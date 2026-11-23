// --ablate on the Qwen3.5 family (docs/ablate.md) on the reproducible route (fast paths off, prefix reuse off): one
// artifact a run, named by the first argument (dense: the 27B, INFERNIX_TEST_ARTIFACT; moe: the 35B-A3B,
// INFERNIX_TEST_MOE_ARTIFACT; offloaded: the 122B-A10B, INFERNIX_TEST_OFFLOADED_ARTIFACT), then the drafters it holds
// (mtp, dflash, dflash2). The test writes its own sidecars: a deterministic unit direction and alpha tables.
//
// A cut lane writes the residual with its own kernels, as without a sidecar (the fused projections, the fused NVFP4
// MLP, the sparse MoE, the MoE combines, a LoRA term), then ops::residual_delta_cut removes the direction's component
// of what the write added. A cut too small to move a residual element (alpha 1e-30) therefore keeps the plain bits.
//
// 0. The sidecar's geometry is the model's: a table naming layer kLayers is refused at startup, kLayers - 1 is not.
// 1. A negligible cut on both lanes of every layer gives the plain run's bits (first-token readouts, 32 greedy
//    tokens) on every prompt: every cut path (the copies, the cut kernels, the trimmed last layer, the decode graph)
//    without the cut's arithmetic. An element whose residual is exactly zero may legitimately differ; none is
//    expected.
// 2. Positive controls: a cut of the last layer's attention lane alone (the trimmed call the first token is read
//    from), of its MLP lane alone, of layer 0's Gated DeltaNet lane alone and of its MLP lane alone each move every
//    first-token readout.
// 3. The decode path is cut: from a one-token prompt, a forced 32-token continuation's log-probabilities move under
//    a strong cut (alpha 2 on every lane) on the decode path at least half as much as on the prefill path.
// 4. Exact identities under the strong cut: a repeated request keeps its bits; two lanes give one lane's tokens; each
//    drafter gives the tokens decoding without it gives (verification is cut as decoding is) and accepts drafts.
// 5. CausalScoring with a sidecar: the strong cut moves the scored log-probabilities, and a repeated window keeps its
//    bits.
// 6. With --lora (dense, INFERNIX_TEST_LORA names a PEFT adapter directory; skipped without it): a negligible cut
//    keeps the adapter run's bits (readouts and greedy tokens), so a cut lane keeps the LoRA term.
//
//   INFERNIX_TEST_ARTIFACT=27b.ninfer infernix_qwen3_5_ablate_real_test dense mtp dflash2

#include "models/resume_cold_probe.h"
#include "real_test_artifact.h"
#include "infernix/engine.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct Kind {
    const char* variable;
    std::uint32_t layers;
    std::uint32_t hidden;
};

// The artifacts' configurations (their text component's num_hidden_layers and hidden_size); check 0 pins them.
Kind kind_of(std::string_view name) {
    if (name == "dense") { return {infernix::test::kDenseArtifactVariable, 64, 5120}; }
    if (name == "moe") { return {infernix::test::kMoeArtifactVariable, 40, 2048}; }
    if (name == "offloaded") { return {infernix::test::kOffloadedArtifactVariable, 48, 3072}; }
    throw std::invalid_argument("the first argument must be dense, moe or offloaded");
}

Kind kind{};
constexpr std::uint32_t kMaxContext = 4096; // the 122B's elastic KV backs this much at startup
constexpr std::uint32_t kChunk      = 1024;
constexpr std::uint32_t kDecode     = 32;
constexpr std::uint32_t kReadouts   = infernix::kMaximumReadoutTokens;

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("  %-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) { ++failures; }
}

// A sidecar in its own directory: a deterministic unit direction and alpha(layer, lane), lane 0 attn, 1 mlp, over
// `layers` layers (more than the model's for check 0).
std::filesystem::path write_sidecar(const std::string& name, const std::function<double(std::uint32_t, int)>& alpha,
                                    std::uint32_t layers = 0) {
    if (layers == 0) { layers = kind.layers; }
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / ("infernix_qwen3_5_ablate_real_" + name);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    std::vector<float> d(kind.hidden);
    std::uint32_t state = 0x5EEDU;
    double norm         = 0.0;
    for (float& v : d) {
        state = state * 1664525U + 1013904223U;
        v     = static_cast<float>(static_cast<std::int32_t>(state >> 8) - (1 << 23)) / static_cast<float>(1 << 23);
        norm += static_cast<double>(v) * v;
    }
    for (float& v : d) { v = static_cast<float>(v / std::sqrt(norm)); }
    {
        std::ofstream out(dir / "d.f32.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(d.data()), static_cast<std::streamsize>(d.size() * sizeof(float)));
    }
    std::string tables[2];
    for (int lane = 0; lane < 2; ++lane) {
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            const double a = alpha(layer, lane);
            if (a == 0.0) { continue; }
            char entry[64];
            std::snprintf(entry, sizeof(entry), "%s\"%u\":%.17g", tables[lane].empty() ? "" : ",", layer, a);
            tables[lane] += entry;
        }
    }
    std::ofstream(dir / "sidecar.json")
        << "{\"format\":\"ablate-direction-v2\",\"direction_dtype\":\"float32\",\"direction_dim\":" << kind.hidden
        << ",\"direction_file\":\"d.f32.bin\",\"generation\":{\"source\":\"infernix_qwen3_5_ablate_real_test\"},"
        << "\"alpha_attn_per_layer\":{" << tables[0] << "},\"alpha_mlp_per_layer\":{" << tables[1] << "}}";
    return dir / "sidecar.json";
}

struct Arm {
    std::filesystem::path sidecar;
    std::filesystem::path lora;
    std::string drafter; // empty: none
    std::uint32_t lanes = 1;
    bool scoring        = false;
};

infernix::EngineOptions engine_options(const char* artifact, const Arm& arm) {
    infernix::EngineOptions options;
    options.artifact_path                     = artifact;
    options.context_cache.host_capacity_bytes = infernix::test::kTestHostTierBytes;
    options.max_context                       = kMaxContext;
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(kMaxContext * arm.lanes);
    options.kv_cache             = infernix::KvCacheStorage::Int8Group64;
    options.max_concurrency      = arm.lanes;
    options.max_pending_requests = 2 * arm.lanes + 2;
    options.prefill_chunk        = kChunk;
    options.enable_vision        = false;
    options.fast_paths           = false; // the reproducible route
    options.ablate_sidecar_path  = arm.sidecar;
    options.lora_path            = arm.lora;
    if (arm.scoring) { options.purpose = infernix::EnginePurpose::CausalScoring; }
    if (arm.drafter == "mtp") {
        options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    } else if (arm.drafter == "dflash") {
        options.speculative.backend      = infernix::SpeculativeBackend::DFlash;
        options.speculative.draft_tokens = 7;
    } else if (arm.drafter == "dflash2") {
        options.speculative.backend      = infernix::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = 7;
    } else if (!arm.drafter.empty()) {
        throw std::invalid_argument("the drafters must be mtp, dflash or dflash2");
    }
    return options;
}

struct Readout {
    std::vector<float> logprobs;
    infernix::TokenId top = 0;
};

// The readout tokens of a prompt: its next token, then distinct prompt tokens, then ids counting up.
std::vector<infernix::TokenId> readout_tokens(const std::vector<infernix::TokenId>& prompt, infernix::TokenId next) {
    std::vector<infernix::TokenId> out{next};
    for (std::size_t i = 0; i < prompt.size() && out.size() < kReadouts / 2; ++i) {
        if (std::find(out.begin(), out.end(), prompt[i]) == out.end()) { out.push_back(prompt[i]); }
    }
    for (infernix::TokenId id = 0; out.size() < kReadouts; ++id) {
        if (std::find(out.begin(), out.end(), id) == out.end()) { out.push_back(id); }
    }
    return out;
}

Readout read(infernix::Engine& engine, const std::vector<infernix::TokenId>& prompt, infernix::TokenId next) {
    infernix::RequestOptions options        = infernix::test::resume_probe_request(1, false);
    options.execution.readout_tokens        = readout_tokens(prompt, next);
    const infernix::GenerationResult result = engine.generate(engine.prepare_tokens(prompt), options);
    if (!result.readout.has_value() || result.readout->logprobs.size() != kReadouts) {
        throw std::runtime_error("a readout returned no log-probabilities");
    }
    return {result.readout->logprobs, result.readout->top_token};
}

struct Decoded {
    std::vector<infernix::TokenId> tokens;
    std::uint64_t accepted = 0;
};

Decoded decode(infernix::Engine& engine, const std::vector<infernix::TokenId>& prompt) {
    const infernix::GenerationResult result =
        engine.generate(engine.prepare_tokens(prompt), infernix::test::resume_probe_request(kDecode, false));
    if (result.generated_token_ids.size() != kDecode) { throw std::runtime_error("a decode stopped early"); }
    return {result.generated_token_ids, result.speculative.accepted_tokens};
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) { return false; }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i])) { return false; }
    }
    return true;
}

bool same_bits(const Readout& a, const Readout& b) { return a.top == b.top && same_bits(a.logprobs, b.logprobs); }

// The mean |a - b| over every readout of every prompt.
double distance(const std::vector<Readout>& a, const std::vector<Readout>& b) {
    double sum        = 0.0;
    std::size_t count = 0;
    for (std::size_t p = 0; p < a.size() && p < b.size(); ++p) {
        for (std::size_t i = 0; i < a[p].logprobs.size() && i < b[p].logprobs.size(); ++i) {
            sum += std::fabs(static_cast<double>(a[p].logprobs[i]) - b[p].logprobs[i]);
            ++count;
        }
    }
    return count == 0 ? 0.0 : sum / static_cast<double>(count);
}

// log p(tokens[k] | prompt + tokens[0..k)) on the decode path: the tokens forced step by step and observed.
std::vector<float> forced_logprobs(infernix::Engine& engine, const std::vector<infernix::TokenId>& prompt,
                                   const std::vector<infernix::TokenId>& tokens) {
    infernix::RequestOptions options = infernix::test::resume_probe_request(static_cast<std::uint32_t>(tokens.size()), false);
    for (const infernix::TokenId token : tokens) {
        options.execution.constraint.steps.push_back({token});
        options.execution.constraint.observe.push_back({token});
    }
    const infernix::GenerationResult result = engine.generate(engine.prepare_tokens(prompt), options);
    if (result.generated_token_ids != tokens || result.constrained_draws.size() != tokens.size()) {
        throw std::runtime_error("a forced decode did not produce its tokens");
    }
    std::vector<float> out;
    for (const infernix::ConstrainedDraw& draw : result.constrained_draws) {
        if (draw.observed_logprobs.size() != 1) { throw std::runtime_error("a forced step observed no log-probability"); }
        out.push_back(draw.observed_logprobs.front());
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        if (argc < 2) { throw std::invalid_argument("usage: infernix_qwen3_5_ablate_real_test dense|moe|offloaded [mtp|dflash|dflash2]..."); }
        const std::string_view name = argv[1];
        kind                        = kind_of(name);
        const char* artifact        = infernix::test::real_test_artifact(kind.variable);
        if (artifact == nullptr) { return 77; }
        const std::uint32_t last = kind.layers - 1; // a full-attention layer in every Qwen3.5-family model, layer 0 a GDN one

        const auto tiny     = write_sidecar("tiny", [](std::uint32_t, int) { return 1e-30; });
        const auto strong   = write_sidecar("strong", [](std::uint32_t, int) { return 2.0; });
        const auto one_lane = [](std::uint32_t at, int lane_at) {
            return [at, lane_at](std::uint32_t l, int lane) { return l == at && lane == lane_at ? 1.0 : 1e-30; };
        };

        std::vector<infernix::TokenId> text;
        std::vector<std::vector<infernix::TokenId>> prompts; // one, two and four prefill calls at kChunk
        const auto start = [&](const std::string& label, const Arm& arm) {
            std::printf("engine: %s\n", label.c_str());
            auto engine = std::make_unique<infernix::Engine>(engine_options(artifact, arm));
            if (text.empty()) {
                text = engine->tokenize_text(infernix::test::resume_probe_text("tide pools and their inhabitants", 400));
                for (const std::size_t length : {std::size_t{700}, std::size_t{1500}, std::size_t{3100}}) {
                    if (text.size() <= length) { throw std::runtime_error("the prompt text is too short"); }
                    prompts.emplace_back(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(length));
                }
            }
            return engine;
        };
        std::map<std::string, std::vector<Readout>> reads;
        std::map<std::string, std::vector<Decoded>> decoded;
        // An arm's first-token readouts of every prompt and, when `decodes`, their 32 greedy tokens.
        const auto run = [&](const std::string& label, const Arm& arm, bool decodes) {
            auto engine = start(label, arm);
            for (const auto& prompt : prompts) {
                reads[label].push_back(read(*engine, prompt, text[prompt.size()]));
                if (decodes) { decoded[label].push_back(decode(*engine, prompt)); }
            }
            return engine;
        };
        const auto at = [&](std::size_t p) { return " (prompt of " + std::to_string(prompts[p].size()) + " tokens)"; };

        // 0. The geometry.
        {
            const auto beyond = write_sidecar("beyond", [&](std::uint32_t l, int) { return l == kind.layers ? 1.0 : 0.0; },
                                              kind.layers + 1);
            bool refused = false;
            try {
                (void)infernix::Engine(engine_options(artifact, {.sidecar = beyond}));
            } catch (const std::exception& error) {
                refused = std::string(error.what()).find("key outside 0.." + std::to_string(kind.layers - 1)) !=
                          std::string::npos;
                std::printf("    layer %u refused: %s\n", kind.layers, error.what());
            }
            check(refused, "0. a sidecar naming layer " + std::to_string(kind.layers) + " is refused at startup");
        }
        {
            auto engine = run("plain", {}, true);
            check(engine->load_summary().ablate_sidecar.empty(), "a start without --ablate names no sidecar");
        }
        {
            auto engine = run("tiny", {.sidecar = tiny}, true);
            check(!engine->load_summary().ablate_sidecar.empty(), "a start with --ablate names its sidecar");
        }
        {
            // 4. A repeated request keeps its bits; the strong arm's tokens are the identities' reference. 3. Its decode path
            // against tiny's below.
            auto engine = run("strong", {.sidecar = strong}, true);
            for (std::size_t p = 0; p < prompts.size(); ++p) {
                check(same_bits(read(*engine, prompts[p], text[prompts[p].size()]), reads["strong"][p]),
                      "4. a repeated request keeps its readout bits under the strong cut" + at(p));
            }
        }

        // 1.
        for (std::size_t p = 0; p < prompts.size(); ++p) {
            check(same_bits(reads["plain"][p], reads["tiny"][p]), "1. a negligible cut keeps the readouts' bits" + at(p));
            check(decoded["plain"][p].tokens == decoded["tiny"][p].tokens, "1. a negligible cut keeps the 32 greedy tokens" + at(p));
        }
        std::printf("    mean |strong - plain| of the first-token log-probabilities: %.4g\n", distance(reads["strong"], reads["plain"]));

        // 2.
        for (const auto& [label, alpha] :
             {std::pair{std::string("last_attn"), one_lane(last, 0)}, std::pair{std::string("last_mlp"), one_lane(last, 1)},
              std::pair{std::string("first_gdn"), one_lane(0, 0)}, std::pair{std::string("first_mlp"), one_lane(0, 1)}}) {
            (void)run(label, {.sidecar = write_sidecar(label, alpha)}, false);
            bool moved = true;
            for (std::size_t p = 0; p < prompts.size(); ++p) { moved = moved && !same_bits(reads["plain"][p], reads[label][p]); }
            check(moved, "2. a cut of " + label + " alone moves every first-token readout");
        }

        // 3. From a one-token prompt every later column runs the decode path. Each comparison stays on one path, so the
        // two routes' own difference cancels; a decode left uncut would move only through the prompt token's state.
        {
            const std::vector<infernix::TokenId> first(text.begin(), text.begin() + 1);
            std::vector<infernix::TokenId> forced;
            std::vector<float> decode_lp[2], prefill_lp[2]; // [tiny, strong]
            for (int arm = 0; arm < 2; ++arm) {
                auto engine = start(std::string(arm == 0 ? "tiny" : "strong") + " (a forced decode against prefill readouts)",
                                    {.sidecar = arm == 0 ? tiny : strong});
                if (forced.empty()) { forced = decode(*engine, first).tokens; }
                decode_lp[arm] = forced_logprobs(*engine, first, forced);
                std::vector<infernix::TokenId> prefix = first;
                for (const infernix::TokenId token : forced) {
                    prefill_lp[arm].push_back(read(*engine, prefix, token).logprobs.front());
                    prefix.push_back(token);
                }
            }
            double on_decode = 0.0, on_prefill = 0.0;
            for (std::size_t k = 0; k < forced.size(); ++k) {
                on_decode += std::fabs(static_cast<double>(decode_lp[1][k]) - decode_lp[0][k]);
                on_prefill += std::fabs(static_cast<double>(prefill_lp[1][k]) - prefill_lp[0][k]);
            }
            on_decode /= static_cast<double>(forced.size());
            on_prefill /= static_cast<double>(forced.size());
            std::printf("    mean |strong - tiny| log-probability of %zu forced tokens: decode %.4g, prefill %.4g\n",
                        forced.size(), on_decode, on_prefill);
            check(on_prefill > 0.0, "3. the strong cut moves the prefill readouts");
            check(on_decode >= 0.5 * on_prefill, "3. the cut moves the decode path as it moves prefill (>= half)");
        }

        // 4. Two lanes, every prompt submitted at once: each request's tokens are its one-lane tokens.
        {
            auto engine = start("strong on two lanes", {.sidecar = strong, .lanes = 2});
            std::vector<infernix::GenerationHandle> handles;
            for (const auto& prompt : prompts) {
                handles.push_back(engine->submit(engine->prepare_tokens(prompt), infernix::test::resume_probe_request(kDecode, false)));
            }
            for (std::size_t p = 0; p < prompts.size(); ++p) {
                check(handles[p].wait().generated_token_ids == decoded["strong"][p].tokens,
                      "4. two lanes give one lane's tokens under the strong cut" + at(p));
            }
        }
        for (int a = 2; a < argc; ++a) {
            const std::string drafter = argv[a];
            const std::string label   = "strong with " + drafter;
            (void)run(label, {.sidecar = strong, .drafter = drafter}, true);
            std::uint64_t accepted = 0;
            for (std::size_t p = 0; p < prompts.size(); ++p) {
                accepted += decoded[label][p].accepted;
                check(decoded[label][p].tokens == decoded["strong"][p].tokens,
                      "4. " + drafter + " gives the tokens decoding without it gives under the strong cut" + at(p));
            }
            std::printf("    %s accepted %llu drafts\n", drafter.c_str(), static_cast<unsigned long long>(accepted));
            check(accepted > 0, "4. " + drafter + " accepts drafts under the strong cut");
        }

        // 5. CausalScoring: one window of 1537 tokens from position 513.
        {
            std::vector<float> scored[2]; // [tiny, strong]
            for (int arm = 0; arm < 2; ++arm) {
                auto engine = start(std::string("CausalScoring ") + (arm == 0 ? "tiny" : "strong"),
                                    {.sidecar = arm == 0 ? tiny : strong, .scoring = true});
                const std::vector<infernix::TokenId> window(text.begin(), text.begin() + 1537);
                scored[arm] = engine->score_tokens(window, 513).logprobs;
                if (arm == 1) {
                    check(same_bits(engine->score_tokens(window, 513).logprobs, scored[arm]),
                          "5. a repeated scoring window keeps its bits under the strong cut");
                }
            }
            bool finite = scored[1].size() == 1024;
            for (const float v : scored[1]) { finite = finite && std::isfinite(v) && v <= 0.0F; }
            check(finite, "5. CausalScoring under the strong cut returns 1024 finite log-probabilities");
            check(!same_bits(scored[0], scored[1]), "5. the strong cut moves the scored log-probabilities");
        }

        // 6.
        if (const char* lora = std::getenv("INFERNIX_TEST_LORA"); name == "dense" && lora != nullptr && *lora != '\0') {
            (void)run("lora", {.lora = lora}, true);
            (void)run("lora + tiny", {.sidecar = tiny, .lora = lora}, true);
            std::printf("    mean |lora - plain| of the first-token log-probabilities: %.4g\n", distance(reads["lora"], reads["plain"]));
            for (std::size_t p = 0; p < prompts.size(); ++p) {
                check(!same_bits(reads["lora"][p], reads["plain"][p]), "6. the adapter moves the readouts" + at(p));
                check(same_bits(reads["lora + tiny"][p], reads["lora"][p]) &&
                          decoded["lora + tiny"][p].tokens == decoded["lora"][p].tokens,
                      "6. a negligible cut keeps the adapter run's readouts and greedy tokens" + at(p));
            }
        } else if (name == "dense") {
            std::printf("  skip 6: INFERNIX_TEST_LORA is not set\n");
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::printf("FAIL: %d check(s)\n", failures);
        return 1;
    }
    std::printf("--ablate: all checks passed\n");
    return 0;
}
