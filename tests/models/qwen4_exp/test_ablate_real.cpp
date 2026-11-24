// --ablate on Qwen3.8-Flash-Next (docs/ablate.md), against the same artifact without it, on the reproducible route
// (fast paths off, one lane, prefix reuse off). The test writes its own sidecars: a deterministic unit direction and an
// alpha table.
//
// 1. A negligible cut (alpha 1e-30 on both lanes of every layer) takes every cut path (the cut injects, the fused MoE
//    combine, the trimmed last layer, the decode graph's layer walks) and must give the plain run's bits: the same 32
//    greedy tokens and the same first-token readouts on prompts of several prefill calls. An element whose lane output
//    and residual are both exactly zero may legitimately differ; none is expected here.
// 2. Positive controls: a cut of the last layer's MLP lane alone (the trimmed call the first token is read from), of its
//    attention lane alone, and of layer 0's MLP lane alone each move the first-token readouts.
// 3. The decode path is cut too: from a one-token prompt, a forced 32-token continuation's log-probabilities move
//    under a strong cut (alpha 2 on both lanes of every layer) on the decode path (forced decode, observed) at least
//    half as much as on the prefill path (readouts at the same prefixes). Each comparison stays on one path, so the
//    two routes' own difference cancels; a decode left uncut would move only through the prompt token's state.
// 4. The cut keeps the route's prefill invariance: wherever the plain route reads prompts of m * G + r tokens with the
//    same bits at prefill chunks 1024 and G = 3072 (the same call grid, test_prefill_width_real's condition and chunks),
//    the cut does too. The plain run is the control, and must keep its bits at some length or the check says nothing
//    (at chunks 512 and 1536 it keeps them at none).
//
// Skips (77) without INFERNIX_QWEN4_ARTIFACT; INFERNIX_QWEN4_NGRAM names the n-gram volume, INFERNIX_QWEN4_KV the KV
// format (default int8).
//
//   INFERNIX_QWEN4_ARTIFACT=out.infernix [INFERNIX_QWEN4_NGRAM=out.ngram] infernix_qwen4_exp_ablate_real_test

#include "kv_cache_storage.h"
#include "models/resume_cold_probe.h"
#include "engine_host.h"
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
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

constexpr std::uint32_t kLayers     = 48;   // Qwen3.8-Flash-Next
constexpr std::uint32_t kHidden     = 2560;
constexpr std::uint32_t kMaxContext = 8192;
constexpr std::uint32_t kNarrow     = 1024; // test_prefill_width_real's chunks, where the route keeps its bits
constexpr std::uint32_t kGrid       = 3072;
constexpr std::uint32_t kDecode     = 32;
constexpr std::uint32_t kReadouts   = infernix::kMaximumReadoutTokens;

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("  %-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) { ++failures; }
}

// The sidecar directories this run wrote, removed when it ends (none stays in the temp directory).
struct SidecarDirs {
    std::vector<std::filesystem::path> dirs;
    ~SidecarDirs() {
        for (const std::filesystem::path& dir : dirs) {
            std::error_code ignored;
            std::filesystem::remove_all(dir, ignored);
        }
    }
} sidecar_dirs;

// A sidecar in its own directory: a deterministic unit direction and alpha(layer, lane), lane 0 attn, 1 mlp.
std::filesystem::path write_sidecar(const std::string& name, const std::function<double(std::uint32_t, int)>& alpha) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / ("infernix_ablate_real_" + name);
    std::filesystem::remove_all(dir);
    sidecar_dirs.dirs.push_back(dir);
    std::filesystem::create_directories(dir);
    std::vector<float> d(kHidden);
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
        for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
            const double a = alpha(layer, lane);
            if (a == 0.0) { continue; }
            char entry[64];
            std::snprintf(entry, sizeof(entry), "%s\"%u\":%.17g", tables[lane].empty() ? "" : ",", layer, a);
            tables[lane] += entry;
        }
    }
    std::ofstream(dir / "sidecar.json")
        << "{\"format\":\"ablate-direction-v2\",\"direction_dtype\":\"float32\",\"direction_dim\":" << kHidden
        << ",\"direction_file\":\"d.f32.bin\",\"generation\":{\"source\":\"infernix_qwen4_exp_ablate_real_test\"},"
        << "\"alpha_attn_per_layer\":{" << tables[0] << "},\"alpha_mlp_per_layer\":{" << tables[1] << "}}";
    return dir / "sidecar.json";
}

infernix::EngineOptions engine_options(const char* artifact, std::uint32_t chunk, const std::filesystem::path& sidecar) {
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    if (const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM"); ngram != nullptr && *ngram != '\0') {
        options.ngram_volume_path = ngram;
    }
    options.max_context          = kMaxContext;
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(kMaxContext);
    const char* kv               = std::getenv("INFERNIX_QWEN4_KV");
    options.kv_cache             = kv != nullptr ? infernix::test::parse_kv_cache_storage(kv) : infernix::KvCacheStorage::Int8Group64;
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    options.prefill_chunk        = chunk;
    options.fast_paths           = false; // the reproducible route
    options.ablate_sidecar_path  = sidecar;
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

std::vector<infernix::TokenId> decode(infernix::Engine& engine, const std::vector<infernix::TokenId>& prompt) {
    const infernix::GenerationResult result =
        engine.generate(engine.prepare_tokens(prompt), infernix::test::resume_probe_request(kDecode, false));
    if (result.generated_token_ids.size() != kDecode) { throw std::runtime_error("a decode stopped early"); }
    return result.generated_token_ids;
}

bool same_bits(const Readout& a, const Readout& b) {
    if (a.top != b.top || a.logprobs.size() != b.logprobs.size()) { return false; }
    for (std::size_t i = 0; i < a.logprobs.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(a.logprobs[i]) != std::bit_cast<std::uint32_t>(b.logprobs[i])) { return false; }
    }
    return true;
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

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT\n");
        return 77;
    }
    try {
        const auto tiny     = write_sidecar("tiny", [](std::uint32_t, int) { return 1e-30; });
        const auto last_mlp = write_sidecar("last_mlp", [](std::uint32_t l, int lane) { return l + 1 == kLayers && lane == 1 ? 1.0 : 0.0; });
        const auto last_att = write_sidecar("last_attn", [](std::uint32_t l, int lane) { return l + 1 == kLayers && lane == 0 ? 1.0 : 0.0; });
        const auto first    = write_sidecar("first_mlp", [](std::uint32_t l, int lane) { return l == 0 && lane == 1 ? 1.0 : 0.0; });
        const auto strong   = write_sidecar("strong", [](std::uint32_t, int) { return 2.0; });

        std::vector<infernix::TokenId> text;
        std::vector<std::vector<infernix::TokenId>> prompts; // several prefill calls each at chunk 512
        std::map<std::string, std::vector<Readout>> first_reads;
        std::map<std::string, std::vector<std::vector<infernix::TokenId>>> decoded;
        const auto run = [&](const std::string& name, const std::filesystem::path& sidecar, bool decodes) {
            std::printf("engine: %s\n", name.c_str());
            infernix::Engine engine(engine_options(artifact, 512, sidecar), infernix::test::engine_host());
            if (text.empty()) {
                text = engine.tokenize_text(infernix::test::resume_probe_text("tide pools and their inhabitants", 400));
                for (const std::size_t length : {std::size_t{700}, std::size_t{1500}, std::size_t{3100}}) {
                    if (text.size() <= length) { throw std::runtime_error("the prompt text is too short"); }
                    prompts.emplace_back(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(length));
                }
            }
            for (const auto& prompt : prompts) {
                first_reads[name].push_back(read(engine, prompt, text[prompt.size()]));
                if (decodes) { decoded[name].push_back(decode(engine, prompt)); }
            }
            return engine.load_summary().ablate_sidecar;
        };
        check(run("plain", {}, true).empty(), "a start without --ablate names no sidecar");
        check(!run("tiny", tiny, true).empty(), "a start with --ablate names its sidecar");
        for (std::size_t p = 0; p < prompts.size(); ++p) {
            const std::string at = " (prompt of " + std::to_string(prompts[p].size()) + " tokens)";
            check(same_bits(first_reads["plain"][p], first_reads["tiny"][p]), "1. a negligible cut keeps the readouts' bits" + at);
            check(decoded["plain"][p] == decoded["tiny"][p], "1. a negligible cut keeps the 32 greedy tokens" + at);
        }
        for (const auto& [name, sidecar] : {std::pair{std::string("last_mlp"), last_mlp}, std::pair{std::string("last_attn"), last_att},
                                            std::pair{std::string("first_mlp"), first}}) {
            (void)run(name, sidecar, false);
            bool moved = true;
            for (std::size_t p = 0; p < prompts.size(); ++p) { moved = moved && !same_bits(first_reads["plain"][p], first_reads[name][p]); }
            check(moved, "2. a cut of " + name + " alone moves every first-token readout");
        }
        {
            // From a one-token prompt every later token's column runs the decode path. The cut moves the decode
            // path's log-probabilities of a forced continuation (cut decode against plain decode) as much as it moves
            // the prefill path's (cut prefill readouts against plain ones): each comparison stays on one path, so the
            // routes' own difference cancels. A decode left uncut would see only the prompt token's cut state.
            const std::vector<infernix::TokenId> start(text.begin(), text.begin() + 1);
            std::vector<infernix::TokenId> forced;
            std::vector<float> decode_lp[2], prefill_lp[2]; // [plain, cut]
            for (int arm = 0; arm < 2; ++arm) {
                std::printf("engine: %s (a forced decode against prefill readouts)\n", arm == 0 ? "plain" : "strong");
                infernix::Engine engine(engine_options(artifact, 512, arm == 0 ? std::filesystem::path() : strong),
                                        infernix::test::engine_host());
                if (forced.empty()) { forced = decode(engine, start); }
                decode_lp[arm]  = forced_logprobs(engine, start, forced);
                std::vector<infernix::TokenId> prefix = start;
                for (const infernix::TokenId token : forced) {
                    prefill_lp[arm].push_back(read(engine, prefix, token).logprobs.front());
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
            std::printf("    mean |cut - plain| log-probability of %zu forced tokens: decode %.4g, prefill %.4g\n",
                        forced.size(), on_decode, on_prefill);
            check(on_prefill > 0.0, "3. the strong cut moves the prefill readouts");
            check(on_decode >= 0.5 * on_prefill, "3. the cut moves the decode path as it moves prefill (>= half)");
        }
        {
            // Without the cut first (the control): the cut must keep every length's invariance the plain route has.
            std::vector<std::size_t> lengths;
            for (const std::uint32_t m : {1U, 2U}) {
                for (const std::uint32_t r : {0U, 1U, 300U}) {
                    if (m * kGrid + r + 1 < text.size() && m * kGrid + r + 1 < kMaxContext) { lengths.push_back(m * kGrid + r); }
                }
            }
            std::map<std::string, std::vector<Readout>> at; // "<arm> <chunk>" -> [length]
            for (const bool cut : {false, true}) {
                for (const std::uint32_t chunk : {kNarrow, kGrid}) {
                    const std::string key = std::string(cut ? "strong " : "plain ") + std::to_string(chunk);
                    std::printf("engine: %s at prefill chunk %u\n", cut ? "strong" : "plain", chunk);
                    infernix::Engine engine(engine_options(artifact, chunk, cut ? strong : std::filesystem::path()),
                                            infernix::test::engine_host());
                    for (const std::size_t length : lengths) {
                        const std::vector<infernix::TokenId> prompt(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(length));
                        at[key].push_back(read(engine, prompt, text[length]));
                    }
                }
            }
            const std::string narrow = std::to_string(kNarrow), grid = std::to_string(kGrid);
            std::size_t invariant = 0;
            for (std::size_t i = 0; i < lengths.size(); ++i) {
                const bool plain_same = same_bits(at["plain " + narrow][i], at["plain " + grid][i]);
                const bool cut_same   = same_bits(at["strong " + narrow][i], at["strong " + grid][i]);
                invariant += plain_same ? 1U : 0U;
                std::printf("    %zu tokens at chunks %u and %u: plain %s, cut %s\n", lengths[i], kNarrow, kGrid,
                            plain_same ? "bit-identical" : "differs", cut_same ? "bit-identical" : "differs");
                check(cut_same || !plain_same,
                      "4. the cut keeps the plain route's bits across prefill chunks (" + std::to_string(lengths[i]) + " tokens)");
            }
            check(invariant > 0, "4. the control is not vacuous: the plain route keeps its bits at some length");
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
