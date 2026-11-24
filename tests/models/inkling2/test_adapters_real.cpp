// --lora and --ablate on a real Inkling 2 artifact, through the public scoring Engine (a generated adapter and sidecar):
//  1. A zero adapter (B = 0, A nonzero, on every module kind of a prelude, a sliding core and a full core layer) leaves
//     the logits' bits, on the default, reproducible and lossless routes; a nonzero adapter changes them, finite.
//  2. The adapted reproducible route is request-invariant in the chunking (chunks of 16 and of 100 give the same bits).
//  3. A sidecar cutting both lanes of every layer at alpha 1e-30 leaves the logits' bits on the default route; at
//     alpha 1 it changes them, finite.
//  4. prompt_hidden_states: 24 rows of 2304 at the last prompt token; the same rows at any prefill chunk; the logits
//     unchanged by the capture.
//  5. --ablate with --lossless is refused; --lora with --lossless runs.
//  6. Generation under a nonzero adapter on all 24 layers and under a cut: DSpark gives plain decoding's tokens on the
//     default, --q8-weights, --a8-activations, --a16-activations and --reproducible routes; on the last, a reply is the
//     same alone and beside another request.
// Checks 1 and 3 also run on those five routes.
//
//   INFERNIX_INKLING2_ARTIFACT  an artifact converted with the inkling2_mini recipe (skips without it)

#include "artifact/safetensors_writer.h"
#include "infernix/engine.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace sw = infernix::test::safetensors_writer;
using Json   = nlohmann::json;
using Tokens = std::vector<infernix::TokenId>;

constexpr std::uint32_t kContext = 512;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

Tokens prompt() {
    Tokens ids;
    std::uint32_t state = 12345;
    for (int i = 0; i < 300; ++i) {
        state = state * 1664525U + 1013904223U;
        ids.push_back(static_cast<infernix::TokenId>(1000 + (state >> 8) % 150000));
    }
    return ids;
}

infernix::EngineOptions scoring(const std::filesystem::path& artifact) {
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = infernix::EnginePurpose::CausalScoring;
    options.max_context   = kContext;
    options.kv_capacity   = infernix::KvCapacityPolicy::explicit_capacity(kContext);
    options.kv_cache      = infernix::KvCacheStorage::BFloat16;
    options.diagnostic_observer.callback = [](const infernix::Diagnostic& d) {
        if (d.level >= infernix::DiagnosticLevel::Warning) { std::cerr << "[engine] " << d.message << '\n'; }
    };
    return options;
}

const std::vector<std::uint32_t> kPositions = {0, 1, 2, 16, 64, 128, 200, 269};
constexpr std::uint32_t kFirstTarget        = 30; // positions above are relative to this (scored positions)

std::vector<float> logits(const infernix::EngineOptions& options, infernix::ScoreResult* full = nullptr,
                          bool hidden = false) {
    infernix::Engine engine(options);
    infernix::ScoreOptions score;
    score.logits_positions     = kPositions;
    score.prompt_hidden_states = hidden;
    infernix::ScoreResult result = engine.score_tokens(prompt(), kFirstTarget, score);
    for (const float v : result.logits) { require(std::isfinite(v), "a non-finite logit"); }
    std::vector<float> out = result.logits;
    if (full != nullptr) { *full = std::move(result); }
    return out;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

constexpr std::uint32_t kGenerationContext = 2048;

infernix::EngineOptions generating(const std::filesystem::path& artifact, std::uint32_t lanes) {
    infernix::EngineOptions options;
    options.artifact_path                     = artifact;
    options.max_context                       = kGenerationContext;
    options.kv_capacity                       = infernix::KvCapacityPolicy::explicit_capacity(lanes * kGenerationContext);
    options.kv_cache                          = infernix::KvCacheStorage::BFloat16;
    options.max_concurrency                   = lanes;
    options.max_pending_requests              = 8;
    options.context_cache.host_capacity_bytes = std::size_t{1} << 30U;
    options.diagnostic_observer.callback      = [](const infernix::Diagnostic& d) {
        if (d.level >= infernix::DiagnosticLevel::Warning) { std::cerr << "[engine] " << d.message << '\n'; }
    };
    return options;
}

const std::vector<std::string>& reply_texts() {
    static const std::vector<std::string> t = {"Explain how a compiler turns source code into machine instructions.",
                                               "Write a short story about a lighthouse keeper who finds a message in a bottle."};
    return t;
}

// Reply 0 is greedy; reply 1 is sampled with its own seed.
infernix::RequestOptions reply_options(std::size_t i) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = i == 0 ? 96 : 64;
    options.execution.sampling.temperature    = i == 0 ? 0.0F : 1.0F;
    options.execution.sampling.top_k          = i == 0 ? 0 : 20;
    options.execution.sampling.seed           = 1000 + i;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    return options;
}

infernix::PreparedPrompt reply_prompt(infernix::Engine& engine, std::size_t i) {
    infernix::PromptInput input;
    infernix::ChatMessage user;
    user.role = infernix::ChatRole::User;
    user.parts.push_back({.text = reply_texts()[i]});
    input.messages.push_back(std::move(user));
    input.options.reasoning_effort = infernix::ReasoningEffort::Low;
    return engine.prepare(std::move(input));
}

std::vector<Tokens> replies(infernix::Engine& engine) {
    std::vector<Tokens> out;
    for (std::size_t i = 0; i < reply_texts().size(); ++i) {
        out.push_back(engine.generate(reply_prompt(engine, i), reply_options(i)).generated_token_ids);
    }
    return out;
}

void require_replies(const std::string& label, const std::vector<Tokens>& expected, const std::vector<Tokens>& got) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(!expected[i].empty() && expected[i] == got[i], label + ": reply " + std::to_string(i) + " differs");
    }
}

// A PEFT adapter over the modules each kind of layer has: `zero_b` writes B = 0; `every_layer` adapts all 24 layers.
void write_adapter(const std::filesystem::path& dir, bool zero_b, bool every_layer = false) {
    struct Module {
        std::string path;
        std::uint64_t in, out;
    };
    std::vector<Module> modules;
    const auto attention = [&](const std::string& layer, bool full) {
        const std::string a = layer + "self_attn.";
        modules.push_back({a + "q_proj.linear", 2304, 3072});
        modules.push_back({a + "k_proj.linear", 2304, 1024});
        modules.push_back({a + "v_proj.linear", 2304, 1024});
        modules.push_back({a + "relative_position_bias.linear", 2304, 384});
        modules.push_back({a + "sink_proj", 2304, 24});
        modules.push_back({a + "o_proj.gate_proj", 2304, 3072});
        modules.push_back({a + "o_proj.linear", 3072, 2304});
        if (full) {
            modules.push_back({a + "k_proj.sconv.weight_proj", 2304, 32});
            modules.push_back({a + "v_proj.sconv.weight_proj", 2304, 32});
        }
    };
    const auto gates = [&](const std::string& layer) {
        for (const char* kind : {"q", "k", "v", "mlp"}) {
            const std::string gate = layer + kind + "_message_gate.";
            modules.push_back({gate + "down_proj", 4608, 64});
            modules.push_back({gate + "forget_gate_proj", 64, 2304});
            modules.push_back({gate + "message_gate_proj", 64, 2304});
        }
    };
    if (every_layer) {
        for (int layer = 0; layer < 24; ++layer) {
            const std::string name = "model.language_model.layers." + std::to_string(layer) + ".";
            attention(name, layer % 6 == 5); // the full-attention layers: 5, 11, 17, 23
            if (layer >= 6 && layer < 18) { gates(name); }
        }
    } else {
        attention("model.language_model.layers.1.", false); // the prelude
        attention("model.language_model.layers.8.", false); // a sliding core layer
        attention("model.language_model.layers.11.", true); // a full core layer
        gates("model.language_model.layers.8.");
    }
    constexpr std::uint64_t r = 8;
    std::vector<sw::TensorSpec> tensors;
    int seed = 1;
    for (const Module& m : modules) {
        std::vector<float> a = sw::pattern(r * m.in, seed++), b = sw::pattern(m.out * r, seed++);
        for (auto& v : a) { v *= 0.02F; }
        for (auto& v : b) { v = zero_b ? 0.0F : v * 0.02F; }
        tensors.push_back({"base_model.model." + m.path + ".lora_A.weight", "F32", {r, m.in}, std::move(a)});
        tensors.push_back({"base_model.model." + m.path + ".lora_B.weight", "F32", {m.out, r}, std::move(b)});
    }
    std::filesystem::create_directories(dir);
    sw::write(dir / "adapter_model.safetensors", tensors);
    sw::write_json(dir / "adapter_config.json", {{"peft_type", "LORA"}, {"r", r}, {"lora_alpha", 8}, {"bias", "none"}});
}

// A sidecar cutting both lanes of every layer at `alpha`, with one shared unit direction.
std::filesystem::path write_sidecar(const std::filesystem::path& dir, float alpha) {
    std::filesystem::create_directories(dir);
    std::vector<float> d(2304);
    double norm = 0;
    for (std::size_t i = 0; i < d.size(); ++i) {
        d[i] = static_cast<float>((i * 31 + 7) % 101) / 101.0F - 0.5F;
        norm += static_cast<double>(d[i]) * d[i];
    }
    for (auto& v : d) { v = static_cast<float>(v / std::sqrt(norm)); }
    {
        std::ofstream out(dir / "direction.f32", std::ios::binary);
        out.write(reinterpret_cast<const char*>(d.data()), static_cast<std::streamsize>(d.size() * 4));
    }
    Json table = Json::object();
    for (int l = 0; l < 24; ++l) { table[std::to_string(l)] = alpha; }
    sw::write_json(dir / "sidecar.json", {{"format", "ablate-direction-v2"},
                                          {"direction_dtype", "float32"},
                                          {"direction_dim", 2304},
                                          {"direction_file", "direction.f32"},
                                          {"alpha_attn_per_layer", table},
                                          {"alpha_mlp_per_layer", table},
                                          {"generation", {{"source", "test_adapters_real"}}}});
    return dir / "sidecar.json";
}

} // namespace

int main() {
    const char* path = std::getenv("INFERNIX_INKLING2_ARTIFACT");
    if (path == nullptr) {
        std::cout << "SKIP: INFERNIX_INKLING2_ARTIFACT is not set\n";
        return 77;
    }
    try {
        const std::filesystem::path artifact = path;
        sw::TempDir temp("inkling2_adapters");
        const std::filesystem::path zero = temp.path / "zero", adapter = temp.path / "adapter";
        write_adapter(zero, true);
        write_adapter(adapter, false);

        const auto route = [&](const char* name, const std::function<void(infernix::EngineOptions&)>& set) {
            infernix::EngineOptions options = scoring(artifact);
            set(options);
            const std::vector<float> base = logits(options);
            infernix::EngineOptions with  = options;
            with.lora_path                = zero;
            require(same_bits(logits(with), base), std::string(name) + ": a zero adapter moved a logit");
            with.lora_path                 = adapter;
            const std::vector<float> moved = logits(with);
            require(!same_bits(moved, base), std::string(name) + ": the adapter changed nothing");
            std::cout << name << ": --lora ok\n";
        };
        using Set = std::function<void(infernix::EngineOptions&)>;
        const std::vector<std::pair<std::string, Set>> fast_routes = {
            {"default", [](infernix::EngineOptions&) {}},
            {"--q8-weights", [](infernix::EngineOptions& o) { o.q8_weights = true; }},
            {"--a8-activations", [](infernix::EngineOptions& o) { o.a8_activations = true; }},
            {"--a16-activations", [](infernix::EngineOptions& o) { o.expert_a16_activations = true; }},
            {"--reproducible", [](infernix::EngineOptions& o) { o.fast_paths = false; }},
        };
        for (const auto& [name, set] : fast_routes) { route(name.c_str(), set); }
        route("lossless", [](infernix::EngineOptions& o) {
            o.lossless   = true;
            o.fast_paths = false;
        });

        // 2. The adapted reproducible route, chunked two ways.
        {
            infernix::EngineOptions o = scoring(artifact);
            o.fast_paths              = false;
            o.lora_path               = adapter;
            o.prefill_chunk           = 16;
            const std::vector<float> a = logits(o);
            o.prefill_chunk            = 100;
            require(same_bits(logits(o), a), "the adapted reproducible route depends on the prefill chunk");
            std::cout << "reproducible --lora: chunk-invariant\n";
        }

        // 3. --ablate.
        const std::filesystem::path tiny_cut = write_sidecar(temp.path / "tiny", 1e-30F);
        const std::filesystem::path full_cut = write_sidecar(temp.path / "one", 1.0F);
        for (const auto& [name, set] : fast_routes) {
            infernix::EngineOptions o = scoring(artifact);
            set(o);
            const std::vector<float> base = logits(o);
            o.ablate_sidecar_path         = tiny_cut;
            require(same_bits(logits(o), base), name + ": alpha 1e-30 moved a logit");
            o.ablate_sidecar_path = full_cut;
            require(!same_bits(logits(o), base), name + ": the cut changed nothing");
            std::cout << name << ": --ablate ok\n";
        }

        // 4. Hidden states.
        {
            infernix::EngineOptions o = scoring(artifact);
            const std::vector<float> base = logits(o);
            o.prefill_chunk               = 16;
            infernix::ScoreResult a;
            require(same_bits(logits(o, &a, true), base), "capturing the hidden states changed the logits");
            require(a.hidden_state_layers == 24 && a.hidden_states.size() == 24U * 2304U,
                    "24 rows of 2304 hidden values");
            for (const float v : a.hidden_states) { require(std::isfinite(v), "a non-finite hidden value"); }
            o.prefill_chunk = 100;
            infernix::ScoreResult b;
            (void)logits(o, &b, true);
            require(same_bits(a.hidden_states, b.hidden_states), "the hidden states depend on the prefill chunk");
            std::cout << "prompt_hidden_states ok\n";
        }

        // 5. --ablate on the lossless route is refused, by name.
        {
            infernix::EngineOptions o = scoring(artifact);
            o.lossless                = true;
            o.fast_paths              = false;
            o.ablate_sidecar_path     = write_sidecar(temp.path / "refused", 1.0F);
            bool refused              = false;
            try {
                infernix::Engine engine(o);
            } catch (const std::invalid_argument& e) {
                refused = std::string(e.what()).find("--lossless") != std::string::npos;
            }
            require(refused, "--ablate --lossless was not refused by name");
        }

        // 6. Generation with a nonzero adapter on all 24 layers, and with a cut of every lane. DSpark (auto, with the
        // shortlist on a ranked artifact) gives plain decoding's tokens on every route: drafts are not adapted, and
        // verification decides. On the reproducible route a reply is the same alone and beside another request.
        const std::filesystem::path everywhere = temp.path / "everywhere";
        write_adapter(everywhere, false, true);
        const std::vector<std::pair<std::string, Set>> changes = {
            {"--lora", [&](infernix::EngineOptions& o) { o.lora_path = everywhere; }},
            {"--ablate", [&](infernix::EngineOptions& o) { o.ablate_sidecar_path = full_cut; }},
        };
        for (const auto& [change, adapt] : changes) {
            for (const auto& [name, set] : fast_routes) {
                const std::string label = change + ", " + name;
                infernix::EngineOptions plain = generating(artifact, 1);
                set(plain);
                adapt(plain);
                std::vector<Tokens> expected;
                {
                    infernix::Engine engine(plain);
                    expected = replies(engine);
                }
                infernix::EngineOptions draft         = plain;
                draft.speculative.backend             = infernix::SpeculativeBackend::DSpark;
                draft.speculative.draft_tokens_auto   = true;
                draft.speculative.draft_tokens        = 7;
                if (std::getenv("INFERNIX_INKLING2_RANKED") != nullptr) {
                    draft.speculative.proposal_head = infernix::ProposalHead::Optimized;
                }
                {
                    infernix::Engine engine(draft);
                    require_replies(label + ": DSpark against plain", expected, replies(engine));
                }
                if (name == "--reproducible") {
                    infernix::EngineOptions lanes = generating(artifact, 3);
                    set(lanes);
                    adapt(lanes);
                    infernix::Engine engine(lanes);
                    const std::vector<Tokens> alone = replies(engine);
                    auto second = engine.submit(reply_prompt(engine, 1), reply_options(1));
                    auto first  = engine.submit(reply_prompt(engine, 0), reply_options(0));
                    const std::vector<Tokens> together = {first.wait().generated_token_ids, second.wait().generated_token_ids};
                    require_replies(label + ": alone against beside another request", alone, together);
                }
                std::cout << label << ": generation ok\n";
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << "\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
