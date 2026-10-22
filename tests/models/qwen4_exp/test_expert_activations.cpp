// Qwen3.8-Flash-Next's routed-expert arithmetic as parse_config resolves it (design §6.1.2): W4A4 for
// banks that carry activation scales, W4A16 for banks without; banks whose 4-bit activations are
// optional (NVIDIA FP4-Local's `expert_a4_activations: optional`) run W4A16 unless
// --expert-a4-activations asks for W4A4. Every other artifact refuses the option. Host only, on a
// synthetic directory.

#include "artifact/schema.h"
#include "models/qwen4_exp/config.h"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <optional>
#include <string>

using namespace infernix;
using namespace infernix::models::qwen4_exp;

namespace {

constexpr const char* kScales = "text/layers/0/moe/expert_input_scales";

// A small but complete Qwen4Exp text config: GDN then attention, the PLE before layer 0.
artifact::Json text_config() {
    artifact::Json config = artifact::parse_json(R"({
        "architectures": ["Qwen4ExpForCausalLM"], "model_type": "qwen4_exp_text",
        "hidden_size": 64, "vocab_size": 256, "num_hidden_layers": 2, "max_position_embeddings": 4096,
        "tie_word_embeddings": false, "rms_norm_eps": 1e-6,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 64,
        "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25, "mrope_section": [3, 3, 2]},
        "linear_num_key_heads": 2, "linear_key_head_dim": 16, "linear_num_value_heads": 4,
        "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4, "output_gate_type": "sigmoid",
        "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32, "norm_topk_prob": true, "hc_count": 4, "hc_lowrank": 8,
        "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 32, "indexer_budget": 64,
        "indexer_compress_ratio": 4, "ple_layer_ids": [1], "ple_embed_dim": 64, "ple_conv_kernel_size": 4,
        "ngram_size": 3, "heads_per_ngram": 2, "ngram_vocab_size_base": 1000,
        "make_ngram_vocab_size_divisible_by": 16, "seed": 1234, "split_ngram_parts": 1, "eos_token_id": 0
    })", "test text config");
    const NgramConfig ngram{.vocab_size = 256, .eos_token_id = 0, .ngram_size = 3, .heads_per_ngram = 2,
                            .ngram_vocab_size_base = 1000, .make_ngram_vocab_size_divisible_by = 16, .seed = 1234};
    // FP8 rows: one byte per element of a 16-wide head.
    const std::uint64_t rows = NgramHash(ngram, 0).table_rows(), row_bytes = 16, per_block = 4096 / row_bytes;
    const std::uint64_t blocks = (rows + per_block - 1) / per_block;
    config["ngram_table"] = {{"format", "fp8_e4m3fn"}, {"rows", rows}, {"row_bytes", row_bytes},
                             {"rows_per_block", per_block}, {"block_bytes", 4096}, {"header_bytes", 4096},
                             {"blocks", blocks}, {"file_bytes", 4096 + blocks * 4096},
                             {"volume_id", "000102030405060708090a0b0c0d0e0f"}};
    return config;
}

struct Case {
    const char* name;
    std::optional<std::string> marker; // the text config's expert_a4_activations
    bool scales;                       // the banks store their activation scales
    bool expert_a4;                    // LoadOptions::expert_a4_activations
    int expected;                     // 1: W4A16, 0: W4A4, -1: refused
    const char* refusal = "";          // a fragment of the refusal
};

} // namespace

int main() {
    const Case cases[] = {
        {"W4A4 bank", std::nullopt, true, false, 0},
        {"weight-only bank", std::nullopt, false, false, 1},
        {"W4A4 bank, option", std::nullopt, true, true, -1, "--expert-a4-activations"},
        {"weight-only bank, option", std::nullopt, false, true, -1, "--expert-a4-activations"},
        {"optional bank", "optional", true, false, 1},
        {"optional bank, option", "optional", true, true, 0},
        {"optional bank without scales", "optional", false, false, -1, "activation scales"},
        {"unknown marker", "required", true, false, -1, "must be optional"},
    };
    int failures = 0;
    for (const Case& c : cases) {
        artifact::Directory directory;
        artifact::Json config = text_config();
        if (c.marker) { config["expert_a4_activations"] = *c.marker; }
        directory.components["text"].config = std::move(config);
        if (c.scales) { directory.bindings[kScales] = artifact::Binding{}; }
        models::LoadOptions options;
        options.expert_a4_activations = c.expert_a4;
        try {
            const models::qwen4_exp::Config parsed = parse_config(directory, options);
            const int got       = parsed.text.moe.a16_experts ? 1 : 0;
            if (got != c.expected || parsed.text.moe.a4_optional != (c.marker == "optional")) {
                std::fprintf(stderr, "FAIL %s: resolved %s\n", c.name, got ? "W4A16" : "W4A4");
                ++failures;
            }
        } catch (const std::exception& error) {
            if (c.expected != -1 || std::string(error.what()).find(c.refusal) == std::string::npos) {
                std::fprintf(stderr, "FAIL %s: %s\n", c.name, error.what());
                ++failures;
            }
        }
    }
    std::printf("%s routed-expert activation resolution (%zu cases)\n", failures ? "FAIL" : "PASS",
                sizeof(cases) / sizeof(cases[0]));
    return failures ? 1 : 0;
}
