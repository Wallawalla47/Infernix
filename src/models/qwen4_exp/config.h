#pragma once

// Qwen4Exp (Qwen3.8-Flash-Next) instance configuration as recorded by the converter
// (tools/convert/qwen4_exp.py). The mathematics is fixed by this implementation; the config only
// supplies dimensions and the n-gram hash and table geometry
// (docs/maintainer/qwen3_8-flash-next-design.md §2, §12).

#include "models/load_options.h"
#include "models/qwen3_5/config.h"
#include "models/qwen4_exp/frontend/ngram_hash.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace infernix::artifact {
struct Directory;
}

namespace infernix::models::qwen4_exp {

enum class MixerKind : std::uint8_t { Attention, Gdn };
enum class GateActivation : std::uint8_t { Sigmoid, Silu };

struct AttentionConfig {
    std::uint32_t heads    = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim = 0;

    [[nodiscard]] std::uint32_t query_width() const noexcept { return heads * head_dim; }
    [[nodiscard]] std::uint32_t key_width() const noexcept { return kv_heads * head_dim; }
};

// Interleaved MRoPE on the first rotary_dim dimensions of a head (half-split rotation).
struct RopeConfig {
    float theta                = 0;
    std::uint32_t rotary_dim   = 0;
    std::array<std::uint32_t, 3> mrope_section{};
    std::vector<std::uint8_t> pair_axes; // [rotary_dim / 2]: 0 = T, 1 = H, 2 = W
    // Text YaRN (the startup option rope_yarn_factor, [1, 4]; 1 = none): like the reference's one
    // shared rotary embedding, it scales every text rotation (attention Q/K, the QSA indexer's
    // queries and pooled keys, the MTP layer), with max_position_embeddings as the original span.
    float yarn_factor = 1.0F;
};

struct GdnConfig {
    std::uint32_t key_heads     = 0;
    std::uint32_t key_head_dim  = 0;
    std::uint32_t value_heads   = 0;
    std::uint32_t value_head_dim = 0;
    std::uint32_t conv_kernel   = 0;
    GateActivation output_gate  = GateActivation::Sigmoid;
    // A GGUF import (tools/convert/qwen4_exp_gguf.py): the output projection reads its value heads
    // in ggml's tiled order (slot g holds head (g % key_heads) * per_key + g / key_heads), and the
    // decay parameter is A = -exp(A_log) itself (gdn/a), not A_log.
    bool output_ggml_tiled = false;
    bool a_direct          = false;

    [[nodiscard]] std::uint32_t key_width() const noexcept { return key_heads * key_head_dim; }
    [[nodiscard]] std::uint32_t value_width() const noexcept { return value_heads * value_head_dim; }
    [[nodiscard]] std::uint32_t conv_channels() const noexcept { return 2 * key_width() + value_width(); }
};

struct MoeConfig {
    std::uint32_t experts             = 0;
    std::uint32_t top_k               = 0;
    std::uint32_t intermediate        = 0;
    std::uint32_t shared_intermediate = 0;
    // The routed experts' arithmetic (parse_config): W4A16 when the banks carry no activation scales
    // (design §16.2.1), else W4A4 (§16.2); banks whose 4-bit activations are optional run W4A16
    // unless LoadOptions::expert_a4_activations asks for W4A4.
    bool a16_experts = false;
    // The text config's expert_a4_activations: optional (NVIDIA FP4-Local, design §6.1.2): the banks
    // carry their calibrated activation scales and permit 4-bit activations, but default to 16-bit.
    bool a4_optional = false;
};

struct HyperConnectionConfig {
    std::uint32_t streams = 0; // hc_count
    std::uint32_t rank    = 0; // hc_lowrank
};

struct QsaConfig {
    std::uint32_t index_heads    = 0;
    std::uint32_t index_head_dim = 0;
    std::uint32_t budget         = 0; // tokens of complete blocks a query selects at most
    std::uint32_t compress_ratio = 0; // tokens pooled into one index block

    [[nodiscard]] std::uint32_t block_budget() const noexcept { return budget / compress_ratio; }
    // Tokens a query attends to at most: the selected blocks plus its incomplete tail block.
    [[nodiscard]] std::uint32_t max_selected() const noexcept {
        return budget + compress_ratio - 1;
    }
};

// The n-gram table volume written beside the artifact (design §12.2): FP8 E4M3FN rows with one
// per-tensor scale, a GGUF's IQ4_NL rows (design §12.5), the unquantized table's BF16 rows (an exact
// conversion bound to a BF16 volume), or NVIDIA FP4-Local's NVFP4 rows (E2M1 codes, then one E4M3FN
// scale per 16 elements, with one FP32 multiplier for the table; design §12.5). Any volume of the
// model's table, in any of the four formats, may replace the artifact's at run time
// (resolve_ngram_volume).
enum class NgramRowFormat : std::uint8_t { Fp8E4m3, Iq4Nl, Bf16, Nvfp4 };

struct NgramTableConfig {
    NgramRowFormat format = NgramRowFormat::Fp8E4m3;
    // An FP8 volume that is not the artifact's own: its per-tensor scale (BF16 bits), from its header
    // or its known volume id. Empty for the artifact's own table (its ngram_scale parameter) and other formats.
    std::optional<std::uint16_t> fp8_scale;
    // An NVFP4 volume that is not the artifact's own: its FP32 multiplier (bits), from its header.
    // Empty for the artifact's own table (its FP32 ngram_scale parameter) and other formats.
    std::optional<std::uint32_t> nvfp4_scale;
    std::uint64_t rows            = 0;
    std::uint32_t row_bytes       = 0;
    std::uint32_t rows_per_block  = 0;
    std::uint32_t block_bytes     = 0;
    std::uint32_t header_bytes    = 0;
    std::uint64_t blocks          = 0;
    std::uint64_t file_bytes      = 0;
    std::array<std::uint8_t, 16> volume_id{};
};

struct PleConfig {
    std::uint32_t layer        = 0; // zero-based decoder layer the injection precedes
    std::uint32_t embed_dim    = 0;
    std::uint32_t conv_kernel  = 0;
    std::uint32_t conv_dilation = 0; // ngram_size
    NgramConfig ngram;
    std::uint32_t split_parts  = 0;
    NgramTableConfig table;

    [[nodiscard]] std::uint32_t heads() const noexcept {
        return (ngram.ngram_size - 1) * ngram.heads_per_ngram;
    }
    [[nodiscard]] std::uint32_t head_width() const noexcept { return embed_dim / heads(); }
    // Past positions the dilated convolution reads: (kernel - 1) * dilation.
    [[nodiscard]] std::uint32_t conv_span() const noexcept {
        return (conv_kernel - 1) * conv_dilation;
    }
};

struct TextConfig {
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    float rms_norm_eps                    = 0;
    std::int32_t eos_token_id             = 0;
    std::vector<MixerKind> layer_types;
    // Index of each layer among the layers of its kind.
    std::vector<std::uint32_t> compact_layer_indices;
    std::uint32_t attention_layers = 0;
    std::uint32_t gdn_layers       = 0;
    AttentionConfig attention;
    RopeConfig rope;
    GdnConfig gdn;
    MoeConfig moe;
    HyperConnectionConfig hc;
    QsaConfig qsa;
    PleConfig ple;
    // Weights imported from a GGUF (tools/convert/qwen4_exp_gguf.py; the converter writes the gdn_a and
    // gdn_output_order keys): ggml block formats, an IQ4_XS embedding and ggml expert banks.
    bool ggml_weights = false;
    // The lossless route (LoadOptions::lossless; docs/maintainer/lossless-route.md §2a): one scored window
    // per call, every operation correctly rounded (Forward's lossless_* steps).
    bool lossless = false;
    // The design reference (LoadOptions::reference, with lossless; lossless-route.md §3b): the lossless window
    // with a W4A4 bank's declared expert activations and the KV format's codec.
    bool design_reference = false;
    // The routed experts run exact A16 arithmetic over their records' own block scales: a GGUF
    // import's ggml banks or Qwen's block-FP8 banks (fp8_expert_b128_v1). Set by the loader.
    bool exact_experts = false;
    // The largest record stride of exact banks (a GGUF import's strides differ by layer: staging,
    // stream-ring and RAM slots take it), set by the loader from the stored banks; 0 means the NVFP4
    // record's. expert_frame_bytes: the mean stride, rounded up (the expert cache's sizing unit: its
    // pools give every layer the same frame count; ExpertResidency::average_frame_bytes).
    std::uint64_t expert_frame_stride = 0;
    std::uint64_t expert_frame_bytes  = 0;

    // The positions a context may use: the native ones times the YaRN factor, at most 1,048,576.
    [[nodiscard]] std::uint32_t context_ceiling() const noexcept {
        return static_cast<std::uint32_t>(
            std::min(double(max_position_embeddings) * double(rope.yarn_factor), 1048576.0));
    }

    [[nodiscard]] std::uint32_t residual_width() const noexcept { return hc.streams * hidden_size; }
};

// A DFlash2 drafter (docs/maintainer/dflash.md) from a companion artifact (--draft-model): its own
// block geometry; the hidden width, vocabulary, token embedding and head are the target's. Every
// layer is a sliding-window layer (the drafter of Qwen3.8-Flash-Next has five).
struct DFlash2Config {
    std::uint32_t layers       = 0;
    AttentionConfig attention;
    std::uint32_t intermediate = 0;
    std::uint32_t window       = 0; // the sliding window and context ring capacity
    float rms_norm_eps         = 0;
    float rope_theta           = 0; // full-head 1-D RoPE at absolute positions
    std::vector<std::uint32_t> target_layer_ids; // the completed target layers whose stream means are the features
    std::uint32_t mask_token_id  = 0;
    std::uint32_t conv_kernel    = 0; // dynamic convolution taps
    std::uint32_t conv_group     = 0; // channels sharing one dynamic coefficient
    std::uint32_t selector_rank  = 0;
    std::uint32_t selector_top_k = 0;
    // Set by the loader from the bound head the drafts read (the proposal head with --lm-head-draft,
    // else the text head): its rows, whether its top-16 runs the fused ops::linear_topk (a Q4
    // proposal head or a Q8 text head), and whether it is a GGUF import's (ggml blocks).
    std::uint32_t head_rows = 0;
    bool fused_head         = false;
    bool ggml_head          = false;

    // The feature width: the target hidden width per tapped layer.
    [[nodiscard]] std::uint32_t feature_width(std::uint32_t hidden) const noexcept {
        return hidden * static_cast<std::uint32_t>(target_layer_ids.size());
    }
};

struct Config {
    TextConfig text;
    bool mtp = false;                // the MTP drafter is bound (--spec mtp)
    std::optional<DFlash2Config> dflash2; // the DFlash2 drafter is bound (--spec dflash2 --draft-model)
    std::uint32_t proposal_rows = 0; // rows of the bound proposal head (--lm-head-draft), else 0
    // The Vision tower (--vision): Qwen3.5's tower except its output width, which the merger's
    // bound fc2 gives (design §19.3.2).
    std::optional<qwen3_5::VisionConfig> vision;
};

[[nodiscard]] Config parse_config(const artifact::Directory& directory, const LoadOptions& options);

// The DFlash2 drafter of a companion artifact (a directory without text whose dflash2 component
// records the target identity it was converted for, tools/convert/qwen4_exp.py), checked against
// the target's text config.
[[nodiscard]] DFlash2Config parse_dflash2_config(const artifact::Directory& companion, const TextConfig& target);

} // namespace infernix::models::qwen4_exp
