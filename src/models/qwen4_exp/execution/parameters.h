#pragma once

// Native execution parameters of a loaded Qwen4Exp model: one Weight per fused projection (the
// converter packs every projection group that shares an input into one parent), BF16 vectors for
// norms and convolutions, and each MoE layer's expert source.

#include "core/arena.h"
#include "core/tensor.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen4_exp/model.h"
#include "infernix/ops/ggml_linear.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "infernix/ops/rope.h"
#include "infernix/ops/weight_input.h"

#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace infernix::models::qwen4_exp::execution {

// One projection: a native Weight, or (a GGUF import) the GGUF block segments of its rows in
// order, run by ops::ggml_linear (then `weight` names the first segment's format and holds no data).
struct LinearParameters {
    Weight weight;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    std::vector<ops::GgmlSegment> ggml;
    const std::int32_t* input_perm = nullptr; // device: the input's block permutation (GDN output)
    std::int32_t perm_block        = 0;
    [[nodiscard]] bool is_ggml() const noexcept { return !ggml.empty(); }
};

struct HyperConnectionParameters {
    Tensor norm;           // BF16 [S*H]
    LinearParameters down; // [rank (+ S), S*H]: down rows, then the injection rows when combining
    LinearParameters up;   // [S*H, rank]
    bool combine = false;
};

struct AttentionParameters {
    // [q | gate | k | v | index q | index k, H] in one parent; or, when the converter stores the index
    // projections in another format (the FP8 dense projections of recipe C beside its BF16 indexer),
    // [q | gate | k | v, H] here and [index q | index k, H] in index_projection.
    LinearParameters projection;
    std::optional<LinearParameters> index_projection;
    Tensor query_norm, key_norm, index_query_norm, index_key_norm;
    LinearParameters output;
};

struct GdnParameters {
    LinearParameters projection; // [q | k | v | z, H]
    LinearParameters control;    // [a | b, H]
    Tensor a_log, dt_bias;       // FP32 [value heads]
    Tensor convolution;          // BF16 [C, K]
    Tensor norm;                 // BF16 [value head dim]
    LinearParameters output;
};

struct MoeParameters {
    Tensor router;                   // BF16 [H, E]: one expert row per column (FP32 logits)
    Tensor shared_score;             // BF16 [H, 1]: the shared-expert gate row
    LinearParameters shared_gate_up; // [2 * I_shared, H]: gate rows, then up rows
    LinearParameters shared_down;    // [H, I_shared]
    const ExpertBank* bank = nullptr;
    const ops::offloaded_moe::ExpertScales* device_scales = nullptr; // [E]
};

struct PleParameters {
    LinearParameters projection; // [key (S*H) | value (H), embed]
    Tensor key_norm, query_norm, conv_norm; // BF16 [S*H]
    Tensor convolution;                     // BF16 [S*H, K]
    Tensor ngram_scale;                     // BF16 [1] (FP8 tables; empty for an IQ4_NL table)
};

// The MTP drafter (design §11.2).
struct MtpParameters {
    Tensor embedding_norm, hidden_norm; // BF16 [H], [S*H]
    LinearParameters embedding_projection, hidden_projection;
    HyperConnectionParameters attn_hc, mlp_hc, final_mixer;
    AttentionParameters attention;
    Tensor router, shared_score; // BF16, as MoeParameters
    LinearParameters shared_gate_up, shared_down;
    Weight experts_gate_up; // [E * 2I, H]: expert e's gate rows, then its up rows
    Weight experts_down;    // [E * H, I]
};

// One DFlash2 layer (docs/maintainer/dflash.md "DFlash2 dynamic convolution").
struct DFlash2LayerParameters {
    Tensor input_norm, post_attention_norm; // BF16 [H]
    Weight query_key_value;                 // Q8 [q | k | v, H] for ops::attn_input_proj
    LinearParameters context_key, context_value; // [kv, H] (the K/V words, shared)
    Tensor query_norm, key_norm;                 // BF16 [head_dim]
    LinearParameters output;                     // [H, q]
    LinearParameters gate_up, down;              // [2I, H], [H, I]
    Tensor attention_base, mlp_base;             // BF16 [H, taps, sides]
    LinearParameters attention_projection, mlp_projection; // BF16 [2 * taps * H / group, H]
};

// The DFlash2 drafter of a companion artifact (--draft-model).
struct DFlash2Parameters {
    ops::PreparedRope rope;              // full-head 1-D, the drafter's theta
    LinearParameters feature_projection; // [H, taps * H]
    Tensor context_norm, final_norm;     // BF16 [H]
    std::vector<DFlash2LayerParameters> layers;
    LinearParameters selector_hidden;                 // [rank, H]
    Tensor predecessor_codebook, successor_codebook;  // BF16 [rank, vocab]
};

// The draft head: the text head, or the proposal head's rows and their token ids.
struct DraftHeadParameters {
    std::optional<LinearParameters> rows; // the proposal head (q4/q8 [rows, H]); empty = text head
    Tensor token_ids;                     // I32 [rows]
};

struct BlockParameters {
    HyperConnectionParameters attn_hc, mlp_hc;
    std::variant<AttentionParameters, GdnParameters> mixer;
    MoeParameters moe;
    std::optional<PleParameters> ple;
};

class Parameters {
public:
    explicit Parameters(const Model& model);
    ~Parameters();
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;

    const Model& model;
    Weight token_embedding; // BF16 (or a GGUF import's IQ4_XS) rows in pinned host memory, read zero-copy
    Tensor output_head; // BF16 [H, V]: one vocabulary row per column (FP32 logits)
    // Recipe B's 8-bit head (q8_g32_fp16 [V, H], FP32 logits), or a GGUF import's head (ggml blocks,
    // FP32 logits from ops::ggml_linear); output_head is then empty.
    std::optional<LinearParameters> output_head_q8;
    HyperConnectionParameters final_mixer;
    std::vector<BlockParameters> layers;
    std::optional<MtpParameters> mtp;
    std::optional<DFlash2Parameters> dflash2;
    DraftHeadParameters draft_head;
    std::optional<qwen3_5::execution::VisionParameters> vision; // --vision (shared tower types)

private:
    // Device copy of every layer's ExpertScales, [layers][E].
    DeviceBuffer scales_;
    // A GGUF import's GDN output permutation: input block j of the output projection is value head
    // gdn_value_perm_[j] (ggml's tiled order).
    DeviceBuffer gdn_value_perm_;
};

} // namespace infernix::models::qwen4_exp::execution
