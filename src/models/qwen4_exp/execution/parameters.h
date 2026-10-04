#pragma once

// Native execution parameters of a loaded Qwen4Exp model: one Weight per fused projection (the
// converter packs every projection group that shares an input into one parent), BF16 vectors for
// norms and convolutions, and each MoE layer's expert source.

#include "core/arena.h"
#include "core/tensor.h"
#include "models/qwen4_exp/model.h"
#include "ninfer/ops/offloaded_sparse_moe.h"
#include "ninfer/ops/weight_input.h"

#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace ninfer::models::qwen4_exp::execution {

using LinearParameters = ops::SingleProjectionWeight;

struct HyperConnectionParameters {
    Tensor norm;           // BF16 [S*H]
    LinearParameters down; // [rank (+ S), S*H]: down rows, then the injection rows when combining
    LinearParameters up;   // [S*H, rank]
    bool combine = false;
};

struct AttentionParameters {
    LinearParameters projection; // [q | gate | k | v | index q | index k, H]
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
    Tensor ngram_scale;                     // BF16 [1]
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
    Weight token_embedding; // BF16 rows in pinned host memory, read zero-copy
    Tensor output_head; // BF16 [H, V]: one vocabulary row per column (FP32 logits)
    HyperConnectionParameters final_mixer;
    std::vector<BlockParameters> layers;

private:
    // Device copy of every layer's ExpertScales, [layers][E].
    DeviceBuffer scales_;
};

} // namespace ninfer::models::qwen4_exp::execution
