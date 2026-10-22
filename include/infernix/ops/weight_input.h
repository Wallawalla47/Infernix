#pragma once

#include "core/weight_view.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/sparse_moe.h"

#include <optional>
#include <span>
#include <variant>

namespace infernix::ops {

// One mathematical use of a logical matrix. The model owns the view and its backing.
struct WeightInput {
    const WeightView& weight;
    LinearPolicy policy = LinearPolicy::A16Only;
    std::optional<float> activation_input_divisor;
    // A static FP8 A8 input's FP32 activation scale (fp8_e4m3fn_row_f32 parents): the checkpoint's
    // calibrated input_scale; fused inputs share one.
    std::optional<float> activation_input_scale;
};

struct SingleProjectionWeight {
    Weight weight;
    LinearPolicy policy = LinearPolicy::A16Only;
};

struct PairedProjectionWeights {
    Weight first, second;
    // A16 at every policy; A16ColumnInvariant selects the column-invariant routes.
    LinearPolicy policy = LinearPolicy::A16Only;
};

using ProjectionWeights = std::variant<SingleProjectionWeight, PairedProjectionWeights>;

// Prepare the existing native forms; no device allocation, upload, execution or graph rewrite.
// Runtime shape/phase choices and scratch remain with the actual calling Op.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(const WeightInput& input);
// Row order is supplied by the calling implementation, for example a Vision Q/K/V bank.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(std::span<const WeightInput> rows);
[[nodiscard]] SingleProjectionWeight prepare_attn_input_proj_weights(const WeightInput& query,
                                                                     const WeightInput& key,
                                                                     const WeightInput& value);
[[nodiscard]] ProjectionWeights prepare_attn_input_proj_weights(const WeightInput& query,
                                                                const WeightInput& key,
                                                                const WeightInput& gate,
                                                                const WeightInput& value);
[[nodiscard]] ProjectionWeights prepare_gdn_input_proj_weights(const WeightInput& query,
                                                               const WeightInput& key,
                                                               const WeightInput& value,
                                                               const WeightInput& z);
[[nodiscard]] ProjectionWeights prepare_gdn_gating_proj_weights(const WeightInput& a,
                                                                const WeightInput& b);
[[nodiscard]] SingleProjectionWeight prepare_linear_swiglu_weight(const WeightInput& gate,
                                                                  const WeightInput& up);
[[nodiscard]] SparseMoeWeights
prepare_sparse_moe_weights(const WeightInput& router, const WeightInput& shared_score,
                           std::span<const WeightInput> expert_gate_up,
                           std::span<const WeightInput> expert_down, const WeightInput& shared_gate,
                           const WeightInput& shared_up, const WeightInput& shared_down);
// A sparse MoE of any geometry whose routed banks run resident_moe_experts (the Qwen3.5-122B-A10B
// drafter's): the router [E, H] and shared score [1, H] BF16 rows in one parent, every expert's
// gate, up and down in one row-split bank of one Q2_G64_FP16, Q4_G64_FP16 or Q8_G32_FP16 codec, the
// shared expert's gate and up a linear parent (gate rows, then up rows) and its down a linear.
[[nodiscard]] SparseMoeWeights
prepare_resident_moe_weights(const WeightInput& router, const WeightInput& shared_score,
                             std::span<const WeightInput> expert_gate_up,
                             std::span<const WeightInput> expert_down, const WeightInput& shared_gate,
                             const WeightInput& shared_up, const WeightInput& shared_down);

} // namespace infernix::ops
