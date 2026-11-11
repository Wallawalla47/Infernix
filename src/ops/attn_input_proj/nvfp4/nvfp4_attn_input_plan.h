#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

[[nodiscard]] std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

// A16ColumnInvariant: one sliced-K schedule family (8 K-warps) for every T, so a column's
// arithmetic does not depend on the call's other columns.
void nvfp4_attn_input_invariant_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void nvfp4_attn_input_a16_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                 Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                Tensor& k, Tensor& v, Nvfp4A4Workspace workspace,
                                cudaStream_t stream);

// Shared by the ordinary A4 launcher and the fused RMSNorm entry: the TMA cutoff and the tiled
// activation-scale plane that route reads. The quantizer that feeds the TMA route must write the
// same layout, so both entries derive it here.
//
// From 129 columns (512 until 2026-11-11), each cut and tile forced in turn (RTX 5090, cold weights): one
// partly filled 256-token tile takes 40-46 us at 129-256 columns against 50-64 for the MMA tiles; at
// 257-512 the 128-token tile takes 55-84 us against 66-97 (the 256-token tile 72-89: its second tile is
// mostly empty); from 513 the 256-token tile takes 90-102 us at 513-768 against 101-125 for the 128-token
// one, and 155 against 181 at 1,023. Same bits (one K order per output).
[[nodiscard]] inline constexpr bool nvfp4_attn_input_tma_route(std::int32_t tokens) {
    return tokens >= 129;
}

[[nodiscard]] inline constexpr Nvfp4ScaleLayout nvfp4_attn_input_scale_layout(std::int32_t tokens) {
    return !nvfp4_attn_input_tma_route(tokens) ? Nvfp4ScaleLayout::RowMajor
           : tokens > 256 && tokens <= 512     ? Nvfp4ScaleLayout::Tiled128
                                               : Nvfp4ScaleLayout::Tiled256;
}

void launch_nvfp4_attn_input_fused_rmsnorm_quantize(const Tensor& residual,
                                                    const Tensor& norm_weight, float eps,
                                                    float input_scale_divisor,
                                                    Nvfp4A4Workspace workspace,
                                                    Nvfp4ScaleLayout layout, cudaStream_t stream);

void nvfp4_attn_input_fused_rmsnorm_launch(const Tensor& residual, const Tensor& norm_weight,
                                           float eps, const Weight& weight, Tensor& q, Tensor& gate,
                                           Tensor& k, Tensor& v, WorkspaceArena& workspace,
                                           cudaStream_t stream);

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream);

} // namespace infernix::ops::detail
