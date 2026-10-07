#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "infernix/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

/**
 * Hyper-connection residual streams (docs/maintainer/qwen3_8-flash-next-design.md §2, §8.3).
 *
 * A residual R holds S streams of H features per column, stream-major: R[s*H + d, t] is feature
 * d of stream s. The mixer of one block computes
 *
 *   Rn      = per-stream RMSNorm of R with unit-offset weight w [S*H]
 *   z       = [W_down; W_inject] Rn
 *   m       = SiLU(z[0:rank] / S)
 *   inject  = 2 * sigmoid(z[rank:rank+S] / S)
 *   u       = W_up m
 *   x[d]    = (1/S) * sum_s sigmoid(u[s*H+d]) * Rn[s*H+d]  -> hyper_connection_mix (all of the above)
 *
 * and after the block computes y from x:
 *
 *   R[s*H+d] += inject[s] * y[d]                            -> hyper_connection_inject
 *
 * Every Op evaluates in FP32 from the represented BF16 inputs and rounds each output once. Its
 * oracle is the formula in FP64; storage rounding belongs to the Op's numerical criterion.
 */

/// The mixer of one block: x = BF16 [H, T] from the residual R (contiguous BF16 [S*H, T]), the norm
/// weight (BF16 [S*H]), W_down [rank (+ S), S*H] and W_up [S*H, rank]; when inject is not null,
/// down carries the S injection rows and inject receives FP32 [S, T]. eps > 0.
///
/// Routes (private, chosen from the weights and T only): Q8_G32_FP16 RowSplit weights of the
/// registered geometry (S = 4, H = 2560, rank a multiple of 8 up to 512) at T <= 16 run two fused
/// kernels with every intermediate (Rn, z, m, u) in FP32, so a column's result does not depend on
/// T within 1..16. Other weights and wider calls run the norm, linear (with `policy`), gates,
/// linear and collapse steps with BF16 intermediates. `workspace` is caller-owned transient
/// storage of hyper_connection_mix_workspace_capacity_bytes; it must not overlap any operand.
void hyper_connection_mix(const Tensor& residual, const Tensor& norm_weight, const Weight& down,
                          const Weight& up, LinearPolicy policy, std::int32_t streams,
                          std::int32_t rank, float eps, Tensor& x, Tensor* inject,
                          WorkspaceArena& workspace, cudaStream_t stream);

/// The workspace hyper_connection_mix needs for every T in [1, max_tokens].
[[nodiscard]] std::size_t hyper_connection_mix_workspace_capacity_bytes(
    const Weight& down, const Weight& up, LinearPolicy policy, std::int32_t streams,
    std::int32_t rank, std::int32_t max_tokens);

/// residual[s*H+d,t] = bf16(residual[s*H+d,t] + inject[s,t] * y[d,t]). y: BF16 [H, T];
/// inject: FP32 [S, T]; residual: BF16 [S*H, T], updated in place.
void hyper_connection_inject(const Tensor& y, const Tensor& inject, Tensor& residual,
                             cudaStream_t stream);

/// residual[s*H+d,t] = x[d,t] for every stream: the first block's input from the embedding.
void hyper_connection_expand(const Tensor& x, std::int32_t streams, Tensor& residual,
                             cudaStream_t stream);

} // namespace infernix::ops
