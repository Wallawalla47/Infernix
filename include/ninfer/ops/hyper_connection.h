#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Hyper-connection residual streams (docs/maintainer/qwen3_8-flash-next-design.md §2, §8.3).
 *
 * A residual R holds S streams of H features per column, stream-major: R[s*H + d, t] is feature
 * d of stream s. The mixer of one block computes
 *
 *   Rn      = per-stream RMSNorm of R with unit-offset weight w [S*H]
 *   z       = [W_down; W_inject] Rn                         (two projections, caller-owned)
 *   m       = SiLU(z[0:rank] / S)                           -> hyper_connection_gates
 *   inject  = 2 * sigmoid(z[rank:rank+S] / S)               -> hyper_connection_gates
 *   u       = W_up m                                        (caller-owned projection)
 *   x[d]    = (1/S) * sum_s sigmoid(u[s*H+d]) * Rn[s*H+d]   -> hyper_connection_collapse
 *
 * and after the block computes y from x:
 *
 *   R[s*H+d] += inject[s] * y[d]                            -> hyper_connection_inject
 *
 * Every Op evaluates in FP32 from the represented BF16 inputs and rounds each output once. Its
 * oracle is the formula in FP64; storage rounding belongs to the Op's numerical criterion.
 */

/// out[s*H+d,t] = R[s*H+d,t] * rsqrt(mean_d R[s*H+d,t]^2 + eps) * (1 + weight[s*H+d]).
/// residual, out: contiguous BF16 [S*H, T]; weight: contiguous BF16 [S*H]. H must be a multiple
/// of 8. Inputs and output must not overlap.
void hyper_connection_norm(const Tensor& residual, const Tensor& weight, std::int32_t streams,
                           float eps, Tensor& out, cudaStream_t stream);

/// mix = SiLU(z[0:rank, t] / S) as BF16 [rank, T]; when inject is not null, inject[s,t] =
/// 2 * sigmoid(z[rank+s, t] / S) as FP32 [S, T]. z is contiguous BF16 [rank (+ S), T].
void hyper_connection_gates(const Tensor& z, std::int32_t rank, std::int32_t streams, Tensor& mix,
                            Tensor* inject, cudaStream_t stream);

/// out[d,t] = (1/S) * sum_s sigmoid(mix_logits[s*H+d,t]) * normalized[s*H+d,t], BF16 [H, T].
void hyper_connection_collapse(const Tensor& mix_logits, const Tensor& normalized,
                               std::int32_t streams, Tensor& out, cudaStream_t stream);

/// residual[s*H+d,t] = bf16(residual[s*H+d,t] + inject[s,t] * y[d,t]). y: BF16 [H, T];
/// inject: FP32 [S, T]; residual: BF16 [S*H, T], updated in place.
void hyper_connection_inject(const Tensor& y, const Tensor& inject, Tensor& residual,
                             cudaStream_t stream);

/// residual[s*H+d,t] = x[d,t] for every stream: the first block's input from the embedding.
void hyper_connection_expand(const Tensor& x, std::int32_t streams, Tensor& residual,
                             cudaStream_t stream);

} // namespace ninfer::ops
