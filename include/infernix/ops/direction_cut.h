#pragma once

#include "core/tensor.h"
#include "infernix/ops/offloaded_sparse_moe.h"

#include <cuda_runtime.h>

namespace infernix::ops {

// One lane's cut (--ablate, docs/ablate.md): a unit direction and its strength at one layer. Inactive (no direction,
// or alpha 0) means no cut: the calls below are then the plain inject's.
struct DirectionCut {
    const float* direction = nullptr; // FP32 [H] on the device, unit length
    float alpha            = 0.0F;

    [[nodiscard]] bool active() const noexcept { return direction != nullptr && alpha != 0.0F; }
};

/**
 * Op: direction cut and inject (directional ablation of a residual write; docs/ablate.md)
 *
 * Math / indexing:
 *   For each column t, with y = the lane's output column y[:, t] and c[t] = sum_h dir[h] * y[h, t]:
 *     residual[s*H + h, t] = bf16(residual[s*H + h, t] + inject[s, t] * (y[h, t] - alpha * c[t] * dir[h]))
 *   hyper_connection_cut_inject takes y as given. moe_combine_cut_inject first forms y as moe_combine_inject does:
 *     y[h, t] = bf16(sum_k weights[k, t] * outputs[h, t*K + k] + shared_gate[t] * shared[h, t]), k in rank order.
 *
 * Logical shapes:
 *   y BF16 [H, T]; dir FP32 [H]; inject FP32 [S, T]; residual BF16 [S*H, T]. outputs BF16 [H, T*K], routing.weights
 *   FP32 [K, T], routing.shared_gate FP32 [T], shared BF16 [H, T]. T >= 1.
 *
 * Supported domain:
 *   H a multiple of 8, at most 4096; S in [1, 4]; K in [1, 16]; alpha finite in (0, 4]; no column mask. An inactive
 *   cut runs hyper_connection_inject / moe_combine_inject unchanged.
 *
 * Numeric:
 *   c and the cut value are FP32; the cut value is not rounded before the inject, so each residual element keeps the
 *   inject's one BF16 rounding (round to nearest even). The combine's y is BF16-rounded, as moe_combine_inject's is.
 *   A column's bits depend only on that column (not on T or the column's position), and repeat call for call.
 *
 * Effects:
 *   residual is updated in place; y, outputs, shared, inject and dir are only read. residual must not overlap them.
 *
 * Workspace: none.
 *
 * Execution:
 *   A programmatic dependent of the preceding kernel while a CUDA Graph captures (as the plain inject); otherwise
 *   stream-ordered.
 */
void hyper_connection_cut_inject(const Tensor& y, const DirectionCut& cut, const Tensor& inject, Tensor& residual,
                                 cudaStream_t stream);

void moe_combine_cut_inject(const Tensor& outputs, const MoeRouting& routing, const Tensor& shared,
                            const DirectionCut& cut, const Tensor& inject, Tensor& residual, cudaStream_t stream);

} // namespace infernix::ops
