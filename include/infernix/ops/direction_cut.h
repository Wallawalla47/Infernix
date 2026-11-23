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

/**
 * Op: residual delta cut (the single-stream residual of Qwen3.5 and Qwen3.5-MoE; docs/ablate.md)
 *
 * The lane has already written its output into x by its own kernels; `before` is x as it was before that write. The
 * cut removes the direction's component of what the write added.
 *
 * Math / indexing:
 *   For each column t, with delta[h, t] = x[h, t] - before[h, t] and c[t] = sum_h dir[h] * delta[h, t]:
 *     x[h, t] = bf16(x[h, t] - alpha * c[t] * dir[h])
 *
 * Logical shapes: before and x BF16 [H, T]; dir FP32 [H]. T >= 1.
 *
 * Supported domain: H a multiple of 8, at most 8192; alpha finite in (0, 4]. An inactive cut leaves x unchanged.
 *
 * Numeric:
 *   delta, c and the cut value are FP32 (delta exact unless the two BF16 values are 2^16 apart in magnitude; c in a
 *   fixed order per column); x is rounded
 *   to BF16 once more (round to nearest even), so a cut too small to move a residual element keeps its bits. Against
 *   the ideal x0 + y - alpha (dir . y) dir of a lane output y written as bf16(x0 + y) with rounding error r, the
 *   result is x0 + y + (r - alpha (dir . r) dir) plus that last rounding: the lane write's own error, its component
 *   along dir scaled by 1 - alpha. A column's bits depend only on that column.
 *
 * Effects: x is updated in place; before and dir are only read; x must not overlap them.
 *
 * Workspace: none. Execution: stream-ordered; graph-capturable.
 */
void residual_delta_cut(const Tensor& before, const DirectionCut& cut, Tensor& x, cudaStream_t stream);

} // namespace infernix::ops
