#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

/**
 * Returns the transient capacity required by rmsnorm_dynamic_grouped_conv_prepare at hidden width
 * H (5120, 2560, 3072 or 2304) for every width/batch pair in the inclusive intervals. Width endpoints lie
 * in [2,16], batch endpoints in [1,8]. Capacity covers every shape-selected route in the requested
 * domain. It is zero where every such route needs none (H 2560 with W x B <= 16: one kernel).
 */
[[nodiscard]] std::size_t rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
    std::int32_t hidden, std::int32_t min_width, std::int32_t max_width,
    std::int32_t min_batch_size, std::int32_t max_batch_size);

/**
 * Op: rmsnorm_dynamic_grouped_conv_prepare
 *
 * Math / indexing:
 *   H=5120 (Qwen3.8-27B's DFlash2), H=2560 (Qwen3.8-Flash-Next's), H=3072 (Qwen3.5-122B-A10B's) or
 *   H=2304 (Inkling 2 Mini's), W=2..16, group width 16,
 *   G=H/16, sides=2, taps=2. For h=16*g+j:
 *
 *     inv[i,b] = rsqrt(sum_h residual[h,i,b]^2 / H + eps)
 *     n[h,i,b] = residual[h,i,b] * inv[i,b] * norm_weight[h]
 *     row(s,t,g) = (s*2+t)*G+g
 *     d[s,t,g,i,b] = sum_h kernel_projection_weight[row(s,t,g),h] * n[h,i,b]
 *
 *     prepared[h,i,b] =
 *         (base_kernel[h,0,0] + d[0,0,g,i,b]) * n[h,i,b]
 *       + I(i>0) * (base_kernel[h,1,0] + d[0,1,g,i,b]) * n[h,i-1,b]
 *
 *     finish_delta[g,t,i,b] = d[1,t,g,i,b].
 *
 * Logical shapes / supported domain:
 *   residual/prepared are contiguous BF16 [H,W,B], W is in [2,16] and B in [1,8];
 *   norm_weight is contiguous BF16 [H]. base_kernel is the runtime view BF16 [H,2,2] with
 * axes [channel,tap,side]; kernel_projection_weight is contiguous BF16 [4G,H]; and
 *   finish_delta is contiguous BF16 [G,2,W,B]. eps is positive and finite. Position zero has
 *   no previous-tap contribution: the Op never reads another request or an earlier round.
 *
 * Numeric:
 *   The oracle evaluates the complete formula naively in FP64 from the represented BF16 inputs.
 *   Both BF16 outputs are promoted and compared directly with those ideal values under their
 *   named reduction criteria. RMS reduction order, Tensor Core operand staging, projection
 *   factorization, split-K reduction, and private intermediate precision are implementation
 *   choices; neither n nor the input-side coefficient plane is an observable cast boundary.
 *
 * Effects:
 *   Writes every element of prepared and finish_delta and preserves every input. Inputs, the
 *   weight payload, both outputs, and live workspace must be pairwise non-overlapping. The Op has
 *   no persistent state side effect.
 *
 * Workspace:
 *   Caller-owned call-scoped storage sized by the capacity query above. The implementation does
 *   not allocate or retain device memory. A shape whose capacity is zero leaves the workspace
 *   unused, so it may then hold the call's tensors.
 */
void rmsnorm_dynamic_grouped_conv_prepare(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const Tensor& base_kernel,
                                          const Weight& kernel_projection_weight, Tensor& prepared,
                                          Tensor& finish_delta, WorkspaceArena& workspace,
                                          cudaStream_t stream);

/**
 * Returns the transient capacity required by linear_dynamic_grouped_conv_add for every width/batch
 * pair in the inclusive intervals. (hidden, input_rows) is (5120, 4096|17408), (2560,
 * 4096|7680), (3072, 4096|9216) or (2304, 4096|6912), widths lie in [2,16], and batch sizes in [1,8]. Capacity follows every
 * shape-selected production route of a projection_weight in `qtype` (Q8_G32_FP16, or NVFP4 at (2304, 6912)).
 */
[[nodiscard]] std::size_t linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
    std::int32_t hidden, std::int32_t input_rows, std::int32_t min_width, std::int32_t max_width,
    std::int32_t min_batch_size, std::int32_t max_batch_size, QType qtype = QType::Q8_G32_FP16);

/**
 * Op: linear_dynamic_grouped_conv_add
 *
 * Math / indexing:
 *   H=5120 with C in {4096, 17408} (Qwen3.8-27B's DFlash2), H=2560 with C in {4096, 7680}
 *   (Qwen3.8-Flash-Next's), H=3072 with C in {4096, 9216} (Qwen3.5-122B-A10B's) or H=2304 with C in
 *   {4096, 6912} (Inkling 2 Mini's), W=2..16, group width 16, G=H/16. For h=16*g+j:
 *
 *     z[h,i,b] = sum_c projection_weight[h,c] * x[c,i,b]
 *
 *     residual[h,i,b] +=
 *         (base_kernel[h,0,1] + finish_delta[g,0,i,b]) * z[h,i,b]
 *       + I(i>0) * (base_kernel[h,1,1] + finish_delta[g,1,i,b]) * z[h,i-1,b].
 *
 * Logical shapes / supported domain:
 *   x is contiguous BF16 [C,W,B]; projection_weight is Q8_G32_FP16 RowSplit [H,C] with
 *   unpadded C, or at (H, C) = (2304, 6912) NVFP4 in block_scale_k16_m128x4 (Inkling 2 Mini's drafter's MLP
 *   down: the NVFP4 linear's A16 route projects, then the same finish); base_kernel is contiguous BF16 [H,2,2] with axes [channel,tap,side];
 *   finish_delta is contiguous BF16 [G,2,W,B]; and residual is contiguous BF16 [H,W,B].
 *   W is in [2,16] and B in [1,8]. Position zero has no previous-tap contribution: the Op never
 * reads another request or an earlier round.
 *
 * Numeric:
 *   The oracle decodes each signed Q8 value with its stored FP16 group scale (an NVFP4 value: its E2M1 code
 *   with its E4M3 block scale and the matrix's divisor) and evaluates the complete formula in FP64 from
 *   represented inputs. The projection is a private intermediate;
 *   the contract does not prescribe its storage or arithmetic precision. The implementation
 *   writes the final residual in BF16.
 *
 * Effects:
 *   Updates every element of residual in place and preserves x, projection_weight, base_kernel,
 *   and finish_delta. All operand storage, both weight planes, and live workspace must be pairwise
 *   non-overlapping. The Op has no persistent state side effect.
 *
 * Workspace:
 *   Caller-owned call-scoped storage sized by the capacity query above. The implementation does
 *   not allocate or retain device memory.
 */
void linear_dynamic_grouped_conv_add(const Tensor& x, const Weight& projection_weight,
                                     const Tensor& base_kernel, const Tensor& finish_delta,
                                     Tensor& residual, WorkspaceArena& workspace,
                                     cudaStream_t stream);

} // namespace infernix::ops
