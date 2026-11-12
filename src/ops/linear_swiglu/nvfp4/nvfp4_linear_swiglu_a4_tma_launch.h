#pragma once
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_operands.h"

namespace infernix::ops::detail {
void launch_nvfp4_linear_swiglu_a4_tma(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                       cudaStream_t stream);
// The same GEMM (x's scales Tiled256) with its SwiGLU product encoded as the down projection's one A4 plane
// (`product`: codes [T, rows / 4], scales `product_layout`, RowMajor or Tiled256, with the down projection's
// divisor) instead of stored: the fused dense MLP's prompt profile. The bytes nvfp4_a4_quantize_kernel writes from
// the stored product. With Tiled256 scales, columns from `row_major_from` (a whole-tile boundary) take row-major scales
// after the whole tiles' (the down projection's MMA call over the remainder, ZK-09).
void launch_nvfp4_linear_swiglu_a4_tma_planes(const Nvfp4A4Operands& p, Nvfp4A4Workspace product,
                                              float down_input_scale_divisor, Nvfp4ScaleLayout product_layout,
                                              cudaStream_t stream, int row_major_from = 0x7fffffff);
// The A4 residual planes (kNvfp4A4ResidualTerms) over one 128-token tile, scales Tiled128.
void launch_nvfp4_linear_swiglu_a4_residual_tma(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                                cudaStream_t stream);
} // namespace infernix::ops::detail
