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

[[nodiscard]] std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                       std::int32_t min_tokens,
                                                                       std::int32_t max_tokens);

void nvfp4_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
void nvfp4_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
// A16ColumnInvariant: the sliced-K arithmetic of 8 K-warps for every T, so a column's arithmetic
// does not depend on the call's other columns. Calls of 17+ columns run the slice-major ring tile,
// narrower ones the sliced-K tiles; the two give the same bits (both are exposed for that test).
void nvfp4_linear_swiglu_invariant_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          cudaStream_t stream);
void nvfp4_linear_swiglu_invariant_sliced_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                                 cudaStream_t stream);
// T >= 17.
void nvfp4_linear_swiglu_invariant_ring_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                               cudaStream_t stream);
void nvfp4_linear_swiglu_a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                   WorkspaceArena& workspace, cudaStream_t stream);
// AllowA4Residual: kNvfp4A4ResidualTerms NVFP4 activation planes into one accumulator.
void nvfp4_linear_swiglu_a4_residual_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                            WorkspaceArena& workspace, cudaStream_t stream);
// The same GEMM (5-96 columns) from the x planes already written, its SwiGLU product encoded with
// `down_input_scale_divisor` into `product_planes` (row-major, K = 17408) instead of stored: the fused dense MLP.
void nvfp4_linear_swiglu_a4_residual_planes_launch(const Weight& weight, Nvfp4A4Workspace x_planes,
                                                   Nvfp4A4Workspace product_planes,
                                                   float down_input_scale_divisor, std::int32_t tokens,
                                                   cudaStream_t stream);
// The fused dense MLP (linear_swiglu.h): x += down(swiglu(gate_up(offset rmsnorm(x)))) under AllowA4Residual (two
// planes, 5-96 columns) or AllowA4 (one plane, prompts of kDenseMlpFusedPromptMinTokens or more).
void nvfp4_dense_mlp_fused_rmsnorm_launch(Tensor& residual, const Tensor& norm_weight, float eps,
                                          const Weight& gate_up, const Weight& down, LinearPolicy policy,
                                          WorkspaceArena& workspace, cudaStream_t stream);
[[nodiscard]] std::size_t nvfp4_dense_mlp_fused_rmsnorm_workspace_bytes(std::int32_t max_tokens);
// The fused norm's encodes alone (the op test compares them with rmsnorm + the residual quantizer, and with rmsnorm +
// the A4 quantizer's Tiled256 plane).
void launch_nvfp4_rmsnorm_a4_residual_quantize(const Tensor& residual, const Tensor& norm_weight, float eps,
                                               float input_scale_divisor, Nvfp4A4Workspace planes,
                                               cudaStream_t stream);
void launch_nvfp4_rmsnorm_a4_quantize_tiled(const Tensor& residual, const Tensor& norm_weight, float eps,
                                            float input_scale_divisor, Nvfp4A4Workspace plane, cudaStream_t stream);

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena& workspace,
                                  cudaStream_t stream);

} // namespace infernix::ops::detail
