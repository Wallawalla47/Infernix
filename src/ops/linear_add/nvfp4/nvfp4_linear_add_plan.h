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

[[nodiscard]] std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                                    std::int32_t input_rows,
                                                                    LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

// `wide_k`: the 8 K-warp K=17408 tiles at 5-16 tokens (LinearRoutes::WideDownTile; see launch_matrix).
void nvfp4_linear_add_a16_launch(const Tensor& x, const Weight& weight, Tensor& residual, bool wide_k,
                                 cudaStream_t stream);

// A16ColumnInvariant: one K-warp count per K for every T (8 at K=6144, 4 at K=17408), so a
// column's arithmetic does not depend on the call's other columns. Calls of 17+ columns (32+ at
// K=17408) run the slice-major ring tile of that count, narrower ones the sliced-K tiles; the two give
// the same bits (both are exposed for that test).
void nvfp4_linear_add_invariant_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                       cudaStream_t stream);
void nvfp4_linear_add_invariant_sliced_launch(const Tensor& x, const Weight& weight,
                                              Tensor& residual, cudaStream_t stream);
// T >= 17.
void nvfp4_linear_add_invariant_ring_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                            cudaStream_t stream);
void nvfp4_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream);

void nvfp4_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                     cudaStream_t stream);

void nvfp4_linear_add_a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                Nvfp4A4Workspace workspace, cudaStream_t stream);
// The A4 route's activation scale layout for a call of `tokens` columns: Tiled256 where it takes the TMA tile, else
// row-major (the MMA tiles).
[[nodiscard]] Nvfp4ScaleLayout nvfp4_linear_add_a4_scale_layout(std::int32_t tokens);
// The A4 GEMM from one plane already written in nvfp4_linear_add_a4_scale_layout(tokens) for `tokens` columns: the
// fused dense MLP's prompt profile (its SwiGLU epilogue wrote the plane).
void nvfp4_linear_add_a4_planes_launch(const Weight& weight, Tensor& residual, Nvfp4A4Workspace planes,
                                       std::int32_t tokens, cudaStream_t stream);
// AllowA4Residual: the workspace's planes (two or three) into one accumulator, MMA tiles only.
void nvfp4_linear_add_a4_residual_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                         Nvfp4A4Workspace workspace, cudaStream_t stream);
// The same GEMM from planes already written (row-major, kNvfp4A4ResidualTerms) for `tokens` columns: the fused
// dense MLP's down projection (its SwiGLU epilogue wrote the planes).
void nvfp4_linear_add_a4_residual_planes_launch(const Weight& weight, Tensor& residual,
                                                Nvfp4A4Workspace planes, std::int32_t tokens,
                                                cudaStream_t stream);

// `wide_down_tile`: LinearRoutes::WideDownTile (the batched-decode fast path's A16 tiles).
void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream,
                               bool wide_down_tile);

} // namespace infernix::ops::detail
