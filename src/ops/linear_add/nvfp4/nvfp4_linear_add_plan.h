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
// column's arithmetic does not depend on the call's other columns.
void nvfp4_linear_add_invariant_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                       cudaStream_t stream);
void nvfp4_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream);

void nvfp4_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                     cudaStream_t stream);

void nvfp4_linear_add_a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                Nvfp4A4Workspace workspace, cudaStream_t stream);
// AllowA4Residual: the workspace's planes (two or three) into one accumulator, MMA tiles only.
void nvfp4_linear_add_a4_residual_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                         Nvfp4A4Workspace workspace, cudaStream_t stream);

// `wide_down_tile`: LinearRoutes::WideDownTile (the batched-decode fast path's A16 tiles).
void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream,
                               bool wide_down_tile);

} // namespace infernix::ops::detail
