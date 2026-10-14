#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/linear.h"

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
// A16ColumnInvariant: one sliced-K schedule (8 K-warps) for every T, so a column's arithmetic does
// not depend on the call's other columns.
void nvfp4_linear_swiglu_invariant_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          cudaStream_t stream);
void nvfp4_linear_swiglu_a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                   WorkspaceArena& workspace, cudaStream_t stream);
// AllowA4Residual: kNvfp4A4ResidualTerms NVFP4 activation planes into one accumulator.
void nvfp4_linear_swiglu_a4_residual_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                            WorkspaceArena& workspace, cudaStream_t stream);

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena& workspace,
                                  cudaStream_t stream);

} // namespace infernix::ops::detail
