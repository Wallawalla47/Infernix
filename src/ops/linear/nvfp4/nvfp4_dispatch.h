#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "infernix/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

[[nodiscard]] std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                                std::int32_t input_rows,
                                                                LinearPolicy policy,
                                                                std::int32_t min_tokens,
                                                                std::int32_t max_tokens);

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream);

// A8Only: x encoded by ggml_a8.h's codec, then the 8-bit tiles (nvfp4_a8_mma.cuh) at any registered shape.
void nvfp4_linear_a8_launch(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
                            cudaStream_t stream);

// Whether the registered [N, K] has an A16ColumnInvariant route (the 27B NVFP4 head's).
[[nodiscard]] bool nvfp4_linear_column_invariant_route(std::int32_t output_rows, std::int32_t input_rows) noexcept;

} // namespace infernix::ops::detail
