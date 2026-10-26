#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

// The capacity of every route the problem may select over the interval, including the
// fp8-a8-pair and f16-accumulate-prefill routes (the query takes no route flags).
[[nodiscard]] std::size_t fp8_linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  LinearRoutes routes, WorkspaceArena* workspace, cudaStream_t stream);

// Whether the problem (fp8_e4m3fn_row_bf16 or fp8_e4m3fn_row_f32) registers a dedicated
// column-invariant route.
[[nodiscard]] bool fp8_linear_column_invariant_route(QType qtype, std::int32_t output_rows,
                                                     std::int32_t input_rows) noexcept;

} // namespace infernix::ops::detail
