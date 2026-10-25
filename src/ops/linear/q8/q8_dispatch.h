#pragma once

#include "core/arena.h"
#include "core/weight.h"
#include "infernix/ops/linear.h"
#include "ops/linear/q8/q8_launch.h"

#include <cstdint>

namespace infernix::ops::detail {

Q8Launch select_q8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearRoutes routes);
// Whether the problem registers a dedicated column-invariant route.
[[nodiscard]] bool q8_column_invariant_route(std::int32_t n, std::int32_t k) noexcept;
Q8Launch select_q8_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy,
                          LinearRoutes routes);

// LinearRoutes::F16Accumulate on a registered problem at prefill widths takes the f16-accumulate route
// (q8_f16acc.h), whose encoded activation needs `workspace`.
void q8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 LinearRoutes routes, WorkspaceArena* workspace, cudaStream_t stream);

} // namespace infernix::ops::detail
