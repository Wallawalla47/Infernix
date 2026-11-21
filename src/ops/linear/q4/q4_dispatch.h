#pragma once

#include "core/weight.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/lora.h"
#include "ops/linear/q4/q4_launch.h"

#include <cstdint>

namespace infernix::ops::detail {

Q4Launch select_q4_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t);
Q4Launch select_q4_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);
// Whether the problem has a LinearPolicy::A16ColumnInvariant route (the proposal heads).
[[nodiscard]] bool q4_column_invariant_route(std::int32_t n, std::int32_t k) noexcept;

void q4_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream);

// The A16 route with a LoRA term (include/infernix/ops/lora.h) added to each projected value before its BF16 store:
// the release instance of the call's width, at the widths linear_swiglu's Materialized route of the [34816,5120]
// gate/up parent runs (q4_lora_launch.cu); other routes throw.
void q4_lora_dispatch(const Tensor& x, const Weight& w, Tensor& out, const LoraAddend& lora, cudaStream_t stream);

} // namespace infernix::ops::detail
