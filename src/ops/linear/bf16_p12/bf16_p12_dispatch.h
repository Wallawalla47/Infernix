#pragma once

// infernix::ops::detail - Linear over bf16_p12 weights (core/weight.h QuantLayout::P12Rows).

#include "core/tensor.h"
#include "core/weight.h"
#include "infernix/ops/linear.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

// A call on a registered shape (bf16_p12_routes.h) runs its route for the call's width; under a column-invariant
// policy, and on another shape, the packed skinny GEMV runs 16 columns a pass (its bits for a column depend only
// on K). No route takes workspace.
void bf16_p12_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy, cudaStream_t stream);

// Decodes a bf16_p12 matrix into a contiguous BF16 [n, k] at `dst` (16-byte aligned): the device decoder every route
// shares, for checks.
void bf16_p12_unpack(const Weight& weight, void* dst, cudaStream_t stream);

} // namespace infernix::ops::detail
