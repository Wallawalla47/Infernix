#pragma once

// projection_fp32 with an explicit mapping, for its benchmark and test. The narrow, wide and tall
// mappings accumulate a column in the same order, so among them the route changes speed only; the
// tensor-core mapping (BF16 vocabulary heads at 1..16 columns) sums in MMA order. Private to the Op.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <span>

namespace infernix::ops::detail {

enum class ProjectionRoute {
    Automatic, // what projection_fp32 selects
    Narrow,    // the narrow (or, from 64 columns, wide) mapping
    Tall,      // the tall mapping (BF16 and q8_g32_fp16, at most 16 columns)
    Mma,       // the tensor-core mapping (one BF16 segment, at most 16 columns)
};

void projection_fp32_route(const Tensor& x, const Weight& weight, Tensor& out, ProjectionRoute route,
                           cudaStream_t stream);
void projection_fp32_route(const Tensor& x, std::span<const Tensor* const> weights, Tensor& out,
                           ProjectionRoute route, cudaStream_t stream);

} // namespace infernix::ops::detail
