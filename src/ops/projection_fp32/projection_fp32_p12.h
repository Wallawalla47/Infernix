#pragma once

// projection_fp32 over a bf16_p12 weight (projection_fp32_p12.cu). Private to the Op.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

void projection_fp32_bf16_p12(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

} // namespace infernix::ops::detail
