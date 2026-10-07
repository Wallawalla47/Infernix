#pragma once

// infernix::ops::detail - private launch prototype for rmsnorm.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

// A non-null z gates the output with SiLU(z), or with sigmoid(z) when sigmoid_gate is set.
void rmsnorm_launch(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                    const Tensor* z, bool sigmoid_gate, Tensor& out, std::int32_t multiprocessor_count,
                    cudaStream_t stream);

} // namespace infernix::ops::detail
