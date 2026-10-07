#pragma once

#include "core/tensor.h"
#include "infernix/ops/gelu.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

void gelu_launch(Tensor& x, GeluMode mode, cudaStream_t stream);

} // namespace infernix::ops::detail
