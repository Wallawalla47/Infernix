#pragma once
#include "core/weight.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace infernix::ops::detail {
using Nvfp4Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);
}
