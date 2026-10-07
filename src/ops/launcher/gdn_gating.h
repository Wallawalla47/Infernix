#pragma once

// infernix::ops::detail - private launch prototype for gdn_gating.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

void gdn_gating_launch(const Tensor& a, const Tensor& b, const Tensor& A_log, const Tensor& dt_bias,
                       Tensor& g, Tensor& beta, cudaStream_t stream);

} // namespace infernix::ops::detail
