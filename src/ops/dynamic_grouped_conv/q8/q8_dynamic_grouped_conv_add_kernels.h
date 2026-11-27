#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

enum class Q8DynamicConvAddSchedule { TiledMma, MmaK128 };

void q8_dynamic_grouped_conv_add_materialized_launch(Q8DynamicConvAddSchedule schedule,
                                                     const Tensor& x, const Weight& weight,
                                                     const Tensor& base_kernel,
                                                     const Tensor& finish_delta, Tensor& residual,
                                                     Tensor& projected, cudaStream_t stream);

// The finish alone over an already projected BF16 [H, W, B] (`width` W, `tokens` W x B): an NVFP4 parent's route, whose
// projection the NVFP4 linear makes.
void dynamic_grouped_conv_add_finish_launch(const Tensor& projected, const Tensor& base_kernel,
                                            const Tensor& finish_delta, Tensor& residual, int width, int tokens,
                                            cudaStream_t stream);

} // namespace infernix::ops::detail
