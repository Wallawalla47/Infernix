#pragma once
#include "core/weight.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace infernix::ops::detail {
struct DynamicConvPrepareRoute {
    int rows;
    int columns;
    int split_k;
    bool fused = false; // bf16_dynamic_grouped_conv_prepare_fused_launch instead of the three launches
};

void bf16_dynamic_grouped_conv_prepare_partial_launch(DynamicConvPrepareRoute route,
                                                      const Tensor& input, const Weight& weight,
                                                      float* partial, cudaStream_t stream);
void bf16_dynamic_grouped_conv_prepare_reduce_launch(DynamicConvPrepareRoute route,
                                                     const Tensor& base, const float* partial,
                                                     Tensor& prepared, Tensor& finish,
                                                     cudaStream_t stream);
// The whole prepare in one kernel (bf16_dynamic_grouped_conv_prepare_fused.cu): H 2560, at most 16 columns.
bool bf16_dynamic_grouped_conv_prepare_fused_supported(int hidden, int tokens);
void bf16_dynamic_grouped_conv_prepare_fused_launch(const Tensor& residual, const Tensor& norm, float eps,
                                                    const Tensor& base, const Weight& weight, Tensor& prepared,
                                                    Tensor& finish, cudaStream_t stream);
} // namespace infernix::ops::detail
