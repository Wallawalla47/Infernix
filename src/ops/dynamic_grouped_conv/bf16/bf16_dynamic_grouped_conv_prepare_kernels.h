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

// The R16 tiles' split of K (64-wide K tiles): 8, or 9 for H 2304 (Inkling 2 Mini's 36 tiles, which 8 does not divide).
constexpr int dynamic_conv_prepare_r16_split(int hidden) { return hidden == 2304 ? 9 : 8; }

void bf16_dynamic_grouped_conv_prepare_partial_launch(DynamicConvPrepareRoute route,
                                                      const Tensor& input, const Weight& weight,
                                                      float* partial, cudaStream_t stream);
void bf16_dynamic_grouped_conv_prepare_reduce_launch(DynamicConvPrepareRoute route,
                                                     const Tensor& base, const float* partial,
                                                     Tensor& prepared, Tensor& finish,
                                                     cudaStream_t stream);
// The whole prepare in one kernel (bf16_dynamic_grouped_conv_prepare_fused.cu): H 2560 at most 16 columns,
// H 2304 any call (chunks of whole rows on the grid's second dimension).
bool bf16_dynamic_grouped_conv_prepare_fused_supported(int hidden, int tokens);
void bf16_dynamic_grouped_conv_prepare_fused_launch(const Tensor& residual, const Tensor& norm, float eps,
                                                    const Tensor& base, const Weight& weight, Tensor& prepared,
                                                    Tensor& finish, cudaStream_t stream);
} // namespace infernix::ops::detail
