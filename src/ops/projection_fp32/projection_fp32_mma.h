#pragma once

// The tensor-core route of projection_fp32: BF16 vocabulary heads at 1..16 columns. Private to the Op.
// (q8_g32_fp16 heads have none: the Q8 MMA tiles dequantize weights to FP16 before the product, which
// misses the Op's FP32 bound; they keep the SIMT mappings.)

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

inline constexpr int kProjectionMmaMaxColumns = 16;

[[nodiscard]] bool projection_fp32_bf16_mma_supported(int rows, int k, int columns) noexcept;
// BF16 [rows, K] row-major weight; out FP32 [rows, T].
void projection_fp32_bf16_mma(const Tensor& x, const void* weight, int rows, Tensor& out, cudaStream_t stream);

} // namespace infernix::ops::detail
