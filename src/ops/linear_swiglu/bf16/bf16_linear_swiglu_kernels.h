#pragma once

// BF16 LinearSwiGLU kernels (Qwen3.8-Flash-Next's shared expert in the bit-exact artifact). Private to
// the Op.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

// Columns the register-streamed route serves; wider calls compose the tuned BF16 GEMM with SwiGLU.
inline constexpr std::int32_t kBf16LinearSwiGluStreamMaxColumns = 16;

[[nodiscard]] bool bf16_linear_swiglu_stream_pair_supported(const Tensor& x, const Weight& w,
                                                             const Tensor& out) noexcept;

// out [M, T] = SiLU(W[0:M] x) * W[M:2M] x for gate/up [2M, 2560] BF16 contiguous, 1..16 columns.
void bf16_linear_swiglu_stream_pair_launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace infernix::ops::detail
