#pragma once

#include "core/tensor.h"
#include "infernix/ops/gated_rmsnorm.h"
#include "infernix/ops/ggml_linear.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops {

// Producers of a GGUF projection's input that write it encoded for ggml_linear_encoded (GgmlEncodedColumns) instead
// of as BF16, so the projection runs without its encoding launch. Each is bit for bit its Op followed by the encoding
// ggml_linear applies to x with the given input permutation (input_perm null: none; else perm_block elements a
// block, a multiple of 8, as ggml_linear takes them). out.k is the column length (ggml_encode_length); out.words and
// out.emax are shaped as GgmlEncodedColumns says. A cluster of out.k / 2048 CTAs takes a column.
inline constexpr std::int32_t kGgmlEncodeMaxElements = 8192;

/** Whether the producers take columns of k elements: a multiple of 2048 up to kGgmlEncodeMaxElements. */
[[nodiscard]] constexpr bool ggml_encode_length(std::int32_t k) noexcept {
    return k > 0 && k % 2048 == 0 && k <= kGgmlEncodeMaxElements;
}

/**
 * gated_rmsnorm over rows of 128, x [128, H T] (z the same shape, weight [128]), whose output, read as [128 H, T] (rows
 * t H .. t H + H - 1 are column t), is encoded into out (out.k = 128 H).
 */
void gated_rmsnorm_ggml_encoded(const Tensor& x, const Tensor& weight, const Tensor& z, RmsGate gate, float eps,
                                const std::int32_t* input_perm, std::int32_t perm_block, GgmlEncodedColumns& out,
                                cudaStream_t stream);

/** sigmoid_mul's x * sigmoid(gate) over contiguous BF16 [K, T] (x is not written), encoded into out (out.k = K). */
void sigmoid_mul_ggml_encoded(const Tensor& gate, const Tensor& x, const std::int32_t* input_perm,
                              std::int32_t perm_block, GgmlEncodedColumns& out, cudaStream_t stream);

} // namespace infernix::ops
