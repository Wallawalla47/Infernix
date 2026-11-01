#pragma once

#include "core/tensor.h"
#include "infernix/ops/gated_rmsnorm.h"
#include "infernix/ops/ggml_linear.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

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

// --a8-activations (LOSSY, opt-in): the same for a projection on ggml_linear's 8-bit narrow route (LinearRoutes::GgmlA8,
// 1-31 columns). The producer writes the projection's 8-bit columns (ops/ggml/ggml_a8.h's codec: per 32-element slice
// int8 codes, an FP32 scale and, for a segment with a min term, the code sum), word for word what ggml_linear's 8-bit
// route would encode from the producer's BF16 result with the given permutation, so the projection's output bits are
// unchanged and it runs without its encoding launch.

/**
 * Whether a projection of `columns` columns of K = k with these segments runs on the 8-bit narrow route under
 * `routes` and takes gated_rmsnorm_ggml_a8_encoded's columns: K a multiple of 128 up to 8,192 and no permutation, or
 * one in blocks of 128 (a norm row each).
 */
[[nodiscard]] bool ggml_a8_takes_encoded(std::span<const GgmlSegment> segments, std::int32_t k, std::int32_t columns,
                                         LinearRoutes routes, const std::int32_t* input_perm, std::int32_t perm_block);

/** The I32 words of one 8-bit encoded column of K = k for these segments. */
[[nodiscard]] std::int32_t ggml_a8_encoded_column_words(std::span<const GgmlSegment> segments, std::int32_t k);

/**
 * gated_rmsnorm over rows of 128, x [128, H T] (z the same shape, weight [128]), whose output, read as [128 H, T], is
 * encoded for the segments' projection into `encoded` (I32 [ggml_a8_encoded_column_words(segments, 128 H), T]).
 */
void gated_rmsnorm_ggml_a8_encoded(const Tensor& x, const Tensor& weight, const Tensor& z, RmsGate gate, float eps,
                                   const std::int32_t* input_perm, std::int32_t perm_block,
                                   std::span<const GgmlSegment> segments, Tensor& encoded, cudaStream_t stream);

/** ggml_linear on the 8-bit narrow route over columns of K = k a producer encoded (`encoded`, as above). */
void ggml_linear_a8_encoded(const Tensor& encoded, std::int32_t k, std::span<const GgmlSegment> segments,
                            std::span<Tensor* const> outs, cudaStream_t stream);

} // namespace infernix::ops
