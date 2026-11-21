#pragma once

// BF16 matrices as 8-bit weights (LOSSY: opt-in switches only): Q8_G32_FP16 RowSplit in the row_split_k128_v1 order,
// int8 codes [n, padded_k] (padded_k = k rounded up to 128) and FP16 scales [n, padded_k / 32]. A group of 32 values
// has scale fl16(m / 127), m its largest magnitude, and codes rne(v / scale) clamped to [-127, 127] (a group whose m /
// 127 rounds to zero in FP16: scale 0, codes 0); padding codes and scales are zero. Used by Flash-Next's
// hyper-connection repack (--a8-activations on a GGUF import) and Inkling 2's --q8-weights.

#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

// Whether `w` is a matrix quantize_bf16_q8_g32 takes: BF16 Contiguous [n, k], k a multiple of 32, 16-byte aligned.
[[nodiscard]] bool q8_g32_quantizable(const Weight& w) noexcept;

// A row's code bytes: k rounded up to 128.
[[nodiscard]] std::int32_t q8_g32_padded_columns(std::int32_t k) noexcept;

// The bytes of a [n, k] matrix's Q8 form with the scales right after the codes.
[[nodiscard]] std::size_t q8_g32_bytes(std::int32_t n, std::int32_t k) noexcept;

// Writes the Q8 form of `src` on `stream`: the codes to `codes` (n * padded_k bytes, 16-byte aligned) and the scales
// to `scales` (n * padded_k / 16 bytes, 2-byte aligned), neither overlapping src. Returns the Weight describing them
// (qdata the codes, scales the scale plane, padded_shape[1] = padded_k).
[[nodiscard]] Weight quantize_bf16_q8_g32(const Weight& src, void* codes, void* scales, cudaStream_t stream);

} // namespace infernix::ops
