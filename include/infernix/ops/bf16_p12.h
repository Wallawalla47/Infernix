#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

/**
 * Lossless 12-bit packing of a BF16 matrix (QType::BF16_P12, QuantLayout::P12Rows; core/weight.h has the
 * planes and docs/maintainer/tensor-formats.md the format). Each weight keeps its sign and mantissa in a byte
 * and its exponent as a 4-bit code from its row's base, the lowest exponent of the 15-exponent window holding
 * most of the row's nonzero weights (ties to the lower base); a zero, a subnormal and every exponent outside
 * the window is an escape, kept exactly in the row's escape list. Every weight decodes to its BF16 bits.
 *
 * Packing runs on the device in two passes over a contiguous BF16 [n, k] matrix (k a multiple of 8, 16-byte
 * aligned): bf16_p12_count finds each row's base and escape count and the matrix's escapes, which size
 * the destination (core/weight_view.h bf16_p12_geometry); bf16_p12_pack then writes the planes there. Both
 * keep their per-row state in caller-owned `scratch` of bf16_p12_pack_scratch_bytes(n) bytes, which must not
 * overlap the source or the destination; the destination must not overlap the source.
 */
[[nodiscard]] std::size_t bf16_p12_pack_scratch_bytes(std::int32_t n);

// Pass 1, enqueued on `stream`: the matrix's escape count lands in the device word `escapes`, so a loader can enqueue
// every matrix's count and read them back at once.
void bf16_p12_count(const void* bf16, std::int32_t n, std::int32_t k, void* scratch, std::uint32_t* escapes,
                    cudaStream_t stream);

// Pass 2, after bf16_p12_count with the same matrix and scratch, `escapes` its result: the planes at `dst`
// (256-byte aligned, of bf16_p12_geometry(n, k, escapes).bytes). Enqueued on `stream`.
void bf16_p12_pack(const void* bf16, std::int32_t n, std::int32_t k, std::uint64_t escapes, void* scratch, void* dst,
                   cudaStream_t stream);

// Whether ops::linear registers tuned routes for a packed [n, k] matrix (every other shape runs the packed skinny
// GEMV, 16 columns a pass): a loader packs only such matrices.
[[nodiscard]] bool bf16_p12_linear_tuned(std::int32_t n, std::int32_t k) noexcept;

// Whether ops::projection_fp32 takes a packed [n, k] head (n a multiple of 16, k of 128 and at least 512).
[[nodiscard]] bool bf16_p12_projection_fp32_supported(std::int32_t n, std::int32_t k) noexcept;

} // namespace infernix::ops
