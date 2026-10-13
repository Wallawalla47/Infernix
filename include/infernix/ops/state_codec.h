#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

// Compact storage forms of an FP32 state tensor (a snapshot's linear-attention recurrent state on
// the prefix cache's Host tier).
enum class PackedStateFormat : std::uint8_t {
    // Sign, exponent and the 15 high mantissa bits: 3 bytes per element.
    Fp24,
    // BF16: 2 bytes per element.
    Bf16,
};

// Packed bytes of `elements` values in `format`.
[[nodiscard]] std::size_t packed_state_bytes(std::int64_t elements, PackedStateFormat format);

/**
 * Op: pack_fp32_state
 *
 * Math / indexing:
 *   packed element i = round_to_nearest_even(source[i]) in `format`, for every logical element i in
 *   contiguous order. Fp24 element i occupies packed bytes [3i, 3i + 3), little endian, and holds
 *   bits 31..8 of the rounded FP32 value; Bf16 element i occupies bytes [2i, 2i + 2).
 *
 * Supported domain:
 *   Source is a contiguous FP32 tensor whose element count is divisible by four, 16-byte aligned;
 *   `packed` is 4-byte aligned and holds packed_state_bytes(numel, format) bytes.
 *
 * Numeric:
 *   Round to nearest, ties to even, on the dropped low mantissa bits; subnormals round on the same
 *   bits. Finite values that round past the largest finite value become infinities; infinities
 *   stay infinities; a NaN stays a NaN (its payload and sign are not defined).
 *
 * Effects:
 *   Writes every packed byte. Source and packed must not alias.
 *
 * Workspace:
 *   None.
 */
void pack_fp32_state(const Tensor& source, void* packed, PackedStateFormat format,
                     cudaStream_t stream);

/**
 * Op: unpack_fp32_state
 *
 * Math / indexing:
 *   destination[i] = the FP32 value of packed element i (exact: the dropped mantissa bits are
 *   zero), with pack_fp32_state's packed layout.
 *
 * Supported domain:
 *   As pack_fp32_state, with destination in place of source.
 *
 * Numeric:
 *   Exact.
 *
 * Effects:
 *   Writes the full destination. Packed and destination must not alias.
 *
 * Workspace:
 *   None.
 */
void unpack_fp32_state(const void* packed, Tensor& destination, PackedStateFormat format,
                       cudaStream_t stream);

} // namespace infernix::ops
