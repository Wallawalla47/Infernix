#pragma once

// The UE8M0 group scales of fp8_e4m3fn_g32_e8m0 weights in g32_scale_rows16_v1 (row r, group g at
// (r / 16) * groups * 16 + g * 16 + r % 16, groups = K / 32). The format admits scale words
// 3 <= e <= 246, so 2^(e - 127) is a normal BF16 value and every code times its scale is an exact
// BF16 value (tools/artifact/codecs/fp8_g32.py).

#include <cstdint>

namespace infernix::ops::detail {

// 2^(e - 127) in both halves of a BF16 pair (a normal value for an admitted word).
__device__ __forceinline__ unsigned mxfp8_scale_bf16x2(std::uint32_t word) { return (word << 7) | (word << 23); }

// The BF16 pair of two E4M3 weights of one row and group times their scale: exact (a power of two
// times a BF16 value that stays normal or exactly subnormal over the admitted words).
__device__ __forceinline__ unsigned mxfp8_scaled_pair(unsigned bf16x2_bits, unsigned scale_bf16x2) {
    unsigned result;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(bf16x2_bits), "r"(scale_bf16x2));
    return result;
}

} // namespace infernix::ops::detail
