#pragma once

// Q4 D256 key codec of the Q4KeyVq2Value format: the fixed normalized Hadamard row y = R*x is
// coded per coordinate with a symmetric 16-level Lloyd-Max quantizer for a unit Gaussian, the
// levels rounded to integers in units of 1/32 (+-{4,12,21,30,40,52,66,87}). One row stores 128
// code bytes (coordinate 2j in the low nibble of byte j) and one FP16 scale S; the represented
// value of coordinate i is S * c[i] with c the signed integer level.
//
// Encoding (exact contract, FP32 from represented BF16 inputs):
//   sigma = sqrt(sum(y^2) / 256); a row with sigma == 0 stores code 7 everywhere and S = 0
//   A_i   = 32 * |y_i| / sigma, m_i = number of thresholds {8,16.5,25.5,35,46,59,76.5} below A_i
//   code  = y_i < 0 ? 7 - m_i : 8 + m_i, c_i = sign * L[m_i]
//   S     = FP16_RNE(sum(y^2) / sum(y * c))

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheQ4CodeBytes = 128;

// Positive integer levels; nibble n decodes to L[n ^ 8] for n >= 8 and -L[n ^ 7] below.
inline constexpr std::int8_t kKVCacheQ4Levels[8] = {4, 12, 21, 30, 40, 52, 66, 87};
inline constexpr float kKVCacheQ4Thresholds[7] = {8.0f, 16.5f, 25.5f, 35.0f, 46.0f, 59.0f, 76.5f};

__device__ __forceinline__ int kv_cache_q4_magnitude_index(float A) {
    return (A > 8.0f ? 1 : 0) + (A > 16.5f ? 1 : 0) + (A > 25.5f ? 1 : 0) + (A > 35.0f ? 1 : 0) +
           (A > 46.0f ? 1 : 0) + (A > 59.0f ? 1 : 0) + (A > 76.5f ? 1 : 0);
}

// Levels packed little-endian: L[0..3] and L[4..7].
inline constexpr std::uint32_t kKVCacheQ4LevelsLow  = 0x1e150c04u;
inline constexpr std::uint32_t kKVCacheQ4LevelsHigh = 0x57423428u;

__device__ __forceinline__ int kv_cache_q4_level(int m) {
    return static_cast<int>(
        __byte_perm(kKVCacheQ4LevelsLow, kKVCacheQ4LevelsHigh, static_cast<unsigned>(m)) & 0xffu);
}

// Eight nibble codes (coordinate j in nibble j of `word`) to eight signed INT8 levels.
__device__ __forceinline__ uint2 kv_cache_q4_decode_word(std::uint32_t word) {
    constexpr std::uint32_t kLow  = kKVCacheQ4LevelsLow;
    constexpr std::uint32_t kHigh = kKVCacheQ4LevelsHigh;
    const std::uint32_t positive = (word >> 3) & 0x11111111u;
    // Magnitude index: n & 7 for positive codes, (n & 7) ^ 7 for negative ones.
    const std::uint32_t index =
        (word & 0x77777777u) ^ (~(positive * 7u) & 0x77777777u);
    const std::uint32_t lo_mag = __byte_perm(kLow, kHigh, index & 0xffffu);
    const std::uint32_t hi_mag = __byte_perm(kLow, kHigh, index >> 16);
    const auto spread = [](std::uint32_t nibbles) {
        // bit 4k of `nibbles` -> byte k all ones when the code is negative
        const std::uint32_t bits = (nibbles & 1u) | ((nibbles >> 4 & 1u) << 8) |
                                   ((nibbles >> 8 & 1u) << 16) | ((nibbles >> 12 & 1u) << 24);
        return (bits ^ 0x01010101u) * 0xffu;
    };
    const std::uint32_t lo_neg = spread(positive & 0xffffu);
    const std::uint32_t hi_neg = spread(positive >> 16);
    return make_uint2(__vsub4(lo_mag ^ lo_neg, lo_neg), __vsub4(hi_mag ^ hi_neg, hi_neg));
}

} // namespace ninfer::ops
