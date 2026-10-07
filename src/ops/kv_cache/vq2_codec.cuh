#pragma once

// VQ2 D256 KV-cache codec: the fixed normalized Hadamard row R*x is split into 32 eight-value
// words; each word stores one 16-bit code (a 9-bit magnitude-pattern index and seven sign bits,
// the eighth sign completing even parity) and each row stores one FP16 scale S. The represented
// value of rotated coordinate i is S * c[i], where c is the signed INT8 codebook entry in units of
// 1/16 of a unit-variance coordinate. 64 code bytes + 2 scale bytes per D256 vector.
//
// Encoding (exact contract, evaluated in FP32 from represented BF16 inputs):
//   y        = normalized_hadamard_d256(x)                      (R*x)
//   sigma    = sqrt(sum(y^2) / 256); a row with sigma == 0 stores all-zero codes and S = 0
//   A_j      = 16 * |y_j| / sigma for the word's coordinates j
//   odd      = number of coordinates with y_j < 0 is odd
//   t_j      = A_j * P[p][j]; dot = t_0 + t_1 + ... + t_7 (left to right); m = min_j t_j
//   D(p)     = fma(-2, dot, |P[p]|^2), then D(p) = fma(4, m, D(p)) if odd
//              (sum_j (A_j - P[p][j])^2 + odd * 4 * m minus the constant |A|^2)
//   p        = argmin D(p), smallest index on ties
//   signs    = sign(y_j) (negative iff y_j < 0); if odd, the sign of the first j minimizing
//              A_j * P[p][j] is flipped
//   c_j      = sign_j * P[p][j]
//   S        = FP16_RNE(sum(y^2) / sum(y * c)) over the row (sum(y*c) > 0 for every nonzero row)
// The scale makes every row's reconstruction unbiased along the input direction: <S*c, y> = |y|^2.

#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace infernix::ops {

inline constexpr int kKVCacheVq2HeadDim   = 256;
inline constexpr int kKVCacheVq2Words     = 32;
inline constexpr int kKVCacheVq2Patterns  = 512;
inline constexpr int kKVCacheVq2CodeBytes = 64;
inline constexpr std::uint32_t kKVCacheVq2PatternMask = 0x1ffu;
inline constexpr int kKVCacheVq2SignShift = 9;

// The INT8 codebook (pattern-major, 8 values per pattern), identical for host oracles and device.
inline constexpr std::int8_t kKVCacheVq2Codebook[kKVCacheVq2Patterns * 8] = {
#include "ops/kv_cache/vq2_codebook.inc"
};

// Device copy of kKVCacheVq2Codebook (defined once in vq2_codebook.cu); kernels stage it into
// shared memory. Only relocatable device code (the Op library) can reference it.
#ifdef __CUDACC_RDC__
extern __device__ std::int8_t g_kv_cache_vq2_codebook[kKVCacheVq2Patterns * 8];
#endif

// Spreads four mask bits to four bytes of 0x00/0xff.
__device__ __forceinline__ unsigned kv_cache_vq2_byte_mask(unsigned bits4) {
    return ((bits4 * 0x00204081u) & 0x01010101u) * 0xffu;
}

// Sign mask (bit j set = coordinate j negative) of one stored word code.
__device__ __forceinline__ unsigned kv_cache_vq2_sign_mask(std::uint32_t code) {
    const unsigned low7 = (code >> kKVCacheVq2SignShift) & 0x7fu;
    return low7 | ((__popc(low7) & 1u) << 7);
}

// Decodes one word to eight signed INT8 codebook values (little-endian bytes, coordinate j in
// byte j). `codebook` is the 4 KiB table in shared memory, 8-byte aligned.
__device__ __forceinline__ uint2 kv_cache_vq2_decode_word(std::uint32_t code,
                                                          const std::int8_t* codebook) {
    const uint2 magnitude =
        *reinterpret_cast<const uint2*>(codebook + 8 * (code & kKVCacheVq2PatternMask));
    const unsigned sign = kv_cache_vq2_sign_mask(code);
    const unsigned lo   = kv_cache_vq2_byte_mask(sign & 0xfu);
    const unsigned hi   = kv_cache_vq2_byte_mask(sign >> 4);
    return make_uint2(__vsub4(magnitude.x ^ lo, lo), __vsub4(magnitude.y ^ hi, hi));
}

// Expands staged code rows (64 bytes per key, row-major in shared memory) of `keys` keys into an
// INT8 tile. `destination(key, word)` returns the 8-byte-aligned shared address of the word's
// eight coordinates. Threads stride over (key, word) pairs.
template <typename Destination>
__device__ __forceinline__ void kv_cache_vq2_decode_tile(const std::uint8_t* staged, int keys,
                                                         const std::int8_t* codebook,
                                                         Destination destination, int thread,
                                                         int threads) {
    const auto* codes = reinterpret_cast<const std::uint16_t*>(staged);
    for (int item = thread; item < keys * kKVCacheVq2Words; item += threads) {
        const int key  = item / kKVCacheVq2Words;
        const int word = item - key * kKVCacheVq2Words;
        *reinterpret_cast<uint2*>(destination(key, word)) =
            kv_cache_vq2_decode_word(codes[item], codebook);
    }
}

// Distance (up to the word's constant |A|^2) of the word with scaled magnitudes A (16 * |y| /
// sigma) and parity `odd` to pattern p: `pattern` its FP32 values, `norm` the exact |P[p]|^2. The
// same expression and order are used by every encoder.
__device__ __forceinline__ float kv_cache_vq2_pattern_distance(const float (&A)[8], bool odd,
                                                               const float* pattern, float norm) {
    float t   = __fmul_rn(A[0], pattern[0]);
    float dot = t;
    float m   = t;
#pragma unroll
    for (int j = 1; j < 8; ++j) {
        t   = __fmul_rn(A[j], pattern[j]);
        dot = __fadd_rn(dot, t);
        m   = fminf(m, t);
    }
    const float d = __fmaf_rn(-2.0f, dot, norm);
    return odd ? __fmaf_rn(4.0f, m, d) : d;
}

// Byte masks (0xff where the coordinate is negative) of a code's eight signs, from its seven
// stored sign bits and the parity sign.
__device__ __forceinline__ uint2 kv_cache_vq2_sign_masks(unsigned stored7) {
    const unsigned sign = stored7 | ((__popc(stored7) & 1u) << 7);
    return make_uint2(kv_cache_vq2_byte_mask(sign & 0xfu), kv_cache_vq2_byte_mask(sign >> 4));
}

// The 128 sign-mask pairs, staged in shared memory for kv_cache_vq2_decode_word_lut.
__device__ __forceinline__ void kv_cache_vq2_stage_sign_masks(uint2* table, int thread,
                                                              int threads) {
    for (int i = thread; i < 128; i += threads) table[i] = kv_cache_vq2_sign_masks(i);
}

// kv_cache_vq2_decode_word with the sign masks read from the staged table.
__device__ __forceinline__ uint2 kv_cache_vq2_decode_word_lut(std::uint32_t code,
                                                              const std::int8_t* codebook,
                                                              const uint2* sign_masks) {
    const uint2 magnitude =
        *reinterpret_cast<const uint2*>(codebook + 8 * (code & kKVCacheVq2PatternMask));
    const uint2 mask = sign_masks[(code >> kKVCacheVq2SignShift) & 0x7fu];
    return make_uint2(__vsub4(magnitude.x ^ mask.x, mask.x), __vsub4(magnitude.y ^ mask.y, mask.y));
}

// Signed INT8 values and the 16-bit code of pattern p for a word with signs of y (bit j set =
// y_j < 0) and scaled magnitudes A.
__device__ __forceinline__ std::uint32_t kv_cache_vq2_word_code(const float (&A)[8],
                                                                unsigned negative, int p,
                                                                const float* pattern,
                                                                int (&c)[8]) {
    unsigned sign = negative;
    if (__popc(negative) & 1) {
        int flip = 0;
        float m  = __fmul_rn(A[0], pattern[0]);
#pragma unroll
        for (int j = 1; j < 8; ++j) {
            const float t = __fmul_rn(A[j], pattern[j]);
            if (t < m) {
                m    = t;
                flip = j;
            }
        }
        sign ^= 1u << flip;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int v = static_cast<int>(pattern[j]);
        c[j]        = ((sign >> j) & 1u) ? -v : v;
    }
    return static_cast<std::uint32_t>(p) | ((sign & 0x7fu) << kKVCacheVq2SignShift);
}

} // namespace infernix::ops
