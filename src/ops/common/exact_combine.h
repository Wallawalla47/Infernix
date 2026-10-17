#pragma once

// Exact integer combination and the one final rounding of the exact expert and dense routes
// (docs/maintainer/qwen3_8-flash-next-design.md §16.2.2): sums of
// integer products kept exactly in int128, then rounded once. Host and device give identical bits.

#include "ops/common/canonical_math.h"

#include <bit>
#include <cstdint>

#if defined(__CUDACC__)
#    define INFERNIX_EXACT_HD __host__ __device__ __forceinline__
#else
#    define INFERNIX_EXACT_HD inline
#endif

namespace infernix::exact {

// The position of the most significant set bit of a nonzero word (one instruction on either side).
INFERNIX_EXACT_HD int top_bit(std::uint64_t v) {
#if defined(__CUDA_ARCH__)
    return 63 - __clzll(static_cast<long long>(v));
#else
    return 63 - std::countl_zero(v);
#endif
}

struct I128 {
    std::uint64_t lo = 0;
    std::int64_t hi  = 0;
};

// acc += v * 2^shift, 0 <= shift <= 62.
INFERNIX_EXACT_HD void add_shifted(I128& acc, std::int64_t v, int shift) {
    const std::uint64_t lo = static_cast<std::uint64_t>(v) << shift;
    const std::int64_t hi  = shift == 0 ? (v < 0 ? -1 : 0) : (v >> (64 - shift));
    const std::uint64_t sum = acc.lo + lo;
    acc.hi += hi + (sum < lo ? 1 : 0);
    acc.lo = sum;
}

INFERNIX_EXACT_HD void add(I128& acc, const I128& b) {
    const std::uint64_t sum = acc.lo + b.lo;
    acc.hi += b.hi + (sum < b.lo ? 1 : 0);
    acc.lo = sum;
}

// Round-to-nearest-even binary32 of R * 2^k (|R| < 2^127, any k): one rounding, at the binary32
// precision of the result's own binade, subnormal results included; beyond the finite range +-inf.
INFERNIX_EXACT_HD float to_f32_scaled_rn(I128 r, int k) {
    const bool negative = r.hi < 0;
    std::uint64_t hi = static_cast<std::uint64_t>(r.hi), lo = r.lo;
    if (negative) { // two's complement negation
        lo = ~lo + 1U;
        hi = ~hi + (lo == 0 ? 1U : 0U);
    }
    const std::uint32_t sign = negative ? 0x80000000U : 0U;
    if (hi == 0 && lo == 0) { return ops::canon::f32_from_bits(sign); }
    const int p = hi != 0 ? 64 + top_bit(hi) : top_bit(lo); // position of the most significant bit of |R|
    // |R| * 2^k = m * 2^q with m an integer: keep 24 bits in a normal binade (q = p + k - 23), or
    // the bits at and above 2^-149 in the subnormal range; s = the low bits of |R| dropped.
    int q = p + k - 23;
    if (q < -149) { q = -149; }
    const int s = q - k;
    std::uint64_t kept;
    bool above = false, equal = false;
    if (s <= 0) {
        const int up = -s; // < 24: the value fits the mantissa exactly
        kept = lo << up;
    } else if (s > 127) {
        kept = 0; // below half the smallest subnormal: rounds to zero (s > p + 1 here)
    } else {
        std::uint64_t rest_hi, rest_lo;
        if (s >= 64) {
            kept    = s >= 128 ? 0 : hi >> (s - 64);
            rest_hi = s == 64 ? 0 : hi & ((1ULL << (s - 64)) - 1ULL);
            rest_lo = lo;
        } else {
            kept    = (lo >> s) | (hi << (64 - s));
            rest_hi = 0;
            rest_lo = lo & ((1ULL << s) - 1ULL);
        }
        if (s - 1 >= 64) {
            const std::uint64_t half = 1ULL << (s - 1 - 64);
            above = rest_hi > half || (rest_hi == half && rest_lo != 0);
            equal = rest_hi == half && rest_lo == 0;
        } else {
            const std::uint64_t half = 1ULL << (s - 1);
            above = rest_hi != 0 || rest_lo > half;
            equal = rest_hi == 0 && rest_lo == half;
        }
    }
    if (above || (equal && (kept & 1U))) { ++kept; }
    // kept * 2^q: renormalize a carry into the next binade (or out of the subnormal range).
    while (kept >= (1ULL << 24)) {
        kept >>= 1; // exact: a carry leaves a power of two
        ++q;
    }
    if (kept == 0) { return ops::canon::f32_from_bits(sign); }
    if (kept < (1ULL << 23)) { // subnormal (q == -149)
        return ops::canon::f32_from_bits(sign | static_cast<std::uint32_t>(kept));
    }
    const int exponent = q + 23; // of the leading bit
    if (exponent > 127) { return ops::canon::f32_from_bits(sign | 0x7F800000U); }
    return ops::canon::f32_from_bits(sign | (static_cast<std::uint32_t>(exponent + 127) << 23) |
                                     (static_cast<std::uint32_t>(kept) & 0x7FFFFFU));
}

// Round-to-nearest-even binary32 of R (|R| < 2^127; never subnormal): to_f32_scaled_rn(r, 0) in the
// short form the GGUF routes' epilogues inline.
INFERNIX_EXACT_HD float to_f32_rn(I128 r) {
    const bool negative = r.hi < 0;
    std::uint64_t hi = static_cast<std::uint64_t>(r.hi), lo = r.lo;
    if (negative) { // two's complement negation
        lo = ~lo + 1U;
        hi = ~hi + (lo == 0 ? 1U : 0U);
    }
    if (hi == 0 && lo == 0) { return negative ? -0.0F : 0.0F; }
    const int p = hi != 0 ? 64 + top_bit(hi) : top_bit(lo); // position of the most significant bit
    std::uint32_t mantissa; // 24 bits
    int exponent = p;       // value = mantissa * 2^(p - 23)
    if (p <= 23) {
        mantissa = static_cast<std::uint32_t>(lo) << (23 - p);
    } else {
        const int s = p - 23; // bits dropped
        std::uint64_t kept, rest_hi, rest_lo;
        if (s >= 64) {
            kept    = hi >> (s - 64);
            rest_hi = s - 64 == 0 ? 0 : hi & ((1ULL << (s - 64)) - 1ULL);
            rest_lo = lo;
        } else {
            kept    = (lo >> s) | (s == 0 ? 0 : hi << (64 - s));
            rest_hi = 0;
            rest_lo = lo & ((1ULL << s) - 1ULL);
        }
        bool above = false, equal = false;
        if (s - 1 >= 64) {
            const std::uint64_t half = 1ULL << (s - 1 - 64);
            above = rest_hi > half || (rest_hi == half && rest_lo != 0);
            equal = rest_hi == half && rest_lo == 0;
        } else {
            const std::uint64_t half = 1ULL << (s - 1);
            above = rest_hi != 0 || rest_lo > half;
            equal = rest_hi == 0 && rest_lo == half;
        }
        mantissa = static_cast<std::uint32_t>(kept);
        if (above || (equal && (mantissa & 1U))) {
            ++mantissa;
            if (mantissa == (1U << 24)) {
                mantissa >>= 1;
                ++exponent;
            }
        }
    }
    const std::uint32_t bits = (negative ? 0x80000000U : 0U) | (static_cast<std::uint32_t>(exponent + 127) << 23) |
                               (mantissa & 0x7FFFFFU);
    return ops::canon::f32_from_bits(bits);
}

// f * 2^k in two exact power-of-two steps (|k| <= 252; each half's factor is a normal binary32).
INFERNIX_EXACT_HD float scale_pow2(float f, int k) {
    const int k1 = k / 2;
    const int k2 = k - k1;
    const float s1 = ops::canon::f32_from_bits(static_cast<std::uint32_t>(k1 + 127) << 23);
    const float s2 = ops::canon::f32_from_bits(static_cast<std::uint32_t>(k2 + 127) << 23);
    return ops::canon::mul_rn(ops::canon::mul_rn(f, s1), s2);
}

} // namespace infernix::exact
