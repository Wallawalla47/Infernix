#pragma once

// Exact fixed-point sums and their correct rounding, for the --lossless route
// (docs/maintainer/lossless-route.md §3).
//  - Wide<Words>: a sum of signed integers times powers of two (m 2^p, |m| < 2^31, p >= 0 relative to
//    the accumulator's unit), held as Words int64 chunks of 32 bits whose carries are deferred, so a
//    thread adds a term into two chunks and threads combine by adding chunks; finish() yields the
//    sign and the magnitude as 32-bit words.
//  - round_magnitude(): a magnitude times 2^unit, rounded to BF16 or binary32 (round to nearest, ties
//    to even, subnormals, overflow to infinity), with a sticky flag for nonzero bits below the
//    magnitude (a truncated quotient).
//  - divide_small(): the magnitude divided by a small integer, truncated, with the remainder's sticky.

#include <cstdint>

namespace infernix::ops::lossless {

template <int Words>
struct Wide {
    long long chunk[Words];

    __device__ __forceinline__ void clear() {
#pragma unroll
        for (int i = 0; i < Words; ++i) { chunk[i] = 0; }
    }

    // += m 2^position (0 <= position, position + 63 < 32 Words).
    __device__ __forceinline__ void add(long long m, int position) {
        const int word = position >> 5, shift = position & 31;
        const long long shifted = m * (1LL << shift); // |m| < 2^31: fits in 63 bits
        const long long low     = shifted & 0xFFFFFFFFLL;
        chunk[word] += low;
        chunk[word + 1] += (shifted - low) >> 32;
    }

    // Sums the warp's accumulators into every lane's.
    __device__ __forceinline__ void warp_sum() {
#pragma unroll
        for (int i = 0; i < Words; ++i) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                chunk[i] += __shfl_xor_sync(0xFFFFFFFFU, chunk[i], offset);
            }
        }
    }

    // Propagates the carries: returns whether the value is negative and its magnitude in mag.
    __device__ __forceinline__ bool finish(std::uint32_t (&mag)[Words]) const {
        long long carry = 0;
#pragma unroll
        for (int i = 0; i < Words; ++i) {
            const long long v = chunk[i] + carry;
            mag[i]            = static_cast<std::uint32_t>(v & 0xFFFFFFFFLL);
            carry             = v >> 32;
        }
        const bool negative = carry < 0;
        if (negative) {
            unsigned long long c = 1;
#pragma unroll
            for (int i = 0; i < Words; ++i) {
                const unsigned long long v = static_cast<unsigned long long>(~mag[i]) + c;
                mag[i]                     = static_cast<std::uint32_t>(v);
                c                          = v >> 32;
            }
        }
        return negative;
    }
};

// A signed 128-bit integer (two's complement over lo and hi).
struct I128 {
    unsigned long long lo;
    long long hi;
};

// acc += term 2^shift (|term| < 2^63, shift <= 64); bits shifted below 2^0 set `dropped`.
__device__ __forceinline__ void add_shifted(I128& acc, long long term, int shift, bool& dropped) {
    if (term == 0) { return; }
    unsigned long long lo;
    long long hi;
    if (shift >= 64) {
        lo = 0;
        hi = term; // shift == 64
    } else if (shift > 0) {
        lo = static_cast<unsigned long long>(term) << shift;
        hi = term >> (64 - shift);
    } else if (shift == 0) {
        lo = static_cast<unsigned long long>(term);
        hi = term < 0 ? -1 : 0;
    } else {
        const int down = -shift;
        if (down >= 63) {
            dropped = true;
            return;
        }
        if ((static_cast<unsigned long long>(term) & ((1ULL << down) - 1)) != 0) { dropped = true; }
        const long long q = term >> down;
        lo                = static_cast<unsigned long long>(q);
        hi                = q < 0 ? -1 : 0;
    }
    const unsigned long long sum = acc.lo + lo;
    acc.hi += hi + (sum < acc.lo ? 1 : 0);
    acc.lo = sum;
}

__device__ __forceinline__ I128 add_i128(I128 a, I128 b) {
    const unsigned long long lo = a.lo + b.lo;
    return {lo, a.hi + b.hi + (lo < a.lo ? 1 : 0)};
}

// Sign and magnitude words of an I128.
__device__ __forceinline__ bool magnitude_i128(const I128& v, std::uint32_t (&mag)[4]) {
    const bool negative   = v.hi < 0;
    unsigned long long lo = v.lo, hi = static_cast<unsigned long long>(v.hi);
    if (negative) {
        lo = ~lo + 1ULL;
        hi = ~hi + (lo == 0 ? 1ULL : 0ULL);
    }
    mag[0] = static_cast<std::uint32_t>(lo);
    mag[1] = static_cast<std::uint32_t>(lo >> 32);
    mag[2] = static_cast<std::uint32_t>(hi);
    mag[3] = static_cast<std::uint32_t>(hi >> 32);
    return negative;
}

// The I128 times 2^unit as a binary64 (rounded; relative error below 2^-52).
__device__ __forceinline__ double double_i128(const I128& v, int unit) {
    std::uint32_t mag[4];
    const bool negative = magnitude_i128(v, mag);
    const double m = ldexp(static_cast<double>((static_cast<unsigned long long>(mag[3]) << 32) | mag[2]), 64) +
                     static_cast<double>((static_cast<unsigned long long>(mag[1]) << 32) | mag[0]);
    const double d = ldexp(m, unit);
    return negative ? -d : d;
}

// mag /= divisor (truncated); returns whether the remainder is nonzero.
template <int Words>
__device__ __forceinline__ bool divide_small(std::uint32_t (&mag)[Words], std::uint32_t divisor) {
    unsigned long long rem = 0;
#pragma unroll
    for (int i = Words - 1; i >= 0; --i) {
        const unsigned long long cur = (rem << 32) | mag[i];
        mag[i]                       = static_cast<std::uint32_t>(cur / divisor);
        rem                          = cur % divisor;
    }
    return rem != 0;
}

// sign, mag (Words little-endian 32-bit words) times 2^unit, plus a nonzero tail below mag's last bit
// when sticky, rounded to BF16 (prec 8) or binary32 (prec 24) bits.
template <int Words>
__device__ std::uint32_t round_magnitude(const std::uint32_t (&mag)[Words], int unit, bool negative, bool bf16,
                                         bool sticky_below = false) {
    const int prec = bf16 ? 8 : 24;
    int top        = -1;
#pragma unroll
    for (int i = Words - 1; i >= 0; --i) {
        if (top < 0 && mag[i] != 0) { top = 32 * i + 31 - __clz(mag[i]); }
    }
    const std::uint32_t sign = negative ? (bf16 ? 0x8000U : 0x80000000U) : 0U;
    if (top < 0) { return sign; } // (a nonzero tail alone stays below every subnormal for our units)
    const int exponent = top + unit;
    const int ulp      = max(exponent - prec + 1, -126 - prec + 1);
    const int shift    = ulp - unit; // bits of mag below the result's ulp
    std::uint64_t q    = 0;
    bool half = false, sticky = sticky_below;
    if (shift <= 0) {
        q = static_cast<std::uint64_t>(mag[0]) | (Words > 1 ? static_cast<std::uint64_t>(mag[1]) << 32 : 0ULL);
        q <<= -shift;
    } else {
        const int word = shift / 32, bit = shift % 32;
        std::uint64_t two = 0;
        if (word < Words) { two = mag[word]; }
        if (word + 1 < Words) { two |= static_cast<std::uint64_t>(mag[word + 1]) << 32; }
        q                   = (two >> bit) & ((1ULL << prec) - 1);
        const int round_bit = shift - 1;
        // A round bit above every word of mag (a result far below the smallest subnormal) is zero.
        half = round_bit / 32 < Words && ((mag[round_bit / 32] >> (round_bit % 32)) & 1U) != 0;
#pragma unroll
        for (int i = 0; i < Words; ++i) {
            const int lo = 32 * i;
            if (lo >= round_bit) { continue; }
            const int n            = min(32, round_bit - lo);
            const std::uint32_t mk = n == 32 ? 0xFFFFFFFFU : ((1U << n) - 1U);
            if ((mag[i] & mk) != 0) { sticky = true; }
        }
    }
    if (half && (sticky || (q & 1ULL) != 0)) { ++q; }
    int e = ulp;
    if (q >= (1ULL << prec)) {
        q >>= 1;
        ++e;
    }
    if (q == 0) { return sign; }
    if (q < (1ULL << (prec - 1))) { return sign | static_cast<std::uint32_t>(q); }
    const int biased = e + prec - 1 + 127;
    if (biased >= 255) { return sign | (bf16 ? 0x7F80U : 0x7F800000U); }
    return sign | (static_cast<std::uint32_t>(biased) << (prec - 1)) | static_cast<std::uint32_t>(q - (1ULL << (prec - 1)));
}

// Words of F32Sum's accumulator: every binary32 (2^-149 to 2^128) and the carries of a few thousand terms.
constexpr int kF32SumWords = 10;

// An exact sum of binary32 values, then rounded once (IEEE results for non-finite terms).
struct F32Sum {
    Wide<kF32SumWords> acc;
    bool nan = false, pos_inf = false, neg_inf = false;

    __device__ __forceinline__ void clear() {
        acc.clear();
        nan = pos_inf = neg_inf = false;
    }

    __device__ __forceinline__ void add(float v) {
        const std::uint32_t bits = __float_as_uint(v);
        const int field = static_cast<int>((bits >> 23) & 0xFF), mant = static_cast<int>(bits & 0x7FFFFF);
        if (field == 255) {
            if (mant != 0) {
                nan = true;
            } else if (bits >> 31) {
                neg_inf = true;
            } else {
                pos_inf = true;
            }
            return;
        }
        if ((bits & 0x7FFFFFFF) == 0) { return; }
        const long long M = field == 0 ? mant : (0x800000 | mant);
        const long long s = (bits >> 31) ? -M : M;
        // 24-bit significand in two pieces, each below 2^31 after the chunk shift.
        acc.add(s & 0xFFF, field == 0 ? 1 : field);
        acc.add(s >> 12, (field == 0 ? 1 : field) + 12);
    }

    __device__ __forceinline__ float result() const {
        if (nan || (pos_inf && neg_inf)) { return NAN; }
        if (pos_inf) { return INFINITY; }
        if (neg_inf) { return -INFINITY; }
        std::uint32_t mag[kF32SumWords];
        const bool negative = acc.finish(mag);
        return __uint_as_float(round_magnitude(mag, -150, negative, false));
    }
};

} // namespace infernix::ops::lossless
