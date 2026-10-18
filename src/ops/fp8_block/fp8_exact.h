#pragma once

// Exact W8A16 arithmetic over block-scaled FP8 (fp8_e4m3fn_b128: Qwen's official FP8 experts;
// docs/maintainer/qwen3_8-flash-next-design.md §16.2.2). Shared by the GPU kernels and the CPU
// engine: every function returns identical bits on both.
//
// A weight is e4m3fn(c) * s with one BF16 scale s = mant(s) * 2^(E(s) - 134) per 128 x 128 tile
// (mant in [128, 255], biased exponent E in [1, 254]; the converter refuses other scales). Every
// E4M3FN value is w * 2^-9 with an integer |w| <= 14 * 2^14 < 2^18 (fp8_int). An activation column
// is the A16 encoding of design §16.2.1 (ops::canon::a16_value): integers X, |X| < 2^21, with
// v = X * 2^(emax - 147). For one row, with E_min the smallest scale exponent of its tiles:
//
//   P_t = sum over tile t's 128 elements of w * X                 exact int64 (< 2^46)
//   R   = sum_t P_t * mant(s_t) * 2^(E(s_t) - E_min)               exact int128 (spread <= 62)
//   y   = bf16_rn( fl32_rn( R * 2^(E_min + emax - 290) ) )         one rounding to binary32
//
// so the sums are exact in any order and one output's bits depend only on its row and column.

#include "ops/common/canonical_math.h"
#include "ops/common/exact_combine.h"

#include <cstdint>

namespace infernix::fp8b128 {

using exact::I128;

constexpr int kTile              = 128;
// A tile term's largest shift (exact::add_shifted's domain). The converter refuses a row whose tile
// scales span more (tools/artifact/codecs/fp8_b128.py MAX_EXPONENT_SPREAD), so add_tile need not check.
constexpr int kMaxExponentSpread = 62;

// The integer e4m3fn(c) * 2^9 of a finite E4M3FN code (the NaN codes 0x7F/0xFF never occur).
INFERNIX_EXACT_HD std::int32_t fp8_int(std::uint8_t c) {
    const int e = (c >> 3) & 15, m = c & 7;
    const std::int32_t magnitude = e == 0 ? m : (8 + m) << (e - 1);
    return (c & 0x80) != 0 ? -magnitude : magnitude;
}

// A BF16 scale's integer mantissa (with its implicit bit) and biased exponent.
INFERNIX_EXACT_HD std::int32_t scale_mant(std::uint16_t word) { return (word & 0x7F) | 0x80; }
INFERNIX_EXACT_HD int scale_exponent(std::uint16_t word) { return (word >> 7) & 0xFF; }

// The smallest biased exponent among `tiles` scale words.
INFERNIX_EXACT_HD int min_exponent(const std::uint16_t* scales, int tiles) {
    int e = 255;
    for (int t = 0; t < tiles; ++t) { e = scale_exponent(scales[t]) < e ? scale_exponent(scales[t]) : e; }
    return e;
}

// Adds one tile's exact product sum P (of its codes and the column's X) into R.
INFERNIX_EXACT_HD void add_tile(I128& r, std::int64_t p, std::uint16_t scale, int e_min) {
    exact::add_shifted(r, p * scale_mant(scale), scale_exponent(scale) - e_min);
}

// The binary32 and BF16 output of R for a row with smallest tile exponent e_min and a column with
// A16 exponent emax (canonical NaN for a non-finite column).
INFERNIX_EXACT_HD float output_f32(const I128& r, int e_min, int emax) {
    if (emax == ops::canon::kA16NonFinite) { return ops::canon::f32_from_bits(ops::canon::kCanonicalNan); }
    return exact::to_f32_scaled_rn(r, e_min + emax - 290);
}

INFERNIX_EXACT_HD std::uint16_t output_bf16(const I128& r, int e_min, int emax) {
    if (emax == ops::canon::kA16NonFinite) { return 0x7FC0U; }
    return ops::canon::f32_to_bf16_rn(output_f32(r, e_min, emax));
}

// The exact R of one row (k code bytes; its k / 128 tile scales) against one encoded column (the
// reference form; kernels split the tiles across threads and add their partial R).
INFERNIX_EXACT_HD I128 row_dot(const std::uint8_t* row, const std::uint16_t* scales, const std::int32_t* x, int k,
                               int e_min) {
    I128 r;
    for (int t = 0; t < k / kTile; ++t) {
        std::int64_t p = 0;
        for (int j = t * kTile; j < (t + 1) * kTile; ++j) {
            p += static_cast<std::int64_t>(fp8_int(row[j])) * x[j];
        }
        add_tile(r, p, scales[t], e_min);
    }
    return r;
}

} // namespace infernix::fp8b128
