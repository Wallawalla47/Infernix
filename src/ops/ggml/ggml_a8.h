#pragma once

// The 8-bit activation route of GGUF (ggml) projections and experts (`--a8-activations`: LOSSY, opt-in;
// docs/maintainer/qwen3_8-flash-next-design.md §19.3.31): llama.cpp's Q8_1 activations with an FP32 scale.
//
// A BF16 column is quantized per 32-element slice. With m the slice's largest magnitude,
//
//   d = fl32(m / 127),  q_j = rne(fl32(v_j * fl32(127 / m))) in [-127, 127]
//
// (a slice whose m is below 2^-120, zero included, has d = 0 and q = 0; a slice holding Inf or NaN has
// d = NaN, so its column's outputs are NaN). A weight slice's codes times their integer sub-scales give the
// exact dot isum = sum_j sub_j * code_j * q_j (|isum| < 2^24, so fl32(isum) is exact), and the slice adds
//
//   acc = fma(fl32(isum * d), w, acc)            w: the weight block's binary16 scale, exactly in binary32
//   acc = fma(fl32(fl32(min * qsum) * d), -wmin, acc)   (Q4_K and Q5_K: min sub-scale, qsum the slice's code sum)
//
// and the row's output is fl32(acc * kappa) (an exact power of two) rounded to the output type once. Every
// route forms the terms the same way; the order of the FP32 sum is the route's (row_order for the expert
// kernels, which the CPU expert route follows, so a CPU-served miss returns the GPU's bits).

#include "core/weight.h"
#include "ops/common/canonical_math.h"
#include "ops/common/exact_combine.h"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#    define INFERNIX_A8_HD __host__ __device__ __forceinline__
#else
#    define INFERNIX_A8_HD inline
#endif

namespace infernix::ggml::a8 {

namespace canon = ::infernix::ops::canon;

// One encoded 32-element slice: eight int32 words of signed byte codes (element 4w + j in byte j of word w).
inline constexpr int kSliceWords = 8;

// An encoded column of K elements (K a multiple of 32): K / 32 slices of kSliceWords words, then K / 32
// binary32 scales d, then (formats with a min term) K / 32 int32 code sums.
INFERNIX_A8_HD constexpr std::size_t column_words(int k, bool sums) {
    return static_cast<std::size_t>(k) / 32 * (kSliceWords + 1 + (sums ? 1 : 0));
}
INFERNIX_A8_HD constexpr std::size_t scales_offset(int k) { return static_cast<std::size_t>(k) / 32 * kSliceWords; }
INFERNIX_A8_HD constexpr std::size_t sums_offset(int k) { return static_cast<std::size_t>(k) / 32 * (kSliceWords + 1); }

// The smallest slice maximum that is quantized (BF16 bits of 2^-120): below it fl32(127 / m) could overflow.
inline constexpr unsigned kMinKey = 7U << 7;

struct SliceScale {
    float d   = 0.0F; // dequantization scale
    float inv = 0.0F; // fl32(127 / m); 0 makes every code 0
};

// The scale of a slice whose largest magnitude has BF16 bits `key` (v & 0x7FFF maximized over the slice).
INFERNIX_A8_HD SliceScale slice_scale(unsigned key) {
    SliceScale s;
    if (key >= 0x7F80U) {
        s.d = canon::f32_from_bits(canon::kCanonicalNan);
        return s;
    }
    if (key < kMinKey) { return s; }
    const float m = canon::bf16_to_f32(static_cast<std::uint16_t>(key));
    s.d           = canon::div_rn(m, 127.0F);
    s.inv         = canon::div_rn(127.0F, m);
    return s;
}

// The code of one finite BF16 element under its slice's scale (|v * inv| <= 127 (1 + 2^-23) < 2^22).
INFERNIX_A8_HD int code(std::uint16_t v, float inv) {
    if (inv == 0.0F) { return 0; }
    const float r = canon::rint_rn(canon::mul_rn(canon::bf16_to_f32(v), inv));
    const int q   = static_cast<int>(r);
    return q > 127 ? 127 : (q < -127 ? -127 : q);
}

// A binary16 weight scale exactly in binary32.
INFERNIX_A8_HD float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16;
    const int field          = (h >> 10) & 0x1F;
    const std::uint32_t mant = h & 0x3FFU;
    if (field == 0x1F) { return canon::f32_from_bits(sign | 0x7F800000U | (mant << 13)); }
    if (field != 0) { return canon::f32_from_bits(sign | (static_cast<std::uint32_t>(field - 15 + 127) << 23) | (mant << 13)); }
    // Subnormal: mant * 2^-24, exact.
    const float v = canon::mul_rn(static_cast<float>(mant), canon::f32_from_bits(103U << 23));
    return sign != 0 ? -v : v;
}

// One slice's term added into acc (see above).
INFERNIX_A8_HD float add_term(float acc, int isum, float xd, float w) {
    return canon::fma_rn(canon::mul_rn(static_cast<float>(isum), xd), w, acc);
}

// Q4_K/Q5_K: the min term of one slice subtracted from acc (wmin: the block's dmin in binary32).
INFERNIX_A8_HD float sub_min_term(float acc, int min_sub, int qsum, float xd, float wmin) {
    return canon::fma_rn(canon::mul_rn(static_cast<float>(min_sub * qsum), xd), -wmin, acc);
}

// The row's output in binary32: acc times the format's kappa (log2 kappa_log2), exact.
INFERNIX_A8_HD float output_f32(float acc, int kappa_log2) { return exact::scale_pow2(acc, kappa_log2); }

// The expert rows' order (the GPU narrow expert kernels and the CPU expert route): a row's K / 32 slices are
// dealt to P partials, slice s to partial s % P, each partial adds its slices in ascending order from 0, then
// the partials pair as a butterfly, p_i = p_i + p_(i + o) for o = P / 2, P / 4, ..., 1 (i < o), leaving the
// row's sum in p_0. P = kGateUpPartials for gate and up rows (K = 2560), kDownPartials for down rows (K = 640).
inline constexpr int kGateUpPartials = 16;
inline constexpr int kDownPartials   = 4;

} // namespace infernix::ggml::a8

#undef INFERNIX_A8_HD
