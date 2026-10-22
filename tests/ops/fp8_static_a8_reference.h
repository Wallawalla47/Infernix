#pragma once

// The static A8 activation rule of fp8_e4m3fn_row_f32 parents (Qwen3.5-122B-A10B; design D4):
// code = e4m3_rn_satfinite(fl32(x * fl32(1 / s))) with the sign of x, representing code * s. Tests of
// the static A8 routes hold the product to the A16 criterion against FP64 sums over these represented
// values, so what remains is FP32 accumulation and the output's rounding.

#include "ops/common/canonical_math.h"

#include "ops/input_projection_test_common.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace infernix::test::fp8_static {

// A static input scale of the checkpoint's kind (0.0025-0.189): its reciprocal is inexact.
inline constexpr float kScale = 0.0123456791F;

// The D4 code of one BF16 value under the reciprocal r = fl32(1 / s).
inline std::uint8_t code(float x, float r) {
    const float y = x * r; // one binary32 rounding (no contraction: a lone product)
    std::uint32_t bits;
    std::memcpy(&bits, &y, 4);
    const std::uint8_t magnitude = ops::canon::e4m3_rn_satfinite(y);
    return static_cast<std::uint8_t>(magnitude | ((bits >> 31) != 0 && magnitude != 0x7F ? 0x80U : 0U));
}

inline double e4m3_value(std::uint8_t c) {
    const int e = (c >> 3) & 15, m = c & 7;
    const double v = e == 0 ? m * std::ldexp(1.0, -9) : (8 + m) * std::ldexp(1.0, e - 10);
    return (c & 0x80) != 0 ? -v : v;
}

// The value one BF16 activation's code represents: code * s (exact in binary64).
inline double represented(float x, float scale = kScale) {
    return e4m3_value(code(x, 1.0F / scale)) * static_cast<double>(scale);
}

inline std::vector<double> represented(const std::vector<float>& x, float scale = kScale) {
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { out[i] = represented(x[i], scale); }
    return out;
}

// BF16 activations [k][tokens] with the cases the encoding must get right: most values small, a few
// outliers beyond 448 s (saturation), some in the E4M3 subnormal range, and negative zeros.
inline std::vector<float> activations(int k, int tokens, std::uint32_t seed, float scale = kScale) {
    std::vector<float> x(static_cast<std::size_t>(k) * tokens);
    std::mt19937 rng(seed);
    std::normal_distribution<float> n(0.0F, 0.02F);
    std::uniform_int_distribution<int> kind(0, 199);
    float negative_zero;
    const std::uint32_t sign = 0x80000000U;
    std::memcpy(&negative_zero, &sign, 4);
    for (auto& v : x) {
        const int pick = kind(rng);
        v = pick == 0 ? 600.0F * scale * (rng() & 1 ? 1.0F : -1.0F)          // saturates
            : pick == 1 ? std::ldexp(scale, -8) * static_cast<float>(rng() % 7) // E4M3 subnormal
            : pick == 2 ? negative_zero
                        : n(rng);
    }
    round_to_bf16(x);
    return x;
}

} // namespace infernix::test::fp8_static
