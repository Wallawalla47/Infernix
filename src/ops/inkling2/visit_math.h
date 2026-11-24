#pragma once

// The scalar arithmetic of Inkling 2's visit chain (docs/maintainer/inkling2_mini-design.md §2.5; the pinned orders of
// .local NUMERICS "Reduction orders" become the reproducible route's bits): host and device give the same bits, so the
// tests' emulations and the kernels share one definition. Only IEEE binary32 operations with round-to-nearest-even and
// explicit fma are used (host translation units compile with -ffp-contract=off).

#include "infernix/ops/inkling2_attention.h"
#include "infernix/ops/inkling2_visit.h"
#include "ops/common/canonical_math.h"

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#    define INFERNIX_VISIT_HD __host__ __device__ __forceinline__
#else
#    define INFERNIX_VISIT_HD inline
#endif

namespace infernix::ops::inkling2 {

inline constexpr int kGroups    = kHidden / 8; // 288 groups of 8 contiguous elements: one a thread
inline constexpr int kWarpsH    = kGroups / 32; // 9
inline constexpr float kMixScale = static_cast<float>(1.0 / 48.0); // fl32(H^-0.5), the mixes' score scale

INFERNIX_VISIT_HD float sqrt_rn(float x) {
#if defined(__CUDA_ARCH__)
    return __fsqrt_rn(x);
#else
    return std::sqrt(x); // IEEE correctly rounded
#endif
}

// 1 / sqrt(sum / n + eps): the unweighted RMS factor of a vector whose squares sum to `sum`.
INFERNIX_VISIT_HD float rms_factor(float sum, float n, float eps = kRmsEps) {
    return canon::div_rn(1.0F, sqrt_rn(canon::add_rn(canon::div_rn(sum, n), eps)));
}

// The L2 clip's factor min(1, threshold / (||x|| + 1e-6)) from the sum of squares.
INFERNIX_VISIT_HD float clip_factor(float sum, float threshold) {
    const float f = canon::div_rn(threshold, canon::add_rn(sqrt_rn(sum), 1e-6F));
    return f < 1.0F ? f : 1.0F;
}

// The full layers' log scaling: 1 + 0.1 ln(max(1, (pos + 1) / 8192)) (the division by 8192 exact).
INFERNIX_VISIT_HD float tau(std::int32_t position, float alpha = kLogScalingAlpha, float floor = kLogScalingFloor) {
    const float ratio = canon::div_rn(static_cast<float>(position + 1), floor);
    return canon::add_rn(1.0F, canon::mul_rn(alpha, canon::log_c(ratio > 1.0F ? ratio : 1.0F)));
}

// A static short conv of one channel: x[0] the current value, x[1..3] the values 1..3 positions back (0 before the
// sequence or the lane's first position); w[lag] lag-major. fold over lags 3, 2, 1, 0 by fma, then (acc + x) / sqrt(5).
INFERNIX_VISIT_HD float conv_static(const float (&w)[4], const float (&x)[4]) {
    float acc = 0.0F;
    acc       = canon::fma_rn(w[3], x[3], acc);
    acc       = canon::fma_rn(w[2], x[2], acc);
    acc       = canon::fma_rn(w[1], x[1], acc);
    acc       = canon::fma_rn(w[0], x[0], acc);
    return canon::div_rn(canon::add_rn(acc, x[0]), 2.2360679774997896964F); // fl32(sqrt 5)
}

// The full layers' dynamic conv: per lag the static tap plus the column's tap of the channel's head, then the fold,
// then (acc + x) / 3.
INFERNIX_VISIT_HD float conv_dynamic(const float (&w)[4], const float (&tap)[4], const float (&x)[4]) {
    float acc = 0.0F;
#pragma unroll
    for (int lag = 3; lag >= 0; --lag) { acc = canon::fma_rn(canon::add_rn(w[lag], tap[lag]), x[lag], acc); }
    return canon::div_rn(canon::add_rn(acc, x[0]), 3.0F);
}

// A loop message gate's update of one channel: (1 - tanh(f)) x + tanh(g) m, each operation rounded, in that order.
INFERNIX_VISIT_HD float message_gate(float x, float m, float f, float g) {
    const float keep = canon::mul_rn(canon::add_rn(1.0F, -canon::tanh_c(f)), x);
    return canon::add_rn(keep, canon::mul_rn(canon::tanh_c(g), m));
}

// ------------------------------------------------------------------ host emulations of the pinned reduction orders

// fold of 8 contiguous products by fma from +0.0.
INFERNIX_VISIT_HD float fold8_dot(const float* a, const float* b) {
    float acc = 0.0F;
    for (int j = 0; j < 8; ++j) { acc = canon::fma_rn(a[j], b[j], acc); }
    return acc;
}

// The recursive-halving tree of 32 values (pairs i, i + 16, then i + 8, ...): the value every lane of a xor butterfly
// holds.
INFERNIX_VISIT_HD float bfly32(const float* v) {
    float t[32];
    for (int i = 0; i < 32; ++i) { t[i] = v[i]; }
    for (int width = 16; width >= 1; width /= 2) {
        for (int i = 0; i < width; ++i) { t[i] = canon::add_rn(t[i], t[i + width]); }
    }
    return t[0];
}

// The H-vector dot product in the pinned order: 288 groups of 8 by fma folds, 9 bfly32 trees, a fold over the 9.
inline float hdot_reference(const float* a, const float* b) {
    float groups[kGroups];
    for (int g = 0; g < kGroups; ++g) { groups[g] = fold8_dot(a + 8 * g, b + 8 * g); }
    float total = 0.0F;
    for (int w = 0; w < kWarpsH; ++w) { total = canon::add_rn(total, bfly32(groups + 32 * w)); }
    return total;
}

// The 128-wide head sum in the pinned order: 32 groups of 4 by fma folds, then bfly32.
inline float head_dot_reference(const float* a, const float* b) {
    float groups[32];
    for (int g = 0; g < 32; ++g) {
        float acc = 0.0F;
        for (int j = 0; j < 4; ++j) { acc = canon::fma_rn(a[4 * g + j], b[4 * g + j], acc); }
        groups[g] = acc;
    }
    return bfly32(groups);
}

// rms(x) = x * rms_factor(sum x^2) per element (H wide), into out (which may be x).
inline void rms_reference(const float* x, float* out) {
    const float r = rms_factor(hdot_reference(x, x), static_cast<float>(kHidden));
    for (int i = 0; i < kHidden; ++i) { out[i] = canon::mul_rn(x[i], r); }
}

// A mix over `count` sources (FP32 values of each, H wide) for one FP32 query: the sources normalized, scores
// hdot(q, rms(s)) * fl32(H^-0.5), softmax (max in source order, exp_c, a fold of the exponentials, one division per
// source), then per element the fold over sources of p_s * rms(s) by fma.
inline void mix_reference(const float* query, const float* const* sources, int count, float* out) {
    static thread_local float normalized[5][kHidden];
    float score[5], e[5];
    for (int s = 0; s < count; ++s) {
        rms_reference(sources[s], normalized[s]);
        score[s] = canon::mul_rn(hdot_reference(query, normalized[s]), kMixScale);
    }
    float m = score[0];
    for (int s = 1; s < count; ++s) { m = score[s] > m ? score[s] : m; }
    float den = 0.0F;
    for (int s = 0; s < count; ++s) {
        e[s] = canon::exp_c(canon::add_rn(score[s], -m));
        den  = canon::add_rn(den, e[s]);
    }
    for (int s = 0; s < count; ++s) { e[s] = canon::div_rn(e[s], den); }
    for (int i = 0; i < kHidden; ++i) {
        float acc = 0.0F;
        for (int s = 0; s < count; ++s) { acc = canon::fma_rn(e[s], normalized[s][i], acc); }
        out[i] = acc;
    }
}

// RMSNorm with a BF16 weight: w * rms(x) per element (FP32; the caller rounds where the model does).
inline void norm_reference(const float* x, const std::uint16_t* weight, float* out) {
    const float r = rms_factor(hdot_reference(x, x), static_cast<float>(kHidden));
    for (int i = 0; i < kHidden; ++i) { out[i] = canon::mul_rn(canon::bf16_to_f32(weight[i]), canon::mul_rn(x[i], r)); }
}

} // namespace infernix::ops::inkling2

#undef INFERNIX_VISIT_HD
