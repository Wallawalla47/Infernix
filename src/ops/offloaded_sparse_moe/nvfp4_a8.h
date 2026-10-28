#pragma once

// The 8-bit activation route of NVFP4 experts stored without activation scales (W4A16 banks: the Qwen3.5-122B,
// Qwen3.8-Flash-Next's uncensored and FP4-Local artifacts) under `--a8-activations` (LOSSY, opt-in, never on the
// reproducible or lossless route; docs/maintainer/qwen3_8-flash-next-design.md §16.2.3). It replaces the exact
// A16 limbs of §16.2.1 with ops/ggml/ggml_a8.h's activations, so the flag means one activation codec on every
// format: a BF16 column is quantized per 32-element slice s to int8 codes q with an FP32 scale d_s
// (ggml::a8::slice_scale, ggml::a8::code; a slice holding Inf or NaN has d = NaN).
//
// Row product of one output row r over K inputs (K / 16 NVFP4 blocks b; slice s holds blocks 2s and 2s + 1):
//
//   P_b  = sum_{j<16} c2(w[r, 16b + j]) * q[16b + j]       |P_b| <= 12 * 16 * 127 < 2^15, exact int32
//   term = fma(fl32(P_b * d_s), S^(ws[r, b]), acc)          c2, S^ as §16.2 (canon::e2m1_x2, canon::e4m3_scaled)
//
// Order: a row's K / 32 slices are dealt to P partials (slice s to partial s % P); each partial adds its slices
// in ascending s, block 2s's term then block 2s + 1's, from +0; then the partials pair as a butterfly,
// p_i = p_i + p_(i + o) for o = P / 2, P / 4, ..., 1 (i < o), leaving the row's sum in p_0. P = kGateUpPartials
// for gate and up rows (K = hidden: 2,560 on Qwen3.8-Flash-Next, 3,072 on the 122B), kDownPartials for down rows
// (K = intermediate: 640 or 1,024). The row's output is
//
//   y = bf16_rn( fl32(p_0 * 2^-10) * m )      2^-10: 2^-1 for the doubled code, 2^-9 for S^ (exact unless
//                                              subnormal, then rounded once); m the record's multiplier
//
// so a NaN slice gives the canonical NaN (f32_to_bf16_rn). SwiGLU between the projections is §16.2's
// (canon::swiglu_bf16 on the BF16 gate and up outputs), and h (BF16) is quantized for the down rows the same way:
//
//   x (BF16) -> A8 -> gate/up -> BF16 -> SwiGLU -> BF16 h -> A8 -> down -> BF16 y_e -> combine
//
// Every route (the GPU 8-bit expert kernels for calls of at most 128 columns, at any column count of an expert,
// and the CPU route for the CPU-served misses of at most 8 columns) forms each term and the order as above, so a
// CPU-served miss returns the GPU's bits and an expert's output depends on neither its placement nor its column
// count in the call.

#include "ops/common/canonical_math.h"
#include "ops/common/exact_combine.h"
#include "ops/ggml/ggml_a8.h"

#include <cstdint>

#if defined(__CUDACC__)
#    define INFERNIX_NVFP4_A8_HD __host__ __device__ __forceinline__
#else
#    define INFERNIX_NVFP4_A8_HD inline
#endif

namespace infernix::ops::offloaded_moe::nvfp4_a8 {

namespace canon = ::infernix::ops::canon;

// The partials of gate/up rows and of down rows (a row's slices dealt as above).
inline constexpr int kGateUpPartials = 16;
inline constexpr int kDownPartials   = 4;
// The widest call the route takes (decode, verification, up to eight lanes); wider calls keep the A16 routes.
inline constexpr int kMaxCallColumns = 128;

// One block's term added into a partial: P_b the exact int32 product sum, d the slice's FP32 scale, scale_word the
// block's E4M3FN scale word (S^ = canon::e4m3_scaled, below 2^18, exact in binary32).
INFERNIX_NVFP4_A8_HD float add_block_term(float acc, std::int32_t p_b, float d, unsigned scale_word) {
    return canon::fma_rn(canon::mul_rn(static_cast<float>(p_b), d), static_cast<float>(canon::e4m3_scaled(scale_word)),
                         acc);
}

// The row's BF16 output from the butterfly's p_0 and the record's multiplier m.
INFERNIX_NVFP4_A8_HD std::uint16_t row_output(float p0, float multiplier) {
    return canon::f32_to_bf16_rn(canon::mul_rn(exact::scale_pow2(p0, -10), multiplier));
}

} // namespace infernix::ops::offloaded_moe::nvfp4_a8

#undef INFERNIX_NVFP4_A8_HD
