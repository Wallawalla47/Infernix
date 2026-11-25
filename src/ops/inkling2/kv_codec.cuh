#pragma once

// The quantized forms of Inkling 2's full-visit pools (include/infernix/ops/inkling2_attention.h KvCodec): a warp's
// encoder of one row of 128 values (lane l holding values 4 l .. 4 l + 3) and a thread's decoder of a 16-value chunk
// into the tile's represented words (BF16 K, FP16 V).

#include "infernix/ops/inkling2_attention.h"
#include "ops/common/canonical_math.h"
#include "ops/inkling2/kv_bf16p_piece.cuh"
#include "ops/inkling2/kv_int8_piece.cuh"
#include "ops/kv_cache/bf16p_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include "ops/kv_cache/q4_lloyd_codec.cuh"
#include "ops/kv_cache/vq2_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

namespace infernix::ops::inkling2::detail {

// A plane's form.
enum class KvForm : int { Words, Int8, Fp8, Nvfp4, Bf16p, Vq2, Q4 };

__host__ __device__ constexpr KvForm key_form(KvCodec c) {
    return c == KvCodec::Int8G64 ? KvForm::Int8
           : c == KvCodec::Fp8Row || c == KvCodec::Fp8KeyNvfp4V ? KvForm::Fp8
           : c == KvCodec::Nvfp4G16 ? KvForm::Nvfp4
           : c == KvCodec::Bf16Packed ? KvForm::Bf16p
           : c == KvCodec::Vq2 ? KvForm::Vq2
           : c == KvCodec::Q4KeyVq2V ? KvForm::Q4
                                     : KvForm::Words;
}
__host__ __device__ constexpr KvForm value_form(KvCodec c) {
    return c == KvCodec::Int8G64 ? KvForm::Int8
           : c == KvCodec::Fp8Row ? KvForm::Fp8
           : c == KvCodec::Nvfp4G16 || c == KvCodec::Fp8KeyNvfp4V ? KvForm::Nvfp4
           : c == KvCodec::Bf16Packed ? KvForm::Bf16p
           : c == KvCodec::Vq2 || c == KvCodec::Q4KeyVq2V ? KvForm::Vq2
                                                           : KvForm::Words;
}
__host__ __device__ constexpr bool rotated(KvForm f) { return f == KvForm::Vq2 || f == KvForm::Q4; }
__host__ __device__ constexpr bool rotated(KvCodec c) { return rotated(key_form(c)) || rotated(value_form(c)); }
__host__ __device__ constexpr int code_bytes(KvForm f) {
    return f == KvForm::Int8 || f == KvForm::Fp8 ? 128
           : f == KvForm::Nvfp4 || f == KvForm::Q4 ? 64
           : f == KvForm::Bf16p                   ? 192
           : f == KvForm::Vq2                     ? 32
                                                  : 0;
}
__host__ __device__ constexpr int side_bytes(KvForm f) {
    return f == KvForm::Int8                          ? 4
           : f == KvForm::Fp8 || rotated(f)            ? 2
           : f == KvForm::Nvfp4                       ? 8
           : f == KvForm::Bf16p                       ? 16
                                                      : 0;
}

// A represented word: K bf16_rn(x), V fp16_rn(x) of an exact product x.
__device__ __forceinline__ std::uint32_t kv_word(float x, bool value) {
    return value ? std::uint32_t{__half_as_ushort(__float2half_rn(x))} : std::uint32_t{canon::f32_to_bf16_rn(x)};
}

__device__ __forceinline__ float e4m3_value(std::uint8_t code) {
    __nv_fp8_e4m3 v;
    v.__x = code;
    return static_cast<float>(v);
}

// ---------------------------------------------------------------------------- encoders (a warp, one row)

// The row's codes and side record written at `codes` / `side` (the row's own), the lane's four represented words
// returned (BF16 K or FP16 V bits, as the BF16 storage's words: low half the first value).
__device__ __forceinline__ uint2 kv_encode_int8(const float (&x)[4], int lane, std::uint8_t* codes, std::uint8_t* side,
                                               bool value) {
    float m = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
#pragma unroll
    for (int offset = 8; offset >= 1; offset >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFU, m, offset)); }
    const __half h      = __float2half_rn(m > 0.0F ? canon::div_rn(m, 127.0F) : 0.0F);
    const float scale   = __half2float(h);
    const float inverse = scale > 0.0F ? canon::div_rn(1.0F, scale) : 0.0F;
    std::uint32_t packed = 0, w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = inverse == 0.0F ? 0 : max(-127, min(127, __float2int_rn(canon::mul_rn(x[j], inverse))));
        packed |= (static_cast<std::uint32_t>(q) & 0xFFU) << (8 * j);
        w[j] = kv_word(static_cast<float>(q) * scale, value);
    }
    *reinterpret_cast<std::uint32_t*>(codes + 4 * lane) = packed;
    if ((lane & 15) == 0) { reinterpret_cast<std::uint16_t*>(side)[lane >> 4] = __half_as_ushort(h); }
    return make_uint2(w[0] | (w[1] << 16), w[2] | (w[3] << 16));
}

__device__ __forceinline__ uint2 kv_encode_fp8(const float (&x)[4], int lane, std::uint8_t* codes, std::uint8_t* side,
                                              bool value) {
    float m = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFU, m, offset)); }
    __half h = __float2half_rn(0.0F);
    if (m != 0.0F) { h = __float2half_rn(fminf(65504.0F, fmaxf(0x1p-24F, canon::div_rn(m, 448.0F)))); }
    const float scale   = __half2float(h);
    const float inverse = m == 0.0F ? 0.0F : canon::div_rn(1.0F, scale);
    std::uint32_t packed = 0, w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const std::uint8_t q =
            inverse == 0.0F ? 0 : __nv_cvt_float_to_fp8(canon::mul_rn(x[j], inverse), __NV_SATFINITE, __NV_E4M3);
        packed |= std::uint32_t{q} << (8 * j);
        w[j] = kv_word(e4m3_value(q) * scale, value);
    }
    *reinterpret_cast<std::uint32_t*>(codes + 4 * lane) = packed;
    if (lane == 0) { *reinterpret_cast<std::uint16_t*>(side) = __half_as_ushort(h); }
    return make_uint2(w[0] | (w[1] << 16), w[2] | (w[3] << 16));
}

// nvfp4: group g (values 16 g ..) is lanes 4 g .. 4 g + 3; lane j of a group tries target j (lane 3 targets 3 and 4,
// in order), and every lane of the group then takes the targets in order exactly as the shared codec's sequential rule
// (target 0 always, a later one when its error is smaller).
__device__ __forceinline__ uint2 kv_encode_nvfp4(const float (&x)[4], int lane, std::uint8_t* codes, std::uint8_t* side,
                                                bool value) {
    constexpr unsigned kAll = 0xFFFFFFFFU;
    const int g = lane >> 2, j = lane & 3;
    float values[16];
    float max_abs = 0.0F;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = __shfl_sync(kAll, x[i & 3], 4 * g + (i >> 2));
        max_abs   = fmaxf(max_abs, fabsf(values[i]));
    }
    KVCacheNvfp4QuantizedGroup16 best{};
    float error = CUDART_INF_F;
    if (max_abs != 0.0F) {
        kv_cache_nvfp4_group16_targets(values, max_abs, j, j == 3 ? kKVCacheNvfp4ScaleTargetCount : j + 1, j == 0, best,
                                       error);
    }
    // The group's lanes in target order: lane 0's (target 0), then lanes 1, 2 and 3 replacing it when smaller.
    KVCacheNvfp4QuantizedGroup16 pick{};
    float pick_error = CUDART_INF_F;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int src            = 4 * g + k;
        const float e            = __shfl_sync(kAll, error, src);
        const std::uint32_t lo   = __shfl_sync(kAll, best.codes_lo, src);
        const std::uint32_t hi   = __shfl_sync(kAll, best.codes_hi, src);
        const auto s             = static_cast<std::uint8_t>(__shfl_sync(kAll, static_cast<unsigned>(best.scale), src));
        if (k == 0 || e < pick_error) {
            pick_error    = e;
            pick.codes_lo = lo;
            pick.codes_hi = hi;
            pick.scale    = s;
        }
    }
    // This lane's values 4 j .. 4 j + 3 of the group: bytes 2 j, 2 j + 1 of its 8.
    const std::uint32_t word = j < 2 ? pick.codes_lo : pick.codes_hi;
    const auto mine          = static_cast<std::uint16_t>(word >> (16 * (j & 1)));
    *reinterpret_cast<std::uint16_t*>(codes + 8 * g + 2 * j) = mine;
    if (j == 0) { side[g] = pick.scale; }
    const float scale = ops::detail::decode_nvfp4_e4m3(pick.scale);
    const float2 a    = ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(mine & 0xFFU));
    const float2 b    = ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(mine >> 8));
    return make_uint2(kv_word(a.x * scale, value) | (kv_word(a.y * scale, value) << 16),
                      kv_word(b.x * scale, value) | (kv_word(b.y * scale, value) << 16));
}

// ---------------------------------------------------------------------------- rotated forms (vq2, q4)

// H128 of the row held 4 a lane (value 4 l + j): H4 over j (spans 1, 2), then H32 over the lanes (hadamard_d256.cuh's
// strides 1..16), then times `scale` (a power of two: exact).
__device__ __forceinline__ void kv_hadamard128(float (&v)[4], int lane, float scale) {
    {
        const float a = v[0], b = v[1], c = v[2], d = v[3];
        v[0] = __fadd_rn(a, b), v[1] = __fsub_rn(a, b), v[2] = __fadd_rn(c, d), v[3] = __fsub_rn(c, d);
    }
    {
        const float a = v[0], b = v[1], c = v[2], d = v[3];
        v[0] = __fadd_rn(a, c), v[1] = __fadd_rn(b, d), v[2] = __fsub_rn(a, c), v[3] = __fsub_rn(b, d);
    }
    ops::hadamard_d32_columns_inplace(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) { v[j] = __fmul_rn(v[j], scale); }
}

// A lane's fma fold of a * b over its 4 values, then the warp's xor tree.
__device__ __forceinline__ float kv_row_dot(const float (&a)[4], const float (&b)[4]) {
    float s = 0.0F;
#pragma unroll
    for (int j = 0; j < 4; ++j) { s = __fmaf_rn(a[j], b[j], s); }
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) { s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFU, s, o)); }
    return s;
}

// The represented words of a rotated row's lane, in the rotated domain: K bf16_rn(S c), V fp16_rn(S c) (S c exact: |c|
// <= 127, S an FP16).
__device__ __forceinline__ uint2 kv_rotated_words(const float (&c)[4], float scale, bool value) {
    return make_uint2(kv_word(c[0] * scale, value) | (kv_word(c[1] * scale, value) << 16),
                      kv_word(c[2] * scale, value) | (kv_word(c[3] * scale, value) << 16));
}

// vq2 at 128: word w (coordinates 8 w ..) is lanes 2 w (its values 0-3) and 2 w + 1 (4-7); each lane of the pair scans
// half the 512 patterns for the first nearest (strict <), the lower half winning ties: the first nearest overall.
__device__ __forceinline__ uint2 kv_encode_vq2(const float (&x)[4], int lane, std::uint8_t* codes, std::uint8_t* side,
                                              bool value) {
    constexpr unsigned kAll = 0xFFFFFFFFU;
    float y[4] = {x[0], x[1], x[2], x[3]};
    kv_hadamard128(y, lane, 0x1p-3F);
    const float sum_sq   = kv_row_dot(y, y);
    const float sigma    = sqrtf(sum_sq * (1.0F / 128.0F));
    const float to_units = sigma > 0.0F ? __fdiv_rn(16.0F, sigma) : 0.0F;
    const int half       = lane & 1;
    float A[8];
    unsigned negative = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float mine  = __fmul_rn(fabsf(y[j]), to_units);
        const float other = __shfl_xor_sync(kAll, mine, 1);
        const float yo    = __shfl_xor_sync(kAll, y[j], 1);
        A[4 * half + j]       = mine;
        A[4 * (1 - half) + j] = other;
        negative |= (y[j] < 0.0F ? 1U : 0U) << (4 * half + j);
        negative |= (yo < 0.0F ? 1U : 0U) << (4 * (1 - half) + j);
    }
    const bool odd = (__popc(negative) & 1) != 0;
    const auto* book = ops::g_kv_cache_vq2_codebook_f32;
    float best       = CUDART_INF_F;
    int best_p       = half * (kKVCacheVq2Patterns / 2);
    for (int p = half * (kKVCacheVq2Patterns / 2); p < (half + 1) * (kKVCacheVq2Patterns / 2); ++p) {
        const float4 lo = __ldg(reinterpret_cast<const float4*>(book + 8 * p));
        const float4 hi = __ldg(reinterpret_cast<const float4*>(book + 8 * p) + 1);
        const float pattern[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        float norm = __fmul_rn(pattern[0], pattern[0]);
#pragma unroll
        for (int j = 1; j < 8; ++j) { norm = __fmaf_rn(pattern[j], pattern[j], norm); }
        const float d = kv_cache_vq2_pattern_distance(A, odd, pattern, norm);
        if (d < best) {
            best   = d;
            best_p = p;
        }
    }
    const float other_best = __shfl_xor_sync(kAll, best, 1);
    const int other_p      = __shfl_xor_sync(kAll, best_p, 1);
    // The pair's first nearest: the lower half's on ties (it holds the lower indices).
    const bool take_other = half == 0 ? other_best < best : !(best < other_best);
    const int p           = take_other ? other_p : best_p;
    float c[4]            = {0.0F, 0.0F, 0.0F, 0.0F};
    std::uint32_t code    = 0;
    if (sigma > 0.0F) {
        const float4 lo = __ldg(reinterpret_cast<const float4*>(book + 8 * p));
        const float4 hi = __ldg(reinterpret_cast<const float4*>(book + 8 * p) + 1);
        const float pattern[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        int levels[8];
        code = kv_cache_vq2_word_code(A, negative, p, pattern, levels);
#pragma unroll
        for (int j = 0; j < 4; ++j) { c[j] = static_cast<float>(levels[4 * half + j]); }
    }
    // A zero row: code 0 (pattern 0, every sign positive) and S = 0, so its words are +0 as the decoder's.
    const float dot = kv_row_dot(y, c);
    const __half s  = __float2half_rn(sigma > 0.0F && dot > 0.0F ? __fdiv_rn(sum_sq, dot) : 0.0F);
    if (half == 0) { reinterpret_cast<std::uint16_t*>(codes)[lane >> 1] = static_cast<std::uint16_t>(code); }
    if (lane == 0) { *reinterpret_cast<std::uint16_t*>(side) = __half_as_ushort(s); }
    return kv_rotated_words(c, __half2float(s), value);
}

// q4 at 128: each lane its 4 coordinates' levels (q4_lloyd_codec.cuh's thresholds at sigma over 128 values).
__device__ __forceinline__ uint2 kv_encode_q4(const float (&x)[4], int lane, std::uint8_t* codes, std::uint8_t* side,
                                             bool value) {
    float y[4] = {x[0], x[1], x[2], x[3]};
    kv_hadamard128(y, lane, 0x1p-3F);
    const float sum_sq   = kv_row_dot(y, y);
    const float sigma    = sqrtf(sum_sq * (1.0F / 128.0F));
    const float to_units = sigma > 0.0F ? __fdiv_rn(32.0F, sigma) : 0.0F;
    std::uint32_t nibbles = 0;
    float c[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int m    = kv_cache_q4_magnitude_index(__fmul_rn(fabsf(y[j]), to_units));
        const bool neg = y[j] < 0.0F;
        nibbles |= static_cast<std::uint32_t>(neg ? 7 - m : 8 + m) << (4 * j);
        const float level = static_cast<float>(kv_cache_q4_level(m));
        c[j]              = neg ? -level : level;
    }
    if (sigma == 0.0F) {
        nibbles = 0x7777U;
#pragma unroll
        for (int j = 0; j < 4; ++j) { c[j] = -static_cast<float>(kv_cache_q4_level(0)); }
    }
    const float dot = kv_row_dot(y, c);
    const __half s  = __float2half_rn(sigma > 0.0F && dot > 0.0F ? __fdiv_rn(sum_sq, dot) : 0.0F);
    reinterpret_cast<std::uint16_t*>(codes)[lane] = static_cast<std::uint16_t>(nibbles);
    if (lane == 0) { *reinterpret_cast<std::uint16_t*>(side) = __half_as_ushort(s); }
    return kv_rotated_words(c, __half2float(s), value);
}

// A rotated full visit's query, the lane's 4 dims of a head (BF16 values): q' = bf16_rn(2^-4 H128 q), so q' . y =
// q . k for y = 2^-3 H128 k (H128 H128 = 128 I). Its output o' (rotated, FP32) is o = 2^-4 H128 o'.
__device__ __forceinline__ void kv_rotate_query(float (&q)[4], int lane) {
    kv_hadamard128(q, lane, 0x1p-4F);
#pragma unroll
    for (int j = 0; j < 4; ++j) { q[j] = canon::bf16_to_f32(canon::f32_to_bf16_rn(q[j])); }
}

// `x` the lane's four values the BF16 storage holds (BF16 K, FP16 V), `bits` their BF16 source words (bf16p's input).
// Sets *unpackable (bf16p: the row has no packed form). Returns the lane's four represented words. Rotated false: an
// instance that never sees vq2 or q4 (it carries no code for them); Packed false: one that never encodes bf16p (the
// bf16p instances' own).
template <bool Rotated = true, bool Packed = true>
__device__ __forceinline__ uint2 kv_encode(KvForm form, const float (&x)[4], const std::uint16_t (&bits)[4], int lane,
                                          std::uint8_t* codes, std::uint8_t* side, bool value, bool& unpackable) {
    unpackable = false;
    switch (form) {
    case KvForm::Int8: return kv_encode_int8(x, lane, codes, side, value);
    case KvForm::Fp8: return kv_encode_fp8(x, lane, codes, side, value);
    case KvForm::Nvfp4: return kv_encode_nvfp4(x, lane, codes, side, value);
    case KvForm::Vq2:
        if constexpr (Rotated) { return kv_encode_vq2(x, lane, codes, side, value); } else { return make_uint2(0U, 0U); }
    case KvForm::Q4:
        if constexpr (Rotated) { return kv_encode_q4(x, lane, codes, side, value); } else { return make_uint2(0U, 0U); }
    default:
        if constexpr (Packed) {
            unpackable = !kv_encode_bf16p(bits, lane, codes, side);
            std::uint32_t w[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) { w[j] = value ? kv_word(canon::bf16_to_f32(bits[j]), true) : std::uint32_t{bits[j]}; }
            return make_uint2(w[0] | (w[1] << 16), w[2] | (w[3] << 16));
        } else {
            return make_uint2(0U, 0U);
        }
    }
}

// ---------------------------------------------------------------------------- decoder (a thread, a 16-value chunk)

// Chunk c (values 16 c .. 16 c + 15) of a row as 16 represented words: lo values 0-7, hi 8-15. Rotated false: as
// kv_encode's.
template <bool Rotated = true>
__device__ __forceinline__ void kv_decode16(KvForm form, const std::uint8_t* codes, const std::uint8_t* side, int c,
                                            bool value, uint4& lo, uint4& hi) {
    std::uint32_t out[8];
    if (form == KvForm::Bf16p) {
        const uint4 code   = *reinterpret_cast<const uint4*>(codes + 16 * c);
        const uint2 nib    = *reinterpret_cast<const uint2*>(codes + 128 + 8 * c);
        const uint4 record = *reinterpret_cast<const uint4*>(side);
        const unsigned base = record.x & 0xFFU;
        // The escapes before this chunk (a nibble 15 is all four bits set).
        unsigned index = 0;
        if (((record.x >> 8) & 0xFU) != 0) {
            for (int k = 0; k < c; ++k) {
                const uint2 n = *reinterpret_cast<const uint2*>(codes + 128 + 8 * k);
                index += __popc(n.x & (n.x >> 1) & (n.x >> 2) & (n.x >> 3) & 0x11111111U) +
                         __popc(n.y & (n.y >> 1) & (n.y >> 2) & (n.y >> 3) & 0x11111111U);
            }
        }
        // Escape k's exponent: record byte 2 + k, its word chosen by two selects (an array indexed at run time lands
        // in local memory).
        const auto escape = [&](unsigned k) {
            const unsigned byte = 2 + k, q = byte >> 2;
            const std::uint32_t low = (q & 1U) != 0 ? record.y : record.x, high = (q & 1U) != 0 ? record.w : record.z;
            return (((q & 2U) != 0 ? high : low) >> (8 * (byte & 3U))) & 0xFFU;
        };
        const std::uint32_t cw[4] = {code.x, code.y, code.z, code.w};
        std::uint32_t half[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const unsigned byte = (cw[i / 4] >> (8 * (i % 4))) & 0xFFU;
            const unsigned n    = ((i < 8 ? nib.x : nib.y) >> (4 * (i % 8))) & 0xFU;
            unsigned exponent   = base + n;
            if (n == kBf16pZeroCode) { exponent = 0; }
            if (n == kBf16pEscapeCode) {
                exponent = index < kBf16p128Escapes ? escape(index) : 0U;
                ++index;
            }
            const auto bf = static_cast<std::uint16_t>(((byte & 0x80U) << 8) | ((exponent & 0xFFU) << 7) | (byte & 0x7FU));
            half[i]       = value ? kv_word(canon::bf16_to_f32(bf), true) : std::uint32_t{bf};
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) { out[i] = half[2 * i] | (half[2 * i + 1] << 16); }
    } else if (Rotated && rotated(form)) {
        // Words 2 c and 2 c + 1 (8 coordinates each) as signed INT8 levels, times S (rotated-domain words).
        uint2 levels[2];
        if (form == KvForm::Vq2) {
            const std::uint32_t pair = *reinterpret_cast<const std::uint32_t*>(codes + 4 * c);
            levels[0]                = kv_cache_vq2_decode_word(pair & 0xFFFFU, ops::g_kv_cache_vq2_codebook);
            levels[1]                = kv_cache_vq2_decode_word(pair >> 16, ops::g_kv_cache_vq2_codebook);
        } else {
            const uint2 nibbles = *reinterpret_cast<const uint2*>(codes + 8 * c);
            levels[0]           = kv_cache_q4_decode_word(nibbles.x);
            levels[1]           = kv_cache_q4_decode_word(nibbles.y);
        }
        const float scale = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(side)));
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const std::uint32_t word = (i < 4 ? (i < 2 ? levels[0].x : levels[0].y) : (i < 6 ? levels[1].x : levels[1].y));
            const int shift          = 16 * (i & 1);
            const float v0 = static_cast<float>(static_cast<std::int8_t>((word >> shift) & 0xFFU));
            const float v1 = static_cast<float>(static_cast<std::int8_t>((word >> (shift + 8)) & 0xFFU));
            out[i]         = kv_word(v0 * scale, value) | (kv_word(v1 * scale, value) << 16);
        }
    } else if (form == KvForm::Nvfp4) {
        const uint2 raw   = *reinterpret_cast<const uint2*>(codes + 8 * c);
        const float scale = ops::detail::decode_nvfp4_e4m3(side[c]);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float2 p = ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>((i < 4 ? raw.x : raw.y) >> (8 * (i & 3))));
            out[i]         = kv_word(p.x * scale, value) | (kv_word(p.y * scale, value) << 16);
        }
    } else {
        const uint4 raw         = *reinterpret_cast<const uint4*>(codes + 16 * c);
        const std::uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
        const bool int8         = form == KvForm::Int8;
        const float scale       = __half2float(
            __ushort_as_half(reinterpret_cast<const std::uint16_t*>(side)[int8 ? c >> 2 : 0]));
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const std::uint32_t pair = w[i / 2] >> (16 * (i % 2));
            const auto b0 = static_cast<std::uint8_t>(pair & 0xFFU), b1 = static_cast<std::uint8_t>((pair >> 8) & 0xFFU);
            const float v0 = int8 ? static_cast<float>(static_cast<std::int8_t>(b0)) : e4m3_value(b0);
            const float v1 = int8 ? static_cast<float>(static_cast<std::int8_t>(b1)) : e4m3_value(b1);
            out[i]         = kv_word(v0 * scale, value) | (kv_word(v1 * scale, value) << 16);
        }
    }
    lo = make_uint4(out[0], out[1], out[2], out[3]);
    hi = make_uint4(out[4], out[5], out[6], out[7]);
}

} // namespace infernix::ops::inkling2::detail
