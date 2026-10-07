#pragma once

// D256 KV-cache NVFP4 codec. Unlike weight artifacts, cache rows have no matrix-level divisor:
// every represented value is exactly E2M1(code) * E4M3(scale) for its contiguous G16 group.

#include "ops/kernel/paged_kv_address.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <cstdint>

namespace infernix::ops {

inline constexpr int kKVCacheNvfp4HeadDim        = 256;
inline constexpr int kKVCacheNvfp4Group          = 16;
inline constexpr int kKVCacheNvfp4Groups         = 16;
inline constexpr int kKVCacheNvfp4CodeBytes      = 128;
inline constexpr float kKVCacheNvfp4MaxFinite    = 6.0F;
inline constexpr float kKVCacheNvfp4ScaleMinimum = 0x1p-9F;
inline constexpr float kKVCacheNvfp4ScaleMaximum = 448.0F;

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_nvfp4_code_index(int physical_page, int kv_head,
                                                                  int d, int page_offset) {
    return paged_kv_element_offset<kKVCacheNvfp4CodeBytes, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, d >> 1);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_nvfp4_scale_index(int physical_page, int kv_head,
                                                                   int group, int page_offset) {
    return paged_kv_element_offset<kKVCacheNvfp4Groups, Geometry::KVHeads>(physical_page, kv_head,
                                                                           page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_nvfp4_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kKVCacheNvfp4HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

struct KVCacheNvfp4QuantizedGroup16 {
    std::uint32_t codes_lo = 0;
    std::uint32_t codes_hi = 0;
    std::uint8_t scale     = 0;
};

// The group's largest magnitude maps to one of these E2M1 targets, tried in order; the scale whose
// codes reconstruct the group with the least squared error wins, and ties keep the earlier target.
// E2M1 steps widen toward 6, so a group whose large values fall between codes is often represented
// better by scaling its maximum to 4-5.5 (Four Over Six, arXiv:2512.02010, generalized to five
// targets). No target saturates the maximum: targets 6.5 and 7 lowered the squared error further
// but made 4K-window perplexity worse. On rotated K rows of the 27B model this set lowers the
// reconstruction error from 9.5 % to 8.5 % RMS. Decoding is unchanged.
//
// Every step is a single IEEE-rounded FP32 operation in a fixed order (no contraction), so a host
// encoder reproduces the stored bytes exactly: scale = E4M3(clamp(max * RN(1 / target))), quotient
// = value * RN(1 / scale), codes = RNE-satfinite E2M1, error = sum over the group in index order of
// RN(RN(value - code * scale)^2).
inline constexpr int kKVCacheNvfp4ScaleTargetCount = 5;

__device__ __forceinline__ float kv_cache_nvfp4_scale_target_inverse(int index) {
    switch (index) {
    case 0:
        return 1.0F / 6.0F;
    case 1:
        return 1.0F / 4.0F;
    case 2:
        return 1.0F / 4.5F;
    case 3:
        return 1.0F / 5.0F;
    default:
        return 1.0F / 5.5F;
    }
}

// The best of targets [first, last) for a group of 16 values with a nonzero maximum. The first
// target is taken unconditionally when `take_first` (so a non-finite group keeps the max -> 6
// encoding); otherwise a target must have a smaller error than `error` on entry.
__device__ __forceinline__ void
kv_cache_nvfp4_group16_targets(const float (&values)[16], float max_abs, int first, int last,
                               bool take_first, KVCacheNvfp4QuantizedGroup16& best, float& error) {
#pragma unroll 1
    for (int target = first; target < last; ++target) {
        const float raw_scale = __fmul_rn(max_abs, kv_cache_nvfp4_scale_target_inverse(target));
        const float bounded =
            fminf(kKVCacheNvfp4ScaleMaximum, fmaxf(kKVCacheNvfp4ScaleMinimum, raw_scale));
        const std::uint8_t scale_code = __nv_cvt_float_to_fp8(bounded, __NV_SATFINITE, __NV_E4M3);
        const float scale             = detail::decode_nvfp4_e4m3(scale_code);
        const float inverse           = __frcp_rn(scale);
        float2 quotients[8];
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            quotients[pair] = make_float2(__fmul_rn(values[2 * pair], inverse),
                                          __fmul_rn(values[2 * pair + 1], inverse));
        }
        std::uint32_t codes_lo = 0;
        std::uint32_t codes_hi = 0;
        detail::pack_nvfp4_e2m1x16(quotients, codes_lo, codes_hi);
        float candidate = 0.0F;
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const std::uint32_t word = pair < 4 ? codes_lo : codes_hi;
            const float2 code =
                detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(word >> (8 * (pair & 3))));
            const float low  = __fsub_rn(values[2 * pair], __fmul_rn(code.x, scale));
            const float high = __fsub_rn(values[2 * pair + 1], __fmul_rn(code.y, scale));
            candidate        = __fadd_rn(candidate, __fmul_rn(low, low));
            candidate        = __fadd_rn(candidate, __fmul_rn(high, high));
        }
        if ((take_first && target == first) || candidate < error) {
            error         = candidate;
            best.codes_lo = codes_lo;
            best.codes_hi = codes_hi;
            best.scale    = scale_code;
        }
    }
}

// Encodes the 16 groups of a D256 row held in `row` (FP32, shared or global) with a whole warp:
// lanes 0-15 try the first three targets of group `lane`, lanes 16-31 the remaining ones of group
// `lane - 16`, and the lower half keeps the better of the two (ties to the lower half, whose
// targets come first), which is exactly the sequential rule above. All 32 lanes must call it; lanes
// 0-15 hold the results.
__device__ __forceinline__ KVCacheNvfp4QuantizedGroup16
kv_cache_nvfp4_quantize_group16_warp(const float* row, int lane) {
    constexpr unsigned FullMask = 0xffffffffU;
    const int group             = lane & (kKVCacheNvfp4Groups - 1);
    const bool lower            = lane < kKVCacheNvfp4Groups;
    float values[16];
    float max_abs = 0.0F;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = row[group * kKVCacheNvfp4Group + i];
        max_abs   = fmaxf(max_abs, fabsf(values[i]));
    }
    KVCacheNvfp4QuantizedGroup16 best{};
    float error = CUDART_INF_F;
    if (max_abs != 0.0F) {
        constexpr int Split = (kKVCacheNvfp4ScaleTargetCount + 1) / 2;
        kv_cache_nvfp4_group16_targets(values, max_abs, lower ? 0 : Split,
                                       lower ? Split : kKVCacheNvfp4ScaleTargetCount, lower, best,
                                       error);
    }
    const float upper_error = __shfl_xor_sync(FullMask, error, 16);
    const auto upper_lo     = __shfl_xor_sync(FullMask, best.codes_lo, 16);
    const auto upper_hi     = __shfl_xor_sync(FullMask, best.codes_hi, 16);
    const auto upper_scale =
        static_cast<std::uint8_t>(__shfl_xor_sync(FullMask, static_cast<unsigned>(best.scale), 16));
    if (lower && max_abs != 0.0F && upper_error < error) {
        best.codes_lo = upper_lo;
        best.codes_hi = upper_hi;
        best.scale    = upper_scale;
    }
    return best;
}

__device__ __forceinline__ int4 kv_cache_nvfp4_dequant_f16x8(const std::uint8_t* codes,
                                                             std::uint8_t scale_code) {
    // Every finite E2M1 value times a legal nonnegative E4M3 cache scale is exactly representable
    // in FP16 (at most four product fraction bits and magnitude <= 2688). Half2 multiplication is
    // therefore the exact FP16 expansion boundary, not an additional approximation.
    const std::uint32_t packed = load_vec<std::uint32_t>(codes);
    const std::uint8_t* bytes  = reinterpret_cast<const std::uint8_t*>(&packed);
    __nv_fp8_e4m3 encoded_scale;
    encoded_scale.__x    = scale_code;
    const __half scale   = static_cast<__half>(encoded_scale);
    const __half2 scale2 = __halves2half2(scale, scale);
    unsigned half_bits[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        __nv_fp4x2_e2m1 encoded;
        encoded.__x         = bytes[pair];
        const __half2 value = __hmul2(static_cast<__half2>(encoded), scale2);
        half_bits[pair]     = *reinterpret_cast<const unsigned*>(&value);
    }
    return make_int4(static_cast<int>(half_bits[0]), static_cast<int>(half_bits[1]),
                     static_cast<int>(half_bits[2]), static_cast<int>(half_bits[3]));
}

struct KVCacheNvfp4DequantizedF16x16 {
    int4 lo;
    int4 hi;
};

__device__ __forceinline__ KVCacheNvfp4DequantizedF16x16
kv_cache_nvfp4_dequant_f16x16(const std::uint8_t* codes, std::uint8_t scale_code) {
    const int2 packed         = load_vec<int2>(codes);
    const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(&packed);
    __nv_fp8_e4m3 encoded_scale;
    encoded_scale.__x    = scale_code;
    const __half scale   = static_cast<__half>(encoded_scale);
    const __half2 scale2 = __halves2half2(scale, scale);
    unsigned half_bits[8];
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        __nv_fp4x2_e2m1 encoded;
        encoded.__x         = bytes[pair];
        const __half2 value = __hmul2(static_cast<__half2>(encoded), scale2);
        half_bits[pair]     = *reinterpret_cast<const unsigned*>(&value);
    }
    return {
        make_int4(static_cast<int>(half_bits[0]), static_cast<int>(half_bits[1]),
                  static_cast<int>(half_bits[2]), static_cast<int>(half_bits[3])),
        make_int4(static_cast<int>(half_bits[4]), static_cast<int>(half_bits[5]),
                  static_cast<int>(half_bits[6]), static_cast<int>(half_bits[7])),
    };
}

} // namespace infernix::ops
