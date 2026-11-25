// Inkling 2's int8 G64 piece decoder: the int8 attention instances' decode of a pool's or a window ring's rows
// (kv_codec.cuh includes it; it depends on no device symbol, so code without relocatable device code can test it).
#pragma once

#include "ops/common/canonical_math.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::inkling2::detail {

// An int8 G64 row's 8-value piece (its 8 codes, the group's FP16 scale bits) as 8 represented words: a key's BF16 word
// (Value false) or a value's FP16 word of q times s, rounded to nearest even once (q a code as a signed integer, s the
// scale); kv_codec.cuh's kv_decode16 int8 path gives the same bits. q times s is exact in FP32 (8 by 11 significant
// bits), so the packed arithmetic rounds the exact product once too: a value's word by HMUL2 of q (exact in FP16) and
// s, a key's by one packed conversion of the FP32 product. q comes from the code byte biased by 128 (u = q + 128) under
// a constant exponent (FP16 1024 + u, FP32 2^23 + u), the bias subtracted exactly. A non-finite scale takes the scalar
// path (a NaN product's key word is the canonical 0x7FC0 there).
template <bool Value>
__device__ __forceinline__ uint4 kv_decode8_int8(uint2 codes, std::uint16_t scale_bits) {
    const __half scale = __ushort_as_half(scale_bits);
    if (!__hisinf(scale) && !__hisnan(scale)) [[likely]] {
        const std::uint32_t lo = codes.x ^ 0x80808080U, hi = codes.y ^ 0x80808080U; // u = q + 128 in each byte
        if constexpr (Value) {
            const __half2 s2   = __half2half2(scale);
            const __half2 bias = __half2half2(__ushort_as_half(static_cast<unsigned short>(0x6480U))); // 1152 = 1024 + 128
            const auto pair    = [&](std::uint32_t w, std::uint32_t selector) {
                const std::uint32_t h = __byte_perm(w, 0x64646464U, selector); // {1024 + u_a, 1024 + u_b}
                const __half2 r       = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&h), bias), s2);
                return *reinterpret_cast<const std::uint32_t*>(&r);
            };
            return make_uint4(pair(lo, 0x5140U), pair(lo, 0x5342U), pair(hi, 0x5140U), pair(hi, 0x5342U));
        } else {
            const float s   = __half2float(scale);
            const auto word = [&](std::uint32_t w, std::uint32_t k) { // byte k of w as q * s (exact)
                return (__uint_as_float(__byte_perm(w, 0x4B000000U, 0x7540U + k)) - 8388736.0F) * s;
            };
            const auto pair = [&](std::uint32_t w, std::uint32_t k) {
                const __nv_bfloat162 r = __floats2bfloat162_rn(word(w, k), word(w, k + 1));
                return *reinterpret_cast<const std::uint32_t*>(&r);
            };
            return make_uint4(pair(lo, 0), pair(lo, 2), pair(hi, 0), pair(hi, 2));
        }
    }
    std::uint32_t out[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const std::uint32_t two = (i < 2 ? codes.x : codes.y) >> (16 * (i % 2));
        const float v0 = static_cast<float>(static_cast<std::int8_t>(two & 0xFFU)) * __half2float(scale);
        const float v1 = static_cast<float>(static_cast<std::int8_t>((two >> 8) & 0xFFU)) * __half2float(scale);
        const auto w   = [](float x) {
            return Value ? std::uint32_t{__half_as_ushort(__float2half_rn(x))} : std::uint32_t{canon::f32_to_bf16_rn(x)};
        };
        out[i] = w(v0) | (w(v1) << 16);
    }
    return make_uint4(out[0], out[1], out[2], out[3]);
}

} // namespace infernix::ops::inkling2::detail
