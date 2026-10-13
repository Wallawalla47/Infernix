#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops {

// Bits 31..8 of `bits` rounded to nearest even on bits 7..0; a NaN stays a quiet NaN.
__device__ __forceinline__ std::uint32_t round_fp24(std::uint32_t bits) {
    if ((bits & 0x7f800000U) == 0x7f800000U && (bits & 0x007fffffU) != 0U) {
        return (bits >> 8U) | 0x4000U;
    }
    return (bits + 0x7fU + ((bits >> 8U) & 1U)) >> 8U;
}

// Four values per thread: one float4 in, three packed 32-bit words out.
__global__ void pack_fp24_kernel(const float4* source, std::uint32_t* packed,
                                 std::int64_t groups) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t g = start; g < groups; g += stride) {
        const float4 value    = source[g];
        const std::uint32_t a = round_fp24(__float_as_uint(value.x));
        const std::uint32_t b = round_fp24(__float_as_uint(value.y));
        const std::uint32_t c = round_fp24(__float_as_uint(value.z));
        const std::uint32_t d = round_fp24(__float_as_uint(value.w));
        std::uint32_t* out    = packed + 3 * g;
        out[0]                = a | (b << 24U);
        out[1]                = (b >> 8U) | (c << 16U);
        out[2]                = (c >> 16U) | (d << 8U);
    }
}

__global__ void unpack_fp24_kernel(const std::uint32_t* packed, float4* destination,
                                   std::int64_t groups) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t g = start; g < groups; g += stride) {
        const std::uint32_t* in = packed + 3 * g;
        const std::uint32_t w0  = in[0];
        const std::uint32_t w1  = in[1];
        const std::uint32_t w2  = in[2];
        destination[g]          = make_float4(__uint_as_float(w0 << 8U),
                                              __uint_as_float(((w0 >> 24U) | (w1 << 8U)) << 8U),
                                              __uint_as_float(((w1 >> 16U) | (w2 << 16U)) << 8U),
                                              __uint_as_float(w2 & 0xffffff00U));
    }
}

__global__ void pack_bf16_kernel(const float4* source, __nv_bfloat162* packed,
                                 std::int64_t groups) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t g = start; g < groups; g += stride) {
        const float4 value = source[g];
        packed[2 * g]      = __floats2bfloat162_rn(value.x, value.y);
        packed[2 * g + 1]  = __floats2bfloat162_rn(value.z, value.w);
    }
}

__global__ void unpack_bf16_kernel(const __nv_bfloat162* packed, float4* destination,
                                   std::int64_t groups) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t g = start; g < groups; g += stride) {
        const float2 low  = __bfloat1622float2(packed[2 * g]);
        const float2 high = __bfloat1622float2(packed[2 * g + 1]);
        destination[g]    = make_float4(low.x, low.y, high.x, high.y);
    }
}

} // namespace infernix::ops
