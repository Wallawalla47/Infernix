#pragma once

// Implements: include/infernix/ops/add_bias.h
// Match: contiguous BF16 [D,C]. A row width that is a multiple of 8 with 16-byte aligned storage
// streams the whole tensor as 16-byte packs, one per thread, so a width that is not a multiple of
// the block (vision fc1's 4304 = 538 packs) leaves no mostly-idle tail block per row; other even
// widths use a BF16x2 stream, and scalar indexing covers odd or unaligned storage.

#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops {

inline constexpr int kAddBiasPairsPerThread = 4;

__device__ __forceinline__ __nv_bfloat162 add_bias_pair(__nv_bfloat162 value, __nv_bfloat162 bias) {
    return __floats2bfloat162_rn(__low2float(value) + __low2float(bias),
                                 __high2float(value) + __high2float(bias));
}

// total = D * C / 8 packs, fewer than 2^32; packs = D / 8 per row.
template <int Block>
__launch_bounds__(Block) __global__
    void add_bias_bf16x8_stream_kernel(const Bf16x8Pack* bias, Bf16x8Pack* x, std::uint32_t packs,
                                       std::uint32_t total) {
    const std::uint32_t index = blockIdx.x * static_cast<std::uint32_t>(Block) + threadIdx.x;
    if (index >= total) { return; }
    const Bf16x8Pack bv = load_vec<Bf16x8Pack>(bias + index % packs);
    Bf16x8Pack value    = load_vec<Bf16x8Pack>(x + index);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) { value.pair[pair] = add_bias_pair(value.pair[pair], bv.pair[pair]); }
    store_vec(x + index, value);
}

template <int Block>
__launch_bounds__(Block) __global__
    void add_bias_bf16x2_kernel(const __nv_bfloat162* bias, __nv_bfloat162* x, std::int32_t pairs,
                                std::int32_t rows) {
    const std::int32_t first =
        (static_cast<std::int32_t>(blockIdx.x) * Block + static_cast<std::int32_t>(threadIdx.x)) *
        kAddBiasPairsPerThread;
    for (std::int32_t row = static_cast<std::int32_t>(blockIdx.y); row < rows;
         row += static_cast<std::int32_t>(gridDim.y)) {
        const std::int64_t base = static_cast<std::int64_t>(row) * pairs;
#pragma unroll
        for (int item = 0; item < kAddBiasPairsPerThread; ++item) {
            const std::int32_t pair = first + item;
            if (pair < pairs) { x[base + pair] = add_bias_pair(x[base + pair], bias[pair]); }
        }
    }
}

__global__ void add_bias_kernel(const __nv_bfloat16* bias, __nv_bfloat16* x, std::int32_t d,
                                std::int64_t n) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const float value = __bfloat162float(x[i]) + __bfloat162float(bias[i % d]);
        x[i]              = __float2bfloat16_rn(value);
    }
}

} // namespace infernix::ops
