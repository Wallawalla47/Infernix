#pragma once

// The skinny GEMV of decode-sized BF16 calls (T <= 8). Every instance accumulates a column in the same
// order (lane l reads chunks l, l + 32, ... in order; a fixed warp butterfly), so the instances differ
// only in speed: a shape selector may choose a different instance per width without changing bits.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/bf16/bf16_launch.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

// Every token column's FMAs over one 16-byte chunk of a warp's rows, in the order the skinny
// kernel's bits are defined by: per token, per row, the chunk's 8 products in k order.
template <int Tokens, int RowsPerWarp>
__device__ __forceinline__ void skinny_accumulate(const __nv_bfloat16* __restrict__ x, std::int32_t K,
                                                  std::int32_t chunk, const uint4 (&w)[RowsPerWarp],
                                                  float (&sums)[RowsPerWarp][Tokens]) {
#pragma unroll
    for (int t = 0; t < Tokens; ++t) {
        const uint4 xp = __ldg(reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(t) * K) + chunk);
        const __nv_bfloat162* xv = reinterpret_cast<const __nv_bfloat162*>(&xp);
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
            const __nv_bfloat162* wv = reinterpret_cast<const __nv_bfloat162*>(&w[r]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 a = __bfloat1622float2(wv[i]);
                const float2 b = __bfloat1622float2(xv[i]);
                sums[r][t]     = fmaf(a.x, b.x, sums[r][t]);
                sums[r][t]     = fmaf(a.y, b.y, sums[r][t]);
            }
        }
    }
}

// Small-T path of the runtime-shape fallback: each warp owns RowsPerWarp weight rows, reads them
// once in 16-byte chunks and accumulates every token column from them, so a decode-sized call
// streams the weight at close to memory bandwidth instead of re-tiling it per 32 tokens. Lane l
// accumulates chunks l, l + 32, ... in order and the warp reduces by a fixed butterfly, so an
// output's bits depend only on K, whatever the instance.
//
// Instances: the default (8 warps, 2 rows per warp) loads a chunk per loop iteration. The small-N
// instance (2 warps, 1 row per warp, Preload = chunks per lane) serves weights of at most
// kSkinnySmallNRows rows, such as the Qwen4Exp GDN control [96, 2560]: its grid spreads over N / 2
// CTAs instead of N / 16, and each lane issues all of its chunk loads before the first FMA.
template <int Tokens, int Warps, int RowsPerWarp, int Preload>
__global__ void __launch_bounds__(Warps * 32)
    bf16_skinny_gemv_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
                            __nv_bfloat16* __restrict__ out, std::int32_t N, std::int32_t K) {
    const std::int32_t lane   = static_cast<std::int32_t>(threadIdx.x % 32);
    const std::int32_t warp   = static_cast<std::int32_t>(threadIdx.x / 32);
    const std::int32_t chunks = K / 8;
    const std::int32_t first  = (static_cast<std::int32_t>(blockIdx.x) * Warps + warp) * RowsPerWarp;
    float sums[RowsPerWarp][Tokens] = {};
    const uint4* rows[RowsPerWarp];
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
        const std::int32_t row = min(first + r, N - 1);
        rows[r] = reinterpret_cast<const uint4*>(weight + static_cast<std::int64_t>(row) * K);
    }
    if constexpr (Preload > 0) {
        uint4 loaded[Preload][RowsPerWarp];
#pragma unroll
        for (int i = 0; i < Preload; ++i) {
            const std::int32_t chunk = lane + 32 * i;
            if (chunk < chunks) {
#pragma unroll
                for (int r = 0; r < RowsPerWarp; ++r) { loaded[i][r] = __ldcs(rows[r] + chunk); }
            }
        }
#pragma unroll
        for (int i = 0; i < Preload; ++i) {
            const std::int32_t chunk = lane + 32 * i;
            if (chunk < chunks) { skinny_accumulate<Tokens, RowsPerWarp>(x, K, chunk, loaded[i], sums); }
        }
    } else {
        for (std::int32_t chunk = lane; chunk < chunks; chunk += 32) {
            uint4 w[RowsPerWarp];
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r) { w[r] = __ldcs(rows[r] + chunk); }
            skinny_accumulate<Tokens, RowsPerWarp>(x, K, chunk, w, sums);
        }
    }
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            float v = sums[r][t];
            for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, offset); }
            if (lane == 0 && first + r < N) {
                out[static_cast<std::int64_t>(t) * N + first + r] = __float2bfloat16_rn(v);
            }
        }
    }
}

template <int Tokens, int Warps, int RowsPerWarp, int Preload>
void launch_bf16_skinny_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr std::int32_t rows_per_block = Warps * RowsPerWarp;
    const dim3 grid(static_cast<unsigned>((weight.n + rows_per_block - 1) / rows_per_block));
    bf16_skinny_gemv_kernel<Tokens, Warps, RowsPerWarp, Preload><<<grid, Warps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        static_cast<__nv_bfloat16*>(out.data), weight.n, weight.k);
    CUDA_CHECK(cudaGetLastError());
}

// The instance <Warps, RowsPerWarp, Preload> for a call of `tokens` columns (1..8). Preload > 0 must
// cover every chunk of a row (Preload * 32 * 8 >= K).
template <int Warps, int RowsPerWarp, int Preload>
Bf16Launch bf16_skinny_launch(std::int32_t tokens) {
    switch (tokens) {
    case 1: return launch_bf16_skinny_gemv<1, Warps, RowsPerWarp, Preload>;
    case 2: return launch_bf16_skinny_gemv<2, Warps, RowsPerWarp, Preload>;
    case 3: return launch_bf16_skinny_gemv<3, Warps, RowsPerWarp, Preload>;
    case 4: return launch_bf16_skinny_gemv<4, Warps, RowsPerWarp, Preload>;
    case 5: return launch_bf16_skinny_gemv<5, Warps, RowsPerWarp, Preload>;
    case 6: return launch_bf16_skinny_gemv<6, Warps, RowsPerWarp, Preload>;
    case 7: return launch_bf16_skinny_gemv<7, Warps, RowsPerWarp, Preload>;
    default: return launch_bf16_skinny_gemv<8, Warps, RowsPerWarp, Preload>;
    }
}

} // namespace infernix::ops::detail
