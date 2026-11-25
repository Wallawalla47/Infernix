#pragma once

// General-purpose BF16 x BF16 GEMM fallback for shapes the specialised bf16 linear path does not
// tile (for example a full-precision vocab projection kept as BF16 by a QAT checkpoint). Computes
// out[T, N] = x[T, K] * weight[N, K]^T with an fp32 accumulation and a BF16 store, matching the
// contiguous token-major layout of the specialised kernels (or an FP32 store: Out = float, a vocabulary
// head's logits). Tiling is fixed and the extents are runtime, so any (N, K, T) is handled without a
// per-shape instantiation.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {
// A 64 x 64 output tile per CTA, 4 x 4 outputs a thread. Every output is one fmaf chain over k = 0 .. K - 1 in order,
// so the tile does not enter the bits: the 32 x 32 tile it replaced gave the same ones at a quarter of the
// arithmetic per shared-memory read (candidates NQ-06: recipe C's QSA indexer [640, 2560], infernix_linear_bench,
// cold L2, three rounds, 32 against 64: 64 columns 270.9 -> 246.4 us, 1,024 373.9 -> 257.2, 4,096 1,131.7 -> 605.3,
// 16,384 4,117.6 -> 2,143.8; ahead at every width of a 64-column sweep, 2026-11-04).
constexpr std::int32_t kTileN         = 64;
constexpr std::int32_t kTileT         = 64;
constexpr std::int32_t kTileK         = 32;
constexpr std::int32_t kThreads       = 256;
constexpr std::int32_t kThreadCols    = kThreads / 16;          // 16
constexpr std::int32_t kRowsPerThread = kTileN / 16;            // 4
constexpr std::int32_t kColsPerThread = kTileT / kThreadCols;   // 4

// An output element from its FP32 sum: rounded to BF16, or the sum itself.
__device__ __forceinline__ void bf16_linear_store(__nv_bfloat16* out, float value) { *out = __float2bfloat16_rn(value); }
__device__ __forceinline__ void bf16_linear_store(float* out, float value) { *out = value; }

template <class Out>
__global__ void bf16_general_gemm_kernel(const __nv_bfloat16* __restrict__ x,
                                         const __nv_bfloat16* __restrict__ weight,
                                         Out* __restrict__ out, std::int32_t N,
                                         std::int32_t K, std::int32_t T) {
    __shared__ __nv_bfloat16 ws[kTileN][kTileK + 1];
    __shared__ __nv_bfloat16 wx[kTileT][kTileK + 1];

    const std::int32_t out_row0 = static_cast<std::int32_t>(blockIdx.x) * kTileN;
    const std::int32_t token0   = static_cast<std::int32_t>(blockIdx.y) * kTileT;
    const std::int32_t tid      = static_cast<std::int32_t>(threadIdx.x);

    const std::int32_t trow = tid / kThreadCols;  // 0..15
    const std::int32_t tcol = tid % kThreadCols;  // 0..15

    float accum[kRowsPerThread][kColsPerThread] = {};

    for (std::int32_t k0 = 0; k0 < K; k0 += kTileK) {
        for (std::int32_t i = tid; i < kTileN * kTileK; i += kThreads) {
            const std::int32_t r  = i / kTileK;
            const std::int32_t c  = i % kTileK;
            const std::int32_t gr = out_row0 + r;
            const std::int32_t gc = k0 + c;
            ws[r][c + 1] = (gr < N && gc < K)
                               ? weight[static_cast<std::int64_t>(gr) * K + gc]
                               : __float2bfloat16_rn(0.0f);
        }
        for (std::int32_t i = tid; i < kTileT * kTileK; i += kThreads) {
            const std::int32_t r  = i / kTileK;
            const std::int32_t c  = i % kTileK;
            const std::int32_t gr = token0 + r;
            const std::int32_t gc = k0 + c;
            wx[r][c + 1] = (gr < T && gc < K)
                               ? x[static_cast<std::int64_t>(gr) * K + gc]
                               : __float2bfloat16_rn(0.0f);
        }
        __syncthreads();

#pragma unroll
        for (std::int32_t rr = 0; rr < kRowsPerThread; ++rr) {
            const std::int32_t row = trow * kRowsPerThread + rr;
#pragma unroll
            for (std::int32_t cc = 0; cc < kColsPerThread; ++cc) {
                const std::int32_t col = tcol * kColsPerThread + cc;
                float sum = accum[rr][cc];
#pragma unroll
                for (std::int32_t kk = 0; kk < kTileK; ++kk) {
                    sum = fmaf(static_cast<float>(wx[col][kk + 1]),
                               static_cast<float>(ws[row][kk + 1]), sum);
                }
                accum[rr][cc] = sum;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (std::int32_t rr = 0; rr < kRowsPerThread; ++rr) {
        const std::int32_t row = out_row0 + trow * kRowsPerThread + rr;
        if (row < N) {
#pragma unroll
            for (std::int32_t cc = 0; cc < kColsPerThread; ++cc) {
                const std::int32_t col = token0 + tcol * kColsPerThread + cc;
                if (col < T) { bf16_linear_store(&out[static_cast<std::int64_t>(col) * N + row], accum[rr][cc]); }
            }
        }
    }
}

// Calls of 9 to 64 columns of a weight of at most kChainMaxRows rows (BF16 output, rows in 16-byte chunks): one
// thread per output, its fmaf chain over k = 0 .. K - 1 in order with the tile's operands (x first) and the tile's
// zero-padded tail, so every output has the tile's bits, and a column's bits stay the same at every width past the
// skinny route. A block holds every column of 64 / Tp rows (Tp: the columns rounded up to 16, 32 or 64), so a weight
// row's chunks are read by one block, from cache after its first thread. The 64 x 64 tile gave Flash-Next's QSA
// indexer [640, 2560] ten CTAs at 16 columns; this gives it 160.
constexpr std::int32_t kChainMaxTokens = 64;
constexpr std::int32_t kChainMaxRows   = 32768;
constexpr std::int32_t kChainThreads   = 64;
constexpr std::int32_t kChainAhead     = 4; // 16-byte chunks loaded before their FMAs

__device__ __forceinline__ float bf16_chain8(const uint4& xv, const uint4& wv, float sum) {
    const auto* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
    const auto* wp = reinterpret_cast<const __nv_bfloat162*>(&wv);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 xf = __bfloat1622float2(xp[i]);
        const float2 wf = __bfloat1622float2(wp[i]);
        sum             = fmaf(xf.x, wf.x, sum);
        sum             = fmaf(xf.y, wf.y, sum);
    }
    return sum;
}

template <class Out>
__global__ void __launch_bounds__(kChainThreads)
    bf16_general_chain_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
                              Out* __restrict__ out, std::int32_t N, std::int32_t K, std::int32_t T,
                              std::int32_t token_shift) {
    const auto tid           = static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t token = tid & ((1 << token_shift) - 1);
    const std::int32_t row   = static_cast<std::int32_t>(blockIdx.x) * (kChainThreads >> token_shift) + (tid >> token_shift);
    if (row >= N || token >= T) { return; }
    const auto* w              = reinterpret_cast<const uint4*>(weight + static_cast<std::int64_t>(row) * K);
    const auto* xs             = reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(token) * K);
    const std::int32_t vectors = K / 8;
    float sum                  = 0.0f;
    std::int32_t v             = 0;
    for (; v + kChainAhead <= vectors; v += kChainAhead) {
        uint4 wv[kChainAhead], xv[kChainAhead];
#pragma unroll
        for (int u = 0; u < kChainAhead; ++u) {
            wv[u] = __ldg(w + v + u);
            xv[u] = __ldg(xs + v + u);
        }
#pragma unroll
        for (int u = 0; u < kChainAhead; ++u) { sum = bf16_chain8(xv[u], wv[u], sum); }
    }
    for (; v < vectors; ++v) { sum = bf16_chain8(__ldg(xs + v), __ldg(w + v), sum); }
    // The tile's last K chunk is zero-padded past K: fmaf(0, 0, sum), which turns -0 into +0 once.
    if (K % kTileK != 0) { sum = fmaf(0.0f, 0.0f, sum); }
    bf16_linear_store(&out[static_cast<std::int64_t>(token) * N + row], sum);
}

} // namespace infernix::ops::detail
