#include "ops/linear/bf16/bf16_general.cuh"
#include "core/device.h"
#include "ops/linear/bf16/bf16_dispatch.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kSkinnyMaxTokens  = 8;
constexpr std::int32_t kSkinnyWarps      = 8;
constexpr std::int32_t kSkinnyRowsPerWarp = 2;

// Small-T path of the runtime-shape fallback: one warp per weight row reads the row once in
// 16-byte chunks and accumulates every token column from it, so a decode-sized call streams the
// weight at close to memory bandwidth instead of re-tiling it per 32 tokens. Lane l accumulates
// chunks l, l + 32, ... in order and the warp reduces by a fixed butterfly.
template <int Tokens>
__global__ void __launch_bounds__(kSkinnyWarps * 32)
    bf16_skinny_gemv_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
                            __nv_bfloat16* __restrict__ out, std::int32_t N, std::int32_t K) {
    const std::int32_t lane   = static_cast<std::int32_t>(threadIdx.x % 32);
    const std::int32_t warp   = static_cast<std::int32_t>(threadIdx.x / 32);
    const std::int32_t chunks = K / 8;
    const std::int32_t first  = (static_cast<std::int32_t>(blockIdx.x) * kSkinnyWarps + warp) * kSkinnyRowsPerWarp;
    float sums[kSkinnyRowsPerWarp][Tokens] = {};
    const uint4* rows[kSkinnyRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kSkinnyRowsPerWarp; ++r) {
        const std::int32_t row = min(first + r, N - 1);
        rows[r] = reinterpret_cast<const uint4*>(weight + static_cast<std::int64_t>(row) * K);
    }
    for (std::int32_t chunk = lane; chunk < chunks; chunk += 32) {
        uint4 w[kSkinnyRowsPerWarp];
#pragma unroll
        for (int r = 0; r < kSkinnyRowsPerWarp; ++r) { w[r] = __ldcs(rows[r] + chunk); }
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            const uint4 xp = __ldg(reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(t) * K) + chunk);
            const __nv_bfloat162* xv = reinterpret_cast<const __nv_bfloat162*>(&xp);
#pragma unroll
            for (int r = 0; r < kSkinnyRowsPerWarp; ++r) {
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
#pragma unroll
    for (int r = 0; r < kSkinnyRowsPerWarp; ++r) {
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

template <int Tokens>
void bf16_skinny_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows_per_block = kSkinnyWarps * kSkinnyRowsPerWarp;
    const dim3 grid(static_cast<unsigned>((weight.n + rows_per_block - 1) / rows_per_block));
    bf16_skinny_gemv_kernel<Tokens><<<grid, kSkinnyWarps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        static_cast<__nv_bfloat16*>(out.data), weight.n, weight.k);
    CUDA_CHECK(cudaGetLastError());
}

void bf16_skinny_dispatch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    switch (x.ne[1]) {
    case 1: return bf16_skinny_gemv<1>(x, weight, out, stream);
    case 2: return bf16_skinny_gemv<2>(x, weight, out, stream);
    case 3: return bf16_skinny_gemv<3>(x, weight, out, stream);
    case 4: return bf16_skinny_gemv<4>(x, weight, out, stream);
    case 5: return bf16_skinny_gemv<5>(x, weight, out, stream);
    case 6: return bf16_skinny_gemv<6>(x, weight, out, stream);
    case 7: return bf16_skinny_gemv<7>(x, weight, out, stream);
    default: return bf16_skinny_gemv<8>(x, weight, out, stream);
    }
}

// Runtime-shape bf16 GEMM used for every linear projection the specialised bf16 kernels do not
// tile. x is [tokens, k], weight is [n, k] row-major, and out is [tokens, n] token-major, matching
// the contiguous layout the specialised kernels produce.
void bf16_general_gemm(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t N = weight.n;
    const std::int32_t K = weight.k;
    const std::int32_t T = x.ne[1];
    const dim3 grid(static_cast<unsigned>((N + kTileN - 1) / kTileN),
                    static_cast<unsigned>((T + kTileT - 1) / kTileT));
    bf16_general_gemm_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        static_cast<__nv_bfloat16*>(out.data), N, K, T);
    CUDA_CHECK(cudaGetLastError());
}

// Decode-sized calls stream each weight row once when the rows can be read in 16-byte chunks.
void bf16_general_small(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const bool vector = weight.k % 8 == 0 && reinterpret_cast<std::uintptr_t>(weight.qdata) % 16 == 0 &&
                        reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0;
    if (vector) {
        bf16_skinny_dispatch(x, weight, out, stream);
    } else {
        bf16_general_gemm(x, weight, out, stream);
    }
}

} // namespace

[[nodiscard]] Bf16Launch select_bf16_general_launch(std::int32_t tokens) {
    return tokens <= kSkinnyMaxTokens ? &bf16_general_small : &bf16_general_gemm;
}

} // namespace ninfer::ops::detail
