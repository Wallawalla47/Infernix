#pragma once

// bf16_p12 device code (core/weight.h QuantLayout::P12Rows): a chunk of 8 weights is 8 code bytes (sign << 7 |
// mantissa) and a 32-bit word of exponent nibbles; it decodes to the same uint4 of BF16 words the BF16 kernels
// load, so a kernel that keeps a BF16 kernel's chunk order keeps its bits.

#include "core/device.h" // CUDA_CHECK
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace infernix::ops::detail {

inline constexpr int kBf16P12SkinnyMaxTokens = 16;

// Whether any of a chunk's 8 nibbles is the escape code 15.
__device__ __forceinline__ bool p12_has_escape(unsigned codes) {
    return (codes & (codes >> 1) & (codes >> 2) & (codes >> 3) & 0x11111111U) != 0;
}

// Two BF16 words (low, high) from a code word's two bytes (sel 0x4140: bytes 0, 1; 0x4342: bytes 2, 3), the
// two nibbles in the low byte of `codes2`, and the row's base in both halves of `base2`.
__device__ __forceinline__ unsigned p12_decode_pair(unsigned code_word, unsigned sel, unsigned codes2, unsigned base2) {
    const unsigned t = __byte_perm(code_word, 0, sel);
    const unsigned e = (codes2 & 0xFU) | ((codes2 & 0xF0U) << 12);
    return ((t & 0x00800080U) << 8) | ((e + base2) << 7) | (t & 0x007F007FU);
}

// The escaped weights of a pair: each from the row's escape list [lo, hi), which is in column order.
__device__ __noinline__ inline unsigned p12_patch_pair(unsigned word, unsigned codes2, unsigned column, unsigned lo, unsigned hi,
                                                const unsigned* __restrict__ escapes) {
#pragma unroll 1
    for (int half = 0; half < 2; ++half) {
        if (((codes2 >> (4 * half)) & 0xFU) != 15U) { continue; }
        const unsigned want = column + static_cast<unsigned>(half);
        unsigned a = lo, b = hi;
        while (a < b) {
            const unsigned mid = (a + b) / 2;
            if ((escapes[mid] >> 16) < want) {
                a = mid + 1;
            } else {
                b = mid;
            }
        }
        word = (word & (0xFFFFU << (16 * (1 - half)))) | ((escapes[a] & 0xFFFFU) << (16 * half));
    }
    return word;
}

// One row's decoding state: its base in both halves and its escape range.
struct P12Row {
    unsigned base2;
    unsigned lo, hi;
};

__device__ __forceinline__ P12Row p12_row(const unsigned* __restrict__ words, std::int64_t row) {
    const unsigned w = words[row], next = words[row + 1];
    return {(w & 0xFFU) * 0x10001U, w >> 8, next >> 8};
}

__device__ __forceinline__ uint4 p12_decode_chunk(uint2 code, unsigned codes, const P12Row& row, std::int32_t chunk,
                                                  const unsigned* __restrict__ escapes) {
    uint4 w;
    w.x = p12_decode_pair(code.x, 0x4140, codes, row.base2);
    w.y = p12_decode_pair(code.x, 0x4342, codes >> 8, row.base2);
    w.z = p12_decode_pair(code.y, 0x4140, codes >> 16, row.base2);
    w.w = p12_decode_pair(code.y, 0x4342, codes >> 24, row.base2);
    if (p12_has_escape(codes)) {
        const unsigned column = static_cast<unsigned>(chunk) * 8U;
        w.x = p12_patch_pair(w.x, codes, column, row.lo, row.hi, escapes);
        w.y = p12_patch_pair(w.y, codes >> 8, column + 2, row.lo, row.hi, escapes);
        w.z = p12_patch_pair(w.z, codes >> 16, column + 4, row.lo, row.hi, escapes);
        w.w = p12_patch_pair(w.w, codes >> 24, column + 6, row.lo, row.hi, escapes);
    }
    return w;
}

// The skinny GEMV of decode-sized calls over bf16_p12 weights: the BF16 skinny kernel's order (bf16_skinny.cuh:
// lane l takes chunks l, l + 32, ... of each of its warp's rows in order; per chunk, per token, per row, 8 FMAs in
// k order; a fixed butterfly), so an output has the bits that kernel gives the decoded weights, whatever the
// instance and the number of columns. `Ahead` chunks of each row are in flight before the first is decoded.
template <int Tokens, int Warps, int RowsPerWarp, int Ahead, class Out>
__global__ void __launch_bounds__(Warps * 32)
    bf16_p12_skinny_gemv_kernel(const __nv_bfloat16* __restrict__ x, const uint2* __restrict__ codes_plane,
                                const unsigned* __restrict__ nibble_plane, const unsigned* __restrict__ words,
                                const unsigned* __restrict__ escapes, Out* __restrict__ out, std::int32_t N,
                                std::int32_t K) {
    const std::int32_t lane   = static_cast<std::int32_t>(threadIdx.x % 32);
    const std::int32_t warp   = static_cast<std::int32_t>(threadIdx.x / 32);
    const std::int32_t chunks = K / 8;
    const std::int32_t first  = (static_cast<std::int32_t>(blockIdx.x) * Warps + warp) * RowsPerWarp;
    float sums[RowsPerWarp][Tokens] = {};
    std::int64_t base[RowsPerWarp];
    P12Row rows[RowsPerWarp];
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
        const std::int64_t row = min(first + r, N - 1);
        base[r]                = row * chunks;
        rows[r]                = p12_row(words, row);
    }
    uint2 code[Ahead][RowsPerWarp];
    unsigned nib[Ahead][RowsPerWarp];
#pragma unroll
    for (int a = 0; a < Ahead; ++a) {
        const std::int32_t chunk = lane + 32 * a;
        if (chunk < chunks) {
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r) {
                code[a][r] = __ldcs(codes_plane + base[r] + chunk);
                nib[a][r]  = __ldcs(nibble_plane + base[r] + chunk);
            }
        }
    }
    for (std::int32_t chunk0 = lane; chunk0 < chunks; chunk0 += 32 * Ahead) {
#pragma unroll
        for (int a = 0; a < Ahead; ++a) {
            const std::int32_t chunk = chunk0 + 32 * a;
            if (chunk >= chunks) { break; }
            uint4 w[RowsPerWarp];
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r) { w[r] = p12_decode_chunk(code[a][r], nib[a][r], rows[r], chunk, escapes); }
            const std::int32_t next = chunk + 32 * Ahead;
            if (next < chunks) {
#pragma unroll
                for (int r = 0; r < RowsPerWarp; ++r) {
                    code[a][r] = __ldcs(codes_plane + base[r] + next);
                    nib[a][r]  = __ldcs(nibble_plane + base[r] + next);
                }
            }
#pragma unroll
            for (int t = 0; t < Tokens; ++t) {
                const uint4 xp = __ldg(reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(t) * K) + chunk);
                const __nv_bfloat162* xv = reinterpret_cast<const __nv_bfloat162*>(&xp);
#pragma unroll
                for (int r = 0; r < RowsPerWarp; ++r) {
                    const __nv_bfloat162* wv = reinterpret_cast<const __nv_bfloat162*>(&w[r]);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float2 wf = __bfloat1622float2(wv[i]);
                        const float2 xf = __bfloat1622float2(xv[i]);
                        sums[r][t]      = fmaf(wf.x, xf.x, sums[r][t]);
                        sums[r][t]      = fmaf(wf.y, xf.y, sums[r][t]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            float v = sums[r][t];
            for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, offset); }
            if (lane == 0 && first + r < N) {
                if constexpr (std::is_same_v<Out, float>) {
                    out[static_cast<std::int64_t>(t) * N + first + r] = v;
                } else {
                    out[static_cast<std::int64_t>(t) * N + first + r] = __float2bfloat16_rn(v);
                }
            }
        }
    }
}

template <int Tokens, int Warps, int RowsPerWarp, int Ahead>
void launch_bf16_p12_skinny(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr std::int32_t rows_per_block = Warps * RowsPerWarp;
    const dim3 grid(static_cast<unsigned>((w.n + rows_per_block - 1) / rows_per_block));
    const auto* xs      = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes   = static_cast<const uint2*>(w.qdata);
    const auto* nibbles = static_cast<const unsigned*>(w.qhigh);
    const auto* words   = static_cast<const unsigned*>(w.scales);
    const auto* escapes = static_cast<const unsigned*>(w.escapes);
    if (out.dtype == DType::FP32) {
        bf16_p12_skinny_gemv_kernel<Tokens, Warps, RowsPerWarp, Ahead><<<grid, Warps * 32, 0, stream>>>(
            xs, codes, nibbles, words, escapes, static_cast<float*>(out.data), w.n, w.k);
    } else {
        bf16_p12_skinny_gemv_kernel<Tokens, Warps, RowsPerWarp, Ahead><<<grid, Warps * 32, 0, stream>>>(
            xs, codes, nibbles, words, escapes, static_cast<__nv_bfloat16*>(out.data), w.n, w.k);
    }
    CUDA_CHECK(cudaGetLastError());
}

using P12Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// The skinny instance <Warps, RowsPerWarp, Ahead> for a call of `tokens` (1 .. kBf16P12SkinnyMaxTokens) columns.
template <int Warps, int RowsPerWarp, int Ahead>
P12Launch bf16_p12_skinny_launch(std::int32_t tokens) {
    switch (tokens) {
    case 1: return launch_bf16_p12_skinny<1, Warps, RowsPerWarp, Ahead>;
    case 2: return launch_bf16_p12_skinny<2, Warps, RowsPerWarp, Ahead>;
    case 3: return launch_bf16_p12_skinny<3, Warps, RowsPerWarp, Ahead>;
    case 4: return launch_bf16_p12_skinny<4, Warps, RowsPerWarp, Ahead>;
    case 5: return launch_bf16_p12_skinny<5, Warps, RowsPerWarp, Ahead>;
    case 6: return launch_bf16_p12_skinny<6, Warps, RowsPerWarp, Ahead>;
    case 7: return launch_bf16_p12_skinny<7, Warps, RowsPerWarp, Ahead>;
    case 8: return launch_bf16_p12_skinny<8, Warps, RowsPerWarp, Ahead>;
    case 9: return launch_bf16_p12_skinny<9, Warps, RowsPerWarp, Ahead>;
    case 10: return launch_bf16_p12_skinny<10, Warps, RowsPerWarp, Ahead>;
    case 11: return launch_bf16_p12_skinny<11, Warps, RowsPerWarp, Ahead>;
    case 12: return launch_bf16_p12_skinny<12, Warps, RowsPerWarp, Ahead>;
    case 13: return launch_bf16_p12_skinny<13, Warps, RowsPerWarp, Ahead>;
    case 14: return launch_bf16_p12_skinny<14, Warps, RowsPerWarp, Ahead>;
    case 15: return launch_bf16_p12_skinny<15, Warps, RowsPerWarp, Ahead>;
    default: return launch_bf16_p12_skinny<16, Warps, RowsPerWarp, Ahead>;
    }
}

} // namespace infernix::ops::detail
