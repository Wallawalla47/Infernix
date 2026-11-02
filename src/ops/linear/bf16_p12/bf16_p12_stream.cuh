#pragma once

// The tensor-core route of bf16_p12 calls wider than the skinny GEMV serves: the weights stream from the packed
// planes straight into registers, are decoded there into the A fragments of m16n8k16 MMAs, and never pass
// through shared memory; only the call's activations are staged (a cp.async ring of swizzled tiles that every
// warp of the CTA shares).
//
// Fragments. Each lane (g = lane / 4, t = lane % 4) loads, per 128-weight step of its K slice, the 32 codes and
// 32 nibbles of chunks 4t .. 4t + 3 of rows g and g + 8 of each of its m16 row groups (two 16-byte code loads and
// one 16-byte nibble load a row). In sub-step j it decodes chunk 4t + j of both rows (bf16_p12_kernels.cuh) and
// feeds its eight weights to two MMAs: the first takes weights 0-3 of the chunk where the fragment layout has k
// 2t, 2t + 1, 2t + 8 and 2t + 9, the second weights 4-7. The B fragments are the same real k of the activation
// row of token g, one 16-byte shared load for both MMAs. The two MMAs of a sub-step thus sum 32 distinct k of the
// row, and the steps cover K once.
//
// Arithmetic. An output (row, column) is the FP32 MMA accumulation over its K slice's steps in order (sub-steps
// j = 0 .. 3, each its two MMAs in order), then the slices added in order 0, 1, ...: it depends on the row's
// weights, the column's activations and the schedule's K slices only. Any two schedules with the same K slices
// give the same bits, at every width (the route is column-invariant for a fixed slice count).

#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/math.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/bf16/bf16_mma_common.cuh"
#include "ops/linear/bf16_p12/bf16_p12_kernels.cuh"
#include "ops/linear/common/output.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace infernix::ops::detail {

// RowGroups: m16 row groups a warp computes; Tokens: columns of a CTA tile (a multiple of 8); RowWarps: warps along
// the rows; KSlices: warps that split each step's K (the CTA's slices, added in order); Ahead: steps of weights in
// flight a lane holds in registers; Stages: activation tiles in the ring.
template <int RowGroups, int Tokens, int RowWarps, int KSlices, int Ahead, int Stages, int MinBlocks = 1>
struct P12StreamSchedule {
    static constexpr int kRowGroups  = RowGroups;
    static constexpr int kTokens     = Tokens;
    static constexpr int kRowWarps   = RowWarps;
    static constexpr int kKSlices    = KSlices;
    static constexpr int kAhead      = Ahead;
    static constexpr int kStages     = Stages;
    static constexpr int kMinBlocks  = MinBlocks;
    static constexpr int kWarpRows   = 16 * RowGroups;
    static constexpr int kBlockRows  = kWarpRows * RowWarps;
    static constexpr int kThreads    = 32 * RowWarps * KSlices;
    static constexpr int kStepK      = 128 * KSlices;
    static constexpr int kTiles      = Tokens / 8;
    static constexpr int kStageBytes = Tokens * KSlices * 256;
    // The slices' partial sums (slices 1 .. KSlices - 1), then the BF16 output tile of each row warp.
    static constexpr int kReduceBytes   = (KSlices - 1) * RowWarps * RowGroups * kTiles * 4 * 32 * 4;
    static constexpr int kEpilogueBytes = RowWarps * kWarpRows * Tokens * 2;
    static constexpr int kSharedBytes   = [] {
        int bytes = Stages * kStageBytes;
        bytes     = bytes > kReduceBytes ? bytes : kReduceBytes;
        return bytes > kEpilogueBytes ? bytes : kEpilogueBytes;
    }();
    static_assert(RowGroups >= 1 && Tokens % 8 == 0 && Tokens >= 8 && RowWarps >= 1 && KSlices >= 1);
    static_assert(Ahead >= 1 && Stages >= 2 && Stages <= 8 && kThreads <= 1024);
    static_assert(kSharedBytes <= 99 * 1024);
};

struct P12StreamOperands {
    const __nv_bfloat16* x;
    const std::uint8_t* codes;
    const std::uint8_t* nibbles;
    const unsigned* words;
    const unsigned* escapes;
    int rows, k, tokens;
};

// The 16-byte unit `u` (of 16) of token row `r`'s 128-k segment, swizzled so that the eight lanes of a quarter warp
// (two rows of one parity each, units 4t + j for t = 0 .. 3) read eight distinct bank groups.
__device__ __forceinline__ int p12_stream_unit(int u, int r) { return u ^ (((u >> 3) & 1) << 1) ^ (r & 1); }

// One row's four chunks of a step: codes of chunks 0-1 and 2-3, and their four nibble words.
struct P12Chunks {
    uint4 lo, hi, nib;
};

template <class S, class Output>
__global__ void __launch_bounds__(S::kThreads, S::kMinBlocks)
    bf16_p12_stream_kernel(P12StreamOperands p, Output output) {
    constexpr int RG = S::kRowGroups, NT = S::kTiles, KS = S::kKSlices, AH = S::kAhead, ST = S::kStages;
    extern __shared__ __align__(16) unsigned char shared[];
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int slice = warp / S::kRowWarps, row_warp = warp % S::kRowWarps;
    const int g = lane >> 2, t = lane & 3;
    const int token_blocks = (p.tokens + S::kTokens - 1) / S::kTokens;
    const int row_block    = static_cast<int>(blockIdx.x) / token_blocks;
    const int n0           = (static_cast<int>(blockIdx.x) - row_block * token_blocks) * S::kTokens;
    const int warp_row     = row_block * S::kBlockRows + row_warp * S::kWarpRows;
    const bool active      = warp_row < p.rows; // rows are whole row groups: a warp is in or out
    const int K = p.k, steps = K / S::kStepK;

    // Each row's decoding state and plane cursors (uint4 units; a step advances codes by kStepK / 16 units and
    // nibbles by kStepK / 32).
    P12Row rows[RG][2];
    const uint4* codes[RG][2];
    const uint4* nibbles[RG][2];
#pragma unroll
    for (int mi = 0; mi < RG; ++mi) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const std::int64_t row = min(warp_row + mi * 16 + g + 8 * h, p.rows - 1);
            rows[mi][h]            = p12_row(p.words, row);
            codes[mi][h]   = reinterpret_cast<const uint4*>(p.codes + row * K + slice * 128 + 32 * t);
            nibbles[mi][h] = reinterpret_cast<const uint4*>(p.nibbles + row * (K / 2) + slice * 64 + 16 * t);
        }
    }

    P12Chunks held[AH][RG][2];
    const auto load = [&](P12Chunks (&into)[RG][2], int step) {
#pragma unroll
        for (int mi = 0; mi < RG; ++mi) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint4* c   = codes[mi][h] + static_cast<std::int64_t>(step) * (S::kStepK / 16);
                into[mi][h].lo  = __ldcs(c);
                into[mi][h].hi  = __ldcs(c + 1);
                into[mi][h].nib = __ldcs(nibbles[mi][h] + static_cast<std::int64_t>(step) * (S::kStepK / 32));
            }
        }
    };

    // The step's activation tile: [token][slice][16 swizzled units]; columns past the call are zero.
    const auto stage = [&](int slot, int step) {
        unsigned char* tile = shared + slot * S::kStageBytes;
#pragma unroll 1
        for (int item = tid; item < S::kTokens * KS * 16; item += S::kThreads) {
            const int r = item / (KS * 16), rest = item - r * (KS * 16);
            const int s = rest >> 4, u = rest & 15;
            const int token  = n0 + r;
            const bool valid = token < p.tokens;
            const auto* src =
                p.x + (valid ? static_cast<std::int64_t>(token) * K + static_cast<std::int64_t>(step) * S::kStepK +
                                   s * 128 + u * 8
                             : 0);
            cp_async_zfill<16, Cache::cg>(tile + ((r * KS + s) * 16 + p12_stream_unit(u, r)) * 16, src, valid ? 16 : 0);
        }
    };

    float acc[RG][NT][4] = {};
    const auto compute = [&](const P12Chunks (&chunks)[RG][2], int slot, int step) {
        const unsigned char* tile = shared + slot * S::kStageBytes;
        const int chunk0          = (step * S::kStepK + slice * 128) / 8 + 4 * t;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint4 w[RG][2];
#pragma unroll
            for (int mi = 0; mi < RG; ++mi) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const P12Chunks& c = chunks[mi][h];
                    const uint2 code   = j == 0 ? make_uint2(c.lo.x, c.lo.y)
                                         : j == 1 ? make_uint2(c.lo.z, c.lo.w)
                                         : j == 2 ? make_uint2(c.hi.x, c.hi.y)
                                                  : make_uint2(c.hi.z, c.hi.w);
                    const unsigned nib = j == 0 ? c.nib.x : j == 1 ? c.nib.y : j == 2 ? c.nib.z : c.nib.w;
                    w[mi][h]           = p12_decode_chunk(code, nib, rows[mi][h], chunk0 + j, p.escapes);
                }
            }
            const int unit = p12_stream_unit(4 * t + j, g);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const uint4 xv = *reinterpret_cast<const uint4*>(tile + (((nt * 8 + g) * KS + slice) * 16 + unit) * 16);
#pragma unroll
                for (int mi = 0; mi < RG; ++mi) {
                    float(&c)[4] = acc[mi][nt];
                    mma_bf16(c[0], c[1], c[2], c[3], w[mi][0].x, w[mi][1].x, w[mi][0].y, w[mi][1].y, xv.x, xv.y);
                    mma_bf16(c[0], c[1], c[2], c[3], w[mi][0].z, w[mi][1].z, w[mi][0].w, w[mi][1].w, xv.z, xv.w);
                }
            }
        }
    };

#pragma unroll
    for (int s = 0; s < ST - 1; ++s) {
        if (s < steps) { stage(s, s); }
        cp_commit();
    }
    if (active) {
#pragma unroll
        for (int a = 0; a < AH; ++a) {
            if (a < steps) { load(held[a], a); }
        }
    }
#pragma unroll 1
    for (int step0 = 0; step0 < steps; step0 += AH) {
#pragma unroll
        for (int a = 0; a < AH; ++a) {
            const int step = step0 + a;
            if (step < steps) {
                cp_wait<ST - 2>();
                __syncthreads(); // tile `step` landed; every warp has left tile step - 1, whose slot is refilled
                if (step + ST - 1 < steps) { stage((step + ST - 1) % ST, step + ST - 1); }
                cp_commit();
                if (active) {
                    compute(held[a], step % ST, step);
                    if (step + AH < steps) { load(held[a], step + AH); }
                }
            }
        }
    }
    cp_wait<0>();
    __syncthreads(); // shared memory is reused below

    if constexpr (KS > 1) {
        auto* partial = reinterpret_cast<float*>(shared);
        const auto at = [&](int s, int mi, int nt, int q) {
            return ((((s - 1) * S::kRowWarps + row_warp) * RG + mi) * NT + nt) * 128 + q * 32 + lane;
        };
        if (slice > 0 && active) {
#pragma unroll
            for (int mi = 0; mi < RG; ++mi)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                    for (int q = 0; q < 4; ++q) { partial[at(slice, mi, nt, q)] = acc[mi][nt][q]; }
        }
        __syncthreads();
        if (slice == 0 && active) {
#pragma unroll 1
            for (int s = 1; s < KS; ++s) {
#pragma unroll
                for (int mi = 0; mi < RG; ++mi)
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                        for (int q = 0; q < 4; ++q) { acc[mi][nt][q] += partial[at(s, mi, nt, q)]; }
            }
        }
        __syncthreads(); // the partial sums are read before the output tile reuses them
    }
    if (slice != 0 || !active) { return; }

    if constexpr (std::is_same_v<Output, LinearBf16Output>) {
        // The warp's [token][row] BF16 tile in shared memory, then 16-byte row runs of each token.
        auto* tile = reinterpret_cast<__nv_bfloat16*>(shared) + row_warp * (S::kWarpRows * S::kTokens);
#pragma unroll
        for (int mi = 0; mi < RG; ++mi) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int r = mi * 16 + g, c = nt * 8 + 2 * t;
                tile[c * S::kWarpRows + r]           = __float2bfloat16_rn(acc[mi][nt][0]);
                tile[(c + 1) * S::kWarpRows + r]     = __float2bfloat16_rn(acc[mi][nt][1]);
                tile[c * S::kWarpRows + r + 8]       = __float2bfloat16_rn(acc[mi][nt][2]);
                tile[(c + 1) * S::kWarpRows + r + 8] = __float2bfloat16_rn(acc[mi][nt][3]);
            }
        }
        __syncwarp();
        constexpr int kRuns = S::kWarpRows / 8; // 16-byte runs a token
#pragma unroll 1
        for (int i = lane; i < S::kTokens * kRuns; i += 32) {
            const int c = i / kRuns, run = i - c * kRuns;
            if (n0 + c < p.tokens) {
                output.store_vector(warp_row + run * 8, n0 + c,
                                    *reinterpret_cast<const uint4*>(tile + c * S::kWarpRows + run * 8));
            }
        }
    } else {
#pragma unroll
        for (int mi = 0; mi < RG; ++mi) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int row = warp_row + mi * 16 + g, token = n0 + nt * 8 + 2 * t;
                if (token < p.tokens) {
                    output.store(row, token, acc[mi][nt][0]);
                    output.store(row + 8, token, acc[mi][nt][2]);
                }
                if (token + 1 < p.tokens) {
                    output.store(row, token + 1, acc[mi][nt][1]);
                    output.store(row + 8, token + 1, acc[mi][nt][3]);
                }
            }
        }
    }
}

// Whether `S` takes a [rows, k] packed weight: whole row groups and whole steps.
template <class S>
[[nodiscard]] constexpr bool p12_stream_fits(std::int32_t rows, std::int32_t k) noexcept {
    return rows % 16 == 0 && k % S::kStepK == 0;
}

template <class S>
void launch_bf16_p12_stream(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (!p12_stream_fits<S>(w.n, w.k)) {
        throw std::invalid_argument("bf16_p12 stream MMA: rows must be whole row groups and K whole steps");
    }
    const P12StreamOperands p{static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
                              static_cast<const std::uint8_t*>(w.qhigh),  static_cast<const unsigned*>(w.scales),
                              static_cast<const unsigned*>(w.escapes),    w.n,
                              w.k,                                        x.ne[1]};
    const std::int64_t blocks = static_cast<std::int64_t>(div_up(w.n, S::kBlockRows)) * div_up(x.ne[1], S::kTokens);
    const dim3 grid(static_cast<unsigned>(blocks));
    with_linear_output(out, w.n, [&](auto output) {
        constexpr auto kernel = bf16_p12_stream_kernel<S, decltype(output)>;
        const int bytes       = bf16_prepare_shared<S::kSharedBytes, kernel>();
        kernel<<<grid, S::kThreads, bytes, stream>>>(p, output);
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace infernix::ops::detail
