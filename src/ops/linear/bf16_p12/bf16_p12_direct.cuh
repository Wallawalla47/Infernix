#pragma once

// The barrier-free tensor-core route of bf16_p12 calls of up to 8 * Tiles columns: every warp streams its rows'
// packed chunks into registers `Ahead` steps ahead of its MMAs and takes the activations' B fragments straight
// from global memory (they stay in L1: a call's activations are at most Tiles * 8 * K * 2 bytes), so no barrier
// paces the stream. A CTA's KSlices warps of a row group split K into contiguous ranges, added in slice order
// through shared memory once at the end.
//
// Fragments. Per 128-weight step a lane (g = lane / 4, t = lane % 4) loads, for rows g and g + 8 of each of its
// row groups, the 16 code bytes and 8 nibble bytes of chunks 2t, 2t + 1 and of chunks 8 + 2t, 9 + 2t (four loads a
// row, each instruction a whole 32-byte sector of the row across the quad). Sub-step j decodes chunk
// c_j = {2t, 2t + 1, 8 + 2t, 9 + 2t}[j] of both rows and feeds its weights 0-3 and 4-7 to two m16n8k16 MMAs where
// the A fragment layout has k 2t, 2t + 1, 2t + 8, 2t + 9; the B fragments are the same real k of token g, one
// 16-byte load for both. A sub-step's two MMAs sum the 8 * 4 = 32 weights of chunks {c_j over the quad}.
//
// Arithmetic. An output (row, column) is its K slices' FP32 MMA accumulations over their steps in order (sub-steps
// j = 0 .. 3, each its two MMAs in order), then the slices added in order 0, 1, ...: it depends on the row's
// weights, the column's activations and the schedule's K slices only, so the route is column-invariant for a fixed
// slice count at every width it takes.

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

// RowGroups: m16 row groups a warp computes; Tiles: n8 token tiles (a call's columns, at most 8 * Tiles);
// RowWarps: warps along the rows of a CTA; KSlices: warps that split K (contiguous ranges of steps); Ahead: steps a
// lane holds in registers ahead of its MMAs.
template <int RowGroups, int Tiles, int RowWarps, int KSlices, int Ahead, int MinBlocks = 1>
struct P12DirectSchedule {
    static constexpr int kRowGroups   = RowGroups;
    static constexpr int kTiles       = Tiles;
    static constexpr int kRowWarps    = RowWarps;
    static constexpr int kKSlices     = KSlices;
    static constexpr int kAhead       = Ahead;
    static constexpr int kMinBlocks   = MinBlocks;
    static constexpr int kWarpRows    = 16 * RowGroups;
    static constexpr int kBlockRows   = kWarpRows * RowWarps;
    static constexpr int kMaxTokens   = 8 * Tiles;
    static constexpr int kThreads     = 32 * RowWarps * KSlices;
    static constexpr int kSharedBytes = (KSlices - 1) * RowWarps * RowGroups * Tiles * 128 * 4;
    static_assert(RowGroups >= 1 && Tiles >= 1 && RowWarps >= 1 && KSlices >= 1 && Ahead >= 1);
    static_assert(kThreads <= 1024 && kSharedBytes <= 48 * 1024);
};

struct P12DirectOperands {
    const __nv_bfloat16* x;
    const std::uint8_t* codes;
    const std::uint8_t* nibbles;
    const unsigned* words;
    const unsigned* escapes;
    int rows, k, tokens;
};

// A row's four chunks of a step: codes of chunks 2t, 2t + 1 and 8 + 2t, 9 + 2t, and their nibble words.
struct P12DirectChunks {
    uint4 lo, hi;
    uint2 nlo, nhi;
};

template <class S, class Output>
__global__ void __launch_bounds__(S::kThreads, S::kMinBlocks)
    bf16_p12_direct_kernel(P12DirectOperands p, Output output) {
    constexpr int RG = S::kRowGroups, NT = S::kTiles, KS = S::kKSlices, AH = S::kAhead;
    __shared__ float partial[S::kSharedBytes > 0 ? S::kSharedBytes / 4 : 1];
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int slice = warp / S::kRowWarps, row_warp = warp % S::kRowWarps;
    const int g = lane >> 2, t = lane & 3;
    const int warp_row = static_cast<int>(blockIdx.x) * S::kBlockRows + row_warp * S::kWarpRows;
    const bool active  = warp_row < p.rows; // rows are whole row groups: a warp is in or out
    const int K = p.k, steps = K / 128;
    const int first = slice * steps / KS, last = (slice + 1) * steps / KS;

    P12Row rows[RG][2];
    const std::uint8_t* code_rows[RG][2];
    const std::uint8_t* nibble_rows[RG][2];
#pragma unroll
    for (int mi = 0; mi < RG; ++mi) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const std::int64_t row = min(warp_row + mi * 16 + g + 8 * h, p.rows - 1);
            rows[mi][h]            = p12_row(p.words, row);
            code_rows[mi][h]       = p.codes + row * K + 16 * t;
            nibble_rows[mi][h]     = p.nibbles + row * (K / 2) + 8 * t;
        }
    }
    // The activation rows of each tile's token g (columns past the call read as zero).
    const __nv_bfloat16* x_rows[NT];
    bool x_valid[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int token = nt * 8 + g;
        x_valid[nt]     = token < p.tokens;
        x_rows[nt]      = p.x + static_cast<std::int64_t>(x_valid[nt] ? token : 0) * K;
    }

    P12DirectChunks held[AH][RG][2];
    const auto load = [&](P12DirectChunks (&into)[RG][2], int step) {
#pragma unroll
        for (int mi = 0; mi < RG; ++mi) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const std::uint8_t* c = code_rows[mi][h] + step * 128;
                const std::uint8_t* n = nibble_rows[mi][h] + step * 64;
                into[mi][h].lo        = __ldcs(reinterpret_cast<const uint4*>(c));
                into[mi][h].hi        = __ldcs(reinterpret_cast<const uint4*>(c + 64));
                into[mi][h].nlo       = __ldcs(reinterpret_cast<const uint2*>(n));
                into[mi][h].nhi       = __ldcs(reinterpret_cast<const uint2*>(n + 32));
            }
        }
    };

    float acc[RG][NT][4] = {};
    const auto compute = [&](const P12DirectChunks (&chunks)[RG][2], int step) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int chunk = step * 16 + (j < 2 ? 2 * t + j : 6 + 2 * t + j); // {2t, 2t + 1, 8 + 2t, 9 + 2t}
            uint4 w[RG][2];
#pragma unroll
            for (int mi = 0; mi < RG; ++mi) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const P12DirectChunks& c = chunks[mi][h];
                    const uint2 code         = j == 0 ? make_uint2(c.lo.x, c.lo.y)
                                               : j == 1 ? make_uint2(c.lo.z, c.lo.w)
                                               : j == 2 ? make_uint2(c.hi.x, c.hi.y)
                                                        : make_uint2(c.hi.z, c.hi.w);
                    const unsigned nib = j == 0 ? c.nlo.x : j == 1 ? c.nlo.y : j == 2 ? c.nhi.x : c.nhi.y;
                    w[mi][h]           = p12_decode_chunk(code, nib, rows[mi][h], chunk, p.escapes);
                }
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                uint4 xv = make_uint4(0, 0, 0, 0);
                if (x_valid[nt]) { xv = __ldg(reinterpret_cast<const uint4*>(x_rows[nt] + chunk * 8)); }
#pragma unroll
                for (int mi = 0; mi < RG; ++mi) {
                    float(&c)[4] = acc[mi][nt];
                    mma_bf16(c[0], c[1], c[2], c[3], w[mi][0].x, w[mi][1].x, w[mi][0].y, w[mi][1].y, xv.x, xv.y);
                    mma_bf16(c[0], c[1], c[2], c[3], w[mi][0].z, w[mi][1].z, w[mi][0].w, w[mi][1].w, xv.z, xv.w);
                }
            }
        }
    };

    if (active) {
#pragma unroll
        for (int a = 0; a < AH; ++a) {
            if (first + a < last) { load(held[a], first + a); }
        }
#pragma unroll 1
        for (int step0 = first; step0 < last; step0 += AH) {
#pragma unroll
            for (int a = 0; a < AH; ++a) {
                const int step = step0 + a;
                if (step < last) {
                    compute(held[a], step);
                    if (step + AH < last) { load(held[a], step + AH); }
                }
            }
        }
    }

    if constexpr (KS > 1) {
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
        if (slice != 0) { return; }
        if (active) {
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
    }
    if (!active) { return; }
#pragma unroll
    for (int mi = 0; mi < RG; ++mi) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int row = warp_row + mi * 16 + g, token = nt * 8 + 2 * t;
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

// Whether `S` takes a [rows, k] packed weight at `tokens` columns: whole row groups and 128-weight steps.
template <class S>
[[nodiscard]] constexpr bool p12_direct_fits(std::int32_t rows, std::int32_t k, std::int32_t tokens) noexcept {
    // A warp stores all kWarpRows rows of its row groups, unchecked (only its loads are clamped).
    return rows % S::kWarpRows == 0 && k % 128 == 0 && k / 128 >= S::kKSlices && tokens >= 1 && tokens <= S::kMaxTokens;
}

template <class S, class Output>
void launch_bf16_p12_direct_output(const Tensor& x, const Weight& w, Output output, cudaStream_t stream) {
    if (!p12_direct_fits<S>(w.n, w.k, x.ne[1])) {
        throw std::invalid_argument("bf16_p12 direct MMA: whole row groups, K whole steps, at most 8 * Tiles columns");
    }
    const P12DirectOperands p{static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
                              static_cast<const std::uint8_t*>(w.qhigh),  static_cast<const unsigned*>(w.scales),
                              static_cast<const unsigned*>(w.escapes),    w.n,
                              w.k,                                        x.ne[1]};
    bf16_p12_direct_kernel<S, Output><<<div_up(w.n, S::kBlockRows), S::kThreads, 0, stream>>>(p, output);
    CUDA_CHECK(cudaGetLastError());
}

template <class S>
void launch_bf16_p12_direct(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    with_linear_output(out, w.n, [&](auto output) { launch_bf16_p12_direct_output<S>(x, w, output, stream); });
}

} // namespace infernix::ops::detail
