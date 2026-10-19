#pragma once

// The ggml-bf16-prefill tiles (docs/maintainer/qwen3_8-flash-next-design.md §19.3.29): BF16 tensor-core
// tiles over decoded GGUF planes (ggml_tile.cuh's CodeStage images, the ones the exact route's tiles
// read). ops::ggml_linear runs them for wide dense projections (LinearRoutes::GgmlBf16Wide) and the
// GGUF expert banks for wide jobs (offloaded_moe::WideA16::kGgmlBf16, ggml_expert.cuh).
//
// Each code is converted exactly to BF16 (|code| <= 127) and multiplied by the BF16 operand as
// stored on m16n8k16 MMAs with FP32 accumulation, so every product is exact and only sums round. Per
// 32-element slice two MMAs give the sums over elements 0-15 and 16-31 (one chained sum where one
// sub-scale covers the slice); the slice adds d * sub * sum (d * sub is exact in binary32) and, for
// the K-quants, - dmin * min * sum(x) (a per-slice column sum the Columns policy supplies) to each output's FP32
// accumulator, slice by slice in K order. An output's bits depend only on its row, its column and K,
// not on how many columns share the call; they are not the exact route's (no A16 encoding, FP32 sums).
//
// The plane stores a thread's codes in m16n8k32 A-fragment order (row g's elements 4t .. 4t + 3 and
// 16 + 4t .. 16 + 4t + 3). An MMA's sum over K does not depend on how K is labelled as long as A and
// B agree, so the m16n8k16 fragments take those elements as their K slots: A's registers 0/2 (K 2t,
// 2t + 1 and 2t + 8, 2t + 9 of row g) carry elements 4t, 4t + 1 and 4t + 2, 4t + 3, and B's two
// registers x[4t .. 4t + 3] of column g. The B operand is staged in shared memory by 8-byte units
// (four elements), unit u of a column's slice at position 2 (u % 4) + u / 4, so one 16-byte load gives
// a thread both halves' B registers.
//
// A CTA computes kRows = 128 rows (two plane tiles, `images`) x kColumns = 128 columns with 8 warps of
// 64 x 32 (four m16 by four n8 tiles; warp row wr reads image wr), stages of 64 elements of K in a
// ring of kStages by cp.async. Column c's K elements come from a Columns policy (contiguous columns
// of a dense call, or an expert's gathered token rows).

#include "core/ggml_format.h"
#include "ops/common/memory.cuh"
#include "ops/ggml/ggml_dp4a.cuh"
#include "ops/ggml/ggml_tile.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ggml::bf16 {

constexpr int kRows        = 2 * tile::kTileRows; // two plane tiles
constexpr int kColumns     = 128;
constexpr int kThreads     = 256;                 // 8 warps: 2 (rows) x 4 (columns)
constexpr int kStages      = 3;
constexpr int kSlices      = tile::kStageSlices;  // 32-element slices per stage
constexpr int kWarpColumns = 32;
constexpr int kMTiles      = tile::kTileRows / 16; // a warp's m16 tiles: its plane tile's rows
constexpr int kNTiles      = kWarpColumns / 8;
static_assert(kThreads / 32 == (kRows / tile::kTileRows) * (kColumns / kWarpColumns));

// How a format's slice is scaled: one sub-scale per 32 elements (kPlain), one per 16 (kHalf: Q6_K,
// IQ2_S), none (kUnit: IQ4_NL, Q2_0), or one per 32 with the K-quants' min term (kMin).
enum class Scaling : int { kPlain, kHalf, kUnit, kMin };

template <QType F>
inline constexpr Scaling kScaling = tile::kHalfSub<F> ? Scaling::kHalf
                                    : fast::kMin<F>   ? Scaling::kMin
                                    : tile::kUnitSub<F> ? Scaling::kUnit
                                                        : Scaling::kPlain;

// Bytes of a stage image the tiles copy (the min terms only where the scaling reads them).
template <Scaling S>
inline constexpr int kImageBytes = S == Scaling::kMin ? static_cast<int>(sizeof(tile::CodeStage))
                                                      : static_cast<int>(offsetof(tile::CodeStage, dmin));

struct Shared {
    alignas(16) std::uint8_t image[kStages][2][sizeof(tile::CodeStage)];
    alignas(16) std::uint16_t x[kStages][kSlices][kColumns][32]; // BF16 bits, 8-byte units permuted
    float xsum[kStages][kSlices][kColumns];                      // kMin only
};

// The accumulators of a CTA's thread: its warp's four m16 by four n8 tiles.
struct Accumulators {
    float r[kMTiles][kNTiles][4];
};

// Output row (within the CTA's 128) and column (within its 128) of accumulator (mi, nt, q).
__device__ __forceinline__ int out_row(int warp, int lane, int mi, int q) {
    return (warp / (kColumns / kWarpColumns)) * tile::kTileRows + 16 * mi + lane / 4 + 8 * (q >> 1);
}
__device__ __forceinline__ int out_column(int warp, int lane, int nt, int q) {
    return (warp % (kColumns / kWarpColumns)) * kWarpColumns + 8 * nt + 2 * (lane % 4) + (q & 1);
}

// Two signed bytes (the low 16 bits of `bytes`) as an exact BF16 pair (low byte in the low half):
// 128 + (b & 127) less 128 or 256 by the byte's sign.
__device__ __forceinline__ unsigned bf16_pair(unsigned bytes) {
    union Bits {
        __nv_bfloat162 pair;
        unsigned bits;
    };
    Bits biased, bias, out;
    biased.bits = __byte_perm(bytes, 0x43004300U, 0x7150) & 0xFF7FFF7FU;
    bias.bits   = 0x43004300U | (bytes & 0x80U) | ((bytes & 0x8000U) << 8);
    out.pair    = __hsub2_rn(biased.pair, bias.pair);
    return out.bits;
}

__device__ __forceinline__ void mma(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float f16_value(std::uint16_t bits) { return __half2float(__ushort_as_half(bits)); }

// The B operand of a dense call: columns col0 .. of x (BF16 [k, columns]), input element e at
// x[perm(e)], and their slice sums (xsum [columns, k / 32]; kMin only).
struct DenseColumns {
    const std::uint16_t* x;
    int k;
    const std::int32_t* perm;
    int perm_block;
    const float* xsum;
    int col0, columns;

    // Column c's elements (null past the call's columns: zero-filled).
    __device__ __forceinline__ const std::uint16_t* column(int c) const {
        return col0 + c < columns ? x + static_cast<std::size_t>(col0 + c) * k : nullptr;
    }
    __device__ __forceinline__ int element(int e) const {
        return perm == nullptr ? e : perm[e / perm_block] * perm_block + e % perm_block;
    }
    __device__ __forceinline__ const float* slice_sum(int c, int s) const {
        return col0 + c < columns ? xsum + static_cast<std::size_t>(col0 + c) * (k / 32) + s : nullptr;
    }
};

// The CTA's K loop: `images` the two plane tiles' stage images (stage st at images[h] + st *
// sizeof(CodeStage)), `stages` stages of kSlices slices, B columns from `cols`. Every thread calls it;
// it leaves the thread's accumulators and returns with every copy landed and every thread past its
// last read of shared memory.
template <Scaling S, class Columns>
__device__ __forceinline__ void compute(Shared& sm, const std::uint8_t* const (&images)[2], int stages, const Columns& cols,
                                        Accumulators& acc) {
    constexpr int kImage          = kImageBytes<S>;
    constexpr std::size_t kStride = sizeof(tile::CodeStage);
    // Stage st's copies into ring slot `buf`: both row tiles' images, the columns' units (zero past the
    // columns) and, for kMin, their slice sums.
    const auto issue = [&](int st, int buf) {
        for (int i = static_cast<int>(threadIdx.x); i < 2 * (kImage / 16); i += kThreads) {
            const int h = i / (kImage / 16), c = i % (kImage / 16);
            tile::cp_async16(&sm.image[buf][h][16 * c], images[h] + st * kStride + 16 * c, 16);
        }
        for (int i = static_cast<int>(threadIdx.x); i < kSlices * kColumns * 8; i += kThreads) {
            const int j = i / (kColumns * 8), c = (i / 8) % kColumns, u = i % 8;
            const std::uint16_t* column = cols.column(c);
            const std::uint16_t* from   = column != nullptr ? column + cols.element((st * kSlices + j) * 32 + 4 * u)
                                                            : cols.column(0);
            ops::cp_async_zfill<8>(&sm.x[buf][j][c][4 * (2 * (u % 4) + u / 4)], from, column != nullptr ? 8 : 0);
        }
        if constexpr (S == Scaling::kMin) {
            for (int i = static_cast<int>(threadIdx.x); i < kSlices * kColumns; i += kThreads) {
                const int j = i / kColumns, c = i % kColumns;
                const float* from = cols.slice_sum(c, st * kSlices + j);
                ops::cp_async_zfill<4>(&sm.xsum[buf][j][c], from != nullptr ? from : cols.slice_sum(0, 0), from != nullptr ? 4 : 0);
            }
        }
    };

    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int g = lane / 4, t = lane % 4;
    const int wr = warp / (kColumns / kWarpColumns), wc = warp % (kColumns / kWarpColumns);
#pragma unroll
    for (int mi = 0; mi < kMTiles; ++mi) {
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) { acc.r[mi][nt][q] = 0.0F; }
        }
    }
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < stages) { issue(s, s); }
        ops::cp_commit();
    }
    for (int st = 0; st < stages; ++st) {
        ops::cp_wait<kStages - 2>(); // stage st has landed
        __syncthreads();             // and every thread finished stage st - 1, whose slot the next issue fills
        if (st + kStages - 1 < stages) { issue(st + kStages - 1, (st + kStages - 1) % kStages); }
        ops::cp_commit();
        const int buf = st % kStages;
        const auto& cs = *reinterpret_cast<const tile::CodeStage*>(sm.image[buf][wr]);
#pragma unroll
        for (int j = 0; j < kSlices; ++j) {
            uint4 b[kNTiles];
#pragma unroll
            for (int nt = 0; nt < kNTiles; ++nt) {
                b[nt] = *reinterpret_cast<const uint4*>(&sm.x[buf][j][wc * kWarpColumns + 8 * nt + g][8 * t]);
            }
            float xs[kNTiles][2] = {};
            if constexpr (S == Scaling::kMin) {
#pragma unroll
                for (int nt = 0; nt < kNTiles; ++nt) {
                    xs[nt][0] = sm.xsum[buf][j][wc * kWarpColumns + 8 * nt + 2 * t];
                    xs[nt][1] = sm.xsum[buf][j][wc * kWarpColumns + 8 * nt + 2 * t + 1];
                }
            }
#pragma unroll
            for (int mi = 0; mi < kMTiles; ++mi) {
                const int4 a = *reinterpret_cast<const int4*>(cs.a[j][mi][g][t]);
                // Elements 0-15 (lo) and 16-31 (hi): rows g, g + 8, then their elements 4t + 2, + 3.
                const unsigned lo[4] = {bf16_pair(static_cast<unsigned>(a.x)), bf16_pair(static_cast<unsigned>(a.y)),
                                        bf16_pair(static_cast<unsigned>(a.x) >> 16), bf16_pair(static_cast<unsigned>(a.y) >> 16)};
                const unsigned hi[4] = {bf16_pair(static_cast<unsigned>(a.z)), bf16_pair(static_cast<unsigned>(a.w)),
                                        bf16_pair(static_cast<unsigned>(a.z) >> 16), bf16_pair(static_cast<unsigned>(a.w) >> 16)};
                const int r = 16 * mi + g; // rows r and r + 8 of the warp's plane tile
                const float d0 = f16_value(cs.d[j][r]), d1 = f16_value(cs.d[j][r + 8]);
                // d * sub per row (0: r, 1: r + 8) and half: exact (an 11-bit mantissa by |sub| <= 127).
                float s00 = d0, s01 = d0, s10 = d1, s11 = d1;
                if constexpr (S != Scaling::kUnit) {
                    const int4 sc = *reinterpret_cast<const int4*>(cs.sub[j][mi][g]);
                    s00 = d0 * static_cast<float>(sc.x);
                    s01 = d0 * static_cast<float>(sc.y);
                    s10 = d1 * static_cast<float>(sc.z);
                    s11 = d1 * static_cast<float>(sc.w);
                }
                float m0 = 0.0F, m1 = 0.0F;
                if constexpr (S == Scaling::kMin) {
                    m0 = f16_value(cs.dmin[j][r]) * static_cast<float>(cs.min[j][r]);
                    m1 = f16_value(cs.dmin[j][r + 8]) * static_cast<float>(cs.min[j][r + 8]);
                }
#pragma unroll
                for (int nt = 0; nt < kNTiles; ++nt) {
                    float(&o)[4] = acc.r[mi][nt];
                    if constexpr (S == Scaling::kHalf) {
                        float p[4] = {0.0F, 0.0F, 0.0F, 0.0F}, q[4] = {0.0F, 0.0F, 0.0F, 0.0F};
                        mma(p, lo, b[nt].x, b[nt].y);
                        mma(q, hi, b[nt].z, b[nt].w);
                        o[0] = fmaf(s01, q[0], fmaf(s00, p[0], o[0]));
                        o[1] = fmaf(s01, q[1], fmaf(s00, p[1], o[1]));
                        o[2] = fmaf(s11, q[2], fmaf(s10, p[2], o[2]));
                        o[3] = fmaf(s11, q[3], fmaf(s10, p[3], o[3]));
                    } else {
                        float p[4] = {0.0F, 0.0F, 0.0F, 0.0F};
                        mma(p, lo, b[nt].x, b[nt].y);
                        mma(p, hi, b[nt].z, b[nt].w);
                        o[0] = fmaf(s00, p[0], o[0]);
                        o[1] = fmaf(s00, p[1], o[1]);
                        o[2] = fmaf(s10, p[2], o[2]);
                        o[3] = fmaf(s10, p[3], o[3]);
                    }
                    if constexpr (S == Scaling::kMin) {
                        o[0] = fmaf(-m0, xs[nt][0], o[0]);
                        o[1] = fmaf(-m0, xs[nt][1], o[1]);
                        o[2] = fmaf(-m1, xs[nt][0], o[2]);
                        o[3] = fmaf(-m1, xs[nt][1], o[3]);
                    }
                }
            }
        }
    }
    ops::cp_wait<0>();
    __syncthreads(); // the caller may reuse shared memory
}

// A dense call's tile: rows [kRows bx, + kRows) of a segment (or slab) whose plane holds `tiles` row
// tiles (the last image pair clamped to the last tile; those rows' outputs are not stored) against
// columns [kColumns by, + kColumns). out[t * n + out_offset + row] for row < rows, t < columns.
template <Scaling S, bool OutF32>
__global__ void __launch_bounds__(kThreads, 1)
    tile_kernel(const std::uint8_t* __restrict__ plane, int tiles, int rows, int slices, DenseColumns cols,
                void* __restrict__ out, int n, int out_offset, int kappa_log2) {
    extern __shared__ __align__(16) unsigned char bf16_shared[];
    auto& sm = *reinterpret_cast<Shared*>(bf16_shared);
    const int stages = slices / kSlices;
    const int tile0  = static_cast<int>(blockIdx.x) * 2;
    cols.col0        = static_cast<int>(blockIdx.y) * kColumns;
    constexpr std::size_t kStride = sizeof(tile::CodeStage);
    const std::uint8_t* const images[2] = {plane + static_cast<std::size_t>(min(tile0, tiles - 1)) * stages * kStride,
                                           plane + static_cast<std::size_t>(min(tile0 + 1, tiles - 1)) * stages * kStride};
    Accumulators acc;
    compute<S>(sm, images, stages, cols, acc);
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
#pragma unroll
    for (int mi = 0; mi < kMTiles; ++mi) {
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int row = static_cast<int>(blockIdx.x) * kRows + out_row(warp, lane, mi, q);
                const int col = cols.col0 + out_column(warp, lane, nt, q);
                if (row < rows && col < cols.columns) {
                    const float v        = ldexpf(acc.r[mi][nt][q], kappa_log2);
                    const std::size_t at = static_cast<std::size_t>(col) * n + out_offset + row;
                    if constexpr (OutF32) {
                        static_cast<float*>(out)[at] = v;
                    } else {
                        static_cast<__nv_bfloat16*>(out)[at] = __float2bfloat16_rn(v);
                    }
                }
            }
        }
    }
}

} // namespace infernix::ggml::bf16
