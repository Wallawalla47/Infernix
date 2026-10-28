#pragma once

// The wide (prefill-width) tensor-core tiles of the 8-bit activation route over GGUF weights
// (`--a8-activations`: LOSSY, opt-in; docs/maintainer/qwen3_8-flash-next-design.md §19.3.31): ops::ggml_linear's
// calls of 32 or more columns under LinearRoutes::GgmlA8Wide (compute_plane, over decoded weight planes of
// PlaneStage images) and the GGUF expert banks' wide jobs under WideA16::kA8 (compute_decode, which decodes each
// K stage of the raw rows in shared memory). The activations are ops/ggml/ggml_a8.h's codec, the narrow routes'
// codes and slice scales: every column quantized per 32-element slice s to int8 codes q with an FP32 scale d_s.
//
// ---- Arithmetic of one output (row r, column t)
//
// P_s is the exact int32 dot of the row's codes of slice s with the column's codes (|P_s| < 2^19, every code
// magnitude <= 127); w_s = d * sub, the slice's binary16 block scale times its integer sub-scale (at most 8
// bits; 1 for the formats without sub-scales), exact in binary32:
//
//   acc = +0; for s = 0 .. K/32 - 1, in order:
//     one sub-scale per slice:            acc = fma(fl32(P_s * w_s), d_s, acc)
//     Q6_K and IQ2_S (one per 16 values): acc = fma(fl32(P_s0 * w_s0), d_s, acc), then the same for half 1
//     Q4_K and Q5_K, after the main term: acc = fma(-(dmin * min_s), fl32(d_s * sum_j q_j), acc)
//   y = fl32(acc * kappa) (ggml::kappa_log2: an exact power of two), rounded once more to BF16 for a BF16 output
//
// fl32(P * w) comes from the int32 MMA without a conversion instruction (ggml_a8_mma.cuh). One thread adds an
// output's terms in K order from +0, so an output depends only on its row's weights, its own column and K: not
// on the call's width or other columns, the tile, the plane slab or the device; a column's bits are the same at
// every width of the wide routes. They are not the narrow routes' bits (ggml_linear_a8.cuh, ggml_expert_a8.cuh:
// fl32(isum * d) then fma(., w, acc), summed in a split order). A non-finite weight scale makes its rows NaN.
//
// ---- Operand layout (both K loops)
//
// The MMA is m16n8k32 s8 with a permuted K labelling: lane (g, t) holds chunk t (elements 8t .. 8t + 7, code
// words 2t and 2t + 1) of rows g and g + 8 as its A fragment (int4: row g's word 2t, row g + 8's, row g's word
// 2t + 1, row g + 8's), and column g's code words 2t and 2t + 1 as B (one 8-byte load of the column's slice in
// its natural word order). A k32 MMA's sum does not depend on the K labels when A and B agree. A format whose
// sub-scale covers 16 elements runs two MMAs per slice, lanes t < 2 (elements 0-15) in the first and t >= 2 in
// the second, the other lanes' A zero.
//
// ---- compute_plane (ops::ggml_linear, GgmlA8Wide)
//
// A plane holds, per tile of kPlaneTileRows rows and K stage of kStageSlices slices, one PlaneStage image: the
// codes in A-fragment order, w per row and half, and the K-quants' -(dmin * min) per row; a8_plane decodes it
// once per call (fast::chunk). A CTA computes kPlaneRows rows (two plane tiles) x kColumns columns with 8 warps
// of 64 x 32 (four m16 by four n8 tiles); stages arrive by cp.async in a ring of kPlaneStages: both tiles'
// images, the columns' codes, their slice scales and (K-quants) fl32(d * sum q) from the encoded columns
// (ggml_linear_a8_wide.cu). The tiles' grid walks the column tiles fastest, so a wave reads each plane tile once
// from DRAM while the encoded columns stay in L2.
//
// ---- compute_decode (GGUF expert banks, WideA16::kA8)
//
// ggml_bf16_tile.cuh's compute_decode with int8 operands: the CTA copies its kDecodeRows rows' raw GGUF bytes by
// cp.async, 256 elements of K per row at a time (one unit ahead, double-buffered), decodes each K stage into
// shared memory (tile::tile_chunk: codes, then w per row and half) before its MMAs, and stages the columns' codes
// and slice scales one stage ahead. Lane (g, t) of a decode task writes chunk t of rows g and g + 8 as one int4,
// the exact A fragment, so the stores are conflict-free.

#include "core/ggml_format.h"
#include "ops/common/memory.cuh"
#include "ops/ggml/ggml_a8.h"
#include "ops/ggml/ggml_a8_mma.cuh"
#include "ops/ggml/ggml_dp4a.cuh"
#include "ops/ggml/ggml_tile.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ggml::a8::wide {

// How a format's slice is scaled: one sub-scale per 32 elements (or none), one per 16 (Q6_K, IQ2_S), or one per
// 32 with the K-quants' min term.
enum class Scaling : int { kPlain, kHalf, kMin };

template <QType F>
inline constexpr Scaling kScaling = tile::kHalfSub<F> ? Scaling::kHalf : fast::kMin<F> ? Scaling::kMin : Scaling::kPlain;

inline constexpr int kStageSlices = 2;   // 64 elements of K per stage
inline constexpr int kColumns     = 128; // columns of a CTA
inline constexpr int kThreads     = 256; // 8 warps: 2 (rows) x 4 (columns)
inline constexpr int kWarps       = kThreads / 32;
inline constexpr int kWarpColumns = 32;  // a warp's columns at four n8 tiles
inline constexpr int kMTiles      = 4;   // a warp's m16 tiles: 64 rows
inline constexpr int kNTiles      = 4;
static_assert(kWarps == 2 * (kColumns / kWarpColumns));

// A binary16 weight scale in binary32 (exact).
__device__ __forceinline__ float weight_scale(std::uint16_t h) { return __half2float(__ushort_as_half(h)); }

// ------------------------------------------------------------------------------------------- one slice step

// One m16 tile's operands at one slice: rows g and g + 8 of the tile (lane (g, t)).
struct RowOperands {
    int4 lo;   // A (kHalf: elements 0-15, lanes t >= 2 zero)
    int4 hi;   // kHalf: elements 16-31 (lanes t < 2 zero)
    float4 w;  // w of row g's halves 0 and 1, then row g + 8's (equal halves without a 16-element sub-scale)
    float4 mw; // -M w (exact)
    float2 mn; // kMin: -(dmin * min) of rows g and g + 8
};

template <Scaling S>
__device__ __forceinline__ RowOperands row_operands(const int4& a, const float4& w, float2 mn, bool low) {
    RowOperands o;
    if constexpr (S == Scaling::kHalf) {
        o.lo = low ? a : make_int4(0, 0, 0, 0);
        o.hi = low ? make_int4(0, 0, 0, 0) : a;
    } else {
        o.lo = a;
        o.hi = make_int4(0, 0, 0, 0);
    }
    o.w  = w;
    o.mw = make_float4(__fmul_rn(kMinusMagic, w.x), __fmul_rn(kMinusMagic, w.y), __fmul_rn(kMinusMagic, w.z),
                       __fmul_rn(kMinusMagic, w.w));
    o.mn = mn;
    return o;
}

// The terms of one slice of an m16 x n8 tile into o (o[q]: row g + 8 (q >> 1), column 2t + (q & 1)): b is column
// g's code words 2t and 2t + 1, x0 and x1 the slice scales of columns 2t and 2t + 1, z0 and z1 (kMin) their
// fl32(d * sum q).
template <Scaling S>
__device__ __forceinline__ void slice_step(float (&o)[4], const RowOperands& r, int2 b, float x0, float x1, float z0,
                                           float z1) {
    int p[4];
    mma_k32_magic(p, r.lo.x, r.lo.y, r.lo.z, r.lo.w, b.x, b.y);
    if constexpr (S == Scaling::kHalf) {
        int q[4];
        mma_k32_magic(q, r.hi.x, r.hi.y, r.hi.z, r.hi.w, b.x, b.y);
        o[0] = __fmaf_rn(scaled_term(p[0], r.w.x, r.mw.x), x0, o[0]);
        o[1] = __fmaf_rn(scaled_term(p[1], r.w.x, r.mw.x), x1, o[1]);
        o[2] = __fmaf_rn(scaled_term(p[2], r.w.z, r.mw.z), x0, o[2]);
        o[3] = __fmaf_rn(scaled_term(p[3], r.w.z, r.mw.z), x1, o[3]);
        o[0] = __fmaf_rn(scaled_term(q[0], r.w.y, r.mw.y), x0, o[0]);
        o[1] = __fmaf_rn(scaled_term(q[1], r.w.y, r.mw.y), x1, o[1]);
        o[2] = __fmaf_rn(scaled_term(q[2], r.w.w, r.mw.w), x0, o[2]);
        o[3] = __fmaf_rn(scaled_term(q[3], r.w.w, r.mw.w), x1, o[3]);
    } else {
        o[0] = __fmaf_rn(scaled_term(p[0], r.w.x, r.mw.x), x0, o[0]);
        o[1] = __fmaf_rn(scaled_term(p[1], r.w.x, r.mw.x), x1, o[1]);
        o[2] = __fmaf_rn(scaled_term(p[2], r.w.z, r.mw.z), x0, o[2]);
        o[3] = __fmaf_rn(scaled_term(p[3], r.w.z, r.mw.z), x1, o[3]);
    }
    if constexpr (S == Scaling::kMin) {
        o[0] = __fmaf_rn(r.mn.x, z0, o[0]);
        o[1] = __fmaf_rn(r.mn.x, z1, o[1]);
        o[2] = __fmaf_rn(r.mn.y, z0, o[2]);
        o[3] = __fmaf_rn(r.mn.y, z1, o[3]);
    }
}

// A thread's outputs: its warp's four m16 by up to four n8 tiles.
struct Accumulators {
    float r[kMTiles][kNTiles][4];
};

__device__ __forceinline__ void zero(Accumulators& acc) {
#pragma unroll
    for (int mi = 0; mi < kMTiles; ++mi) {
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) { acc.r[mi][nt][q] = 0.0F; }
        }
    }
}

// --------------------------------------------------------------------------------------------- the planes

inline constexpr int kPlaneTileRows = 64;
inline constexpr int kPlaneGroups   = kPlaneTileRows / 16;
inline constexpr int kPlaneRows     = 2 * kPlaneTileRows; // a CTA's rows: two plane tiles
inline constexpr int kPlaneStages   = 3;                   // ring slots of compute_plane

// One stage of a plane tile (kPlaneTileRows rows x kStageSlices slices): a[j][m][g][t] is chunk t of rows
// 16 m + g and 16 m + g + 8 of slice j (the A fragment above), w[j][m][g] their w (row 16 m + g's halves 0 and 1,
// then row 16 m + g + 8's), mn[j][m][g] (Q4_K, Q5_K) their -(dmin * min).
struct PlaneStage {
    int4 a[kStageSlices][kPlaneGroups][8][4];
    float4 w[kStageSlices][kPlaneGroups][8];
    float2 mn[kStageSlices][kPlaneGroups][8];
};
static_assert(sizeof(PlaneStage) == 5632 && offsetof(PlaneStage, w) == 4096 && offsetof(PlaneStage, mn) == 5120,
              "stage images copy in 16-byte units");

// Bytes of an image the tiles copy (the min terms only where the scaling reads them).
template <Scaling S>
inline constexpr int kImageBytes = S == Scaling::kMin ? static_cast<int>(sizeof(PlaneStage))
                                                      : static_cast<int>(offsetof(PlaneStage, mn));

// Decodes stage st of the plane tile whose row r is row_of(r) (a valid row: callers clamp past the matrix) into
// `image` (global memory) from the raw GGUF bytes (fast::chunk). Every thread of a CTA of kThreads calls it.
template <QType F, class RowOf>
__device__ __forceinline__ void decode_plane_stage(PlaneStage& image, RowOf row_of, int st) {
    static_assert(kStageSlices * kPlaneGroups * 8 * 4 == kThreads, "one chunk pair per thread");
    const int i = static_cast<int>(threadIdx.x);
    const int j = i >> 7, m = (i >> 5) & 3, g = (i >> 2) & 7, t = i & 3;
    const int s = kStageSlices * st + j, r = 16 * m + g;
    const fast::Chunk c0 = fast::chunk<F>(row_of(r), s, t);
    const fast::Chunk c1 = fast::chunk<F>(row_of(r + 8), s, t);
    image.a[j][m][g][t] = make_int4(c0.w0, c1.w0, c0.w1, c1.w1);
    if ((t & 1) == 0) { // chunk 0 holds half 0's sub-scale, chunk 2 half 1's
        float* w     = reinterpret_cast<float*>(&image.w[j][m][g]);
        w[t / 2]     = __fmul_rn(weight_scale(c0.d), static_cast<float>(c0.sub));
        w[2 + t / 2] = __fmul_rn(weight_scale(c1.d), static_cast<float>(c1.sub));
    }
    if constexpr (fast::kMin<F>) {
        if (t == 0) {
            image.mn[j][m][g] = make_float2(-__fmul_rn(weight_scale(c0.dmin), static_cast<float>(c0.min)),
                                            -__fmul_rn(weight_scale(c1.dmin), static_cast<float>(c1.min)));
        }
    }
}

// Encoded columns of a dense call (ggml_linear_a8_wide.cu): column c of the call at encoded + c * stride words, its
// codes (slice s at words 8 s .. 8 s + 7, element 4w + j in byte j of word w), then the K / 32 slice scales d
// (binary32, at word k / 4), then (K-quants) the K / 32 values fl32(d * sum q) (at word 9 k / 32): ggml_a8.h's
// column layout with the min term's code sums already scaled.
struct DenseColumns {
    const std::int32_t* encoded = nullptr;
    int stride                  = 0; // words between columns
    int k                       = 0;
    int col0                    = 0; // the CTA's first column
    int columns                 = 0; // the batch's columns

    __device__ __forceinline__ const std::int32_t* column(int c) const {
        return col0 + c < columns ? encoded + static_cast<std::size_t>(col0 + c) * stride : nullptr;
    }
};

struct PlaneShared {
    PlaneStage image[kPlaneStages][2];
    alignas(16) std::int32_t x[kPlaneStages][kStageSlices][kColumns][8]; // the columns' codes
    alignas(16) float2 xs[kPlaneStages][kColumns];                        // the stage's two slice scales
    alignas(16) float2 xz[kPlaneStages][kColumns];                        // kMin: fl32(d * sum q) of both
};

// The CTA's K loop over `stages` stages of its two plane tiles (stage st of tile h at images[h] + st *
// sizeof(PlaneStage)) against columns col0 .. col0 + kColumns - 1 of `cols` (zeros past its columns). Every thread
// calls it; it leaves the thread's accumulators (warp (wr, wc): rows 64 wr + 16 mi + g (+ 8), columns 32 wc +
// 8 nt + 2t (+ 1)) and returns with every copy landed and every thread past its last read of shared memory.
template <Scaling S>
__device__ __forceinline__ void compute_plane(PlaneShared& sm, const std::uint8_t* const (&images)[2], int stages,
                                              const DenseColumns& cols, Accumulators& acc) {
    constexpr int kImage       = kImageBytes<S>;
    constexpr int kImageCopies = 2 * kImage / 16;
    constexpr bool kMinTerm    = S == Scaling::kMin;
    const int tid = static_cast<int>(threadIdx.x);
    // A thread's copies, the same every stage: 16 bytes (slice j, half h) of the codes of columns tid / 4 and
    // 64 + tid / 4, and the scales (tid < 128) or the min sums (tid >= 128) of column tid % 128.
    const int xj = (tid >> 1) & 1, xh = tid & 1;
    const std::int32_t* code_col[2] = {cols.column(tid >> 2), cols.column(64 + (tid >> 2))};
    const std::int32_t* scale_col   = cols.column(tid & (kColumns - 1));
    const bool scale_copy           = tid < kColumns || kMinTerm;
    const int scale_words           = tid < kColumns ? cols.k / 4 : cols.k / 32 * 9;
    const std::int32_t* fallback    = cols.encoded; // the source of zero-filled copies
    const auto issue = [&](int st, int buf) {
        for (int i = tid; i < kImageCopies; i += kThreads) {
            const int h = i / (kImage / 16), c = i % (kImage / 16);
            tile::cp_async16(reinterpret_cast<unsigned char*>(&sm.image[buf][h]) + 16 * c,
                             images[h] + static_cast<std::size_t>(st) * sizeof(PlaneStage) + 16 * c, 16);
        }
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int c              = (tid >> 2) + 64 * k;
            const std::int32_t* from = code_col[k] != nullptr ? code_col[k] + (kStageSlices * st + xj) * 8 + 4 * xh : fallback;
            tile::cp_async16(&sm.x[buf][xj][c][4 * xh], from, code_col[k] != nullptr ? 16 : 0);
        }
        if (scale_copy) {
            const int c              = tid & (kColumns - 1);
            const std::int32_t* from = scale_col != nullptr ? scale_col + scale_words + kStageSlices * st : fallback;
            ops::cp_async_zfill<8>(tid < kColumns ? &sm.xs[buf][c] : &sm.xz[buf][c], from, scale_col != nullptr ? 8 : 0);
        }
    };

    const int warp = tid / 32, lane = tid % 32;
    const int g = lane / 4, t = lane % 4;
    const int wr = warp / (kColumns / kWarpColumns), wc = warp % (kColumns / kWarpColumns);
    zero(acc);
#pragma unroll
    for (int s = 0; s < kPlaneStages - 1; ++s) {
        if (s < stages) { issue(s, s); }
        ops::cp_commit();
    }
    for (int st = 0; st < stages; ++st) {
        ops::cp_wait<kPlaneStages - 2>(); // stage st has landed
        __syncthreads();                  // and every thread finished stage st - 1, whose slot the next issue fills
        if (st + kPlaneStages - 1 < stages) { issue(st + kPlaneStages - 1, (st + kPlaneStages - 1) % kPlaneStages); }
        ops::cp_commit();
        const int buf         = st % kPlaneStages;
        const PlaneStage& im  = sm.image[buf][wr];
        // Columns 2t and 2t + 1 of each n8 tile: {d(2t, slice 0), d(2t, slice 1), d(2t + 1, 0), d(2t + 1, 1)}.
        float4 xd[kNTiles];
        [[maybe_unused]] float4 xz[kNTiles];
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
            const int c = wc * kWarpColumns + 8 * nt + 2 * t;
            xd[nt]      = *reinterpret_cast<const float4*>(&sm.xs[buf][c]);
            if constexpr (kMinTerm) { xz[nt] = *reinterpret_cast<const float4*>(&sm.xz[buf][c]); }
        }
#pragma unroll
        for (int j = 0; j < kStageSlices; ++j) {
            int2 b[kNTiles];
#pragma unroll
            for (int nt = 0; nt < kNTiles; ++nt) {
                b[nt] = *reinterpret_cast<const int2*>(&sm.x[buf][j][wc * kWarpColumns + 8 * nt + g][2 * t]);
            }
#pragma unroll
            for (int mi = 0; mi < kMTiles; ++mi) {
                float2 mn = make_float2(0.0F, 0.0F);
                if constexpr (kMinTerm) { mn = im.mn[j][mi][g]; }
                const RowOperands ro = row_operands<S>(im.a[j][mi][g][t], im.w[j][mi][g], mn, t < 2);
#pragma unroll
                for (int nt = 0; nt < kNTiles; ++nt) {
                    const float x0 = j == 0 ? xd[nt].x : xd[nt].y, x1 = j == 0 ? xd[nt].z : xd[nt].w;
                    float z0 = 0.0F, z1 = 0.0F;
                    if constexpr (kMinTerm) {
                        z0 = j == 0 ? xz[nt].x : xz[nt].y;
                        z1 = j == 0 ? xz[nt].z : xz[nt].w;
                    }
                    slice_step<S>(acc.r[mi][nt], ro, b[nt], x0, x1, z0, z1);
                }
            }
        }
    }
    ops::cp_wait<0>();
    __syncthreads(); // the caller may reuse shared memory
}

// ------------------------------------------------------------------------------------------ stage decode

inline constexpr int kDecodeRows   = 128; // a CTA's rows
inline constexpr int kDecodeGroups = kDecodeRows / 16;
inline constexpr int kXStages      = 2;   // the columns' ring slots of compute_decode
static_assert(2 * kWarps == kStageSlices * kDecodeGroups, "two decode tasks per warp cover a stage");

// A raw unit (tile::kUnitSlices slices) of a row of F blocks, and the 16-byte chunks that hold it from its row's
// 16-byte aligned start (up to 15 bytes before it), rounded to an odd count so that the rows a warp decodes
// start in distinct bank quads (ggml_bf16_tile.cuh's DecodeShared).
template <QType F>
inline constexpr int kUnitBytes = tile::kUnitSlices * 32 / static_cast<int>(ggml_block_elems(F)) *
                                  static_cast<int>(ggml_block_bytes(F));
template <QType F>
inline constexpr int kRawChunks = ((15 + kUnitBytes<F> + 15) / 16) | 1;

template <QType F>
struct DecodeShared {
    alignas(16) std::int32_t x[kXStages][kStageSlices][kColumns][8]; // the columns' codes
    alignas(16) float2 xs[kXStages][kColumns];                        // the stage's two slice scales per column
    alignas(16) int4 a[kStageSlices][kDecodeGroups][8][4];            // the stage's codes in A-fragment order
    alignas(16) float4 w[kStageSlices][kDecodeGroups][8];             // w of rows 16 m + g, 16 m + g + 8 by half
    alignas(16) std::uint8_t raw[2][kDecodeRows][kRawChunks<F> * 16];
    int raw_offset[2][kDecodeRows]; // where row r's unit bytes start in raw[.][r]
    fast::SharedCodebook<F> codebook;
};

// Output column (within the CTA's) of accumulator (nt, q) of a tile of NT n8 tiles per warp, and output row.
template <int NT>
__device__ __forceinline__ int decode_out_column(int warp, int lane, int nt, int q) {
    return (warp % (kColumns / kWarpColumns)) * (8 * NT) + 8 * nt + 2 * (lane % 4) + (q & 1);
}
__device__ __forceinline__ int out_row(int warp, int lane, int mi, int q) {
    return (warp / (kColumns / kWarpColumns)) * (kMTiles * 16) + 16 * mi + lane / 4 + 8 * (q >> 1);
}

// The K loop over `stages` stages of rows row_of(0 .. kDecodeRows - 1) (a valid row each; the second warp row
// takes rows 64 ..) against the CTA's columns of `cols`: cols.column(c) is column c's row of codes (slice s at
// byte 32 s) and slice scales (binary32, slice s at byte cols.scales + 4 s), null past the tile's columns, and
// column 0 is valid. NT = 4 (128 columns) or 2 (64; a column's outputs do not depend on its tile's width).
// sm.codebook must hold F's codebook, published by a barrier (the caller loads it once per CTA). Every thread
// calls it; it leaves the thread's accumulators (acc.r[mi][nt], nt < NT) and returns with every copy landed and
// every thread past its last read of shared memory.
template <QType F, int NT, class RowOf, class Cols>
__device__ __forceinline__ void compute_decode(DecodeShared<F>& sm, RowOf row_of, int stages, const Cols& cols,
                                               Accumulators& acc) {
    constexpr Scaling S = kScaling<F>;
    static_assert(S != Scaling::kMin, "the stage-decode tiles serve the expert formats (no min term)");
    static_assert(NT == 4 || NT == 2);
    constexpr int kBlockElems = static_cast<int>(ggml_block_elems(F));
    constexpr int kBlockBytes = static_cast<int>(ggml_block_bytes(F));
    constexpr int kUnitStages = tile::kUnitSlices / kStageSlices;
    constexpr int kCols       = (kColumns / kWarpColumns) * 8 * NT; // the tile's columns
    constexpr int kCodeCopies = kCols * 4 / kThreads;              // 16-byte code copies per thread and stage
    static_assert(kCodeCopies * kThreads == kCols * 4 && kCols <= kThreads);
    const int tid = static_cast<int>(threadIdx.x), warp = tid / 32, lane = tid % 32;
    const int slices = stages * kStageSlices;

    // A thread's column copies, the same every stage: 16 bytes (slice xj, half xh) of the codes of columns
    // tid / 4 + 64 k, and (tid < kCols) the two slice scales of column tid.
    const int xj = (tid >> 1) & 1, xh = tid & 1;
    const std::uint8_t* xcol[kCodeCopies];
#pragma unroll
    for (int k = 0; k < kCodeCopies; ++k) { xcol[k] = cols.column((tid >> 2) + (kThreads / 4) * k); }
    const std::uint8_t* scol     = tid < kCols ? cols.column(tid) : nullptr;
    const std::uint8_t* fallback = cols.column(0);
    const auto issue_x = [&](int st, int buf) {
#pragma unroll
        for (int k = 0; k < kCodeCopies; ++k) {
            const int c              = (tid >> 2) + (kThreads / 4) * k;
            const std::uint8_t* from = xcol[k] != nullptr ? xcol[k] + (kStageSlices * st + xj) * 32 + 16 * xh : fallback;
            tile::cp_async16(&sm.x[buf][xj][c][4 * xh], from, xcol[k] != nullptr ? 16 : 0);
        }
        if (tid < kCols) {
            const std::uint8_t* from = scol != nullptr ? scol + cols.scales + 4 * kStageSlices * st : fallback;
            ops::cp_async_zfill<8>(&sm.xs[buf][tid], from, scol != nullptr ? 8 : 0);
        }
    };
    // Raw unit u's bytes of every row into buffer `buf` (its last unit may be partial).
    const auto issue_unit = [&](int u, int buf) {
        const int elems         = 32 * min(tile::kUnitSlices, slices - u * tile::kUnitSlices);
        const int bytes         = elems / kBlockElems * kBlockBytes;
        const std::size_t begin = static_cast<std::size_t>(u) * kUnitBytes<F>;
        for (int i = tid; i < kDecodeRows * kRawChunks<F>; i += kThreads) {
            const int r = i / kRawChunks<F>, chunk = i % kRawChunks<F>;
            const std::uint8_t* src = row_of(r) + begin;
            const auto* aligned     = reinterpret_cast<const std::uint8_t*>(reinterpret_cast<std::uintptr_t>(src) & ~std::uintptr_t{15});
            const int offset        = static_cast<int>(src - aligned);
            const int need          = offset + bytes - 16 * chunk; // bytes of this chunk the unit reads
            if (chunk == 0) { sm.raw_offset[buf][r] = offset; }
            if (need > 0) { tile::cp_async16(&sm.raw[buf][r][16 * chunk], aligned + 16 * chunk, need < 16 ? need : 16); }
        }
    };

    zero(acc);
    // Copy groups (ggml_bf16_tile.cuh's compute_decode): the first unit and stage 0's columns, then two per stage
    // st (after its first barrier): stage st + 1's columns, then the next unit's bytes at a unit's first stage
    // (else empty). The wait at stage st leaves only the latest group (a unit's or none) in flight.
    issue_unit(0, 0);
    issue_x(0, 0);
    ops::cp_commit();
    ops::cp_commit();

    const int g = lane / 4, t = lane % 4;
    const int wr = warp / (kColumns / kWarpColumns), wc = warp % (kColumns / kWarpColumns);
    for (int st = 0; st < stages; ++st) {
        const int unit = st / kUnitStages, ubuf = unit & 1, buf = st % kXStages;
        ops::cp_wait<1>(); // stage st's columns and its unit's raw bytes have landed
        __syncthreads();   // and every thread finished stage st - 1 (the codes, its column slot, the raw buffer refilled next)
        if (st + 1 < stages) { issue_x(st + 1, (st + 1) % kXStages); }
        ops::cp_commit();
        if (st % kUnitStages == 0 && (unit + 1) * tile::kUnitSlices < slices) { issue_unit(unit + 1, ubuf ^ 1); }
        ops::cp_commit();
        // The decode: task (j, m) = warp, warp + 8 (as j kDecodeGroups + m); lane (g, t) decodes chunk t of
        // rows 16 m + g and 16 m + g + 8 of slice j: its whole A fragment.
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int task = warp + kWarps * k;
            const int j = task / kDecodeGroups, m = task % kDecodeGroups;
            const int e = (st * kStageSlices + j - unit * tile::kUnitSlices) * 32 + 8 * t; // element in the unit
            tile::TileChunk c[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r               = 16 * m + g + 8 * h;
                const std::uint8_t* block = &sm.raw[ubuf][r][sm.raw_offset[ubuf][r]] + e / kBlockElems * kBlockBytes;
                c[h]                      = tile::tile_chunk<F>(block, e % kBlockElems / 8, sm.codebook);
            }
            sm.a[j][m][g][t] = make_int4(c[0].w[0], c[1].w[0], c[0].w[1], c[1].w[1]);
            if ((t & 1) == 0) { // chunk 0 holds half 0's sub-scale, chunk 2 half 1's
                float* w     = reinterpret_cast<float*>(&sm.w[j][m][g]);
                w[t / 2]     = __fmul_rn(weight_scale(c[0].d), static_cast<float>(c[0].sub));
                w[2 + t / 2] = __fmul_rn(weight_scale(c[1].d), static_cast<float>(c[1].sub));
            }
        }
        __syncthreads(); // the stage's codes and scales are stored
        float4 xd[NT];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) { xd[nt] = *reinterpret_cast<const float4*>(&sm.xs[buf][wc * 8 * NT + 8 * nt + 2 * t]); }
#pragma unroll
        for (int j = 0; j < kStageSlices; ++j) {
            int2 b[NT];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                b[nt] = *reinterpret_cast<const int2*>(&sm.x[buf][j][wc * 8 * NT + 8 * nt + g][2 * t]);
            }
#pragma unroll
            for (int mi = 0; mi < kMTiles; ++mi) {
                const int m          = kMTiles * wr + mi;
                const RowOperands ro = row_operands<S>(sm.a[j][m][g][t], sm.w[j][m][g], make_float2(0.0F, 0.0F), t < 2);
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const float x0 = j == 0 ? xd[nt].x : xd[nt].y, x1 = j == 0 ? xd[nt].z : xd[nt].w;
                    slice_step<S>(acc.r[mi][nt], ro, b[nt], x0, x1, 0.0F, 0.0F);
                }
            }
        }
    }
    ops::cp_wait<0>();
    __syncthreads(); // the caller may reuse shared memory
}

} // namespace infernix::ggml::a8::wide
