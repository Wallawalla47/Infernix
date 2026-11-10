#pragma once

// The ggml-bf16-prefill tiles (docs/maintainer/qwen3_8-flash-next-design.md §19.3.29): BF16 tensor-core
// tiles over decoded GGUF weights. ops::ggml_linear runs them for wide dense projections
// (LinearRoutes::GgmlBf16Wide) over a BF16 plane (PlaneStage images: the integers of ggml_tile.cuh's
// CodeStage, the exact route's plane, already converted to the MMAs' operands once per call), and the
// GGUF expert banks for wide jobs (offloaded_moe::WideA16::kGgmlBf16, ggml_expert.cuh), which decode
// each K stage in shared memory (compute_decode: no plane, the same bits).
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
// A CTA computes kRows = 128 rows (two plane tiles, `images`) x kColumns = 128 columns, stages of 64
// elements of K by cp.async: the columns in a ring of kStages, the plane images in two buffers. Column
// c's K elements come from a Columns policy (contiguous columns of a dense call, or an expert's
// gathered token rows). The stage-decode tiles run 8 warps of 64 x 32 (four m16 by four n8 tiles; warp
// row wr reads image wr); the plane tiles, whose threads only load and multiply, run 16 warps of
// 64 x 16 or 32 of 64 x 8 (kPlaneWarpColumns): a CTA owns its SM, and eight warps leave its tensor
// cores short of work.

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
constexpr int kThreads     = 256;                 // the 8-warp shape (2 rows x 4 columns) NT counts n8 tiles of
constexpr int kStages      = 3;
constexpr int kSlices      = tile::kStageSlices;  // 32-element slices per stage
constexpr int kWarpColumns = 32;
constexpr int kMTiles      = tile::kTileRows / 16; // a warp's m16 tiles: its plane tile's rows
constexpr int kNTiles      = kWarpColumns / 8;
static_assert(kThreads / 32 == (kRows / tile::kTileRows) * (kColumns / kWarpColumns));
// Threads of a plane tile CTA whose warps take WC columns each.
template <int WC>
inline constexpr int kPlaneThreads = (kRows / tile::kTileRows) * (kColumns / WC) * 32;

// How a format's slice is scaled: one sub-scale per 32 elements (kPlain), one per 16 (kHalf: Q6_K,
// IQ2_S), none (kUnit: IQ4_NL, Q2_0), or one per 32 with the K-quants' min term (kMin).
enum class Scaling : int { kPlain, kHalf, kUnit, kMin };

template <QType F>
inline constexpr Scaling kScaling = tile::kHalfSub<F> ? Scaling::kHalf
                                    : fast::kMin<F>   ? Scaling::kMin
                                    : tile::kUnitSub<F> ? Scaling::kUnit
                                                        : Scaling::kPlain;


// One stage of a 64-row plane tile as the MMAs read it (the BF16 plane): the codes as exact BF16
// pairs in the k16 fragment order, and the per-row factors already formed, so a column tile's warps
// neither convert a code nor touch a scale's integer parts.
struct PlaneStage {
    // a[j][m][h][g][t]: lane (g, t)'s A registers 0-3 of row group m's elements 16 h .. 16 h + 15
    // (slice_step's lo for h = 0, hi for h = 1).
    alignas(16) uint4 a[kSlices][tile::kRowGroups][2][8][4];
    // scale[j][m][g]: d * sub of row 16 m + g's halves 0 and 1, then row 16 m + g + 8's (d where the
    // format has no sub-scales).
    alignas(16) float4 scale[kSlices][tile::kRowGroups][8];
    // min[j][m][g]: dmin * min of rows 16 m + g and 16 m + g + 8 (kMin only: the image's tail).
    alignas(16) float2 min[kSlices][tile::kRowGroups][8];
};
static_assert(sizeof(PlaneStage) % 16 == 0 && offsetof(PlaneStage, min) % 16 == 0, "plane images copy in 16-byte units");

// Bytes of a stage image the tiles copy (the min terms only where the scaling reads them).
template <Scaling S>
inline constexpr int kImageBytes = S == Scaling::kMin ? static_cast<int>(sizeof(PlaneStage))
                                                      : static_cast<int>(offsetof(PlaneStage, min));

// A plane tile warp's columns. RTX 5090, ggml_linear_bench --routes bf16 --cold (whole call, eight
// shapes x 512 / 2,048 / 4,096 columns, 2026-11-10; equal output hashes), against 32 columns (8 warps):
// 16 columns -3.7 .. -8.3 % on the Q4_K / Q5_K / IQ shapes and -13.6 .. -14.6 % on the Q6_K ssm_out;
// 8 columns -3.3 .. -7.2 % and -17.7 .. -19.2 %. The half-scaled formats (two partial sums a slice)
// take 8, the rest 16.
template <Scaling S>
inline constexpr int kPlaneWarpColumns = S == Scaling::kHalf ? 8 : 16;

constexpr int kImageBufs = 2;

struct Shared {
    alignas(16) std::uint8_t image[kImageBufs][2][sizeof(PlaneStage)];
    alignas(16) std::uint16_t x[kStages][kSlices][kColumns][32]; // BF16 bits, 8-byte units permuted
    float xsum[kStages][kSlices][kColumns];                      // kMin only
};
static_assert(sizeof(Shared) <= 99 * 1024);

// The accumulators of a stage-decode CTA's thread: its warp's four m16 by four n8 tiles.
struct Accumulators {
    float r[kMTiles][kNTiles][4];
};


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

// Item i (< kSlices * tile::kRowGroups * 32) of a decoded stage as its BF16 plane image: lane (g, t)'s
// fragment vectors of slice j's row group m and, from the lane with t = 0, the group's factors. The
// values are the ones the stage-decode tiles form (compute_decode), from the same integers.
template <Scaling S>
__device__ __forceinline__ void plane_item(const tile::CodeStage& cs, PlaneStage& out, int i) {
    const int j = i / (tile::kRowGroups * 32), m = (i / 32) % tile::kRowGroups, g = (i / 4) % 8, t = i % 4;
    const int4 a = *reinterpret_cast<const int4*>(cs.a[j][m][g][t]);
    // Elements 0-15 (lo) and 16-31 (hi): rows g, g + 8, then their elements 4t + 2, + 3.
    out.a[j][m][0][g][t] = make_uint4(bf16_pair(static_cast<unsigned>(a.x)), bf16_pair(static_cast<unsigned>(a.y)),
                                      bf16_pair(static_cast<unsigned>(a.x) >> 16), bf16_pair(static_cast<unsigned>(a.y) >> 16));
    out.a[j][m][1][g][t] = make_uint4(bf16_pair(static_cast<unsigned>(a.z)), bf16_pair(static_cast<unsigned>(a.w)),
                                      bf16_pair(static_cast<unsigned>(a.z) >> 16), bf16_pair(static_cast<unsigned>(a.w) >> 16));
    if (t != 0) { return; }
    const int r = 16 * m + g; // rows r and r + 8 of the tile
    const float d0 = f16_value(cs.d[j][r]), d1 = f16_value(cs.d[j][r + 8]);
    // d * sub per row (0: r, 1: r + 8) and half: exact (an 11-bit mantissa by |sub| <= 127).
    float4 scale = make_float4(d0, d0, d1, d1);
    if constexpr (S != Scaling::kUnit) {
        const int4 sc = *reinterpret_cast<const int4*>(cs.sub[j][m][g]);
        scale = make_float4(d0 * static_cast<float>(sc.x), d0 * static_cast<float>(sc.y), d1 * static_cast<float>(sc.z),
                            d1 * static_cast<float>(sc.w));
    }
    out.scale[j][m][g] = scale;
    if constexpr (S == Scaling::kMin) {
        out.min[j][m][g] = make_float2(f16_value(cs.dmin[j][r]) * static_cast<float>(cs.min[j][r]),
                                       f16_value(cs.dmin[j][r + 8]) * static_cast<float>(cs.min[j][r + 8]));
    }
}

// One slice of an m16 x n8 tile: the MMAs of A's elements 0-15 (lo) and 16-31 (hi) by B (b: both
// halves' registers) and their scaling into the accumulators o, with s00 / s01 row g's d * sub of
// the two halves and s10 / s11 row g + 8's (kHalf; the others read s00 and s10). Both stage sources
// (compute's planes and compute_decode's stage decode) run exactly this, so with the same codes,
// scales and B they give the same bits.
template <Scaling S>
__device__ __forceinline__ void slice_step(float (&o)[4], const unsigned (&lo)[4], const unsigned (&hi)[4], const uint4& b,
                                           float s00, float s01, float s10, float s11) {
    if constexpr (S == Scaling::kHalf) {
        float p[4] = {0.0F, 0.0F, 0.0F, 0.0F}, q[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        mma(p, lo, b.x, b.y);
        mma(q, hi, b.z, b.w);
        o[0] = fmaf(s01, q[0], fmaf(s00, p[0], o[0]));
        o[1] = fmaf(s01, q[1], fmaf(s00, p[1], o[1]));
        o[2] = fmaf(s11, q[2], fmaf(s10, p[2], o[2]));
        o[3] = fmaf(s11, q[3], fmaf(s10, p[3], o[3]));
    } else {
        float p[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        mma(p, lo, b.x, b.y);
        mma(p, hi, b.z, b.w);
        o[0] = fmaf(s00, p[0], o[0]);
        o[1] = fmaf(s00, p[1], o[1]);
        o[2] = fmaf(s10, p[2], o[2]);
        o[3] = fmaf(s10, p[3], o[3]);
    }
}

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
//
// WC: the columns of a warp's tile (kPlaneWarpColumns<S> for the plane tiles). acc[mi][nt]: the thread's
// accumulators of its warp's m16 tile mi and n8 tile nt.
template <Scaling S, int WC, class Columns>
__device__ __forceinline__ void compute(Shared& sm, const std::uint8_t* const (&images)[2], int stages, const Columns& cols,
                                        float (&acc)[kMTiles][WC / 8][4]) {
    constexpr int kThreads     = kPlaneThreads<WC>;
    constexpr int kWarpColumns = WC;
    constexpr int kNTiles      = WC / 8;
    constexpr int kImage          = kImageBytes<S>;
    constexpr std::size_t kStride = sizeof(PlaneStage);
    // Stage st's plane images of both row tiles into image buffer `buf`.
    const auto issue_images = [&](int st, int buf) {
        for (int i = static_cast<int>(threadIdx.x); i < 2 * (kImage / 16); i += kThreads) {
            const int h = i / (kImage / 16), c = i % (kImage / 16);
            tile::cp_async16(&sm.image[buf][h][16 * c], images[h] + st * kStride + 16 * c, 16);
        }
    };
    // Stage st's columns into ring slot `buf`: their units (zero past the columns) and, for kMin, their
    // slice sums.
    const auto issue = [&](int st, int buf) {
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
            for (int q = 0; q < 4; ++q) { acc[mi][nt][q] = 0.0F; }
        }
    }
    // Copy groups: stage 0's images and columns, then stage 1's columns; then two per stage st (after
    // its barrier): stage st + 1's images, then stage st + 2's columns. The wait at stage st leaves only
    // the latest group (columns) in flight, so an image has one stage to arrive and the columns two.
    static_assert(kStages == 3 && kImageBufs == 2, "the copy groups below");
    issue_images(0, 0);
    issue(0, 0);
    ops::cp_commit();
    if (1 < stages) { issue(1, 1); }
    ops::cp_commit();
    for (int st = 0; st < stages; ++st) {
        ops::cp_wait<1>(); // stage st's images and columns have landed
        __syncthreads();   // and every thread finished stage st - 1, whose buffers the next issues fill
        if (st + 1 < stages) { issue_images(st + 1, (st + 1) % kImageBufs); }
        ops::cp_commit();
        if (st + 2 < stages) { issue(st + 2, (st + 2) % kStages); }
        ops::cp_commit();
        const int buf = st % kStages;
        const auto& ps = *reinterpret_cast<const PlaneStage*>(sm.image[st % kImageBufs][wr]);
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
                // Rows 16 mi + g and + 8 of the warp's plane tile: elements 0-15 (lo) and 16-31 (hi).
                const uint4 l = ps.a[j][mi][0][g][t], u = ps.a[j][mi][1][g][t];
                const float4 sc = ps.scale[j][mi][g];
                const unsigned lo[4] = {l.x, l.y, l.z, l.w}, hi[4] = {u.x, u.y, u.z, u.w};
                float m0 = 0.0F, m1 = 0.0F;
                if constexpr (S == Scaling::kMin) {
                    const float2 mins = ps.min[j][mi][g];
                    m0 = mins.x;
                    m1 = mins.y;
                }
#pragma unroll
                for (int nt = 0; nt < kNTiles; ++nt) {
                    float(&o)[4] = acc[mi][nt];
                    slice_step<S>(o, lo, hi, b[nt], sc.x, sc.y, sc.z, sc.w);
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

// ------------------------------------------------------------------------------ stage decode
//
// compute_decode is compute without a plane: the CTA copies the raw GGUF bytes of its kRows rows by
// cp.async, 256 elements of K (one raw unit: tile::kUnitSlices slices) per row at a time,
// double-buffered and issued one unit ahead, and decodes every K stage into shared memory before
// its MMAs (decode-then-MMA, as the exact tiles: decoding the next stage beside this one's MMAs
// measured 3-5 % slower there). The decode gives each chunk the integers the planes hold
// (tile_chunk = fast::chunk) and stores them already converted to BF16 pairs in the k16 fragment
// order the MMAs read (the planes' compute converts each code in every one of the four column
// warps), with d * sub per row and half in FP32 (the same products compute forms); the K loop then
// runs slice_step on the same operands in the same order, so every output is the planes' bits. The
// codebooks are the CTA's shared copy (SharedCodebook). B is staged as compute stages it, in a
// two-stage ring (one stage ahead: the three-stage ring does not fit beside the raw units).
//
// A tile has NT n8 tiles per warp: 4 (128 columns) or 2 (64: a job's last tile when at most 64 of
// its columns remain); a column's outputs do not depend on its tile's width.
//
// Shared memory (DecodeShared): IQ3_S 89 KiB, IQ4_NL / IQ4_XS 95 KiB, IQ2_S 87 KiB, IQ3_XXS 88 KiB,
// Q2_0 79 KiB: one CTA of 256 threads per SM, as the planes' tiles.

constexpr int kDecodeXStages = 2;
constexpr int kRowGroups     = kRows / 16; // m16 row groups of the CTA's rows (warp row wr: groups 4 wr ..)
static_assert(kRowGroups == 2 * kMTiles);

template <QType F>
inline constexpr int kUnitBytes = tile::kUnitSlices * 32 / static_cast<int>(ggml_block_elems(F)) *
                                  static_cast<int>(ggml_block_bytes(F));
// A raw row's 16-byte chunks: a unit from its row's 16-byte aligned start (up to 15 bytes before
// it), rounded to an odd count so that the rows a warp decodes start in distinct bank quads.
template <QType F>
inline constexpr int kRawChunks = ((15 + kUnitBytes<F> + 15) / 16) | 1;

template <QType F>
struct DecodeShared {
    alignas(16) std::uint16_t x[kDecodeXStages][kSlices][kColumns][32]; // as Shared::x
    // The stage's codes as BF16 pairs: a[j][m][h][g][t] = lane (g, t)'s A registers 0-3 of row group
    // m's elements 16 h .. 16 h + 15 (slice_step's lo for h = 0, hi for h = 1).
    alignas(16) uint4 a[kSlices][kRowGroups][2][8][4];
    // scale[j][m][g] = d * sub of row 16 m + g's halves 0 and 1, then row 16 m + g + 8's (d where
    // the format has no sub-scales).
    alignas(16) float4 scale[kSlices][kRowGroups][8];
    alignas(16) std::uint8_t raw[2][kRows][kRawChunks<F> * 16];
    int raw_offset[2][kRows]; // where row r's unit bytes start in raw[.][r]
    fast::SharedCodebook<F> codebook;
};

// The stage-decode tiles' threads: sixteen warps of 64 x 16, one decode task a warp and stage. RTX 5090,
// offloaded_moe_wide_bench --bank ggml --ggml-wide bf16 --cold --graph (512 experts, equal output hashes,
// 2026-11-10), against eight warps of 64 x 32: experts in device frames -5.3 .. -6.5 % at 4,096 tokens and
// -7.5 .. -8.1 % at 16,384 (IQ2_S / IQ3_XXS / IQ3_S gate/up with IQ4_NL down); staged experts (the call
// waits on the link) -0.1 .. -2.6 %.
constexpr int kDecodeThreads = 512;

// A stage-decode CTA of T threads: T / 64 warp columns by two warp rows of 64 rows; a tile of 32 NT
// columns gives each warp kDecodeWarpTiles n8 tiles.
template <int NT, int T>
inline constexpr int kDecodeWarpTiles = NT * kThreads / T;

// Output row (within the CTA's 128) and column (within its columns) of accumulator (mi, nt, q) of a
// tile of 32 NT columns in a CTA of T threads.
template <int T>
__device__ __forceinline__ int decode_out_row(int warp, int lane, int mi, int q) {
    return (warp / (T / 64)) * tile::kTileRows + 16 * mi + lane / 4 + 8 * (q >> 1);
}
template <int NT, int T>
__device__ __forceinline__ int decode_out_column(int warp, int lane, int nt, int q) {
    return (warp % (T / 64)) * (8 * kDecodeWarpTiles<NT, T>) + 8 * nt + 2 * (lane % 4) + (q & 1);
}

// The K loop over `stages` stages of rows row_of(0 .. kRows - 1) (kRows / 2 .. are the second
// warp row's), B columns from `cols`. sm.codebook must hold F's codebook, published by a barrier
// (the caller loads it once per CTA). Every thread calls it; it leaves the thread's accumulators
// (acc.r[mi][nt], nt < kDecodeWarpTiles<NT, T>) and returns with every copy landed and every thread past
// its last read of shared memory. T: the CTA's threads (kDecodeThreads; 256 is the eight-warp shape, two
// decode tasks a warp and stage).
template <Scaling S, QType F, int NT, int T, class RowOf, class Columns>
__device__ __forceinline__ void compute_decode(DecodeShared<F>& sm, RowOf row_of, int stages, const Columns& cols,
                                               Accumulators& acc) {
    static_assert(S != Scaling::kMin, "the stage-decode tiles serve the expert formats (no min term)");
    static_assert(NT == 4 || NT == 2);
    static_assert(T == 256 || T == 512);
    constexpr int kColWarps = T / 64;                     // warp columns
    constexpr int kWNT      = kDecodeWarpTiles<NT, T>;    // a warp's n8 tiles
    constexpr int kBlockElems = static_cast<int>(ggml_block_elems(F));
    constexpr int kBlockBytes = static_cast<int>(ggml_block_bytes(F));
    constexpr int kUnitStages = tile::kUnitSlices / kSlices;
    constexpr int kCols       = (kColumns / kWarpColumns) * 8 * NT; // the tile's columns
    constexpr int kXCopies    = kSlices * kCols * 8 / T;           // 8-byte B copies per thread and stage
    static_assert(kXCopies * T == kSlices * kCols * 8);
    const int tid = static_cast<int>(threadIdx.x), warp = tid / 32, lane = tid % 32;
    const int slices = stages * kSlices;

    // A thread's B copies: copy k (i = tid + T k) is unit u = tid % 8 of column c = tid / 8 +
    // (T / 8) (k % kPer) in the stage's slice j = k / kPer, the same every stage, so its kPer columns
    // are resolved once per tile (a column past the tile's is zero-filled from column 0).
    constexpr int kPer = kXCopies / kSlices;
    static_assert(kCols == (T / 8) * kPer, "copy k's column and slice as below");
    const std::uint16_t* xcol[kPer];
    unsigned valid = 0;
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const std::uint16_t* column = cols.column(tid / 8 + (T / 8) * k);
        xcol[k] = column != nullptr ? column : cols.column(0);
        valid |= column != nullptr ? 1U << k : 0U;
    }
    const int xu = tid % 8, xperm = 4 * (2 * (xu % 4) + xu / 4);
    const auto issue_x = [&](int st, int buf) {
#pragma unroll
        for (int k = 0; k < kXCopies; ++k) {
            const int j = k / kPer, c = tid / 8 + (T / 8) * (k % kPer);
            ops::cp_async_zfill<8>(&sm.x[buf][j][c][xperm], xcol[k % kPer] + cols.element((st * kSlices + j) * 32 + 4 * xu),
                                   (valid >> (k % kPer)) & 1U ? 8 : 0);
        }
    };
    // Raw unit u's bytes of every row into buffer `buf` (its last unit may be partial).
    const auto issue_unit = [&](int u, int buf) {
        const int elems         = 32 * min(tile::kUnitSlices, slices - u * tile::kUnitSlices);
        const int bytes         = elems / kBlockElems * kBlockBytes;
        const std::size_t begin = static_cast<std::size_t>(u) * kUnitBytes<F>;
        for (int i = tid; i < kRows * kRawChunks<F>; i += T) {
            const int r = i / kRawChunks<F>, chunk = i % kRawChunks<F>;
            const std::uint8_t* src = row_of(r) + begin;
            const auto* aligned     = reinterpret_cast<const std::uint8_t*>(reinterpret_cast<std::uintptr_t>(src) & ~std::uintptr_t{15});
            const int offset        = static_cast<int>(src - aligned);
            const int need          = offset + bytes - 16 * chunk; // bytes of this chunk the unit reads
            if (chunk == 0) { sm.raw_offset[buf][r] = offset; }
            if (need > 0) { tile::cp_async16(&sm.raw[buf][r][16 * chunk], aligned + 16 * chunk, need < 16 ? need : 16); }
        }
    };

#pragma unroll
    for (int mi = 0; mi < kMTiles; ++mi) {
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) { acc.r[mi][nt][q] = 0.0F; }
        }
    }
    // Copy groups: the first unit and stage 0's B, then two per stage st (after its first barrier):
    // stage st + 1's B, then the next unit's bytes at a unit's first stage (else empty). The wait at
    // stage st leaves only the latest group (a unit's or none) in flight, so B has one stage to arrive
    // and a unit two (it is needed two stages later still).
    issue_unit(0, 0);
    issue_x(0, 0);
    ops::cp_commit();
    ops::cp_commit();

    // The decode's tasks: thread (warp, lane) decodes chunk dp (elements 8 dp .. 8 dp + 7) of rows
    // 16 m + g and 16 m + g + 8 of slice j for (j, m) = warp (and warp + 8 in a CTA of eight warps; as
    // j kRowGroups + m), so
    // it holds both rows' words 2 dp and 2 dp + 1 and stores two whole fragment vectors (t = 2 (dp
    // % 2) and + 1, half dp / 2). Lane bits: 0 g's bit 0, 1 dp's bit 0, 2 g's bit 1, 3 g's bit 2, 4
    // dp's bit 1; lanes whose g has bit 1 set store their t + 1 vector first, so each quarter warp's
    // 16-byte stores fall in eight distinct bank quads.
    const int dg = (lane & 1) | ((lane >> 1) & 6), ddp = ((lane >> 1) & 1) | ((lane >> 3) & 2);
    const bool swap = ((lane >> 2) & 1) != 0;
    constexpr int kTasks = kSlices * kRowGroups * 32 / T; // decode tasks per warp and stage
    static_assert(kTasks * T == kSlices * kRowGroups * 32, "the warps' decode tasks cover a stage");

    const int g = lane / 4, t = lane % 4;
    const int wr = warp / kColWarps, wc = warp % kColWarps;
    for (int st = 0; st < stages; ++st) {
        const int unit = st / kUnitStages, ubuf = unit & 1, buf = st % kDecodeXStages;
        ops::cp_wait<1>(); // stage st's B and its unit's raw bytes have landed
        __syncthreads();   // and every thread finished stage st - 1 (the codes, its B slot, the raw buffer refilled next)
        if (st + 1 < stages) { issue_x(st + 1, (st + 1) % kDecodeXStages); }
        ops::cp_commit();
        if (st % kUnitStages == 0 && (unit + 1) * tile::kUnitSlices < slices) { issue_unit(unit + 1, ubuf ^ 1); }
        ops::cp_commit();
#pragma unroll
        for (int k = 0; k < kTasks; ++k) {
            const int task = warp + (T / 32) * k;
            const int j = task / kRowGroups, m = task % kRowGroups;
            const int e = (st * kSlices + j - unit * tile::kUnitSlices) * 32 + 8 * ddp; // element in the unit
            tile::TileChunk c[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = 16 * m + dg + 8 * h;
                const std::uint8_t* block = &sm.raw[ubuf][r][sm.raw_offset[ubuf][r]] + e / kBlockElems * kBlockBytes;
                c[h] = tile::tile_chunk<F>(block, e % kBlockElems / 8, sm.codebook);
            }
            const auto vector = [&](int w) {
                return make_uint4(bf16_pair(static_cast<unsigned>(c[0].w[w])), bf16_pair(static_cast<unsigned>(c[1].w[w])),
                                  bf16_pair(static_cast<unsigned>(c[0].w[w]) >> 16),
                                  bf16_pair(static_cast<unsigned>(c[1].w[w]) >> 16));
            };
            const int t0 = 2 * (ddp % 2), half = ddp / 2;
            uint4(&dst)[4] = sm.a[j][m][half][dg];
            if (swap) {
                dst[t0 + 1] = vector(1);
                dst[t0]     = vector(0);
            } else {
                dst[t0]     = vector(0);
                dst[t0 + 1] = vector(1);
            }
            if (ddp % 2 == 0) {
                // d * sub of this half of each row: exact in binary32 (an 11-bit mantissa by |sub| <= 127).
                float* s = reinterpret_cast<float*>(&sm.scale[j][m][dg]);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float d = f16_value(c[h].d);
                    s[2 * h + half] = S == Scaling::kUnit ? d : d * static_cast<float>(c[h].sub);
                }
            }
        }
        __syncthreads(); // the stage's codes and scales are stored
#pragma unroll
        for (int j = 0; j < kSlices; ++j) {
            uint4 b[kWNT];
#pragma unroll
            for (int nt = 0; nt < kWNT; ++nt) {
                b[nt] = *reinterpret_cast<const uint4*>(&sm.x[buf][j][wc * 8 * kWNT + 8 * nt + g][8 * t]);
            }
#pragma unroll
            for (int mi = 0; mi < kMTiles; ++mi) {
                const int m   = kMTiles * wr + mi;
                const uint4 l = sm.a[j][m][0][g][t], u = sm.a[j][m][1][g][t];
                const float4 sc = sm.scale[j][m][g];
                const unsigned lo[4] = {l.x, l.y, l.z, l.w}, hi[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int nt = 0; nt < kWNT; ++nt) { slice_step<S>(acc.r[mi][nt], lo, hi, b[nt], sc.x, sc.y, sc.z, sc.w); }
            }
        }
    }
    ops::cp_wait<0>();
    __syncthreads(); // the caller may reuse shared memory
}

// A dense call's tile: rows [kRows bx, + kRows) of a segment (or slab) whose plane holds `tiles` row
// tiles (the last image pair clamped to the last tile; those rows' outputs are not stored) against
// columns [kColumns by, + kColumns). out[t * n + out_offset + row] for row < rows, t < columns.
template <Scaling S, bool OutF32, int WC = kPlaneWarpColumns<S>>
__global__ void __launch_bounds__(kPlaneThreads<WC>, 1)
    tile_kernel(const std::uint8_t* __restrict__ plane, int tiles, int rows, int slices, DenseColumns cols,
                void* __restrict__ out, int n, int out_offset, int kappa_log2) {
    extern __shared__ __align__(16) unsigned char bf16_shared[];
    auto& sm = *reinterpret_cast<Shared*>(bf16_shared);
    const int stages = slices / kSlices;
    const int tile0  = static_cast<int>(blockIdx.x) * 2;
    cols.col0        = static_cast<int>(blockIdx.y) * kColumns;
    constexpr std::size_t kStride = sizeof(PlaneStage);
    const std::uint8_t* const images[2] = {plane + static_cast<std::size_t>(min(tile0, tiles - 1)) * stages * kStride,
                                           plane + static_cast<std::size_t>(min(tile0 + 1, tiles - 1)) * stages * kStride};
    float acc[kMTiles][WC / 8][4];
    compute<S, WC>(sm, images, stages, cols, acc);
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int wr = warp / (kColumns / WC), wc = warp % (kColumns / WC);
#pragma unroll
    for (int mi = 0; mi < kMTiles; ++mi) {
#pragma unroll
        for (int nt = 0; nt < WC / 8; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int row = static_cast<int>(blockIdx.x) * kRows + wr * tile::kTileRows + 16 * mi + lane / 4 + 8 * (q >> 1);
                const int col = cols.col0 + wc * WC + 8 * nt + 2 * (lane % 4) + (q & 1);
                if (row < rows && col < cols.columns) {
                    const float v        = ldexpf(acc[mi][nt][q], kappa_log2);
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
