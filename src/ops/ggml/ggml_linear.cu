// Exact projections over GGUF block weights (include/infernix/ops/ggml_linear.h).
//
// encode_kernel writes each column once as A16 limbs (ops/ggml/ggml_dp4a.cuh: the column's
// exponent, then per 32-element slice its three limb planes and its X sum, with the optional input
// block permutation). rows_kernel gives each output row a group of `Lanes` lanes (4 to 32, so
// that every lane has several of the row's slices whatever K); a lane decodes its slices once per
// pass of up to kColumns columns and adds their exact terms into one int128 per column; the
// group's partial sums are added exactly (order cannot matter) and its first lane rounds. (Every
// rows CTA encoding a decode call's column into shared memory instead of encode_kernel measured
// 4 % slower end to end: 98.0 -> 94.0 tok/s, design §19.3.25.)
//
// Calls of kMmaMinColumns..kMmaMaxColumns columns (MTP verification) run mma_rows_kernel on tensor
// cores; calls of kTileMinColumns or more (prefill) first decode each segment's weights once into
// a plane (plane_kernel, workspace kPlaneBytes) and run the tensor-core tiles over it. Every route
// returns the same integers. With LinearRoutes::GgmlBf16Wide (ggml-bf16-prefill) the wide calls run
// ops/ggml/ggml_bf16_tile.cuh's BF16 tiles over the same planes instead (no column encoding).

#include "infernix/ops/ggml_linear.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "core/ggml_format.h"
#include "ops/ggml/ggml_bf16_tile.cuh"
#include "ops/ggml/ggml_dp4a.cuh"
#include "ops/ggml/ggml_exact.h"
#include "ops/ggml/ggml_mma_rows.cuh"
#include "ops/ggml/ggml_tile.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infernix::ops {
namespace {

namespace fast = ggml::fast;

constexpr int kThreads       = 128;
constexpr int kWarps         = kThreads / 32;
constexpr int kEncodeThreads = 256;
constexpr int kColumns       = 4; // columns per pass of a row group (calls of 5-8 columns: one pass of 8)
// Calls of at least this many columns run the tensor-core tiles (ops/ggml/ggml_tile.cuh).
constexpr int kTileMinColumns = 32;
// Columns encoded at once: wider calls (prefill chunks) run in batches of this many columns, which
// bounds the workspace (K * 512 * 3.125 bytes) whatever the chunk size.
constexpr int kEncodeColumns = 512;
// Workspace of a wide call's decoded planes (1.5 bytes per weight: every IQ3_S dense call in one pass).
constexpr std::size_t kPlaneBytes = 64ULL << 20;

// Up to kMax segments of one format in one narrow launch; CTAs [first[i], first[i + 1]) take segment
// i's rows into out[i] (n[i] rows per column). A verification launch may hold two formats
// (mma_rows_mixed_kernel): format[i] picks F0 or F1, row_bytes[i] its rows. One launch per call
// instead of one per segment (GDN q/k/v/z at one column 33.4 -> 23.0 us; RTX 5090, IQ3_S, MTP 4 +
// n-gram 7 at 4K 196.1 -> 202.4 tok/s with mixed launches, two rounds, 2026-10-16).
struct RowsGroup {
    static constexpr int kMax = 4;
    const std::uint8_t* data[kMax] = {};
    void* out[kMax]                = {};
    int rows[kMax]                 = {};
    int n[kMax]                    = {};
    int out_offset[kMax]           = {};
    int row_bytes[kMax]            = {};
    int format[kMax]               = {};
    int first[kMax + 1]            = {};
    int count                      = 0;
};

// A CTA's segment of its group and its block index within that segment.
struct GroupSlot {
    const std::uint8_t* weights;
    void* out;
    int rows, n, out_offset, block, row_bytes, format;
};

__device__ __forceinline__ GroupSlot group_slot(const RowsGroup& g) {
    const int b = static_cast<int>(blockIdx.x);
    int i       = 0;
#pragma unroll
    for (int j = 1; j < RowsGroup::kMax; ++j) { i = j < g.count && b >= g.first[j] ? j : i; }
    // Selects with constant indices, so the parameters are read from the constant bank in place.
    GroupSlot s{g.data[0], g.out[0], g.rows[0], g.n[0], g.out_offset[0], b - g.first[0], g.row_bytes[0], g.format[0]};
#pragma unroll
    for (int j = 1; j < RowsGroup::kMax; ++j) {
        if (i == j) {
            s = GroupSlot{g.data[j], g.out[j], g.rows[j], g.n[j], g.out_offset[j], b - g.first[j], g.row_bytes[j], g.format[j]};
        }
    }
    return s;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("ggml_linear: ") + message); }
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("ggml_linear ") + what + ": " + cudaGetErrorString(error)); }
}

// Encodes one column with the whole CTA (Threads threads): its exponent (returned to every thread),
// then its limbs and slice sums into `column`. `warp_key` and `broadcast` are shared scratch.
template <int Threads>
__device__ int encode_column(const std::uint16_t* __restrict__ v, int k, const std::int32_t* __restrict__ perm,
                             int perm_block, std::int32_t* __restrict__ column, unsigned* warp_key, int* broadcast) {
    constexpr int kEncodeWarps = Threads / 32;
    const auto source = [&](int j) { return perm == nullptr ? j : perm[j / perm_block] * perm_block + j % perm_block; };
    unsigned key = 0;
    for (int j = threadIdx.x; j < k; j += Threads) { key = max(key, static_cast<unsigned>(v[j] & 0x7FFFU)); }
    key = __reduce_max_sync(0xFFFFFFFFU, key);
    if (threadIdx.x % 32 == 0) { warp_key[threadIdx.x / 32] = key; }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned m = 0;
        for (int w = 0; w < kEncodeWarps; ++w) { m = max(m, warp_key[w]); }
        *broadcast = m >= 0x7F80U ? canon::kA16NonFinite : max(static_cast<int>(m >> 7), 1);
    }
    __syncthreads();
    const int e = *broadcast;
    const int groups = k / 4, slices = k / 32;
    const auto value = [&](int j) { return e == canon::kA16NonFinite ? 0 : canon::a16_value(v[source(j)], e); };
    // Groups are visited so that the eight groups of a slice sit in eight consecutive lanes.
    for (int base = 0; base < groups; base += Threads) {
        const int g = base + static_cast<int>(threadIdx.x);
        int sum     = 0;
        if (g < groups) { sum = fast::store_group(column, g, value(4 * g), value(4 * g + 1), value(4 * g + 2), value(4 * g + 3)); }
        sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 1);
        sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 2);
        sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 4);
        if (g < groups && g % 8 == 0) { column[slices * fast::kSliceWords + g / 8] = sum; }
    }
    return e;
}

// encode_column with 16-byte loads, for every column it can read (end to end 0.7 % faster, two
// rounds, 2026-10-16). Thread i takes the column's 8-element chunks i, i + Threads, ... (chunk c holds
// groups 2c and 2c + 1, read from the permuted source in one load: needs perm_block % 8 == 0 and a
// 16-byte aligned column), keeps the first kKeep of them in registers from the exponent's pass to the
// limbs' pass, and adds a slice's sum over its four chunks' lanes. Writes the same words as
// encode_column.
template <int Threads>
__device__ int encode_column_vec(const std::uint16_t* __restrict__ v, int k, const std::int32_t* __restrict__ perm,
                                 int perm_block, std::int32_t* __restrict__ column, unsigned* warp_key, int* broadcast) {
    constexpr int kEncodeWarps = Threads / 32;
    constexpr int kKeep        = 4;
    const auto source = [&](int j) { return perm == nullptr ? j : perm[j / perm_block] * perm_block + j % perm_block; };
    const int chunks  = k / 8;
    const auto load   = [&](int c) { return *reinterpret_cast<const uint4*>(v + source(8 * c)); };
    const auto max8   = [](const uint4& u) {
        const unsigned m0 = max(u.x & 0x7FFFU, (u.x >> 16) & 0x7FFFU), m1 = max(u.y & 0x7FFFU, (u.y >> 16) & 0x7FFFU);
        const unsigned m2 = max(u.z & 0x7FFFU, (u.z >> 16) & 0x7FFFU), m3 = max(u.w & 0x7FFFU, (u.w >> 16) & 0x7FFFU);
        return max(max(m0, m1), max(m2, m3));
    };
    uint4 raw[kKeep];
    unsigned key = 0;
#pragma unroll
    for (int m = 0; m < kKeep; ++m) {
        const int c = m * Threads + static_cast<int>(threadIdx.x);
        raw[m]      = c < chunks ? load(c) : make_uint4(0, 0, 0, 0);
        key         = max(key, max8(raw[m]));
    }
    for (int c = kKeep * Threads + static_cast<int>(threadIdx.x); c < chunks; c += Threads) { key = max(key, max8(load(c))); }
    key = __reduce_max_sync(0xFFFFFFFFU, key);
    if (threadIdx.x % 32 == 0) { warp_key[threadIdx.x / 32] = key; }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned m = 0;
        for (int w = 0; w < kEncodeWarps; ++w) { m = max(m, warp_key[w]); }
        *broadcast = m >= 0x7F80U ? canon::kA16NonFinite : max(static_cast<int>(m >> 7), 1);
    }
    __syncthreads();
    const int e = *broadcast;
    const int slices = k / 32;
    const auto value = [&](unsigned bits) {
        return e == canon::kA16NonFinite ? 0 : canon::a16_value(static_cast<std::uint16_t>(bits), e);
    };
    const auto encode = [&](int c, const uint4& u) { // groups 2c and 2c + 1; returns their sum
        int sum = fast::store_group(column, 2 * c, value(u.x & 0xFFFFU), value(u.x >> 16), value(u.y & 0xFFFFU),
                                    value(u.y >> 16));
        sum += fast::store_group(column, 2 * c + 1, value(u.z & 0xFFFFU), value(u.z >> 16), value(u.w & 0xFFFFU),
                                 value(u.w >> 16));
        return sum;
    };
    const auto finish = [&](int c, int sum) { // every lane of the warp: a slice is four chunks' lanes
        sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 1);
        sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 2);
        if (c < chunks && c % 4 == 0) { column[slices * fast::kSliceWords + c / 4] = sum; }
    };
#pragma unroll
    for (int m = 0; m < kKeep; ++m) {
        if (m * Threads >= chunks) { break; } // uniform
        const int c = m * Threads + static_cast<int>(threadIdx.x);
        finish(c, c < chunks ? encode(c, raw[m]) : 0);
    }
    for (int base = kKeep * Threads; base < chunks; base += Threads) {
        const int c = base + static_cast<int>(threadIdx.x);
        finish(c, c < chunks ? encode(c, load(c)) : 0);
    }
    return e;
}

// One CTA per column.
__global__ void __launch_bounds__(kEncodeThreads)
    encode_kernel(const std::uint16_t* __restrict__ x, int k, const std::int32_t* __restrict__ perm, int perm_block,
                  std::int32_t* __restrict__ encoded, std::int32_t* __restrict__ emax, int vec) {
    __shared__ unsigned warp_key[kEncodeThreads / 32];
    __shared__ int broadcast;
    pdl::enter(); // x is the previous kernel's output
    const int t = blockIdx.x;
    const std::uint16_t* v  = x + static_cast<std::size_t>(t) * k;
    std::int32_t* column    = encoded + static_cast<std::size_t>(t) * fast::encoded_column_words(k);
    const int e = vec != 0 ? encode_column_vec<kEncodeThreads>(v, k, perm, perm_block, column, warp_key, &broadcast)
                           : encode_column<kEncodeThreads>(v, k, perm, perm_block, column, warp_key, &broadcast);
    if (threadIdx.x == 0) { emax[t] = e; }
}

__device__ __forceinline__ void shfl_add(ggml::I128& r, int offset) {
    const auto lo = __shfl_xor_sync(0xFFFFFFFFU, static_cast<unsigned long long>(r.lo), offset);
    const auto hi = __shfl_xor_sync(0xFFFFFFFFU, static_cast<long long>(r.hi), offset);
    ggml::add(r, ggml::I128{static_cast<std::uint64_t>(lo), static_cast<std::int64_t>(hi)});
}

// Tiled (ggml_tiles16_v1): the same lanes, slices and arithmetic over a tiled matrix, whose row's
// blocks are 16 apart (row r of tile r / 16 at (r % 16) block_bytes; slice s's block at s / 8 * 16
// block_bytes past it): fast::slice reads block s / 8 at row + s / 8 * block_bytes, so the row
// pointer of slice s is offset by s / 8 * 15 block_bytes.
template <QType F, bool OutF32, int Lanes, int C, bool Tiled = false>
__global__ void __launch_bounds__(kThreads)
    rows_kernel(RowsGroup g, int row_bytes, int slices, const std::int32_t* __restrict__ encoded,
                const std::int32_t* __restrict__ emax, int columns) {
    constexpr int kRowsPerWarp = 32 / Lanes;
    constexpr std::size_t kBlockBytes = ggml_block_bytes(F);
    const GroupSlot seg             = group_slot(g);
    const std::uint8_t* weights     = seg.weights;
    const int rows = seg.rows, n = seg.n, out_offset = seg.out_offset;
    void* const out = seg.out;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int part = lane % Lanes;
    const int row  = (seg.block * kWarps + warp) * kRowsPerWarp + lane / Lanes;
    const bool active = row < rows;
    const int r = active ? row : 0;
    const std::uint8_t* w = Tiled ? weights + static_cast<std::size_t>(r / 16) * 16 * row_bytes + (r % 16) * kBlockBytes
                                  : weights + static_cast<std::size_t>(r) * row_bytes;
    const auto row_of = [&](int s) { // the row pointer fast::slice takes for slice s
        if constexpr (Tiled) {
            static_assert(ggml_block_elems(F) == 256, "tiled rows: 256-element blocks");
            return w + static_cast<std::size_t>(s / 8) * 15 * kBlockBytes;
        } else {
            return w;
        }
    };
    const std::size_t column_words = static_cast<std::size_t>(slices) * (fast::kSliceWords + 1);
    // A programmatic dependent of the encoder (or of the previous segment's kernel): the lane's first
    // slice is read from the immutable weights while the encoder still runs.
    fast::Slice first{};
    if (active) { first = fast::slice<F>(row_of(part), part); }
    pdl::wait_for_dependencies();
    for (int base = 0; base < columns; base += C) {
        const int count = min(C, columns - base);
        ggml::I128 acc[C];
        if (active) {
            for (int s = part; s < slices; s += Lanes) {
                const fast::Slice ws       = base == 0 && s == part ? first : fast::slice<F>(row_of(s), s);
                const fast::SliceScale sc  = fast::slice_scale(ws);
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    if (c < count) { // uniform across the warp
                        const std::int32_t* column = encoded + static_cast<std::size_t>(base + c) * column_words;
                        fast::add_slice(acc[c], ws, sc, column + s * fast::kSliceWords,
                                        fast::kMin<F> ? column[slices * fast::kSliceWords + s] : 0);
                    }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            if (c < count) {
#pragma unroll
                for (int offset = Lanes / 2; offset >= 1; offset >>= 1) { shfl_add(acc[c], offset); }
                if (active && part == 0) {
                    const int t = base + c;
                    const std::size_t at = static_cast<std::size_t>(t) * n + out_offset + row;
                    if constexpr (OutF32) {
                        static_cast<float*>(out)[at] = ggml::output_f32(acc[c], emax[t], F);
                    } else {
                        static_cast<std::uint16_t*>(out)[at] = ggml::output_bf16(acc[c], emax[t], F);
                    }
                }
            }
        }
    }
}

// Narrow calls of kMmaMinColumns..kMmaMaxColumns columns (MTP verification rounds) on tensor cores
// (ops/ggml/ggml_mma_rows.cuh): one CTA per 16 rows, its warps taking K's blocks in turn; the outputs
// are the same integers as rows_kernel's. (Measured at 5 columns, 2026-10-12: four warps taking
// whole blocks beat eight, and eight or sixteen taking 2- or 4-slice parts of a block, at every dense
// shape, the 1,280- and 2,560-row ones included.)
constexpr int kMmaRows       = 16;
constexpr int kMmaWarps      = 4;
constexpr int kMmaMinColumns = 3;
constexpr int kMmaMaxColumns = 8;

static_assert(kMmaWarps > 1, "the mma rows' warps reduce through shared memory");
using MmaPartial = unsigned long long[kMmaWarps - 1][4][2][32];

// The mma rows' weight paths: 2-byte loads from the rows (kPlain: the mixed-format launch), or the
// rows' units staged through shared memory by cp.async (kStaged: single-format launches; MTP 4 +
// n-gram 7 at 4K 202.4 -> 206.7 tok/s, two rounds, 2026-10-16).
enum : int { kPlain = 0, kStaged = 1 };

// One CTA of the mma rows: segment `seg` (rows row_bytes apart) of its launch.
template <QType F, bool OutF32, int Stage>
__device__ __forceinline__ void mma_rows_body(const GroupSlot& seg, int row_bytes, int slices,
                                              const std::int32_t* __restrict__ encoded, const std::int32_t* __restrict__ emax,
                                              int columns, MmaPartial& partial) {
    constexpr int Warps        = kMmaWarps;
    constexpr int kBlockSlices = ggml::tile::kBlockSlices<F>;
    const std::uint8_t* weights = seg.weights;
    const int rows = seg.rows, n = seg.n, out_offset = seg.out_offset;
    void* const out = seg.out;
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int g = lane / 4, t = lane % 4;
    const int row0 = seg.block * kMmaRows;
    const std::uint8_t* wg  = weights + static_cast<std::size_t>(min(row0 + g, rows - 1)) * row_bytes;
    const std::uint8_t* wg8 = weights + static_cast<std::size_t>(min(row0 + g + 8, rows - 1)) * row_bytes;
    const std::size_t column_words = static_cast<std::size_t>(slices) * (fast::kSliceWords + 1);
    // B: column g's limb words 2t, 2t + 1 (columns past the call's repeat its last: ignored outputs).
    const std::int32_t* bcol = encoded + static_cast<std::size_t>(min(g, columns - 1)) * column_words + 2 * t;
    // The min term's slice sums of the lane's output columns 2t, 2t + 1.
    const std::int32_t* xs0 = encoded + static_cast<std::size_t>(min(2 * t, columns - 1)) * column_words +
                              static_cast<std::size_t>(slices) * fast::kSliceWords;
    const std::int32_t* xs1 = encoded + static_cast<std::size_t>(min(2 * t + 1, columns - 1)) * column_words +
                              static_cast<std::size_t>(slices) * fast::kSliceWords;
    ggml::I128 acc[4];
    if constexpr (Stage == kStaged) {
        __shared__ ggml::mma_rows::StagedRows<F> staged[Warps];
        const auto row = [&](int r) { return weights + static_cast<std::size_t>(min(row0 + r, rows - 1)) * row_bytes; };
        // The warp's first unit is immutable weights: copied while the producer still runs.
        ggml::mma_rows::issue_first_unit<F>(staged[warp], row, lane, warp, slices);
        pdl::wait_for_dependencies();
        ggml::mma_rows::accumulate_staged<F>(acc, staged[warp], row, bcol, xs0, xs1, t, g, lane, warp, slices, Warps, true);
    } else {
        pdl::wait_for_dependencies();
        ggml::mma_rows::accumulate<F>(acc, wg, wg8, bcol, xs0, xs1, t, warp, slices / kBlockSlices, Warps);
    }
    ggml::mma_rows::reduce<Warps>(acc, partial, warp, lane);
    if (warp > 0) { return; }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int row = row0 + g + 8 * (q >> 1), c = 2 * t + (q & 1);
        if (row < rows && c < columns) {
            const std::size_t at = static_cast<std::size_t>(c) * n + out_offset + row;
            if constexpr (OutF32) {
                static_cast<float*>(out)[at] = ggml::output_f32(acc[q], emax[c], F);
            } else {
                static_cast<std::uint16_t*>(out)[at] = ggml::output_bf16(acc[q], emax[c], F);
            }
        }
    }
}

template <QType F, bool OutF32, int Stage = kStaged>
__global__ void __launch_bounds__(kMmaWarps * 32, 4)
    mma_rows_kernel(RowsGroup grp, int row_bytes, int slices, const std::int32_t* __restrict__ encoded,
                    const std::int32_t* __restrict__ emax, int columns) {
    __shared__ MmaPartial partial;
    mma_rows_body<F, OutF32, Stage>(group_slot(grp), row_bytes, slices, encoded, emax, columns, partial);
}

// A verification call's segments of two formats in one launch, each CTA running its segment's format
// (RowsGroup::format 0: F0, 1: F1): end to end 1.0 % faster than a launch per format (two rounds,
// 2026-10-16).
template <QType F0, QType F1, bool OutF32>
__global__ void __launch_bounds__(kMmaWarps * 32, 4)
    mma_rows_mixed_kernel(RowsGroup grp, int slices, const std::int32_t* __restrict__ encoded,
                          const std::int32_t* __restrict__ emax, int columns) {
    __shared__ MmaPartial partial;
    const GroupSlot seg = group_slot(grp);
    if (seg.format == 0) {
        mma_rows_body<F0, OutF32, kPlain>(seg, seg.row_bytes, slices, encoded, emax, columns, partial);
    } else {
        mma_rows_body<F1, OutF32, kPlain>(seg, seg.row_bytes, slices, encoded, emax, columns, partial);
    }
}

// ------------------------------------------------------------------ tiled rows (ggml_tiles16_v1)
//
// A ggml_tiles16_v1 matrix (GgmlLayout::Tiles16; formats of 256-element blocks) stores its rows in
// tiles of 16: tile i at 16 i row_bytes, and block b of the tile's row r at (16 b + r) kBlockBytes,
// so the tile's 16 rows' block b are one contiguous, 16-byte aligned run (rows padded to a whole
// tile with zero blocks, d = 0). Narrow calls run the tensor-core tiled kernel: a CTA of W warps
// holds W / split tiles, each tile's `split` warps taking its blocks in turn (b = part, part + split,
// ...); a warp stages its blocks' runs by 16-byte cp.async (L2 evict-first) into a ring of `stages`
// slots (the first stages - 1 before the programmatic wait: the weights are immutable) and runs
// mma_rows::unit_sums over each, mma_rows_kernel's arithmetic; the tile's warps add their int128
// partials through shared memory (exact, in any order). A call's segments of two tiled formats share
// one launch (tiled_mma_mixed_kernel). One-column Q6_K and IQ4_XS calls of one format run rows_kernel
// with tiled addressing instead (tiled_rows_route); every route returns the same integers. Against
// the row-major routes (2026-10-16, RTX 5090, the IQ3_S import's dense calls weighted per token):
// -12 / -23 / -26 / -27 / -32 % at 1 / 2 / 4 / 5 / 8 columns; end to end MTP 4 + n-gram 7 at 4K
// 221.7-222.6 against 214.0-214.5 tok/s, plain decode 99.6-99.7 against 99.5-99.6.
template <QType F>
inline constexpr bool kTiledFormat = ggml_tiles16_supported(F);
template <QType F>
inline constexpr int kTiledRun = 16 * static_cast<int>(ggml_block_bytes(F)); // one block of a tile's 16 rows
constexpr int kTiledMaxThreads = 256;

// A 16-byte cp.async with an L2 evict-first policy: the dense weights (about 1.3 GB a token) are
// read once per call and never reused from L2, so their stream should not push out what other
// kernels reuse (MTP 4 + n-gram 7 at 4K +0.5 %, both runs ahead, 2026-10-16; the head's evict-first
// loads +0.3 %, design §19.3.20d).
__device__ __forceinline__ void cp_async16_evict_first(void* shared, const void* global, std::uint64_t policy) {
    const auto s = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(s), "l"(global), "l"(policy));
}

// Waits until at most n of the warp's cp.async groups are pending (n < 4).
__device__ __forceinline__ void cp_async_wait_upto(int n) {
    switch (n) {
    case 0: ggml::tile::cp_async_wait<0>(); break;
    case 1: ggml::tile::cp_async_wait<1>(); break;
    case 2: ggml::tile::cp_async_wait<2>(); break;
    default: ggml::tile::cp_async_wait<3>(); break;
    }
}

// Dynamic shared memory of a tiled launch: the warps' rings of `slot_bytes` slots, then the partials
// (split > 1).
std::size_t tiled_shared_bytes(int warps, int split, int stages, int slot_bytes) {
    const std::size_t rings    = static_cast<std::size_t>(warps) * stages * slot_bytes;
    const std::size_t partials = split > 1 ? static_cast<std::size_t>(warps / split) * (split - 1) * 32 * 4 * sizeof(ggml::I128) : 0;
    return rings + partials;
}

// One CTA of the tiled kernel: segment `seg` (seg.row_bytes per row) of its launch, ring slots of
// slot_bytes (the launch's largest run).
template <QType F, bool OutF32>
__device__ __forceinline__ void tiled_mma_body(const GroupSlot& seg, int slices, const std::int32_t* __restrict__ encoded,
                                               const std::int32_t* __restrict__ emax, int columns, int split, int stages,
                                               int slot_bytes, unsigned char* shared) {
    static_assert(kTiledFormat<F>, "tiled rows: formats of 256-element blocks");
    constexpr int kBlockBytes = static_cast<int>(ggml_block_bytes(F));
    constexpr int kRun        = kTiledRun<F>;
    static_assert(kRun % 16 == 0, "a tile's block run is whole 16-byte copies");
    void* const out = seg.out;
    const int warps = static_cast<int>(blockDim.x) / 32, warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int per_cta = warps / split, slot = warp / split, part = warp % split;
    const int tiles = (seg.rows + 15) / 16, tile = seg.block * per_cta + slot;
    const bool active = tile < tiles;
    const int blocks  = slices / 8;
    const int mine    = active && part < blocks ? (blocks - part + split - 1) / split : 0; // the warp's blocks
    const std::uint8_t* tile_base = seg.weights + static_cast<std::size_t>(active ? tile : 0) * 16 * seg.row_bytes;
    unsigned char* ring = shared + static_cast<std::size_t>(warp) * stages * slot_bytes;
    const std::size_t column_words = static_cast<std::size_t>(slices) * (fast::kSliceWords + 1);
    std::uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    const auto issue = [&](int i) { // the warp's i-th block into its ring slot, one cp.async group
        if (i < mine) {
            const std::uint8_t* src = tile_base + static_cast<std::size_t>(part + i * split) * kRun;
            unsigned char* dst      = ring + (i % stages) * slot_bytes;
            for (int c = lane; c < kRun / 16; c += 32) { cp_async16_evict_first(dst + 16 * c, src + 16 * c, policy); }
        }
        ggml::tile::cp_async_commit();
    };
    // MMA fragments (mma_rows_body): lane (g, t), column g's limb words 2t, 2t + 1.
    const int g = lane / 4, t = lane % 4;
    const std::int32_t* bcol = encoded + static_cast<std::size_t>(min(g, columns - 1)) * column_words + 2 * t;
    const std::int32_t* xs0  = encoded + static_cast<std::size_t>(min(2 * t, columns - 1)) * column_words +
                              static_cast<std::size_t>(slices) * fast::kSliceWords;
    const std::int32_t* xs1 = encoded + static_cast<std::size_t>(min(2 * t + 1, columns - 1)) * column_words +
                              static_cast<std::size_t>(slices) * fast::kSliceWords;
    ggml::I128 acc[4];
    for (int i = 0; i < stages - 1; ++i) { issue(i); }
    pdl::wait_for_dependencies();
    for (int i = 0; i < mine; ++i) {
        __syncwarp(); // slot (i - 1) % stages, which the next issue fills, was read at i - 1
        issue(i + stages - 1);
        cp_async_wait_upto(stages - 1); // block i has landed
        __syncwarp();
        const unsigned char* run = ring + (i % stages) * slot_bytes;
        ggml::mma_rows::unit_sums<F>(acc, run + g * kBlockBytes, run + (g + 8) * kBlockBytes, part + i * split, 8, bcol, xs0,
                                     xs1, t);
    }
    ggml::tile::cp_async_wait<0>();
    // The tile's other warps' partials into part 0 (exact, any order).
    if (split > 1) {
        auto* partial = reinterpret_cast<unsigned long long*>(shared + static_cast<std::size_t>(warps) * stages * slot_bytes);
        const auto at = [&](int p, int q) {
            return partial + ((static_cast<std::size_t>(slot) * (split - 1) + (p - 1)) * 32 + lane) * 8 + 2 * q;
        };
        if (part > 0) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                at(part, q)[0] = acc[q].lo;
                at(part, q)[1] = static_cast<unsigned long long>(acc[q].hi);
            }
        }
        __syncthreads();
        if (part == 0) {
            for (int p = 1; p < split; ++p) {
#pragma unroll
                for (int q = 0; q < 4; ++q) { ggml::add(acc[q], ggml::I128{at(p, q)[0], static_cast<std::int64_t>(at(p, q)[1])}); }
            }
        }
    }
    if (part != 0 || !active) { return; }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int row = tile * 16 + g + 8 * (q >> 1), c = 2 * t + (q & 1);
        if (row < seg.rows && c < columns) {
            const std::size_t o = static_cast<std::size_t>(c) * seg.n + seg.out_offset + row;
            if constexpr (OutF32) {
                static_cast<float*>(out)[o] = ggml::output_f32(acc[q], emax[c], F);
            } else {
                static_cast<std::uint16_t*>(out)[o] = ggml::output_bf16(acc[q], emax[c], F);
            }
        }
    }
}

template <QType F, bool OutF32>
__global__ void __launch_bounds__(kTiledMaxThreads, 2)
    tiled_mma_kernel(RowsGroup grp, int slices, const std::int32_t* __restrict__ encoded,
                     const std::int32_t* __restrict__ emax, int columns, int split, int stages, int slot_bytes) {
    extern __shared__ __align__(16) unsigned char tiled_shared[];
    tiled_mma_body<F, OutF32>(group_slot(grp), slices, encoded, emax, columns, split, stages, slot_bytes, tiled_shared);
}

// A call's segments of two tiled formats in one launch, each CTA running its segment's format
// (RowsGroup::format 0: F0, 1: F1).
template <QType F0, QType F1, bool OutF32>
__global__ void __launch_bounds__(kTiledMaxThreads, 2)
    tiled_mma_mixed_kernel(RowsGroup grp, int slices, const std::int32_t* __restrict__ encoded,
                           const std::int32_t* __restrict__ emax, int columns, int split, int stages, int slot_bytes) {
    extern __shared__ __align__(16) unsigned char tiled_shared[];
    const GroupSlot seg = group_slot(grp);
    if (seg.format == 0) {
        tiled_mma_body<F0, OutF32>(seg, slices, encoded, emax, columns, split, stages, slot_bytes, tiled_shared);
    } else {
        tiled_mma_body<F1, OutF32>(seg, slices, encoded, emax, columns, split, stages, slot_bytes, tiled_shared);
    }
}


// A decoded plane of one segment (or a slab of its rows): for each tile of kTileRows rows (the last
// padded with the last row), the images of its K stages in order (ggml::tile::CodeStage, written
// whole for every format). One CTA per (row tile, stage). A Tiled (ggml_tiles16_v1) matrix's row
// pointer is offset so that fast::chunk's block of the stage (stages lie inside one 256-element
// block) lands on the row's block in its tile.
template <QType F, bool Tiled>
__global__ void __launch_bounds__(ggml::tile::kThreads)
    plane_kernel(const std::uint8_t* __restrict__ weights, int rows, int row_bytes, std::uint8_t* __restrict__ plane) {
    __shared__ ggml::tile::CodeStage cs;
    const int tile = blockIdx.x, st = blockIdx.y, stages = gridDim.y;
    const auto row_of = [&](int r) {
        const int row = min(tile * ggml::tile::kTileRows + r, rows - 1);
        if constexpr (Tiled) {
            static_assert(ggml::tile::kBlockSlices<F> % ggml::tile::kStageSlices == 0, "a stage inside one block");
            const int b = st * ggml::tile::kStageSlices / ggml::tile::kBlockSlices<F>;
            return weights + static_cast<std::size_t>(row / 16) * 16 * row_bytes +
                   static_cast<std::size_t>(row % 16 + 15 * b) * ggml_block_bytes(F);
        } else {
            return weights + static_cast<std::size_t>(row) * row_bytes;
        }
    };
    ggml::tile::decode_stage<F>(cs, row_of, st);
    __syncthreads();
    auto* dst      = reinterpret_cast<int4*>(plane + (static_cast<std::size_t>(tile) * stages + st) * sizeof(cs));
    const auto* src = reinterpret_cast<const int4*>(&cs);
    for (int i = threadIdx.x; i < ggml::tile::kImageBytes<F> / 16; i += blockDim.x) { dst[i] = src[i]; }
}

// Wide calls: one CTA per tile of ggml::tile::kTileRows rows x kTileColumns encoded columns, its
// stages the images of the segment's decoded plane (rows still give the scales' accumulator choice).
// Measured at 4096 columns against tiles that decode their raw rows in the loop (2026-10-12): 15-34 %
// faster (Q6_K attn_qkv 3,912 -> 2,839 us). Tiles of Q6_K and IQ4_NL fit two CTAs per SM (128
// registers): Q6_K 3-6 % and IQ4_NL 14 % faster than one; Q4_K and Q5_K (min-term accumulators)
// spill and run 60-70 % slower, IQ4_XS 9 % slower.
// The other formats' tiles have 512 threads (16 warps of 16 x 16 outputs, one CTA per SM within 128
// registers; ggml_tile.cuh), measured at 4096 columns against 256 (2026-10-14): Q5_K -2.4 %, Q4_K
// -3.6 %, IQ4_XS -4.1 % time; Q6_K +9-12 % and IQ4_NL +3 %, which keep 256 at two CTAs per SM.
// (Q2_0 and the IQ2/IQ3 formats, which no dense projection of the IQ3_S import uses, unmeasured.)
template <QType F>
inline constexpr int kPlaneTileThreads = F == QType::GGML_Q6_K || F == QType::GGML_IQ4_NL ? 256 : 512;
template <QType F>
inline constexpr int kPlaneTileBlocks = kPlaneTileThreads<F> == 512 ? 1 : F == QType::GGML_Q6_K || F == QType::GGML_IQ4_NL ? 2 : 1;

template <QType F, bool OutF32, bool Tiled>
__global__ void __launch_bounds__(kPlaneTileThreads<F>, kPlaneTileBlocks<F>)
    tile_kernel(const std::uint8_t* __restrict__ weights, int rows, int row_bytes, int slices,
                const std::int32_t* __restrict__ encoded, const std::int32_t* __restrict__ emax, int columns,
                void* __restrict__ out, int n, int out_offset, const std::uint8_t* __restrict__ plane) {
    extern __shared__ __align__(16) unsigned char tile_shared[];
    auto& sm       = *reinterpret_cast<ggml::tile::PlaneShared*>(tile_shared);
    const int row0 = blockIdx.x * ggml::tile::kTileRows, col0 = blockIdx.y * ggml::tile::kTileColumns;
    const std::size_t column_words = static_cast<std::size_t>(slices) * (fast::kSliceWords + 1);
    const auto limbs = [&](int c) { return encoded + static_cast<std::size_t>(min(col0 + c, columns - 1)) * column_words; };
    ggml::tile::Accumulators acc;
    // The raw rows (their scales choose the accumulator): a Tiled row's blocks are 16 apart.
    const auto row_of = [&](int r) {
        const int row = min(row0 + r, rows - 1);
        if constexpr (Tiled) {
            return weights + static_cast<std::size_t>(row / 16) * 16 * row_bytes + static_cast<std::size_t>(row % 16) * ggml_block_bytes(F);
        } else {
            return weights + static_cast<std::size_t>(row) * row_bytes;
        }
    };
    const int tile_columns = min(ggml::tile::kTileColumns, columns - col0);
    const ggml::tile::WarpTile wt = ggml::tile::warp_tile<kPlaneTileThreads<F>>(tile_columns);
    const std::size_t stages = static_cast<std::size_t>(slices / ggml::tile::kStageSlices);
    ggml::tile::compute_plane<F, kPlaneTileThreads<F>, Tiled ? 16 : 1>(sm, row_of, plane + blockIdx.x * stages * sizeof(ggml::tile::CodeStage),
                                                                      slices, limbs, slices * fast::kSliceWords, tile_columns, acc);
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int mi = 0; mi < ggml::tile::kMTiles; ++mi) {
#pragma unroll
        for (int nt = 0; nt < ggml::tile::kNTiles; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int row = row0 + ggml::tile::out_row(wt, lane, mi, q);
                const int t   = col0 + ggml::tile::out_column(wt, lane, nt, q);
                if (mi < wt.mtiles && nt < wt.ntiles && row < rows && t < columns) {
                    const std::size_t at = static_cast<std::size_t>(t) * n + out_offset + row;
                    if constexpr (OutF32) {
                        static_cast<float*>(out)[at] = ggml::output_f32(acc.r[mi][nt][q], emax[t], F);
                    } else {
                        static_cast<std::uint16_t*>(out)[at] = ggml::output_bf16(acc.r[mi][nt][q], emax[t], F);
                    }
                }
            }
        }
    }
}

// Lanes per row: a power of two in [4, 32] leaving each lane about four or more slices.
int lanes_for(int slices) {
    int lanes = 4;
    while (lanes < 32 && slices / (2 * lanes) >= 4) { lanes *= 2; }
    return lanes;
}

// The encoded columns of a batch.
struct Columns {
    const std::int32_t* encoded = nullptr;
    const std::int32_t* emax    = nullptr;
};

// A wide call's segment (or slab of one): the tensor-core tiles over its decoded plane.
template <QType F, bool OutF32, bool Tiled>
void launch_tile_format(const GgmlSegment& s, int k, const Columns& in, int columns, void* out, int n,
                        const std::uint8_t* plane, cudaStream_t stream) {
    const int row_bytes = static_cast<int>(k / ggml_block_elems(F) * ggml_block_bytes(F));
    const int slices    = k / 32;
    require(plane != nullptr, "a wide call's tiles read a decoded plane");
    const dim3 grid((s.rows + ggml::tile::kTileRows - 1) / ggml::tile::kTileRows,
                    (columns + ggml::tile::kTileColumns - 1) / ggml::tile::kTileColumns);
    constexpr int kShared = static_cast<int>(sizeof(ggml::tile::PlaneShared));
    static const bool configured = [] {
        return cudaFuncSetAttribute(tile_kernel<F, OutF32, Tiled>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared) ==
               cudaSuccess;
    }();
    require(configured, "the tile kernel's shared memory exceeds the device's");
    tile_kernel<F, OutF32, Tiled><<<grid, kPlaneTileThreads<F>, kShared, stream>>>(s.data, s.rows, row_bytes, slices, in.encoded,
                                                                                    in.emax, columns, out, n, s.out_offset, plane);
    check_launch("tile");
}

// ---- tiled narrow calls: the route and the CTA shape of a launch
//
// A tiled MMA launch takes its CTA shape (warps, K split; two ring slots each) from its tiles, its
// rows' blocks and the device's residency at the shape's shared memory: the first of kTiledShapes
// whose CTAs all fit in one wave (cudaOccupancyMaxActiveBlocksPerMultiprocessor x SMs) and whose
// split either keeps the call within kTiledWarpBudget warps or leaves each warp kTiledMinBlocks
// blocks; a call no shape fits in one wave (the head) takes 4 warps, unsplit. Measured 2026-10-16
// (RTX 5090, 170 SMs; the IQ3_S import's dense calls at 1-8 columns, every shape): a second wave of
// even two CTAs costs a call 25-30 % (GDN q/k/v/z, 1,024 tiles: 4 warps split 2, 512 CTAs against
// 510 resident, 28.8 us at one column; unsplit, 21.9); within one wave more warps help small calls
// (128 tiles: split 8, 7.6 us; split 2, 10.0), while split 4 over 2.5 blocks a warp loses to split 2
// (384 tiles: 13.3 against 13.0 us).
struct TiledShape {
    int warps  = 4;
    int split  = 1;
    int stages = 2;
};
constexpr TiledShape kTiledShapes[] = {{8, 8, 2}, {4, 4, 2}, {2, 2, 2}, {4, 2, 2}, {4, 1, 2}, {2, 1, 2}};
constexpr TiledShape kTiledManyWaves{4, 1, 2};
constexpr int kTiledWarpBudget = 1024;
constexpr int kTiledMinBlocks  = 4;

// The device's SMs and opt-in shared memory per block, and `kernel`'s resident CTAs per SM at
// `threads` threads and `shared` bytes (cached per thread: the answers never change).
struct TiledDevice {
    int sms   = 0;
    int limit = 0;
};

TiledDevice tiled_device() {
    thread_local int device = -1;
    thread_local TiledDevice d;
    int current = 0;
    CUDA_CHECK(cudaGetDevice(&current));
    if (current != device) {
        CUDA_CHECK(cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, current));
        CUDA_CHECK(cudaDeviceGetAttribute(&d.limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, current));
        device = current;
    }
    return d;
}

// Lets `kernel` take up to the device's opt-in shared memory (once per kernel and thread).
template <class Kernel>
void tiled_allow_shared(Kernel kernel, int limit) {
    thread_local std::vector<const void*> done;
    const void* key = reinterpret_cast<const void*>(kernel);
    if (std::find(done.begin(), done.end(), key) != done.end()) { return; }
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, limit));
    done.push_back(key);
}

template <class Kernel>
int tiled_resident(Kernel kernel, int threads, std::size_t shared) {
    struct Entry {
        const void* kernel;
        int threads;
        std::size_t shared;
        int resident;
    };
    thread_local std::vector<Entry> cache;
    const void* key = reinterpret_cast<const void*>(kernel);
    for (const Entry& e : cache) {
        if (e.kernel == key && e.threads == threads && e.shared == shared) { return e.resident; }
    }
    int resident = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, threads, shared));
    cache.push_back({key, threads, shared, resident});
    return resident;
}

template <class Kernel>
TiledShape tiled_shape(Kernel kernel, int tiles, int blocks, int slot_bytes, const TiledDevice& dev) {
    for (const TiledShape& s : kTiledShapes) {
        if (s.split > 1 && tiles * s.split > kTiledWarpBudget && blocks < kTiledMinBlocks * s.split) { continue; }
        const std::size_t shared = tiled_shared_bytes(s.warps, s.split, s.stages, slot_bytes);
        if (shared > static_cast<std::size_t>(dev.limit)) { continue; }
        const int per_cta = s.warps / s.split;
        const int ctas    = (tiles + per_cta - 1) / per_cta;
        if (ctas <= tiled_resident(kernel, 32 * s.warps, shared) * dev.sms) { return s; }
    }
    return kTiledManyWaves;
}

// The rows route (rows_kernel with tiled addressing) takes a single-format tiled group of Q6_K or
// IQ4_XS at one column and every tiled call wider than kMmaMaxColumns; the MMA route takes the rest
// and every two-format call of 1-8 columns (one launch). Measured at one column (2026-10-16): the GDN
// output projection (Q6_K) 13.0 us against 15.5 by MMA, the head 295.5 against 297.5; a 6,144-row
// Q4_K matrix 11.1 by MMA against 13.3 by rows; GDN q/k/v/z 22.1 in one MMA launch against 26.0 in
// a rows launch per format.
bool tiled_rows_route(QType format, int columns) {
    return columns > kMmaMaxColumns || (columns == 1 && (format == QType::GGML_Q6_K || format == QType::GGML_IQ4_XS));
}

// rows_kernel over a tiled group (launch_narrow_format's rows route, Tiled addressing).
template <QType F, bool OutF32>
void launch_rows_tiled(RowsGroup g, int k, const Columns& in, int columns, cudaStream_t stream) {
    const int row_bytes = static_cast<int>(k / ggml_block_elems(F) * ggml_block_bytes(F));
    const int slices    = k / 32;
    const int lanes     = lanes_for(slices);
    const int per_cta   = kWarps * (32 / lanes);
    g.first[0]          = 0;
    for (int i = 0; i < g.count; ++i) { g.first[i + 1] = g.first[i] + (g.rows[i] + per_cta - 1) / per_cta; }
    const dim3 grid(g.first[g.count]);
#define INFERNIX_GGML_TILED_LANES(LN)                                                                                  \
    case LN:                                                                                                           \
        if (columns > kColumns && columns <= 2 * kColumns) {                                                           \
            CUDA_CHECK(pdl::launch_consumer({grid, dim3(kThreads), 0, stream},                                         \
                                            rows_kernel<F, OutF32, LN, 2 * kColumns, true>, g, row_bytes, slices,       \
                                            in.encoded, in.emax, columns));                                             \
        } else {                                                                                                       \
            CUDA_CHECK(pdl::launch_consumer({grid, dim3(kThreads), 0, stream}, rows_kernel<F, OutF32, LN, kColumns, true>, \
                                            g, row_bytes, slices, in.encoded, in.emax, columns));                       \
        }                                                                                                              \
        break;
    switch (lanes) {
        INFERNIX_GGML_TILED_LANES(4)
        INFERNIX_GGML_TILED_LANES(8)
        INFERNIX_GGML_TILED_LANES(16)
        INFERNIX_GGML_TILED_LANES(32)
    default:
        throw std::logic_error("ggml_linear: lanes");
    }
#undef INFERNIX_GGML_TILED_LANES
    check_launch("rows tiled");
}

// The MMA route over a tiled group (its row_bytes and, for a mixed launch, format filled): the CTA
// prefix, the shape and the shared memory for `kernel`, whose ring slots hold `slot_bytes`.
template <class Kernel>
void launch_tiled_mma(Kernel kernel, RowsGroup g, int k, const Columns& in, int columns, int slot_bytes, cudaStream_t stream) {
    int tiles = 0;
    for (int i = 0; i < g.count; ++i) { tiles += (g.rows[i] + 15) / 16; }
    const TiledDevice dev = tiled_device();
    tiled_allow_shared(kernel, dev.limit);
    const TiledShape shape = tiled_shape(kernel, tiles, k / 256, slot_bytes, dev);
    const int per_cta      = shape.warps / shape.split;
    g.first[0]             = 0;
    for (int i = 0; i < g.count; ++i) { g.first[i + 1] = g.first[i] + ((g.rows[i] + 15) / 16 + per_cta - 1) / per_cta; }
    const std::size_t shared = tiled_shared_bytes(shape.warps, shape.split, shape.stages, slot_bytes);
    CUDA_CHECK(pdl::launch_consumer({dim3(g.first[g.count]), dim3(32 * shape.warps), shared, stream}, kernel, g, k / 32,
                                    in.encoded, in.emax, columns, shape.split, shape.stages, slot_bytes));
    check_launch("tiled mma");
}

template <QType F, bool OutF32>
void launch_tiled_format(RowsGroup g, int k, const Columns& in, int columns, cudaStream_t stream) {
    if constexpr (!kTiledFormat<F>) {
        throw std::logic_error("ggml_linear: a tiled segment's format has no tiled kernels");
    } else {
        if (tiled_rows_route(F, columns)) {
            launch_rows_tiled<F, OutF32>(g, k, in, columns, stream);
            return;
        }
        for (int i = 0; i < g.count; ++i) { g.row_bytes[i] = static_cast<int>(k / ggml_block_elems(F) * ggml_block_bytes(F)); }
        launch_tiled_mma(tiled_mma_kernel<F, OutF32>, g, k, in, columns, kTiledRun<F>, stream);
    }
}

// A call's segments of two tiled formats (F0 before F1 in QType order) in one MMA launch over a group
// whose format[] holds 0 or 1 and row_bytes[] each segment's rows.
template <QType F0, QType F1>
void launch_tiled_mixed_pair(const RowsGroup& g, int k, const Columns& in, int columns, cudaStream_t stream) {
    launch_tiled_mma(tiled_mma_mixed_kernel<F0, F1, false>, g, k, in, columns, std::max(kTiledRun<F0>, kTiledRun<F1>), stream);
}

bool launch_tiled_mixed(QType f0, QType f1, const RowsGroup& g, int k, const Columns& in, int columns, cudaStream_t stream) {
#define INFERNIX_GGML_TILED_PAIR(A, B)                                                                                 \
    if (f0 == QType::A && f1 == QType::B) {                                                                            \
        launch_tiled_mixed_pair<QType::A, QType::B>(g, k, in, columns, stream);                                        \
        return true;                                                                                                   \
    }
    INFERNIX_GGML_TILED_PAIR(GGML_Q4_K, GGML_Q5_K)
    INFERNIX_GGML_TILED_PAIR(GGML_Q4_K, GGML_Q6_K)
    INFERNIX_GGML_TILED_PAIR(GGML_Q4_K, GGML_IQ4_XS)
    INFERNIX_GGML_TILED_PAIR(GGML_Q5_K, GGML_Q6_K)
    INFERNIX_GGML_TILED_PAIR(GGML_Q5_K, GGML_IQ4_XS)
    INFERNIX_GGML_TILED_PAIR(GGML_Q6_K, GGML_IQ4_XS)
#undef INFERNIX_GGML_TILED_PAIR
    return false;
}


// A narrow call's group of segments of format F (RowsGroup; one segment without the switch): the
// mma rows of a verification round, else the rows kernel. Fills the group's CTA prefix.
template <QType F, bool OutF32>
void launch_narrow_format(RowsGroup g, int k, const Columns& in, int columns, cudaStream_t stream) {
    const int row_bytes = static_cast<int>(k / ggml_block_elems(F) * ggml_block_bytes(F));
    const int slices    = k / 32;
    const bool mma      = columns >= kMmaMinColumns && columns <= kMmaMaxColumns;
    const int lanes     = lanes_for(slices);
    const int per_cta   = mma ? kMmaRows : kWarps * (32 / lanes);
    g.first[0]          = 0;
    for (int i = 0; i < g.count; ++i) { g.first[i + 1] = g.first[i] + (g.rows[i] + per_cta - 1) / per_cta; }
    const dim3 grid(g.first[g.count]);
    if (mma) {
        CUDA_CHECK(pdl::launch_consumer({grid, dim3(kMmaWarps * 32), 0, stream}, mma_rows_kernel<F, OutF32, kStaged>,
                                        g, row_bytes, slices, in.encoded, in.emax, columns));
        check_launch("mma rows");
        return;
    }
    // Up to kColumns columns in one pass; 5-8 (an MTP verification round) in one pass of eight, so
    // the weights are read once.
#define INFERNIX_GGML_LANES(LN)                                                                                        \
    case LN:                                                                                                           \
        if (columns > kColumns && columns <= 2 * kColumns) {                                                           \
            CUDA_CHECK(pdl::launch_consumer({grid, dim3(kThreads), 0, stream},                                         \
                                            rows_kernel<F, OutF32, LN, 2 * kColumns>, g, row_bytes, slices,             \
                                            in.encoded, in.emax, columns));                                      \
        } else {                                                                                                       \
            CUDA_CHECK(pdl::launch_consumer({grid, dim3(kThreads), 0, stream}, rows_kernel<F, OutF32, LN, kColumns>,    \
                                            g, row_bytes, slices, in.encoded, in.emax, columns));                \
        }                                                                                                              \
        break;
    switch (lanes) {
        INFERNIX_GGML_LANES(4)
        INFERNIX_GGML_LANES(8)
        INFERNIX_GGML_LANES(16)
        INFERNIX_GGML_LANES(32)
    default:
        throw std::logic_error("ggml_linear: lanes");
    }
#undef INFERNIX_GGML_LANES
    check_launch("rows");
}

// The formats a mixed verification launch may pair (the dense projections of the GSQ-RCO IQ3_S
// import), and the launch of one pair (F0 before F1 in QType order) over a group whose format[] holds
// 0 or 1.
bool mixed_format(QType f) {
    return f == QType::GGML_Q4_K || f == QType::GGML_Q5_K || f == QType::GGML_Q6_K || f == QType::GGML_IQ4_XS;
}

template <QType F0, QType F1>
void launch_mixed_pair(RowsGroup g, int k, const Columns& in, int columns, cudaStream_t stream) {
    g.first[0] = 0;
    for (int i = 0; i < g.count; ++i) { g.first[i + 1] = g.first[i] + (g.rows[i] + kMmaRows - 1) / kMmaRows; }
    CUDA_CHECK(pdl::launch_consumer({dim3(g.first[g.count]), dim3(kMmaWarps * 32), 0, stream},
                                    mma_rows_mixed_kernel<F0, F1, false>, g, k / 32, in.encoded, in.emax, columns));
    check_launch("mma rows mixed");
}

bool launch_mixed(QType f0, QType f1, const RowsGroup& g, int k, const Columns& in, int columns, cudaStream_t stream) {
#define INFERNIX_GGML_PAIR(A, B)                                                                                       \
    if (f0 == QType::A && f1 == QType::B) {                                                                            \
        launch_mixed_pair<QType::A, QType::B>(g, k, in, columns, stream);                                              \
        return true;                                                                                                   \
    }
    INFERNIX_GGML_PAIR(GGML_Q4_K, GGML_Q5_K)
    INFERNIX_GGML_PAIR(GGML_Q4_K, GGML_Q6_K)
    INFERNIX_GGML_PAIR(GGML_Q4_K, GGML_IQ4_XS)
    INFERNIX_GGML_PAIR(GGML_Q5_K, GGML_Q6_K)
    INFERNIX_GGML_PAIR(GGML_Q5_K, GGML_IQ4_XS)
    INFERNIX_GGML_PAIR(GGML_Q6_K, GGML_IQ4_XS)
#undef INFERNIX_GGML_PAIR
    return false;
}

#define INFERNIX_GGML_FORMATS(X)                                                                                       \
    X(QType::GGML_Q2_0)                                                                                                \
    X(QType::GGML_Q4_K)                                                                                                \
    X(QType::GGML_Q5_K)                                                                                                \
    X(QType::GGML_Q6_K)                                                                                                \
    X(QType::GGML_IQ2_S)                                                                                               \
    X(QType::GGML_IQ3_XXS)                                                                                             \
    X(QType::GGML_IQ3_S)                                                                                               \
    X(QType::GGML_IQ4_XS)                                                                                              \
    X(QType::GGML_IQ4_NL)

template <bool OutF32>
void launch_tiles(const GgmlSegment& s, int k, const Columns& in, int columns, void* out, int n,
                  const std::uint8_t* plane, cudaStream_t stream) {
#define INFERNIX_GGML_TILES(Q)                                                                                         \
    case Q:                                                                                                            \
        if (s.layout == GgmlLayout::Tiles16) {                                                                         \
            if constexpr (kTiledFormat<Q>) {                                                                           \
                launch_tile_format<Q, OutF32, true>(s, k, in, columns, out, n, plane, stream);                         \
                break;                                                                                                 \
            }                                                                                                          \
            throw std::logic_error("ggml_linear: a tiled segment's format has no tiled kernels");                      \
        }                                                                                                              \
        launch_tile_format<Q, OutF32, false>(s, k, in, columns, out, n, plane, stream);                                \
        break;
    switch (s.format) {
        INFERNIX_GGML_FORMATS(INFERNIX_GGML_TILES)
    default:
        throw std::invalid_argument("ggml_linear: segment format is not a ggml block format");
    }
#undef INFERNIX_GGML_TILES
}

template <bool OutF32>
void launch_narrow(QType format, GgmlLayout layout, const RowsGroup& g, int k, const Columns& in, int columns,
                   cudaStream_t stream) {
#define INFERNIX_GGML_NARROW(Q)                                                                                        \
    case Q:                                                                                                            \
        if (layout == GgmlLayout::Tiles16) {                                                                           \
            launch_tiled_format<Q, OutF32>(g, k, in, columns, stream);                                                 \
        } else {                                                                                                       \
            launch_narrow_format<Q, OutF32>(g, k, in, columns, stream);                                                \
        }                                                                                                              \
        break;
    switch (format) {
        INFERNIX_GGML_FORMATS(INFERNIX_GGML_NARROW)
    default:
        throw std::invalid_argument("ggml_linear: segment format is not a ggml block format");
    }
#undef INFERNIX_GGML_NARROW
}

void decode_plane(const GgmlSegment& s, int k, std::uint8_t* plane, cudaStream_t stream) {
    const int slices = k / 32;
    const dim3 grid((s.rows + ggml::tile::kTileRows - 1) / ggml::tile::kTileRows, slices / ggml::tile::kStageSlices);
#define INFERNIX_GGML_PLANE(Q)                                                                                         \
    case Q: {                                                                                                          \
        const int row_bytes = static_cast<int>(k / ggml_block_elems(Q) * ggml_block_bytes(Q));                         \
        if (s.layout == GgmlLayout::Tiles16) {                                                                         \
            if constexpr (kTiledFormat<Q>) {                                                                           \
                plane_kernel<Q, true><<<grid, ggml::tile::kThreads, 0, stream>>>(s.data, s.rows, row_bytes, plane);   \
                break;                                                                                                 \
            }                                                                                                          \
            throw std::logic_error("ggml_linear: a tiled segment's format has no tiled kernels");                      \
        }                                                                                                              \
        plane_kernel<Q, false><<<grid, ggml::tile::kThreads, 0, stream>>>(s.data, s.rows, row_bytes, plane);          \
        break;                                                                                                         \
    }
    switch (s.format) {
        INFERNIX_GGML_FORMATS(INFERNIX_GGML_PLANE)
    default:
        throw std::invalid_argument("ggml_linear: segment format is not a ggml block format");
    }
#undef INFERNIX_GGML_PLANE
    check_launch("plane");
}

// Column c's sum of x over each 32-element slice (input element j at x[perm(j)]), in element order:
// the K-quants' min term for the BF16 tiles. One CTA per column.
__global__ void __launch_bounds__(128)
    bf16_xsum_kernel(const std::uint16_t* __restrict__ x, int k, const std::int32_t* __restrict__ perm,
                     int perm_block, float* __restrict__ xsum) {
    const int c = static_cast<int>(blockIdx.x), slices = k / 32;
    const std::uint16_t* column = x + static_cast<std::size_t>(c) * k;
    for (int s = static_cast<int>(threadIdx.x); s < slices; s += static_cast<int>(blockDim.x)) {
        float sum = 0.0F;
        for (int e = 0; e < 32; ++e) {
            const int j   = 32 * s + e;
            const int src = perm == nullptr ? j : perm[j / perm_block] * perm_block + j % perm_block;
            sum += __uint_as_float(static_cast<unsigned>(column[src]) << 16);
        }
        xsum[static_cast<std::size_t>(c) * slices + s] = sum;
    }
}

// The ggml-bf16-prefill tiles over one slab's plane (ggml_bf16_tile.cuh), every column at once.
struct Bf16Input {
    const std::uint16_t* x  = nullptr;
    int k                   = 0;
    const std::int32_t* perm = nullptr;
    int perm_block          = 0;
    const float* xsum       = nullptr; // the K-quants' slice sums
};

template <ggml::bf16::Scaling S, bool OutF32>
void launch_bf16_scaling(const GgmlSegment& s, const Bf16Input& in, int columns, void* out, int n,
                         const std::uint8_t* plane, cudaStream_t stream) {
    namespace b = ggml::bf16;
    constexpr int kShared = static_cast<int>(sizeof(b::Shared));
    static const bool configured = [] {
        return cudaFuncSetAttribute(b::tile_kernel<S, OutF32>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared) ==
               cudaSuccess;
    }();
    require(configured, "the BF16 tile kernel's shared memory exceeds the device's");
    const int tiles = (s.rows + ggml::tile::kTileRows - 1) / ggml::tile::kTileRows;
    const dim3 grid((s.rows + b::kRows - 1) / b::kRows, (columns + b::kColumns - 1) / b::kColumns);
    const b::DenseColumns cols{.x = in.x, .k = in.k, .perm = in.perm, .perm_block = in.perm_block, .xsum = in.xsum,
                               .col0 = 0, .columns = columns};
    b::tile_kernel<S, OutF32><<<grid, b::kThreads, kShared, stream>>>(plane, tiles, s.rows, in.k / 32, cols, out, n,
                                                                     s.out_offset, ggml::kappa_log2(s.format));
    check_launch("bf16 tile");
}

template <bool OutF32>
void launch_bf16(const GgmlSegment& s, const Bf16Input& in, int columns, void* out, int n, const std::uint8_t* plane,
                 cudaStream_t stream) {
#define INFERNIX_GGML_BF16(Q)                                                                                          \
    case Q:                                                                                                            \
        launch_bf16_scaling<ggml::bf16::kScaling<Q>, OutF32>(s, in, columns, out, n, plane, stream);                   \
        break;
    switch (s.format) {
        INFERNIX_GGML_FORMATS(INFERNIX_GGML_BF16)
    default:
        throw std::invalid_argument("ggml_linear: segment format is not a ggml block format");
    }
#undef INFERNIX_GGML_BF16
}
#undef INFERNIX_GGML_FORMATS

// Bytes of a decoded plane of `rows` rows of K = 32 slices.
std::size_t plane_bytes(int rows, int slices) {
    const std::size_t tiles = static_cast<std::size_t>((rows + ggml::tile::kTileRows - 1) / ggml::tile::kTileRows);
    return tiles * static_cast<std::size_t>(slices / ggml::tile::kStageSlices) * sizeof(ggml::tile::CodeStage);
}

// The row slabs of a wide call's segments, each a segment of whole row tiles whose plane fits
// kPlaneBytes, grouped into passes whose planes fit together.
struct Slab {
    GgmlSegment segment;
    std::size_t offset; // of its plane in the pass's workspace
};

std::vector<std::vector<Slab>> plan_slabs(std::span<const GgmlSegment> segments, int k) {
    const int slices          = k / 32;
    const std::size_t per_tile = plane_bytes(ggml::tile::kTileRows, slices);
    const int max_rows = static_cast<int>(std::max<std::size_t>(kPlaneBytes / per_tile, 1)) * ggml::tile::kTileRows;
    std::vector<std::vector<Slab>> passes(1);
    std::size_t used = 0;
    for (const auto& s : segments) {
        const int row_bytes = static_cast<int>(k / ggml_block_elems(s.format) * ggml_block_bytes(s.format));
        for (int r = 0; r < s.rows; r += max_rows) {
            GgmlSegment slab = s;
            slab.rows        = std::min(max_rows, s.rows - r);
            slab.data        = s.data + static_cast<std::size_t>(r) * row_bytes;
            slab.out_offset  = s.out_offset + r;
            const std::size_t bytes = plane_bytes(slab.rows, slices);
            if (used + bytes > kPlaneBytes && !passes.back().empty()) {
                passes.emplace_back();
                used = 0;
            }
            passes.back().push_back(Slab{slab, used});
            used += (bytes + 255) / 256 * 256;
        }
    }
    return passes;
}

// Whether a call of `columns` columns takes the tiles over decoded planes (the exact route's or the
// BF16 route's).
bool wide_call(int k, int columns) { return columns >= kTileMinColumns && (k / 32) % ggml::tile::kStageSlices == 0; }

// The BF16 route of a wide call (LinearRoutes::GgmlBf16Wide): the same slabs and planes as the exact
// route's, every column in one launch per slab, no encoding.
void linear_bf16(const Tensor& x, std::span<const GgmlSegment> segments, std::span<Tensor* const> outs,
                 const std::int32_t* input_perm, std::int32_t perm_block, WorkspaceArena& workspace, cudaStream_t stream) {
    const int k = x.ne[0], columns = x.ne[1];
    require(input_perm == nullptr || perm_block % 4 == 0, "the BF16 route reads x in four-element units of the permutation");
    require(reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0, "the BF16 route reads x in 8-byte units");
    const DType dtype = outs.front()->dtype;
    const auto passes = plan_slabs(segments, k);
    auto* planes      = static_cast<std::uint8_t*>(workspace.alloc(DType::U8, {static_cast<int>(kPlaneBytes)}).data);
    Bf16Input in{.x = static_cast<const std::uint16_t*>(x.data), .k = k, .perm = input_perm, .perm_block = perm_block};
    const bool min_terms = std::any_of(segments.begin(), segments.end(), [](const GgmlSegment& s) {
        return s.format == QType::GGML_Q4_K || s.format == QType::GGML_Q5_K;
    });
    if (min_terms) {
        float* xsum = static_cast<float*>(workspace.alloc(DType::FP32, {k / 32, columns}).data);
        bf16_xsum_kernel<<<columns, 128, 0, stream>>>(in.x, k, input_perm, perm_block, xsum);
        check_launch("bf16 xsum");
        in.xsum = xsum;
    }
    for (const auto& pass : passes) {
        for (const auto& slab : pass) { decode_plane(slab.segment, k, planes + slab.offset, stream); }
        for (const auto& slab : pass) {
            const Tensor& out = *outs[static_cast<std::size_t>(slab.segment.output)];
            if (dtype == DType::FP32) {
                launch_bf16<true>(slab.segment, in, columns, out.data, out.ne[0], planes + slab.offset, stream);
            } else {
                launch_bf16<false>(slab.segment, in, columns, out.data, out.ne[0], planes + slab.offset, stream);
            }
        }
    }
}

} // namespace

std::size_t ggml_linear_workspace_bytes(std::int32_t k, std::int32_t columns) {
    const auto align = [](std::size_t v) { return (v + 255) / 256 * 256; };
    const std::size_t t = static_cast<std::size_t>(std::min(columns, kEncodeColumns));
    if (!wide_call(k, columns)) { return align(fast::encoded_column_words(k) * t * 4) + align(t * 4); }
    // The exact route's encoded batch, or the BF16 route's slice sums (both with the planes).
    const std::size_t exact = align(fast::encoded_column_words(k) * t * 4) + align(t * 4);
    const std::size_t bf16  = align(static_cast<std::size_t>(k / 32) * static_cast<std::size_t>(columns) * 4);
    return std::max(exact, bf16) + kPlaneBytes;
}

std::size_t ggml_row_bytes(QType format, std::int32_t k) {
    return static_cast<std::size_t>(k) / ggml_block_elems(format) * ggml_block_bytes(format);
}

std::vector<GgmlSegment> ggml_split_segments(std::span<const GgmlSegment> segments, std::int32_t k,
                                             std::span<const std::int32_t> bounds) {
    require(bounds.size() >= 2 && bounds.front() == 0 && std::is_sorted(bounds.begin(), bounds.end()),
            "split bounds must ascend from 0");
    std::vector<GgmlSegment> out;
    for (const auto& s : segments) {
        require(s.out_offset >= 0 && s.out_offset + s.rows <= bounds.back(), "segment exceeds the split rows");
        const std::size_t row_bytes = ggml_row_bytes(s.format, k);
        for (std::size_t i = 0; i + 1 < bounds.size(); ++i) {
            const int lo = std::max(s.out_offset, bounds[i]), hi = std::min(s.out_offset + s.rows, bounds[i + 1]);
            if (lo >= hi) { continue; }
            // A tiled matrix splits at whole tiles (tile i at 16 i row_bytes: the same offset).
            require(s.layout != GgmlLayout::Tiles16 || (lo - s.out_offset) % 16 == 0,
                    "a ggml_tiles16_v1 segment splits only at multiples of 16 rows");
            GgmlSegment part = s;
            part.data        = s.data + static_cast<std::size_t>(lo - s.out_offset) * row_bytes;
            part.rows        = hi - lo;
            part.out_offset  = lo - bounds[i];
            part.output      = static_cast<std::int32_t>(i);
            out.push_back(part);
        }
    }
    return out;
}

void ggml_linear(const Tensor& x, std::span<const GgmlSegment> segments, Tensor& out, const std::int32_t* input_perm,
                 std::int32_t perm_block, WorkspaceArena& workspace, cudaStream_t stream, LinearRoutes routes) {
    for (const auto& s : segments) { require(s.output == 0, "a one-output call's segments name output 0"); }
    Tensor* const outs[] = {&out};
    ggml_linear(x, segments, outs, input_perm, perm_block, workspace, stream, routes);
}

void ggml_linear(const Tensor& x, std::span<const GgmlSegment> segments, std::span<Tensor* const> outs,
                 const std::int32_t* input_perm, std::int32_t perm_block, WorkspaceArena& workspace, cudaStream_t stream,
                 LinearRoutes routes) {
    require(valid_linear_routes(routes), "unknown linear route flags");
    require(x.data != nullptr && x.dtype == DType::BF16 && x.is_contiguous(), "x must be contiguous BF16 [K, T]");
    require(!outs.empty() && outs.front() != nullptr, "no outputs");
    const DType dtype = outs.front()->dtype;
    const int k = x.ne[0], columns = x.ne[1];
    for (const Tensor* o : outs) {
        require(o != nullptr && o->data != nullptr && (o->dtype == DType::FP32 || o->dtype == DType::BF16) &&
                    o->dtype == dtype && o->is_contiguous() && o->ne[1] == columns,
                "outputs must be contiguous [N, T] of one dtype, BF16 or FP32");
    }
    require(columns > 0 && k % 32 == 0 && !segments.empty(), "shapes disagree");
    for (const auto& s : segments) {
        require(s.output >= 0 && static_cast<std::size_t>(s.output) < outs.size(), "segment names no output");
        require(is_ggml(s.format) && s.data != nullptr && s.rows > 0 && s.out_offset >= 0 &&
                    s.out_offset + s.rows <= outs[static_cast<std::size_t>(s.output)]->ne[0],
                "segment exceeds its output");
        require(k % static_cast<int>(ggml_block_elems(s.format)) == 0, "K is not whole blocks of a segment's format");
        require(s.layout == GgmlLayout::Rows || ggml_tiles16_supported(s.format),
                "ggml_tiles16_v1 holds Q4_K, Q5_K, Q6_K and IQ4_XS matrices only");
    }
    require(input_perm == nullptr || (perm_block > 0 && k % perm_block == 0), "input permutation needs whole blocks");
    const std::size_t out_bytes = dtype == DType::FP32 ? 4 : 2;
    auto scope = workspace.scope();
    // Wide calls decode each segment's weights once into planes for the tiles; segments whose
    // planes exceed kPlaneBytes run in passes of row slabs, each re-encoding the columns.
    const bool wide = wide_call(k, columns);
    if (wide && has_route(routes, LinearRoutes::GgmlBf16Wide)) {
        linear_bf16(x, segments, outs, input_perm, perm_block, workspace, stream);
        return;
    }
    const int batch = std::min(columns, kEncodeColumns);
    Tensor encoded = workspace.alloc(DType::I32, {static_cast<int>(fast::encoded_column_words(k)), batch});
    Tensor emax    = workspace.alloc(DType::I32, {batch});
    const Columns in{.encoded = static_cast<const std::int32_t*>(encoded.data),
                     .emax    = static_cast<const std::int32_t*>(emax.data)};
    std::vector<std::vector<Slab>> passes;
    std::uint8_t* planes = nullptr;
    if (wide) {
        passes = plan_slabs(segments, k);
        planes = static_cast<std::uint8_t*>(workspace.alloc(DType::U8, {static_cast<int>(kPlaneBytes)}).data);
    } else {
        passes.emplace_back();
        for (const auto& s : segments) { passes.back().push_back(Slab{s, 0}); }
    }
    // The encoder's 16-byte loads (encode_column_vec) wherever the columns allow them.
    const int vec = (input_perm == nullptr || perm_block % 8 == 0) && reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0 ? 1 : 0;
    for (const auto& pass : passes) {
        if (wide) {
            for (const auto& slab : pass) { decode_plane(slab.segment, k, planes + slab.offset, stream); }
        }
        for (int first = 0; first < columns; first += batch) {
            const int count = std::min(batch, columns - first);
            CUDA_CHECK(pdl::launch_consumer({dim3(count), dim3(kEncodeThreads), 0, stream}, encode_kernel,
                                            static_cast<const std::uint16_t*>(x.data) + static_cast<std::size_t>(first) * k,
                                            k, input_perm, perm_block, static_cast<std::int32_t*>(encoded.data),
                                            static_cast<std::int32_t*>(emax.data), vec));
            const auto column_out = [&](const GgmlSegment& s) {
                const Tensor& out = *outs[static_cast<std::size_t>(s.output)];
                return static_cast<void*>(static_cast<std::byte*>(out.data) + static_cast<std::size_t>(first) * out.ne[0] * out_bytes);
            };
            if (wide) {
                for (const auto& slab : pass) {
                    const int n = outs[static_cast<std::size_t>(slab.segment.output)]->ne[0];
                    if (dtype == DType::FP32) {
                        launch_tiles<true>(slab.segment, k, in, count, column_out(slab.segment), n, planes + slab.offset, stream);
                    } else {
                        launch_tiles<false>(slab.segment, k, in, count, column_out(slab.segment), n, planes + slab.offset, stream);
                    }
                }
                continue;
            }
            // A call of exactly two mixed-capable formats (BF16 outputs, at most kMax segments of one
            // layout) runs in one launch: row-major segments at verification widths, tiled ones at
            // every narrow width (by MMA whatever tiled_rows_route says of either format alone: GDN
            // q/k/v/z at one column 21.9 us against 26.0 in a rows launch per format).
            const GgmlLayout layout0 = pass.front().segment.layout;
            const bool tiled_pass    = layout0 == GgmlLayout::Tiles16;
            if (dtype == DType::BF16 && count >= (tiled_pass ? 1 : kMmaMinColumns) && count <= kMmaMaxColumns &&
                pass.size() <= static_cast<std::size_t>(RowsGroup::kMax)) {
                QType f0 = pass.front().segment.format, f1 = f0;
                bool ok = true;
                for (const auto& slab : pass) {
                    const QType f = slab.segment.format;
                    ok = ok && mixed_format(f) && slab.segment.layout == layout0;
                    if (f != f0 && f1 == f0) {
                        f1 = f;
                    } else if (f != f0 && f != f1) {
                        ok = false;
                    }
                }
                if (ok && f1 != f0) {
                    if (static_cast<int>(f1) < static_cast<int>(f0)) { std::swap(f0, f1); }
                    RowsGroup g;
                    for (const auto& slab : pass) {
                        const GgmlSegment& s  = slab.segment;
                        g.data[g.count]       = s.data;
                        g.out[g.count]        = column_out(s);
                        g.rows[g.count]       = s.rows;
                        g.n[g.count]          = outs[static_cast<std::size_t>(s.output)]->ne[0];
                        g.out_offset[g.count] = s.out_offset;
                        g.row_bytes[g.count]  = static_cast<int>(ggml_row_bytes(s.format, k));
                        g.format[g.count]     = s.format == f0 ? 0 : 1;
                        ++g.count;
                    }
                    if (tiled_pass ? launch_tiled_mixed(f0, f1, g, k, in, count, stream)
                                   : launch_mixed(f0, f1, g, k, in, count, stream)) {
                        continue;
                    }
                }
            }
            std::vector<bool> taken(pass.size(), false);
            for (std::size_t i = 0; i < pass.size(); ++i) {
                if (taken[i]) { continue; }
                // The segments of each format and layout, in order of first appearance, share
                // launches of up to RowsGroup::kMax.
                const QType format      = pass[i].segment.format;
                const GgmlLayout layout = pass[i].segment.layout;
                RowsGroup g;
                for (std::size_t j = i; j < pass.size() && g.count < RowsGroup::kMax; ++j) {
                    const GgmlSegment& s = pass[j].segment;
                    if (taken[j] || s.format != format || s.layout != layout) { continue; }
                    taken[j]              = true;
                    g.data[g.count]       = s.data;
                    g.out[g.count]        = column_out(s);
                    g.rows[g.count]       = s.rows;
                    g.n[g.count]          = outs[static_cast<std::size_t>(s.output)]->ne[0];
                    g.out_offset[g.count] = s.out_offset;
                    ++g.count;
                }
                if (dtype == DType::FP32) {
                    launch_narrow<true>(format, layout, g, k, in, count, stream);
                } else {
                    launch_narrow<false>(format, layout, g, k, in, count, stream);
                }
            }
        }
    }
}

} // namespace infernix::ops
