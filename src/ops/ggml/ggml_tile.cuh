#pragma once

// Exact tensor-core tiles of the ggml arithmetic (ops/ggml/ggml_exact.h) for wide calls (prefill).
//
// A CTA computes kTileRows weight rows x kTileColumns encoded columns. K advances in stages of
// kStageSlices 32-element slices; the raw weight bytes arrive by cp.async in units of 256
// elements (one block, or the run of small blocks covering them), each unit one unit ahead, and the
// columns' limbs one stage ahead, both double-buffered, so global and L2 reads overlap the decode
// and MMAs. Every thread decodes 8-element chunks of a stage (tile_chunk: ggml_exact.h's chunk
// integers by word operations, on the raw copy) into signed byte codes; each row keeps its slice's
// two sub-scales. A CTA has 256 threads (8 warps of 32 rows x 16 columns) or 512 (16 warps of 16 x
// 16: half the outputs per thread, so that one CTA fits 128 registers per thread). For each slice a
// warp multiplies the codes by each limb plane with mma.sync s8 (k32, or two k16 halves for formats
// whose sub-scale covers 16 elements) and adds sub * partial to an int32 block sum per limb plane
// (|sum| < 2^kSumLog2 <= 2^27); at the end of each block it adds mant(d) * P * 2^(e(d) - 1) (and the
// min term) to its outputs: the integers of ggml_dp4a.cuh's add_slice, so the outputs are the same
// bits as every other route's. row_bases chooses the accumulators from the rows' scale spans (Accum):
// in 512-thread tiles of formats without a min term usually one int64 per limb plane, each block
// adding M * sum with M = +-mant(d) * 2^(e(d) - 1 - base) < 2^31 (one wide multiply-add), the planes
// combined once at the end; else one int64 per output (narrow rows) or int128 (wide spans).
//
// compute_plane runs the same K loop over a decoded plane instead: each stage's codes, sub-scales
// and scale words are a CodeStage image that decode_stage wrote once per call (ggml_linear's dense
// projections), copied by cp.async one stage ahead with no decode in the loop. The experts' tiles
// decode in the loop: a group-wise plane measured 2 % faster at 16,384 tokens and no faster end to
// end, where the layer walk's MoE calls are 4,096-8,192 tokens (2026-10-12).
//
// Measured on the RTX 5090 (2026-10-11, the IQ3_S model's dense shapes at 512 columns): tiles of
// 32 x 128 columns re-read each column's limbs per 32 rows and were bound by L2 throughput (66 %
// of peak); folding the sub-scales into two byte planes of the weights (6 MMAs per weight instead
// of 3 plus IMADs) was no faster, nor at 4096 columns with Q6_K planes at one CTA per SM (2026-10-13:
// prefill pp32768 4,090 against 4,093 tok/s; Q6_K projections -11 % to +1 %, the experts' calls
// mixed). At 4096 columns (2026-10-12) double-buffering the codes so the
// next stage decodes beside this stage's MMAs (one barrier fewer per stage) was 3-5 % slower for
// Q6_K, Q5_K and Q4_K than this decode-then-MMA order.

#include "core/ggml_format.h"
#include "ops/ggml/ggml_dp4a.cuh"
#include "ops/ggml/ggml_exact.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace infernix::ggml::tile {

constexpr int kThreads      = 256;
constexpr int kTileRows     = 64;  // two warp rows of 32 (two m16 tiles each)
constexpr int kTileColumns  = 64;  // four warp columns of 16 (two n8 tiles each)
constexpr int kMTiles       = 2;   // m16 tiles per warp
constexpr int kNTiles       = 2;   // n8 tiles per warp
constexpr int kStageSlices  = 2;   // 64 elements of K per stage
constexpr int kUnitSlices   = 8;   // 256 elements of K per raw unit
constexpr int kBWords       = 28;  // shared column pitch of a slice's 24 limb words
// 16-byte chunks of one row's raw unit: at most 225 bytes (a 210-byte Q6_K block from a 16-byte
// aligned start, so at most 15 bytes before it).
constexpr int kRawChunks    = 15;

// Elements per block of F (the period of the int128 flush).
template <QType F>
inline constexpr int kBlockSlices = F == QType::GGML_IQ4_NL ? 1 : F == QType::GGML_Q2_0 ? 2 : 8;
static_assert(kUnitSlices % kStageSlices == 0);

// log2 of a bound on one block's term |mant(d) * P| (with the K-quants' min term): elements x
// |sub| x |code| x |X| (< 2^21) x mant (< 2^11). Q6_K 2^8 2^7 2^5; Q5_K 2^8 2^6 2^5 (+ the min term
// < 2^46); Q4_K 2^8 2^6 2^4 (+ 2^46); IQ4_XS 2^8 2^5 2^7; IQ4_NL 2^5 1 2^7; IQ3_S 2^8 2^5 2^4;
// IQ3_XXS 2^8 2^5 2^6; IQ2_S 2^8 2^5 2^6; Q2_0 2^6 1 2^1.
template <QType F>
inline constexpr int kTermLog2 = F == QType::GGML_Q6_K || F == QType::GGML_Q5_K || F == QType::GGML_IQ4_XS ? 52
                                 : F == QType::GGML_Q4_K || F == QType::GGML_IQ3_XXS || F == QType::GGML_IQ2_S ? 51
                                 : F == QType::GGML_IQ3_S                                                          ? 49
                                 : F == QType::GGML_IQ4_NL                                                         ? 44
                                                                                                                   : 39;

// Formats whose sub-scales are all 1 (the MMA sums need no scaling).
template <QType F>
inline constexpr bool kUnitSub = F == QType::GGML_IQ4_NL || F == QType::GGML_Q2_0;

// Formats whose sub-scale covers 16 elements: two k16 MMAs per slice, each scaled by its own.
template <QType F>
inline constexpr bool kHalfSub = F == QType::GGML_Q6_K || F == QType::GGML_IQ2_S;

// d = a * b over one 16x8x32 int8 tile (exact int32).
__device__ __forceinline__ void mma_k32(int (&d)[4], int a0, int a1, int a2, int a3, int b0, int b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}

// d = a * b over one 16x8x16 int8 tile (exact int32).
__device__ __forceinline__ void mma_k16(int (&d)[4], int a0, int a1, int b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}

// Four 8x8 matrices of 16-bit words from shared memory: lanes 8m .. 8m + 7 give the 16-byte row
// addresses of matrix m, and register m of lane (g, t) is the 32-bit word t of row g of matrix m.
__device__ __forceinline__ void ldmatrix_x4(int (&r)[4], const void* row) {
    const auto s = static_cast<unsigned>(__cvta_generic_to_shared(row));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

// 16 bytes into shared memory, of which the first `bytes` are read (the rest zero-filled).
__device__ __forceinline__ void cp_async16(void* shared, const void* global, int bytes) {
    const auto s = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(global), "r"(bytes));
}

__device__ __forceinline__ void cp_async4(void* shared, const void* global) {
    const auto s = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(global));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

constexpr int kRowGroups = kTileRows / 16; // m16 row groups of a tile

// One stage of a tile's decoded weights (kTileRows rows x kStageSlices slices). Codes in A-fragment
// order: a[j][m][g][t] holds words t of rows 16m + g and 16m + g + 8, then words 4 + t of the same
// rows (the m16n8k32 fragment a0..a3, or the two k16 fragments), one 128-bit load per row group;
// sub[j][m][g] holds the two sub-scales of row 16m + g, then those of row 16m + g + 8. A plane
// (decode_planes) stores these images in global memory, stage after stage, row tile after row tile.
struct CodeStage {
    alignas(16) int a[kStageSlices][kRowGroups][8][4][4];
    alignas(16) int sub[kStageSlices][kRowGroups][8][4];
    std::uint16_t d[kStageSlices][kTileRows];
    // The K-quants' min terms (the image's tail: formats without them copy only the bytes before).
    alignas(16) std::uint16_t dmin[kStageSlices][kTileRows];
    int min[kStageSlices][kTileRows];
};
static_assert(sizeof(CodeStage) % 16 == 0, "stage images copy in 16-byte units");

// A code stage of a format whose sub-scales are all 1 and which has no min term (IQ4_NL, Q2_0): the
// codes and d only (CodeStage's a and d), 71 % of a CodeStage.
struct UnitCodeStage {
    alignas(16) int a[kStageSlices][kRowGroups][8][4][4];
    std::uint16_t d[kStageSlices][kTileRows];
};

// Bytes of a stage image a tile reads for format F.
template <QType F>
inline constexpr int kImageBytes = fast::kMin<F> ? static_cast<int>(sizeof(CodeStage))
                                                 : static_cast<int>(offsetof(CodeStage, dmin));
static_assert(offsetof(CodeStage, dmin) % 16 == 0);

// Dynamic shared memory of a tile CTA that decodes its raw units (the kernels declare it extern;
// launchers pass sizeof).
struct Shared {
    static constexpr int kBufs = 2;                            // limb buffers
    CodeStage code;                                            // the current stage's codes
    alignas(16) int b[2][kStageSlices][kTileColumns][kBWords]; // limbs a, b, c (8 words each), two stages
    int xsum[2][kStageSlices][kTileColumns];
    alignas(16) std::uint8_t raw[2][kTileRows][kRawChunks * 16]; // raw units, two
    int raw_offset[2][kTileRows]; // where row r's unit bytes start in raw[.][r]
    int base[kTileRows];          // row r's smallest scale shift (the int64 accumulators' unit)
};

// Dynamic shared memory of a tile CTA that reads decoded planes: stage images and limbs by cp.async
// in a ring of kBufs stages (the K loop copies kBufs - 1 stages ahead).
struct PlaneShared {
    static constexpr int kBufs = 2;
    CodeStage code[kBufs];
    alignas(16) int b[kBufs][kStageSlices][kTileColumns][kBWords];
    int xsum[kBufs][kStageSlices][kTileColumns];
    int base[kTileRows];
};

// A tile's exact outputs: thread (warp, lane) holds, for its warp tile's m16 tiles mi < mtiles and
// n8 tiles nt < ntiles, rows row0 + 16 mi + g (+ 8 for q >= 2) and columns col0 + 8 nt + 2t + (q & 1),
// with g = lane / 4, t = lane % 4.
struct Accumulators {
    I128 r[kMTiles][kNTiles][4];
};

// The warps' share of a tile with `columns` valid columns (the CTA's count, the same for every warp),
// layout 0, 1 or 2 by columns over 32, up to 32 and up to 16. A CTA of 256 threads (8 warps):
// 2 x 4 warps of 32 rows x 16 columns; 4 x 2 of 16 x 16; 4 x 2 of 16 x 8. A CTA of 512 (16 warps,
// half the outputs per thread): 4 x 4 warps of 16 x 16; 4 x 4 of 16 x 8; 4 x 2 of 16 x 8 (warps 8-15
// idle, mtiles 0). A narrow tile's warps compute fewer padded columns, their weights decoded once.
struct WarpTile {
    int row0, col0, mtiles, ntiles;
};

template <int Threads>
struct WarpLayout {
    int row0, col0;
    bool active;
};

template <int Threads, int Layout>
__device__ __forceinline__ WarpLayout<Threads> warp_layout() {
    const int warp = static_cast<int>(threadIdx.x) / 32;
    if constexpr (Threads == 256) {
        if constexpr (Layout == 0) { return {32 * (warp / 4), 16 * (warp % 4), true}; }
        if constexpr (Layout == 1) { return {16 * (warp / 2), 16 * (warp % 2), true}; }
        return {16 * (warp / 2), 8 * (warp % 2), true};
    } else {
        static_assert(Threads == 512, "tiles of 256 or 512 threads");
        if constexpr (Layout == 0) { return {16 * (warp / 4), 16 * (warp % 4), true}; }
        if constexpr (Layout == 1) { return {16 * (warp / 4), 8 * (warp % 4), true}; }
        return {16 * ((warp % 8) / 2), 8 * (warp % 2), warp < 8};
    }
}

// m16 and n8 tiles of a warp of layout Layout.
template <int Threads, int Layout>
inline constexpr int kLayoutM = Threads == 256 && Layout == 0 ? 2 : 1;
template <int Threads, int Layout>
inline constexpr int kLayoutN = Layout == 0 || (Threads == 256 && Layout == 1) ? 2 : 1;

template <int Threads = kThreads>
__device__ __forceinline__ WarpTile warp_tile(int columns) {
    const auto make = [](auto layout, int m, int n) {
        return WarpTile{layout.row0, layout.col0, layout.active ? m : 0, n};
    };
    if (columns > 32) { return make(warp_layout<Threads, 0>(), kLayoutM<Threads, 0>, kLayoutN<Threads, 0>); }
    if (columns > 16) { return make(warp_layout<Threads, 1>(), kLayoutM<Threads, 1>, kLayoutN<Threads, 1>); }
    return make(warp_layout<Threads, 2>(), kLayoutM<Threads, 2>, kLayoutN<Threads, 2>);
}

__device__ __forceinline__ int out_row(const WarpTile& w, int lane, int mi, int q) {
    return w.row0 + 16 * mi + lane / 4 + 8 * (q >> 1);
}
__device__ __forceinline__ int out_column(const WarpTile& w, int lane, int nt, int q) {
    return w.col0 + 8 * nt + 2 * (lane % 4) + (q & 1);
}

// Issues the cp.async copies of raw unit `unit` (slices 8 unit .. of K = 32 slices) into buffer `buf`.
template <QType F, int Threads, class Sm, class RowOf>
__device__ __forceinline__ void issue_unit(Sm& sm, int buf, int unit, int slices, RowOf row_of) {
    constexpr int kBlockElems = static_cast<int>(ggml_block_elems(F));
    constexpr int kBlockBytes = static_cast<int>(ggml_block_bytes(F));
    const int elems         = 32 * min(kUnitSlices, slices - unit * kUnitSlices);
    const int bytes         = elems / kBlockElems * kBlockBytes;
    const std::size_t begin = static_cast<std::size_t>(unit) * kUnitSlices * 32 / kBlockElems * kBlockBytes;
    for (int i = threadIdx.x; i < kTileRows * kRawChunks; i += Threads) {
        const int r = i / kRawChunks, chunk = i % kRawChunks;
        const std::uint8_t* src = row_of(r) + begin;
        const auto* aligned     = reinterpret_cast<const std::uint8_t*>(reinterpret_cast<std::uintptr_t>(src) & ~std::uintptr_t{15});
        const int offset        = static_cast<int>(src - aligned);
        const int need          = offset + bytes - 16 * chunk; // bytes of this chunk the unit reads
        if (chunk == 0) { sm.raw_offset[buf][r] = offset; }
        if (need > 0) { cp_async16(&sm.raw[buf][r][16 * chunk], aligned + 16 * chunk, need < 16 ? need : 16); }
    }
}

// A thread's cp.async copies of every stage's limbs (16 bytes each) and slice sums: the same
// (slice of the stage, column, word) every stage, so their sources (at stage slice 0) and shared
// offsets are resolved once per tile.
constexpr int kStageCopies = kStageSlices * kTileColumns * 6; // 16-byte limb copies per stage
template <int Threads>
inline constexpr int kLimbCopies = (kStageCopies + Threads - 1) / Threads;
static_assert(kStageSlices * kTileColumns <= kThreads, "at most one slice-sum copy per thread");
template <int Threads>
struct LimbCopies {
    const std::int32_t* src[kLimbCopies<Threads>]; // null: no copy (past kStageCopies)
    int dst[kLimbCopies<Threads>];                 // word offset in sm.b[buf]
    const std::int32_t* xsum;  // null: no slice-sum copy (or not a min format)
    int xdst;                  // word offset in sm.xsum[buf]
};

template <QType F, int Threads, class ColumnLimbs>
__device__ __forceinline__ LimbCopies<Threads> limb_copies(ColumnLimbs limbs, int xsum_offset) {
    LimbCopies<Threads> lc{};
#pragma unroll
    for (int k = 0; k < kLimbCopies<Threads>; ++k) {
        const int i = static_cast<int>(threadIdx.x) + Threads * k;
        const int j = i / (kTileColumns * 6), c = (i / 6) % kTileColumns, u = i % 6;
        lc.src[k] = i < kStageCopies ? limbs(c) + j * fast::kSliceWords + 4 * u : nullptr;
        lc.dst[k] = (j * kTileColumns + c) * kBWords + 4 * u;
    }
    if constexpr (fast::kMin<F>) {
        const int i = static_cast<int>(threadIdx.x);
        if (i < kStageSlices * kTileColumns) {
            const int j = i / kTileColumns, c = i % kTileColumns;
            lc.xsum = limbs(c) + xsum_offset + j;
            lc.xdst = i;
        }
    }
    return lc;
}

// Issues the cp.async copies of stage s0's limbs (and slice sums) into buffer `buf`.
template <QType F, int Threads, class Sm>
__device__ __forceinline__ void issue_limbs(Sm& sm, int buf, int s0, const LimbCopies<Threads>& lc) {
#pragma unroll
    for (int k = 0; k < kLimbCopies<Threads>; ++k) {
        if (kStageCopies % Threads != 0 && lc.src[k] == nullptr) { continue; }
        cp_async16(&sm.b[buf][0][0][0] + lc.dst[k], lc.src[k] + s0 * fast::kSliceWords, 16);
    }
    if constexpr (fast::kMin<F>) {
        if (lc.xsum != nullptr) { cp_async4(&sm.xsum[buf][0][0] + lc.xdst, lc.xsum + s0); }
    }
}

// One 8-element chunk of a block as the stage stores it: two code words (byte j of word h is element
// 4h + j), its sub-scale and scale words. The integers of chunk<F> (ggml_exact.h), decoded with
// word operations (ggml_dp4a.cuh's byte tricks) for every ggml format. `b` is the block in the
// stage's raw copy (shared memory at any byte alignment, so words are read byte by byte); `cb` the
// format's codebook (fast::DeviceCodebook or a CTA's fast::SharedCodebook: the same values).
struct TileChunk {
    int w[2];
    int sub;
    std::uint16_t d;
    int min_sub;
    std::uint16_t dmin;
};

template <QType F, class Codebook = fast::DeviceCodebook<F>>
__device__ __forceinline__ TileChunk tile_chunk(const std::uint8_t* b, int part, const Codebook& cb = Codebook{}) {
    TileChunk o{};
    o.sub = 1;
    if constexpr (F == QType::GGML_IQ3_S) {
        const int sb = part / 4, l = part % 4;
        const int qh = b[66 + sb];
        const int i1 = b[2 + 8 * sb + 2 * l] | ((qh << (8 - 2 * l)) & 256);
        const int i2 = b[3 + 8 * sb + 2 * l] | ((qh << (7 - 2 * l)) & 256);
        const unsigned sign = b[74 + 4 * sb + l];
        o.w[0] = fast::apply_signs4(cb.grid32(i1), sign & 0xFU);
        o.w[1] = fast::apply_signs4(cb.grid32(i2), sign >> 4);
        o.sub  = 2 * ((b[106 + sb / 2] >> (4 * (sb % 2))) & 0xF) + 1;
        o.d    = u16_at(b);
    } else if constexpr (F == QType::GGML_IQ2_S) {
        const int ib = part / 4, l = part % 4;
        const int index = b[2 + 4 * ib + l] | ((b[66 + ib] << (8 - 2 * l)) & 0x300);
        const std::uint64_t grid = cb.grid64(index);
        const unsigned sign = b[34 + 4 * ib + l];
        o.w[0] = fast::apply_signs4(static_cast<std::uint32_t>(grid), sign & 0xFU);
        o.w[1] = fast::apply_signs4(static_cast<std::uint32_t>(grid >> 32), sign >> 4);
        o.sub  = 2 * ((b[74 + ib] >> (4 * (l / 2))) & 0xF) + 1;
        o.d    = u16_at(b);
    } else if constexpr (F == QType::GGML_IQ3_XXS) {
        const int ib = part / 4, l = part % 4;
        const std::uint32_t aux = u32_at(b + 66 + 4 * ib);
        const unsigned sign = cb.sign((aux >> (7 * l)) & 127);
        o.w[0] = fast::apply_signs4(cb.grid32(b[2 + 8 * ib + 2 * l]), sign & 0xFU);
        o.w[1] = fast::apply_signs4(cb.grid32(b[3 + 8 * ib + 2 * l]), sign >> 4);
        o.sub  = 2 * static_cast<int>(aux >> 28) + 1;
        o.d    = u16_at(b);
    } else if constexpr (F == QType::GGML_Q2_0) {
        // Four 2-bit codes per byte (element 4i + k at bits 2k), each minus 1.
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const std::uint32_t v = b[2 + 2 * part + h];
            const std::uint32_t e = (v & 3U) | (((v >> 2) & 3U) << 8) | (((v >> 4) & 3U) << 16) | (((v >> 6) & 3U) << 24);
            o.w[h] = fast::sub_bytes4(e, 0x01010101U);
        }
        o.d = u16_at(b);
    } else if constexpr (F == QType::GGML_IQ4_NL || F == QType::GGML_IQ4_XS) {
        // Elements 0-15 of a 32-element group are the low nibbles of its 16 bytes, 16-31 the high.
        const int group = F == QType::GGML_IQ4_NL ? 0 : part / 4, p = part % 4;
        const std::uint8_t* qs = b + (F == QType::GGML_IQ4_NL ? 2 : 8 + 16 * group) + 8 * (p & 1);
        const int2 v0 = fast::iq4nl_values(u32_at(qs)), v1 = fast::iq4nl_values(u32_at(qs + 4));
        o.w[0] = p >= 2 ? v0.y : v0.x;
        o.w[1] = p >= 2 ? v1.y : v1.x;
        o.d    = u16_at(b);
        if constexpr (F == QType::GGML_IQ4_XS) {
            o.sub = (((b[4 + group / 2] >> (4 * (group % 2))) & 0xF) | (((u16_at(b + 2) >> (2 * group)) & 3) << 4)) - 32;
        }
    } else if constexpr (F == QType::GGML_Q4_K || F == QType::GGML_Q5_K) {
        const int e = 8 * part, ch = e / 64, l = e % 64, sb = 2 * ch + (l >= 32 ? 1 : 0), li = l % 32;
        const int shift = l >= 32 ? 4 : 0;
        const std::uint8_t* qs = b + (F == QType::GGML_Q4_K ? 16 : 48) + 32 * ch + li;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            std::uint32_t v = (u32_at(qs + 4 * h) >> shift) & 0x0F0F0F0FU;
            if constexpr (F == QType::GGML_Q5_K) { v |= ((u32_at(b + 16 + li + 4 * h) >> sb) & 0x01010101U) << 4; }
            o.w[h] = static_cast<int>(v);
        }
        scale_min_k4(sb, b + 4, o.sub, o.min_sub);
        o.d    = u16_at(b);
        o.dmin = u16_at(b + 2);
    } else if constexpr (F == QType::GGML_Q6_K) {
        const int e = 8 * part, half = e / 128, r = e % 128, quarter = r / 32, l = r % 32;
        const std::uint8_t* ql = b + 64 * half + (quarter & 1) * 32 + l;
        const std::uint8_t* qh = b + 128 + 32 * half + l;
        const int lshift = quarter >= 2 ? 4 : 0, hshift = 2 * quarter;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const std::uint32_t lo = (u32_at(ql + 4 * h) >> lshift) & 0x0F0F0F0FU;
            const std::uint32_t hi = ((u32_at(qh + 4 * h) >> hshift) & 0x03030303U) << 4;
            o.w[h] = fast::sub_bytes4(lo | hi, 0x20202020U);
        }
        o.sub = static_cast<std::int8_t>(b[192 + 8 * half + l / 16 + 2 * quarter]);
        o.d   = u16_at(b + 208);
    } else {
        static_assert(F == QType::GGML_IQ3_S, "tile_chunk: no decoder for this format");
    }
    return o;
}

// v * 2^shift in two's complement (0 <= shift < 64; the caller bounds the result).
__device__ __forceinline__ long long shl64(long long v, int shift) {
    return static_cast<long long>(static_cast<unsigned long long>(v) << shift);
}

// The scale shifts (binary16 exponent field, at least 1, minus 1) of row r's blocks: d's, and the
// K-quants' dmin's.
template <QType F>
__device__ __forceinline__ void scale_shift_range(const std::uint8_t* block, int& lo, int& hi) {
    const auto shift = [](unsigned w) { return max(static_cast<int>((w >> 10) & 0x1FU), 1) - 1; };
    const auto* d = reinterpret_cast<const unsigned short*>(block + (F == QType::GGML_Q6_K ? 208 : 0));
    lo = min(lo, shift(__ldg(d)));
    hi = max(hi, shift(__ldg(d)));
    if constexpr (fast::kMin<F>) {
        lo = min(lo, shift(__ldg(d + 1)));
        hi = max(hi, shift(__ldg(d + 1)));
    }
}

// How a tile accumulates its outputs (row_bases' choice from the rows' scales, the same for every
// thread of the CTA). Every mode gives the same integers.
enum class Accum {
    kWide,   // int128 per output
    kNarrow, // one int64 per output, in units of 2^base(row)
    kLimbs,  // one int64 per output and limb plane, in units of 2^base(row) (kLimbsMode)
};

// Tiles whose int64 mode is kLimbs (else kNarrow): formats without a min term and CTAs of 512
// threads (8 outputs per thread; 256-thread tiles' 16 would need 96 registers for them).
template <QType F, int Threads>
inline constexpr bool kLimbsMode = !fast::kMin<F> && Threads == 512;

// log2 of a bound on one limb plane's block sum |sum of sub * code * limb| (kTermLog2 with the limb's
// 2^7 for X's 2^21 and without the mantissa's 2^11); formats without a min term.
template <QType F>
inline constexpr int kSumLog2 = kTermLog2<F> - 25;

// Every row's smallest scale shift into sm.base and the CTA's accumulator mode, with span the
// largest difference between a row's shifts and blocks the row's block count:
// - kLimbs (kLimbsMode tiles): span <= 20 and 11 + span + kSumLog2<F> + ceil(log2(blocks)) <= 62,
//   so that M = +-mant(d) * 2^(shift(d) - base) fits int32 (|M| < 2^31) and each limb plane's sum of
//   M * block sum fits int64. (This bound contains kNarrow's but for Q2_0 spans of 21-23, which
//   take kWide.)
// - kNarrow (other tiles): span <= 62 - kTermLog2<F> - ceil(log2(blocks)), so that the row's block
//   terms, each shifted by its scale's shift less the row's smallest, sum in int64: |sum| < blocks *
//   2^(term + span) <= 2^62.
// - kWide otherwise.
// The same for every thread of the CTA (which all call it).
// Stride: a row's blocks are Stride * kBlockBytes apart (16 in a ggml_tiles16_v1 matrix).
template <QType F, int Threads, int Stride = 1, class Sm, class RowOf>
__device__ __forceinline__ Accum row_bases(Sm& sm, RowOf row_of, int slices) {
    constexpr int kBlockElems = static_cast<int>(ggml_block_elems(F));
    constexpr int kBlockStride = Stride * static_cast<int>(ggml_block_bytes(F));
    constexpr int kPerRow      = Threads / kTileRows; // threads per row: 4 or 8
    static_assert(kPerRow == 4 || kPerRow == 8, "four or eight threads per row");
    const int r = static_cast<int>(threadIdx.x) / kPerRow, q = static_cast<int>(threadIdx.x) % kPerRow;
    const int blocks = slices * 32 / kBlockElems;
    const std::uint8_t* row = row_of(r);
    int lo = 31, hi = 0;
    for (int b = q; b < blocks; b += kPerRow) { scale_shift_range<F>(row + static_cast<std::size_t>(b) * kBlockStride, lo, hi); }
#pragma unroll
    for (int offset = 1; offset < kPerRow; offset <<= 1) {
        lo = min(lo, __shfl_xor_sync(0xFFFFFFFFU, lo, offset));
        hi = max(hi, __shfl_xor_sync(0xFFFFFFFFU, hi, offset));
    }
    if (q == 0) { sm.base[r] = lo; }
    const int log2_blocks = 32 - __clz(blocks - 1); // ceil(log2(blocks)); 0 for one block
    if constexpr (kLimbsMode<F, Threads>) {
        return __syncthreads_and(hi - lo <= 20 && 11 + (hi - lo) + kSumLog2<F> + log2_blocks <= 62) != 0 ? Accum::kLimbs
                                                                                                         : Accum::kWide;
    } else {
        return __syncthreads_and(hi - lo <= 62 - kTermLog2<F> - log2_blocks) != 0 ? Accum::kNarrow : Accum::kWide;
    }
}

// Stores chunk dp (elements 8 dp .. 8 dp + 7) of slice dj of tile row dr into a stage: its words
// w = 2 dp and 2 dp + 1 of the slice go to a[dj][dr / 16][dr % 8][w % 4][2 (w / 4) + (dr % 16) / 8].
template <QType F, class Stage = CodeStage>
__device__ __forceinline__ void store_chunk(Stage& cs, int dr, int dj, int dp, int w0, int w1, int sub,
                                            std::uint16_t d, int min_sub, std::uint16_t dmin) {
    const int grp = dr / 16, gr = dr % 8, hr = (dr % 16) / 8;
    cs.a[dj][grp][gr][(2 * dp) % 4][2 * ((2 * dp) / 4) + hr]         = w0;
    cs.a[dj][grp][gr][(2 * dp + 1) % 4][2 * ((2 * dp + 1) / 4) + hr] = w1;
    // Chunks 0-1 are elements 0-15, chunks 2-3 elements 16-31 (a UnitCodeStage has no sub-scales).
    if constexpr (!std::is_same_v<Stage, UnitCodeStage>) {
        if (dp % 2 == 0) { cs.sub[dj][grp][gr][2 * hr + dp / 2] = sub; }
    }
    if (dp == 0) {
        cs.d[dj][dr] = d;
        if constexpr (fast::kMin<F>) {
            cs.min[dj][dr]  = min_sub;
            cs.dmin[dj][dr] = dmin;
        }
    }
}

// Decodes stage st (slices kStageSlices st ..) of tile rows row_of(0 .. kTileRows - 1) from global
// memory into a stage (fast::chunk: the same integers as tile_chunk), with codebook `cb`. Every
// thread calls it.
template <QType F, int Threads = kThreads, class RowOf, class Codebook = fast::DeviceCodebook<F>>
__device__ __forceinline__ void decode_stage(CodeStage& cs, RowOf row_of, int st, const Codebook& cb = Codebook{}) {
    constexpr int kChunks = kTileRows * kStageSlices * 4 / Threads; // 8-element chunks per thread
    static_assert(kChunks * Threads == kTileRows * kStageSlices * 4);
#pragma unroll
    for (int i = 0; i < kChunks; ++i) {
        const int c  = static_cast<int>(threadIdx.x) + Threads * i;
        const int dr = c / (4 * kStageSlices), dj = (c / 4) % kStageSlices, dp = c % 4;
        const fast::Chunk ch = fast::chunk<F>(row_of(dr), st * kStageSlices + dj, dp, cb);
        store_chunk<F>(cs, dr, dj, dp, ch.w0, ch.w1, ch.sub, ch.d, ch.min, ch.dmin);
    }
}

// Issues the cp.async copies of stage st's image of a decoded plane (`plane`: the tile's images).
template <QType F, int Threads>
__device__ __forceinline__ void issue_image(CodeStage& cs, const std::uint8_t* plane, int st) {
    const std::uint8_t* src = plane + static_cast<std::size_t>(st) * sizeof(CodeStage);
    auto* dst               = reinterpret_cast<std::uint8_t*>(&cs);
    for (int i = static_cast<int>(threadIdx.x); i < kImageBytes<F> / 16; i += Threads) {
        cp_async16(dst + 16 * i, src + 16 * i, 16);
    }
}

// The resident tiles' stage type of format F.
template <QType F>
using ResidentStage = std::conditional_t<kUnitSub<F> && !fast::kMin<F>, UnitCodeStage, CodeStage>;

// Dynamic shared memory of a weight-stationary tile CTA: every one of its Stages stages decoded once
// (decode_resident into code[st]) and kept while it computes several column tiles, the columns'
// limbs in a ring of kBufs stages (the K loop copies kBufs - 1 stages ahead).
template <int Stages, class Stage = CodeStage>
struct ResidentShared {
    static constexpr int kBufs = 2;
    Stage code[Stages];
    alignas(16) int b[kBufs][kStageSlices][kTileColumns][kBWords];
    int xsum[kBufs][kStageSlices][kTileColumns];
    int base[kTileRows];
};

// Where the K loop's stages come from.
enum class StageSource {
    kDecode,   // the CTA copies raw units and decodes each stage (Shared)
    kPlane,    // images of a decoded plane, copied one stage ahead (PlaneShared)
    kResident, // decoded before the loop and kept (ResidentShared)
};

__device__ __forceinline__ CodeStage& stage_of(Shared& sm, int, int) { return sm.code; }
__device__ __forceinline__ CodeStage& stage_of(PlaneShared& sm, int buf, int) { return sm.code[buf]; }
template <int Stages, class Stage>
__device__ __forceinline__ Stage& stage_of(ResidentShared<Stages, Stage>& sm, int, int st) { return sm.code[st]; }

// The K loop of compute, accumulating by Mode (Accum; the rows' bases in sm.base for kNarrow and
// kLimbs); the warps' tiles are warp_tile<Threads>'s layout `Layout`, a warp of an inactive layout
// slot (512 threads, layout 2) copying and decoding but computing nothing. Source says where the
// stages come from (StageSource; `plane` for kPlane); kDecode reads codebook `cb`.
template <QType F, Accum Mode, int Layout, StageSource Source, int Threads, class Sm, class RowOf, class Codebook>
__device__ __forceinline__ void compute_loop(Sm& sm, RowOf row_of, int slices, const LimbCopies<Threads>& copies,
                                             Accumulators& acc, const std::uint8_t* plane, const Codebook& cb) {
    constexpr bool Narrow = Mode == Accum::kNarrow;
    constexpr bool Limbs  = Mode == Accum::kLimbs;
    static_assert(!Limbs || kLimbsMode<F, Threads>, "per-limb accumulators: kLimbsMode tiles only");
    constexpr int kBlockElems = static_cast<int>(ggml_block_elems(F));
    constexpr int kBlockBytes = static_cast<int>(ggml_block_bytes(F));
    constexpr int kUnitStages = kUnitSlices / kStageSlices;
    constexpr int kDecodeChunks = kTileRows * kStageSlices * 4 / Threads; // 8-element chunks per thread and stage
    static_assert(kDecodeChunks * Threads == kTileRows * kStageSlices * 4);
    constexpr int kM = kLayoutM<Threads, Layout>, kN = kLayoutN<Threads, Layout>;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int g = lane / 4, t = lane % 4;
    const WarpLayout<Threads> wl = warp_layout<Threads, Layout>();
    const int row0 = wl.row0, col0 = wl.col0;
    // Block sums of sub * (codes . limb) per limb plane (a, b, c): |sum| < 2^kSumLog2 <= 2^27.
    int s[3][kMTiles][kNTiles][4];
    long long mt[kMTiles][kNTiles][4];
    // Narrow: the outputs in units of 2^base of their rows (row ra = row0 + 16 mi + g, and ra + 8).
    long long n64[kMTiles][kNTiles][4];
    // Limbs: per limb plane l, the sum of M * block sum l in units of 2^base (row_bases' bound).
    long long nl[3][kMTiles][kNTiles][4];
    int base[kMTiles][2];
#pragma unroll
    for (int mi = 0; mi < kMTiles; ++mi) {
#pragma unroll
        for (int h = 0; h < 2; ++h) { base[mi][h] = (Narrow || Limbs) && mi < kM ? sm.base[row0 + 16 * mi + g + 8 * h] : 0; }
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
#pragma unroll
                for (int l = 0; l < 3; ++l) {
                    s[l][mi][nt][q]  = 0;
                    nl[l][mi][nt][q] = 0;
                }
                mt[mi][nt][q]    = 0;
                n64[mi][nt][q]   = 0;
                acc.r[mi][nt][q] = I128{};
            }
        }
    }
    constexpr int kBufs = Sm::kBufs;
    const int stages = slices / kStageSlices;
    for (int st = 0; st < stages; ++st) {
        const int s0 = st * kStageSlices, buf = st % kBufs;
        if constexpr (Source == StageSource::kResident) {
            // Ahead: stage st + kBufs - 1's limbs (compute_resident issued the first kBufs - 1; their
            // buffer's last reader, stage st - 1, finished at its closing barrier).
            constexpr int kAhead = kBufs - 1;
            if (st + kAhead < stages) { issue_limbs<F>(sm, (st + kAhead) % kBufs, s0 + kAhead * kStageSlices, copies); }
            cp_async_commit();
            cp_async_wait<kAhead>(); // stage st's copies have landed
            __syncthreads();
        } else if constexpr (Source == StageSource::kPlane) {
            // Ahead: stage st + kBufs - 1's limbs and image (compute_plane issued the first kBufs - 1;
            // their buffers' last reader, stage st - 1, finished at its closing barrier).
            constexpr int kAhead = kBufs - 1;
            if (st + kAhead < stages) {
                issue_limbs<F>(sm, (st + kAhead) % kBufs, s0 + kAhead * kStageSlices, copies);
                issue_image<F, Threads>(sm.code[(st + kAhead) % kBufs], plane, st + kAhead);
            }
            cp_async_commit();
            cp_async_wait<kAhead>(); // stage st's copies have landed
            __syncthreads();
        } else {
            const int unit = st / kUnitStages, ubuf = unit & 1;
            // Ahead: the next stage's limbs; at a unit's first stage, the next unit's raw bytes (their
            // buffer's last reader, unit - 1, finished at the previous stage's closing barrier).
            if (st + 1 < stages) { issue_limbs<F>(sm, buf ^ 1, s0 + kStageSlices, copies); }
            if (st % kUnitStages == 0 && (unit + 1) * kUnitSlices < slices) { issue_unit<F, Threads>(sm, ubuf ^ 1, unit + 1, slices, row_of); }
            cp_async_commit();
            cp_async_wait<1>(); // everything but this stage's issues has landed
            __syncthreads();
#pragma unroll
            for (int i = 0; i < kDecodeChunks; ++i) {
                const int c = static_cast<int>(threadIdx.x) + Threads * i;
                const int dr = c / (4 * kStageSlices), dj = (c / 4) % kStageSlices, dp = c % 4;
                // Element offset of this chunk within the unit's raw copy.
                const int e = (s0 + dj - unit * kUnitSlices) * 32 + 8 * dp;
                const std::uint8_t* block = &sm.raw[ubuf][dr][sm.raw_offset[ubuf][dr]] + e / kBlockElems * kBlockBytes;
                const TileChunk ch = tile_chunk<F>(block, e % kBlockElems / 8, cb);
                store_chunk<F>(sm.code, dr, dj, dp, ch.w[0], ch.w[1], ch.sub, ch.d, ch.min_sub, ch.dmin);
            }
            __syncthreads();
        }
        const auto& cs = stage_of(sm, buf, st);
#pragma unroll
        for (int j = 0; j < kStageSlices; ++j) {
            if (!wl.active) { break; } // warp-uniform
            // The limb fragments by ldmatrix: matrix f = (nt * 3 + l) * 2 + h holds columns
            // col0 + 8 nt .. + 7 (its rows) and words 8 l + 4 h .. + 3 of limb plane l, so register
            // f of lane (g, t) is word 8 l + 4 h + t of column col0 + 8 nt + g; three ldmatrix.x4 per
            // slice.
            int b0[kNTiles][3], b1[kNTiles][3];
            static_assert(kNTiles * 2 == 4, "one ldmatrix.x4 per limb plane and slice");
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                const int f = 4 * q + lane / 8, fnt = f / 6, fl = (f % 6) / 2, fh = f % 2;
                int r[4];
                ldmatrix_x4(r, &sm.b[buf][j][col0 + 8 * fnt + lane % 8][8 * fl + 4 * fh]);
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const int fm = 4 * q + m, nt = fm / 6, l = (fm % 6) / 2;
                    if (fm % 2 == 0) {
                        b0[nt][l] = r[m];
                    } else {
                        b1[nt][l] = r[m];
                    }
                }
            }
#pragma unroll
            for (int mi = 0; mi < kMTiles; ++mi) {
                if (mi >= kM) { continue; }
                const int ra = row0 + 16 * mi + g, grp = row0 / 16 + mi;
                // Rows ra and ra + 8: a = (row ra word t, row ra + 8 word t, the same of word 4 + t),
                // sc = (row ra's two sub-scales, row ra + 8's).
                const int4 a  = *reinterpret_cast<const int4*>(cs.a[j][grp][g][t]);
                int4 sc{1, 1, 1, 1};
                if constexpr (!kUnitSub<F>) { sc = *reinterpret_cast<const int4*>(cs.sub[j][grp][g]); }
                // An n-tile's MMAs, then their scaling, before the next n-tile's (the other n-tile's
                // MMAs overlap the scaling; fewer live products than all of them first).
#pragma unroll
                for (int nt = 0; nt < kNTiles; ++nt) {
                    if (nt >= kN) { continue; }
                    if constexpr (kHalfSub<F>) {
                        int p0[3][4], p1[3][4];
#pragma unroll
                        for (int l = 0; l < 3; ++l) {
                            mma_k16(p0[l], a.x, a.y, b0[nt][l]); // elements 0-15
                            mma_k16(p1[l], a.z, a.w, b1[nt][l]); // elements 16-31
                        }
#pragma unroll
                        for (int l = 0; l < 3; ++l) {
                            // One IMAD per product (o + sc * p, twice), not two products and an add.
                            int (&o)[4] = s[l][mi][nt];
                            o[0] += sc.x * p0[l][0] + sc.y * p1[l][0];
                            o[1] += sc.x * p0[l][1] + sc.y * p1[l][1];
                            o[2] += sc.z * p0[l][2] + sc.w * p1[l][2];
                            o[3] += sc.z * p0[l][3] + sc.w * p1[l][3];
                        }
                    } else {
                        int p[3][4];
#pragma unroll
                        for (int l = 0; l < 3; ++l) { mma_k32(p[l], a.x, a.y, a.z, a.w, b0[nt][l], b1[nt][l]); }
#pragma unroll
                        for (int l = 0; l < 3; ++l) {
                            int (&o)[4] = s[l][mi][nt];
                            if constexpr (kUnitSub<F>) {
                                o[0] += p[l][0];
                                o[1] += p[l][1];
                                o[2] += p[l][2];
                                o[3] += p[l][3];
                            } else {
                                o[0] += sc.x * p[l][0];
                                o[1] += sc.x * p[l][1];
                                o[2] += sc.z * p[l][2];
                                o[3] += sc.z * p[l][3];
                            }
                        }
                    }
                }
#pragma unroll
                for (int nt = 0; nt < kNTiles; ++nt) {
                    if (nt >= kN) { continue; }
                    if constexpr (fast::kMin<F>) {
                        const int m_lo = cs.min[j][ra], m_hi = cs.min[j][ra + 8];
                        const int col  = col0 + 8 * nt + 2 * t;
                        const int x0 = sm.xsum[buf][j][col], x1 = sm.xsum[buf][j][col + 1];
                        mt[mi][nt][0] += static_cast<long long>(m_lo) * x0;
                        mt[mi][nt][1] += static_cast<long long>(m_lo) * x1;
                        mt[mi][nt][2] += static_cast<long long>(m_hi) * x0;
                        mt[mi][nt][3] += static_cast<long long>(m_hi) * x1;
                    }
                }
                if ((s0 + j + 1) % kBlockSlices<F> == 0) {
                    // The block ends: its rows' d (and dmin) are those of any of its slices.
                    std::int64_t mant[2], min_mant[2] = {0, 0};
                    int shift[2], min_shift[2] = {0, 0};
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        bool negative;
                        f16_parts(cs.d[j][ra + 8 * h], mant[h], shift[h], negative);
                        if (negative) { mant[h] = -mant[h]; }
                        if constexpr (fast::kMin<F>) {
                            f16_parts(cs.dmin[j][ra + 8 * h], min_mant[h], min_shift[h], negative);
                            min_mant[h] = negative ? min_mant[h] : -min_mant[h]; // the min term is subtracted
                        }
                    }
                    if constexpr (Limbs) {
                        // M = +-mant(d) * 2^(shift(d) - base) (|M| < 2^31, row_bases); each limb
                        // plane's block sum is added as M * sum (one wide multiply-add).
                        int m32[2];
#pragma unroll
                        for (int h = 0; h < 2; ++h) { m32[h] = static_cast<int>(mant[h]) * (1 << (shift[h] - base[mi][h])); }
#pragma unroll
                        for (int nt = 0; nt < kNTiles; ++nt) {
                            if (nt >= kN) { continue; }
#pragma unroll
                            for (int q = 0; q < 4; ++q) {
#pragma unroll
                                for (int l = 0; l < 3; ++l) {
                                    nl[l][mi][nt][q] += static_cast<long long>(m32[q >> 1]) * s[l][mi][nt][q];
                                    s[l][mi][nt][q] = 0;
                                }
                            }
                        }
                    } else {
#pragma unroll
                        for (int nt = 0; nt < kNTiles; ++nt) {
                            if (nt >= kN) { continue; }
#pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                const int h = q >> 1;
                                const long long p = (static_cast<long long>(s[0][mi][nt][q]) << 16) +
                                                    (static_cast<long long>(s[1][mi][nt][q]) << 8) + s[2][mi][nt][q];
                                if constexpr (Narrow) {
                                    n64[mi][nt][q] += shl64(mant[h] * p, shift[h] - base[mi][h]);
                                } else {
                                    add_shifted(acc.r[mi][nt][q], mant[h] * p, shift[h]);
                                }
#pragma unroll
                                for (int l = 0; l < 3; ++l) { s[l][mi][nt][q] = 0; }
                                if constexpr (fast::kMin<F>) {
                                    if constexpr (Narrow) {
                                        n64[mi][nt][q] += shl64(min_mant[h] * mt[mi][nt][q], min_shift[h] - base[mi][h]);
                                    } else {
                                        add_shifted(acc.r[mi][nt][q], min_mant[h] * mt[mi][nt][q], min_shift[h]);
                                    }
                                    mt[mi][nt][q] = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
        __syncthreads(); // the next stage's decode and copies reuse the codes and these buffers
    }
    if constexpr (Narrow || Limbs) {
#pragma unroll
        for (int mi = 0; mi < kMTiles; ++mi) {
#pragma unroll
            for (int nt = 0; nt < kNTiles; ++nt) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int b = base[mi][q >> 1];
                    if constexpr (Narrow) {
                        add_shifted(acc.r[mi][nt][q], n64[mi][nt][q], b);
                    } else { // limb planes a, b, c weigh 2^16, 2^8, 1
                        add_shifted(acc.r[mi][nt][q], nl[0][mi][nt][q], b + 16);
                        add_shifted(acc.r[mi][nt][q], nl[1][mi][nt][q], b + 8);
                        add_shifted(acc.r[mi][nt][q], nl[2][mi][nt][q], b);
                    }
                }
            }
        }
    }
}

// compute_loop for the CTA's accumulator mode and warp layout (the same for every warp).
template <QType F, StageSource Source, int Threads, class Sm, class RowOf, class Codebook = fast::DeviceCodebook<F>>
__device__ __forceinline__ void dispatch_loop(Sm& sm, RowOf row_of, int slices, const LimbCopies<Threads>& copies,
                                              int columns, Accum mode, Accumulators& acc, const std::uint8_t* plane,
                                              const Codebook& cb = Codebook{}) {
    const auto run = [&](auto layout) {
        constexpr int kLayout = decltype(layout)::value;
        constexpr Accum kInt64 = kLimbsMode<F, Threads> ? Accum::kLimbs : Accum::kNarrow; // row_bases' int64 mode
        mode == kInt64 ? compute_loop<F, kInt64, kLayout, Source, Threads>(sm, row_of, slices, copies, acc, plane, cb)
                       : compute_loop<F, Accum::kWide, kLayout, Source, Threads>(sm, row_of, slices, copies, acc, plane, cb);
    };
    if (columns > 32) {
        run(std::integral_constant<int, 0>{});
    } else if (columns > 16) {
        run(std::integral_constant<int, 1>{});
    } else {
        run(std::integral_constant<int, 2>{});
    }
}

// Computes a tile of kTileRows weight rows (tile row r at row_of(r), a valid row even where the
// caller ignores its outputs) against kTileColumns columns: column c's limbs for slice s at
// limbs(c) + s * kSliceWords (16-byte aligned), its slice sums (min formats only) at
// limbs(c) + xsum_offset + s. `slices` is K / 32, a multiple of kStageSlices. `columns` is the
// tile's valid columns: the thread's outputs (Accumulators) are those of its warp tile
// (warp_tile<Threads>(columns)); columns past the valid ones are computed on padding (column limbs
// that the caller clamps to a valid column) and ignored. Every thread of the CTA (Threads of them)
// calls it. The outputs are the same integers whichever accumulator the rows' scales allow. The
// decode reads codebook `cb`: a fast::SharedCodebook the caller loaded before the call is published
// by row_bases' barrier, before the first decode.
template <QType F, int Threads = kThreads, class RowOf, class ColumnLimbs, class Codebook = fast::DeviceCodebook<F>>
__device__ __forceinline__ void compute(Shared& sm, RowOf row_of, int slices, ColumnLimbs limbs, int xsum_offset,
                                        int columns, Accumulators& acc, const Codebook& cb = Codebook{}) {
    const LimbCopies<Threads> copies = limb_copies<F, Threads>(limbs, xsum_offset);
    issue_unit<F, Threads>(sm, 0, 0, slices, row_of);
    issue_limbs<F>(sm, 0, 0, copies);
    cp_async_commit();
    const Accum mode = row_bases<F, Threads>(sm, row_of, slices);
    dispatch_loop<F, StageSource::kDecode>(sm, row_of, slices, copies, columns, mode, acc, nullptr, cb);
}

// compute over a decoded plane: stage st's image at plane + st * sizeof(CodeStage) (decode_planes);
// row_of(r) still gives the raw rows, whose scales choose the accumulator (blocks Stride *
// kBlockBytes apart: row_bases).
template <QType F, int Threads = kThreads, int Stride = 1, class RowOf, class ColumnLimbs>
__device__ __forceinline__ void compute_plane(PlaneShared& sm, RowOf row_of, const std::uint8_t* plane, int slices,
                                              ColumnLimbs limbs, int xsum_offset, int columns, Accumulators& acc) {
    const LimbCopies<Threads> copies = limb_copies<F, Threads>(limbs, xsum_offset);
    const int stages = slices / kStageSlices;
#pragma unroll
    for (int k = 0; k < PlaneShared::kBufs - 1; ++k) { // the first kBufs - 1 stages, one group each
        if (k < stages) {
            issue_image<F, Threads>(sm.code[k], plane, k);
            issue_limbs<F>(sm, k, k * kStageSlices, copies);
        }
        cp_async_commit();
    }
    const Accum mode = row_bases<F, Threads, Stride>(sm, row_of, slices);
    dispatch_loop<F, StageSource::kPlane>(sm, row_of, slices, copies, columns, mode, acc, plane);
}

// Weight-stationary tiles: decode_resident decodes every stage of the tile's rows into sm.code once
// and returns row_bases' accumulator mode; compute_resident then computes one tile of columns
// against them (as compute does), any number of times. Every thread of the CTA calls both.
// The rows' raw bytes arrive by cp.async in one batch (each row from its 16-byte aligned start, in
// sm.b, which the K loop fills only later), then every stage decodes from shared memory (tile_chunk):
// one global latency per tile instead of one per stage. (Decoding each stage from global memory
// was 12 % of the IQ4_NL down tiles' stall samples, 2026-10-14.)
template <QType F, int Stages, int Threads = kThreads, class Stage, class RowOf>
__device__ __forceinline__ Accum decode_resident(ResidentShared<Stages, Stage>& sm, RowOf row_of) {
    constexpr int kBlockElems = static_cast<int>(ggml_block_elems(F));
    constexpr int kBlockBytes = static_cast<int>(ggml_block_bytes(F));
    constexpr int kRowBytes   = Stages * kStageSlices * 32 / kBlockElems * kBlockBytes;
    constexpr int kRowChunks  = (15 + kRowBytes + 15) / 16; // 16-byte chunks from the aligned start
    constexpr int kChunks     = kTileRows * kStageSlices * 4 / Threads; // 8-element chunks per thread and stage
    static_assert(kChunks * Threads == kTileRows * kStageSlices * 4);
    static_assert(kTileRows * kRowChunks * 16 <= static_cast<int>(sizeof(sm.b)), "a tile's raw rows fit the limb buffers");
    auto* raw = reinterpret_cast<std::uint8_t*>(&sm.b[0][0][0][0]);
    const auto offset = [&](int r) { return static_cast<int>(reinterpret_cast<std::uintptr_t>(row_of(r)) & 15U); };
    for (int i = static_cast<int>(threadIdx.x); i < kTileRows * kRowChunks; i += Threads) {
        const int r = i / kRowChunks, c = i % kRowChunks;
        const std::uint8_t* aligned = row_of(r) - offset(r);
        const int need = offset(r) + kRowBytes - 16 * c; // bytes of this chunk the row reads
        if (need > 0) { cp_async16(raw + (r * kRowChunks + c) * 16, aligned + 16 * c, need < 16 ? need : 16); }
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
#pragma unroll 1
    for (int st = 0; st < Stages; ++st) {
#pragma unroll
        for (int i = 0; i < kChunks; ++i) {
            const int c  = static_cast<int>(threadIdx.x) + Threads * i;
            const int dr = c / (4 * kStageSlices), dj = (c / 4) % kStageSlices, dp = c % 4;
            const int e  = (st * kStageSlices + dj) * 32 + 8 * dp; // element offset in the row
            const std::uint8_t* block = raw + dr * kRowChunks * 16 + offset(dr) + e / kBlockElems * kBlockBytes;
            const TileChunk ch = tile_chunk<F>(block, e % kBlockElems / 8);
            store_chunk<F, Stage>(sm.code[st], dr, dj, dp, ch.w[0], ch.w[1], ch.sub, ch.d, ch.min_sub, ch.dmin);
        }
    }
    return row_bases<F, Threads>(sm, row_of, Stages * kStageSlices); // its barrier also publishes the stages
}

template <QType F, int Stages, int Threads = kThreads, class Stage, class ColumnLimbs>
__device__ __forceinline__ void compute_resident(ResidentShared<Stages, Stage>& sm, Accum mode, ColumnLimbs limbs,
                                                 int xsum_offset, int columns, Accumulators& acc) {
    using Sm = ResidentShared<Stages, Stage>;
    const LimbCopies<Threads> copies = limb_copies<F, Threads>(limbs, xsum_offset);
#pragma unroll
    for (int k = 0; k < Sm::kBufs - 1; ++k) { // the first kBufs - 1 stages' limbs, one group each
        if (k < Stages) { issue_limbs<F>(sm, k, k * kStageSlices, copies); }
        cp_async_commit();
    }
    const auto no_rows = [](int) -> const std::uint8_t* { return nullptr; };
    dispatch_loop<F, StageSource::kResident>(sm, no_rows, Stages * kStageSlices, copies, columns, mode, acc, nullptr);
}

} // namespace infernix::ggml::tile
