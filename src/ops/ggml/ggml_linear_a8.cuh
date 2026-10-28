#pragma once

// The 8-bit activation route of ops::ggml_linear (LinearRoutes::GgmlA8, `--a8-activations`: LOSSY, opt-in;
// docs/maintainer/qwen3_8-flash-next-design.md §19.3.31): calls of 1-31 columns over GGUF block weights, each
// column quantized per 32-element slice by ops/ggml/ggml_a8.h's codec (int8 codes, an FP32 scale d, the code
// sums of the K-quants' min term). A slice's dot with a row's codes and sub-scales is an exact int32 (dp4a or
// an int8 MMA); the row adds the slice's FP32 terms (a8::add_term, then a8::sub_min_term) to an FP32
// accumulator and rounds once to the output (a8::output_f32, then BF16 for a BF16 output). The route keeps the
// exact narrow routes' structure (ops/ggml/ggml_linear.cu) with one int8 plane where they carry three limbs:
//
// - encode (ggml_linear_a8.cu): four threads per slice, each column's a8::column_words at a 16-byte stride.
// - rows (ggml_linear_a8.cu, dp4a): Rows-layout calls of 1-2 columns and Tiles16 Q6_K / IQ4_XS calls of one
//   column, where the exact route also takes its rows kernel. `Lanes` lanes per row (the exact route's
//   lanes_for(K / 32)): lane p adds slices p, p + Lanes, ... in ascending order, then the lanes pair as a
//   butterfly, p_i = p_i + p_(i + o) for o = Lanes / 2, ..., 1.
// - mma (ggml_linear_a8_rows.cu, ggml_linear_a8_tiled.cu): every other call. A CTA of W warps holds W / split
//   tiles of 16 rows; a tile's `split` warps take its 256-element units in turn (unit u to part u % split),
//   each staged by 16-byte cp.async (L2 evict-first) into the warp's own ring (a Tiles16 unit is the tile's
//   contiguous block run, a Rows unit each row's bytes from its 16-byte aligned start). Lane (g, t) decodes
//   chunk t of each slice of rows g and g + 8 (tile::tile_chunk) as the A fragment of one m16n8k32 s8 MMA per
//   8-column tile (two, each half's lanes alone, for formats whose sub-scale covers 16 elements), with
//   column g's code words 2t, 2t + 1 as B, and adds the four terms of rows g, g + 8 and columns 2t, 2t + 1
//   (the scales of those two columns). A warp's NT column tiles (1, 2 or 4: up to 8, 16 or 32 columns)
//   share each decoded chunk, so a call of up to 31 columns reads and decodes its weights once (the exact
//   route's wide tiles take a second column group past 16). A part adds its units' slices in ascending
//   order; part 0 then adds parts 1, 2, ... in turn, which leave their partials in their own rings.
//
// So an output's FP32 order is fixed by K (rows) or by the CTA shape the call selects (mma: the split, from
// the call's rows, NT and the device's residency): a call's bits are deterministic, and a column's do not
// depend on the other columns' values, but they may change with the call's width or rows.

#include "core/device.h"
#include "core/ggml_format.h"
#include "core/pdl.cuh"
#include "infernix/ops/ggml_linear.h"
#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_a8.h"
#include "ops/ggml/ggml_dp4a.cuh"
#include "ops/ggml/ggml_mma_rows.cuh"
#include "ops/ggml/ggml_tile.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix::ops::a8_linear {

namespace a8    = ggml::a8;
namespace fast  = ggml::fast;
namespace tile  = ggml::tile;
namespace canon = ops::canon;

constexpr int kEncodeThreads = 256;
constexpr int kRowsThreads   = 128; // the rows kernel's CTA: four warps (the exact rows_kernel's)
constexpr int kRowsWarps     = kRowsThreads / 32;
constexpr int kRowsMaxColumns = 2;  // Rows-layout calls of at most this many columns take the rows kernel
constexpr int kMmaMinColumns  = 3;  // a Rows-layout mixed (two-format) launch from this width (the exact route's)
constexpr int kMmaMaxThreads  = 256;
constexpr int kUnitSlices     = ggml::mma_rows::kUnitElems / 32; // a staged unit: 256 elements

inline void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("ggml_linear (a8): ") + message); }
}

inline void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("ggml_linear a8 ") + what + ": " + cudaGetErrorString(error)); }
}

// Words between consecutive encoded columns: a8::column_words rounded up to 16 bytes, so every column's code
// words take 16-byte loads.
[[nodiscard]] constexpr int column_stride(int k, bool sums) {
    return static_cast<int>((a8::column_words(k, sums) + 3) / 4 * 4);
}

// The column tiles of 8 a warp holds for a call of `columns` columns (NT of the mma kernels).
[[nodiscard]] constexpr int column_tiles_for(int columns) { return columns <= 8 ? 1 : columns <= 16 ? 2 : 4; }

// Up to kMax segments in one launch (ggml_linear.cu's RowsGroup): CTAs [first[i], first[i + 1]) take segment
// i's rows into out[i] (n[i] rows per column); a mixed launch's format[i] picks F0 (0) or F1 (1).
struct A8Group {
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

// A CTA's segment of its launch and its block index within that segment.
struct Slot {
    const std::uint8_t* weights;
    void* out;
    int rows, n, out_offset, block, row_bytes, format;
};

__device__ __forceinline__ Slot slot_at(const A8Group& g, int b) {
    int i = 0;
#pragma unroll
    for (int j = 1; j < A8Group::kMax; ++j) { i = j < g.count && b >= g.first[j] ? j : i; }
    // Selects with constant indices, so the parameters are read from the constant bank in place.
    Slot s{g.data[0], g.out[0], g.rows[0], g.n[0], g.out_offset[0], b - g.first[0], g.row_bytes[0], g.format[0]};
#pragma unroll
    for (int j = 1; j < A8Group::kMax; ++j) {
        if (i == j) { s = Slot{g.data[j], g.out[j], g.rows[j], g.n[j], g.out_offset[j], b - g.first[j], g.row_bytes[j], g.format[j]}; }
    }
    return s;
}

// A call's encoded columns.
struct Columns {
    const std::int32_t* encoded = nullptr;
    int stride                  = 0; // words between columns (column_stride)
    int k                       = 0;
    int columns                 = 0;
};

// A weight block's binary16 scale in binary32: a8::f16_to_f32's value in one conversion (binary16 to binary32
// is exact; a NaN scale gives NaN outputs either way, and the GPU's FP32 arithmetic returns its canonical NaN).
__device__ __forceinline__ float weight_scale(std::uint16_t h) { return __half2float(__ushort_as_half(h)); }

// A 16-byte cp.async of which the first `bytes` are read (the rest zero-filled), with an L2 cache policy:
// evict-first for the dense weights, read once per call (ggml_linear.cu's cp_async16_evict_first).
__device__ __forceinline__ void cp_async16_hint(void* shared, const void* global, int bytes, std::uint64_t policy) {
    const auto s = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n" ::"r"(s), "l"(global), "r"(bytes),
                 "l"(policy));
}

// Waits until at most n of the thread's cp.async groups are pending (n < 4).
__device__ __forceinline__ void cp_async_wait_upto(int n) {
    switch (n) {
    case 0: tile::cp_async_wait<0>(); break;
    case 1: tile::cp_async_wait<1>(); break;
    case 2: tile::cp_async_wait<2>(); break;
    default: tile::cp_async_wait<3>(); break;
    }
}

// ------------------------------------------------------------------------------------------- the mma route

// A ring slot of format F: a Tiles16 unit is the tile's block run (16 rows' block, contiguous); a Rows unit
// is each row's bytes of the unit from its 16-byte aligned start, mma_rows::kUnitChunks copies a row.
template <QType F, bool Tiled>
inline constexpr int kRowPitch = Tiled ? static_cast<int>(ggml_block_bytes(F)) : 16 * ggml::mma_rows::kUnitChunks<F>;
template <QType F, bool Tiled>
inline constexpr int kSlotBytes = 16 * kRowPitch<F, Tiled>;

// Bytes of one warp's partial sums (NT column tiles of four FP32 outputs a lane), which a split tile's parts
// leave in their rings: every ring (two slots at least) holds four tiles' (kSlotBytes >= 1,536).
inline constexpr int kPartialBytesPerTile = 4 * 32 * static_cast<int>(sizeof(float));

// Column tiles of a warp: for tile nt, lane (g, t)'s B column 8 nt + g (its code words, offset by 2t: the
// B fragment's words 2t, 2t + 1 of each slice), and its output columns 8 nt + 2t and 8 nt + 2t + 1 (their
// scales and code sums). Columns past the call's read its last column (outputs that are not stored).
template <int NT>
struct ColumnTiles {
    const std::int32_t* codes[NT];
    const float* scale0[NT];
    const float* scale1[NT];
    const std::int32_t* sum0[NT];
    const std::int32_t* sum1[NT];
};

template <int NT>
__device__ __forceinline__ ColumnTiles<NT> column_tiles(const std::int32_t* encoded, int stride, int k, int columns, int g,
                                                        int t) {
    ColumnTiles<NT> c;
    const auto column = [&](int i) { return encoded + static_cast<std::size_t>(min(i, columns - 1)) * stride; };
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        c.codes[nt]  = column(8 * nt + g) + 2 * t;
        c.scale0[nt] = reinterpret_cast<const float*>(column(8 * nt + 2 * t) + a8::scales_offset(k));
        c.scale1[nt] = reinterpret_cast<const float*>(column(8 * nt + 2 * t + 1) + a8::scales_offset(k));
        c.sum0[nt]   = column(8 * nt + 2 * t) + a8::sums_offset(k);
        c.sum1[nt]   = column(8 * nt + 2 * t + 1) + a8::sums_offset(k);
    }
    return c;
}

// The terms of one staged unit (its first `unit_slices` slices; slice s0 of K first) of rows g and g + 8,
// whose bytes rg and rg8 hold in shared memory (any alignment), against the first `live` column tiles, added
// into acc slice by slice in K order. acc[nt][q]: row g + 8 (q >> 1), column 8 nt + 2t + (q & 1).
template <QType F, int NT>
__device__ __forceinline__ void unit_terms(float (&acc)[NT][4], const std::uint8_t* rg, const std::uint8_t* rg8, int s0,
                                           int unit_slices, const ColumnTiles<NT>& ct, int live, int t) {
    constexpr int kBlockElems = static_cast<int>(ggml_block_elems(F));
    constexpr int kBlockBytes = static_cast<int>(ggml_block_bytes(F));
#pragma unroll
    for (int j = 0; j < kUnitSlices; ++j) {
        if (j >= unit_slices) { break; } // uniform: a Rows matrix's short last unit
        const int e = 32 * j + 8 * t;    // the lane's chunk of slice j within the unit
        const tile::TileChunk cg = tile::tile_chunk<F>(rg + e / kBlockElems * kBlockBytes, e % kBlockElems / 8);
        const tile::TileChunk ch = tile::tile_chunk<F>(rg8 + e / kBlockElems * kBlockBytes, e % kBlockElems / 8);
        const float wg = weight_scale(cg.d), wh = weight_scale(ch.d);
        [[maybe_unused]] float mg = 0.0F, mh = 0.0F;
        if constexpr (fast::kMin<F>) {
            mg = weight_scale(cg.dmin);
            mh = weight_scale(ch.dmin);
        }
        // Rows g's and g + 8's sub-scales of the other 16-element half (lane t ^ 2's chunks; ggml_mma_rows.cuh).
        [[maybe_unused]] int og = 0, oh = 0;
        if constexpr (tile::kHalfSub<F>) {
            const int packed = (cg.sub & 0xFFFF) | (ch.sub << 16);
            const int other  = __shfl_xor_sync(0xFFFFFFFFU, packed, 2);
            og               = static_cast<short>(other & 0xFFFF);
            oh               = other >> 16;
        }
        const int s = s0 + j;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            if (nt >= live) { break; } // uniform: a tile of padding columns only
            const int2 b    = __ldg(reinterpret_cast<const int2*>(ct.codes[nt] + static_cast<std::size_t>(s) * a8::kSliceWords));
            const float x0  = __ldg(ct.scale0[nt] + s);
            const float x1  = __ldg(ct.scale1[nt] + s);
            int isum[4];
            if constexpr (tile::kHalfSub<F>) {
                // One MMA per half, the other half's lanes giving zero codes; isum = sub0 p0 + sub1 p1 (exact).
                isum[0] = isum[1] = isum[2] = isum[3] = 0;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const bool mine = (t >> 1) == h;
                    int p[4];
                    tile::mma_k32(p, mine ? cg.w[0] : 0, mine ? ch.w[0] : 0, mine ? cg.w[1] : 0, mine ? ch.w[1] : 0, b.x, b.y);
                    const int sg = mine ? cg.sub : og, sh = mine ? ch.sub : oh;
                    isum[0] += sg * p[0];
                    isum[1] += sg * p[1];
                    isum[2] += sh * p[2];
                    isum[3] += sh * p[3];
                }
            } else {
                int p[4];
                tile::mma_k32(p, cg.w[0], ch.w[0], cg.w[1], ch.w[1], b.x, b.y);
                if constexpr (tile::kUnitSub<F>) {
                    isum[0] = p[0];
                    isum[1] = p[1];
                    isum[2] = p[2];
                    isum[3] = p[3];
                } else {
                    isum[0] = cg.sub * p[0];
                    isum[1] = cg.sub * p[1];
                    isum[2] = ch.sub * p[2];
                    isum[3] = ch.sub * p[3];
                }
            }
            acc[nt][0] = a8::add_term(acc[nt][0], isum[0], x0, wg);
            acc[nt][1] = a8::add_term(acc[nt][1], isum[1], x1, wg);
            acc[nt][2] = a8::add_term(acc[nt][2], isum[2], x0, wh);
            acc[nt][3] = a8::add_term(acc[nt][3], isum[3], x1, wh);
            if constexpr (fast::kMin<F>) {
                const int q0 = __ldg(ct.sum0[nt] + s), q1 = __ldg(ct.sum1[nt] + s);
                acc[nt][0] = a8::sub_min_term(acc[nt][0], cg.min_sub, q0, x0, mg);
                acc[nt][1] = a8::sub_min_term(acc[nt][1], cg.min_sub, q1, x1, mg);
                acc[nt][2] = a8::sub_min_term(acc[nt][2], ch.min_sub, q0, x0, mh);
                acc[nt][3] = a8::sub_min_term(acc[nt][3], ch.min_sub, q1, x1, mh);
            }
        }
    }
}

// Whether F has a ggml_tiles16_v1 layout, as a host-evaluated constant (ggml_tiles16_supported is a host
// constexpr function, which device code may not call).
template <QType F>
inline constexpr bool kTiles16Format = ggml_tiles16_supported(F);

// One CTA of the mma route: segment `seg` of its launch (seg.row_bytes per row; Tiled: a ggml_tiles16_v1
// matrix, tile i at 16 i row_bytes), `split` warps per tile, rings of `stages` slots of slot_bytes (the
// launch's largest kSlotBytes) in `shared`. NT column tiles of 8 cover the call's columns.
template <QType F, bool OutF32, int NT, bool Tiled>
__device__ __forceinline__ void mma_body(const Slot& seg, int slices, const std::int32_t* __restrict__ encoded, int stride,
                                         int columns, int split, int stages, int slot_bytes, unsigned char* shared) {
    static_assert(!Tiled || kTiles16Format<F>, "tiled rows: formats of 256-element blocks");
    constexpr int kBlockBytes                  = static_cast<int>(ggml_block_bytes(F));
    [[maybe_unused]] constexpr int kUnitBytes = ggml::mma_rows::kUnitBytes<F>;   // Rows
    [[maybe_unused]] constexpr int kChunks    = ggml::mma_rows::kUnitChunks<F>;  // Rows
    [[maybe_unused]] constexpr int kPitch     = kRowPitch<F, Tiled>;             // Rows
    const int warps = static_cast<int>(blockDim.x) / 32, warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int per_cta = warps / split, slot = warp / split, part = warp % split;
    const int tiles = (seg.rows + 15) / 16, tile_id = seg.block * per_cta + slot;
    const bool active = tile_id < tiles;
    const int units = (slices + kUnitSlices - 1) / kUnitSlices;
    const int mine  = active && part < units ? (units - part + split - 1) / split : 0; // the warp's units
    const int row0  = (active ? tile_id : 0) * 16;
    unsigned char* ring = shared + static_cast<std::size_t>(warp) * stages * slot_bytes;
    // Rows: row r of the tile (rows past the segment's last read its last; their outputs are not stored).
    [[maybe_unused]] const auto row = [&](int r) {
        return seg.weights + static_cast<std::size_t>(min(row0 + r, seg.rows - 1)) * seg.row_bytes;
    };
    [[maybe_unused]] const std::uint8_t* tile_base = seg.weights + static_cast<std::size_t>(row0) * seg.row_bytes; // Tiled
    std::uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    const auto issue = [&](int i) { // the warp's i-th unit into its ring slot, one cp.async group
        if (i < mine) {
            const int u        = part + i * split;
            unsigned char* dst = ring + (i % stages) * slot_bytes;
            if constexpr (Tiled) {
                const std::uint8_t* src = tile_base + static_cast<std::size_t>(u) * 16 * kBlockBytes;
                for (int c = lane; c < kBlockBytes; c += 32) { cp_async16_hint(dst + 16 * c, src + 16 * c, 16, policy); }
            } else {
                const int bytes = ggml::mma_rows::unit_row_bytes<F>(u, slices);
                for (int c = lane; c < 16 * kChunks; c += 32) {
                    const int r = c / kChunks, cc = c % kChunks;
                    const std::uint8_t* src = row(r) + static_cast<std::size_t>(u) * kUnitBytes;
                    const auto* aligned =
                        reinterpret_cast<const std::uint8_t*>(reinterpret_cast<std::uintptr_t>(src) & ~std::uintptr_t{15});
                    const int need = static_cast<int>(src - aligned) + bytes - 16 * cc; // bytes of this copy the unit reads
                    if (need > 0) { cp_async16_hint(dst + r * kPitch + 16 * cc, aligned + 16 * cc, need < 16 ? need : 16, policy); }
                }
            }
        }
        tile::cp_async_commit();
    };
    const int g = lane / 4, t = lane % 4;
    const ColumnTiles<NT> ct = column_tiles<NT>(encoded, stride, slices * 32, columns, g, t);
    const int live = min(NT, (columns + 7) / 8);
    float acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int q = 0; q < 4; ++q) { acc[nt][q] = 0.0F; }
    }
    // The first stages - 1 units are immutable weights: copied while the encoder still runs.
    for (int i = 0; i < stages - 1; ++i) { issue(i); }
    pdl::wait_for_dependencies();
    for (int i = 0; i < mine; ++i) {
        __syncwarp(); // slot (i - 1) % stages, which the next issue fills, was read at i - 1
        issue(i + stages - 1);
        cp_async_wait_upto(stages - 1); // unit i has landed
        __syncwarp();
        const unsigned char* run = ring + (i % stages) * slot_bytes;
        const int u = part + i * split;
        if constexpr (Tiled) {
            unit_terms<F, NT>(acc, run + g * kBlockBytes, run + (g + 8) * kBlockBytes, kUnitSlices * u, kUnitSlices, ct, live, t);
        } else {
            const auto offset = [&](int r) {
                return static_cast<int>(reinterpret_cast<std::uintptr_t>(row(r) + static_cast<std::size_t>(u) * kUnitBytes) & 15U);
            };
            unit_terms<F, NT>(acc, run + g * kPitch + offset(g), run + (g + 8) * kPitch + offset(g + 8), kUnitSlices * u,
                              min(kUnitSlices, slices - kUnitSlices * u), ct, live, t);
        }
    }
    tile::cp_async_wait<0>();
    __syncwarp(); // the warp's ring is free: every copy has landed and been read
    if (split > 1) {
        // Parts 1 .. split - 1 leave their partials in their own rings; part 0 adds them in that order.
        if (part > 0) {
            auto* own = reinterpret_cast<float*>(ring);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                if (nt >= live) { break; }
#pragma unroll
                for (int q = 0; q < 4; ++q) { own[(nt * 4 + q) * 32 + lane] = acc[nt][q]; }
            }
        }
        __syncthreads();
        if (part == 0) {
            for (int p = 1; p < split; ++p) {
                const auto* other = reinterpret_cast<const float*>(shared + static_cast<std::size_t>(warp + p) * stages * slot_bytes);
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    if (nt >= live) { break; }
#pragma unroll
                    for (int q = 0; q < 4; ++q) { acc[nt][q] = canon::add_rn(acc[nt][q], other[(nt * 4 + q) * 32 + lane]); }
                }
            }
        }
    }
    if (part != 0 || !active) { return; }
    const int kappa = ggml::kappa_log2(F);
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        if (nt >= live) { break; }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int r = row0 + g + 8 * (q >> 1), c = 8 * nt + 2 * t + (q & 1);
            if (r < seg.rows && c < columns) {
                const std::size_t o = static_cast<std::size_t>(c) * seg.n + seg.out_offset + r;
                const float y       = a8::output_f32(acc[nt][q], kappa);
                if constexpr (OutF32) {
                    static_cast<float*>(seg.out)[o] = y;
                } else {
                    static_cast<std::uint16_t*>(seg.out)[o] = canon::f32_to_bf16_rn(y);
                }
            }
        }
    }
}

// ---- the mma route's CTA shape (ggml_linear.cu's tiled narrow rule, the same structure)
//
// The first of kShapes whose CTAs all fit in one wave (cudaOccupancyMaxActiveBlocksPerMultiprocessor x SMs)
// and whose split either keeps the call within kWarpBudget warps or leaves each warp kMinUnits units; a call
// no shape fits in one wave takes kManyWaves. The exact route measured this rule on the RTX 5090 at the IQ3_S
// import's dense calls (a second wave costs a narrow call 25-30 %; more warps help small calls); the 8-bit
// kernels keep its per-warp rings and split, so the same reasons hold (their own timing is still owed).
struct Shape {
    int warps  = 4;
    int split  = 1;
    int stages = 2;
};
constexpr Shape kShapes[] = {{8, 8, 2}, {4, 4, 2}, {2, 2, 2}, {4, 2, 2}, {4, 1, 2}, {2, 1, 2}};
constexpr Shape kManyWaves{4, 1, 2};
constexpr int kWarpBudget = 1024;
constexpr int kMinUnits   = 4;

// The rings are the CTA's only shared memory (the partials reuse them).
[[nodiscard]] inline std::size_t mma_shared_bytes(const Shape& s, int slot_bytes) {
    return static_cast<std::size_t>(s.warps) * s.stages * slot_bytes;
}

struct Device {
    int sms   = 0;
    int limit = 0; // opt-in shared memory per block
};

inline Device device() {
    thread_local int current_device = -1;
    thread_local Device d;
    int current = 0;
    CUDA_CHECK(cudaGetDevice(&current));
    if (current != current_device) {
        CUDA_CHECK(cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, current));
        CUDA_CHECK(cudaDeviceGetAttribute(&d.limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, current));
        current_device = current;
    }
    return d;
}

// Lets `kernel` take up to the device's opt-in shared memory (once per kernel and thread).
template <class Kernel>
void allow_shared(Kernel kernel, int limit) {
    thread_local std::vector<const void*> done;
    const void* key = reinterpret_cast<const void*>(kernel);
    if (std::find(done.begin(), done.end(), key) != done.end()) { return; }
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, limit));
    done.push_back(key);
}

// `kernel`'s resident CTAs per SM at `threads` threads and `shared` bytes (cached per thread).
template <class Kernel>
int resident(Kernel kernel, int threads, std::size_t shared) {
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
    int count = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&count, kernel, threads, shared));
    cache.push_back({key, threads, shared, count});
    return count;
}

template <class Kernel>
Shape choose_shape(Kernel kernel, int tiles, int units, int slot_bytes, const Device& dev) {
    for (const Shape& s : kShapes) {
        if (s.split > 1 && tiles * s.split > kWarpBudget && units < kMinUnits * s.split) { continue; }
        const std::size_t shared = mma_shared_bytes(s, slot_bytes);
        if (shared > static_cast<std::size_t>(dev.limit)) { continue; }
        const int per_cta = s.warps / s.split;
        const int ctas    = (tiles + per_cta - 1) / per_cta;
        if (ctas <= resident(kernel, 32 * s.warps, shared) * dev.sms) { return s; }
    }
    return kManyWaves;
}

// One launch of an mma kernel (single-format or mixed; `nt` its column tiles) over group g: its CTA prefix,
// shape and shared memory. Every kernel takes (A8Group, slices, encoded, stride, columns, split, stages,
// slot_bytes).
template <class Kernel>
void launch_mma(Kernel kernel, A8Group g, const Columns& in, int slot_bytes, int nt, cudaStream_t stream) {
    int tiles = 0;
    for (int i = 0; i < g.count; ++i) { tiles += (g.rows[i] + 15) / 16; }
    const Device dev = device();
    allow_shared(kernel, dev.limit);
    const int units   = (in.k / 32 + kUnitSlices - 1) / kUnitSlices;
    const Shape shape = choose_shape(kernel, tiles, units, slot_bytes, dev);
    require(shape.stages * slot_bytes >= nt * kPartialBytesPerTile, "a warp's ring holds its partial sums");
    const int per_cta = shape.warps / shape.split;
    g.first[0]        = 0;
    for (int i = 0; i < g.count; ++i) { g.first[i + 1] = g.first[i] + ((g.rows[i] + 15) / 16 + per_cta - 1) / per_cta; }
    CUDA_CHECK(pdl::launch_consumer({dim3(g.first[g.count]), dim3(32 * shape.warps), mma_shared_bytes(shape, slot_bytes), stream},
                                    kernel, g, in.k / 32, in.encoded, in.stride, in.columns, shape.split, shape.stages,
                                    slot_bytes));
    check_launch("mma");
}

// The formats a mixed (two-format) launch may pair: the GGUF imports' fused dense projections
// (ggml_linear.cu's mixed_format).
[[nodiscard]] inline bool mixed_format(QType f) {
    return f == QType::GGML_Q4_K || f == QType::GGML_Q5_K || f == QType::GGML_Q6_K || f == QType::GGML_IQ4_XS;
}

// The mma launches of each layout (ggml_linear_a8_rows.cu, ggml_linear_a8_tiled.cu): one format's group
// (FP32 or BF16 outputs), or a group of two formats f0 < f1 (BF16 outputs; false: not a registered pair).
void launch_mma_rows(QType format, const A8Group& g, const Columns& in, bool out_f32, cudaStream_t stream);
bool launch_mma_rows_mixed(QType f0, QType f1, const A8Group& g, const Columns& in, cudaStream_t stream);
void launch_mma_tiled(QType format, const A8Group& g, const Columns& in, bool out_f32, cudaStream_t stream);
bool launch_mma_tiled_mixed(QType f0, QType f1, const A8Group& g, const Columns& in, cudaStream_t stream);

} // namespace infernix::ops::a8_linear

#define INFERNIX_GGML_A8_FORMATS(X)                                                                                    \
    X(QType::GGML_Q2_0)                                                                                                \
    X(QType::GGML_Q4_K)                                                                                                \
    X(QType::GGML_Q5_K)                                                                                                \
    X(QType::GGML_Q6_K)                                                                                                \
    X(QType::GGML_IQ2_S)                                                                                               \
    X(QType::GGML_IQ3_XXS)                                                                                             \
    X(QType::GGML_IQ3_S)                                                                                               \
    X(QType::GGML_IQ4_XS)                                                                                              \
    X(QType::GGML_IQ4_NL)
