#pragma once

// The tensor-core narrow route of the exact ggml arithmetic (ops/ggml/ggml_linear.cu's
// mma_rows_kernel, dense calls of 3-8 columns): a CTA owns 16 rows and up to eight encoded columns,
// its warps taking K's blocks in turn. (The expert banks' narrow kernels stay on dp4a: this route,
// 8 intermediates or 16 down rows per CTA of five warps, measured 45-65 % slower at 1 and 5 tokens
// and tg512 112.3 against 121.8 tok/s, 2026-10-12; an expert gets 1-2 columns per call.)
//
// Lane (g, t) decodes chunk t (elements 8t..8t+7, fast::chunk) of each slice of rows g and g + 8:
// they are its k slots 4t.. and 16 + 4t.. of an m16n8k32 MMA, so column g's limb words 2t and 2t + 1
// are its B fragment (one 8-byte load per limb plane) and no code word moves between lanes. Each
// limb plane is its own MMA, so a lane's three products of an output (row g or g + 8, column 2t or
// 2t + 1) are in the same register and combine as in add_slice. Formats whose sub-scale covers 16
// elements run one MMA per half, the other half's lanes giving zero codes, and take the other half's
// sub-scale from lane t ^ 2. Per block an int32 sum per limb plane (the tile's bound: |sum| < 2^29),
// then mant(d) * P into an int128: the outputs are the same integers as add_slice's.
//
// Accumulator q of a lane is row g + 8 * (q >> 1), column 2t + (q & 1).

#include "ops/ggml/ggml_dp4a.cuh"
#include "ops/ggml/ggml_tile.cuh"

namespace infernix::ggml::mma_rows {

// Adds blocks first, first + step, ... (< blocks) of rows `wg` (g) and `wg8` (g + 8) against the
// encoded column words `bcol` (column g's first slice, offset by 2t words; slice s at s *
// kSliceWords) into acc. `xs0` and `xs1` are the slice sums of columns 2t and 2t + 1 (read only by
// formats with a min term).
template <QType F>
__device__ __forceinline__ void accumulate(I128 (&acc)[4], const std::uint8_t* wg, const std::uint8_t* wg8,
                                           const std::int32_t* bcol, const std::int32_t* xs0, const std::int32_t* xs1,
                                           int t, int first, int blocks, int step) {
    namespace fast             = ggml::fast;
    constexpr int kBlockSlices = tile::kBlockSlices<F>;
    for (int blk = first; blk < blocks; blk += step) {
        int sum[3][4];
        long long mt[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            sum[0][q] = sum[1][q] = sum[2][q] = 0;
            mt[q]                             = 0;
        }
        std::uint16_t d[2] = {0, 0}, dmin[2] = {0, 0};
#pragma unroll
        for (int j = 0; j < kBlockSlices; ++j) {
            const int s            = blk * kBlockSlices + j;
            const fast::Chunk cg   = fast::chunk<F>(wg, s, t);
            const fast::Chunk ch   = fast::chunk<F>(wg8, s, t);
            const std::int32_t* bs = bcol + static_cast<std::size_t>(s) * fast::kSliceWords;
            int2 b[3];
#pragma unroll
            for (int l = 0; l < 3; ++l) { b[l] = __ldg(reinterpret_cast<const int2*>(bs + 8 * l)); }
            if constexpr (tile::kHalfSub<F>) {
                // Row g's and g + 8's sub-scales of the other half (lane t ^ 2's chunks).
                const int packed = (cg.sub & 0xFFFF) | (ch.sub << 16);
                const int other  = __shfl_xor_sync(0xFFFFFFFFU, packed, 2);
                const int og = static_cast<short>(other & 0xFFFF), oh = other >> 16;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const bool mine = (t >> 1) == h;
                    int p[3][4];
#pragma unroll
                    for (int l = 0; l < 3; ++l) {
                        tile::mma_k32(p[l], mine ? cg.w0 : 0, mine ? ch.w0 : 0, mine ? cg.w1 : 0, mine ? ch.w1 : 0, b[l].x,
                                      b[l].y);
                    }
                    const int sg = mine ? cg.sub : og, sh = mine ? ch.sub : oh;
#pragma unroll
                    for (int l = 0; l < 3; ++l) {
                        sum[l][0] += sg * p[l][0];
                        sum[l][1] += sg * p[l][1];
                        sum[l][2] += sh * p[l][2];
                        sum[l][3] += sh * p[l][3];
                    }
                }
            } else {
                int p[3][4];
#pragma unroll
                for (int l = 0; l < 3; ++l) { tile::mma_k32(p[l], cg.w0, ch.w0, cg.w1, ch.w1, b[l].x, b[l].y); }
#pragma unroll
                for (int l = 0; l < 3; ++l) {
                    if constexpr (tile::kUnitSub<F>) {
#pragma unroll
                        for (int q = 0; q < 4; ++q) { sum[l][q] += p[l][q]; }
                    } else {
                        sum[l][0] += cg.sub * p[l][0];
                        sum[l][1] += cg.sub * p[l][1];
                        sum[l][2] += ch.sub * p[l][2];
                        sum[l][3] += ch.sub * p[l][3];
                    }
                }
            }
            if constexpr (fast::kMin<F>) {
                const int xa = __ldg(xs0 + s), xb = __ldg(xs1 + s);
                mt[0] += static_cast<long long>(cg.min) * xa;
                mt[1] += static_cast<long long>(cg.min) * xb;
                mt[2] += static_cast<long long>(ch.min) * xa;
                mt[3] += static_cast<long long>(ch.min) * xb;
            }
            if (j == kBlockSlices - 1) {
                d[0]    = cg.d;
                d[1]    = ch.d;
                dmin[0] = cg.dmin;
                dmin[1] = ch.dmin;
            }
        }
        // The block ends: mant(d) * P (and the min term) into the int128 outputs.
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            std::int64_t mant, min_mant = 0;
            int shift, min_shift = 0;
            bool negative;
            f16_parts(d[h], mant, shift, negative);
            if (negative) { mant = -mant; }
            if constexpr (fast::kMin<F>) {
                f16_parts(dmin[h], min_mant, min_shift, negative);
                min_mant = negative ? min_mant : -min_mant; // the min term is subtracted
            }
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int q = 2 * h + c;
                const long long p = (static_cast<long long>(sum[0][q]) << 16) + (static_cast<long long>(sum[1][q]) << 8) +
                                    sum[2][q];
                add_shifted(acc[q], mant * p, shift);
                if constexpr (fast::kMin<F>) { add_shifted(acc[q], min_mant * mt[q], min_shift); }
            }
        }
    }
}

// ------------------------------------------------------------------------------ staged weights
//
// accumulate with the weights staged (single-format launches; mma_rows_kernel kStaged, end to end
// 2.1 % faster than accumulate's 2-byte loads, 2026-10-16): the warp copies its next 256-element unit of the CTA's 16 rows
// (every row from its 16-byte aligned start) into its half of a double buffer by cp.async while it
// decodes the current unit from shared memory (tile_chunk: the same integers as fast::chunk), so a
// unit's loads are 16-byte copies in flight beside the previous unit's MMAs instead of 2-byte loads
// in its own dependency chain (the 5-column Q6_K call measured 31.9 us, 0.67 TB/s, stalled on long
// scoreboard and the LSU queue, 2026-10-14).
constexpr int kUnitElems = 256;
template <QType F>
inline constexpr int kUnitBytes = kUnitElems / static_cast<int>(ggml_block_elems(F)) * static_cast<int>(ggml_block_bytes(F));
template <QType F>
inline constexpr int kUnitChunks = (15 + kUnitBytes<F> + 15) / 16; // 16-byte copies of a row's unit
template <QType F>
struct StagedRows {
    alignas(16) std::uint8_t raw[2][16][kUnitChunks<F> * 16];
};

// Issues the copies of unit u (its first `bytes` bytes) of rows wg (g = 0..15 by `row`) into buf.
template <QType F, class Row>
__device__ __forceinline__ void issue_unit_rows(StagedRows<F>& st, int buf, int u, int bytes, Row row, int lane) {
    for (int i = lane; i < 16 * kUnitChunks<F>; i += 32) {
        const int r = i / kUnitChunks<F>, c = i % kUnitChunks<F>;
        const std::uint8_t* src = row(r) + static_cast<std::size_t>(u) * kUnitBytes<F>;
        const auto* aligned     = reinterpret_cast<const std::uint8_t*>(reinterpret_cast<std::uintptr_t>(src) & ~std::uintptr_t{15});
        const int need          = static_cast<int>(src - aligned) + bytes - 16 * c;
        if (need > 0) { tile::cp_async16(&st.raw[buf][r][16 * c], aligned + 16 * c, need < 16 ? need : 16); }
    }
}

// The bytes of unit u (of K = slices * 32 elements) of a row.
template <QType F>
__device__ __forceinline__ int unit_row_bytes(int u, int slices) {
    constexpr int kBlockElems = static_cast<int>(ggml_block_elems(F));
    constexpr int kBlockBytes = static_cast<int>(ggml_block_bytes(F));
    return (min(kUnitElems, slices * 32 - u * kUnitElems) / kBlockElems) * kBlockBytes;
}

// Issues (and commits) the warp's first unit, `first`, into buffer 0: a programmatic dependent may
// call it before its wait (the weights are immutable), then accumulate_staged with `issued`.
template <QType F, class Row>
__device__ __forceinline__ void issue_first_unit(StagedRows<F>& st, Row row, int lane, int first, int slices) {
    const int units = (slices * 32 + kUnitElems - 1) / kUnitElems;
    if (first < units) { issue_unit_rows<F>(st, 0, first, unit_row_bytes<F>(first, slices), row, lane); }
    tile::cp_async_commit();
}

// The sums of unit u (its first `unit_slices` slices) of rows g and g + 8, whose bytes rg and rg8
// hold in shared memory (any alignment): accumulate's arithmetic, block by block.
template <QType F>
__device__ __forceinline__ void unit_sums(I128 (&acc)[4], const std::uint8_t* rg, const std::uint8_t* rg8, int u,
                                          int unit_slices, const std::int32_t* bcol, const std::int32_t* xs0,
                                          const std::int32_t* xs1, int t) {
    namespace fast             = ggml::fast;
    constexpr int kBlockElems  = static_cast<int>(ggml_block_elems(F));
    constexpr int kBlockBytes  = static_cast<int>(ggml_block_bytes(F));
    constexpr int kBlockSlices = tile::kBlockSlices<F>;
        for (int b0 = 0; b0 < unit_slices; b0 += kBlockSlices) {
            int sum[3][4];
            long long mt[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                sum[0][q] = sum[1][q] = sum[2][q] = 0;
                mt[q]                             = 0;
            }
            std::uint16_t d[2] = {0, 0}, dmin[2] = {0, 0};
#pragma unroll
            for (int j = 0; j < kBlockSlices; ++j) {
                const int e  = (b0 + j) * 32 + 8 * t; // element of the unit
                const int s  = u * (kUnitElems / 32) + b0 + j;
                const tile::TileChunk cg = tile::tile_chunk<F>(rg + e / kBlockElems * kBlockBytes, e % kBlockElems / 8);
                const tile::TileChunk ch = tile::tile_chunk<F>(rg8 + e / kBlockElems * kBlockBytes, e % kBlockElems / 8);
                const std::int32_t* bs = bcol + static_cast<std::size_t>(s) * fast::kSliceWords;
                int2 b[3];
#pragma unroll
                for (int l = 0; l < 3; ++l) { b[l] = __ldg(reinterpret_cast<const int2*>(bs + 8 * l)); }
                if constexpr (tile::kHalfSub<F>) {
                    const int packed = (cg.sub & 0xFFFF) | (ch.sub << 16);
                    const int other  = __shfl_xor_sync(0xFFFFFFFFU, packed, 2);
                    const int og = static_cast<short>(other & 0xFFFF), oh = other >> 16;
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const bool mine = (t >> 1) == h;
                        int p[3][4];
#pragma unroll
                        for (int l = 0; l < 3; ++l) {
                            tile::mma_k32(p[l], mine ? cg.w[0] : 0, mine ? ch.w[0] : 0, mine ? cg.w[1] : 0, mine ? ch.w[1] : 0,
                                          b[l].x, b[l].y);
                        }
                        const int sg = mine ? cg.sub : og, sh = mine ? ch.sub : oh;
#pragma unroll
                        for (int l = 0; l < 3; ++l) {
                            sum[l][0] += sg * p[l][0];
                            sum[l][1] += sg * p[l][1];
                            sum[l][2] += sh * p[l][2];
                            sum[l][3] += sh * p[l][3];
                        }
                    }
                } else {
                    int p[3][4];
#pragma unroll
                    for (int l = 0; l < 3; ++l) { tile::mma_k32(p[l], cg.w[0], ch.w[0], cg.w[1], ch.w[1], b[l].x, b[l].y); }
#pragma unroll
                    for (int l = 0; l < 3; ++l) {
                        if constexpr (tile::kUnitSub<F>) {
#pragma unroll
                            for (int q = 0; q < 4; ++q) { sum[l][q] += p[l][q]; }
                        } else {
                            sum[l][0] += cg.sub * p[l][0];
                            sum[l][1] += cg.sub * p[l][1];
                            sum[l][2] += ch.sub * p[l][2];
                            sum[l][3] += ch.sub * p[l][3];
                        }
                    }
                }
                if constexpr (fast::kMin<F>) {
                    const int xa = __ldg(xs0 + s), xb = __ldg(xs1 + s);
                    mt[0] += static_cast<long long>(cg.min_sub) * xa;
                    mt[1] += static_cast<long long>(cg.min_sub) * xb;
                    mt[2] += static_cast<long long>(ch.min_sub) * xa;
                    mt[3] += static_cast<long long>(ch.min_sub) * xb;
                }
                if (j == kBlockSlices - 1) {
                    d[0]    = cg.d;
                    d[1]    = ch.d;
                    dmin[0] = cg.dmin;
                    dmin[1] = ch.dmin;
                }
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                std::int64_t mant, min_mant = 0;
                int shift, min_shift = 0;
                bool negative;
                f16_parts(d[h], mant, shift, negative);
                if (negative) { mant = -mant; }
                if constexpr (fast::kMin<F>) {
                    f16_parts(dmin[h], min_mant, min_shift, negative);
                    min_mant = negative ? min_mant : -min_mant; // the min term is subtracted
                }
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int q = 2 * h + c;
                    const long long p = (static_cast<long long>(sum[0][q]) << 16) +
                                        (static_cast<long long>(sum[1][q]) << 8) + sum[2][q];
                    add_shifted(acc[q], mant * p, shift);
                    if constexpr (fast::kMin<F>) { add_shifted(acc[q], min_mant * mt[q], min_shift); }
                }
            }
        }
}

// accumulate over units first, first + step, ... (< units) of the K = slices * 32 elements, the
// rows' bytes staged through `st` (this warp's); `issued`: issue_first_unit already ran. Formats as
// accumulate.
template <QType F, class Row>
__device__ __forceinline__ void accumulate_staged(I128 (&acc)[4], StagedRows<F>& st, Row row, const std::int32_t* bcol,
                                                  const std::int32_t* xs0, const std::int32_t* xs1, int t, int g,
                                                  int lane, int first, int slices, int step, bool issued = false) {
    constexpr int kBlockElems = static_cast<int>(ggml_block_elems(F));
    constexpr int kBlockBytes = static_cast<int>(ggml_block_bytes(F));
    const int units = (slices * 32 + kUnitElems - 1) / kUnitElems;
    const auto unit_bytes = [&](int u) { return unit_row_bytes<F>(u, slices); };
    const auto offset = [&](int r, int u) {
        return static_cast<int>(reinterpret_cast<std::uintptr_t>(row(r) + static_cast<std::size_t>(u) * kUnitBytes<F>) & 15U);
    };
    if (!issued) { issue_first_unit<F>(st, row, lane, first, slices); }
    int buf = 0;
    for (int u = first; u < units; u += step, buf ^= 1) {
        // Ahead: the next unit into the other half (its last reader, the previous unit, is done: the
        // warp's lanes reached here together after its decode).
        __syncwarp();
        if (u + step < units) { issue_unit_rows<F>(st, buf ^ 1, u + step, unit_bytes(u + step), row, lane); }
        tile::cp_async_commit();
        tile::cp_async_wait<1>();
        __syncwarp();
        unit_sums<F>(acc, &st.raw[buf][g][offset(g, u)], &st.raw[buf][g + 8][offset(g + 8, u)], u,
                     unit_bytes(u) / kBlockBytes * kBlockElems / 32, bcol, xs0, xs1, t);
    }
    tile::cp_async_wait<0>();
}

// The warps' accumulators summed (exact, any order) into warp 0's through `partial` (shared memory
// of Warps - 1 entries). Every thread of the CTA calls it; warp 0 then holds the sums.
template <int Warps>
__device__ __forceinline__ void reduce(I128 (&acc)[4], unsigned long long (*partial)[4][2][32], int warp, int lane) {
    if constexpr (Warps > 1) {
        if (warp > 0) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                partial[warp - 1][q][0][lane] = acc[q].lo;
                partial[warp - 1][q][1][lane] = static_cast<unsigned long long>(acc[q].hi);
            }
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int w = 0; w < Warps - 1; ++w) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    add(acc[q], I128{partial[w][q][0][lane], static_cast<std::int64_t>(partial[w][q][1][lane])});
                }
            }
        }
    }
}

} // namespace infernix::ggml::mma_rows
