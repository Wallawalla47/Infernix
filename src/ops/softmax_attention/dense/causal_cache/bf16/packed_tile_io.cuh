#pragma once

// Packed BF16 KV (KvCacheStorage::Bf16Packed, ops/kv_cache/bf16p_codec.cuh) for the BF16 attention kernels. The
// grouped kernels' packed instances decode it in producer warps beside their MMA warps (bf16p_kv_produce_tiles); the
// prompt route unpacks a segment of keys at a time into a scratch of the bf16 storage's rows for the bf16 tiled kernel
// (bf16p_kv_unpack_kernel). Either way attention reads what the bf16 storage holds (K as BF16, V as FP16), so its
// arithmetic and bits are the bf16 storage's.

#include "ops/common/mbarrier.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/bf16p_codec.cuh"
#include "ops/softmax_attention/common/causal_tile_io.cuh"

#include <cstdint>

namespace infernix::ops::detail {

// The grouped kernels' packed instances: producer warps beside the MMA warps (eight warps in all, the most one CTA's
// 255-register warps fit: four producers beside up to four MMA warps, two beside six), and the ring of tile buffers
// between them.
inline constexpr int kBf16pTileRing = 3;
template <class S>
inline constexpr int kBf16pProducerWarps = S::kWarps <= 4 ? 4 : 2;

// The ring's handshakes and side records (the tile buffers are the kernel's shared storage): full[s], the P producers
// have decoded slot s's tile; empty[s], the MMA warps are done with it; side[s][role][row], the side records of the
// slot's rows. The MMA warps' queries first occupy slots 0 .. Q - 1, and their release of them is those slots' empty
// phase 0; tile t takes slot (t + Q) % kBf16pTileRing, so the producers stage the first tiles from the kernel's start.
template <int N, int P, int Q>
struct Bf16pTileRing {
    static_assert(Q >= 1 && Q < kBf16pTileRing);
    std::uint64_t full[kBf16pTileRing];
    std::uint64_t empty[kBf16pTileRing];
    alignas(16) uint2 side[kBf16pTileRing][2][N];

    // Tile t's slot, its full barrier's phase parity (each slot takes every third tile) and the parity its stage
    // waits for on the empty barrier: the slot's previous use released (none for a slot's first use).
    __device__ __forceinline__ static int slot(int t) { return (t + Q) % kBf16pTileRing; }
    __device__ __forceinline__ static unsigned full_parity(int t) { return (t / kBf16pTileRing) & 1U; }
    __device__ __forceinline__ static unsigned empty_parity(int t) { return (((t + Q) / kBf16pTileRing) & 1U) ^ 1U; }

    // One thread, before the CTA's first barrier.
    __device__ __forceinline__ void init(int consumer_warps) {
        for (int slot = 0; slot < kBf16pTileRing; ++slot) {
            cta_mbarrier_init(full + slot, P);
            cta_mbarrier_init(empty + slot, consumer_warps);
        }
        cta_mbarrier_fence_init();
    }
};

// One of the P producer warps of a packed grouped kernel. Of each tile (keys k0 .. k0 + N - 1 of one page and KV
// head) it owns K and V rows first .. first + N / P - 1: it copies their 384 bytes (one cp.async instruction a row,
// lanes 0-23) and their side records (16 bytes a lane) into the tile's ring slot, whose rows have the bf16 storage's
// 512 bytes, two tiles ahead of its decode. Its decode (after its own copies' cp.async group) rewrites each row in
// place as the bf16 storage holds it (K as BF16, V as FP16), at the swizzled places the MMA warps read (a row at or
// past load_end as zero), and arrives on the slot's full barrier once its rows are written. (cp.async, not
// cp.async.bulk: a bulk copy takes its addresses in uniform registers, so a warp's per-row copies issue one lane at a
// time, each waiting on the last.)
template <class G, int N, int P, int Q>
__device__ __forceinline__ void bf16p_kv_produce_tiles(std::uint16_t* tiles, Bf16pTileRing<N, P, Q>& ring,
                                                       const std::uint8_t* keys, const std::uint8_t* values,
                                                       const std::uint8_t* key_side, const std::uint8_t* value_side,
                                                       const int* table, int head, int start, int end, int load_end,
                                                       int producer, int lane) {
    constexpr int Rows = N / P, SideParts = Rows * kBf16pSideBytes / 16; // 16-byte parts of a role's side records
    constexpr int TileRow = 2 * G::kHeadDim;                              // a decoded row's bytes in the slot
    constexpr unsigned kFull = 0xFFFFFFFFU;
    static_assert(N % P == 0 && Rows % 2 == 0 && 2 * SideParts <= 32 && kBf16pRowUnits == 24);
    static_assert(G::kHeadDim == kBf16pRowValues && kPagedKVPageSize % N == 0);
    const int first = producer * Rows, count = div_up(end - start, N);
    // The CTA's pages, 32 a load: page i of its range is lane i % 32's after the load of block i / 32.
    const int first_page = start >> kPagedKVPageShift, last_page = (end - 1) >> kPagedKVPageShift;
    int block = -1, lane_page = 0;
    const auto page_of = [&](int k0) {
        const int i = (k0 >> kPagedKVPageShift) - first_page;
        if ((i >> 5) != block) {
            block        = i >> 5;
            const int at = first_page + 32 * block + lane;
            lane_page    = at <= last_page ? table[at] : 0;
        }
        return __shfl_sync(kFull, lane_page, i & 31);
    };
    using Ring = Bf16pTileRing<N, P, Q>;
    const auto tile_of = [&](int t) {
        return reinterpret_cast<std::uint8_t*>(tiles + Ring::slot(t) * 2 * N * G::kHeadDim);
    };
    // Tile t's copies into its slot (one cp.async group, committed by every lane), once the MMA warps are done with
    // the slot's previous use.
    const auto stage = [&](int t) {
        const int slot = Ring::slot(t), k0 = start + t * N, live = min(Rows, load_end - (k0 + first));
        const int page = page_of(k0);
        cta_mbarrier_wait(&ring.empty[slot], Ring::empty_parity(t));
        if (live > 0) {
            const std::int64_t row0 =
                paged_kv_element_offset<1, G::KVHeads>(page, head, (k0 & kPagedKVPageMask) + first, 0);
            std::uint8_t* tile = tile_of(t);
            if (lane < kBf16pRowUnits) {
#pragma unroll
                for (int row = 0; row < Rows; ++row) {
                    if (row < live) {
                        const std::int64_t at = (row0 + row) * kBf16pRowBytes + 16 * lane;
                        cp_async<16, Cache::cg>(tile + (first + row) * TileRow + 16 * lane, keys + at);
                        cp_async<16, Cache::cg>(tile + (N + first + row) * TileRow + 16 * lane, values + at);
                    }
                }
            }
            // The side records of all the warp's rows (live or not: the same page), two records a lane: lanes
            // 0 .. SideParts - 1 the K ones, then the V ones.
            if (lane < 2 * SideParts) {
                const int role = lane / SideParts, part = lane % SideParts;
                cp_async<16, Cache::cg>(&ring.side[slot][role][first + 2 * part],
                                        (role ? value_side : key_side) + row0 * kBf16pSideBytes + 16 * part);
            }
        }
        cp_commit();
    };
    // Tile t's rows rewritten in place, two rows of K and V at a time: every lane reads its bytes of all four, then
    // (__syncwarp) writes its decoded chunks, whose swizzled places overlap other lanes' packed bytes. (Each row patches
    // its own escapes inline: one loop over the rows that need it, with a full BF16 decode and conversion of a V row,
    // lost 0.3-1.1 % end to end on real KV, where about one row in ten has escapes.)
    const auto decode = [&](int t) {
        const int slot = Ring::slot(t), k0 = start + t * N;
        std::uint8_t* tile = tile_of(t);
        // This warp's copies of tile t have landed (only tile t + 1's may still be in flight); __syncwarp shows each
        // lane's copies to the others.
        if (t + 1 < count) {
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncwarp();
#pragma unroll 1
        for (int row = first; row < first + Rows; row += 2) {
            uint2 side[2][2], code[2][2];
            unsigned nib[2][2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int role = 0; role < 2; ++role) {
                    const std::uint8_t* at = tile + (role * N + row + i) * TileRow;
                    side[i][role]          = ring.side[slot][role][row + i];
                    code[i][role]          = *reinterpret_cast<const uint2*>(at + 8 * lane);
                    nib[i][role]           = *reinterpret_cast<const unsigned*>(at + kBf16pNibbles + 4 * lane);
                }
            }
            uint4 out[2][2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int role = 0; role < 2; ++role) {
                    const int r = row + i;
                    out[i][role] = make_uint4(0, 0, 0, 0);
                    if (k0 + r >= load_end) { continue; }
                    out[i][role] = role ? bf16p_decode_regs_f16(code[i][role], nib[i][role], side[i][role], lane)
                                        : bf16p_decode_regs(code[i][role], nib[i][role], side[i][role], lane);
                }
            }
            __syncwarp();
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int role = 0; role < 2; ++role) {
                    const int r = row + i;
                    *reinterpret_cast<uint4*>(tile + (role * N + r) * TileRow + 16 * (lane ^ (r & 7))) = out[i][role];
                }
            }
        }
        __syncwarp();
        if (lane == 0) cta_mbarrier_arrive(&ring.full[slot]);
    };
    stage(0);
    if (count > 1) stage(1);
    for (int t = 0; t < count; ++t) {
        decode(t);
        if (t + 2 < count) stage(t + 2);
    }
}

// Unpacks keys [first, first + count) of one table row of a packed cache (first a multiple of 64) into a scratch of
// the bf16 storage's rows, page-major like the cache (logical page L at scratch page L - first / 64; K as BF16, V as
// FP16), and points `table`'s entries of those pages at them. Keys past the call's last live position (positions of
// one row, `valid_columns` live of `width`) are not read. One warp a (key, KV head, K or V) row.
template <int KVHeads>
__global__ __launch_bounds__(256) void bf16p_kv_unpack_kernel(
    const std::uint8_t* __restrict__ keys, const std::uint8_t* __restrict__ values,
    const std::uint8_t* __restrict__ key_side, const std::uint8_t* __restrict__ value_side,
    const std::int32_t* __restrict__ tables, const std::int32_t* __restrict__ table_rows, int table_stride,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ valid_columns, int width, int first,
    int count, std::uint8_t* __restrict__ scratch_keys, std::uint8_t* __restrict__ scratch_values,
    std::int32_t* __restrict__ table) {
    const int unit = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = static_cast<int>(threadIdx.x & 31);
    if (unit >= count * KVHeads * 2) { return; }
    const int role = unit & 1, head = (unit >> 1) % KVHeads, key = first + (unit >> 1) / KVHeads;
    const int live = valid_columns != nullptr ? valid_columns[0] : width;
    if (live <= 0 || key > positions[live - 1]) { return; }
    const std::int32_t* block_table =
        tables + (table_rows != nullptr ? static_cast<std::int64_t>(table_rows[0]) * table_stride : 0);
    const int logical = key >> kPagedKVPageShift, offset = key & kPagedKVPageMask;
    const std::int64_t element = paged_kv_element_offset<1, KVHeads>(block_table[logical], head, offset, 0);
    const uint2 side =
        __ldg(reinterpret_cast<const uint2*>((role ? value_side : key_side) + element * kBf16pSideBytes));
    const std::uint8_t* row = (role ? values : keys) + element * kBf16pRowBytes;
    const uint2 code        = __ldg(reinterpret_cast<const uint2*>(row + 8 * lane));
    const unsigned nib      = __ldg(reinterpret_cast<const unsigned*>(row + kBf16pNibbles + 4 * lane));
    const uint4 out = role ? bf16p_decode_regs_f16(code, nib, side, lane) : bf16p_decode_regs(code, nib, side, lane);
    const int scratch_page = logical - (first >> kPagedKVPageShift);
    *reinterpret_cast<uint4*>((role ? scratch_values : scratch_keys) +
                              paged_kv_element_offset<2 * kBf16pRowValues, KVHeads>(scratch_page, head, offset, 0) +
                              16 * lane) = out;
    if (role == 0 && head == 0 && offset == 0 && lane == 0) { table[logical] = scratch_page; }
}

} // namespace infernix::ops::detail
