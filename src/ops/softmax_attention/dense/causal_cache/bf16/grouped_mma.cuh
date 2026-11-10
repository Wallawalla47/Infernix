#pragma once

#include "core/pdl.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/epilogue.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/packed_tile_io.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/softmax.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/split_policy.h"
#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/common/causal_tree.cuh"

namespace infernix::ops::detail {

template <class G, class S, class Input>
__device__ __forceinline__ void
bf16_kv_load_grouped_tile(__nv_bfloat16* key_tile, __half* value_tile, const __nv_bfloat16* keys,
                          const __half* values, Input input, int page, int head, int begin, int end,
                          int first_position, int tid) {
    constexpr int D       = G::kHeadDim;
    const auto cache_base = bf16_kv_cache_index<G>(page, head, 0, begin & kPagedKVPageMask);
    // Small query tiles need enough outstanding loads to hide memory latency.
#pragma unroll 1
    for (int i = tid; i < S::kKeyRows * (D / 8); i += S::kThreads) {
        const int row = i / (D / 8), d = i % (D / 8) * 8, key = begin + row;
        const auto offset = row * D + causal_swizzle(row, d);
        auto* kd          = key_tile + offset;
        auto* vd          = value_tile + offset;
        if (key >= end) {
            store_vec(kd, make_int4(0, 0, 0, 0));
            store_vec(vd, make_int4(0, 0, 0, 0));
        } else {
            if constexpr (Input::writes_cache) {
                if (key >= first_position) {
                    const auto index = causal_new_index<G>(head, d, key - first_position);
                    cp_async<16>(kd, input.k + index);
                    store_vec(vd, bf16x8_bits_to_f16x8_bits(load_vec<int4>(input.v + index)));
                    continue;
                }
            }
            const auto index = cache_base + row * D + d;
            cp_async<16>(kd, keys + index);
            cp_async<16>(vd, values + index);
        }
    }
}

// Packed-query tiles and KV partitions are independent grid axes. Each KV row
// has one append owner; all query CTAs read new rows from the immutable inputs. Tree instances
// also apply the per-row ancestor masks of speculative verification trees. Packed instances read the
// packed BF16 storage (packed_tile_io.cuh: the same tiles, so the same bits); its rows are appended
// before the call (no fused append). Their CTAs add kBf16pProducerWarps<S> producer warps after the MMA
// warps: the producers copy the packed rows into a ring of tile buffers and decode them there in place
// ahead of the MMA warps, so the decode overlaps the stream and the MMAs (one CTA an SM). Packed calls
// of 65-96 query rows take a 96-row tile (six MMA warps, two producers), so one CTA decodes each tile.
template <class G, class S, bool MultiBatch, bool Masked, class Input, bool Tree = false, bool Packed = false>
__launch_bounds__(Packed ? S::kThreads + 32 * kBf16pProducerWarps<S> : S::kLaunchBoundThreads,
                  Packed ? 1 : S::kMinBlocks) __global__
    void bf16_kv_grouped_mma_kernel(const __nv_bfloat16* q, Input input, const int* positions,
                                    typename Bf16KvCacheView<Input::writes_cache>::Key* cache_k,
                                    typename Bf16KvCacheView<Input::writes_cache>::Value* cache_v,
                                    const int* tables, const int* validity, const int* table_rows,
                                    int table_stride, int runtime_width, int visible_capacity,
                                    float scale, Bf16KvPartition partition,
                                    CausalPartialView partial, const std::uint32_t* tree_masks,
                                    const std::uint8_t* key_side, const std::uint8_t* value_side) {
    // Every input but the scale and the geometry is a kernel's output (q, the appended K/V, the cache and its
    // tables): wait before any of them (a no-op unless the launch is a programmatic dependent, candidates §58).
    pdl::enter_streaming();
    static_assert(!(Packed && Input::writes_cache), "the packed storage appends before its attention");
    const int width = S::kFixedWidth ? S::kFixedWidth : runtime_width;
    constexpr int D = G::kHeadDim, M = S::kQueryRows, N = S::kKeyRows;
    constexpr int NK = N / S::kWarpsKV, QKNt = NK / 8, QKKs = D / 16;
    constexpr int PVNt = D / 8, PVKs = NK / 16;
    using Storage = Bf16KvGroupedStorage<G, S, Packed ? kBf16pTileRing : 1>;
    static_assert(sizeof(Storage) <= 99 * 1024);
    // The queries stage in the K/V buffer; a packed instance's may span ring slots (QSlots of them, M <= 96).
    constexpr int QSlots = (M + 2 * N - 1) / (2 * N);
    static_assert(Packed ? QSlots < kBf16pTileRing : M <= 2 * N);
    Storage* storage;
    if constexpr (sizeof(Storage) <= 48 * 1024) {
        __shared__ Storage fixed;
        storage = &fixed;
    } else {
        extern __shared__ __align__(16) unsigned char dynamic[];
        storage = reinterpret_cast<Storage*>(dynamic);
    }
    auto* k_s     = reinterpret_cast<__nv_bfloat16*>(storage->qkv);
    auto* v_s     = reinterpret_cast<__half*>(storage->qkv + N * D);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_q = warp % S::kWarpsQ, warp_kv = warp / S::kWarpsQ;
    const int head  = S::kFixedWidth ? blockIdx.x : blockIdx.x % G::KVHeads;
    const int tile  = S::kFixedWidth ? 0 : blockIdx.x / G::KVHeads;
    // The row and split this CTA serves: grid z and y, or, on a proportional launch, the slot the rows' windows give
    // it (bf16_kv_proportional_slot); `base` is the row's first partial slot.
    CausalRowSplit slot{MultiBatch ? static_cast<int>(blockIdx.z) : 0, static_cast<int>(blockIdx.y), 0,
                        (MultiBatch ? static_cast<int>(blockIdx.z) : 0) * partition.capacity};
    if (MultiBatch && partition.proportional != 0) {
        if (!bf16_kv_proportional_slot(partition, positions, Masked ? validity : nullptr, width,
                                       static_cast<int>(gridDim.z), visible_capacity,
                                       static_cast<int>(blockIdx.z * gridDim.y + blockIdx.y), slot))
            return;
    }
    const int split = slot.split, batch = slot.row;
    const int row_begin = tile * M, packed_rows = width * G::GroupSize;
    const int live = Masked ? validity[batch] : width;
    q += static_cast<std::int64_t>(batch) * width * D * G::QHeads;
    positions += batch * width;
    if constexpr (Input::writes_cache) {
        input.k += static_cast<std::int64_t>(batch) * width * D * G::KVHeads;
        input.v += static_cast<std::int64_t>(batch) * width * D * G::KVHeads;
    }
    {
        partial.acc += static_cast<std::int64_t>(slot.base) * width * G::QHeads * D;
        partial.maximum += static_cast<std::int64_t>(slot.base) * width * G::QHeads;
        partial.sum += static_cast<std::int64_t>(slot.base) * width * G::QHeads;
    }
    const auto neutral = [&]() {
        for (int i = tid; i < M * (D / 2); i += S::kThreads) {
            const int row = row_begin + i / (D / 2), d = i % (D / 2) * 2;
            if (row < packed_rows) {
                const int token = row / G::GroupSize;
                const int h     = head * G::GroupSize + row % G::GroupSize;
                bf16_kv_store_pair<G, true>(partial, nullptr, h, token, d, width, split, 0.0f, 0.0f,
                                            -CUDART_INF_F, 0.0f);
            }
        }
    };
    if (row_begin >= live * G::GroupSize) return;
    // Masked tail columns count toward the partition, but no key past the live ones is loaded.
    const int first = positions[0], live_end = positions[live - 1] + 1;
    const int window = causal_row_window(first, width, visible_capacity);
    // A proportional row runs its count of slots as layout() splits them (a slot past its live splits returns).
    const auto work = MultiBatch && partition.proportional != 0
                          ? Bf16KvProportionalShares::layout(partition, window, slot.count)
                          : partition.live(window);
    if (split >= work.splits) return;
    const int start     = split * work.keys_per_split;
    const int stop      = min(window, start + work.keys_per_split);
    const int table_row = table_rows ? table_rows[batch] : 0;
    const int* table    = tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (Input::writes_cache) {
        if (tile == 0) {
            for (int i = tid; i < live * (D / 8); i += S::kThreads) {
                const int token = i / (D / 8), d = i % (D / 8) * 8, key = positions[token];
                if (key >= start && key < stop) {
                    const auto src = causal_new_index<G>(head, d, token);
                    const auto dst = bf16_kv_cache_index<G>(table[key >> kPagedKVPageShift], head,
                                                            d, key & kPagedKVPageMask);
                    store_vec(cache_k + dst, load_vec<int4>(input.k + src));
                    store_vec(cache_v + dst,
                              bf16x8_bits_to_f16x8_bits(load_vec<int4>(input.v + src)));
                }
            }
        }
    }
    const int last_token = min(width, div_up(row_begin + M, G::GroupSize)) - 1;
    const int end        = min(stop, first + last_token + 1);
    const int load_end   = min(end, live_end);
    if (start >= end) {
        neutral();
        return;
    }
    // The page window: the table's pages from page_window on (a CTA's keys start at a multiple of N, so a tile
    // lies in one page).
    int page_window      = -Storage::kPageWindow;
    const auto load_pages = [&](int logical_page) {
        page_window          = logical_page;
        const int page_count = min(Storage::kPageWindow, div_up(end, kPagedKVPageSize) - page_window);
        for (int i = tid; i < page_count; i += S::kThreads)
            storage->pages[i] = table[page_window + i];
        __syncthreads();
    };
    // Packed: warps from S::kWarps on are producers (bf16p_kv_produce_tiles), from the kernel's start; the ring's full
    // and empty barriers hand its tile buffers between them and the MMA warps, which alone run what follows (their
    // queries pass through the ring's first QSlots slots, which they release on those slots' empty barriers).
    using Ring = Bf16pTileRing<N, Packed ? kBf16pProducerWarps<S> : 1, Packed ? QSlots : 1>;
    [[maybe_unused]] Ring* ring = nullptr;
    if constexpr (Packed) {
        __shared__ Ring shared_ring;
        ring = &shared_ring;
        if (tid == 0) ring->init(S::kWarps);
        __syncthreads();
        if (warp >= S::kWarps) {
            bf16p_kv_produce_tiles<G, N, kBf16pProducerWarps<S>, QSlots>(
                storage->qkv, *ring, reinterpret_cast<const std::uint8_t*>(cache_k),
                reinterpret_cast<const std::uint8_t*>(cache_v), key_side, value_side, table, head, start, end,
                load_end, warp - S::kWarps, lane);
            return;
        }
    }
    // The MMA warps' barrier: the CTA's, or theirs alone when producers run beside them.
    [[maybe_unused]] const auto consumer_sync = [&]() {
        if constexpr (Packed) {
            asm volatile("bar.sync 1, %0;" : : "r"(S::kThreads) : "memory");
        } else {
            __syncthreads();
        }
    };
    for (int i = tid; i < M * (D / 8); i += S::kThreads) {
        const int row = i / (D / 8), d = i % (D / 8) * 8, packed = row_begin + row;
        const int token = packed / G::GroupSize, h = head * G::GroupSize + packed % G::GroupSize;
        auto* dst = k_s + row * D + causal_swizzle(row, d);
        if (token < live)
            cp_async<16>(dst, q + causal_q_index<G>(h, d, token));
        else
            store_vec(dst, make_int4(0, 0, 0, 0));
    }
    cp_commit();
    cp_wait<0>();
    consumer_sync();
    unsigned q_frag[QKKs][4];
    const int a_row = warp_q * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
    for (int k = 0; k < QKKs; ++k) {
        const int d = k * 16 + (lane >> 4) * 8;
        ldmatrix_x4(q_frag[k][0], q_frag[k][1], q_frag[k][2], q_frag[k][3],
                    smem_addr(k_s + a_row * D + causal_swizzle(a_row, d)));
    }
    if constexpr (Packed) {
        // This warp's queries have left the ring's first QSlots slots.
        __syncwarp();
        if (lane == 0) {
#pragma unroll
            for (int slot = 0; slot < QSlots; ++slot) cta_mbarrier_arrive(ring->empty + slot);
        }
    } else {
        __syncthreads();
    }
    float acc[PVNt][4] = {};
    Bf16KvSoftmaxRow state[2];
    const float scale_log2 = scale * kLog2E;
    const int gid = lane >> 2, lid = lane & 3;
    const int rows[2]   = {row_begin + warp_q * 16 + gid, row_begin + warp_q * 16 + gid + 8};
    const int tokens[2] = {rows[0] / G::GroupSize, rows[1] / G::GroupSize};
    const int qabs[2]   = {tokens[0] < live ? positions[tokens[0]] : -1,
                         tokens[1] < live ? positions[tokens[1]] : -1};
    const std::uint32_t tree[2] = {
        tokens[0] < live ? causal_tree_mask<Tree>(tree_masks, batch, width, tokens[0]) : ~0u,
        tokens[1] < live ? causal_tree_mask<Tree>(tree_masks, batch, width, tokens[1]) : ~0u};
    // A 96-row tile runs the rows of two of the plan's 64-row tiles: each warp stops where its 64-row tile's CTA
    // stops (a key tile masked for all of a warp's rows still rescales their state by exp2 of a rounding residual),
    // so the bits are the 64-row tiles'.
    int warp_end = end;
    if constexpr (M > 64) {
        const int plan_begin = (row_begin + warp_q * 16) / 64 * 64;
        warp_end = min(stop, first + min(width, div_up(plan_begin + 64, G::GroupSize)));
    }
    // The K and V fragments' shared addresses. causal_swizzle XORs a row's 16-byte chunk with row & 7, which in every
    // fragment this lane loads is lane & 7: an address is a per-lane register (the XOR's result for the chunk's low
    // three bits) plus a compile-time offset (the row block and chunk >> 3). No fragment load computes its address, and
    // few registers stay live across the tile (the computed form kept dozens, and spilled them: up to 30 stores and
    // loads a tile). Instances that never spilled measured 1-8 % slower in this form (their fragment loads sat right
    // before their MMAs, where the address arithmetic had hidden the shared-memory latency): the packed 32-row tile,
    // and on two-KV-head geometries the single-sequence fused append when masked (verification) or 32 rows wide. They
    // keep the computed addresses and one K fragment a load.
    constexpr bool LaneAddresses =
        !((Packed && M == 32) || (G::KVHeads == 2 && Input::writes_cache && !MultiBatch && (Masked || M == 32)));
    const unsigned swizzle = lane & 7, upper = (lane >> 3) & 1;
    const unsigned k_lane  = (warp_kv * NK + (lane >> 4) * 8 + swizzle) * (2 * D);
    const unsigned v_lane  = (warp_kv * NK + upper * 8 + swizzle) * (2 * D);
    [[maybe_unused]] int t = 0;
    for (int k0 = start; k0 < end; k0 += N) {
        if constexpr (Packed) {
            const int slot = Ring::slot(t);
            cta_mbarrier_wait(ring->full + slot, Ring::full_parity(t));
            k_s = reinterpret_cast<__nv_bfloat16*>(storage->qkv + slot * (2 * N * D));
            v_s = reinterpret_cast<__half*>(storage->qkv + slot * (2 * N * D) + N * D);
            if constexpr (M > 64) {
                if (k0 >= warp_end) {
                    // Past this warp's 64-row tile: only hand the buffer back.
                    __syncwarp();
                    if (lane == 0) cta_mbarrier_arrive(ring->empty + slot);
                    ++t;
                    continue;
                }
            }
        } else {
            const int logical_page = k0 >> kPagedKVPageShift;
            if (logical_page >= page_window + Storage::kPageWindow) load_pages(logical_page);
            bf16_kv_load_grouped_tile<G, S>(k_s, v_s, cache_k, cache_v, input,
                                            storage->pages[logical_page - page_window], head, k0, load_end,
                                            first, tid);
            cp_commit();
            cp_wait<0>();
            __syncthreads();
        }
        float score[QKNt][4] = {};
        // One ldmatrix x4 a k step and pair of key blocks: K rows warp_kv * NK + p * 16 + (lane >> 4) * 8 + lane & 7,
        // chunk 2 k + upper, at kq[k & 3] + p * 16 rows + 128 * (k >> 2) bytes; its registers 0-1 are block 2 p's
        // fragment, 2-3 block 2 p + 1's. Each score block still sums its k steps in order (the bits are unchanged).
        const unsigned k_tile = smem_addr(k_s) + k_lane;
        unsigned kq[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) kq[q] = k_tile + 16 * ((2 * q + upper) ^ swizzle);
        // Q and O already occupy 192 registers per lane at D=256. Keep one
        // K fragment (pair) live at a time; double buffering all N fragments spills.
        if constexpr (LaneAddresses) {
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
#pragma unroll
                for (int p = 0; p < QKNt / 2; ++p) {
                    unsigned bf[4];
                    ldmatrix_x4(bf[0], bf[1], bf[2], bf[3], kq[k & 3] + p * (16 * 2 * D) + 128 * (k >> 2));
                    mma_bf16(score[2 * p][0], score[2 * p][1], score[2 * p][2], score[2 * p][3], q_frag[k][0],
                             q_frag[k][1], q_frag[k][2], q_frag[k][3], bf[0], bf[1]);
                    mma_bf16(score[2 * p + 1][0], score[2 * p + 1][1], score[2 * p + 1][2], score[2 * p + 1][3],
                             q_frag[k][0], q_frag[k][1], q_frag[k][2], q_frag[k][3], bf[2], bf[3]);
                }
            }
        } else {
#pragma unroll
            for (int n = 0; n < QKNt; ++n) {
#pragma unroll
                for (int k = 0; k < QKKs; ++k) {
                    unsigned bf[2];
                    const int row = warp_kv * NK + n * 8 + (lane & 7);
                    const int d   = k * 16 + ((lane >> 3) & 1) * 8;
                    ldmatrix_x2(bf[0], bf[1], smem_addr(k_s + row * D + causal_swizzle(row, d)));
                    mma_bf16(score[n][0], score[n][1], score[n][2], score[n][3], q_frag[k][0],
                             q_frag[k][1], q_frag[k][2], q_frag[k][3], bf[0], bf[1]);
                }
            }
        }
        float maximum[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
        for (int n = 0; n < QKNt; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int key = k0 + warp_kv * NK + n * 8 + 2 * lid + (j & 1);
                if (key >= warp_end || key > qabs[j / 2] ||
                    (Tree && !causal_tree_visible(key, first, tree[j / 2])))
                    score[n][j] = -CUDART_INF_F;
                maximum[j / 2] = fmaxf(maximum[j / 2], score[n][j]);
            }
        }
        const float alpha[2] = {state[0].update(warp_max<4>(maximum[0], 0xffffffffu), scale_log2),
                                state[1].update(warp_max<4>(maximum[1], 0xffffffffu), scale_log2)};
        unsigned pf[PVKs][4];
        float tile_sum[2] = {};
#pragma unroll
        for (int n = 0; n < QKNt; ++n) {
            const float p0 = state[0].probability(score[n][0], scale_log2);
            const float p1 = state[0].probability(score[n][1], scale_log2);
            const float p2 = state[1].probability(score[n][2], scale_log2);
            const float p3 = state[1].probability(score[n][3], scale_log2);
            tile_sum[0] += p0 + p1;
            tile_sum[1] += p2 + p3;
            pf[n / 2][(n % 2) * 2]     = pack_f16x2(p0, p1);
            pf[n / 2][(n % 2) * 2 + 1] = pack_f16x2(p2, p3);
        }
        state[0].accumulate(alpha[0], tile_sum[0]);
        state[1].accumulate(alpha[1], tile_sum[1]);
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha[0];
            acc[n][1] *= alpha[0];
            acc[n][2] *= alpha[1];
            acc[n][3] *= alpha[1];
        }
        // Batched and read-only wide tiles benefit from paired V loads. Single-row
        // fused append uses scalar fragments to leave room for its input state.
        if constexpr (S::kWarpsQ >= 4 && (MultiBatch || !Input::writes_cache)) {
            // V rows warp_kv * NK + k * 16 + upper * 8 + lane & 7, chunk n + (lane >> 4) (n even):
            // vp[(n & 7) >> 1] + k * 16 rows + 128 * (n >> 3) bytes.
            const unsigned v_tile = smem_addr(v_s) + v_lane, pair = lane >> 4;
            unsigned vp[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) vp[q] = v_tile + 16 * ((2 * q + pair) ^ swizzle);
            unsigned vf[2][4];
            const auto load_v = [&](int i, int slot) {
                const int k = i / (PVNt / 2), n = i % (PVNt / 2) * 2;
                ldmatrix_x4_t(vf[slot][0], vf[slot][1], vf[slot][2], vf[slot][3],
                              vp[(n & 7) >> 1] + k * (16 * 2 * D) + 128 * (n >> 3));
            };
            load_v(0, 0);
#pragma unroll
            for (int i = 0; i < PVKs * (PVNt / 2); ++i) {
                const int k = i / (PVNt / 2), n = i % (PVNt / 2) * 2, slot = i & 1;
                if (i + 1 < PVKs * (PVNt / 2)) load_v(i + 1, slot ^ 1);
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[k][0], pf[k][1], pf[k][2],
                        pf[k][3], vf[slot][0], vf[slot][1]);
                mma_f16(acc[n + 1][0], acc[n + 1][1], acc[n + 1][2], acc[n + 1][3], pf[k][0],
                        pf[k][1], pf[k][2], pf[k][3], vf[slot][2], vf[slot][3]);
            }
        } else {
            // Limit live V fragments too; the grouped path is bandwidth-bound. V rows as above, chunk n:
            // vq[n & 7] + k * 16 rows + 128 * (n >> 3) bytes.
            const unsigned v_tile = smem_addr(v_s) + v_lane;
            unsigned vq[8];
#pragma unroll
            for (int q = 0; q < 8; ++q) vq[q] = v_tile + 16 * (q ^ swizzle);
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
#pragma unroll
                for (int k = 0; k < PVKs; ++k) {
                    unsigned vf[2];
                    if constexpr (LaneAddresses) {
                        ldmatrix_x2_t(vf[0], vf[1], vq[n & 7] + k * (16 * 2 * D) + 128 * (n >> 3));
                    } else {
                        const int row = warp_kv * NK + k * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                        ldmatrix_x2_t(vf[0], vf[1], smem_addr(v_s + row * D + causal_swizzle(row, n * 8)));
                    }
                    mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[k][0], pf[k][1],
                            pf[k][2], pf[k][3], vf[0], vf[1]);
                }
            }
        }
        if constexpr (Packed) {
            // This warp's reads of the buffer are done: the producers may refill it.
            __syncwarp();
            if (lane == 0) cta_mbarrier_arrive(ring->empty + Ring::slot(t));
            ++t;
        } else {
            __syncthreads();
        }
    }
    // The KV stream is done: a programmatic merge may begin launching as CTAs finish (it waits for
    // this grid before reading any partial).
    pdl::trigger_dependents();
    state[0].finish();
    state[1].finish();
    float m[2]    = {state[0].maximum * scale_log2, state[1].maximum * scale_log2};
    float sums[2] = {state[0].sum, state[1].sum};
    if constexpr (S::kWarpsKV > 1) {
        consumer_sync();
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = warp_kv * M + warp_q * 16 + gid + j * 8;
            if (lid == 0) {
                storage->maximum[row] = m[j];
                storage->sum[row]     = sums[j];
            }
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
                const int d = n * 8 + 2 * lid;
                *reinterpret_cast<float2*>(storage->reduction + row * D + d) =
                    make_float2(acc[n][j * 2], acc[n][j * 2 + 1]);
            }
        }
        consumer_sync();
        if (warp_kv != 0) return;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = warp_q * 16 + gid + j * 8;
            m[j]          = -CUDART_INF_F;
            sums[j]       = 0;
#pragma unroll
            for (int w = 0; w < S::kWarpsKV; ++w) m[j] = fmaxf(m[j], storage->maximum[w * M + row]);
            float weights[S::kWarpsKV];
#pragma unroll
            for (int w = 0; w < S::kWarpsKV; ++w) {
                weights[w] = bf16_kv_state_weight(storage->maximum[w * M + row],
                                                  storage->sum[w * M + row], m[j]);
                sums[j] += weights[w] * storage->sum[w * M + row];
            }
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
                float a = 0, b = 0;
#pragma unroll
                for (int w = 0; w < S::kWarpsKV; ++w) {
                    const auto x = *reinterpret_cast<const float2*>(
                        storage->reduction + (w * M + row) * D + n * 8 + 2 * lid);
                    a += weights[w] * x.x;
                    b += weights[w] * x.y;
                }
                acc[n][2 * j]     = a;
                acc[n][2 * j + 1] = b;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        if (tokens[j] < width) {
            const int h = head * G::GroupSize + rows[j] % G::GroupSize;
#pragma unroll
            for (int n = 0; n < PVNt; ++n)
                bf16_kv_store_pair<G, true>(partial, nullptr, h, tokens[j], n * 8 + 2 * lid, width,
                                            split, acc[n][2 * j], acc[n][2 * j + 1], m[j], sums[j]);
        }
    }
}

} // namespace infernix::ops::detail
