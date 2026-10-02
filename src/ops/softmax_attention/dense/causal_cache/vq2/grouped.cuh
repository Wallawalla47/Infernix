#pragma once
#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/common/causal_epilogue.cuh"
#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/common/causal_softmax.cuh"
#include "ops/softmax_attention/common/causal_tile_io.cuh"
#include "ops/softmax_attention/common/causal_tree.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/fast_tiled_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/vq2/operands.h"
#include "ops/softmax_attention/dense/causal_cache/vq2/tile_io.cuh"

namespace ninfer::ops::detail {

// Grouped attention over a vector-quantized cache for rows of up to eight columns (eight warps),
// and for wider rows as independent sixteen-column CTAs (ParallelQueries, sixteen warps, so a tile's
// codes are expanded once for sixteen columns). The arithmetic is the INT8-G64 grouped
// kernel's: Q is rotated and encoded per G64 group, QK runs on INT8 Tensor Cores with FP32 scales,
// the online softmax runs over 32- or 64-key tiles with FP16 probabilities and PV runs FP16 Tensor Cores
// with FP32 accumulation over V widened in registers. V stays in the rotated basis; the natural
// merge applies the inverse rotation. The append runs as a separate launch before this kernel.
//
// A row's keys split by how its columns read them (vq_exact_key), and each split CTA has one role:
//   * Window CTAs (the first `window_splits` splits) take the sink and recent tiles. Each tile's
//     INT8 window slots, tags and paged codes land in one buffer; a slot whose tag does not match
//     the codes is replaced by its expanded codes. They keep the pairs a column reads exactly.
//   * Code CTAs (the remaining splits) partition the tiles some column reads from codes. The
//     compact code rows stream double-buffered and each tile is expanded once into the INT8 tile
//     (the row scale repeated as each group scale). They keep the pairs a column reads from codes.
// Every (key, column) pair is kept by exactly one CTA, and every CTA publishes a partial (an empty
// one the neutral state) so the merge reads all window_splits + capacity splits. QK spreads the
// tile's 8-key column blocks, times the row tiles, over up to all the warps, which exchange row
// maxima through shared memory; PV splits the dimensions over all the warps.
template <KVCacheVqKeyCodec KeyCodec, int KeyRows>
struct VqKvGroupedShape {
    static_assert(KeyRows == 32 || KeyRows == 64);
    static constexpr int kKeyRows  = KeyRows;
    static constexpr int kKeyBytes = kVqKeyCodeBytes<KeyCodec>;
    // One INT8 tile: K rows, V rows, then K and V G64 scales (as the INT8 kernels).
    static constexpr int kTileBytes = 2 * kKeyRows * 256 + 2 * kKeyRows * kKVCacheInt8Groups * 2;
    // A code stage: K code rows, V code rows, then K and V row scales.
    static constexpr int kStageVOffset     = kKeyRows * kKeyBytes;
    static constexpr int kStageScaleOffset = kStageVOffset + kKeyRows * kVqValueCodeBytes;
    static constexpr int kStageBytes       = (kStageScaleOffset + 2 * kKeyRows * 2 + 15) / 16 * 16;
    static constexpr int kStagesOffset     = kTileBytes;
    // Window CTAs use one stage; their slot tags and kept flags take the second stage's place.
    static constexpr int kTagOffset      = kStagesOffset + kStageBytes;
    static constexpr int kKeptOffset     = kTagOffset + 2 * kKeyRows * 4;
    static constexpr int kCodebookOffset = kStagesOffset + 2 * kStageBytes;
    // VQ2 keys also stage the sign-mask table (K4V2's 8-column CTA has no room left for two per SM;
    // its keys need no table).
    static constexpr bool kSignTable = KeyCodec == KVCacheVqKeyCodec::Vq2;
    static constexpr int kSignOffset = kCodebookOffset + kKVCacheVq2Patterns * 8;
    static constexpr int kArenaBytes = kSignOffset + (kSignTable ? 128 * 8 : 0);
    static_assert(kStageBytes % 16 == 0 && kTileBytes % 16 == 0);
    static_assert(kKeptOffset + 2 * kKeyRows <= kCodebookOffset);
};

// The per-block shared-memory limit of this target (sm_120).
inline constexpr std::uint32_t kVqGroupedSmemLimit = 101376;

// Shared memory one grouped CTA of TokenTile columns needs: the arena plus the static query,
// probability, scale and exchange arrays. The launcher picks the key tile with it, since a 64-key
// tile does not fit every geometry:
//   * H24/KV4, 16 columns, k4v2: 62976 B of arena and 40320 B of static arrays (103296 B) exceed
//     the limit, while vq2's 64-byte K rows need 96128 B;
//   * H16/KV2 has twice the rows per column tile, so only 32-key tiles fit there.
template <KVCacheVqKeyCodec KeyCodec, int KeyRows, int TokenTile, int GroupSize>
inline constexpr std::uint32_t vq_kv_grouped_smem_bytes() {
    constexpr int Br    = (TokenTile * GroupSize + 15) / 16 * 16;
    constexpr int QKNt  = KeyRows / 8;
    return VqKvGroupedShape<KeyCodec, KeyRows>::kArenaBytes + Br * 256 + Br * KeyRows * 2 + Br * 4 +
           Br * QKNt * 4;
}

// Columns of one ParallelQueries CTA, and the most window splits of a row: the sink and recent
// tiles of one CTA's columns (at most 64 + 768 + 16 keys, 28 tiles of 32) spread over that many
// CTAs.
inline constexpr int kVqParallelTile   = 16;
inline constexpr int kVqMaxWindowSplits = 8;

// Warps of a grouped CTA of TokenTile columns.
template <int TokenTile>
inline constexpr int kVqGroupedWarps = TokenTile > 8 ? 16 : 8;

// The partition the merge reads: every one of a row's `splits` partials.
inline CausalKvPartition vq_merge_partition(int splits) {
    CausalKvPartition partition;
    partition.capacity  = splits;
    partition.target    = splits;
    partition.key_shift = 0;
    return partition;
}

template <class Geometry, int TokenTile, bool MultiBatch, bool Masked, KVCacheVqKeyCodec KeyCodec,
          int KeyRows, bool ParallelQueries, bool Tree>
__launch_bounds__(kVqGroupedWarps<TokenTile> * 32, TokenTile > 8 ? 1 : 2) __global__
    void vq_kv_grouped_kernel(const __nv_bfloat16* q, const std::int32_t* pos, VqKvCacheView cache,
                              std::int32_t full_width, std::int32_t logical_capacity,
                              CausalKvPartition partition, std::int32_t window_splits, float scale,
                              float* partial_acc, float* partial_m, float* partial_l) {
    using Shape                 = VqKvGroupedShape<KeyCodec, KeyRows>;
    constexpr int Wc            = kVqGroupedWarps<TokenTile>;
    constexpr int RowCount      = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles      = (RowCount + 15) / 16;
    constexpr int Br            = RowTiles * 16;
    constexpr int Bc            = Shape::kKeyRows;
    constexpr int D             = 256;
    constexpr int DB16          = D / 2;
    constexpr int Threads       = Wc * 32;
    constexpr int Groups        = kKVCacheInt8Groups;
    constexpr int GroupKc       = kKVCacheInt8Group / 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr int KeyBytes      = Shape::kKeyBytes;
    constexpr int Words         = kKVCacheVq2Words;
    constexpr float Log2E       = kLog2E;
    constexpr unsigned FullMask = 0xffffffffu;
    // QK: warp w takes column block w % 4 for the row tiles t with t % (QKWarps / 4) == w / 4.
    constexpr int QKWarps      = Wc < QKNt * RowTiles ? Wc : QKNt * RowTiles;
    constexpr int TileGroups   = QKWarps / QKNt;
    constexpr int WarpRowTiles = (RowTiles + TileGroups - 1) / TileGroups;
    // PV: warp w owns dimensions [w * DimsPerWarp, (w + 1) * DimsPerWarp).
    constexpr int DimsPerWarp = D / Wc;
    constexpr int DBlocks     = DimsPerWarp / 16;

    static_assert(Wc == 8 || Wc == 16);
    static_assert(RowTiles >= 1 && RowTiles * DBlocks <= 8, "PV accumulators: 64 registers");
    static_assert((QKNt == 4 || QKNt == 8) && QKWarps % QKNt == 0);
    static_assert(Bc <= kPagedKVPageSize && kPagedKVPageSize % Bc == 0);
    static_assert(kKVWindowSinkTokens % Bc == 0);

    __shared__ __align__(16) std::int8_t q_s[Br * D];
    extern __shared__ __align__(16) std::int8_t arena[];
    __shared__ __align__(16) __half p_s[Br * Bc];
    __shared__ float alpha_s[Br];
    __shared__ __align__(16) float exchange_s[Br * QKNt];
    // One block must hold the arena and these arrays: the per-block opt-in limit of the target.
    static_assert(Shape::kArenaBytes + Br * D + Br * Bc * 2 + Br * 4 + Br * QKNt * 4 <=
                      kVqGroupedSmemLimit,
                  "grouped kernel: shared memory exceeds one block on this target");
    std::int8_t* q_i8    = q_s;
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);
    std::int8_t* tile_k  = arena;
    std::int8_t* tile_v  = arena + Bc * D;
    __half* tile_ks      = reinterpret_cast<__half*>(arena + 2 * Bc * D);
    __half* tile_vs      = tile_ks + Bc * Groups;
    const auto stage     = [&](int s) {
        return reinterpret_cast<std::uint8_t*>(arena + Shape::kStagesOffset + s * Shape::kStageBytes);
    };
    std::int32_t* tags    = reinterpret_cast<std::int32_t*>(arena + Shape::kTagOffset);
    std::uint8_t* kept    = reinterpret_cast<std::uint8_t*>(arena + Shape::kKeptOffset);
    std::int8_t* codebook = arena + Shape::kCodebookOffset;
    uint2* sign_masks     = reinterpret_cast<uint2*>(arena + Shape::kSignOffset);
    // The query scales pass through the second stage, which nothing touches before the key loop.
    float* q_scale_tmp = reinterpret_cast<float*>(stage(1));
    static_assert(RowCount * kKVCacheInt8Groups * 4 <= Shape::kStageBytes);

    const int kv_head      = ParallelQueries ? blockIdx.x % Geometry::KVHeads : blockIdx.x;
    const int column_begin = ParallelQueries ? (blockIdx.x / Geometry::KVHeads) * TokenTile : 0;
    const int tile_tokens = ParallelQueries ? min(TokenTile, full_width - column_begin) : TokenTile;
    const int partial_width = full_width;
    const int partial_begin = column_begin;
    const int split         = static_cast<int>(blockIdx.y);
    const int batch         = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    // Launched as the append's programmatic dependent: window CTAs read the keys it writes and
    // wait for it first (every one of them, so this grid's completion implies the append's); code
    // CTAs read only keys older than any column's window and overlap it.
    const bool window_role = split < window_splits;
    if (window_role) pdl::wait_for_dependencies();
    const int tid           = static_cast<int>(threadIdx.x);
    const int warp          = tid >> 5;
    const int lane          = tid & 31;
    const int gid           = lane >> 2;
    const int lid           = lane & 3;

    int valid_tokens = tile_tokens;
    if constexpr (Masked) {
        const int remaining = cache.valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : min(remaining, tile_tokens);
    }
    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(256) * Geometry::QHeads * column_base;
    const int last_pos  = pos[(MultiBatch ? batch * full_width : 0) + full_width - 1];
    const int row_first = pos[MultiBatch ? batch * full_width : 0];
    pos += column_base;
    const int table_row  = cache.table_rows == nullptr ? 0 : cache.table_rows[batch];
    const int window_row = cache.window_slots == nullptr ? 0 : cache.window_slots[batch];
    const std::int32_t* block_table =
        cache.tables + static_cast<std::int64_t>(table_row) * cache.table_stride;
    if constexpr (MultiBatch) {
        const int split_count = static_cast<int>(gridDim.y);
        partial_acc +=
            static_cast<std::int64_t>(batch) * 256 * Geometry::QHeads * partial_width * split_count;
        partial_m +=
            static_cast<std::int64_t>(batch) * Geometry::QHeads * partial_width * split_count;
        partial_l +=
            static_cast<std::int64_t>(batch) * Geometry::QHeads * partial_width * split_count;
    }

    if (valid_tokens == 0) return; // Merge writes exact zero for masked columns.
    if (pos[0] < 0 || last_pos < 0 || last_pos >= logical_capacity) return;
    // A masked row's keys end at its last valid column: columns past it were never appended.
    int live_end = last_pos + 1;
    if constexpr (Masked) live_end = row_first + max(0, cache.valid_columns[batch]);
    live_end = min(live_end, causal_row_window(row_first, full_width, logical_capacity));
    const bool has_window = cache.window_tags != nullptr;
    // This CTA's columns sit at positions [query_first, query_last].
    const int query_first = pos[0];
    const int query_last  = pos[valid_tokens - 1];

    // This CTA's tiles: tile i of [tile_begin, tile_end) is key tile tile_of(i).
    const int live_tiles   = div_up(live_end, Bc);
    int sink_tiles = 0, recent_first = 0, tile_begin = 0, tile_end = 0;
    int code_lo = 0, code_hi = 0;
    if (window_role) {
        if (has_window) {
            sink_tiles         = min(kKVWindowSinkTokens / Bc, live_tiles);
            recent_first       = max(sink_tiles, max(0, query_first - kKVWindowRecentTokens) / Bc);
            // The lowest splits take the tiles: a row shorter than window_splits keys
            // merges only its first `window` splits (vq_merge_partition).
            const int count    = sink_tiles + max(0, live_tiles - recent_first);
            const int per      = div_up(count, window_splits);
            tile_begin         = min(count, split * per);
            tile_end           = min(count, tile_begin + per);
        }
    } else {
        code_lo = has_window ? kKVWindowSinkTokens : 0;
        code_hi = has_window ? min(live_end, query_last - kKVWindowRecentTokens) : live_end;
        if (code_hi > code_lo) {
            const int first_tile = code_lo / Bc;
            const int count      = div_up(code_hi, Bc) - first_tile;
            const int active     = partition.active(code_hi);
            const int c          = split - window_splits;
            if (c < active) {
                tile_begin = first_tile + c * count / active;
                tile_end   = first_tile + (c + 1) * count / active;
            }
        }
    }
    const auto tile_of = [&](int i) {
        if (!window_role) return i;
        return i < sink_tiles ? i : recent_first + (i - sink_tiles);
    };
    // Whether this CTA keeps key `key` for a column at position `qabs`.
    const auto role_keeps = [&](int key, int qabs) {
        if (window_role) return key < live_end && vq_exact_key(key, qabs);
        return key >= code_lo && key < code_hi && (!has_window || !vq_exact_key(key, qabs));
    };

    // Code CTAs: stage the compact code rows and row scales of key tile `k0` (live keys only).
    const auto issue_codes = [&](std::uint8_t* target, int k0) {
        const int physical_page = block_table[k0 >> kPagedKVPageShift];
        const int page_offset   = k0 & kPagedKVPageMask;
        const std::int64_t head_page =
            static_cast<std::int64_t>(physical_page) * Geometry::KVHeads + kv_head;
        const std::uint8_t* k_src =
            cache.key_codes + (head_page * kPagedKVPageSize + page_offset) * KeyBytes;
        const std::uint8_t* v_src =
            cache.value_codes + (head_page * kPagedKVPageSize + page_offset) * kVqValueCodeBytes;
        for (int chunk = tid; chunk < Bc * KeyBytes / 16; chunk += Threads) {
            const int key = k0 + chunk * 16 / KeyBytes;
            cp_async_zfill<16>(target + chunk * 16, k_src + chunk * 16, key < live_end ? 16 : 0);
        }
        for (int chunk = tid; chunk < Bc * kVqValueCodeBytes / 16; chunk += Threads) {
            const int key = k0 + chunk * 16 / kVqValueCodeBytes;
            cp_async_zfill<16>(target + Shape::kStageVOffset + chunk * 16, v_src + chunk * 16,
                               key < live_end ? 16 : 0);
        }
        if (tid < 2 * Bc / 8) {
            // 8 FP16 row scales per chunk: K chunks then V chunks; dead keys are cleared at decode.
            const int role  = tid / (Bc / 8);
            const int chunk = tid - role * (Bc / 8);
            const __half* src = (role ? cache.value_scales : cache.key_scales) +
                                head_page * kPagedKVPageSize + page_offset + chunk * 8;
            cp_async<16>(target + Shape::kStageScaleOffset + (role * Bc + chunk * 8) * 2, src);
        }
    };

    // Expands the staged codes of the (key, role) pairs without a kept slot into the INT8 tile,
    // the row scale repeated as each group scale (zero for dead keys). Thread (warp w, lane l)
    // expands word l of keys w, w + Wc, ... of K and V.
    const auto expand_codes = [&](const std::uint8_t* codes, int k0, bool use_kept) {
        static_assert(Words == 32 && Bc % Wc == 0);
        const int word = lane;
#pragma unroll
        for (int i = 0; i < Bc / Wc; ++i) {
            const int key_l = warp + Wc * i;
            const std::uint8_t* k_row = codes + key_l * KeyBytes;
            const std::uint8_t* v_row = codes + Shape::kStageVOffset + key_l * kVqValueCodeBytes;
            if (!use_kept || kept[2 * key_l] == 0) {
                *reinterpret_cast<uint2*>(&tile_k[key_l * D + causal_swizzle(key_l, (word >> 1) * 8) * 2 +
                                                  (word & 1) * 8]) =
                    Shape::kSignTable
                        ? vq_decode_key_word<KeyCodec>(k_row, word, codebook, sign_masks)
                        : vq_decode_key_word<KeyCodec>(k_row, word, codebook);
            }
            if (!use_kept || kept[2 * key_l + 1] == 0) {
                *reinterpret_cast<uint2*>(
                    &tile_v[key_l * D + (((word >> 1) ^ (key_l & 7)) << 4) + (word & 1) * 8]) =
                    Shape::kSignTable ? vq_decode_value_word(v_row, word, codebook, sign_masks)
                                      : vq_decode_value_word(v_row, word, codebook);
            }
        }
        if (tid < 2 * Bc) {
            const int role  = tid / Bc;
            const int key_l = tid - role * Bc;
            if (!use_kept || kept[2 * key_l + role] == 0) {
                const __half* row_scales =
                    reinterpret_cast<const __half*>(codes + Shape::kStageScaleOffset);
                const __half s = k0 + key_l < live_end ? row_scales[role * Bc + key_l]
                                                       : __float2half_rn(0.0f);
                const __half2 pair = __halves2half2(s, s);
                uint2 packed;
                packed.x = load_vec<unsigned>(&pair);
                packed.y = packed.x;
                *reinterpret_cast<uint2*>((role ? tile_vs : tile_ks) + key_l * Groups) = packed;
            }
        }
    };

    // Window CTAs: load key tile `k0` into the INT8 tile (window slots, or expanded codes where a
    // slot's tag does not match the codes). Ends with a barrier.
    const auto load_window_tile = [&](int k0) {
        std::uint8_t* codes = stage(0);
        issue_codes(codes, k0);
        for (int item = tid; item < Bc * 17; item += Threads) {
            // Items 0..15 of a key copy its 16-byte K and V slot chunks, item 16 its scales and tags.
            const int key_l = item / 17;
            const int part  = item - key_l * 17;
            const int key   = k0 + key_l;
            if (key >= live_end) continue;
            const int slot = kv_window_slot(key);
            if (part < 16) {
                const std::int64_t src =
                    kv_window_code_index<Geometry::KVHeads>(window_row, kv_head, slot, part * 16);
                cp_async<16>(&tile_k[key_l * D + causal_swizzle(key_l, part * 8) * 2],
                             cache.window_k + src);
                cp_async<16>(&tile_v[key_l * D + ((part ^ (key_l & 7)) << 4)], cache.window_v + src);
            } else {
                const std::int64_t s =
                    kv_window_scale_index<Geometry::KVHeads>(window_row, kv_head, slot, 0);
                cp_async<8>(&tile_ks[key_l * Groups], cache.window_k_scales + s);
                cp_async<8>(&tile_vs[key_l * Groups], cache.window_v_scales + s);
                cp_async<8>(&tags[2 * key_l],
                            cache.window_tags +
                                kv_window_tag_index<Geometry::KVHeads>(window_row, kv_head, slot, 0));
            }
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();
        const __half* row_scales = reinterpret_cast<const __half*>(codes + Shape::kStageScaleOffset);
        for (int item = tid; item < 2 * Bc; item += Threads) {
            const int key_l = item >> 1;
            const int role  = item & 1;
            const int key   = k0 + key_l;
            bool keep       = false;
            if (key < live_end) {
                const std::uint32_t tag =
                    role ? vq_row_tag<kVqValueCodeBytes>(
                               key, codes + Shape::kStageVOffset + key_l * kVqValueCodeBytes,
                               row_scales[Bc + key_l])
                         : vq_row_tag<KeyBytes>(key, codes + key_l * KeyBytes, row_scales[key_l]);
                keep = static_cast<std::uint32_t>(tags[item]) == tag;
            }
            kept[item] = keep ? 1 : 0;
        }
        __syncthreads();
        expand_codes(codes, k0, true);
        __syncthreads();
    };

    vq_stage_codebook(codebook, tid, Threads);
    if constexpr (Shape::kSignTable) kv_cache_vq2_stage_sign_masks(sign_masks, tid, Threads);
    if (!window_role && tile_begin < tile_end) issue_codes(stage(0), tile_of(tile_begin) * Bc);
    cp_commit();

    for (int i = tid; i < Br * D; i += Threads) { q_i8[i] = 0; }
    for (int i = tid; i < RowCount * Groups; i += Threads) { q_scale_tmp[i] = 0.0f; }
    __syncthreads();

    for (int row = warp; row < RowCount; row += Wc) {
        int q_head = 0;
        int token  = 0;
        causal_row_to_qt<Geometry>(row, kv_head, q_head, token);
        float q_values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            q_values[r] = token < valid_tokens
                              ? __bfloat162float(q[causal_q_index<Geometry>(q_head, d, token)])
                              : 0.0f;
        }
        normalized_hadamard_d256_inplace(q_values, lane);
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            const int d0    = grp * kKVCacheInt8Group + lane;
            const int d1    = d0 + 32;
            const float x0  = q_values[2 * grp];
            const float x1  = q_values[2 * grp + 1];
            float amax      = fmaxf(fabsf(x0), fabsf(x1));
            amax            = warp_max(amax, FullMask);
            const float qs  = amax > 0.0f ? amax / 127.0f : 0.0f;
            const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
            causal_store_query_code(q_i8, row, d0, kv_cache_int8_quant_code(x0, inv));
            causal_store_query_code(q_i8, row, d1, kv_cache_int8_quant_code(x1, inv));
            if (lane == 0) { q_scale_tmp[row * Groups + grp] = qs; }
        }
    }
    __syncthreads();

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    // QK warp state: column block nt, row tiles row_tile(j), and per row tile the two rows a lane
    // holds (gid, gid + 8): query scales, positions, tree masks, running maximum and sum.
    const bool qk_warp = warp < QKWarps;
    const int nt       = warp & (QKNt - 1);
    const auto row_tile = [&](int j) { return (warp / QKNt) + j * TileGroups; };
    float q_scale_r[WarpRowTiles][2][Groups];
    int qabs_r[WarpRowTiles][2];
    std::uint32_t tree_r[WarpRowTiles][2];
    float m_r[WarpRowTiles][2];
    float l_r[WarpRowTiles][2];
#pragma unroll
    for (int j = 0; j < WarpRowTiles; ++j) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int t   = row_tile(j);
            const int row = t * 16 + gid + 8 * h;
            const bool on = qk_warp && t < RowTiles && row < tile_tokens * Geometry::GroupSize;
#pragma unroll
            for (int g = 0; g < Groups; ++g)
                q_scale_r[j][h][g] = on && row < RowCount ? q_scale_tmp[row * Groups + g] : 0.0f;
            int q_head = 0, token = 0;
            causal_row_to_qt<Geometry>(row, kv_head, q_head, token);
            qabs_r[j][h] = on ? pos[token] : -1;
            tree_r[j][h] = on ? causal_tree_mask<Tree>(cache.tree_masks, batch, full_width,
                                                       column_begin + token)
                              : ~0u;
            m_r[j][h] = -CUDART_INF_F;
            l_r[j][h] = 0.0f;
        }
    }

    // acc[row tile][d-block][even/odd n8 tile] over this warp's dimensions.
    float acc[RowTiles][DBlocks][2][4];
#pragma unroll
    for (int t = 0; t < RowTiles; ++t) {
#pragma unroll
        for (int b = 0; b < DBlocks; ++b) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { acc[t][b][e][i] = 0.0f; }
            }
        }
    }
    const int v_group = (warp * DimsPerWarp) / kKVCacheInt8Group;

#pragma unroll 1
    for (int i = tile_begin; i < tile_end; ++i) {
        const int k0 = tile_of(i) * Bc;
        if (window_role) {
            __syncthreads(); // The previous tile's PV is done with the INT8 tile.
            load_window_tile(k0);
        } else {
            const int s = (i - tile_begin) & 1;
            cp_wait<0>();
            __syncthreads();
            if (i + 1 < tile_end) issue_codes(stage(s ^ 1), tile_of(i + 1) * Bc);
            cp_commit();
            expand_codes(stage(s), k0, false);
            __syncthreads();
        }

        // QK over this warp's 8 keys; each row's block maximum goes to the exchange.
        // (Keys before the row's first column precede every column and any tree.)
        const bool full_tile =
            !window_role && k0 >= code_lo && k0 + Bc <= code_hi &&
            k0 + Bc <= (has_window ? query_first - kKVWindowRecentTokens : row_first);
        float score[WarpRowTiles][4];
        if (qk_warp) {
            const __nv_bfloat16* k_b16 = reinterpret_cast<const __nv_bfloat16*>(tile_k);
            const int key0             = k0 + nt * 8 + 2 * lid;
            float ks0[Groups], ks1[Groups];
#pragma unroll
            for (int g = 0; g < Groups; ++g) {
                ks0[g] = __half2float(tile_ks[(nt * 8 + 2 * lid) * Groups + g]);
                ks1[g] = __half2float(tile_ks[(nt * 8 + 2 * lid + 1) * Groups + g]);
            }
            // This warp's 8 keys as B fragments of every 32-dimension chunk.
            unsigned bf[Groups * GroupKc][2];
#pragma unroll
            for (int k = 0; k < Groups * GroupKc; ++k) {
                const int brow = nt * 8 + b_rin;
                const int bcol = k * 16 + b_koff;
                ldmatrix_x2(bf[k][0], bf[k][1],
                            smem_addr(&k_b16[brow * DB16 + causal_swizzle(brow, bcol)]));
            }
#pragma unroll
            for (int j = 0; j < WarpRowTiles; ++j) {
                score[j][0] = score[j][1] = score[j][2] = score[j][3] = 0.0f;
                const int t = row_tile(j);
                if (t >= RowTiles) continue;
#pragma unroll
                for (int g = 0; g < Groups; ++g) {
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                    for (int kk = 0; kk < GroupKc; ++kk) {
                        const int k    = g * GroupKc + kk;
                        const int acol = k * 16 + a_coloff;
                        unsigned af[4];
                        ldmatrix_x4(af[0], af[1], af[2], af[3],
                                    smem_addr(&q_b16[(t * 16 + a_rowoff) * DB16 +
                                                     causal_swizzle(t * 16 + a_rowoff, acol)]));
                        mma_s8(c0, c1, c2, c3, af[0], af[1], af[2], af[3], bf[k][0], bf[k][1]);
                    }
                    score[j][0] += q_scale_r[j][0][g] * ks0[g] * static_cast<float>(c0);
                    score[j][1] += q_scale_r[j][0][g] * ks1[g] * static_cast<float>(c1);
                    score[j][2] += q_scale_r[j][1][g] * ks0[g] * static_cast<float>(c2);
                    score[j][3] += q_scale_r[j][1][g] * ks1[g] * static_cast<float>(c3);
                }
                // A code tile below every column's window is kept whole by each live row.
                const auto keeps = [&](int key, int h) {
                    const int qabs = qabs_r[j][h];
                    if (full_tile) return qabs >= 0;
                    return key <= qabs && role_keeps(key, qabs) &&
                           (!Tree || causal_tree_visible(key, row_first, tree_r[j][h]));
                };
                score[j][0] = keeps(key0, 0) ? score[j][0] * scale : -CUDART_INF_F;
                score[j][1] = keeps(key0 + 1, 0) ? score[j][1] * scale : -CUDART_INF_F;
                score[j][2] = keeps(key0, 1) ? score[j][2] * scale : -CUDART_INF_F;
                score[j][3] = keeps(key0 + 1, 1) ? score[j][3] * scale : -CUDART_INF_F;
                const float bm0 = warp_max<4>(fmaxf(score[j][0], score[j][1]), FullMask);
                const float bm1 = warp_max<4>(fmaxf(score[j][2], score[j][3]), FullMask);
                if (lid == 0) {
                    exchange_s[(t * 16 + gid) * QKNt + nt]     = bm0;
                    exchange_s[(t * 16 + gid + 8) * QKNt + nt] = bm1;
                }
            }
        }
        __syncthreads();

        // Softmax: every warp of a row takes the same new maximum from the exchange.
        if (qk_warp) {
#pragma unroll
            for (int j = 0; j < WarpRowTiles; ++j) {
                const int t = row_tile(j);
                if (t >= RowTiles) continue;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int row = t * 16 + gid + 8 * h;
                    float nm      = m_r[j][h];
#pragma unroll
                    for (int c = 0; c < QKNt; c += 4) {
                        const float4 m = *reinterpret_cast<const float4*>(&exchange_s[row * QKNt + c]);
                        nm = fmaxf(nm, fmaxf(fmaxf(m.x, m.y), fmaxf(m.z, m.w)));
                    }
                    const float alpha =
                        m_r[j][h] == -CUDART_INF_F ? 0.0f
                                                   : causal_exp_difference(m_r[j][h], nm, Log2E);
                    const float s0 = score[j][2 * h];
                    const float s1 = score[j][2 * h + 1];
                    const float p0 = (nm > -CUDART_INF_F && s0 > -CUDART_INF_F)
                                         ? causal_exp_difference(s0, nm, Log2E)
                                         : 0.0f;
                    const float p1 = (nm > -CUDART_INF_F && s1 > -CUDART_INF_F)
                                         ? causal_exp_difference(s1, nm, Log2E)
                                         : 0.0f;
                    const int col0 = nt * 8 + 2 * lid;
                    p_s[row * Bc + causal_probability_swizzle<Bc>(row, col0)]     = __float2half_rn(p0);
                    p_s[row * Bc + causal_probability_swizzle<Bc>(row, col0 + 1)] = __float2half_rn(p1);
                    l_r[j][h] = l_r[j][h] * alpha + warp_sum<4>(p0 + p1, FullMask);
                    m_r[j][h] = nm;
                    if (nt == 0 && lid == 0) alpha_s[row] = alpha;
                }
            }
        }
        __syncthreads();

        // PV: this warp's dimensions for every row tile. Each lane widens its own V codes; the
        // even/odd n8 tiles of a 16-dimension block hold dimensions 16b + 2n and 16b + 2n + 1.
#pragma unroll
        for (int t = 0; t < RowTiles; ++t) {
            const float alpha0 = alpha_s[t * 16 + gid];
            const float alpha1 = alpha_s[t * 16 + gid + 8];
#pragma unroll
            for (int b = 0; b < DBlocks; ++b) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    acc[t][b][e][0] *= alpha0;
                    acc[t][b][e][1] *= alpha0;
                    acc[t][b][e][2] *= alpha1;
                    acc[t][b][e][3] *= alpha1;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < PVKs; ++j) {
            const int key = j * 16 + 2 * lid;
            __half2 lo_scale = __halves2half2(tile_vs[key * Groups + v_group],
                                              tile_vs[(key + 1) * Groups + v_group]);
            __half2 hi_scale = __halves2half2(tile_vs[(key + 8) * Groups + v_group],
                                              tile_vs[(key + 9) * Groups + v_group]);
            const unsigned lo = load_vec<unsigned>(&lo_scale);
            const unsigned hi = load_vec<unsigned>(&hi_scale);
            // r[2b], r[2b + 1]: keys 0-7 and 8-15 of the tile's 16-key step, dimension block b.
            unsigned r[2 * DBlocks];
            {
                const int vkey  = j * 16 + ((a_mat & 1) << 3) + a_rin;
                const int chunk = warp * DBlocks + (DBlocks == 2 ? (a_mat >> 1) : 0);
                const unsigned address = smem_addr(&tile_v[vkey * D + ((chunk ^ (vkey & 7)) << 4)]);
                if constexpr (DBlocks == 2) {
                    ldmatrix_x4_t(r[0], r[1], r[2], r[3], address);
                } else {
                    ldmatrix_x2_t(r[0], r[1], address);
                }
            }
            unsigned vf[DBlocks][2][2]; // [d-block][even/odd][lo/hi keys]
#pragma unroll
            for (int b = 0; b < DBlocks; ++b) {
                causal_prompt_i8_fast_decode_v_pair(r[2 * b], lo, vf[b][0][0], vf[b][1][0]);
                causal_prompt_i8_fast_decode_v_pair(r[2 * b + 1], hi, vf[b][0][1], vf[b][1][1]);
            }
#pragma unroll
            for (int t = 0; t < RowTiles; ++t) {
                const __half* p_consumer = &p_s[t * 16 * Bc];
                unsigned pf[4];
                const int pcol = j * 16 + a_coloff;
                ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                            smem_addr(&p_consumer[a_rowoff * Bc +
                                                  causal_probability_swizzle<Bc>(a_rowoff, pcol)]));
#pragma unroll
                for (int b = 0; b < DBlocks; ++b) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        mma_f16(acc[t][b][e][0], acc[t][b][e][1], acc[t][b][e][2], acc[t][b][e][3],
                                pf[0], pf[1], pf[2], pf[3], vf[b][e][0], vf[b][e][1]);
                    }
                }
            }
        }
    }
    // The KV stream is done: a programmatic merge may begin launching as CTAs finish.
    pdl::trigger_dependents();

    // Row sums: the column-block warps of a row add their partial sums.
    __syncthreads();
    if (qk_warp) {
#pragma unroll
        for (int j = 0; j < WarpRowTiles; ++j) {
            const int t = row_tile(j);
            if (t >= RowTiles || lid != 0) continue;
            exchange_s[(t * 16 + gid) * QKNt + nt]     = l_r[j][0];
            exchange_s[(t * 16 + gid + 8) * QKNt + nt] = l_r[j][1];
        }
    }
    __syncthreads();
    if (qk_warp && nt == 0 && lid == 0) {
#pragma unroll
        for (int j = 0; j < WarpRowTiles; ++j) {
            const int t = row_tile(j);
            if (t >= RowTiles) continue;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = t * 16 + gid + 8 * h;
                if (row >= tile_tokens * Geometry::GroupSize) continue;
                float sum = 0.0f;
#pragma unroll
                for (int c = 0; c < QKNt; c += 4) {
                    const float4 l = *reinterpret_cast<const float4*>(&exchange_s[row * QKNt + c]);
                    sum += (l.x + l.y) + (l.z + l.w);
                }
                int q_head = 0, token = 0;
                causal_row_to_qt<Geometry>(row, kv_head, q_head, token);
                const std::int64_t index =
                    causal_stat_index<Geometry>(q_head, partial_begin + token, split, partial_width);
                partial_m[index] = m_r[j][h];
                partial_l[index] = sum;
            }
        }
    }

    // Lane (g, t) owns dimensions 16b + 4t .. 16b + 4t + 3 of rows g and g + 8 of each tile.
#pragma unroll
    for (int t = 0; t < RowTiles; ++t) {
        const int row0 = t * 16 + gid;
        const int row1 = row0 + 8;
#pragma unroll
        for (int b = 0; b < DBlocks; ++b) {
            const int d0 = (warp * DBlocks + b) * 16 + 4 * lid;
            if (row0 < tile_tokens * Geometry::GroupSize) {
                int q_head = 0;
                int token  = 0;
                causal_row_to_qt<Geometry>(row0, kv_head, q_head, token);
                const std::int64_t dst = causal_partial_index<Geometry>(
                    q_head, d0, partial_begin + token, split, partial_width);
                causal_store_partial_pair(&partial_acc[dst], acc[t][b][0][0], acc[t][b][1][0]);
                causal_store_partial_pair(&partial_acc[dst + 2], acc[t][b][0][1], acc[t][b][1][1]);
            }
            if (row1 < tile_tokens * Geometry::GroupSize) {
                int q_head = 0;
                int token  = 0;
                causal_row_to_qt<Geometry>(row1, kv_head, q_head, token);
                const std::int64_t dst = causal_partial_index<Geometry>(
                    q_head, d0, partial_begin + token, split, partial_width);
                causal_store_partial_pair(&partial_acc[dst], acc[t][b][0][2], acc[t][b][1][2]);
                causal_store_partial_pair(&partial_acc[dst + 2], acc[t][b][0][3], acc[t][b][1][3]);
            }
        }
    }
}

} // namespace ninfer::ops::detail
