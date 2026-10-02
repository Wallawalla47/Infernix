#pragma once

// Causal prompt kernel over a vector-quantized cache (Vq2, Q4KeyVq2Value), derived from the fast
// INT8 prompt kernel (int8/fast_tiled_mma.cuh): each warp owns 16 query rows for the whole key
// sweep, Q is rotated and encoded per G64 group for INT8 QK Tensor Cores, the online softmax
// keeps scores and probabilities in registers, and PV runs FP16 Tensor Cores over V widened in
// registers with FP16 partials promoted once per tile. What differs:
//   * A tile is 32 keys (half a page) with double-buffered INT8-G64 K/V tiles beside the
//     double-buffered code staging, or, once a call sees kVqPromptWideTileKeys keys, 64 keys (a
//     page, as the INT8 kernel) with one INT8 tile: the next tile's codes still stream during
//     this tile's math, and the few tiles of exact rows (the window, the call's own columns) load
//     before their use.
//   * Each key comes from one of three sources: the call's own columns from the per-call INT8
//     staging the append wrote, older keys from their window slots when the slot tag matches the
//     paged codes (both exact), or the key's persistent codes, expanded into the INT8 tile after
//     the stage lands (row scale repeated as each group scale). Which one a query reads follows the
//     position-only rule of vq_exact_key: the CTA's first pass over a tile reads exactly every key
//     some of its rows read exactly, and a tile meeting the band (vq_band) takes a second, masked
//     pass over those keys' codes for the rows that read them from codes.
//   * V is stored rotated: the normalized rows are inverse-rotated in registers (butterflies over
//     the lane's dimension bits and its quad) before they are stored or published as split rows.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/common/causal_geometry.h"
#include "ops/softmax_attention/dense/causal_cache/fast_prompt_common.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/fast_tiled_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/vq2/tile_io.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

// Key tiles of the prompt kernel: 32 keys, or 64 when the call sees kVqPromptWideTileKeys keys
// (measured: short chunks without a prefix, whose tiles are all exact rows, prefer 32).
inline constexpr int kVqPromptBc           = 32;
inline constexpr int kVqPromptWideBc       = 64;
inline constexpr int kVqPromptWideTileKeys = 1536;
inline constexpr int kVqPromptGroups = kCausalPromptHeadDim / kKVCacheInt8Group;

// Per-call INT8 staging of the call's own columns (kv_cache_append_vq_batch_launch).
struct VqPromptStaging {
    const std::int8_t* k_codes;
    const std::int8_t* v_codes;
    const __half* k_scales;
    const __half* v_scales;
};

template <KVCacheVqKeyCodec KeyCodec, int Warps, int Bc>
struct VqPromptShape {
    static_assert(Warps == 4 || Warps == 8);
    static_assert(Bc == kVqPromptBc || Bc == kVqPromptWideBc);
    // 64-key tiles keep one INT8 tile (shared memory); 32-key tiles double-buffer it.
    static constexpr bool SingleTile = Bc == kVqPromptWideBc;
    static constexpr int Tiles      = SingleTile ? 1 : 2;
    static constexpr int Threads    = Warps * 32;
    static constexpr int Br         = Warps * 16;
    static constexpr int KeyBytes   = kVqKeyCodeBytes<KeyCodec>;
    static constexpr int QBytes     = Br * kCausalPromptHeadDim;
    static constexpr int TileBytes  = Bc * kCausalPromptHeadDim;
    static constexpr int ScaleBytes = Bc * kVqPromptGroups * 2;
    static constexpr int StageBytes = 2 * TileBytes + 2 * ScaleBytes;
    // Staged codes: K rows, V rows, K and V row scales, K/V tags, per-key source, per-(key, role)
    // kept flags.
    static constexpr int KCodeOffset  = 0;
    static constexpr int VCodeOffset  = KCodeOffset + Bc * KeyBytes;
    static constexpr int RowScaleOffset = VCodeOffset + Bc * kVqValueCodeBytes;
    static constexpr int TagOffset    = RowScaleOffset + 2 * Bc * 2;
    static constexpr int SourceOffset = TagOffset + 2 * Bc * 4;
    static constexpr int KeptOffset   = SourceOffset + Bc;
    static constexpr int CodeBytes    = KeptOffset + 2 * Bc;
    static constexpr int StagesOffset = QBytes;
    static constexpr int CodesOffset  = StagesOffset + Tiles * StageBytes;
    static constexpr int CodebookOffset = CodesOffset + 2 * CodeBytes;
    static constexpr int SignOffset   = CodebookOffset + kKVCacheVq2Patterns * 8;
    static constexpr int SmemBytes    = SignOffset + 128 * 8;
    static_assert(StageBytes % 16 == 0 && CodeBytes % 16 == 0);
    static_assert(SmemBytes <= 99 * 1024);
};

// Key sources of a staged tile.
inline constexpr std::uint8_t kVqSourceCodes  = 0;
inline constexpr std::uint8_t kVqSourceCall   = 1;
inline constexpr std::uint8_t kVqSourceWindow = 2;

// In-register normalized inverse rotation of the two rows a lane holds: values[b][j] is
// dimension 16b + 4 * lid + j, and the lane quad (lid) holds bits 2..3 of the dimension.
__device__ __forceinline__ void vq_prompt_inverse_rotate(float (&values)[16][4], int lid) {
    constexpr unsigned FullMask = 0xffffffffu;
#pragma unroll
    for (int b = 0; b < 16; ++b) {
        const float a0 = values[b][0] + values[b][1];
        const float a1 = values[b][0] - values[b][1];
        const float a2 = values[b][2] + values[b][3];
        const float a3 = values[b][2] - values[b][3];
        values[b][0]   = a0 + a2;
        values[b][2]   = a0 - a2;
        values[b][1]   = a1 + a3;
        values[b][3]   = a1 - a3;
    }
#pragma unroll
    for (int stride = 1; stride <= 2; stride <<= 1) {
#pragma unroll
        for (int b = 0; b < 16; ++b) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float partner = __shfl_xor_sync(FullMask, values[b][j], stride);
                values[b][j] = (lid & stride) == 0 ? values[b][j] + partner : partner - values[b][j];
            }
        }
    }
#pragma unroll
    for (int span = 1; span < 16; span <<= 1) {
#pragma unroll
        for (int base = 0; base < 16; base += 2 * span) {
#pragma unroll
            for (int offset = 0; offset < span; ++offset) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float low                  = values[base + offset][j];
                    const float high                 = values[base + offset + span][j];
                    values[base + offset][j]        = low + high;
                    values[base + offset + span][j] = low - high;
                }
            }
        }
    }
#pragma unroll
    for (int b = 0; b < 16; ++b) {
#pragma unroll
        for (int j = 0; j < 4; ++j) values[b][j] *= 0x1p-4f;
    }
}

template <typename Geometry, typename Metadata, KVCacheVqKeyCodec KeyCodec, int Warps, bool Split,
          int Bc>
__global__ __launch_bounds__(VqPromptShape<KeyCodec, Warps, Bc>::Threads,
                             1) void vq_prompt_kernel(const __nv_bfloat16* __restrict__ q,
                                                      VqKvCacheView cache, Metadata metadata,
                                                      VqPromptStaging staging,
                                                      const std::int32_t* __restrict__ positions,
                                                      float scale, __nv_bfloat16* __restrict__ out,
                                                      std::int32_t width,
                                                      float* __restrict__ partial_rows,
                                                      float2* __restrict__ partial_stats) {
    using Shape                 = VqPromptShape<KeyCodec, Warps, Bc>;
    constexpr int D             = kCausalPromptHeadDim;
    constexpr int DB16          = D / 2;
    constexpr int Threads       = Shape::Threads;
    constexpr int Br            = Shape::Br;
    constexpr int Groups        = kVqPromptGroups;
    constexpr int GroupKc       = kKVCacheInt8Group / 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr int DBlocks       = D / 16;
    constexpr int GroupDBlocks  = kKVCacheInt8Group / 16;
    constexpr int PassDBlocks   = 2;
    constexpr int KeyBytes      = Shape::KeyBytes;
    constexpr int Words         = kKVCacheVq2Words;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    static_assert(Bc * Groups % 128 == 0);
    static_assert(!Split || Warps == 8, "only eight-warp CTAs split");

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::int8_t* q_i8    = reinterpret_cast<std::int8_t*>(smem_raw);
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);
    std::int8_t* codebook = reinterpret_cast<std::int8_t*>(smem_raw + Shape::CodebookOffset);
    uint2* sign_masks     = reinterpret_cast<uint2*>(smem_raw + Shape::SignOffset);

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    // Grid (QHeads, row blocks, splits). Longest-first issue order: the first QHeads CTAs take the
    // last row block of every head.
    const int row_blocks = static_cast<int>(gridDim.y);
    const int q_head     = static_cast<int>(blockIdx.x);
    const int q0         = (row_blocks - 1 - static_cast<int>(blockIdx.y)) * Br;
    const int kv_head    = q_head / Geometry::GroupSize;
    const int tokens     = metadata.valid_tokens(width);

    const auto store_row = [&](int row, int d0, float v0, float v1, float v2, float v3) {
        const int token = q0 + row;
        if (token >= width) { return; }
        const bool valid   = token < tokens;
        const uint2 packed = make_uint2(pack_bf16x2(valid ? v0 : 0.0f, valid ? v1 : 0.0f),
                                        pack_bf16x2(valid ? v2 : 0.0f, valid ? v3 : 0.0f));
        store_vec(&out[causal_prompt_q_index<Geometry>(q_head, d0, token)], packed);
    };

    if (q0 >= tokens) {
        if constexpr (!Split) {
            for (int element = tid; element < Br * (D / 4); element += Threads) {
                const int row = element / (D / 4);
                store_row(row, (element - row * (D / 4)) * 4, 0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        return;
    }

    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int window_row            = cache.window_slots == nullptr ? 0 : cache.window_slots[0];
    const bool has_window           = cache.window_tags != nullptr;
    const int rows                  = min(Br, tokens - q0);
    const int min_query_abs         = base_pos + q0;
    const int max_query_abs         = base_pos + q0 + rows - 1;
    const int key_blocks            = max_query_abs / Bc + 1;
    const VqBand band               = vq_band(min_query_abs, max_query_abs, has_window);
    const int band_first            = band.lo / Bc;
    const int band_blocks           = band.hi > band.lo ? (band.hi - 1) / Bc + 1 - band_first : 0;
    const int passes                = key_blocks + band_blocks;
    const int split           = Split ? static_cast<int>(blockIdx.z) : 0;
    const int passes_per_split = Split ? div_up(passes, static_cast<int>(gridDim.z)) : 0;
    const int v_begin         = Split ? min(passes, split * passes_per_split) : 0;
    const int v_end           = Split ? min(passes, v_begin + passes_per_split) : passes;

    vq_stage_codebook(codebook, tid, Threads);
    kv_cache_vq2_stage_sign_masks(sign_masks, tid, Threads);

    float q_scale_r[2][Groups];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int row    = warp * 16 + r;
        const int token  = q0 + row;
        const bool valid = row < rows;
        float q_values[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            q_values[k] =
                valid ? __bfloat162float(
                            q[causal_prompt_q_index<Geometry>(q_head, lane + 32 * k, token)])
                      : 0.0f;
        }
        normalized_hadamard_d256_inplace(q_values, lane);
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            const int d0    = grp * kKVCacheInt8Group + lane;
            const float x0  = q_values[2 * grp];
            const float x1  = q_values[2 * grp + 1];
            const float mx  = warp_max(fmaxf(fabsf(x0), fabsf(x1)), FullMask);
            const float qs  = mx > 0.0f ? mx / 127.0f : 0.0f;
            const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
            causal_prompt_store_byte_swizzled(q_i8, row, d0, kv_cache_int8_quant_code(x0, inv));
            causal_prompt_store_byte_swizzled(q_i8, row, d0 + 32,
                                              kv_cache_int8_quant_code(x1, inv));
            if (gid == (r & 7)) { q_scale_r[r >> 3][grp] = qs; }
        }
    }

    const auto stage_base = [&](int stage) {
        return smem_raw + Shape::StagesOffset + (Shape::SingleTile ? 0 : stage) * Shape::StageBytes;
    };
    const auto code_base = [&](int stage) {
        return smem_raw + Shape::CodesOffset + stage * Shape::CodeBytes;
    };
    // Call columns come from staging when the append staged them (a cached-only call reads them
    // through their window slots instead). The first pass over a tile of call columns that every
    // row reads exactly needs nothing else.
    const bool staged      = has_window && staging.k_codes != nullptr;
    const int exact_from   = min_query_abs - kKVWindowRecentTokens;
    const auto staged_tile = [&](VqTilePass pass) {
        const int k0 = pass.block * Bc;
        return staged && !pass.band_codes && k0 >= base_pos && k0 >= exact_from;
    };
    // A tile every row reads from codes (a band pass, or every key before the window).
    const auto codes_only_tile = [&](VqTilePass pass) {
        const int k0 = pass.block * Bc;
        return pass.band_codes || !has_window ||
               (k0 >= kKVWindowSinkTokens && k0 + Bc <= exact_from);
    };

    // Stages pass v's 32-key tile: call columns as INT8 staging rows, window keys as window slots
    // and tags, every other key's code row and row scale (a band pass stages codes only). Keys past
    // the CTA's last visible key are zero-filled so masked columns stay finite.
    const auto issue_tile = [&](int v) {
        const VqTilePass pass = vq_tile_pass(v, band_first, band_blocks);
        const int kb         = pass.block;
        unsigned char* base  = stage_base(v & 1);
        std::int8_t* k_s     = reinterpret_cast<std::int8_t*>(base);
        std::int8_t* v_s     = k_s + Shape::TileBytes;
        __half* ks_s         = reinterpret_cast<__half*>(v_s + Shape::TileBytes);
        __half* vs_s         = ks_s + Bc * Groups;
        const int k0         = kb * Bc;
        const int valid      = min(Bc, max_query_abs + 1 - k0);
        const auto copy_row  = [&](int key, int chunk, const std::int8_t* k_src,
                                  const std::int8_t* v_src, bool live) {
            const int dst = key * D + ((chunk ^ (key & 7)) << 4);
            cp_async_zfill<16, Cache::cg>(k_s + dst, k_src + chunk * 16, live ? 16 : 0);
            cp_async_zfill<16, Cache::cg>(v_s + dst, v_src + chunk * 16, live ? 16 : 0);
        };
        if (staged_tile(pass)) {
            for (int item = tid; item < Bc * 17; item += Threads) {
                const int key    = item / 17;
                const int part   = item - key * 17;
                const bool live  = key < valid;
                const std::int64_t column =
                    static_cast<std::int64_t>(kv_head) * width + (live ? k0 + key - base_pos : 0);
                if (part < 16) {
                    copy_row(key, part, staging.k_codes + column * D, staging.v_codes + column * D,
                             live);
                } else {
                    cp_async_zfill<8>(&ks_s[key * Groups], staging.k_scales + column * Groups,
                                      live ? 8 : 0);
                    cp_async_zfill<8>(&vs_s[key * Groups], staging.v_scales + column * Groups,
                                      live ? 8 : 0);
                }
            }
            cp_commit();
            return;
        }
        unsigned char* codes  = code_base(v & 1);
        std::uint8_t* source  = codes + Shape::SourceOffset;
        const int page        = block_table[k0 >> kPagedKVPageShift];
        const int page_offset = k0 & kPagedKVPageMask;
        const std::int64_t head_page =
            static_cast<std::int64_t>(page) * Geometry::KVHeads + kv_head;
        const std::uint8_t* k_src =
            cache.key_codes + (head_page * kPagedKVPageSize + page_offset) * KeyBytes;
        const std::uint8_t* v_src =
            cache.value_codes + (head_page * kPagedKVPageSize + page_offset) * kVqValueCodeBytes;
        for (int chunk = tid; chunk < Bc * KeyBytes / 16; chunk += Threads) {
            const int key = chunk * 16 / KeyBytes;
            cp_async_zfill<16, Cache::cg>(codes + Shape::KCodeOffset + chunk * 16,
                                          k_src + chunk * 16, key < valid ? 16 : 0);
        }
        for (int chunk = tid; chunk < Bc * kVqValueCodeBytes / 16; chunk += Threads) {
            const int key = chunk * 16 / kVqValueCodeBytes;
            cp_async_zfill<16, Cache::cg>(codes + Shape::VCodeOffset + chunk * 16,
                                          v_src + chunk * 16, key < valid ? 16 : 0);
        }
        if (tid < 2 * Bc / 8) {
            const int role  = tid / (Bc / 8);
            const int chunk = tid - role * (Bc / 8);
            const __half* src = (role ? cache.value_scales : cache.key_scales) +
                                head_page * kPagedKVPageSize + page_offset + chunk * 8;
            cp_async<16>(codes + Shape::RowScaleOffset + (role * Bc + chunk * 8) * 2, src);
        }
        if (codes_only_tile(pass)) {
            cp_commit();
            return;
        }
        // Per-key sources of the first pass: keys some row reads exactly come from staging (call
        // columns) or their window slots; a band pass reads codes only.
        for (int item = tid; item < Bc * 17; item += Threads) {
            const int key_l = item / 17;
            const int part  = item - key_l * 17;
            const int key   = k0 + key_l;
            std::uint8_t kind = kVqSourceCodes;
            if (key_l < valid && !pass.band_codes && has_window &&
                vq_exact_key(key, min_query_abs)) {
                kind = staged && key >= base_pos ? kVqSourceCall : kVqSourceWindow;
            }
            if (part == 16) source[key_l] = kind;
            if (kind == kVqSourceCall) {
                const std::int64_t column = static_cast<std::int64_t>(kv_head) * width + key -
                                            base_pos;
                if (part < 16)
                    copy_row(key_l, part, staging.k_codes + column * D,
                             staging.v_codes + column * D, true);
                else {
                    cp_async<8>(&ks_s[key_l * Groups], staging.k_scales + column * Groups);
                    cp_async<8>(&vs_s[key_l * Groups], staging.v_scales + column * Groups);
                }
            } else if (kind == kVqSourceWindow) {
                const int slot = kv_window_slot(key);
                if (part < 16) {
                    const std::int64_t src =
                        kv_window_code_index<Geometry::KVHeads>(window_row, kv_head, slot, 0);
                    copy_row(key_l, part, cache.window_k + src, cache.window_v + src, true);
                } else {
                    const std::int64_t s =
                        kv_window_scale_index<Geometry::KVHeads>(window_row, kv_head, slot, 0);
                    cp_async<8>(&ks_s[key_l * Groups], cache.window_k_scales + s);
                    cp_async<8>(&vs_s[key_l * Groups], cache.window_v_scales + s);
                    cp_async<8>(reinterpret_cast<std::int32_t*>(codes + Shape::TagOffset) +
                                    2 * key_l,
                                cache.window_tags +
                                    kv_window_tag_index<Geometry::KVHeads>(window_row, kv_head,
                                                                           slot, 0));
                }
            }
        }
        cp_commit();
    };

    // After a staged tile lands: keep call columns and matching window slots, expand every other
    // key's codes into the INT8 tile.
    const auto finish_tile = [&](int v) {
        const VqTilePass pass = vq_tile_pass(v, band_first, band_blocks);
        if (staged_tile(pass)) { return; }
        const int kb                = pass.block;
        unsigned char* base         = stage_base(v & 1);
        std::int8_t* k_s            = reinterpret_cast<std::int8_t*>(base);
        std::int8_t* v_s            = k_s + Shape::TileBytes;
        __half* ks_s                = reinterpret_cast<__half*>(v_s + Shape::TileBytes);
        __half* vs_s                = ks_s + Bc * Groups;
        unsigned char* codes        = code_base(v & 1);
        const std::uint8_t* source  = codes + Shape::SourceOffset;
        std::uint8_t* kept          = codes + Shape::KeptOffset;
        const __half* row_scales    = reinterpret_cast<const __half*>(codes + Shape::RowScaleOffset);
        const std::uint8_t* k_codes = codes + Shape::KCodeOffset;
        const std::uint8_t* v_codes = codes + Shape::VCodeOffset;
        const int k0                = kb * Bc;
        const int valid             = min(Bc, max_query_abs + 1 - k0);
        // A tile of codes only needs no tag check.
        const bool codes_only = codes_only_tile(pass);
        const int word = lane;
        if (codes_only) {
#pragma unroll
            for (int i = 0; i < Bc / Warps; ++i) {
                const int key_l = warp + Warps * i;
                const int at = key_l * D + (((word >> 1) ^ (key_l & 7)) << 4) + (word & 1) * 8;
                *reinterpret_cast<uint2*>(&k_s[at]) = vq_decode_key_word<KeyCodec>(
                    k_codes + key_l * KeyBytes, word, codebook, sign_masks);
                *reinterpret_cast<uint2*>(&v_s[at]) = vq_decode_value_word(
                    v_codes + key_l * kVqValueCodeBytes, word, codebook, sign_masks);
            }
            if (tid < 2 * Bc) {
                const int role  = tid / Bc;
                const int key_l = tid - role * Bc;
                const __half s  = key_l < valid ? row_scales[role * Bc + key_l]
                                                : __float2half_rn(0.0f);
                const __half2 pair = __halves2half2(s, s);
                uint2 packed;
                packed.x = load_vec<unsigned>(&pair);
                packed.y = packed.x;
                *reinterpret_cast<uint2*>((role ? vs_s : ks_s) + key_l * Groups) = packed;
            }
            __syncthreads();
            return;
        }
        {
            for (int item = tid; item < 2 * Bc; item += Threads) {
                const int key_l = item >> 1;
                const int role  = item & 1;
                bool keep       = source[key_l] == kVqSourceCall;
                if (source[key_l] == kVqSourceWindow) {
                    const int key = k0 + key_l;
                    const std::uint32_t tag =
                        role ? vq_row_tag<kVqValueCodeBytes>(
                                   key, v_codes + key_l * kVqValueCodeBytes, row_scales[Bc + key_l])
                             : vq_row_tag<KeyBytes>(key, k_codes + key_l * KeyBytes,
                                                    row_scales[key_l]);
                    keep = static_cast<std::uint32_t>(reinterpret_cast<const std::int32_t*>(
                               codes + Shape::TagOffset)[item]) == tag;
                }
                kept[item] = keep ? 1 : 0;
            }
            __syncthreads();
        }
        // Thread (warp w, lane l) expands word l of keys w, w + Warps, ... of K and V.
        static_assert(Words == 32 && Bc % Warps == 0);
#pragma unroll
        for (int i = 0; i < Bc / Warps; ++i) {
            const int key_l = warp + Warps * i;
            const int at    = key_l * D + (((word >> 1) ^ (key_l & 7)) << 4) + (word & 1) * 8;
            if (kept[2 * key_l] == 0)
                *reinterpret_cast<uint2*>(&k_s[at]) = vq_decode_key_word<KeyCodec>(
                    k_codes + key_l * KeyBytes, word, codebook, sign_masks);
            if (kept[2 * key_l + 1] == 0)
                *reinterpret_cast<uint2*>(&v_s[at]) = vq_decode_value_word(
                    v_codes + key_l * kVqValueCodeBytes, word, codebook, sign_masks);
        }
        if (tid < 2 * Bc) {
            const int role  = tid / Bc;
            const int key_l = tid - role * Bc;
            if (kept[2 * key_l + role] == 0) {
                const __half s = key_l < valid ? row_scales[role * Bc + key_l]
                                               : __float2half_rn(0.0f);
                const __half2 pair = __halves2half2(s, s);
                uint2 packed;
                packed.x = load_vec<unsigned>(&pair);
                packed.y = packed.x;
                *reinterpret_cast<uint2*>((role ? vs_s : ks_s) + key_l * Groups) = packed;
            }
        }
        __syncthreads();
    };

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_koff   = ((lane >> 3) & 1) << 3;
    const int row_base = warp * 16;

    const bool warp_active  = row_base < rows;
    const int warp_min_qabs = base_pos + q0 + row_base;
    const int warp_max_qabs = base_pos + q0 + min(row_base + 15, rows - 1);
    const int row0          = row_base + gid;
    const int row1          = row0 + 8;
    const int qabs0         = row0 < rows ? base_pos + q0 + row0 : -1;
    const int qabs1         = row1 < rows ? base_pos + q0 + row1 : -1;

    float acc[DBlocks][2][4];
#pragma unroll
    for (int b = 0; b < DBlocks; ++b) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[b][p][i] = 0.0f; }
        }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    const float scale_l2 = scale * Log2E;

    unsigned pa[PVKs][4];
    float tile_alpha0 = 0.0f;
    float tile_alpha1 = 0.0f;
    int tile_shift    = 0;
    bool tile_live    = false;

    const auto qk_softmax = [&](int v) {
        tile_live             = false;
        const VqTilePass pass = vq_tile_pass(v, band_first, band_blocks);
        const int k0          = pass.block * Bc;
        if (!warp_active || k0 > warp_max_qabs) { return; }
        const unsigned char* base  = stage_base(v & 1);
        const __nv_bfloat16* k_b16 = reinterpret_cast<const __nv_bfloat16*>(base);
        const __half* ks_s         = reinterpret_cast<const __half*>(base + 2 * Shape::TileBytes);
        const __half* vs_s         = ks_s + Bc * Groups;

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            unsigned af[GroupKc][4];
#pragma unroll
            for (int kk = 0; kk < GroupKc; ++kk) {
                const int acol = (grp * GroupKc + kk) * 16 + a_coloff;
                ldmatrix_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                            smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                             causal_prompt_swz(row_base + a_rowoff, acol)]));
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int brow = nt * 8 + a_rin;
                    const int bcol = (grp * GroupKc + kk) * 16 + b_koff;
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&k_b16[brow * DB16 + causal_prompt_swz(brow, bcol)]));
                    mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                           bf[1]);
                }
                const int keya  = nt * 8 + 2 * lid;
                const float ks0 = __half2float(ks_s[keya * Groups + grp]);
                const float ks1 = __half2float(ks_s[(keya + 1) * Groups + grp]);
                const float qs0 = q_scale_r[0][grp];
                const float qs1 = q_scale_r[1][grp];
                score[nt][0]    = __fmaf_rn(qs0 * ks0, static_cast<float>(c0), score[nt][0]);
                score[nt][1]    = __fmaf_rn(qs0 * ks1, static_cast<float>(c1), score[nt][1]);
                score[nt][2]    = __fmaf_rn(qs1 * ks0, static_cast<float>(c2), score[nt][2]);
                score[nt][3]    = __fmaf_rn(qs1 * ks1, static_cast<float>(c3), score[nt][3]);
            }
        }

        const bool full_tile = k0 + Bc - 1 <= warp_min_qabs;
        const bool band_tile = k0 < band.hi && k0 + Bc > band.lo;
        const auto keeps     = [&](int key, int qabs) {
            return key <= qabs && (!band_tile || vq_pass_keeps(key, qabs, band, pass.band_codes));
        };
        float bm0 = -CUDART_INF_F;
        float bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            if (!full_tile || band_tile) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                score[nt][0]   = keeps(key0, qabs0) ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = keeps(key1, qabs0) ? score[nt][1] : -CUDART_INF_F;
                score[nt][2]   = keeps(key0, qabs1) ? score[nt][2] : -CUDART_INF_F;
                score[nt][3]   = keeps(key1, qabs1) ? score[nt][3] : -CUDART_INF_F;
            }
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0        = fmaxf(running_m0, bm0);
        const float nm1        = fmaxf(running_m1, bm1);
        const float nm0_scaled = nm0 == -CUDART_INF_F ? 0.0f : nm0 * scale_l2;
        const float nm1_scaled = nm1 == -CUDART_INF_F ? 0.0f : nm1 * scale_l2;
        tile_alpha0            = running_m0 == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
        tile_alpha1            = running_m1 == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
        running_m0             = nm0;
        running_m1             = nm1;

        float bl0 = 0.0f;
        float bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
            const float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
            const float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
            const float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            pa[nt >> 1][(nt & 1) * 2 + 0] = pack_f16x2(p00, p01);
            pa[nt >> 1][(nt & 1) * 2 + 1] = pack_f16x2(p10, p11);
        }
        running_l0 = __fmaf_rn(running_l0, tile_alpha0, bl0);
        running_l1 = __fmaf_rn(running_l1, tile_alpha1, bl1);

        // A 32-key FP16 partial is bounded by 32 * 127 * max_scale; the INT8 kernel's limit keeps
        // it representable with twice the margin.
        __half2 vmax2 = __float2half2_rn(0.0f);
#pragma unroll
        for (int c = 0; c < Bc * Groups / 128; ++c) {
            const uint2 v4 = load_vec<uint2>(&vs_s[4 * lane + 128 * c]);
            vmax2          = __hmax2(vmax2, __hmax2(__habs2(load_vec<__half2>(&v4.x)),
                                                    __habs2(load_vec<__half2>(&v4.y))));
        }
        const float vmax = warp_max(fmaxf(__low2float(vmax2), __high2float(vmax2)), FullMask);
        tile_shift       = 0;
        if (vmax > kCausalPromptI8FastF16PartialScaleLimit) {
            const float bounded = fminf(vmax, 65504.0f);
            while (ldexpf(bounded, -tile_shift) > kCausalPromptI8FastF16PartialScaleLimit) {
                ++tile_shift;
            }
        }
        tile_live = true;
    };

    const auto pv = [&](int v) {
        if (!tile_live) { return; }
        const unsigned char* base = stage_base(v & 1);
        const std::int8_t* v_s =
            reinterpret_cast<const std::int8_t*>(base) + Shape::TileBytes;
        const __half* vs_s =
            reinterpret_cast<const __half*>(v_s + Shape::TileBytes) + Bc * Groups;
        const __half2 mul   = __float2half2_rn(ldexpf(1.0f, -tile_shift));
        const float unscale = ldexpf(1.0f, tile_shift);
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            unsigned vsc[PVKs][2];
#pragma unroll
            for (int j = 0; j < PVKs; ++j) {
                const int key = j * 16 + 2 * lid;
                __half2 lo =
                    __halves2half2(vs_s[key * Groups + grp], vs_s[(key + 1) * Groups + grp]);
                __half2 hi =
                    __halves2half2(vs_s[(key + 8) * Groups + grp], vs_s[(key + 9) * Groups + grp]);
                if (tile_shift != 0) {
                    lo = __hmul2(lo, mul);
                    hi = __hmul2(hi, mul);
                }
                vsc[j][0] = load_vec<unsigned>(&lo);
                vsc[j][1] = load_vec<unsigned>(&hi);
            }
#pragma unroll
            for (int pass = 0; pass < GroupDBlocks / PassDBlocks; ++pass) {
                const int db0 = grp * GroupDBlocks + pass * PassDBlocks;
                unsigned h[PassDBlocks][2][2];
#pragma unroll
                for (int b = 0; b < PassDBlocks; ++b) {
#pragma unroll
                    for (int p = 0; p < 2; ++p) { h[b][p][0] = h[b][p][1] = 0u; }
                }
#pragma unroll
                for (int j = 0; j < PVKs; ++j) {
#pragma unroll
                    for (int q2 = 0; q2 < PassDBlocks / 2; ++q2) {
                        const int db    = db0 + 2 * q2;
                        const int key   = j * 16 + ((a_mat & 1) << 3) + a_rin;
                        const int chunk = db + (a_mat >> 1);
                        unsigned r[4];
                        ldmatrix_x4_t(r[0], r[1], r[2], r[3],
                                      smem_addr(&v_s[key * D + ((chunk ^ (key & 7)) << 4)]));
#pragma unroll
                        for (int b = 0; b < 2; ++b) {
                            unsigned even_lo, odd_lo, even_hi, odd_hi;
                            causal_prompt_i8_fast_decode_v_pair(r[2 * b], vsc[j][0], even_lo,
                                                                odd_lo);
                            causal_prompt_i8_fast_decode_v_pair(r[2 * b + 1], vsc[j][1], even_hi,
                                                                odd_hi);
                            unsigned (&he)[2] = h[2 * q2 + b][0];
                            unsigned (&ho)[2] = h[2 * q2 + b][1];
                            causal_prompt_i8_fast_mma_f16_acc(he[0], he[1], pa[j][0], pa[j][1],
                                                              pa[j][2], pa[j][3], even_lo, even_hi);
                            causal_prompt_i8_fast_mma_f16_acc(ho[0], ho[1], pa[j][0], pa[j][1],
                                                              pa[j][2], pa[j][3], odd_lo, odd_hi);
                        }
                    }
                }
#pragma unroll
                for (int b = 0; b < PassDBlocks; ++b) {
#pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        float (&a)[4] = acc[db0 + b][p];
                        float2 r0     = __half22float2(load_vec<__half2>(&h[b][p][0]));
                        float2 r1     = __half22float2(load_vec<__half2>(&h[b][p][1]));
                        if (tile_shift != 0) {
                            r0.x *= unscale;
                            r0.y *= unscale;
                            r1.x *= unscale;
                            r1.y *= unscale;
                        }
                        a[0] = __fmaf_rn(a[0], tile_alpha0, r0.x);
                        a[1] = __fmaf_rn(a[1], tile_alpha0, r0.y);
                        a[2] = __fmaf_rn(a[2], tile_alpha1, r1.x);
                        a[3] = __fmaf_rn(a[3], tile_alpha1, r1.y);
                    }
                }
            }
        }
    };

    // With one INT8 tile, only codes stream ahead: a tile of exact rows loads at its own turn,
    // once the previous tile's math is done with the INT8 tile.
    const auto prefetched = [&](int v) {
        return !Shape::SingleTile || codes_only_tile(vq_tile_pass(v, band_first, band_blocks));
    };
    if (v_begin < v_end && prefetched(v_begin)) { issue_tile(v_begin); }
#pragma unroll 1
    for (int v = v_begin; v < v_end; ++v) {
        cp_wait<0>();
        __syncthreads();
        if (!prefetched(v)) {
            issue_tile(v);
            cp_wait<0>();
            __syncthreads();
        }
        if (v + 1 < v_end && prefetched(v + 1)) { issue_tile(v + 1); }
        finish_tile(v);
        qk_softmax(v);
        pv(v);
    }

    running_l0         = warp_sum<4>(running_l0, FullMask);
    running_l1         = warp_sum<4>(running_l1, FullMask);
    const float inv_l0 = running_l0 > 0.0f ? __frcp_rn(running_l0) : 0.0f;
    const float inv_l1 = running_l1 > 0.0f ? __frcp_rn(running_l1) : 0.0f;
    // Lane (g, t) owns dimensions 16b + 4t + j of its two rows: j = 0..3 are the even and odd n8
    // tiles' first and second registers.
    float row_values[2][DBlocks][4];
#pragma unroll
    for (int b = 0; b < DBlocks; ++b) {
        row_values[0][b][0] = acc[b][0][0] * inv_l0;
        row_values[0][b][1] = acc[b][1][0] * inv_l0;
        row_values[0][b][2] = acc[b][0][1] * inv_l0;
        row_values[0][b][3] = acc[b][1][1] * inv_l0;
        row_values[1][b][0] = acc[b][0][2] * inv_l1;
        row_values[1][b][1] = acc[b][1][2] * inv_l1;
        row_values[1][b][2] = acc[b][0][3] * inv_l1;
        row_values[1][b][3] = acc[b][1][3] * inv_l1;
    }
    vq_prompt_inverse_rotate(row_values[0], lid);
    vq_prompt_inverse_rotate(row_values[1], lid);
    if constexpr (Split) {
        const auto publish = [&](int row, int r, float maximum, float sum) {
            if (row >= rows) { return; }
            const std::int64_t index =
                causal_prompt_fast_partial_row<Geometry>(split, q0 + row, q_head, width);
            if (lid == 0) { partial_stats[index] = make_float2(maximum, sum); }
            float* target = partial_rows + index * D;
#pragma unroll
            for (int b = 0; b < DBlocks; ++b) {
                store_vec(target + b * 16 + 4 * lid,
                          make_float4(row_values[r][b][0], row_values[r][b][1],
                                      row_values[r][b][2], row_values[r][b][3]));
            }
        };
        publish(row0, 0, running_m0, running_l0);
        publish(row1, 1, running_m1, running_l1);
    } else {
#pragma unroll
        for (int b = 0; b < DBlocks; ++b) {
            const int d0 = b * 16 + 4 * lid;
            store_row(row0, d0, row_values[0][b][0], row_values[0][b][1], row_values[0][b][2],
                      row_values[0][b][3]);
            store_row(row1, d0, row_values[1][b][0], row_values[1][b][1], row_values[1][b][2],
                      row_values[1][b][3]);
        }
    }
}

} // namespace ninfer::ops::detail
