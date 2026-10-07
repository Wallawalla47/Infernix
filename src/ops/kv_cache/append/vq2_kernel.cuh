#pragma once

#include "core/pdl.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/kv_window.cuh"
#include "ops/kv_cache/q4_lloyd_codec.cuh"
#include "ops/kv_cache/vq2_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops {

// One warp per word in the arg-min reduction: warp w scans patterns [16w, 16w + 16) for every
// word, then reduces word w over the warps.
inline constexpr int kKVCacheVq2EncodeWarps     = kKVCacheVq2Words;
inline constexpr int kKVCacheVq2PatternsPerWarp = kKVCacheVq2Patterns / kKVCacheVq2EncodeWarps;
inline constexpr int kKVCacheVqAppendThreads    = 32 * kKVCacheVq2EncodeWarps;

// Persistent code bytes per row of each key codec.
enum class KVCacheVqKeyCodec { Vq2, Q4 };

template <KVCacheVqKeyCodec Codec>
inline constexpr int kKVCacheVqKeyCodeBytes =
    Codec == KVCacheVqKeyCodec::Vq2 ? kKVCacheVq2CodeBytes : kKVCacheQ4CodeBytes;

// Where the window copies of an append go: nowhere (no window), the row's window slots, or the
// per-call staging rows [256, width, Hkv] that a wide call reads and commits afterwards.
enum class KVCacheVqWindowMode { None, Slots, Staging };

struct KVCacheVqWindowTarget {
    std::int8_t* k_codes       = nullptr;
    std::int8_t* v_codes       = nullptr;
    __half* k_scales           = nullptr;
    __half* v_scales           = nullptr;
    std::int32_t* tags         = nullptr;
    // Window slot of each batch row (null: the planes belong to the one sequence, slot 0).
    const std::int32_t* slots  = nullptr;
};

// The encoder's FP32 codebook (a warp reads one pattern per step as two float4 broadcasts) and
// each pattern's exact squared norm.
struct KVCacheVqCodebookShared {
    __align__(16) float codebook[kKVCacheVq2Patterns * 8];
    float norm[kKVCacheVq2Patterns];
};

// One row being encoded: the rotated values, its code words and square sum.
struct KVCacheVqRowShared {
    float y[kD256KVCacheHeadDim];
    std::uint32_t code_words[kKVCacheQ4CodeBytes / 4];
    float sum_square;
};

// The block encoder's arg-min exchange.
struct KVCacheVqBlockShared {
    float best_distance[kKVCacheVq2EncodeWarps][kKVCacheVq2Words + 1];
    int best_pattern[kKVCacheVq2EncodeWarps][kKVCacheVq2Words + 1];
    int word_pattern[kKVCacheVq2Words];
};

__device__ __forceinline__ void kv_cache_vq2_stage_encoder_codebook(KVCacheVqCodebookShared& book,
                                                                    int thread, int threads) {
    for (int i = thread; i < kKVCacheVq2Patterns * 8; i += threads)
        book.codebook[i] = static_cast<float>(g_kv_cache_vq2_codebook[i]);
    for (int p = thread; p < kKVCacheVq2Patterns; p += threads) {
        int norm = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int v = g_kv_cache_vq2_codebook[8 * p + j];
            norm += v * v;
        }
        book.norm[p] = static_cast<float>(norm);
    }
}

template <int CodeBytes>
__device__ __forceinline__ std::int64_t kv_cache_vq_code_index(int kv_heads, int physical_page,
                                                               int kv_head, int page_offset) {
    return (static_cast<std::int64_t>(physical_page) * kv_heads + kv_head) * kPagedKVPageSize *
               CodeBytes +
           static_cast<std::int64_t>(page_offset) * CodeBytes;
}

__device__ __forceinline__ std::int64_t kv_cache_vq_scale_index(int kv_heads, int physical_page,
                                                                int kv_head, int page_offset) {
    return (static_cast<std::int64_t>(physical_page) * kv_heads + kv_head) * kPagedKVPageSize +
           page_offset;
}

// One warp: rotate the BF16 row into row.y and its square sum into row.sum_square.
__device__ __forceinline__ void kv_cache_vq_rotate_row(const __nv_bfloat16* source,
                                                       KVCacheVqRowShared& row, int lane) {
    float values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) values[r] = __bfloat162float(source[lane + 32 * r]);
    normalized_hadamard_d256_inplace(values, lane);
    float sq = 0.0f;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        row.y[lane + 32 * r] = values[r];
        sq                   = __fmaf_rn(values[r], values[r], sq);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, offset);
    if (lane == 0) row.sum_square = sq;
}

// Scaled magnitudes and signs of word `lane` (bit j of `negative` set = y_j < 0).
__device__ __forceinline__ void kv_cache_vq2_word_magnitudes(const KVCacheVqRowShared& row,
                                                             int lane, float sigma, float (&A)[8],
                                                             unsigned& negative) {
    const float to_units = sigma > 0.0f ? 16.0f / sigma : 0.0f;
    negative             = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float y = row.y[8 * lane + j];
        A[j]          = __fmul_rn(fabsf(y), to_units);
        negative |= (y < 0.0f ? 1u : 0u) << j;
    }
}

// The first nearest of patterns [first, first + Count) (strict <: the smallest index on ties).
template <int Count>
__device__ __forceinline__ void kv_cache_vq2_scan(const KVCacheVqCodebookShared& book,
                                                  const float (&A)[8], bool odd, int first,
                                                  float& best, int& best_p) {
    best   = 3.402823466e38f;
    best_p = first;
#pragma unroll 4
    for (int i = 0; i < Count; ++i) {
        const int p = first + i;
        float pattern[8];
        *reinterpret_cast<float4*>(&pattern[0]) =
            *reinterpret_cast<const float4*>(&book.codebook[8 * p]);
        *reinterpret_cast<float4*>(&pattern[4]) =
            *reinterpret_cast<const float4*>(&book.codebook[8 * p + 4]);
        const float d = kv_cache_vq2_pattern_distance(A, odd, pattern, book.norm[p]);
        if (d < best) {
            best   = d;
            best_p = p;
        }
    }
}

// Word `lane`'s code for pattern p into row.code_words; returns sum(y * c) over the word.
__device__ __forceinline__ float kv_cache_vq2_finish_word(const KVCacheVqCodebookShared& book,
                                                          KVCacheVqRowShared& row, int lane,
                                                          float sigma, const float (&A)[8],
                                                          unsigned negative, int p) {
    int c[8];
    std::uint32_t code = 0;
    float dot          = 0.0f;
    if (sigma > 0.0f) {
        code = kv_cache_vq2_word_code(A, negative, p, &book.codebook[8 * p], c);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            dot = __fmaf_rn(row.y[8 * lane + j], static_cast<float>(c[j]), dot);
    }
    reinterpret_cast<std::uint16_t*>(row.code_words)[lane] = static_cast<std::uint16_t>(code);
    return dot;
}

// Q4 key encoder in one warp (lane = eight coordinates); returns sum(y * c) per lane.
__device__ __forceinline__ float kv_cache_q4_encode_warp(KVCacheVqRowShared& row, int lane,
                                                         float sigma) {
    const float to_units = sigma > 0.0f ? 32.0f / sigma : 0.0f;
    std::uint32_t word   = 0;
    float dot            = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float y      = row.y[8 * lane + j];
        const int m        = kv_cache_q4_magnitude_index(__fmul_rn(fabsf(y), to_units));
        const bool neg     = y < 0.0f;
        const std::uint32_t n = neg ? static_cast<std::uint32_t>(7 - m)
                                    : static_cast<std::uint32_t>(8 + m);
        word |= n << (4 * j);
        const float level = static_cast<float>(kv_cache_q4_level(m));
        const float c     = neg ? -level : level;
        dot           = __fmaf_rn(y, c, dot);
    }
    if (sigma == 0.0f) {
        word = 0x77777777u;
        dot  = 0.0f;
    }
    row.code_words[lane] = word;
    return dot;
}

// Where an append row goes: its paged codes and scale, and its window slot or staging row (null
// window pointers: none).
struct KVCacheVqRowTarget {
    const __nv_bfloat16* source  = nullptr;
    int position                 = 0;
    bool value                   = false;
    std::uint8_t* codes          = nullptr;
    __half* scale                = nullptr;
    std::int8_t* window_codes    = nullptr;
    __half* window_scales        = nullptr;
    std::int32_t* window_tag     = nullptr;
};

// One warp, after the row's codes are in row.code_words and each lane holds its word's sum(y * c):
// write the paged codes and scale, and optionally the window slot or staging row with the tag of
// the written codes.
template <int CodeWords>
__device__ __forceinline__ void kv_cache_vq_store_row(KVCacheVqRowShared& row, int lane, float dot,
                                                      const KVCacheVqRowTarget& target) {
    constexpr unsigned Full = 0xffffffffu;
    const float sum_square  = row.sum_square;
    const float sigma       = sqrtf(sum_square * (1.0f / 256.0f));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) dot += __shfl_xor_sync(Full, dot, offset);
    __syncwarp();
    const __half s =
        __float2half_rn(sigma > 0.0f && dot > 0.0f ? __fdiv_rn(sum_square, dot) : 0.0f);
    if (lane < CodeWords)
        reinterpret_cast<std::uint32_t*>(target.codes)[lane] = row.code_words[lane];
    if (lane == 0) *target.scale = s;
    if (target.window_codes == nullptr) return;

    // Window copy: INT8-G64 codes of the rotated row, group g = coordinates [64g, 64g + 64).
#pragma unroll
    for (int g = 0; g < kKVWindowGroups; ++g) {
        const float x0   = row.y[64 * g + lane];
        const float x1   = row.y[64 * g + 32 + lane];
        const float amax = warp_max(fmaxf(fabsf(x0), fabsf(x1)), Full);
        const auto quant = kv_cache_int8_quant_params(amax);
        target.window_codes[64 * g + lane]      = kv_cache_int8_quant_code(x0, quant.inverse_scale);
        target.window_codes[64 * g + 32 + lane] = kv_cache_int8_quant_code(x1, quant.inverse_scale);
        if (lane == 0) target.window_scales[g] = quant.scale;
    }
    if (lane == 0) {
        *target.window_tag = static_cast<std::int32_t>(kv_window_tag<CodeWords>(
            target.position, row.code_words, static_cast<std::uint32_t>(__half_as_ushort(s))));
    }
}

// Encodes one row (role K or V) of the block's (token, kv_head) with the whole block: warp 0
// rotates, every warp scans its pattern range, warp w reduces word w, warp 0 finishes the row.
template <KVCacheVqKeyCodec KeyCodec>
__device__ __forceinline__ void kv_cache_vq_append_row_block(const KVCacheVqRowTarget& target,
                                                             const KVCacheVqCodebookShared& book,
                                                             KVCacheVqRowShared& row,
                                                             KVCacheVqBlockShared& block) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (warp == 0) kv_cache_vq_rotate_row(target.source, row, lane);
    __syncthreads();
    const float sigma = sqrtf(row.sum_square * (1.0f / 256.0f));
    if (KeyCodec == KVCacheVqKeyCodec::Q4 && !target.value) {
        if (warp != 0) return;
        const float dot = kv_cache_q4_encode_warp(row, lane, sigma);
        __syncwarp();
        kv_cache_vq_store_row<kKVCacheQ4CodeBytes / 4>(row, lane, dot, target);
        return;
    }
    float A[8];
    unsigned negative = 0;
    kv_cache_vq2_word_magnitudes(row, lane, sigma, A, negative);
    const bool odd = (__popc(negative) & 1) != 0;
    float best     = 0.0f;
    int best_p     = 0;
    kv_cache_vq2_scan<kKVCacheVq2PatternsPerWarp>(book, A, odd, warp * kKVCacheVq2PatternsPerWarp,
                                                  best, best_p);
    block.best_distance[warp][lane] = best;
    block.best_pattern[warp][lane]  = best_p;
    __syncthreads();
    {
        float d = block.best_distance[lane][warp];
        int p   = block.best_pattern[lane][warp];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_d = __shfl_xor_sync(0xffffffffu, d, offset);
            const int other_p   = __shfl_xor_sync(0xffffffffu, p, offset);
            if (other_d < d || (other_d == d && other_p < p)) {
                d = other_d;
                p = other_p;
            }
        }
        if (lane == 0) block.word_pattern[warp] = p;
    }
    __syncthreads();
    if (warp != 0) return;
    const float dot =
        kv_cache_vq2_finish_word(book, row, lane, sigma, A, negative, block.word_pattern[lane]);
    __syncwarp();
    kv_cache_vq_store_row<kKVCacheVq2CodeBytes / 4>(row, lane, dot, target);
}

// Encodes one row with one warp: lane = word scans every pattern.
template <KVCacheVqKeyCodec KeyCodec>
__device__ __forceinline__ void kv_cache_vq_append_row_warp(const KVCacheVqRowTarget& target,
                                                            const KVCacheVqCodebookShared& book,
                                                            KVCacheVqRowShared& row) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    kv_cache_vq_rotate_row(target.source, row, lane);
    __syncwarp();
    const float sigma = sqrtf(row.sum_square * (1.0f / 256.0f));
    if (KeyCodec == KVCacheVqKeyCodec::Q4 && !target.value) {
        const float dot = kv_cache_q4_encode_warp(row, lane, sigma);
        __syncwarp();
        kv_cache_vq_store_row<kKVCacheQ4CodeBytes / 4>(row, lane, dot, target);
        return;
    }
    float A[8];
    unsigned negative = 0;
    kv_cache_vq2_word_magnitudes(row, lane, sigma, A, negative);
    const bool odd = (__popc(negative) & 1) != 0;
    float best     = 0.0f;
    int best_p     = 0;
    kv_cache_vq2_scan<kKVCacheVq2Patterns>(book, A, odd, 0, best, best_p);
    const float dot = kv_cache_vq2_finish_word(book, row, lane, sigma, A, negative, best_p);
    __syncwarp();
    kv_cache_vq_store_row<kKVCacheVq2CodeBytes / 4>(row, lane, dot, target);
}

// Item x of an append: K (x even) or V (x odd) of unit x / 2 = token * KVHeads + kv_head. Window
// copies follow `Mode`: Slots writes positions in the row's final kKVWindowRingTokens and the sink;
// Staging writes every column's staging row.
template <typename Geometry, typename Metadata, KVCacheVqKeyCodec KeyCodec, KVCacheVqWindowMode Mode>
__device__ __forceinline__ KVCacheVqRowTarget kv_cache_append_vq_target(
    int item, int batch, int tokens, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const std::int32_t* __restrict__ positions,
    const Metadata& metadata, std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    __half* __restrict__ scale_k, __half* __restrict__ scale_v, const KVCacheVqWindowTarget& window,
    std::int32_t width) {
    constexpr int KeyBytes = kKVCacheVqKeyCodeBytes<KeyCodec>;
    const int unit     = item >> 1;
    const bool value   = (item & 1) != 0;
    const int kv_head  = unit % Geometry::KVHeads;
    const int token    = unit / Geometry::KVHeads;
    const int position = positions[0] + token;
    const int page     = paged_kv_physical_page(metadata.block_table(), position);
    const int offset   = position & kPagedKVPageMask;
    const std::int64_t source =
        (static_cast<std::int64_t>(token) * Geometry::KVHeads + kv_head) * kD256KVCacheHeadDim;

    KVCacheVqRowTarget target{};
    target.position = position;
    target.value    = value;
    if (value) {
        target.source = v + source;
        target.codes  = cache_v + kv_cache_vq_code_index<kKVCacheVq2CodeBytes>(
                                     Geometry::KVHeads, page, kv_head, offset);
        target.scale = scale_v + kv_cache_vq_scale_index(Geometry::KVHeads, page, kv_head, offset);
    } else {
        target.source = k + source;
        target.codes =
            cache_k + kv_cache_vq_code_index<KeyBytes>(Geometry::KVHeads, page, kv_head, offset);
        target.scale = scale_k + kv_cache_vq_scale_index(Geometry::KVHeads, page, kv_head, offset);
    }
    if constexpr (Mode == KVCacheVqWindowMode::Slots) {
        const int last = positions[0] + tokens - 1;
        if (position < kKVWindowSinkTokens || position > last - kKVWindowRingTokens) {
            const int row  = window.slots == nullptr ? 0 : window.slots[batch];
            const int slot = kv_window_slot(position);
            target.window_codes  = (value ? window.v_codes : window.k_codes) +
                                  kv_window_code_index<Geometry::KVHeads>(row, kv_head, slot, 0);
            target.window_scales = (value ? window.v_scales : window.k_scales) +
                                   kv_window_scale_index<Geometry::KVHeads>(row, kv_head, slot, 0);
            target.window_tag =
                window.tags + kv_window_tag_index<Geometry::KVHeads>(row, kv_head, slot, value);
        }
    } else if constexpr (Mode == KVCacheVqWindowMode::Staging) {
        // Staging rows [256, width, Hkv] per role, scales [4, width, Hkv], tags [2, width, Hkv].
        const std::int64_t staged = (static_cast<std::int64_t>(kv_head) * width + token) +
                                    static_cast<std::int64_t>(batch) * width * Geometry::KVHeads;
        target.window_codes =
            (value ? window.v_codes : window.k_codes) + staged * kD256KVCacheHeadDim;
        target.window_scales =
            (value ? window.v_scales : window.k_scales) + staged * kKVWindowGroups;
        target.window_tag = window.tags + staged * 2 + (value ? 1 : 0);
    }
    return target;
}

// Shifts the per-batch-row operands of an append to batch row `batch`. The pointer types are
// deduced so they keep the kernel parameters' __restrict__ (GCC binds no plain pointer reference
// to a restrict-qualified pointer).
template <typename Geometry, typename Metadata, bool MultiBatch, typename Values, typename Positions>
__device__ __forceinline__ void kv_cache_append_vq_select_batch(int batch, Metadata& metadata, Values& k,
                                                                Values& v, Positions& positions,
                                                                std::int32_t width) {
    if constexpr (MultiBatch) {
        metadata.table_rows += batch;
        if (metadata.valid_columns) metadata.valid_columns += batch;
        const auto offset = static_cast<std::int64_t>(batch) * width * 256 * Geometry::KVHeads;
        k += offset;
        v += offset;
        positions += batch * width;
    }
}

// Grid (blocks, 1, batch) of 1024-thread blocks: each block stages the codebook once, then encodes
// items blockIdx.x, blockIdx.x + gridDim.x, ... of its batch row with the whole block (the lowest
// latency per row: a decode step's few rows).
template <typename Geometry, typename Metadata, KVCacheVqKeyCodec KeyCodec, KVCacheVqWindowMode Mode,
          bool MultiBatch = false>
__launch_bounds__(kKVCacheVqAppendThreads) __global__ void kv_cache_append_vq_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    __half* __restrict__ scale_k, __half* __restrict__ scale_v, KVCacheVqWindowTarget window,
    std::int32_t width) {
    __shared__ KVCacheVqCodebookShared book;
    __shared__ KVCacheVqRowShared row;
    __shared__ KVCacheVqBlockShared block;
    // The attention that follows may launch beside this append: only its window CTAs read what
    // the append writes, and they wait for this grid (vq_kv_grouped_kernel).
    pdl::trigger_dependents();
    const int batch = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    kv_cache_append_vq_select_batch<Geometry, Metadata, MultiBatch>(batch, metadata, k, v,
                                                                   positions, width);
    const int tokens = metadata.valid_tokens(width);
    const int items  = 2 * tokens * Geometry::KVHeads;
    if (static_cast<int>(blockIdx.x) >= items) return;
    kv_cache_vq2_stage_encoder_codebook(book, static_cast<int>(threadIdx.x),
                                        static_cast<int>(blockDim.x));
    __syncthreads();
    for (int item = static_cast<int>(blockIdx.x); item < items;
         item += static_cast<int>(gridDim.x)) {
        const KVCacheVqRowTarget target =
            kv_cache_append_vq_target<Geometry, Metadata, KeyCodec, Mode>(
                item, batch, tokens, k, v, positions, metadata, cache_k, cache_v, scale_k, scale_v,
                window, width);
        kv_cache_vq_append_row_block<KeyCodec>(target, book, row, block);
        __syncthreads();
    }
}

// Warps per block of the row-per-warp append.
inline constexpr int kKVCacheVqAppendWarpRows = 8;

// Grid (blocks, 1, batch) of 8-warp blocks: each block stages the codebook once, then each warp
// encodes items blockIdx.x * 8 + warp, + gridDim.x * 8, ... by itself (the highest throughput: a
// wide call's many rows).
template <typename Geometry, typename Metadata, KVCacheVqKeyCodec KeyCodec, KVCacheVqWindowMode Mode,
          bool MultiBatch = false>
__launch_bounds__(32 * kKVCacheVqAppendWarpRows) __global__ void kv_cache_append_vq_warp_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    __half* __restrict__ scale_k, __half* __restrict__ scale_v, KVCacheVqWindowTarget window,
    std::int32_t width) {
    __shared__ KVCacheVqCodebookShared book;
    __shared__ KVCacheVqRowShared rows[kKVCacheVqAppendWarpRows];
    pdl::trigger_dependents();
    const int batch = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    kv_cache_append_vq_select_batch<Geometry, Metadata, MultiBatch>(batch, metadata, k, v,
                                                                   positions, width);
    const int tokens = metadata.valid_tokens(width);
    const int items  = 2 * tokens * Geometry::KVHeads;
    if (static_cast<int>(blockIdx.x) * kKVCacheVqAppendWarpRows >= items) return;
    kv_cache_vq2_stage_encoder_codebook(book, static_cast<int>(threadIdx.x),
                                        static_cast<int>(blockDim.x));
    __syncthreads();
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    for (int item = static_cast<int>(blockIdx.x) * kKVCacheVqAppendWarpRows + warp; item < items;
         item += static_cast<int>(gridDim.x) * kKVCacheVqAppendWarpRows) {
        const KVCacheVqRowTarget target =
            kv_cache_append_vq_target<Geometry, Metadata, KeyCodec, Mode>(
                item, batch, tokens, k, v, positions, metadata, cache_k, cache_v, scale_k, scale_v,
                window, width);
        kv_cache_vq_append_row_warp<KeyCodec>(target, book, rows[warp]);
        __syncwarp();
    }
}

} // namespace infernix::ops
