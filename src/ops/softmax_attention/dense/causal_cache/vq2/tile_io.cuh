#pragma once

// Tile helpers of the vector-quantized cache kernels. Every kernel stages one tile's persistent
// code rows and row scales in shared memory, expands them into the INT8-G64 tile layout of the
// INT8 kernels (codes widened from the 2-bit codebook or the 4-bit levels, the row scale repeated
// as every group scale) and takes keys of the exact window from their INT8 window slots instead
// when the slot tag matches the staged codes.

#include "ops/kv_cache/append/vq2_kernel.cuh"
#include "ops/kv_cache/kv_window.cuh"
#include "ops/kv_cache/q4_lloyd_codec.cuh"
#include "ops/kv_cache/vq2_codec.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <KVCacheVqKeyCodec KeyCodec>
inline constexpr int kVqKeyCodeBytes = kKVCacheVqKeyCodeBytes<KeyCodec>;
inline constexpr int kVqValueCodeBytes = kKVCacheVq2CodeBytes;

// Copies the 4 KiB INT8 codebook into shared memory (16-byte aligned).
__device__ __forceinline__ void vq_stage_codebook(std::int8_t* codebook, int thread, int threads) {
    const auto* source = reinterpret_cast<const int4*>(g_kv_cache_vq2_codebook);
    auto* destination  = reinterpret_cast<int4*>(codebook);
    for (int i = thread; i < kKVCacheVq2Patterns * 8 / 16; i += threads)
        destination[i] = source[i];
}

// Eight INT8 coordinates [8w, 8w + 8) of one staged key row.
template <KVCacheVqKeyCodec KeyCodec>
__device__ __forceinline__ uint2 vq_decode_key_word(const std::uint8_t* row, int word,
                                                    const std::int8_t* codebook) {
    if constexpr (KeyCodec == KVCacheVqKeyCodec::Q4) {
        return kv_cache_q4_decode_word(reinterpret_cast<const std::uint32_t*>(row)[word]);
    } else {
        return kv_cache_vq2_decode_word(reinterpret_cast<const std::uint16_t*>(row)[word],
                                        codebook);
    }
}

__device__ __forceinline__ uint2 vq_decode_value_word(const std::uint8_t* row, int word,
                                                      const std::int8_t* codebook) {
    return kv_cache_vq2_decode_word(reinterpret_cast<const std::uint16_t*>(row)[word], codebook);
}

// The same words with the VQ2 sign masks read from a staged table (kv_cache_vq2_stage_sign_masks).
template <KVCacheVqKeyCodec KeyCodec>
__device__ __forceinline__ uint2 vq_decode_key_word(const std::uint8_t* row, int word,
                                                    const std::int8_t* codebook,
                                                    const uint2* sign_masks) {
    if constexpr (KeyCodec == KVCacheVqKeyCodec::Q4) {
        return kv_cache_q4_decode_word(reinterpret_cast<const std::uint32_t*>(row)[word]);
    } else {
        return kv_cache_vq2_decode_word_lut(reinterpret_cast<const std::uint16_t*>(row)[word],
                                            codebook, sign_masks);
    }
}

__device__ __forceinline__ uint2 vq_decode_value_word(const std::uint8_t* row, int word,
                                                      const std::int8_t* codebook,
                                                      const uint2* sign_masks) {
    return kv_cache_vq2_decode_word_lut(reinterpret_cast<const std::uint16_t*>(row)[word], codebook,
                                        sign_masks);
}

// The window tag a slot must carry to stand in for this staged row (kv_window.cuh).
template <int CodeBytes>
__device__ __forceinline__ std::uint32_t vq_row_tag(int position, const std::uint8_t* row,
                                                    __half scale) {
    std::uint32_t hash = kv_window_tag_begin(position);
#pragma unroll 8
    for (int i = 0; i < CodeBytes / 4; ++i)
        hash = kv_window_tag_step(hash, reinterpret_cast<const std::uint32_t*>(row)[i]);
    return kv_window_tag_finish(hash, static_cast<std::uint32_t>(__half_as_ushort(scale)));
}

// The read rule: query position `query` reads key `key` from its exact INT8-G64 row (a window slot,
// or the call's own staged row) when the key is a sink or lies within kKVWindowRecentTokens before
// the query, and from its stored codes otherwise. It depends on positions only, never on how a
// sequence was split into calls.
__device__ __forceinline__ bool vq_exact_key(int key, int query) {
    return key < kKVWindowSinkTokens || key >= query - kKVWindowRecentTokens;
}

// For a CTA whose queries span positions [first, last]: keys outside [lo, hi) are read the same way
// by every query (exactly from kKVWindowRecentTokens before `first`, from codes below that), keys
// in the band [lo, hi) exactly by some queries and from codes by others. Without a window every
// key reads its codes and the band is empty.
struct VqBand {
    int lo;
    int hi;
};

__device__ __forceinline__ VqBand vq_band(int first, int last, bool window) {
    if (!window) return {0, 0};
    const int lo = max(kKVWindowSinkTokens, first - kKVWindowRecentTokens);
    return {lo, max(lo, last - kKVWindowRecentTokens)};
}

// A tile's first pass reads band keys exactly and keys outside the band as every query does; a
// tile meeting the band takes a second pass over its band keys' codes. Each (key, query) pair is
// kept in exactly one pass.
__device__ __forceinline__ bool vq_pass_keeps(int key, int query, VqBand band, bool band_codes) {
    if (key < band.lo || key >= band.hi) return !band_codes;
    return band_codes != vq_exact_key(key, query);
}

// Pass v of a key-block sweep whose blocks [band_first, band_first + band_blocks) meet the band:
// every block once, each band block a second time (its codes pass) right after its first pass.
struct VqTilePass {
    int block;
    bool band_codes;
};

__device__ __forceinline__ VqTilePass vq_tile_pass(int v, int band_first, int band_blocks) {
    if (v < band_first) return {v, false};
    const int d = v - band_first;
    if (d < 2 * band_blocks) return {band_first + (d >> 1), (d & 1) != 0};
    return {v - band_blocks, false};
}

} // namespace ninfer::ops::detail
