#pragma once

// Exact recent-key window of the vector-quantized KV formats (core/paged_kv_storage.h): slot
// addressing, the slot tag and the INT8-G64 row codec of a window slot.
//
// A slot of role r (0 = K, 1 = V) and position p holds INT8-G64 codes of the rotated row R*x
// (the same encoding as the INT8-G64 cache format, kv_cache_int8_quant_params/code) and the tag
//   kv_window_tag(p, persistent code words of that row, its FP16 scale bits)
// which is always odd. A reader recomputes the tag from the paged row it would otherwise decode
// and uses the slot only when both agree, so a slot left behind by another history, a rejected
// or compacted draft, or a trimmed tail is never read: no lifecycle event has to clear slots.

#include "core/paged_kv_storage.h"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ int kv_window_slot(int position) {
    return position < kKVWindowSinkTokens
               ? position
               : kKVWindowSinkTokens + (position & (kKVWindowRingTokens - 1));
}

// Element offset of window code d of (slot, kv_head, row) in a [256, slots, Hkv, rows] plane.
template <int KVHeads>
__device__ __forceinline__ std::int64_t kv_window_code_index(int row, int kv_head, int slot,
                                                             int d) {
    return (((static_cast<std::int64_t>(row) * KVHeads + kv_head) * kKVWindowSlots) + slot) *
               kD256KVCacheHeadDim +
           d;
}

template <int KVHeads>
__device__ __forceinline__ std::int64_t kv_window_scale_index(int row, int kv_head, int slot,
                                                              int group) {
    return (((static_cast<std::int64_t>(row) * KVHeads + kv_head) * kKVWindowSlots) + slot) *
               kKVWindowGroups +
           group;
}

template <int KVHeads>
__device__ __forceinline__ std::int64_t kv_window_tag_index(int row, int kv_head, int slot,
                                                            int role) {
    return (((static_cast<std::int64_t>(row) * KVHeads + kv_head) * kKVWindowSlots) + slot) * 2 +
           role;
}

// FNV-1a over 32-bit words, finalized and forced odd. `words` are the persistent code bytes of
// the row as little-endian u32 words; `scale_bits` is its FP16 scale.
__device__ __forceinline__ std::uint32_t kv_window_tag_begin(int position) {
    return (2166136261u ^ static_cast<std::uint32_t>(position)) * 16777619u;
}

__device__ __forceinline__ std::uint32_t kv_window_tag_step(std::uint32_t hash,
                                                            std::uint32_t word) {
    return (hash ^ word) * 16777619u;
}

__device__ __forceinline__ std::uint32_t kv_window_tag_finish(std::uint32_t hash,
                                                              std::uint32_t scale_bits) {
    hash = kv_window_tag_step(hash, scale_bits);
    hash ^= hash >> 15;
    hash *= 0x2c1b3c6du;
    hash ^= hash >> 12;
    return hash | 1u;
}

template <int Words>
__device__ __forceinline__ std::uint32_t kv_window_tag(int position, const std::uint32_t* words,
                                                       std::uint32_t scale_bits) {
    std::uint32_t hash = kv_window_tag_begin(position);
#pragma unroll
    for (int i = 0; i < Words; ++i) hash = kv_window_tag_step(hash, words[i]);
    return kv_window_tag_finish(hash, scale_bits);
}

} // namespace ninfer::ops
