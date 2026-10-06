#pragma once

// The exact-window read rule of QSA's kernels for the vector-quantized KV storages (vq2, k4v2;
// core/paged_kv_storage.h, ops/kv_cache/kv_window.cuh), shared by the decode and prompt kernels.

#include "core/paged_kv_storage.h"
#include "ops/kv_cache/kv_window.cuh"
#include "ops/qsa/qsa_prompt.h"

#include <cstdint>

namespace ninfer::ops::detail {

// How a query at position `query` of a call starting at `first` reads key `key`: 0 from its codes,
// 1 from its window slot (when the slot's tag matches the stored codes), 2 from the call's staged
// row (a wide call's own columns).
__device__ __forceinline__ int vq_read_mode(const QsaVqWindow& w, int key, int query, int first) {
    if (w.tags == nullptr || !(key < kKVWindowSinkTokens || key >= query - kKVWindowRecentTokens)) { return 0; }
    return w.staged_width > 0 && key >= first ? 2 : 1;
}

// Row index of (window row, kv_head, slot) in the window planes (kv_window.cuh's indices for a
// runtime head count): codes at row * 256, group scales at row * 4, tags at row * 2 + role.
__device__ __forceinline__ std::int64_t vq_window_row(int row, int kv_head, int kv_heads, int slot) {
    return (static_cast<std::int64_t>(row) * kv_heads + kv_head) * kKVWindowSlots + slot;
}

// Row index of a staged call column (staging [256, width, Hkv]: kv_head * width + column).
__device__ __forceinline__ std::int64_t vq_staged_row(const QsaVqWindow& w, int kv_head, int column) {
    return static_cast<std::int64_t>(kv_head) * w.staged_width + column;
}

} // namespace ninfer::ops::detail
