#pragma once

#include "core/paged_kv_cache.h"
#include "ops/kv_cache/append/vq2_kernel.cuh"
#include "ops/softmax_attention/common/causal_operands.h"

#include <cuda_fp16.h>

#include <cstdint>

namespace infernix::ops::detail {

// Read view of a vector-quantized cache (Vq2, Q4KeyVq2Value) with its exact recent-key window.
// Window planes are indexed by sequence state slot (window_slots[b], or 0 for a single view).
struct VqKvCacheView {
    const std::uint8_t* key_codes;
    const std::uint8_t* value_codes;
    const __half* key_scales;
    const __half* value_scales;
    const std::int32_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    int table_stride, kv_heads;
    const std::int8_t* window_k;
    const std::int8_t* window_v;
    const __half* window_k_scales;
    const __half* window_v_scales;
    const std::int32_t* window_tags;
    // Window slot of each batch row (null: a single-sequence view, slot 0).
    const std::int32_t* window_slots;
    // Ancestor masks of speculative verification trees, [B][W] (see QuantizedCausalCacheView).
    const std::uint32_t* tree_masks = nullptr;
};

inline VqKvCacheView make_vq_cache_view(const PagedKVBatchLayerView& cache,
                                        const Tensor* valid = nullptr,
                                        const Tensor* rows  = nullptr) {
    return {static_cast<const std::uint8_t*>(cache.k_pages.data),
            static_cast<const std::uint8_t*>(cache.v_pages.data),
            static_cast<const __half*>(cache.k_scale_pages.data),
            static_cast<const __half*>(cache.v_scale_pages.data),
            static_cast<const std::int32_t*>(cache.block_tables.data),
            valid ? static_cast<const std::int32_t*>(valid->data) : nullptr,
            rows ? static_cast<const std::int32_t*>(rows->data) : nullptr,
            cache.block_tables.ne[0],
            cache.num_kv_heads,
            static_cast<const std::int8_t*>(cache.window.k_codes.data),
            static_cast<const std::int8_t*>(cache.window.v_codes.data),
            static_cast<const __half*>(cache.window.k_scales.data),
            static_cast<const __half*>(cache.window.v_scales.data),
            static_cast<const std::int32_t*>(cache.window.tags.data),
            static_cast<const std::int32_t*>(cache.window.slots.data)};
}

template <class G>
void validate_vq_causal_operands(const CausalAttentionOperands& p, const VqKvCacheView& cache) {
    static_assert(G::kHeadDim == kCausalHeadDim, "VQ causal templates require D256");
    if (p.query_heads != G::QHeads || cache.kv_heads != G::KVHeads || !p.q || !p.positions ||
        !p.out || !cache.key_codes || !cache.value_codes || !cache.key_scales ||
        !cache.value_scales || !cache.tables || p.width < 1 || p.batch < 1 ||
        p.visible_capacity < 1 ||
        static_cast<std::int64_t>(p.visible_capacity) >
            static_cast<std::int64_t>(cache.table_stride) * kPagedKVPageSize ||
        (cache.window_tags != nullptr &&
         (!cache.window_k || !cache.window_v || !cache.window_k_scales || !cache.window_v_scales)))
        throw std::invalid_argument("VQ causal attention: invalid operands");
}

} // namespace infernix::ops::detail
