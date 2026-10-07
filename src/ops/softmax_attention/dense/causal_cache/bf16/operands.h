#pragma once
#include "ops/softmax_attention/common/causal_operands.h"

namespace infernix::ops::detail {
template <bool Writable>
struct Bf16KvCacheView {
    static constexpr bool kWritable = Writable;
    using Key   = std::conditional_t<Writable, __nv_bfloat16, const __nv_bfloat16>;
    using Value = std::conditional_t<Writable, __half, const __half>;
    Key* keys;
    Value* values;
    const std::int32_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    int table_stride;
    int head_dim;
    int kv_heads;
    // Optional per-row ancestor masks of speculative verification trees ([B][W]; see
    // causal_softmax_attention).
    const std::uint32_t* tree_masks = nullptr;
};

using Bf16KvReadView  = Bf16KvCacheView<false>;
using Bf16KvWriteView = Bf16KvCacheView<true>;

template <bool Writable>
Bf16KvCacheView<Writable> bf16_kv_cache_view(const PagedKVBatchLayerView& cache,
                                             const Tensor* valid = nullptr,
                                             const Tensor* rows  = nullptr) {
    return {static_cast<typename Bf16KvCacheView<Writable>::Key*>(cache.k_pages.data),
            static_cast<typename Bf16KvCacheView<Writable>::Value*>(cache.v_pages.data),
            static_cast<const std::int32_t*>(cache.block_tables.data),
            valid ? static_cast<const std::int32_t*>(valid->data) : nullptr,
            rows ? static_cast<const std::int32_t*>(rows->data) : nullptr,
            cache.block_tables.ne[0],
            cache.head_dim,
            cache.num_kv_heads};
}

} // namespace infernix::ops::detail
