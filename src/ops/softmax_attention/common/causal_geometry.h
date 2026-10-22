#pragma once
#include "ops/softmax_attention/common/head_mapping.cuh"

namespace infernix::ops::detail {
inline constexpr int kCausalHeadDim = 256;

template <int HeadDim, int QueryHeads, int KVHeads>
struct CausalGeometry : AttentionHeadMapping<QueryHeads, KVHeads> {
    static_assert(HeadDim > 0 && HeadDim % 64 == 0);
    static constexpr int kHeadDim = HeadDim;
};

// The registered head geometries: Qwen3.5-27B (24/4), the 35B-A3B (16/2) and the 122B-A10B (32/2,
// GQA group 16).
using CausalD256H24Kv4 = CausalGeometry<256, 24, 4>;
using CausalD256H16Kv2 = CausalGeometry<256, 16, 2>;
using CausalD256H32Kv2 = CausalGeometry<256, 32, 2>;

[[nodiscard]] constexpr bool causal_registered_heads(int query_heads) noexcept {
    return query_heads == 24 || query_heads == 16 || query_heads == 32;
}
// The KV heads of a registered geometry's query heads.
[[nodiscard]] constexpr int causal_kv_heads(int query_heads) noexcept { return query_heads == 24 ? 4 : 2; }
[[nodiscard]] constexpr int causal_group_size(int query_heads) noexcept {
    return query_heads / causal_kv_heads(query_heads);
}

// The columns of one grouped CTA of the quantized KV formats: a CTA packs at most 64 query rows
// (column, head pairs of one KV head) and at most eight columns, so the 122B's group 16 holds four.
template <class G>
inline constexpr int kCausalGroupedTokens = 64 / G::GroupSize < 8 ? 64 / G::GroupSize : 8;
[[nodiscard]] constexpr int causal_grouped_tokens(int query_heads) noexcept {
    const int group = causal_group_size(query_heads);
    return 64 / group < 8 ? 64 / group : 8;
}

// Calls f.template operator()<G>() with the registered geometry of `query_heads`, which the caller
// has checked (causal_registered_heads).
template <class F>
decltype(auto) with_causal_geometry(int query_heads, F&& f) {
    if (query_heads == 24) { return f.template operator()<CausalD256H24Kv4>(); }
    if (query_heads == 32) { return f.template operator()<CausalD256H32Kv2>(); }
    return f.template operator()<CausalD256H16Kv2>();
}
} // namespace infernix::ops::detail
