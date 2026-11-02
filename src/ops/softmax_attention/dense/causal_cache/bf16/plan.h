#pragma once

#include "infernix/ops/softmax_attention.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/instances.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/split_policy.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

struct Bf16KvCausalPlan {
    Bf16KvInstance instance;
    Bf16KvPartition partition;
    int query_heads;
    int width;
    int batch;
    CausalAttentionExecutionEnvelope envelope;
    // A tiled (prompt-route) plan's key splits per query tile: 1, or more under
    // envelope.bf16_prompt_split (bf16_kv_prompt_splits).
    int prompt_splits = 1;

    bool grouped() const { return bf16_kv_instance_description(instance).grouped; }
};

// The bf16-prompt-split count of a tiled launch: of 1-4 equal key shares per query tile, the one
// whose CTAs fill the last wave of the SMs best (a larger count only when it fills at least 3 % more),
// with every share at least 1,024 keys and its partials within the prompt split workspace.
int bf16_kv_prompt_splits(int query_heads, int width, int query_rows, std::uint32_t visible_keys,
                          std::size_t workspace_bytes, int multiprocessor_count);

Bf16KvCausalPlan make_bf16_kv_causal_plan(int query_heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope,
                                          int multiprocessor_count);
std::size_t bf16_kv_workspace_bytes(int query_heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope,
                                    int multiprocessor_count);

// The packed storage's prompt route (launch.cu): its keys unpacked kBf16pSegmentKeys at a time into a
// scratch of the bf16 storage's rows, which the bf16 tiled kernel reads, its CTAs' state suspended
// between segments (Bf16KvSegment).
inline constexpr int kBf16pSegmentKeys = 16384;

struct Bf16pPromptScratch {
    std::uint8_t* keys   = nullptr; // [pages][kv heads][64][256] BF16 (the segment's pages)
    std::uint8_t* values = nullptr; // the same, FP16
    std::int32_t* table  = nullptr; // logical page -> scratch page
    std::int32_t* zero   = nullptr; // table row 0 (masked calls)
    float* state         = nullptr; // suspended CTAs' state (more than one segment)
    int table_pages      = 0;
    int segments         = 0;
};

// The scratch of a prompt call of `width` columns over `keys` keys (Allocator: WorkspaceArena or
// WorkspaceLayoutBuilder): the state of every CTA of a tiled launch with `query_rows`-row tiles
// and `splits` key splits, each thread's D/8 x 4 accumulators and two rows' softmax state.
template <class Allocator>
Bf16pPromptScratch allocate_bf16p_prompt_scratch(Allocator& allocator, int query_heads, int kv_heads, int width,
                                                 int keys, int splits, int query_rows) {
    Bf16pPromptScratch out;
    const int pages            = (std::min(keys, kBf16pSegmentKeys) + 63) / 64;
    const std::size_t plane    = static_cast<std::size_t>(pages) * kv_heads * 64 * 512;
    out.keys                   = static_cast<std::uint8_t*>(allocator.alloc_bytes(plane).data);
    out.values                 = static_cast<std::uint8_t*>(allocator.alloc_bytes(plane).data);
    out.table_pages            = (keys + 63) / 64;
    out.table                  = static_cast<std::int32_t*>(allocator.alloc_bytes(out.table_pages * 4).data);
    out.zero                   = static_cast<std::int32_t*>(allocator.alloc_bytes(4).data);
    out.segments               = (keys + kBf16pSegmentKeys - 1) / kBf16pSegmentKeys;
    if (out.segments > 1) {
        const std::size_t threads = static_cast<std::size_t>(query_rows) / 16 * 32;
        const std::size_t ctas    = static_cast<std::size_t>((width + query_rows - 1) / query_rows) * query_heads * splits;
        const std::size_t fields  = kCausalHeadDim / 8 * 4 + 6;
        out.state                 = static_cast<float*>(allocator.alloc_bytes(ctas * fields * threads * 4).data);
    }
    return out;
}

// The packed storage's prompt-route workspace (batch 1, widths past the grouped route's).
std::size_t bf16p_kv_prompt_workspace_bytes(int query_heads, int kv_heads, int min_width, int max_width,
                                            CausalAttentionExecutionEnvelope envelope, int multiprocessor_count);

} // namespace infernix::ops::detail
