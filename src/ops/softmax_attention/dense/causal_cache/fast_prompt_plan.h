#pragma once

// Host side of the fast INT8 prompt kernel's key splits: the launch shape, the FP32 partials a
// split launch publishes, their budget and the workspace they need. The kernel uses 128- and
// 64-row CTAs of one query head at one CTA per SM, and a split launch merges its partial rows with
// causal_attention_prompt_fast_merge_kernel (fast_prompt_common.cuh); int8/fast_tiled_plan.h
// chooses the shape and split count.

#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/common/causal_partition.h"

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

// A fast prompt launch: warps per CTA and number of key splits. More than one split divides every
// row block's key pages among CTAs and merges their FP32 partial rows, so a launch whose row blocks
// alone would leave SMs idle still fills them.
struct FastPromptPlan {
    std::int32_t warps  = 8;
    std::int32_t splits = 1;
};

// A split launch's partials: the normalized FP32 row of every (column, query head, split) and its
// (max, sum) statistics.
struct FastPromptPartials {
    Tensor rows;
    Tensor stats;
};

template <class Allocator>
FastPromptPartials allocate_fast_prompt_partials(Allocator& workspace, std::int32_t q_heads,
                                                 std::int32_t width, std::int32_t splits) {
    return {workspace.alloc(DType::FP32, {256, q_heads, width, splits}),
            workspace.alloc(DType::FP32, {2, q_heads, width, splits})};
}

inline std::size_t fast_prompt_split_bytes(std::int32_t q_heads, std::int32_t width,
                                           std::int32_t splits) {
    if (splits <= 1) return 0;
    WorkspaceLayoutBuilder layout;
    (void)allocate_fast_prompt_partials(layout, q_heads, width, splits);
    return layout.peak_bytes(1);
}

// Split partials stay within the launch's split workspace
// (CausalAttentionExecutionEnvelope::prompt_split_workspace_bytes), and every split keeps at least
// eight 64-key pages.
inline constexpr std::int32_t kFastPromptMinPagesPerSplit = 8;

inline std::int32_t fast_prompt_pages(std::uint32_t visible_keys) {
    return static_cast<std::int32_t>(
        (static_cast<std::uint64_t>(visible_keys) + kPagedKVPageSize - 1) / kPagedKVPageSize);
}

// Whether a launch of this width over this many key pages may use this many splits.
inline bool fast_prompt_split_admissible(std::int32_t q_heads, std::int32_t width,
                                         std::int32_t pages, std::int32_t splits,
                                         std::size_t split_budget) {
    return splits <= 1 || (pages >= splits * kFastPromptMinPagesPerSplit &&
                           fast_prompt_split_bytes(q_heads, width, splits) <= split_budget);
}

inline int fast_prompt_multiprocessors() {
    static const int count = [] {
        int device = 0;
        int value  = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device));
        return value;
    }();
    return count;
}

} // namespace infernix::ops::detail
