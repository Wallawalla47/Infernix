#pragma once

// Host plan of the fast NVFP4 prompt kernel: whether it applies, its CTA shape and key splits.

#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "ops/common/math.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {

// A fast prompt launch: warps per CTA and number of key splits. More than one split divides every
// row block's key pages among CTAs and merges their FP32 partial rows, so a launch whose row blocks
// alone would leave SMs idle still fills them.
struct RotatedFastPromptPlan {
    std::int32_t warps  = 8;
    std::int32_t splits = 1;
};

// A split launch's partials: the normalized FP32 row of every (column, query head, split) and its
// (max, sum) statistics.
struct RotatedFastPromptPartials {
    Tensor rows;
    Tensor stats;
};

template <class Allocator>
RotatedFastPromptPartials allocate_rotated_fast_prompt_partials(Allocator& workspace,
                                                                std::int32_t q_heads,
                                                                std::int32_t width,
                                                                std::int32_t splits) {
    return {workspace.alloc(DType::FP32, {256, q_heads, width, splits}),
            workspace.alloc(DType::FP32, {2, q_heads, width, splits})};
}

inline std::size_t rotated_fast_prompt_split_bytes(std::int32_t q_heads, std::int32_t width,
                                                   std::int32_t splits) {
    if (splits <= 1) return 0;
    WorkspaceLayoutBuilder layout;
    (void)allocate_rotated_fast_prompt_partials(layout, q_heads, width, splits);
    return layout.peak_bytes(1);
}

// Every CTA of a launch sweeps about the same key range and one CTA fits an SM, so a launch costs
// about (waves) x (one CTA's sweep); a split CTA sweeps 1/splits of it. A four-warp CTA sweeps in
// 72 % of an eight-warp CTA's time (RTX 5090, 4096 columns over 128K keys) but covers half the
// rows. Split partials may use at most 64 MiB, which the workspace plan reserves for the widest
// launch.
inline RotatedFastPromptPlan rotated_fast_prompt_plan(std::int32_t q_heads, std::int32_t width,
                                                      std::uint32_t max_visible_keys) {
    constexpr std::size_t kSplitBudgetBytes  = std::size_t{64} << 20;
    constexpr std::int32_t kMinPagesPerSplit = 8;
    constexpr std::int32_t kMaxSplits        = 16;
    constexpr std::int64_t kSplitCostPercent = 1;
    struct Cta {
        std::int32_t warps, rows;
        std::int64_t percent;
    };
    constexpr Cta kCtas[] = {{8, 128, 100}, {4, 64, 72}};
    static const int multiprocessors = [] {
        int device = 0;
        int count  = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
        return count;
    }();
    const auto pages = static_cast<std::int32_t>(
        (static_cast<std::uint64_t>(max_visible_keys) + kPagedKVPageSize - 1) / kPagedKVPageSize);
    RotatedFastPromptPlan best{};
    std::int64_t best_cost = std::numeric_limits<std::int64_t>::max();
    for (const Cta& cta : kCtas) {
        const std::int32_t ctas = div_up(width, cta.rows) * q_heads;
        for (std::int32_t splits = 1; splits <= kMaxSplits; ++splits) {
            if (splits > 1 && (pages < splits * kMinPagesPerSplit ||
                               rotated_fast_prompt_split_bytes(q_heads, width, splits) >
                                   kSplitBudgetBytes))
                break;
            const std::int64_t waves = div_up(static_cast<std::int64_t>(ctas) * splits,
                                              static_cast<std::int64_t>(multiprocessors));
            // Scaled by kMaxSplits so every split count divides exactly.
            const std::int64_t cost = waves * cta.percent * kMaxSplits / splits +
                                      (splits - 1) * kSplitCostPercent * kMaxSplits;
            if (cost < best_cost) {
                best_cost = cost;
                best      = {cta.warps, splits};
            }
        }
    }
    return best;
}

// Whether an NVFP4 prompt-route launch runs the fast kernel. It is 33-65 % faster than the tiled
// kernel from 4K visible keys (RTX 5090, 256-4096 columns, 4K-128K keys) and at a 3584-column first
// chunk, but slower for short launches over few keys (1024 columns over none: 185 against 127 us).
inline bool nvfp4_fast_prompt_applies(std::uint32_t max_visible_keys) {
    return max_visible_keys > 2048;
}

// Transient bytes of the widest fast prompt launch among widths [first, last].
inline std::size_t rotated_fast_prompt_workspace_bytes(std::int32_t q_heads, std::int32_t first,
                                                       std::int32_t last,
                                                       std::uint32_t max_visible_keys) {
    std::size_t maximum = 0;
    for (std::int32_t width = first; width <= last; ++width) {
        const RotatedFastPromptPlan plan = rotated_fast_prompt_plan(q_heads, width, max_visible_keys);
        maximum = std::max(maximum, rotated_fast_prompt_split_bytes(q_heads, width, plan.splits));
    }
    return maximum;
}

} // namespace ninfer::ops::detail
