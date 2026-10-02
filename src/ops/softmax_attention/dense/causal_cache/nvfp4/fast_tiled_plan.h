#pragma once

// Host plan of the fast NVFP4 prompt kernel: whether it applies, its CTA shape and key splits.

#include "ops/softmax_attention/dense/causal_cache/fast_prompt_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {

// Whether an NVFP4 prompt-route launch runs the fast kernel. It is 33-65 % faster than the tiled
// kernel from 4K visible keys (RTX 5090, 256-4096 columns, 4K-128K keys) and at a 3584-column first
// chunk, but slower for short launches over few keys (1024 columns over none: 185 against 127 us).
inline bool nvfp4_fast_prompt_applies(std::uint32_t max_visible_keys) {
    return max_visible_keys > 2048;
}

// Every CTA of a launch sweeps about the same key range and one CTA fits an SM, so a launch costs
// about (waves) x (one CTA's sweep); a split CTA sweeps 1/splits of it. A four-warp CTA sweeps in
// 72 % of an eight-warp CTA's time (RTX 5090, 4096 columns over 128K keys) but covers half the
// rows.
inline FastPromptPlan nvfp4_fast_prompt_plan(std::int32_t q_heads, std::int32_t width,
                                             std::uint32_t max_visible_keys) {
    constexpr std::int32_t kMaxSplits        = 16;
    constexpr std::int64_t kSplitCostPercent = 1;
    struct Cta {
        std::int32_t warps, rows;
        std::int64_t percent;
    };

    constexpr Cta kCtas[]              = {{8, 128, 100}, {4, 64, 72}};
    const std::int64_t multiprocessors = fast_prompt_multiprocessors();
    const std::int32_t pages           = fast_prompt_pages(max_visible_keys);
    FastPromptPlan best{};
    std::int64_t best_cost = std::numeric_limits<std::int64_t>::max();
    for (const Cta& cta : kCtas) {
        const std::int64_t ctas = static_cast<std::int64_t>(div_up(width, cta.rows)) * q_heads;
        for (std::int32_t splits = 1; splits <= kMaxSplits; ++splits) {
            if (!fast_prompt_split_admissible(q_heads, width, pages, splits)) break;
            const std::int64_t waves = div_up(ctas * splits, multiprocessors);
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

// Transient bytes of the widest fast prompt launch among widths [first, last]. The plan's cost
// does not depend on the key count, and fewer keys only remove split counts, so the plan at
// max_visible_keys bounds every launch.
inline std::size_t nvfp4_fast_prompt_workspace_bytes(std::int32_t q_heads, std::int32_t first,
                                                     std::int32_t last,
                                                     std::uint32_t max_visible_keys) {
    std::size_t maximum = 0;
    for (std::int32_t width = first; width <= last; ++width) {
        const FastPromptPlan plan = nvfp4_fast_prompt_plan(q_heads, width, max_visible_keys);
        maximum = std::max(maximum, fast_prompt_split_bytes(q_heads, width, plan.splits));
    }
    return maximum;
}

} // namespace ninfer::ops::detail
