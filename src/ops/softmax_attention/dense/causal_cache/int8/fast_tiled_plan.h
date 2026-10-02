#pragma once

// Host plan of the fast INT8 prompt kernel: its CTA shape and key splits.

#include "ops/softmax_attention/dense/causal_cache/fast_prompt_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kInt8FastPromptMaxSplits = 8;

// Launch time in units of one eight-warp CTA's sweep over one visible key (about 59 ns on RTX
// 5090). One CTA fits an SM and the last row block sweeps every visible key, so a launch takes
// (waves) x (visible keys / splits). A four-warp CTA sweeps in 80 % of that time but covers half
// the rows, and runs unsplit. A split launch also stores every split's FP32 row and merges them:
// half a key's sweep per column and split for 24 query heads. Measured with every shape and split
// count 1-8 forced, at 257-1536 columns over 0-64K cached keys (24 query heads): this choice is
// within 0.14 % of the fastest on average and 3.8 % at worst; the shape and split count planned
// without the split cost were up to 54 % slower over a few thousand keys.
inline FastPromptPlan int8_fast_prompt_plan(std::int32_t q_heads, std::int32_t width,
                                            std::uint32_t max_visible_keys) {
    constexpr double kNarrowSweep      = 0.8;
    constexpr double kSplitPerColumn   = 0.5 / 24.0;
    const std::int64_t multiprocessors = fast_prompt_multiprocessors();
    const std::int32_t pages           = fast_prompt_pages(max_visible_keys);
    const double keys                  = static_cast<double>(max_visible_keys);

    const std::int64_t narrow_ctas = static_cast<std::int64_t>(div_up(width, 64)) * q_heads;
    FastPromptPlan best{4, 1};
    double best_cost =
        static_cast<double>(div_up(narrow_ctas, multiprocessors)) * kNarrowSweep * keys;
    const std::int64_t ctas = static_cast<std::int64_t>(div_up(width, 128)) * q_heads;
    for (std::int32_t splits = 1; splits <= kInt8FastPromptMaxSplits; ++splits) {
        if (!fast_prompt_split_admissible(q_heads, width, pages, splits)) break;
        double cost = static_cast<double>(div_up(ctas * splits, multiprocessors)) * keys / splits;
        if (splits > 1) cost += kSplitPerColumn * width * splits * q_heads;
        if (cost < best_cost) {
            best_cost = cost;
            best      = {8, splits};
        }
    }
    return best;
}

// Transient bytes of the widest fast prompt launch among widths [first, last]. Every candidate's
// cost is linear in the key count, and its constant term (the split cost) grows with the split
// count, so fewer keys never select more splits: the plan at max_visible_keys bounds every launch.
inline std::size_t int8_fast_prompt_workspace_bytes(std::int32_t q_heads, std::int32_t first,
                                                    std::int32_t last,
                                                    std::uint32_t max_visible_keys) {
    std::size_t maximum = 0;
    for (std::int32_t width = first; width <= last; ++width) {
        const FastPromptPlan plan = int8_fast_prompt_plan(q_heads, width, max_visible_keys);
        maximum = std::max(maximum, fast_prompt_split_bytes(q_heads, width, plan.splits));
    }
    return maximum;
}

} // namespace ninfer::ops::detail
