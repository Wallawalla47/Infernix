#pragma once

#include "ops/softmax_attention/common/causal_operands.h"
#include "ops/softmax_attention/common/causal_partition.h"
#include <algorithm>
#include <cstddef>

namespace ninfer::ops::detail {

inline constexpr int kMxfp8TiledQueryRows = 128;
inline constexpr int kMxfp8TiledMaxSplits = 8;

// Minimize waves per KV partition, retaining fewer partitions on a tie, with every split's FP32
// partial rows within split_budget (CausalAttentionExecutionEnvelope::prompt_split_workspace_bytes;
// one split's are always written, for the merge). Live rows cap the count at
// ceil(visible_keys / 512), and mxfp8_tiled_active_splits picks the launch's count below it. Count
// changes work and storage, never kernel topology. The partition carries the device's SM count for
// that launch-time choice.
inline CausalKvPartition mxfp8_tiled_partition(int heads, int width, int visible_capacity,
                                               int multiprocessor_count, std::size_t split_budget) {
    const std::int64_t tiles =
        (static_cast<std::int64_t>(width) + kMxfp8TiledQueryRows - 1) / kMxfp8TiledQueryRows;
    const std::int64_t ctas = heads * tiles;
    int selected            = 1;
    auto waves              = (ctas + multiprocessor_count - 1) / multiprocessor_count;
    for (int splits = 2; splits <= kMxfp8TiledMaxSplits; ++splits) {
        const auto next = (ctas * splits + multiprocessor_count - 1) / multiprocessor_count;
        if (next * selected < waves * splits) {
            selected = splits;
            waves    = next;
        }
    }
    while (selected > 1) {
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, selected, 1);
        if (layout.peak_bytes(1) <= split_budget) break;
        --selected;
    }
    CausalKvPartition partition{1, selected, 9};
    partition.capacity        = partition.active(visible_capacity);
    partition.multiprocessors = multiprocessor_count;
    return partition;
}

// Splits a launch over `visible` keys runs: the cheapest count up to partition.active(visible).
// One 128-row CTA fits an SM and the last row block sweeps every visible key, so a launch takes
// (waves) x (visible keys / splits); each split beyond the first also stores every column's FP32
// row and merges it, about half a key's sweep per column and split for 24 query heads (the cost
// the fast INT8 prompt kernel's plan measured). The kernel and the merge both call this, and fewer
// splits than the partition's capacity only leave grid slices idle.
__host__ __device__ inline int mxfp8_tiled_active_splits(const CausalKvPartition& partition,
                                                         int visible, int width, int heads) {
    const int bound = partition.active(visible);
    const int sms   = partition.multiprocessors > 0 ? partition.multiprocessors : 1;
    const int ctas  = ((width + kMxfp8TiledQueryRows - 1) / kMxfp8TiledQueryRows) * heads;
    int best        = 1;
    float best_cost = 0.0F;
    for (int splits = 1; splits <= bound; ++splits) {
        const int waves = (ctas * splits + sms - 1) / sms;
        float cost =
            static_cast<float>(waves) * static_cast<float>(visible) / static_cast<float>(splits);
        if (splits > 1) cost += (0.5F / 24.0F) * static_cast<float>(width * splits * heads);
        if (splits == 1 || cost < best_cost) {
            best      = splits;
            best_cost = cost;
        }
    }
    return best;
}

// Every width is checked: the split budget can lower the split target inside a query-tile
// interval, so the largest allocation need not sit at an interval's last width.
inline std::size_t mxfp8_tiled_workspace_bytes(int heads, int min_width, int max_width,
                                               int visible_capacity, int multiprocessor_count,
                                               std::size_t split_budget) {
    std::size_t maximum = 0;
    for (int width = std::max(min_width, 17); width <= max_width; ++width) {
        const auto partition = mxfp8_tiled_partition(heads, width, visible_capacity,
                                                     multiprocessor_count, split_budget);
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, partition.capacity, 1);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

} // namespace ninfer::ops::detail
