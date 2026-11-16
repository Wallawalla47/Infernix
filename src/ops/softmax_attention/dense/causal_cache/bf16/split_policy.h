#pragma once

#include "ops/softmax_attention/common/causal_partition.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

struct Bf16KvLivePartition {
    int splits;
    int keys_per_split;
};

// Capture fixes capacity; producer and merge derive the same partition from
// the row window (causal_row_window), including ragged batches and Graph replay.
struct Bf16KvPartition {
    int capacity      = 1;
    int normal_target = 1;
    int long_target   = 1;
    int key_rows      = 32;
    // Nonzero (batch-coupled plans of more than two rows; the other formats' YD-02): the rows of a launch form
    // share groups of this many consecutive rows (a decode or verification call: all its rows; a column-invariant
    // round's rows launches: one column's lanes, so a column keeps its one-column call's partition), and the splits
    // a group's rows would run are shared between them in proportion to their windows, each live row at least one
    // (Bf16KvProportionalShares): one long row among short ones takes most of the CTAs instead of the same target
    // as each short row. Rows of equal windows keep live()'s count. A row's partials then start at its first slot
    // (the prefix of the earlier rows' counts), not at row x capacity.
    int proportional = 0;

    __host__ __device__ static int ceil_div(int n, int d) { return (n + d - 1) / d; }

    // The partition of `visible` keys into at most `target` splits of whole key tiles (at least 64 keys each).
    __host__ __device__ Bf16KvLivePartition live_with(int visible, int target) const {
        const int rounded = ceil_div(ceil_div(visible, target), key_rows) * key_rows;
        const int keys    = rounded < 64 ? 64 : rounded;
        return {ceil_div(visible, keys), keys};
    }

    __host__ __device__ Bf16KvLivePartition live(int visible) const {
        return live_with(visible, visible >= 32768 ? long_target : normal_target);
    }

    // Between key-tile rounding points, the split count grows with length. A
    // complete target*chunk boundary reaches target; otherwise high is maximal.
    int interval_capacity(int low, int high, int target) const {
        if (low > high) return 0;
        const int boundary = high / (target * key_rows) * (target * key_rows);
        if (boundary >= low && boundary >= target * 64) return target;
        return live(high).splits;
    }
};

// The most splits one row of a proportional launch takes: the merge's reduction block (Bf16KvMergeSchedule<256>).
inline constexpr int kBf16KvProportionalMaxSplits = 256;

// A proportional launch's shares, in share groups of partition.proportional consecutive rows (at most
// kCausalProportionalMaxRows rows of `width` columns; row r's positions at positions[r * width]; `validity` null, or
// each row's live columns). A row with no live column or a negative first position has window 0 and takes no split.
// A group's budget is the sum of its live rows' plain bounds, min(target of the window, one split per 64 keys,
// capacity): what live() runs at most, so the group runs about as many CTAs as the plain partition and never more
// than its capacity x rows slots. Each live row takes one split and the rest go to the group's rows in proportion to
// their windows (rounded down), a row never more than one split per 64 keys or the merge's block: that is its count,
// the slots it owns, and its partials start at its first slot (the prefix of the earlier rows' counts). It runs them
// as layout(window, count): equal splits of whole key tiles, the last slots left empty where the tile rounding needs
// fewer. Rows of equal windows have their plain bound as count and so live()'s splits, key for key (when the
// envelope's capacity is below the target, the capacity's). Every CTA of a launch (producer and merge alike) computes
// the same shares from the same positions, in integers.
//
// It runs warp-wide: lane r < kCausalProportionalMaxRows holds row r (its window and bound, its group's sums by
// shuffles, its count, and its first slot by a warp scan), so no array is kept (the eight rows' arrays, read by row
// indices a group-bounded loop chose, spilled into local memory). The rows' positions load together, and a group's
// shares divide by its sum of windows through one reciprocal. Every lane of the calling warp calls it with the same
// arguments. 32-bit arithmetic: a window is at most 2^20 keys and the spare slots at most 8 x 256, so
// spare x window < 2^32.
struct Bf16KvProportionalShares {
    // A row's splits within its `count` slots: live_with(window, count), the fewest whole key tiles a split (at
    // least 64 keys) that cover the window in `count` splits, and the splits that then hold keys (the tile rounding
    // can leave the last slots empty: their CTAs return, and the merge reads only the live ones). Only a CTA's own
    // row pays these divisions.
    __device__ static Bf16KvLivePartition layout(const Bf16KvPartition& partition, int window, int count) {
        return partition.live_with(window, count);
    }
};

#if defined(__CUDACC__)
// The calling lane's row's count and first slot, and every row's count.
__device__ inline CausalProportionalRow bf16_kv_proportional_lane(const Bf16KvPartition& partition,
                                                                  const std::int32_t* positions,
                                                                  const std::int32_t* validity, int width, int rows,
                                                                  int visible_capacity) {
    constexpr unsigned kFull = 0xffffffffU;
    const int r              = static_cast<int>(threadIdx.x & 31U);
    const int row            = r < rows ? r : 0; // a lane past the rows reads row 0 (and is given no window below)
    const int first          = positions[static_cast<std::int64_t>(row) * width];
    const int valid          = validity != nullptr ? validity[row] : 1;
    const bool works         = r < rows && r < kCausalProportionalMaxRows && valid > 0 && first >= 0;
    const int value          = works ? causal_row_window(first, width, visible_capacity) : 0;
    const unsigned window    = value > 0 ? static_cast<unsigned>(value) : 0U;
    int bound                = value >= 32768 ? partition.long_target : partition.normal_target;
    const int by_64          = (value + 63) >> 6;
    bound                    = bound < by_64 ? bound : by_64;
    bound                    = bound < partition.capacity ? bound : partition.capacity;
    const int share          = r / partition.proportional * partition.proportional; // its group's first row
    unsigned live = 0, keys = 0, budget = 0;
INFERNIX_CAUSAL_UNROLL
    for (int s = 0; s < kCausalProportionalMaxRows; ++s) {
        const unsigned other_window = __shfl_sync(kFull, window, s);
        const int other_bound       = __shfl_sync(kFull, bound, s);
        if (__shfl_sync(kFull, share, s) == share) {
            live += other_window != 0U ? 1U : 0U;
            keys += other_window;
            budget += other_window != 0U ? static_cast<unsigned>(other_bound) : 0U;
        }
    }
    const unsigned spare      = budget > live ? budget - live : 0U;
    const unsigned reciprocal = keys != 0U ? 0xffffffffU / keys : 0U; // floor((2^32 - 1) / keys)
    int count                 = 0;
    if (window != 0U) {
        // floor(spare x window / keys): the reciprocal's quotient is at most one short.
        const unsigned product = spare * window;
        unsigned quotient = static_cast<unsigned>((static_cast<unsigned long long>(product) * reciprocal) >> 32);
        if (product - quotient * keys >= keys) ++quotient;
        unsigned split_share = 1U + quotient;
        const unsigned per64 = (window + 63U) >> 6;
        split_share          = split_share < per64 ? split_share : per64;
        count                = static_cast<int>(split_share < static_cast<unsigned>(kBf16KvProportionalMaxSplits)
                                                    ? split_share
                                                    : static_cast<unsigned>(kBf16KvProportionalMaxSplits));
    }
    int inclusive = count;
INFERNIX_CAUSAL_UNROLL
    for (int offset = 1; offset < kCausalProportionalMaxRows; offset <<= 1) {
        const int lower = __shfl_up_sync(kFull, inclusive, offset);
        if (r >= offset) { inclusive += lower; }
    }
    return {count, inclusive - count, __shfl_sync(kFull, inclusive, kCausalProportionalMaxRows - 1)};
}

// Row `last`'s first slot (base: the slots of the rows before it) and count (a row past the launch's: every row's
// count, and none).
__device__ inline void bf16_kv_proportional_row(const Bf16KvPartition& partition, const std::int32_t* positions,
                                                const std::int32_t* validity, int width, int rows,
                                                int visible_capacity, int last, int& base, int& count) {
    constexpr unsigned kFull       = 0xffffffffU;
    const CausalProportionalRow me = bf16_kv_proportional_lane(partition, positions, validity, width, rows,
                                                               visible_capacity);
    const bool inside              = last >= 0 && last < kCausalProportionalMaxRows;
    const int source               = inside ? last : 0;
    const int row_base             = __shfl_sync(kFull, me.base, source);
    const int row_count            = __shfl_sync(kFull, me.count, source);
    base                           = inside ? row_base : me.total;
    count                          = inside ? row_count : 0;
}

// The row and split the CTA at slot `flat` (row-major over grid z, then y) of such a launch serves, with that
// row's count and first slot; false for a slot past every row's count.
__device__ inline bool bf16_kv_proportional_slot(const Bf16KvPartition& partition, const std::int32_t* positions,
                                                 const std::int32_t* validity, int width, int rows,
                                                 int visible_capacity, int flat, CausalRowSplit& out) {
    constexpr unsigned kFull       = 0xffffffffU;
    const CausalProportionalRow me = bf16_kv_proportional_lane(partition, positions, validity, width, rows,
                                                               visible_capacity);
    const int r                    = static_cast<int>(threadIdx.x & 31U);
    const unsigned hit             = __ballot_sync(
        kFull, r < kCausalProportionalMaxRows && flat >= me.base && flat < me.base + me.count);
    if (hit == 0U) return false;
    const int row   = __ffs(static_cast<int>(hit)) - 1;
    const int count = __shfl_sync(kFull, me.count, row);
    const int base  = __shfl_sync(kFull, me.base, row);
    out             = {row, flat - base, count, base};
    return true;
}
#endif

} // namespace infernix::ops::detail
