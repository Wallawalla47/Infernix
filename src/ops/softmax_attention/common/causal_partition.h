#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace infernix::ops::detail {

// Capture reserves partials for the largest row window. Producer and merge use
// the same row window; a wider capture never changes a row's work partition.
struct CausalKvPartition {
    static constexpr int kMaxSplits = 256;
    int capacity                    = 1;
    int target                      = 1;
    int key_shift                   = 6; // log2 of the minimum KV keys per split
    // log2 of the producer's key tile. Nonzero balances a row's split count down to the fewest
    // splits that keep the same largest number of key tiles per split; rows of at least
    // balance_limit keys keep the plain count.
    int balance_shift = 0;
    int balance_limit = 0;
    // The device's SM count, for partitions whose kernels choose their live split count at launch
    // (mxfp8_tiled_active_splits); zero elsewhere.
    int multiprocessors = 0;
    // Nonzero (batch-coupled launches of more than one row; candidates YD-02): the launch's split slots (capacity
    // per row, grid y times rows) are shared by its rows in proportion to their windows, each live row at least
    // one (causal_proportional_counts), so one long row among short ones takes most of the CTAs instead of the
    // same `target` as each short row. Rows of equal windows keep active()'s count. A row's partials then start
    // at its first slot (the prefix of the earlier rows' counts), not at row x capacity.
    int proportional = 0;

    // The plain count: the live count of any row window up to `visible` is at most this, so
    // capture sizes partials and grids with it.
    __host__ __device__ int bound(int visible) const {
        const int count = (visible + (1 << key_shift) - 1) >> key_shift;
        return count < target ? count : target;
    }

    // A count of at most `count` splits balanced to the fewest that keep the same largest number of key tiles per
    // split (rows of at least balance_limit keys keep it).
    __host__ __device__ int balanced(int visible, int count) const {
        if (balance_shift == 0 || count <= 1 || visible >= balance_limit) return count;
        const int tiles     = (visible + (1 << balance_shift) - 1) >> balance_shift;
        const int per_split = (tiles + count - 1) / count;
        return (tiles + per_split - 1) / per_split;
    }

    __host__ __device__ int active(int visible) const { return balanced(visible, bound(visible)); }

    // The proportional count of a window given its share of the launch's slots: the plain count's key minimum
    // with `share` in place of target (at most kMaxSplits), balanced.
    __host__ __device__ int shared(int visible, int share) const {
        int count = (visible + (1 << key_shift) - 1) >> key_shift;
        count     = count < share ? count : share;
        count     = count < kMaxSplits ? count : kMaxSplits;
        return balanced(visible, count < 1 ? 1 : count);
    }
};

// The most rows a proportional launch shares its slots between (the batched causal launches' bound).
inline constexpr int kCausalProportionalMaxRows = 8;

// Complete groups of independent query tiles should fill at least 90% of the SMs.
// Each dtype chooses whether a shortfall warrants a larger CTA budget.
inline constexpr bool causal_query_tiles_underfill_sms(int independent_tiles,
                                                       int multiprocessor_count) {
    const std::int64_t filled =
        (multiprocessor_count / independent_tiles) * static_cast<std::int64_t>(independent_tiles);
    return filled < static_cast<std::int64_t>(multiprocessor_count) * 9 / 10;
}

inline constexpr int causal_partition_target(std::int64_t cta_budget, int independent_tiles) {
    return static_cast<int>(
        std::clamp<std::int64_t>(cta_budget / independent_tiles, 1, CausalKvPartition::kMaxSplits));
}

// A row partitions KV as if all of its physical columns were live, so masking trailing columns
// never moves the split boundaries of the live ones. Live positions are sequential from the
// first, so a dense row's window is its last position plus one. The envelope bounds the window
// only where the physical width reaches past the visible capacity.
__host__ __device__ inline int causal_row_window(int first_position, int width,
                                                 int visible_capacity) {
    const int window = first_position + width;
    return window < visible_capacity ? window : visible_capacity;
}

// YD-02's proportional partition: every row's live split count for a launch of `rows` rows of `width` columns
// (row r's positions at positions[r * width]) sharing `slots` split slots. A row whose first position is negative
// (no live work) takes none; each live row takes one, and the rest of the slots go to the live rows in proportion
// to their windows (rounded down), each capped by CausalKvPartition::shared. The counts sum to at most `slots`, and
// rows of equal windows keep CausalKvPartition::active's count whenever every row's share reaches it. Every CTA of
// a launch (producer and merge alike) computes the same counts from the same positions. Returns the total.
__device__ inline int causal_proportional_counts(const CausalKvPartition& partition, const std::int32_t* positions,
                                                 int width, int rows, int visible_capacity, int slots,
                                                 int (&counts)[kCausalProportionalMaxRows]) {
    // 32-bit arithmetic throughout: a window is at most 2^20 keys (kCausalAttentionMaximumVisibleKeys), the spare
    // slots at most 8 x 256, so spare x window < 2^32 and the eight windows' sum < 2^24. Every CTA runs this before
    // its key loop; 64-bit divisions cost about a timer tick (2 us) on 4K-key rounds.
    unsigned windows[kCausalProportionalMaxRows];
    unsigned live = 0, keys = 0;
    for (int r = 0; r < kCausalProportionalMaxRows; ++r) {
        const int first = r < rows ? positions[static_cast<std::int64_t>(r) * width] : -1;
        const int window = first >= 0 ? causal_row_window(first, width, visible_capacity) : 0;
        windows[r]       = window > 0 ? static_cast<unsigned>(window) : 0U;
        live += windows[r] != 0U ? 1U : 0U;
        keys += windows[r];
    }
    const unsigned spare = static_cast<unsigned>(slots) > live ? static_cast<unsigned>(slots) - live : 0U;
    int total            = 0;
    for (int r = 0; r < kCausalProportionalMaxRows; ++r) {
        counts[r] = windows[r] != 0U
                        ? partition.shared(static_cast<int>(windows[r]), 1 + static_cast<int>(spare * windows[r] / keys))
                        : 0;
        total += counts[r];
    }
    return total;
}

// The row and split a proportional launch's CTA at slot `flat` (row-major over grid z, then y) serves, with that
// row's count and first slot; false for a slot past every row's count.
struct CausalRowSplit {
    int row   = 0;
    int split = 0;
    int count = 0;
    int base  = 0;
};

__device__ inline bool causal_proportional_slot(const int (&counts)[kCausalProportionalMaxRows], int flat,
                                                CausalRowSplit& out) {
    int base = 0;
    for (int r = 0; r < kCausalProportionalMaxRows; ++r) {
        if (flat < base + counts[r]) {
            out = {r, flat - base, counts[r], base};
            return true;
        }
        base += counts[r];
    }
    return false;
}

} // namespace infernix::ops::detail
