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

// A share group's shares: every row's window and the group's split budget, for `rows` rows (at most
// kCausalProportionalMaxRows) of `width` columns (row r's positions at positions[r * width]; `validity` null, or
// each row's live columns). A row with no live column or a negative first position has window 0 and takes no
// split. The budget is the sum of the live rows' plain bounds, min(target of the window, one split per 64 keys,
// capacity): what live() runs at most, so the group runs about as many CTAs as the plain partition and never more
// than its capacity x rows slots. Each live row takes one split and the rest go to the rows in proportion to their
// windows (rounded down), a row never more than one split per 64 keys or the merge's block: that is its count, the
// slots it owns. It runs them as layout(window, count): equal splits of whole key tiles, the last slots left empty
// where the tile rounding needs fewer. Rows of equal windows have their plain bound as count and so live()'s
// splits, key for key (when the envelope's capacity is below the target, the capacity's). Every CTA of a launch
// (producer and merge alike) computes the same shares from the same positions, in integers. The rows' positions
// load together, without a branch between them, and the shares divide by the same sum of windows through its
// reciprocal: one load round trip and one division before a CTA's first key load. 32-bit arithmetic: a window is
// at most 2^20 keys and the spare slots at most 8 x 256, so spare x window < 2^32.
struct Bf16KvProportionalShares {
    unsigned windows[kCausalProportionalMaxRows];
    unsigned keys       = 0;
    unsigned spare      = 0;
    unsigned reciprocal = 0; // floor((2^32 - 1) / keys)

    __device__ Bf16KvProportionalShares(const Bf16KvPartition& partition, const std::int32_t* positions,
                                        const std::int32_t* validity, int width, int rows, int visible_capacity) {
        int first[kCausalProportionalMaxRows], valid[kCausalProportionalMaxRows];
        for (int r = 0; r < kCausalProportionalMaxRows; ++r) {
            const int row = r < rows ? r : 0; // a row past the group reads row 0 (and is given no window below)
            first[r]      = positions[static_cast<std::int64_t>(row) * width];
            valid[r]      = validity != nullptr ? validity[row] : 1;
        }
        unsigned live = 0, budget = 0;
        for (int r = 0; r < kCausalProportionalMaxRows; ++r) {
            const bool works = r < rows && valid[r] > 0 && first[r] >= 0;
            const int window = works ? causal_row_window(first[r], width, visible_capacity) : 0;
            windows[r]       = window > 0 ? static_cast<unsigned>(window) : 0U;
            int bound        = window >= 32768 ? partition.long_target : partition.normal_target;
            const int by_64  = (window + 63) >> 6;
            bound            = bound < by_64 ? bound : by_64;
            bound            = bound < partition.capacity ? bound : partition.capacity;
            live += windows[r] != 0U ? 1U : 0U;
            keys += windows[r];
            budget += windows[r] != 0U ? static_cast<unsigned>(bound) : 0U;
        }
        spare      = budget > live ? budget - live : 0U;
        reciprocal = keys != 0U ? 0xffffffffU / keys : 0U;
    }

    // Row r's count: the split slots it owns (0 for a row without work).
    __device__ int count(int r) const {
        const unsigned window = windows[r];
        if (window == 0U) return 0;
        // floor(spare x window / keys): the reciprocal's quotient is at most one short.
        const unsigned product = spare * window;
        unsigned quotient      = static_cast<unsigned>((static_cast<unsigned long long>(product) * reciprocal) >> 32);
        if (product - quotient * keys >= keys) ++quotient;
        unsigned share       = 1U + quotient;
        const unsigned by_64 = (window + 63U) >> 6;
        share                = share < by_64 ? share : by_64;
        return static_cast<int>(share < static_cast<unsigned>(kBf16KvProportionalMaxSplits)
                                    ? share
                                    : static_cast<unsigned>(kBf16KvProportionalMaxSplits));
    }

    // A row's splits within its `count` slots: live_with(window, count), the fewest whole key tiles a split (at
    // least 64 keys) that cover the window in `count` splits, and the splits that then hold keys (the tile rounding
    // can leave the last slots empty: their CTAs return, and the merge reads only the live ones). Only a CTA's own
    // row pays these divisions.
    __device__ static Bf16KvLivePartition layout(const Bf16KvPartition& partition, int window, int count) {
        return partition.live_with(window, count);
    }
};

// A proportional launch of `rows` rows, in share groups of partition.proportional consecutive rows: the count and
// first slot of every row up to `last` (base: the slots of the rows before `last`; count: its own).
__device__ inline void bf16_kv_proportional_row(const Bf16KvPartition& partition, const std::int32_t* positions,
                                                const std::int32_t* validity, int width, int rows,
                                                int visible_capacity, int last, int& base, int& count) {
    base  = 0;
    count = 0;
    for (int begin = 0; begin <= last && begin < rows; begin += partition.proportional) {
        const int group = partition.proportional < rows - begin ? partition.proportional : rows - begin;
        const Bf16KvProportionalShares shares(partition, positions + static_cast<std::int64_t>(begin) * width,
                                              validity != nullptr ? validity + begin : nullptr, width, group,
                                              visible_capacity);
        for (int r = 0; r < group; ++r) {
            const int c = shares.count(r);
            if (begin + r == last) {
                count = c;
                return;
            }
            base += c;
        }
    }
}

// The row and split the CTA at slot `flat` (row-major over grid z, then y) of such a launch serves, with that
// row's count and first slot; false for a slot past every row's count.
__device__ inline bool bf16_kv_proportional_slot(const Bf16KvPartition& partition, const std::int32_t* positions,
                                                 const std::int32_t* validity, int width, int rows,
                                                 int visible_capacity, int flat, CausalRowSplit& out) {
    int base = 0;
    for (int begin = 0; begin < rows; begin += partition.proportional) {
        const int group = partition.proportional < rows - begin ? partition.proportional : rows - begin;
        const Bf16KvProportionalShares shares(partition, positions + static_cast<std::int64_t>(begin) * width,
                                              validity != nullptr ? validity + begin : nullptr, width, group,
                                              visible_capacity);
        for (int r = 0; r < group; ++r) {
            const int c = shares.count(r);
            if (flat < base + c) {
                out = {begin + r, flat - base, c, base};
                return true;
            }
            base += c;
        }
    }
    return false;
}

} // namespace infernix::ops::detail
