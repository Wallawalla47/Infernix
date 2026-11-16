#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

// Full unrolling of the proportional partitions' fixed-trip row loops in device code, so their arrays stay in
// registers; host compilers see nothing.
#if defined(__CUDA_ARCH__)
#    define INFERNIX_CAUSAL_UNROLL _Pragma("unroll")
#else
#    define INFERNIX_CAUSAL_UNROLL
#endif

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
    // one (causal_proportional_lane), so one long row among short ones takes most of the CTAs instead of the
    // same `target` as each short row. Rows of equal windows keep active()'s count. A row's partials then start
    // at its first slot (the prefix of the earlier rows' counts), not at row x capacity. The value is the share
    // group: each run of `proportional` consecutive rows shares its own slots (a column's lanes on the rows
    // route, whose launches hold whole columns, so a column's counts equal its one-column call's; ZK-01).
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
// (row r's positions at positions[r * width]) sharing `slots` split slots (slots / rows a row). The rows form share
// groups of partition.proportional rows (the whole launch when that is 1 or covers it), and each group shares its
// rows' slots: a row whose first position is negative (no live work) takes none; each live row takes one, and the
// rest of the group's slots go to its live rows in proportion to their windows (rounded down), each capped by
// CausalKvPartition::shared. The counts sum to at most `slots`, and rows of equal windows keep
// CausalKvPartition::active's count whenever every row's share reaches it. Every CTA of a launch (producer and merge
// alike) computes the same counts from the same positions.
//
// It runs warp-wide: lane r < kCausalProportionalMaxRows holds row r (its window, its share group's first row, then
// its count and its first slot, the prefix of the earlier rows' counts), the group sums come from shuffles and the
// first slots from a warp scan, so no array is kept (an array of the eight rows' counts, loops over it stepping by
// the run-time group size, put 64-128 bytes of stack and up to 246 local loads into the INT8, FP8, NVFP4 and K8V4
// decode kernels from 56d5726e on, and their register pressure spilled values the key loop reloads). Every lane of the
// calling warp calls it with the same arguments. 32-bit arithmetic throughout: a window is at most 2^20 keys
// (kCausalAttentionMaximumVisibleKeys), the spare slots at most 8 x 256, so spare x window < 2^32 and the eight
// windows' sum < 2^24.
struct CausalProportionalRow {
    int count = 0; // the lane's row's count (0 for lanes past the eight rows)
    int base  = 0; // its first slot (lanes 0..7)
    int total = 0; // every row's count, in every lane
};

#if defined(__CUDACC__)
__device__ inline CausalProportionalRow causal_proportional_lane(const CausalKvPartition& partition,
                                                                 const std::int32_t* positions, int width, int rows,
                                                                 int visible_capacity, int slots) {
    constexpr unsigned kFull = 0xffffffffU;
    const int r              = static_cast<int>(threadIdx.x & 31U);
    const int first  = r < kCausalProportionalMaxRows && r < rows ? positions[static_cast<std::int64_t>(r) * width] : -1;
    const int value  = first >= 0 ? causal_row_window(first, width, visible_capacity) : 0;
    const unsigned window = value > 0 ? static_cast<unsigned>(value) : 0U;
    const int group = partition.proportional > 1 && partition.proportional < rows ? partition.proportional : rows;
    const int share = group > 0 ? r / group * group : 0; // the first row of this row's share group
    unsigned live = 0, keys = 0;
INFERNIX_CAUSAL_UNROLL
    for (int s = 0; s < kCausalProportionalMaxRows; ++s) {
        const unsigned other = __shfl_sync(kFull, window, s);
        if (__shfl_sync(kFull, share, s) == share) {
            live += other != 0U ? 1U : 0U;
            keys += other;
        }
    }
    const unsigned per_row = rows > 0 ? static_cast<unsigned>(slots / rows) : 0U;
    const int end = group > 0 && share + group < kCausalProportionalMaxRows ? share + group : kCausalProportionalMaxRows;
    const int in_group   = (rows < end ? rows : end) - share;
    const unsigned owned = in_group > 0 ? per_row * static_cast<unsigned>(in_group) : 0U;
    const unsigned spare = owned > live ? owned - live : 0U;
    const int count      = r < kCausalProportionalMaxRows && window != 0U
                               ? partition.shared(static_cast<int>(window), 1 + static_cast<int>(spare * window / keys))
                               : 0;
    int inclusive = count;
INFERNIX_CAUSAL_UNROLL
    for (int offset = 1; offset < kCausalProportionalMaxRows; offset <<= 1) {
        const int lower = __shfl_up_sync(kFull, inclusive, offset);
        if (r >= offset) { inclusive += lower; }
    }
    return {count, inclusive - count, __shfl_sync(kFull, inclusive, kCausalProportionalMaxRows - 1)};
}
#endif

// The row and split a proportional launch's CTA at slot `flat` (row-major over grid z, then y) serves, with that
// row's count and first slot; false for a slot past every row's count. Warp-wide, as causal_proportional_lane.
struct CausalRowSplit {
    int row   = 0;
    int split = 0;
    int count = 0;
    int base  = 0;
};

#if defined(__CUDACC__)
__device__ inline bool causal_proportional_slot(const CausalKvPartition& partition, const std::int32_t* positions,
                                                int width, int rows, int visible_capacity, int slots, int flat,
                                                CausalRowSplit& out) {
    constexpr unsigned kFull       = 0xffffffffU;
    const CausalProportionalRow me = causal_proportional_lane(partition, positions, width, rows, visible_capacity, slots);
    const int r                    = static_cast<int>(threadIdx.x & 31U);
    const unsigned hit             = __ballot_sync(
        kFull, r < kCausalProportionalMaxRows && flat >= me.base && flat < me.base + me.count);
    if (hit == 0U) { return false; }
    const int row   = __ffs(static_cast<int>(hit)) - 1;
    const int count = __shfl_sync(kFull, me.count, row);
    const int base  = __shfl_sync(kFull, me.base, row);
    out             = {row, flat - base, count, base};
    return true;
}

// The merge's view: row `row`'s first slot and count (a row past the eight: every count, and none).
__device__ inline void causal_proportional_row(const CausalKvPartition& partition, const std::int32_t* positions,
                                               int width, int rows, int visible_capacity, int slots, int row, int& base,
                                               int& count) {
    constexpr unsigned kFull       = 0xffffffffU;
    const CausalProportionalRow me = causal_proportional_lane(partition, positions, width, rows, visible_capacity, slots);
    const bool inside              = row >= 0 && row < kCausalProportionalMaxRows;
    const int source               = inside ? row : 0;
    const int row_base             = __shfl_sync(kFull, me.base, source);
    const int row_count            = __shfl_sync(kFull, me.count, source);
    base                           = inside ? row_base : me.total;
    count                          = inside ? row_count : 0;
}
#endif

} // namespace infernix::ops::detail
