#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

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

    // The plain count: the live count of any row window up to `visible` is at most this, so
    // capture sizes partials and grids with it.
    __host__ __device__ int bound(int visible) const {
        const int count = (visible + (1 << key_shift) - 1) >> key_shift;
        return count < target ? count : target;
    }

    __host__ __device__ int active(int visible) const {
        const int count = bound(visible);
        if (balance_shift == 0 || count <= 1 || visible >= balance_limit) return count;
        const int tiles     = (visible + (1 << balance_shift) - 1) >> balance_shift;
        const int per_split = (tiles + count - 1) / count;
        return (tiles + per_split - 1) / per_split;
    }
};

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

} // namespace ninfer::ops::detail
