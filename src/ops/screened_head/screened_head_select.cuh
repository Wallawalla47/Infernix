#pragma once

// The screened heads' shared device code (include/infernix/ops/screened_head.h): the candidate order, the
// per-column selection of the rows to rescore from the screen's per-CTA lists, and the warp sums of an exact
// row. Used by the 2,560-wide screens (screened_head.cu) and the 5,120-wide one (screened_head_wide.cu).

#include "ops/common/score_id_order.cuh"
#include "ops/screened_head/screened_head_launch.h"

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

namespace infernix::ops::detail::screened_head {

inline constexpr int kSelectThreads    = 512;
inline constexpr int kSelectWarps      = kSelectThreads / 32;
inline constexpr int kMaxRescored      = 256;
inline constexpr int kMaxEntries       = kMaxGrid * kListLength; // candidates of one column
inline constexpr int kEntriesPerThread = kMaxEntries / kSelectThreads;
static_assert(kMaxEntries % kSelectThreads == 0);

__device__ __forceinline__ float negative_infinity() { return __int_as_float(0xff800000); }

// (v, r) ranks above (w, s): the higher score, on equal scores the lower row.
__device__ __forceinline__ bool ranks_above(float v, int r, float w, int s) {
    return v > w || (v == w && r < s);
}

__device__ __forceinline__ float warp_butterfly(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, offset); }
    return v;
}

// A column's list of kListLength candidates, best first, kept by one thread: inserts (v, row) when it ranks
// above the last. Constant indices only (registers, no local memory).
__device__ __forceinline__ void list_insert(float (&best)[kListLength], int (&best_row)[kListLength], float v,
                                            int row) {
    if (!ranks_above(v, row, best[kListLength - 1], best_row[kListLength - 1])) { return; }
    bool placed = false;
#pragma unroll
    for (int i = kListLength - 1; i > 0; --i) {
        if (!placed) {
            if (ranks_above(v, row, best[i - 1], best_row[i - 1])) {
                best[i]     = best[i - 1];
                best_row[i] = best_row[i - 1];
            } else {
                best[i]     = v;
                best_row[i] = row;
                placed      = true;
            }
        }
    }
    if (!placed) {
        best[0]     = v;
        best_row[0] = row;
    }
}

// ---- Selection (one CTA of kSelectThreads per column)

struct SelectShared {
    std::uint32_t histogram[256];
    std::uint64_t reduce64[kSelectWarps];
    int reduce32[kSelectWarps];
    std::uint64_t prefix;
    int need;
    int digit_above; // keys above the chosen digit, in this pass
    int digit;
    int rows[kMaxRescored];
    float screen[kMaxRescored];
};

__device__ __forceinline__ std::uint64_t block_max(std::uint64_t value, SelectShared& s) {
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const std::uint64_t other = __shfl_xor_sync(0xffffffffu, value, offset);
        value                     = other > value ? other : value;
    }
    if (lane == 0) { s.reduce64[warp] = value; }
    __syncthreads();
    std::uint64_t out = s.reduce64[0];
    for (int w = 1; w < kSelectWarps; ++w) { out = s.reduce64[w] > out ? s.reduce64[w] : out; }
    __syncthreads();
    return out;
}

__device__ __forceinline__ int block_sum(int value, SelectShared& s) {
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_xor_sync(0xffffffffu, value, offset); }
    if (lane == 0) { s.reduce32[warp] = value; }
    __syncthreads();
    int out = 0;
    for (int w = 0; w < kSelectWarps; ++w) { out += s.reduce32[w]; }
    __syncthreads();
    return out;
}

// The `target`-th largest of the block's distinct nonzero keys (the keys at or above it are exactly
// `target`), by eight passes of 8-bit radix selection from the top byte.
__device__ inline std::uint64_t select_key(const std::uint64_t (&keys)[kEntriesPerThread], int target,
                                           SelectShared& s) {
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    if (tid == 0) {
        s.prefix = 0;
        s.need   = target;
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        if (tid < 256) { s.histogram[tid] = 0; }
        __syncthreads();
        const std::uint64_t prefix = s.prefix;
        const std::uint64_t mask   = shift == 56 ? 0 : ~((std::uint64_t{1} << (shift + 8)) - 1);
#pragma unroll
        for (int i = 0; i < kEntriesPerThread; ++i) {
            if (keys[i] != 0 && (keys[i] & mask) == prefix) {
                atomicAdd(&s.histogram[(keys[i] >> shift) & 0xFF], 1u);
            }
        }
        __syncthreads();
        if (warp == 0) {
            // Lane l owns digits 255 - 8l down to 248 - 8l; scan from the top.
            int counts[8];
            int own = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                counts[i] = static_cast<int>(s.histogram[255 - 8 * lane - i]);
                own += counts[i];
            }
            int above = own; // inclusive scan over lanes (lane 0 = the highest digits)
#pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                const int other = __shfl_up_sync(0xffffffffu, above, offset);
                if (lane >= offset) { above += other; }
            }
            const int before = above - own;
            const int need   = s.need;
            const unsigned holder = __ballot_sync(0xffffffffu, before < need && above >= need);
            if (lane == __ffs(holder) - 1) {
                int running = before;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (running + counts[i] >= need) {
                        s.digit       = 255 - 8 * lane - i;
                        s.digit_above = running;
                        break;
                    }
                    running += counts[i];
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            s.prefix |= static_cast<std::uint64_t>(s.digit) << shift;
            s.need -= s.digit_above;
        }
        __syncthreads();
    }
    return s.prefix;
}

// Selects column `column`'s rows to rescore from its lists ([grid][kListLength]) into s.rows / s.screen
// in list order; returns their count.
__device__ inline int select_rows(const float2* lists, int grid, float margin, int ranks, int cap, SelectShared& s) {
    const int tid     = static_cast<int>(threadIdx.x);
    const int entries = grid * kListLength;
    std::uint64_t keys[kEntriesPerThread];
    std::uint64_t top = 0;
#pragma unroll
    for (int i = 0; i < kEntriesPerThread; ++i) {
        const int e = tid + i * kSelectThreads;
        keys[i]     = 0;
        if (e < entries) {
            const float2 entry = __ldcg(lists + e);
            const int row      = __float_as_int(entry.y);
            if (row != INT_MAX) { keys[i] = score_id_order_key(entry.x, row); }
        }
        top = keys[i] > top ? keys[i] : top;
    }
    top = block_max(top, s);
    const float threshold = score_from_order_key(top) - margin;
    int within = 0, valid = 0;
#pragma unroll
    for (int i = 0; i < kEntriesPerThread; ++i) {
        valid += keys[i] != 0;
        within += keys[i] != 0 && score_from_order_key(keys[i]) >= threshold;
    }
    within = block_sum(within, s);
    valid  = block_sum(valid, s);
    const int target = min(min(max(within, ranks), cap), valid);
    // Within the bounds every row within the margin is taken; otherwise the `target` largest keys (the
    // keys at or above the target-th, which is the cut).
    const bool by_margin    = target == within;
    const std::uint64_t cut = by_margin ? 0 : select_key(keys, target, s);
    // Compaction in entry order (entry e = tid + i * kSelectThreads, ranked by (i, tid)).
    int offset = 0;
#pragma unroll
    for (int i = 0; i < kEntriesPerThread; ++i) {
        const bool chosen =
            keys[i] != 0 && (by_margin ? score_from_order_key(keys[i]) >= threshold : keys[i] >= cut);
        const int lane = tid & 31, warp = tid >> 5;
        const unsigned ballot = __ballot_sync(0xffffffffu, chosen);
        if (lane == 0) { s.reduce32[warp] = __popc(ballot); }
        __syncthreads();
        int base = offset;
        for (int w = 0; w < warp; ++w) { base += s.reduce32[w]; }
        int total = 0;
        for (int w = 0; w < kSelectWarps; ++w) { total += s.reduce32[w]; }
        if (chosen) {
            const int position = base + __popc(ballot & ((1u << lane) - 1u));
            if (position < kMaxRescored) {
                s.rows[position]   = id_from_order_key(keys[i]);
                s.screen[position] = score_from_order_key(keys[i]);
            }
        }
        offset += total;
        __syncthreads();
    }
    return min(target, kMaxRescored);
}

} // namespace infernix::ops::detail::screened_head
