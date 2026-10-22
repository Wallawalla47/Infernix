#pragma once

// Implements: include/infernix/ops/top_logprobs.h
// Match: contiguous BF16 or FP32 [physical_rows,C], I32 [K,C], and FP32 [K,C].
// Algorithm assumptions: one 256-thread CTA per column computes the column's logsumexp. For K up
// to kTopLogprobsSerialMaximum (and above kTopLogprobsSelectMaximum) each of K passes then selects
// the largest (value, -row) key below the previous pass's. Between the two, a radix selection
// finds the K-th largest key in a few passes (one per kTopLogprobsDigitBits-bit digit at most,
// usually two or three), one pass gathers the K keys at or above it, and a bitonic sort in shared
// memory orders them. Both select the same rows in the same order and compute the same outputs.

#include "ops/kernel/target_logprobs.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops {

inline constexpr int kTopLogprobsBlock = 256;
// Largest K the K-pass selection serves: the sampled-draft (4) and DFlash2 candidate (16) heads
// stay on it until the radix selection is timed at their shapes.
inline constexpr int kTopLogprobsSerialMaximum = 16;
// Largest K the radix selection serves (its shared key buffer).
inline constexpr int kTopLogprobsSelectMaximum = 1024;
inline constexpr int kTopLogprobsDigitBits     = 11;

// A value's 32-bit key: unsigned order is value order, with zeros of either sign equal.
__device__ __forceinline__ unsigned top_logprobs_ordered(float value) {
    const unsigned bits = __float_as_uint(value == 0.0f ? 0.0f : value);
    return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
}

// Larger keys rank first: descending value, then ascending row. Equal values share their high
// half (zero of either sign included), and every valid row's key is nonzero.
__device__ __forceinline__ unsigned long long top_logprobs_key(float value, std::int32_t row) {
    return (static_cast<unsigned long long>(top_logprobs_ordered(value)) << 32) |
           (0xffffffffu - static_cast<unsigned>(row));
}

template <int BlockSize>
__device__ __forceinline__ unsigned long long top_logprobs_block_max(unsigned long long value) {
    static_assert(BlockSize >= kWarpSize && BlockSize <= 1024);
    static_assert((BlockSize & (BlockSize - 1)) == 0);
    constexpr int kWarps = BlockSize / kWarpSize;
    __shared__ unsigned long long warp_maxima[kWarps];
    __shared__ unsigned long long result;

    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = max(value, __shfl_xor_sync(kFullWarpMask, value, offset));
    }
    if (lane == 0) { warp_maxima[warp] = value; }
    __syncthreads();

    if (warp == 0) {
        value = lane < kWarps ? warp_maxima[lane] : 0ULL;
#pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            value = max(value, __shfl_xor_sync(kFullWarpMask, value, offset));
        }
        if (lane == 0) { result = value; }
    }
    __syncthreads();
    return result;
}

template <int BlockSize, class Logit>
__launch_bounds__(BlockSize) __global__
    void top_logprobs_kernel(const Logit* logits, std::int32_t* ids, float* output,
                             std::int32_t valid_rows, std::int32_t physical_rows,
                             std::int32_t count) {
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    const Logit* values       = logits + static_cast<std::int64_t>(column) * physical_rows;
    const TargetLogprobsColumn lse = target_logprobs_column<BlockSize>(values, valid_rows);

    unsigned long long previous = ~0ULL;
    for (std::int32_t rank = 0; rank < count; ++rank) {
        unsigned long long best = 0;
        for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
             row += BlockSize) {
            const unsigned long long key = top_logprobs_key(logprob_logit(values[row]), row);
            if (key < previous && key > best) { best = key; }
        }
        best = top_logprobs_block_max<BlockSize>(best);
        if (threadIdx.x == 0) {
            const auto row = static_cast<std::int32_t>(0xffffffffu - static_cast<unsigned>(best));
            const std::int64_t index = static_cast<std::int64_t>(column) * count + rank;
            ids[index]               = row;
            output[index] = logprob_logit(values[row]) - lse.maximum - lse.log_sum;
        }
        previous = best;
    }
}

// The exclusive prefix sum of value over the block's threads in thread order, to every thread.
template <int BlockSize>
__device__ __forceinline__ unsigned top_logprobs_exclusive_sum(unsigned value, unsigned* warp_totals) {
    constexpr int kWarps = BlockSize / kWarpSize;
    static_assert(kWarps <= kWarpSize);
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    unsigned inclusive = value;
#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const unsigned lower = __shfl_up_sync(kFullWarpMask, inclusive, offset);
        if (lane >= offset) { inclusive += lower; }
    }
    if (lane == kWarpSize - 1) { warp_totals[warp] = inclusive; }
    __syncthreads();
    if (warp == 0) {
        const unsigned total = lane < kWarps ? warp_totals[lane] : 0u;
        unsigned running     = total;
#pragma unroll
        for (int offset = 1; offset < kWarpSize; offset <<= 1) {
            const unsigned lower = __shfl_up_sync(kFullWarpMask, running, offset);
            if (lane >= offset) { running += lower; }
        }
        if (lane < kWarps) { warp_totals[lane] = running - total; }
    }
    __syncthreads();
    return warp_totals[warp] + inclusive - value;
}

// Radix selection. A valid row's key is its ordered value above row_bits bits of (row_limit - row),
// so keys are distinct and descending key order is the contract's order. Each round histograms the
// next digit of the keys that share the prefix fixed so far and fixes the digit holding the K-th
// largest key; once the chosen bucket holds exactly the keys still needed, the K largest keys are
// exactly those at or above the prefix (the last digit's buckets hold one key each, so this always
// happens).
template <int BlockSize, class Logit>
__launch_bounds__(BlockSize) __global__
    void top_logprobs_select_kernel(const Logit* logits, std::int32_t* ids, float* output,
                                    std::int32_t valid_rows, std::int32_t physical_rows,
                                    std::int32_t count) {
    constexpr int kBins = 1 << kTopLogprobsDigitBits;
    __shared__ unsigned histogram[kBins];
    __shared__ unsigned long long keys[kTopLogprobsSelectMaximum];
    __shared__ unsigned warp_totals[BlockSize / kWarpSize];
    __shared__ unsigned long long prefix;
    __shared__ unsigned remaining;
    __shared__ unsigned gathered;
    __shared__ bool settled;

    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    const Logit* values       = logits + static_cast<std::int64_t>(column) * physical_rows;
    const TargetLogprobsColumn lse = target_logprobs_column<BlockSize>(values, valid_rows);

    // valid_rows <= 2^31 - 1, so row_bits <= 31 and keys fit in 63 bits.
    const int row_bits =
        valid_rows > 1 ? 32 - __clz(static_cast<unsigned>(valid_rows - 1)) : 1;
    const unsigned row_limit = (1u << row_bits) - 1u;
    const auto key_of        = [&](std::int32_t row) {
        return (static_cast<unsigned long long>(top_logprobs_ordered(logprob_logit(values[row])))
                << row_bits) |
               (row_limit - static_cast<unsigned>(row));
    };

    if (threadIdx.x == 0) {
        prefix    = 0;
        remaining = static_cast<unsigned>(count);
        gathered  = 0;
        settled   = false;
    }
    for (int high = 32 + row_bits; high > 0;) {
        const int width = high < kTopLogprobsDigitBits ? high : kTopLogprobsDigitBits;
        const int shift = high - width;
        const int bins  = 1 << width;
        for (int bin = static_cast<int>(threadIdx.x); bin < bins; bin += BlockSize) {
            histogram[bin] = 0;
        }
        __syncthreads();
        const unsigned long long fixed = prefix >> high;
        const unsigned need            = remaining;
        for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
             row += BlockSize) {
            const unsigned long long key = key_of(row);
            if ((key >> high) == fixed) {
                atomicAdd(&histogram[static_cast<int>(key >> shift) & (bins - 1)], 1u);
            }
        }
        __syncthreads();
        // Thread t holds the digits [bins - (t + 1) per, bins - t per): thread 0 the largest.
        const int per    = (bins + BlockSize - 1) / BlockSize;
        const int top    = bins - static_cast<int>(threadIdx.x) * per;
        const int bottom = top - per > 0 ? top - per : 0;
        unsigned local   = 0;
        for (int bin = bottom; bin < top; ++bin) { local += histogram[bin]; }
        const unsigned before = top_logprobs_exclusive_sum<BlockSize>(local, warp_totals);
        if (before < need && need <= before + local) {
            unsigned above = before;
            for (int bin = top - 1; bin >= bottom; --bin) {
                const unsigned held = histogram[bin];
                if (above + held >= need) {
                    prefix |= static_cast<unsigned long long>(bin) << shift;
                    remaining = need - above;
                    settled   = held == need - above;
                    break;
                }
                above += held;
            }
        }
        __syncthreads();
        if (settled) { break; }
        high = shift;
    }

    const unsigned long long threshold = prefix;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        const unsigned long long key = key_of(row);
        if (key >= threshold) { keys[atomicAdd(&gathered, 1u)] = key; }
    }
    int size = 1;
    while (size < count) { size <<= 1; }
    // Keys are nonzero, so zero padding sorts last.
    for (int index = count + static_cast<int>(threadIdx.x); index < size; index += BlockSize) {
        keys[index] = 0;
    }
    __syncthreads();
    for (int span = 2; span <= size; span <<= 1) {
        for (int stride = span >> 1; stride > 0; stride >>= 1) {
            for (int index = static_cast<int>(threadIdx.x); index < size; index += BlockSize) {
                const int partner = index ^ stride;
                if (partner > index) {
                    const unsigned long long a = keys[index];
                    const unsigned long long b = keys[partner];
                    // Descending runs where (index & span) == 0, ascending otherwise: the last
                    // merge (span == size) leaves the whole buffer descending.
                    if ((index & span) == 0 ? a < b : a > b) {
                        keys[index]   = b;
                        keys[partner] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (std::int32_t rank = static_cast<std::int32_t>(threadIdx.x); rank < count; rank += BlockSize) {
        const auto row = static_cast<std::int32_t>(row_limit - static_cast<unsigned>(keys[rank] & row_limit));
        const std::int64_t index = static_cast<std::int64_t>(column) * count + rank;
        ids[index]               = row;
        output[index]            = logprob_logit(values[row]) - lse.maximum - lse.log_sum;
    }
}

} // namespace infernix::ops
