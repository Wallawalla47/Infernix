#pragma once

// Implements: include/ninfer/ops/top_logprobs.h
// Match: contiguous BF16 [physical_rows,C], I32 [K,C], and FP32 [K,C].
// Algorithm assumptions: one 256-thread CTA per column computes the column's logsumexp, then each
// of K passes selects the largest (value, -row) key below the previous pass's.

#include "ops/kernel/target_logprobs.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kTopLogprobsBlock = 256;

// Larger keys rank first: descending value, then ascending row. Equal values share their high
// half (zero of either sign included), and every valid row's key is nonzero.
__device__ __forceinline__ unsigned long long top_logprobs_key(float value, std::int32_t row) {
    const unsigned bits    = __float_as_uint(value == 0.0f ? 0.0f : value);
    const unsigned ordered = (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
    return (static_cast<unsigned long long>(ordered) << 32) |
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

template <int BlockSize>
__launch_bounds__(BlockSize) __global__
    void top_logprobs_kernel(const __nv_bfloat16* logits, std::int32_t* ids, float* output,
                             std::int32_t valid_rows, std::int32_t physical_rows,
                             std::int32_t count) {
    const std::int32_t column   = static_cast<std::int32_t>(blockIdx.x);
    const __nv_bfloat16* values = logits + static_cast<std::int64_t>(column) * physical_rows;
    const TargetLogprobsColumn lse = target_logprobs_column<BlockSize>(values, valid_rows);

    unsigned long long previous = ~0ULL;
    for (std::int32_t rank = 0; rank < count; ++rank) {
        unsigned long long best = 0;
        for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
             row += BlockSize) {
            const unsigned long long key = top_logprobs_key(__bfloat162float(values[row]), row);
            if (key < previous && key > best) { best = key; }
        }
        best = top_logprobs_block_max<BlockSize>(best);
        if (threadIdx.x == 0) {
            const auto row = static_cast<std::int32_t>(0xffffffffu - static_cast<unsigned>(best));
            const std::int64_t index = static_cast<std::int64_t>(column) * count + rank;
            ids[index]               = row;
            output[index] = __bfloat162float(values[row]) - lse.maximum - lse.log_sum;
        }
        previous = best;
    }
}

} // namespace ninfer::ops
