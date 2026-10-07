#pragma once

// Implements: include/infernix/ops/target_logprobs.h
// Match: contiguous BF16 [physical_rows,C], I32 [T,C], and FP32 [T,C].
// Algorithm assumptions: one 256-thread CTA performs a single-pass online logsumexp per column.

#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace infernix::ops {

inline constexpr int kTargetLogprobsBlock = 256;

template <int BlockSize>
__device__ __forceinline__ float target_logprobs_block_max(float value) {
    static_assert(BlockSize >= kWarpSize && BlockSize <= 1024);
    static_assert((BlockSize & (BlockSize - 1)) == 0);
    constexpr int kWarps = BlockSize / kWarpSize;
    __shared__ float warp_maxima[kWarps];
    __shared__ float result;

    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    value          = warp_max(value);
    if (lane == 0) { warp_maxima[warp] = value; }
    __syncthreads();

    if (warp == 0) {
        value = lane < kWarps ? warp_maxima[lane] : -CUDART_INF_F;
        value = warp_max(value);
        if (lane == 0) { result = value; }
    }
    __syncthreads();
    return result;
}

// A column's maximum valid logit and log(sum(exp(logit - maximum))), returned to every thread.
struct TargetLogprobsColumn {
    float maximum;
    float log_sum;
};

template <int BlockSize>
__device__ __forceinline__ TargetLogprobsColumn
target_logprobs_column(const __nv_bfloat16* values, std::int32_t valid_rows) {
    float local_max = -CUDART_INF_F;
    float local_sum = 0.0f;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        const float value   = __bfloat162float(values[row]);
        const float new_max = fmaxf(local_max, value);
        if (new_max > local_max) {
            local_sum *= expf(local_max - new_max);
        }
        local_max = new_max;
        local_sum += expf(value - local_max);
    }
    const float maximum = target_logprobs_block_max<BlockSize>(local_max);

    local_sum *= expf(local_max - maximum);
    __shared__ float warp_sums[BlockSize / kWarpSize];
    __shared__ float log_sum;
    const float sum = block_reduce_sum<BlockSize>(local_sum, warp_sums);
    if (threadIdx.x == 0) { log_sum = logf(sum); }
    __syncthreads();
    return {maximum, log_sum};
}

template <int BlockSize>
__launch_bounds__(BlockSize) __global__
    void target_logprobs_kernel(const __nv_bfloat16* logits, const std::int32_t* target_ids,
                                float* output, std::int32_t valid_rows,
                                std::int32_t physical_rows, std::int32_t targets) {
    const std::int32_t column   = static_cast<std::int32_t>(blockIdx.x);
    const __nv_bfloat16* values = logits + static_cast<std::int64_t>(column) * physical_rows;
    const TargetLogprobsColumn lse = target_logprobs_column<BlockSize>(values, valid_rows);
    for (std::int32_t target = static_cast<std::int32_t>(threadIdx.x); target < targets;
         target += BlockSize) {
        const std::int64_t index = static_cast<std::int64_t>(column) * targets + target;
        output[index] = __bfloat162float(values[target_ids[index]]) - lse.maximum - lse.log_sum;
    }
}

} // namespace infernix::ops
