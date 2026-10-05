#pragma once

// Implements: include/ninfer/ops/token_constraint.h
// Match: contiguous BF16 [physical_rows,C] logits, I32 descriptors/choices, FP32 records.
// Algorithm assumptions: one 256-thread CTA per column; the full-vocabulary log-sum-exp is the
// target_logprobs single-pass reduction, and the permitted set (at most 16 tokens) is resolved by
// one thread.

#include "ninfer/ops/token_constraint.h"
#include "ops/kernel/target_logprobs.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kTokenConstraintBlock = 256;

template <int BlockSize>
__launch_bounds__(BlockSize) __global__
    void constrain_logits_kernel(__nv_bfloat16* logits, std::int32_t* argmax,
                                 const std::int32_t* descriptors, const std::int32_t* choices,
                                 const std::int32_t* choice_counts, std::int32_t sets,
                                 std::int32_t valid_rows, std::int32_t physical_rows,
                                 float* records) {
    const std::int32_t column     = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t descriptor = descriptors[column];
    if (descriptor == -1) { return; }

    __shared__ std::int32_t permitted[kTokenConstraintChoices];
    __shared__ __nv_bfloat16 saved[kTokenConstraintChoices];
    __shared__ std::int32_t count;
    if (threadIdx.x == 0) {
        std::int32_t n = 0;
        if (descriptor >= 0) {
            n = descriptor < sets ? choice_counts[descriptor] : 0;
            if (n < 1 || n > kTokenConstraintChoices) { n = 0; }
            for (std::int32_t i = 0; i < n; ++i) {
                const std::int32_t token = choices[descriptor * kTokenConstraintChoices + i];
                permitted[i]             = token;
                if (token < 0 || token >= valid_rows) { n = 0; }
            }
        } else {
            const std::int32_t token = -(descriptor + 2);
            permitted[0]             = token;
            n                        = token < valid_rows ? 1 : 0;
        }
        count = n;
    }
    __syncthreads();
    const std::int32_t n = count;
    if (n == 0) { return; }

    __nv_bfloat16* values = logits + static_cast<std::int64_t>(column) * physical_rows;
    const TargetLogprobsColumn full = target_logprobs_column<BlockSize>(values, valid_rows);

    if (threadIdx.x == 0) {
        float maximum = -CUDART_INF_F;
        std::int32_t best = 0;
        for (std::int32_t i = 0; i < n; ++i) {
            saved[i]          = values[permitted[i]];
            const float value = __bfloat162float(saved[i]);
            if (value > maximum || (value == maximum && permitted[i] < permitted[best])) {
                maximum = value;
                best    = i;
            }
        }
        float sum = 0.0f;
        for (std::int32_t i = 0; i < n; ++i) {
            sum += expf(__bfloat162float(saved[i]) - maximum);
        }
        float* record = records + static_cast<std::int64_t>(column) * kTokenConstraintRecord;
        for (std::int32_t i = 0; i < kTokenConstraintChoices; ++i) {
            record[i] = i < n ? expf(__bfloat162float(saved[i]) - maximum) / sum : 0.0f;
        }
        record[kTokenConstraintChoices] =
            fminf(1.0f, expf(maximum + logf(sum) - full.maximum - full.log_sum));
        if (argmax != nullptr) { argmax[column] = permitted[best]; }
    }
    __syncthreads();
    const __nv_bfloat16 negative_infinity = __float2bfloat16(-CUDART_INF_F);
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        values[row] = negative_infinity;
    }
    // Block-scope ordering: the masking stores above are visible before the restores below.
    __syncthreads();
    if (static_cast<std::int32_t>(threadIdx.x) < n) {
        values[permitted[threadIdx.x]] = saved[threadIdx.x];
    }
}

} // namespace ninfer::ops
