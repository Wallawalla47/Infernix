#pragma once

// Addressing helpers of the fast prompt kernels: the public Q/output index, the XOR swizzle of
// their shared-memory operand tiles, and the partial rows and merge of a key-split launch.

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kCausalPromptHeadDim = 256;

template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_prompt_q_index(int q_head, int d, int token) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kCausalPromptHeadDim) *
                                              (static_cast<std::int64_t>(q_head) +
                                               static_cast<std::int64_t>(Geometry::QHeads) * token);
}

// XOR-swizzled b16 element address. INT8 operands use the same layout by packing
// two consecutive signed bytes into each b16 lane before ldmatrix.
__device__ __forceinline__ int causal_prompt_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <typename Byte>
__device__ __forceinline__ void causal_prompt_store_byte_swizzled(Byte* tile, int row, int d,
                                                                  Byte code) {
    const int col_b16 = d >> 1;
    const int byte    = d & 1;
    const int off = (row * (kCausalPromptHeadDim / 2) + causal_prompt_swz(row, col_b16)) * 2 + byte;
    tile[off]     = code;
}

// Split partial layout for key split s, column c and query head h: the D256 row at
// ((s * width + c) * QHeads + h) * 256 and its (max, sum) pair at twice that row index.
template <typename Geometry>
__host__ __device__ __forceinline__ std::int64_t
causal_prompt_fast_partial_row(int split, int column, int q_head, int width) {
    return (static_cast<std::int64_t>(split) * width + column) * Geometry::QHeads + q_head;
}

// Combines the key splits of one column and query head: every split row is normalized by its own
// sum, so the merged row is sum_s w_s * row_s / sum_s w_s with w_s = sum_s * 2^((m_s - M) * c).
// Columns past the valid count publish zeros.
template <typename Geometry>
__global__ __launch_bounds__(kCausalPromptHeadDim) void causal_attention_prompt_fast_merge_kernel(
    const float* __restrict__ partial_rows, const float2* __restrict__ partial_stats,
    const std::int32_t* __restrict__ valid_columns, std::int32_t width, std::int32_t splits,
    float scale_l2, __nv_bfloat16* __restrict__ out) {
    const int column = static_cast<int>(blockIdx.x);
    const int q_head = static_cast<int>(blockIdx.y);
    const int d      = static_cast<int>(threadIdx.x);
    const int tokens = valid_columns == nullptr ? width : max(0, min(width, valid_columns[0]));
    const auto index = causal_prompt_q_index<Geometry>(q_head, d, column);
    if (column >= tokens) {
        out[index] = __float2bfloat16(0.0f);
        return;
    }
    float maximum = -CUDART_INF_F;
    for (int split = 0; split < splits; ++split) {
        const float2 stats =
            partial_stats[causal_prompt_fast_partial_row<Geometry>(split, column, q_head, width)];
        if (stats.y > 0.0f) { maximum = fmaxf(maximum, stats.x); }
    }
    float numerator   = 0.0f;
    float denominator = 0.0f;
    for (int split = 0; split < splits; ++split) {
        const std::int64_t row =
            causal_prompt_fast_partial_row<Geometry>(split, column, q_head, width);
        const float2 stats = partial_stats[row];
        if (stats.y > 0.0f) {
            const float weight = stats.y * exp2f((stats.x - maximum) * scale_l2);
            numerator = __fmaf_rn(weight, partial_rows[row * kCausalPromptHeadDim + d], numerator);
            denominator += weight;
        }
    }
    out[index] = __float2bfloat16(denominator > 0.0f ? numerator / denominator : 0.0f);
}

} // namespace ninfer::ops::detail
