// logits_topk's producers (the DFlash2 drafters' heads: Inkling 2's and Qwen3.8-Flash-Next's materialized FP32
// logits): one warp a (512-row group, column), each lane holding 16 rows' keys, the group's top sixteen taken as
// sixteen successive warp-wide maxima and published in descending order, as linear_topk's producers publish theirs
// for the merge. The keys' order is total (score, then the lower id), so the candidates are exactly those of the
// 128-row sorting producer (linear_topk_logits_launch), with a quarter of the partial keys to merge; at Inkling's
// 65,536-row shortlist and 7 columns that producer took 15.9 us (a block merge sort a column, one after another).
#include "core/weight.h"
#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "ops/common/score_id_order.cuh"

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr int kPerLane = kLogitsTopKRows / 32;
static_assert(kLogitsTopKRows % 32 == 0 && kPerLane >= 1);

__device__ __forceinline__ std::uint64_t warp_max(std::uint64_t value) {
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        const std::uint64_t other = __shfl_xor_sync(0xffffffffu, value, offset);
        value                     = other > value ? other : value;
    }
    return value;
}

__global__ __launch_bounds__(32) void logits_topk_warp_kernel(const float* __restrict__ logits, std::int32_t rows,
                                                              std::int32_t valid_rows,
                                                              std::uint64_t* __restrict__ partial_keys,
                                                              std::int32_t producer_groups) {
    const int group = static_cast<int>(blockIdx.x), column = static_cast<int>(blockIdx.y);
    const int lane  = static_cast<int>(threadIdx.x);
    const float* values = logits + static_cast<std::int64_t>(column) * rows;
    std::uint64_t key[kPerLane];
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
        const int row = group * kLogitsTopKRows + j * 32 + lane;
        key[j]        = row < valid_rows ? score_id_order_key(values[row], row) : 0; // 0: below every key
    }
    std::uint64_t* out = partial_keys + (static_cast<std::int64_t>(column) * producer_groups + group) * kLinearTopK;
#pragma unroll 1
    for (int i = 0; i < kLinearTopK; ++i) {
        std::uint64_t local = key[0];
#pragma unroll
        for (int j = 1; j < kPerLane; ++j) { local = key[j] > local ? key[j] : local; }
        const std::uint64_t best = warp_max(local);
        if (lane == 0) { out[i] = best; }
        // Keys are unique but for the sentinel: one lane holds the maximum, and drops it.
        if (best != 0 && local == best) {
#pragma unroll
            for (int j = 0; j < kPerLane; ++j) {
                if (key[j] == best) { key[j] = 0; }
            }
        }
    }
}

} // namespace

void logits_topk_warp_launch(const Tensor& logits, std::int32_t valid_rows, const LinearTopKWorkspace& workspace,
                             cudaStream_t stream) {
    if (workspace.rows_per_producer != kLogitsTopKRows || logits.dtype != DType::FP32 ||
        logits.ne[1] != workspace.columns) {
        throw std::logic_error("logits_topk: workspace does not match the call");
    }
    const dim3 grid(static_cast<unsigned>(workspace.producer_groups), static_cast<unsigned>(workspace.columns));
    logits_topk_warp_kernel<<<grid, 32, 0, stream>>>(static_cast<const float*>(logits.data), logits.ne[0], valid_rows,
                                                    static_cast<std::uint64_t*>(workspace.partial_keys.data),
                                                    workspace.producer_groups);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
