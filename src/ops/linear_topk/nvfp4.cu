// linear_topk over Qwen3.5-122B-A10B's head (nvfp4_rg16_v1 [248320, 3072]): the head's canonical
// W4A16 projection (rg16_linear, exact sums, so the scores are the target head's own FP32 logits)
// into the call's workspace, the head read once for every column; then each 128-row producer group's
// top sixteen per column, published as the other heads' producers publish theirs for the merge.
#include "core/weight.h"
#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "infernix/ops/rg16_linear.h"
#include "ops/common/score_id_order.cuh"

#include <cub/block/block_merge_sort.cuh>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr int kThreads = kLinearTopKGroupedRows; // one row per thread
using GroupSort        = cub::BlockMergeSort<std::uint64_t, kThreads, 1>;

// One CTA per producer group: for every column, its 128 rows' keys (rows past valid_rows the
// sentinel) sorted, the first sixteen published.
__global__ __launch_bounds__(kThreads) void nvfp4_logits_topk_kernel(const float* __restrict__ logits,
                                                                     std::int32_t rows, std::int32_t valid_rows,
                                                                     std::uint64_t* __restrict__ partial_keys,
                                                                     std::int32_t producer_groups, std::int32_t columns) {
    __shared__ typename GroupSort::TempStorage storage;
    const int group = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    const int row   = group * kThreads + tid;
    for (int column = 0; column < columns; ++column) {
        std::uint64_t key[1] = {
            row < valid_rows ? score_id_order_key(logits[static_cast<std::int64_t>(column) * rows + row], row) : 0};
        GroupSort(storage).Sort(key, ScoreIdOrderGreater{});
        if (tid < kLinearTopK) {
            partial_keys[(static_cast<std::int64_t>(column) * producer_groups + group) * kLinearTopK + tid] = key[0];
        }
        __syncthreads(); // the sort's storage is reused by the next column
    }
}

} // namespace

void linear_topk_nvfp4_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                              const LinearTopKWorkspace& workspace, Tensor& logits, WorkspaceArena& arena,
                              cudaStream_t stream) {
    if (workspace.rows_per_producer != kThreads || logits.dtype != DType::FP32 || logits.ne[0] != head.n ||
        logits.ne[1] != hidden.ne[1]) {
        throw std::logic_error("linear_topk NVFP4: workspace does not match the call");
    }
    rg16_linear(hidden, head, logits, arena, stream);
    nvfp4_logits_topk_kernel<<<workspace.producer_groups, kThreads, 0, stream>>>(
        static_cast<const float*>(logits.data), head.n, valid_rows,
        static_cast<std::uint64_t*>(workspace.partial_keys.data), workspace.producer_groups, workspace.columns);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
