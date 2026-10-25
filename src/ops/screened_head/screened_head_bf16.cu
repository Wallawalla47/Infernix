// screened_head_logits' exact rescore of a BF16 head: the listed rows through projection_fp32's
// tensor-core mapping (ops/projection_fp32/projection_fp32_mma.cu), whose Output gathers each tile row
// from the list. An MMA output element depends only on its weight row, its column and the K walk, so
// every listed (row, column) gets the dense call's bits.

#include "ops/screened_head/screened_head_launch.h"

#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_launch.cuh"
#include "ops/linear/common/epilogue.cuh"

#include <cuda_bf16.h>

namespace infernix::ops::detail::screened_head {
namespace {

// Virtual row r is position r % cap of column r / cap's list. Tiles of 32 rows lie inside one column's
// list (cap is a multiple of 32), so a tile past the column's count is skipped whole.
struct ListedRowsOutput {
    float* logits;
    std::int32_t vocab_rows;
    const std::int32_t* entries;
    const std::int32_t* counts;
    std::int32_t cap;

    // The list is the previous kernel's output: read past L1.
    __device__ __forceinline__ std::int64_t weight_row(int row) const {
        return static_cast<std::int64_t>(__ldcg(entries + row));
    }
    __device__ __forceinline__ bool skip_tile(int row0) const {
        return row0 % cap >= __ldcg(counts + row0 / cap);
    }
    // Only the listing column's value is stored: another column keeps its screen score.
    __device__ __forceinline__ void store(int row, int token, float value) const {
        const int column = row / cap;
        if (token == column && row % cap < __ldcg(counts + column)) {
            logits[static_cast<std::int64_t>(token) * vocab_rows + __ldcg(entries + row)] = value;
        }
    }
};

// projection_fp32_mma.cu's schedule.
using HeadSchedule = Bf16A16SlicedR32T16W4;
static_assert(HeadSchedule::kBlockRows == 32 && HeadSchedule::kBlockTokens == 16,
              "the listed rows are gathered in tiles of 32 at up to 16 columns");

} // namespace

void bf16_rescore(const Tensor& x, const void* head_rows, std::int32_t vocab_rows, std::int32_t columns,
                  const std::int32_t* entries, const std::int32_t* counts, std::int32_t cap, float* logits,
                  cudaStream_t stream) {
    const Bf16A16Operands p{static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(head_rows),
                            columns * cap, kHidden, columns};
    launch_bf16_a16_sliced_k_mma<HeadSchedule>(p, ListedRowsOutput{logits, vocab_rows, entries, counts, cap},
                                               LinearIdentityEpilogue{}, stream, pdl::Dependency::Programmatic);
}

} // namespace infernix::ops::detail::screened_head
