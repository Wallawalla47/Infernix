// The tensor-core route of projection_fp32 for BF16 vocabulary heads at 1..16 columns: the BF16 sliced-K
// MMA tile with an FP32 output. BF16 products are exact in FP32 and the MMA accumulates in FP32, so the
// Op's bound holds; one route serves every width, so a column's bits do not depend on the call's width
// (up to 16). The head's logits cost one read of its weights at every width: 701-708 us for
// [248320, 2560] against the narrow SIMT mapping's 1.05 ms at 8 columns and 1.5 ms at 9-16.

#include "ops/projection_fp32/projection_fp32_mma.h"

#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_launch.cuh"
#include "ops/linear/common/epilogue.cuh"

namespace infernix::ops::detail {
namespace {

struct ProjectionFp32Output {
    float* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        data[static_cast<std::int64_t>(token) * rows + row] = value;
    }
};

using HeadSchedule = Bf16A16SlicedR32T16W4;

} // namespace

bool projection_fp32_bf16_mma_supported(int rows, int k, int columns) noexcept {
    return columns >= 1 && columns <= kProjectionMmaMaxColumns && rows % HeadSchedule::kBlockRows == 0 &&
           k % HeadSchedule::kBlockK == 0;
}

void projection_fp32_bf16_mma(const Tensor& x, const void* weight, int rows, Tensor& out, cudaStream_t stream) {
    const Bf16A16Operands p{static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight),
                            rows, static_cast<int>(x.ne[0]), static_cast<int>(x.ne[1])};
    launch_bf16_a16_sliced_k_mma<HeadSchedule>(p, ProjectionFp32Output{static_cast<float*>(out.data), rows},
                                               LinearIdentityEpilogue{}, stream);
}

} // namespace infernix::ops::detail
