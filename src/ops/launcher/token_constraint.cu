// Implements: include/ninfer/ops/token_constraint.h
// Match: wrapper-validated contiguous tensors and valid vocabulary rows.
// Algorithm assumptions: one independent CTA per column; no global workspace.
#include "ops/launcher/token_constraint.h"

#include "core/device.h"
#include "ops/kernel/token_constraint.cuh"

namespace ninfer::ops::detail {

void constrain_logits_launch(Tensor& logits, Tensor* argmax, const Tensor& descriptors,
                             const Tensor& choices, const Tensor& choice_counts,
                             std::int32_t valid_rows, Tensor& records, cudaStream_t stream) {
    const auto columns = static_cast<unsigned int>(logits.ne[1]);
    constrain_logits_kernel<kTokenConstraintBlock><<<columns, kTokenConstraintBlock, 0, stream>>>(
        static_cast<__nv_bfloat16*>(logits.data),
        argmax != nullptr ? static_cast<std::int32_t*>(argmax->data) : nullptr,
        static_cast<const std::int32_t*>(descriptors.data),
        static_cast<const std::int32_t*>(choices.data),
        static_cast<const std::int32_t*>(choice_counts.data), choice_counts.ne[0], valid_rows,
        logits.ne[0], static_cast<float*>(records.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
