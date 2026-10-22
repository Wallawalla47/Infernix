// Implements: include/infernix/ops/top_logprobs.h
// Match: wrapper-validated contiguous tensors, valid vocabulary rows and K<=valid_rows.
// Algorithm assumptions: one independent CTA per column, K-pass selection up to
// kTopLogprobsSerialMaximum and above kTopLogprobsSelectMaximum, radix selection between; no
// global workspace.
#include "ops/launcher/top_logprobs.h"

#include "core/device.h"
#include "ops/kernel/top_logprobs.cuh"

namespace infernix::ops::detail {

void top_logprobs_launch(const Tensor& logits, std::int32_t valid_rows, Tensor& ids, Tensor& output,
                         cudaStream_t stream) {
    const auto columns = static_cast<unsigned int>(logits.ne[1]);
    const auto count   = ids.ne[0];
    const bool select  = count > kTopLogprobsSerialMaximum && count <= kTopLogprobsSelectMaximum;
    const auto launch  = [&](const auto* values) {
        auto* const top_ids = static_cast<std::int32_t*>(ids.data);
        auto* const top     = static_cast<float*>(output.data);
        if (select) {
            top_logprobs_select_kernel<kTopLogprobsBlock><<<columns, kTopLogprobsBlock, 0, stream>>>(
                values, top_ids, top, valid_rows, logits.ne[0], count);
        } else {
            top_logprobs_kernel<kTopLogprobsBlock><<<columns, kTopLogprobsBlock, 0, stream>>>(
                values, top_ids, top, valid_rows, logits.ne[0], count);
        }
    };
    if (logits.dtype == DType::FP32) {
        launch(static_cast<const float*>(logits.data));
    } else {
        launch(static_cast<const __nv_bfloat16*>(logits.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
