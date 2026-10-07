// Implements: include/infernix/ops/top_logprobs.h
// Match: wrapper-validated contiguous tensors, valid vocabulary rows and K<=valid_rows.
// Algorithm assumptions: one independent CTA per column; no global workspace.
#include "ops/launcher/top_logprobs.h"

#include "core/device.h"
#include "ops/kernel/top_logprobs.cuh"

namespace infernix::ops::detail {

void top_logprobs_launch(const Tensor& logits, std::int32_t valid_rows, Tensor& ids, Tensor& output,
                         cudaStream_t stream) {
    const auto columns = static_cast<unsigned int>(logits.ne[1]);
    top_logprobs_kernel<kTopLogprobsBlock><<<columns, kTopLogprobsBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(ids.data),
        static_cast<float*>(output.data), valid_rows, logits.ne[0], ids.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
