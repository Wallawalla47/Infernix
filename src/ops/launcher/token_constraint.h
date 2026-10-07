#pragma once

// infernix::ops::detail - private launch prototype for constrain_logits.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace infernix::ops::detail {

void constrain_logits_launch(Tensor& logits, Tensor* argmax, const Tensor& descriptors,
                             const Tensor& choices, const Tensor& choice_counts,
                             std::int32_t valid_rows, Tensor& records, cudaStream_t stream);

} // namespace infernix::ops::detail
