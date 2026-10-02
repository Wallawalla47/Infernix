#pragma once

// ninfer::ops::detail - private launch prototype for top_logprobs.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void top_logprobs_launch(const Tensor& logits, std::int32_t valid_rows, Tensor& ids, Tensor& output,
                         cudaStream_t stream);

} // namespace ninfer::ops::detail
