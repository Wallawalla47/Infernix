#pragma once

// infernix::ops::detail - private launch prototype for argmax.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

void argmax_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);

// out[t] = row_ids[out[t]] in place.
void argmax_map_launch(const Tensor& row_ids, Tensor& out, cudaStream_t stream);

} // namespace infernix::ops::detail
