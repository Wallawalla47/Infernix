#pragma once

// infernix::ops::detail - private launch prototype for argmax.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

void argmax_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);

// out[t] = row_ids[out[t]] in place.
void argmax_map_launch(const Tensor& row_ids, Tensor& out, cudaStream_t stream);

// probability[t] = 1 / sum_v exp(logits[v,t] - logits[rows[t],t]) (validated by the wrapper).
void argmax_probability_launch(const Tensor& logits, const Tensor& rows, Tensor& probability,
                               std::int32_t valid_rows, cudaStream_t stream);

} // namespace infernix::ops::detail
