#pragma once

// infernix::ops::detail - private launch prototype for argmax.

#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace infernix::ops::detail {

void argmax_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);

// out[t] = row_ids[out[t]] in place.
void argmax_map_launch(const Tensor& row_ids, Tensor& out, cudaStream_t stream);

// argmax_with_probability's row tiles per column (0: one CTA per column, no workspace).
[[nodiscard]] std::int32_t argmax_probability_tiles(DType logits, std::int32_t physical_rows,
                                                    std::int32_t valid_rows, std::int32_t columns);
// Its workspace bytes for that many tiles: the tiles' FP32 maxima and sums and the columns' arrivals.
[[nodiscard]] std::size_t argmax_probability_workspace_bytes(std::int32_t tiles, std::int32_t columns);

// out[t] = argmax, probability[t] = 1 / sum_v exp(logits[v,t] - logits[out[t],t]) (validated by the
// wrapper); `workspace` holds argmax_probability_workspace_bytes for the tiles.
void argmax_probability_launch(const Tensor& logits, Tensor& out, Tensor& probability, std::int32_t valid_rows,
                               DeviceSpan workspace, cudaStream_t stream);

} // namespace infernix::ops::detail
