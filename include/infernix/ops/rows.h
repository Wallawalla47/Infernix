#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace infernix::ops {

/// Splits the rows of a contiguous BF16 [R, T] tensor into contiguous BF16 [r_i, T] tensors in
/// order: output i receives rows [sum_{j<i} r_j, sum_{j<=i} r_j). Exact copy; at most 8 outputs.
void split_rows(const Tensor& in, std::span<Tensor* const> outs, cudaStream_t stream);

/// out[:, i] = in[:, columns[i]] for contiguous BF16 [R, T] in and [R, N] out; columns is I32 [N].
void gather_columns(const Tensor& in, const Tensor& columns, Tensor& out, cudaStream_t stream);

} // namespace infernix::ops
