#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace infernix::ops {

/**
 * Computes one vocabulary argmax per column:
 *
 *   out[t] = min argmax_{0 <= v < valid_rows} float(logits[v,t]).
 *
 * `logits` is contiguous BF16 or FP32 [physical_rows,T], `out` is contiguous I32 [T], and
 * 1 <= valid_rows <= physical_rows. Physical rows [valid_rows,physical_rows) do not
 * participate. Equal maxima select the lowest row index. `out` must not overlap `logits`.
 * The Op has no workspace and changes no state other than writing all of `out`.
 */
void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);

/**
 * The same argmax over a shortlist head's rows, mapped to token ids: out[t] = row_ids[r_t] for the
 * row r_t the overload above selects. `row_ids` is contiguous I32 [physical_rows].
 */
void argmax(const Tensor& logits, const Tensor& row_ids, Tensor& out, std::int32_t valid_rows,
            cudaStream_t stream);

/**
 * The argmax of each column with its softmax probability (a speculative draft's confidence), from one
 * pass over the logits:
 *
 *   out[t]         = min argmax_{0 <= v < valid_rows} float(logits[v,t])   (as argmax above),
 *   probability[t] = 1 / sum_{0 <= v < valid_rows} exp(float(logits[v,t]) - float(logits[out[t],t])).
 *
 * `logits`, `out` and valid_rows as for argmax (out holds rows: map them to token ids afterwards);
 * the logits are finite; `probability` is contiguous FP32 [T], T <= 65535. Each column's sum runs in
 * one fixed order (fast exponentials and FP32 accumulation; each thread rescales its running sum at
 * a new maximum, then a fixed tree combines the threads and, in row order, the row tiles), so the
 * result depends only on the column's logits: an extent policy's input, not a sampling probability.
 * `workspace` supplies argmax_with_probability_workspace_bytes() of transient storage (the row
 * tiles' partial maxima and sums), free again when the call returns.
 */
[[nodiscard]] std::size_t argmax_with_probability_workspace_bytes(DType logits, std::int32_t physical_rows,
                                                                  std::int32_t valid_rows, std::int32_t columns);
void argmax_with_probability(const Tensor& logits, Tensor& out, Tensor& probability, std::int32_t valid_rows,
                             WorkspaceArena& workspace, cudaStream_t stream);

} // namespace infernix::ops
