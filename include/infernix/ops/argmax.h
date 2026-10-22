#pragma once

#include "core/tensor.h"

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
 * The softmax probability of each column's selected row (a speculative draft's confidence):
 *
 *   probability[t] = 1 / sum_{0 <= v < valid_rows} exp(float(logits[v,t]) - float(logits[rows[t],t])).
 *
 * `logits` as for argmax; `rows` contiguous I32 [T] with 0 <= rows[t] < valid_rows (argmax's `out`
 * before any id mapping); `probability` contiguous FP32 [T]. Each column's sum runs in one fixed
 * order (fast exponentials, FP32 accumulation), so the result depends only on the column's logits:
 * an extent policy's input, not a sampling probability.
 */
void argmax_probability(const Tensor& logits, const Tensor& rows, Tensor& probability, std::int32_t valid_rows,
                        cudaStream_t stream);

} // namespace infernix::ops
