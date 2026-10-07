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
 * `logits` is contiguous BF16 [physical_rows,T], `out` is contiguous I32 [T], and
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

} // namespace infernix::ops
