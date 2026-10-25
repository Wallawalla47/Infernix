#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace infernix::ops::detail {

// A column-invariant verification round runs one column at a time (causal_softmax_attention.cpp):
// flags [width][batch] receives 1 where column j of row b is valid (j < counts[b]), else 0, so
// each column's one-column call masks the rows that end before it.
void causal_column_valid_launch(const Tensor& counts, Tensor& flags, std::int32_t width,
                                std::int32_t batch, cudaStream_t stream);

// The rows variant of that round (INFERNIX_TMP_VERIFY_ROWS_ATTENTION) runs every column as a row of
// one launch: rows [width][batch] receives row b's KV table row (table_rows[b], or 0 when the call
// has no table rows) for each column, so a column-major row reads its round row's pages.
void causal_column_table_rows_launch(const Tensor& table_rows, Tensor& rows, std::int32_t width,
                                     std::int32_t batch, cudaStream_t stream);

// One tensor of a verification round whose rows (one per token, `row_bytes` each, a multiple of
// 4) are reordered between the round's row-major order (row b * width + column j) and column-major
// order (column j * batch + row b), so each column's rows are contiguous.
struct CausalColumnPermutation {
    const void* source      = nullptr;
    void* destination       = nullptr;
    std::int32_t row_bytes  = 0;
};

// Reorders up to four tensors of a [width, batch] round in one launch: to column-major with
// `to_columns`, else back to row-major. The column-by-column loop gathers its inputs once and
// scatters its output once, so a round costs a CUDA Graph two copy nodes, not five per column.
void causal_columns_permute_launch(const CausalColumnPermutation* tensors, std::int32_t count,
                                   std::int32_t width, std::int32_t batch, bool to_columns,
                                   cudaStream_t stream);

} // namespace infernix::ops::detail
