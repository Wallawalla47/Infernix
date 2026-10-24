#pragma once

// Private launch declarations of the screened-head Ops (include/infernix/ops/screened_head.h).

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail::screened_head {

inline constexpr std::int32_t kHidden     = 2560; // the head width the screen kernel is built for
inline constexpr std::int32_t kListLength = 8;    // screen candidates kept per CTA and column
inline constexpr std::int32_t kMaxGrid    = 256;  // screen CTAs at most (one per SM)

// The exact rescore of a BF16 head's listed rows at 1..16 columns: projection_fp32's tensor-core
// mapping (the BF16 sliced-K MMA tile) over the listed rows. entries [columns][cap] hold each column's
// rows, counts [columns] how many are live; logits [vocab_rows, columns] receives row entries[c][i] of
// column c for i < counts[c]. Launched as a programmatic dependent of the selection.
void bf16_rescore(const Tensor& x, const void* head_rows, std::int32_t vocab_rows, std::int32_t columns,
                  const std::int32_t* entries, const std::int32_t* counts, std::int32_t cap, float* logits,
                  cudaStream_t stream);

} // namespace infernix::ops::detail::screened_head
