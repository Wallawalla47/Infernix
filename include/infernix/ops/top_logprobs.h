#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace infernix::ops {

/**
 * Op: Most probable token log-probabilities
 *
 * Math / indexing:
 *   Let l[r,c] be the exact real value represented by logits[r,c] and
 *
 *     lse[c] = log(sum_{r=0..valid_rows-1} exp(l[r,c])).
 *
 *   For every column c, ids[0..K-1,c] are the K rows r<valid_rows of largest l[r,c], in
 *   descending order of l[r,c] with equal values in ascending order of r, and
 *
 *     ideal[k,c] = l[ids[k,c],c] - lse[c].
 *
 * Logical shapes:
 *   logits is [physical_rows,C], and ids and output are [K,C], with C>0, 1<=K<=valid_rows and
 *   1<=valid_rows<=physical_rows. Physical rows [valid_rows,physical_rows) participate in neither
 *   the selection nor lse.
 *
 * Supported domain:
 *   logits is contiguous finite BF16, ids is contiguous I32, and output is contiguous FP32.
 *   Storage has its dtype's natural alignment.
 *
 * Numeric:
 *   ids is exact. output is the FP32 numerical approximation of ideal; reduction association and
 *   private accumulator precision are implementation choices, and the independent oracle
 *   evaluates lse in FP64 from the represented BF16 inputs.
 *
 * Effects:
 *   Writes every ids and output element and preserves logits. Neither output overlaps logits or
 *   the other output.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues work on stream and owns no persistent state.
 */
void top_logprobs(const Tensor& logits, std::int32_t valid_rows, Tensor& ids, Tensor& output,
                  cudaStream_t stream);

} // namespace infernix::ops
