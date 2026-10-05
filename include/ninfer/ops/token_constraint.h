#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Most tokens one constrained column may permit.
inline constexpr std::int32_t kTokenConstraintChoices = 16;
// Probability record per constrained column: the restricted distribution over the permitted
// tokens in their declared order, padded with zeros, then the permitted tokens' total mass.
inline constexpr std::int32_t kTokenConstraintRecord = kTokenConstraintChoices + 1;

/**
 * Op: Constrain logits to a permitted token set
 *
 * Math / indexing:
 *   Each column c of logits has a descriptor d = descriptors[c]:
 *     d == -1   the column is unconstrained and every output for it is left untouched;
 *     d >= 0    the permitted tokens are P = choices[0:n,d] with n = choice_counts[d];
 *     d <= -2   the column is forced: P = {-(d+2)}.
 *   Let l[r] be the represented value of logits[r,c] and V = valid_rows. For a constrained
 *   column:
 *
 *     records[i,c]  = exp(l[P_i]) / sum_j exp(l[P_j])              for i < n, else 0;
 *     records[16,c] = sum_j exp(l[P_j]) / sum_{r<V} exp(l[r]);
 *     argmax[c]     = P_k maximizing l[P_k], the smallest token id among equal maxima;
 *     logits[r,c]   = -inf for r < V with r not in P; permitted and r >= V rows are unchanged.
 *
 *   So a sampler reading the column afterwards (greedy or stochastic, with or without penalties)
 *   can only draw a permitted token, and a greedy fast path reading argmax agrees with it.
 *
 * Logical shapes:
 *   logits is [physical_rows,C], descriptors and argmax are [C], choices is
 *   [kTokenConstraintChoices,S] and choice_counts is [S] with S>0, and records is
 *   [kTokenConstraintRecord,C].
 *
 * Supported domain:
 *   logits is contiguous finite BF16; descriptors, choices, choice_counts and argmax are
 *   contiguous I32; records is contiguous FP32. Every descriptor is -1, below -1 naming a token in
 *   [0,V), or a set index in [0,S); every used count is in [1,16] and its tokens are distinct
 *   members of [0,V). A column whose descriptor or set breaks this is left untouched.
 *
 * Numeric:
 *   records are FP32 approximations of the formulas above from the represented BF16 logits; the
 *   independent oracle evaluates them in FP64. argmax and the masked logits are exact.
 *
 * Effects:
 *   Mutates logits in place; writes argmax (when non-null) and records for constrained columns
 *   only. No output may overlap another argument.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues work on stream and owns no persistent state. An unconstrained column costs one
 *   descriptor load, so the Op may sit in every captured round.
 */
void constrain_logits(Tensor& logits, Tensor* argmax, const Tensor& descriptors,
                      const Tensor& choices, const Tensor& choice_counts, std::int32_t valid_rows,
                      Tensor& records, cudaStream_t stream);

} // namespace ninfer::ops
