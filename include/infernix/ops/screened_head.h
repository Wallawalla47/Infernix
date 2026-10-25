#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

/**
 * The selection rule both screened-head Ops share: per column, the head rows whose screen score
 * (below) lies within `margin` of the column's best screen score are rescored exactly, at most
 * `rescored` of them, the highest screen scores first (equal scores: the lower row first), and at
 * least `ranks` (the Op's required rows) even when fewer lie within the margin.
 *
 * The candidates are the screen's per-CTA lists: the screen's g CTAs (one per SM, at most 256) take
 * the head's 16-row items round-robin (item i to CTA i mod g) and each keeps its best 8 rows per
 * column, so a row is a candidate only when fewer than 8 rows of its CTA score above it. A CTA's rows
 * are 16-row runs g * 16 rows apart: more than 8 rows within the margin in one CTA (a run of adjacent
 * near-equal tokens, such as the digits) drops the CTA's lower ones. The candidate set, and with it
 * every output, depends on the device's SM count, never on T or other columns.
 */
struct ScreenedHeadPolicy {
    float margin           = 5.0F;
    std::int32_t rescored  = 64; // a multiple of 32 in [32, 256]
};

// Columns a screened head serves in one call (1..16).
inline constexpr std::int32_t kScreenedHeadMaxColumns = 16;
// The ranks screened_head_top returns per column (1..4).
inline constexpr std::int32_t kScreenedHeadMaxRanks = 4;

/**
 * Caller-owned transient capacity of screened_head_top and screened_head_logits for every column
 * count in [1, max_columns] and every head of at most `head_rows` rows, under `policy`. Invalid
 * intervals or policies throw.
 */
[[nodiscard]] std::size_t screened_head_workspace_bytes(std::int32_t head_rows, std::int32_t max_columns,
                                                        const ScreenedHeadPolicy& policy = {});

/**
 * Op: Screened head top rows
 *
 * Math / indexing:
 *   x is BF16 [K, T]; `head` is the exact head [R, K] and `screen` its 2-bit copy [R, K]. Let
 *     e[r,t] = sum_k w[r,k] * x[k,t]     (the exact head's logit, projection_fp32's arithmetic),
 *     s[r,t] = sum_g S[r,g] * Sx[t,g] * sum_{k in g} c[r,k] * q[k,t]
 *   the screen score, with c and S the copy's signed codes and FP16 group scales, and q[., t] the
 *   column quantized to int8 per 64-group g with FP32 scale Sx[t,g] = max|x[g,t]| / 127. For each
 *   column t let P_t be the rows r < valid_rows the policy selects by s[., t] (with ranks = n).
 *   rows[0..n-1, t] are the n rows of P_t with the largest bf16(e[r,t]), in descending order, equal
 *   values in ascending row order, and
 *     logprobs[j,t] = bf16(e[rows[j,t],t]) - L_t,
 *     L_t = log( sum_{r in P_t} exp(bf16(e[r,t])) + exp(v_t / 2) sum_{r < valid_rows, r not in P_t} exp(s[r,t]) ),
 *   the log-normalizer with the rows outside P_t at their screen scores raised by v_t / 2, the mean of
 *   (bf16(e[r,t]) - s[r,t])^2 over the rows of P_t other than its four largest bf16(e) (ranks order),
 *   capped at 4, or 0 when fewer than 8 such rows remain: the screen's residual e - s is about
 *   independent of s, so E[exp(e) | s] = exp(s + v / 2). When `row_ids` is non-empty,
 *   tokens[t] = row_ids[rows[0,t]].
 *
 * Logical shapes:
 *   x BF16 [K, T] with K = 2560 and T in [1, 16]; head and screen [R, K] with R a positive multiple
 *   of 16; 1 <= valid_rows <= R; rows I32 [n, T], logprobs FP32 [n, T], n = ranks in [1, 4]; row_ids
 *   I32 [R] and tokens I32 [T], both empty or both present.
 *
 * Supported domain:
 *   head: row-split q4_g64_fp16 (a proposal head). screen: row-split q2_g64_fp16 of the same rows.
 *   Weight planes and tensors 16-byte aligned.
 *
 * Numeric:
 *   e[r,t] for every row of P_t carries exactly projection_fp32's bits for that row and column (the
 *   same summation order at every T). rows equal the exact head's top n rows (argmax and
 *   top_logprobs over every row of bf16(e)) whenever those rows lie in P_t; s[., t] and with it P_t
 *   depend on column t alone. L_t is an approximation of the exact head's log-normalizer.
 *
 * Effects:
 *   Writes rows, logprobs and (when present) tokens; preserves every input. Outputs overlap neither
 *   the inputs nor each other.
 *
 * Workspace:
 *   screened_head_workspace_bytes(R, T, policy).
 *
 * Execution:
 *   A fixed launch sequence for each T: capturable in a CUDA Graph and replayable.
 */
void screened_head_top(const Tensor& x, const Weight& head, const Weight& screen, std::int32_t valid_rows,
                       const ScreenedHeadPolicy& policy, Tensor& rows, Tensor& logprobs, const Tensor& row_ids,
                       Tensor& tokens, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Op: Screened head logits
 *
 * Math / indexing:
 *   With e, s and P_t as for screened_head_top (P_t chosen among the rows r < valid_rows, ranks 1):
 *     logits[r,t] = e[r,t] for r in P_t, s[r,t] otherwise.
 *
 * Logical shapes:
 *   x BF16 [K, T] with K = 2560 and T in [1, 16]; head [R, K] and screen [R, K] with R a positive
 *   multiple of 16; logits FP32 [R, T]; 1 <= valid_rows <= R.
 *
 * Supported domain:
 *   head: BF16 rows (`bf16_head`, row-major [R, K]: a BF16 tensor [K, R]) with `head` empty, or a
 *   row-split q8_g32_fp16 `head` with `bf16_head` empty. screen: row-split q2_g64_fp16 of the same
 *   rows.
 *
 * Numeric:
 *   For r in P_t, logits[r,t] carries exactly projection_fp32's bits for that row and column (its
 *   route for a T-column call). The argmax over r < valid_rows of bf16(logits[., t]) equals the
 *   exact head's whenever its row lies in P_t and no screen score outside P_t exceeds it.
 *
 * Effects:
 *   Writes every element of logits; preserves every input.
 *
 * Workspace:
 *   screened_head_workspace_bytes(R, T, policy).
 *
 * Execution:
 *   A fixed launch sequence for each T: capturable in a CUDA Graph and replayable.
 */
void screened_head_logits(const Tensor& x, const Tensor& bf16_head, const Weight& head, const Weight& screen,
                          std::int32_t valid_rows, const ScreenedHeadPolicy& policy, Tensor& logits,
                          WorkspaceArena& workspace, cudaStream_t stream);

} // namespace infernix::ops
