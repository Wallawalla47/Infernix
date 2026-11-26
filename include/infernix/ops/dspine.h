#pragma once

// Ported from NInfer-all (https://github.com/iamwavecut/ninfer-all, PRs #341 and #365; Apache-2.0) and modified
// for Infernix (renamed into Infernix's Op contract and its oracle test, the proposal draw sub-keyed by step
// with the Gumbel-max coupled draw: docs/maintainer/dspine.md).

// The Ops of the DSpine code bridge (Tencent DSpine, docs/maintainer/dspine.md): the write that
// follows every drafter layer, and the serial refinement of the last write that turns a block's
// candidates into a chain of drafts. Each Op states its rounding points; the independent oracle
// evaluates the formula in FP64 from the represented inputs and compares every stored BF16 or
// FP32 value under the Op's tolerance, except the selected ids, which are exact.

#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

/**
 * Op: dspine_bridge_read
 *
 * For every column n of raw, contiguous BF16 [H,N] with H a positive multiple of 8:
 *
 *   rms[n]          = sqrt(max(mean_h raw[h,n]^2, FLT_MIN))           (FP32 [N])
 *   normalized[h,n] = BF16(raw[h,n] / (rms[n] + eps))                (BF16 [H,N])
 *
 * eps is positive and finite. Outputs are contiguous, completely overwritten and do not overlap
 * raw or each other. No workspace.
 */
void dspine_bridge_read(const Tensor& raw, float eps, Tensor& normalized, Tensor& rms,
                        cudaStream_t stream);

/**
 * Op: dspine_bridge_codes
 *
 * For every column n of projected, contiguous BF16 [C,N] with C a positive multiple of 8:
 *
 *   codes[c,n] = BF16(sqrt(C) * projected[c,n] / (sqrt(sum_c projected[c,n]^2) + eps))
 *
 * codes is contiguous BF16 [C,N], completely overwritten, and may alias projected exactly.
 * No workspace.
 */
void dspine_bridge_codes(const Tensor& projected, float eps, Tensor& codes, cudaStream_t stream);

/**
 * Op: dspine_bridge_message
 *
 * packed is contiguous BF16 [3R,W,B] (W in [1,16], B in [1,8], R a positive multiple of 32 up to
 * 512): rows [0,R) are each column's receiver query q, [R,2R) its sender gate s and [2R,3R) its
 * sender value v. Column 0 of a block is its anchor, whose sender rows are those of the anchor's
 * token in sender_table, contiguous BF16 [2R,V] ([s;v] of every token id), not packed's.
 * gate_bias is contiguous BF16 [R] and anchors contiguous I32 [B] with ids in [0,V).
 *
 *   message[r,0,b] = 0
 *   message[r,i,b] = BF16(sigmoid(q[r,i,b] + s[r,i-1,b] + gate_bias[r]) * v[r,i-1,b])   (i >= 1)
 *
 * with the column-0 sender rows read from sender_table[anchors[b]]. message is contiguous BF16
 * [R,W,B], completely overwritten, and overlaps no input. No workspace.
 */
void dspine_bridge_message(const Tensor& packed, const Tensor& anchors, const Tensor& sender_table,
                           const Tensor& gate_bias, Tensor& message, cudaStream_t stream);

/**
 * Op: dspine_bridge_finish
 *
 * The write after a drafter layer and the norm of the next layer's input. raw and delta are
 * contiguous BF16 [H,W,B] (H a positive multiple of 8, W in [1,16], B in [1,8]), rms contiguous
 * FP32 [W,B] (dspine_bridge_read's), norm_weight contiguous BF16 [H], eps positive:
 *
 *   out[h,0,b]    = raw[h,0,b]                                         (the anchor keeps its row)
 *   out[h,i,b]    = BF16(raw[h,i,b] + rms[i,b] * delta[h,i,b])          (i >= 1)
 *   y[h,i,b]      = BF16(out[h,i,b] / sqrt(mean_h out[h,i,b]^2 + eps))
 *   normed[h,i,b] = BF16(y[h,i,b] * norm_weight[h])
 *
 * and, when denominator is non-null (contiguous FP32 [W,B]), denominator[i,b] =
 * sqrt(mean_h out[h,i,b]^2 + eps). out may alias raw exactly; normed overlaps nothing else.
 * Every output is completely overwritten. No workspace.
 */
void dspine_bridge_finish(const Tensor& raw, const Tensor& rms, const Tensor& delta,
                          const Tensor& norm_weight, float eps, Tensor& out, Tensor& normed,
                          Tensor* denominator, cudaStream_t stream);

/**
 * Op: dspine_scale_rows
 *
 * out[h,n] = BF16(rows[h,n] * weight[h]) for contiguous BF16 rows/out [H,N] and weight [H], H a
 * positive multiple of 8. out may alias rows exactly. No workspace. The head-write table's input:
 * each column of the last write (rows: the write transposed) scaled by the drafter's final norm.
 */
void dspine_scale_rows(const Tensor& rows, const Tensor& weight, Tensor& out, cudaStream_t stream);

/**
 * Op: dspine_store_head_write
 *
 * head_write[first + j, v] = BF16(logits[v, j]) for logits contiguous BF16 or FP32 [V,N] and
 * head_write contiguous BF16 [R,V] (R a multiple of 32 up to 512, first + N <= R): N columns of the
 * target head's output over dspine_scale_rows' columns become N entries of every token's
 * head-write row, the table dspine_path reads. A BF16 value is copied exactly, an FP32 one rounded
 * to nearest even. Other entries are unchanged. The two do not overlap. No workspace.
 */
void dspine_store_head_write(const Tensor& logits, std::int32_t first, Tensor& head_write,
                             cudaStream_t stream);

// Transient bytes of dspine_path for K draft rows and B batch rows (K in [1,15], B in [1,8]).
[[nodiscard]] std::size_t dspine_path_workspace_capacity_bytes(std::int32_t max_steps,
                                                               std::int32_t max_batch);

inline constexpr std::int32_t kDSpineCandidates = 16;

/**
 * Op: dspine_path
 *
 * The serial refinement of a block's last bridge write (DSpine's exact sender), for K in [1,15]
 * draft rows of B in [1,8] blocks of width W >= K + 1. Inputs, all contiguous:
 *
 *   candidate_ids    I32  [16,K,B]  distinct ids in [0,V) per draft row, or id 0 at score -inf
 *   candidate_scores FP32 [16,K,B]  their logits u under the target's head at the block's
 *                                   final-norm output (candidate_topk's scores)
 *   denominator      FP32 [W,B]     d, that final norm's denominator (dspine_bridge_finish)
 *   rms              FP32 [W,B]     the last write's scale (dspine_bridge_read)
 *   packed           BF16 [3R,W,B]  the last write's packed rows (q in rows [0,R))
 *   message          BF16 [R,W,B]   the last write's message (dspine_bridge_message)
 *   head_write       BF16 [R,V]     column v: the head's row v times the final norm's weight,
 *                                   through the last write (dspine_store_head_write)
 *   sender_table     BF16 [2R,V]    [s;v] of every token id (dspine_bridge_message)
 *   gate_bias        BF16 [R]
 *   anchors, base_positions I32 [B];  configs: device SamplingConfig[B]
 *
 * Draft row i of block b is block column n = i + 1. Its edge to candidate c given sender token t
 * replaces the block's own message by the one sender t sends:
 *
 *   m_t[r]     = sigmoid(q[r,n,b] + sender_table[t,r] + gate_bias[r]) * sender_table[t,R+r]
 *   edge(t, c) = d[n,b] u[c,i,b] + rms[n,b] sum_r (m_t[r] - message[r,n,b]) head_write[id_c, r]
 *
 * in FP32, id_c = candidate_ids[c,i,b]; a candidate of score -inf has edge -inf. The sender of row
 * 0 is anchors[b]; the sender of row i > 0 is the draft of row i - 1. A row with
 * configs[b].temperature <= 0 takes the candidate of greatest edge (the lowest id on a tie) and
 * writes its one-hot proposal_q. A positive-temperature row takes p = softmax(edge / (d[n,b] *
 * temperature)) over the 16 candidates, keeps the candidates that the joint top_k / top_p rule and
 * min_p keep (rank by p, the lower candidate slot first on a tie: kept when its rank is below
 * top_k and, with top_p < 1, the mass ranked before it at most top_p, and, with min_p > 0, when p
 * is at least min_p times the largest p; the first-ranked candidate of nonzero p always), writes q
 * = the kept p renormalized as proposal_q, and draws the candidate whose cumulative q (in slot
 * order) first exceeds u * the total, u = uniform(seed, base_positions[b] + i,
 * kSamplePurposeDSpineProposal, i) (the step i as sub-key: a later round's draft at the same position has a
 * smaller step, so it never reuses this uniform), the last kept slot at most; a gumbel_max row instead takes the
 * kept candidate of the greatest sampling_gumbel_score(q, seed, base_positions[b] + i + 1, id), the lower slot on
 * a tie (the draw its verification's own noise favours). The drawn id is drafts[i,b] and the
 * next row's sender. With confidence (FP32 [K,B]), the Op also stores the drawn candidate's
 * probability under softmax(edge / d[n,b]).
 *
 * drafts I32 [K,B], proposal_q FP32 [16,K,B]: completely overwritten. Inputs, outputs and the
 * config array are pairwise non-overlapping; the workspace is transient. R is a positive multiple
 * of 32 up to 512 and every id in [0,V). The Op ignores the other SamplingConfig fields.
 */
void dspine_path(const Tensor& candidate_ids, const Tensor& candidate_scores,
                 const Tensor& denominator, const Tensor& rms, const Tensor& packed,
                 const Tensor& message, const Tensor& head_write, const Tensor& sender_table,
                 const Tensor& gate_bias, const Tensor& anchors, const Tensor& base_positions,
                 const SamplingConfig* configs, Tensor& drafts, Tensor& proposal_q,
                 Tensor* confidence, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace infernix::ops
