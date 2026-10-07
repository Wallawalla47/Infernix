#pragma once

#include "core/tensor.h"
#include "core/arena.h"
#include "infernix/ops/sampling.h"
#include "infernix/ops/speculative_tree.h"

#include <cuda_runtime.h>

namespace infernix::ops {

// Capacity for every K/B pair in the inclusive intervals, K=1..15 and B=1..8.
[[nodiscard]] std::size_t candidate_selector_path_workspace_capacity_bytes(std::int32_t min_steps,
                                                                           std::int32_t max_steps,
                                                                           std::int32_t min_batch,
                                                                           std::int32_t max_batch);

/**
 * Op: candidate_selector_path
 *
 * For K in [1,15] and B in [1,8], the inputs are contiguous candidate_ids I32 [16,K,B],
 * unary_scores FP32 [16,K,B], projected_hidden BF16 [256,K,B], anchors I32 [B],
 * predecessor_codebook and successor_codebook BF16 [256,248320], base_positions I32 [B], and a
 * device-resident SamplingConfig[B]. Candidate rank is the fastest axis. The 16 candidate ids in
 * each row are distinct, and all candidate and anchor token ids lie in [0,248077); the registered
 * vocabulary, artifact binding, and linear_topk producer establish that trusted value contract.
 *
 * Starting with predecessor=anchors[b], each position i in [0,K) computes:
 *
 *   edge[c] = unary_scores[c,i,b]
 *           + sum_r predecessor_codebook[r,predecessor]
 *                   * projected_hidden[r,i,b]
 *                   * successor_codebook[r,candidate_ids[c,i,b]].
 *
 * A row with configs[b].temperature<=0 selects the lowest candidate rank attaining max(edge) and
 * writes its exact one-hot distribution. A positive-temperature row writes the FP32 softmax of
 * edge/temperature, then draws a candidate with counter key
 * (configs[b].seed,base_positions[b]+i,kSamplePurposeDFlash2Proposal). The selected global id is
 * written to drafts[i,b] and becomes the next predecessor. The Op ignores all other
 * SamplingConfig fields and never updates token_counts.
 *
 * drafts is contiguous I32 [K,B] and proposal_q is contiguous FP32 [16,K,B]. Both outputs are
 * completely overwritten. Inputs, outputs, codebooks, and the config array must be pairwise
 * non-overlapping. The Op has no persistent state or internal allocation. Caller workspace is
 * transient and must not overlap any input or output.
 */
void candidate_selector_path(const Tensor& candidate_ids, const Tensor& unary_scores,
                             const Tensor& projected_hidden, const Tensor& anchors,
                             const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                             const Tensor& base_positions, const SamplingConfig* configs,
                             Tensor& drafts, Tensor& proposal_q, WorkspaceArena& workspace,
                             cudaStream_t stream);

/**
 * Op: candidate_selector_tree
 *
 * Builds each row's verification tree over the K-position lattice of candidate_selector_path
 * (K in [1,15], B in [1,8], inputs as candidate_selector_path; shape.main_depth == K). W =
 * shape.nodes is the round's verification width. For a node v at depth i-1 the law of its
 * children over the 16 candidates of position i is q_v = softmax(edge(v's rank, .)/temperature),
 * the path walk's conditional law (the anchor row for v = 0).
 *
 * Every row first draws the main chain exactly as candidate_selector_path does (columns 1..K, bit
 * for bit). A row with current_extents[b] == W-1 is a tree row: it then adds side columns K+1..
 * best first until it holds W columns or no node can take another child. The next child of node
 * v has priority score(v) * E[law value of that child], where score(v) is the product of the law
 * values along v's path and the expectation is taken over the child's draw given v's earlier
 * children (the best untaken rank for greedy rows, which score with the temperature-1 law). The
 * winner's next child is drawn without replacement from q_v restricted to its untaken ranks with
 * counter key (seed, base+i-1, kSamplePurposeDFlash2Proposal, ((v+1) << 5) | j) for sibling j;
 * greedy rows take the best untaken rank. A child never exceeds depth K and a row never exceeds
 * shape.paths leaves. Since a child's priority is fixed before it is drawn, the tree's shape never
 * depends on a column's own token, which keeps recursive rejection sampling exact.
 *
 * Outputs: drafts I32 [W-1,B]; column_candidates I32 [16,W-1,B] (the lattice candidates of the
 * column's position); proposal_q FP32 [16,W-1,B] holding the full q of the column's parent (a
 * one-hot of the drawn rank for greedy rows); tree_rows I32 [kSpeculativeTreeRowWords,B] (one
 * SpeculativeTreeRow per row: a chain row's nodes are its extent+1 chain columns); tree_masks I32
 * [W,B] (ancestor masks: bit a of column c is set iff a is c or an ancestor of c; columns past a
 * row's nodes see only themselves); valid_columns I32 [B] is set to each tree row's node count.
 */
void candidate_selector_tree(const Tensor& candidate_ids, const Tensor& unary_scores,
                             const Tensor& projected_hidden, const Tensor& anchors,
                             const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                             const Tensor& base_positions, const Tensor& current_extents,
                             const SamplingConfig* configs, SpeculativeTreeShape shape,
                             Tensor& drafts, Tensor& column_candidates, Tensor& proposal_q,
                             Tensor& tree_rows, Tensor& tree_masks, Tensor& valid_columns,
                             WorkspaceArena& workspace, cudaStream_t stream);

[[nodiscard]] std::size_t candidate_selector_tree_workspace_capacity_bytes(std::int32_t main_depth,
                                                                           std::int32_t min_batch,
                                                                           std::int32_t max_batch);

} // namespace infernix::ops
