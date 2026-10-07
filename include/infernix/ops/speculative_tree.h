#pragma once

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "infernix/ops/sampling.h"
#include "infernix/ops/speculative_round.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

inline constexpr int kSpeculativeTreeMaxNodes      = 32;
inline constexpr int kSpeculativeTreeMaxPaths      = 8;
inline constexpr int kSpeculativeTreeMaxPathLength = 16;

/**
 * One verification row's draft tree, built on the device every round by candidate_selector_tree
 * and read by the tree forms of the verification Ops below and of attention, GDN and acceptance.
 *
 * Columns 0..nodes-1 of the row are live. Column 0 is the anchor. Every other column c has
 * parent[c] < c, depth[c] = depth[parent[c]] + 1 and holds one proposed token. Columns
 * 0..main_depth form the main chain (parent[c] = c-1, the chain proposal) and every other column
 * lies after it at a depth of at most main_depth. Children are linked in draw order through
 * first_child/next_sibling (-1 terminates); a main column's main-chain child is its first child,
 * and sibling_index is a column's position among its siblings.
 *
 * paths lists every root-to-leaf column sequence in increasing leaf column order, so the main
 * chain is path 0. A column belongs to the first path that contains it: path p owns
 * path_columns[p][owned_from[p] .. path_length[p]-1].
 *
 * tree is 0 for a chain row (a row of a tree round whose extent is below the tree's: its live
 * columns are its chain prefix and no other column is used) and 1 for a tree row.
 */
struct SpeculativeTreeRow {
    std::int32_t nodes;
    std::int32_t main_depth;
    std::int32_t paths;
    std::int32_t tree;
    std::int8_t parent[kSpeculativeTreeMaxNodes];
    std::int8_t depth[kSpeculativeTreeMaxNodes];
    std::int8_t first_child[kSpeculativeTreeMaxNodes];
    std::int8_t next_sibling[kSpeculativeTreeMaxNodes];
    std::int8_t sibling_index[kSpeculativeTreeMaxNodes];
    std::int8_t path_length[kSpeculativeTreeMaxPaths];
    std::int8_t owned_from[kSpeculativeTreeMaxPaths];
    std::int8_t path_columns[kSpeculativeTreeMaxPaths][kSpeculativeTreeMaxPathLength];
};

static_assert(offsetof(SpeculativeTreeRow, path_columns) % 16 == 0,
              "path column rows load as 16-byte vectors");

/** I32 words per SpeculativeTreeRow; tree-row tensors are I32 [kSpeculativeTreeRowWords, B]. */
inline constexpr int kSpeculativeTreeRowWords =
    static_cast<int>(sizeof(SpeculativeTreeRow) / sizeof(std::int32_t));
static_assert(sizeof(SpeculativeTreeRow) % sizeof(std::int32_t) == 0);

/**
 * Static limits of an engine's tree rounds: every tree row verifies at most `nodes` columns
 * (2+main_depth .. kSpeculativeTreeMaxNodes) whose leaves number at most `paths`
 * (1..kSpeculativeTreeMaxPaths). main_depth is the drafter's proposal count K (1..15).
 */
struct SpeculativeTreeShape {
    std::int32_t nodes      = 0;
    std::int32_t main_depth = 0;
    std::int32_t paths      = 0;
};

/** Throws std::invalid_argument unless the shape is within the limits above. */
void validate_speculative_tree_shape(SpeculativeTreeShape shape);

/** Throws std::invalid_argument unless tree_rows is a contiguous I32 [words, B] tensor. */
void validate_speculative_tree_rows(const Tensor& tree_rows, std::int32_t batch, const char* op);

/**
 * Op: speculative_prepare_tree_verify_inputs
 *
 * Tensors are contiguous I32: anchors/base_positions [B], drafts [W-1,B], tree_rows
 * [kSpeculativeTreeRowWords,B], verify_ids/positions/rope_positions [W,B]. For row b with base
 * position F = base_positions[b] and n = tree_rows[b].nodes live columns:
 *   verify_ids[c,b] = anchors[b] for c = 0 and for c >= n, else drafts[c-1,b];
 *   positions[c,b]  = F + min(c, n-1)                  (the KV cache slot);
 *   rope_positions[c,b] = rope_positions[0,b] + depth  (depth of column c below the anchor, 0
 *                                                       for c >= n), read-modify-write.
 * A chain row's live prefix is a chain, so this is speculative_prepare_verify_inputs over it.
 */
void speculative_prepare_tree_verify_inputs(const Tensor& anchors, const Tensor& drafts,
                                            const Tensor& base_positions, const Tensor& tree_rows,
                                            Tensor& verify_ids, Tensor& positions,
                                            Tensor& rope_positions, cudaStream_t stream);

// Caller-owned transient capacity over the inclusive batch interval for a tree width.
[[nodiscard]] std::size_t speculative_accept_sparse_tree_workspace_capacity_bytes(
    std::int32_t token_domain, std::int32_t width, std::int32_t min_batch, std::int32_t max_batch);

/**
 * Op: speculative_accept_sparse_tree
 *
 * Tree form of speculative_accept_sparse_drafts over W = logits.ne[1] columns. logits is BF16
 * [248320,W,B], target_tokens I32 [W,B], drafts I32 [W-1,B], candidate_ids I32 [16,W-1,B],
 * proposal_q FP32 [16,W-1,B] (the full proposal law of each column's parent), current_extents/
 * round_lengths/round_anchors/licensed_counts/accepted_drafts/accepted_branch I32 [B],
 * licensed_tokens I32 [W,B], accepted_path I32 [W,B], tree_rows I32 [kSpeculativeTreeRowWords,B].
 *
 * A chain row (tree_rows[b].tree == 0) is speculative_accept_sparse_drafts over its first
 * current_extents[b]+1 columns. A tree row walks from the anchor. At column v with children
 * x_0..x_{m-1} (sibling order), drawn from q_v without replacement, the residual r starts at the
 * processed target distribution p_v; sibling j is accepted with probability min(1, r(x_j)/Q_j(x_j))
 * where Q_j is q_v restricted to the untaken candidates and renormalized, with counter key
 * (seed, L+depth(x_j), kSamplePurposeSpeculativeAccept, j); a rejection replaces r by the
 * normalized max(r-Q_j,0). After every sibling is rejected the correction is drawn from r with
 * (seed, L+depth(v)+1, kSamplePurposeSpeculativeCorrection, 0); at a leaf the bonus is drawn from
 * p_leaf with (seed, L+depth(leaf)+1, kSamplePurposeSpeculativeBonus, 0). Greedy rows accept the
 * child equal to the penalty-adjusted target argmax. Column c's penalty overlay is the drafts of
 * its ancestors and itself, excluding the anchor. The emitted sequence has the processed target
 * distribution of a chain round.
 *
 * Outputs follow speculative_accept_sparse_drafts along the accepted path c_0=0,c_1..c_A:
 * licensed_tokens[0:A] = drafts[c_i-1], then the terminal token; accepted_path[i,b] = c_i for
 * i<=A and -1 beyond; accepted_branch[b] is the first i with c_i != i when the path leaves the main
 * chain (its drafts i..A lie on a side branch), 0 when it stays on it, and -1 for a chain row.
 */
void speculative_accept_sparse_tree(const Tensor& target_tokens, const Tensor& logits,
                                    const Tensor& drafts, const Tensor& candidate_ids,
                                    const Tensor& proposal_q, const Tensor& current_extents,
                                    const Tensor& tree_rows, Tensor& round_lengths,
                                    Tensor& round_anchors, Tensor& licensed_tokens,
                                    Tensor& licensed_counts, Tensor& accepted_drafts,
                                    Tensor& accepted_path, Tensor& accepted_branch,
                                    std::int32_t token_domain, const SamplingConfig* configs,
                                    WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Op: speculative_tree_compact_columns
 *
 * data is an in-place [inner, D, outer...] tensor (D >= W columns) whose outer extent is
 * layers*rows_per_layer. Row b of a layer is outer index layer*rows_per_layer+r, with r =
 * row_outer[b] when row_outer (I32 [B]) is given and r = b otherwise. For each row b and each i
 * in [1, accepted_drafts[b]] with c = accepted_path[i,b] != i, column c is copied onto column i.
 * Every such c lies after the main chain and every i within it, so sources and destinations are
 * disjoint. accepted_path is I32 [W,B]; inner*element size must be a multiple of 16 bytes.
 */
void speculative_tree_compact_columns(Tensor& data, std::int32_t rows_per_layer,
                                      const Tensor& row_outer, const Tensor& accepted_path,
                                      const Tensor& accepted_drafts, cudaStream_t stream);

/**
 * Op: speculative_tree_compact_kv
 *
 * verify_positions is the round's I32 [W,B] cache position matrix; F = verify_positions[0,b].
 * For each full-attention layer view, row b and i in [1, accepted_drafts[b]] with
 * c = accepted_path[i,b] != i, copies every stored plane of cache position F+c (codes and scales
 * of K and V, in any KV storage format) onto F+i through table row kv_table_rows[b]. Other
 * positions are unchanged.
 */
void speculative_tree_compact_kv(const PagedKVBatchLayerView* layers, std::int32_t layer_count,
                                 const Tensor& verify_positions, const Tensor& kv_table_rows,
                                 const Tensor& accepted_path, const Tensor& accepted_drafts,
                                 cudaStream_t stream);

} // namespace infernix::ops
