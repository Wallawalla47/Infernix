#pragma once
#include "ops/candidate_selector/bf16/candidate_selector_path_plan.h"

namespace ninfer::ops::detail {
void candidate_selector_path_launch(SelectorRoute route, const Tensor& candidate_ids,
                                    const Tensor& unary_scores, const Tensor& projected_hidden,
                                    const Tensor& anchors, const Tensor& predecessor_codebook,
                                    const Tensor& successor_codebook, const Tensor& base_positions,
                                    const SamplingConfig* configs, Tensor& drafts,
                                    Tensor& proposal_q, const SelectorWorkspace& workspace,
                                    cudaStream_t stream);

// Lattice plus per-row tree build; workspace must hold the lattice route's edges.
void candidate_selector_tree_launch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                    const Tensor& projected_hidden, const Tensor& anchors,
                                    const Tensor& predecessor_codebook,
                                    const Tensor& successor_codebook, const Tensor& base_positions,
                                    const Tensor& current_extents, const SamplingConfig* configs,
                                    SpeculativeTreeShape shape, Tensor& drafts,
                                    Tensor& column_candidates, Tensor& proposal_q,
                                    Tensor& tree_rows, Tensor& tree_masks, Tensor& valid_columns,
                                    const SelectorWorkspace& workspace, cudaStream_t stream);
}
