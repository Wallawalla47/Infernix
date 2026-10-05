#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/execution_context.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/speculative_tree.h"

namespace ninfer::models::qwen3_5::execution {

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::CausalAttentionExecutionEnvelope envelope) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    const bool tree = frame.tree_rows.data != nullptr;
    if (tree) {
        if (frame.proposal_q.data == nullptr || frame.feature_sink == nullptr) {
            throw std::logic_error("tree verification requires DFlash2 sparse acceptance");
        }
        card.set_verification_tree(&frame.tree_rows, &frame.tree_masks);
    }
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.state_source_slots,
                                 frame.state_destination_slots, envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens, *frame.feature_sink);
    } else {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.state_source_slots,
                                 frame.state_destination_slots, envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens);
    }
    // Constrained rows (chain rows only) restrict their columns before any acceptance reads the
    // logits or the precomputed greedy argmax.
    qwen3_5::constrain_round_logits(
        execution.io, frame.target_logits, &frame.target_tokens,
        dimension(execution.parameters.model.resources().public_token_count),
        execution.device.stream);
    if (tree) {
        card.set_verification_tree(nullptr, nullptr);
        const cudaStream_t stream = execution.device.stream;
        ops::speculative_accept_sparse_tree(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.candidate_ids,
            frame.proposal_q, frame.current_extents, frame.tree_rows, frame.frontiers,
            frame.anchors, frame.licensed_tokens, frame.licensed_counts, frame.accepted_drafts,
            frame.accepted_path, frame.accepted_branch,
            dimension(execution.parameters.model.resources().public_token_count), frame.sampling,
            execution.work, stream);
        // Move each accepted side path onto the main-chain columns so the rest of the round, the
        // commit, Fold and the next context append see an ordinary chain round.
        card.compact_tree_kv(frame.cache_positions, frame.kv_table_rows,
                             frame.state_destination_slots, frame.accepted_path,
                             frame.accepted_drafts);
        const GdnReplayRecords& records = *frame.replay_records;
        const auto dense_columns        = [](const Tensor& plane) {
            return Tensor(plane.data, plane.dtype,
                          {plane.ne[0] * plane.ne[1], plane.ne[2], plane.ne[3]});
        };
        const Tensor none{};
        for (Tensor plane : {records.conv, dense_columns(records.key), dense_columns(records.value),
                             dense_columns(records.gate)}) {
            ops::speculative_tree_compact_columns(plane, records.spec.record_capacity, none,
                                                  frame.accepted_path, frame.accepted_drafts,
                                                  stream);
        }
        Tensor hidden = frame.target_hidden;
        ops::speculative_tree_compact_columns(hidden, hidden.ne[2], none, frame.accepted_path,
                                              frame.accepted_drafts, stream);
        Tensor features = frame.pending_features;
        ops::speculative_tree_compact_columns(features, features.ne[2], frame.active_lanes,
                                              frame.accepted_path, frame.accepted_drafts, stream);
    } else if (frame.proposal_q.data != nullptr) {
        ops::speculative_accept_sparse_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.candidate_ids,
            frame.proposal_q, frame.current_extents, frame.frontiers, frame.anchors,
            frame.licensed_tokens, frame.licensed_counts, frame.accepted_drafts,
            dimension(execution.parameters.model.resources().public_token_count), frame.sampling,
            {false}, execution.work, execution.device.stream);
    } else {
        ops::speculative_accept_greedy_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.current_extents,
            frame.frontiers, frame.anchors, frame.licensed_tokens, frame.licensed_counts,
            frame.accepted_drafts,
            dimension(execution.parameters.model.resources().public_token_count), frame.sampling,
            execution.work, execution.device.stream);
    }
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.state_destination_slots, continuation_hidden_store,
                 execution.device.stream);
}

} // namespace ninfer::models::qwen3_5::execution
