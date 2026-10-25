#include "infernix/ops/candidate_selector.h"
#include "infernix/ops/speculative_round.h"
#include "infernix/ops/speculative_tree.h"
#include "models/qwen3_5/program/graph_execution.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/execution_context.h"
#include "core/nvtx.h"
#include "infernix/ops/mtp_round.h"
#include "infernix/ops/scatter.h"
#include "infernix/ops/scalar.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>

namespace infernix::models::qwen3_5::execution {
void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const Tensor* next_embedding) {
    if (!state.mtp_kv.valid() || !state.execution.io.mtp) {
        throw std::logic_error("MTP bridge requires MTP storage");
    }
    if (rope_position.size() != 3) {
        throw std::invalid_argument("MTP bridge requires one three-axis rope position");
    }
    state.execution.work.reset();
    TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                     state.text_kv, state.execution.state_images, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.state_source_slot,
                        state.state_destination_slot, state.mtp_proposal_extent);

    Tensor position_view = state.execution.io.mtp->target_positions.slice(0, 0, 1);
    ops::set_i32_scalar(position_view, position, state.execution.device.stream);
    Tensor mtp_hidden         = state.execution.io.mtp->ar_hidden;
    Tensor logits             = mtp_draft_logits(state.execution.io);
    Tensor draft0             = state.execution.io.mtp->draft_tokens.slice(0, 0, 1);
    Tensor rope_position_view = state.execution.work.alloc(DType::I32, {1, 3});
    CUDA_CHECK(cudaMemcpyAsync(rope_position_view.data, rope_position.data(),
                               rope_position.size_bytes(), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
    const ops::CausalAttentionExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
    card.mtp_forward_batch(next_token, previous_hidden, position_view, bridge_envelope, mtp_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, &rope_position_view, next_embedding);
    if (!build_proposal) { return; }

    if (state.mtp_proposal_extent == 0 ||
        state.mtp_proposal_extent >
            static_cast<std::uint32_t>(state.execution.io.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP bridge proposal extent is outside the configured window");
    }

    Tensor ar_position = state.execution.io.mtp->position.slice(0, 0, 1);
    ops::set_i32_scalar(ar_position, position + 1, state.execution.device.stream);
    for (int i = 1; i < static_cast<int>(state.mtp_proposal_extent); ++i) {
        Tensor previous_token = state.execution.io.mtp->draft_tokens.slice(0, i - 1, 1);
        Tensor next_draft     = state.execution.io.mtp->draft_tokens.slice(0, i, 1);
        Tensor next_hidden    = state.execution.prefill_hidden.slice(1, i, 1);
        const auto visible    = static_cast<std::uint32_t>(position + i + 1);
        const ops::CausalAttentionExecutionEnvelope envelope{visible, visible};
        card.mtp_forward_ar_step(previous_token, state.execution.io.mtp->ar_hidden, ar_position,
                                 envelope, next_hidden, logits, next_draft);
        CUDA_CHECK(cudaMemcpyAsync(state.execution.io.mtp->ar_hidden.data, next_hidden.data,
                                   state.execution.io.mtp->ar_hidden.bytes(),
                                   cudaMemcpyDeviceToDevice, state.execution.device.stream));
        ops::increment_i32_scalar(ar_position, state.execution.device.stream);
    }
}

namespace {

// One exact-B MTP round on the frame viewed at its widths: Forward (the target verifies the current
// drafts) and Finish (acceptance, the next round's alignment and its proposal steps, then the
// egress). Finish's pieces run in order; a chained one-row round captures its proposal steps as a
// gated chain (MtpBatchContext::proposal_chain). Forward and Finish derive the same frame view, so
// Finish reads what Forward wrote.
class MtpRound {
public:
    MtpRound(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
             MtpCausalAttentionEnvelopes envelopes)
        : state_(state), batch_(batch_size), k_(k), next_k_(state.neural_proposal_drafts), envelopes_(envelopes) {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) || k == 0 ||
            k > kMtpVerifyMaximumDrafts) {
            throw std::logic_error("MTP decode batch state is incomplete");
        }
        if (next_k_ == 0 || next_k_ > kMtpDecodeMaximumDrafts) {
            throw std::logic_error("MTP neural proposal width is outside its supported domain");
        }
        const ExecutionSchedule& schedule = state.execution.execution_schedule;
        if (state.proposal_chain && (batch_size != 1 || !schedule.mtp_proposal_chain)) {
            throw std::logic_error("only a one-row round of the proposal chain's schedule proposes through it");
        }
        if (state.step_chain && (!schedule.mtp_step_chain || state.proposal_chain || state.tree)) {
            throw std::logic_error("only a round of the step chain's schedule proposes through it");
        }
        tree_      = state.tree;
        frame_     = state.frame.narrowed(k, next_k_, tree_ ? state.chain_drafts : 0U);
        sampled_   = schedule.sampled_mtp_drafts;
        // A tree extends each row's chain over its drafts' sparse proposals.
        if ((tree_ || state.tree_proposals) && (!sampled_ || frame_.tree_rows.data == nullptr)) {
            throw std::logic_error("MTP trees need sampled drafts and the frame's tree planes");
        }
        // The confidence policy's, the cut's and the gate's input: each step's drafter probability.
        confidence_ = schedule.mtp_draft_confidence;
        if (sampled_ && (confidence_ || frame_.current_candidates.data == nullptr)) {
            throw std::logic_error("sampled MTP drafts need the MTP draft window's sparse planes, without probabilities");
        }
        state.execution.work.reset();
        card_.emplace(state.execution.device, state.execution.parameters, state.execution.work, PagedKVCacheView{},
                      state.execution.state_images, state.execution.io, state.execution.prefill_hidden,
                      state.execution.prefill_chunk, 0, PagedKVCacheView{}, &state.text_cache, &state.mtp_cache);
        card_->set_execution_schedule(schedule);
        const auto rows = [&](const Tensor& tensor, int dim) { return tensor.slice(dim, 0, batch_size); };
        anchors_            = rows(frame_.anchors, 0);
        frontiers_          = rows(frame_.base_frontiers, 0);
        budgets_            = rows(frame_.remaining_budgets, 0);
        current_extents_    = rows(frame_.current_extents, 0);
        current_drafts_     = rows(frame_.current_drafts, 1);
        target_rope_        = rows(frame_.target_rope_positions, 1);
        mtp_rows_           = rows(frame_.mtp_kv_table_rows, 0);
        state_destinations_ = rows(frame_.state_destination_slots, 0);
        rope_deltas_        = rows(frame_.rope_deltas, 0);
        verify_ids_         = rows(frame_.verify_ids, 1);
        target_positions_   = rows(frame_.target_positions, 1);
        target_hidden_      = rows(frame_.target_hidden, 2);
        licensed_counts_    = rows(frame_.licensed_counts, 0);
        accepted_           = rows(frame_.accepted_drafts, 0);
        alignment_ids_      = rows(frame_.alignment_ids, 1);
        alignment_hidden_   = rows(frame_.alignment_hidden, 2);
        ar_hidden_          = rows(frame_.ar_hidden, 1);
        next_hidden_        = rows(frame_.next_hidden, 1);
        ar_positions_       = rows(frame_.ar_positions, 0);
        ar_rope_positions_  = rows(frame_.ar_rope_positions, 0);
        ar_valid_columns_   = rows(frame_.ar_valid_columns, 0);
        next_drafts_        = rows(frame_.next_drafts, 0);
        next_probabilities_ = rows(frame_.next_probabilities, 0);
        proposal_logits_    = rows(frame_.proposal_logits, 1);
        verify_             = TargetVerifyFrameView{
                        .ids                     = verify_ids_,
                        .cache_positions         = target_positions_,
                        .rope_positions          = target_rope_,
                        .valid_columns           = rows(frame_.target_valid_columns, 0),
                        .kv_table_rows           = rows(frame_.text_kv_table_rows, 0),
                        .state_source_slots      = rows(frame_.state_source_slots, 0),
                        .state_destination_slots = state_destinations_,
                        .target_hidden           = target_hidden_,
                        .target_logits           = rows(frame_.target_logits, 2),
                        .target_tokens           = rows(frame_.target_argmax, 1),
                        .drafts                  = current_drafts_,
                        .current_extents         = current_extents_,
                        // sampled-mtp-drafts: sparse acceptance against each draft's proposal (a
                        // tree round's columns: the tree builder's).
                        .candidate_ids           = tree_      ? rows(frame_.tree_candidates, 2)
                                                   : sampled_ ? rows(frame_.current_candidates, 2)
                                                              : Tensor{},
                        .proposal_q              = tree_      ? rows(frame_.tree_q, 2)
                                                   : sampled_ ? rows(frame_.current_q, 2)
                                                              : Tensor{},
                        .frontiers               = frontiers_,
                        .anchors                 = anchors_,
                        .licensed_tokens         = rows(frame_.licensed_tokens, 1),
                        .licensed_counts         = licensed_counts_,
                        .accepted_drafts         = accepted_,
                        .selected_hidden         = rows(frame_.target_continuation_hidden, 1),
                        .replay_records          = state.execution.replay_records,
                        .sampling                = frame_.sampling,
                        .tree_rows               = tree_ ? rows(frame_.tree_rows, 1) : Tensor{},
                        .tree_masks              = tree_ ? rows(frame_.tree_masks, 1) : Tensor{},
                        .accepted_path           = rows(frame_.accepted_path, 1),
                        .accepted_branch         = rows(frame_.accepted_branch, 0),
        };
    }

    [[nodiscard]] std::uint32_t proposal_steps() const noexcept { return next_k_; }

    // The ingress (the sparse planes only with sampled drafts), the verify inputs and the target.
    void forward() {
        const std::size_t ingress = sampled_ ? sizeof(qwen3_5::MtpDecodeIngress)
                                             : offsetof(qwen3_5::MtpDecodeIngress, current_candidates);
        CUDA_CHECK(cudaMemcpyAsync(frame_.ingress.data, &state_.host_ingress, ingress, cudaMemcpyHostToDevice,
                                   stream()));
        if (tree_) {
            // Each tree row's chain gains its side leaves; the columns' candidates and laws go to
            // the tree planes the acceptance reads.
            const auto rows_of = [&](const Tensor& tensor, int dim) { return tensor.slice(dim, 0, batch_); };
            Tensor drafts            = current_drafts_;
            Tensor column_candidates = rows_of(frame_.tree_candidates, 2);
            Tensor column_q          = rows_of(frame_.tree_q, 2);
            Tensor tree_rows         = verify_.tree_rows;
            Tensor tree_masks        = verify_.tree_masks;
            Tensor valid             = verify_.valid_columns;
            ops::candidate_chain_tree(rows_of(frame_.current_candidates, 2), rows_of(frame_.current_q, 2),
                                      rows_of(frame_.current_scores, 2), frontiers_, current_extents_,
                                      frame_.sampling,
                                      {static_cast<std::int32_t>(k_ + 1U),
                                       static_cast<std::int32_t>(state_.chain_drafts),
                                       static_cast<std::int32_t>(state_.tree_paths)},
                                      drafts, column_candidates, column_q, tree_rows, tree_masks, valid,
                                      stream());
            ops::speculative_prepare_tree_verify_inputs(anchors_, current_drafts_, frontiers_, tree_rows,
                                                        verify_ids_, target_positions_, target_rope_, stream());
        } else {
            ops::speculative_prepare_verify_inputs(anchors_, current_drafts_, frontiers_, current_extents_,
                                                   verify_ids_, target_positions_, stream());
        }
        nvtx::ScopedRange target_range(nvtx::Name::DecodeMtpTarget, nvtx::Category::Mtp,
                                       static_cast<std::uint64_t>(k_ + 1U) * static_cast<std::uint64_t>(batch_));
        target_verify_forward(state_.execution, *card_, verify_, envelopes_.target_verify);
    }

    // Finish's proposal step `step`: step 0 also accepts, prepares the next round and aligns the
    // drafter; step j > 0 runs the drafter on draft j - 1 and drafts j.
    void propose(std::uint32_t step) {
        nvtx::ScopedRange draft_range(nvtx::Name::DecodeMtpDraft, nvtx::Category::Mtp,
                                      static_cast<std::uint64_t>(batch_));
        if (step == 0) {
            align();
            draft(0, ar_hidden_);
            return;
        }
        const auto at                 = static_cast<std::int32_t>(step - 1U);
        const std::int32_t hidden     = dimension(state_.execution.parameters.model.config().text.hidden_size);
        Tensor previous_batch         = draft_tokens(step - 1U).view({1, batch_});
        Tensor position               = ar_positions_.slice(1, at, 1).view({1, batch_});
        Tensor rope                   = ar_rope_positions_.slice(1, at, 1).view({1, batch_});
        Tensor valid                  = ar_valid_columns_.slice(1, at, 1).view({batch_});
        Tensor hidden_batch           = ar_hidden_.view({hidden, 1, batch_});
        Tensor next_hidden_batch      = next_hidden_.view({hidden, 1, batch_});
        card_->mtp_forward_decode_batch(previous_batch, hidden_batch, position, rope, valid, mtp_rows_,
                                        state_destinations_, envelopes_.ar[step - 1U], next_hidden_batch);
        draft(step, next_hidden_);
        CUDA_CHECK(cudaMemcpyAsync(ar_hidden_.data, next_hidden_.data, ar_hidden_.bytes(), cudaMemcpyDeviceToDevice,
                                   stream()));
    }

    // The step chain's body `body` (ExecutionSchedule::mtp_step_chain): the alignment (the drafter's
    // KV over the committed tokens and the first draft's hidden), then draft body - 1.
    void step(std::uint32_t body) {
        if (body == 0) {
            nvtx::ScopedRange draft_range(nvtx::Name::DecodeMtpDraft, nvtx::Category::Mtp,
                                          static_cast<std::uint64_t>(batch_));
            align();
        } else if (body == 1) {
            draft(0, ar_hidden_);
        } else {
            propose(body - 1U);
        }
    }

    // The step chain's gate after body `body`: the next body (draft `body`) runs while some row's
    // proposal limit admits it and, past the alignment, that row's draft body - 1 clears the round's
    // step threshold.
    void step_gate(std::uint32_t body, cudaGraphConditionalHandle next) {
        launch_graph_gate_rows(body == 0 ? nullptr : static_cast<const float*>(step_probability(body - 1U).data),
                               frame_.step_threshold, static_cast<const std::int32_t*>(frame_.proposal_limits.data),
                               batch_, static_cast<std::int32_t>(body), next, stream());
    }

    // The proposal chain's gate after step `step`: the next step runs while the round's proposal
    // limit allows it and, with the device gate, while draft `step` clears --spec-min-p (the host's
    // cut, the same binary64 comparison). A skipped step leaves its draft and probability words
    // stale, beyond the limit and the cut that read them.
    void gate(std::uint32_t step, cudaGraphConditionalHandle next) {
        const ExecutionSchedule& schedule = state_.execution.execution_schedule;
        const double threshold = schedule.mtp_gate ? schedule.mtp_min_p : -std::numeric_limits<double>::infinity();
        launch_graph_gate(static_cast<const float*>(step_probability(step).data), threshold,
                          static_cast<const std::int32_t*>(frame_.proposal_limits.data),
                          static_cast<std::int32_t>(step), next, stream());
    }

    // The round's egress (the sparse planes only with sampled drafts).
    void publish() {
        const std::size_t egress = sampled_ ? sizeof(qwen3_5::MtpDecodeEgress)
                                            : offsetof(qwen3_5::MtpDecodeEgress, next_candidates);
        CUDA_CHECK(cudaMemcpyAsync(&state_.host_egress, frame_.egress.data, egress, cudaMemcpyDeviceToHost, stream()));
    }

    // Every Finish piece in order (eager runs and unchained captures).
    void finish() {
        for (std::uint32_t step = 0; step < next_k_; ++step) { propose(step); }
        publish();
    }

private:
    [[nodiscard]] cudaStream_t stream() const noexcept { return state_.execution.device.stream; }

    // Acceptance, the next round's alignment and the drafter over the committed tokens (its KV, and
    // the accepted column's hidden in ar_hidden_ for the first draft).
    void align() {
        target_accept(state_.execution, state_.continuation_hidden_store, *card_, verify_);
        Tensor next_extents = frame_.next_extents.slice(0, 0, batch_);
        // The alignment reads the accepted path's tokens, which a tree round's compaction moved
        // onto the chain's columns with their hidden states.
        ops::mtp_prepare_next_round(verify_.licensed_tokens, anchors_, accepted_, frontiers_, budgets_,
                                    licensed_counts_, rope_deltas_, alignment_ids_, next_extents,
                                    ar_positions_, ar_rope_positions_, ar_valid_columns_,
                                    static_cast<std::int32_t>(state_.text_cache.max_context()),
                                    static_cast<std::int32_t>(next_k_), stream());
        card_->mtp_forward_decode_batch(alignment_ids_, target_hidden_, target_positions_, target_rope_,
                                        licensed_counts_, mtp_rows_, state_destinations_, envelopes_.batch,
                                        alignment_hidden_);
        ops::speculative_select_accepted_hidden(alignment_hidden_, accepted_, ar_hidden_, stream());
    }

    [[nodiscard]] Tensor draft_tokens(std::uint32_t step) const {
        return next_drafts_.slice(1, static_cast<std::int32_t>(step), 1).view({batch_});
    }

    [[nodiscard]] Tensor step_probability(std::uint32_t step) const {
        return next_probabilities_.slice(1, static_cast<std::int32_t>(step), 1).view({batch_});
    }

    // Draft `step` from the drafter's hidden: the argmax (with its probability under the confidence
    // schedule) or, with sampled drafts, a draw under each row's sampler with its sparse proposal
    // (each row's RNG position is its frontier after acceptance; the step is the draw).
    void draft(std::uint32_t step, const Tensor& hidden) {
        Tensor tokens = draft_tokens(step);
        if (sampled_) {
            constexpr auto slots = static_cast<std::int32_t>(kMtpSparseSlots);
            const auto at        = static_cast<std::int32_t>(step);
            const SampledDraftStep draw{
                .configs    = frame_.sampling,
                .positions  = frontiers_,
                .draw       = step,
                .candidates = frame_.next_candidates.slice(1, 0, batch_).slice(2, at, 1).view({slots, batch_}),
                .q          = frame_.next_q.slice(1, 0, batch_).slice(2, at, 1).view({slots, batch_}),
                .scores     = state_.tree_proposals
                                  ? frame_.next_scores.slice(1, 0, batch_).slice(2, at, 1).view(
                                        {kMtpSampledDraftRanks, batch_})
                                  : Tensor{},
            };
            card_->mtp_propose_batch(hidden, proposal_logits_, tokens, nullptr, &draw);
            return;
        }
        Tensor probability = confidence_ ? step_probability(step) : Tensor{};
        card_->mtp_propose_batch(hidden, proposal_logits_, tokens, confidence_ ? &probability : nullptr);
    }

    MtpBatchContext& state_;
    std::int32_t batch_   = 0;
    std::uint32_t k_      = 0;
    std::uint32_t next_k_ = 0;
    MtpCausalAttentionEnvelopes envelopes_;
    qwen3_5::MtpDecodeState frame_;
    bool sampled_    = false;
    bool confidence_ = false;
    bool tree_       = false;
    std::optional<TextContext> card_;
    Tensor anchors_, frontiers_, budgets_, current_extents_, current_drafts_, target_rope_, mtp_rows_,
        state_destinations_, rope_deltas_, verify_ids_, target_positions_, target_hidden_, licensed_counts_,
        accepted_, alignment_ids_, alignment_hidden_, ar_hidden_, next_hidden_, ar_positions_, ar_rope_positions_,
        ar_valid_columns_, next_drafts_, next_probabilities_, proposal_logits_;
    TargetVerifyFrameView verify_{};
};

} // namespace

void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpCausalAttentionEnvelopes envelopes,
                              DecodeGraphDefinition& definition, SpeculativePhase phase) {
    MtpRound round(state, batch_size, k, envelopes);
    if (phase == SpeculativePhase::Forward) {
        definition.capture(state.execution.device.stream, [&] { round.forward(); });
    } else if (state.step_chain) {
        definition.capture_gated_chain(
            state.execution.device.stream, static_cast<std::int32_t>(round.proposal_steps()) + 1,
            [&](std::int32_t body) { round.step(static_cast<std::uint32_t>(body)); },
            [&](std::int32_t body, cudaGraphConditionalHandle next) {
                round.step_gate(static_cast<std::uint32_t>(body), next);
            },
            [&] { round.publish(); });
    } else if (state.proposal_chain) {
        definition.capture_gated_chain(
            state.execution.device.stream, static_cast<std::int32_t>(round.proposal_steps()),
            [&](std::int32_t step) { round.propose(static_cast<std::uint32_t>(step)); },
            [&](std::int32_t step, cudaGraphConditionalHandle next) {
                round.gate(static_cast<std::uint32_t>(step), next);
            },
            [&] { round.publish(); });
    } else {
        definition.capture(state.execution.device.stream, [&] { round.finish(); });
    }
}

void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpCausalAttentionEnvelopes envelopes, DecodeGraphExecutable* executable,
                      SpeculativePhase phase) {
    if (executable != nullptr) {
        run_prepared(state, executable, [] {});
        return;
    }
    // Eager rounds run every proposal step: the limit and the cut drop what a chain would skip.
    MtpRound round(state, batch_size, k, envelopes);
    run_prepared(state, executable, [&] {
        if (phase == SpeculativePhase::Forward) {
            round.forward();
        } else {
            round.finish();
        }
    });
}

} // namespace infernix::models::qwen3_5::execution
