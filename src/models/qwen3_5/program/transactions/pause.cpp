#include "models/qwen3_5/program/program_impl.h"
#include "core/device.h"

#include <array>
#include <cstdint>
#include <stdexcept>

namespace infernix::models::qwen3_5::detail {

bool ProgramImpl::start_pause(SequenceHandle handle, runtime::ExecutionTiming* observation) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, observation);
    if (context_transaction_ || pending_transaction_) { return false; }
    if (!valid_sequence(handle)) { throw std::logic_error("pause has a stale sequence"); }
    const auto lane = ContractAccess::lane(handle).value;
    auto& state     = sequences[lane];
    auto& control   = requests[lane];
    if (control.lifecycle != Lifecycle::Active && control.lifecycle != Lifecycle::Prefilling &&
        control.lifecycle != Lifecycle::Replaying) {
        throw std::logic_error("pause requires a committed boundary");
    }
    timing.begin_wait();
    device.synchronize();
    timing.end_wait();
    control.recovery.reset();
    if (control.permit) { settle_unit(lane); }
    if (control.prefill && control.prefill->vision) {
        control.prefill->retired_vision_seconds += control.prefill->vision->elapsed_seconds();
        control.prefill->vision.reset();
    }
    if (control.replay) { control.replay->vision.reset(); }
    // Before the replay's lifecycle is unwound below: the snapshot is taken at the state's own frontier.
    hybrid_publish_pause(state, control);
    auto saved      = std::make_unique<ResumeStateImpl>();
    saved->frontier = control.lifecycle == Lifecycle::Prefilling  ? control.prefill->cursor
                      : control.lifecycle == Lifecycle::Replaying ? control.replay_target
                                                                  : state.execution_frontier;
    if (control.lifecycle == Lifecycle::Replaying) {
        control.lifecycle = control.resume_lifecycle;
        control.replay.reset();
    }
    // The lane's committed blocks stay in the prefix cache, with its pause snapshot where one was
    // published; the resume binding restores the deepest cached state up to the frontier and
    // replays the rest of the ledger.
    hybrid_release_lane(state.lane);
    release_sequence_kv(state);
    release_sequence_state(state);
    state.mtp_draft_count = 0;
    saved->sequence       = std::move(state);
    saved->control        = std::move(control);
    state                 = {};
    state.lane            = lane;
    control               = {};
    control.lifecycle     = Lifecycle::Pausing;
    ContextTransaction tx;
    tx.kind  = ContextOperationKind::Pause;
    tx.lane  = lane;
    tx.epoch = lane_epochs[lane];
    tx.paused.emplace(ResumeState(std::move(saved)));
    context_transaction_.emplace(std::move(tx));
    return true;
}

void ProgramImpl::hybrid_publish_pause(SequenceState& sequence, const RequestControl& request) {
    if (!hybrid_ || !sequence.kv || sequence.lane >= max_concurrency) { return; }
    const std::uint32_t frontier = sequence.text_kv_valid;
    const bool consistent =
        (request.lifecycle == Lifecycle::Active && frontier == sequence.execution_frontier) ||
        (request.lifecycle == Lifecycle::Prefilling && request.prefill && frontier == request.prefill->cursor) ||
        (request.lifecycle == Lifecycle::Replaying && frontier == request.replay_cursor);
    if (!consistent || frontier == 0) { return; }
    if (is_masked_draft_backend(speculative_backend) && sequence.dflash_context_frontier < frontier) {
        // Decode leaves the round's accepted tokens out of the drafter's context until the next round; a snapshot
        // carries a whole ring. A drafter with its own paged KV would need pages mapped here, under the pressure
        // that is pausing the lane: it replays instead.
        const std::uint32_t lag = frontier - sequence.dflash_context_frontier;
        if (backend_kv_cache() != nullptr || lag > draft_window + 1U) { return; }
        const std::array<std::uint32_t, 1> lanes{sequence.lane};
        const std::array<std::uint32_t, 1> starts{sequence.dflash_context_frontier};
        const std::array<std::uint32_t, 1> counts{lag};
        enqueue_dflash_context_append(lanes, starts, counts);
        device.synchronize();
        sequence.dflash_context_frontier = frontier;
        commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
        work.reset();
    }
    (void)hybrid_publish_endpoint(sequence, true, true);
}

ResumeState ProgramImpl::complete_pause(ContextTransaction& tx) {
    if (!tx.paused) { throw std::logic_error("pause operation lost its durable request"); }
    requests[tx.lane]       = {};
    sequences[tx.lane]      = {};
    sequences[tx.lane].lane = tx.lane;
    invalidate_lane(tx.lane);
    return std::move(*tx.paused);
}
} // namespace infernix::models::qwen3_5::detail
