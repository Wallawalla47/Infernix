#include "models/qwen3_5/program/program_impl.h"
#include "core/device.h"

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
    auto saved      = std::make_unique<ResumeStateImpl>();
    saved->frontier = control.lifecycle == Lifecycle::Prefilling  ? control.prefill->cursor
                      : control.lifecycle == Lifecycle::Replaying ? control.replay_target
                                                                  : state.execution_frontier;
    if (control.lifecycle == Lifecycle::Replaying) {
        control.lifecycle = control.resume_lifecycle;
        control.replay.reset();
    }
    // The lane's committed blocks stay in the prefix cache; the resume binding restores the
    // deepest cached state up to the frontier and replays the rest of the ledger.
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

ResumeState ProgramImpl::complete_pause(ContextTransaction& tx) {
    if (!tx.paused) { throw std::logic_error("pause operation lost its durable request"); }
    requests[tx.lane]       = {};
    sequences[tx.lane]      = {};
    sequences[tx.lane].lane = tx.lane;
    invalidate_lane(tx.lane);
    return std::move(*tx.paused);
}
} // namespace infernix::models::qwen3_5::detail
