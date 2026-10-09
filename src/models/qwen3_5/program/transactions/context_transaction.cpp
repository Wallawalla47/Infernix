#include "models/qwen3_5/program/program_impl.h"
#include "core/device.h"

#include <stdexcept>

namespace infernix::models::qwen3_5::detail {

// A binding activates its lane from cached blocks without waiting for the Host restore it staged
// (the lane's Device work queues behind it), and a pause only releases the lane: no transaction
// holds another lane's context.
bool ProgramImpl::context_blocks(SequenceHandle) const noexcept { return false; }

ContextProgress ProgramImpl::poll_context(runtime::CancellationFlagView cancellation) {
    if (!context_transaction_) { throw std::logic_error("there is no context operation"); }
    auto& tx = *context_transaction_;
    ContextProgress out{.kind = tx.kind};
    if (tx.kind == ContextOperationKind::Pause && tx.paused && tx.paused->impl_) {
        out.request_timings     = tx.paused->impl_->control.timings;
        out.request_speculative = tx.paused->impl_->control.speculative_stats;
    }
    if (cancellation.requested()) {
        out.operations = tx.operations;
        abort_context();
        out.complete = true;
        return out;
    }
    if (tx.kind == ContextOperationKind::Bind) {
        complete_hybrid_binding(tx, out);
    } else {
        out.paused.emplace(complete_pause(tx));
    }
    out.complete = out.published = true;
    out.operations               = tx.operations;
    context_transaction_.reset();
    return out;
}

void ProgramImpl::abort_context() noexcept {
    if (!context_transaction_) { return; }
    auto& tx = *context_transaction_;
    try {
        // CUDA failure is already fatal to the Engine; it must not prevent CPU ownership cleanup.
        (void)cudaStreamSynchronize(device.transfer_stream);
        if (tx.hybrid) {
            abort_hybrid_binding(tx);
            if (tx.adopted) { hybrid_release_lane(tx.lane); }
        }
        tx.text_activation.reset();
        tx.backend_activation.reset();
        if (tx.kind == ContextOperationKind::Bind && tx.adopted) {
            release_sequence_state(sequences[tx.lane]);
            release_sequence_kv(sequences[tx.lane]);
            tx.reserved_state.reset();
        }
        if (tx.reserved_kv) {
            if (tx.reserved_kv->backend &&
                !backend_kv_addresses->release_after_deactivate(*tx.reserved_kv->backend)) {
                std::terminate();
            }
            if (!text_kv_addresses->release_after_deactivate(tx.reserved_kv->text)) {
                std::terminate();
            }
            tx.reserved_kv.reset();
        }
        if (tx.kind == ContextOperationKind::Bind) {
            sequences[tx.lane].state = {};
            sequences[tx.lane].kv.reset();
            requests[tx.lane] = {};
            invalidate_lane(tx.lane);
        } else {
            requests[tx.lane] = {};
            invalidate_lane(tx.lane);
        }
        if (tx.reserved_state) { (void)state_store->release(*tx.reserved_state); }
        context_transaction_.reset();
    } catch (...) { std::terminate(); }
}

} // namespace infernix::models::qwen3_5::detail
