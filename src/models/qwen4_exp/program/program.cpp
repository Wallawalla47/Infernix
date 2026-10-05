#include "models/qwen4_exp/program/program_impl.h"

namespace ninfer::models::qwen4_exp {

RequestBasePlan::RequestBasePlan(std::unique_ptr<detail::BasePlanImpl> impl) noexcept : impl_(std::move(impl)) {}
RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept            = default;
RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;
RequestBasePlan::~RequestBasePlan()                                     = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept { return impl_->summary; }

Program::Program(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options)
    : impl_(std::make_unique<detail::ProgramImpl>(parameters, device, std::move(options))) {}

Program::~Program() noexcept = default;

ProgramDevicePlan Program::plan_device(const ProgramOptions& options, const Config& config,
                                       std::uint32_t public_tokens) {
    return detail::ProgramImpl::plan_device(options, config, static_cast<std::int32_t>(public_tokens)).bytes;
}

const ProgramDevicePlan& Program::device_plan() const noexcept { return impl_->device_layout().bytes; }

const VramSizing& Program::vram_sizing() const noexcept { return impl_->vram_sizing(); }

void Program::set_maintenance_waker(std::function<void()> waker) { impl_->set_maintenance_waker(std::move(waker)); }

void Program::maintain() { impl_->apply_vram_target(true); }

RequestBasePlan Program::plan_request(const PreparedPrompt& prompt, const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(prompt, options);
}

bool Program::isolated_request_feasible(const RequestBasePlan& base) const noexcept { return impl_->feasible(base); }

HybridAdmissionQuote Program::hybrid_quote(const PreparedPrompt& prompt, const RequestBasePlan& base,
                                           runtime::LaneId destination) {
    return impl_->quote(prompt, base, destination);
}

runtime::ContextTransactionReserveStatus Program::hybrid_reserve_materialization(
    HybridAdmissionQuote&& quote, PreparedPrompt&& prompt, runtime::CancellationFlagView cancellation) {
    return impl_->reserve(std::move(quote), std::move(prompt), cancellation);
}

ContextTransactionProgress Program::progress_context_transaction(runtime::CancellationFlagView cancellation) {
    return impl_->progress(cancellation);
}

void Program::finalize_context_transaction() noexcept { impl_->finalize(); }

bool Program::has_context_transaction() const noexcept { return impl_->in_transaction(); }

PrefillProgress Program::advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming*,
                                         runtime::PrefillStepWidth) {
    return impl_->advance_prefill(sequence);
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming*) {
    return impl_->decode(sequences, budgets);
}

runtime::ExecutionTiming Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
                                                       std::span<const TokenId> row_major_tokens,
                                                       std::uint32_t row_stride,
                                                       std::span<const std::optional<std::uint32_t>>,
                                                       runtime::ExecutionTiming*) {
    return impl_->append_forced(sequences, row_major_tokens, row_stride);
}

CommitResult Program::commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation, runtime::ExecutionTiming*) {
    return impl_->commit(std::move(pending), decisions);
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept { return impl_->discard(std::move(pending)); }

FinishResult Program::finish(SequenceHandle sequence) noexcept { return impl_->finish(sequence); }

AbortResult Program::abort(SequenceHandle sequence) noexcept { return impl_->abort(sequence); }

std::optional<PhysicalUsageSnapshot> Program::fail_all_cleanup() noexcept {
    impl_->release_all();
    return std::nullopt;
}

runtime::ProgramResourceRevision Program::resource_revision() const noexcept {
    return runtime::ProgramResourceRevision{impl_->revision()};
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->usage(); }

MemorySummary Program::memory_summary() const noexcept { return impl_->memory(); }

} // namespace ninfer::models::qwen4_exp
