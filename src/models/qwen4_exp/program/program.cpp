#include "models/qwen4_exp/program/program_impl.h"

namespace infernix::models::qwen4_exp {

RequestBasePlan::RequestBasePlan(std::unique_ptr<detail::BasePlanImpl> impl) noexcept : impl_(std::move(impl)) {}
RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept            = default;
RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;
RequestBasePlan::~RequestBasePlan()                                     = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept { return impl_->summary; }

ResumeState::ResumeState(std::unique_ptr<detail::ResumeStateImpl> impl) noexcept : impl_(std::move(impl)) {}
ResumeState::ResumeState(ResumeState&&) noexcept            = default;
ResumeState& ResumeState::operator=(ResumeState&&) noexcept = default;
ResumeState::~ResumeState()                                 = default;
std::uint32_t ResumeState::frontier() const noexcept { return impl_ ? impl_->target : 0U; }

Program::Program(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options)
    : impl_(std::make_unique<detail::ProgramImpl>(parameters, device, std::move(options))) {}

Program::~Program() noexcept = default;

ProgramDevicePlan Program::plan_device(const ProgramOptions& options, const Config& config,
                                       std::uint32_t public_tokens) {
    return detail::ProgramImpl::plan_device(options, config, static_cast<std::int32_t>(public_tokens)).bytes;
}

const ProgramDevicePlan& Program::device_plan() const noexcept { return impl_->device_layout().bytes; }

const VramSizing& Program::vram_sizing() const noexcept { return impl_->vram_sizing(); }
ExpertRamFill Program::expert_ram_fill() const noexcept { return impl_->expert_ram_fill(); }
std::optional<ExpertCacheTotals> Program::expert_cache_totals() const noexcept { return impl_->expert_cache_totals(); }

void Program::set_maintenance_waker(std::function<void()> waker) { impl_->set_maintenance_waker(std::move(waker)); }

// A verification round's expert-cache update may wait for the next round's proposal (ProgramImpl::DeferredRound), and
// a DFlash2 verification's commit waits on the device behind a host gate until commit() (ProgramImpl::CommitGate):
// the entry points that can touch the expert cache, launch a forward or report on it release the gate with no
// committed column and settle the update first; decode settles it while its proposal drafts, commit defers it, and
// planning and accessors leave both.
void Program::maintain() {
    impl_->settle_deferred();
    impl_->apply_vram_target(true);
    impl_->maintain_expert_state();
    impl_->maintain_ram();
    impl_->maintain_storage();
}

void Program::start_ram_resize(std::uint64_t reserve_bytes) { impl_->start_ram_resize(reserve_bytes); }

HybridPrefixCacheStats Program::hybrid_stats() const noexcept { return impl_->prefix_stats(); }

RequestBasePlan Program::plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(std::move(prompt), options);
}

std::vector<SourceCandidate> Program::hybrid_sources(const RequestBasePlan& base, std::uint32_t maximum_frontier) {
    return impl_->hybrid_sources(base, maximum_frontier);
}

BindingReservation Program::start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                          const SourceCandidate& source, ResumeState* resume,
                                          ExecutionUnitKind, std::uint32_t) {
    impl_->settle_deferred();
    return impl_->start_binding(base, lane, source, resume);
}

ContextProgress Program::poll_context(runtime::CancellationFlagView cancellation) {
    impl_->settle_deferred();
    return impl_->poll_context(cancellation);
}

bool Program::has_context_transaction() const noexcept { return impl_->in_transaction(); }

bool Program::context_blocks(SequenceHandle sequence) const noexcept { return impl_->context_blocks(sequence); }

bool Program::hybrid_reclaim(runtime::ContextResourceUsage shortage) {
    impl_->settle_deferred();
    return impl_->hybrid_reclaim(shortage);
}
std::optional<std::uint32_t> Program::hybrid_prefetch(const RequestBasePlan& base) {
    impl_->settle_deferred();
    return impl_->hybrid_prefetch(base);
}
std::uint32_t Program::hybrid_prefetch_room() const noexcept { return impl_->hybrid_prefetch_room(); }
void Program::hybrid_hold_queue(std::span<const RequestBasePlan* const> queue) { impl_->hybrid_hold_queue(queue); }
std::uint64_t Program::hybrid_cache_epoch() const noexcept { return impl_->hybrid_cache_epoch(); }

bool Program::hybrid_poll_storage() { return impl_->hybrid_poll_storage(); }

bool Program::hybrid_source_pending() const noexcept { return impl_->hybrid_source_pending(); }

bool Program::hybrid_storage_due() const noexcept { return impl_->hybrid_storage_due(); }
std::uint32_t Program::hybrid_resume_grid() const noexcept { return impl_->hybrid_resume_grid(); }

runtime::ResourceReservation Program::reserve_units(std::span<const ExecutionUnit> units) {
    return impl_->reserve_units(units);
}

bool Program::start_pause(SequenceHandle sequence, runtime::ExecutionTiming*) {
    impl_->settle_deferred();
    return impl_->start_pause(sequence);
}

ReplayProgress Program::advance_replay(SequenceHandle sequence, runtime::ExecutionTiming*) {
    impl_->settle_deferred();
    return impl_->advance_replay(sequence);
}

PrefillProgress Program::advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming*,
                                         runtime::TokenMaskProvider* masks) {
    impl_->settle_deferred();
    return impl_->advance_prefill(sequence, masks);
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming*, runtime::TokenMaskProvider* masks) {
    return impl_->decode(sequences, budgets, masks);
}

runtime::ExecutionTiming Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
                                                       std::span<const TokenId> row_major_tokens,
                                                       std::uint32_t row_stride,
                                                       std::span<const std::optional<std::uint32_t>>,
                                                       runtime::ExecutionTiming*) {
    impl_->settle_deferred();
    return impl_->append_forced(sequences, row_major_tokens, row_stride);
}

CommitResult Program::commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation, runtime::ExecutionTiming*) {
    return impl_->commit(std::move(pending), decisions);
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept { return impl_->discard(std::move(pending)); }

FinishResult Program::finish(SequenceHandle sequence) noexcept {
    impl_->settle_deferred_or_drop();
    return impl_->finish(sequence);
}

AbortResult Program::abort(SequenceHandle sequence) noexcept {
    impl_->settle_deferred_or_drop();
    return impl_->abort(sequence);
}

void Program::fail_all_cleanup() noexcept { impl_->release_all(); }

bool Program::recover_after_failure() noexcept { return impl_->recover_after_failure(); }

void Program::shutdown_cleanup() noexcept {
    impl_->settle_deferred_or_drop();
    impl_->report_cache_layers();
    impl_->release_all();
    impl_->save_expert_state();
    impl_->save_prefix_cache_for_shutdown();
}

runtime::prefix_cache::PersistResult Program::attach_prefix_cache_file(const std::filesystem::path& path, std::string fingerprint,
                                                         const StartupObserver& observer) {
    impl_->settle_deferred();
    return impl_->attach_prefix_cache_file(path, std::move(fingerprint), observer);
}

std::optional<runtime::prefix_cache::PersistResult> Program::prefix_shutdown_save() const { return impl_->prefix_shutdown_save(); }

std::optional<runtime::prefix_cache::PersistResult> Program::save_prefix_cache_now(const CancellationView& abandoned) {
    impl_->settle_deferred();
    return impl_->save_prefix_cache_now(abandoned);
}

ScoreResult Program::causal_score(PreparedPrompt&& prompt, std::uint32_t first_target, const ScoreOptions& options) {
    impl_->settle_deferred();
    return impl_->causal_score(std::move(prompt), first_target, options);
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->usage(); }

MemorySummary Program::memory_summary() const noexcept { return impl_->memory(); }

} // namespace infernix::models::qwen4_exp
