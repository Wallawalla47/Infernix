#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/program_impl.h"
#include <stdexcept>
#include <utility>

namespace infernix::models::qwen3_5 {


SequencePlan::SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}

SequencePlan::SequencePlan(SequencePlan&&) noexcept = default;

SequencePlan& SequencePlan::operator=(SequencePlan&&) noexcept = default;

SequencePlan::~SequencePlan() = default;

std::uint32_t SequencePlan::capacity() const noexcept {
    return impl_ != nullptr ? impl_->capacity : 0;
}

std::uint32_t SequencePlan::kv_capacity() const noexcept {
    return impl_ != nullptr ? impl_->kv_capacity : 0;
}

std::uint32_t SequencePlan::max_concurrency() const noexcept {
    return impl_ != nullptr ? impl_->max_concurrency : 0;
}

std::size_t SequencePlan::device_reservation_bytes() const noexcept {
    return impl_ != nullptr ? impl_->device_reservation_bytes : 0;
}

std::size_t SequencePlan::workspace_capacity_bytes() const noexcept {
    return impl_ != nullptr ? impl_->workspace.capacity : 0;
}

std::size_t SequencePlan::host_capacity_bytes() const noexcept {
    return impl_ ? impl_->context_cache.host_capacity_bytes.value_or(0) : 0;
}

bool SequencePlan::draft_tree_auto() const noexcept {
    return impl_ != nullptr && impl_->tree_widths.automatic_mode();
}

SequencePlanner::SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept
    : impl_(std::move(impl)) {}

SequencePlanner::SequencePlanner(SequencePlanner&&) noexcept = default;

SequencePlanner& SequencePlanner::operator=(SequencePlanner&&) noexcept = default;

SequencePlanner::~SequencePlanner() = default;

const runtime::SequenceCapacityCurve& SequencePlanner::capacity_curve() const noexcept {
    static const runtime::SequenceCapacityCurve empty;
    return impl_ != nullptr ? impl_->curve : empty;
}

SequencePlan SequencePlanner::finalize(std::uint32_t main_page_groups) && {
    if (impl_ == nullptr) { throw std::logic_error("sequence planner is empty"); }
    return SequencePlan(detail::finalize_sequence_plan_impl(std::move(impl_), main_page_groups));
}

RequestBasePlan::RequestBasePlan(std::shared_ptr<detail::RequestBasePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}

RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept = default;

RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;

RequestBasePlan::~RequestBasePlan() = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept {
    static const runtime::RequestPlanSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

ResumeState::ResumeState(std::unique_ptr<detail::ResumeStateImpl> impl) noexcept
    : impl_(std::move(impl)) {}

ResumeState::ResumeState(ResumeState&&) noexcept            = default;
ResumeState& ResumeState::operator=(ResumeState&&) noexcept = default;
ResumeState::~ResumeState()                                 = default;

std::uint32_t ResumeState::frontier() const noexcept { return impl_ ? impl_->frontier : 0; }

Program::Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept : impl_(std::move(impl)) {}

Program::~Program() noexcept = default;

RequestBasePlan Program::plan_request(PreparedPrompt&& prompt,
                                      const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(PreparedPromptAccess::take(std::move(prompt)), options);
}

ScoreResult Program::causal_score(PreparedPrompt&& prompt, std::uint32_t first_target,
                                  const ScoreOptions& options) {
    return impl_->causal_score(PreparedPromptAccess::take(std::move(prompt)), first_target,
                               options);
}

runtime::ResourceReservation Program::reserve_units(std::span<const ExecutionUnit> units) {
    return impl_->reserve_units(units);
}

void Program::release_units(std::span<const SequenceHandle> units) noexcept {
    impl_->release_units(units);
}

BindingReservation Program::start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                          const SourceCandidate& source, ResumeState* resume,
                                          ExecutionUnitKind kind, std::uint32_t tokens) {
    return impl_->start_binding(base, lane, source, resume, kind, tokens);
}

bool Program::start_pause(SequenceHandle h, runtime::ExecutionTiming* timing) {
    return impl_->start_pause(h, timing);
}

ContextProgress Program::poll_context(runtime::CancellationFlagView c) {
    return impl_->poll_context(c);
}

bool Program::context_blocks(SequenceHandle sequence) const noexcept {
    return impl_->context_blocks(sequence);
}

bool Program::recovery_pending(SequenceHandle sequence) const noexcept {
    return impl_->recovery_pending(sequence);
}

bool Program::has_context_transaction() const noexcept { return impl_->has_context_transaction(); }

PrefillProgress Program::advance_prefill(SequenceHandle h, runtime::ExecutionTiming* t,
                                         runtime::TokenMaskProvider* m) {
    return impl_->advance_prefill(h, t, m);
}

ReplayProgress Program::advance_replay(SequenceHandle h, runtime::ExecutionTiming* t) {
    return impl_->advance_replay(h, t);
}

PendingBatch Program::decode(std::span<const SequenceHandle> s,
                             std::span<const runtime::RoundBudget> b, runtime::ExecutionTiming* t,
                             runtime::TokenMaskProvider* m) {
    return impl_->decode(s, b, t, m);
}

runtime::ExecutionTiming Program::append_forced_tokens(
    std::span<const SequenceHandle> s, std::span<const TokenId> ids, std::uint32_t stride,
    std::span<const std::optional<std::uint32_t>> splits, runtime::ExecutionTiming* t) {
    return impl_->append_forced_tokens(s, ids, stride, splits, t);
}

CommitResult Program::commit(PendingBatch&& p, std::span<const runtime::CommitDecision> d,
                             runtime::CommitObservation o, runtime::ExecutionTiming* t) {
    return impl_->commit(std::move(p), d, o, t);
}

DiscardResult Program::abort_pending(PendingBatch&& p) noexcept {
    return impl_->abort_pending(std::move(p));
}

FinishResult Program::finish(SequenceHandle s) noexcept { return impl_->finish(s); }

AbortResult Program::abort(SequenceHandle s) noexcept { return impl_->abort(s); }

void Program::fail_all_cleanup() noexcept { impl_->fail_all_cleanup(); }

void Program::shutdown_cleanup() noexcept { impl_->shutdown_cleanup(); }

std::vector<SourceCandidate> Program::hybrid_sources(const RequestBasePlan& base,
                                                     std::uint32_t maximum_frontier) {
    return impl_->hybrid_sources(base, maximum_frontier);
}

bool Program::hybrid_reclaim(runtime::ContextResourceUsage shortage) {
    return impl_->hybrid_reclaim(shortage);
}

std::optional<std::uint32_t> Program::hybrid_prefetch(const RequestBasePlan& base) {
    return impl_->hybrid_prefetch(base);
}

std::uint32_t Program::hybrid_prefetch_room() const noexcept {
    return impl_->hybrid_prefetch_room();
}

void Program::hybrid_hold_queue(std::span<const RequestBasePlan* const> queue) {
    impl_->hybrid_hold_queue(queue);
}

std::uint64_t Program::hybrid_cache_epoch() const noexcept { return impl_->hybrid_cache_epoch(); }

HybridPrefixCacheStats Program::hybrid_stats() const noexcept { return impl_->hybrid_stats(); }

void Program::set_hybrid_cost(const runtime::prefix_cache::CacheCostModel& cost) {
    impl_->set_hybrid_cost(cost);
}

void Program::set_hybrid_coalesce_wait_limit(double seconds) {
    impl_->set_hybrid_coalesce_wait_limit(seconds);
}

HybridCachePersistence Program::attach_hybrid_cache_file(const std::filesystem::path& path,
                                                         std::string fingerprint,
                                                         const StartupObserver& observer) {
    return impl_->attach_hybrid_cache_file(path, std::move(fingerprint), observer);
}

std::optional<HybridCachePersistence> Program::save_prefix_cache_now(const CancellationView& abandoned) {
    return impl_->save_hybrid_cache_now(abandoned);
}

std::optional<HybridCachePersistence> Program::hybrid_shutdown_save() const {
    return impl_->hybrid_shutdown_save();
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->physical_usage(); }

MemorySummary Program::memory_summary() const noexcept { return impl_->memory_summary(); }

void Program::reset_memory_peaks() noexcept { impl_->reset_memory_peaks(); }

SequencePlanner make_sequence_planner(const execution::Parameters& parameters,
                                      DeviceContext& device, const EngineOptions& options) {
    return SequencePlanner(detail::make_sequence_planner_impl(parameters, device, options));
}

std::unique_ptr<Program> create_program(const execution::Parameters& parameters,
                                        SequencePlan&& plan, DeviceContext& device,
                                        const StartupObserver& startup_observer) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }
    if (plan.impl_->parameters != &parameters) {
        throw std::invalid_argument("sequence plan belongs to another model instance");
    }
    if (plan.impl_->multiprocessor_count != device.multiprocessor_count()) {
        throw std::invalid_argument("sequence plan device capacity does not match execution");
    }
    auto impl =
        std::make_unique<detail::ProgramImpl>(parameters, *plan.impl_, device, startup_observer);
    plan.impl_.reset();
    return std::unique_ptr<Program>(new Program(std::move(impl)));
}

} // namespace infernix::models::qwen3_5
