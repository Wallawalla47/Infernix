#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/program_impl.h"
#include <cstdio>
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

std::size_t SequencePlan::kv_payload_bytes() const noexcept {
    return impl_ != nullptr ? impl_->persistent.kv_payload_bytes : 0;
}

std::size_t SequencePlan::graph_allowance_bytes() const noexcept {
    return impl_ != nullptr ? impl_->graph_allowance_bytes : 0;
}

std::uint32_t SequencePlan::draft_window() const noexcept {
    return impl_ != nullptr ? impl_->draft_window : 0;
}

std::size_t SequencePlan::host_capacity_bytes() const noexcept {
    return impl_ ? impl_->context_cache.host_capacity_bytes.value_or(0) : 0;
}

bool SequencePlan::draft_tree_auto() const noexcept {
    return impl_ != nullptr && impl_->tree_widths.automatic_mode();
}

std::uint32_t SequencePlan::prefill_chunk() const noexcept {
    return impl_ != nullptr ? impl_->prefill_chunk : 0;
}

std::uint32_t SequencePlan::idle_prefill_width() const noexcept {
    return impl_ != nullptr ? impl_->idle_prefill_width : 0;
}

std::size_t SequencePlan::offload_wide_pass_bytes() const noexcept {
    return impl_ != nullptr ? std::max(impl_->workspace.text_prefill_wide, impl_->workspace.mtp_prefill_wide) : 0;
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

bool Program::recover_after_failure() noexcept { return impl_->recover_after_failure(); }

void Program::fail_all_cleanup() noexcept { impl_->fail_all_cleanup(); }

void Program::shutdown_cleanup() noexcept { impl_->shutdown_cleanup(); }

void Program::maintain() {
    if (impl_->offload) {
        // An elastic text KV idle for kKvIdleShrink gives back what the prefix cache's idle blocks with
        // host copies hold (one attempt per idle period: blocks without host copies may still hold the top).
        if (impl_->kv_idle_due()) {
            impl_->kv_busy();
            impl_->shrink_kv(true);
        }
        impl_->offload->engine().maintain(impl_->device.stream);
    }
    impl_->maintain_ram();
    // The SSD tier's transfers land while the Engine is idle: apply them and continue its write-back.
    if (impl_->hybrid_) { impl_->hybrid_->service_ssd(); }
}

void Program::set_maintenance_waker(std::function<void()> waker) {
    if (impl_->hybrid_ && impl_->hybrid_->ssd() != nullptr) { impl_->hybrid_->ssd()->set_waker(waker); }
    if (impl_->ram_) { impl_->ram_->set_waker(waker); }
    if (impl_->offload) { impl_->offload->engine().set_maintenance_waker(std::move(waker)); }
}

void Program::start_ram_resize(std::uint64_t reserve_bytes,
                               std::function<void(const std::string&, DiagnosticLevel)> diagnostic) {
    const bool host_tier = impl_->hybrid_ && impl_->hybrid_->host_tier();
    if (!host_tier && impl_->ram_expert_tier() == nullptr) { return; }
    impl_->ram_diagnostic_ = std::move(diagnostic);
    impl_->ram_            = std::make_unique<runtime::expert_offload::RamMonitor>(reserve_bytes);
    impl_->ram_->set_room(impl_->ram_shrinkable(), impl_->ram_growable());
}

runtime::expert_offload::HostExpertTier* detail::ProgramImpl::ram_expert_tier() noexcept {
    return offload ? offload->engine().tier() : nullptr;
}

std::uint64_t detail::ProgramImpl::ram_shrinkable() {
    const auto* tier = ram_expert_tier();
    return (hybrid_ ? hybrid_->host_tier_shrinkable_bytes() : 0U) + (tier ? tier->shrinkable_bytes() : 0U);
}

std::uint64_t detail::ProgramImpl::ram_growable() {
    const auto* tier = ram_expert_tier();
    return (hybrid_ ? hybrid_->host_tier_growable_bytes() : 0U) + (tier ? tier->growable_bytes() : 0U);
}

void detail::ProgramImpl::maintain_ram() {
    if (!ram_ || has_context_transaction()) { return; }
    // The SSD tier resizes only with no promotion queued and no prompt streaming (its chunks with lent or pinned slots
    // stay anyway); the promotions that have landed are published first, which drops their slots' H2D pins (the
    // device's record table is uploaded from its dirty keys before the next round).
    runtime::expert_offload::HostExpertTier* tier = ram_expert_tier();
    if (tier != nullptr && (offload->engine().residency().loads_pending() || offload->engine().stream_lent())) {
        tier = nullptr;
    }
    if (tier != nullptr) { offload->engine().residency().publish_promotions(); }
    const std::uint64_t shrinkable =
        (hybrid_ ? hybrid_->host_tier_shrinkable_bytes() : 0U) + (tier ? tier->shrinkable_bytes() : 0U);
    const std::uint64_t growable =
        (hybrid_ ? hybrid_->host_tier_growable_bytes() : 0U) + (tier ? tier->growable_bytes() : 0U);
    // A release with the reading that asked for it; a growth with what is available after it.
    const auto report = [this](bool release, std::uint64_t available, std::uint64_t prefix, std::uint64_t experts) {
        if (prefix + experts == 0 || !ram_diagnostic_) { return; }
        const double gib = 1024.0 * 1024.0 * 1024.0;
        const double total = static_cast<double>(prefix + experts) / gib, in_prefix = static_cast<double>(prefix) / gib,
                     in_experts = static_cast<double>(experts) / gib, left = static_cast<double>(available) / gib;
        char text[256];
        if (release) {
            std::snprintf(text, sizeof(text),
                          "RAM pressure: %.1f GiB available; gave back %.1f GiB (prefix cache %.1f, experts %.1f)", left,
                          total, in_prefix, in_experts);
        } else {
            std::snprintf(text, sizeof(text),
                          "RAM free again: took back %.1f GiB (prefix cache %.1f, experts %.1f); %.1f GiB available",
                          total, in_prefix, in_experts, left);
        }
        ram_diagnostic_(text, DiagnosticLevel::Info);
    };
    const runtime::expert_offload::RamDecision decision = ram_->decide(shrinkable, growable);
    const std::uint64_t asked_at = ram_->available();
    if (decision.grow == 0) {
        report(false, asked_at, std::exchange(ram_grown_prefix_, 0), std::exchange(ram_grown_experts_, 0));
    }
    if (decision.release == 0 && decision.grow == 0) { return; }
    std::uint64_t prefix = 0, experts = 0;
    if (decision.release != 0) {
        if (hybrid_) { prefix = hybrid_->shrink_host_tier(decision.release); }
        if (tier != nullptr && prefix < decision.release) { experts = tier->shrink(decision.release - prefix); }
    } else {
        // One piece (a step, or the experts' largest chunk): a request that arrives meanwhile waits for one piece's
        // pin and reads at most, and the monitor asks for the next at once.
        const std::uint64_t piece =
            std::min(decision.grow, std::max(runtime::expert_offload::RamControl::kStepBytes,
                                             tier ? tier->largest_chunk_bytes() : std::uint64_t{0}));
        if (tier != nullptr) {
            const std::int32_t* frames = offload->engine().residency().host_table();
            experts = tier->grow(piece, [frames](std::uint32_t key) { return frames[key] >= 0; });
        }
        if (hybrid_ && experts < piece) { prefix = hybrid_->grow_host_tier(piece - experts); }
    }
    const bool more = ram_->resized(decision.release != 0 || prefix + experts == 0, ram_shrinkable(), ram_growable());
    if (decision.release != 0) {
        report(true, asked_at, prefix, experts);
        return;
    }
    ram_grown_prefix_ += prefix;
    ram_grown_experts_ += experts;
    if (!more) {
        report(false, ram_->available(), std::exchange(ram_grown_prefix_, 0), std::exchange(ram_grown_experts_, 0));
    }
}

std::uint64_t Program::engine_thread_cpus() const noexcept {
    return impl_->offload ? impl_->offload->engine().engine_cpus() : 0U;
}

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

void Program::hybrid_hold_continuation(std::chrono::steady_clock::time_point deadline) {
    impl_->hybrid_hold_continuation(deadline);
}

bool Program::hybrid_holds_stale() const noexcept { return impl_->hybrid_holds_stale(); }

std::uint64_t Program::hybrid_cache_epoch() const noexcept { return impl_->hybrid_cache_epoch(); }

bool Program::hybrid_poll_storage() { return impl_->hybrid_ && impl_->hybrid_->poll_ssd(); }

bool Program::hybrid_source_pending() const noexcept { return impl_->hybrid_source_pending_; }

bool Program::hybrid_storage_due() const noexcept { return impl_->hybrid_ && impl_->hybrid_->ssd_due(); }

std::uint32_t Program::hybrid_resume_grid() const noexcept {
    return impl_->execution_schedule.grid_cache ? impl_->prefill_chunk : 1U;
}

HybridPrefixCacheStats Program::hybrid_stats() const noexcept { return impl_->hybrid_stats(); }

void Program::set_hybrid_cost(const runtime::prefix_cache::CacheCostModel& cost) {
    impl_->set_hybrid_cost(cost);
}

void Program::set_hybrid_coalesce_wait_limit(double seconds) {
    impl_->set_hybrid_coalesce_wait_limit(seconds);
}

runtime::prefix_cache::PersistResult Program::attach_hybrid_cache_file(const std::filesystem::path& path,
                                                         std::string fingerprint,
                                                         const StartupObserver& observer) {
    return impl_->attach_hybrid_cache_file(path, std::move(fingerprint), observer);
}

std::optional<runtime::prefix_cache::PersistResult> Program::save_prefix_cache_now(const CancellationView& abandoned) {
    return impl_->save_hybrid_cache_now(abandoned);
}

std::optional<runtime::prefix_cache::PersistResult> Program::hybrid_shutdown_save() const {
    return impl_->hybrid_shutdown_save();
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->physical_usage(); }

MemorySummary Program::memory_summary() const noexcept { return impl_->memory_summary(); }
std::optional<ExpertCacheTotals> Program::expert_cache_totals() const noexcept {
    if (!impl_->offload) { return std::nullopt; }
    return impl_->offload->engine().totals();
}

void Program::reset_memory_peaks() noexcept { impl_->reset_memory_peaks(); }

SequencePlanner make_sequence_planner(const execution::Parameters& parameters,
                                      DeviceContext& device, const EngineOptions& options,
                                      const detail::ProgramOffloadOptions* offload) {
    return SequencePlanner(detail::make_sequence_planner_impl(parameters, device, options, 0, offload));
}

SequencePlanner make_sequence_planner(const execution::Parameters& parameters,
                                      DeviceContext& device, const EngineOptions& options,
                                      std::uint32_t idle_prefill_width,
                                      const detail::ProgramOffloadOptions* offload) {
    if (idle_prefill_width == 0) {
        throw std::invalid_argument("an idle prefill width must be positive");
    }
    return SequencePlanner(
        detail::make_sequence_planner_impl(parameters, device, options, idle_prefill_width, offload));
}

std::vector<std::uint32_t> idle_prefill_widths(const execution::Parameters& parameters,
                                               const EngineOptions& options) {
    return detail::idle_prefill_widths_impl(parameters, options);
}

std::uint32_t resolve_auto_draft_length(const execution::Parameters& parameters, EngineOptions& options) {
    return detail::resolve_auto_draft_length_impl(parameters, options);
}

std::uint32_t resolve_auto_ngram(const Model& model, EngineOptions& options) {
    return detail::resolve_auto_ngram_impl(model, options);
}

std::string mtp_draft_switch_report(const execution::Parameters& parameters, const EngineOptions& options) {
    return detail::mtp_draft_switch_report_impl(parameters, options);
}

std::unique_ptr<Program> create_program(const execution::Parameters& parameters,
                                        SequencePlan&& plan, DeviceContext& device,
                                        const StartupObserver& startup_observer,
                                        const detail::ProgramOffloadOptions* offload) {
    if (parameters.model.offloaded_experts() != (offload != nullptr)) {
        throw std::invalid_argument("offloaded experts need the instance's offload options, and only they do");
    }
    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }
    if (plan.impl_->parameters != &parameters) {
        throw std::invalid_argument("sequence plan belongs to another model instance");
    }
    if (plan.impl_->multiprocessor_count != device.multiprocessor_count()) {
        throw std::invalid_argument("sequence plan device capacity does not match execution");
    }
    auto impl = std::make_unique<detail::ProgramImpl>(parameters, *plan.impl_, device, startup_observer, offload);
    plan.impl_.reset();
    return std::unique_ptr<Program>(new Program(std::move(impl)));
}

} // namespace infernix::models::qwen3_5
