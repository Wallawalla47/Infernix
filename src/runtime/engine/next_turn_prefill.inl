#pragma once

// EngineCore's idle-time prefill of finished turns (hybrid-prefix-cache-spec §7.9): the preparation thread,
// the worker's start and yield of the internal pass, and the cache-coverage skip. Declared, with their state,
// in engine_core.h, which includes this file.

namespace infernix::runtime {

template <class Instance>
void EngineCore<Instance>::prefill_next_turn(PromptInput conversation) noexcept {
    if (next_turn_quiet_.count() == 0 || next_turn_disabled_.load(std::memory_order_acquire)) {
        return;
    }
    try {
        conversation.options.continuation = PromptContinuationMode::NextTurnHistory;
        const Clock::time_point registered = Clock::now();
        std::uint64_t epoch                = 0;
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) { return; }
            epoch = submission_epoch_;
        }
        bool replaced = false;
        {
            std::lock_guard lock(next_turn_mutex_);
            if (next_turn_stop_) { return; }
            replaced = next_turn_input_.has_value();
            next_turn_input_.emplace(
                NextTurnInput{.input = std::move(conversation), .epoch = epoch, .registered = registered});
        }
        if (replaced) { next_turn_dropped_.fetch_add(1, std::memory_order_relaxed); }
        next_turn_cv_.notify_one();
    } catch (...) {}
}

template <class Instance>
ResolvedRequestOptions EngineCore<Instance>::next_turn_options() {
    ResolvedRequestOptions options;
    options.execution.requested_output_tokens = 1;
    options.execution.allow_prefix_reuse      = true;
    options.execution.background              = true;
    options.tool_choice.constraints           = ToolConstraintMode::Automatic;
    return options;
}

template <class Instance>
void EngineCore<Instance>::next_turn_prepare_loop() noexcept {
    for (;;) {
        std::optional<NextTurnInput> turn;
        {
            std::unique_lock lock(next_turn_mutex_);
            next_turn_cv_.wait(lock, [&] { return next_turn_stop_ || next_turn_input_.has_value(); });
            if (next_turn_stop_) { return; }
            turn.swap(next_turn_input_);
        }
        const std::uint64_t epoch = turn->epoch;
        const auto current        = [this, epoch] {
            std::lock_guard lock(queue_mutex_);
            return !stopping_ && !failed_ && submission_epoch_ == epoch;
        };
        bool accepted = false;
        try {
            if (current()) {
                const PreparationControl control{
                    .deadline     = {},
                    .cancellation = CancellationView([this, &current] {
                        {
                            std::lock_guard lock(next_turn_mutex_);
                            if (next_turn_stop_ || next_turn_input_.has_value()) { return true; }
                        }
                        return !current();
                    })};
                const Clock::time_point started = Clock::now();
                PreparedPrompt prompt = instance_.frontend.prepare(std::move(turn->input), control);
                const ResolvedRequestOptions options = next_turn_options();
                auto output = instance_.frontend.make_output_session(
                    prompt, options.stop, options.output, options.execution.thinking,
                    options.constraint, options.tool_choice);
                const PromptSummary summary = prompt.summary();
                const double seconds =
                    std::chrono::duration<double>(Clock::now() - started).count();
                bool replaced = false;
                {
                    std::lock_guard lock(queue_mutex_);
                    if (!stopping_ && !failed_ && submission_epoch_ == epoch &&
                        summary.prompt_tokens <= max_context_) {
                        replaced = next_turn_candidate_.has_value();
                        next_turn_candidate_.emplace(NextTurnCandidate{.prompt          = std::move(prompt),
                                                              .output          = std::move(output),
                                                              .summary         = summary,
                                                              .prepare_seconds = seconds,
                                                              .registered      = turn->registered});
                        ++next_turn_serial_;
                        accepted = true;
                    }
                }
                if (replaced) { next_turn_dropped_.fetch_add(1, std::memory_order_relaxed); }
            }
        } catch (...) {
            // A history the template or the context cannot take is simply not prefilled.
        }
        if (accepted) {
            queue_cv_.notify_all();
        } else {
            next_turn_dropped_.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

template <class Instance>
auto EngineCore<Instance>::next_turn_due() const -> std::optional<Clock::time_point> {
    if (!next_turn_candidate_) { return std::nullopt; }
    return std::max(next_turn_candidate_->registered, worker_busy_at_) + next_turn_quiet_;
}

template <class Instance>
bool EngineCore<Instance>::start_next_turn_locked() {
    const auto due = next_turn_due();
    if (!due || Clock::now() < *due) { return false; }
    NextTurnCandidate candidate = std::move(*next_turn_candidate_);
    next_turn_candidate_.reset();
    if (next_request_id_ == 0 || next_publication_order_ == 0) { return false; }
    auto request = std::make_shared<Request>(
        next_request_id_++, next_publication_order_++, std::move(candidate.prompt),
        std::move(candidate.output), candidate.summary, candidate.prepare_seconds,
        next_turn_options(), OutputConsumerMode::Aggregate, GenerationObservationOptions{},
        Clock::time_point::max(), Clock::now());
    request->background        = true;
    request->consumer_released = true;
    request->capacity_released = true;
    pending_.push_back(request);
    next_turn_request_ = request;
    ++next_turn_counters_.started;
    request_admission_check();
    return true;
}

template <class Instance>
inline void EngineCore<Instance>::yield_next_turn() {
    if (!next_turn_request_) { return; }
    {
        std::lock_guard lock(next_turn_request_->mutex);
        if (next_turn_request_->response_done) {
            next_turn_request_.reset();
            return;
        }
    }
    // A pass that has already reached its end settles at this boundary anyway.
    if (next_turn_request_->cancelled.load(std::memory_order_acquire) ||
        next_turn_request_->is_model_finished()) {
        return;
    }
    bool real = !paused_.empty() || (materializing_ && materializing_->request != next_turn_request_);
    for (std::uint32_t lane = 0; lane < max_concurrency_ && !real; ++lane) {
        real = slots_[lane] != nullptr && slots_[lane] != next_turn_request_;
    }
    if (!real) {
        std::lock_guard lock(queue_mutex_);
        real = stopping_ || std::any_of(pending_.begin(), pending_.end(),
                                        [&](const auto& request) { return request != next_turn_request_; });
    }
    if (!real) { return; }
    next_turn_request_->cancelled.store(true, std::memory_order_release);
    if (next_turn_request_->admitted_at) {
        ++next_turn_counters_.yielded;
    } else {
        next_turn_dropped_.fetch_add(1, std::memory_order_relaxed);
    }
    request_admission_check();
}

template <class Instance>
bool EngineCore<Instance>::next_turn_covered(const Request& request,
                                             std::span<const SourceCandidate> sources) const {
    const std::uint32_t prompt = request.base_plan->summary().prompt_tokens;
    std::uint32_t best         = 0;
    for (const SourceCandidate& source : sources) { best = std::max(best, source.reused_tokens); }
    const std::uint32_t grid = std::max<std::uint32_t>(instance_.program->hybrid_resume_grid(), 1U);
    // The pass's last snapshot sits at its history's end (one token before its prompt's end), on
    // the grid at or before it.
    const std::uint32_t deepest = prompt == 0 ? 0U : (prompt - 1U) / grid * grid;
    return deepest < best + kNextTurnMinimumTokens;
}

} // namespace infernix::runtime
