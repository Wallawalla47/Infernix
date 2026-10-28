#pragma once

// Product-side adapter from one protocol-neutral generation request to the public Engine. Wire
// adapters normalize before this layer and render IDs, usage, and response events after it.

#include "infernix/engine.h"
#include "serve/prefix_miss.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::serve {

struct RequestLifetime;
struct RequestCapacity;

struct GenerationOutcome {
    std::string text;
    std::string reasoning;
    std::vector<infernix::TokenId> generated_token_ids;
    std::vector<infernix::GeneratedToolCall> tool_calls;
    infernix::ToolCallParseDiagnostics tool_call_parse;
    int prompt_tokens     = 0;
    int completion_tokens = 0;
    int reasoning_tokens  = 0;
    infernix::ThinkingBudgetStats thinking;
    std::optional<infernix::ConstraintObservation> constraint;
    infernix::FinishReason finish_reason = infernix::FinishReason::OutputLimit;
    std::optional<std::string> matched_stop_string;
    std::optional<infernix::PromptReadout> readout;
    std::vector<infernix::ConstrainedDraw> constrained_draws;
    // The Engine's measurements (infernix::GenerationResult's fields). timings counts the serve's
    // own preparation too: prepare_seconds, first_token_seconds (the request's TTFT) and
    // total_seconds include it.
    std::uint64_t engine_request_id       = 0;
    std::uint32_t computed_prefill_tokens = 0;
    std::uint32_t reused_prompt_tokens    = 0;
    std::optional<std::uint32_t> cache_created_prompt_tokens;
    infernix::PrefixReusePath prefix_reuse_path = infernix::PrefixReusePath::Root;
    infernix::GenerationTimings timings;
    infernix::GenerationEngineTiming engine_timing;
    std::optional<infernix::GenerationFirstOutputTiming> first_output_timing;
    infernix::GenerationSchedulingStats scheduling;
    infernix::GenerationAdmissionStats admission;
    infernix::SpeculativeStats speculative;
    NgramArchiveStats ngram_archive;
    std::optional<infernix::ExpertCacheStats> expert_cache;
    // With the request JSONL: the output's token 4-grams and those that also occur in the prompt
    // (the room copy proposals from the prompt would have); unset without the log or a prompt
    // 4-gram.
    std::optional<std::uint64_t> output_4grams;
    std::optional<std::uint64_t> output_4grams_in_prompt;
    // With the request JSONL: why a large re-prefill missed the prefix cache (prefix_miss.h).
    std::optional<PrefixMiss> prefix_miss;
};

// A transport may report an interrupted stream through a callback or is_cancelled().
// Generation still settles through Engine cancellation and returns its actual work statistics.
class ClientDisconnected final : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override { return "client disconnected"; }
};

struct StreamSink {
    std::function<void(const infernix::GenerationStart& start)> on_start;
    std::function<void(const infernix::PromptProgress& progress)> on_progress;
    std::function<void(const infernix::GenerationTimingObservation& timing)> on_timing;
    std::function<void(const std::string& delta_text)> on_content;
    std::function<void(const std::string& delta_text)> on_reasoning;
    // A constrained call's next step (infernix::ToolCallDelta), after the content before it.
    std::function<void(const infernix::ToolCallDelta& delta)> on_tool_call;
    std::function<bool()> is_cancelled;
};

enum class GenerationConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

// Translate Engine request failures into the shared protocol-neutral HTTP error contract. A
// queue-pressure answer (429 overloaded, 503 queue timeout) carries retry_after_seconds as its
// Retry-After.
ApiError
request_error_to_api_error(const infernix::RequestError& exception,
                           std::string_view constraint_param = "structured_outputs.grammar",
                           std::span<const std::string> tool_schema_params = {},
                           std::uint32_t retry_after_seconds               = 0);

// The sentence a context-overflow message appends for an aggregate vision overflow
// (ContextLengthDetails::vision): its counts are image tokens against the per-prompt vision budget.
// Empty for a text overflow.
[[nodiscard]] std::string vision_overflow_note(const infernix::ContextLengthDetails& details);

// Preparation ends by synchronously submitting the owning prompt to the Engine FIFO. The returned
// request keeps its ingress/response lifetime reservation until the HTTP response is released and
// is consumed exactly once by run().
struct PreparedRequest {
    std::string constraint_param;
    std::vector<std::string> tool_schema_params;
    infernix::GenerationHandle generation;
    infernix::ResolvedSamplingParameters sampling;
    // Service input acquisition/bookkeeping; Engine timings own prompt and constraint preparation.
    double service_prepare_seconds = 0.0;
    double acquisition_seconds = 0.0;
    PromptPreparationStats preparation;
    int prompt_tokens    = 0;
    // Display size of each submitted media item, in prompt order.
    std::vector<infernix::MediaGeometry> media;
    bool enable_thinking = true;
    std::optional<std::uint32_t> thinking_budget;
    std::optional<infernix::ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    std::shared_ptr<RequestLifetime> lifetime;
    // With the request JSONL: the prompt's token 4-grams (hashed, sorted, unique), for the outcome's
    // output-overlap figure, and its token ids, for the prefix-miss record; empty otherwise.
    std::vector<std::uint64_t> prompt_4grams;
    std::vector<infernix::TokenId> prompt_ids;
    // With idle-time prefill (HybridPrefixCacheOptions::next_turn_prefill_quiet): the request's lowered
    // text conversation, to which run() appends the reply before handing it to
    // Engine::prefill_next_turn. Unset for requests the next turn cannot follow (media, assistant
    // continuation, readouts, token constraints, no prefix reuse).
    std::optional<infernix::PromptInput> next_turn;
};

// What the HTTP routes ask of generation. GenerationService answers it over the Engine; a test of
// the routes answers it with its own.
class GenerationBackend {
public:
    virtual ~GenerationBackend() = default;

    // Engine owns the once-normalized startup configuration. Serving diagnostics must use this
    // value instead of reinterpreting optional defaults from ServeOptions.
    [[nodiscard]] virtual const infernix::EngineOptions& engine_options() const = 0;
    [[nodiscard]] virtual infernix::LoadSummary load_summary() const = 0;
    [[nodiscard]] virtual infernix::ModelMetadata model_metadata() const = 0;
    [[nodiscard]] virtual infernix::MemorySummary memory_summary() const = 0;
    [[nodiscard]] virtual infernix::RuntimeStats runtime_stats() const = 0;
    virtual void reset_execution_maxima() noexcept = 0;
    [[nodiscard]] virtual bool is_available() const = 0;
    // An Engine-wide execution failure, as opposed to an orderly stop.
    [[nodiscard]] virtual bool has_failed() const = 0;
    [[nodiscard]] virtual infernix::ModelSamplingDefaults sampling_defaults() const = 0;
    // Artifact-tokenizer encoding of raw text, with no template or special token added.
    [[nodiscard]] virtual std::vector<infernix::TokenId>
    tokenize_text(std::string_view text) const = 0;

    [[nodiscard]] virtual PreparedRequest
    prepare(const GenerationRequest& req, GenerationConsumerMode consumer_mode,
            infernix::GenerationObservationOptions observation = {},
            std::function<bool()> is_cancelled = {}) const = 0;
    // The next generation of an admitted request that executed a server-side tool (tool search):
    // the HTTP request still holds its ingress slot, so the continuation takes another without
    // the queue bound refusing it, and gets a fresh pending deadline.
    [[nodiscard]] virtual PreparedRequest
    prepare_continuation(const GenerationRequest& req, GenerationConsumerMode consumer_mode,
                         infernix::GenerationObservationOptions observation = {},
                         std::function<bool()> is_cancelled = {}) const = 0;
    [[nodiscard]] virtual int count_prompt_tokens(const GenerationRequest& req,
                                                  std::function<bool()> is_cancelled = {}) const = 0;

    // Consumes prepared.generation. A PreparedRequest is single-use.
    virtual GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink,
                                  std::function<bool()> is_cancelled = {}) = 0;

    // A streaming route calls it before committing its response: blocks until the Engine admits the
    // request, so that the queue's own answers (a queue timeout, an orderly stop) still reach the
    // client as a plain HTTP status, as on the non-streaming routes; after admission only
    // mid-generation failures remain, which belong in the stream. Throws the ApiException run()
    // would; a request cancelled while queued returns, and run() gives its answer.
    virtual void admit(PreparedRequest& prepared,
                       std::function<bool()> is_cancelled = {}) const = 0;

    // Begins the Engine's orderly stop: running and queued generations fail as Unavailable.
    virtual void stop() noexcept = 0;

protected:
    GenerationBackend()                                    = default;
    GenerationBackend(const GenerationBackend&)            = default;
    GenerationBackend& operator=(const GenerationBackend&) = default;
};

class GenerationService final : public GenerationBackend {
public:
    explicit GenerationService(ServeOptions options, StartupObserver startup_observer = {},
                               DiagnosticObserver diagnostic_observer = {});

    [[nodiscard]] const ServeOptions& options() const noexcept { return options_; }

    [[nodiscard]] const infernix::EngineOptions& engine_options() const override {
        return engine_->options();
    }

    [[nodiscard]] infernix::LoadSummary load_summary() const override {
        return engine_->load_summary();
    }

    [[nodiscard]] infernix::ModelMetadata model_metadata() const override {
        return engine_->model_metadata();
    }

    [[nodiscard]] infernix::MemorySummary memory_summary() const override {
        return engine_->memory_summary();
    }

    [[nodiscard]] infernix::RuntimeStats runtime_stats() const override {
        return engine_->runtime_stats();
    }

    void reset_execution_maxima() noexcept override { engine_->reset_execution_maxima(); }

    [[nodiscard]] bool is_available() const override { return engine_->is_available(); }

    [[nodiscard]] bool has_failed() const override { return engine_->has_failed(); }

    [[nodiscard]] infernix::MediaCacheSummary media_cache_summary() const {
        return engine_->media_cache_summary();
    }

    [[nodiscard]] infernix::ModelSamplingDefaults sampling_defaults() const override {
        return engine_->sampling_defaults();
    }

    [[nodiscard]] infernix::ChatTemplateRoleSupport chat_template_role_support() const {
        return engine_->chat_template_role_support();
    }

    [[nodiscard]] std::vector<infernix::TokenId>
    tokenize_text(std::string_view text) const override {
        return engine_->tokenize_text(text);
    }

    [[nodiscard]] PreparedRequest prepare(const GenerationRequest& req,
                                          GenerationConsumerMode consumer_mode,
                                          infernix::GenerationObservationOptions observation = {},
                                          std::function<bool()> is_cancelled = {}) const override;
    [[nodiscard]] PreparedRequest
    prepare_continuation(const GenerationRequest& req, GenerationConsumerMode consumer_mode,
                         infernix::GenerationObservationOptions observation = {},
                         std::function<bool()> is_cancelled = {}) const override;
    [[nodiscard]] int count_prompt_tokens(const GenerationRequest& req,
                                          std::function<bool()> is_cancelled = {}) const override;

    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink,
                          std::function<bool()> is_cancelled = {}) override;

    void admit(PreparedRequest& prepared,
               std::function<bool()> is_cancelled = {}) const override;

    void warmup();

    void stop() noexcept override { engine_->stop(); }

private:
    enum class CacheParticipation : std::uint8_t {
        Disabled,
        ReadWrite,
    };

    enum class DeadlinePolicy : std::uint8_t {
        ClientPendingTimeout,
        UnboundedStartup,
    };

    [[nodiscard]] PreparedRequest
    prepare_impl(const GenerationRequest& req, GenerationConsumerMode consumer_mode,
                 infernix::GenerationObservationOptions observation,
                 std::function<bool()> is_cancelled, CacheParticipation cache_participation,
                 DeadlinePolicy deadline_policy, bool admitted = false) const;
    [[nodiscard]] std::shared_ptr<RequestLifetime>
    acquire_request_lifetime(DeadlinePolicy deadline_policy, bool admitted) const;
    // Hands a finished conversation (prepared.next_turn plus the reply) to Engine::prefill_next_turn.
    void schedule_next_turn(PreparedRequest& prepared, const GenerationOutcome& outcome) const;
    [[nodiscard]] std::shared_ptr<RequestLifetime>
    acquire_lifetime(const std::shared_ptr<RequestCapacity>& capacity,
                     DeadlinePolicy deadline_policy, const char* full_message,
                     bool admitted = false) const;

    // The Retry-After of a queue-pressure answer: the admission window (--pending-timeout-ms), at
    // least 1 s.
    [[nodiscard]] std::uint32_t queue_retry_after_seconds() const noexcept {
        return std::max<std::uint32_t>(1, (options_.pending_timeout_ms + 999U) / 1000U);
    }

    ServeOptions options_;
    std::unique_ptr<infernix::Engine> engine_;
    // The recent contexts behind GenerationOutcome::prefix_miss (with the request JSONL only).
    std::unique_ptr<PrefixMissTracker> prefix_misses_;
    std::shared_ptr<RequestCapacity> request_capacity_;
    // Token counting runs the whole preparation path on handler threads, so it has its own bound:
    // a flood of counts is rejected before it can occupy the threads generation prepares on.
    std::shared_ptr<RequestCapacity> count_capacity_;
};

} // namespace infernix::serve
