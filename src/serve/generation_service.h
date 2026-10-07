#pragma once

// Product-side adapter from one protocol-neutral generation request to the public Engine. Wire
// adapters normalize before this layer and render IDs, usage, and response events after it.

#include "infernix/engine.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::serve {

struct RequestLifetime;
struct RequestCapacity;

struct GenerationMetrics {
    std::uint64_t engine_request_id            = 0;
    std::uint32_t computed_prefill_tokens      = 0;
    double prepare_seconds                     = 0.0;
    double ttft_seconds                        = 0.0;
    double vision_seconds                      = 0.0;
    double prefill_seconds                     = 0.0;
    double decode_seconds                      = 0.0;
    double decode_share_seconds                = 0.0;
    double prompt_wall_seconds                 = 0.0;
    double generation_wall_seconds             = 0.0;
    double total_seconds                       = 0.0;
    double vision_offload_window_seconds       = 0.0;
    double vision_offload_evict_seconds        = 0.0;
    double vision_offload_restore_seconds      = 0.0;
    std::uint64_t vision_offload_evicted_bytes = 0;
    std::uint64_t vision_offload_staged_bytes  = 0;
    infernix::GenerationEngineTiming engine_timing;
    std::optional<infernix::GenerationFirstOutputTiming> first_output_timing;
    infernix::GenerationSchedulingStats scheduling;

    SpeculativeBackend speculative_backend    = SpeculativeBackend::None;
    std::uint32_t speculative_draft_window    = 0;
    std::uint64_t speculative_rounds          = 0;
    std::uint64_t speculative_draft_tokens    = 0;
    std::uint64_t speculative_accepted_tokens = 0;
    std::uint64_t speculative_fallback_steps  = 0;
    std::vector<std::uint64_t> speculative_accepted_per_position;
    std::optional<infernix::ExpertCacheStats> expert_cache;
    std::uint64_t ngram_rounds                  = 0;
    std::uint64_t ngram_drafted_tokens          = 0;
    std::uint64_t ngram_accepted_tokens         = 0;
    std::uint64_t ngram_archive_rounds          = 0;
    std::uint64_t ngram_archive_drafted_tokens  = 0;
    std::uint64_t ngram_archive_accepted_tokens = 0;
    std::uint64_t tree_rounds                   = 0;
    std::uint64_t tree_side_rounds              = 0;
    std::uint64_t tree_side_accepted_tokens     = 0;
    NgramArchiveStats ngram_archive;
    std::uint32_t prefix_cache_hit_tokens     = 0;
    infernix::PrefixReusePath prefix_reuse_path = infernix::PrefixReusePath::Root;
};

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
    infernix::FinishReason finish_reason = infernix::FinishReason::OutputLimit;
    std::optional<std::string> matched_stop_string;
    std::optional<infernix::PromptReadout> readout;
    std::vector<infernix::ConstrainedDraw> constrained_draws;
    GenerationMetrics metrics;
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
    std::function<bool()> is_cancelled;
};

enum class GenerationConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

// Translate Engine request failures into the shared protocol-neutral HTTP error contract.
ApiError request_error_to_api_error(const infernix::RequestError& exception);

// Preparation ends by synchronously submitting the owning prompt to the Engine FIFO. The returned
// request keeps its ingress/response lifetime reservation until the HTTP response is released and
// is consumed exactly once by run().
struct PreparedRequest {
    infernix::GenerationHandle generation;
    infernix::ResolvedSamplingParameters sampling;
    double prepare_seconds     = 0.0;
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
};

class GenerationService {
public:
    explicit GenerationService(ServeOptions options, StartupObserver startup_observer = {},
                               DiagnosticObserver diagnostic_observer = {});

    [[nodiscard]] const ServeOptions& options() const noexcept { return options_; }

    // Engine owns the once-normalized startup configuration. Serving diagnostics must use this
    // value instead of reinterpreting optional defaults from ServeOptions.
    [[nodiscard]] const infernix::EngineOptions& engine_options() const { return engine_->options(); }

    [[nodiscard]] infernix::LoadSummary load_summary() const { return engine_->load_summary(); }

    [[nodiscard]] infernix::ModelMetadata model_metadata() const {
        return engine_->model_metadata();
    }

    [[nodiscard]] infernix::MemorySummary memory_summary() const { return engine_->memory_summary(); }

    [[nodiscard]] infernix::RuntimeStats runtime_stats() const { return engine_->runtime_stats(); }

    [[nodiscard]] bool is_available() const { return engine_->is_available(); }

    [[nodiscard]] infernix::MediaCacheSummary media_cache_summary() const {
        return engine_->media_cache_summary();
    }

    [[nodiscard]] infernix::ModelSamplingDefaults sampling_defaults() const {
        return engine_->sampling_defaults();
    }

    // Artifact-tokenizer encoding of raw text, with no template or special token added.
    [[nodiscard]] std::vector<infernix::TokenId> tokenize_text(std::string_view text) const {
        return engine_->tokenize_text(text);
    }

    [[nodiscard]] PreparedRequest prepare(const GenerationRequest& req,
                                          GenerationConsumerMode consumer_mode,
                                          infernix::GenerationObservationOptions observation = {},
                                          std::function<bool()> is_cancelled               = {},
                                          ContextCacheHints context_cache = {}) const;
    [[nodiscard]] int count_prompt_tokens(const GenerationRequest& req,
                                          std::function<bool()> is_cancelled = {}) const;

    // Consumes prepared.generation. A PreparedRequest is single-use.
    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink,
                          std::function<bool()> is_cancelled = {});

    void warmup();

    // Begins the Engine's orderly stop: running and queued generations fail as Unavailable.
    void stop() noexcept { engine_->stop(); }

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
                 std::function<bool()> is_cancelled, ContextCacheHints context_cache,
                 CacheParticipation cache_participation, DeadlinePolicy deadline_policy) const;
    [[nodiscard]] std::shared_ptr<RequestLifetime>
    acquire_request_lifetime(DeadlinePolicy deadline_policy) const;
    [[nodiscard]] std::shared_ptr<RequestLifetime>
    acquire_lifetime(const std::shared_ptr<RequestCapacity>& capacity,
                     DeadlinePolicy deadline_policy, const char* full_message) const;

    ServeOptions options_;
    std::unique_ptr<infernix::Engine> engine_;
    std::shared_ptr<RequestCapacity> request_capacity_;
    // Token counting runs the whole preparation path on handler threads, so it has its own bound:
    // a flood of counts is rejected before it can occupy the threads generation prepares on.
    std::shared_ptr<RequestCapacity> count_capacity_;
};

} // namespace infernix::serve
