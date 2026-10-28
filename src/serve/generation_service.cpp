#include <charconv>
#include "serve/generation_service.h"

#include "core/temporary_switches.h"
#include "product/media_acquire/acquire.h"
#include "serve/translate.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <iterator>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace infernix::serve {

struct RequestCapacity {
    explicit RequestCapacity(std::size_t limit) : maximum(limit) {}

    std::mutex mutex;
    std::size_t active = 0;
    const std::size_t maximum;
};

struct RequestLifetime {
    RequestLifetime(std::shared_ptr<RequestCapacity> owner,
                    std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point limit)
        : capacity(std::move(owner)), started(begin), deadline(limit) {}

    ~RequestLifetime() {
        std::lock_guard lock(capacity->mutex);
        --capacity->active;
    }

    std::shared_ptr<RequestCapacity> capacity;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point deadline;
};

std::string vision_overflow_note(const infernix::ContextLengthDetails& details) {
    if (!details.vision) { return {}; }
    return " (These are image and video tokens: the prompt's media together exceed the server's per-prompt vision "
           "budget of " +
           std::to_string(details.max_context) + " tokens; drop or summarize earlier images.)";
}

ApiError request_error_to_api_error(const infernix::RequestError& exception,
                                    std::string_view constraint_param,
                                    std::span<const std::string> tool_schema_params,
                                    std::uint32_t retry_after_seconds) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    const auto tool_param = [&] {
        const std::string_view pointer = exception.pointer();
        const auto split               = pointer.find('/', 1);
        std::size_t index              = 0;
        if (pointer.starts_with('/') && split != std::string_view::npos) {
            const auto number = pointer.substr(1, split - 1);
            const auto parsed =
                std::from_chars(number.data(), number.data() + number.size(), index);
            if (parsed.ec == std::errc{} && parsed.ptr == number.data() + number.size() &&
                index < tool_schema_params.size() && !tool_schema_params[index].empty() &&
                (pointer.substr(split) == "/parameters" ||
                 pointer.substr(split).starts_with("/parameters/")))
                return tool_schema_params[index] + std::string(pointer.substr(split + 11));
        }
        return "tools" + exception.pointer();
    };
    switch (exception.kind()) {
    case infernix::RequestErrorKind::InvalidToolConstraint:
        error.status = 400;
        error.code   = "invalid_tool_constraint";
        error.param  = tool_param();
        break;
    case infernix::RequestErrorKind::InvalidGrammar:
    case infernix::RequestErrorKind::InvalidChoice:
    case infernix::RequestErrorKind::InvalidRegex:
    case infernix::RequestErrorKind::ConstraintDeadEnd:
        error.status = 400;
        error.param  = constraint_param;
        error.code =
            exception.kind() == infernix::RequestErrorKind::InvalidGrammar  ? "invalid_grammar"
            : exception.kind() == infernix::RequestErrorKind::InvalidChoice ? "invalid_choice"
            : exception.kind() == infernix::RequestErrorKind::InvalidRegex  ? "invalid_regex"
                                                                          : "constraint_dead_end";
        break;
    case infernix::RequestErrorKind::InvalidJsonSchema:
    case infernix::RequestErrorKind::UnsupportedJsonSchema:
    case infernix::RequestErrorKind::UnsatisfiableJsonSchema:
        error.status = 400;
        error.param  = exception.source() == RequestErrorSource::Tools
                           ? tool_param()
                           : std::string(constraint_param) + exception.pointer();
        error.code = exception.kind() == infernix::RequestErrorKind::InvalidJsonSchema
                         ? "invalid_json_schema"
                     : exception.kind() == infernix::RequestErrorKind::UnsupportedJsonSchema
                         ? "unsupported_json_schema"
                         : "unsatisfiable_json_schema";
        break;
    case infernix::RequestErrorKind::ContextLengthExceeded:
        error.status = 400;
        error.code   = "context_length_exceeded";
        if (const auto& details = exception.context_length()) {
            // OpenAI's wording, which agent clients match to compact and retry. An aggregate vision
            // overflow (the prompt's images together over the per-prompt vision budget) answers in the
            // same form, so a screenshot-heavy session compacts instead of failing on every turn, and
            // says what the counts are.
            error.context_length = details;
            error.message        = "This model's maximum context length is " + std::to_string(details->max_context) +
                            " tokens. However, your messages resulted in " + (details->exact ? "" : "more than ") +
                            std::to_string(details->prompt_tokens) +
                            " tokens. Please reduce the length of the messages." + vision_overflow_note(*details);
        }
        break;
    case infernix::RequestErrorKind::ThinkingBudgetCapacityInsufficient:
        error.param.clear();
        error.status = 400;
        error.code   = "thinking_budget_capacity_insufficient";
        break;
    case infernix::RequestErrorKind::MediaBudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case infernix::RequestErrorKind::InvalidMedia:
        error.status = 400;
        error.code   = "invalid_media";
        break;
    case infernix::RequestErrorKind::Overloaded:
        error.param.clear();
        error.status = 429;
        error.type   = "rate_limit_error";
        error.code   = "server_overloaded";
        error.retry_after_seconds = retry_after_seconds;
        break;
    case infernix::RequestErrorKind::QueueTimeout:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        error.retry_after_seconds = retry_after_seconds;
        break;
    case infernix::RequestErrorKind::Cancelled:
        error.param.clear();
        error.status = 499;
        error.type   = "request_cancelled";
        error.code   = "client_disconnected";
        break;
    case infernix::RequestErrorKind::Unavailable:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "service_unavailable";
        break;
    }
    return error;
}

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void throw_preparation_cancelled();

[[noreturn]] void throw_media_error(const infernix::product::media_acquire::Error& exception,
                                    std::uint32_t retry_after_seconds = 0) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case infernix::product::media_acquire::ErrorKind::BudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case infernix::product::media_acquire::ErrorKind::RemoteUnavailable:
        error.status = 502;
        error.type   = "server_error";
        error.code   = "media_fetch_failed";
        break;
    case infernix::product::media_acquire::ErrorKind::RemoteTimeout:
        error.status = 504;
        error.type   = "server_error";
        error.code   = "media_fetch_timeout";
        break;
    case infernix::product::media_acquire::ErrorKind::DeadlineExceeded:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        error.retry_after_seconds = retry_after_seconds;
        break;
    case infernix::product::media_acquire::ErrorKind::Cancelled:
        throw_preparation_cancelled();
    }
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_invalid_input(const std::exception& exception, const char* code) {
    ApiError error;
    error.status  = 400;
    error.param   = "messages";
    error.code    = code;
    error.message = exception.what();
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_preparation_cancelled() {
    ApiError error;
    error.status  = 499;
    error.type    = "request_cancelled";
    error.code    = "client_disconnected";
    error.message = "client disconnected during media preparation";
    throw ApiException(std::move(error));
}

infernix::OwnedMedia acquire_media(const ContentPart& part, Clock::time_point deadline,
                                   const std::function<bool()>& is_cancelled,
                                   std::size_t& remaining_bytes,
                                   std::uint32_t retry_after_seconds) {
    if (remaining_bytes == 0) {
        throw_media_error(infernix::product::media_acquire::Error(
            infernix::product::media_acquire::ErrorKind::BudgetExceeded,
            "request media exceeds aggregate byte limit"));
    }
    infernix::product::media_acquire::Policy policy;
    policy.max_bytes    = std::min(policy.max_bytes, remaining_bytes);
    policy.deadline     = deadline;
    policy.is_cancelled = is_cancelled;
    std::vector<std::uint8_t> source_bytes;
    try {
        source_bytes = infernix::product::media_acquire::acquire_bytes(part.source, policy);
    } catch (const infernix::product::media_acquire::Error& exception) {
        throw_media_error(exception, retry_after_seconds);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_media");
    }

    remaining_bytes -= source_bytes.size();
    infernix::OwnedMedia media;
    media.kind =
        part.kind == ContentKind::Image ? infernix::MediaKind::Image : infernix::MediaKind::Video;
    media.media_type = part.source.media_type;
    switch (part.source.kind) {
    case infernix::product::media_acquire::SourceKind::Path:
    case infernix::product::media_acquire::SourceKind::Url:
        media.source_name = part.source.value;
        break;
    case infernix::product::media_acquire::SourceKind::Data:
        media.source_name = "inline-data";
        break;
    case infernix::product::media_acquire::SourceKind::Bytes:
        media.source_name = "inline-bytes";
        break;
    }
    media.bytes               = std::move(source_bytes);
    media.image_resize_policy = part.image_resize_policy;
    return media;
}

[[noreturn]] void
throw_request_error(const infernix::RequestError& exception,
                    std::string_view constraint_param               = "structured_outputs.grammar",
                    std::span<const std::string> tool_schema_params = {},
                    std::uint32_t retry_after_seconds               = 0) {
    throw ApiException(request_error_to_api_error(exception, constraint_param, tool_schema_params,
                                                  retry_after_seconds));
}

void check_preparation_control(Clock::time_point deadline,
                               const std::function<bool()>& is_cancelled,
                               std::uint32_t retry_after_seconds) {
    if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
    if (Clock::now() >= deadline) {
        throw_request_error(infernix::RequestError(RequestErrorKind::QueueTimeout,
                                                   "inference request expired during preparation"),
                            "structured_outputs.grammar", {}, retry_after_seconds);
    }
}

class ServiceOutputSink final : public infernix::OutputSink {
public:
    explicit ServiceOutputSink(const StreamSink& sink) : sink_(&sink) {}

    void start(infernix::GenerationStart start) override {
        deliver([&] {
            if (sink_->on_start) { sink_->on_start(start); }
        });
    }

    void progress(infernix::PromptProgress progress) override {
        deliver([&] {
            if (sink_->on_progress) { sink_->on_progress(progress); }
        });
    }

    void timing(infernix::GenerationTimingObservation timing) override {
        deliver([&] {
            if (sink_->on_timing) { sink_->on_timing(timing); }
        });
    }

    void publish(infernix::OutputDelta delta) override {
        if (delta.text.empty()) { return; }
        deliver([&] {
            if (delta.channel == infernix::OutputChannel::Reasoning) {
                if (sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
            } else {
                if (sink_->on_content) { sink_->on_content(delta.text); }
            }
        });
    }

    void tool_call(infernix::ToolCallDelta delta) override {
        deliver([&] {
            if (sink_->on_tool_call) { sink_->on_tool_call(delta); }
        });
    }

    [[nodiscard]] bool disconnected() const noexcept { return disconnected_; }

private:
    template <class Callback>
    void deliver(Callback&& callback) {
        if (disconnected_) { return; }
        try {
            callback();
        } catch (const ClientDisconnected&) { disconnected_ = true; }
    }

    const StreamSink* sink_ = nullptr;
    bool disconnected_      = false;
};

// A token 4-gram's 64-bit hash (a request-log measurement: collisions are negligible at prompt sizes).
std::uint64_t token_4gram_hash(const infernix::TokenId* tokens) {
    std::uint64_t hash = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 4; ++i) {
        hash ^= static_cast<std::uint32_t>(tokens[i]) + 0x9E3779B97F4A7C15ULL + (hash << 6) + (hash >> 2);
        hash *= 0xBF58476D1CE4E5B9ULL;
    }
    return hash;
}

} // namespace

GenerationService::GenerationService(ServeOptions options, StartupObserver startup_observer,
                                     DiagnosticObserver diagnostic_observer)
    : options_(std::move(options)) {
    infernix::EngineOptions engine_options;
    engine_options.artifact_path            = options_.artifact_path;
    engine_options.chat_template_path       = options_.chat_template_path;
    engine_options.ngram_volume_path        = options_.ngram_volume_path;
    engine_options.ngram_residency          = options_.ngram_residency;
    engine_options.expert_stripe_path       = options_.expert_stripe_path;
    engine_options.expert_state_path        = options_.expert_state ? *options_.expert_state
                                                                    : std::filesystem::path(options_.artifact_path +
                                                                                            ".expert-state");
    engine_options.expert_state_off         = options_.expert_state && options_.expert_state->empty(); // --expert-state off
    engine_options.thinking_budget_message  = options_.thinking_budget_message;
    engine_options.route_trace_path        = options_.route_trace;
    engine_options.device                   = options_.device;
    engine_options.max_context              = options_.max_context;
    engine_options.rope_yarn_factor          = options_.rope_yarn_factor;
    engine_options.kv_capacity              = options_.kv_capacity;
    engine_options.vram_headroom_bytes      = options_.vram_headroom_bytes;
    engine_options.vram_past_budget         = options_.vram_past_budget;
    engine_options.expert_tail_skip         = options_.expert_tail_skip;
    engine_options.expert_a4_activations    = options_.expert_a4_activations;
    engine_options.a8_activations           = options_.a8_activations;
    engine_options.embedding_host           = options_.embedding_host;
    engine_options.max_concurrency          = options_.max_concurrency;
    engine_options.max_pending_requests     = options_.max_pending_requests;
    engine_options.pending_timeout_ms       = options_.pending_timeout_ms;
    engine_options.prefill_chunk            = options_.prefill_chunk;
    engine_options.original_int8_prefill_kernel = options_.original_int8_prefill_kernel;
    engine_options.prefill_split_workspace_mib  = options_.prefill_split_workspace_mib;
    engine_options.original_nvfp4_prefill_kernel = options_.original_nvfp4_prefill_kernel;
    // The lossless route runs no fast path.
    engine_options.fast_paths                    = options_.fast_paths && !options_.lossless;
    engine_options.lossless                      = options_.lossless;
    engine_options.fast_path_candidates          = options_.fast_path_candidates;
    engine_options.kv_cache                  = options_.kv_cache;
    engine_options.logits_dtype              = options_.logits_dtype;
    engine_options.enable_vision             = options_.enable_vision;
    engine_options.vision_offload            = options_.vision_offload;
    engine_options.vision_max_merged_tokens  = options_.vision_max_merged_tokens;
    engine_options.use_cuda_graph            = options_.use_cuda_graph;
    engine_options.speculative              = options_.speculative;
    engine_options.context_cache            = options_.context_cache;
    engine_options.context_cost.preset_path = options_.context_cost_presets;
    engine_options.media_cache_bytes        = options_.media_cache_bytes;
    engine_options.media_live_bytes         = options_.media_live_bytes;
    engine_options.ram_free_bytes           = options_.ram_free_bytes;
    engine_options.expert_ram_bytes         = options_.expert_ram_bytes;
    engine_options.expert_ram_layout        = options_.expert_ram_layout;
    engine_options.expert_pin_bytes         = options_.expert_pin_bytes;
    engine_options.media_preprocess_threads = options_.media_preprocess_threads;
    engine_options.startup_observer         = std::move(startup_observer);
    engine_options.diagnostic_observer      = std::move(diagnostic_observer);
    engine_           = std::make_unique<infernix::Engine>(std::move(engine_options));
    if (!options_.request_log_jsonl.empty()) {
        prefix_misses_ = std::make_unique<PrefixMissTracker>(
            [engine = engine_.get()](std::span<const infernix::TokenId> ids) { return engine->detokenize(ids); });
    }
    request_capacity_ = std::make_shared<RequestCapacity>(
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests);
    count_capacity_ = std::make_shared<RequestCapacity>(
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests);
}

std::shared_ptr<RequestLifetime>
GenerationService::acquire_request_lifetime(DeadlinePolicy deadline_policy, bool admitted) const {
    return acquire_lifetime(request_capacity_, deadline_policy, "inference request queue is full",
                            admitted);
}

std::shared_ptr<RequestLifetime>
GenerationService::acquire_lifetime(const std::shared_ptr<RequestCapacity>& capacity,
                                    DeadlinePolicy deadline_policy, const char* full_message,
                                    bool admitted) const {
    const auto started = Clock::now();
    {
        std::lock_guard lock(capacity->mutex);
        // An admitted request continuing after a server-side tool already holds a slot until its
        // response is released, so its next generation is never refused by the bound it counts in.
        if (!admitted && capacity->active >= capacity->maximum) {
            throw_request_error(infernix::RequestError(RequestErrorKind::Overloaded, full_message),
                                "structured_outputs.grammar", {}, queue_retry_after_seconds());
        }
        ++capacity->active;
    }
    try {
        const Clock::time_point deadline =
            deadline_policy == DeadlinePolicy::UnboundedStartup
                ? Clock::time_point::max()
                : started + std::chrono::milliseconds(options_.pending_timeout_ms);
        return std::make_shared<RequestLifetime>(capacity, started, deadline);
    } catch (...) {
        std::lock_guard lock(capacity->mutex);
        --capacity->active;
        throw;
    }
}

PreparedRequest GenerationService::prepare(const GenerationRequest& request,
                                           GenerationConsumerMode consumer_mode,
                                           infernix::GenerationObservationOptions observation,
                                           std::function<bool()> is_cancelled) const {
    return prepare_impl(request, consumer_mode, std::move(observation), std::move(is_cancelled),
                        options_.allow_prefix_reuse ? CacheParticipation::ReadWrite
                                                    : CacheParticipation::Disabled,
                        DeadlinePolicy::ClientPendingTimeout);
}

PreparedRequest GenerationService::prepare_continuation(
    const GenerationRequest& request, GenerationConsumerMode consumer_mode,
    infernix::GenerationObservationOptions observation, std::function<bool()> is_cancelled) const {
    return prepare_impl(request, consumer_mode, std::move(observation), std::move(is_cancelled),
                        options_.allow_prefix_reuse ? CacheParticipation::ReadWrite
                                                    : CacheParticipation::Disabled,
                        DeadlinePolicy::ClientPendingTimeout, /*admitted=*/true);
}

PreparedRequest GenerationService::prepare_impl(const GenerationRequest& request,
                                                GenerationConsumerMode consumer_mode,
                                                infernix::GenerationObservationOptions observation,
                                                std::function<bool()> is_cancelled,
                                                CacheParticipation cache_participation,
                                                DeadlinePolicy deadline_policy,
                                                bool admitted) const {
    PreparedRequest prepared;
    prepared.constraint_param               = request.constraint_param;
    prepared.tool_schema_params             = tool_declaration_params(request);
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    infernix::RequestOptions request_options  = to_request_options(
        request, options_, semantics, cache_participation == CacheParticipation::ReadWrite);
    request_options.ngram_session = request.ngram_session;
    prepared.thinking_budget      = request_options.execution.thinking.budget;
    prepared.reasoning_effort     = semantics.reasoning_effort;
    prepared.preserve_thinking    = semantics.preserve_thinking;
    const bool request_has_media  = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    prepared.lifetime = acquire_request_lifetime(deadline_policy, admitted);

    try {
        const auto acquisition_started = Clock::now();
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, infernix::kMaximumPromptMediaBytes);
        infernix::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, prepared.lifetime->deadline, is_cancelled,
                                     remaining_media_bytes, queue_retry_after_seconds());
            });
        prepared.acquisition_seconds =
            std::chrono::duration<double>(Clock::now() - acquisition_started).count();
        check_preparation_control(prepared.lifetime->deadline, is_cancelled,
                                  queue_retry_after_seconds());
        // Idle-time prefill of the finished turn (hybrid-prefix-cache-spec §7.9): a text chat whose
        // next request appends to it keeps its lowered conversation for run().
        const auto& cache = engine_->options().context_cache;
        if (cache_participation == CacheParticipation::ReadWrite && cache.enabled &&
            cache.hybrid.next_turn_prefill_quiet.value_or(std::chrono::milliseconds{0}).count() > 0 &&
            request.continuation == infernix::PromptContinuationMode::NewAssistantTurn &&
            !request_has_media && request.readout_tokens.empty() &&
            request.token_constraint.empty() && !request.ignore_eos && request.max_tokens > 0) {
            prepared.next_turn = input;
        }
        const PreparationControl control{
            .deadline     = prepared.lifetime->deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        infernix::PreparedPrompt prompt = engine_->prepare(std::move(input), control);
        check_preparation_control(prepared.lifetime->deadline, is_cancelled,
                                  queue_retry_after_seconds());
        prepared.enable_thinking = prompt.summary().starts_in_reasoning;
        if (!prepared.enable_thinking) {
            request_options.execution.thinking.budget.reset();
            prepared.thinking_budget.reset();
        } else if (semantics.enable_thinking != false &&
                   !request_options.execution.thinking.budget) {
            // The answer room follows the limit the request actually has: a prompt that leaves
            // less context than max_tokens would otherwise think to the end of the context.
            request_options.execution.thinking.budget = thinking_answer_room_budget(
                options_, effective_output_limit(
                              request_options.execution.requested_output_tokens,
                              engine_->options().max_context, prompt.summary().prompt_tokens));
            prepared.thinking_budget = request_options.execution.thinking.budget;
        }
        prepared.prompt_tokens           = static_cast<int>(prompt.summary().prompt_tokens);
        // The request log's output-overlap figure (candidates CF-07): the prompt's 4-grams before the
        // Engine takes the prompt; and its ids for the prefix-miss record (CF-08).
        if (const auto ids = prompt.token_ids(); !options_.request_log_jsonl.empty()) {
            prepared.prompt_ids.assign(ids.begin(), ids.end());
            if (ids.size() >= 4) {
                prepared.prompt_4grams.reserve(ids.size() - 3);
                for (std::size_t i = 0; i + 4 <= ids.size(); ++i) {
                    prepared.prompt_4grams.push_back(token_4gram_hash(ids.data() + i));
                }
                std::sort(prepared.prompt_4grams.begin(), prepared.prompt_4grams.end());
                prepared.prompt_4grams.erase(
                    std::unique(prepared.prompt_4grams.begin(), prepared.prompt_4grams.end()),
                    prepared.prompt_4grams.end());
            }
        }
        prepared.media                   = prompt.summary().media;
        prepared.preparation             = prompt.preparation_stats();
        prepared.service_prepare_seconds = std::max(
            0.0, std::chrono::duration<double>(Clock::now() - prepared.lifetime->started).count() -
                     prepared.preparation.seconds);
        if (observation.first_token) {
            observation.first_token = [callback = std::move(observation.first_token),
                                       seconds  = prepared.service_prepare_seconds](
                                          infernix::GenerationFirstTokenObservation first) {
                first.prepare_seconds += seconds;
                callback(first);
            };
        }
        prepared.generation = engine_->submit(std::move(prompt), std::move(request_options),
                                              consumer_mode == GenerationConsumerMode::Streaming
                                                  ? infernix::OutputConsumerMode::Streaming
                                                  : infernix::OutputConsumerMode::Aggregate,
                                              std::move(observation), prepared.lifetime->deadline);
        prepared.sampling   = prepared.generation.resolved_sampling();
    } catch (const ApiException&) { throw; } catch (const infernix::RequestError& exception) {
        throw_request_error(exception, request.constraint_param, prepared.tool_schema_params,
                            queue_retry_after_seconds());
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
    return prepared;
}

int GenerationService::count_prompt_tokens(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled) const {
    const bool request_has_media = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    const std::shared_ptr<RequestLifetime> lifetime = acquire_lifetime(
        count_capacity_, DeadlinePolicy::ClientPendingTimeout, "token count queue is full");
    const Clock::time_point deadline        = lifetime->deadline;
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    try {
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, infernix::kMaximumPromptMediaBytes);
        infernix::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, deadline, is_cancelled, remaining_media_bytes,
                                     queue_retry_after_seconds());
            });
        check_preparation_control(deadline, is_cancelled, queue_retry_after_seconds());
        const PreparationControl control{
            .deadline     = deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        const int prompt_tokens =
            static_cast<int>(engine_->count_tokens(std::move(input), control));
        check_preparation_control(deadline, is_cancelled, queue_retry_after_seconds());
        return prompt_tokens;
    } catch (const ApiException&) { throw; } catch (const infernix::RequestError& exception) {
        throw_request_error(exception, "structured_outputs.grammar", {},
                            queue_retry_after_seconds());
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
}

void GenerationService::admit(PreparedRequest& prepared, std::function<bool()> is_cancelled) const {
    infernix::CancellationView cancellation;
    if (is_cancelled) { cancellation = infernix::CancellationView(std::move(is_cancelled)); }
    try {
        prepared.generation.await_admission(cancellation);
    } catch (const infernix::RequestError& exception) {
        throw_request_error(exception, prepared.constraint_param, prepared.tool_schema_params,
                            queue_retry_after_seconds());
    }
}

GenerationOutcome GenerationService::run(PreparedRequest& prepared, const StreamSink* sink,
                                         std::function<bool()> is_cancelled) {
    std::unique_ptr<ServiceOutputSink> output_sink;
    if (sink != nullptr) { output_sink = std::make_unique<ServiceOutputSink>(*sink); }
    infernix::OutputSink* public_sink = output_sink.get();
    infernix::CancellationView cancellation;
    if (is_cancelled || sink != nullptr) {
        cancellation = infernix::CancellationView(
            [external = std::move(is_cancelled), sink, observed = output_sink.get()]() {
                return (observed && observed->disconnected()) || (external && external()) ||
                       (sink != nullptr && sink->is_cancelled && sink->is_cancelled());
            });
    }

    infernix::GenerationResult result;
    try {
        result = prepared.generation.wait(public_sink, cancellation);
    } catch (const infernix::RequestError& exception) {
        throw_request_error(exception, prepared.constraint_param, prepared.tool_schema_params,
                            queue_retry_after_seconds());
    }
    GenerationOutcome outcome;
    outcome.text                = std::move(result.content);
    outcome.reasoning           = std::move(result.reasoning);
    outcome.prompt_tokens       = static_cast<int>(result.prompt.prompt_tokens);
    outcome.completion_tokens   = static_cast<int>(result.generated_token_ids.size());
    outcome.generated_token_ids = std::move(result.generated_token_ids);
    if (!prepared.prompt_4grams.empty() && outcome.generated_token_ids.size() >= 4) {
        std::uint64_t total = 0, found = 0;
        for (std::size_t i = 0; i + 4 <= outcome.generated_token_ids.size(); ++i) {
            ++total;
            found += std::binary_search(prepared.prompt_4grams.begin(), prepared.prompt_4grams.end(),
                                        token_4gram_hash(outcome.generated_token_ids.data() + i))
                         ? 1U
                         : 0U;
        }
        outcome.output_4grams           = total;
        outcome.output_4grams_in_prompt = found;
    }
    outcome.reasoning_tokens    = static_cast<int>(result.reasoning_tokens);
    outcome.thinking            = result.thinking;
    outcome.constraint          = result.constraint;
    outcome.finish_reason       = result.finish_reason;
    outcome.matched_stop_string = std::move(result.matched_stop_string);
    outcome.readout             = std::move(result.readout);
    outcome.constrained_draws   = std::move(result.constrained_draws);

    // The serve's preparation precedes the Engine's: the request's preparation, first token and
    // total include it.
    outcome.timings = result.timings;
    outcome.timings.prepare_seconds += prepared.service_prepare_seconds;
    outcome.timings.first_token_seconds =
        outcome.timings.prepare_seconds +
        std::max(0.0, result.timings.first_token_seconds - result.timings.prepare_seconds);
    outcome.timings.total_seconds += prepared.service_prepare_seconds;
    outcome.engine_timing               = result.engine_timing;
    outcome.first_output_timing         = std::move(result.first_output_timing);
    outcome.scheduling                  = result.scheduling;
    outcome.admission                   = result.admission;
    outcome.engine_request_id           = result.engine_request_id;
    outcome.computed_prefill_tokens     = result.computed_prefill_tokens;
    outcome.reused_prompt_tokens        = result.reused_prompt_tokens;
    outcome.cache_created_prompt_tokens = result.cache_created_prompt_tokens;
    outcome.prefix_reuse_path           = result.prefix_reuse_path;
    outcome.speculative                 = std::move(result.speculative);
    outcome.expert_cache                = result.expert_cache;
    outcome.ngram_archive               = result.ngram_archive;
    if (prefix_misses_ != nullptr && !prepared.prompt_ids.empty()) {
        outcome.prefix_miss = prefix_misses_->settle(
            outcome.engine_request_id, prepared.prompt_ids, outcome.generated_token_ids,
            outcome.reused_prompt_tokens, outcome.computed_prefill_tokens);
    }

    outcome.tool_calls      = std::move(result.tool_calls);
    outcome.tool_call_parse = result.tool_call_parse;
    schedule_next_turn(prepared, outcome);
    return outcome;
}

// A reply that ended on its own (a stop token or string; not cut by its limit, not cancelled) is
// followed by the conversation's next request: the Engine prefills that request's history while it
// is idle (hybrid-prefix-cache-spec §7.9). The reply is appended as the client receives it and
// sends it back: content, reasoning and tool calls (the template renders no call identifier).
void GenerationService::schedule_next_turn(PreparedRequest& prepared,
                                           const GenerationOutcome& outcome) const {
    if (!prepared.next_turn) { return; }
    infernix::PromptInput conversation = std::move(*prepared.next_turn);
    prepared.next_turn.reset();
    if (outcome.finish_reason != infernix::FinishReason::StopToken &&
        outcome.finish_reason != infernix::FinishReason::StopString) {
        return;
    }
    try {
        infernix::ChatMessage reply;
        reply.role = ChatRole::Assistant;
        if (!outcome.text.empty()) {
            reply.parts.push_back(infernix::MessagePart{
                .kind = infernix::MessagePartKind::Text, .text = outcome.text, .media = {}});
        }
        reply.reasoning_content = outcome.reasoning;
        reply.tool_calls.reserve(outcome.tool_calls.size());
        for (std::size_t i = 0; i < outcome.tool_calls.size(); ++i) {
            reply.tool_calls.push_back(infernix::ToolCall{.id             = "call_" + std::to_string(i),
                                                          .name           = outcome.tool_calls[i].name,
                                                          .arguments_json = outcome.tool_calls[i].arguments_json});
        }
        conversation.messages.push_back(std::move(reply));
    } catch (...) { return; }
    engine_->prefill_next_turn(std::move(conversation));
}

void GenerationService::warmup() {
    const auto generate = [this](int max_tokens, bool ignore_eos) {
        GenerationRequest request;
        ChatTurn turn;
        turn.role = ChatRole::User;
        ContentPart content;
        content.kind     = ContentKind::Text;
        content.text     = "hi";
        content.type_raw = "text";
        turn.content.push_back(std::move(content));
        request.messages.push_back(std::move(turn));
        request.max_tokens = max_tokens;
        request.ignore_eos = ignore_eos;
        PreparedRequest prepared =
            prepare_impl(request, GenerationConsumerMode::Aggregate, {}, {},
                         CacheParticipation::Disabled, DeadlinePolicy::UnboundedStartup);
        run(prepared, nullptr);
    };
    generate(4, false);
    // TEMPORARY (INFERNIX_TMP_WARMUP_GRAPHS, candidates §7.6 D14): before READY, max-concurrency
    // requests at the server's default sampling, of staggered lengths (32 + 16 i tokens, end of
    // sequence ignored), so the decode rounds pass every batch size from 1 to the maximum and back,
    // each for several rounds: a Program's decode, verification and draft graphs run eagerly once and
    // are captured on the next round, so a fresh server's first requests replay them instead. The
    // widths are those the default draft policy picks; requests use no prefix cache. Startup only.
    if (!temporary_switches::flag("INFERNIX_TMP_WARMUP_GRAPHS")) { return; }
    const std::uint32_t rows = std::max<std::uint32_t>(1, options_.max_concurrency);
    std::vector<std::exception_ptr> errors(rows);
    std::vector<std::thread> threads;
    threads.reserve(rows);
    for (std::uint32_t i = 0; i < rows; ++i) {
        threads.emplace_back([&, i] {
            try {
                generate(32 + 16 * static_cast<int>(i), true);
            } catch (...) {
                errors[i] = std::current_exception();
            }
        });
    }
    for (auto& thread : threads) { thread.join(); }
    for (const auto& error : errors) {
        if (error) { std::rethrow_exception(error); }
    }
}

} // namespace infernix::serve
