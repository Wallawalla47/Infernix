#pragma once

#include "infernix/types.h"

#include <chrono>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace infernix {

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();

    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] const PromptSummary& summary() const noexcept;
    [[nodiscard]] const PromptPreparationStats& preparation_stats() const noexcept;
    // The token ids the Engine will process for this prompt (chat template applied), for
    // measurement tools that score a rendered conversation; empty for an empty prompt.
    [[nodiscard]] std::span<const TokenId> token_ids() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    class Impl;
    explicit PreparedPrompt(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class GenerationHandle {
public:
    GenerationHandle() noexcept;
    ~GenerationHandle();

    GenerationHandle(GenerationHandle&&) noexcept;
    GenerationHandle& operator=(GenerationHandle&&) noexcept;

    GenerationHandle(const GenerationHandle&)            = delete;
    GenerationHandle& operator=(const GenerationHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept;

    GenerationResult wait(OutputSink* sink = nullptr, const CancellationView& cancellation = {});

    // Blocks until the Engine admits this handle's request (the boundary that publishes its
    // GenerationStart) or answers it before admission, consuming nothing: wait() still delivers every
    // event. An error answered before admission (a queue timeout, an orderly stop) is rethrown here as
    // wait() would rethrow it, so a streaming route can answer with a plain HTTP status before it
    // commits a response; a request settled without admission and without an error (cancelled while
    // queued) returns, and wait() gives its answer. Valid on a stream-consumed handle only.
    void await_admission(const CancellationView& cancellation = {});

private:
    class Impl;
    explicit GenerationHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class Engine {
public:
    explicit Engine(EngineOptions options);
    ~Engine();

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;

    // Raw token input is retained for repeatable correctness and performance measurement.
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;

    // Artifact-tokenizer raw-text encoding. No chat template or implicit special token is added.
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;

    // Artifact-tokenizer decoding of token ids, special tokens included. The bytes of a character
    // that the ids cut are returned as they are (not necessarily valid UTF-8).
    [[nodiscard]] std::string detokenize(std::span<const TokenId> token_ids) const;

    // The token ids of a text chat rendered with the artifact's chat template, whatever their number: unlike
    // prepare(), no max_context or capacity limit applies (a measurement tool cuts its windows from them).
    // Also marks the tokens the model writes (the assistant messages) apart from the prompt text.
    [[nodiscard]] ChatTokens tokenize_chat(PromptInput input) const;

    // Returns log p(tokens[i] | tokens[0..i)) for i in [first_target,tokens.size()), and the
    // distribution outputs options requests for those positions.
    [[nodiscard]] ScoreResult score_tokens(std::vector<TokenId> tokens, std::uint32_t first_target,
                                           ScoreOptions options = {});

    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    // Establishes queue membership synchronously with a fixed output consumer mode. Destroying an
    // unconsumed handle cancels its request; wait() owns result consumption and may run
    // independently from GPU execution. Streaming mode requires a non-null sink in wait() and
    // publishes one exact GenerationStart before output deltas; Aggregate mode requires a null
    // sink. Observation options request protocol-neutral publication facts without changing the
    // execution request.
    [[nodiscard]] GenerationHandle
    submit(PreparedPrompt prompt, RequestOptions options,
           OutputConsumerMode consumer_mode                       = OutputConsumerMode::Aggregate,
           GenerationObservationOptions observation               = {},
           std::chrono::steady_clock::time_point pending_deadline = {});

    GenerationResult generate(PreparedPrompt prompt, RequestOptions options,
                              OutputSink* sink                     = nullptr,
                              const CancellationView& cancellation = {});

    // Idle-time prefill of a finished turn (docs/maintainer/hybrid-prefix-cache-spec.md §7.9).
    // `conversation` is a request's text messages followed by its finished assistant reply as the
    // client will send it back (content, reasoning, tool calls), with that request's prompt options.
    // Off the caller's thread, the Engine renders it as the conversation's next request will
    // (PromptContinuationMode::NextTurnHistory) and, once it has had nothing to run for
    // HybridPrefixCacheOptions::next_turn_prefill_quiet, prefills that history into the prefix cache as an
    // internal request. A newer conversation replaces one not yet started, a submitted request drops
    // it, and a request arriving while it runs makes it stop at its next unit boundary. Outputs never
    // change: the next request reuses it only where its tokens match. No effect when the option is
    // zero, the prefix cache is off, or the Engine scores; never throws.
    void prefill_next_turn(PromptInput conversation) noexcept;

    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] LoadSummary load_summary() const;
    [[nodiscard]] ModelMetadata model_metadata() const;
    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    // Which instruction positions the active chat template renders; prompt preparation lowers an
    // instruction turn that arrived in an unsupported position to a role the template does render.
    [[nodiscard]] ChatTemplateRoleSupport chat_template_role_support() const;
    [[nodiscard]] bool is_available() const;
    // True once an execution failure has ended every request: the Engine refuses new work until
    // the process starts a new one. An orderly stop() is not a failure.
    [[nodiscard]] bool has_failed() const;

    void reset_memory_peaks() noexcept;
    // Restarts RuntimeStats::execution's maxima (counts and totals keep running), so they describe
    // what follows: a benchmark's measured runs, or serving after the startup warm-up.
    void reset_execution_maxima() noexcept;

    // Begins the orderly stop without waiting for it: new work is refused, and queued and active
    // generation requests end with an Unavailable error within one unit of work. A Generation
    // Engine then saves its prefix cache file, when configured. Destruction waits for the stop.
    // Idempotent and callable from any thread.
    void stop() noexcept;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace infernix
