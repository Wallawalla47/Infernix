#pragma once

#include "infernix/types.h"

#include <chrono>
#include <memory>
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

    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] LoadSummary load_summary() const;
    [[nodiscard]] ModelMetadata model_metadata() const;
    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] bool is_available() const;

    // Elastic Device expert-cache control (Qwen3.8-Flash-Next, design §19.3.7). A release hands
    // Device memory back and a restore takes it again, both applied between requests while the
    // engine keeps running, so another program can have the GPU in between. Engines without an
    // elastic cache report ResidencyStatus::supported == false and ignore requests.
    [[nodiscard]] ResidencyStatus residency_status() const;
    void request_residency(bool release, std::uint32_t frames);

    // The SSD expert tier's pinned Host memory (design §19.3.7), the level its expert records live
    // in. A release hands `bytes` of it back to the OS (0: as much as the tier may give) and a
    // restore takes it again and refills it from the artifact; both run between requests while the
    // engine keeps serving, so a machine that runs out of physical memory can be relieved without
    // stopping the engine. Engines whose model has no SSD tier report
    // ResidencyStatus::host.supported == false and ignore the request.
    void request_host_residency(bool release, std::uint64_t bytes);

    void reset_memory_peaks() noexcept;

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
