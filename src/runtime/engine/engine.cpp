#include "infernix/engine.h"

#include "core/device.h"
#include "core/nvtx.h"
#include "core/power_throttling.h"
#include "core/startup.h"
#include "core/temporary_switches.h"
#include "infernix/ops/lossless_ops.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/request.h"
#include "runtime/engine/causal_score_core.h"
#include "runtime/engine/engine_core.h"
#include "runtime/engine/diagnostics.h"
#include "runtime/engine/context_cache/hybrid_resource_manager.h"
#include "runtime/engine/model_instance.h"
#include "runtime/engine/qwen4_exp_instance.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace infernix {
namespace {

// The per-thread stack an Engine reserves (design note candidates §22 VM-12). The driver keeps stack x resident
// threads of local memory for it (1 KiB by default, ~267 MiB on an RTX 5090), taken before the Programs size their
// frames and KV. Every kernel off the lossless route has a frame of at most 240 bytes (cuobjdump of the 2026-10-26
// build: the GGUF dense tiles), so 256 bytes serve them; the lossless route's attention and fallback GEMMs need up to
// 1,472 (twenty-two functions over 256), so a lossless Engine reserves 1.5 KiB up front instead of the driver growing
// the stack outside the plan at their first launch.
constexpr std::size_t kStackBytes         = 256;
constexpr std::size_t kLosslessStackBytes = 1536;

// --vram-past-budget sizes Qwen3.8-Flash-Next's expert cache past the OS budget. The driver loads a kernel's code
// lazily, at its first launch, so the code of the sampler, prefill and drafter kernels is allocated after the cache
// has taken the budget and lands past it, where the context fails ("unspecified launch failure", 5 of 7 warmups or
// first long requests at --vram-headroom-mib 128, 2 of 6 decode runs at 256; none of 16 without the flag, and none
// of 30 with every module loaded first). Loading every module at context creation puts the code (382 MiB with the
// whole binary; about 100 MiB is what a request launches) inside the budget, ahead of the cache. It costs 7 % of
// single-lane decode (128.2 against 137.9 tok/s, flag off, ABBA over six runs) and 2 % of two-lane decode, so only
// the flag pays it. The user's own CUDA_MODULE_LOADING stands.
void load_kernels_eagerly() {
    constexpr const char* kName = "CUDA_MODULE_LOADING";
    if (std::getenv(kName) != nullptr) { return; }
#ifdef _WIN32
    _putenv_s(kName, "EAGER");
#else
    setenv(kName, "EAGER", 0);
#endif
}

DeviceContext initialize_device(const EngineOptions& options, bool code_before_cache) {
    StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaInitialize);
    if (code_before_cache) { load_kernels_eagerly(); }
    DeviceContext device(options.device);
    const std::size_t stack = options.lossless ? kLosslessStackBytes : kStackBytes;
    device.bind_to_current_thread();
    if (const cudaError_t err = cudaDeviceSetLimit(cudaLimitStackSize, stack); err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaDeviceSetLimit(cudaLimitStackSize) failed: ") + cudaGetErrorString(err));
    }
    phase.complete();
    return device;
}

runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    if (options.execution.thinking.budget && *options.execution.thinking.budget == 0) {
        throw std::invalid_argument("thinking budget must be positive");
    }
    const std::vector<TokenId>& readout = options.execution.readout_tokens;
    if (readout.size() > kMaximumReadoutTokens) {
        throw std::invalid_argument("readout names more than kMaximumReadoutTokens tokens");
    }
    if (!readout.empty() && options.execution.requested_output_tokens == 0) {
        // The readout is taken where the first output token is chosen.
        throw std::invalid_argument("readout requires at least one requested output token");
    }
    std::vector<TokenId> distinct = readout;
    std::ranges::sort(distinct);
    if (std::ranges::adjacent_find(distinct) != distinct.end()) {
        throw std::invalid_argument("readout tokens must be distinct");
    }
    const TokenConstraint& constraint = options.execution.constraint;
    if (!constraint.empty()) {
        if (constraint.steps.size() > kMaximumConstraintSteps) {
            throw std::invalid_argument("constraint has more than kMaximumConstraintSteps steps");
        }
        if (options.execution.thinking.budget) {
            // The budget's injected close would occupy steps the constraint owns.
            throw std::invalid_argument("constraint cannot be combined with a thinking budget");
        }
        if (options.execution.thinking.loop == ReasoningLoopAction::Conclude) {
            // So would the loop guard's.
            throw std::invalid_argument("constraint cannot be combined with reasoning-loop conclude");
        }
        if (options.execution.requested_output_tokens > constraint.steps.size()) {
            throw std::invalid_argument("constraint has fewer steps than requested output tokens");
        }
        std::vector<std::vector<TokenId>> sets;
        for (const std::vector<TokenId>& step : constraint.steps) {
            if (step.empty() || step.size() > kMaximumConstraintChoices) {
                throw std::invalid_argument(
                    "constraint step must permit 1 to kMaximumConstraintChoices tokens");
            }
            std::vector<TokenId> sorted = step;
            std::ranges::sort(sorted);
            if (std::ranges::adjacent_find(sorted) != sorted.end()) {
                throw std::invalid_argument("constraint step tokens must be distinct");
            }
            if (step.size() > 1 && std::ranges::find(sets, step) == sets.end()) {
                sets.push_back(step);
            }
        }
        if (sets.size() > kMaximumConstraintSets) {
            throw std::invalid_argument(
                "constraint uses more than kMaximumConstraintSets distinct multi-token steps");
        }
        if (!constraint.observe.empty() && constraint.observe.size() != constraint.steps.size()) {
            throw std::invalid_argument("constraint observe needs one list per step");
        }
        for (const std::vector<TokenId>& observed : constraint.observe) {
            if (observed.size() > kMaximumObservedTokens) {
                throw std::invalid_argument(
                    "constraint step observes more than kMaximumObservedTokens tokens");
            }
            std::vector<TokenId> sorted = observed;
            std::ranges::sort(sorted);
            if (std::ranges::adjacent_find(sorted) != sorted.end()) {
                throw std::invalid_argument("constraint observed tokens must be distinct");
            }
        }
    }
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling, options.constraint.has_value());
    resolved.execution.requested_output_tokens = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse      = options.execution.allow_prefix_reuse;
    resolved.execution.thinking                = options.execution.thinking;
    resolved.execution.readout_tokens          = std::move(options.execution.readout_tokens);
    resolved.execution.constraint              = std::move(options.execution.constraint);
    resolved.stop                            = std::move(options.stop);
    resolved.output                            = options.output;
    resolved.constraint                        = options.constraint;
    resolved.tool_choice                       = std::move(options.tool_choice);
    resolved.ngram_session                     = std::move(options.ngram_session);
    return resolved;
}

} // namespace

class PreparedPrompt::Impl {
public:
    Impl(PromptSummary prompt_summary, PromptPreparationStats preparation, SamplingMode mode,
         models::qwen3_5::PreparedPrompt prepared)
        : summary(std::move(prompt_summary)), prepare(std::move(preparation)), sampling_mode(mode),
          value(std::move(prepared)) {}

    PromptSummary summary;
    PromptPreparationStats prepare;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    models::qwen3_5::PreparedPrompt value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty;
    return impl_ != nullptr ? impl_->prepare : empty;
}

std::span<const TokenId> PreparedPrompt::token_ids() const noexcept {
    return impl_ != nullptr ? impl_->value.token_ids() : std::span<const TokenId>{};
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

class GenerationHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept() = default;
        virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
        virtual void await_admission(const CancellationView& cancellation) = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

        void await_admission(const CancellationView& cancellation) override {
            submission_.await_admission(cancellation);
        }

    private:
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission,
         ResolvedSamplingParameters sampling)
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))),
          sampling_(sampling) {}

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }

    void await_admission(const CancellationView& cancellation) { state_->await_admission(cancellation); }

    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

private:
    std::unique_ptr<Concept> state_;
    ResolvedSamplingParameters sampling_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

void GenerationHandle::await_admission(const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    impl_->await_admission(cancellation);
}

class Engine::Impl {
public:
    using GenerationCore = runtime::EngineCore<runtime::ModelInstance>;
    using ScoringCore    = runtime::CausalScoreCore<runtime::ModelInstance>;
    // Qwen3.8-Flash-Next: the common cores over its own Program.
    using Qwen4ExpCore        = runtime::EngineCore<runtime::Qwen4ExpInstance>;
    using Qwen4ExpScoringCore = runtime::CausalScoreCore<runtime::Qwen4ExpInstance>;
    using Core = std::variant<std::monostate, std::unique_ptr<GenerationCore>, std::unique_ptr<ScoringCore>,
                              std::unique_ptr<Qwen4ExpCore>, std::unique_ptr<Qwen4ExpScoringCore>>;
    template <class CoreState>
    static constexpr bool kScoring = std::is_same_v<CoreState, std::unique_ptr<ScoringCore>> ||
                                     std::is_same_v<CoreState, std::unique_ptr<Qwen4ExpScoringCore>>;

    explicit Impl(EngineOptions engine_options)
        : options(normalize(std::move(engine_options))),
          device(initialize_device(options, architecture == models::Architecture::Qwen4Exp && options.vram_past_budget)) {
        nvtx::ScopedRange load_range(nvtx::Name::EngineLoad, nvtx::Category::Runtime);
        // The lossless ops' rounding mode is process-wide device state: every lossless Engine sets it, so
        // a noise twin never leaks into a later lossless run.
        if (options.lossless) {
            ops::lossless_set_noise(options.lossless_noise_seed, options.lossless_noise_units,
                                    options.lossless_noise_scale);
        }
        if (architecture == models::Architecture::Qwen4Exp) {
            auto constructed  = runtime::construct_qwen4_exp(options, device);
            options           = std::move(constructed.options);
            qwen4             = std::move(constructed.instance);
            load              = std::move(constructed.load);
            model_metadata    = std::move(constructed.model_metadata);
            load.cuda_sync_mode = device.sync_mode();
            load.temporary_switches = temporary_switches::describe(temporary_switches::Scope::Qwen4Exp);
            sampling_defaults = qwen4->frontend.sampling_defaults();
            StartupPhaseScope finalize_phase(options.startup_observer, StartupPhase::EngineFinalize);
            if (options.purpose == EnginePurpose::CausalScoring) {
                core = std::make_unique<Qwen4ExpScoringCore>(*qwen4, device);
            } else {
                core = std::make_unique<Qwen4ExpCore>(*qwen4, device, options);
            }
            finalize_phase.complete();
            return;
        }
        const bool tree_auto_requested = options.speculative.draft_tree_auto;
        auto constructed               = runtime::construct_model(options, device);
        active                         = std::move(constructed.instance);
        load                           = std::move(constructed.load);
        model_metadata                 = std::move(constructed.model_metadata);
        load.cuda_sync_mode            = device.sync_mode();
        load.temporary_switches        = temporary_switches::describe(temporary_switches::Scope::Qwen3_5);
        if (tree_auto_requested && !options.speculative.draft_tree_auto) {
            runtime::publish_diagnostic(
                options.diagnostic_observer, DiagnosticLevel::Warning,
                "automatic tree verification does not run here (an artifact whose GDN input "
                "projections are not single FP8, NVFP4 or Q8 parents, fixed speculation, MTP without "
                "sampled-mtp-drafts or beside adaptive-mtp's draft lengths, or --draft-tokens "
                "auto's adaptive-dflash widths, which verify chains faster); rounds verify chains");
        }
        sampling_defaults = active->frontend.sampling_defaults();
        StartupPhaseScope finalize_phase(options.startup_observer, StartupPhase::EngineFinalize);
        if (options.purpose == EnginePurpose::CausalScoring) {
            core = std::make_unique<ScoringCore>(*active, device);
        } else {
            core = std::make_unique<GenerationCore>(*active, device, options);
        }
        finalize_phase.complete();
    }

    // Options are checked before the artifact is opened; defaults that depend on the model (the
    // hybrid prefix cache's, --spec auto's drafter) need the artifact's architecture and drafters, read
    // once here. An unreadable artifact is reported by the (Qwen3.5) model construction, after option
    // validation.
    EngineOptions normalize(EngineOptions engine_options) {
        bool offloaded_experts = false;
        std::optional<models::qwen3_5::BundledDrafters> drafters;
        try {
            const artifact::Reader reader(engine_options.artifact_path);
            if (runtime::artifact_is_qwen4_exp(reader)) {
                architecture = models::Architecture::Qwen4Exp;
            } else {
                offloaded_experts = runtime::artifact_offloads_experts(reader);
            }
            drafters = runtime::artifact_drafters(reader);
        } catch (const std::exception&) {}
        runtime::validate_engine_options(engine_options, architecture, offloaded_experts);
        if (drafters) {
            runtime::resolve_auto_speculative_backend(engine_options, architecture, offloaded_experts, *drafters);
        }
        return runtime::normalize_engine_options(std::move(engine_options), architecture, offloaded_experts);
    }

    ~Impl() noexcept {
        device.bind_to_current_thread_noexcept();
        const bool persists = persists_prefix_cache();
        stop();
        // Joins the generation core's orderly stop, which saves the Host tier before dropping it.
        core.emplace<std::monostate>();
        try {
            device.synchronize();
        } catch (...) {}
        if (persists) { report_prefix_cache_save(); }
    }

    void stop() noexcept {
        if (stop_requested.exchange(true)) { return; }
        if (persists_prefix_cache()) {
            runtime::publish_diagnostic(
                options.diagnostic_observer, DiagnosticLevel::Info, "saving the prefix cache to %s",
                options.context_cache.hybrid.persistent_file.string().c_str());
        }
        std::visit(
            [](auto& state) {
                using CoreState = std::remove_cvref_t<decltype(state)>;
                if constexpr (!std::is_same_v<CoreState, std::monostate>) { state->stop(); }
            },
            core);
    }

    [[nodiscard]] bool persists_prefix_cache() const noexcept {
        return (std::holds_alternative<std::unique_ptr<GenerationCore>>(core) ||
                std::holds_alternative<std::unique_ptr<Qwen4ExpCore>>(core)) &&
               options.context_cache.enabled &&
               !options.context_cache.hybrid.persistent_file.empty();
    }

    void report_prefix_cache_save() const noexcept {
        try {
            if (qwen4) {
                report_prefix_cache_save(qwen4->program->prefix_shutdown_save());
            } else {
                report_prefix_cache_save(active->program->hybrid_shutdown_save());
            }
        } catch (const std::exception& error) {
            runtime::publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                                        "prefix cache not saved: %s", error.what());
        } catch (...) {}
    }

    // Either model's shutdown-save result (the same fields).
    template <class Result>
    void report_prefix_cache_save(const std::optional<Result>& result) const {
        if (!result) {
            runtime::publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                                        "prefix cache not saved: the Engine did not stop cleanly");
            return;
        }
        const Result& saved = *result;
        if (saved.ok) {
            runtime::publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                                        "prefix cache saved: %llu blocks, %llu snapshots, "
                                        "%.1f MiB in %.1f s",
                                        static_cast<unsigned long long>(saved.blocks),
                                        static_cast<unsigned long long>(saved.snapshots),
                                        static_cast<double>(saved.bytes) / 1048576.0, saved.seconds);
        } else {
            runtime::publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                                        "prefix cache not saved: %s", saved.message.c_str());
        }
    }

    [[nodiscard]] const models::qwen3_5::Frontend& frontend() const {
        return qwen4 != nullptr ? qwen4->frontend : active->frontend;
    }

    [[nodiscard]] std::uint32_t capacity() const { return qwen4 != nullptr ? qwen4->capacity : active->capacity; }

    models::Architecture architecture = models::Architecture::Qwen3_5; // set by normalize
    EngineOptions options;
    DeviceContext device;
    std::unique_ptr<runtime::ModelInstance> active;
    std::unique_ptr<runtime::Qwen4ExpInstance> qwen4;
    LoadSummary load;
    ModelMetadata model_metadata;
    ModelSamplingDefaults sampling_defaults;
    Core core;
    std::atomic<bool> stop_requested{false};
};

Engine::Engine(EngineOptions options) {
    // Before any host thread starts: a throttled process runs every one of them slower.
    if (const auto refused = exempt_process_from_power_throttling()) {
        runtime::publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                                    "Windows refused the power-throttling exemption (error %lu): run in a "
                                    "background window, the host threads may be slowed many times over",
                                    *refused);
    }
    StartupObserver startup_observer = options.startup_observer;
    StartupPhaseScope startup_phase(startup_observer, StartupPhase::EngineStartup);
    impl_ = std::make_shared<Impl>(std::move(options));
    startup_phase.complete();
}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime);
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    auto prepared      = impl_->frontend().prepare(std::move(input), control);
    PromptSummary info = prepared.summary();
    const SamplingMode sampling_mode =
        info.starts_in_reasoning ? SamplingMode::Thinking : SamplingMode::NonThinking;
    if (info.prompt_tokens > impl_->capacity()) {
        throw std::logic_error("target Frontend admitted a prompt beyond Engine capacity");
    }
    const PromptPreparationStats preparation = prepared.preparation_stats();
    return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(info, preparation, sampling_mode,
                                                                 std::move(prepared)));
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime,
                                    static_cast<std::uint64_t>(token_ids.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (token_ids.size() > impl_->capacity()) {
        throw RequestError(ContextLengthDetails{.prompt_tokens = token_ids.size(),
                                                .max_context   = impl_->capacity()});
    }
    auto prepared = impl_->frontend().prepare_tokens(std::move(token_ids), allow_prefix_identity);
    PromptSummary info = prepared.summary();
    if (info.prompt_tokens > impl_->capacity()) {
        throw std::logic_error("target Frontend admitted prompt tokens beyond capacity");
    }
    const PromptPreparationStats preparation = prepared.preparation_stats();
    return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
        info, preparation, SamplingMode::Thinking, std::move(prepared)));
}

std::vector<TokenId> Engine::tokenize_text(std::string_view text) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->frontend().tokenize_text(text);
}

std::string Engine::detokenize(std::span<const TokenId> token_ids) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->frontend().detokenize(token_ids);
}

ChatTokens Engine::tokenize_chat(PromptInput input) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->frontend().tokenize_chat(std::move(input));
}

ScoreResult Engine::score_tokens(std::vector<TokenId> tokens, std::uint32_t first_target,
                                 ScoreOptions options) {
    nvtx::ScopedRange score_range(nvtx::Name::Score, nvtx::Category::Scoring,
                                  static_cast<std::uint64_t>(tokens.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::CausalScoring) {
        throw std::logic_error("score_tokens requires a CausalScoring Engine");
    }
    if (tokens.size() < 2 || tokens.size() > impl_->options.max_context) {
        throw std::invalid_argument("score_tokens token count must be in [2,max_context]");
    }
    if (first_target == 0 || first_target >= tokens.size()) {
        throw std::invalid_argument("score_tokens first_target must be in [1,token_count-1]");
    }
    const std::size_t positions = tokens.size() - first_target;
    if (options.top_k > kMaximumScoreTopTokens ||
        options.candidates_per_position > kMaximumScoreTopTokens) {
        throw std::invalid_argument("score_tokens top_k and candidates_per_position must be in "
                                    "[0,kMaximumScoreTopTokens]");
    }
    if (options.candidates.size() != positions * options.candidates_per_position) {
        throw std::invalid_argument(
            "score_tokens needs candidates_per_position candidates for every scored position");
    }
    if (options.logits_positions.size() > kMaximumScoreLogitPositions ||
        !std::ranges::is_sorted(options.logits_positions, std::ranges::less_equal{}) ||
        (!options.logits_positions.empty() && options.logits_positions.back() >= positions)) {
        throw std::invalid_argument("score_tokens logits_positions must be strictly ascending scored positions, at "
                                    "most kMaximumScoreLogitPositions");
    }
    const std::size_t logit_rows       = options.logits_positions.size();
    const std::size_t top_values       = positions * options.top_k;
    const std::size_t candidate_values = options.candidates.size();
    PreparedPrompt prompt              = prepare_tokens(std::move(tokens), false);
    ScoreResult result                 = std::visit(
        [&](auto& core) -> ScoreResult {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (Impl::kScoring<CoreState>) {
                return core->score(std::move(prompt.impl_->value), first_target,
                                   std::move(options));
            } else {
                throw std::logic_error("Engine scoring core is unavailable");
            }
        },
        impl_->core);
    if (result.logprobs.size() != positions || result.top_ids.size() != top_values ||
        result.top_logprobs.size() != top_values ||
        result.candidate_logprobs.size() != candidate_values ||
        (logit_rows == 0 ? !result.logits.empty() : result.logits.empty() || result.logits.size() % logit_rows != 0)) {
        throw std::logic_error("target Program returned an invalid causal score count");
    }
    return result;
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->frontend().count_tokens(std::move(input), control);
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->sampling_defaults;
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                OutputConsumerMode consumer_mode,
                                GenerationObservationOptions observation,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("submit requires a Generation Engine");
    }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }
    if (observation.live_timings) { observation.phase_timings = true; }
    if (consumer_mode != OutputConsumerMode::Streaming &&
        (observation.live_timings || observation.prompt_progress)) {
        throw std::invalid_argument("live generation observations require a Streaming consumer");
    }

    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(options));
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;
    const auto in_domain = [&](TokenId id) {
        return id >= 0 && static_cast<std::uint64_t>(id) < impl_->model_metadata.vocab_size;
    };
    for (const TokenId id : resolved_options.execution.readout_tokens) {
        if (!in_domain(id)) {
            throw std::invalid_argument("readout token is outside the tokenizer domain");
        }
    }
    for (const auto& step : resolved_options.execution.constraint.steps) {
        if (!std::ranges::all_of(step, in_domain)) {
            throw std::invalid_argument("constraint token is outside the tokenizer domain");
        }
    }

    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(ContextLengthDetails{.prompt_tokens = prompt_summary.prompt_tokens,
                                                .max_context   = impl_->options.max_context});
    }
    const double prepare_seconds = prompt.impl_->prepare.seconds;
    if (resolved_options.execution.requested_output_tokens == 0) {
        struct ImmediateSubmission {
            GenerationResult result;
            OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;

            GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
                const bool streaming = consumer_mode == OutputConsumerMode::Streaming;
                if (streaming != (sink != nullptr)) {
                    throw std::invalid_argument(
                        "GenerationHandle wait sink does not match its submitted consumer mode");
                }
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                return std::move(result);
            }

            // Answered on submit: no queue, so no admission to wait for.
            void await_admission(const CancellationView&) const noexcept {}
        } immediate{.consumer_mode = consumer_mode};

        immediate.result.prompt                     = prompt_summary;
        immediate.result.finish_reason              = FinishReason::OutputLimit;
        immediate.result.thinking.configured_budget = resolved_options.execution.thinking.budget;
        immediate.result.timings.prepare_seconds    = prepare_seconds;
        immediate.result.timings.total_seconds      = prepare_seconds;
        prompt.impl_.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling));
    }

    return std::visit(
        [&](auto& core) -> GenerationHandle {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else if constexpr (Impl::kScoring<CoreState>) {
                throw std::logic_error("Engine generation core is unavailable");
            } else {
                auto submission =
                    core->submit(std::move(prompt.impl_->value), prompt_summary, prepare_seconds,
                                 std::move(resolved_options), consumer_mode, std::move(observation),
                                 pending_deadline);
                return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
                    impl_, std::move(submission), resolved_sampling));
            }
        },
        impl_->core);
}

void Engine::prefill_next_turn(PromptInput conversation) noexcept {
    if (impl_ == nullptr) { return; }
    std::visit(
        [&](auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (!std::is_same_v<CoreState, std::monostate> && !Impl::kScoring<CoreState>) {
                if (core != nullptr) { core->prefill_next_turn(std::move(conversation)); }
            }
        },
        impl_->core);
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    const OutputConsumerMode consumer_mode =
        sink != nullptr ? OutputConsumerMode::Streaming : OutputConsumerMode::Aggregate;
    return submit(std::move(prompt), std::move(options), consumer_mode, {})
        .wait(sink, cancellation);
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load;
}

ModelMetadata Engine::model_metadata() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->model_metadata;
}

MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> MemorySummary {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->memory_summary();
            }
        },
        impl_->core);
}

MediaCacheSummary Engine::media_cache_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->frontend().media_cache_summary();
}

ChatTemplateRoleSupport Engine::chat_template_role_support() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->frontend().chat_template_role_support();
}

RuntimeStats Engine::runtime_stats() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> RuntimeStats {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->runtime_stats();
            }
        },
        impl_->core);
}

bool Engine::is_available() const {
    if (impl_ == nullptr) { return false; }
    return std::visit(
        [](const auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                return false;
            } else {
                return core != nullptr && core->is_available();
            }
        },
        impl_->core);
}

bool Engine::has_failed() const {
    if (impl_ == nullptr) { return false; }
    return std::visit(
        [](const auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                return false;
            } else if constexpr (requires { core->has_failed(); }) {
                return core != nullptr && core->has_failed();
            } else {
                return false; // a causal scoring core fails its call instead
            }
        },
        impl_->core);
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr) { return; }
    std::visit(
        [](auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (!std::is_same_v<CoreState, std::monostate>) {
                core->reset_memory_peaks();
            }
        },
        impl_->core);
}

void Engine::reset_execution_maxima() noexcept {
    if (impl_ == nullptr) { return; }
    std::visit(
        [](auto& core) {
            if constexpr (requires { core->reset_execution_maxima(); }) {
                core->reset_execution_maxima();
            }
        },
        impl_->core);
}

void Engine::stop() noexcept {
    if (impl_ != nullptr) { impl_->stop(); }
}

} // namespace infernix
