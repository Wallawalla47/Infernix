#include "core/device.h"
#include "models/load_options.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/prefix/test_seams.h"
#include "models/qwen3_5/program/program.h"
#include "models/qwen3_5/program/program_impl.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace infernix;
namespace qwen = models::qwen3_5;

constexpr std::uint32_t kCapacity      = 512;
constexpr std::uint32_t kChunk         = 128;
constexpr std::uint32_t kPromptTokens  = 384;
constexpr std::uint32_t kPauseFrontier = 256;
// The cached lifecycle's long prompt: past the 27B DFlash2 drafter's ring window, paused far from any
// planned read.
constexpr std::uint32_t kLongContext = 8192;
constexpr std::uint32_t kLongPrompt  = 6144;
constexpr std::uint32_t kLongPause   = 3072;

void require(bool condition, std::string_view message) {
    if (!condition) { throw std::runtime_error(std::string(message)); }
}

// A Program without a prefix cache: every source is a root start, and a paused request recovers
// by replaying its ledger.
EngineOptions program_options(SpeculativeBackend backend) {
    EngineOptions options;
    options.max_context                                = kCapacity;
    options.kv_capacity                                = KvCapacityPolicy::explicit_capacity(2 * kCapacity);
    options.prefill_chunk                              = kChunk;
    options.max_concurrency                            = 2;
    options.kv_cache                                   = KvCacheStorage::Fp8E4M3Row256;
    options.use_cuda_graph                             = false;
    options.speculative.backend                        = backend;
    options.speculative.draft_tokens                   = backend == SpeculativeBackend::None ? 0U : 3U;
    options.context_cache.enabled                      = false;
    options.context_cache.hybrid.device_snapshot_slots = 0;
    options.context_cache.host_capacity_bytes          = 0;
    return options;
}

class Fixture {
public:
    Fixture(DeviceContext& device, qwen::Program& program, qwen::Frontend& frontend)
        : device_(device), program_(program), frontend_(frontend),
          empty_(program.physical_usage()) {}

    qwen::RequestBasePlan request(std::uint32_t tokens) {
        std::vector<TokenId> ids(tokens, 198);
        ids.front() = 1000;
        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens    = 8;
        options.allow_prefix_reuse         = false;
        options.sampling.presence_penalty  = 0.5F;
        options.sampling.frequency_penalty = 0.25F;
        return program_.plan_request(frontend_.prepare_tokens(std::move(ids)), options);
    }

    qwen::ContextProgress settle() {
        for (unsigned step = 0; step < 8; ++step) {
            device_.synchronize();
            auto progress = program_.poll_context({});
            if (progress.complete) {
                require(!program_.has_context_transaction(), "completed context remains pending");
                return progress;
            }
        }
        throw std::runtime_error("bounded native context transaction did not complete");
    }

    qwen::SequenceHandle
    bind(const qwen::RequestBasePlan& base, std::uint32_t lane = 0,
         qwen::ResumeState* resume           = nullptr,
         qwen::ExecutionUnitKind resume_kind = qwen::ExecutionUnitKind::Decode) {
        const auto sources = program_.hybrid_sources(base, nullptr);
        require(sources.size() == 1 && sources.front().reused_tokens == 0,
                "a Program without a prefix cache quoted more than a root start");
        require(static_cast<bool>(program_.start_binding(
                    base, {lane}, sources.front(), resume, resume_kind,
                    resume_kind == qwen::ExecutionUnitKind::Prefill ? 0 : 1)),
                "request binding could not reserve its first unit");
        auto progress = settle();
        require(progress.published && progress.sequence.has_value(),
                "binding did not publish a lane");
        require(!resume || progress.replaying, "resumed binding did not enter replay");
        return *progress.sequence;
    }

    void reserve(qwen::SequenceHandle sequence, qwen::ExecutionUnitKind kind,
                 std::uint32_t tokens = 0) {
        const std::array<qwen::ExecutionUnit, 1> units{{{sequence, kind, tokens}}};
        require(static_cast<bool>(program_.reserve_units(units)),
                "finite native unit was not reserved");
    }

    void prefill_prefix(qwen::SequenceHandle sequence, std::uint32_t tokens) {
        std::uint32_t processed = 0;
        while (processed < tokens) {
            reserve(sequence, qwen::ExecutionUnitKind::Prefill);
            auto step = program_.advance_prefill(sequence);
            require(!step.complete && !step.pending, "partial prefill sampled before its target");
            require(step.processed_prompt_tokens != 0, "partial prefill made no progress");
            processed += step.processed_prompt_tokens;
        }
        require(processed == tokens, "partial prefill crossed its requested frontier");
    }

    void finish_prefill(qwen::SequenceHandle sequence, std::uint32_t remaining, bool terminal) {
        std::uint32_t processed = 0;
        for (unsigned iteration = 0; iteration < remaining / kChunk + 4; ++iteration) {
            const bool recovering = program_.recovery_pending(sequence);
            reserve(sequence, qwen::ExecutionUnitKind::Prefill);
            auto step = program_.advance_prefill(sequence);
            processed += step.processed_prompt_tokens;
            if (!step.complete) {
                require(step.processed_prompt_tokens != 0, "prefill made no observable progress");
                continue;
            }
            require(step.pending && step.pending->row_count() == 1,
                    "completed prefill did not return one pending row");
            require(!recovering || program_.recovery_pending(sequence),
                    "uncommitted prefill output released recovery protection");
            require(processed == remaining, "resumed prefill repeated or skipped committed input");
            const std::array<runtime::CommitDecision, 1> decision{
                {{.accepted_tokens = 1, .terminal = terminal}}};
            auto result = program_.commit(std::move(*step.pending), decision);
            require(result.row_count == 1 && result.rows[0].disposition ==
                                                 (terminal ? runtime::CommitDisposition::Finishable
                                                           : runtime::CommitDisposition::Active),
                    "prefill commit returned the wrong request phase");
            return;
        }
        throw std::runtime_error("prefill exceeded its bounded chunk count");
    }

    void decode_and_finish(qwen::SequenceHandle sequence) {
        reserve(sequence, qwen::ExecutionUnitKind::Decode, 1);
        const std::array<qwen::SequenceHandle, 1> members{sequence};
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto pending = program_.decode(members, budgets);
        const std::array<runtime::CommitDecision, 1> decision{
            {{.accepted_tokens = 1, .terminal = true}}};
        auto committed = program_.commit(std::move(pending), decision);
        require(committed.rows[0].disposition == runtime::CommitDisposition::Finishable,
                "request could not decode after recovery");
        require(program_.finish(sequence).status == runtime::ConsumeStatus::Consumed,
                "finished request did not release its execution resources");
    }

    void expect_empty(std::string_view label) {
        require(!program_.has_context_transaction(), "cleanup left a context operation pending");
        require(program_.physical_usage().occupied == empty_.occupied, label);
    }

    void next_request() {
        auto base           = request(kChunk);
        const auto sequence = bind(base);
        finish_prefill(sequence, kChunk, false);
        decode_and_finish(sequence);
        expect_empty("next request did not return typed stores to the initial baseline");
    }

    // A paused request releases its lane and resumes by replaying its ledger: a repeated pause
    // keeps the original target, a stale handle cannot consume the new lane, and destroying the
    // ResumeState of a paused replay releases everything.
    void replay_pause_and_resume() {
        auto base           = request(kPromptTokens);
        const auto original = bind(base);
        prefill_prefix(original, kPauseFrontier);
        require(program_.start_pause(original), "prefill pause was not accepted");
        auto first = settle();
        require(first.paused && first.paused->frontier() == kPauseFrontier,
                "prefill pause lost its replay target");
        expect_empty("pause did not release its execution resources");

        const auto replay = bind(base, 1, &*first.paused, qwen::ExecutionUnitKind::Prefill);
        first.paused.reset();
        reserve(replay, qwen::ExecutionUnitKind::Replay);
        const auto partial = program_.advance_replay(replay);
        require(partial.processed_tokens == kChunk && !partial.complete,
                "replay did not yield after its first finite chunk");
        require(program_.start_pause(replay), "partially replayed request could not pause again");
        auto second = settle();
        require(second.paused && second.paused->frontier() == kPauseFrontier,
                "repeated replay pause replaced the original target with its partial cursor");
        expect_empty("repeated replay pause retained execution resources");

        const auto resumed = bind(base, 0, &*second.paused, qwen::ExecutionUnitKind::Prefill);
        second.paused.reset();
        require(program_.abort(original).status == runtime::ConsumeStatus::InvariantMismatch &&
                    program_.abort(replay).status == runtime::ConsumeStatus::InvariantMismatch,
                "a stale pre-pause sequence handle consumed a newly bound lane");
        std::uint32_t replayed = 0;
        bool complete          = false;
        for (unsigned step = 0; step < kPauseFrontier / kChunk + 1; ++step) {
            reserve(resumed, qwen::ExecutionUnitKind::Replay);
            const auto progress = program_.advance_replay(resumed);
            replayed += progress.processed_tokens;
            if (progress.complete) {
                complete = true;
                break;
            }
        }
        require(complete && replayed == kPauseFrontier,
                "second replay did not recover the entire original target");
        finish_prefill(resumed, kPromptTokens - kPauseFrontier, false);
        decode_and_finish(resumed);
        expect_empty("replayed request retained physical resources after finishing");
        next_request();

        const auto to_cancel = bind(base);
        prefill_prefix(to_cancel, kPauseFrontier);
        require(program_.start_pause(to_cancel), "cancel fixture could not pause prefill");
        auto saved = settle();
        require(saved.paused.has_value(), "cancel fixture lost its paused prefill record");
        const auto rebuilding = bind(base, 1, &*saved.paused, qwen::ExecutionUnitKind::Prefill);
        saved.paused.reset();
        reserve(rebuilding, qwen::ExecutionUnitKind::Replay);
        require(!program_.advance_replay(rebuilding).complete,
                "cancel fixture replay completed early");
        require(program_.start_pause(rebuilding), "cancel fixture could not pause replay");
        auto cancelled = settle();
        require(cancelled.paused && cancelled.paused->frontier() == kPauseFrontier,
                "paused replay cancellation lost its request target");
        cancelled.paused.reset();
        expect_empty("destroyed paused replay retained typed resources");
        next_request();
    }

    // A request paused after its binding and before its first prefill chunk (frontier 0). NInfer #434's
    // server-wide stall was this case: the resume became an empty Replay its scheduler never granted, while
    // the request kept the recovery permit, so nothing ran for hours. Here the resume's one Replay unit is
    // reserved and completes at once with nothing processed, then the whole prompt prefills, decodes and
    // finishes, and the stores return to their baseline.
    void pause_before_first_prefill_chunk() {
        auto base           = request(kPromptTokens);
        const auto original = bind(base);
        require(program_.start_pause(original), "a pause before the first prefill chunk was not accepted");
        auto paused = settle();
        require(paused.paused && paused.paused->frontier() == 0,
                "a pause before the first prefill chunk recorded a frontier");
        expect_empty("a pause before the first prefill chunk did not release its execution resources");

        const auto resumed = bind(base, 1, &*paused.paused, qwen::ExecutionUnitKind::Prefill);
        paused.paused.reset();
        reserve(resumed, qwen::ExecutionUnitKind::Replay);
        const auto replay = program_.advance_replay(resumed);
        require(replay.complete && replay.processed_tokens == 0,
                "an empty replay did not complete at once with nothing processed");
        finish_prefill(resumed, kPromptTokens, false);
        decode_and_finish(resumed);
        expect_empty("a request resumed from frontier 0 retained physical resources");
        next_request();
    }

    void grammar_row_failure() {
        struct Masks final : runtime::TokenMaskProvider {
            void uploaded(std::size_t, std::size_t) noexcept override {}

            bool constrained(std::size_t row) const noexcept override { return row == 0; }

            std::uint64_t fill(std::size_t, std::span<const TokenId> drafts,
                               std::span<std::uint32_t> words) override {
                std::fill(words.begin(), words.end(), 0);
                const auto stride = words.size() / (drafts.size() + 1);
                for (std::size_t col = 0; col <= drafts.size(); ++col) words[col * stride] = 1;
                return 1; // The first predicted position is a real dead end.
            }
        } masks;

        auto base        = request(32);
        const auto first = bind(base, 0);
        finish_prefill(first, 32, false);
        const auto second = bind(base, 1);
        finish_prefill(second, 32, false);
        const std::array<qwen::SequenceHandle, 2> members{first, second};
        const std::array<qwen::ExecutionUnit, 2> units{
            {{first, qwen::ExecutionUnitKind::Decode, 1},
             {second, qwen::ExecutionUnitKind::Decode, 1}}};
        require(static_cast<bool>(program_.reserve_units(units)), "grammar mixed unit reservation");
        const std::array<runtime::RoundBudget, 2> budgets{{{1}, {1}}};
        auto pending = program_.decode(members, budgets, nullptr, &masks);
        require(pending.row_failure(0) == runtime::PendingRowFailure::ConstraintDeadEnd &&
                    pending.row_failure(1) == runtime::PendingRowFailure::None,
                "grammar failure lost its row");
        const std::array<runtime::CommitDecision, 2> decisions{
            {{.terminal = true, .failed = true}, {.accepted_tokens = 1}}};
        auto committed = program_.commit(std::move(pending), decisions);
        require(committed.rows[0].disposition == runtime::CommitDisposition::FailedReleased &&
                    committed.rows[1].disposition == runtime::CommitDisposition::Active,
                "grammar failure contaminated a healthy row");
        require(program_.abort(second).status == runtime::ConsumeStatus::Consumed,
                "healthy row was released by another row's grammar error");
        expect_empty("grammar row failure leaked physical resources");
    }

private:
    DeviceContext& device_;
    qwen::Program& program_;
    qwen::Frontend& frontend_;
    qwen::PhysicalUsageSnapshot empty_;
};

// Replay restores a lane's occurrence counts (presence and frequency penalties) from its committed
// output and reproduces the ledger's execution splits; the exact oracle counts committed output
// events, independently of the execution route.
void replay_sampling_counts(DeviceContext& device, const qwen::execution::Parameters& parameters,
                            qwen::Frontend& frontend, EngineOptions options) {
    options.max_context = 128;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(128);
    auto planner        = qwen::detail::make_sequence_planner_impl(parameters, device, options);
    auto plan           = qwen::detail::finalize_sequence_plan_impl(std::move(planner), 2);
    qwen::detail::ProgramImpl program(parameters, *plan, device, {});

    // The prompt token appears many times but contributes no occurrences of its own.
    const std::vector<TokenId> prompt(32, 200);
    auto prepared = frontend.prepare_tokens(prompt);
    runtime::ResolvedExecutionOptions request;
    request.requested_output_tokens    = 8;
    request.allow_prefix_reuse         = false;
    request.sampling.presence_penalty  = 0.5F;
    request.sampling.frequency_penalty = 0.25F;
    auto base =
        program.plan_request(qwen::PreparedPromptAccess::take(std::move(prepared)), request);

    const auto settle = [&]() {
        for (unsigned step = 0; step < 8; ++step) {
            device.synchronize();
            auto progress = program.poll_context({});
            if (progress.complete) { return progress; }
        }
        throw std::runtime_error("sampling-count recovery transaction did not complete");
    };
    const auto bind = [&](std::uint32_t lane, qwen::ResumeState* resume) {
        const auto sources = program.hybrid_sources(base, nullptr);
        require(sources.size() == 1 &&
                    static_cast<bool>(program.start_binding(base, {lane}, sources.front(), resume,
                                                            qwen::ExecutionUnitKind::Decode, 1)),
                "sampling-count fixture could not bind its lane");
        auto progress = settle();
        require(progress.published && progress.sequence,
                "sampling-count fixture did not publish a sequence");
        require(!resume || progress.replaying, "sampling-count resume did not enter replay");
        return *progress.sequence;
    };
    const auto reserve = [&](qwen::SequenceHandle sequence, qwen::ExecutionUnitKind kind,
                             std::uint32_t tokens = 0) {
        const std::array<qwen::ExecutionUnit, 1> units{{{sequence, kind, tokens}}};
        require(static_cast<bool>(program.reserve_units(units)),
                "sampling-count fixture could not reserve a finite unit");
    };

    const auto initial = bind(0, nullptr);
    reserve(initial, qwen::ExecutionUnitKind::Prefill);
    auto begin = program.advance_prefill(initial, nullptr, nullptr);
    require(begin.complete && begin.pending && begin.pending->tokens().size() == 1,
            "sampling-count fixture did not produce exactly one Begin token");
    const TokenId first = begin.pending->tokens().front();
    const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
    const auto committed = program.commit(std::move(*begin.pending), accepted,
                                          runtime::CommitObservation::AllRows, nullptr);
    require(committed.rows[0].disposition == runtime::CommitDisposition::Active,
            "sampling-count fixture did not commit Begin");

    const std::array<TokenId, 3> forced{198, 198, 199};
    const std::array<qwen::SequenceHandle, 1> members{initial};
    const std::array<std::optional<std::uint32_t>, 1> splits{2U};
    reserve(initial, qwen::ExecutionUnitKind::Control, forced.size());
    (void)program.append_forced_tokens(members, forced, forced.size(), splits, nullptr);

    const auto vocabulary =
        static_cast<std::int32_t>(parameters.model.resources().public_token_count);
    std::vector<std::int32_t> expected_counts(static_cast<std::size_t>(vocabulary), 0);
    ++expected_counts.at(static_cast<std::size_t>(first));
    for (const TokenId token : forced) { ++expected_counts.at(static_cast<std::size_t>(token)); }
    std::vector<TokenId> expected_ledger = prompt;
    expected_ledger.push_back(first);
    expected_ledger.insert(expected_ledger.end(), forced.begin(), forced.end());
    const std::uint32_t expected_frontier = static_cast<std::uint32_t>(expected_ledger.size() - 1U);
    const std::array<std::uint32_t, 1> expected_splits{
        static_cast<std::uint32_t>(prompt.size() + 1U + *splits.front())};

    const auto check = [&](std::uint32_t lane, std::string_view phase) {
        std::vector<std::int32_t> counts(expected_counts.size());
        const auto source =
            program.token_counts.slice(1, static_cast<std::int32_t>(lane), 1).view({vocabulary});
        device.synchronize();
        CUDA_CHECK(cudaMemcpy(counts.data(), source.data, source.bytes(), cudaMemcpyDeviceToHost));
        require(counts == expected_counts,
                std::string(phase) + ": sampling counts differ from committed output occurrences");
        const auto& sequence     = program.sequences[lane];
        const auto actual_splits = sequence.ledger_splits.frontiers();
        require(sequence.ledger == expected_ledger &&
                    sequence.execution_frontier == expected_frontier &&
                    sequence.ledger_frontier == expected_frontier + 1U &&
                    sequence.ledger_splits.size() == expected_ledger.size() &&
                    actual_splits.size() == expected_splits.size() &&
                    std::equal(actual_splits.begin(), actual_splits.end(), expected_splits.begin()),
                std::string(phase) +
                    ": replay changed committed ledger, E/S or forced execution split");
    };
    check(0, "after forced control");
    require(program.start_pause(initial, nullptr), "sampling-count fixture could not pause");
    auto paused = settle();
    require(paused.paused && paused.paused->frontier() == expected_frontier,
            "sampling-count fixture did not retain its committed replay target");
    const auto resumed = bind(1, &*paused.paused);
    paused.paused.reset();
    check(1, "after lane rebinding");
    bool recovered         = false;
    std::uint32_t replayed = 0;
    for (unsigned step = 0; step < 3; ++step) {
        reserve(resumed, qwen::ExecutionUnitKind::Replay);
        const auto progress = program.advance_replay(resumed, nullptr);
        replayed += progress.processed_tokens;
        check(1, "after replay chunk");
        if (progress.complete) {
            recovered = true;
            break;
        }
    }
    require(recovered && replayed == expected_frontier,
            "sampling-count fixture did not finish replaying its execution frontier");
    require(program.abort(resumed).status == runtime::ConsumeStatus::Consumed,
            "sampling-count fixture could not release its restored lane");
    require(program.physical_usage().occupied == runtime::ContextResourceUsage{},
            "sampling-count fixture leaked typed resources");
}

// The prefix cache's pause lifecycle (resource-scheduling §5.2-5.3), on a Device-only cache:
// - pause_replay_tokens, which a pressure turn uses to choose its victim, names the tokens the resume
//   recomputes: a pause that publishes the lane's state at its frontier resumes from it and replays
//   nothing, one that cannot replays from the deepest snapshot (decode and prefill pauses);
// - a resume binding cancelled while its restore is staged releases everything it took, leaves the
//   index consistent, and the paused request still resumes from the same source;
// - shutdown cleanup during a staged resume binding (an Engine stop) leaves no pin behind.
// Pins are checked through the index: with no lane bound, every Device-resident block is evictable.
void cached_pause_lifecycle(DeviceContext& device, const qwen::execution::Parameters& parameters,
                            EngineOptions options) {
    const qwen::Frontend frontend =
        qwen::make_frontend(parameters.model.resources(), {.vision_enabled = false, .max_context = kLongContext});
    options.max_context                                = kLongContext;
    options.kv_capacity                                = KvCapacityPolicy::explicit_capacity(2 * kLongContext);
    options.context_cache.enabled                      = true;
    options.context_cache.hybrid.device_snapshot_slots = 4;
    options.context_cache.hybrid.max_new_taps          = 2;
    options.context_cache.hybrid.tap_ladder_tokens     = 4096;
    options.context_cache.hybrid.tap_min_gap_tokens    = 1024;
    // Main KV pages: two long lanes and the cached blocks of finished requests.
    auto planner = qwen::detail::make_sequence_planner_impl(parameters, device, options);
    auto plan    = qwen::detail::finalize_sequence_plan_impl(std::move(planner), 3 * kLongContext / 64U);
    qwen::detail::ProgramImpl program(parameters, *plan, device, {});
    require(program.hybrid_ != nullptr, "the cached fixture built no prefix cache");
    auto& index = program.hybrid_->index();

    const auto prompt_ids = [](TokenId first, std::uint32_t tokens) {
        std::vector<TokenId> ids(tokens, 198);
        for (std::size_t at = 0; at < ids.size(); at += 7) { ids[at] = first + static_cast<TokenId>(at % 97); }
        return ids;
    };
    // `unechoed`: the prompt's PreparedPromptData::prompt_end_tap_token, as the Frontend sets it for a prompt that
    // opens the reasoning.
    const auto request_of = [&](std::vector<TokenId> ids, std::optional<TokenId> unechoed = std::nullopt,
                                bool reuse = true) {
        runtime::ResolvedExecutionOptions execution;
        execution.requested_output_tokens = 16;
        execution.allow_prefix_reuse      = reuse;
        qwen::PreparedPromptData data = qwen::PreparedPromptAccess::take(frontend.prepare_tokens(std::move(ids)));
        data.prompt_end_tap_token     = unechoed;
        return program.plan_request(std::move(data), execution);
    };
    const auto request = [&](TokenId first, std::uint32_t tokens = kPromptTokens) {
        return request_of(prompt_ids(first, tokens));
    };
    const auto settle = [&](runtime::CancellationFlagView cancellation = {}) {
        for (unsigned step = 0; step < 8; ++step) {
            device.synchronize();
            auto progress = program.poll_context(cancellation);
            if (progress.complete) { return progress; }
        }
        throw std::runtime_error("cached fixture transaction did not complete");
    };
    struct Bound {
        qwen::SequenceHandle sequence;
        std::uint32_t reused = 0;
    };
    const auto start = [&](const qwen::RequestBasePlan& base, std::uint32_t lane, qwen::ResumeState* resume,
                           qwen::ExecutionUnitKind kind) {
        const auto sources = program.hybrid_sources(base, resume);
        require(!sources.empty(), "the cached fixture quoted no source");
        require(static_cast<bool>(program.start_binding(base, {lane}, sources.front(), resume, kind,
                                                        kind == qwen::ExecutionUnitKind::Prefill ? 0U : 1U)),
                "the cached fixture could not reserve its first unit");
        return sources.front().reused_tokens;
    };
    const auto bind = [&](const qwen::RequestBasePlan& base, std::uint32_t lane, qwen::ResumeState* resume,
                          qwen::ExecutionUnitKind kind) {
        const std::uint32_t reused = start(base, lane, resume, kind);
        auto progress              = settle();
        require(progress.published && progress.sequence, "the cached fixture did not publish a lane");
        return Bound{*progress.sequence, reused};
    };
    const auto reserve = [&](qwen::SequenceHandle sequence, qwen::ExecutionUnitKind kind, std::uint32_t tokens = 0) {
        const std::array<qwen::ExecutionUnit, 1> units{{{sequence, kind, tokens}}};
        require(static_cast<bool>(program.reserve_units(units)), "the cached fixture could not reserve a unit");
    };
    // Prefills `tokens` more prompt tokens; reaching the prompt's end commits its first output token.
    const auto prefill = [&](qwen::SequenceHandle sequence, std::uint32_t tokens) {
        std::uint32_t processed = 0;
        while (processed < tokens) {
            reserve(sequence, qwen::ExecutionUnitKind::Prefill);
            auto step = program.advance_prefill(sequence, nullptr, nullptr);
            require(step.processed_prompt_tokens != 0, "cached prefill made no progress");
            processed += step.processed_prompt_tokens;
            if (step.complete) {
                const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
                (void)program.commit(std::move(*step.pending), accepted, runtime::CommitObservation::AllRows, nullptr);
            }
        }
        require(processed == tokens, "cached prefill crossed its frontier");
    };
    const auto decode = [&](qwen::SequenceHandle sequence, bool terminal) {
        reserve(sequence, qwen::ExecutionUnitKind::Decode, 1);
        const std::array<qwen::SequenceHandle, 1> members{sequence};
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto pending = program.decode(members, budgets, nullptr, nullptr);
        const std::array<runtime::CommitDecision, 1> decision{{{.accepted_tokens = 1, .terminal = terminal}}};
        (void)program.commit(std::move(pending), decision, runtime::CommitObservation::AllRows, nullptr);
    };
    const auto replay = [&](qwen::SequenceHandle sequence) {
        std::uint32_t replayed = 0;
        for (unsigned step = 0; step < kLongPrompt / kChunk + 4; ++step) {
            reserve(sequence, qwen::ExecutionUnitKind::Replay);
            const auto progress = program.advance_replay(sequence, nullptr);
            replayed += progress.processed_tokens;
            if (progress.complete) { return replayed; }
        }
        throw std::runtime_error("cached replay did not complete");
    };
    const auto unpinned = [&](std::string_view label) {
        device.synchronize();
        index.check_invariants();
        const std::uint32_t pinned = index.device_pinned_nodes();
        require(pinned == 0, std::string(label) + ": " + std::to_string(pinned) + " of " +
                                 std::to_string(index.stats().device_resident_blocks) +
                                 " Device blocks stayed pinned with no lane bound");
    };
    // Pauses the request bound on lane 0 and resumes it on lane 1; the prediction must name what the
    // resume recomputes.
    const auto pause_and_resume = [&](const qwen::RequestBasePlan& base, Bound bound, qwen::ExecutionUnitKind kind,
                                      std::string_view label) {
        const std::uint32_t predicted = program.pause_replay_tokens(bound.sequence);
        require(program.start_pause(bound.sequence, nullptr), std::string(label) + ": pause refused");
        auto paused = settle();
        require(paused.paused.has_value(), std::string(label) + ": pause lost its request");
        const std::uint32_t frontier = paused.paused->frontier();
        unpinned(std::string(label) + " paused");
        const Bound resumed = bind(base, 1, &*paused.paused, kind);
        paused.paused.reset();
        const std::uint32_t replayed = replay(resumed.sequence);
        std::cout << label << ": paused at " << frontier << ", predicted " << predicted << " to replay, resumed from "
                  << resumed.reused << ", replayed " << replayed << '\n';
        require(resumed.reused + predicted == frontier && replayed == predicted,
                std::string(label) + ": pause_replay_tokens predicted " + std::to_string(predicted) +
                    " tokens, the resume reused " + std::to_string(resumed.reused) + " of " +
                    std::to_string(frontier) + " and replayed " + std::to_string(replayed));
        return resumed;
    };

    // A decode pause.
    {
        const auto base = request(1000);
        Bound bound     = bind(base, 0, nullptr, qwen::ExecutionUnitKind::Prefill);
        prefill(bound.sequence, kPromptTokens);
        for (int round = 0; round < 3; ++round) { decode(bound.sequence, false); }
        const Bound resumed = pause_and_resume(base, bound, qwen::ExecutionUnitKind::Decode, "decode pause");
        decode(resumed.sequence, true);
        require(program.finish(resumed.sequence).status == runtime::ConsumeStatus::Consumed,
                "the resumed request did not finish");
        unpinned("decode pause finished");
    }
    // A prefill pause on the chunk grid.
    {
        const auto base = request(2000);
        Bound bound     = bind(base, 0, nullptr, qwen::ExecutionUnitKind::Prefill);
        prefill(bound.sequence, kPauseFrontier);
        const Bound resumed = pause_and_resume(base, bound, qwen::ExecutionUnitKind::Prefill, "prefill pause");
        prefill(resumed.sequence, kPromptTokens - kPauseFrontier);
        decode(resumed.sequence, true);
        require(program.finish(resumed.sequence).status == runtime::ConsumeStatus::Consumed,
                "the resumed prefill did not finish");
        unpinned("prefill pause finished");
    }
    // A prefill pause far into a prompt longer than the DFlash2 drafter's ring window: with that drafter,
    // YE-01 skipped the positions no planned read reaches, so the pause publishes nothing and the resume
    // replays from the deepest snapshot; the prediction must say so (other backends publish there).
    {
        const auto base = request(5000, kLongPrompt);
        Bound bound     = bind(base, 0, nullptr, qwen::ExecutionUnitKind::Prefill);
        prefill(bound.sequence, kLongPause);
        const Bound resumed = pause_and_resume(base, bound, qwen::ExecutionUnitKind::Prefill, "long prefill pause");
        prefill(resumed.sequence, kLongPrompt - kLongPause);
        decode(resumed.sequence, true);
        require(program.finish(resumed.sequence).status == runtime::ConsumeStatus::Consumed,
                "the resumed long prefill did not finish");
        unpinned("long prefill pause finished");
    }
    // A flexible tap skipped for lack of room stays planned and is taken at the next chunk boundary (after NInfer):
    // with every Device snapshot slot held, the long prompt's first planned tap is skipped; once the slots return,
    // the next boundary takes it, a chunk deeper (with DFlash2 its ring is whole there: the tap stayed a planned
    // read for YE-01), counting one skip and one retry, and a prompt departing just past it resumes there.
    {
        const auto ids  = prompt_ids(9000, kLongPrompt);
        const auto base = request_of(ids);
        const Bound bound = bind(base, 0, nullptr, qwen::ExecutionUnitKind::Prefill);
        std::vector<std::uint32_t> held;
        while (const auto slot = index.acquire_device_slot()) { held.push_back(*slot); }
        require(!held.empty(), "tap retry: no Device snapshot slot to hold");
        const auto before    = program.hybrid_stats();
        std::uint32_t cursor = 0;
        bool skipped         = false;
        while (!skipped && cursor + 2U * kChunk < kLongPrompt) {
            prefill(bound.sequence, kChunk);
            cursor += kChunk;
            skipped = program.hybrid_stats().taps_skipped_no_slot > before.taps_skipped_no_slot;
        }
        require(skipped, "tap retry: no planned tap was skipped with every slot held");
        for (const std::uint32_t slot : held) { index.release_device_slot(slot); }
        prefill(bound.sequence, kChunk);
        cursor += kChunk;
        const auto after = program.hybrid_stats();
        std::cout << "tap retry: skipped at " << cursor - kChunk << ", retried at " << cursor << ": skipped "
                  << after.taps_skipped_no_slot - before.taps_skipped_no_slot << ", retried "
                  << after.taps_retried - before.taps_retried << '\n';
        require(after.taps_retried == before.taps_retried + 1 &&
                    after.taps_skipped_no_slot == before.taps_skipped_no_slot + 1,
                "tap retry: the skipped tap was not taken at the next boundary once, with one skip counted");
        prefill(bound.sequence, kLongPrompt - cursor);
        decode(bound.sequence, true);
        require(program.finish(bound.sequence).status == runtime::ConsumeStatus::Consumed,
                "tap retry: the request did not finish");
        std::vector<TokenId> departed(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(cursor) + 32);
        departed.insert(departed.end(), 40, 201);
        const auto sources = program.hybrid_sources(request_of(departed), nullptr);
        const std::uint32_t resumed = sources.empty() ? 0U : sources.front().reused_tokens;
        std::cout << "tap retry: a prompt departing at " << cursor + 32 << " resumes from " << resumed << '\n';
        require(resumed == cursor, "tap retry: the retried tap does not serve a prompt departing past it");
        unpinned("tap retry");
    }
    // A reply the template will not echo (its first id is the prompt's prompt_end_tap_token: a reasoning closed at
    // once) leaves a tap at the prompt's end, taken as its first token commits; a next prompt departing there
    // resumes from it. The first token is learned from the same prompt with the cache off, then marked.
    // The control marks an id the reply does not start with: no tap, so the departing prompt resumes short of the end.
    for (const bool control : {false, true}) {
        const auto ids = prompt_ids(control ? 7000 : 6000, kPromptTokens);
        TokenId first  = 0;
        {
            const auto probe = request_of(ids, std::nullopt, false);
            const Bound bound = bind(probe, 0, nullptr, qwen::ExecutionUnitKind::Prefill);
            prefill(bound.sequence, kPromptTokens);
            first = program.sequences[0].ledger.at(kPromptTokens);
            require(program.abort(bound.sequence).status == runtime::ConsumeStatus::Consumed,
                    "the probe request could not abort");
        }
        const TokenId other = first == 198 ? 199 : 198;
        const auto marked   = request_of(ids, control ? other : first);
        const Bound bound   = bind(marked, 0, nullptr, qwen::ExecutionUnitKind::Prefill);
        prefill(bound.sequence, kPromptTokens);
        require(program.sequences[0].ledger.at(kPromptTokens) == first, "the marked request's first id changed");
        for (int round = 0; round < 2; ++round) { decode(bound.sequence, round == 1); }
        require(program.finish(bound.sequence).status == runtime::ConsumeStatus::Consumed,
                "the marked request did not finish");
        auto departed = ids;
        departed.push_back(other); // the echo's extra id where the reply began
        departed.insert(departed.end(), 40, 200);
        const auto next              = request_of(departed);
        const auto sources           = program.hybrid_sources(next, nullptr);
        const std::uint32_t resumed  = sources.empty() ? 0U : sources.front().reused_tokens;
        std::cout << "prompt-end tap" << (control ? " (control)" : "") << ": first id " << first
                  << ", the departing prompt resumes from " << resumed << " of " << kPromptTokens << '\n';
        require(control ? resumed < kPromptTokens : resumed == kPromptTokens,
                control ? "a reply the template echoes left a snapshot at its prompt's end"
                        : "a reply the template does not echo left no snapshot at its prompt's end");
        unpinned("prompt-end tap");
    }
    // A resume binding cancelled while its restore is staged, then the same resume again; then shutdown
    // cleanup with a resume binding staged.
    {
        const auto base = request(3000);
        Bound bound     = bind(base, 0, nullptr, qwen::ExecutionUnitKind::Prefill);
        prefill(bound.sequence, kPauseFrontier);
        require(program.start_pause(bound.sequence, nullptr), "cancel fixture: pause refused");
        auto paused = settle();
        require(paused.paused.has_value(), "cancel fixture: pause lost its request");
        const std::uint32_t staged = start(base, 1, &*paused.paused, qwen::ExecutionUnitKind::Prefill);
        const std::atomic<bool> cancelled{true};
        const auto aborted = settle(runtime::CancellationFlagView{&cancelled});
        require(aborted.complete && !aborted.published && !program.has_context_transaction(),
                "a cancelled resume binding published or stayed open");
        unpinned("resume binding cancelled");
        const Bound again = bind(base, 1, &*paused.paused, qwen::ExecutionUnitKind::Prefill);
        require(again.reused == staged, "the paused request lost its source to a cancelled binding");
        (void)replay(again.sequence);
        require(program.abort(again.sequence).status == runtime::ConsumeStatus::Consumed,
                "the resumed request could not abort");
        unpinned("resumed after a cancelled binding");

        (void)start(base, 0, &*paused.paused, qwen::ExecutionUnitKind::Prefill);
        program.shutdown_cleanup();
        require(!program.has_context_transaction(), "shutdown cleanup left a binding open");
        unpinned("shutdown cleanup during a staged resume");
        paused.paused.reset();
    }
}

// A paused generation resumed from a tap whose blocks and image went to the Host tier, with the restore's copies
// held 200 ms (testing::set_restore_delay), against the same request with the copies not held:
// - the Replay passes wait for each layer's copies, so both runs generate the same tokens (2026-11-15: only a
//   prefill pass waited; a Replay pass read restored KV pages and recurrent state before they landed);
// - a reclaim that cannot evict returns while that restore is still landing, having waited for no transfer (it
//   drained every transfer, another lane's restore included);
// - a lane's release writes to the Host tier only the blocks a snapshot can resume through;
// - the prefill unit that publishes a tap reports it (PrefillProgress::snapshot_published), and no other does;
// - emptying the cache drops every entry and the tool-call holds that named them.
void host_restore_lifecycle(DeviceContext& device, const qwen::execution::Parameters& parameters,
                            EngineOptions options) {
    constexpr std::uint32_t kHostPrompt = 2048;
    constexpr std::uint32_t kTap        = kHostPrompt - kChunk; // the prompt-tail tap: the final chunk's start
    const qwen::Frontend frontend =
        qwen::make_frontend(parameters.model.resources(), {.vision_enabled = false, .max_context = kLongContext});
    options.max_context                                = kLongContext;
    options.kv_capacity                                = KvCapacityPolicy::explicit_capacity(2 * kLongContext);
    options.context_cache.enabled                      = true;
    options.context_cache.host_capacity_bytes          = 2ULL << 30;
    options.context_cache.hybrid.device_snapshot_slots = 4;
    options.context_cache.hybrid.max_new_taps          = 2;
    options.context_cache.hybrid.tap_ladder_tokens     = 4096;
    options.context_cache.hybrid.tap_min_gap_tokens    = 1024;
    auto planner = qwen::detail::make_sequence_planner_impl(parameters, device, options);
    auto plan    = qwen::detail::finalize_sequence_plan_impl(std::move(planner), 3 * kLongContext / 64U);
    qwen::detail::ProgramImpl program(parameters, *plan, device, {});
    require(program.hybrid_ != nullptr && program.hybrid_->host_tier(), "the Host-tier fixture built no Host tier");

    const auto settle = [&] {
        for (unsigned step = 0; step < 8; ++step) {
            device.synchronize();
            auto progress = program.poll_context({});
            if (progress.complete) { return progress; }
        }
        throw std::runtime_error("Host-tier fixture transaction did not complete");
    };
    const auto reserve = [&](qwen::SequenceHandle sequence, qwen::ExecutionUnitKind kind, std::uint32_t tokens = 0) {
        const std::array<qwen::ExecutionUnit, 1> units{{{sequence, kind, tokens}}};
        require(static_cast<bool>(program.reserve_units(units)), "the Host-tier fixture could not reserve a unit");
    };
    const auto decode = [&](qwen::SequenceHandle sequence, bool terminal) {
        reserve(sequence, qwen::ExecutionUnitKind::Decode, 1);
        const std::array<qwen::SequenceHandle, 1> members{sequence};
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto pending = program.decode(members, budgets, nullptr, nullptr);
        const std::array<runtime::CommitDecision, 1> decision{{{.accepted_tokens = 1, .terminal = terminal}}};
        (void)program.commit(std::move(pending), decision, runtime::CommitObservation::AllRows, nullptr);
    };
    struct RestoreDelay {
        explicit RestoreDelay(std::uint32_t microseconds) { qwen::testing::set_restore_delay(microseconds); }
        ~RestoreDelay() { qwen::testing::set_restore_delay(0); }
    };

    // One request from an empty cache: prefilled, three tokens generated, paused with no Device snapshot slot to
    // take (so the pause publishes nothing), its blocks evicted to the Host tier, resumed from the tap and replayed,
    // then six more tokens. Returns the tokens it generated.
    const auto run = [&](bool delayed) {
        const std::string label = delayed ? "held restore" : "restore";
        const auto at_start     = program.hybrid_stats();
        std::vector<TokenId> ids(kHostPrompt, 198);
        for (std::size_t at = 0; at < ids.size(); at += 5) { ids[at] = 4000 + static_cast<TokenId>(at % 89); }
        runtime::ResolvedExecutionOptions execution;
        execution.requested_output_tokens = 16;
        execution.allow_prefix_reuse      = true;
        const auto base = program.plan_request(
            qwen::PreparedPromptAccess::take(frontend.prepare_tokens(std::move(ids))), execution);

        auto sources = program.hybrid_sources(base, nullptr);
        require(!sources.empty() && sources.front().reused_tokens == 0, label + ": the empty cache quoted a source");
        require(static_cast<bool>(program.start_binding(base, {0}, sources.front(), nullptr,
                                                        qwen::ExecutionUnitKind::Prefill, 0U)),
                label + ": the request could not bind");
        const qwen::SequenceHandle original = *settle().sequence;
        // The sequence carries a rotary offset, as a prompt with a multi-row image before this text does: every
        // text pass past the prompt's start, and every Replay pass after the pause, must run with it (they ran at 0).
        constexpr std::int32_t kRopeDelta = -7;
        program.sequences[0].rope_delta   = kRopeDelta;
        qwen::testing::record_text_passes(true);
        bool reported = false;
        for (std::uint32_t processed = 0; processed < kHostPrompt;) {
            reserve(original, qwen::ExecutionUnitKind::Prefill);
            const std::uint64_t taps = program.hybrid_stats().taps_created;
            auto step                = program.advance_prefill(original, nullptr, nullptr);
            const bool published     = program.hybrid_stats().taps_created != taps;
            require(step.snapshot_published == published,
                    label + ": a prefill unit's snapshot_published disagrees with the taps it published");
            reported = reported || step.snapshot_published;
            processed += step.processed_prompt_tokens;
            if (step.complete) {
                const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
                (void)program.commit(std::move(*step.pending), accepted, runtime::CommitObservation::AllRows, nullptr);
            }
        }
        require(reported, label + ": no prefill unit reported the tap it published");
        for (int round = 0; round < 3; ++round) { decode(original, false); }

        // The tap's Host write lands, so holding every Device slot takes its slot and leaves its image on the Host.
        device.synchronize();
        program.hybrid_->drain();
        std::vector<std::uint32_t> held;
        while (const auto slot = program.hybrid_->index().acquire_device_slot()) { held.push_back(*slot); }
        const auto before_pause = program.hybrid_stats();
        require(program.start_pause(original, nullptr), label + ": pause refused");
        auto paused = settle();
        require(paused.paused.has_value(), label + ": pause lost its request");
        device.synchronize();
        program.hybrid_->drain();
        for (const std::uint32_t slot : held) { program.hybrid_->index().release_device_slot(slot); }
        // Only the blocks at or above the tap can serve a resume; the ones past it would be recomputed. They went to
        // the Host tier as the tap published (continuous write-through), so the release has nothing left to copy.
        const std::uint64_t at_tap     = before_pause.host_block_writes - at_start.host_block_writes;
        const std::uint64_t at_release = program.hybrid_stats().host_block_writes - before_pause.host_block_writes;
        require(at_tap == kTap / 64U, label + ": the tap wrote " + std::to_string(at_tap) +
                                          " blocks to the Host tier, not the " + std::to_string(kTap / 64U) +
                                          " a snapshot can resume through");
        require(at_release == 0, label + ": the release wrote " + std::to_string(at_release) +
                                     " blocks no snapshot reaches");
        (void)program.hybrid_->index().evict_backed_device_blocks(kLongContext);

        const RestoreDelay delay(delayed ? 200'000U : 0U);
        const auto before_resume = program.hybrid_stats();
        sources                  = program.hybrid_sources(base, &*paused.paused);
        require(!sources.empty() && sources.front().reused_tokens == kTap,
                label + ": the paused request was not quoted its tap");
        require(static_cast<bool>(program.start_binding(base, {1}, sources.front(), &*paused.paused,
                                                        qwen::ExecutionUnitKind::Decode, 1U)),
                label + ": the resume could not bind");
        auto bound = settle();
        require(bound.published && bound.sequence && bound.replaying, label + ": the resume did not enter replay");
        paused.paused.reset();
        const qwen::SequenceHandle resumed = *bound.sequence;
        require(program.hybrid_stats().host_block_restores - before_resume.host_block_restores == kTap / 64U,
                label + ": the resume did not restore its blocks from the Host tier");
        if (delayed) {
            // Nothing a wait could free: the only transfer in flight lands blocks the resumed lane maps.
            const std::uint64_t waits = qwen::testing::transfer_waits();
            runtime::ContextResourceUsage shortage;
            shortage.main_kv_pages = 4U * kLongContext;
            (void)program.hybrid_reclaim(shortage);
            require(qwen::testing::transfer_waits() == waits && program.hybrid_->transfers_pending(),
                    "a reclaim that could evict nothing waited for another lane's restore");
        }
        bool complete = false;
        for (unsigned step = 0; !complete && step < kHostPrompt / kChunk + 4; ++step) {
            reserve(resumed, qwen::ExecutionUnitKind::Replay);
            complete = program.advance_replay(resumed, nullptr).complete;
        }
        require(complete, label + ": the replay did not complete");
        const std::vector<qwen::testing::TextPass> passes = qwen::testing::text_passes();
        qwen::testing::record_text_passes(false);
        std::size_t replayed_passes = 0;
        for (const qwen::testing::TextPass& pass : passes) {
            require(pass.rope_delta == kRopeDelta,
                    label + ": a text pass at " + std::to_string(pass.base) + " ran with rotary offset " +
                        std::to_string(pass.rope_delta) + ", not the sequence's " + std::to_string(kRopeDelta));
            replayed_passes += pass.base >= kTap ? 1U : 0U;
        }
        // The prompt's last pass and the Replay's passes all begin at or past the tap.
        require(replayed_passes >= 2, label + ": no Replay pass was recorded");
        for (int round = 0; round < 6; ++round) { decode(resumed, round == 5); }
        const std::vector<TokenId>& ledger = program.sequences[1].ledger;
        std::vector<TokenId> generated(ledger.begin() + kHostPrompt, ledger.end());
        require(program.finish(resumed).status == runtime::ConsumeStatus::Consumed, label + ": the request did not finish");
        return generated;
    };

    const std::vector<TokenId> plain = run(false);
    // Emptying the cache between the runs: no entry and no tool-call hold survives it.
    program.hybrid_hold_continuation(std::chrono::steady_clock::now() + std::chrono::seconds(30));
    require(!program.hybrid_tool_holds_.empty(), "the finished request left nothing for a tool-call hold");
    program.hybrid_reset_cache();
    require(program.hybrid_tool_holds_.empty() && program.hybrid_->index().stats().nodes == 0 &&
                program.hybrid_->index().stats().snapshots == 0,
            "emptying the prefix cache left entries or tool-call holds behind");
    const std::vector<TokenId> held = run(true);
    require(plain.size() == 10 && held == plain,
            "a resume whose Host restore was held generated other tokens than one whose restore had landed");
}

SpeculativeBackend selected_backend(std::string_view name) {
    if (name == "none") { return SpeculativeBackend::None; }
    if (name == "mtp") { return SpeculativeBackend::Mtp; }
    if (name == "dflash") { return SpeculativeBackend::DFlash; }
    if (name == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("backend must be none, mtp, dflash or dflash2");
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "SKIP: set INFERNIX_TEST_ARTIFACT to an explicit .ninfer artifact\n";
        return 77;
    }
    try {
        require(argc <= 2,
                "usage: infernix_qwen3_5_native_transactions_test [none|mtp|dflash|dflash2]");
        const auto backend = selected_backend(argc == 2 ? argv[1] : "none");
        DeviceContext device;
        // The load options the Program checks against (an MTP Program's include its expert pool).
        const EngineOptions options = program_options(backend);
        auto model = qwen::load_model(artifact, models::load_options(options, models::Architecture::Qwen3_5), device);
        qwen::execution::Parameters parameters(*model);
        auto frontend = qwen::make_frontend(model->resources(),
                                            {.vision_enabled = false, .max_context = kCapacity});
        auto planner = qwen::make_sequence_planner(parameters, device, options);
        auto plan    = std::move(planner).finalize(16);
        auto program = qwen::create_program(parameters, std::move(plan), device, {});
        {
            Fixture fixture(device, *program, frontend);
            fixture.replay_pause_and_resume();
            fixture.pause_before_first_prefill_chunk();
            fixture.grammar_row_failure();
        }
        program.reset();
        std::cout << "pause, replay and row failure: OK\n";
        replay_sampling_counts(device, parameters, frontend, options);
        std::cout << "replay sampling counts: OK\n";
        cached_pause_lifecycle(device, parameters, options);
        std::cout << "cached pause lifecycle: OK\n";
        host_restore_lifecycle(device, parameters, options);
        std::cout << "OK native pause, a pause before the first chunk, replay recovery, row failure, the cached "
                     "pause lifecycle and the Host-restored resume\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL native transactions: " << error.what() << '\n';
        return 1;
    }
}
