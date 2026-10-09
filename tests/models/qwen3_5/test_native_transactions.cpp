#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/program.h"
#include "models/qwen3_5/program/program_impl.h"

#include <algorithm>
#include <array>
#include <atomic>
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
        const auto sources = program_.hybrid_sources(base, UINT32_MAX);
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
        require(pending.constraint_failed(0) && !pending.constraint_failed(1),
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
        const auto sources = program.hybrid_sources(base, UINT32_MAX);
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
        models::LoadOptions selected;
        selected.speculative = backend;
        auto model           = qwen::load_model(artifact, selected, device);
        qwen::execution::Parameters parameters(*model);
        auto frontend = qwen::make_frontend(model->resources(),
                                            {.vision_enabled = false, .max_context = kCapacity});
        const EngineOptions options = program_options(backend);
        auto planner = qwen::make_sequence_planner(parameters, device, options);
        auto plan    = std::move(planner).finalize(16);
        auto program = qwen::create_program(parameters, std::move(plan), device, {});
        {
            Fixture fixture(device, *program, frontend);
            fixture.replay_pause_and_resume();
            fixture.grammar_row_failure();
        }
        program.reset();
        replay_sampling_counts(device, parameters, frontend, options);
        std::cout << "OK native pause, replay recovery and row failure\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL native transactions: " << error.what() << '\n';
        return 1;
    }
}
