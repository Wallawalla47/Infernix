// The fork route of Qwen3.8-Flash-Next's decode and verification expert calls (ops::moe_experts: the misses staged and
// computed on a side stream beside the resident experts), through the public Engine on the real artifact.
// A call takes it only when every job fits the staging slots, so a Program whose staging rule misses a job (a shared
// expert bank's, MoeConfig::shared_bank) loses it at its widest width with the same outputs: NVIDIA NVFP4-QAD's
// eight-column verification calls did with top_k x max_width (80) slots, MTP decode 14-16 % slower (design §6.1.3).
// Per drafter (arguments mtp, dflash2; seven drafts, so calls of up to eight columns), an Engine on the default
// expert layout and route with two lanes captures its graph set at startup and runs a code prompt alone and two
// together: at every width every call offered the fork (ops::testing::moe_forked_calls / moe_lost_forks) must take
// it, and an eight-column call must have. INFERNIX_TMP_STAGING_SLOTS=80 on NVFP4-QAD makes it fail (the eight-column
// calls lose the fork). The rule holds where every routed expert is in RAM or VRAM (the exclusive layout's tier): a
// start short of RAM reads the rest from the artifact and keeps 64 slots by design, so a drafter whose start did is
// skipped, reported (on 2026-11-18 a start right after another process freed its RAM did, and failed at 7-8 columns).
// Skips (77) without INFERNIX_QWEN4_ARTIFACT or when every drafter is skipped; dflash2 skips without
// INFERNIX_QWEN4_DFLASH2.
//
//   INFERNIX_QWEN4_ARTIFACT=out.infernix [INFERNIX_QWEN4_NGRAM=out.ngram] [INFERNIX_QWEN4_DFLASH2=drafter.infernix]
//   infernix_qwen4_exp_fork_route_real_test mtp dflash2

#include "models/qwen4_exp/program/prefix/test_taps.h"
#include "engine_host.h"
#include "infernix/engine.h"
#include "infernix/ops/offloaded_sparse_moe.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct Setup {
    const char* artifact = nullptr;
    const char* ngram    = nullptr;
    const char* drafter  = nullptr;
};

// A code prompt whose answer repeats its own structure (long accepted drafts) and a prose prompt.
const char* const kPrompts[] = {
    "Write a C++ function `int classify(int code)` that returns 1 for each of the codes 100, 101, 102, 103, 104, 105, "
    "106, 107, 108 and 109 and 0 otherwise, as a switch statement with one case line per code.",
    "In two short paragraphs, explain why lighthouses were built on headlands.",
};
constexpr std::uint32_t kLanes     = 2;
constexpr std::uint32_t kDrafts    = 7; // Flash-Next's most: calls of up to eight columns
constexpr std::int32_t kWidest     = 8; // the fork route's widest call (and kDrafts + 1)
constexpr std::uint32_t kMaxTokens = 96;

infernix::EngineOptions engine_options(const Setup& setup, std::string_view drafter) {
    infernix::EngineOptions options;
    options.artifact_path = setup.artifact;
    if (setup.ngram != nullptr) { options.ngram_volume_path = setup.ngram; }
    options.max_context                   = 4096;
    options.kv_capacity                   = infernix::KvCapacityPolicy::explicit_capacity(4096);
    options.kv_cache                      = infernix::KvCacheStorage::Int8Group64;
    options.max_concurrency               = kLanes;
    options.max_pending_requests          = kLanes;
    options.context_cache.enabled         = false;
    options.speculative.draft_tokens      = kDrafts;
    options.speculative.draft_tokens_auto = true;
    if (drafter == "mtp") {
        options.speculative.backend = infernix::SpeculativeBackend::Mtp;
    } else if (drafter == "dflash2") {
        options.speculative.backend     = infernix::SpeculativeBackend::DFlash2;
        options.speculative.draft_model = setup.drafter;
    } else {
        throw std::invalid_argument("the drafter must be mtp or dflash2");
    }
    return options;
}

infernix::PromptInput prompt_input(const char* text) {
    infernix::PromptInput input;
    infernix::ChatMessage message;
    message.role = infernix::ChatRole::User;
    message.parts.push_back({.kind = infernix::MessagePartKind::Text, .text = text});
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

infernix::RequestOptions request_options() {
    infernix::RequestOptions request;
    request.execution.requested_output_tokens = kMaxTokens;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    return request;
}

// A drafter's checks failed, or the case was skipped (its start did not hold every expert in RAM or VRAM).
struct Outcome {
    int failures = 0;
    bool skipped = false;
};

Outcome run(const Setup& setup, std::string_view drafter) {
    namespace testing      = infernix::ops::testing;
    namespace program_taps = infernix::models::qwen4_exp::testing;
    const std::string name(drafter);
    int failures     = 0;
    const auto check = [&](bool condition, const std::string& what) {
        if (!condition) {
            std::fprintf(stderr, "FAIL %s: %s\n", name.c_str(), what.c_str());
            ++failures;
        }
    };
    // Per column count 1..kWidest: calls that forked and that lost the fork, at startup (the graph set's capture) and
    // in all (the runs' eager calls and any graph captured at its first use).
    struct Counts {
        std::uint64_t forked[kWidest + 1] = {}, lost[kWidest + 1] = {};
    };
    const auto read = [] {
        Counts c;
        for (std::int32_t w = 1; w <= kWidest; ++w) {
            c.forked[w] = testing::moe_forked_calls(w);
            c.lost[w]   = testing::moe_lost_forks(w);
        }
        return c;
    };
    const Counts before = read();
    Counts startup;
    program_taps::note_staging(0, false); // cleared: the Engine's Program notes its own
    {
        infernix::Engine engine(engine_options(setup, drafter), infernix::test::engine_host());
        startup                    = read();
        const std::uint32_t drafts = engine.options().speculative.draft_tokens;
        check(drafts == kDrafts, "the Engine drafts at most " + std::to_string(drafts) + " tokens, not " +
                                     std::to_string(kDrafts) + ": the widest calls are not checked");
        // The rule's case: every routed expert in RAM or VRAM (the Program notes its slots and tier at its start).
        const std::int32_t slots = program_taps::staging_slots();
        if (slots == 0) {
            check(false, "the Program noted no staging slots");
            return {.failures = failures, .skipped = false};
        }
        if (!program_taps::staging_full_tier()) {
            std::printf("skip %s: the start reads part of the experts from the artifact (short of RAM), so it keeps %d "
                        "staging slots by design; the rule is not checked\n",
                        name.c_str(), slots);
            return {.failures = failures, .skipped = true};
        }
        std::printf("%s: %d staging slots, every routed expert in RAM or VRAM\n", name.c_str(), slots);
        engine.generate(engine.prepare(prompt_input(kPrompts[0])), request_options());
        std::vector<infernix::GenerationHandle> handles;
        for (std::uint32_t i = 0; i < kLanes; ++i) {
            handles.push_back(engine.submit(engine.prepare(prompt_input(kPrompts[i % 2])), request_options()));
        }
        for (auto& handle : handles) { handle.wait(); }
    }
    const Counts after = read();
    std::printf("%s: calls offered the fork route, forked / lost (startup capture; all):\n", name.c_str());
    for (std::int32_t w = 1; w <= kWidest; ++w) {
        const std::uint64_t forked = after.forked[w] - before.forked[w], lost = after.lost[w] - before.lost[w];
        std::printf("  %d columns: %llu / %llu; %llu / %llu\n", w,
                    static_cast<unsigned long long>(startup.forked[w] - before.forked[w]),
                    static_cast<unsigned long long>(startup.lost[w] - before.lost[w]),
                    static_cast<unsigned long long>(forked), static_cast<unsigned long long>(lost));
        check(lost == 0, std::to_string(lost) + " expert calls of " + std::to_string(w) +
                             " columns lost the fork route: their jobs exceeded the staging slots");
    }
    check(after.forked[kWidest] > before.forked[kWidest],
          "no expert call of " + std::to_string(kWidest) + " columns took the fork route (the widest was not checked)");
    return {.failures = failures, .skipped = false};
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Setup setup;
    setup.artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    setup.ngram    = std::getenv("INFERNIX_QWEN4_NGRAM");
    setup.drafter  = std::getenv("INFERNIX_QWEN4_DFLASH2");
    if (setup.artifact == nullptr || *setup.artifact == '\0') {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT\n");
        return 77;
    }
    if (argc < 2) {
        // No case named: report it instead of passing with nothing checked.
        std::fprintf(stderr, "FAIL: usage: infernix_qwen4_exp_fork_route_real_test mtp|dflash2...\n");
        return 2;
    }
    try {
        int failures = 0;
        int cases    = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string_view drafter = argv[i];
            if (drafter == "dflash2" && (setup.drafter == nullptr || *setup.drafter == '\0')) {
                std::printf("skip dflash2: INFERNIX_QWEN4_DFLASH2 is not set\n");
                continue;
            }
            const Outcome outcome = run(setup, drafter);
            failures += outcome.failures;
            cases += outcome.skipped ? 0 : 1;
        }
        if (cases == 0 && failures == 0) {
            std::printf("SKIP: every named case was skipped\n");
            return 77;
        }
        std::printf(failures == 0 ? "fork route passed (%d cases)\n" : "FAIL: %d checks failed\n",
                    failures == 0 ? cases : failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
