// The Engine's no-progress watch on the real artifact (candidates YJ-02, ZK-02): the warning and the heal had
// never executed, because no test could stall the Engine. Test hooks (runtime::testing::hold_units,
// set_no_progress_bounds) stall it and shorten the watch's bounds to 0.5 s and 2 s:
//
//   heal     A request is admitted and bound, then no unit runs. The watch warns, then heals: the request fails
//            with the documented error, RuntimeStats counts one recovery, and a request queued behind it is
//            admitted again and, once units run, returns the same tokens as before the stall.
//   after    With the hooks cleared the Engine serves a fresh request as before (no stale stall interval).
//
// Exact route (fast paths off). Skips (77) unless INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact
// (INFERNIX_QWEN4_NGRAM: its volume).
//
//   INFERNIX_QWEN4_ARTIFACT=out.ninfer infernix_qwen4_exp_no_progress_real_test

#include "runtime/contract/execution.h"
#include "engine_host.h"
#include "infernix/engine.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

std::vector<infernix::TokenId> synthetic_tokens(std::size_t count, std::uint32_t seed) {
    std::vector<infernix::TokenId> tokens;
    tokens.reserve(count);
    std::uint32_t state = seed * 2654435761U + 1U;
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 1664525U + 1013904223U;
        tokens.push_back(static_cast<infernix::TokenId>(1000U + (state >> 8U) % 30000U));
    }
    return tokens;
}

infernix::RequestOptions greedy(std::uint32_t outputs) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return options;
}

int check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    return ok ? 0 : 1;
}

struct Outcome {
    std::vector<infernix::TokenId> tokens;
    std::string error;
};

// Runs a request on its own thread; `done` once it returned or threw.
class Async {
public:
    Async(infernix::Engine& engine, std::vector<infernix::TokenId> prompt, std::uint32_t outputs)
        : thread_([this, &engine, prompt = std::move(prompt), outputs] {
              Outcome outcome;
              try {
                  outcome.tokens = engine.generate(engine.prepare_tokens(prompt), greedy(outputs)).generated_token_ids;
              } catch (const std::exception& error) { outcome.error = error.what(); }
              std::lock_guard lock(mutex_);
              outcome_ = std::move(outcome);
          }) {}
    ~Async() {
        if (thread_.joinable()) { thread_.join(); }
    }
    // Waits up to `limit` for the outcome.
    std::optional<Outcome> wait(std::chrono::milliseconds limit) {
        const auto until = std::chrono::steady_clock::now() + limit;
        for (;;) {
            {
                std::lock_guard lock(mutex_);
                if (outcome_) { return outcome_; }
            }
            if (std::chrono::steady_clock::now() >= until) { return std::nullopt; }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

private:
    std::mutex mutex_;
    std::optional<Outcome> outcome_;
    std::thread thread_;
};

} // namespace

int main() {
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::printf("SKIP: INFERNIX_QWEN4_ARTIFACT is not set\n");
        return 77;
    }
    const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM");
    namespace testing = infernix::runtime::testing;
    int failures = 0;
    try {
        infernix::EngineOptions options;
        options.artifact_path = artifact;
        if (ngram != nullptr && *ngram != '\0') { options.ngram_volume_path = ngram; }
        options.max_context          = 4096;
        options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(4608);
        options.prefill_chunk        = 512;
        options.kv_cache             = infernix::KvCacheStorage::Int8Group64;
        options.max_concurrency      = 1;
        options.max_pending_requests = 2;
        options.fast_paths           = false;
        std::atomic<int> warnings{0}, heal_lines{0};
        options.diagnostic_observer.callback = [&](const infernix::Diagnostic& diagnostic) {
            if (diagnostic.message.find("no progress for") != std::string::npos) { ++warnings; }
            if (diagnostic.message.find("made progress again") != std::string::npos) { ++heal_lines; }
        };
        infernix::Engine engine(options, infernix::test::engine_host());
        const std::vector<infernix::TokenId> held_prompt = synthetic_tokens(700, 41);
        const std::vector<infernix::TokenId> queued      = synthetic_tokens(600, 42);
        const std::vector<infernix::TokenId> reference =
            engine.generate(engine.prepare_tokens(queued), greedy(16)).generated_token_ids;
        const std::uint64_t recoveries_before = engine.runtime_stats().no_progress_recoveries;

        std::printf("== heal\n");
        testing::set_no_progress_bounds(500, 2000);
        testing::hold_units(true);
        Async held(engine, held_prompt, 16);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        Async waiting(engine, queued, 16);
        const std::optional<Outcome> first = held.wait(std::chrono::seconds(30));
        testing::hold_units(false); // units run again: the queued request may now finish
        failures += check(first.has_value(), "the held request returned within 30 s (the heal ran)");
        if (first) {
            failures += check(first->error.find("made no progress") != std::string::npos,
                              "the held request failed with the no-progress error (got '" + first->error + "')");
        }
        const std::optional<Outcome> second = waiting.wait(std::chrono::seconds(60));
        failures += check(second.has_value() && second->error.empty(),
                          "the request queued behind the stall was admitted again and completed" +
                              (second && !second->error.empty() ? " (error '" + second->error + "')" : std::string()));
        if (second && second->error.empty()) {
            failures += check(second->tokens == reference, "it returned the tokens it returned before the stall");
        }
        failures += check(engine.runtime_stats().no_progress_recoveries == recoveries_before + 1,
                          "RuntimeStats counted one no-progress recovery");
        failures += check(warnings.load() >= 1, "the watch published its no-progress warning");

        std::printf("== after\n");
        testing::set_no_progress_bounds(0, 0);
        const std::vector<infernix::TokenId> again =
            engine.generate(engine.prepare_tokens(queued), greedy(16)).generated_token_ids;
        failures += check(again == reference, "a fresh request after the heal returns the same tokens");
        failures += check(engine.runtime_stats().no_progress_recoveries == recoveries_before + 1,
                          "no further recovery after the hooks were cleared");
        std::printf("   (progress-again lines: %d)\n", heal_lines.load());
    } catch (const std::exception& error) {
        testing::hold_units(false);
        testing::set_no_progress_bounds(0, 0);
        std::printf("  FAIL exception: %s\n", error.what());
        return 1;
    }
    std::printf(failures == 0 ? "all checks passed\n" : "%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
