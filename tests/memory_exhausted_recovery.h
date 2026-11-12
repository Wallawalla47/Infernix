#pragma once

// The Engine's recovery from a unit that ran out of device memory, on a real artifact (fix sweep FS-01); shared
// by the Qwen3.5 and Qwen3.8-Flash-Next real tests. runtime::testing::request_memory_fault makes the N-th
// CUDA_CHECK of the Engine worker's work after the request throw DeviceMemoryExhausted, as a real shortage does
// at a kernel's first (lazily loaded) launch or at a stream synchronization the driver could not page in for.
// Over a sweep of N the failure lands in binding, prefill, decode and commit work:
//
//   failed   a request whose work reached the N-th check fails with the retryable out-of-memory error; one
//            whose work ended first (or whose failing call was cleaned up in place) returns its reference.
//   serving  a fresh request after each one returns its reference tokens.
//   cache    a request that extends the swept one, with prefix reuse, returns the tokens it returns cold: the
//            prefix cache kept nothing the failed unit half computed.
//   decode   a short prompt's request with a longer output, swept past its prefill into the decode rounds (their
//            graph launches, drafting, first-token copies and commits): the same outcomes as above.
//   queued   a request queued behind a failing one is admitted after the recovery and returns its reference.
//   bound    eight failures in a row with no successful unit between them stop the Engine (docs/serving.md,
//            "How failures end").
//
// Exact route (the caller's options turn fast paths off), so prefix reuse may not change any token.

#include "core/device.h"
#include "runtime/contract/execution.h"
#include "infernix/engine.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace infernix::test::memory_exhausted {

inline std::vector<TokenId> synthetic_tokens(std::size_t count, std::uint32_t seed) {
    std::vector<TokenId> tokens;
    tokens.reserve(count);
    std::uint32_t state = seed * 2654435761U + 1U;
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 1664525U + 1013904223U;
        tokens.push_back(static_cast<TokenId>(1000U + (state >> 8U) % 30000U));
    }
    return tokens;
}

inline RequestOptions greedy(std::uint32_t outputs, bool reuse) {
    RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

inline int check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    return ok ? 0 : 1;
}

struct Outcome {
    std::vector<TokenId> tokens;
    std::string error;
};

inline Outcome run(Engine& engine, const std::vector<TokenId>& prompt, std::uint32_t outputs, bool reuse) {
    Outcome outcome;
    try {
        outcome.tokens = engine.generate(engine.prepare_tokens(prompt), greedy(outputs, reuse)).generated_token_ids;
    } catch (const std::exception& error) { outcome.error = error.what(); }
    return outcome;
}

inline bool out_of_memory(const Outcome& outcome) {
    return outcome.error.find("out of device memory") != std::string::npos;
}

// A request on its own thread.
class Async {
public:
    Async(Engine& engine, std::vector<TokenId> prompt, std::uint32_t outputs, bool reuse)
        : thread_([this, &engine, prompt = std::move(prompt), outputs, reuse] {
              Outcome outcome = run(engine, prompt, outputs, reuse);
              std::lock_guard lock(mutex_);
              outcome_ = std::move(outcome);
          }) {}
    ~Async() {
        if (thread_.joinable()) { thread_.join(); }
    }
    Outcome get() {
        thread_.join();
        std::lock_guard lock(mutex_);
        return *outcome_;
    }

private:
    std::mutex mutex_;
    std::optional<Outcome> outcome_;
    std::thread thread_;
};

// Runs every step on `engine` (started on the exact route, one lane, at least two pending requests). The
// prompts are synthetic ids; `prompt_tokens` and `outputs` size them.
inline int run_all(Engine& engine, std::uint32_t prompt_tokens, std::uint32_t outputs) {
    namespace testing = runtime::testing;
    int failures = 0;
    const std::vector<TokenId> fresh = synthetic_tokens(prompt_tokens, 7);
    const std::vector<TokenId> queued = synthetic_tokens(prompt_tokens, 8);
    const Outcome fresh_reference = run(engine, fresh, outputs, false);
    const Outcome queued_reference = run(engine, queued, outputs, false);
    if (!fresh_reference.error.empty() || !queued_reference.error.empty()) {
        std::printf("  FAIL a reference request failed: %s%s\n", fresh_reference.error.c_str(),
                    queued_reference.error.c_str());
        return 1;
    }

    // Swept counts: the failure lands from the first binding work to deep in a decode round's commit.
    const std::vector<std::uint64_t> sweep{1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181};
    struct Swept {
        std::vector<TokenId> prompt, extended;
        Outcome reference, extended_reference;
    };
    std::vector<Swept> swept(sweep.size());
    for (std::size_t i = 0; i < sweep.size(); ++i) {
        swept[i].prompt   = synthetic_tokens(prompt_tokens, 100U + static_cast<std::uint32_t>(i));
        swept[i].extended = swept[i].prompt;
        const std::vector<TokenId> tail = synthetic_tokens(prompt_tokens / 8U, 200U + static_cast<std::uint32_t>(i));
        swept[i].extended.insert(swept[i].extended.end(), tail.begin(), tail.end());
        // Cold: a request without prefix reuse inserts nothing into the cache either.
        swept[i].reference          = run(engine, swept[i].prompt, outputs, false);
        swept[i].extended_reference = run(engine, swept[i].extended, outputs, false);
    }

    std::printf("== sweep\n");
    const std::uint64_t recoveries_before = engine.runtime_stats().device_memory_recoveries;
    std::uint32_t failed = 0;
    // The first swept count that failed its request: the bound step's. (A shortage while a request binds can
    // leave it queued, so it binds again and completes; Flash-Next gives a failed binding back that way.)
    std::uint64_t failing_count = 0;
    for (std::size_t i = 0; i < sweep.size(); ++i) {
        const std::uint64_t injected = infernix::testing::memory_exhausted_injections();
        testing::request_memory_fault(sweep[i]);
        const Outcome outcome = run(engine, swept[i].prompt, outputs, true);
        testing::request_memory_fault(0); // the next worker iteration disarms what did not fire
        const bool fired = infernix::testing::memory_exhausted_injections() > injected;
        const std::string label = "N = " + std::to_string(sweep[i]) + (fired ? " (fired)" : " (not reached)");
        if (outcome.error.empty()) {
            failures += check(outcome.tokens == swept[i].reference.tokens, label + ": completed with its reference tokens");
        } else {
            ++failed;
            if (failing_count == 0 && out_of_memory(outcome)) { failing_count = sweep[i]; }
            failures += check(out_of_memory(outcome), label + ": failed with the out-of-memory error (got '" +
                                                          outcome.error + "')");
        }
        const Outcome after = run(engine, fresh, outputs, false);
        failures += check(after.error.empty() && after.tokens == fresh_reference.tokens,
                          label + ": a fresh request after it returned its reference" +
                              (after.error.empty() ? std::string() : " (error '" + after.error + "')"));
        const Outcome extended = run(engine, swept[i].extended, outputs, true);
        failures += check(extended.error.empty() && extended.tokens == swept[i].extended_reference.tokens,
                          label + ": its extension, with prefix reuse, returned its cold tokens" +
                              (extended.error.empty() ? std::string() : " (error '" + extended.error + "')"));
    }
    const std::uint64_t recoveries = engine.runtime_stats().device_memory_recoveries - recoveries_before;
    std::printf("   %u of %zu swept requests failed; %llu out-of-memory recoveries\n", failed, sweep.size(),
                static_cast<unsigned long long>(recoveries));
    failures += check(failed >= 6 && failing_count != 0, "at least six swept requests failed with the injected shortage");
    failures += check(recoveries >= failed, "RuntimeStats counted a recovery for every failed request");

    std::printf("== decode sweep\n");
    {
        // A short prompt (one small prefill call) and a longer output: counts past the prompt's checks land in
        // decode rounds, their drafting, their graphs' first capture and their commits. Each request is cold, so
        // a count lands at the same place every time; the sweep ends at the first count the request no longer
        // reaches.
        const std::vector<TokenId> short_prompt = synthetic_tokens(48, 400);
        const std::uint32_t long_output         = outputs * 3U;
        const Outcome short_reference           = run(engine, short_prompt, long_output, false);
        std::uint32_t decode_failed = 0;
        // One count: false once the request no longer reaches it.
        const auto inject = [&](std::uint64_t n) {
            const std::uint64_t injected = infernix::testing::memory_exhausted_injections();
            testing::request_memory_fault(n);
            const Outcome outcome = run(engine, short_prompt, long_output, false);
            testing::request_memory_fault(0);
            const bool fired      = infernix::testing::memory_exhausted_injections() > injected;
            const std::string label = "N = " + std::to_string(n);
            if (!fired && outcome.error.empty()) {
                failures += check(outcome.tokens == short_reference.tokens, label + " (not reached): completed with its reference");
                return false;
            }
            if (outcome.error.empty()) {
                failures += check(outcome.tokens == short_reference.tokens, label + ": completed with its reference");
            } else {
                ++decode_failed;
                failures += check(out_of_memory(outcome), label + ": failed with the out-of-memory error at '" +
                                                              outcome.error.substr(0, 150) + "'");
            }
            const Outcome after = run(engine, fresh, outputs, false);
            failures += check(after.error.empty() && after.tokens == fresh_reference.tokens,
                              label + ": a fresh request after it returned its reference");
            return true;
        };
        std::uint64_t reached = 0, end = 0;
        for (std::uint64_t n = 800; n <= 200000; n = n * 11U / 10U) {
            if (!inject(n)) {
                end = n;
                break;
            }
            reached = n;
        }
        // The decode rounds hold few checks (their graphs launch whole): twelve counts between the last one
        // reached and the end land in them.
        if (end > reached + 1) {
            const std::uint64_t step = std::max<std::uint64_t>(1, (end - reached) / 13U);
            for (std::uint64_t n = reached + step; n < end; n += step) {
                if (!inject(n)) { break; }
            }
        }
        failures += check(decode_failed >= 3, "at least three short-prompt requests failed with the shortage (" +
                                                  std::to_string(decode_failed) + ")");
    }

    std::printf("== queued\n");
    {
        testing::request_memory_fault(377);
        Async held(engine, swept[0].prompt, outputs, false);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        Async waiting(engine, queued, outputs, false);
        const Outcome first  = held.get();
        const Outcome second = waiting.get();
        testing::request_memory_fault(0);
        failures += check(first.error.empty() ? first.tokens == swept[0].reference.tokens : out_of_memory(first),
                          "the held request failed with the shortage or returned its reference");
        failures += check(second.error.empty() && second.tokens == queued_reference.tokens,
                          "the request queued behind it completed with its reference" +
                              (second.error.empty() ? std::string() : " (error '" + second.error + "')"));
    }

    std::printf("== bound\n");
    {
        const Outcome before = run(engine, fresh, outputs, false); // a successful unit: the count starts at 0
        failures += check(before.error.empty(), "a request before the bound's failures completed");
        std::uint32_t refused = 0;
        std::printf("   (injected at check %llu, the sweep's first failing count)\n",
                    static_cast<unsigned long long>(failing_count));
        for (int i = 0; i < 8; ++i) {
            testing::request_memory_fault(failing_count != 0 ? failing_count : 1);
            const Outcome outcome = run(engine, synthetic_tokens(prompt_tokens, 300U + static_cast<std::uint32_t>(i)),
                                        outputs, false);
            refused += out_of_memory(outcome) ? 1U : 0U;
        }
        testing::request_memory_fault(0);
        failures += check(refused == 8, "eight requests in a row failed with the shortage (" + std::to_string(refused) + ")");
        const Outcome stopped = run(engine, fresh, outputs, false);
        failures += check(!stopped.error.empty(), "the eighth failure in a row stopped the Engine: a request after it "
                                                  "failed ('" + stopped.error + "')");
    }
    return failures;
}

} // namespace infernix::test::memory_exhausted
