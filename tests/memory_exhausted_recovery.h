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
//   release  a failure inside the Program's finish or abort (the request's release): the request completes with its
//            reference (or ends cancelled), the Engine recovers the device and serves on.
//   bound    eight failures in a row with no successful unit between them stop the Engine (docs/serving.md,
//            "How failures end").
//
// Exact route (the caller's options turn fast paths off), so prefix reuse may not change any token.

#include "core/device.h"
#include "runtime/contract/execution.h"
#include "infernix/engine.h"

#include <algorithm>
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

// Once the Engine is idle, every cached Device block of the prefix cache is evictable: nothing a failed release left
// still pins one. Polled up to 2 s, while blocks pinned only by Host writes in flight land.
inline bool nothing_pinned(Engine& engine, std::string& detail) {
    RuntimeStats stats;
    for (int i = 0; i < 200; ++i) {
        stats = engine.runtime_stats();
        if (stats.hybrid_evictable_blocks == stats.hybrid_cached_blocks) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    detail = std::to_string(stats.hybrid_evictable_blocks) + " of " + std::to_string(stats.hybrid_cached_blocks) +
             " cached blocks evictable";
    return false;
}

// At idle, the occupied Device state slots plus the free Device snapshot slots: constant while nothing leaks (a
// snapshot image moves a slot from one count to the other; a failed binding that kept its reserved slot raises the
// sum by one each time). Zero where the Program reports neither (Flash-Next). Polled up to 2 s, while landings settle.
inline std::int64_t idle_state_slots(Engine& engine) {
    const RuntimeStats stats = engine.runtime_stats();
    return static_cast<std::int64_t>(stats.device_state_occupied_slots) +
           static_cast<std::int64_t>(stats.hybrid_free_device_snapshot_slots);
}
inline bool state_slots_kept(Engine& engine, std::int64_t expected, std::string& detail) {
    std::int64_t now = 0;
    for (int i = 0; i < 200; ++i) {
        now = idle_state_slots(engine);
        if (now == expected) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    detail = std::to_string(now) + " occupied state slots plus free snapshot slots, " + std::to_string(expected) +
             " before the sweep";
    return false;
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
// `cancel_aborts`: cancelling a decoding request releases its lane through Program::abort (Qwen3.5); Qwen3.8-Flash-Next
// releases it inside the round's commit, so its abort faults are not reached there.
inline int run_all(Engine& engine, std::uint32_t prompt_tokens, std::uint32_t outputs, bool cancel_aborts) {
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

    // Swept counts: the failure lands from the first binding work to deep in a decode round's commit. With
    // INFERNIX_TEST_OOM_SWEEP_STRIDE=S every S-th count up to 4181 as well (a dense sweep, a few minutes): far more of
    // the windows between an acquisition and its owner's hold are hit.
    std::vector<std::uint64_t> sweep{1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181};
    if (const char* stride = std::getenv("INFERNIX_TEST_OOM_SWEEP_STRIDE"); stride != nullptr && *stride != '\0') {
        const std::uint64_t step = std::max<std::uint64_t>(1, std::strtoull(stride, nullptr, 10));
        for (std::uint64_t n = 1; n <= 4181; n += step) { sweep.push_back(n); }
        std::sort(sweep.begin(), sweep.end());
        sweep.erase(std::unique(sweep.begin(), sweep.end()), sweep.end());
        std::printf("   dense sweep: %zu counts, every %llu\n", sweep.size(), static_cast<unsigned long long>(step));
    }
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
    std::int64_t idle_slots = 0;
    for (int i = 0; i < 200; ++i) { // settled before the first injection
        idle_slots = idle_state_slots(engine);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    {
        const RuntimeStats stats = engine.runtime_stats();
        std::printf("   state slots at idle: %u occupied, %u snapshot slots free\n",
                    static_cast<unsigned>(stats.device_state_occupied_slots),
                    static_cast<unsigned>(stats.hybrid_free_device_snapshot_slots));
    }
    std::uint32_t failed = 0;
    // The first swept count that failed its request: the bound step's. (A shortage while a request binds can
    // leave it queued, so it binds again and completes; Flash-Next gives a failed binding back that way.)
    std::uint64_t failing_count = 0;
    for (std::size_t i = 0; i < sweep.size(); ++i) {
        const std::uint64_t injected = infernix::testing::memory_exhausted_injections();
        testing::request_memory_fault(sweep[i]);
        const Outcome outcome = run(engine, swept[i].prompt, outputs, true);
        testing::request_memory_fault(0); // disarms at once what did not fire, before the next request's work
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
        // Whatever the failure unwound left no cached block pinned (a pin or a fill not given back) and no state slot
        // held by nothing. Checked after the fresh request, which reuses nothing: its admission polls the swept
        // request's Host writes, and Flash-Next polls none while it idles (a request that reuses a prefix leaves its
        // write-through pinned until the next admission).
        std::string pinned, slots;
        const bool unpinned = nothing_pinned(engine, pinned);
        failures += check(unpinned, label + ": no cached block is left pinned" +
                                        (pinned.empty() ? std::string() : " (" + pinned + ")"));
        const bool slots_kept = state_slots_kept(engine, idle_slots, slots);
        failures += check(slots_kept, label + ": no state slot is left reserved" +
                                          (slots.empty() ? std::string() : " (" + slots + ")"));
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

    std::printf("== release\n");
    {
        // A device failure inside the Program's finish (testing::request_finish_fault: at the count-th check of the finish
        // call; before, the Engine failed with 'terminal native sequence could not finish'): the reply is whole, so the
        // request completes with its reference; the Engine then recovers the device (one more out-of-memory recovery) and
        // serves on. A request queued behind it completes, a fresh request returns its reference, and the extension with
        // prefix reuse returns its cold tokens (the failed finish leaves no endpoint).
        std::uint32_t fired_finishes = 0;
        for (const std::uint64_t count : {1ULL, 2ULL, 4ULL, 8ULL, 16ULL}) {
            const std::uint64_t recoveries = engine.runtime_stats().device_memory_recoveries;
            const std::uint64_t injected   = infernix::testing::memory_exhausted_injections();
            testing::request_finish_fault(count);
            Async finishing(engine, swept[2].prompt, outputs, true);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            Async waiting(engine, queued, outputs, false);
            const Outcome first  = finishing.get();
            const Outcome second = waiting.get();
            testing::request_finish_fault(0);
            const bool fired = infernix::testing::memory_exhausted_injections() > injected;
            fired_finishes += fired ? 1U : 0U;
            const std::string label = "finish fault at check " + std::to_string(count) + (fired ? " (fired)" : " (not reached)");
            failures += check(first.error.empty() && first.tokens == swept[2].reference.tokens,
                              label + ": the finishing request completed with its reference" +
                                  (first.error.empty() ? std::string() : " (error '" + first.error + "')"));
            failures += check(second.error.empty() && second.tokens == queued_reference.tokens,
                              label + ": the request queued behind it completed with its reference" +
                                  (second.error.empty() ? std::string() : " (error '" + second.error + "')"));
            const Outcome after = run(engine, fresh, outputs, false);
            failures += check(after.error.empty() && after.tokens == fresh_reference.tokens,
                              label + ": a fresh request after it returned its reference");
            // A failure finish reported is recovered at the end of its cycle; one the Program contained in its
            // publication (Flash-Next's prefix copies) needs none. Either way nothing may stay pinned.
            std::printf("   %s: %llu device recoveries\n", label.c_str(),
                        static_cast<unsigned long long>(engine.runtime_stats().device_memory_recoveries - recoveries));
            std::string pinned;
            const bool unpinned = nothing_pinned(engine, pinned);
            failures += check(unpinned, label + ": once idle, no cached block stays pinned" +
                                            (pinned.empty() ? std::string() : " (" + pinned + ")"));
            const Outcome extended = run(engine, swept[2].extended, outputs, true);
            failures += check(extended.error.empty() && extended.tokens == swept[2].extended_reference.tokens,
                              label + ": its extension, with prefix reuse, returned its cold tokens");
        }
        failures += check(fired_finishes >= 1, "a finish fault fired (" + std::to_string(fired_finishes) + " of 5)");

        // A device failure inside the Program's abort, the release of a request cancelled while it decodes: it ends
        // cancelled, the Engine recovers the device and a fresh request returns its reference.
        class Count final : public OutputSink {
        public:
            std::atomic<std::uint32_t> deltas{0};
            void start(GenerationStart) override {}
            void progress(PromptProgress) override {}
            void timing(GenerationTimingObservation) override {}
            void publish(OutputDelta) override { deltas.fetch_add(1); }
            void tool_call(ToolCallDelta) override {}
        };
        std::uint32_t fired_aborts = 0;
        for (const std::uint64_t count : {1ULL, 2ULL, 4ULL}) {
            const std::uint64_t recoveries = engine.runtime_stats().device_memory_recoveries;
            const std::uint64_t injected   = infernix::testing::memory_exhausted_injections();
            testing::request_abort_fault(count);
            Count sink;
            std::string error;
            GenerationResult result;
            try {
                GenerationHandle handle = engine.submit(engine.prepare_tokens(swept[3].prompt), greedy(outputs * 8U, false),
                                                        OutputConsumerMode::Streaming);
                result = handle.wait(&sink, CancellationView([&sink] { return sink.deltas.load() >= 2; }));
            } catch (const std::exception& e) { error = e.what(); }
            testing::request_abort_fault(0);
            const bool fired = infernix::testing::memory_exhausted_injections() > injected;
            fired_aborts += fired ? 1U : 0U;
            const std::string label = "abort fault at check " + std::to_string(count) + (fired ? " (fired)" : " (not reached)");
            failures += check(error.empty() && result.finish_reason == FinishReason::Cancelled,
                              label + ": the cancelled request ended cancelled" +
                                  (error.empty() ? std::string() : " (error '" + error + "')"));
            const Outcome after = run(engine, fresh, outputs, false);
            failures += check(after.error.empty() && after.tokens == fresh_reference.tokens,
                              label + ": a fresh request after it returned its reference");
            std::printf("   %s: %llu device recoveries\n", label.c_str(),
                        static_cast<unsigned long long>(engine.runtime_stats().device_memory_recoveries - recoveries));
            std::string pinned;
            const bool unpinned = nothing_pinned(engine, pinned);
            failures += check(unpinned, label + ": once idle, no cached block stays pinned" +
                                            (pinned.empty() ? std::string() : " (" + pinned + ")"));
        }
        if (cancel_aborts) {
            failures += check(fired_aborts >= 1, "an abort fault fired (" + std::to_string(fired_aborts) + " of 3)");
        } else {
            std::printf("   abort faults fired: %u of 3 (cancellation releases the lane in the commit)\n", fired_aborts);
        }
    }

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
