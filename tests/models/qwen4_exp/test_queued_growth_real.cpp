// A request that waits for the KV cache to grow is bound at a long prompt's next walk-span boundary, not when the
// whole prompt ends (GitHub #24). The KV cache grows by taking expert-cache frames; while a prompt streams its
// experts, its ring, wide arena and walk area are lent frames that only go back at a call or span boundary. A
// fresh engine (an empty KV pool and prefix cache, so the short request needs growth) runs a two-span prompt; a
// short prompt is submitted once the first span is under way.
//   reproducible route (no fast paths: a request's ids do not depend on the other lane)
//     order     the short request is bound (GenerationStart) at least kMargin before the long prompt's first output
//     outputs   both requests' ids equal their runs alone
// Skips (77) unless INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact (INFERNIX_QWEN4_NGRAM: its n-gram volume).
//
//   INFERNIX_QWEN4_ARTIFACT=out.ninfer infernix_qwen4_exp_queued_growth_real_test

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

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kOutputs     = 16;
constexpr std::size_t kLongTokens    = 81920; // a 65,536-token walk span, then a 16,384-token call
constexpr std::size_t kShortTokens   = 8192;
constexpr auto kSubmitAfter          = std::chrono::milliseconds(1500); // inside the long prompt's first span
constexpr auto kMargin               = std::chrono::milliseconds(1000);

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

infernix::RequestOptions greedy() {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = kOutputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    return options;
}

int check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    return ok ? 0 : 1;
}

// A streaming consumer that notes the binding (GenerationStart) and the first output.
class Watch final : public infernix::OutputSink {
public:
    std::atomic<bool> started_flag{false};
    void start(infernix::GenerationStart) override {
        set(started);
        started_flag.store(true);
    }
    void progress(infernix::PromptProgress) override {}
    void timing(infernix::GenerationTimingObservation) override {}
    void publish(infernix::OutputDelta) override { set(first_output); }
    void tool_call(infernix::ToolCallDelta) override {}
    std::optional<Clock::time_point> get_started() {
        std::lock_guard lock(mutex_);
        return started;
    }
    std::optional<Clock::time_point> get_first_output() {
        std::lock_guard lock(mutex_);
        return first_output;
    }

private:
    void set(std::optional<Clock::time_point>& at) {
        std::lock_guard lock(mutex_);
        if (!at) { at = Clock::now(); }
    }
    std::mutex mutex_;
    std::optional<Clock::time_point> started, first_output;
};

std::string ms(Clock::duration d) {
    return std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(d).count()) + " ms";
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr) {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT\n");
        return 77;
    }
    const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM");
    int failures      = 0;
    try {
        infernix::EngineOptions options;
        options.artifact_path = artifact;
        if (ngram != nullptr) { options.ngram_volume_path = ngram; }
        options.max_context     = 98304;
        options.max_concurrency = 2;
        options.kv_cache        = infernix::KvCacheStorage::Int8Group64;
        options.kv_capacity     = infernix::KvCapacityPolicy::automatic(); // the elastic pool, as the server's
        options.fast_paths      = false;
        infernix::Engine engine(options, infernix::test::engine_host());
        const std::vector<infernix::TokenId> long_prompt  = synthetic_tokens(kLongTokens, 21);
        const std::vector<infernix::TokenId> short_prompt = synthetic_tokens(kShortTokens, 22);

        // Together first, on the fresh engine: nothing cached covers the short request's pages.
        Watch long_watch, short_watch;
        infernix::GenerationHandle long_handle =
            engine.submit(engine.prepare_tokens(long_prompt), greedy(), infernix::OutputConsumerMode::Streaming);
        infernix::GenerationResult long_together;
        std::exception_ptr long_failed;
        std::atomic<bool> long_done{false};
        std::thread consumer([&] {
            try {
                long_together = long_handle.wait(&long_watch);
            } catch (...) { long_failed = std::current_exception(); }
            long_done.store(true);
        });
        while (!long_watch.started_flag.load() && !long_done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::this_thread::sleep_for(kSubmitAfter);
        std::exception_ptr short_failed;
        infernix::GenerationResult short_together;
        try {
            infernix::GenerationHandle short_handle =
                engine.submit(engine.prepare_tokens(short_prompt), greedy(), infernix::OutputConsumerMode::Streaming);
            short_together = short_handle.wait(&short_watch);
        } catch (...) { short_failed = std::current_exception(); }
        consumer.join();
        if (long_failed) { std::rethrow_exception(long_failed); }
        if (short_failed) { std::rethrow_exception(short_failed); }

        const auto short_bound = short_watch.get_started();
        const auto long_first  = long_watch.get_first_output();
        const auto short_first = short_watch.get_first_output();
        failures += check(short_bound && long_first, "both requests were bound and the long one produced output");
        if (short_bound && long_first) {
            failures += check(*short_bound + kMargin <= *long_first,
                              "the short request was bound " + ms(*long_first - *short_bound) +
                                  " before the long prompt's first output (at least " + ms(kMargin) + ")");
        }
        if (short_first && long_first) {
            std::printf("  info first outputs: short %s the long prompt's\n",
                        *short_first < *long_first ? ("ahead of, by " + ms(*long_first - *short_first)).c_str()
                                                   : ("after, by " + ms(*short_first - *long_first)).c_str());
        }

        const infernix::GenerationResult long_alone  = engine.generate(engine.prepare_tokens(long_prompt), greedy());
        const infernix::GenerationResult short_alone = engine.generate(engine.prepare_tokens(short_prompt), greedy());
        failures += check(long_together.generated_token_ids == long_alone.generated_token_ids,
                          "the long prompt's ids equal its run alone");
        failures += check(short_together.generated_token_ids == short_alone.generated_token_ids,
                          "the short prompt's ids equal its run alone");
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures == 0 ? "PASS\n" : "FAILED (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
