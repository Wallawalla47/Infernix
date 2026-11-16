// A request that finishes while another request's binding spans Engine cycles (a Vision encode lent its window
// runs a layer group a cycle; a test hook holds a text binding open the same way) is completed at once: its final
// event (a non-streaming client's whole response) does not wait for the binding, and its lane is released after
// it (QN-04, engine-architecture.md §5.1).
//   reproducible route (no fast paths: a request's ids do not depend on the other lane)
//     order     the Program finished the request beside the open binding, and its result arrived long before
//               the held binding could end
//     outputs   both requests' ids equal their runs alone
//   default route (free cache resume: a finish publishes its endpoint)
//     usage     the finished request reports the prompt tokens its snapshots made reusable, as many as a cold run
//               alone of a prompt of the same length
//     next turn its next turn resumes past the prompt, from the endpoint it published beside the transaction
// Skips (77) unless INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact (INFERNIX_QWEN4_NGRAM: its n-gram volume).
//
//   INFERNIX_QWEN4_ARTIFACT=out.ninfer infernix_qwen4_exp_finish_beside_binding_real_test

#include "models/qwen4_exp/program/prefix/test_taps.h"
#include "engine_host.h"
#include "infernix/engine.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
namespace testing = infernix::models::qwen4_exp::testing;

constexpr std::uint32_t kOutputs = 24;
constexpr auto kHold             = std::chrono::milliseconds(8000); // the binding's extra time
constexpr auto kPromptly         = std::chrono::milliseconds(3000); // the finished request's bound after it starts

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

infernix::RequestOptions greedy(bool reuse) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = kOutputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    return options;
}

int check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    return ok ? 0 : 1;
}

// A streaming consumer that notes the start and the first output.
class Watch final : public infernix::OutputSink {
public:
    std::atomic<bool> output{false};
    std::optional<Clock::time_point> started;
    void start(infernix::GenerationStart) override { started = Clock::now(); }
    void progress(infernix::PromptProgress) override {}
    void timing(infernix::GenerationTimingObservation) override {}
    void publish(infernix::OutputDelta) override { output.store(true); }
    void tool_call(infernix::ToolCallDelta) override {}
};

infernix::EngineOptions engine_options(const char* artifact, const char* ngram, bool fast_paths) {
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    if (ngram != nullptr) { options.ngram_volume_path = ngram; }
    options.max_context     = 8192;
    options.max_concurrency = 2;
    options.kv_cache        = infernix::KvCacheStorage::Int8Group64;
    options.fast_paths      = fast_paths;
    return options;
}

// The `held` request decodes while the `other` request's binding is held open for kHold.
struct Beside {
    infernix::GenerationResult held;
    infernix::GenerationResult other;
    std::uint64_t held_finishes = 0;
    Clock::duration held_after_other_start{};
};

Beside beside(infernix::Engine& engine, const std::vector<infernix::TokenId>& held_prompt, bool held_reuse,
              const std::vector<infernix::TokenId>& other_prompt) {
    Beside out;
    const std::uint64_t before = testing::held_finishes();
    Watch held_watch;
    infernix::GenerationHandle held_handle =
        engine.submit(engine.prepare_tokens(held_prompt), greedy(held_reuse), infernix::OutputConsumerMode::Streaming);
    std::optional<Clock::time_point> held_done;
    std::thread consumer([&] {
        out.held  = held_handle.wait(&held_watch);
        held_done = Clock::now();
    });
    while (!held_watch.output.load()) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    testing::set_binding_hold(kHold);
    Watch other_watch;
    infernix::GenerationHandle other_handle =
        engine.submit(engine.prepare_tokens(other_prompt), greedy(false), infernix::OutputConsumerMode::Streaming);
    out.other = other_handle.wait(&other_watch);
    consumer.join();
    testing::set_binding_hold(std::chrono::milliseconds(0));
    out.held_finishes = testing::held_finishes() - before;
    if (held_done && other_watch.started) { out.held_after_other_start = *held_done - *other_watch.started; }
    return out;
}

std::vector<infernix::TokenId> next_turn(std::vector<infernix::TokenId> prompt, const infernix::GenerationResult& reply) {
    prompt.insert(prompt.end(), reply.generated_token_ids.begin(), reply.generated_token_ids.end());
    const auto more = synthetic_tokens(64, 99);
    prompt.insert(prompt.end(), more.begin(), more.end());
    return prompt;
}

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
        const std::vector<infernix::TokenId> other_prompt = synthetic_tokens(512, 13);
        {
            std::printf("== reproducible route\n");
            infernix::Engine engine(engine_options(artifact, ngram, false), infernix::test::engine_host());
            const std::vector<infernix::TokenId> held_prompt = synthetic_tokens(1024, 12);
            const infernix::GenerationResult held_alone  = engine.generate(engine.prepare_tokens(held_prompt), greedy(false));
            const infernix::GenerationResult other_alone = engine.generate(engine.prepare_tokens(other_prompt), greedy(false));
            const Beside run = beside(engine, held_prompt, false, other_prompt);
            failures += check(run.held_finishes == 1, "the Program finished the request beside the open binding (" +
                                                          std::to_string(run.held_finishes) + " held finishes)");
            failures += check(run.held_after_other_start < kPromptly,
                              "its result arrived " + ms(run.held_after_other_start) +
                                  " after the binding began, not after the binding's " + ms(kHold));
            failures += check(run.held.generated_token_ids == held_alone.generated_token_ids,
                              "the finished request's ids equal its run alone");
            failures += check(run.other.generated_token_ids == other_alone.generated_token_ids,
                              "the binding request's ids equal its run alone");
        }
        {
            std::printf("== default route\n");
            infernix::Engine engine(engine_options(artifact, ngram, true), infernix::test::engine_host());
            constexpr std::size_t kPrompt = 1536;
            const std::vector<infernix::TokenId> alone_prompt = synthetic_tokens(kPrompt, 21);
            const std::vector<infernix::TokenId> held_prompt  = synthetic_tokens(kPrompt, 22);
            const infernix::GenerationResult alone      = engine.generate(engine.prepare_tokens(alone_prompt), greedy(true));
            const infernix::GenerationResult alone_next = engine.generate(engine.prepare_tokens(next_turn(alone_prompt, alone)),
                                                                          greedy(true));
            const Beside run = beside(engine, held_prompt, true, other_prompt);
            const infernix::GenerationResult held_next = engine.generate(engine.prepare_tokens(next_turn(held_prompt, run.held)),
                                                                         greedy(true));
            const std::uint32_t created       = run.held.cache_created_prompt_tokens.value_or(0);
            const std::uint32_t created_alone = alone.cache_created_prompt_tokens.value_or(0);
            failures += check(run.held_finishes == 1, "the Program finished the request beside the open binding");
            failures += check(run.held.cache_created_prompt_tokens.has_value() && run.held.expert_cache.has_value() &&
                                  run.held.timings.decode_seconds > 0.0,
                              "the finished request reports its usage fields");
            failures += check(created_alone > 0 && created == created_alone,
                              "it made as many prompt tokens reusable as a cold run alone (" + std::to_string(created) +
                                  " vs " + std::to_string(created_alone) + ")");
            // Past the prompt only a decode-computed snapshot resumes: for the held lane, the endpoint it published
            // beside the transaction. How far past depends on its drafting state at the finish (an endpoint with
            // unfinished MTP blocks is not taken), so the run alone, of another prompt, is not a bound on it.
            failures += check(alone_next.reused_prompt_tokens > kPrompt && held_next.reused_prompt_tokens > kPrompt,
                              "its next turn resumes from its endpoint, past the prompt as the next turn alone does (" +
                                  std::to_string(held_next.reused_prompt_tokens) + " and " +
                                  std::to_string(alone_next.reused_prompt_tokens) + " of " +
                                  std::to_string(kPrompt + kOutputs) + " tokens)");
        }
    } catch (const std::exception& error) {
        testing::set_binding_hold(std::chrono::milliseconds(0));
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures == 0 ? "finish beside binding checks passed\n" : "FAIL: %d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
