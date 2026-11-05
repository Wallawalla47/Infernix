// Engine-level recovery from a failed expert read on the real artifact (RecoverableExecutionError):
// a fault injected into the expert error word fails the requests running in that round, the Engine
// keeps serving, and a later request gives the same greedy ids as before the fault (the prefix cache
// the failed round may have published into is emptied). Three requests run against two lanes, so
// one may wait in the queue across the recovery.
//
// The fault is armed for the fourth check (one per prefill call or decode round, process-wide), so which round it
// fails depends on how the three requests interleave.
//
// With INFERNIX_QWEN4_DFLASH2 naming the DFlash2 companion the same recovery runs again under the drafter (7 drafts),
// and then a fault aimed at a verification round: one request alone, the fault armed once its first token is
// committed, so the next check is one of its verification rounds (the concurrent pass's fourth check was the second
// request's prefill call). That round fails with its gated commit pending, and the request after the recovery must
// give the baseline's ids (candidates WF-03; ninferpro f94d72ad6e). The check fails the round before it records its
// width, so this does not reach QA-16 (b)'s round-width resets: with them removed it passes too (2026-11-05); every
// abandonment that could leave a width behind ends the Engine.
// Skips (77) unless INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact.
//
//   INFERNIX_QWEN4_ARTIFACT=out.ninfer [INFERNIX_QWEN4_NGRAM=out.ninfer.ngram]
//   [INFERNIX_QWEN4_DFLASH2=out.dflash2.ninfer] infernix_qwen4_exp_expert_fault_real_test

#include "runtime/expert_offload/expert_offload.h"
#include "infernix/engine.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

infernix::PromptInput story() {
    infernix::PromptInput input;
    infernix::ChatMessage message;
    message.role = infernix::ChatRole::User;
    message.parts.push_back({.kind = infernix::MessagePartKind::Text,
                             .text = "Write a story about a lighthouse keeper who befriends a seagull."});
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

infernix::RequestOptions greedy(std::uint32_t tokens) {
    infernix::RequestOptions request;
    request.execution.requested_output_tokens = tokens;
    request.execution.sampling.temperature    = 0.0F;
    return request;
}

// Arms the fault once the request's first token is committed: its prefill is over, so the next check is one of its
// decode rounds (a verification round under the drafter). Called on the waiting thread only.
class ArmAfterFirstToken final : public infernix::OutputSink {
public:
    void start(infernix::GenerationStart) override {}
    void progress(infernix::PromptProgress) override {}
    void timing(infernix::GenerationTimingObservation timing) override {
        if (!armed_ && timing.generated_tokens >= 1) {
            armed_ = true;
            infernix::runtime::expert_offload::testing::set_expert_fault(1);
        }
    }
    void publish(infernix::OutputDelta) override {}
    void tool_call(infernix::ToolCallDelta) override {}

private:
    bool armed_ = false;
};

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) { ++failures; }
}

// One recovery pass: `drafter` names the DFlash2 companion (plain decoding when null).
int run(const char* artifact, const char* ngram, const char* drafter) {
    failures = 0;
    try {
        std::mutex diagnostics_mutex;
        std::vector<std::string> diagnostics;
        infernix::EngineOptions options;
        options.artifact_path = artifact;
        if (ngram != nullptr) { options.ngram_volume_path = ngram; }
        if (drafter != nullptr) {
            options.speculative.backend      = infernix::SpeculativeBackend::DFlash2;
            options.speculative.draft_tokens = 7;
            options.speculative.draft_model  = drafter;
        }
        options.max_context     = 4096;
        options.max_concurrency = 2;
        options.kv_cache        = infernix::KvCacheStorage::Int8Group64;
        options.kv_capacity     = infernix::KvCapacityPolicy::explicit_capacity(8192);
        options.context_cache.host_capacity_bytes = 2ULL << 30;
        options.diagnostic_observer.callback = [&](const infernix::Diagnostic& d) {
            const std::lock_guard<std::mutex> lock(diagnostics_mutex);
            diagnostics.push_back(d.message);
            std::printf("  [engine] %s\n", d.message.c_str());
        };
        infernix::Engine engine(options);

        constexpr std::uint32_t kTokens = 48;
        const auto baseline = engine.generate(engine.prepare(story()), greedy(kTokens)).generated_token_ids;
        check(baseline.size() == kTokens, "the baseline generates its tokens");

        // The fault lands in the first rounds the three requests run (a verification round under the drafter).
        infernix::runtime::expert_offload::testing::set_expert_fault(4);
        std::vector<std::optional<std::vector<infernix::TokenId>>> ids(3);
        std::vector<std::string> errors(3);
        std::vector<std::thread> threads;
        for (std::size_t r = 0; r < 3; ++r) {
            threads.emplace_back([&, r] {
                try {
                    ids[r] = engine.generate(engine.prepare(story()), greedy(kTokens)).generated_token_ids;
                } catch (const std::exception& error) { errors[r] = error.what(); }
            });
        }
        for (std::thread& thread : threads) { thread.join(); }
        infernix::runtime::expert_offload::testing::set_expert_fault(0);
        std::size_t failed = 0;
        for (std::size_t r = 0; r < 3; ++r) {
            if (ids[r]) {
                check(*ids[r] == baseline, "request " + std::to_string(r) + " completed with the baseline's ids");
            } else {
                ++failed;
                check(errors[r].find("could not be read") != std::string::npos,
                      "request " + std::to_string(r) + " failed with the expert read error: " + errors[r]);
            }
        }
        check(failed >= 1, "the injected fault failed the request(s) running in its round");
        bool reported = false;
        {
            const std::lock_guard<std::mutex> lock(diagnostics_mutex);
            for (const std::string& line : diagnostics) {
                reported = reported || line.find("the Engine keeps serving") != std::string::npos;
            }
        }
        check(reported, "the Engine reported the recovery");

        const auto after = engine.generate(engine.prepare(story()), greedy(kTokens)).generated_token_ids;
        check(after == baseline, "after the recovery a request gives the baseline's ids");

        if (drafter != nullptr) {
            ArmAfterFirstToken sink;
            std::string error;
            try {
                infernix::GenerationObservationOptions observation;
                observation.live_timings = true; // a timing per commit, not only the final one
                (void)engine
                    .submit(engine.prepare(story()), greedy(kTokens), infernix::OutputConsumerMode::Streaming,
                            observation)
                    .wait(&sink);
            } catch (const std::exception& e) { error = e.what(); }
            infernix::runtime::expert_offload::testing::set_expert_fault(0);
            check(error.find("could not be read") != std::string::npos,
                  "a fault in a verification round failed its request: " + error);
            const auto after_verification = engine.generate(engine.prepare(story()), greedy(kTokens)).generated_token_ids;
            check(after_verification == baseline, "after a verification round's recovery a request gives the baseline's ids");
        }
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::printf("%d checks failed\n", failures);
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr) {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT\n");
        return 77;
    }
    const char* ngram   = std::getenv("INFERNIX_QWEN4_NGRAM");
    const char* drafter = std::getenv("INFERNIX_QWEN4_DFLASH2");
    std::printf("plain decoding:\n");
    int status = run(artifact, ngram, nullptr);
    if (drafter != nullptr) {
        std::printf("DFlash2 drafting:\n");
        status += run(artifact, ngram, drafter);
    }
    if (status != 0) { return 1; }
    std::printf("expert fault recovery checks passed\n");
    return 0;
}
