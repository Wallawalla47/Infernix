// Engine-level recovery from a failed expert read on the real artifact (RecoverableExecutionError):
// a fault injected into the expert error word fails the requests running in that round, the Engine
// keeps serving, and a later request gives the same greedy ids as before the fault (the prefix cache
// the failed round may have published into is emptied). Three requests run against two lanes, so
// one may wait in the queue across the recovery.
// Skips (77) unless INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact.
//
//   INFERNIX_QWEN4_ARTIFACT=out.ninfer [INFERNIX_QWEN4_NGRAM=out.ninfer.ngram] infernix_qwen4_exp_expert_fault_real_test

#include "models/qwen4_exp/program/program.h"
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

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) { ++failures; }
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
    try {
        std::mutex diagnostics_mutex;
        std::vector<std::string> diagnostics;
        infernix::EngineOptions options;
        options.artifact_path = artifact;
        if (ngram != nullptr) { options.ngram_volume_path = ngram; }
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

        // The fault lands in the first rounds the three requests run.
        infernix::models::qwen4_exp::testing::set_expert_fault(4);
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
        infernix::models::qwen4_exp::testing::set_expert_fault(0);
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
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::printf("%d checks failed\n", failures);
        return 1;
    }
    std::printf("expert fault recovery checks passed\n");
    return 0;
}
