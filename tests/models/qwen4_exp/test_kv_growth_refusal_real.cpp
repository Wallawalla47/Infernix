// An idle Engine whose KV pool must grow for a request keeps that request when the device refuses the memory the
// free-memory reading offered. Under WDDM at 85 MiB free, with nothing running, the device refused even a 2 MiB KV
// chunk in the memory the expert cache's sizing counted as spare while the cache held 24 GiB of frames, and the idle
// Engine answered a request that fit its pool with HTTP 503 ("more than an idle Engine has": the agentic replay,
// 2026-11-28). A test seam (qwen4_exp::testing::set_phantom_kv_spare) makes the next growth read 1 GiB of spare
// memory the device refuses, so the growth must take its memory from the expert frames instead.
//   fresh engine (only the pool's base pages backed), one request whose prompt needs the pool to grow
//     admitted  the request completes
//     retried   the growth went on from the expert frames once (kv_growth_retries)
//     outputs   its ids equal its run without the refusal (fast paths off)
// Skips (77) unless INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact (INFERNIX_QWEN4_NGRAM: its n-gram volume).
//
//   INFERNIX_QWEN4_ARTIFACT=out.ninfer infernix_qwen4_exp_kv_growth_refusal_real_test

#include "models/qwen4_exp/program/prefix/test_taps.h"
#include "engine_host.h"
#include "infernix/engine.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

namespace {

namespace testing = infernix::models::qwen4_exp::testing;

constexpr std::uint32_t kOutputs   = 16;
constexpr std::size_t kTokens      = 32768; // past the base pages: the pool grows for it
constexpr std::uint64_t kPhantom   = 1ULL << 30;

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
        options.max_context     = 65536;
        options.max_concurrency = 1;
        options.kv_cache        = infernix::KvCacheStorage::Int8Group64;
        options.kv_capacity     = infernix::KvCapacityPolicy::automatic(); // the elastic pool, as the server's
        options.fast_paths      = false;
        infernix::Engine engine(options, infernix::test::engine_host());
        const std::vector<infernix::TokenId> prompt = synthetic_tokens(kTokens, 31);

        const std::uint64_t retries = testing::kv_growth_retries();
        testing::set_phantom_kv_spare(kPhantom);
        infernix::GenerationResult refused_first;
        std::string failed;
        try {
            refused_first = engine.generate(engine.prepare_tokens(prompt), greedy());
        } catch (const std::exception& error) { failed = error.what(); }
        testing::set_phantom_kv_spare(0); // unused if the pool never grew
        failures += check(failed.empty(), "the request needing growth completed" +
                                              (failed.empty() ? std::string() : " (it failed: " + failed + ")"));
        failures += check(testing::kv_growth_retries() == retries + 1,
                          "the growth went on from the expert frames after the refused chunk (" +
                              std::to_string(testing::kv_growth_retries() - retries) + " retries)");
        if (failed.empty()) {
            const infernix::GenerationResult again = engine.generate(engine.prepare_tokens(prompt), greedy());
            failures += check(refused_first.generated_token_ids == again.generated_token_ids,
                              "its ids equal its run without the refusal");
        }
        engine.stop();
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures == 0 ? "PASS\n" : "FAILED (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
