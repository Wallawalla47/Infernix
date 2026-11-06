// Withdrawal after non-finite logits on the real artifact (hybrid-prefix-cache-spec §7.8, candidates
// SQ-01). A test hook (runtime::testing::set_nonfinite_fault) reports a row's logits as non-finite, as a
// NaN reaching the sampler would, after the request has published prefix-cache entries:
//
//   prefill   The prompt's first token fails, after its prefill chunks published blocks and a tap. The
//             request fails alone, the cache holds what it held before, the same prompt again resumes
//             from nothing and equals a cache-off run, and its repeat resumes from that run's tap.
//   decode    A request fails in decode, after blocks of its output were published (free cache resume):
//             the same again; what the earlier prompt cached stays.
//   file      Engine 1 caches a clean prompt, its periodic save writes the file, then a failing request
//             publishes and is withdrawn (a save during it deletes the file). Engine 2 loads the file
//             saved at Engine 1's stop: the clean prompt resumes, the failed one resumes from nothing.
//
// Exact route (fast paths off, free-cache-resume on), as the prefix cache test, so resumes equal cold runs.
// Skips (77) unless INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact (INFERNIX_QWEN4_NGRAM: its volume).
//
//   INFERNIX_QWEN4_ARTIFACT=out.ninfer infernix_qwen4_exp_nonfinite_withdraw_real_test

#include "test_temp_path.h"
#include "runtime/contract/execution.h"
#include "infernix/engine.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <string>
#include <system_error>
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

infernix::RequestOptions greedy(std::uint32_t outputs, bool reuse) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

struct Run {
    std::vector<infernix::TokenId> tokens;
    std::uint32_t reused = 0;
};

Run generate(infernix::Engine& engine, const std::vector<infernix::TokenId>& prompt, std::uint32_t outputs, bool reuse) {
    const infernix::GenerationResult result = engine.generate(engine.prepare_tokens(prompt), greedy(outputs, reuse));
    return Run{result.generated_token_ids, result.reused_prompt_tokens};
}

int check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    return ok ? 0 : 1;
}

// Runs `prompt` with the `rows`-th row check reporting non-finite logits; whether it failed with them.
bool fails_nonfinite(infernix::Engine& engine, const std::vector<infernix::TokenId>& prompt, std::uint32_t outputs,
                     std::uint32_t rows) {
    infernix::runtime::testing::set_nonfinite_fault(rows);
    bool failed = false;
    try {
        (void)generate(engine, prompt, outputs, true);
    } catch (const std::exception& error) {
        failed = std::string(error.what()).find("non-finite") != std::string::npos;
        if (!failed) { std::printf("        unexpected error: %s\n", error.what()); }
    }
    infernix::runtime::testing::set_nonfinite_fault(0);
    return failed;
}

// The Engine publishes its statistics after it answers a request: waits until `ready` holds (at most 5 s).
infernix::RuntimeStats stats_when(infernix::Engine& engine, const std::function<bool(const infernix::RuntimeStats&)>& ready) {
    infernix::RuntimeStats stats = engine.runtime_stats();
    for (int i = 0; i < 250 && !ready(stats); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        stats = engine.runtime_stats();
    }
    return stats;
}

// The statistics of an idle Engine, once its last request's are published.
infernix::RuntimeStats settled_stats(infernix::Engine& engine) {
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    return engine.runtime_stats();
}

infernix::EngineOptions base_options(const char* artifact, const char* ngram) {
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    if (ngram != nullptr) { options.ngram_volume_path = ngram; }
    options.max_context                       = 4096;
    options.kv_capacity                       = infernix::KvCapacityPolicy::explicit_capacity(4608);
    options.prefill_chunk                     = 512;
    options.kv_cache                          = infernix::KvCacheStorage::Int8Group64;
    options.max_concurrency                   = 1;
    options.max_pending_requests              = 1;
    options.context_cache.host_capacity_bytes = 2ULL << 30;
    options.fast_paths                        = false;
    options.fast_path_candidates              = {infernix::FastPath::FreeCacheResume};
    return options;
}

// The prompt's chunks end at 512 and 1024; its final call [1024, 1500) starts with the flexible tap.
const std::vector<infernix::TokenId>& prompt_p() {
    static const std::vector<infernix::TokenId> tokens = synthetic_tokens(1500, 31);
    return tokens;
}
const std::vector<infernix::TokenId>& prompt_q() {
    static const std::vector<infernix::TokenId> tokens = synthetic_tokens(1500, 32);
    return tokens;
}

int run_in_memory(const char* artifact, const char* ngram, std::vector<infernix::TokenId>& cold_p) {
    std::printf("== in memory\n");
    infernix::Engine engine(base_options(artifact, ngram));
    int failures = 0;
    const std::vector<infernix::TokenId>& p = prompt_p();
    const std::vector<infernix::TokenId>& q = prompt_q();
    cold_p                                   = generate(engine, p, 32, false).tokens;
    const std::vector<infernix::TokenId> cold_q = generate(engine, q, 96, false).tokens;

    // prefill: the first sampled row fails, after the prompt's chunks published. Before the failure the
    // tree held the request's blocks, so an empty tree with its taps counted is the state after it.
    const infernix::RuntimeStats before = settled_stats(engine);
    failures += check(fails_nonfinite(engine, p, 32, 1), "the prompt's first token fails with non-finite logits");
    const infernix::RuntimeStats after = stats_when(engine, [&](const infernix::RuntimeStats& s) {
        return s.hybrid_taps_created > before.hybrid_taps_created && s.hybrid_snapshots == before.hybrid_snapshots &&
               s.hybrid_tree_blocks == before.hybrid_tree_blocks;
    });
    std::printf("        the failed request published %llu taps and %llu blocks\n",
                static_cast<unsigned long long>(after.hybrid_taps_created - before.hybrid_taps_created),
                static_cast<unsigned long long>(after.hybrid_blocks_inserted - before.hybrid_blocks_inserted));
    failures += check(after.hybrid_taps_created > before.hybrid_taps_created &&
                          after.hybrid_blocks_inserted > before.hybrid_blocks_inserted,
                      "the failed request had published blocks and a tap (positive control)");
    failures += check(after.hybrid_snapshots == before.hybrid_snapshots && after.hybrid_tree_blocks == before.hybrid_tree_blocks,
                      "its blocks and snapshots are withdrawn (snapshots " + std::to_string(after.hybrid_snapshots) +
                          ", blocks " + std::to_string(after.hybrid_tree_blocks) + ")");
    const Run retry = generate(engine, p, 32, true);
    failures += check(retry.reused == 0, "the same prompt resumes from nothing (reused " + std::to_string(retry.reused) + ")");
    failures += check(retry.tokens == cold_p, "the retry equals the cache-off run");
    const Run again = generate(engine, p, 32, true);
    failures += check(again.reused > 0, "its repeat resumes from the retry's entries (reused " +
                                            std::to_string(again.reused) + ")");
    failures += check(again.tokens == cold_p, "the repeat equals the cache-off run");

    // decode: the 60th row fails (59 output tokens past the 1500-token prompt: block 23 [1472, 1536),
    // which ends in the output, was published), while P's clean entries stay.
    const infernix::RuntimeStats clean = settled_stats(engine);
    failures += check(fails_nonfinite(engine, q, 96, 60), "a decode row fails with non-finite logits");
    const infernix::RuntimeStats decoded = stats_when(engine, [&](const infernix::RuntimeStats& s) {
        return s.hybrid_blocks_inserted >= clean.hybrid_blocks_inserted + 24 &&
               s.hybrid_tree_blocks == clean.hybrid_tree_blocks && s.hybrid_snapshots == clean.hybrid_snapshots;
    });
    failures += check(decoded.hybrid_blocks_inserted >= clean.hybrid_blocks_inserted + 24,
                      "the failed request had published its prompt's 23 blocks and one of its output (" +
                          std::to_string(decoded.hybrid_blocks_inserted - clean.hybrid_blocks_inserted) + ")");
    failures += check(decoded.hybrid_snapshots == clean.hybrid_snapshots && decoded.hybrid_tree_blocks == clean.hybrid_tree_blocks,
                      "the decode failure's entries are withdrawn and P's stay (snapshots " +
                          std::to_string(decoded.hybrid_snapshots) + " of " + std::to_string(clean.hybrid_snapshots) +
                          ", blocks " + std::to_string(decoded.hybrid_tree_blocks) + " of " +
                          std::to_string(clean.hybrid_tree_blocks) + ")");
    const Run q_retry = generate(engine, q, 96, true);
    failures += check(q_retry.reused == 0 && q_retry.tokens == cold_q,
                      "Q again resumes from nothing and equals its cache-off run (reused " + std::to_string(q_retry.reused) + ")");
    const Run p_after = generate(engine, p, 32, true);
    failures += check(p_after.reused > 0 && p_after.tokens == cold_p, "P still resumes from its clean entries");
    return failures;
}

int run_file(const char* artifact, const char* ngram, const std::vector<infernix::TokenId>& cold_p) {
    std::printf("== --prefix-cache-file\n");
    const std::filesystem::path file = infernix::test::unique_temp_path("infernix_qwen4_exp_nonfinite.cache");
    std::error_code ignored;
    std::filesystem::remove(file, ignored);
    const auto options = [&] {
        infernix::EngineOptions o                       = base_options(artifact, ngram);
        o.context_cache.hybrid.persistent_file          = file;
        o.context_cache.hybrid.persistent_identity      = "nonfinite";
        o.context_cache.hybrid.persistent_save_interval = std::chrono::seconds(1);
        return o;
    };
    int failures = 0;
    {
        infernix::Engine engine(options());
        (void)generate(engine, prompt_q(), 16, true); // clean entries the file must keep
        bool saved = false;
        for (int i = 0; i < 300 && !saved; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            saved = std::filesystem::exists(file);
        }
        failures += check(saved, "the periodic save wrote the clean prompt's entries");
        const auto written = std::filesystem::last_write_time(file, ignored);
        failures += check(fails_nonfinite(engine, prompt_p(), 32, 1), "P fails with non-finite logits");
        const bool exists = std::filesystem::exists(file);
        std::printf("        after the failure the file %s\n",
                    !exists ? "was deleted (a save ran during the request)"
                    : std::filesystem::last_write_time(file, ignored) != written ? "was written again"
                                                                                  : "is the one saved before the request");
    } // the stop saves the Host tier
    failures += check(std::filesystem::exists(file), "Engine 1's stop saved the Host tier");
    {
        infernix::Engine engine(options());
        const infernix::LoadSummary::PrefixCacheRestore restore = engine.load_summary().prefix_cache;
        failures += check(restore.restored && restore.snapshots > 0, "Engine 2 restores the saved snapshots");
        const Run p = generate(engine, prompt_p(), 32, true);
        failures += check(p.reused == 0 && p.tokens == cold_p,
                          "the failed prompt resumes from nothing after the restart (reused " + std::to_string(p.reused) + ")");
        const Run q = generate(engine, prompt_q(), 16, true);
        failures += check(q.reused > 0, "the clean prompt resumes from the file (reused " + std::to_string(q.reused) + ")");
    }
    std::filesystem::remove(file, ignored);
    return failures;
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
        std::vector<infernix::TokenId> cold_p;
        int failures = run_in_memory(artifact, ngram, cold_p);
        failures += run_file(artifact, ngram, cold_p);
        std::printf(failures == 0 ? "non-finite withdrawal checks passed\n" : "FAIL: %d checks failed\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
