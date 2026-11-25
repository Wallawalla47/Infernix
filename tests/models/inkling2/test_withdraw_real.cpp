// Inkling 2's withdrawal of what a failed request published (hybrid-prefix-cache-spec §7.8) on a real artifact. A test
// hook (runtime::testing::set_nonfinite_fault) reports a row's logits as non-finite, as a NaN reaching the sampler
// would, after the request has published prefix-cache entries:
//
//   alone     The prompt's first token fails after its prefill published. The same prompt again resumes from nothing
//             and equals a cache-off run; its repeat resumes from that run's entries.
//   shared    Two lanes: request A (prompt P) fails at its first token while request B (P and a long tail) is still
//             prefilling. B's output equals its cache-off run, P resumes from nothing, and B's prompt resumes from
//             B's own entries. (A prompt publishes its blocks at its prefill's end, so B, admitted before that,
//             never maps A's: the withdrawal's pinned remainder, retried at later releases, is not reached here.)
//   file      With --prefix-cache-file, a clean prompt is saved, then a failing request publishes and is withdrawn:
//             a new Engine restores the file; the clean prompt resumes, the failed one resumes from nothing.
//
// Plain decoding (no drafter), prefill chunks of 512. Skips (77) without INFERNIX_INKLING2_ARTIFACT.

#include "test_temp_path.h"
#include "infernix/engine.h"
#include "runtime/contract/execution.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace {

using Tokens = std::vector<infernix::TokenId>;

int check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    return ok ? 0 : 1;
}

// Numbered sentences' raw-text ids, the first `count` (`salt` varies the text).
Tokens prompt(const infernix::Engine& engine, std::size_t count, int salt) {
    std::string text;
    for (int i = 0; text.size() < count * 12; ++i) {
        text += "Entry " + std::to_string(i + salt) + " records the " + std::to_string(i * 37 % 101) +
                "th reading of gauge " + std::to_string((i + salt) % 17) + " at " + std::to_string(i % 24) + ":00.\n";
    }
    Tokens ids = engine.tokenize_text(text);
    if (ids.size() < count) { throw std::runtime_error("the prompt's text encodes to too few ids"); }
    ids.resize(count);
    return ids;
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
    Tokens tokens;
    std::uint32_t reused = 0;
    std::optional<std::uint32_t> created; // cache_created_prompt_tokens
};

Run generate(infernix::Engine& engine, const Tokens& p, std::uint32_t outputs, bool reuse) {
    const infernix::GenerationResult r = engine.generate(engine.prepare_tokens(p), greedy(outputs, reuse));
    return Run{r.generated_token_ids, r.reused_prompt_tokens, r.cache_created_prompt_tokens};
}

bool failed_nonfinite(const std::exception& error) {
    const bool nonfinite = std::string(error.what()).find("non-finite") != std::string::npos;
    if (!nonfinite) { std::printf("        unexpected error: %s\n", error.what()); }
    return nonfinite;
}

// Runs `p` with the next row check reporting non-finite logits; whether it failed with them.
bool fails_nonfinite(infernix::Engine& engine, const Tokens& p, std::uint32_t outputs) {
    infernix::runtime::testing::set_nonfinite_fault(1);
    bool failed = false;
    try {
        (void)generate(engine, p, outputs, true);
    } catch (const std::exception& error) { failed = failed_nonfinite(error); }
    infernix::runtime::testing::set_nonfinite_fault(0);
    return failed;
}

infernix::EngineOptions options(const char* artifact, std::uint32_t lanes) {
    infernix::EngineOptions o;
    o.artifact_path                     = artifact;
    o.max_context                       = 8192;
    o.max_concurrency                   = lanes;
    o.max_pending_requests              = 4;
    o.kv_cache                          = infernix::KvCacheStorage::BFloat16;
    o.kv_capacity                       = infernix::KvCapacityPolicy::explicit_capacity(lanes * 8192);
    o.prefill_chunk                     = 512;
    o.context_cache.enabled             = true;
    o.context_cache.host_capacity_bytes = std::size_t{4} << 30U;
    o.speculative.backend               = infernix::SpeculativeBackend::None;
    o.diagnostic_observer.callback = [](const infernix::Diagnostic& d) {
        if (d.level >= infernix::DiagnosticLevel::Warning) { std::printf("        [engine] %s\n", d.message.c_str()); }
    };
    return o;
}

int run_alone(const char* artifact) {
    std::printf("== alone\n");
    infernix::Engine engine(options(artifact, 1));
    const Tokens p    = prompt(engine, 1500, 0);
    const Tokens cold = generate(engine, p, 16, false).tokens;
    int failures      = check(fails_nonfinite(engine, p, 16), "the prompt's first token fails with non-finite logits");
    const Run retry   = generate(engine, p, 16, true);
    failures += check(retry.reused == 0, "the same prompt resumes from nothing (reused " + std::to_string(retry.reused) + ")");
    failures += check(retry.tokens == cold, "the retry equals the cache-off run");
    // Anthropic's cache_creation_input_tokens: the retry's snapshots made prompt tokens past its (empty) reuse
    // reusable; its repeat creates at most what lies past its own reuse.
    failures += check(retry.created && *retry.created > 0 && *retry.created <= p.size(),
                      "the retry reports the prompt tokens its snapshots cached (" +
                          (retry.created ? std::to_string(*retry.created) : std::string("none")) + ")");
    const Run again = generate(engine, p, 16, true);
    failures += check(again.reused > 0 && again.tokens == cold,
                      "its repeat resumes from the retry's entries (reused " + std::to_string(again.reused) + ")");
    failures += check(again.created && again.reused + *again.created <= p.size(),
                      "the repeat's created tokens lie past its reuse");
    return failures;
}

int run_shared(const char* artifact) {
    std::printf("== shared\n");
    infernix::Engine engine(options(artifact, 2));
    const Tokens p = prompt(engine, 1500, 100);
    Tokens long_p  = p;
    const Tokens tail = prompt(engine, 5000, 200);
    long_p.insert(long_p.end(), tail.begin(), tail.end());
    const Tokens cold_p    = generate(engine, p, 16, false).tokens;
    const Tokens cold_long = generate(engine, long_p, 4, false).tokens;
    // A samples long before B (B's prompt is four times longer and its chunks rotate with A's), so the armed row is A's.
    infernix::runtime::testing::set_nonfinite_fault(1);
    infernix::GenerationHandle a = engine.submit(engine.prepare_tokens(p), greedy(16, true));
    infernix::GenerationHandle b = engine.submit(engine.prepare_tokens(long_p), greedy(4, true));
    bool a_failed = false;
    try {
        (void)a.wait();
    } catch (const std::exception& error) { a_failed = failed_nonfinite(error); }
    infernix::runtime::testing::set_nonfinite_fault(0);
    int failures = check(a_failed, "request A fails at its first token");
    const infernix::GenerationResult b_result = b.wait();
    failures += check(b_result.generated_token_ids == cold_long, "request B, beside the failure, equals its cache-off run");
    // B published its own blocks and snapshots at its prompt's end (after A's withdrawal): they serve B's prompt.
    const Run long_retry = generate(engine, long_p, 4, true);
    failures += check(long_retry.reused > 0 && long_retry.tokens == cold_long,
                      "B's prompt resumes from B's own entries and equals its cache-off run (reused " +
                          std::to_string(long_retry.reused) + ")");
    const Run retry = generate(engine, p, 16, true);
    failures += check(retry.reused == 0, "after B's release P resumes from nothing (reused " + std::to_string(retry.reused) + ")");
    failures += check(retry.tokens == cold_p, "P equals its cache-off run");
    return failures;
}

int run_file(const char* artifact) {
    std::printf("== --prefix-cache-file\n");
    const std::filesystem::path file = infernix::test::unique_temp_path("infernix_inkling2_withdraw.cache");
    std::error_code ignored;
    std::filesystem::remove(file, ignored);
    const auto with_file = [&] {
        infernix::EngineOptions o                       = options(artifact, 1);
        o.context_cache.hybrid.persistent_file          = file;
        o.context_cache.hybrid.persistent_identity      = "withdraw";
        o.context_cache.hybrid.persistent_save_interval = std::chrono::seconds(1);
        return o;
    };
    int failures = 0;
    Tokens p, q, cold_p;
    {
        infernix::Engine engine(with_file());
        p      = prompt(engine, 1500, 300);
        q      = prompt(engine, 1500, 400);
        cold_p = generate(engine, p, 16, false).tokens;
        (void)generate(engine, q, 16, true); // clean entries the file must keep
        bool saved = false;
        for (int i = 0; i < 300 && !saved; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            saved = std::filesystem::exists(file);
        }
        failures += check(saved, "the periodic save wrote the clean prompt's entries");
        failures += check(fails_nonfinite(engine, p, 16), "P fails with non-finite logits");
    } // the stop saves the Host tier
    failures += check(std::filesystem::exists(file), "the stop saved the Host tier");
    {
        infernix::Engine engine(with_file());
        const Run rp = generate(engine, p, 16, true);
        failures += check(rp.reused == 0 && rp.tokens == cold_p,
                          "the failed prompt resumes from nothing after the restart (reused " + std::to_string(rp.reused) + ")");
        const Run rq = generate(engine, q, 16, true);
        failures += check(rq.reused > 0, "the clean prompt resumes from the file (reused " + std::to_string(rq.reused) + ")");
    }
    std::filesystem::remove(file, ignored);
    failures += check(!std::filesystem::exists(file), "the cache file is removed");
    return failures;
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* artifact = std::getenv("INFERNIX_INKLING2_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::printf("SKIP: set INFERNIX_INKLING2_ARTIFACT\n");
        return 77;
    }
    try {
        int failures = run_alone(artifact);
        failures += run_shared(artifact);
        failures += run_file(artifact);
        std::printf(failures == 0 ? "withdrawal checks passed\n" : "FAIL: %d checks failed\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
