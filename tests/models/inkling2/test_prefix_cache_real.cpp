// Inkling 2's prefix cache on a real artifact (src/models/inkling2/program/prefix_cache.h): the cache publishes only
// prompt-computed state, and the prompt route gives the same bits whatever its chunks, so a request resumed from a
// snapshot must give a cold run's output bit for bit (tests/models/resume_cold_probe.h: the tokens, and every
// teacher-forced log-probability of the cold run's tokens, resumed and cold).
// 1. Each --kv-dtype (the full visits' pools), prefill chunks of 4,096: a 2,300-token prompt, then its continuation by
//    one separator token, resumed (it must reuse a snapshot) and cold.
// 2. Chunks of 100 (BF16): the same, every resume point inside a call's span.
// 3. A decoded prefix (BF16): the prompt decodes 24 tokens first; the continuation (prompt, the decoded tokens and the
//    separator) resumes at the prompt's snapshot, re-prefilling the decoded tokens (no decode-computed state is cached).
// 4. DSpark (--spec dspark, auto): the resumed continuation's tokens equal plain decoding's cold ones, and it drafts
//    (its acceptance reported beside a cold run's).
// 5. A Host-tier restore: with a pool that holds only one lane's context, an unrelated 3,500-token prompt evicts the
//    first prompt's cached blocks from the Device, and the continuation restores them from the Host tier.
// 6. Two lanes, the pool exactly their two contexts: two identical prompts at once (duplicate blocks), then two
//    near-full prompts at once, then a continuation of the first: every output equals its prompt's cold one.
// 7. --prefix-cache-file: saved at shutdown, restored by a new Engine, resumed with the cold run's tokens.
//
//   INFERNIX_INKLING2_ARTIFACT  an artifact converted with the inkling2_mini recipe (skips without it)

#include "infernix/engine.h"
#include "models/resume_cold_probe.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Tokens = std::vector<infernix::TokenId>;
constexpr std::uint32_t kContext = 4096;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

infernix::EngineOptions cache_options(const std::filesystem::path& artifact, infernix::KvCacheStorage kv,
                                      std::uint32_t chunk, std::uint32_t pool_tokens) {
    infernix::EngineOptions options;
    options.artifact_path                       = artifact;
    options.max_context                         = kContext;
    options.max_concurrency                     = 1;
    options.max_pending_requests                = 16;
    options.kv_cache                            = kv;
    options.kv_capacity                         = infernix::KvCapacityPolicy::explicit_capacity(pool_tokens);
    options.prefill_chunk                       = chunk;
    options.context_cache.enabled               = true;
    options.context_cache.host_capacity_bytes   = std::size_t{6} << 30U;
    options.speculative.backend                 = infernix::SpeculativeBackend::None;
    options.diagnostic_observer.callback = [](const infernix::Diagnostic& d) {
        if (d.level >= infernix::DiagnosticLevel::Info) { std::cerr << "[engine] " << d.message << '\n'; }
    };
    return options;
}

// Numbered sentences, the first `count` ids of their raw-text encoding (`salt` varies the text).
Tokens long_prompt(const infernix::Engine& engine, std::size_t count, int salt = 0) {
    static const char* const adjectives[] = {"quiet", "amber", "distant", "narrow", "restless", "silver", "patient"};
    static const char* const nouns[]      = {"harbour", "ledger", "orchard", "signal", "bridge", "lantern",
                                             "archive", "valley", "engine", "garden", "compass"};
    std::string text;
    for (int i = 0; text.size() < std::max<std::size_t>(24000, count * 12); ++i) {
        text += "Item " + std::to_string(i + salt) + ": the " + adjectives[(i + salt) % 7] + " " + nouns[i % 11] +
                " in section " + std::to_string(i * 37 % 101) + " was logged at " + std::to_string(i * 13 % 60) +
                " minutes past " + std::to_string(i % 12 + 1) + ".\n";
    }
    Tokens ids = engine.tokenize_text(text);
    require(ids.size() >= count, "the long prompt's text encodes to enough ids");
    ids.resize(count);
    return ids;
}

// INFERNIX_INKLING2_PREFIX_TIMING (a measurement, not a check; run it on a quiet machine): with the cache on, the
// prefill of a cold 10,000-token prompt that publishes (its taps copied to the Host tier) and one that does not
// (allow_prefix_reuse false), then a turn continuing a published prompt by 64 tokens resumed and cold (its first token
// and the decode rate of its 128 tokens: the first round waits for the prompt-end tap's copy). Medians of 3 prompts.
void timing(const std::filesystem::path& artifact, bool dspark) {
    infernix::EngineOptions options = cache_options(artifact, infernix::KvCacheStorage::BFloat16, 4096, 4 * 16384);
    options.max_context             = 16384;
    if (dspark) {
        options.speculative.backend           = infernix::SpeculativeBackend::DSpark;
        options.speculative.draft_tokens_auto = true;
    }
    infernix::Engine engine(options);
    const auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    std::vector<double> publish, plain, warm_first, cold_first, warm_decode, cold_decode;
    const Tokens tail = long_prompt(engine, 64, 101);
    (void)engine.generate(engine.prepare_tokens(long_prompt(engine, 10000, 900)), infernix::test::resume_probe_request(1, false));
    for (int i = 0; i < 3; ++i) {
        const Tokens prompt = long_prompt(engine, 10000, 1000 + i);
        plain.push_back(engine.generate(engine.prepare_tokens(long_prompt(engine, 10000, 2000 + i)),
                                        infernix::test::resume_probe_request(1, false))
                            .timings.prefill_seconds);
        publish.push_back(
            engine.generate(engine.prepare_tokens(prompt), infernix::test::resume_probe_request(1, true)).timings.prefill_seconds);
        Tokens turn = prompt;
        turn.insert(turn.end(), tail.begin(), tail.end());
        for (const bool reuse : {true, false}) {
            const infernix::GenerationResult r =
                engine.generate(engine.prepare_tokens(turn), infernix::test::resume_probe_request(128, reuse));
            (reuse ? warm_first : cold_first).push_back(r.timings.first_token_seconds);
            (reuse ? warm_decode : cold_decode)
                .push_back(static_cast<double>(r.generated_token_ids.size()) / r.timings.decode_seconds);
            if (reuse) { require(r.reused_prompt_tokens >= 9984, "timing: the warm turn did not resume"); }
        }
    }
    std::cout << "timing" << (dspark ? " (dspark)" : "") << ": 10,000-token prefill " << median(plain) * 1e3
              << " ms without publishing, " << median(publish) * 1e3 << " ms publishing; a 64-token continuing turn's "
              << "first token " << median(warm_first) * 1e3 << " ms resumed, " << median(cold_first) * 1e3
              << " ms cold; its decode " << median(warm_decode) << " tok/s resumed, " << median(cold_decode)
              << " tok/s cold\n";
}

void check_probe(const std::string& label, const infernix::test::ResumeProbe& probe, std::size_t at_least) {
    std::cout << label << ": reused " << probe.reused << " of " << probe.continuation << " prompt tokens; mean |dlogprob| "
              << probe.mean_abs_delta() << '\n';
    require(probe.reused >= at_least, label + ": the continuation reused fewer than " + std::to_string(at_least) + " tokens");
    require(probe.bit_identical(), label + ": the resumed continuation differs from the cold one");
}

} // namespace

int main() {
    try {
        const char* artifact = std::getenv("INFERNIX_INKLING2_ARTIFACT");
        if (artifact == nullptr || !std::filesystem::exists(artifact)) {
            std::cout << "SKIP: INFERNIX_INKLING2_ARTIFACT is not set\n";
            return 77;
        }
        using infernix::KvCacheStorage;
        if (std::getenv("INFERNIX_INKLING2_PREFIX_TIMING") != nullptr) {
            timing(artifact, false);
            timing(artifact, true);
            return 0;
        }
        const std::uint32_t pool = 8 * kContext; // the lane's context and a cache-only share
        // 1. Every KV format.
        for (const KvCacheStorage kv :
             {KvCacheStorage::BFloat16, KvCacheStorage::Bf16Packed, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
              KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
              KvCacheStorage::Q4KeyVq2Value}) {
            infernix::Engine engine(cache_options(artifact, kv, 4096, pool));
            const Tokens prompt         = long_prompt(engine, 2300);
            const infernix::TokenId sep = engine.tokenize_text("\n").front();
            check_probe(std::string("kv ") + infernix::kv_cache_cli_name(kv),
                        infernix::test::resume_probe(engine, prompt, 0, sep), 2048);
            if (kv == KvCacheStorage::BFloat16) {
                // 3. A decoded prefix: the continuation reuses the prompt's state and re-prefills the decoded tokens.
                check_probe("decoded prefix", infernix::test::resume_probe(engine, long_prompt(engine, 2200, 7), 24, sep),
                            2048);
            }
        }
        // 2. Chunks of 100.
        {
            infernix::Engine engine(cache_options(artifact, KvCacheStorage::BFloat16, 100, pool));
            const infernix::TokenId sep = engine.tokenize_text("\n").front();
            check_probe("chunks of 100", infernix::test::resume_probe(engine, long_prompt(engine, 2300), 0, sep), 2048);
        }
        // 4. DSpark: the resumed request's tokens are plain decoding's (resume_probe's cold run), and its drafter keeps its
        //    context (the full layer's pages and the sliding rings come back with the snapshot): its acceptance is
        //    reported beside a cold run's of the same continuation.
        if (std::getenv("INFERNIX_INKLING2_NO_DSPARK") == nullptr) {
            infernix::EngineOptions options       = cache_options(artifact, KvCacheStorage::BFloat16, 4096, pool);
            options.speculative.backend           = infernix::SpeculativeBackend::DSpark;
            options.speculative.draft_tokens_auto = true;
            infernix::Engine engine(options);
            const infernix::TokenId sep = engine.tokenize_text("\n").front();
            const Tokens prompt         = long_prompt(engine, 2300);
            check_probe("dspark", infernix::test::resume_probe(engine, prompt, 0, sep), 2048);
            Tokens continuation = prompt;
            continuation.push_back(sep);
            const auto stats = [&](bool reuse) {
                const infernix::GenerationResult r =
                    engine.generate(engine.prepare_tokens(continuation), infernix::test::resume_probe_request(96, reuse));
                std::cout << "dspark " << (reuse ? "resumed" : "cold") << ": reused " << r.reused_prompt_tokens << ", "
                          << r.speculative.accepted_tokens << " of " << r.speculative.drafted_tokens
                          << " drafts accepted in " << r.speculative.rounds << " rounds\n";
                return r;
            };
            const infernix::GenerationResult warm = stats(true), cold = stats(false);
            require(warm.generated_token_ids == cold.generated_token_ids, "dspark: resumed tokens differ from cold");
            require(warm.speculative.drafted_tokens != 0, "dspark: the resumed request drafted nothing");
        }
        // 6. Two lanes at once with the pool holding exactly their two contexts: two identical prompts publish the
        //    same blocks (the second lane's duplicates must not keep a page each beside the first's), then two other
        //    long prompts take every page; then a third prompt resumes. Every token equals its prompt's alone.
        {
            infernix::EngineOptions options = cache_options(artifact, KvCacheStorage::BFloat16, 4096, 2 * kContext);
            options.max_concurrency         = 2;
            infernix::Engine engine(options);
            const Tokens a = long_prompt(engine, 2300, 11), b = long_prompt(engine, 3900, 13), c = long_prompt(engine, 3900, 17);
            const auto cold = [&](const Tokens& prompt) {
                return engine.generate(engine.prepare_tokens(prompt), infernix::test::resume_probe_request(16, false))
                    .generated_token_ids;
            };
            const Tokens cold_a = cold(a), cold_b = cold(b), cold_c = cold(c);
            auto first  = engine.submit(engine.prepare_tokens(a), infernix::test::resume_probe_request(16, true));
            auto second = engine.submit(engine.prepare_tokens(a), infernix::test::resume_probe_request(16, true));
            require(first.wait().generated_token_ids == cold_a, "C = 2: the first of two identical prompts");
            require(second.wait().generated_token_ids == cold_a, "C = 2: the second of two identical prompts");
            auto third  = engine.submit(engine.prepare_tokens(b), infernix::test::resume_probe_request(16, true));
            auto fourth = engine.submit(engine.prepare_tokens(c), infernix::test::resume_probe_request(16, true));
            require(third.wait().generated_token_ids == cold_b, "C = 2: a near-full prompt beside another");
            require(fourth.wait().generated_token_ids == cold_c, "C = 2: the other near-full prompt");
            Tokens continuation = a;
            continuation.push_back(engine.tokenize_text("\n").front());
            const infernix::GenerationResult again =
                engine.generate(engine.prepare_tokens(continuation), infernix::test::resume_probe_request(16, true));
            const Tokens cold_again = cold(continuation);
            std::cout << "C = 2: the continuation reused " << again.reused_prompt_tokens << " prompt tokens\n";
            require(again.generated_token_ids == cold_again, "C = 2: the continuation differs from cold");
        }
        // 5. A Host-tier restore: the pool holds the lane's context alone, so the unrelated prompt takes every page the
        //    cache held and the continuation's path comes back from the Host tier.
        {
            infernix::Engine engine(cache_options(artifact, KvCacheStorage::BFloat16, 4096, kContext));
            const Tokens prompt         = long_prompt(engine, 2300);
            const infernix::TokenId sep = engine.tokenize_text("\n").front();
            (void)engine.generate(engine.prepare_tokens(prompt), infernix::test::resume_probe_request(1, true));
            (void)engine.generate(engine.prepare_tokens(long_prompt(engine, 3500, 3)),
                                  infernix::test::resume_probe_request(1, true));
            Tokens continuation = prompt;
            continuation.push_back(sep);
            const infernix::GenerationResult resumed =
                engine.generate(engine.prepare_tokens(continuation), infernix::test::resume_probe_request(16, true));
            const infernix::GenerationResult cold =
                engine.generate(engine.prepare_tokens(continuation), infernix::test::resume_probe_request(16, false));
            std::cout << "host restore: reused " << resumed.reused_prompt_tokens << " prompt tokens\n";
            require(resumed.reused_prompt_tokens >= 2048, "host restore: the continuation reused too little");
            require(resumed.generated_token_ids == cold.generated_token_ids, "host restore: the tokens differ from cold");
        }
        // 7. --prefix-cache-file: an Engine saves its Host tier at shutdown, a new one restores it, and the continuation
        //    resumes there with the cold run's tokens.
        {
            const std::filesystem::path file = std::filesystem::temp_directory_path() / "infernix_inkling2_prefix_test.cache";
            std::filesystem::remove(file);
            infernix::EngineOptions options = cache_options(artifact, KvCacheStorage::BFloat16, 4096, pool);
            options.context_cache.hybrid.persistent_file     = file;
            options.context_cache.hybrid.persistent_identity = "infernix_inkling2_prefix_cache_real_test";
            Tokens continuation;
            {
                infernix::Engine engine(options);
                const Tokens prompt = long_prompt(engine, 2300, 19);
                (void)engine.generate(engine.prepare_tokens(prompt), infernix::test::resume_probe_request(1, true));
                continuation = prompt;
                continuation.push_back(engine.tokenize_text("\n").front());
                engine.stop(); // the orderly stop saves the Host tier
            }
            require(std::filesystem::exists(file), "persist: no file was saved at shutdown");
            infernix::Engine engine(options);
            const infernix::GenerationResult resumed =
                engine.generate(engine.prepare_tokens(continuation), infernix::test::resume_probe_request(16, true));
            const infernix::GenerationResult cold =
                engine.generate(engine.prepare_tokens(continuation), infernix::test::resume_probe_request(16, false));
            std::cout << "persist: the restarted Engine reused " << resumed.reused_prompt_tokens << " prompt tokens\n";
            require(resumed.reused_prompt_tokens >= 2048, "persist: the restarted Engine reused too little");
            require(resumed.generated_token_ids == cold.generated_token_ids, "persist: the tokens differ from cold");
            std::filesystem::remove(file);
        }
        std::cout << "PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
