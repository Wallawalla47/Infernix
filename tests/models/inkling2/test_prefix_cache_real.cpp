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
// 8. --a16-activations with --reproducible, and --q8-weights: resumed equals cold there too.
// 9. Tap reach: a snapshot after a call ending k tokens past a frontier serves it (k up to 29; 15 with DSpark).
// 10. Images (INFERNIX_INKLING2_VISION_ARTIFACT): another image never resumes past the image; the same image does.
// 11. The SSD tier (INFERNIX_INKLING2_PREFIX_SSD_ONLY runs it alone): what the Host tier gives up is written back, read
//     back on a hit, and resumes as cold.
//
//   INFERNIX_INKLING2_ARTIFACT  an artifact converted with the inkling2_mini recipe (skips without it)

#include "infernix/engine.h"
#include "media/decode/decode.h"
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

// 11. The SSD tier (--ssd-context-gib, --ssd-context-dir): a Host tier of one prompt's snapshot and blocks, so three
//     unrelated prompts push the first one's out to the SSD; its continuation reads them back (a promotion) and
//     equals its cold run. The prompts are 3,900 tokens: reading a shorter prompt's 200 MB image and blocks back from
//     an SSD at the cost model's 2 GB/s takes longer than its prefill (a 2,300-token prompt: about 150 ms against
//     136 ms), so the source choice re-prefills it. INFERNIX_INKLING2_PREFIX_SSD_ONLY runs this case alone.
void ssd_tier(const std::filesystem::path& artifact) {
    using infernix::KvCacheStorage;
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "infernix_inkling2_ssd_tier";
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    std::filesystem::create_directories(dir);
    infernix::EngineOptions options = cache_options(artifact, KvCacheStorage::BFloat16, 4096, kContext);
    options.context_cache.host_capacity_bytes      = std::size_t{500} << 20U;
    options.context_cache.ssd_capacity_bytes       = std::uint64_t{4} << 30U;
    options.context_cache.ssd_write_bytes_per_hour = infernix::ContextCacheOptions::kSsdWritesUnlimited;
    options.context_cache.ssd_directory            = dir;
    {
        infernix::Engine engine(options);
        const Tokens prompt         = long_prompt(engine, 3900, 41);
        const infernix::TokenId sep = engine.tokenize_text("\n").front();
        (void)engine.generate(engine.prepare_tokens(prompt), infernix::test::resume_probe_request(1, true));
        for (int k = 0; k < 3; ++k) {
            (void)engine.generate(engine.prepare_tokens(long_prompt(engine, 3900, 60 + k)),
                                  infernix::test::resume_probe_request(1, true));
        }
        Tokens continuation = prompt;
        continuation.push_back(sep);
        const infernix::RuntimeStats before = engine.runtime_stats();
        const infernix::GenerationResult resumed =
            engine.generate(engine.prepare_tokens(continuation), infernix::test::resume_probe_request(16, true));
        const infernix::RuntimeStats after = engine.runtime_stats();
        const infernix::GenerationResult cold =
            engine.generate(engine.prepare_tokens(continuation), infernix::test::resume_probe_request(16, false));
        std::cout << "ssd tier (" << static_cast<double>(after.hybrid_ssd_capacity_bytes) / 1e9 << " GB): reused "
                  << resumed.reused_prompt_tokens << " prompt tokens; "
                  << static_cast<double>(after.hybrid_ssd_write_bytes) / 1e6 << " MB written ("
                  << after.hybrid_ssd_image_writes << " images, " << after.hybrid_ssd_block_writes << " blocks), "
                  << static_cast<double>(after.hybrid_ssd_read_bytes) / 1e6 << " MB read in "
                  << after.hybrid_ssd_promotions - before.hybrid_ssd_promotions << " promotions, "
                  << after.hybrid_ssd_failures << " failures; Host (" << after.hybrid_host_capacity_bytes / 1048576
                  << " MiB): " << after.hybrid_host_image_writes << " images and " << after.hybrid_host_block_writes
                  << " blocks written, " << after.hybrid_host_image_write_refusals << " and "
                  << after.hybrid_host_block_write_refusals << " refused, " << after.hybrid_host_snapshot_evictions
                  << " snapshots evicted unwritten, " << after.hybrid_unbacked_node_losses << " blocks lost\n";
        require(after.hybrid_ssd_write_bytes != 0, "ssd tier: nothing the Host tier gave up was written back");
        require(after.hybrid_ssd_promotions > before.hybrid_ssd_promotions,
                "ssd tier: the continuation's source was not read back from the SSD tier");
        require(resumed.reused_prompt_tokens >= 3840, "ssd tier: the continuation reused too little");
        require(resumed.generated_token_ids == cold.generated_token_ids, "ssd tier: the tokens differ from cold");
        require(after.hybrid_ssd_failures == 0, "ssd tier: an SSD transfer failed");
    }
    std::filesystem::remove_all(dir, ignored);
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
        if (std::getenv("INFERNIX_INKLING2_PREFIX_SSD_ONLY") != nullptr) {
            ssd_tier(artifact);
            std::cout << "PASS (the SSD tier case alone)\n";
            return 0;
        }
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
        // 8. The other routes: --a16-activations with --reproducible (its prompt calls run the narrow kernels) and
        //    --q8-weights: a resume equals a cold run bit for bit there too.
        for (int arm = 0; arm < 2; ++arm) {
            infernix::EngineOptions options = cache_options(artifact, KvCacheStorage::BFloat16, 4096, pool);
            if (arm == 0) {
                options.expert_a16_activations = true;
                options.fast_paths             = false;
            } else {
                options.q8_weights = true;
            }
            infernix::Engine engine(options);
            const infernix::TokenId sep = engine.tokenize_text("\n").front();
            check_probe(arm == 0 ? "a16 reproducible" : "q8 weights",
                        infernix::test::resume_probe(engine, long_prompt(engine, 2300, 23 + arm), 0, sep), 2048);
        }
        // 9. Taps inside a call's reach: a token prompt whose generation opener (an exact tap) lies k tokens before a
        //    prefill call's end. The image after that call serves it when k <= 29 (15 with DSpark, which splits the call
        //    at k = 16); a prompt that diverges right after the opener resumes there, with the cold run's bits.
        for (const bool dspark : {false, true}) {
            infernix::EngineOptions options = cache_options(artifact, KvCacheStorage::BFloat16, 1024, pool);
            if (dspark) {
                options.speculative.backend           = infernix::SpeculativeBackend::DSpark;
                options.speculative.draft_tokens_auto = true;
            }
            infernix::Engine engine(options);
            // The system and model message ids, from a rendered chat with a reply: its first id, and the reply's first
            // (its first generated position: tokenize_chat marks the model's messages).
            infernix::PromptInput chat;
            chat.messages.push_back({.role = infernix::ChatRole::System, .parts = {{.text = "Be brief."}}});
            chat.messages.push_back({.role = infernix::ChatRole::User, .parts = {{.text = "Hello."}}});
            chat.messages.push_back({.role = infernix::ChatRole::Assistant, .parts = {{.text = "Hi."}}});
            const infernix::ChatTokens rendered = engine.tokenize_chat(chat);
            std::size_t opener = 0;
            while (opener < rendered.generated.size() && rendered.generated[opener] == 0) { ++opener; }
            require(opener < rendered.ids.size(), "reach: the chat has no model message");
            const infernix::TokenId system = rendered.ids.front(), model = rendered.ids[opener];
            for (const std::uint32_t k : dspark ? std::vector<std::uint32_t>{2, 15, 16} : std::vector<std::uint32_t>{2, 15, 29}) {
                const std::uint32_t P = 1024 - k;
                const Tokens filler   = long_prompt(engine, 2300, 31 + static_cast<int>(k));
                Tokens first          = filler;
                first[0]              = system;
                first[P]              = model;
                (void)engine.generate(engine.prepare_tokens(first), infernix::test::resume_probe_request(1, true));
                Tokens second(first.begin(), first.begin() + P + 1);
                const Tokens other = long_prompt(engine, 400, 77 + static_cast<int>(k));
                second.insert(second.end(), other.begin(), other.end());
                const infernix::GenerationResult cold =
                    engine.generate(engine.prepare_tokens(second), infernix::test::resume_probe_request(16, false));
                infernix::RequestOptions read          = infernix::test::resume_probe_request(16, true);
                infernix::RequestOptions read_cold     = infernix::test::resume_probe_request(16, false);
                read.execution.readout_tokens          = {cold.generated_token_ids.front()};
                read_cold.execution.readout_tokens     = {cold.generated_token_ids.front()};
                const infernix::GenerationResult warm  = engine.generate(engine.prepare_tokens(second), read);
                const infernix::GenerationResult again = engine.generate(engine.prepare_tokens(second), read_cold);
                std::cout << "reach" << (dspark ? " (dspark)" : "") << ", opener " << k << " before the call's end: reused "
                          << warm.reused_prompt_tokens << '\n';
                require(warm.reused_prompt_tokens >= P, "reach: the diverging prompt did not resume at the opener");
                require(warm.generated_token_ids == cold.generated_token_ids, "reach: tokens differ from cold");
                require(warm.readout && again.readout && warm.readout->logprobs == again.readout->logprobs,
                        "reach: the first token's log-probability differs from cold");
            }
        }
        // 10. Images (INFERNIX_INKLING2_VISION_ARTIFACT): two images of the same cell count after a long system prompt.
        //     The second must not resume past the image's start (its key differs; placeholders are equal tokens), and
        //     each equals its cold run; the first image again resumes past it. The question after the image is long, so
        //     the generation opener lies more than kMinimumTapSeparation past the user message's start (the planner
        //     keeps one tap of closer hints, the earliest).
        if (const char* vision = std::getenv("INFERNIX_INKLING2_VISION_ARTIFACT"); vision != nullptr) {
            infernix::EngineOptions options = cache_options(vision, KvCacheStorage::BFloat16, 4096, pool);
            options.enable_vision           = true;
            infernix::Engine engine(options);
            const auto picture = [](int phase) {
                std::vector<std::uint8_t> rgb(200 * 140 * 3);
                for (int y = 0; y < 140; ++y) {
                    for (int x = 0; x < 200; ++x) {
                        std::uint8_t* p = rgb.data() + (static_cast<std::size_t>(y) * 200 + x) * 3;
                        const bool disc  = (x - 60 - phase) * (x - 60 - phase) + (y - 65) * (y - 65) < 45 * 45;
                        p[0] = disc ? 220 : 30, p[1] = disc ? 30 : 60, p[2] = disc ? 30 : 200;
                    }
                }
                return infernix::media::decode::encode_raw_rgb(200, 140, rgb);
            };
            std::string system_text, question = "What is in this picture?";
            for (int i = 0; i < 300; ++i) { system_text += "Rule " + std::to_string(i) + ": answer in plain words. "; }
            for (int i = 0; i < 30; ++i) { question += " Describe its shape number " + std::to_string(i) + " and its colours."; }
            const auto ask = [&](const std::vector<std::uint8_t>& image) {
                infernix::PromptInput input;
                input.messages.push_back({.role = infernix::ChatRole::System, .parts = {{.text = system_text}}});
                infernix::ChatMessage user{.role = infernix::ChatRole::User};
                infernix::MessagePart part{.kind = infernix::MessagePartKind::Media};
                part.media.bytes = image;
                user.parts.push_back(std::move(part));
                user.parts.push_back({.text = question});
                input.messages.push_back(std::move(user));
                input.options.reasoning_effort = infernix::ReasoningEffort::Low;
                return input;
            };
            const auto run = [&](const std::vector<std::uint8_t>& image, bool reuse) {
                return engine.generate(engine.prepare(ask(image)), infernix::test::resume_probe_request(12, reuse));
            };
            const std::vector<std::uint8_t> a = picture(0), b = picture(70);
            const infernix::GenerationResult cold_a = run(a, false), cold_b = run(b, false);
            (void)run(a, true); // publishes
            const infernix::GenerationResult warm_b = run(b, true);
            const infernix::GenerationResult warm_a = run(a, true);
            std::cout << "images: another image reused " << warm_b.reused_prompt_tokens << ", the same image reused "
                      << warm_a.reused_prompt_tokens << " of " << warm_a.prompt.prompt_tokens << " prompt tokens\n";
            require(warm_b.generated_token_ids == cold_b.generated_token_ids, "images: another image's tokens differ from cold");
            require(warm_a.generated_token_ids == cold_a.generated_token_ids, "images: the same image's tokens differ from cold");
            require(warm_a.reused_prompt_tokens > warm_b.reused_prompt_tokens,
                    "images: the same image did not resume past the other's divergence");
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
            {
                infernix::Engine engine(options);
                const infernix::GenerationResult resumed =
                    engine.generate(engine.prepare_tokens(continuation), infernix::test::resume_probe_request(16, true));
                const infernix::GenerationResult cold =
                    engine.generate(engine.prepare_tokens(continuation), infernix::test::resume_probe_request(16, false));
                std::cout << "persist: the restarted Engine reused " << resumed.reused_prompt_tokens << " prompt tokens\n";
                require(resumed.reused_prompt_tokens >= 2048, "persist: the restarted Engine reused too little");
                require(resumed.generated_token_ids == cold.generated_token_ids, "persist: the tokens differ from cold");
                engine.stop(); // its stop saves the file again: removed after it
            }
            std::filesystem::remove(file);
            require(!std::filesystem::exists(file), "persist: the test's cache file stays in the temp directory");
        }
        // 11. The SSD tier.
        ssd_tier(artifact);
        std::cout << "PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
