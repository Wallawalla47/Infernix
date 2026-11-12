// Qwen4Exp prefix cache on the real artifact (design §19.3.1). Skips unless INFERNIX_QWEN4_ARTIFACT
// names a Qwen4Exp artifact (INFERNIX_QWEN4_NGRAM: its n-gram volume, default <artifact>.ngram).
// INFERNIX_QWEN4_KV selects the KV storage (bf16, int8, fp8, nvfp4, k8v4, vq2, k4v2; default int8);
// INFERNIX_QWEN4_EXPERT_A4 (any value) runs NVIDIA FP4-Local's experts with 4-bit activations.
// One Engine per drafter mode serves every scenario, so the 64.5 GiB model loads once per mode:
//
//   tap resume   (E3) A request resumes from another's flexible tap at a prefill-chunk boundary; its
//                greedy output equals a cold run of the same prompt (allow_prefix_reuse = false),
//                which has the same call boundaries.
//   host blocks  (E3 through the Host tier) A pressure prompt evicts the cached blocks; the next
//                resume restores them from the slab pool and still equals the cold run.
//   endpoint     (E1) A second turn resumes from the first turn's endpoint left in its lane; after
//                pressure evicts everything Device-side, the same second turn resumes through a
//                Host image restore and must generate the same tokens.
//   X5           Chat turn 2 without turn 1's reasoning block resumes at least at turn 1's generation
//                opener tap.
//   X6           Sessions sharing only a system block resume at its structural tap (an exact split).
//   X3           A repeat of a chat prompt resumes at its generation opener (an exact off-grid tap, taken
//                in place by the capturing run's pair call) and generates what the capturing run generated.
//   X12          (plain mode) Engine 1 serves two chat turns and stops, saving its Host tier to a file;
//                Engine 2 loads the file and resumes turn 2 from it, generating what Engine 1 did.
//                Between the turns, Engine 1's periodic save writes the file once and, with the Host
//                tier unchanged, not again.
//   one context  (MTP mode, own Engine with kv_capacity = max_context) A recomputed cached prefix
//                replaces the cached copies instead of pinning them beside its pages: the request
//                completes, equals its cold run, and the Engine keeps serving.
//   burst        (MTP mode, own Engines, default route, INT8 and FP8 KV) Four rounds of four identical chat
//                prompts at C = 4: every request reaches its limit and the Engine keeps serving.
//
//   next turn    (own Engines, default and reproducible routes) Engine::prefill_next_turn prefills a reply's
//                history while idle: turn 2 resumes at its end (reproducible: equals an uncached run), and a
//                request arriving during a long pass makes it yield within about one call.
//
//   SSD tier     (own Engine, hybrid-prefix-cache-spec §5.7) A 300 MiB Host tier with a 4 GiB SSD tier:
//                after pressure prompts the tap resume and the endpoint resume read their sources back
//                from the SSD tier and equal the cold run and the lane-resident resume. With a prefix
//                cache file the SSD tier is kept (§5.5): after a restart turn 3 resumes at turn 2's
//                endpoint, which only the SSD tier held, and equals the lane-resident resume.
//   SSD scale    (--ssd-scale only: timing) A 100K-token conversation resumed from the Host tier and, once
//                pushed out of it, from the SSD tier (INFERNIX_TEST_SSD_DIR): times to first token.
//
//   infernix_qwen4_exp_prefix_cache_real_test [--mtp] [--plain] [--next-turn] [--ssd] [--ssd-scale]
//   (default: all but --ssd-scale)

#include "test_temp_path.h"
#include "kv_cache_storage.h"
#include "engine_host.h"
#include "infernix/engine.h"

#include <algorithm>
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
#include <tuple>
#include <vector>

namespace {

constexpr std::uint32_t kChunk = 512;

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

std::string first_difference(const std::vector<infernix::TokenId>& a, const std::vector<infernix::TokenId>& b) {
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (a[i] != b[i]) { return "first difference at " + std::to_string(i); }
    }
    return a.size() == b.size() ? "identical" : "lengths differ";
}

infernix::EngineOptions base_options(const char* artifact, const char* ngram) {
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    if (ngram != nullptr) { options.ngram_volume_path = ngram; }
    options.max_context          = 4096;
    // 72 pages (+1 copy-on-write page): a 3,900-token pressure prompt evicts nearly every block.
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(4608);
    options.prefill_chunk        = kChunk;
    const char* kv               = std::getenv("INFERNIX_QWEN4_KV");
    options.kv_cache             = kv != nullptr ? infernix::test::parse_kv_cache_storage(kv)
                                                 : infernix::KvCacheStorage::Int8Group64;
    options.expert_a4_activations = std::getenv("INFERNIX_QWEN4_EXPERT_A4") != nullptr;
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    options.context_cache.host_capacity_bytes = 2ULL << 30;
    // Resumes equal cold runs on the exact route (model-quality.md §6.1); the default route's
    // qualified paths make a column's bits depend on its call. Off-grid and endpoint resumes are the
    // free-cache-resume path.
    options.fast_paths           = false;
    options.fast_path_candidates = {infernix::FastPath::FreeCacheResume};
    return options;
}

std::string rules_text() {
    std::string rules;
    for (int r = 1; r <= 110; ++r) {
        rules += "Rule " + std::to_string(r) + ": items of colour " + std::to_string(r % 9) + " go to shelf " +
                 std::to_string((r * 7) % 13) + ".\n";
    }
    return rules;
}

// User turns longer than kMinimumTapSeparation (64 tokens): an opener closer to the structural tap
// joins its cluster, and only the earlier tap is kept.
std::string user_context() {
    std::string context;
    for (int i = 0; i < 4; ++i) {
        context += "Today a delivery of mixed items arrived at the warehouse and I need to plan where each "
                   "box goes before the afternoon shift starts. ";
    }
    return context;
}

infernix::ChatMessage message(infernix::ChatRole role, std::string text) {
    infernix::ChatMessage m;
    m.role = role;
    m.parts.push_back(infernix::MessagePart{.kind = infernix::MessagePartKind::Text, .text = std::move(text)});
    return m;
}

infernix::GenerationResult chat_on(infernix::Engine& engine, std::vector<infernix::ChatMessage> messages,
                                 std::uint32_t outputs) {
    infernix::PromptInput input;
    input.messages                = std::move(messages);
    input.options.enable_thinking = false;
    infernix::RequestOptions request      = greedy(outputs, true);
    request.stop.include_model_defaults = true;
    return engine.generate(engine.prepare(std::move(input)), request);
}

int run_mode(const char* artifact, const char* ngram, bool mtp) {
    std::printf("== %s\n", mtp ? "MTP drafter (--spec mtp, 3 drafts)" : "plain decode");
    infernix::EngineOptions options = base_options(artifact, ngram);
    if (mtp) {
        options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    }
    infernix::Engine engine(std::move(options), infernix::test::engine_host());
    int failures = 0;

    // tap resume: A leaves a flexible tap at the start of its final call (1024); B shares 1,400
    // tokens, so it resumes at 1024 and prefills 1024..1536..1700 as a cold run does.
    const std::vector<infernix::TokenId> a = synthetic_tokens(1500, 11);
    std::vector<infernix::TokenId> b(a.begin(), a.begin() + 1400);
    const auto suffix = synthetic_tokens(300, 12);
    b.insert(b.end(), suffix.begin(), suffix.end());
    const Run cold_b = generate(engine, b, 48, false);
    (void)generate(engine, a, 16, true);
    const Run warm_b = generate(engine, b, 48, true);
    failures += check(warm_b.reused == 1024, "tap resume reuses 1024 tokens (got " + std::to_string(warm_b.reused) + ")");
    failures += check(warm_b.tokens == cold_b.tokens, "tap resume equals the cold run (" +
                                                          first_difference(warm_b.tokens, cold_b.tokens) + ")");

    // host blocks: pressure evicts the Device copies; B now resumes from its own prompt-tail tap
    // (1536) through Host block restores.
    const std::vector<infernix::TokenId> pressure = synthetic_tokens(3900, 13);
    (void)generate(engine, pressure, 4, true);
    const Run host_b = generate(engine, b, 48, true);
    failures += check(host_b.reused >= 1024, "Host resume reuses " + std::to_string(host_b.reused) + " tokens");
    failures += check(host_b.tokens == cold_b.tokens, "Host block restore equals the cold run (" +
                                                          first_difference(host_b.tokens, cold_b.tokens) + ")");

    // endpoint: turn 2 echoes turn 1 and its output; the lane still holds turn 1's endpoint.
    const std::vector<infernix::TokenId> t1 = synthetic_tokens(700, 14);
    const Run turn1 = generate(engine, t1, 96, true);
    std::vector<infernix::TokenId> t2 = t1;
    t2.insert(t2.end(), turn1.tokens.begin(), turn1.tokens.end());
    const auto more = synthetic_tokens(64, 15);
    t2.insert(t2.end(), more.begin(), more.end());
    const Run resident = generate(engine, t2, 48, true);
    const std::uint32_t endpoint = static_cast<std::uint32_t>(t1.size() + turn1.tokens.size() - 1U);
    failures += check(resident.reused == endpoint, "turn 2 resumes at the endpoint " + std::to_string(endpoint) +
                                                       " (got " + std::to_string(resident.reused) + ")");
    // Re-create turn 1's endpoint, evict, then resume turn 2 through the Host image.
    (void)generate(engine, t1, 96, true);
    (void)generate(engine, pressure, 4, true);
    const Run restored = generate(engine, t2, 48, true);
    failures += check(restored.reused == endpoint, "turn 2 resumes at the endpoint after eviction (got " +
                                                       std::to_string(restored.reused) + ")");
    failures += check(restored.tokens == resident.tokens, "Host-restored endpoint equals the lane-resident resume (" +
                                                              first_difference(restored.tokens, resident.tokens) + ")");

    // Chat prompts carry the structural (end of the system block) and generation-opener taps.
    const auto chat = [&](std::vector<infernix::ChatMessage> messages, std::uint32_t outputs) {
        return chat_on(engine, std::move(messages), outputs);
    };
    const infernix::ChatMessage system = message(infernix::ChatRole::System, rules_text());
    const std::string context        = user_context();
    const auto user = [&](const std::string& question) {
        return message(infernix::ChatRole::User, context + question);
    };

    // X5 chat no-echo: turn 2 re-renders turn 1's reply without its reasoning block; it resumes at
    // least at turn 1's generation opener (a few tokens before turn 1's prompt end).
    const infernix::GenerationResult s1 =
        chat({system, user("Which shelf takes colour 4?")}, 32);
    const std::uint32_t n1 = s1.prompt.prompt_tokens;
    const infernix::GenerationResult x5 =
        chat({system, user("Which shelf takes colour 4?"), message(infernix::ChatRole::Assistant, s1.content),
              message(infernix::ChatRole::User, "And colour 5?")},
             32);
    std::printf("        X5: turn 1 prompt %u, turn 2 prompt %u reused %u\n", n1, x5.prompt.prompt_tokens,
                x5.reused_prompt_tokens);
    failures += check(x5.reused_prompt_tokens + 16U >= n1, "X5 turn 2 reuses at least turn 1's opener (reused " +
                                                               std::to_string(x5.reused_prompt_tokens) + ", turn 1 prompt " +
                                                               std::to_string(n1) + ")");

    // X6 shared preamble: sessions 2 and 3 share only the system block; they resume at its
    // structural tap, an exact split the first session made.
    const infernix::GenerationResult s2 =
        chat({system, user("Summarise rule 12 in one sentence.")}, 32);
    const infernix::GenerationResult s3 = chat({system, user("How many rules send items to shelf 0?")}, 32);
    std::printf("        X6: sessions 2 and 3 reused %u and %u of %u and %u\n", s2.reused_prompt_tokens,
                s3.reused_prompt_tokens, s2.prompt.prompt_tokens, s3.prompt.prompt_tokens);
    failures += check(s2.reused_prompt_tokens >= 1024 && s2.reused_prompt_tokens + 8U < s2.prompt.prompt_tokens &&
                          s3.reused_prompt_tokens == s2.reused_prompt_tokens,
                      "X6 sessions 2 and 3 resume at the shared structural frontier");

    // X3 split equivalence at an exact off-grid tap: session 2 again resumes at its own generation
    // opener. The run that captured it took the tap between the two chunks' mixers of a pair call (design
    // §19.3.56) and ran the tail's routed experts with the head's; the resume runs the tail as a call of its own.
    // With these exact kernels the two agree: the pair call's router and shared expert run per chunk, and the
    // routed experts act column by column.
    const infernix::GenerationResult again = chat({system, user("Summarise rule 12 in one sentence.")}, 32);
    failures += check(again.reused_prompt_tokens + 16U >= s2.prompt.prompt_tokens &&
                          again.reused_prompt_tokens < s2.prompt.prompt_tokens,
                      "X3 the repeat resumes at the opener (reused " + std::to_string(again.reused_prompt_tokens) + ")");
    failures += check(again.generated_token_ids == s2.generated_token_ids,
                      "X3 the opener resume equals the capturing run (" +
                          first_difference(again.generated_token_ids, s2.generated_token_ids) + ")");
    return failures;
}

// One context of KV (issue #1: --kv-capacity == --max-context): B shares A's first 2,000 tokens but
// no snapshot on that path, so it recomputes them from the root while the admission eviction leaves
// A's shallowest blocks cached. Each recomputed block must replace its unpinned cached copy: pinning
// the copy beside B's page fills the pool, and B's first decode page past its admission then has
// no source (the capacity contract error, formerly fatal for the Engine).
int run_one_context(const char* artifact, const char* ngram) {
    std::printf("== one context of KV (MTP drafter, kv_capacity = max_context)\n");
    infernix::EngineOptions options = base_options(artifact, ngram);
    options.kv_capacity               = infernix::KvCapacityPolicy::explicit_capacity(4096);
    options.speculative.backend       = infernix::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    infernix::Engine engine(std::move(options), infernix::test::engine_host());
    int failures = 0;
    const std::vector<infernix::TokenId> a = synthetic_tokens(3000, 21);
    std::vector<infernix::TokenId> b(a.begin(), a.begin() + 2000);
    const auto suffix = synthetic_tokens(1300, 22);
    b.insert(b.end(), suffix.begin(), suffix.end());
    const Run cold_b = generate(engine, b, 700, false);
    (void)generate(engine, a, 16, true);
    const infernix::RuntimeStats before = engine.runtime_stats();
    Run warm_b;
    try {
        warm_b = generate(engine, b, 700, true);
    } catch (const std::exception& error) {
        return check(false, std::string("B completes in one context of KV (") + error.what() + ")");
    }
    const infernix::RuntimeStats after = engine.runtime_stats();
    std::printf("        B reused %u tokens; blocks reattached %llu, duplicate %llu\n", warm_b.reused,
                static_cast<unsigned long long>(after.hybrid_blocks_reattached - before.hybrid_blocks_reattached),
                static_cast<unsigned long long>(after.hybrid_blocks_duplicate - before.hybrid_blocks_duplicate));
    failures += check(warm_b.reused < 2000 && after.hybrid_blocks_reattached > before.hybrid_blocks_reattached,
                      "B recomputes cached blocks and replaces their copies");
    failures += check(warm_b.tokens == cold_b.tokens,
                      "B equals the cold run (" + first_difference(warm_b.tokens, cold_b.tokens) + ")");
    const Run later = generate(engine, synthetic_tokens(200, 23), 8, true);
    failures += check(later.tokens.size() == 8, "the Engine keeps serving");
    return failures;
}

// X12: the Host tier saved at an Engine's stop and loaded by the next Engine.
int run_persistence(const char* artifact, const char* ngram) {
    std::printf("== persistence (X12)\n");
    const std::filesystem::path file = infernix::test::unique_temp_path("infernix_qwen4_exp_x12.cache");
    std::error_code ignored;
    std::filesystem::remove(file, ignored);
    const auto options = [&] {
        infernix::EngineOptions o                       = base_options(artifact, ngram);
        o.context_cache.hybrid.persistent_file     = file;
        o.context_cache.hybrid.persistent_identity = "x12";
        return o;
    };
    const infernix::ChatMessage system = message(infernix::ChatRole::System, rules_text());
    const std::string context        = user_context();
    const std::vector<infernix::ChatMessage> turn1{system, message(infernix::ChatRole::User, context + "Which shelf takes colour 2?")};
    int failures = 0;
    std::vector<infernix::ChatMessage> turn2 = turn1;
    infernix::GenerationResult before;
    std::filesystem::file_time_type periodic_write{};
    {
        // Engine 1 also saves periodically (persistent_save_interval): after turn 1 the file appears
        // while the Engine runs, through its .tmp, once the interval has passed after a change. A save
        // during the request may find the Host tier still filling and the next one rewrite it; once
        // the Host tier stops changing, the file holds the endpoint and is not rewritten.
        infernix::EngineOptions o                       = options();
        o.context_cache.hybrid.persistent_save_interval = std::chrono::seconds(1);
        infernix::Engine engine(o, infernix::test::engine_host());
        failures += check(!engine.load_summary().prefix_cache.restored, "Engine 1 starts without a saved file");
        failures += check(!std::filesystem::exists(file), "the periodic save starts without a file");
        const infernix::GenerationResult r1 = chat_on(engine, turn1, 32);
        std::filesystem::path tmp = file;
        tmp += ".tmp";
        bool saved = false;
        for (int i = 0; i < 300 && !saved; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            saved = std::filesystem::exists(file) && !std::filesystem::exists(tmp);
        }
        failures += check(saved, "the periodic save wrote the file while the Engine was idle");
        bool settled = false;
        for (int i = 0; i < 6 && saved && !settled; ++i) {
            const auto written = std::filesystem::last_write_time(file, ignored);
            std::this_thread::sleep_for(std::chrono::milliseconds(1600)); // more than one interval
            settled = std::filesystem::last_write_time(file, ignored) == written;
        }
        failures += check(settled, "an unchanged Host tier is not saved again");
        const std::uintmax_t periodic_size = saved ? std::filesystem::file_size(file, ignored) : 0;
        failures += check(periodic_size > (std::uintmax_t{64} << 20),
                          "the settled periodic file holds turn 1's snapshot (" + std::to_string(periodic_size) +
                              " bytes)");
        periodic_write = std::filesystem::last_write_time(file, ignored);
        turn2.push_back(message(infernix::ChatRole::Assistant, r1.content));
        turn2.push_back(message(infernix::ChatRole::User, context + "And colour 3?"));
        before = chat_on(engine, turn2, 32);
    } // the stop saves the Host tier
    std::error_code size_error;
    const std::uintmax_t saved_bytes = std::filesystem::file_size(file, size_error);
    failures += check(!size_error && saved_bytes > 0 && std::filesystem::last_write_time(file, ignored) != periodic_write,
                      "the stop saved the Host tier (" + std::to_string(size_error ? 0U : saved_bytes) + " bytes)");
    {
        infernix::Engine engine(options(), infernix::test::engine_host());
        const infernix::LoadSummary::PrefixCacheRestore restore = engine.load_summary().prefix_cache;
        std::printf("        restored %llu blocks and %llu snapshots (%s)\n",
                    static_cast<unsigned long long>(restore.blocks), static_cast<unsigned long long>(restore.snapshots),
                    restore.message.c_str());
        failures += check(restore.restored && restore.snapshots > 0, "Engine 2 restores the saved snapshots");
        // Turn 2 again: Engine 1 left a snapshot at its own generation opener, so the restarted
        // Engine resumes there and repeats Engine 1's computation of turn 2.
        const infernix::GenerationResult after = chat_on(engine, turn2, 32);
        failures += check(after.reused_prompt_tokens + 16U >= before.prompt.prompt_tokens,
                          "after the restart turn 2 resumes at its opener (reused " +
                              std::to_string(after.reused_prompt_tokens) + " of " +
                              std::to_string(after.prompt.prompt_tokens) + ")");
        failures += check(after.generated_token_ids == before.generated_token_ids,
                          "the restored resume equals Engine 1's turn 2 (" +
                              first_difference(after.generated_token_ids, before.generated_token_ids) + ")");
    }
    std::filesystem::remove(file, ignored);
    return failures;
}

// The identical burst (candidates §15 NN-08; NInfer #414, root cause by oldfencetried on 2026-10-25: the worker
// reserves resident permits before fresh admission, so an exact hit could pin a resident writer's partial tail
// page while that writer still held its permit, its next write became a copy-on-write beyond its reservation,
// and the exception failed the whole Engine). Four rounds of four copies of one chat prompt whose length is not a
// page multiple, submitted together at C = 4 with MTP 3 on the default route, with INT8 and FP8 KV: every request
// reaches its output limit and the Engine serves the prompt again afterwards.
int run_identical_burst(const char* artifact, const char* ngram) {
    constexpr std::uint32_t kOutputs = 64;
    int failures                     = 0;
    for (const auto kv : {infernix::KvCacheStorage::Int8Group64, infernix::KvCacheStorage::Fp8E4M3Row256}) {
        const char* label = kv == infernix::KvCacheStorage::Int8Group64 ? "int8" : "fp8";
        std::printf("== identical burst (%s KV, C = 4, MTP 3, default route)\n", label);
        infernix::EngineOptions options;
        options.artifact_path = artifact;
        if (ngram != nullptr) { options.ngram_volume_path = ngram; }
        options.max_context                       = 4096;
        options.kv_capacity                       = infernix::KvCapacityPolicy::explicit_capacity(4 * 4096);
        options.kv_cache                          = kv;
        options.max_concurrency                   = 4;
        options.max_pending_requests              = 4;
        options.context_cache.host_capacity_bytes = 1ULL << 30;
        options.speculative.backend               = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens          = 3;
        infernix::Engine engine(std::move(options), infernix::test::engine_host());
        infernix::PromptInput input;
        infernix::ChatMessage user;
        user.role = infernix::ChatRole::User;
        std::string text = rules_text() + "Which shelf takes colour 4, and why?";
        user.parts.push_back(infernix::MessagePart{.kind = infernix::MessagePartKind::Text, .text = text, .media = {}});
        input.messages.push_back(user);
        input.options.enable_thinking = false;
        std::uint32_t prompt_tokens   = engine.generate(engine.prepare(input), greedy(1, false)).prompt.prompt_tokens;
        if (prompt_tokens % 64 == 0) {
            input.messages.front().parts.front().text += " Thanks.";
            prompt_tokens = engine.generate(engine.prepare(input), greedy(1, false)).prompt.prompt_tokens;
        }
        failures += check(prompt_tokens % 64 != 0, "the prompt (" + std::to_string(prompt_tokens) +
                                                        " tokens) is not a page multiple");
        for (int round = 0; round < 4; ++round) {
            std::vector<infernix::GenerationHandle> burst;
            for (int i = 0; i < 4; ++i) { burst.push_back(engine.submit(engine.prepare(input), greedy(kOutputs, true))); }
            int complete = 0;
            std::uint32_t reused = 0;
            for (auto& handle : burst) {
                const infernix::GenerationResult result = handle.wait();
                complete += result.finish_reason == infernix::FinishReason::OutputLimit &&
                                    result.generated_token_ids.size() == kOutputs
                                ? 1
                                : 0;
                reused = std::max(reused, result.reused_prompt_tokens);
            }
            failures += check(complete == 4, "round " + std::to_string(round) + ": " + std::to_string(complete) +
                                                 " of 4 identical requests reached their limit (most reused " +
                                                 std::to_string(reused) + ")");
        }
        const infernix::GenerationResult after = engine.generate(engine.prepare(input), greedy(kOutputs, true));
        failures += check(after.generated_token_ids.size() == kOutputs, "the Engine serves the prompt after the burst");
    }
    return failures;
}

// Next-turn prefill (hybrid-prefix-cache-spec §7.9), plain decode, one Engine per route. With
// preserve_thinking false a reply renders without its thinking block once a user turn follows, so the
// next request diverges at the reply's opener. Engine::prefill_next_turn prefills the history while
// the Engine is idle:
// - turn 2 resumes at the history's end (the reproducible route: the 16-token grid position at or
//   before it) and, on the reproducible route, generates what an uncached run generates; the default
//   route's qualified paths make bits depend on call boundaries, so there the comparison is reported,
//   not required;
// - a request arriving while a long pass runs is admitted within about one call: the pass yields at
//   its next unit boundary. Its layer walk keeps nothing past the span it abandons (design §19.3.8), so
//   the reuse the request finds is reported, not required.
int run_next_turn(const char* artifact, const char* ngram, bool reproducible) {
    std::printf("== next-turn prefill (%s route)\n", reproducible ? "reproducible" : "default");
    if (reproducible) {
        // The reproducible route refuses the pass: a turn resumed from its grid snapshot inside a cold run's call
        // differed from an uncached run (2026-10-27: first difference at token 16).
        infernix::EngineOptions refused = base_options(artifact, ngram);
        refused.fast_path_candidates.clear();
        refused.context_cache.hybrid.next_turn_prefill_quiet = std::chrono::milliseconds{20};
        std::string what;
        try {
            infernix::Engine engine(std::move(refused), infernix::test::engine_host());
        } catch (const std::exception& error) { what = error.what(); }
        return check(what.find("next-turn prefill") != std::string::npos,
                     "the reproducible route refuses the next-turn prefill (" + what + ")");
    }
    infernix::EngineOptions options = base_options(artifact, ngram);
    options.max_context              = 16384;
    options.kv_capacity              = infernix::KvCapacityPolicy::explicit_capacity(16384 + 4096);
    options.fast_paths               = !reproducible;
    options.fast_path_candidates.clear();
    options.context_cache.hybrid.next_turn_prefill_quiet = std::chrono::milliseconds{20};
    const std::uint32_t grid = reproducible ? 16U : 1U;
    infernix::Engine engine(std::move(options), infernix::test::engine_host());
    int failures = 0;
    const auto chat = [&](std::vector<infernix::ChatMessage> messages, std::uint32_t outputs, bool reuse) {
        infernix::PromptInput input;
        input.messages                  = std::move(messages);
        input.options.enable_thinking   = false;
        input.options.preserve_thinking = false;
        infernix::RequestOptions request    = greedy(outputs, reuse);
        request.stop.include_model_defaults = true;
        return engine.generate(engine.prepare(std::move(input)), request);
    };
    const auto conversation = [](std::vector<infernix::ChatMessage> messages) {
        infernix::PromptInput input;
        input.messages                  = std::move(messages);
        input.options.enable_thinking   = false;
        input.options.preserve_thinking = false;
        return input;
    };
    const auto ended = [&](const infernix::RuntimeStats& before) {
        for (int i = 0; i < 12000; ++i) {
            const infernix::RuntimeStats now = engine.runtime_stats();
            if (now.next_turn_prefills_completed + now.next_turn_prefills_yielded + now.next_turn_prefills_skipped +
                    now.next_turn_prefills_dropped >
                before.next_turn_prefills_completed + before.next_turn_prefills_yielded +
                    before.next_turn_prefills_skipped + before.next_turn_prefills_dropped) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return false;
    };

    const std::vector<infernix::ChatMessage> turn1{
        message(infernix::ChatRole::System, rules_text()),
        message(infernix::ChatRole::User, user_context() + "Explain in detail, in at least 150 words, how to "
                                                           "plan the shelves for colours 1 to 4.")};
    const infernix::GenerationResult first = chat(turn1, 160, true);
    std::vector<infernix::ChatMessage> history = turn1;
    history.push_back(message(infernix::ChatRole::Assistant, first.content));
    std::vector<infernix::ChatMessage> turn2 = history;
    turn2.push_back(message(infernix::ChatRole::User, "Now say it in one sentence."));
    infernix::PromptInput rendered  = conversation(history);
    rendered.options.continuation   = infernix::PromptContinuationMode::NextTurnHistory;
    const std::uint32_t pass_tokens = engine.prepare(std::move(rendered)).summary().prompt_tokens;
    const std::uint32_t expected    = (pass_tokens - 1U) / grid * grid;
    const infernix::RuntimeStats idle = engine.runtime_stats();
    engine.prefill_next_turn(conversation(history));
    if (!ended(idle)) { return check(false, "the next-turn pass ended within two minutes"); }
    const infernix::RuntimeStats passed = engine.runtime_stats();
    const infernix::GenerationResult second = chat(turn2, 24, true);
    std::printf("        reply %zu tokens, history %u tokens; turn 2 reused %u of %u; the pass computed %llu tokens\n",
                first.generated_token_ids.size(), pass_tokens, second.reused_prompt_tokens, second.prompt.prompt_tokens,
                static_cast<unsigned long long>(passed.next_turn_prefill_tokens - idle.next_turn_prefill_tokens));
    failures += check(passed.next_turn_prefills_completed == idle.next_turn_prefills_completed + 1,
                      "the next-turn pass ran to its end");
    failures += check(second.reused_prompt_tokens >= expected && second.reused_prompt_tokens < second.prompt.prompt_tokens,
                      "turn 2 resumes at the pass's history end (expected at least " + std::to_string(expected) + ")");
    const infernix::GenerationResult uncached = chat(turn2, 24, false);
    if (reproducible) {
        failures += check(second.generated_token_ids == uncached.generated_token_ids,
                          "turn 2 after the pass equals an uncached run (" +
                              first_difference(second.generated_token_ids, uncached.generated_token_ids) + ")");
    } else {
        std::printf("        default route: turn 2 against an uncached run: %s\n",
                    first_difference(second.generated_token_ids, uncached.generated_token_ids).c_str());
    }

    // Yield: a long new conversation's pass, interrupted once it has computed something.
    std::string document;
    for (int paragraph = 0; paragraph < 400; ++paragraph) {
        document += "Section " + std::to_string(paragraph) + " records that the warehouse moved " +
                    std::to_string(paragraph * 37 % 101) + " crates of part " + std::to_string(paragraph % 17) +
                    " to bay " + std::to_string(paragraph % 9) + " during the night shift. ";
    }
    const std::vector<infernix::ChatMessage> long_history{
        message(infernix::ChatRole::System, "You summarise warehouse logs."),
        message(infernix::ChatRole::User, document + "Summarise the log."),
        message(infernix::ChatRole::Assistant, "Crates moved between bays every night.")};
    std::vector<infernix::ChatMessage> long_turn = long_history;
    long_turn.push_back(message(infernix::ChatRole::User, "Which bay received the most?"));
    const infernix::RuntimeStats before_long = engine.runtime_stats();
    engine.prefill_next_turn(conversation(long_history));
    bool started = false;
    for (int i = 0; i < 60000 && !started; ++i) {
        const infernix::RuntimeStats now = engine.runtime_stats();
        started = now.next_turn_prefills_started > before_long.next_turn_prefills_started && now.prefilling_requests != 0;
        if (!started) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    }
    const infernix::GenerationResult interrupted = chat(long_turn, 8, true);
    const infernix::RuntimeStats after_long      = engine.runtime_stats();
    std::printf("        the long pass computed %llu tokens before yielding; the request waited %.3f s and reused %u of %u\n",
                static_cast<unsigned long long>(after_long.next_turn_prefill_tokens - before_long.next_turn_prefill_tokens),
                interrupted.engine_timing.queue_wait_seconds, interrupted.reused_prompt_tokens,
                interrupted.prompt.prompt_tokens);
    failures += check(started && after_long.next_turn_prefills_yielded == before_long.next_turn_prefills_yielded + 1,
                      "the long pass yields to the request");
    failures += check(interrupted.engine_timing.queue_wait_seconds < 3.0,
                      "the request is admitted within about one call of the pass");
    if (reproducible) {
        const infernix::GenerationResult cold = chat(long_turn, 8, false);
        failures += check(interrupted.generated_token_ids == cold.generated_token_ids,
                          "the request after the yield equals an uncached run (" +
                              first_difference(interrupted.generated_token_ids, cold.generated_token_ids) + ")");
    }
    return failures;
}

} // namespace

// The SSD tier's directory: INFERNIX_TEST_SSD_DIR (a drive to measure), else a temporary one.
std::filesystem::path ssd_directory() {
    if (const char* dir = std::getenv("INFERNIX_TEST_SSD_DIR"); dir != nullptr && *dir != '\0') { return dir; }
    std::filesystem::path dir = infernix::test::unique_temp_path("qwen4_ssd_tier");
    std::filesystem::create_directories(dir);
    return dir;
}

void print_ssd(const infernix::RuntimeStats& s) {
    std::printf("        SSD tier: %.1f MB written (%llu images, %llu blocks), %.1f MB read in %llu promotions, "
                "%llu image and %llu block Host demotions, %llu SSD-only snapshots lost, %llu failures\n",
                static_cast<double>(s.hybrid_ssd_write_bytes) / 1e6,
                static_cast<unsigned long long>(s.hybrid_ssd_image_writes),
                static_cast<unsigned long long>(s.hybrid_ssd_block_writes),
                static_cast<double>(s.hybrid_ssd_read_bytes) / 1e6, static_cast<unsigned long long>(s.hybrid_ssd_promotions),
                static_cast<unsigned long long>(s.hybrid_ssd_host_image_demotions),
                static_cast<unsigned long long>(s.hybrid_ssd_host_block_demotions),
                static_cast<unsigned long long>(s.hybrid_ssd_snapshot_evictions),
                static_cast<unsigned long long>(s.hybrid_ssd_failures));
    std::printf("        Host tier: %llu snapshots evicted unwritten, %llu blocks lost with no copy, %llu dead reclaims\n",
                static_cast<unsigned long long>(s.hybrid_host_snapshot_evictions),
                static_cast<unsigned long long>(s.hybrid_unbacked_node_losses),
                static_cast<unsigned long long>(s.hybrid_host_dead_reclaims));
}

// SSD tier (hybrid-prefix-cache-spec §5.7): a Host tier of about two images keeps nothing of A once
// three pressure prompts have passed; the SSD tier keeps A's tap and turn 1's endpoint, and their
// resumes read them back (promotions) and equal the cold run and the lane-resident resume. With
// `drafter` (INFERNIX_QWEN4_DFLASH2) the same runs under DFlash2, whose ring the SSD copies leave out:
// a promotion zero-fills it, which may cost acceptance but never output.
int run_ssd_tier(const char* artifact, const char* ngram, const char* drafter = nullptr) {
    std::printf("== SSD tier (Host tier %s MiB, SSD tier 4 GiB%s)\n", drafter != nullptr ? "420" : "300",
                drafter != nullptr ? ", DFlash2" : "");
    const auto with_drafter = [&](infernix::EngineOptions options) {
        if (drafter != nullptr) {
            options.speculative.backend      = infernix::SpeculativeBackend::DFlash2;
            options.speculative.draft_tokens = 3;
            options.speculative.draft_model  = drafter;
        }
        return options;
    };
    // The reference turn 2, resumed at turn 1's endpoint in its lane, from an Engine of its own: in the
    // SSD Engine turn 2's endpoint must be turn 1's fresh one (a re-sent turn 1 republishes an endpoint
    // the first turn 2 superseded, and superseded snapshots are never written to the SSD).
    const std::vector<infernix::TokenId> t1 = synthetic_tokens(700, 14);
    Run turn1_reference, resident, resident3;
    std::vector<infernix::TokenId> t2, t3;
    {
        infernix::Engine reference(with_drafter(base_options(artifact, ngram)), infernix::test::engine_host());
        turn1_reference = generate(reference, t1, 96, true);
        t2              = t1;
        t2.insert(t2.end(), turn1_reference.tokens.begin(), turn1_reference.tokens.end());
        const auto more = synthetic_tokens(64, 15);
        t2.insert(t2.end(), more.begin(), more.end());
        resident = generate(reference, t2, 48, true);
        // Turn 3, which the restart below resumes at turn 2's endpoint from the kept SSD tier.
        t3 = t2;
        t3.insert(t3.end(), resident.tokens.begin(), resident.tokens.end());
        const auto third = synthetic_tokens(48, 16);
        t3.insert(t3.end(), third.begin(), third.end());
        resident3 = generate(reference, t3, 32, true);
    }
    const std::filesystem::path directory = ssd_directory();
    // A prefix cache file keeps the SSD tier across the restart at the end (spec §5.5, candidates YK-02).
    const std::filesystem::path file =
        infernix::test::unique_temp_path(drafter != nullptr ? "infernix_qwen4_exp_ssd_dflash2.cache" : "infernix_qwen4_exp_ssd.cache");
    std::error_code ignored;
    std::filesystem::remove(file, ignored);
    const auto ssd_options = [&] {
        infernix::EngineOptions options = with_drafter(base_options(artifact, ngram));
        // About two and a half images either way (a DFlash2 image also holds its ring: 158 against 117 MB).
        options.context_cache.host_capacity_bytes = (drafter != nullptr ? 420ULL : 300ULL) << 20;
        options.context_cache.ssd_capacity_bytes  = 4ULL << 30;
        // The write budget's level persists in the test directory: exactness and timing run without it.
        options.context_cache.ssd_write_bytes_per_hour = infernix::ContextCacheOptions::kSsdWritesUnlimited;
        options.context_cache.ssd_directory       = directory;
        options.context_cache.hybrid.persistent_file     = file;
        options.context_cache.hybrid.persistent_identity = "ssd";
        return options;
    };
    int failures = 0;
    Run restored;
    std::optional<infernix::Engine> first;
    first.emplace(ssd_options(), infernix::test::engine_host());
    infernix::Engine& engine = *first;
    const auto pressure = [&](std::uint32_t seed) {
        for (std::uint32_t k = 0; k < 3; ++k) { (void)generate(engine, synthetic_tokens(3900, seed + k), 4, true); }
    };

    // Tap resume through the SSD tier: as in run_mode, B resumes at A's tap at 1024.
    const std::vector<infernix::TokenId> a = synthetic_tokens(1500, 11);
    std::vector<infernix::TokenId> b(a.begin(), a.begin() + 1400);
    const auto suffix = synthetic_tokens(300, 12);
    b.insert(b.end(), suffix.begin(), suffix.end());
    const Run cold_b = generate(engine, b, 48, false);
    (void)generate(engine, a, 16, true);
    pressure(100);
    const infernix::RuntimeStats before = engine.runtime_stats();
    const Run ssd_b = generate(engine, b, 48, true);
    const infernix::RuntimeStats after = engine.runtime_stats();
    print_ssd(after);
    failures += check(after.hybrid_ssd_capacity_bytes != 0 && after.hybrid_ssd_write_bytes != 0,
                      "the SSD tier wrote back what the Host tier gave up");
    failures += check(after.hybrid_ssd_promotions > before.hybrid_ssd_promotions,
                      "B's source was read back from the SSD tier");
    failures += check(ssd_b.reused == 1024, "SSD resume reuses 1024 tokens (got " + std::to_string(ssd_b.reused) + ")");
    failures += check(ssd_b.tokens == cold_b.tokens,
                      "SSD tap resume equals the cold run (" + first_difference(ssd_b.tokens, cold_b.tokens) + ")");

    // Endpoint through the SSD tier: turn 2 after pressure equals turn 2 resumed in the lane.
    const Run turn1 = generate(engine, t1, 96, true);
    failures += check(turn1.tokens == turn1_reference.tokens, "turn 1 repeats the reference Engine's turn 1");
    const auto endpoint = static_cast<std::uint32_t>(t1.size() + turn1.tokens.size() - 1U);
    std::printf("        after turn 1:\n");
    print_ssd(engine.runtime_stats());
    pressure(200);
    const std::uint64_t promotions = engine.runtime_stats().hybrid_ssd_promotions;
    restored = generate(engine, t2, 48, true);
    print_ssd(engine.runtime_stats());
    failures += check(engine.runtime_stats().hybrid_ssd_promotions > promotions, "turn 2 was read back from the SSD tier");
    failures += check(restored.reused == endpoint, "turn 2 resumes at the endpoint " + std::to_string(endpoint) +
                                                       " from the SSD tier (got " + std::to_string(restored.reused) + ")");
    failures += check(restored.tokens == resident.tokens, "SSD-restored endpoint equals the lane-resident resume (" +
                                                              first_difference(restored.tokens, resident.tokens) + ")");
    failures += check(engine.runtime_stats().hybrid_ssd_failures == 0, "no SSD transfer failed");
    // Turn 2's endpoint leaves the Host tier for the SSD tier; the stop saves the file.
    pressure(300);
    std::printf("        before the restart:\n");
    print_ssd(engine.runtime_stats());
    first.reset(); // the stop saves the file

    // The restart: a new Engine on the same file and SSD directory takes the SSD copies back unread, and
    // turn 3 resumes at turn 2's endpoint from them.
    const auto endpoint2 = static_cast<std::uint32_t>(t2.size() + restored.tokens.size() - 1U);
    infernix::Engine restarted(ssd_options(), infernix::test::engine_host());
    const infernix::LoadSummary::PrefixCacheRestore restore = restarted.load_summary().prefix_cache;
    std::printf("        restart: restored %llu blocks and %llu snapshots, %llu and %llu of them on the SSD tier (%s%s)\n",
                static_cast<unsigned long long>(restore.blocks), static_cast<unsigned long long>(restore.snapshots),
                static_cast<unsigned long long>(restore.ssd_blocks), static_cast<unsigned long long>(restore.ssd_snapshots),
                restore.message.c_str(), restore.ssd_message.c_str());
    failures += check(restore.restored && restore.ssd_snapshots != 0 && restore.ssd_message.empty(),
                      "the restart takes back the SSD tier's snapshots");
    const Run third = generate(restarted, t3, 32, true);
    print_ssd(restarted.runtime_stats());
    failures += check(restarted.runtime_stats().hybrid_ssd_promotions != 0, "turn 3 was read back from the kept SSD tier");
    failures += check(third.reused == endpoint2, "turn 3 resumes at turn 2's endpoint " + std::to_string(endpoint2) +
                                                     " after the restart (got " + std::to_string(third.reused) + ")");
    failures += check(third.tokens == resident3.tokens, "turn 3 after the restart equals the lane-resident resume (" +
                                                     first_difference(third.tokens, resident3.tokens) + ")");
    failures += check(restarted.runtime_stats().hybrid_ssd_failures == 0, "no SSD transfer failed after the restart");
    return failures;
}

// The SSD tier at scale (timing, not run by default: --ssd-scale): a 100K-token conversation resumes at
// a snapshot near its end from the Host tier, and a second conversation of the same shape resumes at
// the same position from the SSD tier after two more pushed it out of a 4 GiB Host tier
// (INFERNIX_TEST_SSD_DIR picks the drive). Prints the time to first token of a cold prefill, of the first
// resume (at the chunk-grid tap, which prefills 1,760 more tokens), of the Host and the SSD resume, and
// the SSD read rate.
int run_ssd_scale(const char* artifact, const char* ngram) {
    constexpr std::uint32_t kContext = 100'000;
    std::printf("== SSD tier at scale (%u-token conversations, Host tier 4 GiB, SSD tier 16 GiB)\n", kContext);
    const std::filesystem::path directory = ssd_directory();
    std::printf("        SSD directory %s\n", directory.string().c_str());
    infernix::EngineOptions options = base_options(artifact, ngram);
    options.max_context             = kContext + 1024;
    options.kv_capacity             = infernix::KvCapacityPolicy::explicit_capacity(kContext + 4096);
    options.prefill_chunk           = 16384;
    options.fast_paths              = true; // the default route, as served
    options.fast_path_candidates.clear();
    options.context_cache.host_capacity_bytes = 4ULL << 30;
    options.context_cache.ssd_capacity_bytes  = 16ULL << 30;
    // The write budget's level persists in the test directory: exactness and timing run without it.
    options.context_cache.ssd_write_bytes_per_hour = infernix::ContextCacheOptions::kSsdWritesUnlimited;
    options.context_cache.ssd_directory       = directory;
    infernix::Engine engine(std::move(options), infernix::test::engine_host());
    const auto timed = [&](const std::vector<infernix::TokenId>& prompt, bool reuse) {
        const infernix::GenerationResult r = engine.generate(engine.prepare_tokens(prompt), greedy(16, reuse));
        const double ttft = r.first_output_timing ? r.first_output_timing->elapsed_seconds : -1.0;
        return std::make_tuple(Run{r.generated_token_ids, r.reused_prompt_tokens}, ttft);
    };
    const std::vector<infernix::TokenId> a = synthetic_tokens(kContext, 31);
    std::vector<infernix::TokenId> a1      = a;
    const auto next                        = synthetic_tokens(64, 32);
    a1.insert(a1.end(), next.begin(), next.end());
    const auto host_resume = [&](const std::vector<infernix::TokenId>& prompt, const char* label) {
        const infernix::RuntimeStats before = engine.runtime_stats();
        auto result                         = timed(prompt, true);
        const infernix::RuntimeStats after  = engine.runtime_stats();
        std::printf("        %s: %.2f GB restored from the Host tier, %llu SSD promotions\n", label,
                    static_cast<double>(after.hybrid_host_restore_bytes - before.hybrid_host_restore_bytes) / 1e9,
                    static_cast<unsigned long long>(after.hybrid_ssd_promotions - before.hybrid_ssd_promotions));
        return std::make_tuple(std::get<0>(result), std::get<1>(result),
                               after.hybrid_ssd_promotions - before.hybrid_ssd_promotions);
    };
    const auto [first, cold_ttft] = timed(a, true);
    (void)first;
    (void)timed(synthetic_tokens(kContext, 33), true); // its blocks push A's off the Device
    // A1 resumes at A's last chunk-grid tap and leaves its own snapshot S near its end; B2 pushes A1's
    // pages off the Device and A1 again resumes at S from the Host tier. A resume supersedes the snapshot
    // it resumed from, so the SSD resume at the same point runs on a second conversation of the same
    // shape: E, F, E1 (its snapshot at the same position), C and D pushing it off the Host, E1 again.
    const auto [tap_run, tap_ttft, tap_promotions] = host_resume(a1, "Host resume at A's tap");
    (void)tap_promotions;
    (void)timed(synthetic_tokens(kContext, 36), true);
    const auto [host_run, host_ttft, host_promotions] = host_resume(a1, "Host resume at A1's snapshot");
    std::vector<infernix::TokenId> e1 = synthetic_tokens(kContext, 37);
    const auto e_next                 = synthetic_tokens(64, 38);
    (void)timed(e1, true);
    e1.insert(e1.end(), e_next.begin(), e_next.end());
    (void)timed(synthetic_tokens(kContext, 39), true);
    (void)host_resume(e1, "Host resume at E's tap");
    (void)timed(synthetic_tokens(kContext, 34), true);
    (void)timed(synthetic_tokens(kContext, 35), true); // and these E's off the Host
    const infernix::RuntimeStats before = engine.runtime_stats();
    const auto [ssd_run, ssd_ttft] = timed(e1, true);
    const infernix::RuntimeStats after = engine.runtime_stats();
    print_ssd(after);
    const double read_gb = static_cast<double>(after.hybrid_ssd_read_bytes - before.hybrid_ssd_read_bytes) / 1e9;
    std::printf("        TTFT: cold %.2f s (%u tokens) | Host resume at A's tap %.2f s (reused %u) | Host resume %.2f s "
                "(reused %u) | SSD resume %.2f s (reused %u, %.2f GB read from the SSD tier, %.2f GB/s last promotion)\n",
                cold_ttft, kContext, tap_ttft, tap_run.reused, host_ttft, host_run.reused, ssd_ttft, ssd_run.reused,
                read_gb, after.hybrid_ssd_read_bytes_per_second / 1e9);
    // The two resumes restore snapshots at the same position of conversations of the same shape and
    // prefill the same suffix length (the --ssd scenario checks exactness on the exact route).
    int failures = 0;
    failures += check(host_promotions == 0, "the Host resume read nothing from the SSD tier");
    failures += check(after.hybrid_ssd_promotions > before.hybrid_ssd_promotions, "the SSD resume read from the SSD tier");
    failures += check(ssd_run.reused == host_run.reused, "the SSD resume reuses what the Host resume did");
    failures += check(after.hybrid_ssd_failures == 0, "no SSD transfer failed");
    return failures;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr) {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT\n");
        return 77;
    }
    const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM");
    bool plain = true, mtp = true, next_turn = true, ssd = true, ssd_scale = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mtp") { plain = false; next_turn = false; ssd = false; }
        if (arg == "--plain") { mtp = false; next_turn = false; ssd = false; }
        if (arg == "--next-turn") { plain = false; mtp = false; ssd = false; }
        if (arg == "--ssd") { plain = false; mtp = false; next_turn = false; }
        if (arg == "--ssd-scale") { plain = false; mtp = false; next_turn = false; ssd = false; ssd_scale = true; }
    }
    try {
        int failures = 0;
        if (ssd) { failures += run_ssd_tier(artifact, ngram); }
        if (const char* drafter = std::getenv("INFERNIX_QWEN4_DFLASH2"); ssd && drafter != nullptr) {
            failures += run_ssd_tier(artifact, ngram, drafter);
        }
        if (ssd_scale) { failures += run_ssd_scale(artifact, ngram); }
        if (plain) { failures += run_mode(artifact, ngram, false); }
        if (mtp) { failures += run_mode(artifact, ngram, true); }
        if (mtp) { failures += run_one_context(artifact, ngram); }
        if (mtp) { failures += run_identical_burst(artifact, ngram); }
        if (plain) { failures += run_persistence(artifact, ngram); }
        if (next_turn) {
            failures += run_next_turn(artifact, ngram, false);
            failures += run_next_turn(artifact, ngram, true);
        }
        std::printf(failures == 0 ? "qwen4_exp prefix cache checks passed\n" : "FAIL: %d checks failed\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
