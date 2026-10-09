// Qwen4Exp prefix cache on the real artifact (design §19.3.1). Skips unless INFERNIX_QWEN4_ARTIFACT
// names a Qwen4Exp artifact (INFERNIX_QWEN4_NGRAM: its n-gram volume, default <artifact>.ngram).
// INFERNIX_QWEN4_KV selects the KV storage (bf16, int8, fp8, nvfp4, k8v4, vq2, k4v2; default int8).
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
//   X3           A repeat of a chat prompt resumes at its generation opener (an exact off-grid tap) and
//                generates what the capturing run generated.
//   X12          (plain mode) Engine 1 serves two chat turns and stops, saving its Host tier to a file;
//                Engine 2 loads the file and resumes turn 2 from it, generating what Engine 1 did.
//   one context  (MTP mode, own Engine with kv_capacity = max_context) A recomputed cached prefix
//                replaces the cached copies instead of pinning them beside its pages: the request
//                completes, equals its cold run, and the Engine keeps serving.
//
//   infernix_qwen4_exp_prefix_cache_real_test [--mtp] [--plain]   (default: both modes)

#include "kv_cache_storage.h"
#include "infernix/engine.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
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
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    options.context_cache.mode                    = infernix::ContextCacheMode::Hybrid;
    options.context_cache.host_capacity_bytes = 2ULL << 30;
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
    infernix::Engine engine(std::move(options));
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
    // opener; the run that captured it split its prefill there, so the resume repeats its computation.
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
    infernix::Engine engine(std::move(options));
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
    const std::filesystem::path file = std::filesystem::temp_directory_path() / "infernix_qwen4_exp_x12.cache";
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
    {
        infernix::Engine engine(options());
        failures += check(!engine.load_summary().prefix_cache.restored, "Engine 1 starts without a saved file");
        const infernix::GenerationResult r1 = chat_on(engine, turn1, 32);
        turn2.push_back(message(infernix::ChatRole::Assistant, r1.content));
        turn2.push_back(message(infernix::ChatRole::User, context + "And colour 3?"));
        before = chat_on(engine, turn2, 32);
    } // the stop saves the Host tier
    std::error_code size_error;
    const std::uintmax_t saved_bytes = std::filesystem::file_size(file, size_error);
    failures += check(!size_error && saved_bytes > 0, "the stop saved the Host tier (" +
                                                          std::to_string(size_error ? 0U : saved_bytes) + " bytes)");
    {
        infernix::Engine engine(options());
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

    // The periodic save (persistent_save_interval): the file appears while the Engine runs, through
    // its .tmp, once the interval has passed after a change. A save during the request may find the
    // Host tier still filling and the next one rewrite it; once the Host tier stops changing, the
    // file holds the endpoint and is not rewritten.
    {
        infernix::EngineOptions o                       = options();
        o.context_cache.hybrid.persistent_save_interval = std::chrono::seconds(1);
        infernix::Engine engine(o);
        failures += check(!std::filesystem::exists(file), "the periodic test starts without a file");
        (void)chat_on(engine, turn1, 16);
        std::filesystem::path tmp = file;
        tmp += ".tmp";
        bool saved = false;
        for (int i = 0; i < 100 && !saved; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            saved = std::filesystem::exists(file) && !std::filesystem::exists(tmp);
        }
        failures += check(saved, "the periodic save wrote the file while the Engine was idle");
        bool settled = false;
        for (int i = 0; i < 4 && saved && !settled; ++i) {
            const auto written = std::filesystem::last_write_time(file, ignored);
            std::this_thread::sleep_for(std::chrono::milliseconds(2500));
            settled = std::filesystem::last_write_time(file, ignored) == written;
        }
        failures += check(settled, "an unchanged Host tier is not saved again");
        const std::uintmax_t size = saved ? std::filesystem::file_size(file, ignored) : 0;
        failures += check(size > (std::uintmax_t{64} << 20),
                          "the settled periodic file holds turn 1's snapshot (" + std::to_string(size) + " bytes)");
    }
    std::filesystem::remove(file, ignored);
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr) {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT\n");
        return 77;
    }
    const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM");
    bool plain = true, mtp = true;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mtp") { plain = false; }
        if (arg == "--plain") { mtp = false; }
    }
    try {
        int failures = 0;
        if (plain) { failures += run_mode(artifact, ngram, false); }
        if (mtp) { failures += run_mode(artifact, ngram, true); }
        if (mtp) { failures += run_one_context(artifact, ngram); }
        if (plain) { failures += run_persistence(artifact, ngram); }
        std::printf(failures == 0 ? "qwen4_exp prefix cache checks passed\n" : "FAIL: %d checks failed\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
