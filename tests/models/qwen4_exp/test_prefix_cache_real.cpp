// Qwen4Exp prefix cache on the real artifact (design §19.3.1). Skips unless NINFER_QWEN4_ARTIFACT
// names a Qwen4Exp artifact (NINFER_QWEN4_NGRAM: its n-gram volume, default <artifact>.ngram).
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
//
//   ninfer_qwen4_exp_prefix_cache_real_test [--mtp] [--plain]   (default: both modes)

#include "ninfer/engine.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kChunk = 512;

std::vector<ninfer::TokenId> synthetic_tokens(std::size_t count, std::uint32_t seed) {
    std::vector<ninfer::TokenId> tokens;
    tokens.reserve(count);
    std::uint32_t state = seed * 2654435761U + 1U;
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 1664525U + 1013904223U;
        tokens.push_back(static_cast<ninfer::TokenId>(1000U + (state >> 8U) % 30000U));
    }
    return tokens;
}

ninfer::RequestOptions greedy(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

struct Run {
    std::vector<ninfer::TokenId> tokens;
    std::uint32_t reused = 0;
};

Run generate(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt, std::uint32_t outputs, bool reuse) {
    const ninfer::GenerationResult result = engine.generate(engine.prepare_tokens(prompt), greedy(outputs, reuse));
    return Run{result.generated_token_ids, result.reused_prompt_tokens};
}

int check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    return ok ? 0 : 1;
}

std::string first_difference(const std::vector<ninfer::TokenId>& a, const std::vector<ninfer::TokenId>& b) {
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (a[i] != b[i]) { return "first difference at " + std::to_string(i); }
    }
    return a.size() == b.size() ? "identical" : "lengths differ";
}

int run_mode(const char* artifact, const char* ngram, bool mtp) {
    std::printf("== %s\n", mtp ? "MTP drafter (--spec mtp, 3 drafts)" : "plain decode");
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    if (ngram != nullptr) { options.ngram_volume_path = ngram; }
    options.max_context          = 4096;
    // 72 pages (+1 copy-on-write page): a 3,900-token pressure prompt evicts nearly every block.
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(4608);
    options.prefill_chunk        = kChunk;
    options.kv_cache             = ninfer::KvCacheStorage::Int8Group64;
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    if (mtp) {
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    }
    options.context_cache.mode                    = ninfer::ContextCacheMode::Hybrid;
    options.context_cache.host_cache_budget_bytes = 2ULL << 30;
    ninfer::Engine engine(std::move(options));
    int failures = 0;

    // tap resume: A leaves a flexible tap at the start of its final call (1024); B shares 1,400
    // tokens, so it resumes at 1024 and prefills 1024..1536..1700 as a cold run does.
    const std::vector<ninfer::TokenId> a = synthetic_tokens(1500, 11);
    std::vector<ninfer::TokenId> b(a.begin(), a.begin() + 1400);
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
    const std::vector<ninfer::TokenId> pressure = synthetic_tokens(3900, 13);
    (void)generate(engine, pressure, 4, true);
    const Run host_b = generate(engine, b, 48, true);
    failures += check(host_b.reused >= 1024, "Host resume reuses " + std::to_string(host_b.reused) + " tokens");
    failures += check(host_b.tokens == cold_b.tokens, "Host block restore equals the cold run (" +
                                                          first_difference(host_b.tokens, cold_b.tokens) + ")");

    // endpoint: turn 2 echoes turn 1 and its output; the lane still holds turn 1's endpoint.
    const std::vector<ninfer::TokenId> t1 = synthetic_tokens(700, 14);
    const Run turn1 = generate(engine, t1, 96, true);
    std::vector<ninfer::TokenId> t2 = t1;
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
    std::string rules;
    for (int r = 1; r <= 110; ++r) {
        rules += "Rule " + std::to_string(r) + ": items of colour " + std::to_string(r % 9) + " go to shelf " +
                 std::to_string((r * 7) % 13) + ".\n";
    }
    const auto chat = [&](std::vector<ninfer::ChatMessage> messages, std::uint32_t outputs) {
        ninfer::PromptInput input;
        input.messages                = std::move(messages);
        input.options.enable_thinking = false;
        ninfer::RequestOptions request      = greedy(outputs, true);
        request.stop.include_model_defaults = true;
        return engine.generate(engine.prepare(std::move(input)), request);
    };
    const auto message = [](ninfer::ChatRole role, std::string text) {
        ninfer::ChatMessage m;
        m.role = role;
        m.parts.push_back(ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = std::move(text)});
        return m;
    };
    const ninfer::ChatMessage system = message(ninfer::ChatRole::System, rules);
    // User turns longer than kMinimumTapSeparation (64 tokens): an opener closer to the structural
    // tap joins its cluster, and only the earlier tap is kept.
    std::string context;
    for (int i = 0; i < 4; ++i) {
        context += "Today a delivery of mixed items arrived at the warehouse and I need to plan where each "
                   "box goes before the afternoon shift starts. ";
    }
    const auto user = [&](const std::string& question) {
        return message(ninfer::ChatRole::User, context + question);
    };

    // X5 chat no-echo: turn 2 re-renders turn 1's reply without its reasoning block; it resumes at
    // least at turn 1's generation opener (a few tokens before turn 1's prompt end).
    const ninfer::GenerationResult s1 =
        chat({system, user("Which shelf takes colour 4?")}, 32);
    const std::uint32_t n1 = s1.prompt.prompt_tokens;
    const ninfer::GenerationResult x5 =
        chat({system, user("Which shelf takes colour 4?"), message(ninfer::ChatRole::Assistant, s1.content),
              message(ninfer::ChatRole::User, "And colour 5?")},
             32);
    std::printf("        X5: turn 1 prompt %u, turn 2 prompt %u reused %u\n", n1, x5.prompt.prompt_tokens,
                x5.reused_prompt_tokens);
    failures += check(x5.reused_prompt_tokens + 16U >= n1, "X5 turn 2 reuses at least turn 1's opener (reused " +
                                                               std::to_string(x5.reused_prompt_tokens) + ", turn 1 prompt " +
                                                               std::to_string(n1) + ")");

    // X6 shared preamble: sessions 2 and 3 share only the system block; they resume at its
    // structural tap, an exact split the first session made.
    const ninfer::GenerationResult s2 =
        chat({system, user("Summarise rule 12 in one sentence.")}, 32);
    const ninfer::GenerationResult s3 = chat({system, user("How many rules send items to shelf 0?")}, 32);
    std::printf("        X6: sessions 2 and 3 reused %u and %u of %u and %u\n", s2.reused_prompt_tokens,
                s3.reused_prompt_tokens, s2.prompt.prompt_tokens, s3.prompt.prompt_tokens);
    failures += check(s2.reused_prompt_tokens >= 1024 && s2.reused_prompt_tokens + 8U < s2.prompt.prompt_tokens &&
                          s3.reused_prompt_tokens == s2.reused_prompt_tokens,
                      "X6 sessions 2 and 3 resume at the shared structural frontier");

    // X3 split equivalence at an exact off-grid tap: session 2 again resumes at its own generation
    // opener; the run that captured it split its prefill there, so the resume repeats its computation.
    const ninfer::GenerationResult again = chat({system, user("Summarise rule 12 in one sentence.")}, 32);
    failures += check(again.reused_prompt_tokens + 16U >= s2.prompt.prompt_tokens &&
                          again.reused_prompt_tokens < s2.prompt.prompt_tokens,
                      "X3 the repeat resumes at the opener (reused " + std::to_string(again.reused_prompt_tokens) + ")");
    failures += check(again.generated_token_ids == s2.generated_token_ids,
                      "X3 the opener resume equals the capturing run (" +
                          first_difference(again.generated_token_ids, s2.generated_token_ids) + ")");
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* artifact = std::getenv("NINFER_QWEN4_ARTIFACT");
    if (artifact == nullptr) {
        std::printf("SKIP: set NINFER_QWEN4_ARTIFACT\n");
        return 77;
    }
    const char* ngram = std::getenv("NINFER_QWEN4_NGRAM");
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
        std::printf(failures == 0 ? "qwen4_exp prefix cache checks passed\n" : "FAIL: %d checks failed\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
