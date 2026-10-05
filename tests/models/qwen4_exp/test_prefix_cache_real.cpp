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
