// Qwen4Exp preemption on the real artifact (design §19.3.13). Skips unless INFERNIX_QWEN4_ARTIFACT
// names a Qwen4Exp artifact (INFERNIX_QWEN4_NGRAM: its n-gram volume, default <artifact>.ngram).
// INFERNIX_QWEN4_KV selects the KV storage (bf16, int8, fp8, nvfp4, k8v4, vq2, k4v2; default int8).
//
// Two lanes share a KV pool too small for both requests' full extents: each binds its prompt and
// grows per round, so the younger request is paused when the pool runs out and resumes once the
// older one finishes.
//
//   exact    Both requests publish to the prefix cache: the pause publishes the younger one's
//            state at its frontier and the resume restores it, so both outputs equal their solo
//            runs (greedy) and the younger request reports a snapshot restore.
//   replay   The younger request does not publish (allow_prefix_reuse = false): its resume replays
//            the ledger from the root without sampling. Its output keeps every token committed
//            before the pause, completes at full length and reports a replay restore; the tokens
//            after the pause may differ from the solo run (prefill arithmetic), which is reported.
//
//   infernix_qwen4_exp_preemption_real_test [--mtp] [--plain]   (default: both modes)

#include "kv_cache_storage.h"
#include "infernix/engine.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kPrompt  = 1024;
constexpr std::uint32_t kOutputs = 1200;

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

infernix::RequestOptions greedy(bool reuse) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = kOutputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

int check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    return ok ? 0 : 1;
}

std::size_t common_prefix(const std::vector<infernix::TokenId>& a, const std::vector<infernix::TokenId>& b) {
    return static_cast<std::size_t>(std::mismatch(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(std::min(a.size(), b.size())),
                                                  b.begin())
                                        .first -
                                    a.begin());
}

int run_mode(const char* artifact, const char* ngram, bool mtp) {
    std::printf("== %s\n", mtp ? "MTP drafter (--spec mtp, 3 drafts)" : "plain decode");
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    if (ngram != nullptr) { options.ngram_volume_path = ngram; }
    options.max_context = 4096;
    // 48 pages: both prompts bind (2 x 17 pages) but both extents (2 x 35) do not fit.
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(3072);
    options.prefill_chunk        = 512;
    const char* kv               = std::getenv("INFERNIX_QWEN4_KV");
    options.kv_cache             = kv != nullptr ? infernix::test::parse_kv_cache_storage(kv)
                                                 : infernix::KvCacheStorage::Int8Group64;
    options.max_concurrency      = 2;
    options.max_pending_requests = 4;
    options.context_cache.mode                = infernix::ContextCacheMode::Hybrid;
    options.context_cache.host_capacity_bytes = 2ULL << 30;
    if (mtp) {
        options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    }
    infernix::Engine engine(options);
    const auto prompt_a = synthetic_tokens(kPrompt, mtp ? 11U : 1U);
    const auto prompt_b = synthetic_tokens(kPrompt, mtp ? 12U : 2U);
    int failures = 0;

    // Solo references (one resident at a time, no prefix reuse: cold call grids).
    const auto solo_a = engine.generate(engine.prepare_tokens(prompt_a), greedy(false)).generated_token_ids;
    const auto solo_b = engine.generate(engine.prepare_tokens(prompt_b), greedy(false)).generated_token_ids;
    failures += check(solo_a.size() == kOutputs && solo_b.size() == kOutputs, "solo runs generate every token");

    // exact: both publish; B (younger) is paused and restored from its own published state.
    {
        auto a = engine.submit(engine.prepare_tokens(prompt_a), greedy(true));
        auto b = engine.submit(engine.prepare_tokens(prompt_b), greedy(true));
        const infernix::GenerationResult ra = a.wait();
        const infernix::GenerationResult rb = b.wait();
        std::printf("  exact: A preemptions %llu, B preemptions %llu, snapshot restores %llu, replay restores %llu\n",
                    static_cast<unsigned long long>(ra.scheduling.preemptions),
                    static_cast<unsigned long long>(rb.scheduling.preemptions),
                    static_cast<unsigned long long>(rb.scheduling.snapshot_restores),
                    static_cast<unsigned long long>(rb.scheduling.replay_restores));
        failures += check(ra.scheduling.preemptions + rb.scheduling.preemptions >= 1, "the shared pool forces a pause");
        failures += check(ra.generated_token_ids == solo_a, "A equals its solo run (first difference at " +
                                                                std::to_string(common_prefix(ra.generated_token_ids, solo_a)) + ")");
        failures += check(rb.generated_token_ids == solo_b, "B equals its solo run across its pause (first difference at " +
                                                                std::to_string(common_prefix(rb.generated_token_ids, solo_b)) + ")");
        failures += check(rb.scheduling.preemptions == 0 || rb.scheduling.snapshot_restores >= 1,
                          "a paused publishing request resumes from its published state");
    }
    // replay: B publishes nothing, so its resume replays the ledger from the root.
    {
        auto a = engine.submit(engine.prepare_tokens(prompt_a), greedy(true));
        auto b = engine.submit(engine.prepare_tokens(prompt_b), greedy(false));
        const infernix::GenerationResult ra = a.wait();
        const infernix::GenerationResult rb = b.wait();
        const std::size_t kept = common_prefix(rb.generated_token_ids, solo_b);
        std::printf("  replay: B preemptions %llu, replay restores %llu, replayed tokens %llu, tokens equal to solo %zu of %zu\n",
                    static_cast<unsigned long long>(rb.scheduling.preemptions),
                    static_cast<unsigned long long>(rb.scheduling.replay_restores),
                    static_cast<unsigned long long>(rb.scheduling.replayed_tokens), kept, rb.generated_token_ids.size());
        failures += check(ra.generated_token_ids == solo_a, "A equals its solo run");
        failures += check(rb.scheduling.preemptions >= 1, "B is paused");
        failures += check(rb.scheduling.replay_restores >= 1 && rb.scheduling.replayed_tokens >= kPrompt,
                          "B resumes by replaying its ledger");
        failures += check(rb.generated_token_ids.size() == kOutputs, "B completes every token");
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::printf("SKIP: INFERNIX_QWEN4_ARTIFACT is not set\n");
        return 77;
    }
    const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM");
    bool plain = true, mtp = true;
    if (argc > 1) {
        plain = mtp = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--plain") { plain = true; }
            if (arg == "--mtp") { mtp = true; }
        }
    }
    try {
        int failures = 0;
        if (plain) { failures += run_mode(artifact, ngram, false); }
        if (mtp) { failures += run_mode(artifact, ngram, true); }
        std::printf(failures == 0 ? "all preemption checks passed\n" : "%d preemption check(s) failed\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
