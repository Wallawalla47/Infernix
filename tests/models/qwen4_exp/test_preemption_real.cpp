// Qwen4Exp preemption on the real artifact (design §19.3.13). Skips unless INFERNIX_QWEN4_ARTIFACT
// names a Qwen4Exp artifact (INFERNIX_QWEN4_NGRAM: its n-gram volume, default <artifact>.ngram).
// INFERNIX_QWEN4_KV selects the KV storage (bf16, int8, fp8, nvfp4, k8v4, vq2, k4v2; default int8);
// INFERNIX_QWEN4_EXPERT_A4 (any value) runs NVIDIA FP4-Local's experts with 4-bit activations.
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
// The DFlash2 mode (INFERNIX_QWEN4_DFLASH2 names the drafter companion) adds dflash2-pairs: the two
// residents decode in two-row DFlash2 rounds until the pause, so the exact scenario also checks that
// pair rounds and the drafter's ring across a pause and resume keep each request's solo tokens.
//
//   prompt   (plain mode) A decodes; B's long prompt binds the pool's last pages a few tokens
//            before A needs another, so B is paused inside its prompt. B publishes nothing, so it
//            resumes from the root, below the frontier the Engine had counted: it walks the prompt
//            again, reports each token once, and equals its solo run.
//
//   infernix_qwen4_exp_preemption_real_test [--mtp] [--plain] [--dflash2]
//   (default: plain and MTP, and DFlash2 when INFERNIX_QWEN4_DFLASH2 is set)

#include "kv_cache_storage.h"
#include "infernix/engine.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kPrompt  = 1024;
constexpr std::uint32_t kOutputs = 600;

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

enum class Mode { Plain, Mtp, DFlash2Pairs };

int run_mode(const char* artifact, const char* ngram, Mode mode, const char* drafter = nullptr) {
    const bool mtp = mode != Mode::Plain; // the drafter modes share their prompts
    std::printf("== %s\n", mode == Mode::Mtp            ? "MTP drafter (--spec mtp, 3 drafts)"
                           : mode == Mode::DFlash2Pairs ? "DFlash2 drafter (--spec dflash2, 3 drafts) with dflash2-pairs"
                                                        : "plain decode");
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    if (ngram != nullptr) { options.ngram_volume_path = ngram; }
    options.max_context = 4096;
    // 40 pages of 64 tokens: both prompts bind (2 x 17 pages) but both extents (2 x 26: 1,024 +
    // 600 tokens) do not, so the pool runs out about 190 tokens into the paired runs.
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(2560);
    options.prefill_chunk        = 512;
    const char* kv               = std::getenv("INFERNIX_QWEN4_KV");
    options.kv_cache             = kv != nullptr ? infernix::test::parse_kv_cache_storage(kv)
                                                 : infernix::KvCacheStorage::Int8Group64;
    options.expert_a4_activations = std::getenv("INFERNIX_QWEN4_EXPERT_A4") != nullptr;
    options.max_concurrency      = 2;
    options.max_pending_requests = 4;
    options.context_cache.host_capacity_bytes = 2ULL << 30;
    // Solo equality holds on the exact route (model-quality.md §6.1), where the default route's
    // q8-sliced-k and gdn-conv-fused couple a request to its round; pausing a resident is the
    // preemption path, and resuming from its published decode state is free-cache-resume.
    options.fast_paths           = false;
    options.fast_path_candidates = {infernix::FastPath::Preemption,
                                    infernix::FastPath::FreeCacheResume};
    if (mode == Mode::Mtp) {
        options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    }
    if (mode == Mode::DFlash2Pairs) {
        options.speculative.backend      = infernix::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = 3;
        options.speculative.draft_model  = drafter;
        options.fast_path_candidates.push_back(infernix::FastPath::DFlash2Pairs);
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

int run_prompt_pause(const char* artifact, const char* ngram) {
    std::printf("== plain decode, B paused inside its prompt\n");
    // The pool is 62 pages of 64 tokens and the prefix cache's 2 lane pages: the 64 pages mapped at startup
    // (kKvBaseTokens), so no binding waits for the pool to grow from the expert frames (a refused growth defers a
    // fresh binding until the resident finishes). A (1,083 tokens) binds its prompt and one round: 17 pages,
    // through 1,088 positions; B (2,944 tokens) binds the other 47. A's round past 1,088 positions, about 4 tokens
    // into its decode, pauses B, the younger, a few of its 12 prefill calls in (one A round runs between calls).
    constexpr std::uint32_t kShort = 1083, kLong = 2944, kChunk = 256, kShortOutputs = 400, kLongOutputs = 32;
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    if (ngram != nullptr) { options.ngram_volume_path = ngram; }
    options.max_context          = 4096;
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(62U * 64U);
    options.prefill_chunk        = kChunk;
    const char* kv               = std::getenv("INFERNIX_QWEN4_KV");
    options.kv_cache             = kv != nullptr ? infernix::test::parse_kv_cache_storage(kv)
                                                 : infernix::KvCacheStorage::Int8Group64;
    options.expert_a4_activations = std::getenv("INFERNIX_QWEN4_EXPERT_A4") != nullptr;
    options.max_concurrency      = 2;
    options.max_pending_requests = 4;
    options.fast_paths           = false;
    options.fast_path_candidates = {infernix::FastPath::Preemption};
    infernix::Engine engine(options);
    const auto prompt_a = synthetic_tokens(kShort, 21U);
    const auto prompt_b = synthetic_tokens(kLong, 22U);
    const auto request  = [](std::uint32_t outputs) {
        infernix::RequestOptions out = greedy(false);
        out.execution.requested_output_tokens = outputs;
        return out;
    };
    int failures = 0;
    const auto solo_b = engine.generate(engine.prepare_tokens(prompt_b), request(kLongOutputs)).generated_token_ids;

    // B publishes nothing, so its pause leaves no state behind and it resumes from the root.
    auto a = engine.submit(engine.prepare_tokens(prompt_a), request(kShortOutputs));
    auto b = engine.submit(engine.prepare_tokens(prompt_b), request(kLongOutputs));
    std::optional<infernix::GenerationResult> ra, rb;
    std::string error;
    try {
        ra = a.wait();
        rb = b.wait();
    } catch (const std::exception& failure) { error = failure.what(); }
    failures += check(error.empty() && ra && rb, "both requests complete" + (error.empty() ? "" : ": " + error));
    if (!rb) { return failures; }
    std::printf("  prompt: A preemptions %llu; B preemptions %llu, snapshot restores %llu, replay restores %llu, "
                "computed prefill %llu\n",
                static_cast<unsigned long long>(ra->scheduling.preemptions),
                static_cast<unsigned long long>(rb->scheduling.preemptions),
                static_cast<unsigned long long>(rb->scheduling.snapshot_restores),
                static_cast<unsigned long long>(rb->scheduling.replay_restores),
                static_cast<unsigned long long>(rb->computed_prefill_tokens));
    // When B was admitted against A: a B that waited for A's whole decode was never resident beside it.
    std::printf("  prompt: A first token %.2f s, decode %.2f s; B first token %.2f s, prompt wall %.2f s\n",
                ra->timings.first_token_seconds, ra->timings.decode_seconds, rb->timings.first_token_seconds,
                rb->timings.prompt_wall_seconds);
    failures += check(rb->scheduling.preemptions >= 1, "B is paused");
    // A pause after B's first token would resume by replaying its ledger.
    failures += check(rb->scheduling.replay_restores == 0, "B is paused inside its prompt");
    failures += check(rb->computed_prefill_tokens == kLong, "B's prompt is counted once");
    failures += check(rb->generated_token_ids == solo_b, "B equals its solo run (first difference at " +
                                                             std::to_string(common_prefix(rb->generated_token_ids, solo_b)) + ")");
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::printf("SKIP: INFERNIX_QWEN4_ARTIFACT is not set\n");
        return 77;
    }
    const char* ngram   = std::getenv("INFERNIX_QWEN4_NGRAM");
    const char* drafter = std::getenv("INFERNIX_QWEN4_DFLASH2");
    const bool has_drafter = drafter != nullptr && *drafter != '\0';
    bool plain = true, mtp = true, dflash2 = has_drafter;
    if (argc > 1) {
        plain = mtp = dflash2 = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--plain") { plain = true; }
            if (arg == "--mtp") { mtp = true; }
            if (arg == "--dflash2") { dflash2 = true; }
        }
    }
    if (dflash2 && !has_drafter) {
        std::fprintf(stderr, "FAIL: --dflash2 needs INFERNIX_QWEN4_DFLASH2 (the DFlash2 drafter companion)\n");
        return 1;
    }
    try {
        int failures = 0;
        if (plain) {
            failures += run_mode(artifact, ngram, Mode::Plain);
            failures += run_prompt_pause(artifact, ngram);
        }
        if (mtp) { failures += run_mode(artifact, ngram, Mode::Mtp); }
        if (dflash2) { failures += run_mode(artifact, ngram, Mode::DFlash2Pairs, drafter); }
        std::printf(failures == 0 ? "all preemption checks passed\n" : "%d preemption check(s) failed\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
