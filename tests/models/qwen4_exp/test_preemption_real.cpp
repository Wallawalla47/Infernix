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
//   replay   Neither request publishes (allow_prefix_reuse = false), so the paused one resumes by
//            replaying its ledger from the root without sampling. Which one is paused is the turn's:
//            the request first refused a unit owns it and pauses the other, the one whose pause
//            recomputes the fewest tokens (both replay here; resource-scheduling, "Service turns"),
//            and which lane is refused first depends on how its prefill and the other's rounds
//            interleave in time. (With the older request publishing, the younger one was expected
//            to be paused; when the younger one owned the turn the older one, whose pause recomputes
//            nothing, was paused instead, and the replay went unexercised: 2026-11-17, 5 of 5 MTP runs.)
//            The paused request's output keeps every token committed before the pause, completes at
//            full length and reports a replay restore; the tokens after the pause may differ from the
//            solo run (prefill arithmetic), which is reported. The other one equals its solo run.
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
// The exact and replay scenarios hold every Host restore's copies 20 ms (testing::set_restore_delay): a paused
// request's blocks leave the Device for the other request, so its resume restores them from the Host, and a round
// that reads them without waiting for that restore would read them before they land every time, not only when the
// copy happens to lose the race (an exact resume decodes at once, with no prefill call to wait per layer).
//
//   penalties (plain mode) Chat turns thinking under a 16-token budget, greedy with presence and frequency penalties:
//            the forced thinking-close tokens count toward the penalties alike live and in a resume's rebuilt counts,
//            so the paused B equals its solo run.
//
//   infernix_qwen4_exp_preemption_real_test [--mtp] [--plain] [--dflash2] [--penalties]
//   (default: plain, penalties and MTP, and DFlash2 when INFERNIX_QWEN4_DFLASH2 is set)

#include "kv_cache_storage.h"
#include "models/qwen4_exp/program/prefix/test_taps.h"
#include "engine_host.h"
#include "infernix/engine.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr std::uint32_t kPrompt  = 1024;
constexpr std::uint32_t kOutputs = 600;
// What each Host restore of the paired runs is held before its copies (testing::set_restore_delay).
constexpr std::uint32_t kRestoreDelayUs = 20'000;

// The restore delay for one scope.
struct RestoreDelay {
    RestoreDelay() { infernix::models::qwen4_exp::testing::set_restore_delay(kRestoreDelayUs); }
    ~RestoreDelay() { infernix::models::qwen4_exp::testing::set_restore_delay(0); }
    RestoreDelay(const RestoreDelay&)            = delete;
    RestoreDelay& operator=(const RestoreDelay&) = delete;
};

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
    infernix::Engine engine(options, infernix::test::engine_host());
    const auto prompt_a = synthetic_tokens(kPrompt, mtp ? 11U : 1U);
    const auto prompt_b = synthetic_tokens(kPrompt, mtp ? 12U : 2U);
    int failures = 0;

    // Solo references (one resident at a time, no prefix reuse: cold call grids).
    const auto solo_a = engine.generate(engine.prepare_tokens(prompt_a), greedy(false)).generated_token_ids;
    const auto solo_b = engine.generate(engine.prepare_tokens(prompt_b), greedy(false)).generated_token_ids;
    failures += check(solo_a.size() == kOutputs && solo_b.size() == kOutputs, "solo runs generate every token");

    const RestoreDelay delay;
    // exact: both publish; B (younger) is paused and restored from its own published state.
    {
        const std::uint64_t restored = engine.runtime_stats().hybrid_host_restore_bytes;
        auto a = engine.submit(engine.prepare_tokens(prompt_a), greedy(true));
        auto b = engine.submit(engine.prepare_tokens(prompt_b), greedy(true));
        const infernix::GenerationResult ra = a.wait();
        const infernix::GenerationResult rb = b.wait();
        const std::uint64_t restores = engine.runtime_stats().hybrid_host_restore_bytes - restored;
        std::printf("  exact: %.1f MiB restored from the Host tier\n", static_cast<double>(restores) / (1 << 20));
        failures += check(restores > 0, "the resumes restore from the Host tier");
        std::printf("  exact: A preemptions %llu (snapshot restores %llu, replay restores %llu), B preemptions %llu, "
                    "snapshot restores %llu, replay restores %llu\n",
                    static_cast<unsigned long long>(ra.scheduling.preemptions),
                    static_cast<unsigned long long>(ra.scheduling.snapshot_restores),
                    static_cast<unsigned long long>(ra.scheduling.replay_restores),
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
    // replay: neither publishes, so the paused one's resume replays its ledger from the root.
    {
        auto a = engine.submit(engine.prepare_tokens(prompt_a), greedy(false));
        auto b = engine.submit(engine.prepare_tokens(prompt_b), greedy(false));
        const infernix::GenerationResult ra = a.wait();
        const infernix::GenerationResult rb = b.wait();
        std::printf("  replay: A preemptions %llu (replay restores %llu, replayed tokens %llu, tokens equal to solo %zu); "
                    "B preemptions %llu (replay restores %llu, replayed tokens %llu, tokens equal to solo %zu)\n",
                    static_cast<unsigned long long>(ra.scheduling.preemptions),
                    static_cast<unsigned long long>(ra.scheduling.replay_restores),
                    static_cast<unsigned long long>(ra.scheduling.replayed_tokens),
                    common_prefix(ra.generated_token_ids, solo_a),
                    static_cast<unsigned long long>(rb.scheduling.preemptions),
                    static_cast<unsigned long long>(rb.scheduling.replay_restores),
                    static_cast<unsigned long long>(rb.scheduling.replayed_tokens),
                    common_prefix(rb.generated_token_ids, solo_b));
        failures += check(ra.scheduling.preemptions + rb.scheduling.preemptions >= 1, "the shared pool forces a pause");
        for (const auto& [label, result, solo] :
             {std::tuple<const char*, const infernix::GenerationResult&, const std::vector<infernix::TokenId>&>{"A", ra, solo_a},
              std::tuple<const char*, const infernix::GenerationResult&, const std::vector<infernix::TokenId>&>{"B", rb, solo_b}}) {
            if (result.scheduling.preemptions >= 1) {
                failures += check(result.scheduling.replay_restores >= 1 && result.scheduling.replayed_tokens >= kPrompt,
                                  std::string(label) + ", paused, resumes by replaying its ledger");
                failures += check(result.generated_token_ids.size() == kOutputs,
                                  std::string(label) + ", paused, completes every token");
            } else {
                failures += check(result.generated_token_ids == solo,
                                  std::string(label) + ", never paused, equals its solo run (first difference at " +
                                      std::to_string(common_prefix(result.generated_token_ids, solo)) + ")");
            }
        }
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
    infernix::Engine engine(options, infernix::test::engine_host());
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

// A chat turn of about kPrompt tokens that opens in thinking (a user message of synthetic words).
infernix::PreparedPrompt thinking_turn(const infernix::Engine& engine, std::uint32_t seed) {
    std::uint32_t state = seed * 2654435761U + 7U;
    std::string text    = "Summarize these words:";
    for (;;) {
        for (int i = 0; i < 40; ++i) {
            state = state * 1664525U + 1013904223U;
            text += " w" + std::to_string((state >> 8U) % 5000U);
        }
        infernix::PromptInput input;
        infernix::ChatMessage user;
        user.role = infernix::ChatRole::User;
        user.parts.push_back(infernix::MessagePart{.kind = infernix::MessagePartKind::Text, .text = text, .media = {}});
        input.messages.push_back(std::move(user));
        input.options.enable_thinking = true;
        infernix::PreparedPrompt prepared = engine.prepare(std::move(input));
        if (prepared.summary().prompt_tokens >= kPrompt - 64) { return prepared; }
    }
}

// penalties  (plain mode) The exact scenario on chat turns that think under a 16-token budget, decoding greedily
//            with presence and frequency penalties: the Engine forces each turn's thinking close (forced tokens)
//            early, B is paused after them and resumes from its published state, its penalty counts rebuilt from its
//            history, and B must equal its solo run, whose counts took the forced tokens as they were appended. A
//            chat prompt's publication plants taps at its message boundaries, which split its prefill calls, so the
//            solo references publish as well, on a cold cache of their own (a first Engine; the two prompts share no
//            cache block): the pair (a second Engine, also cold) then runs the same call grids.
int run_penalty_pause(const char* artifact, const char* ngram) {
    std::printf("== plain decode, penalties and forced thinking-close tokens across a pause\n");
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    if (ngram != nullptr) { options.ngram_volume_path = ngram; }
    options.max_context                       = 4096;
    options.kv_capacity                       = infernix::KvCapacityPolicy::explicit_capacity(2560);
    options.prefill_chunk                     = 512;
    const char* kv                            = std::getenv("INFERNIX_QWEN4_KV");
    options.kv_cache                          = kv != nullptr ? infernix::test::parse_kv_cache_storage(kv)
                                                              : infernix::KvCacheStorage::Int8Group64;
    options.expert_a4_activations             = std::getenv("INFERNIX_QWEN4_EXPERT_A4") != nullptr;
    options.max_concurrency                   = 2;
    options.max_pending_requests              = 4;
    options.context_cache.host_capacity_bytes = 2ULL << 30;
    options.fast_paths                        = false;
    options.fast_path_candidates = {infernix::FastPath::Preemption, infernix::FastPath::FreeCacheResume};
    // The forced text is common words, so whether its tokens count changes what the penalties let greedy decoding
    // pick after it (the built-in close alone did not: B's tokens were the same either way).
    options.thinking_budget_message = "The answer is the list, and the words of the list are the answer. So the answer "
                                      "is: the words, the list, and the answer.";
    const auto request = [] {
        infernix::RequestOptions out             = greedy(true);
        out.execution.sampling.presence_penalty  = 1.0F;
        out.execution.sampling.frequency_penalty = 0.5F;
        out.execution.thinking.budget            = 16;
        return out;
    };
    int failures = 0;
    infernix::GenerationResult solo_a, solo_b;
    {
        infernix::Engine engine(options, infernix::test::engine_host());
        solo_a = engine.generate(thinking_turn(engine, 31U), request());
        solo_b = engine.generate(thinking_turn(engine, 32U), request());
    }
    failures += check(solo_b.thinking.injected_tokens > 0, "the thinking budget forces B's thinking close (" +
                                                               std::to_string(solo_b.thinking.injected_tokens) + " tokens)");
    infernix::Engine engine(options, infernix::test::engine_host());
    const RestoreDelay delay;
    auto a                              = engine.submit(thinking_turn(engine, 31U), request());
    auto b                              = engine.submit(thinking_turn(engine, 32U), request());
    const infernix::GenerationResult ra = a.wait();
    const infernix::GenerationResult rb = b.wait();
    std::printf("  penalties: A preemptions %llu; B preemptions %llu (snapshot restores %llu, replay restores %llu), "
                "B forced tokens %u, B reused %u prompt tokens\n",
                static_cast<unsigned long long>(ra.scheduling.preemptions),
                static_cast<unsigned long long>(rb.scheduling.preemptions),
                static_cast<unsigned long long>(rb.scheduling.snapshot_restores),
                static_cast<unsigned long long>(rb.scheduling.replay_restores), rb.thinking.injected_tokens,
                rb.reused_prompt_tokens);
    failures += check(ra.scheduling.preemptions + rb.scheduling.preemptions >= 1, "the shared pool forces a pause");
    failures += check(rb.scheduling.preemptions == 0 || rb.scheduling.snapshot_restores >= 1,
                      "a paused B resumes from its published state");
    failures += check(ra.generated_token_ids == solo_a.generated_token_ids,
                      "A equals its solo run (first difference at " +
                          std::to_string(common_prefix(ra.generated_token_ids, solo_a.generated_token_ids)) + ")");
    failures += check(rb.generated_token_ids == solo_b.generated_token_ids,
                      "B equals its solo run across its pause (first difference at " +
                          std::to_string(common_prefix(rb.generated_token_ids, solo_b.generated_token_ids)) + ")");
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
    bool plain = true, mtp = true, dflash2 = has_drafter, penalties = true;
    if (argc > 1) {
        plain = mtp = dflash2 = penalties = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--plain") { plain = true; }
            if (arg == "--mtp") { mtp = true; }
            if (arg == "--dflash2") { dflash2 = true; }
            if (arg == "--penalties") { penalties = true; }
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
        if (penalties) { failures += run_penalty_pause(artifact, ngram); }
        if (mtp) { failures += run_mode(artifact, ngram, Mode::Mtp); }
        if (dflash2) { failures += run_mode(artifact, ngram, Mode::DFlash2Pairs, drafter); }
        std::printf(failures == 0 ? "all preemption checks passed\n" : "%d preemption check(s) failed\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
