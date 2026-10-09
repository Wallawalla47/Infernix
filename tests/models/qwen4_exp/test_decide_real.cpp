// Qwen4Exp /v1/decide primitives and hybrid-cache admission on the real artifact. Skips unless
// INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact (INFERNIX_QWEN4_NGRAM: its n-gram volume, default
// <artifact>.ngram). One Engine per drafter mode serves every scenario (the model loads once per mode):
//
//   readout      A prompt's readout names its greedy first token and 15 others: the top token is the
//                one generated, no named token outscores it, and a resumed prompt reads what its cold
//                run reads (within rounding: their prefill calls differ). With INFERNIX_DECIDE_DUMP=DIR
//                the prompt ids and readout are written there for the FP64 oracle of the gate, which
//                log-softmaxes the forward test's FP32 logits of the same prompt rounded to BF16.
//   forced       One-token steps force their tokens; each draw reports probability 1.
//   chain        Steps {other, greedy}: the constrained greedy output equals the unconstrained one,
//                and each draw renormalizes over its step (with MTP: through draft verification).
//   cross        A step of two tokens the model does not prefer: the draw's probabilities and mass
//                equal what a readout of the same two tokens implies.
//   coalesce     (plain) Two requests submitted together share a 1,024-token prefix nothing has
//                cached: the second waits for the first's snapshot at the divergence, resumes from
//                it, and generates what an uncached run generates (same call boundaries).
//   prefetch     (plain) While both lanes decode, the waiting head's Host-only blocks are copied to
//                the Device; it then resumes at its tap and generates what an uncached run does.
//
//   infernix_qwen4_exp_decide_real_test [--mtp] [--plain]   (default: both modes)

#include "infernix/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
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

std::vector<infernix::TokenId> concat(std::vector<infernix::TokenId> a, const std::vector<infernix::TokenId>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

infernix::RequestOptions greedy(std::uint32_t outputs, bool reuse) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

int check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    return ok ? 0 : 1;
}

// `count` distinct tokens of the synthetic range, none of them in `avoid`.
std::vector<infernix::TokenId> others(std::size_t count, const std::vector<infernix::TokenId>& avoid, std::uint32_t seed) {
    std::vector<infernix::TokenId> out;
    for (const infernix::TokenId t : synthetic_tokens(4 * count + 16, seed)) {
        if (out.size() == count) { break; }
        if (std::find(avoid.begin(), avoid.end(), t) == avoid.end() && std::find(out.begin(), out.end(), t) == out.end()) {
            out.push_back(t);
        }
    }
    return out;
}

infernix::EngineOptions base_options(const char* artifact, const char* ngram, bool mtp) {
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    if (ngram != nullptr) { options.ngram_volume_path = ngram; }
    options.max_context          = 4096;
    // 72 pages (+ copy-on-write pages): a 3,900-token pressure prompt evicts nearly every block.
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(4608);
    options.prefill_chunk        = kChunk;
    options.kv_cache             = infernix::KvCacheStorage::Int8Group64;
    options.max_concurrency      = 2;
    options.max_pending_requests = 4;
    options.context_cache.host_capacity_bytes = 2ULL << 30;
    if (mtp) {
        options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    }
    return options;
}

infernix::GenerationResult run(infernix::Engine& engine, const std::vector<infernix::TokenId>& prompt,
                             const infernix::RequestOptions& options) {
    return engine.generate(engine.prepare_tokens(prompt), options);
}

int readout(infernix::Engine& engine, bool dump) {
    int failures = 0;
    const std::vector<infernix::TokenId> prompt = synthetic_tokens(700, 11);
    const auto cold = run(engine, prompt, greedy(1, false));
    std::vector<infernix::TokenId> named = {cold.generated_token_ids.at(0)};
    for (const infernix::TokenId t : others(15, named, 12)) { named.push_back(t); }
    infernix::RequestOptions options    = greedy(1, false);
    options.execution.readout_tokens  = named;
    const auto read                   = run(engine, prompt, options);
    failures += check(read.readout.has_value(), "readout: a readout is returned");
    if (!read.readout) { return failures; }
    const infernix::PromptReadout& r = *read.readout;
    failures += check(r.logprobs.size() == named.size(), "readout: one log-probability per named token");
    failures += check(read.generated_token_ids == cold.generated_token_ids && r.top_token == named[0],
                      "readout: the top token is the greedy first token");
    bool bounded = std::isfinite(r.top_logprob) && r.top_logprob <= 0.0F;
    for (const float lp : r.logprobs) { bounded = bounded && std::isfinite(lp) && lp <= r.top_logprob + 1e-5F; }
    failures += check(bounded, "readout: finite, at most 0, none above the top token");
    failures += check(std::fabs(r.logprobs[0] - r.top_logprob) <= 1e-4F, "readout: the top token's two readings agree");
    std::printf("        top %d logprob %.6f\n", r.top_token, static_cast<double>(r.top_logprob));
    if (dump) {
        const std::filesystem::path dir = std::getenv("INFERNIX_DECIDE_DUMP");
        std::ofstream ids(dir / "readout_prompt.ids");
        for (std::size_t i = 0; i < prompt.size(); ++i) { ids << (i ? "," : "") << prompt[i]; }
        std::ofstream values(dir / "readout.txt");
        values << "public " << engine.model_metadata().vocab_size << "\n";
        values << "top " << r.top_token << " " << r.top_logprob << "\n";
        for (std::size_t i = 0; i < named.size(); ++i) { values << named[i] << " " << r.logprobs[i] << "\n"; }
    }

    // A prompt resumed from the cache reads (within rounding) what its cold run reads.
    (void)run(engine, prompt, greedy(2, true)); // publishes the prompt's blocks, taps and endpoint
    const std::vector<infernix::TokenId> longer = concat(prompt, synthetic_tokens(150, 13));
    infernix::RequestOptions cold_longer        = greedy(1, false);
    cold_longer.execution.readout_tokens      = named;
    infernix::RequestOptions warm_longer        = cold_longer;
    warm_longer.execution.allow_prefix_reuse  = true;
    const auto a = run(engine, longer, cold_longer);
    const auto b = run(engine, longer, warm_longer);
    failures += check(b.reused_prompt_tokens > 0, "readout: the longer prompt resumes (reused " +
                                                      std::to_string(b.reused_prompt_tokens) + ")");
    double worst = 0.0;
    if (a.readout && b.readout) {
        for (std::size_t i = 0; i < named.size(); ++i) {
            worst = std::max(worst, std::fabs(static_cast<double>(a.readout->logprobs[i] - b.readout->logprobs[i])));
        }
    }
    failures += check(a.readout && b.readout && a.readout->top_token == b.readout->top_token && worst <= 0.05,
                      "readout: resumed equals cold within 0.05 nats (worst " + std::to_string(worst) + ")");
    return failures;
}

int forced(infernix::Engine& engine) {
    const std::vector<infernix::TokenId> prompt = synthetic_tokens(300, 21);
    const std::vector<infernix::TokenId> want   = synthetic_tokens(3, 22);
    infernix::RequestOptions options            = greedy(3, false);
    for (const infernix::TokenId t : want) { options.execution.constraint.steps.push_back({t}); }
    const auto result = run(engine, prompt, options);
    bool draws        = result.constrained_draws.size() == want.size();
    for (std::size_t i = 0; draws && i < want.size(); ++i) {
        const infernix::ConstrainedDraw& d = result.constrained_draws[i];
        draws = d.token == want[i] && d.probabilities.size() == 1 && std::fabs(d.probabilities[0] - 1.0F) <= 1e-6F &&
                d.mass > 0.0F && d.mass <= 1.0F + 1e-6F;
    }
    int failures = check(result.generated_token_ids == want, "forced: the output is the forced tokens");
    return failures + check(draws, "forced: one draw per token, probability 1, mass in (0,1]");
}

// Every step permits one set: the free greedy tokens plus others (16 in all). The constrained greedy
// output equals the free one, and each draw's largest probability is its token's.
int chain(infernix::Engine& engine, std::uint32_t steps, const char* label) {
    const std::vector<infernix::TokenId> prompt = synthetic_tokens(400, 31);
    const auto free_run                       = run(engine, prompt, greedy(steps, false));
    const std::vector<infernix::TokenId>& g     = free_run.generated_token_ids;
    std::vector<infernix::TokenId> set;
    for (const infernix::TokenId t : g) {
        if (std::find(set.begin(), set.end(), t) == set.end()) { set.push_back(t); }
    }
    for (const infernix::TokenId t : others(16 - set.size(), set, 32)) { set.push_back(t); }
    infernix::RequestOptions options = greedy(steps, false);
    for (std::uint32_t i = 0; i < steps && i < g.size(); ++i) { options.execution.constraint.steps.push_back(set); }
    const auto result = run(engine, prompt, options);
    bool draws        = result.constrained_draws.size() == g.size();
    for (std::size_t i = 0; draws && i < g.size(); ++i) {
        const infernix::ConstrainedDraw& d = result.constrained_draws[i];
        const auto at  = static_cast<std::size_t>(std::find(set.begin(), set.end(), g[i]) - set.begin());
        float sum      = 0.0F;
        for (const float p : d.probabilities) { sum += p; }
        draws = d.token == g[i] && d.probabilities.size() == set.size() && std::fabs(sum - 1.0F) <= 1e-3F &&
                std::max_element(d.probabilities.begin(), d.probabilities.end()) - d.probabilities.begin() ==
                    static_cast<std::ptrdiff_t>(at);
    }
    int failures = check(g.size() == steps && result.generated_token_ids == g,
                         std::string(label) + ": constrained greedy equals free greedy");
    failures += check(draws, std::string(label) + ": draws renormalize over each step and name the chosen token");
    std::printf("        speculative: %llu drafted, %llu accepted\n",
                static_cast<unsigned long long>(result.speculative.drafted_tokens),
                static_cast<unsigned long long>(result.speculative.accepted_tokens));
    return failures;
}

int cross(infernix::Engine& engine) {
    const std::vector<infernix::TokenId> prompt = synthetic_tokens(500, 41);
    const auto free_run                       = run(engine, prompt, greedy(1, false));
    const std::vector<infernix::TokenId> pair   = others(2, free_run.generated_token_ids, 42);
    infernix::RequestOptions options            = greedy(1, false);
    options.execution.readout_tokens          = pair;
    options.execution.constraint.steps.push_back(pair);
    const auto result = run(engine, prompt, options);
    if (!result.readout || result.constrained_draws.size() != 1) { return check(false, "cross: a readout and one draw"); }
    const double px = std::exp(static_cast<double>(result.readout->logprobs[0]));
    const double py = std::exp(static_cast<double>(result.readout->logprobs[1]));
    const infernix::ConstrainedDraw& d = result.constrained_draws[0];
    const infernix::TokenId expect     = px > py || (px == py && pair[0] < pair[1]) ? pair[0] : pair[1];
    int failures = check(d.token == expect && result.generated_token_ids.at(0) == expect,
                         "cross: the more probable permitted token is generated");
    failures += check(std::fabs(d.probabilities[0] - px / (px + py)) <= 2e-3,
                      "cross: the draw's probabilities equal the readout's renormalized");
    failures += check(std::fabs(d.mass / (px + py) - 1.0) <= 1e-2,
                      "cross: the draw's mass equals the readout's (relative 1e-2; mass " + std::to_string(d.mass) + ")");
    return failures;
}

int coalesce(infernix::Engine& engine) {
    const std::vector<infernix::TokenId> shared = synthetic_tokens(1024, 51);
    const auto first                          = concat(shared, synthetic_tokens(40, 52));
    const auto second                         = concat(shared, synthetic_tokens(40, 53));
    const auto reference                      = run(engine, second, greedy(16, false)).generated_token_ids;
    infernix::GenerationHandle leader   = engine.submit(engine.prepare_tokens(first), greedy(1, true));
    infernix::GenerationHandle follower = engine.submit(engine.prepare_tokens(second), greedy(16, true));
    const auto led      = leader.wait();
    const auto followed = follower.wait();
    int failures = check(led.reused_prompt_tokens == 0 && followed.reused_prompt_tokens == 1024,
                         "coalesce: the follower resumes at the divergence (leader " +
                             std::to_string(led.reused_prompt_tokens) + ", follower " +
                             std::to_string(followed.reused_prompt_tokens) + ")");
    return failures + check(followed.generated_token_ids == reference, "coalesce: the follower equals an uncached run");
}

int prefetch(infernix::Engine& engine) {
    // A 1,600-token prompt publishes flexible taps at its chunk boundaries; a prompt sharing its first
    // 1,536 tokens resumes at one of them (a chunk boundary, so its calls are the uncached run's).
    const std::vector<infernix::TokenId> base = synthetic_tokens(1600, 61);
    (void)run(engine, base, greedy(1, true));
    const auto head      = concat(std::vector<infernix::TokenId>(base.begin(), base.begin() + 1536), synthetic_tokens(40, 62));
    const auto reference = run(engine, head, greedy(8, false)).generated_token_ids;
    (void)run(engine, synthetic_tokens(3900, 63), greedy(1, false)); // evicts the Device blocks
    const infernix::RuntimeStats before = engine.runtime_stats();
    // A long request's prompt holds 52 of the 72 pages (a binding reserves the prompt plus one round,
    // and decode grows it): the head, with a lane free, waits for KV pages (its path is Host-only),
    // and its blocks are copied into what is free meanwhile.
    infernix::GenerationHandle a = engine.submit(engine.prepare_tokens(synthetic_tokens(3300, 64)), greedy(300, false));
    infernix::GenerationHandle h = engine.submit(engine.prepare_tokens(head), greedy(8, true));
    (void)a.wait();
    const auto waited                = h.wait();
    const infernix::RuntimeStats after = engine.runtime_stats();
    const auto prefetched            = after.hybrid_prefetched_blocks - before.hybrid_prefetched_blocks;
    int failures = check(prefetched > 0, "prefetch: the waiting head's blocks were prefetched (" +
                                             std::to_string(prefetched) + " blocks)");
    failures += check(waited.reused_prompt_tokens >= 1024 && waited.reused_prompt_tokens % kChunk == 0,
                      "prefetch: the head resumes at a chunk-boundary tap (" +
                          std::to_string(waited.reused_prompt_tokens) + ")");
    return failures + check(waited.generated_token_ids == reference, "prefetch: the head equals an uncached run");
}

int run_mode(const char* artifact, const char* ngram, bool mtp) {
    std::printf("== %s\n", mtp ? "MTP drafter (--spec mtp, 3 drafts)" : "plain decode");
    infernix::Engine engine(base_options(artifact, ngram, mtp));
    int failures = 0;
    failures += readout(engine, !mtp && std::getenv("INFERNIX_DECIDE_DUMP") != nullptr);
    failures += forced(engine);
    failures += chain(engine, mtp ? 12U : 6U, mtp ? "chain (MTP verify)" : "chain");
    failures += cross(engine);
    if (!mtp) {
        failures += coalesce(engine);
        failures += prefetch(engine);
    }
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
        std::printf(failures == 0 ? "qwen4_exp decide checks passed\n" : "FAIL: %d checks failed\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
