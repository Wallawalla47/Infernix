// Qwen4Exp with a DFlash2 drafter (--spec dflash2 --draft-model) on the real artifacts. Skips unless
// INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact and INFERNIX_QWEN4_DFLASH2 its DFlash2 companion
// (tools.convert --components dflash2); INFERNIX_QWEN4_NGRAM names its n-gram volume (default
// <artifact>.ngram), INFERNIX_QWEN4_KV the KV storage (default int8). Every engine runs the exact
// route (--disable-fast-paths).
//
//   greedy        Two prompts, 256 greedy tokens: DFlash2 7 (text head, and the proposal head with
//                 --lm-head-draft when the artifact has one) gives plain decoding's tokens, and its
//                 drafts are accepted (a drafter reading wrong features accepts next to nothing).
//   concurrency   Four greedy requests in flight on a four-lane engine (plain rounds of several rows,
//                 whose catch-up writes several rings at once) give the one-lane tokens.
//   tap resume    With the prefix cache (512-token calls), a request resuming from another's tap at
//                 1024 restores the drafter's context ring with the rest of the state: its tokens and
//                 its drafted and accepted counts equal a cold run's (same call boundaries).
//   distribution  N seeded requests (T 1.0, top-k 20, top-p 0.95, G tokens each) from plain decoding
//                 and from DFlash2: each output position's token distribution, and the first two
//                 tokens' jointly, must match (two-sample chi-square, family alpha 0.001,
//                 Bonferroni); plain decoding at T 0.8 must not (the test's power).
//
//   infernix_qwen4_exp_dflash2_real_test [--greedy-only] [--trials N] [--tokens G]

#include "kv_cache_storage.h"
#include "infernix/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Tokens = std::vector<infernix::TokenId>;

constexpr const char* kPrompt =
    "The old lighthouse keeper climbed the stairs one last time before the storm. At the top, he";
constexpr const char* kGreedyPrompts[] = {
    "Write a Python function that returns the n-th Fibonacci number, then explain its complexity.",
    "The three most important causes of the French Revolution were"};
constexpr std::uint32_t kDrafts = 7;

int check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    return ok ? 0 : 1;
}

std::string first_difference(const Tokens& a, const Tokens& b) {
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (a[i] != b[i]) { return "first difference at " + std::to_string(i); }
    }
    return a.size() == b.size() ? "identical" : "lengths differ";
}

// Q(a, x), the regularized upper incomplete gamma function, for chi-square tail probabilities.
double gamma_q(double a, double x) {
    if (x <= 0) { return 1.0; }
    const double log_prefix = -x + a * std::log(x) - std::lgamma(a);
    if (x < a + 1) {
        double term = 1.0 / a, sum = term;
        for (int n = 1; n < 1000 && std::abs(term) > std::abs(sum) * 1e-15; ++n) {
            term *= x / (a + n);
            sum += term;
        }
        return 1.0 - sum * std::exp(log_prefix);
    }
    double b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
    for (int i = 1; i < 1000; ++i) {
        const double an = -i * (i - a);
        b += 2;
        d = an * d + b;
        d = std::abs(d) < 1e-300 ? 1e-300 : d;
        c = b + an / c;
        c = std::abs(c) < 1e-300 ? 1e-300 : c;
        d = 1 / d;
        const double delta = d * c;
        h *= delta;
        if (std::abs(delta - 1) < 1e-15) { break; }
    }
    return std::exp(log_prefix) * h;
}

// Two-sample chi-square homogeneity over categorical keys (categories under 10 observations in
// both samples together pooled into one bin); the p-value.
double homogeneity(const std::map<std::int64_t, std::uint64_t>& a, const std::map<std::int64_t, std::uint64_t>& b) {
    std::map<std::int64_t, std::pair<double, double>> cells;
    for (const auto& [key, n] : a) { cells[key].first += static_cast<double>(n); }
    for (const auto& [key, n] : b) { cells[key].second += static_cast<double>(n); }
    std::vector<std::pair<double, double>> bins;
    std::pair<double, double> pooled{0, 0};
    for (const auto& [key, n] : cells) {
        if (n.first + n.second < 10) {
            pooled.first += n.first;
            pooled.second += n.second;
        } else {
            bins.push_back(n);
        }
    }
    if (pooled.first + pooled.second > 0) { bins.push_back(pooled); }
    if (bins.size() < 2) { return 1.0; }
    double na = 0, nb = 0;
    for (const auto& [x, y] : bins) {
        na += x;
        nb += y;
    }
    double statistic = 0;
    for (const auto& [x, y] : bins) {
        const double total = x + y;
        const double ea = total * na / (na + nb), eb = total * nb / (na + nb);
        statistic += (x - ea) * (x - ea) / ea + (y - eb) * (y - eb) / eb;
    }
    return gamma_q(static_cast<double>(bins.size() - 1) / 2, statistic / 2);
}

std::vector<double> compare(const std::vector<Tokens>& a, const std::vector<Tokens>& b, std::size_t tokens) {
    std::vector<double> out;
    for (std::size_t i = 0; i <= tokens; ++i) {
        std::map<std::int64_t, std::uint64_t> ca, cb;
        const auto key = [&](const Tokens& s) -> std::int64_t {
            if (i == tokens) { return (static_cast<std::int64_t>(s.at(0)) << 32) | static_cast<std::uint32_t>(s.at(1)); }
            return s.at(i);
        };
        for (const auto& s : a) { ++ca[key(s)]; }
        for (const auto& s : b) { ++cb[key(s)]; }
        out.push_back(homogeneity(ca, cb));
    }
    return out;
}

int report(const char* name, const std::vector<double>& p, double alpha, bool expect_same) {
    const double least = *std::min_element(p.begin(), p.end());
    std::printf("  %s: p per position", name);
    for (std::size_t i = 0; i < p.size(); ++i) { std::printf(i + 1 == p.size() ? " | joint(0,1) %.3g" : " %.3g", p[i]); }
    std::printf("; least %.3g against %.3g\n", least, alpha);
    return check(expect_same ? least >= alpha : least < alpha,
                 std::string(name) + (expect_same ? " matches plain sampling" : " differs from plain sampling (power)"));
}

struct Setup {
    const char* artifact = nullptr;
    const char* ngram    = nullptr;
    const char* drafter  = nullptr;
};

enum class Drafter { None, TextHead, ProposalHead };

infernix::EngineOptions engine_options(const Setup& setup, Drafter drafter, std::uint32_t concurrency) {
    infernix::EngineOptions options;
    options.artifact_path = setup.artifact;
    if (setup.ngram != nullptr) { options.ngram_volume_path = setup.ngram; }
    options.max_context          = 4096;
    const char* kv               = std::getenv("INFERNIX_QWEN4_KV");
    options.kv_cache             = kv != nullptr ? infernix::test::parse_kv_cache_storage(kv) : infernix::KvCacheStorage::Int8Group64;
    options.max_concurrency      = concurrency;
    options.max_pending_requests = 2 * concurrency;
    options.fast_paths           = false;
    if (drafter != Drafter::None) {
        options.speculative.backend      = infernix::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = kDrafts;
        options.speculative.draft_model  = setup.drafter;
        if (drafter == Drafter::ProposalHead) { options.speculative.proposal_head = infernix::ProposalHead::Optimized; }
    }
    return options;
}

infernix::RequestOptions request(float temperature, std::uint64_t seed, std::uint32_t tokens, bool reuse = false) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = temperature;
    options.execution.sampling.top_k          = 20;
    options.execution.sampling.top_p          = 0.95F;
    options.execution.sampling.seed           = seed;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

struct Outputs {
    std::vector<Tokens> tokens;
    infernix::SpeculativeStats speculative; // summed
};

void add(infernix::SpeculativeStats& sum, const infernix::SpeculativeStats& one) {
    sum.rounds += one.rounds;
    sum.drafted_tokens += one.drafted_tokens;
    sum.accepted_tokens += one.accepted_tokens;
}

Outputs greedy(infernix::Engine& engine) {
    Outputs out;
    for (const char* text : kGreedyPrompts) {
        const auto result = engine.generate(engine.prepare_tokens(engine.tokenize_text(text)), request(0.0F, 0, 256));
        out.tokens.push_back(result.generated_token_ids);
        add(out.speculative, result.speculative);
    }
    return out;
}

void print_acceptance(const infernix::SpeculativeStats& s) {
    std::printf("  rounds %llu, accepted %llu of %llu drafts (%.1f %%, %.2f tokens per round)\n",
                static_cast<unsigned long long>(s.rounds), static_cast<unsigned long long>(s.accepted_tokens),
                static_cast<unsigned long long>(s.drafted_tokens),
                s.drafted_tokens == 0 ? 0.0 : 100.0 * static_cast<double>(s.accepted_tokens) / static_cast<double>(s.drafted_tokens),
                s.rounds == 0 ? 0.0 : 1.0 + static_cast<double>(s.accepted_tokens) / static_cast<double>(s.rounds));
}

// `count` seeded requests from `first_seed`, `concurrency` in flight.
Outputs sample(infernix::Engine& engine, const Tokens& prompt, float temperature, std::uint64_t first_seed,
               std::uint32_t count, std::uint32_t tokens, std::uint32_t concurrency) {
    Outputs out;
    for (std::uint32_t at = 0; at < count; at += concurrency) {
        std::vector<infernix::GenerationHandle> handles;
        for (std::uint32_t i = at; i < std::min(count, at + concurrency); ++i) {
            handles.push_back(engine.submit(engine.prepare_tokens(prompt), request(temperature, first_seed + i, tokens)));
        }
        for (auto& handle : handles) {
            const infernix::GenerationResult result = handle.wait();
            if (result.generated_token_ids.size() != tokens) {
                throw std::runtime_error("a sampled request generated " + std::to_string(result.generated_token_ids.size()) +
                                         " tokens instead of " + std::to_string(tokens));
            }
            out.tokens.push_back(result.generated_token_ids);
            add(out.speculative, result.speculative);
        }
    }
    return out;
}

Tokens synthetic_tokens(std::size_t count, std::uint32_t seed) {
    Tokens tokens;
    std::uint32_t state = seed * 2654435761U + 1U;
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 1664525U + 1013904223U;
        tokens.push_back(static_cast<infernix::TokenId>(1000U + (state >> 8U) % 30000U));
    }
    return tokens;
}

// The prefix cache's tap resume (test_prefix_cache_real's scenario) with the drafter's ring.
int tap_resume(const Setup& setup) {
    std::printf("== tap resume with the prefix cache (512-token calls)\n");
    infernix::EngineOptions options = engine_options(setup, Drafter::TextHead, 1);
    options.kv_capacity                       = infernix::KvCapacityPolicy::explicit_capacity(4608);
    options.prefill_chunk                     = 512;
    options.context_cache.host_capacity_bytes = 2ULL << 30;
    options.fast_path_candidates              = {infernix::FastPath::FreeCacheResume};
    infernix::Engine engine(std::move(options));
    // A leaves a tap at 1024; B shares 1,400 tokens, resumes at 1024 and prefills 1024..1536..1700.
    const Tokens a = synthetic_tokens(1500, 11);
    Tokens b(a.begin(), a.begin() + 1400);
    const Tokens suffix = synthetic_tokens(300, 12);
    b.insert(b.end(), suffix.begin(), suffix.end());
    const auto cold = engine.generate(engine.prepare_tokens(b), request(0.0F, 0, 96, false));
    (void)engine.generate(engine.prepare_tokens(a), request(0.0F, 0, 16, true));
    const auto warm = engine.generate(engine.prepare_tokens(b), request(0.0F, 0, 96, true));
    std::printf("  cold: ");
    print_acceptance(cold.speculative);
    std::printf("  warm: ");
    print_acceptance(warm.speculative);
    int failures = 0;
    failures += check(warm.reused_prompt_tokens == 1024,
                      "the request resumes at the tap (reused " + std::to_string(warm.reused_prompt_tokens) + ")");
    failures += check(warm.generated_token_ids == cold.generated_token_ids,
                      "the resumed request equals the cold run (" +
                          first_difference(warm.generated_token_ids, cold.generated_token_ids) + ")");
    failures += check(warm.speculative.drafted_tokens == cold.speculative.drafted_tokens &&
                          warm.speculative.accepted_tokens == cold.speculative.accepted_tokens,
                      "the restored ring drafts as the cold run's (drafted and accepted counts)");
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    Setup setup;
    setup.artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    setup.drafter  = std::getenv("INFERNIX_QWEN4_DFLASH2");
    setup.ngram    = std::getenv("INFERNIX_QWEN4_NGRAM");
    if (setup.artifact == nullptr || *setup.artifact == '\0' || setup.drafter == nullptr || *setup.drafter == '\0') {
        std::printf("SKIP: INFERNIX_QWEN4_ARTIFACT and INFERNIX_QWEN4_DFLASH2 are not both set\n");
        return 77;
    }
    bool greedy_only     = false;
    std::uint32_t trials = 1000, tokens = 8;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--greedy-only") { greedy_only = true; }
        if (arg == "--trials" && i + 1 < argc) { trials = static_cast<std::uint32_t>(std::stoul(argv[++i])); }
        if (arg == "--tokens" && i + 1 < argc) { tokens = static_cast<std::uint32_t>(std::max(2UL, std::stoul(argv[++i]))); }
    }
    try {
        int failures = 0;
        Outputs plain_greedy, plain, cold;
        Tokens prompt;
        {
            std::printf("== plain decoding\n");
            infernix::Engine engine(engine_options(setup, Drafter::None, greedy_only ? 1 : 8));
            plain_greedy = greedy(engine);
            prompt       = engine.tokenize_text(kPrompt);
            if (!greedy_only) {
                plain = sample(engine, prompt, 1.0F, 1, trials, tokens, 8);
                cold  = sample(engine, prompt, 0.8F, 3ULL * trials + 1, trials, tokens, 8);
            }
        }
        {
            std::printf("== DFlash2 %u, text head\n", kDrafts);
            infernix::Engine engine(engine_options(setup, Drafter::TextHead, 1));
            const Outputs drafted = greedy(engine);
            print_acceptance(drafted.speculative);
            for (std::size_t i = 0; i < drafted.tokens.size(); ++i) {
                failures += check(drafted.tokens[i] == plain_greedy.tokens[i],
                                  "greedy prompt " + std::to_string(i) + " equals plain decoding (" +
                                      first_difference(drafted.tokens[i], plain_greedy.tokens[i]) + ")");
            }
            failures += check(drafted.speculative.drafted_tokens > 0 &&
                                  4 * drafted.speculative.accepted_tokens > drafted.speculative.drafted_tokens,
                              "more than a quarter of the greedy drafts are accepted");
            if (!greedy_only) {
                const Outputs sampled = sample(engine, prompt, 1.0F, trials + 1ULL, trials, tokens, 1);
                print_acceptance(sampled.speculative);
                failures += check(sampled.speculative.accepted_tokens > 0 &&
                                      sampled.speculative.accepted_tokens < sampled.speculative.drafted_tokens,
                                  "sampled drafts are both accepted and rejected");
                const double alpha = 1e-3 / static_cast<double>(tokens + 1);
                failures += report("DFlash2 sampling", compare(plain.tokens, sampled.tokens, tokens), alpha, true);
                failures += report("plain T 0.8", compare(plain.tokens, cold.tokens, tokens), alpha, false);
            }
        }
        {
            std::printf("== DFlash2 %u, four lanes, four greedy requests in flight\n", kDrafts);
            infernix::Engine engine(engine_options(setup, Drafter::TextHead, 4));
            std::vector<infernix::GenerationHandle> handles;
            for (int copy = 0; copy < 2; ++copy) {
                for (const char* text : kGreedyPrompts) {
                    handles.push_back(engine.submit(engine.prepare_tokens(engine.tokenize_text(text)), request(0.0F, 0, 256)));
                }
            }
            for (std::size_t i = 0; i < handles.size(); ++i) {
                const Tokens got = handles[i].wait().generated_token_ids;
                failures += check(got == plain_greedy.tokens[i % 2],
                                  "concurrent request " + std::to_string(i) + " equals one-lane decoding (" +
                                      first_difference(got, plain_greedy.tokens[i % 2]) + ")");
            }
        }
        std::optional<infernix::Engine> proposal;
        try {
            proposal.emplace(engine_options(setup, Drafter::ProposalHead, 1));
        } catch (const std::exception& error) {
            std::printf("== DFlash2 %u with --lm-head-draft: not run (%s)\n", kDrafts, error.what());
        }
        if (proposal) {
            std::printf("== DFlash2 %u with --lm-head-draft (the proposal head)\n", kDrafts);
            const Outputs drafted = greedy(*proposal);
            print_acceptance(drafted.speculative);
            for (std::size_t i = 0; i < drafted.tokens.size(); ++i) {
                failures += check(drafted.tokens[i] == plain_greedy.tokens[i],
                                  "greedy prompt " + std::to_string(i) + " equals plain decoding (" +
                                      first_difference(drafted.tokens[i], plain_greedy.tokens[i]) + ")");
            }
            failures += check(4 * drafted.speculative.accepted_tokens > drafted.speculative.drafted_tokens,
                              "more than a quarter of the proposal-head drafts are accepted");
            proposal.reset();
        }
        failures += tap_resume(setup);
        std::printf(failures == 0 ? "all DFlash2 checks passed\n" : "%d DFlash2 check(s) failed\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
