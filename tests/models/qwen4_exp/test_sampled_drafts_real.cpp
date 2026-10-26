// Qwen4Exp sampled MTP drafts (the sampled-mtp-drafts fast path) on the real artifact: speculative
// rejection sampling against the drafter's sampled proposals keeps the sampler's output
// distribution, and greedy requests are unchanged. Skips unless INFERNIX_QWEN4_ARTIFACT names a
// Qwen4Exp artifact (INFERNIX_QWEN4_NGRAM: its n-gram volume, default <artifact>.ngram).
// INFERNIX_QWEN4_KV selects the KV storage (default int8). Every engine runs the exact route.
//
//   distribution  N seeded requests per arm (T 1.0, top-k 20, top-p 0.95, G tokens each, disjoint
//                 seeds) from plain decoding (the sampler itself), MTP with one-hot argmax drafts
//                 (the exact route: the positive control) and MTP with sampled drafts. Each MTP
//                 arm's token distribution at every output position, and of the first two tokens
//                 jointly, must match plain decoding's (two-sample chi-square, family alpha 0.001,
//                 Bonferroni), and plain decoding at T 0.8 must not (the test's power).
//   greedy        Two prompts, 256 greedy tokens: sampled drafts give the one-hot arm's tokens.
//
//   infernix_qwen4_exp_sampled_drafts_real_test [--greedy-only] [--trials N] [--tokens G] [--concurrency C]
//
// Every arm keeps C = 8 requests in flight by default (the sparse acceptance takes up to 8 rows):
// the exact route is request-invariant and each request has its own seed, so batching does not
// change an arm's distribution, and the MTP arms finish about as fast as plain decoding.

#include "kv_cache_storage.h"
#include "infernix/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Samples = std::vector<std::vector<infernix::TokenId>>; // one output per request

constexpr const char* kPrompt =
    "The old lighthouse keeper climbed the stairs one last time before the storm. At the top, he";
constexpr const char* kGreedyPrompts[] = {
    "Write a Python function that returns the n-th Fibonacci number, then explain its complexity.",
    "The three most important causes of the French Revolution were"};

int check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    return ok ? 0 : 1;
}

// Q(a, x), the regularized upper incomplete gamma function (series below a + 1, continued
// fraction above), for chi-square tail probabilities: p = Q(df / 2, statistic / 2).
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

// Two-sample chi-square homogeneity test over categorical keys; categories with fewer than 10
// observations in both samples together are pooled into one bin. Returns the p-value.
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
    const double df = static_cast<double>(bins.size() - 1);
    return gamma_q(df / 2, statistic / 2);
}

// The family's p-values: each output position's token, and the first two tokens jointly.
std::vector<double> compare(const Samples& a, const Samples& b, std::size_t tokens) {
    std::vector<double> out;
    for (std::size_t i = 0; i <= tokens; ++i) {
        std::map<std::int64_t, std::uint64_t> ca, cb;
        const auto key = [&](const std::vector<infernix::TokenId>& s) -> std::int64_t {
            if (i == tokens) { return (static_cast<std::int64_t>(s.at(0)) << 32) | static_cast<std::uint32_t>(s.at(1)); }
            return s.at(i);
        };
        for (const auto& s : a) { ++ca[key(s)]; }
        for (const auto& s : b) { ++cb[key(s)]; }
        out.push_back(homogeneity(ca, cb));
    }
    return out;
}

infernix::EngineOptions engine_options(const char* artifact, const char* ngram, bool mtp, bool sampled_drafts,
                                       std::uint32_t concurrency) {
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    if (ngram != nullptr) { options.ngram_volume_path = ngram; }
    options.max_context     = 4096;
    const char* kv          = std::getenv("INFERNIX_QWEN4_KV");
    options.kv_cache        = kv != nullptr ? infernix::test::parse_kv_cache_storage(kv) : infernix::KvCacheStorage::Int8Group64;
    options.max_concurrency = concurrency;
    options.max_pending_requests = 2 * concurrency;
    options.fast_paths      = false;
    if (sampled_drafts) { options.fast_path_candidates = {infernix::FastPath::SampledMtpDrafts}; }
    if (mtp) {
        options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 4;
    }
    return options;
}

infernix::RequestOptions request(float temperature, std::uint64_t seed, std::uint32_t tokens) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = temperature;
    options.execution.sampling.top_k          = 20;
    options.execution.sampling.top_p          = 0.95F;
    options.execution.sampling.seed           = seed;
    options.stop.include_model_defaults       = false;
    return options;
}

struct Arm {
    Samples samples;
    infernix::SpeculativeStats speculative; // summed over the arm's requests
};

// `count` seeded requests from `first_seed`, `concurrency` in flight.
Arm sample(infernix::Engine& engine, const std::vector<infernix::TokenId>& prompt, float temperature,
           std::uint64_t first_seed, std::uint32_t count, std::uint32_t tokens, std::uint32_t concurrency) {
    Arm arm;
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
            arm.samples.push_back(result.generated_token_ids);
            arm.speculative.rounds += result.speculative.rounds;
            arm.speculative.drafted_tokens += result.speculative.drafted_tokens;
            arm.speculative.accepted_tokens += result.speculative.accepted_tokens;
        }
    }
    return arm;
}

std::vector<std::vector<infernix::TokenId>> greedy(infernix::Engine& engine) {
    std::vector<std::vector<infernix::TokenId>> out;
    for (const char* text : kGreedyPrompts) {
        out.push_back(engine.generate(engine.prepare_tokens(engine.tokenize_text(text)), request(0.0F, 0, 256))
                          .generated_token_ids);
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

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::printf("SKIP: INFERNIX_QWEN4_ARTIFACT is not set\n");
        return 77;
    }
    const char* ngram   = std::getenv("INFERNIX_QWEN4_NGRAM");
    bool greedy_only    = false;
    std::uint32_t trials = 2000, tokens = 8, concurrency = 8;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--greedy-only") { greedy_only = true; }
        if (arg == "--concurrency" && i + 1 < argc) {
            concurrency = static_cast<std::uint32_t>(std::clamp(std::stoul(argv[++i]), 1UL, 8UL));
        }
        if (arg == "--trials" && i + 1 < argc) { trials = static_cast<std::uint32_t>(std::stoul(argv[++i])); }
        if (arg == "--tokens" && i + 1 < argc) { tokens = static_cast<std::uint32_t>(std::max(2UL, std::stoul(argv[++i]))); }
    }
    try {
        int failures       = 0;
        const double alpha = 1e-3 / static_cast<double>(tokens + 1); // per test of each family
        Arm plain, cold;
        std::vector<infernix::TokenId> prompt;
        if (!greedy_only) {
            std::printf("== plain decoding (C = %u): %u requests at T 1.0 and at T 0.8, %u tokens\n", concurrency,
                        trials, tokens);
            infernix::Engine engine(engine_options(artifact, ngram, false, false, concurrency));
            prompt = engine.tokenize_text(kPrompt);
            plain  = sample(engine, prompt, 1.0F, 1, trials, tokens, concurrency);
            cold   = sample(engine, prompt, 0.8F, 3ULL * trials + 1, trials, tokens, concurrency);
        }
        std::vector<std::vector<infernix::TokenId>> greedy_one_hot, greedy_sampled;
        {
            std::printf("== MTP 4, one-hot drafts (C = %u)\n", concurrency);
            infernix::Engine engine(engine_options(artifact, ngram, true, false, concurrency));
            greedy_one_hot = greedy(engine);
            if (!greedy_only) {
                const Arm one_hot = sample(engine, prompt, 1.0F, trials + 1ULL, trials, tokens, concurrency);
                std::printf("  rounds %llu, accepted %llu of %llu drafts\n",
                            static_cast<unsigned long long>(one_hot.speculative.rounds),
                            static_cast<unsigned long long>(one_hot.speculative.accepted_tokens),
                            static_cast<unsigned long long>(one_hot.speculative.drafted_tokens));
                failures += report("one-hot drafts", compare(plain.samples, one_hot.samples, tokens), alpha, true);
            }
        }
        {
            std::printf("== MTP 4, sampled drafts (C = %u)\n", concurrency);
            infernix::Engine engine(engine_options(artifact, ngram, true, true, concurrency));
            greedy_sampled = greedy(engine);
            if (!greedy_only) {
                const Arm drafts = sample(engine, prompt, 1.0F, 2ULL * trials + 1, trials, tokens, concurrency);
                std::printf("  rounds %llu, accepted %llu of %llu drafts\n",
                            static_cast<unsigned long long>(drafts.speculative.rounds),
                            static_cast<unsigned long long>(drafts.speculative.accepted_tokens),
                            static_cast<unsigned long long>(drafts.speculative.drafted_tokens));
                failures += check(drafts.speculative.accepted_tokens > 0 &&
                                      drafts.speculative.accepted_tokens < drafts.speculative.drafted_tokens,
                                  "sampled drafts are both accepted and rejected");
                failures += report("sampled drafts", compare(plain.samples, drafts.samples, tokens), alpha, true);
            }
        }
        if (!greedy_only) { failures += report("plain T 0.8", compare(plain.samples, cold.samples, tokens), alpha, false); }
        for (std::size_t i = 0; i < greedy_one_hot.size(); ++i) {
            failures += check(greedy_sampled[i] == greedy_one_hot[i],
                              "greedy prompt " + std::to_string(i) + ": sampled drafts give the one-hot tokens");
        }
        std::printf(failures == 0 ? "all sampled-draft checks passed\n" : "%d sampled-draft check(s) failed\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
