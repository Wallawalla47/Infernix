// Qwen4Exp speculative decoding on the real artifacts against one shared plain-decoding reference:
// MTP drafts (one-hot and the sampled-mtp-drafts fast path) and a DFlash2 companion drafter
// (--spec dflash2 --draft-model). Skips unless INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact;
// the DFlash2 sections run when INFERNIX_QWEN4_DFLASH2 names its companion (tools.convert
// --components dflash2). INFERNIX_QWEN4_NGRAM names the n-gram volume (default <artifact>.ngram),
// INFERNIX_QWEN4_KV the KV storage (default int8). Every engine runs the exact route
// (--reproducible), whose seeded requests give the same tokens whatever runs beside them, so one
// plain arm serves every comparison.
//
//   reference     Plain decoding: two greedy prompts (256 tokens) and N seeded requests at T 1.0
//                 (top-k 20, top-p 0.95, G tokens each) and N at T 0.8, eight in flight.
//   distribution  Each sampled arm's token distribution at every output position, and of the first
//                 two tokens jointly, must match the plain arm's (two-sample chi-square, family
//                 alpha 0.001, Bonferroni), and plain decoding at T 0.8 must not (the test's power,
//                 checked at both arm sizes).
//   MTP           MTP 4 with one-hot argmax drafts (the exact route: the positive control) and with
//                 sampled drafts, N requests each on one lane: distributions; sampled drafts are
//                 both accepted and rejected, and greedy requests give the one-hot arm's tokens.
//   DFlash2       DFlash2 7 (text head, and the proposal head with --lm-head-draft when the
//                 artifact has one): greedy tokens equal plain decoding's and more than a quarter of
//                 the drafts are accepted; N / 2 sampled requests on one lane match the plain arm's
//                 first N / 2. Four greedy requests on a four-lane engine (plain rounds of several
//                 rows, whose catch-up writes several rings at once) give the one-lane tokens. With
//                 dflash2-pairs (two-row rounds share one proposal and one verify width), at DFlash2
//                 7 and with --draft-tokens auto: two greedy requests on a two-lane engine give the
//                 one-lane tokens and draft, and N / 2 sampled requests two at a time match. With the
//                 prefix cache (512-token calls), a request resuming from another's tap at 1024
//                 restores the drafter's context ring: its tokens and drafted and accepted counts
//                 equal a cold run's.
//
//   infernix_qwen4_exp_speculative_sampling_real_test [--greedy-only] [--trials N] [--tokens G]
//                                                     [--concurrency C]
//
// N = 2000, G = 8, C = 8 by default. Each request has its own seed, so batching does not change an
// arm's distribution, but only one-row rounds (two-row with a pairs fast path) draft: rounds of more
// rows decode plain. So the plain arms run C in flight and each speculative arm runs on the lanes its
// rounds draft with, and must draft at least one token per request on average. Each section prints
// its time.

#include "kv_cache_storage.h"
#include "infernix/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
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
constexpr std::uint32_t kDFlash2Drafts = 7;

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

// A section's wall time, printed when it ends.
class SectionTimer {
public:
    ~SectionTimer() {
        std::printf("  (%.1f s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count());
    }

private:
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
};

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
    return gamma_q(static_cast<double>(bins.size() - 1) / 2, statistic / 2);
}

// The family's p-values over the first `count` samples of each: each output position's token, and
// the first two tokens jointly.
std::vector<double> compare(const std::vector<Tokens>& a, const std::vector<Tokens>& b, std::size_t count,
                            std::size_t tokens) {
    std::vector<double> out;
    for (std::size_t i = 0; i <= tokens; ++i) {
        std::map<std::int64_t, std::uint64_t> ca, cb;
        const auto key = [&](const Tokens& s) -> std::int64_t {
            if (i == tokens) { return (static_cast<std::int64_t>(s.at(0)) << 32) | static_cast<std::uint32_t>(s.at(1)); }
            return s.at(i);
        };
        for (std::size_t s = 0; s < count; ++s) {
            ++ca[key(a.at(s))];
            ++cb[key(b.at(s))];
        }
        out.push_back(homogeneity(ca, cb));
    }
    return out;
}

int report(const std::string& name, const std::vector<double>& p, double alpha, bool expect_same) {
    const double least = *std::min_element(p.begin(), p.end());
    std::printf("  %s: p per position", name.c_str());
    for (std::size_t i = 0; i < p.size(); ++i) { std::printf(i + 1 == p.size() ? " | joint(0,1) %.3g" : " %.3g", p[i]); }
    std::printf("; least %.3g against %.3g\n", least, alpha);
    return check(expect_same ? least >= alpha : least < alpha,
                 name + (expect_same ? " matches plain sampling" : " differs from plain sampling (power)"));
}

struct Setup {
    const char* artifact = nullptr;
    const char* ngram    = nullptr;
    const char* drafter  = nullptr;
};

enum class Drafts { None, MtpOneHot, MtpSampled, DFlash2TextHead, DFlash2ProposalHead };

infernix::EngineOptions engine_options(const Setup& setup, Drafts drafts, std::uint32_t concurrency) {
    infernix::EngineOptions options;
    options.artifact_path = setup.artifact;
    if (setup.ngram != nullptr) { options.ngram_volume_path = setup.ngram; }
    options.max_context          = 4096;
    const char* kv               = std::getenv("INFERNIX_QWEN4_KV");
    options.kv_cache             = kv != nullptr ? infernix::test::parse_kv_cache_storage(kv) : infernix::KvCacheStorage::Int8Group64;
    options.max_concurrency      = concurrency;
    options.max_pending_requests = 2 * concurrency;
    options.fast_paths           = false;
    switch (drafts) {
    case Drafts::None: break;
    case Drafts::MtpSampled: options.fast_path_candidates = {infernix::FastPath::SampledMtpDrafts}; [[fallthrough]];
    case Drafts::MtpOneHot:
        options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 4;
        break;
    case Drafts::DFlash2ProposalHead: options.speculative.proposal_head = infernix::ProposalHead::Optimized; [[fallthrough]];
    case Drafts::DFlash2TextHead:
        options.speculative.backend      = infernix::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = kDFlash2Drafts;
        options.speculative.draft_model  = setup.drafter;
        break;
    }
    return options;
}

infernix::RequestOptions request(float temperature, std::uint64_t seed, std::uint32_t tokens, bool reuse) {
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
    infernix::SpeculativeStats speculative; // summed over the requests
};

void add(infernix::SpeculativeStats& sum, const infernix::SpeculativeStats& one) {
    sum.rounds += one.rounds;
    sum.drafted_tokens += one.drafted_tokens;
    sum.accepted_tokens += one.accepted_tokens;
}

Outputs greedy(infernix::Engine& engine) {
    Outputs out;
    for (const char* text : kGreedyPrompts) {
        const auto result =
            engine.generate(engine.prepare_tokens(engine.tokenize_text(text)), request(0.0F, 0, 256, false));
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

// `count` seeded requests from `first_seed` on an engine of `lanes` lanes. One lane takes them back to
// back, the next queued while one decodes; several take them in waves of `lanes` started together.
Outputs sample(infernix::Engine& engine, const Tokens& prompt, float temperature, std::uint64_t first_seed,
               std::uint32_t count, std::uint32_t tokens, std::uint32_t lanes, bool reuse) {
    Outputs out;
    std::deque<infernix::GenerationHandle> handles;
    const std::uint32_t queued = lanes == 1 ? 2 : lanes;
    std::uint32_t submitted    = 0;
    while (out.tokens.size() < count) {
        const bool open = lanes == 1 || handles.empty();
        for (; open && submitted < count && handles.size() < queued; ++submitted) {
            handles.push_back(
                engine.submit(engine.prepare_tokens(prompt), request(temperature, first_seed + submitted, tokens, reuse)));
        }
        const infernix::GenerationResult result = handles.front().wait();
        handles.pop_front();
        if (result.generated_token_ids.size() != tokens) {
            throw std::runtime_error("a sampled request generated " + std::to_string(result.generated_token_ids.size()) +
                                     " tokens instead of " + std::to_string(tokens));
        }
        out.tokens.push_back(result.generated_token_ids);
        add(out.speculative, result.speculative);
    }
    return out;
}

// Speculative rounds have one row (or two, with a pairs fast path): rounds of more rows decode plain.
// A sampled arm whose rounds stop drafting would compare plain decoding with itself, so each must
// draft at least one token per request on average.
int drafts_per_request(const char* arm, const infernix::SpeculativeStats& s, std::uint32_t requests) {
    return check(s.drafted_tokens >= requests, std::string(arm) + " draft at least one token per request (" +
                                                   std::to_string(s.drafted_tokens) + " drafts, " +
                                                   std::to_string(requests) + " requests)");
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
    const SectionTimer timer;
    infernix::EngineOptions options = engine_options(setup, Drafts::DFlash2TextHead, 1);
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

struct Reference {
    Outputs greedy;
    Outputs plain; // T 1.0, seeds 1..N
    Outputs cold;  // T 0.8, seeds 3N + 1..4N
    Tokens prompt;
};

// MTP 4 with one-hot drafts (the positive control) and with sampled drafts, one lane each so every
// round drafts.
int mtp(const Setup& setup, const Reference& ref, bool greedy_only, std::uint32_t trials, std::uint32_t tokens,
        double alpha) {
    int failures = 0;
    Outputs greedy_one_hot, greedy_sampled;
    {
        std::printf("== MTP 4, one-hot drafts (one lane)\n");
        const SectionTimer timer;
        infernix::Engine engine(engine_options(setup, Drafts::MtpOneHot, 1));
        greedy_one_hot = greedy(engine);
        if (!greedy_only) {
            const Outputs one_hot = sample(engine, ref.prompt, 1.0F, trials + 1ULL, trials, tokens, 1, true);
            print_acceptance(one_hot.speculative);
            failures += drafts_per_request("one-hot MTP requests", one_hot.speculative, trials);
            failures += report("one-hot drafts", compare(ref.plain.tokens, one_hot.tokens, trials, tokens), alpha, true);
        }
    }
    {
        std::printf("== MTP 4, sampled drafts (one lane)\n");
        const SectionTimer timer;
        infernix::Engine engine(engine_options(setup, Drafts::MtpSampled, 1));
        greedy_sampled = greedy(engine);
        if (!greedy_only) {
            const Outputs drafts = sample(engine, ref.prompt, 1.0F, 2ULL * trials + 1, trials, tokens, 1, true);
            print_acceptance(drafts.speculative);
            failures += drafts_per_request("sampled MTP requests", drafts.speculative, trials);
            failures += check(drafts.speculative.accepted_tokens > 0 &&
                                  drafts.speculative.accepted_tokens < drafts.speculative.drafted_tokens,
                              "sampled drafts are both accepted and rejected");
            failures += report("sampled drafts", compare(ref.plain.tokens, drafts.tokens, trials, tokens), alpha, true);
        }
    }
    for (std::size_t i = 0; i < greedy_one_hot.tokens.size(); ++i) {
        failures += check(greedy_sampled.tokens[i] == greedy_one_hot.tokens[i],
                          "greedy prompt " + std::to_string(i) + ": sampled MTP drafts give the one-hot tokens");
    }
    return failures;
}

// DFlash2 7: greedy and sampled on one lane, four lanes, dflash2-pairs, the proposal head and the tap
// resume. Its sampled arms take `count` requests against the plain arm's first `count`.
int dflash2(const Setup& setup, const Reference& ref, bool greedy_only, std::uint32_t count, std::uint32_t tokens,
            double alpha) {
    int failures = 0;
    {
        std::printf("== DFlash2 %u, text head\n", kDFlash2Drafts);
        const SectionTimer timer;
        infernix::Engine engine(engine_options(setup, Drafts::DFlash2TextHead, 1));
        const Outputs drafted = greedy(engine);
        print_acceptance(drafted.speculative);
        for (std::size_t i = 0; i < drafted.tokens.size(); ++i) {
            failures += check(drafted.tokens[i] == ref.greedy.tokens[i],
                              "greedy prompt " + std::to_string(i) + " equals plain decoding (" +
                                  first_difference(drafted.tokens[i], ref.greedy.tokens[i]) + ")");
        }
        failures += check(drafted.speculative.drafted_tokens > 0 &&
                              4 * drafted.speculative.accepted_tokens > drafted.speculative.drafted_tokens,
                          "more than a quarter of the greedy drafts are accepted");
        if (!greedy_only) {
            const Outputs sampled = sample(engine, ref.prompt, 1.0F, count + 1ULL, count, tokens, 1, false);
            print_acceptance(sampled.speculative);
            failures += drafts_per_request("DFlash2 requests", sampled.speculative, count);
            failures += check(sampled.speculative.accepted_tokens > 0 &&
                                  sampled.speculative.accepted_tokens < sampled.speculative.drafted_tokens,
                              "sampled drafts are both accepted and rejected");
            failures += report("DFlash2 sampling", compare(ref.plain.tokens, sampled.tokens, count, tokens), alpha, true);
        }
    }
    {
        std::printf("== DFlash2 %u, four lanes, four greedy requests in flight\n", kDFlash2Drafts);
        const SectionTimer timer;
        infernix::Engine engine(engine_options(setup, Drafts::DFlash2TextHead, 4));
        std::vector<infernix::GenerationHandle> handles;
        for (int copy = 0; copy < 2; ++copy) {
            for (const char* text : kGreedyPrompts) {
                handles.push_back(
                    engine.submit(engine.prepare_tokens(engine.tokenize_text(text)), request(0.0F, 0, 256, false)));
            }
        }
        for (std::size_t i = 0; i < handles.size(); ++i) {
            const Tokens got = handles[i].wait().generated_token_ids;
            failures += check(got == ref.greedy.tokens[i % 2],
                              "concurrent request " + std::to_string(i) + " equals one-lane decoding (" +
                                  first_difference(got, ref.greedy.tokens[i % 2]) + ")");
        }
    }
    // dflash2-pairs: both greedy prompts in flight together decode in two-row rounds, which stay on
    // the column-invariant routes (at most 2 x 3 columns), so each equals its one-lane run.
    for (const bool automatic : {false, true}) {
        std::printf("== DFlash2 %s, dflash2-pairs, two lanes, two greedy requests in flight\n",
                    automatic ? "auto" : "7 (pair rounds verify 2)");
        const SectionTimer timer;
        infernix::EngineOptions options       = engine_options(setup, Drafts::DFlash2TextHead, 2);
        options.fast_path_candidates          = {infernix::FastPath::DFlash2Pairs};
        options.speculative.draft_tokens_auto = automatic;
        infernix::Engine engine(std::move(options));
        const auto& paths = engine.load_summary().fast_paths;
        failures += check(std::find(paths.begin(), paths.end(), "dflash2-pairs") != paths.end() &&
                              !engine.load_summary().request_invariant,
                          "the engine runs dflash2-pairs and reports its requests coupled");
        std::vector<infernix::GenerationHandle> handles;
        for (const char* text : kGreedyPrompts) {
            handles.push_back(engine.submit(engine.prepare_tokens(engine.tokenize_text(text)), request(0.0F, 0, 256, false)));
        }
        infernix::SpeculativeStats paired;
        for (std::size_t i = 0; i < handles.size(); ++i) {
            const infernix::GenerationResult result = handles[i].wait();
            add(paired, result.speculative);
            failures += check(result.generated_token_ids == ref.greedy.tokens[i],
                              "paired request " + std::to_string(i) + " equals one-lane decoding (" +
                                  first_difference(result.generated_token_ids, ref.greedy.tokens[i]) + ")");
        }
        print_acceptance(paired);
        // One-row rounds alone (one request finishing first) would draft too; the pair rounds' share
        // of drafts is not observable from the public stats, so this checks only that drafting runs.
        failures += check(paired.drafted_tokens > 0 && paired.accepted_tokens > 0, "the paired requests draft and accept");
        if (!greedy_only) {
            const Outputs sampled = sample(engine, ref.prompt, 1.0F, 5ULL * count + 1, count, tokens, 2, false);
            print_acceptance(sampled.speculative);
            failures += drafts_per_request("paired requests", sampled.speculative, count);
            failures += report(automatic ? "DFlash2 auto pairs sampling" : "DFlash2 pairs sampling",
                               compare(ref.plain.tokens, sampled.tokens, count, tokens), alpha, true);
        }
    }
    std::optional<infernix::Engine> proposal;
    try {
        proposal.emplace(engine_options(setup, Drafts::DFlash2ProposalHead, 1));
    } catch (const std::exception& error) {
        std::printf("== DFlash2 %u with --lm-head-draft: not run (%s)\n", kDFlash2Drafts, error.what());
    }
    if (proposal) {
        std::printf("== DFlash2 %u with --lm-head-draft (the proposal head)\n", kDFlash2Drafts);
        const SectionTimer timer;
        const Outputs drafted = greedy(*proposal);
        print_acceptance(drafted.speculative);
        for (std::size_t i = 0; i < drafted.tokens.size(); ++i) {
            failures += check(drafted.tokens[i] == ref.greedy.tokens[i],
                              "greedy prompt " + std::to_string(i) + " equals plain decoding (" +
                                  first_difference(drafted.tokens[i], ref.greedy.tokens[i]) + ")");
        }
        failures += check(4 * drafted.speculative.accepted_tokens > drafted.speculative.drafted_tokens,
                          "more than a quarter of the proposal-head drafts are accepted");
        proposal.reset();
    }
    failures += tap_resume(setup);
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    Setup setup;
    setup.artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    setup.drafter  = std::getenv("INFERNIX_QWEN4_DFLASH2");
    setup.ngram    = std::getenv("INFERNIX_QWEN4_NGRAM");
    if (setup.artifact == nullptr || *setup.artifact == '\0') {
        std::printf("SKIP: INFERNIX_QWEN4_ARTIFACT is not set\n");
        return 77;
    }
    bool greedy_only     = false;
    std::uint32_t trials = 2000, tokens = 8, concurrency = 8;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--greedy-only") { greedy_only = true; }
        if (arg == "--concurrency" && i + 1 < argc) {
            concurrency = static_cast<std::uint32_t>(std::clamp(std::stoul(argv[++i]), 1UL, 8UL));
        }
        if (arg == "--trials" && i + 1 < argc) { trials = std::max(2U, static_cast<std::uint32_t>(std::stoul(argv[++i]))); }
        if (arg == "--tokens" && i + 1 < argc) { tokens = static_cast<std::uint32_t>(std::max(2UL, std::stoul(argv[++i]))); }
    }
    try {
        int failures       = 0;
        const double alpha = 1e-3 / static_cast<double>(tokens + 1); // per test of each family
        const std::uint32_t half = trials / 2;                       // the DFlash2 arms' size
        Reference ref;
        {
            std::printf("== plain decoding (C = %u): %u requests at T 1.0 and at T 0.8, %u tokens\n", concurrency, trials,
                        tokens);
            const SectionTimer timer;
            infernix::Engine engine(engine_options(setup, Drafts::None, greedy_only ? 1 : concurrency));
            ref.greedy = greedy(engine);
            ref.prompt = engine.tokenize_text(kPrompt);
            if (!greedy_only) {
                ref.plain = sample(engine, ref.prompt, 1.0F, 1, trials, tokens, concurrency, true);
                ref.cold  = sample(engine, ref.prompt, 0.8F, 3ULL * trials + 1, trials, tokens, concurrency, true);
                failures += report("plain T 0.8", compare(ref.plain.tokens, ref.cold.tokens, trials, tokens), alpha, false);
                failures += report("plain T 0.8 (first " + std::to_string(half) + ")",
                                   compare(ref.plain.tokens, ref.cold.tokens, half, tokens), alpha, false);
            }
        }
        failures += mtp(setup, ref, greedy_only, trials, tokens, alpha);
        if (setup.drafter != nullptr && *setup.drafter != '\0') {
            failures += dflash2(setup, ref, greedy_only, half, tokens, alpha);
        } else {
            std::printf("== DFlash2: not run (INFERNIX_QWEN4_DFLASH2 is not set)\n");
        }
        std::printf(failures == 0 ? "all speculative sampling checks passed\n" : "%d speculative sampling check(s) failed\n",
                    failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
