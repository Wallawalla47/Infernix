// Qwen4Exp n-gram trees (--ngram-tree-nodes; docs/maintainer/tree-verification.md, "N-gram trees") on a real
// artifact. Each prompt shows a code fragment twice, as version 1 and as version 2 (seven edits, each after
// more than 64 tokens both share), and asks for one of them again. While a copy runs through shared text both
// versions match equally and the newer, version 2, is the copy's main chain, so a copy of version 1 meets its
// other source as a side branch at every edit. Greedy streams with n-gram copies verified as trees and as
// chains (--ngram-draft-tokens 15, minimum match 12); both arms take every copy, to a two-path tree's depth, and
// the tree arm verifies every eligible tree (testing::set_ngram_policy_override), as the cost policy alone rarely
// would on an offloaded artifact:
//
//   rounds      the tree arm verifies n-gram tree rounds, and some accept the other source's branch;
//   exactness   each arm's stream scored on its own prefix by the lossless route (the exact model, BF16 KV, one
//               prefill): the tree arm has at most kExtraMisses more tokens whose regret (the reference's top
//               log-probability less the token's) exceeds 1 nat than the most any chain arm of the run has on
//               that prompt (copies alone and with DFlash2 are both chain decodings of the route, and differ by
//               up to 3 on NVIDIA Dense8), and none above kStateRegret nats from its first difference with its
//               own chain arm on (before it, its tokens are the chain arm's choices: Dense8's chain arms hold a
//               5.15-nat one, from its 4-bit expert activations). Verification widths differ between the arms,
//               so the fast route's near-ties may flip (FP4-Local: 1.9-2.2 nats, the chain arm's first token
//               among them); a state, gather or compaction error leaves tokens far below the exact model's
//               choice, and a state error keeps doing so. The first difference between the arms is reported
//               with both tokens' regrets.
//
// Without a drafter (n-gram copies alone), and with the DFlash2 companion when INFERNIX_QWEN4_DFLASH2 names it.
// Skips unless INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact; INFERNIX_QWEN4_NGRAM names its n-gram volume
// (default <artifact>.ngram), INFERNIX_QWEN4_KV the KV storage (default int8).

#include "kv_cache_storage.h"
#include "models/qwen4_exp/program/ngram_tree_testing.h"
#include "engine_host.h"
#include "infernix/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <string>
#include <vector>

namespace {

using Tokens = std::vector<infernix::TokenId>;

constexpr std::uint32_t kOutputs     = 600;
constexpr std::uint32_t kContext      = 4096;
constexpr std::uint32_t kExtraMisses  = 2;
constexpr double kStateRegret         = 5.0;

// Per-token regrets of `generated` after `prompt` under the lossless reference.
std::vector<double> regrets(infernix::Engine& reference, const Tokens& prompt, const Tokens& generated) {
    Tokens sequence = prompt;
    sequence.insert(sequence.end(), generated.begin(), generated.end());
    const infernix::ScoreResult scored =
        reference.score_tokens(std::move(sequence), static_cast<std::uint32_t>(prompt.size()), {.top_k = 1});
    std::vector<double> out(generated.size());
    for (std::size_t i = 0; i < generated.size(); ++i) {
        out[i] = static_cast<double>(scored.top_logprobs[i]) - static_cast<double>(scored.logprobs[i]);
    }
    return out;
}

constexpr const char* kVersionOne = R"(def merge_intervals(intervals):
    """Merge overlapping intervals and return them sorted by start."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda pair: pair[0])
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [tuple(pair) for pair in merged]


def total_coverage(intervals, limit=10):
    """Total length covered by the intervals, counting at most `limit` of them."""
    covered = 0
    for start, end in merge_intervals(intervals)[:limit]:
        covered += end - start
    return covered


def largest_gap(intervals):
    """The widest gap between consecutive merged intervals, 0 when they touch."""
    merged = merge_intervals(intervals)
    widest = 0
    for (_, previous_end), (next_start, _) in zip(merged, merged[1:]):
        widest = max(widest, next_start - previous_end)
    return widest


def clip_intervals(intervals, low, high):
    """Clip every interval to [low, high] and drop the ones left empty."""
    clipped = []
    for start, end in intervals:
        start, end = max(start, low), min(end, high)
        if start < end:
            clipped.append((start, end))
    return clipped


def intersect(first, second):
    """The intervals covered by both lists, merged and sorted by start."""
    result = []
    for a_start, a_end in merge_intervals(first):
        for b_start, b_end in merge_intervals(second):
            start, end = max(a_start, b_start), min(a_end, b_end)
            if start < end:
                result.append((start, end))
    return merge_intervals(result)


def summary(intervals):
    """A one-line description of the merged intervals."""
    merged = merge_intervals(intervals)
    count = len(merged)
    covered = total_coverage(merged, limit=count)
    return f"{count} intervals covering {covered} units, widest gap {largest_gap(merged)}"
)";

constexpr const char* kVersionTwo = R"(def merge_intervals(intervals):
    """Merge overlapping intervals and return them sorted by start."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda pair: pair[0])
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        last = merged[-1]
        if start < last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [tuple(pair) for pair in merged]


def total_coverage(intervals, limit=12):
    """Total length covered by the intervals, counting at most `limit` of them."""
    covered = 0
    for start, end in merge_intervals(intervals)[:limit]:
        covered += end - start
    return covered


def largest_gap(intervals):
    """The widest gap between consecutive merged intervals, 0 when they touch."""
    merged = merge_intervals(intervals)
    widest = 0
    for (_, previous_end), (next_start, _) in zip(merged, merged[1:]):
        widest = max(widest, next_start - previous_end - 1)
    return widest


def clip_intervals(intervals, low, high):
    """Clip every interval to [low, high] and drop the ones left empty."""
    clipped = []
    for start, end in intervals:
        start, end = max(start, low), min(end, high)
        if start <= end:
            clipped.append((start, end))
    return clipped


def intersect(first, second):
    """The intervals covered by both lists, merged and sorted by start."""
    result = []
    for a_start, a_end in merge_intervals(first):
        for b_start, b_end in merge_intervals(second):
            start, end = max(a_start, b_start), min(a_end, b_end)
            if start < end:
                result.append([start, end])
    return merge_intervals(result)


def summary(intervals):
    """A one-line description of the merged intervals."""
    merged = merge_intervals(intervals)
    count = len(merged) or 1
    covered = total_coverage(merged, limit=count)
    return f"{count} intervals covering {covered} units, largest gap {largest_gap(merged)}"
)";

int check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    return ok ? 0 : 1;
}

std::string prompt(int variant) {
    // Two prompts: ask for version 1 (the older source, a side branch), or for version 2 (the main chain).
    std::string text = std::string("Two versions of intervals.py follow.\n\nVersion 1:\n```python\n") + kVersionOne +
                       "```\n\nVersion 2:\n```python\n" + kVersionTwo + "```\n\n";
    text += variant == 0 ? "Write version 1 again, exactly as it is above:\n```python\ndef merge_intervals(intervals):\n"
                         : "Write version 2 again, exactly as it is above:\n```python\ndef merge_intervals(intervals):\n";
    return text;
}

struct Arm {
    std::vector<Tokens> streams;
    std::uint64_t tree_rounds = 0, side_rounds = 0, side_drafts = 0, ngram_rounds = 0;
};

Arm run(const char* artifact, const char* ngram, const char* draft_model, std::uint32_t tree_nodes,
        const std::vector<Tokens>& prompts) {
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    if (ngram != nullptr) { options.ngram_volume_path = ngram; }
    const char* kv  = std::getenv("INFERNIX_QWEN4_KV");
    options.kv_cache = kv != nullptr ? infernix::test::parse_kv_cache_storage(kv) : infernix::KvCacheStorage::Int8Group64;
    options.max_context     = kContext;
    options.max_concurrency = 1;
    options.speculative.ngram_draft_tokens = 15;
    options.speculative.ngram_min_match   = 12;
    options.speculative.ngram_tree_nodes  = tree_nodes;
    if (draft_model != nullptr) {
        options.speculative.backend      = infernix::SpeculativeBackend::DFlash2;
        options.speculative.draft_model  = draft_model;
        options.speculative.draft_tokens = 7;
    }
    // Both arms take every copy, to a two-path tree's depth; the tree arm verifies every eligible tree.
    infernix::models::qwen4_exp::testing::set_ngram_policy_override(true);
    infernix::Engine engine(options, infernix::test::engine_host());
    Arm arm;
    for (const Tokens& tokens : prompts) {
        infernix::RequestOptions request;
        request.execution.requested_output_tokens = kOutputs;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.allow_prefix_reuse      = false;
        request.stop.include_model_defaults       = false;
        const infernix::GenerationResult result = engine.submit(engine.prepare_tokens(tokens), request).wait();
        arm.streams.push_back(result.generated_token_ids);
        arm.tree_rounds += result.speculative.ngram_tree_rounds;
        arm.side_rounds += result.speculative.ngram_tree_side_rounds;
        arm.side_drafts += result.speculative.ngram_tree_side_accepted_tokens;
        arm.ngram_rounds += result.speculative.ngram_rounds;
    }
    return arm;
}

// The lossless reference, as infernix_decode_quality_judge runs it: no fast path, BF16 KV, one prefill.
infernix::EngineOptions reference_options(const char* artifact, const char* ngram) {
    infernix::EngineOptions scoring;
    scoring.artifact_path = artifact;
    scoring.purpose       = infernix::EnginePurpose::CausalScoring;
    scoring.max_context   = kContext;
    scoring.prefill_chunk = kContext;
    scoring.kv_cache      = infernix::KvCacheStorage::BFloat16;
    scoring.lossless      = true;
    scoring.fast_paths    = false;
    if (ngram != nullptr) { scoring.ngram_volume_path = ngram; }
    return scoring;
}

std::uint32_t misses(const std::vector<double>& r) {
    return static_cast<std::uint32_t>(std::count_if(r.begin(), r.end(), [](double x) { return x > 1.0; }));
}

// `chains`: every chain arm of the run (`chain`, the tree arm's own, among them), each a chain decoding of the same
// route; the tree arm's misses are held to the most of theirs, since chain arms of different drafters differ from
// each other too (NVIDIA Dense8, whose 4-bit expert activations flip more near-ties: 7 against 4 on one prompt).
int compare(const char* label, infernix::Engine& reference, const std::vector<Tokens>& prompts, const Arm& tree,
            const Arm& chain, const std::vector<const Arm*>& chains) {
    int failures = 0;
    std::printf("%s: tree arm %llu n-gram tree rounds, %llu took the other source (%llu drafts), %llu copy rounds; "
                "chain arm %llu copy rounds\n",
                label, static_cast<unsigned long long>(tree.tree_rounds), static_cast<unsigned long long>(tree.side_rounds),
                static_cast<unsigned long long>(tree.side_drafts), static_cast<unsigned long long>(tree.ngram_rounds),
                static_cast<unsigned long long>(chain.ngram_rounds));
    failures += check(tree.tree_rounds > 0, std::string(label) + ": the tree arm verified n-gram trees");
    failures += check(tree.side_rounds > 0 && tree.side_drafts >= tree.side_rounds,
                      std::string(label) + ": some tree round accepted the other source's branch");
    for (std::size_t p = 0; p < prompts.size(); ++p) {
        const std::string name = std::string(label) + " prompt " + std::to_string(p);
        const Tokens& a        = tree.streams[p];
        const Tokens& b        = chain.streams[p];
        if (a.size() != b.size()) {
            failures += check(false, name + ": one stream stopped early");
            continue;
        }
        if (a == b) {
            failures += check(true, name + ": identical streams");
            continue;
        }
        std::size_t at = 0;
        while (a[at] == b[at]) { ++at; }
        const std::vector<double> tree_regret  = regrets(reference, prompts[p], a);
        const std::vector<double> chain_regret = regrets(reference, prompts[p], b);
        std::uint32_t chain_misses             = misses(chain_regret);
        for (const Arm* other : chains) {
            if (other != &chain && p < other->streams.size()) {
                chain_misses = std::max(chain_misses, misses(regrets(reference, prompts[p], other->streams[p])));
            }
        }
        // Before the first difference the tree arm's tokens are the chain arm's own choices.
        const double tree_worst    = *std::max_element(tree_regret.begin() + static_cast<std::ptrdiff_t>(at), tree_regret.end());
        const double prefix_worst  = at == 0 ? 0.0 : *std::max_element(tree_regret.begin(), tree_regret.begin() +
                                                                                               static_cast<std::ptrdiff_t>(at));
        const double chain_worst   = *std::max_element(chain_regret.begin(), chain_regret.end());
        std::printf("  %s: first difference at %zu (regret tree %.3f, chain %.3f); worst regret from there tree %.3f "
                    "(before it %.3f), chain %.3f; tokens above 1 nat tree %u, chain %u, most of the chain arms %u\n",
                    name.c_str(), at, tree_regret[at], chain_regret[at], tree_worst, prefix_worst, chain_worst,
                    misses(tree_regret), misses(chain_regret), chain_misses);
        failures += check(misses(tree_regret) <= chain_misses + kExtraMisses,
                          name + ": no more tokens above 1 nat of regret than the chain arms' (+" +
                              std::to_string(kExtraMisses) + ")");
        failures += check(tree_worst <= kStateRegret,
                          name + ": no token from the first difference on far below the exact model's choice (worst "
                                 "regret " + std::to_string(tree_worst) + " nats)");
    }
    return failures;
}

} // namespace

int main() {
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::printf("skip: INFERNIX_QWEN4_ARTIFACT is not set\n");
        return 77;
    }
    const char* ngram   = std::getenv("INFERNIX_QWEN4_NGRAM");
    const char* dflash2 = std::getenv("INFERNIX_QWEN4_DFLASH2");
    try {
        std::vector<Tokens> prompts;
        {
            infernix::EngineOptions options;
            options.artifact_path = artifact;
            if (ngram != nullptr) { options.ngram_volume_path = ngram; }
            options.max_context = kContext;
            infernix::Engine engine(options, infernix::test::engine_host());
            for (int variant = 0; variant < 2; ++variant) { prompts.push_back(engine.tokenize_text(prompt(variant))); }
        }
        int failures = 0;
        const bool with_dflash2 = dflash2 != nullptr && *dflash2 != '\0';
        const Arm copies_tree   = run(artifact, ngram, nullptr, 16, prompts);
        const Arm copies_chain  = run(artifact, ngram, nullptr, 0, prompts);
        std::optional<Arm> dflash2_tree, dflash2_chain;
        if (with_dflash2) {
            dflash2_tree.emplace(run(artifact, ngram, dflash2, 16, prompts));
            dflash2_chain.emplace(run(artifact, ngram, dflash2, 0, prompts));
        }
        std::vector<const Arm*> chains{&copies_chain};
        if (with_dflash2) { chains.push_back(&*dflash2_chain); }
        infernix::Engine reference(reference_options(artifact, ngram), infernix::test::engine_host());
        failures += compare("copies alone", reference, prompts, copies_tree, copies_chain, chains);
        if (with_dflash2) { failures += compare("DFlash2 7 + copies", reference, prompts, *dflash2_tree, *dflash2_chain, chains); }
        std::printf(failures == 0 ? "ngram tree real test passed\n" : "ngram tree real test: %d failure(s)\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
}
