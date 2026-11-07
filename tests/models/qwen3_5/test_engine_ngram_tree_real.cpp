// Qwen3.5-family n-gram trees (--ngram-tree-nodes; docs/maintainer/tree-verification.md, "N-gram trees") on a
// real artifact. Each prompt shows a code fragment twice, as version 1 and as version 2 (seven edits, each after
// more than 64 tokens both share), and asks for one of them again. While a copy runs through shared text both
// versions match equally and the newer, version 2, is the copy's main chain, so a copy of version 1 meets its
// other source as a side branch at every edit. Greedy streams with copies of up to 31 drafts (minimum match 12)
// verified as trees of 16 columns and as chains:
//
//   rounds      the tree arm verifies n-gram tree rounds, and some accept the other source's branch;
//   exactness   the arms commit the same tokens up to their first difference, where both tokens must be a
//               near-tie of the target (a scoring pass over the shared prefix: log-probabilities within kNearTie
//               nats). The arms verify at different widths, so a near-tie may flip; a state or compaction error
//               would leave the tree arm's token far below the target's choice.
//
// Arguments: drafter (dflash2, dflash or mtp; default dflash2), streams decoded at once (default 1: with 2 the
// rows without a copy verify their neural proposal in the tree's round), KV codec (default int8). The artifact
// (INFERNIX_TEST_ARTIFACT) must hold the drafter and verify trees (single FP8, NVFP4 or Q8 GDN input projections).

#include "infernix/engine.h"
#include "kv_cache_storage.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

namespace {

using Tokens = std::vector<infernix::TokenId>;

constexpr std::uint32_t kOutputs = 600;
constexpr std::uint32_t kContext = 4096;
constexpr float kNearTie         = 1.5F;

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

infernix::EngineOptions engine_options(const char* artifact, const std::string& drafter, std::uint32_t streams,
                                       infernix::KvCacheStorage kv, std::uint32_t tree_nodes) {
    infernix::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = kContext;
    options.kv_capacity     = infernix::KvCapacityPolicy::explicit_capacity(kContext * streams);
    options.prefill_chunk   = kContext;
    options.max_concurrency = streams;
    options.kv_cache        = kv;
    options.speculative.backend       = drafter == "dflash" ? infernix::SpeculativeBackend::DFlash
                                        : drafter == "mtp"  ? infernix::SpeculativeBackend::Mtp
                                                            : infernix::SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens  = drafter == "mtp" ? 3U : 7U;
    options.speculative.proposal_head = infernix::ProposalHead::Optimized;
    options.speculative.ngram_draft_tokens = 31;
    options.speculative.ngram_min_match    = 12;
    options.speculative.ngram_tree_nodes   = tree_nodes;
    return options;
}

Arm run(const infernix::EngineOptions& options, const std::vector<Tokens>& prompts) {
    infernix::Engine engine(options);
    std::vector<infernix::GenerationHandle> handles;
    for (const Tokens& tokens : prompts) {
        infernix::RequestOptions request;
        request.execution.requested_output_tokens = kOutputs;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.allow_prefix_reuse      = false;
        request.stop.include_model_defaults       = false;
        handles.push_back(engine.submit(engine.prepare_tokens(tokens), request));
    }
    Arm arm;
    for (infernix::GenerationHandle& handle : handles) {
        const infernix::GenerationResult result = handle.wait();
        arm.streams.push_back(result.generated_token_ids);
        arm.tree_rounds += result.speculative.ngram_tree_rounds;
        arm.side_rounds += result.speculative.ngram_tree_side_rounds;
        arm.side_drafts += result.speculative.ngram_tree_side_accepted_tokens;
        arm.ngram_rounds += result.speculative.ngram_rounds;
    }
    return arm;
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::printf("skip: INFERNIX_TEST_ARTIFACT is not set\n");
        return 77;
    }
    try {
        const std::string drafter = argc > 1 ? argv[1] : "dflash2";
        const auto streams        = argc > 2 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 1U;
        const auto kv             = infernix::test::parse_kv_cache_storage(argc > 3 ? argv[3] : "int8");
        std::vector<Tokens> prompts;
        {
            infernix::EngineOptions options;
            options.artifact_path = artifact;
            options.max_context   = kContext;
            options.kv_capacity   = infernix::KvCapacityPolicy::explicit_capacity(kContext);
            infernix::Engine engine(options);
            for (int variant = 0; variant < 2; ++variant) { prompts.push_back(engine.tokenize_text(prompt(variant))); }
        }
        const Arm tree  = run(engine_options(artifact, drafter, streams, kv, 16), prompts);
        const Arm chain = run(engine_options(artifact, drafter, streams, kv, 0), prompts);
        int failures    = 0;
        std::printf("%s x%u: tree arm %llu n-gram tree rounds, %llu took the other source (%llu drafts), %llu copy "
                    "rounds; chain arm %llu copy rounds\n",
                    drafter.c_str(), streams, static_cast<unsigned long long>(tree.tree_rounds),
                    static_cast<unsigned long long>(tree.side_rounds), static_cast<unsigned long long>(tree.side_drafts),
                    static_cast<unsigned long long>(tree.ngram_rounds), static_cast<unsigned long long>(chain.ngram_rounds));
        failures += check(tree.tree_rounds > 0, "the tree arm verified n-gram trees");
        failures += check(tree.side_rounds > 0 && tree.side_drafts >= tree.side_rounds,
                          "some tree round accepted the other source's branch");
        infernix::EngineOptions scoring;
        scoring.artifact_path = artifact;
        scoring.purpose       = infernix::EnginePurpose::CausalScoring;
        scoring.max_context   = kContext;
        scoring.kv_cache      = kv;
        infernix::Engine scorer(scoring);
        for (std::size_t p = 0; p < prompts.size(); ++p) {
            const Tokens& a = tree.streams[p];
            const Tokens& b = chain.streams[p];
            std::size_t at  = 0;
            while (at < a.size() && at < b.size() && a[at] == b[at]) { ++at; }
            const std::string where = "prompt " + std::to_string(p);
            if (at == a.size() && at == b.size()) {
                failures += check(true, where + ": identical streams");
                continue;
            }
            if (at == a.size() || at == b.size()) {
                failures += check(false, where + ": one stream stopped early");
                continue;
            }
            Tokens sequence = prompts[p];
            sequence.insert(sequence.end(), a.begin(), a.begin() + static_cast<std::ptrdiff_t>(at));
            const auto first = static_cast<std::uint32_t>(sequence.size());
            sequence.push_back(a[at]);
            Tokens other = sequence;
            other.back() = b[at];
            const float tree_logprob  = scorer.score_tokens(sequence, first, {.top_k = 1}).logprobs.back();
            const float chain_logprob = scorer.score_tokens(other, first, {.top_k = 1}).logprobs.back();
            failures += check(std::abs(tree_logprob - chain_logprob) <= kNearTie,
                              where + ": first difference at " + std::to_string(at) + " is a near-tie (tree token log p " +
                                  std::to_string(tree_logprob) + ", chain token " + std::to_string(chain_logprob) + ")");
        }
        std::printf(failures == 0 ? "ngram tree real test passed\n" : "ngram tree real test: %d failure(s)\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
}
