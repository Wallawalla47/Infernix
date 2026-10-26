#include "infernix/engine.h"
#include "kv_cache_storage.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Greedy tree rounds of DFlash2, DFlash or MTP, and adaptive-dflash block rounds of either masked
// drafter, against the target's own greedy choice on the real model.
//
// A greedy tree round commits the target's argmax at every position, including positions verified
// on a side branch whose columns compaction then moved onto the main chain. The oracle is one
// CausalScoring pass over each prompt and its committed stream (a prefill without speculative
// state): every position's most probable token, and the committed token's log-probability below
// it. A context built by decode rounds rounds differently from one prefill, so chain decoding
// disagrees with this oracle too, at low-confidence positions: with the 27B NVIDIA artifact
// 1.5-2.7 % of the positions for 8- and 16-column chains and 16-column trees over every KV
// storage at margins up to 0.86 nats, and with the 35B-A3B artifact 1.0 % for trees and 6.4 %
// for the block chain at margins up to 1.70 (2026-10-26; one greedy token from a fresh prefill
// of each committed prefix, the oracle this pass replaced, gave 1.4-3.0 % and 0.94-7.0 % on the
// same runs; earlier 27B runs reached margins of 2.94 nats at a position where a fresh decode
// round of every width and kind picks a third token). A state or
// compaction error after a side branch would leave the target's choice at most later positions,
// so the test requires at most 10 % disagreements and no committed token more than kMaxMargin
// nats below the oracle's.
//
// A block round (--draft-tokens auto with adaptive-dflash) verifies a width of its proposal its
// drafts' probabilities choose; a draft copied or aligned wrongly from the block into the round's
// width would leave the target's choice the same way, and the run must verify more than one width.
// With several streams at once their rounds batch rows, which adaptive-dflash runs as plain rounds
// of the configuration's batch length where it has one.
//
// Arguments: KV codec (default int8), tree table (default "16"; "auto" for automatic widths, "0"
// runs chain rounds as the baseline, "block" block rounds), output tokens per prompt (default
// 160), draft tokens K (default 7; with "block" the ceiling, 0 the Engine's), drafter (default
// dflash2; dflash for an artifact with a DFlash drafter, whose trees take each position's 16 most
// probable tokens; mtp, whose trees add leaves to its chain, with K at most 5), n-gram copy
// drafts beside the drafter (default 0: none; minimum match 12; "auto" the Engine's resolved
// window), streams decoded at once (default 1), a prompt file in place of the built-in prompts
// (4,096-token context).
namespace {

constexpr float kMaxMargin = 6.0F;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

infernix::RequestOptions greedy(std::uint32_t outputs) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return options;
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: INFERNIX_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        const auto kv           = infernix::test::parse_kv_cache_storage(argc > 1 ? argv[1] : "int8");
        const std::string table = argc > 2 ? argv[2] : "16";
        const auto outputs      = argc > 3 ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 160U;
        const auto drafts       = argc > 4 ? static_cast<std::uint32_t>(std::stoul(argv[4])) : 7U;
        const std::string drafter = argc > 5 ? argv[5] : "dflash2";
        const bool ngram_auto     = argc > 6 && std::string(argv[6]) == "auto";
        const auto ngram = argc > 6 && !ngram_auto ? static_cast<std::uint32_t>(std::stoul(argv[6])) : 0U;
        const auto streams        = argc > 7 ? static_cast<std::uint32_t>(std::stoul(argv[7])) : 1U;
        const bool block          = table == "block";
        // A prompt file (whole prompts, each ended by a line "<<<END>>>"; the first six are taken)
        // replaces the built-in short prompts, for example copy-heavy rewrites that run copy rounds.
        std::vector<std::string> texts{
            "def merge_sort(values):\n    \"\"\"Sort a list with merge sort.\"\"\"\n",
            "The history of the printing press begins in",
            "Question: Why is the sky blue?\nAnswer:",
            "Once upon a time, in a small village by the sea,",
            "To configure nginx as a reverse proxy, first",
            "The derivative of x^2 sin(x) is",
        };
        if (argc > 8) {
            std::ifstream file(argv[8], std::ios::binary);
            std::stringstream all;
            all << file.rdbuf();
            const std::string text = all.str(), end = "\n<<<END>>>\n";
            texts.clear();
            for (std::size_t begin = 0, at = text.find(end); at != std::string::npos && texts.size() < 6;
                 begin = at + end.size(), at = text.find(end, begin)) {
                texts.push_back(text.substr(begin, at - begin));
            }
            require(!texts.empty(), "the prompt file holds no prompt ended by <<<END>>>");
        }
        const std::uint32_t kContext = argc > 8 ? 4096U : 2048U;

        infernix::EngineOptions options;
        options.artifact_path               = artifact;
        options.max_context                 = kContext;
        options.kv_capacity                 = infernix::KvCapacityPolicy::explicit_capacity(kContext * streams);
        options.prefill_chunk               = kContext;
        options.max_concurrency             = streams;
        options.kv_cache                    = kv;
        options.speculative.backend         = drafter == "dflash" ? infernix::SpeculativeBackend::DFlash
                                              : drafter == "mtp"  ? infernix::SpeculativeBackend::Mtp
                                                                  : infernix::SpeculativeBackend::DFlash2;
        // MTP trees extend chains over the drafts' sparse proposals (sampled-mtp-drafts).
        if (drafter == "mtp") {
            options.fast_path_candidates = {infernix::FastPath::AdaptiveSpeculation,
                                            infernix::FastPath::SampledMtpDrafts};
        }
        options.speculative.draft_tokens    = drafts;
        options.speculative.proposal_head   = infernix::ProposalHead::Optimized;
        options.speculative.draft_tree_auto = table == "auto";
        options.speculative.ngram_draft_tokens      = ngram;
        options.speculative.ngram_draft_tokens_auto = ngram_auto;
        options.speculative.ngram_min_match         = 12;
        if (block) {
            options.speculative.draft_tokens_auto = true;
            // The policy acts where speculation is adaptive, which not every artifact registers.
            options.fast_path_candidates = {infernix::FastPath::AdaptiveSpeculation,
                                            infernix::FastPath::AdaptiveDFlash};
        }
        std::size_t entry = 0;
        for (std::size_t begin = 0;
             !options.speculative.draft_tree_auto && !block && begin <= table.size() && entry < 8;
             ++entry) {
            const std::size_t end = std::min(table.find(',', begin), table.size());
            options.speculative.draft_tree_nodes[entry] =
                static_cast<std::uint32_t>(std::stoul(table.substr(begin, end - begin)));
            begin = end + 1;
        }
        for (; entry != 0 && entry < 8; ++entry)
            options.speculative.draft_tree_nodes[entry] =
                options.speculative.draft_tree_nodes[entry - 1];
        const bool tree =
            options.speculative.draft_tree_auto || options.speculative.draft_tree_nodes[0] != 0;

        std::uint64_t rounds = 0, side_rounds = 0, side_drafts = 0, positions = 0, copy_rounds = 0;
        std::vector<std::uint64_t> by_length;
        std::vector<std::vector<infernix::TokenId>> prompts;
        std::vector<infernix::GenerationResult> results;
        {
            infernix::Engine engine(options);
            if (ngram_auto) {
                const auto& resolved = engine.options().speculative;
                std::cout << "ngram auto: window " << resolved.ngram_draft_tokens << ", minimum match "
                          << resolved.ngram_min_match << '\n';
                require(!resolved.ngram_draft_tokens_auto, "the automatic ngram window was not resolved");
            }
            // Every stream is submitted at once; the Engine decodes `streams` of them together.
            std::vector<infernix::GenerationHandle> handles;
            for (const std::string& text : texts) {
                prompts.push_back(engine.tokenize_text(text));
                handles.push_back(engine.submit(engine.prepare_tokens(prompts.back()), greedy(outputs)));
            }
            for (infernix::GenerationHandle& handle : handles) { results.push_back(handle.wait()); }
        }
        for (const infernix::GenerationResult& result : results) {
            require(result.generated_token_ids.size() == outputs, "greedy stream stopped before its output budget");
            rounds += result.speculative.rounds;
            copy_rounds += result.speculative.ngram_rounds;
            by_length.resize(std::max(by_length.size(), result.speculative.rounds_by_draft_length.size()));
            for (std::size_t k = 0; k < result.speculative.rounds_by_draft_length.size(); ++k) {
                by_length[k] += result.speculative.rounds_by_draft_length[k];
            }
            side_rounds += result.speculative.tree_side_rounds;
            side_drafts += result.speculative.tree_side_accepted_tokens;
        }
        if (tree) { require(side_rounds != 0, "the greedy streams accepted no side branch"); }
        if (block) {
            const auto widths = std::count_if(by_length.begin(), by_length.end(),
                                              [](std::uint64_t n) { return n != 0; });
            std::cout << "rounds verifying 1..N drafts:";
            for (const std::uint64_t n : by_length) { std::cout << ' ' << n; }
            std::cout << '\n';
            require(widths > 1, "the block rounds verified a single width");
        }

        // The oracle: one scoring pass over each prompt and its committed stream gives every
        // position's most probable token and the committed token's log-probability.
        std::size_t disagreements = 0, above_half = 0, above_one = 0;
        float worst = 0.0F;
        {
            infernix::EngineOptions scoring;
            scoring.artifact_path = artifact;
            scoring.purpose       = infernix::EnginePurpose::CausalScoring;
            scoring.max_context   = kContext;
            scoring.kv_cache      = kv;
            infernix::Engine scorer(scoring);
            for (std::size_t p = 0; p < prompts.size(); ++p) {
                const auto& generated = results[p].generated_token_ids;
                auto stream           = prompts[p];
                stream.insert(stream.end(), generated.begin(), generated.end());
                const auto scores = scorer.score_tokens(stream, static_cast<std::uint32_t>(prompts[p].size()),
                                                        {.top_k = 1});
                require(scores.logprobs.size() == outputs && scores.top_ids.size() == outputs,
                        "scoring returned no score for some position");
                for (std::size_t j = 0; j < outputs; ++j) {
                    ++positions;
                    if (scores.top_ids[j] == generated[j]) continue;
                    const float margin = scores.top_logprobs[j] - scores.logprobs[j];
                    ++disagreements;
                    worst = std::max(worst, margin);
                    above_half += margin > 0.5F ? 1U : 0U;
                    above_one += margin > 1.0F ? 1U : 0U;
                    std::cout << "prompt " << p << " position " << j << ": committed " << generated[j]
                              << " (log p " << std::fixed << std::setprecision(4) << scores.logprobs[j]
                              << "), oracle " << scores.top_ids[j] << " (log p " << scores.top_logprobs[j]
                              << "), margin " << margin << '\n';
                }
            }
        }
        std::cout << "table=" << table << " K=" << drafts << " positions=" << positions
                  << " disagreements=" << disagreements << " above_0.5=" << above_half
                  << " above_1=" << above_one << " worst_margin=" << std::fixed
                  << std::setprecision(4) << worst << " rounds=" << rounds
                  << " side_rounds=" << side_rounds << " side_drafts=" << side_drafts
                  << " ngram_rounds=" << copy_rounds << '\n';
        require(worst <= kMaxMargin, "a committed token is far below the target's greedy choice");
        require(disagreements * 10 <= positions,
                "the stream left the target's greedy choice at too many positions");
        std::cout << "ok\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
