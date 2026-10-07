#include "infernix/engine.h"
#include "kv_cache_storage.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Greedy DFlash2 tree rounds against the target's own greedy choice on the real model.
//
// A greedy tree round commits the target's argmax at every position, including positions verified
// on a side branch whose columns compaction then moved onto the main chain. For every position of
// a tree-decoded stream this test asks the same Engine for one greedy token from the exact
// committed prefix (a fresh prefill without speculative state) and scores each disagreement on a
// CausalScoring Engine. A context built by decode rounds rounds differently from one prefill, so
// chain decoding disagrees with this oracle too, at low-confidence positions: with the 27B NVIDIA
// artifact 3.0-3.3 % of the positions for 8- and 16-column chains and 16-column trees over every
// KV storage, at margins up to 1.25 nats for chains and 2.94 for trees (a position where a fresh
// decode round of every width and kind picks a third token). A state or compaction error after a
// side branch would leave the target's choice at most later positions, so the test requires at
// most 10 % disagreements and no committed token more than kMaxMargin nats below the oracle's.
//
// Arguments: KV codec (default int8), tree table (default "16"; "auto" for automatic widths, "0"
// runs chain rounds as the baseline), output tokens per prompt (default 160), draft tokens K
// (default 7).
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

struct Disagreement {
    std::vector<infernix::TokenId> prefix;
    infernix::TokenId committed = 0;
    infernix::TokenId oracle    = 0;
    std::size_t prompt        = 0;
    std::size_t position      = 0;
};

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
        constexpr std::uint32_t kContext = 2048;

        infernix::EngineOptions options;
        options.artifact_path               = artifact;
        options.max_context                 = kContext;
        options.kv_capacity                 = infernix::KvCapacityPolicy::explicit_capacity(kContext);
        options.prefill_chunk               = kContext;
        options.max_concurrency             = 1;
        options.kv_cache                    = kv;
        options.speculative.backend         = infernix::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens    = drafts;
        options.speculative.proposal_head   = infernix::ProposalHead::Optimized;
        options.speculative.draft_tree_auto = table == "auto";
        std::size_t entry                   = 0;
        for (std::size_t begin = 0;
             !options.speculative.draft_tree_auto && begin <= table.size() && entry < 8; ++entry) {
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

        const std::vector<std::string> texts{
            "def merge_sort(values):\n    \"\"\"Sort a list with merge sort.\"\"\"\n",
            "The history of the printing press begins in",
            "Question: Why is the sky blue?\nAnswer:",
            "Once upon a time, in a small village by the sea,",
            "To configure nginx as a reverse proxy, first",
            "The derivative of x^2 sin(x) is",
        };
        std::vector<Disagreement> disagreements;
        std::uint64_t rounds = 0, side_rounds = 0, side_drafts = 0, positions = 0;
        {
            infernix::Engine engine(options);
            for (std::size_t p = 0; p < texts.size(); ++p) {
                const auto prompt = engine.tokenize_text(texts[p]);
                const auto result = engine.generate(engine.prepare_tokens(prompt), greedy(outputs));
                require(result.generated_token_ids.size() == outputs,
                        "greedy stream stopped before its output budget");
                rounds += result.speculative.rounds;
                side_rounds += result.speculative.tree_side_rounds;
                side_drafts += result.speculative.tree_side_accepted_tokens;
                auto prefix = prompt;
                for (std::size_t j = 0; j < outputs; ++j) {
                    const auto oracle = engine.generate(engine.prepare_tokens(prefix), greedy(1));
                    require(oracle.generated_token_ids.size() == 1, "oracle produced no token");
                    const infernix::TokenId committed = result.generated_token_ids[j];
                    if (oracle.generated_token_ids[0] != committed) {
                        disagreements.push_back(
                            {prefix, committed, oracle.generated_token_ids[0], p, j});
                    }
                    prefix.push_back(committed);
                    ++positions;
                }
            }
        }
        if (tree) { require(side_rounds != 0, "the greedy streams accepted no side branch"); }

        float worst            = 0.0F;
        std::size_t above_half = 0, above_one = 0;
        if (!disagreements.empty()) {
            infernix::EngineOptions scoring;
            scoring.artifact_path = artifact;
            scoring.purpose       = infernix::EnginePurpose::CausalScoring;
            scoring.max_context   = kContext;
            scoring.kv_cache      = kv;
            infernix::Engine scorer(scoring);
            for (const Disagreement& d : disagreements) {
                const auto first    = static_cast<std::uint32_t>(d.prefix.size());
                auto with_committed = d.prefix;
                with_committed.push_back(d.committed);
                auto with_oracle = d.prefix;
                with_oracle.push_back(d.oracle);
                const float committed = scorer.score_tokens(with_committed, first).logprobs.at(0);
                const float oracle    = scorer.score_tokens(with_oracle, first).logprobs.at(0);
                const float margin    = oracle - committed;
                worst                 = std::max(worst, margin);
                above_half += margin > 0.5F ? 1U : 0U;
                above_one += margin > 1.0F ? 1U : 0U;
                std::cout << "prompt " << d.prompt << " position " << d.position << ": committed "
                          << d.committed << " (log p " << std::fixed << std::setprecision(4)
                          << committed << "), oracle " << d.oracle << " (log p " << oracle
                          << "), margin " << margin << '\n';
            }
        }
        std::cout << "table=" << table << " K=" << drafts << " positions=" << positions
                  << " disagreements=" << disagreements.size() << " above_0.5=" << above_half
                  << " above_1=" << above_one << " worst_margin=" << std::fixed
                  << std::setprecision(4) << worst << " rounds=" << rounds
                  << " side_rounds=" << side_rounds << " side_drafts=" << side_drafts << '\n';
        require(worst <= kMaxMargin, "a committed token is far below the target's greedy choice");
        require(disagreements.size() * 10 <= positions,
                "the stream left the target's greedy choice at too many positions");
        std::cout << "ok\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
