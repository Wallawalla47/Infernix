// N-gram trees on the host (docs/maintainer/tree-verification.md, "N-gram trees"): the proposer's
// candidates (propose()'s choice first, one per source position) and the tree they build in
// SpeculativeTreeRow layout (main chain first, side branches after it and no deeper, transport rows of
// deterministic children whose parents' laws hold every child). CPU only.
#include "models/qwen3_5/program/ngram_proposer.h"
#include "models/qwen3_5/program/ngram_tree.h"

#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using infernix::models::qwen3_5::detail::build_ngram_tree;
using infernix::models::qwen3_5::detail::NgramProposer;
using infernix::models::qwen3_5::detail::NgramTree;
using infernix::models::qwen3_5::detail::NgramTreeLimits;
using infernix::models::qwen3_5::NgramMatch;
namespace ops = infernix::ops;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

// Every layout rule tree verification relies on (speculative_tree.h).
void check_layout(const NgramTree& tree, const std::vector<NgramMatch>& candidates, NgramTreeLimits limits) {
    const ops::SpeculativeTreeRow& row = tree.row;
    const int nodes = row.nodes, main = row.main_depth;
    require(nodes == static_cast<int>(tree.columns) && nodes <= static_cast<int>(limits.nodes), "column count");
    require(main >= 1 && main <= static_cast<int>(limits.depth), "main depth");
    require(row.paths >= 2 && row.paths <= static_cast<int>(limits.paths), "path count");
    for (int c = 1; c <= main; ++c) {
        require(row.parent[c] == c - 1 && row.depth[c] == c, "the main chain occupies columns 0..K");
        require(tree.drafts[c - 1] == candidates.front().tokens[c - 1], "the main chain is the best candidate");
    }
    for (int c = 1; c < nodes; ++c) {
        require(row.parent[c] >= 0 && row.parent[c] < c, "parents precede children");
        require(row.depth[c] == row.depth[row.parent[c]] + 1, "depth follows the parent");
        if (c > main) { require(row.depth[c] <= main, "side columns lie no deeper than the main chain"); }
        std::uint32_t mask = 0;
        for (int v = c; v >= 0; v = row.parent[v]) { mask |= 1U << v; }
        require(tree.ancestor_masks[c] == mask, "ancestor mask");
        // The parent's law holds every child of the parent, and the first child is the main one.
        const int p = row.parent[c];
        for (int s = row.first_child[p]; s >= 0; s = row.next_sibling[s]) {
            bool listed = false;
            for (int slot = 0; slot < ops::kSparseSpeculativeCandidates; ++slot) {
                listed |= tree.candidate_ids[(c - 1) * ops::kSparseSpeculativeCandidates + slot] == tree.drafts[s - 1];
            }
            require(listed, "a child's token is among its parent's candidates");
        }
        require(p > main || p == 0 || row.first_child[p] == p + 1 || p == main, "the main chain is a node's first child");
    }
    require(row.first_child[0] == 1, "the main chain starts the anchor's children");
    // Paths: increasing leaf columns, the main chain first, each owned suffix new.
    std::vector<int> owner(static_cast<std::size_t>(nodes), -1);
    int last_leaf = -1;
    for (int p = 0; p < row.paths; ++p) {
        const int length = row.path_length[p];
        const int leaf   = row.path_columns[p][length - 1];
        require(leaf > last_leaf && row.first_child[leaf] < 0, "paths end at leaves in column order");
        last_leaf = leaf;
        require(row.path_columns[p][0] == 0, "paths start at the anchor");
        for (int i = 1; i < length; ++i) {
            require(row.parent[row.path_columns[p][i]] == row.path_columns[p][i - 1], "paths follow parents");
        }
        for (int i = row.owned_from[p]; i < length; ++i) {
            require(owner[static_cast<std::size_t>(row.path_columns[p][i])] < 0, "a column has one owner");
            owner[static_cast<std::size_t>(row.path_columns[p][i])] = p;
        }
    }
    require(row.path_length[0] == main + 1, "path 0 is the main chain");
    for (int c = 1; c < nodes; ++c) { require(owner[static_cast<std::size_t>(c)] >= 0, "every column is owned"); }
    require(row.tree == 1 && row.transport == 1 && row.sampled == 0U, "a transport row of deterministic children");
}

NgramMatch candidate(std::vector<int> tokens, std::uint32_t matched) { return {std::move(tokens), matched}; }

} // namespace

int main() {
    try {
        constexpr std::int32_t kDomain = 1000;
        const NgramTreeLimits limits{.depth = 7, .nodes = 16, .paths = 4};
        // One candidate, or candidates agreeing within the depth: a chain.
        {
            const std::vector<NgramMatch> one{candidate({1, 2, 3, 4, 5, 6, 7, 8, 9}, 20)};
            const NgramTree chain = build_ngram_tree(one, limits, kDomain);
            require(!chain.tree() && chain.columns == 8, "a single candidate is a chain of the main depth");
            const std::vector<NgramMatch> agree{candidate({1, 2, 3, 4, 5, 6, 7, 8}, 20),
                                                candidate({1, 2, 3, 4, 5, 6, 7, 99}, 18)};
            require(!build_ngram_tree(agree, limits, kDomain).tree(), "a divergence past the depth adds nothing");
        }
        // Three sources: the second leaves the main chain at draft 3, the third at draft 1.
        {
            const std::vector<NgramMatch> three{candidate({1, 2, 3, 4, 5, 6, 7, 8}, 30),
                                                candidate({1, 2, 3, 40, 50, 60, 70, 80}, 25),
                                                candidate({10, 20, 30}, 12)};
            const NgramTree tree = build_ngram_tree(three, limits, kDomain);
            check_layout(tree, three, limits);
            require(tree.columns == 1 + 7 + 4 + 3 && tree.row.paths == 3, "main 7, a 4-column branch and a 3-column one");
            require(tree.row.parent[8] == 3 && tree.drafts[7] == 40, "the second branch hangs off draft 3");
            require(tree.row.parent[12] == 0 && tree.drafts[11] == 10, "the third off the anchor");
            require(tree.row.sibling_index[8] == 1 && tree.row.next_sibling[4] == 8, "siblings in candidate order");
        }
        // The main chain yields columns to the other sources' branches, down to three quarters of the
        // tree: a 14-deep copy and a source leaving it at draft 2 share 16 columns as 12 main + 3 side.
        {
            std::vector<int> main(20), other(20);
            std::iota(main.begin(), main.end(), 1);
            other = main;
            for (int i = 2; i < 20; ++i) { other[static_cast<std::size_t>(i)] = 500 + i; }
            const std::vector<NgramMatch> two{candidate(main, 40), candidate(other, 30)};
            const NgramTreeLimits wide{.depth = 14, .nodes = 16, .paths = 4};
            const NgramTree tree = build_ngram_tree(two, wide, kDomain);
            check_layout(tree, two, wide);
            require(tree.row.main_depth == 12 && tree.columns == 16, "main 12 and a 3-column branch");
            require(tree.row.parent[13] == 2 && tree.drafts[12] == 502 && tree.drafts[14] == 504,
                    "the branch leaves after draft 2");
            // A late divergence costs the main chain nothing: leaving at draft 12, the main chain keeps 14
            // and the branch takes the last column.
            for (int i = 2; i < 12; ++i) { other[static_cast<std::size_t>(i)] = main[static_cast<std::size_t>(i)]; }
            const std::vector<NgramMatch> late{candidate(main, 40), candidate(other, 30)};
            const NgramTree deep = build_ngram_tree(late, wide, kDomain);
            check_layout(deep, late, wide);
            require(deep.row.main_depth == 14 && deep.columns == 16 && deep.row.parent[15] == 12,
                    "main 14 and a 1-column branch after draft 12");
        }
        // Limits: columns and paths bound the tree; a cut branch still ends a path.
        {
            std::vector<NgramMatch> many{candidate({1, 2, 3, 4, 5, 6, 7}, 40)};
            for (int k = 0; k < 8; ++k) { many.push_back(candidate({1, 100 + k, 200 + k, 300 + k, 400 + k}, 30 - k)); }
            const NgramTree tree = build_ngram_tree(many, limits, kDomain);
            check_layout(tree, many, limits);
            // Two 4-column branches fill the 16 columns; the third candidate finds no room.
            require(tree.columns == 16 && tree.row.paths == 3, "16 columns, 3 paths");
            bool rejected = false;
            try {
                (void)build_ngram_tree(many, NgramTreeLimits{.depth = 16, .nodes = 32, .paths = 8}, kDomain);
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "a main chain deeper than 15 is refused");
        }
        // The proposer's candidates: propose()'s choice first, one per source, others after it.
        {
            NgramProposer proposer(1U << 12, 1U << 10);
            std::vector<int> context(24);
            std::iota(context.begin(), context.end(), 500);
            std::vector<int> first = context, second = context;
            for (int t : {1, 2, 3, 4, 5, 6}) { first.push_back(t); }
            for (int t : {1, 2, 9, 9, 9, 9}) { second.push_back(t); }
            proposer.ingest(first);
            proposer.ingest(second);
            const auto candidates = proposer.propose_candidates(context, 6, 12, 8);
            const auto best       = proposer.propose(context, 6, 12);
            require(candidates.size() == 2, "two sources match the context (" + std::to_string(candidates.size()) + ")");
            require(candidates.front().tokens == best.tokens && candidates.front().matched == best.matched,
                    "the first candidate is propose()'s choice");
            require(candidates[0].tokens != candidates[1].tokens, "the sources continue differently");
            const NgramTree tree = build_ngram_tree(candidates, NgramTreeLimits{.depth = 6, .nodes = 12, .paths = 2}, kDomain);
            check_layout(tree, candidates, NgramTreeLimits{.depth = 6, .nodes = 12, .paths = 2});
            require(tree.columns == 1 + 6 + 4, "the second source branches after their two shared drafts");
            require(proposer.propose_candidates(context, 6, 12, 1).size() == 1, "the limit bounds the candidates");
            require(proposer.propose_candidates(context, 6, 40, 8).empty(), "the minimum match applies");
        }
        std::cout << "ngram tree tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
