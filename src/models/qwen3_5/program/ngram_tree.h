#pragma once

#include "infernix/ops/sampling.h"
#include "infernix/ops/speculative_tree.h"
#include "models/qwen3_5/ngram.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <span>
#include <stdexcept>
#include <vector>

namespace infernix::models::qwen3_5::detail {

// A request's n-gram tree rounds (decode.cpp): their tokens over their copies' alone, learned with weight
// kNgramTreeGainWeight from kNgramTreeGainPrior; below 1 the request verifies its copies as chains, and a tree
// every kNgramTreeProbeInterval-th eligible round. A copy's acceptance per draft before 64 drafts is the prior.
inline constexpr double kNgramTreeGainPrior            = 1.1;
inline constexpr double kNgramTreeGainWeight           = 0.25;
inline constexpr std::uint32_t kNgramTreeProbeInterval = 8;
inline constexpr double kNgramTreeCopyAcceptancePrior  = 0.9;

// The limits of one row's n-gram tree: its main chain's depth (1..15, drafts), its columns with the
// anchor (main depth + 2 .. 32) and its root-to-leaf paths (2..8).
struct NgramTreeLimits {
    std::uint32_t depth = 0;
    std::uint32_t nodes = 0;
    std::uint32_t paths = 0;
};

// A row's n-gram tree in the layout of ops::SpeculativeTreeRow (docs/maintainer/tree-verification.md,
// "N-gram trees"): column 0 the anchor, columns 1..D the best candidate's first D drafts (the main
// chain: the start of the chain a copy round would verify, D the depth limit or less where the other
// candidates' branches need the columns, never below three quarters of them), then every other candidate's
// continuation from where it leaves the tree built so far, at most D deep, in candidate order while
// columns and paths last. Every draft is a deterministic child: the row is a transport row (UniVer) with no residual
// draw, so a sampled row accepts a child with its target probability and otherwise draws the
// correction from the target outside the children; greedy and Gumbel rows take the child equal to
// the target token. Exactly the processed target distribution either way.
struct NgramTree {
    ops::SpeculativeTreeRow row{};
    std::uint32_t columns = 0; // live columns, the anchor included (row.nodes)
    // Per column c >= 1 (index c-1): its draft, and its parent's law as every child stores it
    // (candidate ids with the children's tokens first, one-hot q on the first).
    std::vector<std::int32_t> drafts;
    std::vector<std::int32_t> candidate_ids; // [columns-1][kSparseSpeculativeCandidates]
    std::vector<float> proposal_q;           // [columns-1][kSparseSpeculativeCandidates]
    std::vector<std::uint32_t> ancestor_masks; // per column: bit a iff a is c or an ancestor of c
    [[nodiscard]] bool tree() const noexcept { return row.tree != 0; }
};

// Builds the tree of `candidates` (best first; the first is the main chain). The row is a chain
// (row.tree == 0, columns = 1 + its drafts) when no other candidate leaves the main chain within its
// depth, or when the limits leave no room for a side column. `domain` is the public token count
// (filler candidate ids stay inside it).
[[nodiscard]] inline NgramTree build_ngram_tree(std::span<const NgramMatch> candidates, NgramTreeLimits limits,
                                                std::int32_t domain) {
    constexpr int kSlots = ops::kSparseSpeculativeCandidates;
    if (limits.depth < 1 || limits.depth >= static_cast<std::uint32_t>(ops::kSpeculativeTreeMaxPathLength) ||
        limits.nodes > static_cast<std::uint32_t>(ops::kSpeculativeTreeMaxNodes) || limits.nodes < limits.depth + 2U ||
        limits.paths < 2 || limits.paths > static_cast<std::uint32_t>(ops::kSpeculativeTreeMaxPaths) || domain <= kSlots) {
        throw std::invalid_argument("n-gram tree limits are out of range");
    }
    NgramTree out;
    if (candidates.empty() || candidates.front().tokens.empty()) { return out; }
    // The main chain's depth: the limit, shortened while the other sources' branches (each from where it
    // first leaves the main copy, as many as the paths allow) would not reach one draft short of it, but
    // never below three quarters of the columns. A copy is usually right, so the main chain keeps the
    // larger share; a late divergence costs it little depth. (NVFP4 27B, DFlash2 auto + n-gram auto, two
    // versions of a file, 16 columns, one request: decode 3.51 s with a floor of half the columns, 2.53 s
    // at three quarters, 2.72 s without yielding; 2026-11-07.)
    const auto& main_tokens = candidates.front().tokens;
    const auto overflow = [&](std::uint32_t main_depth) {
        std::uint32_t columns = 1U + main_depth, branches = 0;
        for (std::size_t k = 1; k < candidates.size() && branches + 1U < limits.paths; ++k) {
            const auto& tokens = candidates[k].tokens;
            std::size_t split  = 0;
            while (split < tokens.size() && split < main_tokens.size() && tokens[split] == main_tokens[split]) {
                ++split;
            }
            const std::size_t end = std::min<std::size_t>(tokens.size(), main_depth);
            if (split >= end) { continue; }
            columns += static_cast<std::uint32_t>(end - split) - 1U; // to depth main_depth - 1
            ++branches;
        }
        return columns > limits.nodes;
    };
    const std::uint32_t floor = 3U * limits.nodes / 4U;
    auto depth_of_main = static_cast<std::uint32_t>(std::min<std::size_t>(main_tokens.size(), limits.depth));
    while (depth_of_main > floor && overflow(depth_of_main)) { --depth_of_main; }
    // Columns as (parent, token, depth); column 0 is the anchor.
    std::vector<std::int32_t> parent{-1}, token{-1}, depth{0};
    std::vector<std::vector<std::int32_t>> children(1);
    const auto add = [&](std::int32_t p, std::int32_t t) {
        const auto c = static_cast<std::int32_t>(parent.size());
        parent.push_back(p);
        token.push_back(t);
        depth.push_back(depth[static_cast<std::size_t>(p)] + 1);
        children.emplace_back();
        children[static_cast<std::size_t>(p)].push_back(c);
        return c;
    };
    for (std::uint32_t i = 0; i < depth_of_main; ++i) {
        (void)add(static_cast<std::int32_t>(i), candidates.front().tokens[i]);
    }
    std::uint32_t leaves = 1;
    for (std::size_t k = 1; k < candidates.size() && leaves < limits.paths; ++k) {
        const auto& tokens = candidates[k].tokens;
        std::int32_t at    = 0;
        std::size_t step   = 0;
        // Follow the tree while the candidate agrees with it.
        for (; step < tokens.size() && step < depth_of_main; ++step) {
            const auto& kids = children[static_cast<std::size_t>(at)];
            const auto next  = std::find_if(kids.begin(), kids.end(), [&](std::int32_t c) {
                return token[static_cast<std::size_t>(c)] == tokens[step];
            });
            if (next == kids.end()) { break; }
            at = *next;
        }
        if (step >= tokens.size() || step >= depth_of_main) { continue; } // nothing new within the depth
        // A sibling under `at`, then its continuation, as far as the depth and the columns allow. A
        // node takes at most kSlots children (its law's candidates).
        if (children[static_cast<std::size_t>(at)].size() >= static_cast<std::size_t>(kSlots) ||
            parent.size() >= limits.nodes) {
            continue;
        }
        for (; step < tokens.size() && step < depth_of_main && parent.size() < limits.nodes; ++step) {
            at = add(at, tokens[step]);
        }
        ++leaves;
    }
    out.columns = static_cast<std::uint32_t>(parent.size());
    const bool tree = leaves > 1;
    ops::SpeculativeTreeRow& row = out.row;
    row.nodes      = static_cast<std::int32_t>(out.columns);
    row.main_depth = static_cast<std::int32_t>(depth_of_main);
    row.tree       = tree ? 1 : 0;
    row.transport  = tree ? 1 : 0;
    row.sampled    = 0U;
    std::fill(std::begin(row.parent), std::end(row.parent), static_cast<std::int8_t>(-1));
    std::fill(std::begin(row.first_child), std::end(row.first_child), static_cast<std::int8_t>(-1));
    std::fill(std::begin(row.next_sibling), std::end(row.next_sibling), static_cast<std::int8_t>(-1));
    for (std::uint32_t c = 0; c < out.columns; ++c) {
        row.parent[c] = static_cast<std::int8_t>(parent[c]);
        row.depth[c]  = static_cast<std::int8_t>(depth[c]);
        const auto& kids = children[c];
        if (!kids.empty()) { row.first_child[c] = static_cast<std::int8_t>(kids.front()); }
        for (std::size_t s = 0; s < kids.size(); ++s) {
            row.sibling_index[kids[s]] = static_cast<std::int8_t>(s);
            if (s + 1 < kids.size()) { row.next_sibling[kids[s]] = static_cast<std::int8_t>(kids[s + 1]); }
        }
    }
    // Paths in increasing leaf column order (the main chain first); a column belongs to the first.
    std::vector<std::uint8_t> owned(out.columns, 0);
    std::int32_t paths = 0;
    for (std::uint32_t c = 0; c < out.columns; ++c) {
        if (!children[c].empty() || (c == 0 && out.columns > 1)) { continue; }
        std::vector<std::int32_t> path;
        for (std::int32_t v = static_cast<std::int32_t>(c); v >= 0; v = parent[static_cast<std::size_t>(v)]) {
            path.push_back(v);
        }
        std::reverse(path.begin(), path.end());
        row.path_length[paths] = static_cast<std::int8_t>(path.size());
        std::int8_t from = static_cast<std::int8_t>(path.size());
        for (std::size_t i = 0; i < path.size(); ++i) {
            row.path_columns[paths][i] = static_cast<std::int8_t>(path[i]);
            if (!owned[static_cast<std::size_t>(path[i])]) {
                owned[static_cast<std::size_t>(path[i])] = 1;
                from = std::min<std::int8_t>(from, static_cast<std::int8_t>(i));
            }
        }
        row.owned_from[paths] = from;
        ++paths;
    }
    row.paths = paths;
    out.drafts.resize(out.columns > 0 ? out.columns - 1 : 0);
    out.candidate_ids.assign(out.drafts.size() * kSlots, 0);
    out.proposal_q.assign(out.drafts.size() * kSlots, 0.0F);
    out.ancestor_masks.assign(out.columns, 0U);
    for (std::uint32_t c = 0; c < out.columns; ++c) {
        for (std::int32_t v = static_cast<std::int32_t>(c); v >= 0; v = parent[static_cast<std::size_t>(v)]) {
            out.ancestor_masks[c] |= 1U << v;
        }
        if (c == 0) { continue; }
        out.drafts[c - 1] = token[c];
        // The parent's law, as each of its children stores it: the children's tokens first, then
        // fillers outside them (distinct ids, so a child's rank is its slot).
        const auto& kids = children[static_cast<std::size_t>(parent[c])];
        std::int32_t* ids = &out.candidate_ids[(c - 1) * kSlots];
        int slot          = 0;
        for (const std::int32_t kid : kids) { ids[slot++] = token[static_cast<std::size_t>(kid)]; }
        for (std::int32_t filler = token[c]; slot < kSlots;) {
            filler = (filler + 1) % domain;
            if (std::find(ids, ids + slot, filler) == ids + slot) { ids[slot++] = filler; }
        }
        out.proposal_q[(c - 1) * kSlots] = 1.0F;
    }
    return out;
}

// A chain row of `extent` drafts (tree 0): what a row of an n-gram tree round that verifies a chain (a
// copy without divergent sources, or a neural proposal) publishes. Acceptance verifies it as a chain over
// its first extent + 1 columns and writes its path itself, so path 0 lists only the columns that fit.
[[nodiscard]] inline ops::SpeculativeTreeRow ngram_chain_row(std::uint32_t extent) {
    if (extent >= static_cast<std::uint32_t>(ops::kSpeculativeTreeMaxNodes)) {
        throw std::invalid_argument("an n-gram tree round's chain row is wider than a tree row");
    }
    ops::SpeculativeTreeRow row{};
    const auto nodes = static_cast<std::int32_t>(extent) + 1;
    row.nodes        = nodes;
    row.main_depth   = static_cast<std::int32_t>(extent);
    row.paths        = 1;
    std::fill(std::begin(row.parent), std::end(row.parent), static_cast<std::int8_t>(-1));
    std::fill(std::begin(row.first_child), std::end(row.first_child), static_cast<std::int8_t>(-1));
    std::fill(std::begin(row.next_sibling), std::end(row.next_sibling), static_cast<std::int8_t>(-1));
    for (std::int32_t c = 0; c < nodes; ++c) {
        row.parent[c] = static_cast<std::int8_t>(c - 1);
        row.depth[c]  = static_cast<std::int8_t>(c);
        if (c + 1 < nodes) { row.first_child[c] = static_cast<std::int8_t>(c + 1); }
    }
    const std::int32_t listed = std::min(nodes, ops::kSpeculativeTreeMaxPathLength);
    row.path_length[0]        = static_cast<std::int8_t>(listed);
    for (std::int32_t i = 0; i < listed; ++i) { row.path_columns[0][i] = static_cast<std::int8_t>(i); }
    return row;
}

// The I32 ancestor masks of `row` over `width` columns (bit a of column c: a is c or an ancestor of c;
// columns past the row's live ones admit only themselves), into out[0..width).
inline void write_ngram_tree_masks(const ops::SpeculativeTreeRow& row, std::uint32_t width, std::span<std::uint32_t> out) {
    if (width > static_cast<std::uint32_t>(ops::kSpeculativeTreeMaxNodes) || out.size() < width) {
        throw std::invalid_argument("n-gram tree masks are wider than a tree row");
    }
    for (std::uint32_t c = 0; c < width; ++c) {
        std::uint32_t mask = 0;
        if (static_cast<std::int32_t>(c) < row.nodes) {
            for (std::int32_t v = static_cast<std::int32_t>(c); v >= 0; v = row.parent[v]) { mask |= 1U << v; }
        } else {
            mask = 1U << c;
        }
        out[c] = mask;
    }
}

} // namespace infernix::models::qwen3_5::detail
