#include "models/qwen3_5/program/speculative/tree_width_controller.h"

#include <algorithm>
#include <stdexcept>

namespace infernix::models::qwen3_5::detail {

namespace {

// Tokens a round of `columns` columns emits on a row whose accepted path is `path`: the drafts
// accepted before the path first uses a column the narrower tree lacks, plus the correction or
// bonus token.
double row_tokens(std::span<const std::int32_t> path, std::uint32_t columns) {
    for (std::size_t i = 1; i < path.size(); ++i) {
        if (path[i] >= static_cast<std::int32_t>(columns)) { return static_cast<double>(i); }
    }
    return static_cast<double>(path.size());
}

// Rounds measured before a width's timing is trusted; until then the smallest sample stands,
// since interference only lengthens a round.
constexpr std::uint32_t kInitialSamples = 3;

} // namespace

TreeWidthController::TreeWidthController(
    std::uint32_t chain_columns,
    const std::array<std::vector<std::uint32_t>, kMaximumConcurrency>& trees)
    : chain_columns_(chain_columns) {
    for (std::size_t b = 0; b < kMaximumConcurrency; ++b) {
        candidates_[b].push_back(chain_columns);
        for (const std::uint32_t columns : trees[b]) {
            if (columns <= candidates_[b].back()) {
                throw std::invalid_argument(
                    "tree widths must be ascending and wider than the chain");
            }
            candidates_[b].push_back(columns);
            if (std::find(tree_widths_.begin(), tree_widths_.end(), columns) ==
                tree_widths_.end()) {
                tree_widths_.push_back(columns);
            }
        }
        for (Cell& cell : cells_[b]) { cell.timings.resize(candidates_[b].size()); }
    }
    std::sort(tree_widths_.begin(), tree_widths_.end());
    gains_.resize(tree_widths_.size());
}

std::size_t TreeWidthController::context_bucket(std::uint32_t context) noexcept {
    std::size_t bucket = 0;
    for (std::uint64_t edge = 8192; bucket + 1 < kContextBuckets && context >= edge; edge *= 2) {
        ++bucket;
    }
    return bucket;
}

double TreeWidthController::gain(std::uint32_t columns) const {
    if (columns == chain_columns_) { return 1.0; }
    const auto it = std::find(tree_widths_.begin(), tree_widths_.end(), columns);
    if (it == tree_widths_.end()) { return 0.0; }
    const Gain& g = gains_[static_cast<std::size_t>(it - tree_widths_.begin())];
    return g.rows >= kMinimumGainRows && g.chain > 0 ? g.tokens / g.chain : 0.0;
}

std::uint32_t TreeWidthController::choose(std::uint32_t batch_size, std::uint32_t context) {
    if (batch_size == 0 || batch_size > kMaximumConcurrency) {
        throw std::invalid_argument("tree width controller: invalid batch size");
    }
    const std::vector<std::uint32_t>& candidates = candidates_[batch_size - 1U];
    if (candidates.size() == 1) { return chain_columns_; }
    Cell& cell                = cells_[batch_size - 1U][context_bucket(context)];
    const std::uint64_t round = ++cell.rounds;
    // Widths still being timed come first, widest first: a wide round also measures the gain of
    // every narrower width.
    for (std::size_t j = candidates.size(); j-- > 0;) {
        if (cell.timings[j].samples < kInitialSamples) { return candidates[j]; }
    }
    const std::uint32_t widest = candidates.back();
    for (std::size_t j = 1; j < candidates.size(); ++j) {
        if (gain(candidates[j]) == 0.0) { return widest; }
    }
    if (round % kExplorePeriod == 0) { return widest; }
    for (std::size_t j = 0; j < candidates.size(); ++j) {
        if (round - cell.timings[j].measured > kStaleRounds) { return candidates[j]; }
    }
    std::size_t best = 0;
    double best_rate = 0.0;
    for (std::size_t j = 0; j < candidates.size(); ++j) {
        const double seconds = cell.timings[j].seconds;
        const double rate    = seconds > 0.0 ? gain(candidates[j]) / seconds : 0.0;
        if (rate > best_rate) {
            best_rate = rate;
            best      = j;
        }
    }
    return candidates[best];
}

void TreeWidthController::observe_tree_row(std::span<const std::int32_t> path,
                                           std::uint32_t columns) {
    if (path.empty() || path[0] != 0) {
        throw std::invalid_argument("tree width controller: a path starts at the anchor column");
    }
    const double chain = row_tokens(path, chain_columns_);
    for (std::size_t w = 0; w < tree_widths_.size() && tree_widths_[w] <= columns; ++w) {
        Gain& g  = gains_[w];
        g.tokens = g.tokens * kGainDecay + row_tokens(path, tree_widths_[w]);
        g.chain  = g.chain * kGainDecay + chain;
        g.rows += 1;
    }
}

void TreeWidthController::observe_round(std::uint32_t batch_size, std::uint32_t context,
                                        std::uint32_t columns, double seconds) {
    if (batch_size == 0 || batch_size > kMaximumConcurrency || !(seconds > 0.0)) {
        throw std::invalid_argument("tree width controller: invalid round observation");
    }
    const std::vector<std::uint32_t>& candidates = candidates_[batch_size - 1U];
    if (candidates.size() == 1) { return; }
    const auto it = std::find(candidates.begin(), candidates.end(), columns);
    if (it == candidates.end()) {
        throw std::invalid_argument("tree width controller: round width is not a candidate");
    }
    Cell& cell     = cells_[batch_size - 1U][context_bucket(context)];
    Timing& timing = cell.timings[static_cast<std::size_t>(it - candidates.begin())];
    if (timing.samples < kInitialSamples) {
        timing.seconds = timing.samples == 0 ? seconds : std::min(timing.seconds, seconds);
    } else {
        // A late spike moves the average by at most half its value.
        const double bounded = std::clamp(seconds, 0.5 * timing.seconds, 1.5 * timing.seconds);
        timing.seconds += kTimeWeight * (bounded - timing.seconds);
    }
    timing.samples += 1;
    timing.measured = cell.rounds;
}

} // namespace infernix::models::qwen3_5::detail
