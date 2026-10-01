#include "models/qwen3_5/program/speculative/tree_width_controller.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// Automatic DFlash2 tree widths (TreeWidthController): exact same-text gains of every narrower
// width from a wide round's accepted paths, the exploration order, the choice that maximizes
// tokens per second in each context bucket, and its bounded response to timing spikes.
namespace {

using ninfer::kMaximumConcurrency;
using ninfer::models::qwen3_5::detail::TreeWidthController;
using Trees = std::array<std::vector<std::uint32_t>, kMaximumConcurrency>;

constexpr std::uint32_t kChain = 8; // draft_tokens 7

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

Trees trees_upto(std::uint32_t batches, const std::vector<std::uint32_t>& widths) {
    Trees trees;
    for (std::uint32_t b = 0; b < batches; ++b) { trees[b] = widths; }
    return trees;
}

// The accepted path a round of `columns` columns takes on a row whose widest-tree path is
// `path`: the same columns until the first one the narrower tree lacks.
std::vector<std::int32_t> narrowed(const std::vector<std::int32_t>& path, std::uint32_t columns) {
    std::vector<std::int32_t> out;
    for (const std::int32_t column : path) {
        if (column >= static_cast<std::int32_t>(columns)) { break; }
        out.push_back(column);
    }
    return out;
}

// Rows with known gains: totals over one cycle are chain 7, 12 columns 9, 16 columns 11 tokens.
const std::vector<std::vector<std::int32_t>> kRows{
    {0, 1, 2, 3},  // stays on the main chain: 4 tokens at every width
    {0, 1, 9, 10}, // leaves it at depth 2 for column 9: chain 2, 12 and 16 columns 4
    {0, 13, 14},   // leaves it at depth 1 for column 13: chain and 12 columns 1, 16 columns 3
};

struct Simulation {
    TreeWidthController controller;
    std::function<double(std::uint32_t columns, std::uint32_t context)> seconds;
    std::size_t row = 0;

    std::uint32_t round(std::uint32_t batch, std::uint32_t context) {
        const std::uint32_t columns = controller.choose(batch, context);
        if (columns != kChain) {
            for (std::uint32_t r = 0; r < batch; ++r) {
                const auto path = narrowed(kRows[row++ % kRows.size()], columns);
                controller.observe_tree_row(path, columns);
            }
        }
        controller.observe_round(batch, context, columns, seconds(columns, context));
        return columns;
    }
};

void context_buckets() {
    const std::vector<std::pair<std::uint32_t, std::size_t>> cases{
        {0, 0},     {8191, 0},  {8192, 1},   {16383, 1},  {16384, 2},
        {65535, 3}, {65536, 4}, {131071, 4}, {131072, 5}, {1048576, 5}};
    for (const auto& [context, bucket] : cases) {
        check(TreeWidthController::context_bucket(context) == bucket,
              "context bucket of " + std::to_string(context));
    }
}

bool near(double value, double expected, double relative) {
    return std::abs(value - expected) <= relative * expected;
}

void gains_from_wide_rounds() {
    // Rows of one kind give exact ratios: chain 1 token, 12 columns 1, 16 columns 3.
    TreeWidthController same(kChain, trees_upto(1, {12, 16}));
    check(same.gain(kChain) == 1.0 && same.gain(12) == 0.0 && same.gain(16) == 0.0,
          "no gain before any row");
    for (int i = 0; i < 31; ++i) { same.observe_tree_row(kRows[2], 16); }
    check(same.gain(16) == 0.0, "a gain needs 32 rows");
    same.observe_tree_row(kRows[2], 16);
    check(near(same.gain(12), 1.0, 1e-12) && near(same.gain(16), 3.0, 1e-12),
          "identical rows give the exact token ratios");
    // A mix: totals 7 (chain), 9 (12 columns), 11 (16 columns) per cycle; the decay weights the
    // latest rows slightly more.
    TreeWidthController mixed(kChain, trees_upto(1, {12, 16}));
    for (int cycle = 0; cycle < 40; ++cycle) {
        for (const auto& path : kRows) { mixed.observe_tree_row(path, 16); }
    }
    check(near(mixed.gain(12), 9.0 / 7.0, 5e-3), "12-column gain is 9/7");
    check(near(mixed.gain(16), 11.0 / 7.0, 5e-3), "16-column gain is 11/7");
    // A 12-column round says nothing about 16 columns.
    TreeWidthController narrow(kChain, trees_upto(1, {12, 16}));
    for (int cycle = 0; cycle < 40; ++cycle) {
        for (const auto& path : kRows) { narrow.observe_tree_row(narrowed(path, 12), 12); }
    }
    check(near(narrow.gain(12), 9.0 / 7.0, 5e-3) && narrow.gain(16) == 0.0,
          "a 12-column round measures only 12 columns");
}

void exploration_order() {
    Simulation sim{TreeWidthController(kChain, trees_upto(1, {12, 16})),
                   [](std::uint32_t columns, std::uint32_t) { return 0.010 + 1e-4 * columns; }};
    std::vector<std::uint32_t> first;
    for (int i = 0; i < 9; ++i) { first.push_back(sim.round(1, 1000)); }
    check(first == std::vector<std::uint32_t>{16, 16, 16, 12, 12, 12, 8, 8, 8},
          "cold start times the widest width first, three rounds each");
    // A new context bucket is timed again.
    check(sim.round(1, 70000) == 16, "a new context bucket starts with the widest width");
    // Batch sizes without tree widths verify chains and ignore their timings.
    check(sim.controller.choose(5, 1000) == kChain, "rounds of five rows verify chains");
    sim.controller.observe_round(5, 1000, kChain, 0.02);
}

std::map<std::uint32_t, int> steady_choices(Simulation& sim, std::uint32_t batch,
                                            std::uint32_t context, int warmup, int rounds) {
    for (int i = 0; i < warmup; ++i) { sim.round(batch, context); }
    std::map<std::uint32_t, int> counts;
    for (int i = 0; i < rounds; ++i) { counts[sim.round(batch, context)] += 1; }
    return counts;
}

void choice_per_context() {
    // Short context: 16 columns cost 10 % for 57 % more tokens. Long context: every extra column
    // costs attention, and the chain wins.
    Simulation sim{TreeWidthController(kChain, trees_upto(2, {12, 16})),
                   [](std::uint32_t columns, std::uint32_t context) {
                       const double base = 0.010;
                       if (context < 65536) { return base * (1.0 + 0.0125 * (columns - kChain)); }
                       return base * (1.0 + 0.1000 * (columns - kChain));
                   }};
    // Besides 16 columns, only the stale-timing refreshes of the chain and 12 columns.
    const auto short_counts = steady_choices(sim, 1, 4000, 200, 1024);
    check(short_counts.at(16) >= 1024 - 6, "short context settles on 16 columns");
    const auto long_counts = steady_choices(sim, 1, 100000, 200, 1024);
    // The chain wins; the widest width is explored every 32 rounds, the other one when stale.
    check(long_counts.at(kChain) >= 1024 - 1024 / 32 - 4, "long context settles on the chain");
    check(long_counts.at(16) >= 1024 / 32 - 1 && long_counts.at(16) <= 1024 / 32 + 3,
          "the widest width is explored every 32 rounds");
    check(long_counts.count(12) == 0 || long_counts.at(12) <= 3,
          "the middle width is only refreshed when stale");
    // Batch size 2 has its own timings.
    check(steady_choices(sim, 2, 4000, 200, 256).at(16) >= 250, "two rows time separately");
}

void middle_width_wins() {
    // 12 columns gain 9/7 at 2 % cost; 16 columns gain 11/7 at 30 % cost.
    Simulation sim{TreeWidthController(kChain, trees_upto(1, {12, 16})),
                   [](std::uint32_t columns, std::uint32_t) {
                       return columns == 16 ? 0.0130 : columns == 12 ? 0.0102 : 0.0100;
                   }};
    const auto counts = steady_choices(sim, 1, 4000, 200, 1024);
    check(counts.at(12) >= 1024 - 1024 / 32 - 4, "the middle width wins when it pays most");
}

void timing_spike_is_bounded() {
    double spike = 1.0;
    Simulation sim{TreeWidthController(kChain, trees_upto(1, {12, 16})),
                   [&spike](std::uint32_t columns, std::uint32_t) {
                       return spike * (columns == 16 ? 0.0110 : columns == 12 ? 0.0105 : 0.0100);
                   }};
    (void)steady_choices(sim, 1, 4000, 200, 64);
    // One round twenty times slower moves the 16-column estimate by at most 1/16, which does not
    // change the choice.
    spike = 20.0;
    (void)sim.round(1, 4000);
    spike = 1.0;
    check(sim.round(1, 4000) == 16, "one timing spike does not change the choice");
}

void invalid_inputs() {
    bool threw = false;
    try {
        TreeWidthController bad(kChain, trees_upto(1, {16, 12}));
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "descending tree widths are rejected");
    TreeWidthController controller(kChain, trees_upto(1, {12, 16}));
    threw = false;
    try {
        controller.observe_round(1, 1000, 20, 0.01);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a width the batch size cannot verify is rejected");
    threw = false;
    try {
        controller.observe_tree_row(std::vector<std::int32_t>{1, 2}, 16);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a path that does not start at the anchor is rejected");
}

} // namespace

int main() {
    try {
        context_buckets();
        gains_from_wide_rounds();
        exploration_order();
        choice_per_context();
        middle_width_wins();
        timing_spike_is_bounded();
        invalid_inputs();
    } catch (const std::exception& error) {
        std::cerr << "tree width controller test threw: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " tree width controller check(s) failed\n";
        return 1;
    }
    std::cout << "tree width controller tests passed\n";
    return 0;
}
