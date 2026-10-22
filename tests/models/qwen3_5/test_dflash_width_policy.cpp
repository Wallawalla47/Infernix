#include "models/qwen3_5/program/speculative/dflash_width_policy.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// adaptive-dflash verify-width policy (DFlashWidthPolicy): the width ladders a ceiling gives one row
// and several rows (none when batches run the plain batch round), the choice of the width with the most expected tokens per round time against an
// exhaustive evaluation (one row and batches, with output limits), ties to the narrower width, the
// periodic probe of the next wider width, and a request's calibration updates.
namespace {

using infernix::models::qwen3_5::detail::dflash_policy_widths;
using infernix::models::qwen3_5::detail::DFlashLaneDraftState;
using infernix::models::qwen3_5::detail::DFlashPolicyRow;
using infernix::models::qwen3_5::detail::DFlashWidthCosts;
using infernix::models::qwen3_5::detail::DFlashWidthPolicy;
using infernix::runtime::speculative::DraftCalibration;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

// Expected tokens of verifying the first `extent` drafts, as a direct sum of products.
double direct_tokens(const DraftCalibration& c, const std::vector<float>& p, std::uint32_t extent) {
    double tokens = 1.0;
    for (std::uint32_t j = 1; j <= std::min<std::uint32_t>(extent, static_cast<std::uint32_t>(p.size())); ++j) {
        double reach = 1.0;
        for (std::uint32_t i = 0; i < j; ++i) { reach *= c.accept[DraftCalibration::bin(p[i])]; }
        tokens += reach;
    }
    return tokens;
}

// The width an exhaustive search over `widths` picks: the highest rate, the narrower on ties, never
// wider than the narrowest width holding the longest row's limit.
std::uint32_t exhaustive(const std::vector<std::uint32_t>& widths, const std::vector<std::vector<float>>& p,
                         const std::vector<DFlashLaneDraftState>& states, const std::vector<std::uint32_t>& limits,
                         DFlashWidthCosts costs) {
    std::uint32_t most = 0;
    for (const std::uint32_t limit : limits) { most = std::max(most, limit); }
    std::uint32_t best = 0;
    double best_rate   = -1.0;
    for (const std::uint32_t width : widths) {
        double tokens = 0.0;
        for (std::size_t r = 0; r < p.size(); ++r) {
            tokens += direct_tokens(states[r].calibration, p[r], std::min(width - 1U, limits[r]));
        }
        const double drafts = width - 1.0;
        const double narrow = std::min(drafts, static_cast<double>(costs.knee));
        const double rows   = static_cast<double>(p.size());
        const double rate   = tokens / (1.0 + costs.proposal +
                                      rows * (costs.per_draft * narrow + costs.per_draft_wide * (drafts - narrow)));
        if (rate > best_rate) {
            best_rate = rate;
            best      = width;
        }
        if (width >= most + 1U) { break; }
    }
    return best;
}

} // namespace

int main() {
    // Ladders: every listed width up to the ceiling's, which is always present.
    check(dflash_policy_widths(15, false) == std::vector<std::uint32_t>{2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16},
          "one-row widths of a 15-draft ceiling");
    check(dflash_policy_widths(15, true) == std::vector<std::uint32_t>{2, 4, 8, 16}, "batch widths of a 15-draft ceiling");
    check(dflash_policy_widths(7, false) == std::vector<std::uint32_t>{2, 3, 4, 5, 6, 7, 8}, "one-row widths of 7 drafts");
    check(dflash_policy_widths(10, true) == std::vector<std::uint32_t>{2, 4, 8, 11}, "batch widths add the ceiling's");
    check(dflash_policy_widths(1, true) == std::vector<std::uint32_t>{2}, "one draft verifies two columns");
    // A policy that leaves batches to the plain batch round has no batch widths and refuses to
    // choose one, while it still chooses for one row.
    {
        DFlashWidthPolicy rows_only(15, 2, {.proposal = 0.1, .per_draft = 0.01, .per_draft_wide = 0.01}, false);
        check(rows_only.widths(1).size() == 11 && rows_only.widths(2).empty(), "a one-row policy's widths");
        DFlashLaneDraftState a, b;
        const std::vector<float> sure(15, 0.97F);
        DFlashPolicyRow one{sure, &a, 15};
        check(rows_only.choose(std::span<const DFlashPolicyRow>(&one, 1)) == 16, "a one-row policy chooses one row");
        const std::vector<DFlashPolicyRow> two{{sure, &a, 15}, {sure, &b, 15}};
        bool refused = false;
        try {
            (void)rows_only.choose(two);
        } catch (const std::logic_error&) { refused = true; }
        check(refused, "a one-row policy refuses rounds of several rows");
    }

    const DFlashWidthCosts costs{.proposal = 0.1, .per_draft = 0.1, .per_draft_wide = 0.1};
    // Confident drafts take the whole block, doubtful ones the narrowest width.
    {
        DFlashWidthPolicy policy(15, 2, costs);
        DFlashLaneDraftState state;
        const std::vector<float> sure(15, 0.97F), doubtful(15, 0.05F);
        DFlashPolicyRow row{sure, &state, 15};
        check(policy.choose(std::span<const DFlashPolicyRow>(&row, 1)) == 16, "confident drafts verify the block");
        row.probabilities = doubtful;
        check(policy.choose(std::span<const DFlashPolicyRow>(&row, 1)) == 2, "doubtful drafts verify one");
        // An output limit of 3 drafts bounds the width at the narrowest holding them.
        row.probabilities = sure;
        row.limit         = 3;
        check(policy.choose(std::span<const DFlashPolicyRow>(&row, 1)) == 4, "the limit bounds the width");
    }
    // Against the exhaustive choice, one row and two rows, random probabilities, limits and costs.
    std::uint32_t seed = 11;
    const auto uniform = [&] {
        seed = seed * 1664525U + 1013904223U;
        return static_cast<float>(seed >> 8) / static_cast<float>(1U << 24);
    };
    for (int trial = 0; trial < 3000; ++trial) {
        const std::uint32_t rows    = 1U + static_cast<std::uint32_t>(trial % 2);
        const std::uint32_t ceiling = 1U + static_cast<std::uint32_t>(trial % 15);
        // Equal slopes (the linear model) and a cheap-then-dear knee at 3, 7 or 11 drafts.
        const DFlashWidthCosts c{.proposal       = 0.05 * (trial % 5),
                                 .per_draft      = 0.02 + 0.04 * (trial % 7),
                                 .per_draft_wide = trial % 3 == 0 ? 0.02 + 0.04 * (trial % 7) : 0.05 + 0.03 * (trial % 4),
                                 .knee           = 3U + 4U * static_cast<std::uint32_t>(trial % 3)};
        DFlashWidthPolicy policy(ceiling, 2, c);
        std::vector<std::vector<float>> p(rows, std::vector<float>(ceiling));
        std::vector<DFlashLaneDraftState> states(rows);
        std::vector<std::uint32_t> limits(rows);
        for (std::uint32_t r = 0; r < rows; ++r) {
            for (float& v : p[r]) { v = uniform(); }
            for (double& a : states[r].calibration.accept) { a = uniform(); }
            limits[r] = static_cast<std::uint32_t>(uniform() * static_cast<float>(ceiling + 1U));
            states[r].rounds = 1; // keep the probe period away from this choice
        }
        std::vector<DFlashPolicyRow> view;
        for (std::uint32_t r = 0; r < rows; ++r) { view.push_back({p[r], &states[r], limits[r]}); }
        const std::uint32_t expected = exhaustive(policy.widths(rows), p, states, limits, c);
        check(policy.choose(view) == expected, "the choice equals the exhaustive one (trial " + std::to_string(trial) + ")");
    }
    // With width 1 allowed (offloaded experts: a masked column routes no expert), doubtful drafts
    // verify nothing when a draft column costs most of a round (the 122B's 0.72), confident ones still
    // verify; the exhaustive choice over the ladder with 1 agrees.
    {
        check(dflash_policy_widths(7, false, true) == std::vector<std::uint32_t>{1, 2, 3, 4, 5, 6, 7, 8},
              "the zero-draft ladder starts at width 1");
        const DFlashWidthCosts dear{.proposal = 0.12, .per_draft = 0.72, .per_draft_wide = 0.72};
        DFlashWidthPolicy policy(15, 2, dear, true, true);
        DFlashLaneDraftState state;
        state.rounds = 1;
        const std::vector<float> doubtful(15, 0.05F), sure(15, 0.995F);
        DFlashPolicyRow row{doubtful, &state, 15};
        check(policy.choose(std::span<const DFlashPolicyRow>(&row, 1)) == 1, "doubtful drafts verify none");
        row.probabilities = sure;
        const std::uint32_t confident = policy.choose(std::span<const DFlashPolicyRow>(&row, 1));
        check(confident >= 2, "confident drafts still verify at a dear column cost");
        const std::vector<std::vector<float>> p{sure};
        const std::vector<DFlashLaneDraftState> states{state};
        check(confident == exhaustive(policy.widths(1), p, states, {15}, dear), "the zero-draft choice is exhaustive's");
    }
    // A knee keeps confident drafts narrow when the columns past it are dear: 15 drafts at 0.97
    // verify the knee's width (8 columns) at a cheap-then-dear cost, the whole block when flat.
    {
        DFlashWidthPolicy dear(15, 1, {.proposal = 0.2, .per_draft = 0.003, .per_draft_wide = 0.12, .knee = 7});
        DFlashLaneDraftState state;
        state.rounds = 1;
        const std::vector<float> sure(15, 0.97F);
        DFlashPolicyRow row{sure, &state, 15};
        check(dear.choose(std::span<const DFlashPolicyRow>(&row, 1)) == 8, "a dear knee keeps the width at 8 columns");
    }
    // Two rows pay each draft twice: drafts of 0.85 calibrated acceptance a lone row verifies to
    // 14 columns stop at 8 in a round of two (0.03 per draft).
    {
        DFlashWidthPolicy policy(15, 2, {.proposal = 0.1, .per_draft = 0.03, .per_draft_wide = 0.03});
        DFlashLaneDraftState a, b;
        a.rounds = b.rounds = 1;
        const std::vector<float> likely(15, 0.86F);
        DFlashPolicyRow one{likely, &a, 15};
        check(policy.choose(std::span<const DFlashPolicyRow>(&one, 1)) == 14, "one row verifies 14 columns");
        const std::vector<DFlashPolicyRow> two{{likely, &a, 15}, {likely, &b, 15}};
        check(policy.choose(two) == 8, "two rows verify narrower");
    }
    // Every kProbePeriod-th choice of a row runs the next wider width.
    {
        DFlashWidthPolicy policy(15, 1, costs);
        DFlashLaneDraftState state;
        const std::vector<float> doubtful(15, 0.05F);
        DFlashPolicyRow row{doubtful, &state, 15};
        std::vector<std::uint32_t> chosen;
        for (std::uint32_t i = 0; i < 2 * DFlashWidthPolicy::kProbePeriod; ++i) {
            chosen.push_back(policy.choose(std::span<const DFlashPolicyRow>(&row, 1)));
        }
        check(chosen[DFlashWidthPolicy::kProbePeriod - 1] == 3 && chosen[2 * DFlashWidthPolicy::kProbePeriod - 1] == 3,
              "the probe runs the next wider width");
        check(std::count(chosen.begin(), chosen.end(), 2U) == static_cast<std::ptrdiff_t>(chosen.size() - 2),
              "other choices stay at the best width");
    }
    // Calibration: a verified row moves its accepted drafts' deciles up and the first rejected one down.
    {
        DFlashLaneDraftState state;
        const std::vector<float> p{0.92F, 0.71F, 0.33F};
        DFlashWidthPolicy::observe(state, p, 3, 1);
        check(state.calibration.accept[9] > 0.95 && state.calibration.accept[7] < 0.75 &&
                  std::abs(state.calibration.accept[3] - 0.35) < 1e-12,
              "observe updates the verified deciles only");
    }
    if (failures != 0) { return 1; }
    std::cout << "dflash width policy checks passed\n";
    return 0;
}
