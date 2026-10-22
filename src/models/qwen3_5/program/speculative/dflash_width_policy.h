#pragma once

#include "infernix/types.h"
#include "runtime/speculative/draft_length.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace infernix::models::qwen3_5::detail {

// Verify-width policy of the adaptive-dflash fast path (FastPath::AdaptiveDFlash).
//
// An all-neural DFlash or DFlash2 round first proposes the drafter's whole block of K drafts with
// each draft's probability (DFlash2: the candidate selector's temperature-1 softmax at the drafted
// rank; DFlash: the drafter head's softmax at its argmax), then verifies one width W among the
// widths captured for its batch size, every row its leading min(W - 1, limit) drafts. The width
// maximizes the rows' expected committed tokens per unit of round time: a row's expected tokens
// follow from its drafts' probabilities through its request's calibration (runtime/speculative/
// draft_length.h), a round's time is DFlashWidthCosts::round (plain rounds of its batch size, each
// draft dearer the more rows verify it). Every
// kProbePeriod choices of a stream the next wider width runs, so the deciles past the chosen width
// stay measured. A configuration with a batch length (fast_paths.h, dflash_batch_drafts) runs its
// rounds of several rows as plain neural rounds at that length instead, one graph with no host
// wait between proposal and verification, and the policy chooses for one row only.
//
// A choice reads the round's own proposal, never the target's draws: speculative acceptance and
// sampling stay exact for any width. Only the verification width (and with it the target's linear
// routes) varies, which is what makes this a fast path.

// The verify widths (anchor included) rounds of one row choose among, and those of several rows:
// the ceiling's width K + 1 is always added. Each width is one captured verify/finish pair per
// batch size it serves.
inline constexpr std::array<std::uint32_t, 11> kDFlashRowWidths{2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16};
inline constexpr std::array<std::uint32_t, 4> kDFlashBatchWidths{2, 4, 8, 16};

// Round costs in plain rounds of the same batch size: the proposal (the drafter's block, paid
// whatever width follows) and each verified draft column of one row, the columns past `knee`
// drafts at their own rate (a target whose verification tiles fill at 8 columns pays little below
// and much above). A round of B rows verifies B columns per draft against a plain round whose
// weight traffic B rows share, so a draft costs B times one row's (measured 2026-10-20 on the
// NVFP4 27B: 0.07, 0.30 and 0.54 ms per draft at 1, 2 and 4 rows against 15 ms rounds).
struct DFlashWidthCosts {
    double proposal       = 0.1;
    double per_draft      = 0.1;
    double per_draft_wide = 0.1;
    std::uint32_t knee    = 7;

    // A round of `rows` rows verifying `drafts` drafts each.
    [[nodiscard]] double round(std::uint32_t drafts, std::uint32_t rows = 1) const noexcept {
        const std::uint32_t narrow = drafts < knee ? drafts : knee;
        const double scale         = static_cast<double>(rows);
        return 1.0 + proposal + scale * (per_draft * static_cast<double>(narrow) +
                                         per_draft_wide * static_cast<double>(drafts - narrow));
    }
};

// The widths, ascending, a ceiling of `drafts` gives rounds of one row (batch == false) or of
// several rows; with `zero`, also width 1: no draft verified (the round runs the narrowest family
// with every row's drafts masked, which only pays where a masked column costs nothing: offloaded
// experts, ops::MoeColumnMask).
[[nodiscard]] std::vector<std::uint32_t> dflash_policy_widths(std::uint32_t drafts, bool batch, bool zero = false);

// A request's memory of the policy: its drafts' calibration and its probe schedule. It lives with
// the request's sequence, so a request paused for preemption resumes (after Replay) with it.
struct DFlashLaneDraftState {
    runtime::speculative::DraftCalibration calibration;
    std::uint32_t rounds = 0; // choices made, for the probe period

    DFlashLaneDraftState() { calibration.reset(); }

    bool operator==(const DFlashLaneDraftState&) const = default;
};

// One row of a round to choose for: its proposal's K probabilities, its request's state and the
// most drafts it may verify (the output budget and KV capacity).
struct DFlashPolicyRow {
    std::span<const float> probabilities;
    DFlashLaneDraftState* state = nullptr;
    std::uint32_t limit         = 0;
};

class DFlashWidthPolicy {
public:
    static constexpr std::uint32_t kProbePeriod = 8;
    // EWMA weight of one verified draft in its decile's calibrated acceptance.
    static constexpr double kCalibrationWeight = 0.1;

    // batches == false: rounds of several rows are not the policy's (they run the configuration's
    // plain batch round, ProgramImpl::dflash_batch_drafts) and have no widths.
    // zero_drafts: width 1 (no draft verified) is a candidate (dflash_policy_widths).
    DFlashWidthPolicy(std::uint32_t ceiling, std::uint32_t max_batch, DFlashWidthCosts costs,
                      bool batches = true, bool zero_drafts = false);

    [[nodiscard]] std::uint32_t ceiling() const noexcept { return ceiling_; }
    [[nodiscard]] const DFlashWidthCosts& costs() const noexcept { return costs_; }

    // The widths a round of batch_size rows may verify, ascending (none: such rounds are not the
    // policy's).
    [[nodiscard]] const std::vector<std::uint32_t>& widths(std::uint32_t batch_size) const;

    // Expected committed tokens per plain round of a round of `rows` at `width`.
    [[nodiscard]] double rate(std::span<const DFlashPolicyRow> rows, std::uint32_t width) const;

    // The width of the next round of rows.size() rows: the highest rate, ties to the narrower width,
    // or every kProbePeriod choices of a one-row stream the next wider width than that. Advances the
    // rows' (one row: its request's; several: the batch size's) probe schedule.
    [[nodiscard]] std::uint32_t choose(std::span<const DFlashPolicyRow> rows);

    // A verified row: its first `verified` drafts were checked and the first `accepted` of them taken.
    static void observe(DFlashLaneDraftState& state, std::span<const float> probabilities,
                        std::uint32_t verified, std::uint32_t accepted) noexcept;

private:
    std::uint32_t ceiling_   = 0;
    std::uint32_t max_batch_ = 0;
    DFlashWidthCosts costs_;
    std::vector<std::uint32_t> row_widths_;
    std::vector<std::uint32_t> batch_widths_;
    std::array<std::uint32_t, kMaximumConcurrency> batch_rounds_{};
};

} // namespace infernix::models::qwen3_5::detail
