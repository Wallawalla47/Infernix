#pragma once

#include "infernix/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace infernix::models::qwen3_5::detail {

// Draft-length policy of the adaptive-mtp fast path (FastPath::AdaptiveMtp).
//
// An MTP round of draft length K verifies the drafts the previous round proposed (at most K of
// them, on K + 1 target columns) and proposes K drafts for the next round. Every length the policy
// may choose is its own captured round family. Before each all-neural round the policy picks the
// length with the most expected committed tokens per unit of round time: one row chooses among
// every length in [1, ceiling], a round of several rows takes one common length among the batch
// lengths (mtp_batch_draft_length). A row's expected tokens follow from its learned per-position
// acceptance; a round's time from a measured table (set_round_seconds; only tests give one) when
// one is given, else the cost learned from the rounds themselves (observe_cycle), else the fixed
// model 1 + kModelDraftCost * K.
//
// A choice reads only earlier rounds, never the round's own draws, so speculative acceptance and
// sampling are unchanged: only the verification width (and with it the target's linear routes)
// varies, which is what makes this a fast path. With the fixed model the choices are a deterministic
// function of the round sequence, so a fixed configuration and workload reproduce run to run.

// Positions an acceptance estimate tracks; at least every MTP draft window.
inline constexpr std::uint32_t kMtpPolicyMaximumDrafts = 8;

// Rounds of two or more rows choose among the lengths 1..kMtpBatchShortDrafts and the ceiling, so
// only those families are captured for every batch size; the other lengths serve one row.
inline constexpr std::uint32_t kMtpBatchShortDrafts = 3;

[[nodiscard]] constexpr bool mtp_batch_draft_length(std::uint32_t k,
                                                    std::uint32_t ceiling) noexcept {
    return k != 0 && k <= ceiling && (k <= kMtpBatchShortDrafts || k == ceiling);
}

// How far one request's MTP drafts get accepted. Draft j is tested only when drafts 0..j-1 were
// accepted, so position j estimates a conditional acceptance. Counts decay every observed round so
// the estimate follows phase changes within a request (prose, then code, then a tool call). Every
// position is shrunk towards the pooled acceptance of all positions, so a few rounds suffice and a
// position never tested takes the pooled value.
class MtpAcceptanceEstimate {
public:
    static constexpr double kDecay           = 0.9; // per observed round
    static constexpr double kPriorAcceptance = 0.7; // pooled prior before any round
    static constexpr double kPriorWeight     = 2.0; // rounds' worth of evidence of that prior
    static constexpr double kShrinkWeight    = 3.0; // tests' worth of pull towards the pool

    // A round verified `extent` drafts and accepted the first `accepted` of them.
    void observe(std::uint32_t extent, std::uint32_t accepted) noexcept;

    // The estimated acceptance of draft `position` given every earlier draft was accepted.
    [[nodiscard]] double acceptance(std::uint32_t position) const noexcept;

    // Expected tokens a round verifying `k` drafts commits: the target's own token plus every draft
    // whose whole prefix is accepted.
    [[nodiscard]] double expected_tokens(std::uint32_t k) const noexcept;

    bool operator==(const MtpAcceptanceEstimate&) const = default;

private:
    [[nodiscard]] double pooled() const noexcept;

    std::array<double, kMtpPolicyMaximumDrafts> tested_{};
    std::array<double, kMtpPolicyMaximumDrafts> accepted_{};
};

// The policy's memory of one choice stream (a request's single-row rounds, or one batch size's
// rounds): the length last chosen, which a challenger must beat by kSwitchMargin, and the probe
// schedule.
struct MtpDraftMemory {
    std::uint32_t draft_length = 0; // 0 before the first choice
    std::uint32_t rounds       = 0; // choices made, for the probe period
    std::uint32_t probe_length = 0;
    std::uint32_t probe_rounds = 0; // probe rounds still to run

    bool operator==(const MtpDraftMemory&) const = default;
};

// A request's draft-length state. It lives with the request's sequence, so a request paused for
// preemption resumes (after Replay) with the estimate it had learned.
struct MtpLaneDraftState {
    MtpAcceptanceEstimate acceptance;
    MtpDraftMemory memory;

    bool operator==(const MtpLaneDraftState&) const = default;
};

class MtpDraftPolicy {
public:
    // A challenger replaces the current length only when its rate is this much higher: switching
    // lengths wastes drafts (a longer round first verifies the shorter round's drafts).
    static constexpr double kSwitchMargin = 1.02;
    // Every kProbePeriod choices a stream runs the next longer length for kProbeRounds rounds: the
    // first proposes the longer drafts, the second verifies them, so the acceptance of the position
    // past the current choice stays measured.
    static constexpr std::uint32_t kProbePeriod = 16;
    static constexpr std::uint32_t kProbeRounds = 2;
    // Without measured or learned round times a round of K drafts costs 1 + kModelDraftCost * K.
    // The cost is the model's (2026-10-18, INT8 KV, from fixed-length runs: about 0.06 on the
    // NVFP4 27B, 0.19 on the 35B-A3B), so the learned fit replaces it once determined.
    static constexpr double kModelDraftCost = 0.2;
    // The learned fit: per observed cycle decay, the rounds and length spread it needs, the gap
    // filter, and the largest cost it reports.
    static constexpr double kCostDecay      = 0.98;
    static constexpr double kCostMinRounds  = 24.0;
    static constexpr double kCostMinSpread  = 0.1; // variance of k
    static constexpr double kCostGapRatio   = 3.0;
    static constexpr double kCostMax        = 1.0;

    // Lengths 1..ceiling for one row, the batch lengths for 2..max_batch rows.
    MtpDraftPolicy(std::uint32_t ceiling, std::uint32_t max_batch);

    [[nodiscard]] std::uint32_t ceiling() const noexcept { return ceiling_; }

    [[nodiscard]] std::uint32_t max_batch() const noexcept { return max_batch_; }

    // Whether a round of batch_size rows may run draft length k.
    [[nodiscard]] bool serves(std::uint32_t k, std::uint32_t batch_size) const noexcept;

    // The lengths a round of batch_size rows chooses among, ascending.
    [[nodiscard]] std::vector<std::uint32_t> lengths(std::uint32_t batch_size) const;

    // The measured time of a round of batch_size rows at draft length k (served lengths only). The
    // Engine sets none (it learns round costs, observe_cycle); tests use it to fix the costs.
    void set_round_seconds(std::uint32_t k, std::uint32_t batch_size, double seconds);

    // Round-cost learning (as Qwen3.8-Flash-Next's, its design 11.3): the start-to-start time of two
    // consecutive all-neural rounds of batch_size rows, the first of draft length k. Fit per batch
    // size as t = a + b k by exponentially weighted least squares; a draft then costs b / a rounds.
    // A cycle past kCostGapRatio predicted rounds held an idle gap and is skipped.
    void observe_cycle(std::uint32_t k, std::uint32_t batch_size, double seconds) noexcept;

    // The learned cost of one draft in rounds of batch_size rows, once its fit has kCostMinRounds
    // rounds of spread lengths and is positive.
    [[nodiscard]] std::optional<double> learned_draft_cost(std::uint32_t batch_size) const noexcept;

    // Whether every length of batch_size has a measured time.
    [[nodiscard]] bool measured(std::uint32_t batch_size) const noexcept;

    // The cost the policy compares: the measured time when the whole batch size is measured, else
    // the fixed model (comparable only within one batch size, which is all the policy compares).
    [[nodiscard]] double round_seconds(std::uint32_t k, std::uint32_t batch_size) const;

    // Expected committed tokens per unit of round time of `rows` at length k.
    [[nodiscard]] double rate(std::span<const MtpAcceptanceEstimate* const> rows,
                              std::uint32_t k) const;

    // The length of the next all-neural round of one row. Advances the lane's memory.
    [[nodiscard]] std::uint32_t choose(MtpLaneDraftState& lane) const;

    // The common length of the next all-neural round of rows.size() >= 2 rows. Advances that batch
    // size's memory.
    [[nodiscard]] std::uint32_t choose_batch(std::span<const MtpAcceptanceEstimate* const> rows);

private:
    [[nodiscard]] std::uint32_t step(MtpDraftMemory& memory,
                                     std::span<const MtpAcceptanceEstimate* const> rows) const;

    struct CostFit {
        double n = 0.0, k = 0.0, kk = 0.0, t = 0.0, kt = 0.0;
    };
    [[nodiscard]] static bool solve(const CostFit& fit, double& a, double& b) noexcept;

    std::uint32_t ceiling_   = 0;
    std::uint32_t max_batch_ = 0;
    std::array<CostFit, kMaximumConcurrency> cost_{};
    // [(batch_size - 1) * ceiling + k - 1]; 0 where unmeasured.
    std::vector<double> seconds_;
    std::array<MtpDraftMemory, kMaximumConcurrency> batches_{};
};

} // namespace infernix::models::qwen3_5::detail
