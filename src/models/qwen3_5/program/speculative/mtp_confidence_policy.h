#pragma once

#include "infernix/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace infernix::models::qwen3_5::detail {

// Confidence-gated MTP extents (Qwen3.5-122B-A10B design §12.2). Every MTP round proposes its next
// drafts with the drafter's probability of each (ops::argmax_with_probability over the proposal head);
// before the next round verifies them, each row keeps the prefix whose expected committed tokens per
// unit of round time is largest:
//
//   extent e maximizes (1 + sum_{j <= e} prod_{i <= j} a(q_i)) / (1 + c * e)
//
// where a(q) is the learned acceptance of a draft of drafter probability q (given its prefix was
// accepted) and c the cost of one verified draft column relative to a round without drafts. The
// verification round masks the columns past each row's extent (ops::MoeColumnMask), so with experts
// held outside device memory an unverified draft costs no expert traffic: there c is large (a
// column's misses, about 0.7 of a plain round on the 5090) and only confident drafts pay.
//
// a(q) is learned per probability bucket from the verified drafts (decayed counts, each bucket's
// prior its midpoint, as if the drafter were calibrated); c from the start-to-start time of
// consecutive rounds against their verified draft columns, fit per batch size as t = a + b x (c =
// b / a, as MtpDraftPolicy's round costs), with `prior_column_cost` until the fit has the rounds and
// spread it needs.
//
// The extent reads only the drafter's probabilities and earlier rounds, never the round's own draws,
// so acceptance and sampling are unchanged; on the exact route (column-invariant verification,
// Gumbel-max draws) the committed text is the same for every extent. On the 122B it is MTP's default
// with offloaded experts (design §0.2: 2 drafts +5.5 % over plain decode on sampled requests, 10 %
// over ungated 2-draft rounds greedy; a fixed probability threshold of 0.9 measured slower).

class MtpConfidencePolicy {
public:
    static constexpr std::size_t kBuckets  = 10;
    static constexpr double kDecay         = 0.995; // per observed draft
    static constexpr double kPriorWeight   = 4.0;   // drafts' worth of the calibrated prior
    static constexpr double kCostDecay     = 0.98;
    static constexpr double kCostMinRounds = 24.0;
    static constexpr double kCostMinSpread = 0.1; // variance of the verified draft columns
    static constexpr double kCostGapRatio  = 3.0; // a cycle this many predicted rounds held a gap
    static constexpr double kCostMax       = 4.0;

    explicit MtpConfidencePolicy(double prior_column_cost);

    // The drafts to verify of the proposed ones (probabilities q, in proposal order), at most
    // `ceiling`, for a round of `batch_size` rows.
    [[nodiscard]] std::uint32_t extent(std::span<const float> q, std::uint32_t ceiling,
                                       std::uint32_t batch_size) const noexcept;

    // A row verified `extent` drafts of probabilities q and accepted the first `accepted`.
    void observe(std::span<const float> q, std::uint32_t extent, std::uint32_t accepted) noexcept;

    // Two consecutive rounds of batch_size rows started `seconds` apart; the first verified
    // `columns` drafts over all its rows.
    void observe_round(std::uint32_t batch_size, std::uint32_t columns, double seconds) noexcept;

    // The learned acceptance of a draft of probability q.
    [[nodiscard]] double acceptance(float q) const noexcept;

    // The cost of one verified draft column relative to a round without drafts (learned or prior).
    [[nodiscard]] double column_cost(std::uint32_t batch_size) const noexcept;

    // TEMPORARY (INFERNIX_TMP_QWEN35_MTP_STEP_GATE, candidates §11 OR-09): the step chain's threshold
    // for a round of batch_size rows, the lowest bucket edge below which every bucket b has
    // a(b) a_max <= c (a_max the best bucket's acceptance; 2 when no bucket clears it). A draft j of
    // a probability below it leaves no later draft worth verifying: draft i > j is reached with at
    // most a(q_j) a_max^(i - j), so the drafts past j add at most a(q_j) a_max <= c tokens per
    // column, and by the mediant inequality no extent past j + 1 beats both extent j + 1 and the
    // plain round (rate 1); extent() keeps the shorter extent on a tie.
    [[nodiscard]] double step_threshold(std::uint32_t batch_size) const noexcept;

private:
    struct CostFit {
        double n = 0.0, x = 0.0, xx = 0.0, t = 0.0, xt = 0.0;
    };
    [[nodiscard]] static bool solve(const CostFit& fit, double& a, double& b) noexcept;
    [[nodiscard]] static std::size_t bucket(float q) noexcept;
    [[nodiscard]] double bucket_acceptance(std::size_t b) const noexcept;

    double prior_column_cost_ = 0.0;
    std::array<double, kBuckets> tested_{};
    std::array<double, kBuckets> accepted_{};
    std::array<CostFit, kMaximumConcurrency> cost_{};
};

// TEMPORARY (INFERNIX_TMP_QWEN35_MTP_SKIP_ROUNDS, candidates §11 OR-09): whether offloaded MTP rounds
// draft. A drafting round runs the drafter's steps whether or not the next round verifies their
// drafts (the 122B's greedy rounds mostly verify none: gated MTP -2 % against plain decode); a
// round without drafting runs only its alignment, the drafter's KV over the committed tokens, and
// the next round verifies no draft. Per batch size the regime keeps decayed committed tokens and
// seconds of each regime's steady cycles (a drafting round after a drafting one, a draft-free round
// after a draft-free one; a cycle is one round's start to the next's) and drafts while the drafting
// regime's tokens per second are at least the other's. Every kProbePeriod rounds the other regime
// runs kProbeRounds rounds (its first cycle a transition, not counted); until both regimes hold
// kMinCycles, runs of kProbeRounds alternate, drafting first. A cycle past kGapRatio times its
// regime's mean is a gap (no work), not counted. Sampling and acceptance are unchanged: rounds only
// verify fewer drafts.
class MtpDraftRegime {
public:
    static constexpr double kDecay              = 0.99; // per counted cycle of a regime
    static constexpr double kMinCycles          = 6.0;
    static constexpr double kGapRatio           = 3.0;
    static constexpr std::uint32_t kProbePeriod = 32;
    static constexpr std::uint32_t kProbeRounds = 4;

    // Whether the next round of batch_size rows drafts.
    [[nodiscard]] bool draft(std::uint32_t batch_size) noexcept;

    // A cycle of batch_size rows whose round drafted (or not) after a round that drafted (or not)
    // committed `tokens` over all its rows in `seconds`.
    void observe(std::uint32_t batch_size, bool previous, bool drafted, std::uint32_t tokens, double seconds) noexcept;

    [[nodiscard]] std::uint64_t rounds() const noexcept { return rounds_; }
    [[nodiscard]] std::uint64_t draft_free_rounds() const noexcept { return draft_free_; }

private:
    struct Regime {
        double tokens = 0.0, seconds = 0.0, cycles = 0.0;
    };
    struct Batch {
        std::array<Regime, 2> regimes{}; // [0] draft-free, [1] drafting
        std::uint32_t round = 0;
    };
    std::array<Batch, kMaximumConcurrency> batches_{};
    std::uint64_t rounds_     = 0;
    std::uint64_t draft_free_ = 0;
};

} // namespace infernix::models::qwen3_5::detail
