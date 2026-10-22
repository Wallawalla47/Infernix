#pragma once

#include "infernix/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace infernix::models::qwen3_5::detail {

// Confidence-gated MTP extents (Qwen3.5-122B-A10B design §12.2). Every MTP round proposes its next
// drafts with the drafter's probability of each (ops::argmax_probability over the proposal head);
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

private:
    struct CostFit {
        double n = 0.0, x = 0.0, xx = 0.0, t = 0.0, xt = 0.0;
    };
    [[nodiscard]] static bool solve(const CostFit& fit, double& a, double& b) noexcept;
    [[nodiscard]] static std::size_t bucket(float q) noexcept;

    double prior_column_cost_ = 0.0;
    std::array<double, kBuckets> tested_{};
    std::array<double, kBuckets> accepted_{};
    std::array<CostFit, kMaximumConcurrency> cost_{};
};

} // namespace infernix::models::qwen3_5::detail
