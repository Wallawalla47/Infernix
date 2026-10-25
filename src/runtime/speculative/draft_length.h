#pragma once

// The calibrated draft length of a block drafter (--draft-tokens auto with DFlash or DFlash2;
// docs/maintainer/dflash.md). A proposal drafts its whole block in one pass, so a round's choice is
// how many of its drafts to verify, made after the proposal from each draft's probability (DFlash2:
// the candidate selector's temperature-1 softmax at the drafted rank; DFlash: the drafter head's
// softmax at its argmax): the count that maximizes the expected tokens per round time. A draft's
// acceptance given the earlier ones' is read from its probability's decile, learned per request,
// since a drafter's probability is not calibrated on its own.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace infernix::runtime::speculative {

struct DraftCalibration {
    static constexpr std::size_t kBins = 10;
    // Conditional acceptance of a draft whose probability falls in decile i.
    std::array<double, kBins> accept{};

    // Each decile's centre: a probability's face value until the request's rounds say otherwise.
    void reset() noexcept {
        for (std::size_t i = 0; i < kBins; ++i) { accept[i] = (static_cast<double>(i) + 0.5) / kBins; }
    }

    [[nodiscard]] static std::size_t bin(float probability) noexcept {
        return static_cast<std::size_t>(std::clamp(static_cast<int>(probability * kBins), 0, static_cast<int>(kBins) - 1));
    }

    // A verified round: drafts [0, verified) carried `probabilities`; the first `accepted` were
    // accepted and, when accepted < verified, draft `accepted` was the first rejected (later drafts
    // say nothing about their own acceptance). EWMA with `weight`.
    void observe(std::span<const float> probabilities, std::uint32_t verified, std::uint32_t accepted,
                 double weight) noexcept {
        for (std::uint32_t j = 0; j < verified && j <= accepted && j < probabilities.size(); ++j) {
            double& a = accept[bin(probabilities[j])];
            a         = (1.0 - weight) * a + weight * (j < accepted ? 1.0 : 0.0);
        }
    }

    bool operator==(const DraftCalibration&) const = default;
};

// Expected tokens a round verifying drafts [0, k) commits: 1 + the sum over j <= k of the product
// of the first j drafts' calibrated acceptances (the target's own token, then every draft whose
// whole prefix is accepted). k beyond the probabilities counts only the drafts there are.
[[nodiscard]] inline double expected_draft_tokens(const DraftCalibration& calibration,
                                                  std::span<const float> probabilities, std::int32_t k) noexcept {
    double reach = 1.0, tokens = 1.0;
    const auto limit = std::min<std::int32_t>(k, static_cast<std::int32_t>(probabilities.size()));
    for (std::int32_t j = 0; j < limit; ++j) {
        reach *= calibration.accept[DraftCalibration::bin(probabilities[static_cast<std::size_t>(j)])];
        tokens += reach;
    }
    return tokens;
}

// The number of drafts, in [0, most], a one-row round verifies: maximizing expected_draft_tokens
// / (1 + proposal + per_draft * k), costs in plain one-row rounds. k = 0 verifies nothing: the
// proposal's cost is already spent, so a draft is kept whenever it pays for its own verification
// column.
[[nodiscard]] inline std::int32_t calibrated_draft_length(const DraftCalibration& calibration,
                                                          std::span<const float> probabilities, std::int32_t most,
                                                          double per_draft, double proposal) noexcept {
    std::int32_t best = 0;
    double best_score = 1.0 / (1.0 + proposal), reach = 1.0, tokens = 1.0;
    const auto limit = std::min<std::int32_t>(most, static_cast<std::int32_t>(probabilities.size()));
    for (std::int32_t k = 1; k <= limit; ++k) {
        reach *= calibration.accept[DraftCalibration::bin(probabilities[static_cast<std::size_t>(k - 1)])];
        tokens += reach;
        const double score = tokens / (1.0 + proposal + per_draft * k);
        if (score > best_score) {
            best_score = score;
            best       = k;
        }
    }
    return best;
}

// The drafts per row, in [least, most], a two-row round verifies (Qwen3.8-Flash-Next dflash2-pairs):
// maximizing (tokens_a[k] + tokens_b[k]) / (1 + proposal + per_draft * k), costs in plain two-row
// rounds. The round verifies both rows over 1 + k columns whatever their own draft counts (a row with
// fewer is padded with filler columns of the same cost), so the width is the one choice and each row
// verifies every draft it has up to k: tokens_r[k] is row r's expected tokens with that prefix verified
// (expected_draft_tokens for a drafter row), non-decreasing in k, for k in [0, most]. `least` is the
// longest proposal the round must verify whole (an n-gram copy row's).
[[nodiscard]] inline std::int32_t pair_draft_width(std::span<const double> tokens_a, std::span<const double> tokens_b,
                                                   std::int32_t least, std::int32_t most, double per_draft,
                                                   double proposal) noexcept {
    most = std::min({most, static_cast<std::int32_t>(tokens_a.size()) - 1, static_cast<std::int32_t>(tokens_b.size()) - 1});
    least = std::clamp(least, 0, std::max(most, 0));
    std::int32_t best = least;
    double best_score = -1.0;
    for (std::int32_t k = least; k <= most; ++k) {
        const auto at     = static_cast<std::size_t>(k);
        const double score = (tokens_a[at] + tokens_b[at]) / (1.0 + proposal + per_draft * k);
        if (score > best_score) {
            best_score = score;
            best       = k;
        }
    }
    return best;
}

} // namespace infernix::runtime::speculative
