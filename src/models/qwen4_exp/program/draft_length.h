#pragma once

// Qwen4Exp's DFlash2 draft-length policy (--draft-tokens auto; design §11.3, docs/maintainer/dflash.md).
// A proposal drafts its whole block in one pass, so a round's choice is how many of its drafts to
// verify, made after the proposal from each draft's selector probability (the candidate selector's
// temperature-1 softmax at the drafted rank): the k that maximizes the expected tokens per round
// time. A draft's acceptance given the earlier ones' is read from its probability's decile, learned
// per request, since the selector's probability over its 16 candidates is not calibrated on its own.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace infernix::models::qwen4_exp::detail {

struct SelectorCalibration {
    static constexpr std::size_t kBins = 10;
    // Conditional acceptance of a draft whose selector probability falls in decile i.
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
};

// The number of drafts, in [0, most], a one-row round verifies: maximizing (1 + sum over j <= k of
// the product of the first j drafts' calibrated acceptances) / (1 + proposal + per_draft * k),
// costs in plain one-row rounds. k = 0 verifies nothing: the proposal's cost is already spent, so a
// draft is kept whenever it pays for its own verification column.
[[nodiscard]] inline std::int32_t calibrated_draft_length(const SelectorCalibration& calibration,
                                                          std::span<const float> probabilities, std::int32_t most,
                                                          double per_draft, double proposal) noexcept {
    std::int32_t best = 0;
    double best_score = 1.0 / (1.0 + proposal), reach = 1.0, tokens = 1.0;
    const auto limit = std::min<std::int32_t>(most, static_cast<std::int32_t>(probabilities.size()));
    for (std::int32_t k = 1; k <= limit; ++k) {
        reach *= calibration.accept[SelectorCalibration::bin(probabilities[static_cast<std::size_t>(k - 1)])];
        tokens += reach;
        const double score = tokens / (1.0 + proposal + per_draft * k);
        if (score > best_score) {
            best_score = score;
            best       = k;
        }
    }
    return best;
}

} // namespace infernix::models::qwen4_exp::detail
