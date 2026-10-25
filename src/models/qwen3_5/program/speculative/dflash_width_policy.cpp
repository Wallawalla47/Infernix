#include "models/qwen3_5/program/speculative/dflash_width_policy.h"

#include "core/temporary_switches.h"

#include <algorithm>
#include <stdexcept>

namespace infernix::models::qwen3_5::detail {

std::vector<std::uint32_t> dflash_policy_widths(std::uint32_t drafts, bool batch, bool zero) {
    std::vector<std::uint32_t> out;
    if (zero) { out.push_back(1U); }
    const auto add = [&](std::uint32_t width) {
        if (width >= 2U && width <= drafts + 1U && std::find(out.begin(), out.end(), width) == out.end()) {
            out.push_back(width);
        }
    };
    if (batch) {
        for (const std::uint32_t width : kDFlashBatchWidths) { add(width); }
    } else {
        for (const std::uint32_t width : kDFlashRowWidths) { add(width); }
    }
    add(drafts + 1U);
    std::sort(out.begin(), out.end());
    return out;
}

DFlashWidthPolicy::DFlashWidthPolicy(std::uint32_t ceiling, std::uint32_t max_batch, DFlashWidthCosts costs,
                                     bool batches, bool zero_drafts)
    : ceiling_(ceiling), max_batch_(max_batch), costs_(costs) {
    if (ceiling == 0 || max_batch == 0 || max_batch > kMaximumConcurrency || !(costs.proposal >= 0.0) ||
        !(costs.per_draft > 0.0) || !(costs.per_draft_wide > 0.0)) {
        throw std::invalid_argument("DFlash width policy dimensions are outside their domain");
    }
    row_widths_   = dflash_policy_widths(ceiling, false, zero_drafts);
    if (batches) { batch_widths_ = dflash_policy_widths(ceiling, true, zero_drafts); }
    pool_ = temporary_switches::flag("INFERNIX_TMP_DRAFT_PROCESS_STATS");
    pooled_.reset();
}

const std::vector<std::uint32_t>& DFlashWidthPolicy::widths(std::uint32_t batch_size) const {
    if (batch_size == 0 || batch_size > max_batch_) {
        throw std::invalid_argument("DFlash width policy batch is outside its domain");
    }
    return batch_size == 1U ? row_widths_ : batch_widths_;
}

double DFlashWidthPolicy::rate(std::span<const DFlashPolicyRow> rows, std::uint32_t width) const {
    double tokens = 0.0;
    for (const DFlashPolicyRow& row : rows) {
        const auto extent = static_cast<std::int32_t>(std::min(width - 1U, row.limit));
        tokens += runtime::speculative::expected_draft_tokens(row.state->calibration, row.probabilities, extent);
    }
    return tokens / costs_.round(width - 1U, static_cast<std::uint32_t>(rows.size()));
}

std::uint32_t DFlashWidthPolicy::choose(std::span<const DFlashPolicyRow> rows) {
    const auto batch_size = static_cast<std::uint32_t>(rows.size());
    const std::vector<std::uint32_t>& candidates = widths(batch_size);
    if (candidates.empty()) { throw std::logic_error("DFlash width policy does not choose rounds of this batch"); }
    std::uint32_t most = 0;
    for (const DFlashPolicyRow& row : rows) {
        if (row.state == nullptr) { throw std::invalid_argument("DFlash width policy row has no state"); }
        most = std::max(most, row.limit);
        // A request's first choice starts from what the process has learned (INFERNIX_TMP_DRAFT_PROCESS_STATS).
        if (pool_ && !row.state->pooled) {
            row.state->calibration = pooled_;
            row.state->pooled      = true;
        }
    }
    // Widths past every row's limit verify nothing more: the narrowest width holding the longest
    // row's drafts bounds the choice.
    std::size_t last = 0;
    while (last + 1 < candidates.size() && candidates[last] < most + 1U) { ++last; }
    std::size_t best      = 0;
    double best_rate      = rate(rows, candidates[0]);
    for (std::size_t i = 1; i <= last; ++i) {
        const double candidate = rate(rows, candidates[i]);
        if (candidate > best_rate) {
            best      = i;
            best_rate = candidate;
        }
    }
    std::uint32_t& rounds = batch_size == 1U ? rows[0].state->rounds : batch_rounds_[batch_size - 1U];
    rounds += 1U;
    if (rounds % kProbePeriod == 0U && best < last) { ++best; }
    return candidates[best];
}

void DFlashWidthPolicy::observe(DFlashLaneDraftState& state, std::span<const float> probabilities,
                                std::uint32_t verified, std::uint32_t accepted) noexcept {
    state.calibration.observe(probabilities, verified, accepted, kCalibrationWeight);
}

void DFlashWidthPolicy::learn(DFlashLaneDraftState& state, std::span<const float> probabilities,
                              std::uint32_t verified, std::uint32_t accepted) noexcept {
    observe(state, probabilities, verified, accepted);
    if (pool_) { pooled_.observe(probabilities, verified, accepted, kCalibrationWeight); }
}

} // namespace infernix::models::qwen3_5::detail
