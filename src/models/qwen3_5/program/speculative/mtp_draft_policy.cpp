#include "models/qwen3_5/program/speculative/mtp_draft_policy.h"

#include <algorithm>
#include <stdexcept>

namespace infernix::models::qwen3_5::detail {

void MtpAcceptanceEstimate::observe(std::uint32_t extent, std::uint32_t accepted) noexcept {
    extent   = std::min(extent, kMtpPolicyMaximumDrafts);
    accepted = std::min(accepted, extent);
    for (std::size_t j = 0; j < tested_.size(); ++j) {
        tested_[j] *= kDecay;
        accepted_[j] *= kDecay;
    }
    // Draft j is tested when drafts 0..j-1 were accepted, and accepted when j < accepted.
    for (std::uint32_t j = 0; j < extent && j <= accepted; ++j) {
        tested_[j] += 1.0;
        if (j < accepted) { accepted_[j] += 1.0; }
    }
}

double MtpAcceptanceEstimate::pooled() const noexcept {
    double tested   = 0.0;
    double accepted = 0.0;
    for (std::size_t j = 0; j < tested_.size(); ++j) {
        tested += tested_[j];
        accepted += accepted_[j];
    }
    return (accepted + kPriorAcceptance * kPriorWeight) / (tested + kPriorWeight);
}

double MtpAcceptanceEstimate::acceptance(std::uint32_t position) const noexcept {
    const double pool = pooled();
    if (position >= kMtpPolicyMaximumDrafts) { return pool; }
    return (accepted_[position] + kShrinkWeight * pool) / (tested_[position] + kShrinkWeight);
}

double MtpAcceptanceEstimate::expected_tokens(std::uint32_t k) const noexcept {
    k                 = std::min(k, kMtpPolicyMaximumDrafts);
    const double pool = pooled();
    double expected   = 1.0;
    double reach      = 1.0;
    for (std::uint32_t j = 0; j < k; ++j) {
        reach *= (accepted_[j] + kShrinkWeight * pool) / (tested_[j] + kShrinkWeight);
        expected += reach;
    }
    return expected;
}

MtpDraftPolicy::MtpDraftPolicy(std::uint32_t ceiling, std::uint32_t max_batch)
    : ceiling_(ceiling), max_batch_(max_batch) {
    if (ceiling == 0 || ceiling > kMtpPolicyMaximumDrafts || max_batch == 0 ||
        max_batch > kMaximumConcurrency) {
        throw std::invalid_argument("MTP draft policy dimensions are outside their domain");
    }
    seconds_.assign(static_cast<std::size_t>(ceiling) * max_batch, 0.0);
}

bool MtpDraftPolicy::serves(std::uint32_t k, std::uint32_t batch_size) const noexcept {
    if (k == 0 || k > ceiling_ || batch_size == 0 || batch_size > max_batch_) { return false; }
    return batch_size == 1 || mtp_batch_draft_length(k, ceiling_);
}

std::vector<std::uint32_t> MtpDraftPolicy::lengths(std::uint32_t batch_size) const {
    std::vector<std::uint32_t> out;
    for (std::uint32_t k = 1; k <= ceiling_; ++k) {
        if (serves(k, batch_size)) { out.push_back(k); }
    }
    return out;
}

void MtpDraftPolicy::set_round_seconds(std::uint32_t k, std::uint32_t batch_size,
                                       double seconds) {
    if (!serves(k, batch_size) || !(seconds > 0.0)) {
        throw std::invalid_argument("MTP draft policy round time is outside its domain");
    }
    seconds_[static_cast<std::size_t>(batch_size - 1U) * ceiling_ + (k - 1U)] = seconds;
}

bool MtpDraftPolicy::measured(std::uint32_t batch_size) const noexcept {
    if (batch_size == 0 || batch_size > max_batch_) { return false; }
    for (std::uint32_t k = 1; k <= ceiling_; ++k) {
        if (serves(k, batch_size) &&
            !(seconds_[static_cast<std::size_t>(batch_size - 1U) * ceiling_ + (k - 1U)] > 0.0)) {
            return false;
        }
    }
    return true;
}

double MtpDraftPolicy::round_seconds(std::uint32_t k, std::uint32_t batch_size) const {
    if (!serves(k, batch_size)) {
        throw std::invalid_argument("MTP draft policy has no round of this length and batch size");
    }
    if (measured(batch_size)) {
        return seconds_[static_cast<std::size_t>(batch_size - 1U) * ceiling_ + (k - 1U)];
    }
    return 1.0 + kModelDraftCost * static_cast<double>(k);
}

double MtpDraftPolicy::rate(std::span<const MtpAcceptanceEstimate* const> rows,
                            std::uint32_t k) const {
    const auto batch_size = static_cast<std::uint32_t>(rows.size());
    double tokens         = 0.0;
    for (const MtpAcceptanceEstimate* row : rows) { tokens += row->expected_tokens(k); }
    return tokens / round_seconds(k, batch_size);
}

std::uint32_t MtpDraftPolicy::step(MtpDraftMemory& memory,
                                   std::span<const MtpAcceptanceEstimate* const> rows) const {
    if (memory.probe_rounds != 0) {
        memory.probe_rounds -= 1;
        return memory.probe_length;
    }
    const std::vector<std::uint32_t> candidates = lengths(static_cast<std::uint32_t>(rows.size()));
    // The current length stands unless a challenger beats it by the margin; the first choice has
    // no incumbent. Ties keep the shorter length.
    const bool incumbent =
        std::find(candidates.begin(), candidates.end(), memory.draft_length) != candidates.end();
    std::uint32_t best = incumbent ? memory.draft_length : candidates.front();
    double best_rate   = rate(rows, best);
    const double bar   = incumbent ? best_rate * kSwitchMargin : best_rate;
    for (const std::uint32_t k : candidates) {
        if (k == best) { continue; }
        const double candidate = rate(rows, k);
        if (candidate > bar && candidate > best_rate) {
            best      = k;
            best_rate = candidate;
        }
    }
    memory.draft_length = best;
    memory.rounds += 1;
    if (memory.rounds % kProbePeriod == 0) {
        const auto longer = std::upper_bound(candidates.begin(), candidates.end(), best);
        if (longer != candidates.end()) {
            memory.probe_length = *longer;
            memory.probe_rounds = kProbeRounds - 1U;
            return *longer;
        }
    }
    return best;
}

std::uint32_t MtpDraftPolicy::choose(MtpLaneDraftState& lane) const {
    const MtpAcceptanceEstimate* row = &lane.acceptance;
    return step(lane.memory, std::span<const MtpAcceptanceEstimate* const>(&row, 1));
}

std::uint32_t MtpDraftPolicy::choose_batch(std::span<const MtpAcceptanceEstimate* const> rows) {
    if (rows.size() < 2 || rows.size() > max_batch_) {
        throw std::invalid_argument("MTP draft policy batch is outside its domain");
    }
    return step(batches_[rows.size() - 1U], rows);
}

} // namespace infernix::models::qwen3_5::detail
