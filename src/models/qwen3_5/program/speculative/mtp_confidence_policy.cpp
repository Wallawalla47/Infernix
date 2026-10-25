#include "models/qwen3_5/program/speculative/mtp_confidence_policy.h"

#include <algorithm>
#include <stdexcept>

namespace infernix::models::qwen3_5::detail {
namespace {

// Bucket b holds probabilities in [kEdges[b], kEdges[b + 1]); its prior is the midpoint. The edges
// resolve the confident end, where the extent decisions are.
constexpr std::array<double, MtpConfidencePolicy::kBuckets + 1> kEdges{0.0,  0.3,  0.5,  0.7,   0.8,  0.9,
                                                                       0.95, 0.98, 0.99, 0.997, 1.0};

} // namespace

MtpConfidencePolicy::MtpConfidencePolicy(double prior_column_cost) : prior_column_cost_(prior_column_cost) {
    if (!(prior_column_cost >= 0.0)) {
        throw std::invalid_argument("MTP confidence policy: a column cost of at least 0");
    }
}

std::size_t MtpConfidencePolicy::bucket(float q) noexcept {
    std::size_t b = 0;
    while (b + 1 < kBuckets && static_cast<double>(q) >= kEdges[b + 1]) { ++b; }
    return b;
}

double MtpConfidencePolicy::bucket_acceptance(std::size_t b) const noexcept {
    const double prior = 0.5 * (kEdges[b] + kEdges[b + 1]);
    return (accepted_[b] + kPriorWeight * prior) / (tested_[b] + kPriorWeight);
}

double MtpConfidencePolicy::acceptance(float q) const noexcept { return bucket_acceptance(bucket(q)); }

double MtpConfidencePolicy::step_threshold(std::uint32_t batch_size) const noexcept {
    const double c = column_cost(batch_size);
    double best    = 0.0;
    for (std::size_t b = 0; b < kBuckets; ++b) { best = std::max(best, bucket_acceptance(b)); }
    for (std::size_t b = 0; b < kBuckets; ++b) {
        if (bucket_acceptance(b) * best > c) { return kEdges[b]; }
    }
    return 2.0;
}

bool MtpDraftRegime::draft(std::uint32_t batch_size) noexcept {
    if (batch_size == 0 || batch_size > kMaximumConcurrency) { return true; }
    Batch& batch            = batches_[batch_size - 1U];
    const Regime& free      = batch.regimes[0];
    const Regime& drafting  = batch.regimes[1];
    const std::uint32_t at  = batch.round++;
    bool draft              = true;
    if (free.cycles < kMinCycles || drafting.cycles < kMinCycles) {
        draft = (at / kProbeRounds) % 2U == 0U;
    } else {
        // Tokens per second, cross-multiplied (both regimes hold seconds).
        const bool drafting_faster = drafting.tokens * free.seconds >= free.tokens * drafting.seconds;
        const bool probe           = at % kProbePeriod >= kProbePeriod - kProbeRounds;
        draft                      = drafting_faster != probe;
    }
    ++rounds_;
    if (!draft) { ++draft_free_; }
    return draft;
}

void MtpDraftRegime::observe(std::uint32_t batch_size, bool previous, bool drafted, std::uint32_t tokens,
                             double seconds) noexcept {
    if (batch_size == 0 || batch_size > kMaximumConcurrency || previous != drafted || !(seconds > 0.0)) { return; }
    Regime& regime = batches_[batch_size - 1U].regimes[drafted ? 1 : 0];
    if (regime.cycles >= kMinCycles && seconds * regime.cycles > kGapRatio * regime.seconds) { return; }
    regime.tokens  = kDecay * regime.tokens + static_cast<double>(tokens);
    regime.seconds = kDecay * regime.seconds + seconds;
    regime.cycles  = kDecay * regime.cycles + 1.0;
}

void MtpConfidencePolicy::observe(std::span<const float> q, std::uint32_t extent, std::uint32_t accepted) noexcept {
    extent   = std::min<std::uint32_t>(extent, static_cast<std::uint32_t>(q.size()));
    accepted = std::min(accepted, extent);
    // Draft j is tested when drafts 0..j-1 were accepted, and accepted when j < accepted.
    for (std::uint32_t j = 0; j < extent && j <= accepted; ++j) {
        for (std::size_t b = 0; b < kBuckets; ++b) {
            tested_[b] *= kDecay;
            accepted_[b] *= kDecay;
        }
        const std::size_t b = bucket(q[j]);
        tested_[b] += 1.0;
        if (j < accepted) { accepted_[b] += 1.0; }
    }
}

bool MtpConfidencePolicy::solve(const CostFit& fit, double& a, double& b) noexcept {
    const double spread = fit.n * fit.xx - fit.x * fit.x;
    if (fit.n < kCostMinRounds || spread < kCostMinSpread * fit.n * fit.n) { return false; }
    b = (fit.n * fit.xt - fit.x * fit.t) / spread;
    a = (fit.t - b * fit.x) / fit.n;
    return a > 0.0 && b > 0.0;
}

void MtpConfidencePolicy::observe_round(std::uint32_t batch_size, std::uint32_t columns, double seconds) noexcept {
    if (batch_size == 0 || batch_size > kMaximumConcurrency || !(seconds > 0.0)) { return; }
    CostFit& fit    = cost_[batch_size - 1U];
    const double x  = static_cast<double>(columns);
    double a = 0.0, b = 0.0;
    // Before the fit, a cycle past kCostGapRatio mean cycles is a gap.
    const double predict = solve(fit, a, b) ? a + b * x : (fit.n > 0.0 ? fit.t / fit.n : seconds);
    if (seconds >= kCostGapRatio * predict) { return; }
    fit.n  = kCostDecay * fit.n + 1.0;
    fit.x  = kCostDecay * fit.x + x;
    fit.xx = kCostDecay * fit.xx + x * x;
    fit.t  = kCostDecay * fit.t + seconds;
    fit.xt = kCostDecay * fit.xt + x * seconds;
}

double MtpConfidencePolicy::column_cost(std::uint32_t batch_size) const noexcept {
    if (batch_size == 0 || batch_size > kMaximumConcurrency) { return prior_column_cost_; }
    double a = 0.0, b = 0.0;
    if (!solve(cost_[batch_size - 1U], a, b)) { return prior_column_cost_; }
    // One row's columns against the round of every row: rows alike, the batch's rate is a row's
    // tokens over 1 + (batch_size * b / a) * extent.
    return std::min(static_cast<double>(batch_size) * b / a, kCostMax);
}

std::uint32_t MtpConfidencePolicy::extent(std::span<const float> q, std::uint32_t ceiling,
                                          std::uint32_t batch_size) const noexcept {
    const auto n = std::min<std::uint32_t>(ceiling, static_cast<std::uint32_t>(q.size()));
    const double c   = column_cost(batch_size);
    std::uint32_t best = 0;
    double best_rate   = 1.0; // no drafts: one token per round
    double tokens = 1.0, reach = 1.0;
    for (std::uint32_t e = 1; e <= n; ++e) {
        reach *= acceptance(q[e - 1]);
        tokens += reach;
        const double rate = tokens / (1.0 + c * static_cast<double>(e));
        if (rate > best_rate) {
            best      = e;
            best_rate = rate;
        }
    }
    return best;
}

} // namespace infernix::models::qwen3_5::detail
