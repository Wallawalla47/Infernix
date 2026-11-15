#pragma once

// Measurements that correct the prefix cache's machine model while the Engine runs
// (runtime/prefix_cache/cost.h, docs/maintainer/hybrid-prefix-cache-spec.md §6.2, §11.4). The model only ranks
// admission sources and values snapshots for eviction (spec §10 invariant 10): nothing here decides feasibility
// or changes what a request computes.
//
// - The Host-to-Device rate: the compiled coefficient is one machine's (about 51.5 GB/s, a PCIe 5.0 x16 link),
//   and a link at x8 moves about half of it, so restores were priced at half their time. A Host restore is itself
//   the measurement: its bytes over the time its copies took on the restore stream.
// - The prefill scale: only a few execution signatures have fitted coefficients, and every other artifact is
//   priced with the generic set. One factor, the measured prefill seconds over the model's prediction for the
//   same prompts, corrects the set's magnitude and keeps its shape. (A per-term fit is not identifiable from
//   ordinary traffic: its calls are one chunk wide, so the call and token terms are collinear.)

#include "runtime/prefix_cache/cost.h"

#include <algorithm>
#include <cstdint>

namespace infernix::runtime {

// Restores smaller than this are dominated by per-copy latency, not by the link.
inline constexpr std::uint64_t kH2dSampleMinimumBytes = 64ULL << 20;
// Rates outside these bounds are a misread interval, not a link.
inline constexpr double kH2dMinimumBytesPerSecond = 1.0e9;
inline constexpr double kH2dMaximumBytesPerSecond = 200.0e9;

// The measured Host-to-Device rate after a restore of `bytes` whose copies took `seconds`: the sample itself
// for the first one (`measured` 0), then a quarter of each new one, so one restore that shared the link with
// another transfer moves the estimate little. A sample too small, or outside the plausible rates, changes nothing.
[[nodiscard]] inline double blend_h2d_rate(double measured, std::uint64_t bytes, double seconds) noexcept {
    if (bytes < kH2dSampleMinimumBytes || !(seconds > 0.0)) { return measured; }
    const double rate = static_cast<double>(bytes) / seconds;
    if (!(rate >= kH2dMinimumBytesPerSecond && rate <= kH2dMaximumBytesPerSecond)) { return measured; }
    return measured > 0.0 ? 0.75 * measured + 0.25 * rate : rate;
}

// The prefill coefficients' scale: exponentially weighted sums of the measured and the predicted seconds of
// whole prompts' prefills (their ratio weights a long prefill by its length, which is where the ranking matters).
class OnlinePrefillScale {
public:
    // One prompt's prefill that ran alone on the device: `measured` seconds against the unscaled model's
    // `predicted` seconds for the same base and tokens.
    void observe(double measured, double predicted) noexcept {
        if (!(measured > 0.0) || !(predicted > 0.0)) { return; }
        measured_  = kDecay * measured_ + measured;
        predicted_ = kDecay * predicted_ + predicted;
        ++samples_;
    }

    // 1 until kMinimumSamples prefills were observed; bounded, so a mismeasured interval cannot invert a ranking
    // by orders of magnitude.
    [[nodiscard]] double scale() const noexcept {
        if (samples_ < kMinimumSamples || !(predicted_ > 0.0)) { return 1.0; }
        return std::clamp(measured_ / predicted_, kMinimumScale, kMaximumScale);
    }

    // `base` with its prefill terms scaled; the transfer terms are untouched.
    [[nodiscard]] prefix_cache::CacheCostModel apply(prefix_cache::CacheCostModel base) const noexcept {
        const double factor = scale();
        base.chunk_seconds *= factor;
        base.token_seconds *= factor;
        base.attention_pair_seconds *= factor;
        return base;
    }

    [[nodiscard]] std::uint32_t samples() const noexcept { return samples_; }

    static constexpr double kDecay                = 0.9;
    static constexpr std::uint32_t kMinimumSamples = 3;
    static constexpr double kMinimumScale         = 0.125;
    static constexpr double kMaximumScale         = 8.0;

private:
    double measured_       = 0.0;
    double predicted_      = 0.0;
    std::uint32_t samples_ = 0;
};

} // namespace infernix::runtime
