#pragma once

// The CPU expert team's speed at run time against its reference rates (the Program's startup reading): each served
// request's compute time over the time those rates predict for it, sum over its jobs of max(1 / expert_rate,
// columns / column_rate) (the prefill CPU split's cost model). The ratios are pooled with exponential forgetting by
// predicted work (kHorizonSeconds of it), so the estimate follows a change within about that much CPU work, and one
// long request weighs as much as many short ones of the same work. Other programs taking the CPU, or a slow state of
// the machine, raise it; the startup reading never sees either.

#include <atomic>
#include <cmath>
#include <cstdint>
#include <span>

namespace infernix::ops::offloaded_moe {

class CpuSpeed {
public:
    // Predicted work pooled into the estimate, and the least before it reads as anything but 1.
    static constexpr double kHorizonSeconds = 0.5;
    static constexpr double kMinimumSeconds = 0.05;

    // Rates in experts and expert columns per second for the whole team (non-positive: no estimate, 1). Any thread;
    // the pooled history restarts.
    void set_reference(double expert_rate, double column_rate) noexcept {
        expert_seconds_.store(expert_rate > 0.0 ? 1.0 / expert_rate : 0.0, std::memory_order_relaxed);
        column_seconds_.store(column_rate > 0.0 ? 1.0 / column_rate : 0.0, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
    }

    // The predicted seconds of a request whose jobs have these column counts.
    [[nodiscard]] double predicted(std::span<const int> columns) const noexcept {
        const double per_expert = expert_seconds_.load(std::memory_order_relaxed);
        const double per_column = column_seconds_.load(std::memory_order_relaxed);
        double seconds          = 0.0;
        for (const int n : columns) { seconds += std::fmax(per_expert, n * per_column); }
        return seconds;
    }

    // One served request: its jobs' column counts and the seconds the team took. One thread calls it.
    void add(std::span<const int> columns, double seconds) noexcept {
        if (const auto generation = generation_.load(std::memory_order_acquire); generation != seen_) {
            seen_     = generation;
            observed_ = expected_ = work_ = 0.0;
        }
        const double expected = predicted(columns);
        if (!(expected > 0.0) || !(seconds >= 0.0)) { return; }
        const double keep = std::exp(-expected / kHorizonSeconds);
        observed_         = observed_ * keep + seconds;
        expected_         = expected_ * keep + expected;
        work_             = std::fmin(work_ + expected, kHorizonSeconds);
        ratio_.store(work_ >= kMinimumSeconds ? observed_ / expected_ : 1.0, std::memory_order_relaxed);
    }

    // Observed over predicted time (1: as fast as the reference); any thread.
    [[nodiscard]] double ratio() const noexcept { return ratio_.load(std::memory_order_relaxed); }

private:
    std::atomic<double> expert_seconds_{0.0}, column_seconds_{0.0};
    std::atomic<std::uint32_t> generation_{0};
    std::uint32_t seen_ = 0;
    double observed_ = 0.0, expected_ = 0.0, work_ = 0.0;
    std::atomic<double> ratio_{1.0};
};

} // namespace infernix::ops::offloaded_moe
