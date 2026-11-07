#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace infernix::runtime::prefix_cache {

// The SSD tier's write budget (docs/maintainer/hybrid-prefix-cache-spec.md §5.7): at most
// `bytes_per_hour` written in any hour, as a bucket holding one hour's bytes that refills continuously.
// Its level is kept across restarts in `state_file` (so restarting the server does not refill it).
// While the bucket is at least half full every candidate is written; below that, a candidate is
// written only when its expected value is among the best (2 x fill) share, by bytes, of the recent
// candidates, so the bytes left go to what is most likely to save the most prefill. (Spending at the
// budget's sustained rate on the most valuable candidates from the start wrote less and kept fewer
// cache hits: the value ranking favours early taps, which are rarely read back; §16.9.)
class SsdWriteBudget {
public:
    using Clock = std::chrono::system_clock;

    SsdWriteBudget(std::uint64_t bytes_per_hour, std::filesystem::path state_file, Clock::time_point now);

    // Whether to write a candidate of `bytes` whose expected value is `value` (seconds of prefill saved
    // per byte, any positive scale); `first` records it among the recent candidates (once per offer).
    [[nodiscard]] bool admit(double value, std::uint64_t bytes, bool first, Clock::time_point now);
    // A write of `bytes` was submitted.
    void charge(std::uint64_t bytes) noexcept;
    // Writes the level and time to the state file (best effort: a failure keeps serving).
    void save(Clock::time_point now) const noexcept;

    [[nodiscard]] std::uint64_t bytes_per_hour() const noexcept { return bytes_per_hour_; }
    [[nodiscard]] std::uint64_t available_bytes() const noexcept;

private:
    struct Offer {
        double value        = 0.0;
        std::uint64_t bytes = 0;
    };
    static constexpr std::size_t kRecent = 256;

    void refill(Clock::time_point now) noexcept;
    [[nodiscard]] double threshold(double share) const;

    std::uint64_t bytes_per_hour_ = 0;
    double level_                 = 0.0; // bytes that may still be written
    Clock::time_point last_{};
    std::filesystem::path state_file_;
    std::array<Offer, kRecent> recent_{};
    std::size_t recent_count_ = 0;
    std::size_t recent_next_  = 0;
};

} // namespace infernix::runtime::prefix_cache
