#pragma once

// The RAM tiers' resize under OS memory pressure (--ram-resize auto|off, docs/serving.md): the host prefix tier and
// the expert tier's pinned slots and unpinned records give memory back when Windows runs short and take it again
// once it is free. Windows stalls once about 1 GiB is left, so the tiers shrink while less than kPressureBytes is
// available until kRelieveBytes is, and grow back, up to their startup sizes, only after more than the growth floor
// (the startup reserve, --ram-free-mib, but at least kRelieveBytes, plus a step) has stayed available for
// kGrowHoldSeconds since the last release; growth then goes on in pieces, one per engine call, while it stays due.
//
// RamControl is the policy (pure, driven with readings and times); RamMonitor is one thread that reads the system's
// available physical memory once a second, keeps RamControl's state and wakes an idle engine (the Program's
// maintenance waker) when a resize is due. The engine thread asks decide() in the Program's maintain() with the
// tiers' room, applies it and reports the room left (resized), which asks for the next growth piece at once.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>

namespace infernix::runtime::expert_offload {

struct RamDecision {
    std::uint64_t release = 0; // bytes the tiers should give back now
    std::uint64_t grow    = 0; // bytes the tiers may take back now (a Program applies one piece of it per call)
};

class RamControl {
public:
    static constexpr std::uint64_t kPressureBytes = 2ULL << 30U;
    static constexpr std::uint64_t kRelieveBytes  = 3ULL << 30U;
    static constexpr std::uint64_t kStepBytes     = 1ULL << 30U;
    static constexpr double kGrowHoldSeconds      = 10.0;
    // Under pressure with no room known, the tiers are asked again this often (pins of the last round may have gone).
    static constexpr double kRecheckSeconds = 30.0;

    explicit RamControl(std::uint64_t reserve_bytes) noexcept : reserve_(reserve_bytes) {}

    // One reading of the available physical memory at `now` (seconds, any epoch); whether the engine should look:
    // a resize is due given the tiers' room (set_room), or, under pressure with no room, kRecheckSeconds have passed
    // since the room was last measured (checked).
    [[nodiscard]] bool observe(std::uint64_t available, double now) noexcept;
    // The room was measured at `now`.
    void checked(double now) noexcept { last_check_ = now; }
    // The resize due at `now` for `available`: a release under pressure (rounded up to whole steps), else a growth
    // of at most what stays above the growth floor (whole steps), once more than it has held for kGrowHoldSeconds
    // since the last hold.
    [[nodiscard]] RamDecision decide(std::uint64_t available, double now) const noexcept;
    // What growth leaves available: the reserve, but never less than kRelieveBytes (a growth into pressure would
    // only be given back), plus a step.
    [[nodiscard]] std::uint64_t growth_floor() const noexcept {
        return (reserve_ > kRelieveBytes ? reserve_ : kRelieveBytes) + kStepBytes;
    }
    // The tiers were resized at `now`; `hold`: growth waits kGrowHoldSeconds again (after a release, or a growth
    // that took nothing), else its next piece is due at once.
    void resized(double now, bool hold) noexcept {
        if (hold) { last_hold_ = now; }
    }
    // What the tiers can still release and take back.
    void set_room(std::uint64_t shrinkable, std::uint64_t growable) noexcept {
        shrinkable_ = shrinkable;
        growable_   = growable;
    }

private:
    std::uint64_t reserve_;
    std::uint64_t shrinkable_ = 0, growable_ = 0;
    double above_since_ = -1.0; // since when the growth floor has been exceeded (-1: not now)
    double last_hold_   = -1.0e30;
    double last_check_  = -1.0e30;
};

class RamMonitor {
public:
    // `available` reads the system's available physical memory (default: core/host_memory); `wake` is called on the
    // monitor thread when a resize is due (the engine's maintenance waker; set later with set_waker).
    explicit RamMonitor(std::uint64_t reserve_bytes, std::function<std::uint64_t()> available = {},
                        std::chrono::milliseconds period = std::chrono::milliseconds(1000));
    ~RamMonitor();
    RamMonitor(const RamMonitor&)            = delete;
    RamMonitor& operator=(const RamMonitor&) = delete;

    void set_waker(std::function<void()> wake);
    // The tiers' room at the start.
    void set_room(std::uint64_t shrinkable, std::uint64_t growable);
    // Engine thread: the resize due now for the tiers' room measured now (reads the memory afresh); once one is
    // applied, resized() with whether growth holds (RamControl::resized) and the room left. Whether more growth is
    // due at once: the engine is then woken again (after any request that waits).
    [[nodiscard]] RamDecision decide(std::uint64_t shrinkable, std::uint64_t growable);
    bool resized(bool hold, std::uint64_t shrinkable, std::uint64_t growable);
    // The latest reading.
    [[nodiscard]] std::uint64_t available() const noexcept { return available_.load(std::memory_order_relaxed); }

private:
    void run();
    [[nodiscard]] double now() const noexcept;

    std::function<std::uint64_t()> read_;
    std::chrono::milliseconds period_;
    std::chrono::steady_clock::time_point epoch_ = std::chrono::steady_clock::now();
    mutable std::mutex mutex_;
    std::condition_variable stop_cv_;
    bool stopping_ = false;
    RamControl control_;
    std::function<void()> wake_;
    std::atomic<std::uint64_t> available_{0};
    std::thread thread_;
};

} // namespace infernix::runtime::expert_offload
