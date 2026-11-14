#pragma once

// The RAM tiers' resize under OS memory pressure (--ram-resize auto|off, docs/serving.md): the host prefix tier and
// the expert tier's pinned slots and unpinned records give memory back when the system runs short and take it again
// once it is free, while requests run as well as between them. With the reserve R (--ram-free-mib) the tiers release
// once less than the pressure point max(R / 2, 2 GiB) is available (Windows stalls once about 1 GiB is left), in whole
// steps until the relieve point max(R, pressure point + a step) is available again, and grow back, up to their startup
// sizes (an automatic Host tier further, into the RAM other programs held at the start: HostSlabStore), only after
// more than the growth floor (the relieve point plus a step) has stayed available for
// kGrowHoldSeconds since the last release; growth then goes on in pieces while it stays due. The band between the
// pressure point and the growth floor (R = 8 GiB: 4 to 9 GiB) keeps a desktop's ordinary swings from cycling the
// tiers between releases and growths.
//
// RamControl is the policy (pure, driven with readings and times). RamMonitor is one thread that reads the system's
// available physical memory once a second, keeps RamControl's state and wakes the engine (the Program's maintenance
// waker, idle or between rounds) when a resize is due, and every second while a resize is in flight (it advances
// when transfers allow); it also owns the HostMemoryWorker the tiers' memory work runs on, off the engine thread. The
// engine thread asks decide() in the Program's maintenance with the tiers' room, starts it and reports the room left
// (resized) once it completed, which asks for the next growth piece at once.

#include "core/host_memory_worker.h"

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
    std::uint64_t grow    = 0; // bytes the tiers may take back now (a Program takes one piece of it at a time)
};

class RamControl {
public:
    // The lowest pressure point: Windows stalls once about 1 GiB is available.
    static constexpr std::uint64_t kPressureFloorBytes = 2ULL << 30U;
    static constexpr std::uint64_t kStepBytes          = 1ULL << 30U;
    static constexpr double kGrowHoldSeconds           = 10.0;
    // Under pressure with no room known, the tiers are asked again this often (pins of the last round may have gone).
    static constexpr double kRecheckSeconds = 30.0;

    explicit RamControl(std::uint64_t reserve_bytes) noexcept;

    // Below this much available the tiers release: half the reserve, at least kPressureFloorBytes.
    [[nodiscard]] std::uint64_t pressure_point() const noexcept { return pressure_; }
    // A release gives back, in whole steps, what makes this much available again: the reserve, at least a step above
    // the pressure point.
    [[nodiscard]] std::uint64_t relieve_point() const noexcept { return relieve_; }
    // What growth leaves available: the relieve point plus a step (a growth into it would soon be given back).
    [[nodiscard]] std::uint64_t growth_floor() const noexcept { return relieve_ + kStepBytes; }

    // One reading of the available physical memory at `now` (seconds, any epoch); whether the engine should look:
    // a resize is due given the tiers' room (set_room), or, under pressure with no room, kRecheckSeconds have passed
    // since the room was last measured (checked).
    [[nodiscard]] bool observe(std::uint64_t available, double now) noexcept;
    // The room was measured at `now`.
    void checked(double now) noexcept { last_check_ = now; }
    // The resize due at `now` for `available`: below the pressure point a release up to the relieve point (whole
    // steps), else a growth of at most what stays above the growth floor (whole steps), once more than it has held
    // for kGrowHoldSeconds since the last hold.
    [[nodiscard]] RamDecision decide(std::uint64_t available, double now) const noexcept;
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
    std::uint64_t pressure_, relieve_;
    std::uint64_t shrinkable_ = 0, growable_ = 0;
    double above_since_ = -1.0; // since when the growth floor has been exceeded (-1: not now)
    double last_hold_   = -1.0e30;
    double last_check_  = -1.0e30;
};

class RamMonitor {
public:
    // `available` reads the system's available physical memory (default: core/host_memory); `device` is the Engine's
    // CUDA device (the worker's pins belong to it). The waker (set_waker) is called on the monitor's and the worker's
    // threads.
    explicit RamMonitor(std::uint64_t reserve_bytes, int device, std::function<std::uint64_t()> available = {},
                        std::chrono::milliseconds period = std::chrono::milliseconds(1000));
    ~RamMonitor();
    RamMonitor(const RamMonitor&)            = delete;
    RamMonitor& operator=(const RamMonitor&) = delete;

    void set_waker(std::function<void()> wake);
    // The tiers' room at the start.
    void set_room(std::uint64_t shrinkable, std::uint64_t growable);
    // Whether the last reading found a resize due (cleared by decide): the cheap test the engine makes between rounds.
    [[nodiscard]] bool due() const noexcept { return due_.load(std::memory_order_relaxed); }
    // While a resize is in flight the engine is woken every period, so it advances once transfers allow it.
    void set_busy(bool busy) noexcept { busy_.store(busy, std::memory_order_relaxed); }
    // Engine thread, between rounds: asks for maintenance again at once (before the next round), for a step that only
    // waits for a transfer the round will finish.
    void poke();
    // Engine thread: the resize due now for the tiers' room measured now (reads the memory afresh); once one has
    // completed, resized() with whether growth holds (RamControl::resized) and the room left. Whether more growth is
    // due at once: the engine is then woken again.
    [[nodiscard]] RamDecision decide(std::uint64_t shrinkable, std::uint64_t growable);
    bool resized(bool hold, std::uint64_t shrinkable, std::uint64_t growable);
    // The latest reading, and a reading taken now.
    [[nodiscard]] std::uint64_t available() const noexcept { return available_.load(std::memory_order_relaxed); }
    std::uint64_t measure();
    [[nodiscard]] const RamControl& control() const noexcept { return control_; }
    // The thread the tiers' memory work runs on.
    [[nodiscard]] HostMemoryWorker& worker() noexcept { return worker_; }

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
    std::atomic<bool> due_{false};
    std::atomic<bool> busy_{false};
    HostMemoryWorker worker_;
    std::thread thread_;
};

} // namespace infernix::runtime::expert_offload
