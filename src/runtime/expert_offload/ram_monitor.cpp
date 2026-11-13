#include "runtime/expert_offload/ram_monitor.h"

#include "core/host_memory.h"

#include <algorithm>

namespace infernix::runtime::expert_offload {

RamControl::RamControl(std::uint64_t reserve_bytes) noexcept
    : pressure_(std::max(reserve_bytes / 2U, kPressureFloorBytes)),
      relieve_(std::max(reserve_bytes, pressure_ + kStepBytes)) {}

bool RamControl::observe(std::uint64_t available, double now) noexcept {
    if (available > growth_floor()) {
        if (above_since_ < 0.0) { above_since_ = now; }
    } else {
        above_since_ = -1.0;
    }
    const RamDecision decision = decide(available, now);
    return decision.release != 0 || decision.grow != 0 ||
           (available < pressure_ && shrinkable_ == 0 && now - last_check_ >= kRecheckSeconds);
}

RamDecision RamControl::decide(std::uint64_t available, double now) const noexcept {
    RamDecision out;
    if (available < pressure_) {
        const std::uint64_t want = (relieve_ - available + kStepBytes - 1U) / kStepBytes * kStepBytes;
        out.release              = std::min(want, shrinkable_);
        return out;
    }
    const std::uint64_t floor = growth_floor();
    if (growable_ != 0 && available > floor && above_since_ >= 0.0 &&
        now - std::max(above_since_, last_hold_) >= kGrowHoldSeconds) {
        out.grow = std::min(growable_, (available - floor) / kStepBytes * kStepBytes);
    }
    return out;
}

RamMonitor::RamMonitor(std::uint64_t reserve_bytes, int device, std::function<std::uint64_t()> available,
                       std::chrono::milliseconds period)
    : read_(available ? std::move(available) : std::function<std::uint64_t()>([] {
          return host_memory_snapshot().available_physical;
      })),
      period_(period), control_(reserve_bytes), worker_(device) {
    available_.store(read_(), std::memory_order_relaxed);
    thread_ = std::thread([this] { run(); });
}

RamMonitor::~RamMonitor() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    stop_cv_.notify_all();
    if (thread_.joinable()) { thread_.join(); }
}

double RamMonitor::now() const noexcept {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_).count();
}

void RamMonitor::set_waker(std::function<void()> wake) {
    worker_.set_waker(wake);
    std::lock_guard lock(mutex_);
    wake_ = std::move(wake);
}

RamDecision RamMonitor::decide(std::uint64_t shrinkable, std::uint64_t growable) {
    const std::uint64_t available = read_();
    available_.store(available, std::memory_order_relaxed);
    std::lock_guard lock(mutex_);
    due_.store(false, std::memory_order_relaxed);
    control_.set_room(shrinkable, growable);
    control_.checked(now());
    (void)control_.observe(available, now());
    return control_.decide(available, now());
}

std::uint64_t RamMonitor::measure() {
    const std::uint64_t available = read_();
    available_.store(available, std::memory_order_relaxed);
    return available;
}

void RamMonitor::poke() {
    std::function<void()> wake;
    {
        std::lock_guard lock(mutex_);
        wake = wake_;
    }
    if (wake) { wake(); } // the engine's waker takes its own lock
}

void RamMonitor::set_room(std::uint64_t shrinkable, std::uint64_t growable) {
    std::lock_guard lock(mutex_);
    control_.set_room(shrinkable, growable);
}

bool RamMonitor::resized(bool hold, std::uint64_t shrinkable, std::uint64_t growable) {
    const std::uint64_t available = read_();
    available_.store(available, std::memory_order_relaxed);
    std::function<void()> wake;
    bool more = false;
    {
        std::lock_guard lock(mutex_);
        control_.resized(now(), hold);
        control_.set_room(shrinkable, growable);
        (void)control_.observe(available, now());
        more = !hold && control_.decide(available, now()).grow != 0;
        if (more) {
            due_.store(true, std::memory_order_relaxed);
            wake = wake_;
        }
    }
    if (wake) { wake(); } // the engine's waker takes its own lock
    return more;
}

void RamMonitor::run() {
    std::unique_lock lock(mutex_);
    while (!stopping_) {
        lock.unlock();
        const std::uint64_t available = read_();
        available_.store(available, std::memory_order_relaxed);
        lock.lock();
        if (stopping_) { break; }
        const bool due = control_.observe(available, now());
        if (due) { due_.store(true, std::memory_order_relaxed); }
        if ((due || busy_.load(std::memory_order_relaxed)) && wake_) {
            const std::function<void()> wake = wake_;
            lock.unlock();
            wake(); // the engine's waker takes its own lock
            lock.lock();
        }
        stop_cv_.wait_for(lock, period_, [this] { return stopping_; });
    }
}

} // namespace infernix::runtime::expert_offload
