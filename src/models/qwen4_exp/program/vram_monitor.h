#pragma once

// The VRAM monitor of the Qwen4Exp Program (design §19.3.7): one thread that reads the device's
// video memory when the OS changes this process's budget, or once a second, keeps the latest
// snapshot, and hands each to the Program's callback, which decides (with VramControl) whether the
// expert cache must resize and wakes an idle engine. It mutates no CUDA state.

#include "core/vram_budget.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace ninfer::models::qwen4_exp {

class VramMonitor {
public:
    // `on_snapshot` runs on the monitor thread after every reading.
    VramMonitor(int device, VramBudgetSource& source, std::function<void(const VramSnapshot&)> on_snapshot,
                std::chrono::milliseconds period = std::chrono::milliseconds(1000));
    ~VramMonitor();
    VramMonitor(const VramMonitor&)            = delete;
    VramMonitor& operator=(const VramMonitor&) = delete;

    // The latest snapshot.
    [[nodiscard]] VramSnapshot latest() const;
    // Reads now on the calling thread (the device is current) and makes it the latest; for the
    // engine after it resized the pool, so its next decision sees the new free memory.
    VramSnapshot refresh();

private:
    void run();

    int device_;
    VramBudgetSource& source_;
    std::function<void(const VramSnapshot&)> on_snapshot_;
    std::chrono::milliseconds period_;
    mutable std::mutex mutex_;
    std::condition_variable stop_cv_;
    bool stopping_ = false;
    VramSnapshot latest_;
    std::thread thread_;
};

namespace testing {
// The grow delay construct_qwen4_exp passes to ProgramOptions::vram_grow_delay_seconds (default 30 s);
// engine-level tests shorten it.
void set_vram_grow_delay(double seconds);
[[nodiscard]] double vram_grow_delay();
} // namespace testing

} // namespace ninfer::models::qwen4_exp
