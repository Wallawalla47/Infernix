#pragma once

// One thread for the host-memory work that must not hold up the engine thread while requests run: page-locking and
// releasing memory, committing and decommitting pages, and copying between Host buffers (the RAM tiers' resize under
// memory pressure, --ram-resize). Jobs run one at a time in the order they were posted, so a job may use what an
// earlier one prepared (a recommit after a decommit of the same pages). After each job the waker runs (the engine's
// maintenance waker), once the job's future is ready, so the engine thread picks the result up at its next
// maintenance, idle or between rounds, without polling. The thread makes `device` current first, so memory it
// page-locks belongs to the Engine's device (-1: none is made current, for jobs that use no CUDA call), and runs on a
// hybrid CPU's slow cores (slow_core_cpus), leaving the fast ones to the threads serving requests. The destructor runs
// the jobs still queued (a queued release must not leak its memory), then stops.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

namespace infernix {

class HostMemoryWorker {
public:
    explicit HostMemoryWorker(int device = -1);
    ~HostMemoryWorker();
    HostMemoryWorker(const HostMemoryWorker&)            = delete;
    HostMemoryWorker& operator=(const HostMemoryWorker&) = delete;

    void set_waker(std::function<void()> wake);

    // Queues `work` (which may own move-only resources, such as the memory it releases); the future holds its result,
    // or its exception.
    template <class Function>
    [[nodiscard]] auto post(Function&& work) -> std::future<std::invoke_result_t<std::decay_t<Function>&>> {
        using Result = std::invoke_result_t<std::decay_t<Function>&>;
        // MSVC's packaged_task keeps its callable in a std::function, which must be copyable: a job owning move-only
        // memory (the chunks a release frees) is held through a shared pointer.
        auto held = std::make_shared<std::decay_t<Function>>(std::forward<Function>(work));
        auto task = std::make_shared<std::packaged_task<Result()>>([held]() -> Result { return (*held)(); });
        std::future<Result> future = task->get_future();
        enqueue([task = std::move(task)] { (*task)(); });
        return future;
    }

    // Whether every job posted so far has run.
    [[nodiscard]] bool idle() const;

private:
    void enqueue(std::function<void()> job);
    void run(int device) noexcept;

    mutable std::mutex mutex_;
    std::condition_variable work_;
    std::deque<std::function<void()>> queue_;
    std::function<void()> wake_;
    bool running_  = false; // a job is running
    bool stopping_ = false;
    std::thread thread_;
};

// A future's result is available now (never blocks).
template <class T>
[[nodiscard]] bool ready(const std::future<T>& future) {
    return future.valid() && future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

} // namespace infernix
