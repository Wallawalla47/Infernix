#include "core/host_memory_worker.h"

#include "core/cpu_topology.h"

#include <cuda_runtime_api.h>

#include <exception>
#include <stdexcept>

namespace infernix {

HostMemoryWorker::HostMemoryWorker(int device) {
    thread_ = std::thread([this, device] { run(device); });
}

HostMemoryWorker::~HostMemoryWorker() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    work_.notify_all();
    if (thread_.joinable()) { thread_.join(); }
}

void HostMemoryWorker::set_waker(std::function<void()> wake) {
    std::lock_guard lock(mutex_);
    wake_ = std::move(wake);
}

bool HostMemoryWorker::idle() const {
    std::lock_guard lock(mutex_);
    return queue_.empty() && !running_;
}

void HostMemoryWorker::enqueue(std::function<void()> job) {
    {
        std::lock_guard lock(mutex_);
        if (stopping_) { throw std::logic_error("host memory worker is stopping"); }
        queue_.push_back(std::move(job));
    }
    work_.notify_one();
}

void HostMemoryWorker::run(int device) noexcept {
    // On a hybrid CPU's slow cores, as the expert tier's agent and copiers: page touching and copies leave the fast
    // cores to the threads serving requests (the CPU expert team, the engine thread).
    try {
        (void)set_current_thread_cpus(slow_core_cpus());
    } catch (...) {} // no topology: unpinned
    if (device >= 0) { (void)cudaSetDevice(device); }
    std::unique_lock lock(mutex_);
    for (;;) {
        work_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
        if (queue_.empty()) { return; } // stopping, nothing left
        std::function<void()> job = std::move(queue_.front());
        queue_.pop_front();
        running_ = true;
        lock.unlock();
        job(); // a packaged task: its exception lands in its future
        lock.lock();
        running_ = false;
        // Under mutex_, so set_waker returns only once no call is in flight: an engine that clears its waker
        // before it goes away is never called afterwards. The waker takes the engine's queue lock, which nothing
        // holds while it posts a job or sets the waker.
        if (wake_) { wake_(); }
    }
}

} // namespace infernix
