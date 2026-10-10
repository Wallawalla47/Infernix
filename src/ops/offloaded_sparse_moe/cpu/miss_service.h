#pragma once

// The host side of CPU-served misses (docs/maintainer/qwen3_8-flash-next-design.md §10.3): a
// service thread watches the mapped request that moe_experts publishes for a layer call, computes
// the requested experts with the worker team (the service thread is worker 0) from the pinned
// expert banks, writes the BF16 outputs to mapped memory and answers with the request's sequence.
// The arithmetic is expert_forward's (A16: expert_forward_a16's), so a CPU-served output equals the
// GPU's bit for bit.
//
// The service owns the mapped channel buffers and the device sequence counter. It must outlive
// every call that uses its channels, and its destructor must run only when no call is pending.

#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/offloaded_sparse_moe/cpu/expert_team.h"
#include "ops/offloaded_sparse_moe/cpu/miss_request.h"
#include "ops/offloaded_sparse_moe/cpu/record_provider.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace infernix::ops::offloaded_moe {

class CpuMissService {
public:
    struct Layer {
        const std::uint8_t* records = nullptr; // pinned host bank, record_stride apart
        std::uint64_t record_stride = 0;
        const ExpertScales* scales  = nullptr; // host [E]
    };
    struct Options {
        int workers     = 8;     // including the service thread
        int max_jobs    = 4;     // CPU-served experts per layer call, at most kMaxCpuJobs
        int max_columns = 64;    // calls with more columns stay on the GPU, at most kMaxCpuCallColumns
        int pcie_divisor = 3;    // misses / pcie_divisor stay on the GPU stage (0: none)
        int max_job_columns = kMaxCpuColumns; // misses with more columns stay on the GPU
        int wide_from   = 0;     // calls of at least this many columns (0: none) take wide_jobs as their cap
        int wide_jobs   = 0;     // prefill assist cap, at most kMaxCpuJobs
        std::vector<int> cpus;   // optional CPU of worker i (worker 0 is the service thread)
        // The arithmetic of every layer's experts; it must equal the calls' MoeExpertSource::activation.
        ExpertActivation activation = ExpertActivation::kA4;
        // Tiered requests (design §19.3.7): a job without a record pointer is read through `records`
        // (demanded before the other jobs run, computed after them); without a provider, or when a
        // read fails, the request is answered with an errno in the status word. The service keeps its
        // heartbeat word changing while it waits for a record and as its team completes work items,
        // so the device waits as long as the request makes progress.
        RecordProvider* records = nullptr;
    };

    CpuMissService(std::vector<Layer> layers, Options options);
    ~CpuMissService();
    CpuMissService(const CpuMissService&)            = delete;
    CpuMissService& operator=(const CpuMissService&) = delete;

    [[nodiscard]] MoeCpuChannel channel(int layer) const;
    [[nodiscard]] std::uint64_t served_requests() const noexcept { return served_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t served_experts() const noexcept { return experts_.load(std::memory_order_relaxed); }
    // Why the service thread stopped serving; empty while it serves. The device's wait for an
    // answer then times out (kErrorHostSilent) on every later call, so the caller treats a failed
    // service as fatal and reports this cause.
    [[nodiscard]] std::string failure() const;
    // A round that may hand the service requests is about to be enqueued: an idle service thread
    // wakes now and spins again. Requests are served without it too, only later: after 20 ms
    // without a request the thread idles in waits of about 0.5 ms (Windows: a high-resolution
    // waitable timer; a plain sleep_for(50 us) there lasts one 15.6 ms timer tick).
    void wake() const noexcept;

private:
    void serve();
    void idle_wait() noexcept;
    void record_failure(std::string what) noexcept;

    std::vector<Layer> layers_;
    Options options_;
    MissRequest* request_ = nullptr; // mapped
    std::uint16_t* x_     = nullptr; // mapped BF16 [H, kMaxCpuXColumns]
    std::uint16_t* y_     = nullptr; // mapped BF16 [H, max(max_jobs, wide_jobs) * kMaxCpuColumns]
    int jobs_             = 0;       // max(max_jobs, wide_jobs)
    std::uint32_t* done_  = nullptr; // mapped
    std::uint32_t* status_ = nullptr; // mapped, beside done_
    std::uint32_t* heartbeat_ = nullptr; // mapped, beside status_: advanced while the service thread lives
    std::uint32_t* sequence_ = nullptr; // device
    std::atomic<bool> stop_{false};
    mutable std::atomic<std::uint32_t> wakes_{0};   // wake() calls the service thread has not seen yet
    void* wake_event_ = nullptr;             // Windows: auto-reset event wake() signals
    void* idle_timer_ = nullptr;             // Windows: the idle wait's high-resolution timer
    std::atomic<std::uint64_t> served_{0}, experts_{0};
    mutable std::mutex failure_mutex_;
    std::string failure_;
    std::thread thread_;
};

} // namespace infernix::ops::offloaded_moe
