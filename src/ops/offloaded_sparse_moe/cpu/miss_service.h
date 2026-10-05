#pragma once

// The host side of CPU-served misses (docs/maintainer/qwen3_8-flash-next-design.md §10.3): a
// service thread watches the mapped request that moe_experts publishes for a layer call, computes
// the requested experts with the worker team (the service thread is worker 0) from the pinned
// expert banks, writes the BF16 outputs to mapped memory and answers with the request's sequence.
// The arithmetic is expert_forward's, so a CPU-served output equals the GPU's bit for bit.
//
// The service owns the mapped channel buffers and the device sequence counter. It must outlive
// every call that uses its channels, and its destructor must run only when no call is pending.

#include "ninfer/ops/offloaded_sparse_moe.h"
#include "ops/offloaded_sparse_moe/cpu/expert_team.h"
#include "ops/offloaded_sparse_moe/cpu/miss_request.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace ninfer::ops::offloaded_moe {

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
        std::vector<int> cpus;   // optional CPU of worker i (worker 0 is the service thread)
    };

    CpuMissService(std::vector<Layer> layers, Options options);
    ~CpuMissService();
    CpuMissService(const CpuMissService&)            = delete;
    CpuMissService& operator=(const CpuMissService&) = delete;

    [[nodiscard]] MoeCpuChannel channel(int layer) const;
    [[nodiscard]] std::uint64_t served_requests() const noexcept { return served_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t served_experts() const noexcept { return experts_.load(std::memory_order_relaxed); }

private:
    void serve();

    std::vector<Layer> layers_;
    Options options_;
    MissRequest* request_ = nullptr; // mapped
    std::uint16_t* x_     = nullptr; // mapped BF16 [H, max_columns]
    std::uint16_t* y_     = nullptr; // mapped BF16 [H, max_jobs * kMaxCpuColumns]
    std::uint32_t* done_  = nullptr; // mapped
    std::uint32_t* sequence_ = nullptr; // device
    std::atomic<bool> stop_{false};
    std::atomic<std::uint64_t> served_{0}, experts_{0};
    std::thread thread_;
};

} // namespace ninfer::ops::offloaded_moe
