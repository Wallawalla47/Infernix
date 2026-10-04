#pragma once

// The CPU expert engine's worker team (docs/maintainer/qwen3_8-flash-next-design.md §10.3).
//
// Worker w of N owns the same parts of every expert: gate/up 16-intermediate units
// [40w/N, 40(w+1)/N) and down row groups [160w/N, 160(w+1)/N). A round runs Phase A (A4 of x,
// the owned gate/up units, SwiGLU and A4 of the owned h blocks) over every job, one barrier, then
// Phase B (the owned down rows) over every job. Sums are exact int64, so the output bits do not
// depend on N, the ISA or the order in which workers finish; they equal expert_forward's.
//
// The calling thread is worker 0 and starts computing at once. Idle workers spin for a bounded
// time, then park on a futex (std::atomic::wait).

#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

namespace ninfer::ops::offloaded_moe {

struct CpuExpertJob {
    const std::uint8_t* record = nullptr;
    ExpertScales scales{};
    int ncols = 0;                          // 1..kMaxColumns
    const std::uint16_t* x[kMaxColumns]{};  // BF16 [kHidden] per column
    std::uint16_t* y[kMaxColumns]{};        // BF16 [kHidden] per column
};

class CpuExpertTeam {
public:
    struct Options {
        int workers         = 1;  // including the calling thread
        CpuIsa isa          = best_cpu_isa();
        int max_jobs        = 16; // per round
        int spin_iterations = 1 << 16; // pause-loop iterations before parking
        std::vector<int> cpus;         // optional: CPU of worker i (worker 0 is the caller, not pinned)
    };

    explicit CpuExpertTeam(Options options);
    ~CpuExpertTeam();
    CpuExpertTeam(const CpuExpertTeam&)            = delete;
    CpuExpertTeam& operator=(const CpuExpertTeam&) = delete;

    // Computes every job; returns when all outputs are written. Not reentrant.
    void run(std::span<const CpuExpertJob> jobs);

    [[nodiscard]] int workers() const { return workers_; }
    [[nodiscard]] CpuIsa isa() const { return isa_; }

private:
    struct alignas(64) WorkerScratch {
        std::vector<canon::A4Block> x_gate; // [job][col][kGateUpBlocks]
        std::vector<canon::A4Block> x_up;
    };

    void worker_main(int w);
    void work(int w);
    void barrier();

    int workers_;
    CpuIsa isa_;
    int max_jobs_;
    int spin_iterations_;
    std::vector<WorkerScratch> scratch_;
    std::vector<canon::A4Block> h_; // [job][col][kHBlocks], shared between the phases
    std::span<const CpuExpertJob> jobs_;

    alignas(64) std::atomic<std::uint64_t> epoch_{0};
    alignas(64) std::atomic<int> sleepers_{0};
    alignas(64) std::atomic<int> arrived_{0};
    alignas(64) std::atomic<std::uint32_t> barrier_generation_{0};
    alignas(64) std::atomic<int> finished_{0};
    alignas(64) std::atomic<bool> stop_{false};
    std::vector<std::thread> threads_;
};

} // namespace ninfer::ops::offloaded_moe
