#pragma once

// The CPU expert engine's worker team (docs/maintainer/qwen3_8-flash-next-design.md §10.3).
//
// A round runs phases separated by barriers, each handing out work through an atomic counter so
// that faster cores (P-cores on hybrid CPUs) take more of it: A4 quantization (A16 encoding) of every
// job's columns, the gate/up 16-intermediate units with SwiGLU and A4 of h (A16: h in BF16, then its
// encoding in a phase of its own, since h's column exponent spans all units), then the down row
// groups. Sums are exact int64, so the output bits do not depend on N, the ISA, which worker took
// which item, or the order in which workers finish; they equal expert_forward's (A16:
// expert_forward_a16's; two-term A4, NarrowA16::kA4Residual: expert_forward_a4r's). The 8-bit
// activation rounds (CpuExpertJob::a8) fix each output's FP32 order by its row instead (GGUF jobs:
// ggml_expert.h's a8 functions; NVFP4 W4A16 jobs: expert_forward_a8's, w4a8_expert.h), with the
// same independence.
//
// The calling thread is worker 0 and starts computing at once. Idle workers spin for a bounded
// time, then park on a futex (std::atomic::wait).
//
// GGUF rounds (both activation routes) run the same items without barriers between the phases: a
// job's gate/up units wait for its own x encodes, the worker that completes a job's last unit encodes
// that job's h, and its down items wait for that job alone, so a slow core's last item delays one job,
// not the round. The items and their arithmetic are the phased round's, so the bits are too. Measured
// 2026-10-28 (candidates §27 CX-01; RTX 5090 + i9-13900K, 16 workers): IQ3_S DFlash2 decode after 8K
// +0.7..+0.85 %, team op 16x1 1035 -> 909 us (with the v2 rows); the NVFP4 rounds keep the phases,
// where it measured -0.16..-0.50 % (Dense8 MTP-3 decode).

#include "ops/offloaded_sparse_moe/cpu/a4r_expert.h"
#include "ops/offloaded_sparse_moe/cpu/ggml_expert.h"
#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"
#include "ops/offloaded_sparse_moe/cpu/w4a8_expert.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

namespace infernix::ops::offloaded_moe {

struct CpuExpertJob {
    const std::uint8_t* record = nullptr;
    ExpertScales scales{};
    GgmlExpertFormat ggml{};                // a GGUF bank's record (ggml_expert.h); scales unused
    // A job of a call on the 8-bit activation route (MissRequest::a8): a GGUF job takes ggml_expert.h's a8
    // functions, an NVFP4 W4A16 job (activation kA16) w4a8_expert.h's. Every job of a round has the same value.
    bool a8   = false;
    int ncols = 0;                          // 1..kMaxColumns
    const std::uint16_t* x[kMaxColumns]{};  // BF16 [H] per column (H: the team's shape.hidden)
    std::uint16_t* y[kMaxColumns]{};        // BF16 [H] per column
};

// A liveness word the device watches (CpuMissService's heartbeat) and its writer's counter. At
// namespace scope: a struct nested in CpuExpertTeam cannot carry default member initializers and
// also be the braced default argument of a CpuExpertTeam member (its initializers are parsed only
// once the enclosing class is complete; GCC bug 96645, and Clang rejects it the same way).
struct CpuExpertHeartbeat {
    volatile std::uint32_t* word = nullptr;
    std::uint32_t* count         = nullptr;
};

class CpuExpertTeam {
public:
    struct Options {
        int workers         = 1;  // including the calling thread
        CpuIsa isa          = best_cpu_isa();
        int max_jobs        = 16; // per round
        int spin_iterations = 1 << 16; // pause-loop iterations before parking
        int prefetch_bytes  = kDefaultPrefetchBytes; // calibrated on the target (§14.2)
        ExpertActivation activation = ExpertActivation::kA4; // of every job's expert
        // The arithmetic of kA16 jobs: exact, or the two-term A4 of expert-a4-residual-decode
        // (a4r_expert.h; NVFP4 banks at Flash-Next's geometry), which must equal the GPU calls' choice.
        NarrowA16 narrow_a16 = NarrowA16::kExact;
        bool ggml = false; // jobs may be GGUF records (ggml_expert.h; activation kA16)
        bool ggml_a8 = false; // GGUF jobs may take the 8-bit activation route (CpuExpertJob::a8; needs ggml)
        // NVFP4 W4A16 jobs may take the 8-bit activation route (CpuExpertJob::a8; w4a8_expert.h; needs
        // activation kA16).
        bool nvfp4_a8 = false;
        // Every job's expert geometry: Flash-Next's for every bank, or another registered one
        // (w4a4_expert.h) for NVFP4 A16 banks.
        ExpertShape shape = kFlashNextShape;
        // Optional: the logical CPUs (a mask, bit i: CPU i) worker i may run on; 0 or a missing entry
        // leaves it unpinned (worker 0 is the caller, not pinned here).
        std::vector<std::uint64_t> cpus;
    };

    explicit CpuExpertTeam(Options options);
    ~CpuExpertTeam();
    CpuExpertTeam(const CpuExpertTeam&)            = delete;
    CpuExpertTeam& operator=(const CpuExpertTeam&) = delete;

    // Computes every job; returns when all outputs are written. Not reentrant. The calling thread
    // advances `heartbeat` after each item it completes, so a long round stays alive to the device
    // while a hung worker (worker 0 then waits at a barrier or for a job's items) still stops it.
    void run(std::span<const CpuExpertJob> jobs, CpuExpertHeartbeat heartbeat = {});

    [[nodiscard]] int workers() const { return workers_; }
    [[nodiscard]] CpuIsa isa() const { return isa_; }
    [[nodiscard]] ExpertActivation activation() const { return activation_; }
    [[nodiscard]] NarrowA16 narrow_a16() const { return narrow_a16_; }
    [[nodiscard]] const ExpertShape& shape() const { return shape_; }

private:

    void worker_main(int w);
    void work(int w);
    void work_a16(int w);
    void work_a4r(int w);
    void work_a8(int w);
    void work_ggml(int w);
    void work_ggml_a8(int w);
    // The GGUF rounds' driver: x items (x_per_job per job), then unit items (units_per_job), each job's
    // h once its units are done (h_job, by the worker completing its last unit), then down items
    // (down_per_job), with per-job waits instead of barriers. Every item callable takes the item index.
    template <class XItem, class UnitItem, class HJob, class DownItem>
    void flow(int w, int x_per_job, int units_per_job, int down_per_job, XItem&& x_item, UnitItem&& unit_item,
              HJob&& h_job, DownItem&& down_item);
    void barrier();
    // Worker 0's beat after an item.
    void beat(int w) const {
        if (w == 0 && heartbeat_.word != nullptr) { *heartbeat_.word = ++*heartbeat_.count; }
    }

    int workers_;
    CpuIsa isa_;
    int max_jobs_;
    int spin_iterations_;
    int prefetch_bytes_;
    ExpertShape shape_;
    std::vector<canon::A4Block> x_gate_; // [job][col][kGateUpBlocks]
    std::vector<canon::A4Block> x_up_;
    std::vector<canon::A4Block> h_; // [job][col][kHBlocks], shared between the phases
    std::vector<canon::A16Block> x16_; // A16 jobs: [job][col][shape_.gate_up_blocks()]
    std::vector<canon::A16Block> h16_; // A16 jobs: [job][col][shape_.h_blocks()]
    std::vector<std::uint16_t> hv_;    // A16 jobs: BF16 h [job][col][shape_.intermediate]
    std::vector<int> x_exp_, h_exp_;   // A16 jobs: column exponents [job][col]
    // Two-term A4 jobs (narrow_a16 kA4Residual): x and h terms [job][col][term][blocks] and row
    // scales [job][col]; h in BF16 is hv_.
    std::vector<canon::A4Block> xr_, hr_;
    std::vector<float> x_scale_, h_scale_;
    // NVFP4 W4A16 jobs on the 8-bit route (options nvfp4_a8): x and h quantized [job][col][slices] (H / 32 and
    // I / 32 slices; h in BF16 is hv_), and per [job][h slice] how many of its two units are done (the second
    // quantizes it; reset by run()).
    std::vector<CpuW4A8Slice> x8_, h8_;
    std::vector<std::atomic<int>> h8_units_;
    NarrowA16 narrow_a16_ = NarrowA16::kExact;
    std::vector<CpuLimbSlice> xg_, hg_; // ggml jobs: [job][col][kGgmlXSlices], [job][col][kGgmlHSlices]
    std::vector<CpuA8Slice> xa_, ha_;   // ggml jobs on the 8-bit route: the same shapes
    ExpertActivation activation_;
    bool ggml_ = false;
    std::span<const CpuExpertJob> jobs_;
    CpuExpertHeartbeat heartbeat_;
    // GGUF rounds, per job: x items not yet done, units not yet done, and whether its h is ready (reset by run()).
    std::vector<std::atomic<int>> x_left_, units_left_, h_ready_;

    alignas(64) std::atomic<std::uint64_t> epoch_{0};
    alignas(64) std::atomic<int> sleepers_{0};
    alignas(64) std::atomic<int> arrived_{0};
    alignas(64) std::atomic<std::uint32_t> barrier_generation_{0};
    alignas(64) std::atomic<int> finished_{0};
    alignas(64) std::atomic<int> next_quantize_{0};
    alignas(64) std::atomic<int> next_unit_{0};
    alignas(64) std::atomic<int> next_encode_h_{0};
    alignas(64) std::atomic<int> next_rows_{0};
    alignas(64) std::atomic<bool> stop_{false};
    std::vector<std::thread> threads_;
};

} // namespace infernix::ops::offloaded_moe
