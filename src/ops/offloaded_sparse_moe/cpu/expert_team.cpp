#include "ops/offloaded_sparse_moe/cpu/expert_team.h"

#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64)
#    include <immintrin.h>
#endif
#if defined(__linux__)
#    include <pthread.h>
#    include <sched.h>
#endif

namespace ninfer::ops::offloaded_moe {
namespace {

inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

void pin_current_thread(int cpu) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)cpu;
#endif
}

std::size_t xs_index(int job, int col) {
    return (static_cast<std::size_t>(job) * kMaxColumns + col) * kGateUpBlocks;
}

std::size_t h_index(int job, int col) {
    return (static_cast<std::size_t>(job) * kMaxColumns + col) * kHBlocks;
}

} // namespace

CpuExpertTeam::CpuExpertTeam(Options options)
    : workers_(options.workers), isa_(options.isa), max_jobs_(options.max_jobs),
      spin_iterations_(options.spin_iterations), prefetch_bytes_(options.prefetch_bytes) {
    if (workers_ < 1 || workers_ > kHBlocks) { throw std::invalid_argument("CpuExpertTeam: workers must be in [1, 40]"); }
    if (max_jobs_ < 1) { throw std::invalid_argument("CpuExpertTeam: max_jobs must be positive"); }
    if (!cpu_isa_supported(isa_)) { throw std::invalid_argument("CpuExpertTeam: unsupported CPU ISA"); }
    scratch_.resize(static_cast<std::size_t>(workers_));
    for (auto& s : scratch_) {
        s.x_gate.resize(xs_index(max_jobs_, 0));
        s.x_up.resize(xs_index(max_jobs_, 0));
    }
    h_.resize(h_index(max_jobs_, 0));
    threads_.reserve(static_cast<std::size_t>(workers_ - 1));
    for (int w = 1; w < workers_; ++w) {
        const int cpu = w < static_cast<int>(options.cpus.size()) ? options.cpus[static_cast<std::size_t>(w)] : -1;
        threads_.emplace_back([this, w, cpu] {
            if (cpu >= 0) { pin_current_thread(cpu); }
            worker_main(w);
        });
    }
}

CpuExpertTeam::~CpuExpertTeam() {
    stop_.store(true, std::memory_order_release);
    epoch_.fetch_add(1, std::memory_order_release);
    epoch_.notify_all();
    for (auto& t : threads_) { t.join(); }
}

void CpuExpertTeam::barrier() {
    if (workers_ == 1) { return; }
    const std::uint32_t generation = barrier_generation_.load(std::memory_order_acquire);
    if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == workers_) {
        arrived_.store(0, std::memory_order_relaxed);
        barrier_generation_.store(generation + 1, std::memory_order_release);
        return;
    }
    while (barrier_generation_.load(std::memory_order_acquire) == generation) { cpu_relax(); }
}

void CpuExpertTeam::work(int w) {
    const int n_jobs = static_cast<int>(jobs_.size());
    WorkerScratch& s = scratch_[static_cast<std::size_t>(w)];
    const int unit_begin = kHBlocks * w / workers_, unit_end = kHBlocks * (w + 1) / workers_;
    const int rg_begin = kDownRowGroups * w / workers_, rg_end = kDownRowGroups * (w + 1) / workers_;

    // Phase A: every job's owned gate/up units, SwiGLU, and A4 of the owned h blocks.
    for (int j = 0; j < n_jobs; ++j) {
        const CpuExpertJob& job = jobs_[static_cast<std::size_t>(j)];
        const bool shared = job.scales.input_gate == job.scales.input_up;
        const canon::A4Block* xg[kMaxColumns];
        const canon::A4Block* xu[kMaxColumns];
        canon::A4Block* hb[kMaxColumns];
        for (int c = 0; c < job.ncols; ++c) {
            quantize_a4(job.x[c], kHidden, job.scales.input_gate, &s.x_gate[xs_index(j, c)]);
            xg[c] = &s.x_gate[xs_index(j, c)];
            if (!shared) { quantize_a4(job.x[c], kHidden, job.scales.input_up, &s.x_up[xs_index(j, c)]); }
            xu[c] = shared ? xg[c] : &s.x_up[xs_index(j, c)];
            hb[c] = &h_[h_index(j, c)];
        }
        if (unit_begin < unit_end) {
            gate_up_units(isa_, job.record, job.scales, xg, xu, job.ncols, unit_begin, unit_end, hb,
                          prefetch_bytes_);
        }
    }
    barrier();
    // Phase B: every job's owned down rows, reading all of A4(h).
    for (int j = 0; j < n_jobs; ++j) {
        const CpuExpertJob& job = jobs_[static_cast<std::size_t>(j)];
        const canon::A4Block* hb[kMaxColumns];
        std::uint16_t* const* y = job.y;
        for (int c = 0; c < job.ncols; ++c) { hb[c] = &h_[h_index(j, c)]; }
        if (rg_begin < rg_end) { down_rows(isa_, job.record, job.scales, hb, job.ncols, rg_begin, rg_end, y, prefetch_bytes_); }
    }
}

void CpuExpertTeam::worker_main(int w) {
    std::uint64_t seen = 0;
    for (;;) {
        std::uint64_t now = epoch_.load(std::memory_order_acquire);
        for (int i = 0; now == seen && i < spin_iterations_; ++i) {
            cpu_relax();
            now = epoch_.load(std::memory_order_acquire);
        }
        while (now == seen) {
            sleepers_.fetch_add(1, std::memory_order_acq_rel);
            epoch_.wait(seen, std::memory_order_acquire);
            sleepers_.fetch_sub(1, std::memory_order_acq_rel);
            now = epoch_.load(std::memory_order_acquire);
        }
        seen = now;
        if (stop_.load(std::memory_order_acquire)) { return; }
        work(w);
        finished_.fetch_add(1, std::memory_order_acq_rel);
    }
}

void CpuExpertTeam::run(std::span<const CpuExpertJob> jobs) {
    if (jobs.empty()) { return; }
    if (static_cast<int>(jobs.size()) > max_jobs_) { throw std::invalid_argument("CpuExpertTeam: too many jobs"); }
    for (const auto& job : jobs) {
        if (job.ncols < 1 || job.ncols > kMaxColumns || !job.record) {
            throw std::invalid_argument("CpuExpertTeam: invalid job");
        }
    }
    jobs_ = jobs;
    finished_.store(0, std::memory_order_relaxed);
    epoch_.fetch_add(1, std::memory_order_release);
    if (sleepers_.load(std::memory_order_acquire) > 0) { epoch_.notify_all(); }
    work(0);
    while (finished_.load(std::memory_order_acquire) != workers_ - 1) { cpu_relax(); }
    jobs_ = {};
}

} // namespace ninfer::ops::offloaded_moe
