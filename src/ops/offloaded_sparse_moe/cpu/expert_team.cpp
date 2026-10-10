#include "ops/offloaded_sparse_moe/cpu/expert_team.h"

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64)
#    include <immintrin.h>
#endif
#if defined(__linux__)
#    include <pthread.h>
#    include <sched.h>
#endif

namespace infernix::ops::offloaded_moe {
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
#elif defined(_WIN32)
    if (cpu >= 0 && cpu < 64) { (void)SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << cpu); }
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
      spin_iterations_(options.spin_iterations), prefetch_bytes_(options.prefetch_bytes),
      activation_(options.activation) {
    if (workers_ < 1 || workers_ > kHBlocks) { throw std::invalid_argument("CpuExpertTeam: workers must be in [1, 40]"); }
    if (max_jobs_ < 1) { throw std::invalid_argument("CpuExpertTeam: max_jobs must be positive"); }
    if (!cpu_isa_supported(isa_)) { throw std::invalid_argument("CpuExpertTeam: unsupported CPU ISA"); }
    if (activation_ == ExpertActivation::kA16) {
        x16_.resize(xs_index(max_jobs_, 0));
        h16_.resize(h_index(max_jobs_, 0));
        hv_.resize(static_cast<std::size_t>(max_jobs_) * kMaxColumns * kIntermediate);
        x_exp_.resize(static_cast<std::size_t>(max_jobs_) * kMaxColumns);
        h_exp_.resize(static_cast<std::size_t>(max_jobs_) * kMaxColumns);
    } else {
        x_gate_.resize(xs_index(max_jobs_, 0));
        x_up_.resize(xs_index(max_jobs_, 0));
        h_.resize(h_index(max_jobs_, 0));
    }
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

// The A16 round (design §16.2.1): the same item scheme as the A4 round, with x encoded under each
// column's exponent, h kept in BF16 by Phase A and encoded under its own column exponent in Phase
// A' (that exponent spans all 40 units of a column), then Phase B on the encoded h.
void CpuExpertTeam::work_a16(int w) {
    const int n_jobs = static_cast<int>(jobs_.size());
    constexpr int kRowsPerItem = 4;
    const auto column = [](int job, int col) { return static_cast<std::size_t>(job) * kMaxColumns + col; };

    // Phase 0: x encoded in slices of a column's blocks. Every slice derives the column exponent
    // itself (a 2,560-element scan), so no slice waits for another; slice 0 stores it.
    constexpr int kSlices = 8, kSliceBlocks = kGateUpBlocks / kSlices;
    const int encode_items = n_jobs * kMaxColumns * kSlices;
    for (int i = next_quantize_.fetch_add(1, std::memory_order_relaxed); i < encode_items;
         i = next_quantize_.fetch_add(1, std::memory_order_relaxed)) {
        const int slice = i % kSlices, j = i / (kSlices * kMaxColumns), c = (i / kSlices) % kMaxColumns;
        const CpuExpertJob& job = jobs_[static_cast<std::size_t>(j)];
        if (c >= job.ncols) { continue; }
        const int emax = canon::a16_column_exponent(job.x[c], kHidden);
        if (slice == 0) { x_exp_[column(j, c)] = emax; }
        encode_a16_blocks(job.x[c], emax, slice * kSliceBlocks, (slice + 1) * kSliceBlocks, &x16_[xs_index(j, c)]);
        beat(w);
    }
    barrier();
    // Phase A: gate/up units with SwiGLU, h in BF16.
    const int unit_items = n_jobs * kHBlocks;
    for (int i = next_unit_.fetch_add(1, std::memory_order_relaxed); i < unit_items;
         i = next_unit_.fetch_add(1, std::memory_order_relaxed)) {
        const int j = i / kHBlocks, u = i % kHBlocks;
        const CpuExpertJob& job = jobs_[static_cast<std::size_t>(j)];
        const canon::A16Block* xb[kMaxColumns];
        std::uint16_t* hv[kMaxColumns];
        for (int c = 0; c < job.ncols; ++c) {
            xb[c] = &x16_[xs_index(j, c)];
            hv[c] = &hv_[column(j, c) * kIntermediate];
        }
        gate_up_units_a16(isa_, job.record, job.scales, xb, &x_exp_[column(j, 0)], job.ncols, u, u + 1, hv,
                          prefetch_bytes_);
        beat(w);
    }
    barrier();
    // Phase A': each column's h encoded under its exponent.
    const int h_items = n_jobs * kMaxColumns;
    for (int i = next_encode_h_.fetch_add(1, std::memory_order_relaxed); i < h_items;
         i = next_encode_h_.fetch_add(1, std::memory_order_relaxed)) {
        const int j = i / kMaxColumns, c = i % kMaxColumns;
        if (c >= jobs_[static_cast<std::size_t>(j)].ncols) { continue; }
        h_exp_[column(j, c)] = encode_a16(&hv_[column(j, c) * kIntermediate], kIntermediate, &h16_[h_index(j, c)]);
        beat(w);
    }
    barrier();
    // Phase B: down row groups.
    const int row_items = n_jobs * (kDownRowGroups / kRowsPerItem);
    for (int i = next_rows_.fetch_add(1, std::memory_order_relaxed); i < row_items;
         i = next_rows_.fetch_add(1, std::memory_order_relaxed)) {
        const int j = i / (kDownRowGroups / kRowsPerItem), rg = (i % (kDownRowGroups / kRowsPerItem)) * kRowsPerItem;
        const CpuExpertJob& job = jobs_[static_cast<std::size_t>(j)];
        const canon::A16Block* hb[kMaxColumns];
        for (int c = 0; c < job.ncols; ++c) { hb[c] = &h16_[h_index(j, c)]; }
        down_rows_a16(isa_, job.record, job.scales, hb, &h_exp_[column(j, 0)], job.ncols, rg, rg + kRowsPerItem, job.y,
                      prefetch_bytes_);
        beat(w);
    }
}

void CpuExpertTeam::work(int w) {
    (void)w;
    if (activation_ == ExpertActivation::kA16) {
        work_a16(w);
        return;
    }
    const int n_jobs = static_cast<int>(jobs_.size());
    constexpr int kRowsPerItem = 4; // down row groups per item

    // Phase 0: A4 of every job's columns, once for the team (gate scale, and up scale if distinct),
    // in slices of a column's blocks: one decode job is otherwise a single serial item while the
    // other workers wait at the barrier. Blocks quantize independently, so slicing changes no bit.
    constexpr int kSlices = 8, kSliceBlocks = kGateUpBlocks / kSlices;
    static_assert(kGateUpBlocks % kSlices == 0);
    const int quantize_items = n_jobs * kMaxColumns * 2 * kSlices;
    for (int i = next_quantize_.fetch_add(1, std::memory_order_relaxed); i < quantize_items;
         i = next_quantize_.fetch_add(1, std::memory_order_relaxed)) {
        const int slice = i % kSlices, column_item = i / kSlices;
        const int j = column_item / (kMaxColumns * 2), c = (column_item / 2) % kMaxColumns, up = column_item % 2;
        const CpuExpertJob& job = jobs_[static_cast<std::size_t>(j)];
        if (c >= job.ncols || (up && job.scales.input_gate == job.scales.input_up)) { continue; }
        const std::uint16_t* x = job.x[c] + static_cast<std::size_t>(slice) * kSliceBlocks * 16;
        const std::size_t first = xs_index(j, c) + static_cast<std::size_t>(slice) * kSliceBlocks;
        if (up) {
            quantize_a4(x, kSliceBlocks * 16, job.scales.input_up, &x_up_[first]);
        } else {
            quantize_a4(x, kSliceBlocks * 16, job.scales.input_gate, &x_gate_[first]);
        }
        beat(w);
    }
    barrier();
    // Phase A: gate/up 16-intermediate units with SwiGLU and A4 of their h block.
    const int unit_items = n_jobs * kHBlocks;
    for (int i = next_unit_.fetch_add(1, std::memory_order_relaxed); i < unit_items;
         i = next_unit_.fetch_add(1, std::memory_order_relaxed)) {
        const int j = i / kHBlocks, u = i % kHBlocks;
        const CpuExpertJob& job = jobs_[static_cast<std::size_t>(j)];
        const bool shared = job.scales.input_gate == job.scales.input_up;
        const canon::A4Block* xg[kMaxColumns];
        const canon::A4Block* xu[kMaxColumns];
        canon::A4Block* hb[kMaxColumns];
        for (int c = 0; c < job.ncols; ++c) {
            xg[c] = &x_gate_[xs_index(j, c)];
            xu[c] = shared ? xg[c] : &x_up_[xs_index(j, c)];
            hb[c] = &h_[h_index(j, c)];
        }
        gate_up_units(isa_, job.record, job.scales, xg, xu, job.ncols, u, u + 1, hb, prefetch_bytes_);
        beat(w);
    }
    barrier();
    // Phase B: down row groups, reading all of A4(h).
    const int row_items = n_jobs * (kDownRowGroups / kRowsPerItem);
    for (int i = next_rows_.fetch_add(1, std::memory_order_relaxed); i < row_items;
         i = next_rows_.fetch_add(1, std::memory_order_relaxed)) {
        const int j = i / (kDownRowGroups / kRowsPerItem), rg = (i % (kDownRowGroups / kRowsPerItem)) * kRowsPerItem;
        const CpuExpertJob& job = jobs_[static_cast<std::size_t>(j)];
        const canon::A4Block* hb[kMaxColumns];
        for (int c = 0; c < job.ncols; ++c) { hb[c] = &h_[h_index(j, c)]; }
        down_rows(isa_, job.record, job.scales, hb, job.ncols, rg, rg + kRowsPerItem, job.y, prefetch_bytes_);
        beat(w);
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

void CpuExpertTeam::run(std::span<const CpuExpertJob> jobs, Heartbeat heartbeat) {
    if (jobs.empty()) { return; }
    if (static_cast<int>(jobs.size()) > max_jobs_) { throw std::invalid_argument("CpuExpertTeam: too many jobs"); }
    for (const auto& job : jobs) {
        if (job.ncols < 1 || job.ncols > kMaxColumns || !job.record) {
            throw std::invalid_argument("CpuExpertTeam: invalid job");
        }
    }
    jobs_      = jobs;
    heartbeat_ = heartbeat;
    finished_.store(0, std::memory_order_relaxed);
    next_quantize_.store(0, std::memory_order_relaxed);
    next_unit_.store(0, std::memory_order_relaxed);
    next_encode_h_.store(0, std::memory_order_relaxed);
    next_rows_.store(0, std::memory_order_relaxed);
    epoch_.fetch_add(1, std::memory_order_release);
    if (sleepers_.load(std::memory_order_acquire) > 0) { epoch_.notify_all(); }
    work(0);
    while (finished_.load(std::memory_order_acquire) != workers_ - 1) { cpu_relax(); }
    jobs_      = {};
    heartbeat_ = {};
}

} // namespace infernix::ops::offloaded_moe
