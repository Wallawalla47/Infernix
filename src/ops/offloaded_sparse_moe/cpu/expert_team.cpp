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

void pin_current_thread(std::uint64_t cpus) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c = 0; c < 64; ++c) {
        if (((cpus >> c) & 1U) != 0) { CPU_SET(c, &set); }
    }
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#elif defined(_WIN32)
    (void)SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(cpus));
#else
    (void)cpus;
#endif
}

// A job column's x blocks and h blocks, `x_blocks` and `h_blocks` per column.
std::size_t xs_index(int job, int col, int x_blocks = kGateUpBlocks) {
    return (static_cast<std::size_t>(job) * kMaxColumns + col) * x_blocks;
}

std::size_t h_index(int job, int col, int h_blocks = kHBlocks) {
    return (static_cast<std::size_t>(job) * kMaxColumns + col) * h_blocks;
}

} // namespace

CpuExpertTeam::CpuExpertTeam(Options options)
    : workers_(options.workers), isa_(options.isa), max_jobs_(options.max_jobs),
      spin_iterations_(options.spin_iterations), prefetch_bytes_(options.prefetch_bytes),
      shape_(options.shape), activation_(options.activation), ggml_(options.ggml) {
    if (!registered_shape(shape_) ||
        (shape_ != kFlashNextShape && (activation_ != ExpertActivation::kA16 || ggml_))) {
        throw std::invalid_argument("CpuExpertTeam: this expert geometry is registered for NVFP4 A16 banks only");
    }
    if (workers_ < 1 || workers_ > shape_.h_blocks()) {
        throw std::invalid_argument("CpuExpertTeam: workers must be in [1, I / 16]");
    }
    if (max_jobs_ < 1) { throw std::invalid_argument("CpuExpertTeam: max_jobs must be positive"); }
    if (!cpu_isa_supported(isa_)) { throw std::invalid_argument("CpuExpertTeam: unsupported CPU ISA"); }
    if (ggml_ && activation_ != ExpertActivation::kA16) {
        throw std::invalid_argument("CpuExpertTeam: GGUF experts take the A16 activation");
    }
    narrow_a16_ = options.narrow_a16;
    shared_x_   = options.shared_x;
    x_owner_.resize(static_cast<std::size_t>(max_jobs_) * kMaxColumns);
    std::size_t table = 16;
    while (table < 2 * x_owner_.size()) { table *= 2; }
    owner_keys_.assign(table, nullptr);
    owner_vals_.assign(table, 0);
    owner_used_.assign(x_owner_.size(), 0);
    if (narrow_a16_ == NarrowA16::kA4Residual &&
        (activation_ != ExpertActivation::kA16 || ggml_ || shape_ != kFlashNextShape)) {
        throw std::invalid_argument("CpuExpertTeam: two-term A4 jobs need NVFP4 W4A16 experts of Flash-Next's geometry");
    }
    if (narrow_a16_ == NarrowA16::kA4Residual) {
        constexpr int kTerms = canon::kA4ResidualTerms;
        xr_.resize(xs_index(max_jobs_, 0) * kTerms);
        hr_.resize(h_index(max_jobs_, 0) * kTerms);
        x_scale_.resize(static_cast<std::size_t>(max_jobs_) * kMaxColumns);
        h_scale_.resize(static_cast<std::size_t>(max_jobs_) * kMaxColumns);
    }
    if (ggml_) {
        xg_.resize(static_cast<std::size_t>(max_jobs_) * kMaxColumns * kGgmlXSlices);
        hg_.resize(static_cast<std::size_t>(max_jobs_) * kMaxColumns * kGgmlHSlices);
    }
    if (activation_ == ExpertActivation::kA16) {
        x16_.resize(xs_index(max_jobs_, 0, shape_.gate_up_blocks()));
        h16_.resize(h_index(max_jobs_, 0, shape_.h_blocks()));
        hv_.resize(static_cast<std::size_t>(max_jobs_) * kMaxColumns * shape_.intermediate);
        x_exp_.resize(static_cast<std::size_t>(max_jobs_) * kMaxColumns);
        h_exp_.resize(static_cast<std::size_t>(max_jobs_) * kMaxColumns);
    } else {
        x_gate_.resize(xs_index(max_jobs_, 0));
        x_up_.resize(xs_index(max_jobs_, 0));
        h_.resize(h_index(max_jobs_, 0));
    }
    threads_.reserve(static_cast<std::size_t>(workers_ - 1));
    for (int w = 1; w < workers_; ++w) {
        const std::uint64_t cpus = w < static_cast<int>(options.cpus.size()) ? options.cpus[static_cast<std::size_t>(w)] : 0;
        threads_.emplace_back([this, w, cpus] {
            if (cpus != 0) { pin_current_thread(cpus); }
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
// A' (that exponent spans all I / 16 units of a column), then Phase B on the encoded h.
void CpuExpertTeam::work_a16(int w) {
    const int n_jobs = static_cast<int>(jobs_.size());
    constexpr int kRowsPerItem = 4;
    const auto column = [](int job, int col) { return static_cast<std::size_t>(job) * kMaxColumns + col; };
    const int hidden = shape_.hidden, intermediate = shape_.intermediate;
    const int x_blocks = shape_.gate_up_blocks(), h_blocks = shape_.h_blocks();
    const int down_items = shape_.down_row_groups() / kRowsPerItem;

    // Phase 0: x encoded in slices of a column's blocks. Every slice derives the column exponent
    // itself (an H-element scan), so no slice waits for another; slice 0 stores it.
    constexpr int kSlices = 8;
    const int slice_blocks = x_blocks / kSlices; // 20 or 24
    const int encode_items = n_jobs * kMaxColumns * kSlices;
    for (int i = next_quantize_.fetch_add(1, std::memory_order_relaxed); i < encode_items;
         i = next_quantize_.fetch_add(1, std::memory_order_relaxed)) {
        const int slice = i % kSlices, j = i / (kSlices * kMaxColumns), c = (i / kSlices) % kMaxColumns;
        const CpuExpertJob& job = jobs_[static_cast<std::size_t>(j)];
        if (c >= job.ncols || x_owner_[column(j, c)] != static_cast<int>(column(j, c))) { continue; }
        const int emax = canon::a16_column_exponent(job.x[c], hidden);
        if (slice == 0) { x_exp_[column(j, c)] = emax; }
        encode_a16_blocks(job.x[c], emax, slice * slice_blocks, (slice + 1) * slice_blocks,
                          &x16_[xs_index(j, c, x_blocks)]);
        beat(w);
    }
    barrier();
    // Phase A: gate/up units with SwiGLU, h in BF16.
    const int unit_items = n_jobs * h_blocks;
    for (int i = next_unit_.fetch_add(1, std::memory_order_relaxed); i < unit_items;
         i = next_unit_.fetch_add(1, std::memory_order_relaxed)) {
        const int j = i / h_blocks, u = i % h_blocks;
        const CpuExpertJob& job = jobs_[static_cast<std::size_t>(j)];
        const canon::A16Block* xb[kMaxColumns];
        std::uint16_t* hv[kMaxColumns];
        int exps[kMaxColumns];
        for (int c = 0; c < job.ncols; ++c) {
            const int owner = x_owner_[column(j, c)];
            xb[c]   = &x16_[xs_index(owner / kMaxColumns, owner % kMaxColumns, x_blocks)];
            exps[c] = x_exp_[static_cast<std::size_t>(owner)];
            hv[c]   = &hv_[column(j, c) * intermediate];
        }
        gate_up_units_a16(isa_, shape_, job.record, job.scales, xb, exps, job.ncols, u, u + 1, hv, prefetch_bytes_);
        beat(w);
    }
    barrier();
    // Phase A': each column's h encoded under its exponent.
    const int h_items = n_jobs * kMaxColumns;
    for (int i = next_encode_h_.fetch_add(1, std::memory_order_relaxed); i < h_items;
         i = next_encode_h_.fetch_add(1, std::memory_order_relaxed)) {
        const int j = i / kMaxColumns, c = i % kMaxColumns;
        if (c >= jobs_[static_cast<std::size_t>(j)].ncols) { continue; }
        h_exp_[column(j, c)] =
            encode_a16(&hv_[column(j, c) * intermediate], intermediate, &h16_[h_index(j, c, h_blocks)]);
        beat(w);
    }
    barrier();
    // Phase B: down row groups.
    const int row_items = n_jobs * down_items;
    for (int i = next_rows_.fetch_add(1, std::memory_order_relaxed); i < row_items;
         i = next_rows_.fetch_add(1, std::memory_order_relaxed)) {
        const int j = i / down_items, rg = (i % down_items) * kRowsPerItem;
        const CpuExpertJob& job = jobs_[static_cast<std::size_t>(j)];
        const canon::A16Block* hb[kMaxColumns];
        for (int c = 0; c < job.ncols; ++c) { hb[c] = &h16_[h_index(j, c, h_blocks)]; }
        down_rows_a16(isa_, shape_, job.record, job.scales, hb, &h_exp_[column(j, 0)], job.ncols, rg,
                      rg + kRowsPerItem, job.y, prefetch_bytes_);
        beat(w);
    }
}

// The two-term A4 round (NarrowA16::kA4Residual, a4r_expert.h): the A16 round's phases, with x and h
// encoded as terms under their row scales (each spans a whole column, so every x slice derives it
// itself and h is encoded in a phase of its own) and the terms' exact sums.
void CpuExpertTeam::work_a4r(int w) {
    constexpr int kTerms = canon::kA4ResidualTerms;
    const int n_jobs = static_cast<int>(jobs_.size());
    constexpr int kRowsPerItem = 4;
    const auto column = [](int job, int col) { return static_cast<std::size_t>(job) * kMaxColumns + col; };
    const auto x_terms = [this](int job, int col) { return &xr_[xs_index(job, col) * kTerms]; };
    const auto h_terms = [this](int job, int col) { return &hr_[h_index(job, col) * kTerms]; };

    // Phase 0: x terms in slices of a column's blocks; every slice scans the column for its row scale.
    constexpr int kSlices = 8, kSliceBlocks = kGateUpBlocks / kSlices;
    static_assert(kGateUpBlocks % kSlices == 0);
    const int encode_items = n_jobs * kMaxColumns * kSlices;
    for (int i = next_quantize_.fetch_add(1, std::memory_order_relaxed); i < encode_items;
         i = next_quantize_.fetch_add(1, std::memory_order_relaxed)) {
        const int slice = i % kSlices, j = i / (kSlices * kMaxColumns), c = (i / kSlices) % kMaxColumns;
        const CpuExpertJob& job = jobs_[static_cast<std::size_t>(j)];
        if (c >= job.ncols || x_owner_[column(j, c)] != static_cast<int>(column(j, c))) { continue; }
        const float row_scale = a4r_row_scale_of(job.x[c], kHidden);
        if (slice == 0) { x_scale_[column(j, c)] = row_scale; }
        encode_a4r_blocks(job.x[c], kGateUpBlocks, row_scale, slice * kSliceBlocks, (slice + 1) * kSliceBlocks,
                          x_terms(j, c));
        beat(w);
    }
    barrier();
    // Phase A: gate/up units with SwiGLU, h in BF16.
    const int unit_items = n_jobs * kHBlocks;
    for (int i = next_unit_.fetch_add(1, std::memory_order_relaxed); i < unit_items;
         i = next_unit_.fetch_add(1, std::memory_order_relaxed)) {
        const int j = i / kHBlocks, u = i % kHBlocks;
        const CpuExpertJob& job = jobs_[static_cast<std::size_t>(j)];
        const canon::A4Block* xb[kMaxColumns];
        std::uint16_t* hv[kMaxColumns];
        float scales[kMaxColumns];
        for (int c = 0; c < job.ncols; ++c) {
            const int owner = x_owner_[column(j, c)];
            xb[c]     = x_terms(owner / kMaxColumns, owner % kMaxColumns);
            scales[c] = x_scale_[static_cast<std::size_t>(owner)];
            hv[c]     = &hv_[column(j, c) * kIntermediate];
        }
        gate_up_units_a4r(isa_, job.record, job.scales, xb, scales, job.ncols, u, u + 1, hv, prefetch_bytes_);
        beat(w);
    }
    barrier();
    // Phase A': each column's h as terms under its row scale.
    const int h_items = n_jobs * kMaxColumns;
    for (int i = next_encode_h_.fetch_add(1, std::memory_order_relaxed); i < h_items;
         i = next_encode_h_.fetch_add(1, std::memory_order_relaxed)) {
        const int j = i / kMaxColumns, c = i % kMaxColumns;
        if (c >= jobs_[static_cast<std::size_t>(j)].ncols) { continue; }
        h_scale_[column(j, c)] = encode_a4r(&hv_[column(j, c) * kIntermediate], kIntermediate, h_terms(j, c));
        beat(w);
    }
    barrier();
    // Phase B: down row groups.
    const int row_items = n_jobs * (kDownRowGroups / kRowsPerItem);
    for (int i = next_rows_.fetch_add(1, std::memory_order_relaxed); i < row_items;
         i = next_rows_.fetch_add(1, std::memory_order_relaxed)) {
        const int j = i / (kDownRowGroups / kRowsPerItem), rg = (i % (kDownRowGroups / kRowsPerItem)) * kRowsPerItem;
        const CpuExpertJob& job = jobs_[static_cast<std::size_t>(j)];
        const canon::A4Block* hb[kMaxColumns];
        for (int c = 0; c < job.ncols; ++c) { hb[c] = h_terms(j, c); }
        down_rows_a4r(isa_, job.record, job.scales, hb, &h_scale_[column(j, 0)], job.ncols, rg, rg + kRowsPerItem,
                      job.y, prefetch_bytes_);
        beat(w);
    }
}

// A round of GGUF records (ggml_expert.h): x encoded per column, the gate/up 16-intermediate
// units with SwiGLU (h in BF16), h encoded per column (its exponent spans all 640), then the down
// rows in groups of 64. Every output is an exact sum rounded once, so the item split changes no bit.
void CpuExpertTeam::work_ggml(int w) {
    const int n_jobs = static_cast<int>(jobs_.size());
    constexpr int kUnitSize = 16, kUnits = kIntermediate / kUnitSize, kRowsPerItem = 64,
                  kRowItems = kHidden / kRowsPerItem;
    const auto column = [](int job, int col) { return static_cast<std::size_t>(job) * kMaxColumns + col; };
    const int column_items = n_jobs * kMaxColumns;
    for (int i = next_quantize_.fetch_add(1, std::memory_order_relaxed); i < column_items;
         i = next_quantize_.fetch_add(1, std::memory_order_relaxed)) {
        const int j = i / kMaxColumns, c = i % kMaxColumns;
        const CpuExpertJob& job = jobs_[static_cast<std::size_t>(j)];
        if (c >= job.ncols) { continue; }
        x_exp_[column(j, c)] = ggml_encode_column(job.x[c], kHidden, &xg_[column(j, c) * kGgmlXSlices]);
        beat(w);
    }
    barrier();
    const int unit_items = n_jobs * kUnits;
    for (int i = next_unit_.fetch_add(1, std::memory_order_relaxed); i < unit_items;
         i = next_unit_.fetch_add(1, std::memory_order_relaxed)) {
        const int j = i / kUnits, u = i % kUnits;
        const CpuExpertJob& job = jobs_[static_cast<std::size_t>(j)];
        const CpuLimbSlice* xs[kMaxColumns];
        std::uint16_t* hv[kMaxColumns];
        for (int c = 0; c < job.ncols; ++c) {
            xs[c] = &xg_[column(j, c) * kGgmlXSlices];
            hv[c] = &hv_[column(j, c) * kIntermediate];
        }
        ggml_gate_up(isa_, job.record, job.ggml, xs, &x_exp_[column(j, 0)], job.ncols, kUnitSize * u, kUnitSize * u + kUnitSize, hv);
        beat(w);
    }
    barrier();
    for (int i = next_encode_h_.fetch_add(1, std::memory_order_relaxed); i < column_items;
         i = next_encode_h_.fetch_add(1, std::memory_order_relaxed)) {
        const int j = i / kMaxColumns, c = i % kMaxColumns;
        if (c >= jobs_[static_cast<std::size_t>(j)].ncols) { continue; }
        h_exp_[column(j, c)] = ggml_encode_column(&hv_[column(j, c) * kIntermediate], kIntermediate,
                                                  &hg_[column(j, c) * kGgmlHSlices]);
        beat(w);
    }
    barrier();
    const int row_items = n_jobs * kRowItems;
    for (int i = next_rows_.fetch_add(1, std::memory_order_relaxed); i < row_items;
         i = next_rows_.fetch_add(1, std::memory_order_relaxed)) {
        const int j = i / kRowItems, r = (i % kRowItems) * kRowsPerItem;
        const CpuExpertJob& job = jobs_[static_cast<std::size_t>(j)];
        const CpuLimbSlice* hs[kMaxColumns];
        for (int c = 0; c < job.ncols; ++c) { hs[c] = &hg_[column(j, c) * kGgmlHSlices]; }
        ggml_down(isa_, job.record, job.ggml, hs, &h_exp_[column(j, 0)], job.ncols, r, r + kRowsPerItem, job.y);
        beat(w);
    }
}

void CpuExpertTeam::work(int w) {
    if (!jobs_.empty() && jobs_.front().ggml.valid()) {
        work_ggml(w);
        return;
    }
    if (activation_ == ExpertActivation::kA16) {
        if (narrow_a16_ == NarrowA16::kA4Residual) {
            work_a4r(w);
        } else {
            work_a16(w);
        }
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

void CpuExpertTeam::run(std::span<const CpuExpertJob> jobs, CpuExpertHeartbeat heartbeat) {
    if (jobs.empty()) { return; }
    if (static_cast<int>(jobs.size()) > max_jobs_) { throw std::invalid_argument("CpuExpertTeam: too many jobs"); }
    const bool ggml = jobs.front().ggml.valid();
    for (const auto& job : jobs) {
        if (job.ncols < 1 || job.ncols > kMaxColumns || !job.record || job.ggml.valid() != ggml ||
            (ggml && (!ggml_ || !ggml_cpu_supported(job.ggml)))) {
            throw std::invalid_argument("CpuExpertTeam: invalid job");
        }
    }
    jobs_      = jobs;
    heartbeat_ = heartbeat;
    // Each job column's owner: itself, or (INFERNIX_TMP_CPU_SHARED_X) the first column with its x pointer,
    // found in an open-addressed table of the round's pointers (no allocation on the miss path).
    const std::size_t mask = owner_keys_.size() - 1;
    for (std::size_t j = 0; j < jobs.size(); ++j) {
        for (int c = 0; c < jobs[j].ncols; ++c) {
            const int self = static_cast<int>(j) * kMaxColumns + c;
            int owner      = self;
            if (shared_x_) {
                const auto key = reinterpret_cast<std::uintptr_t>(jobs[j].x[c]);
                std::size_t slot = (key >> 4) * 0x9E3779B97F4A7C15ULL >> 20 & mask;
                while (owner_keys_[slot] != nullptr && owner_keys_[slot] != jobs[j].x[c]) { slot = (slot + 1) & mask; }
                if (owner_keys_[slot] == nullptr) {
                    owner_keys_[slot] = jobs[j].x[c];
                    owner_vals_[slot] = self;
                    owner_used_[owner_count_++] = static_cast<int>(slot);
                } else {
                    owner = owner_vals_[slot];
                }
            }
            x_owner_[static_cast<std::size_t>(self)] = owner;
        }
    }
    for (int i = 0; i < owner_count_; ++i) { owner_keys_[static_cast<std::size_t>(owner_used_[i])] = nullptr; }
    owner_count_ = 0;
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
