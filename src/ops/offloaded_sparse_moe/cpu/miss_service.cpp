#include "ops/offloaded_sparse_moe/cpu/miss_service.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#    include <immintrin.h>
#endif

#include <cerrno>

namespace infernix::ops::offloaded_moe {
namespace {

void cuda_require(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CPU miss service ") + what + ": " + cudaGetErrorString(error));
    }
}

void relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

void pin(int cpu) {
#if defined(_WIN32)
    if (cpu >= 0 && cpu < 64) { (void)SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << cpu); }
#else
    (void)cpu;
#endif
}

} // namespace

CpuMissService::CpuMissService(std::vector<Layer> layers, Options options)
    : layers_(std::move(layers)), options_(std::move(options)) {
    if (options_.workers < 1 || options_.max_jobs < 1 || options_.max_jobs > kMaxCpuJobs ||
        options_.max_columns < 1 || options_.max_columns > kMaxCpuCallColumns || options_.max_job_columns < 1 ||
        options_.max_job_columns > kMaxCpuColumns || options_.wide_from < 0 || options_.wide_jobs < 0 ||
        options_.wide_jobs > kMaxCpuJobs || (options_.wide_from > 0) != (options_.wide_jobs > 0)) {
        throw std::invalid_argument("CPU miss service options are out of range");
    }
    for (const auto& layer : layers_) {
        // Without a bank (the SSD tier) every request is tiered: it carries its records' addresses.
        if ((layer.records == nullptr && options_.records == nullptr) || layer.scales == nullptr ||
            layer.record_stride < kRecordBytes) {
            throw std::invalid_argument("CPU miss service layer is incomplete");
        }
    }
    const auto mapped = [](std::size_t bytes) {
        void* p = nullptr;
        cuda_require(cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc");
        std::memset(p, 0, bytes);
        return p;
    };
    request_ = static_cast<MissRequest*>(mapped(sizeof(MissRequest)));
    // A service that takes prefill calls sizes for every job a call may hand it: a gated call's cap
    // (MoeCpuChannel::wide_jobs, set per call) may rise to kMaxCpuJobs.
    jobs_    = options_.wide_jobs > 0 ? kMaxCpuJobs : options_.max_jobs;
    x_       = static_cast<std::uint16_t*>(mapped(sizeof(std::uint16_t) * kHidden * kMaxCpuXColumns));
    y_       = static_cast<std::uint16_t*>(mapped(sizeof(std::uint16_t) * kHidden * jobs_ * kMaxCpuColumns));
    done_    = static_cast<std::uint32_t*>(mapped(64));
    status_  = done_ + 1;
    heartbeat_ = done_ + 2;
    cuda_require(cudaMalloc(&sequence_, sizeof(std::uint32_t)), "cudaMalloc");
    cuda_require(cudaMemset(sequence_, 0, sizeof(std::uint32_t)), "cudaMemset");
    thread_ = std::thread([this] { serve(); });
}

CpuMissService::~CpuMissService() {
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) { thread_.join(); }
    cudaFree(sequence_);
    cudaFreeHost(done_);
    cudaFreeHost(y_);
    cudaFreeHost(x_);
    cudaFreeHost(request_);
}

MoeCpuChannel CpuMissService::channel(int layer) const {
    MoeCpuChannel out;
    out.request     = request_;
    out.x           = x_;
    out.y           = y_;
    out.done        = done_;
    out.status      = status_;
    out.heartbeat   = heartbeat_;
    out.sequence    = sequence_;
    out.layer       = layer;
    out.max_jobs    = options_.max_jobs;
    out.max_columns  = options_.max_columns;
    out.pcie_divisor = options_.pcie_divisor;
    out.max_job_columns = options_.max_job_columns;
    out.wide_from       = options_.wide_from;
    out.wide_jobs       = options_.wide_jobs;
    return out;
}

void CpuMissService::serve() {
    try {
        pin(options_.cpus.empty() ? -1 : options_.cpus[0]);
        CpuExpertTeam team({.workers = options_.workers, .max_jobs = jobs_, .cpus = options_.cpus});
        std::vector<CpuExpertJob> jobs(static_cast<std::size_t>(jobs_));
        std::vector<CpuExpertJob> ready, read;
        std::vector<std::uint32_t> tickets;
        std::vector<int> read_jobs;
        auto* volatile_status = reinterpret_cast<volatile std::uint32_t*>(status_);
        auto* volatile_beat   = reinterpret_cast<volatile std::uint32_t*>(heartbeat_);
        std::uint32_t beat    = 0;
        auto* volatile_sequence = reinterpret_cast<volatile std::uint32_t*>(&request_->sequence);
        auto* volatile_done     = reinterpret_cast<volatile std::uint32_t*>(done_);
        std::uint32_t seen      = *volatile_sequence;
        auto last               = std::chrono::steady_clock::now();
        std::uint32_t polls     = 0;
        while (!stop_.load(std::memory_order_acquire)) {
            const std::uint32_t sequence = *volatile_sequence;
            *volatile_beat = ++beat;
            if (sequence == seen) {
                relax();
                // Spin while decoding; after 20 ms without a request, poll every 50 us.
                if (++polls % 4096 == 0 &&
                    std::chrono::steady_clock::now() - last > std::chrono::milliseconds(20)) {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
                continue;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            seen = sequence;
            const MissRequest& r = *request_;
            const Layer& layer   = layers_.at(static_cast<std::size_t>(r.layer));
            const int count      = r.jobs;
            std::uint32_t status = 0;
            ready.clear();
            read_jobs.clear();
            tickets.clear();
            for (int j = 0; j < count; ++j) {
                CpuExpertJob& job = jobs[static_cast<std::size_t>(j)];
                const auto expert = static_cast<std::uint64_t>(r.expert[j]);
                job.record        = r.tiered != 0 ? reinterpret_cast<const std::uint8_t*>(r.record[j])
                                                      : layer.records + expert * layer.record_stride;
                job.scales        = layer.scales[expert];
                job.ncols         = r.ncols[j];
                for (int c = 0; c < job.ncols; ++c) {
                    job.x[c] = x_ + static_cast<std::size_t>(r.column[j][c]) * kHidden;
                    job.y[c] = y_ + (static_cast<std::size_t>(j) * kMaxCpuColumns + c) * kHidden;
                }
                if (job.record != nullptr) {
                    ready.push_back(job);
                } else if (options_.records != nullptr) { // its read starts now, under the other jobs
                    read_jobs.push_back(j);
                    tickets.push_back(options_.records->demand(r.layer, static_cast<int>(expert)));
                } else {
                    status = ENOENT;
                }
            }
            team.run(std::span<const CpuExpertJob>(ready.data(), ready.size()));
            if (!read_jobs.empty()) {
                read.clear();
                for (std::size_t i = 0; i < read_jobs.size(); ++i) {
                    std::uint32_t st = 0;
                    while (!options_.records->landed(tickets[i])) {
                        *volatile_beat = ++beat;
                        relax();
                    }
                    const std::uint8_t* record = options_.records->wait(tickets[i], st);
                    if (record == nullptr) {
                        status = st != 0 ? st : EIO;
                        continue;
                    }
                    CpuExpertJob job = jobs[static_cast<std::size_t>(read_jobs[i])];
                    job.record       = record;
                    read.push_back(job);
                }
                team.run(std::span<const CpuExpertJob>(read.data(), read.size()));
                for (const std::uint32_t t : tickets) { options_.records->done(t); }
            }
            *volatile_status = status;
            std::atomic_thread_fence(std::memory_order_release);
            *volatile_done = sequence;
            served_.fetch_add(1, std::memory_order_relaxed);
            experts_.fetch_add(static_cast<std::uint64_t>(count), std::memory_order_relaxed);
            last  = std::chrono::steady_clock::now();
            polls = 0;
        }
    } catch (const std::exception& error) {
        // A failed service never answers: the device wait ends after its timeout instead of hanging,
        // and failure() tells the caller why.
        record_failure(error.what());
    } catch (...) {
        record_failure("unknown exception");
    }
}

void CpuMissService::record_failure(std::string what) noexcept {
    try {
        const std::lock_guard<std::mutex> lock(failure_mutex_);
        failure_ = what.empty() ? std::string("unknown exception") : std::move(what);
    } catch (...) {}
}

std::string CpuMissService::failure() const {
    const std::lock_guard<std::mutex> lock(failure_mutex_);
    return failure_;
}

} // namespace infernix::ops::offloaded_moe
