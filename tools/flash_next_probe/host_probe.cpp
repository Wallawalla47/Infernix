// Flash-Next M0 host probe (docs/maintainer/qwen3_8-flash-next-design.md §14.2 Stage 1, §19 M0):
// DRAM read bandwidth versus worker threads, and the CPU W4A4 expert kernel's service rate per ISA,
// column count, worker count, and cold versus cache-warm records. No GPU is needed.
//
// Build (from the repository root):
//   g++ -O3 -std=c++20 -pthread -ffp-contract=off -fno-fast-math -Isrc
//     tools/flash_next_probe/host_probe.cpp src/ops/offloaded_sparse_moe/cpu/w4a4_expert.cpp
//     src/ops/offloaded_sparse_moe/cpu/expert_team.cpp -o build/flash_next_host_probe   (one command)
//   ./build/flash_next_host_probe [--arena-gib 8] [--experts 256] [--seconds 0.5] [--threads 1,2,4,8]
//                                 [--prefetch 0,1024,2048,4096,8192]
//
// Every tunable here (worker count, software prefetch distance, ISA variant) is chosen from this
// probe's output on the target machine; numbers from other hosts only show which knobs matter.
//
// Reported numbers:
//   Team: latency of one layer's CPU-served misses (k experts, one column each, cold records)
//   through CpuExpertTeam with N workers, the caller included; the decode critical path (§10.1).
//   DRAM: GB/s of distinct bytes read, all threads together, over an arena far larger than the LLC.
//   Expert: record bytes consumed per second (2,764,800 per expert call) and microseconds per expert
//   call. "cold" walks a record set larger than the LLC so every byte comes from DRAM; "warm" repeats
//   one record per thread. The design's narrow-route claim (§10.2) needs cold n = 8 to stay close to
//   the DRAM rate; on the development VM it did not (n = 1 ~5.6 GB/s cold, n = 8 ~3.1 GB/s warm), so
//   this table decides the CPU worker count and whether n > 4 experts stay on the CPU route.

#include "ops/offloaded_sparse_moe/cpu/expert_team.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <sys/mman.h>

namespace moe = ninfer::ops::offloaded_moe;

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    double arena_gib = 8.0;
    int experts      = 256; // 708 MB of records: far larger than any desktop LLC
    double seconds   = 0.5;
    int repetitions  = 5;
    std::vector<int> threads;
    std::vector<int> prefetch{0, 1024, 2048, 4096, 8192}; // software prefetch distances to sweep
};

std::vector<int> parse_list(const char* s) {
    std::vector<int> out;
    for (const char* p = s; *p;) {
        out.push_back(std::atoi(p));
        while (*p && *p != ',') { ++p; }
        if (*p == ',') { ++p; }
    }
    return out;
}

void* huge_alloc(std::size_t bytes) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        std::perror("mmap");
        std::exit(1);
    }
    madvise(p, bytes, MADV_HUGEPAGE);
    return p;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Runs `body(thread_index, stop)` on `threads` threads for about `seconds`; returns the wall time.
template <class Body> double run_threads(int threads, double seconds, Body&& body) {
    std::atomic<bool> stop{false};
    std::atomic<int> ready{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            ready.fetch_add(1);
            while (ready.load() < threads) {}
            body(t, stop);
        });
    }
    while (ready.load() < threads) {}
    const auto t0 = Clock::now();
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop.store(true);
    for (auto& th : pool) { th.join(); }
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ------------------------------------------------------------------------------------ DRAM

void dram_probe(const Options& o, const std::vector<int>& thread_counts) {
    const std::size_t bytes = static_cast<std::size_t>(o.arena_gib * (1ULL << 30)) & ~std::size_t{4095};
    auto* arena             = static_cast<std::uint64_t*>(huge_alloc(bytes));
    std::memset(arena, 1, bytes); // fault in
    const std::size_t words = bytes / 8;
    std::printf("\nDRAM read bandwidth over a %.1f GiB arena (median of %d)\n", o.arena_gib, o.repetitions);
    std::printf("%8s %10s %10s %10s\n", "threads", "GB/s", "min", "max");
    for (int threads : thread_counts) {
        std::vector<double> rates;
        for (int r = 0; r < o.repetitions; ++r) {
            std::vector<std::uint64_t> sink(threads * 8), done(threads * 8);
            const double wall = run_threads(threads, o.seconds, [&](int t, std::atomic<bool>& stop) {
                const std::size_t begin = words * t / threads, end = words * (t + 1) / threads;
                std::uint64_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0, n = 0;
                while (!stop.load(std::memory_order_relaxed)) {
                    for (std::size_t i = begin; i + 4 <= end; i += 4) {
                        acc0 += arena[i];
                        acc1 += arena[i + 1];
                        acc2 += arena[i + 2];
                        acc3 += arena[i + 3];
                    }
                    n += end - begin;
                }
                sink[t * 8] = acc0 + acc1 + acc2 + acc3;
                done[t * 8] = n;
            });
            std::uint64_t total = 0;
            for (int t = 0; t < threads; ++t) { total += done[t * 8]; }
            rates.push_back(static_cast<double>(total) * 8 / wall / 1e9);
        }
        std::printf("%8d %10.1f %10.1f %10.1f\n", threads, median(rates), *std::min_element(rates.begin(), rates.end()),
                    *std::max_element(rates.begin(), rates.end()));
    }
    munmap(arena, bytes);
}

// ------------------------------------------------------------------------------------ expert kernel

void fill_records(std::uint8_t* records, int experts) {
    std::mt19937_64 rng(7);
    for (int e = 0; e < experts; ++e) {
        std::uint8_t* rec = records + static_cast<std::size_t>(e) * moe::kRecordBytes;
        for (std::size_t unit = 0; unit < moe::kRecordBytes / moe::kUnitBytes; ++unit) {
            std::uint8_t* u = rec + unit * moe::kUnitBytes;
            for (int i = 0; i < 128; i += 8) {
                const std::uint64_t w = rng();
                std::memcpy(u + i, &w, 8);
            }
            for (int i = 0; i < 16; ++i) { u[128 + i] = static_cast<std::uint8_t>(0x30 + rng() % 0x18); } // finite E4M3
        }
    }
}

void expert_probe(const Options& o, const std::vector<int>& thread_counts) {
    const std::size_t bytes = static_cast<std::size_t>(o.experts) * moe::kRecordBytes;
    auto* records           = static_cast<std::uint8_t*>(huge_alloc(bytes));
    fill_records(records, o.experts);
    const moe::ExpertScales scales{0.0117F, 0.0117F, 0.0031F, 0.0117F * 0.002F, 0.0117F * 0.002F, 0.0031F * 0.002F};

    std::mt19937 rng(3);
    std::normal_distribution<float> normal(0.0F, 3.0F);
    const int max_threads = *std::max_element(thread_counts.begin(), thread_counts.end());
    // Per-thread activations and outputs, so threads share nothing they write.
    std::vector<std::vector<std::uint16_t>> x(static_cast<std::size_t>(max_threads) * moe::kMaxColumns),
        y(static_cast<std::size_t>(max_threads) * moe::kMaxColumns);
    for (auto& v : x) {
        v.resize(moe::kHidden);
        for (auto& e : v) {
            const float f = normal(rng);
            std::uint32_t b;
            std::memcpy(&b, &f, 4);
            e = static_cast<std::uint16_t>(b >> 16);
        }
    }
    for (auto& v : y) { v.resize(moe::kHidden); }

    std::printf("\nCPU W4A4 expert kernel, %d records (%.0f MB) (median of %d)\n", o.experts, bytes / 1e6, o.repetitions);
    std::printf("%-12s %5s %3s %8s %6s %12s %12s %12s\n", "isa", "mode", "n", "threads", "pf", "GB/s", "us/expert", "experts/s");
    for (moe::CpuIsa isa : {moe::CpuIsa::kAvx512Vnni, moe::CpuIsa::kAvxVnni, moe::CpuIsa::kAvx2, moe::CpuIsa::kScalar}) {
        if (!moe::cpu_isa_supported(isa)) { continue; }
        for (const bool cold : {true, false}) {
            for (int n : {1, 2, 4, 8}) {
                for (int threads : thread_counts) {
                  for (int pf : cold ? o.prefetch : std::vector<int>{moe::kDefaultPrefetchBytes}) {
                    if (isa == moe::CpuIsa::kScalar && threads > 1) { continue; }
                    std::vector<double> rates, latencies;
                    for (int r = 0; r < o.repetitions; ++r) {
                        std::vector<std::uint64_t> calls(static_cast<std::size_t>(threads) * 8);
                        std::atomic<std::uint64_t> next{0};
                        const double wall = run_threads(threads, o.seconds, [&](int t, std::atomic<bool>& stop) {
                            const std::uint16_t* xs[moe::kMaxColumns];
                            std::uint16_t* ys[moe::kMaxColumns];
                            for (int c = 0; c < n; ++c) {
                                xs[c] = x[static_cast<std::size_t>(t) * moe::kMaxColumns + c].data();
                                ys[c] = y[static_cast<std::size_t>(t) * moe::kMaxColumns + c].data();
                            }
                            std::uint64_t k = 0;
                            while (!stop.load(std::memory_order_relaxed)) {
                                const std::uint64_t e = cold ? next.fetch_add(1, std::memory_order_relaxed) % o.experts
                                                             : static_cast<std::uint64_t>(t % o.experts);
                                moe::expert_forward(isa, records + e * moe::kRecordBytes, scales, n, xs, ys, pf);
                                ++k;
                            }
                            calls[static_cast<std::size_t>(t) * 8] = k;
                        });
                        std::uint64_t total = 0;
                        for (int t = 0; t < threads; ++t) { total += calls[static_cast<std::size_t>(t) * 8]; }
                        rates.push_back(static_cast<double>(total) * moe::kRecordBytes / wall / 1e9);
                        latencies.push_back(wall * threads / static_cast<double>(std::max<std::uint64_t>(total, 1)) * 1e6);
                    }
                    const double rate = median(rates);
                    std::printf("%-12s %5s %3d %8d %6d %12.2f %12.1f %12.0f\n", moe::cpu_isa_name(isa), cold ? "cold" : "warm",
                                n, threads, pf, rate, median(latencies), rate * 1e9 / moe::kRecordBytes);
                  }
                }
            }
        }
    }
    munmap(records, bytes);
}


void team_probe(const Options& o, const std::vector<int>& thread_counts) {
    const std::size_t bytes = static_cast<std::size_t>(o.experts) * moe::kRecordBytes;
    auto* records           = static_cast<std::uint8_t*>(huge_alloc(bytes));
    fill_records(records, o.experts);
    const moe::ExpertScales scales{0.0117F, 0.0117F, 0.0031F, 0.0117F * 0.002F, 0.0117F * 0.002F, 0.0031F * 0.002F};
    std::vector<std::uint16_t> x(static_cast<std::size_t>(moe::kHidden) * 16, 0x3C00), y(x.size());
    std::printf("\nCPU expert team: latency of k cold misses at n = 1 (median of %d x 64 rounds)\n", o.repetitions);
    std::printf("%8s %6s %4s %12s %12s %12s\n", "workers", "pf", "k", "median us", "p90 us", "GB/s");
    for (int workers : thread_counts) {
      for (int pf : o.prefetch) {
        moe::CpuExpertTeam team({.workers = workers, .isa = moe::best_cpu_isa(), .max_jobs = 16,
                                 .spin_iterations = 1 << 20, .prefetch_bytes = pf, .cpus = {}});
        for (int k : {1, 2, 4, 10}) {
            std::vector<double> lat;
            std::size_t next = 0;
            for (int r = 0; r < o.repetitions * 64; ++r) {
                std::vector<moe::CpuExpertJob> jobs(static_cast<std::size_t>(k));
                for (int j = 0; j < k; ++j) {
                    auto& job  = jobs[static_cast<std::size_t>(j)];
                    job.record = records + (next++ % static_cast<std::size_t>(o.experts)) * moe::kRecordBytes;
                    job.scales = scales;
                    job.ncols  = 1;
                    job.x[0]   = &x[static_cast<std::size_t>(j) * moe::kHidden];
                    job.y[0]   = &y[static_cast<std::size_t>(j) * moe::kHidden];
                }
                const auto t0 = Clock::now();
                team.run(jobs);
                lat.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
            }
            std::sort(lat.begin(), lat.end());
            const double med = lat[lat.size() / 2], p90 = lat[lat.size() * 9 / 10];
            std::printf("%8d %6d %4d %12.1f %12.1f %12.2f\n", workers, pf, k, med, p90, k * moe::kRecordBytes / med / 1e3);
        }
      }
    }
    munmap(records, bytes);
}
} // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value          = [&] {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--arena-gib") {
            o.arena_gib = std::atof(value());
        } else if (a == "--experts") {
            o.experts = std::atoi(value());
        } else if (a == "--seconds") {
            o.seconds = std::atof(value());
        } else if (a == "--repetitions") {
            o.repetitions = std::atoi(value());
        } else if (a == "--threads") {
            o.threads = parse_list(value());
        } else if (a == "--prefetch") {
            o.prefetch = parse_list(value());
        } else if (a == "--skip-dram") {
            o.arena_gib = 0;
        } else {
            std::fprintf(stderr, "usage: %s [--arena-gib G] [--skip-dram] [--experts N] [--seconds S] "
                                 "[--repetitions R] [--threads 1,2,4,...] [--prefetch 0,1024,...]\n", argv[0]);
            return 2;
        }
    }
    const int hw = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    if (o.threads.empty()) {
        for (int t = 1; t < hw; t *= 2) { o.threads.push_back(t); }
        o.threads.push_back(hw);
    }
    std::printf("hardware threads %d, best expert ISA %s, record %zu bytes\n", hw,
                moe::cpu_isa_name(moe::best_cpu_isa()), moe::kRecordBytes);
    if (o.arena_gib > 0) { dram_probe(o, o.threads); }
    expert_probe(o, o.threads);
    team_probe(o, o.threads);
    return 0;
}
