// Flash-Next M0 NVMe probe (docs/maintainer/qwen3_8-flash-next-design.md §12, §14.2 Stage 1):
// random 4 KiB O_DIRECT reads from a large file, as the PLE n-gram row cache issues them.
//
// Build (from the repository root):
//   g++ -O2 -std=c++20 -pthread tools/flash_next_probe/nvme_probe.cpp -o build/flash_next_nvme_probe
//   ./build/flash_next_nvme_probe FILE [--seconds 2] [--depths 1,2,4,8,16,32,64]
//
// FILE should be on the NVMe that will hold the n-gram volume and much larger than RAM's page
// cache share (O_DIRECT bypasses the cache anyway; a 20+ GB file avoids drive-internal caching
// effects). Queue depth d is d threads each with one synchronous pread outstanding, which is the
// device-side concurrency io_uring at depth d reaches; io_uring submission overhead and
// SQPOLL/IOPOLL are calibrator (M9) questions. The design assumes QD1 p50 <= ~80 us so that layer 0
// hides a PLE row miss (§12.3, §12.4); this probe confirms or revises it.

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

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kBlock = 4096;

std::vector<int> parse_list(const char* s) {
    std::vector<int> out;
    for (const char* p = s; *p;) {
        out.push_back(std::atoi(p));
        while (*p && *p != ',') { ++p; }
        if (*p == ',') { ++p; }
    }
    return out;
}

double percentile(std::vector<double>& v, double q) {
    if (v.empty()) { return 0; }
    const std::size_t i = std::min(v.size() - 1, static_cast<std::size_t>(q * static_cast<double>(v.size())));
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(i), v.end());
    return v[i];
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s FILE [--seconds S] [--depths 1,2,4,...]\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    double seconds   = 2.0;
    std::vector<int> depths{1, 2, 4, 8, 16, 32, 64};
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--seconds") {
            seconds = std::atof(argv[i + 1]);
        } else if (a == "--depths") {
            depths = parse_list(argv[i + 1]);
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    const int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        std::perror("open O_DIRECT");
        return 1;
    }
    struct stat st{};
    fstat(fd, &st);
    const std::uint64_t blocks = static_cast<std::uint64_t>(st.st_size) / kBlock;
    if (blocks < 1024) {
        std::fprintf(stderr, "file too small for a random-read probe\n");
        return 1;
    }
    std::printf("%s: %.1f GB, random %zu-byte O_DIRECT reads, %.1f s per depth\n", path,
                static_cast<double>(st.st_size) / 1e9, kBlock, seconds);
    std::printf("%6s %10s %10s %9s %9s %9s %9s\n", "depth", "IOPS", "MB/s", "p50 us", "p90 us", "p99 us", "max us");
    for (int depth : depths) {
        std::atomic<bool> stop{false};
        std::vector<std::vector<double>> lat(static_cast<std::size_t>(depth));
        std::vector<std::thread> pool;
        std::atomic<int> errors{0};
        const auto t0 = Clock::now();
        for (int t = 0; t < depth; ++t) {
            pool.emplace_back([&, t] {
                void* buf = nullptr;
                if (posix_memalign(&buf, kBlock, kBlock) != 0) {
                    errors.fetch_add(1);
                    return;
                }
                std::mt19937_64 rng(0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(t + 1));
                auto& mine = lat[static_cast<std::size_t>(t)];
                mine.reserve(1 << 16);
                while (!stop.load(std::memory_order_relaxed)) {
                    const off_t off = static_cast<off_t>((rng() % blocks) * kBlock);
                    const auto a    = Clock::now();
                    if (pread(fd, buf, kBlock, off) != static_cast<ssize_t>(kBlock)) {
                        errors.fetch_add(1);
                        break;
                    }
                    mine.push_back(std::chrono::duration<double, std::micro>(Clock::now() - a).count());
                }
                std::free(buf);
            });
        }
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
        stop.store(true);
        for (auto& th : pool) { th.join(); }
        const double wall = std::chrono::duration<double>(Clock::now() - t0).count();
        std::vector<double> all;
        for (auto& v : lat) { all.insert(all.end(), v.begin(), v.end()); }
        const double iops = static_cast<double>(all.size()) / wall;
        const double mx   = all.empty() ? 0 : *std::max_element(all.begin(), all.end());
        const double p50 = percentile(all, 0.50), p90 = percentile(all, 0.90), p99 = percentile(all, 0.99);
        std::printf("%6d %10.0f %10.1f %9.1f %9.1f %9.1f %9.1f%s\n", depth, iops, iops * kBlock / 1e6, p50, p90, p99,
                    mx, errors.load() ? "  (read errors)" : "");
    }
    close(fd);
    return 0;
}
