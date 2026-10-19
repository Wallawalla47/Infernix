// Other processes' share of the CPU (core/host_load.h): the arithmetic on fixed samples, and on live
// samples that this process's own busy time is not counted as another's.

#include "core/host_load.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

infernix::HostCpuTimes sample(std::uint64_t busy_ns, std::uint64_t own_ns, std::chrono::milliseconds at,
                              std::uint32_t cpus) {
    infernix::HostCpuTimes t;
    t.busy_ns = busy_ns;
    t.own_ns  = own_ns;
    t.at      = std::chrono::steady_clock::time_point{} + at;
    t.cpus    = cpus;
    return t;
}

void test_arithmetic() {
    using std::chrono::milliseconds;
    // 1 s on 32 CPUs: 10 CPU-seconds busy, 2 of them this process's: others 8 / 32.
    const auto a = sample(0, 0, milliseconds(0), 32);
    const auto b = sample(10'000'000'000ULL, 2'000'000'000ULL, milliseconds(1000), 32);
    check(std::abs(infernix::other_process_cpu_share(a, b) - 0.25) < 1e-12, "others' share is (busy - own) / capacity");
    // This process alone busy: no other process's share.
    const auto c = sample(6'000'000'000ULL, 6'000'000'000ULL, milliseconds(1000), 32);
    check(infernix::other_process_cpu_share(a, c) == 0.0, "the process's own time is not another's");
    // Clamped to [0, 1]; no time or a counter going back gives 0.
    const auto d = sample(64'000'000'000ULL, 0, milliseconds(1000), 32);
    check(infernix::other_process_cpu_share(a, d) == 1.0, "the share is clamped to 1");
    check(infernix::other_process_cpu_share(a, a) == 0.0, "no wall time gives 0");
    check(infernix::other_process_cpu_share(b, sample(1, 0, milliseconds(2000), 32)) == 0.0,
          "a busy counter going back gives 0");
}

void test_live_samples() {
    const infernix::HostCpuTimes first = infernix::host_cpu_times();
    check(first.cpus >= 1, "at least one logical CPU");
    // Keep two threads of this process busy until the own time has grown past 0.3 s (on an idle
    // machine ~150 ms; a loaded one only delays it, bounded at 30 s), and the others' share stays a
    // share.
    std::atomic<bool> stop{false};
    std::vector<std::thread> spinners;
    for (int i = 0; i < 2; ++i) {
        spinners.emplace_back([&] {
            volatile std::uint64_t x = 0;
            while (!stop.load(std::memory_order_relaxed)) { x = x * 6364136223846793005ULL + 1; }
        });
    }
    const auto started = std::chrono::steady_clock::now();
    infernix::HostCpuTimes second = first;
    do {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        second = infernix::host_cpu_times();
    } while (static_cast<double>(second.own_ns - first.own_ns) / 1e9 <= 0.35 &&
             std::chrono::steady_clock::now() - started < std::chrono::seconds(30));
    stop = true;
    for (auto& t : spinners) { t.join(); }
    const double own_s = static_cast<double>(second.own_ns - first.own_ns) / 1e9;
    const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const double share = infernix::other_process_cpu_share(first, second);
    std::printf("own CPU %.3f s over %.3f s with two spinning threads; other processes' share %.3f of %u CPUs\n",
                own_s, wall_s, share, second.cpus);
    check(own_s > 0.3, "the process's own CPU time counts its spinning threads");
    check(share >= 0.0 && share <= 1.0, "the live share is a share");
}

} // namespace

int main() {
    test_arithmetic();
    test_live_samples();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all host load checks passed\n");
    return 0;
}
