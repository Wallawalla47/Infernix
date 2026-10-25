// The host's physical cores (core/cpu_topology.h), which the temporary CPU placement switches
// (runtime/expert_offload/placement.h) give one expert worker each: every logical CPU belongs
// to exactly one core, a core's first thread is in it, and a thread restricted to one core's CPU runs
// there and gets its old mask back.

#include "core/cpu_topology.h"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <thread>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

} // namespace

int main() {
    const auto cores = infernix::physical_cores();
    if (cores.empty()) {
        std::printf("no CPU topology on this platform or machine; nothing to check\n");
        return 0;
    }
    std::uint64_t seen = 0;
    bool disjoint = true, nonempty = true;
    for (const auto& core : cores) {
        nonempty &= core.cpus != 0;
        disjoint &= (seen & core.cpus) == 0;
        seen |= core.cpus;
    }
    check(nonempty, "every physical core has a logical CPU");
    check(disjoint, "no logical CPU belongs to two physical cores");
    const unsigned logical = std::thread::hardware_concurrency();
    check(logical == 0 || static_cast<unsigned>(std::popcount(seen)) <= logical,
          "the cores hold at most the machine's logical CPUs");
    check(infernix::first_cpu(cores.front().cpus) >= 0 &&
              ((cores.front().cpus >> infernix::first_cpu(cores.front().cpus)) & 1U) != 0,
          "a core's first CPU is one of its own");
    check(infernix::first_cpu(0) == -1, "an empty mask has no first CPU");

    // Pin to the last core's first thread and back.
    const std::uint64_t one = std::uint64_t{1} << infernix::first_cpu(cores.back().cpus);
    const std::uint64_t before = infernix::set_current_thread_cpus(one);
    check(before != 0, "the thread's previous mask is reported");
    check(infernix::set_current_thread_cpus(before) == one, "the pinned mask is reported when restoring");
    check(infernix::set_current_thread_cpus(0) == 0, "an empty mask changes nothing");

    std::printf("%zu physical cores, %d logical CPUs\n", cores.size(), std::popcount(seen));
    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all CPU topology checks passed\n");
    return 0;
}
