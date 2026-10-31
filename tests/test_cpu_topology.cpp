// The host's physical cores (core/cpu_topology.h), which the temporary CPU placement switches
// (runtime/expert_offload/placement.h) give one expert worker each: every logical CPU belongs
// to exactly one core, a core's first thread is in it, and a thread restricted to one core's CPU runs
// there and gets its old mask back. Also the process's logical CPUs and the default CPU expert team they
// allow (GitHub issue #48: 16 workers on 16 vCPUs starved the engine thread), and the rule that settles
// the team's measured rates.

#include "core/cpu_topology.h"
#include "runtime/expert_offload/placement.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
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

} // namespace

int main() {
    // The default team leaves four logical CPUs, keeps at least one worker and never grows.
    using infernix::runtime::expert_offload::default_workers;
    check(default_workers(16, 16) == 12, "16 GGUF workers on 16 logical CPUs become 12");
    check(default_workers(16, 32) == 16, "a 32-CPU host keeps 16 workers");
    check(default_workers(6, 8) == 4, "6 NVFP4 workers on 8 logical CPUs become 4");
    check(default_workers(6, 4) == 1 && default_workers(16, 2) == 1, "a tiny host keeps one worker");
    check(default_workers(16, 0) == 16, "an unknown CPU count keeps the configured team");
    check(default_workers(0, 16) == 0, "a disabled team stays off");

    // The 122B's decode split: 256 / (1 + 1.46 C t), C t = C x record / link.
    using infernix::runtime::expert_offload::decode_link_share_256;
    check(decode_link_share_256(27.5e9, 10'650.0, 5'308'416.0) == 64, "Ornith on this machine keeps 64/256 on the link");
    check(decode_link_share_256(55.0e9, 10'650.0, 5'308'416.0) > 64, "a faster link takes a larger share");
    check(decode_link_share_256(27.5e9, 21'300.0, 5'308'416.0) < 64, "a faster CPU team takes the larger part");
    check(decode_link_share_256(0.0, 10'650.0, 5'308'416.0) == 0 && decode_link_share_256(27.5e9, 0.0, 1.0) == 0,
          "no measured link or CPU rate gives no share");
    check(decode_link_share_256(1.0e15, 1.0, 1.0) == 256, "a link far faster than the CPU takes every miss");

    // The settled CPU rates: a second reading after a pause, more while one beats the best experts/s by over 10 %,
    // at most three; the best of each rate stands.
    using infernix::runtime::expert_offload::CpuRateReading;
    using infernix::runtime::expert_offload::settle_cpu_rates;
    using infernix::runtime::expert_offload::SettledCpuRates;
    const auto settle = [](std::vector<CpuRateReading> readings, int* pauses) {
        std::size_t next = 0;
        *pauses          = 0;
        return settle_cpu_rates(
            [&]() -> std::optional<CpuRateReading> {
                if (next == readings.size()) { return std::nullopt; }
                return readings[next++];
            },
            [&] { ++*pauses; });
    };
    int pauses = 0;
    SettledCpuRates rates = settle({{10'000.0, 80'000.0, -1.0, "a"}, {10'500.0, 78'000.0, -1.0, "b"}}, &pauses);
    check(rates.readings.size() == 2 && pauses == 1, "a quiet start takes two readings");
    check(rates.experts == 10'500.0 && rates.columns == 80'000.0, "each rate keeps its best reading");
    check(rates.note == "b" && rates.busy_share < 0.0, "the report takes the last reading's note");
    check(rates.readings_text() == "; the best of 2 readings (10000, 10500 experts/s)", "the report lists the readings");
    rates = settle({{12'000.0, 1.0, -1.0, ""}, {9'000.0, 1.0, -1.0, ""}}, &pauses);
    check(rates.readings.size() == 2 && rates.experts == 12'000.0, "a lower second reading leaves the first");
    // The 122B after a peer's large run (2026-10-30): 4,923 experts/s, then the settled 9,515-12,805.
    rates = settle({{4'923.0, 1.0, -1.0, ""}, {9'515.0, 1.0, -1.0, ""}, {12'805.0, 1.0, -1.0, ""}, {20'000.0, 1.0, -1.0, ""}},
                   &pauses);
    check(rates.readings.size() == 3 && pauses == 2 && rates.experts == 12'805.0,
          "a recovering memory reads at most three times");
    rates = settle({{5'000.0, 1.0, -1.0, ""}, {5'400.0, 1.0, -1.0, ""}, {20'000.0, 1.0, -1.0, ""}}, &pauses);
    check(rates.readings.size() == 2 && rates.experts == 5'400.0, "a gain of 10 % or less ends the readings");
    rates = settle({}, &pauses);
    check(rates.readings.empty() && pauses == 0 && rates.experts == 0.0 && rates.readings_text().empty(),
          "no expert record in RAM: no reading, no pause");
    rates = settle({{9'000.0, 1.0, 0.40, ""}, {10'000.0, 1.0, -1.0, ""}}, &pauses);
    check(rates.busy_share < 0.0, "a quiet reading clears the busy warning");
    rates = settle({{9'000.0, 1.0, -1.0, ""}, {10'000.0, 1.0, 0.40, ""}, {9'500.0, 1.0, 0.50, ""}}, &pauses);
    check(rates.busy_share < 0.0, "a busy reading after a quiet one does not warn");
    rates = settle({{5'000.0, 1.0, 0.40, ""}, {9'000.0, 1.0, 0.50, ""}, {9'500.0, 1.0, 0.45, ""}}, &pauses);
    check(rates.busy_share == 0.45, "every reading busy: the warning gives the last one's share");

    const std::uint32_t process = infernix::process_logical_cpus();
    const unsigned machine      = std::thread::hardware_concurrency();
    check(process >= 1, "the process may run on at least one logical CPU");
    check(machine == 0 || process <= machine, "the process's logical CPUs are at most the machine's");

    const auto cores = infernix::physical_cores();
    if (cores.empty()) {
        std::printf("no CPU topology on this platform or machine; no core to check\n");
        return g_failures == 0 ? 0 : 1;
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
