#pragma once

// How busy the rest of the machine is: the CPU time other processes spend over an interval, as a
// share of every logical CPU's time. A startup measurement of this process's own throughput (the
// Flash-Next CPU expert team's rates) reads low while another program loads the cores, so it checks
// this share and measures again once the machine is quieter.

#include <chrono>
#include <cstdint>

namespace infernix {

// One sample: CPU time every logical CPU spent busy, and this process's own, since an arbitrary
// origin, with the wall clock and the logical CPU count.
struct HostCpuTimes {
    std::uint64_t busy_ns = 0; // all processes, summed over the logical CPUs
    std::uint64_t own_ns  = 0; // this process (user and kernel)
    std::chrono::steady_clock::time_point at{};
    std::uint32_t cpus = 1; // logical CPUs
};

// Samples the counters (Windows: GetSystemTimes and GetProcessTimes; Linux: /proc/stat and
// getrusage). Every field stays zero where the platform does not tell.
[[nodiscard]] HostCpuTimes host_cpu_times() noexcept;

// The share of the logical CPUs' capacity between two samples that other processes used: the
// machine's busy time minus this process's, over (wall time x logical CPUs), clamped to [0, 1];
// 0 when the samples carry no time.
[[nodiscard]] double other_process_cpu_share(const HostCpuTimes& begin, const HostCpuTimes& end) noexcept;

} // namespace infernix
