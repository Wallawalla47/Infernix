#pragma once

// The host CPU's physical cores and the logical CPUs of each (SMT siblings), with each core's
// efficiency class, so a placement can give threads one physical core each or keep a core apart.
// Logical CPUs are bits of a 64-bit mask (bit i: logical CPU i of the process's processor group);
// machines with more than 64 logical CPUs report no topology.

#include <cstdint>
#include <vector>

namespace infernix {

struct PhysicalCore {
    std::uint64_t cpus      = 0; // the core's logical CPUs (two with SMT); the lowest bit is its first thread
    std::uint8_t efficiency = 0; // the OS's efficiency class: higher is a faster core (hybrid CPUs: P-cores)
};

// The physical cores in the OS's order (Windows: GetLogicalProcessorInformationEx; Linux: sysfs
// core ids and, on hybrid Intel CPUs, /sys/devices/cpu_core/cpus). Empty when the platform does not
// tell or the machine has more than 64 logical CPUs.
[[nodiscard]] std::vector<PhysicalCore> physical_cores();

// The lowest logical CPU of a non-empty mask.
[[nodiscard]] int first_cpu(std::uint64_t cpus) noexcept;

// Restricts the calling thread to the logical CPUs of `cpus` (0: no change) and returns its previous
// mask (0 when unknown or when the change failed).
std::uint64_t set_current_thread_cpus(std::uint64_t cpus) noexcept;

} // namespace infernix
