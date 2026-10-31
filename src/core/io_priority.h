#pragma once

// Windows I/O priority. A process started by a low-priority parent (a Task Scheduler task at its
// default priority, a service, some IDE or agent terminals) inherits I/O priority Low even when its
// CPU class is Normal, and Windows throttles its reads: on an RTX 5090 D (Ryzen 7 9800X3D, the same
// arguments, each a fresh load) Strata measured a direct-read probe at 8,029 against 121,863 rows/s,
// cold prefill 3-4x slower and decode 134.8 against 160.8 tok/s (Strata #4733, 2026-10-30). Infernix
// reads its weights, the n-gram volume and the SSD expert tier with direct reads.

#include <optional>

namespace infernix {

// I/O priority hints (Windows IO_PRIORITY_HINT): very low 0, low 1, normal 2, high 3.
inline constexpr int kIoPriorityNormal = 2;

struct IoPriorityRaise {
    int before = kIoPriorityNormal; // the inherited priority
    int after  = kIoPriorityNormal; // the priority now (before: Windows refused the raise)
};

// Raises the calling process's I/O priority to normal when it is lower, once per process (every call
// returns the first call's outcome). nullopt when it was normal or higher, could not be read, or the
// platform has no such priority (Linux).
std::optional<IoPriorityRaise> raise_low_io_priority() noexcept;

} // namespace infernix
