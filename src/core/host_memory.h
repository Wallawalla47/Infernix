#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace infernix {

// Physical Host memory the OS can hand out now without paging: free plus reclaimable file cache
// (Windows ullAvailPhys, Linux MemAvailable). Callers that lock large Host blocks check it first,
// since pinned pages cannot be paged out and an oversized lock pushes other processes to swap.
[[nodiscard]] std::uint64_t available_host_memory_bytes();

// One reading of the Host memory a startup ledger plans against.
struct HostMemorySnapshot {
    std::uint64_t total_physical = 0;
    // As available_host_memory_bytes, lowered on Linux to what the process's cgroup still allows.
    std::uint64_t available_physical = 0;
    // Memory the OS can still commit (Windows ullAvailPageFile: the commit limit less what is
    // committed; Linux: MemAvailable plus free swap, as Linux commits lazily).
    std::uint64_t available_commit = 0;
    // This process's private committed bytes (Windows PrivateUsage, Linux VmRSS).
    std::uint64_t process_private = 0;
    // Linux RLIMIT_MEMLOCK when limited; Windows has no such limit for CUDA pinned memory.
    std::optional<std::uint64_t> lock_limit;
};

[[nodiscard]] HostMemorySnapshot host_memory_snapshot();

// Keeps at least `bytes` of this process resident (Windows: a hard minimum working set), so a small
// RAM reserve trims other programs before Infernix's own pageable memory. Empty on success and where
// the OS has no such control (Linux); otherwise the OS's reason (non-fatal: the caller logs it).
[[nodiscard]] std::string reserve_process_working_set(std::uint64_t bytes);

} // namespace infernix
