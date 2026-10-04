#pragma once

#include <cstdint>

namespace ninfer {

// Physical Host memory the OS can hand out now without paging: free plus reclaimable file cache
// (Windows ullAvailPhys, Linux MemAvailable). Callers that lock large Host blocks check it first,
// since pinned pages cannot be paged out and an oversized lock pushes other processes to swap.
[[nodiscard]] std::uint64_t available_host_memory_bytes();

} // namespace ninfer
