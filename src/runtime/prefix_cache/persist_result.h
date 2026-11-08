#pragma once

#include <cstdint>
#include <string>

namespace infernix::runtime::prefix_cache {

// Outcome of saving or loading a prefix cache file (docs/maintainer/hybrid-prefix-cache-spec.md §5.5),
// for every model binding (persist_file.h).
struct PersistResult {
    bool ok = false;
    std::string message; // why nothing was saved or loaded
    std::uint64_t blocks    = 0;
    std::uint64_t snapshots = 0;
    std::uint64_t bytes     = 0;
    double seconds          = 0.0;
    // Load only: the file's entries and the Host tier bytes they all take, against this tier.
    std::uint64_t saved_blocks        = 0;
    std::uint64_t saved_snapshots     = 0;
    std::uint64_t required_host_bytes = 0;
    std::uint64_t host_bytes          = 0;
    // With a kept SSD tier: the entries whose copy is (saved) or came back (loaded) on the SSD tier,
    // and why the file's SSD copies were not used (empty: they were, or there were none).
    std::uint64_t ssd_blocks    = 0;
    std::uint64_t ssd_snapshots = 0;
    std::string ssd_message;
};

} // namespace infernix::runtime::prefix_cache
