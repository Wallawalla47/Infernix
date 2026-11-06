#pragma once

#include <cstdint>

namespace infernix::runtime::prefix_cache {

// What both models' prefix cache statistics report of the SSD tier
// (docs/maintainer/hybrid-prefix-cache-spec.md §5.7); all zero without one.
struct SsdTierStats {
    std::uint64_t capacity_bytes       = 0;
    std::uint64_t used_bytes           = 0;
    std::uint32_t ssd_only_snapshots   = 0;
    std::uint64_t write_bytes          = 0;
    std::uint64_t read_bytes           = 0;
    std::uint64_t image_writes         = 0;
    std::uint64_t block_writes         = 0;
    std::uint64_t promotions           = 0;
    std::uint64_t host_image_demotions = 0; // Host copies given up because the SSD holds them
    std::uint64_t host_block_demotions = 0;
    std::uint64_t snapshot_evictions   = 0; // SSD-only snapshots lost
    std::uint64_t backup_drops         = 0;
    std::uint64_t dead_reclaims        = 0;
    std::uint64_t write_failures       = 0;
    std::uint64_t read_failures        = 0;
    double read_bytes_per_second       = 0.0;
};

} // namespace infernix::runtime::prefix_cache
