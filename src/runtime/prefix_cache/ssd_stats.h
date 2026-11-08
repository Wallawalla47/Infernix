#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace infernix::runtime::prefix_cache {

// SSD tier image fates by snapshot kind (§5.7), indexed by SnapshotKind (tap, endpoint, output
// boundary, boundary): images written (with the slabs of their image and path), image reads into
// the Host tier, written images read at least once, written images superseded or dropped from the
// SSD before any read, and retained snapshots the Host tier evicted without an SSD copy.
struct SsdKindStats {
    std::uint64_t writes            = 0;
    std::uint64_t written_slabs     = 0;
    std::uint64_t reads             = 0;
    std::uint64_t first_reads       = 0;
    std::uint64_t superseded_unread = 0;
    std::uint64_t dropped_unread    = 0;
    std::uint64_t host_losses       = 0;
};
inline constexpr std::size_t kSnapshotKinds = 4;

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
    std::uint64_t promotion_room_waits = 0; // times a promotion waited for a write-back to free Host room
    std::uint64_t host_image_demotions = 0; // Host copies given up because the SSD holds them
    std::uint64_t host_block_demotions = 0;
    std::uint64_t snapshot_evictions   = 0; // SSD-only snapshots lost
    std::uint64_t backup_drops         = 0;
    std::uint64_t dead_reclaims        = 0;
    std::uint64_t write_failures       = 0;
    std::uint64_t read_failures        = 0;
    double read_bytes_per_second       = 0.0;
    std::uint64_t slab_bytes           = 0; // of every SSD slab (SsdKindStats::written_slabs)
    // The write budget (0: none), what it still allows, and the snapshots it passed over.
    std::uint64_t budget_bytes_per_hour  = 0;
    std::uint64_t budget_available_bytes = 0;
    std::uint64_t declined_snapshots     = 0;
    // Images written as codec streams, and the SSD bytes that saved.
    std::uint64_t packed_images      = 0;
    std::uint64_t packed_saved_bytes = 0;
    std::uint64_t policy_nanoseconds = 0; // the write policy's own time (choosing write-backs)
    std::array<SsdKindStats, kSnapshotKinds> by_kind{};
};

} // namespace infernix::runtime::prefix_cache
