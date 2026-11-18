#pragma once

// Private to prefix_index.cpp, prefix_index_host.cpp, prefix_index_ssd.cpp, prefix_index_persist.cpp and
// prefix_index_invariants.cpp (one PrefixCacheIndex, split by concern): the list ids and the invariant failure
// they share, and the few-line private helpers more than one of them calls, defined here so that a call across
// those files stays inlinable (the build has no cross-file inlining).

#include "runtime/prefix_cache/prefix_index.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>

namespace infernix::runtime::prefix_cache {

inline constexpr std::uint8_t kBacked   = 0;
inline constexpr std::uint8_t kUnbacked = 1;
inline constexpr std::uint8_t kNoList   = 0xff;

[[noreturn]] inline void invariant(const char* message) { throw std::logic_error(message); }

// ---- handles (the public valid() is defined in the class) ----

inline PrefixCacheIndex::Node& PrefixCacheIndex::require(NodeRef node) {
    if (!valid(node)) { invariant("stale prefix cache node reference"); }
    return nodes_[node.index];
}

inline const PrefixCacheIndex::Node& PrefixCacheIndex::require(NodeRef node) const {
    if (!valid(node)) { invariant("stale prefix cache node reference"); }
    return nodes_[node.index];
}

inline PrefixCacheIndex::Snapshot& PrefixCacheIndex::require(SnapshotRef snapshot) {
    if (!valid(snapshot)) { invariant("stale prefix cache snapshot reference"); }
    return snapshots_[snapshot.index];
}

inline const PrefixCacheIndex::Snapshot& PrefixCacheIndex::require(SnapshotRef snapshot) const {
    if (!valid(snapshot)) { invariant("stale prefix cache snapshot reference"); }
    return snapshots_[snapshot.index];
}

inline NodeRef PrefixCacheIndex::ref_of_node(std::uint32_t index) const noexcept {
    if (index == kNoId) { return {}; }
    return NodeRef{index, nodes_[index].generation};
}

inline SnapshotRef PrefixCacheIndex::ref_of_snapshot(std::uint32_t index) const noexcept {
    return SnapshotRef{index, snapshots_[index].generation};
}

inline std::uint64_t PrefixCacheIndex::child_key(std::uint32_t parent,
                                          std::uint64_t lookup_hash) noexcept {
    return lookup_hash ^ ((static_cast<std::uint64_t>(parent) + 1ULL) * 0x9e3779b97f4a7c15ULL);
}

// ---- copies and slab counts (the public ssd_image_slab_count() is defined in the class) ----

inline bool PrefixCacheIndex::snapshot_valid(const Snapshot& snapshot) const noexcept {
    const bool image = snapshot.device_slot != kNoId || image_backed(snapshot);
    const bool tail  = snapshot.tail_len == 0 || snapshot.tail_device_copy == CopyState::Resident ||
                       tail_backed(snapshot);
    return image && tail;
}

inline bool PrefixCacheIndex::tail_backed(const Snapshot& snapshot) const noexcept {
    return config_.host_blocks && image_backed(snapshot);
}

inline std::uint32_t PrefixCacheIndex::snapshot_slab_count(const Snapshot& snapshot) const noexcept {
    return config_.image_slabs + (config_.host_blocks && snapshot.tail_len != 0 ? 1U : 0U);
}

inline std::uint32_t PrefixCacheIndex::ssd_snapshot_slab_count(const Snapshot& snapshot) const noexcept {
    return ssd_image_slab_count() + (config_.host_blocks && snapshot.tail_len != 0 ? 1U : 0U);
}

inline std::size_t PrefixCacheIndex::ssd_to_host_position(std::size_t k) const noexcept {
    // The SSD copy holds the kept image slabs, then the tail slab; the Host layout, every image slab first.
    const std::uint32_t kept = ssd_image_slab_count();
    return k < kept ? k : config_.image_slabs + (k - kept);
}

// ---- Host and SSD slabs ----

inline void PrefixCacheIndex::free_slab(std::uint32_t slab) {
    if (slab >= ceiling_) { // retired by the shrink under way
        --past_in_use_;
        if (slab_moving_[slab]) { --moves_live_; } // its planned move has nothing left to switch
        return;
    }
    free_slabs_.push_back(slab);
    std::push_heap(free_slabs_.begin(), free_slabs_.end(), std::greater<>{});
}

inline void PrefixCacheIndex::set_node_slab(std::uint32_t node, std::uint32_t slab) noexcept {
    Node& entry = nodes_[node];
    if (entry.host_slab != kNoId) { node_of_slab_[entry.host_slab] = kNoId; }
    entry.host_slab = slab;
    if (slab != kNoId) { node_of_slab_[slab] = node; }
}

inline bool PrefixCacheIndex::yields_host_room(const Snapshot& snapshot) const noexcept {
    if (!host_shrinking()) { return true; }
    return std::any_of(snapshot.host_slabs.begin(), snapshot.host_slabs.end(),
                       [this](std::uint32_t slab) { return slab < ceiling_; });
}

inline void PrefixCacheIndex::free_ssd_slab(std::uint32_t slab) {
    free_ssd_slabs_.push_back(slab);
    std::push_heap(free_ssd_slabs_.begin(), free_ssd_slabs_.end(), std::greater<>{});
}

} // namespace infernix::runtime::prefix_cache
