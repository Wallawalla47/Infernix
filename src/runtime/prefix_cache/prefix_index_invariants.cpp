// PrefixCacheIndex::check_invariants (prefix_index.h): every structural invariant of the index, recomputed
// from scratch (tests call it after each operation).

#include "runtime/prefix_cache/prefix_index_internal.h"

#include <algorithm>
#include <vector>

namespace infernix::runtime::prefix_cache {

void PrefixCacheIndex::check_invariants() const {
    std::vector<std::uint32_t> live(nodes_.size(), 0);
    std::vector<std::uint32_t> warm(nodes_.size(), 0);
    std::vector<std::uint32_t> slab_uses(config_.host_slabs, 0);
    std::vector<std::uint32_t> ssd_uses(config_.ssd_slabs, 0);
    std::uint64_t clean_nodes = 0, clean_listed_nodes = 0, clean_image_slabs = 0, clean_dead_nodes = 0;
    std::uint32_t dead_nodes          = 0;
    std::uint32_t ssd_dead_nodes      = 0;
    std::uint32_t occupied_nodes      = 0;
    std::uint32_t transfer_held_nodes = 0;
    for (std::uint32_t i = 0; i < nodes_.size(); ++i) {
        const Node& node = nodes_[i];
        if (!node.occupied) {
            if (lru_links_[i].list != kNoList || node.in_dead || node.in_ssd_dead || node.clean_list != kNoList ||
                node.transfer_counted ||
                node.clean_counted) {
                invariant("free node is linked");
            }
            continue;
        }
        ++occupied_nodes;
        if (node.parent != kNoId) {
            const Node& parent = nodes_[node.parent];
            if (!parent.occupied || parent.depth + 1U != node.depth) {
                invariant("node parent is inconsistent");
            }
            if (parent.pins < node.pins || parent.io_pins < node.io_pins) { invariant("pinned node has a less pinned parent"); }
        } else if (node.depth != 0) {
            invariant("root child has nonzero depth");
        }
        if (node.device == CopyState::Absent && node.host == CopyState::Absent && node.ssd == CopyState::Absent) {
            invariant("node has no copy");
        }
        if (node.ssd == CopyState::Filling && node.host != CopyState::Resident) {
            invariant("node SSD write has no Host source");
        }
        if (node.host != CopyState::Absent) {
            if (node.host_slab >= config_.host_slabs) { invariant("node slab out of range"); }
            ++slab_uses[node.host_slab];
        }
        if (node.ssd != CopyState::Absent) {
            if (node.ssd_slab >= config_.ssd_slabs) { invariant("node SSD slab out of range"); }
            ++ssd_uses[node.ssd_slab];
        }
        if (node.device_pins > node.pins || node.transfer_pins > node.device_pins) {
            invariant("node Device holders exceed its pins");
        }
        // Recomputed here, not through node_lru_list: a dead block (no snapshot at or below it, no holder)
        // is on a dead list, every other evictable one on its kind's and hold's list.
        const bool lru  = device_evictable(node);
        const bool dead_copy = lru && node.live_below == 0 && node.pins == 0 && node.io_pins == 0;
        const std::uint8_t list = !lru        ? kNoList
                                  : dead_copy ? (node_backed(node) ? kLruDeadBacked : kLruDeadUnbacked)
                                              : lru_list(node_backed(node), node.hold);
        if (lru_links_[i].list != list) { invariant("node LRU membership is inconsistent"); }
        if (node.host != CopyState::Absent && node_of_slab_[node.host_slab] != i) {
            invariant("node Host slab is not recorded as its own");
        }
        const bool transfer_held = (node.device == CopyState::Resident || node.device == CopyState::Filling) &&
                                   node.transfer_pins != 0 && node.device_pins == node.transfer_pins;
        if (transfer_held != node.transfer_counted) { invariant("node transfer-held membership is inconsistent"); }
        if (transfer_held) { ++transfer_held_nodes; }
        const bool dead =
            node.host == CopyState::Resident && node.live_below == 0 && !held(node);
        if (dead != node.in_dead) { invariant("node dead membership is inconsistent"); }
        const bool ssd_dead = node.ssd == CopyState::Resident && node.live_below == 0 && !held(node);
        if (ssd_dead != node.in_ssd_dead) { invariant("node SSD dead membership is inconsistent"); }
        const bool clean = node.host == CopyState::Resident && node.ssd == CopyState::Resident;
        if (clean != node.clean_counted) { invariant("node clean count is inconsistent"); }
        if (clean) { ++clean_nodes; }
        if ((clean && dead) != node.clean_dead_counted) { invariant("node clean-and-dead count is inconsistent"); }
        if (clean && dead) { ++clean_dead_nodes; }
        const std::uint8_t clean_list = clean && !held(node) && node.hold == kNoHold && node.live_below != 0
                                            ? (node.warm_below == 0 ? kCold : kCleanWarm)
                                            : kNoList;
        if (clean_list != node.clean_list) { invariant("node clean list membership is inconsistent"); }
        if (clean_list != kNoList) { ++clean_listed_nodes; }
        if (node.in_dead) { ++dead_nodes; }
        if (node.in_ssd_dead) { ++ssd_dead_nodes; }
    }
    if (ssd_dead_nodes != ssd_dead_count_) { invariant("SSD dead node count mismatch"); }
    if (clean_nodes != host_clean_nodes_) { invariant("clean node count mismatch"); }
    if (clean_dead_nodes != host_clean_dead_) { invariant("clean-and-dead node count mismatch"); }
    if (clean_listed_nodes != host_clean_listed_) { invariant("clean list node count mismatch"); }
    if (dead_nodes != dead_count_) { invariant("dead node count mismatch"); }
    if (transfer_held_nodes != transfer_releasable_) { invariant("transfer-held node count mismatch"); }
    if (occupied_nodes != node_count_) { invariant("node count mismatch"); }
    std::uint32_t resident = 0;
    for (const Node& node : nodes_) {
        if (node.occupied && node.device == CopyState::Resident) { ++resident; }
    }
    if (resident != device_resident_) { invariant("device resident count mismatch"); }
    if (static_cast<std::uint32_t>(
            std::count(slot_state_.begin(), slot_state_.end(), SlotState::Free)) != free_slots_) {
        invariant("free device slot count mismatch");
    }
    std::uint32_t occupied_snapshots = 0;
    std::vector<std::uint32_t> retained(nodes_.size(), 0);
    std::uint32_t top_level = 0;
    for (std::uint32_t i = 0; i < snapshots_.size(); ++i) {
        const Snapshot& snapshot = snapshots_[i];
        if (!snapshot.occupied) { continue; }
        ++occupied_snapshots;
        if (!snapshot_valid(snapshot)) { invariant("indexed snapshot is not valid"); }
        const bool is_warm = snapshot.device_slot != kNoId || snapshot.host != CopyState::Absent;
        if (is_warm != snapshot.warm_counted) { invariant("snapshot warmth is inconsistent"); }
        const bool is_clean = snapshot.host == CopyState::Resident && snapshot.ssd == CopyState::Resident;
        if (is_clean != snapshot.clean_counted) { invariant("snapshot clean count is inconsistent"); }
        if (is_clean) { clean_image_slabs += snapshot_slab_count(snapshot); }
        if (snapshot.ssd == CopyState::Filling && snapshot.host != CopyState::Resident) {
            invariant("snapshot SSD write has no Host source");
        }
        if (snapshot.ssd != CopyState::Absent) {
            if (snapshot.ssd_packed ? snapshot.ssd_slabs.size() >= ssd_snapshot_slab_count(snapshot) ||
                                          snapshot.ssd_slabs.size() < 1U + (ssd_snapshot_slab_count(snapshot) - ssd_image_slab_count())
                                    : snapshot.ssd_slabs.size() != ssd_snapshot_slab_count(snapshot)) {
                invariant("snapshot SSD slab count is inconsistent");
            }
            for (const std::uint32_t slab : snapshot.ssd_slabs) {
                if (slab >= config_.ssd_slabs) { invariant("snapshot SSD slab out of range"); }
                ++ssd_uses[slab];
            }
        } else if (!snapshot.ssd_slabs.empty()) {
            invariant("snapshot without an SSD copy holds SSD slabs");
        }
        for (std::uint32_t current = snapshot.anchor; current != kNoId;
             current               = nodes_[current].parent) {
            ++live[current];
            if (!snapshot.superseded) { ++retained[current]; }
            if (is_warm) { ++warm[current]; }
        }
        if (snapshot.ancestor != find_ancestor(i)) {
            invariant("snapshot ancestor is not the nearest snapshot above it");
        }
        const std::vector<std::uint32_t>& siblings =
            snapshot.ancestor == kNoId ? top_snapshots_ : snapshots_[snapshot.ancestor].dependents;
        if (std::count(siblings.begin(), siblings.end(), i) != 1) {
            invariant("snapshot is not listed once under its ancestor");
        }
        if (snapshot.ancestor == kNoId) { ++top_level; }
        for (const std::uint32_t other : snapshot.dependents) {
            if (!snapshots_[other].occupied || snapshots_[other].ancestor != i) {
                invariant("snapshot dependent does not name it as its ancestor");
            }
        }
        for (const std::uint32_t slab : snapshot.host_slabs) {
            if (slab >= config_.host_slabs) { invariant("snapshot slab out of range"); }
            ++slab_uses[slab];
        }
        if (snapshot.device_slot != kNoId &&
            (slot_state_[snapshot.device_slot] != SlotState::Owned ||
             slot_owner_[snapshot.device_slot] != i)) {
            invariant("snapshot slot ownership is inconsistent");
        }
    }
    if (occupied_snapshots != snapshot_count_) { invariant("snapshot count mismatch"); }
    if (top_level != top_snapshots_.size()) {
        invariant("top-level snapshot list is inconsistent");
    }
    for (std::uint32_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].occupied && nodes_[i].live_below != live[i]) {
            invariant("live snapshot count is inconsistent");
        }
        if (nodes_[i].occupied && nodes_[i].retained_below != retained[i]) {
            invariant("retained snapshot count is inconsistent");
        }
        if (nodes_[i].occupied && nodes_[i].warm_below != warm[i]) {
            invariant("warm snapshot count is inconsistent");
        }
    }
    if (clean_image_slabs != host_clean_image_slabs_) { invariant("clean image slab count mismatch"); }
    for (const std::uint32_t slab : free_ssd_slabs_) {
        if (slab >= config_.ssd_slabs) { invariant("free SSD slab out of range"); }
        ++ssd_uses[slab];
    }
    for (std::uint32_t slab = 0; slab < ssd_uses.size(); ++slab) {
        if (ssd_uses[slab] != 1U) { invariant("SSD slab is leaked or shared"); }
    }
    for (const std::uint32_t slab : free_slabs_) {
        if (slab >= ceiling_) { invariant("free host slab past the shrink ceiling"); }
        ++slab_uses[slab];
    }
    // Free + owned + reserved + planned move destinations = every slab in service below the shrink ceiling, each
    // exactly once; past the ceiling a slab is in use at most once (until it moves or is freed, it retires); a retired
    // slab holds nothing.
    for (std::uint32_t slab = 0; slab < slab_reserved_.size(); ++slab) {
        if (slab_reserved_[slab]) { ++slab_uses[slab]; }
    }
    std::uint32_t moving      = 0;
    std::uint32_t moves_live  = 0;
    std::uint32_t slab_owners = 0;
    for (const std::uint32_t owner : node_of_slab_) {
        if (owner == kNoId) { continue; }
        ++slab_owners;
        if (owner >= nodes_.size() || !nodes_[owner].occupied || nodes_[owner].host == CopyState::Absent) {
            invariant("Host slab is recorded for a node that does not hold it");
        }
    }
    std::uint32_t host_nodes = 0;
    for (const Node& node : nodes_) { host_nodes += node.occupied && node.host != CopyState::Absent ? 1U : 0U; }
    if (slab_owners != host_nodes) { invariant("Host slab owners and nodes with a Host copy differ"); }
    for (const PendingHostMove& move : host_moves_) {
        if (move.to >= ceiling_ || move.from < ceiling_ || !slab_moving_[move.from]) {
            invariant("planned host move is inconsistent");
        }
        ++slab_uses[move.to];
        if (move.snapshot) {
            const Snapshot& holder = snapshots_[move.entry];
            moves_live += holder.occupied && holder.host == CopyState::Resident &&
                                  move.position < holder.host_slabs.size() && holder.host_slabs[move.position] == move.from
                              ? 1U
                              : 0U;
        } else {
            const Node& holder = nodes_[move.entry];
            moves_live += holder.occupied && holder.host == CopyState::Resident && holder.host_slab == move.from ? 1U : 0U;
        }
    }
    if (moves_live != moves_live_) { invariant("planned host moves still holding their source miscounted"); }
    for (std::uint32_t slab = 0; slab < slab_moving_.size(); ++slab) { moving += slab_moving_[slab] ? 1U : 0U; }
    if (moving != host_moves_.size()) { invariant("planned host moves and moving slabs differ"); }
    if (ceiling_ > host_limit_) { invariant("host shrink ceiling past the limit"); }
    std::uint32_t past_in_use = 0;
    for (std::uint32_t slab = 0; slab < slab_uses.size(); ++slab) {
        const bool ok = slab < ceiling_ ? slab_uses[slab] == 1U : slab < host_limit_ ? slab_uses[slab] <= 1U : slab_uses[slab] == 0U;
        if (!ok) { invariant("host slab is leaked or shared"); }
        past_in_use += slab >= ceiling_ && slab < host_limit_ && slab_uses[slab] == 1U ? 1U : 0U;
    }
    if (host_shrinking() && past_in_use != past_in_use_) { invariant("host slabs in use past the shrink ceiling miscounted"); }
    std::array<std::uint32_t, 2> kinds{};
    for (std::size_t list = 0; list < kLruLists; ++list) {
        std::uint32_t counted = 0;
        for (std::uint32_t entry = lru_[list].head; entry != kNoId;
             entry               = lru_links_[entry].next) {
            ++counted;
        }
        if (counted != lru_[list].count) { invariant("LRU count mismatch"); }
        kinds[list % 2] += counted;
    }
    if (kinds[kBacked] != lru_backed_ || kinds[kUnbacked] != lru_unbacked_) {
        invariant("LRU totals mismatch");
    }
    // Queue holds: each held snapshot names its rank, and each node carries the lowest rank of a
    // held snapshot on whose anchor path it lies.
    std::vector<std::uint8_t> holds(nodes_.size(), kNoHold);
    for (std::uint32_t rank = 0; rank < holds_.size(); ++rank) {
        const std::uint32_t held = holds_[rank];
        if (held == kNoId) { continue; }
        if (!snapshots_[held].occupied || snapshots_[held].hold != rank) {
            invariant("held snapshot does not carry its rank");
        }
        for (std::uint32_t current = snapshots_[held].anchor; current != kNoId;
             current               = nodes_[current].parent) {
            holds[current] = std::min(holds[current], static_cast<std::uint8_t>(rank));
        }
    }
    for (std::uint32_t i = 0; i < snapshots_.size(); ++i) {
        const Snapshot& snapshot = snapshots_[i];
        if (snapshot.hold != kNoHold &&
            (!snapshot.occupied || snapshot.hold >= holds_.size() || holds_[snapshot.hold] != i)) {
            invariant("snapshot hold is not in the queue holds");
        }
        const std::uint32_t entry = config_.max_nodes + i;
        const bool lru            = snapshot.occupied && snapshot.tail_len != 0 &&
                         snapshot.tail_device_copy == CopyState::Resident && snapshot.pins == 0 &&
                         (snapshot.io_pins == 0 || tail_backed(snapshot));
        const bool backed = tail_backed(snapshot);
        if (lru_links_[entry].list != (lru ? lru_list(backed, snapshot.hold) : kNoList)) {
            invariant("snapshot tail LRU membership is inconsistent");
        }
    }
    for (std::uint32_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].occupied && nodes_[i].hold != holds[i]) {
            invariant("node hold is inconsistent");
        }
    }
}

} // namespace infernix::runtime::prefix_cache
