// PrefixCacheIndex's Host tier (prefix_index.h): Host slab eviction and room, slab allocation, and the
// shrink that moves live slabs below a lower limit and retires the rest.

#include "runtime/prefix_cache/prefix_index_internal.h"

#include <algorithm>
#include <functional>

namespace infernix::runtime::prefix_cache {

bool PrefixCacheIndex::evict_for_slabs(std::uint32_t protect_snapshot, double claim, bool for_moves) {
    if (dead_head_ != kNoId) {
        const std::uint32_t victim = dead_head_;
        ++counters_.host_dead_reclaims;
        if (nodes_[victim].device == CopyState::Absent) {
            // The dead node has no other copy worth keeping; its subtree is dead and unpinned too.
            remove_subtree(victim);
        } else {
            drop_node_host_copy(victim);
            if (nodes_[victim].ssd == CopyState::Resident) { drop_node_ssd_copy(victim); }
        }
        return true;
    }
    // With an SSD tier (§5.7) Host copies the SSD also holds go first, where nothing above the SSD
    // still needs them: blocks under no Host- or Device-resident snapshot image.
    if (clean_head_[kCold] != kNoId) {
        drop_node_host_copy(clean_head_[kCold]);
        ++counters_.host_block_demotions;
        return true;
    }
    // During a shrink a snapshot whose slabs all lie past the ceiling frees nothing an allocation can take
    // (they retire, and its planned moves keep their destinations until they commit): it is no victim.
    const HostVictims victims = pick_host_victims(protect_snapshot, !for_moves);
    if (victims.superseded != kNoId) {
        // Only requests diverging before its lineage's newer snapshot could use it: with an SSD
        // copy it moves there, otherwise both copies go, and GDSF time does not advance.
        if (snapshots_[victims.superseded].ssd == CopyState::Resident) {
            demote_snapshot(victims.superseded);
        } else {
            remove_snapshot(victims.superseded);
            ++counters_.superseded_evictions;
            ++counters_.host_snapshot_evictions;
        }
        return true;
    }
    if (victims.clean != kNoId) {
        // Restorable from the SSD: its Host copy yields without loss, and its blocks no other Host or
        // Device image needs follow as cold blocks.
        demote_snapshot(victims.clean);
        return true;
    }
    if (clean_head_[kCleanWarm] != kNoId) {
        drop_node_host_copy(clean_head_[kCleanWarm]);
        ++counters_.host_block_demotions;
        return true;
    }
    std::uint32_t victim = victims.unclean;
    if (victim == kNoId) {
        // A held snapshot's slabs serve a queued request sooner than anything new would.
        if (std::any_of(holds_.begin(), holds_.end(), [&](std::uint32_t held) {
                return held != kNoId && snapshots_[held].host == CopyState::Resident;
            })) {
            ++counters_.held_host_refusals;
        }
        return false;
    }
    // A retained snapshot only yields its slabs to something worth at least as much: GDSF
    // never inserts what it would evict next. One the allocation found within its claim when it began
    // stays a victim after an earlier eviction of this allocation revalued it (its nearest retained
    // snapshot above went, so it now carries that one's prefill): the feasibility check counted it, and
    // stopping here would have lost the earlier victims for nothing.
    if (snapshots_[victim].priority > claim) {
        victim = victims.claimed;
        if (victim == kNoId) { return false; }
    }
    Snapshot& chosen = snapshots_[victim];
    // GDSF time rises to what the claim displaced, never past the claim itself.
    inflation_ = std::max(inflation_, std::min(chosen.priority, claim));
    if (chosen.device_slot != kNoId &&
        (chosen.tail_len == 0 || chosen.tail_device_copy == CopyState::Resident)) {
        // Still complete on the Device: only its host copy yields its slabs.
        for (const std::uint32_t slab : chosen.host_slabs) { free_slab(slab); }
        chosen.host_slabs.clear();
        chosen.host = CopyState::Absent;
        refresh_tail(victim);
    } else {
        ++kind_stats(chosen).host_losses;
        remove_snapshot(victim);
    }
    ++counters_.host_snapshot_evictions;
    return true;
}

std::uint64_t PrefixCacheIndex::host_room(std::uint64_t count, std::uint32_t protect_snapshot, double claim) const {
    std::uint64_t room = free_slabs_.size();
    if (room >= count) { return room; }
    // evict_for_slabs' first victims go whatever the claim: dead blocks and the clean lists' Host copies
    // the SSD tier also holds. The lists hold no dead block (those are on the dead list) and no held
    // one, so each slab counts once and only where evict_for_slabs may take it.
    const bool shrinking = host_shrinking();
    if (!shrinking) {
        room += static_cast<std::uint64_t>(dead_count_) + host_clean_listed_;
    } else {
        // A slab at or past the ceiling retires when freed: only the lists' slabs below it are room. The
        // walks end at `count`, and a shrink's first plan has dropped the dead blocks past its ceiling.
        for (std::uint32_t node = dead_head_; node != kNoId && room < count; node = nodes_[node].dead_next) {
            room += nodes_[node].host_slab < ceiling_ ? 1U : 0U;
        }
        for (const std::uint8_t list : {kCold, kCleanWarm}) {
            for (std::uint32_t node = clean_head_[list]; node != kNoId && room < count; node = nodes_[node].clean_next) {
                room += nodes_[node].host_slab < ceiling_ ? 1U : 0U;
            }
        }
    }
    if (room >= count) { return room; }
    // Then the snapshots its order may take: superseded and SSD-backed ones whatever the claim, retained
    // ones valued at most `claim`; held, pinned and queue-held ones never. A snapshot removed outright
    // (not demoted to the SSD tier, not keeping its Device copy) leaves dead the blocks no other
    // snapshot below needs, and they go too. A removal reaches a node's parent only once every
    // snapshot at or below the node is removed (it passes them all up then), so each node is visited
    // once per snapshot anchored on it and once per child that empties: a check costs O(snapshots +
    // nodes), not the depth of every lineage per snapshot.
    std::unordered_map<std::uint32_t, std::uint32_t> removed_below;
    ++operations_.room_scans;
    for (std::uint32_t i = 0; i < snapshots_.size() && room < count; ++i) {
        const Snapshot& snapshot = snapshots_[i];
        ++operations_.snapshot_visits;
        if (!snapshot.occupied || i == protect_snapshot || held(snapshot) || snapshot.hold != kNoHold ||
            snapshot.host != CopyState::Resident) {
            continue;
        }
        const bool on_ssd = snapshot.ssd == CopyState::Resident;
        if (!snapshot.superseded && !on_ssd && snapshot.priority > claim) { continue; }
        if (shrinking) {
            // As pick_host_victims: one with no slab below the ceiling is no victim.
            const auto below = static_cast<std::uint64_t>(
                std::count_if(snapshot.host_slabs.begin(), snapshot.host_slabs.end(),
                              [this](std::uint32_t slab) { return slab < ceiling_; }));
            if (below == 0) { continue; }
            room += below;
        } else {
            room += snapshot.host_slabs.size();
        }
        const bool keeps_device =
            !snapshot.superseded && snapshot.device_slot != kNoId &&
            (snapshot.tail_len == 0 || snapshot.tail_device_copy == CopyState::Resident);
        if (on_ssd || keeps_device) { continue; }
        std::uint32_t passed = 1;
        for (std::uint32_t node = snapshot.anchor; node != kNoId && room < count; node = nodes_[node].parent) {
            const Node& entry = nodes_[node];
            ++operations_.room_node_visits;
            std::uint32_t& removed = removed_below[node];
            removed += passed;
            if (removed != entry.live_below) { break; }
            // A block already dead, or on a clean list, is counted above.
            if (entry.host == CopyState::Resident && !held(entry) && !entry.in_dead &&
                entry.clean_list == kNoList && entry.host_slab < ceiling_) {
                ++room;
            }
            passed = entry.live_below;
        }
    }
    return room;
}

bool PrefixCacheIndex::allocate_slabs(std::uint32_t count, std::vector<std::uint32_t>& out,
                                      std::uint32_t protect_snapshot, double claim) {
    out.clear();
    return make_slabs_free(count, protect_snapshot, claim) && take_free_slabs(count, out);
}

bool PrefixCacheIndex::make_slabs_free(std::uint32_t count, std::uint32_t protect_snapshot, double claim) {
    if (count > ceiling_) { return false; }
    // Evicted one at a time, victims would be lost even when the rest cannot free enough (a pause's
    // resume point against a Host tier of held snapshots evicted every unheld one, then failed): the
    // eviction starts only when what it may take covers the allocation.
    if (free_slabs_.size() < count && !host_room_possible(count, protect_snapshot, claim)) {
        // As evict_for_slabs counts it: a held snapshot's slabs serve a queued request sooner.
        if (std::any_of(holds_.begin(), holds_.end(), [&](std::uint32_t held_snapshot) {
                return held_snapshot != kNoId && snapshots_[held_snapshot].host == CopyState::Resident;
            })) {
            ++counters_.held_host_refusals;
        }
        return false;
    }
    if (free_slabs_.size() < count && claim < std::numeric_limits<double>::infinity()) {
        // The claim is judged on the values the allocation begins with: what the check above counted
        // stays a victim (evict_for_slabs), so the allocation it found feasible completes.
        ++claim_epoch_;
        // A block write's claim is below every value: no retained snapshot is stamped, so no scan.
        if (claim > -std::numeric_limits<double>::infinity()) {
            for (Snapshot& snapshot : snapshots_) {
                if (snapshot.occupied && snapshot.priority <= claim) { snapshot.claim_stamp = claim_epoch_; }
            }
        }
    }
    while (free_slabs_.size() < count) {
        if (!evict_for_slabs(protect_snapshot, claim)) { return false; }
    }
    return true;
}

std::uint32_t PrefixCacheIndex::begin_host_shrink(std::uint32_t limit, std::uint32_t granule) {
    if (host_shrinking()) { throw std::logic_error("prefix cache host shrink is already under way"); }
    granule          = std::max(granule, 1U);
    limit            = std::min(limit, host_limit_);
    const auto round = [&](std::uint64_t slabs) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(host_limit_, (slabs + granule - 1U) / granule * granule));
    };
    // This may run between rounds, so it visits the slabs from `limit` on and the snapshot table, never every node:
    // - the slabs a transfer reads or writes now (reservations, filling copies, I/O-pinned entries): the ceiling stays
    //   above them, so the shrink never waits for one. Only those at or past `limit` matter (the ceiling never lies
    //   below it);
    // - the nodes holding a slab at or past `limit` (node_of_slab_), for plan_host_moves;
    // - what the eviction below can free at all (host_room: the order's own count, so reservations, filling copies,
    //   pinned, I/O-pinned and queue-held entries, and the blocks only a held or Device-only snapshot keeps alive, are
    //   no room), so nothing is evicted for a shrink that cannot retire a granule.
    std::uint32_t blocked = 0;
    std::vector<std::uint32_t> past;
    for (std::uint32_t slab = limit; slab < host_limit_; ++slab) {
        if (slab_reserved_[slab]) { blocked = slab + 1U; }
        const std::uint32_t index = node_of_slab_[slab];
        if (index == kNoId) { continue; }
        past.push_back(index);
        if (nodes_[index].host == CopyState::Filling || nodes_[index].io_pins != 0) { blocked = slab + 1U; }
    }
    for (const Snapshot& snapshot : snapshots_) {
        if (!snapshot.occupied || (snapshot.host != CopyState::Filling && snapshot.io_pins == 0)) { continue; }
        for (const std::uint32_t slab : snapshot.host_slabs) { blocked = std::max(blocked, slab + 1U); }
    }
    // In use once everything evictable is gone; host_room stops counting at what `limit` needs.
    const std::uint64_t wanted_room = host_limit_ - limit;
    const std::uint64_t room        = host_room(wanted_room, kNoId, std::numeric_limits<double>::infinity());
    const std::uint64_t floor       = host_limit_ - std::min<std::uint64_t>(room, host_limit_);
    const std::uint32_t target      = round(std::max<std::uint64_t>({limit, floor, blocked}));
    if (target >= host_limit_) { return host_limit_; }
    const auto in_use = [this] { return host_limit_ - static_cast<std::uint32_t>(free_slabs_.size()); };
    while (in_use() > target) {
        if (!evict_for_slabs(kNoId, std::numeric_limits<double>::infinity())) { break; }
    }
    const std::uint32_t ceiling = round(std::max(target, in_use()));
    if (ceiling >= host_limit_) { return host_limit_; }
    ceiling_                  = ceiling;
    const std::size_t free_before = free_slabs_.size();
    std::erase_if(free_slabs_, [this](std::uint32_t slab) { return slab >= ceiling_; });
    std::make_heap(free_slabs_.begin(), free_slabs_.end(), std::greater<>{});
    past_in_use_  = (host_limit_ - ceiling_) - static_cast<std::uint32_t>(free_before - free_slabs_.size());
    moves_live_   = 0;
    shrink_nodes_ = std::move(past);
    return ceiling_;
}

std::vector<PrefixCacheIndex::HostMove> PrefixCacheIndex::plan_host_moves() {
    std::vector<HostMove> out;
    if (!host_shrinking()) { return out; }
    // Dead KV past the ceiling (no snapshot resumes through it) goes rather than moves: allocation evicts it first.
    for (const std::uint32_t index : shrink_nodes_) {
        const Node& node = nodes_[index];
        if (!node.occupied || !node.in_dead || node.host != CopyState::Resident || node.host_slab < ceiling_ ||
            held(node)) {
            continue;
        }
        ++counters_.host_dead_reclaims;
        if (node.device == CopyState::Absent) {
            remove_subtree(index); // its subtree is dead and unpinned too
        } else {
            drop_node_host_copy(index);
            if (nodes_[index].ssd == CopyState::Resident) { drop_node_ssd_copy(index); }
        }
    }
    std::erase_if(shrink_nodes_, [this](std::uint32_t index) {
        const Node& node = nodes_[index];
        return !node.occupied || node.host == CopyState::Absent || node.host_slab < ceiling_;
    });
    // Collected once: an eviction below only takes entries away, and what it took is skipped when the slabs are
    // paired (collecting again after every eviction walked the nodes and the snapshot table each time).
    std::vector<PendingHostMove> wanted;
    for (const std::uint32_t index : shrink_nodes_) {
        const Node& node = nodes_[index];
        if (node.occupied && node.host == CopyState::Resident && node.io_pins == 0 && node.host_slab >= ceiling_ &&
            !slab_moving_[node.host_slab]) {
            wanted.push_back({.snapshot = false, .entry = index, .position = 0, .from = node.host_slab});
        }
    }
    for (std::uint32_t index = 0; index < snapshots_.size(); ++index) {
        const Snapshot& snapshot = snapshots_[index];
        if (!snapshot.occupied || snapshot.host != CopyState::Resident || snapshot.io_pins != 0) { continue; }
        for (std::uint32_t k = 0; k < snapshot.host_slabs.size(); ++k) {
            const std::uint32_t slab = snapshot.host_slabs[k];
            if (slab >= ceiling_ && !slab_moving_[slab]) {
                wanted.push_back({.snapshot = true, .entry = index, .position = k, .from = slab});
            }
        }
    }
    // Allocations since begin_host_shrink may have taken the room below the ceiling: they evict as theirs would
    // (and an entry past the ceiling too: one move less). Each slab an eviction frees past the ceiling that no
    // planned move names was one of `wanted`.
    std::size_t needed = wanted.size();
    while (free_slabs_.size() < needed) {
        const std::uint32_t unplanned = past_in_use_ - moves_live_;
        if (!evict_for_slabs(kNoId, std::numeric_limits<double>::infinity(), true)) { break; }
        needed -= std::min<std::size_t>(needed, unplanned - (past_in_use_ - moves_live_));
    }
    // Reserved before the first slab is taken: the loop below cannot throw, so no `to` is taken without its move.
    const std::size_t planned = std::min(wanted.size(), free_slabs_.size());
    out.reserve(planned);
    host_moves_.reserve(host_moves_.size() + planned);
    std::vector<std::uint32_t> to;
    to.reserve(1);
    for (PendingHostMove move : wanted) {
        if (host_move_holder(move) == nullptr) { continue; } // evicted above
        if (!take_free_slabs(1, to)) { break; }              // the rest at a later call
        move.to                 = to.front();
        slab_moving_[move.from] = true;
        ++moves_live_;
        host_moves_.push_back(move);
        out.push_back({move.from, move.to});
    }
    return out;
}

void PrefixCacheIndex::cancel_host_moves(std::span<const HostMove> moves) noexcept {
    // plan_host_moves appended them last, in order; each `to` came off the free list, which kept its capacity.
    for (auto move = moves.rbegin(); move != moves.rend(); ++move) {
        if (host_moves_.empty() || host_moves_.back().from != move->from || host_moves_.back().to != move->to) {
            std::terminate(); // not the last plan: the slabs' ownership is unknown
        }
        slab_moving_[move->from] = false;
        --moves_live_;
        free_slab(move->to);
        host_moves_.pop_back();
    }
}

std::uint32_t* PrefixCacheIndex::host_move_holder(const PendingHostMove& move) noexcept {
    if (move.snapshot) {
        Snapshot& snapshot = snapshots_[move.entry];
        return snapshot.occupied && snapshot.host == CopyState::Resident && move.position < snapshot.host_slabs.size() &&
                       snapshot.host_slabs[move.position] == move.from
                   ? &snapshot.host_slabs[move.position]
                   : nullptr;
    }
    Node& node = nodes_[move.entry];
    return node.occupied && node.host == CopyState::Resident && node.host_slab == move.from ? &node.host_slab : nullptr;
}

bool PrefixCacheIndex::commit_host_moves() {
    std::erase_if(host_moves_, [this](const PendingHostMove& move) {
        if (std::uint32_t* holder = host_move_holder(move)) {
            const std::uint32_t io_pins = move.snapshot ? snapshots_[move.entry].io_pins : nodes_[move.entry].io_pins;
            if (io_pins != 0) { return false; } // an SSD job reads it now: a later call
            if (move.snapshot) {
                *holder = move.to; // `from`, past the ceiling, is retired
            } else {
                set_node_slab(move.entry, move.to);
            }
            --past_in_use_;
            --moves_live_;
        } else {
            free_slab(move.to); // evicted or dropped meanwhile; its `from` retired then
        }
        slab_moving_[move.from] = false;
        return true;
    });
    return host_moves_.empty();
}

bool PrefixCacheIndex::finish_host_shrink() {
    if (!host_shrinking()) { return true; }
    if (!host_moves_.empty() || past_in_use_ != 0) { return false; }
    host_limit_ = ceiling_;
    std::vector<std::uint32_t>().swap(shrink_nodes_);
    return true;
}

void PrefixCacheIndex::grow_host(std::uint32_t limit) {
    if (host_shrinking()) { throw std::logic_error("prefix cache host growth during a shrink"); }
    limit                     = std::min(limit, config_.host_slabs);
    const std::uint32_t first = host_limit_;
    host_limit_               = std::max(host_limit_, limit);
    ceiling_                  = host_limit_;
    for (std::uint32_t slab = first; slab < limit; ++slab) { free_slab(slab); }
}

} // namespace infernix::runtime::prefix_cache
