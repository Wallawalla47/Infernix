// PrefixCacheIndex's SSD tier bookkeeping (prefix_index.h): which Host copies are demoted, SSD slabs and
// their eviction, SSD writes and promotions back to the Host tier.

#include "runtime/prefix_cache/prefix_index_internal.h"

#include <algorithm>
#include <functional>

namespace infernix::runtime::prefix_cache {

PrefixCacheIndex::HostVictims PrefixCacheIndex::pick_host_victims(std::uint32_t protect, bool yielding) const {
    HostVictims victims;
    ++operations_.victim_scans;
    operations_.snapshot_visits += snapshots_.size();
    yielding = yielding && host_shrinking();
    const auto lower = [&](std::uint32_t candidate, std::uint32_t best) {
        const Snapshot& a = snapshots_[candidate];
        const Snapshot& b = snapshots_[best];
        return a.priority < b.priority || (a.priority == b.priority && a.last_hit_tick < b.last_hit_tick);
    };
    for (std::uint32_t i = 0; i < snapshots_.size(); ++i) {
        const Snapshot& snapshot = snapshots_[i];
        if (!snapshot.occupied || i == protect || held(snapshot) || snapshot.hold != kNoHold ||
            snapshot.host != CopyState::Resident || (yielding && !yields_host_room(snapshot))) {
            continue;
        }
        if (snapshot.superseded) {
            if (victims.superseded == kNoId ||
                snapshot.superseded_tick < snapshots_[victims.superseded].superseded_tick) {
                victims.superseded = i;
            }
        } else if (snapshot.ssd == CopyState::Resident) {
            if (victims.clean == kNoId || lower(i, victims.clean)) { victims.clean = i; }
        } else {
            if (victims.unclean == kNoId || lower(i, victims.unclean)) { victims.unclean = i; }
            if (snapshot.claim_stamp == claim_epoch_ && (victims.claimed == kNoId || lower(i, victims.claimed))) {
                victims.claimed = i;
            }
        }
    }
    return victims;
}

void PrefixCacheIndex::demote_snapshot(std::uint32_t index) {
    Snapshot& snapshot = snapshots_[index];
    if (snapshot.host != CopyState::Resident || snapshot.ssd != CopyState::Resident || held(snapshot)) {
        invariant("prefix cache demotes a snapshot the SSD tier does not hold");
    }
    for (const std::uint32_t slab : snapshot.host_slabs) { free_slab(slab); }
    snapshot.host_slabs.clear();
    snapshot.host = CopyState::Absent;
    // The tail stays backed (its SSD slab), and the anchor path's blocks lose a warm image above
    // them: those no other image needs become cold.
    refresh_tail(index);
    ++counters_.host_image_demotions;
}

void PrefixCacheIndex::drop_node_ssd_copy(std::uint32_t index) {
    Node& node = nodes_[index];
    if (node.ssd != CopyState::Resident) { invariant("prefix cache drops a non-resident SSD slab"); }
    const std::uint32_t slab = node.ssd_slab;
    node.ssd                 = CopyState::Absent;
    node.ssd_slab            = kNoId;
    free_ssd_slab(slab);
    refresh_node(index);
}

void PrefixCacheIndex::drop_snapshot_ssd_copy(std::uint32_t index) {
    Snapshot& snapshot = snapshots_[index];
    if (snapshot.ssd != CopyState::Resident) { invariant("prefix cache drops a non-resident SSD image"); }
    if (!snapshot.ssd_read && !snapshot.superseded) { ++kind_stats(snapshot).dropped_unread; }
    for (const std::uint32_t slab : snapshot.ssd_slabs) { free_ssd_slab(slab); }
    snapshot.ssd_slabs.clear();
    snapshot.ssd        = CopyState::Absent;
    snapshot.ssd_read   = false;
    snapshot.ssd_packed = false;
    refresh_tail(index);
}

bool PrefixCacheIndex::evict_for_ssd(std::uint32_t protect, double claim) {
    // 1. SSD copies of dead blocks: nothing can resume through them.
    if (ssd_dead_head_ != kNoId) {
        const std::uint32_t victim = ssd_dead_head_;
        ++counters_.ssd_dead_reclaims;
        const Node& node = nodes_[victim];
        if (node.device == CopyState::Absent && node.host != CopyState::Resident) {
            remove_subtree(victim); // its only copy; its subtree is dead and unpinned too
        } else {
            drop_node_ssd_copy(victim);
        }
        return true;
    }
    // 2. Superseded SSD-only snapshots, oldest first; 3. backups of images still on the Host or the
    // Device worth more than the claim, most valuable first (the last the Host would give up); 4.
    // retained SSD-only snapshots worth at most the claim, lowest first. Held ones stay (§9.6).
    std::uint32_t backup = kNoId, superseded = kNoId, retained = kNoId;
    for (std::uint32_t i = 0; i < snapshots_.size(); ++i) {
        const Snapshot& snapshot = snapshots_[i];
        if (!snapshot.occupied || i == protect || snapshot.ssd != CopyState::Resident || held(snapshot)) {
            continue;
        }
        if (snapshot.warm_counted) {
            if (snapshot.host != CopyState::Filling && snapshot.priority > claim &&
                (backup == kNoId || snapshot.priority > snapshots_[backup].priority)) {
                backup = i;
            }
            continue;
        }
        if (snapshot.hold != kNoHold) { continue; }
        if (snapshot.superseded) {
            if (superseded == kNoId || snapshot.superseded_tick < snapshots_[superseded].superseded_tick) {
                superseded = i;
            }
        } else if (snapshot.priority <= claim &&
                   (retained == kNoId || snapshot.priority < snapshots_[retained].priority ||
                    (snapshot.priority == snapshots_[retained].priority &&
                     snapshot.last_hit_tick < snapshots_[retained].last_hit_tick))) {
            retained = i;
        }
    }
    if (superseded != kNoId) {
        remove_snapshot(superseded);
        ++counters_.ssd_snapshot_evictions;
        return true;
    }
    if (backup != kNoId) {
        drop_snapshot_ssd_copy(backup);
        ++counters_.ssd_backup_drops;
        return true;
    }
    if (retained == kNoId) { return false; }
    inflation_ = std::max(inflation_, snapshots_[retained].priority);
    remove_snapshot(retained);
    ++counters_.ssd_snapshot_evictions;
    return true;
}

std::uint64_t PrefixCacheIndex::ssd_room_for(std::uint32_t protect, double claim) const {
    std::uint64_t room = free_ssd_slabs_.size() + ssd_dead_count_;
    for (std::uint32_t i = 0; i < snapshots_.size(); ++i) {
        const Snapshot& snapshot = snapshots_[i];
        if (!snapshot.occupied || i == protect || snapshot.ssd != CopyState::Resident || held(snapshot)) {
            continue;
        }
        const bool yields = snapshot.warm_counted
                                ? snapshot.host != CopyState::Filling && snapshot.priority > claim
                                : snapshot.hold == kNoHold && (snapshot.superseded || snapshot.priority <= claim);
        if (yields) { room += snapshot.ssd_slabs.size(); }
    }
    return room;
}

bool PrefixCacheIndex::allocate_ssd_slabs(std::uint32_t count, std::vector<std::uint32_t>& out,
                                          std::uint32_t protect, double claim) {
    out.clear();
    if (count > config_.ssd_slabs) { return false; }
    while (free_ssd_slabs_.size() < count) {
        if (!evict_for_ssd(protect, claim)) { return false; }
    }
    for (std::uint32_t taken = 0; taken < count; ++taken) {
        std::pop_heap(free_ssd_slabs_.begin(), free_ssd_slabs_.end(), std::greater<>{});
        out.push_back(free_ssd_slabs_.back());
        free_ssd_slabs_.pop_back();
    }
    return true;
}

std::uint64_t PrefixCacheIndex::host_lossless_slabs() const noexcept {
    return free_slabs_.size() + dead_count_ + host_clean_nodes_ + host_clean_image_slabs_;
}

void PrefixCacheIndex::set_ssd_rate(double bytes_per_second) noexcept {
    if (bytes_per_second > 0.0) { config_.cost.ssd_bytes_per_second = bytes_per_second; }
}

std::optional<SsdJob> PrefixCacheIndex::begin_ssd_write(const std::function<bool(const SsdWriteCandidate&)>& admit) {
    if (config_.ssd_slabs == 0) { return std::nullopt; }
    // Candidates in the order the Host tier would lose them: the lowest priority first.
    std::vector<std::uint32_t> order;
    for (std::uint32_t i = 0; i < snapshots_.size(); ++i) {
        const Snapshot& snapshot = snapshots_[i];
        if (snapshot.occupied && !snapshot.superseded && !held(snapshot) && snapshot.hold == kNoHold &&
            snapshot.host == CopyState::Resident && snapshot.ssd == CopyState::Absent) {
            order.push_back(i);
        }
    }
    std::sort(order.begin(), order.end(), [&](std::uint32_t left, std::uint32_t right) {
        const Snapshot& a = snapshots_[left];
        const Snapshot& b = snapshots_[right];
        return a.priority < b.priority || (a.priority == b.priority && a.last_hit_tick < b.last_hit_tick);
    });
    std::vector<std::uint32_t> nodes;
    // A full SSD tier refuses write-backs worth less than what they would displace; a few later
    // candidates (worth more) may still fit.
    constexpr std::size_t kCandidates = 8;
    std::size_t tried                 = 0;
    for (const std::uint32_t index : order) {
        if (tried == kCandidates) { break; }
        // Every block of its path must have a copy the write can read or already the SSD's, and none
        // may be under another write-back.
        nodes.clear();
        bool writable = true;
        for (std::uint32_t current = snapshots_[index].anchor; current != kNoId && writable;
             current               = nodes_[current].parent) {
            const Node& node = nodes_[current];
            if (node.ssd == CopyState::Resident) { continue; }
            writable = node.ssd == CopyState::Absent && node.host == CopyState::Resident;
            nodes.push_back(current);
        }
        if (!writable) { continue; }
        std::reverse(nodes.begin(), nodes.end()); // path order
        Snapshot& snapshot       = snapshots_[index];
        const std::uint32_t need = static_cast<std::uint32_t>(nodes.size()) + ssd_snapshot_slab_count(snapshot);
        if (admit) {
            const SsdWriteCandidate candidate{.kind        = snapshot.kind,
                                              .hits        = snapshot.hits,
                                              .slabs       = need,
                                              .value       = snapshot.priority - snapshot.priority_base,
                                              .first_offer = !snapshot.ssd_offered};
            snapshot.ssd_offered = true;
            if (!admit(candidate)) {
                if (!snapshot.ssd_declined) {
                    snapshot.ssd_declined = true;
                    ++counters_.ssd_declined_snapshots;
                }
                continue;
            }
        }
        ++tried;
        // Checked before anything is evicted: a write that cannot fit displaces nothing.
        if (ssd_room_for(index, snapshot.priority) < need) { continue; }
        std::vector<std::uint32_t> slabs;
        // The room check can pass and the allocation still fail: each SSD eviction re-values the victim's retained
        // dependents, which may rise above this claim. What it freed stays free for the next candidate.
        if (!allocate_ssd_slabs(need, slabs, index, snapshot.priority)) { continue; }
        // The SSD eviction above removes only other snapshots, which never take a block of this path
        // with them (the path stays live), so the candidate and its blocks are as checked.
        SsdJob job;
        job.snapshot = ref_of_snapshot(index);
        job.image    = true;
        std::size_t next = 0;
        for (const std::uint32_t node : nodes) {
            Node& entry    = nodes_[node];
            entry.ssd_slab = slabs[next];
            entry.ssd      = CopyState::Filling;
            job.nodes.push_back(ref_of_node(node));
            job.transfers.push_back(SsdTransfer{.host_slab = entry.host_slab, .ssd_slab = slabs[next]});
            ++next;
        }
        Snapshot& written = snapshots_[index];
        written.ssd_slabs.assign(slabs.begin() + static_cast<std::ptrdiff_t>(next), slabs.end());
        written.ssd = CopyState::Filling;
        for (std::size_t k = 0; k < written.ssd_slabs.size(); ++k) {
            job.transfers.push_back(
                SsdTransfer{.host_slab = written.host_slabs[ssd_to_host_position(k)], .ssd_slab = written.ssd_slabs[k]});
        }
        io_pin_snapshot(job.snapshot.index);
        for (const std::uint32_t node : nodes) { refresh_node(node); }
        return job;
    }
    return std::nullopt;
}

void PrefixCacheIndex::complete_ssd_write(SsdJob& job) {
    for (const NodeRef ref : job.nodes) {
        Node& node = require(ref);
        if (node.ssd != CopyState::Filling) { invariant("prefix cache SSD write not open"); }
        node.ssd = CopyState::Resident;
        refresh_node(ref.index);
    }
    if (job.image) {
        Snapshot& snapshot = require(job.snapshot);
        if (snapshot.ssd != CopyState::Filling) { invariant("prefix cache SSD image write not open"); }
        snapshot.ssd      = CopyState::Resident;
        snapshot.ssd_read = false;
        SsdKindStats& kind = kind_stats(snapshot);
        ++kind.writes;
        kind.written_slabs += job.transfers.size() + job.image_ssd_slabs.size();
        // Superseded while it was written (begin_ssd_write takes only retained ones).
        if (snapshot.superseded) { ++kind.superseded_unread; }
    }
    io_unpin_snapshot(job.snapshot.index);
    job = SsdJob{};
}

void PrefixCacheIndex::pack_ssd_write(SsdJob& job, std::uint32_t stream_slabs) {
    Snapshot& snapshot       = require(job.snapshot);
    const std::uint32_t kept = ssd_image_slab_count();
    if (!job.image || job.packed || snapshot.ssd != CopyState::Filling || stream_slabs == 0 || stream_slabs >= kept) {
        invariant("prefix cache packs an SSD write it cannot");
    }
    // The job's transfers are its blocks', its kept image slabs', then its tail's.
    const auto image_begin = static_cast<std::ptrdiff_t>(job.nodes.size());
    job.image_host_slabs.assign(snapshot.host_slabs.begin(), snapshot.host_slabs.begin() + kept);
    job.image_ssd_slabs.assign(snapshot.ssd_slabs.begin(), snapshot.ssd_slabs.begin() + stream_slabs);
    for (std::uint32_t k = stream_slabs; k < kept; ++k) { free_ssd_slab(snapshot.ssd_slabs[k]); }
    std::vector<std::uint32_t> slabs(snapshot.ssd_slabs.begin(), snapshot.ssd_slabs.begin() + stream_slabs);
    slabs.insert(slabs.end(), snapshot.ssd_slabs.begin() + kept, snapshot.ssd_slabs.end());
    snapshot.ssd_slabs  = std::move(slabs);
    snapshot.ssd_packed = true;
    job.transfers.erase(job.transfers.begin() + image_begin, job.transfers.begin() + image_begin + kept);
    job.packed = true;
}

void PrefixCacheIndex::abort_ssd_write(SsdJob& job) {
    for (const NodeRef ref : job.nodes) {
        Node& node = require(ref);
        if (node.ssd != CopyState::Filling) { invariant("prefix cache SSD write not open"); }
        free_ssd_slab(node.ssd_slab);
        node.ssd_slab = kNoId;
        node.ssd      = CopyState::Absent;
        refresh_node(ref.index);
    }
    if (job.image) {
        Snapshot& snapshot = require(job.snapshot);
        if (snapshot.ssd != CopyState::Filling) { invariant("prefix cache SSD image write not open"); }
        for (const std::uint32_t slab : snapshot.ssd_slabs) { free_ssd_slab(slab); }
        snapshot.ssd_slabs.clear();
        snapshot.ssd        = CopyState::Absent;
        snapshot.ssd_packed = false;
    }
    io_unpin_snapshot(job.snapshot.index);
    job = SsdJob{};
}

SsdPromotionStart PrefixCacheIndex::begin_ssd_promotion(SnapshotRef ref) {
    const Snapshot& snapshot = require(ref);
    // What the source lacks above the SSD: path blocks with neither a Device nor a Host copy, and an
    // image (with its tail) that is not on the Device or the Host.
    std::vector<std::uint32_t> nodes;
    bool waiting = false;
    for (std::uint32_t current = snapshot.anchor; current != kNoId; current = nodes_[current].parent) {
        const Node& node = nodes_[current];
        if (node.device != CopyState::Absent || node.host == CopyState::Resident) { continue; }
        if (node.host == CopyState::Filling) {
            waiting = true;
        } else if (node.ssd == CopyState::Resident) {
            nodes.push_back(current);
        } else {
            invariant("prefix cache path block has no copy");
        }
    }
    const bool tail_needed = snapshot.tail_len != 0 && snapshot.tail_device_copy != CopyState::Resident;
    bool image             = false;
    if (snapshot.host == CopyState::Filling && (snapshot.device_slot == kNoId || tail_needed)) {
        waiting = true;
    } else if (snapshot.host == CopyState::Absent && (snapshot.device_slot == kNoId || tail_needed)) {
        if (snapshot.ssd != CopyState::Resident) { invariant("prefix cache snapshot has no image copy"); }
        image = true;
    }
    if (nodes.empty() && !image) {
        return SsdPromotionStart{.status = waiting ? SsdPromotionStatus::Waiting : SsdPromotionStatus::Resident,
                                 .job    = std::nullopt};
    }
    std::reverse(nodes.begin(), nodes.end()); // path order
    // Pinned first, so making room never takes a Host copy of the source's own path.
    io_pin_snapshot(ref.index);
    const std::uint32_t image_slabs = image ? snapshot_slab_count(snapshot) : 0U;
    std::vector<std::uint32_t> slabs;
    if (!allocate_slabs(static_cast<std::uint32_t>(nodes.size()) + image_slabs, slabs, ref.index)) {
        io_unpin_snapshot(ref.index);
        return SsdPromotionStart{.status = SsdPromotionStatus::NoRoom, .job = std::nullopt};
    }
    SsdJob job;
    job.snapshot     = ref;
    job.image        = image;
    std::size_t next = 0;
    for (const std::uint32_t node : nodes) {
        Node& entry = nodes_[node];
        set_node_slab(node, slabs[next]);
        entry.host = CopyState::Filling;
        job.nodes.push_back(ref_of_node(node));
        job.transfers.push_back(SsdTransfer{.host_slab = slabs[next], .ssd_slab = entry.ssd_slab});
        refresh_node(node);
        ++next;
    }
    if (image) {
        Snapshot& promoted = snapshots_[ref.index];
        promoted.host_slabs.assign(slabs.begin() + static_cast<std::ptrdiff_t>(next), slabs.end());
        promoted.host = CopyState::Filling;
        if (promoted.ssd_packed) {
            // The stream decodes into the kept image slabs; the tail is a slab copy.
            const bool tail           = config_.host_blocks && promoted.tail_len != 0;
            const std::size_t stream  = promoted.ssd_slabs.size() - (tail ? 1U : 0U);
            job.packed                = true;
            job.image_ssd_slabs.assign(promoted.ssd_slabs.begin(), promoted.ssd_slabs.begin() + static_cast<std::ptrdiff_t>(stream));
            job.image_host_slabs.assign(promoted.host_slabs.begin(), promoted.host_slabs.begin() + ssd_image_slab_count());
            if (tail) {
                job.transfers.push_back(SsdTransfer{.host_slab = promoted.host_slabs[config_.image_slabs],
                                                    .ssd_slab  = promoted.ssd_slabs.back()});
            }
        } else {
            for (std::size_t k = 0; k < promoted.ssd_slabs.size(); ++k) {
                job.transfers.push_back(SsdTransfer{.host_slab = promoted.host_slabs[ssd_to_host_position(k)],
                                                    .ssd_slab  = promoted.ssd_slabs[k]});
            }
        }
        for (std::uint32_t k = ssd_image_slab_count(); k < config_.image_slabs; ++k) {
            job.zero_host_slabs.push_back(promoted.host_slabs[k]);
        }
        refresh_tail(ref.index);
    }
    return SsdPromotionStart{.status = waiting ? SsdPromotionStatus::Waiting : SsdPromotionStatus::Started,
                             .job    = std::move(job)};
}

void PrefixCacheIndex::complete_ssd_promotion(SsdJob& job) {
    for (const NodeRef ref : job.nodes) {
        Node& node = require(ref);
        if (node.host != CopyState::Filling) { invariant("prefix cache SSD promotion not open"); }
        node.host = CopyState::Resident;
        refresh_node(ref.index);
    }
    if (job.image) {
        Snapshot& snapshot = require(job.snapshot);
        if (snapshot.host != CopyState::Filling) { invariant("prefix cache SSD image promotion not open"); }
        snapshot.host = CopyState::Resident;
        refresh_tail(job.snapshot.index);
        SsdKindStats& kind = kind_stats(snapshot);
        ++kind.reads;
        if (!snapshot.ssd_read) { ++kind.first_reads; }
        snapshot.ssd_read = true;
    }
    io_unpin_snapshot(job.snapshot.index);
    job = SsdJob{};
}

void PrefixCacheIndex::cancel_ssd_promotion(SsdJob& job) {
    release_ssd_promotion(job);
    job = SsdJob{};
}

void PrefixCacheIndex::abort_ssd_promotion(SsdJob& job) {
    release_ssd_promotion(job);
    // The SSD copies read may be damaged: they go, and an entry left with no copy goes with its
    // subtree (§9.4). The snapshot first, then the blocks deepest first. An entry another transfer
    // pins keeps its copy (its loss would leave that transfer's source without one).
    if (job.image && valid(job.snapshot)) {
        Snapshot& snapshot = snapshots_[job.snapshot.index];
        if (snapshot.ssd == CopyState::Resident && !held(snapshot)) {
            drop_snapshot_ssd_copy(job.snapshot.index);
            if (!snapshot_valid(snapshot)) { remove_snapshot(job.snapshot.index); }
        }
    }
    for (auto it = job.nodes.rbegin(); it != job.nodes.rend(); ++it) {
        if (!valid(*it) || held(nodes_[it->index]) || nodes_[it->index].ssd != CopyState::Resident) { continue; }
        drop_node_ssd_copy(it->index);
        const Node& node = nodes_[it->index];
        if (node.device == CopyState::Absent && node.host == CopyState::Absent) { remove_subtree(it->index); }
    }
    job = SsdJob{};
}

void PrefixCacheIndex::release_ssd_promotion(SsdJob& job) {
    for (const NodeRef ref : job.nodes) {
        Node& node = require(ref);
        if (node.host != CopyState::Filling) { invariant("prefix cache SSD promotion not open"); }
        free_slab(node.host_slab);
        set_node_slab(ref.index, kNoId);
        node.host = CopyState::Absent;
        refresh_node(ref.index);
    }
    if (job.image) {
        Snapshot& snapshot = require(job.snapshot);
        if (snapshot.host != CopyState::Filling) { invariant("prefix cache SSD image promotion not open"); }
        for (const std::uint32_t slab : snapshot.host_slabs) { free_slab(slab); }
        snapshot.host_slabs.clear();
        snapshot.host = CopyState::Absent;
        refresh_tail(job.snapshot.index);
    }
    io_unpin_snapshot(job.snapshot.index);
}

} // namespace infernix::runtime::prefix_cache
