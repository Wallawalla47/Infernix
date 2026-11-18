// PrefixCacheIndex's persistence (prefix_index.h): what a saved cache file holds, and the blocks and
// snapshots restored from one, with the Host restore plan.

#include "runtime/prefix_cache/prefix_index_internal.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

namespace infernix::runtime::prefix_cache {

void PrefixCacheIndex::collect_persistable(std::vector<NodeRef>& nodes, std::vector<SnapshotRef>& snapshots,
                                           bool ssd) const {
    nodes.clear();
    snapshots.clear();
    std::vector<bool> keep(nodes_.size(), false);
    // With `ssd` (a kept SSD tier) an SSD copy counts as a Host copy does.
    const auto saved = [ssd](CopyState host, CopyState on_ssd) {
        return host == CopyState::Resident || (ssd && on_ssd == CopyState::Resident);
    };
    for (std::uint32_t index = 0; index < snapshots_.size(); ++index) {
        const Snapshot& snapshot = snapshots_[index];
        // Superseded snapshots are saved too: a retry, a regenerate or a re-sent prompt resumes
        // from the tap it resumed from before, which its own endpoint superseded (issue #59).
        if (!snapshot.occupied || !saved(snapshot.host, snapshot.ssd)) { continue; }
        bool backed = true;
        for (std::uint32_t current = snapshot.anchor; current != kNoId && backed;
             current               = nodes_[current].parent) {
            backed = saved(nodes_[current].host, nodes_[current].ssd);
        }
        if (!backed) { continue; }
        for (std::uint32_t current = snapshot.anchor; current != kNoId && !keep[current];
             current               = nodes_[current].parent) {
            keep[current] = true;
        }
        snapshots.push_back(ref_of_snapshot(index));
    }
    // Retained first, then superseded oldest supersession first: a restore in file order gives
    // the superseded ones increasing ticks, so they are evicted in the order they were saved.
    std::stable_sort(snapshots.begin(), snapshots.end(), [&](SnapshotRef left, SnapshotRef right) {
        const Snapshot& a = snapshots_[left.index];
        const Snapshot& b = snapshots_[right.index];
        if (a.superseded != b.superseded) { return !a.superseded; }
        return a.superseded && a.superseded_tick < b.superseded_tick;
    });
    for (std::uint32_t index = 0; index < nodes_.size(); ++index) {
        if (keep[index]) { nodes.push_back(ref_of_node(index)); }
    }
    std::stable_sort(nodes.begin(), nodes.end(), [&](NodeRef left, NodeRef right) {
        return nodes_[left.index].depth < nodes_[right.index].depth;
    });
}

bool PrefixCacheIndex::take_free_slabs(std::uint32_t count, std::vector<std::uint32_t>& out) {
    out.clear();
    if (free_slabs_.size() < count) { return false; }
    for (std::uint32_t taken = 0; taken < count; ++taken) {
        std::pop_heap(free_slabs_.begin(), free_slabs_.end(), std::greater<>{});
        out.push_back(free_slabs_.back());
        free_slabs_.pop_back();
    }
    return true;
}

bool PrefixCacheIndex::claim_ssd_slabs(std::span<const std::uint32_t> slabs) {
    if (ssd_claimed_.size() != config_.ssd_slabs) { ssd_claimed_.assign(config_.ssd_slabs, false); }
    for (std::size_t k = 0; k < slabs.size(); ++k) {
        const std::uint32_t slab = slabs[k];
        if (slab >= config_.ssd_slabs || ssd_claimed_[slab] ||
            std::find(slabs.begin(), slabs.begin() + static_cast<std::ptrdiff_t>(k), slab) !=
                slabs.begin() + static_cast<std::ptrdiff_t>(k)) {
            return false;
        }
    }
    for (const std::uint32_t slab : slabs) { ssd_claimed_[slab] = true; }
    return true;
}

void PrefixCacheIndex::finish_ssd_restore() {
    if (ssd_claimed_.empty()) { return; }
    // The free list is every SSD slab no restored copy holds, lowest first.
    free_ssd_slabs_.clear();
    for (std::uint32_t slab = 0; slab < config_.ssd_slabs; ++slab) {
        if (!ssd_claimed_[slab]) { free_ssd_slabs_.push_back(slab); }
    }
    std::make_heap(free_ssd_slabs_.begin(), free_ssd_slabs_.end(), std::greater<>{});
    ssd_claimed_.clear();
}

std::optional<RestoredBlock> PrefixCacheIndex::restore_block(NodeRef parent, std::uint64_t lookup_hash,
                                                             std::span<const TokenId> tokens, std::uint64_t extra,
                                                             bool host, std::uint32_t ssd_slab) {
    if (!config_.host_blocks || tokens.size() != kBlockTokens || (!host && ssd_slab == kNoId) ||
        (ssd_slab != kNoId && config_.ssd_slabs == 0)) {
        throw std::invalid_argument("prefix cache restore requires a Host tier, a full block and a copy");
    }
    const std::uint32_t parent_index = parent.valid() ? parent.index : kNoId;
    if (parent.valid() && require(parent).host != CopyState::Resident && nodes_[parent_index].ssd != CopyState::Resident) {
        invariant("prefix cache restored block has no restored parent");
    }
    if (find_child(parent, lookup_hash, tokens, extra)) { return std::nullopt; }
    std::vector<std::uint32_t> slab;
    if (free_nodes_.empty() || (host && !take_free_slabs(1, slab))) { return std::nullopt; }
    if (ssd_slab != kNoId && !claim_ssd_slabs(std::span<const std::uint32_t>(&ssd_slab, 1))) {
        for (const std::uint32_t taken : slab) { free_slab(taken); }
        return std::nullopt;
    }
    const std::uint32_t index = free_nodes_.back();
    free_nodes_.pop_back();
    Node& node    = nodes_[index];
    node.occupied = true;
    node.parent   = parent_index;
    node.depth    = parent.valid() ? nodes_[parent_index].depth + 1U : 0U;
    node.hash     = lookup_hash;
    node.extra    = extra;
    std::copy(tokens.begin(), tokens.end(), node.tokens.begin());
    node.children.clear();
    node.snapshots.clear();
    node.device_id      = kNoId;
    node.host_slab      = kNoId;
    set_node_slab(index, host ? slab.front() : kNoId);
    node.host           = host ? CopyState::Resident : CopyState::Absent;
    node.ssd_slab       = ssd_slab;
    node.ssd            = ssd_slab != kNoId ? CopyState::Resident : CopyState::Absent;
    node.pins           = 0;
    node.device_pins    = 0;
    node.transfer_pins  = 0;
    node.io_pins        = 0;
    node.live_below     = 0;
    node.retained_below = 0;
    node.warm_below     = 0;
    node.hold           = kNoHold;
    if (parent.valid()) {
        nodes_[parent_index].children.push_back(index);
    } else {
        root_children_.push_back(index);
    }
    children_.emplace(child_key(parent_index, lookup_hash), index);
    ++node_count_;
    refresh_node(index);
    return RestoredBlock{ref_of_node(index), host ? slab.front() : kNoId};
}

std::optional<SnapshotRef> PrefixCacheIndex::restore_snapshot(NodeRef anchor, std::uint32_t frontier,
                                                              std::span<const TokenId> tail, SnapshotKind kind,
                                                              std::uint32_t hits, bool superseded, bool host,
                                                              std::span<const std::uint32_t> ssd_slabs,
                                                              bool ssd_packed) {
    if (!config_.host_blocks || tail.size() >= kBlockTokens || (!host && ssd_slabs.empty()) ||
        (!ssd_slabs.empty() && config_.ssd_slabs == 0)) {
        throw std::invalid_argument("prefix cache restored snapshot is inconsistent");
    }
    const std::uint32_t anchor_index = anchor.valid() ? anchor.index : kNoId;
    const std::uint32_t base = anchor.valid() ? (require(anchor).depth + 1U) * kBlockTokens : 0U;
    if (frontier == 0 || frontier != base + static_cast<std::uint32_t>(tail.size()) ||
        (anchor.valid() && nodes_[anchor_index].host != CopyState::Resident &&
         nodes_[anchor_index].ssd != CopyState::Resident)) {
        throw std::invalid_argument("prefix cache restored snapshot does not match its anchor");
    }
    // An SSD copy holds the kept image slabs raw (or fewer, packed) and then the tail's slab.
    const std::uint32_t ssd_expected = ssd_image_slab_count() + (tail.empty() ? 0U : 1U);
    if (!ssd_slabs.empty() && (ssd_packed ? ssd_slabs.size() >= ssd_expected : ssd_slabs.size() != ssd_expected)) {
        return std::nullopt;
    }
    std::vector<std::uint32_t>& anchored =
        anchor.valid() ? nodes_[anchor_index].snapshots : root_snapshots_;
    for (const std::uint32_t existing : anchored) {
        const Snapshot& other = snapshots_[existing];
        if (other.tail_len == tail.size() &&
            std::equal(tail.begin(), tail.end(), other.tail.begin())) {
            return std::nullopt;
        }
    }
    std::vector<std::uint32_t> slabs;
    const std::uint32_t count = config_.image_slabs + (tail.empty() ? 0U : 1U);
    if (free_snapshots_.empty() || (host && !take_free_slabs(count, slabs))) {
        for (const std::uint32_t slab : slabs) { free_slab(slab); }
        return std::nullopt;
    }
    if (!ssd_slabs.empty() && !claim_ssd_slabs(ssd_slabs)) {
        for (const std::uint32_t slab : slabs) { free_slab(slab); }
        return std::nullopt;
    }
    const std::uint32_t index = free_snapshots_.back();
    free_snapshots_.pop_back();
    Snapshot& snapshot = snapshots_[index];
    snapshot.occupied  = true;
    snapshot.anchor    = anchor_index;
    snapshot.frontier  = frontier;
    snapshot.tail_len  = static_cast<std::uint32_t>(tail.size());
    std::copy(tail.begin(), tail.end(), snapshot.tail.begin());
    snapshot.tail_device      = kNoId;
    snapshot.tail_device_copy = CopyState::Absent;
    snapshot.device_slot      = kNoId;
    snapshot.host_slabs       = std::move(slabs);
    snapshot.host             = host ? CopyState::Resident : CopyState::Absent;
    snapshot.ssd_slabs.assign(ssd_slabs.begin(), ssd_slabs.end());
    snapshot.ssd              = ssd_slabs.empty() ? CopyState::Absent : CopyState::Resident;
    snapshot.ssd_read         = false;
    snapshot.ssd_packed       = !ssd_slabs.empty() && ssd_packed;
    snapshot.ssd_offered      = false;
    snapshot.ssd_declined     = false;
    snapshot.warm_counted     = false;
    snapshot.clean_counted    = false;
    snapshot.pins             = 0;
    snapshot.io_pins          = 0;
    snapshot.kind             = kind;
    snapshot.hits             = hits;
    snapshot.last_hit_tick    = ++tick_;
    snapshot.priority_base    = inflation_;
    snapshot.superseded       = superseded;
    snapshot.superseded_tick  = superseded ? ++tick_ : 0;
    snapshot.hold             = kNoHold;
    ++snapshot_epoch_;
    anchored.push_back(index);
    ++snapshot_count_;
    if (anchor.valid()) {
        adjust_live(anchor_index, +1);
        if (!superseded) { adjust_retained(anchor_index, +1); }
    }
    link_snapshot(index);
    std::vector<std::uint32_t> dependents;
    collect_retained_dependents(index, dependents);
    revalue_around(index, dependents);
    if (!superseded) { revalue_sibling(anchor_index, index, 1); }
    refresh_tail(index);
    return ref_of_snapshot(index);
}

HostRestorePlan plan_host_restore(std::span<const std::int32_t> block_parents,
                                  std::span<const SavedSnapshotShape> snapshots,
                                  std::uint64_t host_slabs, const PrefixIndexConfig& config,
                                  std::span<const bool> block_host) {
    const std::size_t block_count = block_parents.size();
    if (!block_host.empty() && block_host.size() != block_count) {
        throw std::invalid_argument("saved prefix cache block tiers do not match its blocks");
    }
    // A block or snapshot the file holds no Host bytes of (only an SSD copy) takes no Host slab.
    const auto has_host = [&](std::size_t block) { return block_host.empty() || block_host[block]; };
    HostRestorePlan plan{.blocks    = std::vector<bool>(block_count, false),
                         .snapshots = std::vector<bool>(snapshots.size(), false)};
    // Host-carrying blocks on each block's path, itself included.
    std::vector<std::uint32_t> path_blocks(block_count);
    std::uint64_t total = 0;
    for (std::size_t index = 0; index < block_count; ++index) {
        const std::int32_t parent = block_parents[index];
        if (parent >= static_cast<std::int32_t>(index) || parent < -1) {
            throw std::invalid_argument("saved prefix cache blocks are not parents first");
        }
        const std::uint32_t own = has_host(index) ? 1U : 0U;
        path_blocks[index]      = parent < 0 ? own : path_blocks[static_cast<std::size_t>(parent)] + own;
        total += own;
    }
    for (const SavedSnapshotShape& snapshot : snapshots) {
        if (snapshot.anchor >= static_cast<std::int32_t>(block_count) || snapshot.anchor < -1 ||
            (snapshot.host && snapshot.slabs < config.image_slabs)) {
            throw std::invalid_argument("saved prefix cache snapshot is inconsistent");
        }
        if (snapshot.host) { total += snapshot.slabs; }
    }
    if (total <= host_slabs) {
        for (std::size_t index = 0; index < block_count; ++index) { plan.blocks[index] = has_host(index); }
        for (std::size_t index = 0; index < snapshots.size(); ++index) { plan.snapshots[index] = snapshots[index].host; }
        plan.slabs = total;
        return plan;
    }

    std::vector<double> density(snapshots.size());
    for (std::size_t index = 0; index < snapshots.size(); ++index) {
        const SavedSnapshotShape& snapshot = snapshots[index];
        const std::uint64_t path =
            snapshot.anchor < 0 ? 0U : path_blocks[static_cast<std::size_t>(snapshot.anchor)];
        const double saved = config.cost.prefill_seconds(0, snapshot.frontier) -
                             config.cost.restore_seconds(config.image_bytes);
        const double size =
            static_cast<double>(config.image_bytes) +
            static_cast<double>(snapshot.slabs > config.image_slabs ? config.block_bytes : 0U) +
            static_cast<double>(path) * static_cast<double>(config.block_bytes);
        density[index] =
            (1.0 + static_cast<double>(snapshot.hits)) * std::max(0.0, saved) / std::max(size, 1.0);
    }
    std::vector<std::uint32_t> order(snapshots.size());
    for (std::uint32_t index = 0; index < order.size(); ++index) { order[index] = index; }
    std::stable_sort(order.begin(), order.end(), [&](std::uint32_t left, std::uint32_t right) {
        // A running cache evicts superseded snapshots first: they are restored last.
        if (snapshots[left].superseded != snapshots[right].superseded) {
            return !snapshots[left].superseded;
        }
        if (density[left] != density[right]) { return density[left] > density[right]; }
        return snapshots[left].frontier > snapshots[right].frontier;
    });

    std::uint64_t left = host_slabs;
    for (const std::uint32_t index : order) {
        const SavedSnapshotShape& snapshot = snapshots[index];
        if (!snapshot.host) { continue; }
        std::uint64_t need = snapshot.slabs;
        for (std::int32_t block = snapshot.anchor; block >= 0; block = block_parents[static_cast<std::size_t>(block)]) {
            if (!plan.blocks[static_cast<std::size_t>(block)] && has_host(static_cast<std::size_t>(block))) { ++need; }
        }
        if (need > left) { continue; }
        for (std::int32_t block = snapshot.anchor; block >= 0; block = block_parents[static_cast<std::size_t>(block)]) {
            if (has_host(static_cast<std::size_t>(block))) { plan.blocks[static_cast<std::size_t>(block)] = true; }
        }
        plan.snapshots[index] = true;
        left -= need;
    }
    plan.slabs = host_slabs - left;
    return plan;
}

} // namespace infernix::runtime::prefix_cache
