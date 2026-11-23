#include "runtime/prefix_cache/prefix_index.h"

#include "runtime/prefix_cache/prefix_index_internal.h"

#include <algorithm>
#include <exception>
#include <stdexcept>

namespace infernix::runtime::prefix_cache {

PrefixCacheIndex::PrefixCacheIndex(const PrefixIndexConfig& config, PrefixIndexBackend& backend)
    : config_(config), backend_(&backend) {
    if (config.max_nodes == 0 || config.max_snapshots == 0) {
        throw std::invalid_argument("prefix cache index requires node and snapshot capacity");
    }
    if (config.max_nodes > kNoId / 2 || config.max_snapshots > kNoId / 2) {
        throw std::invalid_argument("prefix cache index capacity is not representable");
    }
    if (config.host_slabs != 0 && config.image_slabs == 0) {
        throw std::invalid_argument("prefix cache host tier requires a positive image slab count");
    }
    if (config.ssd_slabs != 0 && (config.host_slabs == 0 || !config.host_blocks)) {
        throw std::invalid_argument("prefix cache SSD tier requires a Host tier holding blocks");
    }
    if (config.ssd_image_slabs > config.image_slabs) {
        throw std::invalid_argument("prefix cache SSD copies keep more slabs than an image has");
    }
    const std::uint32_t ssd_image = config.ssd_image_slabs == 0 ? config.image_slabs : config.ssd_image_slabs;
    if (config.ssd_slabs != 0 && config.ssd_slabs < ssd_image + 2U) {
        throw std::invalid_argument("prefix cache SSD tier holds less than one snapshot and its block");
    }
    nodes_.resize(config.max_nodes);
    free_nodes_.resize(config.max_nodes);
    for (std::uint32_t i = 0; i < config.max_nodes; ++i) {
        free_nodes_[i] = config.max_nodes - 1U - i;
    }
    snapshots_.resize(config.max_snapshots);
    free_snapshots_.resize(config.max_snapshots);
    for (std::uint32_t i = 0; i < config.max_snapshots; ++i) {
        free_snapshots_[i] = config.max_snapshots - 1U - i;
    }
    lru_links_.resize(static_cast<std::size_t>(config.max_nodes) + config.max_snapshots);
    // A min-heap: allocation always takes the lowest free slab, so a batch of single-slab
    // allocations comes out ascending and mostly consecutive and its copies coalesce into runs.
    free_slabs_.resize(config.host_slabs);
    for (std::uint32_t i = 0; i < config.host_slabs; ++i) { free_slabs_[i] = i; }
    host_limit_ = config.host_slabs;
    ceiling_    = config.host_slabs;
    slab_reserved_.assign(config.host_slabs, false);
    slab_moving_.assign(config.host_slabs, false);
    node_of_slab_.assign(config.host_slabs, kNoId);
    // Ascending ids already form a min-heap.
    free_ssd_slabs_.resize(config.ssd_slabs);
    for (std::uint32_t i = 0; i < config.ssd_slabs; ++i) { free_ssd_slabs_[i] = i; }
    slot_state_.assign(config.device_snapshot_slots, SlotState::Free);
    free_slots_ = config.device_snapshot_slots;
    slot_owner_.assign(config.device_snapshot_slots, kNoId);
    children_.reserve(config.max_nodes);
}

void PrefixCacheIndex::set_cost(const CacheCostModel& cost) {
    config_.cost = cost;
    for (std::uint32_t index = 0; index < snapshots_.size(); ++index) {
        if (snapshots_[index].occupied) { update_priority(index); }
    }
}

// ---- lookup ------------------------------------------------------------------------------------

std::optional<NodeRef> PrefixCacheIndex::find_child(NodeRef parent, std::uint64_t lookup_hash,
                                                    std::span<const TokenId> block_tokens,
                                                    std::uint64_t extra) const {
    if (block_tokens.size() != kBlockTokens) {
        throw std::invalid_argument("prefix cache block must contain 64 tokens");
    }
    const std::uint32_t parent_index = parent.valid() ? parent.index : kNoId;
    if (parent.valid()) { (void)require(parent); }
    const auto [begin, end] = children_.equal_range(child_key(parent_index, lookup_hash));
    for (auto it = begin; it != end; ++it) {
        const Node& node = nodes_[it->second];
        if (node.parent == parent_index && node.hash == lookup_hash && node.extra == extra &&
            std::equal(block_tokens.begin(), block_tokens.end(), node.tokens.begin())) {
            return ref_of_node(it->second);
        }
    }
    return std::nullopt;
}

bool PrefixCacheIndex::tail_matches(const Snapshot& snapshot,
                                    std::span<const TokenId> prompt) const noexcept {
    const std::uint32_t begin = snapshot.frontier - snapshot.tail_len;
    if (prompt.size() < snapshot.frontier) { return false; }
    return std::equal(snapshot.tail.begin(), snapshot.tail.begin() + snapshot.tail_len,
                      prompt.begin() + begin);
}

MatchResult PrefixCacheIndex::match(std::span<const TokenId> tokens,
                                    std::span<const std::uint64_t> block_hashes,
                                    std::span<const std::uint64_t> block_extras,
                                    std::uint32_t prompt_tokens) const {
    if (tokens.size() < prompt_tokens) {
        throw std::invalid_argument("prefix cache match prompt exceeds its tokens");
    }
    MatchResult result;
    if (prompt_tokens < 2) { return result; }
    const std::uint32_t max_frontier = prompt_tokens - 1U;
    const std::uint32_t max_blocks   = max_frontier / kBlockTokens;

    std::vector<std::uint32_t> host_only;
    std::vector<std::uint32_t> filling;
    std::vector<std::uint32_t> ssd_only;
    // Of ssd_only, the blocks no promotion is reading yet: only these still cost an SSD read.
    std::vector<std::uint32_t> ssd_unread;
    NodeRef parent;
    for (std::uint32_t block = 0; block < max_blocks && block < block_hashes.size(); ++block) {
        const std::uint64_t extra = block_extras.empty() ? 0ULL : block_extras[block];
        const auto child          = find_child(
            parent, block_hashes[block],
            tokens.subspan(static_cast<std::size_t>(block) * kBlockTokens, kBlockTokens), extra);
        if (!child) { break; }
        const Node& node                    = nodes_[child->index];
        const std::uint32_t prior_host_only = host_only.empty() ? 0U : host_only.back();
        const std::uint32_t prior_filling   = filling.empty() ? 0U : filling.back();
        const std::uint32_t prior_ssd_only  = ssd_only.empty() ? 0U : ssd_only.back();
        host_only.push_back(prior_host_only + (node.device == CopyState::Absent ? 1U : 0U));
        filling.push_back(prior_filling + (node.device == CopyState::Filling ? 1U : 0U));
        // Neither a Device nor a Host copy: on the SSD tier, or being promoted from it.
        ssd_only.push_back(prior_ssd_only +
                           (node.device == CopyState::Absent && node.host != CopyState::Resident ? 1U : 0U));
        ssd_unread.push_back((ssd_unread.empty() ? 0U : ssd_unread.back()) +
                             (node.device == CopyState::Absent && node.host == CopyState::Absent ? 1U : 0U));
        result.path.push_back(*child);
        parent = *child;
    }

    auto consider = [&](std::uint32_t snapshot_index, std::uint32_t path_blocks) {
        const Snapshot& snapshot = snapshots_[snapshot_index];
        if (snapshot.frontier > max_frontier || !snapshot_valid(snapshot) ||
            !tail_matches(snapshot, tokens)) {
            return;
        }
        MatchCandidate candidate;
        candidate.snapshot         = ref_of_snapshot(snapshot_index);
        candidate.frontier         = snapshot.frontier;
        candidate.path_blocks      = path_blocks;
        candidate.host_only_blocks = path_blocks == 0 ? 0U : host_only[path_blocks - 1U];
        candidate.filling_blocks   = path_blocks == 0 ? 0U : filling[path_blocks - 1U];
        candidate.tail             = snapshot.tail_len != 0;
        candidate.tail_on_device   = snapshot.tail_device_copy == CopyState::Resident;
        candidate.image_on_device  = snapshot.device_slot != kNoId;
        candidate.restore_bytes =
            static_cast<std::uint64_t>(candidate.host_only_blocks) * config_.block_bytes +
            (candidate.image_on_device ? 0ULL : config_.image_bytes) +
            (candidate.tail && !candidate.tail_on_device ? config_.block_bytes : 0ULL);
        candidate.ssd_only_blocks   = path_blocks == 0 ? 0U : ssd_only[path_blocks - 1U];
        candidate.image_on_ssd_only = !candidate.image_on_device && snapshot.host != CopyState::Resident;
        candidate.tail_on_ssd_only =
            candidate.tail && !candidate.tail_on_device && snapshot.host != CopyState::Resident;
        const std::uint64_t image_ssd_bytes =
            snapshot.ssd_packed
                ? static_cast<std::uint64_t>(snapshot.ssd_slabs.size() - (candidate.tail ? 1U : 0U)) * config_.block_bytes
                : config_.image_bytes * ssd_image_slab_count() / config_.image_slabs;
        candidate.ssd_bytes =
            static_cast<std::uint64_t>(candidate.ssd_only_blocks) * config_.block_bytes +
            (candidate.image_on_ssd_only ? image_ssd_bytes : 0ULL) +
            (candidate.tail_on_ssd_only ? config_.block_bytes : 0ULL);
        // What a promotion is reading already (a Filling Host copy) is not read again: charging it in
        // full made a source whose read was nearly done look as far away as an unread one.
        const bool image_unread = snapshot.host == CopyState::Absent;
        candidate.ssd_unread_bytes =
            static_cast<std::uint64_t>(path_blocks == 0 ? 0U : ssd_unread[path_blocks - 1U]) * config_.block_bytes +
            (candidate.image_on_ssd_only && image_unread ? image_ssd_bytes : 0ULL) +
            (candidate.tail_on_ssd_only && image_unread ? config_.block_bytes : 0ULL);
        result.candidates.push_back(candidate);
    };
    for (const std::uint32_t snapshot : root_snapshots_) { consider(snapshot, 0); }
    for (std::size_t depth = 0; depth < result.path.size(); ++depth) {
        for (const std::uint32_t snapshot : nodes_[result.path[depth].index].snapshots) {
            consider(snapshot, static_cast<std::uint32_t>(depth + 1));
        }
    }
    std::stable_sort(result.candidates.begin(), result.candidates.end(),
                     [](const MatchCandidate& left, const MatchCandidate& right) {
                         return left.frontier > right.frontier;
                     });
    return result;
}

AdmissionChoice PrefixCacheIndex::choose(const MatchResult& match,
                                         std::uint32_t prompt_tokens) const {
    AdmissionChoice best;
    best.predicted_seconds     = config_.cost.prefill_seconds(0, prompt_tokens);
    std::uint64_t best_restore = 0;
    for (std::size_t i = 0; i < match.candidates.size(); ++i) {
        const MatchCandidate& candidate = match.candidates[i];
        // A candidate on the SSD tier is read into the Host tier first (§5.7).
        const double seconds =
            config_.cost.ssd_seconds(candidate.ssd_unread_bytes) +
            config_.cost.restore_seconds(candidate.restore_bytes) +
            config_.cost.prefill_seconds(candidate.frontier, prompt_tokens - candidate.frontier);
        const bool better = seconds < best.predicted_seconds ||
                            (seconds == best.predicted_seconds && best.candidate &&
                             candidate.restore_bytes < best_restore);
        if (better) {
            best.candidate         = i;
            best.predicted_seconds = seconds;
            best_restore           = candidate.restore_bytes;
        }
    }
    return best;
}

void PrefixCacheIndex::note_hit(SnapshotRef snapshot, bool counted) {
    Snapshot& entry = require(snapshot);
    ++counters_.snapshot_hits;
    entry.last_hit_tick = ++tick_;
    if (counted) {
        ++entry.hits;
        entry.priority_base = inflation_;
    }
    if (!entry.superseded) {
        update_priority(snapshot.index);
        return;
    }
    // Resumed from again (a retry, or a second branch): retained until this lineage moves on.
    retain_again(snapshot.index);
}

void PrefixCacheIndex::retain_again(std::uint32_t index) {
    Snapshot& entry = snapshots_[index];
    // Collected first: an allocation that fails leaves the snapshot superseded and consistent.
    std::vector<std::uint32_t> dependents;
    collect_retained_dependents(index, dependents);
    entry.superseded      = false;
    entry.superseded_tick = 0;
    adjust_retained(entry.anchor, +1);
    revalue_around(index, dependents);
    revalue_sibling(entry.anchor, index, 1);
}

void PrefixCacheIndex::revalue_sibling(std::uint32_t anchor, std::uint32_t skip, std::uint32_t own) {
    // Up to the first block another retained snapshot shares: below it the count is this snapshot's own.
    std::uint32_t from = kNoId;
    std::uint32_t node = anchor;
    while (node != kNoId && nodes_[node].retained_below == own) {
        from = node;
        node = nodes_[node].parent;
    }
    // Two or more others: none of them had these blocks to itself, before or after.
    if (node == kNoId || nodes_[node].retained_below != own + 1U) { return; }
    // Down to that one snapshot: every block on the way has it alone below (besides `skip`'s branch,
    // `from`, at the fork).
    for (std::uint32_t exclude = from;;) {
        for (const std::uint32_t snapshot : nodes_[node].snapshots) {
            if (snapshot != skip && !snapshots_[snapshot].superseded) {
                update_priority(snapshot);
                return;
            }
        }
        std::uint32_t next = kNoId;
        for (const std::uint32_t child : nodes_[node].children) {
            if (child != exclude && nodes_[child].retained_below != 0) {
                next = child;
                break;
            }
        }
        if (next == kNoId) { return; }
        exclude = kNoId;
        node    = next;
    }
}

bool PrefixCacheIndex::shared_below(const Snapshot& snapshot) const noexcept {
    // The blocks right after the snapshot: children of its anchor whose tokens begin with its
    // tail. Two of them, or a branch further down the one chain they start, mean another
    // conversation has continued from this prefix differently.
    const std::vector<std::uint32_t>& first =
        snapshot.anchor == kNoId ? root_children_ : nodes_[snapshot.anchor].children;
    std::uint32_t chain = kNoId;
    for (const std::uint32_t child : first) {
        const auto tail = snapshot.tail.begin();
        if (!std::equal(tail, tail + snapshot.tail_len, nodes_[child].tokens.begin())) { continue; }
        if (chain != kNoId) { return true; }
        chain = child;
    }
    while (chain != kNoId) {
        const std::vector<std::uint32_t>& children = nodes_[chain].children;
        if (children.size() > 1) { return true; }
        chain = children.empty() ? kNoId : children.front();
    }
    return false;
}

// ---- queue holds -------------------------------------------------------------------------------

void PrefixCacheIndex::set_queue_holds(std::span<const SnapshotRef> holds) {
    std::vector<std::uint32_t> previous;
    previous.swap(holds_);
    for (const std::uint32_t index : previous) {
        if (index == kNoId) { continue; }
        snapshots_[index].hold = kNoHold;
        refresh_tail(index);
    }
    for (const SnapshotRef ref : holds) {
        if (holds_.size() == kMaxQueueHolds) { break; }
        if (!valid(ref) || snapshots_[ref.index].hold != kNoHold) { continue; }
        snapshots_[ref.index].hold = static_cast<std::uint8_t>(holds_.size());
        holds_.push_back(ref.index);
        refresh_tail(ref.index);
    }
    refresh_holds();
}

void PrefixCacheIndex::refresh_holds() {
    std::vector<std::uint32_t> previous;
    previous.swap(held_nodes_);
    for (const std::uint32_t index : previous) { nodes_[index].hold = kNoHold; }
    // Ranks in admission order: a node already marked lies on a lower rank's path, and so does
    // every node above it.
    for (std::uint32_t rank = 0; rank < holds_.size(); ++rank) {
        if (holds_[rank] == kNoId) { continue; }
        for (std::uint32_t current = snapshots_[holds_[rank]].anchor;
             current != kNoId && nodes_[current].hold == kNoHold; current = nodes_[current].parent) {
            nodes_[current].hold = static_cast<std::uint8_t>(rank);
            held_nodes_.push_back(current);
        }
    }
    for (const std::uint32_t index : previous) { refresh_node(index); }
    for (const std::uint32_t index : held_nodes_) { refresh_node(index); }
}

bool PrefixCacheIndex::leaves_path(const Snapshot& snapshot, std::uint32_t from,
                                   std::span<const TokenId> continuation) const noexcept {
    const std::uint32_t end = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(snapshot.frontier, std::uint64_t{from} + continuation.size()));
    const std::uint32_t anchored = snapshot.frontier - snapshot.tail_len;
    for (std::uint32_t p = std::max(from, anchored); p < end; ++p) {
        if (snapshot.tail[p - anchored] != continuation[p - from]) { return true; }
    }
    // The anchor chain, deepest block first, down to the block holding `from`.
    std::uint32_t node = snapshot.anchor;
    for (std::uint32_t begin = anchored; node != kNoId && begin > from;
         node = nodes_[node].parent) {
        begin -= kBlockTokens;
        for (std::uint32_t p = std::max(from, begin); p < std::min(end, begin + kBlockTokens); ++p) {
            if (nodes_[node].tokens[p - begin] != continuation[p - from]) { return true; }
        }
    }
    return false;
}

bool PrefixCacheIndex::supersede(SnapshotRef snapshot, std::span<const TokenId> continuation) {
    // Below a Boundary the endpoints may be other conversations': one conversation passing them
    // says nothing about theirs.
    if (!continuation.empty() && require(snapshot).kind != SnapshotKind::Boundary) {
        const Snapshot& source = require(snapshot);
        std::vector<std::uint32_t> passed;
        for (const std::uint32_t dependent : dependents_of(snapshot.index)) {
            const Snapshot& endpoint = snapshots_[dependent];
            if (endpoint.kind == SnapshotKind::Endpoint && endpoint.hits == 0 &&
                !endpoint.superseded && leaves_path(endpoint, source.frontier, continuation)) {
                passed.push_back(dependent);
            }
        }
        // Before `snapshot`, so they are evicted first.
        for (const std::uint32_t endpoint : passed) { (void)supersede(ref_of_snapshot(endpoint)); }
    }
    Snapshot& entry = require(snapshot);
    // A boundary serves the conversations that share it, so it is superseded only while the tree
    // shows no other conversation below it: until then it is just this lineage's snapshot.
    if (entry.superseded || (entry.kind == SnapshotKind::Boundary && shared_below(entry))) {
        return false;
    }
    std::vector<std::uint32_t> dependents;
    collect_retained_dependents(snapshot.index, dependents);
    if (entry.ssd == CopyState::Resident && !entry.ssd_read) { ++kind_stats(entry).superseded_unread; }
    entry.superseded      = true;
    entry.superseded_tick = ++tick_;
    adjust_retained(entry.anchor, -1);
    // Its retained dependents now value what lies between them and the retained ancestor above.
    revalue_around(snapshot.index, dependents);
    revalue_sibling(entry.anchor, snapshot.index, 0);
    return true;
}

void PrefixCacheIndex::unsupersede(SnapshotRef snapshot) {
    if (!require(snapshot).superseded) { return; }
    retain_again(snapshot.index);
}

// ---- pins --------------------------------------------------------------------------------------

void PrefixCacheIndex::acquire_path(std::span<const NodeRef> path) {
    std::uint32_t expected_parent = kNoId;
    for (const NodeRef ref : path) {
        const Node& node = require(ref);
        if (node.parent != expected_parent) { invariant("prefix cache path is not a root path"); }
        expected_parent = ref.index;
    }
    for (const NodeRef ref : path) {
        ++nodes_[ref.index].pins;
        ++nodes_[ref.index].device_pins;
        refresh_node(ref.index);
    }
}

void PrefixCacheIndex::release_path(std::span<const NodeRef> path) {
    std::uint32_t expected_parent = kNoId;
    for (const NodeRef ref : path) {
        const Node& node = require(ref);
        if (node.parent != expected_parent) { invariant("prefix cache path is not a root path"); }
        if (node.pins == 0 || node.device_pins <= node.transfer_pins) {
            invariant("prefix cache path released without a pin");
        }
        expected_parent = ref.index;
    }
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        --nodes_[it->index].pins;
        --nodes_[it->index].device_pins;
        refresh_node(it->index);
    }
}

void PrefixCacheIndex::pin_path_from(std::uint32_t node) {
    for (std::uint32_t current = node; current != kNoId; current = nodes_[current].parent) {
        ++nodes_[current].pins;
        refresh_node(current);
    }
}

void PrefixCacheIndex::unpin_path_from(std::uint32_t node, bool) {
    for (std::uint32_t current = node; current != kNoId; current = nodes_[current].parent) {
        if (nodes_[current].pins == 0) { invariant("prefix cache node unpinned without a pin"); }
        --nodes_[current].pins;
        refresh_node(current);
    }
}

void PrefixCacheIndex::pin_node(NodeRef node) {
    Node& entry = require(node);
    ++entry.device_pins;
    ++entry.transfer_pins;
    pin_path_from(node.index);
}

void PrefixCacheIndex::unpin_node(NodeRef node) {
    Node& entry = require(node);
    if (entry.transfer_pins == 0 || entry.device_pins == 0) {
        invariant("prefix cache node transfer unpinned without a pin");
    }
    --entry.transfer_pins;
    --entry.device_pins;
    unpin_path_from(node.index, true);
}

void PrefixCacheIndex::retain_node(NodeRef node) {
    (void)require(node);
    pin_path_from(node.index);
}

void PrefixCacheIndex::unretain_node(NodeRef node) {
    (void)require(node);
    unpin_path_from(node.index, true);
}

void PrefixCacheIndex::pin_snapshot(SnapshotRef snapshot) {
    Snapshot& entry = require(snapshot);
    ++entry.pins;
    // A pinned snapshot pins its anchor path, as a pinned node pins its ancestors: an unbacked
    // eviction removes a whole subtree, which must never hold a snapshot a transfer still reads.
    if (entry.anchor != kNoId) { pin_path_from(entry.anchor); }
    refresh_tail(snapshot.index);
}

void PrefixCacheIndex::unpin_snapshot(SnapshotRef snapshot) {
    Snapshot& entry = require(snapshot);
    if (entry.pins == 0) { invariant("prefix cache snapshot unpinned without a pin"); }
    --entry.pins;
    if (entry.anchor != kNoId) { unpin_path_from(entry.anchor, true); }
    refresh_tail(snapshot.index);
}

void PrefixCacheIndex::io_pin_snapshot(std::uint32_t index) {
    ++snapshots_[index].io_pins;
    for (std::uint32_t current = snapshots_[index].anchor; current != kNoId; current = nodes_[current].parent) {
        ++nodes_[current].io_pins;
        refresh_node(current);
    }
    refresh_tail(index);
}

void PrefixCacheIndex::io_unpin_snapshot(std::uint32_t index) {
    Snapshot& entry = snapshots_[index];
    if (entry.io_pins == 0) { invariant("prefix cache snapshot I/O-unpinned without an I/O pin"); }
    --entry.io_pins;
    for (std::uint32_t current = entry.anchor; current != kNoId; current = nodes_[current].parent) {
        if (nodes_[current].io_pins == 0) { invariant("prefix cache node I/O-unpinned without an I/O pin"); }
        --nodes_[current].io_pins;
        refresh_node(current);
    }
    refresh_tail(index);
}

// ---- membership maintenance ------------------------------------------------------------------

std::uint8_t PrefixCacheIndex::lru_list(bool backed, std::uint8_t hold) noexcept {
    const std::uint8_t kind = backed ? kBacked : kUnbacked;
    return hold == kNoHold ? kind : static_cast<std::uint8_t>(2U + 2U * hold + kind);
}

void PrefixCacheIndex::refresh_node(std::uint32_t index) {
    Node& node              = nodes_[index];
    const std::uint8_t list = node_lru_list(node);
    if (lru_links_[index].list != list) {
        if (lru_links_[index].list != kNoList) { lru_remove(index); }
        if (list != kNoList) { lru_append(index, list); }
    }
    const bool transfer_held = node.occupied &&
                               (node.device == CopyState::Resident || node.device == CopyState::Filling) &&
                               node.transfer_pins != 0 && node.device_pins == node.transfer_pins;
    if (transfer_held != node.transfer_counted) {
        node.transfer_counted = transfer_held;
        if (transfer_held) {
            ++transfer_releasable_;
        } else {
            --transfer_releasable_;
        }
    }
    const bool dead =
        node.occupied && node.host == CopyState::Resident && node.live_below == 0 && !held(node);
    if (dead != node.in_dead) {
        if (dead) {
            dead_append(index);
        } else {
            dead_remove(index);
        }
    }
    const bool ssd_dead =
        node.occupied && node.ssd == CopyState::Resident && node.live_below == 0 && !held(node);
    if (ssd_dead != node.in_ssd_dead) {
        if (ssd_dead) {
            ssd_dead_append(index);
        } else {
            ssd_dead_remove(index);
        }
    }
    // A Host copy the SSD tier also holds can go without loss (§5.7): first where no snapshot below
    // has its image on the Device or the Host. Held paths keep theirs (§9.6).
    const bool clean = node.occupied && node.host == CopyState::Resident && node.ssd == CopyState::Resident;
    if (clean != node.clean_counted) {
        node.clean_counted = clean;
        if (clean) {
            ++host_clean_nodes_;
        } else {
            --host_clean_nodes_;
        }
    }
    const std::uint8_t clean_list = clean && !held(node) && node.hold == kNoHold && node.live_below != 0
                                        ? (node.warm_below == 0 ? kCold : kCleanWarm)
                                        : kNoList;
    if (clean_list != node.clean_list) {
        if (node.clean_list != kNoList) { clean_remove(index); }
        if (clean_list != kNoList) { clean_append(index, clean_list); }
    }
}

void PrefixCacheIndex::refresh_tail(std::uint32_t index) {
    const Snapshot& snapshot  = snapshots_[index];
    const std::uint32_t entry = config_.max_nodes + index;
    const bool lru    = snapshot.occupied && snapshot.tail_len != 0 &&
                        snapshot.tail_device_copy == CopyState::Resident && snapshot.pins == 0 &&
                        (snapshot.io_pins == 0 || tail_backed(snapshot));
    const bool backed = tail_backed(snapshot);
    const std::uint8_t list = lru ? lru_list(backed, snapshot.hold) : kNoList;
    if (lru_links_[entry].list != list) {
        if (lru_links_[entry].list != kNoList) { lru_remove(entry); }
        if (list != kNoList) { lru_append(entry, list); }
    }
    if (snapshot.occupied) { sync_snapshot(index); }
}

void PrefixCacheIndex::sync_snapshot(std::uint32_t index) {
    Snapshot& snapshot = snapshots_[index];
    const bool warm =
        snapshot.occupied && (snapshot.device_slot != kNoId || snapshot.host != CopyState::Absent);
    if (warm != snapshot.warm_counted) {
        snapshot.warm_counted = warm;
        if (snapshot.anchor != kNoId) { adjust_warm(snapshot.anchor, warm ? +1 : -1); }
    }
    const bool clean =
        snapshot.occupied && snapshot.host == CopyState::Resident && snapshot.ssd == CopyState::Resident;
    if (clean != snapshot.clean_counted) {
        snapshot.clean_counted = clean;
        const std::uint64_t slabs = snapshot_slab_count(snapshot);
        if (clean) {
            host_clean_image_slabs_ += slabs;
        } else {
            host_clean_image_slabs_ -= slabs;
        }
    }
}

void PrefixCacheIndex::adjust_warm(std::uint32_t anchor, int delta) {
    for (std::uint32_t current = anchor; current != kNoId; current = nodes_[current].parent) {
        Node& node = nodes_[current];
        if (delta < 0 && node.warm_below == 0) { invariant("prefix cache warm snapshot count underflow"); }
        node.warm_below = delta < 0 ? node.warm_below - 1U : node.warm_below + 1U;
        refresh_node(current);
    }
}

void PrefixCacheIndex::lru_remove(std::uint32_t entry) {
    LruLink& link = lru_links_[entry];
    if (link.list == kNoList) { return; }
    LruList& list = lru_[link.list];
    if (link.prev != kNoId) {
        lru_links_[link.prev].next = link.next;
    } else {
        list.head = link.next;
    }
    if (link.next != kNoId) {
        lru_links_[link.next].prev = link.prev;
    } else {
        list.tail = link.prev;
    }
    --list.count;
    --(link.list % 2 == 0 ? lru_backed_ : lru_unbacked_);
    link = LruLink{};
}

void PrefixCacheIndex::lru_append(std::uint32_t entry, std::uint8_t list_id) {
    LruLink& link = lru_links_[entry];
    LruList& list = lru_[list_id];
    link.list     = list_id;
    link.prev     = list.tail;
    link.next     = kNoId;
    if (list.tail != kNoId) {
        lru_links_[list.tail].next = entry;
    } else {
        list.head = entry;
    }
    list.tail = entry;
    ++list.count;
    ++(list_id % 2 == 0 ? lru_backed_ : lru_unbacked_);
}

std::uint32_t PrefixCacheIndex::lru_pop(std::uint8_t list_id) {
    const std::uint32_t entry = lru_[list_id].head;
    if (entry != kNoId) { lru_remove(entry); }
    return entry;
}

std::uint32_t PrefixCacheIndex::lru_pop_victim(bool unbacked, bool unheld_only) {
    // Dead blocks first: no request can resume through them, so their pages are free room, while every
    // other copy is one a waiting or later request restores or recomputes.
    ++operations_.device_list_probes;
    if (lru_[kLruDeadBacked].count != 0) { return lru_pop(kLruDeadBacked); }
    if (unbacked) {
        ++operations_.device_list_probes;
        if (lru_[kLruDeadUnbacked].count != 0) { return lru_pop(kLruDeadUnbacked); }
    }
    for (const bool backed : {true, false}) {
        if (!backed && !unbacked) { break; }
        const std::uint8_t unheld = lru_list(backed, kNoHold);
        ++operations_.device_list_probes;
        if (lru_[unheld].count != 0) { return lru_pop(unheld); }
        if (unheld_only) { continue; }
        for (std::uint32_t rank = kMaxQueueHolds; rank-- > 0;) {
            const std::uint8_t list = lru_list(backed, static_cast<std::uint8_t>(rank));
            ++operations_.device_list_probes;
            if (lru_[list].count != 0) {
                ++counters_.held_device_evictions;
                return lru_pop(list);
            }
        }
    }
    return kNoId;
}

template <std::uint32_t PrefixCacheIndex::Node::*Prev, std::uint32_t PrefixCacheIndex::Node::*Next>
void PrefixCacheIndex::list_unlink(std::uint32_t index, std::uint32_t& head, std::uint32_t& tail) noexcept {
    Node& node = nodes_[index];
    if (node.*Prev != kNoId) {
        nodes_[node.*Prev].*Next = node.*Next;
    } else {
        head = node.*Next;
    }
    if (node.*Next != kNoId) {
        nodes_[node.*Next].*Prev = node.*Prev;
    } else {
        tail = node.*Prev;
    }
    node.*Prev = kNoId;
    node.*Next = kNoId;
}

template <std::uint32_t PrefixCacheIndex::Node::*Prev, std::uint32_t PrefixCacheIndex::Node::*Next>
void PrefixCacheIndex::list_append(std::uint32_t index, std::uint32_t& head, std::uint32_t& tail) noexcept {
    Node& node = nodes_[index];
    node.*Prev = tail;
    node.*Next = kNoId;
    if (tail != kNoId) {
        nodes_[tail].*Next = index;
    } else {
        head = index;
    }
    tail = index;
}

void PrefixCacheIndex::dead_remove(std::uint32_t index) {
    Node& node = nodes_[index];
    if (!node.in_dead) { return; }
    list_unlink<&Node::dead_prev, &Node::dead_next>(index, dead_head_, dead_tail_);
    node.in_dead = false;
    --dead_count_;
}

// Each append binds the node first: re-indexing nodes_ after the head and tail stores would reload its storage.
void PrefixCacheIndex::dead_append(std::uint32_t index) {
    Node& node = nodes_[index];
    list_append<&Node::dead_prev, &Node::dead_next>(index, dead_head_, dead_tail_);
    node.in_dead = true;
    ++dead_count_;
}

void PrefixCacheIndex::ssd_dead_remove(std::uint32_t index) {
    Node& node = nodes_[index];
    if (!node.in_ssd_dead) { return; }
    list_unlink<&Node::ssd_dead_prev, &Node::ssd_dead_next>(index, ssd_dead_head_, ssd_dead_tail_);
    node.in_ssd_dead = false;
    --ssd_dead_count_;
}

void PrefixCacheIndex::ssd_dead_append(std::uint32_t index) {
    Node& node = nodes_[index];
    list_append<&Node::ssd_dead_prev, &Node::ssd_dead_next>(index, ssd_dead_head_, ssd_dead_tail_);
    node.in_ssd_dead = true;
    ++ssd_dead_count_;
}

void PrefixCacheIndex::clean_remove(std::uint32_t index) {
    Node& node = nodes_[index];
    if (node.clean_list == kNoList) { return; }
    const std::uint8_t list = node.clean_list;
    list_unlink<&Node::clean_prev, &Node::clean_next>(index, clean_head_[list], clean_tail_[list]);
    node.clean_list = kNoList;
    --host_clean_listed_;
}

void PrefixCacheIndex::clean_append(std::uint32_t index, std::uint8_t list) {
    Node& node = nodes_[index];
    list_append<&Node::clean_prev, &Node::clean_next>(index, clean_head_[list], clean_tail_[list]);
    node.clean_list = list;
    ++host_clean_listed_;
}

void PrefixCacheIndex::adjust_live(std::uint32_t anchor, int delta) {
    for (std::uint32_t current = anchor; current != kNoId; current = nodes_[current].parent) {
        Node& node = nodes_[current];
        if (delta < 0 && node.live_below == 0) {
            invariant("prefix cache live snapshot count underflow");
        }
        node.live_below = delta < 0 ? node.live_below - 1U : node.live_below + 1U;
        refresh_node(current);
    }
}

void PrefixCacheIndex::set_device(Node& node, CopyState state) noexcept {
    if (node.device == CopyState::Resident) { --device_resident_; }
    if (state == CopyState::Resident) { ++device_resident_; }
    node.device = state;
}

void PrefixCacheIndex::set_slot(std::uint32_t slot, SlotState state) noexcept {
    if (slot_state_[slot] == SlotState::Free) { --free_slots_; }
    if (state == SlotState::Free) { ++free_slots_; }
    slot_state_[slot] = state;
}

// ---- tree mutation -----------------------------------------------------------------------------

InsertResult PrefixCacheIndex::insert_block(NodeRef parent, std::uint64_t lookup_hash,
                                            std::span<const TokenId> block_tokens,
                                            std::uint64_t extra, std::uint32_t device_id,
                                            bool attach) {
    const std::uint32_t parent_index = parent.valid() ? parent.index : kNoId;
    if (parent.valid() && require(parent).pins == 0) {
        invariant("prefix cache insertion parent is not pinned by the inserting sequence");
    }
    if (device_id == kNoId) { throw std::invalid_argument("prefix cache block has no device id"); }
    if (const auto existing = find_child(parent, lookup_hash, block_tokens, extra)) {
        Node& node = nodes_[existing->index];
        InsertResult result{*existing, false, false};
        const bool unpinned = node.pins == 0;
        if (attach && node.device == CopyState::Absent) {
            // A host-only block recomputed by this sequence regains its Device copy.
            node.device_id = device_id;
            set_device(node, CopyState::Resident);
            result.device_attached = true;
        } else if (attach && node.device == CopyState::Resident && unpinned) {
            // A block recomputed by this sequence whose Device copy nobody maps: the sequence's page
            // replaces it, so the two never hold two pages for one block (the sequence pins its
            // own, and an evictable copy would otherwise become pinned beside it).
            result.replaced_device_id = node.device_id;
            node.device_id            = device_id;
            result.device_attached    = true;
        }
        // An SSD transfer (§5.7) may read or write its slabs: they keep, like a pinned node's.
        const bool copies_free = unpinned && node.io_pins == 0;
        if (result.device_attached && copies_free && node.host == CopyState::Resident) {
            // A recomputation need not repeat the earlier bits (the default route's kernels depend on
            // the call's width), so the Host copy of the earlier computation goes: Host restores and
            // saved files would otherwise serve it beside snapshots taken from the new page. The write
            // through backs the new page. A pinned node's slab may be under a restore copy, and keeps.
            drop_node_host_copy(existing->index);
        }
        if (result.device_attached && copies_free && node.ssd == CopyState::Resident) {
            // The SSD tier's copy of the earlier computation goes for the same reason.
            drop_node_ssd_copy(existing->index);
        }
        ++node.pins;
        ++node.device_pins; // the sequence maps it
        refresh_node(existing->index);
        return result;
    }
    if (free_nodes_.empty()) { invariant("prefix cache node capacity exhausted"); }
    const std::uint32_t index = free_nodes_.back();
    free_nodes_.pop_back();
    Node& node    = nodes_[index];
    node.occupied = true;
    node.parent   = parent_index;
    node.depth    = parent.valid() ? nodes_[parent_index].depth + 1U : 0U;
    node.hash     = lookup_hash;
    node.extra    = extra;
    std::copy(block_tokens.begin(), block_tokens.end(), node.tokens.begin());
    node.children.clear();
    node.snapshots.clear();
    node.device_id = device_id;
    node.device    = CopyState::Absent;
    set_device(node, CopyState::Resident);
    node.host_slab      = kNoId;
    node.host           = CopyState::Absent;
    node.ssd_slab       = kNoId;
    node.ssd            = CopyState::Absent;
    node.pins           = 1;
    node.device_pins    = 1; // the inserting sequence maps it
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
    return InsertResult{ref_of_node(index), true, true};
}

void PrefixCacheIndex::begin_device_fill(NodeRef ref, std::uint32_t device_id) {
    Node& node = require(ref);
    if (node.device != CopyState::Absent || node.host != CopyState::Resident || node.pins == 0 ||
        device_id == kNoId) {
        invariant("prefix cache device fill requires a pinned host-only node");
    }
    set_device(node, CopyState::Filling);
    node.device_id = device_id;
    refresh_node(ref.index);
}

void PrefixCacheIndex::complete_device_fill(NodeRef ref) {
    Node& node = require(ref);
    if (node.device != CopyState::Filling) { invariant("prefix cache device fill not open"); }
    set_device(node, CopyState::Resident);
    refresh_node(ref.index);
}

void PrefixCacheIndex::abort_device_fill(NodeRef ref) {
    Node& node = require(ref);
    if (node.device != CopyState::Filling) { invariant("prefix cache device fill not open"); }
    const std::uint32_t id = node.device_id;
    set_device(node, CopyState::Absent);
    node.device_id = kNoId;
    refresh_node(ref.index);
    backend_->release_device_block(id);
}

std::uint32_t PrefixCacheIndex::device_evictable_blocks() const noexcept {
    return lru_backed_ + lru_unbacked_;
}

std::uint32_t PrefixCacheIndex::device_backed_evictable_blocks() const noexcept {
    return lru_backed_;
}

std::uint32_t PrefixCacheIndex::device_pinned_nodes() const noexcept {
    return static_cast<std::uint32_t>(std::count_if(
        nodes_.begin(), nodes_.end(), [](const Node& node) { return node.occupied && node.device_pins != 0; }));
}

void PrefixCacheIndex::drop_node_device_copy(std::uint32_t index) {
    Node& node = nodes_[index];
    if (node.device != CopyState::Resident) { invariant("prefix cache drops a non-resident page"); }
    const std::uint32_t id = node.device_id;
    set_device(node, CopyState::Absent);
    node.device_id = kNoId;
    refresh_node(index);
    backend_->release_device_block(id);
}

void PrefixCacheIndex::drop_node_host_copy(std::uint32_t index) {
    Node& node = nodes_[index];
    if (node.host != CopyState::Resident) { invariant("prefix cache drops a non-resident slab"); }
    const std::uint32_t slab = node.host_slab;
    node.host                = CopyState::Absent;
    set_node_slab(index, kNoId);
    free_slab(slab);
    refresh_node(index);
}

std::uint32_t PrefixCacheIndex::evict_device_blocks(std::uint32_t blocks) {
    return evict_device_entries(blocks, true);
}

std::uint32_t PrefixCacheIndex::evict_backed_device_blocks(std::uint32_t blocks) {
    return evict_device_entries(blocks, false);
}

std::uint32_t PrefixCacheIndex::evict_unheld_backed_device_blocks(std::uint32_t blocks) {
    return evict_device_entries(blocks, false, true);
}

std::uint32_t PrefixCacheIndex::device_unheld_backed_evictable_blocks() const noexcept {
    return lru_[kLruDeadBacked].count + lru_[lru_list(true, kNoHold)].count;
}

std::uint32_t PrefixCacheIndex::evict_device_entries(std::uint32_t blocks, bool unbacked, bool unheld_only) {
    std::uint32_t released = 0;
    while (released < blocks) {
        const std::uint32_t entry = lru_pop_victim(unbacked, unheld_only);
        if (entry == kNoId) { break; }
        if (entry < config_.max_nodes) {
            Node& node = nodes_[entry];
            if (node_backed(node)) {
                drop_node_device_copy(entry);
                ++counters_.device_block_evictions;
                ++released;
            } else {
                // Unbacked: the node loses its only copy together with its (unpinned) subtree.
                std::uint32_t pages = 0;
                std::vector<std::uint32_t> stack{entry};
                while (!stack.empty()) {
                    const std::uint32_t current = stack.back();
                    stack.pop_back();
                    if (nodes_[current].device == CopyState::Resident) { ++pages; }
                    for (const std::uint32_t snapshot : nodes_[current].snapshots) {
                        if (snapshots_[snapshot].tail_device_copy == CopyState::Resident) {
                            ++pages;
                        }
                    }
                    for (const std::uint32_t child : nodes_[current].children) {
                        stack.push_back(child);
                    }
                }
                remove_subtree(entry);
                ++counters_.unbacked_node_losses;
                counters_.device_block_evictions += pages;
                released += pages;
            }
        } else {
            const std::uint32_t index = entry - config_.max_nodes;
            Snapshot& snapshot        = snapshots_[index];
            ++counters_.device_block_evictions;
            ++released;
            if (tail_backed(snapshot)) {
                const std::uint32_t id    = snapshot.tail_device;
                snapshot.tail_device      = kNoId;
                snapshot.tail_device_copy = CopyState::Absent;
                refresh_tail(index);
                backend_->release_device_block(id);
            } else {
                remove_snapshot(index);
            }
        }
    }
    return released;
}

// ---- fills: Host copies of blocks and snapshots, a snapshot tail's Device copy -----------------

std::optional<std::uint32_t> PrefixCacheIndex::begin_host_fill(NodeRef ref) {
    Node& node = require(ref);
    if (node.device != CopyState::Resident || node.host != CopyState::Absent || node.pins == 0) {
        invariant("prefix cache host fill requires a pinned device-only node");
    }
    if (!config_.host_blocks) { return std::nullopt; }
    std::vector<std::uint32_t> slab;
    // A claim below every value: dead blocks, lossless copies and superseded snapshots yield, no retained one.
    // Issue #77: at an infinite claim, a tier smaller than one conversation's blocks and image gave up that
    // conversation's only snapshot for a second copy of a block the Device held, and its next turn re-prefilled.
    // Without an SSD tier one image and its tail stay free for the lineage's next capture; with one, a capture
    // makes its room losslessly (§5.7), and a reserve would only demote more than the write needs.
    const std::uint32_t wanted = config_.ssd_slabs != 0 ? 1U : config_.image_slabs + 2U;
    if (!make_slabs_free(wanted, kNoId, -std::numeric_limits<double>::infinity()) || !take_free_slabs(1, slab)) {
        return std::nullopt;
    }
    nodes_[ref.index].host = CopyState::Filling;
    set_node_slab(ref.index, slab.front());
    refresh_node(ref.index);
    return slab.front();
}

void PrefixCacheIndex::complete_host_fill(NodeRef ref) {
    Node& node = require(ref);
    if (node.host != CopyState::Filling) { invariant("prefix cache host fill not open"); }
    node.host = CopyState::Resident;
    refresh_node(ref.index);
}

void PrefixCacheIndex::abort_host_fill(NodeRef ref) {
    Node& node = require(ref);
    if (node.host != CopyState::Filling) { invariant("prefix cache host fill not open"); }
    free_slab(node.host_slab);
    node.host = CopyState::Absent;
    set_node_slab(ref.index, kNoId);
    refresh_node(ref.index);
}

bool PrefixCacheIndex::begin_snapshot_host_fill(SnapshotRef ref, std::optional<double> claim) {
    Snapshot& snapshot = require(ref);
    if (snapshot.host != CopyState::Absent || snapshot.pins == 0) {
        invariant("prefix cache snapshot host fill requires a pinned device-only snapshot");
    }
    const std::uint32_t count =
        config_.image_slabs + (config_.host_blocks && snapshot.tail_len != 0 ? 1U : 0U);
    std::vector<std::uint32_t> slabs;
    // The image goes to the Host only if it is worth what it displaces; otherwise it stays
    // Device-only.
    if (!allocate_slabs(count, slabs, ref.index, claim.value_or(snapshot.priority))) { return false; }
    Snapshot& filled  = snapshots_[ref.index];
    filled.host_slabs = std::move(slabs);
    filled.host       = CopyState::Filling;
    refresh_tail(ref.index);
    return true;
}

void PrefixCacheIndex::complete_snapshot_host_fill(SnapshotRef ref) {
    Snapshot& snapshot = require(ref);
    if (snapshot.host != CopyState::Filling) { invariant("prefix cache snapshot fill not open"); }
    snapshot.host = CopyState::Resident;
    refresh_tail(ref.index);
}

void PrefixCacheIndex::abort_snapshot_host_fill(SnapshotRef ref) {
    Snapshot& snapshot = require(ref);
    if (snapshot.host != CopyState::Filling) { invariant("prefix cache snapshot fill not open"); }
    for (const std::uint32_t slab : snapshot.host_slabs) { free_slab(slab); }
    snapshot.host_slabs.clear();
    snapshot.host = CopyState::Absent;
    refresh_tail(ref.index);
}

void PrefixCacheIndex::begin_tail_device_fill(SnapshotRef ref, std::uint32_t device_id) {
    Snapshot& snapshot = require(ref);
    if (!config_.host_blocks || snapshot.tail_len == 0 ||
        snapshot.tail_device_copy != CopyState::Absent || snapshot.host != CopyState::Resident ||
        snapshot.pins == 0 || device_id == kNoId) {
        invariant("prefix cache tail fill requires a pinned host-backed tail");
    }
    snapshot.tail_device      = device_id;
    snapshot.tail_device_copy = CopyState::Filling;
    refresh_tail(ref.index);
}

void PrefixCacheIndex::complete_tail_device_fill(SnapshotRef ref) {
    Snapshot& snapshot = require(ref);
    if (snapshot.tail_device_copy != CopyState::Filling) {
        invariant("prefix cache tail fill not open");
    }
    snapshot.tail_device_copy = CopyState::Resident;
    refresh_tail(ref.index);
}

void PrefixCacheIndex::abort_tail_device_fill(SnapshotRef ref) {
    Snapshot& snapshot = require(ref);
    if (snapshot.tail_device_copy != CopyState::Filling) {
        invariant("prefix cache tail fill not open");
    }
    const std::uint32_t id    = snapshot.tail_device;
    snapshot.tail_device      = kNoId;
    snapshot.tail_device_copy = CopyState::Absent;
    refresh_tail(ref.index);
    backend_->release_device_block(id);
}

// ---- snapshots ---------------------------------------------------------------------------------

std::optional<std::uint32_t> PrefixCacheIndex::acquire_device_slot(double claim) {
    for (std::uint32_t slot = 0; slot < slot_state_.size(); ++slot) {
        if (slot_state_[slot] == SlotState::Free) {
            set_slot(slot, SlotState::Staging);
            return slot;
        }
    }
    const auto take = [&](std::uint32_t slot) {
        ++counters_.device_slot_evictions;
        set_slot(slot, SlotState::Staging);
        return slot;
    };
    const auto drop_device_copy = [&](std::uint32_t slot) {
        const std::uint32_t owner = slot_owner_[slot];
        backend_->drop_snapshot_device_image(ref_of_snapshot(owner));
        snapshots_[owner].device_slot = kNoId;
        slot_owner_[slot]             = kNoId;
        refresh_tail(owner); // backed by the SSD alone, its blocks may turn cold
        return take(slot);
    };
    // A superseded owner goes first, backed or not: its lineage resumes from a newer snapshot.
    std::uint32_t superseded = kNoId;
    std::uint32_t unbacked   = kNoId;
    std::uint32_t backed     = kNoId;
    // Owned slots whose (retained) owner has no Host copy, evictable or not.
    std::uint32_t device_only_slots = 0;
    for (std::uint32_t slot = 0; slot < slot_state_.size(); ++slot) {
        if (slot_state_[slot] != SlotState::Owned) { continue; }
        const Snapshot& owner = snapshots_[slot_owner_[slot]];
        if (!owner.superseded && !image_backed(owner)) { ++device_only_slots; }
        if (owner.pins != 0) { continue; }
        // A held owner without a Host copy would be lost with its slot, and so would one whose blocks an
        // SSD promotion is reading (§5.7).
        if ((owner.hold != kNoHold || owner.io_pins != 0) && !image_backed(owner)) { continue; }
        if (owner.superseded) {
            if (superseded == kNoId ||
                owner.superseded_tick < snapshots_[slot_owner_[superseded]].superseded_tick) {
                superseded = slot;
            }
        } else if (!image_backed(owner)) {
            const Snapshot* best = unbacked == kNoId ? nullptr : &snapshots_[slot_owner_[unbacked]];
            if (best == nullptr || owner.priority < best->priority ||
                (owner.priority == best->priority && owner.last_hit_tick < best->last_hit_tick)) {
                unbacked = slot;
            }
        } else if (backed == kNoId ||
                   owner.last_hit_tick < snapshots_[slot_owner_[backed]].last_hit_tick) {
            backed = slot;
        }
    }
    if (superseded != kNoId) {
        if (image_backed(snapshots_[slot_owner_[superseded]])) {
            return drop_device_copy(superseded);
        }
        remove_snapshot(slot_owner_[superseded]);
        return take(superseded);
    }
    // A Device-only owner is lost when its slot is taken, so value decides: the least valuable
    // goes, before any backed owner gives up its Device copy, but only for a new snapshot worth
    // at least as much. Host eviction leaves such owners behind, and without this they hold their
    // slots for good; a short request's snapshots must not push out the one a long conversation
    // resumes from.
    // Device-only owners hold at most D - 1 slots, though: GDSF time advances only on an eviction,
    // so owners worth more than every new claim would otherwise keep all D slots for the rest of
    // the session and no new conversation could publish a snapshot (issue #29). The least
    // valuable evictable owner yields, unless it is the most valuable Device-only owner (the
    // others pinned or held): that one keeps its slot. A single slot keeps value's rule alone.
    bool rotate = false;
    if (unbacked != kNoId && slot_state_.size() >= 2 && device_only_slots == slot_state_.size()) {
        const double candidate = snapshots_[slot_owner_[unbacked]].priority;
        for (std::uint32_t slot = 0; slot < slot_state_.size() && !rotate; ++slot) {
            const Snapshot& owner = snapshots_[slot_owner_[slot]];
            rotate = slot != unbacked && !owner.superseded && !image_backed(owner) &&
                     owner.priority >= candidate;
        }
    }
    if (unbacked != kNoId && (snapshots_[slot_owner_[unbacked]].priority <= claim || rotate)) {
        // GDSF time rises to the value a claim displaced. A rotation takes an owner worth more than the
        // claim, which says nothing about the cache's least valuable snapshot: time then rises no further
        // than that one's value, or every rotation would lift it by a hot owner's value and age the
        // Host tier's snapshots as fast as slots turn over.
        const double lost = snapshots_[slot_owner_[unbacked]].priority;
        inflation_        = std::max(inflation_, lost <= claim ? lost : std::min(lost, lowest_retained_priority()));
        remove_snapshot(slot_owner_[unbacked]);
        return take(unbacked);
    }
    // A backed owner keeps its Host copy: the least recently hit gives up its Device copy.
    if (backed != kNoId) { return drop_device_copy(backed); }
    if (!slot_state_.empty()) { ++counters_.device_slot_refusals; }
    return std::nullopt;
}

double PrefixCacheIndex::estimate_priority(std::uint32_t base_frontier, std::uint32_t frontier,
                                           bool tail) const noexcept {
    // update_priority for a snapshot that has not been published yet: no hits, and the blocks
    // between its nearest retained snapshot and itself are its own.
    if (frontier <= base_frontier) { return inflation_; }
    const double saved = config_.cost.prefill_seconds(base_frontier, frontier - base_frontier) -
                         config_.cost.restore_seconds(config_.image_bytes);
    const std::uint64_t exclusive_blocks = frontier / kBlockTokens - base_frontier / kBlockTokens;
    const double size =
        static_cast<double>(config_.image_bytes) +
        static_cast<double>(tail ? config_.block_bytes : 0ULL) +
        static_cast<double>(exclusive_blocks) * static_cast<double>(config_.block_bytes);
    return inflation_ + std::max(0.0, saved) / std::max(size, 1.0);
}

void PrefixCacheIndex::release_device_slot(std::uint32_t slot) {
    if (slot >= slot_state_.size() || slot_state_[slot] != SlotState::Staging) {
        invariant("prefix cache releases a device slot it did not stage");
    }
    set_slot(slot, SlotState::Free);
}

PublishResult PrefixCacheIndex::publish_snapshot(NodeRef anchor, std::uint32_t frontier,
                                                 std::span<const TokenId> tail,
                                                 std::optional<std::uint32_t> tail_device_id,
                                                 std::uint32_t device_slot, SnapshotKind kind,
                                                 std::optional<double> claim, bool adopt) {
    // A refused call consumes what it was given, as every other outcome does (publish_image).
    if (device_slot >= slot_state_.size() || slot_state_[device_slot] != SlotState::Staging) {
        if (tail_device_id && *tail_device_id != kNoId) { backend_->release_device_block(*tail_device_id); }
        invariant("prefix cache snapshot image is not in a staging slot");
    }
    // A Device image's tail lives on the Device until a Host fill backs the whole snapshot.
    if (!tail.empty() && !tail_device_id) {
        set_slot(device_slot, SlotState::Free);
        throw std::invalid_argument("prefix cache snapshot tail is inconsistent");
    }
    return publish_image(anchor, frontier, tail, tail_device_id, device_slot, {}, kind, claim, adopt);
}

std::optional<HostImageReservation> PrefixCacheIndex::reserve_host_image(bool tail, double claim) {
    // A Host-resident snapshot always has its tail on the Host (snapshot_valid, refresh_tail).
    if (tail && !config_.host_blocks) {
        throw std::invalid_argument("prefix cache Host-born snapshot tail requires Host blocks");
    }
    if (config_.host_slabs == 0) { return std::nullopt; }
    HostImageReservation reservation;
    reservation.tail = tail;
    if (!allocate_slabs(config_.image_slabs + (tail ? 1U : 0U), reservation.slabs, kNoId, claim)) {
        return std::nullopt;
    }
    for (const std::uint32_t slab : reservation.slabs) { slab_reserved_[slab] = true; }
    return reservation;
}

void PrefixCacheIndex::release_host_image(HostImageReservation&& reservation) noexcept {
    for (const std::uint32_t slab : reservation.slabs) {
        // Only slabs still reserved: a reservation already released or published frees nothing.
        if (slab < slab_reserved_.size() && slab_reserved_[slab]) {
            slab_reserved_[slab] = false;
            free_slab(slab);
        }
    }
    reservation.slabs.clear();
    reservation.tail = false;
}

PublishResult PrefixCacheIndex::publish_host_snapshot(NodeRef anchor, std::uint32_t frontier,
                                                      std::span<const TokenId> tail,
                                                      std::optional<std::uint32_t> tail_device_id,
                                                      HostImageReservation&& image,
                                                      SnapshotKind kind, std::optional<double> claim) {
    // A refused call consumes what it was given, as every other outcome does (publish_image); release_host_image
    // frees only slabs still reserved.
    const auto refuse = [&]() noexcept {
        release_host_image(std::move(image));
        if (tail_device_id && *tail_device_id != kNoId) { backend_->release_device_block(*tail_device_id); }
    };
    if (image.tail != !tail.empty()) {
        refuse();
        throw std::invalid_argument(
            "prefix cache Host-born snapshot tail does not match its reservation");
    }
    if (image.slabs.size() != config_.image_slabs + (image.tail ? 1U : 0U) ||
        !std::all_of(image.slabs.begin(), image.slabs.end(), [&](std::uint32_t slab) {
            return slab < slab_reserved_.size() && slab_reserved_[slab];
        })) {
        refuse();
        invariant("prefix cache Host-born snapshot image is not a reservation");
    }
    const PublishResult result =
        publish_image(anchor, frontier, tail, tail_device_id, kNoId, std::move(image.slabs), kind, claim, false);
    image.slabs.clear();
    image.tail = false;
    return result;
}

PublishResult PrefixCacheIndex::publish_image(NodeRef anchor, std::uint32_t frontier,
                                              std::span<const TokenId> tail,
                                              std::optional<std::uint32_t> tail_device_id,
                                              std::uint32_t device_slot,
                                              std::vector<std::uint32_t>&& host_slabs,
                                              SnapshotKind kind, std::optional<double> claim, bool adopt) {
    // Every outcome consumes the image (its staging slot or its reserved slabs) and the tail id, a refusal here
    // included: the caller cannot tell a refusal from a throw past the first change below (an allocation, after
    // which the snapshot owns them), so it never gives them back itself.
    const auto refuse = [&]() noexcept {
        if (device_slot != kNoId) { set_slot(device_slot, SlotState::Free); }
        for (const std::uint32_t slab : host_slabs) {
            slab_reserved_[slab] = false;
            free_slab(slab);
        }
        host_slabs.clear();
        if (tail_device_id && *tail_device_id != kNoId) { backend_->release_device_block(*tail_device_id); }
    };
    if (tail.size() >= kBlockTokens || (tail.empty() && tail_device_id) ||
        (tail_device_id && *tail_device_id == kNoId)) {
        refuse();
        throw std::invalid_argument("prefix cache snapshot tail is inconsistent");
    }
    if (anchor.valid() && !valid(anchor)) {
        refuse();
        invariant("stale prefix cache node reference");
    }
    const std::uint32_t anchor_index = anchor.valid() ? anchor.index : kNoId;
    const std::uint32_t base = anchor.valid() ? (nodes_[anchor_index].depth + 1U) * kBlockTokens : 0U;
    if (frontier == 0 || frontier != base + static_cast<std::uint32_t>(tail.size())) {
        refuse();
        throw std::invalid_argument("prefix cache snapshot frontier does not match its anchor");
    }
    // Validated: the reserved slabs now belong to this publication.
    for (const std::uint32_t slab : host_slabs) { slab_reserved_[slab] = false; }
    const auto release_image = [&] {
        if (device_slot != kNoId) { set_slot(device_slot, SlotState::Free); }
        for (const std::uint32_t slab : host_slabs) { free_slab(slab); }
        host_slabs.clear();
        if (tail_device_id) { backend_->release_device_block(*tail_device_id); }
    };
    std::vector<std::uint32_t>& anchored =
        anchor.valid() ? nodes_[anchor_index].snapshots : root_snapshots_;
    for (const std::uint32_t existing : anchored) {
        Snapshot& snapshot = snapshots_[existing];
        if (snapshot.tail_len != tail.size() || !std::equal(tail.begin(), tail.end(), snapshot.tail.begin())) {
            continue;
        }
        // The same state published again: the lineage that published it resumes from this snapshot next,
        // so it is recent and, if another lineage had moved past it, its lineage's resume point again
        // (left superseded it was the first victim, with the snapshot the lineage came from superseded
        // behind it). It takes the copies it lacks: a tail's Device copy, and the Device slot when the
        // caller attaches the image.
        PublishResult result{ref_of_snapshot(existing), false};
        if (tail_device_id && snapshot.tail_device_copy == CopyState::Absent) {
            snapshot.tail_device      = *tail_device_id;
            snapshot.tail_device_copy = CopyState::Resident;
            tail_device_id.reset();
            result.tail_adopted = true;
        }
        if (adopt && device_slot != kNoId && snapshot.device_slot == kNoId) {
            snapshot.device_slot = device_slot;
            set_slot(device_slot, SlotState::Owned);
            slot_owner_[device_slot] = existing;
            device_slot              = kNoId;
            result.device_adopted    = true;
        }
        release_image();
        snapshot.last_hit_tick = ++tick_;
        if (snapshot.superseded) { retain_again(existing); }
        refresh_tail(existing);
        return result;
    }
    if (free_snapshots_.empty()) {
        // Make room by evicting a superseded snapshot, otherwise the GDSF minimum, and that one only for a
        // snapshot worth at least as much (GDSF never inserts what it would evict next).
        const std::uint32_t victim = pick_victim(false, kNoId);
        const double worth         = claim.value_or(estimate_priority(
            retained_frontier_above(anchor_index, frontier, tail), frontier, !tail.empty()));
        if (victim == kNoId || (!snapshots_[victim].superseded && snapshots_[victim].priority > worth)) {
            release_image();
            return PublishResult{};
        }
        if (snapshots_[victim].superseded) {
            ++counters_.superseded_evictions;
        } else {
            inflation_ = std::max(inflation_, std::min(snapshots_[victim].priority, worth));
        }
        remove_snapshot(victim);
        ++counters_.host_snapshot_evictions;
    }
    const std::uint32_t index = free_snapshots_.back();
    free_snapshots_.pop_back();
    Snapshot& snapshot = snapshots_[index];
    snapshot.occupied  = true;
    snapshot.anchor    = anchor_index;
    snapshot.frontier  = frontier;
    snapshot.tail_len  = static_cast<std::uint32_t>(tail.size());
    std::copy(tail.begin(), tail.end(), snapshot.tail.begin());
    snapshot.tail_device      = tail_device_id.value_or(kNoId);
    snapshot.tail_device_copy = tail_device_id ? CopyState::Resident : CopyState::Absent;
    snapshot.device_slot      = device_slot;
    snapshot.host            = host_slabs.empty() ? CopyState::Absent : CopyState::Resident;
    snapshot.host_slabs      = std::move(host_slabs);
    snapshot.ssd_slabs.clear();
    snapshot.ssd             = CopyState::Absent;
    snapshot.ssd_read        = false;
    snapshot.ssd_packed      = false;
    snapshot.ssd_offered     = false;
    snapshot.ssd_declined    = false;
    snapshot.warm_counted    = false;
    snapshot.clean_counted   = false;
    snapshot.pins            = 0;
    snapshot.io_pins         = 0;
    snapshot.kind            = kind;
    snapshot.hits            = 0;
    snapshot.last_hit_tick   = ++tick_;
    snapshot.priority_base   = inflation_;
    snapshot.superseded      = false;
    snapshot.superseded_tick = 0;
    snapshot.hold            = kNoHold;
    ++snapshot_epoch_;
    if (device_slot != kNoId) {
        set_slot(device_slot, SlotState::Owned);
        slot_owner_[device_slot] = index;
    }
    anchored.push_back(index);
    ++snapshot_count_;
    if (anchor.valid()) {
        adjust_live(anchor_index, +1);
        adjust_retained(anchor_index, +1);
    }
    link_snapshot(index);
    std::vector<std::uint32_t> dependents;
    collect_retained_dependents(index, dependents);
    revalue_around(index, dependents);
    // A retained snapshot on another branch shares its path's upper blocks with this one from now on.
    revalue_sibling(anchor_index, index, 1);
    refresh_tail(index);
    return PublishResult{ref_of_snapshot(index), true};
}

void PrefixCacheIndex::update_priority(std::uint32_t index) {
    // GDSF value of keeping this snapshot, given that superseded snapshots go first: without it,
    // a request continuing its lineage resumes from the nearest retained snapshot above it and
    // prefills the rest, and evicting it frees its image, its tail and the blocks no other
    // retained snapshot keeps.
    Snapshot& snapshot                    = snapshots_[index];
    const std::uint32_t ancestor          = retained_ancestor(index);
    const std::uint32_t ancestor_frontier = ancestor == kNoId ? 0U : snapshots_[ancestor].frontier;
    const std::uint32_t own               = snapshot.superseded ? 0U : 1U;
    std::uint64_t exclusive_blocks        = 0;
    for (std::uint32_t current = snapshot.anchor;
         current != kNoId && nodes_[current].retained_below <= own;
         current = nodes_[current].parent) {
        ++exclusive_blocks;
    }
    const double saved =
        config_.cost.prefill_seconds(ancestor_frontier, snapshot.frontier - ancestor_frontier) -
        config_.cost.restore_seconds(config_.image_bytes);
    const double cost = std::max(0.0, saved);
    const double size =
        static_cast<double>(config_.image_bytes) +
        static_cast<double>(snapshot.tail_len != 0 ? config_.block_bytes : 0ULL) +
        static_cast<double>(exclusive_blocks) * static_cast<double>(config_.block_bytes);
    const double frequency = 1.0 + static_cast<double>(snapshot.hits);
    snapshot.priority      = snapshot.priority_base + frequency * cost / std::max(size, 1.0);
}

// ---- snapshot ancestry -------------------------------------------------------------------------

bool PrefixCacheIndex::on_path(const Snapshot& above, const Snapshot& below) const noexcept {
    if (above.frontier >= below.frontier) { return false; }
    const std::uint32_t above_blocks = above.anchor == kNoId ? 0U : nodes_[above.anchor].depth + 1U;
    const std::uint32_t below_blocks = below.anchor == kNoId ? 0U : nodes_[below.anchor].depth + 1U;
    if (above_blocks > below_blocks) { return false; }
    // `tail_block` is the node of `below`'s path holding `above`'s tail tokens (absent when both
    // share an anchor); the node above it must be `above`'s anchor.
    std::uint32_t tail_block = kNoId;
    std::uint32_t current    = below.anchor;
    while (current != kNoId && nodes_[current].depth >= above_blocks) {
        tail_block = current;
        current    = nodes_[current].parent;
    }
    if (current != above.anchor) { return false; }
    const auto tail = above.tail.begin();
    if (tail_block == kNoId) { return std::equal(tail, tail + above.tail_len, below.tail.begin()); }
    return std::equal(tail, tail + above.tail_len, nodes_[tail_block].tokens.begin());
}

std::uint32_t PrefixCacheIndex::find_ancestor(std::uint32_t index) const noexcept {
    // Walk up the anchor path; the first node anchoring a snapshot on this path holds the nearest.
    const Snapshot& snapshot = snapshots_[index];
    std::uint32_t best       = kNoId;
    std::uint32_t previous   = kNoId;
    std::uint32_t current    = snapshot.anchor;
    for (;;) {
        const std::vector<std::uint32_t>& anchored =
            current != kNoId ? nodes_[current].snapshots : root_snapshots_;
        // Tokens right after this node on the path: the snapshot's own tail at its anchor.
        const TokenId* next =
            previous == kNoId ? snapshot.tail.data() : nodes_[previous].tokens.data();
        for (const std::uint32_t other : anchored) {
            const Snapshot& candidate = snapshots_[other];
            if (other == index || candidate.frontier >= snapshot.frontier ||
                !std::equal(candidate.tail.begin(), candidate.tail.begin() + candidate.tail_len,
                            next)) {
                continue;
            }
            if (best == kNoId || candidate.frontier > snapshots_[best].frontier) { best = other; }
        }
        if (best != kNoId || current == kNoId) { return best; }
        previous = current;
        current  = nodes_[current].parent;
    }
}

std::vector<std::uint32_t>& PrefixCacheIndex::dependents_of(std::uint32_t snapshot) noexcept {
    return snapshot == kNoId ? top_snapshots_ : snapshots_[snapshot].dependents;
}

void PrefixCacheIndex::link_snapshot(std::uint32_t index) {
    Snapshot& snapshot = snapshots_[index];
    snapshot.ancestor  = find_ancestor(index);
    snapshot.dependents.clear();
    // Snapshots below the new one had its ancestor as their nearest; they now hang from it.
    std::vector<std::uint32_t>& siblings = dependents_of(snapshot.ancestor);
    for (std::size_t i = 0; i < siblings.size();) {
        const std::uint32_t other = siblings[i];
        if (on_path(snapshot, snapshots_[other])) {
            snapshots_[other].ancestor = index;
            snapshot.dependents.push_back(other);
            siblings[i] = siblings.back();
            siblings.pop_back();
        } else {
            ++i;
        }
    }
    siblings.push_back(index);
}

void PrefixCacheIndex::unlink_snapshot(std::uint32_t index) {
    Snapshot& snapshot                   = snapshots_[index];
    std::vector<std::uint32_t>& siblings = dependents_of(snapshot.ancestor);
    // Room first (remove_snapshot has made it already): past this point nothing allocates, so a failed
    // allocation cannot leave the dependents half handed over.
    siblings.reserve(siblings.size() + snapshot.dependents.size());
    siblings.erase(std::find(siblings.begin(), siblings.end(), index));
    for (const std::uint32_t other : snapshot.dependents) {
        snapshots_[other].ancestor = snapshot.ancestor;
        siblings.push_back(other);
    }
    snapshot.dependents.clear();
    snapshot.ancestor = kNoId;
}

std::uint32_t PrefixCacheIndex::retained_ancestor(std::uint32_t index) const noexcept {
    std::uint32_t current = snapshots_[index].ancestor;
    while (current != kNoId && snapshots_[current].superseded) {
        current = snapshots_[current].ancestor;
    }
    return current;
}

void PrefixCacheIndex::collect_retained_dependents(std::uint32_t index,
                                                   std::vector<std::uint32_t>& out) const {
    std::vector<std::uint32_t> stack(snapshots_[index].dependents);
    while (!stack.empty()) {
        const std::uint32_t current = stack.back();
        stack.pop_back();
        if (snapshots_[current].superseded) {
            stack.insert(stack.end(), snapshots_[current].dependents.begin(),
                         snapshots_[current].dependents.end());
        } else {
            out.push_back(current);
        }
    }
}

void PrefixCacheIndex::adjust_retained(std::uint32_t anchor, int delta) {
    for (std::uint32_t current = anchor; current != kNoId; current = nodes_[current].parent) {
        Node& node = nodes_[current];
        if (delta < 0 && node.retained_below == 0) {
            invariant("prefix cache retained snapshot count underflow");
        }
        node.retained_below = delta < 0 ? node.retained_below - 1U : node.retained_below + 1U;
    }
}

void PrefixCacheIndex::revalue_around(std::uint32_t index,
                                      std::span<const std::uint32_t> dependents) {
    update_priority(index);
    for (const std::uint32_t other : dependents) { update_priority(other); }
    const std::uint32_t ancestor = retained_ancestor(index);
    if (ancestor != kNoId) { update_priority(ancestor); }
}

std::uint32_t PrefixCacheIndex::pick_victim(bool host_resident, std::uint32_t protect) const {
    std::uint32_t superseded = kNoId;
    std::uint32_t retained   = kNoId;
    for (std::uint32_t i = 0; i < snapshots_.size(); ++i) {
        const Snapshot& snapshot = snapshots_[i];
        if (!snapshot.occupied || i == protect || held(snapshot) || snapshot.hold != kNoHold ||
            (host_resident ? snapshot.host != CopyState::Resident
                           : snapshot.host == CopyState::Filling)) {
            continue;
        }
        if (snapshot.superseded) {
            if (superseded == kNoId ||
                snapshot.superseded_tick < snapshots_[superseded].superseded_tick) {
                superseded = i;
            }
        } else if (retained == kNoId || snapshot.priority < snapshots_[retained].priority ||
                   (snapshot.priority == snapshots_[retained].priority &&
                    snapshot.last_hit_tick < snapshots_[retained].last_hit_tick)) {
            retained = i;
        }
    }
    return superseded != kNoId ? superseded : retained;
}

void PrefixCacheIndex::remove_snapshot(std::uint32_t index) {
    Snapshot& snapshot = snapshots_[index];
    if (!snapshot.occupied) { invariant("prefix cache removes a free snapshot"); }
    if (held(snapshot)) { invariant("prefix cache removes a pinned snapshot"); }
    // Snapshots valued against this one are revalued once it is gone.
    if (snapshot.host == CopyState::Filling || snapshot.ssd == CopyState::Filling) {
        invariant("prefix cache removes a snapshot under a transfer");
    }
    std::vector<std::uint32_t> dependents;
    std::uint32_t retained_above = kNoId;
    const bool retained          = !snapshot.superseded;
    // Everything that allocates comes before the first change (the retained dependents here, the room
    // unlink_snapshot hands the dependents over into): an allocation failing under the memory pressure a
    // Host shrink answers leaves the snapshot whole.
    if (retained) { collect_retained_dependents(index, dependents); }
    {
        std::vector<std::uint32_t>& siblings = dependents_of(snapshot.ancestor);
        siblings.reserve(siblings.size() + snapshot.dependents.size());
    }
    if (retained) {
        retained_above = retained_ancestor(index);
        adjust_retained(snapshot.anchor, -1);
    }
    // Out of the warm counts and the clean slab count while its anchor and slabs are still known.
    if (snapshot.warm_counted) {
        snapshot.warm_counted = false;
        if (snapshot.anchor != kNoId) { adjust_warm(snapshot.anchor, -1); }
    }
    if (snapshot.clean_counted) {
        snapshot.clean_counted = false;
        host_clean_image_slabs_ -= snapshot_slab_count(snapshot);
    }
    if (snapshot.ssd == CopyState::Resident && !snapshot.ssd_read && !snapshot.superseded) {
        ++kind_stats(snapshot).dropped_unread;
    }
    for (const std::uint32_t slab : snapshot.ssd_slabs) { free_ssd_slab(slab); }
    snapshot.ssd_slabs.clear();
    snapshot.ssd        = CopyState::Absent;
    snapshot.ssd_read   = false;
    snapshot.ssd_packed = false;
    unlink_snapshot(index);
    snapshot.superseded      = false;
    snapshot.superseded_tick = 0;
    ++snapshot_epoch_;
    const bool held = snapshot.hold != kNoHold;
    if (held) {
        // Every unheld victim was gone: the queued request resumes from an earlier source.
        holds_[snapshot.hold] = kNoId;
        snapshot.hold         = kNoHold;
        ++counters_.held_snapshot_losses;
    }
    backend_->release_snapshot(ref_of_snapshot(index));
    std::vector<std::uint32_t>& anchored =
        snapshot.anchor != kNoId ? nodes_[snapshot.anchor].snapshots : root_snapshots_;
    anchored.erase(std::find(anchored.begin(), anchored.end(), index));
    if (snapshot.device_slot != kNoId) {
        set_slot(snapshot.device_slot, SlotState::Free);
        slot_owner_[snapshot.device_slot] = kNoId;
        snapshot.device_slot              = kNoId;
    }
    for (const std::uint32_t slab : snapshot.host_slabs) { free_slab(slab); }
    snapshot.host_slabs.clear();
    snapshot.host                   = CopyState::Absent;
    const std::uint32_t tail_device = snapshot.tail_device;
    const bool release_tail         = snapshot.tail_device_copy == CopyState::Resident;
    snapshot.tail_device            = kNoId;
    snapshot.tail_device_copy       = CopyState::Absent;
    snapshot.occupied               = false;
    lru_remove(config_.max_nodes + index);
    const std::uint32_t anchor = snapshot.anchor;
    snapshot.anchor            = kNoId;
    ++snapshot.generation;
    free_snapshots_.push_back(index);
    --snapshot_count_;
    if (anchor != kNoId) { adjust_live(anchor, -1); }
    if (release_tail) { backend_->release_device_block(tail_device); }
    for (const std::uint32_t other : dependents) { update_priority(other); }
    if (retained_above != kNoId) { update_priority(retained_above); }
    // A retained snapshot on another branch that shared path blocks with this one has them to itself again.
    if (retained) { revalue_sibling(anchor, kNoId, 0); }
    if (held) { refresh_holds(); }
}

std::uint32_t PrefixCacheIndex::retained_frontier_above(std::uint32_t anchor, std::uint32_t frontier,
                                                        std::span<const TokenId> tail) const noexcept {
    // As find_ancestor, for a snapshot not yet in the index and past superseded ones.
    std::uint32_t previous = kNoId;
    std::uint32_t current  = anchor;
    for (;;) {
        const std::vector<std::uint32_t>& anchored = current != kNoId ? nodes_[current].snapshots : root_snapshots_;
        // Tokens right after this node on the path: the new snapshot's own tail at its anchor.
        const TokenId* next = previous == kNoId ? tail.data() : nodes_[previous].tokens.data();
        std::uint32_t best  = 0;
        for (const std::uint32_t other : anchored) {
            const Snapshot& candidate = snapshots_[other];
            if (candidate.superseded || candidate.frontier >= frontier || candidate.frontier <= best ||
                !std::equal(candidate.tail.begin(), candidate.tail.begin() + candidate.tail_len, next)) {
                continue;
            }
            best = candidate.frontier;
        }
        if (best != 0 || current == kNoId) { return best; }
        previous = current;
        current  = nodes_[current].parent;
    }
}

double PrefixCacheIndex::lowest_retained_priority() const noexcept {
    double lowest = std::numeric_limits<double>::infinity();
    for (const Snapshot& snapshot : snapshots_) {
        if (snapshot.occupied && !snapshot.superseded) { lowest = std::min(lowest, snapshot.priority); }
    }
    return lowest;
}

void PrefixCacheIndex::release_node_storage(std::uint32_t index) {
    Node& node = nodes_[index];
    if (held(node) || node.device == CopyState::Filling || node.host == CopyState::Filling ||
        node.ssd == CopyState::Filling) {
        invariant("prefix cache removes a pinned or transferring node");
    }
    if (node.host == CopyState::Resident) { free_slab(node.host_slab); }
    node.host = CopyState::Absent;
    set_node_slab(index, kNoId);
    if (node.ssd == CopyState::Resident) { free_ssd_slab(node.ssd_slab); }
    node.ssd      = CopyState::Absent;
    node.ssd_slab = kNoId;
    if (node.clean_counted) {
        node.clean_counted = false;
        --host_clean_nodes_;
    }
    lru_remove(index);
    dead_remove(index);
    ssd_dead_remove(index);
    clean_remove(index);
    const bool release_device = node.device == CopyState::Resident;
    const std::uint32_t id    = node.device_id;
    set_device(node, CopyState::Absent);
    node.device_id = kNoId;
    if (release_device) { backend_->release_device_block(id); }
}

void PrefixCacheIndex::remove_subtree(std::uint32_t root) {
    std::vector<std::uint32_t> order{root};
    for (std::size_t i = 0; i < order.size(); ++i) {
        for (const std::uint32_t child : nodes_[order[i]].children) { order.push_back(child); }
    }
    for (const std::uint32_t index : order) {
        while (!nodes_[index].snapshots.empty()) {
            remove_snapshot(nodes_[index].snapshots.back());
        }
    }
    // Detach the subtree root from its parent; descendants disappear with it.
    Node& top = nodes_[root];
    std::vector<std::uint32_t>& siblings =
        top.parent != kNoId ? nodes_[top.parent].children : root_children_;
    siblings.erase(std::find(siblings.begin(), siblings.end(), root));
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const std::uint32_t index = *it;
        Node& node                = nodes_[index];
        release_node_storage(index);
        const auto [begin, end] = children_.equal_range(child_key(node.parent, node.hash));
        for (auto entry = begin; entry != end; ++entry) {
            if (entry->second == index) {
                children_.erase(entry);
                break;
            }
        }
        node.children.clear();
        node.occupied       = false;
        node.parent         = kNoId;
        node.live_below     = 0;
        node.retained_below = 0;
        node.warm_below     = 0;
        node.hold           = kNoHold;
        ++node.generation;
        free_nodes_.push_back(index);
        --node_count_;
    }
}

// ---- withdrawal --------------------------------------------------------------------------------

std::uint32_t PrefixCacheIndex::withdraw(Withdrawal& withdrawal) {
    std::uint32_t removed = 0;
    const auto remove_unpinned = [&](std::uint32_t snapshot) {
        if (held(snapshots_[snapshot])) { return false; }
        remove_snapshot(snapshot);
        ++removed;
        return true;
    };
    std::erase_if(withdrawal.snapshots,
                  [&](SnapshotRef ref) { return !valid(ref) || remove_unpinned(ref.index); });
    std::vector<std::uint32_t> order;
    std::vector<std::uint32_t> anchored;
    std::erase_if(withdrawal.nodes, [&](NodeRef ref) {
        if (!valid(ref)) { return true; } // evicted, or inside a subtree removed before
        order.assign(1, ref.index);
        for (std::size_t i = 0; i < order.size(); ++i) {
            for (const std::uint32_t child : nodes_[order[i]].children) { order.push_back(child); }
        }
        // Pins are path-wide, so a pinned node anywhere below pins the subtree's root.
        bool pinned = held(nodes_[ref.index]);
        for (const std::uint32_t index : order) {
            anchored = nodes_[index].snapshots; // removal edits the node's list
            for (const std::uint32_t snapshot : anchored) {
                if (!remove_unpinned(snapshot)) { pinned = true; }
            }
        }
        if (pinned) { return false; }
        removed += static_cast<std::uint32_t>(order.size());
        remove_subtree(ref.index);
        return true;
    });
    return removed;
}

// ---- views -------------------------------------------------------------------------------------

NodeView PrefixCacheIndex::node(NodeRef ref) const {
    const Node& node = require(ref);
    NodeView view;
    view.depth                = node.depth;
    view.parent               = ref_of_node(node.parent);
    view.device_id            = node.device_id;
    view.device               = node.device;
    view.host_slab            = node.host_slab;
    view.host                 = node.host;
    view.ssd_slab             = node.ssd_slab;
    view.ssd                  = node.ssd;
    view.pins                 = node.pins;
    view.device_pins          = node.device_pins;
    view.evictable            = lru_links_[ref.index].list != kNoList;
    view.transfer_held        = node.transfer_counted;
    view.live_snapshots_below = node.live_below;
    return view;
}

SnapshotView PrefixCacheIndex::snapshot(SnapshotRef ref) const {
    const Snapshot& snapshot = require(ref);
    SnapshotView view;
    view.anchor           = ref_of_node(snapshot.anchor);
    view.frontier         = snapshot.frontier;
    view.tail_len         = snapshot.tail_len;
    view.tail_device      = snapshot.tail_device;
    view.tail_device_copy = snapshot.tail_device_copy;
    view.device_slot      = snapshot.device_slot;
    view.host             = snapshot.host;
    view.host_slabs       = snapshot.host_slabs;
    view.ssd              = snapshot.ssd;
    view.ssd_slabs        = snapshot.ssd_slabs;
    view.ssd_packed       = snapshot.ssd_packed;
    view.pins             = snapshot.pins;
    view.kind             = snapshot.kind;
    view.hits             = snapshot.hits;
    view.superseded       = snapshot.superseded;
    view.priority         = snapshot.priority;
    view.tail             =std::span<const TokenId>(snapshot.tail.data(), snapshot.tail_len);
    return view;
}

BlockIdentity PrefixCacheIndex::block_identity(NodeRef ref) const {
    const Node& node = require(ref);
    return BlockIdentity{
        .parent      = ref_of_node(node.parent),
        .lookup_hash = node.hash,
        .extra       = node.extra,
        .tokens      = std::span<const TokenId>(node.tokens.data(), node.tokens.size()),
    };
}

std::optional<SnapshotRef> PrefixCacheIndex::slot_owner(std::uint32_t slot) const {
    if (slot >= slot_state_.size() || slot_state_[slot] != SlotState::Owned) {
        return std::nullopt;
    }
    return ref_of_snapshot(slot_owner_[slot]);
}

PrefixIndexStats PrefixCacheIndex::stats() const noexcept {
    PrefixIndexStats stats        = counters_;
    stats.nodes                   = node_count_;
    stats.snapshots               = snapshot_count_;
    stats.host_free_slabs         = static_cast<std::uint32_t>(free_slabs_.size());
    stats.host_slab_limit         = host_limit_;
    stats.gdsf_inflation          = inflation_;
    stats.device_evictable_blocks = device_evictable_blocks();
    stats.device_resident_blocks  = device_resident_;
    stats.free_device_slots       = free_slots_;
    stats.held_snapshots          = static_cast<std::uint32_t>(
        std::count_if(holds_.begin(), holds_.end(), [](std::uint32_t held) { return held != kNoId; }));
    stats.ssd_free_slabs = static_cast<std::uint32_t>(free_ssd_slabs_.size());
    if (config_.ssd_slabs != 0) {
        stats.ssd_only_snapshots = static_cast<std::uint32_t>(
            std::count_if(snapshots_.begin(), snapshots_.end(), [](const Snapshot& snapshot) {
                return snapshot.occupied && !snapshot.warm_counted;
            }));
    }
    return stats;
}

} // namespace infernix::runtime::prefix_cache
