#include "runtime/prefix_cache/block_hash.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::runtime::prefix_cache;

namespace {

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

template <class F>
void require_throws(F&& f, const std::string& message) {
    try {
        f();
    } catch (const std::exception&) { return; }
    throw std::runtime_error("expected an exception: " + message);
}

class RecordingBackend final : public PrefixIndexBackend {
public:
    void release_device_block(std::uint32_t id) noexcept override {
        released.push_back(id);
        live.erase(id);
    }

    std::uint32_t allocate() {
        const std::uint32_t id = next++;
        live.insert(id);
        return id;
    }

    std::vector<std::uint32_t> released;
    std::set<std::uint32_t> live;
    std::uint32_t next = 1;
};

PrefixIndexConfig small_config(std::uint32_t host_slabs = 64, std::uint32_t slots = 2) {
    PrefixIndexConfig config;
    config.max_nodes                 = 256;
    config.max_snapshots             = 32;
    config.host_slabs                = host_slabs;
    config.image_slabs               = 4;
    config.device_snapshot_slots     = slots;
    config.block_bytes               = 1U << 20U;
    config.image_bytes               = 4U << 20U;
    config.cost.token_seconds        = 1.0e-4;
    config.cost.h2d_bytes_per_second = 50.0e9;
    return config;
}

std::vector<TokenId> make_tokens(std::uint32_t count, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<TokenId> tokens(count);
    for (TokenId& token : tokens) { token = static_cast<TokenId>(rng() % 50000U); }
    return tokens;
}

// Inserts every full block of `tokens` below the root as one sequence would, returning its path.
std::vector<NodeRef> insert_sequence(PrefixCacheIndex& index, RecordingBackend& backend,
                                     const std::vector<TokenId>& tokens) {
    const auto hashes = block_lookup_hashes(tokens, {});
    std::vector<NodeRef> path;
    NodeRef parent;
    for (std::size_t b = 0; b < hashes.size(); ++b) {
        const std::span<const TokenId> block(tokens.data() + b * kBlockTokens, kBlockTokens);
        const std::uint32_t id    = backend.allocate();
        const InsertResult result =
            index.insert_block(parent, hashes[b], block, 0, id, true);
        require(result.inserted == result.device_attached || !result.inserted,
                "a new node must own its device id");
        if (!result.device_attached) { backend.release_device_block(id); }
        if (result.replaced_device_id != kNoId) {
            backend.release_device_block(result.replaced_device_id);
        }
        path.push_back(result.node);
        parent = result.node;
    }
    index.check_invariants();
    return path;
}

SnapshotRef publish_tap(PrefixCacheIndex& index, NodeRef anchor) {
    const auto slot = index.acquire_device_slot();
    require(slot.has_value(), "no device slot for a tap");
    const std::uint32_t frontier = (index.node(anchor).depth + 1U) * kBlockTokens;
    const PublishResult result =
        index.publish_snapshot(anchor, frontier, {}, std::nullopt, *slot, SnapshotKind::Tap);
    require(result.created, "tap snapshot not created");
    index.check_invariants();
    return result.snapshot;
}

// Publishes a snapshot at `frontier` of `tokens`, whose full blocks are `path`; a frontier inside
// a block takes its tail from that block's tokens.
SnapshotRef publish_at(PrefixCacheIndex& index, RecordingBackend& backend,
                       const std::vector<NodeRef>& path, const std::vector<TokenId>& tokens,
                       std::uint32_t frontier, SnapshotKind kind) {
    const std::uint32_t full = frontier / kBlockTokens;
    const std::uint32_t tail = frontier % kBlockTokens;
    const auto slot          = index.acquire_device_slot();
    require(slot.has_value(), "no device slot for a snapshot");
    std::optional<std::uint32_t> tail_id;
    if (tail != 0) { tail_id = backend.allocate(); }
    const PublishResult result = index.publish_snapshot(
        full == 0 ? NodeRef{} : path[full - 1], frontier,
        std::span<const TokenId>(tokens.data() + static_cast<std::size_t>(full) * kBlockTokens,
                                 tail),
        tail_id, *slot, kind);
    require(result.created, "snapshot not created");
    index.check_invariants();
    return result.snapshot;
}

void backup_node(PrefixCacheIndex& index, NodeRef node) {
    index.pin_node(node);
    const auto slab = index.begin_host_fill(node);
    require(slab.has_value(), "no slab for a block backup");
    index.complete_host_fill(node);
    index.unpin_node(node);
    index.check_invariants();
}

void backup_snapshot(PrefixCacheIndex& index, SnapshotRef snapshot) {
    index.pin_snapshot(snapshot);
    require(index.begin_snapshot_host_fill(snapshot), "no slabs for a snapshot backup");
    index.complete_snapshot_host_fill(snapshot);
    index.unpin_snapshot(snapshot);
    index.check_invariants();
}

void test_hashes() {
    const auto tokens = make_tokens(256, 1);
    const auto a      = block_lookup_hashes(tokens, {});
    const auto b      = block_lookup_hashes(tokens, {});
    require(a.size() == 4 && a == b, "block hashes are not deterministic");
    auto changed = tokens;
    changed[130] ^= 1;
    const auto c = block_lookup_hashes(changed, {});
    require(c[0] == a[0] && c[1] == a[1] && c[2] != a[2] && c[3] != a[3],
            "a token change must change its block hash and every later chained hash");
    const std::vector<std::uint64_t> extras{0, 7, 0, 0};
    const auto d = block_lookup_hashes(tokens, extras);
    require(d[0] == a[0] && d[1] != a[1], "the extra key must change the block hash");
    require_throws([&] { (void)block_lookup_hashes(tokens, std::vector<std::uint64_t>{1}); },
                   "short extras");
}

void test_match_and_choice() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(), backend);
    const auto tokens = make_tokens(64 * 8 + 10, 2);
    const std::vector<TokenId> first(tokens.begin(), tokens.begin() + 64 * 6);
    const auto path        = insert_sequence(index, backend, first);
    const SnapshotRef tap2 = publish_tap(index, path[1]); // frontier 128
    const SnapshotRef tap5 = publish_tap(index, path[4]); // frontier 320
    index.release_path(path);
    index.check_invariants();

    const auto hashes = block_lookup_hashes(tokens, {});
    const MatchResult match =
        index.match(tokens, hashes, {}, static_cast<std::uint32_t>(tokens.size()));
    require(match.path.size() == 6, "match did not follow every cached block");
    require(match.candidates.size() == 2 && match.candidates[0].snapshot == tap5 &&
                match.candidates[1].snapshot == tap2,
            "candidates are not deepest first");
    require(match.candidates[0].restore_bytes == 0 && match.candidates[0].image_on_device,
            "device-resident candidate should need no restore");
    const AdmissionChoice choice = index.choose(match, static_cast<std::uint32_t>(tokens.size()));
    require(choice.candidate == 0U, "the deepest device-resident snapshot should be chosen");

    // A prompt that ends exactly at the tap frontier cannot use it (at least one token prefills).
    const MatchResult exact = index.match(tokens, hashes, {}, 320);
    require(exact.candidates.size() == 1 && exact.candidates[0].snapshot == tap2,
            "a snapshot at the prompt end must not be offered");
    require(index.match(tokens, hashes, {}, 321).candidates[0].snapshot == tap5,
            "a snapshot one token before the prompt end must be offered");

    // A different token in block 3 stops the path at block 2 and hides tap5.
    auto edited = tokens;
    edited[64 * 3 + 5] += 1;
    const auto edited_hashes = block_lookup_hashes(edited, {});
    const MatchResult partial =
        index.match(edited, edited_hashes, {}, static_cast<std::uint32_t>(edited.size()));
    require(partial.path.size() == 3 && partial.candidates.size() == 1 &&
                partial.candidates[0].snapshot == tap2,
            "an edit must fall back to the snapshot before it");

    // The extra key (media identity) distinguishes identical tokens.
    std::vector<std::uint64_t> extras(hashes.size(), 0);
    extras[0]               = 99;
    const auto media_hashes = block_lookup_hashes(tokens, extras);
    require(index.match(tokens, media_hashes, extras, static_cast<std::uint32_t>(tokens.size()))
                .path.empty(),
            "different media must not match");
}

void test_collisions_and_tails() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(), backend);
    const auto a = make_tokens(64, 3);
    const auto b = make_tokens(64, 4);
    // Force the same lookup hash for two different blocks: exact comparison must separate them.
    const InsertResult first  = index.insert_block({}, 42, a, 0, backend.allocate(), true);
    const InsertResult second = index.insert_block({}, 42, b, 0, backend.allocate(), true);
    require(first.inserted && second.inserted && !(first.node == second.node),
            "colliding blocks must be distinct nodes");
    require(index.find_child({}, 42, a, 0) == first.node &&
                index.find_child({}, 42, b, 0) == second.node,
            "collision lookup returned the wrong node");
    require(!index.find_child({}, 42, a, 1).has_value(), "extra key mismatch must miss");
    index.check_invariants();

    // Endpoint with a tail: frontier 64 + 10 anchored at `first`.
    std::vector<TokenId> prompt = a;
    const auto tail_tokens      = make_tokens(10, 5);
    prompt.insert(prompt.end(), tail_tokens.begin(), tail_tokens.end());
    prompt.push_back(7);
    prompt.push_back(8);
    const auto slot             = index.acquire_device_slot();
    const std::uint32_t tail_id = backend.allocate();
    const PublishResult endpoint =
        index.publish_snapshot(first.node, 74, tail_tokens, tail_id, *slot, SnapshotKind::Endpoint);
    require(endpoint.created, "endpoint snapshot not created");
    index.check_invariants();

    std::vector<std::uint64_t> hashes{42};
    const MatchResult match =
        index.match(prompt, hashes, {}, static_cast<std::uint32_t>(prompt.size()));
    require(match.candidates.size() == 1 && match.candidates[0].frontier == 74 &&
                match.candidates[0].tail && match.candidates[0].tail_on_device,
            "tail snapshot did not match its exact tail");
    auto other_tail = prompt;
    other_tail[70] += 1;
    require(index.match(other_tail, hashes, {}, static_cast<std::uint32_t>(prompt.size()))
                .candidates.empty(),
            "a different tail token must not match");

    // Duplicate publication frees the new slot and tail and returns the existing snapshot.
    const auto slot2              = index.acquire_device_slot();
    const std::uint32_t tail_copy = backend.allocate();
    const PublishResult duplicate = index.publish_snapshot(first.node, 74, tail_tokens, tail_copy,
                                                           *slot2, SnapshotKind::Endpoint);
    require(!duplicate.created && duplicate.snapshot == endpoint.snapshot,
            "duplicate snapshot publication was not deduplicated");
    require(backend.released.back() == tail_copy, "duplicate tail was not released");
    require(index.stats().free_device_slots == 1, "duplicate staging slot was not freed");
    require_throws(
        [&] {
            const auto s = index.acquire_device_slot();
            (void)index.publish_snapshot(first.node, 75, tail_tokens, backend.allocate(), *s,
                                         SnapshotKind::Tap);
        },
        "frontier mismatch");
    index.check_invariants();
}

void test_device_lru_order() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(0), backend);
    const auto tokens = make_tokens(64 * 4, 6);
    const auto path   = insert_sequence(index, backend, tokens);
    (void)publish_tap(index, path[3]);
    require(index.device_evictable_blocks() == 0, "pinned blocks must not be evictable");
    index.release_path(path);
    index.check_invariants();
    require(index.device_evictable_blocks() == 4, "released blocks must become evictable");

    // Unbacked: evicting the deepest block first loses only that node and its snapshot.
    const std::uint32_t deepest = index.node(path[3]).device_id;
    require(index.evict_device_blocks(1) == 1, "one block should be released");
    require(backend.released.back() == deepest, "deepest block must be evicted first");
    require(!index.valid(path[3]) && index.valid(path[2]), "only the deepest node is lost");
    require(index.stats().snapshots == 0, "the snapshot on a lost node must be removed");
    index.check_invariants();

    // Re-pinning a path protects it.
    const std::vector<NodeRef> prefix(path.begin(), path.begin() + 3);
    index.acquire_path(prefix);
    require(index.evict_device_blocks(10) == 0, "pinned path must not be evicted");
    index.release_path(prefix);
    require(index.evict_device_blocks(10) == 3, "whole released path must be evictable");
    require(index.stats().nodes == 0 && backend.live.empty(), "every page must be released");
    index.check_invariants();
}

void test_backed_before_unbacked() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(64), backend);
    const auto a             = make_tokens(64 * 2, 7);
    const auto b             = make_tokens(64 * 2, 8);
    const auto path_a        = insert_sequence(index, backend, a);
    const auto path_b        = insert_sequence(index, backend, b);
    const SnapshotRef snap_a = publish_tap(index, path_a[1]);
    const SnapshotRef snap_b = publish_tap(index, path_b[1]);
    backup_node(index, path_b[0]);
    backup_node(index, path_b[1]);
    backup_snapshot(index, snap_b);
    index.release_path(path_a); // older: unbacked
    index.release_path(path_b); // newer: backed
    index.check_invariants();
    require(index.evict_device_blocks(2) == 2, "two backed blocks should be released");
    require(index.valid(path_b[0]) && index.node(path_b[0]).device == CopyState::Absent &&
                index.node(path_b[0]).host == CopyState::Resident,
            "backed block must survive as host-only");
    require(index.node(path_a[0]).device == CopyState::Resident,
            "unbacked blocks must be evicted only after backed ones");
    require(index.valid(snap_a) && index.valid(snap_b), "no snapshot should be lost");

    // The host-only path is still a candidate and reports its restore bytes.
    const auto hashes = block_lookup_hashes(b, {});
    auto prompt       = b;
    prompt.push_back(1);
    const MatchResult match =
        index.match(prompt, hashes, {}, static_cast<std::uint32_t>(prompt.size()));
    require(match.candidates.size() == 1 && match.candidates[0].host_only_blocks == 2,
            "host-only path blocks must be counted");
    require(match.candidates[0].restore_bytes == 2U * (1U << 20U),
            "restore bytes must cover host-only blocks (image still on device)");

    // Restore one block to device.
    index.acquire_path(match.path);
    const std::uint32_t id = backend.allocate();
    index.begin_device_fill(match.path[0], id);
    require(index.match(prompt, hashes, {}, static_cast<std::uint32_t>(prompt.size()))
                    .candidates[0]
                    .filling_blocks == 1,
            "a filling block must be reported");
    index.complete_device_fill(match.path[0]);
    index.release_path(match.path);
    index.check_invariants();
}

void test_host_dead_and_gdsf() {
    RecordingBackend backend;
    // 12 slabs: images need 4 slabs each.
    PrefixCacheIndex index(small_config(12, 3), backend);
    const auto dead_tokens = make_tokens(64 * 4, 9);
    const auto dead_path   = insert_sequence(index, backend, dead_tokens);
    for (const NodeRef node : dead_path) { backup_node(index, node); }
    index.release_path(dead_path); // no snapshot: dead KV
    index.check_invariants();

    const auto a             = make_tokens(64 * 1, 10);
    const auto path_a        = insert_sequence(index, backend, a);
    const SnapshotRef snap_a = publish_tap(index, path_a[0]);
    // Allocating 4 slabs (8 free) needs no reclaim; the next ones reclaim dead blocks first.
    backup_snapshot(index, snap_a);
    require(index.stats().host_dead_reclaims == 0, "no reclaim needed yet");
    index.release_path(path_a);

    const auto b             = make_tokens(64 * 1, 11);
    const auto path_b        = insert_sequence(index, backend, b);
    const SnapshotRef snap_b = publish_tap(index, path_b[0]);
    backup_snapshot(index, snap_b);
    require(index.stats().host_dead_reclaims == 0, "4 slabs were still free");
    backup_node(index, path_b[0]);
    require(index.stats().host_dead_reclaims == 1, "dead KV must be reclaimed before snapshots");
    require(index.valid(snap_a) && index.valid(snap_b),
            "no snapshot should be evicted for dead KV");
    index.release_path(path_b);

    // Hit b so it is more valuable; a new snapshot must evict a (the GDSF minimum).
    index.note_hit(snap_b);
    const auto c             = make_tokens(64 * 1, 12);
    const auto path_c        = insert_sequence(index, backend, c);
    const SnapshotRef snap_c = publish_tap(index, path_c[0]);
    const double before      = index.stats().gdsf_inflation;
    // Consume the remaining dead nodes first, then a snapshot's host copy must go.
    backup_snapshot(index, snap_c);
    require(index.stats().host_dead_reclaims == 4, "all dead nodes must go before snapshots");
    require(index.stats().host_snapshot_evictions == 1, "one snapshot host copy must be evicted");
    // a is the GDSF minimum; it is still complete on the Device, so only its host copy goes.
    require(index.valid(snap_a) && index.snapshot(snap_a).host == CopyState::Absent &&
                index.snapshot(snap_a).device_slot != kNoId,
            "the least valuable snapshot must yield its host copy and keep its device image");
    require(index.valid(snap_b) && index.snapshot(snap_b).host == CopyState::Resident &&
                index.valid(snap_c) && index.snapshot(snap_c).host == CopyState::Resident,
            "more valuable snapshots keep their host copies");
    require(index.stats().gdsf_inflation >= before, "GDSF inflation must be monotonic");

    // Host pressure left a Device-only: it is lost when its slot is taken, so it goes before a
    // backed owner gives up its Device copy, for a snapshot worth as much.
    const auto slot = index.acquire_device_slot();
    require(slot.has_value() && !index.valid(snap_a) && index.valid(snap_b) &&
                index.snapshot(snap_b).device_slot != kNoId &&
                index.snapshot(snap_c).device_slot != kNoId,
            "the Device-only snapshot must lose its slot before a backed one");
    // With no free or Device-only slot left, the least recently hit backed owner keeps only its
    // Host copy.
    const auto backed_slot = index.acquire_device_slot(0.0);
    require(backed_slot.has_value() && index.valid(snap_b) && index.valid(snap_c) &&
                (index.snapshot(snap_b).device_slot == kNoId ||
                 index.snapshot(snap_c).device_slot == kNoId),
            "a backed device slot must be reclaimable whatever the claim");
    index.release_device_slot(*slot);
    index.release_device_slot(*backed_slot);
    index.release_path(path_c);
    index.check_invariants();
}

void test_host_only_reattach() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(64, 2), backend);
    const auto tokens      = make_tokens(64 * 3, 30);
    const auto path        = insert_sequence(index, backend, tokens);
    const SnapshotRef snap = publish_tap(index, path[2]);
    for (const NodeRef node : path) { backup_node(index, node); }
    backup_snapshot(index, snap);
    index.release_path(path);
    require(index.evict_device_blocks(3) == 3, "backed blocks must be released");
    for (const NodeRef node : path) {
        require(index.node(node).device == CopyState::Absent, "block must be host-only");
    }
    const std::size_t live_before = backend.live.size();

    // A sequence recomputing the same blocks re-establishes their Device copies.
    const auto again = insert_sequence(index, backend, tokens);
    require(again == path, "recomputed blocks must resolve to the existing nodes");
    require(backend.live.size() == live_before + 3, "the index must keep the recomputed pages");
    for (const NodeRef node : path) {
        const NodeView view = index.node(node);
        require(view.device == CopyState::Resident && view.host == CopyState::Resident,
                "a reattached block is resident on both tiers");
    }
    // Duplicates of Device copies another sequence pins stay private to the inserting sequence.
    const auto twice = insert_sequence(index, backend, tokens);
    require(backend.live.size() == live_before + 3, "a pinned resident duplicate must not be retained");
    index.release_path(twice);
    index.release_path(again);
    index.check_invariants();
    require(index.evict_device_blocks(3) == 3, "reattached backed blocks are evictable again");
    index.check_invariants();
}

// A sequence recomputing cached blocks (a resume from a shallower snapshot) must not hold two pages
// per block: its page replaces an unpinned Device copy, which returns to the pool, while a copy
// another sequence pins stays and the sequence keeps its page private. Before this, every
// recomputed block pinned its evictable copy beside the sequence's page, and a pool sized to one
// context ran out of pages for the next unit.
void test_resident_duplicate_replacement() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(64, 1), backend);
    const auto tokens = make_tokens(64 * 3, 62);
    const auto path   = insert_sequence(index, backend, tokens);
    backup_node(index, path[1]);
    index.release_path(path);
    require(index.device_evictable_blocks() == 3 && backend.live.size() == 3,
            "the cached blocks are evictable Device copies");
    std::vector<std::uint32_t> old_ids;
    for (const NodeRef node : path) { old_ids.push_back(index.node(node).device_id); }
    const std::array<NodeRef, 1> reader{path[0]};
    index.acquire_path(reader); // another sequence maps block 0

    const std::uint32_t first_new = backend.next;
    const auto again              = insert_sequence(index, backend, tokens);
    require(again == path, "recomputed blocks resolve to the existing nodes");
    require(index.node(path[0]).device_id == old_ids[0] && backend.live.contains(old_ids[0]) &&
                !backend.live.contains(first_new),
            "a pinned Device copy stays and the sequence's page stays private");
    for (std::uint32_t b = 1; b < 3; ++b) {
        const NodeView view = index.node(path[b]);
        require(view.device == CopyState::Resident && view.device_id == first_new + b &&
                    view.pins == 1,
                "an unpinned Device copy is replaced by the sequence's page");
        require(!backend.live.contains(old_ids[b]), "the replaced copy returns to the pool");
    }
    require(index.node(path[1]).host == CopyState::Resident,
            "a replaced block keeps its Host copy");
    require(backend.live.size() == 3, "one page per block while the sequence holds the path");
    require(index.device_evictable_blocks() == 0, "the sequence pins its path");
    index.check_invariants();

    index.release_path(again);
    index.release_path(reader);
    require(index.device_evictable_blocks() == 3, "the replaced blocks are evictable again");
    require(index.evict_device_blocks(3) == 3 && backend.live.empty(),
            "evicting the blocks releases the adopted pages");
    index.check_invariants();
}

// A saved Host tier keeps exactly the Host-backed snapshots whose paths are Host-resident, and a
// fresh index rebuilt from it matches the same prompts through host-only blocks.
void test_persistence_roundtrip() {
    RecordingBackend backend;
    PrefixCacheIndex source(small_config(64, 2), backend);
    const auto tokens = make_tokens(64 * 3, 40);
    const auto path   = insert_sequence(source, backend, tokens);
    for (const NodeRef node : path) { backup_node(source, node); }
    const SnapshotRef backed = publish_tap(source, path[1]);
    backup_snapshot(source, backed);
    const auto tail_tokens      = make_tokens(10, 41);
    const auto slot             = source.acquire_device_slot();
    const std::uint32_t tail_id = backend.allocate();
    const PublishResult endpoint =
        source.publish_snapshot(path[2], 202, tail_tokens, tail_id, *slot, SnapshotKind::Endpoint);
    backup_snapshot(source, endpoint.snapshot);
    // A Device-only snapshot on an unbacked path is not persistable.
    const auto other      = make_tokens(64, 42);
    const auto other_path = insert_sequence(source, backend, other);
    const auto other_slot = source.acquire_device_slot();
    require(other_slot.has_value(), "no slot for an unbacked snapshot");
    (void)source.publish_snapshot(other_path[0], 64, {}, std::nullopt, *other_slot,
                                  SnapshotKind::Tap);
    source.release_path(path);
    source.release_path(other_path);

    std::vector<NodeRef> nodes;
    std::vector<SnapshotRef> snapshots;
    source.collect_persistable(nodes, snapshots);
    require(nodes.size() == 3 && snapshots.size() == 2,
            "only Host-backed snapshots and their paths are persistable");

    PrefixCacheIndex restored(small_config(64, 2), backend);
    std::vector<std::pair<NodeRef, NodeRef>> mapping;
    const auto mapped = [&](NodeRef ref) {
        for (const auto& [from, to] : mapping) {
            if (from == ref) { return to; }
        }
        return NodeRef{};
    };
    for (const NodeRef node : nodes) {
        const BlockIdentity identity = source.block_identity(node);
        const auto block             = restored.restore_host_block(
            mapped(identity.parent), identity.lookup_hash, identity.tokens, identity.extra);
        require(block.has_value(), "a persisted block did not restore");
        mapping.emplace_back(node, block->node);
    }
    for (const SnapshotRef snapshot : snapshots) {
        const SnapshotView view = source.snapshot(snapshot);
        require(restored
                    .restore_host_snapshot(mapped(view.anchor), view.frontier, view.tail, view.kind,
                                           view.hits)
                    .has_value(),
                "a persisted snapshot did not restore");
    }
    restored.check_invariants();

    auto prompt = tokens;
    prompt.insert(prompt.end(), tail_tokens.begin(), tail_tokens.end());
    prompt.push_back(3);
    const auto hashes = block_lookup_hashes(prompt, {});
    const MatchResult match =
        restored.match(prompt, hashes, {}, static_cast<std::uint32_t>(prompt.size()));
    require(match.candidates.size() == 2 && match.candidates[0].frontier == 202 &&
                match.candidates[0].host_only_blocks == 3 && !match.candidates[0].image_on_device,
            "a restored Host tier must match through host-only blocks and images");

    // A full Host tier restores what fits and nothing else.
    PrefixCacheIndex tiny(small_config(1, 2), backend);
    const BlockIdentity first = source.block_identity(nodes.front());
    require(tiny.restore_host_block(NodeRef{}, first.lookup_hash, first.tokens, first.extra)
                .has_value(),
            "one slab must hold one block");
    const BlockIdentity second = source.block_identity(nodes[1]);
    require(!tiny.restore_host_block(NodeRef{}, second.lookup_hash, second.tokens, second.extra)
                 .has_value(),
            "restoring must not evict to make room");
    tiny.check_invariants();
}

void test_tail_device_fill() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(64, 2), backend);
    const auto tokens = make_tokens(64, 31);
    const auto path   = insert_sequence(index, backend, tokens);
    backup_node(index, path[0]);
    const auto tail_tokens      = make_tokens(10, 32);
    const auto slot             = index.acquire_device_slot();
    const std::uint32_t tail_id = backend.allocate();
    const PublishResult endpoint =
        index.publish_snapshot(path[0], 74, tail_tokens, tail_id, *slot, SnapshotKind::Endpoint);
    require(endpoint.created, "endpoint not created");
    backup_snapshot(index, endpoint.snapshot);
    require(index.snapshot(endpoint.snapshot).host_slabs.size() == 5,
            "a host-backed tail needs its own slab");
    index.release_path(path);
    // Evict the block and the tail; both are host-backed so the snapshot survives.
    require(index.evict_device_blocks(2) == 2, "block and tail must be evictable");
    require(backend.live.count(tail_id) == 0, "the tail's device page must be released");
    const SnapshotView evicted = index.snapshot(endpoint.snapshot);
    require(index.valid(endpoint.snapshot) && evicted.tail_device_copy == CopyState::Absent,
            "a backed tail survives host-only");

    auto prompt = tokens;
    prompt.insert(prompt.end(), tail_tokens.begin(), tail_tokens.end());
    prompt.push_back(7);
    const auto hashes = block_lookup_hashes(prompt, {});
    const MatchResult match =
        index.match(prompt, hashes, {}, static_cast<std::uint32_t>(prompt.size()));
    require(match.candidates.size() == 1 && match.candidates[0].tail &&
                !match.candidates[0].tail_on_device,
            "a host-only tail must be reported");

    // Abort then complete a restore of the tail.
    index.pin_snapshot(endpoint.snapshot);
    require_throws([&] { index.complete_tail_device_fill(endpoint.snapshot); }, "no fill open");
    const std::uint32_t first = backend.allocate();
    index.begin_tail_device_fill(endpoint.snapshot, first);
    index.abort_tail_device_fill(endpoint.snapshot);
    require(backend.live.count(first) == 0, "an aborted tail fill must return its page");
    const std::uint32_t second = backend.allocate();
    index.begin_tail_device_fill(endpoint.snapshot, second);
    index.complete_tail_device_fill(endpoint.snapshot);
    index.unpin_snapshot(endpoint.snapshot);
    index.check_invariants();
    const SnapshotView restored = index.snapshot(endpoint.snapshot);
    require(restored.tail_device_copy == CopyState::Resident && restored.tail_device == second,
            "a completed tail fill is resident");
    require(index.evict_device_blocks(1) == 1 && backend.live.count(second) == 0,
            "a restored tail is evictable again");
    index.check_invariants();
}

void test_device_slots() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(64, 1), backend);
    const auto a             = make_tokens(64, 13);
    const auto path_a        = insert_sequence(index, backend, a);
    const SnapshotRef snap_a = publish_tap(index, path_a[0]);
    require(!index.acquire_device_slot(0.0).has_value(),
            "an unbacked slot must not be taken for a snapshot worth less");
    backup_snapshot(index, snap_a);
    const auto slot = index.acquire_device_slot();
    require(slot.has_value(), "a backed slot must be reusable");
    require(index.valid(snap_a) && index.snapshot(snap_a).device_slot == kNoId,
            "the evicted slot's snapshot must survive host-only");
    index.release_device_slot(*slot);

    // Unbacked with permission: the owner is lost.
    const auto b             = make_tokens(64, 14);
    const auto path_b        = insert_sequence(index, backend, b);
    const SnapshotRef snap_b = publish_tap(index, path_b[0]);
    const auto forced        = index.acquire_device_slot();
    require(forced.has_value() && !index.valid(snap_b), "unbacked slot eviction loses its owner");
    index.release_device_slot(*forced);
    index.release_path(path_a);
    index.release_path(path_b);
    index.check_invariants();
}

void test_supersession() {
    RecordingBackend backend;
    // 20 slabs: a tail-less image takes 4, an image with a tail 5.
    PrefixCacheIndex index(small_config(20, 8), backend);
    const auto tokens = make_tokens(64 * 12 + 1, 30);
    const auto path   = insert_sequence(index, backend, tokens);
    const SnapshotRef shared =
        publish_at(index, backend, path, tokens, 64 * 2, SnapshotKind::Boundary);
    const SnapshotRef first =
        publish_at(index, backend, path, tokens, 64 * 3 + 10, SnapshotKind::Endpoint);
    const SnapshotRef second =
        publish_at(index, backend, path, tokens, 64 * 7 + 20, SnapshotKind::Endpoint);
    // A second branch sharing five blocks continues past the boundary differently: from here on
    // it serves two conversations.
    std::vector<TokenId> other(tokens.begin(), tokens.begin() + 64 * 5);
    const auto suffix = make_tokens(64 * 4 + 1, 31);
    other.insert(other.end(), suffix.begin(), suffix.end());
    const auto other_path = insert_sequence(index, backend, other);
    index.supersede(shared);
    require(!index.snapshot(shared).superseded,
            "a boundary another conversation continued from must not be superseded");
    index.supersede(first);
    require(index.snapshot(first).superseded, "the resumed snapshot must be superseded");
    index.note_hit(first);
    require(!index.snapshot(first).superseded, "a hit must retain a superseded snapshot again");
    // Hits make `first` the most valuable snapshot by GDSF; superseding it must still put it first.
    for (int hit = 0; hit < 4; ++hit) { index.note_hit(first); }
    index.supersede(first);
    index.check_invariants();

    // The second branch hangs below `first`, whose tail lies in a shared block.
    const SnapshotRef branch =
        publish_at(index, backend, other_path, other, 64 * 6 + 3, SnapshotKind::Endpoint);

    for (const SnapshotRef snapshot : {shared, first, second, branch}) {
        backup_snapshot(index, snapshot);
    }
    require(index.stats().host_free_slabs == 1, "the Host tier must be nearly full");
    const SnapshotRef tip = publish_at(index, backend, path, tokens, 64 * 11, SnapshotKind::Tap);
    backup_snapshot(index, tip);
    require(!index.valid(first), "the superseded snapshot must be evicted first, entirely");
    require(index.stats().superseded_evictions == 1 && index.stats().host_snapshot_evictions == 1,
            "exactly one superseded eviction");
    require(index.valid(shared) && index.valid(second) && index.valid(branch) && index.valid(tip),
            "retained snapshots must survive");
    index.release_path(path);
    index.release_path(other_path);
    index.check_invariants();
}

// Without a Host tier a slot's owner is lost when the slot is taken, so value decides (§9.2): a
// short request's snapshot must not push out the one a long conversation resumes from, and a
// snapshot worth less than every owner takes no slot.
void test_unbacked_slot_values() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(0, 2), backend);
    const auto long_tokens = make_tokens(64 * 20, 50);
    const auto long_path   = insert_sequence(index, backend, long_tokens);
    const SnapshotRef long_end =
        publish_at(index, backend, long_path, long_tokens, 64 * 20, SnapshotKind::Endpoint);
    const auto short_tokens = make_tokens(64 * 2, 51);
    const auto short_path   = insert_sequence(index, backend, short_tokens);
    const SnapshotRef short_end =
        publish_at(index, backend, short_path, short_tokens, 64 * 2, SnapshotKind::Endpoint);

    // Another short request's endpoint may take the short endpoint's slot, never the long one's.
    const auto other      = make_tokens(64 * 2, 52);
    const auto other_path = insert_sequence(index, backend, other);
    const auto slot       = index.acquire_device_slot(index.estimate_priority(0, 128, false));
    require(slot.has_value() && index.valid(long_end) && !index.valid(short_end),
            "the least valuable unbacked snapshot must lose its slot first");
    const SnapshotRef other_end =
        index.publish_snapshot(other_path[1], 128, {}, std::nullopt, *slot, SnapshotKind::Endpoint)
            .snapshot;
    require(!index.acquire_device_slot(0.0).has_value() && index.valid(long_end) &&
                index.valid(other_end),
            "a snapshot worth less than every owner must not displace one");
    index.release_path(long_path);
    index.release_path(short_path);
    index.release_path(other_path);
    index.check_invariants();
}

// A snapshot's Host write only displaces snapshots worth no more than itself (§9.3): a tap just
// past a conversation's boundary must not push another conversation's boundary out of the Host
// tier. Rejected, it stays Device-only.
void test_host_write_admission() {
    RecordingBackend backend;
    // 4 slabs: one tail-less image.
    PrefixCacheIndex index(small_config(4, 4), backend);
    const auto a      = make_tokens(64 * 8, 56);
    const auto path_a = insert_sequence(index, backend, a);
    const SnapshotRef valuable =
        publish_at(index, backend, path_a, a, 64 * 8, SnapshotKind::Boundary);
    backup_snapshot(index, valuable);
    const auto b            = make_tokens(64 * 1, 57);
    const auto path_b       = insert_sequence(index, backend, b);
    const SnapshotRef cheap = publish_tap(index, path_b[0]);
    index.pin_snapshot(cheap);
    require(!index.begin_snapshot_host_fill(cheap),
            "a snapshot worth less must not displace a Host snapshot worth more");
    index.unpin_snapshot(cheap);
    require(index.snapshot(valuable).host == CopyState::Resident &&
                index.snapshot(cheap).host == CopyState::Absent,
            "the rejected snapshot stays Device-only and the valuable one keeps its Host copy");
    index.release_path(path_a);
    index.release_path(path_b);
    index.check_invariants();
}

// A boundary no other conversation continued from is just its lineage's snapshot: superseded
// when the lineage moves past it, and retained again by a later hit.
void test_unshared_boundary() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(20, 4), backend);
    const auto tokens = make_tokens(64 * 6, 58);
    const auto path   = insert_sequence(index, backend, tokens);
    const SnapshotRef boundary =
        publish_at(index, backend, path, tokens, 64 * 2 + 5, SnapshotKind::Boundary);
    index.supersede(boundary);
    require(index.snapshot(boundary).superseded,
            "a boundary no other conversation shares must be superseded like any snapshot");
    index.note_hit(boundary);
    require(!index.snapshot(boundary).superseded, "a hit must retain it again");
    index.release_path(path);
    index.check_invariants();
}

// A lineage that resumes from a tap and goes past an endpoint below it (a client re-rendering the
// reply, a retry, a branch) supersedes that endpoint before the tap, unless the endpoint has been
// resumed from or the tap is a Boundary. Device-only, the endpoint then gives up its slot first.
void test_passed_endpoints() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(0, 4), backend);
    const auto main      = make_tokens(64 * 6, 70);
    const auto main_path = insert_sequence(index, backend, main);
    const SnapshotRef tap = publish_at(index, backend, main_path, main, 64 * 2, SnapshotKind::Tap);
    const SnapshotRef dead =
        publish_at(index, backend, main_path, main, 64 * 3 + 10, SnapshotKind::Endpoint);
    // An endpoint below the same tap that a later request resumed from: an exact-replay client.
    std::vector<TokenId> replayed(main.begin(), main.begin() + 64 * 2);
    const auto replayed_suffix = make_tokens(64 * 2, 71);
    replayed.insert(replayed.end(), replayed_suffix.begin(), replayed_suffix.end());
    const auto replayed_path = insert_sequence(index, backend, replayed);
    const SnapshotRef hit =
        publish_at(index, backend, replayed_path, replayed, 64 * 3 + 5, SnapshotKind::Endpoint);
    index.note_hit(hit);
    const auto unrelated      = make_tokens(64 * 2, 73);
    const auto unrelated_path = insert_sequence(index, backend, unrelated);
    const SnapshotRef other =
        publish_at(index, backend, unrelated_path, unrelated, 64 * 2, SnapshotKind::Tap);

    // A continuation that ends before leaving the endpoint's path passes nothing.
    index.supersede(tap, std::span<const TokenId>(main.data() + 64 * 2, 30));
    require(index.snapshot(tap).superseded && !index.snapshot(dead).superseded,
            "a continuation still on the endpoint's path must not supersede it");
    index.note_hit(tap);

    // The next request shares 150 tokens and goes on differently.
    std::vector<TokenId> next(main.begin(), main.begin() + 150);
    const auto next_suffix = make_tokens(64 * 5 - 150, 72);
    next.insert(next.end(), next_suffix.begin(), next_suffix.end());
    const auto next_path = insert_sequence(index, backend, next);
    index.supersede(tap, std::span<const TokenId>(next.data() + 64 * 2, next.size() - 64 * 2));
    require(index.snapshot(dead).superseded && index.snapshot(tap).superseded,
            "the passed endpoint and the resumed tap must both be superseded");
    require(!index.snapshot(hit).superseded, "an endpoint resumed from must stay retained");
    const auto slot = index.acquire_device_slot();
    require(slot.has_value() && !index.valid(dead) && index.valid(tap),
            "the passed endpoint must give up its slot before the lineage's tap");
    index.release_device_slot(*slot);
    require(index.valid(hit) && index.valid(other), "retained snapshots must survive");

    // Below a Boundary the endpoints may be other conversations': they are left alone.
    RecordingBackend boundary_backend;
    PrefixCacheIndex shared(small_config(0, 4), boundary_backend);
    const auto shared_path = insert_sequence(shared, boundary_backend, main);
    const SnapshotRef boundary =
        publish_at(shared, boundary_backend, shared_path, main, 64 * 2, SnapshotKind::Boundary);
    const SnapshotRef theirs =
        publish_at(shared, boundary_backend, shared_path, main, 64 * 3 + 10, SnapshotKind::Endpoint);
    const auto shared_next = insert_sequence(shared, boundary_backend, next);
    shared.supersede(boundary, std::span<const TokenId>(next.data() + 64 * 2, next.size() - 64 * 2));
    require(!shared.snapshot(theirs).superseded,
            "an endpoint below a Boundary must not be superseded by another conversation");

    index.release_path(main_path);
    index.release_path(replayed_path);
    index.release_path(unrelated_path);
    index.release_path(next_path);
    index.check_invariants();
    shared.release_path(shared_path);
    shared.release_path(shared_next);
    shared.check_invariants();
}

// The production failure: a conversation resuming turn after turn under Host pressure, next to a
// deep stale conversation. Each turn resumes from the previous turn's endpoint and publishes a
// deeper one; the new endpoint must always survive to serve the next turn.
void test_lineage_under_pressure() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(16, 4), backend);
    const auto stale_tokens = make_tokens(64 * 40 + 1, 40);
    const auto stale_path   = insert_sequence(index, backend, stale_tokens);
    const SnapshotRef stale =
        publish_at(index, backend, stale_path, stale_tokens, 64 * 40 - 7, SnapshotKind::Endpoint);
    backup_snapshot(index, stale);
    index.release_path(stale_path);

    const auto tokens = make_tokens(64 * 30 + 1, 41);
    const auto path   = insert_sequence(index, backend, tokens);
    // An aligned tap early in the conversation (the Legacy valuation anchored every later
    // endpoint to it and kept evicting them).
    SnapshotRef resume = publish_at(index, backend, path, tokens, 64 * 4, SnapshotKind::Tap);
    backup_snapshot(index, resume);
    for (std::uint32_t turn = 1; turn <= 8; ++turn) {
        const std::uint32_t frontier = 64 * 4 + turn * 200 + 13;
        index.note_hit(resume);
        const SnapshotRef tip =
            publish_at(index, backend, path, tokens, frontier, SnapshotKind::Endpoint);
        index.supersede(resume);
        backup_snapshot(index, tip);
        // Other traffic writes to the Host tier before this conversation's next turn arrives (a
        // short new conversation here): that allocation used to evict the newest endpoint.
        const auto noise_tokens = make_tokens(64 * 2 + 1, 100 + turn);
        const auto noise_path   = insert_sequence(index, backend, noise_tokens);
        backup_snapshot(index, publish_at(index, backend, noise_path, noise_tokens, 64 * 2 - 5,
                                          SnapshotKind::Endpoint));
        index.release_path(noise_path);
        require(index.valid(tip) && index.snapshot(tip).host == CopyState::Resident,
                "the newest endpoint must keep its Host copy");
        const std::vector<TokenId> next(tokens.begin(), tokens.begin() + frontier + 50);
        const auto hashes = block_lookup_hashes(next, {});
        const MatchResult match =
            index.match(next, hashes, {}, static_cast<std::uint32_t>(next.size()));
        require(!match.candidates.empty() && match.candidates.front().snapshot == tip,
                "the next turn must resume from the newest endpoint");
        resume = tip;
    }
    require(index.stats().superseded_evictions >= 6, "superseded endpoints must make the room");
    index.release_path(path);
    index.check_invariants();
}

void test_image_only_host_tier() {
    RecordingBackend backend;
    PrefixIndexConfig config = small_config(8, 1);
    config.image_slabs       = 1;
    config.host_blocks       = false;
    PrefixCacheIndex index(config, backend);
    const auto a      = make_tokens(64, 15);
    const auto path_a = insert_sequence(index, backend, a);
    index.pin_node(path_a[0]);
    require(!index.begin_host_fill(path_a[0]).has_value(),
            "an image-only host tier must not back KV blocks");
    index.unpin_node(path_a[0]);

    // Endpoint with a tail; its host copy holds the image only.
    const auto tail_tokens      = make_tokens(10, 16);
    const auto slot             = index.acquire_device_slot();
    const std::uint32_t tail_id = backend.allocate();
    const PublishResult endpoint =
        index.publish_snapshot(path_a[0], 74, tail_tokens, tail_id, *slot, SnapshotKind::Endpoint);
    backup_snapshot(index, endpoint.snapshot);
    require(index.snapshot(endpoint.snapshot).host_slabs.size() == 1,
            "an image-only host copy must not reserve a tail slab");
    index.release_path(path_a);
    index.check_invariants();

    // The tail is unbacked: evicting it loses the snapshot even though its image is on host.
    std::uint32_t released = 0;
    while (index.valid(endpoint.snapshot) && released < 4) {
        released += index.evict_device_blocks(1);
        index.check_invariants();
    }
    require(!index.valid(endpoint.snapshot), "a snapshot without its tail KV must be removed");
    require(index.stats().host_free_slabs == 8, "the removed snapshot's image slab was leaked");
}

// Host-born snapshots (images copied straight to Host slabs) on an index without Device snapshot
// slots: reservations, publication with and without a tail, duplicates, Device-tail eviction.
void test_host_born_snapshots() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(64, 0), backend);
    require(!index.acquire_device_slot().has_value(), "no Device slot exists to acquire");
    require(index.stats().free_device_slots == 0, "zero Device slots are configured");
    const auto tokens = make_tokens(64 * 2, 50);
    const auto path   = insert_sequence(index, backend, tokens);

    auto image = index.reserve_host_image(false, std::numeric_limits<double>::infinity());
    require(image.has_value() && image->slabs.size() == 4 && !image->tail,
            "an image reservation takes the image's slabs");
    require(index.stats().host_free_slabs == 60, "reserved slabs are not free");
    index.check_invariants();
    const PublishResult tap =
        index.publish_host_snapshot(path[1], 128, {}, std::nullopt, std::move(*image), SnapshotKind::Tap);
    require(tap.created && image->slabs.empty(), "a Host-born tap is published from its reservation");
    const SnapshotView tap_view = index.snapshot(tap.snapshot);
    require(tap_view.device_slot == kNoId && tap_view.host == CopyState::Resident &&
                tap_view.host_slabs.size() == 4 && tap_view.tail_len == 0,
            "a Host-born snapshot is Host-resident with no Device slot");
    index.check_invariants();

    // A duplicate returns the existing snapshot and its slabs to the free list.
    auto again = index.reserve_host_image(false, std::numeric_limits<double>::infinity());
    require(again.has_value() && index.stats().host_free_slabs == 56, "second reservation");
    const PublishResult duplicate =
        index.publish_host_snapshot(path[1], 128, {}, std::nullopt, std::move(*again), SnapshotKind::Tap);
    require(!duplicate.created && duplicate.snapshot == tap.snapshot,
            "a duplicate returns the existing snapshot");
    require(index.stats().host_free_slabs == 60, "a duplicate frees its reserved slabs");
    index.check_invariants();

    // A reservation that does not match the publication stays reserved and can be released.
    const auto tail_tokens = make_tokens(10, 51);
    auto plain             = index.reserve_host_image(false, std::numeric_limits<double>::infinity());
    require(plain.has_value(), "plain reservation");
    require_throws(
        [&] {
            (void)index.publish_host_snapshot(path[1], 138, tail_tokens, std::nullopt, std::move(*plain),
                                              SnapshotKind::Endpoint);
        },
        "a tail needs a reservation with a tail slab");
    require(plain->slabs.size() == 4 && index.stats().host_free_slabs == 56,
            "a refused publication leaves the reservation intact");
    index.check_invariants();
    index.release_host_image(std::move(*plain));
    require(plain->slabs.empty() && index.stats().host_free_slabs == 60, "release frees the slabs");
    index.release_host_image(std::move(*plain)); // already released: nothing to free
    index.check_invariants();

    // An endpoint with a tail and a Device tail copy: the tail slab is part of the reservation.
    auto with_tail = index.reserve_host_image(true, std::numeric_limits<double>::infinity());
    require(with_tail.has_value() && with_tail->slabs.size() == 5 && with_tail->tail,
            "a tail reservation adds the tail slab");
    const std::uint32_t tail_id = backend.allocate();
    const PublishResult endpoint = index.publish_host_snapshot(path[1], 138, tail_tokens, tail_id,
                                                               std::move(*with_tail), SnapshotKind::Endpoint);
    require(endpoint.created, "Host-born endpoint not created");
    const SnapshotView endpoint_view = index.snapshot(endpoint.snapshot);
    require(endpoint_view.host_slabs.size() == 5 && endpoint_view.tail_len == 10 &&
                endpoint_view.tail_device == tail_id && endpoint_view.tail_device_copy == CopyState::Resident,
            "the endpoint keeps its tail on both tiers");
    index.check_invariants();
    index.release_path(path);

    // Its Device tail is Host-backed: evicting it keeps the snapshot, and a match plans the
    // tail's restore from its slab.
    require(index.evict_backed_device_blocks(1) == 1 && backend.live.count(tail_id) == 0,
            "the backed Device tail is evictable");
    require(index.valid(endpoint.snapshot) &&
                index.snapshot(endpoint.snapshot).tail_device_copy == CopyState::Absent,
            "the endpoint survives on its Host tail");
    index.check_invariants();
    auto prompt = tokens;
    prompt.insert(prompt.end(), tail_tokens.begin(), tail_tokens.end());
    prompt.push_back(7);
    const auto hashes = block_lookup_hashes(prompt, {});
    const MatchResult match = index.match(prompt, hashes, {}, static_cast<std::uint32_t>(prompt.size()));
    require(match.candidates.size() == 2 && match.candidates[0].snapshot == endpoint.snapshot &&
                match.candidates[0].tail && !match.candidates[0].tail_on_device &&
                !match.candidates[0].image_on_device,
            "the endpoint is matched with its image and tail on the Host");

    // A Host-born tail without a Device copy is valid from the start.
    const auto other_tail = make_tokens(20, 52);
    auto host_tail        = index.reserve_host_image(true, std::numeric_limits<double>::infinity());
    require(host_tail.has_value(), "tail reservation");
    const PublishResult host_only = index.publish_host_snapshot(path[1], 148, other_tail, std::nullopt,
                                                                std::move(*host_tail), SnapshotKind::Endpoint);
    require(host_only.created && index.snapshot(host_only.snapshot).tail_device_copy == CopyState::Absent,
            "a Host-only tail is published");
    index.check_invariants();

    // Without Host blocks a snapshot's tail cannot live on the Host.
    PrefixIndexConfig image_only = small_config(8, 0);
    image_only.image_slabs       = 1;
    image_only.host_blocks       = false;
    RecordingBackend image_backend;
    PrefixCacheIndex image_index(image_only, image_backend);
    require_throws([&] { (void)image_index.reserve_host_image(true, 0.0); },
                   "a Host-born tail requires Host blocks");
    // And without a Host tier nothing can be reserved.
    PrefixCacheIndex device_only(small_config(0, 1), image_backend);
    require(!device_only.reserve_host_image(false, std::numeric_limits<double>::infinity()).has_value(),
            "no Host tier: no reservation");
}

// Reservations evict like begin_snapshot_host_fill, never past their claim, and outstanding
// reservations are never evicted.
void test_host_image_claims() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(12, 0), backend);
    const auto a      = make_tokens(64, 60);
    const auto path_a = insert_sequence(index, backend, a);
    auto first        = index.reserve_host_image(false, std::numeric_limits<double>::infinity());
    require(first.has_value(), "first reservation");
    const PublishResult snap_a =
        index.publish_host_snapshot(path_a[0], 64, {}, std::nullopt, std::move(*first), SnapshotKind::Tap);
    require(snap_a.created, "snapshot a");
    index.release_path(path_a);

    auto held   = index.reserve_host_image(false, 0.0);
    auto second = index.reserve_host_image(false, 0.0);
    require(held.has_value() && second.has_value() && index.stats().host_free_slabs == 0,
            "free slabs need no eviction");
    index.check_invariants();
    // Every slab is now owned or reserved: a claim below a's value evicts nothing.
    const double value_a = index.estimate_priority(0, 64, false);
    require(!index.reserve_host_image(false, value_a * 0.5).has_value() && index.valid(snap_a.snapshot),
            "a reservation must not evict a snapshot worth more than its claim");
    index.check_invariants();
    // A worthy claim evicts a; the outstanding reservations keep their slabs.
    auto third = index.reserve_host_image(false, std::numeric_limits<double>::infinity());
    require(third.has_value() && !index.valid(snap_a.snapshot) &&
                index.stats().host_snapshot_evictions == 1,
            "the GDSF minimum yields its slabs to a worthy reservation");
    index.check_invariants();
    require(!index.reserve_host_image(false, std::numeric_limits<double>::infinity()).has_value(),
            "reserved slabs are never evicted");
    std::vector<std::uint32_t> all;
    for (const auto* r : {&*held, &*second, &*third}) { all.insert(all.end(), r->slabs.begin(), r->slabs.end()); }
    std::sort(all.begin(), all.end());
    require(std::adjacent_find(all.begin(), all.end()) == all.end() && all.size() == 12,
            "reservations hold distinct slabs");
    index.release_host_image(std::move(*held));
    index.release_host_image(std::move(*second));
    index.release_host_image(std::move(*third));
    require(index.stats().host_free_slabs == 12, "all slabs return");
    index.check_invariants();
}

// Queue holds (§9.6): a held snapshot is never a Host victim, so a write that would need its slabs
// is refused; its path's Device copies go after every unheld entry of the same kind, host-backed
// before unbacked, the highest rank first.
void test_queue_holds() {
    constexpr double kAny = std::numeric_limits<double>::infinity();
    // Two conversations fill a 12-slab Host tier: 2 blocks and a 4-slab endpoint image each.
    const auto host_scenario = [&](RecordingBackend& backend, PrefixCacheIndex& index) {
        std::vector<SnapshotRef> out;
        for (const std::uint32_t seed : {70U, 71U}) {
            const auto path = insert_sequence(index, backend, make_tokens(128, seed));
            for (const NodeRef node : path) { backup_node(index, node); }
            auto image = index.reserve_host_image(false, kAny);
            require(image.has_value(), "an endpoint image fits");
            out.push_back(index.publish_host_snapshot(path[1], 128, {}, std::nullopt, std::move(*image),
                                                      SnapshotKind::Endpoint)
                              .snapshot);
            index.release_path(path);
        }
        require(index.stats().host_free_slabs == 0, "the Host tier is full");
        return out;
    };
    {
        // Control: without holds the older of two equal GDSF values yields.
        RecordingBackend backend;
        PrefixCacheIndex index(small_config(12, 0), backend);
        const auto snaps = host_scenario(backend, index);
        auto image       = index.reserve_host_image(false, kAny);
        require(image && !index.valid(snaps[0]) && index.valid(snaps[1]), "the older endpoint yields");
        index.release_host_image(std::move(*image));
        index.check_invariants();
    }
    {
        RecordingBackend backend;
        PrefixCacheIndex index(small_config(12, 0), backend);
        const auto snaps = host_scenario(backend, index);
        index.set_queue_holds(std::vector<SnapshotRef>{snaps[0], snaps[0], SnapshotRef{}});
        index.check_invariants();
        require(index.stats().held_snapshots == 1, "a repeated or invalid hold takes no rank");
        auto first = index.reserve_host_image(false, kAny);
        require(first && index.valid(snaps[0]) && !index.valid(snaps[1]),
                "an unheld snapshot yields before a held one");
        index.check_invariants();
        // The 4 freed slabs went to `first`; b's 2 dead blocks remain, then only the held endpoint.
        require(!index.reserve_host_image(false, kAny).has_value() && index.valid(snaps[0]) &&
                    index.stats().held_host_refusals == 1 && index.stats().host_dead_reclaims == 2,
                "a write is refused rather than evict a held snapshot");
        index.check_invariants();
        index.set_queue_holds({});
        require(index.stats().held_snapshots == 0, "an empty queue releases every hold");
        index.check_invariants();
        auto second = index.reserve_host_image(false, kAny);
        require(second && !index.valid(snaps[0]), "a released hold is an ordinary victim again");
        index.release_host_image(std::move(*first));
        index.release_host_image(std::move(*second));
        index.check_invariants();
    }
    // Device: b's blocks are older, so plain LRU would evict them first.
    const auto device_scenario = [&](RecordingBackend& backend, PrefixCacheIndex& index,
                                     std::vector<std::vector<NodeRef>>& paths) {
        std::vector<SnapshotRef> out;
        for (const std::uint32_t seed : {72U, 73U}) {
            paths.push_back(insert_sequence(index, backend, make_tokens(128, seed)));
            out.push_back(publish_tap(index, paths.back()[1]));
        }
        index.release_path(paths[1]); // b
        index.release_path(paths[0]); // a, most recently used
        index.check_invariants();
        return out; // a, b
    };
    {
        RecordingBackend backend;
        PrefixCacheIndex index(small_config(64, 4), backend);
        std::vector<std::vector<NodeRef>> paths;
        const auto snaps = device_scenario(backend, index, paths);
        index.set_queue_holds(std::vector<SnapshotRef>{snaps[1]});
        index.check_invariants();
        require(index.evict_device_blocks(1) == 1 && !index.valid(snaps[0]) && index.valid(snaps[1]),
                "an unheld unbacked block goes before a held one, whatever its recency");
        require(index.evict_device_blocks(1) == 1 && index.valid(snaps[1]) &&
                    index.stats().held_device_evictions == 0,
                "the unheld path goes entirely before any held block");
        require(index.evict_device_blocks(1) == 1 && !index.valid(snaps[1]) &&
                    index.stats().held_device_evictions == 1 && index.stats().held_snapshot_losses == 1 &&
                    index.stats().held_snapshots == 0,
                "a held block is evicted only when nothing else is left, and its loss is counted");
        index.check_invariants();
    }
    {
        // Ranks: the request furthest back gives way first.
        RecordingBackend backend;
        PrefixCacheIndex index(small_config(64, 4), backend);
        std::vector<std::vector<NodeRef>> paths;
        const auto snaps = device_scenario(backend, index, paths);
        index.set_queue_holds(std::vector<SnapshotRef>{snaps[0], snaps[1]});
        index.check_invariants();
        require(index.evict_device_blocks(1) == 1 && index.valid(snaps[0]) && !index.valid(snaps[1]),
                "the highest rank's block goes first");
        index.check_invariants();
    }
    {
        // A held host-backed copy goes before an unheld unbacked one: dropping it loses nothing.
        RecordingBackend backend;
        PrefixCacheIndex index(small_config(64, 4), backend);
        std::vector<std::vector<NodeRef>> paths;
        const auto snaps = device_scenario(backend, index, paths);
        for (const NodeRef node : paths[1]) { backup_node(index, node); }
        index.set_queue_holds(std::vector<SnapshotRef>{snaps[1]});
        index.check_invariants();
        require(index.evict_backed_device_blocks(8) == 2 && index.valid(snaps[0]) && index.valid(snaps[1]) &&
                    index.node(paths[1][0]).device == CopyState::Absent &&
                    index.node(paths[0][0]).device == CopyState::Resident &&
                    index.stats().held_device_evictions == 2,
                "backed-only eviction may take held copies but never an unbacked block");
        index.check_invariants();
    }
}

// insert_block without attach pins an identical Host-only child but leaves its Device copy absent:
// the inserting sequence keeps its page private.
void test_insert_without_attach() {
    RecordingBackend backend;
    PrefixCacheIndex index(small_config(64, 1), backend);
    const auto tokens = make_tokens(64 * 2, 61);
    const auto path   = insert_sequence(index, backend, tokens);
    for (const NodeRef node : path) { backup_node(index, node); }
    index.release_path(path);
    require(index.evict_device_blocks(2) == 2, "backed blocks are released");
    const auto hashes = block_lookup_hashes(tokens, {});
    NodeRef parent;
    std::vector<NodeRef> pinned;
    for (std::size_t b = 0; b < hashes.size(); ++b) {
        const std::span<const TokenId> block(tokens.data() + b * kBlockTokens, kBlockTokens);
        const std::uint32_t id    = backend.allocate();
        const InsertResult result = index.insert_block(parent, hashes[b], block, 0, id, false);
        require(!result.inserted && !result.device_attached && result.node == path[b],
                "an existing Host-only child is found without adopting the page");
        const NodeView view = index.node(result.node);
        require(view.pins == 1 && view.device == CopyState::Absent && view.device_id == kNoId,
                "the child is pinned and stays Host-only");
        backend.release_device_block(id); // the caller's private page
        pinned.push_back(result.node);
        parent = result.node;
    }
    index.check_invariants();
    index.release_path(pinned);
    index.check_invariants();
}

// The saturating call cost: rho = 0 is the per-chunk formula exactly; rho > 0 saturates towards
// chunk_seconds and one call is never dearer than two splitting it.
void test_cost_model() {
    CacheCostModel cost;
    cost.chunk_seconds          = 1.28;
    cost.chunk_tokens           = 512;
    cost.token_seconds          = 1.17e-3;
    cost.attention_pair_seconds = 3.0e-9;
    for (const std::uint32_t base : {0U, 100U, 4096U}) {
        for (const std::uint32_t tokens : {0U, 1U, 7U, 511U, 512U, 513U, 1024U, 5000U}) {
            const double s      = tokens;
            const double pairs  = base * s + s * (s + 1.0) / 2.0;
            const double chunks = static_cast<double>((tokens + 511U) / 512U);
            const double before =
                tokens == 0 ? 0.0 : chunks * cost.chunk_seconds + s * cost.token_seconds + pairs * cost.attention_pair_seconds;
            require(cost.prefill_seconds(base, tokens) == before, "rho = 0 must keep the per-chunk cost");
        }
    }
    require(cost.call_seconds(0) == 0.0 && cost.call_seconds(1) == cost.chunk_seconds,
            "rho = 0 charges every call chunk_seconds");

    cost.call_route_fraction = 10.0 / 512.0;
    double previous          = 0.0;
    for (std::uint32_t t = 1; t <= 8192; t = t < 16 ? t + 1 : t * 2) {
        const double got    = cost.call_seconds(t);
        const double oracle = cost.chunk_seconds * (1.0 - std::pow(1.0 - cost.call_route_fraction, t));
        require(std::abs(got - oracle) <= 1e-12 * cost.chunk_seconds, "call_seconds equals its closed form");
        require((got > previous || got == cost.chunk_seconds) && got <= cost.chunk_seconds,
                "call cost grows until it saturates at chunk_seconds");
        previous = got;
    }
    require(cost.chunk_seconds - cost.call_seconds(4096) <= 1e-12 * cost.chunk_seconds,
            "a call far wider than 1 / rho costs chunk_seconds");
    for (const std::uint32_t a : {1U, 8U, 100U}) {
        for (const std::uint32_t b : {1U, 64U, 400U}) {
            require(cost.call_seconds(a + b) <= cost.call_seconds(a) + cost.call_seconds(b),
                    "one call is never dearer than two splitting it");
        }
    }
    const double s    = 1300.0;
    const double want = 2.0 * cost.call_seconds(512) + cost.call_seconds(276) + s * cost.token_seconds +
                        (s * (s + 1.0) / 2.0) * cost.attention_pair_seconds;
    require(std::abs(cost.prefill_seconds(0, 1300) - want) <= 1e-12 * want,
            "prefill charges full calls plus the remainder call");
    require(cost.prefill_seconds(0, 1024) == 2.0 * cost.call_seconds(512) + 1024.0 * cost.token_seconds +
                                                 (1024.0 * 1025.0 / 2.0) * cost.attention_pair_seconds,
            "no remainder call when the tokens fill whole calls");
    cost.call_route_fraction = 1.5;
    require(cost.call_seconds(1) == cost.chunk_seconds, "rho >= 1 charges a full call for any width");
}

void test_random_stress() {
    std::mt19937 rng(1234);
    for (int round = 0; round < 20; ++round) {
        RecordingBackend backend;
        PrefixIndexConfig config = small_config(40, 3);
        config.max_nodes         = 128;
        config.max_snapshots     = 16;
        PrefixCacheIndex index(config, backend);
        // A small vocabulary of shared prefixes so paths overlap.
        std::vector<std::vector<TokenId>> prompts;
        for (int p = 0; p < 6; ++p) {
            auto tokens = make_tokens(64 * (2 + p % 4), 100 + round);
            auto suffix = make_tokens(64 * 3, 200 + p + round * 10);
            tokens.insert(tokens.end(), suffix.begin(), suffix.end());
            tokens.push_back(1);
            prompts.push_back(std::move(tokens));
        }
        std::vector<std::vector<NodeRef>> held;
        std::vector<SnapshotRef> published_snapshots;
        for (int step = 0; step < 300; ++step) {
            const int op = static_cast<int>(rng() % 6U);
            if (op <= 1 && held.size() < 3) {
                const auto& prompt           = prompts[rng() % prompts.size()];
                const auto hashes            = block_lookup_hashes(prompt, {});
                const auto n                 = static_cast<std::uint32_t>(prompt.size());
                MatchResult match            = index.match(prompt, hashes, {}, n);
                const AdmissionChoice choice = index.choose(match, n);
                std::uint32_t reuse_blocks   = 0;
                if (choice.candidate) {
                    const MatchCandidate& candidate = match.candidates[*choice.candidate];
                    reuse_blocks                    = candidate.path_blocks;
                    index.note_hit(candidate.snapshot);
                }
                std::vector<NodeRef> path(match.path.begin(), match.path.begin() + reuse_blocks);
                index.acquire_path(path);
                for (const NodeRef node : path) {
                    if (index.node(node).device == CopyState::Absent) {
                        index.begin_device_fill(node, backend.allocate());
                        index.complete_device_fill(node);
                    }
                }
                NodeRef parent = path.empty() ? NodeRef{} : path.back();
                for (std::uint32_t b = reuse_blocks; b < hashes.size(); ++b) {
                    if (index.stats().nodes + 2 >= config.max_nodes) {
                        (void)index.evict_device_blocks(4);
                    }
                    const std::span<const TokenId> block(prompt.data() + b * kBlockTokens,
                                                         kBlockTokens);
                    const std::uint32_t id    = backend.allocate();
                    const InsertResult result =
                        index.insert_block(parent, hashes[b], block, 0, id, true);
                    if (!result.device_attached) { backend.release_device_block(id); }
                    if (result.replaced_device_id != kNoId) {
                        backend.release_device_block(result.replaced_device_id);
                    }
                    path.push_back(result.node);
                    parent = result.node;
                    if (index.node(result.node).host == CopyState::Absent &&
                        index.node(result.node).device == CopyState::Resident && rng() % 2U) {
                        index.pin_node(result.node);
                        if (index.begin_host_fill(result.node)) {
                            index.complete_host_fill(result.node);
                        }
                        index.unpin_node(result.node);
                    }
                    if (rng() % 3U == 0) {
                        if (const auto slot = index.acquire_device_slot(
                                rng() % 2U ? std::numeric_limits<double>::infinity() : 0.0)) {
                            const PublishResult published = index.publish_snapshot(
                                result.node, (index.node(result.node).depth + 1U) * kBlockTokens,
                                {}, std::nullopt, *slot,
                                rng() % 4U == 0 ? SnapshotKind::Boundary : SnapshotKind::Tap);
                            if (published.created) {
                                published_snapshots.push_back(published.snapshot);
                            }
                            if (published.created && rng() % 2U) {
                                index.pin_snapshot(published.snapshot);
                                if (index.begin_snapshot_host_fill(published.snapshot)) {
                                    index.complete_snapshot_host_fill(published.snapshot);
                                }
                                index.unpin_snapshot(published.snapshot);
                            }
                        }
                    }
                    index.check_invariants();
                }
                held.push_back(std::move(path));
            } else if (op <= 3 && !held.empty()) {
                const std::size_t which = rng() % held.size();
                index.release_path(held[which]);
                held.erase(held.begin() + static_cast<std::ptrdiff_t>(which));
            } else if (op == 4 && !published_snapshots.empty()) {
                const SnapshotRef snapshot =
                    published_snapshots[rng() % published_snapshots.size()];
                if (index.valid(snapshot)) {
                    if (rng() % 2U) {
                        index.supersede(snapshot);
                    } else {
                        index.note_hit(snapshot);
                    }
                }
            } else if (rng() % 2U && !published_snapshots.empty()) {
                // A queue of up to four waiting requests; stale references are skipped.
                std::vector<SnapshotRef> queue;
                for (std::uint32_t i = rng() % 5U; i > 0; --i) {
                    queue.push_back(published_snapshots[rng() % published_snapshots.size()]);
                }
                index.set_queue_holds(queue);
            } else {
                (void)index.evict_device_blocks(1 + rng() % 8U);
            }
            index.check_invariants();
        }
        for (const auto& path : held) { index.release_path(path); }
        (void)index.evict_device_blocks(1000);
        index.check_invariants();
        // Every device page not held by an index node was released exactly once.
        std::uint32_t resident = 0;
        for (std::uint32_t id : backend.live) {
            (void)id;
            ++resident;
        }
        require(resident == index.stats().device_resident_blocks +
                                0U /* tails are never used in this test */,
                "device pages leaked or double-released");
    }
}

void test_tap_planner() {
    TapPlannerConfig config;
    config.max_new_taps   = 8;
    config.ladder_tokens  = 4096;
    config.min_gap_tokens = 1024;
    const std::vector<TapHint> hints{
        {100, TapHintKind::Structural},        {5000, TapHintKind::MessageBoundary},
        {9000, TapHintKind::MessageBoundary},  {21800, TapHintKind::MessageBoundary},
        {29990, TapHintKind::MessageBoundary}, {30010, TapHintKind::GenerationOpener},
        {12345, TapHintKind::Explicit},
    };
    const auto taps = plan_taps(30020, 0, hints, {}, {}, config);
    require(std::is_sorted(
                taps.begin(), taps.end(),
                [](const PlannedTap& a, const PlannedTap& b) { return a.position < b.position; }),
            "taps must be sorted");
    for (const PlannedTap& tap : taps) {
        require(tap.position > 0 && tap.position <= 30019, "tap outside the legal domain");
    }
    auto find = [&](const std::vector<PlannedTap>& planned,
                    std::uint32_t position) -> std::optional<PlannedTap> {
        for (const PlannedTap& tap : planned) {
            if (tap.position == position) { return tap; }
        }
        return std::nullopt;
    };
    require(find(taps, 12345) && find(taps, 12345)->placement == TapPlacement::Exact,
            "an explicit hint must be tapped exactly");
    require(find(taps, 30010) && find(taps, 30010)->placement == TapPlacement::Exact,
            "the generation opener must be tapped exactly");
    require(find(taps, 100) && find(taps, 100)->placement == TapPlacement::Exact,
            "a structural hint must be tapped exactly");
    require(find(taps, 100)->boundary && find(taps, 12345)->boundary &&
                !find(taps, 30010)->boundary,
            "structural and explicit taps are boundaries, the generation opener is not");
    require(!find(taps, 30019),
            "the prompt tail next to the generation opener covers nothing and must be dropped");
    // A protocol-automatic marker is planned like a structural boundary but marks the latest turn,
    // so its snapshot is supersedable; clustered with a shared boundary it stays shared.
    const std::vector<TapHint> automatic{{3000, TapHintKind::Automatic},
                                         {5000, TapHintKind::Automatic},
                                         {5030, TapHintKind::Structural}};
    const auto automatic_taps = plan_taps(8000, 0, automatic, {}, {}, config);
    require(find(automatic_taps, 3000) &&
                find(automatic_taps, 3000)->placement == TapPlacement::Exact &&
                !find(automatic_taps, 3000)->boundary,
            "an automatic marker must be tapped exactly and stay supersedable");
    require(find(automatic_taps, 5000) && find(automatic_taps, 5000)->boundary &&
                !find(automatic_taps, 5030),
            "a cluster keeps its earliest position, shared when any member is");
    // n - 8192 = 21828: the boundary 28 tokens below is within the gap, so the ladder snaps.
    require(find(taps, 21800) && find(taps, 21800)->placement == TapPlacement::Flexible,
            "the ladder must snap to a nearby message boundary and stay flexible");
    // n - 4096 = 25924 has no boundary within the gap: the ladder keeps its target.
    require(find(taps, 25924) && find(taps, 25924)->placement == TapPlacement::Flexible,
            "the ladder must keep its target without a nearby boundary");

    // A short user turn puts the system-block end and the generation opener in one cluster: the
    // earlier boundary is kept because a new conversation sharing the system block diverges
    // there, and the prompt tail next to it is dropped.
    const std::vector<TapHint> short_turn{{1085, TapHintKind::Structural},
                                          {1098, TapHintKind::GenerationOpener}};
    const auto clustered = plan_taps(1104, 0, short_turn, {}, {}, config);
    require(clustered.size() == 1 && clustered[0].position == 1085 &&
                clustered[0].placement == TapPlacement::Exact,
            "a semantic cluster must keep its earliest boundary alone");

    // An explicit breakpoint right after the system-block end does not replace it: only the
    // earlier snapshot serves a new conversation diverging between the two.
    const std::vector<TapHint> marked{{2109, TapHintKind::Structural},
                                      {2115, TapHintKind::Explicit},
                                      {2117, TapHintKind::GenerationOpener}};
    const auto both = plan_taps(2124, 0, marked, {}, {}, config);
    require(both.size() == 2 && both[0].position == 2109 && both[1].position == 2115,
            "an explicit breakpoint must not displace an earlier semantic boundary");

    // Without an opener near the end the prompt tail is a flexible tap.
    const std::vector<TapHint> plain{{100, TapHintKind::Structural}};
    const auto tail = plan_taps(3000, 0, plain, {}, {}, config);
    require(find(tail, 2999) && find(tail, 2999)->placement == TapPlacement::Flexible,
            "the prompt tail must be a flexible tap");

    // Resume base and existing snapshots suppress taps at or near them.
    const std::vector<std::uint32_t> existing{12345};
    const auto resumed = plan_taps(30020, 10000, hints, existing, {}, config);
    for (const PlannedTap& tap : resumed) {
        require(tap.position > 10000 && tap.position != 12345,
                "tap below base or duplicating an existing snapshot");
    }
    const std::vector<std::uint32_t> near_opener{30000};
    require(!find(plan_taps(30020, 0, hints, near_opener, {}, config), 30010),
            "a tap within the minimum separation of an existing snapshot must be dropped");

    // Exclusions move taps to the start of the excluded span.
    const std::vector<TapExclusion> exclusions{{12000, 13000}};
    const auto excluded = plan_taps(30020, 0, hints, {}, exclusions, config);
    for (const PlannedTap& tap : excluded) {
        require(!(tap.position > 12000 && tap.position < 13000), "tap inside a Vision span");
    }
    require(find(excluded, 12000).has_value(), "an excluded explicit tap moves to the span start");

    // Budget: the highest-priority tap survives.
    TapPlannerConfig one = config;
    one.max_new_taps     = 1;
    const auto single    = plan_taps(30020, 0, hints, {}, {}, one);
    require(single.size() == 1 && single[0].position == 12345, "tap budget or priority ignored");

    // Ladder size is logarithmic for a long prompt without hints.
    const auto ladder = plan_taps(240000, 0, {}, {}, {}, config);
    require(ladder.size() <= 7, "ladder must be logarithmic");
    require(plan_taps(1, 0, hints, {}, {}, config).empty(), "no taps for a 1-token prompt");
}

// A Host tier smaller than a saved one restores whole snapshots with their paths, most valuable
// first, and never a block no chosen snapshot resumes through.
void test_restore_plan() {
    const PrefixIndexConfig config = small_config();
    // Two lineages sharing blocks 0-1: 0-1-2-3-4 and 0-1-5-6-7-8-9.
    const std::vector<std::int32_t> parents         = {-1, 0, 1, 2, 3, 1, 5, 6, 7, 8};
    const std::vector<SavedSnapshotShape> snapshots = {
        {.anchor = 4, .frontier = 5 * 64, .slabs = 4, .hits = 0},
        {.anchor = 9, .frontier = 7 * 64 + 10, .slabs = 5, .hits = 5},
        {.anchor = 1, .frontier = 2 * 64, .slabs = 4, .hits = 0},
    };
    const auto chosen_blocks = [](const HostRestorePlan& plan) {
        std::vector<std::uint32_t> out;
        for (std::uint32_t index = 0; index < plan.blocks.size(); ++index) {
            if (plan.blocks[index]) { out.push_back(index); }
        }
        return out;
    };

    const HostRestorePlan all = plan_host_restore(parents, snapshots, 23, config);
    require(all.slabs == 23 && chosen_blocks(all).size() == parents.size() &&
                std::all_of(all.snapshots.begin(), all.snapshots.end(), [](bool b) { return b; }),
            "a tier that holds the whole file must restore all of it");

    // The blocks alone would fill 10 of these 12 slabs and leave no room for any image: the plan
    // instead takes the most used snapshot and exactly its path.
    const HostRestorePlan one = plan_host_restore(parents, snapshots, 12, config);
    require(!one.snapshots[0] && one.snapshots[1] && !one.snapshots[2] && one.slabs == 12,
            "a small tier must keep the most valuable snapshot");
    require(chosen_blocks(one) == std::vector<std::uint32_t>{0, 1, 5, 6, 7, 8, 9},
            "a small tier must restore exactly the kept snapshot's path");

    // Shared path blocks are paid for once, and a snapshot that does not fit is passed over for a
    // later one that does.
    const HostRestorePlan two = plan_host_restore(parents, snapshots, 16, config);
    require(!two.snapshots[0] && two.snapshots[1] && two.snapshots[2] && two.slabs == 16,
            "a snapshot sharing a kept path must cost only its own slabs");

    // Between otherwise equal snapshots, use decides.
    const std::vector<std::int32_t> roots        = {-1, -1};
    const std::vector<SavedSnapshotShape> equals = {
        {.anchor = 0, .frontier = 64, .slabs = 4, .hits = 0},
        {.anchor = 1, .frontier = 64, .slabs = 4, .hits = 3},
    };
    const HostRestorePlan used = plan_host_restore(roots, equals, 5, config);
    require(!used.snapshots[0] && used.snapshots[1] && !used.blocks[0] && used.blocks[1],
            "the more used of two equal snapshots must be kept");

    require_throws([&] { (void)plan_host_restore(std::vector<std::int32_t>{0}, {}, 8, config); },
                   "a block that is its own parent must be rejected");
}

} // namespace

int main() {
    try {
        test_hashes();
        test_match_and_choice();
        test_collisions_and_tails();
        test_device_lru_order();
        test_backed_before_unbacked();
        test_host_dead_and_gdsf();
        test_device_slots();
        test_supersession();
        test_lineage_under_pressure();
        test_unbacked_slot_values();
        test_host_write_admission();
        test_unshared_boundary();
        test_passed_endpoints();
        test_host_only_reattach();
        test_resident_duplicate_replacement();
        test_tail_device_fill();
        test_persistence_roundtrip();
        test_restore_plan();
        test_image_only_host_tier();
        test_host_born_snapshots();
        test_host_image_claims();
        test_queue_holds();
        test_insert_without_attach();
        test_cost_model();
        test_random_stress();
        test_tap_planner();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "prefix cache index tests passed\n";
    return 0;
}
