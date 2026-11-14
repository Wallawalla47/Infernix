#pragma once

#include "runtime/prefix_cache/block_hash.h"
#include "runtime/prefix_cache/cost.h"
#include "runtime/prefix_cache/ssd_stats.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace infernix::runtime::prefix_cache {

// Hybrid prefix cache index (docs/maintainer/hybrid-prefix-cache-spec.md §5, §6, §9).
//
// The index is host-only bookkeeping. It owns exact identity (the block tree and snapshot tails),
// residency state, pins, host slab ids, device snapshot slot ids and every eviction decision.
// Device page bundles are opaque ids supplied by the caller and returned through
// PrefixIndexBackend when the index stops referencing them. The index performs no transfers; the
// caller moves bytes and reports completion.
//
// Pins are path-wide: pinning a node pins it and every ancestor, so an unpinned node always has an
// unpinned subtree. Every active sequence pins its whole mapped path, every in-flight transfer
// pins the path of the node it touches.

inline constexpr std::uint32_t kNoId = std::numeric_limits<std::uint32_t>::max();
// Queued requests whose resume snapshots set_queue_holds holds at most (§9.6).
inline constexpr std::uint32_t kMaxQueueHolds = 32;

struct NodeRef {
    std::uint32_t index      = kNoId;
    std::uint32_t generation = 0;

    [[nodiscard]] bool valid() const noexcept { return index != kNoId; }

    [[nodiscard]] friend bool operator==(NodeRef, NodeRef) noexcept = default;
};

struct SnapshotRef {
    std::uint32_t index      = kNoId;
    std::uint32_t generation = 0;

    [[nodiscard]] bool valid() const noexcept { return index != kNoId; }

    [[nodiscard]] friend bool operator==(SnapshotRef, SnapshotRef) noexcept = default;
};

enum class CopyState : std::uint8_t {
    Absent,
    Filling,
    Resident,
};

enum class SnapshotKind : std::uint8_t {
    Tap,
    Endpoint,
    OutputBoundary,
    // A tap at a boundary that later prompts may share across conversations: a client-named
    // breakpoint, the end of the tools or leading System/Developer block, or the divergence of
    // concurrent requests. A lineage continuing past it supersedes it only while no other
    // conversation has continued from it.
    Boundary,
};

struct PrefixIndexConfig {
    // Upper bound of simultaneously indexed nodes. Every indexed node holds a device page bundle
    // or a host slab, so device pages + host slabs is sufficient.
    std::uint32_t max_nodes     = 0;
    std::uint32_t max_snapshots = 0;
    std::uint32_t host_slabs    = 0;
    // Slabs per snapshot image; a snapshot with a tail needs one more slab for the tail bundle.
    std::uint32_t image_slabs           = 0;
    std::uint32_t device_snapshot_slots = 0;
    std::uint64_t block_bytes           = 0; // one block bundle (all enabled pools)
    std::uint64_t image_bytes           = 0; // one state image
    // Whether the host tier backs KV block bundles and snapshot tails. Without it the host tier
    // holds state images only: tails and blocks are unbacked and their loss removes dependants.
    bool host_blocks = true;
    // SSD tier slabs (spec §5.7), each the size of a Host slab; 0: no SSD tier. Requires a Host
    // tier with host_blocks: every SSD transfer copies one Host slab.
    std::uint32_t ssd_slabs = 0;
    // Image slabs an SSD copy keeps (0: all). The image slabs past them hold state the binding can
    // restore without, as stale-but-finite data (Qwen4Exp's DFlash2 ring, which only steers drafts):
    // a promotion fills them with zeros.
    std::uint32_t ssd_image_slabs = 0;
    CacheCostModel cost;
};

// Physical resources the index references but does not own.
class PrefixIndexBackend {
public:
    virtual ~PrefixIndexBackend() = default;
    // The index dropped its last reference to this device page bundle (block or snapshot tail).
    virtual void release_device_block(std::uint32_t device_id) noexcept = 0;

    // The snapshot is being removed: every copy of its image is released. Called before the
    // snapshot reference becomes stale.
    virtual void release_snapshot(SnapshotRef /*snapshot*/) noexcept {}

    // The snapshot keeps its host image but loses its device image copy.
    virtual void drop_snapshot_device_image(SnapshotRef /*snapshot*/) noexcept {}
};

struct MatchCandidate {
    SnapshotRef snapshot;
    std::uint32_t frontier         = 0;
    std::uint32_t path_blocks      = 0; // full blocks up to and including the anchor
    std::uint32_t host_only_blocks = 0; // path blocks without a device copy (needing H2D)
    std::uint32_t filling_blocks   = 0; // path blocks another admission is already restoring
    // Path blocks with neither a Device nor a Host copy: only on the SSD tier, or being read from it.
    std::uint32_t ssd_only_blocks  = 0;
    bool tail                      = false;
    bool tail_on_device            = false;
    bool image_on_device           = false;
    // The image (or the tail) is only on the SSD tier, or being read from it.
    bool image_on_ssd_only         = false;
    bool tail_on_ssd_only          = false;
    std::uint64_t restore_bytes    = 0; // Host-to-Device bytes
    std::uint64_t ssd_bytes        = 0; // SSD-to-Host bytes before them
};

struct MatchResult {
    std::vector<NodeRef> path;              // matched full blocks in prompt order
    std::vector<MatchCandidate> candidates; // valid snapshots, deepest first
};

struct AdmissionChoice {
    std::optional<std::size_t> candidate; // absent: root
    double predicted_seconds = 0.0;
};

struct InsertResult {
    NodeRef node;
    bool inserted = false;
    // The supplied device id now belongs to the index: a new node, an existing host-only node
    // whose Device copy is re-established from the inserting sequence's page, or an existing node
    // whose unpinned Device copy the sequence's page replaces.
    bool device_attached = false;
    // The replaced Device copy (kNoId otherwise). No sequence maps it and no transfer reads it;
    // the caller releases it.
    std::uint32_t replaced_device_id = kNoId;
};

struct PublishResult {
    SnapshotRef snapshot;
    bool created = false;
};

// Free-list slabs a caller holds for a snapshot image it is copying to the Host before the
// snapshot exists: owned by no node or snapshot until publish_host_snapshot or release_host_image.
// The slabs are the image's, then one tail slab when `tail`.
struct HostImageReservation {
    std::vector<std::uint32_t> slabs;
    bool tail = false;
};

struct NodeView {
    std::uint32_t depth = 0;
    NodeRef parent;
    std::uint32_t device_id            = kNoId;
    CopyState device                   = CopyState::Absent;
    std::uint32_t host_slab            = kNoId;
    CopyState host                     = CopyState::Absent;
    std::uint32_t ssd_slab             = kNoId;
    CopyState ssd                      = CopyState::Absent;
    std::uint32_t pins                 = 0;
    std::uint32_t live_snapshots_below = 0;
};

struct SnapshotView {
    NodeRef anchor;
    std::uint32_t frontier     = 0;
    std::uint32_t tail_len     = 0;
    std::uint32_t tail_device  = kNoId;
    CopyState tail_device_copy = CopyState::Absent;
    std::uint32_t device_slot  = kNoId;
    CopyState host             = CopyState::Absent;
    std::span<const std::uint32_t> host_slabs; // image slabs, then the tail slab when tail_len>0
    CopyState ssd              = CopyState::Absent;
    std::span<const std::uint32_t> ssd_slabs;  // the same layout on the SSD tier
    bool ssd_packed    = false;                // the SSD image is a codec stream (fewer slabs)
    std::uint32_t pins = 0;
    SnapshotKind kind  = SnapshotKind::Tap;
    std::uint32_t hits = 0;
    bool superseded    = false;
    std::span<const TokenId> tail;
};

// Exact identity of one block, for saving the Host tier.
struct BlockIdentity {
    NodeRef parent;
    std::uint64_t lookup_hash = 0;
    std::uint64_t extra       = 0;
    std::span<const TokenId> tokens;
};

struct RestoredBlock {
    NodeRef node;
    std::uint32_t slab = kNoId;
};

// One slab copied between the Host tier and the SSD tier (§5.7).
struct SsdTransfer {
    std::uint32_t host_slab = kNoId;
    std::uint32_t ssd_slab  = kNoId;
};

// A write-back (Host to SSD) or a promotion (SSD to Host) the index has started: the slabs it
// reserved are Filling, and `snapshot` (with its anchor path) is pinned until it completes or aborts.
struct SsdJob {
    SnapshotRef snapshot;
    std::vector<NodeRef> nodes;          // blocks whose copy moves, in path order
    bool image = false;                  // the snapshot's image (and tail) moves
    std::vector<SsdTransfer> transfers;  // the nodes' slabs, then the image's (then the tail's)
    // A promotion's Host image slabs the SSD copy does not hold (PrefixIndexConfig::ssd_image_slabs):
    // the tier fills them with zeros before completing it.
    std::vector<std::uint32_t> zero_host_slabs;
    // A packed image (pack_ssd_write): the SSD slabs holding its codec stream and the Host image slabs
    // it encodes, in order; `transfers` then hold only the blocks' slabs and the tail's.
    bool packed = false;
    std::vector<std::uint32_t> image_ssd_slabs;
    std::vector<std::uint32_t> image_host_slabs;
};

// A write-back candidate offered to the SSD tier's write policy (§5.7).
struct SsdWriteCandidate {
    SnapshotKind kind   = SnapshotKind::Tap;
    std::uint32_t hits  = 0;
    std::uint32_t slabs = 0;  // SSD slabs the write takes: its image and its unwritten path
    double value        = 0.0; // GDSF value density: (1 + hits) x seconds saved per byte its eviction frees
    bool first_offer    = false; // not offered before since it was published or last left the SSD
};

enum class SsdPromotionStatus : std::uint8_t {
    Resident,  // nothing of the snapshot is only on the SSD
    Started,   // a promotion was started (the job)
    Waiting,   // another promotion is reading parts of it
    NoRoom,    // the Host tier cannot free the slabs now
};

struct SsdPromotionStart {
    SsdPromotionStatus status = SsdPromotionStatus::Resident;
    std::optional<SsdJob> job;
};

// What a failed sequence published that PrefixCacheIndex::withdraw still has to remove (§7.8).
struct Withdrawal {
    std::vector<NodeRef> nodes;         // blocks whose Device copy was the sequence's page
    std::vector<SnapshotRef> snapshots; // snapshots the sequence created

    [[nodiscard]] bool empty() const noexcept { return nodes.empty() && snapshots.empty(); }
};

struct PrefixIndexStats {
    std::uint32_t nodes                   = 0;
    std::uint32_t snapshots               = 0;
    std::uint32_t device_resident_blocks  = 0;
    std::uint32_t device_evictable_blocks = 0;
    std::uint32_t host_free_slabs         = 0;
    std::uint32_t host_slab_limit         = 0; // slabs in service (finish_host_shrink / grow_host)
    std::uint32_t free_device_slots       = 0;
    std::uint64_t device_block_evictions  = 0;
    std::uint64_t unbacked_node_losses    = 0;
    std::uint64_t host_dead_reclaims      = 0;
    std::uint64_t host_snapshot_evictions = 0;
    std::uint64_t superseded_evictions    = 0; // of host_snapshot_evictions
    std::uint64_t device_slot_evictions   = 0;
    std::uint64_t device_slot_refusals    = 0; // acquire_device_slot found no slot
    std::uint64_t snapshot_hits           = 0;
    double gdsf_inflation                 = 0.0;
    // Queue holds (§9.6): held snapshots now, held entries whose Device copy was evicted, held
    // snapshots lost, and Host allocations refused because only held snapshots were left.
    std::uint32_t held_snapshots          = 0;
    std::uint64_t held_device_evictions   = 0;
    std::uint64_t held_snapshot_losses    = 0;
    std::uint64_t held_host_refusals      = 0;
    // SSD tier (§5.7): free slabs; snapshots whose image is only on the SSD; Host copies given up
    // because the SSD holds them (images and blocks); SSD copies given up (dead blocks, backups of
    // Host-resident images, SSD-only snapshots lost).
    std::uint32_t ssd_free_slabs           = 0;
    std::uint32_t ssd_only_snapshots       = 0;
    std::uint64_t host_image_demotions     = 0;
    std::uint64_t host_block_demotions     = 0;
    std::uint64_t ssd_dead_reclaims        = 0;
    std::uint64_t ssd_backup_drops         = 0;
    std::uint64_t ssd_snapshot_evictions   = 0;
    std::array<SsdKindStats, kSnapshotKinds> ssd_by_kind{};
    std::uint64_t ssd_declined_snapshots = 0; // snapshots the write policy passed over (once each)
};

class PrefixCacheIndex {
public:
    PrefixCacheIndex(const PrefixIndexConfig& config, PrefixIndexBackend& backend);

    PrefixCacheIndex(const PrefixCacheIndex&)            = delete;
    PrefixCacheIndex& operator=(const PrefixCacheIndex&) = delete;

    [[nodiscard]] const PrefixIndexConfig& config() const noexcept { return config_; }

    // Replaces the machine model used by choose() and snapshot valuation. Existing priorities
    // are refreshed.
    void set_cost(const CacheCostModel& cost);

    // ---- lookup and choice (§6.1, §6.2) -------------------------------------------------------
    // `block_hashes[b]` is block_lookup_hashes(tokens)[b]; `block_extras` is empty or one extra key
    // per full block. Only snapshots with frontier <= prompt_tokens - 1 are returned.
    [[nodiscard]] MatchResult match(std::span<const TokenId> tokens,
                                    std::span<const std::uint64_t> block_hashes,
                                    std::span<const std::uint64_t> block_extras,
                                    std::uint32_t prompt_tokens) const;
    [[nodiscard]] AdmissionChoice choose(const MatchResult& match,
                                         std::uint32_t prompt_tokens) const;
    // Records a hit on the snapshot selected for an admission. A hit on a superseded snapshot
    // shows it still serves requests: it is retained again until its new lineage moves on. A
    // paused request's resume (`counted` false) is the request admitted before, recovering: it
    // refreshes recency and retention but adds no GDSF frequency, so a source a request replays
    // from again and again does not outrank the snapshots of the conversations still moving on.
    void note_hit(SnapshotRef snapshot, bool counted = true);
    // A lineage that resumed from `snapshot`, or captured it as a tap, has published a deeper
    // snapshot on the same path, so its next request resumes from that one: `snapshot` now serves
    // only requests that diverge before it. Superseded snapshots are evicted before every retained
    // one, oldest supersession first, and retained snapshots are valued against their nearest
    // retained ancestor. A Boundary snapshot is superseded only while no other conversation has
    // continued from it (the tree below it is a single chain).
    // `continuation` is the lineage's tokens from `snapshot`'s frontier on, when it resumed from
    // `snapshot`. Unless `snapshot` is a Boundary (whose endpoints below may be other
    // conversations'), an Endpoint whose nearest snapshot above it is `snapshot`, never resumed
    // from, that the continuation leaves before reaching it, is then superseded first: it serves
    // only an exact replay of its reply, and the conversation that went past `snapshot` did not
    // send one (a client re-rendering the reply, a retry, or a branch). Left retained, such
    // endpoints held Device-only slots for good while the lineage's own resume point gave up its
    // slot.
    // Returns whether `snapshot` itself became superseded (not already superseded, not a shared
    // Boundary).
    bool supersede(SnapshotRef snapshot, std::span<const TokenId> continuation = {});
    // Undoes a supersede() that returned true when the deeper snapshot was not published after
    // all (no Device slot, or the publication failed): `snapshot` is its lineage's resume point
    // again. Endpoints that supersede() superseded as passed stay superseded.
    void unsupersede(SnapshotRef snapshot);

    // ---- queue holds (§9.6) --------------------------------------------------------------------
    // The snapshots queued requests will resume from, in admission order (rank 0 is admitted
    // next). Replaces the previous holds; invalid references are skipped, a repeated snapshot
    // keeps its first rank, and at most kMaxQueueHolds are held. A held snapshot is never a
    // Host-allocation or snapshot-table victim, so new snapshots and block writes that would need
    // its slabs are refused instead. Its anchor path's Device copies are evicted after every
    // unheld entry of the same kind (host-backed copies before unbacked ones), the highest rank
    // first, so the request furthest back in the queue gives way first. Pins still take
    // precedence: a hold never blocks an active sequence's growth.
    void set_queue_holds(std::span<const SnapshotRef> holds);
    // Changes whenever a snapshot is published or removed, so the queue's holds are recomputed
    // only when a queued request's choice of snapshot may have changed.
    [[nodiscard]] std::uint64_t snapshot_epoch() const noexcept { return snapshot_epoch_; }

    // ---- pins -----------------------------------------------------------------------------
    // `path` is a root path (path[i] is the parent of path[i+1], path[0] a root child).
    void acquire_path(std::span<const NodeRef> path);
    // Releases pins taken by acquire_path/insert_block; nodes that become unpinned join the device
    // LRU deepest first, so deeper blocks are evicted before shallower ones.
    void release_path(std::span<const NodeRef> path);
    // Pins one node and its ancestors (in-flight transfer of that node).
    void pin_node(NodeRef node);
    void unpin_node(NodeRef node);
    // Pins one snapshot and its anchor path (in-flight transfer of the snapshot's image or tail).
    void pin_snapshot(SnapshotRef snapshot);
    void unpin_snapshot(SnapshotRef snapshot);

    // ---- tree mutation (§7.6) ------------------------------------------------------------------
    [[nodiscard]] std::optional<NodeRef> find_child(NodeRef parent, std::uint64_t lookup_hash,
                                                    std::span<const TokenId> block_tokens,
                                                    std::uint64_t extra) const;
    // Inserts a committed full block owning `device_id`, device Resident, pinned once for the
    // inserting sequence. When an identical child already exists `inserted` is false and the child
    // is pinned; with `attach` a host-only existing child adopts `device_id` as its Device copy, and
    // an unpinned Device-resident one adopts it in place of its copy, returned in
    // `replaced_device_id` (device_attached in both cases). Otherwise the caller keeps its page
    // private: a pinned copy is mapped by another sequence or read by a transfer.
    [[nodiscard]] InsertResult insert_block(NodeRef parent, std::uint64_t lookup_hash,
                                            std::span<const TokenId> block_tokens,
                                            std::uint64_t extra, std::uint32_t device_id,
                                            bool attach);

    // ---- device residency of blocks (§6.4, §9.1)
    // ------------------------------------------------- Host-only pinned node receives a device
    // page being filled from its host slab.
    void begin_device_fill(NodeRef node, std::uint32_t device_id);
    void complete_device_fill(NodeRef node);
    // Returns the device id to the backend.
    void abort_device_fill(NodeRef node);
    // Releases up to `blocks` unpinned device copies, host-backed ones first. Returns the number
    // released. An unbacked release loses the node and its subtree (§9.4).
    std::uint32_t evict_device_blocks(std::uint32_t blocks);
    // The same, releasing only host-backed copies: nothing loses its last copy.
    std::uint32_t evict_backed_device_blocks(std::uint32_t blocks);
    [[nodiscard]] std::uint32_t device_evictable_blocks() const noexcept;
    [[nodiscard]] std::uint32_t device_backed_evictable_blocks() const noexcept;

    // ---- host residency (§9.3)
    // ------------------------------------------------------------------- Allocates a slab for a
    // device-resident pinned node (write-through destination). Absent when no slab can be freed.
    [[nodiscard]] std::optional<std::uint32_t> begin_host_fill(NodeRef node);
    void complete_host_fill(NodeRef node);
    void abort_host_fill(NodeRef node);
    // Allocates image slabs (+1 tail slab) for a pinned snapshot, evicting no retained snapshot valued
    // above `claim` (the snapshot's own priority when absent). False when they cannot be freed.
    [[nodiscard]] bool begin_snapshot_host_fill(SnapshotRef snapshot,
                                                std::optional<double> claim = std::nullopt);
    void complete_snapshot_host_fill(SnapshotRef snapshot);
    void abort_snapshot_host_fill(SnapshotRef snapshot);
    // A pinned host-backed snapshot whose tail lost its Device copy receives `device_id` being
    // filled from its tail slab.
    void begin_tail_device_fill(SnapshotRef snapshot, std::uint32_t device_id);
    void complete_tail_device_fill(SnapshotRef snapshot);
    // Returns the device id to the backend.
    void abort_tail_device_fill(SnapshotRef snapshot);

    // ---- snapshots (§5.3, §7.4, §7.7, §9.2)
    // ------------------------------------------------------- Returns a device snapshot slot for a
    // tap/endpoint destination (staging): a free one, else a superseded owner's, else the least
    // valuable Device-only owner's when the new snapshot's `claim` priority is at least its GDSF
    // priority, or when Device-only owners hold every one of two or more slots and another of them
    // is worth at least as much (that snapshot is lost), else the least
    // recently hit Host-backed owner's (which keeps its Host copy). Absent with no Device slots
    // configured, or when every owner is pinned, held or worth more (counted as a refusal).
    [[nodiscard]] std::optional<std::uint32_t>
    acquire_device_slot(double claim = std::numeric_limits<double>::infinity());
    // GDSF priority of a new snapshot at `frontier` whose nearest retained snapshot on its path
    // is at `base_frontier` and which alone keeps the blocks between them (§9.3).
    [[nodiscard]] double estimate_priority(std::uint32_t base_frontier, std::uint32_t frontier,
                                           bool tail) const noexcept;
    void release_device_slot(std::uint32_t slot);
    // Publishes a snapshot whose image is in staging slot `device_slot`. frontier must equal
    // 64 * (anchor depth + 1) + tail.size() (tail.size() for the root anchor). A tail requires
    // `tail_device_id`. For a duplicate (same anchor and tail) the staging slot is freed, the tail
    // device id is released and the existing snapshot is returned with created=false.
    [[nodiscard]] PublishResult publish_snapshot(NodeRef anchor, std::uint32_t frontier,
                                                 std::span<const TokenId> tail,
                                                 std::optional<std::uint32_t> tail_device_id,
                                                 std::uint32_t device_slot, SnapshotKind kind);
    // Host-born snapshots: the image (and tail) is copied straight to Host slabs, with no Device
    // slot. Reserves the image's slabs, plus one tail slab when `tail` (which needs host_blocks),
    // evicting like begin_snapshot_host_fill but no retained snapshot valued above `claim` (the
    // new snapshot's estimate_priority). Absent when they cannot be freed.
    [[nodiscard]] std::optional<HostImageReservation> reserve_host_image(bool tail, double claim);
    // Returns a reservation's slabs to the free list; the reservation is left empty.
    void release_host_image(HostImageReservation&& reservation) noexcept;
    // Publishes a snapshot whose image and tail already landed in `image`'s slabs: Host Resident,
    // no Device slot. A tail requires a reservation made with `tail`; its Device copy
    // (`tail_device_id`) is optional. Frontier rules as publish_snapshot. A duplicate (same anchor
    // and tail), or no snapshot entry to be had, releases the slabs and the tail device id; a
    // duplicate returns the existing snapshot with created=false. `image` is consumed either way.
    [[nodiscard]] PublishResult publish_host_snapshot(NodeRef anchor, std::uint32_t frontier,
                                                      std::span<const TokenId> tail,
                                                      std::optional<std::uint32_t> tail_device_id,
                                                      HostImageReservation&& image,
                                                      SnapshotKind kind);

    // ---- withdrawal (§7.8) ---------------------------------------------------------------------
    // Removes what a sequence published from state that may be wrong (its logits went non-finite):
    // every snapshot of `withdrawal.snapshots`, and every node of `withdrawal.nodes` with its whole
    // subtree and each snapshot anchored there (later sequences that extended the path computed
    // over its pages). Call once the sequence has released its pins and its transfers have landed.
    // An entry another sequence or a transfer still pins stays in `withdrawal` for a later call;
    // below a pinned node the unpinned snapshots go now, so no admission resumes from them. Stale
    // references (entries evicted meanwhile) are dropped. Returns the entries removed.
    std::uint32_t withdraw(Withdrawal& withdrawal);

    // ---- persistence ---------------------------------------------------------------------------
    // What a saved cache keeps: every snapshot with a Host copy (with `ssd`, a kept SSD tier: or an
    // SSD copy) whose anchor path has such copies throughout, and exactly those paths, parents
    // before children. Other content is dead (no snapshot can resume through it) or unreachable
    // after a restart. Superseded snapshots are kept too (a retry or a re-sent prompt resumes from
    // them, issue #59), after the retained ones and in supersession order, so a restore can keep
    // their eviction order.
    void collect_persistable(std::vector<NodeRef>& nodes, std::vector<SnapshotRef>& snapshots,
                             bool ssd = false) const;
    [[nodiscard]] BlockIdentity block_identity(NodeRef node) const;
    // Rebuild a saved cache into a fresh index, parents before children and anchors before their
    // snapshots: each entry Host-resident (`host`: a free slab is taken, its bytes then read into
    // the returned slabs), at the given SSD slabs of a kept SSD tier, or both. Nothing is evicted to
    // make room; absent means no slab, node or snapshot entry was free, an SSD slab is out of range
    // or held twice, or the entry already exists. A `superseded` snapshot comes back superseded
    // (evicted first, oldest restored first), as it was when saved; a hit retains it again. After
    // the last entry with an SSD copy, finish_ssd_restore() makes every other SSD slab free.
    [[nodiscard]] std::optional<RestoredBlock> restore_block(NodeRef parent, std::uint64_t lookup_hash,
                                                             std::span<const TokenId> tokens, std::uint64_t extra,
                                                             bool host, std::uint32_t ssd_slab);
    [[nodiscard]] std::optional<SnapshotRef>
    restore_snapshot(NodeRef anchor, std::uint32_t frontier, std::span<const TokenId> tail, SnapshotKind kind,
                     std::uint32_t hits, bool superseded, bool host, std::span<const std::uint32_t> ssd_slabs,
                     bool ssd_packed);
    void finish_ssd_restore();

    // ---- Host tier resize (RAM pressure, HostSlabStore) ----------------------------------------
    // Slabs [0, host_slab_limit()) are owned, reserved or free; the configured host_slabs past it are retired and
    // hold nothing, so their memory can be released.
    [[nodiscard]] std::uint32_t host_slab_limit() const noexcept { return host_limit_; }
    // A shrink retires the slabs from a ceiling on while requests run and transfers fly:
    // - begin_host_shrink evicts as an allocation would (dead KV, superseded snapshots, then the GDSF minimum of the
    //   unheld entries, whatever its priority) until what is in use (owned and reserved) fits below `limit`, and sets
    //   the ceiling there, or where what held entries keep in use fits, and above every slab a transfer reads or
    //   writes now (reservations, filling copies, I/O-pinned entries), rounded up to a multiple of `granule` (the
    //   caller's chunk); it evicts nothing when that leaves nothing to retire. From then on no slab at or past the
    //   ceiling is taken: the free ones leave the free list and freed ones are retired. Returns the ceiling;
    //   host_slab_limit() when nothing can be retired now.
    // - plan_host_moves pairs each movable entry's slab past the ceiling with a free slab below it (lowest first,
    //   evicting as an allocation would when they run short): Host-resident blocks and snapshots without an I/O pin
    //   (an SSD job reads or writes them), pinned ones included (a sequence's mapped path; a restore batch that reads
    //   one makes the caller wait before committing). Dead KV past the ceiling is dropped instead (allocation evicts
    //   it first anyway). An entry an SSD job I/O-pins after begin waits until the job ends. Each is planned once; the
    //   caller copies the bytes of `from` into `to`. Only begin_host_shrink scans every node; the rest stays
    //   proportional to what lies past the ceiling (and the snapshots).
    // - commit_host_moves, once the copies have landed and no restore reads a Host slab: each entry still holding a
    //   planned `from` takes its `to` (one I/O-pinned meanwhile keeps its move for a later call), and the `to` of an
    //   entry gone meanwhile is freed. Returns whether no move is left.
    // - finish_host_shrink ends the shrink once no move is left and nothing past the ceiling is owned or reserved: the
    //   limit becomes the ceiling (true); false while something still has to move or land.
    struct HostMove {
        std::uint32_t from = 0;
        std::uint32_t to   = 0;
    };
    std::uint32_t begin_host_shrink(std::uint32_t limit, std::uint32_t granule);
    [[nodiscard]] std::vector<HostMove> plan_host_moves();
    bool commit_host_moves();
    bool finish_host_shrink();
    [[nodiscard]] bool host_shrinking() const noexcept { return ceiling_ < host_limit_; }
    [[nodiscard]] std::uint32_t host_ceiling() const noexcept { return ceiling_; }
    [[nodiscard]] bool host_moves_pending() const noexcept { return !host_moves_.empty(); }
    // Brings the retired slabs below `limit` (at most host_slabs) back as free ones; not while a shrink is under way.
    void grow_host(std::uint32_t limit);

    // ---- SSD tier (§5.7) -------------------------------------------------------------------------
    // Host slabs that can be freed without losing anything: free ones, dead blocks', and every Host
    // copy (block or image) the SSD also holds. Pinned and held entries are counted too.
    [[nodiscard]] std::uint64_t host_lossless_slabs() const noexcept;
    // The SSD image fates by kind so far (PrefixIndexStats::ssd_by_kind, without computing the rest).
    [[nodiscard]] const std::array<SsdKindStats, kSnapshotKinds>& ssd_kind_stats() const noexcept { return counters_.ssd_by_kind; }
    // Starts the next write-back: the retained, unpinned, unheld, Host-resident snapshot of lowest
    // priority whose image has no SSD copy and whose anchor path has a Host or SSD copy throughout
    // (a block another write-back is writing defers it). Reserves SSD slabs for the image (and tail)
    // and for every path block that is Host-resident without an SSD copy, evicting from the SSD tier
    // (dead blocks, backups of Host- or Device-resident images, then SSD-only snapshots by priority,
    // never the job's own), marks them Filling and pins the snapshot. Absent when no snapshot
    // qualifies or the SSD tier cannot free the slabs. `admit`, when given, sees each candidate
    // first (the tier's write policy); a declined one is passed over.
    [[nodiscard]] std::optional<SsdJob>
    begin_ssd_write(const std::function<bool(const SsdWriteCandidate&)>& admit = {});
    // The write landed: the copies are SSD Resident and the snapshot is unpinned.
    void complete_ssd_write(SsdJob& job);
    // Before its write is issued: the job's image is stored as a codec stream in its first
    // `stream_slabs` image SSD slabs (fewer than it reserved); the rest return to the SSD tier.
    void pack_ssd_write(SsdJob& job, std::uint32_t stream_slabs);
    // Image slabs an SSD copy keeps raw, or encodes when packed (PrefixIndexConfig::ssd_image_slabs).
    [[nodiscard]] std::uint32_t ssd_image_slab_count() const noexcept;
    // The write failed: the SSD slabs are freed and the entries stay as they were.
    void abort_ssd_write(SsdJob& job);
    // Starts reading the parts of `snapshot` that are only on the SSD (path blocks, image, tail)
    // into Host slabs reserved now (evicting from the Host tier, losslessly first), marked Filling
    // with the snapshot pinned. Resident when nothing is only on the SSD; Waiting when another
    // promotion reads some of them.
    [[nodiscard]] SsdPromotionStart begin_ssd_promotion(SnapshotRef snapshot);
    // The reads landed: the Host copies are Resident (the SSD copies stay) and the snapshot is
    // unpinned.
    void complete_ssd_promotion(SsdJob& job);
    // The reads failed: the Host slabs are freed, the snapshot unpinned, and the SSD copies read
    // are dropped (an entry left with no copy goes with its subtree, §9.4).
    void abort_ssd_promotion(SsdJob& job);
    // The read rate the cost model uses for SSD candidates (§6.2).
    void set_ssd_rate(double bytes_per_second) noexcept;

    // ---- views
    // ------------------------------------------------------------------------------------
    [[nodiscard]] bool valid(NodeRef node) const noexcept;
    [[nodiscard]] bool valid(SnapshotRef snapshot) const noexcept;
    [[nodiscard]] NodeView node(NodeRef node) const;
    [[nodiscard]] SnapshotView snapshot(SnapshotRef snapshot) const;
    [[nodiscard]] std::optional<SnapshotRef> slot_owner(std::uint32_t slot) const;
    [[nodiscard]] PrefixIndexStats stats() const noexcept;
    // Structural self-check for tests: throws std::logic_error on any invariant violation.
    void check_invariants() const;

private:
    struct Node {
        std::uint32_t generation = 0;
        bool occupied            = false;
        std::uint32_t parent     = kNoId;
        std::uint32_t depth      = 0;
        std::uint64_t hash       = 0;
        std::uint64_t extra      = 0;
        std::array<TokenId, kBlockTokens> tokens{};
        std::vector<std::uint32_t> children;
        std::vector<std::uint32_t> snapshots;
        std::uint32_t device_id  = kNoId;
        CopyState device         = CopyState::Absent;
        std::uint32_t host_slab  = kNoId;
        CopyState host           = CopyState::Absent;
        std::uint32_t ssd_slab   = kNoId;
        CopyState ssd            = CopyState::Absent;
        std::uint32_t pins       = 0;
        // SSD transfers (§5.7), path-wide: the Host and SSD copies stay and the node stays indexed,
        // but its Device copy remains evictable (a pin would take it out of the Device LRU).
        std::uint32_t io_pins    = 0;
        std::uint32_t live_below = 0;
        // Retained (not superseded) snapshots anchored at or below this node.
        std::uint32_t retained_below = 0;
        // Snapshots anchored at or below this node whose image is on the Device or the Host
        // (Filling included): a block whose snapshots below are all only on the SSD is "cold".
        std::uint32_t warm_below = 0;
        std::uint32_t dead_prev      = kNoId;
        std::uint32_t dead_next      = kNoId;
        bool in_dead                 = false;
        // Intrusive membership of the SSD dead list and the clean lists (kNoList: none).
        std::uint32_t ssd_dead_prev = kNoId;
        std::uint32_t ssd_dead_next = kNoId;
        bool in_ssd_dead            = false;
        std::uint32_t clean_prev    = kNoId;
        std::uint32_t clean_next    = kNoId;
        std::uint8_t clean_list     = 0xff;
        // Counted in host_clean_nodes_ (Host and SSD Resident).
        bool clean_counted = false;
        // Lowest queue rank of a held snapshot whose anchor path holds this node (kNoHold: none).
        std::uint8_t hold = kNoHold;
    };

    struct Snapshot {
        std::uint32_t generation = 0;
        bool occupied            = false;
        std::uint32_t anchor     = kNoId; // node index, kNoId for the root
        std::uint32_t frontier   = 0;
        std::uint32_t tail_len   = 0;
        std::array<TokenId, kBlockTokens - 1> tail{};
        std::uint32_t tail_device  = kNoId;
        CopyState tail_device_copy = CopyState::Absent;
        std::uint32_t device_slot  = kNoId;
        std::vector<std::uint32_t> host_slabs;
        CopyState host                = CopyState::Absent;
        std::vector<std::uint32_t> ssd_slabs; // the Host layout: image slabs, then the tail slab
        CopyState ssd                 = CopyState::Absent;
        bool ssd_read                 = false; // its SSD image was read since it was written
        bool ssd_packed               = false; // its SSD image is a codec stream (pack_ssd_write)
        bool ssd_offered              = false; // offered to the write policy (SsdWriteCandidate::first_offer)
        bool ssd_declined             = false; // the write policy passed it over at least once
        // Counted in the anchor path's warm_below, and in host_clean_image_slabs_.
        bool warm_counted             = false;
        bool clean_counted            = false;
        std::uint32_t pins            = 0;
        std::uint32_t io_pins         = 0; // SSD transfers of its image (Node::io_pins)
        SnapshotKind kind             = SnapshotKind::Tap;
        std::uint32_t hits            = 0;
        double priority               = 0.0; // GDSF H = priority_base + F * C / Z
        double priority_base          = 0.0; // GDSF inflation at the last publication or hit
        std::uint64_t last_hit_tick   = 0;
        bool superseded               = false;
        std::uint64_t superseded_tick = 0;
        // Nearest snapshot above this one on its path (kNoId: none), whatever its state, and the
        // snapshots whose nearest one this is.
        std::uint32_t ancestor = kNoId;
        std::vector<std::uint32_t> dependents;
        std::uint8_t hold = kNoHold; // queue rank while held (§9.6)
    };

    enum class SlotState : std::uint8_t { Free, Staging, Owned };

    static constexpr std::uint8_t kNoHold = 0xff;
    // Device LRU lists: 0 backed, 1 unbacked, then for each queue rank r a backed (2 + 2r) and an
    // unbacked (3 + 2r) list of held entries. Even lists are backed.
    static constexpr std::size_t kLruLists = 2 + 2 * static_cast<std::size_t>(kMaxQueueHolds);

    // Device LRU entries: node i is entry i, snapshot tail j is entry max_nodes + j.
    struct LruLink {
        std::uint32_t prev = kNoId;
        std::uint32_t next = kNoId;
        std::uint8_t list  = 0xff; // see kLruLists; 0xff absent
    };

    struct LruList {
        std::uint32_t head  = kNoId;
        std::uint32_t tail  = kNoId;
        std::uint32_t count = 0;
    };

    [[nodiscard]] Node& require(NodeRef node);
    [[nodiscard]] const Node& require(NodeRef node) const;
    [[nodiscard]] Snapshot& require(SnapshotRef snapshot);
    [[nodiscard]] const Snapshot& require(SnapshotRef snapshot) const;
    [[nodiscard]] NodeRef ref_of_node(std::uint32_t index) const noexcept;
    [[nodiscard]] SnapshotRef ref_of_snapshot(std::uint32_t index) const noexcept;
    [[nodiscard]] static std::uint64_t child_key(std::uint32_t parent,
                                                 std::uint64_t lookup_hash) noexcept;

    [[nodiscard]] bool snapshot_valid(const Snapshot& snapshot) const noexcept;
    [[nodiscard]] bool tail_matches(const Snapshot& snapshot,
                                    std::span<const TokenId> prompt) const noexcept;

    void pin_path_from(std::uint32_t node);
    // An SSD job (§5.7) protects its snapshot and anchor path with I/O pins.
    void io_pin_snapshot(std::uint32_t snapshot);
    void io_unpin_snapshot(std::uint32_t snapshot);
    // Pinned or I/O-pinned: its Host and SSD copies stay and it stays indexed.
    [[nodiscard]] static bool held(const Node& node) noexcept { return node.pins != 0 || node.io_pins != 0; }
    [[nodiscard]] static bool held(const Snapshot& snapshot) noexcept { return snapshot.pins != 0 || snapshot.io_pins != 0; }
    void unpin_path_from(std::uint32_t node, bool deepest_first_order);
    void refresh_node(std::uint32_t node);
    void refresh_tail(std::uint32_t snapshot);

    std::uint32_t evict_device_entries(std::uint32_t blocks, bool unbacked);
    void lru_remove(std::uint32_t entry);
    void lru_append(std::uint32_t entry, std::uint8_t list);
    [[nodiscard]] std::uint32_t lru_pop(std::uint8_t list);
    // The list of an evictable entry: its residency kind and queue hold.
    [[nodiscard]] static std::uint8_t lru_list(bool backed, std::uint8_t hold) noexcept;
    // The next Device entry to evict: unheld backed, held backed (highest rank first), then with
    // `unbacked` unheld unbacked and held unbacked. kNoId when none.
    [[nodiscard]] std::uint32_t lru_pop_victim(bool unbacked);
    // Recomputes every node's hold from holds_ and moves the affected LRU entries.
    void refresh_holds();

    void dead_remove(std::uint32_t node);
    void dead_append(std::uint32_t node);
    void ssd_dead_remove(std::uint32_t node);
    void ssd_dead_append(std::uint32_t node);
    // Clean lists: kCold (no warm snapshot below), kCleanWarm (one is).
    void clean_remove(std::uint32_t node);
    void clean_append(std::uint32_t node, std::uint8_t list);

    // ---- SSD tier helpers (§5.7)
    // A copy outside the Device: Host or SSD Resident (the Device LRU's "backed").
    [[nodiscard]] static bool node_backed(const Node& node) noexcept;
    [[nodiscard]] bool tail_backed(const Snapshot& snapshot) const noexcept;
    // An image copy outside the Device (a Device slot owner that is not Device-only, §9.2).
    [[nodiscard]] static bool image_backed(const Snapshot& snapshot) noexcept;
    // Recounts the snapshot's warmth (anchor path warm_below) and clean image slabs.
    void sync_snapshot(std::uint32_t snapshot);
    void adjust_warm(std::uint32_t anchor, int delta);
    // The snapshot gives up its Host image (and tail slab), keeping its SSD copy.
    void demote_snapshot(std::uint32_t snapshot);
    void drop_node_ssd_copy(std::uint32_t node);
    void drop_snapshot_ssd_copy(std::uint32_t snapshot);
    void free_ssd_slab(std::uint32_t slab);
    // One eviction freeing SSD slabs for a write-back worth `claim` (its snapshot's priority), never
    // touching `protect`: dead blocks' copies, superseded SSD-only snapshots, backups of images
    // worth more than the claim (the Host keeps those longer), then SSD-only snapshots worth at most
    // the claim. False when nothing can yield. The bound keeps write-backs from displacing one
    // another's backups in turn.
    [[nodiscard]] bool evict_for_ssd(std::uint32_t protect, double claim);
    // The SSD slabs evict_for_ssd could free for `claim` (a lower bound: the blocks a removed
    // snapshot leaves dead are not counted), free ones included.
    [[nodiscard]] std::uint64_t ssd_room_for(std::uint32_t protect, double claim) const;
    [[nodiscard]] bool allocate_ssd_slabs(std::uint32_t count, std::vector<std::uint32_t>& out,
                                          std::uint32_t protect, double claim);
    [[nodiscard]] std::uint32_t snapshot_slab_count(const Snapshot& snapshot) const noexcept;
    // An SSD copy's slabs with the tail's (PrefixIndexConfig::ssd_image_slabs).
    [[nodiscard]] std::uint32_t ssd_snapshot_slab_count(const Snapshot& snapshot) const noexcept;
    // The Host slab position (in Snapshot::host_slabs) of an SSD copy's slab k.
    [[nodiscard]] std::size_t ssd_to_host_position(std::size_t k) const noexcept;
    [[nodiscard]] SsdKindStats& kind_stats(const Snapshot& snapshot) noexcept {
        return counters_.ssd_by_kind[static_cast<std::size_t>(snapshot.kind)];
    }

    [[nodiscard]] bool take_free_slabs(std::uint32_t count, std::vector<std::uint32_t>& out);
    // Restore: marks SSD slabs held by restored copies; false when one is out of range or held already.
    [[nodiscard]] bool claim_ssd_slabs(std::span<const std::uint32_t> slabs);
    // Evicts dead KV, superseded snapshots, then retained snapshots in GDSF order until `count`
    // slabs are free, but no retained snapshot with a priority above `claim`.
    [[nodiscard]] bool allocate_slabs(std::uint32_t count, std::vector<std::uint32_t>& out,
                                      std::uint32_t protect_snapshot,
                                      double claim = std::numeric_limits<double>::infinity());
    // One eviction of allocate_slabs's order that frees Host slabs; false when nothing (or nothing worth at most
    // `claim`) can yield.
    [[nodiscard]] bool evict_for_slabs(std::uint32_t protect_snapshot, double claim);
    void free_slab(std::uint32_t slab);

    // The publication shared by publish_snapshot and publish_host_snapshot: exactly one of
    // `device_slot` (a staging slot) and `host_slabs` (reserved slabs, already unmarked) holds the
    // image. A duplicate or a full snapshot table releases the image and the tail device id.
    [[nodiscard]] PublishResult publish_image(NodeRef anchor, std::uint32_t frontier,
                                              std::span<const TokenId> tail,
                                              std::optional<std::uint32_t> tail_device_id,
                                              std::uint32_t device_slot,
                                              std::vector<std::uint32_t>&& host_slabs,
                                              SnapshotKind kind);
    void remove_snapshot(std::uint32_t snapshot);
    // Snapshot ancestry: `above` lies on `below`'s path at a smaller frontier.
    [[nodiscard]] bool on_path(const Snapshot& above, const Snapshot& below) const noexcept;
    // Whether the tree branches below the snapshot's frontier (another conversation continued
    // from its prefix).
    [[nodiscard]] bool shared_below(const Snapshot& snapshot) const noexcept;
    // Whether `continuation` (path tokens from position `from` on) differs from the snapshot's
    // path before the snapshot's frontier; false when it ends first without differing.
    [[nodiscard]] bool leaves_path(const Snapshot& snapshot, std::uint32_t from,
                                   std::span<const TokenId> continuation) const noexcept;
    [[nodiscard]] std::uint32_t find_ancestor(std::uint32_t snapshot) const noexcept;
    void link_snapshot(std::uint32_t snapshot);
    void unlink_snapshot(std::uint32_t snapshot);
    [[nodiscard]] std::vector<std::uint32_t>& dependents_of(std::uint32_t snapshot) noexcept;
    [[nodiscard]] std::uint32_t retained_ancestor(std::uint32_t snapshot) const noexcept;
    // Retained snapshots whose nearest retained ancestor is `snapshot` (through superseded ones).
    void collect_retained_dependents(std::uint32_t snapshot, std::vector<std::uint32_t>& out) const;
    void adjust_retained(std::uint32_t anchor, int delta);
    // Revalues the snapshots whose value depends on `snapshot` being retained.
    void revalue_around(std::uint32_t snapshot, std::span<const std::uint32_t> dependents);
    // Next snapshot to give up its Host slabs (host_only) or its index entry: the oldest
    // superseded one, otherwise the GDSF minimum. kNoId when every candidate is protected.
    [[nodiscard]] std::uint32_t pick_victim(bool host_resident, std::uint32_t protect) const;
    // Host-resident, unpinned, unheld snapshots: the oldest superseded one, and the lowest-priority
    // retained ones whose image the SSD tier holds (clean) and does not hold (unclean).
    struct HostVictims {
        std::uint32_t superseded = kNoId;
        std::uint32_t clean      = kNoId;
        std::uint32_t unclean    = kNoId;
    };
    [[nodiscard]] HostVictims pick_host_victims(std::uint32_t protect) const;
    void remove_subtree(std::uint32_t node);
    void release_node_storage(std::uint32_t node);
    void drop_node_device_copy(std::uint32_t node);
    void drop_node_host_copy(std::uint32_t node);
    void adjust_live(std::uint32_t anchor, int delta);
    // Every Device residency and slot-state change goes through these so the gauges stats()
    // reports stay O(1): the Engine publishes statistics after every decode round.
    void set_device(Node& node, CopyState state) noexcept;
    void set_slot(std::uint32_t slot, SlotState state) noexcept;
    void update_priority(std::uint32_t snapshot);

    PrefixIndexConfig config_;
    PrefixIndexBackend* backend_ = nullptr;

    std::vector<Node> nodes_;
    std::vector<std::uint32_t> free_nodes_;
    std::unordered_multimap<std::uint64_t, std::uint32_t> children_;
    std::vector<std::uint32_t> root_children_;
    std::vector<std::uint32_t> root_snapshots_;
    // Snapshots with no snapshot above them.
    std::vector<std::uint32_t> top_snapshots_;

    std::vector<Snapshot> snapshots_;
    std::vector<std::uint32_t> free_snapshots_;

    std::vector<LruLink> lru_links_;
    std::array<LruList, kLruLists> lru_{};
    std::uint32_t lru_backed_   = 0; // entries on even lists
    std::uint32_t lru_unbacked_ = 0; // entries on odd lists

    // Held snapshot indices by queue rank (§9.6), and the nodes refresh_holds marked.
    std::vector<std::uint32_t> holds_;
    std::vector<std::uint32_t> held_nodes_;
    std::uint64_t snapshot_epoch_ = 0;

    std::uint32_t dead_head_ = kNoId;
    std::uint32_t dead_tail_ = kNoId;
    std::uint32_t dead_count_ = 0;
    std::uint32_t ssd_dead_count_ = 0;
    std::uint32_t ssd_dead_head_ = kNoId;
    std::uint32_t ssd_dead_tail_ = kNoId;
    static constexpr std::uint8_t kCold      = 0;
    static constexpr std::uint8_t kCleanWarm = 1;
    std::array<std::uint32_t, 2> clean_head_{kNoId, kNoId};
    std::array<std::uint32_t, 2> clean_tail_{kNoId, kNoId};
    // Host slabs whose bytes the SSD tier also holds: blocks, and snapshot images with their tails.
    std::uint64_t host_clean_nodes_       = 0;
    std::uint64_t host_clean_image_slabs_ = 0;
    std::vector<std::uint32_t> free_ssd_slabs_; // a min-heap: the file fills from its start
    std::vector<bool> ssd_claimed_;             // during a restore: slabs restored copies hold

    std::vector<std::uint32_t> free_slabs_;
    std::uint32_t host_limit_ = 0; // begin/finish_host_shrink, grow_host
    // While a shrink is under way the slabs from here on are never taken and freed ones retire (host_limit_ else).
    std::uint32_t ceiling_ = 0;
    // During a shrink: the slabs at or past the ceiling still owned or reserved (the shrink ends at 0; free_slab and
    // commit_host_moves count them down), and the nodes begin_host_shrink found holding one (plan_host_moves revisits
    // only these: no node takes a slab past the ceiling later; snapshots, far fewer, are scanned).
    std::uint32_t past_in_use_ = 0;
    std::vector<std::uint32_t> shrink_nodes_;
    // A planned shrink move: the entry (a node, or a snapshot and the position in its host_slabs) holding `from`.
    struct PendingHostMove {
        bool snapshot          = false;
        std::uint32_t entry    = 0;
        std::uint32_t position = 0;
        std::uint32_t from     = 0;
        std::uint32_t to       = 0;
    };
    std::vector<PendingHostMove> host_moves_;
    std::vector<bool> slab_moving_; // a planned move's `from`
    // Where an entry holds a planned move's `from` (null when it no longer does).
    [[nodiscard]] std::uint32_t* host_move_holder(const PendingHostMove& move) noexcept;
    // Slabs held by an outstanding HostImageReservation (owned by no node or snapshot).
    std::vector<bool> slab_reserved_;
    std::vector<SlotState> slot_state_;
    std::vector<std::uint32_t> slot_owner_;

    std::uint32_t node_count_      = 0;
    std::uint32_t snapshot_count_  = 0;
    std::uint32_t device_resident_ = 0;
    std::uint32_t free_slots_      = 0;
    std::uint64_t tick_            = 0;
    double inflation_              = 0.0;
    PrefixIndexStats counters_;
};

// ---- restore planning (§5.5) -------------------------------------------------------------------
// A saved snapshot as the restore plan sees it.
struct SavedSnapshotShape {
    // Index of its anchor among the saved blocks; -1 for the root.
    std::int32_t anchor    = -1;
    std::uint32_t frontier = 0;
    // Host slabs it takes: the image's, plus one for a tail.
    std::uint32_t slabs = 0;
    std::uint32_t hits  = 0;
    bool superseded     = false;
    // The file holds its Host bytes (false: only an SSD copy, which takes no Host slab).
    bool host = true;
};

struct HostRestorePlan {
    std::vector<bool> blocks;
    std::vector<bool> snapshots;
    // Host slabs the chosen entries take.
    std::uint64_t slabs = 0;
};

// Chooses what an empty Host tier of `host_slabs` restores from a saved one: everything when it
// fits. Otherwise retained snapshots are taken before superseded ones (which a running cache
// evicts first), each group greedily by the GDSF density of §9.3 without its time-dependent base —
// (1 + hits) times the prefill a restore saves from the root, per byte of image, tail and anchor
// path. Each takes its own slabs and the path blocks not yet chosen; one
// that does not fit is passed over for later ones that may. Blocks are chosen only on a chosen
// snapshot's path, so no restored block is dead. `block_parents` lists every saved block's parent
// index (-1 for the root), parents first. With a kept SSD tier, `block_host` says which blocks the
// file holds Host bytes of, and a snapshot shape's `host` the same: the others take no Host slab and
// are never chosen (their SSD copies come back regardless).
[[nodiscard]] HostRestorePlan plan_host_restore(std::span<const std::int32_t> block_parents,
                                                std::span<const SavedSnapshotShape> snapshots,
                                                std::uint64_t host_slabs, const PrefixIndexConfig& config,
                                                std::span<const bool> block_host = {});

} // namespace infernix::runtime::prefix_cache
