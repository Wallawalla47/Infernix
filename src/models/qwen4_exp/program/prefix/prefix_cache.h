#pragma once

// Physical binding of the generic prefix index (runtime/prefix_cache) to Qwen4Exp's KV page pool,
// lane state and pinned Host slab pool (docs/maintainer/qwen3_8-flash-next-design.md §19.3.1). The
// index decides; this class holds what its opaque ids name and moves the bytes its decisions imply.
//
// - A block id owns one Device page lease: a cached full block, or a snapshot's Device tail. Lanes
//   map cached pages into their block tables without a lease of their own; the index's path pins
//   keep a mapped page's node (and so its lease) alive while any lane maps it.
// - Snapshots are Host-born: an image (and a tail page, when the frontier is not a block boundary)
//   is copied straight from a lane to reserved Host slabs, then published. Optional packed Device
//   slots hold images on the Device when the index has slots.
// - Host writes run on the transfer stream after a producer event; restores run on the restore
//   stream in forward order, one event per decoder layer and one for the MTP group, so a lane's
//   first call waits for each layer's state just before that layer reads it.
// - A tap's tail is the lane's own partial page, read on the transfer stream: a page that returns
//   to the pool (the lane's release, a block's eviction) while such a copy is pending orders the
//   compute stream, where its next writer runs, after the copy (order_page_reuse).

#include "core/arena.h"
#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"
#include "core/startup.h"
#include "models/qwen4_exp/program/prefix/state_image.h"
#include "infernix/types.h"
#include "runtime/prefix_cache/host_slab_store.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/persist_file.h"
#include "runtime/prefix_cache/ssd_tier.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace infernix::models::qwen4_exp::prefix {

struct PrefixCacheCounters {
    std::uint64_t admissions                  = 0;
    std::uint64_t reused_tokens               = 0;
    std::uint64_t blocks_inserted             = 0;
    std::uint64_t blocks_reattached           = 0;
    std::uint64_t blocks_duplicate            = 0;
    std::uint64_t taps_created                = 0;
    std::uint64_t taps_skipped                = 0;
    std::uint64_t endpoints_created           = 0;
    std::uint64_t endpoint_skipped_mtp        = 0;
    std::uint64_t endpoint_mismatch_fallbacks = 0;
    std::uint64_t mtp_continuation_mismatches = 0;
    std::uint64_t mtp_branch_mismatches       = 0;
    std::uint64_t host_image_writes           = 0;
    std::uint64_t host_block_writes           = 0;
    std::uint64_t host_image_restores         = 0;
    std::uint64_t host_block_restores         = 0; // prefetched blocks included
    std::uint64_t prefetched_blocks           = 0;
    std::uint64_t host_tail_restores          = 0;
    std::uint64_t host_write_bytes            = 0;
    std::uint64_t host_restore_bytes          = 0;
};

// Geometry of the Host tier: one slab holds one page record (931,840 bytes with INT8 KV, 4 KiB
// aligned); an image spans image_slabs slabs and a snapshot tail one more.
struct PrefixHostLayout {
    HostKVPageLayout page;
    std::size_t slab_bytes        = 0;
    std::size_t image_bytes       = 0;
    std::uint32_t image_slabs     = 0;
    std::uint32_t slabs           = 0; // 0: no Host tier
    std::uint32_t slabs_per_chunk = 0;
};

[[nodiscard]] PrefixHostLayout plan_prefix_host_layout(const KvPageGeometry& pages, const StateImageLayout& image,
                                                       std::uint64_t host_bytes);

struct PrefixCacheConfig {
    runtime::prefix_cache::PrefixIndexConfig index; // capacities filled by the cache from the layouts
    runtime::prefix_cache::TapPlannerConfig taps;
    std::uint32_t kv_pages             = 0; // Device page groups of the pool
    std::uint32_t device_snapshot_slots = 0;
    // An automatic Host tier's growth room (--host-context-mib auto with --ram-resize auto): the tier is sized to this
    // and page-locks its startup size (host_bytes); RAM freed later grows it towards this. 0 (or below host_bytes):
    // the startup size is the most.
    std::uint64_t host_ceiling_bytes = 0;
    // The SSD tier (hybrid-prefix-cache-spec §5.7): its size, rounded down to whole Host slabs, and
    // the directory of its scratch file. 0: none.
    std::uint64_t ssd_bytes = 0;
    std::filesystem::path ssd_directory;
    std::uint64_t ssd_write_bytes_per_hour = 0; // 0: no write budget
    // With a prefix cache file: the SSD tier's kept file (runtime::prefix_cache::kept_ssd_file), whose
    // slabs the file names across restarts; empty: a scratch file that ends with the process.
    std::filesystem::path ssd_kept_file;
};

// One admission's restore: Host pages into new pages, an image into a lane, then the events a
// lane's first call waits on (one per decoder layer, then the MTP group).
struct RestoreTicket {
    std::uint64_t id = 0;
};

class PrefixCache final : public runtime::prefix_cache::PrefixIndexBackend {
public:
    // `compute` is the stream every writer of a pool page runs on or after (a restore opens behind
    // it): a page returned to the pool while a capture still reads it is ordered there.
    PrefixCache(const PrefixCacheConfig& config, DeviceKVPagePool& pool, const KvPageGeometry& pages,
                const LaneStateImage& lanes, std::uint64_t host_bytes, cudaStream_t compute);
    ~PrefixCache() override;

    PrefixCache(const PrefixCache&)            = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    [[nodiscard]] runtime::prefix_cache::PrefixCacheIndex& index() noexcept { return *index_; }
    [[nodiscard]] const runtime::prefix_cache::PrefixCacheIndex& index() const noexcept { return *index_; }
    [[nodiscard]] const runtime::prefix_cache::TapPlannerConfig& taps() const noexcept { return taps_; }
    [[nodiscard]] PrefixCacheCounters& counters() noexcept { return counters_; }
    [[nodiscard]] const PrefixCacheCounters& counters() const noexcept { return counters_; }
    [[nodiscard]] const PrefixHostLayout& host_layout() const noexcept { return host_; }
    [[nodiscard]] bool host_tier() const noexcept { return host_.slabs != 0; }
    [[nodiscard]] std::uint64_t pinned_bytes() const noexcept { return store_.pinned_bytes(); }

    // ---- blocks ------------------------------------------------------------------------------
    // Inserts a committed full block whose page is `lease`. When the index keeps it (a new node,
    // or a Host-only node adopting it), the lease moves into the cache and the result reports
    // device_attached; otherwise `lease` is left with the caller.
    [[nodiscard]] runtime::prefix_cache::InsertResult insert_block(runtime::prefix_cache::NodeRef parent,
                                                                   std::uint64_t lookup_hash,
                                                                   std::span<const TokenId> tokens,
                                                                   std::uint64_t extra, DeviceKVPageLease& lease,
                                                                   bool attach, std::int32_t mtp_next);
    // The token a node's last MTP cell encodes (history[64(b+1)] of its publisher); -1 when unknown.
    [[nodiscard]] std::int32_t mtp_next(runtime::prefix_cache::NodeRef node) const;
    // The Device page of a Device-resident block or snapshot tail.
    [[nodiscard]] DeviceKVPageHandle page(std::uint32_t device_id) const;
    // Adopts a lane's partial page as a snapshot's Device tail; returns its device id.
    [[nodiscard]] std::uint32_t adopt_tail(DeviceKVPageLease&& lease);

    // ---- snapshot meta -----------------------------------------------------------------------
    [[nodiscard]] const StateImageHeader& meta(runtime::prefix_cache::SnapshotRef snapshot) const;

    // ---- Host-born captures ------------------------------------------------------------------
    // Copies lane `lane`'s state image (and, with a tail, the partial page `tail_page`) to reserved
    // Host slabs after `producer` reaches the current point, in forward order, and publishes the
    // snapshot when the copy lands (poll). The lane's next writes to a layer's state, and to that
    // layer's planes of `tail_page`, wait on that layer's event (capture_events).
    struct Capture {
        std::uint64_t id = 0; // the caller's name for capture_result
        runtime::prefix_cache::NodeRef anchor;
        std::uint32_t frontier = 0;
        std::vector<TokenId> tail;
        std::optional<DeviceKVPageHandle> tail_page;
        std::optional<DeviceKVPageLease> tail_lease; // handed over as the Device tail when present
        runtime::prefix_cache::SnapshotKind kind = runtime::prefix_cache::SnapshotKind::Tap;
        double claim = 0.0; // the new snapshot's estimate_priority
        StateImageHeader meta;
        std::optional<runtime::prefix_cache::SnapshotRef> supersedes;
        // When `supersedes` is the snapshot the lane resumed from: the lane's tokens from its frontier
        // on, so the endpoints the lineage went past are superseded too (PrefixCacheIndex::supersede).
        std::vector<TokenId> continuation;
    };
    // Returns false when no slabs could be reserved for its claim (the capture is skipped).
    [[nodiscard]] bool capture(std::uint32_t lane, Capture&& capture, cudaStream_t producer);
    // A capture taken group by group, for a layer walk that passes the frontier inside a span
    // (prefill_walk.cpp): the walk reaches the frontier in each layer at another point of its queue,
    // so each group (with that layer's planes of the tail page) is copied once the walk has enqueued
    // that layer of the frontier's chunk, and the capture is committed after the span. Opening
    // reserves the slabs (nothing when none could be reserved: the capture is skipped) and the
    // events the lane's next writes to each group wait on (staged_events, layer order).
    [[nodiscard]] std::optional<std::uint64_t> open_staged_capture(bool tail, double claim);
    void capture_group(std::uint64_t staged, std::uint32_t group, std::uint32_t lane,
                       std::optional<DeviceKVPageHandle> tail_page, cudaStream_t producer);
    [[nodiscard]] std::span<const cudaEvent_t> staged_events(std::uint64_t staged) const;
    // Every group copied; `capture` as `capture` takes it (claim and tail as opened). Published when
    // the copies land (poll), in commit order with every other capture.
    void commit_staged_capture(std::uint64_t staged, Capture&& capture);
    // A staged capture that will not be committed: its slabs return once its copies have landed.
    void abandon_staged_capture(std::uint64_t staged) noexcept;
    // The capture's copy-out events in layer order (decoder layers, then MTP): each is recorded once
    // that layer's state (and its planes of the tail page) has been read. Empty once it has landed.
    [[nodiscard]] std::span<const cudaEvent_t> capture_events(std::uint64_t id) const noexcept;
    // Orders the compute stream after every pending capture whose tail is one of `pages`, before
    // those pages return to the pool: a tap or boundary reads the lane's partial page on the
    // transfer stream, and another request writing the page under that copy would put its KV into
    // the snapshot. Free once the copies have landed.
    void order_page_reuse(std::span<const DeviceKVPageLease> pages) const;
    // Copies every Device-only node of `nodes` to a Host slab, after `producer`'s queued work.
    void write_blocks(std::span<const runtime::prefix_cache::NodeRef> nodes, cudaStream_t producer);
    // Publishes landed captures and writes and retires landed restores. Non-blocking.
    void poll();
    // Waits for every capture, write and restore in flight and publishes them.
    void drain();

    // ---- SSD tier (hybrid-prefix-cache-spec §5.7) ----------------------------------------------
    [[nodiscard]] runtime::prefix_cache::SsdTier* ssd() noexcept { return ssd_.get(); }
    [[nodiscard]] const runtime::prefix_cache::SsdTier* ssd() const noexcept { return ssd_.get(); }
    // Applies landed SSD transfers and starts write-backs; returns whether a promotion landed since
    // the last call. service_ssd keeps the landing for the next poll_ssd (idle maintenance).
    bool poll_ssd();
    void service_ssd();
    // Before a capture (capture or open_staged_capture) takes `slabs` Host slabs: while the Host tier
    // cannot free them without losing unwritten content, starts the write-backs it can and waits for
    // them to land (SSD tier only).
    void make_lossless_room(std::uint64_t slabs);
    // Whether service_ssd would start a write-back now.
    [[nodiscard]] bool ssd_due() const noexcept { return ssd_ && ssd_->write_back_due(); }
    // Starts reading the parts of `snapshot` only on the SSD into the Host tier.
    [[nodiscard]] runtime::prefix_cache::SsdTier::Promotion promote(runtime::prefix_cache::SnapshotRef snapshot);
    [[nodiscard]] bool transfers_pending() const noexcept {
        return !captures_.empty() || !writes_.empty() || !landing_.empty();
    }
    // Waits for an endpoint copy-out a prompt extends (its anchor is on the prompt's matched `path`
    // and its tail tokens continue the prompt), and publishes it. Only such a prompt waits; the
    // copy is first in the transfer stream, at most an image's copy time. Returns whether it waited.
    [[nodiscard]] bool await_extended_endpoint(std::span<const runtime::prefix_cache::NodeRef> path,
                                               std::span<const TokenId> tokens);
    // The publication of a landed capture (absent while it is in flight); forget it when done.
    [[nodiscard]] std::optional<runtime::prefix_cache::PublishResult> capture_result(std::uint64_t id) const;
    void forget_capture(std::uint64_t id) noexcept;

    // ---- restores (one admission staged at a time) --------------------------------------------
    void open_restore(cudaStream_t producer);
    // A Host-only block into a new cache page (the lease moves into the cache; node Filling).
    void restore_block(runtime::prefix_cache::NodeRef node, DeviceKVPageLease&& destination);
    // A block's Host page into a lane's private page (a copy-on-write anchor), the block pinned
    // until the batch lands.
    void restore_node_page(runtime::prefix_cache::NodeRef node, DeviceKVPageHandle destination);
    // A snapshot's Host tail page into a lane's private page.
    void restore_tail_page(runtime::prefix_cache::SnapshotRef snapshot, DeviceKVPageHandle destination);
    // A snapshot's Host image into a lane.
    void restore_image(runtime::prefix_cache::SnapshotRef snapshot, std::uint32_t lane);
    // Enqueues the batch per decoder layer (that layer's page planes or state parts), then the MTP
    // group, recording one event after each. Returns the ticket the lane's calls wait with.
    [[nodiscard]] RestoreTicket submit_restore();
    [[nodiscard]] bool restore_open() const noexcept { return restore_.open; }
    // The batch's events in layer order (decoder layers, then MTP); empty once it has landed.
    [[nodiscard]] std::span<const cudaEvent_t> restore_events(RestoreTicket ticket) const noexcept;
    // Orders `consumer` after every copy of a landing batch.
    void order_after_restore(RestoreTicket ticket, cudaStream_t consumer) const;
    // Waits on the host for a landing batch (and every batch before it) and retires what landed.
    void await_restore(RestoreTicket ticket);
    void abort_restore() noexcept;

    // Releases every reference the cache holds; the index is rebuilt empty.
    void clear() noexcept;

    // ---- Host tier resize (RAM pressure, --ram-resize; runtime::prefix_cache::HostSlabStore) -----------------
    // Idle or while lanes run. Starts giving back whole chunks worth at least `bytes`, keeping image_slabs + 2 slabs:
    // the lowest-valued entries go, the entries past the cut are copied below it on `worker`; returns the bytes it
    // aims at (0: held entries keep every chunk in use now).
    std::uint64_t begin_host_release(std::uint64_t bytes, HostMemoryWorker& worker);
    // Starts page-locking chunks of at most `bytes` on `worker`, up to the tier's slabs (an automatic tier's growth
    // room included: PrefixCacheConfig::host_ceiling_bytes); returns the bytes asked.
    std::uint64_t begin_host_growth(std::uint64_t bytes, HostMemoryWorker& worker) {
        return store_.begin_growth(bytes, worker);
    }
    // Applies what the worker finished (never waits): a release commits its moves once no restore is in flight.
    runtime::prefix_cache::HostSlabStore::Progress step_host_resize(HostMemoryWorker& worker);
    [[nodiscard]] bool host_resizing() const noexcept { return store_.resizing(); }
    // What a release could give back and a growth take now (pinned_bytes: what the Host tier holds page-locked).
    [[nodiscard]] std::uint64_t host_tier_shrinkable_bytes() const noexcept { return store_.shrinkable_bytes(); }
    [[nodiscard]] std::uint64_t host_tier_growable_bytes() const noexcept { return store_.growable_bytes(); }

    // ---- persistence (persist.cpp, runtime/prefix_cache/persist_file.h) -------------------------
    // Writes every Host-backed snapshot and its block path to `path` (through `path.tmp`, renamed
    // when complete), and with a kept SSD tier the SSD-backed ones by reference (the tier is drained
    // and flushed first). Requires idle Host transfers. `abandoned` stops the save, deleting the
    // temporary file; the previous file stays.
    [[nodiscard]] runtime::prefix_cache::PersistResult save(const std::filesystem::path& path,
                                                            std::string_view fingerprint,
                                                            const CancellationView& abandoned);
    // Rebuilds a saved cache into this empty cache when the fingerprint and geometry match; a
    // smaller Host tier restores the snapshots it values most, and a kept SSD tier that reopened the
    // previous run's file takes back its copies. A mismatch or damaged file loads nothing.
    [[nodiscard]] runtime::prefix_cache::PersistResult load(const std::filesystem::path& path,
                                                            std::string_view fingerprint,
                                                            const StartupObserver& observer);

    // PrefixIndexBackend
    void release_device_block(std::uint32_t device_id) noexcept override;
    void release_snapshot(runtime::prefix_cache::SnapshotRef snapshot) noexcept override;
    void drop_snapshot_device_image(runtime::prefix_cache::SnapshotRef snapshot) noexcept override;

private:
    struct PendingCapture {
        Capture capture;
        runtime::prefix_cache::HostImageReservation image;
        std::vector<cudaEvent_t> events; // decoder layers, then MTP
    };
    struct StagedCapture {
        runtime::prefix_cache::HostImageReservation image;
        bool tail = false;
        std::vector<cudaEvent_t> events; // decoder layers, then MTP; each recorded once its group is copied
        std::uint32_t copied = 0;        // groups copied
    };
    // The image's segments, as HostImage takes them.
    [[nodiscard]] std::vector<std::byte*> image_segments(const runtime::prefix_cache::HostImageReservation& image) const;
    // Group g's copy-out (its layer's planes of a tail page first) on the transfer stream.
    void copy_capture_group(std::uint32_t group, std::uint32_t lane, const HostImage& destination,
                            std::byte* tail_record, std::uint32_t tail_group, std::optional<DeviceKVPageHandle> tail_page);
    struct PendingWrite {
        std::vector<runtime::prefix_cache::NodeRef> nodes;
        cudaEvent_t done = nullptr;
    };
    struct PageCopy {
        DeviceKVPageHandle page;
        std::byte* record = nullptr;
        std::uint32_t group = 0;
    };
    struct RestoreBatch {
        bool open = false;
        RestoreTicket ticket;
        std::vector<runtime::prefix_cache::NodeRef> nodes; // Filling until the batch lands
        std::vector<runtime::prefix_cache::NodeRef> pins;
        std::vector<runtime::prefix_cache::SnapshotRef> snapshot_pins;
        std::vector<PageCopy> pages;
        std::optional<runtime::prefix_cache::SnapshotRef> image;
        std::uint32_t image_lane = 0;
        std::vector<cudaEvent_t> events; // decoder layers, then MTP
    };

    [[nodiscard]] std::uint32_t allocate_block_id(DeviceKVPageLease&& lease);
    [[nodiscard]] std::byte* slab(std::uint32_t id) const noexcept { return store_.slab(id); }
    [[nodiscard]] runtime::prefix_cache::PersistBinding persist_binding();
    [[nodiscard]] std::uint32_t slab_group(std::uint32_t id) const noexcept { return store_.chunk_of(id); }
    [[nodiscard]] cudaEvent_t take_event();
    void recycle(cudaEvent_t event) noexcept;
    void finish_restore(RestoreBatch& batch) noexcept;
    void copy_pages_from_host(std::span<const PageCopy> copies, std::size_t plane_begin, std::size_t plane_end,
                              cudaStream_t stream) const;

    runtime::prefix_cache::TapPlannerConfig taps_;
    DeviceKVPagePool* pool_ = nullptr;
    KvPageGeometry geometry_;
    const LaneStateImage* lanes_ = nullptr;
    PrefixHostLayout host_;
    runtime::prefix_cache::HostSlabStore store_;
    // Declared after the chunks it reads and writes: destroyed first, waiting for its transfers.
    std::unique_ptr<runtime::prefix_cache::SsdTier> ssd_;
    bool ssd_landed_ = false;
    cudaStream_t compute_  = nullptr; // not owned
    cudaStream_t transfer_ = nullptr;
    cudaStream_t restore_stream_ = nullptr;
    std::vector<std::optional<DeviceKVPageLease>> blocks_;
    std::vector<std::uint32_t> free_blocks_;
    std::vector<std::int32_t> node_mtp_next_;      // by node index
    std::vector<StateImageHeader> snapshot_meta_;  // by snapshot index
    std::deque<PendingCapture> captures_;
    std::unordered_map<std::uint64_t, StagedCapture> staged_;
    std::uint64_t next_staged_ = 1;
    std::deque<PendingWrite> writes_;
    std::unordered_map<std::uint64_t, runtime::prefix_cache::PublishResult> published_;
    RestoreBatch restore_;
    std::deque<RestoreBatch> landing_;
    std::uint64_t next_ticket_ = 1;
    std::vector<cudaEvent_t> spare_events_;
    PrefixCacheCounters counters_;
    std::optional<runtime::prefix_cache::PrefixCacheIndex> index_;
};

} // namespace infernix::models::qwen4_exp::prefix
