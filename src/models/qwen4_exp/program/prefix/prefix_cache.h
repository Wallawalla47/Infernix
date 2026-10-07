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

#include "core/arena.h"
#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"
#include "core/startup.h"
#include "models/qwen4_exp/program/prefix/state_image.h"
#include "infernix/types.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <deque>
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

// The outcome of saving or loading the Host tier (persist.cpp).
struct PersistResult {
    bool ok = false;
    std::string message; // why nothing was saved or loaded
    std::uint64_t blocks    = 0;
    std::uint64_t snapshots = 0;
    std::uint64_t bytes     = 0;
    double seconds          = 0.0;
    // Load only: the file's entries and the Host tier bytes they all take, against this tier.
    std::uint64_t saved_blocks        = 0;
    std::uint64_t saved_snapshots     = 0;
    std::uint64_t required_host_bytes = 0;
    std::uint64_t host_bytes          = 0;
};

struct PrefixCacheConfig {
    runtime::prefix_cache::PrefixIndexConfig index; // capacities filled by the cache from the layouts
    runtime::prefix_cache::TapPlannerConfig taps;
    std::uint32_t kv_pages             = 0; // Device page groups of the pool
    std::uint32_t device_snapshot_slots = 0;
};

// One admission's restore: Host pages into new pages, an image into a lane, then the events a
// lane's first call waits on (one per decoder layer, then the MTP group).
struct RestoreTicket {
    std::uint64_t id = 0;
};

class PrefixCache final : public runtime::prefix_cache::PrefixIndexBackend {
public:
    PrefixCache(const PrefixCacheConfig& config, DeviceKVPagePool& pool, const KvPageGeometry& pages,
                const LaneStateImage& lanes, std::uint64_t host_bytes);
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
    [[nodiscard]] std::uint64_t pinned_bytes() const noexcept;

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
    };
    // Returns false when no slabs could be reserved for its claim (the capture is skipped).
    [[nodiscard]] bool capture(std::uint32_t lane, Capture&& capture, cudaStream_t producer);
    // The capture's copy-out events in layer order (decoder layers, then MTP): each is recorded once
    // that layer's state (and its planes of the tail page) has been read. Empty once it has landed.
    [[nodiscard]] std::span<const cudaEvent_t> capture_events(std::uint64_t id) const noexcept;
    // Copies every Device-only node of `nodes` to a Host slab, after `producer`'s queued work.
    void write_blocks(std::span<const runtime::prefix_cache::NodeRef> nodes, cudaStream_t producer);
    // Publishes landed captures and writes and retires landed restores. Non-blocking.
    void poll();
    // Waits for every capture, write and restore in flight and publishes them.
    void drain();
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

    // ---- persistence (persist.cpp) -------------------------------------------------------------
    // Writes every Host-backed snapshot and its Host block path to `path` (through `path.tmp`,
    // renamed when complete). Requires idle transfers. `abandoned` stops the save, deleting the
    // temporary file; the previous file stays.
    [[nodiscard]] PersistResult save(const std::filesystem::path& path, std::string_view fingerprint,
                                     const CancellationView& abandoned) const;
    // Rebuilds a saved Host tier into this empty cache when the fingerprint and geometry match; a
    // smaller tier restores the snapshots it values most. A mismatch or damaged file loads nothing.
    [[nodiscard]] PersistResult load(const std::filesystem::path& path, std::string_view fingerprint,
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
    [[nodiscard]] std::byte* slab(std::uint32_t id) const noexcept;
    [[nodiscard]] std::uint32_t slab_group(std::uint32_t id) const noexcept { return id / host_.slabs_per_chunk; }
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
    std::vector<PinnedHostBuffer> chunks_;
    cudaStream_t transfer_ = nullptr;
    cudaStream_t restore_stream_ = nullptr;
    std::vector<std::optional<DeviceKVPageLease>> blocks_;
    std::vector<std::uint32_t> free_blocks_;
    std::vector<std::int32_t> node_mtp_next_;      // by node index
    std::vector<StateImageHeader> snapshot_meta_;  // by snapshot index
    std::deque<PendingCapture> captures_;
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
