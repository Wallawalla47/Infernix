#pragma once

// Physical binding of the generic prefix index (runtime/prefix_cache) to Inkling 2's full-visit page pool, lane state
// images and pinned Host slab pool (design docs/maintainer/inkling2_mini-design.md §7.4; the hybrid prefix cache spec,
// docs/maintainer/hybrid-prefix-cache-spec.md). The index decides; this class holds what its opaque ids name and moves
// the bytes its decisions imply. It follows Qwen3.8-Flash-Next's binding (models/qwen4_exp/program/prefix):
//
// - A block id owns one Device page lease: a cached full block, or a snapshot's Device tail. Lanes map cached pages
//   into their block-table rows without a lease of their own; the index's path pins keep a mapped page's node (and so
//   its lease) alive while any lane maps it.
// - Snapshots are Host-born: a lane's state image (its window and conv rings, contiguous) and, when the frontier is not
//   a page boundary, its partial page are copied straight to reserved Host slabs on the transfer stream, then
//   published when the copy lands (poll). There are no Device snapshot slots.
// - Host writes run on the transfer stream after a producer event; a restore runs on the restore stream after the
//   compute stream's queued work, and the restored lane's first call waits for it (one event a batch).
// - A tap's tail is the lane's own partial page read on the transfer stream: a page that returns to the pool while
//   such a copy is pending orders the compute stream, where its next writer runs, after the copy (order_page_reuse).

#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"
#include "core/startup.h"
#include "infernix/types.h"
#include "runtime/prefix_cache/host_slab_store.h"
#include "runtime/prefix_cache/persist_file.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/ssd_tier.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <string_view>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace infernix::models::inkling2::prefix {

struct PrefixCacheCounters {
    std::uint64_t admissions          = 0;
    std::uint64_t reused_tokens       = 0;
    std::uint64_t blocks_inserted     = 0;
    std::uint64_t blocks_reattached   = 0;
    std::uint64_t blocks_duplicate    = 0;
    std::uint64_t taps_created        = 0;
    std::uint64_t taps_skipped        = 0;
    std::uint64_t host_image_writes   = 0;
    std::uint64_t host_block_writes   = 0;
    std::uint64_t host_image_restores = 0;
    std::uint64_t host_block_restores = 0;
    std::uint64_t host_tail_restores  = 0;
    std::uint64_t host_write_bytes    = 0;
    std::uint64_t host_restore_bytes  = 0;
};

// The lanes' state images, each the concatenation of its parts: part p of lane l is `bytes` at device + l * lane_stride
// (the target's state image; with DSpark the drafter's sliding rings).
struct LaneImagePart {
    std::byte* device       = nullptr;
    std::size_t lane_stride = 0;
    std::size_t bytes       = 0;
};
struct LaneImages {
    std::vector<LaneImagePart> parts;
    [[nodiscard]] std::size_t image_bytes() const noexcept {
        std::size_t out = 0;
        for (const LaneImagePart& p : parts) { out += p.bytes; }
        return out;
    }
};

// Geometry of the Host tier: one slab holds one page record (every full visit's planes), 4 KiB aligned; an image spans
// image_slabs slabs and a snapshot tail one more.
struct PrefixHostLayout {
    HostKVPageLayout page;
    std::size_t slab_bytes        = 0;
    std::size_t image_bytes       = 0;
    std::uint32_t image_slabs     = 0;
    std::uint32_t slabs           = 0;
    std::uint32_t slabs_per_chunk = 0;
};

[[nodiscard]] PrefixHostLayout plan_prefix_host_layout(const KVPageGeometry& pages, std::size_t image_bytes,
                                                       std::uint64_t host_bytes);

struct PrefixCacheConfig {
    runtime::prefix_cache::PrefixIndexConfig index; // capacities filled by the cache from the layouts
    runtime::prefix_cache::TapPlannerConfig taps;
    std::uint32_t kv_pages = 0; // Device pages of the pool
    // The SSD tier (hybrid-prefix-cache-spec §5.7): its size (whole Host slabs), the directory of its file, its write
    // budget (0: none) and, with a prefix cache file, the kept file its slabs persist in. 0 bytes: none.
    std::uint64_t ssd_bytes = 0;
    std::filesystem::path ssd_directory;
    std::uint64_t ssd_write_bytes_per_hour = 0;
    std::filesystem::path ssd_kept_file;
    // An automatic Host tier's growth room (--host-context-mib auto with --ram-resize auto): the index is sized for this
    // many bytes, of which the start page-locks its size; a growth links the rest once RAM frees. 0: none.
    std::uint64_t host_ceiling_bytes = 0;
};

struct RestoreTicket {
    std::uint64_t id = 0;
};

class PrefixCache final : public runtime::prefix_cache::PrefixIndexBackend {
public:
    // `compute` is the stream every writer of a pool page runs on (a restore opens behind it): a page returned to the
    // pool while a capture still reads it is ordered there.
    PrefixCache(const PrefixCacheConfig& config, DeviceKVPagePool& pool, const LaneImages& lanes,
                std::uint64_t host_bytes, cudaStream_t compute);
    ~PrefixCache() override;

    PrefixCache(const PrefixCache&)            = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    [[nodiscard]] runtime::prefix_cache::PrefixCacheIndex& index() noexcept { return *index_; }
    [[nodiscard]] const runtime::prefix_cache::PrefixCacheIndex& index() const noexcept { return *index_; }
    [[nodiscard]] const runtime::prefix_cache::TapPlannerConfig& taps() const noexcept { return taps_; }
    [[nodiscard]] PrefixCacheCounters& counters() noexcept { return counters_; }
    [[nodiscard]] const PrefixCacheCounters& counters() const noexcept { return counters_; }
    [[nodiscard]] const PrefixHostLayout& host_layout() const noexcept { return host_; }
    [[nodiscard]] std::uint64_t pinned_bytes() const noexcept { return store_.pinned_bytes(); }
    // The slabs the Host tier holds now (its locked chunks'; host_layout().slabs includes an automatic tier's growth
    // room).
    [[nodiscard]] std::uint32_t pinned_slabs() const noexcept { return store_.pinned_slabs(); }

    // ---- blocks ------------------------------------------------------------------------------
    // Inserts a committed full block whose page is `lease`. When the index keeps it (a new node, or an existing node
    // adopting it), the lease moves into the cache and the result reports device_attached; otherwise `lease` is left
    // with the caller.
    [[nodiscard]] runtime::prefix_cache::InsertResult insert_block(runtime::prefix_cache::NodeRef parent,
                                                                   std::uint64_t lookup_hash,
                                                                   std::span<const TokenId> tokens,
                                                                   std::uint64_t extra, DeviceKVPageLease& lease);
    // The Device page of a Device-resident block or snapshot tail.
    [[nodiscard]] DeviceKVPageHandle page(std::uint32_t device_id) const;

    // ---- Host-born captures ------------------------------------------------------------------
    // Copies lane `lane`'s image (and, with a tail, the partial page `tail_page`) to reserved Host slabs after
    // `producer` reaches the current point, then the Device-only blocks of `path` (the snapshot's blocks below its
    // frontier, root first: a snapshot outlives its blocks' Device copies only with their Host copies) behind it, and
    // publishes the snapshot when the copy lands (poll). The lane's next writes to its image wait on
    // capture_events(id) (the image's copy, not the blocks').
    struct Capture {
        std::uint64_t id = 0; // the caller's name for capture_result
        runtime::prefix_cache::NodeRef anchor;
        std::uint32_t frontier = 0;
        std::vector<TokenId> tail;
        std::optional<DeviceKVPageHandle> tail_page;
        runtime::prefix_cache::SnapshotKind kind = runtime::prefix_cache::SnapshotKind::Tap;
        double claim = 0.0; // the new snapshot's estimate_priority
        std::optional<runtime::prefix_cache::SnapshotRef> supersedes;
    };
    // Returns false when no slabs could be reserved for its claim (the capture is skipped).
    [[nodiscard]] bool capture(std::uint32_t lane, Capture&& capture, std::span<const runtime::prefix_cache::NodeRef> path,
                               cudaStream_t producer);
    // The capture's copy-out event (recorded once the image and tail have been read); empty once it has landed.
    [[nodiscard]] std::span<const cudaEvent_t> capture_events(std::uint64_t id) const noexcept;
    // Orders the compute stream after every pending capture whose tail is one of `pages`, before those pages return to
    // the pool. Free once the copies have landed.
    void order_page_reuse(std::span<const DeviceKVPageLease> pages) const;
    // Copies every Device-only node of `nodes` to a Host slab, after `producer`'s queued work.
    void write_blocks(std::span<const runtime::prefix_cache::NodeRef> nodes, cudaStream_t producer);
    // Publishes landed captures and writes and retires landed restores. Non-blocking.
    void poll();
    // Waits for every capture, write and restore in flight and publishes them.
    void drain();
    // Room-making's wait: for the oldest transfer whose landing can free Device memory. False when nothing in flight
    // could free anything.
    bool settle_oldest_transfer();
    [[nodiscard]] bool transfers_pending() const noexcept {
        return !captures_.empty() || !writes_.empty() || !landing_.empty();
    }
    // Waits for the copy-outs a prompt extends (a capture anchored on its matched `path` whose tail continues it) and
    // publishes them; returns whether it waited.
    [[nodiscard]] bool await_extended_capture(std::span<const runtime::prefix_cache::NodeRef> path,
                                              std::span<const TokenId> tokens);
    [[nodiscard]] std::optional<runtime::prefix_cache::PublishResult> capture_result(std::uint64_t id) const;
    void forget_capture(std::uint64_t id) noexcept;

    // ---- restores (one admission staged at a time) --------------------------------------------
    void open_restore(cudaStream_t producer);
    // A Host-only block into a new cache page (the lease moves into the cache; node Filling).
    void restore_block(runtime::prefix_cache::NodeRef node, DeviceKVPageLease&& destination);
    // A snapshot's Host tail page into a lane's private page.
    void restore_tail_page(runtime::prefix_cache::SnapshotRef snapshot, DeviceKVPageHandle destination);
    // A snapshot's Host image into a lane.
    void restore_image(runtime::prefix_cache::SnapshotRef snapshot, std::uint32_t lane);
    // Enqueues the batch; returns the ticket the lane's first call waits with (order_after_restore).
    [[nodiscard]] RestoreTicket submit_restore();
    [[nodiscard]] bool restore_open() const noexcept { return restore_.open; }
    void order_after_restore(RestoreTicket ticket, cudaStream_t consumer) const;
    void abort_restore() noexcept;
    // A failure with the batch open, possibly part enqueued: waits for the restore stream, then aborts the batch.
    void fail_restore() noexcept;

    // ---- SSD tier (hybrid-prefix-cache-spec §5.7) ---------------------------------------------
    [[nodiscard]] runtime::prefix_cache::SsdTier* ssd() noexcept { return ssd_.get(); }
    [[nodiscard]] const runtime::prefix_cache::SsdTier* ssd() const noexcept { return ssd_.get(); }
    // Applies landed SSD transfers and starts write-backs; returns whether a promotion landed since the last call.
    bool poll_ssd();
    // Starts reading the parts of `snapshot` only on the SSD into the Host tier.
    [[nodiscard]] runtime::prefix_cache::SsdTier::Promotion promote(runtime::prefix_cache::SnapshotRef snapshot);
    [[nodiscard]] bool ssd_due() const noexcept { return ssd_ && ssd_->write_back_due(); }
    [[nodiscard]] std::optional<std::chrono::steady_clock::time_point> ssd_retry_at() const noexcept {
        return ssd_ ? ssd_->write_back_retry_at() : std::nullopt;
    }

    // Releases every reference the cache holds; the index is rebuilt empty.
    void clear() noexcept;

    // ---- Host tier resize (RAM pressure, --ram-resize; runtime::prefix_cache::HostSlabStore) -----------------
    // Idle or while lanes run. Starts giving back whole chunks worth at least `bytes`, keeping image_slabs + 2 slabs:
    // the lowest-valued entries go, the entries past the cut are copied below it on `worker`; returns the bytes it aims
    // at (0: held entries keep every chunk in use now).
    std::uint64_t begin_host_release(std::uint64_t bytes, HostMemoryWorker& worker);
    // Starts page-locking chunks of at most `bytes` on `worker`, up to the tier's slabs (an automatic tier's growth
    // room included: PrefixCacheConfig::host_ceiling_bytes); returns the bytes asked.
    std::uint64_t begin_host_growth(std::uint64_t bytes, HostMemoryWorker& worker) {
        return store_.begin_growth(bytes, worker);
    }
    // Applies what the worker finished (never waits): a release commits its moves once no restore is in flight.
    runtime::prefix_cache::HostSlabStore::Progress step_host_resize(HostMemoryWorker& worker);
    [[nodiscard]] bool host_resizing() const noexcept { return store_.resizing(); }
    // What a release could give back and a growth take now.
    [[nodiscard]] std::uint64_t host_tier_shrinkable_bytes() const noexcept { return store_.shrinkable_bytes(); }
    [[nodiscard]] std::uint64_t host_tier_growable_bytes() const noexcept { return store_.growable_bytes(); }

    // ---- persistence (runtime/prefix_cache/persist_file.h) ------------------------------------
    // Writes every Host-backed snapshot and its block path to `path` (through `path.tmp`). Requires idle transfers.
    [[nodiscard]] runtime::prefix_cache::PersistResult save(const std::filesystem::path& path,
                                                            std::string_view fingerprint,
                                                            const CancellationView& abandoned);
    // Rebuilds a saved cache into this empty cache when the fingerprint and geometry match.
    [[nodiscard]] runtime::prefix_cache::PersistResult load(const std::filesystem::path& path,
                                                            std::string_view fingerprint,
                                                            const StartupObserver& observer);

    // PrefixIndexBackend
    void release_device_block(std::uint32_t device_id) noexcept override;

private:
    struct PendingCapture {
        Capture capture;
        runtime::prefix_cache::HostImageReservation image;
        cudaEvent_t done = nullptr;
    };
    struct PendingWrite {
        std::vector<runtime::prefix_cache::NodeRef> nodes;
        cudaEvent_t done = nullptr;
    };
    struct PageCopy {
        DeviceKVPageHandle page;
        std::byte* record   = nullptr;
        std::uint32_t group = 0;
    };
    struct RestoreBatch {
        bool open = false;
        RestoreTicket ticket;
        std::vector<runtime::prefix_cache::NodeRef> nodes; // Filling until the batch lands
        std::vector<runtime::prefix_cache::NodeRef> pins;  // restore destinations (pin_node)
        std::vector<runtime::prefix_cache::SnapshotRef> snapshot_pins;
        std::vector<PageCopy> pages;
        std::optional<runtime::prefix_cache::SnapshotRef> image;
        std::uint32_t image_lane = 0;
        cudaEvent_t done         = nullptr;
    };

    [[nodiscard]] runtime::prefix_cache::PersistBinding persist_binding();
    [[nodiscard]] std::uint32_t allocate_block_id(DeviceKVPageLease&& lease);
    [[nodiscard]] std::byte* slab(std::uint32_t id) const noexcept { return store_.slab(id); }
    [[nodiscard]] std::uint32_t slab_group(std::uint32_t id) const noexcept { return store_.chunk_of(id); }
    [[nodiscard]] cudaEvent_t take_event();
    void recycle(cudaEvent_t event) noexcept;
    // The lane's image between its device bytes and the image slabs, runs of consecutive slabs as one copy.
    void copy_image(std::uint32_t lane, std::span<const std::uint32_t> slabs, bool to_host, cudaStream_t stream) const;
    void finish_restore(RestoreBatch& batch) noexcept;
    // Before a capture takes `slabs` Host slabs (its image, its tail and its path's blocks): while the Host tier cannot
    // free them without losing unwritten content, lands the captures and block writes in flight (a snapshot is written
    // back only once it and its path are on the Host), starts the write-backs it can and waits for them (SSD tier only).
    void make_lossless_room(std::uint64_t slabs);
    void service_ssd();

    runtime::prefix_cache::TapPlannerConfig taps_;
    DeviceKVPagePool* pool_ = nullptr;
    LaneImages lanes_;
    PrefixHostLayout host_;
    runtime::prefix_cache::HostSlabStore store_;
    // Declared after the slabs it reads and writes: destroyed first, waiting for its transfers.
    std::unique_ptr<runtime::prefix_cache::SsdTier> ssd_;
    bool ssd_landed_ = false;
    cudaStream_t compute_        = nullptr; // not owned
    cudaStream_t transfer_       = nullptr;
    cudaStream_t restore_stream_ = nullptr;
    std::vector<std::optional<DeviceKVPageLease>> blocks_;
    std::vector<std::uint32_t> free_blocks_;
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

} // namespace infernix::models::inkling2::prefix
