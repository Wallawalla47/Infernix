#include "models/qwen4_exp/program/prefix/prefix_cache.h"

#include "core/device.h"
#include "models/qwen4_exp/program/prefix/test_taps.h"
#include "runtime/prefix_cache/prefix_holds.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace infernix::models::qwen4_exp::prefix {

namespace pc = runtime::prefix_cache;

namespace {

// The Host tier is page-locked in chunks of whole slabs (HostSlabStore::kChunkBytes, also its resize step under RAM
// pressure); a slab never crosses chunks.
constexpr std::size_t kHostChunkBytes = pc::HostSlabStore::kChunkBytes;
constexpr std::size_t kSlabAlignment  = 4096;

std::size_t align_slab(std::size_t bytes) { return (bytes + kSlabAlignment - 1U) / kSlabAlignment * kSlabAlignment; }

} // namespace

PrefixHostLayout plan_prefix_host_layout(const KvPageGeometry& pages, const StateImageLayout& image,
                                         std::uint64_t host_bytes) {
    PrefixHostLayout out;
    out.page        = plan_host_kv_page_layout(pages.geometry);
    out.slab_bytes  = align_slab(out.page.page_stride);
    out.image_bytes = image.image_bytes;
    out.image_slabs = static_cast<std::uint32_t>((image.image_bytes + out.slab_bytes - 1U) / out.slab_bytes);
    const std::uint64_t slabs = host_bytes / out.slab_bytes;
    if (slabs > pc::kNoId / 4U) { throw std::invalid_argument("Qwen4Exp prefix cache: the Host tier is too large"); }
    out.slabs           = static_cast<std::uint32_t>(slabs);
    out.slabs_per_chunk = static_cast<std::uint32_t>(std::max<std::size_t>(1U, kHostChunkBytes / out.slab_bytes));
    return out;
}

PrefixCache::PrefixCache(const PrefixCacheConfig& config, DeviceKVPagePool& pool, const KvPageGeometry& pages,
                         const LaneStateImage& lanes, std::uint64_t host_bytes, cudaStream_t compute)
    : taps_(config.taps), pool_(&pool), geometry_(pages), lanes_(&lanes),
      host_(plan_prefix_host_layout(pages, lanes.layout(), std::max(host_bytes, config.host_ceiling_bytes))),
      compute_(compute) {
    if (pool.plane_count() != pages.geometry.planes.size()) {
        throw std::invalid_argument("Qwen4Exp prefix cache: the page geometry does not match the pool");
    }
    // The slabs page-locked at the start; the rest of host_.slabs is growth room.
    const auto locked =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(host_.slabs, host_bytes / std::max<std::size_t>(host_.slab_bytes, 1U)));
    if (locked == 0) {
        // Host-born snapshots are the only snapshot source: without a Host tier there is no cache.
        throw std::invalid_argument("Qwen4Exp prefix cache needs a Host tier (--host-context-mib)");
    }
    if (locked < host_.image_slabs + 2U) {
        const std::uint64_t mib = (static_cast<std::uint64_t>(host_.image_slabs + 2U) * host_.slab_bytes + (1U << 20U) - 1U) >> 20U;
        throw std::invalid_argument("--host-context-mib is too small for the Qwen4Exp prefix cache: one snapshot needs " +
                                    std::to_string(mib) + " MiB");
    }
    // The SSD tier holds whole Host slabs (hybrid-prefix-cache-spec §5.7).
    const std::uint64_t ssd_slabs = config.ssd_bytes / host_.slab_bytes;
    if (config.ssd_bytes != 0 && ssd_slabs < host_.image_slabs + 2U) {
        throw std::invalid_argument("--ssd-context-gib is too small for the Qwen4Exp prefix cache: one snapshot needs " +
                                    std::to_string(((host_.image_slabs + 2ULL) * host_.slab_bytes + (1ULL << 30U) - 1U) >> 30U) +
                                    " GiB");
    }
    if (ssd_slabs > pc::kNoId / 4U) { throw std::invalid_argument("Qwen4Exp prefix cache: the SSD tier is too large"); }
    pc::PrefixIndexConfig index = config.index;
    index.max_nodes             = config.kv_pages + host_.slabs + static_cast<std::uint32_t>(ssd_slabs) + 1U;
    index.max_snapshots         = config.device_snapshot_slots +
                          (host_.slabs + static_cast<std::uint32_t>(ssd_slabs)) / host_.image_slabs + 1U;
    index.host_slabs            = host_.slabs;
    index.ssd_slabs             = static_cast<std::uint32_t>(ssd_slabs);
    index.image_slabs           = host_.image_slabs;
    index.device_snapshot_slots = config.device_snapshot_slots;
    index.block_bytes           = host_.page.page_stride;
    index.image_bytes           = host_.image_bytes;
    index.host_blocks           = true;
    // The DFlash2 ring only steers drafts (dflash2_program.cpp): SSD copies leave out the image slabs
    // holding nothing else, and a promotion leaves zeros there (a stale ring costs acceptance only).
    const StateImageLayout& image_layout = lanes.layout();
    if (image_layout.spec.draft_ring.layers != 0 && ssd_slabs != 0) {
        const auto ring = std::find_if(image_layout.parts.begin(), image_layout.parts.end(), [](const StateImagePart& part) {
            return part.kind == StateImagePart::Kind::DraftRingK;
        });
        if (ring != image_layout.parts.end()) {
            index.ssd_image_slabs = static_cast<std::uint32_t>((ring->offset + host_.slab_bytes - 1U) / host_.slab_bytes);
        }
    }
    if (config.device_snapshot_slots != 0) {
        throw std::invalid_argument("Qwen4Exp prefix cache: Device snapshot slots are not supported");
    }
    store_ = pc::HostSlabStore(host_.slab_bytes, host_.slabs, locked, kHostChunkBytes, "Qwen4Exp prefix cache Host tier");
    // A release keeps one snapshot and its block in service.
    store_.set_floor_chunks((host_.image_slabs + 2U + host_.slabs_per_chunk - 1U) / host_.slabs_per_chunk);
    // The destructor does not run for a constructor that throws (the SSD tier's file cannot be opened, the second
    // stream is refused): the streams made so far are destroyed here, the members unwind by themselves.
    try {
        CUDA_CHECK(cudaStreamCreateWithFlags(&transfer_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&restore_stream_, cudaStreamNonBlocking));
        if (index.ssd_slabs != 0) {
            // Its slabs are Host slabs byte for byte (page records and image segments), so a transfer is a
            // plain copy, and a snapshot's meta stays with its index entry.
            ssd_ = std::make_unique<pc::SsdTier>(config.ssd_directory, index.ssd_slabs, host_.slab_bytes,
                                                 [this](std::uint32_t id) { return slab(id); },
                                                 config.ssd_write_bytes_per_hour, /*codec=*/true, config.ssd_kept_file);
        }
        node_mtp_next_.assign(index.max_nodes, -1);
        snapshot_meta_.assign(index.max_snapshots, StateImageHeader{});
        blocks_.reserve(index.max_nodes);
        index_.emplace(index, *this);
        store_.limit_index(*index_); // only the locked slabs serve until a growth links more
    } catch (...) {
        ssd_.reset(); // its threads end before the slabs they would read
        index_.reset();
        for (cudaStream_t stream : {transfer_, restore_stream_}) {
            if (stream != nullptr) { (void)cudaStreamDestroy(stream); }
        }
        throw;
    }
}

PrefixCache::~PrefixCache() {
    // Only CUDA objects are released here; leases go back to the pool with blocks_.
    for (cudaStream_t stream : {transfer_, restore_stream_}) {
        if (stream != nullptr) {
            (void)cudaStreamSynchronize(stream);
            (void)cudaStreamDestroy(stream);
        }
    }
    for (PendingCapture& c : captures_) {
        for (cudaEvent_t e : c.events) { (void)cudaEventDestroy(e); }
    }
    for (auto& [id, s] : staged_) {
        for (cudaEvent_t e : s.events) { (void)cudaEventDestroy(e); }
    }
    for (PendingWrite& w : writes_) {
        if (w.done != nullptr) { (void)cudaEventDestroy(w.done); }
    }
    for (cudaEvent_t e : restore_.events) { (void)cudaEventDestroy(e); }
    for (RestoreBatch& batch : landing_) {
        for (cudaEvent_t e : batch.events) { (void)cudaEventDestroy(e); }
    }
    for (cudaEvent_t e : spare_events_) { (void)cudaEventDestroy(e); }
    index_.reset();
}

// ---- ids, events, slabs ----------------------------------------------------------------------------

std::uint32_t PrefixCache::allocate_block_id(DeviceKVPageLease&& lease) {
    if (!free_blocks_.empty()) {
        const std::uint32_t id = free_blocks_.back();
        free_blocks_.pop_back();
        blocks_[id].emplace(std::move(lease));
        return id;
    }
    if (blocks_.size() >= pc::kNoId - 1U) { throw std::overflow_error("Qwen4Exp prefix cache block ids exhausted"); }
    blocks_.emplace_back(std::move(lease));
    return static_cast<std::uint32_t>(blocks_.size() - 1U);
}

DeviceKVPageHandle PrefixCache::page(std::uint32_t device_id) const {
    if (device_id >= blocks_.size() || !blocks_[device_id]) {
        throw std::logic_error("Qwen4Exp prefix cache block id is not registered");
    }
    return blocks_[device_id]->handle();
}

cudaEvent_t PrefixCache::take_event() {
    if (!spare_events_.empty()) {
        cudaEvent_t event = spare_events_.back();
        spare_events_.pop_back();
        return event;
    }
    cudaEvent_t event = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
}

void PrefixCache::recycle(cudaEvent_t event) noexcept {
    if (event != nullptr) { spare_events_.push_back(event); }
}

// ---- blocks --------------------------------------------------------------------------------------

pc::InsertResult PrefixCache::insert_block(pc::NodeRef parent, std::uint64_t lookup_hash,
                                           std::span<const TokenId> tokens, std::uint64_t extra,
                                           DeviceKVPageLease& lease, bool attach, std::int32_t mtp_next) {
    if (!lease.valid()) { throw std::invalid_argument("Qwen4Exp prefix cache: a block needs its page"); }
    const DeviceKVPageHandle handle = lease.handle();
    const std::uint32_t id          = allocate_block_id(std::move(lease));
    pc::InsertResult result;
    try {
        result = index_->insert_block(parent, lookup_hash, tokens, extra, id, attach);
    } catch (...) {
        lease = std::move(*blocks_[id]);
        blocks_[id].reset();
        free_blocks_.push_back(id);
        throw;
    }
    if (!result.device_attached) {
        // The index kept an existing copy: the page stays the caller's private page.
        lease = std::move(*blocks_[id]);
        blocks_[id].reset();
        free_blocks_.push_back(id);
        ++counters_.blocks_duplicate;
    } else {
        // The replaced Device copy is unmapped and unread (the index replaces only unpinned ones),
        // so its page returns to the pool as an eviction's would.
        if (result.replaced_device_id != pc::kNoId) { release_device_block(result.replaced_device_id); }
        node_mtp_next_[result.node.index] = mtp_next;
        ++(result.inserted ? counters_.blocks_inserted : counters_.blocks_reattached);
    }
    (void)handle;
    return result;
}

std::int32_t PrefixCache::mtp_next(pc::NodeRef node) const {
    if (!index_->valid(node)) { throw std::logic_error("Qwen4Exp prefix cache: stale node"); }
    return node_mtp_next_[node.index];
}

std::uint32_t PrefixCache::adopt_tail(DeviceKVPageLease&& lease) { return allocate_block_id(std::move(lease)); }

const StateImageHeader& PrefixCache::meta(pc::SnapshotRef snapshot) const {
    if (!index_->valid(snapshot)) { throw std::logic_error("Qwen4Exp prefix cache: stale snapshot"); }
    return snapshot_meta_[snapshot.index];
}

// ---- captures and write-through ------------------------------------------------------------------

bool PrefixCache::capture(std::uint32_t lane, Capture&& capture, cudaStream_t producer) {
    const bool tail = !capture.tail.empty();
    if (tail != capture.tail_page.has_value() || capture.tail.size() >= pc::kBlockTokens) {
        throw std::invalid_argument("Qwen4Exp prefix cache: a capture's tail does not match its page");
    }
    if (ssd_) { make_lossless_room(host_.image_slabs + (tail ? 1U : 0U)); }
    std::optional<pc::SnapshotRef> superseded;
    auto reserved = reserve_capture(tail, capture.claim, capture.replaces_source, capture.continuation,
                                    capture.replaces_opener, superseded);
    if (!reserved) {
        ++counters_.taps_skipped;
        return false;
    }
    capture.source_superseded = superseded.has_value();
    // Until the capture is listed, the reservation and the anchor's retention are held here: owned by nothing else,
    // they would shrink the Host tier until clear(). Declared before the try, so they go back only after the catch
    // has let the copies enqueued so far land (the slabs go back once idle).
    pc::HostImageHold image(*index_, std::move(*reserved));
    // The anchor must outlive the copy: the capturing lane may release its path before it lands.
    pc::NodeRetention anchor;
    if (capture.anchor.valid()) { anchor = pc::NodeRetention(*index_, capture.anchor); }
    const std::optional<DeviceKVPageHandle> tail_page = capture.tail_page;
    PendingCapture pending{std::move(capture), {}, {}};
    cudaEvent_t start = nullptr;
    try {
        start = take_event();
        CUDA_CHECK(cudaEventRecord(start, producer));
        CUDA_CHECK(cudaStreamWaitEvent(transfer_, start, 0));
        recycle(std::exchange(start, nullptr));
        const std::vector<std::byte*> segments = image_segments(image.get());
        write_state_image_header(lanes_->layout(), pending.capture.meta, segments.front());
        const HostImage destination{segments, host_.slab_bytes};
        std::byte* const tail_record   = tail ? slab(image.get().slabs.back()) : nullptr;
        const std::uint32_t tail_group = tail ? slab_group(image.get().slabs.back()) : 0U;
        pending.events.reserve(lanes_->layout().groups());
        for (std::uint32_t g = 0; g < lanes_->layout().groups(); ++g) {
            copy_capture_group(g, lane, destination, tail_record, tail_group, tail_page);
            cudaEvent_t event = take_event();
            pending.events.push_back(event); // reserved
            CUDA_CHECK(cudaEventRecord(event, transfer_));
        }
        captures_.push_back(std::move(pending)); // a deque: unchanged when its new block cannot be allocated
    } catch (...) {
        // A failed copy (a device fault, an allocation): the copies enqueued so far land before the holds go.
        (void)cudaStreamSynchronize(transfer_);
        if (start != nullptr) { recycle(start); }
        for (cudaEvent_t event : pending.events) { recycle(event); }
        if (superseded && index_->valid(*superseded)) { index_->unsupersede(*superseded); }
        throw;
    }
    captures_.back().image = image.release(); // poll publishes it
    (void)anchor.release();                   // poll unretains it
    counters_.host_write_bytes += host_.image_bytes + (tail ? host_.page.page_stride : 0U);
    ++counters_.host_image_writes;
    return true;
}

std::vector<std::byte*> PrefixCache::image_segments(const pc::HostImageReservation& image) const {
    std::vector<std::byte*> segments;
    segments.reserve(host_.image_slabs);
    for (std::uint32_t i = 0; i < host_.image_slabs; ++i) { segments.push_back(slab(image.slabs[i])); }
    return segments;
}

void PrefixCache::copy_capture_group(std::uint32_t group, std::uint32_t lane, const HostImage& destination,
                                     std::byte* tail_record, std::uint32_t tail_group,
                                     std::optional<DeviceKVPageHandle> tail_page) {
    const StateImageLayout& layout = lanes_->layout();
    // KV layer of the group: attention layers in order, then MTP's (when the model keeps one).
    std::optional<std::uint32_t> kv_layer;
    std::uint32_t attention = 0;
    for (std::uint32_t g = 0; g < group && g < layout.spec.layer_types.size(); ++g) {
        if (layout.spec.layer_types[g] == MixerKind::Attention) { ++attention; }
    }
    if (group == layout.drafter_group()) {
        if (geometry_.kv_layers > attention) { kv_layer = attention; }
    } else if (layout.spec.layer_types[group] == MixerKind::Attention) {
        kv_layer = attention;
    }
    if (tail_page && kv_layer) {
        const PlaneRange planes            = kv_layer_planes(geometry_, *kv_layer);
        std::byte* const records[1]        = {tail_record};
        const std::uint32_t groups[1]      = {tail_group};
        const DeviceKVPageHandle pages[1]  = {*tail_page};
        pool_->copy_to_host_records(pages, records, groups, host_.page, planes.begin, planes.end, transfer_);
    }
    lanes_->copy_group_to_host(group, lane, destination, transfer_);
}

std::optional<pc::SnapshotRef> PrefixCache::published_opener(std::uint64_t id) const {
    const auto found = published_.find(id);
    if (found == published_.end() || !found->second.snapshot.valid() || !index_->valid(found->second.snapshot)) {
        return std::nullopt;
    }
    const pc::SnapshotRef snapshot = found->second.snapshot;
    if (index_->snapshot(snapshot).kind != pc::SnapshotKind::Tap || !snapshot_meta_[snapshot.index].opener) {
        return std::nullopt;
    }
    return snapshot;
}

std::optional<pc::HostImageReservation>
PrefixCache::reserve_capture(bool tail, double claim, std::optional<pc::SnapshotRef> replaces_source,
                             std::span<const TokenId> continuation, std::uint64_t replaces_opener,
                             std::optional<pc::SnapshotRef>& superseded) {
    superseded.reset();
    auto reserved = index_->reserve_host_image(tail, claim);
    if (reserved) { return reserved; }
    if (replaces_source && index_->valid(*replaces_source)) {
        const bool changed = index_->supersede(*replaces_source, continuation);
        reserved           = index_->reserve_host_image(tail, claim);
        if (reserved) {
            if (changed) { superseded = replaces_source; }
            return reserved;
        }
        if (changed && index_->valid(*replaces_source)) { index_->unsupersede(*replaces_source); }
    }
    if (replaces_opener != 0) {
        (void)await_capture(replaces_opener); // in flight it is a reservation, which nothing can take
        if (const auto opener = published_opener(replaces_opener)) {
            const bool changed = index_->supersede(*opener);
            reserved           = index_->reserve_host_image(tail, claim);
            if (!reserved && changed && index_->valid(*opener)) { index_->unsupersede(*opener); }
        }
    }
    return reserved;
}

std::optional<std::uint64_t> PrefixCache::open_staged_capture(bool tail, double claim,
                                                              std::optional<pc::SnapshotRef> replaces_source,
                                                              std::span<const TokenId> continuation) {
    if (ssd_) { make_lossless_room(host_.image_slabs + (tail ? 1U : 0U)); }
    std::optional<pc::SnapshotRef> superseded;
    auto reserved = reserve_capture(tail, claim, replaces_source, continuation, 0, superseded);
    if (!reserved) {
        ++counters_.taps_skipped;
        return std::nullopt;
    }
    // Held until the staged capture is listed: an event that cannot be created, or the map's node, would otherwise
    // lose the reservation (and the events taken) until clear().
    pc::HostImageHold image(*index_, std::move(*reserved));
    StagedCapture staged{{}, tail, {}, 0, superseded};
    try {
        staged.events.reserve(lanes_->layout().groups());
        for (std::uint32_t g = 0; g < lanes_->layout().groups(); ++g) { staged.events.push_back(take_event()); }
        const std::uint64_t id = next_staged_++;
        const auto entry       = staged_.emplace(id, std::move(staged)).first; // ids are never reused
        entry->second.image    = image.release();
        return id;
    } catch (...) {
        for (cudaEvent_t event : staged.events) { recycle(event); } // none recorded yet
        if (superseded && index_->valid(*superseded)) { index_->unsupersede(*superseded); }
        throw;
    }
}

void PrefixCache::capture_group(std::uint64_t staged, std::uint32_t group, std::uint32_t lane,
                                std::optional<DeviceKVPageHandle> tail_page, cudaStream_t producer) {
    StagedCapture& s = staged_.at(staged);
    if (group != s.copied || tail_page.has_value() != s.tail) {
        throw std::logic_error("Qwen4Exp prefix cache: a staged capture's groups are copied out of order");
    }
    cudaEvent_t start = take_event();
    CUDA_CHECK(cudaEventRecord(start, producer));
    CUDA_CHECK(cudaStreamWaitEvent(transfer_, start, 0));
    recycle(start);
    const std::vector<std::byte*> segments = image_segments(s.image);
    copy_capture_group(group, lane, HostImage{segments, host_.slab_bytes}, s.tail ? slab(s.image.slabs.back()) : nullptr,
                       s.tail ? slab_group(s.image.slabs.back()) : 0U, tail_page);
    CUDA_CHECK(cudaEventRecord(s.events[group], transfer_));
    ++s.copied;
}

std::span<const cudaEvent_t> PrefixCache::staged_events(std::uint64_t staged) const {
    return staged_.at(staged).events;
}

void PrefixCache::commit_staged_capture(std::uint64_t staged, Capture&& capture) {
    const auto found = staged_.find(staged);
    if (found == staged_.end() || found->second.copied != lanes_->layout().groups() ||
        found->second.tail != !capture.tail.empty() || capture.tail.size() >= pc::kBlockTokens) {
        throw std::logic_error("Qwen4Exp prefix cache: a staged capture is committed incomplete or with another tail");
    }
    StagedCapture s = std::move(found->second);
    staged_.erase(found);
    // Past the erase nothing else holds the staged image, its events and the anchor's retention: held here until
    // the capture is listed, and given back only after the catch has waited for the copies.
    pc::HostImageHold image(*index_, std::move(s.image));
    pc::NodeRetention anchor;
    capture.source_superseded = s.superseded.has_value();
    PendingCapture pending{std::move(capture), {}, std::move(s.events)};
    try {
        if (pending.capture.anchor.valid()) { anchor = pc::NodeRetention(*index_, pending.capture.anchor); }
        write_state_image_header(lanes_->layout(), pending.capture.meta, slab(image.get().slabs[0]));
        captures_.push_back(std::move(pending)); // a deque: unchanged when its new block cannot be allocated
    } catch (...) {
        for (cudaEvent_t event : pending.events) {
            (void)cudaEventSynchronize(event);
            recycle(event);
        }
        throw;
    }
    captures_.back().image = image.release(); // poll publishes it
    (void)anchor.release();                   // poll unretains it
    counters_.host_write_bytes += host_.image_bytes + (s.tail ? host_.page.page_stride : 0U);
    ++counters_.host_image_writes;
}

void PrefixCache::abandon_staged_capture(std::uint64_t staged) noexcept {
    const auto found = staged_.find(staged);
    if (found == staged_.end()) { return; }
    // The slabs may still be written: an unrecorded event's synchronize returns at once.
    for (cudaEvent_t event : found->second.events) {
        (void)cudaEventSynchronize(event);
        recycle(event);
    }
    index_->release_host_image(std::move(found->second.image));
    // No deeper state of the lineage came of it: the source it superseded for room serves again.
    if (const auto& superseded = found->second.superseded; superseded && index_->valid(*superseded)) {
        index_->unsupersede(*superseded);
    }
    staged_.erase(found);
}

std::span<const cudaEvent_t> PrefixCache::capture_events(std::uint64_t id) const noexcept {
    for (const PendingCapture& c : captures_) {
        if (c.capture.id == id) { return c.events; }
    }
    return {};
}

void PrefixCache::order_page_reuse(std::span<const DeviceKVPageLease> pages) const {
    for (const PendingCapture& c : captures_) {
        // An endpoint's tail page is its own (tail_lease); a tap's or boundary's is the lane's.
        if (!c.capture.tail_page || c.capture.tail_lease) { continue; }
        const bool reads = std::any_of(pages.begin(), pages.end(), [&](const DeviceKVPageLease& page) {
            return page.valid() && page.handle() == *c.capture.tail_page;
        });
        if (reads) { CUDA_CHECK(cudaStreamWaitEvent(compute_, c.events.back(), 0)); }
    }
}

void PrefixCache::write_blocks(std::span<const pc::NodeRef> nodes, cudaStream_t producer) {
    std::vector<pc::NodeRef> written;
    std::vector<DeviceKVPageHandle> pages;
    std::vector<std::byte*> records;
    std::vector<std::uint32_t> groups;
    // A batch that fails before it is pending gives its slabs and pins back (as HybridPrefixCache's writes do): a
    // node left pinned with its Host fill begun would never become evictable (2026-11-17: an out-of-memory
    // cudaStreamWaitEvent below, once the Engine survived a failed finish, left blocks pinned for good).
    const auto abort_batch = [&]() noexcept {
        for (const pc::NodeRef node : written) {
            index_->abort_host_fill(node);
            index_->unpin_node(node);
        }
        written.clear();
    };
    try {
        // A node pinned and filling is in `written` before anything else can throw: the pushes cannot allocate, and
        // the pin's hold gives it back when begin_host_fill fails or refuses.
        written.reserve(nodes.size());
        pages.reserve(nodes.size());
        records.reserve(nodes.size());
        groups.reserve(nodes.size());
        for (const pc::NodeRef node : nodes) {
            if (!index_->valid(node)) { continue; }
            const pc::NodeView view = index_->node(node);
            if (view.device != pc::CopyState::Resident || view.host != pc::CopyState::Absent) { continue; }
            pc::NodePin pin(*index_, node);
            const auto slab_id = index_->begin_host_fill(node);
            if (!slab_id) { break; } // the Host tier cannot free a slab for anything worth less
            written.push_back(pin.release());
            pages.push_back(page(view.device_id));
            records.push_back(slab(*slab_id));
            groups.push_back(slab_group(*slab_id));
        }
    } catch (...) {
        abort_batch();
        throw;
    }
    if (written.empty()) { return; }
    cudaEvent_t start = nullptr, done = nullptr;
    try {
        start = take_event();
        CUDA_CHECK(cudaEventRecord(start, producer));
        CUDA_CHECK(cudaStreamWaitEvent(transfer_, start, 0));
        recycle(std::exchange(start, nullptr));
        pool_->copy_to_host_records(pages, records, groups, host_.page, transfer_);
        done = take_event();
        CUDA_CHECK(cudaEventRecord(done, transfer_));
    } catch (...) {
        (void)cudaStreamSynchronize(transfer_); // copies already enqueued land before their slabs go back
        if (start != nullptr) { recycle(start); }
        if (done != nullptr) { recycle(done); }
        abort_batch();
        throw;
    }
    counters_.host_block_writes += written.size();
    counters_.host_write_bytes += written.size() * host_.page.page_stride;
    writes_.push_back(PendingWrite{std::move(written), done});
}

void PrefixCache::poll() {
    while (!captures_.empty()) {
        PendingCapture& c = captures_.front();
        const cudaError_t status = cudaEventQuery(c.events.back());
        if (status == cudaErrorNotReady) { break; }
        CUDA_CHECK(status);
        std::optional<std::uint32_t> tail_device;
        if (c.capture.tail_lease) { tail_device = adopt_tail(std::move(*c.capture.tail_lease)); }
        const pc::PublishResult result =
            index_->publish_host_snapshot(c.capture.anchor, c.capture.frontier, c.capture.tail, tail_device,
                                          std::move(c.image), c.capture.kind, c.capture.publish_claim);
        if (result.created) {
            snapshot_meta_[result.snapshot.index] = c.capture.meta;
            ++(c.capture.kind == pc::SnapshotKind::Endpoint ? counters_.endpoints_created : counters_.taps_created);
        }
        if (c.capture.supersedes && index_->valid(*c.capture.supersedes)) {
            if (!result.snapshot.valid()) {
                // Nothing replaced the source: it stays its lineage's resume point.
                if (c.capture.source_superseded) { index_->unsupersede(*c.capture.supersedes); }
            } else if (*c.capture.supersedes != result.snapshot) {
                index_->supersede(*c.capture.supersedes, c.capture.continuation);
            }
        }
        // A lineage that resumes at endpoints does not read this turn's opener tap: the first to give way.
        if (result.created && c.capture.replaces_opener != 0) {
            if (const auto opener = published_opener(c.capture.replaces_opener)) { (void)index_->supersede(*opener); }
        }
        if (c.capture.anchor.valid()) { index_->unretain_node(c.capture.anchor); }
        published_[c.capture.id] = result;
        for (cudaEvent_t event : c.events) { recycle(event); }
        captures_.pop_front();
    }
    while (!writes_.empty()) {
        PendingWrite& w = writes_.front();
        const cudaError_t status = cudaEventQuery(w.done);
        if (status == cudaErrorNotReady) { break; }
        CUDA_CHECK(status);
        // Deepest first (the nodes are a lane's path, root first): each joins the Device LRU as its pin drops, and
        // the LRU's oldest entry goes first, so the deeper blocks are evicted before the ones every continuation
        // of the path maps.
        for (auto node = w.nodes.rbegin(); node != w.nodes.rend(); ++node) {
            index_->complete_host_fill(*node);
            index_->unpin_node(*node);
        }
        recycle(w.done);
        writes_.pop_front();
    }
    while (!landing_.empty()) {
        RestoreBatch& batch       = landing_.front();
        const cudaError_t status = cudaEventQuery(batch.events.back());
        if (status == cudaErrorNotReady) { break; }
        CUDA_CHECK(status);
        finish_restore(batch);
        landing_.pop_front();
    }
    // SSD transfers land on their own threads; the index learns of them here. New write-backs start
    // only from poll_ssd, between rounds.
    if (ssd_) { ssd_->settle(*index_); }
}

bool PrefixCache::await_capture(std::uint64_t id) {
    for (const PendingCapture& c : captures_) {
        if (c.capture.id != id) { continue; }
        // One transfer stream: every capture queued before it has landed by then, so poll publishes up to it.
        CUDA_CHECK(cudaEventSynchronize(c.events.back()));
        poll();
        return true;
    }
    return false;
}

void PrefixCache::drain() {
    testing::count_drain();
    CUDA_CHECK(cudaStreamSynchronize(transfer_));
    CUDA_CHECK(cudaStreamSynchronize(restore_stream_));
    poll();
}

bool PrefixCache::settle_oldest_transfer() {
    // Each wait lands at least one transfer, so a caller that loops makes progress and ends.
    if (!writes_.empty()) {
        CUDA_CHECK(cudaEventSynchronize(writes_.front().done));
    } else if (!captures_.empty()) {
        CUDA_CHECK(cudaEventSynchronize(captures_.front().events.back()));
    } else if (!landing_.empty() && index_->device_transfer_held_blocks() != 0) {
        CUDA_CHECK(cudaEventSynchronize(landing_.front().events.back()));
    } else {
        return false;
    }
    poll();
    return true;
}

bool PrefixCache::poll_ssd() {
    service_ssd();
    return std::exchange(ssd_landed_, false);
}

void PrefixCache::make_lossless_room(std::uint64_t slabs) {
    // Landed captures and writes join the index, write-backs start for what is unwritten (the lowest
    // value first) and land, until the reservation can displace copies the SSD holds or nothing more
    // can be written. Without the wait a just-finished request's endpoint, published but not yet
    // written when the next request's first capture arrived, was evicted unwritten whenever no
    // write-back had started in between (hybrid-prefix-cache-spec §5.7). Each pass either returns or
    // lands a transfer, and a snapshot is written once, so the loop ends.
    while (index_->host_lossless_slabs() < slabs) {
        service_ssd();
        if (index_->host_lossless_slabs() >= slabs || !ssd_->busy()) { return; }
        ssd_->drain(*index_);
    }
}

void PrefixCache::service_ssd() {
    if (!ssd_) { return; }
    // Landed captures and writes first: a just-finished request's snapshots are written back before
    // the next request's prefill can displace them.
    poll();
    ssd_landed_ |= ssd_->poll(*index_);
}

pc::SsdTier::Promotion PrefixCache::promote(pc::SnapshotRef snapshot) {
    if (!ssd_) { return pc::SsdTier::Promotion::Ready; }
    return ssd_->promote(*index_, snapshot);
}

bool PrefixCache::await_extended_capture(std::span<const pc::NodeRef> path, std::span<const TokenId> tokens,
                                         bool endpoints) {
    // The last one in the transfer stream that the prompt extends: every capture before it lands first.
    const PendingCapture* last = nullptr;
    for (const PendingCapture& c : captures_) {
        if ((!endpoints && c.capture.kind == pc::SnapshotKind::Endpoint) || tokens.size() <= c.capture.frontier) {
            continue;
        }
        const std::uint32_t k = c.capture.frontier / pc::kBlockTokens;
        if (k > 0 && (path.size() < k || path[k - 1] != c.capture.anchor)) { continue; }
        if (!std::equal(c.capture.tail.begin(), c.capture.tail.end(),
                        tokens.begin() + static_cast<std::ptrdiff_t>(k) * pc::kBlockTokens)) {
            continue;
        }
        last = &c;
    }
    if (last == nullptr) { return false; }
    CUDA_CHECK(cudaEventSynchronize(last->events.back()));
    poll();
    return true;
}

std::optional<pc::PublishResult> PrefixCache::capture_result(std::uint64_t id) const {
    const auto found = published_.find(id);
    if (found == published_.end()) { return std::nullopt; }
    return found->second;
}

void PrefixCache::forget_capture(std::uint64_t id) noexcept { published_.erase(id); }

// ---- restores --------------------------------------------------------------------------------------

void PrefixCache::open_restore(cudaStream_t producer) {
    if (restore_.open) { throw std::logic_error("Qwen4Exp prefix cache: a restore is already open"); }
    restore_          = RestoreBatch{};
    restore_.open     = true;
    restore_.ticket   = RestoreTicket{next_ticket_++};
    cudaEvent_t start = take_event();
    CUDA_CHECK(cudaEventRecord(start, producer));
    CUDA_CHECK(cudaStreamWaitEvent(restore_stream_, start, 0));
    recycle(start);
    // A test's delay (testing::set_restore_delay): a host function holds the copy stream, and nothing else.
    if (const std::uint32_t delay = testing::restore_delay(); delay != 0) {
        CUDA_CHECK(cudaLaunchHostFunc(
            restore_stream_,
            [](void* microseconds) {
                std::this_thread::sleep_for(std::chrono::microseconds(reinterpret_cast<std::uintptr_t>(microseconds)));
            },
            reinterpret_cast<void*>(static_cast<std::uintptr_t>(delay))));
    }
}

void PrefixCache::restore_block(pc::NodeRef node, DeviceKVPageLease&& destination) {
    if (!restore_.open) { throw std::logic_error("Qwen4Exp prefix cache: no restore is open"); }
    const pc::NodeView view = index_->node(node);
    if (view.host != pc::CopyState::Resident || view.device != pc::CopyState::Absent) {
        throw std::logic_error("Qwen4Exp prefix cache: restored block is not Host-only");
    }
    // The batch's lists have room before anything is taken: a Filling node and its pin are listed (abort_restore
    // undoes what is listed) with pushes that cannot throw.
    pc::reserve_for_push(restore_.pins);
    pc::reserve_for_push(restore_.nodes);
    pc::reserve_for_push(restore_.pages);
    const DeviceKVPageHandle handle = destination.handle();
    const std::uint32_t id          = allocate_block_id(std::move(destination));
    index_->begin_device_fill(node, id);
    restore_.nodes.push_back(node);
    index_->pin_node(node); // until the batch lands
    restore_.pins.push_back(node);
    restore_.pages.push_back(PageCopy{handle, slab(view.host_slab), slab_group(view.host_slab)});
    ++counters_.host_block_restores;
}

void PrefixCache::restore_node_page(pc::NodeRef node, DeviceKVPageHandle destination) {
    if (!restore_.open) { throw std::logic_error("Qwen4Exp prefix cache: no restore is open"); }
    const pc::NodeView view = index_->node(node);
    if (view.host != pc::CopyState::Resident) { throw std::logic_error("Qwen4Exp prefix cache: copied block has no Host copy"); }
    // The copy reads the node's Host slab only: the node stays indexed with it, its Device copy may go.
    pc::reserve_for_push(restore_.retains);
    pc::reserve_for_push(restore_.pages);
    index_->retain_node(node);
    restore_.retains.push_back(node);
    restore_.pages.push_back(PageCopy{destination, slab(view.host_slab), slab_group(view.host_slab)});
}

void PrefixCache::restore_tail_page(pc::SnapshotRef snapshot, DeviceKVPageHandle destination) {
    if (!restore_.open) { throw std::logic_error("Qwen4Exp prefix cache: no restore is open"); }
    const pc::SnapshotView view = index_->snapshot(snapshot);
    if (view.host != pc::CopyState::Resident || view.tail_len == 0) {
        throw std::logic_error("Qwen4Exp prefix cache: restored tail has no Host copy");
    }
    pc::reserve_for_push(restore_.snapshot_pins);
    pc::reserve_for_push(restore_.pages);
    index_->pin_snapshot(snapshot);
    restore_.snapshot_pins.push_back(snapshot);
    const std::uint32_t tail_slab = view.host_slabs.back();
    restore_.pages.push_back(PageCopy{destination, slab(tail_slab), slab_group(tail_slab)});
    ++counters_.host_tail_restores;
}

void PrefixCache::restore_image(pc::SnapshotRef snapshot, std::uint32_t lane) {
    if (!restore_.open || restore_.image) { throw std::logic_error("Qwen4Exp prefix cache: image restore is not open"); }
    const pc::SnapshotView view = index_->snapshot(snapshot);
    if (view.host != pc::CopyState::Resident) { throw std::logic_error("Qwen4Exp prefix cache: restored image is not on the Host"); }
    pc::reserve_for_push(restore_.snapshot_pins);
    index_->pin_snapshot(snapshot);
    restore_.snapshot_pins.push_back(snapshot);
    restore_.image      = snapshot;
    restore_.image_lane = lane;
    ++counters_.host_image_restores;
}

void PrefixCache::copy_pages_from_host(std::span<const PageCopy> copies, std::size_t plane_begin, std::size_t plane_end,
                                       cudaStream_t stream) const {
    if (copies.empty()) { return; }
    std::vector<const std::byte*> records;
    std::vector<std::uint32_t> groups;
    std::vector<DeviceKVPageHandle> pages;
    records.reserve(copies.size());
    groups.reserve(copies.size());
    pages.reserve(copies.size());
    for (const PageCopy& c : copies) {
        records.push_back(c.record);
        groups.push_back(c.group);
        pages.push_back(c.page);
    }
    pool_->copy_from_host_records(records, groups, pages, host_.page, plane_begin, plane_end, stream);
}

RestoreTicket PrefixCache::submit_restore() {
    if (!restore_.open) { throw std::logic_error("Qwen4Exp prefix cache: no restore is open"); }
    const StateImageLayout& layout = lanes_->layout();
    std::vector<const std::byte*> segments;
    if (restore_.image) {
        const pc::SnapshotView view = index_->snapshot(*restore_.image);
        for (std::uint32_t i = 0; i < host_.image_slabs; ++i) { segments.push_back(slab(view.host_slabs[i])); }
    }
    const HostImageConst image{segments, host_.slab_bytes};
    std::uint32_t attention = 0;
    for (std::uint32_t g = 0; g < layout.groups(); ++g) {
        if (g == layout.drafter_group()) {
            if (geometry_.kv_layers > attention) {
                const PlaneRange planes = kv_layer_planes(geometry_, attention);
                copy_pages_from_host(restore_.pages, planes.begin, planes.end, restore_stream_);
            }
        } else if (layout.spec.layer_types[g] == MixerKind::Attention) {
            const PlaneRange planes = kv_layer_planes(geometry_, attention++);
            copy_pages_from_host(restore_.pages, planes.begin, planes.end, restore_stream_);
        }
        if (restore_.image) { lanes_->copy_group_from_host(g, image, restore_.image_lane, restore_stream_); }
        cudaEvent_t event = take_event();
        CUDA_CHECK(cudaEventRecord(event, restore_stream_));
        restore_.events.push_back(event);
    }
    counters_.host_restore_bytes += restore_.pages.size() * host_.page.page_stride + (restore_.image ? host_.image_bytes : 0U);
    restore_.open             = false;
    const RestoreTicket ticket = restore_.ticket;
    landing_.push_back(std::move(restore_));
    restore_ = RestoreBatch{};
    return ticket;
}

std::span<const cudaEvent_t> PrefixCache::restore_events(RestoreTicket ticket) const noexcept {
    for (const RestoreBatch& batch : landing_) {
        if (batch.ticket.id == ticket.id) { return batch.events; }
    }
    return {};
}

void PrefixCache::order_after_restore(RestoreTicket ticket, cudaStream_t consumer) const {
    const auto events = restore_events(ticket);
    if (!events.empty()) { CUDA_CHECK(cudaStreamWaitEvent(consumer, events.back(), 0)); }
}

void PrefixCache::await_restore(RestoreTicket ticket) {
    // One restore stream completes batches in submission order.
    const auto events = restore_events(ticket);
    if (!events.empty()) { CUDA_CHECK(cudaEventSynchronize(events.back())); }
    poll();
}

void PrefixCache::finish_restore(RestoreBatch& batch) noexcept {
    for (const pc::NodeRef node : batch.nodes) { index_->complete_device_fill(node); }
    for (const pc::NodeRef node : batch.pins) { index_->unpin_node(node); }
    for (const pc::NodeRef node : batch.retains) { index_->unretain_node(node); }
    for (const pc::SnapshotRef snapshot : batch.snapshot_pins) { index_->unpin_snapshot(snapshot); }
    for (cudaEvent_t event : batch.events) { recycle(event); }
    batch.events.clear();
}

void PrefixCache::abort_restore() noexcept {
    if (!restore_.open) { return; }
    // Nothing was enqueued yet: the destinations go back and the pins are dropped.
    for (const pc::NodeRef node : restore_.nodes) { index_->abort_device_fill(node); }
    for (const pc::NodeRef node : restore_.pins) { index_->unpin_node(node); }
    for (const pc::NodeRef node : restore_.retains) { index_->unretain_node(node); }
    for (const pc::SnapshotRef snapshot : restore_.snapshot_pins) { index_->unpin_snapshot(snapshot); }
    restore_ = RestoreBatch{};
}

void PrefixCache::fail_restore() noexcept {
    if (!restore_.open) { return; }
    // Part of the batch may be enqueued (a failure inside submit_restore): those copies land, or fail, before their
    // destination pages return to the pool.
    (void)cudaStreamSynchronize(restore_stream_);
    for (cudaEvent_t event : restore_.events) { recycle(event); }
    restore_.events.clear();
    abort_restore();
}

void PrefixCache::clear() noexcept {
    fail_restore(); // a batch a failed activation left open names this index's entries
    try {
        drain();
    } catch (...) {}
    for (RestoreBatch& batch : landing_) {
        for (cudaEvent_t event : batch.events) { recycle(event); }
    }
    landing_.clear();
    while (!staged_.empty()) { abandon_staged_capture(staged_.begin()->first); }
    // What a failed drain left unpublished: the streams were waited for as far as they answer, so the events go
    // back to the spares (the index these captures and writes name is rebuilt below).
    for (PendingCapture& c : captures_) {
        for (cudaEvent_t event : c.events) { recycle(event); }
    }
    for (PendingWrite& w : writes_) { recycle(w.done); }
    captures_.clear();
    writes_.clear();
    published_.clear();
    // The new index holds no SSD slab: transfers are waited for and forgotten.
    if (ssd_) { ssd_->abandon(); }
    // A resize's copies land first; the new index gets the slabs the Host tier holds now, not its startup size.
    store_.abandon();
    pc::PrefixIndexConfig config = index_->config();
    index_.reset();
    blocks_.clear();
    free_blocks_.clear();
    std::fill(node_mtp_next_.begin(), node_mtp_next_.end(), -1);
    std::fill(snapshot_meta_.begin(), snapshot_meta_.end(), StateImageHeader{});
    index_.emplace(config, *this);
    store_.limit_index(*index_);
}

// ---- Host tier resize ----------------------------------------------------------------------------------

std::uint64_t PrefixCache::begin_host_release(std::uint64_t bytes, HostMemoryWorker& worker) {
    poll(); // landed writes and captures publish first: their entries can move rather than hold the cut
    return store_.begin_release(*index_, bytes, worker);
}

pc::HostSlabStore::Progress PrefixCache::step_host_resize(HostMemoryWorker& worker) {
    if (!store_.resizing()) { return {}; }
    poll(); // retires the restores that have landed
    // A restore batch reads Host slabs from the addresses it was staged with: a move commits only once none is open
    // or landing. Captures and writes target reserved or filling slabs, which never move.
    return store_.step(*index_, !restore_.open && landing_.empty(), worker);
}

// ---- PrefixIndexBackend ------------------------------------------------------------------------------

void PrefixCache::release_device_block(std::uint32_t device_id) noexcept {
    if (device_id >= blocks_.size() || !blocks_[device_id]) { std::terminate(); }
    // A published block was its lane's partial page when a tap captured it, and the tap may still
    // be copying it. A wait that fails has failed the context: nothing writes the page after it.
    try {
        order_page_reuse(std::span<const DeviceKVPageLease>(&*blocks_[device_id], 1));
    } catch (...) {}
    blocks_[device_id].reset(); // the lease returns its page to the pool
    free_blocks_.push_back(device_id);
}

void PrefixCache::release_snapshot(pc::SnapshotRef snapshot) noexcept {
    if (snapshot.index < snapshot_meta_.size()) { snapshot_meta_[snapshot.index] = StateImageHeader{}; }
}

void PrefixCache::drop_snapshot_device_image(pc::SnapshotRef) noexcept {
    // Snapshots are Host-born; there are no Device images to drop.
}

} // namespace infernix::models::qwen4_exp::prefix
