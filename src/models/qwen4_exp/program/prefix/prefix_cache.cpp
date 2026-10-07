#include "models/qwen4_exp/program/prefix/prefix_cache.h"

#include "core/device.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::models::qwen4_exp::prefix {

namespace pc = runtime::prefix_cache;

namespace {

// Largest single pinned allocation of the slab pool: one very large pinned allocation can fail or
// stall under WDDM where several smaller ones succeed. A slab never crosses chunks.
constexpr std::size_t kHostChunkBytes = std::size_t{4} << 30U;
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
                         const LaneStateImage& lanes, std::uint64_t host_bytes)
    : taps_(config.taps), pool_(&pool), geometry_(pages), lanes_(&lanes),
      host_(plan_prefix_host_layout(pages, lanes.layout(), host_bytes)) {
    if (pool.plane_count() != pages.geometry.planes.size()) {
        throw std::invalid_argument("Qwen4Exp prefix cache: the page geometry does not match the pool");
    }
    if (host_.slabs == 0) {
        // Host-born snapshots are the only snapshot source: without a Host tier there is no cache.
        throw std::invalid_argument("Qwen4Exp prefix cache needs a Host tier (--host-context-mib)");
    }
    if (host_.slabs < host_.image_slabs + 2U) {
        const std::uint64_t mib = (static_cast<std::uint64_t>(host_.image_slabs + 2U) * host_.slab_bytes + (1U << 20U) - 1U) >> 20U;
        throw std::invalid_argument("--host-context-mib is too small for the Qwen4Exp prefix cache: one snapshot needs " +
                                    std::to_string(mib) + " MiB");
    }
    pc::PrefixIndexConfig index = config.index;
    index.max_nodes             = config.kv_pages + host_.slabs + 1U;
    index.max_snapshots         = config.device_snapshot_slots + host_.slabs / host_.image_slabs + 1U;
    index.host_slabs            = host_.slabs;
    index.image_slabs           = host_.image_slabs;
    index.device_snapshot_slots = config.device_snapshot_slots;
    index.block_bytes           = host_.page.page_stride;
    index.image_bytes           = host_.image_bytes;
    index.host_blocks           = true;
    if (config.device_snapshot_slots != 0) {
        throw std::invalid_argument("Qwen4Exp prefix cache: Device snapshot slots are not supported");
    }
    const std::uint64_t total = static_cast<std::uint64_t>(host_.slabs) * host_.slab_bytes;
    std::uint64_t pinned      = 0;
    for (std::uint32_t first = 0; first < host_.slabs; first += host_.slabs_per_chunk) {
        const std::uint32_t slabs = std::min(host_.slabs_per_chunk, host_.slabs - first);
        const std::size_t bytes   = static_cast<std::size_t>(slabs) * host_.slab_bytes;
        try {
            chunks_.emplace_back(bytes);
        } catch (const std::exception& error) {
            throw std::runtime_error("pinning the Qwen4Exp prefix cache Host tier failed after " +
                                     std::to_string(pinned >> 20U) + " of " + std::to_string(total >> 20U) + " MiB (" +
                                     error.what() + "); lower --host-context-mib");
        }
        pinned += bytes;
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&restore_stream_, cudaStreamNonBlocking));
    node_mtp_next_.assign(index.max_nodes, -1);
    snapshot_meta_.assign(index.max_snapshots, StateImageHeader{});
    blocks_.reserve(index.max_nodes);
    index_.emplace(index, *this);
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

std::uint64_t PrefixCache::pinned_bytes() const noexcept {
    std::uint64_t bytes = 0;
    for (const PinnedHostBuffer& chunk : chunks_) { bytes += chunk.size(); }
    return bytes;
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

std::byte* PrefixCache::slab(std::uint32_t id) const noexcept {
    const std::uint32_t chunk = id / host_.slabs_per_chunk;
    return static_cast<std::byte*>(chunks_[chunk].data()) +
           static_cast<std::size_t>(id % host_.slabs_per_chunk) * host_.slab_bytes;
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
    auto image = index_->reserve_host_image(tail, capture.claim);
    if (!image) {
        ++counters_.taps_skipped;
        return false;
    }
    // The anchor must outlive the copy: the capturing lane may release its path before it lands.
    if (capture.anchor.valid()) { index_->pin_node(capture.anchor); }
    cudaEvent_t start = take_event();
    CUDA_CHECK(cudaEventRecord(start, producer));
    CUDA_CHECK(cudaStreamWaitEvent(transfer_, start, 0));
    recycle(start);
    std::vector<std::byte*> segments;
    segments.reserve(host_.image_slabs);
    for (std::uint32_t i = 0; i < host_.image_slabs; ++i) { segments.push_back(slab(image->slabs[i])); }
    write_state_image_header(lanes_->layout(), capture.meta, segments.front());
    const HostImage destination{segments, host_.slab_bytes};
    const StateImageLayout& layout = lanes_->layout();
    std::byte* const tail_record[1]  = {tail ? slab(image->slabs.back()) : nullptr};
    const std::uint32_t tail_group[1] = {tail ? slab_group(image->slabs.back()) : 0U};
    const DeviceKVPageHandle tail_page[1] = {tail ? *capture.tail_page : DeviceKVPageHandle{}};
    const auto copy_tail_planes = [&](std::uint32_t kv_layer) {
        if (!tail) { return; }
        const PlaneRange planes = kv_layer_planes(geometry_, kv_layer);
        pool_->copy_to_host_records(tail_page, tail_record, tail_group, host_.page, planes.begin, planes.end, transfer_);
    };
    PendingCapture pending{std::move(capture), std::move(*image), {}};
    std::uint32_t attention = 0;
    for (std::uint32_t g = 0; g < layout.groups(); ++g) {
        if (g == layout.mtp_group()) {
            if (geometry_.kv_layers > attention) { copy_tail_planes(attention); }
        } else if (layout.spec.layer_types[g] == MixerKind::Attention) {
            copy_tail_planes(attention++);
        }
        lanes_->copy_group_to_host(g, lane, destination, transfer_);
        cudaEvent_t event = take_event();
        CUDA_CHECK(cudaEventRecord(event, transfer_));
        pending.events.push_back(event);
    }
    counters_.host_write_bytes += host_.image_bytes + (tail ? host_.page.page_stride : 0U);
    ++counters_.host_image_writes;
    captures_.push_back(std::move(pending));
    return true;
}

std::span<const cudaEvent_t> PrefixCache::capture_events(std::uint64_t id) const noexcept {
    for (const PendingCapture& c : captures_) {
        if (c.capture.id == id) { return c.events; }
    }
    return {};
}

void PrefixCache::write_blocks(std::span<const pc::NodeRef> nodes, cudaStream_t producer) {
    std::vector<pc::NodeRef> written;
    std::vector<DeviceKVPageHandle> pages;
    std::vector<std::byte*> records;
    std::vector<std::uint32_t> groups;
    for (const pc::NodeRef node : nodes) {
        if (!index_->valid(node)) { continue; }
        const pc::NodeView view = index_->node(node);
        if (view.device != pc::CopyState::Resident || view.host != pc::CopyState::Absent) { continue; }
        index_->pin_node(node);
        const auto slab_id = index_->begin_host_fill(node);
        if (!slab_id) {
            index_->unpin_node(node);
            break; // the Host tier cannot free a slab for anything worth less
        }
        written.push_back(node);
        pages.push_back(page(view.device_id));
        records.push_back(slab(*slab_id));
        groups.push_back(slab_group(*slab_id));
    }
    if (written.empty()) { return; }
    cudaEvent_t start = take_event();
    CUDA_CHECK(cudaEventRecord(start, producer));
    CUDA_CHECK(cudaStreamWaitEvent(transfer_, start, 0));
    recycle(start);
    pool_->copy_to_host_records(pages, records, groups, host_.page, transfer_);
    cudaEvent_t done = take_event();
    CUDA_CHECK(cudaEventRecord(done, transfer_));
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
                                          std::move(c.image), c.capture.kind);
        if (result.created) {
            snapshot_meta_[result.snapshot.index] = c.capture.meta;
            ++(c.capture.kind == pc::SnapshotKind::Endpoint ? counters_.endpoints_created : counters_.taps_created);
        }
        if (c.capture.supersedes && index_->valid(*c.capture.supersedes) &&
            (!result.snapshot.valid() || *c.capture.supersedes != result.snapshot)) {
            index_->supersede(*c.capture.supersedes);
        }
        if (c.capture.anchor.valid()) { index_->unpin_node(c.capture.anchor); }
        published_[c.capture.id] = result;
        for (cudaEvent_t event : c.events) { recycle(event); }
        captures_.pop_front();
    }
    while (!writes_.empty()) {
        PendingWrite& w = writes_.front();
        const cudaError_t status = cudaEventQuery(w.done);
        if (status == cudaErrorNotReady) { break; }
        CUDA_CHECK(status);
        for (const pc::NodeRef node : w.nodes) {
            index_->complete_host_fill(node);
            index_->unpin_node(node);
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
}

void PrefixCache::drain() {
    CUDA_CHECK(cudaStreamSynchronize(transfer_));
    CUDA_CHECK(cudaStreamSynchronize(restore_stream_));
    poll();
}

bool PrefixCache::await_extended_endpoint(std::span<const pc::NodeRef> path, std::span<const TokenId> tokens) {
    for (const PendingCapture& c : captures_) {
        if (c.capture.kind != pc::SnapshotKind::Endpoint || tokens.size() <= c.capture.frontier) { continue; }
        const std::uint32_t k = c.capture.frontier / pc::kBlockTokens;
        if (k > 0 && (path.size() < k || path[k - 1] != c.capture.anchor)) { continue; }
        if (!std::equal(c.capture.tail.begin(), c.capture.tail.end(),
                        tokens.begin() + static_cast<std::ptrdiff_t>(k) * pc::kBlockTokens)) {
            continue;
        }
        CUDA_CHECK(cudaEventSynchronize(c.events.back()));
        poll();
        return true;
    }
    return false;
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
}

void PrefixCache::restore_block(pc::NodeRef node, DeviceKVPageLease&& destination) {
    if (!restore_.open) { throw std::logic_error("Qwen4Exp prefix cache: no restore is open"); }
    const pc::NodeView view = index_->node(node);
    if (view.host != pc::CopyState::Resident || view.device != pc::CopyState::Absent) {
        throw std::logic_error("Qwen4Exp prefix cache: restored block is not Host-only");
    }
    const DeviceKVPageHandle handle = destination.handle();
    const std::uint32_t id          = allocate_block_id(std::move(destination));
    index_->begin_device_fill(node, id);
    index_->pin_node(node); // until the batch lands
    restore_.pins.push_back(node);
    restore_.nodes.push_back(node);
    restore_.pages.push_back(PageCopy{handle, slab(view.host_slab), slab_group(view.host_slab)});
    ++counters_.host_block_restores;
}

void PrefixCache::restore_node_page(pc::NodeRef node, DeviceKVPageHandle destination) {
    if (!restore_.open) { throw std::logic_error("Qwen4Exp prefix cache: no restore is open"); }
    const pc::NodeView view = index_->node(node);
    if (view.host != pc::CopyState::Resident) { throw std::logic_error("Qwen4Exp prefix cache: copied block has no Host copy"); }
    index_->pin_node(node);
    restore_.pins.push_back(node);
    restore_.pages.push_back(PageCopy{destination, slab(view.host_slab), slab_group(view.host_slab)});
}

void PrefixCache::restore_tail_page(pc::SnapshotRef snapshot, DeviceKVPageHandle destination) {
    if (!restore_.open) { throw std::logic_error("Qwen4Exp prefix cache: no restore is open"); }
    const pc::SnapshotView view = index_->snapshot(snapshot);
    if (view.host != pc::CopyState::Resident || view.tail_len == 0) {
        throw std::logic_error("Qwen4Exp prefix cache: restored tail has no Host copy");
    }
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
        const bool mtp_group = g == layout.mtp_group();
        if (mtp_group) {
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
    for (const pc::SnapshotRef snapshot : batch.snapshot_pins) { index_->unpin_snapshot(snapshot); }
    for (cudaEvent_t event : batch.events) { recycle(event); }
    batch.events.clear();
}

void PrefixCache::abort_restore() noexcept {
    if (!restore_.open) { return; }
    // Nothing was enqueued yet: the destinations go back and the pins are dropped.
    for (const pc::NodeRef node : restore_.nodes) { index_->abort_device_fill(node); }
    for (const pc::NodeRef node : restore_.pins) { index_->unpin_node(node); }
    for (const pc::SnapshotRef snapshot : restore_.snapshot_pins) { index_->unpin_snapshot(snapshot); }
    restore_ = RestoreBatch{};
}

void PrefixCache::clear() noexcept {
    try {
        drain();
    } catch (...) {}
    for (RestoreBatch& batch : landing_) {
        for (cudaEvent_t event : batch.events) { recycle(event); }
    }
    landing_.clear();
    captures_.clear();
    writes_.clear();
    published_.clear();
    pc::PrefixIndexConfig config = index_->config();
    index_.reset();
    blocks_.clear();
    free_blocks_.clear();
    std::fill(node_mtp_next_.begin(), node_mtp_next_.end(), -1);
    std::fill(snapshot_meta_.begin(), snapshot_meta_.end(), StateImageHeader{});
    index_.emplace(config, *this);
}

// ---- PrefixIndexBackend ------------------------------------------------------------------------------

void PrefixCache::release_device_block(std::uint32_t device_id) noexcept {
    if (device_id >= blocks_.size() || !blocks_[device_id]) { std::terminate(); }
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
