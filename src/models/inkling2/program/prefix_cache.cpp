#include "models/inkling2/program/prefix_cache.h"

#include "core/device.h"
#include "models/inkling2/program/test_taps.h"
#include "runtime/prefix_cache/prefix_holds.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::models::inkling2::testing {
namespace {

std::atomic<std::uint32_t> forced_stalls{0};
std::atomic<std::uint64_t> rows_drafted{0};
std::atomic<std::uint64_t> unequal_rounds{0}, empty_rows{0}, full_rows{0};
std::atomic<bool> skewed{false};
std::atomic<std::uint64_t> captures_known{0};
std::mutex ram_mutex;
std::function<std::uint64_t()> ram_reader; // ram_mutex
ContextObserver context_watcher;           // ram_mutex

} // namespace

void set_ram_available(std::function<std::uint64_t()> reader) {
    const std::lock_guard<std::mutex> lock(ram_mutex);
    ram_reader = std::move(reader);
}

std::function<std::uint64_t()> ram_available() {
    const std::lock_guard<std::mutex> lock(ram_mutex);
    return ram_reader;
}

void set_context_observer(ContextObserver observer) {
    const std::lock_guard<std::mutex> lock(ram_mutex);
    context_watcher = std::move(observer);
}

ContextObserver context_observer() {
    const std::lock_guard<std::mutex> lock(ram_mutex);
    return context_watcher;
}

void force_moe_stall(std::uint32_t rounds) noexcept { forced_stalls.store(rounds); }

void count_rows_drafted(std::uint64_t rows) noexcept { rows_drafted.fetch_add(rows, std::memory_order_relaxed); }
std::uint64_t rows_drafted_together() noexcept { return rows_drafted.load(std::memory_order_relaxed); }

void count_dealing(std::span<const std::int32_t> drafts, std::int32_t steps) noexcept {
    if (drafts.empty()) { return; }
    const auto [lo, hi] = std::minmax_element(drafts.begin(), drafts.end());
    if (*lo != *hi) { unequal_rounds.fetch_add(1, std::memory_order_relaxed); }
    for (const std::int32_t d : drafts) {
        if (d == 0) { empty_rows.fetch_add(1, std::memory_order_relaxed); }
        if (d == steps) { full_rows.fetch_add(1, std::memory_order_relaxed); }
    }
}

Dealing dealing() noexcept {
    return {unequal_rounds.load(std::memory_order_relaxed), empty_rows.load(std::memory_order_relaxed),
            full_rows.load(std::memory_order_relaxed)};
}

void set_skewed_dealing(bool on) noexcept { skewed.store(on, std::memory_order_relaxed); }
bool skewed_dealing() noexcept { return skewed.load(std::memory_order_relaxed); }

void record_known_captures(std::uint64_t count) noexcept { captures_known.store(count, std::memory_order_relaxed); }
std::uint64_t known_captures() noexcept { return captures_known.load(std::memory_order_relaxed); }

bool take_forced_stall() noexcept {
    std::uint32_t left = forced_stalls.load();
    while (left != 0 && !forced_stalls.compare_exchange_weak(left, left - 1U)) {}
    return left != 0;
}

} // namespace infernix::models::inkling2::testing

namespace infernix::models::inkling2::prefix {

namespace pc = runtime::prefix_cache;

namespace {

constexpr std::size_t kHostChunkBytes = pc::HostSlabStore::kChunkBytes;
constexpr std::size_t kSlabAlignment  = 4096;

std::size_t align_slab(std::size_t bytes) { return (bytes + kSlabAlignment - 1U) / kSlabAlignment * kSlabAlignment; }

} // namespace

PrefixHostLayout plan_prefix_host_layout(const KVPageGeometry& pages, std::size_t image_bytes, std::uint64_t host_bytes) {
    PrefixHostLayout out;
    out.page        = plan_host_kv_page_layout(pages);
    out.slab_bytes  = align_slab(out.page.page_stride);
    out.image_bytes = image_bytes;
    out.image_slabs = static_cast<std::uint32_t>((image_bytes + out.slab_bytes - 1U) / out.slab_bytes);
    const std::uint64_t slabs = host_bytes / out.slab_bytes;
    if (slabs > pc::kNoId / 4U) { throw std::invalid_argument("Inkling 2 prefix cache: the Host tier is too large"); }
    out.slabs           = static_cast<std::uint32_t>(slabs);
    out.slabs_per_chunk = static_cast<std::uint32_t>(std::max<std::size_t>(1U, kHostChunkBytes / out.slab_bytes));
    return out;
}

PrefixCache::PrefixCache(const PrefixCacheConfig& config, DeviceKVPagePool& pool, const LaneImages& lanes,
                         std::uint64_t host_bytes, cudaStream_t compute)
    : taps_(config.taps), pool_(&pool), lanes_(lanes),
      host_(plan_prefix_host_layout(pool.geometry(), lanes.image_bytes(), std::max(host_bytes, config.host_ceiling_bytes))),
      compute_(compute) {
    // The slabs page-locked at the start; the rest of host_.slabs is growth room (--ram-resize auto).
    const auto locked = static_cast<std::uint32_t>(std::min<std::uint64_t>(host_bytes / std::max<std::size_t>(host_.slab_bytes, 1U),
                                                                           host_.slabs));
    if (locked == 0) {
        // Host-born snapshots are the only snapshot source: without a Host tier there is no cache.
        throw std::invalid_argument("the Inkling 2 prefix cache needs a Host tier (--host-context-mib)");
    }
    if (locked < host_.image_slabs + 2U) {
        const std::uint64_t mib =
            (static_cast<std::uint64_t>(host_.image_slabs + 2U) * host_.slab_bytes + (1U << 20U) - 1U) >> 20U;
        throw std::invalid_argument("--host-context-mib is too small for the Inkling 2 prefix cache: one snapshot needs " +
                                    std::to_string(mib) + " MiB");
    }
    // The SSD tier holds whole Host slabs (hybrid-prefix-cache-spec §5.7).
    const std::uint64_t ssd_slabs = config.ssd_bytes / host_.slab_bytes;
    if (config.ssd_bytes != 0 && ssd_slabs < host_.image_slabs + 2U) {
        throw std::invalid_argument("--ssd-context-gib is too small for the Inkling 2 prefix cache: one snapshot needs " +
                                    std::to_string(((host_.image_slabs + 2ULL) * host_.slab_bytes + (1ULL << 30U) - 1U) >> 30U) +
                                    " GiB");
    }
    if (ssd_slabs > pc::kNoId / 4U) { throw std::invalid_argument("Inkling 2 prefix cache: the SSD tier is too large"); }
    pc::PrefixIndexConfig index = config.index;
    index.max_nodes             = config.kv_pages + host_.slabs + static_cast<std::uint32_t>(ssd_slabs) + 1U;
    index.max_snapshots         = (host_.slabs + static_cast<std::uint32_t>(ssd_slabs)) / host_.image_slabs + 1U;
    index.host_slabs            = host_.slabs;
    index.ssd_slabs             = static_cast<std::uint32_t>(ssd_slabs);
    index.image_slabs           = host_.image_slabs;
    index.device_snapshot_slots = 0;
    index.block_bytes           = host_.page.page_stride;
    index.image_bytes           = host_.image_bytes;
    index.host_blocks           = true;
    store_ = pc::HostSlabStore(host_.slab_bytes, host_.slabs, locked, kHostChunkBytes, "Inkling 2 prefix cache Host tier");
    store_.set_floor_chunks((host_.image_slabs + 2U + host_.slabs_per_chunk - 1U) / host_.slabs_per_chunk);
    try {
        CUDA_CHECK(cudaStreamCreateWithFlags(&transfer_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&restore_stream_, cudaStreamNonBlocking));
        if (index.ssd_slabs != 0) {
            // Its slabs are Host slabs byte for byte (page records and image segments), images packed where that saves
            // slabs.
            ssd_ = std::make_unique<pc::SsdTier>(config.ssd_directory, index.ssd_slabs, host_.slab_bytes,
                                                 [this](std::uint32_t id) { return slab(id); },
                                                 config.ssd_write_bytes_per_hour, /*codec=*/true, config.ssd_kept_file);
        }
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
    for (cudaStream_t stream : {transfer_, restore_stream_}) {
        if (stream != nullptr) {
            (void)cudaStreamSynchronize(stream);
            (void)cudaStreamDestroy(stream);
        }
    }
    for (PendingCapture& c : captures_) {
        if (c.done != nullptr) { (void)cudaEventDestroy(c.done); }
    }
    for (PendingWrite& w : writes_) {
        if (w.done != nullptr) { (void)cudaEventDestroy(w.done); }
    }
    if (restore_.done != nullptr) { (void)cudaEventDestroy(restore_.done); }
    for (RestoreBatch& batch : landing_) {
        if (batch.done != nullptr) { (void)cudaEventDestroy(batch.done); }
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
    if (blocks_.size() >= pc::kNoId - 1U) { throw std::overflow_error("Inkling 2 prefix cache block ids exhausted"); }
    blocks_.emplace_back(std::move(lease));
    return static_cast<std::uint32_t>(blocks_.size() - 1U);
}

DeviceKVPageHandle PrefixCache::page(std::uint32_t device_id) const {
    if (device_id >= blocks_.size() || !blocks_[device_id]) {
        throw std::logic_error("Inkling 2 prefix cache block id is not registered");
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

void PrefixCache::copy_image(std::uint32_t lane, std::span<const std::uint32_t> slabs, bool to_host,
                             cudaStream_t stream) const {
    // The image is its parts back to back over the slabs: each part's bytes go to (or come from) the slabs that hold
    // its range, a run of consecutive slabs at consecutive addresses (one chunk) in one copy.
    std::size_t image_at = 0;
    for (const LaneImagePart& part : lanes_.parts) {
        std::byte* const device = part.device + static_cast<std::size_t>(lane) * part.lane_stride;
        std::size_t done        = 0;
        while (done < part.bytes) {
            const std::size_t at     = image_at + done;
            const std::size_t first  = at / host_.slab_bytes;
            const std::size_t within = at % host_.slab_bytes;
            std::size_t run          = 1;
            // Slabs of one chunk only: each chunk is its own page-locked registration, and a copy across two aborts.
            while (first + run < slabs.size() && slab_group(slabs[first + run]) == slab_group(slabs[first]) &&
                   slab(slabs[first + run]) == slab(slabs[first]) + run * host_.slab_bytes) {
                ++run;
            }
            const std::size_t bytes = std::min(run * host_.slab_bytes - within, part.bytes - done);
            std::byte* const host   = slab(slabs[first]) + within;
            if (to_host) {
                CUDA_CHECK(cudaMemcpyAsync(host, device + done, bytes, cudaMemcpyDeviceToHost, stream));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(device + done, host, bytes, cudaMemcpyHostToDevice, stream));
            }
            done += bytes;
        }
        image_at += part.bytes;
    }
}

// ---- blocks --------------------------------------------------------------------------------------

pc::InsertResult PrefixCache::insert_block(pc::NodeRef parent, std::uint64_t lookup_hash, std::span<const TokenId> tokens,
                                           std::uint64_t extra, DeviceKVPageLease& lease) {
    if (!lease.valid()) { throw std::invalid_argument("Inkling 2 prefix cache: a block needs its page"); }
    const std::uint32_t id = allocate_block_id(std::move(lease));
    pc::InsertResult result;
    try {
        result = index_->insert_block(parent, lookup_hash, tokens, extra, id, /*attach=*/true);
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
        // The replaced Device copy is unmapped and unread (the index replaces only unpinned ones).
        if (result.replaced_device_id != pc::kNoId) { release_device_block(result.replaced_device_id); }
        ++(result.inserted ? counters_.blocks_inserted : counters_.blocks_reattached);
    }
    return result;
}

// ---- captures and write-through ------------------------------------------------------------------

bool PrefixCache::capture(std::uint32_t lane, Capture&& capture, std::span<const pc::NodeRef> path, cudaStream_t producer) {
    const bool tail = !capture.tail.empty();
    if (tail != capture.tail_page.has_value() || capture.tail.size() >= pc::kBlockTokens) {
        throw std::invalid_argument("Inkling 2 prefix cache: a capture's tail does not match its page");
    }
    if (ssd_) {
        // The image, its tail and the path's blocks with no Host copy yet (write_blocks below): an image several times
        // a prompt's blocks leaves the Host tier room for few snapshots, and the blocks would displace the last one.
        std::uint64_t blocks = 0;
        for (const pc::NodeRef node : path) {
            blocks += index_->valid(node) && index_->node(node).host == pc::CopyState::Absent ? 1U : 0U;
        }
        make_lossless_room(host_.image_slabs + (tail ? 1U : 0U) + blocks);
    }
    auto reserved = index_->reserve_host_image(tail, capture.claim);
    if (!reserved) {
        ++counters_.taps_skipped;
        return false;
    }
    pc::HostImageHold image(*index_, std::move(*reserved));
    // The anchor must outlive the copy: the capturing lane may release its path before it lands.
    pc::NodeRetention anchor;
    if (capture.anchor.valid()) { anchor = pc::NodeRetention(*index_, capture.anchor); }
    PendingCapture pending{std::move(capture), {}, nullptr};
    cudaEvent_t start = nullptr;
    try {
        start = take_event();
        CUDA_CHECK(cudaEventRecord(start, producer));
        CUDA_CHECK(cudaStreamWaitEvent(transfer_, start, 0));
        recycle(std::exchange(start, nullptr));
        const std::span<const std::uint32_t> slabs(image.get().slabs);
        copy_image(lane, slabs.first(host_.image_slabs), /*to_host=*/true, transfer_);
        if (tail) {
            const std::uint32_t t                = slabs.back();
            std::byte* const records[1]          = {slab(t)};
            const std::uint32_t groups[1]        = {slab_group(t)};
            const DeviceKVPageHandle pages[1]    = {*pending.capture.tail_page};
            (void)pool_->copy_to_host_records(pages, records, groups, host_.page, transfer_);
        }
        pending.done = take_event();
        CUDA_CHECK(cudaEventRecord(pending.done, transfer_));
        captures_.push_back(std::move(pending));
    } catch (...) {
        (void)cudaStreamSynchronize(transfer_);
        if (start != nullptr) { recycle(start); }
        if (pending.done != nullptr) { recycle(pending.done); }
        throw;
    }
    captures_.back().image = image.release(); // poll publishes it
    (void)anchor.release();                   // poll unretains it
    counters_.host_write_bytes += host_.image_bytes + (tail ? host_.page.page_stride : 0U);
    ++counters_.host_image_writes;
    write_blocks(path, producer);
    return true;
}

std::span<const cudaEvent_t> PrefixCache::capture_events(std::uint64_t id) const noexcept {
    for (const PendingCapture& c : captures_) {
        if (c.capture.id == id) { return {&c.done, 1}; }
    }
    return {};
}

void PrefixCache::order_page_reuse(std::span<const DeviceKVPageLease> pages) const {
    for (const PendingCapture& c : captures_) {
        if (!c.capture.tail_page) { continue; }
        const bool reads = std::any_of(pages.begin(), pages.end(), [&](const DeviceKVPageLease& page) {
            return page.valid() && page.handle() == *c.capture.tail_page;
        });
        if (reads) { CUDA_CHECK(cudaStreamWaitEvent(compute_, c.done, 0)); }
    }
}

void PrefixCache::write_blocks(std::span<const pc::NodeRef> nodes, cudaStream_t producer) {
    std::vector<pc::NodeRef> written;
    std::vector<DeviceKVPageHandle> pages;
    std::vector<std::byte*> records;
    std::vector<std::uint32_t> groups;
    const auto abort_batch = [&]() noexcept {
        for (const pc::NodeRef node : written) {
            index_->abort_host_fill(node);
            index_->unpin_node(node);
        }
        written.clear();
    };
    try {
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
        (void)pool_->copy_to_host_records(pages, records, groups, host_.page, transfer_);
        done = take_event();
        CUDA_CHECK(cudaEventRecord(done, transfer_));
    } catch (...) {
        (void)cudaStreamSynchronize(transfer_);
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
        PendingCapture& c        = captures_.front();
        const cudaError_t status = cudaEventQuery(c.done);
        if (status == cudaErrorNotReady) { break; }
        CUDA_CHECK(status);
        const pc::PublishResult result = index_->publish_host_snapshot(c.capture.anchor, c.capture.frontier,
                                                                       c.capture.tail, std::nullopt, std::move(c.image),
                                                                       c.capture.kind);
        if (result.created) { ++counters_.taps_created; }
        if (c.capture.supersedes && index_->valid(*c.capture.supersedes) &&
            (!result.snapshot.valid() || *c.capture.supersedes != result.snapshot)) {
            (void)index_->supersede(*c.capture.supersedes);
        }
        if (c.capture.anchor.valid()) { index_->unretain_node(c.capture.anchor); }
        if (!c.forgotten) { published_[c.capture.id] = result; }
        recycle(c.done);
        captures_.pop_front();
    }
    while (!writes_.empty()) {
        PendingWrite& w          = writes_.front();
        const cudaError_t status = cudaEventQuery(w.done);
        if (status == cudaErrorNotReady) { break; }
        CUDA_CHECK(status);
        // Deepest first: each joins the Device LRU as its pin drops, so deeper blocks are evicted first.
        for (auto node = w.nodes.rbegin(); node != w.nodes.rend(); ++node) {
            index_->complete_host_fill(*node);
            index_->unpin_node(*node);
        }
        recycle(w.done);
        writes_.pop_front();
    }
    while (!landing_.empty()) {
        RestoreBatch& batch      = landing_.front();
        const cudaError_t status = cudaEventQuery(batch.done);
        if (status == cudaErrorNotReady) { break; }
        CUDA_CHECK(status);
        finish_restore(batch);
        landing_.pop_front();
    }
    // SSD transfers land on their own threads; the index learns of them here (write-backs start from poll_ssd).
    if (ssd_) { ssd_->settle(*index_); }
}

bool PrefixCache::poll_ssd() {
    service_ssd();
    return std::exchange(ssd_landed_, false);
}

void PrefixCache::service_ssd() {
    if (!ssd_) { return; }
    // Landed captures and writes first: a just-finished request's snapshots are written back before the next request's
    // prefill can displace them.
    poll();
    ssd_landed_ |= ssd_->poll(*index_);
}

void PrefixCache::make_lossless_room(std::uint64_t slabs) {
    if (index_->host_lossless_slabs() >= slabs) { return; }
    // The captures and block writes in flight first: the last request's snapshot is published, and its path on the
    // Host, only once they land, and write-back takes nothing before (requests back to back gave it no poll between).
    CUDA_CHECK(cudaStreamSynchronize(transfer_));
    // Each pass returns or lands a transfer, and a snapshot is written once, so the loop ends.
    while (index_->host_lossless_slabs() < slabs) {
        service_ssd();
        if (index_->host_lossless_slabs() >= slabs || !ssd_->busy()) { return; }
        ssd_->drain(*index_);
    }
}

pc::SsdTier::Promotion PrefixCache::promote(pc::SnapshotRef snapshot) {
    if (!ssd_) { return pc::SsdTier::Promotion::Ready; }
    // The captures and block writes in flight land first and write-back starts for what they publish: a promotion that
    // finds no Host room waits for a write in flight (Pending), and with none would start at the root (Unavailable)
    // although the last request's snapshot, published moments ago, could be written back to make the room.
    CUDA_CHECK(cudaStreamSynchronize(transfer_));
    service_ssd();
    return ssd_->promote(*index_, snapshot);
}

void PrefixCache::drain() {
    CUDA_CHECK(cudaStreamSynchronize(transfer_));
    CUDA_CHECK(cudaStreamSynchronize(restore_stream_));
    poll();
}

bool PrefixCache::settle_oldest_transfer() {
    if (!writes_.empty()) {
        CUDA_CHECK(cudaEventSynchronize(writes_.front().done));
    } else if (!captures_.empty()) {
        CUDA_CHECK(cudaEventSynchronize(captures_.front().done));
    } else if (!landing_.empty() && index_->device_transfer_held_blocks() != 0) {
        CUDA_CHECK(cudaEventSynchronize(landing_.front().done));
    } else {
        return false;
    }
    poll();
    return true;
}

bool PrefixCache::await_extended_capture(std::span<const pc::NodeRef> path, std::span<const TokenId> tokens) {
    const PendingCapture* last = nullptr;
    for (const PendingCapture& c : captures_) {
        if (tokens.size() <= c.capture.frontier) { continue; }
        const std::uint32_t k = c.capture.frontier / pc::kBlockTokens;
        if (k > 0 && (path.size() < k || path[k - 1] != c.capture.anchor)) { continue; }
        if (!std::equal(c.capture.tail.begin(), c.capture.tail.end(),
                        tokens.begin() + static_cast<std::ptrdiff_t>(k) * pc::kBlockTokens)) {
            continue;
        }
        last = &c;
    }
    if (last == nullptr) { return false; }
    CUDA_CHECK(cudaEventSynchronize(last->done));
    poll();
    return true;
}

std::optional<pc::PublishResult> PrefixCache::capture_result(std::uint64_t id) const {
    const auto found = published_.find(id);
    if (found == published_.end()) { return std::nullopt; }
    return found->second;
}

void PrefixCache::forget_capture(std::uint64_t id) noexcept {
    published_.erase(id);
    for (PendingCapture& c : captures_) {
        if (c.capture.id == id) { c.forgotten = true; }
    }
}

// ---- restores --------------------------------------------------------------------------------------

void PrefixCache::open_restore(cudaStream_t producer) {
    if (restore_.open) { throw std::logic_error("Inkling 2 prefix cache: a restore is already open"); }
    restore_          = RestoreBatch{};
    restore_.open     = true;
    restore_.ticket   = RestoreTicket{next_ticket_++};
    cudaEvent_t start = take_event();
    CUDA_CHECK(cudaEventRecord(start, producer));
    CUDA_CHECK(cudaStreamWaitEvent(restore_stream_, start, 0));
    recycle(start);
}

void PrefixCache::restore_block(pc::NodeRef node, DeviceKVPageLease&& destination) {
    if (!restore_.open) { throw std::logic_error("Inkling 2 prefix cache: no restore is open"); }
    const pc::NodeView view = index_->node(node);
    if (view.host != pc::CopyState::Resident || view.device != pc::CopyState::Absent) {
        throw std::logic_error("Inkling 2 prefix cache: a restored block is not Host-only");
    }
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

void PrefixCache::restore_tail_page(pc::SnapshotRef snapshot, DeviceKVPageHandle destination) {
    if (!restore_.open) { throw std::logic_error("Inkling 2 prefix cache: no restore is open"); }
    const pc::SnapshotView view = index_->snapshot(snapshot);
    if (view.host != pc::CopyState::Resident || view.tail_len == 0) {
        throw std::logic_error("Inkling 2 prefix cache: a restored tail has no Host copy");
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
    if (!restore_.open || restore_.image) { throw std::logic_error("Inkling 2 prefix cache: an image restore is not open"); }
    const pc::SnapshotView view = index_->snapshot(snapshot);
    if (view.host != pc::CopyState::Resident) { throw std::logic_error("Inkling 2 prefix cache: a restored image is not on the Host"); }
    pc::reserve_for_push(restore_.snapshot_pins);
    index_->pin_snapshot(snapshot);
    restore_.snapshot_pins.push_back(snapshot);
    restore_.image      = snapshot;
    restore_.image_lane = lane;
    ++counters_.host_image_restores;
}

RestoreTicket PrefixCache::submit_restore() {
    if (!restore_.open) { throw std::logic_error("Inkling 2 prefix cache: no restore is open"); }
    if (!restore_.pages.empty()) {
        std::vector<const std::byte*> records;
        std::vector<std::uint32_t> groups;
        std::vector<DeviceKVPageHandle> pages;
        records.reserve(restore_.pages.size());
        groups.reserve(restore_.pages.size());
        pages.reserve(restore_.pages.size());
        for (const PageCopy& c : restore_.pages) {
            records.push_back(c.record);
            groups.push_back(c.group);
            pages.push_back(c.page);
        }
        (void)pool_->copy_from_host_records(records, groups, pages, host_.page, restore_stream_);
    }
    if (restore_.image) {
        const pc::SnapshotView view = index_->snapshot(*restore_.image);
        copy_image(restore_.image_lane, view.host_slabs.first(host_.image_slabs), /*to_host=*/false, restore_stream_);
    }
    restore_.done = take_event();
    CUDA_CHECK(cudaEventRecord(restore_.done, restore_stream_));
    counters_.host_restore_bytes +=
        restore_.pages.size() * host_.page.page_stride + (restore_.image ? host_.image_bytes : 0U);
    restore_.open              = false;
    const RestoreTicket ticket = restore_.ticket;
    landing_.push_back(std::move(restore_));
    restore_ = RestoreBatch{};
    return ticket;
}

void PrefixCache::order_after_restore(RestoreTicket ticket, cudaStream_t consumer) const {
    for (const RestoreBatch& batch : landing_) {
        if (batch.ticket.id == ticket.id) { CUDA_CHECK(cudaStreamWaitEvent(consumer, batch.done, 0)); }
    }
}

void PrefixCache::finish_restore(RestoreBatch& batch) noexcept {
    for (const pc::NodeRef node : batch.nodes) { index_->complete_device_fill(node); }
    for (const pc::NodeRef node : batch.pins) { index_->unpin_node(node); }
    for (const pc::SnapshotRef snapshot : batch.snapshot_pins) { index_->unpin_snapshot(snapshot); }
    recycle(std::exchange(batch.done, nullptr));
}

void PrefixCache::abort_restore() noexcept {
    if (!restore_.open) { return; }
    for (const pc::NodeRef node : restore_.nodes) { index_->abort_device_fill(node); }
    for (const pc::NodeRef node : restore_.pins) { index_->unpin_node(node); }
    for (const pc::SnapshotRef snapshot : restore_.snapshot_pins) { index_->unpin_snapshot(snapshot); }
    recycle(std::exchange(restore_.done, nullptr));
    restore_ = RestoreBatch{};
}

void PrefixCache::fail_restore() noexcept {
    if (!restore_.open) { return; }
    (void)cudaStreamSynchronize(restore_stream_);
    abort_restore();
}

void PrefixCache::clear() noexcept {
    fail_restore();
    try {
        drain();
    } catch (...) {}
    for (RestoreBatch& batch : landing_) { recycle(std::exchange(batch.done, nullptr)); }
    landing_.clear();
    for (PendingCapture& c : captures_) { recycle(std::exchange(c.done, nullptr)); }
    for (PendingWrite& w : writes_) { recycle(std::exchange(w.done, nullptr)); }
    captures_.clear();
    writes_.clear();
    published_.clear();
    if (ssd_) { ssd_->abandon(); } // the new index holds no SSD slab: transfers are waited for and forgotten
    store_.abandon();
    pc::PrefixIndexConfig config = index_->config();
    index_.reset();
    blocks_.clear();
    free_blocks_.clear();
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
    // A restore batch reads Host slabs from the addresses it was staged with: a move commits only once none is open or
    // landing. Captures and writes target reserved or filling slabs, which never move.
    return store_.step(*index_, !restore_.open && landing_.empty(), worker);
}

// ---- persistence ----------------------------------------------------------------------------------------

namespace {

constexpr std::array<char, 8> kMagic = {'N', 'I', 'N', 'F', 'I', 'K', '2', 'C'};

template <class T>
void append(std::vector<std::byte>& out, const T& value) {
    const auto* bytes = reinterpret_cast<const std::byte*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

} // namespace

pc::PersistBinding PrefixCache::persist_binding() {
    // The geometry a file must match byte for byte: the slab and page record, and the image's parts.
    pc::PersistBinding binding;
    binding.magic = kMagic;
    append(binding.geometry, static_cast<std::uint64_t>(host_.slab_bytes));
    append(binding.geometry, static_cast<std::uint64_t>(host_.image_bytes));
    append(binding.geometry, static_cast<std::uint64_t>(host_.page.page_stride));
    append(binding.geometry, static_cast<std::uint32_t>(host_.image_slabs));
    append(binding.geometry, static_cast<std::uint32_t>(lanes_.parts.size()));
    for (const LaneImagePart& part : lanes_.parts) { append(binding.geometry, static_cast<std::uint64_t>(part.bytes)); }
    binding.slab_bytes  = host_.slab_bytes;
    binding.image_slabs = host_.image_slabs;
    binding.host_slabs  = index_->host_slab_limit();
    binding.slab        = [this](std::uint32_t id) { return slab(id); };
    binding.ssd         = ssd_ && ssd_->kept() ? ssd_.get() : nullptr;
    return binding;
}

pc::PersistResult PrefixCache::save(const std::filesystem::path& path, std::string_view fingerprint,
                                    const CancellationView& abandoned) {
    if (!captures_.empty() || !writes_.empty() || restore_.open || !landing_.empty()) {
        throw std::logic_error("Inkling 2 prefix cache save requires idle transfers");
    }
    // A kept SSD tier's copies are named by the file: every transfer lands before (the file then makes them durable).
    if (ssd_ && ssd_->kept()) { ssd_->drain(*index_); }
    return pc::save_prefix_cache(*index_, persist_binding(), path, fingerprint, abandoned);
}

pc::PersistResult PrefixCache::load(const std::filesystem::path& path, std::string_view fingerprint,
                                    const StartupObserver& observer) {
    if (index_->stats().nodes != 0 || index_->stats().snapshots != 0) {
        throw std::logic_error("Inkling 2 prefix cache load requires an empty cache");
    }
    pc::PersistResult out = pc::load_prefix_cache(*index_, persist_binding(), path, fingerprint, observer);
    // Partially restored entries may hold unread slab bytes: drop them all.
    if (!out.ok && (index_->stats().nodes != 0 || index_->stats().snapshots != 0)) { clear(); }
    return out;
}

// ---- PrefixIndexBackend ------------------------------------------------------------------------------

void PrefixCache::release_device_block(std::uint32_t device_id) noexcept {
    if (device_id >= blocks_.size() || !blocks_[device_id]) { std::terminate(); }
    // A published block was its lane's partial page when a tap captured it, and the tap may still be copying it.
    try {
        order_page_reuse(std::span<const DeviceKVPageLease>(&*blocks_[device_id], 1));
    } catch (...) {}
    blocks_[device_id].reset(); // the lease returns its page to the pool
    free_blocks_.push_back(device_id);
}

} // namespace infernix::models::inkling2::prefix
