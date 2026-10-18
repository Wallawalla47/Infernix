#include "models/qwen4_exp/program/expert_residency.h"

#include "models/qwen4_exp/program/host_expert_tier.h"

#include "core/block_copy.h"
#include "core/copy_batch.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

namespace infernix::models::qwen4_exp {
namespace {

using expert_cache::CacheController;
using expert_cache::ResidencyEntry;
using expert_cache::ResidencyState;
using Command = CacheController::Command;

// Frames kept free for one round's promotions while its victims' frames retire. None: the
// Program calls after_round only on a quiescent compute stream, and its on_quiescent frees every
// retired frame and drains the queue before the next round, so slack frames would never hold an
// expert (they cost 512 frames at 9,443). An asynchronous agent that overlaps rounds needs them.
std::uint32_t slack_frames(std::uint32_t) { return 0; }

bool absent_write(const Command& command) {
    return command.kind == Command::Kind::kWriteEntry && ResidencyEntry::decode(command.word).state != ResidencyState::kReady;
}

} // namespace

std::uint8_t* ExpertResidency::Pool::base() const noexcept {
    return static_cast<std::uint8_t*>(arena ? arena->base() : memory.p);
}

std::uint64_t ExpertResidency::table_bytes(const TextConfig& config, std::int32_t max_columns) noexcept {
    const std::uint64_t keys = static_cast<std::uint64_t>(config.num_hidden_layers) * config.moe.experts;
    const std::uint64_t route =
        static_cast<std::uint64_t>(config.moe.top_k) * static_cast<std::uint64_t>(max_columns) * config.num_hidden_layers;
    const std::uint64_t landing = 2ULL * config.num_hidden_layers * kLandingSlots; // table + landed log
    return (keys + route + landing) * sizeof(std::int32_t);
}

std::uint32_t ExpertResidency::max_frames(const TextConfig& config) noexcept {
    return config.num_hidden_layers * config.moe.experts - 1;
}

std::uint64_t ExpertResidency::average_frame_bytes(std::span<const std::uint64_t> strides) noexcept {
    if (strides.empty()) { return 0; }
    std::uint64_t sum = 0;
    for (const std::uint64_t s : strides) { sum += s; }
    return (sum + strides.size() - 1) / strides.size();
}

ExpertResidency::ExpertResidency(const TextConfig& config, std::vector<const std::uint8_t*> banks,
                                 std::uint64_t record_stride, std::int32_t max_columns, int device,
                                 std::vector<std::uint64_t> host_strides)
    : c_(config), banks_(std::move(banks)), host_strides_(std::move(host_strides)), stride_(record_stride) {
    experts_ = config.moe.experts;
    layers_  = config.num_hidden_layers;
    top_k_   = config.moe.top_k;
    if (banks_.size() != layers_) { throw std::invalid_argument("expert residency: one bank per layer"); }
    if (stride_ == 0) { throw std::invalid_argument("expert residency: empty records"); }
    if (host_strides_.empty()) { host_strides_.assign(layers_, stride_); }
    if (host_strides_.size() != layers_ ||
        std::any_of(host_strides_.begin(), host_strides_.end(), [&](std::uint64_t s) { return s == 0 || s > stride_; })) {
        throw std::invalid_argument("expert residency: every bank's records must fit the largest stride");
    }
    frame_bytes_ = average_frame_bytes(host_strides_);
    const std::size_t keys = static_cast<std::size_t>(layers_) * experts_;
    // One pool per record stride, in increasing stride order.
    std::map<std::uint64_t, std::vector<std::uint32_t>> by_stride;
    for (std::uint32_t l = 0; l < layers_; ++l) { by_stride[host_strides_[l]].push_back(l); }
    pool_of_layer_.assign(layers_, 0);
    std::size_t free_bytes = 0, total_bytes = 0;
    const bool vmm = VmmArena::supported(device);
    if (vmm) { CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes)); }
    for (auto& [pitch, layers] : by_stride) {
        Pool pool;
        pool.pitch  = pitch;
        pool.layers = layers;
        for (const std::uint32_t l : layers) { pool_of_layer_[l] = static_cast<std::uint32_t>(pools_.size()); }
        pool.limit = static_cast<std::uint32_t>(layers.size()) * experts_ - 1;
        if (vmm) {
            // Address space for the whole card: growth never needs a new range.
            pool.arena = std::make_unique<VmmArena>(device, total_bytes, kChunkBytes);
            pool.limit = static_cast<std::uint32_t>(std::min<std::uint64_t>(pool.limit, pool.arena->reserved_bytes() / pitch));
        }
        // The pool ticks once per call of its layers: its halving period is their share.
        const auto period = static_cast<std::uint32_t>(
            std::max<std::uint64_t>(1, static_cast<std::uint64_t>(kLfruHalvingPeriod) * layers.size() / layers_));
        pool.controller = std::make_unique<CacheController>(static_cast<std::uint32_t>(keys), 0, slack_frames(0), 1,
                                                            pool.limit, period);
        pools_.push_back(std::move(pool));
    }
    table_device_ = DeviceBuffer(keys * sizeof(std::int32_t));
    table_host_   = PinnedHostBuffer(keys * sizeof(std::int32_t));
    std::fill_n(static_cast<std::int32_t*>(table_host_.data()), keys, -1);
    table_device_.copy_from_host(table_host_.data(), keys * sizeof(std::int32_t));
    route_stride_ = static_cast<std::size_t>(top_k_) * static_cast<std::size_t>(max_columns);
    route_device_ = DeviceBuffer(route_stride_ * layers_ * sizeof(std::int32_t));
    route_host_   = PinnedHostBuffer(route_stride_ * layers_ * sizeof(std::int32_t));
    const std::size_t landing_bytes = static_cast<std::size_t>(layers_) * kLandingSlots * sizeof(std::int32_t);
    landing_host_   = PinnedHostBuffer(landing_bytes);
    landed_host_    = PinnedHostBuffer(landing_bytes);
    std::memset(landing_host_.data(), 0xFF, landing_bytes);
    std::memset(landed_host_.data(), 0xFF, landing_bytes);
    landing_device_ = DeviceBuffer(landing_bytes);
    landed_device_  = DeviceBuffer(landing_bytes);
    landing_device_.copy_from_host(landing_host_.data(), landing_bytes);
    landed_device_.copy_from_host(landed_host_.data(), landing_bytes);
    seen_.assign(experts_, 0);
    group_.reserve(experts_);
    routed_by_layer_.assign(layers_, 0);
    hits_by_layer_.assign(layers_, 0);
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&table_ready_, cudaEventDisableTiming));
    table_dirty_ = false;
}

ExpertResidency::~ExpertResidency() {
    if (copy_stream_ != nullptr) { (void)cudaStreamSynchronize(copy_stream_); }
    if (d2h_stream_ != nullptr) {
        (void)cudaStreamSynchronize(d2h_stream_);
        for (auto& demotion : demotions_) { (void)cudaEventDestroy(demotion.done); }
        (void)cudaStreamDestroy(d2h_stream_);
    }
    for (auto& batch : in_flight_) { (void)cudaEventDestroy(batch.done); }
    for (auto event : spare_events_) { (void)cudaEventDestroy(event); }
    for (auto event : bounced_) {
        if (event != nullptr) { (void)cudaEventDestroy(event); }
    }
    if (table_ready_ != nullptr) { (void)cudaEventDestroy(table_ready_); }
    if (copy_stream_ != nullptr) { (void)cudaStreamDestroy(copy_stream_); }
}

const std::uint8_t* ExpertResidency::frame_base(std::uint32_t layer) const noexcept {
    return pools_[pool_of_layer_[layer]].base();
}

std::uint64_t ExpertResidency::frame_pitch(std::uint32_t layer) const noexcept {
    return pools_[pool_of_layer_[layer]].pitch;
}

std::uint32_t ExpertResidency::frames() const noexcept {
    std::uint32_t n = 0;
    for (const auto& pool : pools_) { n += pool.frames; }
    return n;
}

std::uint32_t ExpertResidency::frame_limit() const noexcept {
    std::uint64_t n = 0;
    for (const auto& pool : pools_) { n += pool.limit; }
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(n, max_frames(c_)));
}

std::uint64_t ExpertResidency::pool_bytes() const noexcept {
    std::uint64_t bytes = 0;
    for (const auto& pool : pools_) { bytes += pool.arena ? pool.arena->mapped_bytes() : pool.memory.bytes; }
    return bytes;
}

std::vector<std::uint32_t> ExpertResidency::split_frames(std::uint32_t frames) const {
    std::vector<std::uint32_t> out(pools_.size());
    if (pools_.size() == 1) {
        out[0] = std::min(frames, pools_[0].limit);
        return out;
    }
    // Every layer the same count: frames / layers average frames (the pools' bytes then sum to
    // about frames * frame_bytes()).
    const std::uint32_t per_layer = frames / layers_;
    for (std::size_t p = 0; p < pools_.size(); ++p) {
        out[p] = std::min(per_layer * static_cast<std::uint32_t>(pools_[p].layers.size()), pools_[p].limit);
    }
    return out;
}

ExpertResidency::Resize ExpertResidency::resize(std::uint32_t frames, cudaStream_t compute, VramBudgetSource& vram) {
    Resize out;
    for (const auto& pool : pools_) {
        if (pool.controller->loaned_frames() != 0) {
            out.frames = this->frames(); // a lease holds frames: the next boundary after its return resizes
            return out;
        }
    }
    frames = std::min(frames, frame_limit());
    finish_demotions(compute, true); // held frames return before frames move or go
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    publish_landed();
    release_reservations();
    load_commands_.clear();
    const auto targets = split_frames(frames);
    // Shrinks first, so their chunks are released before any pool grows.
    for (std::size_t p = 0; p < pools_.size(); ++p) {
        if (targets[p] < pools_[p].frames) { resize_pool(pools_[p], targets[p], compute, vram, out); }
    }
    for (std::size_t p = 0; p < pools_.size(); ++p) {
        if (targets[p] > pools_[p].frames && !out.refused && out.spilled == 0) {
            resize_pool(pools_[p], targets[p], compute, vram, out);
        }
    }
    if (table_dirty_) { upload_table(compute); }
    issue_loads(compute);
    sync_queue_pins(); // a shrink may drop queued experts
    out.frames = this->frames();
    return out;
}

void ExpertResidency::resize_pool(Pool& pool, std::uint32_t frames, cudaStream_t compute, VramBudgetSource& vram,
                                  Resize& out) {
    auto& commands = pool.commands;
    commands.clear();
    SpillGuard guard(vram);
    if (!pool.arena) {
        // One allocation of the first size; frames the driver placed in system memory are given back
        // once and the pool starts smaller.
        if (pool.frames == 0 && frames > 0) {
            for (int attempt = 0;; ++attempt) {
                guard.begin();
                pool.memory = DeviceBuffer(static_cast<std::size_t>(frames) * pool.pitch);
                pool.memory.fill(0);
                const std::uint64_t spilled = guard.end(pool.memory.bytes);
                if (spilled == 0) { break; }
                out.spilled = spilled;
                pool.memory = DeviceBuffer{};
                const std::uint64_t drop = (spilled + kChunkBytes + pool.pitch - 1) / pool.pitch;
                frames = attempt == 0 && frames > drop ? frames - static_cast<std::uint32_t>(drop) : 0U;
                if (frames == 0) { break; }
            }
            pool.controller->resize(frames, commands);
            take_loads(commands);
            pool.frames = frames;
        }
        return;
    }
    auto* base = pool.base();
    if (frames < pool.frames) {
        pool.controller->resize(frames, commands);
        report_evictions(commands);
        apply_moves(pool, compute);
        if (table_dirty_) { upload_table(compute); }
        // The moves and the table must be complete before the top chunks lose their memory.
        CUDA_CHECK(cudaStreamSynchronize(compute));
        const std::size_t keep = static_cast<std::size_t>(frames) * pool.pitch;
        while (pool.arena->mapped_bytes() >= keep + pool.arena->chunk_bytes()) { pool.arena->unmap_chunk(); }
        pool.frames = frames;
    } else if (frames > pool.frames) {
        const std::size_t need = static_cast<std::size_t>(frames) * pool.pitch;
        while (pool.arena->mapped_bytes() < need) {
            guard.begin();
            if (!pool.arena->map_chunk()) {
                out.refused = true;
                break;
            }
            // Touch the chunk so its memory is resident before free memory is read again.
            CUDA_CHECK(cudaMemsetAsync(base + pool.arena->mapped_bytes() - pool.arena->chunk_bytes(), 0,
                                       pool.arena->chunk_bytes(), compute));
            CUDA_CHECK(cudaStreamSynchronize(compute));
            if (const std::uint64_t spilled = guard.end(pool.arena->chunk_bytes()); spilled != 0) {
                pool.arena->unmap_chunk();
                out.spilled = spilled;
                break;
            }
        }
        const auto backed =
            static_cast<std::uint32_t>(std::min<std::uint64_t>(frames, pool.arena->mapped_bytes() / pool.pitch));
        if (backed > pool.frames) {
            pool.controller->resize(backed, commands);
            pool.frames = backed;
            take_loads(commands);
        }
    }
}

void ExpertResidency::attach_tier(HostExpertTier* tier) {
    if (tier_ != nullptr || tier == nullptr || round_ != 0) { throw std::logic_error("expert residency: tier attached twice or late"); }
    tier_ = tier;
    const std::size_t keys = static_cast<std::size_t>(layers_) * experts_;
    records_device_        = DeviceBuffer(keys * sizeof(const std::uint8_t*));
    records_host_          = PinnedHostBuffer(keys * sizeof(const std::uint8_t*));
    auto* records          = static_cast<const std::uint8_t**>(records_host_.data());
    for (std::size_t k = 0; k < keys; ++k) { records[k] = tier_->record(static_cast<std::uint32_t>(k)); }
    records_device_.copy_from_host(records_host_.data(), keys * sizeof(const std::uint8_t*));
    tier_->take_dirty(dirty_);
    dirty_.clear();
    queue_pinned_.assign(keys, 0);
    queue_mark_.assign(keys, 0);
    copies_in_flight_.assign(keys, 0);
    admissible_ = [this](std::uint32_t key) { return tier_->host_record(key) != nullptr; };
    evictable_  = [this](std::uint32_t key) { return tier_->controller().allow_evict(key); };
    demoting_.assign(keys, 0);
    // The D2H stream: a demotion never waits behind the copy stream's promotion DMA.
    CUDA_CHECK(cudaStreamCreateWithFlags(&d2h_stream_, cudaStreamNonBlocking));
    if (tier_->bounce_slots() > 0) {
        for (auto& event : bounced_) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
    }
}

void ExpertResidency::demote_or_drop(Pool& pool, const std::vector<Command>& commands, std::int32_t* table) {
    auto& host = tier_->controller();
    for (const auto& command : commands) {
        if (command.kind != Command::Kind::kWriteEntry ||
            ResidencyEntry::decode(command.word).state != ResidencyState::kAbsent) {
            continue;
        }
        const std::uint32_t key = command.key, frame = command.frame;
        // Only a published frame (its load completed) can be copied out; an unpublished one's load
        // read a host copy that is still pinned (T3).
        const bool published = table[key] == static_cast<std::int32_t>(frame);
        std::uint32_t slot   = 0;
        if (host.vram_evicted(key, published, slot) != expert_cache::VramEviction::kDemote) { continue; }
        pool.controller->hold(frame); // T4: the frame stays readable in the table until the copy lands
        auto* destination = const_cast<std::uint8_t*>(tier_->slot_bytes(slot));
        CUDA_CHECK(cudaMemcpyAsync(destination, pool.base() + static_cast<std::size_t>(frame) * pool.pitch, pool.pitch,
                                   cudaMemcpyDeviceToHost, d2h_stream_));
        cudaEvent_t done = nullptr;
        if (!spare_events_.empty()) {
            done = spare_events_.back();
            spare_events_.pop_back();
        } else {
            CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(done, d2h_stream_));
        demotions_.push_back({key, frame, slot, done});
        demoting_[key] = 1;
        ++stats_.demotions;
    }
}

void ExpertResidency::finish_demotions(cudaStream_t compute, bool wait) {
    if (tier_ == nullptr || demotions_.empty()) { return; }
    if (wait) { CUDA_CHECK(cudaStreamSynchronize(d2h_stream_)); }
    auto* table = static_cast<std::int32_t*>(table_host_.data());
    auto& host  = tier_->controller();
    load_commands_.clear();
    for (auto it = demotions_.begin(); it != demotions_.end();) {
        const cudaError_t status = cudaEventQuery(it->done);
        if (status == cudaErrorNotReady) {
            ++it;
            continue;
        }
        CUDA_CHECK(status);
        Pool& pool = pool_of(it->key);
        // T7: the record is in RAM. Unless the key was loaded into VRAM again meanwhile, the table
        // stops naming the frame (the slot's pointer is published with this boundary's upload).
        const bool in_vram = pool.controller->entry(it->key).state == ResidencyState::kReady;
        host.demotion_completed(it->key, it->slot, in_vram);
        if (table[it->key] == static_cast<std::int32_t>(it->frame)) {
            table[it->key] = -1;
            table_dirty_   = true;
        }
        demoting_[it->key] = 0;
        pool.commands.clear();
        pool.controller->release_held(it->frame, pool.commands);
        take_loads(pool.commands);
        spare_events_.push_back(it->done);
        it = demotions_.erase(it);
    }
    if (table_dirty_) { upload_table(compute); }
    issue_loads(compute);
    sync_queue_pins();
}

const std::uint8_t* const* ExpertResidency::record_table(std::uint32_t layer) const noexcept {
    if (tier_ == nullptr) { return nullptr; }
    return static_cast<const std::uint8_t* const*>(records_device_.p) + static_cast<std::size_t>(layer) * experts_;
}

void ExpertResidency::report_evictions(const std::vector<Command>& commands) {
    if (tier_ == nullptr) { return; }
    for (const auto& command : commands) {
        if (command.kind == Command::Kind::kWriteEntry &&
            ResidencyEntry::decode(command.word).state == ResidencyState::kAbsent) {
            // R10.1 demotes nothing: a key with a host copy keeps it (T3), the rest become SSD-only (T5).
            std::uint32_t slot = 0;
            (void)tier_->controller().vram_evicted(command.key, false, slot);
        }
    }
}

void ExpertResidency::sync_queue_pins() {
    if (tier_ == nullptr) { return; }
    auto& host = tier_->controller();
    for (const auto& pool : pools_) {
        for (const std::uint32_t key : pool.controller->queued_keys()) { queue_mark_[key] = 1; }
    }
    for (const std::uint32_t key : queue_pinned_keys_) { // dropped from the queue without a load (T9)
        if (queue_pinned_[key] != 0 && queue_mark_[key] == 0) {
            host.unqueued(key);
            queue_pinned_[key] = 0;
        }
    }
    queue_pinned_keys_.clear();
    for (const auto& pool : pools_) {
        for (const std::uint32_t key : pool.controller->queued_keys()) {
            queue_mark_[key] = 0;
            if (queue_pinned_[key] == 0) { // T8
                host.queued(key);
                queue_pinned_[key] = 1;
            }
            queue_pinned_keys_.push_back(key);
        }
    }
}

const std::int32_t* ExpertResidency::table(std::uint32_t layer) const noexcept {
    return static_cast<const std::int32_t*>(table_device_.p) + static_cast<std::size_t>(layer) * experts_;
}

void ExpertResidency::upload_table(cudaStream_t compute) {
    // SM-read, not a copy engine: the table must not wait behind this round's promotion copies.
    upload_pinned(table_device_.p, table_host_.data(), table_device_.bytes, compute);
    table_dirty_ = false;
}

void ExpertResidency::tier_step(cudaStream_t compute) {
    if (tier_ == nullptr) { return; }
    tier_->end_round();
    tier_->begin_round(0);
    dirty_.clear();
    tier_->take_dirty(dirty_);
    if (!dirty_.empty()) {
        auto* records = static_cast<const std::uint8_t**>(records_host_.data());
        for (const std::uint32_t key : dirty_) { records[key] = tier_->record(key); }
        upload_pinned(records_device_.p, records_host_.data(), records_device_.bytes, compute);
    }
}

void ExpertResidency::before_round(cudaStream_t compute, bool landing, bool walking, std::uint32_t stream_slots) {
    decode_round_ = landing;
    publish_landed();
    if (tier_ != nullptr) {
        finish_demotions(compute, false);
        // The last chunk's (or layer walk span's) streamed records have been copied: their pins
        // end here, before this boundary picks victims. A walk's step boundaries (tier_step) and
        // other lanes' rounds between them keep them, since its span's later layers still copy
        // from the records it planned. The span's lent stream slots come back too (the stream gave
        // every one back at the span's end), and a new span borrows its own.
        if (!walking) {
            tier_->release_stream_pins();
            tier_->return_stream();
            if (stream_slots > 0) { (void)tier_->lend_stream(stream_slots); }
        }
        // The tier's boundary (landings admitted, RAM victims evicted) and the host pointers it
        // changed, uploaded before this round's kernels: a victim's slot is rewritten only during
        // this round, which no longer reads it. Decode and verification boundaries allow 32
        // demotions (none inside a walk's span); prefill chunk boundaries any number.
        tier_->begin_round(walking   ? 0
                           : landing ? tier_->controller().config().decode_demotions
                                     : std::numeric_limits<std::uint32_t>::max());
        dirty_.clear();
        tier_->take_dirty(dirty_);
        if (!dirty_.empty()) {
            auto* records = static_cast<const std::uint8_t**>(records_host_.data());
            for (const std::uint32_t key : dirty_) { records[key] = tier_->record(key); }
            upload_pinned(records_device_.p, records_host_.data(), records_device_.bytes, compute);
        }
    }
    release_reservations(); // a discarded round's
    std::uint32_t per_layer = 0;
    if (landing && frames() > 0) {
        // The same count for every layer: what the scarcest pool allows.
        std::uint32_t want = kLandingSlots;
        for (const auto& pool : pools_) {
            want = std::min<std::uint32_t>(want, pool.controller->free_frames() / static_cast<std::uint32_t>(pool.layers.size()));
        }
        if (want > 0) {
            std::vector<std::vector<std::uint32_t>> got(pools_.size());
            per_layer = want;
            for (std::size_t p = 0; p < pools_.size(); ++p) {
                const auto n = static_cast<std::uint32_t>(pools_[p].layers.size());
                pools_[p].controller->reserve_free(want * n, got[p]);
                per_layer = std::min<std::uint32_t>(per_layer, static_cast<std::uint32_t>(got[p].size()) / n);
            }
            reserved_.assign(static_cast<std::size_t>(per_layer) * layers_, 0);
            for (std::size_t p = 0; p < pools_.size(); ++p) {
                std::size_t next = 0;
                for (const std::uint32_t layer : pools_[p].layers) {
                    for (std::uint32_t n = 0; n < per_layer; ++n) { reserved_[layer * per_layer + n] = got[p][next++]; }
                }
                for (; next < got[p].size(); ++next) { pools_[p].controller->release(got[p][next]); }
            }
        }
    }
    landing_per_layer_ = per_layer;
    const std::size_t landing_bytes = static_cast<std::size_t>(layers_) * kLandingSlots * sizeof(std::int32_t);
    if (per_layer > 0 || landing_uploaded_) {
        auto* table = static_cast<std::int32_t*>(landing_host_.data());
        std::fill_n(table, static_cast<std::size_t>(layers_) * kLandingSlots, -1);
        for (std::uint32_t layer = 0; layer < layers_; ++layer) {
            for (std::uint32_t n = 0; n < per_layer; ++n) {
                table[layer * kLandingSlots + n] = static_cast<std::int32_t>(reserved_[layer * per_layer + n]);
            }
        }
        upload_pinned(landing_device_.p, landing_host_.data(), landing_bytes, compute);
        landing_uploaded_ = per_layer > 0;
    }
    if (per_layer > 0) { CUDA_CHECK(cudaMemsetAsync(landed_device_.p, 0xFF, landing_bytes, compute)); }
    if (table_dirty_) { upload_table(compute); }
}

void ExpertResidency::release_reservations() {
    for (std::uint32_t layer = 0; layer < layers_ && landing_per_layer_ > 0; ++layer) {
        for (std::uint32_t n = 0; n < landing_per_layer_; ++n) {
            pools_[pool_of_layer_[layer]].controller->release(reserved_[layer * landing_per_layer_ + n]);
        }
    }
    reserved_.clear();
    landing_per_layer_ = 0;
}

void ExpertResidency::publish_landed() {
    auto* table = static_cast<std::int32_t*>(table_host_.data());
    for (auto it = in_flight_.begin(); it != in_flight_.end();) {
        const cudaError_t status = cudaEventQuery(it->done);
        if (status == cudaErrorNotReady) {
            ++it;
            continue;
        }
        CUDA_CHECK(status);
        for (const auto& load : it->loads) {
            // Publish only the load the policy still wants: the same key, frame and serial (a key
            // evicted and re-admitted into this frame meanwhile has a newer copy in flight).
            const bool published = pool_of(load.key).controller->complete_load(load.key, load.frame, load.serial);
            if (published) {
                table[load.key] = static_cast<std::int32_t>(load.frame);
                table_dirty_    = true;
            }
            // T2: the last copy of the key releases its slot's H2D pin (a shadow when published).
            if (tier_ != nullptr && --copies_in_flight_[load.key] == 0) {
                tier_->controller().promotion_completed(load.key, published);
            }
        }
        spare_events_.push_back(it->done);
        it = in_flight_.erase(it);
    }
}

void ExpertResidency::enqueue_route_download(cudaStream_t compute, std::int32_t columns) {
    const std::size_t used = static_cast<std::size_t>(top_k_) * static_cast<std::size_t>(columns);
    if (used > route_stride_) { throw std::logic_error("expert residency: round exceeds the route log"); }
    // Only the round's columns of each layer's log.
    const std::size_t pitch = route_stride_ * sizeof(std::int32_t);
    CUDA_CHECK(cudaMemcpy2DAsync(route_host_.data(), pitch, route_device_.p, pitch, used * sizeof(std::int32_t),
                                 layers_, cudaMemcpyDeviceToHost, compute));
    if (landing_per_layer_ > 0) {
        CUDA_CHECK(cudaMemcpyAsync(landed_host_.data(), landed_device_.p,
                                   static_cast<std::size_t>(layers_) * kLandingSlots * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, compute));
    }
}

void ExpertResidency::after_round(cudaStream_t compute, std::int32_t columns, std::size_t per_layer_budget,
                                  std::span<const std::uint8_t> live) {
    const auto* routes = static_cast<const std::int32_t*>(route_host_.data());
    auto* table        = static_cast<std::int32_t*>(table_host_.data());
    const std::size_t used = static_cast<std::size_t>(top_k_) * static_cast<std::size_t>(columns);
    ++round_;
    for (auto& pool : pools_) { pool.commands.clear(); }
    if (tier_ != nullptr) {
        tier_->end_round();
        uses_.clear();
    }
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        Pool& pool = pools_[pool_of_layer_[layer]];
        group_.clear();
        const std::int32_t* ids = routes + static_cast<std::size_t>(layer) * route_stride_;
        for (std::size_t i = 0; i < used; ++i) {
            if (!live.empty() && live[i / top_k_] == 0) { continue; }
            const auto expert = static_cast<std::uint32_t>(ids[i]);
            if (expert >= experts_ || seen_[expert]) { continue; }
            seen_[expert] = 1;
            group_.push_back(layer * experts_ + expert);
        }
        if (tier_ != nullptr) { uses_.insert(uses_.end(), group_.begin(), group_.end()); }
        for (const auto key : group_) {
            seen_[key - layer * experts_] = 0;
            ++stats_.routed;
            ++routed_by_layer_[layer];
            if (table[key] >= 0) {
                ++stats_.hits;
                ++hits_by_layer_[layer];
            }
        }
        if (decode_round_) {
            stats_.decode_routed += group_.size();
            for (const auto key : group_) { stats_.decode_hits += table[key] >= 0 ? 1U : 0U; }
        }
        // The layer's landed experts become resident in their frames; the rest of its reserved
        // frames return to the pool.
        std::size_t budget = per_layer_budget;
        if (landing_per_layer_ > 0) {
            const auto* landed = static_cast<const std::int32_t*>(landed_host_.data()) + layer * kLandingSlots;
            std::size_t adopted = 0;
            for (std::uint32_t n = 0; n < landing_per_layer_; ++n) {
                const std::uint32_t frame = reserved_[layer * landing_per_layer_ + n];
                const std::int32_t expert = landed[n];
                const std::uint32_t key   = layer * experts_ + static_cast<std::uint32_t>(expert);
                if (expert >= 0 && static_cast<std::uint32_t>(expert) < experts_ &&
                    pool.controller->adopt(key, frame, group_, round_, pool.commands)) {
                    table[key]   = static_cast<std::int32_t>(frame);
                    table_dirty_ = true;
                    ++adopted;
                } else {
                    pool.controller->release(frame);
                }
            }
            stats_.landed += adopted;
            budget = budget > adopted ? budget - adopted : 0;
        }
        pool.controller->on_route(group_, round_, pool.commands, budget, tier_ != nullptr ? &admissible_ : nullptr,
                                  tier_ != nullptr ? &evictable_ : nullptr);
    }
    if (tier_ != nullptr) {
        // The tier's clock advances with the round's live columns (at most 16 per prefill chunk).
        std::size_t live_columns = static_cast<std::size_t>(columns);
        if (!live.empty()) { live_columns = static_cast<std::size_t>(std::count(live.begin(), live.end(), std::uint8_t{1})); }
        tier_->record_uses(static_cast<double>(std::min<std::size_t>(live_columns, 16)), uses_, 1.0);
    }
    reserved_.clear();
    landing_per_layer_ = 0;
    if (frames() == 0) { return; }
    load_commands_.clear();
    for (auto& pool : pools_) {
        if (pool.frames == 0) { continue; }
        if (tier_ != nullptr) { demote_or_drop(pool, pool.commands, table); } // before on_quiescent frees the victims' frames
        pool.controller->on_quiescent(pool.commands);
        // Evictions take effect on the compute stream before the next round; promotions copy on
        // the copy stream once that table is in place. A demoted expert stays readable in its held frame.
        for (const auto& command : pool.commands) {
            if (absent_write(command) && table[command.key] >= 0 && (tier_ == nullptr || demoting_[command.key] == 0)) {
                table[command.key] = -1;
                table_dirty_       = true;
            }
        }
        take_loads(pool.commands);
    }
    if (table_dirty_) { upload_table(compute); }
    issue_loads(compute);
    sync_queue_pins();
}

void ExpertResidency::seed_prompt(std::span<const std::uint32_t> uses, double scale) {
    if (uses.size() != static_cast<std::size_t>(layers_) * experts_) {
        throw std::invalid_argument("expert residency: prompt uses have another key count");
    }
    if (!(scale > 0.0)) { return; }
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        group_.clear();
        amounts_.clear();
        for (std::uint32_t expert = 0; expert < experts_; ++expert) {
            const std::uint32_t key = layer * experts_ + expert;
            if (uses[key] == 0) { continue; }
            group_.push_back(key);
            amounts_.push_back(scale * static_cast<double>(uses[key]));
        }
        if (!group_.empty()) { pools_[pool_of_layer_[layer]].controller->credit(group_, amounts_); }
    }
}

ExpertResidency::SavedState ExpertResidency::saved_state() const {
    SavedState out;
    const std::size_t keys = static_cast<std::size_t>(layers_) * experts_;
    out.counts.resize(keys);
    // Halved counts are fractions below ~2 kLfruHalvingPeriod; the file keeps them rounded.
    for (std::uint32_t key = 0; key < keys; ++key) {
        out.counts[key] = static_cast<std::uint32_t>(std::lround(pool_of(key).controller->policy().count(key)));
    }
    // Every pool's residents by score, a pool's age (its own ticks) scaled to layer calls of every
    // layer so scores compare across pools (warm_start loads each pool's keys in this order).
    std::vector<std::pair<double, std::uint32_t>> ranked;
    for (const auto& pool : pools_) {
        const auto& policy = pool.controller->policy();
        const double scale = static_cast<double>(layers_) / static_cast<double>(pool.layers.size());
        for (const std::uint32_t key : policy.residents()) {
            const double count = policy.count(key), score = policy.score(key);
            const double age   = score > 0.0 ? count / score - 1.0 : 0.0;
            ranked.emplace_back(score > 0.0 ? count / (age * scale + 1.0) : 0.0, key);
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    out.ranked.reserve(ranked.size());
    for (const auto& [score, key] : ranked) { out.ranked.push_back(key); }
    return out;
}

std::uint32_t ExpertResidency::warm_start(const SavedState& state, std::uint32_t count_cap, std::uint32_t max_keys,
                                          cudaStream_t compute) {
    const std::uint32_t total = frames();
    if (total == 0) { return 0; }
    if (!state.counts.empty() && state.counts.size() != static_cast<std::size_t>(layers_) * experts_) {
        throw std::invalid_argument("expert residency: the saved state has another key count");
    }
    std::vector<std::uint32_t> counts(state.counts);
    for (auto& count : counts) { count = std::min(count, count_cap); }
    // Each pool loads its own keys, best first, up to its share of max_keys (with a tier: only keys
    // the warm pre-fill put in RAM).
    std::vector<std::vector<std::uint32_t>> keys(pools_.size());
    for (const std::uint32_t key : state.ranked) {
        if (key >= static_cast<std::uint32_t>(layers_) * experts_) { continue; }
        if (tier_ != nullptr && tier_->record(key) == nullptr) { continue; }
        keys[pool_of_layer_[key / experts_]].push_back(key);
    }
    load_commands_.clear();
    std::uint32_t loaded = 0;
    for (std::size_t p = 0; p < pools_.size(); ++p) {
        Pool& pool = pools_[p];
        if (pool.frames == 0) { continue; }
        const auto share = static_cast<std::uint32_t>(static_cast<std::uint64_t>(max_keys) * pool.frames / total);
        const std::span<const std::uint32_t> ranked(keys[p].data(), std::min<std::size_t>(keys[p].size(), share));
        pool.commands.clear();
        loaded += pool.controller->seed(ranked, counts, pool.commands);
        take_loads(pool.commands);
    }
    issue_loads(compute);
    stats_.promotions -= loaded; // issue_loads counted them; they are seeds, not promotions
    stats_.seeded += loaded;
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    publish_landed();
    if (table_dirty_) { upload_table(compute); }
    return loaded;
}

std::uint32_t ExpertResidency::pool_lendable(const Pool& pool) const noexcept {
    // A quarter of the frames is never lent, and a lease is one contiguous run: after a lease in the
    // middle of the pool the next one fits only the longest piece left (FP8 SSD tier, 16,384-token
    // prefill: the wide arena, walk area and ring leases once failed "no run of frames to lend").
    const std::uint32_t lent = pool.controller->loaned_frames(), floor = pool.frames / 4;
    const std::uint32_t total = pool.frames > lent + floor ? pool.frames - lent - floor : 0;
    return std::min(total, pool.controller->longest_unloaned_run());
}

std::size_t ExpertResidency::lend_pool() const noexcept {
    std::size_t best = 0;
    std::uint64_t best_bytes = 0;
    for (std::size_t p = 0; p < pools_.size(); ++p) {
        const std::uint64_t bytes = static_cast<std::uint64_t>(pool_lendable(pools_[p])) * pools_[p].pitch;
        if (bytes > best_bytes) {
            best       = p;
            best_bytes = bytes;
        }
    }
    return best;
}

std::uint64_t ExpertResidency::lendable_bytes() const noexcept {
    const Pool& pool = pools_[lend_pool()];
    return static_cast<std::uint64_t>(pool_lendable(pool)) * pool.pitch;
}

std::uint64_t ExpertResidency::lend_pitch() const noexcept { return pools_[lend_pool()].pitch; }

std::uint64_t ExpertResidency::lend_capacity_bytes() const noexcept {
    const Pool& pool = pools_[lend_pool()];
    return static_cast<std::uint64_t>(pool.frames - pool.frames / 4) * pool.pitch;
}

std::uint32_t ExpertResidency::lend_layers() const noexcept {
    return static_cast<std::uint32_t>(pools_[lend_pool()].layers.size());
}

std::uint32_t ExpertResidency::apply_moves(Pool& pool, cudaStream_t compute) {
    auto* table = static_cast<std::int32_t*>(table_host_.data());
    move_to_.clear();
    move_from_.clear();
    for (const auto& command : pool.commands) {
        if (command.kind == Command::Kind::kRelocate) {
            move_to_.push_back(command.frame);
            move_from_.push_back(command.source);
            table[command.key] = static_cast<std::int32_t>(command.frame);
            table_dirty_       = true;
        } else if (absent_write(command) && table[command.key] >= 0) {
            table[command.key] = -1;
            table_dirty_       = true;
        }
    }
    // On `compute`, after every round already enqueued that reads a source frame. One launch: a
    // cudaMemcpyAsync per frame held the host ~10 ms for the ~1,300 moves of an 8K prompt's lease.
    const auto moves = static_cast<std::uint32_t>(move_to_.size());
    copy_device_blocks(pool.base(), pool.pitch, move_to_.data(), move_from_.data(), moves, compute);
    return moves;
}

ExpertResidency::FrameLease ExpertResidency::lend(std::uint64_t bytes, cudaStream_t compute,
                                                  std::span<const cudaStream_t> writers) {
    const std::size_t p = lend_pool();
    Pool& pool          = pools_[p];
    const auto count    = static_cast<std::uint32_t>((bytes + pool.pitch - 1) / pool.pitch);
    if (count == 0 || count > pool_lendable(pool)) { throw std::logic_error("expert residency: cannot lend that many bytes"); }
    finish_demotions(compute, true); // a held frame must not be lent
    // The run's experts move by device copies, which need every promotion landed and published.
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    publish_landed();
    release_reservations();
    const std::vector<std::uint8_t> busy(pool.frames, 0);
    const auto first = pool.controller->choose_run(count, busy);
    if (!first) { throw std::logic_error("expert residency: no run of frames to lend"); }
    pool.commands.clear();
    stats_.lend_evictions += pool.controller->lend(*first, count, pool.commands);
    report_evictions(pool.commands);
    sync_queue_pins();
    auto* base                = pool.base();
    const std::uint32_t moved = apply_moves(pool, compute);
    stats_.lend_relocations += moved;
    if (table_dirty_) { upload_table(compute); }
    // The run's new user writes it only after the moves have read it, and after every round already
    // on `compute` that reads its evicted experts (a run of victims and free frames moves nothing,
    // so the fence does not depend on the moves).
    if (!writers.empty()) {
        CUDA_CHECK(cudaEventRecord(table_ready_, compute));
        for (const cudaStream_t writer : writers) { CUDA_CHECK(cudaStreamWaitEvent(writer, table_ready_, 0)); }
    }
    stats_.lent_frames += count;
    return {static_cast<std::uint32_t>(p), *first, count,
            DeviceSpan{base + static_cast<std::size_t>(*first) * pool.pitch, static_cast<std::size_t>(count) * pool.pitch}};
}

ExpertResidency::FrameLease ExpertResidency::split(FrameLease& lease, std::uint64_t head_bytes) const {
    if (!lease.valid()) { return {}; }
    const std::uint64_t pitch = pools_[lease.pool].pitch;
    const auto head           = static_cast<std::uint32_t>((head_bytes + pitch - 1) / pitch);
    if (head >= lease.count) { return {}; }
    FrameLease tail{lease.pool, lease.first + head, lease.count - head,
                    DeviceSpan{static_cast<std::byte*>(lease.memory.data) + static_cast<std::size_t>(head) * pitch,
                               static_cast<std::size_t>(lease.count - head) * pitch}};
    lease.count        = head;
    lease.memory.bytes = static_cast<std::size_t>(head) * pitch;
    return tail;
}

void ExpertResidency::give_back(FrameLease& lease, cudaStream_t compute) {
    if (!lease.valid()) { return; }
    Pool& pool = pools_[lease.pool];
    pool.commands.clear();
    // Decode rounds reserve landing frames for every layer from the scarcest pool (before_round). With
    // every expert in RAM, when the returned run lets them, the frames stay free and decode's misses
    // land in them (NVFP4 Dense8, one pool: MTP 4 at 4K + 512 154.8 tok/s against 137.5 with the
    // refill, plain equal to the refill-free route, 2026-10-17); when another pool has none to spare
    // (a GGUF import's record sizes), landing reserves nothing and the loans' victims reload instead
    // (IQ3_S: MTP 185.4 against 162.2, plain 129.0 against 115.7). The SSD tier reloads only victims
    // with a RAM copy (`admissible_`) and still lands (FP8: 44.0-44.2 tok/s against 39.6-43.5 without).
    std::uint32_t landing = kLandingSlots;
    for (std::size_t p = 0; p < pools_.size(); ++p) {
        const std::uint32_t free = pools_[p].controller->free_frames() + (p == lease.pool ? lease.count : 0U);
        landing = std::min<std::uint32_t>(landing, free / static_cast<std::uint32_t>(pools_[p].layers.size()));
    }
    pool.controller->give_back(lease.first, lease.count, pool.commands, landing == 0 || tier_ != nullptr,
                               tier_ != nullptr ? &admissible_ : nullptr);
    stats_.lent_frames -= lease.count;
    lease = {};
    load_commands_.clear();
    take_loads(pool.commands);
    issue_loads(compute); // queued experts load into the run once `compute` reaches here
    sync_queue_pins();
}

void ExpertResidency::take_loads(std::vector<Command>& commands) {
    for (const auto& command : commands) {
        if (command.kind == Command::Kind::kCopy) { load_commands_.push_back(command); }
    }
}

void ExpertResidency::issue_loads(cudaStream_t compute) {
    std::vector<Batch::Load> loads;
    for (const auto& command : load_commands_) { loads.push_back({command.key, command.frame, command.serial}); }
    load_commands_.clear();
    if (loads.empty()) { return; }
    CUDA_CHECK(cudaEventRecord(table_ready_, compute));
    CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, table_ready_, 0));
    // A prefill promotes thousands of records (~1.2 s of link at x8): batched copies keep the host
    // free, and an event per group publishes the group's experts as soon as it lands.
    CopyBatch copies(copy_stream_);
    // Unpinned records (the SSD tier past the page-locking limit): copied from pageable memory the
    // driver would stage synchronously at ~13 GB/s while holding this thread (RTX 5090, x8), so
    // they go through the tier's pinned bounce slots instead: the copy threads fill one half while
    // the link drains the other, and the DMA runs at the link rate (~50 GB/s at 5.0 x16).
    struct Bounced {
        std::uint8_t* frame;
        const std::uint8_t* record; // unpinned
        std::uint64_t bytes;        // the layer bank's record stride
    };
    std::vector<Bounced> bounced;
    std::vector<std::pair<std::uint8_t*, const std::uint8_t*>> fill;
    const auto drain_bounced = [&] {
        if (bounced.empty()) { return; }
        copies.flush(); // stream order with the pending pinned copies
        const std::uint32_t half = tier_->bounce_slots() / 2;
        for (std::size_t at = 0; at < bounced.size(); at += half) {
            const std::size_t n = std::min<std::size_t>(half, bounced.size() - at);
            const std::uint32_t h = bounce_half_;
            if (bounce_pending_[h]) { CUDA_CHECK(cudaEventSynchronize(bounced_[h])); } // its last DMA read it
            fill.clear();
            for (std::size_t i = 0; i < n; ++i) { fill.emplace_back(tier_->bounce_slot(h * half + static_cast<std::uint32_t>(i)), bounced[at + i].record); }
            tier_->copy_records(fill);
            CopyBatch from_bounce(copy_stream_);
            for (std::size_t i = 0; i < n; ++i) { from_bounce.add(bounced[at + i].frame, fill[i].first, bounced[at + i].bytes); }
            from_bounce.flush();
            CUDA_CHECK(cudaEventRecord(bounced_[h], copy_stream_));
            bounce_pending_[h] = true;
            bounce_half_       = h ^ 1U;
        }
        stats_.unpinned_promotions += bounced.size();
        bounced.clear();
    };
    for (std::size_t first = 0; first < loads.size(); first += kLoadsPerEvent) {
        Batch batch;
        batch.loads.assign(loads.begin() + static_cast<std::ptrdiff_t>(first),
                           loads.begin() + static_cast<std::ptrdiff_t>(std::min(loads.size(), first + kLoadsPerEvent)));
        for (const auto& load : batch.loads) {
            const std::uint32_t layer  = load.key / experts_;
            const std::uint32_t expert = load.key % experts_;
            const Pool& pool           = pools_[pool_of_layer_[layer]];
            const std::uint8_t* source = banks_[layer] + static_cast<std::size_t>(expert) * host_strides_[layer];
            if (tier_ != nullptr) {
                // T1: the RAM slot is pinned until the copy completes (admission required a host copy,
                // and the Queue pin kept it since). An unpinned record never moves.
                source = tier_->record(load.key);
                if (source == nullptr) {
                    source = tier_->unpinned_record(load.key);
                    if (source == nullptr) { throw std::logic_error("expert residency: a promotion lost its host copy"); }
                }
                auto& host = tier_->controller();
                if (copies_in_flight_[load.key]++ == 0) {
                    if (queue_pinned_[load.key] == 0) { host.queued(load.key); }
                    host.promotion_issued(load.key); // Queue -> H2D
                } else if (queue_pinned_[load.key] != 0) {
                    host.unqueued(load.key); // an earlier copy of the key still holds the H2D pin
                }
                queue_pinned_[load.key] = 0;
            }
            // The bank's own record: its stride, the pool's pitch.
            auto* destination = pool.base() + static_cast<std::size_t>(load.frame) * pool.pitch;
            if (tier_ != nullptr && source == tier_->unpinned_record(load.key)) {
                bounced.push_back({destination, source, host_strides_[layer]});
                continue;
            }
            // A frame written by a pending bounced copy and then by this one keeps stream order.
            for (const auto& b : bounced) {
                if (b.frame == destination) {
                    drain_bounced();
                    break;
                }
            }
            copies.add(destination, source, host_strides_[layer]);
        }
        drain_bounced();
        copies.flush();
        stats_.promotions += batch.loads.size();
        if (spare_events_.empty()) {
            cudaEvent_t event = nullptr;
            CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            spare_events_.push_back(event);
        }
        batch.done = spare_events_.back();
        spare_events_.pop_back();
        CUDA_CHECK(cudaEventRecord(batch.done, copy_stream_));
        in_flight_.push_back(std::move(batch));
    }
}

} // namespace infernix::models::qwen4_exp
