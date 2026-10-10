#include "models/qwen4_exp/program/expert_residency.h"

#include "models/qwen4_exp/program/host_expert_tier.h"

#include "core/copy_batch.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <stdexcept>

namespace infernix::models::qwen4_exp {
namespace {

using expert_cache::CacheController;
using expert_cache::ResidencyEntry;
using expert_cache::ResidencyState;

// Frames kept free for one round's promotions while its victims' frames retire. None: the
// Program calls after_round only on a quiescent compute stream, and its on_quiescent frees every
// retired frame and drains the queue before the next round, so slack frames would never hold an
// expert (they cost 512 frames at 9,443). An asynchronous agent that overlaps rounds needs them.
std::uint32_t slack_frames(std::uint32_t) { return 0; }

} // namespace

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

ExpertResidency::ExpertResidency(const TextConfig& config, std::vector<const std::uint8_t*> banks,
                                 std::uint64_t record_stride, std::int32_t max_columns, int device)
    : c_(config), banks_(std::move(banks)), stride_(record_stride) {
    experts_ = config.moe.experts;
    layers_  = config.num_hidden_layers;
    top_k_   = config.moe.top_k;
    if (banks_.size() != layers_) { throw std::invalid_argument("expert residency: one bank per layer"); }
    if (stride_ == 0) { throw std::invalid_argument("expert residency: empty records"); }
    const std::size_t keys = static_cast<std::size_t>(layers_) * experts_;
    limit_                 = max_frames(config);
    if (VmmArena::supported(device)) {
        // Address space for the whole card: growth never needs a new range.
        std::size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        arena_ = std::make_unique<VmmArena>(device, total_bytes, kChunkBytes);
        limit_ = static_cast<std::uint32_t>(std::min<std::uint64_t>(limit_, arena_->reserved_bytes() / stride_));
    }
    controller_ = std::make_unique<CacheController>(static_cast<std::uint32_t>(keys), 0, slack_frames(0), 1, limit_,
                                                    kLfruHalvingPeriod);
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
    if (table_ready_ != nullptr) { (void)cudaEventDestroy(table_ready_); }
    if (copy_stream_ != nullptr) { (void)cudaStreamDestroy(copy_stream_); }
}

const std::uint8_t* ExpertResidency::frame_base() const noexcept {
    return static_cast<const std::uint8_t*>(arena_ ? arena_->base() : frame_memory_.p);
}

std::uint64_t ExpertResidency::pool_bytes() const noexcept {
    return arena_ ? arena_->mapped_bytes() : frame_memory_.bytes;
}

ExpertResidency::Resize ExpertResidency::resize(std::uint32_t frames, cudaStream_t compute, VramBudgetSource& vram) {
    Resize out;
    if (controller_ && controller_->loaned_frames() != 0) {
        out.frames = frames_; // a lease holds frames: the next boundary after its return resizes
        return out;
    }
    frames = std::min(frames, limit_);
    finish_demotions(compute, true); // held frames return before frames move or go
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    publish_landed();
    release_reservations();
    commands_.clear();
    SpillGuard guard(vram);
    if (!arena_) {
        // One allocation of the first size; frames the driver placed in system memory are given back
        // once and the cache starts smaller.
        if (frames_ == 0 && frames > 0) {
            for (int attempt = 0;; ++attempt) {
                guard.begin();
                frame_memory_ = DeviceBuffer(static_cast<std::size_t>(frames) * stride_);
                frame_memory_.fill(0);
                const std::uint64_t spilled = guard.end(frame_memory_.bytes);
                if (spilled == 0) { break; }
                out.spilled = spilled;
                frame_memory_ = DeviceBuffer{};
                const std::uint64_t drop = (spilled + kChunkBytes + stride_ - 1) / stride_;
                frames = attempt == 0 && frames > drop ? frames - static_cast<std::uint32_t>(drop) : 0U;
                if (frames == 0) { break; }
            }
            controller_->resize(frames, commands_);
            frames_ = frames;
        }
        if (table_dirty_) { upload_table(compute); }
        out.frames = frames_;
        return out;
    }
    auto* base  = static_cast<std::uint8_t*>(arena_->base());
    auto* table = static_cast<std::int32_t*>(table_host_.data());
    if (frames < frames_) {
        controller_->resize(frames, commands_);
        report_evictions();
        for (const auto& command : commands_) {
            if (command.kind == CacheController::Command::Kind::kRelocate) {
                CUDA_CHECK(cudaMemcpyAsync(base + static_cast<std::size_t>(command.frame) * stride_,
                                           base + static_cast<std::size_t>(command.source) * stride_, stride_,
                                           cudaMemcpyDeviceToDevice, compute));
                table[command.key] = static_cast<std::int32_t>(command.frame);
                table_dirty_       = true;
            } else if (command.kind == CacheController::Command::Kind::kWriteEntry &&
                       ResidencyEntry::decode(command.word).state != ResidencyState::kReady && table[command.key] >= 0) {
                table[command.key] = -1;
                table_dirty_       = true;
            }
        }
        if (table_dirty_) { upload_table(compute); }
        // The moves and the table must be complete before the top chunks lose their memory.
        CUDA_CHECK(cudaStreamSynchronize(compute));
        const std::size_t keep = static_cast<std::size_t>(frames) * stride_;
        while (arena_->mapped_bytes() >= keep + arena_->chunk_bytes()) { arena_->unmap_chunk(); }
        frames_ = frames;
    } else if (frames > frames_) {
        const std::size_t need = static_cast<std::size_t>(frames) * stride_;
        while (arena_->mapped_bytes() < need) {
            guard.begin();
            if (!arena_->map_chunk()) {
                out.refused = true;
                break;
            }
            // Touch the chunk so its memory is resident before free memory is read again.
            CUDA_CHECK(cudaMemsetAsync(base + arena_->mapped_bytes() - arena_->chunk_bytes(), 0, arena_->chunk_bytes(),
                                       compute));
            CUDA_CHECK(cudaStreamSynchronize(compute));
            if (const std::uint64_t spilled = guard.end(arena_->chunk_bytes()); spilled != 0) {
                arena_->unmap_chunk();
                out.spilled = spilled;
                break;
            }
        }
        const auto backed = static_cast<std::uint32_t>(std::min<std::uint64_t>(frames, arena_->mapped_bytes() / stride_));
        if (backed > frames_) {
            controller_->resize(backed, commands_);
            frames_ = backed;
            if (table_dirty_) { upload_table(compute); }
            issue_loads(compute);
            sync_queue_pins();
        }
    }
    if (table_dirty_) { upload_table(compute); }
    sync_queue_pins(); // a shrink may drop queued experts
    out.frames = frames_;
    return out;
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
    admissible_ = [this](std::uint32_t key) { return tier_->record(key) != nullptr; };
    evictable_  = [this](std::uint32_t key) { return tier_->controller().allow_evict(key); };
    demoting_.assign(keys, 0);
    // The D2H stream: a demotion never waits behind the copy stream's promotion DMA.
    CUDA_CHECK(cudaStreamCreateWithFlags(&d2h_stream_, cudaStreamNonBlocking));
}

void ExpertResidency::demote_or_drop(std::int32_t* table) {
    auto& host = tier_->controller();
    for (const auto& command : commands_) {
        if (command.kind != CacheController::Command::Kind::kWriteEntry ||
            ResidencyEntry::decode(command.word).state != ResidencyState::kAbsent) {
            continue;
        }
        const std::uint32_t key = command.key, frame = command.frame;
        // Only a published frame (its load completed) can be copied out; an unpublished one's load
        // read a host copy that is still pinned (T3).
        const bool published = table[key] == static_cast<std::int32_t>(frame);
        std::uint32_t slot   = 0;
        if (host.vram_evicted(key, published, slot) != expert_cache::VramEviction::kDemote) { continue; }
        controller_->hold(frame); // T4: the frame stays readable in the table until the copy lands
        auto* destination = const_cast<std::uint8_t*>(tier_->slot_bytes(slot));
        CUDA_CHECK(cudaMemcpyAsync(destination, frame_base() + static_cast<std::size_t>(frame) * stride_, stride_,
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
    commands_.clear();
    for (auto it = demotions_.begin(); it != demotions_.end();) {
        const cudaError_t status = cudaEventQuery(it->done);
        if (status == cudaErrorNotReady) {
            ++it;
            continue;
        }
        CUDA_CHECK(status);
        // T7: the record is in RAM. Unless the key was loaded into VRAM again meanwhile, the table
        // stops naming the frame (the slot's pointer is published with this boundary's upload).
        const bool in_vram = controller_->entry(it->key).state == ResidencyState::kReady;
        host.demotion_completed(it->key, it->slot, in_vram);
        if (table[it->key] == static_cast<std::int32_t>(it->frame)) {
            table[it->key] = -1;
            table_dirty_   = true;
        }
        demoting_[it->key] = 0;
        controller_->release_held(it->frame, commands_);
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

void ExpertResidency::report_evictions() {
    if (tier_ == nullptr) { return; }
    for (const auto& command : commands_) {
        if (command.kind == CacheController::Command::Kind::kWriteEntry &&
            ResidencyEntry::decode(command.word).state == ResidencyState::kAbsent) {
            // R10.1 demotes nothing: a key with a host copy keeps it (T3), the rest become SSD-only (T5).
            std::uint32_t slot = 0;
            (void)tier_->controller().vram_evicted(command.key, false, slot);
        }
    }
}

void ExpertResidency::sync_queue_pins() {
    if (tier_ == nullptr || !controller_) { return; }
    auto& host = tier_->controller();
    const auto& queued = controller_->queued_keys();
    for (const std::uint32_t key : queued) { queue_mark_[key] = 1; }
    for (const std::uint32_t key : queue_pinned_keys_) { // dropped from the queue without a load (T9)
        if (queue_pinned_[key] != 0 && queue_mark_[key] == 0) {
            host.unqueued(key);
            queue_pinned_[key] = 0;
        }
    }
    queue_pinned_keys_.clear();
    for (const std::uint32_t key : queued) {
        queue_mark_[key] = 0;
        if (queue_pinned_[key] == 0) { // T8
            host.queued(key);
            queue_pinned_[key] = 1;
        }
        queue_pinned_keys_.push_back(key);
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

void ExpertResidency::before_round(cudaStream_t compute, bool landing, bool walking) {
    publish_landed();
    if (tier_ != nullptr) {
        finish_demotions(compute, false);
        // The last chunk's (or layer walk span's) streamed records have been copied: their pins
        // end here, before this boundary picks victims. A walk's step boundaries (tier_step) and
        // other lanes' rounds between them keep them, since its span's later layers still copy
        // from the records it planned.
        if (!walking) { tier_->release_stream_pins(); }
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
    if (landing && controller_ && frames_ > 0) {
        const std::uint32_t want = std::min<std::uint32_t>(kLandingSlots, controller_->free_frames() / layers_);
        if (want > 0) {
            const std::uint32_t got = controller_->reserve_free(want * layers_, reserved_);
            per_layer               = got / layers_;
            for (std::size_t i = static_cast<std::size_t>(per_layer) * layers_; i < reserved_.size(); ++i) {
                controller_->release(reserved_[i]);
            }
            reserved_.resize(static_cast<std::size_t>(per_layer) * layers_);
        }
    }
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
    landing_per_layer_ = per_layer;
    if (table_dirty_) { upload_table(compute); }
}

void ExpertResidency::release_reservations() {
    for (const std::uint32_t frame : reserved_) { controller_->release(frame); }
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
            const bool published = controller_->complete_load(load.key, load.frame, load.serial);
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
    commands_.clear();
    if (tier_ != nullptr) {
        tier_->end_round();
        uses_.clear();
    }
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
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
            if (table[key] >= 0) { ++stats_.hits; }
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
                    controller_->adopt(key, frame, group_, round_, commands_)) {
                    table[key]   = static_cast<std::int32_t>(frame);
                    table_dirty_ = true;
                    ++adopted;
                } else {
                    controller_->release(frame);
                }
            }
            stats_.landed += adopted;
            budget = budget > adopted ? budget - adopted : 0;
        }
        if (controller_) {
            controller_->on_route(group_, round_, commands_, budget, tier_ != nullptr ? &admissible_ : nullptr,
                                  tier_ != nullptr ? &evictable_ : nullptr);
        }
    }
    if (tier_ != nullptr) {
        // The tier's clock advances with the round's live columns (at most 16 per prefill chunk).
        std::size_t live_columns = static_cast<std::size_t>(columns);
        if (!live.empty()) { live_columns = static_cast<std::size_t>(std::count(live.begin(), live.end(), std::uint8_t{1})); }
        tier_->record_uses(static_cast<double>(std::min<std::size_t>(live_columns, 16)), uses_, 1.0);
    }
    reserved_.clear();
    landing_per_layer_ = 0;
    if (frames_ == 0) { return; }
    if (tier_ != nullptr) { demote_or_drop(table); } // before on_quiescent frees the victims' frames
    controller_->on_quiescent(commands_);

    // Evictions take effect on the compute stream before the next round; promotions copy on the
    // copy stream once that table is in place. A demoted expert stays readable in its held frame.
    for (const auto& command : commands_) {
        if (command.kind == CacheController::Command::Kind::kWriteEntry) {
            const ResidencyEntry entry = ResidencyEntry::decode(command.word);
            if (entry.state != ResidencyState::kReady && table[command.key] >= 0 &&
                (tier_ == nullptr || demoting_[command.key] == 0)) {
                table[command.key] = -1;
                table_dirty_       = true;
            }
        }
    }
    if (table_dirty_) { upload_table(compute); }
    issue_loads(compute);
    sync_queue_pins();
}

ExpertResidency::SavedState ExpertResidency::saved_state() const {
    SavedState out;
    if (!controller_) { return out; }
    const auto& policy = controller_->policy();
    out.counts.resize(policy.num_keys());
    // Halved counts are fractions below ~2 kLfruHalvingPeriod; the file keeps them rounded.
    for (std::uint32_t key = 0; key < policy.num_keys(); ++key) {
        out.counts[key] = static_cast<std::uint32_t>(std::lround(policy.count(key)));
    }
    out.ranked.assign(policy.residents().begin(), policy.residents().end());
    std::stable_sort(out.ranked.begin(), out.ranked.end(),
                     [&](std::uint32_t a, std::uint32_t b) { return policy.score(a) > policy.score(b); });
    return out;
}

std::uint32_t ExpertResidency::warm_start(const SavedState& state, std::uint32_t count_cap, std::uint32_t max_keys,
                                          cudaStream_t compute) {
    if (!controller_ || frames_ == 0) { return 0; }
    if (!state.counts.empty() && state.counts.size() != static_cast<std::size_t>(layers_) * experts_) {
        throw std::invalid_argument("expert residency: the saved state has another key count");
    }
    std::vector<std::uint32_t> counts(state.counts);
    for (auto& count : counts) { count = std::min(count, count_cap); }
    commands_.clear();
    // With a tier only keys the warm pre-fill put in RAM can be loaded.
    std::vector<std::uint32_t> in_ram;
    const std::vector<std::uint32_t>* keys = &state.ranked;
    if (tier_ != nullptr) {
        for (const std::uint32_t key : state.ranked) {
            if (tier_->record(key) != nullptr) { in_ram.push_back(key); }
        }
        keys = &in_ram;
    }
    const std::span<const std::uint32_t> ranked(keys->data(), std::min<std::size_t>(keys->size(), max_keys));
    const std::uint32_t loaded = controller_->seed(ranked, counts, commands_);
    issue_loads(compute);
    stats_.promotions -= loaded; // issue_loads counted them; they are seeds, not promotions
    stats_.seeded += loaded;
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    publish_landed();
    if (table_dirty_) { upload_table(compute); }
    return loaded;
}

std::uint32_t ExpertResidency::lendable() const noexcept {
    if (!controller_) { return 0; }
    const std::uint32_t lent = controller_->loaned_frames(), floor = frames_ / 4;
    return frames_ > lent + floor ? frames_ - lent - floor : 0;
}

ExpertResidency::FrameLease ExpertResidency::lend(std::uint32_t count, cudaStream_t compute,
                                                  std::span<const cudaStream_t> writers) {
    if (count == 0 || count > lendable()) { throw std::logic_error("expert residency: cannot lend that many frames"); }
    finish_demotions(compute, true); // a held frame must not be lent
    // The run's experts move by device copies, which need every promotion landed and published.
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    publish_landed();
    release_reservations();
    const std::vector<std::uint8_t> busy(frames_, 0);
    const auto first = controller_->choose_run(count, busy);
    if (!first) { throw std::logic_error("expert residency: no run of frames to lend"); }
    commands_.clear();
    stats_.lend_evictions += controller_->lend(*first, count, commands_);
    report_evictions();
    sync_queue_pins();
    auto* base  = const_cast<std::uint8_t*>(frame_base());
    auto* table = static_cast<std::int32_t*>(table_host_.data());
    bool moved  = false;
    for (const auto& command : commands_) {
        if (command.kind == CacheController::Command::Kind::kRelocate) {
            // On `compute`, after every round already enqueued that reads the source frame.
            CUDA_CHECK(cudaMemcpyAsync(base + static_cast<std::size_t>(command.frame) * stride_,
                                       base + static_cast<std::size_t>(command.source) * stride_, stride_,
                                       cudaMemcpyDeviceToDevice, compute));
            table[command.key] = static_cast<std::int32_t>(command.frame);
            table_dirty_       = true;
            moved              = true;
            ++stats_.lend_relocations;
        } else if (command.kind == CacheController::Command::Kind::kWriteEntry &&
                   ResidencyEntry::decode(command.word).state != ResidencyState::kReady && table[command.key] >= 0) {
            table[command.key] = -1;
            table_dirty_       = true;
        }
    }
    if (table_dirty_) { upload_table(compute); }
    // The run's new user writes it only after the moves have read it.
    if (moved && !writers.empty()) {
        CUDA_CHECK(cudaEventRecord(table_ready_, compute));
        for (const cudaStream_t writer : writers) { CUDA_CHECK(cudaStreamWaitEvent(writer, table_ready_, 0)); }
    }
    stats_.lent_frames += count;
    return {*first, count, DeviceSpan{base + static_cast<std::size_t>(*first) * stride_, static_cast<std::size_t>(count) * stride_}};
}

void ExpertResidency::give_back(FrameLease& lease, cudaStream_t compute) {
    if (!lease.valid()) { return; }
    commands_.clear();
    controller_->give_back(lease.first, lease.count, commands_);
    stats_.lent_frames -= lease.count;
    lease = {};
    issue_loads(compute); // queued experts load into the run once `compute` reaches here
    sync_queue_pins();
}

void ExpertResidency::issue_loads(cudaStream_t compute) {
    std::vector<Batch::Load> loads;
    for (const auto& command : commands_) {
        if (command.kind == CacheController::Command::Kind::kCopy) {
            loads.push_back({command.key, command.frame, command.serial});
        }
    }
    if (loads.empty()) { return; }
    CUDA_CHECK(cudaEventRecord(table_ready_, compute));
    CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, table_ready_, 0));
    auto* base = const_cast<std::uint8_t*>(frame_base());
    // A prefill promotes thousands of records (~1.2 s of link at x8): batched copies keep the host
    // free, and an event per group publishes the group's experts as soon as it lands.
    CopyBatch copies(copy_stream_);
    for (std::size_t first = 0; first < loads.size(); first += kLoadsPerEvent) {
        Batch batch;
        batch.loads.assign(loads.begin() + static_cast<std::ptrdiff_t>(first),
                           loads.begin() + static_cast<std::ptrdiff_t>(std::min(loads.size(), first + kLoadsPerEvent)));
        for (const auto& load : batch.loads) {
            const std::uint32_t layer  = load.key / experts_;
            const std::uint32_t expert = load.key % experts_;
            const std::uint8_t* source = banks_[layer] + static_cast<std::size_t>(expert) * stride_;
            if (tier_ != nullptr) {
                // T1: the RAM slot is pinned until the copy completes (admission required a host copy,
                // and the Queue pin kept it since).
                source = tier_->record(load.key);
                if (source == nullptr) { throw std::logic_error("expert residency: a promotion lost its host copy"); }
                auto& host = tier_->controller();
                if (copies_in_flight_[load.key]++ == 0) {
                    if (queue_pinned_[load.key] == 0) { host.queued(load.key); }
                    host.promotion_issued(load.key); // Queue -> H2D
                } else if (queue_pinned_[load.key] != 0) {
                    host.unqueued(load.key); // an earlier copy of the key still holds the H2D pin
                }
                queue_pinned_[load.key] = 0;
            }
            copies.add(base + static_cast<std::size_t>(load.frame) * stride_, source, stride_);
        }
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
