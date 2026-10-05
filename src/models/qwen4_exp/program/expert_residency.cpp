#include "models/qwen4_exp/program/expert_residency.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ninfer::models::qwen4_exp {
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
    return (keys + route) * sizeof(std::int32_t);
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
    controller_ = std::make_unique<CacheController>(static_cast<std::uint32_t>(keys), 0, slack_frames(0), 1, limit_);
    table_device_ = DeviceBuffer(keys * sizeof(std::int32_t));
    table_host_   = PinnedHostBuffer(keys * sizeof(std::int32_t));
    std::fill_n(static_cast<std::int32_t*>(table_host_.data()), keys, -1);
    table_device_.copy_from_host(table_host_.data(), keys * sizeof(std::int32_t));
    route_stride_ = static_cast<std::size_t>(top_k_) * static_cast<std::size_t>(max_columns);
    route_device_ = DeviceBuffer(route_stride_ * layers_ * sizeof(std::int32_t));
    route_host_   = PinnedHostBuffer(route_stride_ * layers_ * sizeof(std::int32_t));
    seen_.assign(experts_, 0);
    group_.reserve(experts_);
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&table_ready_, cudaEventDisableTiming));
    table_dirty_ = false;
}

ExpertResidency::~ExpertResidency() {
    if (copy_stream_ != nullptr) { (void)cudaStreamSynchronize(copy_stream_); }
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
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    publish_landed();
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
        }
    }
    if (table_dirty_) { upload_table(compute); }
    out.frames = frames_;
    return out;
}

const std::int32_t* ExpertResidency::table(std::uint32_t layer) const noexcept {
    return static_cast<const std::int32_t*>(table_device_.p) + static_cast<std::size_t>(layer) * experts_;
}

void ExpertResidency::upload_table(cudaStream_t compute) {
    // SM-read, not a copy engine: the table must not wait behind this round's promotion copies.
    upload_pinned(table_device_.p, table_host_.data(), table_device_.bytes, compute);
    table_dirty_ = false;
}

void ExpertResidency::before_round(cudaStream_t compute) {
    publish_landed();
    if (table_dirty_) { upload_table(compute); }
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
            if (controller_->complete_load(load.key, load.frame, load.serial)) {
                table[load.key] = static_cast<std::int32_t>(load.frame);
                table_dirty_    = true;
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
}

void ExpertResidency::after_round(cudaStream_t compute, std::int32_t columns, std::size_t per_layer_budget,
                                  std::span<const std::uint8_t> live) {
    const auto* routes = static_cast<const std::int32_t*>(route_host_.data());
    auto* table        = static_cast<std::int32_t*>(table_host_.data());
    const std::size_t used = static_cast<std::size_t>(top_k_) * static_cast<std::size_t>(columns);
    ++round_;
    commands_.clear();
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
        for (const auto key : group_) {
            seen_[key - layer * experts_] = 0;
            ++stats_.routed;
            if (table[key] >= 0) { ++stats_.hits; }
        }
        if (controller_) { controller_->on_route(group_, round_, commands_, per_layer_budget); }
    }
    if (frames_ == 0) { return; }
    controller_->on_quiescent(commands_);

    // Evictions take effect on the compute stream before the next round; promotions copy on the
    // copy stream once that table is in place.
    for (const auto& command : commands_) {
        if (command.kind == CacheController::Command::Kind::kWriteEntry) {
            const ResidencyEntry entry = ResidencyEntry::decode(command.word);
            if (entry.state != ResidencyState::kReady && table[command.key] >= 0) {
                table[command.key] = -1;
                table_dirty_       = true;
            }
        }
    }
    if (table_dirty_) { upload_table(compute); }
    issue_loads(compute);
}

ExpertResidency::SavedState ExpertResidency::saved_state() const {
    SavedState out;
    if (!controller_) { return out; }
    const auto& policy = controller_->policy();
    out.counts.resize(policy.num_keys());
    for (std::uint32_t key = 0; key < policy.num_keys(); ++key) { out.counts[key] = policy.count(key); }
    out.ranked.assign(policy.residents().begin(), policy.residents().end());
    std::stable_sort(out.ranked.begin(), out.ranked.end(),
                     [&](std::uint32_t a, std::uint32_t b) { return policy.score(a) > policy.score(b); });
    return out;
}

std::uint32_t ExpertResidency::warm_start(const SavedState& state, std::uint32_t count_cap, cudaStream_t compute) {
    if (!controller_ || frames_ == 0) { return 0; }
    if (!state.counts.empty() && state.counts.size() != static_cast<std::size_t>(layers_) * experts_) {
        throw std::invalid_argument("expert residency: the saved state has another key count");
    }
    std::vector<std::uint32_t> counts(state.counts);
    for (auto& count : counts) { count = std::min(count, count_cap); }
    commands_.clear();
    const std::uint32_t loaded = controller_->seed(state.ranked, counts, commands_);
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
    publish_landed();
    std::vector<std::uint8_t> busy(frames_, 0);
    for (const auto& batch : in_flight_) {
        for (const auto& load : batch.loads) {
            if (load.frame < frames_) { busy[load.frame] = 1; }
        }
    }
    const auto first = controller_->choose_run(count, busy);
    if (!first) { throw std::logic_error("expert residency: no run of frames to lend"); }
    commands_.clear();
    stats_.lend_evictions += controller_->lend(*first, count, commands_);
    auto* table = static_cast<std::int32_t*>(table_host_.data());
    for (const auto& command : commands_) {
        if (command.kind == CacheController::Command::Kind::kWriteEntry &&
            ResidencyEntry::decode(command.word).state != ResidencyState::kReady && table[command.key] >= 0) {
            table[command.key] = -1;
            table_dirty_       = true;
        }
    }
    if (table_dirty_) { upload_table(compute); }
    // Promotions still landing in the run must finish before its new user writes it.
    for (const auto& batch : in_flight_) {
        const bool targets = std::any_of(batch.loads.begin(), batch.loads.end(), [&](const Batch::Load& load) {
            return load.frame >= *first && load.frame < *first + count;
        });
        if (!targets) { continue; }
        CUDA_CHECK(cudaStreamWaitEvent(compute, batch.done, 0));
        for (const cudaStream_t writer : writers) { CUDA_CHECK(cudaStreamWaitEvent(writer, batch.done, 0)); }
    }
    stats_.lent_frames += count;
    auto* base = const_cast<std::uint8_t*>(frame_base());
    return {*first, count, DeviceSpan{base + static_cast<std::size_t>(*first) * stride_, static_cast<std::size_t>(count) * stride_}};
}

void ExpertResidency::give_back(FrameLease& lease, cudaStream_t compute) {
    if (!lease.valid()) { return; }
    commands_.clear();
    controller_->give_back(lease.first, lease.count, commands_);
    stats_.lent_frames -= lease.count;
    lease = {};
    issue_loads(compute); // queued experts load into the run once `compute` reaches here
}

void ExpertResidency::issue_loads(cudaStream_t compute) {
    Batch batch;
    for (const auto& command : commands_) {
        if (command.kind == CacheController::Command::Kind::kCopy) {
            batch.loads.push_back({command.key, command.frame, command.serial});
        }
    }
    if (batch.loads.empty()) { return; }
    CUDA_CHECK(cudaEventRecord(table_ready_, compute));
    CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, table_ready_, 0));
    auto* base = const_cast<std::uint8_t*>(frame_base());
    for (const auto& load : batch.loads) {
        const std::uint32_t layer  = load.key / experts_;
        const std::uint32_t expert = load.key % experts_;
        CUDA_CHECK(cudaMemcpyAsync(base + static_cast<std::size_t>(load.frame) * stride_,
                                   banks_[layer] + static_cast<std::size_t>(expert) * stride_, stride_,
                                   cudaMemcpyHostToDevice, copy_stream_));
    }
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

} // namespace ninfer::models::qwen4_exp
