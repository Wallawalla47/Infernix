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

ExpertResidency::ExpertResidency(const TextConfig& config, std::vector<const std::uint8_t*> banks,
                                 std::uint64_t record_stride, std::uint32_t frames, std::int32_t max_columns)
    : c_(config), banks_(std::move(banks)), stride_(record_stride), frames_(frames) {
    experts_ = config.moe.experts;
    layers_  = config.num_hidden_layers;
    top_k_   = config.moe.top_k;
    if (banks_.size() != layers_) { throw std::invalid_argument("expert residency: one bank per layer"); }
    const std::size_t keys = static_cast<std::size_t>(layers_) * experts_;
    if (frames_ > 0) {
        if (frames_ >= keys) { frames_ = static_cast<std::uint32_t>(keys) - 1; }
        frame_memory_ = DeviceBuffer(static_cast<std::size_t>(frames_) * stride_);
        controller_   = std::make_unique<CacheController>(static_cast<std::uint32_t>(keys), frames_,
                                                         slack_frames(frames_), 1);
    }
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
    return static_cast<const std::uint8_t*>(frame_memory_.p);
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
    if (!controller_) { return; }
    auto* table = static_cast<std::int32_t*>(table_host_.data());
    for (auto it = in_flight_.begin(); it != in_flight_.end();) {
        const cudaError_t status = cudaEventQuery(it->done);
        if (status == cudaErrorNotReady) {
            ++it;
            continue;
        }
        CUDA_CHECK(status);
        for (const auto& [key, frame] : it->loads) {
            // Publish only loads the policy still wants in that frame.
            const ResidencyEntry& entry = controller_->entry(key);
            if (entry.state == ResidencyState::kReady && entry.frame == frame) {
                table[key]   = static_cast<std::int32_t>(frame);
                table_dirty_ = true;
            }
        }
        spare_events_.push_back(it->done);
        it = in_flight_.erase(it);
    }
    if (table_dirty_) { upload_table(compute); }
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
    if (!controller_) { return; }
    controller_->on_quiescent(commands_);

    // Evictions take effect on the compute stream before the next round; promotions copy on the
    // copy stream once that table is in place.
    Batch batch;
    for (const auto& command : commands_) {
        if (command.kind == CacheController::Command::Kind::kWriteEntry) {
            const ResidencyEntry entry = ResidencyEntry::decode(command.word);
            if (entry.state != ResidencyState::kReady && table[command.key] >= 0) {
                table[command.key] = -1;
                table_dirty_       = true;
            }
        } else {
            batch.loads.emplace_back(command.key, command.frame);
        }
    }
    if (table_dirty_) { upload_table(compute); }
    if (batch.loads.empty()) { return; }
    CUDA_CHECK(cudaEventRecord(table_ready_, compute));
    CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, table_ready_, 0));
    auto* base = static_cast<std::uint8_t*>(frame_memory_.p);
    for (const auto& [key, frame] : batch.loads) {
        const std::uint32_t layer  = key / experts_;
        const std::uint32_t expert = key % experts_;
        CUDA_CHECK(cudaMemcpyAsync(base + static_cast<std::size_t>(frame) * stride_,
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
