#include "models/qwen3_5/execution/vision_overlay.h"

#include "models/qwen3_5/execution/vision_overlay.h"

#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace overlay_detail {

using OverlayClock = std::chrono::steady_clock;

std::size_t staging_align(std::size_t bytes) {
    constexpr std::size_t kAlign = 256;
    return (bytes + kAlign - 1) / kAlign * kAlign;
}

// Rebase a pointer that lives inside a group's pinned bytes into the group's staged slot, using
// the segment that contains it. A group's objects are not necessarily adjacent in the pinned
// block, so each object is staged independently.
const void* stage_pointer(const VisionOverlayGroup& group, std::byte* staging,
                          const std::byte* block, const void* pointer) {
    if (pointer == nullptr) { return nullptr; }
    const std::size_t pinned =
        static_cast<std::size_t>(static_cast<const std::byte*>(pointer) - block);
    for (const VisionOverlaySegment& segment : group.segments) {
        if (pinned >= segment.pinned_offset && pinned < segment.pinned_offset + segment.bytes) {
            return staging + segment.staging_offset + (pinned - segment.pinned_offset);
        }
    }
    throw std::logic_error("overlay weight is outside its staged group");
}

Tensor stage_tensor(const VisionOverlayGroup& group, std::byte* staging, const std::byte* block,
                    const Tensor& source) {
    Tensor out = source;
    out.data   = const_cast<std::byte*>(
        static_cast<const std::byte*>(stage_pointer(group, staging, block, source.data)));
    return out;
}

Weight stage_weight(const VisionOverlayGroup& group, std::byte* staging, const std::byte* block,
                    const Weight& source) {
    Weight out   = source;
    out.payload  = static_cast<const std::byte*>(stage_pointer(group, staging, block, source.payload));
    out.qdata    = stage_pointer(group, staging, block, source.qdata);
    out.qhigh    = stage_pointer(group, staging, block, source.qhigh);
    out.scales   = stage_pointer(group, staging, block, source.scales);
    return out;
}

// Copy a group's pinned segments into its device slot through the copy stream.
void stage_group(cudaStream_t stream, std::byte* staging, const VisionOverlayGroup& group,
                 const std::byte* block) {
    for (const VisionOverlaySegment& segment : group.segments) {
        CUDA_CHECK(cudaMemcpyAsync(staging + segment.staging_offset, block + segment.pinned_offset,
                                   segment.bytes, cudaMemcpyHostToDevice, stream));
    }
}

} // namespace overlay_detail

VisionWeightStream::VisionWeightStream(DeviceContext& device, const VisionOverlayAssets& assets,
                                       std::byte* staging)
    : device_(device), assets_(assets) {
    using overlay_detail::staging_align;
    const VisionOverlayLayout& layout = assets.layout;
    prelude_ = staging;
    merger_  = prelude_ + staging_align(layout.prelude.bytes);
    slot_[0] = merger_ + staging_align(layout.merger.bytes);
    slot_[1] = slot_[0] + staging_align(layout.slot_bytes);
    for (cudaEvent_t& event : uploaded_) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&prelude_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&merger_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&compute_fence_, cudaEventDisableTiming));

    const std::byte* block = assets.pinned_block;
    overlay_detail::stage_group(copy_stream(), prelude_, layout.prelude, block);
    CUDA_CHECK(cudaEventRecord(prelude_event_, copy_stream()));
    overlay_detail::stage_group(copy_stream(), merger_, layout.merger, block);
    CUDA_CHECK(cudaEventRecord(merger_event_, copy_stream()));
    upload_bytes_ = layout.prelude.bytes + layout.merger.bytes;
    reset(device_.stream);
}

VisionWeightStream::~VisionWeightStream() {
    // The window synchronizes the device before restore; events are idle here.
    for (cudaEvent_t event : uploaded_) { (void)cudaEventDestroy(event); }
    (void)cudaEventDestroy(prelude_event_);
    (void)cudaEventDestroy(merger_event_);
    (void)cudaEventDestroy(compute_fence_);
}

VisionParameters VisionWeightStream::window_weights(const VisionParameters& host) const {
    using overlay_detail::stage_tensor;
    using overlay_detail::stage_weight;
    const VisionOverlayLayout& layout = assets_.layout;
    const std::byte* block            = assets_.pinned_block;
    const VisionOverlayGroup& prelude = layout.prelude;
    const VisionOverlayGroup& merger  = layout.merger;

    VisionParameters out = host;
    out.patch_embedding.weight =
        stage_weight(prelude, prelude_, block, host.patch_embedding.weight);
    out.patch_embedding_bias = stage_tensor(prelude, prelude_, block, host.patch_embedding_bias);
    out.position_embedding   = stage_tensor(prelude, prelude_, block, host.position_embedding);
    for (std::size_t layer = 0; layer < host.layers.size(); ++layer) {
        const VisionOverlayGroup& group  = layout.layers[layer];
        std::byte* staging               = slot_[layer % 2];
        const VisionBlockParameters& src = host.layers[layer];
        VisionBlockParameters dst        = src;
        dst.norm1.weight  = stage_tensor(group, staging, block, src.norm1.weight);
        dst.norm1.bias    = stage_tensor(group, staging, block, src.norm1.bias);
        dst.norm2.weight  = stage_tensor(group, staging, block, src.norm2.weight);
        dst.norm2.bias    = stage_tensor(group, staging, block, src.norm2.bias);
        dst.qkv.weight    = stage_weight(group, staging, block, src.qkv.weight);
        dst.qkv_bias      = stage_tensor(group, staging, block, src.qkv_bias);
        dst.output.weight = stage_weight(group, staging, block, src.output.weight);
        dst.output_bias   = stage_tensor(group, staging, block, src.output_bias);
        dst.fc1.weight    = stage_weight(group, staging, block, src.fc1.weight);
        dst.fc1_bias      = stage_tensor(group, staging, block, src.fc1_bias);
        dst.fc2.weight    = stage_weight(group, staging, block, src.fc2.weight);
        dst.fc2_bias      = stage_tensor(group, staging, block, src.fc2_bias);
        out.layers[layer] = dst;
    }
    out.merger_norm.weight = stage_tensor(merger, merger_, block, host.merger_norm.weight);
    out.merger_norm.bias   = stage_tensor(merger, merger_, block, host.merger_norm.bias);
    out.merger_fc1.weight  = stage_weight(merger, merger_, block, host.merger_fc1.weight);
    out.merger_fc1_bias    = stage_tensor(merger, merger_, block, host.merger_fc1_bias);
    out.merger_fc2.weight  = stage_weight(merger, merger_, block, host.merger_fc2.weight);
    out.merger_fc2_bias    = stage_tensor(merger, merger_, block, host.merger_fc2_bias);
    return out;
}

void VisionWeightStream::reset(cudaStream_t compute) {
    CUDA_CHECK(cudaEventRecord(compute_fence_, compute));
    CUDA_CHECK(cudaStreamWaitEvent(copy_stream(), compute_fence_, 0));
    next_upload_ = 0;
    upload_next_layer();
    upload_next_layer();
}

void VisionWeightStream::prelude_ready(cudaStream_t compute) {
    CUDA_CHECK(cudaStreamWaitEvent(compute, prelude_event_, 0));
}

void VisionWeightStream::merger_ready(cudaStream_t compute) {
    CUDA_CHECK(cudaStreamWaitEvent(compute, merger_event_, 0));
}

void VisionWeightStream::arrive(std::uint32_t layer, cudaStream_t compute) {
    CUDA_CHECK(cudaStreamWaitEvent(compute, uploaded_[layer % 2], 0));
    const std::uint32_t layers = static_cast<std::uint32_t>(assets_.layout.layers.size());
    if (next_upload_ == layer + 1 && next_upload_ < layers) {
        CUDA_CHECK(cudaEventRecord(compute_fence_, compute));
        CUDA_CHECK(cudaStreamWaitEvent(copy_stream(), compute_fence_, 0));
        upload_next_layer();
    }
}

void VisionWeightStream::upload_next_layer() {
    if (next_upload_ >= static_cast<std::uint32_t>(assets_.layout.layers.size())) { return; }
    const std::uint32_t layer       = next_upload_++;
    const VisionOverlayGroup& group = assets_.layout.layers[layer];
    overlay_detail::stage_group(copy_stream(), slot_[layer % 2], group, assets_.pinned_block);
    CUDA_CHECK(cudaEventRecord(uploaded_[layer % 2], copy_stream()));
    upload_bytes_ += group.bytes;
}

std::vector<PinnedVisionResult>
encode_items_overlay(DeviceContext& device, const Parameters& parameters,
                     const qwen3_5::PreparedPromptData& prompt, const detail::VisionPrefillPlan& plan,
                     std::size_t first_item, VisionOverlayWindowStats* stats) {
    using overlay_detail::OverlayClock;
    using overlay_detail::staging_align;
    const Model& model = parameters.model;
    if (!model.weights().vision) {
        throw std::logic_error("overlay window requires vision weights");
    }
    if (!model.overlay_vision()) {
        throw std::logic_error("overlay window requires overlay assets");
    }
    const VisionOverlayAssets& assets = *model.overlay_vision();
    EvictableWeightPool& pool         = *assets.pool;
    if (plan.control == nullptr || plan.uses.empty()) {
        throw std::invalid_argument("overlay window has no vision items");
    }
    const std::size_t prepared_begin = plan.control->prepared_item_begin;
    const std::size_t prepared_end   = prepared_begin + plan.control->items.size();
    if (first_item < prepared_begin || first_item > prepared_end ||
        prepared_end > prompt.vision_items.size()) {
        throw std::invalid_argument("overlay window start is outside the prepared items");
    }
    const std::size_t first_control = first_item - prepared_begin;

    const auto& config = model.config().vision.value();
    const auto& params = parameters.vision.value();

    // The window is self-contained: weight staging, encode workspace, and the item output all
    // live in memory borrowed from the evicted weight tail.
    std::size_t max_patches = 0;
    std::uint32_t max_merged = 0;
    for (std::size_t index = first_control; index < plan.control->items.size(); ++index) {
        const qwen3_5::VisionItemControl& control = plan.control->items[index];
        max_patches = std::max(max_patches, control.patch_count);
        max_merged  = std::max(max_merged, static_cast<std::uint32_t>(control.merged_count));
    }
    const auto window_plan =
        VisionContext::plan_overlay_window(config, params, max_patches, max_merged);

    const auto window_start = OverlayClock::now();
    std::size_t mapped      = 0;
    std::byte* staging      =
        pool.evict(staging_align(assets.layout.staging_bytes) + window_plan.capacity_bytes,
                   &mapped);
    std::byte* lease = staging + staging_align(assets.layout.staging_bytes);

    struct RestoreGuard {
        EvictableWeightPool& pool;
        cudaStream_t stream;
        ~RestoreGuard() {
            // On the failure path in-flight work may still reference the overlay range;
            // quiesce before remapping so restore stays safe.
            (void)cudaDeviceSynchronize();
            pool.restore(stream);
        }
    } restore_guard{pool, device.stream};

    const DeviceSpan lease_span{lease, window_plan.capacity_bytes};

    std::vector<PinnedVisionResult> results;
    std::size_t staged_bytes = 0;
    {
        VisionWeightStream stream(device, assets, staging);
        const VisionParameters window_view = stream.window_weights(params);
        const VisionContext context(device, config, window_view);
        results.reserve(prepared_end);
        for (std::size_t skipped = 0; skipped < first_item; ++skipped) {
            results.push_back(PinnedVisionResult{});  // prefix-reused: never consumed
        }
        for (std::size_t index = first_control; index < plan.control->items.size(); ++index) {
            const qwen3_5::VisionItemControl& control = plan.control->items[index];
            const std::size_t prepared_index          = prepared_begin + index;
            const qwen3_5::VisionItem& source         = prompt.vision_items.at(prepared_index);
            const std::size_t patch_elements =
                control.patch_count * static_cast<std::size_t>(config.patch_width());
            const auto& payload = prompt.media_payloads.at(prepared_index);
            if (source.patch_begin != control.patch_begin || payload == nullptr ||
                payload->patch_elements != patch_elements) {
                throw std::invalid_argument("overlay item patch payload has an invalid shape");
            }
            Tensor output = VisionContext::bind_output(lease_span, window_plan,
                                                       control.merged_count);

            if (index != first_control) { stream.reset(device.stream); }
            context.encode(VisionItemView{payload->span(), &control}, output, lease_span,
                           window_plan, &stream);

            const std::size_t embedding_bytes = output.bytes();
            results.push_back(PinnedVisionResult{
                std::make_unique<PinnedHostBuffer>(embedding_bytes), embedding_bytes});
            CUDA_CHECK(cudaMemcpyAsync(results.back().buffer->data(), output.data, embedding_bytes,
                                       cudaMemcpyDeviceToHost, device.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
        staged_bytes = stream.uploaded_bytes();
    }

    pool.restore(device.stream);  // the guard's later call becomes a no-op

    if (stats != nullptr) {
        stats->window_seconds  =
            std::chrono::duration<double>(OverlayClock::now() - window_start).count();
        stats->evict_seconds   = pool.last_evict_seconds();
        stats->restore_seconds = pool.last_restore_seconds();
        stats->evicted_bytes   = mapped;
        stats->staged_bytes    = staged_bytes;
    }
    return results;
}

void encode_overlay_suffix(DeviceContext& device, const Parameters& parameters,
                           const qwen3_5::PreparedPromptData& prompt,
                           const detail::VisionPrefillPlan& plan, std::uint32_t reused,
                           VisionPrefillSession& session) {
    const std::size_t item_end = plan.control->prepared_item_begin + plan.control->items.size();
    std::size_t first_needed   = item_end;
    for (const detail::VisionUseSpan& use : plan.uses) {
        if (use.end > reused && use.prepared_item_index < first_needed) {
            first_needed = use.prepared_item_index;
        }
    }
    VisionOverlayWindowStats stats;
    std::vector<PinnedVisionResult> results;
    if (first_needed < item_end) {
        results = encode_items_overlay(device, parameters, prompt, plan, first_needed, &stats);
    } else {
        results.resize(item_end);
    }
    session.set_preencoded(std::move(results), stats);
}

} // namespace ninfer::models::qwen3_5::execution
