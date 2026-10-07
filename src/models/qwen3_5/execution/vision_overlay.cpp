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

namespace infernix::models::qwen3_5::execution {
namespace {

using OverlayClock = std::chrono::steady_clock;

} // namespace

std::vector<PinnedVisionResult>
encode_items_overlay(DeviceContext& device, const Parameters& parameters,
                     const qwen3_5::PreparedPromptData& prompt, const detail::VisionPrefillPlan& plan,
                     std::size_t first_item, VisionOverlayWindowStats* stats) {
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
        pool.evict(vision_staging_align(assets.layout.staging_bytes) + window_plan.capacity_bytes,
                   &mapped);
    std::byte* lease = staging + vision_staging_align(assets.layout.staging_bytes);

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
        stream.finish(device.stream);
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
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

} // namespace infernix::models::qwen3_5::execution
