#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/load/vision_overlay.h"
#include "models/qwen3_5/program/vision_prefill.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

// One vision item's merged embeddings, parked in pinned host memory by an overlay window and
// re-uploaded into the request transient when prefill consumes the item.
struct PinnedVisionResult {
    std::unique_ptr<PinnedHostBuffer> buffer = nullptr;  // null for prefix-reused items
    std::size_t bytes                        = 0;
};

struct VisionOverlayWindowStats {
    double window_seconds  = 0.0;
    double evict_seconds   = 0.0;
    double restore_seconds = 0.0;
    std::size_t evicted_bytes = 0;
    std::size_t staged_bytes  = 0;
};

// Streams vision weights from the pinned block through borrowed device staging: a fixed
// prelude region (patch/position embedding), a fixed merger region, and two layer slots
// refilled on the transfer stream one layer ahead of compute. All synchronization is
// device-side (events); the host never blocks between layers.
class VisionWeightStream {
public:
    VisionWeightStream(DeviceContext& device, const VisionOverlayAssets& assets,
                       std::byte* staging);
    ~VisionWeightStream();

    VisionWeightStream(const VisionWeightStream&)            = delete;
    VisionWeightStream& operator=(const VisionWeightStream&) = delete;

    // Rebased view of the host weights: every layer's tensors point at the slot that will
    // hold the layer when arrive(layer) admits it.
    [[nodiscard]] VisionParameters window_weights(const VisionParameters& host) const;

    // Prepare the next encode pass: uploads of layers 0 and 1 are issued after everything
    // already submitted on the compute stream (the previous item's tail layers still own
    // the slots until then).
    void reset(cudaStream_t compute);

    void prelude_ready(cudaStream_t compute);
    void merger_ready(cudaStream_t compute);

    // Called at the top of the encoder loop for `layer`: gates compute on the slot upload,
    // then refills the slot the previous layer just vacated.
    void arrive(std::uint32_t layer, cudaStream_t compute);

    [[nodiscard]] std::size_t uploaded_bytes() const noexcept { return upload_bytes_; }

private:
    [[nodiscard]] cudaStream_t copy_stream() const noexcept { return device_.transfer_stream; }
    void upload_next_layer();

    DeviceContext& device_;
    const VisionOverlayAssets& assets_;
    std::byte* prelude_      = nullptr;
    std::byte* merger_       = nullptr;
    std::byte* slot_[2]      = {nullptr, nullptr};
    cudaEvent_t uploaded_[2] = {nullptr, nullptr};
    cudaEvent_t prelude_event_  = nullptr;
    cudaEvent_t merger_event_   = nullptr;
    cudaEvent_t compute_fence_  = nullptr;
    std::uint32_t next_upload_  = 0;
    std::size_t upload_bytes_   = 0;
};

// Encode the vision items a request will consume inside a single overlay window: evict the
// staging extent from the weight pool tail, stream the vision tower through it, land each
// item's merged embeddings in pinned host memory, and restore the evicted text weights before
// returning. `first_item` is an absolute prepared-item index; the returned vector covers
// every prepared item (null entries for prefix-reused items).
[[nodiscard]] std::vector<PinnedVisionResult>
encode_items_overlay(DeviceContext& device, const Parameters& parameters,
                     const qwen3_5::PreparedPromptData& prompt, const detail::VisionPrefillPlan& plan,
                     std::size_t first_item, VisionOverlayWindowStats* stats);

class VisionPrefillSession;

// Encode, in one overlay window, the prepared items a prefill consumes past its `reused` prompt
// tokens and install them on `session`. Items wholly inside the reused prefix are not encoded:
// their embeddings already live in the reused context. Item indices are absolute prepared-item
// indices, which start at the plan's prepared_item_begin for a prompt extending a prefix.
void encode_overlay_suffix(DeviceContext& device, const Parameters& parameters,
                           const qwen3_5::PreparedPromptData& prompt,
                           const detail::VisionPrefillPlan& plan, std::uint32_t reused,
                           VisionPrefillSession& session);

} // namespace ninfer::models::qwen3_5::execution
