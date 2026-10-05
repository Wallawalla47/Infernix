#pragma once

// Streams the Vision tower's weights from their pinned host block through device staging
// (docs/maintainer/qwen3_8-flash-next-design.md §19.3.2): a prelude region (patch and position
// embedding), two layer slots refilled one layer ahead of compute, and a merger region. All
// synchronization is device-side (events); the host never blocks between layers. Shared by
// Qwen3.5's overlay window and Qwen4Exp's encode window; Program-free.

#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/load/vision_overlay.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::models::qwen3_5::execution {

// Staging regions start on 256-byte boundaries.
[[nodiscard]] constexpr std::size_t vision_staging_align(std::size_t bytes) noexcept {
    return (bytes + 255) / 256 * 256;
}

class VisionWeightStream {
public:
    // `staging` holds VisionOverlayLayout::staging_bytes. The first upload waits for a fence on the
    // compute stream (everything already submitted there, e.g. work still reading the staging
    // memory) and for every event of `wait_before_first_upload` (e.g. copies still landing in it).
    // Uploads run prelude, layer 0, layer 1, ..., then the merger after the last layer.
    VisionWeightStream(DeviceContext& device, const VisionOverlayAssets& assets, std::byte* staging,
                       std::span<const cudaEvent_t> wait_before_first_upload = {});
    // Without finish(), synchronizes the transfer stream (failure paths) before releasing events.
    ~VisionWeightStream();

    VisionWeightStream(const VisionWeightStream&)            = delete;
    VisionWeightStream& operator=(const VisionWeightStream&) = delete;

    // Rebased view of the host weights: every layer's tensors point at the slot that will hold the
    // layer when arrive(layer) admits it.
    [[nodiscard]] VisionParameters window_weights(const VisionParameters& host) const;

    // Starts another pass over the layers (Qwen3.5 encodes one item per pass): uploads of layers 0
    // and 1 follow everything already submitted on `compute`, which still reads the slots.
    void reset(cudaStream_t compute);

    void prelude_ready(cudaStream_t compute);
    void merger_ready(cudaStream_t compute);

    // At the top of the encoder loop for `layer`: gates compute on the slot upload, then refills the
    // slot the previous layer just vacated.
    void arrive(std::uint32_t layer, cudaStream_t compute);

    // After the last use: `compute` waits for every upload, and the staging memory may be reused
    // once `compute` reaches this point.
    void finish(cudaStream_t compute);

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
    cudaEvent_t prelude_event_ = nullptr;
    cudaEvent_t merger_event_  = nullptr;
    cudaEvent_t compute_fence_ = nullptr;
    cudaEvent_t final_event_   = nullptr;
    std::uint32_t next_upload_ = 0;
    bool merger_uploaded_      = false;
    bool finished_             = false;
    std::size_t upload_bytes_  = 0;
};

} // namespace ninfer::models::qwen3_5::execution
