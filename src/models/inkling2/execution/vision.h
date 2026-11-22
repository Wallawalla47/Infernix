#pragma once

// Inkling 2's vision tower (design docs/maintainer/inkling2_mini-design.md §15): one image at a time, on the Program's
// stream, in buffers the Program allocates with itself (device_bytes() is part of its device plan). An image's rows
// depend on its own patches alone, so its image tokens get the same embedding in any prompt and any batch.
//
// Per image of R x C cells (P = merge^2 R C patches, T = 1 + P tokens): the patch embedding plus the bilinearly resized
// position table, the class token in front, the pre-norm, the layers (LayerNorm, fused QKV with its bias and the
// axial rotary, D64 attention over the whole image, the output projection into the layer-scaled residual; LayerNorm,
// the exact-GELU MLP into the second scaled residual), then each cell's merge^2 patch rows as one [merge^2 H] row
// through the projector. Token-parallel stages run kVisionChunk tokens at a time; Q/K/V and the residual are whole.

#include "core/arena.h"
#include "models/inkling2/config.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/model.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <vector>

namespace infernix::models::inkling2 {

inline constexpr std::int32_t kVisionChunk = 4096;

// One image's processed pixels: BF16 [P][3 p^2], patches in merge-major order (cell row-major, then (dy, dx) within
// the cell), each patch's values in (ky, kx, channel) order.
struct VisionImage {
    std::int32_t rows = 0; // cells down
    std::int32_t cols = 0; // cells across
    std::span<const std::uint16_t> patches;
    [[nodiscard]] std::int32_t cells() const noexcept { return rows * cols; }
};

class VisionTower {
public:
    // Device bytes of the tower's buffers for images of up to max_cells cells, with an offloaded tower's staging
    // (`host`: --vision-offload on).
    [[nodiscard]] static std::uint64_t device_bytes(const VisionConfig& config, std::uint32_t max_cells,
                                                    const VisionHostWeights* host = nullptr);

    // `host` (--vision-offload on, the operands addressing its pinned block): the tower keeps the patch embedding,
    // position table, class row, pre-norm and projector in a device region uploaded here, and streams each layer
    // through one of two device slots per encode (design §15.5): layer l + 1's upload on `transfer` overlaps layer l,
    // each layer's kernels wait for their slot's upload, and a slot is refilled once the layer before has been issued.
    VisionTower(const VisionConfig& config, const VisionOperands& operands, std::uint32_t max_cells,
                const VisionHostWeights* host = nullptr, cudaStream_t transfer = nullptr);
    ~VisionTower();
    VisionTower(const VisionTower&)            = delete;
    VisionTower& operator=(const VisionTower&) = delete;

    // Encodes one image on `stream`; returns its rows, BF16 [cells][output_size] in the tower's buffers, valid until
    // the next encode.
    const std::uint16_t* encode(const VisionImage& image, cudaStream_t stream);

    [[nodiscard]] std::uint32_t max_cells() const noexcept { return max_cells_; }
    // An offloaded tower's bytes uploaded by the last encode (0 when resident).
    [[nodiscard]] std::uint64_t streamed_bytes() const noexcept { return streamed_bytes_; }

private:
    struct Layout;
    void upload_layer(std::uint32_t layer);
    const VisionConfig& c_;
    const VisionOperands* w_ = nullptr;
    std::uint32_t max_cells_ = 0;
    DeviceBuffer buffer_;
    // --vision-offload on: the resident region and the two layer slots, the operands rebased into them, the upload
    // events (a slot's upload done; the compute stream past a slot's last reader).
    const VisionHostWeights* host_ = nullptr;
    cudaStream_t transfer_          = nullptr;
    DeviceBuffer staging_;
    std::byte* slot_[2]              = {nullptr, nullptr};
    VisionOperands staged_;
    cudaEvent_t uploaded_[2]         = {nullptr, nullptr};
    cudaEvent_t fence_               = nullptr;
    std::uint32_t next_upload_       = 0;
    std::uint64_t streamed_bytes_    = 0;
    std::uint16_t *x_ = nullptr, *qkv_ = nullptr, *att_ = nullptr, *h_ = nullptr, *o_ = nullptr, *u_ = nullptr;
    std::uint16_t* patches_     = nullptr;
    std::int32_t* positions_    = nullptr;
    std::int32_t* tap_index_    = nullptr;
    float* tap_weight_          = nullptr;
    void* arena_                = nullptr;
    std::vector<std::int32_t> host_positions_, host_index_;
    std::vector<float> host_weight_;
};

} // namespace infernix::models::inkling2
