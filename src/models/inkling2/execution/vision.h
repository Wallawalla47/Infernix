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
    // Device bytes of the tower's buffers for images of up to max_cells cells.
    [[nodiscard]] static std::uint64_t device_bytes(const VisionConfig& config, std::uint32_t max_cells);

    VisionTower(const VisionConfig& config, const VisionOperands& operands, std::uint32_t max_cells);

    // Encodes one image on `stream`; returns its rows, BF16 [cells][output_size] in the tower's buffers, valid until
    // the next encode.
    const std::uint16_t* encode(const VisionImage& image, cudaStream_t stream);

    [[nodiscard]] std::uint32_t max_cells() const noexcept { return max_cells_; }

private:
    struct Layout;
    const VisionConfig& c_;
    const VisionOperands& w_;
    std::uint32_t max_cells_ = 0;
    DeviceBuffer buffer_;
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
