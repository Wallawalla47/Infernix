#pragma once

// Inkling 2's image processor (transformers' Inkling2ImageProcessor; design docs/maintainer/inkling2_mini-design.md
// §15): RGB as decoded, a transparent image composited over a 204/153 checkerboard of 8-pixel squares
// ((rgb * a + bg * (255 - a) + 127) / 255), padded right and bottom to whole (patch * merge)-pixel cells with -1 on
// the 0-255 scale, (v - 127.5) / 127.5 in FP32, then BF16. One image token per cell. The reference never resizes; an
// image of more cells than the budget (--vision-max-merged) is first downscaled (bicubic) to the largest whole cells
// that fit with its aspect kept (smart_resize's floor), or refused when its request says so.

#include "infernix/types.h"
#include "models/inkling2/config.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace infernix::models::inkling2 {

struct ProcessedImage {
    std::int32_t rows = 0; // cells down
    std::int32_t cols = 0; // cells across
    MediaGeometry source;  // the decoded size
    // BF16 [merge^2 rows cols][3 patch^2]: patches merge-major (cell row-major, then (dy, dx)), values (ky, kx, c).
    std::shared_ptr<const std::vector<std::uint16_t>> patches;
    [[nodiscard]] std::int32_t cells() const noexcept { return rows * cols; }
};

// The cells an image of this size takes (after the budget's downscale), without decoding it.
struct ImageCells {
    std::int32_t rows = 0, cols = 0;
    bool resized = false;
};
[[nodiscard]] ImageCells image_cells(std::int32_t width, std::int32_t height, const VisionConfig& config,
                                     std::uint32_t max_cells);

// Decodes and processes one image (RequestError InvalidMedia or MediaBudgetExceeded on a refused one).
[[nodiscard]] ProcessedImage process_image(const OwnedMedia& media, const VisionConfig& config, std::uint32_t max_cells,
                                           const std::function<void()>& checkpoint = {});

// The geometry of an image without decoding its pixels.
[[nodiscard]] MediaGeometry image_geometry(const OwnedMedia& media);

} // namespace infernix::models::inkling2
