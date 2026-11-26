#include "models/inkling2/frontend/image.h"

#include "media/decode/decode.h"
#include "media/decode/resize.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <string>

namespace infernix::models::inkling2 {
namespace {

std::uint16_t to_bf16(float value) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

// (v - 127.5) / 127.5 in FP32 (torchvision's normalize with the fused mean and std 0.5 * 255), BF16; index v + 1, v in
// [-1, 255] (-1 the padding).
const std::array<std::uint16_t, 257>& normalized() {
    static const std::array<std::uint16_t, 257> table = [] {
        std::array<std::uint16_t, 257> out{};
        for (int v = -1; v <= 255; ++v) { out[v + 1] = to_bf16((static_cast<float>(v) - 127.5F) / 127.5F); }
        return out;
    }();
    return table;
}

media::decode::Policy decode_policy(const std::function<void()>& checkpoint) {
    media::decode::Policy policy;
    policy.keep_alpha = true;
    policy.checkpoint = checkpoint;
    return policy;
}

[[noreturn]] void refuse(const media::decode::Error& error) {
    throw RequestError(error.kind() == media::decode::ErrorKind::BudgetExceeded ? RequestErrorKind::MediaBudgetExceeded
                                                                                 : RequestErrorKind::InvalidMedia,
                       std::string("Inkling 2 image: ") + error.what());
}

} // namespace

ImageCells image_cells(std::int32_t width, std::int32_t height, const VisionConfig& config, std::uint32_t max_cells,
                       std::uint32_t max_long_side) {
    const auto f = static_cast<std::int64_t>(config.cell());
    if (width <= 0 || height <= 0) { throw RequestError(RequestErrorKind::InvalidMedia, "Inkling 2 image: an empty image"); }
    ImageCells out{static_cast<std::int32_t>((height + f - 1) / f), static_cast<std::int32_t>((width + f - 1) / f)};
    // The long-side cap (detail 'low'): the whole cells whose longer side is at most the cap, the aspect kept.
    double h = height, w = width;
    if (const std::int32_t long_side = std::max(width, height);
        max_long_side != 0 && static_cast<std::uint32_t>(long_side) > max_long_side) {
        const double scale = static_cast<double>(max_long_side) / long_side;
        h *= scale;
        w *= scale;
        out.rows    = static_cast<std::int32_t>(std::max<std::int64_t>(1, static_cast<std::int64_t>(std::floor(h / f))));
        out.cols    = static_cast<std::int32_t>(std::max<std::int64_t>(1, static_cast<std::int64_t>(std::floor(w / f))));
        out.resized = true;
    }
    if (static_cast<std::uint64_t>(out.rows) * static_cast<std::uint64_t>(out.cols) <= max_cells) { return out; }
    // smart_resize's floor: the largest whole cells with the aspect kept that the budget holds.
    const double beta = std::sqrt(h * w / (static_cast<double>(max_cells) * f * f));
    out.rows        = static_cast<std::int32_t>(std::max<std::int64_t>(1, static_cast<std::int64_t>(std::floor(h / beta / f))));
    out.cols        = static_cast<std::int32_t>(std::max<std::int64_t>(1, static_cast<std::int64_t>(std::floor(w / beta / f))));
    out.resized     = true;
    out.over_budget = true;
    while (static_cast<std::uint64_t>(out.rows) * static_cast<std::uint64_t>(out.cols) > max_cells) { // a 1-cell side
        if (out.rows >= out.cols) { --out.rows; } else { --out.cols; }
    }
    return out;
}

MediaGeometry image_geometry(const OwnedMedia& media) {
    if (media.kind != MediaKind::Image) {
        throw RequestError(RequestErrorKind::InvalidMedia, "Inkling 2 takes images, not video");
    }
    try {
        const media::decode::ImageInfo info = media::decode::inspect_image(media.bytes, decode_policy({}));
        return {info.width, info.height};
    } catch (const media::decode::Error& error) {
        refuse(error);
    }
}

ProcessedImage process_image(const OwnedMedia& media, const VisionConfig& config, std::uint32_t max_cells,
                             const std::function<void()>& checkpoint) {
    if (media.kind != MediaKind::Image) {
        throw RequestError(RequestErrorKind::InvalidMedia, "Inkling 2 takes images, not video");
    }
    media::decode::Image image;
    try {
        image = media::decode::decode_image(media.bytes, decode_policy(checkpoint));
    } catch (const media::decode::Error& error) {
        refuse(error);
    }
    ProcessedImage out;
    out.source = {image.width, image.height};
    // A transparent picture over the checkerboard (Inkling2ImageProcessor.convert_to_rgb).
    if (!image.alpha.empty()) {
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                const std::size_t at = static_cast<std::size_t>(y) * image.width + x;
                const int a          = image.alpha[at];
                const int bg         = ((y / 8) + (x / 8)) % 2 == 0 ? 204 : 153;
                for (int c = 0; c < 3; ++c) {
                    std::uint8_t& v = image.rgb[3 * at + c];
                    v               = static_cast<std::uint8_t>((v * a + bg * (255 - a) + 127) / 255);
                }
            }
        }
        image.alpha.clear();
    }
    const ImageCells cells = image_cells(image.width, image.height, config, max_cells, media.image_max_long_side);
    const int f            = static_cast<int>(config.cell());
    if (cells.resized) {
        if (cells.over_budget && media.image_resize_policy == ImageResizePolicy::RejectOversized) {
            throw RequestError(RequestErrorKind::MediaBudgetExceeded,
                               "Inkling 2 image: " + std::to_string(image.width) + " x " + std::to_string(image.height) +
                                   " pixels take more than the " + std::to_string(max_cells) +
                                   " image tokens --vision-max-merged allows, and the request refuses downsizing");
        }
        image = media::decode::resize_bicubic(image, cells.cols * f, cells.rows * f, checkpoint);
    }
    out.rows = cells.rows;
    out.cols = cells.cols;

    // Patches: cell (r, c) row-major, then (dy, dx); each patch's values (ky, kx, channel), -1 past the picture.
    const int p = static_cast<int>(config.patch_size), m = static_cast<int>(config.merge);
    const std::size_t width = static_cast<std::size_t>(3) * p * p;
    auto patches = std::make_shared<std::vector<std::uint16_t>>(static_cast<std::size_t>(out.cells()) * m * m * width);
    const auto& lut = normalized();
    std::uint16_t* dst = patches->data();
    for (int r = 0; r < out.rows; ++r) {
        if (checkpoint) { checkpoint(); }
        for (int c = 0; c < out.cols; ++c) {
            for (int dy = 0; dy < m; ++dy) {
                for (int dx = 0; dx < m; ++dx) {
                    for (int ky = 0; ky < p; ++ky) {
                        const int Y = r * f + dy * p + ky;
                        for (int kx = 0; kx < p; ++kx) {
                            const int X = c * f + dx * p + kx;
                            if (Y < image.height && X < image.width) {
                                const std::uint8_t* px = image.rgb.data() + (static_cast<std::size_t>(Y) * image.width + X) * 3;
                                dst[0] = lut[px[0] + 1];
                                dst[1] = lut[px[1] + 1];
                                dst[2] = lut[px[2] + 1];
                            } else {
                                dst[0] = dst[1] = dst[2] = lut[0];
                            }
                            dst += 3;
                        }
                    }
                }
            }
        }
    }
    out.patches = std::move(patches);
    return out;
}

} // namespace infernix::models::inkling2
