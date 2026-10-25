#pragma once

#include <cstdint>
#include <span>

#include "media/decode/decode.h"

namespace infernix::media::decode {

// True when the bytes start with the JPEG start-of-image marker.
bool is_jpeg(std::span<const std::uint8_t> bytes);

// JPEG through libjpeg(-turbo) with Pillow's settings (islow IDCT, fancy upsampling, libjpeg's
// integer YCbCr->RGB; Adobe CMYK/YCCK as Pillow's inverted CMYK composited to RGB), so the pixels
// equal what the Hugging Face and vLLM processors decode. Both return the stored geometry and
// pixels, before EXIF orientation. inspect_jpeg decodes the whole image (as grayscale or CMYK) so
// a truncated or corrupt image fails in probing as it would in decoding. Malformed input throws
// Error(ErrorKind::InvalidInput); the byte and pixel budgets throw Error(ErrorKind::BudgetExceeded);
// Policy::checkpoint runs before every scanline.
ImageInfo inspect_jpeg(std::span<const std::uint8_t> bytes, const Policy& policy);
Image decode_jpeg(std::span<const std::uint8_t> bytes, const Policy& policy);

} // namespace infernix::media::decode
