#pragma once

#include "media/decode/decode.h"

#include <functional>

namespace infernix::media::decode {

// Separable bicubic resampling of 8-bit RGB with the antialiased coefficients torchvision and Pillow use (Keys,
// a = -0.5; the support widened by the scale when downsampling; each output's weights normalized in FP64 and kept in
// FP32), horizontal pass first, each pass's sums in FP32 rounded half to even and clamped to 0-255. `checkpoint`
// (optional) runs every 16 rows of each pass. The alpha plane is not resampled (the result has none).
[[nodiscard]] Image resize_bicubic(const Image& input, int width, int height,
                                   const std::function<void()>& checkpoint = {});

} // namespace infernix::media::decode
