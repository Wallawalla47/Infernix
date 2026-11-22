#include "media/decode/resize.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace infernix::media::decode {
namespace {

int round_even(double value) { return static_cast<int>(std::nearbyint(value)); }

double cubic(double x) {
    // Torchvision's antialiased bicubic path uses the Keys/Pillow coefficient.
    constexpr double a = -0.5;
    x                  = std::abs(x);
    if (x < 1.0) { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0; }
    if (x < 2.0) { return (((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a); }
    return 0.0;
}

struct Coefficients {
    std::vector<int> starts;
    std::vector<int> offsets;
    std::vector<float> weights;
};

Coefficients coefficients(int input, int output) {
    Coefficients out;
    out.starts.resize(static_cast<std::size_t>(output));
    out.offsets.resize(static_cast<std::size_t>(output + 1));
    const double scale    = static_cast<double>(input) / output;
    const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    const double support  = 2.0 * (scale >= 1.0 ? scale : 1.0);
    for (int dst = 0; dst < output; ++dst) {
        const double center = scale * (dst + 0.5);
        const int begin     = std::max(static_cast<int>(center - support + 0.5), 0);
        const int size      = std::min(static_cast<int>(center + support + 0.5), input) - begin;
        out.starts[static_cast<std::size_t>(dst)]  = begin;
        out.offsets[static_cast<std::size_t>(dst)] = static_cast<int>(out.weights.size());
        double sum                                 = 0.0;
        for (int j = 0; j < size; ++j) {
            const double weight = cubic((j + begin - center + 0.5) * invscale);
            out.weights.push_back(static_cast<float>(weight));
            sum += weight;
        }
        if (sum == 0.0) { throw std::runtime_error("bicubic resize produced zero weights"); }
        const int first = out.offsets[static_cast<std::size_t>(dst)];
        for (std::size_t i = static_cast<std::size_t>(first); i < out.weights.size(); ++i) {
            out.weights[i] = static_cast<float>(out.weights[i] / sum);
        }
    }
    out.offsets[static_cast<std::size_t>(output)] = static_cast<int>(out.weights.size());
    return out;
}

} // namespace

Image resize_bicubic(const Image& input, int width, int height, const std::function<void()>& checkpoint) {
    if (width <= 0 || height <= 0 || input.width <= 0 || input.height <= 0) {
        throw std::invalid_argument("bicubic resize needs positive sizes");
    }
    const auto check = [&] {
        if (checkpoint) { checkpoint(); }
    };
    const Coefficients horizontal = coefficients(input.width, width);
    const Coefficients vertical   = coefficients(input.height, height);
    std::vector<std::uint8_t> temp(static_cast<std::size_t>(input.height) * width * 3);
    for (int y = 0; y < input.height; ++y) {
        if (y % 16 == 0) { check(); }
        for (int x = 0; x < width; ++x) {
            const int first        = horizontal.offsets[static_cast<std::size_t>(x)];
            const int last         = horizontal.offsets[static_cast<std::size_t>(x + 1)];
            const int source_begin = horizontal.starts[static_cast<std::size_t>(x)];
            std::array<float, 3> value{};
            const std::uint8_t* source =
                input.rgb.data() + (static_cast<std::size_t>(y) * input.width + source_begin) * 3;
            for (int i = first; i < last; ++i, source += 3) {
                const float weight = horizontal.weights[static_cast<std::size_t>(i)];
                value[0] += weight * source[0];
                value[1] += weight * source[1];
                value[2] += weight * source[2];
            }
            std::uint8_t* destination = temp.data() + (static_cast<std::size_t>(y) * width + x) * 3;
            for (int c = 0; c < 3; ++c) {
                destination[c] = static_cast<std::uint8_t>(std::clamp(round_even(value[c]), 0, 255));
            }
        }
    }

    Image out;
    out.width  = width;
    out.height = height;
    out.rgb.resize(static_cast<std::size_t>(height) * width * 3);
    for (int y = 0; y < height; ++y) {
        if (y % 16 == 0) { check(); }
        const int first        = vertical.offsets[static_cast<std::size_t>(y)];
        const int last         = vertical.offsets[static_cast<std::size_t>(y + 1)];
        const int source_begin = vertical.starts[static_cast<std::size_t>(y)];
        for (int x = 0; x < width; ++x) {
            std::array<float, 3> value{};
            const std::uint8_t* source = temp.data() + (static_cast<std::size_t>(source_begin) * width + x) * 3;
            for (int i = first; i < last; ++i, source += static_cast<std::size_t>(width) * 3) {
                const float weight = vertical.weights[static_cast<std::size_t>(i)];
                value[0] += weight * source[0];
                value[1] += weight * source[1];
                value[2] += weight * source[2];
            }
            std::uint8_t* destination = out.rgb.data() + (static_cast<std::size_t>(y) * width + x) * 3;
            for (int c = 0; c < 3; ++c) {
                destination[c] = static_cast<std::uint8_t>(std::clamp(round_even(value[c]), 0, 255));
            }
        }
    }
    return out;
}

} // namespace infernix::media::decode
