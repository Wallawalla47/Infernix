#pragma once

#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace ninfer::product {

inline float parse_rope_yarn_factor(std::string_view text) {
    double value = 0.0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value) ||
        value < 1.0 || value > 4.0) {
        throw std::invalid_argument("--rope-yarn-factor must be finite and in [1,4]");
    }
    return static_cast<float>(value);
}

} // namespace ninfer::product
