#include "ops/linear/mxfp8/mxfp8_format.h"

#include <cstddef>
#include <limits>
#include <string>

namespace infernix::ops::detail {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

} // namespace

Mxfp8WeightGeometry validate_mxfp8_weight(const Weight& weight, const char* operation) {
    const auto fail = [&](const char* why) { throw std::invalid_argument(std::string(operation) + ": " + why); };
    if (weight.n <= 0 || weight.k <= 0 || weight.n % 16 || weight.k % 32) {
        fail("MXFP8 needs N a multiple of 16 and K a multiple of 32");
    }
    const auto n = static_cast<std::uint64_t>(weight.n), k = static_cast<std::uint64_t>(weight.k);
    if (n > std::numeric_limits<std::uint64_t>::max() / k) { fail("MXFP8 geometry overflows"); }
    Mxfp8WeightGeometry geometry{};
    geometry.code_plane_bytes   = n * k;
    geometry.scale_plane_offset = (geometry.code_plane_bytes + 255) / 256 * 256;
    geometry.scale_plane_bytes  = n * (k / 32);
    const std::int64_t tile = static_cast<std::int64_t>(weight.k / 32) * 16;
    if (weight.qtype != QType::FP8_E4M3FN_G32_E8M0 || weight.layout != QuantLayout::G32ScaleRows16 ||
        weight.scale_dtype != DType::U8 || weight.group_size != 32 || weight.group != 32 || weight.ndim != 2 ||
        weight.shape[0] != weight.n || weight.shape[1] != weight.k || weight.padded_shape[0] != weight.n ||
        weight.padded_shape[1] != weight.k || weight.scale_ne[0] != 16 || weight.scale_ne[1] != weight.k / 32 ||
        weight.scale_ne[2] != weight.n / 16 || weight.scale_nb[0] != 1 || weight.scale_nb[1] != 16 ||
        weight.scale_nb[2] != tile || weight.payload == nullptr || weight.qdata == nullptr ||
        weight.scales == nullptr || weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        weight.payload_bytes < geometry.scale_plane_offset + geometry.scale_plane_bytes ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 16)) {
        fail("invalid MXFP8 weight");
    }
    const auto* payload = static_cast<const std::byte*>(weight.payload);
    if (weight.qdata != payload || weight.scales != payload + geometry.scale_plane_offset) {
        fail("invalid MXFP8 plane geometry");
    }
    return geometry;
}

} // namespace infernix::ops::detail
