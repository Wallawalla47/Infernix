#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {

struct Mxfp8WeightGeometry {
    std::uint64_t code_plane_bytes;
    std::uint64_t scale_plane_offset;
    std::uint64_t scale_plane_bytes;
};

// Validates an fp8_e4m3fn_g32_e8m0 Weight in g32_scale_rows16_v1 (planes, strides, alignment).
Mxfp8WeightGeometry validate_mxfp8_weight(const Weight& weight, const char* operation);

// The A16 operands of an MXFP8 projection: BF16 activations [K, T], E4M3FN codes [N, K] and the
// UE8M0 scale plane in 16-row tiles.
struct Mxfp8Operands {
    const __nv_bfloat16* x;
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    int rows, k, tokens;
};

inline Mxfp8Operands mxfp8_operands(const Tensor& x, const Weight& w) {
    return {static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), w.n, w.k, x.ne[1]};
}

template <class Schedule>
inline void validate_mxfp8_operands(const Mxfp8Operands& p) {
    const auto aligned = [](const void* v, int alignment) {
        return v && reinterpret_cast<std::uintptr_t>(v) % alignment == 0;
    };
    if (p.rows <= 0 || p.k <= 0 || p.tokens <= 0 || p.rows % 16 || p.k % 32 || !aligned(p.x, 16) ||
        !aligned(p.codes, 16) || !aligned(p.scales, 16))
        throw std::invalid_argument("MXFP8 templates require positive N/K/T, 16-row tiles and aligned operands");
    if constexpr (Schedule::kStaticK > 0) {
        if (p.k != Schedule::kStaticK) throw std::invalid_argument("MXFP8 template static K does not match operands");
    }
}

} // namespace infernix::ops::detail
