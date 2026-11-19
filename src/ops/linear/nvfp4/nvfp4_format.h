#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cstdint>

namespace infernix::ops::detail {

struct Nvfp4WeightGeometry {
    std::uint64_t code_plane_bytes;
    std::uint64_t scale_plane_offset;
    std::uint64_t scale_plane_bytes;
    std::uint64_t required_payload_bytes;
};

Nvfp4WeightGeometry validate_nvfp4_weight(const Weight& weight, const char* operation);

// An NVFP4 Weight's matrix coefficient: exactly one of its FP32 divisor (nvfp4) and multiplier (an
// nvfp4_mul BlockScale parent: ModelOpt's weight_scale_2) is set, positive and finite.
[[nodiscard]] inline bool valid_nvfp4_coefficient(const Weight& w) noexcept {
    const bool divisor    = w.weight_scale_divisor > 0.0F && w.weight_scale_divisor <= 3.4028235e38F;
    const bool multiplier = w.weight_multiplier > 0.0F && w.weight_multiplier <= 3.4028235e38F;
    return divisor != multiplier && (divisor || w.weight_scale_divisor == 0.0F) &&
           (multiplier || w.weight_multiplier == 0.0F);
}

// The FP32 factor the 16-bit-activation kernels scale each product sum by: the multiplier as stored,
// or the divisor's reciprocal.
[[nodiscard]] inline float nvfp4_a16_alpha(const Weight& w) noexcept {
    return w.weight_multiplier > 0.0F ? w.weight_multiplier : 1.0F / w.weight_scale_divisor;
}

// The 4-bit-activation kernels' factor: fl32(s * multiplier) for a multiplier-form matrix (s the input's
// stored activation scale: vLLM's alpha, input_scale * weight_scale_2), else 1 / (input divisor * divisor).
// A multiplier-form matrix with no input scale gives 0, which the operand checks refuse.
[[nodiscard]] inline float nvfp4_a4_alpha(const Weight& w) noexcept {
    return w.weight_multiplier > 0.0F ? w.input_scale * w.weight_multiplier
                                      : 1.0F / (w.input_scale_divisor * w.weight_scale_divisor);
}

// The 8-bit-activation kernels' factor (nvfp4_a8_mma.cuh): the 16-bit one halved for their doubled E2M1 codes
// (exact: a power-of-two scaling); the activations' own slice scales are already in the sums.
[[nodiscard]] inline float nvfp4_a8_alpha(const Weight& w) noexcept { return 0.5F * nvfp4_a16_alpha(w); }

} // namespace infernix::ops::detail
