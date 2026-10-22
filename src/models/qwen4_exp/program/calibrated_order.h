#pragma once

// The calibrated expert order of Qwen3.8-Flash-Next: the expert cache's seed when no saved state
// exists (design §19.3.5 S4b). Every artifact of the architecture's shape uses it (FP8, NVFP4,
// Dense8, IQ3_S and their derivatives route through the same experts); the program checks the shape.

#include "runtime/expert_offload/expert_cache.h"

#include <cstdint>

namespace infernix::models::qwen4_exp {

inline constexpr std::uint32_t kCalibratedLayers  = 48;
inline constexpr std::uint32_t kCalibratedExperts = 512;

// The order as a saved state of kCalibratedLayers x kCalibratedExperts keys (key = layer * experts +
// expert): counts are the ranks reversed, and the ranked list takes every layer's first expert, then
// every layer's second, and so on.
[[nodiscard]] runtime::expert_offload::expert_cache::SavedState calibrated_expert_order();

} // namespace infernix::models::qwen4_exp
