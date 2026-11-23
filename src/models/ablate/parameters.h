#pragma once

#include "core/arena.h"
#include "infernix/ops/direction_cut.h"
#include "models/ablate/binding.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace infernix::models {

// --ablate (docs/ablate.md): the operands of a forward's cuts. The sidecar (AblateSidecarBinding, Model data read at
// load) gives the direction and each layer's two alphas; this holds the direction's device copy and each layer's two
// cuts, as a LoRA adapter's factors are a Qwen3.5 Parameters' (qwen3_5::execution::prepare_lora). The Parameters of a
// Model loaded without --ablate hold none.
struct AblateParameters {
    using Lane = artifact::AblateSidecar::Lane;

    DeviceBuffer direction;                              // FP32 [rows, H]: one shared row, or one per layer
    std::vector<std::array<ops::DirectionCut, 2>> cuts; // [layer][Lane]: inactive where the lane is uncut

    [[nodiscard]] ops::DirectionCut cut(std::uint32_t layer, Lane lane) const noexcept {
        return layer < cuts.size() ? cuts[layer][static_cast<std::size_t>(lane)] : ops::DirectionCut{};
    }
    // True when the sidecar cuts this layer's lane (the forward then forms the lane's output before the residual add).
    [[nodiscard]] bool cuts_lane(std::uint32_t layer, Lane lane) const noexcept { return cut(layer, lane).active(); }
};

// The device direction and cuts of `binding` (nullopt without one), checked against the model's geometry.
[[nodiscard]] std::optional<AblateParameters> prepare_ablate(const AblateSidecarBinding* binding, std::uint32_t layers,
                                                             std::uint32_t hidden_size);

} // namespace infernix::models
