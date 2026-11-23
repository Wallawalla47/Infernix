#include "models/qwen4_exp/execution/ablate_parameters.h"

#include <stdexcept>

namespace infernix::models::qwen4_exp::execution {

std::optional<AblateParameters> prepare_ablate(const Model& model) {
    const AblateSidecarBinding* binding = model.ablate();
    if (binding == nullptr) { return std::nullopt; }
    const artifact::AblateSidecar& sidecar = binding->sidecar;
    const auto& text                       = model.config().text;
    if (sidecar.n_layers != text.num_hidden_layers || sidecar.hidden_size != text.hidden_size ||
        sidecar.d_host.size() != static_cast<std::size_t>(sidecar.rows()) * sidecar.hidden_size) {
        throw std::logic_error("Qwen4Exp --ablate: the sidecar does not match the model it was bound to");
    }
    AblateParameters out;
    out.direction = DeviceBuffer(sidecar.d_host.size() * sizeof(float));
    out.direction.copy_from_host(sidecar.d_host.data(), out.direction.bytes);
    const auto* base = static_cast<const float*>(out.direction.p);
    out.cuts.resize(sidecar.n_layers);
    for (std::uint32_t layer = 0; layer < sidecar.n_layers; ++layer) {
        for (const AblateParameters::Lane lane : {AblateParameters::Lane::Attn, AblateParameters::Lane::Mlp}) {
            const float alpha = sidecar.alpha_for(layer, lane);
            if (alpha != 0.0F) {
                out.cuts[layer][static_cast<std::size_t>(lane)] =
                    ops::DirectionCut{.direction = base + sidecar.d_row_offset(layer), .alpha = alpha};
            }
        }
    }
    return out;
}

} // namespace infernix::models::qwen4_exp::execution
