#include "models/qwen4_exp/load/ablate.h"

#include <string>

namespace infernix::models::qwen4_exp {
namespace {

// "8-47" for a contiguous run of cut layers, "8-20, 30" otherwise; "none" when the lane is uncut.
std::string layer_ranges(const artifact::AblateSidecar& sidecar, const artifact::AblateSidecar::Lane lane) {
    std::string out;
    std::uint32_t l = 0;
    while (l < sidecar.n_layers) {
        if (sidecar.alpha_for(l, lane) == 0.0F) {
            ++l;
            continue;
        }
        const std::uint32_t first = l;
        while (l < sidecar.n_layers && sidecar.alpha_for(l, lane) != 0.0F) { ++l; }
        if (!out.empty()) { out += ", "; }
        out += std::to_string(first);
        if (l - 1 != first) { out += "-" + std::to_string(l - 1); }
    }
    return out.empty() ? "none" : out;
}

} // namespace

AblateSidecarBinding bind_ablate(const std::filesystem::path& path, const TextConfig& text) {
    return AblateSidecarBinding{.path    = path,
                                .sidecar = artifact::AblateSidecar::load(path, text.num_hidden_layers,
                                                                         text.hidden_size)};
}

std::string AblateSidecarBinding::describe() const {
    using Lane = artifact::AblateSidecar::Lane;
    std::string line = std::string(sidecar.per_layer_direction ? "a per-layer direction" : "one shared direction") +
                       ", attn lane " + std::to_string(sidecar.layers_cut(Lane::Attn)) + " layers (" +
                       layer_ranges(sidecar, Lane::Attn) + "), mlp lane " +
                       std::to_string(sidecar.layers_cut(Lane::Mlp)) + " layers (" + layer_ranges(sidecar, Lane::Mlp) +
                       ")";
    if (!sidecar.generation.empty()) { line += "; " + sidecar.generation; }
    return line;
}

} // namespace infernix::models::qwen4_exp
