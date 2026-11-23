#include "models/ablate/binding.h"

#include <cstddef>
#include <string>

namespace infernix::models {
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

AblateSidecarBinding bind_ablate(const std::filesystem::path& path, std::uint32_t layers, std::uint32_t hidden_size) {
    return AblateSidecarBinding{.path = path, .sidecar = artifact::AblateSidecar::load(path, layers, hidden_size)};
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

std::string AblateSidecarBinding::artifact_mismatch(const artifact::ArtifactId& loaded) const {
    constexpr char kHex[] = "0123456789abcdef";
    std::string here;
    for (const std::byte byte : loaded) {
        here += kHex[std::to_integer<unsigned>(byte) >> 4U];
        here += kHex[std::to_integer<unsigned>(byte) & 15U];
    }
    if (sidecar.artifact_id.empty() || sidecar.artifact_id == here) { return {}; }
    return "the ablation sidecar was fitted on artifact " + sidecar.artifact_id + ", not this one (" + here +
           "): it loads, as the geometry matches, but its cut was not measured on this artifact";
}

} // namespace infernix::models
