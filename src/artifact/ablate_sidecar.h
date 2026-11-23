#pragma once

// Ablation sidecar (`--ablate`, docs/ablate.md): one write direction d plus a per-layer, per-lane alpha table, read
// from sidecar.json and the float32 direction file beside it. A model that takes it cuts, at each hooked lane of
// layer L and per token column:
//     y -= alpha * dot(d, y_col) * d
// Two lanes are cut independently: `attn` is the attention (mixer) output before it is added to the residual, `mlp`
// the MLP output before it is added.
//
// This is the format reader and its validation, host only. A model binds a sidecar to its own geometry and owns the
// device copy (models/ablate/binding.h and parameters.h).
//
// FORMAT "ablate-direction-v2": a shared unit direction or a per-layer stack, alpha_attn_per_layer /
// alpha_mlp_per_layer (or alpha_per_layer for both lanes), a REQUIRED "generation" block saying where the direction
// came from (the engine logs it, so a running server names the cut it applies) and an optional "artifact_id" naming
// the artifact it was fitted on. Every other key is refused, so a misspelt table cannot load as an uncut lane.
//
// The alpha table is concrete on purpose. Parameterizations (bands, tapers, heretic-style shapes) belong in the
// fitter, which expands them into this table before it writes a sidecar; the engine stays a plain, deterministic
// applier.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace infernix::artifact {

class AblateSidecar {
public:
    // Which residual-write lane of a layer a cut applies to.
    enum class Lane : std::uint8_t { Attn = 0, Mlp = 1, kLanes = 2 };

    std::uint32_t hidden_size = 0;
    std::uint32_t n_layers    = 0;
    bool per_layer_direction  = false;
    std::string generation;  // provenance, flattened to one line for the startup log
    std::string artifact_id; // the artifact the direction was fitted on; empty when the sidecar names none

    std::vector<float> d_host; // shared: [H]; per-layer stack: [n_layers * H]; every row unit length or zero
    // [lane][layer]; layers absent from a table are 0 (no cut on that lane).
    std::array<std::vector<float>, 2> alpha;

    // Reads and validates `json_path` against a model's geometry. Any contract violation (missing file, an unknown
    // key, unsupported or missing format, dtype/dim/layout mismatch, a non-zero default_alpha, a non-finite or
    // out-of-range alpha, an out-of-range layer key, a degenerate direction on a cut layer, no `generation.source`)
    // throws std::runtime_error. Never a silent no-op: a flag pointing at a broken sidecar must fail the boot, not
    // serve uncut.
    [[nodiscard]] static AblateSidecar load(const std::filesystem::path& json_path, std::uint32_t n_layers,
                                            std::uint32_t hidden_size);
    // The direction file `json_path` names, beside it (for the saved prefix cache's fingerprint); empty when the JSON
    // cannot be read or names none. Refusing a broken sidecar is load()'s job.
    [[nodiscard]] static std::filesystem::path direction_path(const std::filesystem::path& json_path) noexcept;

    [[nodiscard]] float alpha_for(const std::uint32_t layer, const Lane lane) const noexcept {
        const auto& a = alpha[static_cast<std::size_t>(lane >= Lane::kLanes ? Lane::Mlp : lane)];
        return layer < static_cast<std::uint32_t>(a.size()) ? a[layer] : 0.0F;
    }
    // True when either lane cuts this layer.
    [[nodiscard]] bool layer_cut(const std::uint32_t layer) const noexcept {
        return alpha_for(layer, Lane::Attn) != 0.0F || alpha_for(layer, Lane::Mlp) != 0.0F;
    }
    // How many layers a lane cuts (for the startup log).
    [[nodiscard]] std::uint32_t layers_cut(Lane lane) const noexcept;
    // Direction rows: one shared vector, or one per layer.
    [[nodiscard]] std::uint32_t rows() const noexcept { return per_layer_direction ? n_layers : 1U; }
    // Row offset into the direction: one row per layer for a stack, always row 0 when shared. One expression serves
    // the host view and a model's device copy, so the two can never walk differently.
    [[nodiscard]] std::size_t d_row_offset(const std::uint32_t layer) const noexcept {
        return per_layer_direction ? static_cast<std::size_t>(layer) * hidden_size : 0U;
    }
    // The direction this layer cuts along, host side: the shared vector, or its row of the stack.
    [[nodiscard]] const float* host_d(const std::uint32_t layer = 0) const noexcept {
        return d_host.empty() ? nullptr : d_host.data() + d_row_offset(layer);
    }
};

inline const char* to_string(AblateSidecar::Lane lane) {
    return lane == AblateSidecar::Lane::Attn ? "attn" : "mlp";
}

} // namespace infernix::artifact
