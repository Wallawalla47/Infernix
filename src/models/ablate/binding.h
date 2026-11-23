#pragma once

#include "artifact/ablate_sidecar.h"
#include "artifact/framing.h"

#include <cstdint>
#include <filesystem>
#include <string>

namespace infernix::models {

// An ablation sidecar (`--ablate`, docs/ablate.md) read at load against a model's layer count and hidden width: Model
// data, as a LoRA adapter's binding is a Qwen3.5 Model's (qwen3_5::bind_lora). Every architecture's Model holds one
// when --ablate is given (Qwen3.5, Qwen3.5-MoE, Qwen3.8-Flash-Next); the direction's device copy and the per-layer cuts
// the forward reads are the Parameters' (models/ablate/parameters.h). A Model loaded without --ablate holds none, and
// nothing downstream changes.
struct AblateSidecarBinding {
    std::filesystem::path path; // sidecar.json
    artifact::AblateSidecar sidecar;

    // The startup log's description, after the path: "one shared direction, attn lane 26 layers (22-47), mlp lane
    // 40 layers (8-47); source=...".
    [[nodiscard]] std::string describe() const;
    // The startup log's warning when the sidecar names another artifact than `loaded` as the one it was fitted on (it
    // loads, as the geometry matches, but was not measured there); empty when it names none or this one.
    [[nodiscard]] std::string artifact_mismatch(const artifact::ArtifactId& loaded) const;
    // The device bytes the direction takes (prepare_ablate's copy): what a plan counts before anything is allocated.
    [[nodiscard]] std::uint64_t device_bytes() const noexcept {
        return static_cast<std::uint64_t>(sidecar.d_host.size()) * sizeof(float);
    }
};

// Reads `path` against the model's geometry (artifact::AblateSidecar::load's checks).
[[nodiscard]] AblateSidecarBinding bind_ablate(const std::filesystem::path& path, std::uint32_t layers,
                                               std::uint32_t hidden_size);

} // namespace infernix::models
