#pragma once

#include "artifact/ablate_sidecar.h"
#include "models/qwen4_exp/config.h"

#include <filesystem>
#include <string>

namespace infernix::models::qwen4_exp {

// An ablation sidecar (`--ablate`, docs/ablate.md) read at load against a Qwen3.8-Flash-Next text configuration: Model
// data, as a LoRA adapter's binding is a Qwen3.5 Model's (bind_lora). It holds the host direction and the alpha
// tables; the direction's device copy and the per-layer cuts the forward reads are the Parameters'
// (execution/ablate_parameters.h). A Model loaded without --ablate holds none, and nothing downstream changes.
struct AblateSidecarBinding {
    std::filesystem::path path; // sidecar.json
    artifact::AblateSidecar sidecar;

    // The startup log's description, after the path: "one shared direction, attn lane 26 layers (22-47), mlp lane
    // 40 layers (8-47); source=...".
    [[nodiscard]] std::string describe() const;
};

// Reads `path` against `text`'s layer count and hidden width (artifact::AblateSidecar::load's checks).
[[nodiscard]] AblateSidecarBinding bind_ablate(const std::filesystem::path& path, const TextConfig& text);

} // namespace infernix::models::qwen4_exp
