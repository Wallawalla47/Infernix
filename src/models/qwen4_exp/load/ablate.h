#pragma once

#include "artifact/ablate_sidecar.h"
#include "models/qwen4_exp/config.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace infernix::models::qwen4_exp {

// An ablation sidecar (`--ablate`, docs/ablate.md) bound to a Qwen3.8-Flash-Next text configuration: the sidecar read
// against the model's layer count and hidden width, and its direction's device copy. The forward asks it per layer
// and lane; an unbound (default) binding cuts nothing, and the forward hooks nothing.
class AblateBinding {
public:
    using Lane = artifact::AblateSidecar::Lane;

    bool enabled = false;
    std::filesystem::path path;
    artifact::AblateSidecar sidecar;

    [[nodiscard]] float alpha_for(const std::uint32_t layer, const Lane lane) const noexcept {
        return enabled ? sidecar.alpha_for(layer, lane) : 0.0F;
    }
    [[nodiscard]] bool layer_cut(const std::uint32_t layer) const noexcept {
        return enabled && sidecar.layer_cut(layer);
    }
    // The direction this layer cuts along on the device (nullptr before upload()).
    [[nodiscard]] const float* device_d(const std::uint32_t layer = 0) const noexcept {
        return d_device_ == nullptr ? nullptr : d_device_ + sidecar.d_row_offset(layer);
    }

    void upload(); // cudaMalloc + copy of the direction; throws on failure
    void release() noexcept;

    // A one-line description for the startup log.
    [[nodiscard]] std::string describe() const;

    AblateBinding() = default;
    AblateBinding(const AblateBinding&)            = delete;
    AblateBinding& operator=(const AblateBinding&) = delete;
    AblateBinding(AblateBinding&& other) noexcept;
    AblateBinding& operator=(AblateBinding&& other) noexcept;
    ~AblateBinding() { release(); }

private:
    float* d_device_ = nullptr; // device FP32, rows() x hidden_size; owned
};

// Reads `path` against `text`'s geometry (artifact::AblateSidecar::load's checks). Nothing on the device until
// upload(): the Program's planner binds the same sidecar to plan with and runs no kernels.
[[nodiscard]] AblateBinding bind_ablate(const std::filesystem::path& path, const TextConfig& text);

} // namespace infernix::models::qwen4_exp
