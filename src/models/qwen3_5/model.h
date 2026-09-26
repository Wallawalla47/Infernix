#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/load/vision_overlay.h"
#include "models/qwen3_5/weights.h"
#include "ninfer/ops/weight_input.h"

#include <memory>
#include <optional>
#include <span>
#include <string>

namespace ninfer::models::qwen3_5 {

struct InstanceInfo {
    std::string name;
    std::string metadata_json;
    std::string provenance_json;
    artifact::ArtifactId artifact_id{};
};

class LoadPlan;

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&)                 = delete;
    Model& operator=(Model&&)      = delete;

    [[nodiscard]] const Config& config() const noexcept { return config_; }

    [[nodiscard]] const LoadOptions& options() const noexcept { return options_; }

    [[nodiscard]] const ModelWeights& weights() const noexcept { return weights_; }

    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }

    [[nodiscard]] ops::WeightInput input(WeightUseId id) const;
    [[nodiscard]] ops::WeightInput input(WeightId id) const;

    [[nodiscard]] std::span<const BoundWeight> weight_data() const noexcept { return bound_; }

    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }

    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

    // Overlay vision assets (evictable pool + pinned block + layout); present only when the
    // model was loaded with vision offload on.
    [[nodiscard]] const std::optional<VisionOverlayAssets>& overlay_vision() const noexcept {
        return overlay_vision_;
    }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    Model(Config config, LoadOptions options, ModelWeights weights, std::vector<BoundWeight> bound,
          FrontendResources resources, InstanceInfo info,
          std::optional<VisionOverlayAssets> overlay_vision,
          artifact::MaterializedArtifact backing);

    // Destroy all borrowers before backing. The caller keeps DeviceContext alive through cleanup.
    std::optional<VisionOverlayAssets> overlay_vision_;
    artifact::MaterializedArtifact backing_;
    Config config_;
    LoadOptions options_;
    ModelWeights weights_;
    std::vector<BoundWeight> bound_;
    FrontendResources resources_;
    InstanceInfo info_;
};

} // namespace ninfer::models::qwen3_5
