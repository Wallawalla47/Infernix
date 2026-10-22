#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/load/vision_overlay.h"
#include "models/qwen3_5/weights.h"
#include "infernix/ops/weight_input.h"
#include "runtime/expert_offload/expert_bank.h"
#include "runtime/expert_offload/expert_store.h"

#include <memory>
#include <optional>
#include <span>
#include <string>

namespace infernix::models::qwen3_5 {

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

    // The offloaded MoE layers (weights.h OffloadedMoeWeights), indexed by text layer: each one's
    // routed bank (pinned host memory, or with LoadOptions::stream_experts its layout over records
    // left in the artifact) and its shared expert's bank (device memory). Both are empty when no
    // layer offloads its experts.
    [[nodiscard]] std::span<const runtime::expert_offload::ExpertBank> expert_banks() const noexcept {
        return expert_banks_;
    }
    [[nodiscard]] std::span<const runtime::expert_offload::ExpertBank> shared_expert_banks() const noexcept {
        return shared_expert_banks_;
    }
    [[nodiscard]] bool offloaded_experts() const noexcept { return !expert_banks_.empty(); }
    // With LoadOptions::stream_experts: where each routed record lies in the artifact (the SSD tier
    // reads them in place); null otherwise.
    [[nodiscard]] const runtime::expert_offload::ExpertStore* expert_store() const noexcept {
        return expert_store_ ? &*expert_store_ : nullptr;
    }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    Model(Config config, LoadOptions options, ModelWeights weights, std::vector<BoundWeight> bound,
          FrontendResources resources, InstanceInfo info,
          std::optional<VisionOverlayAssets> overlay_vision,
          artifact::MaterializedArtifact backing,
          std::vector<runtime::expert_offload::ExpertBank> expert_banks,
          std::vector<runtime::expert_offload::ExpertBank> shared_expert_banks,
          std::vector<std::vector<float>> shared_expert_multipliers,
          std::optional<runtime::expert_offload::ExpertStore> expert_store);

    // Destroy all borrowers before backing. The caller keeps DeviceContext alive through cleanup.
    std::optional<VisionOverlayAssets> overlay_vision_;
    artifact::MaterializedArtifact backing_;
    Config config_;
    LoadOptions options_;
    ModelWeights weights_;
    std::vector<BoundWeight> bound_;
    FrontendResources resources_;
    InstanceInfo info_;
    std::vector<runtime::expert_offload::ExpertBank> expert_banks_;        // views into backing_
    std::vector<runtime::expert_offload::ExpertBank> shared_expert_banks_; // views into backing_
    // The shared banks' scale tails (their records are in device memory only; their planes point here).
    std::vector<std::vector<float>> shared_expert_multipliers_;
    // Owns the streamed banks' multipliers (their planes point into it).
    std::optional<runtime::expert_offload::ExpertStore> expert_store_;
};

} // namespace infernix::models::qwen3_5
