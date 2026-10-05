#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/load/vision_overlay.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/weights.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen4_exp {

// The frontend (tokenizer, chat template, output parsing) is the Qwen3.5 family's, unchanged
// (docs/maintainer/qwen3_8-flash-next-design.md §7).
using FrontendResources = qwen3_5::FrontendResources;

struct InstanceInfo {
    std::string name;
    std::string metadata_json;
    std::string provenance_json;
    artifact::ArtifactId artifact_id{};
};

// The routed experts of one MoE layer: the pinned host bank and each expert's scalars. alpha is
// derived once here from the stored words, alpha = fl32(weight_scale_2 * input_scale), so every
// route (GPU frame, GPU staging, CPU) reads the same values (design §6.1, §16.2).
struct ExpertBank {
    ExpertBankPlanes planes;
    std::vector<ops::offloaded_moe::ExpertScales> scales; // [experts]
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
    [[nodiscard]] const TextWeights& weights() const noexcept { return weights_; }
    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }
    [[nodiscard]] std::span<const BoundWeight> weight_data() const noexcept { return bound_; }
    [[nodiscard]] std::span<const ExpertBank> expert_banks() const noexcept { return banks_; }
    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }
    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }
    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }
    // With vision offload: the pinned tower and its staging layout (no evictable pool: the encode
    // window borrows device memory from the expert cache, design §19.3.2).
    [[nodiscard]] const std::optional<qwen3_5::VisionOverlayAssets>& overlay_vision() const noexcept {
        return overlay_vision_;
    }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    Model(Config config, LoadOptions options, TextWeights weights, std::vector<BoundWeight> bound,
          std::vector<ExpertBank> banks, FrontendResources resources, InstanceInfo info,
          std::optional<qwen3_5::VisionOverlayAssets> overlay_vision, artifact::MaterializedArtifact backing);

    // Borrowers are declared after the backing they view, so they are destroyed first.
    artifact::MaterializedArtifact backing_;
    Config config_;
    LoadOptions options_;
    TextWeights weights_;
    std::vector<BoundWeight> bound_;
    std::vector<ExpertBank> banks_;
    FrontendResources resources_;
    InstanceInfo info_;
    std::optional<qwen3_5::VisionOverlayAssets> overlay_vision_;
};

} // namespace ninfer::models::qwen4_exp
