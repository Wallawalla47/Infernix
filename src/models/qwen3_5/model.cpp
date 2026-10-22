#include "models/qwen3_5/model.h"

#include <stdexcept>
#include <utility>

namespace infernix::models::qwen3_5 {

Model::Model(Config config, LoadOptions options, ModelWeights weights,
             std::vector<BoundWeight> bound, FrontendResources resources, InstanceInfo info,
             std::optional<VisionOverlayAssets> overlay_vision,
             artifact::MaterializedArtifact backing,
             std::vector<runtime::expert_offload::ExpertBank> expert_banks,
             std::vector<runtime::expert_offload::ExpertBank> shared_expert_banks,
             std::vector<std::vector<float>> shared_expert_multipliers,
             std::optional<runtime::expert_offload::ExpertStore> expert_store)
    : overlay_vision_(std::move(overlay_vision)), backing_(std::move(backing)),
      config_(std::move(config)), options_(options), weights_(std::move(weights)),
      bound_(std::move(bound)), resources_(std::move(resources)), info_(std::move(info)),
      expert_banks_(std::move(expert_banks)), shared_expert_banks_(std::move(shared_expert_banks)),
      shared_expert_multipliers_(std::move(shared_expert_multipliers)),
      expert_store_(std::move(expert_store)) {}

Model::~Model() = default;

ops::WeightInput Model::input(WeightUseId id) const {
    const auto& parameter = weight(id.parameter);
    const auto& use       = parameter.uses.at(id.use_index);
    return {parameter.view, use.policy, use.activation_input_divisor, use.activation_input_scale};
}

ops::WeightInput Model::input(WeightId id) const {
    if (weight(id).uses.size() != 1) {
        throw std::invalid_argument("weight input requires an explicit mathematical use");
    }
    return input(WeightUseId{id, 0});
}

} // namespace infernix::models::qwen3_5
