#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "models/inkling2/config.h"
#include "models/inkling2/weights.h"
#include "text/tokenizer.h"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::models::inkling2 {

// The frontend's resources. Byte views borrow the model's Host backing; the tokenizer (Inkling pipeline) owns its
// tables.
struct FrontendResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view chat_template_jinja;
    std::string_view generation_config_json;
    std::shared_ptr<const text::Tokenizer> tokenizer;
    std::uint32_t public_token_count = 0;
};

struct InstanceInfo {
    std::string name;
    std::string metadata_json;
    std::string provenance_json;
    artifact::ArtifactId artifact_id{};
};

class LoadPlan;

// A loaded Inkling 2 model: every weight device-resident (design §20), frozen after materialization.
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
    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }
    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }
    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept { return backing_.stats(); }
    [[nodiscard]] std::uint64_t device_weight_bytes() const noexcept { return backing_.stats().device_capacity_bytes; }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&, const StartupObserver*);
    Model(Config config, LoadOptions options, TextWeights weights, std::vector<BoundWeight> bound,
          FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing);

    // Borrowers are declared after the backing they view, so they are destroyed first.
    artifact::MaterializedArtifact backing_;
    Config config_;
    LoadOptions options_;
    TextWeights weights_;
    std::vector<BoundWeight> bound_;
    FrontendResources resources_;
    InstanceInfo info_;
};

} // namespace infernix::models::inkling2
