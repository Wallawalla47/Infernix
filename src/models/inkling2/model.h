#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "models/ablate/binding.h"
#include "models/bf16_packing.h"
#include "models/q8_weights.h"
#include "models/inkling2/config.h"
#include "models/inkling2/draft_head.h"
#include "models/inkling2/lora.h"
#include "models/inkling2/weights.h"
#include "text/tokenizer.h"

#include <cstddef>
#include <memory>
#include <optional>
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
    std::optional<artifact::ArtifactId> draft_artifact_id; // the DFlash2 companion's (--draft-model)
};

class LoadPlan;

// --vision-offload on (design §15.5): the tower's weights in the pinned host block, grouped as the tower stages them:
// each group's objects in order, each at a 16-byte-aligned offset of its device region.
struct VisionStagingSegment {
    std::size_t pinned_offset  = 0; // in the pinned block
    std::size_t staging_offset = 0; // in the group's device region
    std::size_t bytes          = 0;
};
struct VisionStagingGroup {
    std::vector<VisionStagingSegment> segments;
    std::size_t bytes = 0;
};
struct VisionHostWeights {
    const std::byte* block = nullptr;        // the pinned block (host address: the linear operands' pointers lie inside it)
    const std::byte* block_device = nullptr; // its device-mapped address (the direct parameters' pointers lie inside it)
    VisionStagingGroup resident;          // patch embedding, position table, class row, pre-norm, projector
    std::vector<VisionStagingGroup> layers;
    std::size_t slot_bytes = 0;           // the largest layer group
};

// A loaded Inkling 2 model: every weight device-resident (design §20) but an offloaded vision tower's, frozen after
// materialization.
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
    [[nodiscard]] std::uint64_t device_weight_bytes() const noexcept {
        return backing_.stats().device_capacity_bytes + packed_.storage.bytes + q8_.storage.bytes +
               (draft_head_ ? draft_head_->storage.bytes : 0) +
               (draft_backing_ ? draft_backing_->stats().device_capacity_bytes : 0);
    }
    // --lm-head-draft: the drafter's shortlist tables, built at load.
    [[nodiscard]] const std::optional<DraftHead>& draft_head() const noexcept { return draft_head_; }
    // The matrices packed to bf16_p12 at load (LoadOptions::pack_bf16), which the bound views address.
    [[nodiscard]] const PackedWeights& packed() const noexcept { return packed_; }
    // The matrices converted to 8-bit weights at load (LoadOptions::q8_weights), which the bound views address.
    [[nodiscard]] const Q8Weights& q8() const noexcept { return q8_; }
    // --vision-offload on: the tower's pinned weights (the bound views address the pinned block); otherwise empty.
    [[nodiscard]] const std::optional<VisionHostWeights>& vision_host() const noexcept { return vision_host_; }
    // --lora / --ablate (docs/lora.md, docs/ablate.md): the adapter and the sidecar read at load; null without one.
    [[nodiscard]] const LoraBinding* lora() const noexcept { return lora_ ? &*lora_ : nullptr; }
    [[nodiscard]] const AblateSidecarBinding* ablate() const noexcept { return ablate_ ? &*ablate_ : nullptr; }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&, const StartupObserver*);
    Model(Config config, LoadOptions options, TextWeights weights, std::vector<BoundWeight> bound,
          FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing,
          std::optional<artifact::MaterializedArtifact> draft_backing, PackedWeights packed, Q8Weights q8,
          std::optional<DraftHead> draft_head, std::optional<VisionHostWeights> vision_host);

    // Borrowers are declared after the backing they view, so they are destroyed first.
    artifact::MaterializedArtifact backing_;
    std::optional<artifact::MaterializedArtifact> draft_backing_; // the DFlash2 companion's weights
    PackedWeights packed_;
    Q8Weights q8_;
    std::optional<DraftHead> draft_head_;
    Config config_;
    LoadOptions options_;
    TextWeights weights_;
    std::vector<BoundWeight> bound_;
    FrontendResources resources_;
    InstanceInfo info_;
    std::optional<VisionHostWeights> vision_host_;
    std::optional<LoraBinding> lora_;
    std::optional<AblateSidecarBinding> ablate_;
};

} // namespace infernix::models::inkling2
