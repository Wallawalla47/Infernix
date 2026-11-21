#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "core/host_memory.h"
#include "models/bf16_packing.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/load/vision_overlay.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/expert_offload.h"
#include "models/ablate/binding.h"
#include "models/qwen4_exp/weights.h"
#include "ops/offloaded_sparse_moe/cpu/ggml_expert.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace infernix::models::qwen4_exp {

// The frontend (tokenizer, chat template, output parsing) is the Qwen3.5 family's, unchanged
// (docs/maintainer/qwen3_8-flash-next-design.md §7).
using FrontendResources = qwen3_5::FrontendResources;

struct InstanceInfo {
    std::string name;
    std::string metadata_json;
    std::string provenance_json;
    artifact::ArtifactId artifact_id{};
    // The DFlash2 companion artifact's id (--draft-model), when one is loaded.
    std::optional<artifact::ArtifactId> draft_artifact_id;
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
    // With MoeConfig::shared_bank: each layer's shared expert, one device-resident record of its routed bank's layout
    // and arithmetic (empty otherwise).
    [[nodiscard]] std::span<const ExpertBank> shared_expert_banks() const noexcept { return shared_banks_; }
    // The pinned banks' CSF tables (runtime::expert_offload::compress_banks; empty: the records as stored).
    [[nodiscard]] const CsfTables& csf_tables() const noexcept { return csf_tables_; }
    // With LoadOptions::stream_experts: where each record lies in the artifact (the banks then
    // hold no records, only their layout and multipliers).
    [[nodiscard]] const std::optional<ExpertStore>& expert_store() const noexcept { return expert_store_; }
    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }
    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }
    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }
    [[nodiscard]] const PackedWeights& packed() const noexcept { return packed_; }
    // The device bytes the dense weights hold once loaded: the artifact's device arena, the packed block and the
    // DFlash2 companion's arena.
    [[nodiscard]] std::uint64_t device_weight_bytes() const noexcept {
        return backing_.stats().device_capacity_bytes + packed_.storage.bytes + draft_device_bytes();
    }
    // Of device_weight_bytes, the DFlash2 companion's arena.
    [[nodiscard]] std::uint64_t draft_device_bytes() const noexcept {
        return draft_backing_ ? draft_backing_->stats().device_capacity_bytes : 0;
    }
    // The weights' host memory: the artifact's and the companion's pinned and retained host objects (the expert banks'
    // records among them when every expert is loaded into RAM) and the packed objects kept in pinned host memory.
    [[nodiscard]] std::uint64_t host_weight_bytes() const noexcept {
        std::uint64_t bytes = backing_.stats().pinned_bytes + backing_.stats().retained_host_bytes + packed_.host_bytes;
        if (draft_backing_) { bytes += draft_backing_->stats().pinned_bytes + draft_backing_->stats().retained_host_bytes; }
        return bytes;
    }
    // What the load page-locked, for a caller that guards the load's device allocations by the process's shared
    // system memory use (SpillGuard::end), which page-locked memory raises too: host_weight_bytes (its retained
    // objects are pageable, so the figure errs high, which only costs the guard sensitivity), the CSF tables of the
    // compressed banks (one mapped pinned allocation; their records are compressed in place), and the upload staging
    // of the artifact and of a DFlash2 companion, released by now but perhaps not yet off the process's use.
    [[nodiscard]] std::uint64_t load_pinned_bytes() const noexcept {
        std::uint64_t bytes = host_weight_bytes() + backing_.stats().peak_staging_bytes;
        if (draft_backing_) { bytes += draft_backing_->stats().peak_staging_bytes; }
        for (const ExpertBank& bank : banks_) {
            if (bank.csf) {
                bytes += static_cast<std::uint64_t>(config_.text.moe.experts) *
                         ops::offloaded_moe::csf::Geometry::of(bank.shape()).table_words() * sizeof(std::uint32_t);
            }
        }
        return bytes;
    }
    // With vision offload: the pinned tower and its staging layout (no evictable pool: the encode
    // window borrows device memory from the expert cache, design §19.3.2).
    [[nodiscard]] const std::optional<qwen3_5::VisionOverlayAssets>& overlay_vision() const noexcept {
        return overlay_vision_;
    }
    // LoadOptions::ablate (--ablate, docs/ablate.md): the sidecar read at load; null without one.
    [[nodiscard]] const AblateSidecarBinding* ablate() const noexcept { return ablate_ ? &*ablate_ : nullptr; }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    Model(Config config, LoadOptions options, TextWeights weights, std::vector<BoundWeight> bound,
          std::vector<ExpertBank> banks, FrontendResources resources, InstanceInfo info,
          std::optional<qwen3_5::VisionOverlayAssets> overlay_vision, artifact::MaterializedArtifact backing,
          std::optional<ExpertStore> expert_store, std::optional<artifact::MaterializedArtifact> draft_backing,
          PackedWeights packed, CsfTables csf_tables, std::vector<ExpertBank> shared_banks,
          std::vector<std::vector<float>> shared_multipliers);

    // Borrowers are declared after the backing they view, so they are destroyed first.
    artifact::MaterializedArtifact backing_;
    std::optional<artifact::MaterializedArtifact> draft_backing_; // the DFlash2 companion's weights
    PackedWeights packed_;
    std::optional<ExpertStore> expert_store_; // the streamed banks' multipliers back banks_
    Config config_;
    LoadOptions options_;
    TextWeights weights_;
    std::vector<BoundWeight> bound_;
    std::vector<ExpertBank> banks_;
    CsfTables csf_tables_; // banks_' CSF tables
    std::vector<std::vector<float>> shared_multipliers_; // back shared_banks_' planes
    std::vector<ExpertBank> shared_banks_;
    FrontendResources resources_;
    InstanceInfo info_;
    std::optional<qwen3_5::VisionOverlayAssets> overlay_vision_;
    std::optional<AblateSidecarBinding> ablate_;
};

} // namespace infernix::models::qwen4_exp
