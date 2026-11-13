#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "core/host_memory.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/load/vision_overlay.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/expert_offload.h"
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

// The BF16 projections packed losslessly at load (LoadOptions::pack_bf16; core/weight.h bf16_p12): one device
// block of their planes and a parent per artifact object, which the bound views address in place of the
// artifact's BF16 parents (freed after packing).
struct PackedWeights {
    DeviceBuffer storage;
    // Objects packed into pinned host memory (the token embedding, gathered zero-copy), and their bytes.
    std::unique_ptr<RegisteredHostBuffer> host_storage;
    std::uint64_t host_bytes = 0;
    std::vector<std::unique_ptr<WeightParent>> parents;
    std::uint64_t source_bytes = 0; // the BF16 bytes they replace
    std::uint64_t escapes      = 0;
    double seconds             = 0; // packing's wall time
    // Of which: counts, the block's allocation, packing, freeing the BF16 copies. (The CUDA module load of the
    // process's first kernel launch, ~2.4 s, is a start's CUDA initialization: load_kernel_images.)
    double phase_seconds[4] = {};
};

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
          std::optional<qwen3_5::VisionOverlayAssets> overlay_vision, artifact::MaterializedArtifact backing,
          std::optional<ExpertStore> expert_store, std::optional<artifact::MaterializedArtifact> draft_backing,
          PackedWeights packed, CsfTables csf_tables);

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
    FrontendResources resources_;
    InstanceInfo info_;
    std::optional<qwen3_5::VisionOverlayAssets> overlay_vision_;
};

} // namespace infernix::models::qwen4_exp
