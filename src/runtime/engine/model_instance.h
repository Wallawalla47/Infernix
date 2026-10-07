#pragma once

#include "models/qwen3_5/model.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/runtime_types.h"
#include "models/registry.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/kv_capacity.h"

#include <memory>

namespace infernix::runtime {

// Validates the options and resolves every default; the hybrid prefix cache's Host tier, Device
// snapshot slots and tap budget depend on the model architecture.
// Everything the bytes of a persisted hybrid Host tier depend on besides its geometry (which the
// file records itself); a different value makes a saved file meaningless.
[[nodiscard]] std::string hybrid_cache_fingerprint(const EngineOptions& options, const std::string& signature);

[[nodiscard]] EngineOptions normalize_engine_options(
    EngineOptions options, models::Architecture architecture = models::Architecture::Qwen3_5);

struct ModelInstance {
    using ModelContract = models::qwen3_5::RuntimeTypes;

    std::unique_ptr<models::qwen3_5::Model> model;
    const models::qwen3_5::execution::Parameters parameters;
    models::qwen3_5::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<models::qwen3_5::Program> program;

    ModelInstance(std::unique_ptr<models::qwen3_5::Model> model, const EngineOptions& options);
    ~ModelInstance();
    ModelInstance(const ModelInstance&)            = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;
};

struct ConstructedModel {
    std::unique_ptr<ModelInstance> instance;
    LoadSummary load;
    ModelMetadata model_metadata;
    ContextMachineCostModel context_cost;
};

// The option checks that need no artifact (shared by every architecture, and the Qwen3.5-only
// pairing of --vram-headroom-mib with automatic KV capacity); the Engine runs them before it opens
// the artifact.
void validate_engine_options(const EngineOptions& options, models::Architecture architecture);

[[nodiscard]] ConstructedModel construct_model(EngineOptions& options, DeviceContext& device);

} // namespace infernix::runtime
