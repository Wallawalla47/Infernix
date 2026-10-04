#pragma once

// The Qwen4Exp (Qwen3.8-Flash-Next) model instance behind the public Engine: its resident model,
// native parameters, the Qwen3.5 frontend it shares the tokenizer and chat template with, and its
// Program. The Engine drives it with the common EngineCore over the HybridResourceManager.

#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/model.h"
#include "models/qwen4_exp/program/program.h"
#include "ninfer/types.h"
#include "runtime/contract/resources.h"

#include <filesystem>
#include <memory>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::runtime {

struct Qwen4ExpInstance {
    using ModelContract = models::qwen4_exp::RuntimeTypes;

    std::unique_ptr<models::qwen4_exp::Model> model;
    const models::qwen4_exp::execution::Parameters parameters;
    models::qwen4_exp::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<models::qwen4_exp::Program> program;

    Qwen4ExpInstance(std::unique_ptr<models::qwen4_exp::Model> model, const EngineOptions& options);
    ~Qwen4ExpInstance();
    Qwen4ExpInstance(const Qwen4ExpInstance&)            = delete;
    Qwen4ExpInstance& operator=(const Qwen4ExpInstance&) = delete;
};

struct ConstructedQwen4Exp {
    std::unique_ptr<Qwen4ExpInstance> instance;
    LoadSummary load;
    ModelMetadata model_metadata;
    EngineOptions options;
};

// True when the artifact's text component is Qwen4ExpForCausalLM.
[[nodiscard]] bool artifact_is_qwen4_exp(const std::filesystem::path& artifact);

[[nodiscard]] ConstructedQwen4Exp construct_qwen4_exp(const EngineOptions& options, DeviceContext& device);

} // namespace ninfer::runtime
