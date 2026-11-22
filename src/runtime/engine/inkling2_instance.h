#pragma once

// The Inkling 2 (Inkling2LoopForCausalLM) model instance behind the public Engine: its resident model, native
// parameters, its own frontend and its Program. The Engine drives it with the common EngineCore.

#include "artifact/reader.h"
#include "infernix/types.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/frontend/frontend.h"
#include "models/inkling2/model.h"
#include "models/inkling2/program/program.h"
#include "runtime/engine/kv_capacity.h"

#include <memory>

namespace infernix {
struct DeviceContext;
}

namespace infernix::runtime {

struct Inkling2Instance {
    using ModelContract = models::inkling2::RuntimeTypes;

    std::unique_ptr<models::inkling2::Model> model;
    const models::inkling2::Parameters parameters;
    models::inkling2::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<models::inkling2::Program> program;

    Inkling2Instance(std::unique_ptr<models::inkling2::Model> model, const EngineOptions& options);
    ~Inkling2Instance();
    Inkling2Instance(const Inkling2Instance&)            = delete;
    Inkling2Instance& operator=(const Inkling2Instance&) = delete;
};

struct ConstructedInkling2 {
    std::unique_ptr<Inkling2Instance> instance;
    LoadSummary load;
    ModelMetadata model_metadata;
    EngineOptions options;
};

// True when the artifact's text component is Inkling2LoopForCausalLM.
[[nodiscard]] bool artifact_is_inkling2(const artifact::Reader& reader);
// True when the artifact bundles the DSpark drafter (its dspark component).
[[nodiscard]] bool artifact_has_inkling2_drafter(const artifact::Reader& reader);

// The options this slice runs (models/inkling2/program/program.h): the prefix cache, the n-gram archive, preemption and
// next-turn prefill off, the context limit's KV reserved for every lane; --spec auto drafts with the bundled DSpark
// drafter when `bundled_drafter` (generation, not --lossless); refuses what it cannot run.
[[nodiscard]] EngineOptions normalize_inkling2_options(EngineOptions options, bool bundled_drafter);

// With `plan` (Engine::plan) the construction stops before any weight is read: the device plan lands in *plan and the
// result is empty.
[[nodiscard]] ConstructedInkling2 construct_inkling2(const EngineOptions& options, DeviceContext& device,
                                                     EnginePlan* plan = nullptr);

} // namespace infernix::runtime
