#pragma once

#include "models/inkling2/model.h"

#include <filesystem>
#include <memory>

namespace infernix::artifact {
class Reader;
}

namespace infernix::models::inkling2 {

// Cold load plan; it borrows its Reader until materialization.
class LoadPlan {
public:
    ~LoadPlan();
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] const FrontendResources& resources() const;
    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;
    // Device bytes of the weights, and the tokenizer's public token count.
    [[nodiscard]] std::uint64_t device_bytes() const;
    [[nodiscard]] std::uint32_t public_token_count() const;
    // --vision-offload on: the tower's pinned groups (their block address set at materialization); otherwise empty.
    [[nodiscard]] const std::optional<VisionHostWeights>& vision_host() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit LoadPlan(std::unique_ptr<Impl> impl);
    friend LoadPlan plan_load(const artifact::Reader&, LoadOptions);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&, const StartupObserver*);
};

// True when the artifact's text component is Inkling2Loop.
[[nodiscard]] bool is_inkling2(const artifact::Reader& reader);

[[nodiscard]] LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options = {});
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                                       const StartupObserver* observer = nullptr);

} // namespace infernix::models::inkling2
