#pragma once

#include "models/qwen3_5/model.h"

#include <filesystem>
#include <memory>
#include <span>

namespace infernix::artifact {
class Reader;
struct ParameterReference;
} // namespace infernix::artifact

namespace infernix::models::qwen3_5 {

// Cold load plan borrows its Reader until materialization. Its selected Host bytes and parsed
// resources already have owners; no Program or device allocation is needed for plan_load.
class LoadPlan {
public:
    ~LoadPlan();
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] const ModelWeights& weights() const;
    [[nodiscard]] const FrontendResources& resources() const;
    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;
    // The Host memory the pinned block (an offloaded vision tower, offloaded experts) must leave
    // free, and the pinned bytes Infernix's later allocations need (the RAM ledger's).
    void set_host_reserve(std::uint64_t reserve_bytes, std::uint64_t later_pinned_bytes = 0);
    // An artifact with offloaded experts: the RAM ledger's pinned terms (design §11.2), the routed
    // banks' (zero with LoadOptions::stream_experts) and everything else pinned (the token
    // embedding and, with vision offload, the tower).
    [[nodiscard]] bool offloaded_experts() const;
    [[nodiscard]] std::uint64_t pinned_expert_bytes() const;
    [[nodiscard]] std::uint64_t pinned_other_bytes() const;
    [[nodiscard]] const std::optional<VisionOverlayLayout>& vision_overlay_layout() const;
    [[nodiscard]] const artifact::ParameterReference& parameter(WeightId id) const;
    [[nodiscard]] std::span<const WeightUse> uses(WeightId id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit LoadPlan(std::unique_ptr<Impl> impl);
    friend LoadPlan plan_load(const artifact::Reader&, LoadOptions);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
};

[[nodiscard]] LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options = {});
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                                       const StartupObserver* observer = nullptr);
[[nodiscard]] std::unique_ptr<Model> load_model(const std::filesystem::path& path,
                                                LoadOptions options, DeviceContext& device,
                                                const StartupObserver* observer = nullptr);

} // namespace infernix::models::qwen3_5
