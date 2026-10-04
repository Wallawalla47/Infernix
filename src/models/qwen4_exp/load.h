#pragma once

#include "models/qwen4_exp/model.h"

#include <filesystem>
#include <memory>

namespace ninfer::artifact {
class Reader;
}

namespace ninfer::models::qwen4_exp {

// Cold load plan; it borrows its Reader until materialization.
class LoadPlan {
public:
    ~LoadPlan();
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;
    // Pinned bytes of the expert banks and of the model's other pinned weights (token embedding).
    [[nodiscard]] std::uint64_t pinned_expert_bytes() const;
    [[nodiscard]] std::uint64_t pinned_other_bytes() const;
    // The Host memory the pinned block must leave free at materialization, and the pins the
    // caller makes after it (both checked before anything is read).
    void set_host_reserve(std::uint64_t reserve_bytes, std::uint64_t later_pinned_bytes);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit LoadPlan(std::unique_ptr<Impl> impl);
    friend LoadPlan plan_load(const artifact::Reader&, LoadOptions);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
};

// True when the artifact's text component is Qwen4Exp.
[[nodiscard]] bool is_qwen4_exp(const artifact::Reader& reader);

[[nodiscard]] LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options = {});
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                                       const StartupObserver* observer = nullptr);
[[nodiscard]] std::unique_ptr<Model> load_model(const std::filesystem::path& path,
                                                LoadOptions options, DeviceContext& device,
                                                const StartupObserver* observer = nullptr);

} // namespace ninfer::models::qwen4_exp
