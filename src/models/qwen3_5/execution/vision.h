#pragma once

#include "models/qwen3_5/program/internal.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/qwen3_5/program/vision_control.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/vision_prefill.h"
#include "models/qwen3_5/execution/vision_overlay.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace infernix::models::qwen3_5::execution {

using detail::VisionWorkspacePlan;
using detail::VisionPrefillPlan;
using detail::VisionUseSpan;
using detail::VisionHandoffState;

class VisionWeightStream;
struct PinnedVisionResult;
struct VisionOverlayWindowStats;

struct VisionItemView {
    std::span<const std::uint16_t> patches;
    const qwen3_5::VisionItemControl* control = nullptr;
};

class VisionContext {
public:
    VisionContext(DeviceContext& device, const execution::Parameters& parameters);
    // Overlay windows bind against a rebased per-window weight view.
    VisionContext(DeviceContext& device, const VisionConfig& config,
                  const VisionParameters& parameters);

    [[nodiscard]] static std::size_t workspace_bytes(const VisionConfig& config,
                                                     const VisionParameters& parameters,
                                                     std::size_t patches, std::size_t merged_tokens,
                                                     const VisionWorkspacePlan& plan);
    [[nodiscard]] static VisionWorkspacePlan plan_workspace(const VisionConfig& config,
                                                            const VisionParameters& parameters,
                                                            std::uint32_t max_merged_tokens,
                                                            std::size_t general_capacity_bytes);
    // Window plan for an overlay encode: the output handoff is placed after the encode
    // tensors (no general reservation), so the borrowed lease never aliases live
    // activations. max_patches/max_merged_tokens bound every item the window encodes.
    [[nodiscard]] static VisionWorkspacePlan plan_overlay_window(const VisionConfig& config,
                                                                 const VisionParameters& parameters,
                                                                 std::size_t max_patches,
                                                                 std::uint32_t max_merged_tokens);

    [[nodiscard]] const VisionConfig& config() const noexcept { return config_; }

    [[nodiscard]] static Tensor bind_output(DeviceSpan backing, const VisionWorkspacePlan& plan,
                                            std::size_t merged_tokens);
    // weight_stream, when set, gates each stage on the streamed uploads of an overlay window;
    // resident execution passes nullptr.
    void encode(const VisionItemView& item, Tensor& output, DeviceSpan backing,
                const VisionWorkspacePlan& plan, VisionWeightStream* weight_stream = nullptr) const;

private:
    DeviceContext& ctx_;
    const VisionConfig& config_;
    const VisionParameters& parameters_;
};

struct VisionChunk {
    std::int32_t length                       = 0;
    const qwen3_5::VisionItemControl* control = nullptr;
    Tensor embeddings;
};

class VisionPrefillSession {
public:
    VisionPrefillSession(DeviceContext& device, const execution::Parameters& parameters,
                         DeviceSpan workspace, const VisionWorkspacePlan& workspace_plan,
                         const qwen3_5::PreparedPromptData& prompt, const VisionPrefillPlan& plan,
                         VisionHandoffState& handoff, std::size_t& handoff_peak_bytes);
    ~VisionPrefillSession();

    VisionPrefillSession(const VisionPrefillSession&)            = delete;
    VisionPrefillSession& operator=(const VisionPrefillSession&) = delete;

    [[nodiscard]] VisionChunk prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length);
    void retire_handoff() noexcept;
    [[nodiscard]] double elapsed_seconds() const;

    [[nodiscard]] std::size_t active_handoff_bytes() const noexcept {
        return owns_handoff() ? active_handoff_bytes_ : 0;
    }
    // Install embeddings produced by an overlay window: prepare_chunk then uploads them
    // instead of encoding, and the media payload becomes releasable immediately.
    void set_preencoded(std::vector<PinnedVisionResult> results,
                        const VisionOverlayWindowStats& stats);
    [[nodiscard]] const std::optional<VisionOverlayWindowStats>& overlay_stats() const noexcept {
        return overlay_stats_;
    }

private:
    [[nodiscard]] bool owns_handoff() const noexcept {
        return handoff_.owner_ == this && handoff_.generation_ == active_generation_;
    }

    DeviceContext& device_;
    DeviceSpan workspace_;
    const VisionWorkspacePlan& workspace_plan_;
    const qwen3_5::PreparedPromptData& prompt_;
    const VisionPrefillPlan& plan_;
    VisionHandoffState& handoff_;
    std::size_t& handoff_peak_bytes_;
    VisionContext context_;
    std::optional<std::uint32_t> active_item_;
    std::uint64_t active_generation_  = 0;
    std::size_t active_handoff_bytes_ = 0;
    std::vector<CudaEventTimer> timers_;
    std::vector<PinnedVisionResult> preencoded_;
    std::optional<VisionOverlayWindowStats> overlay_stats_;
};

} // namespace infernix::models::qwen3_5::execution
