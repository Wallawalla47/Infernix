#include "models/qwen4_exp/program/vision_window.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ninfer::models::qwen4_exp {

double vision_seconds(std::span<const std::uint32_t> patches, std::uint32_t hidden) {
    double seconds = 0;
    for (const std::uint32_t p : patches) {
        const double P = p;
        seconds += kVisionGemmFlopsPerPatch * P / kVisionGemmFlops + 27.0 * 4.0 * P * P * hidden / kVisionAttentionFlops;
    }
    return seconds;
}

VisionWindowPlan plan_vision_window(std::span<const VisionWindowItem> items, std::uint32_t reused,
                                    std::size_t pass_bytes, std::size_t staging_bytes, std::size_t work_capacity,
                                    std::uint64_t frame_bytes, std::uint32_t out, std::uint32_t hidden,
                                    std::uint32_t depth) {
    if (frame_bytes == 0 || out == 0 || depth == 0) { throw std::invalid_argument("Vision window: bad geometry"); }
    VisionWindowPlan plan;
    std::vector<std::uint32_t> patches;
    for (std::uint32_t i = 0; i < items.size(); ++i) {
        const VisionWindowItem& item = items[i];
        if (item.token_end <= reused) { continue; }
        // A frontier strictly inside an item is never a resume point (the prefix cache drops them).
        if (item.token_begin < reused) { throw std::logic_error("Vision window: the reused prefix ends inside an item"); }
        plan.items.push_back(i);
        plan.encoded_tokens += item.merged;
        patches.push_back(item.patches);
    }
    if (plan.items.empty()) { return plan; }
    plan.handoff_bytes = 2ULL * out * plan.encoded_tokens;
    plan.window_bytes  = staging_bytes + pass_bytes;
    plan.seconds       = vision_seconds(patches, hidden);
    const auto frames_for = [&](std::size_t bytes) {
        return static_cast<std::uint32_t>((bytes + frame_bytes - 1) / frame_bytes);
    };
    if (plan.window_bytes <= work_capacity && plan.seconds <= kVisionStepSeconds) {
        plan.lease           = false;
        plan.frames          = frames_for(plan.handoff_bytes);
        plan.steps           = 1;
        plan.layers_per_step = depth;
    } else {
        plan.lease           = true;
        // The handoff first, then the window, each starting on a 256-byte boundary.
        plan.frames          = frames_for((plan.handoff_bytes + 255) / 256 * 256 + plan.window_bytes);
        plan.layers_per_step = std::max<std::uint32_t>(
            1, static_cast<std::uint32_t>(std::ceil(depth * kVisionStepSeconds / std::max(plan.seconds, 1e-9))));
        plan.layers_per_step = std::min(plan.layers_per_step, depth);
        // Stage 0 (embedding) and the merger ride with the first and last layer steps.
        plan.steps = (depth + plan.layers_per_step - 1) / plan.layers_per_step;
    }
    return plan;
}

VisualSlice visual_slice(std::span<const std::uint32_t> visual, std::uint32_t a, std::uint32_t b, std::uint32_t base,
                         std::span<std::int32_t> columns) {
    const auto lo = std::lower_bound(visual.begin(), visual.end(), a);
    const auto hi = std::lower_bound(lo, visual.end(), b);
    VisualSlice out{static_cast<std::uint32_t>(lo - visual.begin()), static_cast<std::uint32_t>(hi - lo)};
    if (columns.size() < out.count) { throw std::invalid_argument("Vision: too few column words"); }
    for (std::uint32_t i = 0; i < out.count; ++i) {
        columns[i] = static_cast<std::int32_t>(visual[out.first + i] - base);
    }
    return out;
}

} // namespace ninfer::models::qwen4_exp
