#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/load/vision_overlay.h"
#include "models/qwen3_5/execution/vision_weight_stream.h"
#include "models/qwen3_5/program/vision_prefill.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

// One vision item's merged embeddings, parked in pinned host memory by an overlay window and
// re-uploaded into the request transient when prefill consumes the item.
struct PinnedVisionResult {
    std::unique_ptr<PinnedHostBuffer> buffer = nullptr;  // null for prefix-reused items
    std::size_t bytes                        = 0;
};

struct VisionOverlayWindowStats {
    double window_seconds  = 0.0;
    double evict_seconds   = 0.0;
    double restore_seconds = 0.0;
    std::size_t evicted_bytes = 0;
    std::size_t staged_bytes  = 0;
};

// Encode the vision items a request will consume inside a single overlay window: evict the
// staging extent from the weight pool tail, stream the vision tower through it, land each
// item's merged embeddings in pinned host memory, and restore the evicted text weights before
// returning. `first_item` is an absolute prepared-item index; the returned vector covers
// every prepared item (null entries for prefix-reused items).
[[nodiscard]] std::vector<PinnedVisionResult>
encode_items_overlay(DeviceContext& device, const Parameters& parameters,
                     const qwen3_5::PreparedPromptData& prompt, const detail::VisionPrefillPlan& plan,
                     std::size_t first_item, VisionOverlayWindowStats* stats);

class VisionPrefillSession;

// Encode, in one overlay window, the prepared items a prefill consumes past its `reused` prompt
// tokens and install them on `session`. Items wholly inside the reused prefix are not encoded:
// their embeddings already live in the reused context. Item indices are absolute prepared-item
// indices, which start at the plan's prepared_item_begin for a prompt extending a prefix.
void encode_overlay_suffix(DeviceContext& device, const Parameters& parameters,
                           const qwen3_5::PreparedPromptData& prompt,
                           const detail::VisionPrefillPlan& plan, std::uint32_t reused,
                           VisionPrefillSession& session);

} // namespace ninfer::models::qwen3_5::execution
