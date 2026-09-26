#pragma once

#include "artifact/materializer.h"
#include "core/evictable_weight_pool.h"
#include "models/qwen3_5/weights.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5 {

namespace loading {
struct PendingWeight;
}

// One contiguous run of pinned bytes staged into a group's device slot. The pinned offset is
// the object's position in the pinned block; the staging offset is its position in the group's
// device slot. A group's objects are not necessarily adjacent in the pinned block (the binder
// packs pinned objects in registration order, and the converter stores the fused qkv separately
// from each layer's other weights), so a group is a list of segments rather than one range.
struct VisionOverlaySegment {
    std::size_t pinned_offset  = 0;
    std::size_t staging_offset = 0;
    std::size_t bytes          = 0;
};

struct VisionOverlayGroup {
    std::vector<VisionOverlaySegment> segments;
    std::size_t bytes = 0;  // total staged bytes (sum of segment bytes)
};

// Byte layout of the vision groups inside the pinned weight block, used by the overlay window
// to stage weights through borrowed device memory. slot_bytes is the largest single layer range.
struct VisionOverlayLayout {
    VisionOverlayGroup prelude;  // patch embedding .. position embedding
    std::vector<VisionOverlayGroup> layers;
    VisionOverlayGroup merger;   // merger fc1 .. merger norm bias
    std::size_t slot_bytes    = 0;
    std::size_t staging_bytes = 0;  // prelude + merger + two layer slots, aligned
};

// Runtime assets the overlay window needs, published on the model when the engine runs
// with vision offload on. The pool pointer is non-const because the window calls
// the pool's evict()/restore() during an overlay encode.
struct VisionOverlayAssets {
    EvictableWeightPool* pool               = nullptr;
    const std::byte* pinned_block           = nullptr;
    std::size_t pinned_bytes                = 0;
    std::size_t ladder_bytes                = 0;  // evictable tail a window may borrow
    VisionOverlayLayout layout;
};

// Resolve each vision group's contiguous byte range in the pinned block. Throws when a
// group's objects are not all pinned or are not packed contiguously (gap beyond alignment).
[[nodiscard]] VisionOverlayLayout compute_vision_overlay_layout(
    const ModelWeights& weights, const std::vector<loading::PendingWeight>& pending,
    const artifact::MaterializationPlan& plan);

} // namespace ninfer::models::qwen3_5
