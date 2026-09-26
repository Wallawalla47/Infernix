#include "models/qwen3_5/load/vision_overlay.h"

#include "models/qwen3_5/load/bindings.h"
#include "models/qwen3_5/load/vision_overlay.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5 {
namespace {

// Pinned byte placement (offset, size) for each object, from the materialization plan.
struct PinnedRangeIndex {
    std::unordered_map<std::size_t, std::pair<std::size_t, std::size_t>> by_object;

    explicit PinnedRangeIndex(const artifact::MaterializationPlan& plan) {
        by_object.reserve(plan.pinned_objects.size());
        for (const artifact::PinnedPlacement& placement : plan.pinned_objects) {
            by_object.emplace(placement.object.index,
                              std::make_pair(static_cast<std::size_t>(placement.offset),
                                             static_cast<std::size_t>(placement.bytes)));
        }
    }

    std::pair<std::size_t, std::size_t> lookup(artifact::ObjectHandle handle) const {
        const auto it = by_object.find(handle.index);
        if (it == by_object.end()) {
            throw std::logic_error("vision overlay group tensor is not pinned");
        }
        return it->second;
    }
};

constexpr std::size_t kStagingAlignment = 256;

std::size_t staging_align(std::size_t bytes) {
    return (bytes + kStagingAlignment - 1) / kStagingAlignment * kStagingAlignment;
}

std::vector<artifact::ObjectHandle>
objects_for(const std::vector<loading::PendingWeight>& pending, WeightId id) {
    std::vector<artifact::ObjectHandle> out;
    for (const auto& part : pending.at(id.index).reference.binding.parts) { out.push_back(part.object); }
    return out;
}

// Stage a group's distinct objects in the order listed, recording each object's pinned and
// staged offsets. Objects shared by several bindings (for example a fused qkv) are staged once.
VisionOverlayGroup build_group(const PinnedRangeIndex& index,
                               const std::vector<artifact::ObjectHandle>& handles) {
    VisionOverlayGroup group;
    std::vector<std::size_t> seen;
    seen.reserve(handles.size());
    std::size_t staging = 0;
    for (const artifact::ObjectHandle handle : handles) {
        if (std::find(seen.begin(), seen.end(), handle.index) != seen.end()) { continue; }
        seen.push_back(handle.index);
        const auto [pinned, bytes] = index.lookup(handle);
        group.segments.push_back(VisionOverlaySegment{pinned, staging, bytes});
        staging += bytes;
    }
    group.bytes = staging;
    return group;
}

} // namespace

VisionOverlayLayout compute_vision_overlay_layout(const ModelWeights& weights,
                                                  const std::vector<loading::PendingWeight>& pending,
                                                  const artifact::MaterializationPlan& plan) {
    if (!weights.vision) { throw std::logic_error("vision overlay layout requires vision weights"); }
    const VisionWeights& vision = *weights.vision;
    const PinnedRangeIndex index(plan);
    VisionOverlayLayout out;

    {
        std::vector<artifact::ObjectHandle> handles;
        for (const WeightId id :
             {vision.patch_embedding, vision.patch_embedding_bias, vision.position_embedding}) {
            auto part = objects_for(pending, id);
            handles.insert(handles.end(), part.begin(), part.end());
        }
        out.prelude = build_group(index, handles);
    }

    out.layers.reserve(vision.layers.size());
    for (std::size_t layer = 0; layer < vision.layers.size(); ++layer) {
        const VisionBlockWeights& source = vision.layers[layer];
        std::vector<artifact::ObjectHandle> handles;
        for (const WeightId id :
             {source.norm1.weight, source.norm1.bias, source.norm2.weight, source.norm2.bias,
              source.query, source.key, source.value, source.query_bias, source.key_bias,
              source.value_bias, source.output, source.output_bias, source.fc1,
              source.fc1_bias, source.fc2, source.fc2_bias}) {
            auto part = objects_for(pending, id);
            handles.insert(handles.end(), part.begin(), part.end());
        }
        VisionOverlayGroup group = build_group(index, handles);
        out.slot_bytes           = std::max(out.slot_bytes, group.bytes);
        out.layers.push_back(std::move(group));
    }

    {
        std::vector<artifact::ObjectHandle> handles;
        for (const WeightId id : {vision.merger_norm.weight, vision.merger_norm.bias,
                                  vision.merger_fc1, vision.merger_fc1_bias,
                                  vision.merger_fc2, vision.merger_fc2_bias}) {
            auto part = objects_for(pending, id);
            handles.insert(handles.end(), part.begin(), part.end());
        }
        out.merger = build_group(index, handles);
    }

    out.staging_bytes = staging_align(out.prelude.bytes) + staging_align(out.merger.bytes) +
                        2 * staging_align(out.slot_bytes);
    return out;
}

} // namespace ninfer::models::qwen3_5
