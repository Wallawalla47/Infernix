#pragma once

// Load-time 8-bit weights (LOSSY, EngineOptions::q8_weights): BF16 matrices converted to Q8_G32_FP16 RowSplit
// (include/infernix/ops/q8_g32.h) for a family whose Ops read that form. As for bf16_packing.h: the artifact objects
// marked transient at planning (artifact::Binder::mark_device_transient) are converted into one device block, each
// object gets a Q8 WeightParent (row views of it address their rows' codes and scales), and the caller points its
// bound views at them (repoint) before freeing the BF16 copies (MaterializedArtifact::release_transient).

#include "artifact/materializer.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace infernix::models {

struct Q8Weights {
    DeviceBuffer storage;
    std::vector<std::unique_ptr<WeightParent>> parents;
    std::unordered_map<const WeightParent*, const WeightParent*> replaced; // BF16 parent -> its Q8 parent
    std::uint64_t source_bytes = 0; // the BF16 bytes they replace
    double seconds             = 0; // conversion's wall time (allocation and kernels)
};

// An object's Q8 device bytes for planning (exact: 256-aligned).
[[nodiscard]] std::uint64_t q8_bytes(const WeightGeometry& geometry);

// Converts `objects` (transient BF16 [n, k] device matrices of the backing, k a multiple of 32) into device memory.
[[nodiscard]] Q8Weights quantize_q8_objects(const artifact::MaterializedArtifact& backing,
                                           const std::vector<artifact::ObjectHandle>& objects, DeviceContext& device);

// Points the parts of `view` whose parent was converted at the Q8 parent.
void repoint(WeightView& view, const Q8Weights& q8);

} // namespace infernix::models
