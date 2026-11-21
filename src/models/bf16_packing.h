#pragma once

// Load-time lossless packing of BF16 matrices into bf16_p12 (core/weight.h QuantLayout::P12Rows, ops/bf16_p12.h) for
// a family whose Ops read the packed form: the artifact objects marked transient at planning
// (artifact::Binder::mark_device_transient) are counted, one device block holds every packed plane (or a pinned host
// block, for objects gathered zero-copy), each object gets a packed WeightParent, and the caller points its bound views
// at them (repoint) before freeing the BF16 copies (MaterializedArtifact::release_transient). Shared by Qwen4Exp and
// Inkling 2.

#include "artifact/materializer.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "core/weight_view.h"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace infernix::models {

// The packed matrices: one device block of their planes and a parent per artifact object.
struct PackedWeights {
    DeviceBuffer storage;
    // Objects packed into pinned host memory (gathered zero-copy), and their bytes.
    std::unique_ptr<RegisteredHostBuffer> host_storage;
    std::uint64_t host_bytes = 0;
    std::vector<std::unique_ptr<WeightParent>> parents;
    std::unordered_map<const WeightParent*, const WeightParent*> replaced; // BF16 parent -> its packed parent
    std::uint64_t source_bytes = 0; // the BF16 bytes they replace
    std::uint64_t escapes      = 0;
    double seconds             = 0; // packing's wall time
    // Of which: counts, the block's allocation, packing, and (the caller's) freeing of the BF16 copies.
    double phase_seconds[4] = {};
};

// An object's packed device bytes for planning, its escapes bounded by one weight in 1,024.
[[nodiscard]] std::uint64_t packed_estimate(const WeightGeometry& geometry);

// Packs `objects` into device memory and `host_objects` into pinned host memory (through a device staging block).
// Every object must be a transient BF16 [n, k] matrix of the backing (k a multiple of 8).
[[nodiscard]] PackedWeights pack_bf16_objects(const artifact::MaterializedArtifact& backing,
                                              const std::vector<artifact::ObjectHandle>& objects,
                                              const std::vector<artifact::ObjectHandle>& host_objects,
                                              DeviceContext& device);

// Points the parts of `view` whose parent was packed at the packed parent.
void repoint(WeightView& view, const PackedWeights& packed);

} // namespace infernix::models
