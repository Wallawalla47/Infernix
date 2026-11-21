#include "models/q8_weights.h"

#include "core/device.h"
#include "infernix/ops/q8_g32.h"

#include <array>
#include <chrono>
#include <stdexcept>

namespace infernix::models {
namespace {

std::uint64_t align_256(std::uint64_t bytes) { return (bytes + 255) / 256 * 256; }

WeightGeometry q8_geometry(const WeightGeometry& source) {
    const std::array<std::uint64_t, 2> shape{source.shape[0], source.shape[1]};
    return weight_geometry(QType::Q8_G32_FP16, QuantLayout::RowSplit, shape);
}

} // namespace

std::uint64_t q8_bytes(const WeightGeometry& geometry) { return align_256(q8_geometry(geometry).bytes); }

Q8Weights quantize_q8_objects(const artifact::MaterializedArtifact& backing,
                              const std::vector<artifact::ObjectHandle>& objects, DeviceContext& device) {
    Q8Weights out;
    if (objects.empty()) { return out; }
    const auto start = std::chrono::steady_clock::now();
    struct Item {
        const WeightParent* source;
        WeightGeometry geometry;
        std::uint64_t offset;
    };
    std::vector<Item> items;
    std::uint64_t total = 0;
    for (const auto handle : objects) {
        const WeightParent& parent = backing.device_parent(handle);
        const WeightGeometry& g    = parent.geometry;
        if (g.format != QType::BF16 || g.layout != QuantLayout::Contiguous || g.shape.size() != 2 || g.shape[1] % 32 != 0 ||
            parent.device == nullptr) {
            throw std::invalid_argument("q8 weights: a device-resident contiguous BF16 [n, k] matrix (k a multiple of 32) "
                                        "is needed");
        }
        items.push_back({&parent, q8_geometry(g), total});
        total += align_256(items.back().geometry.bytes);
        out.source_bytes += g.bytes;
    }
    out.storage = DeviceBuffer(total);
    for (const auto& item : items) {
        const WeightGeometry& g = item.source->geometry;
        Weight source;
        source.qdata  = item.source->device;
        source.qtype  = QType::BF16;
        source.layout = QuantLayout::Contiguous;
        source.ndim   = 2;
        source.n = source.shape[0] = source.padded_shape[0] = static_cast<std::int32_t>(g.shape[0]);
        source.k = source.shape[1] = source.padded_shape[1] = static_cast<std::int32_t>(g.shape[1]);
        auto* at = static_cast<std::byte*>(out.storage.p) + item.offset;
        (void)ops::quantize_bf16_q8_g32(source, at, at + item.geometry.scale_offset, device.stream);
        auto parent      = std::make_unique<WeightParent>();
        parent->geometry = item.geometry;
        parent->data     = at;
        parent->device   = at;
        out.replaced.emplace(item.source, parent.get());
        out.parents.push_back(std::move(parent));
    }
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return out;
}

void repoint(WeightView& view, const Q8Weights& q8) {
    for (auto& part : view.parts) {
        if (const auto found = q8.replaced.find(part.parent); found != q8.replaced.end()) { part.parent = found->second; }
    }
}

} // namespace infernix::models
