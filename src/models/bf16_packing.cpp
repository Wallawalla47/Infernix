#include "models/bf16_packing.h"

#include "core/device.h"
#include "infernix/ops/bf16_p12.h"

#include <chrono>
#include <stdexcept>

namespace infernix::models {
namespace {

std::uint64_t align_256(std::uint64_t bytes) { return (bytes + 255) / 256 * 256; }

} // namespace

std::uint64_t packed_estimate(const WeightGeometry& g) {
    return align_256(bf16_p12_geometry(g.shape[0], g.shape[1], g.elements / 1024).bytes);
}

PackedWeights pack_bf16_objects(const artifact::MaterializedArtifact& backing,
                                const std::vector<artifact::ObjectHandle>& objects,
                                const std::vector<artifact::ObjectHandle>& host_objects, DeviceContext& device) {
    PackedWeights out;
    if (objects.empty() && host_objects.empty()) { return out; }
    const auto start = std::chrono::steady_clock::now();
    struct Item {
        const WeightParent* source;
        std::int32_t n, k;
        bool host;
        std::uint64_t scratch, escapes, offset;
    };
    std::vector<Item> items;
    std::uint64_t scratch_bytes = 0;
    for (const auto* list : {&objects, &host_objects}) {
        for (const auto handle : *list) {
            const WeightParent& parent = backing.device_parent(handle);
            const WeightGeometry& g    = parent.geometry;
            if (g.format != QType::BF16 || g.layout != QuantLayout::Contiguous || g.shape.size() != 2 || g.shape[1] % 8 != 0) {
                throw std::invalid_argument("bf16 packing: a contiguous BF16 [n, k] matrix (k a multiple of 8) is needed");
            }
            const auto n = static_cast<std::int32_t>(g.shape[0]);
            const auto k = static_cast<std::int32_t>(g.shape[1]);
            items.push_back({&parent, n, k, list == &host_objects, scratch_bytes, 0, 0});
            scratch_bytes += align_256(ops::bf16_p12_pack_scratch_bytes(n));
        }
    }
    auto lap = [last = start]() mutable {
        const auto now = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(now - last).count();
        last           = now;
        return s;
    };
    // Every count enqueued, then one read of all of them.
    DeviceBuffer scratch(scratch_bytes + 4 * items.size());
    auto* scratch_base = static_cast<std::byte*>(scratch.p);
    auto* counts       = reinterpret_cast<std::uint32_t*>(scratch_base + scratch_bytes);
    for (std::size_t i = 0; i < items.size(); ++i) {
        const auto& item = items[i];
        ops::bf16_p12_count(item.source->device, item.n, item.k, scratch_base + item.scratch, counts + i, device.stream);
    }
    std::vector<std::uint32_t> escapes(items.size());
    CUDA_CHECK(cudaMemcpyAsync(escapes.data(), counts, 4 * items.size(), cudaMemcpyDeviceToHost, device.stream));
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    std::uint64_t total = 0, host_total = 0;
    for (std::size_t i = 0; i < items.size(); ++i) {
        auto& item   = items[i];
        item.escapes = escapes[i];
        auto& end    = item.host ? host_total : total;
        item.offset  = end;
        end          = align_256(end + bf16_p12_geometry(item.n, item.k, item.escapes).bytes);
        out.source_bytes += item.source->geometry.bytes;
        out.escapes += item.escapes;
    }
    out.phase_seconds[0] = lap();
    out.storage          = DeviceBuffer(total);
    DeviceBuffer staging(host_total);
    if (host_total != 0) {
        out.host_storage = std::make_unique<RegisteredHostBuffer>(host_total);
        out.host_storage->pin();
        out.host_bytes = host_total;
    }
    out.phase_seconds[1] = lap();
    for (const auto& item : items) {
        auto* at = static_cast<std::byte*>(item.host ? staging.p : out.storage.p) + item.offset;
        ops::bf16_p12_pack(item.source->device, item.n, item.k, item.escapes, scratch_base + item.scratch, at,
                           device.stream);
        auto parent      = std::make_unique<WeightParent>();
        parent->geometry = bf16_p12_geometry(item.n, item.k, item.escapes);
        parent->data     = item.host ? out.host_storage->data() + item.offset : at;
        parent->device   = item.host ? out.host_storage->device_data() + item.offset : at;
        out.replaced.emplace(item.source, parent.get());
        out.parents.push_back(std::move(parent));
    }
    if (host_total != 0) {
        CUDA_CHECK(cudaMemcpyAsync(out.host_storage->data(), staging.p, host_total, cudaMemcpyDeviceToHost, device.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    out.phase_seconds[2] = lap();
    out.seconds          = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return out;
}

void repoint(WeightView& view, const PackedWeights& packed) {
    for (auto& part : view.parts) {
        if (const auto found = packed.replaced.find(part.parent); found != packed.replaced.end()) {
            part.parent = found->second;
        }
    }
}

} // namespace infernix::models
