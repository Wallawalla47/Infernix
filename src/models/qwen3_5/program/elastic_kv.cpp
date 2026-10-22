// Elastic text KV of a Program with offloaded experts (Qwen3.5-122B design §11.1; Qwen3.8-Flash-Next
// design §19.3.11): the text KV page planes live in a reserved virtual region whose pages
// [0, kv_base_pages) are backed at startup. A request that needs more pages than are free first gets
// what the prefix cache can evict (hybrid_reclaim), then the region grows by memory taken from the
// expert frames; between requests the free top goes back to the frames, and after kKvIdleShrink of
// idleness the cache's idle device blocks that have host copies are released first. Without it the
// whole --max-context x --max-concurrency KV would hold VRAM the expert cache could use.

#include "models/qwen3_5/program/program_impl.h"

#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace infernix::models::qwen3_5::detail {
namespace {

// The region's chunks that pages [0, pages) of every page-major plane need (a plane's page is its
// outermost dimension, so its first pages are a prefix of its storage).
std::vector<std::uint8_t> kv_chunks(const DeviceKVPagePoolLayout& layout, std::size_t region_bytes, std::size_t chunk,
                                    std::uint32_t pages) {
    std::vector<std::uint8_t> need((region_bytes + chunk - 1) / chunk, 0);
    const auto total = static_cast<std::size_t>(layout.spec.page_group_count);
    for (const auto& plane : layout.planes) {
        const std::size_t bytes = plane.storage.region.bytes / total * pages;
        if (bytes == 0) { continue; }
        const std::size_t first = plane.storage.region.offset / chunk;
        const std::size_t last  = (plane.storage.region.offset + bytes - 1) / chunk;
        for (std::size_t c = first; c <= last; ++c) { need[c] = 1; }
    }
    return need;
}

} // namespace

bool ProgramImpl::map_kv(std::uint32_t pages) {
    const auto need = kv_chunks(kv_layout, kv_vmm->reserved_bytes(), kKvChunkBytes, pages);
    for (std::size_t c = 0; c < need.size(); ++c) {
        if (need[c] == 0 || kv_vmm->mapped(c)) { continue; }
        if (!kv_vmm->map(c)) { return false; }
        // Nothing reads a page before it is written, but a zeroed chunk keeps stray reads finite.
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(kv_vmm->base()) + c * kKvChunkBytes, 0, kKvChunkBytes,
                                   device.stream));
    }
    return true;
}

bool ProgramImpl::grow_kv(std::uint32_t need) {
    DeviceKVPagePool& pool = decoder->text_kv.page_pool();
    if (!kv_vmm || !offload || pool.available_pages() >= need) { return pool.available_pages() >= need; }
    const std::uint32_t backed  = pool.backed_pages();
    const std::uint32_t missing = need - pool.available_pages();
    std::uint32_t target        = backed + missing;
    target = std::min(pool.capacity_pages(), (target + kKvGrowPages - 1U) / kKvGrowPages * kKvGrowPages);
    if (target - backed < missing) { return false; }
    const auto chunks = kv_chunks(kv_layout, kv_vmm->reserved_bytes(), kKvChunkBytes, target);
    std::uint64_t added = 0;
    for (std::size_t c = 0; c < chunks.size(); ++c) { added += chunks[c] != 0 && !kv_vmm->mapped(c) ? kKvChunkBytes : 0; }
    auto& engine               = offload->engine();
    const std::uint32_t before = engine.residency().frames();
    device.synchronize(); // the frames given up may hold experts an in-flight call reads
    if (added != 0 && !engine.take_frames(added, device.stream)) {
        // A prompt's stream ring or wide workspace holds frames until its last call. Between calls they
        // can go back: its next call lends again from the smaller cache (or stages its experts).
        offload->return_stream(device.stream);
        if (!engine.take_frames(added, device.stream)) { return false; }
    }
    const std::size_t mapped_before = kv_vmm->mapped_bytes();
    if (!map_kv(target)) {
        // The chunks mapped before the failure stay with the region (the next growth uses them); the
        // frames get back only what was not mapped.
        const std::size_t kept = kv_vmm->mapped_bytes() - mapped_before;
        if (added > kept) { engine.give_frames(added - kept, device.stream); }
        return false;
    }
    pool.set_backed_pages(target);
    kv_grown.store(true);
    kv_busy();
    engine.report("KV pool grew to " + std::to_string(target) + " pages (" + std::to_string(kv_vmm->mapped_bytes() >> 20) +
                  " MiB); expert cache " + std::to_string(before) + " -> " + std::to_string(engine.residency().frames()) +
                  " frames");
    return true;
}

void ProgramImpl::shrink_kv(bool evict) {
    if (!kv_vmm || !offload) { return; }
    DeviceKVPagePool& pool = decoder->text_kv.page_pool();
    if (pool.backed_pages() <= kv_base_pages) { return; }
    if (evict && hybrid_) {
        // The idle cache's device blocks that have host copies go first (a later match restores them),
        // so an ended conversation gives its memory back to the frames.
        runtime::prefix_cache::PrefixCacheIndex& index = hybrid_->index();
        for (std::uint32_t budget = pool.backed_pages() - kv_base_pages;
             budget != 0 && !pool.can_back(kv_base_pages) && index.evict_backed_device_blocks(1) != 0; --budget) {}
    }
    // Down in growth steps, the last one to the base (a pool grown to a capacity off the step grid
    // stops above the base otherwise).
    std::uint32_t target = pool.backed_pages();
    while (target > kv_base_pages) {
        const std::uint32_t next = std::max(kv_base_pages, target - std::min(target, kKvGrowPages));
        if (!pool.can_back(next)) { break; }
        target = next;
    }
    if (target == pool.backed_pages()) { return; }
    auto& engine               = offload->engine();
    const std::uint32_t before = engine.residency().frames();
    device.synchronize();
    const std::size_t mapped = kv_vmm->mapped_bytes();
    pool.set_backed_pages(target);
    const auto need = kv_chunks(kv_layout, kv_vmm->reserved_bytes(), kKvChunkBytes, target);
    for (std::size_t c = 0; c < need.size(); ++c) {
        if (need[c] == 0 && kv_vmm->mapped(c)) { kv_vmm->unmap(c); }
    }
    kv_grown.store(target > kv_base_pages);
    engine.give_frames(mapped - kv_vmm->mapped_bytes(), device.stream);
    engine.report("KV pool shrank to " + std::to_string(target) + " pages; expert cache " + std::to_string(before) +
                  " -> " + std::to_string(engine.residency().frames()) + " frames");
}

bool ProgramImpl::kv_idle_due() const noexcept {
    const auto idle = std::chrono::steady_clock::now().time_since_epoch().count() - kv_busy_ticks.load();
    return kv_grown.load() &&
           idle >= std::chrono::duration_cast<std::chrono::steady_clock::duration>(kKvIdleShrink).count();
}

std::uint32_t ProgramImpl::kv_growable_pages() const noexcept {
    if (!kv_vmm) { return 0; }
    const DeviceKVPagePool& pool = decoder->text_kv.page_pool();
    return pool.capacity_pages() - pool.backed_pages();
}

} // namespace infernix::models::qwen3_5::detail
