// Elastic text KV of Qwen4Exp (docs/maintainer/qwen3_8-flash-next-design.md §19.3.11): the KV page planes live in
// a reserved virtual region whose base pages are backed at startup; a request that needs more pages grows the
// region by memory taken from the expert frames, and between requests (after kKvIdleShrink of idleness, with
// the cache's idle device blocks that have Host copies released first) the free top goes back to the frames.

#include "models/qwen4_exp/program/program_impl.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>

namespace infernix::models::qwen4_exp::detail {

bool ProgramImpl::map_kv(std::uint32_t pages) {
    const auto need = kv_chunks(plan_, pages);
    for (std::size_t c = 0; c < need.size(); ++c) {
        if (!need[c] || kv_vmm_->mapped(c)) { continue; }
        if (!kv_vmm_->map(c)) { return false; }
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(kv_vmm_->base()) + c * kKvChunkBytes, 0, kKvChunkBytes,
                                   device_.stream));
    }
    return true;
}

bool ProgramImpl::grow_kv(std::uint32_t need) {
    if (!kv_vmm_ || pool_->available_pages() >= need) { return pool_->available_pages() >= need; }
    const std::uint32_t backed = pool_->backed_pages();
    std::uint32_t target       = backed + (need - pool_->available_pages());
    target = std::min(pool_->capacity_pages(), (target + kKvGrowPages - 1) / kKvGrowPages * kKvGrowPages);
    if (target - backed < need - pool_->available_pages()) { return false; }
    const std::lock_guard<std::mutex> lock(vram_mutex_);
    // The bytes map_kv(target) will map: the chunks the target needs that are not mapped yet. Chunks an earlier
    // failed growth kept (below) can sit in some planes while another plane still lacks one, so the difference of
    // the two totals is not that amount (and wrapped below zero when a smaller target followed).
    std::size_t added = 0;
    {
        const auto chunks = kv_chunks(plan_, target);
        for (std::size_t c = 0; c < chunks.size(); ++c) {
            if (chunks[c] != 0 && !kv_vmm_->mapped(c)) { added += kKvChunkBytes; }
        }
    }
    const std::uint64_t stride = residency_->frame_bytes();
    const std::uint32_t frames = residency_->frames();
    // Memory the sizing function would give the frames anyway comes first; the rest is taken from
    // the frames, one chunk per pool over so the pools unmap at least that much.
    const std::uint32_t spare =
        control_ ? std::max<std::uint32_t>(control_->target(vram_->query(), pool_now_), frames) - frames : 0;
    const std::uint64_t spare_bytes = static_cast<std::uint64_t>(spare) * stride;
    std::uint32_t give              = 0;
    if (added > spare_bytes) {
        give = static_cast<std::uint32_t>((added - spare_bytes + residency_->pools() * ExpertResidency::kChunkBytes + stride - 1) / stride);
    }
    if (give > frames) { return false; }
    device_.synchronize();
    if (give != 0) {
        const auto resized = residency_->resize(frames - give, device_.stream, *vram_);
        if (resized.frames > frames - give) { // a lease holds the frames (it may have shrunk part-way)
            frames_now_ = residency_->frames();
            pool_now_   = residency_->pool_bytes();
            return false;
        }
    }
    const std::size_t mapped_before = kv_vmm_->mapped_bytes();
    if (!map_kv(target)) {
        // The chunks mapped before the failure stay with the region (the next growth uses them) and count as
        // fixed bytes; the frames get back only what was not mapped (as Qwen3.5's elastic_kv.cpp), so frames and
        // KV never claim the same memory.
        const std::size_t kept = kv_vmm_->mapped_bytes() - mapped_before;
        if (control_ && kept != 0) { control_->account_fixed(static_cast<std::int64_t>(kept)); }
        const auto held = static_cast<std::uint32_t>(std::min<std::uint64_t>(give, (kept + stride - 1) / stride));
        if (give > held) { (void)residency_->resize(frames - held, device_.stream, *vram_, growth_limit()); }
        frames_now_ = residency_->frames();
        pool_now_   = residency_->pool_bytes();
        return false;
    }
    pool_->set_backed_pages(target);
    kv_grown_.store(true);
    if (control_) { control_->account_fixed(static_cast<std::int64_t>(added)); }
    frames_now_ = residency_->frames();
    pool_now_   = residency_->pool_bytes();
    diagnostic("KV pool grew to " + std::to_string(target) + " pages (" +
               std::to_string(kv_vmm_->mapped_bytes() >> 20) + " MiB); expert cache " + std::to_string(frames) +
               " -> " + std::to_string(residency_->frames()) + " frames", DiagnosticLevel::Debug);
    return true;
}

void ProgramImpl::shrink_kv(bool evict) {
    if (!kv_vmm_ || pool_->backed_pages() <= plan_.kv_base_pages) { return; }
    if (evict && prefix_) {
        // The Host writes a lane's release started pin their blocks until poll sees them land, and nothing polls
        // while the engine idles: without this the blocks of the last conversation stayed pinned, the shrink
        // released nothing and the pool kept them (with the frames they took) until the next admission.
        prefix_->poll();
        runtime::prefix_cache::PrefixCacheIndex& index = prefix_->index();
        std::uint32_t evicted = 0;
        for (std::uint32_t budget = pool_->backed_pages() - plan_.kv_base_pages;
             budget != 0 && !pool_->can_back(plan_.kv_base_pages) && index.evict_backed_device_blocks(1) != 0; --budget) {
            ++evicted;
        }
        diagnostic("idle KV shrink: " + std::to_string(evicted) + " cached blocks with Host copies released; " +
                       std::to_string(pool_->allocated_pages()) + " of " + std::to_string(pool_->backed_pages()) +
                       " backed pages in use, " + std::to_string(index.device_evictable_blocks()) +
                       " of them cached blocks without Host copies",
                   DiagnosticLevel::Debug);
    }
    // Down in growth steps, the last one to the base (Qwen3.5's elastic_kv.cpp: a pool grown to a capacity
    // off the step grid stopped above the base when a page below the base was in use, and then woke the
    // engine every kKvIdleShrink for a shrink that did nothing).
    std::uint32_t target = pool_->backed_pages();
    while (target > plan_.kv_base_pages) {
        const std::uint32_t next = std::max(plan_.kv_base_pages, target - std::min(target, kKvGrowPages));
        if (!pool_->can_back(next)) { break; }
        target = next;
    }
    // A release's shrink keeps one growth step above what is in use: the conversation's next turn takes those
    // pages again at once, and giving them back meant unmapping them, a device synchronization and a growth
    // that evicted the experts loaded into them, every turn. The step costs the frames of kKvGrowPages pages
    // until the idle shrink, which goes to the lowest step (and evicts first), returns it after kKvIdleShrink.
    if (!evict) { target = std::min(pool_->backed_pages(), target + kKvGrowPages); }
    if (target == pool_->backed_pages()) { return; }
    const std::lock_guard<std::mutex> lock(vram_mutex_);
    device_.synchronize();
    const std::size_t before = kv_vmm_->mapped_bytes();
    pool_->set_backed_pages(target);
    const auto need = kv_chunks(plan_, target);
    for (std::size_t c = 0; c < need.size(); ++c) {
        if (!need[c] && kv_vmm_->mapped(c)) { kv_vmm_->unmap(c); }
    }
    kv_grown_.store(target > plan_.kv_base_pages);
    const std::size_t freed   = before - kv_vmm_->mapped_bytes();
    const std::uint64_t stride = residency_->frame_bytes();
    const std::uint32_t frames = residency_->frames();
    (void)residency_->resize(frames + static_cast<std::uint32_t>(freed / stride), device_.stream, *vram_,
                             growth_limit());
    if (control_) { control_->account_fixed(-static_cast<std::int64_t>(freed)); }
    frames_now_ = residency_->frames();
    pool_now_   = residency_->pool_bytes();
    diagnostic("KV pool shrank to " + std::to_string(target) + " pages; expert cache " + std::to_string(frames) +
               " -> " + std::to_string(residency_->frames()) + " frames", DiagnosticLevel::Debug);
}

} // namespace infernix::models::qwen4_exp::detail
