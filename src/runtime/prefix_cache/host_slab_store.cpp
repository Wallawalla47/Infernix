#include "runtime/prefix_cache/host_slab_store.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace infernix::runtime::prefix_cache {

HostSlabStore::HostSlabStore(std::size_t slab_bytes, std::uint32_t slabs, std::size_t chunk_bytes, std::string_view tier,
                             const std::function<void(std::uint64_t, std::uint64_t)>& progress)
    : slab_bytes_(slab_bytes), slabs_(slabs),
      per_chunk_(static_cast<std::uint32_t>(std::max<std::size_t>(1U, slab_bytes == 0 ? 1U : chunk_bytes / slab_bytes))) {
    if (slabs == 0) { return; }
    if (slab_bytes == 0 || slab_bytes % 4096U != 0) {
        throw std::invalid_argument("prefix cache Host slabs must be a positive multiple of 4 KiB");
    }
    const std::uint64_t total = capacity_bytes();
    const std::uint32_t count = (slabs + per_chunk_ - 1U) / per_chunk_;
    std::uint64_t pinned      = 0;
    const auto failure        = [&](const std::exception& error) {
        return std::runtime_error("pinning the " + std::string(tier) + " failed after " + std::to_string(pinned >> 20U) +
                                  " of " + std::to_string(total >> 20U) + " MiB (" + error.what() +
                                  "); lower --host-context-mib");
    };
    chunks_.reserve(count);
    // Chunks in batches: committed, the pages of each touched by a thread of its own (the OS zeroes a page at its first
    // touch, the bulk of the time, which cudaMallocHost spent on one thread in its serial call), then registered one
    // after another.
    const std::uint32_t threads = std::clamp(std::thread::hardware_concurrency() / 2U, 1U, 8U);
    for (std::uint32_t first = 0; first < count; first += threads) {
        const std::uint32_t last = std::min(count, first + threads);
        std::vector<Chunk> batch;
        try {
            for (std::uint32_t c = first; c < last; ++c) {
                const std::uint32_t begin = c * per_chunk_;
                batch.push_back(std::make_unique<RegisteredHostBuffer>(
                    static_cast<std::size_t>(std::min(per_chunk_, slabs - begin)) * slab_bytes));
            }
            std::vector<std::jthread> touch; // joined when it goes, a throw included
            for (std::size_t i = 1; i < batch.size(); ++i) {
                touch.emplace_back([chunk = batch[i].get()] { chunk->prefault(0, chunk->size()); });
            }
            batch.front()->prefault(0, batch.front()->size());
        } catch (const std::exception& error) { throw failure(error); }
        for (Chunk& chunk : batch) {
            try {
                chunk->pin();
            } catch (const std::exception& error) { throw failure(error); }
            pinned += chunk->size();
            chunks_.push_back(std::move(chunk));
            if (progress) { progress(pinned, total); }
        }
    }
}

HostSlabStore::~HostSlabStore() {
    // A copy writes into the chunks; a growth's chunks go with its result, a release's with its job.
    if (copy_.valid()) { copy_.wait(); }
    if (growth_.valid()) { growth_.wait(); }
    if (freeing_.valid()) { freeing_.wait(); }
}

HostSlabStore::Chunk HostSlabStore::pin_chunk(std::size_t bytes) {
    auto chunk = std::make_unique<RegisteredHostBuffer>(bytes);
    chunk->prefault(0, bytes);
    chunk->pin();
    return chunk;
}

std::uint64_t HostSlabStore::pinned_bytes() const noexcept {
    std::uint64_t bytes = 0;
    for (const Chunk& chunk : chunks_) { bytes += chunk->size(); }
    return bytes;
}

std::uint32_t HostSlabStore::pinned_slabs() const noexcept {
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(slabs_, static_cast<std::uint64_t>(chunks_.size()) * per_chunk_));
}

std::uint64_t HostSlabStore::shrinkable_bytes() const noexcept {
    if (resizing()) { return 0; }
    std::uint64_t bytes = 0;
    for (std::size_t i = floor_chunks_; i < chunks_.size(); ++i) { bytes += chunks_[i]->size(); }
    return bytes;
}

std::uint64_t HostSlabStore::growable_bytes() const noexcept {
    return resizing() ? 0 : capacity_bytes() - pinned_bytes();
}

std::uint64_t HostSlabStore::begin_release(PrefixCacheIndex& index, std::uint64_t bytes, HostMemoryWorker& worker) {
    const auto have = static_cast<std::uint32_t>(chunks_.size());
    if (resizing() || bytes == 0 || have <= floor_chunks_) { return 0; }
    const std::uint64_t chunk = static_cast<std::uint64_t>(per_chunk_) * slab_bytes_;
    const auto drop           = static_cast<std::uint32_t>(std::min<std::uint64_t>(have, (bytes + chunk - 1U) / chunk));
    const std::uint32_t keep  = std::max(floor_chunks_, have - drop);
    if (keep >= have) { return 0; }
    const std::uint32_t ceiling = index.begin_host_shrink(keep * per_chunk_, per_chunk_);
    if (ceiling >= index.host_slab_limit()) { return 0; } // held entries keep every chunk in use now
    releasing_       = true;
    std::uint64_t aim = 0;
    for (std::size_t i = (ceiling + per_chunk_ - 1U) / per_chunk_; i < chunks_.size(); ++i) { aim += chunks_[i]->size(); }
    copy_moves(index, worker);
    return aim;
}

std::uint64_t HostSlabStore::begin_growth(std::uint64_t bytes, HostMemoryWorker& worker) {
    if (resizing()) { return 0; }
    std::vector<std::size_t> sizes;
    std::uint64_t total = 0;
    for (std::uint64_t first = static_cast<std::uint64_t>(chunks_.size()) * per_chunk_; first < slabs_; first += per_chunk_) {
        const auto size = static_cast<std::size_t>(std::min<std::uint64_t>(per_chunk_, slabs_ - first) * slab_bytes_);
        if (total + size > bytes) { break; }
        sizes.push_back(size);
        total += size;
    }
    if (sizes.empty()) { return 0; }
    growth_ = worker.post([sizes = std::move(sizes)] {
        Grown out;
        for (const std::size_t size : sizes) {
            try {
                out.chunks.push_back(pin_chunk(size));
            } catch (const std::exception&) {
                out.refused = true; // the OS refuses to page-lock more now
                break;
            }
        }
        return out;
    });
    return total;
}

void HostSlabStore::copy_moves(PrefixCacheIndex& index, HostMemoryWorker& worker) {
    const std::vector<PrefixCacheIndex::HostMove> moves = index.plan_host_moves();
    if (moves.empty()) { return; }
    std::vector<std::pair<std::byte*, const std::byte*>> copies;
    copies.reserve(moves.size());
    for (const PrefixCacheIndex::HostMove& move : moves) { copies.emplace_back(slab(move.to), slab(move.from)); }
    copy_ = worker.post([copies = std::move(copies), bytes = slab_bytes_] {
        for (const auto& [to, from] : copies) { copy_streaming(to, from, bytes); }
    });
}

HostSlabStore::Progress HostSlabStore::step(PrefixCacheIndex& index, bool restores_quiet, HostMemoryWorker& worker) {
    Progress out;
    if (ready(growth_)) {
        Grown grown = growth_.get();
        for (Chunk& chunk : grown.chunks) {
            out.grown += chunk->size();
            chunks_.push_back(std::move(chunk));
        }
        index.grow_host(pinned_slabs());
        out.refused     = grown.refused;
        out.growth_done = true;
    }
    if (freeing_.valid()) {
        // The release ends once the worker has freed the chunks, so the next reading of available memory counts them.
        if (!ready(freeing_)) { return out; }
        freeing_.get();
        out.released     = std::exchange(freeing_bytes_, 0);
        out.release_done = true;
        return out;
    }
    if (!releasing_) { return out; }
    if (copy_.valid()) {
        if (!ready(copy_)) { return out; }
        copy_.get();
    }
    if (index.host_moves_pending()) {
        if (!restores_quiet) { // a restore may still read a slab being moved
            out.waiting = true;
            return out;
        }
        (void)index.commit_host_moves(); // I/O-pinned entries keep theirs for a later step
    }
    if (index.finish_host_shrink()) {
        const std::size_t kept = (index.host_slab_limit() + per_chunk_ - 1U) / per_chunk_;
        std::vector<Chunk> gone;
        while (chunks_.size() > kept) {
            freeing_bytes_ += chunks_.back()->size();
            gone.push_back(std::move(chunks_.back()));
            chunks_.pop_back();
        }
        // Nothing reads them any more: every restore that could has landed and the index names no slab in them.
        releasing_ = false;
        if (gone.empty()) {
            out.release_done = true;
            return out;
        }
        // One chunk at a time with a pause between: each unregistration holds the driver (about 1.7 ms for 64 MiB,
        // during which the round thread's launches wait), so back-to-back calls would add up inside one round.
        freeing_ = worker.post([gone = std::move(gone)]() mutable {
            for (std::size_t i = 0; i < gone.size(); ++i) {
                if (i != 0) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
                gone[i].reset();
            }
        });
        return out;
    }
    copy_moves(index, worker); // entries that landed past the ceiling since the last plan
    return out;
}

void HostSlabStore::abandon() noexcept {
    try {
        if (copy_.valid()) { copy_.get(); }
        if (growth_.valid()) {
            Grown grown = growth_.get();
            for (Chunk& chunk : grown.chunks) { chunks_.push_back(std::move(chunk)); }
        }
        if (freeing_.valid()) { freeing_.get(); } // its chunks are gone already; the index never had them again
    } catch (...) {}
    freeing_bytes_ = 0;
    releasing_     = false;
}

void HostSlabStore::limit_index(PrefixCacheIndex& index) const {
    const std::uint32_t pinned = pinned_slabs();
    if (pinned >= index.host_slab_limit()) { return; }
    (void)index.begin_host_shrink(pinned, 1);
    if (!index.finish_host_shrink()) {
        throw std::logic_error("a fresh prefix index keeps slabs past the page-locked Host chunks");
    }
}

} // namespace infernix::runtime::prefix_cache
