#pragma once

// The prefix cache's Host tier memory, shared by Qwen3.8-Flash-Next's and Qwen3.5's caches
// (hybrid-prefix-cache-spec §5.2): `slabs` slabs of slab_bytes, page-locked in chunks of whole slabs, and the tier's
// resize under RAM pressure (--ram-resize auto) while requests run as well as between them.
//
// - A chunk is committed, its pages touched (zeroed; at startup by several threads) and then registered with the
//   driver (RegisteredHostBuffer): the driver call is short (about 0.003 s/GiB, against 0.11 s/GiB for
//   cudaMallocHost), which matters when it runs while rounds launch. One very large page-locked block can fail or stall under WDDM where several smaller ones succeed, so
//   the tier is several chunks; a slab never crosses chunks, so a strided copy run never joins two (chunk_of is a
//   copy's group). A chunk is also the resize step.
// - A release retires the slabs from a ceiling on (PrefixCacheIndex::begin_host_shrink): the entries past it are
//   copied below it on the HostMemoryWorker with streaming stores (copy_streaming: no cache lines evicted from the
//   threads serving requests), the index switches them once the copies have landed and no restore reads a Host
//   slab, and once nothing past the ceiling is in use its chunks are released on the worker.
// - A growth page-locks chunks on the worker, up to the tier's `slabs` (an automatic tier starts with part of them
//   locked: what the start left; the rest is room to grow into RAM freed later), and links them (their slabs freed
//   into the index) at the next step.
// The engine thread never waits for a copy, a pin or a release: it starts them and step() applies what has landed
// (the worker wakes the engine after each job).

#include "core/host_memory.h"
#include "core/host_memory_worker.h"
#include "runtime/prefix_cache/prefix_index.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <string_view>
#include <vector>

namespace infernix::runtime::prefix_cache {

class HostSlabStore {
public:
    // The chunk the caches page-lock in, and their resize step. Registering memory stalls the device's work for a
    // time that grows with the range (RTX 5090, WDDM, 2026-11-13, a 2 ms kernel loop beside a second thread:
    // 1 GiB stretched a round to 38-46 ms, 256 MiB to 11-23 ms, 64 MiB to 3.5-4.7 ms, 16 MiB to 2.6 ms from 1.6),
    // and a strided copy run never spans two chunks: 64 MiB keeps a resize's stall at about a millisecond or two per
    // chunk while copy runs still join dozens of slabs.
    static constexpr std::size_t kChunkBytes = std::size_t{64} << 20U;

    HostSlabStore() = default; // no Host tier
    // A tier of `slabs` slabs of `slab_bytes` (a multiple of 4 KiB: the SSD tier moves slabs with unbuffered I/O) in
    // chunks of at most `chunk_bytes`, of which the chunks holding the first `locked` slabs are page-locked now (the
    // rest a growth may lock later; limit_index then gives a fresh index only the locked ones); `tier` names the tier
    // in the error a refused chunk throws, and `progress` sees the bytes locked so far and the total.
    HostSlabStore(std::size_t slab_bytes, std::uint32_t slabs, std::uint32_t locked, std::size_t chunk_bytes,
                  std::string_view tier, const std::function<void(std::uint64_t, std::uint64_t)>& progress = {});
    // Waits for the worker's jobs on its memory.
    ~HostSlabStore();
    HostSlabStore(const HostSlabStore&)            = delete;
    HostSlabStore& operator=(const HostSlabStore&) = delete;
    // Only before a resize starts (a job in flight names the store's memory, not the object).
    HostSlabStore(HostSlabStore&&) noexcept            = default;
    HostSlabStore& operator=(HostSlabStore&&) noexcept = default;

    [[nodiscard]] std::byte* slab(std::uint32_t id) const noexcept {
        return chunks_[id / per_chunk_]->data() + static_cast<std::size_t>(id % per_chunk_) * slab_bytes_;
    }
    [[nodiscard]] std::uint32_t chunk_of(std::uint32_t id) const noexcept { return id / per_chunk_; }
    [[nodiscard]] std::uint32_t slabs_per_chunk() const noexcept { return per_chunk_; }
    [[nodiscard]] std::size_t slab_bytes() const noexcept { return slab_bytes_; }
    [[nodiscard]] std::uint32_t slabs() const noexcept { return slabs_; }
    [[nodiscard]] bool empty() const noexcept { return slabs_ == 0; }
    // The most the tier may hold (`slabs`), and what is page-locked now.
    [[nodiscard]] std::uint64_t capacity_bytes() const noexcept { return static_cast<std::uint64_t>(slabs_) * slab_bytes_; }
    [[nodiscard]] std::uint64_t pinned_bytes() const noexcept;
    // The slabs the locked chunks hold: the index's limit outside a resize.
    [[nodiscard]] std::uint32_t pinned_slabs() const noexcept;

    // ---- resize (engine thread; the jobs run on `worker`, which outlives the store or is destroyed first)
    // The chunks a release keeps.
    void set_floor_chunks(std::uint32_t chunks) noexcept { floor_chunks_ = chunks; }
    // What a release could give back now (the chunks past the floor) and what a growth could take back; both 0 while
    // a resize is under way.
    [[nodiscard]] std::uint64_t shrinkable_bytes() const noexcept;
    [[nodiscard]] std::uint64_t growable_bytes() const noexcept;
    // A resize is under way until its memory work has run: a release until the worker has freed its chunks.
    [[nodiscard]] bool resizing() const noexcept { return releasing_ || growth_.valid() || freeing_.valid(); }
    // Starts giving back whole chunks worth at least `bytes`, keeping the floor (fewer when held entries keep the
    // slabs in use); returns the bytes it aims at (0: none can go now).
    std::uint64_t begin_release(PrefixCacheIndex& index, std::uint64_t bytes, HostMemoryWorker& worker);
    // Starts page-locking chunks worth at most `bytes`, up to the tier's slabs; returns the bytes asked for.
    std::uint64_t begin_growth(std::uint64_t bytes, HostMemoryWorker& worker);
    struct Progress {
        std::uint64_t released = 0; // chunks the worker released, counted when it has
        std::uint64_t grown    = 0; // chunks linked by this step
        bool release_done      = false;
        bool growth_done       = false;
        bool refused           = false; // the OS refused a chunk of the growth
        bool waiting           = false; // landed moves wait for no restore in flight: worth another step soon
    };
    // Applies what the worker finished: links grown chunks; commits landed moves once `restores_quiet` (no restore
    // batch is in flight or staged: one may read a slab being moved), plans the next copies, hands the chunks past
    // the ceiling to the worker once nothing there is in use, and ends the release once it has freed them. Never waits.
    Progress step(PrefixCacheIndex& index, bool restores_quiet, HostMemoryWorker& worker);
    // The index is being rebuilt (PrefixCache::clear): waits for the jobs in flight, links a growth that landed and
    // drops the release under way (its chunks stay); limit_index then gives the fresh index the locked slabs.
    void abandon() noexcept;
    void limit_index(PrefixCacheIndex& index) const;

private:
    using Chunk = std::unique_ptr<RegisteredHostBuffer>;
    struct Grown {
        std::vector<Chunk> chunks;
        bool refused = false;
    };
    // Commits, zeroes and page-locks one chunk; throws when the OS refuses.
    [[nodiscard]] static Chunk pin_chunk(std::size_t bytes);
    // Plans the next moves and posts their copies.
    void copy_moves(PrefixCacheIndex& index, HostMemoryWorker& worker);

    std::size_t slab_bytes_   = 0;
    std::uint32_t slabs_      = 0;
    std::uint32_t per_chunk_  = 1;
    std::uint32_t floor_chunks_ = 1;
    std::vector<Chunk> chunks_;
    bool releasing_ = false;
    std::future<void> copy_;
    std::future<Grown> growth_;
    std::future<void> freeing_; // the release of the chunks past the ceiling
    std::uint64_t freeing_bytes_ = 0;
};

} // namespace infernix::runtime::prefix_cache
