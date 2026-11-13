// The prefix cache's Host slab store (runtime/prefix_cache/host_slab_store.h) over a real index and page-locked
// memory, as both Programs' caches drive it under RAM pressure (--ram-resize auto): a release while a sequence still
// runs copies what lies past the ceiling below it on the worker with every byte kept, a pinned path included; its
// moves wait while a restore may read a Host slab; the chunks past the ceiling are given back. A growth page-locks
// chunks again and links them, and a rebuilt index gets the slabs page-locked now (limit_index). Needs a CUDA device
// (exit 77 without one).

#include "core/host_memory_worker.h"
#include "runtime/prefix_cache/block_hash.h"
#include "runtime/prefix_cache/host_slab_store.h"
#include "runtime/prefix_cache/prefix_index.h"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace pc = infernix::runtime::prefix_cache;

namespace {

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

class Backend final : public pc::PrefixIndexBackend {
public:
    void release_device_block(std::uint32_t) noexcept override {}
    std::uint32_t next = 1;
};

constexpr std::size_t kSlab        = 64 * 1024;
constexpr std::uint32_t kSlabs     = 64;
constexpr std::uint32_t kPerChunk  = 16;
constexpr std::uint64_t kChunk     = static_cast<std::uint64_t>(kPerChunk) * kSlab;
constexpr double kAny              = std::numeric_limits<double>::infinity();

pc::PrefixIndexConfig config() {
    pc::PrefixIndexConfig out;
    out.max_nodes                 = 256;
    out.max_snapshots             = 32;
    out.host_slabs                = kSlabs;
    out.image_slabs               = 4;
    out.device_snapshot_slots     = 0;
    out.block_bytes               = kSlab;
    out.image_bytes               = 4 * kSlab;
    out.cost.token_seconds        = 1.0e-4;
    out.cost.h2d_bytes_per_second = 50.0e9;
    return out;
}

std::vector<pc::NodeRef> insert_sequence(pc::PrefixCacheIndex& index, Backend& backend, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<infernix::TokenId> tokens(2 * pc::kBlockTokens);
    for (infernix::TokenId& token : tokens) { token = static_cast<infernix::TokenId>(rng() % 50000U); }
    const auto hashes = pc::block_lookup_hashes(tokens, {});
    std::vector<pc::NodeRef> path;
    pc::NodeRef parent;
    for (std::size_t b = 0; b < hashes.size(); ++b) {
        const std::span<const infernix::TokenId> block(tokens.data() + b * pc::kBlockTokens, pc::kBlockTokens);
        const pc::InsertResult result = index.insert_block(parent, hashes[b], block, 0, backend.next++, true);
        path.push_back(result.node);
        parent = result.node;
    }
    return path;
}

// An entry's bytes: a node's by its index, a snapshot's by 1000 + its index, then the byte's offset.
std::byte pattern(std::uint32_t entity, std::size_t at) {
    return static_cast<std::byte>((entity * 131U) ^ (at * 7U) ^ (at >> 9U));
}

void fill(const pc::HostSlabStore& store, std::uint32_t slab, std::uint32_t entity, std::size_t first) {
    std::byte* bytes = store.slab(slab);
    for (std::size_t i = 0; i < kSlab; ++i) { bytes[i] = pattern(entity, first + i); }
}

bool holds(const pc::HostSlabStore& store, std::uint32_t slab, std::uint32_t entity, std::size_t first) {
    const std::byte* bytes = store.slab(slab);
    for (std::size_t i = 0; i < kSlab; ++i) {
        if (bytes[i] != pattern(entity, first + i)) { return false; }
    }
    return true;
}

// Steps the store until `done` holds (the worker's jobs land on its own thread).
template <class Done>
pc::HostSlabStore::Progress step_until(pc::HostSlabStore& store, pc::PrefixCacheIndex& index, bool quiet,
                                       infernix::HostMemoryWorker& worker, Done done) {
    pc::HostSlabStore::Progress total;
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < until) {
        const pc::HostSlabStore::Progress step = store.step(index, quiet, worker);
        total.released += step.released;
        total.grown += step.grown;
        total.release_done = total.release_done || step.release_done;
        total.growth_done  = total.growth_done || step.growth_done;
        total.refused      = total.refused || step.refused;
        if (done(total)) { return total; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("the store did not finish in 10 s");
}

void test_release_and_growth(int device) {
    infernix::HostMemoryWorker worker(device);
    pc::HostSlabStore store(kSlab, kSlabs, kChunk, "test Host tier");
    store.set_floor_chunks(1);
    require(store.slabs_per_chunk() == kPerChunk && store.pinned_bytes() == kSlabs * kSlab &&
                store.pinned_slabs() == kSlabs && store.shrinkable_bytes() == 3 * kChunk && store.growable_bytes() == 0,
            "four chunks page-locked, three can go");
    Backend backend;
    pc::PrefixCacheIndex index(config(), backend);
    std::vector<std::vector<pc::NodeRef>> paths;
    std::vector<pc::SnapshotRef> snapshots;
    for (std::uint32_t s = 0; s < 6; ++s) {
        paths.push_back(insert_sequence(index, backend, 700 + s));
        auto image = index.reserve_host_image(false, kAny);
        require(image.has_value(), "an image reservation");
        snapshots.push_back(
            index.publish_host_snapshot(paths.back()[1], 2 * pc::kBlockTokens, {}, std::nullopt, std::move(*image),
                                        pc::SnapshotKind::Tap)
                .snapshot);
    }
    for (const auto& path : paths) {
        for (const pc::NodeRef node : path) {
            index.pin_node(node);
            require(index.begin_host_fill(node).has_value(), "a block backup");
            index.complete_host_fill(node);
            index.unpin_node(node);
        }
    }
    for (std::size_t s = 0; s + 1 < paths.size(); ++s) { index.release_path(paths[s]); } // the last sequence runs on
    // Every entry's slabs get its pattern.
    const auto for_each_entry = [&](const auto& visit) {
        for (const auto& path : paths) {
            for (const pc::NodeRef node : path) {
                if (index.valid(node) && index.node(node).host == pc::CopyState::Resident) {
                    visit(index.node(node).host_slab, node.index, std::size_t{0});
                }
            }
        }
        for (const pc::SnapshotRef snapshot : snapshots) {
            if (!index.valid(snapshot)) { continue; }
            const auto slabs = index.snapshot(snapshot).host_slabs;
            for (std::size_t k = 0; k < slabs.size(); ++k) { visit(slabs[k], 1000U + snapshot.index, k * kSlab); }
        }
    };
    for_each_entry([&](std::uint32_t slab, std::uint32_t entity, std::size_t first) { fill(store, slab, entity, first); });
    const std::uint32_t running_top = index.node(paths[5][1]).host_slab;
    require(running_top >= kPerChunk, "the running sequence's block lies past the first chunk");

    // Three chunks asked: the shrink keeps the floor, evicts to fit and plans the moves past slab 16.
    require(store.begin_release(index, 3 * kChunk, worker) == 3 * kChunk && store.resizing() &&
                index.host_ceiling() == kPerChunk && store.shrinkable_bytes() == 0 && store.growable_bytes() == 0,
            "a release of three chunks begins at slab 16");
    require(store.begin_release(index, kChunk, worker) == 0 && store.begin_growth(kChunk, worker) == 0,
            "one resize at a time");
    // While a restore may read a Host slab the moves wait (their bytes are copied meanwhile).
    (void)step_until(store, index, false, worker, [&](const auto&) { return worker.idle(); });
    (void)store.step(index, false, worker);
    require(store.resizing() && index.host_moves_pending() && store.pinned_bytes() == kSlabs * kSlab,
            "the moves wait for no restore in flight");
    const pc::HostSlabStore::Progress released =
        step_until(store, index, true, worker, [](const auto& total) { return total.release_done; });
    require(released.released == 3 * kChunk && !store.resizing() && store.pinned_bytes() == kChunk &&
                index.host_slab_limit() == kPerChunk,
            "the release ends with one chunk page-locked and the index's limit at its slabs");
    bool kept = true, below = true;
    for_each_entry([&](std::uint32_t slab, std::uint32_t entity, std::size_t first) {
        below = below && slab < kPerChunk;
        kept  = kept && holds(store, slab, entity, first);
    });
    require(below && kept, "every entry left is below the ceiling with every byte it had");
    require(index.valid(paths[5][1]) && index.node(paths[5][1]).host_slab < kPerChunk,
            "the running sequence's pinned block moved");
    index.check_invariants();
    index.release_path(paths[5]);

    // Growth: two chunks page-locked again on the worker and linked.
    require(store.growable_bytes() == 3 * kChunk && store.begin_growth(2 * kChunk + kSlab, worker) == 2 * kChunk,
            "a growth asks whole chunks");
    const pc::HostSlabStore::Progress grown =
        step_until(store, index, true, worker, [](const auto& total) { return total.growth_done; });
    require(grown.grown == 2 * kChunk && !grown.refused && store.pinned_slabs() == 3 * kPerChunk &&
                index.host_slab_limit() == 3 * kPerChunk,
            "the growth links two chunks");
    index.check_invariants();
    std::vector<std::optional<pc::HostImageReservation>> images;
    for (int i = 0; i < 8; ++i) { images.push_back(index.reserve_host_image(false, kAny)); }
    bool fits = true;
    for (auto& image : images) {
        require(image.has_value(), "a grown tier reserves images");
        for (const std::uint32_t slab : image->slabs) {
            fits = fits && slab < 3 * kPerChunk;
            std::memset(store.slab(slab), 0x5A, kSlab); // the linked memory is usable
        }
        index.release_host_image(std::move(*image));
    }
    require(fits, "reservations stay inside the page-locked chunks");

    // A rebuilt index (PrefixCache::clear) starts at the page-locked slabs, not at the startup size.
    store.abandon();
    pc::PrefixCacheIndex fresh(config(), backend);
    store.limit_index(fresh);
    require(fresh.host_slab_limit() == 3 * kPerChunk && !fresh.host_shrinking(), "a rebuilt index gets the page-locked slabs");
    fresh.check_invariants();
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "skipped: no CUDA device\n";
        return 77;
    }
    try {
        test_release_and_growth(0);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "host slab store: ok\n";
    return 0;
}
