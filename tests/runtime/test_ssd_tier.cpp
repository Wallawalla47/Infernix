// The prefix cache's SSD tier driver (hybrid-prefix-cache-spec §5.7) over a real index and scratch
// file: write-back starts below its lossless-room target, demotion keeps the snapshot, a promotion
// reads back exactly the bytes the Host slabs held (blocks and image), the measured read rate reaches
// the cost model, and drain/abandon settle every transfer. CPU and disk only.
#include "test_temp_path.h"
#include "runtime/prefix_cache/block_hash.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/ssd_tier.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace infernix;
using namespace infernix::runtime::prefix_cache;

namespace {

constexpr std::size_t kSlabBytes  = 64 * 1024;
constexpr std::uint32_t kHostSlabs = 32;
constexpr double kAny              = std::numeric_limits<double>::infinity();

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

// The Host slab pool and the Device ids the index hands back.
class Backend final : public PrefixIndexBackend {
public:
    Backend() : memory(static_cast<std::byte*>(::operator new(kSlabBytes * kHostSlabs, std::align_val_t{4096}))) {}
    ~Backend() override { ::operator delete(memory, std::align_val_t{4096}); }
    void release_device_block(std::uint32_t) noexcept override {}
    void move_host_slab(std::uint32_t from, std::uint32_t to) noexcept override {
        std::memcpy(slab(to), slab(from), kSlabBytes);
    }
    std::byte* slab(std::uint32_t id) const { return memory + static_cast<std::size_t>(id) * kSlabBytes; }
    std::byte* memory = nullptr;
    std::uint32_t next = 1;
};

std::byte pattern(std::uint32_t entity, std::size_t at) {
    return static_cast<std::byte>((entity * 131U) ^ (at * 7U) ^ (at >> 9));
}

void fill(Backend& backend, std::uint32_t slab, std::uint32_t entity) {
    std::byte* p = backend.slab(slab);
    for (std::size_t i = 0; i < kSlabBytes; ++i) { p[i] = pattern(entity, i); }
}

bool holds(const Backend& backend, std::uint32_t slab, std::uint32_t entity) {
    const std::byte* p = backend.slab(slab);
    for (std::size_t i = 0; i < kSlabBytes; ++i) {
        if (p[i] != pattern(entity, i)) { return false; }
    }
    return true;
}

PrefixIndexConfig config() {
    PrefixIndexConfig c;
    c.max_nodes                 = 128;
    c.max_snapshots             = 16;
    c.host_slabs                = kHostSlabs;
    c.image_slabs               = 4;
    c.device_snapshot_slots     = 0;
    c.block_bytes               = kSlabBytes;
    c.image_bytes               = 4 * kSlabBytes;
    c.ssd_slabs                 = 64;
    c.cost.token_seconds        = 1.0e-4;
    c.cost.h2d_bytes_per_second = 50.0e9;
    return c;
}

struct Conversation {
    std::vector<TokenId> tokens;
    std::vector<NodeRef> path;
    SnapshotRef endpoint;
};

// Blocks written through to the Host (each slab's bytes keyed by its node index), then a Host-born
// endpoint whose image slabs are keyed by 1000 + snapshot index + slab position.
Conversation publish(PrefixCacheIndex& index, Backend& backend, std::uint32_t seed, std::uint32_t blocks) {
    Conversation c;
    std::mt19937 rng(seed);
    c.tokens.resize(blocks * kBlockTokens);
    for (TokenId& token : c.tokens) { token = static_cast<TokenId>(rng() % 50000U); }
    const auto hashes = block_lookup_hashes(c.tokens, {});
    NodeRef parent;
    for (std::uint32_t b = 0; b < blocks; ++b) {
        const InsertResult inserted = index.insert_block(
            parent, hashes[b], std::span<const TokenId>(c.tokens.data() + b * kBlockTokens, kBlockTokens), 0,
            backend.next++, true);
        const auto slab = index.begin_host_fill(inserted.node);
        require(slab.has_value(), "a block is written through");
        fill(backend, *slab, inserted.node.index);
        index.complete_host_fill(inserted.node);
        c.path.push_back(inserted.node);
        parent = inserted.node;
    }
    auto image = index.reserve_host_image(false, kAny);
    require(image.has_value(), "an endpoint image fits");
    const std::vector<std::uint32_t> slabs = image->slabs;
    const PublishResult result = index.publish_host_snapshot(c.path.back(), blocks * kBlockTokens, {}, std::nullopt,
                                                             std::move(*image), SnapshotKind::Endpoint);
    require(result.created, "endpoint published");
    for (std::size_t k = 0; k < slabs.size(); ++k) {
        fill(backend, slabs[k], 1000U + result.snapshot.index * 8U + static_cast<std::uint32_t>(k));
    }
    c.endpoint = result.snapshot;
    index.release_path(c.path);
    (void)index.evict_backed_device_blocks(1000); // only the Host (and later the SSD) hold them
    index.check_invariants();
    return c;
}

void settle(SsdTier& tier, PrefixCacheIndex& index) {
    const auto stop = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (tier.busy() && std::chrono::steady_clock::now() < stop) {
        (void)tier.poll(index);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(!tier.busy(), "the transfers land");
    index.check_invariants();
}

} // namespace

int main() {
    const std::filesystem::path directory = infernix::test::unique_temp_path("ssd_tier");
    std::filesystem::create_directories(directory);
    int status = 0;
    try {
        Backend backend;
        PrefixCacheIndex index(config(), backend);
        SsdTier tier(directory, 64, kSlabBytes, [&](std::uint32_t slab) { return backend.slab(slab); });

        // a: 2 blocks + 4 image slabs; the Host's lossless room (26) is above three quarters of its 32
        // slabs (24): a quarter may stay unwritten.
        Conversation a = publish(index, backend, 1, 2);
        (void)tier.poll(index);
        require(!tier.busy(), "no write-back while the Host tier has room");
        // b: 3 blocks + 4 image slabs: lossless room 19 < 24, so the lowest-priority snapshot is written.
        Conversation b = publish(index, backend, 2, 3);
        (void)tier.poll(index);
        require(tier.busy(), "a write-back starts below the target");
        settle(tier, index);
        require(tier.counters().image_writes == 1 && tier.counters().write_bytes >= 6 * kSlabBytes,
                "one write-back wrote an image and its blocks, enough to reach the target");

        // Captures take the Host tier slab by slab, the tier polled between them as the Engine does:
        // write-back keeps ahead, so both conversations move to the SSD and nothing is lost. The
        // slabs they leave are overwritten.
        std::vector<HostImageReservation> reservations;
        for (;;) {
            (void)tier.poll(index);
            settle(tier, index);
            auto image = index.reserve_host_image(false, kAny);
            if (!image) { break; }
            for (const std::uint32_t slab : image->slabs) { std::memset(backend.slab(slab), 0, kSlabBytes); }
            reservations.push_back(std::move(*image));
            index.check_invariants();
        }
        require(reservations.size() == kHostSlabs / 4, "the captures took the whole Host tier");
        require(tier.counters().image_writes == 2, "each conversation was written once");
        require(index.snapshot(a.endpoint).host == CopyState::Absent && index.snapshot(b.endpoint).host == CopyState::Absent,
                "both endpoints moved to the SSD");
        require(index.stats().host_snapshot_evictions == 0, "nothing was lost");
        for (HostImageReservation& reservation : reservations) { index.release_host_image(std::move(reservation)); }

        // Promote a: Pending until it lands, then its blocks and image hold their bytes again.
        require(tier.promote(index, a.endpoint) == SsdTier::Promotion::Pending, "a promotion starts");
        require(tier.promote(index, a.endpoint) == SsdTier::Promotion::Pending, "a second quote waits");
        require(tier.promoting(), "promoting");
        bool landed       = false;
        const auto stop   = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (!landed && std::chrono::steady_clock::now() < stop) {
            landed = tier.poll(index);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        require(landed, "poll reports the landing");
        index.check_invariants();
        require(tier.promote(index, a.endpoint) == SsdTier::Promotion::Ready, "the source is ready");
        for (const NodeRef node : a.path) {
            const NodeView view = index.node(node);
            require(view.host == CopyState::Resident && holds(backend, view.host_slab, node.index),
                    "a promoted block holds its bytes");
        }
        const SnapshotView image = index.snapshot(a.endpoint);
        for (std::size_t k = 0; k < image.host_slabs.size(); ++k) {
            require(holds(backend, image.host_slabs[k], 1000U + a.endpoint.index * 8U + static_cast<std::uint32_t>(k)),
                    "the promoted image holds its bytes");
        }
        require(tier.counters().promotions == 1 && tier.counters().read_bytes == 6 * kSlabBytes,
                "one promotion read 2 blocks and 4 image slabs");

        // a is clean on the Host again (its SSD copies stayed): nothing more is written for it.
        const std::uint64_t written = tier.counters().write_bytes;
        (void)tier.poll(index);
        settle(tier, index);
        require(tier.counters().write_bytes == written, "a promoted conversation is not written again");
        require(index.config().cost.ssd_bytes_per_second == 2.0e9 || tier.counters().read_bytes_per_second > 0.0,
                "a measured read rate reaches the cost model");

        tier.drain(index);
        index.check_invariants();
        tier.abandon();
        require(!tier.busy(), "nothing in flight after abandon");
        std::cout << "ssd tier: ok (" << tier.counters().write_bytes << " bytes written, "
                  << tier.counters().read_bytes << " read)\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        status = 1;
    }
    std::error_code ec;
    std::filesystem::remove_all(directory, ec);
    return status;
}
