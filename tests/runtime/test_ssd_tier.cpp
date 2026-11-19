// The prefix cache's SSD tier driver (hybrid-prefix-cache-spec §5.7) over a real index and scratch
// file: write-back starts below its lossless-room target, demotion keeps the snapshot, a promotion
// reads back exactly the bytes the Host slabs held (blocks and image), the measured read rate reaches
// the cost model, and drain/abandon settle every transfer. CPU and disk only.
#include "test_temp_path.h"
#include "runtime/prefix_cache/block_hash.h"
#include "runtime/prefix_cache/persist_file.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/ssd_tier.h"
#include "runtime/prefix_cache/fp32_row_codec.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
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

PrefixIndexConfig config(std::uint32_t image_slabs = 4) {
    PrefixIndexConfig c;
    c.max_nodes                 = 128;
    c.max_snapshots             = 16;
    c.host_slabs                = kHostSlabs;
    c.image_slabs               = image_slabs;
    c.device_snapshot_slots     = 0;
    c.block_bytes               = kSlabBytes;
    c.image_bytes               = image_slabs * kSlabBytes;
    c.ssd_slabs                 = 64;
    c.cost.token_seconds        = 1.0e-4;
    c.cost.h2d_bytes_per_second = 50.0e9;
    return c;
}

struct Conversation {
    std::vector<TokenId> tokens;
    std::vector<NodeRef> path;
    SnapshotRef endpoint;
    SnapshotRef tap; // with tap_blocks
};

// Blocks written through to the Host (each slab's bytes keyed by its node index), then a Host-born
// endpoint whose image slabs are keyed by 1000 + snapshot index + slab position. With `tap_blocks`, a
// Host-born tap after that many blocks first.
Conversation publish(PrefixCacheIndex& index, Backend& backend, std::uint32_t seed, std::uint32_t blocks,
                     std::uint32_t tap_blocks = 0) {
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
    if (tap_blocks != 0) {
        auto tap_image = index.reserve_host_image(false, kAny);
        require(tap_image.has_value(), "a tap image fits");
        const std::vector<std::uint32_t> tap_slabs = tap_image->slabs;
        const PublishResult tap = index.publish_host_snapshot(c.path[tap_blocks - 1], tap_blocks * kBlockTokens, {}, std::nullopt,
                                                              std::move(*tap_image), SnapshotKind::Tap);
        require(tap.created, "tap published");
        for (std::size_t k = 0; k < tap_slabs.size(); ++k) {
            fill(backend, tap_slabs[k], 1000U + tap.snapshot.index * 8U + static_cast<std::uint32_t>(k));
        }
        c.tap = tap.snapshot;
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

// FP32 words like a recurrent state's: random signs and mantissas, exponents within a few binades.
void fill_state(std::byte* p, std::size_t bytes, std::uint32_t seed) {
    std::mt19937 rng(seed);
    for (std::size_t i = 0; i < bytes; i += 4) {
        const std::uint32_t word = (rng() & 0x807FFFFFU) | ((118U + rng() % 9U) << 23U);
        std::memcpy(p + i, &word, 4);
    }
}

// The image codec (fp32_row_codec): exact for every bit pattern, smaller on state-like words, raw where
// nothing shrinks, and a damaged payload is refused.
void test_codec() {
    constexpr std::size_t kBytes = 64 * 1024;
    std::vector<std::byte> in(kBytes), payload(codec_payload_bound(kBytes)), out(kBytes);
    const auto round_trip = [&](const char* what) {
        const std::size_t size = encode_slab(in.data(), kBytes, payload.data());
        require(size <= codec_payload_bound(kBytes) && payload_size(payload.data()) == size, std::string(what) + ": payload size");
        require(decode_slab(payload.data(), size, out.data(), kBytes) && out == in, std::string(what) + ": round trip");
        return size;
    };
    fill_state(in.data(), kBytes, 7);
    const std::size_t state = round_trip("state-like words");
    require(state < kBytes * 92 / 100, "state-like words shrink by more than 8 % (" + std::to_string(state) + ")");
    // Outliers in one row (escapes), the 14/15 escape boundary and special values in another.
    const std::uint32_t specials[] = {0x00000000U, 0x80000000U, 0x7F800000U, 0xFF800000U, 0x7FC00001U,
                                      0x00000001U, 0x807FFFFFU, 0x3F800000U};
    for (std::size_t i = 0; i < 8; ++i) { std::memcpy(in.data() + 4 * i, &specials[i], 4); }
    const std::uint32_t edge14 = (126U - 14U) << 23U, edge15 = (126U - 15U) << 23U;
    std::memcpy(in.data() + 512, &edge14, 4);
    std::memcpy(in.data() + 516, &edge15, 4);
    (void)round_trip("special values and escapes");
    std::fill(in.begin(), in.end(), std::byte{0});
    require(round_trip("zeros") < kBytes, "zeros shrink");
    std::mt19937 rng(9);
    for (std::byte& b : in) { b = static_cast<std::byte>(rng()); }
    require(round_trip("random bytes") == 4U + kBytes, "an incompressible slab is stored raw");
    // A damaged payload is refused.
    fill_state(in.data(), kBytes, 11);
    const std::size_t size = encode_slab(in.data(), kBytes, payload.data());
    require(!decode_slab(payload.data(), size - 1, out.data(), kBytes), "a truncated payload is refused");
    payload[4] = std::byte{7}; // the first row's kind
    require(!decode_slab(payload.data(), size, out.data(), kBytes), "a damaged row is refused");
}

// With INFERNIX_TEST_CODEC_DUMP naming a saved Flash-Next Host tier (--prefix-cache-file), times the codec
// on its state images (found by their header magic) and reports the ratio and the encode and decode rates,
// one thread and eight. Not a check: the numbers are for the SSD tier's design notes.
void bench_codec(const char* dump) {
    constexpr std::size_t kSlab = 860'160; // FP4-Local INT8 KV with DFlash2 (no MTP planes)
    std::FILE* file = std::fopen(dump, "rb");
    require(file != nullptr, "the codec dump opens");
    std::vector<std::byte> bytes(static_cast<std::size_t>(std::filesystem::file_size(dump)));
    require(std::fread(bytes.data(), 1, bytes.size(), file) == bytes.size(), "the codec dump reads");
    std::fclose(file);
    const std::size_t end = bytes.size() - 16;
    std::vector<std::size_t> images;
    for (std::size_t k = 1; k * kSlab <= end; ++k) {
        if (std::memcmp(bytes.data() + end - k * kSlab, "QWQ4IMG1", 8) == 0) { images.push_back(end - k * kSlab); }
    }
    require(!images.empty(), "the dump holds state images");
    // The SSD copy keeps the image up to its DFlash2 ring: 136 of its 184 slabs (116.5 MB of 157.6).
    constexpr std::size_t kKept = 136;
    std::vector<const std::byte*> slabs;
    for (std::size_t i = 0; i < std::min<std::size_t>(images.size(), 4); ++i) {
        for (std::size_t k = 0; k < kKept; ++k) { slabs.push_back(bytes.data() + images[i] + k * kSlab); }
    }
    std::vector<std::byte> stream(slabs.size() * codec_payload_bound(kSlab));
    std::vector<std::size_t> offsets(slabs.size() + 1, 0);
    auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < slabs.size(); ++i) {
        offsets[i + 1] = offsets[i] + encode_slab(slabs[i], kSlab, stream.data() + offsets[i]);
    }
    const double encode = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const double raw    = static_cast<double>(slabs.size() * kSlab);
    std::vector<std::byte> out(slabs.size() * kSlab);
    const auto decode = [&](unsigned threads) {
        const auto begin = std::chrono::steady_clock::now();
        std::vector<std::thread> pool;
        std::atomic<bool> ok{true};
        for (unsigned t = 0; t < threads; ++t) {
            pool.emplace_back([&, t] {
                for (std::size_t i = slabs.size() * t / threads; i < slabs.size() * (t + 1) / threads; ++i) {
                    if (!decode_slab(stream.data() + offsets[i], offsets[i + 1] - offsets[i], out.data() + i * kSlab, kSlab)) {
                        ok = false;
                    }
                }
            });
        }
        for (std::thread& thread : pool) { thread.join(); }
        require(ok.load(), "the dump's payloads decode");
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    };
    const double one = decode(1), eight = decode(8);
    for (std::size_t i = 0; i < slabs.size(); ++i) {
        require(std::memcmp(out.data() + i * kSlab, slabs[i], kSlab) == 0, "the dump's slabs decode exactly");
    }
    std::cout << "codec on " << images.size() << " dumped images (" << slabs.size() / kKept << " timed, the kept "
              << kKept << " slabs each): " << 100.0 * (1.0 - static_cast<double>(offsets.back()) / raw)
              << " % smaller; encode " << raw / encode / 1e9 << " GB/s (one thread); decode " << raw / one / 1e9
              << " GB/s (one), " << raw / eight / 1e9 << " GB/s (eight)\n";
}

// The codec with a partial SSD copy (Qwen4Exp leaves out its DFlash2 ring): 16 of 17 image slabs are
// packed into 15, and a promotion decodes them and zero-fills the 17th.
void test_packed_partial(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    Backend backend;
    PrefixIndexConfig c = config(17);
    c.ssd_image_slabs   = 16;
    PrefixCacheIndex index(c, backend);
    SsdTier tier(directory, 64, kSlabBytes, [&](std::uint32_t slab) { return backend.slab(slab); }, 0, true);
    const Conversation a = publish(index, backend, 77, 2);
    const SnapshotView view = index.snapshot(a.endpoint);
    std::vector<std::byte> copy;
    for (std::size_t k = 0; k < view.host_slabs.size(); ++k) {
        fill_state(backend.slab(view.host_slabs[k]), kSlabBytes, 500U + static_cast<std::uint32_t>(k));
        copy.insert(copy.end(), backend.slab(view.host_slabs[k]), backend.slab(view.host_slabs[k]) + kSlabBytes);
    }
    for (int i = 0; i < 20 && tier.counters().image_writes == 0; ++i) {
        (void)tier.poll(index);
        settle(tier, index);
    }
    require(tier.counters().image_writes == 1 && tier.counters().packed_images == 1 &&
                tier.counters().packed_saved_bytes == kSlabBytes,
            "a partial image was written packed, one SSD slab smaller (" + std::to_string(tier.counters().image_writes) +
                " writes, " + std::to_string(tier.counters().packed_images) + " packed)");
    auto image = index.reserve_host_image(false, kAny);
    require(image.has_value(), "a second image displaces the first to the SSD");
    for (const std::uint32_t slab : image->slabs) { std::memset(backend.slab(slab), 0xAB, kSlabBytes); }
    index.release_host_image(std::move(*image));
    require(index.snapshot(a.endpoint).host == CopyState::Absent, "the image is SSD-only");
    require(tier.promote(index, a.endpoint) == SsdTier::Promotion::Pending, "a packed partial promotion starts");
    settle(tier, index);
    const SnapshotView back = index.snapshot(a.endpoint);
    require(back.host == CopyState::Resident, "it landed");
    for (std::size_t k = 0; k < 16; ++k) {
        require(std::memcmp(backend.slab(back.host_slabs[k]), copy.data() + k * kSlabBytes, kSlabBytes) == 0,
                "a kept slab decodes to its bytes");
    }
    const std::byte* omitted = backend.slab(back.host_slabs[16]);
    require(std::all_of(omitted, omitted + kSlabBytes, [](std::byte b) { return b == std::byte{0}; }),
            "the omitted slab holds zeros");
    tier.drain(index);
    index.check_invariants();
}

// The prefix cache file's binding for this test's Host tier and kept SSD tier.
PersistBinding test_binding(Backend& backend, SsdTier& tier, std::uint32_t image_slabs) {
    PersistBinding binding;
    binding.magic       = {'T', 'E', 'S', 'T', 'P', 'C', '0', '1'};
    binding.geometry    = {std::byte{1}, std::byte{2}, std::byte{3}};
    binding.slab_bytes  = kSlabBytes;
    binding.image_slabs = image_slabs;
    binding.host_slabs  = kHostSlabs;
    binding.slab        = [&backend](std::uint32_t id) { return backend.slab(id); };
    binding.ssd         = &tier;
    return binding;
}

// The only snapshot a restored index holds.
SnapshotRef only_snapshot(const PrefixCacheIndex& index) {
    std::vector<NodeRef> nodes;
    std::vector<SnapshotRef> snapshots;
    index.collect_persistable(nodes, snapshots, true);
    require(snapshots.size() == 1, "the restored index holds one snapshot (" + std::to_string(snapshots.size()) + ")");
    return snapshots.front();
}

// The SSD tier across a restart (§5.5, candidates YK-02): a snapshot only on the SSD is saved by reference
// in the prefix cache file, a restart reopens the kept file and takes it back without reading it, and its
// first promotion reads exactly the bytes the Host slabs held. A slab overwritten after the save (a crash
// after it) fails that promotion: the entry goes and nothing of it is used.
void test_kept_restart(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    const std::filesystem::path kept = directory / "kept.bin";
    const std::filesystem::path file = directory / "cache.bin";
    const PrefixIndexConfig c        = config(17);
    std::vector<std::byte> image;
    std::uint32_t frontier = 0;
    {
        Backend backend;
        PrefixCacheIndex index(c, backend);
        SsdTier tier(directory, 64, kSlabBytes, [&](std::uint32_t slab) { return backend.slab(slab); }, 0, false, kept);
        require(tier.kept() && !tier.kept_content(), "a new kept file holds no previous run");
        const Conversation a = publish(index, backend, 91, 2);
        const SnapshotView view = index.snapshot(a.endpoint);
        for (const std::uint32_t slab : view.host_slabs) {
            image.insert(image.end(), backend.slab(slab), backend.slab(slab) + kSlabBytes);
        }
        frontier = view.frontier;
        for (int i = 0; i < 20 && tier.counters().image_writes == 0; ++i) {
            (void)tier.poll(index);
            settle(tier, index);
        }
        require(tier.counters().image_writes == 1, "the endpoint was written back");
        auto displacing = index.reserve_host_image(false, kAny);
        require(displacing.has_value(), "a second image displaces the first to the SSD");
        index.release_host_image(std::move(*displacing));
        require(index.snapshot(a.endpoint).host == CopyState::Absent, "the endpoint is SSD-only");
        tier.drain(index);
        const PersistResult saved = save_prefix_cache(index, test_binding(backend, tier, c.image_slabs), file, "fp", {});
        require(saved.ok && saved.snapshots == 1 && saved.ssd_snapshots == 1,
                "the SSD-only snapshot is saved by reference (" + saved.message + ")");
    }
    std::uint32_t first_ssd_slab = kNoId;
    {
        Backend backend;
        PrefixCacheIndex index(c, backend);
        SsdTier tier(directory, 64, kSlabBytes, [&](std::uint32_t slab) { return backend.slab(slab); }, 0, false, kept);
        require(tier.kept_content(), "the restart finds the previous run's SSD file");
        const PersistResult loaded = load_prefix_cache(index, test_binding(backend, tier, c.image_slabs), file, "fp", {});
        require(loaded.ok && loaded.snapshots == 1 && loaded.ssd_snapshots == 1 && loaded.ssd_message.empty(),
                "the SSD-only snapshot comes back (" + loaded.message + loaded.ssd_message + ")");
        index.check_invariants();
        const SnapshotRef snapshot = only_snapshot(index);
        require(index.snapshot(snapshot).frontier == frontier && index.snapshot(snapshot).host == CopyState::Absent &&
                    index.snapshot(snapshot).ssd == CopyState::Resident,
                "it is SSD-only at its frontier");
        first_ssd_slab = index.snapshot(snapshot).ssd_slabs[0];
        require(tier.promote(index, snapshot) == SsdTier::Promotion::Pending, "its promotion starts");
        settle(tier, index);
        const SnapshotView back = index.snapshot(snapshot);
        require(back.host == CopyState::Resident && tier.counters().kept_slab_mismatches == 0, "it landed, checked");
        for (std::size_t k = 0; k < back.host_slabs.size(); ++k) {
            require(std::memcmp(backend.slab(back.host_slabs[k]), image.data() + k * kSlabBytes, kSlabBytes) == 0,
                    "a promoted slab holds the bytes saved before the restart");
        }
        tier.drain(index);
    }
    {
        // Another process wrote over the slab after the save.
        std::fstream damage(kept, std::ios::in | std::ios::out | std::ios::binary);
        damage.seekp(static_cast<std::streamoff>(first_ssd_slab) * static_cast<std::streamoff>(kSlabBytes));
        const std::vector<char> junk(4096, 'x');
        damage.write(junk.data(), static_cast<std::streamsize>(junk.size()));
    }
    {
        Backend backend;
        PrefixCacheIndex index(c, backend);
        SsdTier tier(directory, 64, kSlabBytes, [&](std::uint32_t slab) { return backend.slab(slab); }, 0, false, kept);
        const PersistResult loaded = load_prefix_cache(index, test_binding(backend, tier, c.image_slabs), file, "fp", {});
        require(loaded.ok && loaded.ssd_snapshots == 1, "the file still names the slab");
        const SnapshotRef snapshot = only_snapshot(index);
        require(tier.promote(index, snapshot) == SsdTier::Promotion::Pending, "its promotion starts");
        settle(tier, index);
        require(tier.counters().kept_slab_mismatches == 1 && tier.counters().read_failures == 1,
                "the changed slab fails the promotion");
        require(!index.valid(snapshot), "the snapshot whose bytes changed is gone");
        index.check_invariants();
    }
}

// A promotion that finds no Host room while a write-back is in flight waits for it (§5.7: the write pins
// its entries until it lands, then they are clean and yield without loss; a restart's restored Host
// entries are in that state): promote says Pending and counts the wait, poll reports the write's landing,
// and quoting again at each landing ends with the source Ready and its bytes. False when the write landed
// before the quote (the promotion then simply started), so the caller tries again.
bool room_wait_once(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    Backend backend;
    PrefixCacheIndex index(config(17), backend);
    SsdTier tier(directory, 64, kSlabBytes, [&](std::uint32_t slab) { return backend.slab(slab); });
    // a: 2 blocks + a 17-slab image, written back, then displaced to the SSD (its blocks stay, clean).
    const Conversation a = publish(index, backend, 61, 2);
    for (int i = 0; i < 20 && tier.counters().image_writes == 0; ++i) {
        (void)tier.poll(index);
        settle(tier, index);
    }
    require(tier.counters().image_writes == 1, "the first endpoint was written back");
    auto displacing = index.reserve_host_image(false, kAny);
    require(displacing.has_value(), "a second image displaces the first to the SSD");
    index.release_host_image(std::move(*displacing));
    require(index.snapshot(a.endpoint).host == CopyState::Absent, "the first endpoint is SSD-only");
    // b: 19 Host-only slabs; its write-back starts (lossless room 13 < 24) and pins them. 11 slabs are
    // free, a's image needs 17, and nothing else may yield: no room until b's write lands.
    const Conversation b = publish(index, backend, 62, 2);
    for (int i = 0; i < 20 && !tier.busy(); ++i) { (void)tier.poll(index); }
    require(tier.busy(), "the second conversation's write-back is in flight");
    const SsdTier::Promotion first = tier.promote(index, a.endpoint);
    if (tier.counters().promotions_waiting_for_room == 0) {
        settle(tier, index);
        tier.drain(index);
        return false;
    }
    require(first == SsdTier::Promotion::Pending, "no room while a write-back is in flight: the promotion waits");
    bool ready = false;
    for (int quote = 0; quote < 50 && !ready; ++quote) {
        bool landed     = false;
        const auto stop = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (!landed && std::chrono::steady_clock::now() < stop) {
            landed = tier.poll(index);
            if (!landed) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        }
        require(landed, "poll reports the landing the waiting promotion needs");
        const SsdTier::Promotion again = tier.promote(index, a.endpoint);
        require(again != SsdTier::Promotion::Unavailable, "a waiting promotion is not given up once room comes");
        ready = again == SsdTier::Promotion::Ready;
    }
    require(ready, "the waiting promotion lands");
    require(tier.counters().promotions == 1, "one promotion landed");
    const SnapshotView image = index.snapshot(a.endpoint);
    for (std::size_t k = 0; k < image.host_slabs.size(); ++k) {
        require(holds(backend, image.host_slabs[k], 1000U + a.endpoint.index * 8U + static_cast<std::uint32_t>(k)),
                "the promoted image holds its bytes");
    }
    require(index.snapshot(b.endpoint).ssd == CopyState::Resident, "b's write-back landed");
    tier.drain(index);
    index.check_invariants();
    return true;
}

void test_room_wait(const std::filesystem::path& directory) {
    for (int attempt = 0; attempt < 5; ++attempt) {
        if (room_wait_once(directory / std::to_string(attempt))) { return; }
    }
    require(false, "no promotion met a write-back in flight in 5 attempts (each write landed before the quote)");
}

// The tier with the codec: state-like images are written packed (fewer SSD bytes) and read back exactly;
// incompressible ones (the main test's pattern) still go raw through the same pipeline.
void test_packed_tier(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    Backend backend;
    PrefixCacheIndex index(config(9), backend); // nine 64 KiB slabs: a packed image saves one
    SsdTier tier(directory, 64, kSlabBytes, [&](std::uint32_t slab) { return backend.slab(slab); }, 0, true);
    std::vector<std::vector<std::byte>> images;
    std::vector<Conversation> conversations;
    for (std::uint32_t seed = 1; seed <= 2; ++seed) {
        conversations.push_back(publish(index, backend, 40 + seed, 2 + seed));
        const SnapshotView view = index.snapshot(conversations.back().endpoint);
        std::vector<std::byte> copy;
        for (std::size_t k = 0; k < view.host_slabs.size(); ++k) {
            fill_state(backend.slab(view.host_slabs[k]), kSlabBytes, 100U * seed + static_cast<std::uint32_t>(k));
            copy.insert(copy.end(), backend.slab(view.host_slabs[k]), backend.slab(view.host_slabs[k]) + kSlabBytes);
        }
        images.push_back(std::move(copy));
        (void)tier.poll(index);
        settle(tier, index);
    }
    std::vector<HostImageReservation> reservations;
    for (;;) {
        (void)tier.poll(index);
        settle(tier, index);
        auto image = index.reserve_host_image(false, kAny);
        if (!image) { break; }
        for (const std::uint32_t slab : image->slabs) { std::memset(backend.slab(slab), 0, kSlabBytes); }
        reservations.push_back(std::move(*image));
    }
    require(tier.counters().image_writes == 2 && tier.counters().packed_images == 2 &&
                tier.counters().packed_saved_bytes == 2 * kSlabBytes,
            "both state-like images were written packed, one SSD slab smaller each");
    for (HostImageReservation& reservation : reservations) { index.release_host_image(std::move(reservation)); }
    for (std::size_t c = 0; c < conversations.size(); ++c) {
        require(tier.promote(index, conversations[c].endpoint) == SsdTier::Promotion::Pending, "a packed promotion starts");
        settle(tier, index);
        require(tier.promote(index, conversations[c].endpoint) == SsdTier::Promotion::Ready, "it landed");
        const SnapshotView view = index.snapshot(conversations[c].endpoint);
        for (std::size_t k = 0; k < view.host_slabs.size(); ++k) {
            require(std::memcmp(backend.slab(view.host_slabs[k]), images[c].data() + k * kSlabBytes, kSlabBytes) == 0,
                    "a packed image decodes to its bytes");
        }
    }
    tier.drain(index);
    index.check_invariants();
}

// The write budget (--ssd-context-gib-per-hour): an hour's bytes, refilled over the hour and kept across
// restarts; everything is written while it is at least half full, then only the most valuable share
// (2 x fill, by bytes) of the recent candidates, and nothing larger than what is left.
void test_write_budget(const std::filesystem::path& directory) {
    using Clock      = SsdWriteBudget::Clock;
    const auto start = Clock::time_point{std::chrono::hours(1000)};
    SsdWriteBudget selective(1000, directory / "selective.txt", start);
    require(selective.available_bytes() == 1000, "a budget without a saved level starts full");
    // Ten candidates of 50 bytes worth 1..10, all written while the bucket is at least half full.
    for (int value = 1; value <= 10; ++value) {
        require(selective.admit(value, 50, true, start), "at least half full: every candidate is written");
        selective.charge(50);
    }
    require(selective.available_bytes() == 500, "charged");
    selective.charge(100); // 400 left: fill 0.4, the best 80 % by bytes of the recent candidates
    require(!selective.admit(2.0, 50, false, start), "a candidate below the best 80 % is passed over");
    require(selective.admit(3.0, 50, false, start), "one inside it is written");
    selective.charge(300); // 100 left: fill 0.1, the best 20 %
    require(!selective.admit(8.0, 50, false, start) && selective.admit(9.0, 50, false, start), "the best 20 % only");
    require(!selective.admit(100.0, 150, false, start), "nothing larger than what is left");
    // A candidate larger than an hour's bytes is written from a full bucket (it could never be written
    // otherwise) and empties it; from anything less it waits.
    SsdWriteBudget oversized(1000, directory / "oversized.txt", start);
    require(oversized.admit(1.0, 1500, true, start), "an oversized candidate is written from a full bucket");
    oversized.charge(1500);
    require(oversized.available_bytes() == 0, "it takes the whole bucket");
    require(!oversized.admit(1.0, 1500, false, start + std::chrono::minutes(59)),
            "an oversized candidate waits for a full bucket");
    require(oversized.admit(1.0, 1500, false, start + std::chrono::minutes(61)), "and is written once it is full again");
    // It refills at an hour's bytes per hour, and a restart keeps the level.
    const auto state = directory / "budget.txt";
    SsdWriteBudget budget(1000, state, start);
    budget.charge(900);
    const auto later = start + std::chrono::minutes(30);
    require(budget.admit(1.0, 50, false, later) && budget.available_bytes() == 600, "half an hour refills half the budget");
    budget.save(later);
    SsdWriteBudget restarted(1000, state, later + std::chrono::minutes(15));
    require(restarted.available_bytes() == 850, "a restart keeps the level and refills from when it was saved");
    SsdWriteBudget capped(1000, state, later + std::chrono::hours(2));
    require(capped.available_bytes() == 1000, "the bucket holds at most an hour");
    // A clock that was ahead when the level was saved, then set back: refill runs from the new time.
    SsdWriteBudget ahead(1000, state, later - std::chrono::hours(5));
    require(ahead.available_bytes() == 600, "a level saved in the future is kept as it was");
    ahead.charge(600);
    require(ahead.admit(1.0, 100, false, later - std::chrono::hours(5) + std::chrono::minutes(6)),
            "a level saved in the future refills from now, not from its saved time");
    ahead.save(later - std::chrono::hours(5) + std::chrono::minutes(6));
    SsdWriteBudget back(1000, state, later - std::chrono::hours(5) + std::chrono::minutes(12));
    require(back.available_bytes() == 200, "the save after a clock set back records the new time");
    bool refused = false;
    try {
        SsdWriteBudget empty(0, state, start);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused, "a budget needs a positive hourly size");
}

// A refused write-back is tried again without anything else waking the caller (§5.7): the candidates are
// refused while pinned (as a snapshot whose Host copies are still landing is), the tier names a near time
// for the next attempt, and once that time has come it reports the write-back due with no poll in between
// (an idle Engine waits until then) and the next poll writes.
void test_idle_retry(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    Backend backend;
    PrefixCacheIndex index(config(), backend);
    SsdTier tier(directory, 64, kSlabBytes, [&](std::uint32_t slab) { return backend.slab(slab); });
    const Conversation a = publish(index, backend, 71, 2);
    (void)tier.poll(index);
    require(!tier.write_back_retry_at().has_value() && !tier.write_back_due(), "a Host tier with room wants no write-back");
    const Conversation b = publish(index, backend, 72, 3);
    index.pin_snapshot(a.endpoint);
    index.pin_snapshot(b.endpoint);
    (void)tier.poll(index);
    require(!tier.busy() && tier.counters().image_writes == 0, "pinned candidates are refused");
    const auto retry = tier.write_back_retry_at();
    require(retry.has_value(), "a refused write-back names its next attempt");
    require(*retry - std::chrono::steady_clock::now() < std::chrono::seconds(1), "the next attempt is near");
    // Due first, then the clock: a due answer read its clock no later than this one, so it must be past the retry
    // (the other order fails when this thread loses the CPU between its clock reading and the tier's).
    const bool due_early = tier.write_back_due();
    require(!due_early || std::chrono::steady_clock::now() >= *retry, "not due before its time");
    index.unpin_snapshot(a.endpoint);
    index.unpin_snapshot(b.endpoint);
    std::this_thread::sleep_until(*retry + std::chrono::milliseconds(1));
    require(tier.write_back_due(), "due once its time has come, with no poll in between");
    (void)tier.poll(index);
    require(tier.busy(), "the retried write-back starts");
    settle(tier, index);
    require(tier.counters().image_writes >= 1, "and lands");
    // Refusals with nothing new published back off, so an idle caller is not woken ten times a second.
    index.pin_snapshot(a.endpoint);
    index.pin_snapshot(b.endpoint);
    std::chrono::steady_clock::duration longest{};
    for (int attempt = 0; attempt < 4 && tier.write_back_retry_at().has_value(); ++attempt) {
        (void)tier.poll(index);
        const auto next = tier.write_back_retry_at();
        if (!next) { break; }
        longest = std::max(longest, *next - std::chrono::steady_clock::now());
        std::this_thread::sleep_until(*next + std::chrono::milliseconds(1));
    }
    if (tier.write_back_retry_at().has_value()) {
        require(longest > std::chrono::milliseconds(150), "repeated refusals wait longer than the first");
    }
    index.unpin_snapshot(a.endpoint);
    index.unpin_snapshot(b.endpoint);
    tier.drain(index);
    index.check_invariants();
}

// A write the file refuses at once (here: Host memory that is not 4 KiB-aligned) fails like a write the
// drive fails: after three the tier stops writing, and the poll that met them returns.
void test_refused_submit(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    Backend backend;
    PrefixCacheIndex index(config(), backend);
    SsdTier tier(directory, 64, kSlabBytes, [&](std::uint32_t slab) { return backend.slab(slab) + 512; });
    const Conversation a = publish(index, backend, 81, 2);
    const Conversation b = publish(index, backend, 82, 3);
    (void)tier.poll(index); // spun forever before the failure limit was tested inside the loop
    require(!tier.busy() && tier.counters().write_failures == 3 && !tier.last_error().empty(),
            "three refused writes stop the write-back (" + std::to_string(tier.counters().write_failures) + ")");
    require(index.snapshot(a.endpoint).ssd == CopyState::Absent && index.snapshot(b.endpoint).ssd == CopyState::Absent,
            "the entries stay as they were");
    (void)tier.poll(index);
    require(tier.counters().write_failures == 3 && !tier.write_back_due(), "and nothing is tried again");
    index.check_invariants();
}

// A promotion whose read cannot be submitted (here a misaligned destination, which the file refuses) has read
// nothing, so the entry keeps its SSD copies (PC-03): Unavailable now, and a later promotion reads them back.
void test_refused_promotion(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    Backend backend;
    PrefixCacheIndex index(config(), backend);
    bool misaligned = false;
    SsdTier tier(directory, 64, kSlabBytes,
                 [&](std::uint32_t slab) { return backend.slab(slab) + (misaligned ? 512 : 0); });
    const Conversation a = publish(index, backend, 91, 2);
    (void)publish(index, backend, 92, 3);
    // As the main test: captures take the Host tier while the write-back keeps ahead, so a moves to the SSD.
    std::vector<HostImageReservation> reservations;
    for (;;) {
        (void)tier.poll(index);
        settle(tier, index);
        auto image = index.reserve_host_image(false, kAny);
        if (!image) { break; }
        reservations.push_back(std::move(*image));
    }
    for (HostImageReservation& reservation : reservations) { index.release_host_image(std::move(reservation)); }
    require(index.snapshot(a.endpoint).host == CopyState::Absent && index.snapshot(a.endpoint).ssd == CopyState::Resident,
            "the endpoint is on the SSD only");
    index.check_invariants();

    misaligned = true;
    require(tier.promote(index, a.endpoint) == SsdTier::Promotion::Unavailable,
            "a promotion whose read cannot be submitted is unavailable");
    require(index.valid(a.endpoint) && index.snapshot(a.endpoint).ssd == CopyState::Resident &&
                index.snapshot(a.endpoint).host == CopyState::Absent,
            "...and the entry keeps its SSD copies (nothing was read)");
    index.check_invariants();

    misaligned = false;
    require(tier.promote(index, a.endpoint) == SsdTier::Promotion::Pending, "a later promotion starts");
    bool landed     = false;
    const auto stop = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (!landed && std::chrono::steady_clock::now() < stop) {
        landed = tier.poll(index);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(landed && tier.promote(index, a.endpoint) == SsdTier::Promotion::Ready, "...and reads the copies back");
    const SnapshotView image = index.snapshot(a.endpoint);
    for (std::size_t k = 0; k < image.host_slabs.size(); ++k) {
        require(holds(backend, image.host_slabs[k], 1000U + a.endpoint.index * 8U + static_cast<std::uint32_t>(k)),
                "the promoted image holds its bytes");
    }
    index.check_invariants();
}

// The prefix cache file's binding for a Host tier alone; `slab` maps a slab id to its memory.
PersistBinding host_binding(std::function<std::byte*(std::uint32_t)> slab, std::uint32_t image_slabs,
                            std::uint32_t host_slabs = kHostSlabs) {
    PersistBinding binding;
    binding.magic       = {'T', 'E', 'S', 'T', 'P', 'C', '0', '1'};
    binding.geometry    = {std::byte{1}, std::byte{2}, std::byte{3}};
    binding.slab_bytes  = kSlabBytes;
    binding.image_slabs = image_slabs;
    binding.host_slabs  = host_slabs;
    binding.slab        = std::move(slab);
    return binding;
}

// The size of a prefix cache file's head (its tables, padded to 4 KiB): the 64-bit field after the magic
// and the version.
std::uint64_t head_bytes_of(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    in.seekg(12);
    std::uint64_t bytes = 0;
    in.read(reinterpret_cast<char*>(&bytes), sizeof(bytes));
    require(static_cast<bool>(in), "the file's head size reads");
    return bytes;
}

void flip_byte(const std::filesystem::path& file, std::uint64_t offset) {
    std::fstream damage(file, std::ios::in | std::ios::out | std::ios::binary);
    damage.seekg(static_cast<std::streamoff>(offset));
    char byte = 0;
    damage.read(&byte, 1);
    byte = static_cast<char>(byte ^ 0x5A);
    damage.seekp(static_cast<std::streamoff>(offset));
    damage.write(&byte, 1);
    require(static_cast<bool>(damage), "the file is damaged for the test");
}

// The Host slabs of an image, in order.
std::vector<std::byte> image_bytes(const PrefixCacheIndex& index, SnapshotRef snapshot,
                                   const std::function<std::byte*(std::uint32_t)>& slab) {
    std::vector<std::byte> out;
    for (const std::uint32_t id : index.snapshot(snapshot).host_slabs) { out.insert(out.end(), slab(id), slab(id) + kSlabBytes); }
    return out;
}

// The prefix cache file through its unbuffered writer and reader (§5.5): the slabs go to and from the Host
// tier's own memory, here in chunks of three slabs laid out in reverse (so runs join inside a chunk and
// break between chunks), behind tables whose size is no multiple of 4 KiB; every block and image comes
// back with its bytes, and the file is whole 4 KiB blocks.
void test_file_round_trip(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    const std::filesystem::path file = directory / "cache.bin";
    const std::string fingerprint    = "a fingerprint of odd length";
    const auto chunked = [](Backend& backend) {
        return [&backend](std::uint32_t id) { return id < 30U ? backend.slab((9U - id / 3U) * 3U + id % 3U) : backend.slab(id); };
    };
    std::vector<std::byte> image_a, image_b;
    std::uint32_t frontier_a = 0;
    {
        Backend backend;
        PrefixCacheIndex index(config(), backend);
        const Conversation a = publish(index, backend, 21, 2);
        const Conversation b = publish(index, backend, 22, 3);
        // The bytes as this layout holds them: every slab in use keyed by its id.
        std::vector<NodeRef> nodes;
        std::vector<SnapshotRef> snapshots;
        index.collect_persistable(nodes, snapshots, false);
        const auto slab = chunked(backend);
        const auto write = [&](std::uint32_t id) {
            std::byte* p = slab(id);
            for (std::size_t i = 0; i < kSlabBytes; ++i) { p[i] = pattern(5000U + id, i); }
        };
        for (const NodeRef node : nodes) { write(index.node(node).host_slab); }
        for (const SnapshotRef snapshot : snapshots) {
            for (const std::uint32_t id : index.snapshot(snapshot).host_slabs) { write(id); }
        }
        image_a    = image_bytes(index, a.endpoint, slab);
        image_b    = image_bytes(index, b.endpoint, slab);
        frontier_a = index.snapshot(a.endpoint).frontier;
        const PersistResult saved = save_prefix_cache(index, host_binding(slab, 4), file, fingerprint, {});
        require(saved.ok && saved.blocks == 5 && saved.snapshots == 2, "the Host tier is saved (" + saved.message + ")");
        const std::uint64_t head = head_bytes_of(file);
        require(head % 4096 == 0 && std::filesystem::file_size(file) == head + 13 * kSlabBytes && saved.bytes == head + 13 * kSlabBytes,
                "the file is its padded tables and its 13 slabs");
        require(!std::filesystem::exists(file.string() + ".tmp"), "the temporary file was renamed");
    }
    {
        Backend backend;
        PrefixCacheIndex index(config(), backend);
        const auto slab            = chunked(backend);
        const PersistResult loaded = load_prefix_cache(index, host_binding(slab, 4), file, fingerprint, {});
        require(loaded.ok && loaded.message.empty() && loaded.blocks == 5 && loaded.snapshots == 2,
                "everything comes back, every slab matching its hash (" + loaded.message + ")");
        index.check_invariants();
        std::vector<NodeRef> nodes;
        std::vector<SnapshotRef> snapshots;
        index.collect_persistable(nodes, snapshots, false);
        require(snapshots.size() == 2, "both snapshots are restored");
        for (const SnapshotRef snapshot : snapshots) {
            const std::vector<std::byte>& expected = index.snapshot(snapshot).frontier == frontier_a ? image_a : image_b;
            require(image_bytes(index, snapshot, slab) == expected, "a restored image holds its bytes");
        }
        // Saved again from the restored tier, the file is the same size (the same entries).
        const PersistResult again = save_prefix_cache(index, host_binding(slab, 4), file, fingerprint, {});
        require(again.ok && again.bytes == loaded.bytes, "a restored tier saves the same file size");
        Backend other;
        PrefixCacheIndex fresh(config(), other);
        const PersistResult foreign =
            load_prefix_cache(fresh, host_binding([&](std::uint32_t id) { return other.slab(id); }, 4), file, "another", {});
        require(!foreign.ok && fresh.stats().nodes == 0, "a file of another fingerprint loads nothing");
    }
}

// A Host slab whose bytes changed on the drive (§5.5): its content hash no longer matches, and the entry
// it belongs to is dropped while the rest of the file loads; damage to the tables refuses the whole file.
void test_file_damage(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    const std::filesystem::path file = directory / "cache.bin";
    std::vector<std::byte> image_a, image_b;
    std::uint32_t frontier_a = 0;
    {
        Backend backend;
        PrefixCacheIndex index(config(), backend);
        const auto slab      = [&](std::uint32_t id) { return backend.slab(id); };
        const Conversation a = publish(index, backend, 31, 2);
        const Conversation b = publish(index, backend, 32, 3);
        image_a              = image_bytes(index, a.endpoint, slab);
        image_b              = image_bytes(index, b.endpoint, slab);
        frontier_a           = index.snapshot(a.endpoint).frontier;
        const PersistResult saved = save_prefix_cache(index, host_binding(slab, 4), file, "fp", {});
        require(saved.ok && saved.snapshots == 2, "two conversations are saved (" + saved.message + ")");
    }
    const std::uint64_t head = head_bytes_of(file);
    const auto load = [&](const char* what, bool ok, std::uint64_t snapshots) {
        Backend backend;
        PrefixCacheIndex index(config(), backend);
        const auto slab            = [&](std::uint32_t id) { return backend.slab(id); };
        const PersistResult loaded = load_prefix_cache(index, host_binding(slab, 4), file, "fp", {});
        require(loaded.ok == ok, std::string(what) + ": " + (ok ? "the file still loads (" : "the file is refused (") + loaded.message + ")");
        if (!ok) { return; }
        require(loaded.snapshots == snapshots && loaded.saved_snapshots == 2 && !loaded.message.empty(),
                std::string(what) + ": the damaged entry is dropped and reported, the other kept (" +
                    std::to_string(loaded.snapshots) + " snapshots)");
        index.check_invariants();
        std::vector<NodeRef> nodes;
        std::vector<SnapshotRef> kept;
        index.collect_persistable(nodes, kept, false);
        require(kept.size() == snapshots, std::string(what) + ": the index holds what the result counts");
        for (const SnapshotRef snapshot : kept) {
            const std::vector<std::byte>& expected = index.snapshot(snapshot).frontier == frontier_a ? image_a : image_b;
            require(image_bytes(index, snapshot, slab) == expected, std::string(what) + ": a kept image holds its bytes");
        }
    };
    // The file's last slab is the last snapshot's last image slab: that snapshot goes, alone.
    flip_byte(file, std::filesystem::file_size(file) - 100);
    load("a damaged image slab", true, 1);
    flip_byte(file, std::filesystem::file_size(file) - 100); // restored
    // The first slab is a root block's: it goes with what resumed through it, the other conversation stays.
    flip_byte(file, head + 100);
    load("a damaged block slab", true, 1);
    flip_byte(file, head + 100);
    // A damaged end mark (the last bytes of the tables, before their padding) refuses the file.
    {
        std::ifstream in(file, std::ios::binary);
        std::vector<char> tables(static_cast<std::size_t>(head));
        in.read(tables.data(), static_cast<std::streamsize>(tables.size()));
        std::size_t end = tables.size();
        while (end > 0 && tables[end - 1] == 0) { --end; }
        require(end > 16, "the tables end in their end mark");
        in.close();
        flip_byte(file, end - 10);
        load("damaged tables", false, 0);
        flip_byte(file, end - 10);
        // So does a flipped byte anywhere inside them (a block's tokens or lookup hash, a snapshot's fields): the
        // tables carry their own hash, so a block is never restored under another identity (audit 2026-11-17 PC-04).
        for (const std::size_t at : {end / 2, end * 3 / 4, std::size_t{64}}) {
            flip_byte(file, at);
            load("a flipped byte inside the tables", false, 0);
            flip_byte(file, at);
        }
    }
    // A truncated file is refused before anything is read.
    std::filesystem::resize_file(file, std::filesystem::file_size(file) - kSlabBytes);
    load("a truncated file", false, 0);
}

// A file saved with SSD copies, loaded by a start without the SSD tier into a Host tier smaller than the
// file (§5.5): a snapshot whose path has a block only the SSD held cannot come back, so it takes none of
// the Host tier, and every snapshot that can come back and fits does.
void test_restore_without_ssd(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    const std::filesystem::path kept = directory / "kept.bin";
    const std::filesystem::path file = directory / "cache.bin";
    std::uint32_t frontier_c = 0;
    {
        Backend backend;
        PrefixCacheIndex index(config(), backend);
        SsdTier tier(directory, 64, kSlabBytes, [&](std::uint32_t slab) { return backend.slab(slab); }, 0, false, kept);
        // x: four blocks, a tap after two and an endpoint; c: two blocks and an endpoint. Only the tap is
        // written back (the endpoints are pinned meanwhile), with x's first two blocks.
        const Conversation x = publish(index, backend, 51, 4, 2);
        const Conversation c = publish(index, backend, 52, 2);
        frontier_c           = index.snapshot(c.endpoint).frontier;
        index.pin_snapshot(x.endpoint);
        index.pin_snapshot(c.endpoint);
        for (int i = 0; i < 20 && tier.counters().image_writes == 0; ++i) {
            (void)tier.poll(index);
            settle(tier, index);
        }
        index.unpin_snapshot(x.endpoint);
        index.unpin_snapshot(c.endpoint);
        require(tier.counters().image_writes == 1 && index.snapshot(x.tap).ssd == CopyState::Resident,
                "the tap and its two blocks were written back");
        // Captures displace what the SSD holds: the tap's Host image, then x's first two blocks' Host copies.
        std::vector<HostImageReservation> reservations;
        while (index.node(x.path[0]).host == CopyState::Resident) {
            auto image = index.reserve_host_image(false, kAny);
            require(image.has_value(), "a capture displaces clean Host copies");
            reservations.push_back(std::move(*image));
        }
        require(index.snapshot(x.endpoint).host == CopyState::Resident && index.node(x.path[2]).host == CopyState::Resident &&
                    index.snapshot(c.endpoint).host == CopyState::Resident,
                "the unwritten endpoints and x's last blocks are still on the Host");
        for (HostImageReservation& reservation : reservations) { index.release_host_image(std::move(reservation)); }
        tier.drain(index);
        const PersistResult saved = save_prefix_cache(index, test_binding(backend, tier, 4), file, "fp", {});
        require(saved.ok && saved.snapshots == 3 && saved.ssd_snapshots == 1 && saved.ssd_blocks == 2,
                "x's endpoint is saved over two blocks only the SSD holds (" + saved.message + ")");
    }
    for (const std::uint32_t host_slabs : {8U, kHostSlabs}) {
        Backend backend;
        PrefixCacheIndex index(config(), backend);
        const PersistResult loaded = load_prefix_cache(
            index, host_binding([&](std::uint32_t id) { return backend.slab(id); }, 4, host_slabs), file, "fp", {});
        require(loaded.ok && !loaded.ssd_message.empty(), "the file loads without its SSD copies (" + loaded.message + ")");
        require(loaded.snapshots == 1 && loaded.blocks == 2,
                "the snapshot that can come back does, in a Host tier of " + std::to_string(host_slabs) + " slabs (" +
                    std::to_string(loaded.snapshots) + " snapshots, " + std::to_string(loaded.blocks) + " blocks)");
        require(index.snapshot(only_snapshot(index)).frontier == frontier_c, "it is the conversation the Host tier held whole");
        index.check_invariants();
    }
}

} // namespace

int main() {
    const std::filesystem::path directory = infernix::test::unique_temp_path("ssd_tier");
    std::filesystem::create_directories(directory);
    int status = 0;
    try {
        test_write_budget(directory);
        test_codec();
        if (const char* dump = std::getenv("INFERNIX_TEST_CODEC_DUMP"); dump != nullptr && *dump != '\0') { bench_codec(dump); }
        test_packed_tier(directory / "packed");
        test_packed_partial(directory / "partial");
        test_kept_restart(directory / "kept");
        test_room_wait(directory / "room");
        test_idle_retry(directory / "retry");
        test_refused_submit(directory / "refused");
        test_refused_promotion(directory / "refused-promotion");
        test_file_round_trip(directory / "file");
        test_file_damage(directory / "damage");
        test_restore_without_ssd(directory / "no-ssd");
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
