// Qwen4Exp prefix-cache state images and page records (design §19.3.1, test X8), on synthetic Device
// pools of the real Qwen3.8-Flash-Next geometry with no model: random lane state is captured to
// packed Device slots and to Host images split over shuffled slabs, restored into other lanes group
// by group, and must come back byte for byte without touching any other lane. A KV page moved to a
// Host record and back into a different page per KV layer must also be byte-exact. A DFlash2
// drafter's context ring (five layers of one lane's K and V) rides in the drafter group.

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "core/host_kv_arena.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "models/qwen4_exp/program/prefix/state_image.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::models::qwen4_exp;
using namespace infernix::models::qwen4_exp::prefix;

namespace {

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

template <class F>
void require_throws(F&& f, const std::string& message) {
    try {
        f();
    } catch (const std::exception&) { return; }
    throw std::runtime_error("expected an exception: " + message);
}

bool cuda_unavailable() {
    int n               = 0;
    const cudaError_t e = cudaGetDeviceCount(&n);
    return e != cudaSuccess || n == 0;
}

// Qwen3.8-Flash-Next's text geometry: 48 layers, three GDN layers then one attention layer.
TextConfig flash_next() {
    TextConfig c;
    c.hidden_size       = 2560;
    c.num_hidden_layers = 48;
    for (std::uint32_t l = 0; l < c.num_hidden_layers; ++l) {
        c.layer_types.push_back(l % 4 == 3 ? MixerKind::Attention : MixerKind::Gdn);
    }
    c.attention_layers   = 12;
    c.gdn_layers         = 36;
    c.attention.heads    = 16;
    c.attention.kv_heads = 2;
    c.attention.head_dim = 256;
    c.gdn.key_heads      = 16;
    c.gdn.key_head_dim   = 128;
    c.gdn.value_heads    = 48;
    c.gdn.value_head_dim = 128;
    c.gdn.conv_kernel    = 4;
    c.hc.streams         = 4;
    c.qsa.index_head_dim = 128;
    c.qsa.compress_ratio = 4;
    c.ple.layer          = 1;
    c.ple.conv_kernel    = 4;
    c.ple.conv_dilation  = 3;
    return c;
}

void fill_random(void* device, std::size_t bytes, std::uint32_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<std::uint64_t> host((bytes + 7) / 8);
    for (auto& v : host) { v = rng(); }
    CUDA_CHECK(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice));
}

// Every non-header part of a lane, concatenated.
std::vector<std::byte> lane_bytes(const LaneStateImage& image, std::uint32_t lane) {
    std::vector<std::byte> out;
    for (std::size_t p = 1; p < image.layout().parts.size(); ++p) {
        const StateImagePart& part = image.layout().parts[p];
        const std::size_t at       = out.size();
        out.resize(at + part.bytes);
        CUDA_CHECK(cudaMemcpy(out.data() + at, image.device_part(part, lane), part.bytes, cudaMemcpyDeviceToHost));
    }
    return out;
}

// Pinned slabs handed out in a shuffled order, as the prefix cache's pool does.
struct HostSlabs {
    HostSlabs(std::size_t slabs, std::size_t slab_bytes, std::size_t used, std::uint32_t seed)
        : pool(slabs * slab_bytes), bytes(slab_bytes) {
        std::vector<std::size_t> order(slabs);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), std::mt19937(seed));
        for (std::size_t i = 0; i < used; ++i) {
            segments.push_back(static_cast<std::byte*>(pool.data()) + order[i] * slab_bytes);
        }
        const_segments.assign(segments.begin(), segments.end());
    }

    HostImage image() const { return HostImage{segments, bytes}; }
    HostImageConst const_image() const { return HostImageConst{const_segments, bytes}; }

    PinnedHostBuffer pool;
    std::size_t bytes = 0;
    std::vector<std::byte*> segments;
    std::vector<const std::byte*> const_segments;
};

void test_layout(const TextConfig& c) {
    const StateImageLayout layout = plan_state_image(state_image_spec(c, true, KvCacheStorage::Int8Group64));
    // Design §19.3.1: 115,673,088 bytes per image with MTP (124 slabs of a 931,840-byte block).
    require(layout.image_bytes == 115'673'088ULL, "image bytes " + std::to_string(layout.image_bytes));
    require(layout.groups() == c.num_hidden_layers + 1U, "one group per decoder layer plus MTP");
    require(layout.parts.front().kind == StateImagePart::Kind::Header && layout.parts.front().offset == 0,
            "the header leads the image");
    std::size_t end = 0;
    for (const StateImagePart& part : layout.parts) {
        require(part.offset % kStateImageAlignment == 0 && part.offset >= end, "parts are aligned and ordered");
        end = part.offset + part.bytes;
    }
    require(end <= layout.image_bytes, "parts fit the image");
    std::uint32_t gdn = 0, attention = 0;
    for (std::uint32_t g = 0; g < c.num_hidden_layers; ++g) {
        std::uint32_t p = layout.group_begin[g];
        if (g == c.ple.layer) {
            require(layout.parts[p].kind == StateImagePart::Kind::Ple, "the PLE history leads its layer's group");
            ++p;
        }
        if (c.layer_types[g] == MixerKind::Gdn) {
            require(layout.group_begin[g + 1] - p == 2 && layout.parts[p].kind == StateImagePart::Kind::GdnRecurrent &&
                        layout.parts[p + 1].kind == StateImagePart::Kind::GdnConv && layout.parts[p].index == gdn &&
                        layout.parts[p + 1].index == gdn,
                    "a GDN layer's group is its recurrent then conv state");
            ++gdn;
        } else {
            require(layout.group_begin[g + 1] - p == 1 && layout.parts[p].kind == StateImagePart::Kind::QsaTail &&
                        layout.parts[p].index == attention,
                    "an attention layer's group is its QSA tail");
            ++attention;
        }
    }
    const std::uint32_t m = layout.group_begin[layout.drafter_group()];
    require(layout.group_begin[layout.drafter_group() + 1] - m == 2 && layout.parts[m].kind == StateImagePart::Kind::MtpSaved &&
                layout.parts[m + 1].kind == StateImagePart::Kind::MtpTail,
            "the MTP group is the saved residual then the MTP tail");
    const StateImageLayout plain = plan_state_image(state_image_spec(c, false, KvCacheStorage::Int8Group64));
    require(plain.group_begin[plain.drafter_group()] == plain.group_begin[plain.drafter_group() + 1] &&
                plain.fingerprint != layout.fingerprint && plain.image_bytes < layout.image_bytes,
            "without MTP the MTP group is empty and the layout differs");
}

void test_round_trips(const TextConfig& c) {
    constexpr std::uint32_t kLanes = 3;
    const StateImageLayout layout  = plan_state_image(state_image_spec(c, true, KvCacheStorage::Int8Group64));
    const StateImageSpec& spec     = layout.spec;

    LayoutBuilder builder;
    const LinearAttentionStatePoolSpec gdn_spec{.layers         = c.gdn_layers,
                                                .conv_channels  = static_cast<std::int32_t>(c.gdn.conv_channels()),
                                                .conv_width     = static_cast<std::int32_t>(c.gdn.conv_kernel - 1),
                                                .value_heads    = static_cast<std::int32_t>(c.gdn.value_heads),
                                                .value_head_dim = static_cast<std::int32_t>(c.gdn.value_head_dim),
                                                .key_head_dim   = static_cast<std::int32_t>(c.gdn.key_head_dim),
                                                .slot_count     = static_cast<std::int32_t>(kLanes)};
    const auto gdn_layout = plan_linear_attention_state_pool(builder, gdn_spec);
    DeviceBuffer state(builder.finish(256));
    LinearAttentionStatePool gdn(DeviceSpan{state.p, state.bytes}, gdn_layout);
    DeviceBuffer ple(spec.ple_bytes * kLanes);
    DeviceBuffer tails(spec.qsa_tail_bytes * kLanes * (c.attention_layers + 1U));
    DeviceBuffer saved(spec.mtp_saved_bytes * kLanes);
    const auto randomize = [&](std::uint32_t seed) {
        fill_random(state.p, state.bytes, seed);
        fill_random(ple.p, ple.bytes, seed + 1);
        fill_random(tails.p, tails.bytes, seed + 2);
        fill_random(saved.p, saved.bytes, seed + 3);
    };
    randomize(100);
    const LaneStateImage image(layout, LaneStateBuffers{.gdn       = &gdn,
                                                        .lanes     = kLanes,
                                                        .ple       = static_cast<std::byte*>(ple.p),
                                                        .tails     = static_cast<std::byte*>(tails.p),
                                                        .mtp_saved = static_cast<std::byte*>(saved.p)});
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Lane 0 to a Host image over shuffled slabs, then into lane 1 group by group, last group first.
    constexpr std::size_t kSlab = 933'888; // one 931,840-byte block bundle, 4 KiB aligned
    const std::size_t used      = (layout.image_bytes + kSlab - 1) / kSlab;
    const auto want             = lane_bytes(image, 0);
    const auto lane2            = lane_bytes(image, 2);
    HostSlabs a(used + 9, kSlab, used, 7);
    image.copy_lane_to_host(0, a.image(), stream);
    StateImageHeader header;
    header.frontier     = 4242;
    header.mtp_written  = true;
    header.mtp_next     = 151643;
    header.lineage_echo = true;
    header.opener       = true;
    header.mtp_accept   = {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2};
    write_state_image_header(layout, header, a.segments[0]);
    for (std::uint32_t g = layout.groups(); g-- > 0;) { image.copy_group_from_host(g, a.const_image(), 1, stream); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    require(lane_bytes(image, 1) == want, "a Host image restores the lane byte for byte");
    require(lane_bytes(image, 2) == lane2, "a restore touches no other lane");
    const auto read = read_state_image_header(layout, a.segments[0]);
    require(read && read->frontier == header.frontier && read->mtp_written && read->mtp_next == header.mtp_next &&
                read->lineage_echo && read->opener && read->mtp_accept == header.mtp_accept,
            "the header round-trips");

    // Lane 0 to a packed Device slot, written through to a second Host image, restored into lane 2.
    DeviceBuffer packed(layout.image_bytes);
    packed.fill(0);
    image.copy_lane_to_packed(0, static_cast<std::byte*>(packed.p), stream);
    HostSlabs b(used + 3, kSlab, used, 11);
    image.copy_packed_to_host(static_cast<const std::byte*>(packed.p), b.image(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    require(!read_state_image_header(layout, b.segments[0]), "a packed slot carries no header");
    for (std::uint32_t g = 0; g < layout.groups(); ++g) { image.copy_group_from_host(g, b.const_image(), 2, stream); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    require(lane_bytes(image, 2) == want, "slot write-through restores byte for byte");

    // A packed slot straight into a lane whose state changed meanwhile.
    fill_random(ple.p, ple.bytes, 999);
    fill_random(saved.p, saved.bytes, 998);
    const auto lane0_now = lane_bytes(image, 0);
    for (std::uint32_t g = 0; g < layout.groups(); ++g) {
        image.copy_group_from_packed(g, static_cast<const std::byte*>(packed.p), 1, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    require(lane_bytes(image, 1) == want, "a packed slot restores byte for byte");
    require(lane_bytes(image, 0) == lane0_now, "the source lane is untouched");

    // Header validation and argument checks.
    write_state_image_header(layout, header, b.segments[0]);
    require(read_state_image_header(layout, b.segments[0]).has_value(), "a written header is read back");
    b.segments[0][8] ^= std::byte{1};
    require(!read_state_image_header(layout, b.segments[0]), "a damaged header is rejected");
    require(!read_state_image_header(plan_state_image(state_image_spec(c, false, KvCacheStorage::Int8Group64)), a.segments[0]),
            "a header of another layout is rejected");
    const HostSlabs short_image(used - 1, kSlab, used - 1, 3);
    require_throws([&] { image.copy_lane_to_host(0, short_image.image(), stream); }, "too few segments");
    require_throws([&] { (void)image.device_part(layout.parts[1], kLanes); }, "lane out of range");
    require_throws([&] { image.copy_group_from_host(layout.groups(), a.const_image(), 0, stream); }, "group out of range");
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::printf("state image: %zu bytes in %zu parts, %u groups, %zu slabs; Host, packed and write-through "
                "round trips byte-exact\n",
                layout.image_bytes, layout.parts.size(), layout.groups(), used);
}

// Qwen3.8-Flash-Next's DFlash2 drafter: five sliding layers of 8 KV heads x 128 over a 2,048 ring.
CyclicKVCacheLayout draft_ring_layout(LayoutBuilder& builder, std::uint32_t lanes) {
    return plan_cyclic_kv_cache(builder, 5, 2048, 8, 128, static_cast<std::int32_t>(lanes));
}

DraftRingGeometry draft_ring_geometry(const CyclicKVCacheLayout& ring, std::uint32_t lanes) {
    return {.layers  = static_cast<std::uint32_t>(ring.k.size()),
            .k_bytes = ring.k.front().region.bytes / lanes,
            .v_bytes = ring.v.front().region.bytes / lanes};
}

void test_draft_ring_layout(const TextConfig& c) {
    LayoutBuilder builder;
    const DraftRingGeometry ring = draft_ring_geometry(draft_ring_layout(builder, 1), 1);
    require(ring.k_bytes == 128ULL * 2048 * 8 * 2 && ring.v_bytes == ring.k_bytes, "4 MiB of K and of V per layer");
    const StateImageLayout layout = plan_state_image(state_image_spec(c, false, KvCacheStorage::Int8Group64, ring));
    const StateImageLayout plain  = plan_state_image(state_image_spec(c, false, KvCacheStorage::Int8Group64));
    const std::uint32_t d         = layout.group_begin[layout.drafter_group()];
    require(layout.groups() == c.num_hidden_layers + 1U && layout.group_begin[layout.drafter_group() + 1] - d == 10,
            "the drafter group holds the ring's ten planes");
    for (std::uint32_t l = 0; l < 5; ++l) {
        const StateImagePart& k = layout.parts[d + 2 * l];
        const StateImagePart& v = layout.parts[d + 2 * l + 1];
        require(k.kind == StateImagePart::Kind::DraftRingK && v.kind == StateImagePart::Kind::DraftRingV && k.index == l &&
                    v.index == l && k.bytes == ring.k_bytes && v.bytes == ring.v_bytes,
                "each drafter layer contributes its K then its V");
    }
    require(layout.image_bytes >= plain.image_bytes + 10 * ring.k_bytes && layout.fingerprint != plain.fingerprint,
            "the ring grows the image by its 40 MiB and changes the layout");
    require_throws([&] { (void)state_image_spec(c, true, KvCacheStorage::Int8Group64, ring); }, "MTP and DFlash2 together");
}

// A lane's ring through a Host image into another lane, byte for byte, with the rest of its state.
void test_draft_ring_round_trip(const TextConfig& c) {
    constexpr std::uint32_t kLanes = 3;
    LayoutBuilder ring_builder;
    const CyclicKVCacheLayout ring_layout = draft_ring_layout(ring_builder, kLanes);
    DeviceBuffer ring_backing(ring_builder.finish(256));
    const CyclicKVCache ring(DeviceSpan{ring_backing.p, ring_backing.bytes}, ring_layout);
    const StateImageLayout layout =
        plan_state_image(state_image_spec(c, false, KvCacheStorage::Int8Group64, draft_ring_geometry(ring_layout, kLanes)));
    const StateImageSpec& spec = layout.spec;
    LayoutBuilder builder;
    const LinearAttentionStatePoolSpec gdn_spec{.layers         = c.gdn_layers,
                                                .conv_channels  = static_cast<std::int32_t>(c.gdn.conv_channels()),
                                                .conv_width     = static_cast<std::int32_t>(c.gdn.conv_kernel - 1),
                                                .value_heads    = static_cast<std::int32_t>(c.gdn.value_heads),
                                                .value_head_dim = static_cast<std::int32_t>(c.gdn.value_head_dim),
                                                .key_head_dim   = static_cast<std::int32_t>(c.gdn.key_head_dim),
                                                .slot_count     = static_cast<std::int32_t>(kLanes)};
    const auto gdn_layout = plan_linear_attention_state_pool(builder, gdn_spec);
    DeviceBuffer state(builder.finish(256));
    LinearAttentionStatePool gdn(DeviceSpan{state.p, state.bytes}, gdn_layout);
    DeviceBuffer ple(spec.ple_bytes * kLanes);
    DeviceBuffer tails(spec.qsa_tail_bytes * kLanes * c.attention_layers);
    fill_random(state.p, state.bytes, 200);
    fill_random(ple.p, ple.bytes, 201);
    fill_random(tails.p, tails.bytes, 202);
    fill_random(ring_backing.p, ring_backing.bytes, 203);
    std::vector<std::byte*> ring_k, ring_v;
    for (std::uint32_t l = 0; l < ring.layer_count(); ++l) {
        ring_k.push_back(static_cast<std::byte*>(ring.layer_view(l).k.data));
        ring_v.push_back(static_cast<std::byte*>(ring.layer_view(l).v.data));
    }
    require_throws(
        [&] {
            (void)LaneStateImage(layout, LaneStateBuffers{.gdn   = &gdn,
                                                          .lanes = kLanes,
                                                          .ple   = static_cast<std::byte*>(ple.p),
                                                          .tails = static_cast<std::byte*>(tails.p)});
        },
        "a ring layout without the ring's buffers");
    const LaneStateImage image(layout, LaneStateBuffers{.gdn    = &gdn,
                                                        .lanes  = kLanes,
                                                        .ple    = static_cast<std::byte*>(ple.p),
                                                        .tails  = static_cast<std::byte*>(tails.p),
                                                        .ring_k = ring_k,
                                                        .ring_v = ring_v});
    // Each ring part is that lane's own slice of its layer's plane.
    const std::uint32_t d = layout.group_begin[layout.drafter_group()];
    require(image.device_part(layout.parts[d + 2], 2) ==
                static_cast<std::byte*>(ring.layer_view(1).k.data) + 2 * spec.draft_ring.k_bytes,
            "lane 2's layer-1 K is its slice of the plane");
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    constexpr std::size_t kSlab = 933'888;
    const std::size_t used      = (layout.image_bytes + kSlab - 1) / kSlab;
    const auto want             = lane_bytes(image, 0);
    const auto lane2            = lane_bytes(image, 2);
    HostSlabs host(used + 5, kSlab, used, 13);
    image.copy_lane_to_host(0, host.image(), stream);
    StateImageHeader header;
    header.frontier       = 9000;
    header.draft_frontier = 8192;
    write_state_image_header(layout, header, host.segments[0]);
    for (std::uint32_t g = 0; g < layout.groups(); ++g) { image.copy_group_from_host(g, host.const_image(), 1, stream); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    require(lane_bytes(image, 1) == want, "the ring and the lane state restore byte for byte");
    require(lane_bytes(image, 2) == lane2, "a ring restore touches no other lane");
    const auto read = read_state_image_header(layout, host.segments[0]);
    require(read && read->frontier == 9000 && read->draft_frontier == 8192, "the DFlash2 frontier round-trips");
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::printf("DFlash2 ring: %zu-byte image, %zu ring bytes per lane, Host round trip byte-exact\n",
                layout.image_bytes, 5 * (spec.draft_ring.k_bytes + spec.draft_ring.v_bytes));
}

void test_page_records(const TextConfig& c, KvCacheStorage storage, std::size_t expected_stride) {
    const KvPageGeometry geometry = kv_page_geometry(c, storage, true);
    require(geometry.kv_layers == c.attention_layers + 1U &&
                geometry.geometry.planes.size() == static_cast<std::size_t>(geometry.kv_layers) * geometry.planes_per_layer,
            "every KV layer has the same planes");
    LayoutBuilder builder;
    const auto pool_layout = plan_device_kv_page_pool(builder, {.page_group_count = 8, .geometry = geometry.geometry});
    DeviceBuffer backing(builder.finish(256));
    fill_random(backing.p, backing.bytes, 5);
    DeviceKVPagePool pool(DeviceSpan{backing.p, backing.bytes}, pool_layout);
    const HostKVPageLayout host = plan_host_kv_page_layout(geometry.geometry);
    require(host.page_stride == expected_stride, "page record bytes " + std::to_string(host.page_stride));
    auto reservation = pool.reserve(8);
    require(reservation.has_value(), "pages");
    std::vector<DeviceKVPageLease> leases;
    leases.reserve(8); // materialize never reallocates its destination
    pool.materialize(*reservation, 8, leases);
    PinnedHostBuffer records(3 * host.page_stride);
    std::byte* const rec[3] = {static_cast<std::byte*>(records.data()),
                               static_cast<std::byte*>(records.data()) + host.page_stride,
                               static_cast<std::byte*>(records.data()) + 2 * host.page_stride};
    const DeviceKVPageHandle source = leases[2].handle(), target = leases[5].handle(), other = leases[6].handle();
    pool.copy_to_host_records(std::span(&source, 1), std::span(&rec[0], 1), {}, host, nullptr);
    pool.copy_to_host_records(std::span(&other, 1), std::span(&rec[2], 1), {}, host, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::byte> other_before(rec[2], rec[2] + host.page_stride);
    // Into a different page one KV layer at a time, MTP first.
    const std::byte* const from[1] = {rec[0]};
    for (std::uint32_t k = geometry.kv_layers; k-- > 0;) {
        const PlaneRange planes = kv_layer_planes(geometry, k);
        pool.copy_from_host_records(std::span(from, 1), {}, std::span(&target, 1), host, planes.begin, planes.end, nullptr);
    }
    pool.copy_to_host_records(std::span(&target, 1), std::span(&rec[1], 1), {}, host, nullptr);
    pool.copy_to_host_records(std::span(&other, 1), std::span(&rec[2], 1), {}, host, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    require(std::memcmp(rec[0], rec[1], host.page_stride) == 0, "a page moved by KV layers is byte-exact");
    require(std::memcmp(rec[2], other_before.data(), host.page_stride) == 0, "other pages are untouched");
    pool.dematerialize(*reservation, 0, leases);
    std::printf("page records (%s): %zu bytes, %u KV layers of %u planes, byte-exact per layer\n",
                storage == KvCacheStorage::Int8Group64 ? "int8" : "bf16", host.page_stride, geometry.kv_layers,
                geometry.planes_per_layer);
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        const TextConfig c = flash_next();
        test_layout(c);
        test_draft_ring_layout(c);
        if (cuda_unavailable()) {
            std::printf("SKIP: no usable CUDA device\n");
            return 77;
        }
        test_round_trips(c);
        test_draft_ring_round_trip(c);
        // 13 KV layers x 64 positions: INT8 K/V with FP16 group scales, pooled BF16 index keys.
        test_page_records(c, KvCacheStorage::Int8Group64, 931'840);
        // BF16 K and FP16 V: 2 x 2 heads x 256 x 2 bytes, plus 64 bytes of pooled keys, per position.
        test_page_records(c, KvCacheStorage::BFloat16, 13ULL * 64 * (2 * 2 * 256 * 2 + 64));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("qwen4_exp prefix state checks passed\n");
    return 0;
}
