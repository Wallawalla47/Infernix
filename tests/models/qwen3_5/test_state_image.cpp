#include "core/device.h"
#include "core/host_context_arena.h"
#include "models/qwen3_5/state/state_image.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace q36 = infernix::models::qwen3_5;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

struct PlannedPool {
    q36::StateImageDeviceLayout layout;
    std::size_t bytes = 0;
};

PlannedPool plan_pool(bool dflash, std::int32_t slots = 4, bool dflash2 = false,
                      bool window = false,
                      q36::RecurrentHostFormat recurrent_host = q36::RecurrentHostFormat::Fp32) {
    q36::StateImageSpec spec{
        .linear =
            {
                .layers         = 2,
                .conv_channels  = 5,
                .conv_width     = 3,
                .value_heads    = 2,
                .value_head_dim = 4,
                .key_head_dim   = 3,
                .slot_count     = slots,
                .conv_dtype     = infernix::DType::BF16,
            },
        .hidden = 7,
    };
    if (dflash) {
        spec.dflash_local = dflash2
                                ? q36::DFlashLocalStateSpec{.layers   = 5,
                                                            .capacity = 2048,
                                                            .kv_heads = 8,
                                                            .head_dim = 128}
                                : q36::DFlashLocalStateSpec{
                                      .layers = 2, .capacity = 17, .kv_heads = 2, .head_dim = 4};
    }
    if (window) { spec.kv_window = q36::KVWindowStateSpec{.layers = 3, .kv_heads = 2}; }
    spec.recurrent_host = recurrent_host;
    infernix::LayoutBuilder builder;
    q36::StateImageDeviceLayout layout = q36::plan_state_image_device_pool(builder, spec);
    return {.layout = std::move(layout), .bytes = builder.finish(256)};
}

// The pool copies on the device's non-blocking stream, which does not wait for legacy-stream
// work, so a fill completes before the next pool operation is enqueued.
void set_bytes(const infernix::Tensor& tensor, unsigned char value) {
    CUDA_CHECK(cudaMemset(tensor.data, value, tensor.bytes()));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void expect_bytes(const infernix::Tensor& tensor, unsigned char expected, std::string_view label) {
    std::vector<unsigned char> host(tensor.bytes());
    CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
    for (const unsigned char value : host) {
        if (value != expected) {
            ++failures;
            std::cerr << "FAIL: " << label << " differs\n";
            return;
        }
    }
}

void fill_slot(q36::StateImageDevicePool& pool, std::int32_t slot, unsigned char base) {
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        set_bytes(pool.linear().conv_slot(layer, slot), static_cast<unsigned char>(base + layer));
        set_bytes(pool.linear().recurrent_slot(layer, slot),
                  static_cast<unsigned char>(base + 0x10 + layer));
    }
    set_bytes(pool.continuation_hidden_slot(slot), static_cast<unsigned char>(base + 0x20));
    if (infernix::CyclicKVCache* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            set_bytes(view.k.slice(3, slot, 1), static_cast<unsigned char>(base + 0x30 + layer));
            set_bytes(view.v.slice(3, slot, 1), static_cast<unsigned char>(base + 0x40 + layer));
        }
    }
    for (std::uint32_t layer = 0; pool.has_kv_window() && layer < 3; ++layer) {
        const infernix::PagedKVWindowView window = pool.kv_window_slot_view(layer, slot);
        unsigned char value = static_cast<unsigned char>(base + 0x50 + 8 * layer);
        for (const infernix::Tensor& plane :
             {window.k_codes, window.v_codes, window.k_scales, window.v_scales, window.tags}) {
            set_bytes(plane, value++);
        }
    }
}

void expect_slot(q36::StateImageDevicePool& pool, std::int32_t slot, unsigned char base,
                 std::string_view label) {
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        expect_bytes(pool.linear().conv_slot(layer, slot), static_cast<unsigned char>(base + layer),
                     label);
        expect_bytes(pool.linear().recurrent_slot(layer, slot),
                     static_cast<unsigned char>(base + 0x10 + layer), label);
    }
    expect_bytes(pool.continuation_hidden_slot(slot), static_cast<unsigned char>(base + 0x20),
                 label);
    if (infernix::CyclicKVCache* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            expect_bytes(view.k.slice(3, slot, 1), static_cast<unsigned char>(base + 0x30 + layer),
                         label);
            expect_bytes(view.v.slice(3, slot, 1), static_cast<unsigned char>(base + 0x40 + layer),
                         label);
        }
    }
    for (std::uint32_t layer = 0; pool.has_kv_window() && layer < 3; ++layer) {
        const infernix::PagedKVWindowView window = pool.kv_window_slot_view(layer, slot);
        unsigned char value = static_cast<unsigned char>(base + 0x50 + 8 * layer);
        for (const infernix::Tensor& plane :
             {window.k_codes, window.v_codes, window.k_scales, window.v_scales, window.tags}) {
            expect_bytes(plane, value++, label);
        }
    }
}

void expect_zero_slot(q36::StateImageDevicePool& pool, std::int32_t slot, std::string_view label) {
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        expect_bytes(pool.linear().conv_slot(layer, slot), 0, label);
        expect_bytes(pool.linear().recurrent_slot(layer, slot), 0, label);
    }
    expect_bytes(pool.continuation_hidden_slot(slot), 0, label);
    if (infernix::CyclicKVCache* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            expect_bytes(view.k.slice(3, slot, 1), 0, label);
            expect_bytes(view.v.slice(3, slot, 1), 0, label);
        }
    }
    // A cleared window is one whose tags are zero (slot tags are odd).
    for (std::uint32_t layer = 0; pool.has_kv_window() && layer < 3; ++layer) {
        expect_bytes(pool.kv_window_slot_view(layer, slot).tags, 0, label);
    }
}

void test_host_roundtrip(bool dflash, infernix::DeviceContext& device, bool dflash2 = false,
                         bool window = false) {
    PlannedPool planned = plan_pool(dflash, 2, dflash2, window);
    if (dflash2) {
        expect(q36::fork_local_transfer_work(planned.layout.host).payload_bytes ==
                   40ULL * 1024 * 1024,
               "DFlash2 local snapshot must transfer exactly 40 MiB");
    }
    infernix::DeviceArena arena(planned.bytes);
    q36::StateImageDevicePool pool({arena.base(), arena.capacity()}, planned.layout);
    fill_slot(pool, 0, dflash ? 0x19 : 0x25);
    pool.zero_slot(1, device.stream);

    infernix::HostContextArena host_backing(planned.layout.host.image_bytes,
                                          planned.layout.host.image_bytes);
    auto image = host_backing.allocate(planned.layout.host.image_bytes);
    expect(image.has_value(), "pinned Host backing holds one StateImage");
    pool.copy_to_host(0, {.data = image->data(), .layout = &planned.layout.host}, device.stream);
    device.synchronize();
    pool.copy_from_host({.data = image->data(), .layout = &planned.layout.host}, 1, device.stream);
    device.synchronize();
    expect_slot(pool, 1, dflash ? 0x19 : 0x25,
                dflash ? "DFlash Host roundtrip" : "common Host roundtrip");
}

// The exact KV window is a per-slot component: a layer view spans every slot, a slot view one;
// a fork copies it with DFlash local state but leaves the Linear Attention state alone.
void test_kv_window(infernix::DeviceContext& device) {
    PlannedPool planned = plan_pool(true, 3, false, true);
    infernix::DeviceArena arena(planned.bytes);
    q36::StateImageDevicePool pool({arena.base(), arena.capacity()}, planned.layout);
    expect(pool.has_kv_window() && pool.has_fork_local(), "StateImage KV window is present");
    const infernix::PagedKVWindowView all = pool.kv_window_view(2);
    expect(all.k_codes.ne[0] == 256 && all.k_codes.ne[1] == infernix::kKVWindowSlots &&
               all.k_codes.ne[2] == 2 && all.k_codes.ne[3] == 3 &&
               all.k_scales.ne[0] == infernix::kKVWindowGroups && all.tags.ne[0] == 2 &&
               all.tags.ne[3] == 3,
           "StateImage KV window layer view spans every slot");
    const infernix::PagedKVWindowView one = pool.kv_window_slot_view(2, 1);
    expect(one.k_codes.ne[3] == 1 &&
               static_cast<const std::byte*>(one.k_codes.data) ==
                   static_cast<const std::byte*>(all.k_codes.data) + one.k_codes.bytes(),
           "StateImage KV window slot view selects the slot");
    bool layer_rejected = false;
    try {
        (void)pool.kv_window_view(3);
    } catch (const std::out_of_range&) { layer_rejected = true; }
    expect(layer_rejected, "an out-of-range KV window layer was accepted");

    const infernix::TransferWork fork = q36::fork_local_transfer_work(planned.layout.host);
    expect(fork.payload_bytes == 2ULL * planned.layout.host.dflash_local_layer_bytes * 2 +
                                     planned.layout.host.kv_window->bytes &&
               fork.copy_operations == 2 * 2 + 5 * 3,
           "fork-local work covers DFlash local state and the KV window");

    fill_slot(pool, 0, 0x13);
    fill_slot(pool, 1, 0x61);
    pool.copy_fork_local(0, 1, device.stream);
    device.synchronize();
    for (std::uint32_t layer = 0; layer < 3; ++layer) {
        const infernix::PagedKVWindowView window = pool.kv_window_slot_view(layer, 1);
        unsigned char value = static_cast<unsigned char>(0x13 + 0x50 + 8 * layer);
        for (const infernix::Tensor& plane :
             {window.k_codes, window.v_codes, window.k_scales, window.v_scales, window.tags}) {
            expect_bytes(plane, value++, "fork copy moves the KV window");
        }
    }
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        expect_bytes(pool.linear().recurrent_slot(layer, 1),
                     static_cast<unsigned char>(0x61 + 0x10 + layer),
                     "fork copy leaves the Linear Attention state");
    }
    pool.copy_slot(0, 2, device.stream);
    pool.zero_slot(0, device.stream);
    device.synchronize();
    expect_slot(pool, 2, 0x13, "StateImage D2D copy carries the KV window");
    expect_zero_slot(pool, 0, "StateImage zero clears the KV window tags");
}

// Segmented Host images (hybrid prefix cache slabs): the packed image split into fixed-size
// segments at scattered addresses must hold exactly the contiguous packed bytes, and restore them.
// An odd segment size splits components mid-way.
void test_segmented_roundtrip(bool dflash, infernix::DeviceContext& device, bool dflash2 = false) {
    PlannedPool planned = plan_pool(dflash, 2, dflash2);
    infernix::DeviceArena arena(planned.bytes);
    q36::StateImageDevicePool pool({arena.base(), arena.capacity()}, planned.layout);
    fill_slot(pool, 0, dflash ? 0x31 : 0x47);
    pool.zero_slot(1, device.stream);

    infernix::HostContextArena contiguous_backing(planned.layout.host.image_bytes,
                                                planned.layout.host.image_bytes);
    auto contiguous = contiguous_backing.allocate(planned.layout.host.image_bytes);
    pool.copy_to_host(0, {.data = contiguous->data(), .layout = &planned.layout.host},
                      device.stream);

    const std::size_t image_bytes   = planned.layout.host.image_bytes;
    const std::size_t segment_bytes = 777;
    const std::size_t segments      = (image_bytes + segment_bytes - 1) / segment_bytes;
    infernix::PinnedHostBuffer backing(segments * segment_bytes);
    auto* base = static_cast<std::byte*>(backing.data());
    std::vector<std::byte*> scattered;
    for (std::size_t segment = 0; segment < segments; ++segment) {
        scattered.push_back(base + (segments - 1U - segment) * segment_bytes);
    }
    pool.copy_to_host_segments(0, scattered, segment_bytes, device.stream);
    device.synchronize();

    const std::byte* packed = contiguous->data();
    bool equal              = true;
    for (std::size_t offset = 0; offset < image_bytes; ++offset) {
        equal =
            equal && scattered[offset / segment_bytes][offset % segment_bytes] == packed[offset];
    }
    expect(equal, "segmented StateImage bytes differ from the packed image");

    const std::vector<const std::byte*> sources(scattered.begin(), scattered.end());
    pool.copy_from_host_segments(sources, segment_bytes, 1, device.stream);
    device.synchronize();
    expect_slot(pool, 1, dflash ? 0x31 : 0x47, "segmented Host roundtrip");

    // Part by part (the order a hybrid restore lands them: the rest, then each linear layer)
    // rebuilds the same slot.
    pool.zero_slot(1, device.stream);
    pool.copy_from_host_segments(sources, segment_bytes, 1,
                                 q36::StateImagePart{.kind = q36::StateImagePart::Kind::Rest},
                                 device.stream);
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        pool.copy_from_host_segments(
            sources, segment_bytes, 1,
            q36::StateImagePart{.kind = q36::StateImagePart::Kind::LinearLayer, .layer = layer},
            device.stream);
    }
    device.synchronize();
    expect_slot(pool, 1, dflash ? 0x31 : 0x47, "per-part segmented Host roundtrip");
    bool layer_rejected = false;
    try {
        pool.copy_from_host_segments(
            sources, segment_bytes, 1,
            q36::StateImagePart{.kind  = q36::StateImagePart::Kind::LinearLayer,
                                .layer = pool.linear().layer_count()},
            device.stream);
    } catch (const std::out_of_range&) { layer_rejected = true; }
    expect(layer_rejected, "an out-of-range StateImage linear layer was accepted");

    bool short_rejected = false;
    try {
        pool.copy_to_host_segments(0, std::span<std::byte* const>(scattered.data(), 1),
                                   segment_bytes, device.stream);
    } catch (const std::invalid_argument&) { short_rejected = true; }
    expect(short_rejected || segments == 1, "segments that do not cover the image were accepted");
}

// The recurrent state rounded to nearest even on its dropped low bits, as the packed Host form
// stores it (ops::pack_fp32_state); a NaN is not produced by these inputs.
std::uint32_t rounded_bits(std::uint32_t bits, std::uint32_t dropped) {
    const std::uint32_t half = 1U << (dropped - 1U);
    return (bits + (half - 1U) + ((bits >> dropped) & 1U)) & ~((1U << dropped) - 1U);
}

// A packed recurrent Host form (hybrid prefix cache --cache-host-state): every other component
// round-trips exactly, the recurrent state comes back rounded, and the Host image is smaller.
void test_packed_recurrent(q36::RecurrentHostFormat format, infernix::DeviceContext& device) {
    const PlannedPool exact  = plan_pool(true, 2);
    const PlannedPool packed = plan_pool(true, 2, false, false, format);
    const std::size_t width  = format == q36::RecurrentHostFormat::Fp24 ? 3U : 2U;
    const std::uint32_t dropped = format == q36::RecurrentHostFormat::Fp24 ? 8U : 16U;
    expect(packed.layout.host.linear_recurrent_layer_bytes ==
                   exact.layout.host.linear_recurrent_layer_bytes / 4U * width &&
               packed.layout.host.linear_recurrent.bytes ==
                   packed.layout.host.linear_recurrent_layer_bytes * 2U &&
               packed.layout.host.image_bytes <= exact.layout.host.image_bytes &&
               packed.layout.recurrent_staging && !exact.layout.recurrent_staging,
           "a packed recurrent Host form must shrink the Host image and stage on the Device");

    infernix::DeviceArena arena(packed.bytes);
    q36::StateImageDevicePool pool({arena.base(), arena.capacity()}, packed.layout);
    fill_slot(pool, 0, 0x29);
    // Recurrent state of a realistic scale, with exact ties on both parities.
    std::vector<std::vector<std::uint32_t>> source(pool.linear().layer_count());
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        const infernix::Tensor state = pool.linear().recurrent_slot(layer, 0);
        std::vector<std::uint32_t>& bits = source[layer];
        bits.resize(state.bytes() / 4U);
        for (std::size_t i = 0; i < bits.size(); ++i) {
            const float value = 0.01F * static_cast<float>(i + 1U) - 0.1F;
            std::memcpy(&bits[i], &value, 4);
        }
        bits[0] = 0x3f800000U | (1U << (dropped - 1U));                      // tie, even: down
        bits[1] = 0x3f800000U | (1U << dropped) | (1U << (dropped - 1U));    // tie, odd: up
        CUDA_CHECK(cudaMemcpy(state.data, bits.data(), state.bytes(), cudaMemcpyHostToDevice));
    }
    pool.zero_slot(1, device.stream);

    const std::size_t image_bytes   = packed.layout.host.image_bytes;
    const std::size_t segment_bytes = 777;
    const std::size_t segments      = (image_bytes + segment_bytes - 1) / segment_bytes;
    infernix::PinnedHostBuffer backing(segments * segment_bytes);
    auto* base = static_cast<std::byte*>(backing.data());
    std::vector<std::byte*> scattered;
    for (std::size_t segment = 0; segment < segments; ++segment) {
        scattered.push_back(base + (segments - 1U - segment) * segment_bytes);
    }
    pool.copy_to_host_segments(0, scattered, segment_bytes, device.stream);
    device.synchronize();
    const std::vector<const std::byte*> sources(scattered.begin(), scattered.end());
    // Part by part, as a hybrid restore lands them.
    pool.copy_from_host_segments(sources, segment_bytes, 1,
                                 q36::StateImagePart{.kind = q36::StateImagePart::Kind::Rest},
                                 device.stream);
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        pool.copy_from_host_segments(
            sources, segment_bytes, 1,
            q36::StateImagePart{.kind = q36::StateImagePart::Kind::LinearLayer, .layer = layer},
            device.stream);
    }
    device.synchronize();

    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        expect_bytes(pool.linear().conv_slot(layer, 1), static_cast<unsigned char>(0x29 + layer),
                     "packed Host form keeps the conv state exact");
        const infernix::Tensor state = pool.linear().recurrent_slot(layer, 1);
        std::vector<std::uint32_t> got(state.bytes() / 4U);
        CUDA_CHECK(cudaMemcpy(got.data(), state.data, state.bytes(), cudaMemcpyDeviceToHost));
        bool equal = true;
        for (std::size_t i = 0; i < got.size(); ++i) {
            equal = equal && got[i] == rounded_bits(source[layer][i], dropped);
        }
        expect(equal, "packed Host form must restore the recurrent state rounded to nearest even");
    }
    expect_bytes(pool.continuation_hidden_slot(1), 0x29 + 0x20,
                 "packed Host form keeps the continuation hidden exact");
    for (std::uint32_t layer = 0; layer < pool.dflash_local()->layer_count(); ++layer) {
        const auto view = pool.dflash_local()->layer_view(layer);
        expect_bytes(view.k.slice(3, 1, 1), static_cast<unsigned char>(0x29 + 0x30 + layer),
                     "packed Host form keeps the DFlash local state exact");
    }

    infernix::HostContextArena contiguous_backing(image_bytes, image_bytes);
    auto contiguous = contiguous_backing.allocate(image_bytes);
    bool unsegmented_rejected = false;
    try {
        pool.copy_to_host(0, {.data = contiguous->data(), .layout = &packed.layout.host},
                          device.stream);
    } catch (const std::logic_error&) { unsegmented_rejected = true; }
    expect(unsegmented_rejected, "an unsegmented copy of a packed Host form was accepted");
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);

    infernix::DeviceContext device(0);
    PlannedPool planned = plan_pool(true);
    infernix::DeviceArena arena(planned.bytes);
    q36::StateImageDevicePool pool({arena.base(), arena.capacity()}, planned.layout);

    expect(pool.slot_count() == 4, "StateImage slot count");
    expect(pool.linear().layer_count() == 2, "StateImage Linear Attention layer count");
    expect(pool.continuation_hidden_slot(0).dtype == infernix::DType::BF16 &&
               pool.continuation_hidden_slot(0).ne[0] == 7,
           "StateImage continuation hidden geometry");
    expect(pool.dflash_local() != nullptr && pool.dflash_local()->layer_count() == 2 &&
               pool.dflash_local()->lane_capacity() == 4,
           "StateImage DFlash local geometry");
    const q36::StateImageDeviceSlotView complete = pool.slot_view(0);
    expect(complete.linear.layers == 2 && complete.continuation_hidden.ne[0] == 7 &&
               complete.dflash_local && complete.dflash_local->layers == 2,
           "StateImage slot view exposes every fixed component");

    fill_slot(pool, 0, 0x11);
    fill_slot(pool, 1, 0x52);
    pool.copy_slot(0, 2, device.stream);
    device.synchronize();
    expect_slot(pool, 0, 0x11, "StateImage D2D source isolation");
    expect_slot(pool, 2, 0x11, "StateImage D2D complete destination");

    pool.zero_slot(1, device.stream);
    device.synchronize();
    expect_zero_slot(pool, 1, "StateImage zero complete destination");
    expect_slot(pool, 0, 0x11, "StateImage zero source isolation");

    test_host_roundtrip(false, device);
    test_host_roundtrip(true, device);
    test_host_roundtrip(true, device, true);
    test_host_roundtrip(false, device, false, true);
    test_kv_window(device);
    test_segmented_roundtrip(false, device);
    test_segmented_roundtrip(true, device);
    test_segmented_roundtrip(true, device, true);
    test_packed_recurrent(q36::RecurrentHostFormat::Fp24, device);
    test_packed_recurrent(q36::RecurrentHostFormat::Bf16, device);

    return failures == 0 ? 0 : 1;
}
