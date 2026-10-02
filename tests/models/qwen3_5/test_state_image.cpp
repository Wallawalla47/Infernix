#include "core/device.h"
#include "core/host_kv_arena.h"
#include "models/qwen3_5/state/state_image.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace q36 = ninfer::models::qwen3_5;

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
                      bool window = false) {
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
                .conv_dtype     = ninfer::DType::BF16,
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
    ninfer::LayoutBuilder builder;
    q36::StateImageDeviceLayout layout = q36::plan_state_image_device_pool(builder, spec);
    return {.layout = std::move(layout), .bytes = builder.finish(256)};
}

// The pool copies on the device's non-blocking stream, which does not wait for legacy-stream
// work, so a fill completes before the next pool operation is enqueued.
void set_bytes(const ninfer::Tensor& tensor, unsigned char value) {
    CUDA_CHECK(cudaMemset(tensor.data, value, tensor.bytes()));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void expect_bytes(const ninfer::Tensor& tensor, unsigned char expected, std::string_view label) {
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
    if (ninfer::CyclicKVCache* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            set_bytes(view.k.slice(3, slot, 1), static_cast<unsigned char>(base + 0x30 + layer));
            set_bytes(view.v.slice(3, slot, 1), static_cast<unsigned char>(base + 0x40 + layer));
        }
    }
    for (std::uint32_t layer = 0; pool.has_kv_window() && layer < 3; ++layer) {
        const ninfer::PagedKVWindowView window = pool.kv_window_slot_view(layer, slot);
        unsigned char value = static_cast<unsigned char>(base + 0x50 + 8 * layer);
        for (const ninfer::Tensor& plane :
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
    if (ninfer::CyclicKVCache* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            expect_bytes(view.k.slice(3, slot, 1), static_cast<unsigned char>(base + 0x30 + layer),
                         label);
            expect_bytes(view.v.slice(3, slot, 1), static_cast<unsigned char>(base + 0x40 + layer),
                         label);
        }
    }
    for (std::uint32_t layer = 0; pool.has_kv_window() && layer < 3; ++layer) {
        const ninfer::PagedKVWindowView window = pool.kv_window_slot_view(layer, slot);
        unsigned char value = static_cast<unsigned char>(base + 0x50 + 8 * layer);
        for (const ninfer::Tensor& plane :
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
    if (ninfer::CyclicKVCache* local = pool.dflash_local(); local != nullptr) {
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

void test_host_roundtrip(bool dflash, ninfer::DeviceContext& device, bool dflash2 = false,
                         bool window = false) {
    PlannedPool planned = plan_pool(dflash, 2, dflash2, window);
    if (dflash2) {
        expect(q36::fork_local_transfer_work(planned.layout.host).payload_bytes ==
                   40ULL * 1024 * 1024,
               "DFlash2 local snapshot must transfer exactly 40 MiB");
    }
    ninfer::DeviceArena arena(planned.bytes);
    q36::StateImageDevicePool pool({arena.base(), arena.capacity()}, planned.layout);
    fill_slot(pool, 0, dflash ? 0x19 : 0x25);
    pool.zero_slot(1, device.stream);

    ninfer::HostContextArena host_backing(planned.layout.host.image_bytes,
                                          planned.layout.host.image_bytes);
    q36::HostStatePool host(host_backing, planned.layout.host);
    const auto handle = host.allocate();
    expect(handle.has_value(), "HostStatePool allocates its fixed slot");
    expect(!host.allocate().has_value(), "HostStatePool reports capacity exhaustion");
    expect(host.occupied() == 1, "HostStatePool occupied count after allocation");

    pool.copy_to_host(0, host.writable_view(*handle), device.stream);
    device.synchronize();
    expect(host.publish(*handle), "Host state publication accepts its reservation");
    expect(host_backing.reserved_bytes() == 0 &&
               host_backing.live_bytes() == planned.layout.host.image_bytes,
           "Host state publication converts reservation to one live allocation");
    pool.copy_from_host(host.view(*handle), 1, device.stream);
    device.synchronize();
    expect_slot(pool, 1, dflash ? 0x19 : 0x25,
                dflash ? "DFlash Host roundtrip" : "common Host roundtrip");

    const q36::HostStateSlotHandle stale = *handle;
    expect(host.release(stale), "HostStatePool releases a live handle");
    expect(host.occupied() == 0, "HostStatePool occupied count after release");
    bool stale_view_rejected = false;
    try {
        (void)host.view(stale);
    } catch (const std::invalid_argument&) { stale_view_rejected = true; }
    expect(stale_view_rejected, "HostStatePool rejects a stale view");
    const auto reused = host.allocate();
    expect(reused && reused->index == stale.index && reused->generation != stale.generation,
           "HostStatePool reuse advances generation");
    expect(!host.release(stale), "HostStatePool rejects stale release");
    expect(host.release(*reused), "HostStatePool releases the reused slot");
}

void test_shared_host_capacity() {
    const PlannedPool planned           = plan_pool(false, 2);
    const ninfer::HostKVPageLayout page = ninfer::plan_host_kv_page_layout(
        {.page_tokens = 1,
         .planes      = {{.dtype = ninfer::DType::BF16, .leading_extent = 8, .head_extent = 1}}});
    const std::size_t image_bytes = planned.layout.host.image_bytes;
    expect(image_bytes % page.page_stride == 0,
           "shared Host fixture image consists of complete KV page extents");
    const auto pages_per_image = static_cast<std::uint32_t>(image_bytes / page.page_stride);
    ninfer::HostContextArena backing(image_bytes * 2, std::min(image_bytes, page.page_stride));
    const std::array layouts{page};
    ninfer::HostKVArena kv(backing, layouts);
    q36::HostStatePool state(backing, planned.layout.host);
    q36::HostStatePool other_state(backing, planned.layout.host);
    auto source                     = state.allocate();
    const std::byte* original_bytes = state.view(*source).data;
    expect(state.publish(*source), "shared Host source publication");
    auto destination = kv.allocate(page, pages_per_image);
    expect(destination && backing.live_bytes() == image_bytes &&
               backing.reserved_bytes() == image_bytes && !state.can_allocate() &&
               !state.allocate() && !kv.allocate(page, 1),
           "State and in-flight KV destination consume the same actual backing");
    expect(!other_state.release(*source) && !other_state.publish(*source),
           "State handles cannot release another typed owner's allocation");
    destination.reset();
    expect(backing.reserved_bytes() == 0 && backing.live_bytes() == image_bytes &&
               state.view(*source).data == original_bytes,
           "KV transfer cancellation retains the pinned State source");
    expect(state.release(*source), "shared Host state release");
    auto all_kv = kv.allocate(page, pages_per_image * 2);
    expect(all_kv && kv.view(*all_kv).data() == original_bytes &&
               backing.occupied_bytes() == backing.capacity_bytes(),
           "KV reuses the exact physical extent previously occupied by State");
    all_kv->publish();
    const auto stale = kv.view(*all_kv);
    auto parts       = kv.split(std::move(*all_kv), pages_per_image);
    expect(!stale.valid() && backing.live_bytes() == image_bytes * 2,
           "KV split invalidates old views while keeping unique byte accounting");
    (void)parts.first.release();
    auto recovered = state.allocate();
    expect(recovered && state.view(*recovered).data == original_bytes &&
               backing.occupied_bytes() == backing.capacity_bytes(),
           "State reuses the exact physical extent released by split KV");
    expect(state.release(*recovered), "shared Host recovered state release");
    (void)parts.second.release();
    expect(backing.occupied_bytes() == 0 && backing.allocation_count() == 0 &&
               backing.capacity_bytes() == image_bytes * 2,
           "shared Host typed owners return all live and reserved extents");
}

// The exact KV window is a per-slot component: a layer view spans every slot, a slot view one;
// a fork copies it with DFlash local state but leaves the Linear Attention state alone.
void test_kv_window(ninfer::DeviceContext& device) {
    PlannedPool planned = plan_pool(true, 3, false, true);
    ninfer::DeviceArena arena(planned.bytes);
    q36::StateImageDevicePool pool({arena.base(), arena.capacity()}, planned.layout);
    expect(pool.has_kv_window() && pool.has_fork_local(), "StateImage KV window is present");
    const ninfer::PagedKVWindowView all = pool.kv_window_view(2);
    expect(all.k_codes.ne[0] == 256 && all.k_codes.ne[1] == ninfer::kKVWindowSlots &&
               all.k_codes.ne[2] == 2 && all.k_codes.ne[3] == 3 &&
               all.k_scales.ne[0] == ninfer::kKVWindowGroups && all.tags.ne[0] == 2 &&
               all.tags.ne[3] == 3,
           "StateImage KV window layer view spans every slot");
    const ninfer::PagedKVWindowView one = pool.kv_window_slot_view(2, 1);
    expect(one.k_codes.ne[3] == 1 &&
               static_cast<const std::byte*>(one.k_codes.data) ==
                   static_cast<const std::byte*>(all.k_codes.data) + one.k_codes.bytes(),
           "StateImage KV window slot view selects the slot");
    bool layer_rejected = false;
    try {
        (void)pool.kv_window_view(3);
    } catch (const std::out_of_range&) { layer_rejected = true; }
    expect(layer_rejected, "an out-of-range KV window layer was accepted");

    const ninfer::TransferWork fork = q36::fork_local_transfer_work(planned.layout.host);
    expect(fork.payload_bytes == 2ULL * planned.layout.host.dflash_local_layer_bytes * 2 +
                                     planned.layout.host.kv_window->bytes &&
               fork.copy_operations == 2 * 2 + 5 * 3,
           "fork-local work covers DFlash local state and the KV window");

    fill_slot(pool, 0, 0x13);
    fill_slot(pool, 1, 0x61);
    pool.copy_fork_local(0, 1, device.stream);
    device.synchronize();
    for (std::uint32_t layer = 0; layer < 3; ++layer) {
        const ninfer::PagedKVWindowView window = pool.kv_window_slot_view(layer, 1);
        unsigned char value = static_cast<unsigned char>(0x13 + 0x50 + 8 * layer);
        for (const ninfer::Tensor& plane :
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

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);

    ninfer::DeviceContext device(0);
    PlannedPool planned = plan_pool(true);
    ninfer::DeviceArena arena(planned.bytes);
    q36::StateImageDevicePool pool({arena.base(), arena.capacity()}, planned.layout);

    expect(pool.slot_count() == 4, "StateImage slot count");
    expect(pool.linear().layer_count() == 2, "StateImage Linear Attention layer count");
    expect(pool.continuation_hidden_slot(0).dtype == ninfer::DType::BF16 &&
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
    test_shared_host_capacity();
    test_host_roundtrip(false, device, false, true);
    test_kv_window(device);

    return failures == 0 ? 0 : 1;
}
