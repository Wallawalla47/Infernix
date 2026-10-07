#include "core/layout.h"
#include "models/qwen3_5/state/state_image.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

namespace q36 = infernix::models::qwen3_5;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

q36::StateImageDeviceLayout plan(bool dflash, bool window = false) {
    q36::StateImageSpec spec{
        .linear =
            {
                .layers         = 2,
                .conv_channels  = 5,
                .conv_width     = 3,
                .value_heads    = 2,
                .value_head_dim = 4,
                .key_head_dim   = 3,
                .slot_count     = 2,
                .conv_dtype     = infernix::DType::BF16,
            },
        .hidden = 7,
    };
    if (dflash) {
        spec.dflash_local =
            q36::DFlashLocalStateSpec{.layers = 2, .capacity = 17, .kv_heads = 2, .head_dim = 4};
    }
    if (window) { spec.kv_window = q36::KVWindowStateSpec{.layers = 3, .kv_heads = 4}; }
    infernix::LayoutBuilder builder;
    return q36::plan_state_image_device_pool(builder, spec);
}

} // namespace

int main() {
    const q36::StateImageDeviceLayout common = plan(false);
    const auto common_work                   = q36::state_image_transfer_work(common.host);
    expect(common_work.payload_bytes == common.host.linear_conv.bytes +
                                            common.host.linear_recurrent.bytes +
                                            common.host.continuation_hidden.bytes,
           "common StateImage work payload includes layout padding");
    expect(common_work.copy_operations == 2 * common.host.spec.linear.layers + 1,
           "common StateImage work does not match physical CUDA copies");

    const q36::StateImageDeviceLayout dflash = plan(true);
    const auto full_work                     = q36::state_image_transfer_work(dflash.host);
    const auto local_work                    = q36::fork_local_transfer_work(dflash.host);
    const std::uint64_t local_bytes =
        2ULL * dflash.host.dflash_local_layer_bytes * dflash.host.spec.dflash_local->layers;
    expect(local_work.payload_bytes == local_bytes &&
               local_work.copy_operations == 2 * dflash.host.spec.dflash_local->layers,
           "DFlash-local work does not match physical K/V layer copies");
    expect(full_work.payload_bytes == common_work.payload_bytes + local_work.payload_bytes &&
               full_work.copy_operations ==
                   common_work.copy_operations + local_work.copy_operations,
           "full DFlash StateImage work is not common plus local state");

    // The exact KV window of the vector-quantized formats: per layer and slot, 1088 INT8 K and V
    // rows (256 bytes), their 4 FP16 group scales and 2 I32 tags per (slot, KV head).
    const q36::StateImageDeviceLayout window = plan(false, true);
    const std::uint64_t window_rows          = 1088ULL * 4;
    const std::uint64_t window_layer_bytes   = window_rows * (2 * 256 + 2 * 4 * 2 + 2 * 4);
    expect(window.host.kv_window && window.host.kv_window_layer_bytes == window_layer_bytes &&
               window.host.kv_window->bytes == 3 * window_layer_bytes,
           "KV window Host bytes are its five planes for every layer");
    expect(window.kv_window && window.kv_window->k_codes.shape[3] == 3 * 2 &&
               window.kv_window->tags.shape[0] == 2,
           "KV window Device planes span every (layer, slot)");
    const auto window_work = q36::state_image_transfer_work(window.host);
    const auto window_fork = q36::fork_local_transfer_work(window.host);
    expect(window_fork.payload_bytes == 3 * window_layer_bytes &&
               window_fork.copy_operations == 5 * 3,
           "KV window fork work is its planes");
    expect(window_work.payload_bytes == common_work.payload_bytes + window_fork.payload_bytes &&
               window_work.copy_operations ==
                   common_work.copy_operations + window_fork.copy_operations,
           "full KV-window StateImage work is common plus window state");
    bool common_rejected = false;
    try {
        (void)q36::fork_local_transfer_work(common.host);
    } catch (const std::invalid_argument&) { common_rejected = true; }
    expect(common_rejected, "a StateImage without fork-local state reported fork work");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
