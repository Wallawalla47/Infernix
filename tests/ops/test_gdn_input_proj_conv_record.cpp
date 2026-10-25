#include "ops/op_family.h"
#include "core/weight.h"
#include "core/device.h"
#include "infernix/ops/gdn_input_proj.h"
#include "infernix/ops/speculative_tree.h"

#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace infernix;
using namespace infernix::test;
using namespace infernix::test::input_projection;

namespace {

constexpr std::int32_t kQueryRows = 2048;
constexpr std::int32_t kKeyRows   = 2048;

int test_record_capacity_domain() {
    int failures       = 0;
    const auto rejects = [&](auto&& query) {
        try {
            (void)query();
        } catch (const std::invalid_argument& error) {
            if (std::string_view(error.what()).find("B/T domain") != std::string_view::npos) {
                return;
            }
        }
        std::cerr << "record workspace did not reject unsupported B/T domain\n";
        ++failures;
    };
    for (const auto [batch, width] : {std::pair{2, 65}, {9, 16}, {1, 65}, {1, 1}}) {
        for (int values : {4096, 6144}) {
            rejects([&] {
                return ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                    kQueryRows, kKeyRows, values, batch, width, width);
            });
        }
        for (auto qtype : {QType::NVFP4, QType::FP8_E4M3FN_ROW_BF16}) {
            rejects([&] {
                return ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                    qtype, 16384, 5120, ops::LinearPolicy::A16Only, batch, width, width);
            });
        }
    }
    return failures;
}

std::vector<std::uint16_t> make_bf16_bits(std::size_t elements, std::uint32_t seed, float low,
                                          float high) {
    std::vector<float> values(elements);
    fill_uniform(values, seed, low, high);
    round_to_bf16(values);
    return bf16_bits(values);
}

int verify_equal(std::string_view label, const std::vector<std::uint16_t>& lhs,
                 const std::vector<std::uint16_t>& rhs) {
    if (lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin())) { return 0; }
    std::cerr << label << ": BF16 bits differ\n";
    return 1;
}

int verify_zero_tail(std::string_view label, const std::vector<std::uint16_t>& values,
                     std::int32_t rows, std::int32_t width, std::int32_t batch,
                     const std::vector<std::int32_t>& valid_columns) {
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        for (std::int32_t token = valid_columns[static_cast<std::size_t>(batch_row)]; token < width;
             ++token) {
            const std::size_t base = static_cast<std::size_t>(batch_row * width + token) * rows;
            for (std::int32_t row = 0; row < rows; ++row) {
                if (values[base + row] != 0) {
                    std::cerr << label << ": invalid tail is not exact zero\n";
                    return 1;
                }
            }
        }
    }
    return 0;
}

int verify_conv_record(std::string_view label, const std::vector<std::uint16_t>& snapshot_state,
                       const std::vector<std::uint16_t>& record, std::int32_t channels,
                       std::int32_t width, std::int32_t batch,
                       const std::vector<std::int32_t>& valid_columns,
                       const std::vector<std::int32_t>& snapshot_bases) {
    const std::size_t slot_stride = static_cast<std::size_t>(channels) * 3;
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        for (std::int32_t token = 0; token < valid_columns[static_cast<std::size_t>(batch_row)];
             ++token) {
            const std::size_t snapshot =
                static_cast<std::size_t>(snapshot_bases[static_cast<std::size_t>(batch_row)] +
                                         token) *
                    slot_stride +
                2ULL * channels;
            const std::size_t record_column =
                static_cast<std::size_t>(batch_row * width + token) * channels;
            if (!std::equal(snapshot_state.begin() + static_cast<std::ptrdiff_t>(snapshot),
                            snapshot_state.begin() +
                                static_cast<std::ptrdiff_t>(snapshot + channels),
                            record.begin() + static_cast<std::ptrdiff_t>(record_column))) {
                std::cerr << label << ": conv record differs from snapshot newest column\n";
                return 1;
            }
        }
    }
    return 0;
}

template <class SnapshotLaunch, class RecordLaunch>
int run_case(std::string_view label, std::int32_t hidden, std::int32_t value_rows,
             std::int32_t z_rows, std::int32_t width, std::int32_t batch,
             std::vector<std::int32_t> valid_columns, std::size_t snapshot_workspace_bytes,
             std::size_t record_workspace_bytes, SnapshotLaunch&& snapshot_launch,
             RecordLaunch&& record_launch, std::uint32_t seed) {
    const std::int32_t channels          = kQueryRows + kKeyRows + value_rows;
    const std::int32_t aggregate_columns = width * batch;
    const std::int32_t source_slots      = 8;
    const std::int32_t slots             = aggregate_columns + source_slots;
    const bool dense                     = valid_columns.empty();
    if (valid_columns.empty()) { valid_columns.assign(static_cast<std::size_t>(batch), width); }

    const std::vector<float> activation = make_bf16_activation(hidden, aggregate_columns, seed);
    const std::vector<std::uint16_t> conv_weight_bits =
        make_bf16_bits(static_cast<std::size_t>(channels) * 4, seed + 1, -0.02F, 0.02F);
    const std::vector<std::uint16_t> state_before = make_bf16_bits(
        static_cast<std::size_t>(channels) * 3 * source_slots, seed + 2, -0.05F, 0.05F);

    std::vector<std::uint16_t> snapshot_before(static_cast<std::size_t>(channels) * 3 * slots);
    std::copy(state_before.begin(), state_before.end(),
              snapshot_before.begin() + static_cast<std::size_t>(channels) * 3 * aggregate_columns);
    std::vector<std::int32_t> snapshot_initial_slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> snapshot_bases(static_cast<std::size_t>(batch));
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        snapshot_bases[static_cast<std::size_t>(batch_row)] = batch_row * width;
        // Read-only selectors may repeat; source capacity does not depend on W.
        initial_slots[static_cast<std::size_t>(batch_row)] =
            batch_row == 7 ? 7 : (batch_row * 3 + 7) % source_slots;
        snapshot_initial_slots[static_cast<std::size_t>(batch_row)] =
            aggregate_columns + initial_slots[static_cast<std::size_t>(batch_row)];
    }

    DeviceBuffer device_x           = to_device_bf16(activation);
    DeviceBuffer device_conv_weight = to_device(conv_weight_bits);
    DeviceBuffer snapshot_state     = to_device(snapshot_before);
    DeviceBuffer record_state       = to_device(state_before);
    DeviceBuffer device_valid;
    if (!dense) { device_valid = to_device(valid_columns); }
    DeviceBuffer device_initial          = to_device(initial_slots);
    DeviceBuffer device_snapshot_initial = to_device(snapshot_initial_slots);
    DeviceBuffer device_snapshot         = to_device(snapshot_bases);

    GuardedBf16Tensor snapshot_query(kQueryRows, aggregate_columns);
    GuardedBf16Tensor snapshot_key(kKeyRows, aggregate_columns);
    GuardedBf16Tensor snapshot_value(value_rows, aggregate_columns);
    GuardedBf16Tensor snapshot_z(z_rows, aggregate_columns);
    GuardedBf16Tensor record_query(kQueryRows, aggregate_columns);
    GuardedBf16Tensor record_key(kKeyRows, aggregate_columns);
    GuardedBf16Tensor record_value(value_rows, aggregate_columns);
    GuardedBf16Tensor record_z(z_rows, aggregate_columns);
    GuardedBf16Tensor conv_record(channels, aggregate_columns);

    Tensor x(device_x.p, DType::BF16, {hidden, width, batch});
    Tensor conv_weight(device_conv_weight.p, DType::BF16, {channels, 4});
    Tensor snapshot_state_view(snapshot_state.p, DType::BF16, {channels, 3, slots});
    Tensor record_state_view(record_state.p, DType::BF16, {channels, 3, source_slots});
    Tensor valid;
    if (!dense) { valid = Tensor(device_valid.p, DType::I32, {batch}); }
    Tensor initial(device_initial.p, DType::I32, {batch});
    Tensor snapshot_initial(device_snapshot_initial.p, DType::I32, {batch});
    Tensor snapshot_base(device_snapshot.p, DType::I32, {batch});
    Tensor snapshot_q(snapshot_query.data(), DType::BF16, {kQueryRows, width, batch});
    Tensor snapshot_k(snapshot_key.data(), DType::BF16, {kKeyRows, width, batch});
    Tensor snapshot_v(snapshot_value.data(), DType::BF16, {value_rows, width, batch});
    Tensor snapshot_z_view(snapshot_z.data(), DType::BF16, {z_rows, width, batch});
    Tensor record_q(record_query.data(), DType::BF16, {kQueryRows, width, batch});
    Tensor record_k(record_key.data(), DType::BF16, {kKeyRows, width, batch});
    Tensor record_v(record_value.data(), DType::BF16, {value_rows, width, batch});
    Tensor record_z_view(record_z.data(), DType::BF16, {z_rows, width, batch});
    Tensor conv_record_view(conv_record.data(), DType::BF16, {channels, width, batch});

    WorkspaceArena snapshot_workspace(std::max<std::size_t>(256, snapshot_workspace_bytes));
    WorkspaceArena record_workspace(std::max<std::size_t>(256, record_workspace_bytes));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cuda_synchronize();
    const auto launch = [&] {
        snapshot_launch(x, conv_weight, snapshot_state_view, valid, snapshot_initial, snapshot_base,
                        snapshot_q, snapshot_k, snapshot_v, snapshot_z_view, snapshot_workspace,
                        stream);
        record_launch(x, conv_weight, record_state_view, valid, initial, conv_record_view, record_q,
                      record_k, record_v, record_z_view, record_workspace, stream);
    };
    launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto activation_bits = bf16_bits(activation);
    if (width == 2 || width == 9 || width == 10 || width == 16) {
        // 2 is the bottom of the Q5 parent's split4 band, 9 the count that used to be its top, 10 the
        // count that is its top now, and 16 a grouped representative extent.
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // The same graph must consume changed inputs at the captured addresses.
        activation_bits = bf16_bits(make_bf16_activation(hidden, aggregate_columns, seed + 19));
        CUDA_CHECK(cudaMemcpyAsync(device_x.p, activation_bits.data(),
                                   activation_bits.size() * sizeof(std::uint16_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));

    int failures = 0;
    failures +=
        verify_equal(std::string(label) + " query", snapshot_query.bits(), record_query.bits());
    failures += verify_equal(std::string(label) + " key", snapshot_key.bits(), record_key.bits());
    failures +=
        verify_equal(std::string(label) + " value", snapshot_value.bits(), record_value.bits());
    failures += verify_equal(std::string(label) + " z", snapshot_z.bits(), record_z.bits());
    failures += verify_zero_tail(std::string(label) + " query", record_query.bits(), kQueryRows,
                                 width, batch, valid_columns);
    failures += verify_zero_tail(std::string(label) + " key", record_key.bits(), kKeyRows, width,
                                 batch, valid_columns);
    failures += verify_zero_tail(std::string(label) + " value", record_value.bits(), value_rows,
                                 width, batch, valid_columns);

    const std::vector<std::uint16_t> snapshot_state_after =
        from_device<std::uint16_t>(snapshot_state, snapshot_before.size());
    const std::vector<std::uint16_t> record_state_after =
        from_device<std::uint16_t>(record_state, state_before.size());
    failures += verify_conv_record(label, snapshot_state_after, conv_record.bits(), channels, width,
                                   batch, valid_columns, snapshot_bases);
    failures +=
        verify_equal(std::string(label) + " source state", state_before, record_state_after);

    failures += verify_preserved(std::string(label) + " x", device_x, activation_bits);
    failures +=
        verify_preserved(std::string(label) + " conv weight", device_conv_weight, conv_weight_bits);
    failures += verify_preserved(std::string(label) + " initial", device_initial, initial_slots);
    if (!dense) {
        failures += verify_preserved(std::string(label) + " valid", device_valid, valid_columns);
    }
    failures += snapshot_query.verify_guards(std::string(label) + " snapshot query");
    failures += snapshot_key.verify_guards(std::string(label) + " snapshot key");
    failures += snapshot_value.verify_guards(std::string(label) + " snapshot value");
    failures += snapshot_z.verify_guards(std::string(label) + " snapshot z");
    failures += record_query.verify_guards(std::string(label) + " record query");
    failures += record_key.verify_guards(std::string(label) + " record key");
    failures += record_value.verify_guards(std::string(label) + " record value");
    failures += record_z.verify_guards(std::string(label) + " record z");
    failures += conv_record.verify_guards(std::string(label) + " conv record");
    if (snapshot_workspace.used() != 0 ||
        snapshot_workspace.peak_used() != snapshot_workspace_bytes) {
        std::cerr << label << ": snapshot workspace query/execution mismatch\n";
        ++failures;
    }
    if (record_workspace.used() != 0 || record_workspace.peak_used() != record_workspace_bytes) {
        std::cerr << label << ": record workspace query/execution mismatch\n";
        ++failures;
    }
    return failures;
}

std::vector<std::int32_t> ragged(std::int32_t width, std::int32_t batch) {
    std::vector<std::int32_t> valid(batch);
    for (int b = 0; b < batch; ++b) valid[b] = b == 0 ? width : 1 + (3 * b) % width;
    return valid;
}

int run_q4_q5() {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    DevicePackedWeight qk(
        quantized_weight::make_patterned_weight(QType::Q4_G64_FP16, 4096, kHidden, 1401U));
    DevicePackedWeight value_z(
        quantized_weight::make_patterned_weight(QType::Q5_G64_FP16, 12288, kHidden, 1403U));

    int failures   = 0;
    const auto run = [&](std::int32_t width, std::int32_t batch, std::vector<std::int32_t> valid,
                         std::uint32_t seed, ops::LinearPolicy policy = ops::LinearPolicy::A16Only) {
        const std::size_t snapshot_bytes =
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                kQueryRows, kKeyRows, kValueRows, policy, batch, width, width);
        const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            kQueryRows, kKeyRows, kValueRows, policy, batch, width, width);
        return run_case(
            std::string(ops::column_invariant(policy) ? "Q4/Q5 invariant B=" : "Q4/Q5 B=") +
                std::to_string(batch) + " T=" + std::to_string(width),
            kHidden, kValueRows, kZRows, width, batch, std::move(valid), snapshot_bytes,
            record_bytes,
            [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
                const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_snapshot(x, qk.view(), value_z.view(), conv, state,
                                                  valid_columns, initial, snapshot_base, q, k, v, z,
                                                  policy, workspace, stream);
            },
            [&](const Tensor& x, const Tensor& conv, const Tensor& state,
                const Tensor& valid_columns, const Tensor& initial, Tensor& record, Tensor& q,
                Tensor& k, Tensor& v, Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_record(x, qk.view(), value_z.view(), conv, state,
                                                valid_columns, initial, record, q, k, v, z, policy,
                                                workspace, stream);
            },
            seed);
    };
    // The column-invariant policy: record equals snapshot at the default route's fused widths too.
    for (int width : {2, 3, 5, 6, 9, 16}) {
        failures += run(width, 1, {}, 1700U + width, ops::LinearPolicy::A16ColumnInvariant);
        failures += run(width, 8, ragged(width, 8), 1750U + width, ops::LinearPolicy::A16ColumnInvariant);
    }
    failures += run(33, 1, {}, 1790U, ops::LinearPolicy::A16ColumnInvariant);
    for (int width = 2; width <= 16; ++width) {
        failures += run(width, 1, {}, 1400U + width);
        failures += run(width, 8, ragged(width, 8), 1450U + width);
    }
    failures += run(5, 3, {5, 3, 1}, 1491U);
    failures += run(4, 4, {4, 3, 2, 1}, 1492U);
    // R7-review trial: the batched organisation whose aggregate column count is 8, which is inside the
    // range the fused template covers when the request is read from the flattened token axis. The dense
    // case checks the history reload at the request boundary, and the masked one additionally puts an
    // invalid tail in the second request, so the boundary is checked next to a zeroed column.
    failures += run(4, 2, {}, 1493U);
    failures += run(4, 2, {4, 2}, 1494U);
    for (int width = 17; width <= 64; ++width) {
        failures += run(width, 1, {}, 2400U + width);
        failures += run(width, 1, {width / 2}, 2450U + width);
    }
    // Batched ngram copy rounds above 16 columns.
    for (int width : {17, 32, 64}) {
        failures += run(width, 2, {width, width / 3}, 2480U + width);
        failures += run(width, 8, ragged(width, 8), 2490U + width);
    }
    failures += qk.verify_preserved("Q4 record qk weight");
    failures += value_z.verify_preserved("Q5 record value/z weight");
    return failures;
}

int run_q8() {
    constexpr std::int32_t kHidden    = 2048;
    constexpr std::int32_t kValueRows = 4096;
    constexpr std::int32_t kZRows     = 4096;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, 12288, kHidden, 1501U));

    int failures   = 0;
    const auto run = [&](std::int32_t width, std::int32_t batch, std::vector<std::int32_t> valid,
                         std::uint32_t seed) {
        const std::size_t snapshot_bytes =
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                kQueryRows, kKeyRows, kValueRows, batch, width, width);
        const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            kQueryRows, kKeyRows, kValueRows, batch, width, width);
        return run_case(
            "Q8 B=" + std::to_string(batch) + " T=" + std::to_string(width), kHidden, kValueRows,
            kZRows, width, batch, std::move(valid), snapshot_bytes, record_bytes,
            [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
                const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, state, valid_columns,
                                                  initial, snapshot_base, q, k, v, z, workspace,
                                                  stream);
            },
            [&](const Tensor& x, const Tensor& conv, const Tensor& state,
                const Tensor& valid_columns, const Tensor& initial, Tensor& record, Tensor& q,
                Tensor& k, Tensor& v, Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_record(x, parent.view(), conv, state, valid_columns,
                                                initial, record, q, k, v, z, workspace, stream);
            },
            seed);
    };
    failures += run(2, 1, {1}, 1511U);
    failures += run(16, 1, {}, 1521U);
    failures += run(16, 8, {16, 13, 9, 7, 5, 3, 2, 1}, 1531U);
    for (int width = 17; width <= 64; ++width) {
        failures += run(width, 1, {}, 2500U + width);
        failures += run(width, 1, {width / 2}, 2550U + width);
    }
    for (int width : {17, 48, 64}) {
        failures += run(width, 2, {width, width / 3}, 2580U + width);
        failures += run(width, 8, ragged(width, 8), 2590U + width);
    }
    failures += parent.verify_preserved("Q8 record parent weight");
    return failures;
}

// The Q8 parent's verification-tree form (the 35B-A3B's DFlash trees): the projection materialized
// into the record, then the tree convolution. A chain-shaped tree row is the chain form over its
// valid prefix, so at batch sizes where the record form also composes from the materialized
// projection every output equals the record form's bit for bit, idle columns included. Branched
// rows are the projected tree convolution's (test_speculative_tree).
int run_q8_tree() {
    constexpr std::int32_t kHidden   = 2048;
    constexpr std::int32_t kValueRows = 4096;
    constexpr std::int32_t kZRows    = 4096;
    constexpr std::int32_t kChannels = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kSlots    = 8;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, 12288, kHidden, 1601U));
    int failures = 0;
    struct Shape {
        std::int32_t width;
        std::vector<std::int32_t> valid;
    };
    for (const Shape& shape : {Shape{9, {9, 4}}, Shape{16, {16, 11, 1}}, Shape{24, {16, 7}}}) {
        const std::int32_t width   = shape.width;
        const auto batch           = static_cast<std::int32_t>(shape.valid.size());
        const std::int32_t columns = width * batch;
        const std::string label =
            "Q8 tree B=" + std::to_string(batch) + " T=" + std::to_string(width);
        const std::vector<float> activation = make_bf16_activation(kHidden, columns, 1610U + width);
        const auto conv_bits =
            make_bf16_bits(static_cast<std::size_t>(kChannels) * 4, 1620U + width, -0.02F, 0.02F);
        const auto state_bits = make_bf16_bits(static_cast<std::size_t>(kChannels) * 3 * kSlots,
                                               1630U + width, -0.05F, 0.05F);
        std::vector<std::int32_t> initial(static_cast<std::size_t>(batch));
        std::vector<ops::SpeculativeTreeRow> rows(static_cast<std::size_t>(batch));
        for (std::int32_t b = 0; b < batch; ++b) {
            initial[static_cast<std::size_t>(b)] = (b * 3 + 5) % kSlots;
            const std::int32_t nodes           = shape.valid[static_cast<std::size_t>(b)];
            ops::SpeculativeTreeRow& row       = rows[static_cast<std::size_t>(b)];
            row                                = {};
            row.nodes                          = nodes;
            row.main_depth                     = nodes - 1;
            row.paths                          = 1;
            row.tree                           = 1;
            for (int c = 0; c < ops::kSpeculativeTreeMaxNodes; ++c) {
                const bool live     = c < nodes;
                row.parent[c]       = static_cast<std::int8_t>(live ? c - 1 : -1);
                row.depth[c]        = static_cast<std::int8_t>(live ? c : 0);
                row.first_child[c]  = static_cast<std::int8_t>(c + 1 < nodes ? c + 1 : -1);
                row.next_sibling[c] = -1;
            }
            row.path_length[0] = static_cast<std::int8_t>(nodes);
            for (int c = 0; c < nodes; ++c) row.path_columns[0][c] = static_cast<std::int8_t>(c);
        }
        DeviceBuffer device_x = to_device_bf16(activation), device_conv = to_device(conv_bits),
                     device_state = to_device(state_bits), device_valid = to_device(shape.valid),
                     device_initial = to_device(initial), device_rows = to_device(rows);
        Tensor x(device_x.p, DType::BF16, {kHidden, width, batch});
        Tensor conv(device_conv.p, DType::BF16, {kChannels, 4});
        Tensor state(device_state.p, DType::BF16, {kChannels, 3, kSlots});
        Tensor valid(device_valid.p, DType::I32, {batch});
        Tensor initial_slots(device_initial.p, DType::I32, {batch});
        Tensor tree_rows(device_rows.p, DType::I32, {ops::kSpeculativeTreeRowWords, batch});
        struct Outputs {
            GuardedBf16Tensor record, query, key, value, z;
        };
        const auto run = [&](bool tree) {
            Outputs out{GuardedBf16Tensor(kChannels, columns), GuardedBf16Tensor(kQueryRows, columns),
                        GuardedBf16Tensor(kKeyRows, columns), GuardedBf16Tensor(kValueRows, columns),
                        GuardedBf16Tensor(kZRows, columns)};
            Tensor record(out.record.data(), DType::BF16, {kChannels, width, batch});
            Tensor q(out.query.data(), DType::BF16, {kQueryRows, width, batch});
            Tensor k(out.key.data(), DType::BF16, {kKeyRows, width, batch});
            Tensor v(out.value.data(), DType::BF16, {kValueRows, width, batch});
            Tensor z(out.z.data(), DType::BF16, {kZRows, width, batch});
            const std::size_t bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                kQueryRows, kKeyRows, kValueRows, batch, width, width);
            WorkspaceArena workspace(std::max<std::size_t>(256, bytes));
            if (tree) {
                ops::gdn_input_proj_conv_record(x, parent.view(), conv, state, valid, initial_slots,
                                                tree_rows, record, q, k, v, z,
                                                ops::LinearPolicy::A16Only, workspace, nullptr);
            } else {
                ops::gdn_input_proj_conv_record(x, parent.view(), conv, state, valid, initial_slots,
                                                record, q, k, v, z, workspace, nullptr);
            }
            cuda_synchronize();
            return out;
        };
        Outputs chain = run(false);
        Outputs tree  = run(true);
        failures += verify_equal(label + " record", chain.record.bits(), tree.record.bits());
        failures += verify_equal(label + " query", chain.query.bits(), tree.query.bits());
        failures += verify_equal(label + " key", chain.key.bits(), tree.key.bits());
        failures += verify_equal(label + " value", chain.value.bits(), tree.value.bits());
        failures += verify_equal(label + " z", chain.z.bits(), tree.z.bits());
        for (GuardedBf16Tensor* guarded :
             {&tree.record, &tree.query, &tree.key, &tree.value, &tree.z}) {
            failures += guarded->verify_guards(label + " tree output");
        }
        failures += verify_preserved(label + " state", device_state, state_bits);
    }
    failures += parent.verify_preserved("Q8 tree parent weight");
    return failures;
}

int run_nvfp4() {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    constexpr std::int32_t kRows      = 16384;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::NVFP4, kRows, kHidden, 1601U, options));

    int failures   = 0;
    const auto run = [&](std::int32_t width, std::int32_t batch, std::vector<std::int32_t> valid,
                         ops::LinearPolicy policy, std::uint32_t seed) {
        const std::size_t snapshot_bytes =
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(QType::NVFP4, kRows, kHidden,
                                                                       policy, batch, width, width);
        const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            QType::NVFP4, kRows, kHidden, policy, batch, width, width);
        return run_case(
            std::string("NVFP4 ") + (policy == ops::LinearPolicy::AllowA4 ? "A4" : "A16") +
                " B=" + std::to_string(batch) + " T=" + std::to_string(width),
            kHidden, kValueRows, kZRows, width, batch, std::move(valid), snapshot_bytes,
            record_bytes,
            [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
                const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, state, valid_columns,
                                                  initial, snapshot_base, q, k, v, z, policy,
                                                  workspace, stream);
            },
            [&](const Tensor& x, const Tensor& conv, const Tensor& state,
                const Tensor& valid_columns, const Tensor& initial, Tensor& record, Tensor& q,
                Tensor& k, Tensor& v, Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_record(x, parent.view(), conv, state, valid_columns,
                                                initial, record, q, k, v, z, policy, workspace,
                                                stream);
            },
            seed);
    };
    for (auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA4}) {
        for (int width = 2; width <= 16; ++width) {
            failures += run(width, 1, {}, policy, 1600U + width);
            failures += run(width, 8, ragged(width, 8), policy, 1650U + width);
        }
        for (int width = 17; width <= 64; ++width) {
            failures += run(width, 1, {}, policy, 2600U + width);
            failures += run(width, 1, {width / 2}, policy, 2650U + width);
        }
        for (int width : {17, 32, 64}) {
            failures += run(width, 2, {width, width / 3}, policy, 2680U + width);
            failures += run(width, 8, ragged(width, 8), policy, 2690U + width);
        }
    }
    failures += parent.verify_preserved("NVFP4 record parent weight");
    return failures;
}

// The registered row-scaled FP8 parents: fp8_e4m3fn_row_bf16 [16384,5120] and Qwen3.5-122B-A10B's
// static-scale fp8_e4m3fn_row_f32 [20480,3072] (no A8 residual profile).
struct Fp8Parent {
    QType qtype;
    std::int32_t hidden, value_rows, z_rows;
    float input_scale; // 0: dynamic A8
    [[nodiscard]] constexpr std::int32_t rows() const { return kQueryRows + kKeyRows + value_rows + z_rows; }
    [[nodiscard]] const char* name() const { return input_scale != 0.0F ? "FP8 row-f32" : "FP8"; }
};
constexpr Fp8Parent kFp8RowBf16{QType::FP8_E4M3FN_ROW_BF16, 5120, 6144, 6144, 0.0F};
// A static input scale of the checkpoint's kind: its reciprocal is inexact.
constexpr Fp8Parent kFp8RowF32{QType::FP8_E4M3FN_ROW_F32, 3072, 8192, 8192, 0.0123456791F};

int run_fp8_case(const Fp8Parent& shape, DevicePackedWeight& parent, std::int32_t width, std::int32_t batch,
                 std::vector<std::int32_t> valid, ops::LinearPolicy policy, std::uint32_t seed) {
    const std::size_t snapshot_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        shape.qtype, shape.rows(), shape.hidden, policy, batch, width, width);
    const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
        shape.qtype, shape.rows(), shape.hidden, policy, batch, width, width);
    return run_case(
        std::string(shape.name()) + " policy=" + std::to_string(static_cast<int>(policy)) +
            " B=" + std::to_string(batch) + " W=" + std::to_string(width),
        shape.hidden, shape.value_rows, shape.z_rows, width, batch, std::move(valid), snapshot_bytes, record_bytes,
        [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
            const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
            Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
            ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, state, valid_columns, initial,
                                              snapshot_base, q, k, v, z, policy, workspace, stream);
        },
        [&](const Tensor& x, const Tensor& conv, const Tensor& state, const Tensor& valid_columns,
            const Tensor& initial, Tensor& record, Tensor& q, Tensor& k, Tensor& v, Tensor& z,
            WorkspaceArena& workspace, cudaStream_t stream) {
            ops::gdn_input_proj_conv_record(x, parent.view(), conv, state, valid_columns, initial,
                                            record, q, k, v, z, policy, workspace, stream);
        },
        seed);
}

int run_fp8(const Fp8Parent& shape, std::uint32_t weight_seed) {
    const quantized_weight::PatternedWeightOptions options{.activation_scale = shape.input_scale};
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(shape.qtype, shape.rows(), shape.hidden, weight_seed, options));
    int failures = 0;
    // The static parent's decode and verification policy is A16ColumnInvariant; the dynamic parent
    // keeps its A8 residual profile.
    const auto policies = shape.input_scale != 0.0F
                              ? std::vector{ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8,
                                            ops::LinearPolicy::A16ColumnInvariant}
                              : std::vector{ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8,
                                            ops::LinearPolicy::AllowA8Residual};
    for (auto policy : policies) {
        for (int width = 2; width <= 16; ++width) {
            failures += run_fp8_case(shape, parent, width, 1, {}, policy, 1700U + width);
            failures += run_fp8_case(shape, parent, width, 8, ragged(width, 8), policy, 1750U + width);
        }
        for (int batch : {2, 3, 4}) {
            failures += run_fp8_case(shape, parent, 4, batch, ragged(4, batch), policy, 1810U + batch);
        }
        for (int width = 17; width <= 64; ++width) {
            failures += run_fp8_case(shape, parent, width, 1, {}, policy, 2700U + width);
            failures += run_fp8_case(shape, parent, width, 1, {width / 2}, policy, 2750U + width);
        }
        for (int width : {17, 32, 64}) {
            failures += run_fp8_case(shape, parent, width, 2, {width, width / 3}, policy, 2780U + width);
            failures += run_fp8_case(shape, parent, width, 8, ragged(width, 8), policy, 2790U + width);
        }
    }
    failures += parent.verify_preserved(std::string(shape.name()) + " record parent weight");
    return failures;
}

} // namespace

INFERNIX_OP_TEST(gdn_input_proj_conv_record) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = test_record_capacity_domain();
    failures += run_q4_q5();
    failures += run_q8();
    failures += run_q8_tree();
    failures += run_nvfp4();
    failures += run_fp8(kFp8RowBf16, 1701U);
    failures += run_fp8(kFp8RowF32, 1703U);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_input_proj_conv_record\n";
    return failures == 0 ? 0 : 1;
}
