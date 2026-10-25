#include "ops/op_family.h"
#include "core/weight.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include "infernix/ops/gdn_input_proj.h"
#include "infernix/ops/linear.h"

#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::test;
using namespace infernix::test::input_projection;

namespace {

// This criterion belongs to the complete A16 GDN-input-projection Op.
constexpr ReductionCriterion kGdnInputProjA16Tolerance{3.0e-3, 4.0e-3, 3.5e-3};
constexpr ReductionCriterion kFp8GdnInputProjA16Tolerance{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
constexpr ReductionCriterion kFp8GdnInputProjA8Tolerance{0.04, 1.0 / 256.0, 0.06};
// Two E4M3 terms put the A8 residual profile near A16; one term (A8's error) fails it.
constexpr ReductionCriterion kFp8GdnInputProjA8ResidualTolerance{1.0e-2, 1.0 / 256.0, 2.0e-2};
constexpr ReductionCriterion kGdnInputProjA4Tolerance{0.16, 4.0e-3, 0.16};
constexpr std::int32_t kA8SampleRows = 31;

int verify_output_range(std::string_view label, const GuardedBf16Tensor& output,
                        std::int32_t full_rows, std::int32_t output_row_offset,
                        std::int32_t output_rows, const quantized_weight::PackedWeight& weight,
                        std::int32_t weight_row_offset, const std::vector<float>& activation,
                        std::int32_t hidden, std::int32_t tokens,
                        const ReductionCriterion& criterion = kGdnInputProjA16Tolerance) {
    const std::vector<double> actual =
        gather_rows(output.values(), full_rows, output_row_offset, output_rows, tokens);
    const std::vector<double> expected =
        projection_oracle(weight, weight_row_offset, output_rows, activation, hidden, tokens);
    return compare(label, actual, expected, criterion);
}

int run_q4_q5_case(DevicePackedWeight& query_key, DevicePackedWeight& value_z_weight,
                   std::int32_t tokens, ops::LinearPolicy policy = ops::LinearPolicy::A16Only) {
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kQkRows      = 4096;
    constexpr std::int32_t kValueRows   = 6144;
    constexpr std::int32_t kZRows       = 6144;
    constexpr std::int32_t kRows        = kQkRows + kValueRows;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 401U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor output   = qkv.tensor();
    Tensor z_output = z.tensor();
    ops::gdn_input_proj(x, query_key.view(), value_z_weight.view(), output, z_output, policy,
                        nullptr);
    cuda_synchronize();

    const std::string suffix = std::string(ops::column_invariant(policy) ? " Q4/Q5 invariant T="
                                                                         : " Q4/Q5 A16 T=") +
                               std::to_string(tokens);
    int failures             = qkv.verify_guards("gdn qkv" + suffix);
    failures += z.verify_guards("gdn z" + suffix);
    failures += qkv.verify_fully_written("gdn qkv" + suffix);
    failures += z.verify_fully_written("gdn z" + suffix);
    failures += verify_output_range("gdn qk" + suffix, qkv, kRows, 0, kQkRows, query_key.host, 0,
                                    activation, kHidden, tokens);
    failures += verify_output_range("gdn value" + suffix, qkv, kRows, kQkRows, kValueRows,
                                    value_z_weight.host, 0, activation, kHidden, tokens);
    failures += verify_output_range("gdn z" + suffix, z, kZRows, 0, kZRows, value_z_weight.host,
                                    kValueRows, activation, kHidden, tokens);
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += query_key.verify_preserved("gdn query/key weight" + suffix);
    failures += value_z_weight.verify_preserved("gdn value/z weight" + suffix);
    return failures;
}

int run_q4_q5_graph_case(DevicePackedWeight& query_key, DevicePackedWeight& value_z_weight,
                         std::int32_t tokens,
                         ops::LinearPolicy policy = ops::LinearPolicy::A16Only) {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kQkRows    = 4096;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    constexpr std::int32_t kRows      = kQkRows + kValueRows;

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<float> activation = make_bf16_activation(kHidden, tokens, 701U + tokens);
    std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation             = to_device(activation_bits);
    GuardedBf16Tensor qkv(kRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor output     = qkv.tensor();
    Tensor z_output   = z.tensor();
    const auto launch = [&](cudaStream_t launch_stream) {
        ops::gdn_input_proj(x, query_key.view(), value_z_weight.view(), output, z_output, policy,
                            launch_stream);
    };
    launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));

    // Replay 1, checked on its own: the outputs are poisoned first, so a replay that skipped a
    // range or wrote to a stale address is caught here rather than being masked by replay 2.
    const auto verify_replay = [&](const std::vector<float>& expected, std::string_view tag) {
        int bad = qkv.verify_guards(std::string("gdn qkv") + std::string(tag));
        bad += z.verify_guards(std::string("gdn z") + std::string(tag));
        bad += qkv.verify_fully_written(std::string("gdn qkv") + std::string(tag));
        bad += z.verify_fully_written(std::string("gdn z") + std::string(tag));
        bad += verify_output_range(std::string("gdn qk") + std::string(tag), qkv, kRows, 0, kQkRows,
                                   query_key.host, 0, expected, kHidden, tokens);
        bad += verify_output_range(std::string("gdn value") + std::string(tag), qkv, kRows, kQkRows,
                                   kValueRows, value_z_weight.host, 0, expected, kHidden, tokens);
        bad += verify_output_range(std::string("gdn z") + std::string(tag), z, kZRows, 0, kZRows,
                                   value_z_weight.host, kValueRows, expected, kHidden, tokens);
        return bad;
    };
    const std::string suffix = std::string(ops::column_invariant(policy) ? " Q4/Q5 invariant graph T="
                                                                         : " Q4/Q5 A16 graph T=") +
                               std::to_string(tokens);
    qkv.repaint(stream);
    z.repaint(stream);
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int failures = verify_replay(activation, suffix + " replay1");

    // Replay 2 against a changed activation at the same captured address: a graph that baked its
    // operands would keep reporting the first input here.
    qkv.repaint(stream);
    z.repaint(stream);
    activation      = make_bf16_activation(kHidden, tokens, 811U + tokens);
    activation_bits = bf16_bits(activation);
    CUDA_CHECK(cudaMemcpyAsync(device_activation.p, activation_bits.data(),
                               activation_bits.size() * sizeof(std::uint16_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));

    failures += verify_replay(activation, suffix + " replay2");
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += query_key.verify_preserved("gdn query/key weight" + suffix);
    failures += value_z_weight.verify_preserved("gdn value/z weight" + suffix);
    return failures;
}

int run_q4_q5() {
    constexpr std::int32_t kHidden = 5120;
    DevicePackedWeight query_key(
        quantized_weight::make_patterned_weight(QType::Q4_G64_FP16, 4096, kHidden, 409U));
    DevicePackedWeight value_z_weight(
        quantized_weight::make_patterned_weight(QType::Q5_G64_FP16, 12288, kHidden, 419U));
    int failures = 0;
    // Every route boundary and both of its neighbours: the Q4/Q5 column catalog hands 1..12 to the
    // per-side small-column kernels (the K-split Q4 parent with the split4 Q5 side at 2..10 and the
    // c4 SIMT Q5 side at 11..12), 13..32 to the 32x32 tile, 33..64 to the 32x64 tile, and 65 upward
    // to the 64x128 tile that also supplies the 128-column tail slices. The Q4 K-split band (7..12)
    // is not changed by this work; its two ends are covered by the same list.
    for (const std::int32_t tokens : {1,  2,  3,  4,  5,  6,  7,  8,  9,   10,  11,  12,
                                      13, 15, 16, 17, 32, 33, 64, 65, 127, 128, 129, 193}) {
        failures += run_q4_q5_case(query_key, value_z_weight, tokens);
    }
    // One captured replay per route, including both ends of the split4 band and a 128-column tail
    // slice (129 = 128 + 1).
    for (const std::int32_t tokens : {7, 8, 9, 10, 12, 13, 33, 65, 129}) {
        failures += run_q4_q5_graph_case(query_key, value_z_weight, tokens);
    }
    // The column-invariant route: the Q4 capacity steps, the Q5 tile steps and the 32-column slices
    // of wider calls.
    constexpr auto kInvariant = ops::LinearPolicy::A16ColumnInvariant;
    for (const std::int32_t tokens : {1, 2, 3, 5, 6, 7, 12, 13, 16, 17, 24, 25, 32, 33, 65, 129, 512}) {
        failures += run_q4_q5_case(query_key, value_z_weight, tokens, kInvariant);
    }
    for (const std::int32_t tokens : {1, 9, 33}) {
        failures += run_q4_q5_graph_case(query_key, value_z_weight, tokens, kInvariant);
    }
    return failures;
}

int run_q8_case(DevicePackedWeight& parent, std::int32_t tokens,
                ops::LinearPolicy policy = ops::LinearPolicy::A16Only) {
    constexpr std::int32_t kHidden      = 2048;
    constexpr std::int32_t kQkvRows     = 8192;
    constexpr std::int32_t kZRows       = 4096;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 501U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kQkvRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor qkv_output = qkv.tensor();
    Tensor z_output   = z.tensor();
    WorkspaceArena workspace(256);
    ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, policy, workspace, nullptr);
    cuda_synchronize();

    const std::string suffix = std::string(ops::column_invariant(policy) ? " Q8 invariant T="
                                                                         : " Q8 A16 T=") +
                               std::to_string(tokens);
    int failures             = qkv.verify_guards("gdn qkv" + suffix);
    failures += z.verify_guards("gdn z" + suffix);
    failures += qkv.verify_fully_written("gdn qkv" + suffix);
    failures += z.verify_fully_written("gdn z" + suffix);
    failures += verify_output_range("gdn qkv" + suffix, qkv, kQkvRows, 0, kQkvRows, parent.host, 0,
                                    activation, kHidden, tokens);
    failures += verify_output_range("gdn z" + suffix, z, kZRows, 0, kZRows, parent.host, kQkvRows,
                                    activation, kHidden, tokens);
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("gdn parent weight" + suffix);
    return failures;
}

int run_q8() {
    constexpr std::int32_t kHidden = 2048;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, 12288, kHidden, 503U));
    int failures = 0;
    for (const std::int32_t tokens : {1, 2, 97}) { failures += run_q8_case(parent, tokens); }
    // The column-invariant route: one column, the exact instances' range and its 32-column slices.
    for (const std::int32_t tokens : {1, 2, 32, 33, 64, 65, 512}) {
        failures += run_q8_case(parent, tokens, ops::LinearPolicy::A16ColumnInvariant);
    }
    return failures;
}

// Qwen3.8-Flash-Next's [16384,2560] parent (Q8, or FP4-Local's MXFP8): the oracle criterion, and the
// exact bits of the rows of linear's packed [16384,T] result, which the model used before it wrote
// the split itself.
int run_flash_next_parent_case(DevicePackedWeight& parent, const char* format, std::int32_t tokens,
                               ops::LinearRoutes routes) {
    constexpr std::int32_t kHidden  = 2560;
    constexpr std::int32_t kQkvRows = 10240;
    constexpr std::int32_t kZRows   = 6144;
    constexpr std::int32_t kRows    = kQkvRows + kZRows;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 521U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kQkvRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    GuardedBf16Tensor packed(kRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor qkv_output    = qkv.tensor();
    Tensor z_output      = z.tensor();
    Tensor packed_output = packed.tensor();
    ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, nullptr, routes);
    WorkspaceArena workspace(256);
    ops::linear(x, parent.view(), packed_output, ops::LinearPolicy::A16Only, workspace, nullptr, routes);
    cuda_synchronize();

    const std::string suffix = std::string(" ") + format + " [16384,2560] T=" + std::to_string(tokens) +
                               (routes == ops::LinearRoutes::Q8SlicedK ? " fast" : "");
    int failures             = qkv.verify_guards("gdn qkv" + suffix);
    failures += z.verify_guards("gdn z" + suffix);
    failures += qkv.verify_fully_written("gdn qkv" + suffix);
    failures += z.verify_fully_written("gdn z" + suffix);
    // The form is linear's [16384,2560] result, so linear's A16 criterion applies: one BF16 unit
    // roundoff in relative L2, two in the gross bound.
    constexpr ReductionCriterion kLinearA16{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
    failures += verify_output_range("gdn qkv" + suffix, qkv, kQkvRows, 0, kQkvRows, parent.host, 0,
                                    activation, kHidden, tokens, kLinearA16);
    failures += verify_output_range("gdn z" + suffix, z, kZRows, 0, kZRows, parent.host, kQkvRows,
                                    activation, kHidden, tokens, kLinearA16);
    const std::vector<std::uint16_t> linear_bits = packed.bits(), qkv_bits = qkv.bits(), z_bits = z.bits();
    std::int64_t mismatches = 0;
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t row = 0; row < kRows; ++row) {
            const std::uint16_t actual = row < kQkvRows ? qkv_bits[static_cast<std::size_t>(t) * kQkvRows + row]
                                                        : z_bits[static_cast<std::size_t>(t) * kZRows + row - kQkvRows];
            mismatches += actual != linear_bits[static_cast<std::size_t>(t) * kRows + row];
        }
    }
    if (mismatches != 0) {
        std::cerr << "gdn qkv/z" << suffix << ": " << mismatches << " values differ from linear\n";
        ++failures;
    }
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("gdn parent weight" + suffix);
    return failures;
}

int run_q8_flash_next() {
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, 16384, 2560, 509U));
    int failures = 0;
    // Each route of the shape and its neighbours: register-streamed to 4 and to 8 columns, then the
    // 32x64, 32x96 and 64x128 MMA tiles.
    for (const std::int32_t tokens : {1, 2, 4, 5, 8, 9, 64, 65, 96, 97, 300}) {
        failures += run_flash_next_parent_case(parent, "Q8", tokens, ops::LinearRoutes::Exact);
    }
    // The q8-sliced-k fast routes: GEMV, sliced-K to 8, grouped sliced-K, sliced-K to 40, the MMA tiles.
    for (const std::int32_t tokens : {1, 2, 8, 9, 16, 17, 40, 41, 64, 65, 128, 129, 300, 513}) {
        failures += run_flash_next_parent_case(parent, "Q8", tokens, ops::LinearRoutes::Q8SlicedK);
    }
    return failures;
}

int run_mxfp8_flash_next() {
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_G32_E8M0, 16384, 2560, 511U));
    int failures = 0;
    // Each schedule of the shape and its neighbours: the 16-token tiles (128 and 64 rows), then the 32-,
    // 64- and wider tiles.
    for (const std::int32_t tokens : {1, 2, 8, 9, 10, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257, 300, 513}) {
        failures += run_flash_next_parent_case(parent, "MXFP8", tokens, ops::LinearRoutes::Exact);
    }
    return failures;
}

// The routes form of gdn_input_proj_conv_snapshot (the gdn-conv-fused fast path): Qwen3.8-Flash-Next's
// [16384,2560] parent at W=1, B=1..8. Its outputs carry one BF16 rounding (query/key/value) after FP32
// projection, convolution and SiLU, whose own errors are orders of magnitude below it. That is the
// profile CausalConv1dSiLU qualifies from represented inputs, so its A16 criterion applies here
// against the complete FP64 oracle: the composed route additionally rounds the projection to BF16
// before the convolution, and removing that rounding can only shrink the error. The published state
// and z are exact: the BF16 projection rows of linear's result under the same routes, and the
// initial window's two newest BF16 values.
constexpr ReductionCriterion kFlashNextConvSnapshotA16Tolerance{1.85e-3, 1.0e-3, 3.7e-3};

namespace flash_next_conv {

constexpr std::int32_t kHidden    = 2560;
constexpr std::int32_t kQueryRows = 2048;
constexpr std::int32_t kKeyRows   = 2048;
constexpr std::int32_t kValueRows = 6144;
constexpr std::int32_t kZRows     = 6144;
constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
constexpr std::int32_t kRows      = kChannels + kZRows;
constexpr std::int32_t kSlots     = 17; // more slots than any call names; the rest must stay untouched
constexpr std::int32_t kSamples   = 256; // sampled channels per query/key/value range

double silu_fp64(double value) {
    if (value >= 0.0) { return value / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return value * exponential / (1.0 + exponential);
}

// The parent, the sampled convolution channels with their exactly decoded weight rows, and a BF16
// convolution weight. Projections are O(20) for this parent at the fixture's activation, so the
// weight and history ranges keep all four convolution taps of comparable size.
struct Fixture {
    Fixture()
        : parent(quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, kRows, kHidden, 509U)) {
        for (const std::int32_t row : sampled_rows(kQueryRows, kSamples)) channels.push_back(row);
        for (const std::int32_t row : sampled_rows(kKeyRows, kSamples)) channels.push_back(kQueryRows + row);
        for (const std::int32_t row : sampled_rows(kValueRows, kSamples))
            channels.push_back(kQueryRows + kKeyRows + row);
        decoded.resize(channels.size() * static_cast<std::size_t>(kHidden));
        for (std::size_t index = 0; index < channels.size(); ++index) {
            for (std::int32_t column = 0; column < kHidden; ++column) {
                decoded[index * kHidden + column] =
                    quantized_weight::logical_weight_fp64(parent.host, channels[index], column);
            }
        }
        conv_weight.resize(static_cast<std::size_t>(kChannels) * 4);
        fill_uniform(conv_weight, 1931U, -0.5F, 0.5F);
        round_to_bf16(conv_weight);
        conv_weight_bits   = bf16_bits(conv_weight);
        device_conv_weight = to_device(conv_weight_bits);
    }

    DevicePackedWeight parent;
    std::vector<std::int32_t> channels;
    std::vector<double> decoded;
    std::vector<float> conv_weight;
    std::vector<std::uint16_t> conv_weight_bits;
    DeviceBuffer device_conv_weight;
};

std::vector<std::uint16_t> make_states(std::uint32_t seed) {
    std::vector<float> values(static_cast<std::size_t>(kSlots) * 3 * kChannels);
    fill_uniform(values, seed, -24.0F, 24.0F);
    return bf16_bits(values);
}

// Query/key/value against the FP64 oracle from `before` (the state the call started from), and the
// exact published windows, z and untouched slots against linear's packed result `linear_bits`.
int verify_call(std::string_view tag, const Fixture& fixture, std::int32_t batch,
                const std::vector<float>& activation, const std::vector<std::int32_t>& initial,
                const std::vector<std::int32_t>& snapshot, const std::vector<std::uint16_t>& before,
                const std::vector<std::uint16_t>& after, const std::vector<std::uint16_t>& linear_bits,
                const GuardedBf16Tensor& query, const GuardedBf16Tensor& key,
                const GuardedBf16Tensor& value, const GuardedBf16Tensor& z) {
    const std::string label(tag);
    int failures = 0;
    failures += query.verify_guards("conv snapshot query" + label);
    failures += key.verify_guards("conv snapshot key" + label);
    failures += value.verify_guards("conv snapshot value" + label);
    failures += z.verify_guards("conv snapshot z" + label);
    failures += query.verify_fully_written("conv snapshot query" + label);
    failures += key.verify_fully_written("conv snapshot key" + label);
    failures += value.verify_fully_written("conv snapshot value" + label);
    failures += z.verify_fully_written("conv snapshot z" + label);

    constexpr std::size_t kSlotWidth = static_cast<std::size_t>(kChannels) * 3;
    const std::vector<double> query_values = query.values(), key_values = key.values(),
                              value_values = value.values();
    std::vector<double> actual, expected;
    for (std::size_t index = 0; index < fixture.channels.size(); ++index) {
        const std::int32_t channel = fixture.channels[index];
        const double w0            = fixture.conv_weight[channel];
        const double w1            = fixture.conv_weight[kChannels + channel];
        const double w2            = fixture.conv_weight[2 * kChannels + channel];
        const double w3            = fixture.conv_weight[3 * kChannels + channel];
        for (std::int32_t b = 0; b < batch; ++b) {
            const std::size_t window = static_cast<std::size_t>(initial[b]) * kSlotWidth;
            const double h0          = bf16_to_f32(before[window + channel]);
            const double h1          = bf16_to_f32(before[window + kChannels + channel]);
            const double h2          = bf16_to_f32(before[window + 2 * kChannels + channel]);
            const float* input       = activation.data() + static_cast<std::size_t>(b) * kHidden;
            double projected         = 0.0;
            for (std::int32_t column = 0; column < kHidden; ++column) {
                projected += fixture.decoded[index * kHidden + column] * static_cast<double>(input[column]);
            }
            expected.push_back(silu_fp64(w0 * h0 + w1 * h1 + w2 * h2 + w3 * projected));
            if (channel < kQueryRows) {
                actual.push_back(query_values[static_cast<std::size_t>(b) * kQueryRows + channel]);
            } else if (channel < kQueryRows + kKeyRows) {
                actual.push_back(key_values[static_cast<std::size_t>(b) * kKeyRows + channel - kQueryRows]);
            } else {
                actual.push_back(
                    value_values[static_cast<std::size_t>(b) * kValueRows + channel - kQueryRows - kKeyRows]);
            }
        }
    }
    failures += compare("conv snapshot query/key/value" + label, actual, expected,
                        kFlashNextConvSnapshotA16Tolerance);

    const std::vector<std::uint16_t> z_bits = z.bits();
    std::int64_t z_mismatches = 0, state_mismatches = 0;
    for (std::int32_t b = 0; b < batch; ++b) {
        const std::size_t column = static_cast<std::size_t>(b) * kRows;
        for (std::int32_t row = 0; row < kZRows; ++row) {
            z_mismatches += z_bits[static_cast<std::size_t>(b) * kZRows + row] != linear_bits[column + kChannels + row];
        }
        const std::size_t source      = static_cast<std::size_t>(initial[b]) * kSlotWidth;
        const std::size_t destination = static_cast<std::size_t>(snapshot[b]) * kSlotWidth;
        for (std::int32_t channel = 0; channel < kChannels; ++channel) {
            state_mismatches += after[destination + channel] != before[source + kChannels + channel];
            state_mismatches += after[destination + kChannels + channel] != before[source + 2 * kChannels + channel];
            state_mismatches += after[destination + 2 * kChannels + channel] != linear_bits[column + channel];
        }
    }
    if (z_mismatches != 0) {
        std::cerr << "conv snapshot z" << label << ": " << z_mismatches << " values differ from linear\n";
        ++failures;
    }
    if (state_mismatches != 0) {
        std::cerr << "conv snapshot state" << label << ": " << state_mismatches
                  << " published values differ from the window shift of linear's projection\n";
        ++failures;
    }
    for (std::int32_t slot = 0; slot < kSlots; ++slot) {
        if (std::find(snapshot.begin(), snapshot.begin() + batch, slot) != snapshot.begin() + batch) { continue; }
        const auto begin = static_cast<std::ptrdiff_t>(static_cast<std::size_t>(slot) * kSlotWidth);
        if (!std::equal(before.begin() + begin, before.begin() + begin + static_cast<std::ptrdiff_t>(kSlotWidth),
                        after.begin() + begin)) {
            std::cerr << "conv snapshot state" << label << ": slot " << slot << " was modified\n";
            ++failures;
        }
    }
    return failures;
}

// One call shape: `distinct` publishes to slots disjoint from the initial ones, otherwise every
// sequence publishes over its own initial slot through one shared selector (the decode
// configuration). `replay` captures the call and replays it twice from the advancing state, the
// second time with a new activation at the captured address.
int run_case(Fixture& fixture, std::int32_t batch, ops::LinearRoutes routes, bool distinct, bool replay) {
    DeviceContext context;
    const cudaStream_t stream = context.stream;
    const std::uint32_t seed      = 17U * static_cast<std::uint32_t>(batch);
    std::vector<float> activation = make_bf16_activation(kHidden, batch, 1301U + seed);
    std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation             = to_device(activation_bits);
    std::vector<std::int32_t> initial(batch), snapshot(batch);
    for (std::int32_t b = 0; b < batch; ++b) {
        initial[b]  = 2 * (batch - 1 - b) + 1;
        snapshot[b] = distinct ? 2 * b : initial[b];
    }
    const std::vector<std::uint16_t> states = make_states(1409U + static_cast<std::uint32_t>(batch));
    DeviceBuffer device_states   = to_device(states);
    DeviceBuffer device_initial  = to_device(initial);
    DeviceBuffer device_snapshot = to_device(snapshot);
    GuardedBf16Tensor query(kQueryRows, batch), key(kKeyRows, batch), value(kValueRows, batch), z(kZRows, batch);
    GuardedBf16Tensor packed(kRows, batch);

    const Tensor x(device_activation.p, DType::BF16, {kHidden, 1, batch});
    const Tensor x_columns(device_activation.p, DType::BF16, {kHidden, batch});
    const Tensor conv_weight(fixture.device_conv_weight.p, DType::BF16, {kChannels, 4});
    Tensor conv_states(device_states.p, DType::BF16, {kChannels, 3, kSlots});
    const Tensor initial_slots(device_initial.p, DType::I32, {batch});
    const Tensor snapshot_slots(distinct ? device_snapshot.p : device_initial.p, DType::I32, {batch});
    Tensor query_out(query.data(), DType::BF16, {kQueryRows, 1, batch});
    Tensor key_out(key.data(), DType::BF16, {kKeyRows, 1, batch});
    Tensor value_out(value.data(), DType::BF16, {kValueRows, 1, batch});
    Tensor z_out(z.data(), DType::BF16, {kZRows, 1, batch});
    Tensor packed_out = packed.tensor();
    WorkspaceArena workspace(256);
    const auto launch = [&] {
        ops::gdn_input_proj_conv_snapshot(x, fixture.parent.view(), conv_weight, conv_states, initial_slots,
                                          snapshot_slots, query_out, key_out, value_out, z_out, routes, stream);
    };

    const std::string suffix = " Q8 [16384,2560] B=" + std::to_string(batch) +
                               (routes == ops::LinearRoutes::Q8SlicedK ? " fast" : " exact") +
                               (distinct ? " distinct slots" : " in place");
    // Runs one call (eager or a replay) from the current device state, then linear on the same input
    // for the exact projection bits, and verifies both.
    const auto checked_call = [&](std::string_view tag, bool graph_launch, DecodeGraphExecutable* graph) {
        const std::vector<std::uint16_t> before = from_device<std::uint16_t>(device_states, states.size());
        query.repaint(stream);
        z.repaint(stream);
        key.repaint(stream);
        value.repaint(stream);
        if (graph_launch)
            graph->launch(stream);
        else
            launch();
        ops::linear(x_columns, fixture.parent.view(), packed_out, ops::LinearPolicy::A16Only, workspace, stream,
                    routes);
        cuda_synchronize(stream);
        const std::vector<std::uint16_t> after = from_device<std::uint16_t>(device_states, states.size());
        return verify_call(suffix + std::string(tag), fixture, batch, activation, initial, snapshot, before, after,
                           packed.bits(), query, key, value, z);
    };

    int failures = checked_call("", false, nullptr);
    if (replay) {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(stream, launch);
        graph.instantiate(definition);
        failures += checked_call(" replay1", true, &graph);
        activation      = make_bf16_activation(kHidden, batch, 1511U + seed);
        activation_bits = bf16_bits(activation);
        device_activation.copy_from_host(activation_bits.data(), activation_bits.size() * sizeof(std::uint16_t));
        failures += checked_call(" replay2", true, &graph);
    }
    failures += verify_preserved("conv snapshot x" + suffix, device_activation, activation_bits);
    failures += verify_preserved("conv snapshot conv weight" + suffix, fixture.device_conv_weight,
                                 fixture.conv_weight_bits);
    failures += verify_preserved("conv snapshot initial slots" + suffix, device_initial, initial);
    if (distinct) failures += verify_preserved("conv snapshot snapshot slots" + suffix, device_snapshot, snapshot);
    failures += fixture.parent.verify_preserved("conv snapshot parent weight" + suffix);
    return failures;
}

// Calls outside the routes form's domain are rejected before any launch.
int run_rejections(Fixture& fixture) {
    DeviceBuffer activation(static_cast<std::size_t>(kHidden) * 9 * 2);
    DeviceBuffer states(static_cast<std::size_t>(kSlots) * 3 * kChannels * 2);
    DeviceBuffer slots(9 * sizeof(std::int32_t));
    DeviceBuffer outputs(static_cast<std::size_t>(kRows) * 9 * 2);
    const Tensor conv_weight(fixture.device_conv_weight.p, DType::BF16, {kChannels, 4});
    Tensor conv_states(states.p, DType::BF16, {kChannels, 3, kSlots});
    auto* base = static_cast<std::uint8_t*>(outputs.p);
    const auto attempt = [&](std::int32_t width, std::int32_t batch, const char* what) {
        const Tensor x(activation.p, DType::BF16, {kHidden, width, batch});
        const Tensor selectors(slots.p, DType::I32, {batch});
        const std::size_t columns = static_cast<std::size_t>(width) * batch;
        Tensor q(base, DType::BF16, {kQueryRows, width, batch});
        Tensor k(base + columns * kQueryRows * 2, DType::BF16, {kKeyRows, width, batch});
        Tensor v(base + columns * (kQueryRows + kKeyRows) * 2, DType::BF16, {kValueRows, width, batch});
        Tensor zz(base + columns * kChannels * 2, DType::BF16, {kZRows, width, batch});
        try {
            ops::gdn_input_proj_conv_snapshot(x, fixture.parent.view(), conv_weight, conv_states, selectors,
                                              selectors, q, k, v, zz, ops::LinearRoutes::Exact, nullptr);
        } catch (const std::invalid_argument&) {
            return 0;
        }
        std::cerr << "conv snapshot routes form accepted " << what << '\n';
        return 1;
    };
    int failures = attempt(2, 1, "W=2");
    failures += attempt(1, 9, "B=9");
    return failures;
}

} // namespace flash_next_conv

int run_q8_flash_next_conv_snapshot() {
    flash_next_conv::Fixture fixture;
    int failures = 0;
    // Every decode route of both tables: register-streamed to 4 and to 8 columns (exact), the GEMV
    // at B=1 and the sliced-K MMA at 2..8 (fast).
    for (const ops::LinearRoutes routes : {ops::LinearRoutes::Exact, ops::LinearRoutes::Q8SlicedK}) {
        for (std::int32_t batch = 1; batch <= 8; ++batch) {
            failures += flash_next_conv::run_case(fixture, batch, routes, false, false);
        }
        for (const std::int32_t batch : {1, 5, 8}) {
            failures += flash_next_conv::run_case(fixture, batch, routes, true, false);
        }
        for (const std::int32_t batch : {1, 3, 8}) {
            failures += flash_next_conv::run_case(fixture, batch, routes, false, true);
        }
    }
    failures += flash_next_conv::run_rejections(fixture);
    return failures;
}

int verify_output_range_sampled(std::string_view label, const GuardedBf16Tensor& output,
                                std::int32_t full_rows, std::int32_t output_row_offset,
                                std::int32_t output_rows,
                                const quantized_weight::PackedWeight& weight,
                                std::int32_t weight_row_offset,
                                const std::vector<float>& activation, std::int32_t hidden,
                                std::int32_t tokens, const ReductionCriterion& criterion,
                                std::int32_t sample_count = 7) {
    const std::vector<double> values     = output.values();
    const std::vector<std::int32_t> rows = sampled_rows(output_rows, sample_count);
    std::vector<std::int32_t> selected_tokens;
    for (const std::int32_t token :
         {0, 1, tokens / 4, tokens / 2, (3 * tokens) / 4, tokens - 2, tokens - 1}) {
        if (token >= 0 && token < tokens &&
            std::find(selected_tokens.begin(), selected_tokens.end(), token) ==
                selected_tokens.end()) {
            selected_tokens.push_back(token);
        }
    }
    std::vector<double> actual;
    std::vector<double> expected;
    actual.reserve(rows.size() * selected_tokens.size());
    expected.reserve(rows.size() * selected_tokens.size());
    for (const std::int32_t local_row : rows) {
        const std::int32_t output_row = output_row_offset + local_row;
        const std::int32_t weight_row = weight_row_offset + local_row;
        for (const std::int32_t token : selected_tokens) {
            actual.push_back(values[static_cast<std::size_t>(token) * full_rows + output_row]);
            expected.push_back(quantized_weight::dot_fp64(
                weight, weight_row, activation.data() + static_cast<std::size_t>(token) * hidden,
                hidden));
        }
    }
    return compare(label, actual, expected, criterion);
}

int run_nvfp4_case(DevicePackedWeight& parent, std::int32_t tokens, ops::LinearPolicy policy) {
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kQkvRows     = 10240;
    constexpr std::int32_t kZRows       = 6144;
    constexpr std::int32_t kRows        = kQkvRows + kZRows;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 601U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kQkvRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x                   = Tensor(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor qkv_output          = qkv.tensor();
    Tensor z_output            = z.tensor();
    const std::size_t capacity = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::NVFP4, kRows, kHidden, policy, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, policy, workspace, nullptr);
    cuda_synchronize();

    const bool a4                       = policy == ops::LinearPolicy::AllowA4;
    const ReductionCriterion& criterion = a4 ? kGdnInputProjA4Tolerance : kGdnInputProjA16Tolerance;
    const std::string suffix =
        std::string(" NVFP4 ") + (a4 ? "A4" : "A16") + " T=" + std::to_string(tokens);
    int failures = qkv.verify_guards("gdn qkv" + suffix);
    failures += z.verify_guards("gdn z" + suffix);
    failures += qkv.verify_fully_written("gdn qkv" + suffix);
    failures += z.verify_fully_written("gdn z" + suffix);
    failures += verify_output_range_sampled("gdn query" + suffix, qkv, kQkvRows, 0, 2048,
                                            parent.host, 0, activation, kHidden, tokens, criterion);
    failures +=
        verify_output_range_sampled("gdn key" + suffix, qkv, kQkvRows, 2048, 2048, parent.host,
                                    2048, activation, kHidden, tokens, criterion);
    failures +=
        verify_output_range_sampled("gdn value" + suffix, qkv, kQkvRows, 4096, 6144, parent.host,
                                    4096, activation, kHidden, tokens, criterion);
    failures += verify_output_range_sampled("gdn z" + suffix, z, kZRows, 0, kZRows, parent.host,
                                            kQkvRows, activation, kHidden, tokens, criterion);
    if (workspace.peak_used() != capacity) {
        std::cerr << "gdn workspace" << suffix << ": query/execution high-water mismatch\n";
        ++failures;
    }
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("gdn parent weight" + suffix);
    return failures;
}

int run_nvfp4() {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kRows   = 16384;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::NVFP4, kRows, kHidden, 607U, options));
    int failures = 0;
    failures += run_nvfp4_case(parent, 1, ops::LinearPolicy::A16Only);
    // The column-invariant route: one schedule at every T, sliced beyond its widest tile.
    for (const std::int32_t tokens : {1, 2, 5, 8, 9, 16, 17, 33, 40, 65, 129}) {
        failures += run_nvfp4_case(parent, tokens, ops::LinearPolicy::A16ColumnInvariant);
    }
    failures += run_nvfp4_case(parent, 4, ops::LinearPolicy::AllowA8);
    failures += run_nvfp4_case(parent, 1, ops::LinearPolicy::AllowA4);
    failures += run_nvfp4_case(parent, 2, ops::LinearPolicy::AllowA4);
    failures += run_nvfp4_case(parent, 17, ops::LinearPolicy::AllowA4);
    // 1023, 1024 and 1025 straddle this route's floor. 1024 was the narrowest width it
    // took before; 1025 is the first ragged one it takes now, and its last M tile holds a
    // single real token, which is the emptiest grid this route ever runs.
    failures += run_nvfp4_case(parent, 1023, ops::LinearPolicy::AllowA4);
    failures += run_nvfp4_case(parent, 1024, ops::LinearPolicy::AllowA4);
    failures += run_nvfp4_case(parent, 1025, ops::LinearPolicy::AllowA4);
    return failures;
}

int run_fp8_case(DevicePackedWeight& parent, std::int32_t tokens, ops::LinearPolicy policy,
                 bool convenience = false, bool replay = false) {
    constexpr std::int32_t kHidden  = 5120;
    constexpr std::int32_t kQkvRows = 10240;
    constexpr std::int32_t kZRows   = 6144;
    constexpr std::int32_t kRows    = kQkvRows + kZRows;
    std::vector<float> activation =
        make_bf16_activation(kHidden, tokens, 617U + static_cast<std::uint32_t>(tokens));
    std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation             = to_device(activation_bits);
    GuardedBf16Tensor qkv(kQkvRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x                   = Tensor(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor qkv_output          = qkv.tensor();
    Tensor z_output            = z.tensor();
    const std::size_t capacity = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, policy, tokens, tokens);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 256));
    WorkspaceArena workspace(DeviceSpan{scratch.data(), std::max<std::size_t>(capacity, 256)});
    DeviceContext context;
    const auto launch = [&] {
        if (convenience)
            ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, context.stream);
        else
            ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, policy, workspace,
                                context.stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    cuda_synchronize();
    if (replay) {
        definition.capture(context.stream, launch);
        graph.instantiate(definition);
    }
    int failures = 0;
    for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
        if (phase) {
            for (auto& value : activation) value = -value;
            activation_bits = bf16_bits(activation);
            device_activation.copy_from_host(activation_bits.data(), activation_bits.size() * 2);
        }
        scratch.fill(phase ? 0xa5 : 0x5a);
        cuda_synchronize();
        CUDA_CHECK(cudaMemsetAsync(qkv_output.data, 0xff, std::size_t(kQkvRows) * tokens * 2,
                                   context.stream));
        CUDA_CHECK(
            cudaMemsetAsync(z_output.data, 0xff, std::size_t(kZRows) * tokens * 2, context.stream));
        if (replay)
            graph.launch(context.stream);
        else
            launch();
        cuda_synchronize(context.stream);
        const bool a8 =
            (policy == ops::LinearPolicy::AllowA8 || policy == ops::LinearPolicy::AllowA4) &&
            tokens >= 17;
        const bool a8_residual = ops::allows_a8_residual(policy) && tokens >= 17;
        const ReductionCriterion& criterion = a8            ? kFp8GdnInputProjA8Tolerance
                                              : a8_residual ? kFp8GdnInputProjA8ResidualTolerance
                                                            : kFp8GdnInputProjA16Tolerance;
        const std::int32_t sample_count = (a8 || a8_residual) ? kA8SampleRows : 7;
        const std::string suffix = std::string(" FP8 ") +
                                   (a8 ? "A8" : (a8_residual ? "A8 residual" : "A16")) +
                                   " T=" + std::to_string(tokens);
        failures += qkv.verify_guards("gdn qkv" + suffix);
        failures += z.verify_guards("gdn z" + suffix);
        failures += qkv.verify_fully_written("gdn qkv" + suffix);
        failures += z.verify_fully_written("gdn z" + suffix);
        failures +=
            verify_output_range_sampled("gdn query" + suffix, qkv, kQkvRows, 0, 2048, parent.host,
                                        0, activation, kHidden, tokens, criterion, sample_count);
        failures +=
            verify_output_range_sampled("gdn key" + suffix, qkv, kQkvRows, 2048, 2048, parent.host,
                                        2048, activation, kHidden, tokens, criterion, sample_count);
        failures += verify_output_range_sampled("gdn value" + suffix, qkv, kQkvRows, 4096, 6144,
                                                parent.host, 4096, activation, kHidden, tokens,
                                                criterion, sample_count);
        failures += verify_output_range_sampled("gdn z" + suffix, z, kZRows, 0, kZRows, parent.host,
                                                kQkvRows, activation, kHidden, tokens, criterion,
                                                sample_count);
        if (workspace.peak_used() != capacity) {
            std::cerr << "gdn workspace" << suffix << ": query/execution high-water mismatch\n";
            ++failures;
        }
        failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
        failures += parent.verify_preserved("gdn parent weight" + suffix);
        failures += scratch.verify_guards(suffix);
    }
    return failures;
}

int run_fp8() {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kRows   = 16384;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, 613U));

    int failures = 0;
    for (int columns : {5, 8, 16, 24, 32, 33, 64, 65, 96, 97, 128, 129}) {
        failures += run_fp8_case(parent, columns, ops::LinearPolicy::A16Only);
    }
    // The column-invariant route: one schedule at every T, sliced beyond its widest tile.
    for (int columns : {1, 2, 5, 8, 9, 16, 17, 33, 40, 65, 129}) {
        failures += run_fp8_case(parent, columns, ops::LinearPolicy::A16ColumnInvariant);
    }
    const std::size_t one = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::AllowA8, 1, 1);
    const std::size_t sixteen = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::AllowA8, 16, 16);
    const std::size_t seventeen = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::AllowA8, 17, 17);
    const std::size_t forty_eight = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::AllowA8, 48, 48);
    const std::size_t hot_interval = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::AllowA8, 1, 48);
    const std::size_t exact_1024 = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::AllowA8, 1024, 1024);
    const std::size_t a16 = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::A16Only, 1, 2048);
    if (one != 0 || sixteen != 0 || seventeen == 0 || forty_eight <= seventeen ||
        hot_interval != forty_eight || exact_1024 <= forty_eight || a16 != 0) {
        std::cerr << "FP8 gdn input workspace interval contract mismatch\n";
        ++failures;
    }

    failures += run_fp8_case(parent, 1, ops::LinearPolicy::A16Only, true);
    failures += run_fp8_case(parent, 2, ops::LinearPolicy::A16Only);
    for (const std::int32_t tokens :
         {1,   2,   4,   8,   16,  17,  18,  32,  33,  48,  64,  65,  127,  128, 129,
          191, 192, 193, 255, 256, 257, 383, 384, 385, 511, 512, 513, 1024, 1025}) {
        failures += run_fp8_case(
            parent, tokens, tokens == 17 ? ops::LinearPolicy::AllowA4 : ops::LinearPolicy::AllowA8);
    }
    for (int tokens : {4, 17, 128, 385, 512, 1025})
        failures += run_fp8_case(parent, tokens, ops::LinearPolicy::AllowA8, false, true);
    // The A8 residual route: A16 below 17 columns, then its 32- and 64-token tiles and their edges.
    for (const std::int32_t tokens :
         {16, 17, 32, 33, 48, 64, 65, 96, 97, 128, 129, 192, 256, 257}) {
        failures += run_fp8_case(parent, tokens, ops::LinearPolicy::AllowA8Residual);
    }
    for (int tokens : {17, 65, 97})
        failures += run_fp8_case(parent, tokens, ops::LinearPolicy::AllowA8Residual, false, true);
    return failures;
}

} // namespace

INFERNIX_OP_TEST(gdn_input_proj) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_q4_q5();
    failures += run_q8();
    failures += run_q8_flash_next();
    failures += run_mxfp8_flash_next();
    failures += run_q8_flash_next_conv_snapshot();
    failures += run_nvfp4();
    failures += run_fp8();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_input_proj\n";
    return failures == 0 ? 0 : 1;
}
