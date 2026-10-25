#include "ops/op_family.h"
#include "core/weight.h"
#include "infernix/ops/dynamic_grouped_conv.h"
#include "core/decode_graph.h"

#include "ops/input_projection_test_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

// The drafter geometry under test (run_profile): hidden width H and G = H / 16 groups.
std::int32_t g_hidden = 0;
constexpr std::int32_t kMaximumWidth = 16;
std::int32_t g_groups = 0;
constexpr std::int32_t kTaps         = 2;
constexpr std::int32_t kSides        = 2;
constexpr std::int32_t kMaximumBatch = 8;
constexpr std::int32_t kMaximumCols  = kMaximumWidth * kMaximumBatch;

// Sampled output rows: the candidates below H, then H - 1.
std::vector<std::int32_t> g_sampled_rows;
constexpr std::array<std::int32_t, 27> kRowCandidates{
    0,   7,   8,   15,  16,   17,   31,   32,   63,   64,   127,  128,  255, 256,
    319, 320, 639, 640, 1023, 1024, 2047, 2048, 2559, 2560, 4095, 4096, 5119};
constexpr ReductionCriterion kCriterion{/*relative_l2=*/3.2e-3,
                                        /*gross_absolute=*/1.0e-3,
                                        /*gross_relative=*/2.0e-3};

std::uint32_t mix32(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

float signed_pattern(std::uint32_t index, std::uint32_t seed, float scale) {
    const std::uint32_t mixed = mix32(index ^ (seed * 0x9e3779b9U));
    int centered              = static_cast<int>((mixed >> 8) & 0xffU) - 128;
    if (centered == 0) { centered = (index & 1U) == 0 ? 1 : -1; }
    return static_cast<float>(centered) * scale;
}

std::vector<float> make_activation(std::int32_t input_rows) {
    std::vector<float> result(static_cast<std::size_t>(input_rows) * kMaximumCols);
    for (std::int32_t col = 0; col < kMaximumCols; ++col) {
        for (std::int32_t row = 0; row < input_rows; ++row) {
            result[static_cast<std::size_t>(col) * input_rows + row] = signed_pattern(
                static_cast<std::uint32_t>(static_cast<std::uint64_t>(col) * input_rows + row),
                101U + static_cast<std::uint32_t>(input_rows), 1.0F / 32768.0F);
        }
    }
    round_to_bf16(result);
    return result;
}

std::vector<std::uint16_t> make_base_kernel() {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(g_hidden) * kTaps * kSides);
    for (std::int32_t side = 0; side < kSides; ++side) {
        for (std::int32_t tap = 0; tap < kTaps; ++tap) {
            for (std::int32_t row = 0; row < g_hidden; ++row) {
                const float center =
                    side == 1 ? (tap == 0 ? 0.375F : -0.15625F) : (tap == 0 ? -0.25F : 0.0625F);
                const float variation =
                    signed_pattern(static_cast<std::uint32_t>((side * kTaps + tap) * g_hidden + row),
                                   211U, 1.0F / 32768.0F);
                result[static_cast<std::size_t>(side * kTaps + tap) * g_hidden + row] =
                    f32_to_bf16(center + variation);
            }
        }
    }
    return result;
}

std::vector<std::uint16_t> make_finish_delta() {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(g_groups) * kTaps * kMaximumCols);
    for (std::int32_t col = 0; col < kMaximumCols; ++col) {
        for (std::int32_t tap = 0; tap < kTaps; ++tap) {
            for (std::int32_t group = 0; group < g_groups; ++group) {
                const float center    = tap == 0 ? 0.046875F : -0.03125F;
                const float variation = signed_pattern(
                    static_cast<std::uint32_t>((col * kTaps + tap) * g_groups + group), 307U,
                    1.0F / 65536.0F);
                result[(static_cast<std::size_t>(col) * kTaps + tap) * g_groups + group] =
                    f32_to_bf16(center + variation);
            }
        }
    }
    return result;
}

std::vector<std::uint16_t> make_residual() {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(g_hidden) * kMaximumCols);
    for (std::int32_t col = 0; col < kMaximumCols; ++col) {
        for (std::int32_t row = 0; row < g_hidden; ++row) {
            const float value = signed_pattern(
                static_cast<std::uint32_t>(static_cast<std::uint64_t>(col) * g_hidden + row), 401U,
                1.0F / 4096.0F);
            result[static_cast<std::size_t>(col) * g_hidden + row] = f32_to_bf16(value);
        }
    }
    return result;
}

std::vector<double> projection_oracle(const quantized_weight::PackedWeight& weight,
                                      const std::vector<float>& activation,
                                      std::int32_t input_rows) {
    std::vector<double> projection(g_sampled_rows.size() * kMaximumCols);
    for (std::size_t sample = 0; sample < g_sampled_rows.size(); ++sample) {
        for (std::int32_t col = 0; col < kMaximumCols; ++col) {
            projection[sample * kMaximumCols + col] = quantized_weight::dot_fp64(
                weight, g_sampled_rows[sample],
                activation.data() + static_cast<std::size_t>(col) * input_rows, input_rows);
        }
    }
    return projection;
}

std::vector<double> result_oracle(const std::vector<double>& projection,
                                  const std::vector<std::uint16_t>& base_kernel,
                                  const std::vector<std::uint16_t>& finish_delta,
                                  const std::vector<std::uint16_t>& residual, std::int32_t width,
                                  std::int32_t batch_size) {
    const std::int32_t cols = width * batch_size;
    std::vector<double> result;
    result.reserve(g_sampled_rows.size() * static_cast<std::size_t>(cols));
    for (std::size_t sample = 0; sample < g_sampled_rows.size(); ++sample) {
        const std::int32_t row   = g_sampled_rows[sample];
        const std::int32_t group = row / 16;
        for (std::int32_t col = 0; col < cols; ++col) {
            const std::size_t residual_offset = static_cast<std::size_t>(col) * g_hidden + row;
            const std::size_t delta0_offset =
                (static_cast<std::size_t>(col) * kTaps) * g_groups + group;
            const double current_coefficient =
                static_cast<double>(bf16_to_f32(base_kernel[2 * g_hidden + row])) +
                static_cast<double>(bf16_to_f32(finish_delta[delta0_offset]));
            double value = static_cast<double>(bf16_to_f32(residual[residual_offset])) +
                           current_coefficient * projection[sample * kMaximumCols + col];
            if ((col % width) != 0) {
                const double previous_coefficient =
                    static_cast<double>(bf16_to_f32(base_kernel[3 * g_hidden + row])) +
                    static_cast<double>(bf16_to_f32(finish_delta[delta0_offset + g_groups]));
                value += previous_coefficient * projection[sample * kMaximumCols + col - 1];
            }
            result.push_back(value);
        }
    }
    return result;
}

std::vector<double> gather_actual(const void* device, std::int32_t width, std::int32_t batch_size) {
    const std::int32_t cols = width * batch_size;
    const std::vector<double> full =
        from_device_bf16(device, static_cast<std::size_t>(g_hidden) * cols);
    std::vector<double> result;
    result.reserve(g_sampled_rows.size() * static_cast<std::size_t>(cols));
    for (const std::int32_t row : g_sampled_rows) {
        for (std::int32_t col = 0; col < cols; ++col) {
            result.push_back(full[static_cast<std::size_t>(col) * g_hidden + row]);
        }
    }
    return result;
}

int verify_preserved(std::string_view label, const DeviceBuffer& device,
                     std::span<const std::uint16_t> expected) {
    const std::string owned(label);
    return verify_exact(owned.c_str(), from_device<std::uint16_t>(device, expected.size()),
                        std::vector<std::uint16_t>(expected.begin(), expected.end()));
}

int run_profile(std::int32_t hidden, std::int32_t input_rows) {
    g_hidden = hidden;
    g_groups = hidden / 16;
    g_sampled_rows.clear();
    for (const std::int32_t row : kRowCandidates) {
        if (row < hidden - 1) { g_sampled_rows.push_back(row); }
    }
    g_sampled_rows.push_back(hidden - 1);
    const std::vector<float> activation = make_activation(input_rows);
    std::vector<std::uint16_t> activation_bits(activation.size());
    std::transform(activation.begin(), activation.end(), activation_bits.begin(), f32_to_bf16);
    const std::vector<std::uint16_t> base_kernel  = make_base_kernel();
    const std::vector<std::uint16_t> finish_delta = make_finish_delta();
    const std::vector<std::uint16_t> residual     = make_residual();

    quantized_weight::PatternedWeightOptions weight_options;
    weight_options.row_split_scale = quantized_weight::RowSplitScalePattern::Tiny;
    weight_options.row_split_codes = quantized_weight::RowSplitCodePattern::Hashed;
    input_projection::DevicePackedWeight projection_weight(quantized_weight::make_patterned_weight(
        QType::Q8_G32_FP16, g_hidden, input_rows, 503U + static_cast<std::uint32_t>(input_rows),
        weight_options));
    const std::vector<double> projection =
        projection_oracle(projection_weight.host, activation, input_rows);

    DeviceBuffer activation_device = to_device(activation_bits);
    DeviceBuffer base_device       = to_device(base_kernel);
    DeviceBuffer delta_device      = to_device(finish_delta);
    const std::size_t capacity     = ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
        g_hidden, input_rows, 2, 16, 1, kMaximumBatch);

    Tensor base(base_device.p, DType::BF16, {g_hidden, kTaps, kSides});
    const Weight weight = projection_weight.view();
    int failures        = 0;
    for (int width = 2; width <= 16; ++width)
        for (int batch_size = 1; batch_size <= 8; ++batch_size) {
            const std::size_t bytes =
                static_cast<std::size_t>(g_hidden) * width * batch_size * sizeof(std::uint16_t);
            const auto exact = ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                g_hidden, input_rows, width, width, batch_size, batch_size);
            if (exact > capacity)
                throw std::runtime_error("workspace interval does not cover exact shape");
            GuardedDeviceBuffer scratch(exact);
            WorkspaceArena workspace(DeviceSpan{scratch.data(), exact});
            GuardedDeviceBuffer residual_device(bytes);
            Tensor x(activation_device.p, DType::BF16, {input_rows, width, batch_size});
            Tensor delta(delta_device.p, DType::BF16, {g_groups, kTaps, width, batch_size});
            Tensor residual_view(residual_device.data(), DType::BF16, {g_hidden, width, batch_size});
            const auto expected =
                result_oracle(projection, base_kernel, finish_delta, residual, width, batch_size);
            const bool graph_case =
                (width == 3 && batch_size == 3) || (width == 13 && batch_size == 5) ||
                (width == 11 && batch_size == 7) || (width == 14 && batch_size == 6) ||
                (width == 15 && batch_size == 6) || (width == 16 && batch_size == 8);
            for (int replay = 0; replay < (graph_case ? 2 : 1); ++replay) {
                residual_device.copy_from_host(residual.data(), bytes);
                workspace.reset_peak();
                const auto launch = [&](cudaStream_t stream) {
                    ops::linear_dynamic_grouped_conv_add(x, weight, base, delta, residual_view,
                                                         workspace, stream);
                };
                if (replay) {
                    cudaStream_t stream = nullptr;
                    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                               "create stream");
                    {
                        DecodeGraphDefinition definition;
                        DecodeGraphExecutable executable;
                        definition.capture(stream, [&] { launch(stream); });
                        executable.instantiate(definition);
                        for (int iteration = 0; iteration < 2; ++iteration) {
                            cuda_check(cudaMemcpyAsync(residual_device.data(), residual.data(),
                                                       bytes, cudaMemcpyHostToDevice, stream),
                                       "reset graph residual");
                            executable.launch(stream);
                            cuda_check(cudaStreamSynchronize(stream), "graph synchronize");
                        }
                    }
                    cuda_check(cudaStreamDestroy(stream), "destroy stream");
                } else
                    launch(nullptr);
                cuda_synchronize();
                const std::string label = "dynamic conv add H=" + std::to_string(g_hidden) +
                                          " C=" + std::to_string(input_rows) +
                                          " W=" + std::to_string(width) +
                                          " B=" + std::to_string(batch_size) +
                                          " graph=" + std::to_string(replay);
                failures += verify_reduction(
                    label, gather_actual(residual_device.data(), width, batch_size), expected,
                    kCriterion);
                failures += residual_device.verify_guards(label);
                failures += scratch.verify_guards(label + " workspace");
                if (workspace.used() != 0 || workspace.peak_used() != exact) {
                    std::cerr << label << ": workspace mismatch\n";
                    ++failures;
                }
            }
        }

    failures +=
        verify_preserved("linear dynamic grouped conv add x", activation_device, activation_bits);
    failures += verify_preserved("linear dynamic grouped conv add base", base_device, base_kernel);
    failures +=
        verify_preserved("linear dynamic grouped conv add delta", delta_device, finish_delta);
    failures += projection_weight.verify_preserved("linear dynamic grouped conv add weight");
    return failures;
}

} // namespace

INFERNIX_OP_TEST(linear_dynamic_grouped_conv_add) {
    try {
        if (cuda_unavailable()) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        // Qwen3.8-27B's drafter, Qwen3.8-Flash-Next's and Qwen3.5-122B-A10B's.
        const int failures = run_profile(5120, 4096) + run_profile(5120, 17408) +
                             run_profile(2560, 4096) + run_profile(2560, 7680) +
                             run_profile(3072, 4096) + run_profile(3072, 9216);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " linear_dynamic_grouped_conv_add\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "linear_dynamic_grouped_conv_add test: " << error.what() << '\n';
        return 1;
    }
}
