#include "core/device.h"
#include "ninfer/ops/rope.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr float kTextTheta   = 1.0e7F;
constexpr float kVisionTheta = 10'000.0F;

// Either member of a rotated pair can approach zero through cancellation. The scale-invariant
// RoPE BF16 profile therefore bounds each output by the FP64 norm of its public input pair rather
// than by the cancelled output value.
constexpr double kRopePointwisePairRtol = 6.9e-3;

struct Geometry {
    const char* label;
    int head_dim;
    int rotary_dim;
    int axes;
    int tokens;
    float theta;
    ops::RopeScaling scaling{};
    bool scaled_api = false;
};

std::size_t dense_elements(int head_dim, int heads, int tokens) {
    return static_cast<std::size_t>(head_dim) * static_cast<std::size_t>(heads) *
           static_cast<std::size_t>(tokens);
}

std::size_t dense_index(int head_dim, int heads, int token, int head, int dim) {
    return (static_cast<std::size_t>(token) * static_cast<std::size_t>(heads) +
            static_cast<std::size_t>(head)) *
               static_cast<std::size_t>(head_dim) +
           static_cast<std::size_t>(dim);
}

std::vector<float> make_bf16_input(std::size_t elements, std::uint32_t seed) {
    std::vector<float> input(elements);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-4.0F, 4.0F);
    for (float& value : input) { value = bf16_to_f32(f32_to_bf16(distribution(generator))); }
    return input;
}

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    std::transform(values.begin(), values.end(), bits.begin(),
                   [](float value) { return f32_to_bf16(value); });
    return bits;
}

std::vector<int> make_positions(int axes, int tokens, int first_position) {
    std::vector<int> positions(static_cast<std::size_t>(axes) * tokens);
    for (int axis = 0; axis < axes; ++axis) {
        for (int token = 0; token < tokens; ++token) {
            const int step = 2 * axis + 1;
            positions[static_cast<std::size_t>(axis) * tokens + token] =
                first_position + 97 * axis + step * token;
        }
    }
    return positions;
}

// This is the sole RoPE oracle. It consumes the exact logical BF16 values represented by the
// public input and evaluates the documented split-half rotation naively in FP64. It does not
// reproduce output storage rounding, production staging, coefficient tables, range reduction,
// kernel split, or reduction order.
std::vector<double> rope_oracle(const std::vector<float>& input, const std::vector<int>& positions,
                                const Geometry& geometry, int heads) {
    std::vector<double> output(input.begin(), input.end());
    const int half = geometry.rotary_dim / 2;
    for (int token = 0; token < geometry.tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            for (int pair = 0; pair < half; ++pair) {
                int axis        = 0;
                double exponent = 0.0;
                if (geometry.axes == 2) {
                    axis     = pair / 18;
                    exponent = -2.0 * static_cast<double>(pair % 18) / 36.0;
                } else {
                    axis     = geometry.axes == 3 ? pair % 3 : 0;
                    exponent = -2.0 * static_cast<double>(pair) / geometry.rotary_dim;
                }
                double frequency       = std::pow(static_cast<double>(geometry.theta), exponent);
                double attention_scale = 1.0;
                if (geometry.scaling.factor != 1.0F && geometry.axes != 2) {
                    // Independent mathematical oracle: no production coefficient helpers,
                    // tables, float arithmetic, phase reduction, or output rounding.
                    const double pi = std::acos(-1.0);
                    const double rotations_fast =
                        geometry.scaling.original_max_positions / (64.0 * pi);
                    const double rotations_slow =
                        geometry.scaling.original_max_positions / (2.0 * pi);
                    const double log_theta = std::log(static_cast<double>(geometry.theta));
                    const double low =
                        std::max(0.0, std::floor(half * std::log(rotations_fast) / log_theta));
                    double high = std::min(static_cast<double>(geometry.rotary_dim - 1),
                                           std::ceil(half * std::log(rotations_slow) / log_theta));
                    if (high == low) { high += 0.001; }
                    const double interpolation =
                        std::min(1.0, std::max(0.0, (pair - low) / (high - low)));
                    frequency = (1.0 - interpolation) * frequency +
                                interpolation * frequency / geometry.scaling.factor;
                    attention_scale =
                        1.0 + 0.1 * std::log(static_cast<double>(geometry.scaling.factor));
                }
                const double phase =
                    static_cast<double>(
                        positions[static_cast<std::size_t>(axis) * geometry.tokens + token]) *
                    frequency;
                const double cosine  = attention_scale * std::cos(phase);
                const double sine    = attention_scale * std::sin(phase);
                const std::size_t lo = dense_index(geometry.head_dim, heads, token, head, pair);
                const std::size_t hi =
                    dense_index(geometry.head_dim, heads, token, head, pair + half);
                const double first  = static_cast<double>(input[lo]);
                const double second = static_cast<double>(input[hi]);
                output[lo]          = first * cosine - second * sine;
                output[hi]          = second * cosine + first * sine;
            }
        }
    }
    return output;
}

std::vector<std::uint16_t> make_strided_storage(const std::vector<std::uint16_t>& dense,
                                                int dense_token_elements, int token_stride,
                                                int tokens, std::uint16_t padding) {
    std::vector<std::uint16_t> storage(static_cast<std::size_t>(token_stride) * tokens, padding);
    for (int token = 0; token < tokens; ++token) {
        std::copy_n(dense.data() + static_cast<std::size_t>(token) * dense_token_elements,
                    dense_token_elements,
                    storage.data() + static_cast<std::size_t>(token) * token_stride);
    }
    return storage;
}

std::vector<double> gather_dense(const std::vector<std::uint16_t>& storage,
                                 int dense_token_elements, int token_stride, int tokens) {
    std::vector<double> dense(static_cast<std::size_t>(dense_token_elements) * tokens);
    for (int token = 0; token < tokens; ++token) {
        for (int element = 0; element < dense_token_elements; ++element) {
            dense[static_cast<std::size_t>(token) * dense_token_elements + element] =
                static_cast<double>(
                    bf16_to_f32(storage[static_cast<std::size_t>(token) * token_stride + element]));
        }
    }
    return dense;
}

int verify_rope_profile(const std::string& label, const std::vector<double>& got,
                        const std::vector<double>& expected, const std::vector<float>& input,
                        const Geometry& geometry, int heads) {
    if (got.size() != expected.size() || got.size() != input.size()) {
        std::cerr << label << ": profile input size mismatch\n";
        return 1;
    }

    double case_max_abs            = 0.0;
    double case_required_pair_rtol = 0.0;
    int violations                 = 0;
    const int half                 = geometry.rotary_dim / 2;
    for (int token = 0; token < geometry.tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            for (int dim = 0; dim < geometry.rotary_dim; ++dim) {
                const int pair          = dim < half ? dim : dim - half;
                const std::size_t index = dense_index(geometry.head_dim, heads, token, head, dim);
                const std::size_t lo    = dense_index(geometry.head_dim, heads, token, head, pair);
                const std::size_t hi =
                    dense_index(geometry.head_dim, heads, token, head, pair + half);
                if (!std::isfinite(got[index]) || !std::isfinite(expected[index])) {
                    std::cerr << label << ": non-finite output at index=" << index << '\n';
                    return 1;
                }
                const double abs_error = std::abs(got[index] - expected[index]);
                const double pair_norm =
                    std::hypot(static_cast<double>(input[lo]), static_cast<double>(input[hi]));
                const double required_pair_rtol =
                    pair_norm == 0.0
                        ? (abs_error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
                        : abs_error / pair_norm;
                if (required_pair_rtol > case_required_pair_rtol) {
                    case_required_pair_rtol = required_pair_rtol;
                }
                case_max_abs = std::max(case_max_abs, abs_error);
                if (abs_error > kRopePointwisePairRtol * pair_norm) {
                    ++violations;
                    if (violations == 1) {
                        std::cerr << label << ": pointwise BF16 profile mismatch at index=" << index
                                  << " abs_error=" << abs_error << " pair_norm=" << pair_norm
                                  << '\n';
                    }
                }
            }
        }
    }

    report_scaled_pointwise_stats(
        label, static_cast<std::int64_t>(geometry.tokens) * heads * geometry.rotary_dim,
        case_max_abs, case_required_pair_rtol, kRopePointwisePairRtol);
    if (violations != 0) {
        std::cerr << label << ": " << violations << " values exceed the unified RoPE profile\n";
        return 1;
    }
    return 0;
}

int verify_passthrough(const std::string& label, const std::vector<std::uint16_t>& got_storage,
                       const std::vector<std::uint16_t>& before_dense, int head_dim, int heads,
                       int tokens, int rotary_dim, int token_stride) {
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            for (int dim = rotary_dim; dim < head_dim; ++dim) {
                const std::size_t dense  = dense_index(head_dim, heads, token, head, dim);
                const std::size_t stored = static_cast<std::size_t>(token) * token_stride +
                                           static_cast<std::size_t>(head) * head_dim + dim;
                if (got_storage[stored] != before_dense[dense]) {
                    std::cerr << label << ": non-rotary dimension changed at token=" << token
                              << " head=" << head << " dim=" << dim << '\n';
                    return 1;
                }
            }
        }
    }
    return 0;
}

int verify_padding(const std::string& label, const std::vector<std::uint16_t>& storage,
                   int dense_token_elements, int token_stride, int tokens, std::uint16_t padding) {
    for (int token = 0; token < tokens; ++token) {
        for (int element = dense_token_elements; element < token_stride; ++element) {
            if (storage[static_cast<std::size_t>(token) * token_stride + element] != padding) {
                std::cerr << label << ": token padding changed at token=" << token
                          << " element=" << element << '\n';
                return 1;
            }
        }
    }
    return 0;
}

int run_pair_case(DeviceExecutionView execution, const Geometry& geometry, int q_heads, int k_heads,
                  int first_position, int q_padding = 0, int k_padding = 0, int lane_width = 0,
                  bool graph = false) {
    constexpr std::uint16_t kPadding = 0x3f81U;
    const int q_dense_per_token      = geometry.head_dim * q_heads;
    const int k_dense_per_token      = geometry.head_dim * k_heads;
    const int q_stride               = q_dense_per_token + q_padding;
    const int k_stride               = k_dense_per_token + k_padding;

    const auto q =
        make_bf16_input(dense_elements(geometry.head_dim, q_heads, geometry.tokens), 0x1001U);
    const auto k =
        make_bf16_input(dense_elements(geometry.head_dim, k_heads, geometry.tokens), 0x2001U);
    const auto q_before = to_bf16_bits(q);
    const auto k_before = to_bf16_bits(k);
    const auto q_storage =
        make_strided_storage(q_before, q_dense_per_token, q_stride, geometry.tokens, kPadding);
    const auto k_storage =
        make_strided_storage(k_before, k_dense_per_token, k_stride, geometry.tokens, kPadding);
    auto positions = make_positions(geometry.axes, geometry.tokens, first_position);
    if (lane_width != 0) {
        for (int axis = 0; axis < geometry.axes; ++axis)
            for (int t = 0; t < geometry.tokens; ++t)
                positions[axis * geometry.tokens + t] = first_position + 1009 * (t / lane_width) +
                                                        97 * axis +
                                                        (2 * axis + 1) * (t % lane_width);
    }
    const auto q_expected = rope_oracle(q, positions, geometry, q_heads);
    const auto k_expected = rope_oracle(k, positions, geometry, k_heads);

    GuardedDeviceBuffer q_device(q_storage.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer k_device(k_storage.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer position_device(positions.size() * sizeof(int));
    q_device.copy_from_host(q_storage.data(), q_device.bytes());
    k_device.copy_from_host(k_storage.data(), k_device.bytes());
    position_device.copy_from_host(positions.data(), position_device.bytes());

    Tensor position_tensor(position_device.data(), DType::I32, {geometry.tokens, geometry.axes});
    Tensor q_tensor(q_device.data(), DType::BF16, {geometry.head_dim, q_heads, geometry.tokens});
    Tensor k_tensor(k_device.data(), DType::BF16, {geometry.head_dim, k_heads, geometry.tokens});
    q_tensor.nb[2] = static_cast<std::int64_t>(q_stride) * sizeof(std::uint16_t);
    k_tensor.nb[2] = static_cast<std::int64_t>(k_stride) * sizeof(std::uint16_t);

    const auto prepared = ops::prepare_rope(geometry.rotary_dim, geometry.theta, geometry.scaling);
    const auto launch   = [&](DeviceExecutionView launch_execution) {
        if (geometry.scaled_api) {
            ops::rope(position_tensor, prepared, q_tensor, k_tensor, launch_execution);
        } else {
            ops::rope(position_tensor, geometry.rotary_dim, geometry.theta, q_tensor, k_tensor,
                      launch_execution);
        }
    };
    launch(execution);
    cuda_synchronize();

    if (graph) {
        cudaStream_t stream;
        cudaGraph_t captured;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch(execution.on_stream(stream));
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) {
            CUDA_CHECK(cudaMemcpyAsync(q_device.data(), q_storage.data(), q_device.bytes(),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(k_device.data(), k_storage.data(), k_device.bytes(),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(captured));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    const auto q_got        = from_device<std::uint16_t>(q_device.data(), q_storage.size());
    const auto k_got        = from_device<std::uint16_t>(k_device.data(), k_storage.size());
    const std::string label = std::string(geometry.label) +
                              " factor=" + std::to_string(geometry.scaling.factor) +
                              " pos=" + std::to_string(first_position) +
                              " axes=" + std::to_string(geometry.axes);
    int failures            = 0;
    failures += verify_rope_profile(
        label + " q", gather_dense(q_got, q_dense_per_token, q_stride, geometry.tokens), q_expected,
        q, geometry, q_heads);
    failures += verify_rope_profile(
        label + " k", gather_dense(k_got, k_dense_per_token, k_stride, geometry.tokens), k_expected,
        k, geometry, k_heads);
    failures += verify_passthrough(label + " q", q_got, q_before, geometry.head_dim, q_heads,
                                   geometry.tokens, geometry.rotary_dim, q_stride);
    failures += verify_passthrough(label + " k", k_got, k_before, geometry.head_dim, k_heads,
                                   geometry.tokens, geometry.rotary_dim, k_stride);
    failures +=
        verify_padding(label + " q", q_got, q_dense_per_token, q_stride, geometry.tokens, kPadding);
    failures +=
        verify_padding(label + " k", k_got, k_dense_per_token, k_stride, geometry.tokens, kPadding);
    failures += verify_exact((label + " positions").c_str(),
                             from_device<int>(position_device.data(), positions.size()), positions);
    if (geometry.scaled_api && geometry.scaling.factor == 1.0F) {
        q_device.copy_from_host(q_storage.data(), q_device.bytes());
        k_device.copy_from_host(k_storage.data(), k_device.bytes());
        ops::rope(position_tensor, geometry.rotary_dim, geometry.theta, q_tensor, k_tensor,
                  execution);
        cuda_synchronize();
        failures += verify_exact("factor1 pair q identity",
                                 from_device<std::uint16_t>(q_device.data(), q_storage.size()),
                                 q_got);
        failures += verify_exact("factor1 pair k identity",
                                 from_device<std::uint16_t>(k_device.data(), k_storage.size()),
                                 k_got);
    }
    failures += q_device.verify_guards((label + " q guards").c_str());
    failures += k_device.verify_guards((label + " k guards").c_str());
    failures += position_device.verify_guards((label + " position guards").c_str());
    return failures;
}

int run_single_case(DeviceExecutionView execution, const Geometry& geometry, int heads,
                    int first_position, int padding = 0) {
    constexpr std::uint16_t kPadding = 0x3f81U;
    const int dense_per_token        = geometry.head_dim * heads;
    const int token_stride           = dense_per_token + padding;
    const auto input =
        make_bf16_input(dense_elements(geometry.head_dim, heads, geometry.tokens), 0x3001U);
    const auto before = to_bf16_bits(input);
    const auto storage =
        make_strided_storage(before, dense_per_token, token_stride, geometry.tokens, kPadding);
    const auto positions = make_positions(geometry.axes, geometry.tokens, first_position);
    const auto expected  = rope_oracle(input, positions, geometry, heads);

    GuardedDeviceBuffer device(storage.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer position_device(positions.size() * sizeof(int));
    device.copy_from_host(storage.data(), device.bytes());
    position_device.copy_from_host(positions.data(), position_device.bytes());

    Tensor position_tensor(position_device.data(), DType::I32, {geometry.tokens, geometry.axes});
    Tensor tensor(device.data(), DType::BF16, {geometry.head_dim, heads, geometry.tokens});
    tensor.nb[2] = static_cast<std::int64_t>(token_stride) * sizeof(std::uint16_t);
    const auto prepared = ops::prepare_rope(geometry.rotary_dim, geometry.theta, geometry.scaling);
    if (geometry.scaled_api) {
        ops::rope(position_tensor, prepared, tensor, execution);
    } else {
        ops::rope(position_tensor, geometry.rotary_dim, geometry.theta, tensor, execution);
    }
    cuda_synchronize();

    const auto got          = from_device<std::uint16_t>(device.data(), storage.size());
    const std::string label = std::string(geometry.label) +
                              " single factor=" + std::to_string(geometry.scaling.factor) +
                              " pos=" + std::to_string(first_position) +
                              " axes=" + std::to_string(geometry.axes);
    int failures            = 0;
    failures += verify_rope_profile(
        label, gather_dense(got, dense_per_token, token_stride, geometry.tokens), expected, input,
        geometry, heads);
    failures += verify_passthrough(label, got, before, geometry.head_dim, heads, geometry.tokens,
                                   geometry.rotary_dim, token_stride);
    failures +=
        verify_padding(label, got, dense_per_token, token_stride, geometry.tokens, kPadding);
    failures += verify_exact((label + " positions").c_str(),
                             from_device<int>(position_device.data(), positions.size()), positions);
    if (geometry.scaled_api && geometry.scaling.factor == 1.0F) {
        device.copy_from_host(storage.data(), device.bytes());
        ops::rope(position_tensor, geometry.rotary_dim, geometry.theta, tensor, execution);
        cuda_synchronize();
        failures += verify_exact("factor1 single identity",
                                 from_device<std::uint16_t>(device.data(), storage.size()), got);
    }
    failures += device.verify_guards((label + " guards").c_str());
    failures += position_device.verify_guards((label + " position guards").c_str());
    return failures;
}

int run_vision_packed_case(DeviceExecutionView execution) {
    constexpr int kHeadDim = 72;
    constexpr int kHeads   = 16;
    constexpr int kTokens  = 11;
    constexpr int kPlane   = kHeadDim * kHeads;
    constexpr int kStride  = 3 * kPlane;
    constexpr Geometry geometry{"vision packed qkv", kHeadDim, kHeadDim, 2, kTokens, kVisionTheta};

    const auto q      = make_bf16_input(dense_elements(kHeadDim, kHeads, kTokens), 0x4001U);
    const auto k      = make_bf16_input(dense_elements(kHeadDim, kHeads, kTokens), 0x4002U);
    const auto v      = make_bf16_input(dense_elements(kHeadDim, kHeads, kTokens), 0x4003U);
    const auto q_bits = to_bf16_bits(q);
    const auto k_bits = to_bf16_bits(k);
    const auto v_bits = to_bf16_bits(v);
    std::vector<std::uint16_t> packed(static_cast<std::size_t>(kStride) * kTokens);
    for (int token = 0; token < kTokens; ++token) {
        const std::size_t dense_base  = static_cast<std::size_t>(token) * kPlane;
        const std::size_t packed_base = static_cast<std::size_t>(token) * kStride;
        std::copy_n(q_bits.data() + dense_base, kPlane, packed.data() + packed_base);
        std::copy_n(k_bits.data() + dense_base, kPlane, packed.data() + packed_base + kPlane);
        std::copy_n(v_bits.data() + dense_base, kPlane, packed.data() + packed_base + 2 * kPlane);
    }
    std::vector<int> positions(2 * kTokens);
    for (int token = 0; token < kTokens; ++token) {
        positions[token]           = token / 4;
        positions[kTokens + token] = token % 4;
    }
    const auto q_expected = rope_oracle(q, positions, geometry, kHeads);
    const auto k_expected = rope_oracle(k, positions, geometry, kHeads);

    GuardedDeviceBuffer packed_device(packed.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer position_device(positions.size() * sizeof(int));
    packed_device.copy_from_host(packed.data(), packed_device.bytes());
    position_device.copy_from_host(positions.data(), position_device.bytes());

    auto* packed_data = static_cast<std::uint16_t*>(packed_device.data());
    Tensor position_tensor(position_device.data(), DType::I32, {kTokens, 2});
    Tensor q_tensor(packed_data, DType::BF16, {kHeadDim, kHeads, kTokens});
    Tensor k_tensor(packed_data + kPlane, DType::BF16, {kHeadDim, kHeads, kTokens});
    q_tensor.nb[2] = static_cast<std::int64_t>(kStride) * sizeof(std::uint16_t);
    k_tensor.nb[2] = static_cast<std::int64_t>(kStride) * sizeof(std::uint16_t);
    ops::rope(position_tensor, kHeadDim, kVisionTheta, q_tensor, k_tensor, execution);
    cuda_synchronize();

    const auto got = from_device<std::uint16_t>(packed_device.data(), packed.size());
    std::vector<std::uint16_t> q_storage(static_cast<std::size_t>(kStride) * kTokens);
    std::vector<std::uint16_t> k_storage(static_cast<std::size_t>(kStride) * kTokens);
    for (int token = 0; token < kTokens; ++token) {
        const std::size_t packed_base = static_cast<std::size_t>(token) * kStride;
        std::copy_n(got.data() + packed_base, kPlane, q_storage.data() + packed_base);
        std::copy_n(got.data() + packed_base + kPlane, kPlane, k_storage.data() + packed_base);
    }

    int failures = 0;
    failures +=
        verify_rope_profile("vision packed q", gather_dense(q_storage, kPlane, kStride, kTokens),
                            q_expected, q, geometry, kHeads);
    failures +=
        verify_rope_profile("vision packed k", gather_dense(k_storage, kPlane, kStride, kTokens),
                            k_expected, k, geometry, kHeads);
    for (int token = 0; token < kTokens; ++token) {
        const std::size_t dense_base  = static_cast<std::size_t>(token) * kPlane;
        const std::size_t packed_base = static_cast<std::size_t>(token) * kStride;
        if (!std::equal(v_bits.begin() + static_cast<std::ptrdiff_t>(dense_base),
                        v_bits.begin() + static_cast<std::ptrdiff_t>(dense_base + kPlane),
                        got.begin() + static_cast<std::ptrdiff_t>(packed_base + 2 * kPlane))) {
            std::cerr << "vision packed qkv: V plane changed at token=" << token << '\n';
            ++failures;
            break;
        }
    }
    failures += verify_exact("vision packed positions",
                             from_device<int>(position_device.data(), positions.size()), positions);
    failures += packed_device.verify_guards("vision packed qkv guards");
    failures += position_device.verify_guards("vision packed position guards");
    return failures;
}

int run_scaling_validation(DeviceExecutionView execution) {
    int failures = 0;
    Tensor positions(nullptr, DType::I32, {1});
    Tensor q(nullptr, DType::BF16, {256, 24, 1});
    Tensor k(nullptr, DType::BF16, {256, 4, 1});
    positions.ne[0] = q.ne[2] = k.ne[2] = 0;
    for (const ops::RopeScaling scaling :
         {ops::RopeScaling{0.0F, 262144},
          {0.99F, 262144},
          {4.01F, 262144},
          {std::numeric_limits<float>::infinity(), 262144},
          {std::numeric_limits<float>::quiet_NaN(), 262144},
          {2.0F, 0},
          {1.0F, 0}}) {
        for (bool single : {false, true}) {
            try {
                if (single) {
                    ops::rope(positions, ops::prepare_rope(64, kTextTheta, scaling), q, execution);
                } else {
                    ops::rope(positions, ops::prepare_rope(64, kTextTheta, scaling), q, k,
                              execution);
                }
                std::cerr << "invalid YaRN scaling accepted\n";
                ++failures;
            } catch (const std::invalid_argument&) {}
        }
    }
    for (float factor : {1.0F, 4.0F}) {
        ops::rope(positions, ops::prepare_rope(64, kTextTheta, {factor, 262144}), q, k,
                  execution);
        ops::rope(positions, ops::prepare_rope(64, kTextTheta, {factor, 262144}), q, execution);
    }
    Tensor vision_positions(nullptr, DType::I32, {1, 2});
    Tensor vision(nullptr, DType::BF16, {72, 16, 1});
    vision_positions.ne[0] = vision.ne[2] = 0;
    try {
        ops::rope(vision_positions, ops::prepare_rope(72, kVisionTheta, {2.0F, 262144}), vision,
                  execution);
        ++failures;
    } catch (const std::invalid_argument&) {}
    try {
        ops::rope(positions, ops::prepare_rope(64, 1.0F, {2.0F, 262144}), q, execution);
        ++failures;
    } catch (const std::invalid_argument&) {}
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    DeviceContext device;
    const auto execution = device.execution_view().on_stream(nullptr);
    int failures         = 0;

    failures += run_scaling_validation(execution);
    failures += run_pair_case(
        execution, {"vision default identity", 72, 72, 2, 8, kVisionTheta, {1.0F, 262144}, true},
        16, 16, 0);
    failures += run_single_case(execution,
                                {"vision single default identity", 72, 72, 2, 8, kVisionTheta,
                                 {1.0F, 262144}, true},
                                16, 0);

    // Text pair form: both registered checkpoint geometries, decode/prefill, and 1-D/MRoPE.
    for (int axes : {1, 3}) {
        for (int width : {2, 7, 16}) {
            for (int batch : {1, 8}) {
                failures += run_pair_case(
                    execution, {"27b dflash2 lanes", 256, 64, axes, width * batch, kTextTheta}, 24,
                    4, 131072, batch == 8 ? 16 : 0, batch == 8 ? 8 : 0, width,
                    width == 16 && batch == 8);
            }
        }
    }
    failures += run_pair_case(execution, {"27b text decode", 256, 64, 1, 1, kTextTheta}, 24, 4, 31);
    failures += run_pair_case(execution, {"27b text mrope prefill", 256, 64, 3, 128, kTextTheta},
                              24, 4, 4096);
    failures += run_pair_case(
        execution, {"35b text native-context tail", 256, 64, 1, 7, kTextTheta}, 16, 2, 262'137);
    failures +=
        run_pair_case(execution, {"35b text mrope", 256, 64, 3, 7, kTextTheta}, 16, 2, 2048, 16, 8);

    // Exercise both public forms on either side of the large-block residency boundary.
    const int large_block_capacity = execution.multiprocessor_count * 6;
    for (int tokens : {large_block_capacity, large_block_capacity + 1}) {
        failures += run_pair_case(execution,
                                  {"27b text residency boundary", 256, 64, 1, tokens, kTextTheta},
                                  24, 4, 8192);
        failures += run_single_case(
            execution, {"27b text mrope residency boundary", 256, 64, 3, tokens, kTextTheta}, 24,
            8192);
    }

    // MTP bulk K append uses the single-tensor form; proposal tail uses the pair form above.
    failures +=
        run_single_case(execution, {"27b mtp k mrope", 256, 64, 3, 128, kTextTheta}, 4, 8192);
    failures +=
        run_single_case(execution, {"35b mtp k text", 256, 64, 1, 5, kTextTheta}, 2, 16384, 8);

    failures += run_vision_packed_case(execution);

    // DFlash proposal consumes 2..16 tokens; context append uses the single-K form.
    failures += run_pair_case(execution, {"35b dflash proposal", 128, 128, 1, 16, kTextTheta}, 32,
                              8, 262'128);
    failures += run_single_case(execution, {"35b dflash context k", 128, 128, 1, 128, kTextTheta},
                                8, 131'072, 16);

    // YaRN: all public text modes, factor boundaries/interior, native and extended
    // positions, prefill/decode, padded storage, single/pair calls and captured arguments.
    for (float factor : {1.0F, 1.5F, 2.0F, 4.0F}) {
        for (int axes : {1, 3}) {
            for (int position : {0, 262143, 524288, 1048575}) {
                // The unchanged FP32 default route is qualified at its original native context.
                if (factor == 1.0F && position != 0) { continue; }
                failures += run_pair_case(
                    execution,
                    {"yarn text", 256, 64, axes, 8, kTextTheta, {factor, 262144}, true}, 24, 4,
                    position, 16, 8, 0, factor == 4.0F);
                failures += run_single_case(
                    execution,
                    {"yarn single", 256, 64, axes, 7, kTextTheta, {factor, 262144}, true}, 2,
                    position, 8);
            }
        }
    }
    failures += run_pair_case(
        execution, {"yarn dflash", 128, 128, 1, 16, kTextTheta, {4.0F, 262144}, true}, 32, 8,
        1048575);
    failures += run_single_case(
        execution, {"yarn full rotary", 256, 256, 1, 128, kTextTheta, {2.0F, 32768}, true}, 3,
        65535, 1);
    failures += run_pair_case(
        execution, {"yarn narrow rotary", 256, 2, 1, 1, 10000.0F, {4.0F, 1}, true}, 3, 1,
        -123456);
    failures += run_single_case(
        execution, {"yarn equal correction", 256, 2, 1, 1, kTextTheta, {2.0F, 262144}, true}, 1,
        17);

    // Qualify every reused fixed/split launch policy boundary against the same FP64 oracle,
    // including both sides of the device's large-block residency boundary.
    for (int tokens : {1, 6, 7, 16, 17, 400, 401, large_block_capacity, large_block_capacity + 1,
                       1024, 1025}) {
        for (int axes : {1, 3}) {
            failures += run_pair_case(execution,
                                      {"prepared tuned text", 256, 64, axes, tokens, kTextTheta,
                                       {2.0F, 262144}, true},
                                      24, 4, 524288);
            failures += run_pair_case(execution,
                                      {"prepared tuned moe", 256, 64, axes, tokens, kTextTheta,
                                       {4.0F, 262144}, true},
                                      16, 2, 1048575);
        }
        failures += run_pair_case(execution,
                                  {"prepared tuned dflash", 128, 128, 1, tokens, kTextTheta,
                                   {2.0F, 262144}, true},
                                  32, 8, 524288);
        failures += run_single_case(execution,
                                    {"prepared tuned single", 128, 128, 1, tokens, kTextTheta,
                                     {4.0F, 262144}, true},
                                    8, 1048575);
    }
    failures += run_single_case(
        execution,
        {"prepared generic stride", 256, 64, 3, 7, kTextTheta, {2.0F, 262144}, true}, 4, 524288,
        1);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " rope correctness\n";
    return failures == 0 ? 0 : 1;
}
