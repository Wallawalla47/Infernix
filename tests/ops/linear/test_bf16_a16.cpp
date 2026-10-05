#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::direct_bf16_weight;

constexpr ReductionCriterion kA16Tolerance{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

// The activation pattern of one column does not depend on the call's T, so an oracle computed for
// the first tokens of a long call serves every shorter call too.
std::uint16_t activation_bit(std::int32_t column, std::int32_t token) {
    const int centered = ((column * 29 + token * 71 + 17) & 0xff) - 128;
    return f32_to_bf16(static_cast<float>(centered) * (1.0F / 512.0F));
}

std::vector<std::uint16_t> make_activation_bits(std::int32_t hidden, std::int32_t tokens) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(hidden) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t column = 0; column < hidden; ++column) {
            result[static_cast<std::size_t>(token) * hidden + column] =
                activation_bit(column, token);
        }
    }
    return result;
}

std::vector<float> materialize(std::span<const std::uint16_t> bits) {
    std::vector<float> result(bits.size());
    for (std::size_t index = 0; index < bits.size(); ++index) {
        result[index] = bf16_to_f32(bits[index]);
    }
    return result;
}

std::vector<double> oracle_all_rows(const HostWeight& weight, std::span<const float> activation) {
    std::vector<double> result(static_cast<std::size_t>(weight.n));
    const unsigned available   = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t threads = std::min(weight.n, static_cast<std::int32_t>(available));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    for (std::int32_t thread = 0; thread < threads; ++thread) {
        const std::int32_t begin =
            static_cast<std::int32_t>((static_cast<std::int64_t>(weight.n) * thread) / threads);
        const std::int32_t end = static_cast<std::int32_t>(
            (static_cast<std::int64_t>(weight.n) * (thread + 1)) / threads);
        workers.emplace_back([&, begin, end] {
            for (std::int32_t row = begin; row < end; ++row) {
                result[static_cast<std::size_t>(row)] = dot_fp64(weight, row, activation);
            }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
    return result;
}

std::vector<std::int32_t> sampled_rows(std::int32_t rows) {
    std::vector<std::int32_t> result{0, 1, rows / 4, rows / 2, (3 * rows) / 4, rows - 2, rows - 1};
    if (rows == 14336) {
        result.insert(result.end(), {1023, 6143, 6144, 7167, 7168, 13311, 13312});
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<std::int32_t> sampled_tokens(std::int32_t tokens) {
    if (tokens <= 32) {
        std::vector<std::int32_t> result(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) {
            result[static_cast<std::size_t>(token)] = token;
        }
        return result;
    }
    std::vector<std::int32_t> result{0, 1, tokens / 2, tokens - 2, tokens - 1};
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

int run_bf16_linear_case(DeviceWeight& weight, std::int32_t tokens, bool replay = false) {
    const std::int32_t rows                    = weight.host.n;
    const std::int32_t hidden                  = weight.host.k;
    std::vector<std::uint16_t> activation_bits = make_activation_bits(hidden, tokens);
    std::vector<float> activation              = materialize(activation_bits);
    DeviceBuffer device_activation             = to_device(activation_bits);
    GuardedDeviceBuffer guarded_output(static_cast<std::size_t>(rows) * tokens *
                                       sizeof(std::uint16_t));
    guarded_output.fill(0xff);

    Tensor x(device_activation.p, DType::BF16, {hidden, tokens});
    Tensor output(guarded_output.data(), DType::BF16, {rows, tokens});
    DeviceArena workspace(256);
    ops::linear(x, weight.view(), output, ops::LinearPolicy::A16Only, workspace, nullptr);
    cuda_synchronize();

    if (replay) {
        DeviceContext context;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(context.stream, [&] {
            ops::linear(x, weight.view(), output, ops::LinearPolicy::A16Only, workspace,
                        context.stream);
        });
        graph.instantiate(definition);
        graph.launch(context.stream);
        cuda_synchronize();
        for (auto& bits : activation_bits) bits ^= 0x8000;
        activation = materialize(activation_bits);
        device_activation.copy_from_host(activation_bits.data(), device_activation.bytes);
        guarded_output.fill(0xff);
        cuda_synchronize();
        graph.launch(context.stream);
        cuda_synchronize();
    }
    const std::string suffix = " T=" + std::to_string(tokens) + (replay ? " graph" : " eager");
    int failures             = guarded_output.verify_guards("BF16_A16 Linear output" + suffix);
    const std::vector<std::uint16_t> output_bits =
        from_device<std::uint16_t>(guarded_output.data(), static_cast<std::size_t>(rows) * tokens);
    for (std::size_t index = 0; index < output_bits.size(); ++index) {
        const std::uint16_t bits = output_bits[index];
        if (!std::isfinite(bf16_to_f32(bits))) {
            std::cerr << "BF16_A16 Linear output" << suffix << " element " << index
                      << " is not finite\n";
            ++failures;
            break;
        }
    }

    std::vector<double> actual;
    std::vector<double> expected;
    if (tokens == 1) {
        const std::vector<double> complete =
            oracle_all_rows(weight.host, std::span<const float>(activation));
        actual.reserve(rows);
        for (const std::uint16_t bits : output_bits) { actual.push_back(bf16_to_f32(bits)); }
        expected = complete;
    } else {
        const std::vector<std::int32_t> sampled       = sampled_rows(rows);
        const std::vector<std::int32_t> token_samples = sampled_tokens(tokens);
        actual.reserve(sampled.size() * token_samples.size());
        expected.reserve(actual.capacity());
        for (const std::int32_t row : sampled) {
            for (const std::int32_t token : token_samples) {
                actual.push_back(
                    bf16_to_f32(output_bits[static_cast<std::size_t>(token) * rows + row]));
                expected.push_back(dot_fp64(
                    weight.host, row,
                    std::span<const float>(
                        activation.data() + static_cast<std::size_t>(token) * hidden, hidden)));
            }
        }
    }
    failures += verify_reduction("BF16_A16 Linear [" + std::to_string(rows) + "," +
                                     std::to_string(hidden) + "]" + suffix,
                                 actual, expected, kA16Tolerance);
    const std::vector<std::uint16_t> activation_after =
        from_device<std::uint16_t>(device_activation, activation_bits.size());
    if (activation_after != activation_bits) {
        std::cerr << "BF16_A16 Linear" << suffix << " modified its activation\n";
        ++failures;
    }
    failures += weight.verify_preserved("BF16_A16 Linear weight" + suffix);
    return failures;
}

int run_selector_linear() {
    constexpr int n = 256, k = 5120, max_t = 2048;
    DeviceWeight weight(make_patterned(n, k, 419U));
    const auto bits       = make_activation_bits(k, max_t);
    const auto activation = materialize(bits);
    std::vector<double> oracle(n * max_t);
    std::vector<std::thread> workers;
    const int threads = std::min(32U, std::max(1U, std::thread::hardware_concurrency()));
    for (int worker = 0; worker < threads; ++worker)
        workers.emplace_back([&, worker] {
            for (int index = worker; index < n * max_t; index += threads) {
                const int token = index / n, row = index % n;
                oracle[index] = dot_fp64(weight.host, row,
                                         std::span<const float>(activation.data() + token * k, k));
            }
        });
    for (auto& worker : workers) worker.join();
    DeviceBuffer input = to_device(bits);
    auto negative      = bits;
    for (auto& value : negative) value ^= 0x8000;
    DeviceContext context;
    int failures   = 0;
    const auto run = [&](int tokens, bool replay) {
        const auto capacity = ops::linear_workspace_capacity_bytes(
            QType::BF16, n, k, ops::LinearPolicy::A16Only, tokens, tokens);
        DeviceArena scratch(std::max<std::size_t>(capacity, 256));
        GuardedDeviceBuffer output_buffer(static_cast<std::size_t>(n) * tokens * 2);
        Tensor x(input.p, DType::BF16, {k, tokens});
        Tensor output(output_buffer.data(), DType::BF16, {n, tokens});
        const auto launch = [&] {
            ops::linear(x, weight.view(), output, ops::LinearPolicy::A16Only, scratch,
                        context.stream);
        };
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        cuda_synchronize();
        if (replay) {
            definition.capture(context.stream, launch);
            graph.instantiate(definition);
        }
        const std::string label =
            "BF16 selector T=" + std::to_string(tokens) + (replay ? " graph" : " eager");
        for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
            const auto& represented = phase == 0 ? bits : negative;
            if (phase) input.copy_from_host(represented.data(), input.bytes);
            output_buffer.fill(0xff);
            cuda_synchronize();
            if (replay)
                graph.launch(context.stream);
            else
                launch();
            cuda_synchronize(context.stream);
            failures += output_buffer.verify_guards(label);
            if (scratch.peak_used() > capacity || scratch.used() != 0) {
                std::cerr << label << ": workspace query/scope mismatch\n";
                ++failures;
            }
            const auto actual_bits = from_device<std::uint16_t>(output_buffer.data(), n * tokens);
            std::vector<double> actual(actual_bits.size()),
                expected(oracle.begin(), oracle.begin() + n * tokens);
            for (std::size_t i = 0; i < actual.size(); ++i) actual[i] = bf16_to_f32(actual_bits[i]);
            if (phase)
                for (auto& value : expected) value = -value;
            failures += verify_reduction(label, actual, expected, kA16Tolerance);
            if (from_device<std::uint16_t>(input, bits.size()) != represented) {
                std::cerr << label << ": modified input\n";
                ++failures;
            }
        }
        if (replay) input.copy_from_host(bits.data(), input.bytes);
    };
    for (int tokens = 1; tokens <= 120; ++tokens) run(tokens, false);
    for (int tokens : {121, 127, 128,  129,  159,  160,  161,  162,  256,  639,  640,
                       641, 642, 1023, 1024, 1025, 1026, 1279, 1280, 1281, 1282, 2048})
        run(tokens, false);
    for (int tokens : {1,   7,   8,   15,  16,  63,  64,  65,   76,   77,   80,   81,
                       119, 120, 129, 160, 161, 640, 641, 1024, 1025, 1280, 1281, 2048})
        run(tokens, true);
    failures += weight.verify_preserved("BF16 selector weight");
    return failures;
}

// FP64 oracle for the listed tokens of the fixed activation pattern, result[i * n + row] for
// tokens[i]: every complete dot product accumulated naively in FP64. Rows are taken eight at a time
// so one token's activation stays in cache while it meets eight weight rows.
std::vector<double> oracle_tokens(const HostWeight& weight, std::span<const std::int32_t> tokens) {
    const std::int32_t n = weight.n, k = weight.k;
    const auto columns   = static_cast<std::size_t>(k);
    std::vector<double> w(weight.bits.size());
    for (std::size_t index = 0; index < w.size(); ++index) {
        w[index] = bf16_to_f32(weight.bits[index]);
    }
    std::vector<double> x(tokens.size() * columns);
    for (std::size_t token = 0; token < tokens.size(); ++token) {
        for (std::int32_t column = 0; column < k; ++column) {
            x[token * columns + column] = bf16_to_f32(activation_bit(column, tokens[token]));
        }
    }
    std::vector<double> result(tokens.size() * static_cast<std::size_t>(n));
    constexpr std::int32_t kRows = 8;
    const std::int32_t blocks    = (n + kRows - 1) / kRows;
    std::atomic<std::int32_t> next{0};
    const auto work = [&] {
        for (std::int32_t block = next++; block < blocks; block = next++) {
            const std::int32_t first = block * kRows, count = std::min(kRows, n - first);
            const double* rows       = w.data() + static_cast<std::size_t>(first) * columns;
            for (std::size_t token = 0; token < tokens.size(); ++token) {
                const double* activation = x.data() + token * columns;
                double sums[kRows]       = {};
                for (std::int32_t column = 0; column < k; ++column) {
                    for (std::int32_t row = 0; row < count; ++row) {
                        sums[row] += rows[row * columns + column] * activation[column];
                    }
                }
                for (std::int32_t row = 0; row < count; ++row) {
                    result[token * static_cast<std::size_t>(n) + first + row] = sums[row];
                }
            }
        }
    };
    std::vector<std::thread> workers;
    const unsigned threads = std::max(1U, std::thread::hardware_concurrency());
    for (unsigned thread = 0; thread < threads; ++thread) workers.emplace_back(work);
    for (std::thread& worker : workers) worker.join();
    return result;
}

// One call at T tokens compared against the oracle at `checked` tokens (oracle[i * n + row] for
// checked[i]). A replay captures the call in a graph, negates the activation and launches again.
int run_vision_case(const DeviceWeight& weight, std::int32_t tokens,
                    std::span<const std::int32_t> checked, std::span<const double> oracle,
                    bool replay, cudaStream_t stream) {
    const std::int32_t rows = weight.host.n, hidden = weight.host.k;
    auto activation_bits    = make_activation_bits(hidden, tokens);
    DeviceBuffer device_activation = to_device(activation_bits);
    const std::size_t outputs      = static_cast<std::size_t>(rows) * tokens;
    GuardedDeviceBuffer guarded_output(outputs * sizeof(std::uint16_t));
    guarded_output.fill(0xff);
    Tensor x(device_activation.p, DType::BF16, {hidden, tokens});
    Tensor output(guarded_output.data(), DType::BF16, {rows, tokens});
    DeviceArena workspace(256);
    const auto launch = [&] {
        ops::linear(x, weight.view(), output, ops::LinearPolicy::A16Only, workspace, stream);
    };
    double sign = 1.0;
    if (replay) {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(stream, launch);
        graph.instantiate(definition);
        graph.launch(stream);
        cuda_synchronize(stream);
        for (auto& bits : activation_bits) bits ^= 0x8000;
        device_activation.copy_from_host(activation_bits.data(), device_activation.bytes);
        guarded_output.fill(0xff);
        graph.launch(stream);
        sign = -1.0;
    } else {
        launch();
    }
    cuda_synchronize(stream);

    const std::string label = "BF16_A16 vision Linear [" + std::to_string(rows) + "," +
                              std::to_string(hidden) + "] T=" + std::to_string(tokens) +
                              (replay ? " graph" : " eager");
    int failures = guarded_output.verify_guards(label);
    const auto output_bits = from_device<std::uint16_t>(guarded_output.data(), outputs);
    for (std::size_t index = 0; index < outputs; ++index) {
        if (!std::isfinite(bf16_to_f32(output_bits[index]))) {
            std::cerr << label << ": element " << index << " is not finite\n";
            ++failures;
            break;
        }
    }
    std::vector<double> actual, expected;
    actual.reserve(checked.size() * rows);
    expected.reserve(checked.size() * rows);
    for (std::size_t index = 0; index < checked.size(); ++index) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::size_t at = static_cast<std::size_t>(checked[index]) * rows + row;
            actual.push_back(bf16_to_f32(output_bits[at]));
            expected.push_back(sign * oracle[index * rows + row]);
        }
    }
    failures += verify_reduction(label, actual, expected, kA16Tolerance);
    if (from_device<std::uint16_t>(device_activation, activation_bits.size()) != activation_bits) {
        std::cerr << label << ": modified its activation\n";
        ++failures;
    }
    failures += weight.verify_preserved(label + " weight");
    return failures;
}

// The BF16 vision tower projections of Qwen3.8-Flash-Next and Qwen3.5 (and Quasar's merger
// [5120,4608]) are registered problems. Calls up to T = 4100 compare every output with the FP64
// oracle; T = 16384 and 65536 compare complete columns at tile seams and both ends. [4304,1152]
// and [1152,4304] exercise the N and K tails of the tail-capable TMA route.
int run_vision_bf16_linear() {
    constexpr std::int32_t kFullTokens = 4100;
    const std::vector<std::pair<int, int>> shapes = {
        {1152, 1536}, // patch embedding
        {3456, 1152}, // fused QKV
        {1152, 1152}, // attention output
        {4304, 1152}, // MLP fc1: N tail
        {1152, 4304}, // MLP fc2: K tail
        {4608, 4608}, // merger fc1
        {2560, 4608}, // merger fc2 (Flash-Next)
        {5120, 4608}, // merger fc2 (Quasar)
    };
    std::vector<std::int32_t> first_tokens(kFullTokens);
    for (std::int32_t token = 0; token < kFullTokens; ++token) first_tokens[token] = token;
    const auto first = [&](std::int32_t tokens) {
        return std::span<const std::int32_t>(first_tokens).first(static_cast<std::size_t>(tokens));
    };
    DeviceContext context;
    int failures = 0;
    for (const auto& [n, k] : shapes) {
        const DeviceWeight weight(make_patterned(n, k, 424U));
        const std::vector<double> oracle = oracle_tokens(weight.host, first_tokens);
        for (const std::int32_t tokens :
             {1,   2,    3,    8,    9,    15,   16,   17,   31,   32,   33,   63,   64,
              65,  127,  128,  129,  255,  256,  257,  383,  384,  385,  511,  512,  513,
              767, 768,  769,  1023, 1024, 1025, 1151, 1152, 1153, 2047, 2048, 2049, 2303,
              2304, 2305, 4095, 4096, 4100}) {
            failures +=
                run_vision_case(weight, tokens, first(tokens), oracle, false, context.stream);
        }
        for (const std::int32_t tokens : {1, 9, 65, 1025, 4100}) {
            failures += run_vision_case(weight, tokens, first(tokens), oracle, true, context.stream);
        }
        for (const std::int32_t tokens : {16384, 65536}) {
            std::vector<std::int32_t> checked{0,    1,          127,          128,
                                              4095, 4096,       4097,         tokens / 2,
                                              tokens - 257,     tokens - 256, tokens - 2,
                                              tokens - 1};
            std::sort(checked.begin(), checked.end());
            checked.erase(std::unique(checked.begin(), checked.end()), checked.end());
            failures += run_vision_case(weight, tokens, checked, oracle_tokens(weight.host, checked),
                                        false, context.stream);
        }
    }
    return failures;
}

int run_general_bf16_linear() {
    // Shapes outside the specialised table route to the general runtime-shape GEMM fallback. Cover
    // the minimal tile, non-tile-aligned extents (the row/column boundary guards), multi-tile
    // grids, and the QAT full-precision vocab head's real shape.
    int failures = 0;
    const std::vector<std::pair<int, int>> shapes = {
        {32, 32},   // exactly one 32x32 tile
        {50, 70},   // not a multiple of 32 in either extent
        {63, 33},   // odd extents
        {128, 256}, // multiple tiles
        {248320, 5120},  // QAT bf16 lm_head [vocab, hidden]
    };
    for (const auto& [n, k] : shapes) {
        DeviceWeight weight(make_patterned(n, k, 421U));
        for (int tokens : {1, 2, 32, 33, 128}) {
            failures += run_bf16_linear_case(weight, tokens);
        }
    }
    // Qwen3.8-Flash-Next's BF16 projections (hyper-connection down/up, GDN, QSA, shared expert,
    // PLE). Decode-sized calls (T <= 8) take the small-T path (a row per warp for N <= 256, as the GDN
    // control [96, 2560], at every width); T = 9 and 64 the tile GEMM.
    const std::vector<std::pair<int, int>> qwen4_exp_shapes = {
        {324, 10240}, {320, 10240}, {10240, 320}, {16384, 2560}, {96, 2560},
        {2560, 6144}, {13952, 2560}, {1280, 2560}, {2560, 640}, {12800, 2560},
    };
    for (const auto& [n, k] : qwen4_exp_shapes) {
        DeviceWeight weight(make_patterned(n, k, 423U));
        for (int tokens : {1, 2, 3, 5, 8, 9, 64}) { failures += run_bf16_linear_case(weight, tokens); }
        if (n <= 256) {
            for (int tokens : {4, 6, 7}) { failures += run_bf16_linear_case(weight, tokens); }
        }
        failures += run_bf16_linear_case(weight, 1, true);
        failures += run_bf16_linear_case(weight, 8, true);
    }
    return failures;
}

int run_bf16_linear() {
    int failures = 0;
    DeviceWeight attention_weight(make_patterned(14336, 5120, 401U));
    DeviceWeight output_weight(make_patterned(5120, 6144, 409U));
    for (DeviceWeight* weight : {&attention_weight, &output_weight}) {
        for (int tokens = 1; tokens <= 33; ++tokens) {
            failures += run_bf16_linear_case(*weight, tokens);
        }
        for (int tokens :
             {63, 64, 65, 66, 95, 96, 97, 98, 127, 128, 129, 130, 191, 192, 193, 194, 1024, 1536}) {
            failures += run_bf16_linear_case(*weight, tokens);
        }
        for (int tokens : {3, 7, 13, 19, 23, 25, 29, 33, 64, 65, 97, 129, 193}) {
            failures += run_bf16_linear_case(*weight, tokens, true);
        }
    }
    failures += run_selector_linear();
    failures += run_vision_bf16_linear();
    failures += run_general_bf16_linear();
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = run_bf16_linear();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " BF16_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "BF16_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
