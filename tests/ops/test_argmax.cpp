#include "infernix/ops/argmax.h"
#include "ops/op_family.h"
#include "core/device.h"
#include <algorithm>
#include "ops/op_tester.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

std::vector<std::int32_t> argmax_oracle(const std::vector<float>& logits,
                                        std::int32_t physical_rows, std::int32_t tokens,
                                        std::int32_t valid_rows) {
    std::vector<std::int32_t> expected(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        std::int32_t best      = 0;
        float best_value       = logits[base];
        for (std::int32_t row = 1; row < valid_rows; ++row) {
            const float value = logits[base + row];
            if (value > best_value) {
                best       = row;
                best_value = value;
            }
        }
        expected[static_cast<std::size_t>(token)] = best;
    }
    return expected;
}

// The logits' bytes as the device reads them: the FP32 values, or each value's BF16 bits.
std::vector<std::uint8_t> logit_bytes(const std::vector<float>& values, DType dtype) {
    std::vector<std::uint8_t> bytes(values.size() * (dtype == DType::FP32 ? sizeof(float) : sizeof(std::uint16_t)));
    if (dtype == DType::FP32) {
        std::memcpy(bytes.data(), values.data(), bytes.size());
    } else {
        for (std::size_t i = 0; i < values.size(); ++i) {
            const std::uint16_t bits = f32_to_bf16(values[i]);
            std::memcpy(bytes.data() + i * sizeof(bits), &bits, sizeof(bits));
        }
    }
    return bytes;
}

// Values on the logits' grid: FP32 ones also differ below BF16's step (2^-20 apart), so the FP32
// comparison is exercised at its own resolution. Each column keeps an exact two-way tie at its maximum
// (the lower row wins).
std::vector<float> make_logits(std::int32_t physical_rows, std::int32_t tokens, std::int32_t valid_rows,
                               DType dtype) {
    const bool fp32 = dtype == DType::FP32;
    const auto grid = [&](float value) { return fp32 ? value : bf16_to_f32(f32_to_bf16(value)); };
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        for (std::int32_t row = 0; row < physical_rows; ++row) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u +
                                        static_cast<std::uint32_t>(token + 1) * 1013904223u;
            float value = -24.0f + static_cast<float>(mixed % 3072u) * (1.0f / 256.0f);
            if (fp32) { value += static_cast<float>((mixed >> 12) % 256u) * 0x1p-20f; }
            logits[base + row] = grid(value);
        }

        std::int32_t first  = 17 + token * 7919;
        std::int32_t second = valid_rows - 1 - token * 65537;
        first %= valid_rows;
        second %= valid_rows;
        if (second < 0) { second += valid_rows; }
        if (first == second) { second = (second + 1) % valid_rows; }
        if (second < first) {
            const std::int32_t temporary = first;
            first                        = second;
            second                       = temporary;
        }
        logits[base + first]  = grid(token % 3 == 0 ? -0.5f : 32.0f + static_cast<float>(token));
        logits[base + second] = logits[base + first];

        if (token == 0) {
            std::fill(logits.begin() + base, logits.begin() + base + valid_rows, -4.0f);
        }
        if (valid_rows < physical_rows) {
            logits[base + valid_rows]        = 32768.0f;
            logits[base + physical_rows - 1] = 65536.0f;
        }
    }
    return logits;
}

// argmax_with_probability on `dtype` logits (values it represents) against independent FP64 oracles: the
// rows exactly (argmax_oracle) and each column's softmax probability of its argmax,
// 1 / sum_v exp(l_v - l_max) in FP64 over the represented logits. Fast exponentials, the running rescale
// and FP32 sums of at most 248,077 terms in [0, 1] stay within 1e-4 relative. Two launches give the same
// bits (one fixed order per column).
int probability_case(const std::vector<float>& logits, std::int32_t physical_rows, std::int32_t valid_rows,
                     std::int32_t tokens, const std::string& label, DType dtype) {
    const auto rows = argmax_oracle(logits, physical_rows, tokens, valid_rows);
    std::vector<double> want(static_cast<std::size_t>(tokens));
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::size_t base = static_cast<std::size_t>(t) * physical_rows;
        const double top       = logits[base + static_cast<std::size_t>(rows[static_cast<std::size_t>(t)])];
        double sum             = 0.0;
        for (std::int32_t v = 0; v < valid_rows; ++v) { sum += std::exp(static_cast<double>(logits[base + v]) - top); }
        want[static_cast<std::size_t>(t)] = 1.0 / sum;
    }
    int failures           = 0;
    const auto bytes       = logit_bytes(logits, dtype);
    const std::string name = label + " with probability";
    GuardedDeviceBuffer device_logits(bytes.size());
    device_logits.copy_from_host(bytes.data(), bytes.size());
    Tensor logits_tensor(device_logits.data(), dtype, {physical_rows, tokens});
    const std::size_t workspace_bytes =
        ops::argmax_with_probability_workspace_bytes(dtype, physical_rows, valid_rows, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes + 256));
    std::vector<std::vector<float>> launches;
    for (int launch = 0; launch < 2; ++launch) {
        GuardedDeviceBuffer device_rows(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
        GuardedDeviceBuffer device_probability(static_cast<std::size_t>(tokens) * sizeof(float));
        device_rows.fill(0xcd);
        device_probability.fill(0xcd);
        Tensor rows_tensor(device_rows.data(), DType::I32, {tokens});
        Tensor probability_tensor(device_probability.data(), DType::FP32, {tokens});
        ops::argmax_with_probability(logits_tensor, rows_tensor, probability_tensor, valid_rows, workspace, nullptr);
        cuda_synchronize();
        failures += verify_exact((name + " rows").c_str(),
                                 from_device<std::int32_t>(device_rows.data(), static_cast<std::size_t>(tokens)),
                                 rows);
        launches.push_back(from_device<float>(device_probability.data(), static_cast<std::size_t>(tokens)));
        failures += device_rows.verify_guards((name + " rows").c_str());
        failures += device_probability.verify_guards((name + " probability").c_str());
    }
    failures += workspace.used() == 0 ? 0 : 1; // the call releases its workspace
    std::size_t far = 0, unequal = 0;
    for (std::int32_t t = 0; t < tokens; ++t) {
        const auto i = static_cast<std::size_t>(t);
        far += std::fabs(launches[0][i] - want[i]) > 1e-4 * want[i];
        unequal += std::memcmp(&launches[0][i], &launches[1][i], sizeof(float)) != 0;
    }
    if (far != 0 || unequal != 0) {
        std::cerr << "FAIL: " << name << ": " << far << " columns outside 1e-4 of FP64, " << unequal
                  << " columns differ between launches\n";
        ++failures;
    }
    failures += device_logits.verify_guards((name + " logits").c_str());
    return failures;
}

// Draft-head-like logits on the BF16 grid: one column per draft confidence, a leading row of logit `lead`
// over unit-spread noise, so the probabilities span the extent policy's buckets.
std::vector<float> make_draft_logits(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t tokens) {
    const auto grid = [](float value) { return bf16_to_f32(f32_to_bf16(value)); };
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::size_t base = static_cast<std::size_t>(t) * physical_rows;
        for (std::int32_t row = 0; row < physical_rows; ++row) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 2654435761u + static_cast<std::uint32_t>(t) * 40503u;
            logits[base + row]        = grid(-2.0f + static_cast<float>(mixed % 1024u) * (1.0f / 512.0f));
        }
        const float lead = 8.0f + 2.0f * static_cast<float>(t);
        logits[base + static_cast<std::size_t>((t * 104729 + 31) % valid_rows)] = grid(lead);
    }
    return logits;
}

int run_case(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t tokens,
             DType dtype = DType::BF16) {
    auto logits   = make_logits(physical_rows, tokens, valid_rows, dtype);
    auto expected = argmax_oracle(logits, physical_rows, tokens, valid_rows);
    auto bytes    = logit_bytes(logits, dtype);

    GuardedDeviceBuffer device_logits(bytes.size());
    GuardedDeviceBuffer device_output(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    device_logits.copy_from_host(bytes.data(), bytes.size());
    device_output.fill(0xcd);

    Tensor logits_tensor(device_logits.data(), dtype, {physical_rows, tokens});
    Tensor output_tensor(device_output.data(), DType::I32, {tokens});
    ops::argmax(logits_tensor, output_tensor, valid_rows, nullptr);
    cuda_synchronize();

    if (physical_rows == 248320 && tokens == 128) {
        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ops::argmax(logits_tensor, output_tensor, valid_rows, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int t = 0; t < tokens; ++t)
            logits[static_cast<std::size_t>(t) * physical_rows + valid_rows - 1] = 8192.0f;
        bytes = logit_bytes(logits, dtype);
        CUDA_CHECK(cudaMemcpyAsync(device_logits.data(), bytes.data(), device_logits.bytes(),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
        expected = argmax_oracle(logits, physical_rows, tokens, valid_rows);
    }

    const auto actual =
        from_device<std::int32_t>(device_output.data(), static_cast<std::size_t>(tokens));
    const auto logits_after = from_device<std::uint8_t>(device_logits.data(), bytes.size());
    const std::string label = std::string("argmax ") + (dtype == DType::FP32 ? "FP32" : "BF16") +
                              " rows=" + std::to_string(physical_rows) +
                              " valid=" + std::to_string(valid_rows) +
                              " T=" + std::to_string(tokens);

    int failures = 0;
    failures += verify_exact(label.c_str(), actual, expected);
    failures += verify_exact((label + " preserves logits").c_str(), logits_after, bytes);
    failures += device_logits.verify_guards((label + " logits").c_str());
    failures += device_output.verify_guards((label + " output").c_str());

    failures += probability_case(logits, physical_rows, valid_rows, tokens, label, dtype);
    return failures;
}

} // namespace

INFERNIX_OP_TEST(argmax) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case(248320, 248077, 1);
    failures += run_case(248320, 248077, 2);
    failures += run_case(248320, 248077, 8);
    failures += run_case(248320, 248077, 9);
    failures += run_case(248320, 248077, 64);
    failures += run_case(248320, 248077, 15);
    failures += run_case(248320, 248077, 128);
    failures += run_case(131072, 131072, 1);
    failures += run_case(131072, 131072, 15);
    failures += run_case(131072, 131072, 120);
    // Physical row counts that are not a multiple of 8 take the scalar route; valid rows that end
    // inside an 8-row vector group take its scalar remainder.
    failures += run_case(248319, 248077, 8);
    failures += run_case(131071, 131071, 15);
    failures += run_case(4096, 4093, 3);
    // FP32 logits (every vocabulary head's): the same routes four rows to a vector, and the scalar
    // route where the physical rows are not a multiple of four or the valid rows end inside a vector.
    for (const std::int32_t tokens : {1, 2, 8, 9, 15, 64, 128}) {
        failures += run_case(248320, 248077, tokens, DType::FP32);
    }
    failures += run_case(131072, 131072, 120, DType::FP32);
    failures += run_case(248318, 248077, 8, DType::FP32);
    failures += run_case(4096, 4093, 3, DType::FP32);
    // The MTP draft head's shapes (full vocabulary, the 131,072-row proposal head) at its batch sizes,
    // with confidences across the extent policy's buckets; and a domain of one block (column route).
    for (const auto& [physical, valid] : {std::pair{248320, 248077}, std::pair{131072, 131072}, std::pair{512, 509}}) {
        for (const std::int32_t tokens : {1, 2, 5, 8}) {
            for (const DType dtype : {DType::BF16, DType::FP32}) {
                failures += probability_case(make_draft_logits(physical, valid, tokens), physical, valid, tokens,
                                             std::string("draft head ") + (dtype == DType::FP32 ? "FP32" : "BF16") +
                                                 " rows=" + std::to_string(physical) + " T=" + std::to_string(tokens),
                                             dtype);
            }
        }
    }
    std::cout << (failures ? "FAIL" : "OK") << " argmax\n";
    return failures ? 1 : 0;
}
