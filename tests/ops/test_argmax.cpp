#include "infernix/ops/argmax.h"
#include "core/device.h"
#include <algorithm>
#include "ops/op_tester.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
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

    // The selected rows' softmax probability (argmax_probability) against its FP64 formula: fast
    // exponentials and an FP32 sum of at most 248,077 terms in [0, 1] stay within 1e-4 relative.
    GuardedDeviceBuffer device_probability(static_cast<std::size_t>(tokens) * sizeof(float));
    Tensor probability_tensor(device_probability.data(), DType::FP32, {tokens});
    ops::argmax_probability(logits_tensor, output_tensor, probability_tensor, valid_rows, nullptr);
    cuda_synchronize();
    const auto probability = from_device<float>(device_probability.data(), static_cast<std::size_t>(tokens));
    std::size_t far = 0;
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::size_t base = static_cast<std::size_t>(t) * physical_rows;
        const double reference = logits[base + static_cast<std::size_t>(actual[static_cast<std::size_t>(t)])];
        double sum             = 0.0;
        for (std::int32_t v = 0; v < valid_rows; ++v) { sum += std::exp(logits[base + v] - reference); }
        const double want = 1.0 / sum;
        far += std::fabs(probability[static_cast<std::size_t>(t)] - want) > 1e-4 * want;
    }
    if (far != 0) {
        std::cerr << "FAIL: " << label << " probability: " << far << " columns outside 1e-4 of FP64\n";
        ++failures;
    }
    failures += device_probability.verify_guards((label + " probability").c_str());
    return failures;
}

} // namespace

int main() {
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
    std::cout << (failures ? "FAIL" : "OK") << " argmax\n";
    return failures ? 1 : 0;
}
