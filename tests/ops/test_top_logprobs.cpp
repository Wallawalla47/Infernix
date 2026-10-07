#include "infernix/ops/top_logprobs.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

constexpr ReductionCriterion kTopLogprobsFp32Criterion{
    /*relative_l2=*/2.0e-5,
    /*gross_absolute=*/2.0e-4,
    /*gross_relative_to_max_reference=*/0.0,
};

// Padding rows get a value above every valid one, so selecting one fails the exact ids check.
constexpr float kPaddingValue = 120.0f;

std::vector<std::uint16_t> make_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                       std::int32_t columns, float (*value)(std::int32_t row,
                                                                           std::int32_t column)) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = 0; row < physical_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] =
                f32_to_bf16(row < valid_rows ? value(row, column) : kPaddingValue);
        }
    }
    return logits;
}

// About 6144 distinct values on a 1/128 grid, rounded to BF16: frequent ties in long columns.
float random_value(std::int32_t row, std::int32_t column) {
    const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u +
                                static_cast<std::uint32_t>(column + 1) * 1013904223u;
    return -24.0f + static_cast<float>(mixed % 6144u) * (1.0f / 128.0f);
}

float uniform_value(std::int32_t, std::int32_t) { return 3.5f; }

float extreme_value(std::int32_t row, std::int32_t column) {
    constexpr float values[] = {-80.0f, -32.0f, -1.0f, 0.0f, 1.0f, 32.0f, 80.0f};
    return values[static_cast<std::size_t>(row + column) % std::size(values)];
}

// Zeros of both signs are equal values and rank by row.
float signed_zero_value(std::int32_t row, std::int32_t column) {
    switch ((row + column) % 3) {
    case 0: return -0.0f;
    case 1: return 0.0f;
    default: return -1.0f;
    }
}

struct Expected {
    std::vector<std::int32_t> ids;
    std::vector<double> logprobs;
};

Expected oracle(const std::vector<std::uint16_t>& logits, std::int32_t physical_rows,
                std::int32_t valid_rows, std::int32_t columns, std::int32_t count) {
    Expected expected;
    std::vector<std::int32_t> rows(static_cast<std::size_t>(valid_rows));
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        const auto value       = [&](std::int32_t row) {
            return static_cast<double>(bf16_to_f32(logits[base + static_cast<std::size_t>(row)]));
        };
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::int32_t row = 0; row < valid_rows; ++row) { maximum = std::max(maximum, value(row)); }
        double sum = 0.0;
        for (std::int32_t row = 0; row < valid_rows; ++row) { sum += std::exp(value(row) - maximum); }
        const double lse = maximum + std::log(sum);
        std::iota(rows.begin(), rows.end(), 0);
        std::stable_sort(rows.begin(), rows.end(),
                         [&](std::int32_t lhs, std::int32_t rhs) { return value(lhs) > value(rhs); });
        for (std::int32_t rank = 0; rank < count; ++rank) {
            expected.ids.push_back(rows[static_cast<std::size_t>(rank)]);
            expected.logprobs.push_back(value(rows[static_cast<std::size_t>(rank)]) - lse);
        }
    }
    return expected;
}

int run_case(const std::string& label, std::int32_t physical_rows, std::int32_t valid_rows,
             std::int32_t columns, std::int32_t count, const std::vector<std::uint16_t>& logits) {
    const Expected expected = oracle(logits, physical_rows, valid_rows, columns, count);
    const std::size_t outputs = static_cast<std::size_t>(count) * columns;

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_ids(outputs * sizeof(std::int32_t));
    GuardedDeviceBuffer device_output(outputs * sizeof(float));
    device_logits.copy_from_host(logits.data(), device_logits.bytes());
    device_ids.fill(0xcd);
    device_output.fill(0xcd);

    Tensor logits_tensor(device_logits.data(), DType::BF16, {physical_rows, columns});
    Tensor ids_tensor(device_ids.data(), DType::I32, {count, columns});
    Tensor output_tensor(device_output.data(), DType::FP32, {count, columns});
    ops::top_logprobs(logits_tensor, valid_rows, ids_tensor, output_tensor, nullptr);
    cuda_synchronize();

    const auto output = from_device<float>(device_output.data(), outputs);
    int failures      = verify_exact((label + " ids").c_str(),
                                     from_device<std::int32_t>(device_ids.data(), outputs), expected.ids);
    failures += verify_reduction(label, std::vector<double>(output.begin(), output.end()),
                                 expected.logprobs, kTopLogprobsFp32Criterion);
    failures +=
        verify_exact((label + " preserves logits").c_str(),
                     from_device<std::uint16_t>(device_logits.data(), logits.size()), logits);
    failures += device_logits.verify_guards(label + " logits guards");
    failures += device_ids.verify_guards(label + " ids guards");
    failures += device_output.verify_guards(label + " output guards");
    return failures;
}

template <class Function>
int expect_invalid(const char* label, Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& error) {
        std::cerr << label << ": expected invalid_argument, got " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

int run_validation_cases() {
    DeviceBuffer logits_data(8 * 3 * sizeof(std::uint16_t));
    DeviceBuffer ids_data(4 * 3 * sizeof(std::int32_t));
    DeviceBuffer output_data(4 * 3 * sizeof(float));
    Tensor logits(logits_data.p, DType::BF16, {8, 3});
    Tensor ids(ids_data.p, DType::I32, {4, 3});
    Tensor output(output_data.p, DType::FP32, {4, 3});

    int failures = 0;
    failures += expect_invalid("top_logprobs rejects valid_rows=0",
                               [&] { ops::top_logprobs(logits, 0, ids, output, nullptr); });
    failures += expect_invalid("top_logprobs rejects valid_rows>physical_rows",
                               [&] { ops::top_logprobs(logits, 9, ids, output, nullptr); });
    failures += expect_invalid("top_logprobs rejects K>valid_rows",
                               [&] { ops::top_logprobs(logits, 3, ids, output, nullptr); });
    failures += expect_invalid("top_logprobs rejects a column mismatch", [&] {
        Tensor wrong_ids(ids_data.p, DType::I32, {6, 2});
        ops::top_logprobs(logits, 8, wrong_ids, output, nullptr);
    });
    failures += expect_invalid("top_logprobs rejects an output shape mismatch", [&] {
        Tensor wrong_output(output_data.p, DType::FP32, {3, 3});
        ops::top_logprobs(logits, 8, ids, wrong_output, nullptr);
    });
    failures += expect_invalid("top_logprobs rejects output dtype", [&] {
        Tensor wrong_output(output_data.p, DType::BF16, {4, 3});
        ops::top_logprobs(logits, 8, ids, wrong_output, nullptr);
    });
    failures += expect_invalid("top_logprobs rejects non-contiguous logits", [&] {
        Tensor strided_logits = logits;
        strided_logits.nb[1] += 2;
        ops::top_logprobs(strided_logits, 8, ids, output, nullptr);
    });
    failures += expect_invalid("top_logprobs rejects aliased outputs", [&] {
        Tensor alias_output(ids_data.p, DType::FP32, {4, 3});
        ops::top_logprobs(logits, 8, ids, alias_output, nullptr);
    });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case("top_logprobs full vocabulary K=32", 248320, 248077, 3, 32,
                         make_logits(248320, 248077, 3, random_value));
    failures += run_case("top_logprobs non-aligned rows C=1025 K=5", 523, 509, 1025, 5,
                         make_logits(523, 509, 1025, random_value));
    failures += run_case("top_logprobs uniform logits rank by row", 263, 257, 64, 17,
                         make_logits(263, 257, 64, uniform_value));
    failures += run_case("top_logprobs one valid row", 13, 1, 7, 1,
                         make_logits(13, 1, 7, uniform_value));
    failures += run_case("top_logprobs K=valid_rows", 41, 37, 9, 37,
                         make_logits(41, 37, 9, random_value));
    failures += run_case("top_logprobs extreme finite ties", 263, 257, 17, 20,
                         make_logits(263, 257, 17, extreme_value));
    failures += run_case("top_logprobs signed zeros rank by row", 70, 64, 5, 30,
                         make_logits(70, 64, 5, signed_zero_value));
    failures += run_validation_cases();

    std::cout << (failures ? "FAIL" : "OK") << " top_logprobs\n";
    return failures ? 1 : 0;
}
