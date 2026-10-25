// A column of a fused residual projection must not change its bits with the batch width
// (NInfer issue Neroued/ninfer#374).
//
// The NVFP4 A4 and FP8 A8 routes launch a FullTokens kernel instance when the width fills whole
// token tiles and a predicated one otherwise. If the epilogue scales the accumulator outside the
// live-column predicate and adds the residual inside it, the compiler can contract the multiply
// and the add into one FMA in the full instance only (and in the FP8 TMA split-K reduce), so the
// same column rounds differently in a full tile and in a tail tile: a token's prefill result would
// depend on how many tokens share its chunk, and a prefix-cache resume could disagree with the run
// it resumes.
//
// Every token column here carries the same input. The residual nearly cancels the projection (it
// is the negated output of a zero-residual call), so the result is the FP32 rounding residue of
// scale + add and one FMA versus a multiply then an add shows in the BF16 output. Every column of
// every width must equal the first quantized-activation width's column bit for bit.
//
// FP8 A8 widths take routes with different K splits and summation orders, which would hide the
// epilogue behind accumulator rounding. Its problems therefore use exact rows: every activation
// encodes to an E4M3 code in [256, 448] (a multiple of 32) and every weight code lies in
// +-[1, 1.875] (a multiple of 1/8), so each product is a multiple of 4 below 841 and a K = 17408
// row sums to under 2^24 such quanta: the FP32 accumulator is exact in any order, and only the
// epilogue's rounding can differ between widths.
#include "ops/op_family.h"
#include "core/weight.h"
#include "infernix/ops/linear_add.h"
#include "core/device.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test;

constexpr std::int32_t kMaxTokens = 1025;

// Widths from the first quantized-activation width to past the TMA floor: each schedule band on a
// full and a ragged width, and the 1024/1025 chunk widths.
std::vector<std::int32_t> widths(std::int32_t first) {
    std::vector<std::int32_t> result;
    for (std::int32_t t = first; t <= 48; ++t) { result.push_back(t); }
    for (std::int32_t t : {49, 63, 64, 65, 96, 127, 128, 129, 191, 192, 193, 255, 256, 257, 383, 384,
                           385, 511, 512, 513, 767, 768, 769, 1023, 1024, 1025}) {
        result.push_back(t);
    }
    return result;
}

std::uint32_t mix(std::uint32_t seed, std::uint32_t a, std::uint32_t b = 0U) {
    std::uint32_t value = seed ^ (a * 0x9e3779b9U) ^ (b * 0x85ebca6bU);
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

std::vector<std::uint16_t> column_input(std::int32_t k, std::uint32_t seed) {
    std::vector<std::uint16_t> column(static_cast<std::size_t>(k));
    for (std::int32_t row = 0; row < k; ++row) {
        std::uint32_t value = seed ^ (static_cast<std::uint32_t>(row) * 0x9e3779b9U);
        value ^= value >> 16;
        value *= 0x7feb352dU;
        value ^= value >> 15;
        column[row] = f32_to_bf16(static_cast<float>(static_cast<int>(value & 0xffU) - 128) * (1.0F / 64.0F));
    }
    return column;
}

// Exact-row activations: +-{0.625, 0.75, 0.875, 1}, with |x| = 1 present, so the dynamic token
// scale is fl(1/448) (and the static scale is chosen equal) and every code is in [256, 448].
std::vector<std::uint16_t> exact_column_input(std::int32_t k, std::uint32_t seed) {
    constexpr float kMagnitudes[]{0.625F, 0.75F, 0.875F, 1.0F};
    std::vector<std::uint16_t> column(static_cast<std::size_t>(k));
    for (std::int32_t row = 0; row < k; ++row) {
        const std::uint32_t value = mix(seed, static_cast<std::uint32_t>(row));
        const float magnitude     = row == 0 ? 1.0F : kMagnitudes[value & 3U];
        column[row]               = f32_to_bf16((value & 4U) != 0U ? -magnitude : magnitude);
    }
    return column;
}

// Exact-row weight codes: E4M3 +-[1, 1.875] (0x38..0x3f, sign 0x80), hashed per element. The row
// scales stay the fixture's (BF16 1 and 1.5 times powers of two, or 24-bit FP32 multipliers).
void exact_weight_codes(quantized_weight::PackedWeight& packed, std::int32_t n, std::int32_t k,
                        std::uint32_t seed) {
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t column = 0; column < k; ++column) {
            const std::uint32_t value = mix(seed, static_cast<std::uint32_t>(row), static_cast<std::uint32_t>(column));
            packed.payload[static_cast<std::size_t>(row) * k + column] =
                static_cast<std::uint8_t>(0x38U | (value & 7U) | ((value & 8U) << 4));
        }
    }
}

// Runs the projection over `tokens` identical columns with `residual` (one column, broadcast) and
// returns the output bits [tokens][rows].
std::vector<std::uint16_t> run(const Weight& weight, QType qtype, ops::LinearPolicy policy, std::int32_t rows,
                               std::int32_t k, const GuardedDeviceBuffer& input,
                               const std::vector<std::uint16_t>& residual, std::int32_t tokens) {
    std::vector<std::uint16_t> initial(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        std::copy(residual.begin(), residual.end(), initial.begin() + static_cast<std::size_t>(t) * rows);
    }
    GuardedDeviceBuffer output(initial.size() * sizeof(std::uint16_t));
    output.copy_from_host(initial.data(), output.bytes());
    Tensor x(const_cast<void*>(input.data()), DType::BF16, {k, tokens});
    Tensor out(output.data(), DType::BF16, {rows, tokens});
    const std::size_t capacity = ops::linear_add_workspace_capacity_bytes(qtype, rows, k, policy, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    ops::linear_add(x, weight, out, policy, workspace, nullptr);
    cuda_check(cudaDeviceSynchronize(), "synchronize linear_add");
    std::vector<std::uint16_t> bits(initial.size());
    output.copy_to_host(bits.data(), output.bytes());
    return bits;
}

struct Problem {
    QType qtype;
    ops::LinearPolicy policy;
    std::int32_t rows, k, first;
    std::uint32_t seed;
    bool exact_rows;
    std::string name;
};

int run_problem(const Problem& problem) {
    const std::int32_t rows = problem.rows, k = problem.k;
    quantized_weight::PatternedWeightOptions options;
    if (problem.qtype == QType::FP8_E4M3FN_ROW_F32) {
        options.activation_scale = 1.0F / 448.0F; // the dynamic scale of a column whose maximum is 1
    } else if (problem.qtype != QType::FP8_E4M3FN_ROW_BF16) {
        options.weight_scale_divisor = 0.13F; // an inexact scale, so scale and add both round
        options.input_scale_divisor  = 3.5F;
    }
    quantized_weight::PackedWeight host_weight =
        quantized_weight::make_patterned_weight(problem.qtype, rows, k, problem.seed, options);
    if (problem.exact_rows) { exact_weight_codes(host_weight, rows, k, problem.seed + 7U); }
    GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());

    const std::vector<std::uint16_t> column =
        problem.exact_rows ? exact_column_input(k, problem.seed + 1U) : column_input(k, problem.seed + 1U);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(k) * kMaxTokens);
    for (std::int32_t t = 0; t < kMaxTokens; ++t) {
        std::copy(column.begin(), column.end(), input.begin() + static_cast<std::size_t>(t) * k);
    }
    GuardedDeviceBuffer device_input(input.size() * sizeof(std::uint16_t));
    device_input.copy_from_host(input.data(), device_input.bytes());

    // The residual cancels the projection on even rows (odd rows keep 0).
    const std::vector<std::uint16_t> zero(static_cast<std::size_t>(rows), f32_to_bf16(0.0F));
    const auto projection =
        run(weight, problem.qtype, problem.policy, rows, k, device_input, zero, problem.first);
    std::vector<std::uint16_t> residual(static_cast<std::size_t>(rows));
    for (std::int32_t row = 0; row < rows; ++row) {
        residual[row] = row % 2 ? f32_to_bf16(0.0F) : f32_to_bf16(-bf16_to_f32(projection[row]));
    }
    const auto reference =
        run(weight, problem.qtype, problem.policy, rows, k, device_input, residual, problem.first);
    int failures = 0;
    std::vector<std::int32_t> differing;
    for (const std::int32_t width : widths(problem.first)) {
        const auto bits = run(weight, problem.qtype, problem.policy, rows, k, device_input, residual, width);
        bool same = true;
        for (std::int32_t t = 0; t < width && same; ++t) {
            same = std::equal(reference.begin(), reference.begin() + rows,
                              bits.begin() + static_cast<std::size_t>(t) * rows);
        }
        if (!same) { differing.push_back(width); }
    }
    if (!differing.empty()) {
        std::cerr << problem.name << ": a column's bits change with the batch width at";
        for (const auto w : differing) { std::cerr << ' ' << w; }
        std::cerr << '\n';
        ++failures;
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(linear_add_residual_width) {
    if (infernix::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    using ops::LinearPolicy;
    // The FP8 first widths are the A8 route's floors (fp8_linear_add_plan.cpp): narrower calls take
    // the A16 route, whose activations are not quantized.
    const std::vector<Problem> problems{
        {QType::NVFP4, LinearPolicy::AllowA4, 5120, 17408, 8, 11U, false, "NVFP4 A4 [5120,17408]"},
        {QType::NVFP4, LinearPolicy::AllowA4, 5120, 6144, 8, 13U, false, "NVFP4 A4 [5120,6144]"},
        {QType::FP8_E4M3FN_ROW_BF16, LinearPolicy::AllowA8, 5120, 17408, 20, 17U, true,
         "FP8 A8 exact rows [5120,17408]"},
        {QType::FP8_E4M3FN_ROW_BF16, LinearPolicy::AllowA8, 5120, 6144, 17, 19U, true,
         "FP8 A8 exact rows [5120,6144]"},
        {QType::FP8_E4M3FN_ROW_F32, LinearPolicy::AllowA8, 3072, 8192, 17, 23U, true,
         "FP8 static A8 exact rows [3072,8192]"},
    };
    int failures = 0;
    for (const Problem& problem : problems) { failures += run_problem(problem); }
    if (failures == 0) { std::cout << "residual width checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
