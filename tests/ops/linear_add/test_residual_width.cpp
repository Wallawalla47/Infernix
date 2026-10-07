// A column of a fused residual projection must not change its bits with the batch width
// (NInfer issue Neroued/ninfer#374).
//
// The NVFP4 A4 route launches a FullTokens kernel instance when the width fills whole token tiles
// and a predicated one otherwise. If the epilogue scales the accumulator outside the
// live-column predicate and adds the residual inside it, the compiler can contract the multiply
// and the add into one FMA in the full instance only, so the same column rounds differently in a
// full tile and in a tail tile: a token's prefill result would depend on how many tokens share its
// chunk, and a prefix-cache resume could disagree with the run it resumes.
//
// Every token column here carries the same input. The residual nearly cancels the projection (it
// is the negated output of a zero-residual call), so the result is the FP32 rounding residue of
// scale + add and one FMA versus a multiply then an add shows in the BF16 output. Every column of
// every width must equal the first A4 width's column bit for bit.
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

constexpr std::int32_t kRows      = 5120;
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

// Runs the projection over `tokens` identical columns with `residual` (one column, broadcast) and
// returns the output bits [tokens][kRows].
std::vector<std::uint16_t> run(const Weight& weight, QType qtype, ops::LinearPolicy policy, std::int32_t k,
                               const GuardedDeviceBuffer& input, const std::vector<std::uint16_t>& residual,
                               std::int32_t tokens) {
    std::vector<std::uint16_t> initial(static_cast<std::size_t>(kRows) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        std::copy(residual.begin(), residual.end(), initial.begin() + static_cast<std::size_t>(t) * kRows);
    }
    GuardedDeviceBuffer output(initial.size() * sizeof(std::uint16_t));
    output.copy_from_host(initial.data(), output.bytes());
    Tensor x(const_cast<void*>(input.data()), DType::BF16, {k, tokens});
    Tensor out(output.data(), DType::BF16, {kRows, tokens});
    const std::size_t capacity = ops::linear_add_workspace_capacity_bytes(qtype, kRows, k, policy, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    ops::linear_add(x, weight, out, policy, workspace, nullptr);
    cuda_check(cudaDeviceSynchronize(), "synchronize linear_add");
    std::vector<std::uint16_t> bits(initial.size());
    output.copy_to_host(bits.data(), output.bytes());
    return bits;
}

int run_problem(QType qtype, ops::LinearPolicy policy, std::int32_t k, std::int32_t first, std::uint32_t seed,
                const std::string& name) {
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.13F; // an inexact scale, so scale and add both round
    options.input_scale_divisor  = 3.5F;
    quantized_weight::PackedWeight host_weight =
        quantized_weight::make_patterned_weight(qtype, kRows, k, seed, options);
    GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());

    const std::vector<std::uint16_t> column = column_input(k, seed + 1U);
    std::vector<std::uint16_t> input(static_cast<std::size_t>(k) * kMaxTokens);
    for (std::int32_t t = 0; t < kMaxTokens; ++t) {
        std::copy(column.begin(), column.end(), input.begin() + static_cast<std::size_t>(t) * k);
    }
    GuardedDeviceBuffer device_input(input.size() * sizeof(std::uint16_t));
    device_input.copy_from_host(input.data(), device_input.bytes());

    // The residual cancels the projection on even rows (odd rows keep 0).
    const std::vector<std::uint16_t> zero(kRows, f32_to_bf16(0.0F));
    const auto projection = run(weight, qtype, policy, k, device_input, zero, first);
    std::vector<std::uint16_t> residual(kRows);
    for (std::int32_t row = 0; row < kRows; ++row) {
        residual[row] = row % 2 ? f32_to_bf16(0.0F) : f32_to_bf16(-bf16_to_f32(projection[row]));
    }
    const auto reference = run(weight, qtype, policy, k, device_input, residual, first);
    int failures = 0;
    std::vector<std::int32_t> differing;
    for (const std::int32_t width : widths(first)) {
        const auto bits = run(weight, qtype, policy, k, device_input, residual, width);
        bool same = true;
        for (std::int32_t t = 0; t < width && same; ++t) {
            same = std::equal(reference.begin(), reference.begin() + kRows,
                              bits.begin() + static_cast<std::size_t>(t) * kRows);
        }
        if (!same) { differing.push_back(width); }
    }
    if (!differing.empty()) {
        std::cerr << name << ": a column's bits change with the batch width at";
        for (const auto w : differing) { std::cerr << ' ' << w; }
        std::cerr << '\n';
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (infernix::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_problem(QType::NVFP4, ops::LinearPolicy::AllowA4, 17408, 8, 11U, "NVFP4 A4 [5120,17408]");
    failures += run_problem(QType::NVFP4, ops::LinearPolicy::AllowA4, 6144, 8, 13U, "NVFP4 A4 [5120,6144]");
    if (failures == 0) { std::cout << "residual width checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
