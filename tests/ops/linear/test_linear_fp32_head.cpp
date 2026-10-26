// FP32 vocabulary-head logits: ops::linear with an FP32 `out`, every head format at its real shape and at
// widths selecting each of its routes, on the A16 and column-invariant policies. The FP32 output is the
// same computation as the BF16 output without its final rounding (include/infernix/ops/linear.h): each BF16
// output of the same call is the FP32 output rounded to nearest even, and the FP32 output meets the A16 routes'
// criterion against the FP64 product of the stored weights (decoded with their stored scales) and the BF16
// activations; its worst error in units of the FP32 accumulation bound sum_k |w x| * K * 2^-24 is reported.
#include "ops/linear/linear_test_common.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include "core/arena.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {
using namespace infernix;
using namespace infernix::test;
using namespace infernix::test::linear;

struct HeadCase {
    const char* label;
    WeightGenerator generator;
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
    bool column_invariant; // the format registers an A16ColumnInvariant route for the head
};

// BF16 [t][k] activations, standard normal before rounding.
std::vector<std::uint16_t> activations(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * t);
    for (auto& value : bits) { value = f32_to_bf16(normal(rng)); }
    return bits;
}

// Rows the FP64 oracle covers: spread over the head, the last one included.
std::vector<std::int32_t> oracle_rows(std::int32_t n) {
    std::vector<std::int32_t> rows;
    for (std::int32_t row = 0; row < n; row += n / 61) { rows.push_back(row); }
    rows.push_back(n - 1);
    return rows;
}

// Columns compared: the first, one inside and the last.
std::vector<std::int32_t> compared_columns(std::int32_t t) {
    std::vector<std::int32_t> columns{0, t / 2, t - 1};
    std::sort(columns.begin(), columns.end());
    columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
    return columns;
}

int run_case(const HeadCase& head) {
    const quantized_weight::PackedWeight packed = head.generator(head.n, head.k, head.seed);
    DeviceBuffer device_weight(packed.payload.size());
    device_weight.copy_from_host(packed.payload.data(), device_weight.bytes);
    const Weight weight = packed.device_weight(device_weight.p);
    const std::vector<std::int32_t> rows = oracle_rows(head.n);
    const std::vector<float> stored      = quantized_weight::materialize_rows_fp32(packed, rows);

    constexpr std::int32_t kMaximumT = 512;
    const std::vector<std::uint16_t> x_bits = activations(head.k, kMaximumT, head.seed + 1U);
    DeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(x_bits.data(), device_x.bytes);
    DeviceBuffer out32(static_cast<std::size_t>(head.n) * kMaximumT * sizeof(float));
    DeviceBuffer out16(static_cast<std::size_t>(head.n) * kMaximumT * sizeof(std::uint16_t));

    int failures = 0;
    std::vector<std::pair<std::int32_t, ops::LinearPolicy>> calls;
    for (const std::int32_t t : {1,  2,  3,  4,  8,  9,  16, 17, 24, 25, 32, 33, 34, 40,  41,  48,
                                 49, 56, 57, 64, 65, 72, 73, 80, 81, 96, 97, 112, 113, 128, 129, 256, 512}) {
        calls.emplace_back(t, ops::LinearPolicy::A16Only);
    }
    if (head.column_invariant) {
        for (const std::int32_t t : {1, 2, 8, 9, 16, 17, 33, 64}) {
            calls.emplace_back(t, ops::LinearPolicy::A16ColumnInvariant);
        }
    }
    double worst = 0.0;
    for (const auto& [t, policy] : calls) {
        const std::string label = std::string(head.label) + " T=" + std::to_string(t) +
                                  (policy == ops::LinearPolicy::A16ColumnInvariant ? " invariant" : "");
        const std::size_t capacity =
            ops::linear_workspace_capacity_bytes(weight.qtype, head.n, head.k, policy, t, t);
        DeviceArena workspace(std::max<std::size_t>(capacity, 256));
        const Tensor x(device_x.p, DType::BF16, {head.k, t});
        Tensor wide(out32.p, DType::FP32, {head.n, t});
        Tensor narrow(out16.p, DType::BF16, {head.n, t});
        cuda_check(cudaMemset(out32.p, 0xff, static_cast<std::size_t>(head.n) * t * sizeof(float)), "poison");
        ops::linear(x, weight, wide, policy, workspace, nullptr);
        ops::linear(x, weight, narrow, policy, workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "linear");

        for (const std::int32_t column : compared_columns(t)) {
            const std::size_t offset = static_cast<std::size_t>(column) * head.n;
            std::vector<float> got32(static_cast<std::size_t>(head.n));
            std::vector<std::uint16_t> got16(static_cast<std::size_t>(head.n));
            cuda_check(cudaMemcpy(got32.data(), static_cast<const float*>(out32.p) + offset,
                                  got32.size() * sizeof(float), cudaMemcpyDeviceToHost),
                       "read FP32");
            cuda_check(cudaMemcpy(got16.data(), static_cast<const std::uint16_t*>(out16.p) + offset,
                                  got16.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                       "read BF16");
            std::size_t rounding = 0;
            for (std::size_t row = 0; row < got32.size(); ++row) {
                rounding += f32_to_bf16(got32[row]) != got16[row] || !std::isfinite(got32[row]);
            }
            if (rounding != 0) {
                std::cerr << "FAIL: " << label << " column " << column << ": " << rounding
                          << " BF16 outputs are not the FP32 output rounded to nearest even\n";
                ++failures;
            }
            const std::uint16_t* xc = x_bits.data() + static_cast<std::size_t>(column) * head.k;
            std::vector<double> got(rows.size()), want(rows.size());
            for (std::size_t r = 0; r < rows.size(); ++r) {
                double reference = 0.0, magnitude = 0.0;
                for (std::int32_t i = 0; i < head.k; ++i) {
                    const double product =
                        static_cast<double>(stored[r * head.k + i]) * static_cast<double>(bf16_to_f32(xc[i]));
                    reference += product;
                    magnitude += std::fabs(product);
                }
                got[r]  = got32[static_cast<std::size_t>(rows[r])];
                want[r] = reference;
                worst   = std::max(worst, std::fabs(got[r] - reference) / (magnitude * head.k * std::ldexp(1.0, -24) + 1e-30));
            }
            // The A16 routes' criterion (BF16 output's): one BF16 unit roundoff of relative L2, two of gross error.
            failures += verify_reduction(label + " column " + std::to_string(column), got, want,
                                         {1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0});
        }
    }
    // The routes keep their own arithmetic (the MMA routes' group scaling rounds), so the FP32 accumulation bound
    // is reported, not required.
    std::cout << head.label << ": " << calls.size() << " calls, worst error " << worst
              << " of the FP32 accumulation bound sum|wx| K 2^-24\n";
    return failures;
}

} // namespace

INFERNIX_OP_TEST(linear_fp32_head) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    try {
        // The 27B's and the 35B-A3B's heads (Q6 and Q8 artifacts), the row-FP8 head, and the BF16 runtime-shape
        // route (every BF16 head's FP32 route; a tenth of the vocabulary keeps the fixture small).
        for (const HeadCase& head : {HeadCase{"Q6 [248320,5120]", make_q6_g64_fp16_weight, 248320, 5120, 601U, true},
                                     HeadCase{"Q6 [248320,2048]", make_q6_g64_fp16_weight, 248320, 2048, 607U, true},
                                     HeadCase{"Q8 [248320,5120]", make_q8_g32_fp16_weight, 248320, 5120, 613U, true},
                                     HeadCase{"Q8 [248320,2048]", make_q8_g32_fp16_weight, 248320, 2048, 617U, true},
                                     HeadCase{"FP8 [248320,5120]", make_fp8_weight, 248320, 5120, 619U, true},
                                     HeadCase{"FP8 row-F32 [248320,3072]", make_fp8_f32_weight, 248320, 3072, 623U, true},
                                     HeadCase{"BF16 [24832,5120]", make_bf16_weight, 24832, 5120, 631U, false}}) {
            failures += run_case(head);
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " FP32 vocabulary-head logits\n";
    return failures == 0 ? 0 : 1;
}
