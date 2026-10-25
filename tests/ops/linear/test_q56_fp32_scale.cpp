// Q5/Q6 prefill MMA precision against the FP64 oracle (TEMPORARY with INFERNIX_TMP_Q56_FP32_SCALE).
//
// The wide q5/q6_a16_mma kernels round q x scale to one BF16 weight before the MMA; with the switch
// on they feed the integer codes (exact in BF16) and apply each 64-group scale to the group's FP32
// partial sum, so the only rounding left that the oracle can see is the final BF16 store. Each arm
// prints, per shape and width, the relative L2 error and the fraction of outputs within half a BF16
// ulp (+2 %) of the oracle; ctest runs the binary once per arm. With the switch on, every width the
// MMA kernels serve must keep at least 99.5 % of the outputs correctly rounded to that margin; the
// default arm only reports (its BF16 weight rounding is an error of the same size as the store's),
// so the two logs show how far the error falls. Widths below the MMA floor take the sliced-K kernels, which already
// scale in FP32 and do not read the switch.
#include "ops/op_family.h"
#include "core/weight.h"
#include "infernix/ops/linear.h"
#include "core/arena.h"
#include "core/device.h"

#include "core/temporary_switches.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test;

struct Shape {
    const char* label;
    linear::WeightGenerator generator;
    std::int32_t n, k;
    std::int32_t first_mma; // the narrowest width the dispatcher sends to q5/q6_a16_mma
    std::uint32_t seed;
};

constexpr std::int32_t kWidths[]{64, 128, 256, 512, 1024, 2048, 4096};
constexpr std::int32_t kMaxWidth = 4096;
constexpr int kRows              = 64;
constexpr int kColumns           = 96;

std::vector<std::int32_t> spread(std::int32_t extent, int samples) {
    std::vector<std::int32_t> result;
    for (int sample = 0; sample < samples; ++sample) {
        const auto index = static_cast<std::int32_t>(static_cast<std::int64_t>(extent - 1) * sample / (samples - 1));
        if (result.empty() || result.back() != index) { result.push_back(index); }
    }
    return result;
}

// One BF16 ulp at |value| (the spacing of the binade holding it; the minimum normal's below it).
double bf16_ulp(double value) {
    const double magnitude = std::max(std::abs(value), std::ldexp(1.0, -126));
    return std::ldexp(1.0, std::ilogb(magnitude) - 7);
}

int run(const Shape& shape, bool fp32_scale) {
    quantized_weight::PackedWeight host_weight = shape.generator(shape.n, shape.k, shape.seed);
    const std::vector<std::int32_t> rows = spread(shape.n, kRows);
    const std::vector<float> oracle_weight = quantized_weight::materialize_rows_fp32(host_weight, rows);

    std::vector<std::uint16_t> activation(static_cast<std::size_t>(shape.k) * kMaxWidth);
    for (std::int32_t token = 0; token < kMaxWidth; ++token) {
        for (std::int32_t column = 0; column < shape.k; ++column) {
            std::uint32_t value = shape.seed ^ (static_cast<std::uint32_t>(column) * 0x9e3779b9U) ^
                                  (static_cast<std::uint32_t>(token) * 0x85ebca6bU);
            value ^= value >> 16;
            value *= 0x7feb352dU;
            value ^= value >> 15;
            activation[static_cast<std::size_t>(token) * shape.k + column] =
                f32_to_bf16(static_cast<float>(static_cast<int>(value & 0xffffU) - 32768) * (1.0F / 65536.0F));
        }
    }
    DeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation.data(), device_activation.bytes);
    DeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), device_weight.bytes);
    const Weight weight = host_weight.device_weight(device_weight.p);

    int failures = 0;
    for (const std::int32_t width : kWidths) {
        DeviceBuffer device_output(static_cast<std::size_t>(shape.n) * width * sizeof(std::uint16_t));
        Tensor input(device_activation.p, DType::BF16, {shape.k, width});
        Tensor output(device_output.p, DType::BF16, {shape.n, width});
        const std::size_t capacity = ops::linear_workspace_capacity_bytes(
            weight.qtype, shape.n, shape.k, ops::LinearPolicy::A16Only, width, width);
        DeviceArena workspace(std::max<std::size_t>(capacity, 256));
        ops::linear(input, weight, output, ops::LinearPolicy::A16Only, workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "synchronize linear");
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(shape.n) * width);
        device_output.copy_to_host(bits.data(), device_output.bytes);

        const std::vector<std::int32_t> columns = spread(width, kColumns);
        std::vector<float> oracle_activation(static_cast<std::size_t>(shape.k) * columns.size());
        for (std::size_t c = 0; c < columns.size(); ++c) {
            for (std::int32_t column = 0; column < shape.k; ++column) {
                oracle_activation[c * shape.k + column] =
                    bf16_to_f32(activation[static_cast<std::size_t>(columns[c]) * shape.k + column]);
            }
        }
        std::vector<double> reference(rows.size() * columns.size());
        linear::cpu_linear_gemm_fp64(oracle_weight.data(), oracle_activation.data(), reference.data(),
                                     static_cast<std::int32_t>(rows.size()), shape.k,
                                     static_cast<std::int32_t>(columns.size()));
        double error2 = 0.0, reference2 = 0.0;
        std::size_t rounded = 0;
        for (std::size_t c = 0; c < columns.size(); ++c) {
            for (std::size_t r = 0; r < rows.size(); ++r) {
                const double want = reference[c * rows.size() + r];
                const double got  = bf16_to_f32(bits[static_cast<std::size_t>(columns[c]) * shape.n + rows[r]]);
                error2 += (got - want) * (got - want);
                reference2 += want * want;
                rounded += std::abs(got - want) <= 0.51 * bf16_ulp(want) ? 1U : 0U;
            }
        }
        const double rel_l2   = std::sqrt(error2 / std::max(reference2, 1e-300));
        const double fraction = static_cast<double>(rounded) / static_cast<double>(rows.size() * columns.size());
        const bool mma        = width >= shape.first_mma;
        std::cout << shape.label << " [" << shape.n << ',' << shape.k << "] T=" << width
                  << (mma ? " mma" : " sliced") << " fp32_scale=" << fp32_scale << " rel_l2=" << std::scientific
                  << std::setprecision(3) << rel_l2 << std::defaultfloat << " correctly_rounded=" << std::fixed
                  << std::setprecision(4) << fraction << std::defaultfloat << '\n';
        // The A16 conformance bound of every Q5/Q6 route (linear_test_common.cpp: one BF16 unit roundoff).
        if (!(rel_l2 <= 1.0 / 256.0)) {
            std::cerr << shape.label << " T=" << width << ": rel-L2 " << rel_l2 << " exceeds 2^-8\n";
            ++failures;
        }
        if (fp32_scale && mma && fraction < 0.995) {
            std::cerr << shape.label << " T=" << width << ": only " << fraction
                      << " of the outputs are correctly rounded with the FP32 group scale\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(linear_q56_fp32_scale) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const bool fp32_scale = temporary_switches::q56_fp32_scale();
    // Real model shapes: the 27B groupwise Q5 MLP down (MMA from T = 161) and the Q6 fused gate_up
    // (MMA from T = 33; n34816_k5120.cpp).
    const Shape shapes[]{
        {"Q5_G64", linear::make_q5_g64_fp16_weight, 5120, 17408, 161, 211U},
        {"Q6_G64", linear::make_q6_g64_fp16_weight, 34816, 5120, 33, 223U},
    };
    int failures = 0;
    for (const Shape& shape : shapes) { failures += run(shape, fp32_scale); }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " q5/q6 fp32 group scale (fp32_scale=" << fp32_scale << ")\n";
    return failures == 0 ? 0 : 1;
}
