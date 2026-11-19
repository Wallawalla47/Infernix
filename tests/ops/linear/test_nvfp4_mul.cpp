// ModelOpt's NVFP4 multiplier form (an nvfp4_mul block_scale_k16_m128x4_v1 parent run as QType::NVFP4: NVIDIA's
// NVFP4-QAD 27B, bit-exact), whose matrix coefficient is weight_scale_2 itself rather than a divisor.
//
//   - The kernels' output scale bit for bit: weight_scale_2 under 16-bit activations, and fl32(s * weight_scale_2)
//     (s the input's stored input_scale, vLLM's alpha) under 4-bit ones, whose activations encode with fl32(1 / s);
//     the divisor form keeps 1 / divisor and 1 / (input divisor * divisor).
//   - ops::linear at the 27B's NVFP4 shapes on the A16 routes against the FP64 product of the exactly decoded
//     weights (code * block scale * multiplier), and on the A16 and A4 routes bit for bit against the divisor form
//     where both forms' coefficients are exactly equal (powers of two), at widths selecting each route.
//   - The 27B output head [248320,5120] in this form: under A16ColumnInvariant a column's FP32 logits are the same
//     bits at every width, and the A16 route's tiles above 16 columns meet the FP64 criterion
//     (test_linear_fp32_head covers the narrow widths).
#include "ops/linear/linear_test_common.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include "core/arena.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test;
using namespace infernix::test::linear;

int run_alpha() {
    int failures = 0;
    const auto bits = [](float v) {
        std::uint32_t b = 0;
        std::memcpy(&b, &v, sizeof(b));
        return b;
    };
    Weight mul{};
    mul.weight_multiplier   = kNvfp4MulMultiplier;
    mul.input_scale         = kNvfp4MulInputScale;
    mul.input_scale_divisor = 1.0F / kNvfp4MulInputScale;
    const float vllm_alpha  = kNvfp4MulInputScale * kNvfp4MulMultiplier; // one FP32 rounding
    if (!ops::detail::valid_nvfp4_coefficient(mul) || bits(ops::detail::nvfp4_a16_alpha(mul)) != bits(kNvfp4MulMultiplier) ||
        bits(ops::detail::nvfp4_a4_alpha(mul)) != bits(vllm_alpha)) {
        std::cerr << "FAIL: the multiplier form's output scales are not weight_scale_2 and fl32(s * weight_scale_2)\n";
        ++failures;
    }
    Weight divisor{};
    divisor.weight_scale_divisor = 0.13F;
    divisor.input_scale_divisor  = 3.5F;
    if (!ops::detail::valid_nvfp4_coefficient(divisor) || ops::detail::nvfp4_a16_alpha(divisor) != 1.0F / 0.13F ||
        ops::detail::nvfp4_a4_alpha(divisor) != 1.0F / (3.5F * 0.13F)) {
        std::cerr << "FAIL: the divisor form's output scales changed\n";
        ++failures;
    }
    Weight both = mul;
    both.weight_scale_divisor = 0.13F;
    Weight neither{};
    if (ops::detail::valid_nvfp4_coefficient(both) || ops::detail::valid_nvfp4_coefficient(neither)) {
        std::cerr << "FAIL: a weight with both or neither coefficient was admitted\n";
        ++failures;
    }
    return failures;
}

// The A16 routes at the realistic coefficients against the FP64 product. The A4 routes run the same kernels as the
// divisor form's (test_nvfp4_a4 qualifies them against the FP64 product under the A4 allowance); what the
// multiplier form changes there, the alpha and the input divisor, run_alpha checks bit for bit and
// run_form_equivalence on every route.
int run_shapes() {
    std::vector<Invocation> a16;
    for (int t : {1, 2, 3, 8, 16, 17, 24, 32, 33, 48, 64, 65, 96, 128, 129, 300}) {
        a16.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only});
    }
    int failures = 0;
    std::uint32_t seed = 931U;
    for (const auto& [n, k] : {std::pair{14336, 5120}, std::pair{16384, 5120}, std::pair{34816, 5120},
                               std::pair{5120, 6144}, std::pair{5120, 17408}}) {
        failures += run_shape("NVFP4_MUL_A16", ActivationCompute::A16, make_nvfp4_mul_weight,
                              {n, k, seed, Comparison::Sampled, true, a16});
        seed += 2;
    }
    // The head's A16 route where it leaves the invariant tiles: the four-K-warp tiles (17-32 columns) and the
    // tensor-core GEMMs (33-64, 65-96 and wider).
    std::vector<Invocation> head;
    for (int t : {17, 24, 25, 32, 33, 48, 64, 65, 96, 97, 128, 129, 300}) {
        head.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only});
    }
    failures += run_shape("NVFP4_MUL_HEAD_A16", ActivationCompute::A16, make_nvfp4_mul_weight,
                          {248320, 5120, seed++, Comparison::Sampled, true, head});
    return failures;
}

// The multiplier form's coefficients reach every route of the 27B's NVFP4 shapes. With power-of-two coefficients
// both forms give the same alphas exactly (multiplier 8 and s 0.25 against divisor 0.125 and input divisor 4: 8 under
// A16, 2 under A4, input divisor 4) over the same codes and block scales, so ops::linear's output is the same bits
// for either form at every width, A16 and A4 routes alike, whatever arithmetic a route uses.
int run_form_equivalence() {
    int failures = 0;
    std::uint32_t seed = 1201U;
    for (const auto& [n, k] : {std::pair{14336, 5120}, std::pair{16384, 5120}, std::pair{34816, 5120},
                               std::pair{5120, 6144}, std::pair{5120, 17408}}) {
        quantized_weight::PatternedWeightOptions mul_options, div_options;
        mul_options.weight_multiplier    = 8.0F;
        mul_options.activation_scale     = 0.25F;
        div_options.weight_scale_divisor = 0.125F;
        div_options.input_scale_divisor  = 4.0F;
        const auto mul = quantized_weight::make_patterned_weight(QType::NVFP4, n, k, seed, mul_options);
        const auto div = quantized_weight::make_patterned_weight(QType::NVFP4, n, k, seed, div_options);
        ++seed;
        DeviceBuffer mul_device(mul.payload.size()), div_device(div.payload.size());
        mul_device.copy_from_host(mul.payload.data(), mul_device.bytes);
        div_device.copy_from_host(div.payload.data(), div_device.bytes);
        const Weight mul_weight = mul.device_weight(mul_device.p);
        const Weight div_weight = div.device_weight(div_device.p);

        constexpr std::int32_t kWidest = 1025;
        std::mt19937 rng(seed * 7U);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * kWidest);
        for (auto& v : x) { v = f32_to_bf16(normal(rng)); }
        DeviceBuffer input(x.size() * sizeof(std::uint16_t));
        input.copy_from_host(x.data(), input.bytes);
        DeviceBuffer output(static_cast<std::size_t>(n) * kWidest * sizeof(std::uint16_t));
        const std::size_t workspace_bytes =
            std::max(ops::linear_workspace_capacity_bytes(QType::NVFP4, n, k, ops::LinearPolicy::AllowA4, 1, kWidest),
                     ops::linear_workspace_capacity_bytes(QType::NVFP4, n, k, ops::LinearPolicy::A16Only, 1, kWidest));
        DeviceArena workspace(std::max<std::size_t>(workspace_bytes, 256));
        const auto run = [&](const Weight& weight, ops::LinearPolicy policy, int tokens) {
            const Tensor tx(input.p, DType::BF16, {k, tokens});
            Tensor out(output.p, DType::BF16, {n, tokens});
            workspace.reset();
            ops::linear(tx, weight, out, policy, workspace, nullptr);
            cuda_check(cudaDeviceSynchronize(), "form equivalence");
            std::vector<std::uint16_t> got(static_cast<std::size_t>(n) * tokens);
            cuda_check(cudaMemcpy(got.data(), output.p, got.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                       "read");
            return got;
        };
        const std::pair<ops::LinearPolicy, std::vector<int>> cases[]{
            {ops::LinearPolicy::A16Only, {1, 2, 3, 8, 16, 17, 33, 64, 65, 129, 300}},
            {ops::LinearPolicy::AllowA4, {1, 2, 4, 5, 7, 8, 16, 17, 64, 65, 128, 129, 193, 385, 513, 1024, 1025}},
        };
        for (const auto& [policy, widths] : cases) {
            for (const int tokens : widths) {
                const auto a = run(mul_weight, policy, tokens);
                const auto b = run(div_weight, policy, tokens);
                if (a != b) {
                    std::size_t differing = 0;
                    for (std::size_t i = 0; i < a.size(); ++i) { differing += a[i] != b[i]; }
                    std::cerr << "FAIL: NVFP4 multiplier/divisor forms [" << n << "," << k << "] "
                              << (policy == ops::LinearPolicy::AllowA4 ? "AllowA4" : "A16Only") << " T=" << tokens
                              << ": " << differing << " outputs differ\n";
                    ++failures;
                }
            }
        }
    }
    return failures;
}

// Under A16ColumnInvariant every column of a call of each width equals, bit for bit, the same column alone.
int run_head_invariance() {
    constexpr std::int32_t kRows = 248320, kK = 5120, kWidest = 65;
    const auto packed = make_nvfp4_mul_weight(kRows, kK, 977U);
    DeviceBuffer device_weight(packed.payload.size());
    device_weight.copy_from_host(packed.payload.data(), device_weight.bytes);
    const Weight weight = packed.device_weight(device_weight.p);
    std::mt19937 rng(979U);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(kK) * kWidest);
    for (auto& v : x) { v = f32_to_bf16(normal(rng)); }
    DeviceBuffer input(x.size() * sizeof(std::uint16_t));
    input.copy_from_host(x.data(), input.bytes);
    DeviceBuffer output(static_cast<std::size_t>(kRows) * kWidest * sizeof(float));
    DeviceArena workspace(256);
    const auto run = [&](int column, int tokens) {
        const Tensor tx(static_cast<std::uint16_t*>(input.p) + static_cast<std::size_t>(column) * kK, DType::BF16,
                        {kK, tokens});
        Tensor out(output.p, DType::FP32, {kRows, tokens});
        ops::linear(tx, weight, out, ops::LinearPolicy::A16ColumnInvariant, workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "head invariant");
        std::vector<float> got(static_cast<std::size_t>(kRows) * tokens);
        cuda_check(cudaMemcpy(got.data(), output.p, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "read");
        return got;
    };
    std::vector<std::vector<float>> alone;
    for (int column = 0; column < kWidest; ++column) { alone.push_back(run(column, 1)); }
    int failures = 0;
    for (const int tokens : {2, 8, 9, 16, 17, 33, 64, 65}) {
        const auto together = run(0, tokens);
        std::size_t differing = 0;
        for (int column = 0; column < tokens; ++column) {
            differing += std::memcmp(together.data() + static_cast<std::size_t>(column) * kRows,
                                     alone[static_cast<std::size_t>(column)].data(), kRows * sizeof(float)) != 0;
        }
        if (differing != 0) {
            std::cerr << "FAIL: NVFP4 head T=" << tokens << ": " << differing << " columns differ from the columns alone\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(linear_nvfp4_mul) {
    int failures = run_alpha();
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return failures == 0 ? 77 : 1;
    }
    try {
        failures += run_shapes();
        failures += run_form_equivalence();
        failures += run_head_invariance();
    } catch (const std::exception& error) {
        std::cerr << "NVFP4_MUL Linear: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4 multiplier form\n";
    return failures == 0 ? 0 : 1;
}
