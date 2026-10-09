#include "core/weight.h"
#include "ops/linear/linear_test_common.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <array>
#include <exception>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test::linear;

std::vector<Invocation> a16_capacity_calls() {
    std::vector<Invocation> calls;
    for (int t = 1; t <= 33; ++t) calls.push_back({t});
    for (int t : {63, 64, 65, 95, 96, 97, 127, 128, 129, 512, 1024}) calls.push_back({t});
    for (int t : {3, 7, 11, 15, 19, 23, 25, 33})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    calls.push_back({1, CallForm::A16Convenience});
    for (int t : {64, 97, 129})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    return calls;
}

int run_fp8_a16() {
    const auto attn_invocations = a16_capacity_calls();
    int failures                = run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                                            {14336, 5120, 811U, Comparison::Sampled, true, attn_invocations});
    const auto gdn_invocations  = a16_capacity_calls();
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {16384, 5120, 817U, Comparison::Sampled, true, gdn_invocations});
    const auto mlp_invocations = a16_capacity_calls();
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {34816, 5120, 821U, Comparison::Sampled, true, mlp_invocations});
    std::vector<Invocation> vocabulary_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{9, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{24, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{25, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{41, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{42, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{49, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{96, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{97, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{160, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{161, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{288, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{289, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    // Every width of the sliced-K route.
    for (int t = 1; t <= 64; ++t)
        vocabulary_invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only});
    for (int t : {7, 25, 41, 48, 56, 64, 65, 128})
        vocabulary_invocations.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {248320, 5120, 823U, Comparison::Sampled, true, vocabulary_invocations});
    constexpr std::array vocabulary_policies{
        ops::LinearPolicy::A16Only,
        ops::LinearPolicy::AllowA8,
        ops::LinearPolicy::AllowA4,
    };
    for (const ops::LinearPolicy policy : vocabulary_policies) {
        try {
            const std::size_t capacity = ops::linear_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16, 248320, 5120, policy, 1, 2048);
            if (capacity != 0) {
                std::cerr << "FP8 vocabulary A16 route reported nonzero workspace\n";
                ++failures;
            }
        } catch (const std::exception& error) {
            std::cerr << "FP8 vocabulary policy was rejected: " << error.what() << '\n';
            ++failures;
        }
    }
    const auto residual6144_invocations = a16_capacity_calls();
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {5120, 6144, 827U, Comparison::Sampled, true, residual6144_invocations});
    const auto residual17408_invocations = a16_capacity_calls();
    failures +=
        run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                  {5120, 17408, 829U, Comparison::Sampled, true, residual17408_invocations});

    auto packed = make_fp8_weight(14336, 5120, 831U);
    try {
        (void)ops::detail::validate_fp8_weight(packed.weight, "FP8 validator test");
    } catch (const std::exception& error) {
        std::cerr << "valid FP8 metadata was rejected: " << error.what() << '\n';
        ++failures;
    }
    const auto expect_invalid = [&](const char* label, Weight invalid) {
        try {
            (void)ops::detail::validate_fp8_weight(invalid, "FP8 validator test");
            std::cerr << "invalid FP8 " << label << " was accepted\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    Weight invalid = packed.weight;
    invalid.layout = QuantLayout::Contiguous;
    expect_invalid("layout", invalid);
    invalid             = packed.weight;
    invalid.scale_nb[1] = invalid.scale_nb[1] - 2;
    expect_invalid("scale stride", invalid);
    invalid               = packed.weight;
    invalid.payload_bytes = invalid.payload_bytes - 1;
    expect_invalid("payload bound", invalid);
    for (auto [n, k] : {std::pair{14336, 5120}, std::pair{16384, 5120}, std::pair{34816, 5120},
                        std::pair{248320, 5120}, std::pair{5120, 6144}, std::pair{5120, 17408},
                        std::pair{16384, 2560}, std::pair{13312, 2560}, std::pair{2560, 6144},
                        std::pair{1280, 2560}, std::pair{2560, 640}}) {
        failures += verify_workspace_envelopes(QType::FP8_E4M3FN_ROW_BF16, n, k);
    }
    return failures;
}

// The Qwen3.8-Flash-Next shapes of recipe C (W8A16): GDN query/key/value/z, QSA query/gate/key/value,
// the GDN and QSA outputs, the shared expert's gate/up and down, against the FP64 oracle at every
// decode and verification width and the prefill tile boundaries.
int run_flash_next_fp8_a16() {
    int failures   = 0;
    std::uint32_t seed = 901U;
    for (const auto& [n, k] : {std::pair{16384, 2560}, std::pair{13312, 2560}, std::pair{2560, 6144},
                               std::pair{1280, 2560}, std::pair{2560, 640}}) {
        std::vector<Invocation> calls;
        for (int t = 1; t <= 33; ++t) calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only});
        for (int t : {48, 63, 64, 65, 96, 97, 127, 128, 129, 192, 257, 384, 512, 769, 1024, 1537, 2049, 4097})
            calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only});
        for (int t : {3, 9, 17, 33, 97})
            calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
        calls.push_back({1, CallForm::A16Convenience});
        failures += run_shape("FP8_A16 Flash-Next", ActivationCompute::A16, make_fp8_weight,
                              {n, k, seed++, Comparison::Sampled, true, calls});
    }
    return failures;
}

// Speculative verification relies on this for one row (design §11.3): an output column is
// bit-identical whether it is computed alone or beside other columns, up to 8 columns on every
// Flash-Next shape (its GEMV and SIMT routes reduce a column identically), and to 48 on the shared
// expert's down, whose TMA tiles all share one reduction order.
int fp8_a16_column_invariance() {
    int failures = 0;
    for (const auto& [n, k, columns] : {std::tuple{16384, 2560, 8}, std::tuple{13312, 2560, 8},
                                        std::tuple{2560, 6144, 8}, std::tuple{1280, 2560, 8},
                                        std::tuple{2560, 640, 48}}) {
        // Checkpoint-like operands: every finite E4M3 code and activations over 25 binades, so the
        // FP32 sums round and any difference in reduction order changes bits. (The patterned
        // weight with uniform activations sums exactly in any order and cannot show one.)
        auto host           = make_fp8_weight(n, k, 951U + static_cast<std::uint32_t>(n));
        std::uint32_t state = static_cast<std::uint32_t>(k) ^ static_cast<std::uint32_t>(n);
        const auto next     = [&state] { return state = state * 1664525U + 1013904223U; };
        for (std::uint64_t i = 0; i < host.code_plane_bytes; ++i) {
            auto code = static_cast<std::uint8_t>(next() >> 24);
            if ((code & 0x7FU) == 0x7FU) code ^= 1U; // not NaN
            host.payload[i] = code;
        }
        std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * columns);
        for (auto& bits : activation) {
            const std::uint32_t r = next();
            bits = static_cast<std::uint16_t>(((r >> 31) << 15) | ((107U + (r >> 8) % 25U) << 7) | (r & 0x7FU));
        }
        const std::size_t out_bytes = static_cast<std::size_t>(n) * columns * 2;
        void *weight = nullptr, *x = nullptr, *y = nullptr;
        infernix::test::cuda_check(cudaMalloc(&weight, host.payload.size()), "weight");
        infernix::test::cuda_check(cudaMalloc(&x, activation.size() * 2), "activation");
        infernix::test::cuda_check(cudaMalloc(&y, out_bytes), "output");
        infernix::test::cuda_check(cudaMemcpy(weight, host.payload.data(), host.payload.size(), cudaMemcpyHostToDevice),
                                   "upload weight");
        infernix::test::cuda_check(cudaMemcpy(x, activation.data(), activation.size() * 2, cudaMemcpyHostToDevice),
                                   "upload activation");
        const Weight w = host.device_weight(weight);
        std::vector<std::uint16_t> alone(static_cast<std::size_t>(n) * columns);
        for (int column = 0; column < columns; ++column) {
            Tensor input(static_cast<std::uint16_t*>(x) + static_cast<std::size_t>(column) * k, DType::BF16, {k, 1});
            Tensor output(y, DType::BF16, {n, 1});
            ops::linear(input, w, output, nullptr);
            infernix::test::cuda_check(cudaMemcpy(alone.data() + static_cast<std::size_t>(column) * n, y,
                                                  static_cast<std::size_t>(n) * 2, cudaMemcpyDeviceToHost),
                                       "read single column");
        }
        for (int t = 2; t <= columns; ++t) {
            Tensor input(x, DType::BF16, {k, t});
            Tensor output(y, DType::BF16, {n, t});
            ops::linear(input, w, output, nullptr);
            std::vector<std::uint16_t> together(static_cast<std::size_t>(n) * t);
            infernix::test::cuda_check(cudaMemcpy(together.data(), y, together.size() * 2, cudaMemcpyDeviceToHost),
                                       "read columns");
            std::size_t differing = 0;
            for (std::size_t i = 0; i < together.size(); ++i) differing += together[i] != alone[i];
            if (differing != 0) {
                std::cerr << "FP8_A16 [" << n << "," << k << "] T=" << t << ": " << differing
                          << " outputs differ from the columns computed alone\n";
                ++failures;
            }
        }
        cudaFree(weight);
        cudaFree(x);
        cudaFree(y);
    }
    return failures;
}

} // namespace

int main() {
    if (!infernix::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_fp8_a16() + run_flash_next_fp8_a16() + fp8_a16_column_invariance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " FP8 A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP8 A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
