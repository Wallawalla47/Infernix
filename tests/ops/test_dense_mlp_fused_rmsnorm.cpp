// ops::dense_mlp_fused_rmsnorm_nvfp4 (SI-02+): bit for bit rmsnorm(), linear_swiglu() and linear_add() under
// LinearPolicy::AllowA4Residual (each qualified against its own FP64 oracle by its Op test), at every width of its
// domain, with the planes its fused producers write compared byte for byte with the quantizer's.
#include "ops/op_family.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "infernix/ops/linear_add.h"
#include "infernix/ops/linear_swiglu.h"
#include "infernix/ops/rmsnorm.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::test;
using namespace infernix::test::input_projection;

namespace {

constexpr int kHidden       = 5120;
constexpr int kGateUpRows   = 34816;
constexpr int kIntermediate = 17408;
constexpr float kEps        = 1.0e-6F;
constexpr auto kPolicy      = ops::LinearPolicy::AllowA4Residual;
constexpr int kTerms        = ops::detail::kNvfp4A4ResidualTerms;

int compare_bytes(const std::string& label, const void* reference, const void* fused, std::size_t bytes) {
    const auto expected = from_device<std::uint8_t>(reference, bytes);
    const auto actual   = from_device<std::uint8_t>(fused, bytes);
    if (actual == expected) { return 0; }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << label << ": byte mismatch at " << i << " of " << bytes << " (reference "
                      << unsigned(expected[i]) << ", fused " << unsigned(actual[i]) << ")\n";
            break;
        }
    }
    return 1;
}

// The planes' written bytes: each term's row-major codes and scales over `tokens` columns.
int compare_planes(const std::string& label, const ops::detail::Nvfp4A4Workspace& reference,
                   const ops::detail::Nvfp4A4Workspace& fused, int tokens, int k) {
    int failures = 0;
    for (int term = 0; term < kTerms; ++term) {
        const std::string suffix = " term " + std::to_string(term);
        failures += compare_bytes(label + " codes" + suffix, reference.codes + term * reference.code_plane,
                                  fused.codes + term * fused.code_plane, static_cast<std::size_t>(tokens) * k / 2);
        failures += compare_bytes(label + " scales" + suffix, reference.scales + term * reference.scale_plane,
                                  fused.scales + term * fused.scale_plane, static_cast<std::size_t>(tokens) * k / 16);
    }
    return failures;
}

int run_case(const Weight& gate_up, const Weight& down, int tokens) {
    const std::string suffix  = " T=" + std::to_string(tokens);
    const auto residual_bits  = bf16_bits(make_bf16_activation(kHidden, tokens, 4049U + tokens));
    const auto gain_bits      = bf16_bits(make_bf16_activation(kHidden, 1, 227U));
    DeviceBuffer reference_io = to_device(residual_bits);
    DeviceBuffer fused_io     = to_device(residual_bits);
    DeviceBuffer gain_buffer  = to_device(gain_bits);
    DeviceBuffer hidden_buffer(static_cast<std::size_t>(kHidden) * tokens * 2);
    DeviceBuffer product_buffer(static_cast<std::size_t>(kIntermediate) * tokens * 2);
    Tensor reference(reference_io.p, DType::BF16, {kHidden, tokens});
    Tensor fused(fused_io.p, DType::BF16, {kHidden, tokens});
    Tensor gain(gain_buffer.p, DType::BF16, {kHidden});
    Tensor hidden(hidden_buffer.p, DType::BF16, {kHidden, tokens});
    Tensor product(product_buffer.p, DType::BF16, {kIntermediate, tokens});

    const std::size_t capacity =
        std::max({ops::linear_swiglu_workspace_capacity_bytes(QType::NVFP4, kGateUpRows, kHidden, kPolicy, tokens,
                                                              tokens),
                  ops::linear_add_workspace_capacity_bytes(QType::NVFP4, kHidden, kIntermediate, kPolicy, tokens,
                                                           tokens),
                  ops::dense_mlp_fused_rmsnorm_nvfp4_workspace_bytes(tokens)});
    DeviceArena reference_workspace(capacity * 2 + 4096);
    DeviceArena fused_workspace(ops::dense_mlp_fused_rmsnorm_nvfp4_workspace_bytes(tokens));
    int failures = 0;

    // The unfused composition the fused Op reproduces.
    ops::rmsnorm(reference, gain, kEps, true, hidden, nullptr);
    ops::linear_swiglu(hidden, gate_up, product, kPolicy, reference_workspace, nullptr);
    {
        // Each fused producer's planes against the quantizer's bytes (from the stored BF16 values).
        auto scope            = reference_workspace.scope();
        const auto x_ref      = ops::detail::allocate_nvfp4_a4_workspace(reference_workspace, tokens, kHidden, kTerms);
        const auto x_fused    = ops::detail::allocate_nvfp4_a4_workspace(reference_workspace, tokens, kHidden, kTerms);
        const auto h_ref      = ops::detail::allocate_nvfp4_a4_workspace(reference_workspace, tokens, kIntermediate, kTerms);
        const auto h_fused    = ops::detail::allocate_nvfp4_a4_workspace(reference_workspace, tokens, kIntermediate, kTerms);
        ops::detail::launch_nvfp4_a4_quantize(hidden, gate_up, x_ref, ops::detail::Nvfp4ScaleLayout::RowMajor, nullptr);
        ops::detail::launch_nvfp4_a4_quantize(product, down, h_ref, ops::detail::Nvfp4ScaleLayout::RowMajor, nullptr);
        ops::detail::launch_nvfp4_rmsnorm_a4_residual_quantize(reference, gain, kEps, gate_up.input_scale_divisor,
                                                               x_fused, nullptr);
        ops::detail::nvfp4_linear_swiglu_a4_residual_planes_launch(gate_up, x_ref, h_fused, down.input_scale_divisor,
                                                                   tokens, nullptr);
        cuda_synchronize();
        failures += compare_planes("norm planes" + suffix, x_ref, x_fused, tokens, kHidden);
        failures += compare_planes("SwiGLU product planes" + suffix, h_ref, h_fused, tokens, kIntermediate);
    }
    ops::linear_add(product, down, reference, kPolicy, reference_workspace, nullptr);
    ops::dense_mlp_fused_rmsnorm_nvfp4(fused, gain, kEps, gate_up, down, fused_workspace, nullptr);
    cuda_synchronize();
    failures += compare_bytes("residual" + suffix, reference_io.p, fused_io.p, reference_io.bytes);
    failures += compare_bytes("norm weight preserved" + suffix, gain_buffer.p, to_device(gain_bits).p,
                              gain_buffer.bytes);

    // Captured in a CUDA Graph (the decode graphs' programmatic launches), from the same input again.
    if (tokens == 16 || tokens == 33) {
        DeviceBuffer graph_io = to_device(residual_bits);
        Tensor graph_residual(graph_io.p, DType::BF16, {kHidden, tokens});
        DeviceContext device;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(device.stream, [&] {
            ops::dense_mlp_fused_rmsnorm_nvfp4(graph_residual, gain, kEps, gate_up, down, fused_workspace,
                                               device.stream);
        });
        graph.instantiate(definition);
        graph.launch(device.stream);
        cuda_synchronize(device.stream);
        failures += compare_bytes("graph residual" + suffix, reference_io.p, graph_io.p, reference_io.bytes);
    }
    if (reference_workspace.used() != 0 || fused_workspace.used() != 0) {
        std::cerr << "dense MLP workspace scope was not released" << suffix << '\n';
        ++failures;
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(dense_mlp_fused_rmsnorm) {
    try {
        if (cuda_unavailable()) { return 77; }
        quantized_weight::PatternedWeightOptions gate_up_options;
        gate_up_options.weight_scale_divisor = 0.125F;
        gate_up_options.input_scale_divisor  = 3.5F;
        quantized_weight::PatternedWeightOptions down_options;
        down_options.weight_scale_divisor = 0.25F;
        down_options.input_scale_divisor  = 2.75F;
        DevicePackedWeight gate_up_parent(
            quantized_weight::make_patterned_weight(QType::NVFP4, kGateUpRows, kHidden, 5101U, gate_up_options));
        DevicePackedWeight down_parent(
            quantized_weight::make_patterned_weight(QType::NVFP4, kHidden, kIntermediate, 5102U, down_options));
        const Weight gate_up = gate_up_parent.view();
        const Weight down    = down_parent.view();
        if (ops::dense_mlp_fused_rmsnorm_nvfp4_eligible(gate_up, ops::LinearPolicy::AllowA4, down, kPolicy, 16) ||
            ops::dense_mlp_fused_rmsnorm_nvfp4_eligible(gate_up, kPolicy, down, ops::LinearPolicy::A16Only, 16) ||
            ops::dense_mlp_fused_rmsnorm_nvfp4_eligible(gate_up, kPolicy, down, kPolicy, 4) ||
            ops::dense_mlp_fused_rmsnorm_nvfp4_eligible(gate_up, kPolicy, down, kPolicy, 97) ||
            ops::dense_mlp_fused_rmsnorm_nvfp4_eligible(down, kPolicy, gate_up, kPolicy, 16) ||
            !ops::dense_mlp_fused_rmsnorm_nvfp4_eligible(gate_up, kPolicy, down, kPolicy, 5) ||
            !ops::dense_mlp_fused_rmsnorm_nvfp4_eligible(gate_up, kPolicy, down, kPolicy, 96)) {
            std::cerr << "fused dense MLP eligibility mismatch\n";
            return 1;
        }
        int failures = 0;
        // Every gate/up tile (16, 32, 64x128 and 96x128 tokens) and down tile, full and ragged.
        for (const int tokens : {5, 7, 8, 9, 13, 16, 17, 32, 33, 48, 64, 65, 80, 96}) {
            failures += run_case(gate_up, down, tokens);
        }
        failures += gate_up_parent.verify_preserved("gate/up weight");
        failures += down_parent.verify_preserved("down weight");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "fused dense MLP test: " << error.what() << '\n';
        return 1;
    }
}
