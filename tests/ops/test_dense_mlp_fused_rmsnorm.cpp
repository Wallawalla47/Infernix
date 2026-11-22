// ops::dense_mlp_fused_rmsnorm_nvfp4 (SI-02+ and the 27B's prompt MLP): bit for bit rmsnorm(), linear_swiglu() and
// linear_add() under the call's policy (each qualified against its own FP64 oracle by its Op test), at every width of
// its domain, with the planes its fused producers write compared byte for byte with the quantizer's.
//
// LinearPolicy::AllowA4Residual (verification rounds, 5-96 columns): two row-major planes, every gate/up and down
// tile, full and ragged, and two widths captured in a CUDA Graph. LinearPolicy::AllowA4 (prompt calls of 129 columns
// or more): one plane, x's scales in 256-token tiles (the gate/up projection's TMA tile) and the product's in the
// layout the down projection reads at that width (row-major MMA tiles to 512 columns, 256-token tiles above); the
// tiled planes are compared over their padded tokens, whose scales both writers zero (the fused planes are filled
// with 0xA5 first). Three widths also repeat under a side stream's traffic (verify_repeatable).
#include "ops/op_family.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "infernix/ops/linear_add.h"
#include "infernix/ops/linear_swiglu.h"
#include "infernix/ops/lora.h"
#include "infernix/ops/rmsnorm.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_operands.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace infernix;
using namespace infernix::test;
using namespace infernix::test::input_projection;

namespace {

namespace detail = ops::detail;

constexpr int kHidden       = 5120;
constexpr int kGateUpRows   = 34816;
constexpr int kIntermediate = 17408;
constexpr float kEps        = 1.0e-6F;
constexpr auto kRounds      = ops::LinearPolicy::AllowA4Residual;
constexpr auto kPrompt      = ops::LinearPolicy::AllowA4;
constexpr int kTerms        = detail::kNvfp4A4ResidualTerms;

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

// The planes' written bytes: each term's codes over `tokens` columns and scales over `scale_tokens` (the padded
// count for a tiled plane).
int compare_planes(const std::string& label, const detail::Nvfp4A4Workspace& reference,
                   const detail::Nvfp4A4Workspace& fused, int tokens, int scale_tokens, int k, int terms) {
    int failures = 0;
    for (int term = 0; term < terms; ++term) {
        const std::string suffix = " term " + std::to_string(term);
        failures += compare_bytes(label + " codes" + suffix, reference.codes + term * reference.code_plane,
                                  fused.codes + term * fused.code_plane, static_cast<std::size_t>(tokens) * k / 2);
        failures += compare_bytes(label + " scales" + suffix, reference.scales + term * reference.scale_plane,
                                  fused.scales + term * fused.scale_plane,
                                  static_cast<std::size_t>(scale_tokens) * k / 16);
    }
    return failures;
}

void poison_scales(const detail::Nvfp4A4Workspace& planes) {
    cuda_check(cudaMemset(planes.scales, 0xA5, planes.scale_bytes), "poison scales");
}

int run_case(const Weight& gate_up, const Weight& down, int tokens, ops::LinearPolicy policy) {
    const bool prompt         = policy == kPrompt;
    const int terms           = prompt ? 1 : kTerms;
    const std::string suffix  = std::string(prompt ? " prompt" : " rounds") + " T=" + std::to_string(tokens);
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
        std::max({ops::linear_swiglu_workspace_capacity_bytes(QType::NVFP4, kGateUpRows, kHidden, policy, tokens,
                                                              tokens),
                  ops::linear_add_workspace_capacity_bytes(QType::NVFP4, kHidden, kIntermediate, policy, tokens,
                                                           tokens),
                  ops::dense_mlp_fused_rmsnorm_nvfp4_workspace_bytes(tokens)});
    DeviceArena reference_workspace(capacity * 2 + 4096);
    DeviceArena fused_workspace(ops::dense_mlp_fused_rmsnorm_nvfp4_workspace_bytes(tokens));
    int failures = 0;

    // The unfused composition the fused Op reproduces.
    ops::rmsnorm(reference, gain, kEps, true, hidden, nullptr);
    ops::linear_swiglu(hidden, gate_up, product, policy, reference_workspace, nullptr);
    {
        // Each fused producer's planes against the quantizer's bytes (from the stored BF16 values).
        auto scope         = reference_workspace.scope();
        const auto x_ref   = detail::allocate_nvfp4_a4_workspace(reference_workspace, tokens, kHidden, terms);
        const auto x_fused = detail::allocate_nvfp4_a4_workspace(reference_workspace, tokens, kHidden, terms);
        const auto h_ref   = detail::allocate_nvfp4_a4_workspace(reference_workspace, tokens, kIntermediate, terms);
        const auto h_fused = detail::allocate_nvfp4_a4_workspace(reference_workspace, tokens, kIntermediate, terms);
        if (prompt) {
            const auto down_layout = detail::nvfp4_linear_add_a4_scale_layout(tokens);
            poison_scales(x_fused);
            poison_scales(h_fused);
            detail::launch_nvfp4_a4_quantize(hidden, gate_up, x_ref, detail::Nvfp4ScaleLayout::Tiled256, nullptr);
            detail::launch_nvfp4_a4_quantize(product, down, h_ref, down_layout, nullptr);
            detail::launch_nvfp4_rmsnorm_a4_quantize_tiled(reference, gain, kEps, gate_up.input_scale_divisor,
                                                           x_fused, nullptr);
            detail::launch_nvfp4_linear_swiglu_a4_tma_planes(
                detail::nvfp4_a4_operands(gate_up, x_ref, tokens, detail::Nvfp4ScaleLayout::Tiled256), h_fused,
                down.input_scale_divisor, down_layout, nullptr);
            cuda_synchronize();
            const int padded      = detail::nvfp4_a4_padded_tokens(tokens);
            const int down_scales = down_layout == detail::Nvfp4ScaleLayout::RowMajor ? tokens : padded;
            failures += compare_planes("norm plane" + suffix, x_ref, x_fused, tokens, padded, kHidden, 1);
            failures += compare_planes("SwiGLU product plane" + suffix, h_ref, h_fused, tokens, down_scales,
                                       kIntermediate, 1);
        } else {
            detail::launch_nvfp4_a4_quantize(hidden, gate_up, x_ref, detail::Nvfp4ScaleLayout::RowMajor, nullptr);
            detail::launch_nvfp4_a4_quantize(product, down, h_ref, detail::Nvfp4ScaleLayout::RowMajor, nullptr);
            detail::launch_nvfp4_rmsnorm_a4_residual_quantize(reference, gain, kEps, gate_up.input_scale_divisor,
                                                              x_fused, nullptr);
            detail::nvfp4_linear_swiglu_a4_residual_planes_launch(gate_up, x_ref, h_fused, down.input_scale_divisor,
                                                                  tokens, nullptr);
            cuda_synchronize();
            failures += compare_planes("norm planes" + suffix, x_ref, x_fused, tokens, tokens, kHidden, kTerms);
            failures += compare_planes("SwiGLU product planes" + suffix, h_ref, h_fused, tokens, tokens,
                                       kIntermediate, kTerms);
        }
    }
    ops::linear_add(product, down, reference, policy, reference_workspace, nullptr);
    ops::dense_mlp_fused_rmsnorm_nvfp4(fused, gain, kEps, gate_up, down, policy, fused_workspace, nullptr);
    cuda_synchronize();
    failures += compare_bytes("residual" + suffix, reference_io.p, fused_io.p, reference_io.bytes);
    failures += compare_bytes("norm weight preserved" + suffix, gain_buffer.p, to_device(gain_bits).p,
                              gain_buffer.bytes);

    // Repeated calls under a side stream's traffic give the same bits (its cp.async and TMA pipelines; op-development.md,
    // "Repeat and compare"): a round width, the down projection's first TMA width and a ragged wide prompt call. Each
    // run restores the in-place residual first.
    if (tokens == 33 || tokens == 513 || tokens == 4160) {
        DeviceBuffer pristine = to_device(residual_bits);
        failures += verify_repeatable(
            "repeat" + suffix, 4, nullptr,
            [&] {
                cuda_check(cudaMemcpyAsync(fused_io.p, pristine.p, fused_io.bytes, cudaMemcpyDeviceToDevice, nullptr),
                           "restore the residual");
                ops::dense_mlp_fused_rmsnorm_nvfp4(fused, gain, kEps, gate_up, down, policy, fused_workspace, nullptr);
            },
            {OutputRegion{fused_io.p, fused_io.bytes}});
        failures += compare_bytes("repeat residual" + suffix, reference_io.p, fused_io.p, reference_io.bytes);
    }

    // Captured in a CUDA Graph (the decode graphs' programmatic launches), from the same input again.
    if (!prompt && (tokens == 16 || tokens == 33)) {
        DeviceBuffer graph_io = to_device(residual_bits);
        Tensor graph_residual(graph_io.p, DType::BF16, {kHidden, tokens});
        DeviceContext device;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(device.stream, [&] {
            ops::dense_mlp_fused_rmsnorm_nvfp4(graph_residual, gain, kEps, gate_up, down, policy, fused_workspace,
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

// --lora (include/infernix/ops/lora.h): the adapter form against the composition rmsnorm(), lora_shrink(h),
// linear_swiglu(+term), lora_shrink(product), linear_add(+term), each Op qualified with its term by the lora Op test.
// Gate/up stacks two A's (at rank 32, gate rows read U rows 0-31, up rows 32-63); down has one. Both sites, and each
// alone (the other projection then runs unadapted). Rank 32 runs the narrow adapter instances, 128 the wide ones.
struct LoraFactors {
    DeviceBuffer a_gate_up, b_gate_up, a_down, b_down;
    ops::LoraSite gate_up, down;
};

std::vector<std::uint16_t> uniform_bf16(std::size_t count, std::uint32_t seed, float bound) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-bound, bound);
    std::vector<std::uint16_t> out(count);
    for (auto& v : out) { v = f32_to_bf16(dist(rng)); }
    return out;
}

LoraFactors make_lora_factors(bool zero_b = false, int rank = 32) {
    const int kRank = rank;
    LoraFactors f;
    f.a_gate_up = to_device(uniform_bf16(static_cast<std::size_t>(2 * kRank) * kHidden, 61U, 0.03F));
    f.b_gate_up = to_device(zero_b ? std::vector<std::uint16_t>(static_cast<std::size_t>(kGateUpRows) * kRank, 0) : uniform_bf16(static_cast<std::size_t>(kGateUpRows) * kRank, 62U, 0.05F));
    f.a_down    = to_device(uniform_bf16(static_cast<std::size_t>(kRank) * kIntermediate, 63U, 0.03F));
    f.b_down    = to_device(zero_b ? std::vector<std::uint16_t>(static_cast<std::size_t>(kHidden) * kRank, 0) : uniform_bf16(static_cast<std::size_t>(kHidden) * kRank, 64U, 0.05F));
    f.gate_up.a = {f.a_gate_up.p, 2 * kRank, kHidden};
    f.gate_up.addend.b        = f.b_gate_up.p;
    f.gate_up.addend.rank     = kRank;
    f.gate_up.addend.u_rows   = 2 * kRank;
    f.gate_up.addend.segments = 2;
    f.gate_up.addend.segment[0] = {kIntermediate, 0, 2.0F};
    f.gate_up.addend.segment[1] = {kGateUpRows, kRank, 0.5F};
    f.down.a = {f.a_down.p, kRank, kIntermediate};
    f.down.addend.b        = f.b_down.p;
    f.down.addend.rank     = kRank;
    f.down.addend.u_rows   = kRank;
    f.down.addend.segments = 1;
    f.down.addend.segment[0] = {kHidden, 0, 1.25F};
    return f;
}

int run_lora_case(const Weight& gate_up, const Weight& down, const LoraFactors& factors, int tokens,
                  ops::LinearPolicy policy, bool with_gate_up, bool with_down) {
    const ops::LoraSite* gate_up_site = with_gate_up ? &factors.gate_up : nullptr;
    const ops::LoraSite* down_site    = with_down ? &factors.down : nullptr;
    const int gate_up_rows            = with_gate_up ? factors.gate_up.a.rows : 0;
    const int down_rows               = with_down ? factors.down.a.rows : 0;
    const std::string suffix = std::string(" lora rank ") + std::to_string(factors.down.addend.rank) +
                               (with_gate_up ? " gate_up" : "") + (with_down ? " down" : "") +
                               (policy == kPrompt ? " prompt" : " rounds") + " T=" + std::to_string(tokens);
    const auto residual_bits  = bf16_bits(make_bf16_activation(kHidden, tokens, 5051U + tokens));
    const auto gain_bits      = bf16_bits(make_bf16_activation(kHidden, 1, 229U));
    DeviceBuffer reference_io = to_device(residual_bits);
    DeviceBuffer fused_io     = to_device(residual_bits);
    DeviceBuffer gain_buffer  = to_device(gain_bits);
    Tensor reference(reference_io.p, DType::BF16, {kHidden, tokens});
    Tensor fused(fused_io.p, DType::BF16, {kHidden, tokens});
    Tensor gain(gain_buffer.p, DType::BF16, {kHidden});

    const std::size_t capacity =
        std::max(ops::linear_swiglu_workspace_capacity_bytes(QType::NVFP4, kGateUpRows, kHidden, policy, tokens, tokens),
                 ops::linear_add_workspace_capacity_bytes(QType::NVFP4, kHidden, kIntermediate, policy, tokens, tokens));
    DeviceArena reference_workspace(capacity +
                                    static_cast<std::size_t>(kHidden + kIntermediate + gate_up_rows + down_rows) *
                                        tokens * 2 +
                                    (1U << 16));
    DeviceArena fused_workspace(ops::dense_mlp_fused_rmsnorm_nvfp4_workspace_bytes(tokens, gate_up_rows, down_rows));

    // The composition.
    {
        Tensor hidden  = reference_workspace.alloc(DType::BF16, {kHidden, tokens});
        Tensor product = reference_workspace.alloc(DType::BF16, {kIntermediate, tokens});
        ops::rmsnorm(reference, gain, kEps, true, hidden, nullptr);
        ops::LoraAddend gate_up_term = factors.gate_up.addend, down_term = factors.down.addend;
        if (with_gate_up) {
            Tensor u = reference_workspace.alloc(DType::BF16, {gate_up_rows, tokens});
            ops::lora_shrink(hidden, factors.gate_up.a, u, nullptr);
            gate_up_term.u = u.data;
        }
        {
            auto scope = reference_workspace.scope();
            ops::linear_swiglu(hidden, gate_up, product, policy, reference_workspace, nullptr,
                               ops::LinearRoutes::Exact, with_gate_up ? &gate_up_term : nullptr);
        }
        if (with_down) {
            Tensor u = reference_workspace.alloc(DType::BF16, {down_rows, tokens});
            ops::lora_shrink(product, factors.down.a, u, nullptr);
            down_term.u = u.data;
        }
        auto scope = reference_workspace.scope();
        ops::linear_add(product, down, reference, policy, reference_workspace, nullptr, ops::LinearRoutes::Exact,
                        with_down ? &down_term : nullptr);
    }
    ops::dense_mlp_fused_rmsnorm_nvfp4(fused, gain, kEps, gate_up, down, policy, fused_workspace, nullptr, gate_up_site,
                                       down_site);
    cuda_synchronize();
    int failures = compare_bytes("residual" + suffix, reference_io.p, fused_io.p, reference_io.bytes);
    // The term is observable: the unadapted call differs.
    {
        DeviceBuffer plain_io = to_device(residual_bits);
        Tensor plain(plain_io.p, DType::BF16, {kHidden, tokens});
        DeviceArena plain_workspace(ops::dense_mlp_fused_rmsnorm_nvfp4_workspace_bytes(tokens));
        ops::dense_mlp_fused_rmsnorm_nvfp4(plain, gain, kEps, gate_up, down, policy, plain_workspace, nullptr);
        cuda_synchronize();
        if (from_device<std::uint16_t>(plain_io.p, residual_bits.size()) ==
            from_device<std::uint16_t>(fused_io.p, residual_bits.size())) {
            std::cerr << "the adapter changed nothing" << suffix << '\n';
            ++failures;
        }
    }
    // Captured in a CUDA Graph (the decode graphs' programmatic launches).
    if (policy == kRounds && tokens == 16) {
        DeviceBuffer graph_io = to_device(residual_bits);
        Tensor graph_residual(graph_io.p, DType::BF16, {kHidden, tokens});
        DeviceContext device;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(device.stream, [&] {
            ops::dense_mlp_fused_rmsnorm_nvfp4(graph_residual, gain, kEps, gate_up, down, policy, fused_workspace,
                                               device.stream, gate_up_site, down_site);
        });
        graph.instantiate(definition);
        graph.launch(device.stream);
        cuda_synchronize(device.stream);
        failures += compare_bytes("graph residual" + suffix, reference_io.p, graph_io.p, reference_io.bytes);
    }
    if (fused_workspace.used() != 0) {
        std::cerr << "dense MLP workspace scope was not released" << suffix << '\n';
        ++failures;
    }
    return failures;
}

// A trained adapter's initial state (B = 0): the fused call's adapter form returns the no-adapter call's bits (its
// norm/quantize kernel that also stores h, its tiles' adapter instances).
int run_zero_b_case(const Weight& gate_up, const Weight& down, const LoraFactors& zero, int tokens,
                    ops::LinearPolicy policy, bool with_gate_up, bool with_down) {
    const std::string suffix = std::string(" zero-B identity") + (with_gate_up ? " gate_up" : "") +
                               (with_down ? " down" : "") + (policy == kPrompt ? " prompt" : " rounds") +
                               " T=" + std::to_string(tokens);
    const auto residual_bits = bf16_bits(make_bf16_activation(kHidden, tokens, 6067U + tokens));
    const auto gain_bits     = bf16_bits(make_bf16_activation(kHidden, 1, 231U));
    DeviceBuffer plain_io    = to_device(residual_bits);
    DeviceBuffer adapted_io  = to_device(residual_bits);
    DeviceBuffer gain_buffer = to_device(gain_bits);
    Tensor plain(plain_io.p, DType::BF16, {kHidden, tokens});
    Tensor adapted(adapted_io.p, DType::BF16, {kHidden, tokens});
    Tensor gain(gain_buffer.p, DType::BF16, {kHidden});
    DeviceArena plain_workspace(ops::dense_mlp_fused_rmsnorm_nvfp4_workspace_bytes(tokens));
    DeviceArena adapted_workspace(ops::dense_mlp_fused_rmsnorm_nvfp4_workspace_bytes(
        tokens, with_gate_up ? zero.gate_up.a.rows : 0, with_down ? zero.down.a.rows : 0));
    ops::dense_mlp_fused_rmsnorm_nvfp4(plain, gain, kEps, gate_up, down, policy, plain_workspace, nullptr);
    ops::dense_mlp_fused_rmsnorm_nvfp4(adapted, gain, kEps, gate_up, down, policy, adapted_workspace, nullptr,
                                       with_gate_up ? &zero.gate_up : nullptr, with_down ? &zero.down : nullptr);
    cuda_synchronize();
    return compare_bytes("residual" + suffix, plain_io.p, adapted_io.p, plain_io.bytes);
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
        const auto eligible  = [&](const Weight& g, ops::LinearPolicy gp, const Weight& d, ops::LinearPolicy dp, int t) {
            return ops::dense_mlp_fused_rmsnorm_nvfp4_eligible(g, gp, d, dp, t);
        };
        if (eligible(gate_up, kPrompt, down, kRounds, 16) ||
            eligible(gate_up, kRounds, down, ops::LinearPolicy::A16Only, 16) ||
            eligible(gate_up, kRounds, down, kRounds, 4) || eligible(gate_up, kRounds, down, kRounds, 97) ||
            eligible(down, kRounds, gate_up, kRounds, 16) || !eligible(gate_up, kRounds, down, kRounds, 5) ||
            !eligible(gate_up, kRounds, down, kRounds, 96) || eligible(gate_up, kPrompt, down, kPrompt, 96) ||
            eligible(gate_up, kPrompt, down, kPrompt, 128) || eligible(gate_up, kPrompt, down, kRounds, 512) ||
            eligible(gate_up, kRounds, down, kRounds, 512) || !eligible(gate_up, kPrompt, down, kPrompt, 129) ||
            !eligible(gate_up, kPrompt, down, kPrompt, 4096)) {
            std::cerr << "fused dense MLP eligibility mismatch\n";
            return 1;
        }
        int failures = 0;
        // Every gate/up tile (16, 32, 64x128 and 96x128 tokens) and down tile, full and ragged.
        for (const int tokens : {5, 7, 8, 9, 13, 16, 17, 32, 33, 48, 64, 65, 80, 96}) {
            failures += run_case(gate_up, down, tokens, kRounds);
        }
        // The prompt profile: one partly filled 256-token gate/up tile (129-255 columns), the down projection's
        // row-major MMA tiles (to 512 columns) and its TMA tile, full and ragged 256-token tiles, up to the idle-wide
        // prefill's 4,096-column calls (and one ragged past them).
        for (const int tokens : {129, 136, 192, 193, 255, 256, 257, 300, 512, 513, 777, 1024, 1088, 2048, 4096, 4160}) {
            failures += run_case(gate_up, down, tokens, kPrompt);
        }
        // --lora: both sites over both profiles' tiles; each site alone at a round and a prompt width.
        const LoraFactors factors = make_lora_factors();
        for (const int tokens : {5, 9, 16, 33, 64, 96}) {
            failures += run_lora_case(gate_up, down, factors, tokens, kRounds, true, true);
        }
        for (const int tokens : {129, 257, 513, 1088, 4160}) {
            failures += run_lora_case(gate_up, down, factors, tokens, kPrompt, true, true);
        }
        for (const auto [gu, dn] : {std::pair{true, false}, std::pair{false, true}}) {
            failures += run_lora_case(gate_up, down, factors, 33, kRounds, gu, dn);
            failures += run_lora_case(gate_up, down, factors, 513, kPrompt, gu, dn);
        }
        // The wide adapter instances (rank 128) on both profiles.
        const LoraFactors wide = make_lora_factors(false, 128);
        for (const int tokens : {5, 16, 96}) { failures += run_lora_case(gate_up, down, wide, tokens, kRounds, true, true); }
        for (const int tokens : {129, 513, 4160}) {
            failures += run_lora_case(gate_up, down, wide, tokens, kPrompt, true, true);
        }
        const LoraFactors zero = make_lora_factors(true);
        for (const auto [gu, dn] : {std::pair{true, true}, std::pair{true, false}, std::pair{false, true}}) {
            for (const int tokens : {5, 8, 9, 16, 17, 33, 64, 96}) {
                failures += run_zero_b_case(gate_up, down, zero, tokens, kRounds, gu, dn);
            }
            for (const int tokens : {129, 257, 513, 1088}) {
                failures += run_zero_b_case(gate_up, down, zero, tokens, kPrompt, gu, dn);
            }
        }
        failures += gate_up_parent.verify_preserved("gate/up weight");
        failures += down_parent.verify_preserved("down weight");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "fused dense MLP test: " << error.what() << '\n';
        return 1;
    }
}
