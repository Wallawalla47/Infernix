#include "ops/op_family.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test::linear;

int run_fp8_a8() {
    constexpr std::array attn_decode_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    int failures =
        run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                  {14336, 5120, 827U, Comparison::Sampled, true, attn_decode_invocations});
    constexpr std::array attn_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1023, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {14336, 5120, 829U, Comparison::Sampled, true, attn_invocations});
    constexpr std::array attn_bulk_invocations{
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{288, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{289, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {14336, 5120, 833U, Comparison::Sampled, true, attn_bulk_invocations});
    constexpr std::array gdn_decode_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {16384, 5120, 837U, Comparison::Sampled, true, gdn_decode_invocations});
    constexpr std::array gdn_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {16384, 5120, 839U, Comparison::Sampled, true, gdn_invocations});
    constexpr std::array gdn_bulk_invocations{
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{384, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {16384, 5120, 841U, Comparison::Sampled, true, gdn_bulk_invocations});
    constexpr std::array mlp_decode_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{3, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                          {34816, 5120, 851U, Comparison::Sampled, true, mlp_decode_invocations});
    constexpr std::array mlp_invocations{
        Invocation{5, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {34816, 5120, 853U, Comparison::Sampled, true, mlp_invocations});

    constexpr std::array mlp_bulk_invocations{
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{511, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{2048, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {34816, 5120, 863U, Comparison::Sampled, true, mlp_bulk_invocations});

    constexpr std::array residual6144_decode_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape(
        "FP8_A16", ActivationCompute::A16, make_fp8_weight,
        {5120, 6144, 855U, Comparison::Sampled, true, residual6144_decode_invocations});
    constexpr std::array residual6144_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{192, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{193, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{768, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{769, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape("FP8_A8", ActivationCompute::A8, make_fp8_weight,
                          {5120, 6144, 857U, Comparison::Sampled, true, residual6144_invocations});
    constexpr std::array residual17408_decode_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape(
        "FP8_A16", ActivationCompute::A16, make_fp8_weight,
        {5120, 17408, 867U, Comparison::Sampled, true, residual17408_decode_invocations});
    constexpr std::array residual17408_invocations{
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{48, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{256, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{384, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{385, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{513, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
        Invocation{768, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{769, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{1025, CallForm::Policy, ops::LinearPolicy::AllowA8, true},
    };
    failures += run_shape(
        "FP8_A8", ActivationCompute::A8, make_fp8_weight,
        {5120, 17408, 859U, Comparison::Sampled, true, residual17408_invocations});

    struct Problem {
        std::int32_t rows;
        std::int32_t input_rows;
        bool a8_at_one;
        bool a8_at_two;
    };

    for (const Problem problem :
         {Problem{14336, 5120, false, false}, Problem{16384, 5120, false, false},
          Problem{34816, 5120, false, false}, Problem{5120, 6144, false, false},
          Problem{5120, 17408, false, false}}) {
        const std::size_t one = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 1, 1);
        const std::size_t two = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 2, 2);
        const std::size_t forty_eight = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 48, 48);
        const std::size_t early_interval = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 2, 4);
        const std::size_t hot_interval = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 1, 48);
        const std::size_t exact_1024 = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 1024, 1024);
        const std::size_t exact_1048 = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 1048, 1048);
        const std::size_t spanning = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows,
            ops::LinearPolicy::AllowA8, 1000, 1048);
        const std::size_t a16 = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows,
            ops::LinearPolicy::A16Only, 1, 2048);
        // A16Only still sizes the flagged routes (pair, pair block32, FP16 accumulate) its interval
        // reaches: the query does not see the route flags (fp8_linear_workspace_capacity_bytes).
        const std::size_t flagged_routes = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, problem.rows, problem.input_rows, ops::LinearPolicy::A16Only, 2048, 2048);
        if ((one != 0) != problem.a8_at_one || (two != 0) != problem.a8_at_two ||
            early_interval != 0 || forty_eight <= two || hot_interval != forty_eight ||
            exact_1024 <= forty_eight || exact_1048 <= exact_1024 || spanning != exact_1048 ||
            a16 != flagged_routes) {
            std::cerr << "FP8 A8 workspace interval contract mismatch for N=" << problem.rows
                      << " K=" << problem.input_rows << '\n';
            ++failures;
        }
    }
    return failures;
}

// fp8-a8-pair (LinearRoutes::Fp8A8Pair) on Qwen3.8-Flash-Next's FP8 problems (recipe C, A16Only
// permission): GDN query/key/value/z, QSA query/gate/key/value, GDN and QSA output, shared-expert
// gate/up and down.
//
// Named criterion: the route carries the activation as two E4M3 terms and meets the A16 criterion
// (linear_test_common.cpp: relative L2 <= 2^-8, |error| <= 2^-8 + 2^-7 max|reference|), so the
// suite checks it with ActivationCompute::A16. Derivation. A column x is represented as
// 2^s1*q1 + 2^s2*q2 with power-of-two scales: s1 the least exponent with max|x| * 2^-s1 <= 448,
// q1 = e4m3(x * 2^-s1), r = x - 2^s1*q1 (exact in FP32), s2 likewise for max|r|,
// q2 = e4m3(r * 2^-s2). A BF16 value has eight significant bits: q1 keeps the top four (rounded)
// and r the remaining four, which E4M3's four significant bits hold exactly while r stays in its
// normal or subnormal grid, that is for every value within 2^14 of the column maximum
// (pair_quantizer_exact checks this against the exact oracle). Smaller values contribute at most
// 2^-20 of the column maximum each. The MMAs multiply exact E4M3 products and sum them in FP32;
// the epilogue's single BF16 rounding of the FP32 sum of both terms (at most 2^-9 relative, RMS
// about 1.1e-3) is the A16 route's own error, so the route sits at the A16 route's error against
// the 3.9e-3 budget, not above it.
//
// fp8-a8-pair-block32 (LinearRoutes::Fp8A8PairBlock32) runs the same problems at the same widths
// with s1 and s2 chosen per 32-value block of K instead of per column, and the block-scaled MMA
// applying them; the derivation holds per block, so every value within 2^14 of its block's maximum
// is carried exactly (pair_block32_quantizer_exact) and the route meets the same criterion.
// Each problem's pair admission (its ops/linear/fp8/shapes kPairMinTokens: the width from which
// the pair route is faster than the exact route; the block32 route shares it).
struct PairProblem {
    std::int32_t n, k, min_tokens;
};
constexpr std::array<PairProblem, 5> kFlashNextProblems{{{16384, 2560, 33},
                                                         {13312, 2560, 97},
                                                         {2560, 6144, 513},
                                                         {1280, 2560, 1025},
                                                         {2560, 640, 513}}};
constexpr ops::LinearRoutes kPair        = ops::LinearRoutes::Fp8A8Pair;
constexpr ops::LinearRoutes kPairBlock32 = ops::LinearRoutes::Fp8A8PairBlock32;
// The width at which both routes move from 64- to 128-column tiles (fp8_a8_pair_tma_mma.cuh
// kFp8A8PairNarrowTokens).
constexpr std::int32_t kPairNarrowTokens = 256;

// The two pair routes, each with its label and seed offset.
struct PairRoute {
    const char* label;
    ops::LinearRoutes routes;
    std::uint32_t seed_offset;
};
constexpr std::array<PairRoute, 2> kPairRoutes{{{"FP8_A8Pair", kPair, 0U},
                                                {"FP8_A8PairBlock32", kPairBlock32, 1000U}}};

int run_flash_next_fp8_a8_pair(const PairRoute& route) {
    const ops::LinearRoutes routes = route.routes;
    const std::string label        = route.label;
    int failures                   = 0;
    std::uint32_t seed             = 941U + route.seed_offset;
    for (const auto& [n, k, min_tokens] : kFlashNextProblems) {
        std::vector<Invocation> calls;
        // Below the admission (the exact route), the admission, the tile seams and the bulk.
        for (const int t : {1, min_tokens - 1, min_tokens, min_tokens + 1, 63, 64, 65, 128,
                            kPairNarrowTokens - 1, kPairNarrowTokens, kPairNarrowTokens + 1, 384,
                            385, 1024, 2049})
            calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, false, routes});
        for (const int t : {min_tokens, std::max(min_tokens, kPairNarrowTokens) + 1})
            calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true, routes});
        // These problems register no single-term A8 route, so the wider permissions run the pair.
        const std::int32_t wide = std::max(min_tokens, 129);
        calls.push_back({wide, CallForm::Policy, ops::LinearPolicy::AllowA8, false, routes});
        calls.push_back({wide, CallForm::Policy, ops::LinearPolicy::AllowA4, false, routes});
        failures += run_shape(label + " Flash-Next", ActivationCompute::A16, make_fp8_weight,
                              {n, k, seed++, Comparison::Sampled, true, calls});
    }
    // A column the first term represents to FP32 rounding (alternating +-0.25: a near-zero second
    // term), and columns with only the last 16 channels set (the K tail of the 640-channel down),
    // each at its problem's admission and in the 128-column tiles.
    const auto pattern_calls = [routes](std::int32_t min_tokens) {
        return std::vector<Invocation>{
            {min_tokens, CallForm::Policy, ops::LinearPolicy::A16Only, false, routes},
            {std::max(min_tokens, kPairNarrowTokens) + 44, CallForm::Policy,
             ops::LinearPolicy::A16Only, false, routes},
        };
    };
    failures += run_shape(label + " Flash-Next cancellation", ActivationCompute::A16,
                          make_fp8_weight,
                          {16384, 2560, 947U + route.seed_offset, Comparison::Sampled, true,
                           pattern_calls(17), ActivationPattern::Cancellation});
    failures += run_shape(label + " Flash-Next K tail", ActivationCompute::A16, make_fp8_weight,
                          {2560, 640, 949U + route.seed_offset, Comparison::Sampled, true,
                           pattern_calls(513), ActivationPattern::KTail});
    return failures;
}

// fp8-a8-pair-block32 on NVIDIA FP4-Local's MXFP8 problems: the same two-term activation, the weight's
// UE8M0 group scales as the block-scaled MMA's weight scales. The represented weights (codes x their
// group's power of two) enter the MMA exactly and the products are exact, so the derivation above holds
// per 32-value block and the route meets the A16 criterion against the FP64 oracle of the represented
// weights. Admissions are mxfp8_shapes.h's kMxfp8PairMinTokens* (recipe C's per [N,K], untimed).
int run_flash_next_mxfp8_pair_block32() {
    constexpr std::array<PairProblem, 5> kMxfp8Problems{{{16384, 2560, 33},
                                                         {13952, 2560, 97},
                                                         {2560, 6144, 513},
                                                         {1280, 2560, 1025},
                                                         {2560, 640, 513}}};
    int failures       = 0;
    std::uint32_t seed = 2941U;
    for (const auto& [n, k, min_tokens] : kMxfp8Problems) {
        std::vector<Invocation> calls;
        for (const int t : {1, min_tokens - 1, min_tokens, min_tokens + 1, 64, 65, kPairNarrowTokens,
                            kPairNarrowTokens + 1, 385, 1024, 2049})
            calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, false, kPairBlock32});
        for (const int t : {min_tokens, std::max(min_tokens, kPairNarrowTokens) + 1})
            calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true, kPairBlock32});
        // The column-invariant policy keeps the exact route whatever the flag.
        calls.push_back({std::max(min_tokens, 129), CallForm::Policy, ops::LinearPolicy::A16ColumnInvariant, false,
                         kPairBlock32});
        failures += run_shape("MXFP8_A8PairBlock32 FP4-Local", ActivationCompute::A16, make_mxfp8_weight,
                              {n, k, seed++, Comparison::Sampled, true, calls});
    }
    const auto pattern_calls = [](std::int32_t min_tokens) {
        return std::vector<Invocation>{
            {min_tokens, CallForm::Policy, ops::LinearPolicy::A16Only, false, kPairBlock32},
            {std::max(min_tokens, kPairNarrowTokens) + 44, CallForm::Policy, ops::LinearPolicy::A16Only, false,
             kPairBlock32},
        };
    };
    failures += run_shape("MXFP8_A8PairBlock32 FP4-Local cancellation", ActivationCompute::A16, make_mxfp8_weight,
                          {16384, 2560, 2947U, Comparison::Sampled, true, pattern_calls(33),
                           ActivationPattern::Cancellation});
    failures += run_shape("MXFP8_A8PairBlock32 FP4-Local K tail", ActivationCompute::A16, make_mxfp8_weight,
                          {2560, 640, 2949U, Comparison::Sampled, true, pattern_calls(513),
                           ActivationPattern::KTail});
    return failures;
}

struct DeviceProblem {
    test::quantized_weight::PackedWeight host;
    DeviceBuffer weight;
    Weight view;

    DeviceProblem(std::int32_t n, std::int32_t k, std::uint32_t seed)
        : host(make_fp8_weight(n, k, seed)), weight(host.payload.size()) {
        weight.copy_from_host(host.payload.data(), host.payload.size());
        view = host.device_weight(weight.p);
    }
};

// One public call; returns the BF16 output bits.
std::vector<std::uint16_t> run_linear(const Weight& weight, const DeviceBuffer& x, std::int32_t t,
                                      ops::LinearPolicy policy, ops::LinearRoutes routes) {
    const std::size_t capacity = ops::linear_workspace_capacity_bytes(weight.qtype, weight.n,
                                                                      weight.k, policy, t, t);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    DeviceBuffer y(static_cast<std::size_t>(weight.n) * t * sizeof(std::uint16_t));
    Tensor input(x.p, DType::BF16, {weight.k, t});
    Tensor output(y.p, DType::BF16, {weight.n, t});
    ops::linear(input, weight, output, policy, workspace, nullptr, routes);
    test::cuda_check(cudaDeviceSynchronize(), "pair linear");
    if (workspace.peak_used() > capacity || workspace.used() != 0) {
        throw std::runtime_error("pair linear: workspace query/scope mismatch");
    }
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(weight.n) * t);
    y.copy_to_host(bits.data(), bits.size() * sizeof(std::uint16_t));
    return bits;
}

// Activations shaped like a transformer's: a few outlier channels (every 61st, 2^8 larger) set
// each column's maximum, column magnitudes vary over nine binades, and every 37th column is zero.
// Most values then sit far below their column's maximum, where the first E4M3 term alone is coarse
// or subnormal and the second term must carry them; a zero column has two zero scales.
std::vector<std::uint16_t> outlier_activations(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * t);
    std::uint32_t state = seed * 2654435761U + static_cast<std::uint32_t>(t);
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t channel = 0; channel < k; ++channel) {
            state            = state * 1664525U + 1013904223U;
            const float unit = static_cast<float>(static_cast<std::int32_t>(state >> 9) - (1 << 22)) /
                               static_cast<float>(1 << 22);
            float value = std::ldexp(unit, token % 9 - 4);
            if (channel % 61 == 7) value *= 256.0F;
            if (token % 37 == 5) value = 0.0F;
            bits[static_cast<std::size_t>(token) * k + channel] = test::f32_to_bf16(value);
        }
    }
    return bits;
}

int run_flash_next_pair_outliers(const PairRoute& route) {
    constexpr test::ReductionCriterion kA16Criterion{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
    int failures       = 0;
    std::uint32_t seed = 971U + route.seed_offset;
    for (const auto& [n, k, min_tokens] : kFlashNextProblems) {
        DeviceProblem problem(n, k, seed++);
        std::vector<std::int32_t> rows;
        for (std::int32_t i = 0; i < 64; ++i)
            rows.push_back(static_cast<std::int32_t>((static_cast<std::int64_t>(n) - 1) * i / 63));
        const std::vector<float> oracle_weight =
            test::quantized_weight::materialize_rows_fp32(problem.host, rows);
        // The problem's admission and a width in the 128-column tiles.
        for (const std::int32_t t : {min_tokens, std::max(min_tokens, kPairNarrowTokens) + 44}) {
            const std::vector<std::uint16_t> bits = outlier_activations(k, t, seed);
            std::vector<float> values(bits.size());
            for (std::size_t index = 0; index < bits.size(); ++index)
                values[index] = test::bf16_to_f32(bits[index]);
            DeviceBuffer x(bits.size() * sizeof(std::uint16_t));
            x.copy_from_host(bits.data(), bits.size() * sizeof(std::uint16_t));
            const auto output =
                run_linear(problem.view, x, t, ops::LinearPolicy::A16Only, route.routes);
            const auto row_count = static_cast<std::int32_t>(rows.size());
            std::vector<double> reference(static_cast<std::size_t>(row_count) * t);
            cpu_linear_gemm_fp64(oracle_weight.data(), values.data(), reference.data(), row_count,
                                 k, t);
            std::vector<double> actual(reference.size());
            for (std::int32_t token = 0; token < t; ++token)
                for (std::int32_t r = 0; r < row_count; ++r)
                    actual[static_cast<std::size_t>(token) * row_count + r] = test::bf16_to_f32(
                        output[static_cast<std::size_t>(token) * n + rows[static_cast<std::size_t>(r)]]);
            const std::string label = std::string(route.label) + " outliers [" +
                                      std::to_string(n) + "," + std::to_string(k) +
                                      "] T=" + std::to_string(t);
            failures += test::verify_reduction(label, actual, reference, kA16Criterion);
        }
    }
    return failures;
}

// The exact value of an E4M3 code.
double e4m3_value(std::uint8_t code) {
    const int exponent = (code >> 3) & 0xF;
    const int mantissa = code & 0x7;
    const double magnitude = exponent == 0 ? std::ldexp(mantissa, -9)
                                           : std::ldexp(8 + mantissa, exponent - 10);
    return (code & 0x80) != 0 ? -magnitude : magnitude;
}

// The pair quantizer against the exact oracle: decoded with its column scales, the two terms equal
// the BF16 activation exactly wherever a value lies within 2^14 of its column's maximum (the high
// term keeps the top four significant bits, the low term the remaining four), and stay within
// 2^-20 of the column maximum below that.
int check_pair_quantizer(std::int32_t n, std::int32_t k, std::int32_t tokens,
                         const std::vector<std::uint16_t>& bits) {
    {
        DeviceBuffer x(bits.size() * sizeof(std::uint16_t));
        x.copy_from_host(bits.data(), bits.size() * sizeof(std::uint16_t));
        const std::size_t capacity = ops::detail::fp8_a8_pair_workspace_capacity_bytes(tokens, k);
        DeviceArena arena(capacity);
        const auto workspace = ops::detail::allocate_fp8_a8_pair_workspace(arena, tokens, k);
        Weight weight{};
        weight.k = k;
        weight.n = n;
        ops::detail::launch_fp8_a8_pair_quantize(Tensor(x.p, DType::BF16, {k, tokens}), weight,
                                                 workspace, nullptr);
        test::cuda_check(cudaDeviceSynchronize(), "pair quantize");
        const auto host = [&](const std::uint8_t* device, std::size_t size) {
            std::vector<std::uint8_t> out(size);
            test::cuda_check(cudaMemcpy(out.data(), device, size, cudaMemcpyDeviceToHost),
                             "pair quantize copy");
            return out;
        };
        const std::size_t codes = static_cast<std::size_t>(k) * tokens;
        const auto hi = host(workspace.codes_hi, codes), lo = host(workspace.codes_lo, codes);
        std::vector<float> s_hi(tokens), s_lo(tokens);
        test::cuda_check(cudaMemcpy(s_hi.data(), workspace.scales_hi, tokens * sizeof(float),
                                    cudaMemcpyDeviceToHost),
                         "pair quantize copy");
        test::cuda_check(cudaMemcpy(s_lo.data(), workspace.scales_lo, tokens * sizeof(float),
                                    cudaMemcpyDeviceToHost),
                         "pair quantize copy");
        int mismatches = 0;
        for (std::int32_t token = 0; token < tokens; ++token) {
            double maximum = 0.0;
            for (std::int32_t channel = 0; channel < k; ++channel)
                maximum = std::max(maximum, std::fabs(static_cast<double>(test::bf16_to_f32(
                                                bits[static_cast<std::size_t>(token) * k + channel]))));
            for (const float scale : {s_hi[static_cast<std::size_t>(token)],
                                      s_lo[static_cast<std::size_t>(token)]}) {
                int exponent = 0;
                if (scale != 0.0F && std::frexp(scale, &exponent) != 0.5F && mismatches++ < 3)
                    std::cerr << "FP8 A8 pair quantizer K=" << k << " token " << token
                              << ": scale " << scale << " is not a power of two\n";
            }
            for (std::int32_t channel = 0; channel < k; ++channel) {
                const std::size_t at = static_cast<std::size_t>(token) * k + channel;
                const double value   = test::bf16_to_f32(bits[at]);
                const double decoded = e4m3_value(hi[at]) * s_hi[static_cast<std::size_t>(token)] +
                                       e4m3_value(lo[at]) * s_lo[static_cast<std::size_t>(token)];
                const bool in_range  = std::fabs(value) >= std::ldexp(maximum, -14);
                const bool ok        = in_range ? decoded == value
                                                : std::fabs(decoded - value) <= std::ldexp(maximum, -20);
                if (!ok && mismatches++ < 3) {
                    std::cerr << "FP8 A8 pair quantizer K=" << k << " T=" << tokens << " token "
                              << token << " channel " << channel << ": " << value << " decodes to "
                              << decoded << '\n';
                }
            }
        }
        return mismatches != 0 ? 1 : 0;
    }
}

// Columns: random values over nine binades with an outlier channel every 61st (2^8 larger), a zero
// column, a column at the BF16 extremes, and one whose every 32nd value lies 2^20 below the rest;
// then the outlier test's own activations.
int pair_quantizer_exact() {
    int failures = 0;
    std::uint32_t seed = 972U; // run_flash_next_pair_outliers' activation seeds
    for (const auto& [n, k, min_tokens] : kFlashNextProblems) {
        constexpr std::int32_t tokens = 40;
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * tokens);
        std::uint32_t state = 1009U * static_cast<std::uint32_t>(k);
        for (std::int32_t token = 0; token < tokens; ++token) {
            for (std::int32_t channel = 0; channel < k; ++channel) {
                state = state * 1664525U + 1013904223U;
                const float unit =
                    static_cast<float>(static_cast<std::int32_t>(state >> 9) - (1 << 22)) /
                    static_cast<float>(1 << 22);
                float value = std::ldexp(unit, token % 9 - 4);
                if (channel % 61 == 7) value *= 256.0F;
                if (token == 5) value = 0.0F;
                if (token == 6) value = std::ldexp(unit, channel % 2 ? 120 : -120);
                if (token == 7 && channel % 32 == 3) value = std::ldexp(unit, -20 - 4);
                bits[static_cast<std::size_t>(token) * k + channel] = test::f32_to_bf16(value);
            }
        }
        failures += check_pair_quantizer(n, k, tokens, bits);
        // The outlier test's own activations at both of its widths.
        for (const std::int32_t t : {min_tokens, std::max(min_tokens, kPairNarrowTokens) + 44})
            failures += check_pair_quantizer(n, k, t, outlier_activations(k, t, seed));
        ++seed;
    }
    return failures;
}

// The block32 quantizer against the exact oracle: decoded with its UE8M0 block scales (word
// (k_tile, token) at k_tile * stride + token, block 4 * k_tile + j in byte j), the two terms equal
// the BF16 activation exactly wherever a value lies within 2^14 of its 32-value block's maximum, and
// stay within 2^-20 of the block maximum below that. No scale byte is 0xFF (NaN in UE8M0).
int check_pair_block32_quantizer(std::int32_t n, std::int32_t k, std::int32_t tokens,
                                 const std::vector<std::uint16_t>& bits) {
    DeviceBuffer x(bits.size() * sizeof(std::uint16_t));
    x.copy_from_host(bits.data(), bits.size() * sizeof(std::uint16_t));
    const std::size_t capacity =
        ops::detail::fp8_a8_pair_block32_workspace_capacity_bytes(tokens, k);
    DeviceArena arena(capacity);
    const auto workspace = ops::detail::allocate_fp8_a8_pair_block32_workspace(arena, tokens, k);
    Weight weight{};
    weight.k = k;
    weight.n = n;
    ops::detail::launch_fp8_a8_pair_block32_quantize(Tensor(x.p, DType::BF16, {k, tokens}), weight,
                                                     workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "pair block32 quantize");
    const auto host = [&](const void* device, std::size_t size) {
        std::vector<std::uint8_t> out(size);
        test::cuda_check(cudaMemcpy(out.data(), device, size, cudaMemcpyDeviceToHost),
                         "pair block32 quantize copy");
        return out;
    };
    const std::size_t codes = static_cast<std::size_t>(k) * tokens;
    const std::int32_t stride = ops::detail::fp8_a8_pair_block32_scale_stride(tokens);
    const std::size_t scale_bytes = static_cast<std::size_t>(k / 128) * stride * 4;
    const auto hi = host(workspace.codes_hi, codes), lo = host(workspace.codes_lo, codes);
    const auto s_hi = host(workspace.block_scales_hi, scale_bytes);
    const auto s_lo = host(workspace.block_scales_lo, scale_bytes);
    int mismatches = 0;
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t block = 0; block < k / 32; ++block) {
            const std::size_t at_scale =
                (static_cast<std::size_t>(block / 4) * stride + token) * 4 + block % 4;
            const int byte_hi = s_hi[at_scale], byte_lo = s_lo[at_scale];
            if ((byte_hi == 0xFF || byte_lo == 0xFF) && mismatches++ < 3) {
                std::cerr << "FP8 A8 pair block32 quantizer K=" << k << " token " << token
                          << " block " << block << ": a NaN scale byte\n";
            }
            double maximum = 0.0;
            for (int i = 0; i < 32; ++i)
                maximum = std::max(maximum, std::fabs(static_cast<double>(test::bf16_to_f32(
                                                bits[static_cast<std::size_t>(token) * k +
                                                     block * 32 + i]))));
            for (int i = 0; i < 32; ++i) {
                const std::size_t at = static_cast<std::size_t>(token) * k + block * 32 + i;
                const double value   = test::bf16_to_f32(bits[at]);
                const double decoded = e4m3_value(hi[at]) * std::ldexp(1.0, byte_hi - 127) +
                                       e4m3_value(lo[at]) * std::ldexp(1.0, byte_lo - 127);
                const bool in_range = std::fabs(value) >= std::ldexp(maximum, -14);
                const bool ok       = in_range ? decoded == value
                                               : std::fabs(decoded - value) <= std::ldexp(maximum, -20);
                if (!ok && mismatches++ < 3) {
                    std::cerr << "FP8 A8 pair block32 quantizer K=" << k << " T=" << tokens
                              << " token " << token << " channel " << block * 32 + i << ": "
                              << value << " decodes to " << decoded << '\n';
                }
            }
        }
    }
    return mismatches != 0 ? 1 : 0;
}

// The columns of pair_quantizer_exact, plus one whose first block holds a value 2^20 above every
// other value of the column (a massive activation: only that block's other values fall below 2^14 of
// their maximum), at T = 40 and at T = 41 (a scale stride with padding); then the block32 outlier
// test's own activations.
int pair_block32_quantizer_exact() {
    int failures       = 0;
    std::uint32_t seed = 1972U; // run_flash_next_pair_outliers' block32 activation seeds
    for (const auto& [n, k, min_tokens] : kFlashNextProblems) {
        for (const std::int32_t tokens : {40, 41}) {
            std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * tokens);
            std::uint32_t state = 1013U * static_cast<std::uint32_t>(k);
            for (std::int32_t token = 0; token < tokens; ++token) {
                for (std::int32_t channel = 0; channel < k; ++channel) {
                    state = state * 1664525U + 1013904223U;
                    const float unit =
                        static_cast<float>(static_cast<std::int32_t>(state >> 9) - (1 << 22)) /
                        static_cast<float>(1 << 22);
                    float value = std::ldexp(unit, token % 9 - 4);
                    if (channel % 61 == 7) value *= 256.0F;
                    if (token == 5) value = 0.0F;
                    if (token == 6) value = std::ldexp(unit, channel % 2 ? 120 : -120);
                    if (token == 7 && channel % 32 == 3) value = std::ldexp(unit, -20 - 4);
                    if (token == 8) value = channel == 5 ? std::ldexp(1.5F, 20) : unit;
                    bits[static_cast<std::size_t>(token) * k + channel] = test::f32_to_bf16(value);
                }
            }
            failures += check_pair_block32_quantizer(n, k, tokens, bits);
        }
        for (const std::int32_t t : {min_tokens, std::max(min_tokens, kPairNarrowTokens) + 44})
            failures += check_pair_block32_quantizer(n, k, t, outlier_activations(k, t, seed));
        ++seed;
    }
    return failures;
}

// Each pair route is deterministic: repeated calls on one input give the same bits, in both tile
// widths of every problem (test_fp8_a8_pair_repeat.cu repeats them thousands of times).
int pair_deterministic(const PairRoute& route) {
    int failures       = 0;
    std::uint32_t seed = 991U + route.seed_offset;
    for (const auto& [n, k, min_tokens] : kFlashNextProblems) {
        DeviceProblem problem(n, k, seed++);
        for (const std::int32_t t : {std::max(min_tokens, 200), std::max(min_tokens, 1024)}) {
            const std::vector<std::uint16_t> bits = outlier_activations(k, t, seed);
            DeviceBuffer x(bits.size() * sizeof(std::uint16_t));
            x.copy_from_host(bits.data(), bits.size() * sizeof(std::uint16_t));
            const auto first =
                run_linear(problem.view, x, t, ops::LinearPolicy::A16Only, route.routes);
            for (int repeat = 0; repeat < 4; ++repeat) {
                const auto again =
                    run_linear(problem.view, x, t, ops::LinearPolicy::A16Only, route.routes);
                int differing    = 0;
                for (std::size_t i = 0; i < first.size(); ++i) {
                    if (first[i] == again[i]) continue;
                    if (differing++ < 4) {
                        std::cerr << route.label << " [" << n << "," << k << "] T=" << t
                                  << " repeat " << repeat << ": token " << i / n << " row "
                                  << i % n << " changed\n";
                    }
                }
                if (differing != 0) {
                    std::cerr << route.label << " [" << n << "," << k << "] T=" << t << ": "
                              << differing << " of " << first.size()
                              << " outputs differ between calls\n";
                    ++failures;
                    break;
                }
            }
        }
    }
    return failures;
}

// Admission seams: below its problem's admission each pair flag runs the exact route bit for bit,
// and a problem that registers no pair route (the Qwen3.8-27B FP8 problems) ignores both flags at
// every width. Where both flags are set the block32 route runs (its precedence). The capacity query
// covers both pair routes from their admission.
int pair_admission_seams() {
    int failures = 0;
    const auto activation = [](const DeviceProblem& problem, std::int32_t t) {
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(problem.view.k) * t);
        for (std::size_t i = 0; i < bits.size(); ++i) {
            const auto hashed = static_cast<std::uint32_t>(i) * 2654435761U;
            bits[i] = test::f32_to_bf16(static_cast<float>(static_cast<int>(hashed >> 24) - 128) /
                                        64.0F);
        }
        DeviceBuffer x(bits.size() * sizeof(std::uint16_t));
        x.copy_from_host(bits.data(), bits.size() * sizeof(std::uint16_t));
        return x;
    };
    const auto compare = [&](const DeviceProblem& problem, std::int32_t t,
                             ops::LinearPolicy policy) {
        const DeviceBuffer x = activation(problem, t);
        const auto exact = run_linear(problem.view, x, t, policy, ops::LinearRoutes::Exact);
        for (const PairRoute& route : kPairRoutes) {
            if (exact != run_linear(problem.view, x, t, policy, route.routes)) {
                std::cerr << "FP8 [" << problem.view.n << "," << problem.view.k << "] T=" << t
                          << ": the " << route.label << " flag changed a width it does not serve\n";
                ++failures;
            }
        }
    };
    const auto precedence = [&](const DeviceProblem& problem, std::int32_t t) {
        const DeviceBuffer x = activation(problem, t);
        const auto block32   = run_linear(problem.view, x, t, ops::LinearPolicy::A16Only, kPairBlock32);
        const auto both      = run_linear(problem.view, x, t, ops::LinearPolicy::A16Only,
                                          kPair | kPairBlock32);
        if (block32 != both) {
            std::cerr << "FP8 [" << problem.view.n << "," << problem.view.k << "] T=" << t
                      << ": with both pair flags the block32 route does not take precedence\n";
            ++failures;
        }
    };
    std::uint32_t seed = 983U;
    for (const auto& [n, k, min_tokens] : kFlashNextProblems) {
        DeviceProblem problem(n, k, seed++);
        compare(problem, min_tokens - 1, ops::LinearPolicy::A16Only);
        compare(problem, 1, ops::LinearPolicy::A16Only);
        precedence(problem, min_tokens);
        precedence(problem, std::max(min_tokens, kPairNarrowTokens) + 1);
        const auto capacity = [n = n, k = k](std::int32_t first, std::int32_t last) {
            return ops::linear_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16, n, k,
                                                        ops::LinearPolicy::A16Only, first, last);
        };
        const std::size_t planes = 2U * static_cast<std::size_t>(k) * 4096U;
        // Below the pair routes only the FP16-accumulate route (where registered) may need workspace.
        if (capacity(1, min_tokens - 1) > capacity(min_tokens, min_tokens) || capacity(min_tokens, min_tokens) == 0 ||
            capacity(1, 4096) < planes || capacity(4096, 4096) != capacity(1, 4096)) {
            std::cerr << "FP8 A8 pair workspace contract mismatch for [" << n << "," << k << "]\n";
            ++failures;
        }
    }
    DeviceProblem dense(14336, 5120, seed++);
    for (const std::int32_t t : {16, 64, 300}) {
        compare(dense, t, ops::LinearPolicy::A16Only);
        compare(dense, t, ops::LinearPolicy::AllowA8);
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(linear_fp8_a8) {
    if (!infernix::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = run_fp8_a8() + pair_quantizer_exact() + pair_block32_quantizer_exact();
        for (const PairRoute& route : kPairRoutes) {
            failures += run_flash_next_fp8_a8_pair(route) + pair_deterministic(route) +
                        run_flash_next_pair_outliers(route);
        }
        failures += pair_admission_seams();
        failures += run_flash_next_mxfp8_pair_block32();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " FP8 A8 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP8 A8 Linear: " << error.what() << '\n';
        return 1;
    }
}
