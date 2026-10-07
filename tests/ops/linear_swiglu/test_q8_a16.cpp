#include "core/arena.h"
#include "core/weight.h"
#include "infernix/ops/linear_swiglu.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

int dflash2_conformance() {
    using namespace infernix;
    using namespace infernix::test::linear_swiglu;
    std::vector<std::int32_t> tokens;
    for (int t = 1; t <= 128; ++t) tokens.push_back(t);
    for (int t : {129, 256, 1024}) tokens.push_back(t);
    constexpr std::array graphs{1,  16, 32, 40, 41, 51, 52, 63,  64,
                                65, 80, 81, 88, 89, 96, 97, 128, 129};
    return run_profile("LinearSwiGLU Q8_A16 DFlash2",
                       {QType::Q8_G32_FP16, 34816, 5120, 17408, 1603U, ActivationCompute::A16},
                       tokens, graphs);
}

// Qwen3.8-Flash-Next's shared expert, gate/up [1280, 2560] -> [640]: FP64 conformance across the
// stream-pair route (T <= 16) and the paired MMA tiles, graph replays at the seams, and column
// invariance at T <= 16 (each column of a 16-column call equals that column alone).
int flash_next_shared() {
    using namespace infernix;
    using namespace infernix::test::linear_swiglu;
    std::vector<std::int32_t> tokens;
    for (int t = 1; t <= 17; ++t) tokens.push_back(t);
    for (int t : {33, 64, 65, 512}) tokens.push_back(t);
    constexpr std::array graphs{1, 8, 16, 17, 65};
    const Profile profile{QType::Q8_G32_FP16, 1280, 2560, 640, 1607U, ActivationCompute::A16};
    int failures = run_profile("LinearSwiGLU Q8_A16 Flash-Next shared expert", profile, tokens, graphs);

    constexpr int kT = 16, kK = 2560, kM = 640;
    const auto host = infernix::test::quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, 2 * kM, kK, 1609U, {});
    std::vector<std::uint16_t> x(static_cast<std::size_t>(kK) * kT);
    std::uint32_t state = 1611U;
    for (auto& bits : x) {
        state = state * 1664525U + 1013904223U;
        bits  = infernix::test::f32_to_bf16(static_cast<float>(static_cast<std::int32_t>(state >> 8) - (1 << 23)) /
                                          static_cast<float>(1 << 23));
    }
    void *w = nullptr, *xd = nullptr, *yd = nullptr;
    infernix::test::cuda_check(cudaMalloc(&w, host.payload.size()), "weight");
    infernix::test::cuda_check(cudaMalloc(&xd, x.size() * 2), "x");
    infernix::test::cuda_check(cudaMalloc(&yd, static_cast<std::size_t>(kM) * kT * 2), "y");
    infernix::test::cuda_check(cudaMemcpy(w, host.payload.data(), host.payload.size(), cudaMemcpyHostToDevice), "upload");
    infernix::test::cuda_check(cudaMemcpy(xd, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "upload");
    const Weight weight = host.device_weight(w);
    WorkspaceArena ws(1 << 20);
    const auto call = [&](int first, int columns, std::vector<std::uint16_t>& out) {
        const Tensor in(static_cast<std::uint16_t*>(xd) + static_cast<std::size_t>(first) * kK, DType::BF16, {kK, columns});
        Tensor y(yd, DType::BF16, {kM, columns});
        ops::linear_swiglu(in, weight, y, ws, nullptr);
        out.resize(static_cast<std::size_t>(kM) * columns);
        infernix::test::cuda_check(cudaMemcpy(out.data(), yd, out.size() * 2, cudaMemcpyDeviceToHost), "read");
    };
    std::vector<std::uint16_t> all, one;
    call(0, kT, all);
    bool same = true;
    for (int c = 0; c < kT && same; ++c) {
        call(c, 1, one);
        same = std::equal(one.begin(), one.end(), all.begin() + static_cast<std::ptrdiff_t>(c) * kM);
    }
    if (!same) {
        std::cerr << "FAIL LinearSwiGLU Flash-Next shared expert: a column of a 16-column call differs from the column alone\n";
        ++failures;
    }
    cudaFree(w);
    cudaFree(xd);
    cudaFree(yd);
    return failures;
}

int main() {
    using namespace infernix;
    using namespace infernix::test::linear_swiglu;

    try {
        // One public numerical case begins each materially distinct Q8 implementation interval;
        // selected endpoints exercise exact-T and predicated tails without inspecting selectors.
        constexpr std::array<std::int32_t, 25> kTokenCases{
            1,   2,   6,   32,  33,  40,  41,  48,  49,  65,  81,  97,  129,
            193, 241, 256, 257, 265, 289, 321, 385, 449, 513, 560, 561,
        };
        int failures = run_profile(
            "LinearSwiGLU Q8_A16",
            {QType::Q8_G32_FP16, 12288, 2048, 6144, 1601U, ActivationCompute::A16}, kTokenCases);

        failures += dflash2_conformance();
        failures += flash_next_shared();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU Q8_A16 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU Q8_A16 test failed: " << error.what() << '\n';
        return 1;
    }
}
