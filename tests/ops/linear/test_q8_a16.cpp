#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <utility>
#include <vector>

namespace {
using namespace ninfer::test::linear;

struct Geometry {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
};

constexpr std::array kGeometries{
    Geometry{1024, 2048, 257U},  Geometry{1024, 5120, 223U},  Geometry{2048, 4096, 251U},
    Geometry{2048, 4608, 271U},  Geometry{2048, 16384, 283U}, Geometry{4608, 4608, 277U},
    Geometry{5120, 4608, 281U},  Geometry{5120, 6144, 239U},  Geometry{5120, 10240, 211U},
    Geometry{5120, 17408, 241U}, Geometry{5120, 25600, 293U}, Geometry{6144, 5120, 227U},
    Geometry{9216, 2048, 263U},  Geometry{12288, 2048, 269U}, Geometry{14336, 5120, 229U},
    Geometry{17408, 5120, 243U}, Geometry{34816, 5120, 233U}, Geometry{248320, 5120, 197U},
    // Qwen3.8-Flash-Next's dense Q8 classes on the register-streamed route: hyper-connection
    // down (+inject), final mixer down, hyper-connection up (K 320, padded to 384), GDN projection,
    // GDN and QSA output, shared expert gate/up and down, PLE projection, the MTP attention group
    // and the MTP embedding/hidden projections.
    Geometry{324, 10240, 301U},  Geometry{320, 10240, 307U},  Geometry{10240, 320, 311U},
    Geometry{16384, 2560, 313U}, Geometry{2560, 6144, 317U},  Geometry{1280, 2560, 337U},
    Geometry{2560, 640, 347U},   Geometry{12800, 2560, 349U}, Geometry{13952, 2560, 353U},
    Geometry{2560, 2560, 359U},
    // An unregistered geometry on the runtime-shape route (predicated SIMT, then MMA tiles).
    Geometry{2560, 4096, 331U}};

int q8_a16_conformance() {
    int failures = 0;
    for (const auto& shape : kGeometries) {
        std::vector<Invocation> calls;
        // Cover live-column tails and the transitions from K-split to tiled contractions.
        for (int t : {1,  2,  3,  4,  5,  7,  8,  9,  15,  16,  17,  23,  24,  25,
                      31, 32, 33, 39, 40, 41, 44, 47, 48,  49,  55,  56,  57,  63,
                      64, 65, 79, 80, 81, 95, 96, 97, 127, 128, 129, 256, 1024}) {
            calls.push_back({t});
        }
        if (shape.n == 2048 && shape.k == 4096) {
            for (int t : {895, 896, 897}) calls.push_back({t});
        }
        if (shape.n == 6144 && shape.k == 5120) {
            for (int t : {191, 192, 193}) calls.push_back({t});
        }
        if (shape.k == 4608) {
            for (int t : {6, 11, 12, 13, 14, 19, 20, 21, 27, 28, 29}) calls.push_back({t});
            if (shape.n == 2048) {
                for (int t : {870, 871, 872}) calls.push_back({t});
            }
            if (shape.n == 4608) {
                for (int t : {255, 257}) calls.push_back({t});
            }
        }
        if (shape.n == 9216 && shape.k == 2048) {
            for (int t : {12, 13, 14}) calls.push_back({t});
        }
        if (shape.n == 248320) calls.push_back({34});
        if (shape.n == 2048 && shape.k == 16384) {
            for (int t : {383,  384,  385,  479,  480,  481,  639,  640,  641,  703,
                          704,  705,  959,  960,  961,  1343, 1344, 1345, 1679, 1680,
                          1681, 2015, 2016, 2017, 2111, 2112, 2113, 4096}) {
                calls.push_back({t});
            }
        }
        for (int t : {1, 8, 16, 32, 48, 64, 65, 128}) {
            calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
        }
        for (int t : {1, 16, 64, 128}) calls.push_back({t, CallForm::A16Convenience});
        calls.push_back({17, CallForm::Policy, ninfer::ops::LinearPolicy::AllowA8});
        calls.push_back({64, CallForm::Policy, ninfer::ops::LinearPolicy::AllowA4});
        failures += run_shape("Q8_A16", ActivationCompute::A16, make_q8_g32_fp16_weight,
                              {shape.n, shape.k, shape.seed, Comparison::Sampled, true, calls});
    }
    return failures;
}

// Speculative verification relies on this for one row (design §11.3): an output column is
// bit-identical whether it is computed alone or beside other columns, up to 8 columns on every route
// (the register-streamed and SIMT routes reduce a column identically, so a shape may mix them by
// width), and as far as each Flash-Next shape keeps the register-streamed route beyond 8 (passes,
// token slices and K-split warps): 64 for the hyper-connection and final-mixer downs, 48 for the
// shared expert gate/up, 16 for its down.
int q8_a16_column_invariance() {
    int failures = 0;
    for (const auto& [shape, kColumns] :
         {std::pair{Geometry{324, 10240, 401U}, 64},  std::pair{Geometry{320, 10240, 403U}, 64},
          std::pair{Geometry{10240, 320, 419U}, 8},   std::pair{Geometry{16384, 2560, 421U}, 8},
          std::pair{Geometry{2560, 6144, 431U}, 8},   std::pair{Geometry{1280, 2560, 433U}, 48},
          std::pair{Geometry{2560, 640, 439U}, 16},   std::pair{Geometry{12800, 2560, 443U}, 8},
          std::pair{Geometry{13952, 2560, 449U}, 8},  std::pair{Geometry{2560, 2560, 457U}, 8},
          std::pair{Geometry{2560, 4096, 409U}, 8}}) {
        const auto host = make_q8_g32_fp16_weight(shape.n, shape.k, shape.seed);
        std::vector<std::uint16_t> activation(static_cast<std::size_t>(shape.k) * kColumns);
        std::uint32_t state = shape.seed;
        for (auto& bits : activation) {
            state = state * 1664525U + 1013904223U;
            bits  = ninfer::test::f32_to_bf16(
                static_cast<float>(static_cast<std::int32_t>(state >> 8) - (1 << 23)) /
                static_cast<float>(1 << 23));
        }
        const std::size_t out_bytes = static_cast<std::size_t>(shape.n) * kColumns * 2;
        void *weight = nullptr, *x = nullptr, *y = nullptr;
        ninfer::test::cuda_check(cudaMalloc(&weight, host.payload.size()), "weight");
        ninfer::test::cuda_check(cudaMalloc(&x, activation.size() * 2), "activation");
        ninfer::test::cuda_check(cudaMalloc(&y, out_bytes), "output");
        ninfer::test::cuda_check(
            cudaMemcpy(weight, host.payload.data(), host.payload.size(), cudaMemcpyHostToDevice),
            "upload weight");
        ninfer::test::cuda_check(
            cudaMemcpy(x, activation.data(), activation.size() * 2, cudaMemcpyHostToDevice),
            "upload activation");
        const ninfer::Weight w = host.device_weight(weight);
        std::vector<std::uint16_t> alone(static_cast<std::size_t>(shape.n) * kColumns);
        for (int column = 0; column < kColumns; ++column) {
            ninfer::Tensor input(static_cast<std::uint16_t*>(x) +
                                     static_cast<std::size_t>(column) * shape.k,
                                 ninfer::DType::BF16, {shape.k, 1});
            ninfer::Tensor output(y, ninfer::DType::BF16, {shape.n, 1});
            ninfer::ops::linear(input, w, output, nullptr);
            ninfer::test::cuda_check(
                cudaMemcpy(alone.data() + static_cast<std::size_t>(column) * shape.n, y,
                           static_cast<std::size_t>(shape.n) * 2, cudaMemcpyDeviceToHost),
                "read single column");
        }
        for (int t = 2; t <= kColumns; ++t) {
            ninfer::Tensor input(x, ninfer::DType::BF16, {shape.k, t});
            ninfer::Tensor output(y, ninfer::DType::BF16, {shape.n, t});
            ninfer::ops::linear(input, w, output, nullptr);
            std::vector<std::uint16_t> together(static_cast<std::size_t>(shape.n) * t);
            ninfer::test::cuda_check(
                cudaMemcpy(together.data(), y, together.size() * 2, cudaMemcpyDeviceToHost),
                "read columns");
            std::size_t differing = 0;
            for (std::size_t i = 0; i < together.size(); ++i) differing += together[i] != alone[i];
            if (differing != 0) {
                std::cerr << "Q8_A16 [" << shape.n << "," << shape.k << "] T=" << t << ": "
                          << differing << " outputs differ from the columns computed alone\n";
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
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = q8_a16_conformance() + q8_a16_column_invariance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q8_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q8_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
