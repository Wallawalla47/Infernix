// BF16 LinearSwiGLU (Qwen3.8-Flash-Next's shared expert in the bit-exact artifact, gate/up [1280, 2560]
// -> [640]) against an FP64 oracle: out = SiLU(W[i] x) * W[640 + i] x from the represented BF16 weights
// and activations. The route keeps gate and up in FP32 and rounds only the output to BF16, so
// |out - ref| <= 2^-8 |ref| + 2^-14 rms(ref). Also: each column of a 16-column call equals that column
// alone (column invariance), and widths outside 1..16 are refused.

#include "infernix/ops/linear_swiglu.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace t = infernix::test;
using infernix::DType;
using infernix::Tensor;

constexpr int kK = 2560, kM = 640, kMaxT = 16;
int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::uint32_t next(std::uint32_t& state) { return state = state * 1664525U + 1013904223U; }
float uniform(std::uint32_t& state) {
    return static_cast<float>(static_cast<std::int32_t>(next(state) >> 8) - (1 << 23)) / static_cast<float>(1 << 23);
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    try {
        // Model-scale weights (|w| ~ 2 / sqrt(K)) so SiLU's argument is of order one.
        std::uint32_t state = 7001U;
        std::vector<std::uint16_t> w(static_cast<std::size_t>(2 * kM) * kK), x(static_cast<std::size_t>(kK) * kMaxT);
        for (auto& v : w) { v = t::f32_to_bf16(uniform(state) * 3.0F / std::sqrt(static_cast<float>(kK))); }
        for (auto& v : x) { v = t::f32_to_bf16(uniform(state)); }
        void *wd = nullptr, *xd = nullptr, *yd = nullptr;
        t::cuda_check(cudaMalloc(&wd, w.size() * 2), "weight");
        t::cuda_check(cudaMalloc(&xd, x.size() * 2), "x");
        t::cuda_check(cudaMalloc(&yd, static_cast<std::size_t>(kM) * kMaxT * 2), "y");
        t::cuda_check(cudaMemcpy(wd, w.data(), w.size() * 2, cudaMemcpyHostToDevice), "upload");
        t::cuda_check(cudaMemcpy(xd, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "upload");
        infernix::Weight weight{};
        weight.payload = weight.qdata = wd;
        weight.payload_bytes          = w.size() * 2;
        weight.qtype                  = infernix::QType::BF16;
        weight.layout                 = infernix::QuantLayout::Contiguous;
        weight.ndim                   = 2;
        weight.shape[0] = weight.padded_shape[0] = weight.n = 2 * kM;
        weight.shape[1] = weight.padded_shape[1] = weight.k = kK;
        infernix::WorkspaceArena ws(1 << 20);
        const auto call = [&](int first, int columns) {
            const Tensor in(static_cast<std::uint16_t*>(xd) + static_cast<std::size_t>(first) * kK, DType::BF16,
                            {kK, columns});
            Tensor y(yd, DType::BF16, {kM, columns});
            infernix::ops::linear_swiglu(in, weight, y, ws, nullptr);
            t::cuda_check(cudaDeviceSynchronize(), "sync");
            std::vector<std::uint16_t> out(static_cast<std::size_t>(kM) * columns);
            t::cuda_check(cudaMemcpy(out.data(), yd, out.size() * 2, cudaMemcpyDeviceToHost), "read");
            return out;
        };
        for (int T = 1; T <= kMaxT; ++T) {
            const auto got = call(0, T);
            std::vector<double> ref(static_cast<std::size_t>(kM) * T);
            double ss = 0;
            for (int c = 0; c < T; ++c) {
                for (int i = 0; i < kM; ++i) {
                    double g = 0, u = 0;
                    for (int k = 0; k < kK; ++k) {
                        const double xv = t::bf16_to_f32(x[static_cast<std::size_t>(c) * kK + k]);
                        g += t::bf16_to_f32(w[static_cast<std::size_t>(i) * kK + k]) * xv;
                        u += t::bf16_to_f32(w[static_cast<std::size_t>(kM + i) * kK + k]) * xv;
                    }
                    const double v = g / (1.0 + std::exp(-g)) * u;
                    ref[static_cast<std::size_t>(c) * kM + i] = v;
                    ss += v * v;
                }
            }
            const double rms = std::sqrt(ss / static_cast<double>(ref.size()));
            std::size_t bad = 0;
            for (std::size_t i = 0; i < ref.size(); ++i) {
                const double e = std::fabs(static_cast<double>(t::bf16_to_f32(got[i])) - ref[i]);
                bad += e > std::ldexp(1.0, -8) * std::fabs(ref[i]) + std::ldexp(1.0, -14) * rms ? 1U : 0U;
            }
            check(bad == 0, "T=" + std::to_string(T) + ": within the bound (" + std::to_string(bad) + " outside)");
        }
        const auto all = call(0, kMaxT);
        for (int c = 0; c < kMaxT; ++c) {
            const auto one = call(c, 1);
            check(std::equal(one.begin(), one.end(), all.begin() + static_cast<std::ptrdiff_t>(c) * kM),
                  "column " + std::to_string(c) + " of a 16-column call equals the column alone");
        }
        bool refused = false;
        try {
            (void)call(0, kMaxT + 1);
        } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "17 columns are refused (the stream route serves 1..16)");
        cudaFree(wd);
        cudaFree(xd);
        cudaFree(yd);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ++failures;
    }
    std::printf(failures == 0 ? "BF16 LinearSwiGLU checks passed\n" : "FAIL: %d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
