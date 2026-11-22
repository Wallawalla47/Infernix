// Inkling 2's vision-tower elementwise Ops (include/infernix/ops/inkling2_vision.h) against FP64 oracles from the
// represented BF16 inputs. Criteria (stated before the run):
//   vision_qkv_rotary: v and the class token's (0, 0) rows equal bf16_rn(x + b) exactly (no rotation); rotated q/k
//     |gpu - ref| <= 2^-8 |ref| + 2^-14 max|x + b| (BF16 storage, the FP32 angle p * inv at p <= 400 and sincosf);
//   vision_scaled_residual: |gpu - ref| <= 2^-8 |ref| + 2^-16 max|ref|;
//   a single-row call equals that row of a many-row call, and a repeated call is bit for bit the first.

#include "infernix/ops/inkling2_vision.h"
#include "ops/common/canonical_math.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace canon = infernix::ops::canon;
namespace ik    = infernix::ops::inkling2;
using infernix::test::cuda_check;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

template <class T>
T* upload(const std::vector<T>& v) {
    T* p = nullptr;
    cuda_check(cudaMalloc(&p, std::max<std::size_t>(v.size(), 1) * sizeof(T)), "cudaMalloc");
    cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}
template <class T>
std::vector<T> download(const T* p, std::size_t n) {
    std::vector<T> v(n);
    cuda_check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}

std::vector<std::uint16_t> random_bf16(std::size_t n, float lo, float hi, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<std::uint16_t> out(n);
    for (auto& w : out) { w = canon::f32_to_bf16_rn(u(rng)); }
    return out;
}

double f(std::uint16_t w) { return static_cast<double>(canon::bf16_to_f32(w)); }

void qkv_case(int tokens, std::uint32_t seed) {
    constexpr int heads = 16, D = 64, W = heads * D, R = D / 4;
    constexpr float theta = 10000.0F;
    const auto qkv  = random_bf16(static_cast<std::size_t>(tokens) * 3 * W, -4.0F, 4.0F, seed);
    const auto bias = random_bf16(3 * W, -1.0F, 1.0F, seed + 1);
    std::vector<std::int32_t> positions(2 * tokens, 0); // token 0: the class token at (0, 0)
    std::mt19937 rng(seed + 2);
    std::uniform_int_distribution<int> p(1, 400);
    for (int t = 1; t < tokens; ++t) { positions[2 * t] = p(rng), positions[2 * t + 1] = p(rng); }
    std::uint16_t* d_qkv = upload(qkv);
    const std::uint16_t* d_bias = upload(bias);
    const std::int32_t* d_pos   = upload(positions);
    ik::vision_qkv_rotary({d_qkv, d_bias, d_pos, tokens, heads, D, theta}, nullptr);
    const auto got = download(d_qkv, qkv.size());

    bool exact = true, within = true;
    double worst = 0.0;
    for (int t = 0; t < tokens; ++t) {
        double zmax = 0.0;
        for (int i = 0; i < 2 * W; ++i) { zmax = std::max(zmax, std::abs(f(qkv[t * 3 * W + i]) + f(bias[i]))); }
        for (int i = 0; i < 3 * W; i += 2) {
            const std::size_t at = static_cast<std::size_t>(t) * 3 * W + i;
            const double z0 = f(qkv[at]) + f(bias[i]), z1 = f(qkv[at + 1]) + f(bias[i + 1]);
            if (i >= 2 * W || t == 0) { // v, and the class token's identity rotation: one exact rounding
                exact &= got[at] == canon::f32_to_bf16_rn(static_cast<float>(z0)) &&
                         got[at + 1] == canon::f32_to_bf16_rn(static_cast<float>(z1));
                continue;
            }
            const int j    = (i % D) / 2;
            const int axis = j < R ? 1 : 0;
            const double a = positions[2 * t + axis] * std::pow(static_cast<double>(theta), -static_cast<double>(j % R) / R);
            const double r0 = z0 * std::cos(a) - z1 * std::sin(a), r1 = z0 * std::sin(a) + z1 * std::cos(a);
            const double e0 = std::abs(f(got[at]) - r0), e1 = std::abs(f(got[at + 1]) - r1);
            const double b0 = std::ldexp(std::abs(r0), -8) + std::ldexp(zmax, -14);
            const double b1 = std::ldexp(std::abs(r1), -8) + std::ldexp(zmax, -14);
            within &= e0 <= b0 && e1 <= b1;
            worst = std::max({worst, e0 / b0, e1 / b1});
        }
    }
    std::printf("vision_qkv_rotary T=%d: worst error / bound %.3f\n", tokens, worst);
    check(exact, "vision_qkv_rotary: v or the class token is not bf16_rn(x + b)");
    check(within, "vision_qkv_rotary: a rotated q/k value is outside its bound");

    // One row alone equals that row of the many-row call; a repeated call is bit for bit the first.
    const int row = tokens / 2;
    std::vector<std::uint16_t> one(qkv.begin() + row * 3 * W, qkv.begin() + (row + 1) * 3 * W);
    std::uint16_t* d_one = upload(one);
    const std::int32_t* d_one_pos = upload(std::vector<std::int32_t>{positions[2 * row], positions[2 * row + 1]});
    ik::vision_qkv_rotary({d_one, d_bias, d_one_pos, 1, heads, D, theta}, nullptr);
    check(download(d_one, one.size()) == std::vector<std::uint16_t>(got.begin() + row * 3 * W, got.begin() + (row + 1) * 3 * W),
          "vision_qkv_rotary: a single-row call differs from its row in a many-row call");
    cuda_check(cudaMemcpy(d_qkv, qkv.data(), qkv.size() * 2, cudaMemcpyHostToDevice), "reset");
    ik::vision_qkv_rotary({d_qkv, d_bias, d_pos, tokens, heads, D, theta}, nullptr);
    check(download(d_qkv, qkv.size()) == got, "vision_qkv_rotary: a repeated call differs");
    cudaFree(d_qkv), cudaFree(const_cast<std::uint16_t*>(d_bias)), cudaFree(const_cast<std::int32_t*>(d_pos));
    cudaFree(d_one), cudaFree(const_cast<std::int32_t*>(d_one_pos));
}

void residual_case(int tokens, int width, std::uint32_t seed) {
    const std::size_t n = static_cast<std::size_t>(tokens) * width;
    const auto x = random_bf16(n, -8.0F, 8.0F, seed), y = random_bf16(n, -6.0F, 6.0F, seed + 1);
    const auto bias = random_bf16(width, -1.0F, 1.0F, seed + 2), scale = random_bf16(width, -0.5F, 0.5F, seed + 3);
    std::uint16_t* d_x = upload(x);
    const std::uint16_t *d_y = upload(y), *d_b = upload(bias), *d_s = upload(scale);
    ik::vision_scaled_residual({d_x, d_y, d_b, d_s, tokens, width}, nullptr);
    const auto got = download(d_x, n);
    std::vector<double> ref(n);
    double rmax = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const int d = static_cast<int>(i % width);
        ref[i]      = f(x[i]) + f(scale[d]) * (f(y[i]) + f(bias[d]));
        rmax        = std::max(rmax, std::abs(ref[i]));
    }
    bool within  = true;
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double bound = std::ldexp(std::abs(ref[i]), -8) + std::ldexp(rmax, -16);
        const double e     = std::abs(f(got[i]) - ref[i]);
        within &= e <= bound;
        worst = std::max(worst, e / bound);
    }
    std::printf("vision_scaled_residual T=%d W=%d: worst error / bound %.3f\n", tokens, width, worst);
    check(within, "vision_scaled_residual is outside its bound");
    check(download(d_y, n) == y, "vision_scaled_residual changed y");
    cuda_check(cudaMemcpy(d_x, x.data(), n * 2, cudaMemcpyHostToDevice), "reset");
    ik::vision_scaled_residual({d_x, d_y, d_b, d_s, tokens, width}, nullptr);
    check(download(d_x, n) == got, "vision_scaled_residual: a repeated call differs");
    cudaFree(d_x), cudaFree(const_cast<std::uint16_t*>(d_y)), cudaFree(const_cast<std::uint16_t*>(d_b));
    cudaFree(const_cast<std::uint16_t*>(d_s));
}

} // namespace

INFERNIX_OP_TEST(inkling2_vision) {
    qkv_case(1, 3);    // the class token alone
    qkv_case(49, 5);   // 3 x 4 cells
    qkv_case(4097, 7); // 32 x 32 cells
    residual_case(49, 1024, 11);
    residual_case(1, 1024, 13);
    residual_case(4097, 1024, 17);
    std::printf(g_failures == 0 ? "PASS\n" : "FAILED (%d)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
