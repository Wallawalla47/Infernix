// The canonical arithmetic of offloaded_sparse_moe on the GPU (docs/maintainer/qwen3_8-flash-next-design.md §16.2):
// the narrow kernels' scalar functions and decoders, against the same header on the CPU. The narrow kernels
// themselves are qualified against the CPU engine by test_offloaded_moe_layer.cu.
//
// Oracles:
//   - the canonical scalar functions (E4M3/E2M1 encoders, exp_c, silu_c) on the GPU against the
//     same header on the CPU: exhaustively over BF16 for exp/SiLU, and over every FP32 word near
//     each encoder rounding boundary plus a strided sweep of all FP32 words;
//   - the E2M1 decoders (scalar on host and device, four codes at once on the device) against the
//     E2M1 table, exhaustively.

#include "ops/op_family.h"
#include "ops/common/canonical_math.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace canon = infernix::ops::canon;
using infernix::test::cuda_check;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

template <class T> T* device_copy(const std::vector<T>& v) {
    T* p = nullptr;
    cuda_check(cudaMalloc(&p, v.size() * sizeof(T)), "cudaMalloc");
    cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
    return p;
}

// ---------------------------------------------------------------------------- scalar functions

__global__ void scalar_kernel(const std::uint32_t* in, std::uint32_t* out, std::size_t n) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n) { return; }
    const float x = canon::f32_from_bits(in[i]);
    const float a = canon::f32_from_bits(in[i] & 0x7FFFFFFFU);
    std::uint32_t r = canon::e4m3_rn_satfinite(a);
    r |= static_cast<std::uint32_t>(canon::e2m1_rn_satfinite(x)) << 8;
    out[2 * i]     = r;
    out[2 * i + 1] = 0;
    if ((in[i] & 0xFFFFU) == 0) { // a BF16 input: exp_c and silu_c
        out[2 * i + 1] = canon::f32_bits(canon::silu_c(x)) ^ (canon::f32_bits(canon::exp_c(x)) * 2654435761U);
    }
}

std::uint32_t scalar_cpu(std::uint32_t bits, std::uint32_t& transcendental) {
    const float x = canon::f32_from_bits(bits);
    const float a = canon::f32_from_bits(bits & 0x7FFFFFFFU);
    std::uint32_t r = canon::e4m3_rn_satfinite(a);
    r |= static_cast<std::uint32_t>(canon::e2m1_rn_satfinite(x)) << 8;
    transcendental = 0;
    if ((bits & 0xFFFFU) == 0) {
        transcendental = canon::f32_bits(canon::silu_c(x)) ^ (canon::f32_bits(canon::exp_c(x)) * 2654435761U);
    }
    return r;
}

void test_scalar_functions() {
    std::vector<std::uint32_t> words;
    // Every BF16 pattern.
    for (std::uint32_t h = 0; h < 65536; ++h) { words.push_back(h << 16); }
    // +-4096 words around every E4M3 and E2M1 rounding boundary (midpoints of adjacent grid values,
    // and the saturation thresholds), for both signs.
    std::vector<double> mids;
    for (int w = 0; w < 126; ++w) {
        const double v0 = canon::e4m3_value(static_cast<unsigned>(w)), v1 = canon::e4m3_value(static_cast<unsigned>(w + 1));
        mids.push_back((v0 + v1) / 2);
    }
    mids.push_back(464.0);
    for (double m : {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 7.0}) { mids.push_back(m); }
    for (double m : mids) {
        const std::uint32_t c = canon::f32_bits(static_cast<float>(m));
        for (std::int32_t d = -4096; d <= 4096; ++d) {
            words.push_back(c + static_cast<std::uint32_t>(d));
            words.push_back((c + static_cast<std::uint32_t>(d)) | 0x80000000U);
        }
    }
    // A strided sweep of all words (non-finite inputs included; the encoders define them).
    for (std::uint64_t w = 0; w < (1ULL << 32); w += 997) { words.push_back(static_cast<std::uint32_t>(w)); }

    std::uint32_t* d_in  = device_copy(words);
    std::uint32_t* d_out = nullptr;
    cuda_check(cudaMalloc(&d_out, words.size() * 2 * sizeof(std::uint32_t)), "cudaMalloc");
    const unsigned blocks = static_cast<unsigned>((words.size() + 255) / 256);
    scalar_kernel<<<blocks, 256>>>(d_in, d_out, words.size());
    cuda_check(cudaDeviceSynchronize(), "scalar kernel");
    std::vector<std::uint32_t> out(words.size() * 2);
    cuda_check(cudaMemcpy(out.data(), d_out, out.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
    cudaFree(d_in);
    cudaFree(d_out);
    long bad = 0;
    for (std::size_t i = 0; i < words.size(); ++i) {
        std::uint32_t t = 0;
        const std::uint32_t r = scalar_cpu(words[i], t);
        if (r != out[2 * i] || t != out[2 * i + 1]) {
            if (bad < 5) { std::fprintf(stderr, "scalar mismatch at %08x\n", words[i]); }
            ++bad;
        }
    }
    std::printf("canonical scalar functions: %zu inputs, %ld mismatches\n", words.size(), bad);
    check(bad == 0, "GPU canonical scalar functions equal the CPU's");
}

// ------------------------------------------------------------------------------- E2M1 decoding

// Thread i decodes the four codes of i (nibbles 0-3) with e2m1_x2_quad from byte lanes, and code
// i & 15 with the device e2m1_x2.
__global__ void e2m1_decode_kernel(std::uint32_t* quads, std::int32_t* scalars) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 65536U) { return; }
    const std::uint32_t lanes = (i & 15U) | ((i >> 4) & 15U) << 8 | ((i >> 8) & 15U) << 16 | ((i >> 12) & 15U) << 24;
    quads[i] = canon::e2m1_x2_quad(lanes);
    if (i < 16U) { scalars[i] = canon::e2m1_x2(i); }
}

// Exhaustive against the E2M1 table itself: twice {0, 0.5, 1, 1.5, 2, 3, 4, 6}, negated by bit 3.
void test_e2m1_decode() {
    constexpr std::int32_t kDoubled[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    const auto doubled = [&](std::uint32_t code) {
        return (code & 8U) != 0 ? -kDoubled[code & 7U] : kDoubled[code & 7U];
    };
    std::uint32_t* d_quads = nullptr;
    std::int32_t* d_scalars = nullptr;
    cuda_check(cudaMalloc(&d_quads, 65536 * sizeof(std::uint32_t)), "cudaMalloc");
    cuda_check(cudaMalloc(&d_scalars, 16 * sizeof(std::int32_t)), "cudaMalloc");
    e2m1_decode_kernel<<<256, 256>>>(d_quads, d_scalars);
    cuda_check(cudaDeviceSynchronize(), "e2m1 decode kernel");
    std::vector<std::uint32_t> quads(65536);
    std::vector<std::int32_t> scalars(16);
    cuda_check(cudaMemcpy(quads.data(), d_quads, quads.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "copy");
    cuda_check(cudaMemcpy(scalars.data(), d_scalars, scalars.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost),
               "copy");
    cudaFree(d_quads);
    cudaFree(d_scalars);
    long bad = 0;
    for (std::uint32_t c = 0; c < 16; ++c) {
        bad += scalars[c] != doubled(c);
        bad += canon::e2m1_x2(c) != doubled(c); // the host form
    }
    for (std::uint32_t i = 0; i < 65536; ++i) {
        for (int j = 0; j < 4; ++j) {
            const auto byte = static_cast<std::int8_t>((quads[i] >> (8 * j)) & 0xFFU);
            bad += byte != doubled((i >> (4 * j)) & 15U);
        }
    }
    std::printf("E2M1 decode: 16 codes, 65536 code quads, %ld mismatches\n", bad);
    check(bad == 0, "E2M1 decoders equal the E2M1 table");
}

} // namespace

INFERNIX_OP_TEST(offloaded_moe_cuda) {
    if (infernix::test::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        test_scalar_functions();
        test_e2m1_decode();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all offloaded_moe GPU checks passed\n");
    return 0;
}
