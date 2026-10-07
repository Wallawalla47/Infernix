// GPU narrow route of offloaded_sparse_moe against the CPU engine
// (docs/maintainer/qwen3_8-flash-next-design.md §16.2, §19 M3 "placement invariance").
//
// Oracles:
//   - the pinned golden output hashes that every CPU ISA reproduces (exact);
//   - the CPU engine's output for random experts, column counts 1-8, split and shared input scales,
//     records in device memory (cache frame) and in mapped host memory (zero-copy), bit for bit;
//   - the canonical scalar functions (E4M3/E2M1 encoders, exp_c, silu_c) on the GPU against the
//     same header on the CPU: exhaustively over BF16 for exp/SiLU, and over every FP32 word near
//     each encoder rounding boundary plus a strided sweep of all FP32 words;
//   - the E2M1 decoders (scalar on host and device, four codes at once on the device) against the
//     E2M1 table, exhaustively.

#include "ops/common/canonical_math.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"
#include "ops/offloaded_sparse_moe/cuda/narrow_expert.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace canon    = infernix::ops::canon;
namespace moe      = infernix::ops::offloaded_moe;
namespace fixtures = infernix::test::offloaded_moe;
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

std::vector<std::uint16_t> cpu_expert(const fixtures::Expert& e, const std::vector<std::uint16_t>& x, int ncols) {
    std::vector<std::uint16_t> y(static_cast<std::size_t>(ncols) * moe::kHidden);
    const std::uint16_t* xp[moe::kMaxColumns];
    std::uint16_t* yp[moe::kMaxColumns];
    for (int c = 0; c < ncols; ++c) {
        xp[c] = &x[static_cast<std::size_t>(c) * moe::kHidden];
        yp[c] = &y[static_cast<std::size_t>(c) * moe::kHidden];
    }
    moe::expert_forward(moe::best_cpu_isa(), e.record.data(), e.scales, ncols, xp, yp);
    return y;
}

struct Case {
    fixtures::Expert expert;
    std::vector<std::uint16_t> x;
    int ncols;
    bool mapped; // record read zero-copy from mapped pinned host memory
};

// Runs every case as one launch (one job each) and returns the GPU outputs.
std::vector<std::vector<std::uint16_t>> gpu_experts(const std::vector<Case>& cases) {
    std::vector<moe::NarrowJob> jobs;
    std::vector<void*> device_allocs, host_allocs;
    std::vector<std::uint16_t*> outputs;
    for (const Case& c : cases) {
        moe::NarrowJob job{};
        if (c.mapped) {
            void* h = nullptr;
            cuda_check(cudaHostAlloc(&h, moe::kRecordBytes, cudaHostAllocMapped), "cudaHostAlloc");
            std::memcpy(h, c.expert.record.data(), moe::kRecordBytes);
            void* d = nullptr;
            cuda_check(cudaHostGetDevicePointer(&d, h, 0), "cudaHostGetDevicePointer");
            job.record = static_cast<const std::uint8_t*>(d);
            host_allocs.push_back(h);
        } else {
            auto* d = device_copy(c.expert.record);
            job.record = d;
            device_allocs.push_back(d);
        }
        auto* x = device_copy(c.x);
        device_allocs.push_back(x);
        std::uint16_t* y = nullptr;
        cuda_check(cudaMalloc(&y, c.x.size() * sizeof(std::uint16_t)), "cudaMalloc");
        cuda_check(cudaMemset(y, 0xFF, c.x.size() * sizeof(std::uint16_t)), "cudaMemset");
        device_allocs.push_back(y);
        outputs.push_back(y);
        job.scales = c.expert.scales;
        job.x      = x;
        job.y      = y;
        job.ncols  = c.ncols;
        jobs.push_back(job);
    }
    auto* d_jobs = device_copy(jobs);
    void* workspace = nullptr;
    cuda_check(cudaMalloc(&workspace, moe::narrow_workspace_bytes(static_cast<int>(jobs.size()))), "cudaMalloc");
    moe::launch_narrow_experts(d_jobs, static_cast<int>(jobs.size()), workspace, nullptr);
    cuda_check(cudaDeviceSynchronize(), "narrow experts");
    std::vector<std::vector<std::uint16_t>> result;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        std::vector<std::uint16_t> y(cases[i].x.size());
        cuda_check(cudaMemcpy(y.data(), outputs[i], y.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
        result.push_back(std::move(y));
    }
    for (void* p : device_allocs) { cudaFree(p); }
    for (void* p : host_allocs) { cudaFreeHost(p); }
    cudaFree(d_jobs);
    cudaFree(workspace);
    return result;
}

void test_golden() {
    std::vector<Case> cases;
    for (int n : {1, 4}) {
        auto g = fixtures::golden_case(n);
        cases.push_back({std::move(g.expert), std::move(g.x), n, false});
    }
    const auto y  = gpu_experts(cases);
    const auto h1 = fixtures::output_hash(y[0]), h4 = fixtures::output_hash(y[1]);
    if (h1 != fixtures::kGolden1 || h4 != fixtures::kGolden4) {
        std::fprintf(stderr, "golden gpu: %016llx %016llx\n", static_cast<unsigned long long>(h1),
                     static_cast<unsigned long long>(h4));
    }
    check(h1 == fixtures::kGolden1 && h4 == fixtures::kGolden4, "GPU reproduces the golden expert output hashes");
}

void test_cpu_equality() {
    std::mt19937 rng(91);
    std::vector<Case> cases;
    for (int i = 0; i < 16; ++i) {
        const int n = 1 + i % moe::kMaxColumns;
        auto e      = fixtures::random_expert(rng, i % 3 == 0);
        auto x      = fixtures::random_activations(rng, n);
        cases.push_back({std::move(e), std::move(x), n, i % 4 == 1});
    }
    const auto y = gpu_experts(cases);
    long mismatches = 0;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto ref = cpu_expert(cases[i].expert, cases[i].x, cases[i].ncols);
        for (std::size_t k = 0; k < ref.size(); ++k) { mismatches += ref[k] != y[i][k]; }
    }
    std::printf("GPU vs CPU: %zu jobs in one launch, %ld mismatching outputs\n", cases.size(), mismatches);
    check(mismatches == 0, "GPU narrow route equals the CPU engine bit for bit");
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

int main() {
    if (infernix::test::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        test_scalar_functions();
        test_e2m1_decode();
        test_golden();
        test_cpu_equality();
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
