// The canonical functions Inkling 2's routed experts and router add to ops/common/canonical_math.h (design
// docs/maintainer/inkling2_mini-design.md §2.5): expm1_c, tanh_c, sigmoid_c, softcap_c (caps 4 and 25) and the
// four-over-six NVFP4 activation rule.
//
// Oracles:
//   - every unary function on the GPU against the same header on the CPU, bit for bit, exhaustively over the 65,536
//     BF16 inputs (the experts' gate and up values are BF16), and against FP64 (std::expm1, std::tanh, 1 / (1 + e^-x)):
//     at most 2 ulp of the binary32 result (log_c over the positive inputs); inputs below 2^-120 (where x / cap is subnormal) to an absolute 2^-140;
//     results beyond binary32's range are the infinity;
//   - four_over_six_block on the GPU against the CPU, bit for bit, and both against the independent reference
//     (tests/fixtures/inkling2/four_over_six.json, from the CPU reference's torch implementation, written by
//     tests/models/inkling2/generate_four_over_six_fixture.py; its path is the first argument): every scale word and code.

#include "ops/common/canonical_math.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
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

constexpr int kFunctions = 6; // expm1, tanh, sigmoid, softcap 4, softcap 25, log

__device__ __host__ float apply(int f, float x) {
    switch (f) {
    case 0: return canon::expm1_c(x);
    case 1: return canon::tanh_c(x);
    case 2: return canon::sigmoid_c(x);
    case 3: return canon::softcap_c(x, 4.0F);
    case 4: return canon::softcap_c(x, 25.0F);
    default: return canon::log_c(x);
    }
}

__global__ void unary_kernel(std::uint32_t* out) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= 65536) { return; }
    const float x = canon::bf16_to_f32(static_cast<std::uint16_t>(i));
    for (int f = 0; f < kFunctions; ++f) { out[f * 65536 + i] = canon::f32_bits(apply(f, x)); }
}

double exact(int f, double x) {
    switch (f) {
    case 0: return std::expm1(x);
    case 1: return std::tanh(x);
    case 2: return 1.0 / (1.0 + std::exp(-x));
    case 3: return 4.0 * std::tanh(x / 4.0);
    case 4: return 25.0 * std::tanh(x / 25.0);
    default: return std::log(x);
    }
}

// The distance in binary32 ulps of `value` from `reference` (the ulp of the reference's binade).
double ulps(float value, double reference) {
    if (reference == 0.0) { return value == 0.0F ? 0.0 : 1e30; }
    int exponent = 0;
    (void)std::frexp(reference, &exponent);
    const double ulp = std::ldexp(1.0, std::max(exponent - 24, -149));
    return std::fabs(static_cast<double>(value) - reference) / ulp;
}

void unary_functions() {
    std::uint32_t* device = nullptr;
    cuda_check(cudaMalloc(&device, kFunctions * 65536 * sizeof(std::uint32_t)), "cudaMalloc");
    unary_kernel<<<256, 256>>>(device);
    cuda_check(cudaGetLastError(), "unary kernel");
    std::vector<std::uint32_t> gpu(kFunctions * 65536);
    cuda_check(cudaMemcpy(gpu.data(), device, gpu.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "copy");
    cuda_check(cudaFree(device), "cudaFree");
    const char* names[kFunctions] = {"expm1_c", "tanh_c", "sigmoid_c", "softcap_c(4)", "softcap_c(25)", "log_c"};
    for (int f = 0; f < kFunctions; ++f) {
        int mismatches  = 0;
        double worst    = 0.0;
        float worst_at  = 0.0F;
        for (int i = 0; i < 65536; ++i) {
            const float x = canon::bf16_to_f32(static_cast<std::uint16_t>(i));
            const float y = apply(f, x);
            if (canon::f32_bits(y) != gpu[f * 65536 + i] && ++mismatches <= 3) {
                std::printf("  %s(%.9g = 0x%04x): CPU 0x%08x, GPU 0x%08x\n", names[f], x, i, canon::f32_bits(y),
                            gpu[f * 65536 + i]);
            }
            if (!std::isfinite(x) || (f == 5 && x <= 0.0F)) { continue; }
            const double reference = exact(f, static_cast<double>(x));
            if (std::fabs(reference) > 3.4028234663852886e38) { // beyond binary32: the rounded result is the infinity
                check(std::isinf(y) && (y > 0) == (reference > 0), "an overflowing result is the infinity");
                continue;
            }
            if (!std::isfinite(reference)) {
                check(std::isinf(y) && (y > 0) == (reference > 0), "an overflowing result is the infinity");
                continue;
            }
            if (f != 5 && std::fabs(x) < 0x1p-120F) {
                // BF16 subnormals and the least normals: x / cap is subnormal in binary32, whose fixed 2^-149 spacing
                // bounds every form's accuracy; the bound here is absolute.
                check(std::fabs(static_cast<double>(y) - reference) <= 0x1p-140, "a tiny input's result is within 2^-140");
                continue;
            }
            const double distance = ulps(y, reference);
            if (distance > worst) {
                worst    = distance;
                worst_at = x;
            }
        }
        std::printf("%-14s GPU/CPU mismatches %d, worst %.3f ulp at %.9g\n", names[f], mismatches, worst, worst_at);
        check(mismatches == 0, "a canonical function's GPU bits differ from its CPU bits");
        check(worst <= 2.0, "a canonical function is more than 2 ulp from FP64");
    }
}

// ---------------------------------------------------------------------------- four-over-six

struct Case {
    std::vector<float> row; // K values (BF16 values as binary32), K a multiple of 16
    std::vector<int> scale_words;
    std::vector<int> codes;
};

__global__ void four_over_six_kernel(const float* rows, const int* offsets, int count, std::uint8_t* words,
                                     std::uint8_t* codes) {
    const int r = static_cast<int>(blockIdx.x);
    if (r >= count) { return; }
    const float* row = rows + offsets[r];
    const int k      = offsets[r + 1] - offsets[r];
    float amax       = 0.0F;
    for (int j = 0; j < k; ++j) { amax = fmaxf(amax, fabsf(row[j])); }
    const canon::FourOverSixRow scale = canon::four_over_six_row(amax);
    for (int b = static_cast<int>(threadIdx.x); b < k / 16; b += static_cast<int>(blockDim.x)) {
        float x[16];
        for (int j = 0; j < 16; ++j) { x[j] = row[16 * b + j]; }
        const canon::FourOverSixBlock block = canon::four_over_six_block(x, scale);
        words[offsets[r] / 16 + b] = block.scale_word;
        for (int j = 0; j < 16; ++j) { codes[offsets[r] + 16 * b + j] = block.code[j]; }
    }
}

void four_over_six(const char* path) {
    std::ifstream file(path, std::ios::binary);
    check(static_cast<bool>(file), "the four-over-six fixture opens");
    if (!file) { return; }
    std::ostringstream text;
    text << file.rdbuf();
    const auto fixture = nlohmann::json::parse(text.str());
    std::vector<Case> cases;
    std::vector<float> rows;
    std::vector<int> offsets{0};
    for (const auto& item : fixture.at("cases")) {
        Case c;
        c.row         = item.at("row").get<std::vector<float>>();
        c.scale_words = item.at("scale_words").get<std::vector<int>>();
        c.codes       = item.at("codes").get<std::vector<int>>();
        rows.insert(rows.end(), c.row.begin(), c.row.end());
        offsets.push_back(static_cast<int>(rows.size()));
        cases.push_back(std::move(c));
    }
    float* d_rows = nullptr;
    int* d_offsets = nullptr;
    std::uint8_t *d_words = nullptr, *d_codes = nullptr;
    cuda_check(cudaMalloc(&d_rows, rows.size() * sizeof(float)), "cudaMalloc");
    cuda_check(cudaMalloc(&d_offsets, offsets.size() * sizeof(int)), "cudaMalloc");
    cuda_check(cudaMalloc(&d_words, rows.size() / 16), "cudaMalloc");
    cuda_check(cudaMalloc(&d_codes, rows.size()), "cudaMalloc");
    cuda_check(cudaMemcpy(d_rows, rows.data(), rows.size() * sizeof(float), cudaMemcpyHostToDevice), "copy");
    cuda_check(cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice), "copy");
    four_over_six_kernel<<<static_cast<unsigned>(cases.size()), 64>>>(d_rows, d_offsets, static_cast<int>(cases.size()),
                                                                      d_words, d_codes);
    cuda_check(cudaGetLastError(), "four-over-six kernel");
    std::vector<std::uint8_t> words(rows.size() / 16), codes(rows.size());
    cuda_check(cudaMemcpy(words.data(), d_words, words.size(), cudaMemcpyDeviceToHost), "copy");
    cuda_check(cudaMemcpy(codes.data(), d_codes, codes.size(), cudaMemcpyDeviceToHost), "copy");
    for (void* p : {static_cast<void*>(d_rows), static_cast<void*>(d_offsets), static_cast<void*>(d_words),
                    static_cast<void*>(d_codes)}) {
        cuda_check(cudaFree(p), "cudaFree");
    }
    int differ = 0, gpu_differ = 0, fours = 0, blocks = 0;
    for (std::size_t r = 0; r < cases.size(); ++r) {
        const Case& c = cases[r];
        float amax    = 0.0F;
        for (const float v : c.row) { amax = std::max(amax, std::fabs(v)); }
        const canon::FourOverSixRow scale = canon::four_over_six_row(amax);
        for (std::size_t b = 0; b < c.row.size() / 16; ++b) {
            float x[16];
            for (int j = 0; j < 16; ++j) { x[j] = c.row[16 * b + j]; }
            const canon::FourOverSixBlock block = canon::four_over_six_block(x, scale);
            const std::size_t at = static_cast<std::size_t>(offsets[r]) + 16 * b;
            bool same = block.scale_word == c.scale_words[b];
            bool gpu  = block.scale_word == words[at / 16];
            for (int j = 0; j < 16; ++j) {
                same = same && block.code[j] == c.codes[16 * b + j];
                gpu  = gpu && block.code[j] == codes[at + j];
            }
            differ += same ? 0 : 1;
            gpu_differ += gpu ? 0 : 1;
            ++blocks;
            // Which candidate won: the reference's choice is recorded per block.
            fours += fixture.at("cases")[r].at("four")[b].get<bool>() ? 1 : 0;
        }
    }
    std::printf("four-over-six: %d blocks, %d took the 4 scale, %d differ from the reference, %d GPU/CPU differ\n",
                blocks, fours, differ, gpu_differ);
    check(differ == 0, "four-over-six blocks differ from the independent reference");
    check(gpu_differ == 0, "four-over-six blocks differ between the GPU and the CPU");
    check(fours > 0 && fours < blocks, "the fixture exercises both candidates");
}

} // namespace

INFERNIX_OP_TEST(inkling2_canon) {
    unary_functions();
    if (argc < 2) {
        std::fprintf(stderr, "usage: inkling2_canon <tests/fixtures/inkling2/four_over_six.json>\n");
        return 2;
    }
    four_over_six(argv[1]);
    std::printf(g_failures == 0 ? "PASS\n" : "FAILED (%d)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
