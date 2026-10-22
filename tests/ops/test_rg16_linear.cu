// rg16_linear (include/infernix/ops/rg16_linear.h): the canonical W4A16 projection of
// Qwen3.5-122B-A10B's nvfp4_rg16_v1 head [248320, 3072], FP32 and BF16 outputs, on the dp4a GEMV
// (1-8 columns) and the integer tensor-core route (9 and more).
//
// Oracle: the exact int64 row sums of the CPU engine (offloaded_moe::rg16_row_sums_a16, itself checked
// against direct integer dot products in test_offloaded_moe_a16_cpu) and canon::a16_row_output_f32,
// bit for bit, for every row and column; the BF16 output is that value's one rounding. Since the sums
// are exact, the check also proves each column independent of the call's width and route. A column
// with an Inf gives NaN rows.
//
// linear_topk over the same head (its NVFP4 profile): every column's sixteen best rows below
// valid_rows (248077) of that exact oracle, in descending score with ties to the lower row, ids
// exactly and scores bit for bit, at the widths of a DFlash2 proposal (16 columns per request) and
// beyond the 128-column chunk.

#include "infernix/ops/linear_topk.h"
#include "infernix/ops/rg16_linear.h"
#include "ops/common/canonical_math.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace canon    = infernix::ops::canon;
namespace moe      = infernix::ops::offloaded_moe;
namespace fixtures = infernix::test::offloaded_moe;
using infernix::DType;
using infernix::QType;
using infernix::QuantLayout;
using infernix::Tensor;
using infernix::Weight;
using infernix::test::cuda_check;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

constexpr int kRows = 248320, kK = 3072, kBlocks = kK / 16, kGroups = kRows / 16;
constexpr std::size_t kBytes = static_cast<std::size_t>(kGroups) * kBlocks * moe::kUnitBytes;

std::uint32_t f32_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// Random units: codes uniform, scale bytes realistic plus extremes (zero, subnormal, 448).
std::vector<std::uint8_t> random_matrix(std::mt19937& rng) {
    std::vector<std::uint8_t> m(kBytes);
    std::uniform_int_distribution<int> byte(0, 255), scale(40, 80), rare(0, 99);
    for (auto& b : m) { b = static_cast<std::uint8_t>(byte(rng)); }
    for (std::size_t u = 0; u < static_cast<std::size_t>(kGroups) * kBlocks; ++u) {
        std::uint8_t* s = &m[u * moe::kUnitBytes + 128];
        for (int r = 0; r < 16; ++r) {
            const int pick = rare(rng);
            s[r] = static_cast<std::uint8_t>(pick == 0 ? 0 : pick == 1 ? 3 : pick == 2 ? 0x7E : scale(rng));
        }
    }
    return m;
}

// The oracle's FP32 outputs [kRows][T] (column-major, as the op writes them).
std::vector<float> oracle(const std::vector<std::uint8_t>& matrix, const std::vector<std::uint16_t>& x, int columns,
                          float multiplier) {
    std::vector<float> y(static_cast<std::size_t>(kRows) * columns);
    std::vector<canon::A16Block> blocks(static_cast<std::size_t>(columns) * kBlocks);
    std::vector<int> emax(static_cast<std::size_t>(columns));
    std::vector<const canon::A16Block*> acts(static_cast<std::size_t>(columns));
    for (int c = 0; c < columns; ++c) {
        emax[static_cast<std::size_t>(c)] = moe::encode_a16(&x[static_cast<std::size_t>(c) * kK], kK, &blocks[static_cast<std::size_t>(c) * kBlocks]);
        acts[static_cast<std::size_t>(c)] = &blocks[static_cast<std::size_t>(c) * kBlocks];
    }
    std::atomic<int> next{0};
    const auto work = [&] {
        std::vector<std::int64_t> s(16 * moe::kMaxColumns);
        for (int g = next++; g < kGroups; g = next++) {
            for (int c0 = 0; c0 < columns; c0 += moe::kMaxColumns) {
                const int n = std::min(moe::kMaxColumns, columns - c0);
                moe::rg16_row_sums_a16(moe::best_cpu_isa(), matrix.data(), kBlocks, g, g + 1, &acts[static_cast<std::size_t>(c0)], n,
                                       s.data());
                for (int r = 0; r < 16; ++r) {
                    for (int c = 0; c < n; ++c) {
                        y[static_cast<std::size_t>(c0 + c) * kRows + static_cast<std::size_t>(g) * 16 + r] =
                            canon::a16_row_output_f32(s[static_cast<std::size_t>(r) * n + c], emax[static_cast<std::size_t>(c0 + c)],
                                                      multiplier);
                    }
                }
            }
        }
    };
    std::vector<std::thread> threads;
    for (unsigned w = 0; w < std::min(24U, std::max(1U, std::thread::hardware_concurrency())); ++w) { threads.emplace_back(work); }
    for (auto& t : threads) { t.join(); }
    return y;
}

void run(const std::uint8_t* d_matrix, const std::vector<std::uint8_t>& matrix, float multiplier, int columns,
         bool with_inf, std::mt19937& rng) {
    auto x = fixtures::wide_range_activations(rng, columns, kK);
    if (with_inf) { x[static_cast<std::size_t>(columns - 1) * kK + 77] = 0x7F80; } // last column: +Inf
    const auto expected = oracle(matrix, x, columns, multiplier);
    std::uint16_t* d_x = nullptr;
    cuda_check(cudaMalloc(&d_x, x.size() * sizeof(std::uint16_t)), "cudaMalloc");
    cuda_check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice), "cudaMemcpy");
    float* d_f32 = nullptr;
    std::uint16_t* d_bf16 = nullptr;
    cuda_check(cudaMalloc(&d_f32, expected.size() * sizeof(float)), "cudaMalloc");
    cuda_check(cudaMalloc(&d_bf16, expected.size() * sizeof(std::uint16_t)), "cudaMalloc");
    void* d_workspace = nullptr;
    const std::size_t ws = infernix::ops::rg16_linear_workspace_bytes(kRows, kK, columns);
    cuda_check(cudaMalloc(&d_workspace, ws), "cudaMalloc");
    infernix::WorkspaceArena arena(infernix::DeviceSpan{d_workspace, ws});

    Weight w{};
    w.qtype             = QType::NVFP4_MUL;
    w.layout            = QuantLayout::Rg16;
    w.n                 = kRows;
    w.k                 = kK;
    w.qdata             = d_matrix;
    w.payload           = d_matrix;
    w.payload_bytes     = kBytes;
    w.weight_multiplier = multiplier;
    Tensor tx(d_x, DType::BF16, {kK, columns});
    Tensor out32(d_f32, DType::FP32, {kRows, columns});
    Tensor out16(d_bf16, DType::BF16, {kRows, columns});
    cuda_check(cudaMemset(d_f32, 0xFF, expected.size() * sizeof(float)), "cudaMemset");
    cuda_check(cudaMemset(d_bf16, 0xFF, expected.size() * sizeof(std::uint16_t)), "cudaMemset");
    infernix::ops::rg16_linear(tx, w, out32, arena, nullptr);
    infernix::ops::rg16_linear(tx, w, out16, arena, nullptr);
    cuda_check(cudaDeviceSynchronize(), "rg16_linear");
    std::vector<float> got32(expected.size());
    std::vector<std::uint16_t> got16(expected.size());
    cuda_check(cudaMemcpy(got32.data(), d_f32, got32.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    cuda_check(cudaMemcpy(got16.data(), d_bf16, got16.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
    std::size_t bad32 = 0, bad16 = 0, nan_rows = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        bad32 += f32_bits(got32[i]) != f32_bits(expected[i]);
        const std::uint16_t want16 = std::isnan(expected[i]) ? 0x7FC0 : canon::f32_to_bf16_rn(expected[i]);
        bad16 += got16[i] != want16;
        nan_rows += std::isnan(got32[i]) ? 1 : 0;
    }
    const std::string where = "T " + std::to_string(columns) + (with_inf ? " (Inf column)" : "");
    std::printf("%s: %zu FP32 and %zu BF16 outputs differ of %zu (workspace %zu B)\n", where.c_str(), bad32, bad16,
                expected.size(), ws);
    check(bad32 == 0, where + ": FP32 outputs equal the exact oracle");
    check(bad16 == 0, where + ": BF16 outputs are the oracle's one rounding");
    check(!with_inf || nan_rows == static_cast<std::size_t>(kRows), where + ": exactly the Inf column is NaN");
    cudaFree(d_workspace);
    cudaFree(d_bf16);
    cudaFree(d_f32);
    cudaFree(d_x);
}

// linear_topk's NVFP4 profile on the same head and the same exact outputs.
void run_topk(const std::uint8_t* d_matrix, const std::vector<std::uint8_t>& matrix, float multiplier, int columns,
              std::mt19937& rng) {
    constexpr int kTop = 16, kValid = 248077;
    const auto x        = fixtures::wide_range_activations(rng, columns, kK);
    const auto expected = oracle(matrix, x, columns, multiplier);
    std::uint16_t* d_x  = nullptr;
    std::int32_t* d_ids = nullptr;
    float* d_scores     = nullptr;
    cuda_check(cudaMalloc(&d_x, x.size() * sizeof(std::uint16_t)), "cudaMalloc");
    cuda_check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice), "cudaMemcpy");
    cuda_check(cudaMalloc(&d_ids, static_cast<std::size_t>(kTop) * columns * sizeof(std::int32_t)), "cudaMalloc");
    cuda_check(cudaMalloc(&d_scores, static_cast<std::size_t>(kTop) * columns * sizeof(float)), "cudaMalloc");
    const std::size_t ws = infernix::ops::linear_topk_workspace_capacity_bytes(QType::NVFP4_MUL, kRows, kK, columns, columns);
    void* d_workspace    = nullptr;
    cuda_check(cudaMalloc(&d_workspace, ws), "cudaMalloc");
    infernix::WorkspaceArena arena(infernix::DeviceSpan{d_workspace, ws});
    Weight w{};
    w.qtype             = QType::NVFP4_MUL;
    w.layout            = QuantLayout::Rg16;
    w.n                 = kRows;
    w.k                 = kK;
    w.qdata             = d_matrix;
    w.payload           = d_matrix;
    w.payload_bytes     = kBytes;
    w.weight_multiplier = multiplier;
    Tensor tx(d_x, DType::BF16, {kK, columns});
    Tensor ids(d_ids, DType::I32, {kTop, columns});
    Tensor scores(d_scores, DType::FP32, {kTop, columns});
    infernix::ops::linear_topk(tx, w, kValid, ids, scores, arena, nullptr);
    cuda_check(cudaDeviceSynchronize(), "linear_topk");
    std::vector<std::int32_t> got_ids(static_cast<std::size_t>(kTop) * columns);
    std::vector<float> got_scores(got_ids.size());
    cuda_check(cudaMemcpy(got_ids.data(), d_ids, got_ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
    cuda_check(cudaMemcpy(got_scores.data(), d_scores, got_scores.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy");
    std::size_t bad = 0;
    std::vector<int> order(kValid);
    for (int c = 0; c < columns; ++c) {
        const float* y = &expected[static_cast<std::size_t>(c) * kRows];
        for (int r = 0; r < kValid; ++r) { order[static_cast<std::size_t>(r)] = r; }
        std::partial_sort(order.begin(), order.begin() + kTop, order.end(), [&](int a, int b) {
            return y[a] != y[b] ? y[a] > y[b] : a < b;
        });
        for (int k = 0; k < kTop; ++k) {
            const std::size_t at = static_cast<std::size_t>(c) * kTop + k;
            bad += got_ids[at] != order[static_cast<std::size_t>(k)] ||
                   f32_bits(got_scores[at]) != f32_bits(y[order[static_cast<std::size_t>(k)]]);
        }
    }
    const std::string where = "linear_topk T " + std::to_string(columns);
    std::printf("%s: %zu of %zu candidates differ (workspace %zu B)\n", where.c_str(), bad, got_ids.size(), ws);
    check(bad == 0, where + ": the sixteen best rows of the exact outputs, in order");
    cudaFree(d_workspace);
    cudaFree(d_scores);
    cudaFree(d_ids);
    cudaFree(d_x);
}

} // namespace

int main() {
    std::mt19937 rng(20261022);
    const auto matrix = random_matrix(rng);
    std::uint8_t* d_matrix = nullptr;
    cuda_check(cudaMalloc(&d_matrix, kBytes), "cudaMalloc");
    cuda_check(cudaMemcpy(d_matrix, matrix.data(), kBytes, cudaMemcpyHostToDevice), "cudaMemcpy");
    const float multiplier = 7.01e-5F; // the order of Ornith's lm_head weight_scale_2 multiplier
    // The GEMV's column templates (1, 2, 4, 8 at 1, 2, 3, 8), the wide route at one partial tile (9),
    // two tiles (100) and a non-finite column.
    for (const int columns : {1, 2, 3, 8, 9, 100}) { run(d_matrix, matrix, multiplier, columns, false, rng); }
    run(d_matrix, matrix, multiplier, 4, true, rng);
    run(d_matrix, matrix, multiplier, 70, true, rng);
    for (const int columns : {1, 16, 100, 130}) { run_topk(d_matrix, matrix, multiplier, columns, rng); }
    cudaFree(d_matrix);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("rg16_linear: all checks passed\n");
    return 0;
}
