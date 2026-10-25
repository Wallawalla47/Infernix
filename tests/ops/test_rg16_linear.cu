// rg16_linear (include/infernix/ops/rg16_linear.h): the canonical W4A16 projection of an
// nvfp4_rg16_v1 matrix at every registered shape, FP32 and BF16 outputs: Qwen3.5-122B-A10B's head
// [248320, 3072] and Qwen3.8-Flash-Next FP4-Local's vision MLP ([4304, 1152]: rows not whole
// 128-row blocks; [1152, 4304]: K not whole 128-element stages) and merger ([4608, 4608],
// [2560, 4608]), on the dp4a GEMV (1-8 columns; 1-4 at K 4304 and 4608) and the integer tensor-core
// route (beyond, in passes of 4,096 columns).
//
// Exact oracle, every shape: the exact int64 row sums of the CPU engine (offloaded_moe::rg16_row_sums_a16,
// itself checked against direct integer dot products in test_offloaded_moe_a16_cpu) and
// canon::a16_row_output_f32, bit for bit, for every row and column; the BF16 output is that value's
// one rounding. Since the sums are exact, the check also proves each column independent of the
// call's width, route and pass. A column with an Inf gives NaN rows. The widths cover every route
// boundary: the GEMV's column templates, the GEMV's last width and the wide route's first, a partial,
// a whole and a second tile, an interior width, and the pass boundary (4,095 / 4,096 / 4,097).
//
// FP64 oracle, the vision shapes: the weights decoded independently from the units (E2M1 code times
// E4M3 scale times the FP32 multiplier, in binary64) times the BF16 activations, summed naively in
// binary64. The BF16 outputs meet the canonical W4A16 criterion: normwise, ||y - y_ref|| <=
// 2^-8 ||y_ref||; pointwise (gross cap), |y - y_ref| <= 2^-8 |y_ref| + 2^-22 (|y_ref| + a) + 1.01 a +
// 2^-40 sum_k |W x| + 2^-126, with a = 2^(emax - 148) sum_k |W[n, k]| bounding the A16 encoding's
// rounding of elements more than 13 binades below their column's largest (design §16.2.1).
// Non-finite columns are excluded (the exact check covers them).
//
// linear_topk over the head (its NVFP4 profile): every column's sixteen best rows below
// valid_rows (248077) of the exact oracle, in descending score with ties to the lower row, ids
// exactly and scores bit for bit, at the widths of a DFlash2 proposal (16 columns per request) and
// beyond the 128-column chunk.

#include "ops/op_family.h"
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
#include <functional>
#include <random>
#include <stdexcept>
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

std::uint32_t f32_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// A registered matrix [rows, k] and its units on the host and the device.
struct Matrix {
    const char* name;
    int rows, k;
    float multiplier;
    std::vector<std::uint8_t> units;
    std::uint8_t* device = nullptr;

    [[nodiscard]] int blocks() const { return k / 16; }
    [[nodiscard]] int groups() const { return rows / 16; }
    [[nodiscard]] std::size_t bytes() const { return static_cast<std::size_t>(groups()) * blocks() * moe::kUnitBytes; }
};

void parallel(int count, const std::function<void(int)>& body) {
    std::atomic<int> next{0};
    std::vector<std::thread> threads;
    const unsigned workers = std::min(24U, std::max(1U, std::thread::hardware_concurrency()));
    for (unsigned w = 0; w < workers; ++w) {
        threads.emplace_back([&] {
            for (int i = next++; i < count; i = next++) { body(i); }
        });
    }
    for (auto& t : threads) { t.join(); }
}

// Random units: codes uniform, scale bytes realistic plus extremes (zero, subnormal, 448).
Matrix make_matrix(const char* name, int rows, int k, float multiplier, std::mt19937& rng) {
    Matrix m{name, rows, k, multiplier, {}, nullptr};
    m.units.resize(m.bytes());
    std::uniform_int_distribution<int> byte(0, 255), scale(40, 80), rare(0, 99);
    for (auto& b : m.units) { b = static_cast<std::uint8_t>(byte(rng)); }
    for (std::size_t u = 0; u < static_cast<std::size_t>(m.groups()) * m.blocks(); ++u) {
        std::uint8_t* s = &m.units[u * moe::kUnitBytes + 128];
        for (int r = 0; r < 16; ++r) {
            const int pick = rare(rng);
            s[r] = static_cast<std::uint8_t>(pick == 0 ? 0 : pick == 1 ? 3 : pick == 2 ? 0x7E : scale(rng));
        }
    }
    cuda_check(cudaMalloc(&m.device, m.bytes()), "cudaMalloc");
    cuda_check(cudaMemcpy(m.device, m.units.data(), m.bytes(), cudaMemcpyHostToDevice), "cudaMemcpy");
    return m;
}

Weight weight_of(const Matrix& m) {
    Weight w{};
    w.qtype             = QType::NVFP4_MUL;
    w.layout            = QuantLayout::Rg16;
    w.n                 = m.rows;
    w.k                 = m.k;
    w.qdata             = m.device;
    w.payload           = m.device;
    w.payload_bytes     = m.bytes();
    w.weight_multiplier = m.multiplier;
    return w;
}

// The exact oracle's FP32 outputs [rows][T] (column-major, as the op writes them), and each column's
// exponent.
std::vector<float> exact_oracle(const Matrix& m, const std::vector<std::uint16_t>& x, int columns,
                                std::vector<int>* exponents = nullptr) {
    std::vector<float> y(static_cast<std::size_t>(m.rows) * columns);
    std::vector<canon::A16Block> blocks(static_cast<std::size_t>(columns) * m.blocks());
    std::vector<int> emax(static_cast<std::size_t>(columns));
    std::vector<const canon::A16Block*> acts(static_cast<std::size_t>(columns));
    for (int c = 0; c < columns; ++c) {
        emax[static_cast<std::size_t>(c)] =
            moe::encode_a16(&x[static_cast<std::size_t>(c) * m.k], m.k, &blocks[static_cast<std::size_t>(c) * m.blocks()]);
        acts[static_cast<std::size_t>(c)] = &blocks[static_cast<std::size_t>(c) * m.blocks()];
    }
    parallel(m.groups(), [&](int g) {
        std::vector<std::int64_t> s(16 * moe::kMaxColumns);
        for (int c0 = 0; c0 < columns; c0 += moe::kMaxColumns) {
            const int n = std::min(moe::kMaxColumns, columns - c0);
            moe::rg16_row_sums_a16(moe::best_cpu_isa(), m.units.data(), m.blocks(), g, g + 1,
                                   &acts[static_cast<std::size_t>(c0)], n, s.data());
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < n; ++c) {
                    y[static_cast<std::size_t>(c0 + c) * m.rows + static_cast<std::size_t>(g) * 16 + r] =
                        canon::a16_row_output_f32(s[static_cast<std::size_t>(r) * n + c], emax[static_cast<std::size_t>(c0 + c)],
                                                  m.multiplier);
                }
            }
        }
    });
    if (exponents != nullptr) { *exponents = emax; }
    return y;
}

// ------------------------------------------------------------------ the independent FP64 oracle

double e2m1_value(unsigned code) {
    static constexpr double kMagnitude[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double v = kMagnitude[code & 7U];
    return (code & 8U) != 0 ? -v : v;
}

// E4M3FN: (8 + m) 2^(e - 10) for e > 0, m 2^-9 for e = 0.
double e4m3_value(unsigned word) {
    const unsigned e = (word >> 3) & 15U, m = word & 7U;
    return e == 0 ? std::ldexp(static_cast<double>(m), -9) : std::ldexp(static_cast<double>(8 + m), static_cast<int>(e) - 10);
}

double bf16_value(std::uint16_t h) {
    const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return static_cast<double>(f);
}

// W[n][k] decoded from the units (storage layouts §6: byte 32q + 4j + t of unit (g, b) holds rows
// 16g + j (low nibble) and 16g + 8 + j (high nibble) at k = 16b + 4q + t; bytes 128 + r the scales).
std::vector<double> decode_weights(const Matrix& m) {
    std::vector<double> w(static_cast<std::size_t>(m.rows) * m.k);
    parallel(m.groups(), [&](int g) {
        for (int b = 0; b < m.blocks(); ++b) {
            const std::uint8_t* unit = &m.units[(static_cast<std::size_t>(g) * m.blocks() + b) * moe::kUnitBytes];
            for (int r = 0; r < 16; ++r) {
                const double scale = e4m3_value(unit[128 + r]) * static_cast<double>(m.multiplier);
                for (int j = 0; j < 16; ++j) {
                    const std::uint8_t byte = unit[32 * (j / 4) + 4 * (r & 7) + j % 4];
                    const unsigned code     = r < 8 ? byte & 15U : byte >> 4;
                    w[static_cast<std::size_t>(16 * g + r) * m.k + 16 * b + j] = e2m1_value(code) * scale;
                }
            }
        }
    });
    return w;
}

// The BF16 outputs of one call against the FP64 oracle (finite columns only).
void check_fp64(const Matrix& m, const std::vector<double>& w, const std::vector<std::uint16_t>& x, int columns,
                const std::vector<int>& emax, const std::vector<std::uint16_t>& got, const std::string& where) {
    std::vector<double> error2(static_cast<std::size_t>(m.rows), 0.0), norm2(static_cast<std::size_t>(m.rows), 0.0);
    std::vector<int> gross(static_cast<std::size_t>(m.rows), 0);
    parallel(m.rows, [&](int n) {
        const double* row = &w[static_cast<std::size_t>(n) * m.k];
        double abs_w = 0.0;
        for (int k = 0; k < m.k; ++k) { abs_w += std::fabs(row[k]); }
        for (int c = 0; c < columns; ++c) {
            if (emax[static_cast<std::size_t>(c)] == canon::kA16NonFinite) { continue; }
            const std::uint16_t* xc = &x[static_cast<std::size_t>(c) * m.k];
            double y = 0.0, abs_wx = 0.0;
            for (int k = 0; k < m.k; ++k) {
                const double p = row[k] * bf16_value(xc[k]);
                y += p;
                abs_wx += std::fabs(p);
            }
            const double a     = std::ldexp(abs_w, emax[static_cast<std::size_t>(c)] - 148);
            const double bound = std::ldexp(std::fabs(y), -8) + std::ldexp(std::fabs(y) + a, -22) + 1.01 * a +
                                 std::ldexp(abs_wx, -40) + std::ldexp(1.0, -126);
            const double out = bf16_value(got[static_cast<std::size_t>(c) * m.rows + n]);
            const double e   = out - y;
            error2[static_cast<std::size_t>(n)] += e * e;
            norm2[static_cast<std::size_t>(n)] += y * y;
            gross[static_cast<std::size_t>(n)] += !(std::fabs(e) <= bound);
        }
    });
    double error = 0.0, norm = 0.0;
    long long over = 0;
    for (int n = 0; n < m.rows; ++n) {
        error += error2[static_cast<std::size_t>(n)];
        norm += norm2[static_cast<std::size_t>(n)];
        over += gross[static_cast<std::size_t>(n)];
    }
    const double relative = norm > 0.0 ? std::sqrt(error / norm) : std::sqrt(error);
    std::printf("%s: FP64 oracle, normwise relative error %.3e, %lld outputs over the pointwise cap\n", where.c_str(),
                relative, over);
    check(relative <= std::ldexp(1.0, -8), where + ": normwise error within 2^-8 of the FP64 oracle");
    check(over == 0, where + ": every output within the pointwise cap of the FP64 oracle");
}

// ------------------------------------------------------------------ one call against the oracles

void run(const Matrix& m, int columns, bool with_inf, std::mt19937& rng, const std::vector<double>* w = nullptr) {
    auto x = fixtures::wide_range_activations(rng, columns, m.k);
    if (with_inf) { x[static_cast<std::size_t>(columns - 1) * m.k + 77] = 0x7F80; } // last column: +Inf
    std::vector<int> emax;
    const auto expected = exact_oracle(m, x, columns, &emax);
    std::uint16_t* d_x = nullptr;
    cuda_check(cudaMalloc(&d_x, x.size() * sizeof(std::uint16_t)), "cudaMalloc");
    cuda_check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice), "cudaMemcpy");
    float* d_f32 = nullptr;
    std::uint16_t* d_bf16 = nullptr;
    cuda_check(cudaMalloc(&d_f32, expected.size() * sizeof(float)), "cudaMalloc");
    cuda_check(cudaMalloc(&d_bf16, expected.size() * sizeof(std::uint16_t)), "cudaMalloc");
    void* d_workspace = nullptr;
    const std::size_t ws = infernix::ops::rg16_linear_workspace_bytes(m.rows, m.k, columns);
    cuda_check(cudaMalloc(&d_workspace, ws), "cudaMalloc");
    infernix::WorkspaceArena arena(infernix::DeviceSpan{d_workspace, ws});

    const Weight w16 = weight_of(m);
    Tensor tx(d_x, DType::BF16, {m.k, columns});
    Tensor out32(d_f32, DType::FP32, {m.rows, columns});
    Tensor out16(d_bf16, DType::BF16, {m.rows, columns});
    cuda_check(cudaMemset(d_f32, 0xFF, expected.size() * sizeof(float)), "cudaMemset");
    cuda_check(cudaMemset(d_bf16, 0xFF, expected.size() * sizeof(std::uint16_t)), "cudaMemset");
    infernix::ops::rg16_linear(tx, w16, out32, arena, nullptr);
    infernix::ops::rg16_linear(tx, w16, out16, arena, nullptr);
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
    const std::string where = std::string(m.name) + " T " + std::to_string(columns) + (with_inf ? " (Inf column)" : "");
    std::printf("%s: %zu FP32 and %zu BF16 outputs differ of %zu (workspace %zu B)\n", where.c_str(), bad32, bad16,
                expected.size(), ws);
    check(bad32 == 0, where + ": FP32 outputs equal the exact oracle");
    check(bad16 == 0, where + ": BF16 outputs are the oracle's one rounding");
    check(!with_inf || nan_rows == static_cast<std::size_t>(m.rows), where + ": exactly the Inf column is NaN");
    if (w != nullptr) { check_fp64(m, *w, x, columns, emax, got16, where); }
    // The narrow widths' calls repeat bit for bit, also with a copy streaming beside them (the GEMV
    // and the streamed GEMV; op_tester.h verify_repeatable).
    if (columns <= 8) {
        check(infernix::test::verify_repeatable(
                  where + " repeated", 3, nullptr, [&] { infernix::ops::rg16_linear(tx, w16, out16, arena, nullptr); },
                  {{d_bf16, expected.size() * sizeof(std::uint16_t)}}) == 0,
              where + ": repeated calls give the same bits");
    }
    cudaFree(d_workspace);
    cudaFree(d_bf16);
    cudaFree(d_f32);
    cudaFree(d_x);
}

// linear_topk's NVFP4 profile on the head and the same exact outputs.
void run_topk(const Matrix& head, int columns, std::mt19937& rng) {
    constexpr int kTop = 16, kValid = 248077;
    const auto x        = fixtures::wide_range_activations(rng, columns, head.k);
    const auto expected = exact_oracle(head, x, columns);
    std::uint16_t* d_x  = nullptr;
    std::int32_t* d_ids = nullptr;
    float* d_scores     = nullptr;
    cuda_check(cudaMalloc(&d_x, x.size() * sizeof(std::uint16_t)), "cudaMalloc");
    cuda_check(cudaMemcpy(d_x, x.data(), x.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice), "cudaMemcpy");
    cuda_check(cudaMalloc(&d_ids, static_cast<std::size_t>(kTop) * columns * sizeof(std::int32_t)), "cudaMalloc");
    cuda_check(cudaMalloc(&d_scores, static_cast<std::size_t>(kTop) * columns * sizeof(float)), "cudaMalloc");
    const std::size_t ws =
        infernix::ops::linear_topk_workspace_capacity_bytes(QType::NVFP4_MUL, head.rows, head.k, columns, columns);
    void* d_workspace = nullptr;
    cuda_check(cudaMalloc(&d_workspace, ws), "cudaMalloc");
    infernix::WorkspaceArena arena(infernix::DeviceSpan{d_workspace, ws});
    const Weight w = weight_of(head);
    Tensor tx(d_x, DType::BF16, {head.k, columns});
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
        const float* y = &expected[static_cast<std::size_t>(c) * head.rows];
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

// An unregistered [N, K] is refused by the workspace query and the projection.
void check_unregistered() {
    bool refused = false;
    try {
        (void)infernix::ops::rg16_linear_workspace_bytes(4304, 1168, 64);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    check(refused, "an unregistered [N, K] is refused");
}

} // namespace

INFERNIX_OP_TEST(rg16_linear) {
    std::mt19937 rng(20261022);
    check_unregistered();
    {
        // The order of Ornith's lm_head weight_scale_2 multiplier.
        Matrix head = make_matrix("head [248320, 3072]", 248320, 3072, 7.01e-5F, rng);
        // The GEMV's column templates (1, 2, 4, 8 at 1, 2, 3, 8), the wide route at one partial tile (9),
        // two tiles (100) and a non-finite column.
        for (const int columns : {1, 2, 3, 8, 9, 100}) { run(head, columns, false, rng); }
        run(head, 4, true, rng);
        run(head, 70, true, rng);
        for (const int columns : {1, 16, 100, 130}) { run_topk(head, columns, rng); }
        cudaFree(head.device);
    }
    // The vision tower's matrices at multipliers of the order of a vision weight_scale_2.
    struct VisionCase {
        const char* name;
        int rows, k;
        float multiplier;
        std::vector<int> widths;       // exact oracle
        std::vector<int> inf_widths;   // exact oracle, last column +Inf
        std::vector<int> fp64_widths;  // exact and FP64 oracles
    };
    const VisionCase cases[] = {
        // Rows past the last whole 128-row block (33 blocks and 5 row groups); the GEMV to 8 columns.
        {"vision fc1 [4304, 1152]", 4304, 1152, 3.1e-4F, {2, 3, 4, 5, 8, 63, 64, 4095, 4096, 4097}, {6, 70},
         {1, 9, 65, 300}},
        // K past the last whole stage (33 stages and 5 units); the GEMV to 4 columns.
        {"vision fc2 [1152, 4304]", 1152, 4304, 2.3e-4F, {2, 3, 4, 8, 9, 63, 64, 4095, 4096, 4097}, {3, 70},
         {1, 5, 65, 300}},
        {"merger fc1 [4608, 4608]", 4608, 4608, 1.7e-4F, {1, 3, 64, 300}, {4, 66}, {4, 5, 65}},
        {"merger fc2 [2560, 4608]", 2560, 4608, 5.3e-4F, {1, 4, 64, 129}, {2, 70}, {5, 300}},
    };
    for (const VisionCase& c : cases) {
        Matrix m = make_matrix(c.name, c.rows, c.k, c.multiplier, rng);
        const std::vector<double> w = decode_weights(m);
        for (const int columns : c.widths) { run(m, columns, false, rng); }
        for (const int columns : c.inf_widths) { run(m, columns, true, rng); }
        for (const int columns : c.fp64_widths) { run(m, columns, false, rng, &w); }
        cudaFree(m.device);
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("rg16_linear: all checks passed\n");
    return 0;
}
