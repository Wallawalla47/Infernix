// Inkling 2's LoRA term (include/infernix/ops/inkling2_lora.h, --lora) against an FP64 oracle over the represented BF16
// factors, U and y: y' = y + s sum_j B[n, j] U[j, t], one BF16 rounding (BF16 outputs) or none (FP32 outputs).
//
// Criterion (pointwise, finite inputs): |y' - ref| <= rounding(ref + d) + d, with d = (rank + 2) 2^-23 |s| sum_j |B U| +
// 2^-22 |y|: the FP32 term chain over `rank` products, its product with s and the FP32 add; rounding(v) = 2^-8 |v| for a
// BF16 output, 2^-23 |v| for an FP32 one.
//
// Also: B = 0 leaves y's bits; rows past a target's `rows` and columns of other targets are untouched; every column's
// bits equal the same column computed alone (one column, any position); a repeated call gives the same bits. Shapes are
// the model's: K2's outputs (the query's 3072 rows, the sink's 24: a partial 16-row group, the taps' 32), W_o's 2304, a
// gate's down (64) and its forget and message rows, FP32 in one column of 4608, at ranks 16, 64 and 128, at the narrow
// (T <= 8) and wide tiles; W_o, the gate and the sink also at the other padded ranks (each k-step count is its own
// compiled instance).

#include "infernix/ops/inkling2_lora.h"
#include "ops/common/canonical_math.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace canon = infernix::ops::canon;
namespace ik    = infernix::ops::inkling2;
using infernix::test::cuda_check;

namespace {

int g_failures = 0;

void fail(const char* what, int rank, int columns, int target, int n, int t) {
    if (g_failures++ < 20) {
        std::fprintf(stderr, "FAIL: %s (rank %d, T %d, target %d, row %d, column %d)\n", what, rank, columns, target, n, t);
    }
}

template <class T>
T* dalloc(std::size_t n) {
    T* p = nullptr;
    cuda_check(cudaMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    return p;
}

template <class T>
void up(T* p, const std::vector<T>& v) {
    cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}

template <class T>
std::vector<T> down(const T* p, std::size_t n) {
    std::vector<T> v(n);
    cuda_check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}

std::vector<std::uint16_t> random_bf16(std::size_t n, float sd, std::mt19937& gen) {
    std::normal_distribution<float> d(0.0F, sd);
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) { x = canon::f32_to_bf16_rn(d(gen)); }
    return v;
}

double bf16_value(std::uint16_t v) { return static_cast<double>(canon::bf16_to_f32(v)); }

struct Shape {
    const char* name;
    int rows;
    int stride;   // y's elements a column
    int first;    // the target's first row within the column (a gate's message rows start at H)
    bool fp32;
};

// One target's operands on the host and the device.
struct Target {
    Shape shape{};
    int rank    = 0;
    float scale = 0;
    std::vector<std::uint16_t> b, u, y16;
    std::vector<float> y32;
    std::uint16_t *db = nullptr, *du = nullptr, *dy16 = nullptr;
    float* dy32 = nullptr;
};

Target make_target(const Shape& s, int rank, int T, std::mt19937& gen, bool zero_b) {
    Target t;
    t.shape = s;
    t.rank  = rank;
    t.scale = std::uniform_real_distribution<float>(0.25F, 4.0F)(gen);
    const std::size_t padded = static_cast<std::size_t>((s.rows + 15) / 16 * 16);
    t.b = zero_b ? std::vector<std::uint16_t>(padded * rank, 0) : random_bf16(padded * rank, 0.2F, gen);
    for (std::size_t r = s.rows; r < padded; ++r) { // the padded rows are zero, as the loader stores them
        std::fill_n(t.b.begin() + static_cast<std::ptrdiff_t>(r * rank), rank, std::uint16_t{0});
    }
    t.u  = random_bf16(static_cast<std::size_t>(T) * rank, 1.0F, gen);
    t.db = dalloc<std::uint16_t>(t.b.size());
    t.du = dalloc<std::uint16_t>(t.u.size());
    up(t.db, t.b);
    up(t.du, t.u);
    const std::size_t y_elements = static_cast<std::size_t>(T) * s.stride;
    if (s.fp32) {
        std::normal_distribution<float> d(0.0F, 2.0F);
        t.y32.resize(y_elements);
        for (auto& v : t.y32) { v = d(gen); }
        t.dy32 = dalloc<float>(y_elements);
    } else {
        t.y16  = random_bf16(y_elements, 2.0F, gen);
        t.dy16 = dalloc<std::uint16_t>(y_elements);
    }
    return t;
}

void reset_y(Target& t) { t.shape.fp32 ? up(t.dy32, t.y32) : up(t.dy16, t.y16); }

ik::LoraTarget op_target(const Target& t, int column0) {
    ik::LoraTarget o;
    o.b = t.db;
    o.u = t.du + static_cast<std::size_t>(column0) * t.rank;
    o.y = t.shape.fp32 ? static_cast<void*>(t.dy32 + static_cast<std::size_t>(column0) * t.shape.stride + t.shape.first)
                       : static_cast<void*>(t.dy16 + static_cast<std::size_t>(column0) * t.shape.stride + t.shape.first);
    o.rows     = t.shape.rows;
    o.y_stride = t.shape.stride;
    o.scale    = t.scale;
    return o;
}

void free_target(Target& t) {
    cudaFree(t.db), cudaFree(t.du);
    if (t.dy16 != nullptr) { cudaFree(t.dy16); }
    if (t.dy32 != nullptr) { cudaFree(t.dy32); }
}

// One launch over `shapes` (one output type a launch).
void run_case(const std::vector<Shape>& shapes, int rank, int T, std::uint32_t seed) {
    std::mt19937 gen(seed);
    const bool fp32 = shapes.front().fp32;
    std::vector<Target> targets;
    for (const Shape& s : shapes) { targets.push_back(make_target(s, rank, T, gen, false)); }
    ik::LoraTermArgs a;
    a.columns  = T;
    a.rank     = rank;
    a.fp32_out = fp32;
    for (Target& t : targets) {
        reset_y(t);
        a.targets[a.count++] = op_target(t, 0);
    }
    ik::lora_term_add(a, nullptr);
    cuda_check(cudaDeviceSynchronize(), "lora_term_add");
    const double round_rel = fp32 ? std::ldexp(1.0, -23) : std::ldexp(1.0, -8);
    std::vector<std::vector<std::uint16_t>> got16(targets.size());
    std::vector<std::vector<float>> got32(targets.size());
    for (std::size_t i = 0; i < targets.size(); ++i) {
        const Target& t = targets[i];
        const Shape& s  = t.shape;
        if (fp32) {
            got32[i] = down(t.dy32, static_cast<std::size_t>(T) * s.stride);
        } else {
            got16[i] = down(t.dy16, static_cast<std::size_t>(T) * s.stride);
        }
        for (int c = 0; c < T; ++c) {
            for (int e = 0; e < s.stride; ++e) {
                const std::size_t at = static_cast<std::size_t>(c) * s.stride + e;
                const double before  = fp32 ? static_cast<double>(t.y32[at]) : bf16_value(t.y16[at]);
                const double after   = fp32 ? static_cast<double>(got32[i][at]) : bf16_value(got16[i][at]);
                const int n          = e - s.first;
                if (n < 0 || n >= s.rows) { // outside the target's rows
                    if (std::memcmp(&before, &after, sizeof(double)) != 0) {
                        fail("a row outside the target changed", rank, T, static_cast<int>(i), e, c);
                    }
                    continue;
                }
                double sum = 0, magnitude = 0;
                for (int j = 0; j < rank; ++j) {
                    const double p = bf16_value(t.b[static_cast<std::size_t>(n) * rank + j]) *
                                     bf16_value(t.u[static_cast<std::size_t>(c) * rank + j]);
                    sum += p;
                    magnitude += std::abs(p);
                }
                const double ref   = before + static_cast<double>(t.scale) * sum;
                const double d     = (rank + 2) * std::ldexp(1.0, -23) * std::abs(t.scale) * magnitude +
                                     std::ldexp(1.0, -22) * std::abs(before);
                if (std::abs(after - ref) > round_rel * (std::abs(ref) + d) + d) {
                    fail("outside the FP64 criterion", rank, T, static_cast<int>(i), n, c);
                }
            }
        }
    }
    // Every column alone (all targets, position 0 of a one-column call) gives its bits in the wide call.
    for (int c = 0; c < T; c += std::max(1, T / 5)) {
        for (Target& t : targets) { reset_y(t); }
        ik::LoraTermArgs one;
        one.columns  = 1;
        one.rank     = rank;
        one.fp32_out = fp32;
        for (const Target& t : targets) { one.targets[one.count++] = op_target(t, c); }
        ik::lora_term_add(one, nullptr);
        cuda_check(cudaDeviceSynchronize(), "lora_term_add column");
        for (std::size_t i = 0; i < targets.size(); ++i) {
            const Shape& s = targets[i].shape;
            if (fp32) {
                const auto column = down(targets[i].dy32 + static_cast<std::size_t>(c) * s.stride, static_cast<std::size_t>(s.stride));
                if (std::memcmp(column.data(), got32[i].data() + static_cast<std::size_t>(c) * s.stride, column.size() * 4) != 0) {
                    fail("a column alone differs", rank, T, static_cast<int>(i), -1, c);
                }
            } else {
                const auto column = down(targets[i].dy16 + static_cast<std::size_t>(c) * s.stride, static_cast<std::size_t>(s.stride));
                if (std::memcmp(column.data(), got16[i].data() + static_cast<std::size_t>(c) * s.stride, column.size() * 2) != 0) {
                    fail("a column alone differs", rank, T, static_cast<int>(i), -1, c);
                }
            }
        }
    }
    for (Target& t : targets) { free_target(t); }
}

// B = 0: y's bits exactly, in every output form.
void zero_b_case(const Shape& shape, int rank, int T, std::uint32_t seed) {
    std::mt19937 gen(seed);
    Target t = make_target(shape, rank, T, gen, true);
    reset_y(t);
    ik::LoraTermArgs a;
    a.columns    = T;
    a.rank       = rank;
    a.fp32_out   = shape.fp32;
    a.targets[0] = op_target(t, 0);
    a.count      = 1;
    ik::lora_term_add(a, nullptr);
    cuda_check(cudaDeviceSynchronize(), "lora_term_add");
    if (shape.fp32) {
        const auto got = down(t.dy32, t.y32.size());
        if (std::memcmp(got.data(), t.y32.data(), got.size() * 4) != 0) { fail("B = 0 changed y", rank, T, 0, -1, -1); }
    } else {
        const auto got = down(t.dy16, t.y16.size());
        if (std::memcmp(got.data(), t.y16.data(), got.size() * 2) != 0) { fail("B = 0 changed y", rank, T, 0, -1, -1); }
    }
    free_target(t);
}

} // namespace

INFERNIX_OP_TEST(inkling2_lora) {
    std::uint32_t seed = 20261123;
    const std::vector<Shape> k2 = {{"query", 3072, 3072, 0, false},  {"relative", 384, 384, 0, false},
                                   {"gate", 3072, 3072, 0, false},   {"sink", 24, 24, 0, false},
                                   {"key", 1024, 1024, 0, false},    {"value", 1024, 1024, 0, false},
                                   {"key taps", 32, 32, 0, false},   {"value taps", 32, 32, 0, false}};
    const std::vector<Shape> wo    = {{"W_o", 2304, 2304, 0, false}};
    const std::vector<Shape> downs = {{"gate down q", 64, 64, 0, false}, {"gate down k", 64, 64, 0, false},
                                      {"gate down v", 64, 64, 0, false}};
    // A gate's forget rows, then its message rows in the same 4608-element column.
    const std::vector<Shape> gate_fg = {{"forget", 2304, 4608, 0, true}, {"message", 2304, 4608, 2304, true}};
    for (const int rank : {16, 64, 128}) {
        for (const int T : {1, 2, 8, 9, 33}) { run_case(k2, rank, T, seed++); }
        for (const int T : {1, 8, 100}) {
            run_case(wo, rank, T, seed++);
            run_case(downs, rank, T, seed++);
            run_case(gate_fg, rank, T, seed++);
        }
        run_case({k2[3]}, rank, 257, seed++); // the sink's partial row group over a wide call
    }
    for (const Shape& s : {k2[0], k2[3], gate_fg[1]}) {
        for (const int T : {1, 16}) { zero_b_case(s, 64, T, seed++); }
    }
    // Every other k-step count: each has its own compiled instances, narrow and wide, BF16 and FP32.
    for (const int rank : {32, 48, 80, 96, 112}) {
        run_case(wo, rank, 8, seed++);
        run_case(wo, rank, 100, seed++);
        run_case(gate_fg, rank, 8, seed++);
        run_case(gate_fg, rank, 100, seed++);
        run_case({k2[3]}, rank, 257, seed++);
    }
    std::printf(g_failures == 0 ? "PASS\n" : "FAILED (%d)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
