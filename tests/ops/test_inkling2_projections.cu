// Inkling 2's fused projections (include/infernix/ops/inkling2_projections.h) against ops::linear's BF16 route on the
// same matrices: every output bit for bit (the contract's arithmetic is linear's skinny order), at the real shapes of
// one visit (K2: query, relative, gate and sink of x_q, key and its taps of x_k, value and its taps of x_v; the three
// message gates' downs [64, 4608]; their forget and gain [4608, 64] with FP32 out; W_o [2304, 3072], which wide calls
// run here), at widths 1, 3, 8, 16, 17, 40, 100 and 300 (passes of 16 and a partial one; past 16 columns BF16 storage
// runs the wide kernel, the weights held across passes). linear() runs 8 columns a call (its column-invariant skinny
// route), so the oracle is linear's arithmetic itself; its accuracy is the Linear Op's qualification.
// Q8 storage (the lossy 8-bit weights) against an FP64 oracle on the stored codes and scales decoded on the host, within
// the FP32 fold's error bound, and every column equal to the same column projected alone (run_q8). 8-bit inputs
// (projections_mma's x8, --a8-activations): bit for bit the contract's order emulated on the host from the codec's
// definition, within FP64's bound, every column equal to the column alone (run_a8).

#include "core/tensor.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "infernix/ops/bf16_p12.h"
#include "infernix/ops/inkling2_projections.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/q8_g32.h"
#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_a8.h"
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
using infernix::DType;
using infernix::QType;
using infernix::QuantLayout;
using infernix::Tensor;
using infernix::Weight;
using infernix::test::cuda_check;

namespace {

int g_failures = 0;

template <class T>
T* dalloc(std::size_t n) {
    T* p = nullptr;
    cuda_check(cudaMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    return p;
}

std::vector<std::uint16_t> random_bf16(std::size_t n, float sd, std::mt19937& gen) {
    std::normal_distribution<float> d(0.0F, sd);
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) { x = canon::f32_to_bf16_rn(d(gen)); }
    return v;
}

Weight bf16_weight(const std::uint16_t* data, int n, int k) {
    Weight w;
    w.qdata  = data;
    w.qtype  = QType::BF16;
    w.layout = QuantLayout::Contiguous;
    w.ndim   = 2;
    w.n = w.shape[0] = w.padded_shape[0] = n;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    return w;
}

struct Case {
    const char* name;
    int k;
    bool fp32;
    std::vector<std::pair<int, int>> segments; // (rows, input)
};

// The matrix packed on the device as bf16_p12 planes: (codes, nibbles, row words, escapes).
struct Packed {
    std::uint8_t* base = nullptr;
    const void* codes = nullptr;
    const void* nibbles = nullptr;
    const unsigned* words = nullptr;
    const unsigned* escapes = nullptr;
};
Packed pack(const std::uint16_t* src, int n, int k) {
    void* scratch = dalloc<std::uint8_t>(infernix::ops::bf16_p12_pack_scratch_bytes(n));
    auto* d_escapes = dalloc<std::uint32_t>(1);
    infernix::ops::bf16_p12_count(src, n, k, scratch, d_escapes, nullptr);
    std::uint32_t escapes = 0;
    cuda_check(cudaMemcpy(&escapes, d_escapes, 4, cudaMemcpyDeviceToHost), "escapes");
    const infernix::WeightGeometry g = infernix::bf16_p12_geometry(n, k, escapes);
    Packed p;
    p.base = dalloc<std::uint8_t>(g.bytes);
    infernix::ops::bf16_p12_pack(src, n, k, escapes, scratch, p.base, nullptr);
    cuda_check(cudaDeviceSynchronize(), "pack");
    cudaFree(scratch);
    cudaFree(d_escapes);
    p.codes   = p.base;
    p.nibbles = p.base + g.high_offset;
    p.words   = reinterpret_cast<const unsigned*>(p.base + g.scale_offset);
    p.escapes = reinterpret_cast<const unsigned*>(p.base + g.escape_offset);
    return p;
}

// An FP16 value (finite) exactly.
double f16_value(std::uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    const double s = (h & 0x8000U) != 0 ? -1.0 : 1.0;
    return e == 0 ? s * std::ldexp(m, -24) : s * std::ldexp(1024 + m, e - 25);
}

// Q8 storage (LOSSY): the matrix quantized by ops::quantize_bf16_q8_g32 (its codes and scales are qualified exactly
// by test_hyper_connection's repack case, which runs the same Op). The oracle decodes the stored codes with their
// stored scales on the host and sums in FP64; the criterion is the FP32 fold's error bound (each lane folds K / 32
// products by fma, then 5 butterfly adds: |err| <= (K / 32 + 5) u sum |w x|, u = 2^-24) plus BF16 rounding of the
// output (<= 2^-8 |y|, one ulp's half on either side of a rounding). Every column of the call must also equal the
// same column projected alone (T = 1, the narrow kernel) bit for bit: width invariance, and wide = narrow.
using ProjectionsFn = void (*)(const ik::ProjectionArgs&, cudaStream_t);
// fn: the fold (projections, bound (K / 32 + 5) 2^-24 sum|wx|) or an MMA route (projections_mma, the prompt route's, and
// projections_dmma, the decode route's: (K / 16 + 5) 2^-23 sum|wx|: per 32-group an MMA sum then an fma with its scale).
void run_q8(const Case& c, int columns, std::uint32_t seed, ProjectionsFn fn = ik::projections, bool mma = false,
            const char* label = nullptr) {
    std::mt19937 gen(seed);
    const int word = c.fp32 ? 4 : 2;
    std::uint16_t* x[3];
    std::vector<std::uint16_t> xh[3];
    for (int i = 0; i < 3; ++i) {
        x[i]  = dalloc<std::uint16_t>(static_cast<std::size_t>(columns) * c.k);
        xh[i] = random_bf16(static_cast<std::size_t>(columns) * c.k, 1.0F, gen);
        cuda_check(cudaMemcpy(x[i], xh[i].data(), xh[i].size() * 2, cudaMemcpyHostToDevice), "upload");
    }
    ik::ProjectionArgs args;
    args.columns  = columns;
    args.k        = c.k;
    args.fp32_out = c.fp32;
    for (int i = 0; i < 3; ++i) { args.x[i] = x[i]; }
    std::vector<void*> blocks;
    std::vector<std::vector<double>> want, scale_sum;
    for (const auto& [rows, input] : c.segments) {
        auto* w   = dalloc<std::uint16_t>(static_cast<std::size_t>(rows) * c.k);
        auto host = random_bf16(static_cast<std::size_t>(rows) * c.k, 0.03F, gen);
        for (std::size_t i = 0; i < host.size(); i += 997) { // zeros and outliers among the groups
            host[i] = (i / 997) % 2 == 0 ? std::uint16_t{0} : canon::f32_to_bf16_rn(3.0F);
        }
        std::fill(host.begin(), host.begin() + 32, std::uint16_t{0}); // a zero group: scale 0, codes 0
        cuda_check(cudaMemcpy(w, host.data(), host.size() * 2, cudaMemcpyHostToDevice), "upload");
        void* block = dalloc<std::uint8_t>(infernix::ops::q8_g32_bytes(rows, c.k));
        auto* scale_plane = static_cast<std::uint8_t*>(block) +
                            static_cast<std::size_t>(rows) * infernix::ops::q8_g32_padded_columns(c.k);
        const Weight q = infernix::ops::quantize_bf16_q8_g32(bf16_weight(w, rows, c.k), block, scale_plane, nullptr);
        cuda_check(cudaDeviceSynchronize(), "quantize");
        cudaFree(w);
        blocks.push_back(block);
        const int stride = static_cast<int>(q.padded_shape[1]);
        std::vector<std::int8_t> codes(static_cast<std::size_t>(rows) * stride);
        std::vector<std::uint16_t> scales(static_cast<std::size_t>(rows) * (stride / 32));
        cuda_check(cudaMemcpy(codes.data(), q.qdata, codes.size(), cudaMemcpyDeviceToHost), "download");
        cuda_check(cudaMemcpy(scales.data(), q.scales, scales.size() * 2, cudaMemcpyDeviceToHost), "download");
        std::vector<double> y(static_cast<std::size_t>(columns) * rows), bound(y.size());
        for (int t = 0; t < columns; ++t) {
            const std::uint16_t* xt = xh[input].data() + static_cast<std::size_t>(t) * c.k;
            for (int r = 0; r < rows; ++r) {
                double s = 0, a = 0;
                for (int j = 0; j < c.k; ++j) {
                    const double wv = codes[static_cast<std::size_t>(r) * stride + j] *
                                      f16_value(scales[static_cast<std::size_t>(r) * (stride / 32) + j / 32]);
                    const double p = wv * canon::bf16_to_f32(xt[j]);
                    s += p, a += std::fabs(p);
                }
                y[static_cast<std::size_t>(t) * rows + r]     = s;
                bound[static_cast<std::size_t>(t) * rows + r] = a;
            }
        }
        want.push_back(std::move(y));
        scale_sum.push_back(std::move(bound));
        ik::ProjectionSegment seg;
        seg.rows      = rows;
        seg.input     = input;
        seg.out       = dalloc<std::uint8_t>(static_cast<std::size_t>(columns) * rows * word);
        seg.codes     = q.qdata;
        seg.q8_scales = static_cast<const std::uint16_t*>(q.scales);
        seg.q8_stride = stride;
        args.segments[args.count++] = seg;
    }
    fn(args, nullptr);
    // Each column alone: the same segments, the inputs and outputs offset to that column.
    std::vector<void*> alone;
    for (int s = 0; s < args.count; ++s) {
        alone.push_back(dalloc<std::uint8_t>(static_cast<std::size_t>(columns) * args.segments[s].rows * word));
    }
    for (int t = 0; t < columns; ++t) {
        ik::ProjectionArgs one = args;
        one.columns            = 1;
        for (int i = 0; i < 3; ++i) { one.x[i] = x[i] + static_cast<std::size_t>(t) * c.k; }
        for (int s = 0; s < args.count; ++s) {
            one.segments[s].out = static_cast<std::uint8_t*>(alone[s]) + static_cast<std::size_t>(t) * args.segments[s].rows * word;
        }
        fn(one, nullptr);
    }
    cuda_check(cudaDeviceSynchronize(), "projections");
    long outside = 0, differ = 0, total = 0;
    double worst = 0;
    const double depth = mma ? 2.0 * (c.k / 16 + 5) : c.k / 32 + 5; // in units of 2^-24
    for (int s = 0; s < args.count; ++s) {
        const std::size_t n = static_cast<std::size_t>(columns) * args.segments[s].rows;
        std::vector<std::uint8_t> got(n * word), solo(n * word);
        cuda_check(cudaMemcpy(got.data(), args.segments[s].out, got.size(), cudaMemcpyDeviceToHost), "download");
        cuda_check(cudaMemcpy(solo.data(), alone[s], solo.size(), cudaMemcpyDeviceToHost), "download");
        for (std::size_t i = 0; i < n; ++i) {
            double g = 0;
            if (c.fp32) {
                float f = 0;
                std::memcpy(&f, &got[i * 4], 4);
                g = f;
            } else {
                std::uint16_t b = 0;
                std::memcpy(&b, &got[i * 2], 2);
                g = canon::bf16_to_f32(b);
            }
            const double y   = want[s][i];
            const double tol = 1.001 * depth * 0x1p-24 * scale_sum[s][i] + (c.fp32 ? 0.0 : 0x1p-8 * std::fabs(y) + 1e-30);
            const double err = std::fabs(g - y);
            outside += err > tol ? 1 : 0;
            worst = std::max(worst, scale_sum[s][i] > 0 ? err / scale_sum[s][i] : 0.0);
            differ += std::memcmp(&got[i * word], &solo[i * word], word) != 0 ? 1 : 0;
        }
        total += static_cast<long>(n);
    }
    std::printf("%-44s %s T = %2d: %ld of %ld outside the bound (worst |err| / sum|wx| %.2e), %ld differ from the "
                "column alone\n",
                c.name, label != nullptr ? label : (mma ? "Q8MMA" : "Q8  "), columns, outside, total, worst, differ);
    if (outside != 0 || differ != 0) { ++g_failures; }
    for (auto* p : x) { cudaFree(p); }
    for (auto* p : blocks) { cudaFree(p); }
    for (int s = 0; s < args.count; ++s) { cudaFree(args.segments[s].out); }
    for (auto* p : alone) { cudaFree(p); }
}

// The tensor-core projections (projections_mma: ops::linear's MMA arithmetic, k16 steps in ascending K chained through
// the FP32 accumulator). Oracle: FP64 over the BF16 operands (every product exact). Criterion, fixed before the first
// run: |err| <= (K / 16 + 5) 2^-23 sum |w x| (one rounding of the accumulator a step, plus a 5-level tree inside a
// step, at a unit that also covers truncating internal adds) plus BF16 rounding of the output (2^-8 |y|). Every
// column must equal the same column projected alone (a 1-column call: the narrow tile shape) bit for bit, so the bits
// depend neither on the call's width nor on its tile shape. Wide calls check a sample of their columns.
void run_mma(const Case& c, int columns, std::uint32_t seed, ProjectionsFn fn = ik::projections_mma, const char* label = "MMA ") {
    std::mt19937 gen(seed);
    const int word = c.fp32 ? 4 : 2;
    std::uint16_t* x[3];
    std::vector<std::uint16_t> xh[3];
    for (int i = 0; i < 3; ++i) {
        x[i]  = dalloc<std::uint16_t>(static_cast<std::size_t>(columns) * c.k);
        xh[i] = random_bf16(static_cast<std::size_t>(columns) * c.k, 1.0F, gen);
        cuda_check(cudaMemcpy(x[i], xh[i].data(), xh[i].size() * 2, cudaMemcpyHostToDevice), "upload");
    }
    ik::ProjectionArgs args;
    args.columns  = columns;
    args.k        = c.k;
    args.fp32_out = c.fp32;
    for (int i = 0; i < 3; ++i) { args.x[i] = x[i]; }
    std::vector<std::vector<std::uint16_t>> wh;
    std::vector<std::uint16_t*> weights;
    for (const auto& [rows, input] : c.segments) {
        auto* w   = dalloc<std::uint16_t>(static_cast<std::size_t>(rows) * c.k);
        auto host = random_bf16(static_cast<std::size_t>(rows) * c.k, 0.03F, gen);
        for (std::size_t i = 0; i < host.size(); i += 997) {
            host[i] = (i / 997) % 3 == 0 ? std::uint16_t{0} : canon::f32_to_bf16_rn((i / 997) % 3 == 1 ? 3.0F : 1.0e-9F);
        }
        cuda_check(cudaMemcpy(w, host.data(), host.size() * 2, cudaMemcpyHostToDevice), "upload");
        weights.push_back(w);
        wh.push_back(std::move(host));
        ik::ProjectionSegment seg;
        seg.rows   = rows;
        seg.input  = input;
        seg.weight = w;
        seg.out    = dalloc<std::uint8_t>(static_cast<std::size_t>(columns) * rows * word);
        args.segments[args.count++] = seg;
    }
    fn(args, nullptr);
    // The checked columns: all up to 40, else the first 3, the last 2 and 5 between.
    std::vector<int> checked;
    if (columns <= 40) {
        for (int t = 0; t < columns; ++t) { checked.push_back(t); }
    } else {
        checked = {0, 1, 2, columns / 7, columns / 3, columns / 2, (2 * columns) / 3, columns - 9, columns - 2, columns - 1};
    }
    std::vector<void*> alone;
    for (int s = 0; s < args.count; ++s) {
        alone.push_back(dalloc<std::uint8_t>(static_cast<std::size_t>(columns) * args.segments[s].rows * word));
    }
    for (const int t : checked) {
        ik::ProjectionArgs one = args;
        one.columns            = 1;
        for (int i = 0; i < 3; ++i) { one.x[i] = x[i] + static_cast<std::size_t>(t) * c.k; }
        for (int s = 0; s < args.count; ++s) {
            one.segments[s].out =
                static_cast<std::uint8_t*>(alone[s]) + static_cast<std::size_t>(t) * args.segments[s].rows * word;
        }
        fn(one, nullptr);
    }
    cuda_check(cudaDeviceSynchronize(), "projections_mma");
    long outside = 0, differ = 0, total = 0;
    double worst = 0;
    const double depth = c.k / 16 + 5;
    for (int s = 0; s < args.count; ++s) {
        const int rows = args.segments[s].rows, input = c.segments[s].second;
        const std::size_t n = static_cast<std::size_t>(columns) * rows;
        std::vector<std::uint8_t> got(n * word), solo(n * word);
        cuda_check(cudaMemcpy(got.data(), args.segments[s].out, got.size(), cudaMemcpyDeviceToHost), "download");
        cuda_check(cudaMemcpy(solo.data(), alone[s], solo.size(), cudaMemcpyDeviceToHost), "download");
        for (const int t : checked) {
            const std::uint16_t* xt = xh[input].data() + static_cast<std::size_t>(t) * c.k;
            for (int r = 0; r < rows; ++r) {
                const std::uint16_t* wr = wh[s].data() + static_cast<std::size_t>(r) * c.k;
                double y = 0, a = 0;
                for (int j = 0; j < c.k; ++j) {
                    const double p = static_cast<double>(canon::bf16_to_f32(wr[j])) * canon::bf16_to_f32(xt[j]);
                    y += p, a += std::fabs(p);
                }
                const std::size_t i = static_cast<std::size_t>(t) * rows + r;
                double g = 0;
                if (c.fp32) {
                    float f = 0;
                    std::memcpy(&f, &got[i * 4], 4);
                    g = f;
                } else {
                    std::uint16_t b = 0;
                    std::memcpy(&b, &got[i * 2], 2);
                    g = canon::bf16_to_f32(b);
                }
                const double tol = depth * 0x1p-23 * a + (c.fp32 ? 0.0 : 0x1p-8 * std::fabs(y)) + 1e-30;
                const double err = std::fabs(g - y);
                outside += err > tol ? 1 : 0;
                worst = std::max(worst, a > 0 ? err / (depth * 0x1p-23 * a + (c.fp32 ? 0.0 : 0x1p-8 * std::fabs(y))) : 0.0);
                differ += std::memcmp(&got[i * word], &solo[i * word], word) != 0 ? 1 : 0;
                ++total;
            }
        }
    }
    std::printf("%-44s %s T = %4d: %ld of %ld outside the bound (worst %.3f of it), %ld differ from the column "
                "alone\n",
                c.name, label, columns, outside, total, worst, differ);
    if (outside != 0 || differ != 0) { ++g_failures; }
    for (auto* p : x) { cudaFree(p); }
    for (auto* p : weights) { cudaFree(p); }
    for (int s = 0; s < args.count; ++s) { cudaFree(args.segments[s].out); }
    for (auto* p : alone) { cudaFree(p); }
}

// A matrix of Q8 storage on the device (quantize_bf16_q8_g32 of random BF16 rows with zeros, outliers and a zero
// group) and its stored codes and scales on the host.
struct Q8Matrix {
    void* block = nullptr;
    Weight q;
    int stride = 0;
    std::vector<std::int8_t> codes;
    std::vector<std::uint16_t> scales;
};
Q8Matrix make_q8(int rows, int k, std::mt19937& gen) {
    auto* w   = dalloc<std::uint16_t>(static_cast<std::size_t>(rows) * k);
    auto host = random_bf16(static_cast<std::size_t>(rows) * k, 0.03F, gen);
    for (std::size_t i = 0; i < host.size(); i += 997) { host[i] = (i / 997) % 2 == 0 ? std::uint16_t{0} : canon::f32_to_bf16_rn(3.0F); }
    std::fill(host.begin(), host.begin() + 32, std::uint16_t{0});
    cuda_check(cudaMemcpy(w, host.data(), host.size() * 2, cudaMemcpyHostToDevice), "upload");
    Q8Matrix m;
    m.block = dalloc<std::uint8_t>(infernix::ops::q8_g32_bytes(rows, k));
    auto* scale_plane = static_cast<std::uint8_t*>(m.block) + static_cast<std::size_t>(rows) * infernix::ops::q8_g32_padded_columns(k);
    m.q = infernix::ops::quantize_bf16_q8_g32(bf16_weight(w, rows, k), m.block, scale_plane, nullptr);
    cuda_check(cudaDeviceSynchronize(), "quantize");
    cudaFree(w);
    m.stride = static_cast<int>(m.q.padded_shape[1]);
    m.codes.resize(static_cast<std::size_t>(rows) * m.stride);
    m.scales.resize(static_cast<std::size_t>(rows) * (m.stride / 32));
    cuda_check(cudaMemcpy(m.codes.data(), m.q.qdata, m.codes.size(), cudaMemcpyDeviceToHost), "download");
    cuda_check(cudaMemcpy(m.scales.data(), m.q.scales, m.scales.size() * 2, cudaMemcpyDeviceToHost), "download");
    return m;
}

// 8-bit inputs (projections_mma with x8 over Q8 storage; LOSSY). Oracle: the inputs encoded on the host by the codec's
// definition (ops/ggml/ggml_a8.h: slice scale d = fl32(m / 127), codes rne(fl32(v fl32(127 / m))) clamped to +-127),
// then (1) the contract's arithmetic in binary32 on the host (acc = +0; per slice in order acc = fma(fl32(P w), d, acc),
// P the exact integer dot): every output bit for bit; (2) FP64 sum_s P_s w_s d_s within the order's bound
// (K / 32 + 1) 2^-24 sum_s |P_s w_s d_s| plus BF16 rounding (2^-8 |y|): the arithmetic's accuracy; (3) every checked
// column equal to the same column projected alone (a 1-column call: another tile shape). Inputs: N(0, 1) with, at 4 or
// more columns, a zero slice (d = 0) in column 1, a slice below 2^-120 (d = 0) in column 2 and a NaN in the last
// column (that column's every output NaN). `alias`: the three inputs one buffer (a visit without messages: encoded once).
void run_a8(const Case& c, int columns, std::uint32_t seed, bool alias) {
    namespace a8 = infernix::ggml::a8;
    std::mt19937 gen(seed);
    const int word = c.fp32 ? 4 : 2, slices = c.k / 32;
    std::uint16_t* x[3];
    std::vector<std::uint16_t> xh[3];
    for (int i = 0; i < 3; ++i) {
        if (alias && i > 0) {
            x[i]  = x[0];
            xh[i] = xh[0];
            continue;
        }
        x[i]  = dalloc<std::uint16_t>(static_cast<std::size_t>(columns) * c.k);
        xh[i] = random_bf16(static_cast<std::size_t>(columns) * c.k, 1.0F, gen);
        if (columns >= 4) {
            auto* v = xh[i].data();
            std::fill(v + c.k, v + c.k + 32, std::uint16_t{0});
            for (int j = 0; j < 32; ++j) { v[2 * static_cast<std::size_t>(c.k) + 32 + j] = canon::f32_to_bf16_rn(1.0e-38F * (j % 3)); }
            v[static_cast<std::size_t>(columns - 1) * c.k + 40] = 0x7FC0U;
        }
        cuda_check(cudaMemcpy(x[i], xh[i].data(), xh[i].size() * 2, cudaMemcpyHostToDevice), "upload");
    }
    // The host's codes and slice scales of every input column.
    std::vector<std::int8_t> qx[3];
    std::vector<float> dx[3];
    for (int i = 0; i < 3; ++i) {
        qx[i].resize(xh[i].size());
        dx[i].resize(static_cast<std::size_t>(columns) * slices);
        for (int t = 0; t < columns; ++t) {
            for (int s = 0; s < slices; ++s) {
                const std::uint16_t* v = xh[i].data() + static_cast<std::size_t>(t) * c.k + 32 * s;
                unsigned key           = 0;
                for (int j = 0; j < 32; ++j) { key = std::max(key, static_cast<unsigned>(v[j] & 0x7FFFU)); }
                const a8::SliceScale sc = a8::slice_scale(key);
                dx[i][static_cast<std::size_t>(t) * slices + s] = sc.d;
                for (int j = 0; j < 32; ++j) {
                    const bool finite = (v[j] & 0x7F80U) != 0x7F80U;
                    qx[i][static_cast<std::size_t>(t) * c.k + 32 * s + j] = static_cast<std::int8_t>(finite ? a8::code(v[j], sc.inv) : 0);
                }
            }
        }
    }
    const std::int32_t words = ik::projection_a8_column_words(c.k);
    std::int32_t* x8[3];
    for (auto& p : x8) { p = dalloc<std::int32_t>(static_cast<std::size_t>(columns) * words); }
    ik::ProjectionArgs args;
    args.columns  = columns;
    args.k        = c.k;
    args.fp32_out = c.fp32;
    for (int i = 0; i < 3; ++i) { args.x[i] = x[i], args.x8[i] = x8[i]; }
    std::vector<Q8Matrix> mats;
    for (const auto& [rows, input] : c.segments) {
        mats.push_back(make_q8(rows, c.k, gen));
        ik::ProjectionSegment seg;
        seg.rows      = rows;
        seg.input     = input;
        seg.out       = dalloc<std::uint8_t>(static_cast<std::size_t>(columns) * rows * word);
        seg.codes     = mats.back().q.qdata;
        seg.q8_scales = static_cast<const std::uint16_t*>(mats.back().q.scales);
        seg.q8_stride = mats.back().stride;
        args.segments[args.count++] = seg;
    }
    ik::projections_mma(args, nullptr);
    std::vector<int> checked;
    if (columns <= 40) {
        for (int t = 0; t < columns; ++t) { checked.push_back(t); }
    } else {
        checked = {0, 1, 2, 3, columns / 7, columns / 3, columns / 2, (2 * columns) / 3, columns - 9, columns - 2, columns - 1};
    }
    std::vector<void*> alone;
    for (int s = 0; s < args.count; ++s) {
        alone.push_back(dalloc<std::uint8_t>(static_cast<std::size_t>(columns) * args.segments[s].rows * word));
    }
    for (const int t : checked) {
        ik::ProjectionArgs one = args;
        one.columns            = 1;
        for (int i = 0; i < 3; ++i) { one.x[i] = x[i] + static_cast<std::size_t>(t) * c.k; }
        for (int s = 0; s < args.count; ++s) {
            one.segments[s].out = static_cast<std::uint8_t*>(alone[s]) + static_cast<std::size_t>(t) * args.segments[s].rows * word;
        }
        ik::projections_mma(one, nullptr);
    }
    cuda_check(cudaDeviceSynchronize(), "projections_mma (a8)");
    long outside = 0, wrong = 0, differ = 0, total = 0;
    double worst = 0;
    for (int s = 0; s < args.count; ++s) {
        const Q8Matrix& m = mats[s];
        const int rows = args.segments[s].rows, input = c.segments[s].second;
        const std::size_t n = static_cast<std::size_t>(columns) * rows;
        std::vector<std::uint8_t> got(n * word), solo(n * word);
        cuda_check(cudaMemcpy(got.data(), args.segments[s].out, got.size(), cudaMemcpyDeviceToHost), "download");
        cuda_check(cudaMemcpy(solo.data(), alone[s], solo.size(), cudaMemcpyDeviceToHost), "download");
        for (const int t : checked) {
            for (int r = 0; r < rows; ++r) {
                float acc = 0.0F;
                double y = 0, a = 0;
                for (int sl = 0; sl < slices; ++sl) {
                    long long dot = 0;
                    for (int j = 0; j < 32; ++j) {
                        dot += static_cast<long long>(m.codes[static_cast<std::size_t>(r) * m.stride + 32 * sl + j]) *
                               qx[input][static_cast<std::size_t>(t) * c.k + 32 * sl + j];
                    }
                    const float w = a8::f16_to_f32(m.scales[static_cast<std::size_t>(r) * (m.stride / 32) + sl]);
                    const float d = dx[input][static_cast<std::size_t>(t) * slices + sl];
                    acc           = canon::fma_rn(canon::mul_rn(static_cast<float>(dot), w), d, acc);
                    const double term = static_cast<double>(dot) * w * d;
                    y += term, a += std::fabs(term);
                }
                const std::size_t i = static_cast<std::size_t>(t) * rows + r;
                double g = 0;
                bool same = false;
                if (c.fp32) {
                    float f = 0;
                    std::memcpy(&f, &got[i * 4], 4);
                    g    = f;
                    same = std::isnan(acc) ? std::isnan(f) : std::memcmp(&f, &acc, 4) == 0;
                } else {
                    std::uint16_t b = 0;
                    std::memcpy(&b, &got[i * 2], 2);
                    g = canon::bf16_to_f32(b);
                    const std::uint16_t want = canon::f32_to_bf16_rn(acc);
                    same = std::isnan(acc) ? std::isnan(g) : b == want;
                }
                wrong += same ? 0 : 1;
                if (!std::isnan(y)) {
                    const double tol = (slices + 1) * 0x1p-24 * a + (c.fp32 ? 0.0 : 0x1p-8 * std::fabs(y)) + 1e-30;
                    const double err = std::fabs(g - y);
                    outside += err > tol ? 1 : 0;
                    worst = std::max(worst, err / tol);
                }
                differ += std::memcmp(&got[i * word], &solo[i * word], word) != 0 ? 1 : 0;
                ++total;
            }
        }
    }
    std::printf("%-44s A8%s T = %4d: %ld of %ld differ from the order's bits, %ld outside the bound (worst %.3f of "
                "it), %ld differ from the column alone\n",
                c.name, alias ? " (one input)" : "", columns, wrong, total, outside, worst, differ);
    if (wrong != 0 || outside != 0 || differ != 0) { ++g_failures; }
    for (int i = 0; i < 3; ++i) {
        if (!(alias && i > 0)) { cudaFree(x[i]); }
        cudaFree(x8[i]);
    }
    for (auto& m : mats) { cudaFree(m.block); }
    for (int s = 0; s < args.count; ++s) { cudaFree(args.segments[s].out); }
    for (auto* p : alone) { cudaFree(p); }
}

void run_case(const Case& c, int columns, std::uint32_t seed, bool packed) {
    std::mt19937 gen(seed);
    const int word = c.fp32 ? 4 : 2;
    std::uint16_t* x[3];
    for (auto& p : x) {
        p = dalloc<std::uint16_t>(static_cast<std::size_t>(columns) * c.k);
        const auto host = random_bf16(static_cast<std::size_t>(columns) * c.k, 1.0F, gen);
        cuda_check(cudaMemcpy(p, host.data(), host.size() * 2, cudaMemcpyHostToDevice), "upload");
    }
    ik::ProjectionArgs args;
    args.columns = columns;
    args.k       = c.k;
    args.fp32_out = c.fp32;
    for (int i = 0; i < 3; ++i) { args.x[i] = x[i]; }
    std::vector<std::uint16_t*> weights;
    std::vector<std::uint8_t*> packs;
    std::vector<void*> outs, wants;
    for (const auto& [rows, input] : c.segments) {
        auto* w = dalloc<std::uint16_t>(static_cast<std::size_t>(rows) * c.k);
        auto host = random_bf16(static_cast<std::size_t>(rows) * c.k, 0.03F, gen);
        // A few weights outside a row's 15-exponent window (zeros and wide magnitudes): the packed rows' escapes.
        for (std::size_t i = 0; i < host.size(); i += 997) {
            host[i] = (i / 997) % 3 == 0 ? std::uint16_t{0} : canon::f32_to_bf16_rn((i / 997) % 3 == 1 ? 3.0e4F : 1.0e-9F);
        }
        cuda_check(cudaMemcpy(w, host.data(), host.size() * 2, cudaMemcpyHostToDevice), "upload");
        weights.push_back(w);
        void* out  = dalloc<std::uint8_t>(static_cast<std::size_t>(columns) * rows * word);
        void* want = dalloc<std::uint8_t>(static_cast<std::size_t>(columns) * rows * word);
        outs.push_back(out);
        wants.push_back(want);
        ik::ProjectionSegment seg;
        seg.rows  = rows;
        seg.input = input;
        seg.out   = out;
        if (packed) {
            const Packed p = pack(w, rows, c.k);
            packs.push_back(p.base);
            seg.codes = p.codes, seg.nibbles = p.nibbles, seg.words = p.words, seg.escapes = p.escapes;
        } else {
            seg.weight = w;
        }
        args.segments[args.count++] = seg;
        // The oracle: linear() in calls of 8 columns.
        const Weight weight = bf16_weight(w, rows, c.k);
        for (int c0 = 0; c0 < columns; c0 += 8) {
            const int n = std::min(8, columns - c0);
            Tensor xt(x[input] + static_cast<std::size_t>(c0) * c.k, DType::BF16, {c.k, n});
            Tensor ot(static_cast<std::uint8_t*>(want) + static_cast<std::size_t>(c0) * rows * word,
                      c.fp32 ? DType::FP32 : DType::BF16, {rows, n});
            infernix::ops::linear(xt, weight, ot, nullptr);
        }
    }
    ik::projections(args, nullptr);
    cuda_check(cudaDeviceSynchronize(), "projections");
    long differ = 0, total = 0;
    for (std::size_t s = 0; s < outs.size(); ++s) {
        const std::size_t bytes = static_cast<std::size_t>(columns) * c.segments[s].first * word;
        std::vector<std::uint8_t> got(bytes), want(bytes);
        cuda_check(cudaMemcpy(got.data(), outs[s], bytes, cudaMemcpyDeviceToHost), "download");
        cuda_check(cudaMemcpy(want.data(), wants[s], bytes, cudaMemcpyDeviceToHost), "download");
        for (std::size_t i = 0; i < bytes; i += word) { differ += std::memcmp(&got[i], &want[i], word) != 0 ? 1 : 0; }
        total += static_cast<long>(bytes / word);
    }
    std::printf("%-44s %s T = %2d: %ld of %ld outputs differ from linear()\n", c.name, packed ? "p12 " : "BF16", columns,
                differ, total);
    if (differ != 0) { ++g_failures; }
    for (auto* p : x) { cudaFree(p); }
    for (auto* p : weights) { cudaFree(p); }
    for (auto* p : packs) { cudaFree(p); }
    for (auto* p : outs) { cudaFree(p); }
    for (auto* p : wants) { cudaFree(p); }
}

} // namespace

INFERNIX_OP_TEST(inkling2_projections) {
    const std::vector<Case> cases = {
        {"K2 of a full visit (8 segments, 3 inputs)", 2304, false,
         {{3072, 0}, {384, 0}, {3072, 0}, {24, 0}, {1024, 1}, {32, 1}, {1024, 2}, {32, 2}}},
        {"message-gate downs [64, 4608] x 3", 4608, false, {{64, 0}, {64, 1}, {64, 2}}},
        {"forget and gain [4608, 64] x 3, FP32 out", 64, true, {{4608, 0}, {4608, 1}, {4608, 2}}},
        {"W_o [2304, 3072] (wide calls)", 3072, false, {{2304, 0}}},
    };
    std::uint32_t seed = 1;
    for (const bool packed : {false, true}) {
        for (const auto& c : cases) {
            for (const int columns : {1, 3, 8, 16, 17, 40, 100, 300}) { run_case(c, columns, seed++, packed); }
        }
    }
    // Q8 storage: the shapes an 8-bit load converts (the forget/gain stay BF16), and the head's arithmetic (FP32 out,
    // K = 2304; 4,096 of its rows: the kernel's shape for any row count past 256).
    const std::vector<Case> q8_cases = {cases[0], cases[1], cases[3], {"head rows [4096, 2304], FP32 out", 2304, true, {{4096, 0}}}};
    for (const auto& c : q8_cases) {
        for (const int columns : {1, 3, 8, 16, 17, 40, 100}) { run_q8(c, columns, seed++); }
        // The prompt route's Q8 tensor-core arithmetic.
        for (const int columns : {1, 3, 16, 17, 100, 300}) { run_q8(c, columns, seed++, ik::projections_mma, true); }
        // The decode route's Q8 tensor-core arithmetic (projections_dmma).
        for (const int columns : {1, 2, 3, 8, 9, 16, 17, 40}) {
            run_q8(c, columns, seed++, ik::projections_dmma, true, "Q8DMMA");
        }
        // 8-bit inputs on the prompt route (every tile shape: <= 16 columns, <= 128 or few rows, wide).
        for (const int columns : {1, 4, 16, 17, 100, 300, 2048}) { run_a8(c, columns, seed++, false); }
    }
    run_a8(cases[0], 40, seed++, true);
    run_a8(cases[0], 300, seed++, true);
    // The message gates' forget and gain (K = 64: one half stage) and the router's shape, on 8-bit inputs.
    for (const int columns : {1, 4, 17, 300}) { run_a8(cases[2], columns, seed++, false); }
    for (const int columns : {1, 17, 300}) { run_a8({"router [128, 2304], FP32 out", 2304, true, {{128, 0}}}, columns, seed++, false); }
    // Output rows of no 16-byte multiple (the head's FP32 logits rows: V = 200,002 is 8 bytes off).
    for (const int columns : {1, 3, 17, 130}) {
        run_a8({"head rows [4098, 2304], FP32 out, unaligned rows", 2304, true, {{4098, 0}}}, columns, seed++, false);
    }
    // The tensor-core projections at the same shapes, the router's [128, 2304] FP32 and the head's arithmetic, at
    // narrow and wide widths (tile shapes: <= 16 columns, few rows, wide).
    const std::vector<Case> mma_cases = {cases[0], cases[1], cases[2], cases[3],
                                         {"router [128, 2304], FP32 out", 2304, true, {{128, 0}}},
                                         {"head rows [4096, 2304], FP32 out", 2304, true, {{4096, 0}}}};
    for (const auto& c : mma_cases) {
        for (const int columns : {1, 3, 16, 17, 40, 300, 2048}) { run_mma(c, columns, seed++); }
        // The decode-width MMA arithmetic (projections_dmma) at decode and verify widths.
        for (const int columns : {1, 2, 3, 8, 9, 16, 17, 40, 128}) { run_mma(c, columns, seed++, ik::projections_dmma, "DMMA"); }
    }
    std::printf(g_failures == 0 ? "PASS\n" : "FAILED (%d)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
