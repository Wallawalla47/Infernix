// Inkling 2's fused projections (include/infernix/ops/inkling2_projections.h) against ops::linear's BF16 route on the
// same matrices: every output bit for bit (the contract's arithmetic is linear's skinny order), at the real shapes of
// one visit (K2: query, relative, gate and sink of x_q, key and its taps of x_k, value and its taps of x_v; the three
// message gates' downs [64, 4608]; their forget and gain [4608, 64] with FP32 out; W_o [2304, 3072], which wide calls
// run here), at widths 1, 3, 8, 16, 17, 40, 100 and 300 (passes of 16 and a partial one; past 16 columns BF16 storage
// runs the wide kernel, the weights held across passes). linear() runs 8 columns a call (its column-invariant skinny
// route), so the oracle is linear's arithmetic itself; its accuracy is the Linear Op's qualification.

#include "core/tensor.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "infernix/ops/bf16_p12.h"
#include "infernix/ops/inkling2_projections.h"
#include "infernix/ops/linear.h"
#include "ops/common/canonical_math.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
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
    std::printf(g_failures == 0 ? "PASS\n" : "FAILED (%d)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
