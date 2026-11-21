// The bf16_p12 packing sweep for Inkling 2's projections (design docs/maintainer/inkling2_mini-design.md §11 L4: a
// matrix loads packed only where its packed routes measure faster at the decode and verify widths): every shape the
// decode round streams, BF16 against bf16_p12 at T = 1, 2, 4, 8, 16 columns, through the Op the forward uses for it
// (ops::inkling2::projections for a visit's K2 and its message gates, ops::linear for W_o and the head). Random
// weights of the checkpoint's scale (escapes rare, as the census found: 84-966 per million); each launch reads its
// own copy, so the weights stream from DRAM; median of repeats of a 16-launch CUDA graph, microseconds a call.

#include "core/tensor.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "infernix/ops/bf16_p12.h"
#include "infernix/ops/inkling2_projections.h"
#include "infernix/ops/linear.h"
#include "ops/common/canonical_math.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace ik    = infernix::ops::inkling2;
namespace canon = infernix::ops::canon;
using infernix::DType;
using infernix::QType;
using infernix::QuantLayout;
using infernix::Tensor;
using infernix::Weight;

namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

std::mt19937 g_rng(3);

std::uint16_t* random_bf16(std::size_t n, float sd) {
    std::normal_distribution<float> d(0.0F, sd);
    std::vector<std::uint16_t> host(n);
    for (auto& x : host) { x = canon::f32_to_bf16_rn(d(g_rng)); }
    std::uint16_t* p = nullptr;
    check(cudaMalloc(&p, n * 2), "cudaMalloc");
    check(cudaMemcpy(p, host.data(), n * 2, cudaMemcpyHostToDevice), "upload");
    return p;
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

// The matrix packed on the device (ops::bf16_p12_count / bf16_p12_pack) as a P12Rows Weight.
Weight packed_weight(const std::uint16_t* src, int n, int k) {
    void* scratch = nullptr;
    std::uint32_t* d_escapes = nullptr;
    check(cudaMalloc(&scratch, infernix::ops::bf16_p12_pack_scratch_bytes(n)), "cudaMalloc");
    check(cudaMalloc(&d_escapes, 4), "cudaMalloc");
    infernix::ops::bf16_p12_count(src, n, k, scratch, d_escapes, nullptr);
    std::uint32_t escapes = 0;
    check(cudaMemcpy(&escapes, d_escapes, 4, cudaMemcpyDeviceToHost), "escapes");
    const infernix::WeightGeometry g = infernix::bf16_p12_geometry(n, k, escapes);
    std::uint8_t* dst = nullptr;
    check(cudaMalloc(&dst, g.bytes), "cudaMalloc");
    infernix::ops::bf16_p12_pack(src, n, k, escapes, scratch, dst, nullptr);
    check(cudaDeviceSynchronize(), "pack");
    cudaFree(scratch);
    cudaFree(d_escapes);
    Weight w = bf16_weight(nullptr, n, k);
    w.qtype   = QType::BF16_P12;
    w.layout  = QuantLayout::P12Rows;
    w.qdata   = dst;
    w.qhigh   = dst + g.high_offset;
    w.scales  = dst + g.scale_offset;
    w.escapes = dst + g.escape_offset;
    return w;
}

double time_graph(const std::function<void(int, cudaStream_t)>& call, cudaStream_t stream, int launches, int repeat) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
    for (int i = 0; i < launches; ++i) { call(i, stream); }
    check(cudaStreamEndCapture(stream, &graph), "capture");
    check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    cudaEvent_t start = nullptr, stop = nullptr;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");
    std::vector<double> us;
    for (int r = 0; r < repeat + 2; ++r) {
        check(cudaEventRecord(start, stream), "record");
        check(cudaGraphLaunch(exec, stream), "launch");
        check(cudaEventRecord(stop, stream), "record");
        check(cudaEventSynchronize(stop), "sync");
        float ms = 0;
        check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        if (r >= 2) { us.push_back(1000.0 * ms / launches); }
    }
    std::sort(us.begin(), us.end());
    check(cudaGraphExecDestroy(exec), "destroy");
    check(cudaGraphDestroy(graph), "destroy");
    return us[us.size() / 2];
}

struct Shape {
    int rows, input;
};

// A fused projection set (copies of its matrices, rotated per launch) at BF16 and bf16_p12.
void projection_set(const char* name, int k, bool fp32, const std::vector<Shape>& shapes, int copies,
                    const std::vector<int>& widths, int repeat, cudaStream_t stream) {
    std::vector<std::vector<Weight>> bf16(copies), p12(copies);
    double bytes_bf16 = 0, bytes_p12 = 0;
    for (int c = 0; c < copies; ++c) {
        for (const Shape& s : shapes) {
            const auto* w = random_bf16(static_cast<std::size_t>(s.rows) * k, 0.02F);
            bf16[c].push_back(bf16_weight(w, s.rows, k));
            p12[c].push_back(packed_weight(w, s.rows, k));
        }
    }
    for (const Shape& s : shapes) { bytes_bf16 += 2.0 * s.rows * k, bytes_p12 += 1.5 * s.rows * k; }
    std::uint16_t* x[3];
    for (auto& p : x) { p = random_bf16(static_cast<std::size_t>(16) * k, 1.0F); }
    std::vector<void*> outs;
    for (const Shape& s : shapes) {
        void* o = nullptr;
        check(cudaMalloc(&o, static_cast<std::size_t>(16) * s.rows * 4), "cudaMalloc");
        outs.push_back(o);
    }
    for (const int T : widths) {
        double times[2];
        for (int packed = 0; packed < 2; ++packed) {
            const auto call = [&](int i, cudaStream_t st) {
                const auto& set = packed ? p12[i % copies] : bf16[i % copies];
                ik::ProjectionArgs a;
                a.columns  = T;
                a.k        = k;
                a.fp32_out = fp32;
                for (int j = 0; j < 3; ++j) { a.x[j] = x[j]; }
                for (std::size_t s = 0; s < shapes.size(); ++s) {
                    ik::ProjectionSegment g;
                    g.rows  = shapes[s].rows;
                    g.input = shapes[s].input;
                    g.out   = outs[s];
                    if (packed) {
                        g.codes = set[s].qdata, g.nibbles = set[s].qhigh;
                        g.words   = static_cast<const unsigned*>(set[s].scales);
                        g.escapes = static_cast<const unsigned*>(set[s].escapes);
                    } else {
                        g.weight = static_cast<const std::uint16_t*>(set[s].qdata);
                    }
                    a.segments[a.count++] = g;
                }
                ik::projections(a, st);
            };
            times[packed] = time_graph(call, stream, 16, repeat);
        }
        std::printf("  %-34s T=%2d  BF16 %8.2f us (%5.0f GB/s)  p12 %8.2f us (%5.0f GB/s)  p12/BF16 %.3f\n", name, T,
                    times[0], bytes_bf16 / times[0] / 1e3, times[1], bytes_p12 / times[1] / 1e3, times[1] / times[0]);
    }
}

// A matrix through ops::linear at BF16 and bf16_p12.
void linear_shape(const char* name, int n, int k, bool fp32, int copies, const std::vector<int>& widths, int repeat,
                  cudaStream_t stream) {
    std::vector<Weight> bf16, p12;
    for (int c = 0; c < copies; ++c) {
        const auto* w = random_bf16(static_cast<std::size_t>(n) * k, 0.02F);
        bf16.push_back(bf16_weight(w, n, k));
        p12.push_back(packed_weight(w, n, k));
    }
    auto* x = random_bf16(static_cast<std::size_t>(16) * k, 1.0F);
    void* out = nullptr;
    check(cudaMalloc(&out, static_cast<std::size_t>(16) * n * 4), "cudaMalloc");
    for (const int T : widths) {
        double times[2];
        for (int packed = 0; packed < 2; ++packed) {
            const auto call = [&](int i, cudaStream_t st) {
                const Weight& w = packed ? p12[i % copies] : bf16[i % copies];
                // The forward's projections run linear 8 columns a call (the BF16 skinny route); packed weights take
                // 16 a pass inside linear.
                for (int c0 = 0; c0 < T; c0 += 8) {
                    const int cols = std::min(8, T - c0);
                    Tensor xt(x + static_cast<std::size_t>(c0) * k, DType::BF16, {k, cols});
                    Tensor ot(static_cast<std::uint8_t*>(out) + static_cast<std::size_t>(c0) * n * (fp32 ? 4 : 2),
                              fp32 ? DType::FP32 : DType::BF16, {n, cols});
                    infernix::ops::linear(xt, w, ot, st);
                }
            };
            times[packed] = time_graph(call, stream, std::max(1, 16 / copies) * copies, repeat);
        }
        std::printf("  %-34s T=%2d  BF16 %8.2f us (%5.0f GB/s)  p12 %8.2f us (%5.0f GB/s)  p12/BF16 %.3f\n", name, T,
                    times[0], 2.0 * n * k / times[0] / 1e3, times[1], 1.5 * n * k / times[1] / 1e3, times[1] / times[0]);
    }
}

} // namespace

int main(int argc, char** argv) {
    int repeat = 20;
    std::vector<int> widths = {1, 2, 4, 8, 16};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--repeat" && i + 1 < argc) { repeat = std::atoi(argv[++i]); }
        else {
            std::fprintf(stderr, "usage: %s [--repeat N]\n", argv[0]);
            return 2;
        }
    }
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    std::printf("Inkling 2 projections, BF16 vs bf16_p12 (median of %d, us a call)\n", repeat);
    projection_set("K2, sliding visit", 2304, false,
                   {{3072, 0}, {384, 0}, {3072, 0}, {24, 0}, {1024, 1}, {1024, 2}}, 4, widths, repeat, stream);
    projection_set("K2, full visit", 2304, false,
                   {{3072, 0}, {384, 0}, {3072, 0}, {24, 0}, {1024, 1}, {32, 1}, {1024, 2}, {32, 2}}, 4, widths, repeat,
                   stream);
    projection_set("message-gate downs 3x[64,4608]", 4608, false, {{64, 0}, {64, 1}, {64, 2}}, 8, widths, repeat, stream);
    projection_set("forget+gain 3x[4608,64] FP32", 64, true, {{4608, 0}, {4608, 1}, {4608, 2}}, 8, widths, repeat, stream);
    linear_shape("W_o [2304,3072]", 2304, 3072, false, 8, widths, repeat, stream);
    linear_shape("head [200058,2304] FP32", 200058, 2304, true, 1, widths, repeat, stream);
    return 0;
}
