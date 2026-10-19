// GGUF (ggml) expert banks through the layer route of offloaded_sparse_moe (moe_route, moe_dispatch,
// moe_experts; docs/maintainer/qwen3_8-flash-next-design.md §16.2.2, §19.3.25).
//
// Records are random ggml blocks (finite binary16 scales) in a pinned host bank, some copied into
// device frames whose pitch exceeds the record stride (the shared frame pool of a model whose
// layers' records differ in size). Misses are read zero-copy, staged through 3 or 64 slots, or
// forked, or computed by the CPU route through the miss service; calls of 1 and 3 columns cover
// decode and verification, and calls of 9, 40 and 300
// columns the wide route (jobs of more than 8 columns run the tensor-core tiles, the others the
// narrow kernels; at 300 columns two favoured experts take several 64-column tiles each). Oracle:
// each routed (column, expert) pair computed on the host by
// the exact arithmetic (ops/ggml/ggml_exact.h, whose decode, sums and rounding
// infernix_ggml_linear_test qualifies against an independent bignum oracle): x and h A16-encoded,
// gate/up rows rounded to BF16, SwiGLU, the down rows. Every output must be bit-equal to it in
// every configuration. The ggml-bf16-prefill route (run_bf16) is checked against the exact route and a
// binary64 chain within bounds instead.
//
// `--bench [tokens]` times one layer's expert call of a prefill chunk (512 experts, top-10, every
// record resident) per format pair instead.

#include "infernix/ops/offloaded_sparse_moe.h"
#include "core/ggml_format.h"
#include "ops/ggml/ggml_exact.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace moe = infernix::ops::offloaded_moe;
using infernix::DType;
using infernix::QType;
using infernix::Tensor;
using infernix::test::cuda_check;
namespace canon = infernix::ops::canon;
namespace ggml  = infernix::ggml;

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

std::uint32_t row_bytes(QType q, int k) { return k / infernix::ggml_block_elems(q) * infernix::ggml_block_bytes(q); }

// The rows' scale exponent fields, which choose the tiles' accumulators (ggml_tile.cuh row_bases):
// kModel 10..11 (the model's rows: almost all within one binade), kMixed 3..12 (spans of up to 9),
// both within the per-limb int64 accumulators' bound; kEdge 1 or 21 with the largest mantissa (span
// 20, |M| = 2047 * 2^20: the per-limb bound's limit, which every expert format's sum bound allows);
// kWide 1..30 (spans past 20: the int128 accumulators).
enum class Spread { kMixed, kModel, kEdge, kWide };

void random_rows(QType q, int rows, int k, std::mt19937& rng, std::uint8_t* out, Spread spread = Spread::kMixed) {
    const int blocks = k / static_cast<int>(infernix::ggml_block_elems(q)), size = static_cast<int>(infernix::ggml_block_bytes(q));
    for (int i = 0; i < rows * blocks * size; ++i) { out[i] = static_cast<std::uint8_t>(rng()); }
    for (int i = 0; i < rows * blocks; ++i) {
        std::uint8_t* b = out + static_cast<std::size_t>(i) * size;
        const unsigned field = spread == Spread::kModel  ? 10 + rng() % 2
                               : spread == Spread::kMixed ? 3 + rng() % 10
                               : spread == Spread::kEdge  ? (rng() % 2 == 0 ? 1 : 21)
                                                          : 1 + rng() % 30;
        const unsigned mant = spread == Spread::kEdge ? 0x3FFU : rng() & 0x3FFU;
        const auto w = static_cast<std::uint16_t>((rng() & 0x8000U) | mant | (field << 10));
        b[0] = static_cast<std::uint8_t>(w & 0xFF);
        b[1] = static_cast<std::uint8_t>(w >> 8);
    }
}

template <QType F> ggml::I128 dot(const std::uint8_t* row, const std::int32_t* x, int k) { return ggml::row_dot<F>(row, x, k); }

ggml::I128 row_dot(QType q, const std::uint8_t* row, const std::int32_t* x, int k) {
    switch (q) {
    case QType::GGML_IQ2_S: return dot<QType::GGML_IQ2_S>(row, x, k);
    case QType::GGML_IQ3_XXS: return dot<QType::GGML_IQ3_XXS>(row, x, k);
    case QType::GGML_IQ3_S: return dot<QType::GGML_IQ3_S>(row, x, k);
    case QType::GGML_IQ4_XS: return dot<QType::GGML_IQ4_XS>(row, x, k);
    case QType::GGML_IQ4_NL: return dot<QType::GGML_IQ4_NL>(row, x, k);
    case QType::GGML_Q2_0: return dot<QType::GGML_Q2_0>(row, x, k);
    default: throw std::logic_error("format");
    }
}

std::vector<std::int32_t> encode(const std::uint16_t* v, int n, int& emax) {
    emax = canon::a16_column_exponent(v, n);
    std::vector<std::int32_t> x(static_cast<std::size_t>(n), 0);
    if (emax != canon::kA16NonFinite) {
        for (int j = 0; j < n; ++j) { x[static_cast<std::size_t>(j)] = canon::a16_value(v[j], emax); }
    }
    return x;
}

// The exact expert on the host.
std::vector<std::uint16_t> expert(const std::uint8_t* record, QType gu, QType down, const std::uint16_t* xv) {
    const int H = moe::kHidden, I = moe::kIntermediate;
    const std::uint32_t gr = row_bytes(gu, H), dr = row_bytes(down, I);
    int ex = 0;
    const auto x = encode(xv, H, ex);
    std::vector<std::uint16_t> h(static_cast<std::size_t>(I));
    for (int i = 0; i < I; ++i) {
        const auto yg = ggml::output_bf16(row_dot(gu, record + static_cast<std::size_t>(i) * gr, x.data(), H), ex, gu);
        const auto yu = ggml::output_bf16(row_dot(gu, record + static_cast<std::size_t>(I + i) * gr, x.data(), H), ex, gu);
        h[static_cast<std::size_t>(i)] = canon::swiglu_bf16(yg, yu);
    }
    int eh = 0;
    const auto hx = encode(h.data(), I, eh);
    std::vector<std::uint16_t> y(static_cast<std::size_t>(H));
    const std::uint8_t* d = record + 2ULL * I * gr;
    for (int r = 0; r < H; ++r) {
        y[static_cast<std::size_t>(r)] = ggml::output_bf16(row_dot(down, d + static_cast<std::size_t>(r) * dr, hx.data(), I), eh, down);
    }
    return y;
}

void run(QType gu, QType down, int columns, std::mt19937& rng, Spread spread = Spread::kMixed) {
    constexpr int experts = 24, top_k = 4;
    const int H = moe::kHidden, I = moe::kIntermediate;
    const std::uint64_t record = 2ULL * I * row_bytes(gu, H) + static_cast<std::uint64_t>(H) * row_bytes(down, I);
    const std::uint64_t stride = (record + 4095) / 4096 * 4096;
    const std::uint64_t pitch  = stride + 8192; // the frame pool's pitch: a larger record elsewhere
    std::uint8_t* host = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host), stride * experts, cudaHostAllocMapped), "cudaHostAlloc");
    std::memset(host, 0, stride * experts);
    for (int e = 0; e < experts; ++e) {
        std::uint8_t* r = host + e * stride;
        random_rows(gu, 2 * I, H, rng, r, spread);
        random_rows(down, H, I, rng, r + 2ULL * I * row_bytes(gu, H), spread);
    }
    std::uint8_t* host_device = nullptr;
    cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_device), host, 0), "cudaHostGetDevicePointer");
    // Every third expert is resident in a frame.
    std::vector<std::int32_t> frames(experts, -1);
    int resident = 0;
    for (int e = 0; e < experts; e += 3) { frames[static_cast<std::size_t>(e)] = resident++; }
    std::uint8_t* d_frames_base = nullptr;
    cuda_check(cudaMalloc(&d_frames_base, pitch * std::max(resident, 1)), "cudaMalloc");
    for (int e = 0; e < experts; ++e) {
        if (frames[static_cast<std::size_t>(e)] >= 0) {
            cuda_check(cudaMemcpy(d_frames_base + frames[static_cast<std::size_t>(e)] * pitch, host + e * stride, stride,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
        }
    }
    auto* d_frames = device_copy(frames);

    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> logits(static_cast<std::size_t>(experts + 1) * columns);
    for (auto& v : logits) { v = n(rng); }
    if (columns > 64) {
        // Two favoured experts take most columns: jobs of several 64-column tiles, beside small ones.
        for (int t = 0; t < columns; ++t) {
            logits[static_cast<std::size_t>(t) * (experts + 1) + 1] += 3.0F;
            logits[static_cast<std::size_t>(t) * (experts + 1) + 2] += 2.0F;
        }
    }
    auto* d_logits = device_copy(logits);
    std::int32_t* d_ids = nullptr;
    float *d_weights = nullptr, *d_gate = nullptr;
    cuda_check(cudaMalloc(&d_ids, sizeof(std::int32_t) * top_k * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_weights, sizeof(float) * top_k * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_gate, sizeof(float) * columns), "cudaMalloc");
    infernix::ops::MoeRouting routing{Tensor(d_ids, DType::I32, {top_k, columns}), Tensor(d_weights, DType::FP32, {top_k, columns}),
                                      Tensor(d_gate, DType::FP32, {columns})};
    infernix::ops::moe_route(Tensor(d_logits, DType::FP32, {experts + 1, columns}), top_k, routing, nullptr);
    void* d_dispatch = nullptr;
    cuda_check(cudaMalloc(&d_dispatch, infernix::ops::moe_dispatch_bytes(experts, top_k * columns)), "cudaMalloc");
    auto dispatch = infernix::ops::carve_moe_dispatch(d_dispatch, experts, top_k * columns);

    std::vector<std::uint16_t> x(static_cast<std::size_t>(H) * columns);
    for (auto& v : x) {
        const float f = n(rng);
        std::uint32_t u;
        std::memcpy(&u, &f, 4);
        v = static_cast<std::uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
    }
    auto* d_x = device_copy(x);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(top_k) * columns);
    cuda_check(cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(H) * top_k * columns);
    {
        // The oracle over (token, slot) pairs on every host thread.
        std::atomic<int> next{0};
        const auto work = [&] {
            for (int i = next++; i < columns * top_k; i = next++) {
                const int t = i / top_k;
                const int e = ids[static_cast<std::size_t>(i)];
                const auto y = expert(host + e * stride, gu, down, &x[static_cast<std::size_t>(t) * H]);
                std::memcpy(&expected[static_cast<std::size_t>(i) * H], y.data(), H * sizeof(std::uint16_t));
            }
        };
        std::vector<std::thread> threads;
        for (unsigned w = 0; w < std::max(1U, std::thread::hardware_concurrency()); ++w) { threads.emplace_back(work); }
        for (auto& th : threads) { th.join(); }
    }

    const int max_jobs = std::min(experts, top_k * columns);
    std::uint8_t* d_staging = nullptr;
    cuda_check(cudaMalloc(&d_staging, pitch * 64), "cudaMalloc");
    void* d_workspace = nullptr;
    cuda_check(cudaMalloc(&d_workspace, infernix::ops::moe_experts_workspace_bytes(max_jobs, top_k * columns, columns,
                                                                                  moe::ExpertActivation::kA16)),
               "cudaMalloc");
    std::vector<moe::ExpertScales> scales(experts);
    auto* d_scales = device_copy(scales);
    std::uint16_t* d_out = nullptr;
    cuda_check(cudaMalloc(&d_out, expected.size() * sizeof(std::uint16_t)), "cudaMalloc");
    cudaStream_t fork_stream = nullptr;
    cudaEvent_t fork_events[2]{};
    cuda_check(cudaStreamCreateWithFlags(&fork_stream, cudaStreamNonBlocking), "stream");
    for (auto& e : fork_events) { cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event"); }

    // The CPU route (ops/offloaded_sparse_moe/cpu/ggml_expert.h) serving misses through the mapped
    // channel, as the Program's service does for decode and short calls.
    std::vector<moe::ExpertScales> no_scales(experts);
    const moe::GgmlExpertFormat format{.gate_up = gu, .down = down, .gate_row_bytes = row_bytes(gu, H),
                                       .down_row_bytes = row_bytes(down, I)};
    moe::CpuMissService service({{.records = host, .record_stride = stride, .scales = no_scales.data(), .ggml = format}},
                                {.workers = 4, .max_jobs = 8, .max_columns = 64, .pcie_divisor = 2, .cpus = {},
                                 .activation = moe::ExpertActivation::kA16});

    struct Config {
        int slots;
        bool fork;
        bool cpu;
        const char* name;
    };
    const Config configs[] = {{0, false, false, "zero-copy"},        {3, false, false, "3 staging slots"},
                              {64, false, false, "64 staging slots"}, {64, true, false, "forked"},
                              {64, false, true, "CPU-served misses"}, {64, true, true, "forked, CPU-served misses"}};
    for (const auto& config : configs) {
        if (config.fork && columns > moe::kMaxColumns) { continue; }
        if (config.cpu && columns > 64) { continue; }
        infernix::ops::moe_dispatch(routing, experts, dispatch, nullptr, nullptr);
        cuda_check(cudaMemset(d_out, 0xFF, expected.size() * sizeof(std::uint16_t)), "cudaMemset");
        infernix::ops::MoeExpertSource source{.frame_base    = d_frames_base,
                                              .frames        = d_frames,
                                              .host_records  = host_device,
                                              .record_stride = stride,
                                              .scales        = d_scales,
                                              .staging_base  = config.slots > 0 ? d_staging : nullptr,
                                              .staging_slots = config.slots};
        source.activation   = moe::ExpertActivation::kA16;
        source.frame_stride = pitch;
        source.ggml         = {.gate_up_format = gu,
                               .down_format    = down,
                               .gate_row_bytes = row_bytes(gu, H),
                               .down_row_bytes = row_bytes(down, I),
                               .record_bytes   = record};
        if (config.fork) {
            source.fork_stream    = fork_stream;
            source.fork_events[0] = fork_events[0];
            source.fork_events[1] = fork_events[1];
        }
        if (config.cpu) {
            service.wake();
            source.cpu = service.channel(0);
        }
        Tensor tx(d_x, DType::BF16, {H, columns});
        Tensor out(d_out, DType::BF16, {H, top_k * columns});
        const std::uint64_t served = service.served_experts();
        infernix::ops::moe_experts(tx, dispatch, source, top_k, max_jobs, d_workspace, out, nullptr);
        cuda_check(cudaDeviceSynchronize(), "moe_experts");
        if (config.cpu) {
            char none[128];
            std::snprintf(none, sizeof none, "ggml experts (%d, %d) at %d columns, %s: the CPU served no expert",
                          static_cast<int>(gu), static_cast<int>(down), columns, config.name);
            check(service.served_experts() > served, none);
        }
        std::vector<std::uint16_t> actual(expected.size());
        cuda_check(cudaMemcpy(actual.data(), d_out, actual.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
        std::size_t bad = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) { bad += actual[i] != expected[i]; }
        char what[160];
        std::snprintf(what, sizeof what, "ggml experts (%d, %d)%s at %d columns, %s: %zu of %zu outputs differ",
                      static_cast<int>(gu), static_cast<int>(down),
                      spread == Spread::kModel  ? " (model scales)"
                      : spread == Spread::kEdge ? " (edge scales)"
                      : spread == Spread::kWide ? " (wide scales)"
                                                : "",
                      columns, config.name, bad, actual.size());
        check(bad == 0, what);
    }
    cudaFree(d_out);
    cudaFree(d_scales);
    cudaFree(d_workspace);
    cudaFree(d_staging);
    cudaFree(d_x);
    cudaFree(d_dispatch);
    cudaFree(d_ids);
    cudaFree(d_weights);
    cudaFree(d_gate);
    cudaFree(d_logits);
    cudaFree(d_frames);
    cudaFree(d_frames_base);
    cudaFreeHost(host);
    cudaStreamDestroy(fork_stream);
    for (auto e : fork_events) { cudaEventDestroy(e); }
}

// One block's exact values in binary64 (llama.cpp's dequantize_row_* through ggml_exact.h's chunks):
// every element of a row of `k` elements.
void decode_row(QType q, const std::uint8_t* row, int k, double* out) {
    for (int i = 0; i < k / 8; ++i) {
        ggml::Chunk c{};
        switch (q) {
        case QType::GGML_IQ2_S: c = ggml::row_chunk<QType::GGML_IQ2_S>(row, i); break;
        case QType::GGML_IQ3_XXS: c = ggml::row_chunk<QType::GGML_IQ3_XXS>(row, i); break;
        case QType::GGML_IQ3_S: c = ggml::row_chunk<QType::GGML_IQ3_S>(row, i); break;
        case QType::GGML_IQ4_XS: c = ggml::row_chunk<QType::GGML_IQ4_XS>(row, i); break;
        case QType::GGML_IQ4_NL: c = ggml::row_chunk<QType::GGML_IQ4_NL>(row, i); break;
        case QType::GGML_Q2_0: c = ggml::row_chunk<QType::GGML_Q2_0>(row, i); break;
        default: throw std::logic_error("format");
        }
        std::int64_t mant;
        int shift;
        bool negative;
        ggml::f16_parts(c.d, mant, shift, negative);
        const double d = std::ldexp(static_cast<double>(negative ? -mant : mant), shift - 24 + ggml::kappa_log2(q));
        for (int j = 0; j < 8; ++j) { out[8 * i + j] = d * c.sub * c.code[j]; }
    }
}

double bf16_value(std::uint16_t bits) {
    const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

std::uint16_t bf16_round(double v) {
    const auto f = static_cast<float>(v);
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

// The ggml-bf16-prefill route (WideA16::kGgmlBf16) of a call wide enough for its planes
// (kGgmlBf16MinColumns): 64 experts, top-4, some resident, the rest read zero-copy in one pass or staged
// through 24 slots (passes of 24 jobs, each one sub-batch), a few rarely routed experts left to
// the narrow kernels. Its arithmetic is not the exact route's (x and h as stored, FP32 sums), so the
// exact route of the same call (bit-equal to the host oracle at the other widths) is the control: per
// output column the normwise difference must stay within 2^-6 and at least 90 % of the outputs must
// be equal. A sample of 48 (token, slot) pairs is checked against a binary64 chain with the route's
// semantic boundaries (gate and up rounded to BF16, swiglu_bf16, BF16 h, the down rows): within
// 2^-7 of sum |w h| plus half a BF16 ulp per output.
void run_bf16(QType gu, QType down, int columns, std::mt19937& rng) {
    constexpr int experts = 64, top_k = 4;
    const int H = moe::kHidden, I = moe::kIntermediate;
    const std::uint32_t gr = row_bytes(gu, H), dr = row_bytes(down, I);
    const std::uint64_t record = 2ULL * I * gr + static_cast<std::uint64_t>(H) * dr;
    const std::uint64_t stride = (record + 4095) / 4096 * 4096;
    std::uint8_t* host = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host), stride * experts, cudaHostAllocMapped), "cudaHostAlloc");
    std::memset(host, 0, stride * experts);
    for (int e = 0; e < experts; ++e) {
        random_rows(gu, 2 * I, H, rng, host + e * stride, Spread::kModel);
        random_rows(down, H, I, rng, host + e * stride + 2ULL * I * gr, Spread::kModel);
    }
    std::uint8_t* host_device = nullptr;
    cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_device), host, 0), "cudaHostGetDevicePointer");
    std::vector<std::int32_t> frames(experts, -1);
    int resident = 0;
    for (int e = 0; e < experts; e += 3) { frames[static_cast<std::size_t>(e)] = resident++; }
    std::uint8_t* d_frames_base = nullptr;
    cuda_check(cudaMalloc(&d_frames_base, stride * resident), "cudaMalloc");
    for (int e = 0; e < experts; ++e) {
        if (frames[static_cast<std::size_t>(e)] >= 0) {
            cuda_check(cudaMemcpy(d_frames_base + frames[static_cast<std::size_t>(e)] * stride, host + e * stride, stride,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
        }
    }
    auto* d_frames = device_copy(frames);
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> logits(static_cast<std::size_t>(experts + 1) * columns);
    for (auto& v : logits) { v = n(rng); }
    for (int t = 0; t < columns; ++t) {
        for (int e = 60; e < 64; ++e) { logits[static_cast<std::size_t>(t) * (experts + 1) + e] -= 7.0F; } // narrow jobs
        logits[static_cast<std::size_t>(t) * (experts + 1) + 1] += 2.0F; // a job of many tiles
    }
    auto* d_logits = device_copy(logits);
    std::int32_t* d_ids = nullptr;
    float *d_weights = nullptr, *d_gate = nullptr;
    cuda_check(cudaMalloc(&d_ids, sizeof(std::int32_t) * top_k * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_weights, sizeof(float) * top_k * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_gate, sizeof(float) * columns), "cudaMalloc");
    infernix::ops::MoeRouting routing{Tensor(d_ids, DType::I32, {top_k, columns}), Tensor(d_weights, DType::FP32, {top_k, columns}),
                                      Tensor(d_gate, DType::FP32, {columns})};
    infernix::ops::moe_route(Tensor(d_logits, DType::FP32, {experts + 1, columns}), top_k, routing, nullptr);
    void* d_dispatch = nullptr;
    cuda_check(cudaMalloc(&d_dispatch, infernix::ops::moe_dispatch_bytes(experts, top_k * columns)), "cudaMalloc");
    auto dispatch = infernix::ops::carve_moe_dispatch(d_dispatch, experts, top_k * columns);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(H) * columns);
    for (auto& v : x) { v = bf16_round(n(rng)); }
    auto* d_x = device_copy(x);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(top_k) * columns);
    cuda_check(cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
    const int max_jobs  = std::min(experts, top_k * columns);
    const std::size_t outputs = static_cast<std::size_t>(H) * top_k * columns;
    std::uint8_t* d_staging = nullptr;
    cuda_check(cudaMalloc(&d_staging, stride * 64), "cudaMalloc");
    void* d_workspace = nullptr;
    cuda_check(cudaMalloc(&d_workspace, infernix::ops::moe_experts_workspace_bytes(max_jobs, top_k * columns, columns,
                                                                                  moe::ExpertActivation::kA16,
                                                                                  moe::WideA16::kGgmlBf16)),
               "cudaMalloc");
    std::vector<moe::ExpertScales> scales(experts);
    auto* d_scales = device_copy(scales);
    std::uint16_t* d_out = nullptr;
    cuda_check(cudaMalloc(&d_out, outputs * sizeof(std::uint16_t)), "cudaMalloc");
    const auto call = [&](moe::WideA16 route, int slots) {
        infernix::ops::moe_dispatch(routing, experts, dispatch, nullptr, nullptr);
        cuda_check(cudaMemset(d_out, 0xFF, outputs * sizeof(std::uint16_t)), "cudaMemset");
        infernix::ops::MoeExpertSource source{.frame_base = d_frames_base, .frames = d_frames, .host_records = host_device,
                                              .record_stride = stride, .scales = d_scales,
                                              .staging_base = slots > 0 ? d_staging : nullptr, .staging_slots = slots};
        source.activation   = moe::ExpertActivation::kA16;
        source.wide_a16     = route;
        source.frame_stride = stride;
        source.ggml         = {.gate_up_format = gu, .down_format = down, .gate_row_bytes = gr, .down_row_bytes = dr,
                               .record_bytes = record};
        Tensor tx(d_x, DType::BF16, {H, columns});
        Tensor out(d_out, DType::BF16, {H, top_k * columns});
        infernix::ops::moe_experts(tx, dispatch, source, top_k, max_jobs, d_workspace, out, nullptr);
        cuda_check(cudaDeviceSynchronize(), "moe_experts");
        std::vector<std::uint16_t> got(outputs);
        cuda_check(cudaMemcpy(got.data(), d_out, outputs * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
        return got;
    };
    const auto exact = call(moe::WideA16::kExact, 0);
    for (const int slots : {0, 24}) {
        const auto fast = call(moe::WideA16::kGgmlBf16, slots);
        std::size_t same = 0, bad_columns = 0;
        double worst = 0;
        for (int c = 0; c < top_k * columns; ++c) {
            double diff = 0, norm = 0;
            for (int r = 0; r < H; ++r) {
                const std::size_t i = static_cast<std::size_t>(c) * H + r;
                same += fast[i] == exact[i];
                const double a = bf16_value(fast[i]), b = bf16_value(exact[i]);
                diff += (a - b) * (a - b);
                norm += b * b;
            }
            const double rel = norm > 0 ? std::sqrt(diff / norm) : std::sqrt(diff);
            worst = std::max(worst, std::isnan(rel) ? 1e30 : rel);
            bad_columns += !(rel <= 1.0 / 64);
        }
        char what[200];
        std::snprintf(what, sizeof what, "ggml bf16 experts (%d, %d) x %d columns, %d slots: %zu output columns beyond 2^-6 of the exact route",
                      static_cast<int>(gu), static_cast<int>(down), columns, slots, bad_columns);
        check(bad_columns == 0, what);
        std::snprintf(what, sizeof what, "ggml bf16 experts (%d, %d) x %d columns, %d slots: only %.1f %% of outputs equal the exact route's",
                      static_cast<int>(gu), static_cast<int>(down), columns, slots, 100.0 * static_cast<double>(same) / outputs);
        check(same * 10 >= outputs * 9, what);
        std::printf("info ggml bf16 experts (%d, %d) x %d columns, %d slots: %.2f %% of outputs equal the exact route's, worst column %.2e\n",
                    static_cast<int>(gu), static_cast<int>(down), columns, slots, 100.0 * static_cast<double>(same) / outputs, worst);
        if (slots != 0) { continue; }
        // The binary64 chain on a sample of (token, slot) pairs.
        std::vector<double> wg(static_cast<std::size_t>(H)), wu(static_cast<std::size_t>(H)), wd(static_cast<std::size_t>(I));
        std::size_t sample_bad = 0;
        for (int s = 0; s < 48; ++s) {
            const int pair = static_cast<int>(rng() % static_cast<unsigned>(top_k * columns));
            const int t = pair / top_k, e = ids[static_cast<std::size_t>(pair)];
            const std::uint8_t* rec = host + e * stride;
            std::vector<std::uint16_t> h(static_cast<std::size_t>(I));
            for (int i = 0; i < I; ++i) {
                decode_row(gu, rec + static_cast<std::size_t>(i) * gr, H, wg.data());
                decode_row(gu, rec + static_cast<std::size_t>(I + i) * gr, H, wu.data());
                double g = 0, u = 0;
                for (int j = 0; j < H; ++j) {
                    const double xv = bf16_value(x[static_cast<std::size_t>(t) * H + j]);
                    g += wg[static_cast<std::size_t>(j)] * xv;
                    u += wu[static_cast<std::size_t>(j)] * xv;
                }
                h[static_cast<std::size_t>(i)] = canon::swiglu_bf16(bf16_round(g), bf16_round(u));
            }
            for (int r = 0; r < H; ++r) {
                decode_row(down, rec + 2ULL * I * gr + static_cast<std::size_t>(r) * dr, I, wd.data());
                double y = 0, mag = 0;
                for (int i = 0; i < I; ++i) {
                    const double p = wd[static_cast<std::size_t>(i)] * bf16_value(h[static_cast<std::size_t>(i)]);
                    y += p;
                    mag += std::fabs(p);
                }
                const double got = bf16_value(fast[static_cast<std::size_t>(pair) * H + r]);
                sample_bad += !(std::fabs(got - y) <= mag / 128 + std::ldexp(std::fabs(y), -8));
            }
        }
        std::snprintf(what, sizeof what, "ggml bf16 experts (%d, %d) x %d columns: %zu sampled outputs beyond the binary64 chain's bound",
                      static_cast<int>(gu), static_cast<int>(down), columns, sample_bad);
        check(sample_bad == 0, what);
    }
    cudaFree(d_out);
    cudaFree(d_scales);
    cudaFree(d_workspace);
    cudaFree(d_staging);
    cudaFree(d_x);
    cudaFree(d_dispatch);
    cudaFree(d_ids);
    cudaFree(d_weights);
    cudaFree(d_gate);
    cudaFree(d_logits);
    cudaFree(d_frames);
    cudaFree(d_frames_base);
    cudaFreeHost(host);
}

// --bench: one layer call of the real geometry (512 experts, top-10; 4,096 tokens, a prefill chunk of
// about 80 columns per expert, or a decode or verification width) with every record resident and
// the model's job bound, timed over repeated moe_experts calls.
void bench(QType gu, QType down, int columns, std::mt19937& rng, moe::WideA16 route) {
    constexpr int experts = 512, top_k = 10, reps = 10;
    const int H = moe::kHidden, I = moe::kIntermediate;
    const std::uint64_t record = 2ULL * I * row_bytes(gu, H) + static_cast<std::uint64_t>(H) * row_bytes(down, I);
    const std::uint64_t stride = (record + 4095) / 4096 * 4096;
    std::vector<std::uint8_t> host(stride * experts);
    for (int e = 0; e < experts; ++e) {
        std::uint8_t* r = host.data() + e * stride;
        random_rows(gu, 2 * I, H, rng, r, Spread::kModel);
        random_rows(down, H, I, rng, r + 2ULL * I * row_bytes(gu, H), Spread::kModel);
    }
    auto* d_frames_base = device_copy(host);
    std::vector<std::int32_t> frames(experts);
    for (int e = 0; e < experts; ++e) { frames[static_cast<std::size_t>(e)] = e; }
    auto* d_frames = device_copy(frames);
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> logits(static_cast<std::size_t>(experts + 1) * columns);
    for (auto& v : logits) { v = n(rng); }
    auto* d_logits = device_copy(logits);
    std::int32_t* d_ids = nullptr;
    float *d_weights = nullptr, *d_gate = nullptr;
    cuda_check(cudaMalloc(&d_ids, sizeof(std::int32_t) * top_k * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_weights, sizeof(float) * top_k * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_gate, sizeof(float) * columns), "cudaMalloc");
    infernix::ops::MoeRouting routing{Tensor(d_ids, DType::I32, {top_k, columns}), Tensor(d_weights, DType::FP32, {top_k, columns}),
                                      Tensor(d_gate, DType::FP32, {columns})};
    infernix::ops::moe_route(Tensor(d_logits, DType::FP32, {experts + 1, columns}), top_k, routing, nullptr);
    void* d_dispatch = nullptr;
    cuda_check(cudaMalloc(&d_dispatch, infernix::ops::moe_dispatch_bytes(experts, top_k * columns)), "cudaMalloc");
    auto dispatch = infernix::ops::carve_moe_dispatch(d_dispatch, experts, top_k * columns);
    infernix::ops::moe_dispatch(routing, experts, dispatch, nullptr, nullptr);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(H) * columns);
    for (auto& v : x) {
        const float f = n(rng);
        std::uint32_t u;
        std::memcpy(&u, &f, 4);
        v = static_cast<std::uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
    }
    auto* d_x = device_copy(x);
    void* d_workspace = nullptr;
    cuda_check(cudaMalloc(&d_workspace, infernix::ops::moe_experts_workspace_bytes(experts, top_k * columns, columns,
                                                                                  moe::ExpertActivation::kA16, route)),
               "cudaMalloc");
    std::vector<moe::ExpertScales> scales(experts);
    auto* d_scales = device_copy(scales);
    std::uint16_t* d_out = nullptr;
    cuda_check(cudaMalloc(&d_out, sizeof(std::uint16_t) * H * top_k * columns), "cudaMalloc");
    // Every expert is resident, so the host records (required by the source) are never read.
    infernix::ops::MoeExpertSource source{.frame_base = d_frames_base, .frames = d_frames, .host_records = d_frames_base,
                                          .record_stride = stride, .scales = d_scales, .staging_base = nullptr,
                                          .staging_slots = 0};
    source.activation   = moe::ExpertActivation::kA16;
    source.frame_stride = stride;
    source.wide_a16     = route;
    source.ggml         = {.gate_up_format = gu, .down_format = down, .gate_row_bytes = row_bytes(gu, H),
                           .down_row_bytes = row_bytes(down, I), .record_bytes = record};
    Tensor tx(d_x, DType::BF16, {H, columns});
    Tensor out(d_out, DType::BF16, {H, top_k * columns});
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "event");
    cuda_check(cudaEventCreate(&stop), "event");
    // The model's job bound: at most one job per routed (column, slot) pair.
    const int max_jobs = std::min(experts, top_k * columns);
    infernix::ops::moe_experts(tx, dispatch, source, top_k, max_jobs, d_workspace, out, nullptr); // warm-up
    cuda_check(cudaEventRecord(start), "event");
    for (int r = 0; r < reps; ++r) { infernix::ops::moe_experts(tx, dispatch, source, top_k, max_jobs, d_workspace, out, nullptr); }
    cuda_check(cudaEventRecord(stop), "event");
    cuda_check(cudaEventSynchronize(stop), "event");
    float ms = 0.0F;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "event");
    ms /= reps;
    const double macs = static_cast<double>(top_k) * columns * (2.0 * I * H + static_cast<double>(H) * I);
    std::printf("experts (%d, %d) x %d tokens, %s: %8.3f ms per layer call, %6.1f effective TOPS\n", static_cast<int>(gu),
                static_cast<int>(down), columns, route == moe::WideA16::kGgmlBf16 ? "bf16" : "exact", ms,
                2.0 * macs / (ms * 1e-3) / 1e12);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_out);
    cudaFree(d_scales);
    cudaFree(d_workspace);
    cudaFree(d_x);
    cudaFree(d_dispatch);
    cudaFree(d_ids);
    cudaFree(d_weights);
    cudaFree(d_gate);
    cudaFree(d_logits);
    cudaFree(d_frames);
    cudaFree(d_frames_base);
}

} // namespace

int main(int argc, char** argv) {
    std::mt19937 rng(20261010);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        // --bench [tokens] [bf16]: the exact route and, with bf16, the ggml-bf16-prefill route too.
        const int columns = argc > 2 ? std::atoi(argv[2]) : 4096;
        const bool bf16   = argc > 3 && std::strcmp(argv[3], "bf16") == 0;
        for (const auto& [gu, down] : {std::pair{QType::GGML_IQ2_S, QType::GGML_IQ4_NL}, std::pair{QType::GGML_IQ3_XXS, QType::GGML_IQ4_NL},
                                       std::pair{QType::GGML_IQ3_S, QType::GGML_IQ4_NL}, std::pair{QType::GGML_IQ3_S, QType::GGML_Q2_0}}) {
            try {
                bench(gu, down, columns, rng, moe::WideA16::kExact);
                if (bf16) { bench(gu, down, columns, rng, moe::WideA16::kGgmlBf16); }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "bench: %s\n", e.what());
                return 1;
            }
        }
        return 0;
    }
    const std::pair<QType, QType> pairs[] = {
        {QType::GGML_IQ2_S, QType::GGML_IQ4_NL}, {QType::GGML_IQ2_S, QType::GGML_Q2_0},
        {QType::GGML_IQ3_XXS, QType::GGML_IQ4_NL}, {QType::GGML_IQ3_XXS, QType::GGML_Q2_0},
        {QType::GGML_IQ3_S, QType::GGML_IQ4_NL}, {QType::GGML_IQ3_S, QType::GGML_Q2_0},
        {QType::GGML_IQ4_XS, QType::GGML_IQ4_NL}, {QType::GGML_IQ4_XS, QType::GGML_Q2_0}};
    for (const auto& [gu, down] : pairs) {
        // 9, 40 and 300 columns: calls of the wide route (tensor-core tiles for jobs over 8 columns).
        for (const int columns : {1, 3, 9, 40, 300}) { run(gu, down, columns, rng); }
        // The model's scale spread, the per-limb bound's limit, and spreads past it (int128).
        for (const int columns : {40, 300}) {
            run(gu, down, columns, rng, Spread::kModel);
            run(gu, down, columns, rng, Spread::kEdge);
            run(gu, down, columns, rng, Spread::kWide);
        }
        // The ggml-bf16-prefill route at a width that carves its planes (a partial last tile).
        run_bf16(gu, down, 14336 + 37, rng); // wide_expert.h kGgmlBf16MinColumns = 14336
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d ggml expert checks failed\n", g_failures);
        return 1;
    }
    std::printf("ggml experts: every format pair, width and placement bit-equal to the exact host expert; the "
                "ggml-bf16-prefill route within its bounds of the exact route and a binary64 chain\n");
    return 0;
}
