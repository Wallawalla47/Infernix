// GGUF (ggml) expert banks on the 8-bit activation route (--a8-activations, MoeGgmlBank::a8; LOSSY, opt-in;
// ops/ggml/ggml_a8.h) through the layer route of offloaded_sparse_moe (moe_route, moe_dispatch, moe_experts).
//
// Records are random ggml blocks (finite binary16 scales) in a pinned host bank, every third expert copied into
// a device frame whose pitch exceeds the record stride. Misses are read zero-copy, staged through 3 or 64 slots,
// forked (and split), or computed by the CPU route through the miss service with its 8-bit functions. Calls of
// 1, 3 and 8 columns cover decode and verification, 9, 40 and 128 columns the route's widest calls (every job on
// the 8-bit kernels, whatever its column count). Oracle: the CPU route's scalar 8-bit expert
// (ggml_a8_expert_forward, CpuIsa::kScalar), which test_offloaded_moe_ggml_a8_cpu qualifies against a binary64
// oracle; every output must be bit-equal to it in every configuration. A call of 200 columns is past the route
// and must return the exact route's bits (the exact host arithmetic of ops/ggml/ggml_exact.h).

#include "ops/op_family.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "core/ggml_format.h"
#include "ops/ggml/ggml_exact.h"
#include "ops/offloaded_sparse_moe/cpu/ggml_expert.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
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

void random_rows(QType q, int rows, int k, std::mt19937& rng, std::uint8_t* out) {
    const int blocks = k / static_cast<int>(infernix::ggml_block_elems(q)), size = static_cast<int>(infernix::ggml_block_bytes(q));
    for (int i = 0; i < rows * blocks * size; ++i) { out[i] = static_cast<std::uint8_t>(rng()); }
    for (int i = 0; i < rows * blocks; ++i) {
        std::uint8_t* b = out + static_cast<std::size_t>(i) * size;
        const unsigned field = 3 + rng() % 10; // finite scales of mixed binades
        const auto w = static_cast<std::uint16_t>((rng() & 0x8000U) | (rng() & 0x3FFU) | (field << 10));
        b[0] = static_cast<std::uint8_t>(w & 0xFF);
        b[1] = static_cast<std::uint8_t>(w >> 8);
    }
}

// The exact route on the host (the control of calls past the 8-bit route).
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

std::vector<std::uint16_t> exact_expert(const std::uint8_t* record, QType gu, QType down, const std::uint16_t* xv) {
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

// The 8-bit route on the host: the CPU route's scalar expert (one column).
std::vector<std::uint16_t> a8_expert(const std::uint8_t* record, const moe::GgmlExpertFormat& format, const std::uint16_t* xv) {
    std::vector<std::uint16_t> y(static_cast<std::size_t>(moe::kHidden));
    const std::uint16_t* xs[] = {xv};
    std::uint16_t* ys[]       = {y.data()};
    moe::ggml_a8_expert_forward(moe::CpuIsa::kScalar, record, format, 1, xs, ys);
    return y;
}

void run(QType gu, QType down, int columns, std::mt19937& rng) {
    constexpr int experts = 24, top_k = 4;
    const int H = moe::kHidden, I = moe::kIntermediate;
    const bool a8_route = columns <= 128;
    const std::uint64_t record = 2ULL * I * row_bytes(gu, H) + static_cast<std::uint64_t>(H) * row_bytes(down, I);
    const std::uint64_t stride = (record + 4095) / 4096 * 4096;
    const std::uint64_t pitch  = stride + 8192;
    std::uint8_t* host = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host), stride * experts, cudaHostAllocMapped), "cudaHostAlloc");
    std::memset(host, 0, stride * experts);
    for (int e = 0; e < experts; ++e) {
        std::uint8_t* r = host + e * stride;
        random_rows(gu, 2 * I, H, rng, r);
        random_rows(down, H, I, rng, r + 2ULL * I * row_bytes(gu, H));
    }
    std::uint8_t* host_device = nullptr;
    cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_device), host, 0), "cudaHostGetDevicePointer");
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
    if (columns > 8) {
        // Two favoured experts take most columns: jobs of more than eight columns (passes of eight).
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

    // x with one large value per column (the activations' outliers) and a zero slice in column 0.
    std::vector<std::uint16_t> x(static_cast<std::size_t>(H) * columns);
    for (auto& v : x) {
        const float f = n(rng);
        std::uint32_t u;
        std::memcpy(&u, &f, 4);
        v = static_cast<std::uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
    }
    for (int t = 0; t < columns; ++t) { x[static_cast<std::size_t>(t) * H + static_cast<std::size_t>(rng() % H)] = 0x4300; } // 128
    for (int j = 0; j < 32; ++j) { x[static_cast<std::size_t>(64 + j)] = 0; }
    auto* d_x = device_copy(x);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(top_k) * columns);
    cuda_check(cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
    const moe::GgmlExpertFormat format{.gate_up = gu, .down = down, .gate_row_bytes = row_bytes(gu, H),
                                       .down_row_bytes = row_bytes(down, I)};
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(H) * top_k * columns);
    {
        std::atomic<int> next{0};
        const auto work = [&] {
            for (int i = next++; i < columns * top_k; i = next++) {
                const int t = i / top_k;
                const int e = ids[static_cast<std::size_t>(i)];
                const std::uint16_t* xv = &x[static_cast<std::size_t>(t) * H];
                const auto y = a8_route ? a8_expert(host + e * stride, format, xv) : exact_expert(host + e * stride, gu, down, xv);
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
    cuda_check(cudaMalloc(&d_workspace, infernix::ops::moe_experts_workspace_bytes(moe::kFlashNextShape, max_jobs, top_k * columns,
                                                                                  columns, moe::ExpertActivation::kA16)),
               "cudaMalloc");
    std::vector<moe::ExpertScales> scales(experts);
    auto* d_scales = device_copy(scales);
    std::uint16_t* d_out = nullptr;
    cuda_check(cudaMalloc(&d_out, expected.size() * sizeof(std::uint16_t)), "cudaMalloc");
    cudaStream_t fork_stream = nullptr, split_stream = nullptr;
    cudaEvent_t fork_events[2]{}, split_events[2]{};
    cuda_check(cudaStreamCreateWithFlags(&fork_stream, cudaStreamNonBlocking), "stream");
    cuda_check(cudaStreamCreateWithFlags(&split_stream, cudaStreamNonBlocking), "stream");
    for (auto& e : fork_events) { cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event"); }
    for (auto& e : split_events) { cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event"); }

    std::vector<moe::ExpertScales> no_scales(experts);
    moe::CpuMissService service({{.records = host, .record_stride = stride, .scales = no_scales.data(), .ggml = format}},
                                {.workers = 4, .max_jobs = 8, .max_columns = 64, .pcie_divisor = 2, .cpus = {},
                                 .activation = moe::ExpertActivation::kA16, .ggml_a8 = true});

    struct Config {
        int slots;
        bool fork;
        bool cpu;
        const char* name;
        bool split = false;
    };
    const Config configs[] = {{0, false, false, "zero-copy"},
                              {3, false, false, "3 staging slots"},
                              {64, false, false, "64 staging slots"},
                              {64, true, false, "forked"},
                              {64, true, false, "forked, split staging", true},
                              {64, false, true, "CPU-served misses"},
                              {64, true, true, "forked, CPU-served misses"},
                              {64, true, true, "forked, split staging, CPU-served misses", true}};
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
                               .record_bytes   = record,
                               .a8             = true};
        if (config.fork) {
            source.fork_stream    = fork_stream;
            source.fork_events[0] = fork_events[0];
            source.fork_events[1] = fork_events[1];
        }
        if (config.split) {
            source.split_stream    = split_stream;
            source.split_events[0] = split_events[0];
            source.split_events[1] = split_events[1];
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
            std::snprintf(none, sizeof none, "a8 ggml experts (%d, %d) at %d columns, %s: the CPU served no expert",
                          static_cast<int>(gu), static_cast<int>(down), columns, config.name);
            check(service.served_experts() > served, none);
        }
        std::vector<std::uint16_t> actual(expected.size());
        cuda_check(cudaMemcpy(actual.data(), d_out, actual.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
        std::size_t bad = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) { bad += actual[i] != expected[i]; }
        char what[192];
        std::snprintf(what, sizeof what, "a8 ggml experts (%d, %d) at %d columns (%s), %s: %zu of %zu outputs differ",
                      static_cast<int>(gu), static_cast<int>(down), columns, a8_route ? "8-bit route" : "exact route",
                      config.name, bad, actual.size());
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
    cudaStreamDestroy(split_stream);
    for (auto e : fork_events) { cudaEventDestroy(e); }
    for (auto e : split_events) { cudaEventDestroy(e); }
}

// `--bench`: one layer's expert call at decode and verification widths (512 experts, top-10, every record
// resident in frames), the exact route against the 8-bit route, per format pair. Eight routings rotate so a
// call does not find the previous call's records in L2 (each touches 10-80 experts of ~2 MB).
void bench(QType gu, QType down, int columns, std::mt19937& rng) {
    constexpr int experts = 512, top_k = 10, kRoutings = 8, kWarmup = 20, kIterations = 200;
    const int H = moe::kHidden, I = moe::kIntermediate;
    const std::uint64_t record = 2ULL * I * row_bytes(gu, H) + static_cast<std::uint64_t>(H) * row_bytes(down, I);
    const std::uint64_t stride = (record + 4095) / 4096 * 4096;
    std::vector<std::uint8_t> host(stride * experts, 0);
    for (int e = 0; e < experts; ++e) {
        random_rows(gu, 2 * I, H, rng, host.data() + e * stride);
        random_rows(down, H, I, rng, host.data() + e * stride + 2ULL * I * row_bytes(gu, H));
    }
    auto* d_frames_base = device_copy(host);
    std::vector<std::int32_t> frames(experts);
    for (int e = 0; e < experts; ++e) { frames[static_cast<std::size_t>(e)] = e; }
    auto* d_frames = device_copy(frames);
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<infernix::ops::MoeDispatch> dispatches;
    std::vector<void*> buffers;
    for (int r = 0; r < kRoutings; ++r) {
        std::vector<float> logits(static_cast<std::size_t>(experts + 1) * columns);
        for (auto& v : logits) { v = n(rng); }
        auto* d_logits = device_copy(logits);
        std::int32_t* d_ids = nullptr;
        float *d_weights = nullptr, *d_gate = nullptr;
        cuda_check(cudaMalloc(&d_ids, sizeof(std::int32_t) * top_k * columns), "cudaMalloc");
        cuda_check(cudaMalloc(&d_weights, sizeof(float) * top_k * columns), "cudaMalloc");
        cuda_check(cudaMalloc(&d_gate, sizeof(float) * columns), "cudaMalloc");
        infernix::ops::MoeRouting routing{Tensor(d_ids, DType::I32, {top_k, columns}),
                                          Tensor(d_weights, DType::FP32, {top_k, columns}), Tensor(d_gate, DType::FP32, {columns})};
        infernix::ops::moe_route(Tensor(d_logits, DType::FP32, {experts + 1, columns}), top_k, routing, nullptr);
        void* d_dispatch = nullptr;
        cuda_check(cudaMalloc(&d_dispatch, infernix::ops::moe_dispatch_bytes(experts, top_k * columns)), "cudaMalloc");
        dispatches.push_back(infernix::ops::carve_moe_dispatch(d_dispatch, experts, top_k * columns));
        infernix::ops::moe_dispatch(routing, experts, dispatches.back(), nullptr, nullptr);
        buffers.insert(buffers.end(), {d_logits, d_ids, d_weights, d_gate, d_dispatch});
    }
    std::vector<std::uint16_t> x(static_cast<std::size_t>(H) * columns);
    for (auto& v : x) { v = static_cast<std::uint16_t>(0x3C00U + rng() % 0x0400U) | static_cast<std::uint16_t>((rng() & 1U) << 15); }
    auto* d_x = device_copy(x);
    const int max_jobs = std::min(experts, top_k * columns);
    void* d_workspace = nullptr;
    cuda_check(cudaMalloc(&d_workspace, infernix::ops::moe_experts_workspace_bytes(moe::kFlashNextShape, max_jobs, top_k * columns,
                                                                                  columns, moe::ExpertActivation::kA16)),
               "cudaMalloc");
    std::vector<moe::ExpertScales> scales(experts);
    auto* d_scales = device_copy(scales);
    std::uint16_t* d_out = nullptr;
    cuda_check(cudaMalloc(&d_out, sizeof(std::uint16_t) * H * top_k * columns), "cudaMalloc");
    cudaEvent_t start = nullptr, stop = nullptr;
    cuda_check(cudaEventCreate(&start), "event");
    cuda_check(cudaEventCreate(&stop), "event");
    double ms[2] = {0.0, 0.0};
    for (int a8 = 0; a8 < 2; ++a8) {
        infernix::ops::MoeExpertSource source{.frame_base = d_frames_base, .frames = d_frames, .host_records = d_frames_base,
                                              .record_stride = stride, .scales = d_scales};
        source.activation = moe::ExpertActivation::kA16;
        source.ggml       = {.gate_up_format = gu, .down_format = down, .gate_row_bytes = row_bytes(gu, H),
                             .down_row_bytes = row_bytes(down, I), .record_bytes = (record + 15) / 16 * 16, .a8 = a8 != 0};
        Tensor tx(d_x, DType::BF16, {H, columns});
        Tensor out(d_out, DType::BF16, {H, top_k * columns});
        for (int i = 0; i < kWarmup + kIterations; ++i) {
            if (i == kWarmup) { cuda_check(cudaEventRecord(start, nullptr), "record"); }
            infernix::ops::moe_experts(tx, dispatches[static_cast<std::size_t>(i % kRoutings)], source, top_k, max_jobs, d_workspace,
                                       out, nullptr);
        }
        cuda_check(cudaEventRecord(stop, nullptr), "record");
        cuda_check(cudaEventSynchronize(stop), "sync");
        float elapsed = 0.0F;
        cuda_check(cudaEventElapsedTime(&elapsed, start, stop), "elapsed");
        ms[a8] = elapsed / kIterations;
    }
    std::printf("experts (%d, %d) x %d columns: exact %.4f ms, 8-bit %.4f ms per layer call (%.2fx)\n", static_cast<int>(gu),
                static_cast<int>(down), columns, ms[0], ms[1], ms[0] / ms[1]);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_out);
    cudaFree(d_scales);
    cudaFree(d_workspace);
    cudaFree(d_x);
    for (void* p : buffers) { cudaFree(p); }
    cudaFree(d_frames);
    cudaFree(d_frames_base);
}

} // namespace

INFERNIX_OP_TEST(offloaded_moe_ggml_a8) {
    std::mt19937 rng(20261027);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        for (const auto& [gu, down] : {std::pair{QType::GGML_IQ3_S, QType::GGML_IQ4_NL}, std::pair{QType::GGML_IQ3_XXS, QType::GGML_IQ4_NL},
                                       std::pair{QType::GGML_IQ2_S, QType::GGML_Q2_0}, std::pair{QType::GGML_IQ4_XS, QType::GGML_IQ4_NL}}) {
            for (const int columns : {1, 2, 4, 8, 16}) { bench(gu, down, columns, rng); }
        }
        return 0;
    }
    const std::pair<QType, QType> pairs[] = {
        {QType::GGML_IQ2_S, QType::GGML_IQ4_NL},   {QType::GGML_IQ2_S, QType::GGML_Q2_0},
        {QType::GGML_IQ3_XXS, QType::GGML_IQ4_NL}, {QType::GGML_IQ3_XXS, QType::GGML_Q2_0},
        {QType::GGML_IQ3_S, QType::GGML_IQ4_NL},   {QType::GGML_IQ3_S, QType::GGML_Q2_0},
        {QType::GGML_IQ4_XS, QType::GGML_IQ4_NL},  {QType::GGML_IQ4_XS, QType::GGML_Q2_0}};
    for (const auto& [gu, down] : pairs) {
        for (const int columns : {1, 3, 9, 128}) { run(gu, down, columns, rng); }
    }
    for (const int columns : {8, 40, 200}) { run(QType::GGML_IQ3_S, QType::GGML_IQ4_NL, columns, rng); }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("offloaded_moe_ggml_a8: ok\n");
    return 0;
}
