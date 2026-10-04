// The layer route of offloaded_sparse_moe (moe_route, moe_dispatch, moe_experts) against the CPU
// engine, with expert records split between device frames and the pinned host bank and misses
// read zero-copy, staged through 1, 3 or 64 device slots (one or several staging passes), or
// served by the host expert engine through the CPU miss channel
// (docs/maintainer/qwen3_8-flash-next-design.md §8.6, §10.3, §16.2).
//
// Oracle: the CPU engine's output of each routed (column, expert) pair, bit for bit. Expert
// arithmetic is exact and placement-invariant, so neither the record's location nor the staging
// pass a job falls in may change an output bit.
#include "ninfer/ops/offloaded_sparse_moe.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace moe      = ninfer::ops::offloaded_moe;
namespace fixtures = ninfer::test::offloaded_moe;
using ninfer::DType;
using ninfer::Tensor;
using ninfer::test::cuda_check;

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

std::vector<std::uint16_t> cpu_column(const fixtures::Expert& e, const std::uint16_t* x) {
    std::vector<std::uint16_t> y(moe::kHidden);
    const std::uint16_t* xp[1] = {x};
    std::uint16_t* yp[1]       = {y.data()};
    moe::expert_forward(moe::best_cpu_isa(), e.record.data(), e.scales, 1, xp, yp);
    return y;
}

void test_layer(int experts, int columns, int top_k, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const std::size_t stride = moe::kRecordBytes;
    std::vector<fixtures::Expert> bank;
    std::vector<moe::ExpertScales> scales;
    for (int e = 0; e < experts; ++e) {
        bank.push_back(fixtures::random_expert(rng, e % 3 == 0));
        scales.push_back(bank.back().scales);
    }
    // Pinned host bank of every record; every third expert also resident in a device frame.
    std::uint8_t* host = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host), stride * experts, cudaHostAllocMapped), "cudaHostAlloc");
    for (int e = 0; e < experts; ++e) { std::memcpy(host + stride * e, bank[e].record.data(), moe::kRecordBytes); }
    std::uint8_t* host_device = nullptr;
    cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_device), host, 0), "cudaHostGetDevicePointer");
    std::vector<std::int32_t> frames(experts, -1);
    std::vector<std::uint8_t> frame_bytes;
    int resident = 0;
    for (int e = 0; e < experts; e += 3) {
        frames[e] = resident++;
        frame_bytes.insert(frame_bytes.end(), bank[e].record.begin(), bank[e].record.end());
    }
    auto* d_frame_base = device_copy(frame_bytes);
    auto* d_frames     = device_copy(frames);
    auto* d_scales     = device_copy(scales);

    // Routing from random FP32 logits.
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> logits(static_cast<std::size_t>(experts + 1) * columns);
    for (auto& v : logits) { v = n(rng); }
    auto* d_logits = device_copy(logits);
    std::int32_t* d_ids  = nullptr;
    float* d_weights     = nullptr;
    float* d_shared_gate = nullptr;
    cuda_check(cudaMalloc(&d_ids, sizeof(std::int32_t) * top_k * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_weights, sizeof(float) * top_k * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_shared_gate, sizeof(float) * columns), "cudaMalloc");
    ninfer::ops::MoeRouting routing{Tensor(d_ids, DType::I32, {top_k, columns}),
                                    Tensor(d_weights, DType::FP32, {top_k, columns}),
                                    Tensor(d_shared_gate, DType::FP32, {columns})};
    ninfer::ops::moe_route(Tensor(d_logits, DType::FP32, {experts + 1, columns}), top_k, routing, nullptr);
    void* d_dispatch = nullptr;
    cuda_check(cudaMalloc(&d_dispatch, ninfer::ops::moe_dispatch_bytes(experts, top_k * columns)), "cudaMalloc");
    auto dispatch = ninfer::ops::carve_moe_dispatch(d_dispatch, experts, top_k * columns);
    ninfer::ops::moe_dispatch(routing, experts, dispatch, nullptr);

    const auto x  = fixtures::random_activations(rng, columns);
    auto* d_x     = device_copy(x);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(top_k) * columns);
    cuda_check(cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost), "cudaMemcpy");

    // Expected outputs: column t * k + slot = expert ids[slot, t] applied to x[:, t].
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(moe::kHidden) * top_k * columns);
    for (int t = 0; t < columns; ++t) {
        for (int s = 0; s < top_k; ++s) {
            const auto y = cpu_column(bank[ids[static_cast<std::size_t>(t) * top_k + s]], &x[static_cast<std::size_t>(t) * moe::kHidden]);
            std::memcpy(&expected[(static_cast<std::size_t>(t) * top_k + s) * moe::kHidden], y.data(),
                        moe::kHidden * sizeof(std::uint16_t));
        }
    }

    const int max_jobs = std::min(experts, top_k * columns);
    std::uint8_t* d_staging = nullptr;
    cuda_check(cudaMalloc(&d_staging, stride * 64), "cudaMalloc");
    void* d_workspace = nullptr;
    cuda_check(cudaMalloc(&d_workspace, ninfer::ops::moe_experts_workspace_bytes(max_jobs, top_k * columns)),
               "cudaMalloc");
    std::uint16_t* d_out = nullptr;
    cuda_check(cudaMalloc(&d_out, expected.size() * sizeof(std::uint16_t)), "cudaMalloc");
    std::vector<moe::CpuMissService::Layer> layers{{.records = host, .record_stride = stride, .scales = scales.data()}};
    moe::CpuMissService service_two(layers, {.workers = 2, .max_jobs = 2, .max_columns = 64, .cpus = {}});
    moe::CpuMissService service_eight(layers, {.workers = 4, .max_jobs = 8, .max_columns = 64, .cpus = {}});
    struct Config {
        int slots;
        const moe::CpuMissService* service;
        bool fork = false;
    };
    // The fork stream and its events, for the one-pass decode/verification route.
    cudaStream_t fork_stream = nullptr;
    cudaEvent_t fork_events[2] = {};
    cuda_check(cudaStreamCreateWithFlags(&fork_stream, cudaStreamNonBlocking), "cudaStreamCreate");
    for (auto& event : fork_events) { cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "event"); }
    for (const Config config : {Config{0, nullptr}, Config{1, nullptr}, Config{3, nullptr}, Config{64, nullptr},
                                Config{3, &service_two}, Config{64, &service_eight}, Config{0, &service_eight},
                                Config{64, nullptr, true}, Config{64, &service_two, true}, Config{3, nullptr, true}}) {
        const int slots = config.slots;
        cuda_check(cudaMemset(d_out, 0xFF, expected.size() * sizeof(std::uint16_t)), "cudaMemset");
        cuda_check(cudaMemset(d_staging, 0, stride * 64), "cudaMemset");
        ninfer::ops::MoeExpertSource source{.frame_base    = d_frame_base,
                                            .frames        = d_frames,
                                            .host_records  = host_device,
                                            .record_stride = stride,
                                            .scales        = d_scales,
                                            .staging_base  = slots > 0 ? d_staging : nullptr,
                                            .staging_slots = slots,
                                            .cpu = config.service != nullptr ? config.service->channel(0)
                                                                             : ninfer::ops::MoeCpuChannel{}};
        if (config.fork) {
            source.fork_stream    = fork_stream;
            source.fork_events[0] = fork_events[0];
            source.fork_events[1] = fork_events[1];
        }
        Tensor tx(d_x, DType::BF16, {moe::kHidden, columns});
        Tensor out(d_out, DType::BF16, {moe::kHidden, top_k * columns});
        ninfer::ops::moe_experts(tx, dispatch, source, top_k, max_jobs, d_workspace, out, nullptr);
        cuda_check(cudaDeviceSynchronize(), "moe_experts");
        std::vector<std::uint16_t> got(expected.size());
        cuda_check(cudaMemcpy(got.data(), d_out, got.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
        long mismatches = 0;
        for (std::size_t i = 0; i < got.size(); ++i) { mismatches += got[i] != expected[i]; }
        std::printf("E=%d T=%d k=%d staging slots %2d%s, CPU jobs %d (served %llu): %ld mismatching outputs of %zu\n",
                    experts, columns, top_k, slots, config.fork ? " (fork)" : "",
                    config.service != nullptr ? config.service->channel(0).max_jobs : 0,
                    config.service != nullptr ? static_cast<unsigned long long>(config.service->served_experts()) : 0ULL,
                    mismatches, got.size());
        check(mismatches == 0, "layer route equals the CPU engine for every placement and staging pass");
    }
    for (auto event : fork_events) { cudaEventDestroy(event); }
    cudaStreamDestroy(fork_stream);
    cudaFree(d_out);
    cudaFree(d_workspace);
    cudaFree(d_staging);
    cudaFree(d_x);
    cudaFree(d_dispatch);
    cudaFree(d_shared_gate);
    cudaFree(d_weights);
    cudaFree(d_ids);
    cudaFree(d_logits);
    cudaFree(d_scales);
    cudaFree(d_frames);
    cudaFree(d_frame_base);
    cudaFreeHost(host);
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        test_layer(12, 1, 10, 7);  // decode: one column, ten experts
        test_layer(24, 8, 10, 11); // verify width: more jobs than one 3-slot pass
        test_layer(9, 5, 3, 13);   // several columns per expert
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all offloaded_moe layer checks passed\n");
    return 0;
}
