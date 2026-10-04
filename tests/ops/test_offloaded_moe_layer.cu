// The layer route of offloaded_sparse_moe (moe_route, moe_dispatch, moe_experts) against the CPU
// engine, with expert records split between device frames and the pinned host bank and misses
// read zero-copy, staged through 1, 3 or 64 device slots (one or several staging passes), or
// served by the host expert engine through the CPU miss channel
// (docs/maintainer/qwen3_8-flash-next-design.md §8.6, §10.3, §16.2).
//
// Oracle: the CPU engine's output of each routed (column, expert) pair, bit for bit. Expert
// arithmetic is exact and placement-invariant, so neither the record's location nor the staging
// pass a job falls in may change an output bit. moe_dispatch has its own exact oracle (test_dispatch).
#include "ninfer/ops/offloaded_sparse_moe.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
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
    ninfer::ops::moe_dispatch(routing, experts, dispatch, nullptr, nullptr);

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

// moe_dispatch against an exact oracle: counts, offsets, jobs and job_count exact, each expert's
// entries as a set, and the route-log copy. The dispatch arrays start as garbage, and a captured
// call is replayed on new ids, so neither route may depend on memory cleared outside the call.
// Entries <= 1024 take the one-CTA kernel, more the count/scan/scatter kernels.
void test_dispatch(int experts, int entries, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, experts - 1);
    const auto random_ids = [&] {
        std::vector<std::int32_t> ids(entries);
        // A few hot experts and many cold ones, as routing produces; some experts stay unused.
        for (auto& id : ids) { id = (rng() % 4 == 0) ? pick(rng) % 8 : pick(rng); }
        return ids;
    };
    std::int32_t* d_ids = nullptr;
    std::int32_t* d_log = nullptr;
    void* d_dispatch    = nullptr;
    const std::size_t bytes = ninfer::ops::moe_dispatch_bytes(experts, entries);
    cuda_check(cudaMalloc(&d_ids, sizeof(std::int32_t) * entries), "cudaMalloc");
    cuda_check(cudaMalloc(&d_log, sizeof(std::int32_t) * entries), "cudaMalloc");
    cuda_check(cudaMalloc(&d_dispatch, bytes), "cudaMalloc");
    cuda_check(cudaMemset(d_dispatch, 0x5A, bytes), "cudaMemset");
    cuda_check(cudaMemset(d_log, 0xFF, sizeof(std::int32_t) * entries), "cudaMemset");
    auto dispatch = ninfer::ops::carve_moe_dispatch(d_dispatch, experts, entries);
    ninfer::ops::MoeRouting routing{Tensor(d_ids, DType::I32, {entries, 1}), Tensor{}, Tensor{}};

    const auto verify = [&](const std::vector<std::int32_t>& ids, const char* phase) {
        std::vector<std::int32_t> counts(experts), offsets(experts + 1), jobs(experts), got_entries(entries), log(entries);
        std::int32_t job_count = -1;
        cuda_check(cudaMemcpy(counts.data(), dispatch.counts, sizeof(std::int32_t) * experts, cudaMemcpyDeviceToHost), "copy");
        cuda_check(cudaMemcpy(offsets.data(), dispatch.offsets, sizeof(std::int32_t) * (experts + 1), cudaMemcpyDeviceToHost),
                   "copy");
        cuda_check(cudaMemcpy(jobs.data(), dispatch.jobs, sizeof(std::int32_t) * experts, cudaMemcpyDeviceToHost), "copy");
        cuda_check(cudaMemcpy(&job_count, dispatch.job_count, sizeof(std::int32_t), cudaMemcpyDeviceToHost), "copy");
        cuda_check(cudaMemcpy(got_entries.data(), dispatch.entries, sizeof(std::int32_t) * entries, cudaMemcpyDeviceToHost),
                   "copy");
        cuda_check(cudaMemcpy(log.data(), d_log, sizeof(std::int32_t) * entries, cudaMemcpyDeviceToHost), "copy");
        std::vector<std::vector<std::int32_t>> by_expert(experts);
        for (int i = 0; i < entries; ++i) { by_expert[ids[i]].push_back(i); }
        bool ok = log == ids;
        int used = 0, start = 0;
        for (int e = 0; e < experts; ++e) {
            const int c = static_cast<int>(by_expert[e].size());
            ok = ok && counts[e] == c && offsets[e] == start;
            if (c > 0) { ok = ok && jobs[used++] == e; }
            if (ok && c > 0) {
                std::vector<std::int32_t> mine(got_entries.begin() + start, got_entries.begin() + start + c);
                std::sort(mine.begin(), mine.end());
                ok = mine == by_expert[e];
            }
            start += c;
        }
        ok = ok && offsets[experts] == entries && job_count == used;
        std::printf("dispatch E=%d entries=%d (%s): %s\n", experts, entries, phase, ok ? "exact" : "MISMATCH");
        check(ok, "moe_dispatch equals the exact oracle");
    };

    const auto first = random_ids();
    cuda_check(cudaMemcpy(d_ids, first.data(), sizeof(std::int32_t) * entries, cudaMemcpyHostToDevice), "copy");
    ninfer::ops::moe_dispatch(routing, experts, dispatch, d_log, nullptr);
    cuda_check(cudaDeviceSynchronize(), "dispatch");
    verify(first, "eager");

    cudaStream_t stream = nullptr;
    cudaGraph_t graph   = nullptr;
    cudaGraphExec_t exec = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
    ninfer::ops::moe_dispatch(routing, experts, dispatch, d_log, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "capture");
    cuda_check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    for (int replay = 0; replay < 2; ++replay) {
        const auto ids = random_ids();
        cuda_check(cudaMemcpy(d_ids, ids.data(), sizeof(std::int32_t) * entries, cudaMemcpyHostToDevice), "copy");
        cuda_check(cudaGraphLaunch(exec, stream), "launch");
        cuda_check(cudaStreamSynchronize(stream), "replay");
        verify(ids, replay == 0 ? "graph replay 1" : "graph replay 2");
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_dispatch);
    cudaFree(d_log);
    cudaFree(d_ids);
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        for (const int entries : {1, 10, 1024, 1025, 40960}) { test_dispatch(512, entries, 100U + entries); }
        test_dispatch(1024, 1024, 31); // every expert slot of the one-CTA kernel
        test_dispatch(3, 1000, 37);    // few experts, long runs per expert
        test_layer(12, 1, 10, 7);  // decode: one column, ten experts
        test_layer(24, 8, 10, 11); // verify width: more jobs than one 3-slot pass
        test_layer(9, 5, 3, 13);   // several columns per expert
        test_layer(9, 4, 3, 17);   // MTP verification width: one-column kernels, several passes per job
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
