// Qwen3.8-Flash-Next recipe C's shared expert (row-scaled FP8 [1280, 2560] gate/up and [2560, 640] down,
// 16-bit activations) on its own stream while offloaded_sparse_moe's routed experts run beside it, as
// the Qwen4Exp Forward overlaps them in decode: resident jobs on the compute stream, the misses staged
// and computed on the fork stream, the shared expert beside both passes. Every output must equal a
// serialized run's bits, over many repeats, eager and graph-replayed, at decode widths 1-8 with W4A16
// and W4A4 banks. With TMA staging and no proxy fence before its consumers released a stage, the FP8
// A16 kernel's outputs came out wrong in such overlaps (2026-10-20: written but corrupted, a warp's
// MMA fragment at a time).

#include "infernix/ops/linear.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/linear/linear_test_common.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <random>
#include <vector>

namespace {

namespace moe      = infernix::ops::offloaded_moe;
namespace fixtures = infernix::test::offloaded_moe;
using infernix::DType;
using infernix::Tensor;
using infernix::Weight;
using infernix::test::cuda_check;

constexpr int kExperts = 128;
constexpr int kTopK    = 10;
constexpr int kMaxT    = 8;
constexpr int kSharedI = 640;
constexpr int kRepeats = 150;

template <class T> T* device_copy(const std::vector<T>& v) {
    T* p = nullptr;
    cuda_check(cudaMalloc(&p, v.size() * sizeof(T)), "cudaMalloc");
    cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
    return p;
}

template <class T> std::vector<T> host_copy(const T* p, std::size_t n) {
    std::vector<T> v(n);
    cuda_check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");
    return v;
}

struct DeviceWeight {
    Weight weight;
    void* payload = nullptr;
    explicit DeviceWeight(const infernix::test::quantized_weight::PackedWeight& host) {
        cuda_check(cudaMalloc(&payload, host.payload.size()), "cudaMalloc");
        cuda_check(cudaMemcpy(payload, host.payload.data(), host.payload.size(), cudaMemcpyHostToDevice), "upload");
        weight = host.device_weight(payload);
    }
};

// One bank, its routing and the buffers of a decode call of T columns; every third expert resident.
struct Case {
    moe::ExpertActivation activation;
    int columns;
    std::uint8_t* host_device       = nullptr; // pinned, mapped: every record
    std::uint8_t* frame_base        = nullptr;
    std::int32_t* frames            = nullptr;
    moe::ExpertScales* scales       = nullptr;
    std::uint8_t* staging           = nullptr;
    infernix::ops::MoeDispatch dispatch;
    std::uint16_t* x       = nullptr; // [kHidden, T]
    std::uint16_t* h       = nullptr; // [kSharedI, T]: the down projection's input
    void* workspace        = nullptr;
    std::uint16_t* out     = nullptr; // [kHidden, K * T]
    std::uint16_t* gate_up = nullptr; // [1280, T]
    std::uint16_t* down    = nullptr; // [kHidden, T]
    std::uint32_t* error   = nullptr;
    int max_jobs           = 0;
};

Case make_case(moe::ExpertActivation activation, int columns, std::uint32_t seed) {
    Case c{activation, columns};
    std::mt19937 rng(seed);
    const bool a16           = activation == moe::ExpertActivation::kA16;
    const std::size_t stride = moe::kRecordBytes;
    std::vector<fixtures::Expert> bank;
    std::vector<moe::ExpertScales> scales;
    for (int e = 0; e < kExperts; ++e) {
        bank.push_back(a16 ? fixtures::random_a16_expert(rng) : fixtures::random_expert(rng, e % 3 == 0));
        scales.push_back(bank.back().scales);
    }
    std::uint8_t* host = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host), stride * kExperts, cudaHostAllocMapped), "cudaHostAlloc");
    for (int e = 0; e < kExperts; ++e) { std::memcpy(host + stride * e, bank[e].record.data(), stride); }
    cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&c.host_device), host, 0), "cudaHostGetDevicePointer");
    std::vector<std::int32_t> frames(kExperts, -1);
    std::vector<std::uint8_t> frame_bytes;
    for (int e = 0, f = 0; e < kExperts; e += 3, ++f) {
        frames[e] = f;
        frame_bytes.insert(frame_bytes.end(), bank[e].record.begin(), bank[e].record.end());
    }
    c.frame_base = device_copy(frame_bytes);
    c.frames     = device_copy(frames);
    c.scales     = device_copy(scales);
    cuda_check(cudaMalloc(&c.staging, stride * kTopK * kMaxT), "cudaMalloc");

    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> logits(static_cast<std::size_t>(kExperts + 1) * columns);
    for (auto& v : logits) { v = n(rng); }
    auto* d_logits      = device_copy(logits);
    std::int32_t* ids   = nullptr;
    float* weights      = nullptr;
    float* shared_gate  = nullptr;
    cuda_check(cudaMalloc(&ids, sizeof(std::int32_t) * kTopK * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&weights, sizeof(float) * kTopK * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&shared_gate, sizeof(float) * columns), "cudaMalloc");
    infernix::ops::MoeRouting routing{Tensor(ids, DType::I32, {kTopK, columns}),
                                      Tensor(weights, DType::FP32, {kTopK, columns}),
                                      Tensor(shared_gate, DType::FP32, {columns})};
    infernix::ops::moe_route(Tensor(d_logits, DType::FP32, {kExperts + 1, columns}), kTopK, routing, nullptr);
    void* dispatch_bytes = nullptr;
    cuda_check(cudaMalloc(&dispatch_bytes, infernix::ops::moe_dispatch_bytes(kExperts, kTopK * columns)), "cudaMalloc");
    c.dispatch = infernix::ops::carve_moe_dispatch(dispatch_bytes, kExperts, kTopK * columns);
    infernix::ops::moe_dispatch(routing, kExperts, c.dispatch, nullptr, nullptr);

    c.x = device_copy(a16 ? fixtures::wide_range_activations(rng, columns) : fixtures::random_activations(rng, columns));
    std::vector<std::uint16_t> h(static_cast<std::size_t>(kSharedI) * columns);
    for (auto& v : h) { v = fixtures::to_bf16(n(rng)); }
    c.h        = device_copy(h);
    c.max_jobs = std::min(kExperts, kTopK * columns);
    cuda_check(cudaMalloc(&c.workspace,
                          infernix::ops::moe_experts_workspace_bytes(infernix::ops::offloaded_moe::kFlashNextShape, c.max_jobs, kTopK * columns, columns, activation)),
               "cudaMalloc");
    cuda_check(cudaMalloc(&c.out, sizeof(std::uint16_t) * moe::kHidden * kTopK * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&c.gate_up, sizeof(std::uint16_t) * 1280 * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&c.down, sizeof(std::uint16_t) * moe::kHidden * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&c.error, sizeof(std::uint32_t)), "cudaMalloc");
    cuda_check(cudaMemset(c.error, 0, sizeof(std::uint32_t)), "cudaMemset");
    cuda_check(cudaDeviceSynchronize(), "setup");
    return c;
}

struct Streams {
    cudaStream_t main = nullptr, fork = nullptr, side = nullptr;
    cudaEvent_t fork_events[2] = {};
    cudaEvent_t start = nullptr, join = nullptr;
    Streams() {
        for (cudaStream_t* s : {&main, &fork, &side}) {
            cuda_check(cudaStreamCreateWithFlags(s, cudaStreamNonBlocking), "cudaStreamCreate");
        }
        for (cudaEvent_t* e : {&fork_events[0], &fork_events[1], &start, &join}) {
            cuda_check(cudaEventCreateWithFlags(e, cudaEventDisableTiming), "cudaEventCreate");
        }
    }
};

// One decode MoE as the Forward issues it: the routed experts on the compute stream, their misses
// staged and computed on the fork stream, and the shared expert after them on the compute stream
// (serialized) or on the side stream from the start (overlapped).
void issue(const Case& c, const Streams& s, const DeviceWeight& gate_up, const DeviceWeight& down, bool serialized) {
    const int T = c.columns;
    infernix::ops::MoeExpertSource source{.frame_base    = c.frame_base,
                                          .frames        = c.frames,
                                          .host_records  = c.host_device,
                                          .record_stride = moe::kRecordBytes,
                                          .scales        = c.scales,
                                          .staging_base  = c.staging,
                                          .staging_slots = kTopK * kMaxT};
    source.activation     = c.activation;
    source.error          = c.error;
    source.fork_stream    = s.fork;
    source.fork_events[0] = s.fork_events[0];
    source.fork_events[1] = s.fork_events[1];
    const cudaStream_t shared = serialized ? s.main : s.side;
    if (!serialized) {
        cuda_check(cudaEventRecord(s.start, s.main), "cudaEventRecord");
        cuda_check(cudaStreamWaitEvent(s.side, s.start, 0), "cudaStreamWaitEvent");
    }
    Tensor tx(c.x, DType::BF16, {moe::kHidden, T});
    Tensor out(c.out, DType::BF16, {moe::kHidden, kTopK * T});
    infernix::ops::moe_experts(tx, c.dispatch, source, kTopK, c.max_jobs, c.workspace, out, s.main);
    Tensor y1(c.gate_up, DType::BF16, {1280, T});
    infernix::ops::linear(tx, gate_up.weight, y1, shared);
    Tensor th(c.h, DType::BF16, {kSharedI, T});
    Tensor y2(c.down, DType::BF16, {moe::kHidden, T});
    infernix::ops::linear(th, down.weight, y2, shared);
    if (!serialized) {
        cuda_check(cudaEventRecord(s.join, s.side), "cudaEventRecord");
        cuda_check(cudaStreamWaitEvent(s.main, s.join, 0), "cudaStreamWaitEvent");
    }
}

struct Outputs {
    std::vector<std::uint16_t> routed, gate_up, down;
    bool operator==(const Outputs&) const = default;
};

Outputs read(const Case& c) {
    const auto T = static_cast<std::size_t>(c.columns);
    return {host_copy(c.out, moe::kHidden * kTopK * T), host_copy(c.gate_up, 1280 * T), host_copy(c.down, moe::kHidden * T)};
}

void poison(const Case& c, cudaStream_t s) {
    const auto T = static_cast<std::size_t>(c.columns);
    cuda_check(cudaMemsetAsync(c.out, 0xFF, sizeof(std::uint16_t) * moe::kHidden * kTopK * T, s), "cudaMemsetAsync");
    cuda_check(cudaMemsetAsync(c.gate_up, 0xFF, sizeof(std::uint16_t) * 1280 * T, s), "cudaMemsetAsync");
    cuda_check(cudaMemsetAsync(c.down, 0xFF, sizeof(std::uint16_t) * moe::kHidden * T, s), "cudaMemsetAsync");
}

// Which outputs differ from the reference: bit 0 routed, 1 gate/up, 2 down.
int differing(const Outputs& got, const Outputs& want) {
    return (got.routed != want.routed ? 1 : 0) | (got.gate_up != want.gate_up ? 2 : 0) | (got.down != want.down ? 4 : 0);
}

int run(moe::ExpertActivation activation, int columns, const DeviceWeight& gate_up, const DeviceWeight& down,
        std::uint32_t seed) {
    const Case c = make_case(activation, columns, seed);
    Streams s;
    poison(c, s.main);
    issue(c, s, gate_up, down, /*serialized=*/true);
    cuda_check(cudaStreamSynchronize(s.main), "reference");
    const Outputs reference = read(c);
    const char* name = activation == moe::ExpertActivation::kA16 ? "W4A16" : "W4A4";
    int eager_bad = 0, kinds = 0;
    for (int r = 0; r < kRepeats; ++r) {
        poison(c, s.main);
        issue(c, s, gate_up, down, false);
        cuda_check(cudaStreamSynchronize(s.main), "eager");
        const int d = differing(read(c), reference);
        eager_bad += d != 0 ? 1 : 0;
        kinds |= d;
    }
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    cuda_check(cudaStreamBeginCapture(s.main, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
    poison(c, s.main);
    issue(c, s, gate_up, down, false);
    cuda_check(cudaStreamEndCapture(s.main, &graph), "cudaStreamEndCapture");
    cuda_check(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
    int graph_bad = 0;
    for (int r = 0; r < kRepeats; ++r) {
        cuda_check(cudaGraphLaunch(exec, s.main), "cudaGraphLaunch");
        cuda_check(cudaStreamSynchronize(s.main), "replay");
        const int d = differing(read(c), reference);
        graph_bad += d != 0 ? 1 : 0;
        kinds |= d;
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    std::uint32_t error = 0;
    cuda_check(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost), "cudaMemcpy");
    const int failures = (eager_bad + graph_bad != 0 ? 1 : 0) + (error != 0 ? 1 : 0);
    std::printf("%s%s T=%d: eager %d of %d, graph %d of %d overlapped calls differ from the serialized run%s%s%s%s\n",
                failures != 0 ? "FAIL: " : "", name, columns, eager_bad, kRepeats, graph_bad, kRepeats,
                kinds & 1 ? " [routed]" : "", kinds & 2 ? " [gate/up]" : "", kinds & 4 ? " [down]" : "",
                error != 0 ? " [error word set]" : "");
    return failures;
}

} // namespace

int main() {
    try {
        namespace lt = infernix::test::linear;
        const DeviceWeight gate_up(lt::make_fp8_weight(1280, moe::kHidden, 7101U));
        const DeviceWeight down(lt::make_fp8_weight(moe::kHidden, kSharedI, 7102U));
        int failures = 0;
        std::uint32_t seed = 1;
        for (const auto activation : {moe::ExpertActivation::kA16, moe::ExpertActivation::kA4}) {
            for (int columns = 1; columns <= kMaxT; ++columns) { failures += run(activation, columns, gate_up, down, seed++); }
        }
        if (failures != 0) {
            std::printf("%d failure(s)\n", failures);
            return 1;
        }
        std::printf("fp8 shared expert beside the routed experts: all overlapped calls equal the serialized ones\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
