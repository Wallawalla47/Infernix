// NVIDIA FP4-Local's shared expert (MXFP8 [1280, 2560] gate/up and [2560, 640] down, 16-bit
// activations) on its own stream while offloaded_sparse_moe's routed experts run beside it, as the
// Qwen4Exp Forward overlaps them in decode: resident jobs on the compute stream, the misses staged and
// computed on the fork stream. Every output must equal a serialized run's bits, over many repeats,
// eager and graph-replayed, at decode widths 1-8 with W4A16 and W4A4 banks, through the shapes'
// selection and through a TMA-staged schedule (the producer of prefill widths). Without the proxy
// fence before a consumer warp releases a stage, TMA-staged outputs differed in up to 37 % of the
// calls (design §6.1.2, 2026-10-20). The serialized outputs' hashes are printed so that two builds or
// producers can be compared bit for bit.

#include "ops/op_family.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/linear/linear_test_common.h"
#include "ops/linear/mxfp8/mxfp8_launch.cuh"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// How the shared expert's projections run: the shapes' selection (infernix::ops::linear) or one
// TMA-staged schedule, whatever the selection picks at these widths.
using Projection = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void selected(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) { infernix::ops::linear(x, w, out, stream); }

template <int K>
void tma_staged(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    namespace detail = infernix::ops::detail;
    detail::mxfp8_linear_a16_tma<detail::Fp8Geometry<16, K>, detail::Mxfp8Tma<detail::Mxfp8TmaT16R64>>(x, w, out, stream);
}

struct Producer {
    const char* name;
    Projection gate_up, down;
};

struct DeviceMxfp8 {
    Weight weight;
    void* payload = nullptr;
    explicit DeviceMxfp8(const infernix::test::quantized_weight::PackedWeight& host) {
        cuda_check(cudaMalloc(&payload, host.payload.size()), "cudaMalloc");
        cuda_check(cudaMemcpy(payload, host.payload.data(), host.payload.size(), cudaMemcpyHostToDevice), "upload");
        weight = host.device_weight(payload);
    }
};

// One bank, its routing and its buffers for a decode call of T columns.
struct Case {
    moe::ExpertActivation activation;
    int columns;
    std::uint8_t* host_device = nullptr; // every record, pinned and mapped
    std::uint8_t* frame_base = nullptr;  // every third expert resident
    std::int32_t* frames = nullptr;
    moe::ExpertScales* scales = nullptr;
    std::uint8_t* staging = nullptr;
    infernix::ops::MoeDispatch dispatch;
    std::uint16_t* x = nullptr;       // [kHidden, T]
    std::uint16_t* h = nullptr;       // [kSharedI, T]: the down projection's input
    void* workspace = nullptr;
    std::uint16_t* out = nullptr;     // [kHidden, K * T]
    std::uint16_t* gate_up = nullptr; // [1280, T]
    std::uint16_t* down = nullptr;    // [kHidden, T]
    std::uint32_t* error = nullptr;
    int max_jobs = 0;
};

Case make_case(moe::ExpertActivation activation, int columns, std::uint32_t seed) {
    Case c{activation, columns};
    std::mt19937 rng(seed);
    const bool a16 = activation == moe::ExpertActivation::kA16;
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
    auto* d_logits = device_copy(logits);
    std::int32_t* ids = nullptr;
    float *weights = nullptr, *shared_gate = nullptr;
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
    cuda_check(cudaMalloc(&c.workspace, infernix::ops::moe_experts_workspace_bytes(infernix::ops::offloaded_moe::kFlashNextShape, c.max_jobs, kTopK * columns, columns,
                                                                                   activation)),
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

// One decode MoE as the Forward issues it: the shared expert on the side stream from the start,
// beside both of the routed call's passes, or (`serialized`) after them on the compute stream.
void issue(const Case& c, const Streams& s, const Producer& producer, const DeviceMxfp8& gate_up, const DeviceMxfp8& down,
           bool serialized) {
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
    producer.gate_up(tx, gate_up.weight, y1, shared);
    Tensor th(c.h, DType::BF16, {kSharedI, T});
    Tensor y2(c.down, DType::BF16, {moe::kHidden, T});
    producer.down(th, down.weight, y2, shared);
    if (!serialized) {
        cuda_check(cudaEventRecord(s.join, s.side), "cudaEventRecord");
        cuda_check(cudaStreamWaitEvent(s.main, s.join, 0), "cudaStreamWaitEvent");
    }
}

struct Outputs {
    std::vector<std::uint16_t> routed, gate_up, down;
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

unsigned long long fnv(const std::vector<std::uint16_t>& v) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const std::uint16_t e : v) { h = (h ^ e) * 1099511628211ULL; }
    return static_cast<unsigned long long>(h);
}

struct Result {
    int failures = 0;
    unsigned long long gate_up = 0, down = 0; // the serialized outputs' hashes
};

Result run(moe::ExpertActivation activation, int columns, int repeats, const Producer& producer,
           const DeviceMxfp8& gate_up, const DeviceMxfp8& down, std::uint32_t seed) {
    const Case c = make_case(activation, columns, seed);
    Streams s;
    poison(c, s.main);
    issue(c, s, producer, gate_up, down, /*serialized=*/true);
    cuda_check(cudaStreamSynchronize(s.main), "reference");
    const Outputs reference = read(c);
    const char* name = activation == moe::ExpertActivation::kA16 ? "W4A16" : "W4A4";
    std::printf("%s T=%d %s serialized outputs: routed %016llx gate/up %016llx down %016llx\n", name, columns,
                producer.name, fnv(reference.routed), fnv(reference.gate_up), fnv(reference.down));
    int eager_bad = 0, kinds = 0;
    for (int r = 0; r < repeats; ++r) {
        poison(c, s.main);
        issue(c, s, producer, gate_up, down, false);
        cuda_check(cudaStreamSynchronize(s.main), "eager");
        const int d = differing(read(c), reference);
        eager_bad += d != 0 ? 1 : 0;
        kinds |= d;
    }
    // Captured once (the side stream joins the capture through its events), replayed.
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    cuda_check(cudaStreamBeginCapture(s.main, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
    poison(c, s.main);
    issue(c, s, producer, gate_up, down, false);
    cuda_check(cudaStreamEndCapture(s.main, &graph), "cudaStreamEndCapture");
    cuda_check(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
    int graph_bad = 0;
    for (int r = 0; r < repeats; ++r) {
        cuda_check(cudaGraphLaunch(exec, s.main), "cudaGraphLaunch");
        cuda_check(cudaStreamSynchronize(s.main), "replay");
        const int d = differing(read(c), reference);
        graph_bad += d != 0 ? 1 : 0;
        kinds |= d;
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    std::printf("%s T=%d %s: eager %d of %d, graph %d of %d calls differ from the serialized run%s%s%s\n", name, columns,
                producer.name, eager_bad, repeats, graph_bad, repeats, kinds & 1 ? " [routed]" : "",
                kinds & 2 ? " [gate/up]" : "", kinds & 4 ? " [down]" : "");
    int failures = eager_bad + graph_bad != 0 ? 1 : 0;
    std::uint32_t error = 0;
    cuda_check(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost), "cudaMemcpy");
    if (error != 0) {
        std::printf("FAIL: %s T=%d set the error word %u\n", name, columns, error);
        ++failures;
    }
    return {failures, fnv(reference.gate_up), fnv(reference.down)};
}

} // namespace

INFERNIX_OP_TEST(mxfp8_moe_overlap) {
    const int repeats = argc > 1 ? std::max(1, std::atoi(argv[1])) : 200;
    try {
        const DeviceMxfp8 gate_up(infernix::test::linear::make_mxfp8_weight(1280, 2560, 7001U));
        const DeviceMxfp8 down(infernix::test::linear::make_mxfp8_weight(moe::kHidden, kSharedI, 7002U));
        const Producer producers[] = {{"selected", selected, selected},
                                      {"TMA-staged", tma_staged<2560>, tma_staged<kSharedI>}};
        int failures = 0;
        std::vector<Result> first; // the first producer's results, case by case
        for (const Producer& producer : producers) {
            std::uint32_t seed = 811U;
            std::size_t index  = 0;
            for (const auto activation : {moe::ExpertActivation::kA16, moe::ExpertActivation::kA4}) {
                for (const int columns : {1, 4, 6, 7, 8}) {
                    const Result r = run(activation, columns, repeats, producer, gate_up, down, seed++);
                    failures += r.failures;
                    if (&producer == producers) {
                        first.push_back(r);
                    } else if (r.gate_up != first[index].gate_up || r.down != first[index].down) {
                        std::printf("FAIL: T=%d: the %s producer's serialized outputs differ from the %s one's\n",
                                    columns, producer.name, producers[0].name);
                        ++failures;
                    }
                    ++index;
                }
            }
        }
        std::printf(failures == 0 ? "the overlapped shared expert and routed experts equal the serialized run\n"
                                  : "FAIL: overlapping the shared expert with the routed experts changes bits\n");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
