// The TMA-staged linear routes on their own stream while offloaded_sparse_moe's routed experts run
// beside them, as a Forward overlaps a side-stream projection with a routed call (the misses staged
// and computed on the fork stream). A TMA refill is an async-proxy write into a stage the consumers
// read through the generic proxy; without a proxy fence before a consumer warp releases its stage,
// the MXFP8 route's outputs came out wrong in up to 37 % of such calls (test_mxfp8_overlap.cu,
// 2026-10-20). The victim is the TMA kernel family whose race this harness exposes: FP8 A8
// (fp8_a8_tma_mma) [14336,5120] at T = 128 under AllowA8, which without the fence returned wrong
// outputs in 20 and 7 of 200 calls beside W4A16 banks (eager, graph) and 147 and 133 beside W4A4.
// BF16 A16 ([2560,640] at T = 40, [1280,2560] at T = 192), the FP8 A8 pair ([1280,2560] at T = 1536)
// and NVFP4 A4 ([5120,6144] at T = 1024) never failed here unfenced (0 of 200 each), so they are not
// victims; they fence for the same proxy rule. fp8-a8-pair-block32 ([2560,6144] at T = 1024,
// [16384,2560] at T = 200), whose first, unfenced form was nondeterministic even in isolation, is one.
//
// Every victim and routed output must equal a serialized run's bits, over many repeats, eager and
// graph-replayed, beside W4A16 and W4A4 banks. The serialized outputs' hashes are printed so that
// two builds can be compared bit for bit.
//
//   infernix_tma_moe_overlap_test [repeats (200)] [victim name filter]

#include "ops/op_family.h"
#include "core/arena.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/linear/linear_test_common.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

namespace moe      = infernix::ops::offloaded_moe;
namespace fixtures = infernix::test::offloaded_moe;
namespace qw       = infernix::test::quantized_weight;
using infernix::DType;
using infernix::QType;
using infernix::Tensor;
using infernix::Weight;
using infernix::ops::LinearPolicy;
using infernix::ops::LinearRoutes;
using infernix::test::cuda_check;

constexpr int kExperts     = 128;
constexpr int kTopK        = 10;
constexpr int kRoutedT     = 8; // a decode round's routed call
constexpr int kRoutedCalls = 4; // back to back beside one victim call, so the overlap spans it

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

// One routed bank, its routing and its buffers for a call of kRoutedT columns; every third expert
// is resident, so the call forks a staged pass for the misses.
struct Routed {
    moe::ExpertActivation activation;
    std::uint8_t* host_device = nullptr;
    std::uint8_t* frame_base  = nullptr;
    std::int32_t* frames      = nullptr;
    moe::ExpertScales* scales = nullptr;
    std::uint8_t* staging     = nullptr;
    infernix::ops::MoeDispatch dispatch;
    std::uint16_t* x     = nullptr;
    void* workspace      = nullptr;
    std::uint16_t* out   = nullptr;
    std::uint32_t* error = nullptr;
    int max_jobs         = 0;
};

Routed make_routed(moe::ExpertActivation activation, std::uint32_t seed) {
    Routed c{activation};
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
    cuda_check(cudaMalloc(&c.staging, stride * kTopK * kRoutedT), "cudaMalloc");

    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> logits(static_cast<std::size_t>(kExperts + 1) * kRoutedT);
    for (auto& v : logits) { v = n(rng); }
    auto* d_logits = device_copy(logits);
    std::int32_t* ids = nullptr;
    float *weights = nullptr, *shared_gate = nullptr;
    cuda_check(cudaMalloc(&ids, sizeof(std::int32_t) * kTopK * kRoutedT), "cudaMalloc");
    cuda_check(cudaMalloc(&weights, sizeof(float) * kTopK * kRoutedT), "cudaMalloc");
    cuda_check(cudaMalloc(&shared_gate, sizeof(float) * kRoutedT), "cudaMalloc");
    infernix::ops::MoeRouting routing{Tensor(ids, DType::I32, {kTopK, kRoutedT}),
                                      Tensor(weights, DType::FP32, {kTopK, kRoutedT}),
                                      Tensor(shared_gate, DType::FP32, {kRoutedT})};
    infernix::ops::moe_route(Tensor(d_logits, DType::FP32, {kExperts + 1, kRoutedT}), kTopK, routing, nullptr);
    void* dispatch_bytes = nullptr;
    cuda_check(cudaMalloc(&dispatch_bytes, infernix::ops::moe_dispatch_bytes(kExperts, kTopK * kRoutedT)), "cudaMalloc");
    c.dispatch = infernix::ops::carve_moe_dispatch(dispatch_bytes, kExperts, kTopK * kRoutedT);
    infernix::ops::moe_dispatch(routing, kExperts, c.dispatch, nullptr, nullptr);
    c.x = device_copy(a16 ? fixtures::wide_range_activations(rng, kRoutedT) : fixtures::random_activations(rng, kRoutedT));
    c.max_jobs = std::min(kExperts, kTopK * kRoutedT);
    cuda_check(cudaMalloc(&c.workspace,
                          infernix::ops::moe_experts_workspace_bytes(infernix::ops::offloaded_moe::kFlashNextShape, c.max_jobs, kTopK * kRoutedT, kRoutedT, activation)),
               "cudaMalloc");
    cuda_check(cudaMalloc(&c.out, sizeof(std::uint16_t) * moe::kHidden * kTopK * kRoutedT), "cudaMalloc");
    cuda_check(cudaMalloc(&c.error, sizeof(std::uint32_t)), "cudaMalloc");
    cuda_check(cudaMemset(c.error, 0, sizeof(std::uint32_t)), "cudaMemset");
    cuda_check(cudaDeviceSynchronize(), "setup");
    return c;
}

std::size_t routed_elements() { return static_cast<std::size_t>(moe::kHidden) * kTopK * kRoutedT; }

// One linear call on a TMA route, into its own output.
struct Victim {
    std::string name;
    std::uint16_t* out = nullptr;
    std::size_t elements = 0;
    std::function<void(cudaStream_t)> issue;
};

std::vector<std::uint16_t> random_bf16(std::size_t count, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<std::uint16_t> v(count);
    for (auto& e : v) { e = fixtures::to_bf16(n(rng)); }
    return v;
}

// Owns a victim's weight, activation, output and workspace.
struct VictimStorage {
    qw::PackedWeight host;
    void* payload = nullptr;
    std::uint16_t* x = nullptr;
    std::unique_ptr<infernix::DeviceArena> workspace;
};

Victim make_victim(std::vector<std::unique_ptr<VictimStorage>>& keep, std::string name, qw::PackedWeight weight, int t,
                   bool with_policy, LinearPolicy policy, LinearRoutes routes) {
    auto storage  = std::make_unique<VictimStorage>();
    storage->host = std::move(weight);
    cuda_check(cudaMalloc(&storage->payload, storage->host.payload.size()), "cudaMalloc");
    cuda_check(cudaMemcpy(storage->payload, storage->host.payload.data(), storage->host.payload.size(),
                          cudaMemcpyHostToDevice),
               "upload");
    const Weight w = storage->host.device_weight(storage->payload);
    storage->x     = device_copy(random_bf16(static_cast<std::size_t>(w.k) * t, 0x7a11U + static_cast<std::uint32_t>(t)));
    if (with_policy) {
        const std::size_t capacity = infernix::ops::linear_workspace_capacity_bytes(w.qtype, w.n, w.k, policy, t, t);
        storage->workspace         = std::make_unique<infernix::DeviceArena>(std::max<std::size_t>(capacity, 256));
    }
    Victim v{std::move(name), nullptr, static_cast<std::size_t>(w.n) * t, {}};
    cuda_check(cudaMalloc(&v.out, sizeof(std::uint16_t) * v.elements), "cudaMalloc");
    VictimStorage* s = storage.get();
    std::uint16_t* out = v.out;
    v.issue = [s, w, t, out, with_policy, policy, routes](cudaStream_t stream) {
        const Tensor x(s->x, DType::BF16, {w.k, t});
        Tensor y(out, DType::BF16, {w.n, t});
        if (with_policy) {
            s->workspace->reset();
            infernix::ops::linear(x, w, y, policy, *s->workspace, stream, routes);
        } else {
            infernix::ops::linear(x, w, y, stream);
        }
    };
    keep.push_back(std::move(storage));
    return v;
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

// The victim on the side stream from the start, beside kRoutedCalls routed calls on the compute stream,
// or (`serialized`) after them on the compute stream.
void issue(const Routed& c, const Streams& s, const Victim& victim, bool serialized) {
    infernix::ops::MoeExpertSource source{.frame_base    = c.frame_base,
                                          .frames        = c.frames,
                                          .host_records  = c.host_device,
                                          .record_stride = moe::kRecordBytes,
                                          .scales        = c.scales,
                                          .staging_base  = c.staging,
                                          .staging_slots = kTopK * kRoutedT};
    source.activation     = c.activation;
    source.error          = c.error;
    source.fork_stream    = s.fork;
    source.fork_events[0] = s.fork_events[0];
    source.fork_events[1] = s.fork_events[1];
    if (!serialized) {
        cuda_check(cudaEventRecord(s.start, s.main), "cudaEventRecord");
        cuda_check(cudaStreamWaitEvent(s.side, s.start, 0), "cudaStreamWaitEvent");
        victim.issue(s.side);
    }
    const Tensor tx(c.x, DType::BF16, {moe::kHidden, kRoutedT});
    Tensor out(c.out, DType::BF16, {moe::kHidden, kTopK * kRoutedT});
    for (int i = 0; i < kRoutedCalls; ++i) {
        infernix::ops::moe_experts(tx, c.dispatch, source, kTopK, c.max_jobs, c.workspace, out, s.main);
    }
    if (serialized) {
        victim.issue(s.main);
    } else {
        cuda_check(cudaEventRecord(s.join, s.side), "cudaEventRecord");
        cuda_check(cudaStreamWaitEvent(s.main, s.join, 0), "cudaStreamWaitEvent");
    }
}

void poison(const Routed& c, const Victim& v, cudaStream_t s) {
    cuda_check(cudaMemsetAsync(c.out, 0xFF, sizeof(std::uint16_t) * routed_elements(), s), "cudaMemsetAsync");
    cuda_check(cudaMemsetAsync(v.out, 0xFF, sizeof(std::uint16_t) * v.elements, s), "cudaMemsetAsync");
}

unsigned long long fnv(const std::vector<std::uint16_t>& v) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const std::uint16_t e : v) { h = (h ^ e) * 1099511628211ULL; }
    return static_cast<unsigned long long>(h);
}

int run(const Routed& c, const Victim& victim, int repeats) {
    Streams s;
    poison(c, victim, s.main);
    issue(c, s, victim, /*serialized=*/true);
    cuda_check(cudaStreamSynchronize(s.main), "reference");
    const auto want_routed = host_copy(c.out, routed_elements());
    const auto want_victim = host_copy(victim.out, victim.elements);
    const char* bank = c.activation == moe::ExpertActivation::kA16 ? "W4A16" : "W4A4";
    std::printf("%s beside %s: serialized victim %016llx, routed %016llx\n", victim.name.c_str(), bank, fnv(want_victim),
                fnv(want_routed));
    int kinds = 0;
    const auto check = [&]() {
        const int d = (host_copy(c.out, routed_elements()) != want_routed ? 1 : 0) |
                      (host_copy(victim.out, victim.elements) != want_victim ? 2 : 0);
        kinds |= d;
        return d != 0 ? 1 : 0;
    };
    int eager_bad = 0;
    for (int r = 0; r < repeats; ++r) {
        poison(c, victim, s.main);
        issue(c, s, victim, false);
        cuda_check(cudaStreamSynchronize(s.main), "eager");
        eager_bad += check();
    }
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    cuda_check(cudaStreamBeginCapture(s.main, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
    poison(c, victim, s.main);
    issue(c, s, victim, false);
    cuda_check(cudaStreamEndCapture(s.main, &graph), "cudaStreamEndCapture");
    cuda_check(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
    int graph_bad = 0;
    for (int r = 0; r < repeats; ++r) {
        cuda_check(cudaGraphLaunch(exec, s.main), "cudaGraphLaunch");
        cuda_check(cudaStreamSynchronize(s.main), "replay");
        graph_bad += check();
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    std::printf("%s beside %s: eager %d of %d, graph %d of %d calls differ from the serialized run%s%s\n",
                victim.name.c_str(), bank, eager_bad, repeats, graph_bad, repeats, kinds & 1 ? " [routed]" : "",
                kinds & 2 ? " [victim]" : "");
    int failures = eager_bad + graph_bad != 0 ? 1 : 0;
    std::uint32_t error = 0;
    cuda_check(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost), "cudaMemcpy");
    if (error != 0) {
        std::printf("FAIL: %s beside %s set the error word %u\n", victim.name.c_str(), bank, error);
        ++failures;
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(tma_moe_overlap) {
    const int repeats       = argc > 1 ? std::max(1, std::atoi(argv[1])) : 200;
    const std::string only  = argc > 2 ? argv[2] : "";
    namespace linear_test   = infernix::test::linear;
    try {
        std::vector<std::unique_ptr<VictimStorage>> keep;
        std::vector<Victim> victims;
        victims.push_back(make_victim(keep, "FP8 A8 [14336,5120] T=128", linear_test::make_fp8_weight(14336, 5120, 9103U),
                                      128, true, LinearPolicy::AllowA8, LinearRoutes::Exact));
        // fp8-a8-pair-block32 (fenced from the start): its 2026-10-17 unfenced form changed bits between
        // identical isolated calls on [2560,6144] in the 128-column tiles; also the 64-column tiles.
        victims.push_back(make_victim(keep, "FP8 pair-block32 [2560,6144] T=1024",
                                      linear_test::make_fp8_weight(2560, 6144, 9107U), 1024, true,
                                      LinearPolicy::A16Only, LinearRoutes::Fp8A8PairBlock32));
        victims.push_back(make_victim(keep, "FP8 pair-block32 [16384,2560] T=200",
                                      linear_test::make_fp8_weight(16384, 2560, 9109U), 200, true,
                                      LinearPolicy::A16Only, LinearRoutes::Fp8A8PairBlock32));
        int failures       = 0;
        std::uint32_t seed = 911U;
        for (const auto activation : {moe::ExpertActivation::kA16, moe::ExpertActivation::kA4}) {
            const Routed c = make_routed(activation, seed++);
            for (const Victim& victim : victims) {
                if (!only.empty() && victim.name.find(only) == std::string::npos) continue;
                failures += run(c, victim, repeats);
            }
        }
        std::printf(failures == 0 ? "every TMA victim and the routed experts equal the serialized run\n"
                                  : "FAIL: overlapping a TMA route with the routed experts changes bits\n");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
