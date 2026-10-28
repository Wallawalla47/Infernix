// NVFP4 W4A16 expert banks on the 8-bit activation route (--a8-activations, NarrowA16::kA8; LOSSY, opt-in;
// ops/offloaded_sparse_moe/nvfp4_a8.h) through the layer route of offloaded_sparse_moe (moe_route, moe_dispatch,
// moe_experts) at both registered expert geometries: Qwen3.8-Flash-Next's (H 2560, I 640) and Qwen3.5-122B-A10B's
// (H 3072, I 1024, with its shared expert as the call's last job).
//
// Records are random NVFP4 units (fixtures::random_a16_expert: random codes, block scales including zero,
// subnormal and 448, random multipliers) in a pinned host bank, every third expert in a device frame whose pitch
// exceeds the record stride. Misses are read zero-copy, staged through 3 or 64 slots, read from copied-ahead
// slots (MoeExpertSource::prefetched), forked (split, or landed in free frames) or computed by the CPU route
// through the miss service; calls run eagerly and captured into CUDA graphs, on the exact wide route's workspace
// and on the A4 residual one's (whose x term plane then holds the route's x codes). Every x column has a wide
// range (fixtures::wide_range_activations), a zero slice, a slice below the codec's smallest quantized maximum
// (2^-120: scale 0), a tiny quantized slice (a subnormal scale), and column 1 an Inf (NaN outputs).
//
// Oracle: the route's arithmetic written out here as scalar code from nvfp4_a8.h's specification, over
// canonical_math.h's IEEE operations and ggml_a8.h's slice codec only. Calls of 1, 3, 8, 9, 16, 64 and 128 columns
// (every job on the 8-bit kernels, whatever its column count) must equal it bit for bit in every configuration,
// and an expert's output must not depend on its column count in the call (one column against nine). A call of
// 129 columns is past the route and must return the exact W4A16 routes' bits (the CPU engine's expert_forward_a16).

#include "ops/op_family.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_a8.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace moe      = infernix::ops::offloaded_moe;
namespace fixtures = infernix::test::offloaded_moe;
namespace canon    = infernix::ops::canon;
namespace a8       = infernix::ggml::a8;
using infernix::DType;
using infernix::Tensor;
using infernix::test::cuda_check;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

template <class T> T* device_copy(const std::vector<T>& v) {
    T* p = nullptr;
    cuda_check(cudaMalloc(&p, v.size() * sizeof(T)), "cudaMalloc");
    cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
    return p;
}

constexpr int kExperts      = 24;
constexpr int kTopK         = 8;
constexpr int kStagingSlots = 64;
constexpr int kLandingSlots = 4; // a forked call's landing entries: three free frames and one -1

// ------------------------------------------------------------------------------------------- the oracle
//
// nvfp4_a8.h's arithmetic, written out from its specification.

// K elements as 8-bit slices: one int8 code per element and the FP32 scale of each 32-element slice.
struct A8Vector {
    std::vector<int> code;
    std::vector<float> scale;
};

A8Vector encode_a8(const std::uint16_t* v, int k) {
    A8Vector a{std::vector<int>(static_cast<std::size_t>(k)), std::vector<float>(static_cast<std::size_t>(k / 32))};
    for (int s = 0; s < k / 32; ++s) {
        unsigned key = 0; // the slice's largest magnitude, as BF16 bits
        for (int j = 0; j < 32; ++j) { key = std::max(key, static_cast<unsigned>(v[32 * s + j] & 0x7FFFU)); }
        const a8::SliceScale sc              = a8::slice_scale(key);
        a.scale[static_cast<std::size_t>(s)] = sc.d;
        for (int j = 0; j < 32; ++j) { a.code[static_cast<std::size_t>(32 * s + j)] = a8::code(v[32 * s + j], sc.inv); }
    }
    return a;
}

// Twice the E2M1 value of the weight at row r (0-15) and element k (0-15) of a 144-byte unit (layout
// nvfp4_expert_rg16_v1: byte 32 (k / 4) + 4 (r % 8) + k % 4, rows 8-15 in the high nibbles).
int weight_c2(const std::uint8_t* unit, int r, int k) {
    const unsigned byte = unit[32 * (k / 4) + 4 * (r % 8) + k % 4];
    return canon::e2m1_x2(r < 8 ? byte & 15U : byte >> 4);
}

// Row `row` of a matrix stored as row groups of `blocks` units against `a`: P_b exact per block; slice s's blocks
// 2s, then 2s + 1, add fma(fl32(P_b d_s), S(w), p) into partial s % partials, slices in ascending order, from +0;
// the butterfly p_i += p_(i + o), o = partials / 2 .. 1; y = bf16(fl32(fl32(p_0 2^-10) m)), the 2^-10 as two
// exact steps of 2^-5 (rounded once if subnormal).
std::uint16_t row_a8(const std::uint8_t* matrix, int blocks, int row, const A8Vector& a, int partials, float m) {
    const int group = row / 16, r = row % 16;
    float p[16] = {};
    for (int s = 0; s < blocks / 2; ++s) {
        for (int b = 2 * s; b <= 2 * s + 1; ++b) {
            const std::uint8_t* unit = matrix + (static_cast<std::size_t>(group) * blocks + b) * moe::kUnitBytes;
            std::int32_t pb          = 0;
            for (int k = 0; k < 16; ++k) { pb += weight_c2(unit, r, k) * a.code[static_cast<std::size_t>(16 * b + k)]; }
            const float sw  = static_cast<float>(canon::e4m3_scaled(unit[128 + r]));
            float& partial  = p[s % partials];
            partial         = canon::fma_rn(canon::mul_rn(static_cast<float>(pb), a.scale[static_cast<std::size_t>(s)]), sw,
                                            partial);
        }
    }
    for (int o = partials / 2; o >= 1; o /= 2) {
        for (int i = 0; i < o; ++i) { p[i] = canon::add_rn(p[i], p[i + o]); }
    }
    const float scaled = canon::mul_rn(canon::mul_rn(p[0], 0x1p-5F), 0x1p-5F);
    return canon::f32_to_bf16_rn(canon::mul_rn(scaled, m));
}

// One column through one expert: x -> A8 -> gate/up (16 partials) -> BF16 -> SwiGLU -> BF16 h -> A8 -> down (4
// partials) -> BF16.
std::vector<std::uint16_t> oracle_expert(const moe::ExpertShape& shape, const fixtures::Expert& e, const std::uint16_t* x) {
    const int H = shape.hidden, I = shape.intermediate;
    const A8Vector ax = encode_a8(x, H);
    std::vector<std::uint16_t> h(static_cast<std::size_t>(I));
    for (int i = 0; i < I; ++i) { // gate row 2i, up row 2i + 1
        const std::uint16_t g = row_a8(e.record.data(), H / 16, 2 * i, ax, 16, e.scales.alpha_gate);
        const std::uint16_t u = row_a8(e.record.data(), H / 16, 2 * i + 1, ax, 16, e.scales.alpha_up);
        h[static_cast<std::size_t>(i)] = canon::swiglu_bf16(g, u);
    }
    const A8Vector ah = encode_a8(h.data(), I);
    std::vector<std::uint16_t> y(static_cast<std::size_t>(H));
    for (int r = 0; r < H; ++r) {
        y[static_cast<std::size_t>(r)] =
            row_a8(e.record.data() + shape.gate_up_bytes(), I / 16, r, ah, 4, e.scales.alpha_down);
    }
    return y;
}

// ------------------------------------------------------------------------------------------- the bank

struct Bank {
    moe::ExpertShape shape;
    bool shared         = false; // the 122B's shared expert as the call's last job
    const char* name    = "";
    std::uint64_t stride = 0, pitch = 0;
    std::uint8_t* host        = nullptr; // pinned, mapped
    std::uint8_t* host_device = nullptr;
    std::vector<fixtures::Expert> experts;
    fixtures::Expert shared_expert;
    std::vector<moe::ExpertScales> scales; // the routed experts' (the CPU service's)
    std::vector<std::int32_t> frames;
    int resident = 0;
    std::uint8_t* d_frame_base    = nullptr; // the resident frames, then three free ones (landings)
    std::int32_t* d_frames        = nullptr;
    moe::ExpertScales* d_scales   = nullptr; // routed experts, then the shared expert's
    std::uint8_t* d_shared_record = nullptr;
    std::int32_t* d_prefetched    = nullptr; // copied-ahead slots of the experts e % 3 == 2 (never resident)
    std::uint8_t* d_prefetch_base = nullptr;
    std::int32_t* d_landing       = nullptr;
    std::int32_t* d_landed        = nullptr;
    std::uint8_t* d_staging       = nullptr;
    std::unique_ptr<moe::CpuMissService> service; // null when the CPU route refused the bank
};

void make_bank(Bank& bank, const moe::ExpertShape& shape, bool shared, const char* name, std::mt19937& rng) {
    bank.shape  = shape;
    bank.shared = shared;
    bank.name   = name;
    bank.stride = (shape.record_bytes() + 4095) / 4096 * 4096;
    bank.pitch  = bank.stride + 4096; // the frame pool's pitch exceeds the record stride
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&bank.host), bank.stride * kExperts, cudaHostAllocMapped),
               "cudaHostAlloc");
    std::memset(bank.host, 0, bank.stride * kExperts);
    for (int e = 0; e < kExperts; ++e) {
        bank.experts.push_back(fixtures::random_a16_expert(rng, shape));
        std::memcpy(bank.host + e * bank.stride, bank.experts.back().record.data(), shape.record_bytes());
        bank.scales.push_back(bank.experts.back().scales);
    }
    cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&bank.host_device), bank.host, 0),
               "cudaHostGetDevicePointer");
    bank.shared_expert   = fixtures::random_a16_expert(rng, shape);
    bank.d_shared_record = device_copy(bank.shared_expert.record);
    std::vector<moe::ExpertScales> all = bank.scales;
    all.push_back(bank.shared_expert.scales);
    bank.d_scales = device_copy(all);
    bank.frames.assign(kExperts, -1);
    for (int e = 0; e < kExperts; e += 3) { bank.frames[static_cast<std::size_t>(e)] = bank.resident++; }
    cuda_check(cudaMalloc(&bank.d_frame_base, bank.pitch * (bank.resident + 3)), "cudaMalloc");
    std::vector<std::int32_t> prefetched(kExperts, -1);
    int slots = 0;
    for (int e = 2; e < kExperts; e += 3) { prefetched[static_cast<std::size_t>(e)] = slots++; }
    cuda_check(cudaMalloc(&bank.d_prefetch_base, bank.pitch * slots), "cudaMalloc");
    for (int e = 0; e < kExperts; ++e) {
        const std::uint8_t* record = bank.host + e * bank.stride;
        if (bank.frames[static_cast<std::size_t>(e)] >= 0) {
            cuda_check(cudaMemcpy(bank.d_frame_base + bank.frames[static_cast<std::size_t>(e)] * bank.pitch, record,
                                  bank.stride, cudaMemcpyHostToDevice), "cudaMemcpy");
        }
        if (prefetched[static_cast<std::size_t>(e)] >= 0) {
            cuda_check(cudaMemcpy(bank.d_prefetch_base + prefetched[static_cast<std::size_t>(e)] * bank.pitch, record,
                                  bank.stride, cudaMemcpyHostToDevice), "cudaMemcpy");
        }
    }
    bank.d_frames     = device_copy(bank.frames);
    bank.d_prefetched = device_copy(prefetched);
    bank.d_landing    = device_copy(std::vector<std::int32_t>{bank.resident, bank.resident + 1, -1, bank.resident + 2});
    cuda_check(cudaMalloc(&bank.d_landed, sizeof(std::int32_t) * kLandingSlots), "cudaMalloc");
    cuda_check(cudaMalloc(&bank.d_staging, bank.pitch * kStagingSlots), "cudaMalloc");
    try {
        const std::vector<moe::CpuMissService::Layer> layers{
            {.records = bank.host, .record_stride = bank.stride, .scales = bank.scales.data()}};
        bank.service = std::make_unique<moe::CpuMissService>(
            layers, moe::CpuMissService::Options{.workers      = 4,
                                                 .max_jobs     = 8,
                                                 .max_columns  = 64,
                                                 .pcie_divisor = 2,
                                                 .cpus         = {},
                                                 .activation   = moe::ExpertActivation::kA16,
                                                 .narrow_a16   = moe::NarrowA16::kA8,
                                                 .nvfp4_a8     = true,
                                                 .shape        = shape});
    } catch (const std::exception& error) {
        check(false, std::string(name) + ": the CPU route takes NVFP4 W4A16 jobs on the 8-bit route (" + error.what() + ")");
    }
}

void free_bank(Bank& bank) {
    bank.service.reset(); // no call is pending
    cudaFree(bank.d_staging);
    cudaFree(bank.d_landed);
    cudaFree(bank.d_landing);
    cudaFree(bank.d_prefetched);
    cudaFree(bank.d_prefetch_base);
    cudaFree(bank.d_frames);
    cudaFree(bank.d_frame_base);
    cudaFree(bank.d_scales);
    cudaFree(bank.d_shared_record);
    cudaFreeHost(bank.host);
}

struct Streams {
    cudaStream_t fork = nullptr, split = nullptr, capture = nullptr;
    cudaEvent_t fork_events[2]{}, split_events[2]{};
    Streams() {
        for (cudaStream_t* s : {&fork, &split, &capture}) {
            cuda_check(cudaStreamCreateWithFlags(s, cudaStreamNonBlocking), "cudaStreamCreate");
        }
        for (auto& e : fork_events) { cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event"); }
        for (auto& e : split_events) { cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event"); }
    }
    ~Streams() {
        for (cudaStream_t s : {fork, split, capture}) { cudaStreamDestroy(s); }
        for (auto e : fork_events) { cudaEventDestroy(e); }
        for (auto e : split_events) { cudaEventDestroy(e); }
    }
    Streams(const Streams&)            = delete;
    Streams& operator=(const Streams&) = delete;
};

// ------------------------------------------------------------------------------------------- calls

struct Config {
    const char* name;
    int slots     = 0;
    bool fork     = false;
    bool split    = false; // a forked call's misses staged in two launches (MoeExpertSource::split_stream)
    bool landing  = false; // a forked call's misses landed in free frames (MoeExpertSource::landing)
    bool prefetch = false; // copied-ahead slots (MoeExpertSource::prefetched)
    bool cpu      = false; // CPU-served misses
    bool planned  = false; // the CPU plan fused with the route (moe_route_dispatch_plan)
    bool graph    = false; // captured into a CUDA graph, replayed twice
};

// The routed ids of `logits` (moe_route).
std::vector<std::int32_t> route_ids(const std::vector<float>& logits, int columns) {
    auto* d_logits      = device_copy(logits);
    std::int32_t* d_ids = nullptr;
    float *d_weights = nullptr, *d_gate = nullptr;
    cuda_check(cudaMalloc(&d_ids, sizeof(std::int32_t) * kTopK * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_weights, sizeof(float) * kTopK * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_gate, sizeof(float) * columns), "cudaMalloc");
    infernix::ops::MoeRouting routing{Tensor(d_ids, DType::I32, {kTopK, columns}),
                                      Tensor(d_weights, DType::FP32, {kTopK, columns}), Tensor(d_gate, DType::FP32, {columns})};
    infernix::ops::moe_route(Tensor(d_logits, DType::FP32, {kExperts + 1, columns}), kTopK, routing, nullptr);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(kTopK) * columns);
    cuda_check(cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
    cudaFree(d_gate);
    cudaFree(d_weights);
    cudaFree(d_ids);
    cudaFree(d_logits);
    return ids;
}

// The outputs of one layer call (moe_route, moe_dispatch, moe_experts) of `columns` columns: one eager run, or two
// graph replays; and the experts the CPU served.
struct CallResult {
    std::vector<std::vector<std::uint16_t>> runs;
    std::uint64_t cpu_served = 0;
};

CallResult layer_call(Bank& bank, Streams& streams, const std::vector<float>& logits, std::uint16_t* d_x, int columns,
                      const Config& config, moe::WideA16 wide) {
    const int H = bank.shape.hidden;
    const int entries = (kTopK + (bank.shared ? 1 : 0)) * columns;
    const int dispatch_experts = kExperts + (bank.shared ? 1 : 0);
    const int max_jobs         = std::min(kExperts, kTopK * columns) + (bank.shared ? 1 : 0);
    auto* d_logits             = device_copy(logits);
    std::int32_t* d_ids        = nullptr;
    float *d_weights = nullptr, *d_gate = nullptr;
    cuda_check(cudaMalloc(&d_ids, sizeof(std::int32_t) * kTopK * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_weights, sizeof(float) * kTopK * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_gate, sizeof(float) * columns), "cudaMalloc");
    infernix::ops::MoeRouting routing{Tensor(d_ids, DType::I32, {kTopK, columns}),
                                      Tensor(d_weights, DType::FP32, {kTopK, columns}), Tensor(d_gate, DType::FP32, {columns})};
    const Tensor logits_tensor(d_logits, DType::FP32, {kExperts + 1, columns});
    infernix::ops::moe_route(logits_tensor, kTopK, routing, nullptr);
    void* d_dispatch = nullptr;
    cuda_check(cudaMalloc(&d_dispatch, infernix::ops::moe_dispatch_bytes(dispatch_experts, entries)), "cudaMalloc");
    auto dispatch     = infernix::ops::carve_moe_dispatch(d_dispatch, dispatch_experts, entries);
    void* d_workspace = nullptr;
    cuda_check(cudaMalloc(&d_workspace, infernix::ops::moe_experts_workspace_bytes(bank.shape, max_jobs, entries, columns,
                                                                                  moe::ExpertActivation::kA16, wide)),
               "cudaMalloc");
    std::uint16_t* d_out    = nullptr;
    const std::size_t bytes = sizeof(std::uint16_t) * H * entries;
    cuda_check(cudaMalloc(&d_out, bytes), "cudaMalloc");

    infernix::ops::MoeExpertSource source{.frame_base    = bank.d_frame_base,
                                          .frames        = bank.d_frames,
                                          .host_records  = bank.host_device,
                                          .record_stride = bank.stride,
                                          .scales        = bank.d_scales,
                                          .staging_base  = config.slots > 0 ? bank.d_staging : nullptr,
                                          .staging_slots = config.slots};
    source.activation   = moe::ExpertActivation::kA16;
    source.wide_a16     = wide;
    source.narrow_a16   = moe::NarrowA16::kA8;
    source.frame_stride = bank.pitch;
    source.shape        = bank.shape;
    if (bank.shared) {
        source.shared_expert = kExperts;
        source.shared_record = bank.d_shared_record;
    }
    if (config.fork) {
        source.fork_stream    = streams.fork;
        source.fork_events[0] = streams.fork_events[0];
        source.fork_events[1] = streams.fork_events[1];
    }
    if (config.split) {
        source.split_stream    = streams.split;
        source.split_events[0] = streams.split_events[0];
        source.split_events[1] = streams.split_events[1];
    }
    if (config.landing) {
        source.landing       = bank.d_landing;
        source.landed        = bank.d_landed;
        source.landing_slots = kLandingSlots;
    }
    if (config.prefetch) {
        source.prefetched    = bank.d_prefetched;
        source.prefetch_base = bank.d_prefetch_base;
    }
    if (config.cpu) {
        bank.service->wake();
        source.cpu = bank.service->channel(0);
    }
    Tensor tx(d_x, DType::BF16, {H, columns});
    Tensor out(d_out, DType::BF16, {H, entries});
    CallResult result;
    const auto read = [&] {
        std::vector<std::uint16_t> got(static_cast<std::size_t>(H) * entries);
        cuda_check(cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy");
        result.runs.push_back(std::move(got));
    };
    if (config.graph) {
        cudaGraph_t graph    = nullptr;
        cudaGraphExec_t exec = nullptr;
        cuda_check(cudaDeviceSynchronize(), "moe_route");
        cuda_check(cudaStreamBeginCapture(streams.capture, cudaStreamCaptureModeThreadLocal), "capture");
        infernix::ops::moe_dispatch(routing, kExperts, dispatch, nullptr, streams.capture, bank.shared);
        infernix::ops::moe_experts(tx, dispatch, source, kTopK, max_jobs, d_workspace, out, streams.capture);
        cuda_check(cudaStreamEndCapture(streams.capture, &graph), "capture");
        cuda_check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        for (int replay = 0; replay < 2; ++replay) {
            cuda_check(cudaMemsetAsync(d_out, 0xFF, bytes, streams.capture), "cudaMemsetAsync");
            cuda_check(cudaGraphLaunch(exec, streams.capture), "cudaGraphLaunch");
            cuda_check(cudaStreamSynchronize(streams.capture), "graph replay");
            read();
        }
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
    } else {
        cuda_check(cudaMemset(d_out, 0xFF, bytes), "cudaMemset");
        const std::uint64_t served = config.cpu ? bank.service->served_experts() : 0;
        bool planned               = false;
        if (config.planned) {
            planned = infernix::ops::moe_route_dispatch_plan(logits_tensor, kTopK, routing, dispatch, nullptr, tx, source,
                                                             max_jobs, d_workspace, out, nullptr,
                                                             infernix::ops::RouterLogits::kFp32, bank.shared);
        } else {
            infernix::ops::moe_dispatch(routing, kExperts, dispatch, nullptr, nullptr, bank.shared);
        }
        infernix::ops::moe_experts(tx, dispatch, source, kTopK, max_jobs, d_workspace, out, nullptr, true, planned);
        cuda_check(cudaDeviceSynchronize(), "moe_experts");
        if (config.cpu) { result.cpu_served = bank.service->served_experts() - served; }
        read();
    }
    cudaFree(d_out);
    cudaFree(d_workspace);
    cudaFree(d_dispatch);
    cudaFree(d_gate);
    cudaFree(d_weights);
    cudaFree(d_ids);
    cudaFree(d_logits);
    return result;
}

// Entry i's expert applied to its column (routed entries t k + slot, then the shared job's k T + t): the oracle on
// the route, else the exact W4A16 expert (the CPU engine). Threads over entries.
std::vector<std::uint16_t> expected_outputs(const Bank& bank, const std::vector<std::int32_t>& ids,
                                            const std::vector<std::uint16_t>& x, int columns, bool a8_route) {
    const int H = bank.shape.hidden, routed = kTopK * columns;
    const int entries = routed + (bank.shared ? columns : 0);
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(H) * entries);
    std::atomic<int> next{0};
    const auto work = [&] {
        for (int i = next++; i < entries; i = next++) {
            const bool shared         = i >= routed;
            const int t               = shared ? i - routed : i / kTopK;
            const fixtures::Expert& e = shared ? bank.shared_expert : bank.experts[static_cast<std::size_t>(ids[static_cast<std::size_t>(i)])];
            const std::uint16_t* xv   = &x[static_cast<std::size_t>(t) * H];
            std::uint16_t* y          = &expected[static_cast<std::size_t>(i) * H];
            if (a8_route) {
                const std::vector<std::uint16_t> v = oracle_expert(bank.shape, e, xv);
                std::memcpy(y, v.data(), sizeof(std::uint16_t) * H);
            } else {
                const std::uint16_t* xp[1] = {xv};
                std::uint16_t* yp[1]       = {y};
                moe::expert_forward_a16(moe::best_cpu_isa(), bank.shape, e.record.data(), e.scales, 1, xp, yp);
            }
        }
    };
    std::vector<std::thread> threads;
    for (unsigned w = 0; w < std::max(1U, std::thread::hardware_concurrency()); ++w) { threads.emplace_back(work); }
    for (auto& th : threads) { th.join(); }
    return expected;
}

void compare(const std::vector<std::uint16_t>& got, const std::vector<std::uint16_t>& expected, int H, int routed,
             const std::string& where) {
    std::size_t bad = 0, bad_shared = 0, first = got.size();
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] == expected[i]) { continue; }
        ++bad;
        bad_shared += i >= static_cast<std::size_t>(H) * routed ? 1 : 0;
        first = std::min(first, i);
    }
    if (bad != 0) {
        std::fprintf(stderr, "%s: %zu of %zu outputs differ (%zu of the shared expert's); first: entry %zu row %zu, 0x%04x for 0x%04x\n",
                     where.c_str(), bad, got.size(), bad_shared, first / static_cast<std::size_t>(H),
                     first % static_cast<std::size_t>(H), static_cast<unsigned>(got[first]),
                     static_cast<unsigned>(expected[first]));
    }
    check(bad == 0, where + ": every output equals the oracle's bits");
}

// x columns (see the header): wide-range values, a ±0 slice, a slice below 2^-120 (scale 0), a slice in
// [2^-120, 2^-119) (a subnormal scale), and an Inf in column 1.
std::vector<std::uint16_t> activations(std::mt19937& rng, int columns, int hidden) {
    std::vector<std::uint16_t> x = fixtures::wide_range_activations(rng, columns, hidden);
    const int slices             = hidden / 32;
    for (int t = 0; t < columns; ++t) {
        std::uint16_t* v = x.data() + static_cast<std::size_t>(t) * hidden;
        const int zero = (3 + 7 * t) % slices, below = (zero + 1 + t % 5) % slices, tiny = (below + 1 + t % 3) % slices;
        for (int j = 0; j < 32; ++j) {
            v[32 * zero + j]  = static_cast<std::uint16_t>((j & 1) != 0 ? 0x8000U : 0U);
            v[32 * below + j] = static_cast<std::uint16_t>((rng() & 0x8000U) | (2U << 7) | (rng() & 0x7FU));
            v[32 * tiny + j]  = static_cast<std::uint16_t>((rng() & 0x8000U) | (7U << 7) | (rng() & 0x7FU));
        }
    }
    if (columns > 1) { x[static_cast<std::size_t>(hidden) + 5 * 32 + 3] = std::uint16_t{0x7F80}; } // column 1, slice 5
    return x;
}

std::vector<float> random_logits(std::mt19937& rng, int columns) {
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> logits(static_cast<std::size_t>(kExperts + 1) * columns);
    for (auto& v : logits) { v = n(rng); }
    if (columns > 8) {
        // Two favoured experts take most columns: jobs of more than eight columns (passes of eight).
        for (int t = 0; t < columns; ++t) {
            logits[static_cast<std::size_t>(t) * (kExperts + 1) + 1] += 3.0F;
            logits[static_cast<std::size_t>(t) * (kExperts + 1) + 2] += 2.0F;
        }
    }
    return logits;
}

void run(Bank& bank, Streams& streams, int columns, std::mt19937& rng) {
    const int H = bank.shape.hidden, routed = kTopK * columns;
    const bool a8_route  = columns <= 128;
    const auto logits    = random_logits(rng, columns);
    auto x               = activations(rng, columns, H);
    auto* d_x            = device_copy(x);
    const auto ids       = route_ids(logits, columns);
    const auto expected  = expected_outputs(bank, ids, x, columns, a8_route);
    int most = 0; // the most columns any routed expert holds
    for (int e = 0; e < kExperts; ++e) { most = std::max<int>(most, static_cast<int>(std::count(ids.begin(), ids.end(), e))); }

    const Config configs[] = {{.name = "zero-copy"},
                              {.name = "3 staging slots", .slots = 3},
                              {.name = "64 staging slots", .slots = kStagingSlots},
                              {.name = "copied-ahead slots", .slots = kStagingSlots, .prefetch = true},
                              {.name = "forked", .slots = kStagingSlots, .fork = true},
                              {.name = "forked, split staging", .slots = kStagingSlots, .fork = true, .split = true},
                              {.name = "forked, landed in free frames", .slots = kStagingSlots, .fork = true, .landing = true},
                              {.name = "CPU-served misses", .slots = kStagingSlots, .cpu = true},
                              {.name = "forked, CPU-served misses, fused plan", .slots = kStagingSlots, .fork = true,
                               .cpu = true, .planned = true},
                              {.name = "graph", .slots = kStagingSlots, .graph = true},
                              {.name = "forked, split staging, graph", .slots = kStagingSlots, .fork = true, .split = true,
                               .graph = true}};
    struct Variant {
        moe::WideA16 wide;
        const char* name;
    };
    const Variant variants[] = {{moe::WideA16::kExact, "exact wide workspace"},
                                {moe::WideA16::kA4Residual, "A4 residual wide workspace"}};
    for (const Variant& variant : variants) {
        // Past the route only the exact wide route's bits are known exactly.
        if (!a8_route && variant.wide != moe::WideA16::kExact) { continue; }
        for (const Config& config : configs) {
            if (config.fork && columns > moe::kMaxColumns) { continue; }        // forked calls are narrow
            if (config.cpu && (columns > 16 || bank.service == nullptr)) { continue; }
            if (!a8_route && (config.fork || config.cpu || config.graph)) { continue; }
            // The A4 residual workspace: one serial and one forked layout suffice (the region is the difference).
            if (variant.wide != moe::WideA16::kExact && std::strcmp(config.name, "64 staging slots") != 0 &&
                std::strcmp(config.name, "forked") != 0) {
                continue;
            }
            const CallResult result = layer_call(bank, streams, logits, d_x, columns, config, variant.wide);
            const std::string where = std::string(bank.name) + ", T " + std::to_string(columns) + " (most columns per expert " +
                                      std::to_string(most) + ", " + (a8_route ? "8-bit route" : "past the route") + "), " +
                                      variant.name + ", " + config.name;
            for (std::size_t r = 0; r < result.runs.size(); ++r) {
                compare(result.runs[r], expected, H, routed, result.runs.size() > 1 ? where + " (replay " + std::to_string(r + 1) + ")" : where);
            }
            if (config.cpu) { check(result.cpu_served > 0, where + ": the CPU served experts"); }
        }
    }
    std::printf("%s, T %d (most columns per expert %d): checked\n", bank.name, columns, most);
    cudaFree(d_x);
}

// An expert's output does not depend on its column count in the call: column 3 of a nine-column call in which
// every column routes to expert 1 (its job holds nine columns: two passes) against the same column alone (its job
// one column).
void invariance(Bank& bank, Streams& streams, std::mt19937& rng) {
    constexpr int kWide = 9, kColumn = 3, kExpert = 1;
    const int H = bank.shape.hidden;
    auto logits = random_logits(rng, kWide);
    for (int t = 0; t < kWide; ++t) { logits[static_cast<std::size_t>(t) * (kExperts + 1) + kExpert] += 20.0F; }
    const std::vector<float> one(logits.begin() + kColumn * (kExperts + 1), logits.begin() + (kColumn + 1) * (kExperts + 1));
    auto x = activations(rng, kWide, H);
    std::vector<std::uint16_t> x_one(x.begin() + static_cast<std::ptrdiff_t>(kColumn) * H,
                                     x.begin() + static_cast<std::ptrdiff_t>(kColumn + 1) * H);
    auto* d_x     = device_copy(x);
    auto* d_x_one = device_copy(x_one);
    const auto ids_wide = route_ids(logits, kWide), ids_one = route_ids(one, 1);
    const int in_wide = static_cast<int>(std::count(ids_wide.begin(), ids_wide.end(), kExpert));
    int slot_wide = -1, slot_one = -1;
    for (int k = 0; k < kTopK; ++k) {
        if (ids_wide[static_cast<std::size_t>(kColumn * kTopK + k)] == kExpert) { slot_wide = k; }
        if (ids_one[static_cast<std::size_t>(k)] == kExpert) { slot_one = k; }
    }
    const std::string where = std::string(bank.name) + ": expert " + std::to_string(kExpert) + " at nine columns and at one";
    check(in_wide == kWide && slot_wide >= 0 && slot_one >= 0, where + ": the routing gives the expert every column");
    if (slot_wide >= 0 && slot_one >= 0) {
        const Config config{.name = "64 staging slots", .slots = kStagingSlots};
        const CallResult wide = layer_call(bank, streams, logits, d_x, kWide, config, moe::WideA16::kExact);
        const CallResult single = layer_call(bank, streams, one, d_x_one, 1, config, moe::WideA16::kExact);
        const auto* a = &wide.runs[0][static_cast<std::size_t>(kColumn * kTopK + slot_wide) * H];
        const auto* b = &single.runs[0][static_cast<std::size_t>(slot_one) * H];
        const std::vector<std::uint16_t> want = oracle_expert(bank.shape, bank.experts[kExpert], x_one.data());
        check(std::equal(a, a + H, b), where + ": the same bits");
        check(std::equal(a, a + H, want.begin()), where + ": the oracle's bits");
    }
    cudaFree(d_x_one);
    cudaFree(d_x);
}

} // namespace

INFERNIX_OP_TEST(offloaded_moe_w4a8) {
    std::mt19937 rng(20261027);
    Streams streams;
    struct Geometry {
        moe::ExpertShape shape;
        bool shared;
        const char* name;
    };
    for (const Geometry& geometry : {Geometry{moe::kFlashNextShape, false, "Qwen3.8-Flash-Next geometry"},
                                     Geometry{moe::kQwen3_5_122BShape, true, "Qwen3.5-122B-A10B geometry"}}) {
        Bank bank;
        make_bank(bank, geometry.shape, geometry.shared, geometry.name, rng);
        // Decode and verification widths (1, 3, 8), jobs of more than eight columns (9, 16: two passes), the
        // widest calls (64, 128: up to sixteen passes; the shared job wide) and one past the route (129).
        for (const int columns : {1, 3, 8, 9, 16, 64, 128, 129}) { run(bank, streams, columns, rng); }
        invariance(bank, streams, rng);
        free_bank(bank);
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("offloaded_moe_w4a8: ok\n");
    return 0;
}
