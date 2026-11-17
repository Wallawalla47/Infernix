// offloaded_sparse_moe's shared job on NVFP4 banks at Qwen3.8-Flash-Next's expert geometry (H 2560, I 640; NVIDIA's
// NVFP4-QAD: routed and shared experts NVFP4 with their own activation scales, design §6.1.3, §8.6) in every
// arithmetic its artifacts run: W4A4 (the default, §16.2; equal and distinct gate/up input scales, so both the
// tensor-core and the one-column narrow kernels), W4A16 exact (--a16-activations, §16.2.1), the expert-a4-residual
// fast paths' two NVFP4 terms (registered for --a16-activations' configuration, §19.3.23) and the 8-bit route
// (--a8-activations, §19.3.31). moe_dispatch puts the shared expert last; moe_experts reads the routed records from
// device frames, zero-copy from the pinned host bank, through 3 or 64 staging slots, on the forked route (one pass,
// split staging) and (W4A4, W4A16) with CPU-served misses, at decode, verification and prefill widths: the shared job
// takes the narrow route through eight columns and the wide one beyond (the 8-bit kernels through 128 columns and its
// int8 tiles beyond).
//
// Oracles:
// - every output of an exact route, routed and shared, against the CPU engine's output of its (column, expert) pair,
//   bit for bit (expert_forward, expert_forward_a16, each checked against FP64 in its CPU test), whatever the
//   record's location, the staging pass or the CPU's share: W4A16 at every width (its wide route is exact limbs),
//   W4A4 on the narrow route (jobs of at most eight columns, or distinct up scales at any width); W4A4's wide route
//   accumulates on FP4 tensor cores in FP32, held to its FP64 oracle by test_offloaded_moe_wide, and its shared jobs
//   (every call of more than eight columns) are checked against the routed reference below;
// - in the A4-residual and 8-bit arithmetics, the routed outputs that the CPU engine shares bit for bit (experts of
//   at most eight columns, expert_forward_a4r; calls of at most 128 columns, expert_forward_a8), and every shared
//   output against the same record run as a routed expert of the same call width (the shared job takes the routed
//   experts' kernels in the bank's arithmetic, so their bits must be equal);
// - the combine of the routed and shared outputs against its FP64 formula within its FP32 rounding bound.

#include "ops/op_family.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/offloaded_sparse_moe/cpu/a4r_expert.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"
#include "ops/offloaded_sparse_moe/cpu/w4a8_expert.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace moe      = infernix::ops::offloaded_moe;
namespace fixtures = infernix::test::offloaded_moe;
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

constexpr moe::ExpertShape kShape = moe::kFlashNextShape;
constexpr int H = kShape.hidden;
constexpr int kExperts = 24, kTopK = 10;
constexpr int kA8CallColumns = 128; // nvfp4_a8::kMaxCallColumns: wider calls take the int8 tiles

// The arithmetic of a call: the bank's activation and the W4A16 banks' narrow and wide routes.
enum class Mode { kA4, kA4SplitUp, kA16, kA4Residual, kA8 };

const char* mode_name(Mode m) {
    switch (m) {
    case Mode::kA4: return "W4A4";
    case Mode::kA4SplitUp: return "W4A4, distinct up input scales";
    case Mode::kA16: return "W4A16 exact";
    case Mode::kA4Residual: return "W4A16, expert-a4-residual(-decode) terms";
    case Mode::kA8: return "W4A16, --a8-activations route";
    }
    return "?";
}

bool a4(Mode m) { return m == Mode::kA4 || m == Mode::kA4SplitUp; }
bool exact(Mode m) { return a4(m) || m == Mode::kA16; }

void apply_mode(Mode m, infernix::ops::MoeExpertSource& source) {
    source.activation = a4(m) ? moe::ExpertActivation::kA4 : moe::ExpertActivation::kA16;
    source.a4_shared_input = m == Mode::kA4;
    source.narrow_a16 = m == Mode::kA4Residual ? moe::NarrowA16::kA4Residual
                        : m == Mode::kA8       ? moe::NarrowA16::kA8
                                               : moe::NarrowA16::kExact;
    source.wide_a16 = m == Mode::kA4Residual ? moe::WideA16::kA4Residual
                      : m == Mode::kA8       ? moe::WideA16::kA8
                                             : moe::WideA16::kExact;
}

moe::WideA16 workspace_wide(Mode m) {
    return m == Mode::kA4Residual ? moe::WideA16::kA4Residual : m == Mode::kA8 ? moe::WideA16::kA8 : moe::WideA16::kExact;
}

// One (column, expert) output of the CPU engine in the mode's arithmetic.
void cpu_expert(Mode m, const fixtures::Expert& e, const std::uint16_t* x, std::uint16_t* y) {
    const std::uint16_t* xp[1] = {x};
    std::uint16_t* yp[1]       = {y};
    switch (m) {
    case Mode::kA4:
    case Mode::kA4SplitUp: moe::expert_forward(moe::best_cpu_isa(), e.record.data(), e.scales, 1, xp, yp); return;
    case Mode::kA16: moe::expert_forward_a16(moe::best_cpu_isa(), kShape, e.record.data(), e.scales, 1, xp, yp); return;
    case Mode::kA4Residual: moe::expert_forward_a4r(moe::best_cpu_isa(), e.record.data(), e.scales, 1, xp, yp); return;
    case Mode::kA8: moe::expert_forward_a8(moe::best_cpu_isa(), kShape, e.record.data(), e.scales, 1, xp, yp); return;
    }
}

void run(int columns, Mode mode, std::mt19937& rng) {
    const std::uint64_t stride = (kShape.record_bytes() + 4095) / 4096 * 4096;
    // The routed bank, then (record kExperts) the shared expert's copy that the reference call routes as an expert.
    std::uint8_t* host = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host), stride * (kExperts + 1), cudaHostAllocMapped),
               "cudaHostAlloc");
    std::memset(host, 0, stride * (kExperts + 1));
    const auto random = [&] {
        return a4(mode) ? fixtures::random_expert(rng, mode == Mode::kA4SplitUp, kShape) : fixtures::random_a16_expert(rng, kShape);
    };
    std::vector<fixtures::Expert> bank;
    std::vector<moe::ExpertScales> scales;
    for (int e = 0; e < kExperts; ++e) {
        bank.push_back(random());
        std::memcpy(host + e * stride, bank.back().record.data(), kShape.record_bytes());
        scales.push_back(bank.back().scales);
    }
    // The shared expert: its record in device memory, its scales after the routed experts'.
    const fixtures::Expert shared_expert = random();
    std::memcpy(host + kExperts * stride, shared_expert.record.data(), kShape.record_bytes());
    auto* d_shared_record = device_copy(shared_expert.record);
    std::vector<moe::ExpertScales> all_scales = scales;
    all_scales.push_back(shared_expert.scales);
    std::uint8_t* host_device = nullptr;
    cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_device), host, 0), "cudaHostGetDevicePointer");
    // A third of the routed experts resident; the shared expert's copy is not.
    std::vector<std::int32_t> frames(kExperts + 1, -1);
    int resident = 0;
    for (int e = 1; e < kExperts; e += 3) { frames[static_cast<std::size_t>(e)] = resident++; }
    std::uint8_t* d_frame_base = nullptr;
    cuda_check(cudaMalloc(&d_frame_base, stride * resident), "cudaMalloc");
    for (int e = 0; e < kExperts; ++e) {
        if (frames[static_cast<std::size_t>(e)] >= 0) {
            cuda_check(cudaMemcpy(d_frame_base + frames[static_cast<std::size_t>(e)] * stride, host + e * stride, stride,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
        }
    }
    auto* d_frames = device_copy(frames);
    auto* d_scales = device_copy(all_scales);

    // Routing: kTopK distinct experts per column, so the wide widths give experts of more than eight columns.
    std::vector<std::int32_t> ids(static_cast<std::size_t>(kTopK) * columns);
    std::vector<float> weights(ids.size(), 0.1F), gate(static_cast<std::size_t>(columns), 0.5F);
    for (int t = 0; t < columns; ++t) {
        std::vector<int> order(kExperts);
        for (int e = 0; e < kExperts; ++e) { order[static_cast<std::size_t>(e)] = e; }
        std::shuffle(order.begin(), order.end(), rng);
        for (int k = 0; k < kTopK; ++k) { ids[static_cast<std::size_t>(t) * kTopK + k] = order[static_cast<std::size_t>(k)]; }
    }
    auto* d_ids     = device_copy(ids);
    auto* d_weights = device_copy(weights);
    auto* d_gate    = device_copy(gate);
    infernix::ops::MoeRouting routing{Tensor(d_ids, DType::I32, {kTopK, columns}),
                                      Tensor(d_weights, DType::FP32, {kTopK, columns}),
                                      Tensor(d_gate, DType::FP32, {columns})};
    // The dispatch holds the shared job: E + 1 experts, k*T + T entries.
    const int entries = (kTopK + 1) * columns;
    void* d_dispatch  = nullptr;
    cuda_check(cudaMalloc(&d_dispatch, infernix::ops::moe_dispatch_bytes(kExperts + 1, entries)), "cudaMalloc");
    auto dispatch = infernix::ops::carve_moe_dispatch(d_dispatch, kExperts + 1, entries);

    const auto x = fixtures::random_activations(rng, columns, H);
    auto* d_x    = device_copy(x);

    // Expected outputs: column t * k + slot = expert ids[slot, t] applied to x[:, t], then column k * T + t = the shared
    // expert applied to x[:, t]; `cpu_checked` marks the outputs the CPU engine's bits bind in this arithmetic.
    std::vector<int> per_expert(kExperts, 0);
    for (const int id : ids) { ++per_expert[static_cast<std::size_t>(id)]; }
    // The narrow routes share the CPU engine's bits. W4A4 jobs of more than eight columns with equal gate and up input
    // scales take the wide route, whose tensor-core FP32 accumulation is held to its FP64 oracle by
    // test_offloaded_moe_wide (it equals the exact sum only where FP32 holds it); W4A16's wide route is exact limbs.
    const auto cpu_binds = [&](int entry) {
        const bool shared = entry >= kTopK * columns;
        const int count   = shared ? columns : per_expert[static_cast<std::size_t>(ids[static_cast<std::size_t>(entry)])];
        switch (mode) {
        case Mode::kA16:
        case Mode::kA4SplitUp: return true;
        case Mode::kA4: return count <= moe::kMaxColumns;
        case Mode::kA8: return !shared && columns <= kA8CallColumns;
        case Mode::kA4Residual: return !shared && count <= moe::kMaxColumns;
        }
        return false;
    };
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(H) * entries, 0);
    {
        std::atomic<int> next{0};
        const auto work = [&] {
            for (int i = next++; i < entries; i = next++) {
                if (!cpu_binds(i)) { continue; }
                const bool shared = i >= kTopK * columns;
                const int t       = shared ? i - kTopK * columns : i / kTopK;
                const fixtures::Expert& e =
                    shared ? shared_expert : bank[static_cast<std::size_t>(ids[static_cast<std::size_t>(i)])];
                cpu_expert(mode, e, &x[static_cast<std::size_t>(t) * H], &expected[static_cast<std::size_t>(i) * H]);
            }
        };
        std::vector<std::thread> threads;
        for (unsigned w = 0; w < std::min(16U, std::max(1U, std::thread::hardware_concurrency())); ++w) {
            threads.emplace_back(work);
        }
        for (auto& th : threads) { th.join(); }
    }
    const int most = *std::max_element(per_expert.begin(), per_expert.end());

    const int max_jobs = std::min(kExperts, kTopK * columns) + 1; // and the shared job
    std::uint8_t* d_staging = nullptr;
    cuda_check(cudaMalloc(&d_staging, stride * 64), "cudaMalloc");
    const moe::ExpertActivation activation = a4(mode) ? moe::ExpertActivation::kA4 : moe::ExpertActivation::kA16;
    void* d_workspace = nullptr;
    cuda_check(cudaMalloc(&d_workspace, infernix::ops::moe_experts_workspace_bytes(kShape, max_jobs, entries, columns,
                                                                                  activation, workspace_wide(mode))),
               "cudaMalloc");
    std::uint16_t* d_out = nullptr;
    cuda_check(cudaMalloc(&d_out, expected.size() * sizeof(std::uint16_t)), "cudaMalloc");
    cudaStream_t fork_stream = nullptr, split_stream = nullptr;
    cudaEvent_t fork_events[2]{}, split_events[2]{};
    cuda_check(cudaStreamCreateWithFlags(&fork_stream, cudaStreamNonBlocking), "stream");
    cuda_check(cudaStreamCreateWithFlags(&split_stream, cudaStreamNonBlocking), "stream");
    for (auto& e : fork_events) { cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event"); }
    for (auto& e : split_events) { cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event"); }
    // The CPU-served configurations run the exact arithmetics (the other routes' CPU jobs are their own tests').
    const std::vector<moe::CpuMissService::Layer> layers{{.records = host, .record_stride = stride, .scales = scales.data()}};
    std::unique_ptr<moe::CpuMissService> service;
    if (exact(mode)) {
        service = std::make_unique<moe::CpuMissService>(
            layers, moe::CpuMissService::Options{.workers = 6, .max_jobs = 8, .max_columns = 64, .pcie_divisor = 2,
                                                 .cpus = {}, .activation = activation, .shape = kShape});
    }

    // The reference of the shared outputs: the shared expert's host copy (expert kExperts) as the only routed expert
    // of a one-slot call of the same width, in the same arithmetic.
    std::vector<std::uint16_t> reference(static_cast<std::size_t>(H) * columns);
    {
        std::vector<std::int32_t> ref_ids(static_cast<std::size_t>(columns), kExperts);
        std::vector<float> ref_weights(static_cast<std::size_t>(columns), 1.0F);
        auto* d_ref_ids     = device_copy(ref_ids);
        auto* d_ref_weights = device_copy(ref_weights);
        infernix::ops::MoeRouting ref_routing{Tensor(d_ref_ids, DType::I32, {1, columns}),
                                              Tensor(d_ref_weights, DType::FP32, {1, columns}),
                                              Tensor(d_gate, DType::FP32, {columns})};
        void* d_ref_dispatch = nullptr;
        cuda_check(cudaMalloc(&d_ref_dispatch, infernix::ops::moe_dispatch_bytes(kExperts + 1, columns)), "cudaMalloc");
        auto ref_dispatch = infernix::ops::carve_moe_dispatch(d_ref_dispatch, kExperts + 1, columns);
        infernix::ops::moe_dispatch(ref_routing, kExperts + 1, ref_dispatch, nullptr, nullptr);
        void* d_ref_workspace = nullptr;
        cuda_check(cudaMalloc(&d_ref_workspace, infernix::ops::moe_experts_workspace_bytes(kShape, 1, columns, columns,
                                                                                          activation, workspace_wide(mode))),
                   "cudaMalloc");
        std::uint16_t* d_ref_out = nullptr;
        cuda_check(cudaMalloc(&d_ref_out, reference.size() * sizeof(std::uint16_t)), "cudaMalloc");
        infernix::ops::MoeExpertSource source{.frame_base    = d_frame_base,
                                              .frames        = d_frames,
                                              .host_records  = host_device,
                                              .record_stride = stride,
                                              .scales        = d_scales,
                                              .staging_base  = d_staging,
                                              .staging_slots = 64};
        apply_mode(mode, source);
        source.shape = kShape;
        Tensor tx(d_x, DType::BF16, {H, columns});
        Tensor out(d_ref_out, DType::BF16, {H, columns});
        infernix::ops::moe_experts(tx, ref_dispatch, source, 1, 1, d_ref_workspace, out, nullptr, true, false);
        cuda_check(cudaDeviceSynchronize(), "moe_experts reference");
        cuda_check(cudaMemcpy(reference.data(), d_ref_out, reference.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                   "cudaMemcpy");
        if (exact(mode) && (mode != Mode::kA4 || columns <= moe::kMaxColumns)) {
            // The routed reference itself equals the CPU engine's bits on an exact route.
            const std::size_t shared_first = static_cast<std::size_t>(H) * kTopK * columns;
            check(std::equal(reference.begin(), reference.end(), expected.begin() + static_cast<std::ptrdiff_t>(shared_first)),
                  "T " + std::to_string(columns) + ", " + mode_name(mode) +
                      ": the shared record run as a routed expert equals the CPU engine's bits");
        }
        cudaFree(d_ref_out);
        cudaFree(d_ref_workspace);
        cudaFree(d_ref_dispatch);
        cudaFree(d_ref_weights);
        cudaFree(d_ref_ids);
    }

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
                              {64, true, true, "forked, CPU-served misses"}};
    std::vector<std::uint16_t> last; // the last configuration's outputs, which the combine reads
    for (const auto& config : configs) {
        if (config.fork && columns > moe::kMaxColumns) { continue; }
        if (config.cpu && (columns > 64 || !service)) { continue; }
        infernix::ops::moe_dispatch(routing, kExperts, dispatch, nullptr, nullptr, /*shared_job=*/true);
        cuda_check(cudaMemset(d_out, 0xFF, expected.size() * sizeof(std::uint16_t)), "cudaMemset");
        infernix::ops::MoeExpertSource source{.frame_base    = d_frame_base,
                                              .frames        = d_frames,
                                              .host_records  = host_device,
                                              .record_stride = stride,
                                              .scales        = d_scales,
                                              .staging_base  = config.slots > 0 ? d_staging : nullptr,
                                              .staging_slots = config.slots};
        apply_mode(mode, source);
        source.shape         = kShape;
        source.shared_expert = kExperts;
        source.shared_record = d_shared_record;
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
        if (config.cpu) { source.cpu = service->channel(0); }
        Tensor tx(d_x, DType::BF16, {H, columns});
        Tensor out(d_out, DType::BF16, {H, entries});
        const std::uint64_t served_before = service ? service->served_experts() : 0;
        infernix::ops::moe_experts(tx, dispatch, source, kTopK, max_jobs, d_workspace, out, nullptr, true, false);
        cuda_check(cudaDeviceSynchronize(), "moe_experts");
        std::vector<std::uint16_t> got(expected.size());
        cuda_check(cudaMemcpy(got.data(), d_out, got.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
        std::size_t bad = 0, bound = 0, bad_shared = 0;
        for (int i = 0; i < entries; ++i) {
            const std::size_t at = static_cast<std::size_t>(i) * H;
            const bool shared    = i >= kTopK * columns;
            if (cpu_binds(i)) {
                ++bound;
                bad += !std::equal(got.begin() + static_cast<std::ptrdiff_t>(at),
                                   got.begin() + static_cast<std::ptrdiff_t>(at + H),
                                   expected.begin() + static_cast<std::ptrdiff_t>(at));
            }
            if (shared) {
                const std::size_t t = static_cast<std::size_t>(i - kTopK * columns);
                bad_shared += !std::equal(got.begin() + static_cast<std::ptrdiff_t>(at),
                                          got.begin() + static_cast<std::ptrdiff_t>(at + H),
                                          reference.begin() + static_cast<std::ptrdiff_t>(t * H));
            }
        }
        const std::string where = "T " + std::to_string(columns) + ", " + mode_name(mode) + " (most columns per expert " +
                                  std::to_string(most) + "), " + config.name;
        std::printf("%s: %zu of %zu CPU-bound outputs differ, %zu of %d shared outputs differ from the routed reference",
                    where.c_str(), bad, bound, bad_shared, columns);
        if (config.cpu) {
            std::printf(" (CPU served %llu experts)", static_cast<unsigned long long>(service->served_experts() - served_before));
        }
        std::printf("\n");
        check(bad == 0, where + ": every CPU-bound routed and shared output equals the CPU engine's bits");
        check(bad_shared == 0, where + ": every shared output equals the record run as a routed expert");
        last = std::move(got);
    }

    // The combine reads the routed entries and the shared job's T columns as its `shared` input (the last call's).
    if (exact(mode)) {
        Tensor routed(d_out, DType::BF16, {H, kTopK * columns});
        Tensor shared(d_out + static_cast<std::size_t>(H) * kTopK * columns, DType::BF16, {H, columns});
        std::uint16_t* d_y = nullptr;
        cuda_check(cudaMalloc(&d_y, sizeof(std::uint16_t) * H * columns), "cudaMalloc");
        Tensor y(d_y, DType::BF16, {H, columns});
        infernix::ops::moe_combine(routed, routing, shared, y, nullptr);
        cuda_check(cudaDeviceSynchronize(), "moe_combine");
        std::vector<std::uint16_t> got(static_cast<std::size_t>(H) * columns);
        cuda_check(cudaMemcpy(got.data(), d_y, got.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
        // The combine's oracle (ops/offloaded_sparse_moe.h): sum_slot w * y + s * shared in FP64. The FP32 sum of
        // eleven products is within 11 * 2^-24 of their magnitudes' sum, so its BF16 output must lie between the BF16
        // roundings of the FP64 value minus and plus that bound.
        const auto value = [](std::uint16_t bits) {
            const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
            float f;
            std::memcpy(&f, &u, 4);
            return static_cast<double>(f);
        };
        std::size_t bad = 0;
        for (int t = 0; t < columns; ++t) {
            for (int row = 0; row < H; ++row) {
                double sum = 0.0, magnitude = 0.0;
                for (int k = 0; k <= kTopK; ++k) {
                    const std::size_t entry = k < kTopK ? static_cast<std::size_t>(t) * kTopK + k
                                                        : static_cast<std::size_t>(kTopK) * columns + t;
                    const double w    = k < kTopK ? weights[entry] : gate[static_cast<std::size_t>(t)];
                    const double term = w * value(last[entry * H + row]);
                    sum += term;
                    magnitude += std::fabs(term);
                }
                const double tolerance = 11.0 * std::ldexp(magnitude, -24);
                const double lo   = value(infernix::ops::canon::f32_to_bf16_rn(static_cast<float>(sum - tolerance)));
                const double hi   = value(infernix::ops::canon::f32_to_bf16_rn(static_cast<float>(sum + tolerance)));
                const double have = value(got[static_cast<std::size_t>(t) * H + row]);
                bad += !(have >= lo && have <= hi);
            }
        }
        check(bad == 0, "T " + std::to_string(columns) + ", " + mode_name(mode) + ": the combine of the routed and shared "
                            "job's outputs within its FP32 error of the FP64 formula (" + std::to_string(bad) + " outside)");
        cudaFree(d_y);
    }

    service.reset(); // no call is pending
    for (auto& e : fork_events) { cudaEventDestroy(e); }
    for (auto& e : split_events) { cudaEventDestroy(e); }
    cudaStreamDestroy(fork_stream);
    cudaStreamDestroy(split_stream);
    cudaFree(d_out);
    cudaFree(d_workspace);
    cudaFree(d_staging);
    cudaFree(d_x);
    cudaFree(d_dispatch);
    cudaFree(d_ids);
    cudaFree(d_weights);
    cudaFree(d_gate);
    cudaFree(d_scales);
    cudaFree(d_frames);
    cudaFree(d_frame_base);
    cudaFree(d_shared_record);
    cudaFreeHost(host);
}

} // namespace

INFERNIX_OP_TEST(offloaded_moe_shared_a4) {
    std::mt19937 rng(0x5EA4ED);
    // Decode (1), verification (2, 4, 8: the shared job narrow), a wide shared job (9, 16, 64, 65), a prefill chunk (300:
    // the 8-bit route's int8 tiles beyond 128 columns).
    for (const int columns : {1, 2, 4, 8, 9, 16, 64, 65, 300}) {
        for (const Mode mode : {Mode::kA4, Mode::kA16, Mode::kA4Residual, Mode::kA8}) { run(columns, mode, rng); }
    }
    // Distinct up input scales: every W4A4 job on the one-column narrow kernels, at any width.
    for (const int columns : {1, 8, 9}) { run(columns, Mode::kA4SplitUp, rng); }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("offloaded_moe shared job at Qwen3.8-Flash-Next's geometry (W4A4, W4A16, A4-residual, 8-bit): all checks "
                "passed\n");
    return 0;
}
