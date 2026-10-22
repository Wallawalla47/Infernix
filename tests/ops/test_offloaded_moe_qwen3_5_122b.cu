// offloaded_sparse_moe at Qwen3.5-122B-A10B's expert geometry (H 3072, I 1024; NVFP4 W4A16 banks,
// docs/maintainer/qwen3_5-122b-offload-design.md §8.5-8.6): the layer route (moe_route with BF16
// router logits, moe_dispatch with the shared expert as the last job, moe_experts, moe_combine) with
// records in device frames, read zero-copy from the pinned host bank, staged through 3 or 64 device
// slots (serial passes) or 16 (double-buffered on an overlap stream), on the forked route, and served
// by the host expert engine through the CPU miss channel. Calls of at most 64 columns run their narrow
// experts on the narrow ring (each of its column variants), wider ones on the one-pass kernels.
//
// Oracles: the routing against its FP64 formula over the logits rounded once to BF16 (lower ids win
// ties), with near-ties that FP32 and BF16 order differently; every expert output (routed and shared)
// against the CPU engine's output (expert_forward_a16 at this geometry, itself checked against FP64 in
// test_offloaded_moe_a16_cpu) of its (column, expert) pair, bit for bit: the narrow and the wide route
// are both exact at this geometry, so every output equals it whatever the record's location, the
// staging pass, the call width or the CPU's share; the combine against its FP32 formula evaluated in
// FP64 within its rounding bound.

#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

constexpr moe::ExpertShape kShape = moe::kQwen3_5_122BShape;
constexpr int H = kShape.hidden;
constexpr int kExperts = 24, kTopK = 8;

float bf16_float(std::uint16_t bits) {
    const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// A logit as the route reads it: rounded once to BF16 (round to nearest even).
double bf16_logit(float v) { return bf16_float(infernix::ops::canon::f32_to_bf16_rn(v)); }

// The routing oracle of one column: top-k of the BF16 logits by repeated arg-max (lower id wins
// ties), the softmax of the selected logits, sigmoid of the shared gate's logit, all in FP64.
struct RouteOracle {
    std::vector<int> ids;
    std::vector<double> weights;
    double gate = 0.0;
};

RouteOracle route_oracle(const float* column) {
    std::vector<double> v(kExperts);
    for (int e = 0; e < kExperts; ++e) { v[e] = bf16_logit(column[e]); }
    RouteOracle out;
    std::vector<bool> taken(kExperts, false);
    for (int k = 0; k < kTopK; ++k) {
        int best = -1;
        for (int e = 0; e < kExperts; ++e) {
            if (!taken[e] && (best < 0 || v[e] > v[best])) { best = e; }
        }
        taken[best] = true;
        out.ids.push_back(best);
    }
    double sum = 0.0;
    for (int id : out.ids) { sum += std::exp(v[id] - v[out.ids[0]]); }
    for (int id : out.ids) { out.weights.push_back(std::exp(v[id] - v[out.ids[0]]) / sum); }
    out.gate = 1.0 / (1.0 + std::exp(-bf16_logit(column[kExperts])));
    return out;
}

// The combine's oracle (ops/offloaded_sparse_moe.h): sum_slot w * y + s * shared in FP64. The route's
// FP32 sum of nine products is within 9 * 2^-24 of their magnitudes' sum, so its BF16 output must lie
// between the BF16 roundings of the FP64 value minus and plus that bound.
bool combine_ok(const std::vector<std::uint16_t>& outputs, const std::vector<float>& weights, float gate,
                int columns, int t, int row, std::uint16_t got) {
    double sum = 0.0, magnitude = 0.0;
    for (int s = 0; s < kTopK; ++s) {
        const std::size_t entry = static_cast<std::size_t>(t) * kTopK + s;
        const double term       = static_cast<double>(weights[entry]) * bf16_float(outputs[entry * H + row]);
        sum += term;
        magnitude += std::fabs(term);
    }
    const std::size_t shared = static_cast<std::size_t>(kTopK) * columns + t;
    const double term        = static_cast<double>(gate) * bf16_float(outputs[shared * H + row]);
    sum += term;
    magnitude += std::fabs(term);
    const double bound = 9.0 * std::ldexp(magnitude, -24);
    const double lo = bf16_float(infernix::ops::canon::f32_to_bf16_rn(static_cast<float>(sum - bound)));
    const double hi = bf16_float(infernix::ops::canon::f32_to_bf16_rn(static_cast<float>(sum + bound)));
    const double value = bf16_float(got);
    return value >= lo && value <= hi;
}

void run(int columns, std::mt19937& rng) {
    const std::uint64_t stride = (kShape.record_bytes() + 4095) / 4096 * 4096;
    const std::uint64_t pitch  = stride + 4096; // the frame pool's pitch exceeds the record stride
    std::uint8_t* host = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host), stride * kExperts, cudaHostAllocMapped), "cudaHostAlloc");
    std::memset(host, 0, stride * kExperts);
    std::vector<fixtures::Expert> bank;
    std::vector<moe::ExpertScales> scales;
    for (int e = 0; e < kExperts; ++e) {
        bank.push_back(fixtures::random_a16_expert(rng, kShape));
        std::memcpy(host + e * stride, bank.back().record.data(), kShape.record_bytes());
        scales.push_back(bank.back().scales);
    }
    // The shared expert: its record in device memory, its multipliers after the routed experts'.
    const fixtures::Expert shared_expert = fixtures::random_a16_expert(rng, kShape);
    auto* d_shared_record = device_copy(shared_expert.record);
    std::vector<moe::ExpertScales> all_scales = scales;
    all_scales.push_back(shared_expert.scales);
    std::uint8_t* host_device = nullptr;
    cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_device), host, 0), "cudaHostGetDevicePointer");
    // A third of the experts resident.
    std::vector<std::int32_t> frames(kExperts, -1);
    int resident = 0;
    for (int e = 1; e < kExperts; e += 3) { frames[static_cast<std::size_t>(e)] = resident++; }
    std::uint8_t* d_frame_base = nullptr;
    cuda_check(cudaMalloc(&d_frame_base, pitch * resident), "cudaMalloc");
    for (int e = 0; e < kExperts; ++e) {
        if (frames[static_cast<std::size_t>(e)] >= 0) {
            cuda_check(cudaMemcpy(d_frame_base + frames[static_cast<std::size_t>(e)] * pitch, host + e * stride, stride,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
        }
    }
    auto* d_frames = device_copy(frames);
    auto* d_scales = device_copy(all_scales);

    // Logits with near-ties: in every column an expert pair one FP32 step apart, the higher id larger,
    // which BF16 makes equal: FP32 logits would rank the higher id first, BF16 ones the lower.
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> logits(static_cast<std::size_t>(kExperts + 1) * columns);
    for (auto& v : logits) { v = n(rng); }
    for (int t = 0; t < columns; ++t) {
        float* column    = logits.data() + static_cast<std::size_t>(t) * (kExperts + 1);
        const float base = bf16_float(infernix::ops::canon::f32_to_bf16_rn(2.5F + 0.01F * static_cast<float>(t % 5)));
        const int lo     = (5 * t) % (kExperts - 1);
        column[lo]       = base;
        column[lo + 1]   = std::nextafter(base, 10.0F);
    }
    auto* d_logits = device_copy(logits);
    std::int32_t* d_ids = nullptr;
    float *d_weights = nullptr, *d_gate = nullptr;
    cuda_check(cudaMalloc(&d_ids, sizeof(std::int32_t) * kTopK * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_weights, sizeof(float) * kTopK * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_gate, sizeof(float) * columns), "cudaMalloc");
    infernix::ops::MoeRouting routing{Tensor(d_ids, DType::I32, {kTopK, columns}),
                                      Tensor(d_weights, DType::FP32, {kTopK, columns}),
                                      Tensor(d_gate, DType::FP32, {columns})};
    const Tensor logits_tensor(d_logits, DType::FP32, {kExperts + 1, columns});
    infernix::ops::moe_route(logits_tensor, kTopK, routing, nullptr, infernix::ops::RouterLogits::kBf16);
    // The dispatch holds the shared job: E + 1 experts, k*T + T entries.
    const int entries = (kTopK + 1) * columns;
    void* d_dispatch  = nullptr;
    cuda_check(cudaMalloc(&d_dispatch, infernix::ops::moe_dispatch_bytes(kExperts + 1, entries)), "cudaMalloc");
    auto dispatch = infernix::ops::carve_moe_dispatch(d_dispatch, kExperts + 1, entries);

    const auto x = fixtures::wide_range_activations(rng, columns, H);
    auto* d_x    = device_copy(x);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(kTopK) * columns);
    std::vector<float> weights(ids.size()), gate(static_cast<std::size_t>(columns));
    cuda_check(cudaDeviceSynchronize(), "moe_route");
    cuda_check(cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
    cuda_check(cudaMemcpy(weights.data(), d_weights, weights.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    cuda_check(cudaMemcpy(gate.data(), d_gate, gate.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
    {
        std::size_t wrong_ids = 0, wrong_values = 0;
        for (int t = 0; t < columns; ++t) {
            const RouteOracle o = route_oracle(logits.data() + static_cast<std::size_t>(t) * (kExperts + 1));
            for (int k = 0; k < kTopK; ++k) {
                const std::size_t i = static_cast<std::size_t>(t) * kTopK + k;
                wrong_ids += ids[i] != o.ids[static_cast<std::size_t>(k)];
                wrong_values += std::fabs(weights[i] - o.weights[static_cast<std::size_t>(k)]) >
                                1e-6 * o.weights[static_cast<std::size_t>(k)] + 1e-30;
            }
            wrong_values += std::fabs(gate[static_cast<std::size_t>(t)] - o.gate) > 1e-6 * o.gate;
        }
        check(wrong_ids == 0 && wrong_values == 0,
              "T " + std::to_string(columns) + ": routing at BF16 logits matches its FP64 formula (" +
                  std::to_string(wrong_ids) + " ids, " + std::to_string(wrong_values) + " values differ)");
    }

    // Expected outputs: column t * k + slot = expert ids[slot, t] applied to x[:, t], then column
    // k * T + t = the shared expert applied to x[:, t] (threads over pairs).
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(H) * entries);
    {
        std::atomic<int> next{0};
        const auto work = [&] {
            for (int i = next++; i < entries; i = next++) {
                const bool shared         = i >= kTopK * columns;
                const int t               = shared ? i - kTopK * columns : i / kTopK;
                const std::uint16_t* xp[1] = {&x[static_cast<std::size_t>(t) * H]};
                std::uint16_t* yp[1]       = {&expected[static_cast<std::size_t>(i) * H]};
                const fixtures::Expert& e =
                    shared ? shared_expert : bank[static_cast<std::size_t>(ids[static_cast<std::size_t>(i)])];
                moe::expert_forward_a16(moe::best_cpu_isa(), kShape, e.record.data(), e.scales, 1, xp, yp);
            }
        };
        std::vector<std::thread> threads;
        for (unsigned w = 0; w < std::min(16U, std::max(1U, std::thread::hardware_concurrency())); ++w) {
            threads.emplace_back(work);
        }
        for (auto& th : threads) { th.join(); }
    }
    int most = 0; // the most columns any expert holds: above eight, the wide route serves it
    for (int e = 0; e < kExperts; ++e) { most = std::max<int>(most, static_cast<int>(std::count(ids.begin(), ids.end(), e))); }

    const int max_jobs = std::min(kExperts, kTopK * columns) + 1; // and the shared job
    std::uint8_t* d_staging = nullptr;
    cuda_check(cudaMalloc(&d_staging, pitch * 64), "cudaMalloc");
    void* d_workspace = nullptr;
    cuda_check(cudaMalloc(&d_workspace, infernix::ops::moe_experts_workspace_bytes(kShape, max_jobs, entries, columns,
                                                                                  moe::ExpertActivation::kA16)),
               "cudaMalloc");
    std::uint16_t* d_out = nullptr;
    cuda_check(cudaMalloc(&d_out, expected.size() * sizeof(std::uint16_t)), "cudaMalloc");
    cudaStream_t fork_stream = nullptr, overlap_stream = nullptr;
    cudaEvent_t fork_events[2]{}, overlap_events[5]{};
    cuda_check(cudaStreamCreateWithFlags(&fork_stream, cudaStreamNonBlocking), "stream");
    cuda_check(cudaStreamCreateWithFlags(&overlap_stream, cudaStreamNonBlocking), "stream");
    for (auto& e : fork_events) { cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event"); }
    for (auto& e : overlap_events) { cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event"); }
    const std::vector<moe::CpuMissService::Layer> layers{{.records = host, .record_stride = stride, .scales = scales.data()}};
    moe::CpuMissService service(layers, {.workers = 6, .max_jobs = 8, .max_columns = 64, .pcie_divisor = 2, .cpus = {},
                                         .activation = moe::ExpertActivation::kA16, .shape = kShape});

    struct Config {
        int slots;
        bool fork;
        bool cpu;
        const char* name;
        bool overlap = false;
    };
    const Config configs[] = {{0, false, false, "zero-copy"},
                              {3, false, false, "3 staging slots"},
                              {64, false, false, "64 staging slots"},
                              {16, false, false, "16 staging slots, double-buffered", true},
                              {64, true, false, "forked"},
                              {64, false, true, "CPU-served misses"},
                              {64, true, true, "forked, CPU-served misses"}};
    for (const auto& config : configs) {
        if (config.fork && columns > moe::kMaxColumns) { continue; }
        if (config.cpu && columns > 64) { continue; }
        infernix::ops::moe_dispatch(routing, kExperts, dispatch, nullptr, nullptr, /*shared_job=*/true);
        cuda_check(cudaMemset(d_out, 0xFF, expected.size() * sizeof(std::uint16_t)), "cudaMemset");
        infernix::ops::MoeExpertSource source{.frame_base    = d_frame_base,
                                              .frames        = d_frames,
                                              .host_records  = host_device,
                                              .record_stride = stride,
                                              .scales        = d_scales,
                                              .staging_base  = config.slots > 0 ? d_staging : nullptr,
                                              .staging_slots = config.slots};
        source.activation    = moe::ExpertActivation::kA16;
        source.frame_stride  = pitch;
        source.shape         = kShape;
        source.shared_expert = kExperts;
        source.shared_record = d_shared_record;
        if (config.fork) {
            source.fork_stream    = fork_stream;
            source.fork_events[0] = fork_events[0];
            source.fork_events[1] = fork_events[1];
        }
        if (config.overlap) {
            source.overlap_stream = overlap_stream;
            for (int i = 0; i < 5; ++i) { source.overlap_events[i] = overlap_events[i]; }
        }
        if (config.cpu) { source.cpu = service.channel(0); }
        Tensor tx(d_x, DType::BF16, {H, columns});
        Tensor out(d_out, DType::BF16, {H, entries});
        bool planned = false;
        if (config.cpu) {
            planned = infernix::ops::moe_route_dispatch_plan(logits_tensor, kTopK, routing, dispatch, nullptr, tx, source,
                                                             max_jobs, d_workspace, out, nullptr,
                                                             infernix::ops::RouterLogits::kBf16, /*shared_job=*/true);
        }
        const std::uint64_t served_before = service.served_experts();
        infernix::ops::moe_experts(tx, dispatch, source, kTopK, max_jobs, d_workspace, out, nullptr, true, planned);
        cuda_check(cudaDeviceSynchronize(), "moe_experts");
        std::vector<std::uint16_t> got(expected.size());
        cuda_check(cudaMemcpy(got.data(), d_out, got.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
        std::size_t bad = 0, bad_shared = 0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const bool differs = got[i] != expected[i];
            bad += differs;
            bad_shared += differs && i >= static_cast<std::size_t>(H) * kTopK * columns;
        }
        const std::string where = "T " + std::to_string(columns) + " (most columns per expert " +
                                  std::to_string(most) + "), " + config.name;
        std::printf("%s: %zu of %zu outputs differ from the CPU engine (%zu of the shared expert's)", where.c_str(), bad,
                    got.size(), bad_shared);
        if (config.cpu) {
            std::printf(" (CPU served %llu experts)",
                        static_cast<unsigned long long>(service.served_experts() - served_before));
        }
        std::printf("\n");
        check(bad == 0, where + ": every routed and shared output equals the CPU engine's bits");
    }

    // A verification round's column mask (MoeColumnMask): rows of `width` columns, row r live in its
    // first valid[r] columns. A masked column routes nowhere (no dispatch entry, routed or shared, so
    // no output is written) and combines to 0; live columns keep every bit;
    // ids and the route log stay the router's for every column. Covers the fused route-dispatch-plan
    // and route-dispatch kernels (T <= 32), the one-CTA dispatch (64) and the three-kernel one (160).
    const int width = columns <= 8 ? columns : 8;
    if (columns > 1 && columns % width == 0) {
        const int rows = columns / width;
        std::vector<std::int32_t> valid(static_cast<std::size_t>(rows));
        for (int r = 0; r < rows; ++r) {
            valid[static_cast<std::size_t>(r)] = rows == 1 ? std::max(1, width / 2) : 1 + (3 * r) % width;
        }
        const auto live = [&](int t) { return t % width < valid[static_cast<std::size_t>(t / width)]; };
        int live_columns = 0;
        for (int t = 0; t < columns; ++t) { live_columns += live(t) ? 1 : 0; }
        auto* d_valid = device_copy(valid);
        std::int32_t* d_log = nullptr;
        cuda_check(cudaMalloc(&d_log, sizeof(std::int32_t) * kTopK * columns), "cudaMalloc");
        infernix::ops::MoeRouting masked = routing;
        masked.mask                      = {d_valid, width};
        const auto entry_live = [&](int i) { return live(i < kTopK * columns ? i / kTopK : i - kTopK * columns); };
        for (const bool cpu : {false, true}) {
            if (cpu && columns > 64) { continue; }
            cuda_check(cudaMemset(d_out, 0xFF, expected.size() * sizeof(std::uint16_t)), "cudaMemset");
            cuda_check(cudaMemset(d_log, 0xFF, sizeof(std::int32_t) * kTopK * columns), "cudaMemset");
            infernix::ops::MoeExpertSource source{.frame_base    = d_frame_base,
                                                  .frames        = d_frames,
                                                  .host_records  = host_device,
                                                  .record_stride = stride,
                                                  .scales        = d_scales,
                                                  .staging_base  = d_staging,
                                                  .staging_slots = 64};
            source.activation    = moe::ExpertActivation::kA16;
            source.frame_stride  = pitch;
            source.shape         = kShape;
            source.shared_expert = kExperts;
            source.shared_record = d_shared_record;
            if (cpu) { source.cpu = service.channel(0); }
            const Tensor tx(d_x, DType::BF16, {H, columns});
            Tensor out(d_out, DType::BF16, {H, entries});
            bool planned = false;
            if (cpu) {
                planned = infernix::ops::moe_route_dispatch_plan(logits_tensor, kTopK, masked, dispatch, d_log, tx, source,
                                                                 max_jobs, d_workspace, out, nullptr,
                                                                 infernix::ops::RouterLogits::kBf16, true);
            } else {
                infernix::ops::moe_route_dispatch(logits_tensor, kTopK, masked, dispatch, d_log, nullptr,
                                                  infernix::ops::RouterLogits::kBf16, true);
            }
            infernix::ops::moe_experts(tx, dispatch, source, kTopK, max_jobs, d_workspace, out, nullptr, true, planned);
            cuda_check(cudaDeviceSynchronize(), "moe_experts (masked)");
            std::vector<std::uint16_t> got(expected.size());
            std::vector<std::int32_t> log(static_cast<std::size_t>(kTopK) * columns), offsets(kExperts + 2);
            std::vector<std::int32_t> listed(static_cast<std::size_t>(entries));
            cuda_check(cudaMemcpy(got.data(), d_out, got.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
            cuda_check(cudaMemcpy(log.data(), d_log, log.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
            cuda_check(cudaMemcpy(offsets.data(), dispatch.offsets, offsets.size() * sizeof(std::int32_t),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
            const int total = offsets[kExperts + 1];
            if (total > 0 && total <= entries) {
                cuda_check(cudaMemcpy(listed.data(), dispatch.entries, static_cast<std::size_t>(total) * sizeof(std::int32_t),
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
            }
            std::size_t bad_live = 0, written_masked = 0;
            for (int i = 0; i < entries; ++i) {
                for (int r = 0; r < H; ++r) {
                    const std::size_t at = static_cast<std::size_t>(i) * H + r;
                    if (entry_live(i)) {
                        bad_live += got[at] != expected[at];
                    } else {
                        written_masked += got[at] != 0xFFFF;
                    }
                }
            }
            std::vector<int> want, have(listed.begin(), listed.begin() + std::clamp(total, 0, entries));
            for (int i = 0; i < entries; ++i) {
                if (entry_live(i)) { want.push_back(i); }
            }
            std::sort(have.begin(), have.end());
            const std::string where = "T " + std::to_string(columns) + " in rows of " + std::to_string(width) + ", " +
                                      std::to_string(live_columns) + " live, " + (cpu ? "CPU-served" : "no CPU");
            check(bad_live == 0 && written_masked == 0,
                  where + ": live outputs keep their bits and masked entries are never computed (" +
                      std::to_string(bad_live) + " live wrong, " + std::to_string(written_masked) + " masked written)");
            check(total == live_columns * (kTopK + 1) && have == want,
                  where + ": the dispatch lists exactly the live columns' routed and shared entries");
            check(log == ids, where + ": the route log keeps the router's ids for every column");
        }

        // The combine: live columns equal the unmasked combine's bits, masked columns are 0.
        std::uint16_t *d_plain = nullptr, *d_masked = nullptr;
        cuda_check(cudaMalloc(&d_plain, sizeof(std::uint16_t) * H * columns), "cudaMalloc");
        cuda_check(cudaMalloc(&d_masked, sizeof(std::uint16_t) * H * columns), "cudaMalloc");
        cuda_check(cudaMemcpy(d_out, expected.data(), expected.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
                   "cudaMemcpy");
        const Tensor all(d_out, DType::BF16, {H, entries});
        Tensor y_plain(d_plain, DType::BF16, {H, columns}), y_masked(d_masked, DType::BF16, {H, columns});
        infernix::ops::moe_combine(all.slice(1, 0, kTopK * columns), routing, all.slice(1, kTopK * columns, columns),
                                   y_plain, nullptr);
        infernix::ops::moe_combine(all.slice(1, 0, kTopK * columns), masked, all.slice(1, kTopK * columns, columns),
                                   y_masked, nullptr);
        cuda_check(cudaDeviceSynchronize(), "moe_combine (masked)");
        std::vector<std::uint16_t> plain(static_cast<std::size_t>(H) * columns), got_masked(plain.size());
        cuda_check(cudaMemcpy(plain.data(), d_plain, plain.size() * 2, cudaMemcpyDeviceToHost), "cudaMemcpy");
        cuda_check(cudaMemcpy(got_masked.data(), d_masked, got_masked.size() * 2, cudaMemcpyDeviceToHost), "cudaMemcpy");
        std::size_t combine_wrong = 0;
        for (int t = 0; t < columns; ++t) {
            for (int r = 0; r < H; ++r) {
                const std::size_t at = static_cast<std::size_t>(t) * H + r;
                combine_wrong += got_masked[at] != (live(t) ? plain[at] : std::uint16_t{0});
            }
        }
        check(combine_wrong == 0, "T " + std::to_string(columns) + ": the masked combine keeps live columns' bits and "
                                                                    "writes 0 for masked ones");

        cudaFree(d_plain);
        cudaFree(d_masked);
        cudaFree(d_log);
        cudaFree(d_valid);
    }

    // The combine at this H: the routed outputs and the shared job's, against its formula.
    std::uint16_t* d_y = nullptr;
    cuda_check(cudaMalloc(&d_y, sizeof(std::uint16_t) * H * columns), "cudaMalloc");
    cuda_check(cudaMemcpy(d_out, expected.data(), expected.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
               "cudaMemcpy");
    const Tensor all(d_out, DType::BF16, {H, entries});
    Tensor y(d_y, DType::BF16, {H, columns});
    infernix::ops::moe_combine(all.slice(1, 0, kTopK * columns), routing, all.slice(1, kTopK * columns, columns), y,
                               nullptr);
    cuda_check(cudaDeviceSynchronize(), "moe_combine");
    std::vector<std::uint16_t> got_y(static_cast<std::size_t>(H) * columns);
    cuda_check(cudaMemcpy(got_y.data(), d_y, got_y.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
    std::size_t combine_bad = 0;
    for (int t = 0; t < columns; ++t) {
        for (int r = 0; r < H; ++r) {
            combine_bad += !combine_ok(expected, weights, gate[static_cast<std::size_t>(t)], columns, t, r,
                                       got_y[static_cast<std::size_t>(t) * H + r]);
        }
    }
    check(combine_bad == 0, "T " + std::to_string(columns) + ": combine within its FP32 error of the FP64 formula (" +
                                std::to_string(combine_bad) + " outside)");

    cudaFree(d_y);
    cudaFree(d_out);
    cudaFree(d_workspace);
    cudaFree(d_staging);
    cudaFree(d_x);
    cudaFree(d_dispatch);
    cudaFree(d_ids);
    cudaFree(d_weights);
    cudaFree(d_gate);
    cudaFree(d_logits);
    cudaFree(d_scales);
    cudaFree(d_frames);
    cudaFree(d_frame_base);
    cudaFree(d_shared_record);
    cudaFreeHost(host);
    cudaStreamDestroy(fork_stream);
    cudaStreamDestroy(overlap_stream);
    for (auto e : fork_events) { cudaEventDestroy(e); }
    for (auto e : overlap_events) { cudaEventDestroy(e); }
}

} // namespace

int main() {
    std::mt19937 rng(20261022);
    // Decode (1), MTP and verification widths (2, 3, 5, 8: the narrow ring's 1-, 2-, 4- and 8-column
    // variants), mixed calls (24 and 64, the ring's widest: some experts above eight columns, others
    // below; the shared job wide), 65 (narrow experts on the one-pass kernels) and a prefill chunk (160:
    // every expert on the wide route; the large dispatch path).
    for (const int columns : {1, 2, 3, 5, 8, 24, 64, 65, 160}) { run(columns, rng); }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("offloaded_moe at Qwen3.5-122B-A10B's geometry: all checks passed\n");
    return 0;
}
