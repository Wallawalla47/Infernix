// The expert-a4-residual fast paths (WideA16::kA4Residual: two NVFP4 terms per x and h row;
// kA4Residual3: three) at Qwen3.5-122B-A10B's expert geometry (H 3072, I 1024; NVFP4 W4A16 banks with
// the shared expert as the call's last job, docs/maintainer/qwen3_5-122b-offload-design.md §8.6), through
// the layer route (moe_route at BF16 logits, moe_dispatch with the shared job, moe_experts) at 24 (a
// narrow-ring call with wide experts beside it), 160, 1,024, 4,096 and 16,384 columns over 32 experts of
// skewed popularity (top 8), so wide experts fill one or many 64-column tiles, partly or wholly.
//
// Oracle: the expert's mathematics in FP64, the stored weights decoded independently (doubled E2M1 code
// times the scaled E4M3 block scale times 2^-10 times the expert's multiplier) against the BF16 x as
// given, with the canonical BF16 boundaries (y_gate, y_up, h) taken from the oracle's own values. On a
// sample of each call's wide entries (routed and shared) the criteria, fixed before the first run, are,
// per stage, relative L2 error per entry:
//   G  gate/up: the route's BF16 h (the entry's h row of the call's workspace, wide_expert.h) against
//      h = silu(bf16(g)) bf16(u) in FP64;
//   D  down: the route's output against the FP64 down product of that same h (its BF16 values);
//   E  end to end: the route's output against the FP64 chain (h rounded to BF16 at its boundary);
// two terms: every sampled entry <= 2^-5 and the mean <= 2^-7 plus the rounding control's mean; three
// terms: <= 2^-6 and <= 2^-9 plus the control's mean (the bounds of the Flash-Next geometry's X1 in
// test_offloaded_moe_wide, whose measured means were 0.31-0.41 % and 0.09-0.10 %). The rounding control
// is what the exact route's BF16 boundary alone leaves against the same oracle: bf16(h) for G, bf16(y)
// for D, the CPU engine's exact output for E. One 4-bit term misses by several per cent; a layout,
// scale or token-mapping error (a shared entry reading another token's x or row scale) by ~100 %.
//   N  narrow entries (at most eight columns, GPU narrow ring or kernels): the CPU engine's bits.
//   R  repeat determinism and placement: the outputs of a rerun, and of every record staged through
//      64 device slots instead of read from frames, are bit-identical to the first run's (the route
//      depends on the data and each expert's column count only).
//   F  a column holding an Inf (T 160): its routed and shared outputs are the canonical NaN.

#include "ops/op_family.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/common/canonical_math.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"
#include "ops/offloaded_sparse_moe/cuda/wide_expert.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace canon    = infernix::ops::canon;
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

template <class T> std::vector<T> host_copy(const T* p, std::size_t n) {
    std::vector<T> v(n);
    cuda_check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");
    return v;
}

constexpr moe::ExpertShape kShape = moe::kQwen3_5_122BShape;
constexpr int H = kShape.hidden, I = kShape.intermediate;
constexpr int kExperts = 32, kTopK = 8;
constexpr int kWideSamples = 192, kSharedSamples = 32, kNarrowSamples = 256;

double bf16_value(std::uint16_t v) { return static_cast<double>(canon::bf16_to_f32(v)); }
double bf16_round(double v) { return bf16_value(canon::f32_to_bf16_rn(static_cast<float>(v))); }

// The stored weight of row `row` at k as an integer (doubled code times scaled block scale); its value is
// this times 2^-10 times the expert's multiplier.
float weight_int(const std::uint8_t* matrix, int blocks, int row, int k) {
    const int rg = row / 16, r = row % 16, b = k / 16, kk = k % 16;
    const std::uint8_t* unit = matrix + (static_cast<std::size_t>(rg) * blocks + b) * moe::kUnitBytes;
    const std::uint8_t byte  = unit[32 * (kk / 4) + 4 * (r % 8) + kk % 4];
    const unsigned code      = r < 8 ? (byte & 15U) : (byte >> 4);
    return static_cast<float>(canon::e2m1_x2(code) * canon::e4m3_scaled(unit[128 + r]));
}

// One expert's matrices decoded once (integers, exact in binary32).
struct Decoded {
    std::vector<float> gate_up; // [2I][H], rows interleaved gate/up
    std::vector<float> down;    // [H][I]
};

Decoded decode(const fixtures::Expert& e) {
    Decoded d;
    d.gate_up.resize(static_cast<std::size_t>(2) * I * H);
    d.down.resize(static_cast<std::size_t>(H) * I);
    const std::uint8_t* gu = e.record.data();
    const std::uint8_t* dn = e.record.data() + kShape.gate_up_bytes();
    for (int row = 0; row < 2 * I; ++row) {
        for (int k = 0; k < H; ++k) { d.gate_up[static_cast<std::size_t>(row) * H + k] = weight_int(gu, kShape.gate_up_blocks(), row, k); }
    }
    for (int row = 0; row < H; ++row) {
        for (int k = 0; k < I; ++k) { d.down[static_cast<std::size_t>(row) * I + k] = weight_int(dn, kShape.down_blocks(), row, k); }
    }
    return d;
}

// FP64 h = silu(bf16(g)) bf16(u) of one column.
std::vector<double> fp64_h(const Decoded& d, const moe::ExpertScales& s, const std::uint16_t* x) {
    std::vector<double> xv(H), h(I);
    for (int k = 0; k < H; ++k) { xv[static_cast<std::size_t>(k)] = bf16_value(x[k]); }
    for (int i = 0; i < I; ++i) {
        const float* wg = &d.gate_up[static_cast<std::size_t>(2 * i) * H];
        const float* wu = wg + H;
        double g = 0.0, u = 0.0;
        for (int k = 0; k < H; ++k) {
            g += static_cast<double>(wg[k]) * xv[static_cast<std::size_t>(k)];
            u += static_cast<double>(wu[k]) * xv[static_cast<std::size_t>(k)];
        }
        const double gb = bf16_round(g * std::ldexp(1.0, -10) * s.alpha_gate);
        const double ub = bf16_round(u * std::ldexp(1.0, -10) * s.alpha_up);
        h[static_cast<std::size_t>(i)] = gb / (1.0 + std::exp(-gb)) * ub;
    }
    return h;
}

// FP64 y of the down matrix over h.
std::vector<double> fp64_down(const Decoded& d, const moe::ExpertScales& s, const std::vector<double>& h) {
    std::vector<double> y(H);
    for (int row = 0; row < H; ++row) {
        const float* w = &d.down[static_cast<std::size_t>(row) * I];
        double acc = 0.0;
        for (int k = 0; k < I; ++k) { acc += static_cast<double>(w[k]) * h[static_cast<std::size_t>(k)]; }
        y[static_cast<std::size_t>(row)] = acc * std::ldexp(1.0, -10) * s.alpha_down;
    }
    return y;
}

template <class Got>
double relative_l2(const Got& got, const std::vector<double>& want) {
    double diff = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double g = got(i);
        diff += (g - want[i]) * (g - want[i]);
        norm += want[i] * want[i];
    }
    return norm > 0.0 ? std::sqrt(diff / norm) : std::sqrt(diff);
}

std::uint64_t fnv1a(const std::vector<std::uint16_t>& v) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const std::uint16_t x : v) { h = (h ^ x) * 1099511628211ULL; }
    return h;
}

// Routes `columns` columns to top-8 distinct experts by popularity 1 / (rank + 1)^0.7: logits that make
// exactly those the top 8, and a shared-gate row.
std::vector<float> routing_logits(std::mt19937& rng, int columns) {
    std::vector<double> weight(kExperts);
    for (int e = 0; e < kExperts; ++e) { weight[static_cast<std::size_t>(e)] = 1.0 / std::pow(e + 1.0, 0.7); }
    std::shuffle(weight.begin(), weight.end(), rng);
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> logits(static_cast<std::size_t>(kExperts + 1) * columns);
    for (int t = 0; t < columns; ++t) {
        float* column = &logits[static_cast<std::size_t>(t) * (kExperts + 1)];
        for (int e = 0; e <= kExperts; ++e) { column[e] = n(rng); }
        std::vector<double> w = weight;
        for (int k = 0; k < kTopK; ++k) {
            std::discrete_distribution<int> pick(w.begin(), w.end());
            const int e                    = pick(rng);
            w[static_cast<std::size_t>(e)] = 0.0;
            column[e]                      = 20.0F + static_cast<float>(kTopK - k);
        }
    }
    return logits;
}

struct Bank {
    std::vector<fixtures::Expert> experts; // the routed experts, then the shared one
    std::uint8_t* d_frames_memory = nullptr;
    std::int32_t* d_frames        = nullptr; // every routed expert in frame e
    std::int32_t* d_no_frames     = nullptr;
    std::uint8_t* host            = nullptr;
    std::uint8_t* host_device     = nullptr;
    std::uint8_t* d_shared_record = nullptr;
    moe::ExpertScales* d_scales   = nullptr;
    std::uint64_t stride          = 0;
};

Bank make_bank(std::mt19937& rng) {
    Bank b;
    b.stride = (kShape.record_bytes() + 4095) / 4096 * 4096;
    for (int e = 0; e <= kExperts; ++e) { b.experts.push_back(fixtures::random_a16_expert(rng, kShape)); }
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&b.host), b.stride * kExperts, cudaHostAllocMapped), "cudaHostAlloc");
    std::memset(b.host, 0, b.stride * kExperts);
    for (int e = 0; e < kExperts; ++e) {
        std::memcpy(b.host + e * b.stride, b.experts[static_cast<std::size_t>(e)].record.data(), kShape.record_bytes());
    }
    cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&b.host_device), b.host, 0), "cudaHostGetDevicePointer");
    cuda_check(cudaMalloc(&b.d_frames_memory, b.stride * kExperts), "cudaMalloc");
    cuda_check(cudaMemcpy(b.d_frames_memory, b.host, b.stride * kExperts, cudaMemcpyHostToDevice), "cudaMemcpy");
    std::vector<std::int32_t> frames(kExperts), none(kExperts, -1);
    for (int e = 0; e < kExperts; ++e) { frames[static_cast<std::size_t>(e)] = e; }
    b.d_frames    = device_copy(frames);
    b.d_no_frames = device_copy(none);
    b.d_shared_record = device_copy(b.experts.back().record);
    std::vector<moe::ExpertScales> scales;
    for (const auto& e : b.experts) { scales.push_back(e.scales); }
    b.d_scales = device_copy(scales);
    return b;
}

void free_bank(Bank& b) {
    cudaFree(b.d_frames_memory);
    cudaFree(b.d_frames);
    cudaFree(b.d_no_frames);
    cudaFree(b.d_shared_record);
    cudaFree(b.d_scales);
    cudaFreeHost(b.host);
}

// The criteria of one stage over its sampled entries.
struct Stage {
    const char* name;
    std::vector<double> error, control;

    void judge(const std::string& where, bool three) const {
        if (error.empty()) { return; }
        double worst = 0.0, mean = 0.0, control_mean = 0.0;
        for (std::size_t i = 0; i < error.size(); ++i) {
            worst = std::max(worst, error[i]);
            mean += error[i];
            control_mean += control[i];
        }
        mean /= static_cast<double>(error.size());
        control_mean /= static_cast<double>(error.size());
        const double worst_bound = std::ldexp(1.0, three ? -6 : -5);
        const double mean_bound  = std::ldexp(1.0, three ? -9 : -7) + control_mean;
        std::printf("   %s %-8s relative L2: mean %.5f (bound %.5f, control %.5f), worst %.5f (bound %.5f), %zu entries\n",
                    where.c_str(), name, mean, mean_bound, control_mean, worst, worst_bound, error.size());
        check(worst <= worst_bound, where + " " + name + ": every sampled wide entry within its bound of the FP64 oracle");
        check(mean <= mean_bound, where + " " + name + ": the sampled wide entries' mean within its bound");
    }
};

void run(Bank& bank, int columns, moe::WideA16 route, std::mt19937& rng) {
    const bool three = route == moe::WideA16::kA4Residual3;
    const std::string where = std::string(three ? "three-term" : "two-term") + " T " + std::to_string(columns);
    std::printf("== %s\n", where.c_str());
    const bool nonfinite = columns == 160;
    const auto logits    = routing_logits(rng, columns);
    auto x               = fixtures::wide_range_activations(rng, columns, H);
    if (nonfinite) { x[5] = 0x7F80U; } // +Inf in column 0
    auto* d_logits = device_copy(logits);
    auto* d_x      = device_copy(x);
    std::int32_t* d_ids = nullptr;
    float *d_weights = nullptr, *d_gate = nullptr;
    cuda_check(cudaMalloc(&d_ids, sizeof(std::int32_t) * kTopK * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_weights, sizeof(float) * kTopK * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_gate, sizeof(float) * columns), "cudaMalloc");
    infernix::ops::MoeRouting routing{Tensor(d_ids, DType::I32, {kTopK, columns}),
                                      Tensor(d_weights, DType::FP32, {kTopK, columns}), Tensor(d_gate, DType::FP32, {columns})};
    infernix::ops::moe_route(Tensor(d_logits, DType::FP32, {kExperts + 1, columns}), kTopK, routing, nullptr,
                             infernix::ops::RouterLogits::kBf16);
    const int entries = (kTopK + 1) * columns;
    void* d_dispatch  = nullptr;
    cuda_check(cudaMalloc(&d_dispatch, infernix::ops::moe_dispatch_bytes(kExperts + 1, entries)), "cudaMalloc");
    auto dispatch = infernix::ops::carve_moe_dispatch(d_dispatch, kExperts + 1, entries);
    infernix::ops::moe_dispatch(routing, kExperts, dispatch, nullptr, nullptr, /*shared_job=*/true);
    cuda_check(cudaDeviceSynchronize(), "dispatch");
    const auto ids       = host_copy(d_ids, static_cast<std::size_t>(kTopK) * columns);
    const auto offsets   = host_copy(dispatch.offsets, static_cast<std::size_t>(kExperts) + 2);
    const auto positions = host_copy(dispatch.entries, static_cast<std::size_t>(entries));
    // Output column -> (expert, token, dispatch position), and each expert's column count.
    std::vector<int> expert_of(static_cast<std::size_t>(entries)), token_of(static_cast<std::size_t>(entries)),
        position_of(static_cast<std::size_t>(entries));
    for (int i = 0; i < entries; ++i) {
        const bool shared = i >= kTopK * columns;
        expert_of[static_cast<std::size_t>(i)] = shared ? kExperts : ids[static_cast<std::size_t>(i)];
        token_of[static_cast<std::size_t>(i)]  = shared ? i - kTopK * columns : i / kTopK;
    }
    for (int p = 0; p < entries; ++p) { position_of[static_cast<std::size_t>(positions[static_cast<std::size_t>(p)])] = p; }
    const auto count = [&](int expert) { return offsets[static_cast<std::size_t>(expert) + 1] - offsets[static_cast<std::size_t>(expert)]; };

    const int max_jobs = std::min(kExperts, kTopK * columns) + 1;
    const std::size_t workspace_bytes =
        infernix::ops::moe_experts_workspace_bytes(kShape, max_jobs, entries, columns, moe::ExpertActivation::kA16, route);
    void* d_workspace = nullptr;
    cuda_check(cudaMalloc(&d_workspace, workspace_bytes), "cudaMalloc");
    const moe::wide::ExpertsWorkspace layout = moe::wide::carve_experts_workspace(
        d_workspace, kShape, max_jobs, entries, columns, moe::ExpertActivation::kA16, route);
    std::uint8_t* d_staging = nullptr;
    cuda_check(cudaMalloc(&d_staging, bank.stride * 64), "cudaMalloc");
    std::uint16_t* d_out = nullptr;
    const std::size_t out_elements = static_cast<std::size_t>(H) * entries;
    cuda_check(cudaMalloc(&d_out, out_elements * sizeof(std::uint16_t)), "cudaMalloc");

    struct Config {
        const char* name;
        bool staged;
    };
    const Config configs[] = {{"frames", false}, {"frames, rerun", false}, {"64 staging slots", true}};
    std::vector<std::uint16_t> first;
    std::vector<std::uint16_t> h_rows; // the first run's BF16 h of every dispatch position
    for (const Config& config : configs) {
        cuda_check(cudaMemset(d_out, 0xFF, out_elements * sizeof(std::uint16_t)), "cudaMemset");
        infernix::ops::MoeExpertSource source{.frame_base    = bank.d_frames_memory,
                                              .frames        = config.staged ? bank.d_no_frames : bank.d_frames,
                                              .host_records  = bank.host_device,
                                              .record_stride = bank.stride,
                                              .scales        = bank.d_scales,
                                              .staging_base  = d_staging,
                                              .staging_slots = 64};
        source.activation    = moe::ExpertActivation::kA16;
        source.wide_a16      = route;
        source.shape         = kShape;
        source.shared_expert = kExperts;
        source.shared_record = bank.d_shared_record;
        Tensor tx(d_x, DType::BF16, {H, columns});
        Tensor out(d_out, DType::BF16, {H, entries});
        infernix::ops::moe_experts(tx, dispatch, source, kTopK, max_jobs, d_workspace, out, nullptr);
        cuda_check(cudaDeviceSynchronize(), "moe_experts");
        auto got = host_copy(d_out, out_elements);
        if (first.empty()) {
            first = std::move(got);
            const std::size_t row_elements = static_cast<std::size_t>(moe::wide::A16Planes<moe::Qwen3_5_122BExperts>::kHRowElements);
            h_rows = host_copy(reinterpret_cast<const std::uint16_t*>(layout.h_rows), row_elements * entries);
            continue;
        }
        long differing = 0;
        for (std::size_t i = 0; i < got.size(); ++i) { differing += got[i] != first[i]; }
        std::printf("   R %-18s: %ld outputs differ from the first run (fnv1a %016llx)\n", config.name, differing,
                    static_cast<unsigned long long>(fnv1a(got)));
        check(differing == 0, where + ": outputs bit-identical on a rerun and with staged records");
    }

    // The samples: wide routed entries spread over the call, wide shared entries, narrow entries.
    std::vector<int> wide_routed, wide_shared, narrow;
    for (int i = 0; i < entries; ++i) {
        const int e = expert_of[static_cast<std::size_t>(i)];
        const bool wide = count(e) > moe::kMaxColumns;
        if (nonfinite && token_of[static_cast<std::size_t>(i)] == 0) { continue; }
        (wide ? (e == kExperts ? wide_shared : wide_routed) : narrow).push_back(i);
    }
    const auto spread = [](const std::vector<int>& all, int n) {
        std::vector<int> out;
        if (all.empty()) { return out; }
        const std::size_t take = std::min<std::size_t>(all.size(), static_cast<std::size_t>(n));
        for (std::size_t s = 0; s < take; ++s) { out.push_back(all[s * all.size() / take]); }
        return out;
    };
    std::vector<int> wide_samples = spread(wide_routed, kWideSamples);
    for (const int i : spread(wide_shared, kSharedSamples)) { wide_samples.push_back(i); }
    const std::vector<int> narrow_samples = spread(narrow, kNarrowSamples);
    std::printf("   %zu wide routed, %zu wide shared and %zu narrow entries; %zu wide and %zu narrow sampled\n",
                wide_routed.size(), wide_shared.size(), narrow.size(), wide_samples.size(), narrow_samples.size());

    // N: narrow entries against the CPU engine, bit for bit.
    long narrow_bad = 0;
    for (const int i : narrow_samples) {
        const auto& e = bank.experts[static_cast<std::size_t>(expert_of[static_cast<std::size_t>(i)])];
        std::vector<std::uint16_t> cpu(H);
        const std::uint16_t* xp[1] = {&x[static_cast<std::size_t>(token_of[static_cast<std::size_t>(i)]) * H]};
        std::uint16_t* yp[1]       = {cpu.data()};
        moe::expert_forward_a16(moe::best_cpu_isa(), kShape, e.record.data(), e.scales, 1, xp, yp);
        narrow_bad += std::memcmp(cpu.data(), &first[static_cast<std::size_t>(i) * H], H * 2) != 0;
    }
    std::printf("   N narrow entries differing from the CPU engine: %ld of %zu\n", narrow_bad, narrow_samples.size());
    check(narrow_bad == 0, where + ": every sampled narrow entry equals the CPU engine bit for bit");

    // G, D, E on the wide samples, grouped by expert (each decoded once), threads over experts.
    Stage g{"gate/up", {}, {}}, d{"down", {}, {}}, e2e{"end2end", {}, {}};
    g.error.assign(wide_samples.size(), 0.0);
    g.control = d.error = d.control = e2e.error = e2e.control = g.error;
    std::vector<std::vector<std::size_t>> by_expert(kExperts + 1);
    for (std::size_t s = 0; s < wide_samples.size(); ++s) {
        by_expert[static_cast<std::size_t>(expert_of[static_cast<std::size_t>(wide_samples[s])])].push_back(s);
    }
    std::atomic<int> next{0};
    const auto work = [&] {
        for (int expert = next++; expert <= kExperts; expert = next++) {
            const auto& samples = by_expert[static_cast<std::size_t>(expert)];
            if (samples.empty()) { continue; }
            const fixtures::Expert& ex = bank.experts[static_cast<std::size_t>(expert)];
            const Decoded dec          = decode(ex); // 38 MB a thread, not kept (host RAM is shared)
            for (const std::size_t s : samples) {
                const int i = wide_samples[s];
                const std::uint16_t* xc = &x[static_cast<std::size_t>(token_of[static_cast<std::size_t>(i)]) * H];
                const std::uint16_t* y  = &first[static_cast<std::size_t>(i) * H];
                const std::uint16_t* hg =
                    &h_rows[static_cast<std::size_t>(position_of[static_cast<std::size_t>(i)]) *
                            moe::wide::A16Planes<moe::Qwen3_5_122BExperts>::kHRowElements];
                const std::vector<double> h = fp64_h(dec, ex.scales, xc);
                g.error[s]   = relative_l2([&](std::size_t k) { return bf16_value(hg[k]); }, h);
                g.control[s] = relative_l2([&](std::size_t k) { return bf16_round(h[k]); }, h);
                std::vector<double> h_route(I), h_bf16(I);
                for (int k = 0; k < I; ++k) {
                    h_route[static_cast<std::size_t>(k)] = bf16_value(hg[k]);
                    h_bf16[static_cast<std::size_t>(k)]  = bf16_round(h[static_cast<std::size_t>(k)]);
                }
                const std::vector<double> y_down = fp64_down(dec, ex.scales, h_route);
                d.error[s]   = relative_l2([&](std::size_t r) { return bf16_value(y[r]); }, y_down);
                d.control[s] = relative_l2([&](std::size_t r) { return bf16_round(y_down[r]); }, y_down);
                const std::vector<double> y_chain = fp64_down(dec, ex.scales, h_bf16);
                std::vector<std::uint16_t> cpu(H);
                const std::uint16_t* xp[1] = {xc};
                std::uint16_t* yp[1]       = {cpu.data()};
                moe::expert_forward_a16(moe::best_cpu_isa(), kShape, ex.record.data(), ex.scales, 1, xp, yp);
                e2e.error[s]   = relative_l2([&](std::size_t r) { return bf16_value(y[r]); }, y_chain);
                e2e.control[s] = relative_l2([&](std::size_t r) { return bf16_value(cpu[r]); }, y_chain);
            }
        }
    };
    std::vector<std::thread> threads;
    for (unsigned w = 0; w < std::min(16U, std::max(1U, std::thread::hardware_concurrency())); ++w) { threads.emplace_back(work); }
    for (auto& t : threads) { t.join(); }
    g.judge(where, three);
    d.judge(where, three);
    e2e.judge(where, three);
    check(!wide_samples.empty() && (columns <= moe::kMaxColumns || !wide_shared.empty()),
          where + ": the call has wide routed and shared entries to check");

    // F: the Inf column's routed and shared outputs are the canonical NaN.
    if (nonfinite) {
        int nan_entries = 0;
        for (int i = 0; i < entries; ++i) {
            if (token_of[static_cast<std::size_t>(i)] == 0) { nan_entries += first[static_cast<std::size_t>(i) * H] == 0x7FC0U; }
        }
        std::printf("   F the Inf column's outputs: %d of %d canonical NaN\n", nan_entries, kTopK + 1);
        check(nan_entries == kTopK + 1, where + ": the non-finite column's outputs are the canonical NaN");
    }

    cudaFree(d_out);
    cudaFree(d_staging);
    cudaFree(d_workspace);
    cudaFree(d_dispatch);
    cudaFree(d_gate);
    cudaFree(d_weights);
    cudaFree(d_ids);
    cudaFree(d_x);
    cudaFree(d_logits);
}

} // namespace

INFERNIX_OP_TEST(offloaded_moe_a4_residual_122b) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (infernix::test::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        std::mt19937 rng(20261023);
        Bank bank = make_bank(rng);
        for (const moe::WideA16 route : {moe::WideA16::kA4Residual, moe::WideA16::kA4Residual3}) {
            for (const int columns : {24, 160, 1024, 4096, 16384}) { run(bank, columns, route, rng); }
        }
        free_bank(bank);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("expert-a4-residual at Qwen3.5-122B-A10B's geometry: all checks passed\n");
    return 0;
}
