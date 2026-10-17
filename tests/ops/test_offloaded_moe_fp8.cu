// Block-FP8 expert banks (fp8_expert_b128_v1) through the layer route of offloaded_sparse_moe
// (moe_route, moe_dispatch, moe_experts; docs/maintainer/qwen3_8-flash-next-design.md §16.2.2, §19.3.25).
//
// Records are random E4M3FN codes (no NaN) with random positive BF16 tile scales spanning several
// binades, in a pinned host bank, some copied into device frames whose pitch exceeds the record
// stride. Misses are read zero-copy, staged through 3 or 64 slots, forked, or computed by the CPU
// route through the miss service; calls of 1, 3, 9, 40 and 300 columns. Oracle: each routed
// (column, expert) pair on the host by the scalar reference of ops/fp8_block/fp8_exact.h (x and h
// A16-encoded, gate/up rounded to BF16, SwiGLU, the down rows); that reference is itself checked
// on sampled rows against an independent bignum sum of decoded values. Every output must be
// bit-equal in every configuration. Calls of 40 and 300 columns run the tensor-core tiles for jobs
// of more than 8 columns.
//
// `--bench [tokens]` times one layer's expert call of a prefill chunk (512 experts, top-10, every
// record resident) instead.

#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/fp8_block/fp8_exact.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
namespace fp8   = infernix::fp8b128;

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

constexpr int H = moe::kHidden, I = moe::kIntermediate, T = fp8::kTile;
constexpr std::size_t kGate   = static_cast<std::size_t>(I) * H;
constexpr int kGateTiles      = (I / T) * (H / T);
constexpr std::uint64_t kRecord = 3 * kGate + 3ULL * kGateTiles * 2;

// A random record: codes without NaN, BF16 scales 2^-14..2^-7 (mantissas random), so a row's tile
// exponents differ by up to 7 binades.
void random_record(std::mt19937& rng, std::uint8_t* r) {
    for (std::size_t i = 0; i < 3 * kGate; ++i) {
        auto c = static_cast<std::uint8_t>(rng());
        if ((c & 0x7F) == 0x7F) { c ^= 0x01; }
        r[i] = c;
    }
    auto* s = reinterpret_cast<std::uint16_t*>(r + 3 * kGate);
    for (int t = 0; t < 3 * kGateTiles; ++t) {
        const int exponent = 127 - 14 + static_cast<int>(rng() % 8);
        s[t]               = static_cast<std::uint16_t>((exponent << 7) | (rng() & 0x7F));
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

// One row's output by the scalar reference.
std::uint16_t row_output(const std::uint8_t* codes, const std::uint16_t* tiles, const std::int32_t* x, int k, int emax) {
    const int e = fp8::min_exponent(tiles, k / T);
    return fp8::output_bf16(fp8::row_dot(codes, tiles, x, k, e), e, emax);
}

// The exact expert on the host.
std::vector<std::uint16_t> expert(const std::uint8_t* record, const std::uint16_t* xv) {
    const auto* s = reinterpret_cast<const std::uint16_t*>(record + 3 * kGate);
    int ex = 0;
    const auto x = encode(xv, H, ex);
    std::vector<std::uint16_t> h(static_cast<std::size_t>(I));
    for (int i = 0; i < I; ++i) {
        const std::uint16_t* tiles = s + (i / T) * (H / T);
        const auto yg = row_output(record + static_cast<std::size_t>(i) * H, tiles, x.data(), H, ex);
        const auto yu = row_output(record + kGate + static_cast<std::size_t>(i) * H, tiles + kGateTiles, x.data(), H, ex);
        h[static_cast<std::size_t>(i)] = canon::swiglu_bf16(yg, yu);
    }
    int eh = 0;
    const auto hx = encode(h.data(), I, eh);
    std::vector<std::uint16_t> y(static_cast<std::size_t>(H));
    for (int r = 0; r < H; ++r) {
        y[static_cast<std::size_t>(r)] = row_output(record + 2 * kGate + static_cast<std::size_t>(r) * I,
                                                    s + 2 * kGateTiles + (r / T) * (I / T), hx.data(), I, eh);
    }
    return y;
}

// ---- the independent check of the scalar reference: a bignum sum of decoded values.

// value * 2^(32 i) per limb; every added term < 2^56.
struct Big {
    std::vector<std::int64_t> limbs = std::vector<std::int64_t>(8, 0);
    void add(std::int64_t term) {
        limbs[0] += term & 0xFFFFFFFF;
        limbs[1] += term >> 32;
        for (std::size_t i = 0; i + 1 < limbs.size(); ++i) {
            const std::int64_t carry = limbs[i] >> 32;
            limbs[i] -= carry << 32;
            limbs[i + 1] += carry;
        }
    }
    // Round-to-nearest-even binary32 of value * 2^k (normal results only), by bits.
    float to_float(int k) const {
        std::vector<std::int64_t> v = limbs;
        const bool negative = v.back() < 0;
        if (negative) {
            Big t;
            for (auto& l : v) { l = -l; }
            t.limbs = v;
            t.add(0);
            v = t.limbs;
        }
        std::vector<int> bits;
        for (const auto l : v) {
            for (int j = 0; j < 32; ++j) { bits.push_back(static_cast<int>((l >> j) & 1)); }
        }
        int msb = static_cast<int>(bits.size()) - 1;
        while (msb >= 0 && bits[static_cast<std::size_t>(msb)] == 0) { --msb; }
        if (msb < 0) { return negative ? -0.0F : 0.0F; }
        const int lsb = std::max(0, msb - 23);
        double mant = 0;
        for (int i = msb; i >= lsb; --i) { mant = mant * 2 + bits[static_cast<std::size_t>(i)]; }
        if (lsb > 0) {
            const bool half = bits[static_cast<std::size_t>(lsb - 1)] != 0;
            bool sticky = false;
            for (int i = 0; i < lsb - 1; ++i) { sticky = sticky || bits[static_cast<std::size_t>(i)] != 0; }
            if (half && (sticky || std::fmod(mant, 2.0) != 0.0)) { mant += 1; }
        }
        const double value = std::ldexp(mant, lsb + k);
        return static_cast<float>(negative ? -value : value);
    }
};

// e4m3fn(c) * 2^9 by floating-point decoding (not fp8_int's bit form).
std::int64_t decoded_int(std::uint8_t c) {
    const int e = (c >> 3) & 15, m = c & 7;
    const double v = e == 0 ? std::ldexp(m, -9) : std::ldexp(1.0 + m / 8.0, e - 7);
    return static_cast<std::int64_t>(std::ldexp(c & 0x80 ? -v : v, 9));
}

// The row's binary32 output, each term w * mant(s) * X * 2^(E(s) - E_lowest) summed as a bignum.
float bignum_row(const std::uint8_t* codes, const std::uint16_t* tiles, const std::int32_t* x, int k, int emax) {
    int lowest = 255;
    for (int t = 0; t < k / T; ++t) { lowest = std::min(lowest, (tiles[t] >> 7) & 0xFF); }
    Big sum;
    for (int j = 0; j < k; ++j) {
        const std::uint16_t s = tiles[j / T];
        const std::int64_t mant = (s & 0x7F) | 0x80;
        sum.add(decoded_int(codes[j]) * mant * x[j] * (std::int64_t{1} << (((s >> 7) & 0xFF) - lowest)));
    }
    return sum.to_float(lowest + emax - 290);
}

void check_reference(std::mt19937& rng) {
    std::vector<std::uint8_t> record(kRecord);
    random_record(rng, record.data());
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<std::uint16_t> xv(H);
    for (auto& v : xv) { v = canon::f32_to_bf16_rn(n(rng)); }
    int emax = 0;
    const auto x = encode(xv.data(), H, emax);
    const auto* s = reinterpret_cast<const std::uint16_t*>(record.data() + 3 * kGate);
    int bad = 0;
    for (int i = 0; i < I; i += 7) {
        const std::uint16_t* tiles = s + (i / T) * (H / T);
        const int e = fp8::min_exponent(tiles, H / T);
        const float ref = fp8::output_f32(fp8::row_dot(record.data() + static_cast<std::size_t>(i) * H, tiles, x.data(), H, e), e, emax);
        const float big = bignum_row(record.data() + static_cast<std::size_t>(i) * H, tiles, x.data(), H, emax);
        bad += std::memcmp(&ref, &big, 4) != 0;
    }
    check(bad == 0, "the scalar FP8 reference differs from the bignum sum of decoded values");
}

void run(int columns, std::mt19937& rng) {
    constexpr int experts = 24, top_k = 4;
    const std::uint64_t stride = (kRecord + 4095) / 4096 * 4096;
    const std::uint64_t pitch  = stride + 8192; // the frame pool's pitch exceeds the record stride
    std::uint8_t* host = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host), stride * experts, cudaHostAllocMapped), "cudaHostAlloc");
    std::memset(host, 0, stride * experts);
    for (int e = 0; e < experts; ++e) { random_record(rng, host + e * stride); }
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
    if (columns > 64) {
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

    std::vector<std::uint16_t> x(static_cast<std::size_t>(H) * columns);
    for (auto& v : x) { v = canon::f32_to_bf16_rn(n(rng)); }
    auto* d_x = device_copy(x);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(top_k) * columns);
    cuda_check(cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(H) * top_k * columns);
    {
        std::atomic<int> next{0};
        const auto work = [&] {
            for (int i = next++; i < columns * top_k; i = next++) {
                const int t = i / top_k;
                const int e = ids[static_cast<std::size_t>(i)];
                const auto y = expert(host + e * stride, &x[static_cast<std::size_t>(t) * H]);
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
    cuda_check(cudaMalloc(&d_workspace, infernix::ops::moe_experts_workspace_bytes(max_jobs, top_k * columns, columns,
                                                                                  moe::ExpertActivation::kA16)),
               "cudaMalloc");
    std::vector<moe::ExpertScales> scales(experts);
    auto* d_scales = device_copy(scales);
    std::uint16_t* d_out = nullptr;
    cuda_check(cudaMalloc(&d_out, expected.size() * sizeof(std::uint16_t)), "cudaMalloc");
    cudaStream_t fork_stream = nullptr;
    cudaEvent_t fork_events[2]{};
    cuda_check(cudaStreamCreateWithFlags(&fork_stream, cudaStreamNonBlocking), "stream");
    for (auto& e : fork_events) { cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event"); }

    // The CPU route (ops/offloaded_sparse_moe/cpu/fp8_expert.h through the ggml team) serving misses.
    std::vector<moe::ExpertScales> no_scales(experts);
    const moe::GgmlExpertFormat format{.gate_up = QType::FP8_E4M3FN_B128, .down = QType::FP8_E4M3FN_B128,
                                       .gate_row_bytes = static_cast<std::uint32_t>(H),
                                       .down_row_bytes = static_cast<std::uint32_t>(I)};
    moe::CpuMissService service({{.records = host, .record_stride = stride, .scales = no_scales.data(), .ggml = format}},
                                {.workers = 4, .max_jobs = 8, .max_columns = 64, .pcie_divisor = 2, .cpus = {},
                                 .activation = moe::ExpertActivation::kA16});

    struct Config {
        int slots;
        bool fork;
        bool cpu;
        const char* name;
    };
    const Config configs[] = {{0, false, false, "zero-copy"},        {3, false, false, "3 staging slots"},
                              {64, false, false, "64 staging slots"}, {64, true, false, "forked"},
                              {64, false, true, "CPU-served misses"}, {64, true, true, "forked, CPU-served misses"}};
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
        source.ggml         = {.gate_up_format = QType::FP8_E4M3FN_B128,
                               .down_format    = QType::FP8_E4M3FN_B128,
                               .gate_row_bytes = static_cast<std::uint32_t>(H),
                               .down_row_bytes = static_cast<std::uint32_t>(I),
                               .record_bytes   = kRecord + (16 - kRecord % 16) % 16};
        if (config.fork) {
            source.fork_stream    = fork_stream;
            source.fork_events[0] = fork_events[0];
            source.fork_events[1] = fork_events[1];
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
            std::snprintf(none, sizeof none, "fp8 experts at %d columns, %s: the CPU served no expert", columns, config.name);
            check(service.served_experts() > served, none);
        }
        std::vector<std::uint16_t> actual(expected.size());
        cuda_check(cudaMemcpy(actual.data(), d_out, actual.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
        std::size_t bad = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) { bad += actual[i] != expected[i]; }
        char what[160];
        std::snprintf(what, sizeof what, "fp8 experts at %d columns, %s: %zu of %zu outputs differ", columns, config.name,
                      bad, actual.size());
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
    for (auto e : fork_events) { cudaEventDestroy(e); }
}

// --bench: one prefill chunk of the real geometry (512 experts, top-10, 4,096 tokens: about 80
// columns per expert) with every record resident (16 random records repeated), timed over
// repeated moe_experts calls.
void bench(int columns, std::mt19937& rng) {
    constexpr int experts = 512, top_k = 10, distinct = 16, reps = 5;
    const std::uint64_t stride = (kRecord + 4095) / 4096 * 4096;
    std::vector<std::uint8_t> records(stride * distinct, 0);
    for (int e = 0; e < distinct; ++e) { random_record(rng, records.data() + e * stride); }
    std::uint8_t* d_frames_base = nullptr;
    cuda_check(cudaMalloc(&d_frames_base, stride * experts), "cudaMalloc");
    for (int e = 0; e < experts; ++e) {
        cuda_check(cudaMemcpy(d_frames_base + e * stride, records.data() + (e % distinct) * stride, stride, cudaMemcpyHostToDevice),
                   "cudaMemcpy");
    }
    std::vector<std::int32_t> frames(experts);
    for (int e = 0; e < experts; ++e) { frames[static_cast<std::size_t>(e)] = e; }
    auto* d_frames = device_copy(frames);
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> logits(static_cast<std::size_t>(experts + 1) * columns);
    for (auto& v : logits) { v = n(rng); }
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
    infernix::ops::moe_dispatch(routing, experts, dispatch, nullptr, nullptr);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(H) * columns);
    for (auto& v : x) { v = canon::f32_to_bf16_rn(n(rng)); }
    auto* d_x = device_copy(x);
    void* d_workspace = nullptr;
    cuda_check(cudaMalloc(&d_workspace, infernix::ops::moe_experts_workspace_bytes(experts, top_k * columns, columns,
                                                                                  moe::ExpertActivation::kA16)),
               "cudaMalloc");
    std::vector<moe::ExpertScales> scales(experts);
    auto* d_scales = device_copy(scales);
    std::uint16_t* d_out = nullptr;
    cuda_check(cudaMalloc(&d_out, sizeof(std::uint16_t) * H * top_k * columns), "cudaMalloc");
    // Every expert is resident, so the host records (required by the source) are never read.
    infernix::ops::MoeExpertSource source{.frame_base = d_frames_base, .frames = d_frames, .host_records = d_frames_base,
                                          .record_stride = stride, .scales = d_scales, .staging_base = nullptr,
                                          .staging_slots = 0};
    source.activation   = moe::ExpertActivation::kA16;
    source.frame_stride = stride;
    source.ggml         = {.gate_up_format = QType::FP8_E4M3FN_B128, .down_format = QType::FP8_E4M3FN_B128,
                           .gate_row_bytes = H, .down_row_bytes = I, .record_bytes = (kRecord + 15) / 16 * 16};
    Tensor tx(d_x, DType::BF16, {H, columns});
    Tensor out(d_out, DType::BF16, {H, top_k * columns});
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "event");
    cuda_check(cudaEventCreate(&stop), "event");
    infernix::ops::moe_experts(tx, dispatch, source, top_k, experts, d_workspace, out, nullptr); // warm-up
    cuda_check(cudaEventRecord(start), "event");
    for (int r = 0; r < reps; ++r) { infernix::ops::moe_experts(tx, dispatch, source, top_k, experts, d_workspace, out, nullptr); }
    cuda_check(cudaEventRecord(stop), "event");
    cuda_check(cudaEventSynchronize(stop), "event");
    float ms = 0.0F;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "event");
    ms /= reps;
    const double macs = static_cast<double>(top_k) * columns * 3.0 * I * H;
    std::printf("fp8 experts x %d tokens: %8.3f ms per layer call, %6.1f effective TOPS\n", columns, ms,
                2.0 * macs / (ms * 1e-3) / 1e12);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (void* p : {static_cast<void*>(d_out), static_cast<void*>(d_scales), d_workspace, static_cast<void*>(d_x), d_dispatch,
                    static_cast<void*>(d_ids), static_cast<void*>(d_weights), static_cast<void*>(d_gate),
                    static_cast<void*>(d_logits), static_cast<void*>(d_frames), static_cast<void*>(d_frames_base)}) {
        cudaFree(p);
    }
}

} // namespace

int main(int argc, char** argv) {
    std::mt19937 rng(20261011);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        try {
            bench(argc > 2 ? std::atoi(argv[2]) : 4096, rng);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "bench: %s\n", e.what());
            return 1;
        }
        return 0;
    }
    check_reference(rng);
    for (const int columns : {1, 3, 9, 40, 300}) { run(columns, rng); }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d fp8 expert checks failed\n", g_failures);
        return 1;
    }
    std::printf("fp8 experts: every width and placement bit-equal to the exact host expert\n");
    return 0;
}
