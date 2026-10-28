// The wide 8-bit activation route of offloaded_sparse_moe's experts (WideA16::kA8: `--a8-activations` at prefill
// widths; LOSSY, opt-in) through the layer route (moe_route, moe_dispatch, moe_experts): GGUF banks
// (MoeGgmlBank::a8; ops/ggml/ggml_a8_tile.cuh's stage-decode tiles) at Qwen3.8-Flash-Next's geometry, and NVFP4
// W4A16 banks (NarrowA16::kA8; cuda/wide_expert_a8.cuh's grouped GEMM) at both registered geometries, the
// Qwen3.5-122B-A10B one with its shared expert as the call's last job.
//
// Calls of more than 128 columns (129, 200, 300, 600, and 2,048 for the IQ3_S pair and Flash-Next's NVFP4 bank,
// zero-copy and staged only) route with a descending expert popularity and two favoured
// experts, so one call holds narrow experts (at most eight columns), wide experts of one partial 64-column tile,
// of one 128-entry tile, and of several. Records are random (finite GGUF scales; NVFP4 codes and scales including
// zero, subnormal and 448) in a pinned host bank, every third expert in a device frame; misses are read zero-copy,
// staged through 3 or 64 slots (GGUF: also in the double-buffered passes), read from copied-ahead slots (NVFP4) or
// computed by the CPU route through the miss service. One token's x holds an Inf.
//
// Oracles and criteria (fixed before the first run):
//   N  An expert with at most eight columns takes the 8-bit narrow kernels: bit-equal to the narrow arithmetic
//      (GGUF: the CPU route's scalar 8-bit expert; NVFP4: nvfp4_a8.h's arithmetic written out as scalar code),
//      both qualified against binary64 in their own suites.
//   W1 A wider expert, against the FP64 chain over the codec (independently decoded weights, ggml_a8.h's codes and
//      scales of x, FP64 dot products, BF16 at the semantic boundaries gate, up, h and y, SwiGLU as
//      canon::swiglu_bf16, h's codes from the BF16 h): every entry's relative L2 error <= 2^-5 (a code of h may
//      differ where the route's FP32 gate or up lands next to a BF16 tie; a layout or scale error moves an entry
//      by ~100 %).
//   W2 At least 90 % of the wide entries are clean: relative L2 error <= 2^-7 (the outputs' BF16 rounding and the
//      odd BF16 tie of an FP32 sum at gate, up or y, about 2^-9 together; no flipped block of h codes).
//   W3 Every output of the Inf token's entries is NaN.
//   P  Placement invariance: every configuration's outputs equal the zero-copy call's, bit for bit (the narrow and
//      the CPU route included).
//   C  Column-count invariance: an expert's output for a token is bit-equal in calls where the expert holds 300 and
//      140 columns (two and three tiles, partial last tiles of different widths) and with the other tokens changed.

#include "ops/op_family.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "core/ggml_format.h"
#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_a8.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/offloaded_sparse_moe/cpu/ggml_expert.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "ops/op_tester.h"
#include "ggml_block_oracle.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace moe      = infernix::ops::offloaded_moe;
namespace fixtures = infernix::test::offloaded_moe;
namespace canon    = infernix::ops::canon;
namespace a8       = infernix::ggml::a8;
using infernix::DType;
using infernix::QType;
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

template <class Fn>
void parallel_for(int n, const Fn& fn) {
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    const unsigned workers = std::max(1U, std::thread::hardware_concurrency());
    for (unsigned w = 0; w < workers; ++w) {
        pool.emplace_back([&] {
            for (int i = next++; i < n; i = next++) { fn(i); }
        });
    }
    for (auto& t : pool) { t.join(); }
}

constexpr int kExperts      = 24;
constexpr int kStagingSlots = 64;
constexpr int kFavoured     = 1; // the expert every column of the invariance calls routes to

// ------------------------------------------------------------------------------------------- the codec

struct A8Vector {
    std::vector<int> code;
    std::vector<float> scale;
};

A8Vector encode_a8(const std::uint16_t* v, int k) {
    A8Vector a{std::vector<int>(static_cast<std::size_t>(k)), std::vector<float>(static_cast<std::size_t>(k / 32))};
    for (int s = 0; s < k / 32; ++s) {
        unsigned key = 0;
        for (int j = 0; j < 32; ++j) { key = std::max(key, static_cast<unsigned>(v[32 * s + j] & 0x7FFFU)); }
        const a8::SliceScale sc              = a8::slice_scale(key);
        a.scale[static_cast<std::size_t>(s)] = sc.d;
        for (int j = 0; j < 32; ++j) { a.code[static_cast<std::size_t>(32 * s + j)] = a8::code(v[32 * s + j], sc.inv); }
    }
    return a;
}

double dot_a8(const double* w, const A8Vector& a, int k) {
    double s = 0.0;
    for (int j = 0; j < k; ++j) {
        s += w[j] * static_cast<double>(a.code[static_cast<std::size_t>(j)]) * static_cast<double>(a.scale[static_cast<std::size_t>(j / 32)]);
    }
    return s;
}

std::uint16_t bf16_of(double v) { return canon::f32_to_bf16_rn(static_cast<float>(v)); }

// --------------------------------------------------------------------------------- the FP64 wide oracle (W1)

// An expert's exactly decoded weights: gate and up rows [I][H], down rows [H][I], and each projection's output
// multiplier (GGUF 1; NVFP4 2^-10 m, its doubled codes times e4m3_scaled being integers).
struct Decoded {
    int hidden = 0, intermediate = 0;
    std::vector<double> gate, up, down;
    double gate_scale = 1.0, up_scale = 1.0, down_scale = 1.0;
};

// x -> A8 -> gate/up (FP64) -> BF16 -> SwiGLU -> BF16 h -> A8 -> down (FP64) -> BF16.
std::vector<std::uint16_t> wide_chain(const Decoded& e, const std::uint16_t* x) {
    const int H = e.hidden, I = e.intermediate;
    const A8Vector ax = encode_a8(x, H);
    std::vector<std::uint16_t> h(static_cast<std::size_t>(I));
    for (int i = 0; i < I; ++i) {
        const std::uint16_t g = bf16_of(dot_a8(&e.gate[static_cast<std::size_t>(i) * H], ax, H) * e.gate_scale);
        const std::uint16_t u = bf16_of(dot_a8(&e.up[static_cast<std::size_t>(i) * H], ax, H) * e.up_scale);
        h[static_cast<std::size_t>(i)] = canon::swiglu_bf16(g, u);
    }
    const A8Vector ah = encode_a8(h.data(), I);
    std::vector<std::uint16_t> y(static_cast<std::size_t>(H));
    for (int r = 0; r < H; ++r) { y[static_cast<std::size_t>(r)] = bf16_of(dot_a8(&e.down[static_cast<std::size_t>(r) * I], ah, I) * e.down_scale); }
    return y;
}

std::uint32_t row_bytes(QType q, int k) { return static_cast<std::uint32_t>(k / infernix::ggml_block_elems(q) * infernix::ggml_block_bytes(q)); }

void decode_ggml_rows(QType q, const std::uint8_t* rows, int count, int k, double* out) {
    const std::size_t rb = row_bytes(q, k), be = infernix::ggml_block_elems(q), bb = infernix::ggml_block_bytes(q);
    parallel_for(count, [&](int r) {
        for (std::size_t b = 0; b < static_cast<std::size_t>(k) / be; ++b) {
            infernix::test::ggml_oracle::decode_block(q, rows + r * rb + b * bb, out + static_cast<std::size_t>(r) * k + b * be);
        }
    });
}

Decoded decode_ggml(const std::uint8_t* record, QType gu, QType down) {
    const int H = moe::kHidden, I = moe::kIntermediate;
    Decoded e;
    e.hidden = H;
    e.intermediate = I;
    e.gate.resize(static_cast<std::size_t>(I) * H);
    e.up.resize(e.gate.size());
    e.down.resize(static_cast<std::size_t>(H) * I);
    const std::uint32_t gr = row_bytes(gu, H);
    decode_ggml_rows(gu, record, I, H, e.gate.data());
    decode_ggml_rows(gu, record + static_cast<std::size_t>(I) * gr, I, H, e.up.data());
    decode_ggml_rows(down, record + 2ULL * I * gr, H, I, e.down.data());
    return e;
}

// Twice the E2M1 value of the weight at row r (0-15) and element k (0-15) of a 144-byte unit (nvfp4_expert_rg16_v1).
int weight_c2(const std::uint8_t* unit, int r, int k) {
    const unsigned byte = unit[32 * (k / 4) + 4 * (r % 8) + k % 4];
    return canon::e2m1_x2(r < 8 ? byte & 15U : byte >> 4);
}

// Row `row` of a matrix of row groups of `blocks` units: c2 * e4m3_scaled per element (exact integers).
void decode_nvfp4_row(const std::uint8_t* matrix, int blocks, int row, double* out) {
    const int group = row / 16, r = row % 16;
    for (int b = 0; b < blocks; ++b) {
        const std::uint8_t* unit = matrix + (static_cast<std::size_t>(group) * blocks + b) * moe::kUnitBytes;
        const double s           = canon::e4m3_scaled(unit[128 + r]);
        for (int k = 0; k < 16; ++k) { out[16 * b + k] = weight_c2(unit, r, k) * s; }
    }
}

Decoded decode_nvfp4(const moe::ExpertShape& shape, const fixtures::Expert& expert) {
    const int H = shape.hidden, I = shape.intermediate;
    Decoded e;
    e.hidden = H;
    e.intermediate = I;
    e.gate.resize(static_cast<std::size_t>(I) * H);
    e.up.resize(e.gate.size());
    e.down.resize(static_cast<std::size_t>(H) * I);
    parallel_for(I, [&](int i) { // gate row 2i, up row 2i + 1
        decode_nvfp4_row(expert.record.data(), H / 16, 2 * i, &e.gate[static_cast<std::size_t>(i) * H]);
        decode_nvfp4_row(expert.record.data(), H / 16, 2 * i + 1, &e.up[static_cast<std::size_t>(i) * H]);
    });
    parallel_for(H, [&](int r) { decode_nvfp4_row(expert.record.data() + shape.gate_up_bytes(), I / 16, r, &e.down[static_cast<std::size_t>(r) * I]); });
    e.gate_scale = std::ldexp(static_cast<double>(expert.scales.alpha_gate), -10);
    e.up_scale   = std::ldexp(static_cast<double>(expert.scales.alpha_up), -10);
    e.down_scale = std::ldexp(static_cast<double>(expert.scales.alpha_down), -10);
    return e;
}

// ------------------------------------------------------------------------------ the narrow oracles (N)

// nvfp4_a8.h's arithmetic written out (test_offloaded_moe_w4a8.cu's oracle).
std::uint16_t nvfp4_row_a8(const std::uint8_t* matrix, int blocks, int row, const A8Vector& a, int partials, float m) {
    const int group = row / 16, r = row % 16;
    float p[16] = {};
    for (int s = 0; s < blocks / 2; ++s) {
        for (int b = 2 * s; b <= 2 * s + 1; ++b) {
            const std::uint8_t* unit = matrix + (static_cast<std::size_t>(group) * blocks + b) * moe::kUnitBytes;
            std::int32_t pb          = 0;
            for (int k = 0; k < 16; ++k) { pb += weight_c2(unit, r, k) * a.code[static_cast<std::size_t>(16 * b + k)]; }
            const float sw = static_cast<float>(canon::e4m3_scaled(unit[128 + r]));
            float& partial = p[s % partials];
            partial = canon::fma_rn(canon::mul_rn(static_cast<float>(pb), a.scale[static_cast<std::size_t>(s)]), sw, partial);
        }
    }
    for (int o = partials / 2; o >= 1; o /= 2) {
        for (int i = 0; i < o; ++i) { p[i] = canon::add_rn(p[i], p[i + o]); }
    }
    const float scaled = canon::mul_rn(canon::mul_rn(p[0], 0x1p-5F), 0x1p-5F);
    return canon::f32_to_bf16_rn(canon::mul_rn(scaled, m));
}

std::vector<std::uint16_t> nvfp4_narrow(const moe::ExpertShape& shape, const fixtures::Expert& e, const std::uint16_t* x) {
    const int H = shape.hidden, I = shape.intermediate;
    const A8Vector ax = encode_a8(x, H);
    std::vector<std::uint16_t> h(static_cast<std::size_t>(I));
    for (int i = 0; i < I; ++i) {
        const std::uint16_t g = nvfp4_row_a8(e.record.data(), H / 16, 2 * i, ax, 16, e.scales.alpha_gate);
        const std::uint16_t u = nvfp4_row_a8(e.record.data(), H / 16, 2 * i + 1, ax, 16, e.scales.alpha_up);
        h[static_cast<std::size_t>(i)] = canon::swiglu_bf16(g, u);
    }
    const A8Vector ah = encode_a8(h.data(), I);
    std::vector<std::uint16_t> y(static_cast<std::size_t>(H));
    for (int r = 0; r < H; ++r) { y[static_cast<std::size_t>(r)] = nvfp4_row_a8(e.record.data() + shape.gate_up_bytes(), I / 16, r, ah, 4, e.scales.alpha_down); }
    return y;
}

std::vector<std::uint16_t> ggml_narrow(const std::uint8_t* record, const moe::GgmlExpertFormat& format, const std::uint16_t* x) {
    std::vector<std::uint16_t> y(static_cast<std::size_t>(moe::kHidden));
    const std::uint16_t* xs[] = {x};
    std::uint16_t* ys[]       = {y.data()};
    moe::ggml_a8_expert_forward(moe::CpuIsa::kScalar, record, format, 1, xs, ys);
    return y;
}

// ------------------------------------------------------------------------------------------- the banks

void random_ggml_rows(QType q, int rows, int k, std::mt19937& rng, std::uint8_t* out) {
    const int blocks = k / static_cast<int>(infernix::ggml_block_elems(q)), size = static_cast<int>(infernix::ggml_block_bytes(q));
    for (int i = 0; i < rows * blocks * size; ++i) { out[i] = static_cast<std::uint8_t>(rng()); }
    for (int i = 0; i < rows * blocks; ++i) {
        std::uint8_t* b      = out + static_cast<std::size_t>(i) * size;
        const unsigned field = 3 + rng() % 10; // finite scales of mixed binades
        const auto w         = static_cast<std::uint16_t>((rng() & 0x8000U) | (rng() & 0x3FFU) | (field << 10));
        b[0]                 = static_cast<std::uint8_t>(w & 0xFF);
        b[1]                 = static_cast<std::uint8_t>(w >> 8);
    }
}

struct Bank {
    std::string name;
    bool ggml = false;
    QType gu = QType::NVFP4_MUL, down = QType::NVFP4_MUL;
    moe::GgmlExpertFormat format{};
    moe::ExpertShape shape = moe::kFlashNextShape;
    int top_k              = 4;
    bool shared            = false; // NVFP4 122B: the shared expert as the call's last job
    std::uint64_t record = 0, stride = 0, pitch = 0;
    std::uint8_t* host        = nullptr; // pinned, mapped
    std::uint8_t* host_device = nullptr;
    std::vector<fixtures::Expert> experts; // NVFP4
    fixtures::Expert shared_expert;
    std::vector<moe::ExpertScales> scales; // routed (host)
    std::vector<std::int32_t> frames;
    std::uint8_t* d_frame_base    = nullptr;
    std::int32_t* d_frames        = nullptr;
    moe::ExpertScales* d_scales   = nullptr;
    std::uint8_t* d_shared_record = nullptr;
    std::int32_t* d_prefetched    = nullptr;
    std::uint8_t* d_prefetch_base = nullptr;
    std::uint8_t* d_staging       = nullptr;
    std::unique_ptr<moe::CpuMissService> service;
    cudaStream_t overlap = nullptr;
    cudaEvent_t events[5]{};

    [[nodiscard]] const std::uint8_t* host_record(int e) const { return host + static_cast<std::uint64_t>(e) * stride; }
};

void finish_bank(Bank& bank, std::mt19937& rng) {
    cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&bank.host_device), bank.host, 0), "cudaHostGetDevicePointer");
    bank.frames.assign(kExperts, -1);
    int resident = 0;
    for (int e = 0; e < kExperts; e += 3) { bank.frames[static_cast<std::size_t>(e)] = resident++; }
    std::vector<std::int32_t> prefetched(kExperts, -1);
    int slots = 0;
    for (int e = 2; e < kExperts; e += 3) { prefetched[static_cast<std::size_t>(e)] = slots++; }
    cuda_check(cudaMalloc(&bank.d_frame_base, bank.pitch * resident), "cudaMalloc");
    cuda_check(cudaMalloc(&bank.d_prefetch_base, bank.pitch * slots), "cudaMalloc");
    for (int e = 0; e < kExperts; ++e) {
        if (bank.frames[static_cast<std::size_t>(e)] >= 0) {
            cuda_check(cudaMemcpy(bank.d_frame_base + bank.frames[static_cast<std::size_t>(e)] * bank.pitch, bank.host_record(e),
                                  bank.stride, cudaMemcpyHostToDevice), "cudaMemcpy");
        }
        if (prefetched[static_cast<std::size_t>(e)] >= 0) {
            cuda_check(cudaMemcpy(bank.d_prefetch_base + prefetched[static_cast<std::size_t>(e)] * bank.pitch, bank.host_record(e),
                                  bank.stride, cudaMemcpyHostToDevice), "cudaMemcpy");
        }
    }
    bank.d_frames     = device_copy(bank.frames);
    bank.d_prefetched = device_copy(prefetched);
    cuda_check(cudaMalloc(&bank.d_staging, bank.pitch * kStagingSlots), "cudaMalloc");
    std::vector<moe::ExpertScales> all = bank.scales;
    if (bank.shared) {
        bank.shared_expert   = fixtures::random_a16_expert(rng, bank.shape);
        bank.d_shared_record = device_copy(bank.shared_expert.record);
        all.push_back(bank.shared_expert.scales);
    }
    bank.d_scales = device_copy(all);
    cuda_check(cudaStreamCreateWithFlags(&bank.overlap, cudaStreamNonBlocking), "stream");
    for (auto& e : bank.events) { cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event"); }
    try {
        moe::CpuMissService::Options options;
        options.workers     = 4;
        options.max_jobs    = 8;
        options.max_columns = 1024;
        options.pcie_divisor = 2;
        options.activation  = moe::ExpertActivation::kA16;
        options.shape       = bank.shape;
        if (bank.ggml) {
            options.ggml_a8 = true;
        } else {
            options.narrow_a16 = moe::NarrowA16::kA8;
            options.nvfp4_a8   = true;
        }
        bank.service = std::make_unique<moe::CpuMissService>(
            std::vector<moe::CpuMissService::Layer>{{.records = bank.host, .record_stride = bank.stride,
                                                     .scales = bank.scales.data(), .ggml = bank.format}},
            options);
    } catch (const std::exception& error) {
        check(false, bank.name + ": the CPU route takes the bank's 8-bit jobs (" + error.what() + ")");
    }
}

void make_ggml_bank(Bank& bank, QType gu, QType down, std::mt19937& rng) {
    const int H = moe::kHidden, I = moe::kIntermediate;
    bank.ggml   = true;
    bank.gu     = gu;
    bank.down   = down;
    bank.top_k  = 4;
    bank.name   = "GGUF (" + std::to_string(static_cast<int>(gu)) + ", " + std::to_string(static_cast<int>(down)) + ")";
    bank.format = {.gate_up = gu, .down = down, .gate_row_bytes = row_bytes(gu, H), .down_row_bytes = row_bytes(down, I)};
    bank.record = 2ULL * I * row_bytes(gu, H) + static_cast<std::uint64_t>(H) * row_bytes(down, I);
    bank.stride = (bank.record + 4095) / 4096 * 4096;
    bank.pitch  = bank.stride + 8192;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&bank.host), bank.stride * kExperts, cudaHostAllocMapped), "cudaHostAlloc");
    std::memset(bank.host, 0, bank.stride * kExperts);
    for (int e = 0; e < kExperts; ++e) {
        std::uint8_t* r = bank.host + e * bank.stride;
        random_ggml_rows(gu, 2 * I, H, rng, r);
        random_ggml_rows(down, H, I, rng, r + 2ULL * I * row_bytes(gu, H));
    }
    bank.scales.assign(kExperts, moe::ExpertScales{});
    finish_bank(bank, rng);
}

void make_nvfp4_bank(Bank& bank, const moe::ExpertShape& shape, bool shared, const char* name, std::mt19937& rng) {
    bank.shape  = shape;
    bank.shared = shared;
    bank.top_k  = 8;
    bank.name   = name;
    bank.record = shape.record_bytes();
    bank.stride = (bank.record + 4095) / 4096 * 4096;
    bank.pitch  = bank.stride + 4096;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&bank.host), bank.stride * kExperts, cudaHostAllocMapped), "cudaHostAlloc");
    std::memset(bank.host, 0, bank.stride * kExperts);
    for (int e = 0; e < kExperts; ++e) {
        bank.experts.push_back(fixtures::random_a16_expert(rng, shape));
        std::memcpy(bank.host + e * bank.stride, bank.experts.back().record.data(), shape.record_bytes());
        bank.scales.push_back(bank.experts.back().scales);
    }
    finish_bank(bank, rng);
}

void free_bank(Bank& bank) {
    bank.service.reset();
    for (auto e : bank.events) { cudaEventDestroy(e); }
    cudaStreamDestroy(bank.overlap);
    cudaFree(bank.d_staging);
    cudaFree(bank.d_prefetched);
    cudaFree(bank.d_prefetch_base);
    cudaFree(bank.d_frames);
    cudaFree(bank.d_frame_base);
    cudaFree(bank.d_scales);
    cudaFree(bank.d_shared_record);
    cudaFreeHost(bank.host);
}

// ------------------------------------------------------------------------------------------- calls

struct Config {
    const char* name;
    int slots     = 0;
    bool overlap  = false; // GGUF: the double-buffered passes (MoeExpertSource::overlap_stream)
    bool prefetch = false; // copied-ahead slots (MoeExpertSource::prefetched)
    bool cpu      = false; // CPU-served misses
};

// Logits of `columns` columns: a descending popularity (expert e's logit lowered by 0.3 e) and two favoured
// experts (1 and 2), so a call holds narrow and wide experts of several tile counts; with `all_favoured` every
// column routes to kFavoured.
std::vector<float> make_logits(std::mt19937& rng, int columns, bool all_favoured) {
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> logits(static_cast<std::size_t>(kExperts + 1) * columns);
    for (int t = 0; t < columns; ++t) {
        float* column = &logits[static_cast<std::size_t>(t) * (kExperts + 1)];
        for (int e = 0; e <= kExperts; ++e) { column[e] = n(rng) - 0.3F * static_cast<float>(e); }
        column[1] += 3.0F;
        column[2] += 1.5F;
        if (all_favoured) { column[kFavoured] += 20.0F; }
    }
    return logits;
}

// Wide-range activations with a zero slice per column and an Inf in column 3's slice 5.
std::vector<std::uint16_t> make_x(std::mt19937& rng, int columns, int hidden) {
    std::vector<std::uint16_t> x = fixtures::wide_range_activations(rng, columns, hidden);
    for (int t = 0; t < columns; ++t) {
        const int zero = (3 + 7 * t) % (hidden / 32);
        for (int j = 0; j < 32; ++j) { x[static_cast<std::size_t>(t) * hidden + 32 * zero + j] = 0; }
    }
    if (columns > 3) { x[static_cast<std::size_t>(3) * hidden + 5 * 32 + 7] = 0x7F80; }
    return x;
}

std::vector<std::int32_t> route_ids(const std::vector<float>& logits, int columns, int top_k) {
    auto* d_logits      = device_copy(logits);
    std::int32_t* d_ids = nullptr;
    float *d_weights = nullptr, *d_gate = nullptr;
    cuda_check(cudaMalloc(&d_ids, sizeof(std::int32_t) * top_k * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_weights, sizeof(float) * top_k * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_gate, sizeof(float) * columns), "cudaMalloc");
    infernix::ops::MoeRouting routing{Tensor(d_ids, DType::I32, {top_k, columns}), Tensor(d_weights, DType::FP32, {top_k, columns}),
                                      Tensor(d_gate, DType::FP32, {columns})};
    infernix::ops::moe_route(Tensor(d_logits, DType::FP32, {kExperts + 1, columns}), top_k, routing, nullptr);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(top_k) * columns);
    cuda_check(cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
    cudaFree(d_gate);
    cudaFree(d_weights);
    cudaFree(d_ids);
    cudaFree(d_logits);
    return ids;
}

// One layer call's outputs (BF16 [H, entries]).
std::vector<std::uint16_t> layer_call(Bank& bank, const std::vector<float>& logits, const std::vector<std::uint16_t>& x, int columns,
                                      const Config& config, bool& cpu_served) {
    const int H = bank.shape.hidden, top_k = bank.top_k;
    const int entries          = (top_k + (bank.shared ? 1 : 0)) * columns;
    const int dispatch_experts = kExperts + (bank.shared ? 1 : 0);
    const int max_jobs         = std::min(kExperts, top_k * columns) + (bank.shared ? 1 : 0);
    auto* d_logits             = device_copy(logits);
    auto* d_x                  = device_copy(x);
    std::int32_t* d_ids        = nullptr;
    float *d_weights = nullptr, *d_gate = nullptr;
    cuda_check(cudaMalloc(&d_ids, sizeof(std::int32_t) * top_k * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_weights, sizeof(float) * top_k * columns), "cudaMalloc");
    cuda_check(cudaMalloc(&d_gate, sizeof(float) * columns), "cudaMalloc");
    infernix::ops::MoeRouting routing{Tensor(d_ids, DType::I32, {top_k, columns}), Tensor(d_weights, DType::FP32, {top_k, columns}),
                                      Tensor(d_gate, DType::FP32, {columns})};
    infernix::ops::moe_route(Tensor(d_logits, DType::FP32, {kExperts + 1, columns}), top_k, routing, nullptr);
    void* d_dispatch = nullptr;
    cuda_check(cudaMalloc(&d_dispatch, infernix::ops::moe_dispatch_bytes(dispatch_experts, entries)), "cudaMalloc");
    auto dispatch = infernix::ops::carve_moe_dispatch(d_dispatch, dispatch_experts, entries);
    infernix::ops::moe_dispatch(routing, kExperts, dispatch, nullptr, nullptr, bank.shared);
    void* d_workspace = nullptr;
    cuda_check(cudaMalloc(&d_workspace, infernix::ops::moe_experts_workspace_bytes(bank.shape, max_jobs, entries, columns,
                                                                                  moe::ExpertActivation::kA16, moe::WideA16::kA8)),
               "cudaMalloc");
    std::uint16_t* d_out = nullptr;
    const std::size_t bytes = sizeof(std::uint16_t) * H * entries;
    cuda_check(cudaMalloc(&d_out, bytes), "cudaMalloc");
    cuda_check(cudaMemset(d_out, 0xFF, bytes), "cudaMemset");

    infernix::ops::MoeExpertSource source{.frame_base    = bank.d_frame_base,
                                          .frames        = bank.d_frames,
                                          .host_records  = bank.host_device,
                                          .record_stride = bank.stride,
                                          .scales        = bank.d_scales,
                                          .staging_base  = config.slots > 0 ? bank.d_staging : nullptr,
                                          .staging_slots = config.slots};
    source.activation   = moe::ExpertActivation::kA16;
    source.wide_a16     = moe::WideA16::kA8;
    source.frame_stride = bank.pitch;
    source.shape        = bank.shape;
    if (bank.ggml) {
        source.ggml = {.gate_up_format = bank.gu,
                       .down_format    = bank.down,
                       .gate_row_bytes = bank.format.gate_row_bytes,
                       .down_row_bytes = bank.format.down_row_bytes,
                       .record_bytes   = (bank.record + 15) / 16 * 16,
                       .a8             = true};
    } else {
        source.narrow_a16 = moe::NarrowA16::kA8;
    }
    if (bank.shared) {
        source.shared_expert = kExperts;
        source.shared_record = bank.d_shared_record;
    }
    if (config.overlap) {
        source.overlap_stream = bank.overlap;
        for (int i = 0; i < 5; ++i) { source.overlap_events[i] = bank.events[i]; }
    }
    if (config.prefetch) {
        source.prefetched    = bank.d_prefetched;
        source.prefetch_base = bank.d_prefetch_base;
    }
    const std::uint64_t served = config.cpu && bank.service ? bank.service->served_experts() : 0;
    if (config.cpu && bank.service) {
        bank.service->wake();
        source.cpu = bank.service->channel(0);
    }
    Tensor tx(d_x, DType::BF16, {H, columns});
    Tensor out(d_out, DType::BF16, {H, entries});
    infernix::ops::moe_experts(tx, dispatch, source, top_k, max_jobs, d_workspace, out, nullptr);
    cuda_check(cudaDeviceSynchronize(), "moe_experts");
    cpu_served = config.cpu && bank.service && bank.service->served_experts() > served;
    std::vector<std::uint16_t> got(static_cast<std::size_t>(H) * entries);
    cuda_check(cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy");
    cudaFree(d_out);
    cudaFree(d_workspace);
    cudaFree(d_dispatch);
    cudaFree(d_gate);
    cudaFree(d_weights);
    cudaFree(d_ids);
    cudaFree(d_x);
    cudaFree(d_logits);
    return got;
}

// ------------------------------------------------------------------------------------------- checks

// Entry i's expert (routed entries t k + slot, then the shared job's k T + t) and token.
struct EntryInfo {
    int expert; // kExperts: the shared expert
    int token;
};

EntryInfo entry_info(const Bank& bank, const std::vector<std::int32_t>& ids, int columns, int i) {
    const int routed = bank.top_k * columns;
    if (i >= routed) { return {kExperts, i - routed}; }
    return {ids[static_cast<std::size_t>(i)], i / bank.top_k};
}

void check_call(Bank& bank, int columns, std::mt19937& rng) {
    const int H = bank.shape.hidden;
    const auto logits = make_logits(rng, columns, false);
    const auto x      = make_x(rng, columns, H);
    const auto ids    = route_ids(logits, columns, bank.top_k);
    const int entries = (bank.top_k + (bank.shared ? 1 : 0)) * columns;
    std::vector<int> count(kExperts + 1, 0);
    for (const int e : ids) { ++count[static_cast<std::size_t>(e)]; }
    count[kExperts] = bank.shared ? columns : 0;
    int narrow_experts = 0, wide_experts = 0, most = 0;
    for (int e = 0; e <= kExperts; ++e) {
        if (count[static_cast<std::size_t>(e)] == 0) { continue; }
        (count[static_cast<std::size_t>(e)] > moe::kMaxColumns ? wide_experts : narrow_experts) += 1;
        most = std::max(most, count[static_cast<std::size_t>(e)]);
    }
    const std::string where = bank.name + ", T " + std::to_string(columns) + " (narrow experts " + std::to_string(narrow_experts) +
                              ", wide " + std::to_string(wide_experts) + ", most columns " + std::to_string(most) + ")";

    // The expected outputs: the narrow arithmetic for narrow experts, the FP64 chain for wide ones.
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(H) * entries);
    std::vector<char> wide(static_cast<std::size_t>(entries), 0);
    std::vector<int> narrow_entries;
    for (int i = 0; i < entries; ++i) {
        const EntryInfo info = entry_info(bank, ids, columns, i);
        if (count[static_cast<std::size_t>(info.expert)] > moe::kMaxColumns) {
            wide[static_cast<std::size_t>(i)] = 1;
        } else {
            narrow_entries.push_back(i);
        }
    }
    parallel_for(static_cast<int>(narrow_entries.size()), [&](int n) {
        const int i          = narrow_entries[static_cast<std::size_t>(n)];
        const EntryInfo info = entry_info(bank, ids, columns, i);
        const std::uint16_t* xv = &x[static_cast<std::size_t>(info.token) * H];
        const auto y = bank.ggml ? ggml_narrow(bank.host_record(info.expert), bank.format, xv)
                                 : nvfp4_narrow(bank.shape, bank.experts[static_cast<std::size_t>(info.expert)], xv);
        std::memcpy(&expected[static_cast<std::size_t>(i) * H], y.data(), sizeof(std::uint16_t) * H);
    });
    for (int e = 0; e <= kExperts; ++e) {
        if (count[static_cast<std::size_t>(e)] <= moe::kMaxColumns) { continue; }
        const Decoded decoded = bank.ggml ? decode_ggml(bank.host_record(e), bank.gu, bank.down)
                                          : decode_nvfp4(bank.shape, e == kExperts ? bank.shared_expert : bank.experts[static_cast<std::size_t>(e)]);
        std::vector<int> mine;
        for (int i = 0; i < entries; ++i) {
            if (entry_info(bank, ids, columns, i).expert == e) { mine.push_back(i); }
        }
        parallel_for(static_cast<int>(mine.size()), [&](int n) {
            const int i = mine[static_cast<std::size_t>(n)];
            const auto y = wide_chain(decoded, &x[static_cast<std::size_t>(entry_info(bank, ids, columns, i).token) * H]);
            std::memcpy(&expected[static_cast<std::size_t>(i) * H], y.data(), sizeof(std::uint16_t) * H);
        });
    }

    const Config configs[] = {{.name = "zero-copy"},
                              {.name = "3 staging slots", .slots = 3},
                              {.name = "64 staging slots", .slots = kStagingSlots},
                              {.name = "64 staging slots, double-buffered passes", .slots = kStagingSlots, .overlap = true},
                              {.name = "copied-ahead slots", .slots = kStagingSlots, .prefetch = true},
                              {.name = "CPU-served misses", .slots = kStagingSlots, .cpu = true}};
    std::vector<std::uint16_t> first;
    for (const Config& config : configs) {
        if (config.overlap && !bank.ggml) { continue; } // an NVFP4 wide call keeps serial passes
        if (config.cpu && bank.service == nullptr) { continue; }
        // A prefill-scale call: the zero-copy call and the staged one (the other placements are the narrower calls').
        if (columns > 1024 && (config.overlap || config.prefetch || config.cpu || config.slots == 3)) { continue; }
        bool cpu_served = false;
        const auto got   = layer_call(bank, logits, x, columns, config, cpu_served);
        const std::string at = where + ", " + config.name;
        if (config.cpu) { check(cpu_served, at + ": the CPU served experts"); }
        if (!first.empty()) {
            std::size_t differ = 0;
            for (std::size_t i = 0; i < got.size(); ++i) { differ += got[i] != first[i]; }
            check(differ == 0, at + ": " + std::to_string(differ) + " outputs differ from the zero-copy call's (P)");
            continue;
        }
        first = got;
        // N: narrow entries bit for bit.
        std::size_t bad_narrow = 0;
        for (const int i : narrow_entries) {
            for (int r = 0; r < H; ++r) {
                const std::size_t k = static_cast<std::size_t>(i) * H + r;
                bad_narrow += got[k] != expected[k];
            }
        }
        check(bad_narrow == 0, at + ": " + std::to_string(bad_narrow) + " narrow outputs differ from the narrow arithmetic (N)");
        // W1-W3: wide entries against the FP64 chain.
        int wide_entries = 0, clean = 0, bad = 0, missing_nan = 0;
        double worst = 0.0;
        for (int i = 0; i < entries; ++i) {
            if (wide[static_cast<std::size_t>(i)] == 0) { continue; }
            const bool inf_token = entry_info(bank, ids, columns, i).token == 3;
            double err2 = 0.0, ref2 = 0.0;
            for (int r = 0; r < H; ++r) {
                const std::size_t k = static_cast<std::size_t>(i) * H + r;
                const double y = canon::bf16_to_f32(got[k]), ref = canon::bf16_to_f32(expected[k]);
                if (inf_token) {
                    missing_nan += !std::isnan(y);
                    continue;
                }
                err2 += (y - ref) * (y - ref);
                ref2 += ref * ref;
            }
            if (inf_token) { continue; }
            ++wide_entries;
            const double rel = std::sqrt(err2) / std::max(std::sqrt(ref2), 1e-30);
            worst = std::max(worst, std::isfinite(rel) ? rel : 1e30);
            bad += !(rel <= std::ldexp(1.0, -5));
            clean += rel <= std::ldexp(1.0, -7) ? 1 : 0;
        }
        check(bad == 0, at + ": " + std::to_string(bad) + " of " + std::to_string(wide_entries) + " wide entries past 2^-5 relative L2 (W1)");
        check(wide_entries == 0 || clean * 10 >= wide_entries * 9,
              at + ": " + std::to_string(clean) + " of " + std::to_string(wide_entries) + " wide entries clean (W2)");
        check(missing_nan == 0, at + ": " + std::to_string(missing_nan) + " outputs of the Inf token are not NaN (W3)");
        std::printf("%s: %d wide entries, %d clean, worst relative L2 %.2e\n", at.c_str(), wide_entries, clean, worst);
    }
}

// C: kFavoured holds every column of a 300-column call and of a 140-column call over the same first 140 tokens
// (whose other tokens differ): its outputs for those tokens are bit-equal.
void check_invariance(Bank& bank, std::mt19937& rng) {
    constexpr int kWide = 300, kNarrower = 140;
    const int H     = bank.shape.hidden;
    auto logits     = make_logits(rng, kWide, true);
    auto x          = make_x(rng, kWide, H);
    const std::vector<float> logits2(logits.begin(), logits.begin() + static_cast<std::ptrdiff_t>(kNarrower) * (kExperts + 1));
    const std::vector<std::uint16_t> x2(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(kNarrower) * H);
    const auto ids = route_ids(logits, kWide, bank.top_k), ids2 = route_ids(logits2, kNarrower, bank.top_k);
    const Config config{.name = "64 staging slots", .slots = kStagingSlots};
    bool unused = false;
    const auto wide   = layer_call(bank, logits, x, kWide, config, unused);
    const auto narrow = layer_call(bank, logits2, x2, kNarrower, config, unused);
    std::size_t differ = 0, compared = 0;
    for (int t = 0; t < kNarrower; ++t) {
        int a = -1, b = -1;
        for (int k = 0; k < bank.top_k; ++k) {
            if (ids[static_cast<std::size_t>(t * bank.top_k + k)] == kFavoured) { a = t * bank.top_k + k; }
            if (ids2[static_cast<std::size_t>(t * bank.top_k + k)] == kFavoured) { b = t * bank.top_k + k; }
        }
        if (a < 0 || b < 0) { continue; }
        ++compared;
        differ += !std::equal(wide.begin() + static_cast<std::ptrdiff_t>(a) * H, wide.begin() + static_cast<std::ptrdiff_t>(a + 1) * H,
                              narrow.begin() + static_cast<std::ptrdiff_t>(b) * H);
    }
    check(compared == static_cast<std::size_t>(kNarrower), bank.name + ": every column routes to the favoured expert");
    check(differ == 0, bank.name + ": " + std::to_string(differ) + " of the favoured expert's entries differ between calls of 300 "
                                       "and 140 columns (C)");
}

} // namespace

INFERNIX_OP_TEST(offloaded_moe_a8_wide) {
    std::mt19937 rng(20261028);
    const std::pair<QType, QType> pairs[] = {{QType::GGML_IQ3_S, QType::GGML_IQ4_NL},
                                             {QType::GGML_IQ2_S, QType::GGML_Q2_0},
                                             {QType::GGML_IQ3_XXS, QType::GGML_IQ4_NL},
                                             {QType::GGML_IQ4_XS, QType::GGML_Q2_0}};
    bool first_pair = true;
    for (const auto& [gu, down] : pairs) {
        Bank bank;
        make_ggml_bank(bank, gu, down, rng);
        for (const int columns : {129, 200, 600}) { check_call(bank, columns, rng); }
        if (first_pair) { check_call(bank, 2048, rng); } // the IQ3_S import's pair at a prefill chunk's width
        first_pair = false;
        check_invariance(bank, rng);
        free_bank(bank);
    }
    struct Geometry {
        moe::ExpertShape shape;
        bool shared;
        const char* name;
    };
    for (const Geometry& geometry : {Geometry{moe::kFlashNextShape, false, "NVFP4 W4A16, Qwen3.8-Flash-Next geometry"},
                                     Geometry{moe::kQwen3_5_122BShape, true, "NVFP4 W4A16, Qwen3.5-122B-A10B geometry"}}) {
        Bank bank;
        make_nvfp4_bank(bank, geometry.shape, geometry.shared, geometry.name, rng);
        for (const int columns : {129, 300}) { check_call(bank, columns, rng); }
        if (!geometry.shared) { check_call(bank, 2048, rng); } // Flash-Next at a prefill chunk's width
        check_invariance(bank, rng);
        free_bank(bank);
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("offloaded_moe_a8_wide: ok\n");
    return 0;
}
