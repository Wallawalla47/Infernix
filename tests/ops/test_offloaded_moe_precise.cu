// The reference route's experts (offloaded_moe::WideA16::kPrecise; docs/maintainer/model-quality.md
// §6.2) through the layer route of offloaded_sparse_moe (moe_route, moe_dispatch, moe_experts), for
// every A16 bank kind: NVFP4 rg16 records with their FP32 multipliers, GGUF records (IQ3_S and IQ4_XS
// gate/up with IQ4_NL and Q2_0 down rows) and block-FP8 records.
//
// Oracle: each routed (column, expert) pair on the host from weights decoded by this file's own
// readers of the record layouts (GGUF blocks by ggml_block_oracle.h): the gate and up sums exact
// (double-double), times the matrix multiplier, each correctly rounded to BF16; h = BF16(SiLU(g) * u)
// evaluated in binary64; the down sums likewise. Where a sum lies within the route's error
// (precise_dot.cuh) of a rounding boundary, the outputs that depend on it are counted, not compared.
// Every other output must equal the oracle's bits whatever the call width (1, 5, 40 columns), and
// every output must be bit-identical wherever the record lives (device frames whose pitch exceeds
// the record stride, the pinned host bank read zero-copy, 3 or 64 staging slots, the forked route).

#include "core/ggml_format.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "ggml_block_oracle.h"
#include "offloaded_moe_fixtures.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace moe      = infernix::ops::offloaded_moe;
namespace fixtures = infernix::test::offloaded_moe;
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

constexpr int H = moe::kHidden, I = moe::kIntermediate;

double bf16_value(std::uint16_t bits) {
    const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// binary64 -> BF16, one round-to-nearest-even (round to odd binary32 first).
std::uint16_t bf16_bits(double x) {
    float f = static_cast<float>(x);
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    const double d = f;
    if ((x > 0.0 && d > x) || (x < 0.0 && d < x)) { --u; }
    if (d != x && (u << 1) <= 0xFF000000U) { u |= 1U; }
    return static_cast<std::uint16_t>((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
}

double e4m3(std::uint8_t w) {
    const int e = (w >> 3) & 15, m = w & 7;
    const double v = e == 0 ? m * std::ldexp(1.0, -9) : (8 + m) * std::ldexp(1.0, e - 10);
    return (w & 0x80) != 0 ? -v : v;
}

double e2m1(unsigned code) {
    static const double kMagnitude[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return (code & 8U) != 0 ? -kMagnitude[code & 7U] : kMagnitude[code & 7U];
}

// A bank kind: its record geometry, how the device reads it, and the decoded matrices of a record
// (gate and up [I][H], down [H][I]) with each matrix's multiplier.
struct Kind {
    std::string name;
    std::uint64_t record_bytes = 0;
    infernix::ops::MoeGgmlBank ggml{};
    std::function<void(std::mt19937&, std::uint8_t*, moe::ExpertScales&)> make;
    std::function<void(const std::uint8_t*, std::vector<double>&, std::vector<double>&, std::vector<double>&)> decode;
};

// nvfp4_expert_rg16_v1: row r's block b in unit (r / 16, b); gate row i is row 2 i, up row 2 i + 1.
double rg16(const std::uint8_t* matrix, int blocks, int row, int k) {
    const std::uint8_t* unit = matrix + (static_cast<std::size_t>(row / 16) * blocks + k / 16) * moe::kUnitBytes;
    const int r = row % 16, kk = k % 16;
    const unsigned byte = unit[32 * (kk / 4) + 4 * (r % 8) + kk % 4];
    return e2m1((r < 8 ? byte : byte >> 4) & 15U) * e4m3(unit[128 + r]);
}

Kind nvfp4_kind() {
    Kind kind;
    kind.name         = "nvfp4 rg16";
    kind.record_bytes = moe::kRecordBytes;
    kind.ggml         = {}; // NVFP4_MUL: not a ggml bank
    kind.make = [](std::mt19937& rng, std::uint8_t* record, moe::ExpertScales& scales) {
        const fixtures::Expert e = fixtures::random_a16_expert(rng);
        std::memcpy(record, e.record.data(), moe::kRecordBytes);
        scales = e.scales;
    };
    kind.decode = [](const std::uint8_t* record, std::vector<double>& g, std::vector<double>& u, std::vector<double>& d) {
        for (int i = 0; i < I; ++i) {
            for (int k = 0; k < H; ++k) {
                g[static_cast<std::size_t>(i) * H + k] = rg16(record, moe::kGateUpBlocks, 2 * i, k);
                u[static_cast<std::size_t>(i) * H + k] = rg16(record, moe::kGateUpBlocks, 2 * i + 1, k);
            }
        }
        for (int r = 0; r < H; ++r) {
            for (int k = 0; k < I; ++k) {
                d[static_cast<std::size_t>(r) * I + k] = rg16(record + moe::kGateUpBytes, moe::kDownBlocks, r, k);
            }
        }
    };
    return kind;
}

// Random GGUF rows: random bytes with finite binary16 block scales (exponent fields 5..20).
void random_ggml_rows(std::mt19937& rng, QType q, int rows, int k, std::uint8_t* out) {
    const int blocks = k / static_cast<int>(infernix::ggml_block_elems(q));
    const int size   = static_cast<int>(infernix::ggml_block_bytes(q));
    for (int i = 0; i < rows * blocks * size; ++i) { out[i] = static_cast<std::uint8_t>(rng()); }
    for (int b = 0; b < rows * blocks; ++b) {
        const auto w = static_cast<std::uint16_t>((rng() & 0x8000U) | (rng() & 0x3FFU) | ((5U + rng() % 16U) << 10));
        out[static_cast<std::size_t>(b) * size]     = static_cast<std::uint8_t>(w & 0xFF);
        out[static_cast<std::size_t>(b) * size + 1] = static_cast<std::uint8_t>(w >> 8);
    }
}

void decode_ggml_rows(QType q, const std::uint8_t* rows, int count, int k, double* out) {
    const std::size_t size = infernix::ggml_block_bytes(q), elems = infernix::ggml_block_elems(q);
    const std::size_t row_bytes = k / elems * size;
    for (int r = 0; r < count; ++r) {
        for (std::size_t b = 0; b < k / elems; ++b) {
            infernix::test::ggml_oracle::decode_block(q, rows + r * row_bytes + b * size, out + static_cast<std::size_t>(r) * k + b * elems);
        }
    }
}

Kind ggml_kind(QType gate_up, QType down) {
    Kind kind;
    kind.name = "ggml " + std::to_string(static_cast<int>(gate_up)) + "/" + std::to_string(static_cast<int>(down));
    const auto gate_row = static_cast<std::uint32_t>(H / infernix::ggml_block_elems(gate_up) * infernix::ggml_block_bytes(gate_up));
    const auto down_row = static_cast<std::uint32_t>(I / infernix::ggml_block_elems(down) * infernix::ggml_block_bytes(down));
    kind.record_bytes = 2ULL * I * gate_row + static_cast<std::uint64_t>(H) * down_row;
    kind.ggml         = {.gate_up_format = gate_up,
                         .down_format    = down,
                         .gate_row_bytes = gate_row,
                         .down_row_bytes = down_row,
                         .record_bytes   = (kind.record_bytes + 15) / 16 * 16};
    kind.make = [=](std::mt19937& rng, std::uint8_t* record, moe::ExpertScales& scales) {
        random_ggml_rows(rng, gate_up, 2 * I, H, record);
        random_ggml_rows(rng, down, H, I, record + 2ULL * I * gate_row);
        scales = {};
    };
    kind.decode = [=](const std::uint8_t* record, std::vector<double>& g, std::vector<double>& u, std::vector<double>& d) {
        decode_ggml_rows(gate_up, record, I, H, g.data());
        decode_ggml_rows(gate_up, record + static_cast<std::size_t>(I) * gate_row, I, H, u.data());
        decode_ggml_rows(down, record + 2ULL * I * gate_row, H, I, d.data());
    };
    return kind;
}

// fp8_expert_b128_v1: gate, up, down codes, then the BF16 tile scales of gate, up ([I/128][H/128])
// and down ([H/128][I/128]).
Kind fp8_kind() {
    constexpr std::size_t kGate = static_cast<std::size_t>(I) * H;
    constexpr int kTiles        = (I / 128) * (H / 128);
    Kind kind;
    kind.name         = "fp8 b128";
    kind.record_bytes = 3 * kGate + 3ULL * kTiles * 2;
    kind.ggml         = {.gate_up_format = QType::FP8_E4M3FN_B128,
                         .down_format    = QType::FP8_E4M3FN_B128,
                         .gate_row_bytes = static_cast<std::uint32_t>(H),
                         .down_row_bytes = static_cast<std::uint32_t>(I),
                         .record_bytes   = (kind.record_bytes + 15) / 16 * 16};
    kind.make = [](std::mt19937& rng, std::uint8_t* record, moe::ExpertScales& scales) {
        for (std::size_t i = 0; i < 3 * kGate; ++i) {
            auto c = static_cast<std::uint8_t>(rng());
            if ((c & 0x7F) == 0x7F) { c ^= 0x01; } // no NaN
            record[i] = c;
        }
        auto* s = reinterpret_cast<std::uint16_t*>(record + 3 * kGate);
        for (int t = 0; t < 3 * kTiles; ++t) {
            s[t] = static_cast<std::uint16_t>(((127 - 14 + static_cast<int>(rng() % 8)) << 7) | (rng() & 0x7F));
        }
        scales = {};
    };
    kind.decode = [](const std::uint8_t* record, std::vector<double>& g, std::vector<double>& u, std::vector<double>& d) {
        const auto* s = reinterpret_cast<const std::uint16_t*>(record + 3 * kGate);
        for (int i = 0; i < I; ++i) {
            for (int k = 0; k < H; ++k) {
                const std::size_t tile = static_cast<std::size_t>(i / 128) * (H / 128) + k / 128;
                g[static_cast<std::size_t>(i) * H + k] = e4m3(record[static_cast<std::size_t>(i) * H + k]) * bf16_value(s[tile]);
                u[static_cast<std::size_t>(i) * H + k] =
                    e4m3(record[kGate + static_cast<std::size_t>(i) * H + k]) * bf16_value(s[kTiles + tile]);
            }
        }
        for (int r = 0; r < H; ++r) {
            for (int k = 0; k < I; ++k) {
                const std::size_t tile = static_cast<std::size_t>(r / 128) * (I / 128) + k / 128;
                d[static_cast<std::size_t>(r) * I + k] =
                    e4m3(record[2 * kGate + static_cast<std::size_t>(r) * I + k]) * bf16_value(s[2 * kTiles + tile]);
            }
        }
    };
    return kind;
}

// An exact dot product (double-double) and the route's error bound around it (precise_dot.cuh:
// (4096 + 2 K) u^2 times the terms' magnitudes, plus the binary64 conversion and scaling).
struct Sum {
    double value;
    double bound;
};

Sum exact_dot(const double* w, const double* v, int k) {
    double hi = 0.0, lo = 0.0, magnitude = 0.0;
    for (int j = 0; j < k; ++j) {
        const double p = w[j] * v[j], pe = std::fma(w[j], v[j], -p);
        const double s = hi + p, bb = s - hi, se = (hi - (s - bb)) + (p - bb);
        hi = s;
        lo += se + pe;
        magnitude += std::fabs(p);
    }
    const double value = hi + lo;
    return {value, (4096.0 + 2.0 * k) * std::ldexp(magnitude, -48) + std::ldexp(std::fabs(value), -51)};
}

// BF16(s * a) for every s within the bound, or false where the interval holds a rounding boundary.
bool scaled_bf16(const Sum& s, double a, std::uint16_t& bits) {
    bits = bf16_bits(s.value * a);
    return bf16_bits((s.value - s.bound) * a) == bits && bf16_bits((s.value + s.bound) * a) == bits;
}

// The oracle's output of one expert for one column; determined[r] is false where the output depends
// on a sum within the route's error of a rounding boundary.
void oracle(const std::vector<double>& g, const std::vector<double>& u, const std::vector<double>& d,
            const moe::ExpertScales& scales, bool multipliers, const std::uint16_t* x, std::uint16_t* y,
            std::uint8_t* determined) {
    const double ag = multipliers ? scales.alpha_gate : 1.0, au = multipliers ? scales.alpha_up : 1.0;
    const double ad = multipliers ? scales.alpha_down : 1.0;
    std::vector<double> v(static_cast<std::size_t>(H)), h(static_cast<std::size_t>(I));
    for (int k = 0; k < H; ++k) { v[static_cast<std::size_t>(k)] = bf16_value(x[k]); }
    bool h_determined = true;
    for (int i = 0; i < I; ++i) {
        std::uint16_t gb = 0, ub = 0;
        h_determined = scaled_bf16(exact_dot(&g[static_cast<std::size_t>(i) * H], v.data(), H), ag, gb) && h_determined;
        h_determined = scaled_bf16(exact_dot(&u[static_cast<std::size_t>(i) * H], v.data(), H), au, ub) && h_determined;
        const double gv = bf16_value(gb), uv = bf16_value(ub);
        h[static_cast<std::size_t>(i)] = bf16_value(bf16_bits(gv / (1.0 + std::exp(-gv)) * uv));
    }
    for (int r = 0; r < H; ++r) {
        const bool ok = scaled_bf16(exact_dot(&d[static_cast<std::size_t>(r) * I], h.data(), I), ad, y[r]);
        determined[r] = ok && h_determined ? 1 : 0;
    }
}

void run(const Kind& kind, int columns, std::mt19937& rng) {
    constexpr int experts = 12, top_k = 4;
    const std::uint64_t stride = (kind.record_bytes + 4095) / 4096 * 4096;
    const std::uint64_t pitch  = stride + 8192; // the frame pool's pitch exceeds the record stride
    std::uint8_t* host = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host), stride * experts, cudaHostAllocMapped), "cudaHostAlloc");
    std::memset(host, 0, stride * experts);
    std::vector<moe::ExpertScales> scales(experts);
    for (int e = 0; e < experts; ++e) { kind.make(rng, host + e * stride, scales[static_cast<std::size_t>(e)]); }
    std::uint8_t* host_device = nullptr;
    cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_device), host, 0), "cudaHostGetDevicePointer");
    std::vector<std::int32_t> frames(experts, -1);
    int resident = 0;
    for (int e = 0; e < experts; e += 3) { frames[static_cast<std::size_t>(e)] = resident++; }
    std::uint8_t* d_frames_base = nullptr;
    cuda_check(cudaMalloc(&d_frames_base, pitch * resident), "cudaMalloc");
    for (int e = 0; e < experts; ++e) {
        if (frames[static_cast<std::size_t>(e)] >= 0) {
            cuda_check(cudaMemcpy(d_frames_base + frames[static_cast<std::size_t>(e)] * pitch, host + e * stride, stride,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
        }
    }
    auto* d_frames = device_copy(frames);
    auto* d_scales = device_copy(scales);

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

    const auto x = fixtures::wide_range_activations(rng, columns);
    auto* d_x    = device_copy(x);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(top_k) * columns);
    cuda_check(cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost), "cudaMemcpy");

    // The oracle, one decoded expert at a time (each used expert decoded once).
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(H) * top_k * columns);
    std::vector<std::uint8_t> determined(expected.size());
    const bool multipliers = kind.ggml.gate_up_format == QType::NVFP4_MUL;
    {
        std::atomic<int> next{0};
        const auto work = [&] {
            std::vector<double> g(static_cast<std::size_t>(I) * H), u(g.size()), d(static_cast<std::size_t>(H) * I);
            for (int e = next++; e < experts; e = next++) {
                bool used = false;
                for (const std::int32_t id : ids) { used = used || id == e; }
                if (!used) { continue; }
                kind.decode(host + e * stride, g, u, d);
                for (std::size_t i = 0; i < ids.size(); ++i) {
                    if (ids[i] != e) { continue; }
                    const std::size_t t = i / top_k;
                    oracle(g, u, d, scales[static_cast<std::size_t>(e)], multipliers, &x[t * H], &expected[i * H],
                           &determined[i * H]);
                }
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
                                                                                  moe::ExpertActivation::kA16,
                                                                                  moe::WideA16::kPrecise)),
               "cudaMalloc");
    std::uint16_t* d_out = nullptr;
    cuda_check(cudaMalloc(&d_out, expected.size() * sizeof(std::uint16_t)), "cudaMalloc");
    cudaStream_t fork_stream = nullptr;
    cudaEvent_t fork_events[2]{};
    cuda_check(cudaStreamCreateWithFlags(&fork_stream, cudaStreamNonBlocking), "stream");
    for (auto& e : fork_events) { cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event"); }

    struct Config {
        int slots;
        bool fork;
        const char* name;
    };
    const Config configs[] = {{0, false, "zero-copy"}, {3, false, "3 staging slots"}, {64, false, "64 staging slots"},
                              {64, true, "forked"}};
    std::vector<std::uint16_t> first_actual;
    for (const auto& config : configs) {
        if (config.fork && columns > moe::kMaxColumns) { continue; }
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
        source.wide_a16     = moe::WideA16::kPrecise;
        source.frame_stride = pitch;
        if (kind.ggml.gate_up_format != QType::NVFP4_MUL) { source.ggml = kind.ggml; }
        if (config.fork) {
            source.fork_stream    = fork_stream;
            source.fork_events[0] = fork_events[0];
            source.fork_events[1] = fork_events[1];
        }
        Tensor tx(d_x, DType::BF16, {H, columns});
        Tensor out(d_out, DType::BF16, {H, top_k * columns});
        infernix::ops::moe_experts(tx, dispatch, source, top_k, max_jobs, d_workspace, out, nullptr);
        cuda_check(cudaDeviceSynchronize(), "moe_experts");
        std::vector<std::uint16_t> actual(expected.size());
        cuda_check(cudaMemcpy(actual.data(), d_out, actual.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "cudaMemcpy");
        std::size_t bad = 0, undetermined = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (determined[i] != 0) {
                bad += actual[i] != expected[i];
            } else {
                ++undetermined;
            }
        }
        const std::string where = kind.name + " at " + std::to_string(columns) + " columns, " + config.name;
        check(bad == 0, where + ": " + std::to_string(bad) + " of " + std::to_string(actual.size()) + " outputs differ");
        if (first_actual.empty()) {
            first_actual = actual;
            if (undetermined != 0) {
                std::printf("%s: %zu of %zu outputs depend on a sum within the route's error of a tie\n", where.c_str(),
                            undetermined, actual.size());
            }
        } else {
            std::size_t moved = 0;
            for (std::size_t i = 0; i < actual.size(); ++i) { moved += actual[i] != first_actual[i]; }
            check(moved == 0, where + ": " + std::to_string(moved) + " outputs differ from the " + configs[0].name + " run's");
        }
    }
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
    cudaFree(d_frames_base);
    cudaFreeHost(host);
    cudaStreamDestroy(fork_stream);
    for (auto e : fork_events) { cudaEventDestroy(e); }
}

} // namespace

int main() {
    std::mt19937 rng(20261019);
    const Kind kinds[] = {nvfp4_kind(), ggml_kind(QType::GGML_IQ3_S, QType::GGML_IQ4_NL),
                          ggml_kind(QType::GGML_IQ4_XS, QType::GGML_Q2_0), fp8_kind()};
    for (const Kind& kind : kinds) {
        for (const int columns : {1, 5, 40}) { run(kind, columns, rng); }
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("offloaded_moe precise route: all checks passed\n");
    return 0;
}
