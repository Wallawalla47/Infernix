// hyper_connection_mix against an FP64 oracle (docs/maintainer/qwen3_8-flash-next-design.md §8.3,
// §19.3.3 Phase 1b). The oracle evaluates the closed formula in FP64 from the represented BF16
// residual and norm weight and the Q8 weights decoded independently from their payload:
//
//   Rn = R * (mean R^2 + eps)^-1/2 * (1 + w) per stream; z = W_down Rn; m = SiLU(z[:rank] / S);
//   inject = 2 sigmoid(z[rank:] / S); u = W_up m; x = (1/S) sum_s sigmoid(u[sH + d]) Rn[sH + d].
//
// Cases: Flash-Next's geometry (S = 4, H = 2560, rank 320) with and without injects at T = 1, 4, 5,
// 8, 9, 16 (the fused route) and 17, 64 (the composed route); column invariance of the fused route
// (each column of a 16-column call bit-identical to that column alone, and a 5-column call equal to
// the 16-column call's first five); one graph-captured call bit-identical to the eager call; the
// workspace capacity covers every T. Criteria: the fused route rounds only x to BF16 (FP32
// intermediates), so |x - ref| <= 2^-8 |ref| + 2^-12 rms(ref), and injects to 1e-5 relative + 1e-6; the
// composed route rounds Rn, z, m and u to BF16, so its bound is 2^-5 |ref| + 2^-6 rms(ref) and
// injects to 2^-7 relative + 2^-9.

#include "ninfer/ops/hyper_connection.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace t = ninfer::test;
using ninfer::DType;
using ninfer::Tensor;

constexpr int kS = 4, kH = 2560, kRank = 320, kW = kS * kH;
constexpr float kEps = 1e-6F;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::uint32_t next(std::uint32_t& state) { return state = state * 1664525U + 1013904223U; }
float uniform(std::uint32_t& state) {
    return static_cast<float>(static_cast<std::int32_t>(next(state) >> 8) - (1 << 23)) / static_cast<float>(1 << 23);
}

struct Device {
    void* p = nullptr;
    explicit Device(std::size_t bytes) { t::cuda_check(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~Device() { cudaFree(p); }
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
};

struct Reference {
    std::vector<double> x;      // [T][H]
    std::vector<double> inject; // [T][S]
};

Reference oracle(const std::vector<std::uint16_t>& r, const std::vector<std::uint16_t>& w, const std::vector<float>& down,
                 int down_rows, const std::vector<float>& up, int T) {
    Reference out;
    out.x.assign(static_cast<std::size_t>(T) * kH, 0.0);
    out.inject.assign(static_cast<std::size_t>(T) * kS, 0.0);
    std::vector<double> rn(kW), z(down_rows), m(kRank);
    for (int c = 0; c < T; ++c) {
        for (int s = 0; s < kS; ++s) {
            double ss = 0;
            for (int d = 0; d < kH; ++d) {
                const double v = t::bf16_to_f32(r[static_cast<std::size_t>(c) * kW + s * kH + d]);
                ss += v * v;
            }
            const double inv = 1.0 / std::sqrt(ss / kH + kEps);
            for (int d = 0; d < kH; ++d) {
                const std::size_t i = static_cast<std::size_t>(s) * kH + d;
                rn[i] = t::bf16_to_f32(r[static_cast<std::size_t>(c) * kW + i]) * inv * (1.0 + t::bf16_to_f32(w[i]));
            }
        }
        for (int k = 0; k < down_rows; ++k) {
            double acc = 0;
            for (int i = 0; i < kW; ++i) { acc += static_cast<double>(down[static_cast<std::size_t>(k) * kW + i]) * rn[i]; }
            z[k] = acc / kS;
        }
        for (int k = 0; k < kRank; ++k) { m[k] = z[k] / (1.0 + std::exp(-z[k])); }
        for (int k = kRank; k < down_rows; ++k) { out.inject[static_cast<std::size_t>(c) * kS + (k - kRank)] = 2.0 / (1.0 + std::exp(-z[k])); }
        for (int d = 0; d < kH; ++d) {
            double sum = 0;
            for (int s = 0; s < kS; ++s) {
                const std::size_t row = static_cast<std::size_t>(s) * kH + d;
                double u = 0;
                for (int k = 0; k < kRank; ++k) { u += static_cast<double>(up[row * kRank + k]) * m[k]; }
                sum += rn[row] / (1.0 + std::exp(-u));
            }
            out.x[static_cast<std::size_t>(c) * kH + d] = sum / kS;
        }
    }
    return out;
}

// The weight's values decoded independently from its payload: int8 code times its group's FP16
// scale (codes [n][padded_k], then FP16 scales [n][padded_k / 32]).
float half_to_float(std::uint16_t h) {
    const int sign = (h >> 15) & 1, exponent = (h >> 10) & 31, mantissa = h & 1023;
    double value;
    if (exponent == 0) {
        value = std::ldexp(static_cast<double>(mantissa), -24);
    } else if (exponent == 31) {
        value = mantissa == 0 ? INFINITY : NAN;
    } else {
        value = std::ldexp(static_cast<double>(mantissa + 1024), exponent - 25);
    }
    return static_cast<float>(sign ? -value : value);
}

std::vector<float> decode_q8(const ninfer::test::quantized_weight::PackedWeight& w) {
    const int n = w.weight.n, k = w.weight.k, padded = w.weight.padded_shape[1];
    std::vector<float> out(static_cast<std::size_t>(n) * k);
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < k; ++c) {
            const std::size_t scale_at =
                static_cast<std::size_t>(w.scale_plane_offset) + (static_cast<std::size_t>(r) * (padded / 32) + c / 32) * 2;
            const auto bits = static_cast<std::uint16_t>(w.payload[scale_at] | (w.payload[scale_at + 1] << 8));
            const auto code = static_cast<std::int8_t>(w.payload[static_cast<std::size_t>(r) * padded + c]);
            out[static_cast<std::size_t>(r) * k + c] = static_cast<float>(code) * half_to_float(bits);
        }
    }
    return out;
}

// Random weights of the model's scale written into the fixture's payload layout: codes uniform in
// [-127, 127] (0 in the padding), each group's FP16 scale g / (127 sqrt(K)) with g in [2, 6), so
// z / S and u stay of order one as in the model (the fixture's patterned codes are structured and
// push the gates into saturation, where any intermediate rounding flips them).
std::uint16_t half_bits(double value) {
    int exponent = 0;
    const double mantissa = std::frexp(value, &exponent); // value = mantissa * 2^exponent, 0.5 <= mantissa < 1
    const int e = exponent - 1;                           // value = (2 * mantissa) * 2^e
    const auto fraction = static_cast<int>(std::lround((2.0 * mantissa - 1.0) * 1024.0));
    if (e < -14 || e > 15) { throw std::logic_error("scale outside FP16's normal range"); }
    return static_cast<std::uint16_t>(((e + 15) << 10) | std::min(fraction, 1023));
}

void randomize_q8(ninfer::test::quantized_weight::PackedWeight& w, std::uint32_t seed) {
    const int n = w.weight.n, k = w.weight.k, padded = w.weight.padded_shape[1];
    std::uint32_t state = seed;
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < padded; ++c) {
            const auto code = c < k ? static_cast<std::int32_t>(next(state) % 255U) - 127 : 0;
            w.payload[static_cast<std::size_t>(r) * padded + c] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
        }
        for (int g = 0; g < padded / 32; ++g) {
            const double gain  = 2.0 + 4.0 * static_cast<double>(next(state) >> 8) / static_cast<double>(1 << 24);
            const auto bits    = half_bits(gain / (127.0 * std::sqrt(static_cast<double>(k))));
            const std::size_t at = static_cast<std::size_t>(w.scale_plane_offset) +
                                   (static_cast<std::size_t>(r) * (padded / 32) + g) * 2;
            w.payload[at]     = static_cast<std::uint8_t>(bits & 0xFFU);
            w.payload[at + 1] = static_cast<std::uint8_t>(bits >> 8);
        }
    }
}

struct Problem {
    int down_rows = 0;
    ninfer::test::quantized_weight::PackedWeight down, up;
    std::vector<float> down_values, up_values; // decoded independently
    std::unique_ptr<Device> down_payload, up_payload;
    ninfer::Weight down_w, up_w;
    std::vector<std::uint16_t> norm;
    std::unique_ptr<Device> norm_device;
};

Problem make_problem(bool inject) {
    Problem p;
    p.down_rows    = kRank + (inject ? kS : 0);
    p.down         = t::linear::make_q8_g32_fp16_weight(p.down_rows, kW, inject ? 501U : 503U);
    p.up           = t::linear::make_q8_g32_fp16_weight(kW, kRank, 509U);
    randomize_q8(p.down, inject ? 601U : 603U);
    randomize_q8(p.up, 607U);
    p.down_values  = decode_q8(p.down);
    p.up_values    = decode_q8(p.up);
    p.down_payload = std::make_unique<Device>(p.down.payload.size());
    p.up_payload   = std::make_unique<Device>(p.up.payload.size());
    t::cuda_check(cudaMemcpy(p.down_payload->p, p.down.payload.data(), p.down.payload.size(), cudaMemcpyHostToDevice), "down");
    t::cuda_check(cudaMemcpy(p.up_payload->p, p.up.payload.data(), p.up.payload.size(), cudaMemcpyHostToDevice), "up");
    p.down_w = p.down.device_weight(p.down_payload->p);
    p.up_w   = p.up.device_weight(p.up_payload->p);
    std::uint32_t state = 77U;
    p.norm.resize(kW);
    for (auto& v : p.norm) { v = t::f32_to_bf16(0.25F * uniform(state)); }
    p.norm_device = std::make_unique<Device>(kW * 2);
    t::cuda_check(cudaMemcpy(p.norm_device->p, p.norm.data(), kW * 2, cudaMemcpyHostToDevice), "norm");
    return p;
}

std::vector<std::uint16_t> residual(int T, std::uint32_t seed) {
    std::vector<std::uint16_t> r(static_cast<std::size_t>(T) * kW);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < r.size(); ++i) {
        // Streams at different scales, so the per-stream norms matter.
        const int s = static_cast<int>((i % kW) / kH);
        r[i]        = t::f32_to_bf16(uniform(state) * static_cast<float>(1 << s));
    }
    return r;
}

struct Result {
    std::vector<std::uint16_t> x;
    std::vector<float> inject;
};

Result run(Problem& p, const std::vector<std::uint16_t>& r, int T, bool inject, ninfer::WorkspaceArena& ws,
           cudaStream_t stream, bool graph = false) {
    Device rd(r.size() * 2), xd(static_cast<std::size_t>(kH) * T * 2), id(static_cast<std::size_t>(kS) * T * 4);
    t::cuda_check(cudaMemcpy(rd.p, r.data(), r.size() * 2, cudaMemcpyHostToDevice), "residual");
    const Tensor R(rd.p, DType::BF16, {kW, T});
    const Tensor w(p.norm_device->p, DType::BF16, {kW});
    Tensor x(xd.p, DType::BF16, {kH, T});
    Tensor inj(id.p, DType::FP32, {kS, T});
    const auto call = [&] {
        ninfer::ops::hyper_connection_mix(R, w, p.down_w, p.up_w, ninfer::ops::LinearPolicy::A16Only, kS, kRank, kEps,
                                          x, inject ? &inj : nullptr, ws, stream);
    };
    if (graph) {
        cudaGraph_t g = nullptr;
        cudaGraphExec_t exec = nullptr;
        t::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        call();
        t::cuda_check(cudaStreamEndCapture(stream, &g), "end capture");
        t::cuda_check(cudaGraphInstantiate(&exec, g, 0), "instantiate");
        t::cuda_check(cudaGraphLaunch(exec, stream), "launch");
        t::cuda_check(cudaStreamSynchronize(stream), "sync");
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(g);
    } else {
        call();
        t::cuda_check(cudaStreamSynchronize(stream), "sync");
    }
    Result out;
    out.x.resize(static_cast<std::size_t>(kH) * T);
    t::cuda_check(cudaMemcpy(out.x.data(), xd.p, out.x.size() * 2, cudaMemcpyDeviceToHost), "x");
    if (inject) {
        out.inject.resize(static_cast<std::size_t>(kS) * T);
        t::cuda_check(cudaMemcpy(out.inject.data(), id.p, out.inject.size() * 4, cudaMemcpyDeviceToHost), "inject");
    }
    return out;
}

void compare(const Result& got, const Reference& ref, bool fused, bool inject, const std::string& label) {
    double ss = 0;
    for (const double v : ref.x) { ss += v * v; }
    const double rms = std::sqrt(ss / static_cast<double>(ref.x.size()));
    const double rel = fused ? std::ldexp(1.0, -8) : std::ldexp(1.0, -5);
    // The composed route rounds Rn, z, m and u to BF16; where the S streams cancel, x is small against
    // their sizes, hence its absolute floor (first set to 2^-7 rms: 4 of 163,840 outputs at T = 64 reached
    // 1.15 of that bound).
    const double abs = rms * (fused ? std::ldexp(1.0, -12) : std::ldexp(1.0, -6));
    double worst = 0;
    std::size_t bad = 0;
    for (std::size_t i = 0; i < ref.x.size(); ++i) {
        const double e = std::fabs(static_cast<double>(t::bf16_to_f32(got.x[i])) - ref.x[i]);
        worst          = std::max(worst, e / (rel * std::fabs(ref.x[i]) + abs));
        bad += e > rel * std::fabs(ref.x[i]) + abs ? 1U : 0U;
    }
    check(bad == 0, label + ": x within its bound (" + std::to_string(bad) + " outside, worst " + std::to_string(worst) +
                        " of the bound)");
    if (inject) {
        // inject lies in (0, 2): a relative bound plus an absolute floor where it saturates toward 0.
        const double irel = fused ? 1e-5 : std::ldexp(1.0, -7);
        const double iabs = fused ? 1e-6 : std::ldexp(1.0, -9);
        double iworst     = 0;
        for (std::size_t i = 0; i < ref.inject.size(); ++i) {
            iworst = std::max(iworst, std::fabs(got.inject[i] - ref.inject[i]) / (irel * std::fabs(ref.inject[i]) + iabs));
        }
        check(iworst <= 1.0, label + ": inject within its bound (worst " + std::to_string(iworst) + " of the bound)");
    }
}

} // namespace

int main() {
    try {
        if (!t::linear::cuda_available()) {
            std::printf("SKIP: no CUDA device\n");
            return 77;
        }
        cudaStream_t stream = nullptr;
        t::cuda_check(cudaStreamCreate(&stream), "stream");
        for (const bool inject : {true, false}) {
            Problem p = make_problem(inject);
            const std::size_t capacity = ninfer::ops::hyper_connection_mix_workspace_capacity_bytes(
                p.down_w, p.up_w, ninfer::ops::LinearPolicy::A16Only, kS, kRank, 64);
            ninfer::WorkspaceArena ws(capacity);
            const std::string tag = inject ? "inject" : "no inject";
            for (const int T : {1, 4, 5, 8, 9, 16, 17, 64}) {
                const auto r     = residual(T, 1000U + static_cast<std::uint32_t>(T));
                const Result got = run(p, r, T, inject, ws, stream);
                compare(got, oracle(r, p.norm, p.down_values, p.down_rows, p.up_values, T), T <= 16, inject,
                        tag + " T=" + std::to_string(T));
                check(ws.peak_used() <= capacity, tag + ": the workspace capacity covers T=" + std::to_string(T));
            }
            // Column invariance of the fused route.
            const auto r16    = residual(16, 4242U);
            const Result all  = run(p, r16, 16, inject, ws, stream);
            bool same         = true;
            for (int c = 0; c < 16 && same; ++c) {
                const std::vector<std::uint16_t> one(r16.begin() + static_cast<std::ptrdiff_t>(c) * kW,
                                                     r16.begin() + static_cast<std::ptrdiff_t>(c + 1) * kW);
                const Result alone = run(p, one, 1, inject, ws, stream);
                same = std::equal(alone.x.begin(), alone.x.end(), all.x.begin() + static_cast<std::ptrdiff_t>(c) * kH) &&
                       (!inject || std::equal(alone.inject.begin(), alone.inject.end(),
                                              all.inject.begin() + static_cast<std::ptrdiff_t>(c) * kS));
            }
            check(same, tag + ": each column of a 16-column call equals that column alone");
            const std::vector<std::uint16_t> r5(r16.begin(), r16.begin() + 5 * kW);
            const Result five = run(p, r5, 5, inject, ws, stream);
            check(std::equal(five.x.begin(), five.x.end(), all.x.begin()), tag + ": a 5-column call equals the first five");
            const Result captured = run(p, r5, 5, inject, ws, stream, true);
            check(captured.x == five.x && captured.inject == five.inject, tag + ": a graph-captured call equals the eager call");
        }
        // Shapes that disagree are refused.
        {
            Problem p = make_problem(true);
            ninfer::WorkspaceArena ws(1 << 20);
            Device rd(static_cast<std::size_t>(kW) * 2), xd(kH * 2);
            const Tensor R(rd.p, DType::BF16, {kW, 1});
            const Tensor w(p.norm_device->p, DType::BF16, {kW});
            Tensor x(xd.p, DType::BF16, {kH, 1});
            bool threw = false;
            try {
                // The down projection carries injection rows the call does not ask for.
                ninfer::ops::hyper_connection_mix(R, w, p.down_w, p.up_w, ninfer::ops::LinearPolicy::A16Only, kS, kRank,
                                                  kEps, x, nullptr, ws, stream);
            } catch (const std::invalid_argument&) { threw = true; }
            check(threw, "a down projection with unused injection rows is refused");
        }
        cudaStreamDestroy(stream);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ++failures;
    }
    std::printf(failures == 0 ? "hyper_connection checks passed\n" : "FAIL: %d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
