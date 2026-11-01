// hyper_connection_mix against an FP64 oracle (docs/maintainer/qwen3_8-flash-next-design.md §8.3,
// §19.3.3 Phase 1b), for Q8_G32_FP16 weights (Dense8) and BF16 weights (the bit-exact artifact), each
// with a BF16 norm weight and with a full-precision FP32 one (NVIDIA FP4-Local). The oracle evaluates
// the closed formula in FP64 from the represented BF16 residual and norm weight and
// the weights decoded independently from their payload:
//
//   Rn = R * (mean R^2 + eps)^-1/2 * (1 + w) per stream; z = W_down Rn; m = SiLU(z[:rank] / S);
//   inject = 2 sigmoid(z[rank:] / S); u = W_up m; x = (1/S) sum_s sigmoid(u[sH + d]) Rn[sH + d].
//
// Cases: Flash-Next's geometry (S = 4, H = 2560, rank 320) with and without injects at T = 1, 4, 5,
// 8, 9, 16 (the fused route) and 17, 64, 96, 97, 129, 193 (the composed route; with Q8 weights and
// more than 96 columns its up projection and collapse run as one kernel, whose tile changes at 129
// columns and leaves partial token tiles at 97, 129 and 193); column invariance of the fused route
// (each column of a 16-column call bit-identical to that column alone, and a 5-column call equal to
// the 16-column call's first five); one graph-captured call bit-identical to the eager call; the
// workspace capacity covers every T; hyper_connection_repack_q8 (--a8-activations) rewriting the BF16 problem's
// matrices as Q8 (every code and scale exactly its rule), then the mix on them against the oracle over the
// repacked values. Criteria: the fused route rounds only x to BF16 (FP32
// intermediates), so |x - ref| <= 2^-8 |ref| + 2^-12 rms(ref), and injects to 1e-5 relative + 1e-6; the
// composed route rounds Rn, z, m and u to BF16, so its bound is 2^-5 |ref| + 2^-6 rms(ref) and
// injects to 2^-7 relative + 2^-9.
//
// The up projection and collapse of more than 96 columns run as one kernel for a Q8 W_up; it claims the
// bits of the separate linear and collapse, so it is compared with them exactly at T = 97, 128, 129, 192,
// 256, 300 and 1025 (each tile width, full and partial token tiles). A BF16 W_up takes the separate steps.

#include "infernix/ops/hyper_connection.h"
#include "ops/ggml/ggml_a8.h"
#include "ops/op_family.h"
#include "ops/hyper_connection/hyper_connection_up_collapse.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace t = infernix::test;
using infernix::DType;
using infernix::Tensor;

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

// `w`: the norm weight's represented values (BF16 widened, or FP32).
Reference oracle(const std::vector<std::uint16_t>& r, const std::vector<double>& w, const std::vector<float>& down,
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
                rn[i] = t::bf16_to_f32(r[static_cast<std::size_t>(c) * kW + i]) * inv * (1.0 + w[i]);
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

std::vector<float> decode_q8(const infernix::test::quantized_weight::PackedWeight& w) {
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

void randomize_q8(infernix::test::quantized_weight::PackedWeight& w, std::uint32_t seed) {
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

// BF16 weights of the same scale as randomize_q8's: per 32-element group a gain g in [2, 6), values
// uniform in (-1, 1) * g / sqrt(K), rounded to BF16 (the oracle reads the rounded values).
void randomize_bf16(infernix::test::quantized_weight::PackedWeight& w, std::uint32_t seed) {
    const int n = w.weight.n, k = w.weight.k;
    std::uint32_t state = seed;
    for (int r = 0; r < n; ++r) {
        double gain = 0;
        for (int c = 0; c < k; ++c) {
            if (c % 32 == 0) { gain = 2.0 + 4.0 * static_cast<double>(next(state) >> 8) / static_cast<double>(1 << 24); }
            const auto bits = t::f32_to_bf16(static_cast<float>(uniform(state) * gain / std::sqrt(static_cast<double>(k))));
            const std::size_t at = (static_cast<std::size_t>(r) * k + c) * 2;
            w.payload[at]        = static_cast<std::uint8_t>(bits & 0xFFU);
            w.payload[at + 1]    = static_cast<std::uint8_t>(bits >> 8);
        }
    }
}

std::vector<float> decode_bf16(const infernix::test::quantized_weight::PackedWeight& w) {
    std::vector<float> out(static_cast<std::size_t>(w.weight.n) * w.weight.k);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = t::bf16_to_f32(static_cast<std::uint16_t>(w.payload[2 * i] | (w.payload[2 * i + 1] << 8)));
    }
    return out;
}

enum class Codec { Q8, Bf16 };

struct Problem {
    int down_rows = 0;
    infernix::test::quantized_weight::PackedWeight down, up;
    std::vector<float> down_values, up_values; // decoded independently
    std::unique_ptr<Device> down_payload, up_payload;
    infernix::Weight down_w, up_w;
    std::vector<std::uint16_t> norm;
    std::unique_ptr<Device> norm_device;
    // An FP32 norm weight of full precision (NVIDIA FP4-Local's trained norms).
    std::vector<float> norm32;
    std::unique_ptr<Device> norm32_device;

    std::vector<double> norm_values(bool fp32) const {
        std::vector<double> out(kW);
        for (int i = 0; i < kW; ++i) { out[i] = fp32 ? norm32[i] : t::bf16_to_f32(norm[i]); }
        return out;
    }
};

Problem make_problem(bool inject, Codec codec) {
    Problem p;
    p.down_rows = kRank + (inject ? kS : 0);
    if (codec == Codec::Q8) {
        p.down = t::linear::make_q8_g32_fp16_weight(p.down_rows, kW, inject ? 501U : 503U);
        p.up   = t::linear::make_q8_g32_fp16_weight(kW, kRank, 509U);
        randomize_q8(p.down, inject ? 601U : 603U);
        randomize_q8(p.up, 607U);
        p.down_values = decode_q8(p.down);
        p.up_values   = decode_q8(p.up);
    } else {
        p.down = t::linear::make_bf16_weight(p.down_rows, kW, inject ? 511U : 513U);
        p.up   = t::linear::make_bf16_weight(kW, kRank, 519U);
        randomize_bf16(p.down, inject ? 611U : 613U);
        randomize_bf16(p.up, 617U);
        p.down_values = decode_bf16(p.down);
        p.up_values   = decode_bf16(p.up);
    }
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
    p.norm32.resize(kW);
    for (auto& v : p.norm32) { v = 0.25F * uniform(state) + 0x1p-20F * uniform(state); } // below BF16's resolution
    p.norm32_device = std::make_unique<Device>(kW * 4);
    t::cuda_check(cudaMemcpy(p.norm32_device->p, p.norm32.data(), kW * 4, cudaMemcpyHostToDevice), "norm32");
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

Result run(Problem& p, const std::vector<std::uint16_t>& r, int T, bool inject, infernix::WorkspaceArena& ws,
           cudaStream_t stream, bool graph = false, bool fp32_norm = false) {
    Device rd(r.size() * 2), xd(static_cast<std::size_t>(kH) * T * 2), id(static_cast<std::size_t>(kS) * T * 4);
    t::cuda_check(cudaMemcpy(rd.p, r.data(), r.size() * 2, cudaMemcpyHostToDevice), "residual");
    const Tensor R(rd.p, DType::BF16, {kW, T});
    const Tensor w = fp32_norm ? Tensor(p.norm32_device->p, DType::FP32, {kW}) : Tensor(p.norm_device->p, DType::BF16, {kW});
    Tensor x(xd.p, DType::BF16, {kH, T});
    Tensor inj(id.p, DType::FP32, {kS, T});
    const auto call = [&] {
        infernix::ops::hyper_connection_mix(R, w, p.down_w, p.up_w, infernix::ops::LinearPolicy::A16Only, kS, kRank, kEps,
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

// hyper_connection_repack_q8 (--a8-activations): a BF16 matrix rewritten as Q8 in its own bytes. Every code and
// scale is compared exactly with the rule evaluated here (s = fl16(fl32(m / 127)), m the group's largest magnitude;
// q = rne(fl32(v / s)) clamped to [-127, 127]; a zero group and the padding past K: 0), and the repacked values
// decoded from the read-back bytes are returned for the oracle.
std::vector<float> repack_and_check(infernix::Weight& w, const std::vector<float>& values, cudaStream_t stream,
                                    const std::string& what) {
    const int n = w.n, k = w.k;
    const std::size_t scratch = infernix::ops::hyper_connection_repack_q8_scratch_bytes(w);
    check(scratch > 0, what + ": repackable");
    if (scratch == 0) { return values; }
    Device s(scratch);
    infernix::ops::hyper_connection_repack_q8(w, s.p, stream);
    t::cuda_check(cudaStreamSynchronize(stream), "repack");
    const int padded = w.padded_shape[1];
    check(w.qtype == infernix::QType::Q8_G32_FP16 && w.layout == infernix::QuantLayout::RowSplit && padded % 128 == 0 &&
              padded >= k && w.scales == static_cast<const std::uint8_t*>(w.qdata) + static_cast<std::size_t>(n) * padded,
          what + ": a Q8_G32_FP16 RowSplit Weight with its scales after the codes");
    const int groups = padded / 32;
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(n) * padded + static_cast<std::size_t>(n) * groups * 2);
    t::cuda_check(cudaMemcpy(bytes.data(), w.qdata, bytes.size(), cudaMemcpyDeviceToHost), "read back");
    std::vector<float> decoded(static_cast<std::size_t>(n) * k);
    std::size_t bad = 0;
    for (int r = 0; r < n; ++r) {
        for (int g = 0; g < groups; ++g) {
            float m = 0.0F;
            for (int j = 0; j < 32; ++j) {
                const int c = 32 * g + j;
                if (c < k) { m = std::max(m, std::fabs(values[static_cast<std::size_t>(r) * k + c])); }
            }
            const std::uint16_t s16 = __half_as_ushort(__float2half_rn(m / 127.0F));
            const float scale       = half_to_float(s16);
            const std::size_t at    = static_cast<std::size_t>(n) * padded + (static_cast<std::size_t>(r) * groups + g) * 2;
            const auto stored       = static_cast<std::uint16_t>(bytes[at] | (bytes[at + 1] << 8));
            bad += stored != (scale > 0.0F ? s16 : std::uint16_t{0}) ? 1U : 0U;
            for (int j = 0; j < 32; ++j) {
                const int c = 32 * g + j;
                int q       = 0;
                if (c < k && scale > 0.0F) {
                    q = static_cast<int>(std::nearbyint(values[static_cast<std::size_t>(r) * k + c] / scale));
                    q = std::clamp(q, -127, 127);
                }
                const auto code = static_cast<std::int8_t>(bytes[static_cast<std::size_t>(r) * padded + c]);
                bad += code != q ? 1U : 0U;
                if (c < k) { decoded[static_cast<std::size_t>(r) * k + c] = static_cast<float>(q) * scale; }
            }
        }
    }
    check(bad == 0, what + ": every code and scale is the rule's (" + std::to_string(bad) + " differ)");
    return decoded;
}

// A BF16 problem's matrices repacked, then the mix on them (its Q8 routes, fused and composed) against the FP64
// oracle over the repacked values, with the Q8 problem's bounds.
void repack_q8(bool inject, cudaStream_t stream) {
    Problem p = make_problem(inject, Codec::Bf16);
    const std::string tag = std::string("repacked q8 ") + (inject ? "inject" : "no inject");
    const std::vector<float> down = repack_and_check(p.down_w, p.down_values, stream, tag + " down");
    const std::vector<float> up   = repack_and_check(p.up_w, p.up_values, stream, tag + " up");
    const std::size_t capacity    = infernix::ops::hyper_connection_mix_workspace_capacity_bytes(
        p.down_w, p.up_w, infernix::ops::LinearPolicy::A16Only, kS, kRank, 129);
    infernix::WorkspaceArena ws(capacity);
    const std::vector<double> norm = p.norm_values(false);
    for (const int T : {1, 8, 16, 17, 97, 129}) {
        const auto r = residual(T, 3000U + static_cast<std::uint32_t>(T));
        compare(run(p, r, T, inject, ws, stream), oracle(r, norm, down, p.down_rows, up, T), T <= 16, inject,
                tag + " T=" + std::to_string(T));
    }
}

// The fused up + collapse against the separate linear (A16Only) and collapse, bit for bit. m is of a
// SiLU output's range and normalized of a normed residual's.
void up_collapse_bit_identity(Codec codec, cudaStream_t stream) {
    Problem p                = make_problem(false, codec);
    const std::string tag    = codec == Codec::Q8 ? "q8 up+collapse" : "bf16 up+collapse";
    constexpr int kMaxTokens = 1025;
    infernix::WorkspaceArena ws(std::max<std::size_t>(infernix::ops::linear_workspace_capacity_bytes(
        p.up_w.qtype, p.up_w.n, p.up_w.k, infernix::ops::LinearPolicy::A16Only, 1, kMaxTokens), 256));
    for (const int T : {97, 128, 129, 192, 256, 300, kMaxTokens}) {
        const std::string label = tag + " T=" + std::to_string(T);
        if (!infernix::ops::detail::hc_up_collapse_supported(p.up_w, kH, kS, T)) {
            check(false, label + ": the fused route serves this call");
            continue;
        }
        std::uint32_t state = 9000U + static_cast<std::uint32_t>(T);
        std::vector<std::uint16_t> m(static_cast<std::size_t>(kRank) * T), normalized(static_cast<std::size_t>(kW) * T);
        for (auto& v : m) { v = t::f32_to_bf16(0.6F + 0.9F * uniform(state)); }
        for (auto& v : normalized) { v = t::f32_to_bf16(2.0F * uniform(state)); }
        Device md(m.size() * 2), nd(normalized.size() * 2), ud(static_cast<std::size_t>(kW) * T * 2),
            composed_d(static_cast<std::size_t>(kH) * T * 2), fused_d(static_cast<std::size_t>(kH) * T * 2);
        t::cuda_check(cudaMemcpy(md.p, m.data(), m.size() * 2, cudaMemcpyHostToDevice), "m");
        t::cuda_check(cudaMemcpy(nd.p, normalized.data(), normalized.size() * 2, cudaMemcpyHostToDevice), "normalized");
        const Tensor mt(md.p, DType::BF16, {kRank, T}), nt(nd.p, DType::BF16, {kW, T});
        Tensor u(ud.p, DType::BF16, {kW, T}), composed(composed_d.p, DType::BF16, {kH, T}),
            fused(fused_d.p, DType::BF16, {kH, T});
        {
            auto scope = ws.scope();
            infernix::ops::linear(mt, p.up_w, u, infernix::ops::LinearPolicy::A16Only, ws, stream);
        }
        infernix::ops::detail::hc_collapse(u, nt, kS, composed, stream);
        infernix::ops::detail::hc_up_collapse(mt, p.up_w, nt, kS, fused, stream);
        t::cuda_check(cudaStreamSynchronize(stream), "sync");
        std::vector<std::uint16_t> a(static_cast<std::size_t>(kH) * T), b(a.size());
        t::cuda_check(cudaMemcpy(a.data(), composed_d.p, a.size() * 2, cudaMemcpyDeviceToHost), "composed");
        t::cuda_check(cudaMemcpy(b.data(), fused_d.p, b.size() * 2, cudaMemcpyDeviceToHost), "fused");
        std::size_t differ = 0;
        for (std::size_t i = 0; i < a.size(); ++i) { differ += a[i] != b[i] ? 1U : 0U; }
        check(differ == 0, label + ": the fused kernel's bits equal the linear + collapse (" + std::to_string(differ) +
                               " of " + std::to_string(a.size()) + " differ)");
    }
}

// hyper_connection_mix_a8_encoded (--a8-activations): x and inject bit for bit hyper_connection_mix's, and the 8-bit
// columns word for word ops/ggml/ggml_a8.h's codec over that x (what ggml_linear's 8-bit route encodes), with and
// without slice sums, at the fused route's two-to-sixteen-column widths; a one-column call is declined.
void a8_encoded(Codec codec, cudaStream_t stream) {
    namespace a8 = infernix::ggml::a8;
    Problem p = make_problem(true, codec);
    const std::size_t capacity = infernix::ops::hyper_connection_mix_workspace_capacity_bytes(
        p.down_w, p.up_w, infernix::ops::LinearPolicy::A16Only, kS, kRank, 16);
    infernix::WorkspaceArena ws(capacity);
    const Tensor w(p.norm_device->p, DType::BF16, {kW});
    for (const bool sums : {false, true}) {
        const int stride = static_cast<int>((a8::column_words(kH, sums) + 3) / 4 * 4);
        for (const int T : {1, 2, 3, 4, 5, 8, 16}) {
            const std::string tag = std::string(codec == Codec::Q8 ? "q8" : "bf16") + " a8 copy" + (sums ? " sums" : "") +
                                    " T=" + std::to_string(T);
            const auto r      = residual(T, 5000U + static_cast<std::uint32_t>(T));
            const Result want = run(p, r, T, true, ws, stream);
            Device rd(r.size() * 2), xd(static_cast<std::size_t>(kH) * T * 2), id(static_cast<std::size_t>(kS) * T * 4),
                ed(static_cast<std::size_t>(stride) * T * 4);
            t::cuda_check(cudaMemcpy(rd.p, r.data(), r.size() * 2, cudaMemcpyHostToDevice), "residual");
            const Tensor R(rd.p, DType::BF16, {kW, T});
            Tensor x(xd.p, DType::BF16, {kH, T}), inj(id.p, DType::FP32, {kS, T}), encoded(ed.p, DType::I32, {stride, T});
            const bool served = infernix::ops::hyper_connection_mix_a8_encoded(R, w, p.down_w, p.up_w, kS, kRank, kEps, x,
                                                                               &inj, encoded, sums, ws, stream);
            t::cuda_check(cudaStreamSynchronize(stream), "a8 mix");
            if (T == 1) {
                check(!served, tag + ": a one-column call is declined");
                continue;
            }
            check(served, tag + ": served");
            Result got;
            got.x.resize(static_cast<std::size_t>(kH) * T);
            got.inject.resize(static_cast<std::size_t>(kS) * T);
            std::vector<std::int32_t> words(static_cast<std::size_t>(stride) * T);
            t::cuda_check(cudaMemcpy(got.x.data(), xd.p, got.x.size() * 2, cudaMemcpyDeviceToHost), "x");
            t::cuda_check(cudaMemcpy(got.inject.data(), id.p, got.inject.size() * 4, cudaMemcpyDeviceToHost), "inject");
            t::cuda_check(cudaMemcpy(words.data(), ed.p, words.size() * 4, cudaMemcpyDeviceToHost), "words");
            check(got.x == want.x && got.inject == want.inject, tag + ": x and inject are hyper_connection_mix's bits");
            std::size_t bad = 0;
            for (int c = 0; c < T; ++c) {
                const std::int32_t* column = words.data() + static_cast<std::size_t>(c) * stride;
                for (int s = 0; s < kH / 32; ++s) {
                    const std::uint16_t* v = want.x.data() + static_cast<std::size_t>(c) * kH + 32 * s;
                    unsigned key           = 0;
                    for (int j = 0; j < 32; ++j) { key = std::max(key, static_cast<unsigned>(v[j] & 0x7FFFU)); }
                    const a8::SliceScale sc = a8::slice_scale(key);
                    int sum                 = 0;
                    for (int g = 0; g < 8; ++g) {
                        std::uint32_t word = 0;
                        for (int j = 0; j < 4; ++j) {
                            const int q = a8::code(v[4 * g + j], sc.inv);
                            sum += q;
                            word |= (static_cast<std::uint32_t>(q) & 0xFFU) << (8 * j);
                        }
                        bad += static_cast<std::uint32_t>(column[s * a8::kSliceWords + g]) != word ? 1U : 0U;
                    }
                    float d = 0.0F;
                    std::memcpy(&d, column + a8::scales_offset(kH) + s, 4);
                    bad += !(d == sc.d || (std::isnan(d) && std::isnan(sc.d))) ? 1U : 0U;
                    if (sums) { bad += column[a8::sums_offset(kH) + s] != sum ? 1U : 0U; }
                }
            }
            check(bad == 0, tag + ": every word is the codec's (" + std::to_string(bad) + " differ)");
        }
    }
}

} // namespace

INFERNIX_OP_TEST(hyper_connection) {
    try {
        if (!t::linear::cuda_available()) {
            std::printf("SKIP: no CUDA device\n");
            return 77;
        }
        cudaStream_t stream = nullptr;
        t::cuda_check(cudaStreamCreate(&stream), "stream");
        // Both codecs, with a BF16 norm weight and with an FP32 one (NVIDIA FP4-Local's).
        for (const Codec codec : {Codec::Q8, Codec::Bf16})
        for (const bool fp32_norm : {false, true})
        for (const bool inject : {true, false}) {
            Problem p = make_problem(inject, codec);
            const std::size_t capacity = infernix::ops::hyper_connection_mix_workspace_capacity_bytes(
                p.down_w, p.up_w, infernix::ops::LinearPolicy::A16Only, kS, kRank, 193);
            infernix::WorkspaceArena ws(capacity);
            const std::string tag = std::string(codec == Codec::Q8 ? "q8 " : "bf16 ") +
                                    (fp32_norm ? "fp32-norm " : "") + (inject ? "inject" : "no inject");
            const std::vector<double> norm = p.norm_values(fp32_norm);
            for (const int T : {1, 4, 5, 8, 9, 16, 17, 64, 96, 97, 129, 193}) {
                const auto r     = residual(T, 1000U + static_cast<std::uint32_t>(T));
                const Result got = run(p, r, T, inject, ws, stream, false, fp32_norm);
                compare(got, oracle(r, norm, p.down_values, p.down_rows, p.up_values, T), T <= 16, inject,
                        tag + " T=" + std::to_string(T));
                check(ws.peak_used() <= capacity, tag + ": the workspace capacity covers T=" + std::to_string(T));
            }
            // Column invariance of the fused route.
            const auto r16    = residual(16, 4242U);
            const Result all  = run(p, r16, 16, inject, ws, stream, false, fp32_norm);
            bool same         = true;
            for (int c = 0; c < 16 && same; ++c) {
                const std::vector<std::uint16_t> one(r16.begin() + static_cast<std::ptrdiff_t>(c) * kW,
                                                     r16.begin() + static_cast<std::ptrdiff_t>(c + 1) * kW);
                const Result alone = run(p, one, 1, inject, ws, stream, false, fp32_norm);
                same = std::equal(alone.x.begin(), alone.x.end(), all.x.begin() + static_cast<std::ptrdiff_t>(c) * kH) &&
                       (!inject || std::equal(alone.inject.begin(), alone.inject.end(),
                                              all.inject.begin() + static_cast<std::ptrdiff_t>(c) * kS));
            }
            check(same, tag + ": each column of a 16-column call equals that column alone");
            const std::vector<std::uint16_t> r5(r16.begin(), r16.begin() + 5 * kW);
            const Result five = run(p, r5, 5, inject, ws, stream, false, fp32_norm);
            check(std::equal(five.x.begin(), five.x.end(), all.x.begin()), tag + ": a 5-column call equals the first five");
            const Result captured = run(p, r5, 5, inject, ws, stream, true, fp32_norm);
            check(captured.x == five.x && captured.inject == five.inject, tag + ": a graph-captured call equals the eager call");
        }
        up_collapse_bit_identity(Codec::Q8, stream);
        repack_q8(true, stream);
        repack_q8(false, stream);
        a8_encoded(Codec::Q8, stream);
        a8_encoded(Codec::Bf16, stream);
        // Shapes that disagree are refused.
        {
            Problem p = make_problem(true, Codec::Q8);
            infernix::WorkspaceArena ws(1 << 20);
            Device rd(static_cast<std::size_t>(kW) * 2), xd(kH * 2);
            const Tensor R(rd.p, DType::BF16, {kW, 1});
            const Tensor w(p.norm_device->p, DType::BF16, {kW});
            Tensor x(xd.p, DType::BF16, {kH, 1});
            bool threw = false;
            try {
                // The down projection carries injection rows the call does not ask for.
                infernix::ops::hyper_connection_mix(R, w, p.down_w, p.up_w, infernix::ops::LinearPolicy::A16Only, kS, kRank,
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
