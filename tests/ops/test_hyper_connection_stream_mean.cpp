// hyper_connection_stream_mean against an FP64 oracle: out[d,t] = (1/S) sum_s R[s*H+d,t] from the
// represented BF16 residual, the target feature of Qwen3.8-Flash-Next's DFlash2 drafter (the mean of
// the four hyper-connection streams of a completed layer). The Op sums the streams in FP32 in order,
// divides by S (exact) and rounds once to BF16, so with e = 3 * 2^-24 * sum|R| / S (the bound of an
// FP32 sum of four terms) each output satisfies |out - ref| <= 2^-8 (|ref| + e) + e. The cases
// include exponent spreads beyond FP32's 24 bits and cancelling streams, where e dominates. Cases:
// Flash-Next's geometry
// (S = 4, H = 2560) at T = 1, 8, 17 and 512 into a contiguous output and into a row slice of a
// wider feature matrix (pitch 5 H, the DFlash2 feature layout: the other rows stay untouched), a
// graph-captured replay bit-identical to the eager call, and input preservation.
//
// A call wider than a grid's 65,535 rows (core/grid_rows.h), on small rows (S = 2, H = 16) at 70,001
// columns: the stream mean against the same bound, hyper_connection_expand exactly (every stream is the
// column's x), and hyper_connection_inject (R += inject * y per stream, one FP32 multiply-add rounded once
// to BF16) within one BF16 rounding of the FP64 value.
//
// hyper_connection_normed_stream_mean (the prefill copy-ahead's guess input) against the FP64 value
// ref = (1/S) sum_s R_s r_s (1 + w_s), r_s = 1 / sqrt(mean R_s^2 + eps), from the represented residual and
// norm weights: |out - ref| <= 2^-8 (|ref| + e) + e with e = 2^-16 (1/S) sum_s |R_s r_s (1 + w_s)|, the FP32
// sums of squares, rsqrtf and the four-term sum (each well inside 2^-16 relative) and the one BF16 rounding.
// Flash-Next's geometry with FP32 norm weights (FP4-Local) and BF16 ones at T = 1, 17 and 512, and S = 3,
// H = 200 (25 vectors: a partly idle warp) at T = 5; the residual is left as it was.

#include "ops/op_family.h"
#include "infernix/ops/hyper_connection.h"
#include "core/decode_graph.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace {

namespace t = infernix::test;
using infernix::DType;
using infernix::Tensor;

constexpr int kS = 4, kH = 2560, kW = kS * kH, kTaps = 5;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::uint32_t next(std::uint32_t& state) { return state = state * 1664525U + 1013904223U; }

// Residual words with wide exponent spreads across streams (some columns span 2^-20..2^4).
std::vector<std::uint16_t> make_residual(int columns, std::uint32_t seed) {
    std::vector<std::uint16_t> r(static_cast<std::size_t>(kW) * columns);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < r.size(); ++i) {
        const float unit = static_cast<float>(static_cast<std::int32_t>(next(state) >> 8) - (1 << 23)) /
                           static_cast<float>(1 << 23);
        const int scale  = static_cast<int>(next(state) % 25U) - 20;
        r[i]             = t::f32_to_bf16(std::ldexp(unit, scale));
    }
    return r;
}

void run_case(int columns, int pitch, std::uint32_t seed, bool graph) {
    const std::vector<std::uint16_t> host = make_residual(columns, seed);
    infernix::DeviceBuffer residual = t::to_device(host);
    const std::size_t out_words = static_cast<std::size_t>(pitch) * columns;
    t::GuardedDeviceBuffer out(out_words * sizeof(std::uint16_t));
    out.fill(0xa5);
    Tensor input(residual.p, DType::BF16, {kW, columns});
    // A row slice [kH, columns] at row offset kH (the second tap) of a [pitch, columns] matrix.
    const int offset = pitch == kH ? 0 : kH;
    Tensor whole(out.data(), DType::BF16, {pitch, columns});
    Tensor view = whole.slice(0, offset, kH);
    const auto launch = [&](cudaStream_t stream) {
        infernix::ops::hyper_connection_stream_mean(input, kS, view, stream);
    };
    launch(nullptr);
    t::cuda_synchronize();
    const std::vector<std::uint16_t> eager = t::from_device<std::uint16_t>(out.data(), out_words);
    const std::string label = "stream mean T=" + std::to_string(columns) + " pitch=" + std::to_string(pitch);
    for (int c = 0; c < columns; ++c) {
        for (int row = 0; row < pitch; ++row) {
            const std::uint16_t got = eager[static_cast<std::size_t>(c) * pitch + row];
            if (row < offset || row >= offset + kH) {
                if (got != 0xa5a5U) {
                    check(false, label + ": a row outside the slice was written");
                    return;
                }
                continue;
            }
            const int d = row - offset;
            double sum = 0.0, magnitude = 0.0;
            for (int s = 0; s < kS; ++s) {
                const double x =
                    t::bf16_to_f32(host[static_cast<std::size_t>(c) * kW + static_cast<std::size_t>(s) * kH + d]);
                sum += x;
                magnitude += std::abs(x);
            }
            const double ref   = sum / kS;
            const double value = t::bf16_to_f32(got);
            const double fp32  = 3.0 * std::ldexp(magnitude / kS, -24);
            if (!(std::abs(value - ref) <= std::ldexp(std::abs(ref) + fp32, -8) + fp32)) {
                check(false, label + ": column " + std::to_string(c) + " row " + std::to_string(d) +
                                 " is " + std::to_string(value) + ", expected " + std::to_string(ref));
                return;
            }
        }
    }
    failures += out.verify_guards(label + " guards");
    check(t::from_device<std::uint16_t>(residual, host.size()) == host, label + ": residual modified");
    if (graph) {
        out.fill(0xa5);
        t::cuda_synchronize();
        cudaStream_t stream = nullptr;
        t::cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        {
            infernix::DecodeGraphDefinition definition;
            infernix::DecodeGraphExecutable executable;
            definition.capture(stream, [&] { launch(stream); });
            executable.instantiate(definition);
            executable.launch(stream);
            t::cuda_check(cudaStreamSynchronize(stream), "graph synchronize");
        }
        t::cuda_check(cudaStreamDestroy(stream), "destroy stream");
        check(t::from_device<std::uint16_t>(out.data(), out_words) == eager, label + ": graph replay differs");
    }
}

void wide_call(int columns) {
    constexpr int S = 2, H = 16, W = S * H;
    const std::string label = "wide call T=" + std::to_string(columns);
    std::uint32_t state     = 0x77A1U;
    const auto value        = [&](float range) {
        const float unit = static_cast<float>(static_cast<std::int32_t>(next(state) >> 8) - (1 << 23)) /
                           static_cast<float>(1 << 23);
        return unit * range;
    };
    std::vector<std::uint16_t> r(static_cast<std::size_t>(W) * columns), y(static_cast<std::size_t>(H) * columns);
    std::vector<float> inject(static_cast<std::size_t>(S) * columns);
    for (auto& v : r) { v = t::f32_to_bf16(value(4.0F)); }
    for (auto& v : y) { v = t::f32_to_bf16(value(2.0F)); }
    for (auto& v : inject) { v = value(2.0F); }
    infernix::DeviceBuffer d_r = t::to_device(r), d_y = t::to_device(y), d_inject = t::to_device(inject);
    infernix::DeviceBuffer d_mean(static_cast<std::size_t>(H) * columns * sizeof(std::uint16_t));
    infernix::DeviceBuffer d_expanded(r.size() * sizeof(std::uint16_t));
    const Tensor residual(d_r.p, DType::BF16, {W, columns});
    Tensor mean(d_mean.p, DType::BF16, {H, columns});
    infernix::ops::hyper_connection_stream_mean(residual, S, mean, nullptr);
    Tensor expanded(d_expanded.p, DType::BF16, {W, columns});
    infernix::ops::hyper_connection_expand(Tensor(d_y.p, DType::BF16, {H, columns}), S, expanded, nullptr);
    Tensor injected(d_r.p, DType::BF16, {W, columns}); // in place, after the mean has read it
    infernix::ops::hyper_connection_inject(Tensor(d_y.p, DType::BF16, {H, columns}),
                                           Tensor(d_inject.p, DType::FP32, {S, columns}), injected, nullptr);
    t::cuda_synchronize();
    const auto got_mean     = t::from_device<std::uint16_t>(d_mean, static_cast<std::size_t>(H) * columns);
    const auto got_expanded = t::from_device<std::uint16_t>(d_expanded, r.size());
    const auto got_injected = t::from_device<std::uint16_t>(d_r, r.size());
    for (int c = 0; c < columns; ++c) {
        for (int d = 0; d < H; ++d) {
            const std::size_t at = static_cast<std::size_t>(c) * W + d;
            const double a = t::bf16_to_f32(r[at]), b = t::bf16_to_f32(r[at + H]);
            const double ref = (a + b) / S, magnitude = (std::abs(a) + std::abs(b)) / S;
            const double fp32 = 3.0 * std::ldexp(magnitude, -24);
            const double got  = t::bf16_to_f32(got_mean[static_cast<std::size_t>(c) * H + d]);
            if (!(std::abs(got - ref) <= std::ldexp(std::abs(ref) + fp32, -8) + fp32)) {
                check(false, label + ": mean column " + std::to_string(c) + " row " + std::to_string(d));
                return;
            }
            for (int s = 0; s < S; ++s) {
                const std::size_t i = static_cast<std::size_t>(c) * W + static_cast<std::size_t>(s) * H + d;
                const std::uint16_t x = y[static_cast<std::size_t>(c) * H + d];
                if (got_expanded[i] != x) {
                    check(false, label + ": expand column " + std::to_string(c) + " stream " + std::to_string(s));
                    return;
                }
                const double want = t::bf16_to_f32(r[i]) +
                                    static_cast<double>(inject[static_cast<std::size_t>(c) * S + s]) * t::bf16_to_f32(x);
                const double have = t::bf16_to_f32(got_injected[i]);
                if (!(std::abs(have - want) <= std::ldexp(std::abs(want), -8) + 1.0e-6)) {
                    check(false, label + ": inject column " + std::to_string(c) + " stream " + std::to_string(s));
                    return;
                }
            }
        }
    }
}

void normed_case(int streams, int hidden, int columns, bool fp32_weight, std::uint32_t seed) {
    const int width = streams * hidden;
    const std::string label = "normed mean S=" + std::to_string(streams) + " H=" + std::to_string(hidden) +
                              " T=" + std::to_string(columns) + (fp32_weight ? " fp32" : " bf16");
    std::uint32_t state = seed;
    const auto unit     = [&] {
        return static_cast<float>(static_cast<std::int32_t>(next(state) >> 8) - (1 << 23)) / static_cast<float>(1 << 23);
    };
    std::vector<std::uint16_t> r(static_cast<std::size_t>(width) * columns), w16(static_cast<std::size_t>(width));
    std::vector<float> w32(static_cast<std::size_t>(width));
    for (auto& v : r) { v = t::f32_to_bf16(std::ldexp(unit(), static_cast<int>(next(state) % 13U) - 8)); }
    for (std::size_t i = 0; i < w32.size(); ++i) {
        w32[i] = 0.5F * unit();
        w16[i] = t::f32_to_bf16(w32[i]);
    }
    constexpr float kEps = 1.0e-6F;
    infernix::DeviceBuffer d_r = t::to_device(r), d_w = fp32_weight ? t::to_device(w32) : t::to_device(w16);
    t::GuardedDeviceBuffer out(static_cast<std::size_t>(hidden) * columns * sizeof(std::uint16_t));
    out.fill(0xa5);
    const Tensor residual(d_r.p, DType::BF16, {width, columns});
    const Tensor weight(d_w.p, fp32_weight ? DType::FP32 : DType::BF16, {width});
    Tensor mean(out.data(), DType::BF16, {hidden, columns});
    infernix::ops::hyper_connection_normed_stream_mean(residual, weight, streams, kEps, mean, nullptr);
    t::cuda_synchronize();
    const auto got = t::from_device<std::uint16_t>(out.data(), static_cast<std::size_t>(hidden) * columns);
    for (int c = 0; c < columns; ++c) {
        std::vector<double> inv(static_cast<std::size_t>(streams));
        for (int s = 0; s < streams; ++s) {
            double squares = 0.0;
            for (int d = 0; d < hidden; ++d) {
                const double x = t::bf16_to_f32(r[static_cast<std::size_t>(c) * width + static_cast<std::size_t>(s) * hidden + d]);
                squares += x * x;
            }
            inv[static_cast<std::size_t>(s)] = 1.0 / std::sqrt(squares / hidden + static_cast<double>(kEps));
        }
        for (int d = 0; d < hidden; ++d) {
            double sum = 0.0, magnitude = 0.0;
            for (int s = 0; s < streams; ++s) {
                const std::size_t i = static_cast<std::size_t>(s) * hidden + d;
                const double wv = fp32_weight ? static_cast<double>(w32[i]) : t::bf16_to_f32(w16[i]);
                const double n  = t::bf16_to_f32(r[static_cast<std::size_t>(c) * width + i]) *
                                 inv[static_cast<std::size_t>(s)] * (1.0 + wv);
                sum += n;
                magnitude += std::abs(n);
            }
            const double ref = sum / streams, e = std::ldexp(magnitude / streams, -16);
            const double value = t::bf16_to_f32(got[static_cast<std::size_t>(c) * hidden + d]);
            if (!(std::abs(value - ref) <= std::ldexp(std::abs(ref) + e, -8) + e)) {
                check(false, label + ": column " + std::to_string(c) + " row " + std::to_string(d) + " is " +
                                 std::to_string(value) + ", expected " + std::to_string(ref));
                return;
            }
        }
    }
    failures += out.verify_guards(label + " guards");
    check(t::from_device<std::uint16_t>(d_r, r.size()) == r, label + ": residual modified");
}

} // namespace

INFERNIX_OP_TEST(hyper_connection_stream_mean) {
    if (t::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        for (const int columns : {1, 8, 17, 512}) {
            run_case(columns, kH, 0x5101U + static_cast<std::uint32_t>(columns), columns == 8);
            run_case(columns, kTaps * kH, 0x5201U + static_cast<std::uint32_t>(columns), columns == 8);
        }
        wide_call(70001);
        for (const int columns : {1, 17, 512}) {
            normed_case(kS, kH, columns, true, 0x5301U + static_cast<std::uint32_t>(columns));
            normed_case(kS, kH, columns, false, 0x5401U + static_cast<std::uint32_t>(columns));
        }
        normed_case(3, 200, 5, true, 0x5501U);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "hyper_connection_stream_mean: %s\n", error.what());
        return 1;
    }
    std::printf("%s hyper_connection_stream_mean\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0 ? 0 : 1;
}
