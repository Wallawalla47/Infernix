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

} // namespace

int main() {
    if (t::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        for (const int columns : {1, 8, 17, 512}) {
            run_case(columns, kH, 0x5101U + static_cast<std::uint32_t>(columns), columns == 8);
            run_case(columns, kTaps * kH, 0x5201U + static_cast<std::uint32_t>(columns), columns == 8);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "hyper_connection_stream_mean: %s\n", error.what());
        return 1;
    }
    std::printf("%s hyper_connection_stream_mean\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0 ? 0 : 1;
}
