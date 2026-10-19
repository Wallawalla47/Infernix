// PLE n-gram embedding injection (include/infernix/ops/ple.h) against independent oracles:
//   - ple_embed: exact. An E4M3 code times a BF16 scale is exact in FP32 (4 x 8 significant bits),
//     so each output must equal the FP64 product rounded once to BF16. The E4M3 decoding here is
//     independent of the device conversion; NaN codes are not represented inputs.
//   - ple_gate: the FP64 formula (unit-offset RMSNorm per stream, signed-sqrt gate, sigmoid,
//     conv-input RMSNorm). Each output is rounded once from an FP32 evaluation, so it lies within
//     one BF16 ULP of the FP64 value (relative 2^-7, with an absolute floor for values near 0).
//   - ple_conv_inject: the residual update R += gated + SiLU(causal dilated depthwise conv of the
//     conv inputs over the sequence's history), against FP64 with the same one-rounding bound on
//     the updated residual, and the state shift exactly: the history becomes the trailing span
//     columns of (old history, this call's conv inputs), in place or into another slot; an empty
//     destination leaves every state unchanged; untouched slots and guards stay intact.

#include "infernix/ops/ple.h"

#include "core/arena.h"
#include "ops/op_check.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

int g_failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

// FP8 E4M3FN: 1 sign, 4 exponent (bias 7), 3 mantissa bits; exponent 0 is subnormal; S.1111.111 is
// NaN (no infinities).
double decode_e4m3(std::uint8_t code) {
    const int sign     = code >> 7;
    const int exponent = (code >> 3) & 0xF;
    const int mantissa = code & 0x7;
    const double magnitude = exponent == 0 ? std::ldexp(mantissa / 8.0, -6)
                                           : std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
    return sign != 0 ? -magnitude : magnitude;
}

bool e4m3_nan(std::uint8_t code) { return (code & 0x7F) == 0x7F; }

std::uint16_t bf16_of(double value) { return f32_to_bf16(static_cast<float>(value)); }

double bf16_value(std::uint16_t bits) { return static_cast<double>(bf16_to_f32(bits)); }

std::vector<std::uint16_t> random_bf16(std::size_t count, std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<std::uint16_t> out(count);
    for (auto& v : out) { v = f32_to_bf16(dist(rng)); }
    return out;
}

// One BF16 ULP of the FP64 value (2^-7 relative at worst), with an absolute floor near zero.
constexpr PointwiseCriterion kOneRounding{.absolute = 1.0e-6, .relative = 1.0 / 128.0};

void test_embed(int width, int heads, int columns, std::uint32_t seed) {
    const std::string label = "ple_embed W=" + std::to_string(width) + " heads=" + std::to_string(heads) +
                              " T=" + std::to_string(columns);
    std::mt19937 rng(seed);
    const std::size_t elements = static_cast<std::size_t>(width) * heads;
    std::vector<std::uint8_t> rows(elements * columns);
    // Every non-NaN code appears (including both zeros and the subnormals), then random ones.
    std::size_t next = 0;
    for (int code = 0; code < 256 && next < rows.size(); ++code) {
        if (!e4m3_nan(static_cast<std::uint8_t>(code))) { rows[next++] = static_cast<std::uint8_t>(code); }
    }
    std::uniform_int_distribution<int> any_code(0, 255);
    for (; next < rows.size(); ++next) {
        std::uint8_t code = 0x7F;
        while (e4m3_nan(code)) { code = static_cast<std::uint8_t>(any_code(rng)); }
        rows[next] = code;
    }
    for (const float scale_value : {0.0234375F, -1.75F, 3.0e-3F}) {
        const std::uint16_t scale = f32_to_bf16(scale_value);
        DeviceBuffer d_rows  = to_device(rows);
        DeviceBuffer d_scale = to_device(std::vector<std::uint16_t>{scale});
        GuardedDeviceBuffer d_out(elements * columns * sizeof(std::uint16_t));
        d_out.fill(0xFF);
        Tensor t_rows(d_rows.p, DType::U8, {width, heads, columns});
        Tensor t_scale(d_scale.p, DType::BF16, {1});
        Tensor t_out(d_out.data(), DType::BF16, {static_cast<std::int32_t>(elements), columns});
        ops::ple_embed(t_rows, t_scale, t_out, nullptr);
        cuda_synchronize();
        const auto got = from_device<std::uint16_t>(d_out.data(), elements * columns);
        std::vector<std::uint16_t> want(elements * columns);
        // out[h * W + j, t] = bf16(e4m3(rows[j, h, t]) * scale): rows are [W, heads, T] with W fastest,
        // so the flat index is the same on both sides.
        for (std::size_t i = 0; i < want.size(); ++i) { want[i] = bf16_of(decode_e4m3(rows[i]) * bf16_value(scale)); }
        g_failures += verify_exact((label + " scale " + std::to_string(scale_value)).c_str(), got, want);
        g_failures += d_out.verify_guards(label);
    }
}

struct GateReference {
    std::vector<double> gated, normalized;
};

GateReference gate_reference(const std::vector<std::uint16_t>& key, const std::vector<std::uint16_t>& value,
                             const std::vector<std::uint16_t>& residual, const std::vector<std::uint16_t>& key_norm,
                             const std::vector<std::uint16_t>& query_norm, const std::vector<std::uint16_t>& conv_norm,
                             int streams, int hidden, int columns, double eps) {
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    GateReference out{std::vector<double>(width * columns), std::vector<double>(width * columns)};
    for (int t = 0; t < columns; ++t) {
        for (int s = 0; s < streams; ++s) {
            const std::size_t base = static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden;
            double key_sq = 0, query_sq = 0;
            for (int d = 0; d < hidden; ++d) {
                key_sq += bf16_value(key[base + d]) * bf16_value(key[base + d]);
                query_sq += bf16_value(residual[base + d]) * bf16_value(residual[base + d]);
            }
            const double key_inv   = 1.0 / std::sqrt(key_sq / hidden + eps);
            const double query_inv = 1.0 / std::sqrt(query_sq / hidden + eps);
            double dot = 0;
            for (int d = 0; d < hidden; ++d) {
                const std::size_t c = static_cast<std::size_t>(s) * hidden + d;
                dot += bf16_value(key[base + d]) * key_inv * (1.0 + bf16_value(key_norm[c])) *
                       bf16_value(residual[base + d]) * query_inv * (1.0 + bf16_value(query_norm[c]));
            }
            double g = dot / std::sqrt(static_cast<double>(hidden));
            g        = std::copysign(std::sqrt(std::max(std::abs(g), 1.0e-6)), g);
            const double gate = 1.0 / (1.0 + std::exp(-g));
            double gated_sq   = 0;
            for (int d = 0; d < hidden; ++d) {
                const double v = gate * bf16_value(value[static_cast<std::size_t>(t) * hidden + d]);
                out.gated[base + d] = v;
                gated_sq += v * v;
            }
            const double gated_inv = 1.0 / std::sqrt(gated_sq / hidden + eps);
            for (int d = 0; d < hidden; ++d) {
                const std::size_t c = static_cast<std::size_t>(s) * hidden + d;
                out.normalized[base + d] = out.gated[base + d] * gated_inv * (1.0 + bf16_value(conv_norm[c]));
            }
        }
    }
    return out;
}

void test_gate(int streams, int hidden, int columns, std::uint32_t seed) {
    const std::string label = "ple_gate S=" + std::to_string(streams) + " H=" + std::to_string(hidden) +
                              " T=" + std::to_string(columns);
    std::mt19937 rng(seed);
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    const auto key          = random_bf16(width * columns, rng, -2.0F, 2.0F);
    const auto value        = random_bf16(static_cast<std::size_t>(hidden) * columns, rng, -1.5F, 1.5F);
    auto residual           = random_bf16(width * columns, rng, -3.0F, 3.0F);
    // One column whose stream correlates with its key: a large gate on top of the near-zero ones.
    for (int d = 0; d < hidden; ++d) { residual[d] = key[d]; }
    const auto key_norm   = random_bf16(width, rng, -0.5F, 0.5F);
    const auto query_norm = random_bf16(width, rng, -0.5F, 0.5F);
    const auto conv_norm  = random_bf16(width, rng, -0.5F, 0.5F);
    const float eps       = 1.0e-6F;

    DeviceBuffer d_key = to_device(key), d_value = to_device(value), d_residual = to_device(residual);
    DeviceBuffer d_key_norm = to_device(key_norm), d_query_norm = to_device(query_norm),
                 d_conv_norm = to_device(conv_norm);
    GuardedDeviceBuffer d_gated(width * columns * 2), d_normalized(width * columns * 2);
    d_gated.fill(0xFF);
    d_normalized.fill(0xFF);
    const auto W = static_cast<std::int32_t>(width);
    Tensor t_key(d_key.p, DType::BF16, {W, columns});
    Tensor t_value(d_value.p, DType::BF16, {hidden, columns});
    Tensor t_residual(d_residual.p, DType::BF16, {W, columns});
    Tensor t_key_norm(d_key_norm.p, DType::BF16, {W});
    Tensor t_query_norm(d_query_norm.p, DType::BF16, {W});
    Tensor t_conv_norm(d_conv_norm.p, DType::BF16, {W});
    Tensor t_gated(d_gated.data(), DType::BF16, {W, columns});
    Tensor t_normalized(d_normalized.data(), DType::BF16, {W, columns});
    ops::ple_gate(t_key, t_value, t_residual, t_key_norm, t_query_norm, t_conv_norm, streams, eps, t_gated,
                  t_normalized, nullptr);
    cuda_synchronize();

    const GateReference ref =
        gate_reference(key, value, residual, key_norm, query_norm, conv_norm, streams, hidden, columns, eps);
    const auto gated      = from_device_bf16(d_gated.data(), width * columns);
    const auto normalized = from_device_bf16(d_normalized.data(), width * columns);
    g_failures += verify_pointwise(label + " gated", gated, ref.gated, kOneRounding);
    g_failures += verify_pointwise(label + " normalized", normalized, ref.normalized, kOneRounding);
    g_failures += d_gated.verify_guards(label + " gated");
    g_failures += d_normalized.verify_guards(label + " normalized");
    // The inputs are read only.
    g_failures += verify_exact((label + " residual preserved").c_str(),
                               from_device<std::uint16_t>(d_residual, residual.size()), residual);
}

// sequences x width columns over `slots` state slots; sequence i reads source[i] and, when
// `commit`, writes destination[i].
void test_conv(int channels, int taps, int dilation, int sequences, int width, bool commit, bool distinct,
               std::uint32_t seed) {
    const std::string label = "ple_conv_inject C=" + std::to_string(channels) + " K=" + std::to_string(taps) +
                              " d=" + std::to_string(dilation) + " B=" + std::to_string(sequences) +
                              " W=" + std::to_string(width) + (commit ? (distinct ? " moved" : " in place") : " verify");
    std::mt19937 rng(seed);
    const int span    = (taps - 1) * dilation;
    const int columns = sequences * width;
    const int slots   = 2 * sequences + 1;
    const std::size_t C = static_cast<std::size_t>(channels);
    const auto gated      = random_bf16(C * columns, rng, -1.0F, 1.0F);
    const auto normalized = random_bf16(C * columns, rng, -2.0F, 2.0F);
    const auto weight     = random_bf16(C * taps, rng, -0.8F, 0.8F);
    const auto states     = random_bf16(C * span * slots, rng, -2.0F, 2.0F);
    const auto residual   = random_bf16(C * columns, rng, -4.0F, 4.0F);
    std::vector<std::int32_t> source(sequences), destination(sequences);
    for (int i = 0; i < sequences; ++i) {
        source[i]      = 2 * i + 1;
        destination[i] = distinct ? 2 * i : source[i];
    }

    DeviceBuffer d_gated = to_device(gated), d_normalized = to_device(normalized), d_weight = to_device(weight);
    DeviceBuffer d_source = to_device(source), d_destination = to_device(destination);
    GuardedDeviceBuffer d_states(states.size() * 2), d_residual(residual.size() * 2);
    d_states.copy_from_host(states.data(), states.size() * 2);
    d_residual.copy_from_host(residual.data(), residual.size() * 2);
    Tensor t_gated(d_gated.p, DType::BF16, {channels, columns});
    Tensor t_normalized(d_normalized.p, DType::BF16, {channels, columns});
    Tensor t_weight(d_weight.p, DType::BF16, {channels, taps});
    Tensor t_states(d_states.data(), DType::BF16, {channels, span, slots});
    Tensor t_source(d_source.p, DType::I32, {sequences});
    Tensor t_destination = commit ? Tensor(d_destination.p, DType::I32, {sequences}) : Tensor{};
    Tensor t_residual(d_residual.data(), DType::BF16, {channels, columns});
    ops::ple_conv_inject(t_gated, t_normalized, t_weight, dilation, t_states, t_source, t_destination, t_residual,
                         nullptr);
    cuda_synchronize();

    // The history of sequence i, oldest first: its slot's span columns, then this call's inputs.
    const auto history = [&](int sequence, int channel, int index) -> double {
        if (index < span) {
            return bf16_value(states[(static_cast<std::size_t>(source[sequence]) * span + index) * C + channel]);
        }
        return bf16_value(normalized[(static_cast<std::size_t>(sequence) * width + (index - span)) * C + channel]);
    };
    std::vector<double> want_residual(residual.size());
    for (int sequence = 0; sequence < sequences; ++sequence) {
        for (int local = 0; local < width; ++local) {
            for (int c = 0; c < channels; ++c) {
                double acc = 0;
                for (int j = 0; j < taps; ++j) {
                    acc += bf16_value(weight[static_cast<std::size_t>(j) * C + c]) *
                           history(sequence, c, span + local - (taps - 1 - j) * dilation);
                }
                const std::size_t i = (static_cast<std::size_t>(sequence) * width + local) * C + c;
                want_residual[i]    = bf16_value(residual[i]) + bf16_value(gated[i]) + acc / (1.0 + std::exp(-acc));
            }
        }
    }
    g_failures += verify_pointwise(label + " residual", from_device_bf16(d_residual.data(), residual.size()),
                                   want_residual, kOneRounding);

    std::vector<std::uint16_t> want_states = states;
    if (commit) {
        for (int sequence = 0; sequence < sequences; ++sequence) {
            for (int k = 0; k < span; ++k) {
                for (int c = 0; c < channels; ++c) {
                    want_states[(static_cast<std::size_t>(destination[sequence]) * span + k) * C + c] =
                        bf16_of(history(sequence, c, width + k));
                }
            }
        }
    }
    g_failures += verify_exact((label + " states").c_str(), from_device<std::uint16_t>(d_states.data(), states.size()),
                               want_states);
    g_failures += d_states.verify_guards(label + " states");
    g_failures += d_residual.verify_guards(label + " residual");
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        // Qwen3.8-Flash-Next's n-gram rows: 16 heads of 160 (embed 2,560); decode, verification and a
        // prefill chunk, plus a small odd geometry.
        test_embed(160, 16, 1, 11);
        test_embed(160, 16, 5, 12);
        test_embed(160, 16, 300, 13);
        test_embed(24, 3, 7, 14);
        // Four streams of 2,048 and an odd geometry (H not a multiple of the 256-thread block).
        for (const int columns : {1, 4, 33}) { test_gate(4, 2048, columns, 100U + columns); }
        test_gate(2, 300, 5, 150);
        test_gate(3, 1000, 2, 151);
        // Dilation is the n-gram size (3); widths shorter and longer than the span, a sequence per
        // row, the in-place, moved and verification (no destination) state updates.
        for (const bool distinct : {false, true}) {
            test_conv(4 * 2048, 4, 3, 1, 1, true, distinct, 200U + distinct);
            test_conv(4 * 2048, 4, 3, 2, 5, true, distinct, 210U + distinct);
            test_conv(4 * 2048, 4, 3, 1, 40, true, distinct, 220U + distinct);
        }
        test_conv(4 * 2048, 4, 3, 3, 4, false, false, 230U);
        test_conv(300, 2, 2, 2, 3, true, false, 240U);
        test_conv(300, 5, 1, 1, 9, true, true, 241U);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d ple check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all ple checks passed\n");
    return 0;
}
