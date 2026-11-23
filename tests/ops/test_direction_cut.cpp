// The direction cut and inject (include/infernix/ops/direction_cut.h; --ablate) against an FP64 oracle:
//   out[s*H+h, t] = r + inject[s, t] * (y[h, t] - alpha * c[t] * dir[h]),  c[t] = sum_h dir[h] * y[h, t]
// from the represented BF16 y and residual, FP32 dir and inject. For moe_combine_cut_inject y is the combine before
// its BF16 rounding (FP64 sum of the routed outputs and the gated shared expert), so its rounding is in the bound.
//
// Criterion (pointwise, finite inputs, every output finite): |out - ref| <= 2^-8 |ref| + e, with
//   e = 2^-20 (|r| + |a| (|y| + alpha |dir_h| |c|)) + |a| alpha |dir_h| 2^-17 sum_j |dir_j y_j|
//       [+ |a| (dy_h + alpha |dir_h| sum_j |dir_j| dy_j), dy = 2^-8 |y| + 2^-20 sum |terms| for the combine]
// the inject's one BF16 rounding (2^-8 relative, nearest even halves it), the FP32 cut and multiply-add, the FP32
// column dot over H = 2560 products (a 45-deep sum: 2^-17 of its magnitude bounds it with room) and the combine's
// BF16 rounding of y. Cases cancel on purpose: a direction nearly parallel to y with alpha 1 leaves v ~ 0.
//
// Also: an inactive cut is the plain inject bit for bit; the combine form equals moe_combine + the cut inject of its
// y bit for bit (the same FMA chain); a column's bits do not depend on T or its position; a captured graph replays
// the eager bits; inputs and the residual's guards are untouched; out-of-domain calls are refused. Geometry: Flash-Next
// (S = 4, H = 2560, top-8) at T = 1, 2, 31, 32, 33 and 257, top-1 and top-16, H = 200 (a partial warp), and a call
// wider than a grid's 65,535 rows (S = 2, H = 16, T = 70,001). residual_delta_cut (Qwen3.5's single-stream residual,
// after the lane's own write) at H = 5120, 3072 and 2048 (the 27B, the 122B, the 35B): see single_case below.

#include "ops/op_family.h"
#include "infernix/ops/direction_cut.h"
#include "infernix/ops/hyper_connection.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "core/decode_graph.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace t = infernix::test;
using infernix::DType;
using infernix::Tensor;
using infernix::ops::DirectionCut;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

struct Rng {
    std::uint32_t state;
    float unit() { // (-1, 1)
        state = state * 1664525U + 1013904223U;
        return static_cast<float>(static_cast<std::int32_t>(state >> 8) - (1 << 23)) / static_cast<float>(1 << 23);
    }
    int below(int n) {
        state = state * 1664525U + 1013904223U;
        return static_cast<int>((state >> 8) % static_cast<std::uint32_t>(n));
    }
};

// One case's host inputs.
struct Case {
    int streams = 4, hidden = 2560, columns = 1, top_k = 8;
    float alpha = 1.0F;
    bool combine = false;
    std::vector<float> dir;               // FP32 [H], unit
    std::vector<std::uint16_t> y;         // BF16 [H, T] (the cut-inject form)
    std::vector<std::uint16_t> outputs;   // BF16 [H, T*K] (the combine form)
    std::vector<float> weights;           // FP32 [K, T]
    std::vector<float> gate;              // FP32 [T]
    std::vector<std::uint16_t> shared;    // BF16 [H, T]
    std::vector<float> inject;            // FP32 [S, T]
    std::vector<std::uint16_t> residual;  // BF16 [S*H, T]
};

// A unit direction; `parallel_to` (a column of y's values) makes it nearly parallel to that column, so alpha 1
// cancels almost all of it.
std::vector<float> make_direction(int hidden, Rng& rng, const std::vector<double>* parallel_to) {
    std::vector<double> v(static_cast<std::size_t>(hidden));
    for (int h = 0; h < hidden; ++h) {
        v[static_cast<std::size_t>(h)] =
            parallel_to != nullptr ? (*parallel_to)[static_cast<std::size_t>(h)] + 1.0e-3 * rng.unit() : rng.unit();
    }
    double norm = 0.0;
    for (const double x : v) { norm += x * x; }
    norm = std::sqrt(norm);
    std::vector<float> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) { out[i] = static_cast<float>(v[i] / norm); }
    return out;
}

Case make_case(int streams, int hidden, int columns, int top_k, float alpha, bool combine, bool parallel,
               std::uint32_t seed) {
    Case c{.streams = streams, .hidden = hidden, .columns = columns, .top_k = top_k, .alpha = alpha, .combine = combine};
    Rng rng{seed};
    const auto bf = [&](float range) { return t::f32_to_bf16(rng.unit() * range); };
    const std::size_t H = static_cast<std::size_t>(hidden), T = static_cast<std::size_t>(columns);
    c.y.resize(H * T);
    for (auto& v : c.y) { v = t::f32_to_bf16(std::ldexp(rng.unit(), rng.below(9) - 6)); }
    if (combine) {
        c.outputs.resize(H * T * static_cast<std::size_t>(top_k));
        for (auto& v : c.outputs) { v = bf(2.0F); }
        c.weights.resize(static_cast<std::size_t>(top_k) * T);
        for (auto& v : c.weights) { v = 0.5F * (rng.unit() + 1.0F) / static_cast<float>(top_k); }
        c.gate.resize(T);
        for (auto& v : c.gate) { v = 0.5F * (rng.unit() + 1.0F); }
        c.shared.resize(H * T);
        for (auto& v : c.shared) { v = bf(1.0F); }
    }
    std::vector<double> first(H);
    for (std::size_t h = 0; h < H; ++h) { first[h] = t::bf16_to_f32(c.y[h]); }
    c.dir = make_direction(hidden, rng, parallel && !combine ? &first : nullptr);
    c.inject.resize(static_cast<std::size_t>(streams) * T);
    for (auto& v : c.inject) { v = 1.5F * rng.unit(); }
    c.residual.resize(static_cast<std::size_t>(streams) * H * T);
    for (auto& v : c.residual) { v = t::f32_to_bf16(std::ldexp(rng.unit(), rng.below(9) - 4)); }
    return c;
}

// The device result of one call, and the stream it ran on (graph: captured and replayed).
std::vector<std::uint16_t> run(const Case& c, bool graph, const std::string& label) {
    const std::size_t H = static_cast<std::size_t>(c.hidden), T = static_cast<std::size_t>(c.columns);
    infernix::DeviceBuffer d_dir = t::to_device(c.dir), d_inject = t::to_device(c.inject);
    t::GuardedDeviceBuffer residual(c.residual.size() * sizeof(std::uint16_t));
    residual.copy_from_host(c.residual.data(), residual.bytes());
    infernix::DeviceBuffer d_y, d_outputs, d_weights, d_gate, d_shared, d_ids;
    if (c.combine) {
        d_outputs = t::to_device(c.outputs);
        d_weights = t::to_device(c.weights);
        d_gate    = t::to_device(c.gate);
        d_shared  = t::to_device(c.shared);
        d_ids     = infernix::DeviceBuffer(static_cast<std::size_t>(c.top_k) * T * sizeof(std::int32_t));
        d_ids.fill(0);
    } else {
        d_y = t::to_device(c.y);
    }
    const DirectionCut cut{.direction = static_cast<const float*>(d_dir.p), .alpha = c.alpha};
    const Tensor inject(d_inject.p, DType::FP32, {c.streams, c.columns});
    Tensor out(residual.data(), DType::BF16, {c.streams * c.hidden, c.columns});
    infernix::ops::MoeRouting routing{
        Tensor(d_ids.p, DType::I32, {c.top_k, c.columns}), Tensor(d_weights.p, DType::FP32, {c.top_k, c.columns}),
        Tensor(d_gate.p, DType::FP32, {c.columns}), {}};
    const auto launch = [&](cudaStream_t stream) {
        if (c.combine) {
            infernix::ops::moe_combine_cut_inject(
                Tensor(d_outputs.p, DType::BF16, {c.hidden, c.columns * c.top_k}), routing,
                Tensor(d_shared.p, DType::BF16, {c.hidden, c.columns}), cut, inject, out, stream);
        } else {
            infernix::ops::hyper_connection_cut_inject(Tensor(d_y.p, DType::BF16, {c.hidden, c.columns}), cut, inject,
                                                       out, stream);
        }
    };
    if (graph) {
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
    } else {
        launch(nullptr);
        t::cuda_synchronize();
    }
    failures += residual.verify_guards(label + " guards");
    if (c.combine) {
        check(t::from_device<std::uint16_t>(d_outputs, c.outputs.size()) == c.outputs, label + ": outputs modified");
        check(t::from_device<std::uint16_t>(d_shared, c.shared.size()) == c.shared, label + ": shared modified");
    } else {
        check(t::from_device<std::uint16_t>(d_y, c.y.size()) == c.y, label + ": y modified");
    }
    return t::from_device<std::uint16_t>(residual.data(), c.residual.size());
}

// The oracle check of every output element.
void verify(const Case& c, const std::vector<std::uint16_t>& got, const std::string& label) {
    const std::size_t H = static_cast<std::size_t>(c.hidden), S = static_cast<std::size_t>(c.streams);
    const std::size_t K = static_cast<std::size_t>(c.top_k);
    std::vector<double> y(H), dy(H);
    for (int col = 0; col < c.columns; ++col) {
        const std::size_t t0 = static_cast<std::size_t>(col);
        for (std::size_t h = 0; h < H; ++h) {
            if (!c.combine) {
                y[h]  = t::bf16_to_f32(c.y[t0 * H + h]);
                dy[h] = 0.0;
                continue;
            }
            double sum = 0.0, magnitude = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                const double term = static_cast<double>(c.weights[t0 * K + k]) *
                                    t::bf16_to_f32(c.outputs[(t0 * K + k) * H + h]);
                sum += term;
                magnitude += std::abs(term);
            }
            const double own = static_cast<double>(c.gate[t0]) * t::bf16_to_f32(c.shared[t0 * H + h]);
            sum += own;
            magnitude += std::abs(own);
            y[h]  = sum;
            dy[h] = std::ldexp(std::abs(sum), -8) + std::ldexp(magnitude, -20);
        }
        double dot = 0.0, dot_magnitude = 0.0, dir_dy = 0.0;
        for (std::size_t h = 0; h < H; ++h) {
            dot += static_cast<double>(c.dir[h]) * y[h];
            dot_magnitude += std::abs(static_cast<double>(c.dir[h]) * y[h]);
            dir_dy += std::abs(static_cast<double>(c.dir[h])) * dy[h];
        }
        for (std::size_t s = 0; s < S; ++s) {
            const double a = c.inject[t0 * S + s];
            for (std::size_t h = 0; h < H; ++h) {
                const std::size_t at = t0 * S * H + s * H + h;
                const double r = t::bf16_to_f32(c.residual[at]), d = c.dir[h];
                const double ref   = r + a * (y[h] - c.alpha * dot * d);
                const double cut_d = c.alpha * std::abs(d);
                const double e     = std::ldexp(std::abs(r) + std::abs(a) * (std::abs(y[h]) + cut_d * std::abs(dot)), -20) +
                                 std::abs(a) * cut_d * std::ldexp(dot_magnitude, -17) +
                                 std::abs(a) * (dy[h] + cut_d * dir_dy);
                const double value = t::bf16_to_f32(got[at]);
                if (!std::isfinite(value) || !(std::abs(value - ref) <= std::ldexp(std::abs(ref), -8) + e)) {
                    check(false, label + ": column " + std::to_string(col) + " stream " + std::to_string(s) + " row " +
                                     std::to_string(h) + " is " + std::to_string(value) + ", expected " +
                                     std::to_string(ref) + " (bound " + std::to_string(e) + ")");
                    return;
                }
            }
        }
    }
}

// Column `col` of `c` as a T = 1 case (the same bytes, at position 0).
Case column_alone(const Case& c, int col) {
    Case one = c;
    one.columns           = 1;
    const std::size_t H   = static_cast<std::size_t>(c.hidden), S = static_cast<std::size_t>(c.streams);
    const std::size_t K   = static_cast<std::size_t>(c.top_k), t0 = static_cast<std::size_t>(col);
    const auto slice = [](const auto& v, std::size_t from, std::size_t n) {
        return std::vector<typename std::decay_t<decltype(v)>::value_type>(v.begin() + from, v.begin() + from + n);
    };
    one.inject   = slice(c.inject, t0 * S, S);
    one.residual = slice(c.residual, t0 * S * H, S * H);
    if (c.combine) {
        one.outputs = slice(c.outputs, t0 * K * H, K * H);
        one.weights = slice(c.weights, t0 * K, K);
        one.gate    = slice(c.gate, t0, 1);
        one.shared  = slice(c.shared, t0 * H, H);
    } else {
        one.y = slice(c.y, t0 * H, H);
    }
    return one;
}

void run_case(const Case& c, const std::string& label, bool invariance) {
    const std::vector<std::uint16_t> eager = run(c, false, label);
    verify(c, eager, label);
    check(run(c, false, label + " repeat") == eager, label + ": a repeated call differs");
    check(run(c, true, label + " graph") == eager, label + ": the graph replay differs");
    if (invariance && c.columns > 1) {
        const std::size_t width = static_cast<std::size_t>(c.streams) * c.hidden;
        for (const int col : {0, c.columns / 2, c.columns - 1}) {
            const std::vector<std::uint16_t> alone = run(column_alone(c, col), false, label + " alone");
            check(std::equal(alone.begin(), alone.end(), eager.begin() + static_cast<std::ptrdiff_t>(col * width)),
                  label + ": column " + std::to_string(col) + " differs from its T = 1 call");
        }
    }
}

// An inactive cut runs the plain inject: the bits of hyper_connection_inject / moe_combine_inject.
void inactive_is_plain(Case c, const std::string& label) {
    const std::vector<std::uint16_t> plain = [&] {
        Case off = c;
        off.alpha = 0.0F;
        return run(off, false, label + " alpha 0");
    }();
    const std::size_t H = static_cast<std::size_t>(c.hidden), T = static_cast<std::size_t>(c.columns);
    infernix::DeviceBuffer d_inject = t::to_device(c.inject), residual = t::to_device(c.residual);
    Tensor out(residual.p, DType::BF16, {c.streams * c.hidden, c.columns});
    const Tensor inject(d_inject.p, DType::FP32, {c.streams, c.columns});
    if (c.combine) {
        infernix::DeviceBuffer d_outputs = t::to_device(c.outputs), d_weights = t::to_device(c.weights),
                               d_gate = t::to_device(c.gate), d_shared = t::to_device(c.shared);
        infernix::DeviceBuffer d_ids(static_cast<std::size_t>(c.top_k) * T * sizeof(std::int32_t));
        d_ids.fill(0);
        const infernix::ops::MoeRouting routing{
            Tensor(d_ids.p, DType::I32, {c.top_k, c.columns}), Tensor(d_weights.p, DType::FP32, {c.top_k, c.columns}),
            Tensor(d_gate.p, DType::FP32, {c.columns}), {}};
        infernix::ops::moe_combine_inject(Tensor(d_outputs.p, DType::BF16, {c.hidden, c.columns * c.top_k}), routing,
                                          Tensor(d_shared.p, DType::BF16, {c.hidden, c.columns}), inject, out, nullptr);
        t::cuda_synchronize();
    } else {
        infernix::DeviceBuffer d_y = t::to_device(c.y);
        infernix::ops::hyper_connection_inject(Tensor(d_y.p, DType::BF16, {c.hidden, c.columns}), inject, out, nullptr);
        t::cuda_synchronize();
    }
    check(t::from_device<std::uint16_t>(residual, c.residual.size()) == plain,
          label + ": an inactive cut is not the plain inject");
    (void)H;
}

// The combine form's y is moe_combine's: its result equals moe_combine followed by the cut inject of that BF16 y.
void combine_matches_composition(const Case& c, const std::string& label) {
    const std::vector<std::uint16_t> fused = run(c, false, label);
    const std::size_t H = static_cast<std::size_t>(c.hidden), T = static_cast<std::size_t>(c.columns);
    infernix::DeviceBuffer d_outputs = t::to_device(c.outputs), d_weights = t::to_device(c.weights),
                           d_gate = t::to_device(c.gate), d_shared = t::to_device(c.shared);
    infernix::DeviceBuffer d_ids(static_cast<std::size_t>(c.top_k) * T * sizeof(std::int32_t));
    d_ids.fill(0);
    infernix::DeviceBuffer d_y(H * T * sizeof(std::uint16_t));
    const infernix::ops::MoeRouting routing{
        Tensor(d_ids.p, DType::I32, {c.top_k, c.columns}), Tensor(d_weights.p, DType::FP32, {c.top_k, c.columns}),
        Tensor(d_gate.p, DType::FP32, {c.columns}), {}};
    Tensor y(d_y.p, DType::BF16, {c.hidden, c.columns});
    infernix::ops::moe_combine(Tensor(d_outputs.p, DType::BF16, {c.hidden, c.columns * c.top_k}), routing,
                               Tensor(d_shared.p, DType::BF16, {c.hidden, c.columns}), y, nullptr);
    t::cuda_synchronize();
    Case composed  = c;
    composed.combine = false;
    composed.y       = t::from_device<std::uint16_t>(d_y, H * T);
    check(run(composed, false, label + " composed") == fused,
          label + ": the combine form differs from moe_combine followed by the cut inject");
}

// residual_delta_cut (Qwen3.5's single-stream residual), after a lane wrote after = bf16(x0 + y) over before = x0:
// - the op's contract, pointwise against FP64: out = after - alpha c dir, c = dir . (after - before), with the same
//   bound as the cut inject at inject 1 (delta = after - before is exact);
// - the lane's ideal x0 + y - alpha (dir . y) dir, per column in L2: the error is the write's own rounding r with its
//   component along dir scaled by 1 - alpha, plus the last rounding, so
//     |out - ideal| <= max(1, |1 - alpha|) |2^-8 after| + |2^-8 out| + the FP32 dot's error,
//   at realistic ratios (|x0| well above |y|, and a few residual elements in the thousands, as Qwen3.5's residual
//   has); the largest error against the write's own rounding bound is printed;
// - a cut too small to move an element keeps its bits (alpha 1e-30: out = after), an inactive cut leaves x
//   unchanged, a column's bits whatever T or its position, a graph replay and a repeat give the eager bits, before and
//   the guards are untouched.
struct SingleCase {
    int hidden = 5120, columns = 1;
    float alpha = 1.0F;
    std::vector<float> dir;
    std::vector<double> y;                    // the lane's output (FP64), for the ideal
    std::vector<std::uint16_t> before, after; // BF16 x0 and bf16(x0 + y)
};

SingleCase make_single(int hidden, int columns, float alpha, std::uint32_t seed) {
    SingleCase c{.hidden = hidden, .columns = columns, .alpha = alpha};
    Rng rng{seed};
    const std::size_t H = static_cast<std::size_t>(hidden), n = H * columns;
    c.y.resize(n);
    c.before.resize(n);
    c.after.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        // A residual element up to 16, four per column in the thousands; a lane output a few times smaller.
        const bool outlier = i % H < 4;
        const double x0    = outlier ? std::ldexp(1.0 + 0.5 * rng.unit(), 10 + rng.below(2)) * (rng.unit() < 0 ? -1 : 1)
                                     : std::ldexp(rng.unit(), rng.below(9) - 4);
        c.before[i]        = t::f32_to_bf16(static_cast<float>(x0));
        double y           = std::ldexp(rng.unit(), rng.below(9) - 6);
        double sum         = t::bf16_to_f32(c.before[i]) + y;
        if (sum == 0.0) { // no exact zero: a cut moves a zero element however small it is
            y += 0.5;
            sum = t::bf16_to_f32(c.before[i]) + y;
        }
        c.after[i] = t::f32_to_bf16(static_cast<float>(sum));
        c.y[i]     = y;
    }
    c.dir = make_direction(hidden, rng, nullptr);
    return c;
}

std::vector<std::uint16_t> run_single(const SingleCase& c, bool graph, const std::string& label) {
    infernix::DeviceBuffer d_dir = t::to_device(c.dir), d_before = t::to_device(c.before);
    t::GuardedDeviceBuffer x(c.after.size() * sizeof(std::uint16_t));
    x.copy_from_host(c.after.data(), x.bytes());
    const Tensor before(d_before.p, DType::BF16, {c.hidden, c.columns});
    Tensor out(x.data(), DType::BF16, {c.hidden, c.columns});
    const DirectionCut cut{.direction = c.alpha != 0.0F ? static_cast<const float*>(d_dir.p) : nullptr, .alpha = c.alpha};
    const auto launch = [&](cudaStream_t stream) { infernix::ops::residual_delta_cut(before, cut, out, stream); };
    if (graph) {
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
    } else {
        launch(nullptr);
        t::cuda_synchronize();
    }
    failures += x.verify_guards(label + " guards");
    check(t::from_device<std::uint16_t>(d_before, c.before.size()) == c.before, label + ": before modified");
    return t::from_device<std::uint16_t>(x.data(), c.after.size());
}

double worst_single_ratio = 0.0; // the largest |out - ideal| against the write's own rounding bound, over every case

void verify_single(const SingleCase& c, const std::vector<std::uint16_t>& got, const std::string& label) {
    const std::size_t H = static_cast<std::size_t>(c.hidden);
    const double a      = c.alpha;
    for (int col = 0; col < c.columns; ++col) {
        double dot = 0.0, magnitude = 0.0, ideal_dot = 0.0;
        for (std::size_t h = 0; h < H; ++h) {
            const std::size_t at = col * H + h;
            const double delta   = static_cast<double>(t::bf16_to_f32(c.after[at])) - t::bf16_to_f32(c.before[at]);
            dot += static_cast<double>(c.dir[h]) * delta;
            magnitude += std::abs(static_cast<double>(c.dir[h]) * delta);
            ideal_dot += static_cast<double>(c.dir[h]) * c.y[at];
        }
        double error2 = 0.0, write2 = 0.0, out2 = 0.0;
        for (std::size_t h = 0; h < H; ++h) {
            const std::size_t at = col * H + h;
            const double after = t::bf16_to_f32(c.after[at]), d = c.dir[h];
            const double ref   = after - a * dot * d;
            const double cut_d = a * std::abs(d);
            const double e     = std::ldexp(std::abs(after) + cut_d * std::abs(dot), -20) + cut_d * std::ldexp(magnitude, -17);
            const double value = t::bf16_to_f32(got[at]);
            if (!std::isfinite(value) || !(std::abs(value - ref) <= std::ldexp(std::abs(ref), -8) + e)) {
                check(false, label + ": column " + std::to_string(col) + " row " + std::to_string(h) + " is " +
                                 std::to_string(value) + ", expected " + std::to_string(ref));
                return;
            }
            const double ideal = t::bf16_to_f32(c.before[at]) + c.y[at] - a * ideal_dot * d;
            error2 += (value - ideal) * (value - ideal);
            write2 += std::ldexp(after, -8) * std::ldexp(after, -8);
            out2 += std::ldexp(value, -8) * std::ldexp(value, -8);
        }
        // after was rounded through FP32 first (2^-24 more), hence the slop on the write's bound.
        const double write = std::sqrt(write2) * (1.0 + std::ldexp(1.0, -10));
        const double bound = std::max(1.0, std::abs(1.0 - a)) * write + std::sqrt(out2) + a * std::ldexp(magnitude, -17);
        worst_single_ratio = std::max(worst_single_ratio, std::sqrt(error2) / write);
        if (!(std::sqrt(error2) <= bound)) {
            check(false, label + ": column " + std::to_string(col) + " is " + std::to_string(std::sqrt(error2)) +
                             " from the ideal in L2, past " + std::to_string(bound));
            return;
        }
    }
}

void single_case(int hidden, int columns, float alpha, std::uint32_t seed, bool invariance) {
    const SingleCase c   = make_single(hidden, columns, alpha, seed);
    const std::string label = "residual delta cut H=" + std::to_string(hidden) + " T=" + std::to_string(columns) +
                              " alpha " + std::to_string(alpha);
    const std::vector<std::uint16_t> eager = run_single(c, false, label);
    verify_single(c, eager, label);
    check(run_single(c, false, label + " repeat") == eager, label + ": a repeated call differs");
    check(run_single(c, true, label + " graph") == eager, label + ": the graph replay differs");
    if (invariance && columns > 1) {
        const std::size_t H = static_cast<std::size_t>(hidden);
        for (const int col : {0, columns / 2, columns - 1}) {
            SingleCase one = c;
            one.columns = 1;
            const auto from = static_cast<std::ptrdiff_t>(col * H), to = static_cast<std::ptrdiff_t>((col + 1) * H);
            one.y.assign(c.y.begin() + from, c.y.begin() + to);
            one.before.assign(c.before.begin() + from, c.before.begin() + to);
            one.after.assign(c.after.begin() + from, c.after.begin() + to);
            check(run_single(one, false, label + " alone") ==
                      std::vector<std::uint16_t>(eager.begin() + from, eager.begin() + to),
                  label + ": column " + std::to_string(col) + " differs from its T = 1 call");
        }
    }
    SingleCase off = c;
    off.alpha = 0.0F;
    check(run_single(off, false, label + " inactive") == c.after, label + ": an inactive cut changed x");
    SingleCase tiny = c;
    tiny.alpha = 1e-30F;
    check(run_single(tiny, false, label + " alpha 1e-30") == c.after, label + ": a negligible cut changed x");
}

void refusals() {
    for (const int hidden : {5116, 8200}) {
        SingleCase bad = make_single(hidden, 2, 1.0F, 7U);
        try {
            (void)run_single(bad, false, "refused");
            check(false, "residual delta cut at H=" + std::to_string(hidden) + " was not refused");
        } catch (const std::invalid_argument&) {}
    }
    Case c = make_case(4, 2560, 2, 8, 1.0F, false, false, 9U);
    const auto refused = [&](const std::string& what, const auto& mutate) {
        Case bad = c;
        mutate(bad);
        try {
            (void)run(bad, false, what);
        } catch (const std::invalid_argument&) { return; }
        check(false, what + " was not refused");
    };
    refused("alpha above 4", [](Case& b) { b.alpha = 4.5F; });
    refused("a negative alpha", [](Case& b) { b.alpha = -1.0F; });
    for (const int hidden : {2556, 4104}) {
        refused("a hidden width of " + std::to_string(hidden) + " (not a multiple of 8, or past 4096)", [&](Case& b) {
            b.hidden = hidden;
            b.dir.resize(static_cast<std::size_t>(hidden));
            b.y.resize(static_cast<std::size_t>(hidden) * b.columns);
            b.residual.resize(static_cast<std::size_t>(b.streams) * hidden * b.columns);
        });
    }
}

} // namespace

INFERNIX_OP_TEST(direction_cut) {
    if (t::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        std::uint32_t seed = 0xA11CE5U;
        for (const int columns : {1, 2, 31, 32, 33, 257}) {
            for (const bool combine : {false, true}) {
                const Case c = make_case(4, 2560, columns, 8, 1.0F, combine, false, ++seed);
                run_case(c, std::string(combine ? "combine" : "inject") + " T=" + std::to_string(columns), true);
            }
        }
        for (const float alpha : {0.5F, 1.3F, 4.0F}) {
            run_case(make_case(4, 2560, 5, 8, alpha, false, false, ++seed), "inject alpha " + std::to_string(alpha), false);
            run_case(make_case(4, 2560, 5, 8, alpha, true, false, ++seed), "combine alpha " + std::to_string(alpha), false);
        }
        run_case(make_case(4, 2560, 3, 8, 1.0F, false, true, ++seed), "inject, a direction nearly parallel to y", false);
        for (const int top_k : {1, 16}) {
            run_case(make_case(4, 2560, 7, top_k, 1.0F, true, false, ++seed), "combine top-" + std::to_string(top_k), true);
        }
        run_case(make_case(3, 200, 5, 4, 1.0F, false, false, ++seed), "inject H=200 S=3", true);
        run_case(make_case(3, 200, 5, 4, 1.0F, true, false, ++seed), "combine H=200 S=3", true);
        run_case(make_case(2, 16, 70001, 2, 1.0F, false, false, ++seed), "inject T=70001", false);
        run_case(make_case(2, 16, 70001, 2, 1.0F, true, false, ++seed), "combine T=70001", false);
        inactive_is_plain(make_case(4, 2560, 9, 8, 1.0F, false, false, ++seed), "inject");
        inactive_is_plain(make_case(4, 2560, 9, 8, 1.0F, true, false, ++seed), "combine");
        combine_matches_composition(make_case(4, 2560, 33, 8, 1.0F, true, false, ++seed), "combine T=33");
        // Qwen3.5's widths: the 27B (5120), the 122B (3072), the 35B (2048); a partial warp (200) and a wide call.
        for (const int hidden : {5120, 3072, 2048}) {
            for (const int columns : {1, 2, 33, 257}) { single_case(hidden, columns, 1.0F, ++seed, columns == 33); }
            single_case(hidden, 5, 2.0F, ++seed, false);
            single_case(hidden, 5, 4.0F, ++seed, false);
        }
        single_case(200, 5, 1.0F, ++seed, true);
        single_case(16, 70001, 1.0F, ++seed, false);
        std::printf("residual delta cut: the largest error against the lane's ideal is %.3g of the write's own rounding "
                    "bound (L2 per column)\n",
                    worst_single_ratio);
        refusals();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("direction cut: all checks passed\n");
    return 0;
}
