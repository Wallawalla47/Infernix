// NVFP4 weights with 8-bit activations (LinearPolicy::A8Only: `--a8-activations` on NVIDIA's NVFP4 artifacts).
//
//   - ops::linear, ops::linear_swiglu and ops::linear_add at the 27B's NVFP4 shapes, every route (the sliced-K tiles
//     through 64 columns, the 5,120-row projections' 64-row MMA tiles through 256, the 128 x 128 ring wider, each
//     full and with a partly filled last token tile), against an FP64 oracle over the codec: each column encoded on
//     the host by ops/ggml/ggml_a8.h's own functions (the int8 codes and FP32 slice scales the kernels read), the
//     weights decoded exactly from their E2M1 codes, E4M3 block scales and matrix coefficient, the products summed
//     in FP64.
//     The kernels add a block's exact P_b S_b times the slice scale into FP32 (one rounding a block) and round the
//     output once, so the criterion is a BF16 rounding plus an FP32 accumulation bound over the |terms|.
//   - gdn_input_proj and attn_input_proj (the split QKV/Z and Q/K/gate/V outputs) bit for bit against ops::linear's
//     8-bit route over the same parent: the same tiles and sums written through another output.
//   - Workspace queries and the refusal of A8Only for a format without an 8-bit route.
#include "ops/linear/linear_test_common.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include "infernix/ops/attn_input_proj.h"
#include "infernix/ops/gdn_input_proj.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/linear_add.h"
#include "infernix/ops/linear_swiglu.h"

#include "core/arena.h"
#include "core/device.h"
#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_a8.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test;
using namespace infernix::test::linear;
namespace a8 = infernix::ggml::a8;

constexpr float kMultiplier = 7.011777e-02F; // a weight_scale_2 that is not a power of two

// The 8-bit codec's view of BF16 columns [K, T] (host, the kernels' own functions).
struct Encoded {
    std::vector<int> q;      // [T][K]
    std::vector<double> d;   // [T][K / 32]
};

Encoded encode(const std::vector<std::uint16_t>& x, int k, int t) {
    Encoded e;
    e.q.resize(static_cast<std::size_t>(k) * t);
    e.d.resize(static_cast<std::size_t>(k / 32) * t);
    for (int c = 0; c < t; ++c) {
        for (int s = 0; s < k / 32; ++s) {
            const std::uint16_t* v = x.data() + static_cast<std::size_t>(c) * k + 32 * s;
            unsigned key            = 0;
            for (int j = 0; j < 32; ++j) { key = std::max(key, static_cast<unsigned>(v[j] & 0x7FFFU)); }
            const a8::SliceScale sc = a8::slice_scale(key);
            e.d[static_cast<std::size_t>(c) * (k / 32) + s] = sc.d;
            for (int j = 0; j < 32; ++j) {
                e.q[static_cast<std::size_t>(c) * k + 32 * s + j] = a8::code(v[j], sc.inv);
            }
        }
    }
    return e;
}

// The packed weight's exact values: E2M1 code times its E4M3 block scale (the matrix coefficient left out).
struct HostNvfp4 {
    int n, k;
    const quantized_weight::PackedWeight* packed;

    double element(int row, int col) const {
        static const double kE2m1[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
        const std::uint8_t byte = packed->payload[static_cast<std::size_t>(row) * (k / 2) + col / 2];
        const int code          = (col & 1) ? byte >> 4 : byte & 0xF;
        const double value      = (code & 8 ? -1.0 : 1.0) * kE2m1[code & 7];
        const int block = col / 16, row_tile = row / 128, row_inner = row % 128;
        const std::size_t offset = packed->scale_plane_offset +
                                   static_cast<std::size_t>(row_tile * (k / 64) + block / 4) * 512 +
                                   static_cast<std::size_t>(row_inner % 32) * 16 +
                                   static_cast<std::size_t>(row_inner / 32) * 4 + static_cast<std::size_t>(block % 4);
        const double scale = std::ldexp(static_cast<double>(ops::canon::e4m3_scaled(packed->payload[offset])), -9);
        return value * scale;
    }
};

// y[row, column] in FP64 and the sum of the |terms| (the accumulation bound's scale).
struct Reference {
    double y, magnitude;
};

Reference reference(const HostNvfp4& w, const Encoded& e, int row, int column) {
    double sum = 0, magnitude = 0;
    for (int s = 0; s < w.k / 32; ++s) {
        double slice = 0;
        for (int j = 0; j < 32; ++j) {
            slice += w.element(row, 32 * s + j) * e.q[static_cast<std::size_t>(column) * w.k + 32 * s + j];
        }
        const double term = slice * e.d[static_cast<std::size_t>(column) * (w.k / 32) + s];
        sum += term;
        magnitude += std::abs(term);
    }
    return {sum * kMultiplier, magnitude * kMultiplier};
}

bool close(double got, Reference ref) {
    // One BF16 rounding of the output and the FP32 sums (one rounding a block and the K-warp partials).
    return std::abs(got - ref.y) <= std::ldexp(std::abs(ref.y), -8) + std::ldexp(ref.magnitude, -18) + 1e-30;
}

double bf16(std::uint16_t bits) { return ops::canon::bf16_to_f32(bits); }

std::vector<int> sampled_rows(int n) {
    std::vector<int> rows;
    for (int r = 0; r < n; r += 97) { rows.push_back(r); }
    for (int r : {1, 7, 8, 15, 16, 127, 128, n / 2 - 1, n / 2, n - 1}) { rows.push_back(std::clamp(r, 0, n - 1)); }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

const std::vector<int> kWidths{1, 3, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129, 300};

quantized_weight::PackedWeight make_packed(int n, int k, std::uint32_t seed) {
    quantized_weight::PatternedWeightOptions options;
    options.weight_multiplier = kMultiplier;
    options.activation_scale  = 0.25F;
    return quantized_weight::make_patterned_weight(QType::NVFP4, n, k, seed, options);
}

// A multiplier-form NVFP4 parent on the device (its host payload kept for the oracle).
struct DeviceNvfp4 {
    quantized_weight::PackedWeight packed;
    DeviceBuffer payload;
    Weight weight;

    DeviceNvfp4(int n, int k, std::uint32_t seed) : packed(make_packed(n, k, seed)), payload(packed.payload.size()) {
        payload.copy_from_host(packed.payload.data(), payload.bytes);
        weight = packed.device_weight(payload.p);
    }
};

std::vector<std::uint16_t> make_x(int k, int t, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * t);
    for (auto& v : x) { v = f32_to_bf16(normal(rng)); }
    // A few outliers and a zero slice exercise the slice scales.
    if (t > 0) {
        x[5] = f32_to_bf16(37.5F);
        for (int j = 32; j < 64 && j < k; ++j) { x[j] = 0; }
    }
    return x;
}

template <class T>
std::vector<T> read(const DeviceBuffer& b, std::size_t count) {
    std::vector<T> out(count);
    cuda_check(cudaMemcpy(out.data(), b.p, count * sizeof(T), cudaMemcpyDeviceToHost), "read");
    return out;
}

int run_linear() {
    int failures = 0;
    std::uint32_t seed = 4401U;
    for (const auto& [n, k] : {std::pair{14336, 5120}, std::pair{16384, 5120}, std::pair{34816, 5120},
                               std::pair{5120, 6144}, std::pair{5120, 17408}}) {
        DeviceNvfp4 w(n, k, seed++);
        const int max = kWidths.back();
        const auto x  = make_x(k, max, seed++);
        DeviceBuffer input(x.size() * 2), output(static_cast<std::size_t>(n) * max * 2);
        input.copy_from_host(x.data(), input.bytes);
        DeviceArena workspace(std::max<std::size_t>(
            ops::linear_workspace_capacity_bytes(QType::NVFP4, n, k, ops::LinearPolicy::A8Only, 1, max), 256));
        const HostNvfp4 host{n, k, &w.packed};
        const auto rows = sampled_rows(n);
        for (const int t : kWidths) {
            const Encoded e = encode(x, k, t);
            Tensor tx(input.p, DType::BF16, {k, t});
            Tensor out(output.p, DType::BF16, {n, t});
            workspace.reset();
            ops::linear(tx, w.weight, out, ops::LinearPolicy::A8Only, workspace, nullptr);
            cuda_check(cudaDeviceSynchronize(), "linear A8");
            const auto got = read<std::uint16_t>(output, static_cast<std::size_t>(n) * t);
            int bad = 0;
            for (const int r : rows) {
                for (int c = 0; c < t; c += (t > 33 ? 7 : 1)) {
                    const Reference ref = reference(host, e, r, c);
                    if (!close(bf16(got[static_cast<std::size_t>(c) * n + r]), ref) && bad++ == 0) {
                        std::cerr << "FAIL: NVFP4 A8 linear [" << n << "," << k << "] T=" << t << " row " << r
                                  << " column " << c << ": " << bf16(got[static_cast<std::size_t>(c) * n + r])
                                  << " against " << ref.y << '\n';
                    }
                }
            }
            failures += bad != 0;
        }
    }
    return failures;
}

// The columns checked at a width: every one through 33, then every fifth (every eleventh past 129) and the last.
std::vector<int> sampled_columns(int t) {
    std::vector<int> columns;
    const int step = t > 129 ? 11 : t > 33 ? 5 : 1;
    for (int c = 0; c < t; c += step) { columns.push_back(c); }
    if (columns.back() != t - 1) { columns.push_back(t - 1); }
    return columns;
}

// The widths of the gate/up and the 5,120-row projections: the sliced-K tiles, the 5,120-row ones' MMA tiles (64 x 64
// and 64 x 128 through 256 columns) and the 128 x 128 ring (the gate/up from 65 columns, the others from 257), each
// full (128, 256, 384) and with a partly filled last token tile.
constexpr int kSwiGluAddWidths[] = {1, 8, 16, 17, 33, 65, 128, 129, 256, 257, 384};

int run_swiglu_and_add() {
    int failures = 0;
    constexpr int kMax = 384;
    // Gate/up [34816,5120]: h = silu(gate) * up of the two halves' rows.
    {
        constexpr int n = 34816, k = 5120, half = n / 2;
        DeviceNvfp4 w(n, k, 4501U);
        const auto x = make_x(k, kMax, 4502U);
        DeviceBuffer input(x.size() * 2), output(static_cast<std::size_t>(half) * kMax * 2);
        input.copy_from_host(x.data(), input.bytes);
        DeviceArena workspace(std::max<std::size_t>(
            ops::linear_swiglu_workspace_capacity_bytes(QType::NVFP4, n, k, ops::LinearPolicy::A8Only, 1, kMax), 256));
        const HostNvfp4 host{n, k, &w.packed};
        for (const int t : kSwiGluAddWidths) {
            const Encoded e = encode(x, k, t);
            Tensor tx(input.p, DType::BF16, {k, t});
            Tensor out(output.p, DType::BF16, {half, t});
            workspace.reset();
            ops::linear_swiglu(tx, w.weight, out, ops::LinearPolicy::A8Only, workspace, nullptr);
            cuda_check(cudaDeviceSynchronize(), "linear_swiglu A8");
            const auto got = read<std::uint16_t>(output, static_cast<std::size_t>(half) * t);
            int bad = 0;
            for (const int r : sampled_rows(half)) {
                for (const int c : sampled_columns(t)) {
                    const Reference g = reference(host, e, r, c), u = reference(host, e, r + half, c);
                    const double h    = g.y / (1.0 + std::exp(-g.y)) * u.y;
                    const double mag  = (std::abs(g.y) + g.magnitude) * (std::abs(u.y) + u.magnitude);
                    const double v    = bf16(got[static_cast<std::size_t>(c) * half + r]);
                    if (std::abs(v - h) > std::ldexp(std::abs(h), -7) + std::ldexp(mag, -17) + 1e-30 && bad++ == 0) {
                        std::cerr << "FAIL: NVFP4 A8 linear_swiglu T=" << t << " row " << r << " column " << c << ": "
                                  << v << " against " << h << '\n';
                    }
                }
            }
            failures += bad != 0;
        }
    }
    // Output and down [5120,6144], [5120,17408]: residual + y.
    for (const auto& [n, k] : {std::pair{5120, 6144}, std::pair{5120, 17408}}) {
        DeviceNvfp4 w(n, k, 4600U + static_cast<std::uint32_t>(k));
        const auto x = make_x(k, kMax, 4700U + static_cast<std::uint32_t>(k));
        std::vector<std::uint16_t> residual(static_cast<std::size_t>(n) * kMax);
        std::mt19937 rng(91U);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        for (auto& v : residual) { v = f32_to_bf16(normal(rng)); }
        DeviceBuffer input(x.size() * 2), output(residual.size() * 2);
        input.copy_from_host(x.data(), input.bytes);
        DeviceArena workspace(std::max<std::size_t>(
            ops::linear_add_workspace_capacity_bytes(QType::NVFP4, n, k, ops::LinearPolicy::A8Only, 1, kMax), 256));
        const HostNvfp4 host{n, k, &w.packed};
        for (const int t : kSwiGluAddWidths) {
            const Encoded e = encode(x, k, t);
            output.copy_from_host(residual.data(), static_cast<std::size_t>(n) * t * 2);
            Tensor tx(input.p, DType::BF16, {k, t});
            Tensor res(output.p, DType::BF16, {n, t});
            workspace.reset();
            ops::linear_add(tx, w.weight, res, ops::LinearPolicy::A8Only, workspace, nullptr);
            cuda_check(cudaDeviceSynchronize(), "linear_add A8");
            const auto got = read<std::uint16_t>(output, static_cast<std::size_t>(n) * t);
            int bad = 0;
            for (const int r : sampled_rows(n)) {
                for (const int c : sampled_columns(t)) {
                    Reference ref = reference(host, e, r, c);
                    ref.y += bf16(residual[static_cast<std::size_t>(c) * n + r]);
                    ref.magnitude += 1.0;
                    if (!close(bf16(got[static_cast<std::size_t>(c) * n + r]), ref) && bad++ == 0) {
                        std::cerr << "FAIL: NVFP4 A8 linear_add [" << n << "," << k << "] T=" << t << " row " << r
                                  << " column " << c << '\n';
                    }
                }
            }
            failures += bad != 0;
        }
    }
    return failures;
}

// The split outputs equal ops::linear's 8-bit route over the same parent bit for bit.
int run_split_outputs() {
    int failures = 0;
    constexpr int kMax = 257;
    for (const bool gdn : {true, false}) {
        const int n = gdn ? 16384 : 14336, k = 5120;
        DeviceNvfp4 w(n, k, gdn ? 4801U : 4802U);
        const auto x = make_x(k, kMax, 4803U);
        DeviceBuffer input(x.size() * 2), whole(static_cast<std::size_t>(n) * kMax * 2);
        input.copy_from_host(x.data(), input.bytes);
        const std::size_t bytes = std::max(
            ops::linear_workspace_capacity_bytes(QType::NVFP4, n, k, ops::LinearPolicy::A8Only, 1, kMax),
            gdn ? ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, n, k, ops::LinearPolicy::A8Only, 1, kMax)
                : ops::attn_input_proj_workspace_capacity_bytes(QType::NVFP4, n, k, ops::LinearPolicy::A8Only, 1,
                                                                kMax));
        DeviceArena workspace(std::max<std::size_t>(bytes, 256));
        // GDN: QKV rows [0, 10240), Z rows [10240, 16384). Attention: Q [0, 6144), K [6144, 7168), gate
        // [7168, 13312), V [13312, 14336) (the A16 tiles' segmented outputs).
        for (const int t : {1, 9, 33, 129, 257}) {
            Tensor tx(input.p, DType::BF16, {k, t});
            Tensor all(whole.p, DType::BF16, {n, t});
            workspace.reset();
            ops::linear(tx, w.weight, all, ops::LinearPolicy::A8Only, workspace, nullptr);
            const auto expected = read<std::uint16_t>(whole, static_cast<std::size_t>(n) * t);
            std::vector<std::uint16_t> joined(static_cast<std::size_t>(n) * t);
            if (gdn) {
                DeviceBuffer qkv(static_cast<std::size_t>(10240) * t * 2), z(static_cast<std::size_t>(6144) * t * 2);
                Tensor tq(qkv.p, DType::BF16, {10240, t}), tz(z.p, DType::BF16, {6144, t});
                workspace.reset();
                ops::gdn_input_proj(tx, w.weight, tq, tz, ops::LinearPolicy::A8Only, workspace, nullptr);
                const auto a = read<std::uint16_t>(qkv, static_cast<std::size_t>(10240) * t);
                const auto b = read<std::uint16_t>(z, static_cast<std::size_t>(6144) * t);
                for (int c = 0; c < t; ++c) {
                    std::copy_n(a.data() + static_cast<std::size_t>(c) * 10240, 10240,
                                joined.data() + static_cast<std::size_t>(c) * n);
                    std::copy_n(b.data() + static_cast<std::size_t>(c) * 6144, 6144,
                                joined.data() + static_cast<std::size_t>(c) * n + 10240);
                }
            } else {
                DeviceBuffer q(static_cast<std::size_t>(6144) * t * 2), g(static_cast<std::size_t>(6144) * t * 2),
                    kk(static_cast<std::size_t>(1024) * t * 2), v(static_cast<std::size_t>(1024) * t * 2);
                Tensor tq(q.p, DType::BF16, {6144, t}), tg(g.p, DType::BF16, {6144, t}),
                    tk(kk.p, DType::BF16, {1024, t}), tv(v.p, DType::BF16, {1024, t});
                workspace.reset();
                ops::attn_input_proj(tx, w.weight, tq, tg, tk, tv, ops::LinearPolicy::A8Only, workspace, nullptr);
                const auto aq = read<std::uint16_t>(q, static_cast<std::size_t>(6144) * t);
                const auto ag = read<std::uint16_t>(g, static_cast<std::size_t>(6144) * t);
                const auto ak = read<std::uint16_t>(kk, static_cast<std::size_t>(1024) * t);
                const auto av = read<std::uint16_t>(v, static_cast<std::size_t>(1024) * t);
                for (int c = 0; c < t; ++c) {
                    auto* col = joined.data() + static_cast<std::size_t>(c) * n;
                    std::copy_n(aq.data() + static_cast<std::size_t>(c) * 6144, 6144, col);
                    std::copy_n(ak.data() + static_cast<std::size_t>(c) * 1024, 1024, col + 6144);
                    std::copy_n(ag.data() + static_cast<std::size_t>(c) * 6144, 6144, col + 7168);
                    std::copy_n(av.data() + static_cast<std::size_t>(c) * 1024, 1024, col + 13312);
                }
            }
            if (joined != expected) {
                std::cerr << "FAIL: NVFP4 A8 " << (gdn ? "gdn_input_proj" : "attn_input_proj") << " T=" << t
                          << " differs from linear's 8-bit route\n";
                ++failures;
            }
        }
    }
    return failures;
}

int run_refusals() {
    int failures = 0;
    try {
        (void)ops::linear_workspace_capacity_bytes(QType::Q8_G32_FP16, 5120, 5120, ops::LinearPolicy::A8Only, 1, 1);
        std::cerr << "FAIL: A8Only admitted for a Q8 weight\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    if (ops::linear_workspace_capacity_bytes(QType::NVFP4, 5120, 6144, ops::LinearPolicy::A8Only, 1, 64) <
        static_cast<std::size_t>(ggml::a8::column_words(6144, false)) * 4 * 64) {
        std::cerr << "FAIL: the A8 workspace does not hold the encoded columns\n";
        ++failures;
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(linear_nvfp4_a8) {
    int failures = run_refusals();
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return failures == 0 ? 77 : 1;
    }
    try {
        failures += run_linear();
        failures += run_swiglu_and_add();
        failures += run_split_outputs();
    } catch (const std::exception& error) {
        std::cerr << "NVFP4 A8: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4 8-bit activations\n";
    return failures == 0 ? 0 : 1;
}
