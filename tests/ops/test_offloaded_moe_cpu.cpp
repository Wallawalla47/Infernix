// Canonical W4A4 arithmetic and the CPU narrow route of offloaded_sparse_moe
// (docs/maintainer/qwen3_8-flash-next-design.md §16.2). Host-only: needs no GPU.
//
// Oracles:
//   - E4M3FN / E2M1 encoders against nearest-value enumeration of the code grids;
//   - exp_c / SiLU against binary64 libm, exhaustively over BF16 inputs;
//   - every CPU ISA against the scalar reference, bit for bit;
//   - the int64 row sums against an independent binary64 evaluation of the decoded operands,
//     which is exact while |S| < 2^53, and the BF16 outputs against an FP64 expert oracle.

#include "ops/common/canonical_math.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"
#include "ops/offloaded_moe_fixtures.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace canon = ninfer::ops::canon;
namespace moe   = ninfer::ops::offloaded_moe;
namespace fixtures = ninfer::test::offloaded_moe;
using fixtures::Expert;
using fixtures::random_activations;
using fixtures::random_expert;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

float bits_to_float(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

std::uint32_t float_to_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// ---------------------------------------------------------------------------- code grids

std::vector<double> e4m3_grid() { // value of words 0..0x7E
    std::vector<double> v(127);
    for (int w = 0; w < 127; ++w) {
        const int e = (w >> 3) & 15, m = w & 7;
        v[w]        = e == 0 ? m * std::ldexp(1.0, -9) : (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
    }
    return v;
}

int nearest_even(const std::vector<double>& grid, double x) {
    // Grid is increasing; returns the index of the nearest value, ties to the even index.
    if (x >= grid.back()) { return static_cast<int>(grid.size()) - 1; }
    const auto it = std::upper_bound(grid.begin(), grid.end(), x);
    const int hi  = static_cast<int>(it - grid.begin());
    const int lo  = hi - 1;
    const double dl = x - grid[lo], dh = grid[hi] - x;
    if (dl < dh) { return lo; }
    if (dh < dl) { return hi; }
    return (lo % 2 == 0) ? lo : hi;
}

void test_e4m3() {
    const auto grid = e4m3_grid();
    for (int w = 0; w < 127; ++w) {
        check(static_cast<double>(canon::e4m3_value(w)) == grid[w], "e4m3_value matches grid");
        check(canon::e4m3_scaled(w) == static_cast<std::int32_t>(grid[w] * 512.0), "e4m3_scaled = value * 2^9");
        check(canon::e4m3_rn_satfinite(static_cast<float>(grid[w])) == w, "e4m3 encode of a grid value");
    }
    std::vector<float> inputs;
    for (int w = 0; w < 126; ++w) {
        const float mid = static_cast<float>((grid[w] + grid[w + 1]) / 2.0); // exact in binary32
        std::uint32_t u = float_to_bits(mid);
        for (int k = -8; k <= 8; ++k) { inputs.push_back(bits_to_float(u + k)); }
    }
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> e(-40.0F, 10.0F);
    for (int i = 0; i < 2'000'000; ++i) { inputs.push_back(std::exp2(e(rng))); }
    inputs.push_back(0.0F);
    inputs.push_back(1.0e-30F);
    inputs.push_back(447.99F);
    inputs.push_back(448.0F);
    inputs.push_back(1.0e30F);
    inputs.push_back(std::numeric_limits<float>::infinity());
    int bad = 0;
    for (float x : inputs) {
        const int expect = nearest_even(grid, x);
        if (canon::e4m3_rn_satfinite(x) != expect) { ++bad; }
    }
    check(bad == 0, "e4m3_rn_satfinite = nearest-even saturating encode");
}

void test_e2m1() {
    const std::vector<double> grid = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    for (int c = 0; c < 16; ++c) {
        const double v = (c & 8 ? -1 : 1) * grid[c & 7];
        check(canon::e2m1_x2(c) == static_cast<int>(2 * v), "e2m1_x2 decode");
    }
    std::vector<float> inputs;
    for (int i = 0; i < 7; ++i) {
        const float mid = static_cast<float>((grid[i] + grid[i + 1]) / 2.0);
        for (int k = -8; k <= 8; ++k) { inputs.push_back(bits_to_float(float_to_bits(mid) + k)); }
    }
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(0.0F, 9.0F);
    for (int i = 0; i < 1'000'000; ++i) { inputs.push_back(u(rng)); }
    int bad = 0;
    for (float a : inputs) {
        for (float x : {a, -a}) {
            const int mag   = nearest_even(grid, std::fabs(static_cast<double>(x)));
            const unsigned c = canon::e2m1_rn_satfinite(x);
            if ((c & 7U) != static_cast<unsigned>(mag) || ((c & 8U) != 0) != std::signbit(x)) { ++bad; }
        }
    }
    check(bad == 0, "e2m1_rn_satfinite = nearest-even saturating encode with sign");
}

double ulp_distance(float a, double exact) {
    if (std::isinf(exact) || exact > std::numeric_limits<float>::max()) { return std::isinf(a) ? 0.0 : 1e9; }
    const double ulp = std::ldexp(1.0, std::max(std::ilogb(static_cast<float>(exact)), -126) - 23);
    return std::fabs(static_cast<double>(a) - exact) / ulp;
}

void test_exp_silu() {
    double worst_exp = 0, worst_silu = 0;
    for (std::uint32_t h = 0; h < 65536; ++h) {
        const float x = canon::bf16_to_f32(static_cast<std::uint16_t>(h));
        if (!std::isfinite(x)) { continue; }
        const double ex = std::exp(static_cast<double>(x));
        if (ex >= std::ldexp(1.0, -126) && x <= 88.7F) { worst_exp = std::max(worst_exp, ulp_distance(canon::exp_c(x), ex)); }
        const double s = static_cast<double>(x) / (1.0 + std::exp(-static_cast<double>(x)));
        const float got = canon::bf16_to_f32(canon::swiglu_bf16(static_cast<std::uint16_t>(h), 0x3F80)); // * 1.0
        const float ref = canon::bf16_to_f32(canon::f32_to_bf16_rn(static_cast<float>(s)));
        // BF16 rounding of a value within 2 FP32 ulps can differ from the reference by one BF16 ulp.
        if (std::isfinite(ref) && std::fabs(ref) >= std::ldexp(1.0F, -126)) { // normal results
            const double bf_ulp = std::ldexp(1.0, std::ilogb(ref) - 7);
            worst_silu          = std::max(worst_silu, std::fabs(static_cast<double>(got) - ref) / bf_ulp);
        }
    }
    for (float x = -103.0F; x < 88.7F; x = std::nextafter(x, 100.0F) + 0.0013F) {
        const double ex = std::exp(static_cast<double>(x));
        if (ex >= std::ldexp(1.0, -126)) { worst_exp = std::max(worst_exp, ulp_distance(canon::exp_c(x), ex)); }
    }
    std::printf("exp_c worst error %.3f ulp; SiLU worst error %.3f BF16 ulp\n", worst_exp, worst_silu);
    check(worst_exp <= 2.0, "exp_c within 2 ulp of exp");
    check(worst_silu <= 1.0, "SiLU within 1 BF16 ulp of the correctly rounded value");
    check(canon::exp_c(100.0F) == std::numeric_limits<float>::infinity(), "exp_c overflow");
    check(canon::exp_c(-200.0F) == 0.0F, "exp_c underflow");
    check(canon::swiglu_bf16(0xC780 /* -65536 */, 0x3F80) == 0x8000, "SiLU of a large negative is -0");
}

void test_bf16_round() {
    std::mt19937 rng(3);
    for (int i = 0; i < 1'000'000; ++i) {
        const std::uint32_t u = rng() % 0x7F7F0000U; // finite, below BF16 overflow
        const float x         = bits_to_float(u);
        const std::uint16_t h = canon::f32_to_bf16_rn(x);
        const double lo = canon::bf16_to_f32(h), d = std::fabs(static_cast<double>(x) - lo);
        const double next = canon::bf16_to_f32(static_cast<std::uint16_t>(h + 1));
        const double prev = canon::bf16_to_f32(static_cast<std::uint16_t>(h - 1));
        if (d > std::fabs(static_cast<double>(x) - next) || d > std::fabs(static_cast<double>(x) - prev)) {
            check(false, "f32_to_bf16_rn is nearest");
            break;
        }
    }
}

// ---------------------------------------------------------------------------- quantizer

using fixtures::to_bf16;

void test_quantizer() {
    std::uint16_t zero[16] = {};
    const auto z = canon::quantize_a4_block(zero, 0.01F);
    check(z.scale_scaled == 0 && z.c2_sum == 0, "zero block");
    std::mt19937 rng(5);
    std::normal_distribution<float> n(0.0F, 1.0F);
    int bad = 0;
    for (int it = 0; it < 200000; ++it) {
        const float g  = std::exp2(std::uniform_real_distribution<float>(-12.0F, -2.0F)(rng));
        const float sc = std::exp2(std::uniform_real_distribution<float>(-8.0F, 8.0F)(rng));
        std::uint16_t v[16];
        for (auto& e : v) { e = to_bf16(n(rng) * sc); }
        const auto q   = canon::quantize_a4_block(v, g);
        const float d  = canon::e4m3_value(q.scale_word) * g;
        int sum        = 0;
        for (int j = 0; j < 16; ++j) {
            sum += q.c2[j];
            if (q.scale_scaled == 0) { continue; }
            const double x = canon::bf16_to_f32(v[j]), r = q.c2[j] / 2.0 * d;
            // Unsaturated: the code is the nearest grid point of x / d, so the error is at most half
            // the local grid spacing (1 below 2, 2 below 4, else 2 up to the 6 saturation).
            const double xd = std::fabs(x / d);
            const double spacing = xd < 2 ? 0.5 : (xd < 4 ? 1.0 : 2.0);
            // Beyond 6 (a block scale that rounded down, or saturated at 448) the code clamps to +-6.
            if (xd > 6.0) {
                if (std::abs(q.c2[j]) != 12) { ++bad; }
            } else if (std::fabs(x - r) > (spacing / 2) * d * (1 + 1e-6)) {
                ++bad;
            }
        }
        if (sum != q.c2_sum) { ++bad; }
    }
    check(bad == 0, "A4 codes are nearest grid points and c2_sum is their sum");
    // Saturation: amax / (6 g) > 448 clamps the scale to 448 and the codes to +-6.
    std::uint16_t big[16];
    for (int j = 0; j < 16; ++j) { big[j] = to_bf16(j % 2 ? 1000.0F : -1000.0F); }
    const auto s = canon::quantize_a4_block(big, 0.01F);
    check(s.scale_word == 0x7E && s.c2[0] == -12 && s.c2[1] == 12, "saturating block");
}

// ---------------------------------------------------------------------------- experts

std::vector<std::uint16_t> run(moe::CpuIsa isa, const Expert& e, const std::vector<std::uint16_t>& x, int ncols) {
    std::vector<std::uint16_t> y(static_cast<std::size_t>(ncols) * moe::kHidden);
    const std::uint16_t* xp[moe::kMaxColumns];
    std::uint16_t* yp[moe::kMaxColumns];
    for (int c = 0; c < ncols; ++c) {
        xp[c] = &x[static_cast<std::size_t>(c) * moe::kHidden];
        yp[c] = &y[static_cast<std::size_t>(c) * moe::kHidden];
    }
    moe::expert_forward(isa, e.record.data(), e.scales, ncols, xp, yp);
    return y;
}

void test_isa_equality() {
    std::mt19937 rng(42);
    for (moe::CpuIsa isa : {moe::CpuIsa::kAvx2, moe::CpuIsa::kAvxVnni, moe::CpuIsa::kAvx512Vnni}) {
        std::printf("CPU ISA %-12s %s\n", moe::cpu_isa_name(isa), moe::cpu_isa_supported(isa) ? "tested" : "not supported on this host (skipped)");
    }
    for (int trial = 0; trial < 4; ++trial) {
        const Expert e = random_expert(rng, trial == 3);
        for (int ncols : {1, 2, 3, 4, 5, 8}) {
            const auto x   = random_activations(rng, ncols);
            const auto ref = run(moe::CpuIsa::kScalar, e, x, ncols);
            for (moe::CpuIsa isa : {moe::CpuIsa::kAvx2, moe::CpuIsa::kAvxVnni, moe::CpuIsa::kAvx512Vnni}) {
                if (!moe::cpu_isa_supported(isa)) { continue; }
                check(run(isa, e, x, ncols) == ref, "ISA output bit-identical to the scalar reference");
            }
            // Per-column independence: a column's result does not depend on its batch.
            if (ncols > 1) {
                const std::vector<std::uint16_t> x0(x.begin(), x.begin() + moe::kHidden);
                const auto y0 = run(moe::best_cpu_isa(), e, x0, 1);
                check(std::equal(y0.begin(), y0.end(), ref.begin()), "column result independent of batch width");
            }
        }
    }
}

void test_split_invariance() {
    std::mt19937 rng(9);
    const Expert e = random_expert(rng, false);
    const int ncols = 3;
    const auto x    = random_activations(rng, ncols);
    const auto ref  = run(moe::CpuIsa::kScalar, e, x, ncols);
    const moe::CpuIsa isa = moe::best_cpu_isa();
    // Phase A and B split into uneven worker ranges, in reverse order.
    std::vector<canon::A4Block> xq(static_cast<std::size_t>(ncols) * moe::kGateUpBlocks), hb(static_cast<std::size_t>(ncols) * moe::kHBlocks);
    const canon::A4Block* xp[8];
    canon::A4Block* hp[8];
    const canon::A4Block* hcp[8];
    std::vector<std::uint16_t> y(static_cast<std::size_t>(ncols) * moe::kHidden);
    std::uint16_t* yp[8];
    for (int c = 0; c < ncols; ++c) {
        moe::quantize_a4(&x[static_cast<std::size_t>(c) * moe::kHidden], moe::kHidden, e.scales.input_gate, &xq[static_cast<std::size_t>(c) * moe::kGateUpBlocks]);
        xp[c] = &xq[static_cast<std::size_t>(c) * moe::kGateUpBlocks];
        hp[c] = &hb[static_cast<std::size_t>(c) * moe::kHBlocks];
        hcp[c]          = hp[c];
        yp[c]           = &y[static_cast<std::size_t>(c) * moe::kHidden];
    }
    const int ucuts[] = {40, 29, 13, 3, 0};
    for (int i = 0; i + 1 < 5; ++i) { moe::gate_up_units(isa, e.record.data(), e.scales, xp, xp, ncols, ucuts[i + 1], ucuts[i], hp); }
    const int rcuts[] = {160, 111, 57, 1, 0};
    for (int i = 0; i + 1 < 5; ++i) { moe::down_rows(isa, e.record.data(), e.scales, hcp, ncols, rcuts[i + 1], rcuts[i], yp); }
    check(y == ref, "worker splits do not change a bit");
}

// Exactness of the int64 row sums, and the BF16 output against an FP64 expert oracle.
void test_fp64_oracle() {
    std::mt19937 rng(21);
    const Expert e  = random_expert(rng, false);
    const auto x    = random_activations(rng, 1);
    std::vector<canon::A4Block> xq(moe::kGateUpBlocks);
    moe::quantize_a4(x.data(), moe::kHidden, e.scales.input_gate, xq.data());
    const canon::A4Block* xp[1] = {xq.data()};
    std::vector<std::int64_t> s(static_cast<std::size_t>(moe::kGateUpRowGroups) * 16);
    moe::rg16_row_sums(moe::best_cpu_isa(), e.record.data(), moe::kGateUpBlocks, 0, moe::kGateUpRowGroups, xp, 1, s.data());
    int inexact = 0, checked = 0;
    for (int row = 0; row < moe::kGateUpRowGroups * 16; ++row) {
        const int rg = row / 16, r = row % 16;
        double sum = 0; // decoded operands (weights without weight_scale_2, activations without g)
        for (int b = 0; b < moe::kGateUpBlocks; ++b) {
            const std::uint8_t* unit = &e.record[(static_cast<std::size_t>(rg) * moe::kGateUpBlocks + b) * moe::kUnitBytes];
            for (int k = 0; k < 16; ++k) {
                const std::uint8_t byte = unit[32 * (k / 4) + 4 * (r % 8) + k % 4];
                const unsigned code     = r < 8 ? (byte & 15U) : (byte >> 4);
                const double w = canon::e2m1_x2(code) / 2.0 * static_cast<double>(canon::e4m3_value(unit[128 + r]));
                const double a = xq[b].c2[k] / 2.0 * static_cast<double>(canon::e4m3_value(xq[b].scale_word));
                sum += w * a; // every partial sum is a multiple of 2^-20 below 2^53: exact
            }
        }
        if (std::fabs(sum) < std::ldexp(1.0, 30)) {
            ++checked;
            if (sum != std::ldexp(static_cast<double>(s[row]), -20)) { ++inexact; }
        }
    }
    check(checked > 1000 && inexact == 0, "int64 row sums equal the exact decoded dot products");
    // Expert output against FP64 with the same A4 activations: two roundings (FP32, BF16) per output.
    const auto y = run(moe::CpuIsa::kScalar, e, x, 1);
    std::vector<canon::A4Block> hq(moe::kHBlocks);
    const canon::A4Block* hp_c[1] = {hq.data()};
    canon::A4Block* hp[1]         = {hq.data()};
    moe::gate_up_units(moe::CpuIsa::kScalar, e.record.data(), e.scales, xp, xp, 1, 0, moe::kHBlocks, hp);
    int bad = 0;
    for (int row = 0; row < moe::kHidden; ++row) {
        const int rg = row / 16, r = row % 16;
        double acc = 0, mag = 0;
        for (int b = 0; b < moe::kDownBlocks; ++b) {
            const std::uint8_t* unit = &e.record[moe::kGateUpBytes + (static_cast<std::size_t>(rg) * moe::kDownBlocks + b) * moe::kUnitBytes];
            for (int k = 0; k < 16; ++k) {
                const std::uint8_t byte = unit[32 * (k / 4) + 4 * (r % 8) + k % 4];
                const unsigned code     = r < 8 ? (byte & 15U) : (byte >> 4);
                const double w = canon::e2m1_x2(code) / 2.0 * static_cast<double>(canon::e4m3_value(unit[128 + r]));
                const double a = hq[b].c2[k] / 2.0 * static_cast<double>(canon::e4m3_value(hq[b].scale_word));
                acc += w * a;
                mag += std::fabs(w * a);
            }
        }
        const double ref = acc * static_cast<double>(e.scales.alpha_down);
        const double got = canon::bf16_to_f32(y[row]);
        // BF16 output: half a BF16 ulp of the result plus the FP32 conversion of the sum.
        const double tol = std::ldexp(std::fabs(ref), -8) + mag * static_cast<double>(e.scales.alpha_down) * std::ldexp(1.0, -23) + 1e-30;
        if (std::fabs(got - ref) > tol) { ++bad; }
    }
    (void)hp_c;
    check(bad == 0, "expert output within BF16 rounding of the FP64 oracle");
}


std::uint64_t golden_hash(moe::CpuIsa isa, int ncols) {
    const auto g = fixtures::golden_case(ncols);
    const auto y = run(isa, g.expert, g.x, ncols);
    const auto nonzero = std::count_if(y.begin(), y.end(), [](std::uint16_t v) { return (v & 0x7FFF) != 0; });
    check(nonzero > static_cast<long>(y.size() * 9 / 10), "golden outputs are not degenerate");
    return fixtures::output_hash(y);
}

void test_golden() {
    using fixtures::kGolden1;
    using fixtures::kGolden4;
    for (moe::CpuIsa isa : {moe::CpuIsa::kScalar, moe::CpuIsa::kAvx2, moe::CpuIsa::kAvxVnni, moe::CpuIsa::kAvx512Vnni}) {
        if (!moe::cpu_isa_supported(isa)) { continue; }
        const std::uint64_t h1 = golden_hash(isa, 1), h4 = golden_hash(isa, 4);
        if (h1 != kGolden1 || h4 != kGolden4) {
            std::fprintf(stderr, "golden %s: %016llx %016llx\n", moe::cpu_isa_name(isa),
                         static_cast<unsigned long long>(h1), static_cast<unsigned long long>(h4));
        }
        check(h1 == kGolden1 && h4 == kGolden4, "golden expert output hash");
    }
}

} // namespace

int main() {
    test_e4m3();
    test_e2m1();
    test_bf16_round();
    test_exp_silu();
    test_quantizer();
    test_isa_equality();
    test_split_invariance();
    test_fp64_oracle();
    test_golden();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all offloaded_moe CPU checks passed\n");
    return 0;
}
