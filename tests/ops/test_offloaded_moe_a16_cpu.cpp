// Canonical W4A16 arithmetic and the CPU narrow route for experts stored without activation scales
// (docs/maintainer/qwen3_8-flash-next-design.md §16.2.1). Host-only: needs no GPU.
//
// Oracles:
//   - the A16 encoding against exact binary64 rounding of v / 2^(emax - 147), for every BF16 value
//     and every column exponent at or above its own;
//   - the int64 row sums against a direct int64 evaluation from decoded integers (no limbs);
//   - every CPU ISA against the scalar reference, bit for bit;
//   - the BF16 outputs against an FP64 expert oracle on the unencoded BF16 activations.

#include "ops/op_family.h"
#include "ops/common/canonical_math.h"
#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"
#include "ops/offloaded_moe_fixtures.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace canon    = infernix::ops::canon;
namespace moe      = infernix::ops::offloaded_moe;
namespace fixtures = infernix::test::offloaded_moe;
using fixtures::Expert;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------- encoding

// rne(v / 2^(emax - 147)) in binary64: v and the quotient are exact, and so is the rounding.
double expected_x(std::uint16_t v, int emax) {
    const double q = std::ldexp(static_cast<double>(canon::bf16_to_f32(v)), 147 - emax);
    return std::nearbyint(q); // the default rounding mode is to nearest, ties to even
}

void test_encoding() {
    int bad = 0, exact_bad = 0;
    for (std::uint32_t u = 0; u < 65536; ++u) {
        const auto v = static_cast<std::uint16_t>(u);
        if ((v & 0x7F80U) == 0x7F80U) { continue; } // Inf and NaN
        const int field = (v >> 7) & 0xFF, own = field == 0 ? 1 : field;
        for (int emax = own; emax <= 254; ++emax) {
            const std::int32_t x = canon::a16_value(v, emax);
            if (static_cast<double>(x) != expected_x(v, emax) || std::abs(x) >= (1 << 21)) { ++bad; }
            // Exact while the element is within 2^13 of the column's largest exponent.
            if (emax - own <= 13 && std::ldexp(static_cast<double>(x), emax - 147) != canon::bf16_to_f32(v)) { ++exact_bad; }
        }
    }
    check(bad == 0, "a16_value is rne(v / 2^(emax - 147)) with |X| < 2^21");
    check(exact_bad == 0, "a16_value is exact within 2^13 of the column exponent");

    std::mt19937 rng(1);
    std::uniform_int_distribution<int> any(0, 65535);
    int limb_bad = 0;
    for (int it = 0; it < 200000; ++it) {
        std::uint16_t v[16];
        for (auto& e : v) {
            do { e = static_cast<std::uint16_t>(any(rng)); } while ((e & 0x7F80U) == 0x7F80U);
        }
        const int emax = canon::a16_column_exponent(v, 16);
        const auto b   = canon::a16_block(v, emax);
        int sum        = 0;
        for (int j = 0; j < 16; ++j) {
            sum += b.hi[j];
            if (canon::a16_x(b, j) != canon::a16_value(v[j], emax)) { ++limb_bad; }
        }
        if (sum != b.hi_sum) { ++limb_bad; }
    }
    check(limb_bad == 0, "limbs reconstruct X and hi_sum is their sum");

    std::uint16_t zeros[32] = {};
    check(canon::a16_column_exponent(zeros, 32) == 1, "all-zero column exponent is 1");
    zeros[5] = 0x8000; // -0
    check(canon::a16_column_exponent(zeros, 32) == 1 && canon::a16_value(0x8000, 1) == 0, "-0 encodes as 0");
    zeros[7] = 0x7F80; // +inf
    check(canon::a16_column_exponent(zeros, 32) == canon::kA16NonFinite, "Inf makes the column non-finite");
    zeros[7] = 0xFFC1; // NaN
    check(canon::a16_column_exponent(zeros, 32) == canon::kA16NonFinite, "NaN makes the column non-finite");
    check(canon::a16_row_output(12345, canon::kA16NonFinite, 1.0F) == 0x7FC0, "non-finite column gives NaN");
}

void test_row_output() {
    std::mt19937 rng(2);
    std::uniform_int_distribution<std::int64_t> s_dist(-(std::int64_t{1} << 56), std::int64_t{1} << 56);
    std::uniform_int_distribution<int> e_dist(1, 254);
    std::uniform_real_distribution<float> m_dist(-20.0F, 4.0F);
    int bad = 0;
    for (int it = 0; it < 1'000'000; ++it) {
        const std::int64_t s = it % 7 == 0 ? s_dist(rng) >> (it % 50) : s_dist(rng);
        const int emax       = e_dist(rng);
        const float m        = std::exp2(m_dist(rng));
        const std::uint16_t y = canon::a16_row_output(s, emax, m);
        // Reference: the same three roundings in binary64-then-binary32 form.
        const float t   = static_cast<float>(s);
        const double ud = std::ldexp(static_cast<double>(t), emax - 157);
        // Binary32 overflows from 2^128 - 2^103 (half an ulp above the largest finite value) up.
        const double overflow = std::ldexp(1.0, 128) - std::ldexp(1.0, 103);
        const float u = std::fabs(ud) >= overflow ? std::copysign(INFINITY, static_cast<float>(t))
                                                  : static_cast<float>(ud); // one rounding
        const float w   = u * m;
        if (y != canon::f32_to_bf16_rn(w)) { ++bad; }
    }
    check(bad == 0, "a16_row_output = bf16(fl32(fl32(S) * 2^(emax - 157)) * m)");
}

// ---------------------------------------------------------------------------- experts
//
// Every expert check runs at both registered geometries (w4a4_expert.h): Qwen3.8-Flash-Next's
// H 2560 / I 640 and Qwen3.5-122B-A10B's H 3072 / I 1024.

constexpr moe::ExpertShape kShapes[] = {moe::kFlashNextShape, moe::kQwen3_5_122BShape};

std::vector<std::uint16_t> run(moe::CpuIsa isa, const moe::ExpertShape& shape, const Expert& e,
                               const std::vector<std::uint16_t>& x, int ncols) {
    const auto h = static_cast<std::size_t>(shape.hidden);
    std::vector<std::uint16_t> y(static_cast<std::size_t>(ncols) * h);
    const std::uint16_t* xp[moe::kMaxColumns];
    std::uint16_t* yp[moe::kMaxColumns];
    for (int c = 0; c < ncols; ++c) {
        xp[c] = &x[static_cast<std::size_t>(c) * h];
        yp[c] = &y[static_cast<std::size_t>(c) * h];
    }
    moe::expert_forward_a16(isa, shape, e.record.data(), e.scales, ncols, xp, yp);
    return y;
}

// Weight (doubled code times scaled block scale) of row `row` at k in a matrix of `blocks` blocks.
std::int64_t weight_int(const std::uint8_t* matrix, int blocks, int row, int k) {
    const int rg = row / 16, r = row % 16, b = k / 16, kk = k % 16;
    const std::uint8_t* unit = matrix + (static_cast<std::size_t>(rg) * blocks + b) * moe::kUnitBytes;
    const std::uint8_t byte  = unit[32 * (kk / 4) + 4 * (r % 8) + kk % 4];
    const unsigned code      = r < 8 ? (byte & 15U) : (byte >> 4);
    return static_cast<std::int64_t>(canon::e2m1_x2(code)) * canon::e4m3_scaled(unit[128 + r]);
}

void test_row_sums(const moe::ExpertShape& shape) {
    std::mt19937 rng(3);
    const Expert e  = fixtures::random_a16_expert(rng, shape);
    const int ncols = 3;
    const int hidden = shape.hidden, blocks = shape.gate_up_blocks(), groups = shape.gate_up_row_groups();
    const auto x    = fixtures::wide_range_activations(rng, ncols, hidden);
    std::vector<canon::A16Block> xb(static_cast<std::size_t>(ncols) * blocks);
    const canon::A16Block* xp[moe::kMaxColumns];
    int emax[moe::kMaxColumns];
    for (int c = 0; c < ncols; ++c) {
        emax[c] = moe::encode_a16(&x[static_cast<std::size_t>(c) * hidden], hidden, &xb[static_cast<std::size_t>(c) * blocks]);
        xp[c]   = &xb[static_cast<std::size_t>(c) * blocks];
    }
    std::vector<std::int64_t> s(static_cast<std::size_t>(groups) * 16 * ncols);
    for (moe::CpuIsa isa : {moe::CpuIsa::kScalar, moe::CpuIsa::kAvx2, moe::CpuIsa::kAvxVnni, moe::CpuIsa::kAvx512Vnni}) {
        if (!moe::cpu_isa_supported(isa)) { continue; }
        moe::rg16_row_sums_a16(isa, e.record.data(), blocks, 0, groups, xp, ncols, s.data());
        int bad = 0;
        for (int row = 0; row < groups * 16; ++row) {
            for (int c = 0; c < ncols; ++c) {
                std::int64_t ref = 0;
                for (int k = 0; k < hidden; ++k) {
                    ref += weight_int(e.record.data(), blocks, row, k) *
                           canon::a16_value(x[static_cast<std::size_t>(c) * hidden + k], emax[c]);
                }
                if (ref != s[static_cast<std::size_t>(row) * ncols + c]) { ++bad; }
            }
        }
        std::printf("row sums H %d %-12s %s\n", hidden, moe::cpu_isa_name(isa), bad == 0 ? "exact" : "WRONG");
        check(bad == 0, "int64 row sums equal the direct integer dot products");
    }
}

void test_isa_equality(const moe::ExpertShape& shape) {
    std::mt19937 rng(42);
    for (moe::CpuIsa isa : {moe::CpuIsa::kAvx2, moe::CpuIsa::kAvxVnni, moe::CpuIsa::kAvx512Vnni}) {
        std::printf("CPU ISA %-12s %s\n", moe::cpu_isa_name(isa), moe::cpu_isa_supported(isa) ? "tested" : "not supported on this host (skipped)");
    }
    const int hidden = shape.hidden;
    for (int trial = 0; trial < 4; ++trial) {
        const Expert e = fixtures::random_a16_expert(rng, shape);
        for (int ncols = 1; ncols <= moe::kMaxColumns; ++ncols) {
            const auto x   = trial % 2 ? fixtures::wide_range_activations(rng, ncols, hidden)
                                       : fixtures::random_activations(rng, ncols, hidden);
            const auto ref = run(moe::CpuIsa::kScalar, shape, e, x, ncols);
            for (moe::CpuIsa isa : {moe::CpuIsa::kAvx2, moe::CpuIsa::kAvxVnni, moe::CpuIsa::kAvx512Vnni}) {
                if (!moe::cpu_isa_supported(isa)) { continue; }
                check(run(isa, shape, e, x, ncols) == ref, "ISA output bit-identical to the scalar reference");
            }
            if (ncols > 1) {
                const std::vector<std::uint16_t> x0(x.begin(), x.begin() + hidden);
                const auto y0 = run(moe::best_cpu_isa(), shape, e, x0, 1);
                check(std::equal(y0.begin(), y0.end(), ref.begin()), "column result independent of batch width");
            }
        }
    }
}

void test_split_invariance(const moe::ExpertShape& shape) {
    std::mt19937 rng(9);
    const Expert e  = fixtures::random_a16_expert(rng, shape);
    const int ncols = 3;
    const int hidden = shape.hidden, intermediate = shape.intermediate;
    const int x_blocks = shape.gate_up_blocks(), h_blocks = shape.h_blocks();
    const auto x    = fixtures::wide_range_activations(rng, ncols, hidden);
    const auto ref  = run(moe::CpuIsa::kScalar, shape, e, x, ncols);
    const moe::CpuIsa isa = moe::best_cpu_isa();
    std::vector<canon::A16Block> xb(static_cast<std::size_t>(ncols) * x_blocks), hb(static_cast<std::size_t>(ncols) * h_blocks);
    std::vector<std::uint16_t> hv(static_cast<std::size_t>(ncols) * intermediate), y(static_cast<std::size_t>(ncols) * hidden);
    const canon::A16Block* xp[8];
    const canon::A16Block* hp[8];
    std::uint16_t* hvp[8];
    std::uint16_t* yp[8];
    int xe[8], he[8];
    for (int c = 0; c < ncols; ++c) {
        const std::uint16_t* xc = &x[static_cast<std::size_t>(c) * hidden];
        xe[c] = canon::a16_column_exponent(xc, hidden);
        // Encoded in uneven block slices, as the worker team does.
        const int cuts[] = {0, 7, 64, 101, x_blocks};
        for (int i = 0; i + 1 < 5; ++i) {
            moe::encode_a16_blocks(xc, xe[c], cuts[i], cuts[i + 1], &xb[static_cast<std::size_t>(c) * x_blocks]);
        }
        xp[c]  = &xb[static_cast<std::size_t>(c) * x_blocks];
        hvp[c] = &hv[static_cast<std::size_t>(c) * intermediate];
        hp[c]  = &hb[static_cast<std::size_t>(c) * h_blocks];
        yp[c]  = &y[static_cast<std::size_t>(c) * hidden];
    }
    const int ucuts[] = {h_blocks, 29, 13, 3, 0};
    for (int i = 0; i + 1 < 5; ++i) {
        moe::gate_up_units_a16(isa, shape, e.record.data(), e.scales, xp, xe, ncols, ucuts[i + 1], ucuts[i], hvp);
    }
    for (int c = 0; c < ncols; ++c) { he[c] = moe::encode_a16(hvp[c], intermediate, &hb[static_cast<std::size_t>(c) * h_blocks]); }
    const int rcuts[] = {shape.down_row_groups(), 111, 57, 1, 0};
    for (int i = 0; i + 1 < 5; ++i) {
        moe::down_rows_a16(isa, shape, e.record.data(), e.scales, hp, he, ncols, rcuts[i + 1], rcuts[i], yp);
    }
    check(y == ref, "worker splits do not change a bit");
}

// The expert output against FP64 on the unencoded BF16 activations, with the canonical BF16
// boundaries (y_gate, y_up, h) reproduced from the oracle's own values.
void test_fp64_oracle(const moe::ExpertShape& shape) {
    std::mt19937 rng(21);
    const int hidden = shape.hidden, intermediate = shape.intermediate;
    for (int trial = 0; trial < 3; ++trial) {
        const Expert e = fixtures::random_a16_expert(rng, shape);
        const auto x   = trial == 2 ? fixtures::wide_range_activations(rng, 1, hidden)
                                    : fixtures::random_activations(rng, 1, hidden);
        const auto y   = run(moe::CpuIsa::kScalar, shape, e, x, 1);
        const std::uint8_t* gu = e.record.data();
        const std::uint8_t* dn = e.record.data() + shape.gate_up_bytes();
        std::vector<double> h(static_cast<std::size_t>(intermediate));
        std::vector<std::uint16_t> hq(static_cast<std::size_t>(intermediate));
        for (int i = 0; i < intermediate; ++i) {
            double g = 0, u = 0;
            for (int k = 0; k < hidden; ++k) {
                const double xv = canon::bf16_to_f32(x[static_cast<std::size_t>(k)]);
                g += static_cast<double>(weight_int(gu, shape.gate_up_blocks(), 2 * i, k)) * xv;
                u += static_cast<double>(weight_int(gu, shape.gate_up_blocks(), 2 * i + 1, k)) * xv;
            }
            g *= std::ldexp(1.0, -10) * e.scales.alpha_gate;
            u *= std::ldexp(1.0, -10) * e.scales.alpha_up;
            const float gb = canon::bf16_to_f32(canon::f32_to_bf16_rn(static_cast<float>(g)));
            const float ub = canon::bf16_to_f32(canon::f32_to_bf16_rn(static_cast<float>(u)));
            h[static_cast<std::size_t>(i)]  = gb / (1.0 + std::exp(-static_cast<double>(gb))) * ub;
            hq[static_cast<std::size_t>(i)] = canon::f32_to_bf16_rn(static_cast<float>(h[static_cast<std::size_t>(i)]));
        }
        int bad = 0;
        for (int row = 0; row < hidden; ++row) {
            double acc = 0, mag = 0;
            for (int k = 0; k < intermediate; ++k) {
                const double t = static_cast<double>(weight_int(dn, shape.down_blocks(), row, k)) *
                                 canon::bf16_to_f32(hq[static_cast<std::size_t>(k)]);
                acc += t;
                mag += std::fabs(t);
            }
            const double scale = std::ldexp(1.0, -10) * e.scales.alpha_down;
            const double ref   = acc * scale;
            const double got   = canon::bf16_to_f32(y[static_cast<std::size_t>(row)]);
            // One BF16 rounding of the output and a few BF16 ulps of propagated y_gate/y_up/h rounding
            // differences (the oracle's and the route's BF16 boundaries can round different ways).
            const double tol = std::ldexp(std::fabs(ref), -7) + mag * scale * std::ldexp(1.0, -7) + 1e-30;
            if (std::fabs(got - ref) > tol) { ++bad; }
        }
        std::printf("FP64 oracle H %d trial %d: %d of %d outputs outside tolerance\n", hidden, trial, bad, hidden);
        check(bad == 0, "expert output within BF16 rounding of the FP64 oracle");
    }
}

void test_non_finite(const moe::ExpertShape& shape) {
    std::mt19937 rng(5);
    const int hidden = shape.hidden;
    const Expert e = fixtures::random_a16_expert(rng, shape);
    auto x         = fixtures::random_activations(rng, 2, hidden);
    x[100]         = 0x7F80; // column 0: +inf
    const auto y   = run(moe::best_cpu_isa(), shape, e, x, 2);
    bool col0_nan = true, col1_finite = true;
    for (int i = 0; i < hidden; ++i) {
        col0_nan    = col0_nan && y[static_cast<std::size_t>(i)] == 0x7FC0;
        col1_finite = col1_finite && (y[static_cast<std::size_t>(hidden + i)] & 0x7F80) != 0x7F80;
    }
    check(col0_nan, "a non-finite column gives NaN outputs");
    check(col1_finite, "other columns are unaffected by a non-finite column");
}

using fixtures::kGoldenA16_1;
using fixtures::kGoldenA16_4;

std::uint64_t golden_hash(moe::CpuIsa isa, int ncols) {
    const auto g = fixtures::golden_a16_case(ncols);
    const auto y = run(isa, moe::kFlashNextShape, g.expert, g.x, ncols);
    const auto nonzero = std::count_if(y.begin(), y.end(), [](std::uint16_t v) { return (v & 0x7FFF) != 0; });
    check(nonzero > static_cast<long>(y.size() * 9 / 10), "golden outputs are not degenerate");
    return fixtures::output_hash(y);
}

void test_golden() {
    for (moe::CpuIsa isa : {moe::CpuIsa::kScalar, moe::CpuIsa::kAvx2, moe::CpuIsa::kAvxVnni, moe::CpuIsa::kAvx512Vnni}) {
        if (!moe::cpu_isa_supported(isa)) { continue; }
        const std::uint64_t h1 = golden_hash(isa, 1), h4 = golden_hash(isa, 4);
        if (h1 != kGoldenA16_1 || h4 != kGoldenA16_4) {
            std::fprintf(stderr, "golden A16 %s: %016llx %016llx\n", moe::cpu_isa_name(isa),
                         static_cast<unsigned long long>(h1), static_cast<unsigned long long>(h4));
        }
        check(h1 == kGoldenA16_1 && h4 == kGoldenA16_4, "golden A16 expert output hash");
    }
}

} // namespace

INFERNIX_OP_TEST(offloaded_moe_a16_cpu) {
    test_encoding();
    test_row_output();
    for (const moe::ExpertShape& shape : kShapes) {
        test_row_sums(shape);
        test_isa_equality(shape);
        test_split_invariance(shape);
        test_fp64_oracle(shape);
        test_non_finite(shape);
    }
    test_golden();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all offloaded_moe A16 CPU checks passed\n");
    return 0;
}
