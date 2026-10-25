// The two-term A4 activations of W4A16 experts (expert-a4-residual-decode; canon::a4r_*, a4r_expert.h;
// docs/maintainer/qwen3_8-flash-next-design.md §19.3.23) and their CPU route. Host-only: needs no GPU.
// The GPU narrow kernels are checked against the same oracles and against these CPU bits in
// test_offloaded_moe_wide (W4A16 layer) and test_offloaded_moe_layer.
//
// Criteria, fixed before the first run:
//   E1  canon::a4r_block_terms equals the independent binary64 oracle (offloaded_moe_a4r_oracle.h)
//       code for code and scale for scale, signed zeros included, on random, wide-range and
//       adversarial blocks (rounding ties of E2M1 and E4M3, saturation, E4M3 subnormal second terms,
//       zero, -0, single-element and tiny rows).
//   E2  rg16_row_sums_a4r equals a direct int64 evaluation from the decoded terms on every ISA and
//       every column count 1..8 (the four-column passes).
//   E3  every ISA, the worker team at several sizes and uneven worker splits give the scalar route's
//       bits; a column's result does not depend on the job's width.
//   E4  each output within BF16 rounding of the FP64 expert on the decoded terms (oracle_outliers == 0).
//   E5  against the exact W4A16 route (expert_forward_a16): relative L2 per column <= 2^-5 and mean
//       <= 2^-7, the wide two-term route's bounds (test_offloaded_moe_wide X1); one 4-bit term misses by
//       several percent, a layout or scale error by ~100 %.
//   E6  a non-finite column gives the canonical NaN, a zero column zeros, other columns are unaffected.
// (AVX-512: run under Intel SDE `-spr` on hosts without it; the test prints the ISAs it covered.)

#include "ops/op_family.h"
#include "ops/common/canonical_math.h"
#include "ops/offloaded_moe_a4r_oracle.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/offloaded_sparse_moe/cpu/a4r_expert.h"
#include "ops/offloaded_sparse_moe/cpu/expert_team.h"
#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
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

constexpr int kTerms = canon::kA4ResidualTerms;
constexpr moe::CpuIsa kIsas[] = {moe::CpuIsa::kScalar, moe::CpuIsa::kAvx2, moe::CpuIsa::kAvxVnni, moe::CpuIsa::kAvx512Vnni};

// ---------------------------------------------------------------------------- E1 encoding

bool same_terms(const std::uint16_t* v, float multiplier) {
    canon::A4Codes got[kTerms];
    canon::a4r_block_terms<kTerms>(v, multiplier, got);
    const auto want = fixtures::oracle_terms(v, multiplier);
    for (int t = 0; t < kTerms; ++t) {
        if (got[t].scale_word != want[static_cast<std::size_t>(t)].scale) { return false; }
        for (int j = 0; j < 16; ++j) {
            if (got[t].code[j] != want[static_cast<std::size_t>(t)].code[static_cast<std::size_t>(j)]) { return false; }
        }
    }
    return true;
}

void test_encoding() {
    std::mt19937 rng(11);
    long bad = 0, blocks = 0;
    // Rows as the routes see them: the row scale over the row, then every block.
    const auto check_row = [&](const std::vector<std::uint16_t>& row) {
        const float rs = moe::a4r_row_scale_of(row.data(), static_cast<int>(row.size()));
        if (std::isnan(rs) != std::isnan(fixtures::oracle_row_scale(row.data(), static_cast<int>(row.size()))) ||
            (!std::isnan(rs) && rs != fixtures::oracle_row_scale(row.data(), static_cast<int>(row.size()))) ||
            canon::a4r_multiplier(rs) != fixtures::oracle_multiplier(rs)) {
            ++bad;
        }
        const float d = canon::a4r_multiplier(rs);
        for (std::size_t b = 0; b + 16 <= row.size(); b += 16) {
            bad += same_terms(&row[b], d) ? 0 : 1;
            ++blocks;
        }
    };
    for (int it = 0; it < 400; ++it) {
        check_row(it % 2 ? fixtures::wide_range_activations(rng, 1) : fixtures::random_activations(rng, 1));
    }
    // Adversarial blocks under fixed multipliers: values on E2M1 midpoints of their block's scale, exact
    // grid values, saturating maxima, second terms in the E4M3 subnormal range, single elements, -0.
    std::uniform_int_distribution<int> any(0, 65535), sign(0, 1), pick(0, 15);
    std::uniform_real_distribution<float> lg(-24.0F, 12.0F);
    static constexpr float kMid[] = {0.25F, 0.75F, 1.25F, 1.75F, 2.5F, 3.5F, 5.0F, 6.0F, 7.0F};
    for (int it = 0; it < 200000; ++it) {
        std::uint16_t v[16];
        const int kind = it % 5;
        const float base = std::exp2(lg(rng));
        for (int j = 0; j < 16; ++j) {
            float f = 0.0F;
            switch (kind) {
            case 0: do { v[j] = static_cast<std::uint16_t>(any(rng)); } while ((v[j] & 0x7F80U) == 0x7F80U); continue;
            case 1: f = base * kMid[pick(rng) % 9] / 6.0F; break;                      // midpoints of the block's grid
            case 2: f = base * static_cast<float>(pick(rng)) / 16.0F; break;           // small integers of a step
            case 3: f = j == 0 ? base : base * std::exp2(-static_cast<float>(pick(rng) + 8)); break; // one outlier
            default: f = j == pick(rng) ? base : 0.0F; break;                          // sparse
            }
            v[j] = fixtures::to_bf16(sign(rng) ? -f : f);
            if (kind == 4 && f == 0.0F && sign(rng)) { v[j] = 0x8000U; } // -0
        }
        // Multipliers of real rows (amax / 2688 for amax across the BF16 range) and exact powers of two.
        const float amax = std::exp2(lg(rng));
        const float d = it % 3 == 0 ? std::exp2(static_cast<float>(it % 40 - 20)) : canon::a4r_multiplier(amax / 2688.0F);
        bad += same_terms(v, d) ? 0 : 1;
        ++blocks;
    }
    std::uint16_t zeros[16] = {};
    bad += same_terms(zeros, 1.0F) ? 0 : 1;
    std::printf("E1 term encoding: %ld of %ld blocks differ from the binary64 oracle\n", bad, blocks);
    check(bad == 0, "E1: the term encoding equals the independent oracle");
    // The edge rows: zero, non-finite and below 2^-116 (multiplier 0: zero terms).
    std::vector<std::uint16_t> row(64, 0);
    check(moe::a4r_row_scale_of(row.data(), 64) == 0.0F && canon::a4r_multiplier(0.0F) == 0.0F, "a zero row has scale 0");
    row[3] = 0x7F80U;
    check(std::isnan(moe::a4r_row_scale_of(row.data(), 64)) && canon::a4r_multiplier(moe::a4r_row_scale_of(row.data(), 64)) == 0.0F,
          "an Inf makes the row scale NaN and its terms zero");
    row[3] = 0x0001U; // the smallest BF16 subnormal: 2^-133
    check(canon::a4r_multiplier(moe::a4r_row_scale_of(row.data(), 64)) == 0.0F, "a row below 2^-116 has zero terms");
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
    moe::expert_forward_a4r(isa, e.record.data(), e.scales, ncols, xp, yp);
    return y;
}

std::vector<std::uint16_t> run_exact(const Expert& e, const std::vector<std::uint16_t>& x, int ncols) {
    std::vector<std::uint16_t> y(static_cast<std::size_t>(ncols) * moe::kHidden);
    const std::uint16_t* xp[moe::kMaxColumns];
    std::uint16_t* yp[moe::kMaxColumns];
    for (int c = 0; c < ncols; ++c) {
        xp[c] = &x[static_cast<std::size_t>(c) * moe::kHidden];
        yp[c] = &y[static_cast<std::size_t>(c) * moe::kHidden];
    }
    moe::expert_forward_a16(moe::best_cpu_isa(), moe::kFlashNextShape, e.record.data(), e.scales, ncols, xp, yp);
    return y;
}

// E2: S of every gate/up row from the decoded terms as integers (doubled codes times scaled scales).
void test_row_sums() {
    std::mt19937 rng(3);
    const Expert e = fixtures::random_a16_expert(rng);
    for (int ncols = 1; ncols <= moe::kMaxColumns; ++ncols) {
        const auto x = fixtures::wide_range_activations(rng, ncols);
        constexpr std::size_t kRow = static_cast<std::size_t>(kTerms) * moe::kGateUpBlocks;
        std::vector<canon::A4Block> terms(static_cast<std::size_t>(ncols) * kRow);
        const canon::A4Block* tp[moe::kMaxColumns];
        for (int c = 0; c < ncols; ++c) {
            (void)moe::encode_a4r(&x[static_cast<std::size_t>(c) * moe::kHidden], moe::kHidden, &terms[c * kRow]);
            tp[c] = &terms[c * kRow];
        }
        // Rows of the first two row groups and the last two (the widest scales' columns).
        for (const int rg : {0, moe::kGateUpRowGroups - 2}) {
            std::vector<std::int64_t> want(static_cast<std::size_t>(32) * ncols, 0);
            for (int r = 0; r < 32; ++r) {
                for (int c = 0; c < ncols; ++c) {
                    std::int64_t s = 0;
                    for (int t = 0; t < kTerms; ++t) {
                        for (int k = 0; k < moe::kHidden; ++k) {
                            const canon::A4Block& a = tp[c][static_cast<std::size_t>(t) * moe::kGateUpBlocks + k / 16];
                            s += static_cast<std::int64_t>(fixtures::oracle_weight(e.record.data(), moe::kGateUpBlocks, 16 * rg + r, k)) *
                                 a.c2[k % 16] * a.scale_scaled;
                        }
                    }
                    want[static_cast<std::size_t>(r) * ncols + c] = s;
                }
            }
            for (const moe::CpuIsa isa : kIsas) {
                if (!moe::cpu_isa_supported(isa)) { continue; }
                std::vector<std::int64_t> got(want.size());
                moe::rg16_row_sums_a4r(isa, e.record.data(), moe::kGateUpBlocks, rg, rg + 2, tp, ncols, got.data());
                check(got == want, "E2: two-term row sums equal the direct int64 evaluation");
            }
        }
    }
}

void test_isa_and_width() {
    std::mt19937 rng(42);
    for (const moe::CpuIsa isa : kIsas) {
        std::printf("CPU ISA %-12s %s\n", moe::cpu_isa_name(isa),
                    moe::cpu_isa_supported(isa) ? "tested" : "not supported on this host (skipped)");
    }
    for (int trial = 0; trial < 3; ++trial) {
        const Expert e = fixtures::random_a16_expert(rng);
        for (int ncols = 1; ncols <= moe::kMaxColumns; ++ncols) {
            const auto x   = trial == 1 ? fixtures::wide_range_activations(rng, ncols) : fixtures::random_activations(rng, ncols);
            const auto ref = run(moe::CpuIsa::kScalar, e, x, ncols);
            for (const moe::CpuIsa isa : kIsas) {
                if (isa == moe::CpuIsa::kScalar || !moe::cpu_isa_supported(isa)) { continue; }
                check(run(isa, e, x, ncols) == ref, "E3: ISA output bit-identical to the scalar reference");
            }
            for (int c = 0; c < ncols; ++c) {
                const std::vector<std::uint16_t> xc(x.begin() + static_cast<long>(c) * moe::kHidden,
                                                    x.begin() + static_cast<long>(c + 1) * moe::kHidden);
                const auto yc = run(moe::best_cpu_isa(), e, xc, 1);
                check(std::equal(yc.begin(), yc.end(), ref.begin() + static_cast<long>(c) * moe::kHidden),
                      "E3: a column's result does not depend on the job's width");
            }
        }
    }
}

// E3: the worker team (any size, any split of its items) gives expert_forward_a4r's bits.
void test_team() {
    std::mt19937 rng(17);
    constexpr int kJobs = 5;
    std::vector<Expert> bank;
    std::vector<std::vector<std::uint16_t>> xs, ys, want;
    const int widths[kJobs] = {1, 8, 3, 5, 2};
    for (int j = 0; j < kJobs; ++j) {
        bank.push_back(fixtures::random_a16_expert(rng));
        xs.push_back(j % 2 ? fixtures::wide_range_activations(rng, widths[j]) : fixtures::random_activations(rng, widths[j]));
        want.push_back(run(moe::CpuIsa::kScalar, bank.back(), xs.back(), widths[j]));
        ys.emplace_back(xs.back().size());
    }
    for (const int workers : {1, 3, 7}) {
        moe::CpuExpertTeam team({.workers = workers, .max_jobs = kJobs, .activation = moe::ExpertActivation::kA16,
                                 .narrow_a16 = moe::NarrowA16::kA4Residual});
        std::vector<moe::CpuExpertJob> jobs(kJobs);
        for (int j = 0; j < kJobs; ++j) {
            jobs[j].record = bank[static_cast<std::size_t>(j)].record.data();
            jobs[j].scales = bank[static_cast<std::size_t>(j)].scales;
            jobs[j].ncols  = widths[j];
            std::fill(ys[static_cast<std::size_t>(j)].begin(), ys[static_cast<std::size_t>(j)].end(), std::uint16_t{0xFFFF});
            for (int c = 0; c < widths[j]; ++c) {
                jobs[j].x[c] = &xs[static_cast<std::size_t>(j)][static_cast<std::size_t>(c) * moe::kHidden];
                jobs[j].y[c] = &ys[static_cast<std::size_t>(j)][static_cast<std::size_t>(c) * moe::kHidden];
            }
        }
        for (int round = 0; round < 2; ++round) { team.run(jobs); }
        bool same = true;
        for (int j = 0; j < kJobs; ++j) { same = same && ys[static_cast<std::size_t>(j)] == want[static_cast<std::size_t>(j)]; }
        std::printf("E3 team of %d workers: %s\n", workers, same ? "expert_forward_a4r's bits" : "DIFFERENT");
        check(same, "E3: the worker team gives expert_forward_a4r's bits");
    }
    // A W4A4 or GGUF team cannot take two-term jobs.
    bool refused = false;
    try {
        moe::CpuExpertTeam team({.workers = 1, .activation = moe::ExpertActivation::kA4, .narrow_a16 = moe::NarrowA16::kA4Residual});
    } catch (const std::invalid_argument&) { refused = true; }
    check(refused, "a W4A4 team refuses the two-term arithmetic");
}

void test_split_invariance() {
    std::mt19937 rng(9);
    const Expert e  = fixtures::random_a16_expert(rng);
    const int ncols = 3;
    const auto x    = fixtures::wide_range_activations(rng, ncols);
    const auto ref  = run(moe::CpuIsa::kScalar, e, x, ncols);
    const moe::CpuIsa isa = moe::best_cpu_isa();
    constexpr std::size_t kX = static_cast<std::size_t>(kTerms) * moe::kGateUpBlocks;
    constexpr std::size_t kH = static_cast<std::size_t>(kTerms) * moe::kHBlocks;
    std::vector<canon::A4Block> xt(ncols * kX), ht(ncols * kH);
    std::vector<std::uint16_t> hv(static_cast<std::size_t>(ncols) * moe::kIntermediate), y(static_cast<std::size_t>(ncols) * moe::kHidden);
    const canon::A4Block* xp[8];
    const canon::A4Block* hp[8];
    std::uint16_t* hvp[8];
    std::uint16_t* yp[8];
    float xs[8], hs[8];
    for (int c = 0; c < ncols; ++c) {
        const std::uint16_t* xc = &x[static_cast<std::size_t>(c) * moe::kHidden];
        xs[c] = moe::a4r_row_scale_of(xc, moe::kHidden);
        const int cuts[] = {0, 7, 64, 101, moe::kGateUpBlocks}; // uneven slices, as the team's
        for (int i = 0; i + 1 < 5; ++i) { moe::encode_a4r_blocks(xc, moe::kGateUpBlocks, xs[c], cuts[i], cuts[i + 1], &xt[c * kX]); }
        xp[c]  = &xt[c * kX];
        hvp[c] = &hv[static_cast<std::size_t>(c) * moe::kIntermediate];
        hp[c]  = &ht[c * kH];
        yp[c]  = &y[static_cast<std::size_t>(c) * moe::kHidden];
    }
    const int ucuts[] = {moe::kHBlocks, 29, 13, 3, 0};
    for (int i = 0; i + 1 < 5; ++i) { moe::gate_up_units_a4r(isa, e.record.data(), e.scales, xp, xs, ncols, ucuts[i + 1], ucuts[i], hvp); }
    for (int c = 0; c < ncols; ++c) { hs[c] = moe::encode_a4r(hvp[c], moe::kIntermediate, &ht[c * kH]); }
    const int rcuts[] = {moe::kDownRowGroups, 111, 57, 1, 0};
    for (int i = 0; i + 1 < 5; ++i) { moe::down_rows_a4r(isa, e.record.data(), e.scales, hp, hs, ncols, rcuts[i + 1], rcuts[i], yp); }
    check(y == ref, "E3: worker splits do not change a bit");
}

// E4 and E5.
void test_oracles() {
    std::mt19937 rng(21);
    double sum = 0.0, worst = 0.0;
    int columns = 0;
    for (int trial = 0; trial < 6; ++trial) {
        const Expert e = fixtures::random_a16_expert(rng);
        const auto x   = trial % 2 ? fixtures::wide_range_activations(rng, 2) : fixtures::random_activations(rng, 2);
        const auto y   = run(moe::CpuIsa::kScalar, e, x, 2);
        const auto ex  = run_exact(e, x, 2);
        for (int c = 0; c < 2; ++c) {
            const std::uint16_t* yc = &y[static_cast<std::size_t>(c) * moe::kHidden];
            const auto ref          = fixtures::oracle_expert(e, &x[static_cast<std::size_t>(c) * moe::kHidden]);
            const int bad           = fixtures::oracle_outliers(ref, yc);
            double err = 0.0, norm = 0.0;
            for (int i = 0; i < moe::kHidden; ++i) {
                const double r = canon::bf16_to_f32(ex[static_cast<std::size_t>(c) * moe::kHidden + i]);
                const double d = canon::bf16_to_f32(yc[i]) - r;
                err += d * d;
                norm += r * r;
            }
            const double rel = norm > 0.0 ? std::sqrt(err / norm) : std::sqrt(err);
            worst = std::max(worst, rel);
            sum += rel;
            ++columns;
            std::printf("E4/E5 trial %d column %d: %d of %d outputs outside the FP64 tolerance; relative L2 %.5f "
                        "against the exact route\n", trial, c, bad, moe::kHidden, rel);
            check(bad == 0, "E4: every output within BF16 rounding of the FP64 expert on the decoded terms");
        }
    }
    const double mean = sum / columns;
    std::printf("E5 relative L2 against the exact W4A16 route: mean %.5f, worst %.5f\n", mean, worst);
    check(worst <= std::ldexp(1.0, -5), "E5: every column within 2^-5 relative L2 of the exact arithmetic");
    check(mean <= std::ldexp(1.0, -7), "E5: the mean relative L2 within 2^-7");
}

void test_non_finite_and_zero() {
    std::mt19937 rng(5);
    const Expert e = fixtures::random_a16_expert(rng);
    auto x         = fixtures::random_activations(rng, 3);
    x[100]         = 0x7F80; // column 0: +inf
    std::fill(x.begin() + 2 * moe::kHidden, x.end(), std::uint16_t{0}); // column 2: zeros
    const auto y   = run(moe::best_cpu_isa(), e, x, 3);
    bool nan0 = true, finite1 = true, zero2 = true;
    for (int i = 0; i < moe::kHidden; ++i) {
        nan0    = nan0 && y[static_cast<std::size_t>(i)] == 0x7FC0;
        finite1 = finite1 && (y[static_cast<std::size_t>(moe::kHidden + i)] & 0x7F80) != 0x7F80;
        zero2   = zero2 && y[static_cast<std::size_t>(2 * moe::kHidden + i)] == 0;
    }
    check(nan0, "E6: a non-finite column gives NaN outputs");
    check(finite1, "E6: other columns are unaffected by a non-finite column");
    check(zero2, "E6: a zero column gives zero outputs");
}

} // namespace

INFERNIX_OP_TEST(offloaded_moe_a4r_cpu) {
    try {
        test_encoding();
        test_row_sums();
        test_isa_and_width();
        test_team();
        test_split_invariance();
        test_oracles();
        test_non_finite_and_zero();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all offloaded_moe two-term A4 CPU checks passed\n");
    return 0;
}
