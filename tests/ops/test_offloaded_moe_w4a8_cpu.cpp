// The 8-bit activation route of NVFP4 experts stored without activation scales on the CPU
// (src/ops/offloaded_sparse_moe/cpu/w4a8_expert.h; `--a8-activations`, LOSSY, opt-in), whose arithmetic is
// src/ops/offloaded_sparse_moe/nvfp4_a8.h's. Host-only.
//
// Records are random `nvfp4_expert_rg16_v1` records (offloaded_moe_fixtures.h: random codes, block scale words in
// the realistic range plus zero, subnormal and 448, random multipliers) at both registered geometries:
// Qwen3.8-Flash-Next's H 2560 / I 640 and Qwen3.5-122B-A10B's H 3072 / I 1024.
//
// The oracle is this file's transcription of the spec, sharing no code with the route beyond canonical_math.h's
// IEEE primitives and SwiGLU: ggml_a8.h's codec (d = fl32(m / 127), q = rne(fl32(v * fl32(127 / m))), zero below
// 2^-120, NaN for Inf or NaN), the weights decoded from the unit layout (design §6.2) with their own E2M1 and E4M3
// tables, each 16-element block's exact integer P_b, the terms fma(fl32(P_b * d_s), S^, acc) with slice s in
// partial s % P (16 for gate/up rows, 4 for down rows) in ascending s, block 2s then 2s + 1, from +0, the
// butterfly, the row output bf16(fl32(fl32(p_0 * 2^-10) * m)) (the power of two in nvfp4_a8.h's two exact steps of
// 2^-5), SwiGLU on the BF16 gate and up outputs, and h re-quantized for the down rows.
//
// The checks, on every ISA this host runs (--isa NAME: only that ISA, for a run under Intel SDE), bit for bit:
// - codec: encode_a8_slices' codes and scales;
// - rows: rg16_rows_a8's binary32 p_0 of every gate/up and down row at widths 1-8 (the scalar route, one code
//   path for every width: 1, 3 and 8; NaN rows: any NaN);
// - outputs: gate_up_units_a8's h and down_rows_a8's y (on the oracle's h, quantized) at the same widths with
//   uneven item splits, and expert_forward_a8 at widths 1, 3 and 8;
// - the worker team's 8-bit NVFP4 rounds (CpuExpertTeam, CpuExpertJob::a8) against expert_forward_a8 at 1, 3 and
//   16 workers, with and without shared x columns, and a team without the route refusing such a job.
// The columns: normal values, a wide dynamic range, an all-zero slice (both zero signs), slices whose values are
// all below 2^-120 (quantized to zero) beside a slice whose largest magnitude is 2^-120 (a subnormal scale), large
// values, and an infinity (every output of its column the canonical NaN). --columns (information only) times
// expert_forward_a8 against expert_forward_a16 on one thread at 1-8 columns.

#include "ops/op_family.h"
#include "ops/common/canonical_math.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/offloaded_sparse_moe/cpu/expert_team.h"
#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"
#include "ops/offloaded_sparse_moe/cpu/w4a8_expert.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace canon    = infernix::ops::canon;
namespace moe      = infernix::ops::offloaded_moe;
namespace fixtures = infernix::test::offloaded_moe;
using fixtures::Expert;

namespace {

int g_failures = 0;
std::string g_only_isa; // --isa: only this ISA's kernels (an emulated run)

constexpr int kCols = moe::kMaxColumns;
constexpr moe::ExpertShape kShapes[] = {moe::kFlashNextShape, moe::kQwen3_5_122BShape};
// The spec's partials: gate and up rows, down rows.
constexpr int kGateUpPartials = 16, kDownPartials = 4;

void fail(const std::string& message) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++g_failures;
}

std::uint32_t f32_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

float bf16_value(std::uint16_t b) {
    const std::uint32_t u = std::uint32_t{b} << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// The same binary32 value, any NaN equal to any NaN (NaN payloads and signs are not part of the contract: every
// NaN output is the canonical BF16 NaN).
bool same_f32(float a, float b) { return std::isnan(a) ? std::isnan(b) : f32_bits(a) == f32_bits(b); }

// ---------------------------------------------------------------------- the oracle, from the spec

struct Q8 {
    std::int32_t q[32];
    float d;
};

// ggml_a8.h's codec: m the slice's largest magnitude; Inf or NaN: d = NaN, q = 0; m below 2^-120 (BF16 0x0380):
// d = 0, q = 0; else d = fl32(m / 127), q = rne(fl32(v * fl32(127 / m))) clamped to [-127, 127].
Q8 quantize(const std::uint16_t* v) {
    unsigned m = 0;
    for (int j = 0; j < 32; ++j) { m = std::max(m, static_cast<unsigned>(v[j] & 0x7FFFU)); }
    Q8 o{};
    if (m >= 0x7F80U) {
        o.d = std::numeric_limits<float>::quiet_NaN();
        return o;
    }
    if (m < 0x0380U) { return o; }
    const float mf  = bf16_value(static_cast<std::uint16_t>(m));
    o.d             = mf / 127.0F;
    const float inv = 127.0F / mf;
    for (int j = 0; j < 32; ++j) {
        o.q[j] = std::clamp(static_cast<int>(std::nearbyint(bf16_value(v[j]) * inv)), -127, 127);
    }
    return o;
}

std::vector<Q8> quantize_column(const std::uint16_t* v, int n) {
    std::vector<Q8> out(static_cast<std::size_t>(n / 32));
    for (int s = 0; s < n / 32; ++s) { out[static_cast<std::size_t>(s)] = quantize(v + 32 * s); }
    return out;
}

// Twice the E2M1 value of each code (bit 3: the sign), and an E4M3FN scale word's value times 2^9.
constexpr int kE2m1x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

std::int32_t e4m3_x512(unsigned word) {
    const unsigned e = (word >> 3) & 15U, m = word & 7U;
    return e == 0 ? static_cast<std::int32_t>(m) : static_cast<std::int32_t>((8U + m) << (e - 1U));
}

// The p_0 of rows [0, rows) of a matrix of row-group-major 144-byte units (`blocks` blocks per row; unit byte
// 32 (k / 4) + 4 (r % 8) + k % 4 holds row r's code at k, rows 0-7 in the low nibbles, 8-15 in the high; byte
// 128 + r its scale word) with `partials` partials, against the quantized columns xq[c]: out[c][row].
void oracle_rows(const std::uint8_t* matrix, int rows, int blocks, int partials, const std::vector<std::vector<Q8>>& xq,
                 std::vector<std::vector<float>>& out) {
    const int slices = blocks / 2;
    out.assign(xq.size(), std::vector<float>(static_cast<std::size_t>(rows)));
    std::vector<int> c2(static_cast<std::size_t>(blocks) * 16);
    std::vector<std::int32_t> sw(static_cast<std::size_t>(blocks));
    for (int row = 0; row < rows; ++row) {
        const int rg = row / 16, r = row % 16;
        for (int b = 0; b < blocks; ++b) {
            const std::uint8_t* unit = matrix + (static_cast<std::size_t>(rg) * blocks + b) * moe::kUnitBytes;
            for (int k = 0; k < 16; ++k) {
                const std::uint8_t byte = unit[32 * (k / 4) + 4 * (r % 8) + k % 4];
                c2[static_cast<std::size_t>(b) * 16 + k] = kE2m1x2[r < 8 ? (byte & 15U) : (byte >> 4)];
            }
            sw[static_cast<std::size_t>(b)] = e4m3_x512(unit[128 + r]);
        }
        for (std::size_t c = 0; c < xq.size(); ++c) {
            float part[kGateUpPartials] = {};
            for (int s = 0; s < slices; ++s) {
                const Q8& x = xq[c][static_cast<std::size_t>(s)];
                for (int half = 0; half < 2; ++half) {
                    const int b    = 2 * s + half;
                    std::int32_t p = 0;
                    for (int j = 0; j < 16; ++j) { p += c2[static_cast<std::size_t>(b) * 16 + j] * x.q[16 * half + j]; }
                    float& acc = part[s % partials];
                    acc = canon::fma_rn(canon::mul_rn(static_cast<float>(p), x.d), static_cast<float>(sw[static_cast<std::size_t>(b)]), acc);
                }
            }
            for (int o = partials / 2; o >= 1; o /= 2) {
                for (int i = 0; i < o; ++i) { part[i] = canon::add_rn(part[i], part[i + o]); }
            }
            out[c][static_cast<std::size_t>(row)] = part[0];
        }
    }
}

std::uint16_t row_output(float p0, float multiplier) {
    const float scaled = canon::mul_rn(canon::mul_rn(p0, 0x1p-5F), 0x1p-5F);
    return canon::f32_to_bf16_rn(canon::mul_rn(scaled, multiplier));
}

// ---------------------------------------------------------------------------------- inputs

// kCols columns of `hidden` values: 0 normal with an outlier, 1 a wide dynamic range, 2 an all-zero slice (both
// zero signs) and a slice of a single nonzero value, 3 ten slices whose values are all below 2^-120 and a slice
// whose largest magnitude is 2^-120 (a subnormal scale), 4 an infinity, 5 large values (around 2^20, every
// output still finite), 6-7 normal.
std::vector<std::uint16_t> columns(std::mt19937& rng, int hidden) {
    std::vector<std::uint16_t> x(static_cast<std::size_t>(kCols) * hidden);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_real_distribution<float> lg(-30.0F, 12.0F);
    for (int c = 0; c < kCols; ++c) {
        std::uint16_t* v = x.data() + static_cast<std::size_t>(c) * hidden;
        for (int j = 0; j < hidden; ++j) {
            float f = normal(rng) * 0.05F;
            if (c == 1) { f = std::exp2(lg(rng)) * (j % 2 ? -1.0F : 1.0F); }
            if (c == 5) { f = normal(rng) * 0x1p20F; }
            v[j] = canon::f32_to_bf16_rn(f);
        }
        if (c != 5) { v[17 * (c + 1) % hidden] = canon::f32_to_bf16_rn(9.0F); }
    }
    std::uint16_t* zero = x.data() + static_cast<std::size_t>(2) * hidden;
    for (int j = 0; j < 32; ++j) {
        zero[96 + j]  = static_cast<std::uint16_t>(j % 2 ? 0x8000U : 0U); // slice 3
        zero[224 + j] = static_cast<std::uint16_t>(j % 3 ? 0x8000U : 0U); // slice 7: one value
    }
    zero[224 + 5] = 0xBF80; // -1
    std::uint16_t* tiny = x.data() + static_cast<std::size_t>(3) * hidden;
    for (int j = 0; j < 11 * 32; ++j) {
        tiny[j] = static_cast<std::uint16_t>((rng() % 0x0380U) | (j % 3 == 0 ? 0x8000U : 0U)); // slices 0-10
    }
    tiny[4]       = 0x0001;  // the smallest BF16 subnormal
    tiny[9]       = 0x837F;  // the largest magnitude below 2^-120
    tiny[320 + 5] = 0x8380;  // slice 10: -2^-120
    x[static_cast<std::size_t>(4) * hidden + 100] = 0x7F80; // +inf
    return x;
}

// The columns of a width-n call: n consecutive columns from (3 n) mod kCols, so each width meets other specials.
int column_of(int ncols, int c) { return (3 * ncols + c) % kCols; }

std::size_t count_diff(const std::uint16_t* a, const std::uint16_t* b, std::size_t n) {
    std::size_t bad = 0;
    for (std::size_t i = 0; i < n; ++i) { bad += a[i] != b[i]; }
    return bad;
}

void check_codec(const std::uint16_t* v, int n, const std::vector<Q8>& want, const std::string& what) {
    std::vector<moe::CpuW4A8Slice> got(static_cast<std::size_t>(n / 32));
    moe::encode_a8_slices(v, 0, n / 32, got.data());
    int bad = 0;
    for (int s = 0; s < n / 32; ++s) {
        const moe::CpuW4A8Slice& g = got[static_cast<std::size_t>(s)];
        const Q8& w                = want[static_cast<std::size_t>(s)];
        bool same                  = same_f32(g.d, w.d);
        for (int j = 0; j < 32; ++j) { same = same && g.q[j] == w.q[j]; }
        bad += !same;
    }
    if (bad != 0) { fail(what + ": " + std::to_string(bad) + " slices differ from the codec's formula"); }
}

// ------------------------------------------------------------------------------ one geometry

struct Oracle {
    std::vector<std::vector<float>> gate_up, down;   // p_0 [col][row]
    std::vector<std::vector<std::uint16_t>> h, y;    // BF16 [col][I], [col][H]
};

Oracle oracle(const moe::ExpertShape& shape, const Expert& e, const std::vector<std::uint16_t>& x) {
    const int H = shape.hidden, I = shape.intermediate;
    Oracle o;
    std::vector<std::vector<Q8>> xq(kCols), hq(kCols);
    for (int c = 0; c < kCols; ++c) { xq[static_cast<std::size_t>(c)] = quantize_column(x.data() + static_cast<std::size_t>(c) * H, H); }
    oracle_rows(e.record.data(), 2 * I, H / 16, kGateUpPartials, xq, o.gate_up);
    o.h.assign(kCols, std::vector<std::uint16_t>(static_cast<std::size_t>(I)));
    for (int c = 0; c < kCols; ++c) {
        const auto& p = o.gate_up[static_cast<std::size_t>(c)];
        for (int t = 0; t < I; ++t) { // gate_t is row 2t, up_t row 2t + 1
            o.h[static_cast<std::size_t>(c)][static_cast<std::size_t>(t)] =
                canon::swiglu_bf16(row_output(p[static_cast<std::size_t>(2 * t)], e.scales.alpha_gate),
                                   row_output(p[static_cast<std::size_t>(2 * t + 1)], e.scales.alpha_up));
        }
        hq[static_cast<std::size_t>(c)] = quantize_column(o.h[static_cast<std::size_t>(c)].data(), I);
    }
    oracle_rows(e.record.data() + shape.gate_up_bytes(), H, I / 16, kDownPartials, hq, o.down);
    o.y.assign(kCols, std::vector<std::uint16_t>(static_cast<std::size_t>(H)));
    for (int c = 0; c < kCols; ++c) {
        for (int r = 0; r < H; ++r) {
            o.y[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)] =
                row_output(o.down[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)], e.scales.alpha_down);
        }
    }
    // The inputs do what the header says: column 3's tiny slices quantize to zero beside a subnormal scale, column
    // 4's outputs are all the canonical NaN, every other column's are finite and mostly nonzero.
    if (!(f32_bits(xq[3][0].d) == 0 && xq[3][10].d > 0.0F && xq[3][10].d < FLT_MIN)) {
        fail("the tiny column must hold zero-scale slices and a subnormal scale");
    }
    for (int c = 0; c < kCols; ++c) {
        const auto& y = o.y[static_cast<std::size_t>(c)];
        if (c == 4) {
            if (std::any_of(y.begin(), y.end(), [](std::uint16_t v) { return v != 0x7FC0; })) {
                fail("the infinite column's outputs must be the canonical NaN");
            }
            continue;
        }
        const auto finite  = std::count_if(y.begin(), y.end(), [](std::uint16_t v) { return (v & 0x7F80) != 0x7F80; });
        const auto nonzero = std::count_if(y.begin(), y.end(), [](std::uint16_t v) { return (v & 0x7FFF) != 0; });
        if (finite != static_cast<std::ptrdiff_t>(y.size()) || nonzero < static_cast<std::ptrdiff_t>(y.size() * 9 / 10)) {
            fail("column " + std::to_string(c) + "'s outputs must be finite and mostly nonzero");
        }
    }
    for (int c = 0; c < kCols; ++c) {
        check_codec(x.data() + static_cast<std::size_t>(c) * H, H, xq[static_cast<std::size_t>(c)], "x column " + std::to_string(c));
        check_codec(o.h[static_cast<std::size_t>(c)].data(), I, hq[static_cast<std::size_t>(c)], "h column " + std::to_string(c));
    }
    return o;
}

void check_isa(moe::CpuIsa isa, const moe::ExpertShape& shape, const Expert& e, const std::vector<std::uint16_t>& x,
               const Oracle& o) {
    const int H = shape.hidden, I = shape.intermediate;
    const int x_slices = H / 32, h_slices = I / 32, units = shape.h_blocks(), groups = shape.down_row_groups();
    const std::uint8_t* gate_up = e.record.data();
    const std::uint8_t* down    = e.record.data() + shape.gate_up_bytes();
    const std::string where     = "H " + std::to_string(H) + ", " + moe::cpu_isa_name(isa);
    for (int ncols = 1; ncols <= kCols; ++ncols) {
        // The scalar route has one code path for every width (the vector variants one per width): 1, 3 and 8 only.
        if (isa == moe::CpuIsa::kScalar && ncols != 1 && ncols != 3 && ncols != kCols) { continue; }
        const std::string at = where + ", " + std::to_string(ncols) + " columns: ";
        std::vector<moe::CpuW4A8Slice> xs(static_cast<std::size_t>(ncols) * x_slices), hs(static_cast<std::size_t>(ncols) * h_slices);
        std::vector<std::uint16_t> h(static_cast<std::size_t>(ncols) * I, 0xFFFF), y(static_cast<std::size_t>(ncols) * H, 0xFFFF);
        const moe::CpuW4A8Slice* xp[kCols];
        const moe::CpuW4A8Slice* hp[kCols];
        const std::uint16_t* xv[kCols];
        std::uint16_t* hv[kCols];
        std::uint16_t* yv[kCols];
        for (int c = 0; c < ncols; ++c) {
            const auto col = static_cast<std::size_t>(column_of(ncols, c));
            xv[c]          = x.data() + col * H;
            xp[c]          = &xs[static_cast<std::size_t>(c) * x_slices];
            hp[c]          = &hs[static_cast<std::size_t>(c) * h_slices];
            hv[c]          = h.data() + static_cast<std::size_t>(c) * I;
            yv[c]          = y.data() + static_cast<std::size_t>(c) * H;
            // x in two uneven pieces, as the worker team's chunks; the down rows read the oracle's h.
            moe::encode_a8_slices(xv[c], 0, 7, &xs[static_cast<std::size_t>(c) * x_slices]);
            moe::encode_a8_slices(xv[c], 7, x_slices, &xs[static_cast<std::size_t>(c) * x_slices]);
            moe::encode_a8_slices(o.h[col].data(), 0, h_slices, &hs[static_cast<std::size_t>(c) * h_slices]);
        }
        // Rows: every p_0 against the oracle's binary32 value.
        std::vector<float> p(static_cast<std::size_t>(2 * I) * ncols);
        moe::rg16_rows_a8(isa, gate_up, H / 16, kGateUpPartials, 0, 2 * I / 16, xp, ncols, p.data());
        std::size_t bad = 0;
        for (int row = 0; row < 2 * I; ++row) {
            for (int c = 0; c < ncols; ++c) {
                bad += !same_f32(p[static_cast<std::size_t>(row) * ncols + c],
                                 o.gate_up[static_cast<std::size_t>(column_of(ncols, c))][static_cast<std::size_t>(row)]);
            }
        }
        if (bad != 0) { fail(at + std::to_string(bad) + " gate/up p_0 differ from the oracle"); }
        p.assign(static_cast<std::size_t>(H) * ncols, 0.0F);
        moe::rg16_rows_a8(isa, down, I / 16, kDownPartials, 0, groups, hp, ncols, p.data());
        bad = 0;
        for (int row = 0; row < H; ++row) {
            for (int c = 0; c < ncols; ++c) {
                bad += !same_f32(p[static_cast<std::size_t>(row) * ncols + c],
                                 o.down[static_cast<std::size_t>(column_of(ncols, c))][static_cast<std::size_t>(row)]);
            }
        }
        if (bad != 0) { fail(at + std::to_string(bad) + " down p_0 differ from the oracle"); }
        // Outputs, in uneven item splits.
        const int ucuts[] = {0, 1, 7, 22, units};
        for (int i = 0; i + 1 < 5; ++i) { moe::gate_up_units_a8(isa, shape, e.record.data(), e.scales, xp, ncols, ucuts[i], ucuts[i + 1], hv); }
        const int rcuts[] = {0, 2, 37, 101, groups};
        for (int i = 0; i + 1 < 5; ++i) { moe::down_rows_a8(isa, shape, e.record.data(), e.scales, hp, ncols, rcuts[i], rcuts[i + 1], yv); }
        std::size_t bad_h = 0, bad_y = 0;
        for (int c = 0; c < ncols; ++c) {
            const auto col = static_cast<std::size_t>(column_of(ncols, c));
            bad_h += count_diff(hv[c], o.h[col].data(), static_cast<std::size_t>(I));
            bad_y += count_diff(yv[c], o.y[col].data(), static_cast<std::size_t>(H));
        }
        if (bad_h != 0) { fail(at + std::to_string(bad_h) + " h outputs differ from the oracle"); }
        if (bad_y != 0) { fail(at + std::to_string(bad_y) + " down outputs differ from the oracle"); }
        if (ncols == 1 || ncols == 3 || ncols == kCols) {
            std::vector<std::uint16_t> f(static_cast<std::size_t>(ncols) * H, 0xFFFF);
            std::uint16_t* fv[kCols];
            for (int c = 0; c < ncols; ++c) { fv[c] = f.data() + static_cast<std::size_t>(c) * H; }
            moe::expert_forward_a8(isa, shape, e.record.data(), e.scales, ncols, xv, fv);
            std::size_t bad_f = 0;
            for (int c = 0; c < ncols; ++c) {
                bad_f += count_diff(fv[c], o.y[static_cast<std::size_t>(column_of(ncols, c))].data(), static_cast<std::size_t>(H));
            }
            if (bad_f != 0) { fail(at + std::to_string(bad_f) + " expert_forward_a8 outputs differ from the oracle"); }
        }
    }
}

// The worker team's 8-bit NVFP4 rounds (CpuExpertTeam::work_a8) against expert_forward_a8: three jobs of one, two
// and three columns of distinct experts at 1, 3 and 16 workers, the jobs' columns shared (the same x pointers, as
// in a decode round).
void check_team(const moe::ExpertShape& shape, std::mt19937& rng) {
    const int H = shape.hidden;
    constexpr int kJobs = 3, kJobCols = 3;
    std::vector<Expert> experts;
    for (int j = 0; j < kJobs; ++j) { experts.push_back(fixtures::random_a16_expert(rng, shape)); }
    const auto x       = columns(rng, H);
    const auto x_of    = [&](int c) { return x.data() + static_cast<std::size_t>(c + 1) * H; }; // columns 1-3
    const auto y_index = [&](int j, int c) { return static_cast<std::size_t>(j * kJobCols + c) * H; };
    std::vector<std::uint16_t> want(static_cast<std::size_t>(H) * kJobCols * kJobs, 0);
    for (int j = 0; j < kJobs; ++j) {
        const std::uint16_t* xp[kJobCols];
        std::uint16_t* yp[kJobCols];
        for (int c = 0; c < kJobCols; ++c) {
            xp[c] = x_of(c);
            yp[c] = want.data() + y_index(j, c);
        }
        moe::expert_forward_a8(moe::best_cpu_isa(), shape, experts[static_cast<std::size_t>(j)].record.data(),
                               experts[static_cast<std::size_t>(j)].scales, j + 1, xp, yp);
    }
    std::vector<moe::CpuExpertJob> jobs(kJobs);
    std::vector<std::uint16_t> got(want.size(), 0);
    for (int j = 0; j < kJobs; ++j) {
        auto& job  = jobs[static_cast<std::size_t>(j)];
        job.record = experts[static_cast<std::size_t>(j)].record.data();
        job.scales = experts[static_cast<std::size_t>(j)].scales;
        job.a8     = true;
        job.ncols  = j + 1;
        for (int c = 0; c < job.ncols; ++c) {
            job.x[c] = x_of(c);
            job.y[c] = got.data() + y_index(j, c);
        }
    }
    for (const int workers : {1, 3, 16}) {
        moe::CpuExpertTeam team({.workers    = workers,
                                 .max_jobs   = kJobs,
                                 .activation = moe::ExpertActivation::kA16,
                                 .nvfp4_a8   = true,
                                 .shape      = shape});
        std::fill(got.begin(), got.end(), 0);
        team.run(jobs);
        if (got != want) {
            fail("H " + std::to_string(H) + ": the worker team's 8-bit NVFP4 round at " + std::to_string(workers) +
                 " workers differs from expert_forward_a8");
        }
    }
    // A team without the route has no buffers for it: such a job is refused, not computed on the A16 route.
    moe::CpuExpertTeam exact({.workers = 1, .max_jobs = kJobs, .activation = moe::ExpertActivation::kA16, .shape = shape});
    bool refused = false;
    try {
        exact.run(jobs);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    if (!refused) { fail("a team without nvfp4_a8 must refuse an 8-bit NVFP4 job"); }
}

// --columns (information only): one expert call on one thread at 1-8 columns, every supported vector ISA, on the
// 8-bit route and on the exact A16 route, at both geometries.
void columns_timing() {
    std::mt19937 rng(20261027);
    for (const moe::ExpertShape& shape : kShapes) {
        const int H  = shape.hidden;
        const Expert e = fixtures::random_a16_expert(rng, shape);
        const auto x   = fixtures::random_activations(rng, kCols, H);
        std::vector<std::uint16_t> y(static_cast<std::size_t>(H) * kCols);
        for (const auto isa : {moe::CpuIsa::kAvx2, moe::CpuIsa::kAvxVnni, moe::CpuIsa::kAvx512Vnni}) {
            if (!moe::cpu_isa_supported(isa)) { continue; }
            for (int ncols = 1; ncols <= kCols; ++ncols) {
                const std::uint16_t* xp[kCols];
                std::uint16_t* yp[kCols];
                for (int c = 0; c < ncols; ++c) {
                    xp[c] = x.data() + static_cast<std::size_t>(c) * H;
                    yp[c] = y.data() + static_cast<std::size_t>(c) * H;
                }
                constexpr int kRepeats = 30;
                double us[2];
                for (int route = 0; route < 2; ++route) {
                    const auto call = [&] {
                        if (route == 0) {
                            moe::expert_forward_a8(isa, shape, e.record.data(), e.scales, ncols, xp, yp);
                        } else {
                            moe::expert_forward_a16(isa, shape, e.record.data(), e.scales, ncols, xp, yp);
                        }
                    };
                    call();
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < kRepeats; ++i) { call(); }
                    us[route] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / kRepeats;
                }
                std::printf("columns H %d %-12s ncols %d: a8 %8.1f us per call (%7.1f per column), a16 %8.1f us\n", H,
                            moe::cpu_isa_name(isa), ncols, us[0], us[0] / ncols, us[1]);
            }
        }
    }
}

} // namespace

INFERNIX_OP_TEST(offloaded_moe_w4a8_cpu) {
    if (argc > 1 && std::string(argv[1]) == "--columns") {
        columns_timing();
        return 0;
    }
    // --isa NAME (cpu_isa_name, e.g. avx512-vnni): only that ISA's kernels against the oracle, for a run under an
    // emulator (Intel SDE) on a CPU without it.
    if (argc > 2 && std::string(argv[1]) == "--isa") {
        g_only_isa = argv[2];
        if (moe::cpu_isa_name(moe::best_cpu_isa()) != g_only_isa) {
            std::fprintf(stderr, "error: --isa %s is not this CPU's best ISA (%s)\n", argv[2], moe::cpu_isa_name(moe::best_cpu_isa()));
            return 1;
        }
    }
    try {
        const moe::CpuIsa isas[] = {moe::CpuIsa::kScalar, moe::CpuIsa::kAvx2, moe::CpuIsa::kAvxVnni, moe::CpuIsa::kAvx512Vnni};
        for (const auto isa : isas) {
            const bool run = moe::cpu_isa_supported(isa) && (g_only_isa.empty() || g_only_isa == moe::cpu_isa_name(isa));
            std::printf("CPU ISA %-12s %s\n", moe::cpu_isa_name(isa), run ? "tested" : "skipped (not supported or not selected)");
        }
        std::mt19937 rng(20261027);
        for (const moe::ExpertShape& shape : kShapes) {
            const Expert e = fixtures::random_a16_expert(rng, shape);
            const auto x   = columns(rng, shape.hidden);
            const Oracle o = oracle(shape, e, x);
            for (const auto isa : isas) {
                if (!moe::cpu_isa_supported(isa) || (!g_only_isa.empty() && g_only_isa != moe::cpu_isa_name(isa))) { continue; }
                check_isa(isa, shape, e, x, o);
            }
            check_team(shape, rng);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d W4A8 CPU expert checks failed\n", g_failures);
        return 1;
    }
    std::printf("W4A8 CPU experts: codec, both geometries, every ISA and width and the worker team bit-equal to the spec's oracle\n");
    return 0;
}
