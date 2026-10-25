#pragma once

// Independent oracles of the two-term A4 activations of W4A16 experts (expert-a4-residual-decode,
// canon::a4r_*; docs/maintainer/qwen3_8-flash-next-design.md §19.3.23), shared by the CPU test and the
// GPU narrow-route checks of offloaded_sparse_moe.
//
//   - oracle_terms: the term encoding in binary64 with each step rounded to binary32 (exact for +, -, *,
//     / of binary32 operands: 53 >= 2 * 24 + 2), and the E4M3 / E2M1 roundings as nearest-value
//     searches over the formats' value sets with ties to the even code. It shares no code with
//     canonical_math.h.
//   - oracle_expert: the expert in FP64 on the decoded terms (the fast path's represented inputs: the
//     terms are its semantic quantization boundary), with the canonical BF16 boundaries (y_gate, y_up, h)
//     reproduced from the oracle's own values and h re-encoded by oracle_terms.

#include "ops/offloaded_moe_fixtures.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace infernix::test::offloaded_moe {

inline constexpr int kOracleTerms = 2;

inline float oracle_f32(double v) { return static_cast<float>(v); }

inline float oracle_bf16_value(std::uint16_t v) {
    const std::uint32_t u = static_cast<std::uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

// The finite non-negative E4M3FN values by word (0x00..0x7E): subnormals m * 2^-9, normals
// (8 + m) * 2^(e - 10).
inline double oracle_e4m3(unsigned word) {
    const unsigned e = (word >> 3) & 15U, m = word & 7U;
    return e == 0 ? std::ldexp(static_cast<double>(m), -9) : std::ldexp(static_cast<double>(8 + m), static_cast<int>(e) - 10);
}

// Round to nearest E4M3FN word, ties to the even word, saturating at 448 (0x7E).
inline std::uint8_t oracle_e4m3_rn(double x) {
    if (x >= 448.0) { return 0x7E; }
    unsigned best = 0;
    for (unsigned w = 1; w <= 0x7E; ++w) {
        const double d = std::fabs(oracle_e4m3(w) - x), b = std::fabs(oracle_e4m3(best) - x);
        if (d < b || (d == b && (w & 1U) == 0)) { best = w; }
    }
    return static_cast<std::uint8_t>(best);
}

// Round to the nearest E2M1 magnitude {0, 0.5, 1, 1.5, 2, 3, 4, 6}, ties to the even code, saturating
// at 6; the sign of x (-0 included) sets bit 3.
inline std::uint8_t oracle_e2m1_rn(double x) {
    static constexpr double kValues[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double a = std::fabs(x);
    unsigned best  = 0;
    for (unsigned c = 1; c < 8; ++c) {
        const double d = std::fabs(kValues[c] - a), b = std::fabs(kValues[best] - a);
        if (d < b || (d == b && (c & 1U) == 0)) { best = c; }
    }
    if (a > 6.0) { best = 7; }
    return static_cast<std::uint8_t>((std::signbit(x) ? 8U : 0U) | best);
}

inline double oracle_e2m1_value(unsigned code) {
    static constexpr double kValues[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double m = kValues[code & 7U];
    return (code & 8U) != 0 ? -m : m;
}

struct OracleTerm {
    std::array<std::uint8_t, 16> code{};
    std::uint8_t scale = 0;
};

// The row scale of a BF16 row (NaN for a row with an Inf or NaN) and its multiplier.
inline float oracle_row_scale(const std::uint16_t* v, int n) {
    double amax = 0.0;
    for (int j = 0; j < n; ++j) {
        const float f = oracle_bf16_value(v[j]);
        if (!std::isfinite(f)) { return std::nanf(""); }
        amax = std::fmax(amax, std::fabs(static_cast<double>(f)));
    }
    return oracle_f32(amax / 2688.0);
}

inline float oracle_multiplier(float row_scale) {
    if (!(row_scale > 0.0F)) { return 0.0F; }
    const float d = oracle_f32(1.0 / static_cast<double>(row_scale));
    return std::isinf(d) ? 0.0F : d;
}

inline std::array<OracleTerm, kOracleTerms> oracle_terms(const std::uint16_t* v, float multiplier) {
    std::array<OracleTerm, kOracleTerms> out{};
    if (multiplier == 0.0F) { return out; }
    float rem[16];
    for (int j = 0; j < 16; ++j) { rem[j] = oracle_f32(static_cast<double>(oracle_bf16_value(v[j])) * multiplier); }
    for (int t = 0; t < kOracleTerms; ++t) {
        double amax = 0.0;
        for (const float r : rem) { amax = std::fmax(amax, std::fabs(static_cast<double>(r))); }
        out[t].scale = oracle_e4m3_rn(oracle_f32(amax / 6.0));
        if (out[t].scale == 0) { continue; }
        const double scale = oracle_e4m3(out[t].scale);
        for (int j = 0; j < 16; ++j) { out[t].code[j] = oracle_e2m1_rn(oracle_f32(static_cast<double>(rem[j]) / scale)); }
        if (t + 1 == kOracleTerms) { break; }
        for (int j = 0; j < 16; ++j) {
            rem[j] = oracle_f32(static_cast<double>(rem[j]) - oracle_e2m1_value(out[t].code[j]) * scale);
        }
    }
    return out;
}

// The decoded activation r * sum_t code * scale of each element of a row (exact in binary64), NaN for
// a non-finite row.
inline std::vector<double> oracle_decoded_row(const std::uint16_t* v, int n) {
    std::vector<double> out(static_cast<std::size_t>(n));
    const float rs = oracle_row_scale(v, n);
    const float d  = oracle_multiplier(rs);
    for (int b = 0; b < n / 16; ++b) {
        const auto terms = oracle_terms(v + 16 * b, d);
        for (int j = 0; j < 16; ++j) {
            double s = 0.0;
            for (const OracleTerm& t : terms) { s += oracle_e2m1_value(t.code[j]) * oracle_e4m3(t.scale); }
            out[static_cast<std::size_t>(16 * b + j)] = static_cast<double>(rs) * s;
        }
    }
    return out;
}

// Weight (doubled code times scaled block scale, i.e. 2^10 * the weight) of row `row` at k of a matrix
// of `blocks` blocks.
inline double oracle_weight(const std::uint8_t* matrix, int blocks, int row, int k) {
    const int rg = row / 16, r = row % 16, b = k / 16, kk = k % 16;
    const std::uint8_t* unit = matrix + (static_cast<std::size_t>(rg) * blocks + b) * moe::kUnitBytes;
    const std::uint8_t byte  = unit[32 * (kk / 4) + 4 * (r % 8) + kk % 4];
    const unsigned code      = r < 8 ? (byte & 15U) : (byte >> 4);
    return 2.0 * oracle_e2m1_value(code) * 512.0 * oracle_e4m3(unit[128 + r]);
}

struct OracleOutput {
    std::vector<double> y;   // [kHidden]
    std::vector<double> mag; // [kHidden]: sum of |terms| of each output, in output units
};

// One column of a Flash-Next W4A16 expert under the two-term A4 activations.
inline OracleOutput oracle_expert(const Expert& e, const std::uint16_t* x) {
    const std::uint8_t* gu = e.record.data();
    const std::uint8_t* dn = e.record.data() + moe::kGateUpBytes;
    const auto xd = oracle_decoded_row(x, moe::kHidden);
    std::vector<std::uint16_t> hq(moe::kIntermediate);
    for (int i = 0; i < moe::kIntermediate; ++i) {
        double g = 0.0, u = 0.0;
        for (int k = 0; k < moe::kHidden; ++k) {
            g += oracle_weight(gu, moe::kGateUpBlocks, 2 * i, k) * xd[static_cast<std::size_t>(k)];
            u += oracle_weight(gu, moe::kGateUpBlocks, 2 * i + 1, k) * xd[static_cast<std::size_t>(k)];
        }
        g *= std::ldexp(1.0, -10) * e.scales.alpha_gate;
        u *= std::ldexp(1.0, -10) * e.scales.alpha_up;
        const float gb = oracle_bf16_value(to_bf16(static_cast<float>(g)));
        const float ub = oracle_bf16_value(to_bf16(static_cast<float>(u)));
        const double h = gb / (1.0 + std::exp(-static_cast<double>(gb))) * ub;
        hq[static_cast<std::size_t>(i)] = to_bf16(static_cast<float>(h));
    }
    const auto hd = oracle_decoded_row(hq.data(), moe::kIntermediate);
    OracleOutput out{std::vector<double>(moe::kHidden), std::vector<double>(moe::kHidden)};
    const double scale = std::ldexp(1.0, -10) * e.scales.alpha_down;
    for (int row = 0; row < moe::kHidden; ++row) {
        double acc = 0.0, mag = 0.0;
        for (int k = 0; k < moe::kIntermediate; ++k) {
            const double t = oracle_weight(dn, moe::kDownBlocks, row, k) * hd[static_cast<std::size_t>(k)];
            acc += t;
            mag += std::fabs(t);
        }
        out.y[static_cast<std::size_t>(row)]   = acc * scale;
        out.mag[static_cast<std::size_t>(row)] = mag * scale;
    }
    return out;
}

// Outputs of `got` (BF16, one column) outside one BF16 rounding of the oracle plus a few BF16 ulps of
// propagated boundary differences (the oracle's and the route's y_gate / y_up / h can round different
// ways, and a flipped h element can move one of its terms' codes).
inline int oracle_outliers(const OracleOutput& ref, const std::uint16_t* got) {
    int bad = 0;
    for (int row = 0; row < moe::kHidden; ++row) {
        const double r   = ref.y[static_cast<std::size_t>(row)];
        const double g   = oracle_bf16_value(got[row]);
        const double tol = std::ldexp(std::fabs(r), -7) + ref.mag[static_cast<std::size_t>(row)] * std::ldexp(1.0, -7) + 1e-30;
        if (!(std::fabs(g - r) <= tol)) { ++bad; }
    }
    return bad;
}

} // namespace infernix::test::offloaded_moe
