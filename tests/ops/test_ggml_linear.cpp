// ggml_linear, ggml_embedding and ple_embed_iq4_nl against an independent exact oracle
// (docs/maintainer/qwen3_8-flash-next-design.md §16.2.2).
//
// Weights are random block bytes (every byte pattern is a valid encoding once the binary16 scales
// are finite) decoded by this file's own transcription of llama.cpp's dequantize_row_* formulas;
// every decoded value is a multiple of 2^-27 below 2^28 in magnitude, so binary64 holds it exactly.
// Activations are BF16 columns encoded by the A16 rule (canon::a16_value, the semantic boundary the
// Op documents). The oracle sums w * X exactly in a bignum, rounds once to binary32 (ties to even),
// scales by 2^(emax - 147) (exact for these normal-range outputs) and rounds to BF16. The Op must
// return those bits for every format, K, column count (across its 512-column batches) and with the
// input permutation, into one output or split among several (ggml_split_segments). The
// ggml-bf16-prefill route (run_bf16) is checked against a binary64 reference within an FP32-sum
// bound instead. The row gathers must return each element's decoded value (exact in binary32)
// rounded once to BF16.

#include "core/device.h"
#include "core/ggml_format.h"
#include "infernix/ops/ggml_linear.h"
#include "infernix/ops/ggml_rows.h"
#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_tables.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <algorithm>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace infernix;

namespace {

// ------------------------------------------------------------------------------- reference decode

float f16(std::uint16_t w) {
    const int sign = (w >> 15) & 1, e = (w >> 10) & 0x1F, m = w & 0x3FF;
    const double v = e == 0 ? std::ldexp(m, -24) : std::ldexp(1024 + m, e - 25);
    return static_cast<float>(sign ? -v : v);
}

std::uint16_t u16(const std::uint8_t* p) { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }

// One block's exact values (llama.cpp ggml-quants.c dequantize_row_*, evaluated in binary64).
void decode_block(QType q, const std::uint8_t* b, double* y) {
    using namespace infernix::ggml;
    switch (q) {
    case QType::GGML_Q2_0: {
        const double d = f16(u16(b));
        for (int j = 0; j < 64; ++j) { y[j] = ((b[2 + j / 4] >> (2 * (j % 4))) & 3) - 1.0; y[j] *= d; }
        break;
    }
    case QType::GGML_IQ4_NL: {
        const double d = f16(u16(b));
        for (int j = 0; j < 16; ++j) {
            y[j]      = d * k_kvalues_iq4nl[b[2 + j] & 0xF];
            y[j + 16] = d * k_kvalues_iq4nl[b[2 + j] >> 4];
        }
        break;
    }
    case QType::GGML_IQ4_XS: {
        const double d = f16(u16(b));
        const int scales_h = u16(b + 2);
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((b[4 + ib / 2] >> 4 * (ib % 2)) & 0xF) | (((scales_h >> 2 * ib) & 3) << 4);
            const double dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[32 * ib + j]      = dl * k_kvalues_iq4nl[b[8 + 16 * ib + j] & 0xF];
                y[32 * ib + j + 16] = dl * k_kvalues_iq4nl[b[8 + 16 * ib + j] >> 4];
            }
        }
        break;
    }
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K: {
        const double d = f16(u16(b)), dmin = f16(u16(b + 2));
        const std::uint8_t* s = b + 4;
        const auto scale_min = [&](int j, int& sc, int& m) {
            if (j < 4) { sc = s[j] & 63; m = s[j + 4] & 63; }
            else { sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4); m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
        };
        const bool five = q == QType::GGML_Q5_K;
        const std::uint8_t* qh = b + 16;
        const std::uint8_t* ql = b + (five ? 48 : 16);
        for (int ch = 0; ch < 4; ++ch) {
            int sc1, m1, sc2, m2;
            scale_min(2 * ch, sc1, m1);
            scale_min(2 * ch + 1, sc2, m2);
            for (int l = 0; l < 32; ++l) {
                const int lo = (ql[32 * ch + l] & 0xF) + (five && (qh[l] >> (2 * ch) & 1) ? 16 : 0);
                const int hi = (ql[32 * ch + l] >> 4) + (five && (qh[l] >> (2 * ch + 1) & 1) ? 16 : 0);
                y[64 * ch + l]      = d * sc1 * lo - dmin * m1;
                y[64 * ch + 32 + l] = d * sc2 * hi - dmin * m2;
            }
        }
        break;
    }
    case QType::GGML_Q6_K: {
        const double d = f16(u16(b + 208));
        for (int n = 0; n < 2; ++n) {
            const std::uint8_t* ql = b + 64 * n;
            const std::uint8_t* qh = b + 128 + 32 * n;
            const auto* sc = reinterpret_cast<const std::int8_t*>(b + 192 + 8 * n);
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                y[128 * n + l]      = d * sc[is + 0] * (((ql[l] & 0xF) | ((qh[l] >> 0 & 3) << 4)) - 32);
                y[128 * n + l + 32] = d * sc[is + 2] * (((ql[l + 32] & 0xF) | ((qh[l] >> 2 & 3) << 4)) - 32);
                y[128 * n + l + 64] = d * sc[is + 4] * (((ql[l] >> 4) | ((qh[l] >> 4 & 3) << 4)) - 32);
                y[128 * n + l + 96] = d * sc[is + 6] * (((ql[l + 32] >> 4) | ((qh[l] >> 6 & 3) << 4)) - 32);
            }
        }
        break;
    }
    case QType::GGML_IQ2_S: {
        const double d = f16(u16(b));
        const std::uint8_t *qs = b + 2, *signs = b + 34, *qh = b + 66, *scales = b + 74;
        for (int ib = 0; ib < 8; ++ib) {
            for (int l = 0; l < 4; ++l) {
                const double dl = d * (0.5 + ((scales[ib] >> 4 * (l / 2)) & 0xF)) * 0.25;
                const std::uint64_t g = k_iq2s_grid[qs[4 * ib + l] | ((qh[ib] << (8 - 2 * l)) & 0x300)];
                for (int j = 0; j < 8; ++j) {
                    y[32 * ib + 8 * l + j] = dl * ((g >> 8 * j) & 0xFF) * ((signs[4 * ib + l] & k_kmask_iq2xs[j]) ? -1 : 1);
                }
            }
        }
        break;
    }
    case QType::GGML_IQ3_XXS: {
        const double d = f16(u16(b));
        for (int ib = 0; ib < 8; ++ib) {
            std::uint32_t aux;
            std::memcpy(&aux, b + 66 + 4 * ib, 4);
            const double db = d * (0.5 + (aux >> 28)) * 0.5;
            for (int l = 0; l < 4; ++l) {
                const std::uint8_t signs = k_ksigns_iq2xs[(aux >> 7 * l) & 127];
                const std::uint32_t g1 = k_iq3xxs_grid[b[2 + 8 * ib + 2 * l]], g2 = k_iq3xxs_grid[b[2 + 8 * ib + 2 * l + 1]];
                for (int j = 0; j < 4; ++j) {
                    y[32 * ib + 8 * l + j]     = db * ((g1 >> 8 * j) & 0xFF) * ((signs & k_kmask_iq2xs[j]) ? -1 : 1);
                    y[32 * ib + 8 * l + j + 4] = db * ((g2 >> 8 * j) & 0xFF) * ((signs & k_kmask_iq2xs[j + 4]) ? -1 : 1);
                }
            }
        }
        break;
    }
    case QType::GGML_IQ3_S: {
        const double d = f16(u16(b));
        const std::uint8_t *qs = b + 2, *qh = b + 66, *signs = b + 74, *scales = b + 106;
        for (int sb = 0; sb < 8; ++sb) {
            const double db = d * (1 + 2 * ((scales[sb / 2] >> 4 * (sb % 2)) & 0xF));
            for (int l = 0; l < 4; ++l) {
                const std::uint32_t g1 = k_iq3s_grid[qs[8 * sb + 2 * l] | ((qh[sb] << (8 - 2 * l)) & 256)];
                const std::uint32_t g2 = k_iq3s_grid[qs[8 * sb + 2 * l + 1] | ((qh[sb] << (7 - 2 * l)) & 256)];
                for (int j = 0; j < 4; ++j) {
                    y[32 * sb + 8 * l + j]     = db * ((g1 >> 8 * j) & 0xFF) * ((signs[4 * sb + l] & k_kmask_iq2xs[j]) ? -1 : 1);
                    y[32 * sb + 8 * l + j + 4] = db * ((g2 >> 8 * j) & 0xFF) * ((signs[4 * sb + l] & k_kmask_iq2xs[j + 4]) ? -1 : 1);
                }
            }
        }
        break;
    }
    default:
        throw std::logic_error("decode_block: format");
    }
}

// ------------------------------------------------------------------------------------- bignum

// A signed fixed-point accumulator: sum of terms (integer * 2^-27), as base-2^32 limbs.
struct Big {
    std::vector<std::int64_t> limbs = std::vector<std::int64_t>(8, 0); // value = sum limbs[i] * 2^(32 i)
    void add(std::int64_t mantissa_2m27, std::int64_t x) { // adds mantissa * x
        // |mantissa| < 2^56, |x| < 2^21: split mantissa into 32-bit halves so every product fits.
        const std::int64_t lo = mantissa_2m27 & 0xFFFFFFFF, hi = mantissa_2m27 >> 32;
        limbs[0] += lo * x; // < 2^53
        limbs[1] += hi * x; // < 2^45
        normalize();
    }
    void normalize() {
        for (std::size_t i = 0; i + 1 < limbs.size(); ++i) {
            const std::int64_t carry = limbs[i] >> 32; // floor division
            limbs[i] -= carry << 32;
            limbs[i + 1] += carry;
        }
    }
    // Round-to-nearest-even binary32 of the value times 2^-27, then times 2^k.
    float to_float(int k) const {
        // Collect the value as sign and magnitude bits.
        std::vector<std::int64_t> v = limbs;
        const bool negative = v.back() < 0;
        if (negative) {
            for (auto& l : v) { l = -l; }
            Big t;
            t.limbs = v;
            t.normalize();
            v = t.limbs;
        }
        // v limbs now in [0, 2^32) except the top one.
        std::vector<int> bits; // little-endian
        for (std::size_t i = 0; i < v.size(); ++i) {
            for (int j = 0; j < 32; ++j) { bits.push_back(static_cast<int>((v[i] >> j) & 1)); }
        }
        int msb = static_cast<int>(bits.size()) - 1;
        while (msb >= 0 && bits[static_cast<std::size_t>(msb)] == 0) { --msb; }
        if (msb < 0) { return negative ? -0.0F : 0.0F; }
        double mant = 0;
        int lsb = msb - 23; // keep 24 bits
        if (lsb < 0) { lsb = 0; }
        for (int i = msb; i >= lsb; --i) { mant = mant * 2 + bits[static_cast<std::size_t>(i)]; }
        if (lsb > 0) {
            const bool half = bits[static_cast<std::size_t>(lsb - 1)] != 0;
            bool sticky = false;
            for (int i = 0; i < lsb - 1; ++i) { sticky = sticky || bits[static_cast<std::size_t>(i)] != 0; }
            if (half && (sticky || std::fmod(mant, 2.0) != 0.0)) { mant += 1; }
        }
        const double value = std::ldexp(mant, lsb - 27 + k);
        return static_cast<float>(negative ? -value : value);
    }
};

std::uint16_t bf16_rne(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

// ------------------------------------------------------------------------------------- the check

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL " << what << "\n";
    }
}

// Block scales, binary16 exponent fields (the tiles' accumulators: ggml_tile.cuh row_bases):
// Narrow 12..13 (as the GSQ-RCO model's rows: almost all within one binade); Wide 5..20 (spans of
// up to 15: per-limb int64 for the formats without a min term, int128 for the K-quants); Wider
// 1..30 (spans past 20: int128); Edge 1 or 21 with the largest mantissa (span 20, |M| = 2047 * 2^20:
// the per-limb form's limit where a format's sum bound allows it, else int128).
enum class Scales { Wide, Narrow, Wider, Edge };

std::vector<std::uint8_t> random_blocks(QType q, int rows, int k, std::mt19937& rng, Scales scales = Scales::Wide) {
    const int blocks = k / static_cast<int>(ggml_block_elems(q)), size = static_cast<int>(ggml_block_bytes(q));
    std::vector<std::uint8_t> data(static_cast<std::size_t>(rows) * blocks * size);
    for (auto& v : data) { v = static_cast<std::uint8_t>(rng()); }
    const auto field = [&]() -> unsigned {
        switch (scales) {
        case Scales::Narrow: return 12 + rng() % 2;
        case Scales::Wider: return 1 + rng() % 30;
        case Scales::Edge: return rng() % 2 == 0 ? 1 : 21;
        default: return 5 + rng() % 16;
        }
    };
    const auto scale = [&](std::uint8_t* p) {
        const unsigned mant = scales == Scales::Edge ? 0x3FFU : rng() & 0x3FFU;
        const std::uint16_t w = static_cast<std::uint16_t>((rng() & 0x8000U) | mant | (field() << 10));
        p[0] = static_cast<std::uint8_t>(w & 0xFF);
        p[1] = static_cast<std::uint8_t>(w >> 8);
    };
    for (int i = 0; i < rows * blocks; ++i) {
        std::uint8_t* b = data.data() + static_cast<std::size_t>(i) * size;
        if (q == QType::GGML_Q6_K) {
            scale(b + 208);
        } else {
            scale(b);
            if (q == QType::GGML_Q4_K || q == QType::GGML_Q5_K) { scale(b + 2); }
        }
    }
    return data;
}

// Q6_K rows at the tiles' int64 bound (K = 6144: 24 blocks, scale shifts spanning 62 - 52 - 5 = 5
// binades): every code -32 with sub-scale -128 and mantissa 2047, so every block term is within a
// factor 2 of its bound; with all-equal inputs at the column maximum the sums reach ~2^61.
std::vector<std::uint8_t> extreme_q6k(int rows, int k) {
    const int blocks = k / 256;
    std::vector<std::uint8_t> data(static_cast<std::size_t>(rows) * blocks * 210, 0); // codes 0: -32
    for (int i = 0; i < rows * blocks; ++i) {
        std::uint8_t* b = data.data() + static_cast<std::size_t>(i) * 210;
        for (int j = 192; j < 208; ++j) { b[j] = 0x80; } // sub-scales -128
        const std::uint16_t w = static_cast<std::uint16_t>(0x3FF | ((i % 2 == 0 ? 10 : 15) << 10));
        b[208] = static_cast<std::uint8_t>(w & 0xFF);
        b[209] = static_cast<std::uint8_t>(w >> 8);
    }
    return data;
}

// The oracle's outputs [rows, columns] of `weights` (rows of q blocks) over x (BF16 [k, columns]),
// column element j read from x[perm(j)] (perm: blocks of pb, null for none): binary32 bits when
// `f32`, else BF16 bits.
std::vector<std::uint32_t> oracle_bits(QType q, const std::vector<std::uint8_t>& weights, int rows, int k,
                                       const std::vector<std::uint16_t>& x, int columns,
                                       const std::vector<std::int32_t>* perm, int pb, bool f32) {
    const int row_bytes = k / static_cast<int>(ggml_block_elems(q)) * static_cast<int>(ggml_block_bytes(q));
    std::vector<double> w(static_cast<std::size_t>(rows) * k);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < k / static_cast<int>(ggml_block_elems(q)); ++b) {
            decode_block(q, weights.data() + static_cast<std::size_t>(r) * row_bytes + b * ggml_block_bytes(q),
                         w.data() + static_cast<std::size_t>(r) * k + b * ggml_block_elems(q));
        }
    }
    std::vector<std::uint32_t> expected(static_cast<std::size_t>(rows) * columns);
    for (int t = 0; t < columns; ++t) {
        const std::uint16_t* col = x.data() + static_cast<std::size_t>(t) * k;
        std::vector<std::uint16_t> source(static_cast<std::size_t>(k));
        for (int j = 0; j < k; ++j) { source[static_cast<std::size_t>(j)] = col[perm != nullptr ? (*perm)[j / pb] * pb + j % pb : j]; }
        const int emax = ops::canon::a16_column_exponent(source.data(), k);
        std::vector<std::int32_t> xs(static_cast<std::size_t>(k));
        for (int j = 0; j < k; ++j) { xs[static_cast<std::size_t>(j)] = ops::canon::a16_value(source[static_cast<std::size_t>(j)], emax); }
        for (int r = 0; r < rows; ++r) {
            Big sum;
            for (int j = 0; j < k; ++j) {
                const double v = w[static_cast<std::size_t>(r) * k + j] * 134217728.0; // * 2^27: an integer
                sum.add(static_cast<std::int64_t>(v), xs[static_cast<std::size_t>(j)]);
            }
            const float v = sum.to_float(emax - 147);
            std::uint32_t bits = 0;
            std::memcpy(&bits, &v, 4);
            expected[static_cast<std::size_t>(t) * rows + r] = f32 ? bits : bf16_rne(v);
        }
    }
    return expected;
}

std::vector<std::uint16_t> oracle(QType q, const std::vector<std::uint8_t>& weights, int rows, int k,
                                  const std::vector<std::uint16_t>& x, int columns, const std::vector<std::int32_t>* perm,
                                  int pb) {
    const auto bits = oracle_bits(q, weights, rows, k, x, columns, perm, pb, false);
    std::vector<std::uint16_t> out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) { out[i] = static_cast<std::uint16_t>(bits[i]); }
    return out;
}

void run_linear(QType q, int rows, int k, int columns, bool permute, std::mt19937& rng, bool wide_range = false,
                Scales scales = Scales::Wide, bool extreme = false) {
    const std::string label = "format " + std::to_string(static_cast<int>(q)) + " k=" + std::to_string(k) +
                              " T=" + std::to_string(columns) + (permute ? " perm" : "") + (wide_range ? " wide" : "") +
                              (scales == Scales::Narrow  ? " narrow scales"
                               : scales == Scales::Wider ? " wider scales"
                               : scales == Scales::Edge  ? " edge scales"
                                                         : "") +
                              (extreme ? " extreme" : "");
    const int row_bytes = k / static_cast<int>(ggml_block_elems(q)) * static_cast<int>(ggml_block_bytes(q));
    const auto weights  = extreme ? extreme_q6k(rows, k) : random_blocks(q, rows, k, rng, scales);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * columns);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (auto& v : x) { v = bf16_rne(normal(rng)); }
    if (wide_range) {
        // Magnitudes over 2^-40..2^12 with zeros and both signs: A16 rounds the small ones and the
        // limb split sees every residue class, negative values and the column maximum.
        std::uniform_real_distribution<float> exponent(-40.0F, 12.0F);
        for (std::size_t i = 0; i < x.size(); ++i) {
            const float m = std::ldexp(1.0F, static_cast<int>(exponent(rng))) * (1.0F + normal(rng) * 0.25F);
            x[i] = i % 17 == 0 ? std::uint16_t{0} : bf16_rne(i % 2 == 0 ? m : -m);
        }
    }
    if (extreme) {
        // Every input 1.9921875 (A16 X = 255 * 2^13, the largest), negated in odd columns.
        for (std::size_t i = 0; i < x.size(); ++i) { x[i] = (i / static_cast<std::size_t>(k)) % 2 == 0 ? 0x3FFF : 0xBFFF; }
    }
    const int pb = 128;
    std::vector<std::int32_t> perm(static_cast<std::size_t>(k / pb));
    for (std::size_t i = 0; i < perm.size(); ++i) { perm[i] = static_cast<std::int32_t>(i); }
    if (permute) { std::shuffle(perm.begin(), perm.end(), rng); }
    const auto expected = oracle(q, weights, rows, k, x, columns, permute ? &perm : nullptr, pb);

    // The Op.
    DeviceBuffer dw(weights.size()), dx(x.size() * 2), dout(expected.size() * 2), dperm(perm.size() * 4);
    dw.copy_from_host(weights.data(), weights.size(), 0);
    dx.copy_from_host(x.data(), x.size() * 2, 0);
    dperm.copy_from_host(perm.data(), perm.size() * 4, 0);
    const std::size_t ws = ops::ggml_linear_workspace_bytes(k, columns) + (1 << 20);
    WorkspaceArena work(ws);
    Tensor tx(dx.p, DType::BF16, {k, columns});
    Tensor tout(dout.p, DType::BF16, {rows, columns});
    // Two segments over the same rows' halves, to exercise row offsets.
    const int half = rows / 2;
    const ops::GgmlSegment segments[] = {
        {q, static_cast<const std::uint8_t*>(dw.p), half, 0},
        {q, static_cast<const std::uint8_t*>(dw.p) + static_cast<std::size_t>(half) * row_bytes, rows - half, half}};
    ops::ggml_linear(tx, segments, tout, permute ? static_cast<const std::int32_t*>(dperm.p) : nullptr, pb, work, nullptr, ops::LinearRoutes::Exact);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint16_t> actual(expected.size());
    dout.copy_to_host(actual.data(), actual.size() * 2, 0);
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) { mismatches += actual[i] != expected[i]; }
    check(mismatches == 0, label + ": " + std::to_string(mismatches) + " of " + std::to_string(actual.size()) + " outputs differ");

    // The multi-output form: the rows split among three tensors at bounds inside both segments.
    const std::int32_t bounds[] = {0, half - 3, half + 5, rows};
    std::vector<DeviceBuffer> parts;
    std::vector<Tensor> part_tensors;
    for (int i = 0; i < 3; ++i) {
        const int n = bounds[i + 1] - bounds[i];
        parts.emplace_back(static_cast<std::size_t>(n) * columns * 2);
        part_tensors.push_back(Tensor(parts.back().p, DType::BF16, {n, columns}));
    }
    Tensor* const outs[] = {&part_tensors[0], &part_tensors[1], &part_tensors[2]};
    ops::ggml_linear(tx, ops::ggml_split_segments(segments, k, bounds), outs,
                     permute ? static_cast<const std::int32_t*>(dperm.p) : nullptr, pb, work, nullptr, ops::LinearRoutes::Exact);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::size_t split_mismatches = 0;
    for (int i = 0; i < 3; ++i) {
        const int n = bounds[i + 1] - bounds[i];
        std::vector<std::uint16_t> got(static_cast<std::size_t>(n) * columns);
        parts[static_cast<std::size_t>(i)].copy_to_host(got.data(), got.size() * 2, 0);
        for (int t = 0; t < columns; ++t) {
            for (int r = 0; r < n; ++r) {
                split_mismatches += got[static_cast<std::size_t>(t) * n + r] !=
                                    expected[static_cast<std::size_t>(t) * rows + bounds[i] + r];
            }
        }
    }
    check(split_mismatches == 0, label + " split into three outputs: " + std::to_string(split_mismatches) + " outputs differ");
}

// A ggml_tiles16_v1 matrix of `rows` ggml_rows_v1 rows (include/infernix/ops/ggml_linear.h): tile i
// at 16 i row_bytes, block b of its row r at (16 b + r) * block_bytes, missing rows zero blocks.
std::vector<std::uint8_t> tiles16(QType q, int rows, int k, const std::uint8_t* data) {
    const std::size_t bb = ggml_block_bytes(q), blocks = static_cast<std::size_t>(k) / ggml_block_elems(q), row_bytes = blocks * bb;
    const int tiles = (rows + 15) / 16;
    std::vector<std::uint8_t> out(static_cast<std::size_t>(tiles) * 16 * row_bytes, 0);
    for (int row = 0; row < rows; ++row) {
        for (std::size_t b = 0; b < blocks; ++b) {
            std::memcpy(out.data() + static_cast<std::size_t>(row / 16) * 16 * row_bytes + (16 * b + row % 16) * bb,
                        data + static_cast<std::size_t>(row) * row_bytes + b * bb, bb);
        }
    }
    return out;
}

// ggml_linear over ggml_tiles16_v1 segments against the oracle: segments of one tiled matrix cut at
// a tile boundary (ggml_split_segments) plus a second matrix tiled on its own and placed at an odd
// output row, so partial last tiles, row offsets and both cuts occur; the route the width takes (the
// rows kernel, the MMA tiles in the shape the call's tiles select, or the prefill planes from 32
// columns); FP32 outputs when `f32`.
void run_tiled(QType q, int rows, int k, int columns, bool permute, bool f32, std::mt19937& rng, Scales scales) {
    const std::string label = "tiled format " + std::to_string(static_cast<int>(q)) + " rows=" + std::to_string(rows) +
                              " k=" + std::to_string(k) + " T=" + std::to_string(columns) + (permute ? " perm" : "") +
                              (f32 ? " f32" : "") + (scales == Scales::Narrow ? " narrow" : scales == Scales::Wider ? " wider" : "");
    const std::size_t row_bytes = ops::ggml_row_bytes(q, k);
    const int first = rows / 2 / 16 * 16 + 16; // rows of the first matrix: a whole number of tiles
    const auto weights = random_blocks(q, rows, k, rng, scales);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * columns);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (auto& v : x) { v = bf16_rne(normal(rng)); }
    const int pb = 128;
    std::vector<std::int32_t> perm(static_cast<std::size_t>(k / pb));
    for (std::size_t i = 0; i < perm.size(); ++i) { perm[i] = static_cast<std::int32_t>(i); }
    if (permute) { std::shuffle(perm.begin(), perm.end(), rng); }
    const auto expected = oracle_bits(q, weights, rows, k, x, columns, permute ? &perm : nullptr, pb, f32);
    const auto t0 = tiles16(q, first, k, weights.data());
    const auto t1 = tiles16(q, rows - first, k, weights.data() + static_cast<std::size_t>(first) * row_bytes);
    DeviceBuffer d0(t0.size()), d1(t1.size()), dx(x.size() * 2), dperm(perm.size() * 4);
    d0.copy_from_host(t0.data(), t0.size(), 0);
    d1.copy_from_host(t1.data(), t1.size(), 0);
    dx.copy_from_host(x.data(), x.size() * 2, 0);
    dperm.copy_from_host(perm.data(), perm.size() * 4, 0);
    const std::size_t out_bytes = f32 ? 4 : 2;
    DeviceBuffer dout(static_cast<std::size_t>(rows) * columns * out_bytes);
    WorkspaceArena work(ops::ggml_linear_workspace_bytes(k, columns) + (1 << 20));
    Tensor tx(dx.p, DType::BF16, {k, columns});
    Tensor tout(dout.p, f32 ? DType::FP32 : DType::BF16, {rows, columns});
    // The first matrix cut at 16 rows into two segments of output 0's rows; the second matrix after them.
    const ops::GgmlSegment whole{q, static_cast<const std::uint8_t*>(d0.p), first, 0, 0, ops::GgmlLayout::Tiles16};
    const std::int32_t cut[] = {0, 16, first};
    auto segments = ops::ggml_split_segments(std::span(&whole, 1), k, cut);
    for (auto& s : segments) {
        s.out_offset += s.output == 1 ? 16 : 0;
        s.output = 0;
    }
    segments.push_back({q, static_cast<const std::uint8_t*>(d1.p), rows - first, first, 0, ops::GgmlLayout::Tiles16});
    const auto run_once = [&](const std::string& what) {
        CUDA_CHECK(cudaMemset(dout.p, 0xFF, dout.bytes));
        ops::ggml_linear(tx, segments, tout, permute ? static_cast<const std::int32_t*>(dperm.p) : nullptr, pb, work, nullptr, ops::LinearRoutes::Exact);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint32_t> actual(expected.size());
        if (f32) {
            dout.copy_to_host(actual.data(), actual.size() * 4, 0);
        } else {
            std::vector<std::uint16_t> half(expected.size());
            dout.copy_to_host(half.data(), half.size() * 2, 0);
            for (std::size_t i = 0; i < half.size(); ++i) { actual[i] = half[i]; }
        }
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) { mismatches += actual[i] != expected[i]; }
        check(mismatches == 0, label + " " + what + ": " + std::to_string(mismatches) + " of " + std::to_string(actual.size()) +
                                   " outputs differ");
    };
    run_once(columns >= 32 ? "planes" : "narrow");
}

// A projection whose rows are two formats (rows [0, half) of q0, the rest of q1, as the GGUF imports'
// fused projections), into one output and split among three at bounds inside both parts: the same
// oracle as run_linear. Verification widths take the mixed-format launch, the others one grouped
// launch per format; `tiled`: both parts ggml_tiles16_v1 (one tiled mixed launch at every narrow
// width).
void run_mixed(QType q0, QType q1, int rows, int k, int columns, std::mt19937& rng, bool tiled = false) {
    const std::string label = std::string(tiled ? "tiled " : "") + "mixed formats " + std::to_string(static_cast<int>(q0)) + "+" +
                              std::to_string(static_cast<int>(q1)) + " T=" + std::to_string(columns);
    const int half = rows / 2;
    const auto w0 = random_blocks(q0, half, k, rng, Scales::Narrow), w1 = random_blocks(q1, rows - half, k, rng, Scales::Narrow);
    std::vector<double> w(static_cast<std::size_t>(rows) * k);
    for (int r = 0; r < rows; ++r) {
        const QType q = r < half ? q0 : q1;
        const auto& src = r < half ? w0 : w1;
        const int rr = r < half ? r : r - half;
        const std::size_t row_bytes = static_cast<std::size_t>(k / ggml_block_elems(q)) * ggml_block_bytes(q);
        for (int b = 0; b < k / static_cast<int>(ggml_block_elems(q)); ++b) {
            decode_block(q, src.data() + rr * row_bytes + b * ggml_block_bytes(q), w.data() + static_cast<std::size_t>(r) * k + b * ggml_block_elems(q));
        }
    }
    std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * columns);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (auto& v : x) { v = bf16_rne(normal(rng)); }
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(rows) * columns);
    for (int t = 0; t < columns; ++t) {
        const std::uint16_t* col = x.data() + static_cast<std::size_t>(t) * k;
        const int emax = ops::canon::a16_column_exponent(col, k);
        for (int r = 0; r < rows; ++r) {
            Big sum;
            for (int j = 0; j < k; ++j) {
                sum.add(static_cast<std::int64_t>(w[static_cast<std::size_t>(r) * k + j] * 134217728.0), ops::canon::a16_value(col[j], emax));
            }
            expected[static_cast<std::size_t>(t) * rows + r] = bf16_rne(sum.to_float(emax - 147));
        }
    }
    // Tiled: each part a ggml_tiles16_v1 matrix of its own, split at whole tiles of each part.
    const auto t0 = tiled ? tiles16(q0, half, k, w0.data()) : w0, t1 = tiled ? tiles16(q1, rows - half, k, w1.data()) : w1;
    const ops::GgmlLayout layout = tiled ? ops::GgmlLayout::Tiles16 : ops::GgmlLayout::Rows;
    DeviceBuffer d0(t0.size()), d1(t1.size()), dx(x.size() * 2), dout(expected.size() * 2);
    d0.copy_from_host(t0.data(), t0.size(), 0);
    d1.copy_from_host(t1.data(), t1.size(), 0);
    dx.copy_from_host(x.data(), x.size() * 2, 0);
    WorkspaceArena work(ops::ggml_linear_workspace_bytes(k, columns) + (1 << 20));
    Tensor tx(dx.p, DType::BF16, {k, columns});
    Tensor tout(dout.p, DType::BF16, {rows, columns});
    const ops::GgmlSegment segments[] = {{q0, static_cast<const std::uint8_t*>(d0.p), half, 0, 0, layout},
                                         {q1, static_cast<const std::uint8_t*>(d1.p), rows - half, half, 0, layout}};
    const std::int32_t bounds[] = {0, tiled ? 16 : half - 3, tiled ? half + 16 : half + 5, rows};
    std::vector<DeviceBuffer> parts;
    std::vector<Tensor> part_tensors;
    for (int i = 0; i < 3; ++i) {
        parts.emplace_back(static_cast<std::size_t>(bounds[i + 1] - bounds[i]) * columns * 2);
        part_tensors.push_back(Tensor(parts.back().p, DType::BF16, {bounds[i + 1] - bounds[i], columns}));
    }
    Tensor* const outs[] = {&part_tensors[0], &part_tensors[1], &part_tensors[2]};
    const auto run_once = [&](const std::string& what) {
        CUDA_CHECK(cudaMemset(dout.p, 0xFF, dout.bytes));
        ops::ggml_linear(tx, segments, tout, nullptr, 0, work, nullptr, ops::LinearRoutes::Exact);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint16_t> actual(expected.size());
        dout.copy_to_host(actual.data(), actual.size() * 2, 0);
        std::size_t bad = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) { bad += actual[i] != expected[i]; }
        check(bad == 0, label + what + ": " + std::to_string(bad) + " of " + std::to_string(actual.size()) + " outputs differ");
        ops::ggml_linear(tx, ops::ggml_split_segments(segments, k, bounds), outs, nullptr, 0, work, nullptr, ops::LinearRoutes::Exact);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::size_t split_bad = 0;
        for (int i = 0; i < 3; ++i) {
            const int n = bounds[i + 1] - bounds[i];
            std::vector<std::uint16_t> got(static_cast<std::size_t>(n) * columns);
            parts[static_cast<std::size_t>(i)].copy_to_host(got.data(), got.size() * 2, 0);
            for (int t = 0; t < columns; ++t) {
                for (int r = 0; r < n; ++r) { split_bad += got[static_cast<std::size_t>(t) * n + r] != expected[static_cast<std::size_t>(t) * rows + bounds[i] + r]; }
            }
        }
        check(split_bad == 0, label + what + " split into three outputs: " + std::to_string(split_bad) + " outputs differ");
    };
    run_once("");
}

// ---- the ggml-bf16-prefill route (LinearRoutes::GgmlBf16Wide)
//
// Its arithmetic is not the exact route's: every product is exact (codes exact in BF16, x as stored)
// and only FP32 sums round, so it is checked against a binary64 reference over the exactly decoded
// weights and the BF16 inputs themselves (no A16 encoding): |y - ref| <= (K / 32 + 48) * 2^-24 *
// sum |w x| per output (an FP32 accumulator over K / 32 slices plus one MMA sum each), plus half a
// BF16 ulp of ref for BF16 outputs. Calls narrower than 32 columns must keep the exact route's bits.
// An information line gives both routes' largest and mean error against the reference, in units of
// sum |w x|, and the share of BF16 outputs equal to the exact route's.
void run_bf16(QType q, int rows, int k, int columns, bool permute, bool tiled, bool f32, Scales scales, std::mt19937& rng) {
    const std::string label = "bf16 route format " + std::to_string(static_cast<int>(q)) + " rows=" + std::to_string(rows) +
                              " k=" + std::to_string(k) + " T=" + std::to_string(columns) + (permute ? " perm" : "") +
                              (tiled ? " tiled" : "") + (f32 ? " f32" : "");
    const std::size_t row_bytes = ops::ggml_row_bytes(q, k);
    const auto weights          = random_blocks(q, rows, k, rng, scales);
    // The K-quants' min term on its own (each block decoded with its codes zeroed: -dmin * min): the
    // route sums it apart from the codes' term, so its error scales with both magnitudes.
    const bool min_format = q == QType::GGML_Q4_K || q == QType::GGML_Q5_K;
    std::vector<double> w(static_cast<std::size_t>(rows) * k), minterm(min_format ? w.size() : 0);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < k / static_cast<int>(ggml_block_elems(q)); ++b) {
            const std::uint8_t* block = weights.data() + static_cast<std::size_t>(r) * row_bytes + b * ggml_block_bytes(q);
            decode_block(q, block, w.data() + static_cast<std::size_t>(r) * k + b * ggml_block_elems(q));
            if (min_format) {
                std::vector<std::uint8_t> zeroed(block, block + ggml_block_bytes(q));
                std::fill(zeroed.begin() + 16, zeroed.end(), std::uint8_t{0}); // Q4_K qs, Q5_K qh and qs
                decode_block(q, zeroed.data(), minterm.data() + static_cast<std::size_t>(r) * k + b * ggml_block_elems(q));
            }
        }
    }
    std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * columns);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (std::size_t i = 0; i < x.size(); ++i) { x[i] = bf16_rne(normal(rng) * (i % 97 == 0 ? 64.0F : 1.0F)); }
    const int pb = 128;
    std::vector<std::int32_t> perm(static_cast<std::size_t>(k / pb));
    for (std::size_t i = 0; i < perm.size(); ++i) { perm[i] = static_cast<std::int32_t>(i); }
    if (permute) { std::shuffle(perm.begin(), perm.end(), rng); }
    const auto value = [](std::uint16_t bits) {
        const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return static_cast<double>(f);
    };
    std::vector<double> ref(static_cast<std::size_t>(rows) * columns), mag(ref.size());
    for (int t = 0; t < columns; ++t) {
        const std::uint16_t* col = x.data() + static_cast<std::size_t>(t) * k;
        for (int r = 0; r < rows; ++r) {
            double sum = 0, abs_sum = 0;
            for (int j = 0; j < k; ++j) {
                const double xv = value(col[permute ? perm[j / pb] * pb + j % pb : j]);
                const double p  = w[static_cast<std::size_t>(r) * k + j] * xv;
                sum += p;
                abs_sum += std::fabs(p) + (min_format ? 2.0 * std::fabs(minterm[static_cast<std::size_t>(r) * k + j] * xv) : 0.0);
            }
            ref[static_cast<std::size_t>(t) * rows + r] = sum;
            mag[static_cast<std::size_t>(t) * rows + r] = abs_sum;
        }
    }
    const auto weights_on_device = tiled ? tiles16(q, rows, k, weights.data()) : weights;
    DeviceBuffer dw(weights_on_device.size()), dx(x.size() * 2), dperm(perm.size() * 4);
    dw.copy_from_host(weights_on_device.data(), weights_on_device.size(), 0);
    dx.copy_from_host(x.data(), x.size() * 2, 0);
    dperm.copy_from_host(perm.data(), perm.size() * 4, 0);
    const std::size_t out_bytes = f32 ? 4 : 2;
    DeviceBuffer dout(ref.size() * out_bytes);
    WorkspaceArena work(ops::ggml_linear_workspace_bytes(k, columns) + (1 << 20));
    Tensor tx(dx.p, DType::BF16, {k, columns});
    Tensor tout(dout.p, f32 ? DType::FP32 : DType::BF16, {rows, columns});
    const ops::GgmlLayout layout = tiled ? ops::GgmlLayout::Tiles16 : ops::GgmlLayout::Rows;
    // Rows-layout calls in two segments (a row offset); a tiled matrix as one.
    std::vector<ops::GgmlSegment> segments;
    if (tiled) {
        segments.push_back({q, static_cast<const std::uint8_t*>(dw.p), rows, 0, 0, layout});
    } else {
        const int half = rows / 2;
        segments.push_back({q, static_cast<const std::uint8_t*>(dw.p), half, 0, 0, layout});
        segments.push_back({q, static_cast<const std::uint8_t*>(dw.p) + static_cast<std::size_t>(half) * row_bytes, rows - half, half, 0, layout});
    }
    const auto run = [&](ops::LinearRoutes routes) {
        CUDA_CHECK(cudaMemset(dout.p, 0xFF, dout.bytes));
        ops::ggml_linear(tx, segments, tout, permute ? static_cast<const std::int32_t*>(dperm.p) : nullptr, pb, work, nullptr, routes);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> got(ref.size());
        std::vector<std::uint32_t> bits(ref.size());
        if (f32) {
            std::vector<float> v(ref.size());
            dout.copy_to_host(v.data(), v.size() * 4, 0);
            for (std::size_t i = 0; i < v.size(); ++i) {
                got[i] = v[i];
                std::memcpy(&bits[i], &v[i], 4);
            }
        } else {
            std::vector<std::uint16_t> v(ref.size());
            dout.copy_to_host(v.data(), v.size() * 2, 0);
            for (std::size_t i = 0; i < v.size(); ++i) {
                got[i]  = value(v[i]);
                bits[i] = v[i];
            }
        }
        return std::pair(got, bits);
    };
    const auto [fast, fast_bits]   = run(ops::LinearRoutes::GgmlBf16Wide);
    const auto [exact, exact_bits] = run(ops::LinearRoutes::Exact);
    if (columns < 32) {
        check(fast_bits == exact_bits, label + ": a narrow call must keep the exact route's bits");
        return;
    }
    // Up to three FP32 roundings per slice (two halves' scaled sums, the min term) and the MMAs'
    // sums of 16 products, each within 2^-24 of the magnitudes it adds.
    const double slack = (3 * (k / 32) + 64) * std::ldexp(1.0, -24);
    std::size_t bad = 0, same = 0;
    double worst_fast = 0, worst_exact = 0, mean_fast = 0, mean_exact = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double tol = slack * mag[i] + (f32 ? 0.0 : std::ldexp(std::fabs(ref[i]), -8)) + 1e-30;
        const double e   = std::fabs(fast[i] - ref[i]);
        bad += !(e <= tol); // NaN fails
        same += fast_bits[i] == exact_bits[i];
        const double scale = mag[i] > 0 ? mag[i] : 1.0;
        worst_fast  = std::max(worst_fast, e / scale);
        worst_exact = std::max(worst_exact, std::fabs(exact[i] - ref[i]) / scale);
        mean_fast += e / scale;
        mean_exact += std::fabs(exact[i] - ref[i]) / scale;
    }
    check(bad == 0, label + ": " + std::to_string(bad) + " of " + std::to_string(ref.size()) + " outputs outside the FP32-sum bound");
    std::printf("info %s: error / sum|wx| max %.2e mean %.2e (exact route %.2e / %.2e); %.1f %% of outputs equal the exact route's\n",
                label.c_str(), worst_fast, mean_fast / static_cast<double>(ref.size()), worst_exact,
                mean_exact / static_cast<double>(ref.size()), 100.0 * static_cast<double>(same) / static_cast<double>(ref.size()));
}

// ---- benchmarks (information only)
//
// Calls of the GSQ-RCO IQ3_S import's dense shapes, captured `calls` at a time in a CUDA Graph on a
// non-blocking stream (programmatic dependents as in the model), over weight copies cycling about
// 256 MB (past the L2 cache; the LM head's one copy is 521 MB); microseconds per call, the encoder
// included. `tiled`: the same matrices in ggml_tiles16_v1.
double time_calls(const std::function<void(int, cudaStream_t)>& run, int calls, cudaStream_t stream) {
    constexpr int kLaunches = 10;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try {
        for (int i = 0; i < calls; ++i) { run(i, stream); }
    } catch (...) { // a refused shape: end the capture, so the stream takes the next one
        cudaGraph_t abandoned = nullptr;
        cudaStreamEndCapture(stream, &abandoned);
        if (abandoned != nullptr) { cudaGraphDestroy(abandoned); }
        cudaGetLastError();
        throw;
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    // Warm-up of at least 50 ms: the clocks drop while the host copies a call's weights (the head's
    // 521 MB), and a few launches did not bring them back (its first routes timed up to 2x slow).
    for (float warm = 0; warm < 50.0F;) {
        CUDA_CHECK(cudaEventRecord(a, stream));
        for (int i = 0; i < 3; ++i) { CUDA_CHECK(cudaGraphLaunch(exec, stream)); }
        CUDA_CHECK(cudaEventRecord(b, stream));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        warm += ms;
    }
    CUDA_CHECK(cudaEventRecord(a, stream));
    for (int i = 0; i < kLaunches; ++i) { CUDA_CHECK(cudaGraphLaunch(exec, stream)); }
    CUDA_CHECK(cudaEventRecord(b, stream));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return ms * 1000.0 / (kLaunches * calls);
}

struct BenchPart {
    QType q;
    int rows;
};

struct BenchCall {
    const char* what;
    int k;
    std::vector<BenchPart> parts; // segments of one call, each its own output
    bool f32  = false;            // FP32 outputs (the LM head)
    bool perm = false;            // the GDN output projection's input permutation (blocks of 128)
};

// One layer's fused projections (GDN q, k, v Q6_K and z IQ4_XS; attention q, gate IQ4_XS and k, v
// Q6_K; shared expert gate Q4_K and up Q6_K), the GDN output projection, single matrices of the other
// dense formats, and the LM head.
std::vector<BenchCall> bench_calls() {
    return {{"gdn q/k/v/z", 2560, {{QType::GGML_Q6_K, 2048}, {QType::GGML_Q6_K, 2048}, {QType::GGML_Q6_K, 6144}, {QType::GGML_IQ4_XS, 6144}}},
            {"attn q/gate/k/v", 2560, {{QType::GGML_IQ4_XS, 6144}, {QType::GGML_IQ4_XS, 6144}, {QType::GGML_Q6_K, 512}, {QType::GGML_Q6_K, 512}}},
            {"shexp gate/up", 2560, {{QType::GGML_Q4_K, 640}, {QType::GGML_Q6_K, 640}}},
            {"ssm_out (perm)", 6144, {{QType::GGML_Q6_K, 2560}}, false, true},
            {"attn_qkv Q5_K", 2560, {{QType::GGML_Q5_K, 10240}}},
            {"attn_gate Q4_K", 2560, {{QType::GGML_Q4_K, 6144}}},
            {"lm_head (FP32)", 2560, {{QType::GGML_Q6_K, 248320}}, true}};
}

// A call's device buffers at `columns` columns, in one layout; time() captures and times it under
// the current switches.
class PreparedCall {
public:
    PreparedCall(const BenchCall& call, int columns, bool tiled) : call_(call), columns_(columns), tiled_(tiled) {
        const int k = call.k;
        std::mt19937 rng(2);
        for (const auto& part : call.parts) { bytes_ += ops::ggml_row_bytes(part.q, k) * part.rows; }
        head_   = bytes_ > (256ULL << 20);
        copies_ = head_ ? 1 : static_cast<int>(std::clamp<std::size_t>((256ULL << 20) / bytes_ + 1, 8, 256));
        for (const auto& part : call.parts) {
            auto w = random_blocks(part.q, part.rows, k, rng, Scales::Narrow);
            if (tiled) { w = tiles16(part.q, part.rows, k, w.data()); }
            part_bytes_.push_back(w.size());
            weights_.emplace_back(w.size() * copies_);
            for (int c = 0; c < copies_; ++c) { weights_.back().copy_from_host(w.data(), w.size(), static_cast<std::size_t>(c) * w.size()); }
            outs_.emplace_back(static_cast<std::size_t>(part.rows) * (call.f32 ? 4 : 2) * columns);
        }
        for (std::size_t i = 0; i < call.parts.size(); ++i) {
            out_tensors_.push_back(Tensor(outs_[i].p, call.f32 ? DType::FP32 : DType::BF16, {call.parts[i].rows, columns}));
        }
        for (auto& t : out_tensors_) { out_ptrs_.push_back(&t); }
        x_ = DeviceBuffer(static_cast<std::size_t>(k) * 2 * columns);
        std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * columns);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        for (auto& v : x) { v = bf16_rne(normal(rng)); }
        x_.copy_from_host(x.data(), x.size() * 2, 0);
        std::vector<std::int32_t> perm(static_cast<std::size_t>(k / 128));
        for (std::size_t i = 0; i < perm.size(); ++i) { perm[i] = static_cast<std::int32_t>((i * 7) % perm.size()); }
        perm_ = DeviceBuffer(perm.size() * 4);
        perm_.copy_from_host(perm.data(), perm.size() * 4, 0);
        work_ = std::make_unique<WorkspaceArena>(ops::ggml_linear_workspace_bytes(k, columns) + (1 << 20));
    }

    void time(const std::string& route, ops::LinearRoutes routes, cudaStream_t stream) {
        Tensor tx(x_.p, DType::BF16, {call_.k, columns_});
        const auto run = [&](int i, cudaStream_t s) {
            std::vector<ops::GgmlSegment> segs;
            for (std::size_t j = 0; j < call_.parts.size(); ++j) {
                segs.push_back({call_.parts[j].q,
                                static_cast<const std::uint8_t*>(weights_[j].p) + static_cast<std::size_t>(i % copies_) * part_bytes_[j],
                                call_.parts[j].rows, 0, static_cast<std::int32_t>(j),
                                tiled_ ? ops::GgmlLayout::Tiles16 : ops::GgmlLayout::Rows});
            }
            ops::ggml_linear(tx, segs, out_ptrs_, call_.perm ? static_cast<const std::int32_t*>(perm_.p) : nullptr, 128, *work_, s, routes);
        };
        const int calls = head_ || columns_ >= 32 ? 4 : 50;
        double us       = 0;
        try {
            us = time_calls(run, calls, stream);
        } catch (const std::exception& error) {
            std::printf("bench %-16s T=%-5d %-6s %-16s skipped: %s\n", call_.what, columns_, tiled_ ? "tiled" : "rows", route.c_str(), error.what());
            std::fflush(stdout);
            return;
        }
        std::printf("bench %-16s T=%-5d %-6s %-16s %9.2f us  %7.1f GB/s\n", call_.what, columns_, tiled_ ? "tiled" : "rows",
                    route.c_str(), us, static_cast<double>(bytes_) / us / 1e3);
        std::fflush(stdout);
    }

private:
    BenchCall call_;
    int columns_;
    bool tiled_;
    std::size_t bytes_ = 0;
    bool head_         = false;
    int copies_        = 1;
    std::vector<DeviceBuffer> weights_, outs_;
    std::vector<std::size_t> part_bytes_;
    std::vector<Tensor> out_tensors_;
    std::vector<Tensor*> out_ptrs_;
    DeviceBuffer x_, perm_;
    std::unique_ptr<WorkspaceArena> work_;
};

// --bench N [tiled] [bf16]: every bench call at N columns, rows or tiled, on the exact route and, with
// bf16, also on the ggml-bf16-prefill route (LinearRoutes::GgmlBf16Wide; 32 or more columns).
void bench(int columns, bool tiled, bool bf16) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (const auto& call : bench_calls()) {
        if (call.f32 && columns > 8) { continue; } // the LM head runs at decode widths only
        PreparedCall prepared(call, columns, tiled);
        prepared.time("exact", ops::LinearRoutes::Exact, stream);
        if (bf16) { prepared.time("bf16", ops::LinearRoutes::GgmlBf16Wide, stream); }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// The row gathers' outputs are their elements' values; a zero's sign is not part of them (an
// IQ4_XS sub-scale of 0 gives d x 0 x kvalue, whose sign follows the evaluation order).
bool same_value(std::uint16_t a, std::uint16_t b) {
    return a == b || ((a & 0x7FFFU) == 0 && (b & 0x7FFFU) == 0);
}

// An element's exact value as BF16 bits: the oracle's binary64 value must be exact in binary32 (the
// rows Ops' contract: at most 24 significant bits), then one rounding to BF16.
std::uint16_t exact_bf16(double v, const std::string& label) {
    const auto f = static_cast<float>(v);
    if (static_cast<double>(f) != v) { throw std::logic_error(label + ": an element is not exact in binary32"); }
    return bf16_rne(f);
}

// ggml_embedding: out[:, t] = bf16(row ids[t]) for every id (first, last and repeated rows).
void run_embedding(QType q, int vocab, int hidden, Scales scales, std::mt19937& rng) {
    const std::string label = "embedding format " + std::to_string(static_cast<int>(q)) + " H=" + std::to_string(hidden);
    const int elems = static_cast<int>(ggml_block_elems(q)), size = static_cast<int>(ggml_block_bytes(q));
    const int row_bytes = hidden / elems * size;
    const auto table = random_blocks(q, vocab, hidden, rng, scales);
    std::vector<std::int32_t> ids = {0, vocab - 1, 7, 7};
    for (int i = 0; i < 37; ++i) { ids.push_back(static_cast<std::int32_t>(rng() % vocab)); }
    const int columns = static_cast<int>(ids.size());
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(hidden) * columns);
    std::vector<double> y(static_cast<std::size_t>(elems));
    for (int t = 0; t < columns; ++t) {
        const std::uint8_t* row = table.data() + static_cast<std::size_t>(ids[static_cast<std::size_t>(t)]) * row_bytes;
        for (int b = 0; b < hidden / elems; ++b) {
            decode_block(q, row + static_cast<std::size_t>(b) * size, y.data());
            for (int j = 0; j < elems; ++j) {
                expected[static_cast<std::size_t>(t) * hidden + b * elems + j] = exact_bf16(y[static_cast<std::size_t>(j)], label);
            }
        }
    }
    DeviceBuffer dtable(table.size()), dids(ids.size() * 4), dout(expected.size() * 2);
    dtable.copy_from_host(table.data(), table.size(), 0);
    dids.copy_from_host(ids.data(), ids.size() * 4, 0);
    Weight weight;
    weight.qtype  = q;
    weight.qdata  = dtable.p;
    weight.n      = vocab;
    weight.k      = hidden;
    weight.layout = QuantLayout::GgmlRows;
    Tensor tids(dids.p, DType::I32, {columns});
    Tensor tout(dout.p, DType::BF16, {hidden, columns});
    ops::ggml_embedding(tids, weight, tout, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint16_t> actual(expected.size());
    dout.copy_to_host(actual.data(), actual.size() * 2, 0);
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) { mismatches += !same_value(actual[i], expected[i]); }
    check(mismatches == 0, label + ": " + std::to_string(mismatches) + " of " + std::to_string(actual.size()) + " outputs differ");
}

// ple_embed_iq4_nl: rows U8 [row_bytes, heads, T] to out BF16 [heads * W, T], element by element.
void run_ple(int heads, int width, int columns, Scales scales, std::mt19937& rng) {
    const std::string label = "ple iq4_nl heads=" + std::to_string(heads) + " W=" + std::to_string(width) +
                              " T=" + std::to_string(columns);
    const QType q = QType::GGML_IQ4_NL;
    const int elems = static_cast<int>(ggml_block_elems(q)), size = static_cast<int>(ggml_block_bytes(q));
    const int row_bytes = width / elems * size;
    const auto rows = random_blocks(q, heads * columns, width, rng, scales);
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(heads) * width * columns);
    std::vector<double> y(static_cast<std::size_t>(elems));
    for (int r = 0; r < heads * columns; ++r) { // row r = t * heads + h
        for (int b = 0; b < width / elems; ++b) {
            decode_block(q, rows.data() + static_cast<std::size_t>(r) * row_bytes + static_cast<std::size_t>(b) * size, y.data());
            for (int j = 0; j < elems; ++j) {
                expected[static_cast<std::size_t>(r) * width + b * elems + j] = exact_bf16(y[static_cast<std::size_t>(j)], label);
            }
        }
    }
    DeviceBuffer drows(rows.size()), dout(expected.size() * 2);
    drows.copy_from_host(rows.data(), rows.size(), 0);
    Tensor trows(drows.p, DType::U8, {row_bytes, heads, columns});
    Tensor tout(dout.p, DType::BF16, {heads * width, columns});
    ops::ple_embed_iq4_nl(trows, tout, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint16_t> actual(expected.size());
    dout.copy_to_host(actual.data(), actual.size() * 2, 0);
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) { mismatches += !same_value(actual[i], expected[i]); }
    check(mismatches == 0, label + ": " + std::to_string(mismatches) + " of " + std::to_string(actual.size()) + " outputs differ");
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        bool tiled = false, bf16 = false;
        for (int i = 3; i < argc; ++i) {
            tiled = tiled || std::string(argv[i]) == "tiled";
            bf16  = bf16 || std::string(argv[i]) == "bf16";
        }
        bench(argc > 2 ? std::atoi(argv[2]) : 1, tiled, bf16);
        return 0;
    }
    try {
        std::mt19937 rng(20261010);
        const QType formats[] = {QType::GGML_Q2_0, QType::GGML_Q4_K, QType::GGML_Q5_K, QType::GGML_Q6_K, QType::GGML_IQ2_S,
                                 QType::GGML_IQ3_XXS, QType::GGML_IQ3_S, QType::GGML_IQ4_XS, QType::GGML_IQ4_NL};
        for (const QType q : formats) {
            const int k = ggml_block_elems(q) == 256 ? 2560 : 640;
            for (const int columns : {1, 3, 8, 9}) { run_linear(q, 34, k, columns, false, rng); }
            run_linear(q, 18, k, 600, false, rng); // two encode batches
            run_linear(q, 18, k == 2560 ? 6144 : 2560, 2, true, rng);
            run_linear(q, 21, k, 5, false, rng, true);
            run_linear(q, 21, k, 1, false, rng, true);
            run_linear(q, 17, 256, 17, false, rng); // one block per row, five column passes, a partial last
            run_linear(q, 130, k, 40, false, rng);      // tensor-core tiles: three row tiles, a partial column tile
            run_linear(q, 67, k, 33, false, rng, true);
            run_linear(q, 70, k == 2560 ? 6144 : 2560, 40, true, rng); // tiles over a permuted input
            // Spans past the per-limb bound (the tiles' int128 route) and at its limit.
            run_linear(q, 130, k, 40, false, rng, false, Scales::Wider);
            run_linear(q, 130, k, 40, false, rng, false, Scales::Edge);
            // The model's narrow rows (the K-quants' single int64 accumulators).
            run_linear(q, 130, k, 40, false, rng, false, Scales::Narrow);
            run_linear(q, 67, k, 33, false, rng, true, Scales::Narrow);
            run_linear(q, 70, k == 2560 ? 6144 : 2560, 40, true, rng, false, Scales::Narrow);
            // Last column tiles of 24 and 8 columns (the tiles' 16 x 16 and 16 x 8 warp layouts).
            run_linear(q, 130, k, 88, false, rng, false, Scales::Narrow);
            run_linear(q, 67, k, 72, false, rng, true);
        }
        run_linear(QType::GGML_Q6_K, 64, 6144, 40, false, rng, false, Scales::Narrow, true); // at the int64 bound
        // Projections of two formats (the mixed verification launch, and per-format launches otherwise).
        const std::pair<QType, QType> mixed[] = {{QType::GGML_Q6_K, QType::GGML_IQ4_XS}, {QType::GGML_IQ4_XS, QType::GGML_Q6_K},
                                                 {QType::GGML_Q4_K, QType::GGML_Q6_K},   {QType::GGML_Q5_K, QType::GGML_IQ4_XS},
                                                 {QType::GGML_Q4_K, QType::GGML_Q5_K},   {QType::GGML_IQ4_XS, QType::GGML_Q4_K}};
        for (const auto& [q0, q1] : mixed) {
            for (const int columns : {1, 3, 5, 8}) { run_mixed(q0, q1, 66, 2560, columns, rng); }
            for (const int columns : {1, 2, 4, 8}) { run_mixed(q0, q1, 66, 2560, columns, rng, true); }
        }
        // Tiled mixed launches of 800 and 250 tiles (one-block rows): the unsplit and 4-way shapes.
        run_mixed(QType::GGML_Q6_K, QType::GGML_IQ4_XS, 12800, 256, 2, rng, true);
        run_mixed(QType::GGML_Q4_K, QType::GGML_Q6_K, 4000, 256, 5, rng, true);
        // ggml_tiles16_v1: every tiled format and route (the rows kernel at one column and past eight,
        // the MMA tiles between, the prefill planes from 32 columns), partial tiles, one-block rows,
        // the input permutation, FP32 outputs and the scale spans of every accumulator; one-block rows
        // of 200 to 3,000 tiles reach every CTA shape of the occupancy rule (on an RTX 5090: 4 warps
        // split 4, 2 split 2, 4 unsplit, 2 unsplit, and the many-wave 4 unsplit).
        for (const QType q : {QType::GGML_Q4_K, QType::GGML_Q5_K, QType::GGML_Q6_K, QType::GGML_IQ4_XS}) {
            for (const int columns : {1, 2, 3, 4, 5, 8, 9, 17}) { run_tiled(q, 45, 2560, columns, false, false, rng, Scales::Narrow); }
            for (const int tiles : {200, 400, 800, 2200, 3000}) { run_tiled(q, 16 * tiles - 5, 256, 2, false, false, rng, Scales::Wide); }
            run_tiled(q, 37, 6144, 5, true, false, rng, Scales::Wide);
            run_tiled(q, 37, 6144, 1, true, true, rng, Scales::Wide);
            run_tiled(q, 20, 256, 2, false, false, rng, Scales::Wide); // one block per row: idle split warps
            run_tiled(q, 61, 2560, 8, false, true, rng, Scales::Wider);
            run_tiled(q, 130, 2560, 40, false, false, rng, Scales::Narrow); // planes: three row tiles
            run_tiled(q, 70, 6144, 33, true, false, rng, Scales::Wider);
            run_tiled(q, 67, 2560, 72, false, true, rng, Scales::Edge);
        }
        // The ggml-bf16-prefill route within its FP32-sum bound: every format, partial row tiles (two
        // segments of 100 or 65 rows; 45 tiled rows) and column tiles, the input permutation, FP32
        // outputs, every scale span; a call narrower than 32 columns keeps the exact route's bits.
        for (const QType q : formats) {
            const int k = ggml_block_elems(q) == 256 ? 2560 : 640;
            run_bf16(q, 200, k, 40, false, false, false, Scales::Narrow, rng);
            run_bf16(q, 130, k, 300, false, false, true, Scales::Wide, rng);
            run_bf16(q, 70, k == 2560 ? 6144 : 2560, 129, true, false, false, Scales::Wider, rng);
            run_bf16(q, 64, k, 8, false, false, false, Scales::Narrow, rng);
        }
        for (const QType q : {QType::GGML_Q4_K, QType::GGML_Q5_K, QType::GGML_Q6_K, QType::GGML_IQ4_XS}) {
            run_bf16(q, 300, 2560, 200, false, true, false, Scales::Narrow, rng);
            run_bf16(q, 45, 6144, 33, true, true, true, Scales::Edge, rng);
        }
        // The row gathers: the IQ4_XS token embedding (the model's H = 2560 and a one-block row) and
        // the PLE n-gram rows (the model's 16 heads of W = 160 per token, and one-block rows), over every
        // scale span.
        for (const Scales scales : {Scales::Narrow, Scales::Wide, Scales::Wider, Scales::Edge}) {
            run_embedding(QType::GGML_IQ4_XS, 300, 2560, scales, rng);
            run_embedding(QType::GGML_IQ4_XS, 50, 256, scales, rng);
            run_ple(16, 160, 1, scales, rng);
            run_ple(16, 160, 37, scales, rng);
            run_ple(3, 32, 300, scales, rng);
        }
        if (failures != 0) {
            std::cerr << failures << " ggml_linear checks failed\n";
            return 1;
        }
        std::cout << "ggml_linear: every format, K, width and output split bit-equal to the exact oracle, the "
                     "ggml-bf16-prefill route within its FP32-sum bound; ggml_embedding and ple_embed_iq4_nl element "
                     "for element\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
