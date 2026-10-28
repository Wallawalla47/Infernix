// The 8-bit activation route of the CPU route for GGUF (ggml) expert banks (src/ops/offloaded_sparse_moe/cpu/
// ggml_expert.h's a8 functions; `--a8-activations`, LOSSY, opt-in; docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.31). Host-only.
//
// Records are random ggml blocks (finite binary16 scales) of every expert format pair at Flash-Next's geometry
// (H 2560, I 640). The checks:
// - codec: ggml_a8_encode_column against this file's transcription of ggml_a8.h's formula (codes, scale bits
//   and code sum), including an all-zero slice, slices whose largest magnitude is below, at and just above
//   2^-120 (subnormal scales), Inf and NaN slices, and codes of +127 and -127;
// - order: every gate/up (h) and down (y) output against this file's transcription of the GPU kernels' binary32
//   order (cuda/ggml_expert_a8.cuh), bit for bit, with each slice's isum recovered from ggml_block_oracle.h's
//   independent decode, so a CPU-served miss returns the GPU's bits; every ISA this host runs equals the scalar
//   route at every width 1-8, called with the worker team's 16-intermediate units and 64-row items, and
//   ggml_a8_expert_forward composes them;
// - accuracy: each row of that order (binary32, before BF16) against the FP64 value of the same quantized
//   column and decoded weights within the bound below, and the scalar route's BF16 down outputs directly
//   within it plus the BF16 rounding;
// - the worker team's 8-bit rounds against ggml_a8_expert_forward.
//
// The bound. A row's exact value is E = kappa * sum_s T_s, T_s = isum_s * xd_s * wd_s. Its binary32 order
// rounds each slice's product fl32(isum * xd) once and each fma once; a partial takes n = K / (32 P) = 5
// slices, so its first term sees at most n + 1 roundings, and the butterfly adds log2 P more; kappa is an
// exact power of two. With u = 2^-24 and N = n + 1 + log2 P (10 gate/up, 8 down),
//   |y32 - E| <= gamma_N * kappa * sum_s |T_s| + U,    gamma_N = N u / (1 - N u),
// where U = 2^-149 * (kappa * (sum_s |wd_s| + K / 32 + P) + 1) covers roundings in the subnormal range (each at
// most 2^-150 absolute; a product's then scaled by its wd, and kappa's two steps). The FP64 oracle forms each
// S_s = kappa * wd_s * isum_s exactly, rounds S_s * xd_s and sums K / 32 terms in order, so its own error is at
// most gamma64_(K/32 + 1) * kappa * sum |T_s| (u = 2^-53), which the check adds. A BF16 output adds
// |bf16(y32) - y32| <= 2^-8 |y32| + 2^-134.

#include "ops/op_family.h"
#include "core/ggml_format.h"
#include "ggml_block_oracle.h"
#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_a8.h"
#include "ops/offloaded_sparse_moe/cpu/expert_team.h"
#include "ops/offloaded_sparse_moe/cpu/ggml_expert.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace moe    = infernix::ops::offloaded_moe;
namespace a8     = infernix::ggml::a8;
namespace canon  = infernix::ops::canon;
namespace oracle = infernix::test::ggml_oracle;
using infernix::QType;

namespace {

int g_failures = 0;
std::string g_only_isa; // --isa: only this ISA's kernels (an emulated run)

constexpr int H = moe::kHidden, I = moe::kIntermediate;
constexpr int kXSlices = H / 32, kHSlices = I / 32;
constexpr int kCols    = moe::kMaxColumns;

void fail(const std::string& message) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    ++g_failures;
}

std::string pair_name(QType gu, QType down) {
    return "pair (" + std::to_string(static_cast<int>(gu)) + ", " + std::to_string(static_cast<int>(down)) + ")";
}

std::uint32_t row_bytes(QType q, int k) { return k / infernix::ggml_block_elems(q) * infernix::ggml_block_bytes(q); }

void random_rows(QType q, int rows, int k, std::mt19937& rng, std::uint8_t* out) {
    const int blocks = k / static_cast<int>(infernix::ggml_block_elems(q)), size = static_cast<int>(infernix::ggml_block_bytes(q));
    for (int i = 0; i < rows * blocks * size; ++i) { out[i] = static_cast<std::uint8_t>(rng()); }
    for (int i = 0; i < rows * blocks; ++i) {
        std::uint8_t* b = out + static_cast<std::size_t>(i) * size;
        // Small finite nonzero scales (exponent field 3..12) keep the outputs in BF16's normal range.
        const auto w = static_cast<std::uint16_t>((rng() & 0x83FF) | ((3 + rng() % 10) << 10));
        b[0] = static_cast<std::uint8_t>(w & 0xFF);
        b[1] = static_cast<std::uint8_t>(w >> 8);
    }
}

std::uint16_t bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

float bf16_value(std::uint16_t b) {
    const std::uint32_t u = std::uint32_t{b} << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

std::uint32_t f32_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// ------------------------------------------------------- the codec, transcribed from ggml_a8.h

struct Q8 {
    std::int32_t q[32];
    float d;
    std::int32_t sum;
};

// d = fl32(m / 127), q = rne(fl32(v * fl32(127 / m))) clamped to [-127, 127]; m below 2^-120 (BF16 0x0380): d = 0
// and q = 0; m Inf or NaN: d = NaN and q = 0.
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
        const float p = bf16_value(v[j]) * inv;
        const int q   = std::clamp(static_cast<int>(std::nearbyint(p)), -127, 127);
        o.q[j]        = q;
        o.sum += q;
    }
    return o;
}

std::vector<Q8> quantize_column(const std::uint16_t* v, int n) {
    std::vector<Q8> out(static_cast<std::size_t>(n / 32));
    for (int s = 0; s < n / 32; ++s) { out[static_cast<std::size_t>(s)] = quantize(v + 32 * s); }
    return out;
}

bool same_slice(const moe::CpuA8Slice& a, const Q8& b) {
    for (int j = 0; j < 32; ++j) {
        if (a.q[j] != b.q[j]) { return false; }
    }
    if (a.sum != b.sum) { return false; }
    return std::isnan(a.d) ? std::isnan(b.d) : f32_bits(a.d) == f32_bits(b.d);
}

// The route's codec against the transcription over a column of n values.
void check_encode(const std::uint16_t* v, int n, const std::string& what) {
    std::vector<moe::CpuA8Slice> got(static_cast<std::size_t>(n / 32));
    moe::ggml_a8_encode_column(v, n, got.data());
    const auto want = quantize_column(v, n);
    int bad = 0;
    for (int s = 0; s < n / 32; ++s) { bad += !same_slice(got[static_cast<std::size_t>(s)], want[static_cast<std::size_t>(s)]); }
    if (bad != 0) { fail(what + ": " + std::to_string(bad) + " slices differ from the codec's formula"); }
}

// The codec's edge cases, one 32-value slice each at v + 32 k: 0 zeros of both signs; 1 every magnitude below
// 2^-120 (BF16 subnormals included); 2 largest magnitude exactly 2^-120, at a negative element (a subnormal
// scale); 3 +m and -m (codes +127 and -127); 4 largest magnitude 1.5 * 2^-120 (a subnormal scale);
// then, with `nonfinite`, 5 an infinity, 6 NaNs, 7 a negative infinity (else normal values).
constexpr int kEdgeSlices = 8;

void edge_slices(std::uint16_t* v, std::mt19937& rng, bool nonfinite) {
    std::uniform_real_distribution<float> uniform(-2.0F, 2.0F);
    const auto sign = [](int j) { return static_cast<std::uint16_t>(j % 3 == 0 ? 0x8000U : 0U); };
    for (int j = 0; j < 32; ++j) {
        v[j]       = static_cast<std::uint16_t>(j % 2 ? 0x8000U : 0U);
        v[32 + j]  = static_cast<std::uint16_t>((rng() % 0x0380U) | sign(j));
        v[64 + j]  = static_cast<std::uint16_t>((0x0300U + rng() % 0x80U) | sign(j));
        v[96 + j]  = bf16(uniform(rng));
        v[128 + j] = static_cast<std::uint16_t>((0x0380U + rng() % 0x40U) | sign(j)); // below 1.5 * 2^-120
        for (int k = 5; k < kEdgeSlices; ++k) { v[32 * k + j] = bf16(uniform(rng)); }
    }
    v[32 + 4]  = 0x0001; // the smallest BF16 subnormal
    v[32 + 9]  = 0x037F; // the largest magnitude below 2^-120
    v[64 + 5]  = 0x8380; // -2^-120
    v[96 + 0]  = 0x4040; // +3
    v[96 + 7]  = 0xC040; // -3
    v[128 + 2] = 0x03C0; // 1.5 * 2^-120
    if (nonfinite) {
        v[160 + 9]  = 0x7F80; // +inf
        v[192 + 3]  = 0x7FC1; // NaNs of both signs
        v[192 + 30] = 0xFFC0;
        v[224 + 31] = 0xFF80; // -inf
    }
}

void check_codec(std::mt19937& rng) {
    std::vector<std::uint16_t> v(32 * kEdgeSlices);
    edge_slices(v.data(), rng, true);
    std::vector<moe::CpuA8Slice> got(kEdgeSlices);
    moe::ggml_a8_encode_column(v.data(), 32 * kEdgeSlices, got.data());
    const auto zero_codes = [](const moe::CpuA8Slice& s) {
        for (const auto q : s.q) {
            if (q != 0) { return false; }
        }
        return s.sum == 0;
    };
    if (f32_bits(got[0].d) != 0 || !zero_codes(got[0])) { fail("codec: an all-zero slice must have scale +0 and codes 0"); }
    if (f32_bits(got[1].d) != 0 || !zero_codes(got[1])) { fail("codec: a slice below 2^-120 must have scale +0 and codes 0"); }
    const float d120 = std::ldexp(1.0F, -120) / 127.0F;
    if (f32_bits(got[2].d) != f32_bits(d120) || !(got[2].d > 0.0F && got[2].d < FLT_MIN) || got[2].q[5] != -127) {
        fail("codec: a slice whose largest magnitude is 2^-120 must have the subnormal scale fl32(2^-120 / 127) and code -127 there");
    }
    if (got[3].q[0] != 127 || got[3].q[7] != -127) { fail("codec: +m and -m must have codes +127 and -127"); }
    if (!(got[4].d > 0.0F && got[4].d < FLT_MIN) || got[4].q[2] != 127) {
        fail("codec: a slice whose largest magnitude is 1.5 * 2^-120 must have a subnormal scale and code 127 there");
    }
    for (int s = 5; s < kEdgeSlices; ++s) {
        if (!std::isnan(got[static_cast<std::size_t>(s)].d) || !zero_codes(got[static_cast<std::size_t>(s)])) {
            fail("codec: a slice holding Inf or NaN must have a NaN scale and codes 0 (slice " + std::to_string(s) + ")");
        }
    }
    check_encode(v.data(), 32 * kEdgeSlices, "codec edge slices");
    // Random columns: normal values, then values over a wide exponent range.
    std::vector<std::uint16_t> r(H);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_real_distribution<float> exponent(-130.0F, 120.0F);
    for (auto& e : r) { e = bf16(normal(rng)); }
    check_encode(r.data(), H, "codec, normal column");
    for (int j = 0; j < H; ++j) {
        r[static_cast<std::size_t>(j)] = bf16(std::ldexp(1.0F + normal(rng) * 0.1F, static_cast<int>(exponent(rng))) * (j % 2 ? -1.0F : 1.0F));
    }
    check_encode(r.data(), H, "codec, wide-range column");
}

// --------------------------------------------- the GPU kernels' order on the independent decode

double kappa(QType q) { return q == QType::GGML_IQ2_S ? 0.125 : q == QType::GGML_IQ3_XXS ? 0.25 : 1.0; }
int kappa_log2(QType q) { return q == QType::GGML_IQ2_S ? -3 : q == QType::GGML_IQ3_XXS ? -2 : 0; }

struct RowResult {
    float y32    = 0.0F; // the GPU order's binary32 output (output_f32 of p_0)
    double exact = 0.0;  // the FP64 oracle's value
    double bound = 0.0;  // the bound of |y32 - exact|
};

// Rows [0, rows) of a matrix of `format` (K elements per row, `partials` partials) against quantized columns
// xq[c]: out[c][row]. Each slice's isum is S_s / (kappa * wd_s), S_s the exact FP64 dot of the decoded weights
// with the slice's codes; the terms then follow ggml_expert_a8.cuh: m = fl32(isum * xd), fma(m, wd, p) into
// partial s % P from +0 in ascending s, the butterfly, output_f32. The FP64 sum of S_s * xd_s and the bound
// (header) come with each row.
void oracle_rows(QType format, const std::uint8_t* matrix, std::uint32_t rb, int rows, int k, int partials,
                 const std::vector<std::vector<Q8>>& xq, std::vector<std::vector<RowResult>>& out,
                 const std::string& what) {
    const int elems = static_cast<int>(infernix::ggml_block_elems(format));
    const int bytes = static_cast<int>(infernix::ggml_block_bytes(format));
    const int slices = k / 32, n = slices / partials;
    const int depth  = static_cast<int>(std::lround(std::log2(partials)));
    const double u32 = std::ldexp(1.0, -24), u64 = std::ldexp(1.0, -53);
    const double gamma32 = (n + 1 + depth) * u32 / (1.0 - (n + 1 + depth) * u32);
    const double gamma64 = (slices + 1) * u64 / (1.0 - (slices + 1) * u64);
    const double kap = kappa(format), tiny = std::ldexp(1.0, -149);
    const std::size_t cols = xq.size();
    out.assign(cols, std::vector<RowResult>(static_cast<std::size_t>(rows)));
    std::vector<double> w(static_cast<std::size_t>(k));
    std::vector<float> part(static_cast<std::size_t>(partials));
    int bad_isum = 0;
    for (int row = 0; row < rows; ++row) {
        const std::uint8_t* r = matrix + static_cast<std::size_t>(row) * rb;
        for (int b = 0; b < k / elems; ++b) { oracle::decode_block(format, r + static_cast<std::size_t>(b) * bytes, w.data() + static_cast<std::size_t>(b) * elems); }
        for (std::size_t c = 0; c < cols; ++c) {
            std::fill(part.begin(), part.end(), 0.0F);
            double exact = 0.0, sum_abs = 0.0, sum_wd = 0.0;
            for (int s = 0; s < slices; ++s) {
                const Q8& x     = xq[c][static_cast<std::size_t>(s)];
                const float wd  = oracle::f16(oracle::u16(r + static_cast<std::size_t>(32 * s / elems) * bytes));
                double dot      = 0.0; // exact: multiples of one unit below 2^53 of them
                for (int e = 0; e < 32; ++e) { dot += w[static_cast<std::size_t>(32 * s + e)] * x.q[e]; }
                const double isum = dot / (kap * wd);
                if (isum != std::trunc(isum) || std::fabs(isum) >= 16777216.0) { ++bad_isum; }
                const float m = static_cast<float>(isum) * x.d;
                float& p      = part[static_cast<std::size_t>(s % partials)];
                p             = std::fmaf(m, wd, p);
                const double t = dot * static_cast<double>(x.d);
                exact += t;
                sum_abs += std::fabs(t);
                sum_wd += std::fabs(static_cast<double>(wd));
            }
            for (int o = partials / 2; o >= 1; o /= 2) {
                for (int i = 0; i < o; ++i) { part[static_cast<std::size_t>(i)] = part[static_cast<std::size_t>(i)] + part[static_cast<std::size_t>(i + o)]; }
            }
            RowResult& res = out[c][static_cast<std::size_t>(row)];
            res.y32        = a8::output_f32(part[0], kappa_log2(format));
            res.exact      = exact;
            res.bound = (gamma32 + gamma64) * sum_abs * (1.0 + 1e-12) + tiny * (kap * (sum_wd + slices + partials) + 1.0);
        }
    }
    if (bad_isum != 0) { fail(what + ": " + std::to_string(bad_isum) + " slice dots are not integers of the decode"); }
}

// Whether the column has a slice the codec maps to a NaN scale (its outputs are then NaN).
bool nonfinite(const std::vector<Q8>& xq) {
    for (const auto& s : xq) {
        if (std::isnan(s.d)) { return true; }
    }
    return false;
}

// The binary32 rows of finite columns against the FP64 oracle; returns the worst |y32 - exact| / bound.
double check_bound(const std::vector<std::vector<RowResult>>& rows, const std::vector<std::vector<Q8>>& xq,
                   const std::string& what) {
    double worst = 0.0;
    std::size_t bad = 0;
    for (std::size_t c = 0; c < rows.size(); ++c) {
        if (nonfinite(xq[c])) { continue; }
        for (const RowResult& r : rows[c]) {
            const double err = std::fabs(static_cast<double>(r.y32) - r.exact);
            if (!(err <= r.bound)) { ++bad; }
            if (r.bound > 0.0) { worst = std::max(worst, err / r.bound); }
        }
    }
    if (bad != 0) { fail(what + ": " + std::to_string(bad) + " rows of the binary32 order exceed the FP32 bound"); }
    return worst;
}

// ---------------------------------------------------------------------------- one format pair

struct Widths {
    std::vector<std::uint16_t> h, y, forward; // [col][I], [col][H], [col][H] (forward: widths 1, 3, 8)
};

bool forward_width(int ncols) { return ncols == 1 || ncols == 3 || ncols == kCols; }

std::size_t count_diff(const std::uint16_t* a, const std::uint16_t* b, std::size_t n) {
    std::size_t bad = 0;
    for (std::size_t i = 0; i < n; ++i) { bad += a[i] != b[i]; }
    return bad;
}

void run(QType gu, QType down, std::mt19937& rng) {
    const std::string name = pair_name(gu, down);
    const moe::GgmlExpertFormat f{.gate_up = gu, .down = down, .gate_row_bytes = row_bytes(gu, H), .down_row_bytes = row_bytes(down, I)};
    std::vector<std::uint8_t> record(2ULL * I * f.gate_row_bytes + static_cast<std::size_t>(H) * f.down_row_bytes);
    random_rows(gu, 2 * I, H, rng, record.data());
    random_rows(down, H, I, rng, record.data() + 2ULL * I * f.gate_row_bytes);
    const std::uint8_t* gate_rows = record.data();
    const std::uint8_t* up_rows   = record.data() + static_cast<std::size_t>(I) * f.gate_row_bytes;
    const std::uint8_t* down_rows = record.data() + 2ULL * I * f.gate_row_bytes;

    // Eight columns: normal values (0-4), a wide dynamic range with zeros (5), the codec's finite edge slices
    // and then normal values (6), and one with an infinity and NaNs (7: NaN outputs).
    std::vector<std::uint16_t> x(static_cast<std::size_t>(H) * kCols);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_real_distribution<float> exponent(-40.0F, 12.0F);
    for (int c = 0; c < kCols; ++c) {
        for (int j = 0; j < H; ++j) {
            float v = normal(rng);
            if (c == 5) { v = j % 17 == 0 ? 0.0F : std::ldexp(1.0F, static_cast<int>(exponent(rng))) * (j % 2 ? -1.0F : 1.0F); }
            x[static_cast<std::size_t>(c) * H + j] = bf16(v);
        }
    }
    edge_slices(x.data() + static_cast<std::size_t>(6) * H, rng, false);
    x[static_cast<std::size_t>(7) * H + 123]  = 0x7F80;
    x[static_cast<std::size_t>(7) * H + 1000] = 0x7FC1;
    for (int c = 0; c < kCols; ++c) { check_encode(x.data() + static_cast<std::size_t>(c) * H, H, name + " x column " + std::to_string(c)); }

    // The GPU order: gate and up rows, h = SwiGLU of their BF16 outputs, then the down rows on that h.
    std::vector<std::vector<Q8>> xq(kCols), hq(kCols);
    for (int c = 0; c < kCols; ++c) { xq[static_cast<std::size_t>(c)] = quantize_column(x.data() + static_cast<std::size_t>(c) * H, H); }
    std::vector<std::vector<RowResult>> g, u, yr;
    oracle_rows(gu, gate_rows, f.gate_row_bytes, I, H, a8::kGateUpPartials, xq, g, name + " gate");
    oracle_rows(gu, up_rows, f.gate_row_bytes, I, H, a8::kGateUpPartials, xq, u, name + " up");
    std::vector<std::vector<std::uint16_t>> h_want(kCols, std::vector<std::uint16_t>(I)), y_want(kCols, std::vector<std::uint16_t>(H));
    for (int c = 0; c < kCols; ++c) {
        for (int i = 0; i < I; ++i) {
            h_want[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)] =
                canon::swiglu_bf16(canon::f32_to_bf16_rn(g[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)].y32),
                                   canon::f32_to_bf16_rn(u[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)].y32));
        }
        hq[static_cast<std::size_t>(c)] = quantize_column(h_want[static_cast<std::size_t>(c)].data(), I);
        check_encode(h_want[static_cast<std::size_t>(c)].data(), I, name + " h column " + std::to_string(c));
    }
    oracle_rows(down, down_rows, f.down_row_bytes, H, I, a8::kDownPartials, hq, yr, name + " down");
    for (int c = 0; c < kCols; ++c) {
        for (int r = 0; r < H; ++r) {
            y_want[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)] = canon::f32_to_bf16_rn(yr[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)].y32);
        }
    }
    for (std::uint16_t v : y_want[7]) {
        if (v != 0x7FC0) {
            fail(name + ": the non-finite column's outputs must be the canonical NaN");
            break;
        }
    }
    const double worst_gate = std::max(check_bound(g, xq, name + " gate"), check_bound(u, xq, name + " up"));
    const double worst_down = check_bound(yr, hq, name + " down");

    // Every ISA at every width: the GPU order's bits, and the scalar route's.
    std::vector<Widths> scalar(kCols + 1);
    bool have_scalar = false;
    const moe::CpuIsa isas[] = {moe::CpuIsa::kScalar, moe::CpuIsa::kAvx2, moe::CpuIsa::kAvxVnni, moe::CpuIsa::kAvx512Vnni};
    for (const auto isa : isas) {
        if (!moe::cpu_isa_supported(isa) || (!g_only_isa.empty() && g_only_isa != moe::cpu_isa_name(isa))) { continue; }
        const std::string where = name + ", " + moe::cpu_isa_name(isa);
        for (int ncols = 1; ncols <= kCols; ++ncols) {
            // Columns are taken from the end for the narrow widths, so the special columns are covered.
            const int first = kCols - ncols;
            std::vector<moe::CpuA8Slice> xs(static_cast<std::size_t>(ncols) * kXSlices), hs(static_cast<std::size_t>(ncols) * kHSlices);
            Widths got;
            got.h.assign(static_cast<std::size_t>(ncols) * I, 0xFFFF);
            got.y.assign(static_cast<std::size_t>(ncols) * H, 0xFFFF);
            const moe::CpuA8Slice* xp[kCols];
            const moe::CpuA8Slice* hp[kCols];
            const std::uint16_t* xv[kCols];
            std::uint16_t* hv[kCols];
            std::uint16_t* yv[kCols];
            for (int c = 0; c < ncols; ++c) {
                xv[c] = x.data() + static_cast<std::size_t>(first + c) * H;
                moe::ggml_a8_encode_column(xv[c], H, &xs[static_cast<std::size_t>(c) * kXSlices]);
                // The down rows read the GPU order's h, so they are checked apart from gate/up.
                moe::ggml_a8_encode_column(h_want[static_cast<std::size_t>(first + c)].data(), I, &hs[static_cast<std::size_t>(c) * kHSlices]);
                xp[c] = &xs[static_cast<std::size_t>(c) * kXSlices];
                hp[c] = &hs[static_cast<std::size_t>(c) * kHSlices];
                hv[c] = got.h.data() + static_cast<std::size_t>(c) * I;
                yv[c] = got.y.data() + static_cast<std::size_t>(c) * H;
            }
            for (int i = 0; i < I; i += 16) { moe::ggml_a8_gate_up(isa, record.data(), f, xp, ncols, i, i + 16, hv); }
            for (int r = 0; r < H; r += 64) { moe::ggml_a8_down(isa, record.data(), f, hp, ncols, r, r + 64, yv); }
            if (forward_width(ncols)) {
                got.forward.assign(static_cast<std::size_t>(ncols) * H, 0xFFFF);
                std::uint16_t* fv[kCols];
                for (int c = 0; c < ncols; ++c) { fv[c] = got.forward.data() + static_cast<std::size_t>(c) * H; }
                moe::ggml_a8_expert_forward(isa, record.data(), f, ncols, xv, fv);
            }
            std::size_t bad_h = 0, bad_y = 0, bad_f = 0;
            for (int c = 0; c < ncols; ++c) {
                bad_h += count_diff(hv[c], h_want[static_cast<std::size_t>(first + c)].data(), I);
                bad_y += count_diff(yv[c], y_want[static_cast<std::size_t>(first + c)].data(), H);
                if (!got.forward.empty()) {
                    bad_f += count_diff(got.forward.data() + static_cast<std::size_t>(c) * H, y_want[static_cast<std::size_t>(first + c)].data(), H);
                }
            }
            const std::string at = where + ", " + std::to_string(ncols) + " columns: ";
            if (bad_h != 0) { fail(at + std::to_string(bad_h) + " h outputs differ from the GPU order"); }
            if (bad_y != 0) { fail(at + std::to_string(bad_y) + " down outputs differ from the GPU order"); }
            if (bad_f != 0) { fail(at + std::to_string(bad_f) + " ggml_a8_expert_forward outputs differ from the GPU order"); }
            if (isa == moe::CpuIsa::kScalar) {
                if (ncols == kCols) {
                    // The scalar route's BF16 down outputs against the FP64 oracle directly.
                    std::size_t bad = 0;
                    for (int c = 0; c < ncols; ++c) {
                        if (nonfinite(hq[static_cast<std::size_t>(c)])) { continue; }
                        for (int r = 0; r < H; ++r) {
                            const RowResult& rr = yr[static_cast<std::size_t>(c)][static_cast<std::size_t>(r)];
                            const double err    = std::fabs(static_cast<double>(bf16_value(yv[c][r])) - rr.exact);
                            bad += !(err <= rr.bound + std::ldexp(std::fabs(rr.exact) + rr.bound, -8) + std::ldexp(1.0, -134));
                        }
                    }
                    if (bad != 0) { fail(at + std::to_string(bad) + " BF16 down outputs exceed the FP32 bound plus BF16 rounding"); }
                }
                scalar[static_cast<std::size_t>(ncols)] = std::move(got);
                have_scalar                             = true;
                continue;
            }
            if (!have_scalar) { continue; } // --isa: no scalar run
            const Widths& ref = scalar[static_cast<std::size_t>(ncols)];
            const std::size_t dh = count_diff(got.h.data(), ref.h.data(), got.h.size());
            const std::size_t dy = count_diff(got.y.data(), ref.y.data(), got.y.size());
            const std::size_t df = got.forward.empty() ? 0 : count_diff(got.forward.data(), ref.forward.data(), got.forward.size());
            if (dh + dy + df != 0) {
                fail(at + std::to_string(dh) + " h, " + std::to_string(dy) + " down and " + std::to_string(df) +
                     " forward outputs differ from the scalar route");
            }
        }
    }
    std::printf("%s: binary32 rows within %.3f (gate/up) and %.3f (down) of the FP32 bound at worst\n", name.c_str(), worst_gate,
                worst_down);
}

// The worker team's 8-bit rounds (CpuExpertTeam::work_ggml_a8) against ggml_a8_expert_forward: three jobs of one,
// two and three columns at 1, 3 and 16 workers.
void check_team(std::mt19937& rng) {
    const QType gu = QType::GGML_IQ3_S, down = QType::GGML_IQ4_NL;
    const moe::GgmlExpertFormat f{.gate_up = gu, .down = down, .gate_row_bytes = row_bytes(gu, H), .down_row_bytes = row_bytes(down, I)};
    const std::size_t record = 2ULL * I * f.gate_row_bytes + static_cast<std::size_t>(H) * f.down_row_bytes;
    constexpr int kJobs = 3, kJobCols = 3;
    std::vector<std::uint8_t> bank(record * kJobs);
    for (int e = 0; e < kJobs; ++e) {
        random_rows(gu, 2 * I, H, rng, bank.data() + e * record);
        random_rows(down, H, I, rng, bank.data() + e * record + 2ULL * I * f.gate_row_bytes);
    }
    std::vector<std::uint16_t> x(static_cast<std::size_t>(H) * kJobCols);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (auto& v : x) { v = bf16(normal(rng)); }
    const auto at = [&](int j, int c) { return static_cast<std::size_t>(j * kJobCols + c) * H; };
    std::vector<std::uint16_t> want(static_cast<std::size_t>(H) * kJobCols * kJobs, 0);
    for (int j = 0; j < kJobs; ++j) {
        const std::uint16_t* xp[kJobCols];
        std::uint16_t* yp[kJobCols];
        for (int c = 0; c < kJobCols; ++c) {
            xp[c] = x.data() + static_cast<std::size_t>(c) * H;
            yp[c] = want.data() + at(j, c);
        }
        moe::ggml_a8_expert_forward(moe::best_cpu_isa(), bank.data() + j * record, f, j + 1, xp, yp);
    }
    for (const int workers : {1, 3, 16}) {
        moe::CpuExpertTeam team(
            {.workers = workers, .max_jobs = kJobs, .activation = moe::ExpertActivation::kA16, .ggml = true, .ggml_a8 = true});
        std::vector<std::uint16_t> got(want.size(), 0);
        std::vector<moe::CpuExpertJob> jobs(kJobs);
        for (int j = 0; j < kJobs; ++j) {
            auto& job  = jobs[static_cast<std::size_t>(j)];
            job.record = bank.data() + j * record;
            job.ggml   = f;
            job.a8     = true;
            job.ncols  = j + 1;
            for (int c = 0; c < job.ncols; ++c) {
                job.x[c] = x.data() + static_cast<std::size_t>(c) * H;
                job.y[c] = got.data() + at(j, c);
            }
        }
        team.run(jobs);
        if (got != want) { fail("the worker team's 8-bit GGUF route at " + std::to_string(workers) + " workers differs from ggml_a8_expert_forward"); }
    }
}

// --columns (information only): one expert call on one thread at 1-8 columns, every supported vector ISA, on the
// 8-bit route and on the exact route, for the GGUF pairs of the IQ3_S and IQ4_XS imports.
void columns() {
    std::mt19937 rng(20261027);
    for (const auto& [gu, down] : {std::pair{QType::GGML_IQ3_S, QType::GGML_IQ4_NL}, std::pair{QType::GGML_IQ4_XS, QType::GGML_IQ4_NL}}) {
        const moe::GgmlExpertFormat f{.gate_up = gu, .down = down, .gate_row_bytes = row_bytes(gu, H), .down_row_bytes = row_bytes(down, I)};
        std::vector<std::uint8_t> record(2ULL * I * f.gate_row_bytes + static_cast<std::size_t>(H) * f.down_row_bytes);
        random_rows(gu, 2 * I, H, rng, record.data());
        random_rows(down, H, I, rng, record.data() + 2ULL * I * f.gate_row_bytes);
        std::vector<std::uint16_t> x(static_cast<std::size_t>(H) * kCols), y(static_cast<std::size_t>(H) * kCols);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        for (auto& v : x) { v = bf16(normal(rng)); }
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
                            moe::ggml_a8_expert_forward(isa, record.data(), f, ncols, xp, yp);
                        } else {
                            moe::ggml_expert_forward(isa, record.data(), f, ncols, xp, yp);
                        }
                    };
                    call();
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < kRepeats; ++i) { call(); }
                    us[route] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / kRepeats;
                }
                std::printf("columns pair (%d, %d) %-12s ncols %d: a8 %8.1f us per call (%7.1f per column), exact %8.1f us\n",
                            static_cast<int>(gu), static_cast<int>(down), moe::cpu_isa_name(isa), ncols, us[0], us[0] / ncols, us[1]);
            }
        }
    }
}

} // namespace

INFERNIX_OP_TEST(offloaded_moe_ggml_a8_cpu) {
    if (argc > 1 && std::string(argv[1]) == "--columns") {
        columns();
        return 0;
    }
    // --isa NAME (cpu_isa_name, e.g. avx512-vnni): only that ISA's kernels against the GPU order and the oracle,
    // for a run under an emulator (Intel SDE) on a CPU without it.
    if (argc > 2 && std::string(argv[1]) == "--isa") {
        g_only_isa = argv[2];
        if (moe::cpu_isa_name(moe::best_cpu_isa()) != g_only_isa) {
            std::fprintf(stderr, "error: --isa %s is not this CPU's best ISA (%s)\n", argv[2], moe::cpu_isa_name(moe::best_cpu_isa()));
            return 1;
        }
    }
    try {
        std::mt19937 rng(20261027);
        check_codec(rng);
        const std::pair<QType, QType> pairs[] = {
            {QType::GGML_IQ2_S, QType::GGML_IQ4_NL},   {QType::GGML_IQ2_S, QType::GGML_Q2_0},
            {QType::GGML_IQ3_XXS, QType::GGML_IQ4_NL}, {QType::GGML_IQ3_XXS, QType::GGML_Q2_0},
            {QType::GGML_IQ3_S, QType::GGML_IQ4_NL},   {QType::GGML_IQ3_S, QType::GGML_Q2_0},
            {QType::GGML_IQ4_XS, QType::GGML_IQ4_NL},  {QType::GGML_IQ4_XS, QType::GGML_Q2_0}};
        for (const auto& [gu, down] : pairs) { run(gu, down, rng); }
        check_team(rng);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d ggml a8 CPU expert checks failed\n", g_failures);
        return 1;
    }
    std::printf("ggml a8 CPU experts: codec, every format pair, ISA and width bit-equal to the GPU order and within the FP32 bound\n");
    return 0;
}
