// lossless_linear (the --lossless route's projections) against an exact oracle.
//
// Weights are built from known codes and scales in every format the op takes: row-split Q2/Q4/Q5/Q6
// per 64 and Q8 per 32 (packed by the fixtures' packer), nvfp4 (E2M1 codes, swizzled E4M3 scales, an
// FP32 divisor), fp8_e4m3fn_row_bf16 (E4M3 codes, BF16 row scales), fp8_e4m3fn_row_f32 (FP32 row
// multipliers of every exponent range and both signs), dense nvfp4_rg16_v1 (the NVFP4 head's layout, one
// FP32 multiplier) and BF16. Each weight is an integer
// W times a power of two (and per row a factor: the FP8 row scale multiplies, the NVFP4 divisor
// divides), so each output's exact sum is accumulated on the host in a wide integer, the factor applied
// exactly (a quotient keeps a sticky bit), and rounded once to BF16 or FP32 (round to nearest, ties to
// even). Every output must equal it bit for bit: through the tensor-core route and through the exact
// fallback (columns whose 64-element blocks span more than 21 binades, cancelling sums, terms far below
// the largest, subnormal results, non-finite inputs), through the grouped form (per-tile rows), and
// through the tile mode (per-tile expert records, input and output column maps). Also MXFP8 (E4M3 code
// times a power of two per 32), GGUF blocks (ggml_rows_v1, ggml_tiles16_v1; random block bytes decoded
// on the host by ggml::chunk: kappa (d sub code - dmin min_sub), the Q4_K/Q5_K min term a second term
// per element), NVFP4 expert units (times an FP32 multiplier per row parity) and block FP8 records.
// Shapes: rows that do not fill a tile, a row range into an output offset, T across the 512-column
// chunk, K of the Qwen3.5 projections. The noise twin (ops::lossless_set_noise_seed, IQS-2's noise floor):
// every output one of its exact value's two neighbours, as often away from zero as the discarded fractions
// say, the same bits for the same seed.

#include "ops/op_family.h"
#include "core/arena.h"
#include "core/device.h"
#include "infernix/ops/ggml_linear.h"
#include "infernix/ops/lossless_linear.h"
#include "infernix/ops/lossless_ops.h"
#include "ops/ggml/ggml_exact.h"
#include "quantized_weight.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <memory>
#include <vector>

using namespace infernix;
namespace qwd = infernix::test::quantized_weight::detail;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL " << what << "\n";
    }
}

std::uint16_t bf16_bits_of_float(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

std::uint32_t f32_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

std::uint16_t f16_bits(float f) { return qwd::f32_to_f16(f); }

int bits_of(QType q) {
    return q == QType::Q2_G64_FP16 ? 2 : q == QType::Q4_G64_FP16 ? 4 : q == QType::Q5_G64_FP16 ? 5
           : q == QType::Q6_G64_FP16 ? 6 : 8;
}

enum class Factor { None, Multiply, Divide };

// A weight as the oracle sees it, element (r, k) = W 2^wexp, row r's sum times fm 2^fe (Multiply) or
// divided by fm 2^fe (Divide); and its device form.
struct HostWeight {
    std::string name;
    int n = 0, k = 0;
    std::vector<int> W, wexp;
    std::vector<int> W2, wexp2; // a second term per element (the GGUF K-quants' min term), or empty
    Factor factor = Factor::None;
    std::vector<std::uint32_t> fm;
    std::vector<int> fe;
    std::vector<std::uint8_t> fneg; // Multiply: the row factor's sign (empty: positive)
    DeviceBuffer payload{};
    Weight weight{};
};

void fp16_parts(std::uint16_t bits, int& m, int& e) {
    const int field = (bits >> 10) & 0x1F, mantissa = bits & 0x3FF;
    m = field == 0 ? mantissa : (1024 | mantissa);
    e = field == 0 ? 1 : field;
    if ((bits & 0x8000U) != 0) { m = -m; }
}

void e4m3_parts(std::uint32_t bits, int& m, int& e) {
    const int field = static_cast<int>((bits >> 3) & 0xF), mantissa = static_cast<int>(bits & 7U);
    m = field == 0 ? mantissa : (8 | mantissa);
    e = field == 0 ? 1 : field;
    if ((bits & 0x80U) != 0) { m = -m; }
}

// ------------------------------------------------------------------------------- formats

HostWeight make_row_split(QType qtype, int n, int k, std::mt19937& rng, bool wide_scales) {
    HostWeight h;
    h.name = "q" + std::to_string(bits_of(qtype));
    h.n    = n;
    h.k    = k;
    const qwd::QuantSpec spec = qwd::quant_spec(qtype);
    const int group = spec.group_size, padded_k = (k + 127) / 128 * 128, groups = padded_k / group;
    const int bits = bits_of(qtype);
    const int lo = bits == 8 ? -127 : -(1 << (bits - 1)), hi = bits == 8 ? 127 : (1 << (bits - 1)) - 1;
    std::uniform_int_distribution<int> code(lo, hi);
    std::vector<int> codes(static_cast<std::size_t>(n) * k);
    for (auto& c : codes) { c = code(rng); }
    std::uniform_real_distribution<float> unit(0.5F, 1.0F);
    std::uniform_int_distribution<int> exponent(wide_scales ? -24 : -9, wide_scales ? 8 : -5);
    std::vector<std::uint16_t> scales(static_cast<std::size_t>(n) * groups);
    for (std::size_t i = 0; i < scales.size(); ++i) {
        float s = std::ldexp(unit(rng), exponent(rng));
        if (i % 37 == 0) { s = -s; }
        if (i % 101 == 0) { s = 0.0F; }
        scales[i] = f16_bits(s);
    }
    h.W.resize(codes.size());
    h.wexp.resize(codes.size());
    for (int r = 0; r < n; ++r) {
        for (int i = 0; i < k; ++i) {
            int m = 0, e = 1;
            fp16_parts(scales[static_cast<std::size_t>(r) * groups + i / group], m, e);
            const std::size_t at = static_cast<std::size_t>(r) * k + i;
            h.W[at]    = codes[at] * m;
            h.wexp[at] = e - 25;
        }
    }
    // Device payload: codes, high planes, scales (256-byte aligned), as the loader binds them.
    const std::size_t code_row = static_cast<std::size_t>(groups) * qwd::nibble_bytes_per_group(spec);
    const std::size_t high_row = static_cast<std::size_t>(groups) * qwd::high_bytes_per_group(spec);
    const std::size_t code_bytes = code_row * n, high_bytes = high_row * n;
    const std::size_t high_off = (code_bytes + 255) / 256 * 256;
    const std::size_t scale_off = (high_off + high_bytes + 255) / 256 * 256;
    std::vector<std::uint8_t> bytes(scale_off + static_cast<std::size_t>(n) * groups * 2, 0);
    std::vector<std::int8_t> group_codes(static_cast<std::size_t>(group));
    for (int r = 0; r < n; ++r) {
        for (int g = 0; g < groups; ++g) {
            for (int i = 0; i < group; ++i) {
                const int kk = g * group + i;
                group_codes[static_cast<std::size_t>(i)] =
                    static_cast<std::int8_t>(kk < k ? codes[static_cast<std::size_t>(r) * k + kk] : 0);
            }
            qwd::pack_lowbit_group(group_codes.data(), spec,
                                   bytes.data() + r * code_row + g * qwd::nibble_bytes_per_group(spec),
                                   high_row == 0 ? nullptr
                                                 : bytes.data() + high_off + r * high_row + g * qwd::high_bytes_per_group(spec));
            std::memcpy(bytes.data() + scale_off + (static_cast<std::size_t>(r) * groups + g) * 2,
                        &scales[static_cast<std::size_t>(r) * groups + g], 2);
        }
    }
    h.payload = DeviceBuffer(bytes.size());
    h.payload.copy_from_host(bytes.data(), bytes.size(), 0);
    Weight w{};
    w.qtype           = qtype;
    w.layout          = QuantLayout::RowSplit;
    w.ndim            = 2;
    w.n               = n;
    w.k               = k;
    w.shape[0]        = w.padded_shape[0] = n;
    w.shape[1]        = k;
    w.padded_shape[1] = padded_k;
    w.group_size      = static_cast<std::uint32_t>(group);
    w.scale_dtype     = DType::FP16;
    w.qdata           = h.payload.p;
    w.qhigh           = high_row == 0 ? nullptr : static_cast<std::uint8_t*>(h.payload.p) + high_off;
    w.scales          = static_cast<std::uint8_t*>(h.payload.p) + scale_off;
    h.weight          = w;
    return h;
}

// nvfp4 in block_scale_k16_m128x4_v1: codes [n][k / 2] (low nibble first), the scales swizzled.
HostWeight make_nvfp4(int n, int k, std::mt19937& rng, float divisor) {
    HostWeight h;
    h.name = "nvfp4";
    h.n    = n;
    h.k    = k;
    std::uniform_int_distribution<int> nibble(0, 15), scale_word(0, 0x7E);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(n) * k / 2, 0);
    const std::size_t scale_bytes = static_cast<std::size_t>(n / 128) * (k / 64) * 512;
    std::vector<std::uint8_t> scales(scale_bytes, 0);
    h.W.resize(static_cast<std::size_t>(n) * k);
    h.wexp.resize(h.W.size());
    static const int twice[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    for (int r = 0; r < n; ++r) {
        for (int g = 0; g < k / 16; ++g) {
            int word = scale_word(rng);
            if ((r * 7 + g) % 53 == 0) { word = 0; }       // a zero scale
            if ((r * 3 + g) % 29 == 0) { word = 0x03; }    // a subnormal scale
            const int inner = r % 128;
            const std::size_t at = (static_cast<std::size_t>(r / 128) * (k / 64) + g / 4) * 512 +
                                   static_cast<std::size_t>(inner % 32) * 16 + (inner / 32) * 4 + g % 4;
            scales[at] = static_cast<std::uint8_t>(word);
            int m = 0, e = 1;
            e4m3_parts(static_cast<std::uint32_t>(word), m, e);
            for (int i = 0; i < 16; ++i) {
                const int kk = g * 16 + i, c = nibble(rng);
                std::uint8_t& byte = codes[static_cast<std::size_t>(r) * (k / 2) + kk / 2];
                byte = static_cast<std::uint8_t>(byte | (c << (4 * (kk & 1))));
                const int v = twice[c & 7] * ((c & 8) ? -1 : 1);
                h.W[static_cast<std::size_t>(r) * k + kk]    = v * m;
                h.wexp[static_cast<std::size_t>(r) * k + kk] = e - 11;
            }
        }
    }
    h.factor = Factor::Divide;
    const std::uint32_t d = f32_bits(divisor);
    const int field = static_cast<int>((d >> 23) & 0xFF), mantissa = static_cast<int>(d & 0x7FFFFF);
    h.fm.assign(static_cast<std::size_t>(n), static_cast<std::uint32_t>(field == 0 ? mantissa : ((1 << 23) | mantissa)));
    h.fe.assign(static_cast<std::size_t>(n), (field == 0 ? 1 : field) - 150);
    const std::size_t scale_off = (codes.size() + 255) / 256 * 256;
    std::vector<std::uint8_t> bytes(scale_off + scale_bytes, 0);
    std::memcpy(bytes.data(), codes.data(), codes.size());
    std::memcpy(bytes.data() + scale_off, scales.data(), scale_bytes);
    h.payload = DeviceBuffer(bytes.size());
    h.payload.copy_from_host(bytes.data(), bytes.size(), 0);
    Weight w{};
    w.qtype                = QType::NVFP4;
    w.layout               = QuantLayout::BlockScaleK16M128x4;
    w.ndim                 = 2;
    w.n                    = n;
    w.k                    = k;
    w.shape[0]             = w.padded_shape[0] = n;
    w.shape[1]             = w.padded_shape[1] = k;
    w.group_size           = 16;
    w.qdata                = h.payload.p;
    w.scales               = static_cast<std::uint8_t*>(h.payload.p) + scale_off;
    w.weight_scale_divisor = divisor;
    h.weight               = w;
    return h;
}

// fp8_e4m3fn_row_bf16 in row_scale_v1: codes [n][k], then the BF16 row scales.
HostWeight make_fp8_row(int n, int k, std::mt19937& rng) {
    HostWeight h;
    h.name = "fp8row";
    h.n    = n;
    h.k    = k;
    std::uniform_int_distribution<int> word(0, 255);
    std::uniform_real_distribution<float> unit(0.5F, 1.0F);
    std::uniform_int_distribution<int> exponent(-12, -2);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(n) * k);
    std::vector<std::uint16_t> scales(static_cast<std::size_t>(n));
    h.W.resize(codes.size());
    h.wexp.assign(codes.size(), -9);
    h.factor = Factor::Multiply;
    h.fm.resize(static_cast<std::size_t>(n));
    h.fe.resize(static_cast<std::size_t>(n));
    for (int r = 0; r < n; ++r) {
        const bool zero_row = r % 23 == 7; // a zero scale: its codes are zeros of either sign
        std::uint16_t s = zero_row ? 0 : bf16_bits_of_float(std::ldexp(unit(rng), exponent(rng)));
        if (r % 31 == 3) { s = 0x0005; } // a subnormal scale
        scales[static_cast<std::size_t>(r)] = s;
        const int field = (s >> 7) & 0xFF, mantissa = s & 0x7F;
        h.fm[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(field == 0 ? mantissa : (128 | mantissa));
        h.fe[static_cast<std::size_t>(r)] = (field == 0 ? 1 : field) - 134;
        for (int i = 0; i < k; ++i) {
            int c = word(rng);
            if ((c & 0x7F) == 0x7F) { c ^= 1; } // not NaN
            if (zero_row) { c &= 0x80; }
            codes[static_cast<std::size_t>(r) * k + i] = static_cast<std::uint8_t>(c);
            int m = 0, e = 1;
            e4m3_parts(static_cast<std::uint32_t>(c), m, e);
            h.W[static_cast<std::size_t>(r) * k + i] = m * (1 << (e - 1));
        }
    }
    const std::size_t scale_off = (codes.size() + 255) / 256 * 256;
    std::vector<std::uint8_t> bytes(scale_off + scales.size() * 2, 0);
    std::memcpy(bytes.data(), codes.data(), codes.size());
    std::memcpy(bytes.data() + scale_off, scales.data(), scales.size() * 2);
    h.payload = DeviceBuffer(bytes.size());
    h.payload.copy_from_host(bytes.data(), bytes.size(), 0);
    Weight w{};
    w.qtype       = QType::FP8_E4M3FN_ROW_BF16;
    w.layout      = QuantLayout::RowScale;
    w.ndim        = 2;
    w.n           = n;
    w.k           = k;
    w.shape[0]    = w.padded_shape[0] = n;
    w.shape[1]    = w.padded_shape[1] = k;
    w.scale_dtype = DType::BF16;
    w.qdata       = h.payload.p;
    w.scales      = static_cast<std::uint8_t*>(h.payload.p) + scale_off;
    h.weight      = w;
    return h;
}

// fp8_e4m3fn_row_f32 in row_scale_v1: codes [n][k], then the FP32 row multipliers (full 24-bit
// significands, exponents from 2^-140 (subnormal) to 2^20, some negative, a zero row).
HostWeight make_fp8_row_f32(int n, int k, std::mt19937& rng) {
    HostWeight h;
    h.name = "fp8row f32";
    h.n    = n;
    h.k    = k;
    std::uniform_int_distribution<int> word(0, 255);
    std::uniform_real_distribution<float> unit(0.5F, 1.0F);
    std::uniform_int_distribution<int> exponent(-30, 20);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(n) * k);
    std::vector<float> scales(static_cast<std::size_t>(n));
    h.W.resize(codes.size());
    h.wexp.assign(codes.size(), -9);
    h.factor = Factor::Multiply;
    h.fm.resize(static_cast<std::size_t>(n));
    h.fe.resize(static_cast<std::size_t>(n));
    h.fneg.resize(static_cast<std::size_t>(n));
    for (int r = 0; r < n; ++r) {
        const bool zero_row = r % 23 == 7;
        float s = zero_row ? 0.0F : std::ldexp(unit(rng), exponent(rng));
        if (r % 31 == 3) { s = std::ldexp(unit(rng), -140); } // a subnormal multiplier
        if (r % 17 == 5) { s = -s; }                            // a negative multiplier
        scales[static_cast<std::size_t>(r)] = s;
        const std::uint32_t bits = f32_bits(s);
        const int field = static_cast<int>((bits >> 23) & 0xFF), mantissa = static_cast<int>(bits & 0x7FFFFF);
        h.fm[static_cast<std::size_t>(r)]   = static_cast<std::uint32_t>(field == 0 ? mantissa : ((1 << 23) | mantissa));
        h.fe[static_cast<std::size_t>(r)]   = (field == 0 ? 1 : field) - 150;
        h.fneg[static_cast<std::size_t>(r)] = static_cast<std::uint8_t>(bits >> 31);
        for (int i = 0; i < k; ++i) {
            int c = word(rng);
            if ((c & 0x7F) == 0x7F) { c ^= 1; } // not NaN
            if (zero_row) { c &= 0x80; }
            codes[static_cast<std::size_t>(r) * k + i] = static_cast<std::uint8_t>(c);
            int m = 0, e = 1;
            e4m3_parts(static_cast<std::uint32_t>(c), m, e);
            h.W[static_cast<std::size_t>(r) * k + i] = m * (1 << (e - 1));
        }
    }
    const std::size_t scale_off = (codes.size() + 255) / 256 * 256;
    std::vector<std::uint8_t> bytes(scale_off + scales.size() * 4, 0);
    std::memcpy(bytes.data(), codes.data(), codes.size());
    std::memcpy(bytes.data() + scale_off, scales.data(), scales.size() * 4);
    h.payload = DeviceBuffer(bytes.size());
    h.payload.copy_from_host(bytes.data(), bytes.size(), 0);
    Weight w{};
    w.qtype       = QType::FP8_E4M3FN_ROW_F32;
    w.layout      = QuantLayout::RowScale;
    w.ndim        = 2;
    w.n           = n;
    w.k           = k;
    w.shape[0]    = w.padded_shape[0] = n;
    w.shape[1]    = w.padded_shape[1] = k;
    w.scale_dtype = DType::FP32;
    w.qdata       = h.payload.p;
    w.scales      = static_cast<std::uint8_t*>(h.payload.p) + scale_off;
    h.weight      = w;
    return h;
}

// BF16 [n][k]. Spread: normal weights, some subnormal, zero and far-apart exponents (most rows hold a
// weight more than 43 binades below its block's largest: the exact fallback) and an infinite weight.
// Grid: model-like weights; in the first two 64-row tiles some weights 22-43 binades below their block's
// largest (the lo grid) and blocks led by a large weight, one row with a weight further below, a zero row.
HostWeight make_bf16(int n, int k, std::mt19937& rng, bool spread) {
    HostWeight h;
    h.name = spread ? "bf16 spread" : "bf16 grid";
    h.n    = n;
    h.k    = k;
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> values(static_cast<std::size_t>(n) * k);
    h.W.resize(values.size());
    h.wexp.resize(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const int r = static_cast<int>(i / k), c = static_cast<int>(i % k);
        float v = normal(rng) * 0.05F;
        if (spread) {
            if (i % 97 == 5) { v = std::ldexp(v, 60); }
            if (i % 89 == 3) { v = std::ldexp(v, -125); }
        } else {
            if (r < 128 && r % 5 == 1 && c % 61 == 7) { v = std::ldexp(v, -30); }
            if (r < 128 && r % 7 == 3 && c % 64 == 0) { v = std::ldexp(v, 25); }
            if (r == 77 && c == k / 2) { v = std::ldexp(v, -60); }
            if (r == n - 5) { v = 0.0F; }
        }
        if (i % 13 == 1) { v = 0.0F; }
        std::uint16_t b = bf16_bits_of_float(v);
        if (spread && i == values.size() / 2 + 3) { b = 0x7F80; } // +Inf
        values[i] = b;
        const int field = (b >> 7) & 0xFF, mantissa = b & 0x7F;
        if (field == 255) {
            h.W[i] = 1 << 30; // marks Inf for the oracle
            h.wexp[i] = 0;
            continue;
        }
        const int M = field == 0 ? mantissa : (128 | mantissa);
        h.W[i]    = (b & 0x8000) ? -M : M;
        h.wexp[i] = (field == 0 ? 1 : field) - 134;
    }
    h.payload = DeviceBuffer(values.size() * 2);
    h.payload.copy_from_host(values.data(), values.size() * 2, 0);
    Weight w{};
    w.qtype    = QType::BF16;
    w.layout   = QuantLayout::Contiguous;
    w.ndim     = 2;
    w.n        = n;
    w.k        = k;
    w.shape[0] = w.padded_shape[0] = n;
    w.shape[1] = w.padded_shape[1] = k;
    w.qdata    = h.payload.p;
    h.weight   = w;
    return h;
}

// E4M3FN value times 2^9 as an integer (no NaN codes are generated).
int e4m3_int(std::uint8_t b) {
    const int field = (b >> 3) & 0xF, mant = b & 7;
    const int M = field == 0 ? mant : (8 | mant);
    const int w = M << ((field == 0 ? 1 : field) - 1);
    return (b & 0x80U) != 0 ? -w : w;
}

std::uint8_t random_e4m3(std::mt19937& rng) {
    std::uniform_int_distribution<int> byte(0, 255);
    for (;;) {
        const int b = byte(rng);
        if ((b & 0x7F) != 0x7F) { return static_cast<std::uint8_t>(b); }
    }
}

int e2m1_twice_host(int nibble) {
    static const int kTable[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    const int v = kTable[nibble & 7];
    return (nibble & 8) != 0 ? -v : v;
}

void f32_parts(std::uint32_t bits, std::uint32_t& m, int& e) {
    const int field = static_cast<int>((bits >> 23) & 0xFF), mant = static_cast<int>(bits & 0x7FFFFF);
    m = static_cast<std::uint32_t>(field == 0 ? mant : ((1 << 23) | mant));
    e = (field == 0 ? 1 : field) - 150;
}

// A generated matrix: its code bytes, its scale bytes (formats with a separate plane), its host decode.
struct GenMatrix {
    std::vector<std::uint8_t> codes, scales;
    HostWeight h;
    int row_bytes = 0;
};

// GGUF [n][k] (rows, or tiles of 16 rows padded with zero blocks): random block bytes, then a binary16 d
// (and dmin) of varied magnitude; every 13th block's d zero.
GenMatrix gen_ggml(QType q, int n, int k, std::mt19937& rng, bool tiles16) {
    GenMatrix g;
    const int elems = static_cast<int>(ggml_block_elems(q)), bytes = static_cast<int>(ggml_block_bytes(q));
    const int blocks = k / elems;
    g.row_bytes = blocks * bytes;
    const int padded_n = tiles16 ? (n + 15) / 16 * 16 : n;
    g.codes.assign(static_cast<std::size_t>(padded_n) * g.row_bytes, 0);
    std::uniform_int_distribution<int> byte(0, 255), exponent(-14, -4);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    const int d_at = q == QType::GGML_Q6_K ? 208 : 0;
    const bool two = q == QType::GGML_Q4_K || q == QType::GGML_Q5_K;
    const auto block_at = [&](int r, int b) -> std::uint8_t* {
        return tiles16 ? g.codes.data() + static_cast<std::size_t>(r / 16) * 16 * g.row_bytes +
                             static_cast<std::size_t>(16 * b + r % 16) * bytes
                       : g.codes.data() + static_cast<std::size_t>(r) * g.row_bytes + static_cast<std::size_t>(b) * bytes;
    };
    for (int r = 0; r < n; ++r) {
        for (int b = 0; b < blocks; ++b) {
            std::uint8_t* blk = block_at(r, b);
            for (int i = 0; i < bytes; ++i) { blk[i] = static_cast<std::uint8_t>(byte(rng)); }
            const int spread = (r * 7 + b) % 11 == 0 ? 12 : 0; // some blocks far from the others
            const std::uint16_t d = (r * blocks + b) % 13 == 5 ? 0 : f16_bits(std::ldexp(normal(rng), exponent(rng) - spread));
            blk[d_at] = static_cast<std::uint8_t>(d & 0xFF);
            blk[d_at + 1] = static_cast<std::uint8_t>(d >> 8);
            if (two) {
                const std::uint16_t dm = f16_bits(std::ldexp(std::fabs(normal(rng)), exponent(rng)));
                blk[2] = static_cast<std::uint8_t>(dm & 0xFF);
                blk[3] = static_cast<std::uint8_t>(dm >> 8);
            }
        }
    }
    HostWeight& h = g.h;
    h.n = n;
    h.k = k;
    h.W.resize(static_cast<std::size_t>(n) * k);
    h.wexp.resize(h.W.size());
    if (two) {
        h.W2.resize(h.W.size());
        h.wexp2.resize(h.W.size());
    }
    const int kappa = ggml::kappa_log2(q);
    for (int r = 0; r < n; ++r) {
        for (int i = 0; i < k; i += 8) {
            const std::uint8_t* blk = block_at(r, i / elems);
            ggml::Chunk c{};
            switch (q) {
            case QType::GGML_Q2_0: c = ggml::chunk<QType::GGML_Q2_0>(blk, (i % elems) / 8); break;
            case QType::GGML_Q4_K: c = ggml::chunk<QType::GGML_Q4_K>(blk, (i % elems) / 8); break;
            case QType::GGML_Q5_K: c = ggml::chunk<QType::GGML_Q5_K>(blk, (i % elems) / 8); break;
            case QType::GGML_Q6_K: c = ggml::chunk<QType::GGML_Q6_K>(blk, (i % elems) / 8); break;
            case QType::GGML_IQ2_S: c = ggml::chunk<QType::GGML_IQ2_S>(blk, (i % elems) / 8); break;
            case QType::GGML_IQ3_XXS: c = ggml::chunk<QType::GGML_IQ3_XXS>(blk, (i % elems) / 8); break;
            case QType::GGML_IQ3_S: c = ggml::chunk<QType::GGML_IQ3_S>(blk, (i % elems) / 8); break;
            case QType::GGML_IQ4_XS: c = ggml::chunk<QType::GGML_IQ4_XS>(blk, (i % elems) / 8); break;
            case QType::GGML_IQ4_NL: c = ggml::chunk<QType::GGML_IQ4_NL>(blk, (i % elems) / 8); break;
            default: throw std::logic_error("gen_ggml: format");
            }
            int m = 0, e = 1, mm = 0, me = 1;
            fp16_parts(c.d, m, e);
            fp16_parts(c.dmin, mm, me);
            for (int j = 0; j < 8; ++j) {
                const std::size_t at = static_cast<std::size_t>(r) * k + i + j;
                h.W[at]    = c.code[j] * m * c.sub;
                h.wexp[at] = e - 25 + kappa;
                if (two) {
                    h.W2[at]    = -mm * c.min_sub;
                    h.wexp2[at] = me - 25 + kappa;
                }
            }
        }
    }
    return g;
}

// MXFP8 [n][k]: E4M3 codes, UE8M0 scales in 16-row tiles; one row with extreme scales (2^-126, 2^123).
GenMatrix gen_mxfp8(int n, int k, std::mt19937& rng) {
    GenMatrix g;
    g.codes.resize(static_cast<std::size_t>(n) * k);
    g.scales.assign(static_cast<std::size_t>((n + 15) / 16) * 16 * (k / 32), 127);
    std::uniform_int_distribution<int> e8(118, 131);
    for (auto& c : g.codes) { c = random_e4m3(rng); }
    HostWeight& h = g.h;
    h.n = n;
    h.k = k;
    h.W.resize(g.codes.size());
    h.wexp.resize(g.codes.size());
    for (int r = 0; r < n; ++r) {
        for (int gr = 0; gr < k / 32; ++gr) {
            int e = e8(rng);
            if (r == n / 2) { e = gr % 2 == 0 ? 1 : 250; }
            g.scales[(static_cast<std::size_t>(r / 16) * (k / 32) + gr) * 16 + r % 16] = static_cast<std::uint8_t>(e);
            for (int j = 0; j < 32; ++j) {
                const std::size_t at = static_cast<std::size_t>(r) * k + gr * 32 + j;
                h.W[at]    = e4m3_int(g.codes[at]);
                h.wexp[at] = e - 136;
            }
        }
    }
    return g;
}

// nvfp4_expert_rg16_v1 [n][k] (n a multiple of 16): 144-byte units, positive E4M3 scales; the sum times
// alpha[r & 1] (FP32 bits).
GenMatrix gen_rg16(int n, int k, std::mt19937& rng, const std::uint32_t (&alpha)[2]) {
    GenMatrix g;
    const int blocks = k / 16;
    g.codes.resize(static_cast<std::size_t>(n / 16) * blocks * 144);
    std::uniform_int_distribution<int> byte(0, 255), field(1, 14), mant(0, 7);
    for (auto& c : g.codes) { c = static_cast<std::uint8_t>(byte(rng)); }
    HostWeight& h = g.h;
    h.n = n;
    h.k = k;
    h.W.resize(static_cast<std::size_t>(n) * k);
    h.wexp.resize(h.W.size());
    h.factor = Factor::Multiply;
    h.fm.resize(static_cast<std::size_t>(n));
    h.fe.resize(static_cast<std::size_t>(n));
    for (int r = 0; r < n; ++r) {
        f32_parts(alpha[r & 1], h.fm[static_cast<std::size_t>(r)], h.fe[static_cast<std::size_t>(r)]);
        for (int b = 0; b < blocks; ++b) {
            std::uint8_t* unit = g.codes.data() + (static_cast<std::size_t>(r / 16) * blocks + b) * 144;
            unit[128 + r % 16] = static_cast<std::uint8_t>((field(rng) << 3) | mant(rng));
            int m = 0, e = 1;
            e4m3_parts(unit[128 + r % 16], m, e);
            for (int j = 0; j < 16; ++j) {
                const std::uint8_t byte_j = unit[32 * (j / 4) + 4 * (r % 8) + j % 4];
                const int nibble = (r % 16) < 8 ? (byte_j & 0xF) : (byte_j >> 4);
                const std::size_t at = static_cast<std::size_t>(r) * k + b * 16 + j;
                h.W[at]    = e2m1_twice_host(nibble) * m;
                h.wexp[at] = e - 11;
            }
        }
    }
    return g;
}

// Block FP8 [n][k] (multiples of 128): E4M3 codes, BF16 tile scales [n / 128][k / 128] of both signs.
GenMatrix gen_fp8b128(int n, int k, std::mt19937& rng) {
    GenMatrix g;
    g.codes.resize(static_cast<std::size_t>(n) * k);
    for (auto& c : g.codes) { c = random_e4m3(rng); }
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_int_distribution<int> exponent(-20, -6);
    const int tk = k / 128;
    std::vector<std::uint16_t> sc(static_cast<std::size_t>(n / 128) * tk);
    for (std::size_t i = 0; i < sc.size(); ++i) { sc[i] = bf16_bits_of_float(std::ldexp(normal(rng), exponent(rng) - (i == 1 ? 60 : 0))); }
    g.scales.resize(sc.size() * 2);
    std::memcpy(g.scales.data(), sc.data(), g.scales.size());
    HostWeight& h = g.h;
    h.n = n;
    h.k = k;
    h.W.resize(g.codes.size());
    h.wexp.resize(g.codes.size());
    for (int r = 0; r < n; ++r) {
        for (int i = 0; i < k; ++i) {
            const std::uint16_t bits = sc[static_cast<std::size_t>(r / 128) * tk + i / 128];
            const int field = (bits >> 7) & 0xFF, mantissa = bits & 0x7F;
            int m = field == 0 ? mantissa : (128 | mantissa);
            if ((bits & 0x8000U) != 0) { m = -m; }
            const std::size_t at = static_cast<std::size_t>(r) * k + i;
            h.W[at]    = e4m3_int(g.codes[at]) * m;
            h.wexp[at] = (field == 0 ? 1 : field) - 143;
        }
    }
    return g;
}

HostWeight make_ggml(QType q, int n, int k, std::mt19937& rng, bool tiles16, const std::string& name) {
    GenMatrix g = gen_ggml(q, n, k, rng, tiles16);
    HostWeight h = std::move(g.h);
    h.name = name + (tiles16 ? " tiles16" : " rows");
    h.payload = DeviceBuffer(g.codes.size());
    h.payload.copy_from_host(g.codes.data(), g.codes.size(), 0);
    Weight w{};
    w.qtype    = q;
    w.layout   = tiles16 ? QuantLayout::GgmlTiles16 : QuantLayout::GgmlRows;
    w.ndim     = 2;
    w.n        = n;
    w.k        = k;
    w.shape[0] = w.padded_shape[0] = n;
    w.shape[1] = w.padded_shape[1] = k;
    w.qdata    = h.payload.p;
    h.weight   = w;
    return h;
}

HostWeight make_mxfp8(int n, int k, std::mt19937& rng) {
    GenMatrix g = gen_mxfp8(n, k, rng);
    HostWeight h = std::move(g.h);
    h.name = "mxfp8";
    const std::size_t scale_at = (g.codes.size() + 255) / 256 * 256;
    h.payload = DeviceBuffer(scale_at + g.scales.size());
    h.payload.copy_from_host(g.codes.data(), g.codes.size(), 0);
    h.payload.copy_from_host(g.scales.data(), g.scales.size(), scale_at);
    Weight w{};
    w.qtype    = QType::FP8_E4M3FN_G32_E8M0;
    w.layout   = QuantLayout::G32ScaleRows16;
    w.ndim     = 2;
    w.n        = n;
    w.k        = k;
    w.shape[0] = w.padded_shape[0] = n;
    w.shape[1] = w.padded_shape[1] = k;
    w.qdata    = h.payload.p;
    w.scales   = static_cast<const std::uint8_t*>(h.payload.p) + scale_at;
    h.weight   = w;
    return h;
}

// A dense nvfp4_rg16_v1 matrix [n][k] (n a multiple of 16): gen_rg16's units, one FP32 multiplier for
// every row (the Qwen3.5 NVFP4 head).
HostWeight make_rg16_dense(int n, int k, std::mt19937& rng, float multiplier) {
    const std::uint32_t alpha[2] = {f32_bits(multiplier), f32_bits(multiplier)};
    GenMatrix g  = gen_rg16(n, k, rng, alpha);
    HostWeight h = std::move(g.h);
    h.name    = "nvfp4 rg16 dense";
    h.payload = DeviceBuffer(g.codes.size());
    h.payload.copy_from_host(g.codes.data(), g.codes.size(), 0);
    Weight w{};
    w.qtype             = QType::NVFP4_MUL;
    w.layout            = QuantLayout::Rg16;
    w.ndim              = 2;
    w.n                 = n;
    w.k                 = k;
    w.shape[0]          = w.padded_shape[0] = n;
    w.shape[1]          = w.padded_shape[1] = k;
    w.group_size        = 16;
    w.qdata             = h.payload.p;
    w.weight_multiplier = multiplier;
    h.weight            = w;
    return h;
}

// ------------------------------------------------------------------------------- exact oracle

// sum_k W M at exponent e - 134 + wexp, held in 32-bit chunks (deferred carries) from 2^-300.
constexpr int kOracleWords = 26;
constexpr int kOracleBias  = 300;

// The exact value's two neighbours at the output precision (toward and away from zero; equal when the value
// is representable) and its position between them in units of their gap: the noise twin rounds away with
// that probability.
struct Neighbours {
    std::uint32_t toward = 0;
    std::uint32_t away   = 0;
    double fraction      = 0.0;
};

// The correctly rounded BF16 (prec 8) or FP32 (prec 24) bits of sign * (mag + tail) * 2^unit, the tail
// a nonzero fraction of mag's last bit when sticky; with neighbours, also the value's two neighbours.
std::uint32_t round_exact(const std::vector<std::uint32_t>& mag, int unit, bool negative, bool bf16, bool sticky_tail,
                          Neighbours* neighbours = nullptr) {
    const int prec = bf16 ? 8 : 24;
    const std::uint32_t sign = negative ? (bf16 ? 0x8000U : 0x80000000U) : 0U;
    int top = -1;
    for (int i = static_cast<int>(mag.size()) - 1; i >= 0 && top < 0; --i) {
        for (int b = 31; b >= 0; --b) {
            if ((mag[static_cast<std::size_t>(i)] >> b) & 1U) {
                top = 32 * i + b;
                break;
            }
        }
    }
    if (top < 0) {
        if (neighbours != nullptr) { *neighbours = Neighbours{sign, sign, 0.0}; }
        return sign;
    }
    const auto bit = [&](int p) -> int {
        if (p < 0 || p >= 32 * static_cast<int>(mag.size())) { return 0; }
        return static_cast<int>((mag[static_cast<std::size_t>(p / 32)] >> (p % 32)) & 1U);
    };
    const int exponent = top + unit;
    const int ulp      = std::max(exponent - prec + 1, -126 - prec + 1);
    const int low      = ulp - unit; // position of the result's last bit within mag
    if (low <= 0 && sticky_tail) { throw std::logic_error("oracle: a sticky tail needs bits below the result"); }
    std::uint64_t q = 0;
    for (int p = top; p >= std::max(low, 0); --p) { q = (q << 1) | static_cast<std::uint64_t>(bit(p)); }
    if (low < 0) { q <<= -low; }
    const auto pack = [&](std::uint64_t value) -> std::uint32_t {
        int e = ulp;
        if (value >= (1ULL << prec)) {
            value >>= 1;
            ++e;
        }
        if (value < (1ULL << (prec - 1))) { return sign | static_cast<std::uint32_t>(value); }
        const int biased = e + prec - 1 + 127;
        if (biased >= 255) { return sign | (bf16 ? 0x7F80U : 0x7F800000U); }
        return sign | (static_cast<std::uint32_t>(biased) << (prec - 1)) |
               static_cast<std::uint32_t>(value - (1ULL << (prec - 1)));
    };
    bool up = false;
    if (low > 0) {
        const int half = bit(low - 1);
        bool sticky = sticky_tail;
        for (int p = 0; p < low - 1 && !sticky; ++p) { sticky = bit(p) != 0; }
        up = half != 0 && (sticky || (q & 1) != 0);
    }
    if (neighbours != nullptr) {
        double fraction = 0.0, scale = 0.5;
        bool inexact = sticky_tail;
        for (int p = low - 1; p >= 0; --p, scale *= 0.5) {
            if (bit(p) != 0) {
                inexact = true;
                fraction += scale;
            }
        }
        *neighbours = Neighbours{pack(q), inexact ? pack(q + 1) : pack(q), inexact ? std::max(fraction, 0x1p-60) : 0.0};
    }
    return pack(up ? q + 1 : q);
}

// The design reference's activation scales (FP32 bits, 0: none): a multiplier of the row (tiles: its
// parity's), a multiplier of the column, and a divisor.
struct ActScales {
    std::uint32_t row_multiplier = 0;
    std::uint32_t column_multiplier = 0;
    std::uint32_t divisor = 0;
};

std::uint32_t oracle(const HostWeight& w, int row, const std::uint16_t* x, bool bf16, const ActScales& act = {},
                     Neighbours* neighbours = nullptr) {
    std::vector<long long> chunks(kOracleWords + 1, 0);
    bool nan = false, pos_inf = false, neg_inf = false;
    for (int k = 0; k < w.k; ++k) {
        const std::size_t at = static_cast<std::size_t>(row) * w.k + k;
        const long long wi = w.W[at];
        const bool w_inf = wi == (1LL << 30);
        const bool two   = !w.W2.empty();
        const std::uint16_t xb = x[k];
        const int field = (xb >> 7) & 0xFF, mant = xb & 0x7F;
        const bool xneg = (xb >> 15) != 0;
        const bool x_zero = field == 0 && mant == 0;
        if (field == 255 || w_inf) {
            int sign = wi > 0 ? 1 : (wi < 0 ? -1 : 0); // the weight's sign (zero: a zero weight)
            if (two) {
                const double v = std::ldexp(static_cast<double>(wi), w.wexp[at]) + std::ldexp(static_cast<double>(w.W2[at]), w.wexp2[at]);
                sign = v > 0.0 ? 1 : (v < 0.0 ? -1 : 0);
            }
            if ((field == 255 && mant != 0) || (field == 255 && sign == 0) || (w_inf && x_zero)) {
                nan = true;
            } else if (xneg != (w_inf ? false : sign < 0)) {
                neg_inf = true;
            } else {
                pos_inf = true;
            }
            continue;
        }
        const long long M = field == 0 ? mant : (128 | mant);
        if (M == 0) { continue; }
        const int e = field == 0 ? 1 : field;
        const auto add = [&](long long Wt, int wexpt) {
            long long p = Wt * M;
            if (xneg) { p = -p; }
            // |p| < 2^35: 24-bit pieces, each split at its bit offset into the 32-bit chunks it touches.
            for (int piece = 0; piece < 2; ++piece) {
                const long long part = piece == 0 ? p - (p >> 24) * (1LL << 24) : (p >> 24);
                const int offset = e - 134 + wexpt + kOracleBias + 24 * piece;
                const int word = offset / 32, shift = offset % 32;
                const long long lo = (part * (1LL << shift)) % (1LL << 32);
                const long long hi = (part * (1LL << shift) - lo) / (1LL << 32);
                chunks[static_cast<std::size_t>(word)] += lo;
                chunks[static_cast<std::size_t>(word + 1)] += hi;
            }
        };
        if (wi != 0) { add(wi, w.wexp[at]); }
        if (two && w.W2[at] != 0) { add(w.W2[at], w.wexp2[at]); }
    }
    // Non-finite results are IEEE's and take no rounding, noise twin or not.
    if (nan || (pos_inf && neg_inf)) {
        const std::uint32_t quiet = bf16 ? 0x7FC0U : 0x7FC00000U;
        if (neighbours != nullptr) { *neighbours = Neighbours{quiet, quiet, 0.0}; }
        return quiet;
    }
    if (pos_inf || neg_inf) {
        const std::uint32_t infinity =
            (neg_inf ? (bf16 ? 0x8000U : 0x80000000U) : 0U) | (bf16 ? 0x7F80U : 0x7F800000U);
        if (neighbours != nullptr) { *neighbours = Neighbours{infinity, infinity, 0.0}; }
        return infinity;
    }
    // Normalize to a signed magnitude.
    std::vector<long long> v(chunks);
    for (std::size_t i = 0; i + 1 < v.size(); ++i) {
        long long c = v[i] >> 32; // floor division by 2^32
        v[i] -= c * (1LL << 32);
        v[i + 1] += c;
    }
    bool negative = v.back() < 0;
    std::vector<std::uint32_t> mag(v.size());
    if (negative) {
        // |value|: negate the base-2^32 digits (two's complement over the digit vector).
        long long borrow = 0;
        for (std::size_t i = 0; i < v.size(); ++i) {
            long long d = -v[i] - borrow;
            borrow = 0;
            while (d < 0) {
                d += 1LL << 32;
                ++borrow;
            }
            mag[i] = static_cast<std::uint32_t>(d);
        }
    } else {
        for (std::size_t i = 0; i < v.size(); ++i) { mag[i] = static_cast<std::uint32_t>(v[i]); }
    }
    // The factors, exactly: every multiplier first (the product stays an integer), then each division of
    // the product padded by 2^64 (floor(floor(x / a) / b) = floor(x / ab), exact exactly when both are).
    int unit = -kOracleBias;
    bool sticky = false;
    const auto multiply = [&](std::uint64_t f) {
        std::uint64_t carry = 0;
        for (auto& d : mag) {
            const std::uint64_t t = static_cast<std::uint64_t>(d) * f + carry;
            d = static_cast<std::uint32_t>(t);
            carry = t >> 32;
        }
        mag.push_back(static_cast<std::uint32_t>(carry));
    };
    const auto divide = [&](std::uint64_t d) {
        mag.insert(mag.begin(), 2, 0U); // times 2^64
        unit -= 64;
        std::uint64_t rem = 0;
        for (int i = static_cast<int>(mag.size()) - 1; i >= 0; --i) {
            const std::uint64_t cur = (rem << 32) | mag[static_cast<std::size_t>(i)];
            mag[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(cur / d);
            rem = cur % d;
        }
        sticky = sticky || rem != 0;
    };
    const auto fp32_factor = [](std::uint32_t bits, std::uint64_t& m, int& e) {
        const int field = static_cast<int>((bits >> 23) & 0xFF), mantissa = static_cast<int>(bits & 0x7FFFFF);
        m = static_cast<std::uint64_t>(field == 0 ? mantissa : ((1 << 23) | mantissa));
        e = (field == 0 ? 1 : field) - 150;
    };
    if (w.factor == Factor::Multiply) {
        multiply(w.fm[static_cast<std::size_t>(row)]);
        unit += w.fe[static_cast<std::size_t>(row)];
        if (!w.fneg.empty() && w.fneg[static_cast<std::size_t>(row)] != 0) { negative = !negative; }
    }
    for (const std::uint32_t bits : {act.row_multiplier, act.column_multiplier}) {
        if (bits == 0) { continue; }
        std::uint64_t m = 0;
        int e = 0;
        fp32_factor(bits, m, e);
        multiply(m);
        unit += e;
    }
    if (w.factor == Factor::Divide) {
        divide(w.fm[static_cast<std::size_t>(row)]);
        unit -= w.fe[static_cast<std::size_t>(row)];
    }
    if (act.divisor != 0) {
        std::uint64_t m = 0;
        int e = 0;
        fp32_factor(act.divisor, m, e);
        divide(m);
        unit -= e;
    }
    bool zero = !sticky;
    for (const auto d : mag) { zero = zero && d == 0; }
    return round_exact(mag, unit, negative && !zero, bf16, sticky, neighbours);
}

// ------------------------------------------------------------------------------- activations

std::vector<std::uint16_t> make_columns(int k, int columns, std::mt19937& rng, int column_seed) {
    std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * columns);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (int t = 0; t < columns; ++t) {
        const int kind = (t + column_seed) % 9;
        for (int i = 0; i < k; ++i) {
            float v = normal(rng);
            switch (kind) {
            case 1: if (i % 7 == 0) { v *= 1e4F; } break;                       // outlier channels
            case 2: if (i % 64 == 5) { v = std::ldexp(v, 30); } break;          // a block spanning > 21 binades
            case 3: if (i % 3 == 0) { v = 0.0F; } else if (i % 5 == 0) { v = std::ldexp(v, -130); } break; // subnormals
            case 4: v = std::ldexp(v, -100); break;                             // tiny column
            case 5: v = std::ldexp(v, 100); break;                              // huge column (FP32 and BF16 overflow paths)
            case 6: if (i == k / 2) { v = std::numeric_limits<float>::infinity(); } break; // +Inf
            case 7: if (i == 3) { v = std::nanf(""); } break;                   // NaN
            default: break;
            }
            x[static_cast<std::size_t>(t) * k + i] = bf16_bits_of_float(v);
        }
    }
    return x;
}

// Columns whose exact sums cancel almost completely for rows whose adjacent weight pairs are equal
// (and partially otherwise): x = +a, -a pairs plus a tiny remainder.
std::vector<std::uint16_t> cancelling_columns(int k, int columns, std::mt19937& rng) {
    std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * columns, 0);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (int t = 0; t < columns; ++t) {
        for (int i = 0; i + 1 < k; i += 2) {
            const std::uint16_t a = bf16_bits_of_float(normal(rng) * 8.0F);
            x[static_cast<std::size_t>(t) * k + i]     = a;
            x[static_cast<std::size_t>(t) * k + i + 1] = static_cast<std::uint16_t>(a ^ 0x8000U);
        }
        x[static_cast<std::size_t>(t) * k + (t % k)] = bf16_bits_of_float(std::ldexp(normal(rng), -40 - t));
    }
    return x;
}

// ------------------------------------------------------------------------------- runs

template <class Get>
void compare(const std::string& label, int rows, int columns, const Get& got_want) {
    std::size_t wrong = 0;
    std::string first;
    for (int t = 0; t < columns; ++t) {
        for (int r = 0; r < rows; ++r) {
            std::uint32_t g = 0, want = 0;
            if (!got_want(t, r, g, want)) { continue; }
            if (g != want) {
                if (wrong == 0) {
                    char buf[160];
                    std::snprintf(buf, sizeof(buf), "column %d row %d got 0x%x want 0x%x", t, r, g, want);
                    first = buf;
                }
                ++wrong;
            }
        }
    }
    if (wrong != 0) { std::cerr << label << ": first mismatch " << first << "\n"; }
    check(wrong == 0, label + ": " + std::to_string(wrong) + " outputs differ from the correctly rounded exact sum");
    std::cout << label << ": " << static_cast<long long>(rows) * columns << " outputs checked\n";
}

void run(const std::string& label, const HostWeight& w, const std::vector<std::uint16_t>& x, int columns, int row_begin,
         int rows, int out_rows, int out_row, bool bf16) {
    DeviceBuffer dx(x.size() * 2);
    dx.copy_from_host(x.data(), x.size() * 2, 0);
    const int esize = bf16 ? 2 : 4;
    DeviceBuffer dout(static_cast<std::size_t>(out_rows) * columns * esize);
    CUDA_CHECK(cudaMemset(dout.p, 0xCD, static_cast<std::size_t>(out_rows) * columns * esize));
    Tensor tx(dx.p, DType::BF16, {w.k, columns});
    Tensor tout(dout.p, bf16 ? DType::BF16 : DType::FP32, {out_rows, columns});
    DeviceBuffer ws(ops::lossless_linear_workspace_bytes(w.weight, rows, columns));
    WorkspaceArena arena(DeviceSpan{ws.p, ws.bytes});
    ops::lossless_linear(tx, w.weight, row_begin, rows, tout, out_row, arena, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint8_t> got(static_cast<std::size_t>(out_rows) * columns * esize);
    dout.copy_to_host(got.data(), got.size(), 0);
    compare(label, rows, columns, [&](int t, int r, std::uint32_t& g, std::uint32_t& want) {
        const std::size_t i = static_cast<std::size_t>(t) * out_rows + out_row + r;
        std::memcpy(&g, got.data() + i * esize, esize);
        want = oracle(w, row_begin + r, x.data() + static_cast<std::size_t>(t) * w.k, bf16);
        return true;
    });
}

// The noise twin (ops::lossless_set_noise_seed): every output is one of its exact value's two neighbours, one
// seed gives the same bits every time and another seed other bits, and the inexact outputs rounded away from
// zero number what the discarded fractions predict (within six binomial standard deviations).
void run_noise(const std::string& label, const HostWeight& w, const std::vector<std::uint16_t>& x, int columns,
               bool bf16) {
    const int esize = bf16 ? 2 : 4;
    const std::size_t outputs = static_cast<std::size_t>(w.n) * columns;
    const auto launch = [&](std::uint64_t seed, std::uint32_t units = ops::kAllLosslessNoiseUnits, float scale = 1.0F) {
        ops::lossless_set_noise(seed, units, scale);
        DeviceBuffer dx(x.size() * 2);
        dx.copy_from_host(x.data(), x.size() * 2, 0);
        DeviceBuffer dout(outputs * esize);
        Tensor tx(dx.p, DType::BF16, {w.k, columns});
        Tensor tout(dout.p, bf16 ? DType::BF16 : DType::FP32, {w.n, columns});
        DeviceBuffer ws(ops::lossless_linear_workspace_bytes(w.weight, w.n, columns));
        WorkspaceArena arena(DeviceSpan{ws.p, ws.bytes});
        ops::lossless_linear(tx, w.weight, 0, w.n, tout, 0, arena, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint32_t> got(outputs, 0);
        std::vector<std::uint8_t> raw(outputs * esize);
        dout.copy_to_host(raw.data(), raw.size(), 0);
        for (std::size_t i = 0; i < outputs; ++i) { std::memcpy(&got[i], raw.data() + i * esize, esize); }
        return got;
    };
    const std::vector<std::uint32_t> first = launch(0x1234567ULL), again = launch(0x1234567ULL),
                                     other = launch(0x89ABCDEF01ULL);
    // The attribution: noise confined to other operation groups leaves this one round to nearest even, and a
    // weaker noise moves proportionally fewer values.
    const std::vector<std::uint32_t> nearest = launch(0);
    const std::vector<std::uint32_t> elsewhere =
        launch(0x1234567ULL, ops::kAllLosslessNoiseUnits & ~static_cast<std::uint32_t>(ops::LosslessNoiseUnit::Linear));
    const std::vector<std::uint32_t> weak = launch(0x1234567ULL, ops::kAllLosslessNoiseUnits, 0.25F);
    ops::lossless_set_noise_seed(0);
    check(elsewhere == nearest, label + ": noise confined to other groups changed this group's outputs");
    std::size_t outside = 0, inexact = 0, away = 0, reseeded = 0, weak_away = 0;
    double expected = 0.0, variance = 0.0;
    for (int t = 0; t < columns; ++t) {
        for (int r = 0; r < w.n; ++r) {
            const std::size_t i = static_cast<std::size_t>(t) * w.n + r;
            Neighbours n;
            (void)oracle(w, r, x.data() + static_cast<std::size_t>(t) * w.k, bf16, {}, &n);
            outside += first[i] != n.toward && first[i] != n.away ? 1U : 0U;
            reseeded += other[i] != first[i] ? 1U : 0U;
            if (n.toward == n.away) { continue; }
            ++inexact;
            away += first[i] == n.away ? 1U : 0U;
            weak_away += weak[i] == n.away ? 1U : 0U;
            expected += n.fraction;
            variance += n.fraction * (1.0 - n.fraction);
        }
    }
    check(outside == 0, label + ": " + std::to_string(outside) + " outputs are not a neighbour of the exact value");
    check(first == again, label + ": one seed gave different bits");
    check(reseeded > 0, label + ": another seed changed nothing");
    const double deviation = std::abs(static_cast<double>(away) - expected);
    check(inexact > 1000 && deviation <= 6.0 * std::sqrt(variance) + 1.0,
          label + ": " + std::to_string(away) + " of " + std::to_string(inexact) + " inexact outputs rounded away, " +
              std::to_string(expected) + " expected");
    // At a quarter of the strength a value moves from its nearest neighbour to the other with a quarter of the
    // probability the twin gives it (its distance from the nearest, in units of the gap): a value at fraction
    // f <= 1/2 of the way from `toward` to `away` ends away with 0.25 f, one at f > 1/2 (its nearest is `away`)
    // with 1 - 0.25 (1 - f).
    double weak_expected = 0.0, weak_variance = 0.0;
    for (int t = 0; t < columns; ++t) {
        for (int r = 0; r < w.n; ++r) {
            const std::size_t i = static_cast<std::size_t>(t) * w.n + r;
            Neighbours n;
            (void)oracle(w, r, x.data() + static_cast<std::size_t>(t) * w.k, bf16, {}, &n);
            if (n.toward == n.away) { continue; }
            const double p = n.fraction <= 0.5 ? 0.25 * n.fraction : 1.0 - 0.25 * (1.0 - n.fraction);
            weak_expected += p;
            weak_variance += p * (1.0 - p);
        }
    }
    check(std::abs(static_cast<double>(weak_away) - weak_expected) <= 6.0 * std::sqrt(weak_variance) + 1.0,
          label + ": at a quarter of the strength " + std::to_string(weak_away) + " values rounded away, " +
              std::to_string(weak_expected) + " expected");
    std::cout << label << ": " << outputs << " outputs, " << inexact << " inexact, " << away << " away (expected "
              << expected << "), " << reseeded << " changed by another seed; at 1/4 strength " << weak_away
              << " away (expected " << weak_expected << ")\n";
}

// The grouped form: tiles of 64 columns, each an expert's rows [e stride + offset, + rows) (some tiles
// skipped).
void run_grouped(const std::string& label, const HostWeight& w, int stride, int offset, int rows, int tiles,
                 std::mt19937& rng) {
    const int columns = 64 * tiles;
    const auto x = make_columns(w.k, columns, rng, 2);
    std::vector<std::int32_t> tile_experts(static_cast<std::size_t>(tiles));
    std::uniform_int_distribution<int> expert(0, (w.n - offset - rows) / stride);
    for (int i = 0; i < tiles; ++i) { tile_experts[static_cast<std::size_t>(i)] = i % 5 == 3 ? -1 : expert(rng); }
    DeviceBuffer dx(x.size() * 2), dexperts(tile_experts.size() * 4);
    dx.copy_from_host(x.data(), x.size() * 2, 0);
    dexperts.copy_from_host(tile_experts.data(), tile_experts.size() * 4, 0);
    DeviceBuffer dout(static_cast<std::size_t>(rows) * columns * 2);
    CUDA_CHECK(cudaMemset(dout.p, 0xCD, static_cast<std::size_t>(rows) * columns * 2));
    Tensor tx(dx.p, DType::BF16, {w.k, columns});
    Tensor texperts(dexperts.p, DType::I32, {tiles});
    Tensor tout(dout.p, DType::BF16, {rows, columns});
    DeviceBuffer ws(ops::lossless_linear_workspace_bytes(w.weight, rows, columns));
    WorkspaceArena arena(DeviceSpan{ws.p, ws.bytes});
    ops::lossless_linear_grouped(tx, w.weight, texperts, stride, offset, rows, tout, arena, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint16_t> got(static_cast<std::size_t>(rows) * columns);
    dout.copy_to_host(got.data(), got.size() * 2, 0);
    compare(label, rows, columns, [&](int t, int r, std::uint32_t& g, std::uint32_t& want) {
        const int e = tile_experts[static_cast<std::size_t>(t / 64)];
        const int first = e < 0 ? -1 : e * stride + offset;
        g = got[static_cast<std::size_t>(t) * rows + r];
        if (first < 0) {
            if (g != 0xCDCDU) { want = 0xCDCDU; } // a skipped tile stays unwritten
            else { return false; }
            return true;
        }
        want = oracle(w, first + r, x.data() + static_cast<std::size_t>(t) * w.k, true);
        return true;
    });
}

// The tile mode: a bank of `experts` records of one matrix (codes at codes_at, scales at scales_at in
// each record), `tiles` tiles each naming a record (every fifth none), column c reading x column
// in_columns[c] (some -1: a zero column) and written to out column out_columns[c] (some -1: none).
struct HostBank {
    std::string name;
    ops::LosslessBankMatrix matrix;
    std::vector<HostWeight> experts;
    std::vector<std::array<std::uint32_t, 2>> alpha;
    DeviceBuffer records{};
    std::size_t stride = 0;
};

template <class Gen>
HostBank make_bank(const std::string& name, QType q, int rows, int k, int experts, std::size_t codes_at, Gen gen) {
    HostBank b;
    b.name = name;
    std::vector<GenMatrix> mats;
    for (int e = 0; e < experts; ++e) { mats.push_back(gen(e)); }
    const std::size_t scales_at = (codes_at + mats[0].codes.size() + 255) / 256 * 256;
    b.stride = (scales_at + mats[0].scales.size() + 4095) / 4096 * 4096;
    std::vector<std::uint8_t> host(b.stride * experts, 0);
    for (int e = 0; e < experts; ++e) {
        std::memcpy(host.data() + b.stride * e + codes_at, mats[static_cast<std::size_t>(e)].codes.data(),
                    mats[static_cast<std::size_t>(e)].codes.size());
        if (!mats[static_cast<std::size_t>(e)].scales.empty()) {
            std::memcpy(host.data() + b.stride * e + scales_at, mats[static_cast<std::size_t>(e)].scales.data(),
                        mats[static_cast<std::size_t>(e)].scales.size());
        }
        b.experts.push_back(std::move(mats[static_cast<std::size_t>(e)].h));
    }
    b.records = DeviceBuffer(host.size());
    b.records.copy_from_host(host.data(), host.size(), 0);
    b.matrix.qtype         = q;
    b.matrix.codes_offset  = codes_at;
    b.matrix.scales_offset = scales_at;
    b.matrix.rows          = rows;
    b.matrix.k             = k;
    b.matrix.row_bytes     = mats[0].row_bytes;
    return b;
}

// lossless_linear with a design-reference activation scale: a divisor (0: none) and per-column multipliers
// (empty: none).
void run_scaled(const std::string& label, const HostWeight& w, const std::vector<std::uint16_t>& x, int columns, bool bf16,
                float divisor, const std::vector<float>& column_scales) {
    DeviceBuffer dx(x.size() * 2);
    dx.copy_from_host(x.data(), x.size() * 2, 0);
    DeviceBuffer dscales(std::max<std::size_t>(column_scales.size(), 1) * 4);
    if (!column_scales.empty()) { dscales.copy_from_host(column_scales.data(), column_scales.size() * 4, 0); }
    const int esize = bf16 ? 2 : 4;
    DeviceBuffer dout(static_cast<std::size_t>(w.n) * columns * esize);
    Tensor tx(dx.p, DType::BF16, {w.k, columns});
    Tensor tout(dout.p, bf16 ? DType::BF16 : DType::FP32, {w.n, columns});
    DeviceBuffer ws(ops::lossless_linear_workspace_bytes(w.weight, w.n, columns));
    WorkspaceArena arena(DeviceSpan{ws.p, ws.bytes});
    const ops::LosslessActivationScale scale{divisor,
                                             column_scales.empty() ? nullptr : static_cast<const float*>(dscales.p)};
    ops::lossless_linear(tx, w.weight, 0, w.n, tout, 0, arena, nullptr, scale);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint8_t> got(static_cast<std::size_t>(w.n) * columns * esize);
    dout.copy_to_host(got.data(), got.size(), 0);
    compare(label, w.n, columns, [&](int t, int r, std::uint32_t& g, std::uint32_t& want) {
        g = 0;
        std::memcpy(&g, got.data() + (static_cast<std::size_t>(t) * w.n + r) * esize, esize);
        ActScales act;
        act.divisor           = f32_bits(divisor);
        act.column_multiplier = column_scales.empty() ? 0U : f32_bits(column_scales[static_cast<std::size_t>(t)]);
        want = oracle(w, r, x.data() + static_cast<std::size_t>(t) * w.k, bf16, act);
        return true;
    });
}

// act: each expert's activation scales of even and odd rows (LosslessTile::act; null: none).
void run_tiles(const std::string& label, const HostBank& b, int tiles, int source_columns, bool bf16, std::mt19937& rng,
               const std::vector<std::array<std::uint32_t, 2>>* act = nullptr) {
    const int k = b.matrix.k, rows = b.matrix.rows, columns = 64 * tiles, out_columns = columns + 7;
    const auto x = make_columns(k, source_columns, rng, 4);
    std::vector<std::int32_t> in(static_cast<std::size_t>(columns)), out(static_cast<std::size_t>(columns));
    std::vector<std::int32_t> slots(static_cast<std::size_t>(out_columns));
    for (int i = 0; i < out_columns; ++i) { slots[static_cast<std::size_t>(i)] = i; }
    std::shuffle(slots.begin(), slots.end(), rng);
    std::uniform_int_distribution<int> source(0, source_columns - 1), expert(0, static_cast<int>(b.experts.size()) - 1);
    for (int c = 0; c < columns; ++c) {
        in[static_cast<std::size_t>(c)]  = c % 23 == 7 ? -1 : source(rng);
        out[static_cast<std::size_t>(c)] = c % 29 == 11 ? -1 : slots[static_cast<std::size_t>(c)];
    }
    std::vector<int> tile_expert(static_cast<std::size_t>(tiles));
    std::vector<ops::LosslessTile> host_tiles(static_cast<std::size_t>(tiles));
    for (int t = 0; t < tiles; ++t) {
        const int e = t % 5 == 3 ? -1 : expert(rng);
        tile_expert[static_cast<std::size_t>(t)] = e;
        host_tiles[static_cast<std::size_t>(t)].record =
            e < 0 ? nullptr : static_cast<const std::uint8_t*>(b.records.p) + b.stride * static_cast<std::size_t>(e);
        const auto a = e < 0 || b.alpha.empty() ? std::array<std::uint32_t, 2>{0U, 0U} : b.alpha[static_cast<std::size_t>(e)];
        host_tiles[static_cast<std::size_t>(t)].alpha[0] = a[0];
        host_tiles[static_cast<std::size_t>(t)].alpha[1] = a[1];
        const auto g = e < 0 || act == nullptr ? std::array<std::uint32_t, 2>{0U, 0U} : (*act)[static_cast<std::size_t>(e)];
        host_tiles[static_cast<std::size_t>(t)].act[0] = g[0];
        host_tiles[static_cast<std::size_t>(t)].act[1] = g[1];
    }
    DeviceBuffer dx(x.size() * 2), din(in.size() * 4), dout_map(out.size() * 4),
        dtiles(host_tiles.size() * sizeof(ops::LosslessTile));
    dx.copy_from_host(x.data(), x.size() * 2, 0);
    din.copy_from_host(in.data(), in.size() * 4, 0);
    dout_map.copy_from_host(out.data(), out.size() * 4, 0);
    dtiles.copy_from_host(host_tiles.data(), host_tiles.size() * sizeof(ops::LosslessTile), 0);
    const int esize = bf16 ? 2 : 4;
    DeviceBuffer dout(static_cast<std::size_t>(rows) * out_columns * esize);
    CUDA_CHECK(cudaMemset(dout.p, 0xCD, static_cast<std::size_t>(rows) * out_columns * esize));
    Tensor tx(dx.p, DType::BF16, {k, source_columns});
    Tensor tout(dout.p, bf16 ? DType::BF16 : DType::FP32, {rows, out_columns});
    DeviceBuffer ws(ops::lossless_linear_tiles_workspace_bytes(b.matrix, columns));
    WorkspaceArena arena(DeviceSpan{ws.p, ws.bytes});
    ops::lossless_linear_tiles(tx, static_cast<const std::int32_t*>(din.p), columns, b.matrix,
                               static_cast<const ops::LosslessTile*>(dtiles.p), tout,
                               static_cast<const std::int32_t*>(dout_map.p), arena, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint8_t> got(static_cast<std::size_t>(rows) * out_columns * esize);
    dout.copy_to_host(got.data(), got.size(), 0);
    const std::vector<std::uint16_t> zeros(static_cast<std::size_t>(k), 0);
    // Every output column written by some c (others must keep the fill).
    std::vector<int> writer(static_cast<std::size_t>(out_columns), -1);
    for (int c = 0; c < columns; ++c) {
        if (out[static_cast<std::size_t>(c)] >= 0 && tile_expert[static_cast<std::size_t>(c / 64)] >= 0) {
            writer[static_cast<std::size_t>(out[static_cast<std::size_t>(c)])] = c;
        }
    }
    const std::uint32_t fill = bf16 ? 0xCDCDU : 0xCDCDCDCDU;
    compare(label, rows, out_columns, [&](int oc, int r, std::uint32_t& g, std::uint32_t& want) {
        std::memcpy(&g, got.data() + (static_cast<std::size_t>(oc) * rows + r) * esize, esize);
        const int c = writer[static_cast<std::size_t>(oc)];
        if (c < 0) {
            want = fill;
            return true;
        }
        const int e = tile_expert[static_cast<std::size_t>(c / 64)];
        const int src = in[static_cast<std::size_t>(c)];
        ActScales scales;
        if (act != nullptr) { scales.row_multiplier = (*act)[static_cast<std::size_t>(e)][static_cast<std::size_t>(r & 1)]; }
        want = oracle(b.experts[static_cast<std::size_t>(e)], r,
                      src < 0 ? zeros.data() : x.data() + static_cast<std::size_t>(src) * k, bf16, scales);
        return true;
    });
}

// Projections of one input through one shared-encode call (ops::lossless_linear_shared), each against the
// oracle: row ranges into offset outputs, BF16 and FP32 outputs, the first projection's format not reading
// the half-block sums that a later one (Q4_K) reads, a BF16 weight's block scan, across the 512-column chunk.
void run_shared(const std::string& label, const std::vector<const HostWeight*>& weights, int columns, std::mt19937& rng) {
    const int k  = weights[0]->k;
    const auto x = make_columns(k, columns, rng, 8);
    DeviceBuffer dx(x.size() * 2);
    dx.copy_from_host(x.data(), x.size() * 2, 0);
    Tensor tx(dx.p, DType::BF16, {k, columns});
    struct Slot {
        std::unique_ptr<DeviceBuffer> buffer;
        Tensor tensor;
        int begin, rows, out_rows, out_row;
        bool bf16;
    };
    std::vector<Slot> slots(weights.size());
    std::vector<ops::LosslessProjection> projections;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const HostWeight& w = *weights[i];
        Slot& slot          = slots[i];
        slot.begin    = i % 2 == 1 ? std::min(7, w.n / 4) : 0;
        slot.rows     = w.n - slot.begin;
        slot.out_row  = static_cast<int>(3 * i);
        slot.out_rows = slot.rows + slot.out_row + 5;
        slot.bf16     = i % 2 == 0;
        const int esize = slot.bf16 ? 2 : 4;
        slot.buffer = std::make_unique<DeviceBuffer>(static_cast<std::size_t>(slot.out_rows) * columns * esize);
        CUDA_CHECK(cudaMemset(slot.buffer->p, 0xCD, static_cast<std::size_t>(slot.out_rows) * columns * esize));
        slot.tensor = Tensor(slot.buffer->p, slot.bf16 ? DType::BF16 : DType::FP32, {slot.out_rows, columns});
    }
    for (std::size_t i = 0; i < weights.size(); ++i) {
        projections.push_back({&weights[i]->weight, slots[i].begin, slots[i].rows, &slots[i].tensor, slots[i].out_row});
    }
    DeviceBuffer ws(ops::lossless_linear_shared_workspace_bytes(projections, columns));
    WorkspaceArena arena(DeviceSpan{ws.p, ws.bytes});
    ops::lossless_linear_shared(tx, projections, arena, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const HostWeight& w = *weights[i];
        const Slot& slot    = slots[i];
        const int esize     = slot.bf16 ? 2 : 4;
        std::vector<std::uint8_t> got(static_cast<std::size_t>(slot.out_rows) * columns * esize);
        slot.buffer->copy_to_host(got.data(), got.size(), 0);
        compare(label + " " + w.name, slot.rows, columns, [&](int t, int r, std::uint32_t& g, std::uint32_t& want) {
            const std::size_t j = static_cast<std::size_t>(t) * slot.out_rows + slot.out_row + r;
            g = 0;
            std::memcpy(&g, got.data() + j * esize, esize);
            want = oracle(w, slot.begin + r, x.data() + static_cast<std::size_t>(t) * w.k, slot.bf16);
            return true;
        });
    }
}

void suite(const HostWeight& w, std::mt19937& rng) {
    for (const int columns : {1, 9, 37}) {
        const auto x = make_columns(w.k, columns, rng, columns);
        run(w.name + " k=" + std::to_string(w.k) + " T=" + std::to_string(columns) + " bf16", w, x, columns, 0, w.n, w.n, 0,
            true);
        run(w.name + " k=" + std::to_string(w.k) + " T=" + std::to_string(columns) + " fp32", w, x, columns, 0, w.n, w.n, 0,
            false);
    }
    // A row range into an output offset.
    const int begin = std::min(37, w.n / 4), count = std::min(136, w.n - begin);
    const auto x = make_columns(w.k, 21, rng, 3);
    run(w.name + " rows " + std::to_string(begin) + ".." + std::to_string(begin + count) + " at 11", w, x, 21, begin,
        count, count + 24, 11, true);
    // Cancellation: results far below the terms (the fallback's wide sum and subnormals).
    const auto c = cancelling_columns(w.k, 17, rng);
    run(w.name + " cancelling bf16", w, c, 17, 0, w.n, w.n, 0, true);
    run(w.name + " cancelling fp32", w, c, 17, 0, w.n, w.n, 0, false);
}

} // namespace

INFERNIX_OP_TEST(lossless_linear) {
    try {
        std::mt19937 rng(20261021);
        const QType formats[] = {QType::Q4_G64_FP16, QType::Q5_G64_FP16, QType::Q6_G64_FP16, QType::Q8_G32_FP16,
                                 QType::Q2_G64_FP16};
        for (const QType q : formats) { suite(make_row_split(q, 200, 640, rng, false), rng); }
        suite(make_nvfp4(256, 640, rng, 2.75e-3F), rng);
        suite(make_nvfp4(128, 640, rng, 3.0F), rng);
        suite(make_fp8_row(200, 640, rng), rng);
        suite(make_fp8_row_f32(200, 640, rng), rng);
        suite(make_rg16_dense(192, 640, rng, 0.0123F), rng);
        suite(make_bf16(70, 640, rng, true), rng);
        suite(make_bf16(200, 640, rng, false), rng);
        // Real projection depths, wide scale exponents (terms far below the largest), across the
        // 512-column chunk.
        {
            const HostWeight w = make_row_split(QType::Q4_G64_FP16, 96, 5120, rng, true);
            const auto x = make_columns(w.k, 600, rng, 0);
            run("q4 k=5120 T=600 wide scales bf16", w, x, 600, 0, w.n, w.n, 0, true);
        }
        {
            const HostWeight w = make_row_split(QType::Q5_G64_FP16, 64, 17408, rng, false);
            const auto x = make_columns(w.k, 13, rng, 5);
            run("q5 k=17408 T=13 fp32", w, x, 13, 0, w.n, w.n, 0, false);
        }
        {
            const HostWeight w = make_nvfp4(128, 17408, rng, 1.5e-4F);
            const auto x = make_columns(w.k, 70, rng, 1);
            run("nvfp4 k=17408 T=70 bf16", w, x, 70, 0, w.n, w.n, 0, true);
        }
        {
            const HostWeight w = make_fp8_row(96, 6144, rng);
            const auto x = make_columns(w.k, 530, rng, 4);
            run("fp8row k=6144 T=530 bf16", w, x, 530, 0, w.n, w.n, 0, true);
        }
        {
            const HostWeight w = make_bf16(48, 5120, rng, true);
            const auto x = make_columns(w.k, 40, rng, 6);
            run("bf16 spread k=5120 T=40 fp32", w, x, 40, 0, w.n, w.n, 0, false);
        }
        {
            const HostWeight w = make_bf16(160, 4096, rng, false);
            const auto x = make_columns(w.k, 530, rng, 7);
            run("bf16 grid k=4096 T=530 bf16", w, x, 530, 0, w.n, w.n, 0, true);
            run("bf16 grid k=4096 rows 70..150 T=530 fp32", w, x, 530, 70, 80, 90, 5, false);
        }
        // Qwen3.5-122B-A10B's depths: its FP8 projections of the hidden state (K 3072) and of the attention
        // output (K 8192), its NVFP4 head (K 3072) with a multiplier far from one.
        {
            const HostWeight w = make_fp8_row_f32(96, 3072, rng);
            const auto x = make_columns(w.k, 530, rng, 4);
            run("fp8row f32 k=3072 T=530 bf16", w, x, 530, 0, w.n, w.n, 0, true);
        }
        {
            const HostWeight w = make_fp8_row_f32(64, 8192, rng);
            const auto x = make_columns(w.k, 13, rng, 5);
            run("fp8row f32 k=8192 T=13 fp32", w, x, 13, 0, w.n, w.n, 0, false);
        }
        {
            const HostWeight w = make_rg16_dense(256, 3072, rng, std::ldexp(1.613F, -11));
            const auto x = make_columns(w.k, 70, rng, 1);
            run("nvfp4 rg16 dense k=3072 T=70 fp32", w, x, 70, 0, w.n, w.n, 0, false);
            run("nvfp4 rg16 dense k=3072 rows 48..208 T=70 bf16", w, x, 70, 48, 160, 170, 7, true);
        }
        // The noise twin: BF16 and FP32 outputs, a BF16 weight and an NVFP4 one (its divisor's sticky tails).
        {
            const HostWeight w = make_bf16(160, 4096, rng, false);
            const auto x = make_columns(w.k, 530, rng, 7);
            run_noise("noise twin bf16 k=4096 T=530 bf16", w, x, 530, true);
            run_noise("noise twin bf16 k=4096 T=530 fp32", w, x, 530, false);
        }
        {
            const HostWeight w = make_nvfp4(128, 17408, rng, 1.5e-4F);
            const auto x = make_columns(w.k, 70, rng, 1);
            run_noise("noise twin nvfp4 k=17408 T=70 bf16", w, x, 70, true);
        }
        // MXFP8 (FP4-Local's dense projections) and GGUF blocks (the GGUF imports' dense projections).
        suite(make_mxfp8(200, 640, rng), rng);
        {
            const struct {
                QType q;
                int k;
                const char* name;
            } ggml_formats[] = {{QType::GGML_Q4_K, 512, "q4_k"},     {QType::GGML_Q5_K, 512, "q5_k"},
                                {QType::GGML_Q6_K, 512, "q6_k"},     {QType::GGML_IQ4_XS, 512, "iq4_xs"},
                                {QType::GGML_IQ4_NL, 640, "iq4_nl"}, {QType::GGML_Q2_0, 640, "q2_0"},
                                {QType::GGML_IQ2_S, 512, "iq2_s"},   {QType::GGML_IQ3_XXS, 512, "iq3_xxs"},
                                {QType::GGML_IQ3_S, 512, "iq3_s"}};
            for (const auto& f : ggml_formats) {
                suite(make_ggml(f.q, 72, f.k, rng, false, f.name), rng);
                const HostWeight t = make_ggml(f.q, 72, f.k, rng, true, f.name);
                const auto x = make_columns(t.k, 37, rng, 1);
                run(t.name + " T=37 bf16", t, x, 37, 0, t.n, t.n, 0, true);
                run(t.name + " rows 16..64 T=37 fp32", t, x, 37, 16, 48, 60, 3, false);
            }
            const HostWeight q6 = make_ggml(QType::GGML_Q6_K, 64, 2560, rng, true, "q6_k");
            const auto x = make_columns(q6.k, 530, rng, 2);
            run("q6_k tiles16 k=2560 T=530 bf16", q6, x, 530, 0, q6.n, q6.n, 0, true);
        }
        // The tile mode: the routed experts' records (Flash-Next's NVFP4, official FP8 and GGUF banks).
        {
            HostBank rg = make_bank("nvfp4 rg16 bank", QType::NVFP4_MUL, 64, 640, 4, 0, [&](int e) {
                const std::uint32_t alpha[2] = {f32_bits(std::ldexp(1.37F, -7 - e)), f32_bits(std::ldexp(1.91F, 3 * e - 2))};
                return gen_rg16(64, 640, rng, alpha);
            });
            for (int e = 0; e < 4; ++e) {
                rg.alpha.push_back({f32_bits(std::ldexp(1.37F, -7 - e)), f32_bits(std::ldexp(1.91F, 3 * e - 2))});
            }
            run_tiles("nvfp4 rg16 bank tiles bf16", rg, 9, 40, true, rng);
            run_tiles("nvfp4 rg16 bank tiles fp32", rg, 3, 70, false, rng);
            // The design reference's expert input scales g (gate rows even, up rows odd), exactly with the
            // weights' multipliers.
            std::vector<std::array<std::uint32_t, 2>> g;
            for (int e = 0; e < 4; ++e) {
                g.push_back({f32_bits(std::ldexp(1.0F + 0.29F * e, -9 + e)), f32_bits(std::ldexp(1.71F, -3 - 2 * e))});
            }
            run_tiles("nvfp4 rg16 bank tiles with act scales bf16", rg, 9, 40, true, rng, &g);
            run_tiles("nvfp4 rg16 bank tiles with act scales fp32", rg, 3, 70, false, rng, &g);
            HostBank fp8 = make_bank("fp8 b128 bank", QType::FP8_E4M3FN_B128, 128, 640, 3, 4096,
                                     [&](int) { return gen_fp8b128(128, 640, rng); });
            run_tiles("fp8 b128 bank tiles bf16", fp8, 9, 40, true, rng);
            HostBank iq3 = make_bank("iq3_s bank", QType::GGML_IQ3_S, 48, 512, 3, 1024,
                                     [&](int) { return gen_ggml(QType::GGML_IQ3_S, 48, 512, rng, false); });
            run_tiles("iq3_s bank tiles bf16", iq3, 9, 40, true, rng);
            HostBank q4k = make_bank("q4_k bank", QType::GGML_Q4_K, 48, 512, 3, 0,
                                     [&](int) { return gen_ggml(QType::GGML_Q4_K, 48, 512, rng, false); });
            run_tiles("q4_k bank tiles fp32", q4k, 6, 40, false, rng);
            HostBank nl = make_bank("iq4_nl bank", QType::GGML_IQ4_NL, 64, 640, 3, 0,
                                    [&](int) { return gen_ggml(QType::GGML_IQ4_NL, 64, 640, rng, false); });
            run_tiles("iq4_nl bank tiles bf16", nl, 6, 40, true, rng);
        }
        // The design reference's activation scales (LosslessActivationScale): an NVFP4 activation divisor with
        // the weight's own divisor (two exact divisions), per-column multipliers with an FP8 row scale, both at
        // once, and a divisor on a weight without a factor.
        {
            const HostWeight n4 = make_nvfp4(128, 640, rng, 2.75e-3F);
            const HostWeight f8 = make_fp8_row(96, 640, rng);
            const HostWeight q4 = make_row_split(QType::Q4_G64_FP16, 64, 640, rng, false);
            const auto x = make_columns(640, 37, rng, 2);
            std::vector<float> scales(37);
            for (int t = 0; t < 37; ++t) {
                scales[static_cast<std::size_t>(t)] = std::ldexp(1.0F + 0.37F * static_cast<float>(t % 5), (t % 11) * 7 - 40);
            }
            run_scaled("nvfp4 activation divisor bf16", n4, x, 37, true, 26.953125F, {});
            run_scaled("nvfp4 activation divisor fp32", n4, x, 37, false, 2741.5F, {});
            run_scaled("fp8row column scales bf16", f8, x, 37, true, 0.0F, scales);
            // Qwen3.5-122B-A10B's static A8 inputs: one static activation scale for every column.
            const HostWeight f8f32 = make_fp8_row_f32(96, 640, rng);
            run_scaled("fp8row f32 static column scale fp32", f8f32, x, 37, false, 0.0F,
                       std::vector<float>(37, 0.0217285F));
            run_scaled("fp8row f32 column scales bf16", f8f32, x, 37, true, 0.0F, scales);
            run_scaled("nvfp4 divisor and column scales fp32", n4, x, 37, false, 0.0371F, scales);
            run_scaled("q4 activation divisor fp32", q4, x, 37, false, 3.0e-3F, {});
        }
        // The grouped form (routed experts): Q4 gate/up banks and the Q5 down bank's depth.
        // (the 35B's layout: expert e's gate at e 1024, its up at e 1024 + 512; down at e 2048).
        run_grouped("q4 grouped up k=640", make_row_split(QType::Q4_G64_FP16, 1024, 640, rng, false), 256, 128, 128, 11,
                    rng);
        run_grouped("q5 grouped k=512", make_row_split(QType::Q5_G64_FP16, 1024, 512, rng, false), 192, 0, 64, 9, rng);
        // Projections sharing one encode.
        {
            const HostWeight a = make_row_split(QType::Q4_G64_FP16, 96, 512, rng, false);
            const HostWeight b = make_ggml(QType::GGML_Q4_K, 72, 512, rng, false, "q4_k");
            const HostWeight c = make_bf16(64, 512, rng, true);
            const HostWeight d = make_ggml(QType::GGML_Q6_K, 72, 512, rng, true, "q6_k tiles16");
            for (const int columns : {1, 600}) { run_shared("shared T=" + std::to_string(columns), {&a, &b, &c, &d}, columns, rng); }
        }
        // Qwen3.5-122B-A10B's GDN input: the FP8 projection and the BF16 control (a, b) of one hidden state.
        {
            const HostWeight f = make_fp8_row_f32(128, 3072, rng);
            const HostWeight c = make_bf16(64, 3072, rng, false);
            run_shared("shared fp8 f32 + bf16 k=3072 T=37", {&f, &c}, 37, rng);
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL exception: " << e.what() << "\n";
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "lossless_linear: every output equals the correctly rounded exact sum\n";
    return 0;
}
