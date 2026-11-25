// PLE n-gram embedding injection (include/infernix/ops/ple.h) against independent oracles:
//   - ple_embed_bf16_p12: exact. The row's base-plus-offset exponent and its sign-and-mantissa byte
//     reproduce each BF16 word bit for bit, escapes included; the row bytes are decoded here.
//   - ple_embed: exact. An E4M3 code times a BF16 scale is exact in FP32 (4 x 8 significant bits),
//     so each output must equal the FP64 product rounded once to BF16. The E4M3 decoding here is
//     independent of the device conversion; NaN codes are not represented inputs.
//   - ple_gate: the FP64 formula (unit-offset RMSNorm per stream, signed-sqrt gate, sigmoid,
//     conv-input RMSNorm). Each output is rounded once from an FP32 evaluation, so it lies within
//     one BF16 ULP of the FP64 value (relative 2^-7, with an absolute floor for values near 0).
//   - ple_conv_inject: the residual update R += gated + SiLU(causal dilated depthwise conv of the
//     conv inputs over the sequence's history), against FP64 with the same one-rounding bound on
//     the updated residual, and the state shift exactly: the history becomes the trailing span
//     columns of (old history, this call's conv inputs), in place or into another slot; an empty
//     destination leaves every state unchanged; untouched slots and guards stay intact.

#include "infernix/ngram_blockdir.h"
#include "infernix/ops/ple.h"
#include "ops/op_family.h"

#include "core/arena.h"
#include "ops/op_check.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

int g_failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

// FP8 E4M3FN: 1 sign, 4 exponent (bias 7), 3 mantissa bits; exponent 0 is subnormal; S.1111.111 is
// NaN (no infinities).
double decode_e4m3(std::uint8_t code) {
    const int sign     = code >> 7;
    const int exponent = (code >> 3) & 0xF;
    const int mantissa = code & 0x7;
    const double magnitude = exponent == 0 ? std::ldexp(mantissa / 8.0, -6)
                                           : std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
    return sign != 0 ? -magnitude : magnitude;
}

bool e4m3_nan(std::uint8_t code) { return (code & 0x7F) == 0x7F; }

std::uint16_t bf16_of(double value) { return f32_to_bf16(static_cast<float>(value)); }

double bf16_value(std::uint16_t bits) { return static_cast<double>(bf16_to_f32(bits)); }

std::vector<std::uint16_t> random_bf16(std::size_t count, std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<std::uint16_t> out(count);
    for (auto& v : out) { v = f32_to_bf16(dist(rng)); }
    return out;
}

// One BF16 ULP of the FP64 value (2^-7 relative at worst), with an absolute floor near zero.
constexpr PointwiseCriterion kOneRounding{.absolute = 1.0e-6, .relative = 1.0 / 128.0};

void test_embed(int width, int heads, int columns, std::uint32_t seed) {
    const std::string label = "ple_embed W=" + std::to_string(width) + " heads=" + std::to_string(heads) +
                              " T=" + std::to_string(columns);
    std::mt19937 rng(seed);
    const std::size_t elements = static_cast<std::size_t>(width) * heads;
    std::vector<std::uint8_t> rows(elements * columns);
    // Every non-NaN code appears (including both zeros and the subnormals), then random ones.
    std::size_t next = 0;
    for (int code = 0; code < 256 && next < rows.size(); ++code) {
        if (!e4m3_nan(static_cast<std::uint8_t>(code))) { rows[next++] = static_cast<std::uint8_t>(code); }
    }
    std::uniform_int_distribution<int> any_code(0, 255);
    for (; next < rows.size(); ++next) {
        std::uint8_t code = 0x7F;
        while (e4m3_nan(code)) { code = static_cast<std::uint8_t>(any_code(rng)); }
        rows[next] = code;
    }
    for (const float scale_value : {0.0234375F, -1.75F, 3.0e-3F}) {
        const std::uint16_t scale = f32_to_bf16(scale_value);
        DeviceBuffer d_rows  = to_device(rows);
        DeviceBuffer d_scale = to_device(std::vector<std::uint16_t>{scale});
        GuardedDeviceBuffer d_out(elements * columns * sizeof(std::uint16_t));
        d_out.fill(0xFF);
        Tensor t_rows(d_rows.p, DType::U8, {width, heads, columns});
        Tensor t_scale(d_scale.p, DType::BF16, {1});
        Tensor t_out(d_out.data(), DType::BF16, {static_cast<std::int32_t>(elements), columns});
        ops::ple_embed(t_rows, t_scale, t_out, nullptr);
        cuda_synchronize();
        const auto got = from_device<std::uint16_t>(d_out.data(), elements * columns);
        std::vector<std::uint16_t> want(elements * columns);
        // out[h * W + j, t] = bf16(e4m3(rows[j, h, t]) * scale): rows are [W, heads, T] with W fastest,
        // so the flat index is the same on both sides.
        for (std::size_t i = 0; i < want.size(); ++i) { want[i] = bf16_of(decode_e4m3(rows[i]) * bf16_value(scale)); }
        g_failures += verify_exact((label + " scale " + std::to_string(scale_value)).c_str(), got, want);
        g_failures += d_out.verify_guards(label);
    }
}

// The BF16 P12 row's bytes for `words` at `base`: W sign-and-mantissa bytes, W/2 exponent-offset
// nibbles, the base and the escape count, then each escaped value's (column, BF16 bits) in column
// order. The test packs the rows from known BF16 words, and those words are the oracle: the codec is
// lossless, so the kernel must return each one bit for bit.
void pack_p12_row(const std::uint16_t* words, int width, int base, std::vector<std::uint8_t>& row) {
    row.assign(static_cast<std::size_t>(width) + width / 2 + 16, 0);
    const int base_at = width + width / 2;
    row[base_at]      = static_cast<std::uint8_t>(base);
    unsigned escapes  = 0;
    for (int j = 0; j < width; ++j) {
        const std::uint16_t bits = words[j];
        const int q              = static_cast<int>((bits >> 7) & 0xFFU) - base;
        row[j]                   = static_cast<std::uint8_t>(((bits >> 8) & 0x80U) | (bits & 0x7FU));
        if (q < 0 || q > 14) {
            if (escapes == 4) { throw std::invalid_argument("a BF16 P12 row holds at most 4 escapes"); }
            std::uint8_t* escape = row.data() + base_at + 2 + 3 * escapes;
            escape[0]            = static_cast<std::uint8_t>(j);
            escape[1]            = static_cast<std::uint8_t>(bits & 0xFFU);
            escape[2]            = static_cast<std::uint8_t>(bits >> 8);
            ++escapes;
        }
        row[width + j / 2] |= static_cast<std::uint8_t>((q < 0 || q > 14 ? 15 : q) << (4 * (j & 1)));
    }
    row[base_at + 1] = static_cast<std::uint8_t>(escapes);
}

// ple_embed_bf16_p12 against the packed words: zeros, negative zero, subnormals, an in-window row, rows
// with one and four escapes, a row whose sign and every mantissa bit must survive, and random rows.
void test_embed_bf16_p12(int width, int heads, int columns, std::uint32_t seed) {
    constexpr int kSpecialRows = 7; // before the random fill: zero, -0, subnormal, window, 1/4 escapes, sign
    if (heads * columns < kSpecialRows) {
        throw std::invalid_argument("ple_embed_bf16_p12 needs at least " + std::to_string(kSpecialRows) + " rows");
    }
    const std::string label = "ple_embed_bf16_p12 W=" + std::to_string(width) + " heads=" + std::to_string(heads) +
                              " T=" + std::to_string(columns);
    const int row_bytes = width + width / 2 + 16;
    std::vector<std::uint8_t> rows(static_cast<std::size_t>(row_bytes) * heads * columns, 0);
    std::mt19937 rng(seed);
    std::vector<std::uint16_t> words(width);
    const std::size_t elements = static_cast<std::size_t>(width) * heads;
    std::vector<std::uint16_t> want(elements * columns); // out[h * W + j, t]: row (t, h)'s word j
    int filled = 0;
    const auto put = [&](int base) {
        const int t = filled / heads, h = filled % heads;
        ++filled;
        std::vector<std::uint8_t> row;
        pack_p12_row(words.data(), width, base, row);
        std::memcpy(rows.data() + static_cast<std::size_t>(t * heads + h) * row_bytes, row.data(), row_bytes);
        std::copy(words.begin(), words.end(), want.begin() + static_cast<std::ptrdiff_t>(t * elements + h * width));
    };
    std::fill(words.begin(), words.end(), std::uint16_t{0}); // all zeros, base 0
    put(0);
    std::fill(words.begin(), words.end(), std::uint16_t{0x8000}); // negative zero
    put(0);
    for (int j = 0; j < width; ++j) { // subnormals (exponent 0): the zero exponent is in window at base 0
        words[j] = static_cast<std::uint16_t>(((j & 1) << 15) | (1 + j % 127));
    }
    put(0);
    for (int j = 0; j < width; ++j) { // every offset 0..14 in window
        words[j] = static_cast<std::uint16_t>(((j & 1) << 15) | ((j % 15) << 7) | (j % 128));
    }
    put(0);
    for (int j = 0; j < width; ++j) { words[j] = static_cast<std::uint16_t>((17 + j % 15) << 7 | (j % 128)); }
    words[width / 2] = static_cast<std::uint16_t>((200 << 7) | 0x55); // one escape
    put(17);
    for (int j = 0; j < width; ++j) { words[j] = static_cast<std::uint16_t>((40 << 7) | (j % 128)); }
    words[0]         = static_cast<std::uint16_t>((0 << 7) | 1);    // four escapes
    words[1]         = static_cast<std::uint16_t>((1 << 7) | 2);
    words[2]         = static_cast<std::uint16_t>((220 << 7) | 3);
    words[3]         = static_cast<std::uint16_t>((255 << 7) | 4);
    put(33);
    for (int j = 0; j < width; ++j) { // sign and every mantissa bit must survive
        words[j] = static_cast<std::uint16_t>(((j & 1) << 15) | (10 << 7) | (j % 128));
    }
    put(5);
    std::uniform_int_distribution<int> base_dist(0, 240), offset_dist(0, 14);
    std::uniform_int_distribution<int> sign_dist(0, 1), mantissa_dist(0, 127);
    while (filled < heads * columns) {
        const int base = base_dist(rng);
        for (int j = 0; j < width; ++j) {
            words[j] = static_cast<std::uint16_t>((sign_dist(rng) << 15) | ((base + offset_dist(rng)) << 7) |
                                                  mantissa_dist(rng));
        }
        put(base);
    }
    expect(filled == heads * columns, label + " fills every row");

    DeviceBuffer d_rows = to_device(rows);
    GuardedDeviceBuffer d_out(elements * columns * sizeof(std::uint16_t));
    d_out.fill(0xFF);
    Tensor t_rows(d_rows.p, DType::U8, {row_bytes, heads, columns});
    Tensor t_out(d_out.data(), DType::BF16, {static_cast<std::int32_t>(elements), columns});
    ops::ple_embed_bf16_p12(t_rows, t_out, nullptr);
    cuda_synchronize();

    g_failures += verify_exact(label.c_str(), from_device<std::uint16_t>(d_out.data(), want.size()), want);
    g_failures += d_out.verify_guards(label);
    if (width == 160 && heads == 16 && columns == 3) {
        // A blockdir volume's reader hands the Op the same canonical P12 row, so the op must decode
        // the same rows to the same BF16 words.
        GuardedDeviceBuffer d_blockdir(static_cast<std::size_t>(elements) * columns * sizeof(std::uint16_t));
        d_blockdir.fill(0xFF);
        Tensor t_blockdir(d_blockdir.data(), DType::BF16, {static_cast<std::int32_t>(elements), columns});
        ops::ple_embed_bf16_p12_blockdir(t_rows, t_blockdir, nullptr);
        cuda_synchronize();
        g_failures += verify_exact((label + " blockdir").c_str(),
                                   from_device<std::uint16_t>(d_blockdir.data(), want.size()), want);
        g_failures += d_blockdir.verify_guards(label + " blockdir");
    }
}

// The BF16 P12 blockdir rows a resident table stages: the block's 240 B data planes, its 16-byte
// directory and the row's index at byte 256 (the rest zero). The fixture's values are known exactly:
// every value's exponent is its row's base + j % 15 (sign 0, mantissa 0x11), except the escapes, whose
// exponents the directory's slots carry. The op must reproduce those words, and agree with
// ple_embed_bf16_p12 on the canonical rows built from the same values.
void test_embed_bf16_p12_blockdir_packed() {
    constexpr int kWidth = 160, kRowBytes = 272, kCanonical = 256;
    constexpr int kHeads = 18, kColumns = 2; // 17 block rows and one side-stream (canonical) row
    const std::array<std::uint8_t, 8> bases = {53, 54, 105, 106, 107, 108, 109, 200};
    const std::string label = "ple_embed_bf16_p12_blockdir_packed";
    // Row -> its escapes (column, exponent): row 1 holds two of the block's four slots, so the
    // fixture covers every slot and a row with more than one escape.
    std::vector<std::vector<std::pair<int, int>>> escapes(static_cast<std::size_t>(kHeads));
    escapes[1] = {{7, 73}, {9, 79}};
    escapes[2] = {{3, 131}};
    escapes[3] = {{5, 135}};
    escapes[kHeads - 1] = {{4, 77}}; // the side-stream row
    std::vector<unsigned> codes(static_cast<std::size_t>(kHeads));
    for (int row = 0; row < kHeads; ++row) { codes[row] = static_cast<unsigned>(row) % 8; }
    // A row's 240 data bytes: every value's exponent is its row's base + j % 15 (sign 0, mantissa
    // 0x11), except its escaped columns, which take q = 15 and their exact exponent.
    const auto make_plane = [&](int row, std::uint8_t* out) {
        std::memset(out, 0, 240);
        for (int j = 0; j < kWidth; ++j) {
            unsigned q = static_cast<unsigned>(j % 15);
            for (const auto& escape : escapes[row]) {
                if (escape.first == j) { q = 15; }
            }
            out[j] = 0x11;
            out[160 + j / 2] |= static_cast<std::uint8_t>(q << (4 * (j & 1)));
        }
    };
    const auto plane_value = [&](int row, int j) {
        unsigned exponent = static_cast<unsigned>(bases[codes[row]] + j % 15);
        for (const auto& escape : escapes[row]) {
            if (escape.first == j) { exponent = static_cast<unsigned>(escape.second); }
        }
        return static_cast<std::uint16_t>((exponent << 7) | 0x11);
    };
    const auto set_bits = [](std::uint8_t* at, unsigned offset, unsigned count, std::uint32_t value) {
        for (unsigned k = 0; k < count; ++k) {
            if ((value >> k) & 1U) { at[(offset + k) / 8] |= static_cast<std::uint8_t>(1U << ((offset + k) % 8)); }
        }
    };
    std::vector<std::uint8_t> block(4096, 0);
    for (int row = 0; row < 17; ++row) { make_plane(row, block.data() + row * 240); }
    std::uint8_t* directory = block.data() + 17 * 240;
    for (int row = 0; row < 17; ++row) { set_bits(directory, 3 * row, 3, codes[row]); }
    unsigned slots = 0;
    for (int row = 0; row < 17 && slots < 4; ++row) {
        for (const auto& escape : escapes[row]) {
            if (slots == 4) { break; }
            const unsigned position = static_cast<unsigned>(row) * 160U + static_cast<unsigned>(escape.first);
            const unsigned delta = static_cast<unsigned>((escape.second - static_cast<int>(bases[codes[row]])) & 0x3F);
            set_bits(directory, 3 * 17 + 3 + 18 * slots, 18, position | (delta << 12));
            ++slots;
        }
    }
    set_bits(directory, 3 * 17, 3, slots);

    std::vector<std::uint8_t> rows(static_cast<std::size_t>(kHeads) * kColumns * kRowBytes, 0);
    std::vector<std::uint8_t> canonical(static_cast<std::size_t>(kHeads) * kColumns * kCanonical, 0);
    std::vector<std::uint16_t> want(static_cast<std::size_t>(kHeads) * kColumns * kWidth, 0);
    for (int column = 0; column < kColumns; ++column) {
        for (int row = 0; row < kHeads; ++row) {
            std::uint8_t* staged = rows.data() + (static_cast<std::size_t>(column) * kHeads + row) * kRowBytes;
            std::uint8_t* plain = canonical.data() + (static_cast<std::size_t>(column) * kHeads + row) * kCanonical;
            if (row < 17) {
                std::memcpy(staged, block.data() + row * 240, 240);
                std::memcpy(staged + 240, directory, 16);
            } else {
                make_plane(row, staged);
                std::memcpy(staged + 240, directory, 16); // overwritten by the canonical trailer below
            }
            staged[256] = static_cast<std::uint8_t>(row);
            std::memcpy(plain, staged, 240);
            plain[240] = bases[codes[row]];
            plain[241] = static_cast<std::uint8_t>(escapes[row].size());
            for (unsigned e = 0; e < escapes[row].size(); ++e) {
                plain[242 + 3 * e] = static_cast<std::uint8_t>(escapes[row][e].first);
                const auto bits = static_cast<std::uint16_t>((escapes[row][e].second << 7) | 0x11);
                plain[243 + 3 * e] = static_cast<std::uint8_t>(bits & 0xFF);
                plain[244 + 3 * e] = static_cast<std::uint8_t>(bits >> 8);
            }
            for (int j = 0; j < kWidth; ++j) {
                want[(static_cast<std::size_t>(column) * kHeads + row) * kWidth + j] = plane_value(row, j);
            }
        }
        // The last row stands for a block whose escapes went to the file's side stream: the reader
        // staged its canonical row instead, marked at byte 256.
        std::uint8_t* staged = rows.data() + (static_cast<std::size_t>(column) * kHeads + kHeads - 1) * kRowBytes;
        std::memcpy(staged + 240,
                    canonical.data() + (static_cast<std::size_t>(column) * kHeads + kHeads - 1) * kCanonical + 240,
                    16);
        staged[256] = 0xFF;
    }

    DeviceBuffer d_rows = to_device(rows);
    GuardedDeviceBuffer d_packed(static_cast<std::size_t>(kHeads) * kWidth * kColumns * sizeof(std::uint16_t));
    d_packed.fill(0xFF);
    Tensor t_rows(d_rows.p, DType::U8, {kRowBytes, kHeads, kColumns});
    Tensor t_packed(d_packed.data(), DType::BF16, {kHeads * kWidth, kColumns});
    ops::ple_embed_bf16_p12_blockdir_packed(t_rows, bases, t_packed, nullptr);
    cuda_synchronize();
    g_failures += verify_exact(label.c_str(), from_device<std::uint16_t>(d_packed.data(), want.size()), want);
    g_failures += d_packed.verify_guards(label);

    // The same values as canonical P12 rows: the shipped kernel must give the same words.
    DeviceBuffer d_canonical = to_device(canonical);
    GuardedDeviceBuffer d_reference(static_cast<std::size_t>(kHeads) * kWidth * kColumns * sizeof(std::uint16_t));
    d_reference.fill(0xFF);
    Tensor t_canonical(d_canonical.p, DType::U8, {kCanonical, kHeads, kColumns});
    Tensor t_reference(d_reference.data(), DType::BF16, {kHeads * kWidth, kColumns});
    ops::ple_embed_bf16_p12(t_canonical, t_reference, nullptr);
    cuda_synchronize();
    g_failures += verify_exact((label + " vs canonical").c_str(),
                               from_device<std::uint16_t>(d_reference.data(), want.size()), want);
    g_failures += d_reference.verify_guards(label + " vs canonical");
}

struct GateReference {
    std::vector<double> gated, normalized;
};

GateReference gate_reference(const std::vector<std::uint16_t>& key, const std::vector<std::uint16_t>& value,
                             const std::vector<std::uint16_t>& residual, const std::vector<std::uint16_t>& key_norm,
                             const std::vector<std::uint16_t>& query_norm, const std::vector<std::uint16_t>& conv_norm,
                             int streams, int hidden, int columns, double eps) {
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    GateReference out{std::vector<double>(width * columns), std::vector<double>(width * columns)};
    for (int t = 0; t < columns; ++t) {
        for (int s = 0; s < streams; ++s) {
            const std::size_t base = static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden;
            double key_sq = 0, query_sq = 0;
            for (int d = 0; d < hidden; ++d) {
                key_sq += bf16_value(key[base + d]) * bf16_value(key[base + d]);
                query_sq += bf16_value(residual[base + d]) * bf16_value(residual[base + d]);
            }
            const double key_inv   = 1.0 / std::sqrt(key_sq / hidden + eps);
            const double query_inv = 1.0 / std::sqrt(query_sq / hidden + eps);
            double dot = 0;
            for (int d = 0; d < hidden; ++d) {
                const std::size_t c = static_cast<std::size_t>(s) * hidden + d;
                dot += bf16_value(key[base + d]) * key_inv * (1.0 + bf16_value(key_norm[c])) *
                       bf16_value(residual[base + d]) * query_inv * (1.0 + bf16_value(query_norm[c]));
            }
            double g = dot / std::sqrt(static_cast<double>(hidden));
            g        = std::copysign(std::sqrt(std::max(std::abs(g), 1.0e-6)), g);
            const double gate = 1.0 / (1.0 + std::exp(-g));
            double gated_sq   = 0;
            for (int d = 0; d < hidden; ++d) {
                const double v = gate * bf16_value(value[static_cast<std::size_t>(t) * hidden + d]);
                out.gated[base + d] = v;
                gated_sq += v * v;
            }
            const double gated_inv = 1.0 / std::sqrt(gated_sq / hidden + eps);
            for (int d = 0; d < hidden; ++d) {
                const std::size_t c = static_cast<std::size_t>(s) * hidden + d;
                out.normalized[base + d] = out.gated[base + d] * gated_inv * (1.0 + bf16_value(conv_norm[c]));
            }
        }
    }
    return out;
}

void test_gate(int streams, int hidden, int columns, std::uint32_t seed) {
    const std::string label = "ple_gate S=" + std::to_string(streams) + " H=" + std::to_string(hidden) +
                              " T=" + std::to_string(columns);
    std::mt19937 rng(seed);
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    const auto key          = random_bf16(width * columns, rng, -2.0F, 2.0F);
    const auto value        = random_bf16(static_cast<std::size_t>(hidden) * columns, rng, -1.5F, 1.5F);
    auto residual           = random_bf16(width * columns, rng, -3.0F, 3.0F);
    // One column whose stream correlates with its key: a large gate on top of the near-zero ones.
    for (int d = 0; d < hidden; ++d) { residual[d] = key[d]; }
    const auto key_norm   = random_bf16(width, rng, -0.5F, 0.5F);
    const auto query_norm = random_bf16(width, rng, -0.5F, 0.5F);
    const auto conv_norm  = random_bf16(width, rng, -0.5F, 0.5F);
    const float eps       = 1.0e-6F;

    DeviceBuffer d_key = to_device(key), d_value = to_device(value), d_residual = to_device(residual);
    DeviceBuffer d_key_norm = to_device(key_norm), d_query_norm = to_device(query_norm),
                 d_conv_norm = to_device(conv_norm);
    GuardedDeviceBuffer d_gated(width * columns * 2), d_normalized(width * columns * 2);
    d_gated.fill(0xFF);
    d_normalized.fill(0xFF);
    const auto W = static_cast<std::int32_t>(width);
    Tensor t_key(d_key.p, DType::BF16, {W, columns});
    Tensor t_value(d_value.p, DType::BF16, {hidden, columns});
    Tensor t_residual(d_residual.p, DType::BF16, {W, columns});
    Tensor t_key_norm(d_key_norm.p, DType::BF16, {W});
    Tensor t_query_norm(d_query_norm.p, DType::BF16, {W});
    Tensor t_conv_norm(d_conv_norm.p, DType::BF16, {W});
    Tensor t_gated(d_gated.data(), DType::BF16, {W, columns});
    Tensor t_normalized(d_normalized.data(), DType::BF16, {W, columns});
    ops::ple_gate(t_key, t_value, t_residual, t_key_norm, t_query_norm, t_conv_norm, streams, eps, t_gated,
                  t_normalized, nullptr);
    cuda_synchronize();

    const GateReference ref =
        gate_reference(key, value, residual, key_norm, query_norm, conv_norm, streams, hidden, columns, eps);
    const auto gated      = from_device_bf16(d_gated.data(), width * columns);
    const auto normalized = from_device_bf16(d_normalized.data(), width * columns);
    g_failures += verify_pointwise(label + " gated", gated, ref.gated, kOneRounding);
    g_failures += verify_pointwise(label + " normalized", normalized, ref.normalized, kOneRounding);
    g_failures += d_gated.verify_guards(label + " gated");
    g_failures += d_normalized.verify_guards(label + " normalized");
    // The inputs are read only.
    g_failures += verify_exact((label + " residual preserved").c_str(),
                               from_device<std::uint16_t>(d_residual, residual.size()), residual);
}

// sequences x width columns over `slots` state slots; sequence i reads source[i] and, when
// `commit`, writes destination[i].
void test_conv(int channels, int taps, int dilation, int sequences, int width, bool commit, bool distinct,
               std::uint32_t seed) {
    const std::string label = "ple_conv_inject C=" + std::to_string(channels) + " K=" + std::to_string(taps) +
                              " d=" + std::to_string(dilation) + " B=" + std::to_string(sequences) +
                              " W=" + std::to_string(width) + (commit ? (distinct ? " moved" : " in place") : " verify");
    std::mt19937 rng(seed);
    const int span    = (taps - 1) * dilation;
    const int columns = sequences * width;
    const int slots   = 2 * sequences + 1;
    const std::size_t C = static_cast<std::size_t>(channels);
    const auto gated      = random_bf16(C * columns, rng, -1.0F, 1.0F);
    const auto normalized = random_bf16(C * columns, rng, -2.0F, 2.0F);
    const auto weight     = random_bf16(C * taps, rng, -0.8F, 0.8F);
    const auto states     = random_bf16(C * span * slots, rng, -2.0F, 2.0F);
    const auto residual   = random_bf16(C * columns, rng, -4.0F, 4.0F);
    std::vector<std::int32_t> source(sequences), destination(sequences);
    for (int i = 0; i < sequences; ++i) {
        source[i]      = 2 * i + 1;
        destination[i] = distinct ? 2 * i : source[i];
    }

    DeviceBuffer d_gated = to_device(gated), d_normalized = to_device(normalized), d_weight = to_device(weight);
    DeviceBuffer d_source = to_device(source), d_destination = to_device(destination);
    GuardedDeviceBuffer d_states(states.size() * 2), d_residual(residual.size() * 2);
    d_states.copy_from_host(states.data(), states.size() * 2);
    d_residual.copy_from_host(residual.data(), residual.size() * 2);
    Tensor t_gated(d_gated.p, DType::BF16, {channels, columns});
    Tensor t_normalized(d_normalized.p, DType::BF16, {channels, columns});
    Tensor t_weight(d_weight.p, DType::BF16, {channels, taps});
    Tensor t_states(d_states.data(), DType::BF16, {channels, span, slots});
    Tensor t_source(d_source.p, DType::I32, {sequences});
    Tensor t_destination = commit ? Tensor(d_destination.p, DType::I32, {sequences}) : Tensor{};
    Tensor t_residual(d_residual.data(), DType::BF16, {channels, columns});
    ops::ple_conv_inject(t_gated, t_normalized, t_weight, dilation, t_states, t_source, t_destination, t_residual,
                         nullptr);
    cuda_synchronize();

    // The history of sequence i, oldest first: its slot's span columns, then this call's inputs.
    const auto history = [&](int sequence, int channel, int index) -> double {
        if (index < span) {
            return bf16_value(states[(static_cast<std::size_t>(source[sequence]) * span + index) * C + channel]);
        }
        return bf16_value(normalized[(static_cast<std::size_t>(sequence) * width + (index - span)) * C + channel]);
    };
    std::vector<double> want_residual(residual.size());
    for (int sequence = 0; sequence < sequences; ++sequence) {
        for (int local = 0; local < width; ++local) {
            for (int c = 0; c < channels; ++c) {
                double acc = 0;
                for (int j = 0; j < taps; ++j) {
                    acc += bf16_value(weight[static_cast<std::size_t>(j) * C + c]) *
                           history(sequence, c, span + local - (taps - 1 - j) * dilation);
                }
                const std::size_t i = (static_cast<std::size_t>(sequence) * width + local) * C + c;
                want_residual[i]    = bf16_value(residual[i]) + bf16_value(gated[i]) + acc / (1.0 + std::exp(-acc));
            }
        }
    }
    g_failures += verify_pointwise(label + " residual", from_device_bf16(d_residual.data(), residual.size()),
                                   want_residual, kOneRounding);

    std::vector<std::uint16_t> want_states = states;
    if (commit) {
        for (int sequence = 0; sequence < sequences; ++sequence) {
            for (int k = 0; k < span; ++k) {
                for (int c = 0; c < channels; ++c) {
                    want_states[(static_cast<std::size_t>(destination[sequence]) * span + k) * C + c] =
                        bf16_of(history(sequence, c, width + k));
                }
            }
        }
    }
    g_failures += verify_exact((label + " states").c_str(), from_device<std::uint16_t>(d_states.data(), states.size()),
                               want_states);
    g_failures += d_states.verify_guards(label + " states");
    g_failures += d_residual.verify_guards(label + " residual");
}

} // namespace

INFERNIX_OP_TEST(ple) {
    if (cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        // Qwen3.8-Flash-Next's n-gram rows: 16 heads of 160 (embed 2,560); decode, verification and a
        // prefill chunk, plus a small odd geometry.
        test_embed(160, 16, 1, 11);
        test_embed(160, 16, 5, 12);
        test_embed(160, 16, 300, 13);
        test_embed(24, 3, 7, 14);
        // The lossless 12-bit rows: zeros, -0, subnormals, 0/1/4 escapes, sign and mantissa, random
        // rows, at the real width and at the escape column's largest even width.
        test_embed_bf16_p12(160, 16, 3, 3001);
        test_embed_bf16_p12(160, 4, 5, 3002);
        test_embed_bf16_p12(254, 3, 3, 3003); // 9 rows: the seven fixed cases plus the random fill
        // The packed blockdir rows: eight base codes, four slots and a side-stream row.
        test_embed_bf16_p12_blockdir_packed();
        // Four streams of 2,048 and an odd geometry (H not a multiple of the 256-thread block).
        for (const int columns : {1, 4, 33}) { test_gate(4, 2048, columns, 100U + columns); }
        test_gate(2, 300, 5, 150);
        test_gate(3, 1000, 2, 151);
        // Dilation is the n-gram size (3); widths shorter and longer than the span, a sequence per
        // row, the in-place, moved and verification (no destination) state updates.
        for (const bool distinct : {false, true}) {
            test_conv(4 * 2048, 4, 3, 1, 1, true, distinct, 200U + distinct);
            test_conv(4 * 2048, 4, 3, 2, 5, true, distinct, 210U + distinct);
            test_conv(4 * 2048, 4, 3, 1, 40, true, distinct, 220U + distinct);
        }
        test_conv(4 * 2048, 4, 3, 3, 4, false, false, 230U);
        test_conv(300, 2, 2, 2, 3, true, false, 240U);
        test_conv(300, 5, 1, 1, 9, true, true, 241U);
        // Calls wider than a grid's 65,535 rows (core/grid_rows.h), on small rows.
        test_embed(8, 1, 70000, 16);
        test_gate(2, 8, 70000, 152);
        test_conv(8, 4, 3, 1, 70000, true, false, 250U);
        test_conv(8, 2, 2, 2, 40000, true, true, 251U);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d ple check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all ple checks passed\n");
    return 0;
}
