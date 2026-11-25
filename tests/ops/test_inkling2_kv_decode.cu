// Inkling 2's piece decoders (the int8 and bf16p attention instances' decode of pool and window-ring rows) and its bf16p
// row encoder against exact oracles, bit for bit.
//
// int8 G64 (kv_int8_piece.cuh kv_decode8_int8), exhaustively: every code byte (256) under every FP16 scale bit pattern
// (65,536: subnormal, infinite and NaN scales included), for keys (BF16 words) and values (FP16 words). Oracle (host):
// the code as a signed integer q and the scale s decoded from their bits; q * s formed in FP64 (exact: 8 by 11
// significant bits) and rounded to nearest even once to the word's format (BF16 for keys, FP16 for values), by bit
// arithmetic on the FP64 value. A NaN product's key word is 0x7FC0 (canon::f32_to_bf16_rn's), a value's the device
// conversion's NaN (any NaN is accepted there).
//
// bf16p at 128 (kv_bf16p_piece.cuh kv_decode8_bf16p), a thread a piece of every row: (a) exhaustively over the rows a
// base gives alone: every base 0..240 under every exponent nibble 0..13 and every code byte; (b) rows with zeros
// (nibble 14) and escapes (nibble 15) - random rows of 0..40 escapes (past the record's 14 included) and 0..8 zeros at
// random columns, with random escape exponents 0..255 (exponent 0 and Inf / NaN values included), and structured rows
// (every value an escape; escapes in the last piece only; every value a zero). Oracle (host): the format's definition
// - value d's code byte (sign, seven fraction bits) and exponent: base + nibble, 0 for nibble 14, the k-th escape's
// (column order) record byte 2 + k for nibble 15 (0 past the record's 14) - as the BF16 key word; the value's word that
// BF16 value rounded to nearest even to FP16 by the same bit arithmetic (a NaN: any FP16 NaN).
//
// bf16p encoder (kv_bf16p_piece.cuh kv_encode_bf16p, a warp a row), byte for byte: random rows of BF16 words whose
// exponents span 1 to 254 binades (narrow rows, rows just past the 14-binade window, wide and very wide ones, rows
// at the exponent range's ends), with zeros, subnormals, infinities and NaNs mixed in. Oracle (host): the format's
// rule over all bases - the base clamp(lowest, 1, 240) of a row within one window (1 with no normal exponent), else
// the lowest b in 1..240 whose window [b, b + 13] holds the most normal exponents, counted for every b; the nibbles,
// code bytes and record from it; packed iff at most 14 escapes.

#include "ops/inkling2/kv_bf16p_piece.cuh"
#include "ops/inkling2/kv_int8_piece.cuh"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace detail = infernix::ops::inkling2::detail;
using infernix::test::cuda_check;

namespace {

// Block b: scale pattern b, 32 threads; thread t decodes piece t of the 256 code bytes (codes 8 t .. 8 t + 7) as keys
// and as values into out[plane][scale][256].
__global__ void decode_kernel(std::uint16_t* out) {
    const unsigned scale = blockIdx.x;
    const int t          = static_cast<int>(threadIdx.x);
    std::uint8_t c[8];
    for (int i = 0; i < 8; ++i) { c[i] = static_cast<std::uint8_t>(8 * t + i); }
    uint2 codes;
    std::memcpy(&codes, c, 8);
    const uint4 k = detail::kv_decode8_int8<false>(codes, static_cast<std::uint16_t>(scale));
    const uint4 v = detail::kv_decode8_int8<true>(codes, static_cast<std::uint16_t>(scale));
    const std::uint32_t kw[4] = {k.x, k.y, k.z, k.w}, vw[4] = {v.x, v.y, v.z, v.w};
    for (int i = 0; i < 8; ++i) {
        out[(0 * 65536ULL + scale) * 256 + 8 * t + i] = static_cast<std::uint16_t>(kw[i / 2] >> (16 * (i % 2)));
        out[(1 * 65536ULL + scale) * 256 + 8 * t + i] = static_cast<std::uint16_t>(vw[i / 2] >> (16 * (i % 2)));
    }
}

constexpr int kRowBytes = 192, kSideBytes = 16, kValues = 128, kPieces = 16;

// Thread (row, piece): the piece's key words and value words of a bf16p row into out[plane][row][128].
__global__ void bf16p_kernel(const std::uint8_t* rows, const std::uint8_t* sides, int count, std::uint16_t* out) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count * kPieces) { return; }
    const int row = i / kPieces, piece = i % kPieces;
    const std::uint8_t* codes = rows + static_cast<std::size_t>(row) * kRowBytes;
    const std::uint8_t* side  = sides + static_cast<std::size_t>(row) * kSideBytes;
    const uint4 k = detail::kv_decode8_bf16p<false>(codes, side, piece);
    const uint4 v = detail::kv_decode8_bf16p<true>(codes, side, piece);
    const std::uint32_t kw[4] = {k.x, k.y, k.z, k.w}, vw[4] = {v.x, v.y, v.z, v.w};
    for (int j = 0; j < 8; ++j) {
        out[static_cast<std::size_t>(row) * kValues + 8 * piece + j] = static_cast<std::uint16_t>(kw[j / 2] >> (16 * (j % 2)));
        out[(static_cast<std::size_t>(count) + row) * kValues + 8 * piece + j] =
            static_cast<std::uint16_t>(vw[j / 2] >> (16 * (j % 2)));
    }
}

// Warp w: row w's 128 BF16 words (lane l values 4 l ..) encoded into rows / sides; packed[w] its return.
__global__ void bf16p_encode_kernel(const std::uint16_t* words, int count, std::uint8_t* rows, std::uint8_t* sides,
                                    int* packed) {
    const int w = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / 32), lane = static_cast<int>(threadIdx.x % 32);
    if (w >= count) { return; } // (whole warps)
    std::uint16_t bits[4];
    for (int j = 0; j < 4; ++j) { bits[j] = words[static_cast<std::size_t>(w) * kValues + 4 * lane + j]; }
    const bool ok = detail::kv_encode_bf16p(bits, lane, rows + static_cast<std::size_t>(w) * kRowBytes,
                                            sides + static_cast<std::size_t>(w) * kSideBytes);
    if (lane == 0) { packed[w] = ok ? 1 : 0; }
}

double half_value(std::uint16_t h) {
    const int e = (h >> 10) & 0x1F, m = h & 0x3FF;
    const double sign = (h & 0x8000) != 0 ? -1.0 : 1.0;
    if (e == 0x1F) { return m != 0 ? std::nan("") : sign * INFINITY; }
    return sign * (e == 0 ? std::ldexp(m, -24) : std::ldexp(1024 + m, e - 25));
}

double bf16_value(std::uint16_t b) {
    const int e = (b >> 7) & 0xFF, m = b & 0x7F;
    const double sign = (b & 0x8000) != 0 ? -1.0 : 1.0;
    if (e == 0xFF) { return m != 0 ? std::nan("") : sign * INFINITY; }
    return sign * (e == 0 ? std::ldexp(m, -133) : std::ldexp(128 + m, e - 134));
}

// x rounded to nearest even to a format of `mbits` stored mantissa bits, exponent bias `bias`, `ebits` exponent bits;
// returns the format's bits (an infinity past its largest finite value).
std::uint16_t round_to(double x, int mbits, int ebits, int bias) {
    const std::uint16_t sign = std::signbit(x) ? static_cast<std::uint16_t>(1U << (mbits + ebits)) : 0;
    const double a = std::fabs(x);
    const std::uint16_t inf = static_cast<std::uint16_t>(sign | (((1U << ebits) - 1) << mbits));
    if (std::isinf(a)) { return inf; }
    if (a == 0.0) { return sign; }
    int e = std::ilogb(a);                        // a in [2^e, 2^(e+1))
    const int emin = 1 - bias;
    const int q    = std::max(e, emin) - mbits;   // the quantum of a's binade (subnormals: emin's)
    const double scaled = std::ldexp(a, -q);      // exact (a power-of-two scaling)
    double n = std::floor(scaled);
    const double frac = scaled - n;
    if (frac > 0.5 || (frac == 0.5 && std::fmod(n, 2.0) != 0.0)) { n += 1.0; }
    const double r   = std::ldexp(n, q);
    const double max = std::ldexp(2.0 - std::ldexp(1.0, -mbits), (1 << (ebits - 1)) - 1); // the largest finite value
    if (r > max) { return inf; } // (a tie above max rounds to the next power of two: past max)
    if (r < std::ldexp(1.0, emin)) { return static_cast<std::uint16_t>(sign | static_cast<unsigned>(std::ldexp(r, mbits - emin))); }
    e = std::ilogb(r);
    const unsigned mant = static_cast<unsigned>(std::ldexp(r, mbits - e)) - (1U << mbits);
    return static_cast<std::uint16_t>(sign | (static_cast<unsigned>(e + bias) << mbits) | mant);
}

// A bf16p row under construction: code bytes, nibbles, and the record's base, escapes and zero flag.
struct Bf16pRow {
    std::uint8_t row[kRowBytes] = {};
    std::uint8_t side[kSideBytes] = {};
    void set(int d, unsigned code, unsigned nibble) {
        row[d] = static_cast<std::uint8_t>(code);
        std::uint8_t& byte = row[128 + d / 2];
        byte = static_cast<std::uint8_t>(d % 2 == 0 ? (byte & 0xF0U) | nibble : (byte & 0x0FU) | (nibble << 4));
    }
    unsigned nibble(int d) const { return (row[128 + d / 2] >> (4 * (d % 2))) & 0xFU; }
};

// The format's definition: value d's BF16 word.
std::uint16_t bf16p_oracle(const Bf16pRow& r, int d) {
    unsigned escapes = 0; // the row's escapes before d
    for (int c = 0; c < d; ++c) { escapes += r.nibble(c) == 15 ? 1U : 0U; }
    const unsigned n = r.nibble(d), code = r.row[d];
    const unsigned e = n < 14 ? r.side[0] + n : n == 14 ? 0U : escapes < 14 ? r.side[2 + escapes] : 0U;
    return static_cast<std::uint16_t>(((code & 0x80U) << 8) | ((e & 0xFFU) << 7) | (code & 0x7FU));
}

// Rows of a base alone (every base, nibble and code byte), then the zero and escape rows.
std::vector<Bf16pRow> bf16p_rows(std::size_t& plain) {
    std::vector<Bf16pRow> rows;
    for (unsigned base = 0; base <= 240; ++base) {
        for (unsigned n = 0; n < 14; ++n) {
            for (int half = 0; half < 2; ++half) {
                Bf16pRow r;
                for (int d = 0; d < kValues; ++d) { r.set(d, static_cast<unsigned>(128 * half + d), n); }
                r.side[0] = static_cast<std::uint8_t>(base);
                rows.push_back(r);
            }
        }
    }
    plain = rows.size();
    std::mt19937 rng(20261125U);
    const auto finish = [](Bf16pRow& r, unsigned base, const std::vector<unsigned>& exponents, bool zeros) {
        r.side[0] = static_cast<std::uint8_t>(base);
        const unsigned count = static_cast<unsigned>(std::min<std::size_t>(exponents.size(), 14));
        r.side[1] = static_cast<std::uint8_t>(count | (zeros ? 0x80U : 0U));
        for (unsigned k = 0; k < count; ++k) { r.side[2 + k] = static_cast<std::uint8_t>(exponents[k]); }
    };
    for (int i = 0; i < 40000; ++i) {
        Bf16pRow r;
        const unsigned base  = 1 + rng() % 240;
        const int escapes    = static_cast<int>(rng() % 41) * (i % 4 == 0 ? 1 : 0) + static_cast<int>(rng() % 15) * (i % 4 != 0 ? 1 : 0);
        const int zero_count = static_cast<int>(rng() % 9);
        std::vector<unsigned> nib(kValues);
        for (auto& n : nib) { n = rng() % 14; }
        for (int k = 0; k < zero_count; ++k) { nib[rng() % kValues] = 14; }
        for (int k = 0; k < escapes; ++k) { nib[rng() % kValues] = 15; }
        std::vector<unsigned> exponents;
        bool zeros = false;
        for (int d = 0; d < kValues; ++d) {
            r.set(d, rng() & 0xFFU, nib[d]);
            if (nib[d] == 15) { exponents.push_back(rng() % 256); }
            zeros = zeros || nib[d] == 14;
        }
        finish(r, base, exponents, zeros);
        rows.push_back(r);
    }
    for (int shape = 0; shape < 3; ++shape) {
        for (int i = 0; i < 64; ++i) {
            Bf16pRow r;
            std::vector<unsigned> exponents;
            for (int d = 0; d < kValues; ++d) {
                const unsigned n = shape == 0 ? 15U : shape == 1 ? (d >= 120 ? 15U : rng() % 14) : 14U;
                r.set(d, rng() & 0xFFU, n);
                if (n == 15) { exponents.push_back(rng() % 256); }
            }
            finish(r, 1 + rng() % 240, exponents, shape == 2);
            rows.push_back(r);
        }
    }
    return rows;
}

int int8_part() {
    const std::size_t words = 2ULL * 65536 * 256;
    std::uint16_t* d = nullptr;
    cuda_check(cudaMalloc(&d, words * sizeof(std::uint16_t)), "malloc");
    decode_kernel<<<65536, 32>>>(d);
    cuda_check(cudaGetLastError(), "launch");
    std::vector<std::uint16_t> got(words);
    cuda_check(cudaMemcpy(got.data(), d, words * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "copy");
    cuda_check(cudaFree(d), "free");
    std::size_t bad[2] = {0, 0}, nans = 0, subnormal = 0;
    for (int plane = 0; plane < 2; ++plane) {
        for (unsigned s = 0; s < 65536; ++s) {
            const double scale = half_value(static_cast<std::uint16_t>(s));
            for (int b = 0; b < 256; ++b) {
                const double x = static_cast<double>(static_cast<std::int8_t>(b)) * scale; // exact
                const std::uint16_t g = got[(plane * 65536ULL + s) * 256 + b];
                bool ok;
                if (std::isnan(x)) {
                    ok = plane == 0 ? g == 0x7FC0U : ((g & 0x7C00U) == 0x7C00U && (g & 0x3FFU) != 0);
                    ++nans;
                } else {
                    const std::uint16_t want = plane == 0 ? round_to(x, 7, 8, 127) : round_to(x, 10, 5, 15);
                    ok = g == want;
                    if (x != 0.0 && std::fabs(x) < (plane == 0 ? 0x1p-126 : 0x1p-14)) { ++subnormal; }
                }
                if (!ok && bad[plane]++ == 0) {
                    std::printf("  first %s mismatch: scale 0x%04X code 0x%02X (q * s = %.17g): got 0x%04X\n",
                                plane == 0 ? "key" : "value", s, b, x, g);
                }
            }
        }
    }
    std::printf("int8 piece decoder: 256 codes x 65536 scales: %zu key and %zu value words differ from the exact "
                "rounding (%zu NaN products, %zu subnormal results checked)\n",
                bad[0], bad[1], nans, subnormal);
    return bad[0] + bad[1] == 0 ? 0 : 1;
}

int bf16p_part() {
    std::size_t plain = 0;
    const std::vector<Bf16pRow> rows = bf16p_rows(plain);
    const int count = static_cast<int>(rows.size());
    std::vector<std::uint8_t> codes(rows.size() * kRowBytes), sides(rows.size() * kSideBytes);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        std::memcpy(codes.data() + r * kRowBytes, rows[r].row, kRowBytes);
        std::memcpy(sides.data() + r * kSideBytes, rows[r].side, kSideBytes);
    }
    std::uint8_t *d_codes = nullptr, *d_sides = nullptr;
    std::uint16_t* d_out = nullptr;
    const std::size_t words = 2ULL * rows.size() * kValues;
    cuda_check(cudaMalloc(&d_codes, codes.size()), "malloc codes");
    cuda_check(cudaMalloc(&d_sides, sides.size()), "malloc sides");
    cuda_check(cudaMalloc(&d_out, words * sizeof(std::uint16_t)), "malloc out");
    cuda_check(cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice), "upload codes");
    cuda_check(cudaMemcpy(d_sides, sides.data(), sides.size(), cudaMemcpyHostToDevice), "upload sides");
    const int threads = 256, blocks = (count * kPieces + threads - 1) / threads;
    bf16p_kernel<<<blocks, threads>>>(d_codes, d_sides, count, d_out);
    cuda_check(cudaGetLastError(), "launch bf16p");
    std::vector<std::uint16_t> got(words);
    cuda_check(cudaMemcpy(got.data(), d_out, words * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "copy bf16p");
    cuda_check(cudaFree(d_codes), "free");
    cuda_check(cudaFree(d_sides), "free");
    cuda_check(cudaFree(d_out), "free");
    std::size_t bad[2][2] = {}, nans = 0, escapes = 0, zeros = 0, past = 0;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const int set = r < plain ? 0 : 1;
        unsigned seen = 0;
        for (int d = 0; d < kValues; ++d) {
            const unsigned n = rows[r].nibble(d);
            if (n == 15) { ++escapes, past += seen++ >= 14 ? 1U : 0U; }
            zeros += n == 14 ? 1U : 0U;
            const std::uint16_t key = bf16p_oracle(rows[r], d);
            const double x          = bf16_value(key);
            const std::uint16_t gk  = got[r * kValues + d], gv = got[(rows.size() + r) * kValues + d];
            bool value_ok;
            if (std::isnan(x)) {
                value_ok = (gv & 0x7C00U) == 0x7C00U && (gv & 0x3FFU) != 0;
                ++nans;
            } else {
                value_ok = gv == round_to(x, 10, 5, 15);
            }
            if (gk != key && bad[set][0]++ == 0) {
                std::printf("  first key mismatch (row %zu, value %d, nibble %u): got 0x%04X want 0x%04X\n", r, d, n, gk, key);
            }
            if (!value_ok && bad[set][1]++ == 0) {
                std::printf("  first value mismatch (row %zu, value %d, nibble %u, BF16 0x%04X): got 0x%04X\n", r, d, n, key, gv);
            }
        }
    }
    std::printf("bf16p piece decoder: %zu rows of a base alone (bases 0..240 x nibbles 0..13 x code bytes): %zu key and "
                "%zu value words differ; %zu zero / escape rows (%zu escapes, %zu past the record's 14, %zu zeros, %zu NaN "
                "values): %zu key and %zu value words differ\n",
                plain, bad[0][0], bad[0][1], rows.size() - plain, escapes, past, zeros, nans, bad[1][0], bad[1][1]);
    return bad[0][0] + bad[0][1] + bad[1][0] + bad[1][1] == 0 ? 0 : 1;
}

// The format's encoding of a row of 128 BF16 words (every base counted); returns whether it has a packed form.
bool bf16p_encode_oracle(const std::uint16_t* words, std::uint8_t* row, std::uint8_t* side) {
    unsigned e[kValues];
    unsigned lowest = 255, highest = 0;
    for (int d = 0; d < kValues; ++d) {
        e[d] = (words[d] >> 7) & 0xFFU;
        if (e[d] != 0 && e[d] != 255) { lowest = std::min(lowest, e[d]), highest = std::max(highest, e[d]); }
    }
    unsigned base;
    if (highest < lowest) {
        base = 1;
    } else if (highest - lowest < 14) {
        base = std::min(std::max(lowest, 1U), 240U);
    } else {
        unsigned most = 0;
        base          = 1;
        for (unsigned b = 1; b <= 240; ++b) {
            unsigned n = 0;
            for (int d = 0; d < kValues; ++d) { n += e[d] != 0 && e[d] != 255 && e[d] >= b && e[d] <= b + 13 ? 1U : 0U; }
            if (n > most) { most = n, base = b; }
        }
    }
    std::memset(row, 0, kRowBytes);
    std::memset(side, 0, kSideBytes);
    unsigned escapes = 0;
    bool zeros       = false;
    for (int d = 0; d < kValues; ++d) {
        const bool normal = e[d] != 0 && e[d] != 255;
        const unsigned n  = e[d] == 0 ? 14U : normal && e[d] >= base && e[d] - base < 14 ? e[d] - base : 15U;
        row[d]            = static_cast<std::uint8_t>(((words[d] >> 8) & 0x80U) | (words[d] & 0x7FU));
        row[128 + d / 2] |= static_cast<std::uint8_t>(n << (4 * (d % 2)));
        if (n == 15) {
            if (escapes < 14) { side[2 + escapes] = static_cast<std::uint8_t>(e[d]); }
            ++escapes;
        }
        zeros = zeros || e[d] == 0;
    }
    side[0] = static_cast<std::uint8_t>(base);
    side[1] = static_cast<std::uint8_t>(std::min(escapes, 14U) | (zeros ? 0x80U : 0U));
    return escapes <= 14;
}

// Random rows: exponents around a centre over a span (narrow to very wide), with zeros, subnormals, infinities and NaNs.
std::vector<std::uint16_t> bf16p_encode_rows(int count) {
    std::mt19937 rng(20261126U);
    std::vector<std::uint16_t> words(static_cast<std::size_t>(count) * kValues);
    const unsigned spans[] = {1, 6, 13, 14, 15, 16, 18, 22, 30, 60, 120, 254};
    for (int r = 0; r < count; ++r) {
        if (r % 2 == 0) {
            // As the model's rows (census 11-25: 5-8 % with escapes, mostly 1-3): exponents inside a band of 8-13
            // binades, a third of the rows with 1-3 values pushed 1-20 binades past it, a few with zeros.
            const unsigned width = 8 + rng() % 6;
            const unsigned low   = 1 + rng() % (254 - width);
            for (int d = 0; d < kValues; ++d) {
                const unsigned ex = low + rng() % (width + 1);
                words[static_cast<std::size_t>(r) * kValues + d] =
                    static_cast<std::uint16_t>((rng() & 0x8000U) | (ex << 7) | (rng() & 0x7FU));
            }
            const int outliers = rng() % 3 == 0 ? 1 + static_cast<int>(rng() % 3) : 0;
            for (int k = 0; k < outliers; ++k) {
                const int step = 1 + static_cast<int>(rng() % 20);
                const int ex   = std::clamp(rng() % 2 == 0 ? static_cast<int>(low) - step : static_cast<int>(low + width) + step, 1, 254);
                words[static_cast<std::size_t>(r) * kValues + rng() % kValues] =
                    static_cast<std::uint16_t>((rng() & 0x8000U) | (static_cast<unsigned>(ex) << 7) | (rng() & 0x7FU));
            }
            if (rng() % 50 == 0) { words[static_cast<std::size_t>(r) * kValues + rng() % kValues] = static_cast<std::uint16_t>(rng() & 0x8000U); }
            continue;
        }
        const unsigned span   = spans[r % 12];
        const unsigned centre = r % 7 == 0 ? 1 + rng() % 20 : r % 7 == 1 ? 234 + rng() % 20 : 1 + rng() % 253;
        const int specials    = r % 5 == 0 ? static_cast<int>(rng() % 6) : 0;
        for (int d = 0; d < kValues; ++d) {
            // A skewed spread (most exponents near the centre, a tail below it), clamped to the normal range.
            const int offset = static_cast<int>(rng() % (span + 1)) - static_cast<int>((rng() % 4 == 0 ? span : span / 4));
            const int ex     = std::clamp(static_cast<int>(centre) + offset, 1, 254);
            words[static_cast<std::size_t>(r) * kValues + d] =
                static_cast<std::uint16_t>((rng() & 0x8000U) | (static_cast<unsigned>(ex) << 7) | (rng() & 0x7FU));
        }
        for (int k = 0; k < specials; ++k) {
            const unsigned kind = rng() % 4; // zero, subnormal, infinity, NaN
            const std::uint16_t w = kind == 0 ? 0 : kind == 1 ? static_cast<std::uint16_t>(1 + rng() % 127)
                                    : kind == 2 ? 0x7F80 : static_cast<std::uint16_t>(0x7F80 | (1 + rng() % 127));
            words[static_cast<std::size_t>(r) * kValues + rng() % kValues] = static_cast<std::uint16_t>(w | (rng() & 0x8000U));
        }
    }
    return words;
}

int encode_part() {
    const int count = 120000;
    const std::vector<std::uint16_t> words = bf16p_encode_rows(count);
    std::uint16_t* d_words = nullptr;
    std::uint8_t *d_rows = nullptr, *d_sides = nullptr;
    int* d_packed = nullptr;
    cuda_check(cudaMalloc(&d_words, words.size() * sizeof(std::uint16_t)), "malloc words");
    cuda_check(cudaMalloc(&d_rows, static_cast<std::size_t>(count) * kRowBytes), "malloc rows");
    cuda_check(cudaMalloc(&d_sides, static_cast<std::size_t>(count) * kSideBytes), "malloc sides");
    cuda_check(cudaMalloc(&d_packed, static_cast<std::size_t>(count) * sizeof(int)), "malloc packed");
    cuda_check(cudaMemcpy(d_words, words.data(), words.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice), "upload words");
    bf16p_encode_kernel<<<(count * 32 + 255) / 256, 256>>>(d_words, count, d_rows, d_sides, d_packed);
    cuda_check(cudaGetLastError(), "launch encode");
    std::vector<std::uint8_t> rows(static_cast<std::size_t>(count) * kRowBytes), sides(static_cast<std::size_t>(count) * kSideBytes);
    std::vector<int> packed(count);
    cuda_check(cudaMemcpy(rows.data(), d_rows, rows.size(), cudaMemcpyDeviceToHost), "copy rows");
    cuda_check(cudaMemcpy(sides.data(), d_sides, sides.size(), cudaMemcpyDeviceToHost), "copy sides");
    cuda_check(cudaMemcpy(packed.data(), d_packed, packed.size() * sizeof(int), cudaMemcpyDeviceToHost), "copy packed");
    for (void* p : {static_cast<void*>(d_words), static_cast<void*>(d_rows), static_cast<void*>(d_sides), static_cast<void*>(d_packed)}) {
        cuda_check(cudaFree(p), "free");
    }
    std::size_t bad = 0, wide = 0, few = 0, unpackable = 0;
    for (int r = 0; r < count; ++r) {
        std::uint8_t row[kRowBytes], side[kSideBytes];
        const bool ok = bf16p_encode_oracle(words.data() + static_cast<std::size_t>(r) * kValues, row, side);
        wide += (side[1] & 0x7FU) != 0 ? 1U : 0U;
        few += (side[1] & 0x7FU) >= 1 && (side[1] & 0x7FU) <= 3 ? 1U : 0U;
        unpackable += ok ? 0U : 1U;
        const bool same = std::memcmp(row, rows.data() + static_cast<std::size_t>(r) * kRowBytes, kRowBytes) == 0 &&
                          std::memcmp(side, sides.data() + static_cast<std::size_t>(r) * kSideBytes, kSideBytes) == 0 &&
                          (packed[r] != 0) == ok;
        if (!same && bad++ == 0) {
            std::printf("  first encoder mismatch: row %d (base want %u got %u, record byte 1 want 0x%02X got 0x%02X)\n", r, side[0],
                        sides[static_cast<std::size_t>(r) * kSideBytes], side[1], sides[static_cast<std::size_t>(r) * kSideBytes + 1]);
        }
    }
    std::printf("bf16p encoder: %d rows (%zu with escapes, %zu of them 1-3, %zu with no packed form): %zu differ from the "
                "format's encoding\n",
                count, wide, few, unpackable, bad);
    return bad == 0 ? 0 : 1;
}

} // namespace

INFERNIX_OP_TEST(inkling2_kv_decode) {
    (void)argc, (void)argv;
    const int failed = int8_part() + bf16p_part() + encode_part();
    std::printf(failed == 0 ? "PASS\n" : "FAILED\n");
    return failed == 0 ? 0 : 1;
}
