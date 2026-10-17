#pragma once

// The fast form of the exact ggml arithmetic (ops/ggml/ggml_exact.h) on the GPU: every R it adds
// up is the same integer, so the outputs are the same bits.
//
// A column's A16 integers X (|X| < 2^21) are stored as three signed byte limbs,
// X = a * 2^16 + b * 2^8 + c with b, c in [-128, 127] and |a| <= 32, packed four elements to a
// word and grouped by 32-element slice (EncodedColumn). A weight slice (32 consecutive elements of
// a row) is decoded once into eight words of signed byte codes and its sub-scales (two per slice:
// elements 0-15 and 16-31). Its contribution to R is then
//
//   P = sub0 * dot(codes[0..16), X) + sub1 * dot(codes[16..32), X)          exact, int64
//   R += mant(d) * P * 2^(e(d) - 1)  [ - mant(dmin) * min * sum(X) * 2^(e(dmin) - 1) ]
//
// where each 16-element dot is three dp4a chains over the limbs (|partial| < 2^18), so a slice
// costs 24 dp4a per column and one int128 addition. Bounds: |sub| <= 127 and |code| <= 127 give
// |sub * chain| < 2^26 (int32), |P| < 2^42 and |mant * P| < 2^53 (int64).

#include "core/weight.h"
#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_exact.h"

#include <cstdint>

namespace infernix::ggml::fast {

// --------------------------------------------------------------------------- encoded columns

// One 32-element slice of an encoded column: limb a, b and c words of its eight 4-element groups.
constexpr int kSliceWords = 24;

// Layout of an encoded column of K elements (K a multiple of 32): K / 32 slices of kSliceWords
// int32, then K / 32 int32 slice sums (sum of X over the slice, |sum| < 2^26).
__host__ __device__ constexpr std::size_t encoded_column_words(int k) { return static_cast<std::size_t>(k) / 32 * (kSliceWords + 1); }

// X = a * 2^16 + b * 2^8 + c, b and c in [-128, 127].
__device__ __forceinline__ void split_limbs(int x, int& a, int& b, int& c) {
    c = static_cast<int>(static_cast<std::int8_t>(x & 0xFF));
    const int x1 = (x - c) >> 8;
    b = static_cast<int>(static_cast<std::int8_t>(x1 & 0xFF));
    a = (x1 - b) >> 8;
}

__device__ __forceinline__ int pack4(int v0, int v1, int v2, int v3) {
    return static_cast<int>((static_cast<unsigned>(v0) & 0xFFU) | ((static_cast<unsigned>(v1) & 0xFFU) << 8) |
                            ((static_cast<unsigned>(v2) & 0xFFU) << 16) | (static_cast<unsigned>(v3) << 24));
}

// Encodes four consecutive A16 integers (group g of the column, elements 4g..4g+3) into `column`.
// Returns their sum.
__device__ __forceinline__ int store_group(std::int32_t* column, int g, int x0, int x1, int x2, int x3) {
    int a[4], b[4], c[4];
    split_limbs(x0, a[0], b[0], c[0]);
    split_limbs(x1, a[1], b[1], c[1]);
    split_limbs(x2, a[2], b[2], c[2]);
    split_limbs(x3, a[3], b[3], c[3]);
    std::int32_t* slice = column + (g / 8) * kSliceWords;
    const int w         = g % 8;
    slice[w]      = pack4(a[0], a[1], a[2], a[3]);
    slice[8 + w]  = pack4(b[0], b[1], b[2], b[3]);
    slice[16 + w] = pack4(c[0], c[1], c[2], c[3]);
    return x0 + x1 + x2 + x3;
}

// ------------------------------------------------------------------------------- weight slices

// Formats with a min term (whose slices read the column's slice sums).
template <QType F>
inline constexpr bool kMin = F == QType::GGML_Q4_K || F == QType::GGML_Q5_K;

struct Slice {
    int q[8];          // signed byte codes, element 4w + j in byte j of q[w]
    int sub0, sub1;    // sub-scales of elements 0-15 and 16-31
    std::uint16_t d;
    int min;           // Q4_K/Q5_K: the slice's min sub-scale (0 otherwise)
    std::uint16_t dmin;
};

__device__ __forceinline__ std::uint32_t ld_u16(const std::uint8_t* p) {
    return __ldg(reinterpret_cast<const unsigned short*>(p));
}

// Four bytes at a 2-byte-aligned address.
__device__ __forceinline__ std::uint32_t ld_u32_a2(const std::uint8_t* p) {
    return ld_u16(p) | (ld_u16(p + 2) << 16);
}

__device__ __forceinline__ std::uint32_t ld_u32_a4(const std::uint8_t* p) {
    return __ldg(reinterpret_cast<const unsigned int*>(p));
}

// N little-endian words at a 2-byte-aligned address: N aligned 4-byte loads (and a further one when
// the address is not 4-byte aligned, never past the words' last byte otherwise) and funnel shifts,
// half the load instructions of N ld_u32_a2 (decode at one or two columns is bound by the LSU queue).
template <int N>
__device__ __forceinline__ void ld_words_a2(const std::uint8_t* p, std::uint32_t (&out)[N]) {
    const auto at        = reinterpret_cast<std::uintptr_t>(p);
    const unsigned shift = static_cast<unsigned>(at & 3U) * 8U;
    const auto* w        = reinterpret_cast<const std::uint8_t*>(at & ~std::uintptr_t{3});
    std::uint32_t v[N + 1];
#pragma unroll
    for (int i = 0; i < N; ++i) { v[i] = ld_u32_a4(w + 4 * i); }
    v[N] = shift != 0 ? ld_u32_a4(w + 4 * N) : 0U;
#pragma unroll
    for (int i = 0; i < N; ++i) { out[i] = __funnelshift_r(v[i], v[i + 1], shift); }
}

// Byte-wise sign application: bytes of `v` whose bit in `signs` (bit j for byte j, signs < 16)
// is set negate. Every byte of `v` is a nonzero grid value, so (b ^ 0xFF) + 1 never carries into
// the next byte: native integer instructions only (the packed-byte intrinsics __vcmpne4/__vsub4
// are emulated, about ten instructions each). `one` holds bit j of `signs` in byte j: the four
// shifted copies 1, 2^7, 2^14, 2^21 of `signs` do not overlap.
__device__ __forceinline__ int apply_signs4(std::uint32_t v, unsigned signs) {
    const std::uint32_t one = (signs * 0x00204081U) & 0x01010101U;
    return static_cast<int>((v ^ (one * 0xFFU)) + one);
}

// Each byte of `v` minus `offset` (bytes below 0x80 each, as is every byte of `v`) as signed bytes:
// raising every byte to 0x80 first keeps a byte's difference from borrowing from its neighbour.
__device__ __forceinline__ int sub_bytes4(std::uint32_t v, std::uint32_t offset) {
    return static_cast<int>(((v | 0x80808080U) - offset) ^ 0x80808080U);
}

// The 16 kvalues_iq4nl bytes as four little-endian words (immediate operands).
__host__ __device__ constexpr std::uint32_t iq4nl_word(int w) {
    std::uint32_t v = 0;
    for (int j = 0; j < 4; ++j) { v |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(k_kvalues_iq4nl[4 * w + j])) << (8 * j); }
    return v;
}

// kvalues_iq4nl of the eight nibbles of `q4`: .x the low nibbles of its four bytes, .y the high
// nibbles (llama.cpp's get_int_from_table_16).
__device__ __forceinline__ int2 iq4nl_values(std::uint32_t q4) {
    constexpr std::uint32_t t0 = iq4nl_word(0), t1 = iq4nl_word(1), t2 = iq4nl_word(2), t3 = iq4nl_word(3);
    const std::uint32_t select = 0x32103210U | ((q4 & 0x88888888U) >> 1);
    std::uint32_t tmp[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const std::uint32_t shift = 16 * i;
        const std::uint32_t low   = __byte_perm(t0, t1, q4 >> shift);
        const std::uint32_t high  = __byte_perm(t2, t3, q4 >> shift);
        tmp[i]                    = __byte_perm(low, high, select >> shift);
    }
    return make_int2(static_cast<int>(__byte_perm(tmp[0], tmp[1], 0x6420)),
                     static_cast<int>(__byte_perm(tmp[0], tmp[1], 0x7531)));
}

// Slice s (elements 32s..32s+31) of a row of F blocks.
template <QType F>
__device__ __forceinline__ Slice slice(const std::uint8_t* row, int s) {
    Slice o;
    o.min  = 0;
    o.dmin = 0;
    if constexpr (F == QType::GGML_Q2_0) {
        const std::uint8_t* b = row + (s / 2) * 18;
        o.d    = static_cast<std::uint16_t>(ld_u16(b));
        o.sub0 = o.sub1 = 1;
        std::uint32_t qw[2];
        ld_words_a2<2>(b + 2 + 8 * (s % 2), qw);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const std::uint32_t w = qw[h];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const std::uint32_t v = (w >> (8 * j)) & 0xFFU; // four 2-bit codes, element 4i+k at bits 2k
                const std::uint32_t e = (v & 3U) | (((v >> 2) & 3U) << 8) | (((v >> 4) & 3U) << 16) | (((v >> 6) & 3U) << 24);
                o.q[4 * h + j] = sub_bytes4(e, 0x01010101U);
            }
        }
    } else if constexpr (F == QType::GGML_IQ4_NL) {
        const std::uint8_t* b = row + s * 18;
        o.d    = static_cast<std::uint16_t>(ld_u16(b));
        o.sub0 = o.sub1 = 1;
        std::uint32_t qs[4];
        ld_words_a2<4>(b + 2, qs);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int2 v = iq4nl_values(qs[w]);
            o.q[w]     = v.x;
            o.q[w + 4] = v.y;
        }
    } else if constexpr (F == QType::GGML_IQ4_XS) {
        const std::uint8_t* b = row + (s / 8) * 136;
        const int ib = s % 8;
        o.d = static_cast<std::uint16_t>(ld_u16(b));
        const std::uint32_t scales_h = ld_u16(b + 2);
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
        o.sub0 = o.sub1 = ls - 32;
        const std::uint8_t* qs = b + 8 + 16 * ib;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const int2 v = iq4nl_values(ld_u32_a4(qs + 4 * w));
            o.q[w]     = v.x;
            o.q[w + 4] = v.y;
        }
    } else if constexpr (F == QType::GGML_Q4_K || F == QType::GGML_Q5_K) {
        constexpr int kBytes = F == QType::GGML_Q4_K ? 144 : 176;
        const std::uint8_t* b = row + (s / 8) * kBytes;
        const int sb = s % 8, ch = sb / 2, shift = 4 * (sb % 2);
        o.d    = static_cast<std::uint16_t>(ld_u16(b));
        o.dmin = static_cast<std::uint16_t>(ld_u16(b + 2));
        int sc, m;
        scale_min_k4(sb, b + 4, sc, m);
        o.sub0 = o.sub1 = sc;
        o.min           = m;
        const std::uint8_t* qs = b + (F == QType::GGML_Q4_K ? 16 : 48) + 32 * ch;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            std::uint32_t v = (ld_u32_a4(qs + 4 * w) >> shift) & 0x0F0F0F0FU;
            if constexpr (F == QType::GGML_Q5_K) { v |= ((ld_u32_a4(b + 16 + 4 * w) >> sb) & 0x01010101U) << 4; }
            o.q[w] = static_cast<int>(v);
        }
    } else if constexpr (F == QType::GGML_Q6_K) {
        const std::uint8_t* b = row + (s / 8) * 210;
        const int sb = s % 8, half = sb / 4, quarter = sb % 4;
        o.d = static_cast<std::uint16_t>(ld_u16(b + 208));
        const std::uint8_t* ql = b + 64 * half + 32 * (quarter & 1);
        const std::uint8_t* qh = b + 128 + 32 * half;
        const int lshift = quarter >= 2 ? 4 : 0, hshift = 2 * quarter;
        const std::uint32_t subs = ld_u16(b + 192 + 8 * half + 2 * quarter);
        o.sub0 = static_cast<std::int8_t>(subs & 0xFFU);
        o.sub1 = static_cast<std::int8_t>(subs >> 8);
        std::uint32_t lw[8], hw[8];
        ld_words_a2<8>(ql, lw);
        ld_words_a2<8>(qh, hw);
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const std::uint32_t lo = (lw[w] >> lshift) & 0x0F0F0F0FU;
            const std::uint32_t hi = ((hw[w] >> hshift) & 0x03030303U) << 4;
            o.q[w] = sub_bytes4(lo | hi, 0x20202020U);
        }
    } else if constexpr (F == QType::GGML_IQ2_S) {
        const std::uint8_t* b = row + (s / 8) * 82;
        const int ib = s % 8;
        o.d = static_cast<std::uint16_t>(ld_u16(b));
        const int qh = b[66 + ib], sc = b[74 + ib];
        o.sub0 = 2 * (sc & 0xF) + 1;
        o.sub1 = 2 * (sc >> 4) + 1;
        const std::uint32_t qs = ld_u32_a2(b + 2 + 4 * ib), signs = ld_u32_a2(b + 34 + 4 * ib);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int index = ((qs >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300);
            const std::uint64_t grid = __ldg(reinterpret_cast<const unsigned long long*>(d_iq2s_grid) + index);
            const unsigned sign      = (signs >> (8 * l)) & 0xFFU;
            o.q[2 * l]     = apply_signs4(static_cast<std::uint32_t>(grid), sign & 0xFU);
            o.q[2 * l + 1] = apply_signs4(static_cast<std::uint32_t>(grid >> 32), sign >> 4);
        }
    } else if constexpr (F == QType::GGML_IQ3_XXS) {
        const std::uint8_t* b = row + (s / 8) * 98;
        const int ib = s % 8;
        o.d = static_cast<std::uint16_t>(ld_u16(b));
        const std::uint32_t aux = ld_u32_a2(b + 66 + 4 * ib);
        o.sub0 = o.sub1 = 2 * static_cast<int>(aux >> 28) + 1;
        std::uint32_t qw[2];
        ld_words_a2<2>(b + 2 + 8 * ib, qw);
        const std::uint32_t qs0 = qw[0], qs1 = qw[1];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t qs = l < 2 ? qs0 : qs1;
            const int i1 = (qs >> (16 * (l % 2))) & 0xFF, i2 = (qs >> (16 * (l % 2) + 8)) & 0xFF;
            const unsigned sign = __ldg(d_ksigns_iq2xs + ((aux >> (7 * l)) & 127));
            o.q[2 * l]     = apply_signs4(__ldg(d_iq3xxs_grid + i1), sign & 0xFU);
            o.q[2 * l + 1] = apply_signs4(__ldg(d_iq3xxs_grid + i2), sign >> 4);
        }
    } else if constexpr (F == QType::GGML_IQ3_S) {
        const std::uint8_t* b = row + (s / 8) * 110;
        const int sb = s % 8;
        o.d = static_cast<std::uint16_t>(ld_u16(b));
        const int qh = b[66 + sb];
        o.sub0 = o.sub1 = 2 * ((b[106 + sb / 2] >> (4 * (sb % 2))) & 0xF) + 1;
        std::uint32_t qw[2];
        ld_words_a2<2>(b + 2 + 8 * sb, qw);
        const std::uint32_t qs0 = qw[0], qs1 = qw[1];
        const std::uint32_t signs = ld_u32_a2(b + 74 + 4 * sb);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t qs = l < 2 ? qs0 : qs1;
            const int i1 = ((qs >> (16 * (l % 2))) & 0xFF) | ((qh << (8 - 2 * l)) & 256);
            const int i2 = ((qs >> (16 * (l % 2) + 8)) & 0xFF) | ((qh << (7 - 2 * l)) & 256);
            const unsigned sign = (signs >> (8 * l)) & 0xFFU;
            o.q[2 * l]     = apply_signs4(__ldg(d_iq3s_grid + i1), sign & 0xFU);
            o.q[2 * l + 1] = apply_signs4(__ldg(d_iq3s_grid + i2), sign >> 4);
        }
    }
    return o;
}

// Chunk p (elements 8p..8p+7, words 2p and 2p+1 of slice()'s q) of slice s of a row of F blocks,
// with the sub-scale of the slice half that holds it and the slice's scale words: the integers of
// slice(), decoded from the 2-byte-aligned loads of the chunk's bytes alone.
struct Chunk {
    int w0, w1;
    int sub;
    std::uint16_t d;
    int min;
    std::uint16_t dmin;
};

template <QType F>
__device__ __forceinline__ Chunk chunk(const std::uint8_t* row, int s, int p) {
    Chunk o;
    o.sub  = 1;
    o.min  = 0;
    o.dmin = 0;
    if constexpr (F == QType::GGML_Q2_0) {
        const std::uint8_t* b = row + (s / 2) * 18;
        o.d                   = static_cast<std::uint16_t>(ld_u16(b));
        const std::uint32_t v = ld_u16(b + 2 + 8 * (s % 2) + 2 * p); // bytes 2p, 2p + 1: words 2p, 2p + 1
        const auto codes      = [](std::uint32_t x) {
            return sub_bytes4((x & 3U) | (((x >> 2) & 3U) << 8) | (((x >> 4) & 3U) << 16) | (((x >> 6) & 3U) << 24),
                              0x01010101U);
        };
        o.w0 = codes(v & 0xFFU);
        o.w1 = codes(v >> 8);
    } else if constexpr (F == QType::GGML_IQ4_NL || F == QType::GGML_IQ4_XS) {
        const std::uint8_t* qs;
        if constexpr (F == QType::GGML_IQ4_NL) {
            const std::uint8_t* b = row + s * 18;
            o.d                   = static_cast<std::uint16_t>(ld_u16(b));
            qs                    = b + 2;
        } else {
            const std::uint8_t* b = row + (s / 8) * 136;
            const int ib          = s % 8;
            o.d                   = static_cast<std::uint16_t>(ld_u16(b));
            const std::uint32_t scales_h = ld_u16(b + 2);
            o.sub = (((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4)) - 32;
            qs    = b + 8 + 16 * ib;
        }
        // Words w < 4 are the low nibbles of bytes 4w.., words 4 + w the high nibbles.
        const int w = 2 * p % 4;
        const int2 v0 = iq4nl_values(ld_u32_a2(qs + 4 * w)), v1 = iq4nl_values(ld_u32_a2(qs + 4 * w + 4));
        o.w0 = p >= 2 ? v0.y : v0.x;
        o.w1 = p >= 2 ? v1.y : v1.x;
    } else if constexpr (F == QType::GGML_Q4_K || F == QType::GGML_Q5_K) {
        constexpr int kBytes  = F == QType::GGML_Q4_K ? 144 : 176;
        const std::uint8_t* b = row + (s / 8) * kBytes;
        const int sb = s % 8, ch = sb / 2, shift = 4 * (sb % 2);
        o.d    = static_cast<std::uint16_t>(ld_u16(b));
        o.dmin = static_cast<std::uint16_t>(ld_u16(b + 2));
        scale_min_k4(sb, b + 4, o.sub, o.min);
        const std::uint8_t* qs = b + (F == QType::GGML_Q4_K ? 16 : 48) + 32 * ch + 8 * p;
        std::uint32_t v[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            v[h] = (ld_u32_a4(qs + 4 * h) >> shift) & 0x0F0F0F0FU;
            if constexpr (F == QType::GGML_Q5_K) { v[h] |= ((ld_u32_a4(b + 16 + 8 * p + 4 * h) >> sb) & 0x01010101U) << 4; }
        }
        o.w0 = static_cast<int>(v[0]);
        o.w1 = static_cast<int>(v[1]);
    } else if constexpr (F == QType::GGML_Q6_K) {
        const std::uint8_t* b = row + (s / 8) * 210;
        const int sb = s % 8, half = sb / 4, quarter = sb % 4;
        o.d = static_cast<std::uint16_t>(ld_u16(b + 208));
        const std::uint8_t* ql = b + 64 * half + 32 * (quarter & 1) + 8 * p;
        const std::uint8_t* qh = b + 128 + 32 * half + 8 * p;
        const int lshift = quarter >= 2 ? 4 : 0, hshift = 2 * quarter;
        o.sub = static_cast<std::int8_t>(b[192 + 8 * half + 2 * quarter + p / 2]);
        int w[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const std::uint32_t lo = (ld_u32_a2(ql + 4 * h) >> lshift) & 0x0F0F0F0FU;
            const std::uint32_t hi = ((ld_u32_a2(qh + 4 * h) >> hshift) & 0x03030303U) << 4;
            w[h] = sub_bytes4(lo | hi, 0x20202020U);
        }
        o.w0 = w[0];
        o.w1 = w[1];
    } else if constexpr (F == QType::GGML_IQ2_S) {
        const std::uint8_t* b = row + (s / 8) * 82;
        const int ib = s % 8;
        o.d = static_cast<std::uint16_t>(ld_u16(b));
        const int qh = b[66 + ib], sc = b[74 + ib];
        o.sub = 2 * (p < 2 ? sc & 0xF : sc >> 4) + 1;
        const int index = b[2 + 4 * ib + p] | ((qh << (8 - 2 * p)) & 0x300);
        const std::uint64_t grid = __ldg(reinterpret_cast<const unsigned long long*>(d_iq2s_grid) + index);
        const unsigned sign      = b[34 + 4 * ib + p];
        o.w0 = apply_signs4(static_cast<std::uint32_t>(grid), sign & 0xFU);
        o.w1 = apply_signs4(static_cast<std::uint32_t>(grid >> 32), sign >> 4);
    } else if constexpr (F == QType::GGML_IQ3_XXS) {
        const std::uint8_t* b = row + (s / 8) * 98;
        const int ib = s % 8;
        o.d = static_cast<std::uint16_t>(ld_u16(b));
        const std::uint32_t aux = ld_u32_a2(b + 66 + 4 * ib);
        o.sub = 2 * static_cast<int>(aux >> 28) + 1;
        const std::uint32_t qs = ld_u16(b + 2 + 8 * ib + 2 * p);
        const unsigned sign    = __ldg(d_ksigns_iq2xs + ((aux >> (7 * p)) & 127));
        o.w0 = apply_signs4(__ldg(d_iq3xxs_grid + (qs & 0xFF)), sign & 0xFU);
        o.w1 = apply_signs4(__ldg(d_iq3xxs_grid + (qs >> 8)), sign >> 4);
    } else if constexpr (F == QType::GGML_IQ3_S) {
        const std::uint8_t* b = row + (s / 8) * 110;
        const int sb = s % 8;
        o.d = static_cast<std::uint16_t>(ld_u16(b));
        const int qh = b[66 + sb];
        o.sub = 2 * ((b[106 + sb / 2] >> (4 * (sb % 2))) & 0xF) + 1;
        const std::uint32_t qs = ld_u16(b + 2 + 8 * sb + 2 * p);
        const int i1 = (qs & 0xFF) | ((qh << (8 - 2 * p)) & 256);
        const int i2 = (qs >> 8) | ((qh << (7 - 2 * p)) & 256);
        const unsigned sign = b[74 + 4 * sb + p];
        o.w0 = apply_signs4(__ldg(d_iq3s_grid + i1), sign & 0xFU);
        o.w1 = apply_signs4(__ldg(d_iq3s_grid + i2), sign >> 4);
    }
    return o;
}

// ------------------------------------------------------------------------------------- terms

// The decoded scale words of a slice, applied once per column.
struct SliceScale {
    long long mant;
    int shift;
    long long min_mant; // signed, 0 without a min term
    int min_shift;
};

__device__ __forceinline__ SliceScale slice_scale(const Slice& w) {
    SliceScale o;
    std::int64_t m;
    bool negative;
    f16_parts(w.d, m, o.shift, negative);
    o.mant = negative ? -m : m;
    o.min_mant  = 0;
    o.min_shift = 0;
    if (w.min != 0) {
        f16_parts(w.dmin, m, o.min_shift, negative);
        // The min term is subtracted: R -= dmin * min * sum(X).
        o.min_mant = (negative ? m : -m) * w.min;
    }
    return o;
}

// The dot of a slice's eight code words with one limb plane's eight words (two 16-element halves).
__device__ __forceinline__ int limb_dot(const Slice& w, const int4& l0, const int4& l1) {
    const int p0 = __dp4a(w.q[3], l0.w, __dp4a(w.q[2], l0.z, __dp4a(w.q[1], l0.y, __dp4a(w.q[0], l0.x, 0))));
    const int p1 = __dp4a(w.q[7], l1.w, __dp4a(w.q[6], l1.z, __dp4a(w.q[5], l1.y, __dp4a(w.q[4], l1.x, 0))));
    return w.sub0 * p0 + w.sub1 * p1;
}

// R += slice term for one encoded column slice (`limbs`: its kSliceWords words, 16-byte aligned, in
// global or shared memory) and its X sum (read only with a min term).
__device__ __forceinline__ void add_slice(I128& r, const Slice& w, const SliceScale& sc, const std::int32_t* limbs,
                                          int xsum) {
    const int4* l4 = reinterpret_cast<const int4*>(limbs);
    long long p = limb_dot(w, l4[4], l4[5]);
    p += static_cast<long long>(limb_dot(w, l4[2], l4[3])) << 8;
    p += static_cast<long long>(limb_dot(w, l4[0], l4[1])) << 16;
    add_shifted(r, sc.mant * p, sc.shift);
    if (sc.min_mant != 0) { add_shifted(r, sc.min_mant * xsum, sc.min_shift); }
}

} // namespace infernix::ggml::fast
