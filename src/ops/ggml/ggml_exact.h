#pragma once

// Exact arithmetic over GGUF (ggml) block formats (docs/maintainer/qwen3_8-flash-next-design.md
// §16.2.2). Shared by the GPU kernels and the CPU engine: every function returns identical bits on both.
//
// A weight is kappa * (d * sub * code - dmin * min_sub): an integer code, an integer sub-scale, a
// binary16 block scale (and a second term for Q4_K/Q5_K) and a power-of-two format constant. An
// activation column is the A16 encoding of design §16.2.1 (ops::canon::a16_value): integers X with
// v = X * 2^(emax - 147). With d = mant * 2^(e - 25) (e >= 1; subnormals take e = 1), one output is
//
//   R = sum over 8-element chunks of mant(d) * sub * (sum code * X) * 2^(e(d) - 1)
//       - mant(dmin) * min_sub * (sum X) * 2^(e(dmin) - 1)              exact, int128
//   y = fl32(R) * 2^(log2 kappa + emax - 171)                           two exact power-of-two steps
//
// so the sums are exact in any order and one output's bits depend on nothing but its row and
// column. Chunk sums fit int32 (|code| <= 127, |X| < 2^21, 8 terms), chunk terms int64 (< 2^49)
// and R int128 (< 2^89 for K <= 8192).

#include "core/ggml_format.h"
#include "core/weight.h"
#include "ops/common/canonical_math.h"
#include "ops/common/exact_combine.h"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#    define INFERNIX_GGML_HD __host__ __device__ __forceinline__
#    include "ops/ggml/ggml_tables_device.cuh"
#else
#    define INFERNIX_GGML_HD inline
#endif
#include "ops/ggml/ggml_tables.h"

#if defined(__CUDA_ARCH__)
#    define INFERNIX_GGML_TABLE(name) ::infernix::ggml::d_##name
#else
#    define INFERNIX_GGML_TABLE(name) ::infernix::ggml::k_##name
#endif

namespace infernix::ggml {

// --------------------------------------------- int128 (ops/common/exact_combine.h, shared)

using exact::add;
using exact::add_shifted;
using exact::I128;
using exact::scale_pow2;
using exact::to_f32_rn;

// log2 of a format's kappa.
INFERNIX_GGML_HD int kappa_log2(QType q) {
    return q == QType::GGML_IQ2_S ? -3 : q == QType::GGML_IQ3_XXS ? -2 : 0;
}

// The binary32 output of R for a column with A16 exponent emax (canonical NaN for a non-finite one).
INFERNIX_GGML_HD float output_f32(const I128& r, int emax, QType q) {
    if (emax == ops::canon::kA16NonFinite) { return ops::canon::f32_from_bits(ops::canon::kCanonicalNan); }
    return scale_pow2(to_f32_rn(r), kappa_log2(q) + emax - 171);
}

INFERNIX_GGML_HD std::uint16_t output_bf16(const I128& r, int emax, QType q) {
    if (emax == ops::canon::kA16NonFinite) { return 0x7FC0U; }
    return ops::canon::f32_to_bf16_rn(output_f32(r, emax, q));
}

// ----------------------------------------------------------------------------------- chunks

// Eight consecutive elements of a block: codes, their shared sub-scale and the block's scale
// words (Q4_K/Q5_K: the min term too).
struct Chunk {
    std::int32_t code[8];
    std::int32_t sub;
    std::uint16_t d;
    std::int32_t min_sub;
    std::uint16_t dmin;
};

INFERNIX_GGML_HD std::uint16_t u16_at(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

INFERNIX_GGML_HD std::uint32_t u32_at(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

// Q4_K/Q5_K get_scale_min_k4.
INFERNIX_GGML_HD void scale_min_k4(int j, const std::uint8_t* q, std::int32_t& sc, std::int32_t& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

INFERNIX_GGML_HD void apply_signs(std::int32_t (&code)[8], std::uint8_t signs) {
    for (int j = 0; j < 8; ++j) {
        if ((signs >> j) & 1U) { code[j] = -code[j]; }
    }
}

// Elements [8 * part, 8 * part + 8) of the block at `b`.
template <QType F>
INFERNIX_GGML_HD Chunk chunk(const std::uint8_t* b, int part) {
    Chunk c{};
    c.sub     = 1;
    c.min_sub = 0;
    c.dmin    = 0;
    if constexpr (F == QType::GGML_Q2_0) {
        c.d = u16_at(b);
        const std::uint8_t q0 = b[2 + 2 * part], q1 = b[3 + 2 * part];
        for (int j = 0; j < 4; ++j) {
            c.code[j]     = ((q0 >> (2 * j)) & 3) - 1;
            c.code[j + 4] = ((q1 >> (2 * j)) & 3) - 1;
        }
    } else if constexpr (F == QType::GGML_IQ4_NL) {
        c.d = u16_at(b);
        const std::uint8_t* qs = b + 2 + 8 * (part & 1);
        const int shift = part >= 2 ? 4 : 0;
        for (int j = 0; j < 8; ++j) { c.code[j] = INFERNIX_GGML_TABLE(kvalues_iq4nl)[(qs[j] >> shift) & 0xF]; }
    } else if constexpr (F == QType::GGML_IQ4_XS) {
        c.d = u16_at(b);
        const int ib = part / 4, p = part % 4;
        const std::uint16_t scales_h = u16_at(b + 2);
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
        c.sub = ls - 32;
        const std::uint8_t* qs = b + 8 + 16 * ib + 8 * (p & 1);
        const int shift = p >= 2 ? 4 : 0;
        for (int j = 0; j < 8; ++j) { c.code[j] = INFERNIX_GGML_TABLE(kvalues_iq4nl)[(qs[j] >> shift) & 0xF]; }
    } else if constexpr (F == QType::GGML_Q4_K || F == QType::GGML_Q5_K) {
        c.d    = u16_at(b);
        c.dmin = u16_at(b + 2);
        const int e = 8 * part, ch = e / 64, l = e % 64, sb = 2 * ch + (l >= 32 ? 1 : 0), li = l % 32;
        scale_min_k4(sb, b + 4, c.sub, c.min_sub);
        const std::uint8_t* qs = b + (F == QType::GGML_Q4_K ? 16 : 48) + 32 * ch + li;
        const int shift = l >= 32 ? 4 : 0;
        for (int j = 0; j < 8; ++j) {
            int q = (qs[j] >> shift) & 0xF;
            if constexpr (F == QType::GGML_Q5_K) { q += ((b[16 + li + j] >> sb) & 1) << 4; }
            c.code[j] = q;
        }
    } else if constexpr (F == QType::GGML_Q6_K) {
        c.d = u16_at(b + 208);
        const int e = 8 * part, half = e / 128, r = e % 128, quarter = r / 32, l = r % 32;
        const std::uint8_t* ql = b + 64 * half + (quarter & 1) * 32 + l;
        const std::uint8_t* qh = b + 128 + 32 * half + l;
        const int lshift = quarter >= 2 ? 4 : 0, hshift = 2 * quarter;
        c.sub = static_cast<std::int8_t>(b[192 + 8 * half + l / 16 + 2 * quarter]);
        for (int j = 0; j < 8; ++j) {
            c.code[j] = (((ql[j] >> lshift) & 0xF) | (((qh[j] >> hshift) & 3) << 4)) - 32;
        }
    } else if constexpr (F == QType::GGML_IQ2_S) {
        c.d = u16_at(b);
        const int ib = part / 4, l = part % 4;
        const int index = b[2 + 4 * ib + l] | ((b[66 + ib] << (8 - 2 * l)) & 0x300);
        const std::uint64_t grid = INFERNIX_GGML_TABLE(iq2s_grid)[index];
        for (int j = 0; j < 8; ++j) { c.code[j] = static_cast<std::int32_t>((grid >> (8 * j)) & 0xFF); }
        apply_signs(c.code, b[34 + 4 * ib + l]);
        c.sub = 2 * ((b[74 + ib] >> (4 * (l / 2))) & 0xF) + 1;
    } else if constexpr (F == QType::GGML_IQ3_XXS) {
        c.d = u16_at(b);
        const int ib = part / 4, l = part % 4;
        const std::uint32_t aux = u32_at(b + 66 + 4 * ib);
        const std::uint32_t g1 = INFERNIX_GGML_TABLE(iq3xxs_grid)[b[2 + 8 * ib + 2 * l]];
        const std::uint32_t g2 = INFERNIX_GGML_TABLE(iq3xxs_grid)[b[2 + 8 * ib + 2 * l + 1]];
        for (int j = 0; j < 4; ++j) {
            c.code[j]     = static_cast<std::int32_t>((g1 >> (8 * j)) & 0xFF);
            c.code[j + 4] = static_cast<std::int32_t>((g2 >> (8 * j)) & 0xFF);
        }
        apply_signs(c.code, INFERNIX_GGML_TABLE(ksigns_iq2xs)[(aux >> (7 * l)) & 127]);
        c.sub = 2 * static_cast<std::int32_t>(aux >> 28) + 1;
    } else if constexpr (F == QType::GGML_IQ3_S) {
        c.d = u16_at(b);
        const int sb = part / 4, l = part % 4;
        const int qh = b[66 + sb];
        const int i1 = b[2 + 8 * sb + 2 * l] | ((qh << (8 - 2 * l)) & 256);
        const int i2 = b[2 + 8 * sb + 2 * l + 1] | ((qh << (7 - 2 * l)) & 256);
        const std::uint32_t g1 = INFERNIX_GGML_TABLE(iq3s_grid)[i1];
        const std::uint32_t g2 = INFERNIX_GGML_TABLE(iq3s_grid)[i2];
        for (int j = 0; j < 4; ++j) {
            c.code[j]     = static_cast<std::int32_t>((g1 >> (8 * j)) & 0xFF);
            c.code[j + 4] = static_cast<std::int32_t>((g2 >> (8 * j)) & 0xFF);
        }
        apply_signs(c.code, b[74 + 4 * sb + l]);
        c.sub = 2 * ((b[106 + sb / 2] >> (4 * (sb % 2))) & 0xF) + 1;
    }
    return c;
}

// The fp16 word's integer mantissa and its shift e - 1 (e = max(exponent field, 1)).
INFERNIX_GGML_HD void f16_parts(std::uint16_t word, std::int64_t& mant, int& shift, bool& negative) {
    const int field = (word >> 10) & 0x1F;
    mant            = field == 0 ? (word & 0x3FF) : (0x400 | (word & 0x3FF));
    shift           = (field == 0 ? 1 : field) - 1;
    negative        = (word & 0x8000U) != 0;
}

// Adds one chunk's terms for an activation slice x[0..8) (A16 integers) into R.
INFERNIX_GGML_HD void accumulate(I128& r, const Chunk& c, const std::int32_t* x) {
    std::int32_t dot = 0, xsum = 0;
    for (int j = 0; j < 8; ++j) {
        dot += c.code[j] * x[j];
        xsum += x[j];
    }
    std::int64_t mant;
    int shift;
    bool negative;
    f16_parts(c.d, mant, shift, negative);
    std::int64_t term = mant * c.sub * static_cast<std::int64_t>(dot);
    add_shifted(r, negative ? -term : term, shift);
    if (c.min_sub != 0) {
        f16_parts(c.dmin, mant, shift, negative);
        const std::int64_t m = mant * c.min_sub * static_cast<std::int64_t>(xsum);
        add_shifted(r, negative ? m : -m, shift);
    }
}

// Rows of a ggml matrix: `row_bytes` bytes of whole blocks.
template <QType F>
INFERNIX_GGML_HD Chunk row_chunk(const std::uint8_t* row, int index) {
    constexpr int kParts      = static_cast<int>(ggml_block_elems(F)) / 8;
    constexpr int kBlockBytes = static_cast<int>(ggml_block_bytes(F));
    return chunk<F>(row + static_cast<std::size_t>(index / kParts) * kBlockBytes, index % kParts);
}

// The exact R of one row against one encoded column of K elements (reference form; kernels split
// the chunk loop across threads and add their partial R).
template <QType F>
INFERNIX_GGML_HD I128 row_dot(const std::uint8_t* row, const std::int32_t* x, int k) {
    I128 r;
    for (int i = 0; i < k / 8; ++i) { accumulate(r, row_chunk<F>(row, i), x + 8 * i); }
    return r;
}

} // namespace infernix::ggml

#undef INFERNIX_GGML_HD
