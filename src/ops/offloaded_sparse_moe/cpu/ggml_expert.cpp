#include "ops/offloaded_sparse_moe/cpu/ggml_expert.h"

#include "core/ggml_format.h"
#include "ops/ggml/ggml_a8.h"
#include "ops/offloaded_sparse_moe/cpu/fp8_expert.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "ops/offloaded_sparse_moe/cpu/x86_mad.h"

#if defined(INFERNIX_CPU_X86)
#    define INFERNIX_GGML_X86 1
#endif

namespace infernix::ops::offloaded_moe {
namespace {

// Slices per block (the period of the d words).
constexpr int block_slices(QType q) { return q == QType::GGML_IQ4_NL ? 1 : q == QType::GGML_Q2_0 ? 2 : 8; }

// The slice's X values from its limbs.
void limbs_to_x(const CpuLimbSlice& l, std::int32_t (&x)[32]) {
    for (int e = 0; e < 32; ++e) { x[e] = static_cast<std::int32_t>(l.hi[e]) * 2048 + l.lo[e]; }
}

// ------------------------------------------------------------------------------------- scalar

// The reference form: ggml_exact.h's chunks and terms, chunk by chunk.
template <QType F>
void row_sums_scalar(const std::uint8_t* matrix, int row_bytes, int slices, int row_begin, int row_end,
                     const CpuLimbSlice* const* columns, int ncols, ggml::I128* r) {
    constexpr int kBlockSlices = block_slices(F);
    constexpr int kBlockBytes  = static_cast<int>(ggml_block_bytes(F));
    for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t* w = matrix + static_cast<std::size_t>(row) * row_bytes;
        for (int c = 0; c < ncols; ++c) {
            ggml::I128 acc;
            for (int s = 0; s < slices; ++s) {
                std::int32_t x[32];
                limbs_to_x(columns[c][s], x);
                const std::uint8_t* block = w + static_cast<std::size_t>(s / kBlockSlices) * kBlockBytes;
                for (int p = 0; p < 4; ++p) {
                    ggml::accumulate(acc, ggml::chunk<F>(block, 4 * (s % kBlockSlices) + p), x + 8 * p);
                }
            }
            r[static_cast<std::size_t>(row - row_begin) * ncols + c] = acc;
        }
    }
}

#if defined(INFERNIX_GGML_X86)

// ------------------------------------------------------------------------- AVX2 slice decoding

struct Decoded {
    __m256i q;      // 32 signed byte codes, element e in byte e
    int sub0, sub1; // elements 0-15, 16-31
    std::uint16_t d;
};

inline std::uint16_t u16(const std::uint8_t* p) {
    std::uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

inline std::uint32_t u32(const std::uint8_t* p) {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

// 0xFF in byte e where bit e of `bits` is set.
INFERNIX_TARGET("avx2")
inline __m256i sign_mask(std::uint32_t bits) {
    const __m256i v    = _mm256_set1_epi32(static_cast<int>(bits));
    const __m256i shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
                                          3, 3, 3, 3, 3);
    const __m256i bit  = _mm256_set1_epi64x(static_cast<long long>(0x8040201008040201ULL));
    return _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(v, shuf), bit), bit);
}

INFERNIX_TARGET("avx2")
inline __m256i apply_signs(__m256i g, std::uint32_t bits) {
    const __m256i m = sign_mask(bits);
    return _mm256_sub_epi8(_mm256_xor_si256(g, m), m);
}

// kvalues_iq4nl of 16 bytes' low nibbles (elements 0-15) and high nibbles (16-31).
INFERNIX_TARGET("avx2")
inline __m256i iq4nl_codes(const std::uint8_t* qs) {
    const __m128i table = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ggml::k_kvalues_iq4nl));
    const __m128i m4    = _mm_set1_epi8(0x0F);
    const __m128i q     = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
    const __m128i lo    = _mm_shuffle_epi8(table, _mm_and_si128(q, m4));
    const __m128i hi    = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
    return _mm256_set_m128i(hi, lo);
}

// The IQ grid entries are read by scalar loads, not vpgather (microcoded and slow on hybrid CPUs'
// efficiency cores): scalar loads measured 0.9 % faster end to end than vpgather (RTX 5090, IQ3_S
// artifact, MTP 4 + n-gram 7 at 4K: 202.0 against 200.2 tok/s, 2026-10-16), and gathers on P-cores only
// gained nothing end to end (IQ3_S DFlash2 decode after 8K, 2026-10-28, candidates §27 CX-03).
template <QType F>
INFERNIX_TARGET("avx2")
inline Decoded decode(const std::uint8_t* row, int s) {
    Decoded o;
    if constexpr (F == QType::GGML_IQ4_NL) {
        const std::uint8_t* b = row + static_cast<std::size_t>(s) * 18;
        o.d    = u16(b);
        o.sub0 = o.sub1 = 1;
        o.q    = iq4nl_codes(b + 2);
    } else if constexpr (F == QType::GGML_IQ4_XS) {
        const std::uint8_t* b = row + static_cast<std::size_t>(s / 8) * 136;
        const int ib          = s % 8;
        o.d                   = u16(b);
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((u16(b + 2) >> (2 * ib)) & 3) << 4);
        o.sub0 = o.sub1 = ls - 32;
        o.q             = iq4nl_codes(b + 8 + 16 * ib);
    } else if constexpr (F == QType::GGML_Q2_0) {
        const std::uint8_t* b = row + static_cast<std::size_t>(s / 2) * 18;
        o.d    = u16(b);
        o.sub0 = o.sub1 = 1;
        std::uint64_t q;
        std::memcpy(&q, b + 2 + 8 * (s % 2), 8);
        // Element 4i + j is bits 2j of byte i: t_j holds them in byte i, then the bytes interleave.
        const __m128i m   = _mm_set1_epi8(3);
        const __m128i v   = _mm_cvtsi64_si128(static_cast<long long>(q));
        const __m128i t0  = _mm_and_si128(v, m);
        const __m128i t1  = _mm_and_si128(_mm_srli_epi64(v, 2), m);
        const __m128i t2  = _mm_and_si128(_mm_srli_epi64(v, 4), m);
        const __m128i t3  = _mm_and_si128(_mm_srli_epi64(v, 6), m);
        const __m128i t01 = _mm_unpacklo_epi8(t0, t1);
        const __m128i t23 = _mm_unpacklo_epi8(t2, t3);
        const __m256i u   = _mm256_set_m128i(_mm_unpackhi_epi16(t01, t23), _mm_unpacklo_epi16(t01, t23));
        o.q               = _mm256_sub_epi8(u, _mm256_set1_epi8(1));
    } else if constexpr (F == QType::GGML_IQ3_S) {
        const std::uint8_t* b = row + static_cast<std::size_t>(s / 8) * 110;
        const int sb          = s % 8;
        o.d                   = u16(b);
        o.sub0 = o.sub1        = 2 * ((b[106 + sb / 2] >> (4 * (sb % 2))) & 0xF) + 1;
        // Entry k: qs[k] | bit k of qh << 8.
        const std::uint8_t* qs = b + 2 + 8 * sb;
        const unsigned qh      = b[66 + sb];
        const auto e = [&](int k) { return static_cast<int>(ggml::k_iq3s_grid[qs[k] | (((qh >> k) & 1U) << 8)]); };
        o.q = apply_signs(_mm256_setr_epi32(e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7)), u32(b + 74 + 4 * sb));
    } else if constexpr (F == QType::GGML_IQ3_XXS) {
        const std::uint8_t* b = row + static_cast<std::size_t>(s / 8) * 98;
        const int ib          = s % 8;
        o.d                   = u16(b);
        const std::uint32_t aux = u32(b + 66 + 4 * ib);
        o.sub0 = o.sub1 = 2 * static_cast<int>(aux >> 28) + 1;
        const std::uint8_t* qs = b + 2 + 8 * ib;
        const auto e           = [&](int k) { return static_cast<int>(ggml::k_iq3xxs_grid[qs[k]]); };
        const __m256i g        = _mm256_setr_epi32(e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7));
        const std::uint32_t signs = static_cast<std::uint32_t>(ggml::k_ksigns_iq2xs[aux & 127]) |
                                    static_cast<std::uint32_t>(ggml::k_ksigns_iq2xs[(aux >> 7) & 127]) << 8 |
                                    static_cast<std::uint32_t>(ggml::k_ksigns_iq2xs[(aux >> 14) & 127]) << 16 |
                                    static_cast<std::uint32_t>(ggml::k_ksigns_iq2xs[(aux >> 21) & 127]) << 24;
        o.q = apply_signs(g, signs);
    } else if constexpr (F == QType::GGML_IQ2_S) {
        const std::uint8_t* b = row + static_cast<std::size_t>(s / 8) * 82;
        const int ib          = s % 8;
        o.d                   = u16(b);
        const int sc          = b[74 + ib];
        o.sub0                = 2 * (sc & 0xF) + 1;
        o.sub1                = 2 * (sc >> 4) + 1;
        const std::uint8_t* qs = b + 2 + 4 * ib;
        const unsigned qh      = b[66 + ib];
        const int i0 = qs[0] | ((qh << 8) & 0x300), i1 = qs[1] | ((qh << 6) & 0x300);
        const int i2 = qs[2] | ((qh << 4) & 0x300), i3 = qs[3] | ((qh << 2) & 0x300);
        const __m256i g =
            _mm256_setr_epi64x(static_cast<long long>(ggml::k_iq2s_grid[i0]), static_cast<long long>(ggml::k_iq2s_grid[i1]),
                               static_cast<long long>(ggml::k_iq2s_grid[i2]), static_cast<long long>(ggml::k_iq2s_grid[i3]));
        o.q = apply_signs(g, u32(b + 34 + 4 * ib));
    }
    return o;
}

// ------------------------------------------------------------------------ AVX2 dot products
//
// The sub-scales go into the codes (|code * sub| <= 127 * 32 = 4064, a signed 16-bit word), so a
// column's slice is four 16-bit multiply-adds (vpmaddwd, or vpdpwssd with VNNI) against its two
// limb planes: per lane |w * x| < 2^24 per pair, below 2^29 over the four slices of a unit.

struct Weights {
    __m256i lo, hi; // code * sub as int16: elements 0-15, 16-31
};

INFERNIX_TARGET("avx2")
inline Weights scaled(const Decoded& d) {
    return {_mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(d.q)), _mm256_set1_epi16(static_cast<short>(d.sub0))),
            _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(d.q, 1)), _mm256_set1_epi16(static_cast<short>(d.sub1)))};
}

using x86::Avx2Mad;
using x86::AvxVnniMad;
using x86::hsum8;

// Slices per unit (one int32 accumulation), and units per block.
template <QType F>
inline constexpr int kUnitSlices = F == QType::GGML_IQ4_NL ? 1 : F == QType::GGML_Q2_0 ? 2 : 4;


// ------------------------------------------------------------------- AVX-512 (512-bit vectors)
//
// A slice's 32 weights (code * sub, int16) are one zmm and each limb plane of a column one zmm, so
// a column's slice is two vpdpwssd (the 256-bit routes take four), and all of a call's columns
// (up to kMaxColumns, 16 accumulators of the 32 zmm registers) share each decoded slice: a call
// decodes every slice once (the 256-bit routes take columns four at a time). A unit's 16 lanes
// fold to 8 (lane i + lane i + 8: per lane at most 16 products, the 256-bit routes' bound) before
// the same batch reduction, so every block's P is the same exact integer: the same bits.
INFERNIX_TARGET("avx2,avx512f,avx512bw")
inline __m256i fold16(__m512i v) { return _mm256_add_epi32(_mm512_castsi512_si256(v), _mm512_extracti64x4_epi64(v, 1)); }

template <QType F, int NC>
INFERNIX_TARGET("avx2,avx512f,avx512bw,avx512vl,avx512vnni")
void row_sums_avx512_n(const std::uint8_t* matrix, int row_bytes, int slices, int row_begin, int row_end,
                       const CpuLimbSlice* const* columns, int ncols, ggml::I128* r) {
    constexpr int kUnit          = kUnitSlices<F>;
    constexpr int kUnitsPerBlock = block_slices(F) / kUnit;
    static_assert(8 % kUnitsPerBlock == 0, "a batch holds whole blocks");
    const int units = slices / kUnit;
    for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t* w = matrix + static_cast<std::size_t>(row) * row_bytes;
        ggml::I128 out[NC];
        __m256i batch_lo[NC][8], batch_hi[NC][8];
        std::uint16_t batch_d[8];
        int n = 0;
        for (int u = 0; u < units; ++u) {
            __m512i lo[NC], hi[NC];
            for (int k = 0; k < NC; ++k) { lo[k] = hi[k] = _mm512_setzero_si512(); }
            std::uint16_t d = 0;
            for (int j = 0; j < kUnit; ++j) {
                const int s      = u * kUnit + j;
                const Decoded dw = decode<F>(w, s);
                __m512i sub;
                if constexpr (F == QType::GGML_IQ2_S) { // elements 0-15 and 16-31 have their own sub-scales
                    sub = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16(static_cast<short>(dw.sub0))),
                                             _mm256_set1_epi16(static_cast<short>(dw.sub1)), 1);
                } else {
                    sub = _mm512_set1_epi16(static_cast<short>(dw.sub0));
                }
                const __m512i sw = _mm512_mullo_epi16(_mm512_cvtepi8_epi16(dw.q), sub);
                d                = dw.d;
                for (int k = 0; k < NC; ++k) {
                    const CpuLimbSlice& l = columns[k][s];
                    lo[k] = _mm512_dpwssd_epi32(lo[k], sw, _mm512_loadu_si512(l.lo));
                    hi[k] = _mm512_dpwssd_epi32(hi[k], sw, _mm512_loadu_si512(l.hi));
                }
            }
            for (int k = 0; k < NC; ++k) {
                batch_lo[k][n] = fold16(lo[k]);
                batch_hi[k][n] = fold16(hi[k]);
            }
            batch_d[n++] = d;
            if (n == 8 || u + 1 == units) {
                for (int k = 0; k < NC; ++k) {
                    for (int b = n; b < 8; ++b) { batch_lo[k][b] = batch_hi[k][b] = _mm256_setzero_si256(); }
                    alignas(32) std::int32_t sl[8], sh[8];
                    _mm256_store_si256(reinterpret_cast<__m256i*>(sl), hsum8(batch_lo[k]));
                    _mm256_store_si256(reinterpret_cast<__m256i*>(sh), hsum8(batch_hi[k]));
                    for (int b = 0; b < n; b += kUnitsPerBlock) {
                        std::int64_t p = 0;
                        for (int i = 0; i < kUnitsPerBlock; ++i) { p += static_cast<std::int64_t>(sh[b + i]) * 2048 + sl[b + i]; }
                        std::int64_t mant;
                        int shift;
                        bool negative;
                        ggml::f16_parts(batch_d[b], mant, shift, negative);
                        ggml::add_shifted(out[k], (negative ? -mant : mant) * p, shift);
                    }
                }
                n = 0;
            }
        }
        for (int k = 0; k < NC; ++k) { r[static_cast<std::size_t>(row - row_begin) * ncols + k] = out[k]; }
    }
}

template <QType F>
void row_sums_avx512(const std::uint8_t* matrix, int row_bytes, int slices, int row_begin, int row_end,
                     const CpuLimbSlice* const* columns, int ncols, ggml::I128* r) {
    static_assert(kMaxColumns == 8, "one pass serves up to eight columns");
    switch (ncols) {
#    define INFERNIX_GGML_NC(n)                                                                                        \
    case n: row_sums_avx512_n<F, n>(matrix, row_bytes, slices, row_begin, row_end, columns, ncols, r); return;
        INFERNIX_GGML_NC(1)
        INFERNIX_GGML_NC(2)
        INFERNIX_GGML_NC(3)
        INFERNIX_GGML_NC(4)
        INFERNIX_GGML_NC(5)
        INFERNIX_GGML_NC(6)
        INFERNIX_GGML_NC(7)
    default: row_sums_avx512_n<F, 8>(matrix, row_bytes, slices, row_begin, row_end, columns, ncols, r); return;
#    undef INFERNIX_GGML_NC
    }
}

#endif // INFERNIX_GGML_X86

// The ISAs without the tiled rows (AVX-512, scalar); the 256-bit ISAs take ggml_rows.
template <QType F>
void row_sums_format(CpuIsa isa, const std::uint8_t* matrix, int row_bytes, int slices, int row_begin, int row_end,
                     const CpuLimbSlice* const* columns, int ncols, ggml::I128* r) {
    switch (isa) {
#if defined(INFERNIX_GGML_X86)
    case CpuIsa::kAvx512Vnni:
        row_sums_avx512<F>(matrix, row_bytes, slices, row_begin, row_end, columns, ncols, r);
        return;
#endif
    default:
        row_sums_scalar<F>(matrix, row_bytes, slices, row_begin, row_end, columns, ncols, r);
        return;
    }
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("offloaded_moe ggml: ") + message); }
}

// ---------------------------------------------------------------- AVX2 and AVX-VNNI tiled rows
//
// The unit and batch arithmetic of the AVX-512 rows (each unit's int32 lanes, eight units' hsum8, each
// block's exact int64 P into I128), so every output is the same integer on every ISA:
// - a unit's kUnit slices of R matrices (R = 2: a gate row and its up row) are decoded once into a
//   tile, then every column (1-8) runs its multiply-adds from the tile, sharing its x loads across
//   the R rows: no second decode pass at 5-8 columns, and two independent decode chains;
// - the block pointer and slice index come from unsigned unit arithmetic, not a signed s / 8 per slice;
// - IQ3_XXS signs are bit arithmetic: ksigns[i] = i | parity(i) << 7 for each 7-bit field.
// Measured 2026-10-28 (candidates §27 CX-04, CX-10; RTX 5090 + i9-13900K): IQ3_S DFlash2 decode after 8K
// +0.38..+0.58 % against the rows that decoded each slice per four columns and per matrix.

#if defined(INFERNIX_GGML_X86)

// The four 7-bit sign fields of an IQ3_XXS aux word as ksigns_iq2xs bytes (field | parity << 7).
inline std::uint32_t iq3xxs_signs(std::uint32_t aux) {
    const std::uint32_t f = (aux & 0x7FU) | ((aux << 1) & 0x7F00U) | ((aux << 2) & 0x7F0000U) | ((aux << 3) & 0x7F000000U);
    std::uint32_t p = f ^ ((f >> 4) & 0x0F0F0F0FU);
    p ^= (p >> 2) & 0x03030303U;
    p ^= (p >> 1) & 0x01010101U;
    return f | ((p & 0x01010101U) << 7);
}

// Slice sb of the block at b (decode<F>'s values).
template <QType F>
INFERNIX_TARGET("avx2")
inline Decoded decode_block(const std::uint8_t* b, unsigned sb) {
    Decoded o;
    if constexpr (F == QType::GGML_IQ3_XXS) {
        o.d                     = u16(b);
        const std::uint32_t aux = u32(b + 66 + 4 * sb);
        o.sub0 = o.sub1        = 2 * static_cast<int>(aux >> 28) + 1;
        const std::uint8_t* qs = b + 2 + 8 * sb;
        const auto e           = [&](int k) { return static_cast<int>(ggml::k_iq3xxs_grid[qs[k]]); };
        o.q = apply_signs(_mm256_setr_epi32(e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7)), iq3xxs_signs(aux));
    } else if constexpr (F == QType::GGML_IQ3_S) {
        o.d                    = u16(b);
        o.sub0 = o.sub1        = 2 * ((b[106 + sb / 2] >> (4 * (sb % 2))) & 0xF) + 1;
        const std::uint8_t* qs = b + 2 + 8 * sb;
        const unsigned qh      = b[66 + sb];
        const auto e = [&](int k) { return static_cast<int>(ggml::k_iq3s_grid[qs[k] | (((qh >> k) & 1U) << 8)]); };
        o.q = apply_signs(_mm256_setr_epi32(e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7)), u32(b + 74 + 4 * sb));
    } else if constexpr (F == QType::GGML_IQ2_S) {
        o.d                    = u16(b);
        const int sc           = b[74 + sb];
        o.sub0                 = 2 * (sc & 0xF) + 1;
        o.sub1                 = 2 * (sc >> 4) + 1;
        const std::uint8_t* qs = b + 2 + 4 * sb;
        const unsigned qh      = b[66 + sb];
        const int i0 = qs[0] | ((qh << 8) & 0x300), i1 = qs[1] | ((qh << 6) & 0x300);
        const int i2 = qs[2] | ((qh << 4) & 0x300), i3 = qs[3] | ((qh << 2) & 0x300);
        const __m256i g =
            _mm256_setr_epi64x(static_cast<long long>(ggml::k_iq2s_grid[i0]), static_cast<long long>(ggml::k_iq2s_grid[i1]),
                               static_cast<long long>(ggml::k_iq2s_grid[i2]), static_cast<long long>(ggml::k_iq2s_grid[i3]));
        o.q = apply_signs(g, u32(b + 34 + 4 * sb));
    } else if constexpr (F == QType::GGML_IQ4_XS) {
        o.d          = u16(b);
        const int ls = ((b[4 + sb / 2] >> (4 * (sb % 2))) & 0xF) | (((u16(b + 2) >> (2 * sb)) & 3) << 4);
        o.sub0 = o.sub1 = ls - 32;
        o.q             = iq4nl_codes(b + 8 + 16 * sb);
    } else if constexpr (F == QType::GGML_IQ4_NL) {
        o.d    = u16(b);
        o.sub0 = o.sub1 = 1;
        o.q    = iq4nl_codes(b + 2);
    } else if constexpr (F == QType::GGML_Q2_0) {
        o = decode<F>(b, static_cast<int>(sb)); // block-relative: decode<Q2_0> indexes 18-byte blocks by s / 2
    }
    return o;
}

// Rows [row_begin, row_end) of R matrices (mats[r], row_bytes apart, `slices` slices) against NC columns:
// out[r][(row - row_begin) * NC + c].
#    define INFERNIX_GGML_TILED_ROWS(Name, Target, Mad)                                                                     \
    template <QType F, int NC, int R>                                                                                    \
    INFERNIX_TARGET(Target)                                                                                              \
void Name(const std::uint8_t* const* mats, int row_bytes, int slices, int row_begin, int row_end,                        \
               const CpuLimbSlice* const* columns, ggml::I128* const* out) {                                             \
    constexpr unsigned kUnit         = kUnitSlices<F>;                                                                   \
    constexpr unsigned kBlockSlices  = block_slices(F);                                                                  \
    constexpr unsigned kBlockBytes   = static_cast<unsigned>(ggml_block_bytes(F));                                       \
    constexpr int kUnitsPerBlock     = static_cast<int>(kBlockSlices / kUnit);                                           \
    static_assert(8 % kUnitsPerBlock == 0, "a batch holds whole blocks");                                                \
    const unsigned units = static_cast<unsigned>(slices) / kUnit;                                                        \
    for (int row = row_begin; row < row_end; ++row) {                                                                    \
        const std::uint8_t* w[R];                                                                                        \
        for (int r = 0; r < R; ++r) { w[r] = mats[r] + static_cast<std::size_t>(row) * static_cast<std::size_t>(row_bytes); } \
        ggml::I128 acc[R][NC];                                                                                           \
        __m256i batch_lo[R][NC][8], batch_hi[R][NC][8];                                                                  \
        std::uint16_t batch_d[R][8];                                                                                     \
        int n = 0;                                                                                                       \
        for (unsigned u = 0; u < units; ++u) {                                                                           \
            const unsigned first = u * kUnit;                                                                            \
            Weights tile[R][kUnit];                                                                                      \
            for (int r = 0; r < R; ++r) {                                                                                \
                const std::uint8_t* block = w[r] + static_cast<std::size_t>(first / kBlockSlices) * kBlockBytes;         \
                for (unsigned j = 0; j < kUnit; ++j) {                                                                   \
                    const Decoded dw = decode_block<F>(block, first % kBlockSlices + j);                                  \
                    tile[r][j]       = scaled(dw);                                                                       \
                    batch_d[r][n]    = dw.d;                                                                             \
                }                                                                                                        \
            }                                                                                                            \
            for (int c = 0; c < NC; ++c) {                                                                               \
                const CpuLimbSlice* col = columns[c] + first;                                                            \
                __m256i lo[R], hi[R];                                                                                    \
                for (int r = 0; r < R; ++r) { lo[r] = hi[r] = _mm256_setzero_si256(); }                                  \
                for (unsigned j = 0; j < kUnit; ++j) {                                                                   \
                    const auto* xl   = reinterpret_cast<const __m256i*>(col[j].lo);                                      \
                    const auto* xh   = reinterpret_cast<const __m256i*>(col[j].hi);                                      \
                    const __m256i l0 = _mm256_load_si256(xl), l1 = _mm256_load_si256(xl + 1);                            \
                    const __m256i h0 = _mm256_load_si256(xh), h1 = _mm256_load_si256(xh + 1);                            \
                    for (int r = 0; r < R; ++r) {                                                                        \
                        lo[r] = Mad::mad(Mad::mad(lo[r], tile[r][j].lo, l0), tile[r][j].hi, l1);                         \
                        hi[r] = Mad::mad(Mad::mad(hi[r], tile[r][j].lo, h0), tile[r][j].hi, h1);                         \
                    }                                                                                                    \
                }                                                                                                        \
                for (int r = 0; r < R; ++r) {                                                                            \
                    batch_lo[r][c][n] = lo[r];                                                                           \
                    batch_hi[r][c][n] = hi[r];                                                                           \
                }                                                                                                        \
            }                                                                                                            \
            ++n;                                                                                                         \
            if (n == 8 || u + 1 == units) {                                                                              \
                for (int r = 0; r < R; ++r) {                                                                            \
                    for (int c = 0; c < NC; ++c) {                                                                       \
                        for (int b = n; b < 8; ++b) { batch_lo[r][c][b] = batch_hi[r][c][b] = _mm256_setzero_si256(); }  \
                        alignas(32) std::int32_t sl[8], sh[8];                                                           \
                        _mm256_store_si256(reinterpret_cast<__m256i*>(sl), hsum8(batch_lo[r][c]));                       \
                        _mm256_store_si256(reinterpret_cast<__m256i*>(sh), hsum8(batch_hi[r][c]));                       \
                        for (int b = 0; b < n; b += kUnitsPerBlock) {                                                    \
                            std::int64_t p = 0;                                                                          \
                            for (int i = 0; i < kUnitsPerBlock; ++i) { p += static_cast<std::int64_t>(sh[b + i]) * 2048 + sl[b + i]; } \
                            std::int64_t mant;                                                                           \
                            int shift;                                                                                   \
                            bool negative;                                                                               \
                            ggml::f16_parts(batch_d[r][b], mant, shift, negative);                                       \
                            ggml::add_shifted(acc[r][c], (negative ? -mant : mant) * p, shift);                          \
                        }                                                                                                \
                    }                                                                                                    \
                }                                                                                                        \
                n = 0;                                                                                                   \
            }                                                                                                            \
        }                                                                                                                \
        for (int r = 0; r < R; ++r) {                                                                                    \
            for (int c = 0; c < NC; ++c) { out[r][static_cast<std::size_t>(row - row_begin) * NC + c] = acc[r][c]; }     \
        }                                                                                                                \
    }                                                                                                                    \
}
INFERNIX_GGML_TILED_ROWS(rows_avx2_n, "avx2", Avx2Mad)
INFERNIX_GGML_TILED_ROWS(rows_avxvnni_n, "avx2,avxvnni", AvxVnniMad)
#    undef INFERNIX_GGML_TILED_ROWS

template <QType F, int R, class Mad>
void rows_mad(const std::uint8_t* const* mats, int row_bytes, int slices, int row_begin, int row_end,
              const CpuLimbSlice* const* columns, int ncols, ggml::I128* const* out) {
    switch (ncols) {
#    define INFERNIX_GGML_TILED_NC(n)                                                                                    \
    case n:                                                                                                            \
        if constexpr (std::is_same_v<Mad, AvxVnniMad>) {                                                               \
            rows_avxvnni_n<F, n, R>(mats, row_bytes, slices, row_begin, row_end, columns, out);                        \
        } else {                                                                                                       \
            rows_avx2_n<F, n, R>(mats, row_bytes, slices, row_begin, row_end, columns, out);                           \
        }                                                                                                              \
        return;
        INFERNIX_GGML_TILED_NC(1)
        INFERNIX_GGML_TILED_NC(2)
        INFERNIX_GGML_TILED_NC(3)
        INFERNIX_GGML_TILED_NC(4)
        INFERNIX_GGML_TILED_NC(5)
        INFERNIX_GGML_TILED_NC(6)
        INFERNIX_GGML_TILED_NC(7)
    default:
        if constexpr (std::is_same_v<Mad, AvxVnniMad>) {
            rows_avxvnni_n<F, 8, R>(mats, row_bytes, slices, row_begin, row_end, columns, out);
        } else {
            rows_avx2_n<F, 8, R>(mats, row_bytes, slices, row_begin, row_end, columns, out);
        }
        return;
#    undef INFERNIX_GGML_TILED_NC
    }
}

// The tiled rows of format F on a 256-bit ISA (false: another ISA, which takes row_sums_format).
template <QType F, int R>
bool rows_256(CpuIsa isa, const std::uint8_t* const* mats, int row_bytes, int slices, int row_begin, int row_end,
              const CpuLimbSlice* const* columns, int ncols, ggml::I128* const* out) {
    if (isa == CpuIsa::kAvxVnni) {
        rows_mad<F, R, AvxVnniMad>(mats, row_bytes, slices, row_begin, row_end, columns, ncols, out);
        return true;
    }
    if (isa == CpuIsa::kAvx2) {
        rows_mad<F, R, Avx2Mad>(mats, row_bytes, slices, row_begin, row_end, columns, ncols, out);
        return true;
    }
    return false;
}

// X of 8 elements v[0..8) of a column with exponent emax (encode_column_avx2).
INFERNIX_TARGET("avx2")
inline __m256i a16_values8(const std::uint16_t* p, int emax) {
    const __m256i em    = _mm256_set1_epi32(emax);
    const __m256i one   = _mm256_set1_epi32(1);
    const __m256i zero  = _mm256_setzero_si256();
    const __m256i vv    = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    const __m256i field = _mm256_and_si256(_mm256_srli_epi32(vv, 7), _mm256_set1_epi32(0xFF));
    const __m256i sub   = _mm256_cmpeq_epi32(field, zero);
    const __m256i m     = _mm256_or_si256(_mm256_and_si256(vv, _mm256_set1_epi32(0x7F)),
                                          _mm256_andnot_si256(sub, _mm256_set1_epi32(0x80)));
    const __m256i d     = _mm256_sub_epi32(em, _mm256_max_epi32(field, one));
    const __m256i q     = _mm256_slli_epi32(m, 13);
    // Round half to even at 2^d: (q + 2^(d-1) - 1 + (q >> d & 1)) >> d; counts past 31 give 0 in
    // vpsllvd/vpsrlvd, which is the scalar d >= 22 case's 0 (q < 2^21); d = 0 keeps q.
    const __m256i half  = _mm256_sllv_epi32(one, _mm256_sub_epi32(d, one));
    const __m256i odd   = _mm256_and_si256(_mm256_srlv_epi32(q, d), one);
    const __m256i r     = _mm256_srlv_epi32(_mm256_add_epi32(_mm256_sub_epi32(_mm256_add_epi32(q, half), one), odd), d);
    const __m256i x     = _mm256_blendv_epi8(r, q, _mm256_cmpeq_epi32(d, zero));
    return _mm256_sign_epi32(x, _mm256_or_si256(_mm256_slli_epi32(vv, 16), one));
}

// canon::a16_column_exponent and a16_value with AVX2 (8 elements per vector, 32-bit lanes), then the
// limb planes: the same integers.
INFERNIX_TARGET("avx2")
int encode_column_avx2(const std::uint16_t* v, int n, CpuLimbSlice* out) {
    __m256i key = _mm256_setzero_si256();
    for (int j = 0; j < n; j += 16) {
        key = _mm256_max_epu16(key, _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(v + j)),
                                                     _mm256_set1_epi16(0x7FFF)));
    }
    alignas(32) std::uint16_t lanes[16];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), key);
    unsigned top = 0;
    for (const std::uint16_t k : lanes) { top = k > top ? k : top; }
    if (top >= 0x7F80U) {
        std::memset(out, 0, sizeof(CpuLimbSlice) * static_cast<std::size_t>(n / 32));
        return canon::kA16NonFinite;
    }
    const int emax = (top >> 7) == 0 ? 1 : static_cast<int>(top >> 7);
    for (int j = 0; j < n; j += 16) {
        const __m256i a = a16_values8(v + j, emax), b = a16_values8(v + j + 8, emax);
        const __m256i lo = _mm256_permute4x64_epi64(
            _mm256_packs_epi32(_mm256_and_si256(a, _mm256_set1_epi32(2047)), _mm256_and_si256(b, _mm256_set1_epi32(2047))), 0xD8);
        const __m256i hi =
            _mm256_permute4x64_epi64(_mm256_packs_epi32(_mm256_srai_epi32(a, 11), _mm256_srai_epi32(b, 11)), 0xD8);
        CpuLimbSlice& l = out[j / 32];
        _mm256_store_si256(reinterpret_cast<__m256i*>(l.lo + j % 32), lo);
        _mm256_store_si256(reinterpret_cast<__m256i*>(l.hi + j % 32), hi);
    }
    return emax;
}

#endif // INFERNIX_GGML_X86

// The tiled rows of `format` (R = 1: one matrix; R = 2: a gate and up pair); false when the ISA has none.
template <int R>
bool ggml_rows(CpuIsa isa, QType format, const std::uint8_t* const* mats, int row_bytes, int slices, int row_begin,
               int row_end, const CpuLimbSlice* const* columns, int ncols, ggml::I128* const* out) {
#if defined(INFERNIX_GGML_X86)
    switch (format) {
#    define INFERNIX_GGML_TILED_FORMAT(Q)                                                                                   \
    case Q:                                                                                                            \
        if constexpr (R == 1 || (Q != QType::GGML_IQ4_NL && Q != QType::GGML_Q2_0)) { /* pairs: gate/up formats */     \
            return rows_256<Q, R>(isa, mats, row_bytes, slices, row_begin, row_end, columns, ncols, out);              \
        } else {                                                                                                       \
            return false;                                                                                              \
        }
        INFERNIX_GGML_TILED_FORMAT(QType::GGML_IQ2_S)
        INFERNIX_GGML_TILED_FORMAT(QType::GGML_IQ3_XXS)
        INFERNIX_GGML_TILED_FORMAT(QType::GGML_IQ3_S)
        INFERNIX_GGML_TILED_FORMAT(QType::GGML_IQ4_XS)
        INFERNIX_GGML_TILED_FORMAT(QType::GGML_IQ4_NL)
        INFERNIX_GGML_TILED_FORMAT(QType::GGML_Q2_0)
#    undef INFERNIX_GGML_TILED_FORMAT
    default: return false;
    }
#else
    (void)isa, (void)format, (void)mats, (void)row_bytes, (void)slices, (void)row_begin, (void)row_end, (void)columns,
        (void)ncols, (void)out;
    return false;
#endif
}

// ======================================================================= the 8-bit activation route
//
// ggml_expert.h's a8 functions (design §19.3.31; ops/ggml/ggml_a8.h; the GPU's cuda/ggml_expert_a8.cuh). The
// weight decode is the exact route's (decode<F>; ggml_exact.h's chunks on the scalar route), so the codes,
// sub-scales and scale words are the GPU's (ggml_dp4a.cuh's fast::slice). A slice's isum is an exact int32 in
// any order and any ISA: 32 terms of |code * sub| <= 4064 times |q| <= 127, |isum| < 2^24, so fl32(isum) is
// exact too. The binary32 part is the canonical sequence per slice, m = fl32(isum * d_x) then
// fma(m, d_w, partial), slice s into partial s % P in ascending s from +0, then the butterfly: the vector
// routes keep partial p in lane p, so each lane runs exactly that sequence (IEEE mul and fma, never
// reassociated or contracted further).

namespace a8 = ggml::a8;

static_assert(a8::kGateUpPartials == 16 && kGgmlXSlices == 5 * a8::kGateUpPartials,
              "a gate/up row is five groups of 16 slices, slice j of a group in partial j");
static_assert(a8::kDownPartials == 4 && kGgmlHSlices == 5 * a8::kDownPartials,
              "a down row is five groups of 4 slices, slice j of a group in partial j");
constexpr int kHGroups = kGgmlHSlices / a8::kDownPartials; // 5

// The formats whose sub-scales are all 1: their weights are their codes.
template <QType F>
inline constexpr bool kUnitSub = F == QType::GGML_IQ4_NL || F == QType::GGML_Q2_0;

// A row's BF16 output from its sum p_0 (the kernels' f32_to_bf16_rn(output_f32(p_0, kappa))).
template <QType F>
inline std::uint16_t a8_output(float p0) {
    return canon::f32_to_bf16_rn(a8::output_f32(p0, ggml::kappa_log2(F)));
}

// ggml_a8.h's butterfly over P partials: p_i = p_i + p_(i + o), o = P / 2 .. 1 (i < o); the sum in p_0.
template <int P>
inline float a8_butterfly(float (&p)[P]) {
    for (int o = P / 2; o >= 1; o /= 2) {
        for (int i = 0; i < o; ++i) { p[i] = canon::add_rn(p[i], p[i + o]); }
    }
    return p[0];
}

// The 8-bit route's pairs: the exact route's GGUF pairs (block-FP8 banks keep their own route).
bool a8_supported(const GgmlExpertFormat& format) {
    return format.gate_up != QType::FP8_E4M3FN_B128 && ggml_cpu_supported(format);
}

// ---------------------------------------------------------------------------------- a8, scalar

// Slice s of a row: its 32 weights code * sub (ggml_exact.h's chunks) and its block's scale word.
template <QType F>
std::uint16_t a8_slice_weights(const std::uint8_t* row, int s, std::int32_t (&w)[32]) {
    constexpr int kBlockSlices = block_slices(F);
    constexpr int kBlockBytes  = static_cast<int>(ggml_block_bytes(F));
    const std::uint8_t* block  = row + static_cast<std::size_t>(s / kBlockSlices) * kBlockBytes;
    std::uint16_t d            = 0;
    for (int p = 0; p < 4; ++p) {
        const ggml::Chunk c = ggml::chunk<F>(block, 4 * (s % kBlockSlices) + p);
        for (int j = 0; j < 8; ++j) { w[8 * p + j] = c.code[j] * c.sub; }
        d = c.d;
    }
    return d;
}

// The reference form: rows [row_begin, row_end) of a matrix of F rows (`slices` slices each, P partials)
// against ncols columns, every slice decoded once; out[(row - row_begin) * ncols + c] is the row's p_0.
template <QType F, int P>
void a8_rows_scalar(const std::uint8_t* matrix, int row_bytes, int slices, int row_begin, int row_end,
                    const CpuA8Slice* const* x, int ncols, float* out) {
    for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t* w = matrix + static_cast<std::size_t>(row) * row_bytes;
        float part[kMaxColumns][P] = {};
        for (int s = 0; s < slices; ++s) {
            std::int32_t ws[32];
            const float wd = a8::f16_to_f32(a8_slice_weights<F>(w, s, ws));
            for (int c = 0; c < ncols; ++c) {
                const CpuA8Slice& xs = x[c][s];
                std::int32_t isum    = 0;
                for (int e = 0; e < 32; ++e) { isum += ws[e] * xs.q[e]; }
                part[c][s % P] = a8::add_term(part[c][s % P], isum, xs.d, wd);
            }
        }
        for (int c = 0; c < ncols; ++c) { out[static_cast<std::size_t>(row - row_begin) * ncols + c] = a8_butterfly(part[c]); }
    }
}

template <QType F>
void a8_gate_up_scalar(const std::uint8_t* record, int row_bytes, const CpuA8Slice* const* x, int ncols, int i_begin,
                       int i_end, std::uint16_t* const* h) {
    constexpr int kRows = 16;
    float g[kRows * kMaxColumns], u[kRows * kMaxColumns];
    const std::uint8_t* up = record + static_cast<std::size_t>(kIntermediate) * row_bytes;
    for (int i0 = i_begin; i0 < i_end; i0 += kRows) {
        const int i1 = std::min(i_end, i0 + kRows);
        a8_rows_scalar<F, a8::kGateUpPartials>(record, row_bytes, kGgmlXSlices, i0, i1, x, ncols, g);
        a8_rows_scalar<F, a8::kGateUpPartials>(up, row_bytes, kGgmlXSlices, i0, i1, x, ncols, u);
        for (int i = i0; i < i1; ++i) {
            for (int c = 0; c < ncols; ++c) {
                const std::size_t at = static_cast<std::size_t>(i - i0) * ncols + c;
                h[c][i]              = canon::swiglu_bf16(a8_output<F>(g[at]), a8_output<F>(u[at]));
            }
        }
    }
}

template <QType F>
void a8_down_scalar(const std::uint8_t* down, int row_bytes, const CpuA8Slice* const* h, int ncols, int row_begin,
                    int row_end, std::uint16_t* const* y) {
    constexpr int kRows = 32;
    float r[kRows * kMaxColumns];
    for (int r0 = row_begin; r0 < row_end; r0 += kRows) {
        const int r1 = std::min(row_end, r0 + kRows);
        a8_rows_scalar<F, a8::kDownPartials>(down, row_bytes, kGgmlHSlices, r0, r1, h, ncols, r);
        for (int row = r0; row < r1; ++row) {
            for (int c = 0; c < ncols; ++c) { y[c][row] = a8_output<F>(r[static_cast<std::size_t>(row - r0) * ncols + c]); }
        }
    }
}

#if defined(INFERNIX_GGML_X86)

// The vector routes' targets: FMA for the terms, F16C for the weight scales (both on every AVX2 CPU).
#    define INFERNIX_A8_AVX2 "avx2,fma,f16c"
#    define INFERNIX_A8_AVXVNNI "avx2,fma,f16c,avxvnni"
#    define INFERNIX_A8_AVX512 "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni"

// --------------------------------------------------------------------------- a8, shared x86 pieces

// The slice's weights code * sub as int16 (scaled(); the codes alone where every sub-scale is 1).
template <QType F>
INFERNIX_TARGET("avx2")
inline Weights a8_weights(const Decoded& d) {
    if constexpr (kUnitSub<F>) {
        return {_mm256_cvtepi8_epi16(_mm256_castsi256_si128(d.q)), _mm256_cvtepi8_epi16(_mm256_extracti128_si256(d.q, 1))};
    } else {
        return scaled(d);
    }
}

// Binary16 weight scales in binary32 by F16C: exact for every finite or infinite word, so a8::f16_to_f32's
// values (a NaN word gives a NaN either way, and every NaN output is the canonical one). The two blocks of a
// gate/up group: lane 0 the scale word d0, lane 1 d1.
INFERNIX_TARGET(INFERNIX_A8_AVX2)
inline __m128 a8_pair_scales(std::uint16_t d0, std::uint16_t d1) {
    return _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<int>(d0 | (static_cast<unsigned>(d1) << 16))));
}

// A down row's 20 slice scales, four per group (wd[t]: slices 4t..4t+3).
INFERNIX_TARGET(INFERNIX_A8_AVX2)
inline void a8_down_scales(const std::uint16_t (&d)[kGgmlHSlices], __m128 (&wd)[kHGroups]) {
    const __m256 a = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(d)));
    const __m256 b = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(d + 8)));
    wd[0]          = _mm256_castps256_ps128(a);
    wd[1]          = _mm256_extractf128_ps(a, 1);
    wd[2]          = _mm256_castps256_ps128(b);
    wd[3]          = _mm256_extractf128_ps(b, 1);
    wd[4]          = _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(d + 16)));
}

// Eight slices' terms, lane j: acc_j = fma(fl32(isum_j * xd_j), wd_j, acc_j) (a8::add_term).
INFERNIX_TARGET(INFERNIX_A8_AVX2)
inline __m256 a8_terms(__m256 acc, __m256i isum, __m256 xd, __m256 wd) {
    return _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(isum), xd), wd, acc);
}

// The butterfly's p_0 over four partials (lane i: p_i): o = 2 (lanes 0, 1: p_i + p_(i + 2)), then o = 1.
INFERNIX_TARGET("avx2")
inline float a8_butterfly4(__m128 p) {
    const __m128 q = _mm_add_ps(p, _mm_movehl_ps(p, p));
    return _mm_cvtss_f32(_mm_add_ss(q, _mm_shuffle_ps(q, q, 1)));
}

// ... over 16 partials, lo holding p_0..p_7 and hi p_8..p_15: o = 8, o = 4, then a8_butterfly4.
INFERNIX_TARGET("avx2")
inline float a8_butterfly16(__m256 lo, __m256 hi) {
    const __m256 p = _mm256_add_ps(lo, hi);
    return a8_butterfly4(_mm_add_ps(_mm256_castps256_ps128(p), _mm256_extractf128_ps(p, 1)));
}

// A down row's p_0 from its five groups' products m = fl32(isum * xd) (m01: groups 0 and 1, m23: groups 2
// and 3, m4: group 4) and scales wd[t]: lane p is partial p, the groups added in order from +0.
INFERNIX_TARGET(INFERNIX_A8_AVX2)
inline float a8_down_sum(__m256 m01, __m256 m23, __m128 m4, const __m128 (&wd)[kHGroups]) {
    __m128 p = _mm_fmadd_ps(_mm256_castps256_ps128(m01), wd[0], _mm_setzero_ps());
    p        = _mm_fmadd_ps(_mm256_extractf128_ps(m01, 1), wd[1], p);
    p        = _mm_fmadd_ps(_mm256_castps256_ps128(m23), wd[2], p);
    p        = _mm_fmadd_ps(_mm256_extractf128_ps(m23, 1), wd[3], p);
    p        = _mm_fmadd_ps(m4, wd[4], p);
    return a8_butterfly4(p);
}

// hsum8's last steps from its first horizontal adds t[k] = hadd(v[2k], v[2k + 1]): lane j is v[j]'s sum.
INFERNIX_TARGET("avx2")
inline __m256i a8_hsum_tail(const __m256i (&t)[4]) {
    const __m256i u0 = _mm256_hadd_epi32(t[0], t[1]), u1 = _mm256_hadd_epi32(t[2], t[3]);
    return _mm256_add_epi32(_mm256_permute2x128_si256(u0, u1, 0x20), _mm256_permute2x128_si256(u0, u1, 0x31));
}

// Slices s0..s0+15 of a gate/up row (s0 a multiple of 16; the gate/up formats' blocks are 8 slices): their
// weights, and their two blocks' scales broadcast over partials 0-7 (lo) and 8-15 (hi).
template <QType F>
INFERNIX_TARGET(INFERNIX_A8_AVX2)
inline void a8_decode16(const std::uint8_t* row, int s0, Weights (&w)[16], __m256& lo, __m256& hi) {
    static_assert(block_slices(F) == 8, "a group of 16 slices is two whole blocks");
    std::uint16_t d[2] = {};
    for (int j = 0; j < 16; ++j) {
        const Decoded dw = decode<F>(row, s0 + j);
        w[j]             = a8_weights<F>(dw);
        d[j / 8]         = dw.d;
    }
    const __m128 f = a8_pair_scales(d[0], d[1]);
    lo             = _mm256_broadcastss_ps(f);
    hi             = _mm256_broadcastss_ps(_mm_movehdup_ps(f));
}

// A down row's 20 slices: their weights and scales.
template <QType F>
INFERNIX_TARGET(INFERNIX_A8_AVX2)
inline void a8_decode20(const std::uint8_t* row, Weights (&w)[kGgmlHSlices], __m128 (&wd)[kHGroups]) {
    std::uint16_t d[kGgmlHSlices];
    for (int s = 0; s < kGgmlHSlices; ++s) {
        const Decoded dw = decode<F>(row, s);
        w[s]             = a8_weights<F>(dw);
        d[s]             = dw.d;
    }
    a8_down_scales(d, wd);
}

// -------------------------------------------------------- a8, 256-bit routes (AVX2 and AVX-VNNI)
//
// Mad: x86::Avx2Mad or x86::AvxVnniMad. A slice's dot with one column is two 16-bit multiply-adds into
// eight int32 lanes (each at most 4 * 4064 * 127 < 2^21); hsum8's horizontal adds turn eight slices' lanes
// into their isums in slice order, so a lane of the isums is a partial (gate/up: two vectors per group of 16
// slices; down: one vector holds two groups of 4). A row's group of slices is decoded once for all of a
// call's columns (each column's work then reads the decoded weights from L1), and a gate row and its up row
// share each column's code loads.
#    define INFERNIX_GGML_A8_ROUTES(Name, Target, Mad)                                                                  \
        INFERNIX_TARGET(Target)                                                                                         \
        inline __m256i Name##_dot(const Weights& w, __m256i x0, __m256i x1) {                                           \
            return Mad::mad(_mm256_madd_epi16(w.lo, x0), w.hi, x1);                                                     \
        }                                                                                                               \
        INFERNIX_TARGET(Target)                                                                                         \
        inline __m256i Name##_slice(const Weights& w, const CpuA8Slice& x) {                                            \
            const auto* q = reinterpret_cast<const __m256i*>(x.q);                                                      \
            return Name##_dot(w, _mm256_loadu_si256(q), _mm256_loadu_si256(q + 1));                                     \
        }                                                                                                               \
        /* Eight slices of one column against a gate row's and an up row's weights: their isums in slice order. */     \
        INFERNIX_TARGET(Target)                                                                                         \
        inline void Name##_isums8x2(const Weights* wg, const Weights* wu, const CpuA8Slice* xs, __m256i& ig,           \
                                    __m256i& iu) {                                                                      \
            __m256i tg[4], tu[4];                                                                                       \
            for (int k = 0; k < 4; ++k) {                                                                               \
                const auto* a    = reinterpret_cast<const __m256i*>(xs[2 * k].q);                                       \
                const auto* b    = reinterpret_cast<const __m256i*>(xs[2 * k + 1].q);                                   \
                const __m256i a0 = _mm256_loadu_si256(a), a1 = _mm256_loadu_si256(a + 1);                               \
                const __m256i b0 = _mm256_loadu_si256(b), b1 = _mm256_loadu_si256(b + 1);                               \
                tg[k] = _mm256_hadd_epi32(Name##_dot(wg[2 * k], a0, a1), Name##_dot(wg[2 * k + 1], b0, b1));            \
                tu[k] = _mm256_hadd_epi32(Name##_dot(wu[2 * k], a0, a1), Name##_dot(wu[2 * k + 1], b0, b1));            \
            }                                                                                                           \
            ig = a8_hsum_tail(tg);                                                                                      \
            iu = a8_hsum_tail(tu);                                                                                      \
        }                                                                                                               \
        /* Eight (four) slices of one column: their isums in slice order. */                                          \
        INFERNIX_TARGET(Target)                                                                                         \
        inline __m256i Name##_isums8(const Weights* w, const CpuA8Slice* xs) {                                          \
            __m256i t[4];                                                                                               \
            for (int k = 0; k < 4; ++k) {                                                                               \
                t[k] = _mm256_hadd_epi32(Name##_slice(w[2 * k], xs[2 * k]), Name##_slice(w[2 * k + 1], xs[2 * k + 1])); \
            }                                                                                                           \
            return a8_hsum_tail(t);                                                                                     \
        }                                                                                                               \
        INFERNIX_TARGET(Target)                                                                                         \
        inline __m128i Name##_isums4(const Weights* w, const CpuA8Slice* xs) {                                          \
            const __m256i t = _mm256_hadd_epi32(_mm256_hadd_epi32(Name##_slice(w[0], xs[0]), Name##_slice(w[1], xs[1])), \
                                                _mm256_hadd_epi32(Name##_slice(w[2], xs[2]), Name##_slice(w[3], xs[3]))); \
            return _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));                            \
        }                                                                                                               \
        template <QType F>                                                                                              \
        INFERNIX_TARGET(Target)                                                                                         \
        void Name##_gate_up(const std::uint8_t* record, int row_bytes, const CpuA8Slice* const* x,                      \
                            const float (*xd)[kGgmlXSlices], int ncols, int i_begin, int i_end,                         \
                            std::uint16_t* const* h) {                                                                  \
            constexpr int P        = a8::kGateUpPartials;                                                               \
            const std::uint8_t* up = record + static_cast<std::size_t>(kIntermediate) * row_bytes;                      \
            for (int i = i_begin; i < i_end; ++i) {                                                                     \
                const std::uint8_t* g = record + static_cast<std::size_t>(i) * row_bytes;                               \
                const std::uint8_t* u = up + static_cast<std::size_t>(i) * row_bytes;                                   \
                __m256 acc[kMaxColumns][4]; /* gate p0-7, gate p8-15, up p0-7, up p8-15 */                              \
                for (int c = 0; c < ncols; ++c) {                                                                       \
                    for (int k = 0; k < 4; ++k) { acc[c][k] = _mm256_setzero_ps(); }                                    \
                }                                                                                                       \
                for (int s0 = 0; s0 < kGgmlXSlices; s0 += P) {                                                          \
                    Weights wg[P], wu[P];                                                                               \
                    __m256 dg[2], du[2];                                                                                \
                    a8_decode16<F>(g, s0, wg, dg[0], dg[1]);                                                            \
                    a8_decode16<F>(u, s0, wu, du[0], du[1]);                                                            \
                    for (int c = 0; c < ncols; ++c) {                                                                   \
                        const CpuA8Slice* xs = x[c] + s0;                                                               \
                        __m256i ig0, iu0, ig1, iu1;                                                                     \
                        Name##_isums8x2(wg, wu, xs, ig0, iu0);                                                          \
                        Name##_isums8x2(wg + 8, wu + 8, xs + 8, ig1, iu1);                                              \
                        const __m256 x0 = _mm256_loadu_ps(xd[c] + s0), x1 = _mm256_loadu_ps(xd[c] + s0 + 8);            \
                        acc[c][0]       = a8_terms(acc[c][0], ig0, x0, dg[0]);                                          \
                        acc[c][1]       = a8_terms(acc[c][1], ig1, x1, dg[1]);                                          \
                        acc[c][2]       = a8_terms(acc[c][2], iu0, x0, du[0]);                                          \
                        acc[c][3]       = a8_terms(acc[c][3], iu1, x1, du[1]);                                          \
                    }                                                                                                   \
                }                                                                                                       \
                for (int c = 0; c < ncols; ++c) {                                                                       \
                    h[c][i] = canon::swiglu_bf16(a8_output<F>(a8_butterfly16(acc[c][0], acc[c][1])),                    \
                                                 a8_output<F>(a8_butterfly16(acc[c][2], acc[c][3])));                   \
                }                                                                                                       \
            }                                                                                                           \
        }                                                                                                               \
        template <QType F>                                                                                              \
        INFERNIX_TARGET(Target)                                                                                         \
        void Name##_down(const std::uint8_t* down, int row_bytes, const CpuA8Slice* const* h,                           \
                         const float (*hd)[kGgmlHSlices], int ncols, int row_begin, int row_end,                        \
                         std::uint16_t* const* y) {                                                                     \
            for (int row = row_begin; row < row_end; ++row) {                                                           \
                Weights w[kGgmlHSlices];                                                                                \
                __m128 wd[kHGroups];                                                                                    \
                a8_decode20<F>(down + static_cast<std::size_t>(row) * row_bytes, w, wd);                                \
                for (int c = 0; c < ncols; ++c) {                                                                       \
                    const CpuA8Slice* hs = h[c];                                                                        \
                    const __m256 m01 =                                                                                  \
                        _mm256_mul_ps(_mm256_cvtepi32_ps(Name##_isums8(w, hs)), _mm256_loadu_ps(hd[c]));                \
                    const __m256 m23 =                                                                                  \
                        _mm256_mul_ps(_mm256_cvtepi32_ps(Name##_isums8(w + 8, hs + 8)), _mm256_loadu_ps(hd[c] + 8));    \
                    const __m128 m4 =                                                                                   \
                        _mm_mul_ps(_mm_cvtepi32_ps(Name##_isums4(w + 16, hs + 16)), _mm_loadu_ps(hd[c] + 16));          \
                    y[c][row] = a8_output<F>(a8_down_sum(m01, m23, m4, wd));                                            \
                }                                                                                                       \
            }                                                                                                           \
        }

INFERNIX_GGML_A8_ROUTES(a8_avx2, INFERNIX_A8_AVX2, Avx2Mad)
INFERNIX_GGML_A8_ROUTES(a8_avxvnni, INFERNIX_A8_AVXVNNI, AvxVnniMad)
#    undef INFERNIX_GGML_A8_ROUTES

// ---------------------------------------------------------------- a8, AVX-512 (512-bit vectors)
//
// A slice's 32 weights are one zmm and a column's slice codes another, so a slice's dot is one vpmaddwd
// (16 int32 lanes, each at most 2 * 4064 * 127 < 2^20). Sixteen slices' lanes become their 16 isums in slice
// order by a transpose-add (a8_quad per four slices, then a8_combine16: 2.8 single-µop instructions per slice,
// where fold16 and hsum8 take about 4.6 µops), and the 16 partials of a gate/up row are one zmm: one mul and
// one fma per group and column. As on the 256-bit routes, a group is decoded once for all columns and gate and
// up share loads.

template <QType F>
INFERNIX_TARGET(INFERNIX_A8_AVX512)
inline __m512i a8_weights512(const Decoded& d) {
    const __m512i codes = _mm512_cvtepi8_epi16(d.q);
    if constexpr (kUnitSub<F>) {
        return codes;
    } else if constexpr (F == QType::GGML_IQ2_S) { // elements 0-15 and 16-31 have their own sub-scales
        return _mm512_mullo_epi16(codes, _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16(static_cast<short>(d.sub0))),
                                                            _mm256_set1_epi16(static_cast<short>(d.sub1)), 1));
    } else {
        return _mm512_mullo_epi16(codes, _mm512_set1_epi16(static_cast<short>(d.sub0)));
    }
}

INFERNIX_TARGET(INFERNIX_A8_AVX512)
inline __m512i a8_codes512(const CpuA8Slice& x) { return _mm512_loadu_si512(x.q); }

// Four slices' lanes a, b, c, d: 128-bit lane k of the result is [a, b, c, d], each summed over its own
// four elements of lane k.
INFERNIX_TARGET(INFERNIX_A8_AVX512)
inline __m512i a8_quad(__m512i a, __m512i b, __m512i c, __m512i d) {
    const __m512i ab = _mm512_add_epi32(_mm512_unpacklo_epi32(a, b), _mm512_unpackhi_epi32(a, b));
    const __m512i cd = _mm512_add_epi32(_mm512_unpacklo_epi32(c, d), _mm512_unpackhi_epi32(c, d));
    return _mm512_add_epi32(_mm512_unpacklo_epi64(ab, cd), _mm512_unpackhi_epi64(ab, cd));
}

// Sixteen slices' sums in slice order from the a8_quad of slices 0-3, 4-7, 8-11 and 12-15: each step adds
// 128-bit lanes in pairs ([q.l0 + q.l1, q.l2 + q.l3] of two quads), twice.
INFERNIX_TARGET(INFERNIX_A8_AVX512)
inline __m512i a8_combine16(__m512i q0, __m512i q1, __m512i q2, __m512i q3) {
    const __m512i p = _mm512_add_epi32(_mm512_shuffle_i32x4(q0, q1, 0x88), _mm512_shuffle_i32x4(q0, q1, 0xDD));
    const __m512i r = _mm512_add_epi32(_mm512_shuffle_i32x4(q2, q3, 0x88), _mm512_shuffle_i32x4(q2, q3, 0xDD));
    return _mm512_add_epi32(_mm512_shuffle_i32x4(p, r, 0x88), _mm512_shuffle_i32x4(p, r, 0xDD));
}

// Four slices' sums in slice order from their a8_quad.
INFERNIX_TARGET(INFERNIX_A8_AVX512)
inline __m128i a8_fold_quad(__m512i q) {
    const __m256i h = _mm256_add_epi32(_mm512_castsi512_si256(q), _mm512_extracti64x4_epi64(q, 1));
    return _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
}

INFERNIX_TARGET(INFERNIX_A8_AVX512)
inline __m256 a8_high256(__m512 v) { return _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1)); }

// Slices s0..s0+15 of a gate/up row: their weights, and their two blocks' scales over partials 0-7 and 8-15.
template <QType F>
INFERNIX_TARGET(INFERNIX_A8_AVX512)
inline void a8_decode16_512(const std::uint8_t* row, int s0, __m512i (&w)[16], __m512& wd) {
    static_assert(block_slices(F) == 8, "a group of 16 slices is two whole blocks");
    std::uint16_t d[2] = {};
    for (int j = 0; j < 16; ++j) {
        const Decoded dw = decode<F>(row, s0 + j);
        w[j]             = a8_weights512<F>(dw);
        d[j / 8]         = dw.d;
    }
    const __m128 f = a8_pair_scales(d[0], d[1]);
    wd = _mm512_mask_blend_ps(0xFF00, _mm512_broadcastss_ps(f), _mm512_broadcastss_ps(_mm_movehdup_ps(f)));
}

// Sixteen slices of one column against a gate row's and an up row's weights: their isums in slice order.
INFERNIX_TARGET(INFERNIX_A8_AVX512)
inline void a8_isums16x2(const __m512i (&wg)[16], const __m512i (&wu)[16], const CpuA8Slice* xs, __m512i& ig,
                         __m512i& iu) {
    __m512i qg[4], qu[4];
    for (int k = 0; k < 4; ++k) {
        const __m512i x0 = a8_codes512(xs[4 * k]), x1 = a8_codes512(xs[4 * k + 1]);
        const __m512i x2 = a8_codes512(xs[4 * k + 2]), x3 = a8_codes512(xs[4 * k + 3]);
        qg[k] = a8_quad(_mm512_madd_epi16(wg[4 * k], x0), _mm512_madd_epi16(wg[4 * k + 1], x1),
                        _mm512_madd_epi16(wg[4 * k + 2], x2), _mm512_madd_epi16(wg[4 * k + 3], x3));
        qu[k] = a8_quad(_mm512_madd_epi16(wu[4 * k], x0), _mm512_madd_epi16(wu[4 * k + 1], x1),
                        _mm512_madd_epi16(wu[4 * k + 2], x2), _mm512_madd_epi16(wu[4 * k + 3], x3));
    }
    ig = a8_combine16(qg[0], qg[1], qg[2], qg[3]);
    iu = a8_combine16(qu[0], qu[1], qu[2], qu[3]);
}

template <QType F>
INFERNIX_TARGET(INFERNIX_A8_AVX512)
void a8_avx512_gate_up(const std::uint8_t* record, int row_bytes, const CpuA8Slice* const* x,
                       const float (*xd)[kGgmlXSlices], int ncols, int i_begin, int i_end, std::uint16_t* const* h) {
    constexpr int P        = a8::kGateUpPartials;
    const std::uint8_t* up = record + static_cast<std::size_t>(kIntermediate) * row_bytes;
    for (int i = i_begin; i < i_end; ++i) {
        const std::uint8_t* g = record + static_cast<std::size_t>(i) * row_bytes;
        const std::uint8_t* u = up + static_cast<std::size_t>(i) * row_bytes;
        __m512 accg[kMaxColumns], accu[kMaxColumns]; // lane p: partial p
        for (int c = 0; c < ncols; ++c) { accg[c] = accu[c] = _mm512_setzero_ps(); }
        for (int s0 = 0; s0 < kGgmlXSlices; s0 += P) {
            __m512i wg[P], wu[P];
            __m512 dg, du;
            a8_decode16_512<F>(g, s0, wg, dg);
            a8_decode16_512<F>(u, s0, wu, du);
            for (int c = 0; c < ncols; ++c) {
                __m512i ig, iu;
                a8_isums16x2(wg, wu, x[c] + s0, ig, iu);
                const __m512 xv = _mm512_loadu_ps(xd[c] + s0);
                accg[c]         = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(ig), xv), dg, accg[c]);
                accu[c]         = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(iu), xv), du, accu[c]);
            }
        }
        for (int c = 0; c < ncols; ++c) {
            h[c][i] = canon::swiglu_bf16(a8_output<F>(a8_butterfly16(_mm512_castps512_ps256(accg[c]), a8_high256(accg[c]))),
                                         a8_output<F>(a8_butterfly16(_mm512_castps512_ps256(accu[c]), a8_high256(accu[c]))));
        }
    }
}

template <QType F>
INFERNIX_TARGET(INFERNIX_A8_AVX512)
void a8_avx512_down(const std::uint8_t* down, int row_bytes, const CpuA8Slice* const* h,
                    const float (*hd)[kGgmlHSlices], int ncols, int row_begin, int row_end, std::uint16_t* const* y) {
    for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t* w = down + static_cast<std::size_t>(row) * row_bytes;
        __m512i ws[kGgmlHSlices];
        std::uint16_t d[kGgmlHSlices];
        for (int s = 0; s < kGgmlHSlices; ++s) {
            const Decoded dw = decode<F>(w, s);
            ws[s]            = a8_weights512<F>(dw);
            d[s]             = dw.d;
        }
        __m128 wd[kHGroups];
        a8_down_scales(d, wd);
        for (int c = 0; c < ncols; ++c) {
            const CpuA8Slice* hs = h[c];
            __m512i q[kHGroups];
            for (int k = 0; k < kHGroups; ++k) {
                q[k] = a8_quad(_mm512_madd_epi16(ws[4 * k], a8_codes512(hs[4 * k])),
                               _mm512_madd_epi16(ws[4 * k + 1], a8_codes512(hs[4 * k + 1])),
                               _mm512_madd_epi16(ws[4 * k + 2], a8_codes512(hs[4 * k + 2])),
                               _mm512_madd_epi16(ws[4 * k + 3], a8_codes512(hs[4 * k + 3])));
            }
            // Groups 0-3 (16 lanes) and group 4: m = fl32(isum * xd), lane-wise.
            const __m512 m16 = _mm512_mul_ps(_mm512_cvtepi32_ps(a8_combine16(q[0], q[1], q[2], q[3])), _mm512_loadu_ps(hd[c]));
            const __m128 m4  = _mm_mul_ps(_mm_cvtepi32_ps(a8_fold_quad(q[4])), _mm_loadu_ps(hd[c] + 16));
            y[c][row]        = a8_output<F>(a8_down_sum(_mm512_castps512_ps256(m16), a8_high256(m16), m4, wd));
        }
    }
}

#    undef INFERNIX_A8_AVX2
#    undef INFERNIX_A8_AVXVNNI
#    undef INFERNIX_A8_AVX512

#endif // INFERNIX_GGML_X86

// ------------------------------------------------------------------------------- a8, dispatch

template <QType F>
void a8_gate_up_format(CpuIsa isa, const std::uint8_t* record, int row_bytes, const CpuA8Slice* const* x, int ncols,
                       int i_begin, int i_end, std::uint16_t* const* h) {
#if defined(INFERNIX_GGML_X86)
    if (isa != CpuIsa::kScalar) {
        // The columns' slice scales, contiguous: a group's 16 load as partials' lanes.
        alignas(64) float xd[kMaxColumns][kGgmlXSlices];
        for (int c = 0; c < ncols; ++c) {
            for (int s = 0; s < kGgmlXSlices; ++s) { xd[c][s] = x[c][s].d; }
        }
        switch (isa) {
        case CpuIsa::kAvx2: a8_avx2_gate_up<F>(record, row_bytes, x, xd, ncols, i_begin, i_end, h); return;
        case CpuIsa::kAvxVnni: a8_avxvnni_gate_up<F>(record, row_bytes, x, xd, ncols, i_begin, i_end, h); return;
        case CpuIsa::kAvx512Vnni: a8_avx512_gate_up<F>(record, row_bytes, x, xd, ncols, i_begin, i_end, h); return;
        default: break;
        }
    }
#endif
    a8_gate_up_scalar<F>(record, row_bytes, x, ncols, i_begin, i_end, h);
}

template <QType F>
void a8_down_format(CpuIsa isa, const std::uint8_t* down, int row_bytes, const CpuA8Slice* const* h, int ncols,
                    int row_begin, int row_end, std::uint16_t* const* y) {
#if defined(INFERNIX_GGML_X86)
    if (isa != CpuIsa::kScalar) {
        alignas(64) float hd[kMaxColumns][kGgmlHSlices];
        for (int c = 0; c < ncols; ++c) {
            for (int s = 0; s < kGgmlHSlices; ++s) { hd[c][s] = h[c][s].d; }
        }
        switch (isa) {
        case CpuIsa::kAvx2: a8_avx2_down<F>(down, row_bytes, h, hd, ncols, row_begin, row_end, y); return;
        case CpuIsa::kAvxVnni: a8_avxvnni_down<F>(down, row_bytes, h, hd, ncols, row_begin, row_end, y); return;
        case CpuIsa::kAvx512Vnni: a8_avx512_down<F>(down, row_bytes, h, hd, ncols, row_begin, row_end, y); return;
        default: break;
        }
    }
#endif
    a8_down_scalar<F>(down, row_bytes, h, ncols, row_begin, row_end, y);
}

} // namespace

bool ggml_cpu_supported(const GgmlExpertFormat& format) noexcept {
    if (format.gate_up == QType::FP8_E4M3FN_B128) { // fp8_expert.cpp
        return format.down == QType::FP8_E4M3FN_B128 && format.gate_row_bytes == kHidden &&
               format.down_row_bytes == kIntermediate;
    }
    const bool gate_up = format.gate_up == QType::GGML_IQ2_S || format.gate_up == QType::GGML_IQ3_XXS ||
                         format.gate_up == QType::GGML_IQ3_S || format.gate_up == QType::GGML_IQ4_XS;
    const bool down    = format.down == QType::GGML_IQ4_NL || format.down == QType::GGML_Q2_0;
    return gate_up && down &&
           format.gate_row_bytes == kHidden / ggml_block_elems(format.gate_up) * ggml_block_bytes(format.gate_up) &&
           format.down_row_bytes == kIntermediate / ggml_block_elems(format.down) * ggml_block_bytes(format.down);
}

int ggml_encode_column(const std::uint16_t* v, int n, CpuLimbSlice* out) {
    require(n % 32 == 0, "a column is whole 32-element slices");
#if defined(INFERNIX_GGML_X86)
    static const bool avx2 = cpu_isa_supported(CpuIsa::kAvx2);
    if (avx2) { return encode_column_avx2(v, n, out); }
#endif
    const int emax = canon::a16_column_exponent(v, n);
    for (int j = 0; j < n; ++j) {
        const std::int32_t x = emax == canon::kA16NonFinite ? 0 : canon::a16_value(v[j], emax);
        CpuLimbSlice& l      = out[j / 32];
        l.lo[j % 32] = static_cast<std::int16_t>(x & 2047);
        l.hi[j % 32] = static_cast<std::int16_t>(x >> 11);
    }
    return emax;
}

void ggml_row_sums(CpuIsa isa, QType format, const std::uint8_t* matrix, int row_bytes, int slices, int row_begin,
                   int row_end, const CpuLimbSlice* const* columns, int ncols, ggml::I128* r) {
    require(cpu_isa_supported(isa), "unsupported CPU ISA");
    require(ncols >= 1 && ncols <= kMaxColumns, "ncols must be in [1, 8]");
#define INFERNIX_GGML_FORMAT(Q)                                                                                        \
    case Q:                                                                                                            \
        require(slices % (block_slices(Q) > 4 ? 4 : block_slices(Q)) == 0, "K is not whole units");                  \
        {                                                                                                              \
            const std::uint8_t* const mats[1] = {matrix};                                                              \
            ggml::I128* const outs[1]         = {r};                                                                   \
            if (ggml_rows<1>(isa, Q, mats, row_bytes, slices, row_begin, row_end, columns, ncols, outs)) { return; }   \
        }                                                                                                              \
        row_sums_format<Q>(isa, matrix, row_bytes, slices, row_begin, row_end, columns, ncols, r);                     \
        return;
    switch (format) {
        INFERNIX_GGML_FORMAT(QType::GGML_IQ2_S)
        INFERNIX_GGML_FORMAT(QType::GGML_IQ3_XXS)
        INFERNIX_GGML_FORMAT(QType::GGML_IQ3_S)
        INFERNIX_GGML_FORMAT(QType::GGML_IQ4_XS)
        INFERNIX_GGML_FORMAT(QType::GGML_IQ4_NL)
        INFERNIX_GGML_FORMAT(QType::GGML_Q2_0)
    default:
        throw std::invalid_argument("offloaded_moe ggml: not an expert format of the CPU route");
    }
#undef INFERNIX_GGML_FORMAT
}

void ggml_gate_up(CpuIsa isa, const std::uint8_t* record, const GgmlExpertFormat& format,
                  const CpuLimbSlice* const* x, const int* x_emax, int ncols, int i_begin, int i_end,
                  std::uint16_t* const* h) {
    if (format.gate_up == QType::FP8_E4M3FN_B128) {
        fp8_gate_up(isa, record, x, x_emax, ncols, i_begin, i_end, h);
        return;
    }
    constexpr int kUnit = 16;
    ggml::I128 g[kUnit * kMaxColumns], u[kUnit * kMaxColumns];
    const int row_bytes = static_cast<int>(format.gate_row_bytes);
    const std::uint8_t* up = record + static_cast<std::size_t>(kIntermediate) * row_bytes;
    for (int i0 = i_begin; i0 < i_end; i0 += kUnit) {
        const int i1 = std::min(i_end, i0 + kUnit);
        // A gate row and its up row in one pass on the 256-bit ISAs.
        const std::uint8_t* const mats[2] = {record, up};
        ggml::I128* const outs[2]         = {g, u};
        if (ncols < 1 || ncols > kMaxColumns || !cpu_isa_supported(isa) ||
            !ggml_rows<2>(isa, format.gate_up, mats, row_bytes, kGgmlXSlices, i0, i1, x, ncols, outs)) {
            ggml_row_sums(isa, format.gate_up, record, row_bytes, kGgmlXSlices, i0, i1, x, ncols, g);
            ggml_row_sums(isa, format.gate_up, up, row_bytes, kGgmlXSlices, i0, i1, x, ncols, u);
        }
        for (int i = i0; i < i1; ++i) {
            for (int c = 0; c < ncols; ++c) {
                const std::size_t at = static_cast<std::size_t>(i - i0) * ncols + c;
                h[c][i] = canon::swiglu_bf16(ggml::output_bf16(g[at], x_emax[c], format.gate_up),
                                             ggml::output_bf16(u[at], x_emax[c], format.gate_up));
            }
        }
    }
}

void ggml_down(CpuIsa isa, const std::uint8_t* record, const GgmlExpertFormat& format, const CpuLimbSlice* const* h,
               const int* h_emax, int ncols, int row_begin, int row_end, std::uint16_t* const* y) {
    if (format.down == QType::FP8_E4M3FN_B128) {
        fp8_down(isa, record, h, h_emax, ncols, row_begin, row_end, y);
        return;
    }
    constexpr int kRows = 32;
    ggml::I128 r[kRows * kMaxColumns];
    const std::uint8_t* down = record + 2 * static_cast<std::size_t>(kIntermediate) * format.gate_row_bytes;
    for (int r0 = row_begin; r0 < row_end; r0 += kRows) {
        const int r1 = std::min(row_end, r0 + kRows);
        ggml_row_sums(isa, format.down, down, static_cast<int>(format.down_row_bytes), kGgmlHSlices, r0, r1, h, ncols, r);
        for (int row = r0; row < r1; ++row) {
            for (int c = 0; c < ncols; ++c) {
                y[c][row] = ggml::output_bf16(r[static_cast<std::size_t>(row - r0) * ncols + c], h_emax[c], format.down);
            }
        }
    }
}

void ggml_expert_forward(CpuIsa isa, const std::uint8_t* record, const GgmlExpertFormat& format, int ncols,
                         const std::uint16_t* const* x, std::uint16_t* const* y) {
    require(ncols >= 1 && ncols <= kMaxColumns && ggml_cpu_supported(format), "unsupported expert or width");
    static thread_local CpuLimbSlice xs[kMaxColumns][kGgmlXSlices], hs[kMaxColumns][kGgmlHSlices];
    static thread_local std::uint16_t hv[kMaxColumns][kIntermediate];
    int x_emax[kMaxColumns], h_emax[kMaxColumns];
    const CpuLimbSlice* xp[kMaxColumns];
    const CpuLimbSlice* hp[kMaxColumns];
    std::uint16_t* hvp[kMaxColumns];
    for (int c = 0; c < ncols; ++c) {
        x_emax[c] = ggml_encode_column(x[c], kHidden, xs[c]);
        xp[c]     = xs[c];
        hvp[c]    = hv[c];
    }
    ggml_gate_up(isa, record, format, xp, x_emax, ncols, 0, kIntermediate, hvp);
    for (int c = 0; c < ncols; ++c) {
        h_emax[c] = ggml_encode_column(hv[c], kIntermediate, hs[c]);
        hp[c]     = hs[c];
    }
    ggml_down(isa, record, format, hp, h_emax, ncols, 0, kHidden, y);
}

void ggml_a8_encode_column(const std::uint16_t* v, int n, CpuA8Slice* out) {
    require(n % 32 == 0, "a column is whole 32-element slices");
    for (int s = 0; s < n / 32; ++s) {
        const std::uint16_t* e = v + static_cast<std::size_t>(s) * 32;
        unsigned key           = 0;
        for (int j = 0; j < 32; ++j) { key = std::max(key, static_cast<unsigned>(e[j] & 0x7FFFU)); }
        const a8::SliceScale scale = a8::slice_scale(key);
        CpuA8Slice& o              = out[s];
        std::int32_t sum           = 0;
        for (int j = 0; j < 32; ++j) {
            const int q = a8::code(e[j], scale.inv);
            o.q[j]      = static_cast<std::int16_t>(q);
            sum += q;
        }
        o.d   = scale.d;
        o.sum = sum;
    }
}

void ggml_a8_gate_up(CpuIsa isa, const std::uint8_t* record, const GgmlExpertFormat& format, const CpuA8Slice* const* x,
                     int ncols, int i_begin, int i_end, std::uint16_t* const* h) {
    require(cpu_isa_supported(isa), "unsupported CPU ISA");
    require(a8_supported(format), "not a GGUF expert pair of the 8-bit route");
    require(ncols >= 1 && ncols <= kMaxColumns && 0 <= i_begin && i_begin <= i_end && i_end <= kIntermediate,
            "8-bit gate/up: width or intermediates out of range");
    const int row_bytes = static_cast<int>(format.gate_row_bytes);
    switch (format.gate_up) {
    case QType::GGML_IQ2_S:
        a8_gate_up_format<QType::GGML_IQ2_S>(isa, record, row_bytes, x, ncols, i_begin, i_end, h);
        return;
    case QType::GGML_IQ3_XXS:
        a8_gate_up_format<QType::GGML_IQ3_XXS>(isa, record, row_bytes, x, ncols, i_begin, i_end, h);
        return;
    case QType::GGML_IQ3_S:
        a8_gate_up_format<QType::GGML_IQ3_S>(isa, record, row_bytes, x, ncols, i_begin, i_end, h);
        return;
    case QType::GGML_IQ4_XS:
        a8_gate_up_format<QType::GGML_IQ4_XS>(isa, record, row_bytes, x, ncols, i_begin, i_end, h);
        return;
    default: throw std::invalid_argument("offloaded_moe ggml: not a gate/up format of the 8-bit route");
    }
}

void ggml_a8_down(CpuIsa isa, const std::uint8_t* record, const GgmlExpertFormat& format, const CpuA8Slice* const* h,
                  int ncols, int row_begin, int row_end, std::uint16_t* const* y) {
    require(cpu_isa_supported(isa), "unsupported CPU ISA");
    require(a8_supported(format), "not a GGUF expert pair of the 8-bit route");
    require(ncols >= 1 && ncols <= kMaxColumns && 0 <= row_begin && row_begin <= row_end && row_end <= kHidden,
            "8-bit down: width or rows out of range");
    const std::uint8_t* down = record + 2 * static_cast<std::size_t>(kIntermediate) * format.gate_row_bytes;
    const int row_bytes      = static_cast<int>(format.down_row_bytes);
    switch (format.down) {
    case QType::GGML_IQ4_NL:
        a8_down_format<QType::GGML_IQ4_NL>(isa, down, row_bytes, h, ncols, row_begin, row_end, y);
        return;
    case QType::GGML_Q2_0: a8_down_format<QType::GGML_Q2_0>(isa, down, row_bytes, h, ncols, row_begin, row_end, y); return;
    default: throw std::invalid_argument("offloaded_moe ggml: not a down format of the 8-bit route");
    }
}

void ggml_a8_expert_forward(CpuIsa isa, const std::uint8_t* record, const GgmlExpertFormat& format, int ncols,
                            const std::uint16_t* const* x, std::uint16_t* const* y) {
    require(ncols >= 1 && ncols <= kMaxColumns && a8_supported(format), "unsupported expert or width");
    static thread_local CpuA8Slice xs[kMaxColumns][kGgmlXSlices], hs[kMaxColumns][kGgmlHSlices];
    static thread_local std::uint16_t hv[kMaxColumns][kIntermediate];
    const CpuA8Slice* xp[kMaxColumns];
    const CpuA8Slice* hp[kMaxColumns];
    std::uint16_t* hvp[kMaxColumns];
    for (int c = 0; c < ncols; ++c) {
        ggml_a8_encode_column(x[c], kHidden, xs[c]);
        xp[c]  = xs[c];
        hvp[c] = hv[c];
    }
    ggml_a8_gate_up(isa, record, format, xp, ncols, 0, kIntermediate, hvp);
    for (int c = 0; c < ncols; ++c) {
        ggml_a8_encode_column(hv[c], kIntermediate, hs[c]);
        hp[c] = hs[c];
    }
    ggml_a8_down(isa, record, format, hp, ncols, 0, kHidden, y);
}

} // namespace infernix::ops::offloaded_moe
