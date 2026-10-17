#include "ops/offloaded_sparse_moe/cpu/ggml_expert.h"

#include "core/ggml_format.h"
#include "ops/offloaded_sparse_moe/cpu/fp8_expert.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

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
// artifact, MTP 4 + n-gram 7 at 4K: 202.0 against 200.2 tok/s, 2026-10-16).
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

// Rows [row_begin, row_end) against NC columns. Each unit's lane vectors are kept in a batch of
// eight, reduced together, and each block's exact P = 2048 sum(w * hi) + sum(w * lo) (int64,
// |P| < 2^42) enters R as mant(d) * P * 2^(e(d) - 1). One copy per ISA, each for its target.
#define INFERNIX_GGML_ROW_SUMS(Name, Target, Mad)                                                                      \
    template <QType F, int NC>                                                                                         \
    INFERNIX_TARGET(Target)                                                                                            \
    void Name##_n(const std::uint8_t* matrix, int row_bytes, int slices, int row_begin, int row_end,                   \
                  const CpuLimbSlice* const* columns, int c0, int ncols, ggml::I128* r) {                              \
        constexpr int kUnit          = kUnitSlices<F>;                                                                 \
        constexpr int kUnitsPerBlock = block_slices(F) / kUnit;                                                        \
        static_assert(8 % kUnitsPerBlock == 0, "a batch holds whole blocks");                                         \
        const int units = slices / kUnit;                                                                              \
        for (int row = row_begin; row < row_end; ++row) {                                                              \
            const std::uint8_t* w = matrix + static_cast<std::size_t>(row) * row_bytes;                                \
            ggml::I128 out[NC];                                                                                        \
            __m256i batch_lo[NC][8], batch_hi[NC][8];                                                                  \
            std::uint16_t batch_d[8];                                                                                  \
            int n = 0;                                                                                                 \
            for (int u = 0; u < units; ++u) {                                                                          \
                __m256i lo[NC], hi[NC];                                                                                \
                for (int k = 0; k < NC; ++k) { lo[k] = hi[k] = _mm256_setzero_si256(); }                               \
                std::uint16_t d = 0;                                                                                   \
                for (int j = 0; j < kUnit; ++j) {                                                                      \
                    const int s       = u * kUnit + j;                                                                 \
                    const Decoded dw  = decode<F>(w, s);                                                               \
                    const Weights sw  = scaled(dw);                                                                    \
                    d                 = dw.d;                                                                          \
                    for (int k = 0; k < NC; ++k) {                                                                     \
                        const CpuLimbSlice& l = columns[c0 + k][s];                                                    \
                        const auto* xl = reinterpret_cast<const __m256i*>(l.lo);                                       \
                        const auto* xh = reinterpret_cast<const __m256i*>(l.hi);                                       \
                        lo[k] = Mad::mad(Mad::mad(lo[k], sw.lo, _mm256_load_si256(xl)), sw.hi, _mm256_load_si256(xl + 1)); \
                        hi[k] = Mad::mad(Mad::mad(hi[k], sw.lo, _mm256_load_si256(xh)), sw.hi, _mm256_load_si256(xh + 1)); \
                    }                                                                                                  \
                }                                                                                                      \
                for (int k = 0; k < NC; ++k) {                                                                         \
                    batch_lo[k][n] = lo[k];                                                                            \
                    batch_hi[k][n] = hi[k];                                                                            \
                }                                                                                                      \
                batch_d[n++] = d;                                                                                      \
                if (n == 8 || u + 1 == units) {                                                                        \
                    for (int k = 0; k < NC; ++k) {                                                                     \
                        for (int b = n; b < 8; ++b) { batch_lo[k][b] = batch_hi[k][b] = _mm256_setzero_si256(); }      \
                        alignas(32) std::int32_t sl[8], sh[8];                                                         \
                        _mm256_store_si256(reinterpret_cast<__m256i*>(sl), hsum8(batch_lo[k]));                       \
                        _mm256_store_si256(reinterpret_cast<__m256i*>(sh), hsum8(batch_hi[k]));                       \
                        for (int b = 0; b < n; b += kUnitsPerBlock) {                                                  \
                            std::int64_t p = 0;                                                                        \
                            for (int i = 0; i < kUnitsPerBlock; ++i) {                                                 \
                                p += static_cast<std::int64_t>(sh[b + i]) * 2048 + sl[b + i];                          \
                            }                                                                                          \
                            std::int64_t mant;                                                                         \
                            int shift;                                                                                 \
                            bool negative;                                                                             \
                            ggml::f16_parts(batch_d[b], mant, shift, negative);                                        \
                            ggml::add_shifted(out[k], (negative ? -mant : mant) * p, shift);                           \
                        }                                                                                              \
                    }                                                                                                  \
                    n = 0;                                                                                             \
                }                                                                                                      \
            }                                                                                                          \
            for (int k = 0; k < NC; ++k) { r[static_cast<std::size_t>(row - row_begin) * ncols + c0 + k] = out[k]; }  \
        }                                                                                                              \
    }                                                                                                                  \
    template <QType F>                                                                                                 \
    void Name(const std::uint8_t* matrix, int row_bytes, int slices, int row_begin, int row_end,                       \
              const CpuLimbSlice* const* columns, int ncols, ggml::I128* r) {                                          \
        for (int c0 = 0; c0 < ncols; c0 += 4) {                                                                        \
            switch (std::min(4, ncols - c0)) {                                                                         \
            case 1: Name##_n<F, 1>(matrix, row_bytes, slices, row_begin, row_end, columns, c0, ncols, r); break;       \
            case 2: Name##_n<F, 2>(matrix, row_bytes, slices, row_begin, row_end, columns, c0, ncols, r); break;       \
            case 3: Name##_n<F, 3>(matrix, row_bytes, slices, row_begin, row_end, columns, c0, ncols, r); break;       \
            default: Name##_n<F, 4>(matrix, row_bytes, slices, row_begin, row_end, columns, c0, ncols, r); break;      \
            }                                                                                                          \
        }                                                                                                              \
    }

INFERNIX_GGML_ROW_SUMS(row_sums_avx2, "avx2", Avx2Mad)
INFERNIX_GGML_ROW_SUMS(row_sums_avxvnni, "avx2,avxvnni", AvxVnniMad)
#undef INFERNIX_GGML_ROW_SUMS

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

template <QType F>
void row_sums_format(CpuIsa isa, const std::uint8_t* matrix, int row_bytes, int slices, int row_begin, int row_end,
                     const CpuLimbSlice* const* columns, int ncols, ggml::I128* r) {
    switch (isa) {
#if defined(INFERNIX_GGML_X86)
    case CpuIsa::kAvx2: row_sums_avx2<F>(matrix, row_bytes, slices, row_begin, row_end, columns, ncols, r); return;
    case CpuIsa::kAvxVnni: row_sums_avxvnni<F>(matrix, row_bytes, slices, row_begin, row_end, columns, ncols, r); return;
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
#define INFERNIX_GGML_FORMAT(Q)                                                                                        \
    case Q:                                                                                                            \
        require(slices % (block_slices(Q) > 4 ? 4 : block_slices(Q)) == 0, "K is not whole units");                  \
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
        ggml_row_sums(isa, format.gate_up, record, row_bytes, kGgmlXSlices, i0, i1, x, ncols, g);
        ggml_row_sums(isa, format.gate_up, up, row_bytes, kGgmlXSlices, i0, i1, x, ncols, u);
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

} // namespace infernix::ops::offloaded_moe
