#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#    include <immintrin.h>
#    if defined(_MSC_VER)
#        include <intrin.h>
#    else
#        include <cpuid.h>
#    endif
#    define INFERNIX_MOE_X86 1
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#    define INFERNIX_TARGET(x)
#else
#    define INFERNIX_TARGET(x) __attribute__((target(x)))
#endif

namespace infernix::ops::offloaded_moe {
namespace {

using canon::A4Block;

// Code nibble of row r (0..15) at k (0..15) inside one 144-byte unit (design §6.2).
inline unsigned unit_code(const std::uint8_t* unit, int r, int k) {
    const std::uint8_t byte = unit[32 * (k / 4) + 4 * (r % 8) + k % 4];
    return r < 8 ? (byte & 15U) : (byte >> 4);
}

void row_sums_scalar(const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end,
                     const A4Block* const* acts, int ncols, std::int64_t* out) {
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        for (int c = 0; c < ncols; ++c) {
            std::int64_t s[16] = {};
            for (int b = 0; b < blocks; ++b) {
                const std::uint8_t* unit =
                    matrix + (static_cast<std::size_t>(rg) * blocks + b) * kUnitBytes;
                const A4Block& a = acts[c][b];
                for (int r = 0; r < 16; ++r) {
                    std::int32_t p = 0;
                    for (int k = 0; k < 16; ++k) { p += canon::e2m1_x2(unit_code(unit, r, k)) * a.c2[k]; }
                    s[r] += static_cast<std::int64_t>(p * a.scale_scaled) * canon::e4m3_scaled(unit[128 + r]);
                }
            }
            for (int r = 0; r < 16; ++r) { out[(static_cast<std::size_t>(rg - rg_begin) * 16 + r) * ncols + c] = s[r]; }
        }
    }
}

#if defined(INFERNIX_MOE_X86)

// Software prefetch `distance` bytes ahead along the unit stream (design §10.3). The best distance
// depends on the host's memory latency and core speed, so calibration chooses it (§14.2).
inline void prefetch_unit(const std::uint8_t* unit, int distance) {
    if (distance > 0) {
        const char* p = reinterpret_cast<const char*>(unit) + distance;
        _mm_prefetch(p, _MM_HINT_T0);
        _mm_prefetch(p + 64, _MM_HINT_T0);
        _mm_prefetch(p + 128, _MM_HINT_T0);
    }
}

// u8 table: 2 * e2m1(code) + 12, so a weight is unsigned for vpdpbusd / vpmaddubsw.
constexpr std::array<std::int8_t, 16> kBiasedCodes = {12, 13, 14, 15, 16, 18, 20, 24,
                                                      12, 11, 10, 9,  8,  6,  4,  0};
constexpr int kBias = 12;

inline std::int32_t quad_word(const A4Block& a, int q) {
    std::int32_t w;
    std::memcpy(&w, &a.c2[4 * q], 4);
    return w;
}

// ----------------------------------------------------------------------------------- AVX-512

INFERNIX_TARGET("avx512f,avx512bw,avx512vnni")
inline __m512i scales16_avx512(const std::uint8_t* s) {
    const __m512i w    = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s)));
    const __m512i m    = _mm512_and_si512(w, _mm512_set1_epi32(7));
    const __m512i e    = _mm512_and_si512(_mm512_srli_epi32(w, 3), _mm512_set1_epi32(15));
    const __m512i norm = _mm512_sllv_epi32(_mm512_or_si512(m, _mm512_set1_epi32(8)),
                                           _mm512_sub_epi32(e, _mm512_set1_epi32(1)));
    const __mmask16 sub = _mm512_cmpeq_epi32_mask(e, _mm512_setzero_si512());
    return _mm512_mask_blend_epi32(sub, norm, m);
}

template <int N>
INFERNIX_TARGET("avx512f,avx512bw,avx512vnni")
void row_sums_avx512(const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end,
                     const A4Block* const* acts, std::int64_t* out, int prefetch_bytes) {
    const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kBiasedCodes.data()));
    const __m512i lut    = _mm512_broadcast_i32x4(lut128);
    const __m512i low4   = _mm512_set1_epi8(0x0F);
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        __m512i even[N], odd[N];
        for (int c = 0; c < N; ++c) { even[c] = odd[c] = _mm512_setzero_si512(); }
        for (int b = 0; b < blocks; ++b) {
            const std::uint8_t* unit = matrix + (static_cast<std::size_t>(rg) * blocks + b) * kUnitBytes;
            prefetch_unit(unit, prefetch_bytes);
            __m512i p[N];
            for (int c = 0; c < N; ++c) { p[c] = _mm512_setzero_si512(); }
            for (int q = 0; q < 4; ++q) {
                __m512i v = _mm512_broadcast_i64x4(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(unit + 32 * q)));
                v         = _mm512_mask_srli_epi16(v, 0xFFFF0000U, v, 4); // upper half: high nibbles
                v         = _mm512_shuffle_epi8(lut, _mm512_and_si512(v, low4));
                for (int c = 0; c < N; ++c) {
                    p[c] = _mm512_dpbusd_epi32(p[c], v, _mm512_set1_epi32(quad_word(acts[c][b], q)));
                }
            }
            const __m512i sw     = scales16_avx512(unit + 128);
            const __m512i sw_odd = _mm512_srli_epi64(sw, 32);
            for (int c = 0; c < N; ++c) {
                const A4Block& a = acts[c][b];
                const __m512i pc = _mm512_sub_epi32(p[c], _mm512_set1_epi32(kBias * a.c2_sum));
                const __m512i pa = _mm512_mullo_epi32(pc, _mm512_set1_epi32(a.scale_scaled));
                even[c] = _mm512_add_epi64(even[c], _mm512_mul_epi32(pa, sw));
                odd[c]  = _mm512_add_epi64(odd[c], _mm512_mul_epi32(_mm512_srli_epi64(pa, 32), sw_odd));
            }
        }
        for (int c = 0; c < N; ++c) {
            alignas(64) std::int64_t e[8], o[8];
            _mm512_store_si512(reinterpret_cast<__m512i*>(e), even[c]);
            _mm512_store_si512(reinterpret_cast<__m512i*>(o), odd[c]);
            for (int i = 0; i < 8; ++i) {
                const std::size_t base = static_cast<std::size_t>(rg - rg_begin) * 16;
                out[(base + 2 * i) * N + c]     = e[i];
                out[(base + 2 * i + 1) * N + c] = o[i];
            }
        }
    }
}

// ----------------------------------------------------------------------------------- AVX2

INFERNIX_TARGET("avx2")
inline __m256i scales8_avx2(const std::uint8_t* s) {
    const __m256i w    = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(s)));
    const __m256i m    = _mm256_and_si256(w, _mm256_set1_epi32(7));
    const __m256i e    = _mm256_and_si256(_mm256_srli_epi32(w, 3), _mm256_set1_epi32(15));
    const __m256i norm = _mm256_sllv_epi32(_mm256_or_si256(m, _mm256_set1_epi32(8)),
                                           _mm256_sub_epi32(e, _mm256_set1_epi32(1)));
    const __m256i sub  = _mm256_cmpeq_epi32(e, _mm256_setzero_si256());
    return _mm256_blendv_epi8(norm, m, sub);
}

INFERNIX_TARGET("avx2")
inline __m256i dot4_avx2(__m256i acc, __m256i w, __m256i a) {
    const __m256i pairs = _mm256_maddubs_epi16(w, a); // |pair| <= 2 * 24 * 12: no saturation
    return _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
}

INFERNIX_TARGET("avx2,avxvnni")
inline __m256i dot4_avxvnni(__m256i acc, __m256i w, __m256i a) {
    return _mm256_dpbusd_avx_epi32(acc, w, a);
}

// One body, two target attributes: AVX-VNNI must not leak into the plain AVX2 variant.
#define INFERNIX_AVX2_ROW_SUMS(NAME, DOT) \
void NAME(const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end, \
                   const A4Block* const* acts, std::int64_t* out, int prefetch_bytes) { \
    const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kBiasedCodes.data())); \
    const __m256i lut    = _mm256_broadcastsi128_si256(lut128); \
    const __m256i low4   = _mm256_set1_epi8(0x0F); \
    for (int rg = rg_begin; rg < rg_end; ++rg) { \
        __m256i even[2][N], odd[2][N]; \
        for (int h = 0; h < 2; ++h) { \
            for (int c = 0; c < N; ++c) { even[h][c] = odd[h][c] = _mm256_setzero_si256(); } \
        } \
        for (int b = 0; b < blocks; ++b) { \
            const std::uint8_t* unit = matrix + (static_cast<std::size_t>(rg) * blocks + b) * kUnitBytes; \
            prefetch_unit(unit, prefetch_bytes); \
            __m256i p[2][N]; \
            for (int h = 0; h < 2; ++h) { \
                for (int c = 0; c < N; ++c) { p[h][c] = _mm256_setzero_si256(); } \
            } \
            for (int q = 0; q < 4; ++q) { \
                const __m256i v  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(unit + 32 * q)); \
                const __m256i lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(v, low4)); \
                const __m256i hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v, 4), low4)); \
                for (int c = 0; c < N; ++c) { \
                    const __m256i a = _mm256_set1_epi32(quad_word(acts[c][b], q)); \
                    p[0][c] = DOT(p[0][c], lo, a); \
                    p[1][c] = DOT(p[1][c], hi, a); \
                } \
            } \
            for (int h = 0; h < 2; ++h) { \
                const __m256i sw     = scales8_avx2(unit + 128 + 8 * h); \
                const __m256i sw_odd = _mm256_srli_epi64(sw, 32); \
                for (int c = 0; c < N; ++c) { \
                    const A4Block& a = acts[c][b]; \
                    const __m256i pc = _mm256_sub_epi32(p[h][c], _mm256_set1_epi32(kBias * a.c2_sum)); \
                    const __m256i pa = _mm256_mullo_epi32(pc, _mm256_set1_epi32(a.scale_scaled)); \
                    even[h][c] = _mm256_add_epi64(even[h][c], _mm256_mul_epi32(pa, sw)); \
                    odd[h][c]  = _mm256_add_epi64(odd[h][c], _mm256_mul_epi32(_mm256_srli_epi64(pa, 32), sw_odd)); \
                } \
            } \
        } \
        for (int h = 0; h < 2; ++h) { \
            for (int c = 0; c < N; ++c) { \
                alignas(32) std::int64_t e[4], o[4]; \
                _mm256_store_si256(reinterpret_cast<__m256i*>(e), even[h][c]); \
                _mm256_store_si256(reinterpret_cast<__m256i*>(o), odd[h][c]); \
                for (int i = 0; i < 4; ++i) { \
                    const std::size_t base = static_cast<std::size_t>(rg - rg_begin) * 16 + 8 * h; \
                    out[(base + 2 * i) * N + c]     = e[i]; \
                    out[(base + 2 * i + 1) * N + c] = o[i]; \
                } \
            } \
        } \
    } \
}

template <int N>
INFERNIX_TARGET("avx2")
INFERNIX_AVX2_ROW_SUMS(row_sums_avx2, dot4_avx2)

template <int N>
INFERNIX_TARGET("avx2,avxvnni")
INFERNIX_AVX2_ROW_SUMS(row_sums_avxvnni, dot4_avxvnni)

#undef INFERNIX_AVX2_ROW_SUMS

// --------------------------------------------------------------- AVX2 canonical functions
// canonical_math.h's exp_c, silu_c, f32_to_bf16_rn, e4m3_rn_satfinite and quantize_a4_block on eight
// lanes: the same IEEE operations in the same order (vdivps, vaddps and vmulps round like the scalar
// operations, and vfmadd once like std::fmaf), so every lane equals the scalar function bit for bit,
// NaN lanes included. Every vector ISA here has AVX2 with FMA (detect_features).

INFERNIX_TARGET("avx2")
inline __m256 bf16x8_to_f32(const std::uint16_t* p) {
    const __m256i w = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    return _mm256_castsi256_ps(_mm256_slli_epi32(w, 16));
}

INFERNIX_TARGET("avx2")
inline __m256 canonical_nan8() { return _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(canon::kCanonicalNan))); }

INFERNIX_TARGET("avx2,fma")
inline __m256 exp_c_avx2(__m256 x) {
    const __m256 magic = _mm256_set1_ps(12582912.0F);
    const __m256 n  = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269502162933349609375F)), magic), magic);
    const __m256 r0 = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693145751953125F), x); // fma(-n, c, x)
    const __m256 r  = _mm256_fnmadd_ps(n, _mm256_set1_ps(1.428606765330187045e-06F), r0);
    __m256 p = _mm256_set1_ps(1.0F / 5040.0F);
    p        = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F / 720.0F));
    p        = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F / 120.0F));
    p        = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F / 24.0F));
    p        = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F / 6.0F));
    p        = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5F));
    p        = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F));
    p        = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0F));
    // 2^n in two exact steps; k / 2 truncates toward zero like the scalar division. Lanes outside the
    // range below hold garbage here and take their special value at the end.
    const __m256i k  = _mm256_cvttps_epi32(n);
    const __m256i k1 = _mm256_srai_epi32(_mm256_add_epi32(k, _mm256_srli_epi32(k, 31)), 1);
    const __m256i k2 = _mm256_sub_epi32(k, k1);
    const __m256i bias = _mm256_set1_epi32(127);
    const __m256 s1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(k1, bias), 23));
    const __m256 s2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(k2, bias), 23));
    __m256 y = _mm256_mul_ps(_mm256_mul_ps(p, s1), s2);
    y = _mm256_blendv_ps(y, _mm256_castsi256_ps(_mm256_set1_epi32(0x7F800000)),
                         _mm256_cmp_ps(x, _mm256_set1_ps(88.72283935546875F), _CMP_GT_OQ));
    y = _mm256_blendv_ps(y, _mm256_setzero_ps(), _mm256_cmp_ps(x, _mm256_set1_ps(-103.97208404541015625F), _CMP_LT_OQ));
    return _mm256_blendv_ps(y, canonical_nan8(), _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
}

// silu_c: g >= 0 takes g / (1 + exp_c(-g)), else (g e) / (1 + e) with e = exp_c(g).
INFERNIX_TARGET("avx2,fma")
inline __m256 silu_c_avx2(__m256 g) {
    const __m256 nonneg = _mm256_cmp_ps(g, _mm256_setzero_ps(), _CMP_GE_OQ);
    const __m256 e      = exp_c_avx2(_mm256_blendv_ps(g, _mm256_xor_ps(g, _mm256_set1_ps(-0.0F)), nonneg));
    const __m256 num    = _mm256_blendv_ps(_mm256_mul_ps(g, e), g, nonneg);
    const __m256 r      = _mm256_div_ps(num, _mm256_add_ps(_mm256_set1_ps(1.0F), e));
    return _mm256_blendv_ps(r, canonical_nan8(), _mm256_cmp_ps(r, r, _CMP_UNORD_Q));
}

// f32_to_bf16_rn, the BF16 word in each 32-bit lane.
INFERNIX_TARGET("avx2")
inline __m256i f32_to_bf16_rn_avx2(__m256 x) {
    const __m256i u   = _mm256_castps_si256(x);
    const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
    const __m256i r   = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(u, _mm256_set1_epi32(0x7FFF)), lsb), 16);
    return _mm256_blendv_epi8(r, _mm256_set1_epi32(0x7FC0), _mm256_castps_si256(_mm256_cmp_ps(x, x, _CMP_UNORD_Q)));
}

INFERNIX_TARGET("avx2,fma")
int swiglu_avx2(const std::uint16_t* g, const std::uint16_t* u, int n, std::uint16_t* h) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256i w = f32_to_bf16_rn_avx2(_mm256_mul_ps(silu_c_avx2(bf16x8_to_f32(g + i)), bf16x8_to_f32(u + i)));
        // Words 0..3 and 4..7 sit in quads 0 and 2 after the in-lane pack.
        _mm_storeu_si128(reinterpret_cast<__m128i*>(h + i),
                         _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi32(w, w), 0x08)));
    }
    return i;
}

// e4m3_rn_satfinite of non-negative lanes (NaN gives 0x7F), one word per 32-bit lane.
INFERNIX_TARGET("avx2")
inline __m256i e4m3_rn_satfinite_avx2(__m256 x) {
    const __m256i u   = _mm256_and_si256(_mm256_castps_si256(x), _mm256_set1_epi32(0x7FFFFFFF));
    const __m256i exp = _mm256_srli_epi32(u, 23);
    // Subnormal E4M3 (exp < 121): the value in units of 2^-9, rounded to nearest even by rint_rn.
    const __m256 magic = _mm256_set1_ps(12582912.0F);
    const __m256 q     = _mm256_mul_ps(_mm256_castsi256_ps(u), _mm256_set1_ps(512.0F));
    const __m256i sub_word = _mm256_cvttps_epi32(_mm256_sub_ps(_mm256_add_ps(q, magic), magic));
    // Normal: exponent and the 3 kept mantissa bits are u >> 20; a round-up carries into the exponent.
    const __m256i rest  = _mm256_and_si256(u, _mm256_set1_epi32(0xFFFFF));
    const __m256i half  = _mm256_set1_epi32(0x80000);
    const __m256i odd   = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_srli_epi32(u, 20), _mm256_set1_epi32(1)), _mm256_set1_epi32(1));
    const __m256i up    = _mm256_or_si256(_mm256_cmpgt_epi32(rest, half), _mm256_and_si256(_mm256_cmpeq_epi32(rest, half), odd));
    const __m256i norm  = _mm256_min_epi32(_mm256_sub_epi32(_mm256_sub_epi32(_mm256_srli_epi32(u, 20), _mm256_set1_epi32(120 << 3)), up),
                                           _mm256_set1_epi32(0x7E));
    __m256i word = _mm256_blendv_epi8(norm, sub_word, _mm256_cmpgt_epi32(_mm256_set1_epi32(121), exp));
    word = _mm256_blendv_epi8(word, _mm256_set1_epi32(0x7E), _mm256_cmpgt_epi32(u, _mm256_set1_epi32(0x43DFFFFF)));
    return _mm256_blendv_epi8(word, _mm256_set1_epi32(0x7F), _mm256_cmpgt_epi32(u, _mm256_set1_epi32(0x7F800000)));
}

// e4m3_scaled of each lane's word.
INFERNIX_TARGET("avx2")
inline __m256i e4m3_scaled_avx2(__m256i w) {
    const __m256i m    = _mm256_and_si256(w, _mm256_set1_epi32(7));
    const __m256i e    = _mm256_and_si256(_mm256_srli_epi32(w, 3), _mm256_set1_epi32(15));
    const __m256i norm = _mm256_sllv_epi32(_mm256_or_si256(m, _mm256_set1_epi32(8)), _mm256_sub_epi32(e, _mm256_set1_epi32(1)));
    return _mm256_blendv_epi8(norm, m, _mm256_cmpeq_epi32(e, _mm256_setzero_si256()));
}

// e2m1_x2(e2m1_rn_satfinite(q)) per lane. The scalar if-chain's code is the number of its failed
// tests (all of them for NaN), so each unordered-true negated compare adds one code step: steps 1..4
// add 1 to the doubled value, 5 and 6 add 2, 7 adds 4; the sign of q (-0 included) negates it.
INFERNIX_TARGET("avx2")
inline __m256i e2m1_x2_avx2(__m256 q) {
    const __m256 a = _mm256_and_ps(q, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    const __m256i s1 = _mm256_castps_si256(_mm256_cmp_ps(a, _mm256_set1_ps(0.25F), _CMP_NLE_UQ));
    const __m256i s2 = _mm256_castps_si256(_mm256_cmp_ps(a, _mm256_set1_ps(0.75F), _CMP_NLT_UQ));
    const __m256i s3 = _mm256_castps_si256(_mm256_cmp_ps(a, _mm256_set1_ps(1.25F), _CMP_NLE_UQ));
    const __m256i s4 = _mm256_castps_si256(_mm256_cmp_ps(a, _mm256_set1_ps(1.75F), _CMP_NLT_UQ));
    const __m256i s5 = _mm256_castps_si256(_mm256_cmp_ps(a, _mm256_set1_ps(2.5F), _CMP_NLE_UQ));
    const __m256i s6 = _mm256_castps_si256(_mm256_cmp_ps(a, _mm256_set1_ps(3.5F), _CMP_NLT_UQ));
    const __m256i s7 = _mm256_castps_si256(_mm256_cmp_ps(a, _mm256_set1_ps(5.0F), _CMP_NLE_UQ));
    const __m256i two   = _mm256_set1_epi32(2);
    const __m256i minus = _mm256_add_epi32(_mm256_add_epi32(s1, s2), _mm256_add_epi32(s3, s4)); // -(steps 1..4)
    const __m256i plus  = _mm256_add_epi32(_mm256_add_epi32(_mm256_and_si256(s5, two), _mm256_and_si256(s6, two)),
                                           _mm256_and_si256(s7, _mm256_set1_epi32(4)));
    return _mm256_sign_epi32(_mm256_sub_epi32(plus, minus), _mm256_castps_si256(q));
}

// Lane b = the maximum over the eight lanes of m[b] (non-NaN lanes, so the order is free).
INFERNIX_TARGET("avx2")
inline __m256 lane_max8(const __m256 (&m)[8]) {
    __m256 t[4];
    for (int i = 0; i < 4; ++i) {
        t[i] = _mm256_max_ps(_mm256_unpacklo_ps(m[2 * i], m[2 * i + 1]), _mm256_unpackhi_ps(m[2 * i], m[2 * i + 1]));
    }
    const __m256 u0 = _mm256_max_ps(_mm256_shuffle_ps(t[0], t[1], 0x44), _mm256_shuffle_ps(t[0], t[1], 0xEE));
    const __m256 u1 = _mm256_max_ps(_mm256_shuffle_ps(t[2], t[3], 0x44), _mm256_shuffle_ps(t[2], t[3], 0xEE));
    return _mm256_max_ps(_mm256_permute2f128_ps(u0, u1, 0x20), _mm256_permute2f128_ps(u0, u1, 0x31));
}

// quantize_a4_block for `blocks` consecutive blocks, eight at a time: the eight block scales in one
// vector, then each block's 16 codes in two.
INFERNIX_TARGET("avx2,fma")
void quantize_a4_avx2(const std::uint16_t* v, int blocks, float input_scale, A4Block* out) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 six_g    = _mm256_set1_ps(canon::mul_rn(6.0F, input_scale));
    for (int b0 = 0; b0 < blocks; b0 += 8) {
        const int nb = blocks - b0 < 8 ? blocks - b0 : 8;
        __m256 m[8];
        for (int b = 0; b < 8; ++b) {
            m[b] = _mm256_setzero_ps();
            if (b < nb) {
                // amax = a > amax ? a : amax, which is vmaxps(a, amax): NaN elements are skipped.
                const std::uint16_t* p = v + 16 * static_cast<std::size_t>(b0 + b);
                m[b] = _mm256_max_ps(_mm256_and_ps(bf16x8_to_f32(p), abs_mask), m[b]);
                m[b] = _mm256_max_ps(_mm256_and_ps(bf16x8_to_f32(p + 8), abs_mask), m[b]);
            }
        }
        const __m256i words  = e4m3_rn_satfinite_avx2(_mm256_div_ps(lane_max8(m), six_g));
        const __m256i scaled = e4m3_scaled_avx2(words);
        const __m256 d = _mm256_mul_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(scaled), _mm256_set1_ps(1.0F / 512.0F)),
                                       _mm256_set1_ps(input_scale));
        alignas(32) std::int32_t word_of[8], scaled_of[8];
        alignas(32) float d_of[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(word_of), words);
        _mm256_store_si256(reinterpret_cast<__m256i*>(scaled_of), scaled);
        _mm256_store_ps(d_of, d);
        for (int b = 0; b < nb; ++b) {
            A4Block& o     = out[b0 + b];
            o.scale_word   = static_cast<std::uint8_t>(word_of[b]);
            o.scale_scaled = scaled_of[b];
            if (o.scale_scaled == 0) {
                std::memset(o.c2, 0, sizeof o.c2);
                o.c2_sum = 0;
                continue;
            }
            const std::uint16_t* p = v + 16 * static_cast<std::size_t>(b0 + b);
            const __m256 db    = _mm256_broadcast_ss(&d_of[b]);
            const __m256i c_lo = e2m1_x2_avx2(_mm256_div_ps(bf16x8_to_f32(p), db));
            const __m256i c_hi = e2m1_x2_avx2(_mm256_div_ps(bf16x8_to_f32(p + 8), db));
            const __m128i w0 = _mm_packs_epi32(_mm256_castsi256_si128(c_lo), _mm256_extracti128_si256(c_lo, 1));
            const __m128i w1 = _mm_packs_epi32(_mm256_castsi256_si128(c_hi), _mm256_extracti128_si256(c_hi, 1));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(o.c2), _mm_packs_epi16(w0, w1));
            const __m256i s8 = _mm256_add_epi32(c_lo, c_hi);
            __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s8), _mm256_extracti128_si256(s8, 1));
            s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
            s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
            o.c2_sum = _mm_cvtsi128_si32(s4);
        }
    }
}

#define INFERNIX_MOE_DISPATCH(CALL)                                                                   \
    switch (ncols) {                                                                                \
    case 1: CALL(1); break;                                                                         \
    case 2: CALL(2); break;                                                                         \
    case 3: CALL(3); break;                                                                         \
    case 4: CALL(4); break;                                                                         \
    case 5: CALL(5); break;                                                                         \
    case 6: CALL(6); break;                                                                         \
    case 7: CALL(7); break;                                                                         \
    case 8: CALL(8); break;                                                                         \
    default: throw std::invalid_argument("offloaded_moe: ncols must be in [1, 8]");                 \
    }

struct CpuFeatures {
    bool avx2 = false, avxvnni = false, avx512 = false;
};

CpuFeatures detect_features() {
    CpuFeatures f;
    unsigned a = 0, b = 0, c = 0, d = 0;
#    if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    const unsigned max_leaf = static_cast<unsigned>(r[0]);
    __cpuid(r, 1);
    c = static_cast<unsigned>(r[2]);
    const unsigned leaf1_c = c;
    const bool osxsave = (c & (1U << 27)) != 0;
    const unsigned long long xcr0 = osxsave ? _xgetbv(0) : 0;
    if (max_leaf < 7) { return f; }
    __cpuidex(r, 7, 0);
    b = static_cast<unsigned>(r[1]);
    c = static_cast<unsigned>(r[2]);
    __cpuidex(r, 7, 1);
    a = static_cast<unsigned>(r[0]);
#    else
    if (__get_cpuid_max(0, nullptr) < 7) { return f; }
    __cpuid(1, a, b, c, d);
    const unsigned leaf1_c = c;
    const bool osxsave = (c & (1U << 27)) != 0;
    unsigned long long xcr0 = 0;
    if (osxsave) {
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        xcr0 = (static_cast<unsigned long long>(hi) << 32) | lo;
    }
    __cpuid_count(7, 0, a, b, c, d);
    const unsigned b7 = b, c7 = c;
    __cpuid_count(7, 1, a, b, c, d);
    b = b7;
    c = c7;
#    endif
    const bool ymm = (xcr0 & 0x6) == 0x6;
    const bool zmm = (xcr0 & 0xE6) == 0xE6;
    // AVX2 (leaf 7 EBX 5) with FMA (leaf 1 ECX 12) and F16C (ECX 29): the GGUF 8-bit route's terms and weight
    // scales (ggml_expert.cpp); every AVX2 CPU has both, but they are separate CPUID bits.
    f.avx2    = ymm && (b & (1U << 5)) != 0 && (leaf1_c & (1U << 12)) != 0 && (leaf1_c & (1U << 29)) != 0;
    f.avxvnni = f.avx2 && (a & (1U << 4)) != 0;
    // AVX-512 F (EBX 16), BW (EBX 30), VL (EBX 31; the GGUF and FP8 routes' 256-bit EVEX forms) and
    // VNNI (ECX 11), with the OS saving the opmask and ZMM state.
    f.avx512 = f.avx2 && zmm && (b & (1U << 16)) != 0 && (b & (1U << 30)) != 0 && (b & (1U << 31)) != 0 &&
               (c & (1U << 11)) != 0;
    return f;
}

const CpuFeatures& features() {
    static const CpuFeatures f = detect_features();
    return f;
}

#endif // INFERNIX_MOE_X86

} // namespace

const char* cpu_isa_name(CpuIsa isa) {
    switch (isa) {
    case CpuIsa::kScalar: return "scalar";
    case CpuIsa::kAvx2: return "avx2";
    case CpuIsa::kAvxVnni: return "avx-vnni";
    case CpuIsa::kAvx512Vnni: return "avx512-vnni";
    }
    return "unknown";
}

bool cpu_isa_supported(CpuIsa isa) {
#if defined(INFERNIX_MOE_X86)
    switch (isa) {
    case CpuIsa::kScalar: return true;
    case CpuIsa::kAvx2: return features().avx2;
    case CpuIsa::kAvxVnni: return features().avxvnni;
    case CpuIsa::kAvx512Vnni: return features().avx512;
    }
    return false;
#else
    return isa == CpuIsa::kScalar;
#endif
}

CpuIsa best_cpu_isa() {
    for (CpuIsa isa : {CpuIsa::kAvx512Vnni, CpuIsa::kAvxVnni, CpuIsa::kAvx2}) {
        if (cpu_isa_supported(isa)) { return isa; }
    }
    return CpuIsa::kScalar;
}

void rg16_row_sums(CpuIsa isa, const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end,
                   const canon::A4Block* const* acts, int ncols, std::int64_t* out, int prefetch_bytes) {
    if (ncols < 1 || ncols > kMaxColumns) { throw std::invalid_argument("offloaded_moe: ncols must be in [1, 8]"); }
    if (!cpu_isa_supported(isa)) { throw std::invalid_argument("offloaded_moe: unsupported CPU ISA"); }
    switch (isa) {
    case CpuIsa::kScalar: row_sums_scalar(matrix, blocks, rg_begin, rg_end, acts, ncols, out); return;
#if defined(INFERNIX_MOE_X86)
    case CpuIsa::kAvx512Vnni:
#    define INFERNIX_CALL(n) row_sums_avx512<n>(matrix, blocks, rg_begin, rg_end, acts, out, prefetch_bytes)
        INFERNIX_MOE_DISPATCH(INFERNIX_CALL)
#    undef INFERNIX_CALL
        return;
    case CpuIsa::kAvxVnni:
#    define INFERNIX_CALL(n) row_sums_avxvnni<n>(matrix, blocks, rg_begin, rg_end, acts, out, prefetch_bytes)
        INFERNIX_MOE_DISPATCH(INFERNIX_CALL)
#    undef INFERNIX_CALL
        return;
    case CpuIsa::kAvx2:
#    define INFERNIX_CALL(n) row_sums_avx2<n>(matrix, blocks, rg_begin, rg_end, acts, out, prefetch_bytes)
        INFERNIX_MOE_DISPATCH(INFERNIX_CALL)
#    undef INFERNIX_CALL
        return;
#else
    default: break;
#endif
    }
    throw std::invalid_argument("offloaded_moe: unsupported CPU ISA");
}

void quantize_a4(CpuIsa isa, const std::uint16_t* v, int n, float input_scale, canon::A4Block* out) {
#if defined(INFERNIX_MOE_X86)
    if (isa != CpuIsa::kScalar) {
        quantize_a4_avx2(v, n / 16, input_scale, out);
        return;
    }
#else
    (void)isa;
#endif
    for (int b = 0; b < n / 16; ++b) { out[b] = canon::quantize_a4_block(v + 16 * b, input_scale); }
}

void swiglu_bf16_n(CpuIsa isa, const std::uint16_t* g, const std::uint16_t* u, int n, std::uint16_t* h) {
    int i = 0;
#if defined(INFERNIX_MOE_X86)
    if (isa != CpuIsa::kScalar) { i = swiglu_avx2(g, u, n, h); }
#else
    (void)isa;
#endif
    for (; i < n; ++i) { h[i] = canon::swiglu_bf16(g[i], u[i]); }
}

void gate_up_units(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales,
                   const canon::A4Block* const* x_gate, const canon::A4Block* const* x_up,
                   int ncols, int unit_begin, int unit_end, canon::A4Block* const* h_blocks,
                   int prefetch_bytes) {
    // No allocation on the miss path: one unit's sums fit on the stack.
    std::int64_t s_gate[32 * kMaxColumns];
    std::int64_t s_up[32 * kMaxColumns];
    for (int u = unit_begin; u < unit_end; ++u) {
        // Unit u is row groups 2u, 2u+1: rows 32u..32u+31, gate_i at even rows, up_i at odd rows.
        rg16_row_sums(isa, record, kGateUpBlocks, 2 * u, 2 * u + 2, x_gate, ncols, s_gate, prefetch_bytes);
        const std::int64_t* up_sums = s_gate;
        if (x_up != x_gate) {
            rg16_row_sums(isa, record, kGateUpBlocks, 2 * u, 2 * u + 2, x_up, ncols, s_up, prefetch_bytes);
            up_sums = s_up;
        }
        std::uint16_t g[16 * kMaxColumns], p[16 * kMaxColumns], h[16 * kMaxColumns];
        for (int c = 0; c < ncols; ++c) {
            for (int i = 0; i < 16; ++i) {
                g[16 * c + i] = canon::a4_row_output(s_gate[static_cast<std::size_t>(2 * i) * ncols + c], scales.alpha_gate);
                p[16 * c + i] = canon::a4_row_output(up_sums[static_cast<std::size_t>(2 * i + 1) * ncols + c], scales.alpha_up);
            }
        }
        swiglu_bf16_n(isa, g, p, 16 * ncols, h);
        canon::A4Block hq[kMaxColumns];
        quantize_a4(isa, h, 16 * ncols, scales.input_down, hq);
        for (int c = 0; c < ncols; ++c) { h_blocks[c][u] = hq[c]; }
    }
}

void down_rows(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales,
               const canon::A4Block* const* h_blocks, int ncols, int rg_begin, int rg_end,
               std::uint16_t* const* y, int prefetch_bytes) {
    std::int64_t s[16 * kMaxColumns]; // one row group at a time: no allocation on the miss path
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        rg16_row_sums(isa, record + kGateUpBytes, kDownBlocks, rg, rg + 1, h_blocks, ncols, s, prefetch_bytes);
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < ncols; ++c) {
                y[c][rg * 16 + r] = canon::a4_row_output(s[r * ncols + c], scales.alpha_down);
            }
        }
    }
}

void expert_forward(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales, int ncols,
                    const std::uint16_t* const* x, std::uint16_t* const* y, int prefetch_bytes) {
    if (ncols < 1 || ncols > kMaxColumns) { throw std::invalid_argument("offloaded_moe: ncols must be in [1, 8]"); }
    std::vector<canon::A4Block> xg(static_cast<std::size_t>(ncols) * kGateUpBlocks);
    std::vector<canon::A4Block> xu;
    std::vector<canon::A4Block> hb(static_cast<std::size_t>(ncols) * kHBlocks);
    const canon::A4Block* g_ptr[kMaxColumns];
    const canon::A4Block* u_ptr[kMaxColumns];
    canon::A4Block* h_ptr[kMaxColumns];
    const canon::A4Block* h_cptr[kMaxColumns];
    const bool shared = scales.input_gate == scales.input_up;
    if (!shared) { xu.resize(xg.size()); }
    for (int c = 0; c < ncols; ++c) {
        quantize_a4(isa, x[c], kHidden, scales.input_gate, &xg[static_cast<std::size_t>(c) * kGateUpBlocks]);
        g_ptr[c] = &xg[static_cast<std::size_t>(c) * kGateUpBlocks];
        if (!shared) {
            quantize_a4(isa, x[c], kHidden, scales.input_up, &xu[static_cast<std::size_t>(c) * kGateUpBlocks]);
            u_ptr[c] = &xu[static_cast<std::size_t>(c) * kGateUpBlocks];
        }
        h_ptr[c]  = &hb[static_cast<std::size_t>(c) * kHBlocks];
        h_cptr[c] = h_ptr[c];
    }
    gate_up_units(isa, record, scales, g_ptr, shared ? g_ptr : u_ptr, ncols, 0, kHBlocks, h_ptr, prefetch_bytes);
    down_rows(isa, record, scales, h_cptr, ncols, 0, kDownRowGroups, y, prefetch_bytes);
}

} // namespace infernix::ops::offloaded_moe
