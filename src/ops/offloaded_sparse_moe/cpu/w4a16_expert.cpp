#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#    include <immintrin.h>
#    define INFERNIX_MOE_X86 1
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#    define INFERNIX_TARGET(x)
#else
#    define INFERNIX_TARGET(x) __attribute__((target(x)))
#endif

namespace infernix::ops::offloaded_moe {
namespace {

using canon::A16Block;

// Code nibble of row r (0..15) at k (0..15) inside one 144-byte unit (design §6.2).
inline unsigned unit_code(const std::uint8_t* unit, int r, int k) {
    const std::uint8_t byte = unit[32 * (k / 4) + 4 * (r % 8) + k % 4];
    return r < 8 ? (byte & 15U) : (byte >> 4);
}

void row_sums_scalar(const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end,
                     const A16Block* const* acts, int ncols, std::int64_t* out) {
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        for (int c = 0; c < ncols; ++c) {
            std::int64_t s[16] = {};
            for (int b = 0; b < blocks; ++b) {
                const std::uint8_t* unit = matrix + (static_cast<std::size_t>(rg) * blocks + b) * kUnitBytes;
                const A16Block& a        = acts[c][b];
                for (int r = 0; r < 16; ++r) {
                    std::int32_t p = 0; // |P| < 2^31 (design §16.2.1)
                    for (int k = 0; k < 16; ++k) { p += canon::e2m1_x2(unit_code(unit, r, k)) * canon::a16_x(a, k); }
                    s[r] += static_cast<std::int64_t>(p) * canon::e4m3_scaled(unit[128 + r]);
                }
            }
            for (int r = 0; r < 16; ++r) { out[(static_cast<std::size_t>(rg - rg_begin) * 16 + r) * ncols + c] = s[r]; }
        }
    }
}

#if defined(INFERNIX_MOE_X86)

// Software prefetch `distance` bytes ahead along the unit stream (design §10.3, §14.2).
inline void prefetch_unit(const std::uint8_t* unit, int distance) {
    if (distance > 0) {
        const char* p = reinterpret_cast<const char*>(unit) + distance;
        _mm_prefetch(p, _MM_HINT_T0);
        _mm_prefetch(p + 64, _MM_HINT_T0);
        _mm_prefetch(p + 128, _MM_HINT_T0);
    }
}

// Code -> doubled E2M1 value as a signed byte (for unsigned activation bytes), and biased by +12 as
// an unsigned byte (for the signed high byte of X).
constexpr std::array<std::int8_t, 16> kSignedCodes = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
constexpr std::array<std::int8_t, 16> kBiasedCodes = {12, 13, 14, 15, 16, 18, 20, 24,
                                                      12, 11, 10, 9,  8,  6,  4,  0};
constexpr int kBias = 12;

template <class T>
inline std::int32_t quad_word(const T (&bytes)[16], int q) {
    std::int32_t w;
    std::memcpy(&w, &bytes[4 * q], 4);
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

// Lane r of the 16-lane vectors is row r of the unit (rows 0-7: low nibbles of the first 256 bits,
// rows 8-15: high nibbles moved into the second 256 bits). Per block and column: lo and mid bytes
// (unsigned) times signed codes, hi bytes (signed) times biased codes, P = lo + 256 mid + 65536 hi
// in wrapping int32 (exact: |P| < 2^31), then P * Sw into int64 per row.
template <int N>
INFERNIX_TARGET("avx512f,avx512bw,avx512vnni")
void row_sums_avx512(const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end, const A16Block* const* acts,
                     std::int64_t* out, int prefetch_bytes) {
    const __m512i lut_s = _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kSignedCodes.data())));
    const __m512i lut_b = _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kBiasedCodes.data())));
    const __m512i low4  = _mm512_set1_epi8(0x0F);
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        __m512i even[N], odd[N];
        for (int c = 0; c < N; ++c) { even[c] = odd[c] = _mm512_setzero_si512(); }
        for (int b = 0; b < blocks; ++b) {
            const std::uint8_t* unit = matrix + (static_cast<std::size_t>(rg) * blocks + b) * kUnitBytes;
            prefetch_unit(unit, prefetch_bytes);
            __m512i plo[N], pmid[N], phi[N];
            for (int c = 0; c < N; ++c) { plo[c] = pmid[c] = phi[c] = _mm512_setzero_si512(); }
            for (int q = 0; q < 4; ++q) {
                __m512i v = _mm512_broadcast_i64x4(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(unit + 32 * q)));
                v         = _mm512_mask_srli_epi16(v, 0xFFFF0000U, v, 4); // upper half: high nibbles
                v         = _mm512_and_si512(v, low4);
                const __m512i ws = _mm512_shuffle_epi8(lut_s, v);
                const __m512i wb = _mm512_shuffle_epi8(lut_b, v);
                for (int c = 0; c < N; ++c) {
                    const A16Block& a = acts[c][b];
                    plo[c]  = _mm512_dpbusd_epi32(plo[c], _mm512_set1_epi32(quad_word(a.lo, q)), ws);
                    pmid[c] = _mm512_dpbusd_epi32(pmid[c], _mm512_set1_epi32(quad_word(a.mid, q)), ws);
                    phi[c]  = _mm512_dpbusd_epi32(phi[c], wb, _mm512_set1_epi32(quad_word(a.hi, q)));
                }
            }
            const __m512i sw     = scales16_avx512(unit + 128);
            const __m512i sw_odd = _mm512_srli_epi64(sw, 32);
            for (int c = 0; c < N; ++c) {
                const __m512i hi = _mm512_sub_epi32(phi[c], _mm512_set1_epi32(kBias * acts[c][b].hi_sum));
                const __m512i p  = _mm512_add_epi32(_mm512_add_epi32(plo[c], _mm512_slli_epi32(pmid[c], 8)),
                                                    _mm512_slli_epi32(hi, 16));
                even[c] = _mm512_add_epi64(even[c], _mm512_mul_epi32(p, sw));
                odd[c]  = _mm512_add_epi64(odd[c], _mm512_mul_epi32(_mm512_srli_epi64(p, 32), sw_odd));
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

// acc + sum of four (unsigned byte of u) * (signed byte of s) per lane.
INFERNIX_TARGET("avx2")
inline __m256i dot4_avx2(__m256i acc, __m256i u, __m256i s) {
    const __m256i pairs = _mm256_maddubs_epi16(u, s); // |pair| <= 2 * 255 * 12 or 2 * 24 * 128: no saturation
    return _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
}

INFERNIX_TARGET("avx2,avxvnni")
inline __m256i dot4_avxvnni(__m256i acc, __m256i u, __m256i s) {
    return _mm256_dpbusd_avx_epi32(acc, u, s);
}

// One body, two target attributes: AVX-VNNI must not leak into the plain AVX2 variant. Half h holds
// rows 8h..8h+7 (low nibbles for h = 0, high nibbles for h = 1).
#define INFERNIX_AVX2_ROW_SUMS_A16(NAME, DOT) \
void NAME(const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end, \
          const A16Block* const* acts, std::int64_t* out, int prefetch_bytes) { \
    const __m256i lut_s = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kSignedCodes.data()))); \
    const __m256i lut_b = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kBiasedCodes.data()))); \
    const __m256i low4  = _mm256_set1_epi8(0x0F); \
    for (int rg = rg_begin; rg < rg_end; ++rg) { \
        __m256i even[2][N], odd[2][N]; \
        for (int h = 0; h < 2; ++h) { \
            for (int c = 0; c < N; ++c) { even[h][c] = odd[h][c] = _mm256_setzero_si256(); } \
        } \
        for (int b = 0; b < blocks; ++b) { \
            const std::uint8_t* unit = matrix + (static_cast<std::size_t>(rg) * blocks + b) * kUnitBytes; \
            prefetch_unit(unit, prefetch_bytes); \
            __m256i plo[2][N], pmid[2][N], phi[2][N]; \
            for (int h = 0; h < 2; ++h) { \
                for (int c = 0; c < N; ++c) { plo[h][c] = pmid[h][c] = phi[h][c] = _mm256_setzero_si256(); } \
            } \
            for (int q = 0; q < 4; ++q) { \
                const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(unit + 32 * q)); \
                const __m256i idx[2] = {_mm256_and_si256(v, low4), _mm256_and_si256(_mm256_srli_epi16(v, 4), low4)}; \
                for (int h = 0; h < 2; ++h) { \
                    const __m256i ws = _mm256_shuffle_epi8(lut_s, idx[h]); \
                    const __m256i wb = _mm256_shuffle_epi8(lut_b, idx[h]); \
                    for (int c = 0; c < N; ++c) { \
                        const A16Block& a = acts[c][b]; \
                        plo[h][c]  = DOT(plo[h][c], _mm256_set1_epi32(quad_word(a.lo, q)), ws); \
                        pmid[h][c] = DOT(pmid[h][c], _mm256_set1_epi32(quad_word(a.mid, q)), ws); \
                        phi[h][c]  = DOT(phi[h][c], wb, _mm256_set1_epi32(quad_word(a.hi, q))); \
                    } \
                } \
            } \
            for (int h = 0; h < 2; ++h) { \
                const __m256i sw     = scales8_avx2(unit + 128 + 8 * h); \
                const __m256i sw_odd = _mm256_srli_epi64(sw, 32); \
                for (int c = 0; c < N; ++c) { \
                    const __m256i hi = _mm256_sub_epi32(phi[h][c], _mm256_set1_epi32(kBias * acts[c][b].hi_sum)); \
                    const __m256i p  = _mm256_add_epi32(_mm256_add_epi32(plo[h][c], _mm256_slli_epi32(pmid[h][c], 8)), \
                                                        _mm256_slli_epi32(hi, 16)); \
                    even[h][c] = _mm256_add_epi64(even[h][c], _mm256_mul_epi32(p, sw)); \
                    odd[h][c]  = _mm256_add_epi64(odd[h][c], _mm256_mul_epi32(_mm256_srli_epi64(p, 32), sw_odd)); \
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
INFERNIX_AVX2_ROW_SUMS_A16(row_sums_avx2, dot4_avx2)

template <int N>
INFERNIX_TARGET("avx2,avxvnni")
INFERNIX_AVX2_ROW_SUMS_A16(row_sums_avxvnni, dot4_avxvnni)

#undef INFERNIX_AVX2_ROW_SUMS_A16

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

#endif // INFERNIX_MOE_X86

} // namespace

int encode_a16(const std::uint16_t* v, int n, canon::A16Block* out) {
    if (n <= 0 || n % 16 != 0) { throw std::invalid_argument("offloaded_moe: A16 columns hold whole blocks"); }
    const int emax = canon::a16_column_exponent(v, n);
    encode_a16_blocks(v, emax, 0, n / 16, out);
    return emax;
}

void encode_a16_blocks(const std::uint16_t* v, int emax, int block_begin, int block_end, canon::A16Block* out) {
    for (int b = block_begin; b < block_end; ++b) { out[b] = canon::a16_block(v + 16 * b, emax); }
}

void rg16_row_sums_a16(CpuIsa isa, const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end,
                       const canon::A16Block* const* acts, int ncols, std::int64_t* out, int prefetch_bytes) {
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

void gate_up_units_a16(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales,
                       const canon::A16Block* const* x, const int* x_exp, int ncols, int unit_begin, int unit_end,
                       std::uint16_t* const* h, int prefetch_bytes) {
    std::int64_t s[32 * kMaxColumns]; // one unit's sums: no allocation on the miss path
    for (int u = unit_begin; u < unit_end; ++u) {
        // Unit u is row groups 2u, 2u+1: rows 32u..32u+31, gate_i at even rows, up_i at odd rows.
        rg16_row_sums_a16(isa, record, kGateUpBlocks, 2 * u, 2 * u + 2, x, ncols, s, prefetch_bytes);
        for (int c = 0; c < ncols; ++c) {
            for (int i = 0; i < 16; ++i) {
                const std::uint16_t g = canon::a16_row_output(s[static_cast<std::size_t>(2 * i) * ncols + c], x_exp[c], scales.alpha_gate);
                const std::uint16_t p = canon::a16_row_output(s[static_cast<std::size_t>(2 * i + 1) * ncols + c], x_exp[c], scales.alpha_up);
                h[c][16 * u + i] = canon::swiglu_bf16(g, p);
            }
        }
    }
}

void down_rows_a16(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales,
                   const canon::A16Block* const* h, const int* h_exp, int ncols, int rg_begin, int rg_end,
                   std::uint16_t* const* y, int prefetch_bytes) {
    std::int64_t s[16 * kMaxColumns]; // one row group at a time
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        rg16_row_sums_a16(isa, record + kGateUpBytes, kDownBlocks, rg, rg + 1, h, ncols, s, prefetch_bytes);
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < ncols; ++c) {
                y[c][rg * 16 + r] = canon::a16_row_output(s[r * ncols + c], h_exp[c], scales.alpha_down);
            }
        }
    }
}

void expert_forward_a16(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales, int ncols,
                        const std::uint16_t* const* x, std::uint16_t* const* y, int prefetch_bytes) {
    if (ncols < 1 || ncols > kMaxColumns) { throw std::invalid_argument("offloaded_moe: ncols must be in [1, 8]"); }
    std::vector<canon::A16Block> xb(static_cast<std::size_t>(ncols) * kGateUpBlocks);
    std::vector<canon::A16Block> hb(static_cast<std::size_t>(ncols) * kHBlocks);
    std::vector<std::uint16_t> hv(static_cast<std::size_t>(ncols) * kIntermediate);
    const canon::A16Block* x_ptr[kMaxColumns];
    const canon::A16Block* h_ptr[kMaxColumns];
    std::uint16_t* hv_ptr[kMaxColumns];
    int x_exp[kMaxColumns], h_exp[kMaxColumns];
    for (int c = 0; c < ncols; ++c) {
        x_exp[c]  = encode_a16(x[c], kHidden, &xb[static_cast<std::size_t>(c) * kGateUpBlocks]);
        x_ptr[c]  = &xb[static_cast<std::size_t>(c) * kGateUpBlocks];
        hv_ptr[c] = &hv[static_cast<std::size_t>(c) * kIntermediate];
        h_ptr[c]  = &hb[static_cast<std::size_t>(c) * kHBlocks];
    }
    gate_up_units_a16(isa, record, scales, x_ptr, x_exp, ncols, 0, kHBlocks, hv_ptr, prefetch_bytes);
    for (int c = 0; c < ncols; ++c) {
        h_exp[c] = encode_a16(hv_ptr[c], kIntermediate, &hb[static_cast<std::size_t>(c) * kHBlocks]);
    }
    down_rows_a16(isa, record, scales, h_ptr, h_exp, ncols, 0, kDownRowGroups, y, prefetch_bytes);
}

} // namespace infernix::ops::offloaded_moe
