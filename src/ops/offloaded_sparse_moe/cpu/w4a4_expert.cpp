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
#    define NINFER_MOE_X86 1
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#    define NINFER_TARGET(x)
#else
#    define NINFER_TARGET(x) __attribute__((target(x)))
#endif

namespace ninfer::ops::offloaded_moe {
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

#if defined(NINFER_MOE_X86)

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

NINFER_TARGET("avx512f,avx512bw,avx512vnni")
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
NINFER_TARGET("avx512f,avx512bw,avx512vnni")
void row_sums_avx512(const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end,
                     const A4Block* const* acts, std::int64_t* out) {
    const __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kBiasedCodes.data()));
    const __m512i lut    = _mm512_broadcast_i32x4(lut128);
    const __m512i low4   = _mm512_set1_epi8(0x0F);
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        __m512i even[N], odd[N];
        for (int c = 0; c < N; ++c) { even[c] = odd[c] = _mm512_setzero_si512(); }
        for (int b = 0; b < blocks; ++b) {
            const std::uint8_t* unit = matrix + (static_cast<std::size_t>(rg) * blocks + b) * kUnitBytes;
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

NINFER_TARGET("avx2")
inline __m256i scales8_avx2(const std::uint8_t* s) {
    const __m256i w    = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(s)));
    const __m256i m    = _mm256_and_si256(w, _mm256_set1_epi32(7));
    const __m256i e    = _mm256_and_si256(_mm256_srli_epi32(w, 3), _mm256_set1_epi32(15));
    const __m256i norm = _mm256_sllv_epi32(_mm256_or_si256(m, _mm256_set1_epi32(8)),
                                           _mm256_sub_epi32(e, _mm256_set1_epi32(1)));
    const __m256i sub  = _mm256_cmpeq_epi32(e, _mm256_setzero_si256());
    return _mm256_blendv_epi8(norm, m, sub);
}

NINFER_TARGET("avx2")
inline __m256i dot4_avx2(__m256i acc, __m256i w, __m256i a) {
    const __m256i pairs = _mm256_maddubs_epi16(w, a); // |pair| <= 2 * 24 * 12: no saturation
    return _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
}

NINFER_TARGET("avx2,avxvnni")
inline __m256i dot4_avxvnni(__m256i acc, __m256i w, __m256i a) {
    return _mm256_dpbusd_avx_epi32(acc, w, a);
}

// One body, two target attributes: AVX-VNNI must not leak into the plain AVX2 variant.
#define NINFER_AVX2_ROW_SUMS(NAME, DOT) \
void NAME(const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end, \
                   const A4Block* const* acts, std::int64_t* out) { \
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
NINFER_TARGET("avx2")
NINFER_AVX2_ROW_SUMS(row_sums_avx2, dot4_avx2)

template <int N>
NINFER_TARGET("avx2,avxvnni")
NINFER_AVX2_ROW_SUMS(row_sums_avxvnni, dot4_avxvnni)

#undef NINFER_AVX2_ROW_SUMS

#define NINFER_MOE_DISPATCH(CALL)                                                                   \
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
    f.avx2    = ymm && (b & (1U << 5)) != 0;
    f.avxvnni = f.avx2 && (a & (1U << 4)) != 0;
    f.avx512  = zmm && (b & (1U << 16)) != 0 && (b & (1U << 30)) != 0 && (c & (1U << 11)) != 0;
    return f;
}

const CpuFeatures& features() {
    static const CpuFeatures f = detect_features();
    return f;
}

#endif // NINFER_MOE_X86

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
#if defined(NINFER_MOE_X86)
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
                   const canon::A4Block* const* acts, int ncols, std::int64_t* out) {
    if (ncols < 1 || ncols > kMaxColumns) { throw std::invalid_argument("offloaded_moe: ncols must be in [1, 8]"); }
    if (!cpu_isa_supported(isa)) { throw std::invalid_argument("offloaded_moe: unsupported CPU ISA"); }
    switch (isa) {
    case CpuIsa::kScalar: row_sums_scalar(matrix, blocks, rg_begin, rg_end, acts, ncols, out); return;
#if defined(NINFER_MOE_X86)
    case CpuIsa::kAvx512Vnni:
#    define NINFER_CALL(n) row_sums_avx512<n>(matrix, blocks, rg_begin, rg_end, acts, out)
        NINFER_MOE_DISPATCH(NINFER_CALL)
#    undef NINFER_CALL
        return;
    case CpuIsa::kAvxVnni:
#    define NINFER_CALL(n) row_sums_avxvnni<n>(matrix, blocks, rg_begin, rg_end, acts, out)
        NINFER_MOE_DISPATCH(NINFER_CALL)
#    undef NINFER_CALL
        return;
    case CpuIsa::kAvx2:
#    define NINFER_CALL(n) row_sums_avx2<n>(matrix, blocks, rg_begin, rg_end, acts, out)
        NINFER_MOE_DISPATCH(NINFER_CALL)
#    undef NINFER_CALL
        return;
#else
    default: break;
#endif
    }
    throw std::invalid_argument("offloaded_moe: unsupported CPU ISA");
}

void quantize_a4(const std::uint16_t* v, int n, float input_scale, canon::A4Block* out) {
    for (int b = 0; b < n / 16; ++b) { out[b] = canon::quantize_a4_block(v + 16 * b, input_scale); }
}

void gate_up_units(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales,
                   const canon::A4Block* const* x_gate, const canon::A4Block* const* x_up,
                   int ncols, int unit_begin, int unit_end, canon::A4Block* const* h_blocks) {
    // No allocation on the miss path: one unit's sums fit on the stack.
    std::int64_t s_gate[32 * kMaxColumns];
    std::int64_t s_up[32 * kMaxColumns];
    for (int u = unit_begin; u < unit_end; ++u) {
        // Unit u is row groups 2u, 2u+1: rows 32u..32u+31, gate_i at even rows, up_i at odd rows.
        rg16_row_sums(isa, record, kGateUpBlocks, 2 * u, 2 * u + 2, x_gate, ncols, s_gate);
        const std::int64_t* up_sums = s_gate;
        if (x_up != x_gate) {
            rg16_row_sums(isa, record, kGateUpBlocks, 2 * u, 2 * u + 2, x_up, ncols, s_up);
            up_sums = s_up;
        }
        for (int c = 0; c < ncols; ++c) {
            std::uint16_t h[16];
            for (int i = 0; i < 16; ++i) {
                const std::uint16_t g = canon::a4_row_output(s_gate[static_cast<std::size_t>(2 * i) * ncols + c], scales.alpha_gate);
                const std::uint16_t p = canon::a4_row_output(up_sums[static_cast<std::size_t>(2 * i + 1) * ncols + c], scales.alpha_up);
                h[i] = canon::swiglu_bf16(g, p);
            }
            h_blocks[c][u] = canon::quantize_a4_block(h, scales.input_down);
        }
    }
}

void down_rows(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales,
               const canon::A4Block* const* h_blocks, int ncols, int rg_begin, int rg_end,
               std::uint16_t* const* y) {
    std::int64_t s[16 * kMaxColumns]; // one row group at a time: no allocation on the miss path
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        rg16_row_sums(isa, record + kGateUpBytes, kDownBlocks, rg, rg + 1, h_blocks, ncols, s);
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < ncols; ++c) {
                y[c][rg * 16 + r] = canon::a4_row_output(s[r * ncols + c], scales.alpha_down);
            }
        }
    }
}

void expert_forward(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales, int ncols,
                    const std::uint16_t* const* x, std::uint16_t* const* y) {
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
        quantize_a4(x[c], kHidden, scales.input_gate, &xg[static_cast<std::size_t>(c) * kGateUpBlocks]);
        g_ptr[c] = &xg[static_cast<std::size_t>(c) * kGateUpBlocks];
        if (!shared) {
            quantize_a4(x[c], kHidden, scales.input_up, &xu[static_cast<std::size_t>(c) * kGateUpBlocks]);
            u_ptr[c] = &xu[static_cast<std::size_t>(c) * kGateUpBlocks];
        }
        h_ptr[c]  = &hb[static_cast<std::size_t>(c) * kHBlocks];
        h_cptr[c] = h_ptr[c];
    }
    gate_up_units(isa, record, scales, g_ptr, shared ? g_ptr : u_ptr, ncols, 0, kHBlocks, h_ptr);
    down_rows(isa, record, scales, h_cptr, ncols, 0, kDownRowGroups, y);
}

} // namespace ninfer::ops::offloaded_moe
