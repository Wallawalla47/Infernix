#include "ops/offloaded_sparse_moe/cpu/w4a8_expert.h"

#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_a8.h"
#include "ops/offloaded_sparse_moe/nvfp4_a8.h"

#include <algorithm>
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

namespace a8 = ::infernix::ggml::a8;

// A row's partials at most (gate/up rows; down rows take nvfp4_a8::kDownPartials).
constexpr int kMaxPartials = nvfp4_a8::kGateUpPartials;
static_assert(nvfp4_a8::kDownPartials <= kMaxPartials);

// The vector variants' weights are c2 + kBias (unsigned bytes), so a block's dot with codes q is
// P_b + kBias * sum(q); CpuW4A8Slice::bias holds -kBias * sum(q).
constexpr int kBias = 12;

// Code nibble of row r (0..15) at k (0..15) inside one 144-byte unit (design §6.2).
inline unsigned unit_code(const std::uint8_t* unit, int r, int k) {
    const std::uint8_t byte = unit[32 * (k / 4) + 4 * (r % 8) + k % 4];
    return r < 8 ? (byte & 15U) : (byte >> 4);
}

// nvfp4_a8.h's butterfly over P partials: p_i = p_i + p_(i + o), o = P / 2 .. 1 (i < o); the sum in p_0.
inline float butterfly(float* p, int partials) {
    for (int o = partials / 2; o >= 1; o /= 2) {
        for (int i = 0; i < o; ++i) { p[i] = canon::add_rn(p[i], p[i + o]); }
    }
    return p[0];
}

// The reference form: each block's P_b from the codes themselves (no correction), its term
// (nvfp4_a8::add_block_term) into partial s % P in ascending s, block 2s then 2s + 1, from +0, then the butterfly.
void rows_scalar(const std::uint8_t* matrix, int blocks, int partials, int rg_begin, int rg_end,
                 const CpuW4A8Slice* const* acts, int ncols, float* out) {
    const int slices = blocks / 2;
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        for (int c = 0; c < ncols; ++c) {
            float part[16][kMaxPartials] = {}; // [row][partial]
            for (int s = 0; s < slices; ++s) {
                const CpuW4A8Slice& a = acts[c][s];
                for (int half = 0; half < 2; ++half) {
                    const std::uint8_t* unit =
                        matrix + (static_cast<std::size_t>(rg) * blocks + 2 * s + half) * kUnitBytes;
                    for (int r = 0; r < 16; ++r) {
                        std::int32_t p = 0; // |P_b| <= 12 * 16 * 127 < 2^15
                        for (int k = 0; k < 16; ++k) { p += canon::e2m1_x2(unit_code(unit, r, k)) * a.q[16 * half + k]; }
                        float& acc = part[r][s % partials];
                        acc        = nvfp4_a8::add_block_term(acc, p, a.d, unit[128 + r]);
                    }
                }
            }
            for (int r = 0; r < 16; ++r) {
                out[(static_cast<std::size_t>(rg - rg_begin) * 16 + r) * ncols + c] = butterfly(part[r], partials);
            }
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

// u8 table: 2 * e2m1(code) + kBias, so a weight is unsigned for vpdpbusd / vpmaddubsw.
constexpr std::array<std::int8_t, 16> kBiasedCodes = {12, 13, 14, 15, 16, 18, 20, 24,
                                                      12, 11, 10, 9,  8,  6,  4,  0};

// Codes 4q..4q+3 of a block (the activation word every row lane multiplies).
inline std::int32_t code_word(const std::int8_t* codes, int q) {
    std::int32_t w;
    std::memcpy(&w, codes + 4 * q, 4);
    return w;
}

// The vector variants. A unit's 16 rows are lanes (rows 0-7: the code bytes' low nibbles, rows 8-15: their high
// nibbles), so a block's dot with one column is four byte dot products per row vector, started from the block's
// correction: P_b exactly, in int32 (|P_b + correction| < 2^16). Its term is lane-wise fl32(P_b * d) (cvt is exact,
// |P_b| < 2^15) and fma(., S^, partial), S^ the rows' scale words (exact below 2^18), into the row vector of the
// block's partial, which acc keeps for the row group (L1): stored after each term, read by the partial's next one,
// and +0 for the partial's first term (slice s < P, block 2s). The butterfly adds the partials' row vectors in
// nvfp4_a8.h's pairs. Every lane thus runs the reference's IEEE operations in its order. A unit's codes and scales
// are decoded once for all of a call's columns.

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

// One zmm holds a unit's 16 rows (the code bytes broadcast to both halves, the upper half's high nibbles shifted
// down), so a block's dot with one column is four vpdpbusd with broadcast code words.
template <int N>
INFERNIX_TARGET("avx512f,avx512bw,avx512vnni")
void rows_avx512(const std::uint8_t* matrix, int blocks, int partials, int rg_begin, int rg_end,
                 const CpuW4A8Slice* const* acts, float* out, int prefetch_bytes) {
    const __m512i lut  = _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kBiasedCodes.data())));
    const __m512i low4 = _mm512_set1_epi8(0x0F);
    const int last     = partials - 1;
    alignas(64) float acc[kMaxPartials][N][16]; // [partial][column][row]
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        const std::uint8_t* units = matrix + static_cast<std::size_t>(rg) * blocks * kUnitBytes;
        for (int b = 0; b < blocks; ++b) {
            const std::uint8_t* unit = units + static_cast<std::size_t>(b) * kUnitBytes;
            prefetch_unit(unit, prefetch_bytes);
            const int s = b >> 1, half = b & 1;
            const bool fresh = half == 0 && s < partials; // the partial's first term: added to +0
            __m512i w[4];
            for (int q = 0; q < 4; ++q) {
                __m512i v = _mm512_broadcast_i64x4(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(unit + 32 * q)));
                v         = _mm512_mask_srli_epi16(v, 0xFFFF0000U, v, 4); // upper half: high nibbles
                w[q]      = _mm512_shuffle_epi8(lut, _mm512_and_si512(v, low4));
            }
            const __m512 sw   = _mm512_cvtepi32_ps(scales16_avx512(unit + 128));
            float(*part)[16]  = acc[s & last];
            for (int c = 0; c < N; ++c) {
                const CpuW4A8Slice& a    = acts[c][s];
                const std::int8_t* codes = a.q + 16 * half;
                __m512i p                = _mm512_set1_epi32(a.bias[half]);
                for (int q = 0; q < 4; ++q) { p = _mm512_dpbusd_epi32(p, w[q], _mm512_set1_epi32(code_word(codes, q))); }
                const __m512 m    = _mm512_mul_ps(_mm512_cvtepi32_ps(p), _mm512_set1_ps(a.d));
                const __m512 base = fresh ? _mm512_setzero_ps() : _mm512_load_ps(part[c]);
                _mm512_store_ps(part[c], _mm512_fmadd_ps(m, sw, base));
            }
        }
        for (int o = partials / 2; o >= 1; o /= 2) {
            for (int i = 0; i < o; ++i) {
                for (int c = 0; c < N; ++c) {
                    _mm512_store_ps(acc[i][c], _mm512_add_ps(_mm512_load_ps(acc[i][c]), _mm512_load_ps(acc[i + o][c])));
                }
            }
        }
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < N; ++c) { out[(static_cast<std::size_t>(rg - rg_begin) * 16 + r) * N + c] = acc[0][c][r]; }
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
    const __m256i pairs = _mm256_maddubs_epi16(u, s); // |pair| <= 2 * 24 * 127: no saturation
    return _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
}

INFERNIX_TARGET("avx2,avxvnni")
inline __m256i dot4_avxvnni(__m256i acc, __m256i u, __m256i s) {
    return _mm256_dpbusd_avx_epi32(acc, u, s);
}

// One body, two target attributes: AVX-VNNI must not leak into the plain AVX2 variant. Half h holds rows
// 8h..8h+7 (low nibbles for h = 0, high nibbles for h = 1), each an 8-float half of a partial's row vector.
#define INFERNIX_AVX2_ROWS_A8(NAME, DOT) \
void NAME(const std::uint8_t* matrix, int blocks, int partials, int rg_begin, int rg_end, \
          const CpuW4A8Slice* const* acts, float* out, int prefetch_bytes) { \
    const __m256i lut  = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kBiasedCodes.data()))); \
    const __m256i low4 = _mm256_set1_epi8(0x0F); \
    const int last     = partials - 1; \
    alignas(64) float acc[kMaxPartials][N][16]; /* [partial][column][row] */ \
    for (int rg = rg_begin; rg < rg_end; ++rg) { \
        const std::uint8_t* units = matrix + static_cast<std::size_t>(rg) * blocks * kUnitBytes; \
        for (int b = 0; b < blocks; ++b) { \
            const std::uint8_t* unit = units + static_cast<std::size_t>(b) * kUnitBytes; \
            prefetch_unit(unit, prefetch_bytes); \
            const int s = b >> 1, half = b & 1; \
            const bool fresh = half == 0 && s < partials; /* the partial's first term: added to +0 */ \
            __m256i w[2][4]; \
            for (int q = 0; q < 4; ++q) { \
                const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(unit + 32 * q)); \
                w[0][q]         = _mm256_shuffle_epi8(lut, _mm256_and_si256(v, low4)); \
                w[1][q]         = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v, 4), low4)); \
            } \
            const __m256 sw[2] = {_mm256_cvtepi32_ps(scales8_avx2(unit + 128)), \
                                  _mm256_cvtepi32_ps(scales8_avx2(unit + 136))}; \
            float(*part)[16] = acc[s & last]; \
            for (int c = 0; c < N; ++c) { \
                const CpuW4A8Slice& a    = acts[c][s]; \
                const std::int8_t* codes = a.q + 16 * half; \
                __m256i p[2]; \
                p[0] = p[1] = _mm256_set1_epi32(a.bias[half]); \
                for (int q = 0; q < 4; ++q) { \
                    const __m256i x = _mm256_set1_epi32(code_word(codes, q)); \
                    p[0]            = DOT(p[0], w[0][q], x); \
                    p[1]            = DOT(p[1], w[1][q], x); \
                } \
                const __m256 d = _mm256_set1_ps(a.d); \
                for (int h = 0; h < 2; ++h) { \
                    float* at         = part[c] + 8 * h; \
                    const __m256 base = fresh ? _mm256_setzero_ps() : _mm256_load_ps(at); \
                    _mm256_store_ps(at, _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(p[h]), d), sw[h], base)); \
                } \
            } \
        } \
        for (int o = partials / 2; o >= 1; o /= 2) { \
            for (int i = 0; i < o; ++i) { \
                for (int c = 0; c < N; ++c) { \
                    for (int h = 0; h < 2; ++h) { \
                        float* at = acc[i][c] + 8 * h; \
                        _mm256_store_ps(at, _mm256_add_ps(_mm256_load_ps(at), _mm256_load_ps(acc[i + o][c] + 8 * h))); \
                    } \
                } \
            } \
        } \
        for (int r = 0; r < 16; ++r) { \
            for (int c = 0; c < N; ++c) { out[(static_cast<std::size_t>(rg - rg_begin) * 16 + r) * N + c] = acc[0][c][r]; } \
        } \
    } \
}

template <int N>
INFERNIX_TARGET("avx2,fma")
INFERNIX_AVX2_ROWS_A8(rows_avx2, dot4_avx2)

template <int N>
INFERNIX_TARGET("avx2,fma,avxvnni")
INFERNIX_AVX2_ROWS_A8(rows_avxvnni, dot4_avxvnni)

#undef INFERNIX_AVX2_ROWS_A8

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

void encode_a8_slices(const std::uint16_t* v, int slice_begin, int slice_end, CpuW4A8Slice* out) {
    for (int s = slice_begin; s < slice_end; ++s) {
        const std::uint16_t* e = v + static_cast<std::size_t>(s) * 32;
        unsigned key           = 0;
        for (int j = 0; j < 32; ++j) { key = std::max(key, static_cast<unsigned>(e[j] & 0x7FFFU)); }
        const a8::SliceScale scale = a8::slice_scale(key);
        CpuW4A8Slice& o            = out[s];
        std::int32_t sum[2]        = {};
        for (int j = 0; j < 32; ++j) {
            const int q = a8::code(e[j], scale.inv);
            o.q[j]      = static_cast<std::int8_t>(q);
            sum[j / 16] += q;
        }
        o.bias[0] = -kBias * sum[0];
        o.bias[1] = -kBias * sum[1];
        o.d       = scale.d;
    }
}

void rg16_rows_a8(CpuIsa isa, const std::uint8_t* matrix, int blocks, int partials, int rg_begin, int rg_end,
                  const CpuW4A8Slice* const* acts, int ncols, float* out, int prefetch_bytes) {
    if (ncols < 1 || ncols > kMaxColumns) { throw std::invalid_argument("offloaded_moe: ncols must be in [1, 8]"); }
    if (!cpu_isa_supported(isa)) { throw std::invalid_argument("offloaded_moe: unsupported CPU ISA"); }
    if (blocks < 2 || blocks % 2 != 0 || partials < 1 || partials > kMaxPartials || (partials & (partials - 1)) != 0 ||
        blocks / 2 < partials) {
        throw std::invalid_argument(
            "offloaded_moe: 8-bit rows take whole slices and 1-16 partials (a power of two) of at least one slice each");
    }
    switch (isa) {
    case CpuIsa::kScalar: rows_scalar(matrix, blocks, partials, rg_begin, rg_end, acts, ncols, out); return;
#if defined(INFERNIX_MOE_X86)
    case CpuIsa::kAvx512Vnni:
#    define INFERNIX_CALL(n) rows_avx512<n>(matrix, blocks, partials, rg_begin, rg_end, acts, out, prefetch_bytes)
        INFERNIX_MOE_DISPATCH(INFERNIX_CALL)
#    undef INFERNIX_CALL
        return;
    case CpuIsa::kAvxVnni:
#    define INFERNIX_CALL(n) rows_avxvnni<n>(matrix, blocks, partials, rg_begin, rg_end, acts, out, prefetch_bytes)
        INFERNIX_MOE_DISPATCH(INFERNIX_CALL)
#    undef INFERNIX_CALL
        return;
    case CpuIsa::kAvx2:
#    define INFERNIX_CALL(n) rows_avx2<n>(matrix, blocks, partials, rg_begin, rg_end, acts, out, prefetch_bytes)
        INFERNIX_MOE_DISPATCH(INFERNIX_CALL)
#    undef INFERNIX_CALL
        return;
#else
    default: break;
#endif
    }
    throw std::invalid_argument("offloaded_moe: unsupported CPU ISA");
}

void gate_up_units_a8(CpuIsa isa, const ExpertShape& shape, const std::uint8_t* record, const ExpertScales& scales,
                      const CpuW4A8Slice* const* x, int ncols, int unit_begin, int unit_end, std::uint16_t* const* h,
                      int prefetch_bytes) {
    float p0[32 * kMaxColumns]; // one unit's rows: no allocation on the miss path
    for (int u = unit_begin; u < unit_end; ++u) {
        // Unit u is row groups 2u, 2u+1: rows 32u..32u+31, gate_i at even rows, up_i at odd rows.
        rg16_rows_a8(isa, record, shape.gate_up_blocks(), nvfp4_a8::kGateUpPartials, 2 * u, 2 * u + 2, x, ncols, p0,
                     prefetch_bytes);
        for (int c = 0; c < ncols; ++c) {
            for (int i = 0; i < 16; ++i) {
                const std::uint16_t g = nvfp4_a8::row_output(p0[static_cast<std::size_t>(2 * i) * ncols + c], scales.alpha_gate);
                const std::uint16_t p = nvfp4_a8::row_output(p0[static_cast<std::size_t>(2 * i + 1) * ncols + c], scales.alpha_up);
                h[c][16 * u + i]      = canon::swiglu_bf16(g, p);
            }
        }
    }
}

void down_rows_a8(CpuIsa isa, const ExpertShape& shape, const std::uint8_t* record, const ExpertScales& scales,
                  const CpuW4A8Slice* const* h, int ncols, int rg_begin, int rg_end, std::uint16_t* const* y,
                  int prefetch_bytes) {
    float p0[16 * kMaxColumns]; // one row group at a time
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        rg16_rows_a8(isa, record + shape.gate_up_bytes(), shape.down_blocks(), nvfp4_a8::kDownPartials, rg, rg + 1, h,
                     ncols, p0, prefetch_bytes);
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < ncols; ++c) {
                y[c][rg * 16 + r] = nvfp4_a8::row_output(p0[r * ncols + c], scales.alpha_down);
            }
        }
    }
}

void expert_forward_a8(CpuIsa isa, const ExpertShape& shape, const std::uint8_t* record, const ExpertScales& scales,
                       int ncols, const std::uint16_t* const* x, std::uint16_t* const* y, int prefetch_bytes) {
    if (ncols < 1 || ncols > kMaxColumns) { throw std::invalid_argument("offloaded_moe: ncols must be in [1, 8]"); }
    if (!registered_shape(shape)) { throw std::invalid_argument("offloaded_moe: unregistered expert geometry"); }
    const int x_slices = shape.hidden / 32, h_slices = shape.intermediate / 32;
    std::vector<CpuW4A8Slice> xs(static_cast<std::size_t>(ncols) * x_slices);
    std::vector<CpuW4A8Slice> hs(static_cast<std::size_t>(ncols) * h_slices);
    std::vector<std::uint16_t> hv(static_cast<std::size_t>(ncols) * shape.intermediate);
    const CpuW4A8Slice* x_ptr[kMaxColumns];
    const CpuW4A8Slice* h_ptr[kMaxColumns];
    std::uint16_t* hv_ptr[kMaxColumns];
    for (int c = 0; c < ncols; ++c) {
        encode_a8_slices(x[c], 0, x_slices, &xs[static_cast<std::size_t>(c) * x_slices]);
        x_ptr[c]  = &xs[static_cast<std::size_t>(c) * x_slices];
        h_ptr[c]  = &hs[static_cast<std::size_t>(c) * h_slices];
        hv_ptr[c] = &hv[static_cast<std::size_t>(c) * shape.intermediate];
    }
    gate_up_units_a8(isa, shape, record, scales, x_ptr, ncols, 0, shape.h_blocks(), hv_ptr, prefetch_bytes);
    for (int c = 0; c < ncols; ++c) { encode_a8_slices(hv_ptr[c], 0, h_slices, &hs[static_cast<std::size_t>(c) * h_slices]); }
    down_rows_a8(isa, shape, record, scales, h_ptr, ncols, 0, shape.down_row_groups(), y, prefetch_bytes);
}

} // namespace infernix::ops::offloaded_moe
