#include "ops/offloaded_sparse_moe/cpu/fp8_expert.h"

#include "ops/common/canonical_math.h"
#include "ops/fp8_block/fp8_exact.h"
#include "ops/offloaded_sparse_moe/cpu/x86_mad.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace infernix::ops::offloaded_moe {
namespace {

using fp8b128::kTile;

constexpr std::size_t kGateBytes   = static_cast<std::size_t>(kIntermediate) * kHidden;
constexpr std::size_t kScaleOffset = 3 * kGateBytes;
constexpr int kGateTiles           = (kIntermediate / kTile) * (kHidden / kTile); // per matrix

// Exact R of rows [row_begin, row_end) of a [rows, k] block-FP8 matrix (codes row-major; its tile
// scales [rows / 128][k / 128]) against `ncols` encoded columns: r[(row - row_begin) * ncols + c],
// with e_min[row - row_begin] each row's smallest tile exponent. The reference form.
void row_sums_scalar(const std::uint8_t* codes, const std::uint16_t* scales, int k, int row_begin, int row_end,
                     const CpuLimbSlice* const* columns, int ncols, exact::I128* r, int* e_min) {
    const int tiles = k / kTile;
    for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t* w      = codes + static_cast<std::size_t>(row) * k;
        const std::uint16_t* tile  = scales + static_cast<std::size_t>(row / kTile) * tiles;
        const int e                = fp8b128::min_exponent(tile, tiles);
        e_min[row - row_begin]     = e;
        for (int c = 0; c < ncols; ++c) {
            exact::I128 acc;
            for (int t = 0; t < tiles; ++t) {
                std::int64_t p = 0;
                for (int j = t * kTile; j < (t + 1) * kTile; ++j) {
                    const CpuLimbSlice& s = columns[c][j / 32];
                    const std::int64_t x  = static_cast<std::int64_t>(s.hi[j % 32]) * 2048 + s.lo[j % 32];
                    p += fp8b128::fp8_int(w[j]) * x;
                }
                fp8b128::add_tile(acc, p, tile[t], e);
            }
            r[static_cast<std::size_t>(row - row_begin) * ncols + c] = acc;
        }
    }
}

#if defined(INFERNIX_CPU_X86)

// ------------------------------------------------------------------------------------ AVX2
//
// A weight's integer w = e4m3fn(c) * 2^9 (fp8_int) is split by its exponent field e into
//
//   w = A + 2^8 * B,  A = (e <= 8 ? +-(lead(e) + m) * 2^max(e - 1, 0) : 0)  in [-1920, 1920],
//                     B = (e >= 9 ? +-(8 + m) * 2^(e - 9)              : 0)  in [-896, 896],
//
// (lead(e) = 8 for a normal code, 0 for a subnormal one), so exactly one of A and B is nonzero.
// Both are a signed 4-bit mantissa times a power of two at most 128, which two 16-entry vpshufb
// tables give per exponent and vpmaddubsw multiplies into int16: the decode costs a few byte
// operations per 32 codes. vpmaddubsw adds neighbouring bytes' products, so each table is applied
// to the even and the odd bytes separately; the column slices are split the same way
// (SplitLimbSlice), and every product is still a weight's with its own column element.

// A column slice's X = hi * 2^11 + lo (CpuLimbSlice), even elements then odd ones.
struct SplitLimbSlice {
    alignas(32) std::int16_t lo[2][16];
    std::int16_t hi[2][16];
};

INFERNIX_TARGET("avx2")
void split_slices(const CpuLimbSlice* columns, int slices, SplitLimbSlice* out) {
    // Per 128-bit lane: even words to the low 8 bytes, odd words to the high 8 bytes; then the
    // 64-bit quarters [e0-6, o1-7, e8-14, o9-15] become [e0-6, e8-14, o1-7, o9-15].
    const __m256i words = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
                                           0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
    for (int s = 0; s < slices; ++s) {
        const std::int16_t* src[2] = {columns[s].lo, columns[s].hi};
        std::int16_t (*dst[2])[16] = {out[s].lo, out[s].hi};
        for (int part = 0; part < 2; ++part) {
            const __m256i a  = _mm256_shuffle_epi8(_mm256_load_si256(reinterpret_cast<const __m256i*>(src[part])), words);
            const __m256i b  = _mm256_shuffle_epi8(_mm256_load_si256(reinterpret_cast<const __m256i*>(src[part] + 16)), words);
            const __m256i qa = _mm256_permute4x64_epi64(a, 0xD8);
            const __m256i qb = _mm256_permute4x64_epi64(b, 0xD8);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst[part][0]), _mm256_permute2x128_si256(qa, qb, 0x20));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst[part][1]), _mm256_permute2x128_si256(qa, qb, 0x31));
        }
    }
}

// A slice's 32 weights as A and B (above), int16: [0] its even elements, [1] its odd ones.
struct SliceWeights {
    __m256i a[2], b[2];
};

INFERNIX_TARGET("avx2")
inline SliceWeights slice_weights(const std::uint8_t* codes) {
    // Per exponent field e: the implicit leading 8, and A's and B's power of two (0: not that part).
    const __m256i lead = _mm256_setr_epi8(0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                                          0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8);
    const __m256i pow_a = _mm256_setr_epi8(1, 1, 2, 4, 8, 16, 32, 64, -128, 0, 0, 0, 0, 0, 0, 0,
                                           1, 1, 2, 4, 8, 16, 32, 64, -128, 0, 0, 0, 0, 0, 0, 0);
    const __m256i pow_b = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64,
                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64);
    const __m256i c    = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes));
    const __m256i e    = _mm256_and_si256(_mm256_srli_epi16(c, 3), _mm256_set1_epi8(15));
    const __m256i m    = _mm256_and_si256(c, _mm256_set1_epi8(7));
    // The signed mantissa: vpsignb negates where the code's sign bit is set (a zero code's
    // mantissa is zero either way).
    const __m256i mant = _mm256_sign_epi8(_mm256_or_si256(_mm256_shuffle_epi8(lead, e), m), c);
    const __m256i pa = _mm256_shuffle_epi8(pow_a, e), pb = _mm256_shuffle_epi8(pow_b, e);
    const __m256i even = _mm256_set1_epi16(0x00FF), odd = _mm256_set1_epi16(static_cast<short>(0xFF00));
    SliceWeights out;
    // vpmaddubsw: unsigned powers times signed mantissas; one byte of each pair is zero.
    out.a[0] = _mm256_maddubs_epi16(_mm256_and_si256(pa, even), mant);
    out.a[1] = _mm256_maddubs_epi16(_mm256_and_si256(pa, odd), mant);
    out.b[0] = _mm256_maddubs_epi16(_mm256_and_si256(pb, even), mant);
    out.b[1] = _mm256_maddubs_epi16(_mm256_and_si256(pb, odd), mant);
    return out;
}

// Rows [row_begin, row_end) against NC columns. Per tile and column eight int32 lane streams
// (A * lo, A * hi, B * lo, B * hi over the even elements, then the odd ones; each lane sums 8
// products of at most 1920 * 2047 per tile, below 2^25) give the tile's exact
// P = 2^19 S_Bhi + 2^11 S_Ahi + 2^8 S_Blo + S_Alo (int64), which enters R as fp8_exact.h's. From
// three columns on, the even and odd halves share their four streams (16 products per lane per
// tile, below 2^26): the accumulators fit the 16 registers instead of spilling 32.
#define INFERNIX_FP8_ROW_SUMS(Name, Target, Mad)                                                                       \
    template <int NC>                                                                                                  \
    INFERNIX_TARGET(Target)                                                                                            \
    void Name##_n(const std::uint8_t* codes, const std::uint16_t* scales, int k, int row_begin, int row_end,           \
                  const SplitLimbSlice* const* columns, int c0, int ncols, exact::I128* r, int* e_min) {               \
        constexpr int kStreams = NC >= 3 ? 4 : 8;                                                                      \
        const int tiles = k / kTile;                                                                                   \
        for (int row = row_begin; row < row_end; ++row) {                                                              \
            const std::uint8_t* w     = codes + static_cast<std::size_t>(row) * k;                                     \
            const std::uint16_t* tile = scales + static_cast<std::size_t>(row / kTile) * tiles;                        \
            const int e               = fp8b128::min_exponent(tile, tiles);                                            \
            e_min[row - row_begin]    = e;                                                                             \
            exact::I128 out[NC];                                                                                       \
            for (int t = 0; t < tiles; ++t) {                                                                          \
                __m256i v[NC][8];                                                                                      \
                for (int c = 0; c < NC; ++c) {                                                                         \
                    for (int i = 0; i < 8; ++i) { v[c][i] = _mm256_setzero_si256(); }                                  \
                }                                                                                                      \
                for (int s = 4 * t; s < 4 * t + 4; ++s) {                                                              \
                    const SliceWeights sw = slice_weights(w + 32 * s);                                                 \
                    for (int c = 0; c < NC; ++c) {                                                                     \
                        const SplitLimbSlice& l = columns[c0 + c][s];                                                  \
                        for (int half = 0; half < 2; ++half) {                                                         \
                            const int at     = kStreams == 8 ? 4 * half : 0;                                           \
                            const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(l.lo[half]));        \
                            const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(l.hi[half]));        \
                            v[c][at + 0] = Mad::mad(v[c][at + 0], sw.a[half], lo);                                     \
                            v[c][at + 1] = Mad::mad(v[c][at + 1], sw.a[half], hi);                                     \
                            v[c][at + 2] = Mad::mad(v[c][at + 2], sw.b[half], lo);                                     \
                            v[c][at + 3] = Mad::mad(v[c][at + 3], sw.b[half], hi);                                     \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
                for (int c = 0; c < NC; ++c) {                                                                         \
                    alignas(32) std::int32_t sum[8];                                                                   \
                    _mm256_store_si256(reinterpret_cast<__m256i*>(sum), x86::hsum8(v[c]));                             \
                    const std::int64_t p = ((static_cast<std::int64_t>(sum[3]) + sum[7]) << 19) +                      \
                                           ((static_cast<std::int64_t>(sum[1]) + sum[5]) << 11) +                      \
                                           ((static_cast<std::int64_t>(sum[2]) + sum[6]) << 8) +                       \
                                           (static_cast<std::int64_t>(sum[0]) + sum[4]);                               \
                    fp8b128::add_tile(out[c], p, tile[t], e);                                                          \
                }                                                                                                      \
            }                                                                                                          \
            for (int c = 0; c < NC; ++c) { r[static_cast<std::size_t>(row - row_begin) * ncols + c0 + c] = out[c]; }  \
        }                                                                                                              \
    }                                                                                                                  \
    void Name(const std::uint8_t* codes, const std::uint16_t* scales, int k, int row_begin, int row_end,               \
              const SplitLimbSlice* const* columns, int ncols, exact::I128* r, int* e_min) {                           \
        for (int c0 = 0; c0 < ncols; c0 += 4) {                                                                        \
            switch (std::min(4, ncols - c0)) {                                                                         \
            case 1: Name##_n<1>(codes, scales, k, row_begin, row_end, columns, c0, ncols, r, e_min); break;            \
            case 2: Name##_n<2>(codes, scales, k, row_begin, row_end, columns, c0, ncols, r, e_min); break;            \
            case 3: Name##_n<3>(codes, scales, k, row_begin, row_end, columns, c0, ncols, r, e_min); break;            \
            default: Name##_n<4>(codes, scales, k, row_begin, row_end, columns, c0, ncols, r, e_min); break;           \
            }                                                                                                          \
        }                                                                                                              \
    }

INFERNIX_FP8_ROW_SUMS(row_sums_avx2, "avx2", x86::Avx2Mad)
INFERNIX_FP8_ROW_SUMS(row_sums_avxvnni, "avx2,avxvnni", x86::AvxVnniMad)
#undef INFERNIX_FP8_ROW_SUMS

// ------------------------------------------------------------------------------------ AVX-512
//
// The same arithmetic on 512-bit registers: one decode takes two slices (64 codes), their even and
// odd halves side by side in each register (PairSlice: slice 2p's half, then slice 2p + 1's), and
// each column keeps four streams that both halves share (a lane sums 8 products per tile, below
// 2^25), reduced once per tile.

struct PairSlice {
    alignas(64) std::int16_t lo[2][32];
    std::int16_t hi[2][32];
};

void pair_slices(const SplitLimbSlice* s, int slices, PairSlice* out) {
    for (int p = 0; p < slices / 2; ++p) {
        for (int half = 0; half < 2; ++half) {
            std::memcpy(out[p].lo[half], s[2 * p].lo[half], sizeof(s[0].lo[0]));
            std::memcpy(out[p].lo[half] + 16, s[2 * p + 1].lo[half], sizeof(s[0].lo[0]));
            std::memcpy(out[p].hi[half], s[2 * p].hi[half], sizeof(s[0].hi[0]));
            std::memcpy(out[p].hi[half] + 16, s[2 * p + 1].hi[half], sizeof(s[0].hi[0]));
        }
    }
}

// Two slices' 64 weights as A and B (slice_weights' split), int16: [0] even elements, [1] odd ones.
INFERNIX_TARGET("avx512f,avx512bw")
inline void pair_weights(const std::uint8_t* codes, __m512i (&a)[2], __m512i (&b)[2]) {
    const __m512i lead  = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8));
    const __m512i pow_a = _mm512_broadcast_i32x4(_mm_setr_epi8(1, 1, 2, 4, 8, 16, 32, 64, -128, 0, 0, 0, 0, 0, 0, 0));
    const __m512i pow_b = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64));
    const __m512i c = _mm512_loadu_si512(codes);
    const __m512i e = _mm512_and_si512(_mm512_srli_epi16(c, 3), _mm512_set1_epi8(15));
    const __m512i m = _mm512_and_si512(c, _mm512_set1_epi8(7));
    const __m512i v = _mm512_or_si512(_mm512_shuffle_epi8(lead, e), m);
    // vpsignb has no 512-bit form: the mantissa is negated where the code's sign bit is set (a zero
    // code's is zero either way).
    const __m512i mant = _mm512_mask_sub_epi8(v, _mm512_movepi8_mask(c), _mm512_setzero_si512(), v);
    const __m512i pa = _mm512_shuffle_epi8(pow_a, e), pb = _mm512_shuffle_epi8(pow_b, e);
    const __m512i even = _mm512_set1_epi16(0x00FF), odd = _mm512_set1_epi16(static_cast<short>(0xFF00));
    a[0] = _mm512_maddubs_epi16(_mm512_and_si512(pa, even), mant);
    a[1] = _mm512_maddubs_epi16(_mm512_and_si512(pa, odd), mant);
    b[0] = _mm512_maddubs_epi16(_mm512_and_si512(pb, even), mant);
    b[1] = _mm512_maddubs_epi16(_mm512_and_si512(pb, odd), mant);
}

template <int NC>
INFERNIX_TARGET("avx512f,avx512bw,avx512vnni")
void row_sums_avx512_n(const std::uint8_t* codes, const std::uint16_t* scales, int k, int row_begin, int row_end,
                       const PairSlice* const* columns, int c0, int ncols, exact::I128* r, int* e_min) {
    const int tiles = k / kTile;
    for (int row = row_begin; row < row_end; ++row) {
        const std::uint8_t* w     = codes + static_cast<std::size_t>(row) * k;
        const std::uint16_t* tile = scales + static_cast<std::size_t>(row / kTile) * tiles;
        const int e               = fp8b128::min_exponent(tile, tiles);
        e_min[row - row_begin]    = e;
        exact::I128 out[NC];
        for (int t = 0; t < tiles; ++t) {
            __m512i v[NC][4];
            for (int c = 0; c < NC; ++c) {
                for (int i = 0; i < 4; ++i) { v[c][i] = _mm512_setzero_si512(); }
            }
            for (int q = 2 * t; q < 2 * t + 2; ++q) { // the tile's two slice pairs
                __m512i a[2], b[2];
                pair_weights(w + 64 * q, a, b);
                for (int c = 0; c < NC; ++c) {
                    const PairSlice& l = columns[c0 + c][q];
                    for (int half = 0; half < 2; ++half) {
                        const __m512i lo = _mm512_load_si512(l.lo[half]);
                        const __m512i hi = _mm512_load_si512(l.hi[half]);
                        v[c][0] = _mm512_dpwssd_epi32(v[c][0], a[half], lo);
                        v[c][1] = _mm512_dpwssd_epi32(v[c][1], a[half], hi);
                        v[c][2] = _mm512_dpwssd_epi32(v[c][2], b[half], lo);
                        v[c][3] = _mm512_dpwssd_epi32(v[c][3], b[half], hi);
                    }
                }
            }
            for (int c = 0; c < NC; ++c) {
                const std::int64_t p = (static_cast<std::int64_t>(_mm512_reduce_add_epi32(v[c][3])) << 19) +
                                       (static_cast<std::int64_t>(_mm512_reduce_add_epi32(v[c][1])) << 11) +
                                       (static_cast<std::int64_t>(_mm512_reduce_add_epi32(v[c][2])) << 8) +
                                       static_cast<std::int64_t>(_mm512_reduce_add_epi32(v[c][0]));
                fp8b128::add_tile(out[c], p, tile[t], e);
            }
        }
        for (int c = 0; c < NC; ++c) { r[static_cast<std::size_t>(row - row_begin) * ncols + c0 + c] = out[c]; }
    }
}

void row_sums_avx512(const std::uint8_t* codes, const std::uint16_t* scales, int k, int row_begin, int row_end,
                     const PairSlice* const* columns, int ncols, exact::I128* r, int* e_min) {
    for (int c0 = 0; c0 < ncols; c0 += 4) {
        switch (std::min(4, ncols - c0)) {
        case 1: row_sums_avx512_n<1>(codes, scales, k, row_begin, row_end, columns, c0, ncols, r, e_min); break;
        case 2: row_sums_avx512_n<2>(codes, scales, k, row_begin, row_end, columns, c0, ncols, r, e_min); break;
        case 3: row_sums_avx512_n<3>(codes, scales, k, row_begin, row_end, columns, c0, ncols, r, e_min); break;
        default: row_sums_avx512_n<4>(codes, scales, k, row_begin, row_end, columns, c0, ncols, r, e_min); break;
        }
    }
}

#endif // INFERNIX_CPU_X86

// The columns of one call as the row sums read them: the reference form takes CpuLimbSlice, the
// vector routes their even/odd split (built once per call, shared by every row).
class Columns {
public:
    Columns(CpuIsa isa, const CpuLimbSlice* const* columns, int ncols, int slices) : isa_(isa), columns_(columns) {
        if (!cpu_isa_supported(isa)) { throw std::invalid_argument("offloaded_moe fp8: unsupported CPU ISA"); }
#if defined(INFERNIX_CPU_X86)
        if (isa != CpuIsa::kScalar) {
            for (int c = 0; c < ncols; ++c) {
                split_slices(columns[c], slices, split_storage_ + static_cast<std::size_t>(c) * kMaxSlices);
                split_[c] = split_storage_ + static_cast<std::size_t>(c) * kMaxSlices;
                if (isa == CpuIsa::kAvx512Vnni) { // the 512-bit route's slice pairs
                    pair_slices(split_[c], slices, pair_storage_ + static_cast<std::size_t>(c) * (kMaxSlices / 2));
                    pairs_[c] = pair_storage_ + static_cast<std::size_t>(c) * (kMaxSlices / 2);
                }
            }
        }
#else
        (void)ncols;
        (void)slices;
#endif
    }

    void row_sums(const std::uint8_t* codes, const std::uint16_t* scales, int k, int row_begin, int row_end, int ncols,
                  exact::I128* r, int* e_min) const {
        switch (isa_) {
#if defined(INFERNIX_CPU_X86)
        case CpuIsa::kAvx2: row_sums_avx2(codes, scales, k, row_begin, row_end, split_, ncols, r, e_min); return;
        case CpuIsa::kAvxVnni: row_sums_avxvnni(codes, scales, k, row_begin, row_end, split_, ncols, r, e_min); return;
        case CpuIsa::kAvx512Vnni: row_sums_avx512(codes, scales, k, row_begin, row_end, pairs_, ncols, r, e_min); return;
#endif
        default: row_sums_scalar(codes, scales, k, row_begin, row_end, columns_, ncols, r, e_min); return;
        }
    }

private:
    static constexpr int kMaxSlices = kHidden / 32;
    CpuIsa isa_;
    const CpuLimbSlice* const* columns_;
#if defined(INFERNIX_CPU_X86)
    // 80 KiB per worker thread (and the 512-bit route's pairs, as much again): kept off the stack.
    static thread_local SplitLimbSlice split_storage_[kMaxColumns * kMaxSlices];
    static thread_local PairSlice pair_storage_[kMaxColumns * (kMaxSlices / 2)];
    const SplitLimbSlice* split_[kMaxColumns]{};
    const PairSlice* pairs_[kMaxColumns]{};
#endif
};

#if defined(INFERNIX_CPU_X86)
thread_local SplitLimbSlice Columns::split_storage_[kMaxColumns * kMaxSlices];
thread_local PairSlice Columns::pair_storage_[kMaxColumns * (kMaxSlices / 2)];
#endif

} // namespace

void fp8_gate_up(CpuIsa isa, const std::uint8_t* record, const CpuLimbSlice* const* x, const int* x_emax, int ncols,
                 int i_begin, int i_end, std::uint16_t* const* h) {
    constexpr int kUnit = 16;
    exact::I128 g[kUnit * kMaxColumns], u[kUnit * kMaxColumns];
    int ge[kUnit], ue[kUnit];
    const auto* scales = reinterpret_cast<const std::uint16_t*>(record + kScaleOffset);
    const Columns columns(isa, x, ncols, kHidden / 32);
    for (int i0 = i_begin; i0 < i_end; i0 += kUnit) {
        const int i1 = std::min(i_end, i0 + kUnit);
        columns.row_sums(record, scales, kHidden, i0, i1, ncols, g, ge);
        columns.row_sums(record + kGateBytes, scales + kGateTiles, kHidden, i0, i1, ncols, u, ue);
        for (int i = i0; i < i1; ++i) {
            for (int c = 0; c < ncols; ++c) {
                const std::size_t at = static_cast<std::size_t>(i - i0) * ncols + c;
                h[c][i] = canon::swiglu_bf16(fp8b128::output_bf16(g[at], ge[i - i0], x_emax[c]),
                                             fp8b128::output_bf16(u[at], ue[i - i0], x_emax[c]));
            }
        }
    }
}

void fp8_down(CpuIsa isa, const std::uint8_t* record, const CpuLimbSlice* const* h, const int* h_emax, int ncols,
              int row_begin, int row_end, std::uint16_t* const* y) {
    constexpr int kRows = 32;
    exact::I128 r[kRows * kMaxColumns];
    int e[kRows];
    const auto* scales = reinterpret_cast<const std::uint16_t*>(record + kScaleOffset) + 2 * kGateTiles;
    const Columns columns(isa, h, ncols, kIntermediate / 32);
    for (int r0 = row_begin; r0 < row_end; r0 += kRows) {
        const int r1 = std::min(row_end, r0 + kRows);
        columns.row_sums(record + 2 * kGateBytes, scales, kIntermediate, r0, r1, ncols, r, e);
        for (int row = r0; row < r1; ++row) {
            for (int c = 0; c < ncols; ++c) {
                y[c][row] = fp8b128::output_bf16(r[static_cast<std::size_t>(row - r0) * ncols + c], e[row - r0], h_emax[c]);
            }
        }
    }
}

} // namespace infernix::ops::offloaded_moe
