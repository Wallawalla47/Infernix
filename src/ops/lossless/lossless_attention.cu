#include "infernix/ops/lossless_ops.h"

#include "core/device.h"
#include "ops/kv_cache/q4_lloyd_codec.cuh"
#include "ops/kv_cache/vq2_codec.cuh"
#include "ops/lossless/cr_math.cuh"
#include "ops/lossless/wide.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

// The --lossless route's causal attention (include/infernix/ops/lossless_ops.h,
// docs/maintainer/lossless-route.md §2): s_j = (q . k_j) 2^-4 exact, m = max s_j,
// p_j = fp32(exp(fp32(s_j - m))), out = bf16(sum p_j v_j / sum p_j).
//
// Keys come from the call's own K/V (a window: query column t at position t) or from the paged cache
// (sequences of `width` columns at their given positions, each reading its keys 0..position through its
// block table row). A sequence's query rows are its (column, head) pairs, those of one KV head together.
// Main kernels, one CTA per 16 query rows of one KV head, key tiles of 32: pass A (each query's maximum
// over the CTA's key range, its own launch with only the q and k tiles in shared memory, several CTAs per
// SM), then pass B from the maximum, rounded in place. When a call has few CTAs, the keys split into ranges
// over more CTAs (pass A's partial maxima, then pass B from the overall maximum into partial int128 sums,
// then a combine kernel adds them and rounds), which changes no result: every sum is exact.
//  - each 64-element block of a q or k row is put on a grid of 29-bit integers M 2^(e - e_b + 21) relative
//    to its largest exponent e_b; a block's score part sum_d X_q X_k is exact in two int64 halves and placed
//    in an int128 at 2^(e_qb + e_kb) relative to the rows' largest block exponents, so a score is exact
//    unless a block lies more than 60 binades below them (then its query takes the exact fallback); a pair
//    whose row has an element off its grid (more than 21 binades under its block's largest) is scored
//    product by product from the inputs instead;
//  - pass A finds each query's exact maximum; pass B recomputes the scores, forms fp32(s - m) exactly
//    (a 320-bit difference) and p = cr_exp of it, sums p exactly in int128, and accumulates each channel's
//    sum p v in int128 from p on a 2^-48 grid (two 24-bit halves) and v on its channel's 29-bit grid
//    below the window's largest exponent, the grids' dropped bits bounded;
//  - the output is rounded to BF16 when the quotient's error interval (conversions and dropped bits)
//    lies within one rounding interval; otherwise its query takes the exact fallback.
// A codec key row whose exact values need more than 29 bits (a rotated FP8 row: an 18-bit E4M3 code times an
// 11-bit scale, then H_256 / 16) keeps the bits a shift to the grid would drop as a second 29-bit limb,
// x = (hi 2^down + lo) 2^(e_b - 155 - down) with 0 <= lo < 2^down: its scores add q . lo exactly at 2^-down
// below q . hi, so the row stays on the main path (rows needing more than 29 + 29 bits take the fallback).
// Fallback kernel, one warp per flagged query: every score as an exact fixed-point sum, the maximum and
// the differences exact, p stored, sum p and each sum p v exact, and the quotient's BF16 rounding decided
// by exact comparisons with the neighbouring midpoints (ties to even).
// A query whose own q, or a visible key's k or v, holds an Inf or NaN gives NaN in every channel.
// With a selection (QSA), query t sees key j <= t when j lies in its tail (at or after its last complete
// block of `ratio` keys) or in a block whose bit is set in its row of the selection.

namespace infernix::ops {
namespace {

using lossless::add_i128;
using lossless::add_shifted;
using lossless::I128;
using lossless::round_magnitude;
using lossless::Wide;

constexpr int kDim        = 256;
constexpr int kQueries    = 16;
constexpr int kKeys       = 32;
constexpr int kThreads    = 256;
constexpr int kScoreShift = 272;   // q k 2^-4 = M_q M_k 2^(e_q + e_k - 272)
constexpr int kNonFinite  = -1;    // kmax marker: the key column's k or v holds an Inf or NaN
constexpr std::uint16_t kNaN = 0x7FC0U;
constexpr int kPage         = 64;  // paged cache: positions per page (kPagedKVPageSize)
constexpr int kTargetCtas   = 384; // key ranges split until a call has about this many CTAs
constexpr int kMinSplitKeys = 256; // ... of at least this many keys each
constexpr int kMaximaKeys   = 64;  // keys per warp item of the maxima kernel
// kMaxima: pass A over the CTA's key range, its partial maximum stored (launched with only pass A's shared
// memory, kPassASmem, and at most 128 registers: two CTAs per SM). kSums: pass B from the ranges' maxima into
// partial sums (the split route; combine_kernel rounds). kFinal: pass B from the one range's maximum, rounded
// in place (a call of one range per CTA).
enum Mode { kMaxima, kSums, kFinal };

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("lossless_attention: ") + message); }
}

__device__ __forceinline__ void bf16_parts(std::uint16_t bits, int& m, int& e) {
    const int field = (bits >> 7) & 0xFF, mant = bits & 0x7F;
    m               = field == 0 ? mant : (128 | mant);
    e               = field == 0 ? 1 : field;
    if ((bits & 0x8000U) != 0) { m = -m; }
}

__device__ __forceinline__ int bf16_field(std::uint16_t bits) {
    const int field = (bits >> 7) & 0xFF;
    return (bits & 0x7FFF) == 0 ? 0 : (field == 0 ? 1 : field);
}

__device__ __forceinline__ bool bf16_nonfinite(std::uint16_t bits) { return ((bits >> 7) & 0xFF) == 0xFF; }

// ------------------------------------------------------------------------- call geometry

struct PartialMax { // one key range's maximum of a query row
    unsigned long long lo;
    long long hi;
    int unit;
    int flags; // kHave, kNan, kFail
};

struct PartialSum { // one key range's sums of a query row
    I128 num[kDim];
    I128 den;
    double p_err;
    float den_err;
    int flags;
};

constexpr int kHave = 1, kNan = 2, kFail = 4;

// One plane of the design reference's stored KV rows (lossless_codec_attention, LosslessKvPlane): row r's
// codes at codes + r code_bytes(codec) and scales at scales + r scale_bytes(codec).
struct CodecPlane {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    int codec; // LosslessKvCodec
    int rotated;
};

struct Args {
    const std::uint16_t* q;   // [256, heads, columns]
    std::uint16_t* out;       // [256, heads, columns]
    const std::uint16_t* k;   // window: [256, kv_heads, keys]; paged: the K pages
    const std::uint16_t* v;
    const std::int32_t* tables;     // paged: block tables (rows of table_stride); null: window
    const std::int32_t* table_rows; // paged: each sequence's table row
    int table_stride;
    const std::int32_t* positions; // [columns]; null: column c at position c (window)
    int width;                     // columns per sequence
    int sequences;
    int key_capacity;              // kmax entries per sequence
    std::int16_t* kmax;            // [sequences][key_capacity][kv_heads]
    int* vmax;                     // [sequences][kv_heads][256]
    int* vinexact;                 // [sequences][kv_heads][256] (key splits)
    std::uint32_t* flagged;        // one bit per (query column, head)
    int heads;
    int kv_heads;
    int columns;
    const std::uint32_t* select; // null: causal; else [columns][select_words] bits of the selected blocks (QSA)
    int select_words;
    int ratio;
    int tiles;      // 16-row tiles per sequence and KV head
    int splits;     // key ranges
    int split_keys; // keys per range (a multiple of kKeys)
    PartialMax* pmax; // [splits][columns][heads]
    PartialSum* psum;
    // The window form's keys and values on their grids, once per call (pack_kernel; null: each CTA encodes
    // the rows it reads): row (key j, KV head h) = j kv_heads + h.
    const std::int16_t* kpack;  // [rows][256] (M << 5) | s, M 2^s the grid integer (0 off the grid)
    const std::int16_t* kblk;   // [rows][4] the 64-blocks' exponents
    const std::uint8_t* koff;   // [rows] an element off its block's grid
    const std::int16_t* vpack;  // [rows][256] the value grid integers, packed alike
    const std::uint32_t* vround; // [rows][8] elements rounded to their channel's grid
    // The design reference's codec rows (lossless_codec_attention; kpack32 null: BF16 rows). Representation
    // rep 0 is the code planes, rep 1 the VQ formats' window rows (reps 2); a (rep, key, KV head) row is
    // rep codec_rows(columns, kv_heads) + codec_row(j, kvh). Its grid integers (int32, below 2^29) in kpack32 and vpack32,
    // its block exponents (the row's one exponent four times) in kblk, its off-grid flag in koff, its
    // rounded value elements in vround; kmax per (rep, key, KV head).
    const std::int32_t* kpack32;
    // Two-limb codec key rows (koff 1 + down, down in [1, 29]): the low limbs lo (0 <= lo < 2^down) of
    // x = hi 2^down + lo; undefined for other rows.
    const std::int32_t* kpack_lo;
    const std::int32_t* vpack32;
    CodecPlane kplane[2];
    CodecPlane vplane[2];
    int reps;
    int window_sinks;
    int window_recent;
};

__device__ __forceinline__ int position(const Args& a, int c) { return a.positions != nullptr ? a.positions[c] : c; }

// A codec plane's row of key j and KV head h: the production page order (64-key pages, heads within a page),
// as lossless_kv_encode stores the rows through an identity block table; a representation holds
// codec_rows(columns, kv_heads) rows (whole pages).
__host__ __device__ __forceinline__ std::size_t codec_row(int kv_heads, int j, int kvh) {
    return (static_cast<std::size_t>(j / kPage) * kv_heads + kvh) * kPage + j % kPage;
}
__host__ __device__ __forceinline__ std::size_t codec_rows(int columns, int kv_heads) {
    return static_cast<std::size_t>((columns + kPage - 1) / kPage) * kPage * kv_heads;
}

// The representation the pair (query column c, key j) reads: 0 its code rows, 1 its window rows (a VQ format's
// exact window: j < window_sinks or j >= the query's position - window_recent).
__device__ __forceinline__ int pair_rep(const Args& a, int c, int j) {
    return a.reps == 2 && (j < a.window_sinks || j >= position(a, c) - a.window_recent) ? 1 : 0;
}

// The last key of sequence b: its last column's position.
__device__ __forceinline__ int last_position(const Args& a, int b) { return position(a, b * a.width + a.width - 1); }

// Key (or value) row j of KV head kvh for sequence b, in elements from the K (V) base.
__device__ __forceinline__ std::size_t row_offset(const Args& a, int b, int j, int kvh) {
    if (a.tables == nullptr) { return (static_cast<std::size_t>(j) * a.kv_heads + kvh) * kDim; }
    const int page = a.tables[static_cast<std::size_t>(a.table_rows[b]) * a.table_stride + j / kPage];
    return ((static_cast<std::size_t>(page) * a.kv_heads + kvh) * kPage + j % kPage) * kDim;
}

__device__ __forceinline__ bool visible(const Args& a, int c, int j) {
    if (j > position(a, c)) { return false; }
    if (a.select == nullptr) { return true; }
    const int tail = (position(a, c) + 1) / a.ratio * a.ratio; // the query's last complete block's end
    if (j >= tail) { return true; }
    const int blk = j / a.ratio;
    return ((a.select[static_cast<std::size_t>(c) * a.select_words + blk / 32] >> (blk % 32)) & 1U) != 0;
}

// ------------------------------------------------------------------------- exponent maxima

// kmax[(b key_capacity + j) Hkv + h]: a key column's largest exponent field (kNonFinite when its k or v
// holds an Inf or NaN); vmax[(b Hkv + h) 256 + c]: a value channel's largest over the sequence's keys. One
// warp per (sequence, KV head, 64 keys).
__global__ void maxima_kernel(const Args a, int max_keys) {
    const int lane    = static_cast<int>(threadIdx.x) & 31;
    const int chunks  = (max_keys + kMaximaKeys - 1) / kMaximaKeys;
    const long long items = static_cast<long long>(a.sequences) * a.kv_heads * chunks;
    const long long warps = static_cast<long long>(gridDim.x) * (blockDim.x / 32);
    for (long long it = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32; it < items; it += warps) {
        const int chunk = static_cast<int>(it % chunks);
        const long long rest = it / chunks;
        const int kvh = static_cast<int>(rest % a.kv_heads), b = static_cast<int>(rest / a.kv_heads);
        const int last = last_position(a, b), j0 = chunk * kMaximaKeys;
        if (j0 > last) { continue; }
        int vm[kDim / 32] = {};
        for (int j = j0; j < min(j0 + kMaximaKeys, last + 1); ++j) {
            const std::size_t off   = row_offset(a, b, j, kvh);
            const std::uint16_t* kr = a.k + off;
            const std::uint16_t* vr = a.v + off;
            int e    = 0;
            bool bad = false;
#pragma unroll
            for (int i = 0; i < kDim / 32; ++i) {
                const std::uint16_t kb = kr[lane + 32 * i], vb = vr[lane + 32 * i];
                e   = max(e, bf16_field(kb));
                bad = bad || bf16_nonfinite(kb) || bf16_nonfinite(vb);
                if (!bf16_nonfinite(vb)) { vm[i] = max(vm[i], bf16_field(vb)); }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) { e = max(e, __shfl_xor_sync(0xFFFFFFFFU, e, o)); }
            bad = __any_sync(0xFFFFFFFFU, bad);
            if (lane == 0) {
                a.kmax[(static_cast<std::size_t>(b) * a.key_capacity + j) * a.kv_heads + kvh] =
                    static_cast<std::int16_t>(bad ? kNonFinite : (e == 0 ? 1 : e));
            }
        }
#pragma unroll
        for (int i = 0; i < kDim / 32; ++i) {
            if (vm[i] != 0) { atomicMax(a.vmax + (static_cast<std::size_t>(b) * a.kv_heads + kvh) * kDim + lane + 32 * i, vm[i]); }
        }
    }
}

// ------------------------------------------------------------------------- scores

struct Score {
    I128 value;
    int unit; // the score is value 2^unit
};

// Sign and top-bit exponent of a score; exponent INT_MIN for zero.
__device__ __forceinline__ int score_exponent(const Score& s, std::uint32_t (&mag)[4], bool& negative) {
    negative = lossless::magnitude_i128(s.value, mag);
#pragma unroll
    for (int i = 3; i >= 0; --i) {
        if (mag[i] != 0) { return 32 * i + 31 - __clz(mag[i]) + s.unit; }
    }
    return INT_MIN;
}

__device__ __forceinline__ void shift_left4(std::uint32_t (&w)[4], int n) {
    for (; n >= 32; n -= 32) {
        w[3] = w[2];
        w[2] = w[1];
        w[1] = w[0];
        w[0] = 0;
    }
    if (n > 0) {
        w[3] = (w[3] << n) | (w[2] >> (32 - n));
        w[2] = (w[2] << n) | (w[1] >> (32 - n));
        w[1] = (w[1] << n) | (w[0] >> (32 - n));
        w[0] <<= n;
    }
}

// Whether a > b (exactly).
__device__ inline bool score_greater(const Score& a, const Score& b) {
    std::uint32_t ma[4], mb[4];
    bool na, nb;
    const int ea = score_exponent(a, ma, na), eb = score_exponent(b, mb, nb);
    const bool za = ea == INT_MIN, zb = eb == INT_MIN;
    if (za && zb) { return false; }
    if (za) { return nb; }
    if (zb) { return !na; }
    if (na != nb) { return nb; }
    int cmp; // |a| against |b|
    if (ea != eb) {
        cmp = ea > eb ? 1 : -1;
    } else {
        // Equal top exponents: shift the larger-unit magnitude left by the unit difference (<= 127 here).
        const int shift = a.unit - b.unit;
        if (shift > 0) { shift_left4(ma, shift); } else if (shift < 0) { shift_left4(mb, -shift); }
        cmp = 0;
        for (int i = 3; i >= 0 && cmp == 0; --i) {
            if (ma[i] != mb[i]) { cmp = ma[i] > mb[i] ? 1 : -1; }
        }
    }
    return na ? cmp < 0 : cmp > 0;
}

// fp32(s - m) for scores whose units differ by at most 96: both placed in a 320-bit fixed point at the
// smaller unit (16-bit pieces, so each add stays below 2^31), subtracted, rounded once. ok = false when
// the units are further apart.
__device__ inline float score_difference(const Score& s, const Score& m, bool& ok) {
    const int base = min(s.unit, m.unit);
    if (max(s.unit, m.unit) - base > 96) {
        ok = false;
        return 0.0F;
    }
    Wide<10> acc;
    acc.clear();
    auto add128 = [&](const I128& v, int position, int sign) {
        std::uint32_t mag[4];
        const int sg = lossless::magnitude_i128(v, mag) ? -sign : sign;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if ((mag[i] & 0xFFFFU) != 0) { acc.add(sg * static_cast<long long>(mag[i] & 0xFFFFU), position + 32 * i); }
            if ((mag[i] >> 16) != 0) { acc.add(sg * static_cast<long long>(mag[i] >> 16), position + 32 * i + 16); }
        }
    };
    add128(s.value, s.unit - base, 1);
    add128(m.value, m.unit - base, -1);
    std::uint32_t mag[10];
    const bool negative = acc.finish(mag);
    return __uint_as_float(round_magnitude(mag, base, negative, false));
}

// ------------------------------------------------------------------------- main kernel

constexpr int kBlocks   = kDim / 64; // 64-element blocks of a q or k row
constexpr int kGridBits = 21;        // grid integers M 2^(e - e_block + 21): exact within 21 binades
constexpr int kPBits    = 48;        // p on a grid of 2^-48 (two 24-bit halves)

// A q or k row in shared memory is padded by one word: a row's 16 threads read 16 rows at one element.
constexpr int kRowStride = kDim + 1;

// Pass A's members first: the maxima kernel (kMaxima) is launched with only that prefix
// (kPassASmem), so several of its CTAs fit an SM.
struct Shared {
    int xq[kQueries][kRowStride]; // query grid integers (0 off the grid)
    int xk[kKeys][kRowStride];    // key grid integers
    std::int16_t qb[kQueries][kBlocks];
    std::int16_t kb[kKeys][kBlocks];
    int qmax[kQueries];
    int qpos[kQueries];      // the row's position (-1: inactive)
    int qbad[kQueries];
    int qoff[kQueries];      // the query row has an element off its block's grid
    int kmax[kKeys];         // the key row's largest exponent field, or kNonFinite
    int koff[kKeys];
    int klo[kKeys];          // a two-limb codec row's down (its low limb's place below xk's), else 0
    // Pass B only.
    int vg[kKeys][kDim];     // value grid integers: M 2^(e - vmax_c + 21), rounded below the grid
    float p[kQueries][kKeys];
    int ph[kQueries][kKeys]; // p 2^48 = ph 2^24 + pl
    int pl[kQueries][kKeys];
    int vmax[kDim];
    int vinexact[kDim];      // a value element of the channel was rounded to its grid
};
constexpr int kPassASmem = static_cast<int>(offsetof(Shared, vg));

// One warp encodes a 64-element block (lane: elements 2 lane, 2 lane + 1; src null: zeros): its largest
// exponent field e_b, the grid integers M 2^(e - e_b + 21), whether an element fell off the grid.
__device__ __forceinline__ void encode_block(const std::uint16_t* src, int lane, int* x, std::int16_t& block_exp,
                                             int& off) {
    int m[2] = {0, 0}, e[2] = {0, 0};
    if (src != nullptr) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const std::uint16_t bits = src[2 * lane + i];
            if (!bf16_nonfinite(bits)) {
                bf16_parts(bits, m[i], e[i]);
                if (m[i] == 0) { e[i] = 0; }
            }
        }
    }
    int eb = max(e[0], e[1]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { eb = max(eb, __shfl_xor_sync(0xFFFFFFFFU, eb, o)); }
    if (eb == 0) { eb = 1; }
    bool lost = false;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int shift = e[i] - eb + kGridBits;
        int X           = 0;
        if (m[i] != 0) {
            if (shift >= 0) {
                X = m[i] * (1 << shift);
            } else {
                lost = true;
            }
        }
        x[2 * lane + i] = X;
    }
    lost = __any_sync(0xFFFFFFFFU, lost);
    if (lane == 0) {
        block_exp = static_cast<std::int16_t>(eb);
        if (lost) { off = 1; }
    }
}

// A grid integer M 2^s (|M| <= 255, 0 <= s <= 21) packed in 16 bits as (M << 5) | s.
__device__ __forceinline__ std::int16_t pack_grid(int m, int shift) { return static_cast<std::int16_t>(m * 32 + shift); }
__device__ __forceinline__ int unpack_grid(std::int16_t p) {
    const int v = p;
    return (v >> 5) * (1 << (v & 31));
}

// v 2^(155 - vmax), packed: the value's grid integer for its channel (rounded to nearest even below the grid).
__device__ __forceinline__ std::int16_t value_pack(std::uint16_t bits, int vmax, bool& rounded) {
    int m = 0, e = 1;
    bf16_parts(bits, m, e);
    if (m == 0) { return 0; }
    const int shift = e - vmax + kGridBits;
    if (shift >= 0) { return pack_grid(m, shift); }
    rounded         = true;
    const int down  = -shift;
    const int mag   = m < 0 ? -m : m;
    if (down >= 9) { return 0; }
    const int q = mag >> down, r = mag & ((1 << down) - 1), half = 1 << (down - 1);
    const int v = q + ((r > half || (r == half && (q & 1) != 0)) ? 1 : 0);
    return pack_grid(m < 0 ? -v : v, 0);
}

__device__ __forceinline__ int value_grid(std::uint16_t bits, int vmax, bool& rounded) {
    return unpack_grid(value_pack(bits, vmax, rounded));
}

// One warp packs key row `row` (window form) as encode_block encodes it: its blocks' grid integers (0 off
// the grid), exponents and off-grid flag; and its values with their rounded-element mask. A row whose k or
// v holds an Inf or NaN (kmax kNonFinite) is zeros, as load_keys encodes it.
__global__ void pack_kernel(const Args a, std::int16_t* kpack, std::int16_t* kblk, std::uint8_t* koff,
                            std::int16_t* vpack, std::uint32_t* vround) {
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const long long items = static_cast<long long>(a.columns) * a.kv_heads;
    const long long warps = static_cast<long long>(gridDim.x) * (blockDim.x / 32);
    for (long long it = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32; it < items; it += warps) {
        const std::size_t row = static_cast<std::size_t>(it);
        const int kvh         = static_cast<int>(it % a.kv_heads);
        const bool valid      = a.kmax[row] != kNonFinite;
        bool lost             = false;
#pragma unroll
        for (int blk = 0; blk < kDim / 64; ++blk) {
            int m[2] = {0, 0}, e[2] = {0, 0};
            if (valid) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const std::uint16_t bits = a.k[row * kDim + 64 * blk + 2 * lane + i];
                    if (!bf16_nonfinite(bits)) {
                        bf16_parts(bits, m[i], e[i]);
                        if (m[i] == 0) { e[i] = 0; }
                    }
                }
            }
            int eb = max(e[0], e[1]);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) { eb = max(eb, __shfl_xor_sync(0xFFFFFFFFU, eb, o)); }
            if (eb == 0) { eb = 1; }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int shift = e[i] - eb + kGridBits;
                std::int16_t x  = 0;
                if (m[i] != 0) {
                    if (shift >= 0) {
                        x = pack_grid(m[i], shift);
                    } else {
                        lost = true;
                    }
                }
                kpack[row * kDim + 64 * blk + 2 * lane + i] = x;
            }
            if (lane == 0) { kblk[row * (kDim / 64) + blk] = static_cast<std::int16_t>(eb); }
        }
        lost = __any_sync(0xFFFFFFFFU, lost);
        if (lane == 0) { koff[row] = lost ? 1 : 0; }
#pragma unroll
        for (int i = 0; i < kDim / 32; ++i) {
            const int d       = lane + 32 * i;
            bool rounded      = false;
            std::int16_t x    = 0;
            if (valid) { x = value_pack(a.v[row * kDim + d], max(a.vmax[kvh * kDim + d], 1), rounded); }
            vpack[row * kDim + d] = x;
            const unsigned mask   = __ballot_sync(0xFFFFFFFFU, rounded);
            if (lane == 0) { vround[row * (kDim / 32) + i] = mask; }
        }
    }
}

// A signed 128-bit B (two int64 partial sums) times 2^shift (shift <= 60) into acc; bits below 2^0 set
// `dropped`.
__device__ __forceinline__ void add_block(I128& acc, long long s0, long long s1, int shift, bool& dropped) {
    I128 b = add_i128(I128{static_cast<unsigned long long>(s0), s0 < 0 ? -1LL : 0LL},
                      I128{static_cast<unsigned long long>(s1), s1 < 0 ? -1LL : 0LL});
    if (b.lo == 0 && b.hi == 0) { return; }
    if (shift > 0) {
        b.hi = static_cast<long long>((static_cast<unsigned long long>(b.hi) << shift) | (b.lo >> (64 - shift)));
        b.lo <<= shift;
    } else if (shift < 0) {
        const int down = -shift;
        if (down >= 64) {
            dropped = true;
            return;
        }
        if ((b.lo & ((1ULL << down) - 1)) != 0) { dropped = true; }
        b.lo = (b.lo >> down) | (static_cast<unsigned long long>(b.hi) << (64 - down));
        b.hi >>= down;
    }
    acc = add_i128(acc, b);
}

// The exact score of a pair whose row has an element off its grid, from the inputs (rare).
__device__ Score exact_pair_score(const Args& a, int b, int c, int h, int kvh, int j, int qmax, int kmax,
                                  bool& dropped) {
    const std::uint16_t* qr = a.q + (static_cast<std::size_t>(c) * a.heads + h) * kDim;
    const std::uint16_t* kr = a.k + row_offset(a, b, j, kvh);
    I128 acc{0ULL, 0LL};
    for (int d = 0; d < kDim; ++d) {
        int mq, eq, mk, ek;
        bf16_parts(qr[d], mq, eq);
        bf16_parts(kr[d], mk, ek);
        if (mq == 0 || mk == 0) { continue; }
        add_shifted(acc, static_cast<long long>(mq * mk), 64 + (eq - qmax) + (ek - kmax), dropped);
    }
    return {acc, qmax + kmax - kScoreShift - 64};
}

// The exact score of a pair whose query row has an element off its grid, against an on-grid codec row (its
// grid integers are exact: k_d = X_d 2^(kmax - 155), a BF16-like element of significand X_d, below 2^29, and
// field kmax - 21).
__device__ Score codec_pair_score(const Args& a, int c, int h, int kvh, int j, int qmax, int kmax, bool& dropped) {
    const std::uint16_t* qr = a.q + (static_cast<std::size_t>(c) * a.heads + h) * kDim;
    const std::size_t row   = static_cast<std::size_t>(pair_rep(a, c, j)) * codec_rows(a.columns, a.kv_heads) +
                            codec_row(a.kv_heads, j, kvh);
    const std::int32_t* kr = a.kpack32 + row * kDim;
    const int off          = a.koff[row];
    const int down         = off >= 2 ? off - 1 : 0; // a two-limb row's low limb place
    const std::int32_t* lo = down > 0 ? a.kpack_lo + row * kDim : nullptr;
    I128 acc{0ULL, 0LL};
    for (int d = 0; d < kDim; ++d) {
        int mq, eq;
        bf16_parts(qr[d], mq, eq);
        if (mq == 0) { continue; }
        if (kr[d] != 0) { add_shifted(acc, static_cast<long long>(mq) * kr[d], 64 + (eq - qmax), dropped); }
        if (lo != nullptr && lo[d] != 0) {
            add_shifted(acc, static_cast<long long>(mq) * lo[d], 64 + (eq - qmax) - down, dropped);
        }
    }
    return {acc, qmax + (kmax - 21) - kScoreShift - 64};
}

// The score of query row i and key row jj: per 64-block, the grid integers' products summed exactly in
// int64 halves, placed at the block's exponents relative to the rows' largest.
__device__ __forceinline__ Score tile_score(const Shared& sh, const Args& a, int seq, int c, int h, int kvh, int j0,
                                            int i, int jj, bool& dropped) {
    if (sh.qoff[i] != 0 || sh.koff[jj] != 0) {
        if (a.kpack32 != nullptr) {
            if (sh.koff[jj] != 0) { // a codec row off its grid: the pair's query takes the exact fallback
                dropped = true;
                return {I128{0ULL, 0LL}, 0};
            }
            return codec_pair_score(a, c, h, kvh, j0 + jj, sh.qmax[i], sh.kmax[jj], dropped);
        }
        return exact_pair_score(a, seq, c, h, kvh, j0 + jj, sh.qmax[i], sh.kmax[jj], dropped);
    }
    I128 acc{0ULL, 0LL};
#pragma unroll
    for (int b = 0; b < kBlocks; ++b) {
        long long s0 = 0, s1 = 0;
#pragma unroll 8
        for (int d = 0; d < 32; ++d) {
            s0 += static_cast<long long>(sh.xq[i][64 * b + d]) * sh.xk[jj][64 * b + d];
            s1 += static_cast<long long>(sh.xq[i][64 * b + 32 + d]) * sh.xk[jj][64 * b + 32 + d];
        }
        // q k 2^-4 per block: s 2^(e_qb + e_kb - 314); the unit 2^(E_q + E_k - 374).
        const int shift = 60 - (sh.qmax[i] - sh.qb[i][b]) - (sh.kmax[jj] - sh.kb[jj][b]);
        add_block(acc, s0, s1, shift, dropped);
    }
    return {acc, sh.qmax[i] + sh.kmax[jj] - 374};
}

// The scores of query row i with key rows jj and jj + 16 where wanted, as tile_score gives them: when both
// rows are on their grids, one pass over the q row serves both keys.
__device__ __forceinline__ void tile_scores(const Shared& sh, const Args& a, int seq, int c, int h, int kvh, int j0,
                                            int i, int jj, const bool (&want)[2], Score (&s)[2], bool& dropped) {
    const int k0 = jj, k1 = jj + 16;
    if (!want[0] || !want[1] || sh.qoff[i] != 0 || sh.koff[k0] != 0 || sh.koff[k1] != 0) {
        if (want[0]) { s[0] = tile_score(sh, a, seq, c, h, kvh, j0, i, k0, dropped); }
        if (want[1]) { s[1] = tile_score(sh, a, seq, c, h, kvh, j0, i, k1, dropped); }
        return;
    }
    I128 acc0{0ULL, 0LL}, acc1{0ULL, 0LL};
#pragma unroll
    for (int b = 0; b < kBlocks; ++b) {
        long long s00 = 0, s01 = 0, s10 = 0, s11 = 0;
#pragma unroll 8
        for (int d = 0; d < 32; ++d) {
            const long long qa = sh.xq[i][64 * b + d], qb = sh.xq[i][64 * b + 32 + d];
            s00 += qa * sh.xk[k0][64 * b + d];
            s01 += qb * sh.xk[k0][64 * b + 32 + d];
            s10 += qa * sh.xk[k1][64 * b + d];
            s11 += qb * sh.xk[k1][64 * b + 32 + d];
        }
        const int qshift = 60 - (sh.qmax[i] - sh.qb[i][b]);
        add_block(acc0, s00, s01, qshift - (sh.kmax[k0] - sh.kb[k0][b]), dropped);
        add_block(acc1, s10, s11, qshift - (sh.kmax[k1] - sh.kb[k1][b]), dropped);
    }
    s[0] = {acc0, sh.qmax[i] + sh.kmax[k0] - 374};
    s[1] = {acc1, sh.qmax[i] + sh.kmax[k1] - 374};
}

// The low limbs of two-limb codec rows, xk holding them (load_low_limbs): q . lo per block added to the
// pair's score at 2^-down below where tile_scores placed q . hi. A query off its grid scored its pair
// product by product already (codec_pair_score, both limbs).
__device__ __forceinline__ void add_low_limbs(const Shared& sh, int i, int sub, const bool (&want)[2], Score (&s)[2],
                                              bool& dropped) {
    if (sh.qoff[i] != 0) { return; }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int jj = sub + 16 * half;
        if (!want[half] || sh.klo[jj] == 0) { continue; }
#pragma unroll
        for (int b = 0; b < kBlocks; ++b) {
            long long s0 = 0, s1 = 0;
#pragma unroll 8
            for (int d = 0; d < 32; ++d) {
                s0 += static_cast<long long>(sh.xq[i][64 * b + d]) * sh.xk[jj][64 * b + d];
                s1 += static_cast<long long>(sh.xq[i][64 * b + 32 + d]) * sh.xk[jj][64 * b + 32 + d];
            }
            const int shift = 60 - (sh.qmax[i] - sh.qb[i][b]) - (sh.kmax[jj] - sh.kb[jj][b]) - sh.klo[jj];
            add_block(s[half].value, s0, s1, shift, dropped);
        }
    }
}

// blockIdx: x = sequence b's 16-row tile, y = KV head, z = key range. Row r of the tile is the sequence's
// (column, head) pair p = 16 tile + r: column b width + p / G, head kvh G + p % G (G = heads / kv_heads).
template <int M>
__global__ __launch_bounds__(kThreads, M == kMaxima ? 2 : 1) void attention_kernel(const Args args) {
    extern __shared__ __align__(16) unsigned char raw[];
    Shared& sh    = *reinterpret_cast<Shared*>(raw);
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int b = static_cast<int>(blockIdx.x) / args.tiles, tile = static_cast<int>(blockIdx.x) % args.tiles;
    const int kvh = static_cast<int>(blockIdx.y), split = static_cast<int>(blockIdx.z);
    const int group = args.heads / args.kv_heads, pairs = args.width * group;
    auto row_active = [&](int r) { return tile * kQueries + r < pairs; };
    auto row_column = [&](int r) { return b * args.width + (tile * kQueries + r) / group; };
    auto row_head   = [&](int r) { return kvh * group + (tile * kQueries + r) % group; };
    const int i = tid / 16, sub = tid % 16; // this thread's query row and its slot among the row's 16 threads
    const bool active = row_active(i);
    const int c = active ? row_column(i) : 0, h = active ? row_head(i) : 0;

    // Query tile: positions, non-finite rows, the grid integers per block, each row's largest block exponent.
    if (tid < kQueries) {
        sh.qbad[tid] = 0;
        sh.qoff[tid] = 0;
        sh.qpos[tid] = row_active(tid) ? position(args, row_column(tid)) : -1;
    }
    if constexpr (M != kMaxima) { // pass B's members lie past kPassASmem
        for (int d = tid; d < kDim; d += kThreads) {
            sh.vmax[d]     = max(args.vmax[(static_cast<std::size_t>(b) * args.kv_heads + kvh) * kDim + d], 1);
            sh.vinexact[d] = 0;
        }
    }
    __syncthreads();
    for (int idx = tid; idx < kQueries * kDim; idx += kThreads) {
        const int r = idx / kDim, d = idx % kDim;
        if (row_active(r) &&
            bf16_nonfinite(args.q[(static_cast<std::size_t>(row_column(r)) * args.heads + row_head(r)) * kDim + d])) {
            sh.qbad[r] = 1;
        }
    }
    for (int unit = warp; unit < kQueries * kBlocks; unit += kThreads / 32) {
        const int r = unit / kBlocks, blk = unit % kBlocks;
        const std::uint16_t* src =
            row_active(r) ? args.q + (static_cast<std::size_t>(row_column(r)) * args.heads + row_head(r)) * kDim + 64 * blk
                          : nullptr;
        encode_block(src, lane, &sh.xq[r][64 * blk], sh.qb[r][blk], sh.qoff[r]);
    }
    __syncthreads();
    if (tid < kQueries) {
        int e = 1;
        for (int blk = 0; blk < kBlocks; ++blk) { e = max(e, static_cast<int>(sh.qb[tid][blk])); }
        sh.qmax[tid] = e;
    }
    __syncthreads();

    // This CTA's keys: its range of the tile's keys 0..(the largest row position).
    int last_key = -1;
    for (int r = 0; r < kQueries; ++r) { last_key = max(last_key, sh.qpos[r]); }
    const int key_begin = split * args.split_keys;
    const int key_end   = min(key_begin + args.split_keys, last_key + 1);
    bool nan = active && sh.qbad[i] != 0;
    bool tile_lo = false; // the loaded tile has two-limb codec rows
    auto load_keys = [&](int j0, bool values, int rep) {
        if (args.kpack32 != nullptr) {
            // Codec rows (representation rep), packed once per call (codec_pack_kernel): copied.
            const std::size_t base = static_cast<std::size_t>(rep) * codec_rows(args.columns, args.kv_heads);
            int two_limb = 0;
            for (int r = tid; r < kKeys; r += kThreads) {
                const int j   = j0 + r;
                sh.kmax[r]    = j < key_end ? args.kmax[(static_cast<std::size_t>(rep) * args.key_capacity + j) * args.kv_heads + kvh] : 1;
                const int off = j < key_end ? args.koff[base + codec_row(args.kv_heads, j, kvh)] : 0;
                sh.koff[r]    = off == 1 ? 1 : 0;      // past two limbs: the exact fallback
                sh.klo[r]     = off >= 2 ? off - 1 : 0; // two limbs: the low limb's down
                two_limb |= sh.klo[r] != 0 ? 1 : 0;
            }
            tile_lo = __syncthreads_or(two_limb) != 0;
            __syncthreads();
            for (int idx = tid; idx < kKeys * kDim; idx += kThreads) {
                const int r = idx / kDim, d = idx % kDim, j = j0 + r;
                const bool ok         = j < key_end && sh.kmax[r] != kNonFinite;
                const std::size_t row = ok ? base + codec_row(args.kv_heads, j, kvh) : 0;
                sh.xk[r][d]           = ok ? args.kpack32[row * kDim + d] : 0;
                if (values) {
                    int x = 0;
                    if (ok) {
                        x = args.vpack32[row * kDim + d];
                        if (((args.vround[row * (kDim / 32) + d / 32] >> (d % 32)) & 1U) != 0) { sh.vinexact[d] = 1; }
                    }
                    sh.vg[r][d] = x;
                }
            }
            for (int idx = tid; idx < kKeys * kBlocks; idx += kThreads) {
                const int r = idx / kBlocks, blk = idx % kBlocks, j = j0 + r;
                const bool ok = j < key_end && sh.kmax[r] != kNonFinite;
                sh.kb[r][blk] = ok ? args.kblk[(base + codec_row(args.kv_heads, j, kvh)) * kBlocks + blk] : 1;
            }
            __syncthreads();
            return;
        }
        tile_lo = false;
        for (int r = tid; r < kKeys; r += kThreads) {
            const int j = j0 + r;
            sh.kmax[r]  = j < key_end ? args.kmax[(static_cast<std::size_t>(b) * args.key_capacity + j) * args.kv_heads + kvh] : 1;
            sh.koff[r]  = 0;
        }
        __syncthreads();
        if (args.kpack != nullptr) {
            // The window form's rows, packed once per call (pack_kernel): copied and unpacked.
            for (int idx = tid; idx < kKeys * kDim; idx += kThreads) {
                const int r = idx / kDim, d = idx % kDim, j = j0 + r;
                const bool ok         = j < key_end && sh.kmax[r] != kNonFinite;
                const std::size_t row = static_cast<std::size_t>(j) * args.kv_heads + kvh;
                sh.xk[r][d]           = ok ? unpack_grid(args.kpack[row * kDim + d]) : 0;
                if (values) {
                    int x = 0;
                    if (ok) {
                        x = unpack_grid(args.vpack[row * kDim + d]);
                        if (((args.vround[row * (kDim / 32) + d / 32] >> (d % 32)) & 1U) != 0) { sh.vinexact[d] = 1; }
                    }
                    sh.vg[r][d] = x;
                }
            }
            for (int idx = tid; idx < kKeys * kBlocks; idx += kThreads) {
                const int r = idx / kBlocks, blk = idx % kBlocks, j = j0 + r;
                const bool ok = j < key_end && sh.kmax[r] != kNonFinite;
                sh.kb[r][blk] = ok ? args.kblk[(static_cast<std::size_t>(j) * args.kv_heads + kvh) * kBlocks + blk] : 1;
                if (blk == 0 && ok && args.koff[static_cast<std::size_t>(j) * args.kv_heads + kvh] != 0) { sh.koff[r] = 1; }
            }
            __syncthreads();
            return;
        }
        for (int unit = warp; unit < kKeys * kBlocks; unit += kThreads / 32) {
            const int r = unit / kBlocks, blk = unit % kBlocks, j = j0 + r;
            const std::uint16_t* src =
                j < key_end && sh.kmax[r] != kNonFinite ? args.k + row_offset(args, b, j, kvh) + 64 * blk : nullptr;
            encode_block(src, lane, &sh.xk[r][64 * blk], sh.kb[r][blk], sh.koff[r]);
        }
        if (values) {
            for (int idx = tid; idx < kKeys * kDim; idx += kThreads) {
                const int r = idx / kDim, d = idx % kDim, j = j0 + r;
                int x = 0;
                if (j < key_end && sh.kmax[r] != kNonFinite) {
                    bool rounded = false;
                    x = value_grid(args.v[row_offset(args, b, j, kvh) + d], sh.vmax[d], rounded);
                    if (rounded) { sh.vinexact[d] = 1; }
                }
                sh.vg[r][d] = x;
            }
        }
        __syncthreads();
    };

    // After the high limbs' scores of a tile with two-limb codec rows (tile_lo): xk takes their low limbs
    // (others 0) for add_low_limbs. Every thread calls it; xk is not read again before the next load_keys.
    auto load_low_limbs = [&](int j0, int rep) {
        __syncthreads();
        const std::size_t base = static_cast<std::size_t>(rep) * codec_rows(args.columns, args.kv_heads);
        for (int idx = tid; idx < kKeys * kDim; idx += kThreads) {
            const int r = idx / kDim, d = idx % kDim, j = j0 + r;
            const bool lo         = j < key_end && sh.klo[r] != 0 && sh.kmax[r] != kNonFinite;
            sh.xk[r][d]           = lo ? args.kpack_lo[(base + codec_row(args.kv_heads, j, kvh)) * kDim + d] : 0;
        }
        __syncthreads();
    };

    Score best{{0ULL, 0LL}, 0};
    bool have = false, fail = false;
    // Codec rows with a window read each tile in up to two representations (a pair reads one of them); a
    // representation no pair of the tile reads is skipped.
    const int reps = args.reps > 1 ? 2 : 1;
    auto pair_wanted = [&](int j0, int half, int rep) {
        const int j = j0 + sub + 16 * half;
        return active && j < key_end && visible(args, c, j) && pair_rep(args, c, j) == rep;
    };
    if constexpr (M == kMaxima) {
        // Pass A: each query's exact maximum (each thread: keys sub and sub + 16 of every tile).
        for (int j0 = key_begin; j0 < key_end; j0 += kKeys) {
            for (int rep = 0; rep < reps; ++rep) {
                bool want[2] = {pair_wanted(j0, 0, rep), pair_wanted(j0, 1, rep)};
                if (reps > 1 && __syncthreads_or(want[0] || want[1] ? 1 : 0) == 0) { continue; }
                load_keys(j0, false, rep);
                Score sc[2];
                if (active) {
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        if (want[half] && sh.kmax[sub + 16 * half] == kNonFinite) {
                            nan        = true;
                            want[half] = false;
                        }
                    }
                    tile_scores(sh, args, b, c, h, kvh, j0, i, sub, want, sc, fail);
                }
                if (tile_lo) {
                    load_low_limbs(j0, rep);
                    if (active) { add_low_limbs(sh, i, sub, want, sc, fail); }
                }
                if (active) {
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        if (want[half] && (!have || score_greater(sc[half], best))) {
                            best = sc[half];
                            have = true;
                        }
                    }
                }
                __syncthreads();
            }
        }
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) {
            Score other;
            other.value.lo        = __shfl_xor_sync(0xFFFFFFFFU, best.value.lo, o);
            other.value.hi        = __shfl_xor_sync(0xFFFFFFFFU, best.value.hi, o);
            other.unit            = __shfl_xor_sync(0xFFFFFFFFU, best.unit, o);
            const bool other_have = __shfl_xor_sync(0xFFFFFFFFU, have ? 1 : 0, o) != 0;
            if (other_have && (!have || score_greater(other, best))) {
                best = other;
                have = true;
            }
            nan = __shfl_xor_sync(0xFFFFFFFFU, nan ? 1 : 0, o) != 0 || nan;
        }
    }
    if constexpr (M == kMaxima) {
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) { fail = __shfl_xor_sync(0xFFFFFFFFU, fail ? 1 : 0, o) != 0 || fail; }
        if (active && sub == 0) {
            args.pmax[(static_cast<std::size_t>(split) * args.columns + c) * args.heads + h] =
                PartialMax{best.value.lo, best.value.hi, best.unit,
                           (have ? kHave : 0) | (nan ? kNan : 0) | (fail ? kFail : 0)};
        }
        return;
    }
    if constexpr (M == kSums || M == kFinal) {
        // The overall maximum from every range's (each of the row's 16 threads alike).
        if (active) {
            for (int s = 0; s < args.splits; ++s) {
                const PartialMax pm = args.pmax[(static_cast<std::size_t>(s) * args.columns + c) * args.heads + h];
                nan  = nan || (pm.flags & kNan) != 0;
                fail = fail || (pm.flags & kFail) != 0;
                const Score sc{{pm.lo, pm.hi}, pm.unit};
                if ((pm.flags & kHave) != 0 && (!have || score_greater(sc, best))) {
                    best = sc;
                    have = true;
                }
            }
        }
    }

    // Pass B: p, sum p (int128 at 2^-87; p <= 1) and the thread's 16 channels' sum p v (int128 at
    // 2^(vmax_c - 203): p on its 2^-48 grid, v on its channel's grid).
    // This thread's channels: sub + 16 cc (the row's 16 threads read 16 banks of a value row).
    I128 num[16];
#pragma unroll
    for (int cc = 0; cc < 16; ++cc) { num[cc] = I128{0ULL, 0LL}; }
    I128 den{0ULL, 0LL};
    float den_err = 0.0F;
    double p_err  = 0.0; // the p grid's dropped bits (each below 2^-49), summed over this thread's keys
    // A row pass A flagged takes the exact fallback: its sums are not formed.
    const bool work = active && !nan && !fail;
    for (int j0 = key_begin; j0 < key_end; j0 += kKeys) {
        for (int rep = 0; rep < reps; ++rep) {
            bool want[2] = {work && pair_wanted(j0, 0, rep), work && pair_wanted(j0, 1, rep)};
            if (reps > 1 && __syncthreads_or(want[0] || want[1] ? 1 : 0) == 0) { continue; }
            load_keys(j0, true, rep);
            Score sc[2];
            if (work) { tile_scores(sh, args, b, c, h, kvh, j0, i, sub, want, sc, fail); }
            if (tile_lo) {
                load_low_limbs(j0, rep);
                if (work) { add_low_limbs(sh, i, sub, want, sc, fail); }
            }
            if (work) {
    #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int jj = sub + 16 * half;
                    float p = 0.0F;
                    if (want[half]) {
                        bool ok       = true;
                        const Score s = sc[half];
                        const float d = score_difference(s, best, ok);
                        if (!ok) { fail = true; }
                        p = lossless::cr_exp(d);
                        const std::uint32_t bits = __float_as_uint(p);
                        const int field = static_cast<int>(bits >> 23), mant = static_cast<int>(bits & 0x7FFFFF);
                        if (p != 0.0F) {
                            // p = M 2^(e - 150) at position e - 63 above 2^-87.
                            bool lost = false;
                            add_shifted(den, field == 0 ? mant : (0x800000 | mant), (field == 0 ? 1 : field) - 63, lost);
                            if (lost) { den_err += p; }
                        }
                    }
                    // p 2^48 = M 2^(e - 102), rounded to nearest below the grid (p < 2^-25).
                    long long P = 0;
                    if (p != 0.0F) {
                        const std::uint32_t bits = __float_as_uint(p);
                        const int field = static_cast<int>(bits >> 23), mant = static_cast<int>(bits & 0x7FFFFF);
                        const long long Mp = field == 0 ? mant : (0x800000 | mant);
                        const int shift = (field == 0 ? 1 : field) - 150 + kPBits;
                        if (shift >= 0) {
                            P = Mp << shift;
                        } else {
                            P = shift > -64 ? ((Mp + (1LL << (-shift - 1))) >> -shift) : 0;
                            p_err += 0x1p-49;
                        }
                    }
                    sh.p[i][jj]  = p;
                    sh.ph[i][jj] = static_cast<int>(P >> 24);
                    sh.pl[i][jj] = static_cast<int>(P & 0xFFFFFF);
                }
            }
            __syncthreads();
            if (work) {
    #pragma unroll
                for (int cg = 0; cg < 16; cg += 4) {
                    long long hi[4] = {0, 0, 0, 0}, lo[4] = {0, 0, 0, 0};
                    for (int jj = 0; jj < kKeys; ++jj) {
                        const long long ph = sh.ph[i][jj], pl = sh.pl[i][jj];
                        if ((ph | pl) == 0) { continue; }
    #pragma unroll
                        for (int cc = 0; cc < 4; ++cc) {
                            const long long v = sh.vg[jj][sub + 16 * (cg + cc)];
                            hi[cc] += ph * v; // |.| < 2^24 2^29 per key: 32 keys below 2^59
                            lo[cc] += pl * v;
                        }
                    }
    #pragma unroll
                    for (int cc = 0; cc < 4; ++cc) {
                        bool unused = false;
                        add_shifted(num[cg + cc], hi[cc], 24, unused);
                        add_shifted(num[cg + cc], lo[cc], 0, unused);
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int o = 8; o > 0; o >>= 1) {
        I128 other;
        other.lo = __shfl_xor_sync(0xFFFFFFFFU, den.lo, o);
        other.hi = __shfl_xor_sync(0xFFFFFFFFU, den.hi, o);
        den      = add_i128(den, other);
        den_err += __shfl_xor_sync(0xFFFFFFFFU, den_err, o);
        p_err += __shfl_xor_sync(0xFFFFFFFFU, p_err, o);
        fail = __shfl_xor_sync(0xFFFFFFFFU, fail ? 1 : 0, o) != 0 || fail;
    }
    if constexpr (M == kSums) {
        if (active) {
            PartialSum& ps = args.psum[(static_cast<std::size_t>(split) * args.columns + c) * args.heads + h];
#pragma unroll
            for (int cc = 0; cc < 16; ++cc) { ps.num[sub + 16 * cc] = num[cc]; }
            if (sub == 0) {
                ps.den     = den;
                ps.p_err   = p_err;
                ps.den_err = den_err;
                ps.flags   = fail ? kFail : 0;
            }
        }
        // Value elements rounded to their grid widen the combine's error bound.
        for (int d = tid; d < kDim; d += kThreads) {
            if (sh.vinexact[d] != 0) { atomicOr(args.vinexact + (static_cast<std::size_t>(b) * args.kv_heads + kvh) * kDim + d, 1); }
        }
        return;
    }
    bool flag = false;
    if (active) {
        std::uint16_t* out = args.out + (static_cast<std::size_t>(c) * args.heads + h) * kDim + sub;
        if (nan) {
#pragma unroll
            for (int cc = 0; cc < 16; ++cc) { out[16 * cc] = kNaN; }
        } else if (fail) {
            flag = true;
        } else {
            const double dsum = lossless::double_i128(den, -87);
#pragma unroll
            for (int cc = 0; cc < 16; ++cc) {
                const int vm = sh.vmax[sub + 16 * cc];
                const double n = lossless::double_i128(num[cc], vm - 203);
                const double o = n / dsum;
                // The num error: p's dropped grid bits times the channel's largest |v| (below 2^(vm - 126)),
                // and half a v grid unit per rounded value element times sum p.
                const double num_err = p_err * ldexp(1.0, vm - 126) +
                                       (sh.vinexact[sub + 16 * cc] != 0 ? (dsum + den_err) * ldexp(1.0, vm - 156) : 0.0);
                // The conversions (2^-51 each), the division (2^-53), and the dropped terms.
                const double err = fabs(o) * 0x1p-48 + (num_err + fabs(o) * den_err) / dsum * 1.01;
                std::uint16_t r;
                if (lossless::certain_bf16(o, err, r)) {
                    out[16 * cc] = r;
                } else {
                    flag = true;
                }
            }
        }
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) { flag = __shfl_xor_sync(0xFFFFFFFFU, flag ? 1 : 0, o) != 0 || flag; }
    if (active && sub == 0 && flag) {
        const std::size_t bit = static_cast<std::size_t>(c) * args.heads + h;
        atomicOr(args.flagged + bit / 32, 1U << (bit % 32));
    }
}

// The key ranges' partial sums of query row (column blockIdx.x, head blockIdx.y) added (exactly) and
// rounded; thread d: channel d.
__global__ __launch_bounds__(kDim) void combine_kernel(const Args a) {
    const int c = static_cast<int>(blockIdx.x), h = static_cast<int>(blockIdx.y), d = static_cast<int>(threadIdx.x);
    const int b = c / a.width, kvh = h / (a.heads / a.kv_heads);
    I128 num{0ULL, 0LL}, den{0ULL, 0LL};
    double p_err  = 0.0;
    float den_err = 0.0F;
    bool nan = false, fail = false;
    for (int s = 0; s < a.splits; ++s) {
        const std::size_t e = (static_cast<std::size_t>(s) * a.columns + c) * a.heads + h;
        const int mf = a.pmax[e].flags, sf = a.psum[e].flags;
        nan  = nan || (mf & kNan) != 0;
        fail = fail || (mf & kFail) != 0 || (sf & kFail) != 0;
        num  = add_i128(num, a.psum[e].num[d]);
        den  = add_i128(den, a.psum[e].den);
        p_err += a.psum[e].p_err;
        den_err += a.psum[e].den_err;
    }
    std::uint16_t* out = a.out + (static_cast<std::size_t>(c) * a.heads + h) * kDim;
    bool flag = false;
    if (nan) {
        out[d] = kNaN;
    } else if (fail) {
        flag = true;
    } else {
        const std::size_t channel = (static_cast<std::size_t>(b) * a.kv_heads + kvh) * kDim + d;
        const int vm      = max(a.vmax[channel], 1);
        const double dsum = lossless::double_i128(den, -87);
        const double n    = lossless::double_i128(num, vm - 203);
        const double o    = n / dsum;
        const double num_err =
            p_err * ldexp(1.0, vm - 126) + (a.vinexact[channel] != 0 ? (dsum + den_err) * ldexp(1.0, vm - 156) : 0.0);
        const double err = fabs(o) * 0x1p-48 + (num_err + fabs(o) * den_err) / dsum * 1.01;
        std::uint16_t r;
        if (lossless::certain_bf16(o, err, r)) {
            out[d] = r;
        } else {
            flag = true;
        }
    }
    if (__syncthreads_or(flag ? 1 : 0) != 0 && d == 0) {
        const std::size_t bit = static_cast<std::size_t>(c) * a.heads + h;
        atomicOr(a.flagged + bit / 32, 1U << (bit % 32));
    }
}

// ------------------------------------------------------------------------- exact fallback

constexpr int kScoreWords = 18; // scores from 2^-272: positions e_q + e_k in [2, 508], plus 24 bits
constexpr int kScoreBase  = -272;
constexpr int kSumWords   = 15; // p and p v from 2^-300: positions in [18, 397], plus 46 bits
constexpr int kSumBase    = -300;
// The exact fallback runs one flagged (column, head) query per warp at a time, each warp holding a
// key_capacity row of probabilities: as many warps as kFallbackBytes of rows allow (whole blocks of
// eight), at most kFallbackMaxWarps: 1,024 for a 14,336-key scoring window, 64 at 262,144 keys.
constexpr std::size_t kFallbackBytes = std::size_t{64} << 20;
constexpr int kFallbackMaxWarps      = 1024;

inline int fallback_warps(int key_capacity) {
    const std::size_t row = std::max<std::size_t>(static_cast<std::size_t>(key_capacity) * sizeof(float), 1);
    const auto fit        = static_cast<int>(std::min<std::size_t>(kFallbackMaxWarps, kFallbackBytes / row));
    return std::max(8, fit / 8 * 8);
}

template <int W>
__device__ __forceinline__ int top_bit(const std::uint32_t (&m)[W]) {
#pragma unroll
    for (int i = W - 1; i >= 0; --i) {
        if (m[i] != 0) { return 32 * i + 31 - __clz(m[i]); }
    }
    return -1;
}

template <int W>
__device__ __forceinline__ int compare_mag(const std::uint32_t (&a)[W], const std::uint32_t (&b)[W]) {
#pragma unroll
    for (int i = W - 1; i >= 0; --i) {
        if (a[i] != b[i]) { return a[i] > b[i] ? 1 : -1; }
    }
    return 0;
}

__device__ void exact_score(const Args& a, int seq, int c, int h, int kvh, int j, Wide<kScoreWords>& acc) {
    acc.clear();
    const std::uint16_t* qr = a.q + (static_cast<std::size_t>(c) * a.heads + h) * kDim;
    const std::uint16_t* kr = a.k + row_offset(a, seq, j, kvh);
    for (int d = 0; d < kDim; ++d) {
        int mq, eq, mk, ek;
        bf16_parts(qr[d], mq, eq);
        bf16_parts(kr[d], mk, ek);
        if (mq == 0 || mk == 0) { continue; }
        acc.add(static_cast<long long>(mq) * mk, eq + ek - kScoreShift - kScoreBase);
    }
}

// Exact sign comparison of two (negative, magnitude) pairs: whether a > b.
__device__ __forceinline__ bool exact_greater(bool na, const std::uint32_t (&a)[kScoreWords], bool nb,
                                              const std::uint32_t (&b)[kScoreWords]) {
    if (na != nb) { return nb; }
    const int c = compare_mag(a, b);
    return na ? c < 0 : c > 0;
}

// N / D (magnitudes over the same unit) against BF16 midpoint mid = odd 2^e2: sign of N - mid D.
__device__ int compare_quotient(const std::uint32_t (&n)[kSumWords], const std::uint32_t (&d)[kSumWords], long long odd,
                                int e2) {
    constexpr int W2 = 2 * kSumWords + 8;
    std::uint32_t lhs[W2] = {}, rhs[W2] = {};
    std::uint32_t prod[kSumWords + 1];
    unsigned long long carry = 0;
    for (int i = 0; i < kSumWords; ++i) {
        const unsigned long long v = static_cast<unsigned long long>(d[i]) * static_cast<unsigned long long>(odd) + carry;
        prod[i]                    = static_cast<std::uint32_t>(v);
        carry                      = v >> 32;
    }
    prod[kSumWords] = static_cast<std::uint32_t>(carry);
    auto place = [](std::uint32_t (&dst)[W2], const std::uint32_t* src, int count, int bits) {
        const int w = bits / 32, b = bits % 32;
        for (int i = 0; i < count; ++i) {
            const unsigned long long v = static_cast<unsigned long long>(src[i]) << b;
            if (i + w >= 0 && i + w < W2) { dst[i + w] |= static_cast<std::uint32_t>(v); }
            if (i + w + 1 >= 0 && i + w + 1 < W2) { dst[i + w + 1] |= static_cast<std::uint32_t>(v >> 32); }
        }
    };
    // N vs odd D 2^e2 with e2 >= -160 (the midpoints of BF16 values): offset both by 160 bits.
    constexpr int kOffset = 160;
    place(lhs, n, kSumWords, kOffset);
    place(rhs, prod, kSumWords + 1, kOffset + e2);
    for (int i = W2 - 1; i >= 0; --i) {
        if (lhs[i] != rhs[i]) { return lhs[i] > rhs[i] ? 1 : -1; }
    }
    return 0;
}

// The BF16 magnitude bits b's midpoint with b + dir as odd 2^e2.
__device__ void midpoint(std::uint16_t b, int dir, long long& odd, int& e2) {
    const double lo = __bfloat162float(__ushort_as_bfloat16(b));
    const double hi = __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(b + dir)));
    const double mid = (lo + hi) / 2; // 9 significant bits: exact
    int e;
    const double fr = frexp(mid, &e); // mid = fr 2^e, fr in [0.5, 1)
    odd = static_cast<long long>(ldexp(fr, 10));
    e2  = e - 10;
    while ((odd & 1) == 0 && odd != 0) {
        odd >>= 1;
        ++e2;
    }
}

struct FallbackArgs {
    Args a;
    float* p_store; // warps x key_capacity
    int warps;
};

__global__ void attention_fallback_kernel(const FallbackArgs f) {
    const Args& a  = f.a;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int total = a.columns * a.heads;
    float* pstore   = f.p_store + static_cast<std::size_t>(warp) * a.key_capacity;
    for (int query = warp; query < total; query += f.warps) {
        if (((a.flagged[query / 32] >> (query % 32)) & 1U) == 0) { continue; }
        const int t = query / a.heads, h = query % a.heads, kvh = h / (a.heads / a.kv_heads);
        const int seq = t / a.width, last = position(a, t);
        // The maximum: per lane over its keys, then across the warp.
        std::uint32_t best[kScoreWords] = {};
        bool best_neg = false, have = false;
        for (int j = lane; j <= last; j += 32) {
            if (!visible(a, t, j)) { continue; }
            Wide<kScoreWords> acc;
            exact_score(a, seq, t, h, kvh, j, acc);
            std::uint32_t m[kScoreWords];
            const bool neg = acc.finish(m);
            if (!have || exact_greater(neg, m, best_neg, best)) {
                for (int w = 0; w < kScoreWords; ++w) { best[w] = m[w]; }
                best_neg = neg;
                have     = true;
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            std::uint32_t other[kScoreWords];
            for (int w = 0; w < kScoreWords; ++w) { other[w] = __shfl_xor_sync(0xFFFFFFFFU, best[w], o); }
            const bool other_neg  = __shfl_xor_sync(0xFFFFFFFFU, best_neg ? 1 : 0, o) != 0;
            const bool other_have = __shfl_xor_sync(0xFFFFFFFFU, have ? 1 : 0, o) != 0;
            if (other_have && (!have || exact_greater(other_neg, other, best_neg, best))) {
                for (int w = 0; w < kScoreWords; ++w) { best[w] = other[w]; }
                best_neg = other_neg;
                have     = true;
            }
        }
        // p_j = cr_exp(fp32(s_j - m)), stored; sum p exact (p = M 2^(e - 150) at e - 150 - kSumBase).
        Wide<kSumWords> den;
        den.clear();
        for (int j = lane; j <= last; j += 32) {
            if (!visible(a, t, j)) {
                pstore[j] = 0.0F;
                continue;
            }
            Wide<kScoreWords> acc;
            exact_score(a, seq, t, h, kvh, j, acc);
            for (int w = 0; w + 1 < kScoreWords; ++w) { // subtract the maximum in 16-bit pieces
                const long long lo = best[w] & 0xFFFFU, hi = best[w] >> 16;
                if (lo != 0) { acc.add(best_neg ? lo : -lo, 32 * w); }
                if (hi != 0) { acc.add(best_neg ? hi : -hi, 32 * w + 16); }
            }
            std::uint32_t m[kScoreWords];
            const bool neg = acc.finish(m);
            const float d  = __uint_as_float(round_magnitude(m, kScoreBase, neg, false));
            const float p  = lossless::cr_exp(d);
            pstore[j]      = p;
            if (p != 0.0F) {
                const std::uint32_t bits = __float_as_uint(p);
                const int pf = static_cast<int>(bits >> 23), pm = static_cast<int>(bits & 0x7FFFFF);
                const long long M = pf == 0 ? pm : (0x800000 | pm);
                const int pos = (pf == 0 ? 1 : pf) - 150 - kSumBase;
                den.add(M & 0xFFFF, pos);
                den.add(M >> 16, pos + 16);
            }
        }
        __syncwarp();
        den.warp_sum();
        std::uint32_t dmag[kSumWords];
        (void)den.finish(dmag);
        for (int c = 0; c < kDim; ++c) {
            Wide<kSumWords> num;
            num.clear();
            for (int j = lane; j <= last; j += 32) {
                const float p = pstore[j];
                if (p == 0.0F) { continue; }
                int mv, ev;
                bf16_parts(a.v[row_offset(a, seq, j, kvh) + c], mv, ev);
                if (mv == 0) { continue; }
                const std::uint32_t bits = __float_as_uint(p);
                const int pf = static_cast<int>(bits >> 23), pm = static_cast<int>(bits & 0x7FFFFF);
                const long long Mp = pf == 0 ? pm : (0x800000 | pm);
                const int pos = (pf == 0 ? 1 : pf) + ev - 284 - kSumBase;
                num.add((Mp & 0xFFFF) * mv, pos);
                num.add((Mp >> 16) * mv, pos + 16);
            }
            num.warp_sum();
            if (lane == 0) {
                std::uint32_t nmag[kSumWords];
                const bool nneg = num.finish(nmag);
                std::uint16_t r = 0;
                if (top_bit(nmag) >= 0) {
                    // A candidate from the quotient's leading words, then exact steps to the right one.
                    auto lead = [](const std::uint32_t (&m)[kSumWords]) {
                        const int tb = top_bit(m), w = tb / 32;
                        double v = 0.0;
                        for (int k = w; k >= 0 && k >= w - 2; --k) { v += ldexp(static_cast<double>(m[k]), 32 * k); }
                        return v;
                    };
                    r = lossless::bf16_of_double(lead(nmag) / lead(dmag));
                    for (int step = 0; step < 4; ++step) {
                        long long odd;
                        int e2;
                        midpoint(r, 1, odd, e2);
                        const int up = compare_quotient(nmag, dmag, odd, e2);
                        if (up > 0) {
                            ++r;
                            continue;
                        }
                        if (up == 0) {
                            if ((r & 1) != 0) { ++r; }
                            break;
                        }
                        if ((r & 0x7FFF) == 0) { break; }
                        midpoint(r, -1, odd, e2);
                        const int down = compare_quotient(nmag, dmag, odd, e2);
                        if (down < 0) {
                            --r;
                            continue;
                        }
                        if (down == 0 && (r & 1) != 0) { --r; }
                        break;
                    }
                }
                if (nneg) { r |= 0x8000U; }
                a.out[(static_cast<std::size_t>(t) * a.heads + h) * kDim + c] = r;
            }
        }
        __syncwarp();
    }
}

// ------------------------------------------------------------------------- codec rows (the design reference)

constexpr int kCodecInt8 = static_cast<int>(LosslessKvCodec::Int8G64);
constexpr int kCodecFp8  = static_cast<int>(LosslessKvCodec::Fp8Row);
constexpr int kCodecFp4  = static_cast<int>(LosslessKvCodec::Nvfp4G16);
constexpr int kCodecQ4   = static_cast<int>(LosslessKvCodec::Q4Lloyd);
// A two-limb codec key row's low limb holds at most this many bits (below 2^29, as every grid integer).
constexpr int kLowLimbBits = 29;

// Bytes of a row's codes and of its scales.
__host__ __device__ constexpr int codec_code_bytes(int codec) {
    return codec == kCodecInt8 || codec == kCodecFp8 ? 256 : (codec == kCodecFp4 || codec == kCodecQ4 ? 128 : 64);
}
__host__ __device__ constexpr int codec_scale_bytes(int codec) {
    return codec == kCodecInt8 ? 8 : (codec == kCodecFp4 ? 16 : 2);
}

// A scale's value m 2^f (m = 0 for a zero scale); false for a non-finite one.
__device__ __forceinline__ bool fp16_scale(std::uint16_t bits, int& m, int& f) {
    const int e = (bits >> 10) & 0x1F, man = bits & 0x3FF;
    if (e == 0x1F) { return false; }
    m = e == 0 ? man : (1024 | man);
    f = e == 0 ? -24 : e - 25;
    if ((bits & 0x8000U) != 0) { m = -m; }
    return true;
}
__device__ __forceinline__ bool e4m3_scale(std::uint8_t bits, int& m, int& f) {
    if ((bits & 0x7F) == 0x7F) { return false; }
    const int e = (bits >> 3) & 0xF, man = bits & 7;
    m = e == 0 ? man : (8 | man);
    f = e == 0 ? -9 : e - 10;
    if ((bits & 0x80U) != 0) { m = -m; }
    return true;
}

// Element i's code value C: the element is C 2^code_exp times its group's scale (code_exp -9 for FP8's E4M3,
// -1 for NVFP4's E2M1, else 0); `bad` for a non-finite code.
__device__ __forceinline__ int codec_code(int codec, const std::uint8_t* codes, int i, bool& bad) {
    if (codec == kCodecInt8) { return static_cast<std::int8_t>(codes[i]); }
    if (codec == kCodecFp8) {
        const int b = codes[i];
        if ((b & 0x7F) == 0x7F) {
            bad = true;
            return 0;
        }
        const int e = (b >> 3) & 0xF, man = b & 7;
        const int mag = e == 0 ? man : (8 | man) << (e - 1);
        return (b & 0x80) != 0 ? -mag : mag;
    }
    if (codec == kCodecFp4 || codec == kCodecQ4) {
        const int n = (codes[i >> 1] >> (4 * (i & 1))) & 0xF;
        if (codec == kCodecQ4) { return n >= 8 ? kv_cache_q4_level(n ^ 8) : -kv_cache_q4_level(n ^ 7); }
        const int a = n & 7, mag = a < 4 ? a : ((a & 1) != 0 ? 3 : 2) << ((a >> 1) - 1); // twice E2M1
        return (n & 8) != 0 ? -mag : mag;
    }
    const std::uint32_t word = reinterpret_cast<const std::uint16_t*>(codes)[i >> 3];
    const int mag            = g_kv_cache_vq2_codebook[8 * (word & kKVCacheVq2PatternMask) + (i & 7)];
    return ((kv_cache_vq2_sign_mask(word) >> (i & 7)) & 1U) != 0 ? -mag : mag;
}

__device__ __forceinline__ I128 i128_of(long long v) { return {static_cast<unsigned long long>(v), v < 0 ? -1LL : 0LL}; }
__device__ __forceinline__ I128 neg_i128(I128 a) {
    const unsigned long long lo = ~a.lo + 1ULL;
    return {lo, ~a.hi + (lo == 0 ? 1LL : 0LL)};
}
__device__ __forceinline__ I128 sub_i128(I128 a, I128 b) { return add_i128(a, neg_i128(b)); }
// v 2^s exactly (0 <= s < 64, |v| 2^s < 2^127).
__device__ __forceinline__ I128 shl_i128(long long v, int s) {
    if (s == 0) { return i128_of(v); }
    return {static_cast<unsigned long long>(v) << s, v >> (64 - s)};
}
// The top bit of |v| (-1 for zero).
__device__ __forceinline__ int i128_top_bit(const I128& v) {
    std::uint32_t mag[4];
    (void)lossless::magnitude_i128(v, mag);
#pragma unroll
    for (int i = 3; i >= 0; --i) {
        if (mag[i] != 0) { return 32 * i + 31 - __clz(mag[i]); }
    }
    return -1;
}

// H_256 (Sylvester's order, unnormalized) applied exactly to a row whose element lane + 32 r is X[r].
__device__ __forceinline__ void hadamard_i128(I128 (&X)[8], int lane) {
#pragma unroll
    for (int stride = 1; stride <= 16; stride <<= 1) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            I128 peer;
            peer.lo = __shfl_xor_sync(0xFFFFFFFFU, X[r].lo, stride);
            peer.hi = __shfl_xor_sync(0xFFFFFFFFU, X[r].hi, stride);
            X[r]    = (lane & stride) == 0 ? add_i128(X[r], peer) : sub_i128(peer, X[r]);
        }
    }
#pragma unroll
    for (int span = 1; span < 8; span <<= 1) {
#pragma unroll
        for (int base = 0; base < 8; base += 2 * span) {
#pragma unroll
            for (int off = 0; off < span; ++off) {
                const I128 lo = X[base + off], hi = X[base + off + span];
                X[base + off]        = add_i128(lo, hi);
                X[base + off + span] = sub_i128(lo, hi);
            }
        }
    }
}

// The exact values of row `row` of a codec plane, one warp (lane: elements lane + 32 r): x = X[r] 2^unit. Each
// element's code times its group scale on the row's smallest scale exponent (below 2^58), then for a rotated
// plane x = R y = H y / 16 (below 2^66). False for a row holding a non-finite code or scale (the encoders store
// none).
__device__ bool decode_codec_row(const CodecPlane& p, std::size_t row, int lane, I128 (&X)[8], int& unit) {
    const int codec            = p.codec;
    const std::uint8_t* codes  = p.codes + row * codec_code_bytes(codec);
    const std::uint8_t* scales = p.scales + row * codec_scale_bytes(codec);
    const int groups           = codec == kCodecInt8 ? 4 : (codec == kCodecFp4 ? 16 : 1);
    int sm[16], sf[16];
    bool bad  = false;
    int fmin  = INT_MAX;
    for (int g = 0; g < groups; ++g) {
        const bool ok = codec == kCodecFp4 ? e4m3_scale(scales[g], sm[g], sf[g])
                                           : fp16_scale(reinterpret_cast<const std::uint16_t*>(scales)[g], sm[g], sf[g]);
        if (!ok) {
            bad   = true;
            sm[g] = 0;
            sf[g] = 0;
        }
        if (sm[g] != 0) { fmin = min(fmin, sf[g]); }
    }
    if (fmin == INT_MAX) { fmin = 0; } // every scale zero: a zero row
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int i = lane + 32 * r;
        const int g = codec == kCodecInt8 ? i >> 6 : (codec == kCodecFp4 ? i >> 4 : 0);
        const long long cm = static_cast<long long>(codec_code(codec, codes, i, bad)) * sm[g]; // below 2^29
        X[r] = shl_i128(cm, sm[g] != 0 ? sf[g] - fmin : 0);
    }
    const int code_exp = codec == kCodecFp8 ? -9 : (codec == kCodecFp4 ? -1 : 0);
    unit               = fmin + code_exp;
    if (p.rotated != 0) {
        hadamard_i128(X, lane);
        unit -= 4;
    }
    return !__any_sync(0xFFFFFFFFU, bad);
}

// One warp per (rep, row) of the call's codec planes. Mode 0: the key row's grid integers M 2^s with the row's
// top below 2^29 (a row whose exact bits a right shift would lose is off its grid: its pairs take the exact
// fallback), its one block exponent e_b (x = X_grid 2^(e_b - 155), so e_b is the BF16 exponent field of its
// largest element) into kblk and kmax (kNonFinite when its k or v holds a non-finite code), and the value row's
// element exponents into vmax. Mode 1 (after mode 0): the value row's grid integers on each channel's grid
// (v 2^(155 - vmax_c), below 2^29), rounded to nearest even below it (vround).
__global__ void codec_pack_kernel(const Args a, int mode, std::int32_t* kpack, std::int32_t* kpack_lo,
                                  std::int16_t* kblk, std::uint8_t* koff, std::int32_t* vpack, std::uint32_t* vround) {
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const long long rows  = static_cast<long long>(a.columns) * a.kv_heads;
    const long long items = rows * (a.reps > 1 ? 2 : 1);
    const long long warps = static_cast<long long>(gridDim.x) * (blockDim.x / 32);
    for (long long it = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32; it < items; it += warps) {
        const int rep          = static_cast<int>(it / rows);
        const long long rr     = it % rows;
        const int kvh          = static_cast<int>(rr % a.kv_heads), j = static_cast<int>(rr / a.kv_heads);
        const std::size_t prow = codec_row(a.kv_heads, j, kvh);
        const std::size_t out  = static_cast<std::size_t>(rep) * codec_rows(a.columns, a.kv_heads) + prow;
        I128 X[8];
        int unit = 0;
        if (mode == 0) {
            const bool kok = decode_codec_row(a.kplane[rep], prow, lane, X, unit);
            int top        = -1;
#pragma unroll
            for (int r = 0; r < 8; ++r) { top = max(top, i128_top_bit(X[r])); }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) { top = max(top, __shfl_xor_sync(0xFFFFFFFFU, top, o)); }
            const int s = top < 0 ? 0 : 28 - top; // |X| 2^s < 2^29
            bool lost   = false;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                std::int32_t x;
                if (s >= 0) {
                    x = static_cast<std::int32_t>(static_cast<long long>(X[r].lo) << s);
                } else {
                    // An arithmetic shift right by -s (at most 38: |X| < 2^67), exact only when the dropped
                    // bits are zero (two's complement keeps a magnitude's trailing zeros).
                    const int down = -s;
                    if ((X[r].lo & ((1ULL << down) - 1)) != 0) { lost = true; }
                    x = static_cast<std::int32_t>(static_cast<long long>(
                        (X[r].lo >> down) | (static_cast<unsigned long long>(X[r].hi) << (64 - down))));
                }
                kpack[out * kDim + lane + 32 * r] = x;
            }
            lost         = __any_sync(0xFFFFFFFFU, lost);
            // The bits the shift dropped become the low limb when they fit the grid (down <= 29): x = hi 2^down
            // + lo, lo the row's X mod 2^down (two's complement keeps it in [0, 2^down)).
            const int down = lost && -s <= kLowLimbBits ? -s : 0;
            if (down > 0) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    kpack_lo[out * kDim + lane + 32 * r] = static_cast<std::int32_t>(X[r].lo & ((1ULL << down) - 1));
                }
            }
            const int eb = unit + 155 - s;
            const bool vok = decode_codec_row(a.vplane[rep], prow, lane, X, unit);
            if (lane < kBlocks) { kblk[out * kBlocks + lane] = static_cast<std::int16_t>(eb); }
            if (lane == 0) {
                koff[out] = static_cast<std::uint8_t>(down > 0 ? 1 + down : (lost ? 1 : 0));
                a.kmax[(static_cast<std::size_t>(rep) * a.key_capacity + j) * a.kv_heads + kvh] =
                    static_cast<std::int16_t>(kok && vok ? eb : kNonFinite);
            }
            if (vok) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int t = i128_top_bit(X[r]);
                    if (t >= 0) { atomicMax(a.vmax + static_cast<std::size_t>(kvh) * kDim + lane + 32 * r, t + unit + 127); }
                }
            }
        } else {
            const bool vok = decode_codec_row(a.vplane[rep], prow, lane, X, unit);
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int d    = lane + 32 * r;
                const int vm   = max(a.vmax[static_cast<std::size_t>(kvh) * kDim + d], 1);
                const int sh   = unit + 155 - vm;
                std::int32_t x = 0;
                bool rounded   = false;
                if (vok) {
                    if (sh >= 0) {
                        x = static_cast<std::int32_t>(static_cast<long long>(X[r].lo) << sh);
                    } else {
                        // Round to nearest even below the grid: |X| 2^sh < 2^29.
                        const int down = -sh;
                        std::uint32_t mag[4];
                        const bool neg = lossless::magnitude_i128(X[r], mag);
                        unsigned long long lo = (static_cast<unsigned long long>(mag[1]) << 32) | mag[0];
                        unsigned long long hi = (static_cast<unsigned long long>(mag[3]) << 32) | mag[2];
                        unsigned long long q, rem_hi_bit, rem_low;
                        if (down >= 128) {
                            q = 0;
                            rem_hi_bit = 0;
                            rem_low    = lo | hi;
                        } else if (down >= 64) {
                            const int dd = down - 64;
                            q          = dd == 0 ? hi : hi >> dd;
                            rem_hi_bit = dd == 0 ? (lo >> 63) : ((hi >> (dd - 1)) & 1ULL);
                            rem_low    = dd == 0 ? (lo & ~(1ULL << 63)) : ((hi & ((1ULL << (dd - 1)) - 1)) | lo);
                        } else {
                            q          = (lo >> down) | (down == 0 ? 0ULL : hi << (64 - down));
                            rem_hi_bit = (lo >> (down - 1)) & 1ULL;
                            rem_low    = lo & ((1ULL << (down - 1)) - 1);
                        }
                        rounded = rem_hi_bit != 0 || rem_low != 0;
                        if (rem_hi_bit != 0 && (rem_low != 0 || (q & 1ULL) != 0)) { ++q; }
                        x = neg ? -static_cast<std::int32_t>(q) : static_cast<std::int32_t>(q);
                    }
                }
                vpack[out * kDim + d] = x;
                const unsigned mask   = __ballot_sync(0xFFFFFFFFU, rounded);
                if (lane == 0) { vround[out * (kDim / 32) + r] = mask; }
            }
        }
    }
}

// The codec exact fallback: scores from 2^-176 (q 2^-133 at least, k 2^-37 at least, the 2^-4 scale) to beyond
// 2^330 (|q| < 2^129, |X| < 2^67, units to 2^5, 256 terms); p and p v from 2^-188 (p 2^-149, v 2^-37) to beyond
// 2^92 (|v| < 2^72, 2^20 keys).
constexpr int kCodecScoreWords = 16;
constexpr int kCodecScoreBase  = -176;
constexpr int kCodecSumWords   = 12;
constexpr int kCodecSumBase    = -188;

// compare_quotient for W-word magnitudes.
template <int W>
__device__ int compare_quotient_w(const std::uint32_t (&n)[W], const std::uint32_t (&d)[W], long long odd, int e2) {
    constexpr int W2 = 2 * W + 8;
    std::uint32_t lhs[W2] = {}, rhs[W2] = {};
    std::uint32_t prod[W + 1];
    unsigned long long carry = 0;
    for (int i = 0; i < W; ++i) {
        const unsigned long long v = static_cast<unsigned long long>(d[i]) * static_cast<unsigned long long>(odd) + carry;
        prod[i]                    = static_cast<std::uint32_t>(v);
        carry                      = v >> 32;
    }
    prod[W] = static_cast<std::uint32_t>(carry);
    auto place = [](std::uint32_t (&dst)[W2], const std::uint32_t* src, int count, int bits) {
        const int w = bits / 32, b = bits % 32;
        for (int i = 0; i < count; ++i) {
            const unsigned long long v = static_cast<unsigned long long>(src[i]) << b;
            if (i + w >= 0 && i + w < W2) { dst[i + w] |= static_cast<std::uint32_t>(v); }
            if (i + w + 1 >= 0 && i + w + 1 < W2) { dst[i + w + 1] |= static_cast<std::uint32_t>(v >> 32); }
        }
    };
    constexpr int kOffset = 160;
    place(lhs, n, W, kOffset);
    place(rhs, prod, W + 1, kOffset + e2);
    for (int i = W2 - 1; i >= 0; --i) {
        if (lhs[i] != rhs[i]) { return lhs[i] > rhs[i] ? 1 : -1; }
    }
    return 0;
}

// The BF16 rounding of N / D (magnitudes on one unit, D nonzero), by the quotient's leading words and exact
// steps across the neighbouring midpoints (ties to even).
template <int W>
__device__ std::uint16_t quotient_bf16(const std::uint32_t (&n)[W], const std::uint32_t (&d)[W]) {
    if (top_bit(n) < 0) { return 0; }
    auto lead = [](const std::uint32_t (&m)[W]) {
        const int tb = top_bit(m), w = tb / 32;
        double v = 0.0;
        for (int k = w; k >= 0 && k >= w - 2; --k) { v += ldexp(static_cast<double>(m[k]), 32 * k); }
        return v;
    };
    std::uint16_t r = lossless::bf16_of_double(lead(n) / lead(d));
    for (int step = 0; step < 4; ++step) {
        long long odd;
        int e2;
        midpoint(r, 1, odd, e2);
        const int up = compare_quotient_w(n, d, odd, e2);
        if (up > 0) {
            ++r;
            continue;
        }
        if (up == 0) {
            if ((r & 1) != 0) { ++r; }
            break;
        }
        if ((r & 0x7FFF) == 0) { break; }
        midpoint(r, -1, odd, e2);
        const int down = compare_quotient_w(n, d, odd, e2);
        if (down < 0) {
            --r;
            continue;
        }
        if (down == 0 && (r & 1) != 0) { --r; }
        break;
    }
    return r;
}

// The exact score of query (column t, head h; each lane's eight elements qm 2^(qe - 134)) against key j of its
// pair's representation, summed over the warp into every lane's acc. False for a non-finite key row.
__device__ bool codec_exact_score(const Args& a, int t, int j, int kvh, const int (&qm)[8], const int (&qe)[8], int lane,
                                  Wide<kCodecScoreWords>& acc) {
    I128 X[8];
    int unit      = 0;
    const bool ok = decode_codec_row(a.kplane[pair_rep(a, t, j)], codec_row(a.kv_heads, j, kvh), lane, X, unit);
    acc.clear();
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        if (qm[r] == 0) { continue; }
        std::uint32_t mag[4];
        const bool neg = lossless::magnitude_i128(X[r], mag);
        const int sign = neg ? -1 : 1;
#pragma unroll
        for (int c = 0; c < 5; ++c) { // |X| < 2^67: five 16-bit pieces
            const long long piece = (mag[c / 2] >> (16 * (c % 2))) & 0xFFFF;
            if (piece != 0) { acc.add(sign * qm[r] * piece, qe[r] - 134 + unit - 4 + 16 * c - kCodecScoreBase); }
        }
    }
    acc.warp_sum();
    return ok;
}

// One warp per flagged query of a codec call: lossless_attention's exact fallback over the codec rows, each
// key row decoded exactly once per pass.
__global__ void codec_fallback_kernel(const Args a, int warps) {
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int total = a.columns * a.heads;
    for (int query = warp; query < total; query += warps) {
        if (((a.flagged[query / 32] >> (query % 32)) & 1U) == 0) { continue; }
        const int t = query / a.heads, h = query % a.heads, kvh = h / (a.heads / a.kv_heads);
        const int last = position(a, t);
        int qm[8], qe[8];
        bool nan = false;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const std::uint16_t bits = a.q[(static_cast<std::size_t>(t) * a.heads + h) * kDim + lane + 32 * r];
            if (bf16_nonfinite(bits)) { nan = true; }
            bf16_parts(bits, qm[r], qe[r]);
        }
        nan = __any_sync(0xFFFFFFFFU, nan);
        // The maximum.
        std::uint32_t best[kCodecScoreWords] = {};
        bool best_neg = false, have = false;
        for (int j = 0; j <= last && !nan; ++j) {
            if (!visible(a, t, j)) { continue; }
            Wide<kCodecScoreWords> acc;
            if (!codec_exact_score(a, t, j, kvh, qm, qe, lane, acc)) {
                nan = true;
                break;
            }
            std::uint32_t m[kCodecScoreWords];
            const bool neg = acc.finish(m);
            bool greater   = !have;
            if (have) {
                const int cmp = compare_mag(m, best);
                greater       = neg != best_neg ? best_neg : (neg ? cmp < 0 : cmp > 0);
            }
            if (greater) {
                for (int w = 0; w < kCodecScoreWords; ++w) { best[w] = m[w]; }
                best_neg = neg;
                have     = true;
            }
        }
        // p_j = cr_exp(fp32(s_j - m)); sum p (lane 0) and each lane's channels' sum p v, exact.
        Wide<kCodecSumWords> den, num[8];
        den.clear();
#pragma unroll
        for (int r = 0; r < 8; ++r) { num[r].clear(); }
        for (int j = 0; j <= last && !nan; ++j) {
            if (!visible(a, t, j)) { continue; }
            Wide<kCodecScoreWords> acc;
            (void)codec_exact_score(a, t, j, kvh, qm, qe, lane, acc);
            for (int w = 0; w + 1 < kCodecScoreWords; ++w) { // subtract the maximum in 16-bit pieces
                const long long lo = best[w] & 0xFFFFU, hi = best[w] >> 16;
                if (lo != 0) { acc.add(best_neg ? lo : -lo, 32 * w); }
                if (hi != 0) { acc.add(best_neg ? hi : -hi, 32 * w + 16); }
            }
            std::uint32_t m[kCodecScoreWords];
            const bool neg = acc.finish(m);
            const float p  = lossless::cr_exp(__uint_as_float(round_magnitude(m, kCodecScoreBase, neg, false)));
            if (p == 0.0F) { continue; }
            const std::uint32_t bits = __float_as_uint(p);
            const int pf = static_cast<int>(bits >> 23), pm = static_cast<int>(bits & 0x7FFFFF);
            const long long Mp = pf == 0 ? pm : (0x800000 | pm);
            const int pe       = (pf == 0 ? 1 : pf) - 150; // p = Mp 2^pe
            if (lane == 0) {
                den.add(Mp & 0xFFFF, pe - kCodecSumBase);
                den.add(Mp >> 16, pe + 16 - kCodecSumBase);
            }
            I128 X[8];
            int unit = 0;
            if (!decode_codec_row(a.vplane[pair_rep(a, t, j)], codec_row(a.kv_heads, j, kvh), lane, X, unit)) {
                nan = true;
                break;
            }
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                std::uint32_t mag[4];
                const bool vneg = lossless::magnitude_i128(X[r], mag);
                const int sign  = vneg ? -1 : 1;
#pragma unroll
                for (int c = 0; c < 5; ++c) {
                    const long long piece = (mag[c / 2] >> (16 * (c % 2))) & 0xFFFF;
                    if (piece == 0) { continue; }
                    num[r].add(sign * (Mp & 0xFFF) * piece, pe + unit + 16 * c - kCodecSumBase);
                    num[r].add(sign * (Mp >> 12) * piece, pe + unit + 12 + 16 * c - kCodecSumBase);
                }
            }
        }
        std::uint16_t* out = a.out + (static_cast<std::size_t>(t) * a.heads + h) * kDim;
        if (nan) {
#pragma unroll
            for (int r = 0; r < 8; ++r) { out[lane + 32 * r] = kNaN; }
            continue;
        }
        den.warp_sum();
        std::uint32_t dmag[kCodecSumWords];
        (void)den.finish(dmag);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            std::uint32_t nmag[kCodecSumWords];
            const bool nneg   = num[r].finish(nmag);
            std::uint16_t res = quotient_bf16(nmag, dmag);
            if (nneg) { res |= 0x8000U; } // a negative quotient rounding to zero is -0
            out[lane + 32 * r] = res;
        }
    }
}

// ------------------------------------------------------------------------- cache append

// One thread per 16 bytes of a K or V row: row (t, h) of the call stored bit for bit at position
// positions[t] of its run's table row.
__global__ void kv_append_kernel(const uint4* k, const uint4* v, const std::int32_t* positions,
                                 const std::int32_t* tables, const std::int32_t* table_rows, int table_stride,
                                 int width, int kv_heads, int columns, uint4* k_pages, uint4* v_pages) {
    constexpr int kChunks = kDim * 2 / 16;
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(columns) * kv_heads * kChunks) { return; }
    const int chunk = static_cast<int>(i % kChunks);
    const long long row = i / kChunks;
    const int h = static_cast<int>(row % kv_heads), t = static_cast<int>(row / kv_heads);
    const int pos = positions[t];
    const int page = tables[static_cast<std::size_t>(table_rows[t / width]) * table_stride + pos / kPage];
    const std::size_t dst = ((static_cast<std::size_t>(page) * kv_heads + h) * kPage + pos % kPage) * kChunks + chunk;
    k_pages[dst] = k[row * kChunks + chunk];
    v_pages[dst] = v[row * kChunks + chunk];
}

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

void require_lossless_cache(const PagedKVBatchLayerView& cache, int heads) {
    require(cache.storage == KvCacheStorage::BFloat16 && cache.k_pages.dtype == DType::BF16 &&
                cache.v_pages.dtype == DType::BF16 && cache.k_pages.is_contiguous() && cache.v_pages.is_contiguous() &&
                cache.head_dim == kDim && cache.num_kv_heads >= 1 && cache.num_kv_heads <= 8 &&
                heads % cache.num_kv_heads == 0,
            "the cache is bf16 storage with BF16 K and V pages (the lossless profile), head dim 256");
    require(cache.block_tables.dtype == DType::I32 && cache.block_tables.is_contiguous(),
            "block tables I32 [entries, rows]");
}

struct Geometry {
    int tiles;      // 16-row tiles per sequence and KV head
    int splits;     // key ranges
    int split_keys; // keys per range
    bool split;     // the split route (partial maxima, partial sums, combine), even for one range
};

// A call with fewer CTAs than kTargetCtas takes the split route whatever its keys, so the launches of a
// shape (a captured decode round) do not depend on the key bound.
bool split_route(long long ctas) { return ctas < kTargetCtas; }

// Key ranges until the call has about kTargetCtas CTAs, each of at least kMinSplitKeys keys (an upper
// bound of the range count, non-decreasing in max_keys).
long long split_bound(long long ctas, int max_keys) {
    if (ctas >= kTargetCtas) { return 1; }
    return std::max(1LL, std::min((kTargetCtas + ctas - 1) / ctas,
                                  (static_cast<long long>(max_keys) + kMinSplitKeys - 1) / kMinSplitKeys));
}

Geometry geometry(int heads, int kv_heads, int width, int sequences, int max_keys) {
    const int tiles       = (width * (heads / kv_heads) + kQueries - 1) / kQueries;
    const long long ctas  = static_cast<long long>(sequences) * tiles * kv_heads;
    const long long bound = split_bound(ctas, max_keys);
    int split_keys        = static_cast<int>((max_keys + bound - 1) / bound);
    split_keys            = (split_keys + kKeys - 1) / kKeys * kKeys;
    return {tiles, (max_keys + split_keys - 1) / split_keys, split_keys, split_route(ctas)};
}

// The workspace of a call of these dimensions with at most key_capacity keys per sequence.
std::size_t call_bytes(int heads, int kv_heads, int width, int sequences, int key_capacity) {
    const std::size_t columns = static_cast<std::size_t>(width) * sequences;
    const int tiles           = (width * (heads / kv_heads) + kQueries - 1) / kQueries;
    const long long ctas      = static_cast<long long>(sequences) * tiles * kv_heads;
    const auto splits         = static_cast<std::size_t>(split_bound(ctas, key_capacity));
    std::size_t bytes =
        align256(static_cast<std::size_t>(sequences) * key_capacity * kv_heads * sizeof(std::int16_t)) +
        2 * align256(static_cast<std::size_t>(sequences) * kv_heads * kDim * sizeof(int)) +
        align256((columns * heads + 31) / 32 * 4) +
        align256(static_cast<std::size_t>(fallback_warps(key_capacity)) * key_capacity * sizeof(float));
    // Pass A's partial maxima (one range per row without the split route), and the split route's sums.
    bytes += align256(splits * columns * heads * sizeof(PartialMax));
    if (split_route(ctas)) { bytes += align256(splits * columns * heads * sizeof(PartialSum)); }
    return bytes;
}

template <int M>
void configure() {
    static const bool configured = [] {
        CUDA_CHECK(cudaFuncSetAttribute(attention_kernel<M>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        M == kMaxima ? kPassASmem : static_cast<int>(sizeof(Shared))));
        return true;
    }();
    (void)configured;
}

// Pass A over every range, then pass B: into partial sums and their combine (the split route), or rounded in
// place (one range per CTA).
void launch_passes(const Args& args, const Geometry& g, cudaStream_t stream) {
    constexpr int kSmem = static_cast<int>(sizeof(Shared));
    const dim3 grid(static_cast<unsigned>(args.sequences * g.tiles), static_cast<unsigned>(args.kv_heads),
                    static_cast<unsigned>(g.splits));
    configure<kMaxima>();
    attention_kernel<kMaxima><<<grid, kThreads, kPassASmem, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
    if (g.split) {
        configure<kSums>();
        attention_kernel<kSums><<<grid, kThreads, kSmem, stream>>>(args);
        CUDA_CHECK(cudaGetLastError());
        combine_kernel<<<dim3(static_cast<unsigned>(args.columns), static_cast<unsigned>(args.heads)), kDim, 0, stream>>>(args);
        CUDA_CHECK(cudaGetLastError());
    } else {
        configure<kFinal>();
        attention_kernel<kFinal><<<grid, kThreads, kSmem, stream>>>(args);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Carves the workspace (call_bytes' order) and runs the kernels; args holds everything but the geometry
// and the workspace pointers.
std::size_t pack_bytes(int kv_heads, int columns) {
    const std::size_t rows = static_cast<std::size_t>(columns) * kv_heads;
    return 2 * align256(rows * kDim * sizeof(std::int16_t)) + align256(rows * (kDim / 64) * sizeof(std::int16_t)) +
           align256(rows) + align256(rows * (kDim / 32) * sizeof(std::uint32_t));
}

void run(Args args, int max_keys, void* workspace, cudaStream_t stream, bool pack = false) {
    const Geometry g = geometry(args.heads, args.kv_heads, args.width, args.sequences, max_keys);
    args.tiles       = g.tiles;
    args.splits      = g.splits;
    args.split_keys  = g.split_keys;
    auto* base       = static_cast<std::uint8_t*>(workspace);
    auto take        = [&](std::size_t bytes) {
        void* p = base;
        base += align256(bytes);
        return p;
    };
    const std::size_t channels = static_cast<std::size_t>(args.sequences) * args.kv_heads * kDim * sizeof(int);
    const std::size_t flag_bytes = (static_cast<std::size_t>(args.columns) * args.heads + 31) / 32 * 4;
    args.kmax = static_cast<std::int16_t*>(
        take(static_cast<std::size_t>(args.sequences) * args.key_capacity * args.kv_heads * sizeof(std::int16_t)));
    args.vmax     = static_cast<int*>(take(channels));
    args.vinexact = static_cast<int*>(take(channels));
    args.flagged  = static_cast<std::uint32_t*>(take(flag_bytes));
    const int warps = fallback_warps(args.key_capacity);
    auto* pstore    = static_cast<float*>(take(static_cast<std::size_t>(warps) * args.key_capacity * sizeof(float)));
    const std::size_t partials = static_cast<std::size_t>(g.splits) * args.columns * args.heads;
    args.pmax = static_cast<PartialMax*>(take(partials * sizeof(PartialMax)));
    if (g.split) { args.psum = static_cast<PartialSum*>(take(partials * sizeof(PartialSum))); }
    std::int16_t *kpack = nullptr, *kblk = nullptr, *vpack = nullptr;
    std::uint8_t* koff  = nullptr;
    std::uint32_t* vround = nullptr;
    if (pack) {
        const std::size_t rows = static_cast<std::size_t>(args.columns) * args.kv_heads;
        kpack  = static_cast<std::int16_t*>(take(rows * kDim * sizeof(std::int16_t)));
        vpack  = static_cast<std::int16_t*>(take(rows * kDim * sizeof(std::int16_t)));
        kblk   = static_cast<std::int16_t*>(take(rows * (kDim / 64) * sizeof(std::int16_t)));
        koff   = static_cast<std::uint8_t*>(take(rows));
        vround = static_cast<std::uint32_t*>(take(rows * (kDim / 32) * sizeof(std::uint32_t)));
    }
    // vmax, vinexact and flagged lie together: one memset zeroes them.
    CUDA_CHECK(cudaMemsetAsync(args.vmax, 0,
                               static_cast<std::size_t>(reinterpret_cast<std::uint8_t*>(args.flagged) + flag_bytes -
                                                        reinterpret_cast<std::uint8_t*>(args.vmax)),
                               stream));
    const long long items =
        static_cast<long long>(args.sequences) * args.kv_heads * ((max_keys + kMaximaKeys - 1) / kMaximaKeys);
    maxima_kernel<<<static_cast<int>(std::min<long long>((items + 7) / 8, 4096)), 256, 0, stream>>>(args, max_keys);
    CUDA_CHECK(cudaGetLastError());
    if (pack) {
        const long long rows = static_cast<long long>(args.columns) * args.kv_heads;
        pack_kernel<<<static_cast<int>(std::min<long long>((rows + 7) / 8, 4096)), 256, 0, stream>>>(args, kpack, kblk, koff,
                                                                                                    vpack, vround);
        CUDA_CHECK(cudaGetLastError());
        args.kpack  = kpack;
        args.kblk   = kblk;
        args.koff   = koff;
        args.vpack  = vpack;
        args.vround = vround;
    }
    launch_passes(args, g, stream);
    attention_fallback_kernel<<<warps / 8, 256, 0, stream>>>(FallbackArgs{args, pstore, warps});
    CUDA_CHECK(cudaGetLastError());
}

// The codec form (one window, sequences 1): kmax per representation, vmax, vinexact, flags, the partials of the
// split route, then each representation's packed rows (whole pages).
constexpr int kCodecFallbackWarps = 512;

std::size_t codec_call_bytes(int heads, int kv_heads, int columns, int reps) {
    const int tiles         = (columns * (heads / kv_heads) + kQueries - 1) / kQueries;
    const long long ctas    = static_cast<long long>(tiles) * kv_heads;
    const auto splits       = static_cast<std::size_t>(split_bound(ctas, columns));
    const std::size_t rows  = static_cast<std::size_t>(reps) * codec_rows(columns, kv_heads);
    const std::size_t calls = static_cast<std::size_t>(columns) * heads;
    std::size_t bytes = align256(static_cast<std::size_t>(reps) * columns * kv_heads * sizeof(std::int16_t)) +
                        2 * align256(static_cast<std::size_t>(kv_heads) * kDim * sizeof(int)) + align256((calls + 31) / 32 * 4) +
                        3 * align256(rows * kDim * sizeof(std::int32_t)) + align256(rows * kBlocks * sizeof(std::int16_t)) +
                        align256(rows) + align256(rows * (kDim / 32) * sizeof(std::uint32_t));
    bytes += align256(splits * calls * sizeof(PartialMax));
    if (split_route(ctas)) { bytes += align256(splits * calls * sizeof(PartialSum)); }
    return bytes;
}

void run_codec(Args args, void* workspace, cudaStream_t stream) {
    const int reps   = args.reps > 1 ? 2 : 1;
    const Geometry g = geometry(args.heads, args.kv_heads, args.width, 1, args.columns);
    args.tiles       = g.tiles;
    args.splits      = g.splits;
    args.split_keys  = g.split_keys;
    auto* base       = static_cast<std::uint8_t*>(workspace);
    auto take        = [&](std::size_t bytes) {
        void* p = base;
        base += align256(bytes);
        return p;
    };
    const std::size_t channels   = static_cast<std::size_t>(args.kv_heads) * kDim * sizeof(int);
    const std::size_t flag_bytes = (static_cast<std::size_t>(args.columns) * args.heads + 31) / 32 * 4;
    args.kmax     = static_cast<std::int16_t*>(
        take(static_cast<std::size_t>(reps) * args.columns * args.kv_heads * sizeof(std::int16_t)));
    args.vmax     = static_cast<int*>(take(channels));
    args.vinexact = static_cast<int*>(take(channels));
    args.flagged  = static_cast<std::uint32_t*>(take(flag_bytes));
    const std::size_t partials = static_cast<std::size_t>(g.splits) * args.columns * args.heads;
    args.pmax = static_cast<PartialMax*>(take(partials * sizeof(PartialMax)));
    if (g.split) { args.psum = static_cast<PartialSum*>(take(partials * sizeof(PartialSum))); }
    const std::size_t rows = static_cast<std::size_t>(reps) * codec_rows(args.columns, args.kv_heads);
    auto* kpack  = static_cast<std::int32_t*>(take(rows * kDim * sizeof(std::int32_t)));
    auto* klow   = static_cast<std::int32_t*>(take(rows * kDim * sizeof(std::int32_t)));
    auto* vpack  = static_cast<std::int32_t*>(take(rows * kDim * sizeof(std::int32_t)));
    auto* kblk   = static_cast<std::int16_t*>(take(rows * kBlocks * sizeof(std::int16_t)));
    auto* koff   = static_cast<std::uint8_t*>(take(rows));
    auto* vround = static_cast<std::uint32_t*>(take(rows * (kDim / 32) * sizeof(std::uint32_t)));
    // vmax, vinexact and flagged lie together: one memset zeroes them.
    CUDA_CHECK(cudaMemsetAsync(args.vmax, 0,
                               static_cast<std::size_t>(reinterpret_cast<std::uint8_t*>(args.flagged) + flag_bytes -
                                                        reinterpret_cast<std::uint8_t*>(args.vmax)),
                               stream));
    const long long items = static_cast<long long>(reps) * args.columns * args.kv_heads;
    const int blocks      = static_cast<int>(std::min<long long>((items + 7) / 8, 4096));
    codec_pack_kernel<<<blocks, 256, 0, stream>>>(args, 0, kpack, klow, kblk, koff, vpack, vround);
    CUDA_CHECK(cudaGetLastError());
    codec_pack_kernel<<<blocks, 256, 0, stream>>>(args, 1, kpack, klow, kblk, koff, vpack, vround);
    CUDA_CHECK(cudaGetLastError());
    args.kpack32  = kpack;
    args.kpack_lo = klow;
    args.vpack32 = vpack;
    args.kblk    = kblk;
    args.koff    = koff;
    args.vround  = vround;
    launch_passes(args, g, stream);
    codec_fallback_kernel<<<kCodecFallbackWarps / 8, 256, 0, stream>>>(args, kCodecFallbackWarps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t lossless_attention_workspace_bytes(int query_heads, int kv_heads, int columns) {
    return call_bytes(query_heads, kv_heads, columns, 1, columns) + pack_bytes(kv_heads, columns);
}

void lossless_attention(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out, void* workspace,
                        cudaStream_t stream, const std::uint32_t* select, int select_words, int ratio) {
    require(q.dtype == DType::BF16 && k.dtype == DType::BF16 && v.dtype == DType::BF16 && out.dtype == DType::BF16 &&
                q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && out.is_contiguous() &&
                q.ne[0] == kDim && k.ne[0] == kDim && v.ne[0] == kDim && out.ne[0] == kDim,
            "q, k, v and out must be contiguous BF16 [256, heads, T]");
    const int heads = q.ne[1], kv_heads = k.ne[1], columns = q.ne[2];
    require(k.ne[2] == columns && v.ne[1] == kv_heads && v.ne[2] == columns && out.ne[1] == heads &&
                out.ne[2] == columns && kv_heads <= 8 && heads % kv_heads == 0,
            "q, k, v and out shapes disagree");
    require(select == nullptr || (ratio > 0 && select_words >= (columns / ratio + 31) / 32),
            "a selection needs its ratio and a bit per complete block");
    if (columns == 0) { return; }
    Args args{};
    args.q            = static_cast<const std::uint16_t*>(q.data);
    args.out          = static_cast<std::uint16_t*>(out.data);
    args.k            = static_cast<const std::uint16_t*>(k.data);
    args.v            = static_cast<const std::uint16_t*>(v.data);
    args.width        = columns;
    args.sequences    = 1;
    args.key_capacity = columns;
    args.heads        = heads;
    args.kv_heads     = kv_heads;
    args.columns      = columns;
    args.select       = select;
    args.select_words = select_words;
    args.ratio        = ratio;
    run(args, columns, workspace, stream, true);
}

std::size_t lossless_paged_attention_workspace_bytes(int query_heads, int kv_heads, int max_width, int sequences,
                                                     int key_capacity) {
    // The split route (partials) needs sequences * tiles * kv_heads < kTargetCtas, so a run narrower than
    // kTargetCtas * 16 / query_heads columns; wider runs differ only in their flag bits.
    const int split_widths = std::min(max_width, kTargetCtas * kQueries / query_heads + kQueries);
    std::size_t bytes      = call_bytes(query_heads, kv_heads, max_width, sequences, key_capacity);
    for (int b = 1; b <= sequences; ++b) {
        for (int w = 1; w <= split_widths; ++w) {
            bytes = std::max(bytes, call_bytes(query_heads, kv_heads, w, b, key_capacity));
        }
    }
    return bytes;
}

void lossless_paged_attention(const Tensor& q, const Tensor& positions, int sequences, const Tensor& table_rows,
                              const PagedKVBatchLayerView& cache, int max_keys, int key_capacity, Tensor& out,
                              void* workspace, cudaStream_t stream, const std::uint32_t* select, int select_words,
                              int ratio) {
    require(q.dtype == DType::BF16 && out.dtype == DType::BF16 && q.is_contiguous() && out.is_contiguous() &&
                q.ne[0] == kDim && out.ne[0] == kDim && q.numel() == out.numel(),
            "q and out must be contiguous BF16 [256, heads, T]");
    const int heads   = q.ne[1];
    const int columns = static_cast<int>(q.numel() / (static_cast<std::int64_t>(kDim) * heads));
    require_lossless_cache(cache, heads);
    require(sequences >= 1 && columns % sequences == 0, "the columns are `sequences` equal runs");
    require(positions.dtype == DType::I32 && positions.is_contiguous() && positions.numel() == columns &&
                table_rows.dtype == DType::I32 && table_rows.is_contiguous() && table_rows.numel() >= sequences,
            "positions I32 [T] and table rows I32 [B]");
    require(max_keys >= 1 && max_keys <= key_capacity &&
                static_cast<std::int64_t>(key_capacity) <= static_cast<std::int64_t>(cache.block_tables.ne[0]) * kPage,
            "max_keys in [1, key_capacity], key_capacity within the block table");
    require(select == nullptr || (ratio > 0 && select_words >= (max_keys / ratio + 31) / 32),
            "a selection needs its ratio and a bit per complete block");
    if (columns == 0) { return; }
    Args args{};
    args.q            = static_cast<const std::uint16_t*>(q.data);
    args.out          = static_cast<std::uint16_t*>(out.data);
    args.k            = static_cast<const std::uint16_t*>(cache.k_pages.data);
    args.v            = static_cast<const std::uint16_t*>(cache.v_pages.data);
    args.tables       = static_cast<const std::int32_t*>(cache.block_tables.data);
    args.table_rows   = static_cast<const std::int32_t*>(table_rows.data);
    args.table_stride = cache.block_tables.ne[0];
    args.positions    = static_cast<const std::int32_t*>(positions.data);
    args.width        = columns / sequences;
    args.sequences    = sequences;
    args.key_capacity = key_capacity;
    args.heads        = heads;
    args.kv_heads     = cache.num_kv_heads;
    args.columns      = columns;
    args.select       = select;
    args.select_words = select_words;
    args.ratio        = ratio;
    run(args, max_keys, workspace, stream);
}

void lossless_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions, int sequences,
                        const Tensor& table_rows, const PagedKVBatchLayerView& cache, cudaStream_t stream) {
    require(k.dtype == DType::BF16 && v.dtype == DType::BF16 && k.is_contiguous() && v.is_contiguous() &&
                k.ne[0] == kDim && k.numel() == v.numel() && k.ne[1] == cache.num_kv_heads,
            "k and v contiguous BF16 [256, Hkv, T]");
    require_lossless_cache(cache, cache.num_kv_heads);
    const int columns = static_cast<int>(k.numel() / (static_cast<std::int64_t>(kDim) * cache.num_kv_heads));
    require(sequences >= 1 && columns % sequences == 0, "the columns are `sequences` equal runs");
    require(positions.dtype == DType::I32 && positions.is_contiguous() && positions.numel() == columns &&
                table_rows.dtype == DType::I32 && table_rows.is_contiguous() && table_rows.numel() >= sequences,
            "positions I32 [T] and table rows I32 [B]");
    if (columns == 0) { return; }
    const long long n = static_cast<long long>(columns) * cache.num_kv_heads * (kDim * 2 / 16);
    kv_append_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        static_cast<const uint4*>(k.data), static_cast<const uint4*>(v.data),
        static_cast<const std::int32_t*>(positions.data), static_cast<const std::int32_t*>(cache.block_tables.data),
        static_cast<const std::int32_t*>(table_rows.data), cache.block_tables.ne[0], columns / sequences,
        cache.num_kv_heads, columns, static_cast<uint4*>(cache.k_pages.data), static_cast<uint4*>(cache.v_pages.data));
    CUDA_CHECK(cudaGetLastError());
}

std::size_t lossless_codec_attention_workspace_bytes(int query_heads, int kv_heads, int columns) {
    return codec_call_bytes(query_heads, kv_heads, columns, 2);
}

void lossless_codec_attention(const Tensor& q, const LosslessKvCodes& kv, int kv_heads, Tensor& out, void* workspace,
                              cudaStream_t stream, const std::uint32_t* select, int select_words, int ratio) {
    require(q.dtype == DType::BF16 && out.dtype == DType::BF16 && q.is_contiguous() && out.is_contiguous() &&
                q.ne[0] == kDim && out.ne[0] == kDim && q.numel() == out.numel(),
            "q and out must be contiguous BF16 [256, heads, T]");
    const int heads = q.ne[1], columns = q.ne[2];
    require(kv_heads >= 1 && kv_heads <= 8 && heads % kv_heads == 0, "Hq a multiple of Hkv (at most 8)");
    const auto valid = [](const LosslessKvPlane& p) {
        return p.codes != nullptr && p.scales != nullptr && static_cast<int>(p.codec) <= static_cast<int>(LosslessKvCodec::Vq2);
    };
    const bool window = kv.window_k.codes != nullptr;
    require(valid(kv.k) && valid(kv.v) && (!window || (valid(kv.window_k) && valid(kv.window_v) && kv.window_sinks >= 0 &&
                                                       kv.window_recent >= 0)),
            "the K and V planes (and a window's both) need codes and scales");
    require(select == nullptr || (ratio > 0 && select_words >= (columns / ratio + 31) / 32),
            "a selection needs its ratio and a bit per complete block");
    if (columns == 0) { return; }
    auto plane = [](const LosslessKvPlane& p) {
        return CodecPlane{p.codes, p.scales, static_cast<int>(p.codec), p.rotated ? 1 : 0};
    };
    Args args{};
    args.q             = static_cast<const std::uint16_t*>(q.data);
    args.out           = static_cast<std::uint16_t*>(out.data);
    args.width         = columns;
    args.sequences     = 1;
    args.key_capacity  = columns;
    args.heads         = heads;
    args.kv_heads      = kv_heads;
    args.columns       = columns;
    args.select        = select;
    args.select_words  = select_words;
    args.ratio         = ratio;
    args.kplane[0]     = plane(kv.k);
    args.vplane[0]     = plane(kv.v);
    args.reps          = window ? 2 : 1;
    args.window_sinks  = kv.window_sinks;
    args.window_recent = kv.window_recent;
    if (window) {
        args.kplane[1] = plane(kv.window_k);
        args.vplane[1] = plane(kv.window_v);
    }
    run_codec(args, workspace, stream);
}

} // namespace infernix::ops

// This unit's copy of the noise twin's seed (ops/lossless/noise.cuh).
INFERNIX_LOSSLESS_NOISE_UNIT(attention)
