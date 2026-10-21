#include "infernix/ops/lossless_ops.h"

#include "core/device.h"
#include "ops/lossless/cr_math.cuh"
#include "ops/lossless/wide.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

// The --lossless route's causal attention (include/infernix/ops/lossless_ops.h,
// docs/maintainer/lossless-route.md §2): s_j = (q . k_j) 2^-4 exact, m = max s_j,
// p_j = fp32(exp(fp32(s_j - m))), out = bf16(sum p_j v_j / sum p_j).
//
// Main kernel, one CTA per 16 query columns of one head, key tiles of 32:
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

// ------------------------------------------------------------------------- exponent maxima

// kmax[t * Hkv + h]: a key column's largest exponent field (kNonFinite when its k or v holds an Inf or
// NaN); vmax[h * 256 + c]: a value channel's largest over the window.
__global__ void maxima_kernel(const std::uint16_t* k, const std::uint16_t* v, int kv_heads, int columns,
                              std::int16_t* kmax, int* vmax) {
    const int row = static_cast<int>(blockIdx.x), lane = static_cast<int>(threadIdx.x); // row = t * Hkv + h
    if (row >= columns * kv_heads) { return; }
    const std::uint16_t* kr = k + static_cast<std::size_t>(row) * kDim;
    const std::uint16_t* vr = v + static_cast<std::size_t>(row) * kDim;
    int e = 0;
    bool bad = false;
    for (int d = lane; d < kDim; d += 32) {
        e = max(e, bf16_field(kr[d]));
        bad = bad || bf16_nonfinite(kr[d]) || bf16_nonfinite(vr[d]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { e = max(e, __shfl_xor_sync(0xFFFFFFFFU, e, o)); }
    bad = __any_sync(0xFFFFFFFFU, bad);
    if (lane == 0) { kmax[row] = static_cast<std::int16_t>(bad ? kNonFinite : (e == 0 ? 1 : e)); }
    const int h = row % kv_heads;
    for (int d = lane; d < kDim; d += 32) {
        if (!bf16_nonfinite(vr[d])) { atomicMax(vmax + h * kDim + d, bf16_field(vr[d])); }
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

struct Shared {
    int xq[kQueries][kDim];  // query grid integers (0 off the grid)
    int xk[kKeys][kDim];     // key grid integers
    int vg[kKeys][kDim];     // value grid integers: M 2^(e - vmax_c + 21), rounded below the grid
    float p[kQueries][kKeys];
    int ph[kQueries][kKeys]; // p 2^48 = ph 2^24 + pl
    int pl[kQueries][kKeys];
    std::int16_t qb[kQueries][kBlocks];
    std::int16_t kb[kKeys][kBlocks];
    int qmax[kQueries];
    int qbad[kQueries];
    int qoff[kQueries];      // the query row has an element off its block's grid
    int kmax[kKeys];         // the key row's largest exponent field, or kNonFinite
    int koff[kKeys];
    int vmax[kDim];
    int vinexact[kDim];      // a value element of the channel was rounded to its grid
};

struct Args {
    const std::uint16_t* q;
    const std::uint16_t* k;
    const std::uint16_t* v;
    std::uint16_t* out;
    const std::int16_t* kmax;
    const int* vmax;
    std::uint32_t* flagged; // one bit per (query column, head)
    int heads;
    int kv_heads;
    int columns;
    const std::uint32_t* select; // null: causal; else [columns][select_words] bits of the selected blocks
    int select_words;
    int ratio;
};

__device__ __forceinline__ bool visible(const Args& a, int t, int j) {
    if (j > t) { return false; }
    if (a.select == nullptr) { return true; }
    const int tail = (t + 1) / a.ratio * a.ratio;
    if (j >= tail) { return true; }
    const int b = j / a.ratio;
    return ((a.select[static_cast<std::size_t>(t) * a.select_words + b / 32] >> (b % 32)) & 1U) != 0;
}

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

// v 2^(155 - vmax): the value's grid integer for its channel (rounded to nearest even below the grid).
__device__ __forceinline__ int value_grid(std::uint16_t bits, int vmax, bool& rounded) {
    int m = 0, e = 1;
    bf16_parts(bits, m, e);
    if (m == 0) { return 0; }
    const int shift = e - vmax + kGridBits;
    if (shift >= 0) { return m * (1 << shift); }
    rounded         = true;
    const int down  = -shift;
    const int mag   = m < 0 ? -m : m;
    if (down >= 9) { return 0; }
    const int q = mag >> down, r = mag & ((1 << down) - 1), half = 1 << (down - 1);
    const int v = q + ((r > half || (r == half && (q & 1) != 0)) ? 1 : 0);
    return m < 0 ? -v : v;
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
__device__ Score exact_pair_score(const Args& a, int t, int h, int kvh, int j, int qmax, int kmax, bool& dropped) {
    const std::uint16_t* qr = a.q + (static_cast<std::size_t>(t) * a.heads + h) * kDim;
    const std::uint16_t* kr = a.k + (static_cast<std::size_t>(j) * a.kv_heads + kvh) * kDim;
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

// The score of query row i and key row jj: per 64-block, the grid integers' products summed exactly in
// int64 halves, placed at the block's exponents relative to the rows' largest.
__device__ __forceinline__ Score tile_score(const Shared& sh, const Args& a, int t, int h, int kvh, int j0, int i,
                                            int jj, bool& dropped) {
    if (sh.qoff[i] != 0 || sh.koff[jj] != 0) {
        return exact_pair_score(a, t, h, kvh, j0 + jj, sh.qmax[i], sh.kmax[jj], dropped);
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

__global__ __launch_bounds__(kThreads) void attention_kernel(const Args args) {
    extern __shared__ __align__(16) unsigned char raw[];
    Shared& sh    = *reinterpret_cast<Shared*>(raw);
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int h = static_cast<int>(blockIdx.y), t0 = static_cast<int>(blockIdx.x) * kQueries;
    const int kvh = h / (args.heads / args.kv_heads);
    const int i = tid / 16, sub = tid % 16; // this thread's query row and its slot among the row's 16 threads
    const int t = t0 + i;
    const bool active = t < args.columns;

    // Query tile: non-finite rows, the grid integers per block, each row's largest block exponent.
    if (tid < kQueries) {
        sh.qbad[tid] = 0;
        sh.qoff[tid] = 0;
    }
    for (int d = tid; d < kDim; d += kThreads) {
        sh.vmax[d]     = max(args.vmax[kvh * kDim + d], 1);
        sh.vinexact[d] = 0;
    }
    __syncthreads();
    for (int idx = tid; idx < kQueries * kDim; idx += kThreads) {
        const int r = idx / kDim, d = idx % kDim;
        if (t0 + r < args.columns &&
            bf16_nonfinite(args.q[(static_cast<std::size_t>(t0 + r) * args.heads + h) * kDim + d])) {
            sh.qbad[r] = 1;
        }
    }
    for (int unit = warp; unit < kQueries * kBlocks; unit += kThreads / 32) {
        const int r = unit / kBlocks, b = unit % kBlocks;
        const std::uint16_t* src =
            t0 + r < args.columns ? args.q + (static_cast<std::size_t>(t0 + r) * args.heads + h) * kDim + 64 * b : nullptr;
        encode_block(src, lane, &sh.xq[r][64 * b], sh.qb[r][b], sh.qoff[r]);
    }
    __syncthreads();
    if (tid < kQueries) {
        int e = 1;
        for (int b = 0; b < kBlocks; ++b) { e = max(e, static_cast<int>(sh.qb[tid][b])); }
        sh.qmax[tid] = e;
    }
    __syncthreads();

    const int last_key = min(t0 + kQueries, args.columns) - 1;
    bool nan = active && sh.qbad[i] != 0;
    auto load_keys = [&](int j0, bool values) {
        for (int r = tid; r < kKeys; r += kThreads) {
            const int j = j0 + r;
            sh.kmax[r] = j <= last_key ? args.kmax[static_cast<std::size_t>(j) * args.kv_heads + kvh] : 1;
            sh.koff[r] = 0;
        }
        __syncthreads();
        for (int unit = warp; unit < kKeys * kBlocks; unit += kThreads / 32) {
            const int r = unit / kBlocks, b = unit % kBlocks, j = j0 + r;
            const std::uint16_t* src =
                j <= last_key && sh.kmax[r] != kNonFinite
                    ? args.k + (static_cast<std::size_t>(j) * args.kv_heads + kvh) * kDim + 64 * b
                    : nullptr;
            encode_block(src, lane, &sh.xk[r][64 * b], sh.kb[r][b], sh.koff[r]);
        }
        if (values) {
            for (int idx = tid; idx < kKeys * kDim; idx += kThreads) {
                const int r = idx / kDim, d = idx % kDim, j = j0 + r;
                int x = 0;
                if (j <= last_key && sh.kmax[r] != kNonFinite) {
                    bool rounded = false;
                    x = value_grid(args.v[(static_cast<std::size_t>(j) * args.kv_heads + kvh) * kDim + d], sh.vmax[d],
                                   rounded);
                    if (rounded) { sh.vinexact[d] = 1; }
                }
                sh.vg[r][d] = x;
            }
        }
        __syncthreads();
    };

    // Pass A: each query's exact maximum (each thread: keys sub and sub + 16 of every tile).
    Score best{{0ULL, 0LL}, 0};
    bool have = false, fail = false;
    for (int j0 = 0; j0 <= last_key; j0 += kKeys) {
        load_keys(j0, false);
        if (active) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int jj = sub + 16 * half, j = j0 + jj;
                if (!visible(args, t, j)) { continue; }
                if (sh.kmax[jj] == kNonFinite) {
                    nan = true;
                    continue;
                }
                const Score s = tile_score(sh, args, t, h, kvh, j0, i, jj, fail);
                if (!have || score_greater(s, best)) {
                    best = s;
                    have = true;
                }
            }
        }
        __syncthreads();
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

    // Pass B: p, sum p (int128 at 2^-87; p <= 1) and the thread's 16 channels' sum p v (int128 at
    // 2^(vmax_c - 203): p on its 2^-48 grid, v on its channel's grid).
    const int c0 = sub * 16;
    I128 num[16];
#pragma unroll
    for (int c = 0; c < 16; ++c) { num[c] = I128{0ULL, 0LL}; }
    I128 den{0ULL, 0LL};
    float den_err = 0.0F;
    double p_err  = 0.0; // the p grid's dropped bits (each below 2^-49), summed over this thread's keys
    for (int j0 = 0; j0 <= last_key; j0 += kKeys) {
        load_keys(j0, true);
        if (active && !nan) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int jj = sub + 16 * half, j = j0 + jj;
                float p = 0.0F;
                if (visible(args, t, j)) {
                    bool ok       = true;
                    const Score s = tile_score(sh, args, t, h, kvh, j0, i, jj, fail);
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
                    const long long M = field == 0 ? mant : (0x800000 | mant);
                    const int shift = (field == 0 ? 1 : field) - 150 + kPBits;
                    if (shift >= 0) {
                        P = M << shift;
                    } else {
                        P = shift > -64 ? ((M + (1LL << (-shift - 1))) >> -shift) : 0;
                        p_err += 0x1p-49;
                    }
                }
                sh.p[i][jj]  = p;
                sh.ph[i][jj] = static_cast<int>(P >> 24);
                sh.pl[i][jj] = static_cast<int>(P & 0xFFFFFF);
            }
        }
        __syncthreads();
        if (active && !nan) {
#pragma unroll
            for (int cg = 0; cg < 16; cg += 4) {
                long long hi[4] = {0, 0, 0, 0}, lo[4] = {0, 0, 0, 0};
                for (int jj = 0; jj < kKeys; ++jj) {
                    const long long ph = sh.ph[i][jj], pl = sh.pl[i][jj];
                    if ((ph | pl) == 0) { continue; }
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const long long v = sh.vg[jj][c0 + cg + c];
                        hi[c] += ph * v; // |.| < 2^24 2^29 per key: 32 keys below 2^59
                        lo[c] += pl * v;
                    }
                }
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    bool unused = false;
                    add_shifted(num[cg + c], hi[c], 24, unused);
                    add_shifted(num[cg + c], lo[c], 0, unused);
                }
            }
        }
        __syncthreads();
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
    bool flag = false;
    if (active) {
        std::uint16_t* out = args.out + (static_cast<std::size_t>(t) * args.heads + h) * kDim + c0;
        if (nan) {
#pragma unroll
            for (int c = 0; c < 16; ++c) { out[c] = kNaN; }
        } else if (fail) {
            flag = true;
        } else {
            const double dsum = lossless::double_i128(den, -87);
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                const int vm = sh.vmax[c0 + c];
                const double n = lossless::double_i128(num[c], vm - 203);
                const double o = n / dsum;
                // The num error: p's dropped grid bits times the channel's largest |v| (below 2^(vm - 126)),
                // and half a v grid unit per rounded value element times sum p.
                const double num_err = p_err * ldexp(1.0, vm - 126) +
                                       (sh.vinexact[c0 + c] != 0 ? (dsum + den_err) * ldexp(1.0, vm - 156) : 0.0);
                // The conversions (2^-51 each), the division (2^-53), and the dropped terms.
                const double err = fabs(o) * 0x1p-48 + (num_err + fabs(o) * den_err) / dsum * 1.01;
                std::uint16_t r;
                if (lossless::certain_bf16(o, err, r)) {
                    out[c] = r;
                } else {
                    flag = true;
                }
            }
        }
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) { flag = __shfl_xor_sync(0xFFFFFFFFU, flag ? 1 : 0, o) != 0 || flag; }
    if (active && sub == 0 && flag) {
        const std::size_t bit = static_cast<std::size_t>(t) * args.heads + h;
        atomicOr(args.flagged + bit / 32, 1U << (bit % 32));
    }
}

// ------------------------------------------------------------------------- exact fallback

constexpr int kScoreWords = 18; // scores from 2^-272: positions e_q + e_k in [2, 508], plus 24 bits
constexpr int kScoreBase  = -272;
constexpr int kSumWords   = 15; // p and p v from 2^-300: positions in [18, 397], plus 46 bits
constexpr int kSumBase    = -300;
constexpr int kFallbackWarps = 64;

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

__device__ void exact_score(const Args& a, int t, int h, int kvh, int j, Wide<kScoreWords>& acc) {
    acc.clear();
    const std::uint16_t* qr = a.q + (static_cast<std::size_t>(t) * a.heads + h) * kDim;
    const std::uint16_t* kr = a.k + (static_cast<std::size_t>(j) * a.kv_heads + kvh) * kDim;
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
    float* p_store; // kFallbackWarps x columns
};

__global__ void attention_fallback_kernel(const FallbackArgs f) {
    const Args& a  = f.a;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int total = a.columns * a.heads;
    float* pstore   = f.p_store + static_cast<std::size_t>(warp) * a.columns;
    for (int query = warp; query < total; query += kFallbackWarps) {
        if (((a.flagged[query / 32] >> (query % 32)) & 1U) == 0) { continue; }
        const int t = query / a.heads, h = query % a.heads, kvh = h / (a.heads / a.kv_heads);
        // The maximum: per lane over its keys, then across the warp.
        std::uint32_t best[kScoreWords] = {};
        bool best_neg = false, have = false;
        for (int j = lane; j <= t; j += 32) {
            if (!visible(a, t, j)) { continue; }
            Wide<kScoreWords> acc;
            exact_score(a, t, h, kvh, j, acc);
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
        for (int j = lane; j <= t; j += 32) {
            if (!visible(a, t, j)) {
                pstore[j] = 0.0F;
                continue;
            }
            Wide<kScoreWords> acc;
            exact_score(a, t, h, kvh, j, acc);
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
            for (int j = lane; j <= t; j += 32) {
                const float p = pstore[j];
                if (p == 0.0F) { continue; }
                int mv, ev;
                bf16_parts(a.v[(static_cast<std::size_t>(j) * a.kv_heads + kvh) * kDim + c], mv, ev);
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

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

} // namespace

std::size_t lossless_attention_workspace_bytes(int query_heads, int columns) {
    return align256(static_cast<std::size_t>(columns) * 8 * sizeof(std::int16_t)) + align256(8 * kDim * sizeof(int)) +
           align256((static_cast<std::size_t>(columns) * query_heads + 31) / 32 * 4) +
           static_cast<std::size_t>(kFallbackWarps) * columns * sizeof(float);
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
    auto* base = static_cast<std::uint8_t*>(workspace);
    auto* kmax = reinterpret_cast<std::int16_t*>(base);
    base += align256(static_cast<std::size_t>(columns) * 8 * sizeof(std::int16_t));
    auto* vmax = reinterpret_cast<int*>(base);
    base += align256(8 * kDim * sizeof(int));
    auto* flagged                = reinterpret_cast<std::uint32_t*>(base);
    const std::size_t flag_bytes = (static_cast<std::size_t>(columns) * heads + 31) / 32 * 4;
    base += align256(flag_bytes);
    auto* pstore = reinterpret_cast<float*>(base);
    CUDA_CHECK(cudaMemsetAsync(vmax, 0, 8 * kDim * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(flagged, 0, flag_bytes, stream));
    maxima_kernel<<<columns * kv_heads, 32, 0, stream>>>(static_cast<const std::uint16_t*>(k.data),
                                                         static_cast<const std::uint16_t*>(v.data), kv_heads,
                                                         columns, kmax, vmax);
    CUDA_CHECK(cudaGetLastError());
    const Args args{static_cast<const std::uint16_t*>(q.data),
                    static_cast<const std::uint16_t*>(k.data),
                    static_cast<const std::uint16_t*>(v.data),
                    static_cast<std::uint16_t*>(out.data),
                    kmax,
                    vmax,
                    flagged,
                    heads,
                    kv_heads,
                    columns,
                    select,
                    select_words,
                    ratio};
    constexpr int kSmem = static_cast<int>(sizeof(Shared));
    static const bool configured = [] {
        CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
        return true;
    }();
    (void)configured;
    attention_kernel<<<dim3((columns + kQueries - 1) / kQueries, heads), kThreads, kSmem, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
    attention_fallback_kernel<<<kFallbackWarps / 8, 256, 0, stream>>>(FallbackArgs{args, pstore});
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops
