// Inkling 2's visit-chain kernels of the decode and verify route (include/infernix/ops/inkling2_visit.h). One CTA a
// column; thread g holds elements 8g .. 8g + 7 of every H-wide row, so an H-vector sum is the pinned order of
// visit_math.h: a thread's 8 products folded by fma, the warp's 32 by a xor butterfly (the bfly32 tree), the 9 warps
// folded from +0.0 through shared memory. Each reduction phase has its own shared scratch, so a phase needs one
// barrier. Immutable weights (queries, norms, conv taps) are read before the PDL wait, and so are rows of earlier calls
// (the column map, the state's conv rings: column.cuh early_history); everything a kernel of the call wrote is read
// after it with ordinary loads (never the non-coherent path, which may be hoisted above the wait).

#include "infernix/ops/inkling2_visit.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_a8.h"
#include "ops/inkling2/column.cuh"
#include "ops/offloaded_sparse_moe/cuda/a16_limbs.cuh"
#include "ops/inkling2/visit_math.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops::inkling2 {
namespace {

namespace moe = infernix::ops::w4a4_moe;
namespace a16 = infernix::ops::offloaded_moe::wide;

constexpr int kThreads  = kGroups;
constexpr unsigned kAll = 0xFFFFFFFFU;
constexpr int kTopK     = moe::kTopK;
constexpr int kNvfp4RowBytes = moe::row_bytes(moe::ExpertFormat::Nvfp4FourOverSix, kHidden);
constexpr int kMxfp8RowBytes = moe::row_bytes(moe::ExpertFormat::Mxfp8, kHidden);

// ---------------------------------------------------------------------------- loads and stores of 8 elements

__device__ __forceinline__ void unpack8(uint4 w, float (&v)[8]) {
    const std::uint32_t u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[2 * j]     = canon::f32_from_bits(u[j] << 16);
        v[2 * j + 1] = canon::f32_from_bits(u[j] & 0xFFFF0000U);
    }
}

// Producer-written rows.
__device__ __forceinline__ void load8(const std::uint16_t* p, float (&v)[8]) {
    unpack8(*reinterpret_cast<const uint4*>(p), v);
}
__device__ __forceinline__ void load8(const float* p, float (&v)[8]) {
    const float4 a = reinterpret_cast<const float4*>(p)[0];
    const float4 b = reinterpret_cast<const float4*>(p)[1];
    v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
}

// Immutable weights.
__device__ __forceinline__ void weight8(const std::uint16_t* p, float (&v)[8]) {
    unpack8(__ldg(reinterpret_cast<const uint4*>(p)), v);
}
__device__ __forceinline__ void weight8(const float* p, float (&v)[8]) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(p));
    const float4 b = __ldg(reinterpret_cast<const float4*>(p) + 1);
    v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
}

__device__ __forceinline__ void store8(std::uint16_t* p, const float (&v)[8]) {
    std::uint32_t u[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        u[j] = std::uint32_t{canon::f32_to_bf16_rn(v[2 * j])} | (std::uint32_t{canon::f32_to_bf16_rn(v[2 * j + 1])} << 16);
    }
    *reinterpret_cast<uint4*>(p) = make_uint4(u[0], u[1], u[2], u[3]);
}
__device__ __forceinline__ void store8(float* p, const float (&v)[8]) {
    reinterpret_cast<float4*>(p)[0] = make_float4(v[0], v[1], v[2], v[3]);
    reinterpret_cast<float4*>(p)[1] = make_float4(v[4], v[5], v[6], v[7]);
}

__device__ __forceinline__ void zero8(float (&v)[8]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) { v[j] = 0.0F; }
}

// The BF16 value of each element (round to nearest even).
__device__ __forceinline__ void round8(float (&v)[8]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) { v[j] = canon::bf16_to_f32(canon::f32_to_bf16_rn(v[j])); }
}

__device__ __forceinline__ float fold8(const float (&a)[8], const float (&b)[8]) {
    float acc = 0.0F;
#pragma unroll
    for (int j = 0; j < 8; ++j) { acc = canon::fma_rn(a[j], b[j], acc); }
    return acc;
}

// ---------------------------------------------------------------------------- CTA reductions

// One round of a transposing xor butterfly over M values (offset o): the lane with bit o clear keeps the even values of
// each pair, its partner the odd ones, and each adds the value its partner sent; (M + 1) / 2 values remain. Values past
// M are zero (a pair's missing odd member), so a lane's padding slots stay zero.
template <int M>
__device__ __forceinline__ void transpose_round(const float (&x)[M], float (&y)[(M + 1) / 2], int lane, int o) {
    const bool upper = (lane & o) != 0;
#pragma unroll
    for (int i = 0; i < (M + 1) / 2; ++i) {
        const float even = x[2 * i];
        const float odd  = 2 * i + 1 < M ? x[2 * i + 1] : 0.0F;
        const float keep = upper ? odd : even;
        const float send = upper ? even : odd;
        y[i]             = canon::add_rn(keep, __shfl_xor_sync(kAll, send, o));
    }
}

// N H-vector sums at once: each thread's partials in v, the totals back in v (every thread holds them). The pinned
// order (visit_math.h): a warp's 32 partials by the xor butterfly (offsets 16, 8, 4, 2, 1), the 9 warps folded from
// +0.0. N >= 2 runs the butterfly transposed (the same pairs and order, FP32 addition being commutative): after the five
// rounds value n is held by the lane whose 5 bits are n's reversed, which stores the warp's sum; then lane n of every
// warp folds value n's 9 warp sums and the warp broadcasts the totals.
template <int N>
__device__ __forceinline__ void cta_sum(float (&v)[N], float (&scratch)[N][kWarpsH]) {
    static_assert(N >= 1 && N <= 32, "cta_sum: 1-32 sums");
    const int lane = static_cast<int>(threadIdx.x & 31U), warp = static_cast<int>(threadIdx.x >> 5);
    if constexpr (N == 1) {
#pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) { v[0] = canon::add_rn(v[0], __shfl_xor_sync(kAll, v[0], offset)); }
        if (lane == 0) { scratch[0][warp] = v[0]; }
        __syncthreads();
        float total = 0.0F;
#pragma unroll
        for (int w = 0; w < kWarpsH; ++w) { total = canon::add_rn(total, scratch[0][w]); }
        v[0] = total;
    } else {
        constexpr int M1 = (N + 1) / 2, M2 = (M1 + 1) / 2, M3 = (M2 + 1) / 2, M4 = (M3 + 1) / 2;
        static_assert((M4 + 1) / 2 == 1, "cta_sum: five rounds leave one value a lane");
        float y1[M1], y2[M2], y3[M3], y4[M4], y5[1];
        transpose_round<N>(v, y1, lane, 16);
        transpose_round<M1>(y1, y2, lane, 8);
        transpose_round<M2>(y2, y3, lane, 4);
        transpose_round<M3>(y3, y4, lane, 2);
        transpose_round<M4>(y4, y5, lane, 1);
        const int held = static_cast<int>(__brev(static_cast<unsigned>(lane)) >> 27);
        if (held < N) { scratch[held][warp] = y5[0]; }
        __syncthreads();
        float total = 0.0F;
        if (lane < N) {
#pragma unroll
            for (int w = 0; w < kWarpsH; ++w) { total = canon::add_rn(total, scratch[lane][w]); }
        }
#pragma unroll
        for (int n = 0; n < N; ++n) { v[n] = __shfl_sync(kAll, total, n); }
    }
}

// The largest of the threads' non-negative values (order-free).
__device__ __forceinline__ float cta_max(float v, float (&scratch)[kWarpsH]) {
    const int lane = static_cast<int>(threadIdx.x & 31U), warp = static_cast<int>(threadIdx.x >> 5);
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        const float o = __shfl_xor_sync(kAll, v, offset);
        v             = o > v ? o : v;
    }
    if (lane == 0) { scratch[warp] = v; }
    __syncthreads();
    float m = 0.0F;
#pragma unroll
    for (int w = 0; w < kWarpsH; ++w) { m = scratch[w] > m ? scratch[w] : m; }
    return m;
}

// ---------------------------------------------------------------------------- mixes

// QA queries over group A's sources (at most kMaxSources: the completed blocks and the partial) and QB queries over
// group B's two (the previous pass's message sources), in two reduction phases for both groups: the sources' sums
// of squares, then every (query, source) score.
template <int QA, int QB>
struct Mix {
    static constexpr int kSB = QB > 0 ? 2 : 0;
    static constexpr int kS  = kMaxSources + kSB;
    static constexpr int kQ  = QA + QB;
    static constexpr int kD  = QA * kMaxSources + QB * kSB;
};

// The partial source's fixed slot in group A: after every completed block, so slot order is source order (a slot
// chosen by the block count would be a dynamic index, which puts the sources in local memory).
constexpr int kPartial = kMaxSources - 1;

// src: group A's `blocks` completed blocks at 0 .. blocks - 1 and its partial at kPartial (unused slots zero),
// group B's sources at kMaxSources ..; normalized in place. Slot order is source order.
template <int QA, int QB>
__device__ __forceinline__ void mix_columns(float (&src)[Mix<QA, QB>::kS][8], int blocks,
                                            const float (&q)[Mix<QA, QB>::kQ][8], float (&out)[Mix<QA, QB>::kQ][8],
                                            float (&s1)[Mix<QA, QB>::kS][kWarpsH],
                                            float (&s2)[Mix<QA, QB>::kD][kWarpsH]) {
    using M = Mix<QA, QB>;
    const auto in_a = [blocks](int s) { return s < blocks || s == kPartial; };
    float sq[M::kS];
#pragma unroll
    for (int s = 0; s < M::kS; ++s) { sq[s] = s >= kMaxSources || in_a(s) ? fold8(src[s], src[s]) : 0.0F; }
    cta_sum<M::kS>(sq, s1);
#pragma unroll
    for (int s = 0; s < M::kS; ++s) {
        const float r = rms_factor(sq[s], static_cast<float>(kHidden));
#pragma unroll
        for (int j = 0; j < 8; ++j) { src[s][j] = canon::mul_rn(src[s][j], r); }
    }
    float d[M::kD];
#pragma unroll
    for (int qi = 0; qi < QA; ++qi) {
#pragma unroll
        for (int s = 0; s < kMaxSources; ++s) { d[qi * kMaxSources + s] = in_a(s) ? fold8(q[qi], src[s]) : 0.0F; }
    }
#pragma unroll
    for (int qi = 0; qi < QB; ++qi) {
#pragma unroll
        for (int s = 0; s < M::kSB; ++s) {
            d[QA * kMaxSources + qi * M::kSB + s] = fold8(q[QA + qi], src[kMaxSources + s]);
        }
    }
    cta_sum<M::kD>(d, s2);
#pragma unroll
    for (int qi = 0; qi < M::kQ; ++qi) {
        const bool a       = qi < QA;
        const int base     = a ? qi * kMaxSources : QA * kMaxSources + (qi - QA) * M::kSB;
        const int first    = a ? 0 : kMaxSources;
        constexpr int kMax = kMaxSources;
        const auto active  = [&](int s) { return a ? in_a(s) : s < M::kSB; };
        float p[kMax];
        float m = canon::mul_rn(d[a ? base + kPartial : base], kMixScale); // an active source's score
#pragma unroll
        for (int s = 0; s < kMax; ++s) {
            p[s] = active(s) ? canon::mul_rn(d[base + s], kMixScale) : 0.0F;
            if (active(s)) { m = p[s] > m ? p[s] : m; }
        }
        float den = 0.0F;
#pragma unroll
        for (int s = 0; s < kMax; ++s) {
            if (active(s)) {
                p[s] = canon::exp_c(canon::add_rn(p[s], -m));
                den  = canon::add_rn(den, p[s]);
            }
        }
#pragma unroll
        for (int s = 0; s < kMax; ++s) { p[s] = active(s) ? canon::div_rn(p[s], den) : 0.0F; }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float acc = 0.0F;
#pragma unroll
            for (int s = 0; s < kMax; ++s) {
                if (active(s)) { acc = canon::fma_rn(p[s], src[first + s][j], acc); }
            }
            out[qi][j] = acc;
        }
    }
}

// ---------------------------------------------------------------------------- short-conv history

using detail::Column;
using detail::column_of;
using detail::early_history;
using detail::history_row;
using detail::history_source;
using detail::ring_row;
using detail::Source;
using detail::writes_ring;

// The static conv of the thread's 8 channels: taps w[lag][j], values x[lag][j] (lag 0 the current position).
__device__ __forceinline__ void conv8(const float (&w)[4][8], const float (&x)[4][8], float (&out)[8]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float wj[4] = {w[0][j], w[1][j], w[2][j], w[3][j]};
        const float xj[4] = {x[0][j], x[1][j], x[2][j], x[3][j]};
        out[j]            = conv_static(wj, xj);
    }
    round8(out);
}

// A column's combined expert output: the 8 slots folded in slot order from +0.0, rounded once.
__device__ __forceinline__ void combine8(const std::uint16_t* experts, int t, int c, float (&v)[8]) {
    float e[kTopK][8];
#pragma unroll
    for (int j = 0; j < kTopK; ++j) { load8(experts + static_cast<std::size_t>(t * kTopK + j) * kHidden + c, e[j]); }
    zero8(v);
#pragma unroll
    for (int j = 0; j < kTopK; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { v[i] = canon::add_rn(v[i], e[j][i]); }
    }
    round8(v);
}

// Raw rows, issued before anything waits on them (a warp issues in order: an unpack right after its load would hold
// back every later load).
__device__ __forceinline__ uint4 raw8(const std::uint16_t* p) { return *reinterpret_cast<const uint4*>(p); }
struct Raw8F {
    float4 a, b;
};
__device__ __forceinline__ Raw8F raw8(const float* p) {
    return {reinterpret_cast<const float4*>(p)[0], reinterpret_cast<const float4*>(p)[1]};
}
__device__ __forceinline__ void unpack8(const Raw8F& r, float (&v)[8]) {
    v[0] = r.a.x, v[1] = r.a.y, v[2] = r.a.z, v[3] = r.a.w, v[4] = r.b.x, v[5] = r.b.y, v[6] = r.b.z, v[7] = r.b.w;
}

// The current column's 8 expert rows and cand, loaded with the visit's other rows.
struct TailRows {
    uint4 e[kTopK];
    Raw8F cand;
};
__device__ __forceinline__ TailRows tail_rows(const MlpTail& tail, int t, int c) {
    TailRows r;
#pragma unroll
    for (int j = 0; j < kTopK; ++j) { r.e[j] = raw8(tail.experts + static_cast<std::size_t>(t * kTopK + j) * kHidden + c); }
    r.cand = raw8(tail.cand + static_cast<std::size_t>(t) * kHidden + c);
    return r;
}

// The 8 slots folded in slot order from +0.0, rounded once (combine8's arithmetic on rows already loaded).
__device__ __forceinline__ void combine_rows(const uint4 (&e)[kTopK], float (&v)[8]) {
    zero8(v);
#pragma unroll
    for (int j = 0; j < kTopK; ++j) {
        float f[8];
        unpack8(e[j], f);
#pragma unroll
        for (int i = 0; i < 8; ++i) { v[i] = canon::add_rn(v[i], f[i]); }
    }
    round8(v);
}

// A column's conv-history rows k = 1 .. 3 that come from the ring (Source::Ring), raw (the others zero): before the
// PDL wait when early_history(ring).
__device__ __forceinline__ void ring_history(const Ring& ring, const ColumnMap& map, const Column& col, int c,
                                             uint4 (&h)[3]) {
#pragma unroll
    for (int k = 1; k < 4; ++k) {
        h[k - 1] = history_source(map, col, k) == Source::Ring ? raw8(history_row(ring, col.lane, col.pos - k) + c)
                                                                : make_uint4(0U, 0U, 0U, 0U);
    }
}

// The MLP tail of the previous visit for column col: out = bf16(cand + bf16(conv(combined))), the combined output
// written to the ring. hist: ring_history's rows of tail.ring.
__device__ __forceinline__ void mlp_tail(const MlpTail& tail, const ColumnMap& map, const Column& col, int c,
                                         const float (&w)[4][8], const TailRows& rows, const uint4 (&hist)[3],
                                         float (&out)[8]) {
    float x[4][8];
    combine_rows(rows.e, x[0]);
#pragma unroll
    for (int k = 1; k < 4; ++k) {
        switch (history_source(map, col, k)) {
        case Source::Zero: zero8(x[k]); break;
        case Source::Call: combine8(tail.experts, col.t - k, c, x[k]); break;
        case Source::Ring: unpack8(hist[k - 1], x[k]); break;
        }
    }
    float m[8];
    conv8(w, x, m);
    float cand[8];
    unpack8(rows.cand, cand);
#pragma unroll
    for (int j = 0; j < 8; ++j) { out[j] = canon::add_rn(cand[j], m[j]); }
    round8(out);
    if (writes_ring(map, col)) { store8(ring_row(tail.ring, col.lane, col.pos) + c, x[0]); }
}

// ---------------------------------------------------------------------------- 8-bit rows (--a8-activations)

// The thread's 8 values (elements 8g .. 8g + 7 of a K = kHidden row; 4 threads a 32-slice) as an A8 row: the codec of
// ops/ggml/ggml_a8.h over their BF16 bits (the slice's largest magnitude's key, then the codes), the row's 2,304
// codes then its 72 FP32 scales (a8_encode_columns' layout).
__device__ __forceinline__ void a8_row8(const float (&v)[8], std::uint8_t* row) {
    namespace a8 = ::infernix::ggml::a8;
    const int g = static_cast<int>(threadIdx.x);
    std::uint16_t bits[8];
    unsigned key = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        bits[j] = canon::f32_to_bf16_rn(v[j]);
        key     = max(key, static_cast<unsigned>(bits[j] & 0x7FFFU));
    }
    key = max(key, __shfl_xor_sync(kAll, key, 1));
    key = max(key, __shfl_xor_sync(kAll, key, 2));
    const a8::SliceScale sc = a8::slice_scale(key);
    unsigned packed[2] = {0, 0};
#pragma unroll
    for (int j = 0; j < 8; ++j) { packed[j / 4] |= (static_cast<unsigned>(a8::code(bits[j], sc.inv)) & 0xFFU) << (8 * (j % 4)); }
    *reinterpret_cast<uint2*>(row + 8 * g) = make_uint2(packed[0], packed[1]);
    if ((g & 3) == 0) { reinterpret_cast<float*>(row + kHidden)[g / 4] = sc.d; }
}

// ---------------------------------------------------------------------------- the experts' activation rows

// The column's MLP input (BF16 values, the thread's 8) as the experts' rows: four-over-six NVFP4 (codes, a scale word
// per 16 elements held by a thread pair, the row amax) or MXFP8 (codes, a UE8M0 word per 32 held by 4 threads).
__device__ __forceinline__ void quantize_row(const float (&v)[8], const ExpertRows& rows, int t,
                                             float (&scratch)[kWarpsH]) {
    const int g = static_cast<int>(threadIdx.x);
    if (rows.a8) {
        a8_row8(v, rows.rows + static_cast<std::size_t>(t) * moe::kA8XRowBytes);
        return;
    }
    if (rows.emax != nullptr) {
        // A16: the row's exponent from its largest BF16 magnitude bits (a NaN's bits exceed every finite one's), then a
        // block's two halves on a thread pair (store_a16_half: store_a16_block's bytes).
        std::uint16_t bits[8];
        unsigned key = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            bits[j] = canon::f32_to_bf16_rn(v[j]);
            key     = max(key, static_cast<unsigned>(bits[j] & 0x7FFFU));
        }
        key = __reduce_max_sync(kAll, key);
        auto& keys = reinterpret_cast<unsigned (&)[kWarpsH]>(scratch);
        if ((g & 31) == 0) { keys[g / 32] = key; }
        __syncthreads();
        unsigned all = 0;
#pragma unroll
        for (int w = 0; w < kWarpsH; ++w) { all = max(all, keys[w]); }
        const int e = a16::a16_emax(all);
        a16::store_a16_half(rows.rows + static_cast<std::size_t>(t) * moe::kA16XRowBytes, g / 2, g & 1, bits, e);
        if (g == 0) { rows.emax[t] = e; }
        return;
    }
    float a     = 0.0F;
    bool nan    = false;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float m = canon::f32_from_bits(canon::f32_bits(v[j]) & 0x7FFFFFFFU);
        a             = m > a ? m : a;
        nan           = nan || v[j] != v[j];
    }
    if (rows.format == moe::ExpertFormat::Nvfp4FourOverSix) {
        const float amax = cta_max(a, scratch);
        if (g == 0) { rows.amax[t] = amax; }
        const canon::FourOverSixRow row = canon::four_over_six_row(amax);
        // A block is a thread pair's 16 elements: each thread evaluates both candidates over its own 8 and the pair
        // adds the two half-trees, the top level of the block's error tree (canon::four_over_six_block's bits).
        const bool even = (g & 1) == 0;
        const float o   = __shfl_xor_sync(kAll, a, 1);
        const canon::FourOverSixWords w = canon::four_over_six_words(o > a ? o : a, row);
        std::uint8_t c6[8], c4[8];
        const float e6 = canon::four_over_six_half(v, w.six, row, c6);
        const float e4 = canon::four_over_six_half(v, w.four, row, c4);
        const float p6 = __shfl_xor_sync(kAll, e6, 1), p4 = __shfl_xor_sync(kAll, e4, 1);
        const float err6 = even ? canon::add_rn(e6, p6) : canon::add_rn(p6, e6);
        const float err4 = even ? canon::add_rn(e4, p4) : canon::add_rn(p4, e4);
        const bool four  = err4 < err6;
        std::uint32_t packed = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const std::uint32_t lo = four ? c4[2 * i] : c6[2 * i];
            const std::uint32_t hi = four ? c4[2 * i + 1] : c6[2 * i + 1];
            packed |= (lo | (hi << 4)) << (8 * i);
        }
        std::uint8_t* base = rows.rows + static_cast<std::size_t>(t) * kNvfp4RowBytes;
        *reinterpret_cast<std::uint32_t*>(base + 4 * g) = packed;
        if (even) { base[kHidden / 2 + g / 2] = four ? w.four : w.six; }
    } else {
#pragma unroll
        for (int offset = 1; offset <= 2; offset <<= 1) {
            const float o = __shfl_xor_sync(kAll, a, offset);
            a             = o > a ? o : a;
            nan           = __shfl_xor_sync(kAll, nan ? 1 : 0, offset) != 0 || nan;
        }
        const std::uint32_t e = canon::mxfp8_exponent(a, nan);
        std::uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            lo |= std::uint32_t{canon::mxfp8_code(v[j], e)} << (8 * j);
            hi |= std::uint32_t{canon::mxfp8_code(v[j + 4], e)} << (8 * j);
        }
        std::uint8_t* base = rows.rows + static_cast<std::size_t>(t) * kMxfp8RowBytes;
        *reinterpret_cast<uint2*>(base + 8 * g) = make_uint2(lo, hi);
        if ((g & 3) == 0) { base[kHidden + g / 4] = static_cast<std::uint8_t>(e); }
    }
}

// ---------------------------------------------------------------------------- kernels

__global__ void __launch_bounds__(kThreads) embed_kernel(const std::int32_t* tokens, const std::uint16_t* embedding,
                                                         const std::uint16_t* norm, std::uint16_t* out) {
    __shared__ float s1[1][kWarpsH];
    const int t = static_cast<int>(blockIdx.x), c = 8 * static_cast<int>(threadIdx.x);
    float w[8];
    weight8(norm + c, w);
    pdl::enter();
    const std::int32_t token = tokens[t];
    float x[8];
    weight8(embedding + static_cast<std::size_t>(token) * kHidden + c, x);
    float sq[1] = {fold8(x, x)};
    cta_sum<1>(sq, s1);
    const float r = rms_factor(sq[0], static_cast<float>(kHidden));
#pragma unroll
    for (int j = 0; j < 8; ++j) { x[j] = canon::mul_rn(w[j], canon::mul_rn(x[j], r)); }
    store8(out + static_cast<std::size_t>(t) * kHidden + c, x);
}

// visit_entry and final_mix. Layer0: the input unmixed; NQ: the attention mix's queries; Messages: loop passes 2-3.
template <int NQ, bool Messages, bool Layer0>
__global__ void __launch_bounds__(kThreads) entry_kernel(const EntryArgs a) {
    constexpr int QA = NQ, QB = Messages ? 3 : 0;
    using M          = Mix<QA, QB>;
    __shared__ float s1[M::kS][kWarpsH];
    __shared__ float s2[M::kD][kWarpsH];
    __shared__ float s3[NQ + QB][kWarpsH];
    const int t = static_cast<int>(blockIdx.x), c = 8 * static_cast<int>(threadIdx.x);
    const bool tail = a.tail.experts != nullptr;

    float conv[4][8];
    if (tail) {
#pragma unroll
        for (int lag = 0; lag < 4; ++lag) { weight8(a.tail.conv + lag * kHidden + c, conv[lag]); }
    }
    float q[M::kQ][8];
    if constexpr (!Layer0) {
#pragma unroll
        for (int i = 0; i < QA; ++i) { weight8(a.queries + i * kHidden + c, q[i]); }
    }
    if constexpr (Messages) {
#pragma unroll
        for (int k = 0; k < 3; ++k) { weight8(a.message_queries + k * kHidden + c, q[QA + k]); }
    }
    float nw[NQ][8];
    if constexpr (!Messages) {
#pragma unroll
        for (int i = 0; i < NQ; ++i) { weight8(a.norms[i] + c, nw[i]); }
    }
    // The tail's column and its ring history (rows of earlier calls) before the wait when early_history; a call with a
    // history snapshot (a prompt) reads them after it.
    const bool early = tail && early_history(a.tail.ring);
    Column col{};
    uint4 hist[3] = {};
    if (early) {
        col = column_of(a.map, t);
        ring_history(a.tail.ring, a.map, col, c, hist);
    }
    pdl::enter();

    // Every row the column reads, issued before the first use: the tail's rows, the input, the blocks and the
    // message sources. A source that is the tail's own output (a pass restart's second message source) comes from
    // the values computed here, never from memory.
    const std::size_t row = static_cast<std::size_t>(t) * kHidden + c;
    TailRows trows{};
    if (tail) { trows = tail_rows(a.tail, t, c); }
    uint4 rin{};
    if (a.input != nullptr) { rin = raw8(a.input + row); }
    uint4 rblk[kPartial] = {};
    bool blk_tail[kPartial] = {};
    if constexpr (!Layer0) {
#pragma unroll
        for (int s = 0; s < kPartial; ++s) {
            blk_tail[s] = tail && a.blocks.rows[s] == a.tail_out;
            if (s < a.blocks.count && !blk_tail[s]) { rblk[s] = raw8(a.blocks.rows[s] + row); }
        }
    }
    uint4 rmsg[2]    = {};
    bool msg_tail[2] = {};
    if constexpr (Messages) {
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            msg_tail[s] = tail && a.message_sources[s] == a.tail_out;
            if (!msg_tail[s]) { rmsg[s] = raw8(a.message_sources[s] + row); }
        }
    }
    if (tail && !early) {
        col = column_of(a.map, t);
        ring_history(a.tail.ring, a.map, col, c, hist);
    }
    float h[8], tail_h[8] = {};
    if (tail) {
        mlp_tail(a.tail, a.map, col, c, conv, trows, hist, tail_h);
        store8(a.tail_out + row, tail_h);
#pragma unroll
        for (int j = 0; j < 8; ++j) { h[j] = tail_h[j]; }
    }
    if (a.input != nullptr) { unpack8(rin, h); }

    float out[M::kQ][8];
    if constexpr (Layer0) {
#pragma unroll
        for (int j = 0; j < 8; ++j) { out[0][j] = h[j]; }
    } else {
        float src[M::kS][8];
#pragma unroll
        for (int s = 0; s < kPartial; ++s) {
            if (s < a.blocks.count) {
                if (blk_tail[s]) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) { src[s][j] = tail_h[j]; }
                } else {
                    unpack8(rblk[s], src[s]);
                }
            } else {
                zero8(src[s]);
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) { src[kPartial][j] = h[j]; }
        if constexpr (Messages) {
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                if (msg_tail[s]) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) { src[kMaxSources + s][j] = tail_h[j]; }
                } else {
                    unpack8(rmsg[s], src[kMaxSources + s]);
                }
            }
        }
        mix_columns<QA, QB>(src, a.blocks.count, q, out, s1, s2);
    }

    float sq[NQ + QB];
#pragma unroll
    for (int i = 0; i < NQ + QB; ++i) { sq[i] = fold8(out[i], out[i]); }
    cta_sum<NQ + QB>(sq, s3);
    if constexpr (!Messages) {
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
            const float r = rms_factor(sq[i], static_cast<float>(kHidden));
            float x[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) { x[j] = canon::mul_rn(nw[i][j], canon::mul_rn(out[i][j], r)); }
            if constexpr (NQ == 1) {
#pragma unroll
                for (int k = 0; k < 3; ++k) {
                    if (a.x[k] != nullptr) { store8(a.x[k] + row, x); }
                    if (a.x8[k] != nullptr) {
                        a8_row8(x, reinterpret_cast<std::uint8_t*>(a.x8[k] + static_cast<std::size_t>(t) * a.x8_stride));
                    }
                }
            } else {
                store8(a.x[i] + row, x);
                if (a.x8[i] != nullptr) {
                    a8_row8(x, reinterpret_cast<std::uint8_t*>(a.x8[i] + static_cast<std::size_t>(t) * a.x8_stride));
                }
            }
        }
    } else {
        const std::size_t plane = static_cast<std::size_t>(gridDim.x) * kHidden;
#pragma unroll
        for (int i = 0; i < NQ; ++i) { store8(a.mixed + i * plane + row, out[i]); }
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            store8(a.message + k * plane + row, out[NQ + k]);
            const int xi  = NQ == 1 ? 0 : k;
            const float rx = rms_factor(sq[xi], static_cast<float>(kHidden));
            const float rm = rms_factor(sq[NQ + k], static_cast<float>(kHidden));
            float gx[8], gm[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                gx[j] = canon::mul_rn(out[xi][j], rx);
                gm[j] = canon::mul_rn(out[NQ + k][j], rm);
            }
            std::uint16_t* gi = a.gate_input + (static_cast<std::size_t>(k) * gridDim.x + t) * (2 * kHidden);
            store8(gi + c, gx);
            store8(gi + kHidden + c, gm);
        }
    }
}

template <bool Messages>
__global__ void __launch_bounds__(kThreads) mlp_entry_kernel(const MlpEntryArgs a) {
    constexpr int QB = Messages ? 1 : 0;
    using M          = Mix<1, QB>;
    __shared__ float s1[M::kS][kWarpsH];
    __shared__ float s2[M::kD][kWarpsH];
    __shared__ float s3[1 + QB][kWarpsH];
    __shared__ float s4[kWarpsH];
    const int t = static_cast<int>(blockIdx.x), c = 8 * static_cast<int>(threadIdx.x);

    float conv[4][8];
#pragma unroll
    for (int lag = 0; lag < 4; ++lag) { weight8(a.conv + lag * kHidden + c, conv[lag]); }
    float q[M::kQ][8];
    weight8(a.query + c, q[0]);
    if constexpr (Messages) { weight8(a.message_query + c, q[1]); }
    float nw[8];
    if constexpr (!Messages) { weight8(a.norm + c, nw); }
    // The column and its ring history (rows of earlier calls) before the wait when early_history; a call with a history
    // snapshot (a prompt) reads them after it.
    const bool early = early_history(a.ring);
    Column col{};
    uint4 hist[3] = {};
    if (early) {
        col = column_of(a.map, t);
        ring_history(a.ring, a.map, col, c, hist);
    }
    pdl::enter();

    // Every row the column reads, issued before the first use (none aliases what this kernel writes: the ring is
    // state, cand FP32).
    const std::size_t row = static_cast<std::size_t>(t) * kHidden + c;
    const uint4 ro = raw8(a.o + row);
    uint4 rh{};
    if (!a.boundary) { rh = raw8(a.h + row); }
    uint4 rblk[kPartial] = {};
#pragma unroll
    for (int s = 0; s < kPartial; ++s) {
        if (s < a.blocks.count) { rblk[s] = raw8(a.blocks.rows[s] + row); }
    }
    uint4 rmsg[2] = {};
    if constexpr (Messages) {
#pragma unroll
        for (int s = 0; s < 2; ++s) { rmsg[s] = raw8(a.message_sources[s] + row); }
    }
    if (!early) {
        col = column_of(a.map, t);
        ring_history(a.ring, a.map, col, c, hist);
    }
    // The post-attention conv of W_o's output (the call holds every column's o, so in-call history is a load).
    float x[4][8];
    unpack8(ro, x[0]);
#pragma unroll
    for (int k = 1; k < 4; ++k) {
        switch (history_source(a.map, col, k)) {
        case Source::Zero: zero8(x[k]); break;
        case Source::Call: load8(a.o + row - static_cast<std::size_t>(k) * kHidden, x[k]); break;
        case Source::Ring: unpack8(hist[k - 1], x[k]); break;
        }
    }
    float attn[8];
    conv8(conv, x, attn);
    if (writes_ring(a.map, col)) { store8(ring_row(a.ring, col.lane, col.pos) + c, x[0]); }

    float cand[8];
    if (a.boundary) {
#pragma unroll
        for (int j = 0; j < 8; ++j) { cand[j] = attn[j]; }
    } else {
        float h[8];
        unpack8(rh, h);
#pragma unroll
        for (int j = 0; j < 8; ++j) { cand[j] = canon::add_rn(h[j], attn[j]); }
    }
    store8(a.cand + row, cand);

    float src[M::kS][8];
#pragma unroll
    for (int s = 0; s < kPartial; ++s) {
        if (s < a.blocks.count) {
            unpack8(rblk[s], src[s]);
        } else {
            zero8(src[s]);
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) { src[kPartial][j] = cand[j]; }
    if constexpr (Messages) {
#pragma unroll
        for (int s = 0; s < 2; ++s) { unpack8(rmsg[s], src[kMaxSources + s]); }
    }
    float out[M::kQ][8];
    mix_columns<1, QB>(src, a.blocks.count, q, out, s1, s2);

    float sq[1 + QB];
#pragma unroll
    for (int i = 0; i < 1 + QB; ++i) { sq[i] = fold8(out[i], out[i]); }
    cta_sum<1 + QB>(sq, s3);
    if constexpr (!Messages) {
        const float r = rms_factor(sq[0], static_cast<float>(kHidden));
        float xm[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { xm[j] = canon::mul_rn(nw[j], canon::mul_rn(out[0][j], r)); }
        round8(xm);
        store8(a.x + row, xm);
        if (a.rows.rows != nullptr) { quantize_row(xm, a.rows, t, s4); }
    } else {
        store8(a.mixed + row, out[0]);
        store8(a.message + row, out[1]);
        const float rx = rms_factor(sq[0], static_cast<float>(kHidden));
        const float rm = rms_factor(sq[1], static_cast<float>(kHidden));
        float gx[8], gm[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            gx[j] = canon::mul_rn(out[0][j], rx);
            gm[j] = canon::mul_rn(out[1][j], rm);
        }
        std::uint16_t* gi = a.gate_input + static_cast<std::size_t>(t) * (2 * kHidden);
        store8(gi + c, gx);
        store8(gi + kHidden + c, gm);
    }
}

// One CTA per (column, kind): a kind's gating and norm need only its own vectors, so the kinds run side by side.
__global__ void __launch_bounds__(kThreads) gate_exit_kernel(const GateExitArgs a) {
    __shared__ float s1[1][kWarpsH];
    __shared__ float s2[kWarpsH];
    const int t = static_cast<int>(blockIdx.x), k = static_cast<int>(blockIdx.y), c = 8 * static_cast<int>(threadIdx.x);
    // Selects, not a parameter array indexed at run time (which would copy the parameters to local memory).
    const std::uint16_t* norm = k == 0 ? a.norms[0] : (k == 1 ? a.norms[1] : a.norms[2]);
    std::uint16_t* xout       = k == 0 ? a.x[0] : (k == 1 ? a.x[1] : a.x[2]);
    float nw[8];
    weight8(norm + c, nw);
    pdl::enter();

    const int columns       = static_cast<int>(gridDim.x);
    const std::size_t plane = static_cast<std::size_t>(columns) * kHidden;
    const std::size_t row   = static_cast<std::size_t>(t) * kHidden + c;
    float x[8], xv[8], m[8], f[8], g[8];
    load8(a.mixed + (a.shared_x ? 0 : k) * plane + row, xv);
    load8(a.message + k * plane + row, m);
    const float* fg = a.fg + (static_cast<std::size_t>(k) * columns + t) * (2 * kHidden);
    load8(fg + c, f);
    load8(fg + kHidden + c, g);
#pragma unroll
    for (int j = 0; j < 8; ++j) { x[j] = message_gate(xv[j], m[j], f[j], g[j]); }
    float sq[1] = {fold8(x, x)};
    cta_sum<1>(sq, s1);
    const float r = rms_factor(sq[0], static_cast<float>(kHidden));
#pragma unroll
    for (int j = 0; j < 8; ++j) { x[j] = canon::mul_rn(nw[j], canon::mul_rn(x[j], r)); }
    round8(x);
    store8(xout + row, x);
    std::int32_t* x8 = k == 0 ? a.x8[0] : (k == 1 ? a.x8[1] : a.x8[2]);
    if (x8 != nullptr) { a8_row8(x, reinterpret_cast<std::uint8_t*>(x8 + static_cast<std::size_t>(t) * a.x8_stride)); }
    if (a.rows.rows != nullptr) { quantize_row(x, a.rows, t, s2); }
}

__global__ void softcap_kernel(float* logits, std::int64_t count, float cap) {
    pdl::enter();
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
        logits[i] = canon::softcap_c(logits[i], cap);
    }
}

// ---------------------------------------------------------------------------- host side

template <class P>
bool aligned16(const P* p) {
    return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
}

void check_columns(std::int32_t columns, const char* op) {
    if (columns <= 0 || columns > 65535) { throw std::invalid_argument(std::string(op) + ": columns must be in [1, 65535]"); }
}

void check_map(const ColumnMap& map, std::int32_t columns, const char* op) {
    if (map.lane == nullptr || map.position == nullptr || map.first == nullptr || map.columns != columns) {
        throw std::invalid_argument(std::string(op) + ": the column map is incomplete or not the call's width");
    }
}

void check_ring(const Ring& ring, std::int32_t width, const char* op) {
    if (ring.base == nullptr || ring.width != width || !aligned16(ring.base) ||
        ring.lane_stride < std::int64_t{kRingPositions} * width || ring.lane_stride % 8 != 0) {
        throw std::invalid_argument(std::string(op) + ": a conv ring of width " + std::to_string(width) +
                                    " and a 16-byte lane stride of at least its positions is needed");
    }
    if (ring.history != nullptr && (!aligned16(ring.history) || ring.history_lane_stride % 8 != 0 ||
                                    ring.history_lane_stride < std::int64_t{kHistorySlots} * width)) {
        throw std::invalid_argument(std::string(op) + ": a ring's history needs a 16-byte lane stride of its slots");
    }
}

void check_blocks(const Blocks& blocks, const char* op) {
    if (blocks.count < 0 || blocks.count > kMaxSources - 1) {
        throw std::invalid_argument(std::string(op) + ": a mix takes at most 4 completed blocks");
    }
    for (int s = 0; s < blocks.count; ++s) {
        if (blocks.rows[s] == nullptr || !aligned16(blocks.rows[s])) {
            throw std::invalid_argument(std::string(op) + ": a block source is missing or misaligned");
        }
    }
}

// The experts' rows: checkpoint rows (NVFP4 with its amax) or, with emax, NVFP4 A16 limb rows, or NVFP4 A8 rows.
void check_rows(const ExpertRows& rows, const char* op) {
    if (rows.rows == nullptr || reinterpret_cast<std::uintptr_t>(rows.rows) % 8 != 0 ||
        (rows.format == moe::ExpertFormat::Nvfp4FourOverSix && rows.amax == nullptr && rows.emax == nullptr && !rows.a8) ||
        (rows.emax != nullptr && rows.format != moe::ExpertFormat::Nvfp4FourOverSix) ||
        (rows.a8 && (rows.format != moe::ExpertFormat::Nvfp4FourOverSix || rows.emax != nullptr))) {
        throw std::invalid_argument(std::string(op) + ": the experts' rows (and NVFP4's amax, or A16's exponents) are needed");
    }
}

pdl::LaunchConfig columns_launch(std::int32_t columns, cudaStream_t stream) {
    return {dim3(static_cast<unsigned>(columns)), dim3(kThreads), 0, stream};
}

void launch_entry(const EntryArgs& a, cudaStream_t stream) {
    const bool messages = a.message_queries != nullptr;
    const auto config   = columns_launch(a.columns, stream);
    if (a.queries == nullptr) {
        CUDA_CHECK(pdl::launch_consumer(config, entry_kernel<1, false, true>, a));
    } else if (a.nq == 1) {
        CUDA_CHECK(messages ? pdl::launch_consumer(config, entry_kernel<1, true, false>, a)
                            : pdl::launch_consumer(config, entry_kernel<1, false, false>, a));
    } else {
        CUDA_CHECK(messages ? pdl::launch_consumer(config, entry_kernel<3, true, false>, a)
                            : pdl::launch_consumer(config, entry_kernel<3, false, false>, a));
    }
}

} // namespace

void embed(const std::int32_t* tokens, std::int32_t columns, const std::uint16_t* embedding, const std::uint16_t* norm,
           std::uint16_t* out, cudaStream_t stream) {
    check_columns(columns, "inkling2 embed");
    if (tokens == nullptr || embedding == nullptr || norm == nullptr || out == nullptr || !aligned16(embedding) ||
        !aligned16(out)) {
        throw std::invalid_argument("inkling2 embed: tokens, a 16-byte aligned table, norm and output are needed");
    }
    CUDA_CHECK(pdl::launch_consumer(columns_launch(columns, stream), embed_kernel, tokens, embedding, norm, out));
}

// --a8-activations' x8 rows: 16-byte aligned, at least a column's codes and scales apart.
void check_x8(std::int32_t* const (&x8)[3], std::int32_t stride, const char* op) {
    for (const std::int32_t* p : x8) {
        if (p != nullptr && (reinterpret_cast<std::uintptr_t>(p) % 16 != 0 || stride * 4 < kHidden + kHidden / 8)) {
            throw std::invalid_argument(std::string(op) + ": 8-bit rows need 16-byte alignment and a column's 2,592 bytes");
        }
    }
}

void visit_entry(const EntryArgs& a, cudaStream_t stream) {
    constexpr const char* op = "inkling2 visit_entry";
    check_columns(a.columns, op);
    check_map(a.map, a.columns, op);
    check_x8(a.x8, a.x8_stride, op);
    const bool tail = a.tail.experts != nullptr;
    if (tail) {
        check_ring(a.tail.ring, kHidden, op);
        if (a.tail.cand == nullptr || a.tail.conv == nullptr || a.tail_out == nullptr) {
            throw std::invalid_argument("inkling2 visit_entry: a tail needs its cand, conv taps and output");
        }
    }
    if (a.input == nullptr && !tail) { throw std::invalid_argument("inkling2 visit_entry: the visit needs an input"); }
    if (a.queries != nullptr) {
        check_blocks(a.blocks, op);
        if (a.nq != 1 && a.nq != 3) { throw std::invalid_argument("inkling2 visit_entry: 1 or 3 mix queries"); }
    } else if (a.message_queries != nullptr) {
        throw std::invalid_argument("inkling2 visit_entry: layer 0 has no loop messages");
    }
    if (a.message_queries != nullptr) {
        if (a.message_sources[0] == nullptr || a.message_sources[1] == nullptr || a.mixed == nullptr ||
            a.message == nullptr || a.gate_input == nullptr) {
            throw std::invalid_argument("inkling2 visit_entry: messages need two sources and three outputs");
        }
    } else {
        const int norms = a.queries == nullptr ? 1 : a.nq;
        for (int i = 0; i < norms; ++i) {
            if (a.norms[i] == nullptr) { throw std::invalid_argument("inkling2 visit_entry: a norm is missing"); }
        }
        if (norms == 3 && (a.x[0] == nullptr || a.x[1] == nullptr || a.x[2] == nullptr)) {
            throw std::invalid_argument("inkling2 visit_entry: a full layer writes x_q, x_k and x_v");
        }
        if (a.x[0] == nullptr && a.x[1] == nullptr && a.x[2] == nullptr) {
            throw std::invalid_argument("inkling2 visit_entry: no output");
        }
    }
    launch_entry(a, stream);
}

void mlp_entry(const MlpEntryArgs& a, cudaStream_t stream) {
    constexpr const char* op = "inkling2 mlp_entry";
    check_columns(a.columns, op);
    check_map(a.map, a.columns, op);
    check_ring(a.ring, kHidden, op);
    check_blocks(a.blocks, op);
    if (a.o == nullptr || a.conv == nullptr || (!a.boundary && a.h == nullptr) || a.cand == nullptr ||
        a.query == nullptr) {
        throw std::invalid_argument("inkling2 mlp_entry: o, the conv taps, h, cand and the query are needed");
    }
    const auto config = columns_launch(a.columns, stream);
    if (a.message_query != nullptr) {
        if (a.message_sources[0] == nullptr || a.message_sources[1] == nullptr || a.mixed == nullptr ||
            a.message == nullptr || a.gate_input == nullptr) {
            throw std::invalid_argument("inkling2 mlp_entry: messages need two sources and three outputs");
        }
        CUDA_CHECK(pdl::launch_consumer(config, mlp_entry_kernel<true>, a));
    } else {
        if (a.norm == nullptr || a.x == nullptr) { throw std::invalid_argument("inkling2 mlp_entry: norm and x"); }
        if (a.rows.rows != nullptr) { check_rows(a.rows, op); } // none: an MXFP8 layer under --a16-activations
        CUDA_CHECK(pdl::launch_consumer(config, mlp_entry_kernel<false>, a));
    }
}

void gate_exit(const GateExitArgs& a, cudaStream_t stream) {
    constexpr const char* op = "inkling2 gate_exit";
    check_columns(a.columns, op);
    check_x8(a.x8, a.x8_stride, op);
    if (a.kinds != 1 && a.kinds != 3) { throw std::invalid_argument("inkling2 gate_exit: 1 or 3 kinds"); }
    if (a.mixed == nullptr || a.message == nullptr || a.fg == nullptr) {
        throw std::invalid_argument("inkling2 gate_exit: mixed, message and the projections are needed");
    }
    for (int k = 0; k < a.kinds; ++k) {
        if (a.norms[k] == nullptr || a.x[k] == nullptr) {
            throw std::invalid_argument("inkling2 gate_exit: a kind's norm or output is missing");
        }
    }
    if (a.rows.rows != nullptr) {
        if (a.kinds != 1) { throw std::invalid_argument("inkling2 gate_exit: the experts' rows belong to the mlp kind"); }
        check_rows(a.rows, op);
    }
    const pdl::LaunchConfig config{dim3(static_cast<unsigned>(a.columns), static_cast<unsigned>(a.kinds)), dim3(kThreads),
                                   0, stream};
    CUDA_CHECK(pdl::launch_consumer(config, gate_exit_kernel, a));
}

void final_mix(const FinalArgs& f, cudaStream_t stream) {
    EntryArgs a;
    a.columns  = f.columns;
    a.map      = f.map;
    a.tail     = f.tail;
    a.tail_out = f.tail_out;
    a.blocks   = f.blocks;
    a.queries  = f.query;
    a.nq       = 1;
    a.norms[0] = f.norm;
    a.x[0]     = f.out;
    if (f.tail.experts == nullptr || f.query == nullptr) {
        throw std::invalid_argument("inkling2 final_mix: the last visit's tail and the output query are needed");
    }
    visit_entry(a, stream);
}

namespace {

// Block (ring, lag): the ring's row of position `position - lag - 1` to its history slot.
__global__ void conv_history_kernel(const ConvHistoryArgs a) {
    const int ring = static_cast<int>(blockIdx.x), pos = a.position - static_cast<int>(blockIdx.y) - 1;
    if (pos < a.first) { return; }
    const std::int64_t* d = a.rings + 3 * ring;
    const auto width      = static_cast<int>(d[2]);
    const auto* src = reinterpret_cast<const uint4*>(a.image + d[0] + static_cast<std::int64_t>(pos & (kRingPositions - 1)) * width);
    auto* dst       = reinterpret_cast<uint4*>(a.history + d[1] + static_cast<std::int64_t>(pos & (kHistorySlots - 1)) * width);
    for (int i = static_cast<int>(threadIdx.x); i < width / 8; i += static_cast<int>(blockDim.x)) { dst[i] = src[i]; }
}

} // namespace

void conv_history(const ConvHistoryArgs& a, cudaStream_t stream) {
    if (a.rings == nullptr || a.count <= 0 || a.image == nullptr || a.history == nullptr || !aligned16(a.image) ||
        !aligned16(a.history) || a.first < 0 || a.position < a.first) {
        throw std::invalid_argument("inkling2 conv_history: the rings, the image, the history and a position are needed");
    }
    conv_history_kernel<<<dim3(static_cast<unsigned>(a.count), kHistorySlots - 1), 128, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

void softcap(float* logits, std::int64_t count, float cap, cudaStream_t stream) {
    if (logits == nullptr || count <= 0) { throw std::invalid_argument("inkling2 softcap: logits are needed"); }
    const std::int64_t blocks = std::min<std::int64_t>((count + 255) / 256, 4 * 170);
    CUDA_CHECK(pdl::launch_consumer({dim3(static_cast<unsigned>(blocks)), dim3(256), 0, stream}, softcap_kernel, logits,
                                    count, cap));
}

} // namespace infernix::ops::inkling2
