// Inkling 2's decode and verify attention (include/infernix/ops/inkling2_attention.h).
//
// prologue_kernel: a CTA per (column, KV head) of 5 warps: K, V, and the group's 3 query heads (q and r). Lane l
//   holds elements 4l .. 4l + 3 of a head, so a head sum is the pinned order (a lane's 4 folded by fma, the bfly32
//   tree). Padding columns write zero outputs and nothing else.
// attention_kernel: a CTA per (split, KV head, lane) of 16 warps over its rows (3 heads x the lane's columns) and the
//   union of its rows' units; per unit the K and V pages are copied to shared memory, then scores (a thread per
//   (row group, key), its rows' dot products as 4 fold chains), the unit softmax (a warp per row), P x V (a thread
//   per (row group, 4 dims)) and the split's merge. The CTA completing its (lane, KV head) counter combines every
//   row's splits.

#include "infernix/ops/inkling2_attention.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/canonical_math.h"
#include "ops/inkling2/column.cuh"
#include "ops/inkling2/visit_math.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace infernix::ops::inkling2 {
namespace {

using detail::Column;
using detail::column_of;
using detail::history_source;
using detail::ring_row;
using detail::Source;

constexpr unsigned kAll        = 0xFFFFFFFFU;
constexpr int kD               = kHeadDim;
constexpr int kKvWidth         = kKvHeads * kD;     // 1024
constexpr int kQWidth          = kHeads * kD;       // 3072
constexpr int kRWidth          = kHeads * kRelDim;  // 384
constexpr int kStatWords       = 132;
constexpr int kAccOffset       = 4;
constexpr float kInvSqrtD      = static_cast<float>(0.08838834764831845);
constexpr float kLog2e         = static_cast<float>(1.4426950408889634);
constexpr float kClipQK        = static_cast<float>(22.627416997969522);
constexpr float kClipR         = 8.0F;

__device__ __forceinline__ float bf16f(std::uint16_t v) { return canon::bf16_to_f32(v); }
__device__ __forceinline__ float neg_inf() { return canon::f32_from_bits(0xFF800000U); }

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void unpack4(uint2 w, float (&v)[4]) {
    v[0] = canon::f32_from_bits(w.x << 16);
    v[1] = canon::f32_from_bits(w.x & 0xFFFF0000U);
    v[2] = canon::f32_from_bits(w.y << 16);
    v[3] = canon::f32_from_bits(w.y & 0xFFFF0000U);
}
__device__ __forceinline__ void load4(const std::uint16_t* p, float (&v)[4]) {
    unpack4(*reinterpret_cast<const uint2*>(p), v);
}
__device__ __forceinline__ void weight4(const std::uint16_t* p, float (&v)[4]) {
    unpack4(__ldg(reinterpret_cast<const uint2*>(p)), v);
}
__device__ __forceinline__ uint2 pack4(const float (&v)[4]) {
    return make_uint2(std::uint32_t{canon::f32_to_bf16_rn(v[0])} | (std::uint32_t{canon::f32_to_bf16_rn(v[1])} << 16),
                      std::uint32_t{canon::f32_to_bf16_rn(v[2])} | (std::uint32_t{canon::f32_to_bf16_rn(v[3])} << 16));
}
__device__ __forceinline__ void round4(float (&v)[4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) { v[j] = bf16f(canon::f32_to_bf16_rn(v[j])); }
}
__device__ __forceinline__ std::uint16_t f16_bits(float x) { return __half_as_ushort(__float2half_rn(x)); }
__device__ __forceinline__ float f16f(std::uint16_t u) { return __half2float(__ushort_as_half(u)); }

// The pinned head sum of 128: a lane's 4 folded by fma, then the bfly32 tree.
__device__ __forceinline__ float head_sum(const float (&a)[4], const float (&b)[4]) {
    float acc = 0.0F;
#pragma unroll
    for (int j = 0; j < 4; ++j) { acc = canon::fma_rn(a[j], b[j], acc); }
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) { acc = canon::add_rn(acc, __shfl_xor_sync(kAll, acc, offset)); }
    return acc;
}

// The element offset of a position's row of 128 in a visit's K or V.
__device__ __forceinline__ std::size_t kv_row(const KvView& kv, int lane, int head, int pos) {
    if (kv.table == nullptr) {
        const int page = (pos / kKeyUnit) % kWindowRingPages;
        return lane * kv.lane_stride +
               ((static_cast<std::size_t>(head) * kWindowRingPages + page) * kKeyUnit + (pos % kKeyUnit)) * kD;
    }
    const std::int64_t page = kv.table[static_cast<std::int64_t>(lane) * kv.table_stride + pos / kKeyUnit];
    return ((static_cast<std::size_t>(head) * kv.pool_pages + page) * kKeyUnit + (pos % kKeyUnit)) * kD;
}

// ---------------------------------------------------------------------------- prologue

constexpr int kPrologueThreads = 5 * 32;

__global__ void __launch_bounds__(kPrologueThreads) prologue_kernel(const AttentionArgs a) {
    const int t = static_cast<int>(blockIdx.x), kv = static_cast<int>(blockIdx.y);
    const int warp = static_cast<int>(threadIdx.x >> 5), lane = static_cast<int>(threadIdx.x & 31U);
    const int ch   = kv * kD + 4 * lane; // the lane's K / V channels
    const int head = kv * kGroupHeads + (warp - 2);
    float w[4][4];
    float scale[4] = {};
    float rscale   = 0.0F;
    if (warp < 2) {
        const std::uint16_t* conv = warp == 0 ? a.layer.key_conv : a.layer.value_conv;
#pragma unroll
        for (int lag = 0; lag < 4; ++lag) { weight4(conv + lag * kKvWidth + ch, w[lag]); }
    } else {
        weight4(a.layer.query_scale + head * kD + 4 * lane, scale);
        if (lane < kRelDim) { rscale = bf16f(__ldg(a.layer.relative_scale + head * kRelDim + lane)); }
    }
    pdl::enter();

    const Column col = column_of(a.map, t);
    if (col.lane < 0) {
        for (int i = static_cast<int>(threadIdx.x); i < kGroupHeads * kD; i += kPrologueThreads) {
            a.out[static_cast<std::size_t>(t) * kQWidth + kv * kGroupHeads * kD + i] = 0;
        }
        return;
    }
    const float tau_p = a.layer.full ? tau(col.pos) : 1.0F;
    if (warp < 2) {
        const std::uint16_t* raw = warp == 0 ? a.in.k : a.in.v;
        const Ring& ring         = warp == 0 ? a.key_ring : a.value_ring;
        float x[4][4];
        load4(raw + static_cast<std::size_t>(t) * kKvWidth + ch, x[0]);
#pragma unroll
        for (int k = 1; k < 4; ++k) {
            switch (history_source(a.map, col, k)) {
            case Source::Zero:
#pragma unroll
                for (int j = 0; j < 4; ++j) { x[k][j] = 0.0F; }
                break;
            case Source::Call: load4(raw + static_cast<std::size_t>(t - k) * kKvWidth + ch, x[k]); break;
            case Source::Ring: load4(ring_row(ring, col.lane, col.pos - k) + ch, x[k]); break;
            }
        }
        float c[4];
        if (a.layer.full) {
            const std::uint16_t* taps = (warp == 0 ? a.in.key_taps : a.in.value_taps) +
                                        static_cast<std::size_t>(t) * (kKvHeads * 4) + kv * 4;
            const float tap[4] = {bf16f(taps[0]), bf16f(taps[1]), bf16f(taps[2]), bf16f(taps[3])};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float wj[4] = {w[0][j], w[1][j], w[2][j], w[3][j]};
                const float xj[4] = {x[0][j], x[1][j], x[2][j], x[3][j]};
                c[j]              = conv_dynamic(wj, tap, xj);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float wj[4] = {w[0][j], w[1][j], w[2][j], w[3][j]};
                const float xj[4] = {x[0][j], x[1][j], x[2][j], x[3][j]};
                c[j]              = conv_static(wj, xj);
            }
        }
        round4(c);
        const float sum = head_sum(c, c);
        const std::size_t at = kv_row(a.kv, col.lane, kv, col.pos) + 4 * lane;
        if (warp == 0) {
            const float f = clip_factor(sum, kClipQK);
#pragma unroll
            for (int j = 0; j < 4; ++j) { c[j] = canon::mul_rn(c[j], f); }
            *reinterpret_cast<uint2*>(a.kv.k + at) = pack4(c);
        } else {
            const float r = rms_factor(sum, static_cast<float>(kD));
            std::uint16_t h[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) { h[j] = f16_bits(bf16f(canon::f32_to_bf16_rn(canon::mul_rn(c[j], r)))); }
            *reinterpret_cast<uint2*>(a.kv.v + at) =
                make_uint2(std::uint32_t{h[0]} | (std::uint32_t{h[1]} << 16), std::uint32_t{h[2]} | (std::uint32_t{h[3]} << 16));
        }
        *reinterpret_cast<uint2*>(ring_row(ring, col.lane, col.pos) + ch) = pack4(x[0]);
        return;
    }
    // q: scaled, clipped, rounded; full visits times tau, rounded again.
    float q[4];
    load4(a.in.q + static_cast<std::size_t>(t) * kQWidth + head * kD + 4 * lane, q);
#pragma unroll
    for (int j = 0; j < 4; ++j) { q[j] = canon::mul_rn(q[j], scale[j]); }
    const float f = clip_factor(head_sum(q, q), kClipQK);
#pragma unroll
    for (int j = 0; j < 4; ++j) { q[j] = canon::mul_rn(q[j], f); }
    round4(q);
    if (a.layer.full) {
#pragma unroll
        for (int j = 0; j < 4; ++j) { q[j] = canon::mul_rn(q[j], tau_p); }
        round4(q);
    }
    *reinterpret_cast<uint2*>(a.ws.q + (static_cast<std::size_t>(t) * kHeads + head) * kD + 4 * lane) = pack4(q);
    // r: the fold of its 16 squares in order.
    float rv = lane < kRelDim ? canon::mul_rn(bf16f(a.in.r[static_cast<std::size_t>(t) * kRWidth + head * kRelDim + lane]),
                                              rscale)
                              : 0.0F;
    float sum = 0.0F;
#pragma unroll
    for (int i = 0; i < kRelDim; ++i) {
        const float v = __shfl_sync(kAll, rv, i);
        sum           = canon::fma_rn(v, v, sum);
    }
    rv = bf16f(canon::f32_to_bf16_rn(canon::mul_rn(rv, clip_factor(sum, kClipR))));
    if (a.layer.full) { rv = bf16f(canon::f32_to_bf16_rn(canon::mul_rn(rv, tau_p))); }
    if (lane < kRelDim) {
        a.ws.r[(static_cast<std::size_t>(t) * kHeads + head) * kRelDim + lane] = canon::f32_to_bf16_rn(rv);
    }
}

// ---------------------------------------------------------------------------- attention

constexpr int kThreads    = 512;
constexpr int kWarps      = kThreads / 32;
constexpr int kRowStride  = kD + 8; // shared rows of 128 BF16/FP16 words, 16 bytes of padding
constexpr int kMaxRows    = kGroupHeads * kMaxAttentionLaneColumns;
constexpr int kScoreRows  = kMaxRows / (kThreads / kKeyUnit); // rows a score thread holds: 6
constexpr int kPvRows     = kMaxRows / (kThreads / 32);       // rows a P x V thread holds: 3

struct RowState {
    int t, head, pos, start, end, slot0; // the row's units of this split: [start, end] (start > end: none)
    float m, l, c, cu;
    int act; // this unit: 0 none, 1 the split's first visible unit, 2 a merge
};

struct Smem {
    std::uint16_t* k;  // [64][kRowStride]
    std::uint16_t* v;  // [64][kRowStride]
    std::uint16_t* q;  // [rows][kRowStride]
    float* r;          // [rows][16]
    float* p;          // [rows][64]
    RowState* rows;    // [rows]
};

std::size_t smem_bytes(int rows) {
    return 2 * std::size_t{kKeyUnit} * kRowStride * 2 + std::size_t(rows) * kRowStride * 2 +
           std::size_t(rows) * kRelDim * 4 + std::size_t(rows) * kKeyUnit * 4 + std::size_t(rows) * sizeof(RowState);
}

__device__ __forceinline__ Smem carve(unsigned char* base, int rows) {
    Smem s;
    s.k    = reinterpret_cast<std::uint16_t*>(base);
    s.v    = s.k + kKeyUnit * kRowStride;
    s.q    = s.v + kKeyUnit * kRowStride;
    s.r    = reinterpret_cast<float*>(s.q + rows * kRowStride);
    s.p    = s.r + rows * kRelDim;
    s.rows = reinterpret_cast<RowState*>(s.p + rows * kKeyUnit);
    return s;
}

__device__ __forceinline__ void cp_async16(void* smem, const void* global) {
    const auto s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(global) : "memory");
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

// The unit range of a row (its window's units, or all of a full row's) and, for full rows, its split geometry.
struct RowUnits {
    int lo, hi;
};
__device__ __forceinline__ RowUnits row_units(bool full, int pos, int first) {
    const int lo = full ? first / kKeyUnit : max(first, max(pos - (kWindow - 1), 0)) / kKeyUnit;
    return {lo, pos / kKeyUnit};
}
__device__ __forceinline__ int full_split_units(int n) { return (n + kFullSplits - 1) / kFullSplits; }

__global__ void __launch_bounds__(kThreads, 1) attention_kernel(const AttentionArgs a) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ int range[2];
    __shared__ int last;
    const int split = static_cast<int>(blockIdx.x), kv = static_cast<int>(blockIdx.y), lane = static_cast<int>(blockIdx.z);
    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, wl = tid & 31;
    const bool full = a.layer.full;
    pdl::enter();
    const int c0 = a.lane_columns[2 * lane], count = a.lane_columns[2 * lane + 1];
    if (count == 0) { return; }
    const int rows  = kGroupHeads * count;
    const int first = a.map.first[lane];
    const Smem s    = carve(smem_raw, kGroupHeads * a.max_lane_columns);

    // Rows: their units of this split.
    const int lane_lo = row_units(full, a.map.position[c0], first).lo;
    if (tid < rows) {
        RowState& r = s.rows[tid];
        r.t         = c0 + tid / kGroupHeads;
        r.head      = kv * kGroupHeads + tid % kGroupHeads;
        r.pos       = a.map.position[r.t];
        const RowUnits u = row_units(full, r.pos, first);
        if (full) {
            const int bs = full_split_units(u.hi - u.lo + 1);
            r.start      = u.lo + split * bs;
            r.end        = min(r.start + bs - 1, u.hi);
            r.slot0      = split;
        } else {
            const int b = lane_lo + split;
            r.start     = b >= u.lo && b <= u.hi ? b : 1;
            r.end       = b >= u.lo && b <= u.hi ? b : 0;
            r.slot0     = b - u.lo;
        }
        r.m = neg_inf(), r.l = 0.0F, r.c = 0.0F, r.cu = 0.0F, r.act = 0;
    }
    __syncthreads();
    if (tid == 0) {
        int lo = 0x7FFFFFFF, hi = -1;
        for (int i = 0; i < rows; ++i) {
            if (s.rows[i].start <= s.rows[i].end) {
                lo = min(lo, s.rows[i].start);
                hi = max(hi, s.rows[i].end);
            }
        }
        range[0] = lo, range[1] = hi;
    }
    // The rows' q and r.
    for (int i = tid; i < rows * (kD / 8); i += kThreads) {
        const int row = i / (kD / 8), chunk = i % (kD / 8);
        const int t = c0 + row / kGroupHeads, head = kv * kGroupHeads + row % kGroupHeads;
        *reinterpret_cast<uint4*>(s.q + row * kRowStride + 8 * chunk) =
            *reinterpret_cast<const uint4*>(a.ws.q + (static_cast<std::size_t>(t) * kHeads + head) * kD + 8 * chunk);
    }
    for (int i = tid; i < rows * kRelDim; i += kThreads) {
        const int row = i / kRelDim, e = i % kRelDim;
        const int t = c0 + row / kGroupHeads, head = kv * kGroupHeads + row % kGroupHeads;
        s.r[i] = bf16f(a.ws.r[(static_cast<std::size_t>(t) * kHeads + head) * kRelDim + e]);
    }
    __syncthreads();

    // P x V accumulators of this thread: rows pv_rg + 16 m, dims 4 dc .. 4 dc + 3.
    const int dc = tid & 31, pv_rg = tid >> 5;
    float acc[kPvRows][4] = {};
    const int key = tid & (kKeyUnit - 1), sc_rg = tid >> 6;

    for (int b = range[0]; b <= range[1]; ++b) {
        // The unit's K and V pages.
        const std::size_t page = kv_row(a.kv, lane, kv, b * kKeyUnit);
        for (int i = tid; i < 2 * kKeyUnit * (kD / 8); i += kThreads) {
            const int plane = i / (kKeyUnit * (kD / 8)), rest = i % (kKeyUnit * (kD / 8));
            const int row = rest / (kD / 8), chunk = rest % (kD / 8);
            const std::uint16_t* src = (plane == 0 ? a.kv.k : a.kv.v) + page + row * kD + 8 * chunk;
            cp_async16((plane == 0 ? s.k : s.v) + row * kRowStride + 8 * chunk, src);
        }
        cp_async_wait_all();
        __syncthreads();

        // Scores: thread (sc_rg, key) for rows sc_rg + 8 m.
        {
            const int kpos = b * kKeyUnit + key;
            const std::uint16_t* krow = s.k + key * kRowStride;
#pragma unroll
            for (int m = 0; m < kScoreRows; ++m) {
                const int row = sc_rg + 8 * m;
                if (row >= rows) { break; }
                const RowState& r = s.rows[row];
                if (b < r.start || b > r.end) { continue; }
                float chain[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
                for (int d0 = 0; d0 < 32; d0 += 8) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const uint4 kw = *reinterpret_cast<const uint4*>(krow + 32 * i + d0);
                        const uint4 qw = *reinterpret_cast<const uint4*>(s.q + row * kRowStride + 32 * i + d0);
                        const std::uint32_t ku[4] = {kw.x, kw.y, kw.z, kw.w}, qu[4] = {qw.x, qw.y, qw.z, qw.w};
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            chain[i] = canon::fma_rn(canon::f32_from_bits(qu[e] << 16), canon::f32_from_bits(ku[e] << 16),
                                                     chain[i]);
                            chain[i] = canon::fma_rn(canon::f32_from_bits(qu[e] & 0xFFFF0000U),
                                                     canon::f32_from_bits(ku[e] & 0xFFFF0000U), chain[i]);
                        }
                    }
                }
                const float dot = canon::add_rn(canon::add_rn(chain[0], chain[1]), canon::add_rn(chain[2], chain[3]));
                const int dist  = r.pos - kpos;
                float bias      = 0.0F;
                if (dist >= 0 && dist < kRelativeExtent) {
                    float acc16 = 0.0F;
#pragma unroll
                    for (int i = 0; i < kRelDim; ++i) {
                        acc16 = canon::fma_rn(s.r[row * kRelDim + i], bf16f(__ldg(a.layer.relative_table + i * kRelativeExtent + dist)),
                                              acc16);
                    }
                    bias = bf16f(canon::f32_to_bf16_rn(acc16));
                }
                const bool visible = dist >= 0 && (full || dist < kWindow) && kpos >= first;
                const float score  = canon::add_rn(canon::mul_rn(dot, kInvSqrtD), bias);
                s.p[row * kKeyUnit + key] = visible ? canon::mul_rn(score, kLog2e) : neg_inf();
            }
        }
        __syncthreads();

        // The unit's softmax and the split's merge scalars: a warp per row.
        for (int row = warp; row < rows; row += kWarps) {
            RowState& r = s.rows[row];
            if (b < r.start || b > r.end) {
                if (wl == 0) { r.act = 0; }
                continue;
            }
            float t0 = s.p[row * kKeyUnit + wl], t1 = s.p[row * kKeyUnit + wl + 32];
            float mu = t0 > t1 ? t0 : t1;
#pragma unroll
            for (int offset = 16; offset >= 1; offset >>= 1) {
                const float o = __shfl_xor_sync(kAll, mu, offset);
                mu            = o > mu ? o : mu;
            }
            if (mu == neg_inf()) {
                if (wl == 0) { r.act = 0; }
                continue;
            }
            const float p0 = ex2(canon::add_rn(t0, -mu)), p1 = ex2(canon::add_rn(t1, -mu));
            s.p[row * kKeyUnit + wl]      = p0;
            s.p[row * kKeyUnit + wl + 32] = p1;
            float lu = canon::add_rn(p0, p1);
#pragma unroll
            for (int offset = 16; offset >= 1; offset >>= 1) { lu = canon::add_rn(lu, __shfl_xor_sync(kAll, lu, offset)); }
            if (wl == 0) {
                if (r.m == neg_inf()) {
                    r.m = mu, r.l = lu, r.act = 1;
                } else {
                    const float m = mu > r.m ? mu : r.m;
                    r.c           = ex2(canon::add_rn(r.m, -m));
                    r.cu          = ex2(canon::add_rn(mu, -m));
                    r.l           = canon::fma_rn(lu, r.cu, canon::mul_rn(r.l, r.c));
                    r.m           = m;
                    r.act         = 2;
                }
            }
        }
        __syncthreads();

        // P x V: thread (pv_rg, dc) for rows pv_rg + 16 m, keys in order.
        {
            float au[kPvRows][4] = {};
            int act[kPvRows];
#pragma unroll
            for (int m = 0; m < kPvRows; ++m) {
                const int row = pv_rg + 16 * m;
                act[m]        = row < rows ? s.rows[row].act : 0;
            }
            if (act[0] != 0 || act[1] != 0 || act[2] != 0) {
                for (int j = 0; j < kKeyUnit; ++j) {
                    const uint2 vw = *reinterpret_cast<const uint2*>(s.v + j * kRowStride + 4 * dc);
                    const float vf[4] = {f16f(static_cast<std::uint16_t>(vw.x & 0xFFFFU)), f16f(static_cast<std::uint16_t>(vw.x >> 16)),
                                         f16f(static_cast<std::uint16_t>(vw.y & 0xFFFFU)), f16f(static_cast<std::uint16_t>(vw.y >> 16))};
#pragma unroll
                    for (int m = 0; m < kPvRows; ++m) {
                        if (act[m] == 0) { continue; }
                        const float p = s.p[(pv_rg + 16 * m) * kKeyUnit + j];
#pragma unroll
                        for (int e = 0; e < 4; ++e) { au[m][e] = canon::fma_rn(p, vf[e], au[m][e]); }
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < kPvRows; ++m) {
                if (act[m] == 1) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) { acc[m][e] = au[m][e]; }
                } else if (act[m] == 2) {
                    const RowState& r = s.rows[pv_rg + 16 * m];
#pragma unroll
                    for (int e = 0; e < 4; ++e) { acc[m][e] = canon::fma_rn(au[m][e], r.cu, canon::mul_rn(acc[m][e], r.c)); }
                }
            }
        }
        __syncthreads(); // the next unit's pages overwrite K, V and p
    }

    // This split's statistics.
#pragma unroll
    for (int m = 0; m < kPvRows; ++m) {
        const int row = pv_rg + 16 * m;
        if (row >= rows) { continue; }
        const RowState& r = s.rows[row];
        if (r.start > r.end) { continue; }
        float* st = a.ws.stats + ((static_cast<std::size_t>(r.t) * kHeads + r.head) * kFullSplits + r.slot0) * kStatWords;
        if (dc == 0) {
            st[0] = r.m;
            st[1] = r.l;
        }
        *reinterpret_cast<float4*>(st + kAccOffset + 4 * dc) = make_float4(acc[m][0], acc[m][1], acc[m][2], acc[m][3]);
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const unsigned done = atomicAdd(a.ws.counters + lane * kKvHeads + kv, 1U);
        last                = done == gridDim.x - 1 ? 1 : 0;
    }
    __syncthreads();
    if (last == 0) { return; }
    __threadfence();

    // Combine every row's splits in split order, then the sink, the division and the gate.
    for (int row = warp; row < rows; row += kWarps) {
        const int t = c0 + row / kGroupHeads, head = kv * kGroupHeads + row % kGroupHeads;
        const int pos = a.map.position[t];
        const RowUnits u = row_units(full, pos, first);
        const int n      = u.hi - u.lo + 1;
        const int splits = full ? (n + full_split_units(n) - 1) / full_split_units(n) : n;
        const float* base = a.ws.stats + (static_cast<std::size_t>(t) * kHeads + head) * kFullSplits * kStatWords;
        const float ms = wl < splits ? __ldcg(base + wl * kStatWords) : neg_inf();
        const float ls = wl < splits ? __ldcg(base + wl * kStatWords + 1) : 0.0F;
        float m        = ms;
#pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            const float o = __shfl_xor_sync(kAll, m, offset);
            m             = o > m ? o : m;
        }
        const float c = wl < splits ? ex2(canon::add_rn(ms, -m)) : 0.0F;
        float l       = 0.0F;
        float o4[4]   = {0.0F, 0.0F, 0.0F, 0.0F};
        for (int sidx = 0; sidx < splits; ++sidx) {
            const float cs = __shfl_sync(kAll, c, sidx);
            l              = canon::fma_rn(__shfl_sync(kAll, ls, sidx), cs, l);
            const float4 v = __ldcg(reinterpret_cast<const float4*>(base + sidx * kStatWords + kAccOffset) + wl);
            o4[0]          = canon::fma_rn(v.x, cs, o4[0]);
            o4[1]          = canon::fma_rn(v.y, cs, o4[1]);
            o4[2]          = canon::fma_rn(v.z, cs, o4[2]);
            o4[3]          = canon::fma_rn(v.w, cs, o4[3]);
        }
        const float sink = bf16f(a.in.sink[static_cast<std::size_t>(t) * kHeads + head]);
        l                = canon::add_rn(l, ex2(canon::add_rn(canon::mul_rn(sink, kLog2e), -m)));
        float gate[4];
        load4(a.in.gate + static_cast<std::size_t>(t) * kQWidth + head * kD + 4 * wl, gate);
        float out[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float y = bf16f(canon::f32_to_bf16_rn(canon::div_rn(o4[e], l)));
            out[e]        = canon::mul_rn(y, canon::sigmoid_c(gate[e]));
        }
        *reinterpret_cast<uint2*>(a.out + static_cast<std::size_t>(t) * kQWidth + head * kD + 4 * wl) = pack4(out);
    }
    if (tid == 0) { a.ws.counters[lane * kKvHeads + kv] = 0; }
}

// ---------------------------------------------------------------------------- host side

template <class P>
bool aligned16(const P* p) {
    return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
}

std::size_t align256(std::size_t v) { return (v + 255) / 256 * 256; }

struct WorkspaceLayout {
    std::size_t q, r, stats, counters, total;
};
WorkspaceLayout workspace_layout() {
    WorkspaceLayout l{};
    l.q        = 0;
    l.r        = align256(l.q + std::size_t{kMaxAttentionColumns} * kHeads * kD * 2);
    l.stats    = align256(l.r + std::size_t{kMaxAttentionColumns} * kHeads * kRelDim * 2);
    l.counters = align256(l.stats + std::size_t{kMaxAttentionColumns} * kHeads * kFullSplits * kStatWords * 4);
    l.total    = align256(l.counters + std::size_t{kMaxAttentionLanes} * kKvHeads * 4);
    return l;
}

} // namespace

std::size_t attention_workspace_bytes() { return workspace_layout().total; }

AttentionWorkspace carve_attention_workspace(void* base) {
    const WorkspaceLayout l = workspace_layout();
    auto* b                 = static_cast<unsigned char*>(base);
    AttentionWorkspace w;
    w.q        = reinterpret_cast<std::uint16_t*>(b + l.q);
    w.r        = reinterpret_cast<std::uint16_t*>(b + l.r);
    w.stats    = reinterpret_cast<float*>(b + l.stats);
    w.counters = reinterpret_cast<std::uint32_t*>(b + l.counters);
    return w;
}

void attention(const AttentionArgs& a, cudaStream_t stream) {
    if (a.columns <= 0 || a.columns > kMaxAttentionColumns || a.lanes <= 0 || a.lanes > kMaxAttentionLanes ||
        a.max_lane_columns <= 0 || a.max_lane_columns > kMaxAttentionLaneColumns) {
        throw std::invalid_argument("inkling2 attention: 1-128 columns over 1-8 lanes of at most 16 columns each");
    }
    if (a.map.lane == nullptr || a.map.position == nullptr || a.map.first == nullptr || a.lane_columns == nullptr) {
        throw std::invalid_argument("inkling2 attention: the column map and the lanes' columns are needed");
    }
    const AttentionInputs& in = a.in;
    if (in.q == nullptr || in.r == nullptr || in.k == nullptr || in.v == nullptr || in.sink == nullptr ||
        in.gate == nullptr || (a.layer.full && (in.key_taps == nullptr || in.value_taps == nullptr))) {
        throw std::invalid_argument("inkling2 attention: a projection output is missing");
    }
    const AttentionLayer& l = a.layer;
    if (l.query_scale == nullptr || l.relative_scale == nullptr || l.relative_table == nullptr || l.key_conv == nullptr ||
        l.value_conv == nullptr) {
        throw std::invalid_argument("inkling2 attention: a layer parameter is missing");
    }
    for (const Ring* ring : {&a.key_ring, &a.value_ring}) {
        if (ring->base == nullptr || ring->width != kKvWidth || !aligned16(ring->base) ||
            ring->lane_stride < std::int64_t{kRingPositions} * kKvWidth || ring->lane_stride % 8 != 0) {
            throw std::invalid_argument("inkling2 attention: the K and V conv rings of width 1024 are needed");
        }
    }
    constexpr std::int64_t kLaneRing = std::int64_t{kKvHeads} * kWindowRingPages * kKeyUnit * kD;
    if (a.kv.k == nullptr || a.kv.v == nullptr || !aligned16(a.kv.k) || !aligned16(a.kv.v) ||
        (l.full != (a.kv.table != nullptr)) || (l.full && (a.kv.table_stride <= 0 || a.kv.pool_pages <= 0)) ||
        (!l.full && (a.kv.lane_stride < kLaneRing || a.kv.lane_stride % 8 != 0))) {
        throw std::invalid_argument("inkling2 attention: sliding visits use a ring per lane, full visits a paged pool");
    }
    if (a.ws.q == nullptr || a.ws.stats == nullptr || a.out == nullptr) {
        throw std::invalid_argument("inkling2 attention: the workspace and the output are needed");
    }
    CUDA_CHECK(pdl::launch_consumer({dim3(static_cast<unsigned>(a.columns), kKvHeads), dim3(kPrologueThreads), 0, stream},
                                    prologue_kernel, a));
    const std::size_t bytes = smem_bytes(kGroupHeads * a.max_lane_columns);
    static const bool attribute = [] {
        CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem_bytes(kMaxRows))));
        return true;
    }();
    (void)attribute;
    const unsigned splits = l.full ? kFullSplits : kSlidingSplits;
    CUDA_CHECK(pdl::launch_consumer({dim3(splits, kKvHeads, static_cast<unsigned>(a.lanes)), dim3(kThreads), bytes, stream},
                                    attention_kernel, a));
}

} // namespace infernix::ops::inkling2
