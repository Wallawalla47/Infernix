// Inkling 2's DSpark drafter Ops (include/infernix/ops/inkling2_dspark.h).
//
// dspark_mix: a CTA per (column, consumer) of 288 threads, 8 of H each.
// dspark_rotate: a warp per (row, head), lane l holding elements 4l .. 4l + 3 (RoPE partners 16 lanes apart).
// dspark_attention: a CTA per (key split of 128, KV head) over the head's 24 query rows (8 block rows x 3 heads),
//   tiles of 32 keys (K transposed in shared memory), an online softmax per row; then a CTA per KV head merges the
//   active splits in order.
// dspark_add_norm: a CTA per row; dspark_swiglu: elementwise.
// dspark_markov: the step-invariant projection (a warp per row of 768), then per step a gate kernel (a warp per row,
//   the last CTA applying the gates) and a logits kernel (a warp per vocabulary row, the last CTA merging the CTAs'
//   argmax and softmax sums in CTA order).

#include "infernix/ops/inkling2_dspark.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/canonical_math.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops::inkling2 {
namespace {

constexpr int kH          = 2304;
constexpr int kD          = kDSparkHeadDim;
constexpr int kRowThreads = kH / 8; // 288
constexpr int kRowWarps   = kRowThreads / 32;
constexpr unsigned kAll   = 0xFFFFFFFFU;
constexpr float kInvH     = static_cast<float>(1.0 / 2304.0);
constexpr float kInvD     = static_cast<float>(1.0 / 128.0);

__device__ __forceinline__ float bf(std::uint16_t v) { return canon::bf16_to_f32(v); }
__device__ __forceinline__ std::uint16_t rb(float x) { return canon::f32_to_bf16_rn(x); }

__device__ __forceinline__ void unpack8(uint4 w, float (&v)[8]) {
    const std::uint32_t a[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[2 * j]     = canon::f32_from_bits(a[j] << 16);
        v[2 * j + 1] = canon::f32_from_bits(a[j] & 0xFFFF0000U);
    }
}
__device__ __forceinline__ void load8(const std::uint16_t* p, float (&v)[8]) {
    unpack8(*reinterpret_cast<const uint4*>(p), v);
}
__device__ __forceinline__ void store8(std::uint16_t* p, const float (&v)[8]) {
    std::uint32_t a[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { a[j] = std::uint32_t{rb(v[2 * j])} | (std::uint32_t{rb(v[2 * j + 1])} << 16); }
    *reinterpret_cast<uint4*>(p) = make_uint4(a[0], a[1], a[2], a[3]);
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o >= 1; o /= 2) { v = canon::add_rn(v, __shfl_xor_sync(kAll, v, o)); }
    return v;
}
// The sum of every thread's v over a 288-thread CTA: the warps' bfly32 sums added in warp order.
__device__ __forceinline__ float row_sum(float v, float* red) {
    v = warp_sum(v);
    __syncthreads(); // red's previous readers are done
    if ((threadIdx.x & 31U) == 0) { red[threadIdx.x / 32] = v; }
    __syncthreads();
    float total = 0.0F;
#pragma unroll
    for (int w = 0; w < kRowWarps; ++w) { total = canon::add_rn(total, red[w]); }
    return total;
}
__device__ __forceinline__ float rms_scale(float sum_squares, float inv_n, float eps) {
    return canon::div_rn(1.0F, __fsqrt_rn(canon::add_rn(canon::mul_rn(sum_squares, inv_n), eps)));
}

__device__ __forceinline__ std::uint32_t acq_rel_add(std::uint32_t* word) {
    std::uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;\n" : "=r"(old) : "l"(word) : "memory");
    return old;
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { throw std::runtime_error(std::string("inkling2 dspark ") + what + ": " + cudaGetErrorString(e)); }
}
pdl::LaunchConfig config(dim3 grid, dim3 block, std::size_t smem, cudaStream_t stream) {
    return {grid, block, smem, stream};
}

// ---------------------------------------------------------------------------- mix

__global__ void __launch_bounds__(kRowThreads) mix_kernel(const DSparkMixArgs a) {
    __shared__ float red[kRowWarps];
    const int c = static_cast<int>(blockIdx.x), m = static_cast<int>(blockIdx.y), e = 8 * static_cast<int>(threadIdx.x);
    // alpha = bf16(softmax_fp32(the consumer's row)) over the 4 taps.
    float l[kDSparkFeatures], alpha[kDSparkFeatures];
    float top = -INFINITY;
#pragma unroll
    for (int j = 0; j < kDSparkFeatures; ++j) {
        l[j] = bf(a.fuse_weight[a.rows[m] * kDSparkFeatures + j]);
        top  = fmaxf(top, l[j]);
    }
    float sum = 0.0F;
#pragma unroll
    for (int j = 0; j < kDSparkFeatures; ++j) {
        alpha[j] = canon::exp_c(__fsub_rn(l[j], top));
        sum      = canon::add_rn(sum, alpha[j]);
    }
#pragma unroll
    for (int j = 0; j < kDSparkFeatures; ++j) { alpha[j] = bf(rb(canon::div_rn(alpha[j], sum))); }
    float w[8];
    load8(a.norms[m] + e, w);
    pdl::enter();
    float mix[8] = {};
#pragma unroll
    for (int j = 0; j < kDSparkFeatures; ++j) {
        float t[8];
        load8(a.features + (static_cast<std::size_t>(c) * kDSparkFeatures + j) * kH + e, t);
#pragma unroll
        for (int k = 0; k < 8; ++k) { mix[k] = __fmaf_rn(t[k], alpha[j], mix[k]); }
    }
    float sq = 0.0F;
#pragma unroll
    for (int k = 0; k < 8; ++k) { sq = __fmaf_rn(mix[k], mix[k], sq); }
    const float r = rms_scale(row_sum(sq, red), kInvH, a.eps);
    float out[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { out[k] = canon::mul_rn(canon::mul_rn(mix[k], r), w[k]); }
    const int row = a.ring[m] > 0 ? a.positions[c] % a.ring[m] : c;
    std::uint16_t* base = a.out[m];
    if (a.ring[m] > 0 && a.lanes != nullptr) { base += static_cast<std::size_t>(a.lanes[c]) * a.lane_stride[m]; }
    store8(base + static_cast<std::size_t>(row) * kH + e, out);
}

// ---------------------------------------------------------------------------- rotate

__global__ void __launch_bounds__(32) rotate_kernel(const DSparkRotateArgs a) {
    const int r = static_cast<int>(blockIdx.x), h = static_cast<int>(blockIdx.y), lane = static_cast<int>(threadIdx.x);
    float w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { w[j] = bf(a.norm[4 * lane + j]); }
    // theta^(-i / 64) for the lane's pair indices (the table's FP64 values rounded to FP32).
    float inv_f[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { inv_f[j] = a.inv_freq[(4 * lane + j) % (kD / 2)]; }
    pdl::enter();
    const std::uint16_t* xr = a.x + static_cast<std::size_t>(r) * a.x_stride + h * kD + 4 * lane;
    const uint2 raw         = *reinterpret_cast<const uint2*>(xr);
    float x[4]              = {canon::f32_from_bits(raw.x << 16), canon::f32_from_bits(raw.x & 0xFFFF0000U),
                               canon::f32_from_bits(raw.y << 16), canon::f32_from_bits(raw.y & 0xFFFF0000U)};
    float sq = 0.0F;
#pragma unroll
    for (int j = 0; j < 4; ++j) { sq = __fmaf_rn(x[j], x[j], sq); }
    const float inv = rms_scale(warp_sum(sq), kInvD, a.eps);
    float n[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { n[j] = bf(rb(canon::mul_rn(canon::mul_rn(x[j], inv), w[j]))); }
    const int position = a.positions != nullptr ? a.positions[r] : a.round->first + r;
    float out[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float partner = __shfl_xor_sync(kAll, n[j], 16);
        const float angle   = canon::mul_rn(static_cast<float>(position), inv_f[j]);
        float s = 0.0F, c = 0.0F;
        sincosf(angle, &s, &c);
        s = bf(rb(s));
        c = bf(rb(c));
        // (i, i + 64): out[i] = x[i] cos - x[i + 64] sin, out[i + 64] = x[i + 64] cos + x[i] sin.
        out[j] = lane < 16 ? __fsub_rn(canon::mul_rn(n[j], c), canon::mul_rn(partner, s))
                           : canon::add_rn(canon::mul_rn(n[j], c), canon::mul_rn(partner, s));
    }
    const uint2 packed = make_uint2(std::uint32_t{rb(out[0])} | (std::uint32_t{rb(out[1])} << 16),
                                    std::uint32_t{rb(out[2])} | (std::uint32_t{rb(out[3])} << 16));
    if (a.dense != nullptr) {
        *reinterpret_cast<uint2*>(a.dense + (static_cast<std::size_t>(r) * a.heads + h) * kD + 4 * lane) = packed;
        return;
    }
    const int lane_index = a.lanes != nullptr ? a.lanes[r] : (a.lane >= 0 ? a.lane : a.round->lane);
    std::size_t at;
    if (a.ring > 0) {
        at = static_cast<std::size_t>(lane_index) * a.lane_stride +
             (static_cast<std::size_t>(h) * a.ring + position % a.ring) * kD + 4 * lane;
    } else { // the full layer's page (the lane's block-table row)
        const std::int64_t page = a.table[static_cast<std::int64_t>(lane_index) * a.table_stride + position / 64];
        at = ((static_cast<std::size_t>(page) * kDSparkKvHeads + h) * 64 + position % 64) * kD + 4 * lane;
    }
    *reinterpret_cast<uint2*>(a.cache_k + at) = packed;
    if (a.values != nullptr) {
        *reinterpret_cast<uint2*>(a.cache_v + at) =
            *reinterpret_cast<const uint2*>(a.values + static_cast<std::size_t>(r) * a.values_stride + h * kD + 4 * lane);
    }
}

// ---------------------------------------------------------------------------- attention

constexpr int kSplitKeys    = 128;
constexpr int kTileKeys     = 32;
constexpr int kGroupRows    = kDSparkBlock * (kDSparkHeads / kDSparkKvHeads); // 24 query rows a KV head
constexpr int kAttnThreads  = 256;
constexpr int kPartialWords = kD + 2; // m, l, acc[128]

struct AttnSmem {
    float q[kGroupRows][kD];
    float kt[kD][kTileKeys + 1]; // K transposed: [dim][key]
    float v[kTileKeys][kD];
    float p[kGroupRows][kTileKeys];
    float m[kGroupRows], l[kGroupRows], scale[kGroupRows];
    std::uint32_t rows[kTileKeys]; // the tile's keys' rows (of 128 values) in the cache
};

__device__ __forceinline__ void key_range(const DSparkAttentionArgs& a, int& lo, int& hi) {
    const int first = a.round->first;
    hi              = first + kDSparkBlock;
    lo              = a.ring > 0 ? max(0, first - kDSparkWindow) : 0;
}

__global__ void __launch_bounds__(kAttnThreads, 1) attention_split_kernel(const DSparkAttentionArgs a) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm       = *reinterpret_cast<AttnSmem*>(smem_raw);
    const int split = static_cast<int>(blockIdx.x), g = static_cast<int>(blockIdx.y), tid = static_cast<int>(threadIdx.x);
    const int lane = tid % 32, warp = tid / 32;
    pdl::enter();
    int lo = 0, hi = 0;
    key_range(a, lo, hi);
    const int k0 = lo + split * kSplitKeys, k1 = min(hi, k0 + kSplitKeys);
    float* partial = a.scratch + (static_cast<std::size_t>(g) * a.splits + split) * kGroupRows * kPartialWords;
    if (k0 >= k1) { return; } // an inactive split (the merge reads only the active ones)
    // The group's query rows: row q = block row i * 3 + head j (query head 3 g + j).
    for (int i = tid; i < kGroupRows * kD; i += kAttnThreads) {
        const int q = i / kD, d = i % kD, row = q / 3, head = 3 * g + q % 3;
        sm.q[q][d] = bf(a.q[(static_cast<std::size_t>(row) * kDSparkHeads + head) * kD + d]);
    }
    if (tid < kGroupRows) {
        sm.m[tid] = -INFINITY;
        sm.l[tid] = 0.0F;
    }
    const int lane_index = a.round->lane;
    // A key's row of 128 values: a sliding layer's ring slot, or the full layer's page row (the lane's block-table row).
    // Rows fit 32 bits (a pool of 2^32 / 512 pages), offsets do not.
    const auto row_of = [&](int position) -> std::uint32_t {
        if (a.ring > 0) {
            return static_cast<std::uint32_t>(static_cast<std::size_t>(lane_index) * a.lane_stride / kD +
                                              static_cast<std::size_t>(g) * a.ring + position % a.ring);
        }
        const std::uint32_t page = static_cast<std::uint32_t>(
            a.table[static_cast<std::int64_t>(lane_index) * a.table_stride + position / 64]);
        return (page * kDSparkKvHeads + static_cast<std::uint32_t>(g)) * 64U + static_cast<std::uint32_t>(position % 64);
    };
    // P x V: thread t < 192 owns row t / 8 and dims 16 (t % 8) .. + 15.
    const int pv_row = tid / 8, pv_d = 16 * (tid % 8);
    float acc[16] = {};
    const float score_scale = static_cast<float>(0.088388347648318440550); // fl32(128^-0.5)
    for (int t0 = k0; t0 < k1; t0 += kTileKeys) {
        const int n = min(kTileKeys, k1 - t0);
        __syncthreads(); // the previous tile's readers are done
        if (tid < n) { sm.rows[tid] = row_of(t0 + tid); } // each key's row, looked up once a tile
        __syncthreads();
        for (int i = tid; i < kTileKeys * kD; i += kAttnThreads) {
            const int k = i / kD, d = i % kD;
            float kv = 0.0F, vv = 0.0F;
            if (k < n) {
                const std::size_t at = static_cast<std::size_t>(sm.rows[k]) * kD + d;
                kv                   = bf(a.cache_k[at]);
                vv                   = bf(a.cache_v[at]);
            }
            sm.kt[d][k] = kv;
            sm.v[k][d]  = vv;
        }
        __syncthreads();
        // Scores: thread t: key t % 32, rows 3 (t / 32) .. + 2.
        {
            const int k = lane, q0 = 3 * warp;
            float s[3] = {0.0F, 0.0F, 0.0F};
            for (int d = 0; d < kD; ++d) {
                const float kv = sm.kt[d][k];
#pragma unroll
                for (int j = 0; j < 3; ++j) { s[j] = __fmaf_rn(sm.q[q0 + j][d], kv, s[j]); }
            }
#pragma unroll
            for (int j = 0; j < 3; ++j) { sm.p[q0 + j][k] = k < n ? canon::mul_rn(s[j], score_scale) : -INFINITY; }
        }
        __syncthreads();
        // The online softmax: a warp per 3 rows, lane = key.
        for (int j = 0; j < 3; ++j) {
            const int q = 3 * warp + j;
            const float s = sm.p[q][lane];
            float top     = s;
#pragma unroll
            for (int o = 16; o >= 1; o /= 2) { top = fmaxf(top, __shfl_xor_sync(kAll, top, o)); }
            const float m_old = sm.m[q], m_new = fmaxf(m_old, top);
            const float p     = lane < n ? canon::exp_c(__fsub_rn(s, m_new)) : 0.0F;
            const float psum  = warp_sum(p);
            sm.p[q][lane]     = p;
            __syncwarp();
            if (lane == 0) {
                const float scale = m_old == -INFINITY ? 0.0F : canon::exp_c(__fsub_rn(m_old, m_new));
                sm.scale[q]       = scale;
                sm.l[q]           = canon::add_rn(canon::mul_rn(sm.l[q], scale), psum);
                sm.m[q]           = m_new;
            }
        }
        __syncthreads();
        if (tid < kGroupRows * 8) {
            const float scale = sm.scale[pv_row];
#pragma unroll
            for (int d = 0; d < 16; ++d) { acc[d] = canon::mul_rn(acc[d], scale); }
            for (int k = 0; k < n; ++k) {
                const float p = sm.p[pv_row][k];
#pragma unroll
                for (int d = 0; d < 16; ++d) { acc[d] = __fmaf_rn(p, sm.v[k][pv_d + d], acc[d]); }
            }
        }
    }
    __syncthreads();
    if (tid < kGroupRows * 8) {
        float* out = partial + static_cast<std::size_t>(pv_row) * kPartialWords;
        if (tid % 8 == 0) {
            out[0] = sm.m[pv_row];
            out[1] = sm.l[pv_row];
        }
#pragma unroll
        for (int d = 0; d < 16; ++d) { out[2 + pv_d + d] = acc[d]; }
    }
}

// The splits' merge in split order: a CTA per KV head, a thread per (row, dim).
__global__ void __launch_bounds__(kD) attention_merge_kernel(const DSparkAttentionArgs a) {
    const int g = static_cast<int>(blockIdx.x), q = static_cast<int>(blockIdx.y), d = static_cast<int>(threadIdx.x);
    pdl::enter();
    int lo = 0, hi = 0;
    key_range(a, lo, hi);
    const int active = (hi - lo + kSplitKeys - 1) / kSplitKeys;
    const float* base = a.scratch + static_cast<std::size_t>(g) * a.splits * kGroupRows * kPartialWords +
                        static_cast<std::size_t>(q) * kPartialWords;
    float top = -INFINITY;
    for (int s = 0; s < active; ++s) { top = fmaxf(top, base[static_cast<std::size_t>(s) * kGroupRows * kPartialWords]); }
    float l = 0.0F, acc = 0.0F;
    for (int s = 0; s < active; ++s) {
        const float* p    = base + static_cast<std::size_t>(s) * kGroupRows * kPartialWords;
        const float scale = canon::exp_c(__fsub_rn(p[0], top));
        l                 = __fmaf_rn(p[1], scale, l);
        acc               = __fmaf_rn(p[2 + d], scale, acc);
    }
    const int row = q / 3, head = 3 * g + q % 3;
    a.out[(static_cast<std::size_t>(row) * kDSparkHeads + head) * kD + d] = rb(canon::div_rn(acc, l));
}

// ---------------------------------------------------------------------------- add_norm, swiglu

__global__ void __launch_bounds__(kRowThreads) add_norm_kernel(const DSparkAddNormArgs a) {
    __shared__ float red[kRowWarps];
    const int r = static_cast<int>(blockIdx.x), e = 8 * static_cast<int>(threadIdx.x);
    float w[8];
    load8(a.norm + e, w);
    pdl::enter();
    const std::size_t at = static_cast<std::size_t>(r) * kH + e;
    float s[8];
    load8(a.residual + at, s);
    if (a.x != nullptr) {
        float x[8];
        load8(a.x + at, x);
#pragma unroll
        for (int k = 0; k < 8; ++k) { s[k] = canon::add_rn(x[k], s[k]); }
        store8(a.residual + at, s);
    }
    float sq = 0.0F;
#pragma unroll
    for (int k = 0; k < 8; ++k) { sq = __fmaf_rn(s[k], s[k], sq); }
    const float scale = rms_scale(row_sum(sq, red), kInvH, a.eps);
    float out[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { out[k] = canon::mul_rn(canon::mul_rn(s[k], scale), w[k]); }
    store8(a.normed + at, out);
}

__global__ void __launch_bounds__(256) swiglu_kernel(const std::uint16_t* gate, const std::uint16_t* up,
                                                     std::int64_t groups, std::uint16_t* h) {
    pdl::enter();
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= groups) { return; }
    float g[8], u[8], out[8];
    load8(gate + 8 * i, g);
    load8(up + 8 * i, u);
#pragma unroll
    for (int k = 0; k < 8; ++k) { out[k] = canon::mul_rn(canon::silu_c(g[k]), u[k]); }
    store8(h + 8 * i, out);
}

// ---------------------------------------------------------------------------- Markov head

constexpr int kRank3        = 3 * kMarkovRank;       // 768
constexpr int kProjStride   = 2 * kMarkovRank + 2 * kH; // 5120
constexpr int kInvK         = 2 * kH;                 // the invariant columns
constexpr int kStepK        = 2 * kMarkovRank;        // the step columns
constexpr int kMarkovWarps  = 8;
constexpr int kGateCtas     = kRank3 / kMarkovWarps / 4; // 4 rows a warp: 24 CTAs
constexpr int kLogitCtas    = 512;

// The scratch: s and l (BF16 [256] each), u (BF16 [768]), the invariant (FP32 [steps][768]), the gate and logits
// counters, and the logits CTAs' partials (max, its index, the running max and sum).
struct MarkovScratch {
    std::uint16_t s[kMarkovRank];
    std::uint16_t lat[kMarkovRank];
    std::uint16_t u[kRank3];
    float invariant[kMarkovSteps][kRank3];
    std::uint32_t gate_done;
    std::uint32_t logits_done;
    float best[kLogitCtas];
    std::int32_t best_index[kLogitCtas];
    float run_max[kLogitCtas];
    float run_sum[kLogitCtas];
};

__device__ __forceinline__ MarkovScratch& markov_scratch(const DSparkMarkovArgs& a) {
    return *reinterpret_cast<MarkovScratch*>(a.scratch);
}

// invariant[i][row] = sum_k W[row][2 rank + k] [x_out[i + 1]; f]_k + bias[row] (FP32): a warp per row.
__global__ void __launch_bounds__(32 * kMarkovWarps) markov_invariant_kernel(const DSparkMarkovArgs a, int steps) {
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int row  = static_cast<int>(blockIdx.x) * kMarkovWarps + warp;
    MarkovScratch& sc = markov_scratch(a);
    pdl::enter();
    if (blockIdx.x == 0) {
        for (int i = static_cast<int>(threadIdx.x); i < kMarkovRank; i += 32 * kMarkovWarps) { sc.s[i] = 0; }
    }
    const int first = a.round->first;
    const std::uint16_t* f = a.f_ring + static_cast<std::size_t>(a.round->lane) * a.f_lane_stride +
                             static_cast<std::size_t>((first - 1) % a.f_ring_rows) * kH;
    float acc[kMarkovSteps] = {};
    const std::uint16_t* w = a.projection + static_cast<std::size_t>(row) * kProjStride + kStepK;
    for (int q = 0; q < kInvK / 256; ++q) {
        const int k = 8 * (lane + 32 * q);
        float wv[8];
        load8(w + k, wv);
        // Every step unrolled (the accumulators stay in registers), the steps past `steps` skipped.
        if (k < kH) {
#pragma unroll
            for (int i = 0; i < kMarkovSteps; ++i) {
                if (i < steps) {
                    float z[8];
                    load8(a.block + static_cast<std::size_t>(i + 1) * kH + k, z);
#pragma unroll
                    for (int j = 0; j < 8; ++j) { acc[i] = __fmaf_rn(wv[j], z[j], acc[i]); }
                }
            }
        } else {
            float z[8];
            load8(f + (k - kH), z);
#pragma unroll
            for (int i = 0; i < kMarkovSteps; ++i) {
                if (i < steps) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) { acc[i] = __fmaf_rn(wv[j], z[j], acc[i]); }
                }
            }
        }
    }
    const float b = bf(a.bias[row]);
#pragma unroll
    for (int i = 0; i < kMarkovSteps; ++i) {
        const float total = warp_sum(acc[i]);
        if (lane == 0 && i < steps) { sc.invariant[i][row] = canon::add_rn(total, b); }
    }
}

// Step `step` (1-based): u = bf16(invariant + W_step [s; w1[prev]]), a warp per 4 rows; the last CTA applies the gates.
__global__ void __launch_bounds__(32 * kMarkovWarps) markov_gate_kernel(const DSparkMarkovArgs a, int step) {
    __shared__ int last;
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    MarkovScratch& sc = markov_scratch(a);
    pdl::enter();
    const int prev = step == 1 ? a.round->anchor : a.tokens[step - 2];
    // The lane's 16 of [s (256); w1[prev] (256)].
    float z[16];
    {
        const int k = 16 * lane;
        const std::uint16_t* src = k < kMarkovRank ? sc.s + k : a.embedding + static_cast<std::size_t>(prev) * kMarkovRank + (k - kMarkovRank);
        float lo[8], hi[8];
        unpack8(__ldcg(reinterpret_cast<const uint4*>(src)), lo);
        unpack8(__ldcg(reinterpret_cast<const uint4*>(src + 8)), hi);
#pragma unroll
        for (int j = 0; j < 8; ++j) { z[j] = lo[j], z[8 + j] = hi[j]; }
    }
    for (int i = 0; i < 4; ++i) {
        const int row = (static_cast<int>(blockIdx.x) * kMarkovWarps + warp) * 4 + i;
        const std::uint16_t* w = a.projection + static_cast<std::size_t>(row) * kProjStride + 16 * lane;
        float lo[8], hi[8];
        load8(w, lo);
        load8(w + 8, hi);
        float acc = 0.0F;
#pragma unroll
        for (int j = 0; j < 8; ++j) { acc = __fmaf_rn(lo[j], z[j], acc); }
#pragma unroll
        for (int j = 0; j < 8; ++j) { acc = __fmaf_rn(hi[j], z[8 + j], acc); }
        const float total = warp_sum(acc);
        if (lane == 0) { sc.u[row] = rb(canon::add_rn(sc.invariant[step - 1][row], total)); }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        last = acq_rel_add(&sc.gate_done) == gridDim.x - 1U ? 1 : 0;
    }
    __syncthreads();
    if (last == 0) { return; }
    __threadfence();
    // The gates (a thread a channel): every CTA has read s.
    const int c = static_cast<int>(threadIdx.x);
    const float g     = bf(rb(canon::sigmoid_c(bf(__ldcg(sc.u + c)))));
    const float kept  = bf(rb(canon::mul_rn(g, bf(sc.s[c]))));
    const float cand  = bf(rb(canon::tanh_c(bf(__ldcg(sc.u + kMarkovRank + c)))));
    const float blend = bf(rb(canon::mul_rn(bf(rb(__fsub_rn(1.0F, g))), cand)));
    sc.s[c]           = rb(canon::add_rn(kept, blend));
    sc.lat[c]         = rb(canon::tanh_c(bf(__ldcg(sc.u + 2 * kMarkovRank + c))));
    if (c == 0) { sc.gate_done = 0U; }
}

// Step `step`: the vocabulary's logits, a warp per row; each CTA's argmax (lowest id on ties) and its running softmax
// sum, merged in CTA order by the last CTA. Q8: the 8-bit Fold (a lane's 8 codes and its group's scale).
template <bool Q8>
__global__ void __launch_bounds__(32 * kMarkovWarps) markov_logits_kernel(const DSparkMarkovArgs a, int step) {
    __shared__ float lat[kMarkovRank];
    __shared__ float wbest[kMarkovWarps], wmax[kMarkovWarps], wsum[kMarkovWarps];
    __shared__ int windex[kMarkovWarps];
    __shared__ int last;
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    MarkovScratch& sc = markov_scratch(a);
    pdl::enter();
    for (int i = static_cast<int>(threadIdx.x); i < kMarkovRank; i += 32 * kMarkovWarps) { lat[i] = bf(__ldcg(sc.lat + i)); }
    __syncthreads();
    float l8[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) { l8[j] = lat[8 * lane + j]; }
    const int per_cta = (a.vocab + kLogitCtas - 1) / kLogitCtas;
    const int v0 = static_cast<int>(blockIdx.x) * per_cta, v1 = min(a.vocab, v0 + per_cta);
    const float* base = a.base + static_cast<std::size_t>(step - 1) * a.vocab;
    float best = -INFINITY, run_max = -INFINITY, run_sum = 0.0F;
    int best_index = 0x7FFFFFFF;
    for (int v = v0 + warp; v < v1; v += kMarkovWarps) {
        float fv[8];
        if constexpr (Q8) {
            // code * scale, exact in FP32 (8 by 11 significant bits): each byte c as 2^23 + (c + 128) by a byte
            // permute into 0x4B000000, less 2^23 + 128.
            const uint2 code = __ldg(reinterpret_cast<const uint2*>(a.fold_codes + static_cast<std::size_t>(v) * kMarkovRank) + lane);
            const float s    = __half2float(__ushort_as_half(__ldg(a.fold_scales + static_cast<std::size_t>(v) * (kMarkovRank / 32) + lane / 4)));
            const unsigned lo = code.x ^ 0x80808080U, hi = code.y ^ 0x80808080U;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                fv[j]     = (__uint_as_float(__byte_perm(lo, 0x4B000000U, 0x7650U + j)) - 8388736.0F) * s;
                fv[4 + j] = (__uint_as_float(__byte_perm(hi, 0x4B000000U, 0x7650U + j)) - 8388736.0F) * s;
            }
        } else {
            load8(a.fold + static_cast<std::size_t>(v) * kMarkovRank + 8 * lane, fv);
        }
        float acc = 0.0F;
#pragma unroll
        for (int j = 0; j < 8; ++j) { acc = __fmaf_rn(fv[j], l8[j], acc); }
        acc = warp_sum(acc);
        const float corr = bf(rb(canon::add_rn(acc, bf(a.fold_bias[v]))));
        const float z    = bf(rb(canon::add_rn(bf(rb(base[v])), corr)));
        const float L    = bf(rb(canon::softcap_c(z, a.softcap)));
        if (L > best) { // rows ascend within a warp: the first maximum is the lowest id
            best       = L;
            best_index = v;
        }
        if (L > run_max) {
            run_sum = run_max == -INFINITY ? 1.0F : __fmaf_rn(run_sum, canon::exp_c(__fsub_rn(run_max, L)), 1.0F);
            run_max = L;
        } else {
            run_sum = canon::add_rn(run_sum, canon::exp_c(__fsub_rn(L, run_max)));
        }
    }
    if (lane == 0) {
        wbest[warp]  = best;
        windex[warp] = best_index;
        wmax[warp]   = run_max;
        wsum[warp]   = run_sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float b = -INFINITY, m = -INFINITY, s = 0.0F;
        int bi  = 0x7FFFFFFF;
        for (int w = 0; w < kMarkovWarps; ++w) {
            if (wbest[w] > b || (wbest[w] == b && windex[w] < bi)) {
                b  = wbest[w];
                bi = windex[w];
            }
            if (wmax[w] == -INFINITY) { continue; }
            if (wmax[w] > m) {
                s = __fmaf_rn(s, canon::exp_c(__fsub_rn(m, wmax[w])), wsum[w]);
                m = wmax[w];
            } else {
                s = __fmaf_rn(wsum[w], canon::exp_c(__fsub_rn(wmax[w], m)), s);
            }
        }
        const int cta         = static_cast<int>(blockIdx.x);
        sc.best[cta]          = b;
        sc.best_index[cta]    = bi;
        sc.run_max[cta]       = m;
        sc.run_sum[cta]       = s;
        __threadfence();
        last = acq_rel_add(&sc.logits_done) == gridDim.x - 1U ? 1 : 0;
    }
    __syncthreads();
    if (last == 0 || threadIdx.x != 0) { return; }
    __threadfence();
    float b = -INFINITY, m = -INFINITY, s = 0.0F;
    int bi  = 0x7FFFFFFF;
    for (int c = 0; c < static_cast<int>(gridDim.x); ++c) {
        const float cb = __ldcg(sc.best + c), cm = __ldcg(sc.run_max + c), cs = __ldcg(sc.run_sum + c);
        const int ci   = __ldcg(sc.best_index + c);
        if (cb > b || (cb == b && ci < bi)) {
            b  = cb;
            bi = ci;
        }
        if (cm == -INFINITY) { continue; }
        if (cm > m) {
            s = __fmaf_rn(s, canon::exp_c(__fsub_rn(m, cm)), cs);
            m = cm;
        } else {
            s = __fmaf_rn(cs, canon::exp_c(__fsub_rn(cm, m)), s);
        }
    }
    a.tokens[step - 1]        = bi;
    a.probabilities[step - 1] = canon::div_rn(canon::exp_c(__fsub_rn(b, m)), s);
    sc.logits_done            = 0U;
}

} // namespace

void dspark_mix(const DSparkMixArgs& a, cudaStream_t stream) {
    if (a.features == nullptr || a.fuse_weight == nullptr || a.columns <= 0 || a.count <= 0 || a.count > 4) {
        throw std::invalid_argument("inkling2 dspark_mix: features, fuse weights, 1-4 consumers and columns");
    }
    for (int m = 0; m < a.count; ++m) {
        if (a.norms[m] == nullptr || a.out[m] == nullptr || a.rows[m] < 0 || a.rows[m] >= 4 ||
            (a.ring[m] > 0 && a.positions == nullptr)) {
            throw std::invalid_argument("inkling2 dspark_mix: each consumer needs its row, norm and output");
        }
    }
    check(pdl::launch_consumer(config(dim3(a.columns, a.count), dim3(kRowThreads), 0, stream), mix_kernel, a), "mix");
}

void dspark_rotate(const DSparkRotateArgs& a, cudaStream_t stream) {
    if (a.x == nullptr || a.norm == nullptr || a.rows <= 0 || a.heads <= 0 || a.x_stride < a.heads * kD ||
        (a.positions == nullptr && a.round == nullptr) || a.inv_freq == nullptr ||
        (a.dense == nullptr && (a.cache_k == nullptr || (a.values != nullptr && a.cache_v == nullptr) ||
                                (a.lanes == nullptr && a.lane < 0 && a.round == nullptr) ||
                                (a.ring <= 0 && (a.table == nullptr || a.table_stride <= 0))))) {
        throw std::invalid_argument("inkling2 dspark_rotate: rows, heads, a norm, positions and one destination");
    }
    check(pdl::launch_consumer(config(dim3(a.rows, a.heads), dim3(32), 0, stream), rotate_kernel, a), "rotate");
}

void dspark_inv_freq(double theta, float (&inv_freq)[kDSparkHeadDim / 2]) {
    for (int i = 0; i < kDSparkHeadDim / 2; ++i) {
        inv_freq[i] = static_cast<float>(std::pow(theta, -static_cast<double>(i) / (kDSparkHeadDim / 2)));
    }
}

std::size_t dspark_attention_scratch_bytes(std::int32_t splits) {
    return static_cast<std::size_t>(kDSparkKvHeads) * splits * kGroupRows * kPartialWords * sizeof(float);
}

void dspark_attention(const DSparkAttentionArgs& a, cudaStream_t stream) {
    if (a.q == nullptr || a.cache_k == nullptr || a.cache_v == nullptr || a.round == nullptr || a.scratch == nullptr ||
        a.out == nullptr || a.splits <= 0 || (a.ring <= 0 && (a.table == nullptr || a.table_stride <= 0))) {
        throw std::invalid_argument("inkling2 dspark_attention: queries, a cache, the round, scratch and splits");
    }
    static const bool configured = [] {
        check(cudaFuncSetAttribute(attention_split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(sizeof(AttnSmem))),
              "attribute");
        return true;
    }();
    (void)configured;
    check(pdl::launch_consumer(config(dim3(a.splits, kDSparkKvHeads), dim3(kAttnThreads), sizeof(AttnSmem), stream),
                               attention_split_kernel, a),
          "attention");
    check(pdl::launch_consumer(config(dim3(kDSparkKvHeads, kGroupRows), dim3(kD), 0, stream), attention_merge_kernel, a),
          "attention merge");
}

void dspark_add_norm(const DSparkAddNormArgs& a, cudaStream_t stream) {
    if (a.residual == nullptr || a.norm == nullptr || a.normed == nullptr || a.rows <= 0) {
        throw std::invalid_argument("inkling2 dspark_add_norm: a residual, a norm, an output and rows");
    }
    check(pdl::launch_consumer(config(dim3(a.rows), dim3(kRowThreads), 0, stream), add_norm_kernel, a), "add_norm");
}

void dspark_swiglu(const std::uint16_t* gate, const std::uint16_t* up, std::int32_t rows, std::int32_t width,
                   std::uint16_t* h, cudaStream_t stream) {
    if (gate == nullptr || up == nullptr || h == nullptr || rows <= 0 || width <= 0 || width % 8 != 0) {
        throw std::invalid_argument("inkling2 dspark_swiglu: gate, up and output rows of a multiple of 8");
    }
    const std::int64_t groups = static_cast<std::int64_t>(rows) * width / 8;
    check(pdl::launch_consumer(config(dim3(static_cast<unsigned>((groups + 255) / 256)), dim3(256), 0, stream),
                               swiglu_kernel, gate, up, groups, h),
          "swiglu");
}

std::size_t dspark_markov_scratch_bytes(std::int32_t) { return sizeof(MarkovScratch); }

void dspark_markov(const DSparkMarkovArgs& a, std::int32_t steps, cudaStream_t stream) {
    const bool q8 = a.fold_codes != nullptr && a.fold_scales != nullptr;
    if (a.projection == nullptr || a.bias == nullptr || a.block == nullptr || a.f_ring == nullptr || a.f_ring_rows <= 0 ||
        a.embedding == nullptr || (a.fold == nullptr && !q8) || a.fold_bias == nullptr || a.base == nullptr || a.vocab <= 0 ||
        a.round == nullptr || a.tokens == nullptr || a.probabilities == nullptr || a.scratch == nullptr || steps < 1 ||
        steps > kMarkovSteps) {
        throw std::invalid_argument("inkling2 dspark_markov: every operand and 1-7 steps");
    }
    static_assert(kRank3 % (kMarkovWarps * 4) == 0 && kRank3 / kMarkovWarps / 4 == kGateCtas, "gate rows");
    check(pdl::launch_consumer(config(dim3(kRank3 / kMarkovWarps), dim3(32 * kMarkovWarps), 0, stream),
                               markov_invariant_kernel, a, static_cast<int>(steps)),
          "markov invariant");
    for (int step = 1; step <= steps; ++step) {
        check(pdl::launch_consumer(config(dim3(kGateCtas), dim3(32 * kMarkovWarps), 0, stream), markov_gate_kernel, a, step),
              "markov gates");
        check(q8 ? pdl::launch_consumer(config(dim3(kLogitCtas), dim3(32 * kMarkovWarps), 0, stream),
                                        markov_logits_kernel<true>, a, step)
                 : pdl::launch_consumer(config(dim3(kLogitCtas), dim3(32 * kMarkovWarps), 0, stream),
                                        markov_logits_kernel<false>, a, step),
              "markov logits");
    }
}

} // namespace infernix::ops::inkling2
