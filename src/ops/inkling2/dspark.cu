// Inkling 2's DSpark drafter Ops (include/infernix/ops/inkling2_dspark.h).
//
// dspark_mix: a CTA per (column, consumer) of 288 threads, 8 of H each.
// dspark_rotate: a warp per (row, head), lane l holding elements 4l .. 4l + 3 (RoPE partners 16 lanes apart).
// dspark_attention: a CTA per (key split, KV head) over the head's 24 query rows (8 block rows x 3 heads): splits of
//   whole 32-key tiles, at least 128 keys, at most `splits` of them (the full layer's 42: two CTAs an SM however long
//   the context), the tiles streamed through a two-stage cp.async ring as BF16, each warp's 3 rows scored a lane a key
//   with their online softmax in registers; then a CTA per (KV head, row) merges the active splits in order.
// dspark_add_norm: a CTA per row; dspark_swiglu: elementwise.
// dspark_markov: the step-invariant projection (a warp per row of 768), then per step a gate kernel (a warp per row,
//   the last CTA applying the gates) and a logits kernel (a warp per vocabulary row, the last CTA merging the CTAs'
//   argmax and softmax sums in CTA order).

#include "infernix/ops/inkling2_dspark.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/canonical_math.h"
#include "ops/common/mma.cuh"
#include "ops/inkling2/kv_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
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
    // A ring row a later column of the lane also takes (a lane's columns are consecutive positions) is left to it:
    // concurrent CTAs storing one row would leave whichever landed last (a call wider than the ring, as a prompt's).
    if (a.ring[m] > 0) {
        const int later = c + a.ring[m];
        if (later < a.columns && (a.lanes == nullptr || a.lanes[later] == a.lanes[c])) { return; }
    }
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
    // Rows from rounds: the launch's round, or (block_rows) block r / block_rows's.
    const DSparkRound* round = a.round == nullptr ? nullptr : a.round + (a.block_rows > 0 ? r / a.block_rows : 0);
    const int in_block       = a.block_rows > 0 ? r % a.block_rows : r;
    const int position = a.positions != nullptr ? a.positions[r] : round->first + in_block;
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
    const int lane_index = a.lanes != nullptr ? a.lanes[r] : (a.lane >= 0 ? a.lane : round->lane);
    if (a.ring == 0 && a.int8.codes_k != nullptr) { // DSparkInt8Cache::active (a host function)
        // The full layer in INT8: the row's K (its BF16 rounding, as the BF16 cache holds it) and V encoded by the warp
        // (the target pools' int8 row codec: lane l holds values 4 l .. 4 l + 3).
        const std::int64_t page = a.table[static_cast<std::int64_t>(lane_index) * a.table_stride + position / 64];
        const std::size_t row   = (static_cast<std::size_t>(page) * kDSparkKvHeads + h) * 64 + position % 64;
        const float key[4]      = {bf(rb(out[0])), bf(rb(out[1])), bf(rb(out[2])), bf(rb(out[3]))};
        (void)detail::kv_encode_int8(key, lane, a.int8.codes_k + row * kD, a.int8.side_k + row * 4, false);
        const uint2 raw_v =
            *reinterpret_cast<const uint2*>(a.values + static_cast<std::size_t>(r) * a.values_stride + h * kD + 4 * lane);
        const float value[4] = {canon::f32_from_bits(raw_v.x << 16), canon::f32_from_bits(raw_v.x & 0xFFFF0000U),
                                canon::f32_from_bits(raw_v.y << 16), canon::f32_from_bits(raw_v.y & 0xFFFF0000U)};
        (void)detail::kv_encode_int8(value, lane, a.int8.codes_v + row * kD, a.int8.side_v + row * 4, false);
        return;
    }
    std::size_t at;
    if (a.ring > 0) {
        // A slot a later row of the lane also takes is left to it (mix_kernel's rule; rows a lane's consecutive positions).
        const int later = r + a.ring;
        if (a.positions != nullptr && later < a.rows && (a.lanes == nullptr || a.lanes[later] == lane_index)) { return; }
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

// A KV head's keys [lo, hi) go to at most a.splits splits of whole 32-key tiles, at least kMinSplitKeys each: a long
// context takes few long CTAs (two an SM for the full layer, so its partials are few to merge) and a short one many
// short ones. The tiles stream through a two-stage cp.async ring as BF16 (attention_split_kernel).
constexpr int kMinSplitKeys = 128;
constexpr int kTileKeys     = 32;
constexpr int kGroupRows    = kDSparkBlock * (kDSparkHeads / kDSparkKvHeads); // 24 query rows a KV head
constexpr int kAttnThreads  = 256;
constexpr int kPartialWords = kD + 2;     // m, l, acc[128]
// The full layer's splits a KV head at most: 8 x 42 = 336 CTAs, two an SM of an RTX 5090's 170.
constexpr int kTargetSplits = 42;

__device__ __forceinline__ void key_range(const DSparkAttentionArgs& a, int& lo, int& hi) {
    const int first = a.round->first;
    hi              = first + kDSparkBlock;
    lo              = a.ring > 0 ? max(0, first - kDSparkWindow) : 0;
}

// The keys a split takes: whole tiles, at least kMinSplitKeys, enough that a.splits splits cover [lo, hi).
__device__ __forceinline__ int split_keys(const DSparkAttentionArgs& a, int lo, int hi) {
    const int per = (hi - lo + a.splits - 1) / a.splits;
    return max(kMinSplitKeys, (per + kTileKeys - 1) / kTileKeys * kTileKeys);
}

__device__ __forceinline__ void cp_async16(void* smem, const void* global) {
    const auto s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(global) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// Block blockIdx.z's arguments (a launch of several blocks: each its own round, queries, output and scratch).
__device__ __forceinline__ DSparkAttentionArgs attention_block(const DSparkAttentionArgs& in) {
    DSparkAttentionArgs a = in;
    const int b           = static_cast<int>(blockIdx.z);
    a.round += b;
    a.q += b * in.q_stride;
    a.out += b * in.out_stride;
    a.scratch += b * in.scratch_stride;
    return a;
}

// A split on tensor cores (m16n8k16 BF16, FP32 accumulation; drafts only: other bits than FP32 folds would give,
// within the same FP64 criterion). Scores S^T = K Q^T: warp w < 6 the 16-key tile w / 3 against the 8-row tile w % 3 (24
// rows: no padding), the head dimension in 16-element steps in order, into s [row][key]; the online softmax (a warp per
// 3 rows, a lane a key, in FP32), each p written as two BF16 terms hi = bf16(p), lo = bf16(p - hi);
// P x V as O^T = V^T P^T: warp w the dims 16 w .. + 15 of every 8-row tile, per 16 keys in order the hi terms' MMA and
// then the lo terms'. Rows padded by 16 bytes (ldmatrix without bank conflicts); a partial tile's missing V rows are
// zeros (their p are 0, and a stale or uninitialized row could hold a NaN).
constexpr int kMmaRow  = kD + 8;              // a BF16 row of 128 and 8 halves of padding
constexpr int kSRow    = kTileKeys + 4;       // scores [row][key]: conflict-free fragment stores
constexpr int kPRow    = 2 * kTileKeys + 8;   // p's hi then lo terms [row][key], 16 bytes of padding

struct AttnSmem {
    std::uint16_t q[kGroupRows][kMmaRow];
    std::uint16_t k[2][kTileKeys][kMmaRow];
    std::uint16_t v[2][kTileKeys][kMmaRow];
    float s[kGroupRows][kSRow];
    std::uint16_t p[kGroupRows][kPRow];
    float scale[kGroupRows];
};

__device__ __forceinline__ unsigned smem_u32(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }

// An INT8 tile piece in registers (8 values of a K row and of its V row, their groups' FP16 scales), fetched a tile
// ahead and decoded into the tile stage after the tile before it computes.
struct Int8Piece {
    uint2 k = {}, v = {};
    std::uint16_t k_scale = 0, v_scale = 0;
    bool live = false;
};

// 8 int8 codes and their group's FP16 scale as 8 represented BF16 words, bf16(code * s) (the product exact in FP32),
// a pair a hardware conversion (cvt.rn.bf16x2.f32: round to nearest even, canon::f32_to_bf16_rn's bits for these
// finite products).
__device__ __forceinline__ uint4 int8_words(uint2 codes, std::uint16_t scale_bits) {
    const float s          = __half2float(__ushort_as_half(scale_bits));
    const std::uint32_t c[2] = {codes.x, codes.y};
    std::uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const std::uint32_t word = c[i / 2] >> (16 * (i % 2));
        const float lo           = static_cast<float>(static_cast<std::int8_t>(word & 0xFFU));
        const float hi           = static_cast<float>(static_cast<std::int8_t>((word >> 8) & 0xFFU));
        const __nv_bfloat162 pair = __floats2bfloat162_rn(canon::mul_rn(lo, s), canon::mul_rn(hi, s));
        w[i]                      = *reinterpret_cast<const std::uint32_t*>(&pair);
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

template <bool Int8>
__global__ void __launch_bounds__(kAttnThreads, 2) attention_split_kernel(const DSparkAttentionArgs args) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const DSparkAttentionArgs a = attention_block(args);
    auto& sm        = *reinterpret_cast<AttnSmem*>(smem_raw);
    const int split = static_cast<int>(blockIdx.x), g = static_cast<int>(blockIdx.y), tid = static_cast<int>(threadIdx.x);
    const int lane = tid % 32, warp = tid / 32;
    pdl::enter();
    int lo = 0, hi = 0;
    key_range(a, lo, hi);
    const int per = split_keys(a, lo, hi);
    const int k0 = lo + split * per, k1 = min(hi, k0 + per);
    float* partial = a.scratch + (static_cast<std::size_t>(g) * a.splits + split) * kGroupRows * kPartialWords;
    if (k0 >= k1) { return; }
    const int lane_index = a.round->lane;
    const auto row_of = [&](int position) -> std::uint32_t {
        if (a.ring > 0) {
            return static_cast<std::uint32_t>(static_cast<std::size_t>(lane_index) * a.lane_stride / kD +
                                              static_cast<std::size_t>(g) * a.ring + position % a.ring);
        }
        const std::uint32_t page = static_cast<std::uint32_t>(
            a.table[static_cast<std::int64_t>(lane_index) * a.table_stride + position / 64]);
        return (page * kDSparkKvHeads + static_cast<std::uint32_t>(g)) * 64U + static_cast<std::uint32_t>(position % 64);
    };
    // A tile's K and V rows into `buffer` (thread t pieces t and t + 256, key = piece / 16); a key past the split: its
    // V row zeros (its K row's scores are masked).
    const auto stage = [&](int t0, int buffer) {
        const int n = min(kTileKeys, k1 - t0);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int piece = tid + j * kAttnThreads, key = piece / 16, c = piece % 16;
            if (key < n) {
                const std::size_t at = static_cast<std::size_t>(row_of(t0 + key)) * kD + 8 * c;
                cp_async16(&sm.k[buffer][key][8 * c], a.cache_k + at);
                cp_async16(&sm.v[buffer][key][8 * c], a.cache_v + at);
            } else {
                *reinterpret_cast<uint4*>(&sm.v[buffer][key][8 * c]) = make_uint4(0U, 0U, 0U, 0U);
            }
        }
        cp_async_commit();
    };
    // INT8: a tile's pieces (thread t pieces t and t + 256 as stage's) fetched into registers, then decoded into a stage.
    Int8Piece ahead[2];
    const auto fetch = [&](int t0) {
        const int n = min(kTileKeys, k1 - t0);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int piece = tid + j * kAttnThreads, key = piece / 16, c = piece % 16;
            ahead[j].live = key < n;
            if (ahead[j].live) {
                const std::size_t row = row_of(t0 + key);
                ahead[j].k            = *reinterpret_cast<const uint2*>(a.int8.codes_k + row * kD + 8 * c);
                ahead[j].v            = *reinterpret_cast<const uint2*>(a.int8.codes_v + row * kD + 8 * c);
                ahead[j].k_scale      = reinterpret_cast<const std::uint16_t*>(a.int8.side_k + row * 4)[c >= 8 ? 1 : 0];
                ahead[j].v_scale      = reinterpret_cast<const std::uint16_t*>(a.int8.side_v + row * 4)[c >= 8 ? 1 : 0];
            }
        }
    };
    const auto put = [&](int buffer) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int piece = tid + j * kAttnThreads, key = piece / 16, c = piece % 16;
            if (ahead[j].live) {
                *reinterpret_cast<uint4*>(&sm.k[buffer][key][8 * c]) = int8_words(ahead[j].k, ahead[j].k_scale);
                *reinterpret_cast<uint4*>(&sm.v[buffer][key][8 * c]) = int8_words(ahead[j].v, ahead[j].v_scale);
            } else {
                *reinterpret_cast<uint4*>(&sm.v[buffer][key][8 * c]) = make_uint4(0U, 0U, 0U, 0U);
            }
        }
    };
    for (int i = tid; i < kGroupRows * kD / 8; i += kAttnThreads) {
        const int q = i / (kD / 8), c = i % (kD / 8), row = q / 3, head = 3 * g + q % 3;
        cp_async16(&sm.q[q][8 * c], a.q + (static_cast<std::size_t>(row) * kDSparkHeads + head) * kD + 8 * c);
    }
    if constexpr (Int8) {
        cp_async_commit(); // the queries' group
        fetch(k0);
        put(0);
    } else {
        stage(k0, 0);
    }
    float m[3] = {-INFINITY, -INFINITY, -INFINITY}, l[3] = {0.0F, 0.0F, 0.0F};
    // P x V accumulators (C fragments of O^T): row tile t's dims 16 w + (lane >> 2) (slots 0, 1) and + 8 (slots 2, 3),
    // rows 8 t + 2 (lane & 3) (slots 0, 2) and + 1 (slots 1, 3).
    float acc[3][4] = {};
    const float score_scale = static_cast<float>(0.088388347648318440550); // fl32(128^-0.5)
    int buffer = 0;
    for (int t0 = k0; t0 < k1; t0 += kTileKeys, buffer ^= 1) {
        const int n = min(kTileKeys, k1 - t0);
        __syncthreads(); // the last tile's readers are done with the other stage, the scores and the probabilities
        if constexpr (Int8) {
            // This tile was decoded into `buffer` at the end of the last one (or before the loop); the next one's codes
            // load into registers now and decode into the other stage once this tile has computed.
            if (t0 + kTileKeys < k1) { fetch(t0 + kTileKeys); }
            cp_async_wait<0>(); // the queries
        } else if (t0 + kTileKeys < k1) {
            stage(t0 + kTileKeys, buffer ^ 1);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        if (warp < 6) {
            const int mt = warp / 3, nt = warp % 3;
            float c[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            const std::uint16_t* ka = &sm.k[buffer][16 * mt + (lane & 15)][8 * (lane >> 4)];
            const std::uint16_t* qb = &sm.q[8 * nt + (lane & 7)][8 * (lane >> 3)];
#pragma unroll
            for (int step = 0; step < kD / 16; step += 2) {
                unsigned kf[4], qf[4];
                ldmatrix_x4(qf[0], qf[1], qf[2], qf[3], smem_u32(qb + 16 * step)); // steps step, step + 1
                ldmatrix_x4(kf[0], kf[1], kf[2], kf[3], smem_u32(ka + 16 * step));
                mma_bf16(c[0], c[1], c[2], c[3], kf[0], kf[1], kf[2], kf[3], qf[0], qf[1]);
                ldmatrix_x4(kf[0], kf[1], kf[2], kf[3], smem_u32(ka + 16 * (step + 1)));
                mma_bf16(c[0], c[1], c[2], c[3], kf[0], kf[1], kf[2], kf[3], qf[2], qf[3]);
            }
            const int key = 16 * mt + (lane >> 2), row = 8 * nt + 2 * (lane & 3);
            sm.s[row][key]         = c[0];
            sm.s[row + 1][key]     = c[1];
            sm.s[row][key + 8]     = c[2];
            sm.s[row + 1][key + 8] = c[3];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int row = 3 * warp + j;
            const float x = lane < n ? canon::mul_rn(sm.s[row][lane], score_scale) : -INFINITY;
            float top     = x;
#pragma unroll
            for (int o = 16; o >= 1; o /= 2) { top = fmaxf(top, __shfl_xor_sync(kAll, top, o)); }
            const float m_new = fmaxf(m[j], top);
            const float p     = lane < n ? canon::exp_c(__fsub_rn(x, m_new)) : 0.0F;
            const float psum  = warp_sum(p);
            const float scale = m[j] == -INFINITY ? 0.0F : canon::exp_c(__fsub_rn(m[j], m_new));
            l[j]              = canon::add_rn(canon::mul_rn(l[j], scale), psum);
            m[j]              = m_new;
            const std::uint16_t ph = rb(p);
            sm.p[row][lane]             = ph;
            sm.p[row][kTileKeys + lane] = rb(__fsub_rn(p, bf(ph)));
            if (lane == 0) { sm.scale[row] = scale; }
        }
        __syncthreads();
        {
#pragma unroll
            for (int nt = 0; nt < 3; ++nt) {
                const float s0 = sm.scale[8 * nt + 2 * (lane & 3)], s1 = sm.scale[8 * nt + 2 * (lane & 3) + 1];
                acc[nt][0] = canon::mul_rn(acc[nt][0], s0);
                acc[nt][1] = canon::mul_rn(acc[nt][1], s1);
                acc[nt][2] = canon::mul_rn(acc[nt][2], s0);
                acc[nt][3] = canon::mul_rn(acc[nt][3], s1);
            }
#pragma unroll
            for (int ks = 0; ks < kTileKeys / 16; ++ks) {
                unsigned vf[4];
                ldmatrix_x4_t(vf[0], vf[1], vf[2], vf[3],
                              smem_u32(&sm.v[buffer][16 * ks + (lane & 7) + 8 * (lane >> 4)][16 * warp + 8 * ((lane >> 3) & 1)]));
#pragma unroll
                for (int nt = 0; nt < 3; ++nt) {
                    unsigned ph[2], pl[2];
                    const std::uint16_t* pb = &sm.p[8 * nt + (lane & 7)][16 * ks + 8 * ((lane >> 3) & 1)];
                    ldmatrix_x2(ph[0], ph[1], smem_u32(pb));
                    ldmatrix_x2(pl[0], pl[1], smem_u32(pb + kTileKeys));
                    mma_bf16(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], vf[0], vf[1], vf[2], vf[3], ph[0], ph[1]);
                    mma_bf16(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], vf[0], vf[1], vf[2], vf[3], pl[0], pl[1]);
                }
            }
        }
        // INT8: the next tile into the other stage (its last readers finished before this tile's first barrier; the
        // next tile's first barrier publishes it).
        if constexpr (Int8) {
            if (t0 + kTileKeys < k1) { put(buffer ^ 1); }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            float* out = partial + static_cast<std::size_t>(3 * warp + j) * kPartialWords;
            out[0]     = m[j];
            out[1]     = l[j];
        }
    }
#pragma unroll
    for (int nt = 0; nt < 3; ++nt) {
        const int dim = 16 * warp + (lane >> 2), row = 8 * nt + 2 * (lane & 3);
        partial[static_cast<std::size_t>(row) * kPartialWords + 2 + dim]           = acc[nt][0];
        partial[static_cast<std::size_t>(row + 1) * kPartialWords + 2 + dim]       = acc[nt][1];
        partial[static_cast<std::size_t>(row) * kPartialWords + 2 + dim + 8]       = acc[nt][2];
        partial[static_cast<std::size_t>(row + 1) * kPartialWords + 2 + dim + 8]   = acc[nt][3];
    }
}

// The splits' merge in split order: a CTA per (KV head, row), a thread per dim.
__global__ void __launch_bounds__(kD) attention_merge_kernel(const DSparkAttentionArgs args) {
    const DSparkAttentionArgs a = attention_block(args);
    const int g = static_cast<int>(blockIdx.x), q = static_cast<int>(blockIdx.y), d = static_cast<int>(threadIdx.x);
    pdl::enter();
    int lo = 0, hi = 0;
    key_range(a, lo, hi);
    const int per    = split_keys(a, lo, hi);
    const int active = (hi - lo + per - 1) / per;
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

// Block blockIdx.y's arguments (a launch of several blocks: each its own round, rows, base, scratch and drafts).
__device__ __forceinline__ DSparkMarkovArgs markov_block(const DSparkMarkovArgs& in) {
    DSparkMarkovArgs a = in;
    const int b        = static_cast<int>(blockIdx.y);
    a.round += b;
    a.block += b * in.block_stride;
    a.base += b * in.base_stride;
    a.scratch += b * in.scratch_stride;
    a.tokens += b * kDSparkBlock;
    a.probabilities += b * kDSparkBlock;
    return a;
}

// invariant[i][row] = sum_k W[row][2 rank + k] [x_out[i + 1]; f]_k + bias[row] (FP32): a warp per row.
__global__ void __launch_bounds__(32 * kMarkovWarps) markov_invariant_kernel(const DSparkMarkovArgs args, int steps) {
    const DSparkMarkovArgs a = markov_block(args);
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
__global__ void __launch_bounds__(32 * kMarkovWarps) markov_gate_kernel(const DSparkMarkovArgs args, int step) {
    const DSparkMarkovArgs a = markov_block(args);
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
__global__ void __launch_bounds__(32 * kMarkovWarps) markov_logits_kernel(const DSparkMarkovArgs args, int step) {
    const DSparkMarkovArgs a = markov_block(args);
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
    // The whole vocabulary over 512 CTAs; a shortlist over CTAs of one 32-row batch a warp (dspark_markov's grid).
    const int per_cta = (a.vocab + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x);
    const int v0 = static_cast<int>(blockIdx.x) * per_cta, v1 = min(a.vocab, v0 + per_cta);
    const float* base = a.base + static_cast<std::size_t>(step - 1) * a.vocab;
    float best = -INFINITY, run_max = -INFINITY, run_sum = 0.0F;
    int best_index = 0x7FFFFFFF;
    // A warp takes 32 rows at a time: each lane its 8 of every row's 256 products, then a transposing butterfly (31
    // shuffles) leaves row v + lane's dot on lane `lane`, which alone forms its logit (one lane a row instead of 32).
    for (int v = v0 + 32 * warp; v < v1; v += 32 * kMarkovWarps) {
        float p[32];
#pragma unroll
        for (int r = 0; r < 32; ++r) {
            p[r] = 0.0F;
            if (v + r >= v1) { continue; }
            const std::size_t row = static_cast<std::size_t>(v + r);
            float fv[8];
            if constexpr (Q8) {
                // code * scale, exact in FP32 (8 by 11 significant bits): each byte c as 2^23 + (c + 128) by a byte
                // permute into 0x4B000000, less 2^23 + 128.
                const uint2 code = __ldg(reinterpret_cast<const uint2*>(a.fold_codes + row * kMarkovRank) + lane);
                const float s    = __half2float(__ushort_as_half(__ldg(a.fold_scales + row * (kMarkovRank / 32) + lane / 4)));
                const unsigned lo = code.x ^ 0x80808080U, hi = code.y ^ 0x80808080U;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    fv[j]     = (__uint_as_float(__byte_perm(lo, 0x4B000000U, 0x7650U + j)) - 8388736.0F) * s;
                    fv[4 + j] = (__uint_as_float(__byte_perm(hi, 0x4B000000U, 0x7650U + j)) - 8388736.0F) * s;
                }
            } else {
                load8(a.fold + row * kMarkovRank + 8 * lane, fv);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) { p[r] = __fmaf_rn(fv[j], l8[j], p[r]); }
        }
        // Level k: rows r and r + k pair up; lanes without bit k keep row r, lanes with it keep r + k, each adding its
        // partner's partial of the row it keeps. After the five levels p[0] holds row v + lane's sum.
#pragma unroll
        for (int k = 16; k >= 1; k >>= 1) {
            const bool upper = (lane & k) != 0;
#pragma unroll
            for (int r = 0; r < k; ++r) {
                const float send = upper ? p[r] : p[r + k];
                const float keep = upper ? p[r + k] : p[r];
                p[r]             = __fadd_rn(keep, __shfl_xor_sync(0xFFFFFFFFU, send, k));
            }
        }
        const int mine = v + lane;
        if (mine < v1) {
            const float corr = bf(rb(canon::add_rn(p[0], bf(a.fold_bias[mine]))));
            const float z    = bf(rb(canon::add_rn(bf(rb(base[mine])), corr)));
            const float L    = bf(rb(canon::softcap_c(z, a.softcap)));
            if (L > best) { // a lane's rows ascend: the first maximum is its lowest id
                best       = L;
                best_index = mine;
            }
            if (L > run_max) {
                run_sum = run_max == -INFINITY ? 1.0F : __fmaf_rn(run_sum, canon::exp_c(__fsub_rn(run_max, L)), 1.0F);
                run_max = L;
            } else {
                run_sum = canon::add_rn(run_sum, canon::exp_c(__fsub_rn(L, run_max)));
            }
        }
    }
    // The warp's lanes in lane order (the lowest id on ties), lane 0 the result.
    for (int src = 1; src < 32; ++src) {
        const float cb = __shfl_sync(0xFFFFFFFFU, best, src), cm = __shfl_sync(0xFFFFFFFFU, run_max, src);
        const float cs = __shfl_sync(0xFFFFFFFFU, run_sum, src);
        const int ci   = __shfl_sync(0xFFFFFFFFU, best_index, src);
        if (lane == 0) {
            if (cb > best || (cb == best && ci < best_index)) {
                best       = cb;
                best_index = ci;
            }
            if (cm != -INFINITY) {
                if (run_max == -INFINITY) {
                    run_max = cm;
                    run_sum = cs;
                } else if (cm > run_max) {
                    run_sum = __fmaf_rn(run_sum, canon::exp_c(__fsub_rn(run_max, cm)), cs);
                    run_max = cm;
                } else {
                    run_sum = __fmaf_rn(cs, canon::exp_c(__fsub_rn(cm, run_max)), run_sum);
                }
            }
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
    if (last == 0) { return; }
    __threadfence();
    // The CTAs' partials merged by a fixed binary tree in CTA order (each thread two CTAs, then halving): the argmax
    // (the lowest id on ties) is the one-thread merge's; the softmax sum is associated by the tree, deterministically.
    // (A one-thread pass over 512 CTAs' partials held each step for most of its time: ncu 2026-11-23, issue slots 12 %
    // busy.)
    static_assert(kLogitCtas == 2 * 32 * kMarkovWarps, "a thread for each two CTAs' partials");
    const auto merge = [](float& b, int& bi, float& m, float& s, float cb, int ci, float cm, float cs) {
        if (cb > b || (cb == b && ci < bi)) {
            b  = cb;
            bi = ci;
        }
        if (cm == -INFINITY) { return; }
        if (m == -INFINITY) {
            m = cm;
            s = cs;
        } else if (cm > m) {
            s = __fmaf_rn(s, canon::exp_c(__fsub_rn(m, cm)), cs);
            m = cm;
        } else {
            s = __fmaf_rn(cs, canon::exp_c(__fsub_rn(cm, m)), s);
        }
    };
    __shared__ float tb[kLogitCtas / 2], tm[kLogitCtas / 2], ts[kLogitCtas / 2];
    __shared__ int ti[kLogitCtas / 2];
    const int t = static_cast<int>(threadIdx.x);
    {
        float b = -INFINITY, m = -INFINITY, s = 0.0F;
        int bi  = 0x7FFFFFFF;
        for (int c = 2 * t; c < min(2 * t + 2, static_cast<int>(gridDim.x)); ++c) { // a shortlist's grid may be smaller
            merge(b, bi, m, s, __ldcg(sc.best + c), __ldcg(sc.best_index + c), __ldcg(sc.run_max + c), __ldcg(sc.run_sum + c));
        }
        tb[t] = b, ti[t] = bi, tm[t] = m, ts[t] = s;
    }
    for (int stride = 1; stride < kLogitCtas / 2; stride *= 2) {
        __syncthreads();
        if (t % (2 * stride) == 0) {
            float b = tb[t], m = tm[t], s = ts[t];
            int bi  = ti[t];
            merge(b, bi, m, s, tb[t + stride], ti[t + stride], tm[t + stride], ts[t + stride]);
            tb[t] = b, ti[t] = bi, tm[t] = m, ts[t] = s;
        }
    }
    __syncthreads();
    if (t != 0) { return; }
    const float b = tb[0], m = tm[0], s = ts[0];
    // Every logit NaN (a non-finite drafter state): no row won, so the draft is row 0 with probability 0 (a valid id the
    // verification rejects or accepts on the target's own terms), never an index past the rows.
    const bool any = ti[0] < a.vocab;
    const int bi   = any ? ti[0] : 0;
    a.tokens[step - 1]        = a.row_ids != nullptr ? a.row_ids[bi] : bi;
    a.probabilities[step - 1] = any ? canon::div_rn(canon::exp_c(__fsub_rn(b, m)), s) : 0.0F;
    sc.logits_done            = 0U;
}

__global__ void __launch_bounds__(256) gather_rows_kernel(const std::uint16_t* src, int row_words, const std::int32_t* ids,
                                                          int rows, std::uint16_t* out) {
    const long long i = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<long long>(rows) * row_words) { return; }
    const long long row = i / row_words, word = i % row_words;
    out[i] = src[static_cast<long long>(ids[row]) * row_words + word];
}

// A CTA per (block column j, row b): column first[b] + j of the call (j < counts[b]) to column b * width + j.
__global__ void __launch_bounds__(256) gather_columns_kernel(const std::uint16_t* features, std::int32_t column_words,
                                                             const std::int32_t* positions, const std::int32_t* words,
                                                             std::int32_t padded, std::int32_t width,
                                                             std::uint16_t* out_features, std::int32_t* out_positions) {
    const int j = static_cast<int>(blockIdx.x), b = static_cast<int>(blockIdx.y);
    const std::int32_t count = words[b], first = words[padded + b];
    if (j >= count) { return; }
    const long long from = static_cast<long long>(first) + j, to = static_cast<long long>(b) * width + j;
    const auto* src = reinterpret_cast<const uint4*>(features + from * column_words);
    auto* dst       = reinterpret_cast<uint4*>(out_features + to * column_words);
    for (int i = static_cast<int>(threadIdx.x); i < column_words / 8; i += 256) { dst[i] = src[i]; }
    if (threadIdx.x == 0) { out_positions[to] = positions[from]; }
}

} // namespace

void dflash2_gather_columns(const std::uint16_t* features, std::int32_t column_words, const std::int32_t* positions,
                            const std::int32_t* words, std::int32_t padded, std::int32_t rows, std::int32_t width,
                            std::uint16_t* out_features, std::int32_t* out_positions, cudaStream_t stream) {
    if (features == nullptr || positions == nullptr || words == nullptr || out_features == nullptr ||
        out_positions == nullptr || column_words <= 0 || column_words % 8 != 0 || rows <= 0 || width <= 0 ||
        padded < rows) {
        throw std::invalid_argument("inkling2 dflash2_gather_columns: features, positions, the rows' words and outputs");
    }
    gather_columns_kernel<<<dim3(static_cast<unsigned>(width), static_cast<unsigned>(rows)), 256, 0, stream>>>(
        features, column_words, positions, words, padded, width, out_features, out_positions);
    check(cudaGetLastError(), "gather columns");
}

void dspark_gather_rows(const std::uint16_t* src, std::int32_t row_words, const std::int32_t* ids, std::int32_t rows,
                        std::uint16_t* out, cudaStream_t stream) {
    if (src == nullptr || ids == nullptr || out == nullptr || row_words <= 0 || rows <= 0) {
        throw std::invalid_argument("inkling2 dspark_gather_rows: a table, its row ids and an output");
    }
    const long long words = static_cast<long long>(rows) * row_words;
    gather_rows_kernel<<<static_cast<unsigned>((words + 255) / 256), 256, 0, stream>>>(src, row_words, ids, rows, out);
    check(cudaGetLastError(), "gather rows");
}

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
    const bool int8 = a.dense == nullptr && a.ring <= 0 && a.int8.active();
    if (a.x == nullptr || a.norm == nullptr || a.rows <= 0 || a.heads <= 0 || a.x_stride < a.heads * kD ||
        (a.positions == nullptr && a.round == nullptr) || a.inv_freq == nullptr ||
        (a.dense == nullptr && ((!int8 && a.cache_k == nullptr) || (a.values != nullptr && !int8 && a.cache_v == nullptr) ||
                                (a.lanes == nullptr && a.lane < 0 && a.round == nullptr) ||
                                (a.ring <= 0 && (a.table == nullptr || a.table_stride <= 0))))) {
        throw std::invalid_argument("inkling2 dspark_rotate: rows, heads, a norm, positions and one destination");
    }
    if (int8 && (a.values == nullptr || a.heads != kDSparkKvHeads || a.int8.codes_v == nullptr || a.int8.side_k == nullptr ||
                 a.int8.side_v == nullptr)) {
        throw std::invalid_argument("inkling2 dspark_rotate: the INT8 full layer takes keys and values, all four planes");
    }
    check(pdl::launch_consumer(config(dim3(a.rows, a.heads), dim3(32), 0, stream), rotate_kernel, a), "rotate");
}

void dspark_inv_freq(double theta, float (&inv_freq)[kDSparkHeadDim / 2]) {
    for (int i = 0; i < kDSparkHeadDim / 2; ++i) {
        inv_freq[i] = static_cast<float>(std::pow(theta, -static_cast<double>(i) / (kDSparkHeadDim / 2)));
    }
}

std::int32_t dspark_attention_splits(bool sliding, std::uint32_t max_context) {
    // A sliding layer's at most 1,032 keys in splits of 128 (9); the full layer's in at most kTargetSplits.
    const std::uint64_t keys = sliding ? std::uint64_t{kDSparkWindow} + kDSparkBlock : std::uint64_t{max_context} + kDSparkBlock;
    const auto splits = static_cast<std::int32_t>((keys + kMinSplitKeys - 1) / kMinSplitKeys);
    return sliding ? splits : std::min(splits, kTargetSplits);
}

std::size_t dspark_attention_scratch_bytes(std::int32_t splits) {
    return static_cast<std::size_t>(kDSparkKvHeads) * splits * kGroupRows * kPartialWords * sizeof(float);
}

void dspark_attention(const DSparkAttentionArgs& a, cudaStream_t stream) {
    const bool int8 = a.ring <= 0 && a.int8.active();
    if (a.q == nullptr || (!int8 && (a.cache_k == nullptr || a.cache_v == nullptr)) || a.round == nullptr ||
        a.scratch == nullptr || a.out == nullptr || a.splits <= 0 || (a.ring <= 0 && (a.table == nullptr || a.table_stride <= 0))) {
        throw std::invalid_argument("inkling2 dspark_attention: queries, a cache, the round, scratch and splits");
    }
    if (int8 && (a.int8.codes_v == nullptr || a.int8.side_k == nullptr || a.int8.side_v == nullptr ||
                 reinterpret_cast<std::uintptr_t>(a.int8.codes_k) % 8 != 0 ||
                 reinterpret_cast<std::uintptr_t>(a.int8.codes_v) % 8 != 0 ||
                 reinterpret_cast<std::uintptr_t>(a.int8.side_k) % 4 != 0 ||
                 reinterpret_cast<std::uintptr_t>(a.int8.side_v) % 4 != 0)) {
        throw std::invalid_argument("inkling2 dspark_attention: the INT8 full layer's four planes, 8- and 4-byte aligned");
    }
    if (reinterpret_cast<std::uintptr_t>(a.q) % 16 != 0 ||
        (!int8 && (reinterpret_cast<std::uintptr_t>(a.cache_k) % 16 != 0 || reinterpret_cast<std::uintptr_t>(a.cache_v) % 16 != 0))) {
        throw std::invalid_argument("inkling2 dspark_attention: 16-byte aligned queries and caches (16-byte row loads)");
    }
    static const bool configured = [] {
        check(cudaFuncSetAttribute(attention_split_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(sizeof(AttnSmem))),
              "attribute");
        check(cudaFuncSetAttribute(attention_split_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(sizeof(AttnSmem))),
              "attribute");
        return true;
    }();
    (void)configured;
    if (a.blocks < 1 || (a.blocks > 1 && (a.q_stride <= 0 || a.out_stride <= 0 || a.scratch_stride <= 0))) {
        throw std::invalid_argument("inkling2 dspark_attention: blocks need their strides");
    }
    const auto blocks = static_cast<unsigned>(a.blocks);
    const auto split_config = config(dim3(a.splits, kDSparkKvHeads, blocks), dim3(kAttnThreads), sizeof(AttnSmem), stream);
    check(int8 ? pdl::launch_consumer(split_config, attention_split_kernel<true>, a)
               : pdl::launch_consumer(split_config, attention_split_kernel<false>, a),
          "attention");
    check(pdl::launch_consumer(config(dim3(kDSparkKvHeads, kGroupRows, blocks), dim3(kD), 0, stream),
                               attention_merge_kernel, a),
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
    if (a.blocks < 1 || (a.blocks > 1 && (a.block_stride <= 0 || a.base_stride <= 0 || a.scratch_stride <= 0))) {
        throw std::invalid_argument("inkling2 dspark_markov: blocks need their strides");
    }
    if (reinterpret_cast<std::uintptr_t>(a.scratch) % 16 != 0 || (a.blocks > 1 && a.scratch_stride % 4 != 0)) {
        throw std::invalid_argument("inkling2 dspark_markov: 16-byte aligned scratch slices (16-byte loads)");
    }
    const auto blocks = static_cast<unsigned>(a.blocks);
    static_assert(kRank3 % (kMarkovWarps * 4) == 0 && kRank3 / kMarkovWarps / 4 == kGateCtas, "gate rows");
    check(pdl::launch_consumer(config(dim3(kRank3 / kMarkovWarps, blocks), dim3(32 * kMarkovWarps), 0, stream),
                               markov_invariant_kernel, a, static_cast<int>(steps)),
          "markov invariant");
    // A shortlist: one 32-row batch a warp (fewer rows than 512 CTAs' would leave warps idle).
    const int batch_ctas      = (a.vocab + 32 * kMarkovWarps - 1) / (32 * kMarkovWarps);
    const unsigned logit_ctas = static_cast<unsigned>(a.row_ids != nullptr && batch_ctas < kLogitCtas ? batch_ctas : kLogitCtas);
    for (int step = 1; step <= steps; ++step) {
        check(pdl::launch_consumer(config(dim3(kGateCtas, blocks), dim3(32 * kMarkovWarps), 0, stream), markov_gate_kernel, a, step),
              "markov gates");
        check(q8 ? pdl::launch_consumer(config(dim3(logit_ctas, blocks), dim3(32 * kMarkovWarps), 0, stream),
                                        markov_logits_kernel<true>, a, step)
                 : pdl::launch_consumer(config(dim3(logit_ctas, blocks), dim3(32 * kMarkovWarps), 0, stream),
                                        markov_logits_kernel<false>, a, step),
              "markov logits");
    }
}

} // namespace infernix::ops::inkling2
