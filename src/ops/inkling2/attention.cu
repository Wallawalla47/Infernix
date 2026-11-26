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
#include "ops/ggml/ggml_a8.h"
#include "ops/inkling2/attention_common.cuh"
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
using detail::early_history;
using detail::history_row;
using detail::history_source;
using detail::ring_row;
using detail::Source;
using detail::writes_ring;

using detail::ex2;
using detail::kD;
using detail::kInvSqrtD;
using detail::kKvWidth;
using detail::kLog2e;
using detail::kQWidth;
using detail::kRWidth;
using detail::kv_row;
using detail::neg_inf;

constexpr unsigned kAll        = 0xFFFFFFFFU;
constexpr int kStatWords       = 132;
constexpr int kAccOffset       = 4;
constexpr float kClipQK        = kQkClipThreshold;
constexpr float kClipR         = kRelativeClipThreshold;
// The lean quantized instances' forms (new_kv's Form).
constexpr int kInt8            = static_cast<int>(detail::KvForm::Int8);
constexpr int kBf16p           = static_cast<int>(detail::KvForm::Bf16p);

__device__ __forceinline__ float bf16f(std::uint16_t v) { return canon::bf16_to_f32(v); }

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
// The A8 codes of v (a row's elements at .. at + 3; aligned groups of 8 lanes hold one 32-slice) and, on a group's
// first lane, the slice's scale, into a K = kQWidth A8 row: ops/ggml/ggml_a8.h's codec of the BF16 values.
__device__ __forceinline__ void a8_store4(const float (&v)[4], std::uint8_t* row, int at) {
    namespace a8 = ::infernix::ggml::a8;
    std::uint16_t bits[4];
    unsigned key = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        bits[e] = canon::f32_to_bf16_rn(v[e]);
        key     = max(key, static_cast<unsigned>(bits[e] & 0x7FFFU));
    }
#pragma unroll
    for (int o = 1; o <= 4; o <<= 1) { key = max(key, __shfl_xor_sync(kAll, key, o)); }
    const a8::SliceScale sc = a8::slice_scale(key);
    unsigned packed = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) { packed |= (static_cast<unsigned>(a8::code(bits[e], sc.inv)) & 0xFFU) << (8 * e); }
    *reinterpret_cast<unsigned*>(row + at) = packed;
    if (at % 32 == 0) { reinterpret_cast<float*>(row + kQWidth)[at / 32] = sc.d; }
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

// ---------------------------------------------------------------------------- prologue

constexpr int kPrologueThreads = 5 * 32;

// The conv taps of a KV head's lane channels 4 lane .. + 3 (which: 0 K, 1 V), [lag][channel].
__device__ __forceinline__ void kv_conv_weights(const AttentionArgs& a, int kv, int which, int lane, float (&w)[4][4]) {
    const std::uint16_t* conv = which == 0 ? a.layer.key_conv : a.layer.value_conv;
#pragma unroll
    for (int lag = 0; lag < 4; ++lag) { weight4(conv + lag * kKvWidth + kv * kD + 4 * lane, w[lag]); }
}

// Whether a call's K and V ring reads may be issued before the PDL wait (column.cuh).
__device__ __forceinline__ bool early_kv_history(const AttentionArgs& a) {
    return early_history(a.key_ring) && early_history(a.value_ring);
}

// Column col's conv-history rows k = 1 .. 3 of the K (which 0) or V ring, KV head kv, lane channels 4 lane .. + 3,
// that come from the ring (Source::Ring), raw (the others zero).
__device__ __forceinline__ void kv_history(const AttentionArgs& a, const Column& col, int kv, int which, int lane,
                                           uint2 (&h)[3]) {
    const Ring& ring = which == 0 ? a.key_ring : a.value_ring;
    const int ch     = kv * kD + 4 * lane;
#pragma unroll
    for (int k = 1; k < 4; ++k) {
        h[k - 1] = history_source(a.map, col, k) == Source::Ring
                       ? *reinterpret_cast<const uint2*>(history_row(ring, col.lane, col.pos - k) + ch)
                       : make_uint2(0U, 0U);
    }
}

// kv_history's rows and the conv taps (kv_conv_weights') into L2: the fused attention's new rows, before its PDL wait,
// holding no registers over the unit loop.
__device__ __forceinline__ void kv_prefetch(const AttentionArgs& a, const Column& col, int kv, int which, int lane) {
    const Ring& ring           = which == 0 ? a.key_ring : a.value_ring;
    const std::uint16_t* conv  = which == 0 ? a.layer.key_conv : a.layer.value_conv;
    const int ch               = kv * kD + 4 * lane;
#pragma unroll
    for (int k = 1; k < 4; ++k) {
        if (history_source(a.map, col, k) == Source::Ring) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(history_row(ring, col.lane, col.pos - k) + ch));
        }
    }
#pragma unroll
    for (int lag = 0; lag < 4; ++lag) { asm volatile("prefetch.global.L2 [%0];" ::"l"(conv + lag * kKvWidth + ch)); }
}

// KvView::encode: a new row of a BF16 view standing in for a quantized window ring. Its codes and side record (at the
// view's encode planes) from the lane's four stored values `held` and their BF16 source words
// `bits`; returns the represented words the view stores (those a ring of the codec decodes the codes to).
// Form: the window codec's form (its K and V forms are the same), a compile-time constant so an instance carries one encoder.
template <int Form>
__device__ __forceinline__ uint2 encode_window_row(const KvView& kv, int which, std::size_t row_element, const float (&held)[4],
                                                   const std::uint16_t (&bits)[4], int lane, int col_lane, int pos) {
    constexpr detail::KvForm form = static_cast<detail::KvForm>(Form);
    std::uint8_t* codes = kv.encode_codes[which] + row_element / kD * detail::code_bytes(form);
    std::uint8_t* side  = kv.encode_side[which] + row_element / kD * detail::side_bytes(form);
    bool unpackable     = false;
    const uint2 words   = detail::kv_encode<false, Form == kBf16p>(form, held, bits, lane, codes, side, which == 1, unpackable);
    if (unpackable && lane == 0 && kv.unpackable != nullptr) { atomicMin(kv.unpackable + col_lane, static_cast<unsigned>(pos)); }
    return words;
}

// Column col's new key (which 0) or value (which 1) row of KV head kv, lane channels 4 lane .. + 3: the short conv of
// its raw input and history (hist: kv_history's rows), rounded; a key clipped by its head norm, a value RMS-normed to
// FP16. Stores the cache row and (writes_ring) the raw input's ring row; returns the stored words (BF16 K or FP16 V
// bits).
// Quantized: the kernel instance for a quantized pool or window ring (KvCodec other than Bf16); the BF16 instance
// carries no codec code. Encode: the BF16 prologue instance of a view with KvView::encode (a staging ring), the window
// codec's detail::KvForm (0: none). Rotated: the quantized instance of a vq2 / k4v2 pool (the others carry no code for them).
// Form: a lean quantized instance's detail::KvForm (Int8 or Bf16p: both planes of that form, one encoder and decoder;
// 0: the planes' forms read from the view).
template <bool Quantized, int Encode = 0, bool Rotated = true, int Form = 0>
__device__ __forceinline__ uint2 new_kv(const AttentionArgs& a, const Column& col, int kv, int which, int lane,
                                        const float (&w)[4][4], const uint2 (&hist)[3]) {
    const int ch = kv * kD + 4 * lane;
    const int t  = col.t;
    {
        const std::uint16_t* raw = which == 0 ? a.in.k : a.in.v;
        const Ring& ring         = which == 0 ? a.key_ring : a.value_ring;
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
            case Source::Ring: unpack4(hist[k - 1], x[k]); break;
            }
        }
        float c[4];
        if (a.layer.full) {
            const std::uint16_t* taps = (which == 0 ? a.in.key_taps : a.in.value_taps) +
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
        const detail::KvForm form = Form != 0 ? static_cast<detail::KvForm>(Form)
                                    : Quantized ? detail::plane_form(a.kv, which) : detail::KvForm::Words;
        const bool quantized      = Quantized && form != detail::KvForm::Words;
        uint2 stored;
        [[maybe_unused]] float held[4];         // the values the BF16 storage holds (a quantized pool's encoder input)
        [[maybe_unused]] std::uint16_t bits[4]; // their BF16 source words (bf16p's input)
        if (which == 0) {
            const float f = clip_factor(sum, kClipQK);
#pragma unroll
            for (int j = 0; j < 4; ++j) { c[j] = canon::mul_rn(c[j], f); }
            stored = pack4(c);
            if constexpr (Quantized || Encode != 0) {
                unpack4(stored, held);
#pragma unroll
                for (int j = 0; j < 4; ++j) { bits[j] = canon::f32_to_bf16_rn(held[j]); }
            }
            if constexpr (Encode != 0) { stored = encode_window_row<Encode>(a.kv, 0, at - 4 * lane, held, bits, lane, col.lane, col.pos); }
            if (!quantized) { *reinterpret_cast<uint2*>(a.kv.k + at) = stored; }
        } else {
            const float r = rms_factor(sum, static_cast<float>(kD));
            std::uint16_t h[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const std::uint16_t b = canon::f32_to_bf16_rn(canon::mul_rn(c[j], r));
                h[j]                  = f16_bits(bf16f(b));
                if constexpr (Quantized || Encode != 0) {
                    bits[j] = b;
                    held[j] = f16f(h[j]);
                }
            }
            stored = make_uint2(std::uint32_t{h[0]} | (std::uint32_t{h[1]} << 16), std::uint32_t{h[2]} | (std::uint32_t{h[3]} << 16));
            if constexpr (Encode != 0) { stored = encode_window_row<Encode>(a.kv, 1, at - 4 * lane, held, bits, lane, col.lane, col.pos); }
            if (!quantized) { *reinterpret_cast<uint2*>(a.kv.v + at) = stored; }
        }
        if (writes_ring(a.map, col)) { *reinterpret_cast<uint2*>(ring_row(ring, col.lane, col.pos) + ch) = pack4(x[0]); }
        if constexpr (Quantized) {
            if (quantized) {
                // The quantized pool: the row's codes and side record over the BF16 storage's values, the represented
                // words returned for the tile.
                const std::size_t row = kv_row(a.kv, col.lane, kv, col.pos);
                bool unpackable       = false;
                const uint2 words     = detail::kv_encode<Rotated, Form == kBf16p>(form, held, bits, lane, detail::row_codes(a.kv, which, form, row),
                                                          detail::row_side(a.kv, which, form, row), which == 1, unpackable);
                if (unpackable && lane == 0 && a.kv.unpackable != nullptr) {
                    atomicMin(a.kv.unpackable + col.lane, static_cast<unsigned>(col.pos));
                }
                return words;
            }
        }
        return stored;
    }
}

// Column col's query (lane dims 4 lane .. + 3, BF16 words) and relative row (lane < 16) of query head `head`: scaled,
// clipped, rounded; full visits times tau, rounded again; a rotated pool's visit then rotated (kv_rotate_query).
template <bool Quantized>
__device__ __forceinline__ void new_qr(const AttentionArgs& a, const Column& col, int head, int lane,
                                       const float (&scale)[4], float rscale, uint2& q_out, std::uint16_t& r_out) {
    const int t       = col.t;
    const float tau_p = a.layer.full ? tau(col.pos) : 1.0F;
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
    if constexpr (Quantized) {
        if (detail::rotated_pool(a)) { detail::kv_rotate_query(q, lane); }
    }
    q_out = pack4(q);
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
    r_out = canon::f32_to_bf16_rn(rv);
}

__device__ __forceinline__ void qr_scales(const AttentionArgs& a, int head, int lane, float (&scale)[4], float& rscale) {
    weight4(a.layer.query_scale + head * kD + 4 * lane, scale);
    rscale = lane < kRelDim ? bf16f(__ldg(a.layer.relative_scale + head * kRelDim + lane)) : 0.0F;
}

// A CTA per (column, KV head): warps 0 and 1 its new K and V rows, warps 2-4 its 3 query heads' q and r into the
// workspace. Padding columns write zero outputs and nothing else. Encode: a staging ring's view (KvView::encode).
// Rotated, Form: as new_kv's.
template <bool Quantized, int Encode, bool Rotated, int Form = 0>
__global__ void __launch_bounds__(kPrologueThreads) prologue_kernel(const AttentionArgs a) {
    const int t = static_cast<int>(blockIdx.x), kv = static_cast<int>(blockIdx.y);
    const int warp = static_cast<int>(threadIdx.x >> 5), lane = static_cast<int>(threadIdx.x & 31U);
    const int head = kv * kGroupHeads + (warp - 2);
    float w[4][4]  = {};
    float scale[4] = {};
    float rscale   = 0.0F;
    if (warp < 2) {
        kv_conv_weights(a, kv, warp, lane, w);
    } else {
        qr_scales(a, head, lane, scale, rscale);
    }
    // The column (the call's upload) and the K / V warps' ring history (rows of earlier calls) before the wait.
    const Column col = column_of(a.map, t);
    const bool early = early_kv_history(a);
    uint2 hist[3]    = {};
    if (warp < 2 && early) { kv_history(a, col, kv, warp, lane, hist); }
    pdl::enter();
    if (col.lane < 0) {
        for (int i = static_cast<int>(threadIdx.x); i < kGroupHeads * kD; i += kPrologueThreads) {
            a.out[static_cast<std::size_t>(t) * kQWidth + kv * kGroupHeads * kD + i] = 0;
        }
        if (a.out8 != nullptr) { // the group's codes and slice scales zero (a zero slice's encoding)
            auto* row = reinterpret_cast<std::uint8_t*>(a.out8 + static_cast<std::size_t>(t) * a.out8_stride);
            for (int i = static_cast<int>(threadIdx.x); i < kGroupHeads * kD / 4; i += kPrologueThreads) {
                reinterpret_cast<std::uint32_t*>(row + kv * kGroupHeads * kD)[i] = 0U;
            }
            for (int i = static_cast<int>(threadIdx.x); i < kGroupHeads * kD / 32; i += kPrologueThreads) {
                reinterpret_cast<float*>(row + kQWidth)[kv * kGroupHeads * kD / 32 + i] = 0.0F;
            }
        }
        return;
    }
    if (warp < 2) {
        if (!early) { kv_history(a, col, kv, warp, lane, hist); }
        (void)new_kv<Quantized, Encode, Rotated, Form>(a, col, kv, warp, lane, w, hist);
        return;
    }
    uint2 q;
    std::uint16_t r;
    new_qr<Quantized && Rotated>(a, col, head, lane, scale, rscale, q, r);
    *reinterpret_cast<uint2*>(a.ws.q + (static_cast<std::size_t>(t) * kHeads + head) * kD + 4 * lane) = q;
    if (lane < kRelDim) { a.ws.r[(static_cast<std::size_t>(t) * kHeads + head) * kRelDim + lane] = r; }
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
    std::uint16_t* k;     // [64][kRowStride]
    std::uint16_t* v;     // [64][kRowStride]
    std::uint8_t* stage;  // the staged INT8 instance's code stage: [2 planes][64 rows x 128 codes, then 64 x 4 side]
    std::uint16_t* q;     // [rows][kRowStride]
    float* r;             // [rows][16]
    float* p;             // [rows][64]
    RowState* rows;       // [rows]
};

// The staged INT8 instance's code stage: a unit's two planes, each its rows' 128 codes and then their 4-byte side
// records (16-byte aligned: 8,192 + 256 bytes a plane).
constexpr int kStagePlane = kKeyUnit * (kD + 4);
constexpr int kStageBytes = 2 * kStagePlane;

// (The other quantized instances decode a unit's codes from global memory into k and v: no code stage.)
std::size_t smem_bytes(int rows, bool staged = false) {
    return 2 * std::size_t{kKeyUnit} * kRowStride * 2 + (staged ? std::size_t{kStageBytes} : 0U) +
           std::size_t(rows) * kRowStride * 2 + std::size_t(rows) * kRelDim * 4 + std::size_t(rows) * kKeyUnit * 4 +
           std::size_t(rows) * sizeof(RowState);
}

__device__ __forceinline__ Smem carve(unsigned char* base, int rows, bool staged = false) {
    Smem s;
    s.k     = reinterpret_cast<std::uint16_t*>(base);
    s.v     = s.k + kKeyUnit * kRowStride;
    s.stage = reinterpret_cast<std::uint8_t*>(s.v + kKeyUnit * kRowStride);
    s.q     = reinterpret_cast<std::uint16_t*>(s.stage + (staged ? kStageBytes : 0));
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

// Form: the lean instance of an int8 G64 or a bf16p pool or window ring (new_kv's: both planes of that form, its piece
// decoder and its encoder). The int8 instance is staged: a unit's codes are copied into the code stage (cp.async) while
// the unit before it computes, then decoded from the stage into k and v (kv_decode8_int8, the generic decode_unit's
// words), without a unit's load latency on the chain (2026-11-26: the same bits as its unstaged decode from global
// memory, which it replaced).
template <bool Quantized, bool Rotated, int Form = 0>
__global__ void __launch_bounds__(kThreads, 1) attention_kernel(const AttentionArgs a) {
    constexpr bool Staged = Form == static_cast<int>(detail::KvForm::Int8);
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ int range[2];
    __shared__ int last;
    const int split = static_cast<int>(blockIdx.x), kv = static_cast<int>(blockIdx.y), group = static_cast<int>(blockIdx.z);
    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, wl = tid & 31;
    const bool full = a.layer.full;
    // The call plan (groups, column map) is uploaded before the forward's first kernel, so it is read before the wait,
    // off the dependent chain to the keys.
    const int c0    = a.groups_columns[3 * group], count = a.groups_columns[3 * group + 1];
    const int lane  = a.groups_columns[3 * group + 2];
    if (count == 0) { return; }
    const int rows  = kGroupHeads * count;
    const int first = a.map.first[lane];
    const int pos0  = a.map.position[c0];
    const Smem s    = carve(smem_raw, kGroupHeads * a.max_group_columns, Staged);

    // Rows: their units of this split (the call plan only).
    const int lane_lo = row_units(full, pos0, first).lo;
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
    // A unit's K and V pages into the tile (BF16), or its codes decoded into the tile's represented words (quantized,
    // synchronously: a shared code stage would hold 26 KB more shared memory a CTA, keeping a next kernel's early CTAs
    // off the SM).
    const auto issue_unit = [&](int b) {
        const std::size_t page = kv_row(a.kv, lane, kv, b * kKeyUnit);
        if constexpr (Staged) {
            // The unit's rows are consecutive: each plane's 8,192 code bytes and 256 side bytes, 16 bytes a copy.
            constexpr int kCodeCopies = kKeyUnit * kD / 16, kPlaneCopies = kCodeCopies + kKeyUnit * 4 / 16;
            for (int i = tid; i < 2 * kPlaneCopies; i += kThreads) {
                const int plane = i / kPlaneCopies, j = i % kPlaneCopies;
                std::uint8_t* dst = s.stage + plane * kStagePlane;
                if (j < kCodeCopies) {
                    cp_async16(dst + 16 * j, detail::row_codes(a.kv, plane, detail::KvForm::Int8, page) + 16 * j);
                } else {
                    const int k = j - kCodeCopies;
                    cp_async16(dst + kKeyUnit * kD + 16 * k, detail::row_side(a.kv, plane, detail::KvForm::Int8, page) + 16 * k);
                }
            }
        } else if constexpr (Form == static_cast<int>(detail::KvForm::Bf16p)) {
            detail::decode_unit_bf16p(a.kv, page, s.k, s.v, [](int row, int d) { return row * kRowStride + d; }, tid,
                                      kThreads);
        } else if constexpr (Quantized) {
            detail::decode_unit<Rotated>(a.kv, page, s.k, s.v, [](int row, int d) { return row * kRowStride + d; }, tid,
                                         kThreads);
        } else {
            for (int i = tid; i < 2 * kKeyUnit * (kD / 8); i += kThreads) {
                const int plane = i / (kKeyUnit * (kD / 8)), rest = i % (kKeyUnit * (kD / 8));
                const int row = rest / (kD / 8), chunk = rest % (kD / 8);
                const std::uint16_t* src = (plane == 0 ? a.kv.k : a.kv.v) + page + row * kD + 8 * chunk;
                cp_async16((plane == 0 ? s.k : s.v) + row * kRowStride + 8 * chunk, src);
            }
        }
    };
    // Fused: no kernel of the round writes the visit's past keys and values (the call's own rows are computed here and
    // patched into the tile), so the first unit's pages are issued before the dependency wait.
    bool early = false;
    if (a.fused) {
        __syncthreads();
        early = range[0] <= range[1];
        if (early) { issue_unit(range[0]); }
        // The new rows' ring history (rows of earlier calls) and conv taps into L2.
        if (early_kv_history(a)) {
            for (int i = warp; i < 2 * count; i += kWarps) { kv_prefetch(a, column_of(a.map, c0 + (i >> 1)), kv, i & 1, wl); }
        }
    }
    // A grid wider than half the SMs does not trigger in pdl::enter; trigger_at_entry lets the next kernel's CTAs launch
    // (and load what they may before their own wait) beside this one's (AttentionArgs).
    if (a.trigger_at_entry) { pdl::trigger_dependents(); }
    pdl::enter();
    // The rows' q and r: from the prologue's workspace, or (fused) computed here in its arithmetic, a warp a row.
    if (a.fused) {
        for (int row = warp; row < rows; row += kWarps) {
            const int t = c0 + row / kGroupHeads, head = kv * kGroupHeads + row % kGroupHeads;
            float scale[4], rscale;
            qr_scales(a, head, wl, scale, rscale);
            uint2 q;
            std::uint16_t r;
            new_qr<Quantized && Rotated>(a, column_of(a.map, t), head, wl, scale, rscale, q, r);
            *reinterpret_cast<uint2*>(s.q + row * kRowStride + 4 * wl) = q;
            if (wl < kRelDim) { s.r[row * kRelDim + wl] = bf16f(r); }
            if (split == 0) { // the workspace's q and r too, as the prologue kernel leaves them
                *reinterpret_cast<uint2*>(a.ws.q + (static_cast<std::size_t>(t) * kHeads + head) * kD + 4 * wl) = q;
                if (wl < kRelDim) { a.ws.r[(static_cast<std::size_t>(t) * kHeads + head) * kRelDim + wl] = r; }
            }
        }
    } else {
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
    }
    __syncthreads();

    // P x V accumulators of this thread: rows pv_rg + 16 m, dims 4 dc .. 4 dc + 3.
    const int dc = tid & 31, pv_rg = tid >> 5;
    float acc[kPvRows][4] = {};
    const int key = tid & (kKeyUnit - 1), sc_rg = tid >> 6;

    for (int b = range[0]; b <= range[1]; ++b) {
        if constexpr (Staged) {
            if (b == range[0] && !early) { issue_unit(b); }
            cp_async_wait_all();
            __syncthreads(); // the stage holds unit b's codes (the previous unit's readers of k and v are done)
            // The unit's words, from the stage: a thread decodes 8-value pieces (kv_decode8_int8), consecutive threads
            // consecutive 16-byte pieces of a row (stores without bank conflicts), each plane in its own loop.
            for (int plane = 0; plane < 2; ++plane) {
                const std::uint8_t* codes = s.stage + plane * kStagePlane;
                const std::uint8_t* side  = codes + kKeyUnit * kD;
                std::uint16_t* tile       = plane == 0 ? s.k : s.v;
                for (int i = tid; i < kKeyUnit * (kD / 8); i += kThreads) {
                    const int row = i / (kD / 8), piece = i % (kD / 8);
                    const uint2 c = *reinterpret_cast<const uint2*>(codes + row * kD + 8 * piece);
                    const std::uint16_t scale = reinterpret_cast<const std::uint16_t*>(side + row * 4)[piece / 8];
                    *reinterpret_cast<uint4*>(tile + row * kRowStride + 8 * piece) =
                        plane == 0 ? detail::kv_decode8_int8<false>(c, scale) : detail::kv_decode8_int8<true>(c, scale);
                }
            }
            __syncthreads(); // k and v hold unit b; the stage is free for the next unit's codes
            if (b + 1 <= range[1]) { issue_unit(b + 1); }
        } else {
            if (!early || b != range[0]) { issue_unit(b); }
            cp_async_wait_all();
            __syncthreads();
        }
        if (a.fused) {
            // The group's new keys and values whose positions lie in this unit: computed here (the prologue's
            // arithmetic: each stores its cache and ring rows; a full visit's unit read by two CTAs stores them twice,
            // the same words) and patched into the tile, which the copy may have read before or after the store.
            for (int i = warp; i < 2 * count; i += kWarps) {
                const int t = c0 + (i >> 1), which = i & 1;
                const int pos = a.map.position[t];
                if (pos < b * kKeyUnit || pos >= (b + 1) * kKeyUnit) { continue; }
                float w[4][4];
                kv_conv_weights(a, kv, which, wl, w);
                const Column col = column_of(a.map, t);
                uint2 hist[3];
                kv_history(a, col, kv, which, wl, hist);
                const uint2 v = new_kv<Quantized, 0, Rotated, Form>(a, col, kv, which, wl, w, hist);
                *reinterpret_cast<uint2*>((which == 0 ? s.k : s.v) + (pos - b * kKeyUnit) * kRowStride + 4 * wl) = v;
            }
            __syncthreads();
        }

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
            // fmaxf ignores a NaN in either operand, so every lane ends on the same maximum and the branch below is
            // warp-uniform (a NaN score cannot leave lanes behind at the full-mask shuffles that follow); on finite
            // scores it is the select's maximum (a zero's sign aside, which no later value sees).
#pragma unroll
            for (int offset = 16; offset >= 1; offset >>= 1) { mu = fmaxf(mu, __shfl_xor_sync(kAll, mu, offset)); }
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
        float* st = a.ws.stats +
                    ((static_cast<std::size_t>(r.t - a.stats_base) * kHeads + r.head) * kFullSplits + r.slot0) * kStatWords;
        if (dc == 0) {
            st[0] = r.m;
            st[1] = r.l;
        }
        *reinterpret_cast<float4*>(st + kAccOffset + 4 * dc) = make_float4(acc[m][0], acc[m][1], acc[m][2], acc[m][3]);
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const unsigned done = atomicAdd(a.ws.counters + group * kKvHeads + kv, 1U);
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
        const float* base =
            a.ws.stats + (static_cast<std::size_t>(t - a.stats_base) * kHeads + head) * kFullSplits * kStatWords;
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
        if constexpr (Quantized && Rotated) {
            if (detail::rotated_pool(a)) { detail::kv_hadamard128(o4, wl, 0x1p-4F); } // back from the rotated domain
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
        if (a.out8 != nullptr) { // W_o's 8-bit inputs: lanes 8s .. 8s + 7 hold the head's slice s
            a8_store4(out, reinterpret_cast<std::uint8_t*>(a.out8 + static_cast<std::size_t>(t) * a.out8_stride),
                      head * kD + 4 * wl);
        }
    }
    if (tid == 0) { a.ws.counters[group * kKvHeads + kv] = 0; }
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
WorkspaceLayout workspace_layout(std::int32_t capacity) {
    if (capacity <= 0) { throw std::invalid_argument("inkling2 attention: a workspace holds at least one column"); }
    const std::size_t c = static_cast<std::size_t>(capacity);
    WorkspaceLayout l{};
    l.q        = 0;
    l.r        = align256(l.q + c * kHeads * kD * 2);
    l.stats    = align256(l.r + c * kHeads * kRelDim * 2);
    l.counters = align256(l.stats + std::size_t{kMaxAttentionColumns} * kHeads * kFullSplits * kStatWords * 4);
    l.total    = align256(l.counters + std::size_t{kMaxAttentionGroups} * kKvHeads * 4);
    return l;
}

} // namespace

std::size_t attention_workspace_bytes(std::int32_t capacity) { return workspace_layout(capacity).total; }

AttentionWorkspace carve_attention_workspace(void* base, std::int32_t capacity) {
    const WorkspaceLayout l = workspace_layout(capacity);
    auto* b                 = static_cast<unsigned char*>(base);
    AttentionWorkspace w;
    w.capacity = capacity;
    w.q        = reinterpret_cast<std::uint16_t*>(b + l.q);
    w.r        = reinterpret_cast<std::uint16_t*>(b + l.r);
    w.stats    = reinterpret_cast<float*>(b + l.stats);
    w.counters = reinterpret_cast<std::uint32_t*>(b + l.counters);
    return w;
}

void detail::check_attention_args(const AttentionArgs& a) {
    if (a.map.lane == nullptr || a.map.position == nullptr || a.map.first == nullptr || a.map.columns != a.columns ||
        a.groups_columns == nullptr) {
        throw std::invalid_argument("inkling2 attention: the call's column map and its groups are needed");
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
            ring->lane_stride < std::int64_t{kRingPositions} * kKvWidth || ring->lane_stride % 8 != 0 ||
            (ring->history != nullptr && (!aligned16(ring->history) || ring->history_lane_stride % 8 != 0 ||
                                          ring->history_lane_stride < std::int64_t{kHistorySlots} * kKvWidth))) {
            throw std::invalid_argument("inkling2 attention: the K and V conv rings of width 1024 are needed");
        }
    }
    if (a.kv.k == nullptr || a.kv.v == nullptr || !aligned16(a.kv.k) || !aligned16(a.kv.v) ||
        (l.full != (a.kv.table != nullptr)) || (l.full && (a.kv.table_stride <= 0 || a.kv.pool_pages <= 0)) ||
        (!l.full && (a.kv.ring_pages < kWindowRingPages ||
                     a.kv.lane_stride < std::int64_t{kKvHeads} * a.kv.ring_pages * kKeyUnit * kD ||
                     a.kv.lane_stride % 8 != 0))) {
        throw std::invalid_argument("inkling2 attention: sliding visits use a ring per lane, full visits a paged pool");
    }
    if (a.kv.codec != KvCodec::Bf16 && (a.kv.k_side == nullptr || a.kv.v_side == nullptr)) {
        throw std::invalid_argument("inkling2 attention: a quantized pool or ring needs its key and value side records");
    }
    if (!l.full && !window_codec_supported(a.kv.codec)) {
        throw std::invalid_argument("inkling2 attention: a window ring holds bf16, bf16p or int8");
    }
    if ((a.kv.codec == KvCodec::Bf16Packed || a.kv.encode == KvCodec::Bf16Packed) && a.kv.unpackable == nullptr) {
        throw std::invalid_argument("inkling2 attention: a bf16p pool or ring needs its lanes' unpackable flags");
    }
    if (a.ws.q == nullptr || a.ws.stats == nullptr || a.out == nullptr) {
        throw std::invalid_argument("inkling2 attention: the workspace and the output are needed");
    }
}

void detail::launch_attention_prologue(const AttentionArgs& a, cudaStream_t stream) {
    const pdl::LaunchConfig config{dim3(static_cast<unsigned>(a.columns), kKvHeads), dim3(kPrologueThreads), 0, stream};
    if (a.kv.encode != KvCodec::Bf16 && (a.kv.encode_codes[0] == nullptr || a.kv.encode_codes[1] == nullptr ||
                                         a.kv.encode_side[0] == nullptr || a.kv.encode_side[1] == nullptr)) {
        throw std::invalid_argument("inkling2 attention: an encoding view needs its code and side planes");
    }
    if (quantized_pool(a)) {
        CUDA_CHECK(rotated_pool(a)                    ? pdl::launch_consumer(config, prologue_kernel<true, 0, true>, a)
                   : a.kv.codec == KvCodec::Int8G64    ? pdl::launch_consumer(config, prologue_kernel<true, 0, false, kInt8>, a)
                   : a.kv.codec == KvCodec::Bf16Packed ? pdl::launch_consumer(config, prologue_kernel<true, 0, false, kBf16p>, a)
                                                       : pdl::launch_consumer(config, prologue_kernel<true, 0, false>, a));
        return;
    }
    switch (a.kv.encode) { // a staging ring's window codec: one encoder an instance
    case KvCodec::Bf16: CUDA_CHECK(pdl::launch_consumer(config, prologue_kernel<false, 0, false>, a)); return;
    case KvCodec::Int8G64: CUDA_CHECK(pdl::launch_consumer(config, prologue_kernel<false, kInt8, false>, a)); return;
    case KvCodec::Bf16Packed: CUDA_CHECK(pdl::launch_consumer(config, prologue_kernel<false, kBf16p, false>, a)); return;
    default: throw std::invalid_argument("inkling2 attention: a staging ring encodes bf16p or int8");
    }
}

void attention(const AttentionArgs& a, cudaStream_t stream) {
    if (a.columns <= 0 || a.columns > a.ws.capacity || a.groups < 0 ||
        (a.groups > 0 && (a.max_group_columns <= 0 || a.max_group_columns > kMaxAttentionLaneColumns))) {
        throw std::invalid_argument("inkling2 attention: 1..capacity columns in groups of at most 16 columns each");
    }
    if (a.host_groups == nullptr && (a.groups > kMaxAttentionGroups || a.columns > kMaxAttentionColumns)) {
        throw std::invalid_argument("inkling2 attention: a call wider than one launch passes its groups on the host");
    }
    detail::check_attention_args(a);
    const AttentionLayer& l = a.layer;
    if (a.fused && a.groups == 0) { throw std::invalid_argument("inkling2 attention: a fused call has no padding column"); }
    if (a.fused && a.kv.encode != KvCodec::Bf16) {
        throw std::invalid_argument("inkling2 attention: a staging ring's call runs the prologue kernel (not fused)");
    }
    if (a.out8 != nullptr && (reinterpret_cast<std::uintptr_t>(a.out8) % 16 != 0 || a.out8_stride * 4 < kQWidth + kQWidth / 8)) {
        throw std::invalid_argument("inkling2 attention: 8-bit outputs need 16-byte aligned rows of 3,072 codes and 96 scales");
    }
    if (!a.fused) { detail::launch_attention_prologue(a, stream); }
    if (a.groups == 0) { return; } // every column is padding: the prologue wrote their zero outputs
    static const bool attribute = [] {
        for (auto* kernel : {attention_kernel<false, false>, attention_kernel<true, false>, attention_kernel<true, true>,
                             attention_kernel<true, false, kBf16p>}) {
            CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(smem_bytes(kMaxRows))));
        }
        CUDA_CHECK(cudaFuncSetAttribute(attention_kernel<true, false, kInt8>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem_bytes(kMaxRows, true))));
        return true;
    }();
    (void)attribute;
    const unsigned splits = l.full ? kFullSplits : kSlidingSplits;
    const bool quantized  = detail::quantized_pool(a);
    const bool rotated    = detail::rotated_pool(a);
    const bool int8       = a.kv.codec == KvCodec::Int8G64;
    const bool bf16p      = a.kv.codec == KvCodec::Bf16Packed;
    const std::size_t bytes = smem_bytes(kGroupHeads * a.max_group_columns, int8); // the int8 instance's code stage
    const auto launch = [&](std::int32_t g0, std::int32_t count, std::int32_t base) {
        AttentionArgs b   = a;
        b.groups_columns  = a.groups_columns + 3 * g0;
        b.groups          = count;
        b.stats_base      = base;
        const pdl::LaunchConfig config{dim3(splits, kKvHeads, static_cast<unsigned>(count)), dim3(kThreads), bytes, stream};
        CUDA_CHECK(!quantized ? pdl::launch_consumer(config, attention_kernel<false, false>, b)
                   : rotated  ? pdl::launch_consumer(config, attention_kernel<true, true>, b)
                   : int8     ? pdl::launch_consumer(config, attention_kernel<true, false, kInt8>, b)
                   : bf16p    ? pdl::launch_consumer(config, attention_kernel<true, false, kBf16p>, b)
                              : pdl::launch_consumer(config, attention_kernel<true, false>, b));
    };
    if (a.host_groups == nullptr) {
        launch(0, a.groups, 0);
        return;
    }
    // Launches of consecutive groups spanning at most kMaxAttentionColumns columns (the statistics' rows) and at most
    // kMaxAttentionGroups groups (the counters).
    for (std::int32_t g0 = 0; g0 < a.groups;) {
        const std::int32_t base = a.host_groups[3 * g0];
        std::int32_t g1         = g0;
        while (g1 < a.groups && g1 - g0 < kMaxAttentionGroups &&
               a.host_groups[3 * g1] + a.host_groups[3 * g1 + 1] - base <= kMaxAttentionColumns) {
            ++g1;
        }
        launch(g0, g1 - g0, base);
        g0 = g1;
    }
}

namespace {

// Block (position, KV head): its K row and V row, a half-warp each: the row's 256 word bytes, or its codes (16 bytes a
// thread) and side record (thread 0 of the half).
__global__ void window_copy_kernel(const WindowCopyArgs a) {
    const int pos = a.from + static_cast<int>(blockIdx.x), head = static_cast<int>(blockIdx.y);
    const int i = static_cast<int>(threadIdx.x) % 16, plane = static_cast<int>(threadIdx.x) / 16;
    const std::size_t s = kv_row(a.src, a.src_lane, head, pos), d = kv_row(a.dst, a.dst_lane, head, pos);
    const detail::KvForm form = detail::plane_form(a.src, plane);
    if (form == detail::KvForm::Words) {
        const auto* src = reinterpret_cast<const uint4*>((plane == 0 ? a.src.k : a.src.v) + s);
        auto* dst       = reinterpret_cast<uint4*>((plane == 0 ? a.dst.k : a.dst.v) + d);
        dst[i]          = src[i];
        return;
    }
    if (a.dst.codec == KvCodec::Bf16) { // a quantized ring into a BF16 staging ring: the represented words, 16 a thread
        if (i < kD / 16) {
            uint4 lo, hi;
            detail::kv_decode16(form, detail::row_codes(a.src, plane, form, s), detail::row_side(a.src, plane, form, s), i,
                                plane == 1, lo, hi);
            std::uint16_t* dst = (plane == 0 ? a.dst.k : a.dst.v) + d + 16 * i;
            *reinterpret_cast<uint4*>(dst)     = lo;
            *reinterpret_cast<uint4*>(dst + 8) = hi;
        }
        return;
    }
    const int code = detail::code_bytes(form), side = detail::side_bytes(form);
    if (16 * i < code) {
        *reinterpret_cast<uint4*>(detail::row_codes(a.dst, plane, form, d) + 16 * i) =
            *reinterpret_cast<const uint4*>(detail::row_codes(a.src, plane, form, s) + 16 * i);
    }
    if (i == 0) {
        const std::uint8_t* from = detail::row_side(a.src, plane, form, s);
        std::uint8_t* to         = detail::row_side(a.dst, plane, form, d);
        for (int b = 0; b < side; ++b) { to[b] = from[b]; }
    }
}

} // namespace

void window_copy(const WindowCopyArgs& a, cudaStream_t stream) {
    for (const KvView* v : {&a.src, &a.dst}) {
        if (v->k == nullptr || v->v == nullptr || v->table != nullptr || !aligned16(v->k) || !aligned16(v->v) ||
            v->ring_pages < kWindowRingPages || v->lane_stride % 8 != 0 ||
            v->lane_stride < std::int64_t{kKvHeads} * v->ring_pages * kKeyUnit * kD) {
            throw std::invalid_argument("inkling2 window_copy: two sliding ring views are needed");
        }
        if (!window_codec_supported(v->codec) || (v->codec != KvCodec::Bf16 && (v->k_side == nullptr || v->v_side == nullptr))) {
            throw std::invalid_argument("inkling2 window_copy: a quantized ring needs its side records");
        }
    }
    if (a.src.codec != a.dst.codec && a.dst.codec != KvCodec::Bf16) {
        throw std::invalid_argument("inkling2 window_copy: both rings hold one codec, or a quantized ring decodes into BF16");
    }
    if (a.from < 0 || a.to < a.from || a.src_lane < 0 || a.dst_lane < 0 ||
        a.to - a.from > kKeyUnit * std::min(a.src.ring_pages, a.dst.ring_pages)) {
        throw std::invalid_argument("inkling2 window_copy: positions within both rings are needed");
    }
    if (a.to == a.from) { return; }
    window_copy_kernel<<<dim3(static_cast<unsigned>(a.to - a.from), kKvHeads), 32, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::inkling2
