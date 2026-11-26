// Inkling 2's attention on the prompt route (include/infernix/ops/inkling2_attention.h attention_prompt): the decode
// route's prologue (the K/V convs, clips, norms and appends; the final q and r), then a tensor-core kernel per (64 of
// the call's columns, KV head) over the rows' fixed absolute 64-key units.
//
// A CTA's 12 warps take 16 rows each (16 consecutive columns of one of the group's 3 query heads). Per unit (K and V
// pages in a double-buffered cp.async ring, 16-byte chunks XOR-swizzled by row; a quantized pool's codes in a cp.async
// code stage while the unit before computes, decoded into one tile stage):
//   bias    the warp's rows' relative bias for the unit's 79 distances, bf16(r . P[:, d]) by m16n8k16 (A the rows' r,
//           B the table's columns; 0 past distance 1023), staged as BF16 [16][80]
//   scores  S = q k^T by m16n8k16 in BF16 from zero over the head dimension, s = S * fl32(1/sqrt 128) + bias,
//           t = s * fl32(log2 e), masked keys -inf (past the row, before the lane's first position, or out of a
//           sliding row's window)
//   softmax per row the unit's m_u; a row with no visible key in the unit is left exactly as it was; else
//           m = max(m, m_u), c = ex2(m_old - m), p = ex2(t - m), l = l c + the row sum of p (four lanes' sums in xor
//           order), acc = acc c + p v by m16n8k16 in FP16 (p and the cached FP16 v), FP32 accumulation over the unit's
//           keys in order
// then l += ex2(sink log2 e - m), a = bf16(bf16(acc / l) * sigmoid(gate)).
// A row folds its units in ascending order from the lane's first unit it can see; the units a CTA adds for its other
// rows change nothing in it, and every value it reads is its own (an MMA's output element reads only its row and
// column), so a column's bits depend only on its position and lane, never on the call's width or its other columns.

#include "infernix/ops/inkling2_attention.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/canonical_math.h"
#include "ops/common/mma.cuh"
#include "ops/inkling2/attention_common.cuh"
#include "ops/inkling2/column.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace infernix::ops::inkling2 {
namespace {

using detail::ex2;
using detail::kD;
using detail::kInvSqrtD;
using detail::kLog2e;
using detail::kQWidth;
using detail::kv_row;
using detail::neg_inf;

constexpr unsigned kAll      = 0xFFFFFFFFU;
constexpr int kTileColumns   = 64;                         // a CTA's columns
constexpr int kWarps         = kGroupHeads * kTileColumns / 16; // 12: 16 rows a warp
constexpr int kThreads       = kWarps * 32;
constexpr int kPageElements  = kKeyUnit * kD;              // a K or V page: 64 x 128
constexpr int kBiasColumns   = 80;                         // 79 distances a unit, a whole n8 tile more
constexpr int kStages        = 2;

constexpr int kTableColumns  = 128; // the distances a CTA's rows meet in one unit: 64 rows + 64 keys - 1, whole

// A call of several lanes' columns (prefill groups): each CTA's columns [begin, end), at most kTileColumns of one lane
// (each lane's columns from its first in tiles of 64). count 0: one lane's call, CTA t its columns 64 t .. 64 t + 63.
constexpr int kMaxPromptTiles = 160; // 8 lanes, a call of at most 9,728 columns
struct PromptTiles {
    std::int32_t count = 0;
    std::int32_t begin[kMaxPromptTiles] = {}, end[kMaxPromptTiles] = {};
};

// The quantized instance's code stage: a unit's codes and side records, a plane each (codes, then side records at a
// fixed offset), sized for the largest form's rows (bf16p: 192 code bytes and 16 side bytes).
constexpr int kCodeRowBytes = 192, kSideRowBytes = 16;
constexpr int kCodePlane    = kKeyUnit * (kCodeRowBytes + kSideRowBytes);

// BF16 pools: two tile stages (a unit's pages copied by cp.async while the unit before it computes). Quantized pools:
// one tile stage and the code stage (a unit's codes copied by cp.async while the unit before it computes, then decoded
// from the stage into the tile).
template <bool Quantized>
struct SharedT {
    static constexpr int kTiles = Quantized ? 1 : kStages;
    std::uint16_t k[kTiles][kPageElements];
    std::uint16_t v[kTiles][kPageElements];
    std::uint16_t bias[kWarps][16][kBiasColumns];
    std::uint16_t table[kTableColumns][kRelDim]; // the relative table's columns of the unit's distances, transposed
    alignas(16) std::uint8_t code[Quantized ? 2 * kCodePlane : 16];
};
static_assert(sizeof(SharedT<false>) <= 99 * 1024 && sizeof(SharedT<true>) <= 99 * 1024);

// Element (row, column) of a swizzled page: 16-byte chunk (column / 8) XOR (row & 7).
__device__ __forceinline__ int swz(int row, int col) { return row * kD + (((col >> 3) ^ (row & 7)) << 3) + (col & 7); }

__device__ __forceinline__ unsigned pack_bf16(std::uint16_t lo, std::uint16_t hi) {
    return static_cast<unsigned>(lo) | (static_cast<unsigned>(hi) << 16);
}
__device__ __forceinline__ unsigned pack_half2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}

// A rotated pool's output (attention_common.cuh rotated_pool): the accumulators' two rows (g: elements 0, 1; g + 8: 2,
// 3; dim 8 n + 2 t4 + (e & 1)) through 2^-4 H128 in FP32, the dims' bit 0 (e), bits 1-2 (the quad's lanes t4), then
// bits 3-6 (n).
__device__ __forceinline__ void unrotate_rows(float (&acc)[16][4], int t4) {
#pragma unroll
    for (int n = 0; n < 16; ++n) {
#pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
            const float x = acc[n][2 * hr], y = acc[n][2 * hr + 1];
            acc[n][2 * hr]     = canon::add_rn(x, y);
            acc[n][2 * hr + 1] = canon::add_rn(x, -y);
        }
    }
#pragma unroll
    for (int s = 1; s <= 2; s <<= 1) {
#pragma unroll
        for (int n = 0; n < 16; ++n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float v    = acc[n][e];
                const float peer = __shfl_xor_sync(kAll, v, s);
                acc[n][e]        = (t4 & s) == 0 ? canon::add_rn(v, peer) : canon::add_rn(peer, -v);
            }
        }
    }
#pragma unroll
    for (int s = 1; s < 16; s <<= 1) {
#pragma unroll
        for (int n = 0; n < 16; ++n) {
            if ((n & s) != 0) { continue; }
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float x = acc[n][e], y = acc[n + s][e];
                acc[n][e]     = canon::add_rn(x, y);
                acc[n + s][e] = canon::add_rn(x, -y);
            }
        }
    }
#pragma unroll
    for (int n = 0; n < 16; ++n) {
#pragma unroll
        for (int e = 0; e < 4; ++e) { acc[n][e] = canon::mul_rn(acc[n][e], 0x1p-4F); }
    }
}

// Quantized: the instance for a quantized pool or window ring (attention.cu's split): the BF16 instance has no decoders.
template <bool Quantized>
__global__ void __launch_bounds__(kThreads, 1) attention_prompt_kernel(const AttentionArgs a, const PromptTiles tiles) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    SharedT<Quantized>& sm = *reinterpret_cast<SharedT<Quantized>*>(smem_raw);
    const int tile = static_cast<int>(blockIdx.x), kv = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, ln = tid & 31;
    const int g = ln >> 2, t4 = ln & 3;
    const bool full = a.layer.full;
    const int c0    = tiles.count == 0 ? tile * kTileColumns : tiles.begin[tile];
    const int c_end = tiles.count == 0 ? min(a.columns, c0 + kTileColumns) : tiles.end[tile];
    pdl::enter();
    const int lane  = a.map.lane[c0];
    const int first = a.map.first[lane];
    // The warp's rows: columns wc0 .. wc0 + 15 (clamped to the call for reads; past it nothing is stored), head hh.
    const int hh = warp / 4, wc0 = c0 + 16 * (warp % 4);
    const int head = kv * kGroupHeads + hh;
    const int p_first = a.map.position[c0];
    const int wpos0   = p_first + (wc0 - c0); // a prompt call's lane: consecutive positions
    const int last_col = c_end - 1;
    const int p_last   = a.map.position[last_col];
    const bool warp_live = wc0 < c_end;
    const int w_last = min(wc0 + 15, last_col);
    const int wpos_last = p_first + (w_last - c0);

    // The CTA's units: from the first row's lowest visible unit to the last row's own.
    const int lo = (full ? first : max(first, max(p_first - (kWindow - 1), 0))) / kKeyUnit;
    const int hi = p_last / kKeyUnit;
    // The warp's units (outside them every row of the warp sees nothing: skipped, a no-op).
    const int w_lo = (full ? first : max(first, max(wpos0 - (kWindow - 1), 0))) / kKeyUnit;
    const int w_hi = wpos_last / kKeyUnit;

    // q of the warp's rows as A fragments (8 k16 steps of the head dimension); r of row (lane & 15) for the bias.
    unsigned qa[8][4];
    std::uint32_t rrow[kRelDim / 2]; // BF16 pairs
    {
        const int rr = min(wc0 + (ln & 15), last_col);
        const std::uint16_t* rp = a.ws.r + (static_cast<std::size_t>(rr) * kHeads + head) * kRelDim;
#pragma unroll
        for (int i = 0; i < kRelDim / 2; ++i) { rrow[i] = *reinterpret_cast<const std::uint32_t*>(rp + 2 * i); }
    }
    {
        const int r0 = min(wc0 + g, last_col), r1 = min(wc0 + g + 8, last_col);
        const std::uint16_t* q0 = a.ws.q + (static_cast<std::size_t>(r0) * kHeads + head) * kD;
        const std::uint16_t* q1 = a.ws.q + (static_cast<std::size_t>(r1) * kHeads + head) * kD;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            qa[s][0] = *reinterpret_cast<const unsigned*>(q0 + 16 * s + 2 * t4);
            qa[s][1] = *reinterpret_cast<const unsigned*>(q1 + 16 * s + 2 * t4);
            qa[s][2] = *reinterpret_cast<const unsigned*>(q0 + 16 * s + 8 + 2 * t4);
            qa[s][3] = *reinterpret_cast<const unsigned*>(q1 + 16 * s + 8 + 2 * t4);
        }
    }
    const int pos_g[2] = {wpos0 + g, wpos0 + g + 8};

    float acc[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0F; }
    float m_row[2] = {neg_inf(), neg_inf()}, l_row[2] = {0.0F, 0.0F};

    // BF16: a unit's K and V pages into a tile stage.
    const auto load_unit = [&](int stage, int unit) {
        const std::size_t page = kv_row(a.kv, lane, kv, unit * kKeyUnit);
        for (int i = tid; i < 2 * kKeyUnit * (kD / 8); i += kThreads) {
            const int plane = i / (kKeyUnit * (kD / 8)), rest = i % (kKeyUnit * (kD / 8));
            const int row = rest / (kD / 8), chunk = rest % (kD / 8);
            const std::uint16_t* src = (plane == 0 ? a.kv.k : a.kv.v) + page + row * kD + 8 * chunk;
            std::uint16_t* dst = (plane == 0 ? sm.k[stage] : sm.v[stage]) + swz(row, 8 * chunk);
            ops::cp_async<16, Cache::cg>(dst, src);
        }
        ops::cp_commit();
    };
    // Quantized: a unit's codes and side records into the code stage (its rows are consecutive, so each plane's codes
    // and side records are contiguous, whole 16-byte copies), then, from the stage, into the tile's represented words
    // by decode_unit over a view of the stage: the words a decode from global memory gives.
    const auto stage_codes = [&](int unit) {
        if constexpr (Quantized) {
            const std::size_t page = kv_row(a.kv, lane, kv, unit * kKeyUnit);
            for (int plane = 0; plane < 2; ++plane) {
                const detail::KvForm form = detail::plane_form(a.kv, plane);
                const int code_copies = kKeyUnit * detail::code_bytes(form) / 16;
                const int side_copies = kKeyUnit * detail::side_bytes(form) / 16;
                const std::uint8_t* codes = detail::row_codes(a.kv, plane, form, page);
                const std::uint8_t* side  = detail::row_side(a.kv, plane, form, page);
                std::uint8_t* dst         = sm.code + plane * kCodePlane;
                for (int i = tid; i < code_copies + side_copies; i += kThreads) {
                    if (i < code_copies) {
                        ops::cp_async<16, Cache::cg>(dst + 16 * i, codes + 16 * i);
                    } else {
                        ops::cp_async<16, Cache::cg>(dst + kKeyUnit * kCodeRowBytes + 16 * (i - code_copies),
                                                     side + 16 * (i - code_copies));
                    }
                }
            }
            ops::cp_commit();
        }
    };
    const auto decode_staged = [&] {
        if constexpr (Quantized) {
            KvView staged = a.kv;
            staged.k      = reinterpret_cast<std::uint16_t*>(sm.code);
            staged.v      = reinterpret_cast<std::uint16_t*>(sm.code + kCodePlane);
            staged.k_side = sm.code + kKeyUnit * kCodeRowBytes;
            staged.v_side = sm.code + kCodePlane + kKeyUnit * kCodeRowBytes;
            detail::decode_unit(staged, 0, sm.k[0], sm.v[0], [](int row, int d) { return swz(row, d); }, tid, kThreads);
        }
    };

    if (lo <= hi) {
        if constexpr (Quantized) {
            stage_codes(lo);
            ops::cp_wait<0>();
            __syncthreads();
            decode_staged();
        } else {
            load_unit(0, lo);
        }
    }
    for (int unit = lo; unit <= hi; ++unit) {
        int stage = 0;
        if constexpr (Quantized) {
            __syncthreads(); // the tile holds this unit; every reader of the code stage is done
            if (unit + 1 <= hi) { stage_codes(unit + 1); }
        } else {
            stage = (unit - lo) & 1;
            if (unit + 1 <= hi) {
                load_unit(stage ^ 1, unit + 1);
                ops::cp_wait<1>();
            } else {
                ops::cp_wait<0>();
            }
            __syncthreads();
        }
        // The relative table's columns for this unit's distances cdmin .. cdmin + 127 (cdmin = p_first - j0 - 63),
        // transposed; zero past the table. A unit whose nearest distance is past the table (far: every key of it at
        // least kRelativeExtent before the CTA's first row) has each bias the fma fold of zeros, +0: it builds no table
        // and folds no bias, the same bits (CTA-uniform; units ascend, so the far ones come first and no later unit
        // reads a table a far one skipped).
        const int j0 = unit * kKeyUnit, cdmin = p_first - j0 - 63;
        const bool far = cdmin >= kRelativeExtent;
        if (!far) {
            for (int i = tid; i < kTableColumns * kRelDim; i += kThreads) {
                const int dim = i / kTableColumns, dl = i % kTableColumns, d = cdmin + dl;
                sm.table[dl][dim] = d >= 0 && d < kRelativeExtent ? __ldg(a.layer.relative_table + dim * kRelativeExtent + d)
                                                                   : std::uint16_t{0};
            }
            __syncthreads();
        }
        if (warp_live && unit >= w_lo && unit <= w_hi) {
            // The warp's bias for distances wdmin .. wdmin + 79 (wdmin = wpos0 - j0 - 63; row r, key c: index r - c + 63)
            // in the decode route's arithmetic: bf16 of the fma fold of r[i] P[i][d] over i in order. Lane l: row l & 15,
            // indices (l >> 4) + 2k.
            const int off = wc0 - c0;
            for (int idx = far ? kBiasColumns : ln >> 4; idx < kBiasColumns - 1; idx += 2) {
                const uint4* col = reinterpret_cast<const uint4*>(sm.table[off + idx]);
                const uint4 w0 = col[0], w1 = col[1];
                const std::uint32_t words[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
                float acc16 = 0.0F;
#pragma unroll
                for (int i = 0; i < kRelDim; ++i) {
                    const std::uint32_t wv = words[i / 2];
                    const float pv = canon::f32_from_bits((i & 1) != 0 ? (wv & 0xFFFF0000U) : (wv << 16));
                    const float rv = canon::f32_from_bits((i & 1) != 0 ? (rrow[i / 2] & 0xFFFF0000U) : (rrow[i / 2] << 16));
                    acc16 = canon::fma_rn(rv, pv, acc16);
                }
                sm.bias[warp][ln & 15][idx] = canon::f32_to_bf16_rn(acc16);
            }
            __syncwarp();
            // Scores: 8 key tiles of 8.
            const std::uint16_t* ks = sm.k[stage];
            float sc[8][4];
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.0F;
                const int krow = 8 * nt + (ln & 7);
#pragma unroll
                for (int s = 0; s < 8; s += 2) {
                    // ldmatrix x4: k16 steps s and s + 1 of keys 8nt .. 8nt + 7.
                    const int col = 16 * s + 8 * (ln >> 3);
                    unsigned b[4];
                    ldmatrix_x4(b[0], b[1], b[2], b[3], smem_addr(ks + swz(krow, col)));
                    mma_bf16(sc[nt][0], sc[nt][1], sc[nt][2], sc[nt][3], qa[s][0], qa[s][1], qa[s][2], qa[s][3], b[0], b[1]);
                    mma_bf16(sc[nt][0], sc[nt][1], sc[nt][2], sc[nt][3], qa[s + 1][0], qa[s + 1][1], qa[s + 1][2],
                             qa[s + 1][3], b[2], b[3]);
                }
            }
            // t = (S / sqrt 128 + bias) log2 e, masked -inf; the unit's row maxima.
            float mu[2] = {neg_inf(), neg_inf()};
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int hr = e >> 1, key = 8 * nt + 2 * t4 + (e & 1);
                    const int r  = g + 8 * hr;
                    const int kpos = j0 + key, dist = pos_g[hr] - kpos;
                    const bool visible = dist >= 0 && (full || dist < kWindow) && kpos >= first;
                    const float bias = far ? 0.0F : canon::bf16_to_f32(sm.bias[warp][r][r - key + 63]);
                    const float s    = canon::add_rn(canon::mul_rn(sc[nt][e], kInvSqrtD), bias);
                    sc[nt][e]        = visible ? canon::mul_rn(s, kLog2e) : neg_inf();
                    mu[hr]           = sc[nt][e] > mu[hr] ? sc[nt][e] : mu[hr];
                }
            }
            float scale[2];
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
#pragma unroll
                for (int o = 1; o <= 2; o <<= 1) {
                    const float other = __shfl_xor_sync(kAll, mu[hr], o);
                    mu[hr]            = other > mu[hr] ? other : mu[hr];
                }
                // A row with no visible key here: c = 1, p = 0 (its state unchanged).
                const bool live = mu[hr] != neg_inf();
                const float m   = live ? (mu[hr] > m_row[hr] ? mu[hr] : m_row[hr]) : m_row[hr];
                scale[hr]       = live ? (m_row[hr] == neg_inf() ? 0.0F : ex2(canon::add_rn(m_row[hr], -m))) : 1.0F;
                mu[hr]          = live ? m : neg_inf(); // the exponent base of this unit's p
                m_row[hr]       = m;
            }
            float lsum[2] = {0.0F, 0.0F};
            unsigned pa[4][4]; // P as FP16 A fragments of the 4 k16 steps over the unit's keys
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                float p[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int hr = e >> 1;
                    p[e] = mu[hr] == neg_inf() || sc[nt][e] == neg_inf() ? 0.0F : ex2(canon::add_rn(sc[nt][e], -mu[hr]));
                    lsum[hr] = canon::add_rn(lsum[hr], p[e]);
                }
                pa[nt >> 1][(nt & 1) * 2]     = pack_half2(p[0], p[1]);
                pa[nt >> 1][(nt & 1) * 2 + 1] = pack_half2(p[2], p[3]);
            }
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
#pragma unroll
                for (int o = 1; o <= 2; o <<= 1) { lsum[hr] = canon::add_rn(lsum[hr], __shfl_xor_sync(kAll, lsum[hr], o)); }
                l_row[hr] = canon::fma_rn(l_row[hr], scale[hr], lsum[hr]);
            }
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                acc[n][0] = canon::mul_rn(acc[n][0], scale[0]);
                acc[n][1] = canon::mul_rn(acc[n][1], scale[0]);
                acc[n][2] = canon::mul_rn(acc[n][2], scale[1]);
                acc[n][3] = canon::mul_rn(acc[n][3], scale[1]);
            }
            // P x V: 4 k16 steps (keys) x 16 dim tiles; V by ldmatrix.trans.
            const std::uint16_t* vs = sm.v[stage];
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                // a's register order: (row g, keys 2t..) (row g+8, keys 2t..) (row g, keys 8+2t..) (row g+8, ...).
                const unsigned a0 = pa[s][0], a1 = pa[s][1], a2 = pa[s][2], a3 = pa[s][3];
#pragma unroll
                for (int n = 0; n < 16; n += 2) {
                    // ldmatrix x4 trans: keys 16s .. 16s + 15, dims 8n .. 8n + 15.
                    const int key = 16 * s + (ln & 15), col = 8 * n + 8 * (ln >> 4);
                    unsigned b[4];
                    ldmatrix_x4_t(b[0], b[1], b[2], b[3], smem_addr(vs + swz(key, col)));
                    mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], a0, a1, a2, a3, b[0], b[1]);
                    mma_f16(acc[n + 1][0], acc[n + 1][1], acc[n + 1][2], acc[n + 1][3], a0, a1, a2, a3, b[2], b[3]);
                }
            }
        }
        __syncthreads(); // the stage is refilled next
        if constexpr (Quantized) {
            if (unit + 1 <= hi) {
                ops::cp_wait<0>();
                __syncthreads(); // every thread's copies into the code stage have landed
                decode_staged();
            }
        }
    }
    pdl::trigger_dependents();
    if (!warp_live) { return; }
    if constexpr (Quantized) {
        if (detail::rotated_pool(a)) { unrotate_rows(acc, t4); }
    }
    // The sink, the division, the gate.
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
        const int col = wc0 + g + 8 * hr;
        if (col > last_col) { continue; }
        const float sink = canon::bf16_to_f32(a.in.sink[static_cast<std::size_t>(col) * kHeads + head]);
        const float l    = canon::add_rn(l_row[hr], ex2(canon::add_rn(canon::mul_rn(sink, kLog2e), -m_row[hr])));
        const std::uint16_t* gate = a.in.gate + static_cast<std::size_t>(col) * kQWidth + head * kD;
        std::uint16_t* out        = a.out + static_cast<std::size_t>(col) * kQWidth + head * kD;
#pragma unroll
        for (int n = 0; n < 16; ++n) {
            const int d = 8 * n + 2 * t4;
            std::uint16_t y[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const float v = canon::add_rn(acc[n][2 * hr + e], 0.0F); // -0 as +0
                const float q = canon::bf16_to_f32(canon::f32_to_bf16_rn(canon::div_rn(v, l)));
                y[e] = canon::f32_to_bf16_rn(canon::mul_rn(q, canon::sigmoid_c(canon::bf16_to_f32(gate[d + e]))));
            }
            *reinterpret_cast<unsigned*>(out + d) = pack_bf16(y[0], y[1]);
        }
    }
}

} // namespace

void attention_prompt(const AttentionArgs& a, cudaStream_t stream) {
    if (a.columns <= 0 || a.columns > a.ws.capacity || a.host_groups == nullptr || a.groups <= 0) {
        throw std::invalid_argument("inkling2 attention_prompt: 1..capacity columns, its groups on the host");
    }
    // Several lanes: each lane's consecutive columns (no padding) in tiles of its own from its first column.
    PromptTiles tiles;
    bool one_lane = true;
    for (int g = 1; g < a.groups; ++g) { one_lane = one_lane && a.host_groups[3 * g + 2] == a.host_groups[2]; }
    if (!one_lane) {
        std::int32_t covered = 0, tile_lane = -1;
        for (int g = 0; g < a.groups; ++g) {
            const std::int32_t col = a.host_groups[3 * g], count = a.host_groups[3 * g + 1], lane = a.host_groups[3 * g + 2];
            covered += count;
            for (std::int32_t c = col; c < col + count; ++c) {
                // The column extends the open tile if it is the same lane's next column and the tile has room.
                const bool extend = tiles.count > 0 && lane == tile_lane && tiles.end[tiles.count - 1] == c &&
                                    tiles.end[tiles.count - 1] - tiles.begin[tiles.count - 1] < kTileColumns;
                if (extend) {
                    ++tiles.end[tiles.count - 1];
                    continue;
                }
                if (tiles.count == kMaxPromptTiles) {
                    throw std::invalid_argument("inkling2 attention_prompt: a call of several lanes takes at most 160 tiles");
                }
                tiles.begin[tiles.count] = c;
                tiles.end[tiles.count]   = c + 1;
                ++tiles.count;
                tile_lane = lane;
            }
        }
        if (covered != a.columns) {
            throw std::invalid_argument("inkling2 attention_prompt: a call of several lanes has no padding columns");
        }
    }
    detail::check_attention_args(a);
    detail::launch_attention_prologue(a, stream);
    constexpr int plain_bytes = static_cast<int>(sizeof(SharedT<false>));
    constexpr int coded_bytes = static_cast<int>(sizeof(SharedT<true>));
    static const bool attribute = [] {
        CUDA_CHECK(cudaFuncSetAttribute(attention_prompt_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        plain_bytes));
        CUDA_CHECK(cudaFuncSetAttribute(attention_prompt_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        coded_bytes));
        return true;
    }();
    (void)attribute;
    const bool quantized = detail::quantized_pool(a);
    const dim3 grid(static_cast<unsigned>(tiles.count != 0 ? tiles.count : (a.columns + kTileColumns - 1) / kTileColumns),
                    kKvHeads);
    const pdl::LaunchConfig config{grid, dim3(kThreads), static_cast<std::size_t>(quantized ? coded_bytes : plain_bytes),
                                   stream};
    CUDA_CHECK(quantized ? pdl::launch_consumer(config, attention_prompt_kernel<true>, a, tiles)
                         : pdl::launch_consumer(config, attention_prompt_kernel<false>, a, tiles));
}

} // namespace infernix::ops::inkling2
