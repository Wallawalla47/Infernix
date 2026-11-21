// Inkling 2's routed experts, the narrow route of decode and verification (include/infernix/ops/w4a4_sparse_moe.h,
// design docs/maintainer/inkling2_mini-design.md §10.3).
//
// - Dispatch (one CTA): the call's entries grouped by expert (ascending entry within an expert), jobs in ascending
//   expert id, and the per-call state cleared: each job's completion count and each entry's h amax.
// - Gate/up (a CTA of 8 warps per 4 row groups of a job's gate/up matrix: 2 K halves a row group, each streaming its
//   half through a private cp.async ring): the job's columns in passes of 8 (one MMA's n). NVFP4: the exact int64 core
//   (ops/common/a4_mma.cuh) and the row scale (amax_x * amax_w) * rcp_rn(1536 * den); MXFP8: block-scaled FP8 MMAs
//   accumulating in FP32 in block order, the halves added half 0 + half 1. Interleaved rows put gate_i and up_i on
//   lanes 4 apart; h = the softcapped SwiGLU times gamma, BF16. NVFP4: h to global with its row amax (atomicMax of the
//   magnitude's bits, order-free), and the job's last CTA (the one completing its count) quantizes the job's h rows by
//   four-over-six once; MXFP8: a CTA's 32 intermediates are one MXFP8 group, quantized in place.
// - Down (a CTA per kDownWarps output row groups of a job): its weight units copied before it waits for its job's h
//   (the job's count, a release from each gate/up CTA), then the same core over K = I.
// Each entry's arithmetic depends only on its own column, in a fixed order, so a column's outputs never depend on the
// call's other columns.

#include "infernix/ops/w4a4_sparse_moe.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/a4_mma.cuh"
#include "ops/common/canonical_math.h"
#include "ops/common/mma.cuh"
#include "ops/ggml/ggml_tile.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops::w4a4_moe {
namespace {

namespace tile = infernix::ggml::tile;

// The geometry the kernels are compiled for (Inkling 2 Mini); a bank of another shape is refused.
struct Inkling {
    static constexpr int kHidden = 2304;
    static constexpr int kInter  = 1152;
};

constexpr int kWarps      = 4; // gate/up row groups a CTA
constexpr int kSplit      = 2; // K halves of a gate/up row group
constexpr int kGuThreads  = 32 * kWarps * kSplit;
constexpr int kDepth      = 3; // stages of a warp's ring
constexpr int kPass       = 8; // columns of one MMA (n = 8)
constexpr int kCtaInter   = 8 * kWarps; // intermediates of a gate/up CTA (its 64 interleaved rows)
constexpr int kEntries    = kMaxColumns * kTopK;

template <class G, bool A8>
struct Shape {
    static constexpr int kUnit             = A8 ? 528 : 144;
    static constexpr int kBlockK           = A8 ? 32 : 16;
    static constexpr int kGateUpRowGroups  = 2 * G::kInter / 16;
    static constexpr int kGateUpBlocks     = G::kHidden / kBlockK;
    static constexpr int kDownRowGroups    = G::kHidden / 16;
    static constexpr int kDownBlocks       = G::kInter / kBlockK;
    static constexpr std::size_t kGateUpBytes = std::size_t{kGateUpRowGroups} * kGateUpBlocks * kUnit;
    static constexpr int kHalfBlocks       = kGateUpBlocks / kSplit;
    static constexpr int kStageUnits       = A8 ? 2 : 8;
    static constexpr int kStages           = kHalfBlocks / kStageUnits;
    static constexpr int kStageBytes       = kStageUnits * kUnit;
    static constexpr int kGateUpCtas       = kGateUpRowGroups / kWarps;
    static constexpr int kDownWarps        = A8 ? 2 : 4; // a row group's units are 19 KB (A8) or 10 KB (A4)
    static constexpr int kDownThreads      = 32 * kDownWarps;
    static constexpr int kDownCtas         = kDownRowGroups / kDownWarps;
    static constexpr int kDownBytes        = kDownBlocks * kUnit;
    static constexpr int kXRowBytes        = A8 ? G::kHidden + G::kHidden / 32 : G::kHidden / 2 + G::kHidden / 16;
    static constexpr int kXStride          = (kXRowBytes + 15) / 16 * 16; // a shared x row, 16-byte aligned
    static constexpr int kHRowBytes        = A8 ? G::kInter + G::kInter / 32 : G::kInter / 2 + G::kInter / 16;
    static constexpr int kHStride          = (kHRowBytes + 15) / 16 * 16;
    static constexpr int kXScales          = A8 ? G::kHidden : G::kHidden / 2; // byte offset of a row's scale words
    static constexpr int kHScales          = A8 ? G::kInter : G::kInter / 2;
    static_assert(kHalfBlocks % kStageUnits == 0 && kStages >= kDepth, "whole stages in each half, a full ring");
    static_assert(kGateUpRowGroups % kWarps == 0 && kDownRowGroups % kDownWarps == 0, "whole row groups a CTA");
    static_assert(kStageBytes % 16 == 0 && kDownBytes % 16 == 0 && kXRowBytes % 8 == 0 && kHRowBytes % 4 == 0,
                  "16-byte weight copies, 8-byte x and 4-byte h copies");
    static_assert(G::kInter % kCtaInter == 0 && kCtaInter == 32, "a gate/up CTA's intermediates are one MXFP8 group");
};

// ---------------------------------------------------------------------------- shared helpers

__device__ __forceinline__ float bf16f(std::uint16_t v) { return canon::bf16_to_f32(v); }

__device__ __forceinline__ std::uint16_t swiglu(std::uint16_t g, std::uint16_t u, float gamma, float gate_cap,
                                                float up_cap) {
    const float gate = bf16f(g), up = bf16f(u);
    const float a    = canon::mul_rn(canon::softcap_c(gate, gate_cap), canon::sigmoid_c(gate));
    return canon::f32_to_bf16_rn(canon::mul_rn(canon::mul_rn(a, canon::softcap_c(up, up_cap)), gamma));
}

// Spins until *word reaches `target` (thread 0 of a CTA; an acquire for the CTA after the following barrier).
__device__ __forceinline__ void wait_count(const std::uint32_t* word, std::uint32_t target) {
    while (true) {
        std::uint32_t v;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(word) : "memory");
        if (v >= target) { return; }
        __nanosleep(64);
    }
}

__device__ __forceinline__ void release_add(std::uint32_t* word) {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(word) : "memory");
}

__device__ __forceinline__ std::uint32_t acq_rel_add(std::uint32_t* word) {
    std::uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;\n" : "=r"(old) : "l"(word) : "memory");
    return old;
}

struct Args {
    Activations x;
    Routing routing;
    Bank bank;
    Workspace ws;
    std::uint16_t* outputs;
};

struct JobView {
    int expert = 0, first = 0, count = 0;
    const std::uint8_t* record = nullptr;
};

__device__ __forceinline__ bool job_view(const Args& a, int job, JobView& out) {
    if (job >= *a.ws.job_count) { return false; }
    out.expert = a.ws.jobs[job];
    out.first  = a.ws.offsets[out.expert];
    out.count  = a.ws.offsets[out.expert + 1] - out.first;
    out.record = a.bank.records + static_cast<std::size_t>(out.expert) * a.bank.record_stride;
    return true;
}

// ---------------------------------------------------------------------------- dispatch

constexpr int kDispatchThreads = 256;

// The dispatch of n = columns * kTopK entries by one CTA of kDispatchThreads threads: counts, offsets, jobs in
// ascending expert id, entries in ascending order within each expert, and the per-call state cleared. Counts live
// in shared memory and one block scan gives each expert its offset and its job index together (count in the low
// 16 bits, has-entries in the high ones), so no thread walks the experts.
__device__ void dispatch_block(const std::int32_t* routed, int columns, int experts, const Workspace& ws,
                               std::int32_t* cursor) {
    __shared__ std::int32_t counts[kDispatchThreads];
    __shared__ std::int32_t warp_sums[kDispatchThreads / 32];
    const int tid = static_cast<int>(threadIdx.x), lane = tid % 32, warp = tid / 32, n = columns * kTopK;
    counts[tid] = 0;
    if (tid < experts) { ws.job_done[tid] = 0U; }
    for (int i = tid; i < n; i += kDispatchThreads) { ws.h_amax[i] = 0U; }
    __syncthreads();
    for (int i = tid; i < n; i += kDispatchThreads) { atomicAdd(&counts[routed[i]], 1); }
    __syncthreads();
    const int count = counts[tid];
    const int packed = count | ((count != 0 ? 1 : 0) << 16);
    int inclusive    = packed;
#pragma unroll
    for (int o = 1; o < 32; o *= 2) {
        const int v = __shfl_up_sync(0xFFFFFFFFU, inclusive, o);
        if (lane >= o) { inclusive += v; }
    }
    if (lane == 31) { warp_sums[warp] = inclusive; }
    __syncthreads();
    int before = 0;
    for (int w = 0; w < warp; ++w) { before += warp_sums[w]; }
    const int exclusive = before + inclusive - packed;
    const int offset = exclusive & 0xFFFF, job = exclusive >> 16;
    if (tid < experts) {
        ws.counts[tid]  = count;
        ws.offsets[tid] = offset;
        cursor[tid]     = offset;
        if (count != 0) { ws.jobs[job] = tid; }
    }
    if (tid == kDispatchThreads - 1) {
        const int total = before + inclusive; // every expert slot past `experts` has count 0
        ws.offsets[experts] = total & 0xFFFF;
        *ws.job_count       = total >> 16;
    }
    __syncthreads();
    // Entries in ascending order within each expert: one warp, 32 entries a step, ranks by matching lanes.
    if (tid < 32) {
        const unsigned lane_mask_lt = (1U << tid) - 1U;
        for (int base = 0; base < n; base += 32) {
            const int i      = base + tid;
            const int e      = i < n ? routed[i] : -1 - tid; // distinct sentinels past the end
            const unsigned m = __match_any_sync(0xFFFFFFFFU, e);
            if (i < n) { ws.entries[cursor[e] + __popc(m & lane_mask_lt)] = i; }
            __syncwarp();
            if (i < n && (m & lane_mask_lt) == 0U) { cursor[e] += __popc(m); }
            __syncwarp();
        }
    }
    __syncthreads();
}

__global__ void __launch_bounds__(kDispatchThreads) dispatch_kernel(Routing routing, int columns, int experts,
                                                                     Workspace ws) {
    __shared__ std::int32_t cursor[256];
    dispatch_block(routing.experts, columns, experts, ws, cursor);
}

// ---------------------------------------------------------------------------- router

constexpr int kRouteWarps   = 8;  // expert rows a router CTA (one a warp)
constexpr int kRouteThreads = 32 * kRouteWarps;
constexpr int kRouteColumns = 16; // columns a router CTA
constexpr int kRouteChunks  = Inkling::kHidden / 256; // 16-byte chunks of a row a lane holds

__global__ void __launch_bounds__(kRouteThreads) route_kernel(const std::uint16_t* __restrict__ x, int columns,
                                                              const std::uint16_t* __restrict__ router,
                                                              const float* __restrict__ bias, int experts,
                                                              float route_scale, Workspace ws) {
    __shared__ std::int32_t cursor[256];
    __shared__ float keys[kRouteWarps][256];
    __shared__ float sel[kRouteWarps][kTopK];
    __shared__ int last;
    const int tid = static_cast<int>(threadIdx.x), lane = tid % 32, warp = tid / 32;
    const int row = static_cast<int>(blockIdx.x) * kRouteWarps + warp;
    // The row's weights before the dependency wait (immutable): lane l holds elements 8 (l + 32 j) .. + 7.
    uint4 w[kRouteChunks];
    if (row < experts) {
#pragma unroll
        for (int j = 0; j < kRouteChunks; ++j) {
            w[j] = *reinterpret_cast<const uint4*>(router + static_cast<std::size_t>(row) * Inkling::kHidden + 8 * (lane + 32 * j));
        }
    }
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    const int t0 = static_cast<int>(blockIdx.y) * kRouteColumns, t1 = min(columns, t0 + kRouteColumns);
    if (row < experts) {
        for (int t = t0; t < t1; ++t) {
            const std::uint16_t* xt = x + static_cast<std::size_t>(t) * Inkling::kHidden;
            float acc = 0.0F;
#pragma unroll
            for (int j = 0; j < kRouteChunks; ++j) {
                const uint4 xv = *reinterpret_cast<const uint4*>(xt + 8 * (lane + 32 * j));
                const std::uint32_t wa[4] = {w[j].x, w[j].y, w[j].z, w[j].w}, xa[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    acc = __fmaf_rn(canon::f32_from_bits(wa[q] << 16), canon::f32_from_bits(xa[q] << 16), acc);
                    acc = __fmaf_rn(canon::f32_from_bits(wa[q] & 0xFFFF0000U), canon::f32_from_bits(xa[q] & 0xFFFF0000U), acc);
                }
            }
#pragma unroll
            for (int o = 16; o >= 1; o /= 2) { acc = __fadd_rn(acc, __shfl_xor_sync(0xFFFFFFFFU, acc, o)); }
            if (lane == 0) { ws.logits[static_cast<std::size_t>(t) * experts + row] = acc; }
        }
    }
    // The last CTA selects every column and dispatches the call.
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        last = acq_rel_add(ws.route_done) == gridDim.x * gridDim.y - 1U ? 1 : 0;
    }
    __syncthreads();
    if (last == 0) { return; }
    __threadfence();
    for (int t = warp; t < columns; t += kRouteWarps) {
        float* k = keys[warp];
        for (int e = lane; e < experts; e += 32) {
            const float s = canon::sigmoid_c(__ldcg(ws.logits + static_cast<std::size_t>(t) * experts + e));
            k[e]          = canon::add_rn(s, bias[e]);
        }
        __syncwarp();
        for (int e = lane; e < experts; e += 32) {
            const float key = k[e];
            int rank        = 0;
            for (int f = 0; f < experts; ++f) {
                const float other = k[f];
                rank += (other > key || (other == key && f < e)) ? 1 : 0;
            }
            if (rank < kTopK) {
                ws.routed[t * kTopK + rank] = e;
                sel[warp][rank] = canon::sigmoid_c(__ldcg(ws.logits + static_cast<std::size_t>(t) * experts + e));
            }
        }
        __syncwarp();
        if (lane < kTopK) {
            const float* s  = sel[warp];
            const float sum = canon::add_rn(canon::add_rn(canon::add_rn(s[0], s[2]), canon::add_rn(s[1], s[3])),
                                            canon::add_rn(canon::add_rn(s[4], s[6]), canon::add_rn(s[5], s[7])));
            ws.gamma[t * kTopK + lane] = canon::mul_rn(canon::div_rn(s[lane], sum), route_scale);
        }
        __syncwarp();
    }
    __syncthreads();
    dispatch_block(ws.routed, columns, experts, ws, cursor);
    if (tid == 0) { *ws.route_done = 0U; }
}

// ---------------------------------------------------------------------------- NVFP4 gate/up

template <class G>
struct GateUpA4Smem {
    using S = Shape<G, false>;
    std::int32_t scaled[256];
    alignas(16) std::uint8_t ring[kSplit * kWarps][kDepth][S::kStageBytes];
    alignas(16) std::uint8_t x[kPass][S::kXStride];
    float rs[kPass];
    std::uint16_t h[kPass][kCtaInter];
    int last;
};

template <class G>
__global__ void __launch_bounds__(kGuThreads) gate_up_a4_kernel(Args a) {
    using S = Shape<G, false>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm = *reinterpret_cast<GateUpA4Smem<G>*>(smem_raw);
    pdl::trigger_dependents(); // the down pass may launch and copy its weights; it reads h after its job's count
    const int job = static_cast<int>(blockIdx.y);
    JobView v;
    if (!job_view(a, job, v)) { return; }
    const int tid = static_cast<int>(threadIdx.x), lane = tid % 32, warp = tid / 32;
    const int g = lane / 4, t = lane % 4;
    const int rg = warp % kWarps, k_half = warp / kWarps;
    const int group = static_cast<int>(blockIdx.x) * kWarps + rg;
    const std::uint8_t* units =
        v.record + (static_cast<std::size_t>(group) * S::kGateUpBlocks + static_cast<std::size_t>(k_half) * S::kHalfBlocks) *
                       S::kUnit;
    auto& ring        = sm.ring[warp];
    const auto issue  = [&](int stage) {
        const std::uint8_t* src = units + static_cast<std::size_t>(stage) * S::kStageBytes;
        for (int i = lane; i < S::kStageBytes / 16; i += 32) { tile::cp_async16(ring[stage % kDepth] + 16 * i, src + 16 * i, 16); }
    };
    a4_mma_fill_scaled(sm.scaled, kGuThreads);
    const float w_amax = a.bank.amax[2 * v.expert];
    const float rcp    = canon::div_rn(1.0F, canon::mul_rn(1536.0F, a.bank.denominators[0]));
    const int h_first  = static_cast<int>(blockIdx.x) * kCtaInter;
    for (int base = 0; base < v.count; base += kPass) {
        const int n = min(kPass, v.count - base);
#pragma unroll
        for (int k = 0; k < kDepth - 1; ++k) {
            issue(k);
            tile::cp_async_commit();
        }
        // The pass's x rows and row scales while the rings fill.
        constexpr int kWords = S::kXRowBytes / 8;
        for (int i = tid; i < n * kWords; i += kGuThreads) {
            const int c      = i / kWords;
            const int column = a.ws.entries[v.first + base + c] / kTopK;
            reinterpret_cast<uint2*>(sm.x[c])[i % kWords] =
                reinterpret_cast<const uint2*>(a.x.rows + static_cast<std::size_t>(column) * S::kXRowBytes)[i % kWords];
        }
        if (tid < n) {
            const int column = a.ws.entries[v.first + base + tid] / kTopK;
            sm.rs[tid]       = canon::mul_rn(canon::mul_rn(a.x.amax[column], w_amax), rcp);
        }
        __syncthreads();
        const int b0            = k_half * S::kHalfBlocks;
        const std::uint8_t* xg  = sm.x[min(g, n - 1)] + 8 * b0 + 2 * t;
        const std::uint8_t* xs0 = sm.x[min(2 * t, n - 1)] + S::kXScales + b0;
        const std::uint8_t* xs1 = sm.x[min(2 * t + 1, n - 1)] + S::kXScales + b0;
        long long s[4]          = {0, 0, 0, 0};
        for (int st = 0; st < S::kStages; ++st) {
            tile::cp_async_wait<kDepth - 2>();
            __syncwarp();
            if (st + kDepth - 1 < S::kStages) { issue(st + kDepth - 1); }
            tile::cp_async_commit();
            const std::uint8_t* stage = ring[st % kDepth];
#pragma unroll
            for (int u = 0; u < S::kStageUnits; ++u) {
                const int b = st * S::kStageUnits + u;
                a4_mma_unit(stage + u * S::kUnit, a4_x_quad(*reinterpret_cast<const unsigned short*>(xg + 8 * b)), sm.scaled,
                            sm.scaled[xs0[b]], sm.scaled[xs1[b]], g, t, s);
            }
            __syncwarp();
        }
        tile::cp_async_wait<0>();
        __syncthreads();
        // The second half's sums join the first's (exact int64 adds) through the rings' memory.
        auto* half = reinterpret_cast<long long (*)[32][4]>(&sm.ring[0][0][0]);
        if (k_half == 1) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { half[rg][lane][i] = s[i]; }
        }
        __syncthreads();
        if (k_half == 0) {
            std::uint16_t y[4], up[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[i] += half[rg][lane][i];
                const float rs = sm.rs[min(2 * t + (i & 1), n - 1)];
                y[i] = canon::f32_to_bf16_rn(canon::mul_rn(canon::mul_rn(canon::i64_to_f32_rn(s[i]), 1.0F / 1048576.0F), rs));
            }
            // Rows g and g + 8 are gate rows for even g (row 2i is intermediate i's gate, 2i + 1 its up): the up
            // row's outputs come from lane + 4.
#pragma unroll
            for (int i = 0; i < 4; ++i) { up[i] = static_cast<std::uint16_t>(__shfl_down_sync(0xFFFFFFFFU, y[i], 4)); }
            if ((g & 1) == 0) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int c = 2 * t + (i & 1);
                    if (c < n) {
                        const int entry = a.ws.entries[v.first + base + c];
                        sm.h[c][rg * 8 + g / 2 + (i < 2 ? 0 : 4)] =
                            swiglu(y[i], up[i], a.routing.gamma[entry], a.bank.gate_softcap, a.bank.up_softcap);
                    }
                }
            }
        }
        __syncthreads();
        // h to global (a column's 32 intermediates, coalesced) and each column's amax over them.
        for (int i = tid; i < n * kCtaInter; i += kGuThreads) {
            const int c = i / kCtaInter, k = i % kCtaInter;
            const int entry = a.ws.entries[v.first + base + c];
            a.ws.h[static_cast<std::size_t>(entry) * G::kInter + h_first + k] = sm.h[c][k];
        }
        if (warp < n) {
            const std::uint32_t mag = canon::f32_bits(bf16f(sm.h[warp][lane])) & 0x7FFFFFFFU;
            std::uint32_t m         = mag;
#pragma unroll
            for (int o = 16; o >= 1; o /= 2) { m = max(m, __shfl_xor_sync(0xFFFFFFFFU, m, o)); }
            if (lane == 0) { atomicMax(&a.ws.h_amax[a.ws.entries[v.first + base + warp]], m); }
        }
        __syncthreads(); // the next pass reuses x, rs and h
    }
    // This CTA's h is stored: count it for the job. The job's last CTA quantizes the job's h rows once, then counts
    // again: the down CTAs wait for kGateUpCtas + 1.
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        sm.last = acq_rel_add(&a.ws.job_done[job]) == static_cast<std::uint32_t>(S::kGateUpCtas - 1) ? 1 : 0;
    }
    __syncthreads();
    if (sm.last == 0) { return; }
    __threadfence();
    constexpr int kBlocks = G::kInter / 16;
    for (int i = tid; i < v.count * kBlocks; i += kGuThreads) {
        const int c = i / kBlocks, b = i % kBlocks;
        const int entry = a.ws.entries[v.first + c];
        const float amax = canon::f32_from_bits(__ldcg(a.ws.h_amax + entry));
        const canon::FourOverSixRow row = canon::four_over_six_row(amax);
        const auto* src = reinterpret_cast<const uint4*>(a.ws.h + static_cast<std::size_t>(entry) * G::kInter + 16 * b);
        const uint4 lo = __ldcg(src), hi = __ldcg(src + 1);
        const std::uint32_t w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        float xv[16];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            xv[2 * j]     = canon::f32_from_bits(w[j] << 16);
            xv[2 * j + 1] = canon::f32_from_bits(w[j] & 0xFFFF0000U);
        }
        const canon::FourOverSixBlock block = canon::four_over_six_block(xv, row);
        std::uint32_t packed[2] = {0, 0};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const std::uint32_t byte = block.code[2 * j] | (static_cast<std::uint32_t>(block.code[2 * j + 1]) << 4);
            packed[j / 4] |= byte << (8 * (j % 4));
        }
        std::uint8_t* out = a.ws.h_rows + static_cast<std::size_t>(entry) * S::kHRowBytes;
        *reinterpret_cast<uint2*>(out + 8 * b) = make_uint2(packed[0], packed[1]);
        out[S::kHScales + b]                   = block.scale_word;
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        release_add(&a.ws.job_done[job]);
    }
}

// ---------------------------------------------------------------------------- NVFP4 down

template <class G>
struct DownA4Smem {
    using S = Shape<G, false>;
    std::int32_t scaled[256];
    alignas(16) std::uint8_t units[S::kDownWarps][S::kDownBytes];
    alignas(16) std::uint8_t h[kPass][S::kHStride];
    float rs[kPass];
};

template <class G>
__global__ void __launch_bounds__(Shape<G, false>::kDownThreads) down_a4_kernel(Args a) {
    using S = Shape<G, false>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm = *reinterpret_cast<DownA4Smem<G>*>(smem_raw);
    pdl::trigger_dependents();
    const int job = static_cast<int>(blockIdx.y);
    JobView v;
    if (!job_view(a, job, v)) { return; }
    const int tid = static_cast<int>(threadIdx.x), lane = tid % 32, warp = tid / 32;
    const int g = lane / 4, t = lane % 4;
    const int group = static_cast<int>(blockIdx.x) * S::kDownWarps + warp;
    // The weights do not depend on h: their copies start before the wait.
    const std::uint8_t* src = v.record + S::kGateUpBytes + static_cast<std::size_t>(group) * S::kDownBytes;
    for (int i = lane; i < S::kDownBytes / 16; i += 32) { tile::cp_async16(sm.units[warp] + 16 * i, src + 16 * i, 16); }
    tile::cp_async_commit();
    a4_mma_fill_scaled(sm.scaled, S::kDownThreads);
    if (tid == 0) { wait_count(a.ws.job_done + job, static_cast<std::uint32_t>(S::kGateUpCtas + 1)); }
    __syncthreads();
    const float w_amax = a.bank.amax[2 * v.expert + 1];
    const float rcp    = canon::div_rn(1.0F, canon::mul_rn(1536.0F, a.bank.denominators[1]));
    for (int base = 0; base < v.count; base += kPass) {
        const int n = min(kPass, v.count - base);
        constexpr int kWords = S::kHRowBytes / 4;
        for (int i = tid; i < n * kWords; i += S::kDownThreads) {
            const int c = i / kWords, entry = a.ws.entries[v.first + base + c];
            reinterpret_cast<std::uint32_t*>(sm.h[c])[i % kWords] =
                __ldcg(reinterpret_cast<const std::uint32_t*>(a.ws.h_rows + static_cast<std::size_t>(entry) * S::kHRowBytes) +
                       i % kWords);
        }
        if (tid < n) {
            const int entry = a.ws.entries[v.first + base + tid];
            sm.rs[tid] = canon::mul_rn(canon::mul_rn(canon::f32_from_bits(__ldcg(a.ws.h_amax + entry)), w_amax), rcp);
        }
        tile::cp_async_wait<0>();
        __syncthreads();
        const std::uint8_t* hg  = sm.h[min(g, n - 1)] + 2 * t;
        const std::uint8_t* hs0 = sm.h[min(2 * t, n - 1)] + S::kHScales;
        const std::uint8_t* hs1 = sm.h[min(2 * t + 1, n - 1)] + S::kHScales;
        const std::uint8_t* units = sm.units[warp];
        long long s[4] = {0, 0, 0, 0};
#pragma unroll 8
        for (int b = 0; b < S::kDownBlocks; ++b) {
            a4_mma_unit(units + b * S::kUnit, a4_x_quad(*reinterpret_cast<const unsigned short*>(hg + 8 * b)), sm.scaled,
                        sm.scaled[hs0[b]], sm.scaled[hs1[b]], g, t, s);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = 2 * t + (i & 1);
            if (c < n) {
                const int entry = a.ws.entries[v.first + base + c];
                const int row   = group * 16 + g + (i < 2 ? 0 : 8);
                a.outputs[static_cast<std::size_t>(entry) * G::kHidden + row] = canon::f32_to_bf16_rn(
                    canon::mul_rn(canon::mul_rn(canon::i64_to_f32_rn(s[i]), 1.0F / 1048576.0F), sm.rs[c]));
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------- MXFP8 gate/up

// One mxfp8_expert_rg16_v1 unit times column g's codes (B: K 4t..4t+3 and 16+4t..+3 of the block) with the rows' and
// the column's UE8M0 scales, accumulated into c in FP32.
__device__ __forceinline__ void a8_mma_unit(const std::uint8_t* unit, const std::uint8_t* column_codes,
                                            std::uint8_t column_scale, int lane, int g, int t, float (&c)[4]) {
    const uint4 a       = *reinterpret_cast<const uint4*>(unit + 16 * lane);
    const unsigned b0   = *reinterpret_cast<const unsigned*>(column_codes + 4 * t);
    const unsigned b1   = *reinterpret_cast<const unsigned*>(column_codes + 16 + 4 * t);
    const unsigned sfa  = unit[512 + ((t & 1) != 0 ? g + 8 : g)];
    mma_fp8_e4m3_block_scaled<0, 0>(c[0], c[1], c[2], c[3], a.x, a.y, a.z, a.w, b0, b1, sfa, column_scale);
}

template <class G>
struct GateUpA8Smem {
    using S = Shape<G, true>;
    alignas(16) std::uint8_t ring[kSplit * kWarps][kDepth][S::kStageBytes];
    alignas(16) std::uint8_t x[kPass][S::kXStride];
    std::uint16_t h[kPass][kCtaInter];
};

template <class G>
__global__ void __launch_bounds__(kGuThreads) gate_up_a8_kernel(Args a) {
    using S = Shape<G, true>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm = *reinterpret_cast<GateUpA8Smem<G>*>(smem_raw);
    pdl::trigger_dependents();
    const int job = static_cast<int>(blockIdx.y);
    JobView v;
    if (!job_view(a, job, v)) { return; }
    const int tid = static_cast<int>(threadIdx.x), lane = tid % 32, warp = tid / 32;
    const int g = lane / 4, t = lane % 4;
    const int rg = warp % kWarps, k_half = warp / kWarps;
    const int group = static_cast<int>(blockIdx.x) * kWarps + rg;
    const std::uint8_t* units =
        v.record + (static_cast<std::size_t>(group) * S::kGateUpBlocks + static_cast<std::size_t>(k_half) * S::kHalfBlocks) *
                       S::kUnit;
    auto& ring       = sm.ring[warp];
    const auto issue = [&](int stage) {
        const std::uint8_t* src = units + static_cast<std::size_t>(stage) * S::kStageBytes;
        for (int i = lane; i < S::kStageBytes / 16; i += 32) { tile::cp_async16(ring[stage % kDepth] + 16 * i, src + 16 * i, 16); }
    };
    const int h_first = static_cast<int>(blockIdx.x) * kCtaInter;
    for (int base = 0; base < v.count; base += kPass) {
        const int n = min(kPass, v.count - base);
#pragma unroll
        for (int k = 0; k < kDepth - 1; ++k) {
            issue(k);
            tile::cp_async_commit();
        }
        constexpr int kWords = S::kXRowBytes / 8;
        for (int i = tid; i < n * kWords; i += kGuThreads) {
            const int c      = i / kWords;
            const int column = a.ws.entries[v.first + base + c] / kTopK;
            reinterpret_cast<uint2*>(sm.x[c])[i % kWords] =
                reinterpret_cast<const uint2*>(a.x.rows + static_cast<std::size_t>(column) * S::kXRowBytes)[i % kWords];
        }
        __syncthreads();
        const int b0           = k_half * S::kHalfBlocks;
        const std::uint8_t* xg = sm.x[min(g, n - 1)];
        float c[4]             = {0.0F, 0.0F, 0.0F, 0.0F};
        for (int st = 0; st < S::kStages; ++st) {
            tile::cp_async_wait<kDepth - 2>();
            __syncwarp();
            if (st + kDepth - 1 < S::kStages) { issue(st + kDepth - 1); }
            tile::cp_async_commit();
            const std::uint8_t* stage = ring[st % kDepth];
#pragma unroll
            for (int u = 0; u < S::kStageUnits; ++u) {
                const int b = b0 + st * S::kStageUnits + u;
                a8_mma_unit(stage + u * S::kUnit, xg + 32 * b, xg[S::kXScales + b], lane, g, t, c);
            }
            __syncwarp();
        }
        tile::cp_async_wait<0>();
        __syncthreads();
        auto* half = reinterpret_cast<float (*)[32][4]>(&sm.ring[0][0][0]);
        if (k_half == 1) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { half[rg][lane][i] = c[i]; }
        }
        __syncthreads();
        if (k_half == 0) {
            std::uint16_t y[4], up[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) { y[i] = canon::f32_to_bf16_rn(canon::add_rn(c[i], half[rg][lane][i])); }
#pragma unroll
            for (int i = 0; i < 4; ++i) { up[i] = static_cast<std::uint16_t>(__shfl_down_sync(0xFFFFFFFFU, y[i], 4)); }
            if ((g & 1) == 0) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int col = 2 * t + (i & 1);
                    if (col < n) {
                        const int entry = a.ws.entries[v.first + base + col];
                        sm.h[col][rg * 8 + g / 2 + (i < 2 ? 0 : 4)] =
                            swiglu(y[i], up[i], a.routing.gamma[entry], a.bank.gate_softcap, a.bank.up_softcap);
                    }
                }
            }
        }
        __syncthreads();
        // A column's 32 intermediates are one MXFP8 group: quantized here, codes and the UE8M0 word to the h row.
        if (tid < n) {
            float values[32];
#pragma unroll
            for (int j = 0; j < 32; ++j) { values[j] = bf16f(sm.h[tid][j]); }
            const canon::Mxfp8Group q = canon::mxfp8_group(values);
            const int entry           = a.ws.entries[v.first + base + tid];
            std::uint8_t* out         = a.ws.h_rows + static_cast<std::size_t>(entry) * S::kHRowBytes;
            auto* words               = reinterpret_cast<std::uint32_t*>(out + h_first);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                words[j] = q.code[4 * j] | (static_cast<std::uint32_t>(q.code[4 * j + 1]) << 8) |
                           (static_cast<std::uint32_t>(q.code[4 * j + 2]) << 16) |
                           (static_cast<std::uint32_t>(q.code[4 * j + 3]) << 24);
            }
            out[S::kHScales + static_cast<int>(blockIdx.x)] = q.exponent;
        }
        __syncthreads();
    }
    if (tid == 0) {
        __threadfence();
        release_add(&a.ws.job_done[job]);
    }
}

// ---------------------------------------------------------------------------- MXFP8 down

template <class G>
struct DownA8Smem {
    using S = Shape<G, true>;
    alignas(16) std::uint8_t units[S::kDownWarps][S::kDownBytes];
    alignas(16) std::uint8_t h[kPass][S::kHStride];
};

template <class G>
__global__ void __launch_bounds__(Shape<G, true>::kDownThreads) down_a8_kernel(Args a) {
    using S = Shape<G, true>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm = *reinterpret_cast<DownA8Smem<G>*>(smem_raw);
    pdl::trigger_dependents();
    const int job = static_cast<int>(blockIdx.y);
    JobView v;
    if (!job_view(a, job, v)) { return; }
    const int tid = static_cast<int>(threadIdx.x), lane = tid % 32, warp = tid / 32;
    const int g = lane / 4, t = lane % 4;
    const int group = static_cast<int>(blockIdx.x) * S::kDownWarps + warp;
    const std::uint8_t* src = v.record + S::kGateUpBytes + static_cast<std::size_t>(group) * S::kDownBytes;
    for (int i = lane; i < S::kDownBytes / 16; i += 32) { tile::cp_async16(sm.units[warp] + 16 * i, src + 16 * i, 16); }
    tile::cp_async_commit();
    if (tid == 0) { wait_count(a.ws.job_done + job, static_cast<std::uint32_t>(S::kGateUpCtas)); }
    __syncthreads();
    for (int base = 0; base < v.count; base += kPass) {
        const int n = min(kPass, v.count - base);
        constexpr int kWords = S::kHRowBytes / 4;
        for (int i = tid; i < n * kWords; i += S::kDownThreads) {
            const int c = i / kWords, entry = a.ws.entries[v.first + base + c];
            reinterpret_cast<std::uint32_t*>(sm.h[c])[i % kWords] =
                __ldcg(reinterpret_cast<const std::uint32_t*>(a.ws.h_rows + static_cast<std::size_t>(entry) * S::kHRowBytes) +
                       i % kWords);
        }
        tile::cp_async_wait<0>();
        __syncthreads();
        const std::uint8_t* hg = sm.h[min(g, n - 1)];
        float c[4]             = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll 4
        for (int b = 0; b < S::kDownBlocks; ++b) {
            a8_mma_unit(sm.units[warp] + b * S::kUnit, hg + 32 * b, hg[S::kHScales + b], lane, g, t, c);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = 2 * t + (i & 1);
            if (col < n) {
                const int entry = a.ws.entries[v.first + base + col];
                const int row   = group * 16 + g + (i < 2 ? 0 : 8);
                a.outputs[static_cast<std::size_t>(entry) * G::kHidden + row] = canon::f32_to_bf16_rn(c[i]);
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------- launches

std::size_t align256(std::size_t v) { return (v + 255) / 256 * 256; }

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("w4a4_sparse_moe ") + what + ": " + cudaGetErrorString(error)); }
}

template <class Kernel>
void allow_shared(Kernel kernel, std::size_t bytes) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
}

// The workspace's planes in order (workspace_bytes and carve_workspace walk the same list).
template <class Take>
Workspace layout(std::int32_t experts, std::int32_t intermediate, Take&& take) {
    const std::size_t e = static_cast<std::size_t>(experts), m = kEntries, i = static_cast<std::size_t>(intermediate);
    const std::size_t h_row = std::max<std::size_t>(i / 2 + i / 16, i + i / 32);
    Workspace w;
    w.logits     = reinterpret_cast<float*>(take(std::size_t{kMaxColumns} * e * 4));
    w.routed     = reinterpret_cast<std::int32_t*>(take(m * 4));
    w.gamma      = reinterpret_cast<float*>(take(m * 4));
    w.route_done = reinterpret_cast<std::uint32_t*>(take(4));
    w.counts     = reinterpret_cast<std::int32_t*>(take(e * 4));
    w.offsets    = reinterpret_cast<std::int32_t*>(take((e + 1) * 4));
    w.entries    = reinterpret_cast<std::int32_t*>(take(m * 4));
    w.jobs       = reinterpret_cast<std::int32_t*>(take(e * 4));
    w.job_count  = reinterpret_cast<std::int32_t*>(take(4));
    w.job_done   = reinterpret_cast<std::uint32_t*>(take(e * 4));
    w.h          = reinterpret_cast<std::uint16_t*>(take(m * i * 2));
    w.h_amax     = reinterpret_cast<std::uint32_t*>(take(m * 4));
    w.h_rows     = reinterpret_cast<std::uint8_t*>(take(m * h_row));
    return w;
}

} // namespace

std::size_t workspace_bytes(std::int32_t experts, std::int32_t intermediate) {
    std::size_t total = 0;
    (void)layout(experts, intermediate, [&](std::size_t bytes) {
        total += align256(bytes);
        return static_cast<std::byte*>(nullptr);
    });
    return total;
}

Workspace carve_workspace(void* base, std::int32_t experts, std::int32_t intermediate) {
    auto* p = static_cast<std::byte*>(base);
    return layout(experts, intermediate, [&](std::size_t bytes) {
        std::byte* out = p;
        p += align256(bytes);
        return out;
    });
}

void moe_route(const std::uint16_t* x, std::int32_t columns, const std::uint16_t* router, const float* bias,
               std::int32_t experts, std::int32_t hidden, float route_scale, const Workspace& workspace,
               cudaStream_t stream) {
    if (hidden != Inkling::kHidden || experts <= 0 || experts > 256 || columns <= 0 || columns > kMaxColumns) {
        throw std::invalid_argument("w4a4_sparse_moe router: H = 2304, at most 256 experts, 1..kMaxColumns columns");
    }
    const dim3 grid((experts + kRouteWarps - 1) / kRouteWarps, (columns + kRouteColumns - 1) / kRouteColumns);
    CUDA_CHECK(pdl::launch_with(pdl::Dependency::Programmatic, {grid, dim3(kRouteThreads), 0, stream}, route_kernel, x,
                                columns, router, bias, experts, route_scale, workspace));
    check_launch("router");
}

void moe_dispatch(const Routing& routing, std::int32_t columns, std::int32_t experts, const Workspace& workspace,
                  cudaStream_t stream) {
    if (columns <= 0 || columns > kMaxColumns || experts <= 0 || experts > 256) {
        throw std::invalid_argument("w4a4_sparse_moe dispatch: 1..kMaxColumns columns and at most 256 experts");
    }
    CUDA_CHECK(pdl::launch_with(pdl::Dependency::Serialized, {dim3(1), dim3(kDispatchThreads), 0, stream},
                                dispatch_kernel, routing, columns, experts, workspace));
    check_launch("dispatch");
}

void moe_experts(const Activations& x, const Routing& routing, const Bank& bank, const Workspace& workspace,
                 std::uint16_t* outputs, cudaStream_t stream) {
    using G = Inkling;
    if (bank.hidden != G::kHidden || bank.intermediate != G::kInter || bank.experts <= 0 || bank.experts > 256) {
        throw std::invalid_argument("w4a4_sparse_moe: the kernels are built for Inkling 2's [E, 2304, 1152] banks");
    }
    if (x.columns <= 0 || x.columns > kMaxColumns) { throw std::invalid_argument("w4a4_sparse_moe: 1..kMaxColumns columns"); }
    const Args args{x, routing, bank, workspace, outputs};
    // A job per expert with entries: at most min(E, columns * top_k) jobs; CTAs past the job count exit.
    const int jobs = std::min(bank.experts, x.columns * kTopK);
    if (bank.format == ExpertFormat::Nvfp4FourOverSix) {
        using S = Shape<G, false>;
        static const bool configured = [] {
            allow_shared(gate_up_a4_kernel<G>, sizeof(GateUpA4Smem<G>));
            allow_shared(down_a4_kernel<G>, sizeof(DownA4Smem<G>));
            return true;
        }();
        (void)configured;
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Serialized,
                                    {dim3(S::kGateUpCtas, jobs), dim3(kGuThreads), sizeof(GateUpA4Smem<G>), stream},
                                    gate_up_a4_kernel<G>, args));
        check_launch("NVFP4 gate/up");
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Programmatic,
                                    {dim3(S::kDownCtas, jobs), dim3(S::kDownThreads), sizeof(DownA4Smem<G>), stream},
                                    down_a4_kernel<G>, args));
        check_launch("NVFP4 down");
        return;
    }
    using S = Shape<G, true>;
    static const bool configured = [] {
        allow_shared(gate_up_a8_kernel<G>, sizeof(GateUpA8Smem<G>));
        allow_shared(down_a8_kernel<G>, sizeof(DownA8Smem<G>));
        return true;
    }();
    (void)configured;
    CUDA_CHECK(pdl::launch_with(pdl::Dependency::Serialized,
                                {dim3(S::kGateUpCtas, jobs), dim3(kGuThreads), sizeof(GateUpA8Smem<G>), stream},
                                gate_up_a8_kernel<G>, args));
    check_launch("MXFP8 gate/up");
    CUDA_CHECK(pdl::launch_with(pdl::Dependency::Programmatic,
                                {dim3(S::kDownCtas, jobs), dim3(S::kDownThreads), sizeof(DownA8Smem<G>), stream},
                                down_a8_kernel<G>, args));
    check_launch("MXFP8 down");
}

} // namespace infernix::ops::w4a4_moe
