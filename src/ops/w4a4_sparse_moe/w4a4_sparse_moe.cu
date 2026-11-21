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
#include "ops/w4a4_sparse_moe/expert_math.cuh"

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

using detail::swiglu;

__device__ __forceinline__ std::uint64_t global_ns() {
    std::uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

// Spins until *word reaches `target` (thread 0 of a CTA; an acquire for the CTA after the following barrier). A count
// that makes no progress for kWaitTimeoutNs sets kErrorDownStalled in *error and gives up (the CTA then writes nothing):
// a lost count fails the call through its error word, never hangs the GPU.
constexpr std::uint64_t kWaitTimeoutNs = 2'000'000'000ULL;

__device__ __forceinline__ bool wait_count(const std::uint32_t* word, std::uint32_t target, std::uint32_t* error) {
    std::uint32_t seen = 0;
    std::uint64_t since = 0;
    for (bool first = true;; first = false) {
        std::uint32_t v;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(word) : "memory");
        if (v >= target) { return true; }
        const std::uint64_t now = global_ns();
        if (first || v != seen) {
            seen  = v;
            since = now;
        } else if (now - since > kWaitTimeoutNs) {
            atomicOr(error, kErrorDownStalled);
            return false;
        }
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
        detail::select_column(ws.logits + static_cast<std::size_t>(t) * experts, bias, experts, route_scale, keys[warp],
                              sel[warp], lane, ws.routed + t * kTopK, ws.gamma + t * kTopK);
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
    // A block's halves on a thread pair (canon::four_over_six_half: each thread both candidates over its 8, the pair's
    // half-trees added: four_over_six_block's bits). Whole rounds of the CTA, so every lane takes part in the pair
    // shuffles; lanes past the work store nothing.
    constexpr int kBlocks = G::kInter / 16;
    const int items  = v.count * kBlocks * 2;
    const int rounds = (items + kGuThreads - 1) / kGuThreads;
    for (int r = 0; r < rounds; ++r) {
        const int i    = r * kGuThreads + tid;
        const bool on  = i < items;
        const int half = i & 1, blk = i >> 1;
        const int c = on ? blk / kBlocks : 0, b = on ? blk % kBlocks : 0;
        const int entry  = on ? a.ws.entries[v.first + c] : 0;
        const float amax = on ? canon::f32_from_bits(__ldcg(a.ws.h_amax + entry)) : 1.0F;
        const canon::FourOverSixRow row = canon::four_over_six_row(amax);
        float xv[8] = {};
        float mag   = 0.0F;
        if (on) {
            const uint4 q = __ldcg(reinterpret_cast<const uint4*>(a.ws.h + static_cast<std::size_t>(entry) * G::kInter + 16 * b) + half);
            const std::uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                xv[2 * j]     = canon::f32_from_bits(w[j] << 16);
                xv[2 * j + 1] = canon::f32_from_bits(w[j] & 0xFFFF0000U);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float m = canon::f32_from_bits(canon::f32_bits(xv[j]) & 0x7FFFFFFFU);
                mag           = m > mag ? m : mag;
            }
        }
        const float other = __shfl_xor_sync(0xFFFFFFFFU, mag, 1);
        const canon::FourOverSixWords words = canon::four_over_six_words(other > mag ? other : mag, row);
        std::uint8_t c6[8], c4[8];
        const float e6 = canon::four_over_six_half(xv, words.six, row, c6);
        const float e4 = canon::four_over_six_half(xv, words.four, row, c4);
        const float p6 = __shfl_xor_sync(0xFFFFFFFFU, e6, 1), p4 = __shfl_xor_sync(0xFFFFFFFFU, e4, 1);
        const float err6 = half == 0 ? canon::add_rn(e6, p6) : canon::add_rn(p6, e6);
        const float err4 = half == 0 ? canon::add_rn(e4, p4) : canon::add_rn(p4, e4);
        const bool four  = err4 < err6;
        if (!on) { continue; }
        std::uint32_t packed = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t lo = four ? c4[2 * j] : c6[2 * j];
            const std::uint32_t hi = four ? c4[2 * j + 1] : c6[2 * j + 1];
            packed |= (lo | (hi << 4)) << (8 * j);
        }
        std::uint8_t* out = a.ws.h_rows + static_cast<std::size_t>(entry) * S::kHRowBytes;
        *reinterpret_cast<std::uint32_t*>(out + 8 * b + 4 * half) = packed;
        if (half == 0) { out[S::kHScales + b] = four ? words.four : words.six; }
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
    __shared__ int stalled;
    if (tid == 0) { stalled = wait_count(a.ws.job_done + job, static_cast<std::uint32_t>(S::kGateUpCtas + 1), a.ws.error) ? 0 : 1; }
    __syncthreads();
    if (stalled != 0) {
        tile::cp_async_wait<0>();
        return;
    }
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
    __shared__ int stalled;
    if (tid == 0) { stalled = wait_count(a.ws.job_done + job, static_cast<std::uint32_t>(S::kGateUpCtas), a.ws.error) ? 0 : 1; }
    __syncthreads();
    if (stalled != 0) {
        tile::cp_async_wait<0>();
        return;
    }
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

// ---------------------------------------------------------------------------- one column in one launch

// moe_single: a single column's router and experts as one persistent grid of kSingleCtas CTAs, one an SM, so every CTA
// is resident and the grid's waits complete. CTAs 0-15 compute the router's logits (route_kernel's rows: a warp a row,
// its order); every CTA waits for them and selects the column's experts itself (select_column's rule, a thread an
// expert); CTA c then takes job
// c / 18 (jobs in ascending expert id) and its warps the row groups (c % 18) * 8 + warp, each streaming its gate/up row
// group over all of K (NVFP4: the exact int64 sum of both halves; MXFP8: the two halves' FP32 chains added half 0 +
// half 1, as gate_up_a8_kernel) and then its down row group through one cp.async ring, so the down weights stream
// while the CTA waits for its job's h. h goes to global as BF16; once the job's 144 row groups are counted, each of
// its 18 CTAs forms the job's quantized h row itself (the same four-over-six blocks or MXFP8 groups) and runs its down
// row groups. Every value is the separate kernels' bit for bit. The last CTA to finish resets the counters.
constexpr int kSingleWarps      = 8;
constexpr int kSingleThreads    = 32 * kSingleWarps;
constexpr int kSingleDepth      = 9;                                // stages of a warp's ring
constexpr int kSingleRowGroups  = 2 * Inkling::kInter / 16;         // a job's gate/up row groups = down's (H / 16)
constexpr int kSingleCtasPerJob = kSingleRowGroups / kSingleWarps;  // 18
constexpr int kSingleCtas       = kTopK * kSingleCtasPerJob;        // 144
constexpr int kSingleExperts    = 128;
constexpr int kRouterCtas       = kSingleExperts / kSingleWarps;    // 16
static_assert(Inkling::kHidden / 16 == kSingleRowGroups && kSingleRowGroups % kSingleWarps == 0, "row groups a CTA");
static_assert(kRouterCtas <= kSingleCtas, "the router's rows on the grid's first CTAs");
// Workspace::single: [0] router CTAs done, [1] CTAs finished, [2 + job] the job's gate/up row groups done.
constexpr int kSingleCounters = 2 + kTopK;

struct SingleArgs {
    Args e;                          // the experts' operands (e.routing unused: the routing is the grid's own)
    const std::uint16_t* x = nullptr; // the router's input row, BF16 [H]
    const std::uint16_t* router = nullptr;
    const float* bias = nullptr;
    float route_scale = 0.0F;
};

template <class G, bool A8>
struct SingleSmem {
    using S = Shape<G, A8>;
    alignas(16) std::uint8_t ring[kSingleWarps][kSingleDepth][S::kStageBytes];
    alignas(16) std::uint8_t x[S::kXStride];
    alignas(16) std::uint8_t h[S::kHStride];
    std::int32_t scaled[256];
    float keys[kSingleExperts];
    float sel[kTopK];
    std::int32_t routed[kTopK];
    float gamma[kTopK];
    std::int32_t job_slot[kTopK];
    std::uint32_t red[kSingleWarps];
    float h_amax;
    int stalled;
};

template <class G, bool A8>
__global__ void __launch_bounds__(kSingleThreads, 1) single_kernel(const SingleArgs a) {
    using S = Shape<G, A8>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm      = *reinterpret_cast<SingleSmem<G, A8>*>(smem_raw);
    const int tid = static_cast<int>(threadIdx.x), lane = tid % 32, warp = tid / 32;
    const int g = lane / 4, t = lane % 4;
    const int cta = static_cast<int>(blockIdx.x);
    std::uint32_t* const counters = a.e.ws.single;
    // The router's rows (CTAs 0-15, a warp a row): the weights before the dependency wait, lane l holding elements
    // 8 (l + 32 j) .. + 7, as route_kernel.
    const bool router = cta < kRouterCtas;
    const int row     = cta * kSingleWarps + warp;
    uint4 w[kRouteChunks];
    if (router) {
#pragma unroll
        for (int j = 0; j < kRouteChunks; ++j) {
            w[j] = *reinterpret_cast<const uint4*>(a.router + static_cast<std::size_t>(row) * G::kHidden + 8 * (lane + 32 * j));
        }
    }
    if constexpr (!A8) { a4_mma_fill_scaled(sm.scaled, kSingleThreads); }
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    if (router) {
        float acc = 0.0F;
#pragma unroll
        for (int j = 0; j < kRouteChunks; ++j) {
            const uint4 xv = *reinterpret_cast<const uint4*>(a.x + 8 * (lane + 32 * j));
            const std::uint32_t wa[4] = {w[j].x, w[j].y, w[j].z, w[j].w}, xa[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                acc = __fmaf_rn(canon::f32_from_bits(wa[q] << 16), canon::f32_from_bits(xa[q] << 16), acc);
                acc = __fmaf_rn(canon::f32_from_bits(wa[q] & 0xFFFF0000U), canon::f32_from_bits(xa[q] & 0xFFFF0000U), acc);
            }
        }
#pragma unroll
        for (int o = 16; o >= 1; o /= 2) { acc = __fadd_rn(acc, __shfl_xor_sync(0xFFFFFFFFU, acc, o)); }
        if (lane == 0) { a.e.ws.logits[row] = acc; }
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            release_add(counters + 0);
        }
    }
    // The column's activation row (the gate/up's B operand) while the router finishes.
    for (int i = tid; i < S::kXRowBytes / 8; i += kSingleThreads) {
        reinterpret_cast<uint2*>(sm.x)[i] = reinterpret_cast<const uint2*>(a.e.x.rows)[i];
    }
    if (tid == 0) { sm.stalled = wait_count(counters + 0, kRouterCtas, a.e.ws.error) ? 0 : 1; }
    __syncthreads();
    if (sm.stalled != 0) { return; }
    // The column's selection (select_column's rule, a thread an expert): keys sigmoid_c(l_e) + bias_e, the top 8 by
    // key (ties to the lower index) in slots of descending key, gamma_j = (s_j / sum8(s)) * route_scale.
    static_assert(kSingleExperts <= kSingleThreads, "a thread an expert");
    float sig = 0.0F;
    if (tid < kSingleExperts) {
        sig          = canon::sigmoid_c(__ldcg(a.e.ws.logits + tid));
        sm.keys[tid] = canon::add_rn(sig, a.bias[tid]);
    }
    __syncthreads();
    if (tid < kSingleExperts) {
        const float key = sm.keys[tid];
        int rank        = 0;
        for (int f = 0; f < kSingleExperts; ++f) {
            const float other = sm.keys[f];
            rank += (other > key || (other == key && f < tid)) ? 1 : 0;
        }
        if (rank < kTopK) {
            sm.routed[rank] = tid;
            sm.sel[rank]    = sig;
        }
    }
    __syncthreads();
    if (tid < kTopK) {
        const float sum = canon::add_rn(canon::add_rn(canon::add_rn(sm.sel[0], sm.sel[2]), canon::add_rn(sm.sel[1], sm.sel[3])),
                                        canon::add_rn(canon::add_rn(sm.sel[4], sm.sel[6]), canon::add_rn(sm.sel[5], sm.sel[7])));
        sm.gamma[tid] = canon::mul_rn(canon::div_rn(sm.sel[tid], sum), a.route_scale);
    }
    __syncthreads();
    // Jobs in ascending expert id: job k is the slot whose expert has k smaller ones among the eight.
    if (tid < kTopK) {
        int rank = 0;
#pragma unroll
        for (int j = 0; j < kTopK; ++j) { rank += sm.routed[j] < sm.routed[tid] ? 1 : 0; }
        sm.job_slot[rank] = tid;
        if (cta == 0) { // the call's routing, as moe_route leaves it
            a.e.ws.routed[tid] = sm.routed[tid];
            a.e.ws.gamma[tid]  = sm.gamma[tid];
        }
    }
    __syncthreads();
    const int job    = cta / kSingleCtasPerJob;
    const int slot   = sm.job_slot[job]; // the entry: slot j of column 0
    const int expert = sm.routed[slot];
    const float gamma = sm.gamma[slot];
    const int group  = (cta % kSingleCtasPerJob) * kSingleWarps + warp;
    const std::uint8_t* record = a.e.bank.records + static_cast<std::size_t>(expert) * a.e.bank.record_stride;
    const std::uint8_t* gu     = record + static_cast<std::size_t>(group) * S::kGateUpBlocks * S::kUnit;
    const std::uint8_t* dn     = record + S::kGateUpBytes + static_cast<std::size_t>(group) * S::kDownBytes;
    constexpr int kGu = S::kGateUpBlocks / S::kStageUnits, kDn = S::kDownBlocks / S::kStageUnits, kTotal = kGu + kDn;
    static_assert(kGu % 2 == 0 && (kGu / 2) * S::kStageUnits == S::kHalfBlocks && kTotal >= kSingleDepth,
                  "the K halves are whole stages; a full ring");
    auto& ring       = sm.ring[warp];
    const auto issue = [&](int stage) {
        const std::uint8_t* src = stage < kGu ? gu + static_cast<std::size_t>(stage) * S::kStageBytes
                                              : dn + static_cast<std::size_t>(stage - kGu) * S::kStageBytes;
        for (int i = lane; i < S::kStageBytes / 16; i += 32) {
            tile::cp_async16(ring[stage % kSingleDepth] + 16 * i, src + 16 * i, 16);
        }
    };
#pragma unroll
    for (int k = 0; k < kSingleDepth - 1; ++k) {
        issue(k);
        tile::cp_async_commit();
    }
    // Stage st's units (gate/up blocks st * units .., or down blocks (st - kGu) * units ..) into acc / s.
    const auto step = [&](int st, float (&acc)[4], long long (&s)[4]) {
        tile::cp_async_wait<kSingleDepth - 2>();
        __syncwarp();
        if (st + kSingleDepth - 1 < kTotal) { issue(st + kSingleDepth - 1); }
        tile::cp_async_commit();
        const std::uint8_t* stage = ring[st % kSingleDepth];
        const bool down           = st >= kGu;
        const std::uint8_t* b_row = down ? sm.h : sm.x;
        const int scales          = down ? S::kHScales : S::kXScales;
        const int first           = (down ? st - kGu : st) * S::kStageUnits;
#pragma unroll
        for (int u = 0; u < S::kStageUnits; ++u) {
            const int b = first + u;
            if constexpr (A8) {
                a8_mma_unit(stage + u * S::kUnit, b_row + 32 * b, b_row[scales + b], lane, g, t, acc);
            } else {
                const int sa = sm.scaled[b_row[scales + b]];
                a4_mma_unit(stage + u * S::kUnit, a4_x_quad(*reinterpret_cast<const unsigned short*>(b_row + 8 * b + 2 * t)),
                            sm.scaled, sa, sa, g, t, s);
            }
        }
        __syncwarp();
    };
    // Gate/up: NVFP4 one exact sum s over both halves; MXFP8 c0 over half 0's blocks, c1 over half 1's.
    float c0[4] = {0.0F, 0.0F, 0.0F, 0.0F}, c1[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    long long s[4] = {0, 0, 0, 0};
    for (int st = 0; st < kGu / 2; ++st) { step(st, c0, s); }
    for (int st = kGu / 2; st < kGu; ++st) { step(st, c1, s); }
    std::uint16_t y[4], up[4];
    if constexpr (A8) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { y[i] = canon::f32_to_bf16_rn(canon::add_rn(c0[i], c1[i])); }
    } else {
        const float rs = detail::nvfp4_row_scale(a.e.x.amax[0], a.e.bank.amax[2 * expert],
                                                 detail::nvfp4_rcp(a.e.bank.denominators[0]));
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            y[i] = canon::f32_to_bf16_rn(canon::mul_rn(canon::mul_rn(canon::i64_to_f32_rn(s[i]), 1.0F / 1048576.0F), rs));
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) { up[i] = static_cast<std::uint16_t>(__shfl_down_sync(0xFFFFFFFFU, y[i], 4)); }
    // Column 0's intermediates (t = 0): gate rows g (even) and g + 8, their up rows from lane + 4.
    std::uint16_t* h_row = a.e.ws.h + static_cast<std::size_t>(slot) * G::kInter;
    if (t == 0 && (g & 1) == 0) {
        h_row[group * 8 + g / 2]     = swiglu(y[0], up[0], gamma, a.e.bank.gate_softcap, a.e.bank.up_softcap);
        h_row[group * 8 + g / 2 + 4] = swiglu(y[2], up[2], gamma, a.e.bank.gate_softcap, a.e.bank.up_softcap);
        __threadfence();
    }
    __syncwarp();
    if (lane == 0) { release_add(counters + 2 + job); }
    // The job's h, once its 144 row groups are counted (the down weights keep streaming meanwhile).
    __syncthreads();
    if (tid == 0) { sm.stalled = wait_count(counters + 2 + job, kSingleRowGroups, a.e.ws.error) ? 0 : 1; }
    __syncthreads();
    if (sm.stalled != 0) {
        tile::cp_async_wait<0>();
        return;
    }
    if constexpr (A8) {
        // MXFP8: each 32 intermediates one group (gate_up_a8_kernel's).
        if (tid < G::kInter / 32) {
            float values[32];
#pragma unroll
            for (int j = 0; j < 32; ++j) { values[j] = bf16f(__ldcg(h_row + 32 * tid + j)); }
            const canon::Mxfp8Group q = canon::mxfp8_group(values);
            auto* words               = reinterpret_cast<std::uint32_t*>(sm.h + 32 * tid);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                words[j] = q.code[4 * j] | (static_cast<std::uint32_t>(q.code[4 * j + 1]) << 8) |
                           (static_cast<std::uint32_t>(q.code[4 * j + 2]) << 16) |
                           (static_cast<std::uint32_t>(q.code[4 * j + 3]) << 24);
            }
            sm.h[S::kHScales + tid] = q.exponent;
        }
    } else {
        // NVFP4: the row's amax (the largest magnitude's bits), then four-over-six by block halves on thread pairs
        // (gate_up_a4_kernel's last-CTA quantization).
        std::uint32_t m = 0;
        for (int k = tid; k < G::kInter; k += kSingleThreads) {
            m = max(m, canon::f32_bits(bf16f(__ldcg(h_row + k))) & 0x7FFFFFFFU);
        }
#pragma unroll
        for (int o = 16; o >= 1; o /= 2) { m = max(m, __shfl_xor_sync(0xFFFFFFFFU, m, o)); }
        if (lane == 0) { sm.red[warp] = m; }
        __syncthreads();
        if (tid == 0) {
            std::uint32_t all = 0;
#pragma unroll
            for (int i = 0; i < kSingleWarps; ++i) { all = max(all, sm.red[i]); }
            sm.h_amax = canon::f32_from_bits(all);
        }
        __syncthreads();
        constexpr int kBlocks = G::kInter / 16;
        static_assert(2 * kBlocks <= kSingleThreads, "one round of block halves");
        const int i    = tid;
        const bool on  = i < 2 * kBlocks;
        const int half = i & 1, b = on ? i >> 1 : 0;
        const canon::FourOverSixRow fos = canon::four_over_six_row(on ? sm.h_amax : 1.0F);
        float xv[8] = {};
        float mag   = 0.0F;
        if (on) {
            const uint4 q = __ldcg(reinterpret_cast<const uint4*>(h_row + 16 * b) + half);
            const std::uint32_t hw[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                xv[2 * j]     = canon::f32_from_bits(hw[j] << 16);
                xv[2 * j + 1] = canon::f32_from_bits(hw[j] & 0xFFFF0000U);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float mj = canon::f32_from_bits(canon::f32_bits(xv[j]) & 0x7FFFFFFFU);
                mag            = mj > mag ? mj : mag;
            }
        }
        const float other = __shfl_xor_sync(0xFFFFFFFFU, mag, 1);
        const canon::FourOverSixWords words = canon::four_over_six_words(other > mag ? other : mag, fos);
        std::uint8_t c6[8], c4[8];
        const float e6 = canon::four_over_six_half(xv, words.six, fos, c6);
        const float e4 = canon::four_over_six_half(xv, words.four, fos, c4);
        const float p6 = __shfl_xor_sync(0xFFFFFFFFU, e6, 1), p4 = __shfl_xor_sync(0xFFFFFFFFU, e4, 1);
        const float err6 = half == 0 ? canon::add_rn(e6, p6) : canon::add_rn(p6, e6);
        const float err4 = half == 0 ? canon::add_rn(e4, p4) : canon::add_rn(p4, e4);
        const bool four  = err4 < err6;
        if (on) {
            std::uint32_t packed = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const std::uint32_t lo = four ? c4[2 * j] : c6[2 * j];
                const std::uint32_t hi = four ? c4[2 * j + 1] : c6[2 * j + 1];
                packed |= (lo | (hi << 4)) << (8 * j);
            }
            *reinterpret_cast<std::uint32_t*>(sm.h + 8 * b + 4 * half) = packed;
            if (half == 0) { sm.h[S::kHScales + b] = four ? words.four : words.six; }
        }
    }
    __syncthreads();
    // Down: one chain over K = I (MXFP8 one FP32 chain in block order; NVFP4 the exact sum).
    float cd[4]     = {0.0F, 0.0F, 0.0F, 0.0F};
    long long sd[4] = {0, 0, 0, 0};
    for (int st = kGu; st < kTotal; ++st) { step(st, cd, sd); }
    tile::cp_async_wait<0>();
    if (t == 0) {
        std::uint16_t lo, hi;
        if constexpr (A8) {
            lo = canon::f32_to_bf16_rn(cd[0]);
            hi = canon::f32_to_bf16_rn(cd[2]);
        } else {
            const float rs = detail::nvfp4_row_scale(sm.h_amax, a.e.bank.amax[2 * expert + 1],
                                                     detail::nvfp4_rcp(a.e.bank.denominators[1]));
            lo = canon::f32_to_bf16_rn(canon::mul_rn(canon::mul_rn(canon::i64_to_f32_rn(sd[0]), 1.0F / 1048576.0F), rs));
            hi = canon::f32_to_bf16_rn(canon::mul_rn(canon::mul_rn(canon::i64_to_f32_rn(sd[2]), 1.0F / 1048576.0F), rs));
        }
        std::uint16_t* out = a.e.outputs + static_cast<std::size_t>(slot) * G::kHidden + group * 16 + g;
        out[0] = lo;
        out[8] = hi;
    }
    // The last CTA resets the counters for the next call (every CTA has passed every wait).
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        if (acq_rel_add(counters + 1) == gridDim.x - 1U) {
#pragma unroll
            for (int i = 0; i < kSingleCounters; ++i) { counters[i] = 0U; }
        }
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
Workspace layout(std::int32_t experts, std::int32_t intermediate, std::int32_t capacity, Take&& take) {
    if (capacity <= 0 || capacity > kMaxCapacity) {
        throw std::invalid_argument("w4a4_sparse_moe: a workspace holds 1..8191 columns");
    }
    const std::size_t e = static_cast<std::size_t>(experts), c = static_cast<std::size_t>(capacity);
    const std::size_t m = c * kTopK, i = static_cast<std::size_t>(intermediate);
    const std::size_t h_row = std::max<std::size_t>(i / 2 + i / 16, i + i / 32);
    Workspace w;
    w.capacity   = capacity;
    w.logits     = reinterpret_cast<float*>(take(c * e * 4));
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
    w.error      = reinterpret_cast<std::uint32_t*>(take(4));
    w.single     = reinterpret_cast<std::uint32_t*>(take(kSingleCounters * 4));
    w.x_plane    = reinterpret_cast<std::uint8_t*>(take(m * kPromptXRowBytes));
    w.h_plane    = reinterpret_cast<std::uint8_t*>(take(m * kPromptHRowBytes));
    w.tiles      = reinterpret_cast<std::int32_t*>(take(2 * (e + m / 64 + 1) * 4));
    w.pass_tiles = reinterpret_cast<std::int32_t*>(take(2 * 4));
    w.job_records = reinterpret_cast<const std::uint8_t**>(take(e * sizeof(void*)));
    return w;
}

} // namespace

std::size_t workspace_bytes(std::int32_t experts, std::int32_t intermediate, std::int32_t capacity) {
    std::size_t total = 0;
    (void)layout(experts, intermediate, capacity, [&](std::size_t bytes) {
        total += align256(bytes);
        return static_cast<std::byte*>(nullptr);
    });
    return total;
}

Workspace carve_workspace(void* base, std::int32_t experts, std::int32_t intermediate, std::int32_t capacity) {
    auto* p = static_cast<std::byte*>(base);
    return layout(experts, intermediate, capacity, [&](std::size_t bytes) {
        std::byte* out = p;
        p += align256(bytes);
        return out;
    });
}

void moe_route(const std::uint16_t* x, std::int32_t columns, const std::uint16_t* router, const float* bias,
               std::int32_t experts, std::int32_t hidden, float route_scale, const Workspace& workspace,
               cudaStream_t stream) {
    if (hidden != Inkling::kHidden || experts <= 0 || experts > 256 || columns <= 0 || columns > workspace.capacity) {
        throw std::invalid_argument("w4a4_sparse_moe router: H = 2304, at most 256 experts, 1..capacity columns");
    }
    const dim3 grid((experts + kRouteWarps - 1) / kRouteWarps, (columns + kRouteColumns - 1) / kRouteColumns);
    CUDA_CHECK(pdl::launch_with(pdl::Dependency::Programmatic, {grid, dim3(kRouteThreads), 0, stream}, route_kernel, x,
                                columns, router, bias, experts, route_scale, workspace));
    check_launch("router");
}

namespace {
// The prompt route's selection: a warp per column, eight a CTA.
__global__ void __launch_bounds__(kRouteThreads) select_kernel(const float* __restrict__ logits, int columns,
                                                               const float* __restrict__ bias, int experts,
                                                               float route_scale, Workspace ws) {
    __shared__ float keys[kRouteWarps][256];
    __shared__ float sel[kRouteWarps][kTopK];
    const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
    pdl::wait_for_dependencies();
    const int t = static_cast<int>(blockIdx.x) * kRouteWarps + warp;
    if (t >= columns) { return; }
    detail::select_column(logits + static_cast<std::size_t>(t) * experts, bias, experts, route_scale, keys[warp],
                          sel[warp], lane, ws.routed + t * kTopK, ws.gamma + t * kTopK);
}
} // namespace

void moe_select(const float* logits, std::int32_t columns, const float* bias, std::int32_t experts, float route_scale,
                const Workspace& workspace, cudaStream_t stream) {
    if (experts <= 0 || experts > 256 || columns <= 0 || columns > workspace.capacity) {
        throw std::invalid_argument("w4a4_sparse_moe select: at most 256 experts, 1..capacity columns");
    }
    const dim3 grid((columns + kRouteWarps - 1) / kRouteWarps);
    CUDA_CHECK(pdl::launch_with(pdl::Dependency::Programmatic, {grid, dim3(kRouteThreads), 0, stream}, select_kernel,
                                logits, columns, bias, experts, route_scale, workspace));
    check_launch("select");
    moe_dispatch(routed_routing(workspace), columns, experts, workspace, stream);
}

void moe_dispatch(const Routing& routing, std::int32_t columns, std::int32_t experts, const Workspace& workspace,
                  cudaStream_t stream) {
    if (columns <= 0 || columns > workspace.capacity || experts <= 0 || experts > 256) {
        throw std::invalid_argument("w4a4_sparse_moe dispatch: 1..capacity columns and at most 256 experts");
    }
    CUDA_CHECK(pdl::launch_with(pdl::Dependency::Serialized, {dim3(1), dim3(kDispatchThreads), 0, stream},
                                dispatch_kernel, routing, columns, experts, workspace));
    check_launch("dispatch");
}

namespace {
template <bool A8>
bool single_fits() {
    using G = Inkling;
    allow_shared(single_kernel<G, A8>, sizeof(SingleSmem<G, A8>));
    int device = 0, sms = 0, blocks = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, single_kernel<G, A8>, kSingleThreads,
                                                             sizeof(SingleSmem<G, A8>)));
    return blocks >= 1 && static_cast<long long>(blocks) * sms >= kSingleCtas;
}
} // namespace

bool moe_single_supported(ExpertFormat format) {
    static const bool a4 = single_fits<false>();
    static const bool a8 = single_fits<true>();
    return format == ExpertFormat::Mxfp8 ? a8 : a4;
}

void moe_single(const std::uint16_t* x, const Activations& rows, const std::uint16_t* router, const float* bias,
                float route_scale, const Bank& bank, const Workspace& workspace, std::uint16_t* outputs,
                cudaStream_t stream) {
    using G = Inkling;
    if (bank.hidden != G::kHidden || bank.intermediate != G::kInter || bank.experts != kSingleExperts ||
        rows.columns != 1 || workspace.capacity < 1) {
        throw std::invalid_argument("w4a4_sparse_moe single: one column of Inkling 2's [128, 2304, 1152] banks");
    }
    if (!moe_single_supported(bank.format)) {
        throw std::invalid_argument("w4a4_sparse_moe single: the device cannot hold its grid resident");
    }
    SingleArgs args;
    args.e           = Args{rows, Routing{workspace.routed, workspace.gamma}, bank, workspace, outputs};
    args.x           = x;
    args.router      = router;
    args.bias        = bias;
    args.route_scale = route_scale;
    if (bank.format == ExpertFormat::Nvfp4FourOverSix) {
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Programmatic,
                                    {dim3(kSingleCtas), dim3(kSingleThreads), sizeof(SingleSmem<G, false>), stream},
                                    single_kernel<G, false>, args));
    } else {
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Programmatic,
                                    {dim3(kSingleCtas), dim3(kSingleThreads), sizeof(SingleSmem<G, true>), stream},
                                    single_kernel<G, true>, args));
    }
    check_launch("single");
}

void moe_experts(const Activations& x, const Routing& routing, const Bank& bank, const Workspace& workspace,
                 std::uint16_t* outputs, cudaStream_t stream) {
    using G = Inkling;
    if (bank.hidden != G::kHidden || bank.intermediate != G::kInter || bank.experts <= 0 || bank.experts > 256) {
        throw std::invalid_argument("w4a4_sparse_moe: the kernels are built for Inkling 2's [E, 2304, 1152] banks");
    }
    if (x.columns <= 0 || x.columns > workspace.capacity) {
        throw std::invalid_argument("w4a4_sparse_moe: 1..capacity columns");
    }
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
