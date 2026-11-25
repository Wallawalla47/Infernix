// Inkling 2's routed experts, the narrow route of decode and verification (include/infernix/ops/w4a4_sparse_moe.h,
// design docs/maintainer/inkling2_mini-design.md §10.3).
//
// - Dispatch (one CTA): the call's entries grouped by expert (ascending entry within an expert), jobs in ascending
//   expert id, and the per-call state cleared: each job's completion count and each entry's h amax.
// - Gate/up (a CTA of 8 warps per 4 row groups of a job's gate/up matrix: 2 K halves a row group, each streaming its
//   half through a private cp.async ring): the job's columns in passes of 8 (one MMA's n). NVFP4: the native core
//   (ops/common/a4_native.cuh: block-scaled e2m1 MMAs, an FP32 chain a K half, the halves added half 0 + half 1; x
//   staged in its slot order) or Nvfp4Core::Exact's int64 core (ops/common/a4_mma.cuh), and the row scale
//   (amax_x * amax_w) * rcp_rn(1536 * den); MXFP8: block-scaled FP8 MMAs accumulating in FP32 in block order, the
//   halves added half 0 + half 1. Interleaved rows put gate_i and up_i on
//   lanes 4 apart; h = the softcapped SwiGLU times gamma, BF16. NVFP4: h to global with its row amax (atomicMax of the
//   magnitude's bits, order-free), and the job's last CTA (the one completing its count) quantizes the job's h rows by
//   four-over-six once; MXFP8: a CTA's 32 intermediates are one MXFP8 group, quantized in place.
// - Down (a CTA per kDownWarps output row groups of a job): its weight units copied before it waits for its job's h
//   (the job's count, a release from each gate/up CTA), then the same core over K = I (native: one FP32 chain in
//   block order, h staged in its slot order).
// Each entry's arithmetic depends only on its own column, in a fixed order, so a column's outputs never depend on the
// call's other columns.

#include "infernix/ops/w4a4_sparse_moe.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "core/tensor.h"
#include "ops/common/a4_mma.cuh"
#include "ops/common/a4_native.cuh"
#include "ops/common/canonical_math.h"
#include "ops/common/mma.cuh"
#include "ops/ggml/ggml_a8.h"
#include "ops/ggml/ggml_a8_mma.cuh"
#include "ops/ggml/ggml_linear_a8.h"
#include "ops/ggml/ggml_tile.cuh"
#include "ops/offloaded_sparse_moe/nvfp4_a8.h"
#include "ops/linear/fp8/fp8_a16_codec.cuh"
#include "ops/offloaded_sparse_moe/cuda/a16_limbs.cuh"
#include "ops/w4a4_sparse_moe/expert_math.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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
// a lost count fails the call through its error word, never hangs the GPU. A wait that has lasted kErrorPollNs reads
// *error once every kErrorPollNs (the word is mapped host memory: one PCIe read a millisecond, never on the fast path)
// and gives up at once when it is set: the round has already failed, so every later wait of it (other CTAs, later
// visits of the same graph) costs a millisecond, not its own bound (candidates JM-02; the 11-22 skeleton arm spun ~12 h,
// one 2 s bound a CTA wave a launch).
constexpr std::uint64_t kWaitTimeoutNs = 2'000'000'000ULL;
constexpr std::uint64_t kErrorPollNs   = 1'000'000ULL;

__device__ __forceinline__ bool wait_count(const std::uint32_t* word, std::uint32_t target, std::uint32_t* error) {
    std::uint32_t seen = 0;
    std::uint64_t since = 0;
    std::uint64_t poll  = 0;
    for (bool first = true;; first = false) {
        std::uint32_t v;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(word) : "memory");
        if (v >= target) { return true; }
        const std::uint64_t now = global_ns();
        if (first) { poll = now + kErrorPollNs; }
        if (first || v != seen) {
            seen  = v;
            since = now;
        } else if (now - since > kWaitTimeoutNs) {
            atomicOr(error, kErrorDownStalled);
            return false;
        }
        if (now >= poll) {
            poll = now + kErrorPollNs;
            if (*static_cast<volatile std::uint32_t*>(error) != 0U) { return false; }
        }
        __nanosleep(64);
    }
}

__device__ __forceinline__ void release_add(std::uint32_t* word) {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(word) : "memory");
}
__device__ __forceinline__ void release_add(std::uint32_t* word, std::uint32_t n) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(word), "r"(n) : "memory");
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
                                                              float route_scale, Workspace ws, std::int32_t* record) {
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
    if (record != nullptr && tid < kTopK) { // the last column's experts: a later call's prefetch hint (relaxed)
        asm volatile("st.relaxed.gpu.global.s32 [%0], %1;" ::"l"(record + tid), "r"(ws.routed[(columns - 1) * kTopK + tid])
                     : "memory");
    }
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

// Native: a K half's FP32 chain of native steps, the halves added half 0 + half 1 (single_kernel's chain); not Native
// (Nvfp4Core::Exact): the exact int64 sums.
template <class G, bool Native>
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
        // The pass's x rows and row scales while the rings fill (Native: the codes in the core's slot order).
        constexpr int kWords = S::kXRowBytes / 8;
        for (int i = tid; i < n * kWords; i += kGuThreads) {
            const int c      = i / kWords;
            const int column = a.ws.entries[v.first + base + c] / kTopK;
            uint2 word = reinterpret_cast<const uint2*>(a.x.rows + static_cast<std::size_t>(column) * S::kXRowBytes)[i % kWords];
            if constexpr (Native) {
                if (i % kWords < S::kXScales / 8) { word = make_uint2(a4_native_order(word.x), a4_native_order(word.y)); }
            }
            reinterpret_cast<uint2*>(sm.x[c])[i % kWords] = word;
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
        const std::uint8_t* xc  = sm.x[min(g, n - 1)]; // Native: column g's codes and scales
        long long s[4]          = {0, 0, 0, 0};
        float c[4]              = {0.0F, 0.0F, 0.0F, 0.0F};
        for (int st = 0; st < S::kStages; ++st) {
            tile::cp_async_wait<kDepth - 2>();
            __syncwarp();
            if (st + kDepth - 1 < S::kStages) { issue(st + kDepth - 1); }
            tile::cp_async_commit();
            const std::uint8_t* stage = ring[st % kDepth];
#pragma unroll
            for (int u = 0; u < S::kStageUnits; ++u) {
                const int b = st * S::kStageUnits + u;
                if constexpr (Native) {
                    if (u % 4 == 0) {
                        a4_native_step(stage + u * S::kUnit, xc + 8 * (b0 + b),
                                       *reinterpret_cast<const std::uint32_t*>(xc + S::kXScales + b0 + b), g, t, c);
                    }
                } else {
                    a4_mma_unit(stage + u * S::kUnit, a4_x_quad(*reinterpret_cast<const unsigned short*>(xg + 8 * b)),
                                sm.scaled, sm.scaled[xs0[b]], sm.scaled[xs1[b]], g, t, s);
                }
            }
            __syncwarp();
        }
        tile::cp_async_wait<0>();
        __syncthreads();
        // The second half's sums join the first's (exact int64 adds; Native: half 0 + half 1 in FP32) through the
        // rings' memory.
        auto* half  = reinterpret_cast<long long (*)[32][4]>(&sm.ring[0][0][0]);
        auto* halfc = reinterpret_cast<float (*)[32][4]>(&sm.ring[0][0][0]);
        if (k_half == 1) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if constexpr (Native) {
                    halfc[rg][lane][i] = c[i];
                } else {
                    half[rg][lane][i] = s[i];
                }
            }
        }
        __syncthreads();
        if (k_half == 0) {
            std::uint16_t y[4], up[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float rs = sm.rs[min(2 * t + (i & 1), n - 1)];
                if constexpr (Native) {
                    y[i] = canon::f32_to_bf16_rn(canon::mul_rn(canon::add_rn(c[i], halfc[rg][lane][i]), rs));
                } else {
                    s[i] += half[rg][lane][i];
                    y[i] = canon::f32_to_bf16_rn(
                        canon::mul_rn(canon::mul_rn(canon::i64_to_f32_rn(s[i]), 1.0F / 1048576.0F), rs));
                }
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

// Native: one FP32 chain of native steps in block order (single_kernel's down chain); not Native (Nvfp4Core::Exact):
// the exact int64 sum.
template <class G, bool Native>
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
            std::uint32_t word =
                __ldcg(reinterpret_cast<const std::uint32_t*>(a.ws.h_rows + static_cast<std::size_t>(entry) * S::kHRowBytes) +
                       i % kWords);
            if constexpr (Native) {
                if (i % kWords < S::kHScales / 4) { word = a4_native_order(word); } // the codes in the core's slot order
            }
            reinterpret_cast<std::uint32_t*>(sm.h[c])[i % kWords] = word;
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
        float cn[4]    = {0.0F, 0.0F, 0.0F, 0.0F};
        if constexpr (Native) {
            const std::uint8_t* hc = sm.h[min(g, n - 1)];
#pragma unroll 6
            for (int b = 0; b < S::kDownBlocks; b += 4) {
                a4_native_step(units + b * S::kUnit, hc + 8 * b, *reinterpret_cast<const std::uint32_t*>(hc + S::kHScales + b),
                               g, t, cn);
            }
        } else {
#pragma unroll 8
            for (int b = 0; b < S::kDownBlocks; ++b) {
                a4_mma_unit(units + b * S::kUnit, a4_x_quad(*reinterpret_cast<const unsigned short*>(hg + 8 * b)), sm.scaled,
                            sm.scaled[hs0[b]], sm.scaled[hs1[b]], g, t, s);
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = 2 * t + (i & 1);
            if (c < n) {
                const int entry = a.ws.entries[v.first + base + c];
                const int row   = group * 16 + g + (i < 2 ? 0 : 8);
                const float value = Native ? canon::mul_rn(cn[i], sm.rs[c])
                                           : canon::mul_rn(canon::mul_rn(canon::i64_to_f32_rn(s[i]), 1.0F / 1048576.0F), sm.rs[c]);
                a.outputs[static_cast<std::size_t>(entry) * G::kHidden + row] = canon::f32_to_bf16_rn(value);
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------- NVFP4 A16 (--a16-activations)

namespace a16 = infernix::ops::offloaded_moe::wide;

// The A16 kernels keep the A4 kernels' shapes (row groups, K halves, rings, passes of kPass columns, the job counts);
// a unit's B operand is a column's limb row read from L2 (x: complete before the call; h: the job's limb rows,
// read after its count), and an output is a16_row_output(S, emax, m).
__device__ __forceinline__ float a16_multiplier(float amax_w, float denominator) { return canon::div_rn(amax_w, denominator); }
__device__ __forceinline__ int a16_h_emax(std::uint32_t h_amax_bits) { return a16::a16_emax(h_amax_bits >> 16); }

template <class G>
struct GateUpA16Smem {
    using S = Shape<G, false>;
    std::int32_t scaled[256];
    alignas(16) std::uint8_t ring[kSplit * kWarps][kDepth][S::kStageBytes];
    std::uint16_t h[kPass][kCtaInter];
    int last;
};

template <class G>
__global__ void __launch_bounds__(kGuThreads) gate_up_a16_kernel(Args a) {
    using S = Shape<G, false>;
    constexpr int kXLimbs = kA16XRowBytes, kHLimbs = kA16HRowBytes;
    static_assert(G::kHidden / 64 * 192 == kXLimbs && G::kInter / 64 * 192 == kHLimbs, "the limb rows of the geometry");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm = *reinterpret_cast<GateUpA16Smem<G>*>(smem_raw);
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
    auto& ring       = sm.ring[warp];
    const auto issue = [&](int stage) {
        const std::uint8_t* src = units + static_cast<std::size_t>(stage) * S::kStageBytes;
        for (int i = lane; i < S::kStageBytes / 16; i += 32) { tile::cp_async16(ring[stage % kDepth] + 16 * i, src + 16 * i, 16); }
    };
    a4_mma_fill_scaled(sm.scaled, kGuThreads);
    const float m     = a16_multiplier(a.bank.amax[2 * v.expert], a.bank.denominators[0]);
    const int h_first = static_cast<int>(blockIdx.x) * kCtaInter;
    const auto column_of_entry = [&](int c) { return a.ws.entries[v.first + c] / kTopK; };
    for (int base = 0; base < v.count; base += kPass) {
        const int n = min(kPass, v.count - base);
#pragma unroll
        for (int k = 0; k < kDepth - 1; ++k) {
            issue(k);
            tile::cp_async_commit();
        }
        __syncthreads(); // the scale table
        const int b0               = k_half * S::kHalfBlocks;
        const std::uint8_t* limbs  = a.x.rows + static_cast<std::size_t>(column_of_entry(base + min(g, n - 1))) * kXLimbs;
        long long s[4]             = {0, 0, 0, 0};
        for (int st = 0; st < S::kStages; ++st) {
            tile::cp_async_wait<kDepth - 2>();
            __syncwarp();
            if (st + kDepth - 1 < S::kStages) { issue(st + kDepth - 1); }
            tile::cp_async_commit();
            const std::uint8_t* stage = ring[st % kDepth];
#pragma unroll
            for (int u = 0; u < S::kStageUnits; ++u) {
                const int b = b0 + st * S::kStageUnits + u;
                a16::a16_unit<a16::LimbLoad::Nc>(stage + u * S::kUnit, limbs + a16::a16_block_offset(b, t), sm.scaled, g, t, s);
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
                const int c = min(2 * t + (i & 1), n - 1);
                y[i]        = canon::a16_row_output(s[i], a.x.emax[column_of_entry(base + c)], m);
            }
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
        // h to global and each column's largest magnitude over this CTA's intermediates.
        for (int i = tid; i < n * kCtaInter; i += kGuThreads) {
            const int c = i / kCtaInter, k = i % kCtaInter;
            const int entry = a.ws.entries[v.first + base + c];
            a.ws.h[static_cast<std::size_t>(entry) * G::kInter + h_first + k] = sm.h[c][k];
        }
        if (warp < n) {
            const std::uint32_t mag = canon::f32_bits(bf16f(sm.h[warp][lane])) & 0x7FFFFFFFU;
            std::uint32_t mx        = mag;
#pragma unroll
            for (int o = 16; o >= 1; o /= 2) { mx = max(mx, __shfl_xor_sync(0xFFFFFFFFU, mx, o)); }
            if (lane == 0) { atomicMax(&a.ws.h_amax[a.ws.entries[v.first + base + warp]], mx); }
        }
        __syncthreads(); // the next pass reuses the rings and h
    }
    // The job's last CTA encodes the job's h rows as limbs once, then counts again: the down CTAs wait for
    // kGateUpCtas + 1.
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
        const auto* p   = reinterpret_cast<const uint4*>(a.ws.h + static_cast<std::size_t>(entry) * G::kInter + 16 * b);
        std::uint16_t hv[16];
        a16::unpack16(__ldcg(p), __ldcg(p + 1), hv);
        a16::store_a16_block(a.ws.h_limbs + static_cast<std::size_t>(entry) * kHLimbs, b, hv,
                             a16_h_emax(__ldcg(a.ws.h_amax + entry)));
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        release_add(&a.ws.job_done[job]);
    }
}

template <class G>
struct DownA16Smem {
    using S = Shape<G, false>;
    std::int32_t scaled[256];
    alignas(16) std::uint8_t units[S::kDownWarps][S::kDownBytes];
};

template <class G>
__global__ void __launch_bounds__(Shape<G, false>::kDownThreads) down_a16_kernel(Args a) {
    using S = Shape<G, false>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm = *reinterpret_cast<DownA16Smem<G>*>(smem_raw);
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
    tile::cp_async_wait<0>();
    __syncthreads();
    if (stalled != 0) { return; }
    const float m = a16_multiplier(a.bank.amax[2 * v.expert + 1], a.bank.denominators[1]);
    const std::uint8_t* units = sm.units[warp];
    for (int base = 0; base < v.count; base += kPass) {
        const int n = min(kPass, v.count - base);
        const std::uint8_t* limbs =
            a.ws.h_limbs + static_cast<std::size_t>(a.ws.entries[v.first + base + min(g, n - 1)]) * kA16HRowBytes;
        long long s[4] = {0, 0, 0, 0};
#pragma unroll 8
        for (int b = 0; b < S::kDownBlocks; ++b) {
            a16::a16_unit<a16::LimbLoad::Cg>(units + b * S::kUnit, limbs + a16::a16_block_offset(b, t), sm.scaled, g, t, s);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = 2 * t + (i & 1);
            if (c < n) {
                const int entry = a.ws.entries[v.first + base + c];
                const int row   = group * 16 + g + (i < 2 ? 0 : 8);
                a.outputs[static_cast<std::size_t>(entry) * G::kHidden + row] =
                    canon::a16_row_output(s[i], a16_h_emax(__ldcg(a.ws.h_amax + entry)), m);
            }
        }
    }
}

// ---------------------------------------------------------------------------- NVFP4 A8 (--a8-activations)

// The header's A8 contract on the narrow route (decode and verification calls of more than one column; moe_single
// takes one column, the prompt route wide_expert_a8.cuh's grouped GEMM, both with these bits): a warp owns a row group
// (16 rows: eight intermediates' interleaved gate and up rows, or 16 down rows) over the whole K, in passes of
// kPass columns, one m16n8k16 s8 MMA per 144-byte unit (lane (g, t): rows g and g + 8 at K 4t .. 4t + 3 of the block,
// the low and high nibbles of the unit's code word 32t + 4g; column g's codes at the same K), its int32 accumulators
// started at 1.5 * 2^23 (ggml_a8_mma.cuh) so that P_b * S_b is one FFMA with {S_b, -1.5 * 2^23 * S_b} from a shared
// table, and the term one more with the column's slice scale. One thread adds an output's terms in block order, so an
// entry's bits depend on its own codes and its expert's record alone. A gate/up CTA's 32 intermediates are one slice of
// h: it encodes each column's slice itself, and the down CTAs wait for every gate/up CTA's count.
//
// Residency (ncu, T = 3 and 8): the gate/up CTA reads the columns' x rows through L1 instead of staging them (20.7 KB
// of shared memory a CTA), so four CTAs (16 warps) share an SM, as the native core's gate/up; a down CTA's 8 warps share
// one copy of the pass's h rows (one SM a CTA, 8 warps, as the native down's two CTAs of 4).
constexpr int kA8Warps   = 4; // gate/up row groups a CTA
constexpr int kA8Threads = 32 * kA8Warps;
constexpr int kA8DownWarps   = 8; // down row groups a CTA
constexpr int kA8DownThreads = 32 * kA8DownWarps;
constexpr int kA8Depth   = 4; // stages of a gate/up warp's ring
constexpr int kA8StageUnits = 8;

__device__ __forceinline__ void fill_a8_scaled(float2* scaled, int threads) {
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += threads) {
        const float s = static_cast<float>(canon::e4m3_scaled(static_cast<unsigned>(i)));
        scaled[i]     = make_float2(s, __fmul_rn(ggml::a8::kMinusMagic, s));
    }
}

// One unit's terms for rows g and g + 8 and columns 2t, 2t + 1 (c[i]: row g + 8 (i >> 1), column 2t + (i & 1)); b0:
// column g's four codes, d0 / d1: the slice scales of columns 2t and 2t + 1.
__device__ __forceinline__ void a8_unit_terms(const std::uint8_t* unit, unsigned b0, float d0, float d1, const float2* scaled,
                                              int g, int t, float (&c)[4]) {
    const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(unit + 32 * t + 4 * g);
    const unsigned a0 = canon::e2m1_x2_quad(word & 0x0F0F0F0FU), a1 = canon::e2m1_x2_quad((word >> 4) & 0x0F0F0F0FU);
    int p[4];
    ggml::a8::mma_k16_magic(p, a0, a1, b0);
    const float2 lo = scaled[unit[128 + g]], hi = scaled[unit[136 + g]];
    c[0] = __fmaf_rn(ggml::a8::scaled_term(p[0], lo.x, lo.y), d0, c[0]);
    c[1] = __fmaf_rn(ggml::a8::scaled_term(p[1], lo.x, lo.y), d1, c[1]);
    c[2] = __fmaf_rn(ggml::a8::scaled_term(p[2], hi.x, hi.y), d0, c[2]);
    c[3] = __fmaf_rn(ggml::a8::scaled_term(p[3], hi.x, hi.y), d1, c[3]);
}

// One 32-element BF16 slice v (shared memory) encoded by ggml_a8.h's codec as slice s of an A8 row of K = k codes: its
// codes at row + 32 s, its FP32 scale at row + k + 4 s (one thread; the h rows' encoder in every A8 kernel).
__device__ __forceinline__ void a8_encode_slice(const std::uint16_t* v, std::uint8_t* row, int k, int s) {
    unsigned key = 0;
#pragma unroll
    for (int j = 0; j < 32; ++j) { key = max(key, static_cast<unsigned>(v[j] & 0x7FFFU)); }
    const ggml::a8::SliceScale sc = ggml::a8::slice_scale(key);
    auto* words = reinterpret_cast<std::uint32_t*>(row + 32 * s);
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        std::uint32_t packed = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            packed |= (static_cast<std::uint32_t>(ggml::a8::code(v[4 * w + j], sc.inv)) & 0xFFU) << (8 * j);
        }
        words[w] = packed;
    }
    reinterpret_cast<float*>(row + k)[s] = sc.d;
}

template <class G>
struct GateUpA8iSmem {
    using S = Shape<G, false>;
    float2 scaled[256];
    alignas(16) std::uint8_t ring[kA8Warps][kA8Depth][kA8StageUnits * S::kUnit];
    std::uint16_t h[kPass][32]; // the pass's columns' BF16 h of this CTA's 32 intermediates (one slice)
};

template <class G>
__global__ void __launch_bounds__(kA8Threads) gate_up_a8i_kernel(Args a) {
    using S = Shape<G, false>;
    static_assert(G::kHidden + G::kHidden / 8 == kA8XRowBytes && G::kInter + G::kInter / 8 == kA8HRowBytes, "A8 rows");
    constexpr int kStages = S::kGateUpBlocks / kA8StageUnits; // 18
    constexpr int kStageBytes = kA8StageUnits * S::kUnit;
    static_assert(S::kGateUpBlocks % kA8StageUnits == 0 && kStages >= kA8Depth && S::kGateUpRowGroups % kA8Warps == 0);
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm = *reinterpret_cast<GateUpA8iSmem<G>*>(smem_raw);
    pdl::trigger_dependents(); // the down pass may launch and copy its weights; it reads h after its job's count
    const int job = static_cast<int>(blockIdx.y);
    JobView v;
    if (!job_view(a, job, v)) { return; }
    const int tid = static_cast<int>(threadIdx.x), lane = tid % 32, warp = tid / 32;
    const int g = lane / 4, t = lane % 4;
    const int group = static_cast<int>(blockIdx.x) * kA8Warps + warp;
    const std::uint8_t* units = v.record + static_cast<std::size_t>(group) * S::kGateUpBlocks * S::kUnit;
    auto& ring       = sm.ring[warp];
    const auto issue = [&](int stage) {
        const std::uint8_t* src = units + static_cast<std::size_t>(stage) * kStageBytes;
        for (int i = lane; i < kStageBytes / 16; i += 32) { tile::cp_async16(ring[stage % kA8Depth] + 16 * i, src + 16 * i, 16); }
    };
    fill_a8_scaled(sm.scaled, kA8Threads);
    const float m = a16_multiplier(a.bank.amax[2 * v.expert], a.bank.denominators[0]);
    const auto column_of_entry = [&](int c) { return a.ws.entries[v.first + c] / kTopK; };
    for (int base = 0; base < v.count; base += kPass) {
        const int n = min(kPass, v.count - base);
#pragma unroll
        for (int k = 0; k < kA8Depth - 1; ++k) {
            issue(k);
            tile::cp_async_commit();
        }
        // The pass's columns' A8 rows through L1 (written by kernels that finished before this one; a missing column
        // repeats the last: its outputs are not stored): column g's codes, columns 2t and 2t + 1's slice scales.
        const auto x_row = [&](int c) { return a.x.rows + static_cast<std::size_t>(column_of_entry(base + min(c, n - 1))) * kA8XRowBytes; };
        const std::uint8_t* xg = x_row(g);
        const float* d0s = reinterpret_cast<const float*>(x_row(2 * t) + G::kHidden);
        const float* d1s = reinterpret_cast<const float*>(x_row(2 * t + 1) + G::kHidden);
        float c[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        for (int st = 0; st < kStages; ++st) {
            tile::cp_async_wait<kA8Depth - 2>();
            if (st == 0) {
                __syncthreads(); // the scale table
            } else {
                __syncwarp();
            }
            if (st + kA8Depth - 1 < kStages) { issue(st + kA8Depth - 1); }
            tile::cp_async_commit();
            const std::uint8_t* stage = ring[st % kA8Depth];
#pragma unroll
            for (int u = 0; u < kA8StageUnits; ++u) {
                const int b = st * kA8StageUnits + u;
                a8_unit_terms(stage + u * S::kUnit, __ldg(reinterpret_cast<const unsigned*>(xg + 16 * b + 4 * t)),
                              __ldg(d0s + b / 2), __ldg(d1s + b / 2), sm.scaled, g, t, c);
            }
            __syncwarp();
        }
        tile::cp_async_wait<0>();
        // Rows g (even: gate) and g + 1 (up) of an intermediate pair through a shuffle; h = the softcapped SwiGLU, staged
        // per column (the CTA's 32 intermediates are slice blockIdx.x of h), then each column's slice encoded here.
        std::uint16_t y[4], up[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { y[i] = offloaded_moe::nvfp4_a8::row_output(c[i], m); }
#pragma unroll
        for (int i = 0; i < 4; ++i) { up[i] = static_cast<std::uint16_t>(__shfl_down_sync(0xFFFFFFFFU, y[i], 4)); }
        if ((g & 1) == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int col = 2 * t + (i & 1);
                if (col < n) {
                    const int entry = a.ws.entries[v.first + base + col];
                    sm.h[col][warp * 8 + g / 2 + (i < 2 ? 0 : 4)] =
                        swiglu(y[i], up[i], a.routing.gamma[entry], a.bank.gate_softcap, a.bank.up_softcap);
                }
            }
        }
        __syncthreads();
        if (tid < n) { // a thread a column: its slice of h (the single launch's bits)
            const int entry = a.ws.entries[v.first + base + tid];
            a8_encode_slice(sm.h[tid], a.ws.h_a8 + static_cast<std::size_t>(entry) * kA8HRowBytes, G::kInter,
                            static_cast<int>(blockIdx.x));
        }
        __syncthreads(); // the next pass reuses x, the rings and h
    }
    // The job's count: the down CTAs wait for every gate/up CTA's slices.
    if (tid == 0) {
        __threadfence();
        release_add(&a.ws.job_done[job]);
    }
}

template <class G>
struct DownA8iSmem {
    using S = Shape<G, false>;
    float2 scaled[256];
    alignas(16) std::uint8_t units[kA8DownWarps][S::kDownBytes];
    alignas(16) std::uint8_t h[kPass][kA8HRowBytes];
};

template <class G>
__global__ void __launch_bounds__(kA8DownThreads) down_a8i_kernel(Args a) {
    using S = Shape<G, false>;
    static_assert(S::kDownRowGroups % kA8DownWarps == 0, "whole row groups a CTA");
    static_assert(sizeof(DownA8iSmem<G>) <= 99 * 1024, "a down CTA's shared memory fits an SM");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm = *reinterpret_cast<DownA8iSmem<G>*>(smem_raw);
    pdl::trigger_dependents();
    const int job = static_cast<int>(blockIdx.y);
    JobView v;
    if (!job_view(a, job, v)) { return; }
    const int tid = static_cast<int>(threadIdx.x), lane = tid % 32, warp = tid / 32;
    const int g = lane / 4, t = lane % 4;
    const int group = static_cast<int>(blockIdx.x) * kA8DownWarps + warp;
    // The weights do not depend on h: their copies start before the wait.
    const std::uint8_t* src = v.record + S::kGateUpBytes + static_cast<std::size_t>(group) * S::kDownBytes;
    for (int i = lane; i < S::kDownBytes / 16; i += 32) { tile::cp_async16(sm.units[warp] + 16 * i, src + 16 * i, 16); }
    tile::cp_async_commit();
    fill_a8_scaled(sm.scaled, kA8DownThreads);
    __shared__ int stalled;
    if (tid == 0) {
        stalled = wait_count(a.ws.job_done + job, static_cast<std::uint32_t>(S::kGateUpRowGroups / kA8Warps), a.ws.error)
                      ? 0
                      : 1;
    }
    tile::cp_async_wait<0>();
    __syncthreads();
    if (stalled != 0) { return; }
    const float m = a16_multiplier(a.bank.amax[2 * v.expert + 1], a.bank.denominators[1]);
    const std::uint8_t* units = sm.units[warp];
    for (int base = 0; base < v.count; base += kPass) {
        const int n = min(kPass, v.count - base);
        for (int i = tid; i < kPass * (kA8HRowBytes / 16); i += kA8DownThreads) {
            const int c = i / (kA8HRowBytes / 16), k = i % (kA8HRowBytes / 16);
            const std::uint8_t* row =
                a.ws.h_a8 + static_cast<std::size_t>(a.ws.entries[v.first + base + min(c, n - 1)]) * kA8HRowBytes;
            tile::cp_async16(sm.h[c] + 16 * k, row + 16 * k, 16);
        }
        tile::cp_async_commit();
        tile::cp_async_wait<0>();
        __syncthreads();
        const std::uint8_t* hg = sm.h[g];
        const float* d0s = reinterpret_cast<const float*>(sm.h[2 * t] + G::kInter);
        const float* d1s = reinterpret_cast<const float*>(sm.h[2 * t + 1] + G::kInter);
        float c[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll 8
        for (int b = 0; b < S::kDownBlocks; ++b) {
            a8_unit_terms(units + b * S::kUnit, *reinterpret_cast<const unsigned*>(hg + 16 * b + 4 * t), d0s[b / 2],
                          d1s[b / 2], sm.scaled, g, t, c);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = 2 * t + (i & 1);
            if (col < n) {
                const int entry = a.ws.entries[v.first + base + col];
                const int row   = group * 16 + g + (i < 2 ? 0 : 8);
                a.outputs[static_cast<std::size_t>(entry) * G::kHidden + row] = offloaded_moe::nvfp4_a8::row_output(c[i], m);
            }
        }
        __syncthreads(); // the next pass's h rows
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

// ---------------------------------------------------------------------------- MXFP8 A16 (--a16-activations)

// W8A16: one mxfp8_expert_rg16_v1 unit times column g's BF16 values at the unit's 32-block (`col`: the block's first
// element). Lane (g, t)'s codes (rows g and g + 8, elements 4t..4t+3 and 16+4t..+3) widen exactly to BF16 as two
// m16n8k16 A fragments (k-step 0: elements 4t..4t+3 in slots 2t, 2t+1, 2t+8, 2t+9; k-step 1: 16 more), the column's
// values in the same slots; the block's two MMAs run from zero, then acc = fma(block, 2^(s_row - 127), acc) (a power of
// two: the product is exact). L: the column's load (non-coherent for x, L2 for the h another grid writes).
__device__ __forceinline__ float ue8m0_value(std::uint32_t e) {
    return e == 0 ? canon::f32_from_bits(0x00400000U) : e == 255 ? canon::f32_from_bits(0x7FC00000U) : canon::f32_from_bits(e << 23);
}

template <a16::LimbLoad L>
__device__ __forceinline__ void mx_a16_unit(const std::uint8_t* unit, const std::uint16_t* col, int lane, int g, int t,
                                            float (&c)[4]) {
    const uint4 w = *reinterpret_cast<const uint4*>(unit + 16 * lane);
    uint2 x0, x1;
    if constexpr (L == a16::LimbLoad::Cg) {
        x0 = __ldcg(reinterpret_cast<const uint2*>(col + 4 * t));
        x1 = __ldcg(reinterpret_cast<const uint2*>(col + 16 + 4 * t));
    } else {
        x0 = __ldg(reinterpret_cast<const uint2*>(col + 4 * t));
        x1 = __ldg(reinterpret_cast<const uint2*>(col + 16 + 4 * t));
    }
    const auto bf = [](unsigned pair) { return ::infernix::ops::detail::fp8_e4m3x2_to_bf16x2_bits(pair); };
    float d[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    mma_bf16(d[0], d[1], d[2], d[3], bf(w.x), bf(w.y), bf(w.x >> 16), bf(w.y >> 16), x0.x, x0.y);
    mma_bf16(d[0], d[1], d[2], d[3], bf(w.z), bf(w.w), bf(w.z >> 16), bf(w.w >> 16), x1.x, x1.y);
    const float s_lo = ue8m0_value(unit[512 + g]), s_hi = ue8m0_value(unit[520 + g]);
    c[0] = __fmaf_rn(d[0], s_lo, c[0]);
    c[1] = __fmaf_rn(d[1], s_lo, c[1]);
    c[2] = __fmaf_rn(d[2], s_hi, c[2]);
    c[3] = __fmaf_rn(d[3], s_hi, c[3]);
}

template <class G>
struct GateUpMxA16Smem {
    using S = Shape<G, true>;
    alignas(16) std::uint8_t ring[kSplit * kWarps][kDepth][S::kStageBytes];
    std::uint16_t h[kPass][kCtaInter];
};

// gate_up_a8_kernel's shape over BF16 x (two K-half chains, half 0 + half 1); h stays BF16 in the workspace.
template <class G>
__global__ void __launch_bounds__(kGuThreads) gate_up_mx_a16_kernel(Args a) {
    using S = Shape<G, true>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm = *reinterpret_cast<GateUpMxA16Smem<G>*>(smem_raw);
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
        const int b0 = k_half * S::kHalfBlocks;
        const std::uint16_t* xg =
            a.x.x16 + static_cast<std::size_t>(a.ws.entries[v.first + base + min(g, n - 1)] / kTopK) * G::kHidden;
        float c[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        for (int st = 0; st < S::kStages; ++st) {
            tile::cp_async_wait<kDepth - 2>();
            __syncwarp();
            if (st + kDepth - 1 < S::kStages) { issue(st + kDepth - 1); }
            tile::cp_async_commit();
            const std::uint8_t* stage = ring[st % kDepth];
#pragma unroll
            for (int u = 0; u < S::kStageUnits; ++u) {
                const int b = b0 + st * S::kStageUnits + u;
                mx_a16_unit<a16::LimbLoad::Nc>(stage + u * S::kUnit, xg + 32 * b, lane, g, t, c);
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
        for (int i = tid; i < n * kCtaInter; i += kGuThreads) {
            const int col = i / kCtaInter, k = i % kCtaInter;
            const int entry = a.ws.entries[v.first + base + col];
            a.ws.h[static_cast<std::size_t>(entry) * G::kInter + h_first + k] = sm.h[col][k];
        }
        __syncthreads(); // the next pass reuses the rings and h
    }
    if (tid == 0) {
        __threadfence();
        release_add(&a.ws.job_done[job]);
    }
}

template <class G>
struct DownMxA16Smem {
    using S = Shape<G, true>;
    alignas(16) std::uint8_t units[S::kDownWarps][S::kDownBytes];
};

// down_a8_kernel's shape over the job's BF16 h rows (one FP32 chain in block order).
template <class G>
__global__ void __launch_bounds__(Shape<G, true>::kDownThreads) down_mx_a16_kernel(Args a) {
    using S = Shape<G, true>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm = *reinterpret_cast<DownMxA16Smem<G>*>(smem_raw);
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
    tile::cp_async_wait<0>();
    __syncthreads();
    if (stalled != 0) { return; }
    for (int base = 0; base < v.count; base += kPass) {
        const int n = min(kPass, v.count - base);
        const std::uint16_t* hg =
            a.ws.h + static_cast<std::size_t>(a.ws.entries[v.first + base + min(g, n - 1)]) * G::kInter;
        float c[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll 4
        for (int b = 0; b < S::kDownBlocks; ++b) {
            mx_a16_unit<a16::LimbLoad::Cg>(sm.units[warp] + b * S::kUnit, hg + 32 * b, lane, g, t, c);
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
    }
}

// ---------------------------------------------------------------------------- one column in one launch

// moe_single: a single column's router and experts as one persistent grid of kSingleCtas CTAs, one an SM, so every CTA
// is resident and the grid's waits complete. CTAs 0-15 compute the router's logits (route_kernel's rows: a warp a row,
// its order); every CTA waits for them and selects the column's experts itself (select_column's rule, a thread an
// expert); CTA c then takes job
// c / 18 (jobs in ascending expert id) and its warps the row groups (c % 18) * 8 + warp, each streaming its gate/up row
// group over all of K (NVFP4: the core of gate_up_a4_kernel, native or exact; MXFP8: the two halves' FP32 chains
// added half 0 + half 1, as gate_up_a8_kernel) and then its down row group through one cp.async ring, so the down
// weights stream while the CTA waits for its job's h. NVFP4: h goes to global as BF16; once the job's 144 row groups
// are counted, each of its 18 CTAs loads the job's h row once (its amax from the same loads) and forms its
// four-over-six blocks itself. MXFP8: each CTA quantizes its 64 intermediates (two whole groups) and publishes them to
// the job's h row, which every down CTA loads. Every value is the separate kernels' bit for bit (with the same core).
// The last CTA to finish resets the counters.
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

// A16 (NVFP4 only): x and h as limb rows, with their exponents. I8 (NVFP4 only, ExpertActivation::A8): x and h as A8
// rows.
template <class G, bool A8, bool A16 = false, bool I8 = false>
struct SingleSmem {
    using S = Shape<G, A8>;
    static_assert(!(A8 && A16) && !(A8 && I8) && !(A16 && I8), "A16 limb rows and A8 rows are the NVFP4 layers'");
    alignas(16) std::uint8_t ring[kSingleWarps][kSingleDepth][S::kStageBytes];
    alignas(16) std::uint8_t x[A16 ? kA16XRowBytes : I8 ? kA8XRowBytes : S::kXStride];
    alignas(16) std::uint8_t h[A16 ? kA16HRowBytes : I8 ? kA8HRowBytes : S::kHStride];
    int x_emax;
    int h_emax;
    std::int32_t scaled[256];
    float2 a8_scaled[I8 ? 256 : 1]; // I8: {S, -1.5 * 2^23 * S} of every E4M3 scale word
    alignas(16) float keys[kSingleExperts];
    float sel[kTopK];
    std::int32_t routed[kTopK];
    float gamma[kTopK];
    std::int32_t job_slot[kTopK];
    std::uint32_t red[kSingleWarps];
    float h_amax;
    int stalled;
};

// Native (NVFP4 A4): the native block-scaled core (a4_native_step) in gate_up_a4_kernel's and down_a4_kernel's chains:
// gate/up two FP32 chains over the K halves added half 0 + half 1, down one chain in block order; x and h staged in
// the core's slot order. Not Native (Nvfp4Core::Exact): the exact int64 core. I8 (NVFP4 A8): gate_up_a8i_kernel's and
// down_a8i_kernel's single chains (a8_unit_terms in block order); each CTA encodes its 64 intermediates (two slices)
// and publishes them to the job's A8 h row, which every down CTA loads (as MXFP8's groups).
template <class G, bool A8, bool A16 = false, bool Native = false, bool I8 = false>
__global__ void __launch_bounds__(kSingleThreads, 1) single_kernel(const SingleArgs a) {
    static_assert(!Native || (!A8 && !A16 && !I8), "the native core is NVFP4 A4's");
    using S = Shape<G, A8>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm      = *reinterpret_cast<SingleSmem<G, A8, A16, I8>*>(smem_raw);
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
    if constexpr (I8) {
        fill_a8_scaled(sm.a8_scaled, kSingleThreads);
    } else if constexpr (!A8) {
        a4_mma_fill_scaled(sm.scaled, kSingleThreads);
    }
    // Dependents launch once the gate/up stream is done (below): launched beside this grid they would pack onto the
    // SMs it leaves free and run there (pdl::enter's note).
    pdl::wait_for_dependencies();
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
    constexpr int kXBytes = A16 ? kA16XRowBytes : I8 ? kA8XRowBytes : S::kXRowBytes;
    for (int i = tid; i < kXBytes / 8; i += kSingleThreads) {
        uint2 v = reinterpret_cast<const uint2*>(a.e.x.rows)[i];
        if constexpr (Native) {
            if (i < S::kXScales / 8) { v = make_uint2(a4_native_order(v.x), a4_native_order(v.y)); }
        }
        reinterpret_cast<uint2*>(sm.x)[i] = v;
    }
    if constexpr (A16) {
        if (tid == 0) { sm.x_emax = a.e.x.emax[0]; }
    }
    if (tid == 0) { sm.stalled = wait_count(counters + 0, kRouterCtas, a.e.ws.error) ? 0 : 1; }
    __syncthreads();
    if (sm.stalled != 0) { return; }
    // The column's selection (select_column's rule, a thread an expert): keys selection_key(sigmoid_c(l_e), bias_e), the top 8 by
    // key (ties to the lower index) in slots of descending key, gamma_j = (s_j / sum8(s)) * route_scale.
    static_assert(kSingleExperts <= kSingleThreads, "a thread an expert");
    float sig = 0.0F;
    if (tid < kSingleExperts) {
        sig          = canon::sigmoid_c(__ldcg(a.e.ws.logits + tid));
        sm.keys[tid] = detail::selection_key(sig, a.bias[tid]);
    }
    __syncthreads();
    if (tid < kSingleExperts) {
        // Four keys a shared load, every load issued before the compares (a loop of dependent loads was ~2 us).
        const float key = sm.keys[tid];
        int rank        = 0;
#pragma unroll
        for (int f = 0; f < kSingleExperts; f += 4) {
            const float4 o = *reinterpret_cast<const float4*>(sm.keys + f);
            rank += (o.x > key || (o.x == key && f < tid)) ? 1 : 0;
            rank += (o.y > key || (o.y == key && f + 1 < tid)) ? 1 : 0;
            rank += (o.z > key || (o.z == key && f + 2 < tid)) ? 1 : 0;
            rank += (o.w > key || (o.w == key && f + 3 < tid)) ? 1 : 0;
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
            if constexpr (Native) {
                static_assert(S::kStageUnits % 4 == 0, "whole 64-element steps a stage");
                if (u % 4 == 0) {
                    a4_native_step(stage + u * S::kUnit, b_row + 8 * b,
                                   *reinterpret_cast<const std::uint32_t*>(b_row + scales + b), g, t, acc);
                }
            } else if constexpr (I8) {
                // The column's codes at K 16b + 4t .. + 3 for every lane (one column), its slice scale for both.
                (void)scales;
                const int k     = down ? G::kInter : G::kHidden;
                const float d   = *reinterpret_cast<const float*>(b_row + k + 4 * (b / 2));
                a8_unit_terms(stage + u * S::kUnit, *reinterpret_cast<const unsigned*>(b_row + 16 * b + 4 * t), d, d,
                              sm.a8_scaled, g, t, acc);
            } else if constexpr (A8) {
                a8_mma_unit(stage + u * S::kUnit, b_row + 32 * b, b_row[scales + b], lane, g, t, acc);
            } else if constexpr (A16) {
                (void)scales;
                a16::a16_unit<a16::LimbLoad::Shared>(stage + u * S::kUnit, b_row + a16::a16_block_offset(b, t), sm.scaled,
                                                     g, t, s);
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
    if constexpr (I8) { // one chain over all of K
        for (int st = 0; st < kGu; ++st) { step(st, c0, s); }
    } else {
        for (int st = 0; st < kGu / 2; ++st) { step(st, c0, s); }
        for (int st = kGu / 2; st < kGu; ++st) { step(st, c1, s); }
    }
    std::uint16_t y[4], up[4];
    if constexpr (I8) {
        const float m = a16_multiplier(a.e.bank.amax[2 * expert], a.e.bank.denominators[0]);
#pragma unroll
        for (int i = 0; i < 4; ++i) { y[i] = offloaded_moe::nvfp4_a8::row_output(c0[i], m); }
    } else if constexpr (A8) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { y[i] = canon::f32_to_bf16_rn(canon::add_rn(c0[i], c1[i])); }
    } else if constexpr (Native) {
        const float rs = detail::nvfp4_row_scale(a.e.x.amax[0], a.e.bank.amax[2 * expert],
                                                 detail::nvfp4_rcp(a.e.bank.denominators[0]));
#pragma unroll
        for (int i = 0; i < 4; ++i) { y[i] = canon::f32_to_bf16_rn(canon::mul_rn(canon::add_rn(c0[i], c1[i]), rs)); }
    } else if constexpr (A16) {
        const float m = a16_multiplier(a.e.bank.amax[2 * expert], a.e.bank.denominators[0]);
#pragma unroll
        for (int i = 0; i < 4; ++i) { y[i] = canon::a16_row_output(s[i], sm.x_emax, m); }
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
    if constexpr (I8) {
        // A8: the CTA's 64 intermediates are two whole slices: encoded here (the codec the separate kernels' last CTA
        // applies, the same values) and published to the job's A8 h row, which every down CTA loads.
        __shared__ std::uint16_t cta_h[2 * 32];
        if (t == 0 && (g & 1) == 0) {
            cta_h[warp * 8 + g / 2]     = swiglu(y[0], up[0], gamma, a.e.bank.gate_softcap, a.e.bank.up_softcap);
            cta_h[warp * 8 + g / 2 + 4] = swiglu(y[2], up[2], gamma, a.e.bank.gate_softcap, a.e.bank.up_softcap);
        }
        __syncthreads();
        if (tid < 2) {
            a8_encode_slice(cta_h + 32 * tid, a.e.ws.h_a8 + static_cast<std::size_t>(slot) * kA8HRowBytes, G::kInter,
                            2 * (cta % kSingleCtasPerJob) + tid);
            __threadfence();
        }
        __syncthreads();
        if (tid == 0) { release_add(counters + 2 + job, kSingleWarps); } // the CTA's 8 row groups
    } else if constexpr (A8) {
        // MXFP8: the CTA's 8 row groups are 64 consecutive intermediates, two whole groups: quantized here
        // (gate_up_a8_kernel's rule, the same values) and published to the job's h row (codes at 32 a group, the UE8M0
        // words after the 1,152 codes), so each down CTA loads them instead of forming all 36 groups itself.
        __shared__ std::uint16_t cta_h[2 * 32];
        if (t == 0 && (g & 1) == 0) {
            cta_h[warp * 8 + g / 2]     = swiglu(y[0], up[0], gamma, a.e.bank.gate_softcap, a.e.bank.up_softcap);
            cta_h[warp * 8 + g / 2 + 4] = swiglu(y[2], up[2], gamma, a.e.bank.gate_softcap, a.e.bank.up_softcap);
        }
        __syncthreads();
        if (tid < 2) {
            float values[32];
#pragma unroll
            for (int j = 0; j < 32; ++j) { values[j] = bf16f(cta_h[32 * tid + j]); }
            const canon::Mxfp8Group q = canon::mxfp8_group(values);
            const int gi              = 2 * (cta % kSingleCtasPerJob) + tid;
            std::uint8_t* out         = a.e.ws.h_rows + static_cast<std::size_t>(slot) * S::kHRowBytes;
            auto* words               = reinterpret_cast<std::uint32_t*>(out + 32 * gi);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                words[j] = q.code[4 * j] | (static_cast<std::uint32_t>(q.code[4 * j + 1]) << 8) |
                           (static_cast<std::uint32_t>(q.code[4 * j + 2]) << 16) |
                           (static_cast<std::uint32_t>(q.code[4 * j + 3]) << 24);
            }
            out[S::kHScales + gi] = q.exponent;
            __threadfence();
        }
        __syncthreads();
        if (tid == 0) { release_add(counters + 2 + job, kSingleWarps); } // the CTA's 8 row groups
    } else {
        if (t == 0 && (g & 1) == 0) {
            h_row[group * 8 + g / 2]     = swiglu(y[0], up[0], gamma, a.e.bank.gate_softcap, a.e.bank.up_softcap);
            h_row[group * 8 + g / 2 + 4] = swiglu(y[2], up[2], gamma, a.e.bank.gate_softcap, a.e.bank.up_softcap);
            __threadfence();
        }
        __syncwarp();
        if (lane == 0) { release_add(counters + 2 + job); }
    }
    pdl::trigger_dependents();
    // The job's h, once its 144 row groups are counted (the down weights keep streaming meanwhile).
    __syncthreads();
    if (tid == 0) { sm.stalled = wait_count(counters + 2 + job, kSingleRowGroups, a.e.ws.error) ? 0 : 1; }
    __syncthreads();
    if (sm.stalled != 0) {
        tile::cp_async_wait<0>();
        return;
    }
    if constexpr (I8) {
        // A8: the job's A8 h row as its 18 CTAs published it (codes, then the FP32 slice scales).
        (void)h_row;
        static_assert(kA8HRowBytes % 4 == 0, "the h row in words");
        const auto* src = reinterpret_cast<const std::uint32_t*>(a.e.ws.h_a8 + static_cast<std::size_t>(slot) * kA8HRowBytes);
        for (int i = tid; i < kA8HRowBytes / 4; i += kSingleThreads) { reinterpret_cast<std::uint32_t*>(sm.h)[i] = __ldcg(src + i); }
    } else if constexpr (A8) {
        // MXFP8: the job's h row as its 18 CTAs published it (codes and UE8M0 words, in sm.h's layout).
        (void)h_row;
        static_assert(S::kHRowBytes % 4 == 0 && S::kHRowBytes <= S::kHStride, "the h row in words");
        const auto* src = reinterpret_cast<const std::uint32_t*>(a.e.ws.h_rows + static_cast<std::size_t>(slot) * S::kHRowBytes);
        for (int i = tid; i < S::kHRowBytes / 4; i += kSingleThreads) { reinterpret_cast<std::uint32_t*>(sm.h)[i] = __ldcg(src + i); }
    } else {
        // NVFP4: the row's amax (the largest magnitude's bits), then four-over-six by block halves on thread pairs
        // (gate_up_a4_kernel's last-CTA quantization). A block half's 8 values are loaded once, by the thread that
        // quantizes them, and the amax is the maximum of the threads' 8-value maxima (the same values, the same max).
        constexpr int kBlocks = G::kInter / 16;
        static_assert(2 * kBlocks <= kSingleThreads, "one round of block halves");
        const int i    = tid;
        const bool on  = i < 2 * kBlocks;
        const int half = i & 1, b = on ? i >> 1 : 0;
        const uint4 hq = on ? __ldcg(reinterpret_cast<const uint4*>(h_row + 16 * b) + half) : make_uint4(0, 0, 0, 0);
        std::uint32_t m = 0;
        {
            const std::uint32_t hw[4] = {hq.x, hq.y, hq.z, hq.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                m = max(m, canon::f32_bits(bf16f(static_cast<std::uint16_t>(hw[j] & 0xFFFFU))) & 0x7FFFFFFFU);
                m = max(m, canon::f32_bits(bf16f(static_cast<std::uint16_t>(hw[j] >> 16))) & 0x7FFFFFFFU);
            }
        }
#pragma unroll
        for (int o = 16; o >= 1; o /= 2) { m = max(m, __shfl_xor_sync(0xFFFFFFFFU, m, o)); }
        if (lane == 0) { sm.red[warp] = m; }
        __syncthreads();
        if (tid == 0) {
            std::uint32_t all = 0;
#pragma unroll
            for (int wi = 0; wi < kSingleWarps; ++wi) { all = max(all, sm.red[wi]); }
            sm.h_amax = canon::f32_from_bits(all);
            if constexpr (A16) { sm.h_emax = a16::a16_emax(all >> 16); } // the BF16 bits of max |h|
        }
        __syncthreads();
        if constexpr (A16) {
            // A16: the row's limbs, a block's halves on a thread pair.
            if (on) {
                const uint4 q = hq;
                const std::uint32_t hw[4] = {q.x, q.y, q.z, q.w};
                std::uint16_t bits[8];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    bits[2 * j]     = static_cast<std::uint16_t>(hw[j] & 0xFFFFU);
                    bits[2 * j + 1] = static_cast<std::uint16_t>(hw[j] >> 16);
                }
                a16::store_a16_half(sm.h, b, half, bits, sm.h_emax);
            }
        } else {
            const canon::FourOverSixRow fos = canon::four_over_six_row(on ? sm.h_amax : 1.0F);
            float xv[8] = {};
            float mag   = 0.0F;
            if (on) {
                const uint4 q = hq;
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
                if constexpr (Native) { packed = a4_native_order(packed); }
                *reinterpret_cast<std::uint32_t*>(sm.h + 8 * b + 4 * half) = packed;
                if (half == 0) { sm.h[S::kHScales + b] = four ? words.four : words.six; }
            }
        }
    }
    __syncthreads();
    // Down: one chain over K = I (MXFP8 and native NVFP4 one FP32 chain in block order; exact NVFP4 the int64 sum).
    float cd[4]     = {0.0F, 0.0F, 0.0F, 0.0F};
    long long sd[4] = {0, 0, 0, 0};
    for (int st = kGu; st < kTotal; ++st) { step(st, cd, sd); }
    tile::cp_async_wait<0>();
    if (t == 0) {
        std::uint16_t lo, hi;
        if constexpr (I8) {
            const float m = a16_multiplier(a.e.bank.amax[2 * expert + 1], a.e.bank.denominators[1]);
            lo            = offloaded_moe::nvfp4_a8::row_output(cd[0], m);
            hi            = offloaded_moe::nvfp4_a8::row_output(cd[2], m);
        } else if constexpr (A8) {
            lo = canon::f32_to_bf16_rn(cd[0]);
            hi = canon::f32_to_bf16_rn(cd[2]);
        } else if constexpr (Native) {
            const float rs = detail::nvfp4_row_scale(sm.h_amax, a.e.bank.amax[2 * expert + 1],
                                                     detail::nvfp4_rcp(a.e.bank.denominators[1]));
            lo = canon::f32_to_bf16_rn(canon::mul_rn(cd[0], rs));
            hi = canon::f32_to_bf16_rn(canon::mul_rn(cd[2], rs));
        } else if constexpr (A16) {
            const float m = a16_multiplier(a.e.bank.amax[2 * expert + 1], a.e.bank.denominators[1]);
            lo            = canon::a16_row_output(sd[0], sm.h_emax, m);
            hi            = canon::a16_row_output(sd[2], sm.h_emax, m);
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
Workspace layout(std::int32_t experts, std::int32_t intermediate, std::int32_t capacity, ExpertActivation activation,
                 Take&& take) {
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
    if (activation == ExpertActivation::A16) { w.h_limbs = reinterpret_cast<std::uint8_t*>(take(m * (i / 64 * 192))); }
    if (activation == ExpertActivation::A8) { w.h_a8 = reinterpret_cast<std::uint8_t*>(take(m * (i + i / 8))); }
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

std::size_t workspace_bytes(std::int32_t experts, std::int32_t intermediate, std::int32_t capacity,
                            ExpertActivation activation) {
    std::size_t total = 0;
    (void)layout(experts, intermediate, capacity, activation, [&](std::size_t bytes) {
        total += align256(bytes);
        return static_cast<std::byte*>(nullptr);
    });
    return total;
}

Workspace carve_workspace(void* base, std::int32_t experts, std::int32_t intermediate, std::int32_t capacity,
                          ExpertActivation activation) {
    auto* p = static_cast<std::byte*>(base);
    return layout(experts, intermediate, capacity, activation, [&](std::size_t bytes) {
        std::byte* out = p;
        p += align256(bytes);
        return out;
    });
}

void moe_route(const std::uint16_t* x, std::int32_t columns, const std::uint16_t* router, const float* bias,
               std::int32_t experts, std::int32_t hidden, float route_scale, const Workspace& workspace,
               cudaStream_t stream, std::int32_t* record) {
    if (hidden != Inkling::kHidden || experts <= 0 || experts > 256 || columns <= 0 || columns > workspace.capacity) {
        throw std::invalid_argument("w4a4_sparse_moe router: H = 2304, at most 256 experts, 1..capacity columns");
    }
    const dim3 grid((experts + kRouteWarps - 1) / kRouteWarps, (columns + kRouteColumns - 1) / kRouteColumns);
    CUDA_CHECK(pdl::launch_with(pdl::Dependency::Programmatic, {grid, dim3(kRouteThreads), 0, stream}, route_kernel, x,
                                columns, router, bias, experts, route_scale, workspace, record));
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
// Sets the instance's shared-memory attributes (every launch of it follows a check here) and whether its grid fits.
template <bool A8, bool A16, bool Native = false, bool I8 = false>
bool single_fits() {
    using G = Inkling;
    allow_shared(single_kernel<G, A8, A16, Native, I8>, sizeof(SingleSmem<G, A8, A16, I8>));
    int device = 0, sms = 0, blocks = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, single_kernel<G, A8, A16, Native, I8>, kSingleThreads,
                                                             sizeof(SingleSmem<G, A8, A16, I8>)));
    return blocks >= 1 && static_cast<long long>(blocks) * sms >= kSingleCtas;
}
} // namespace

bool moe_single_supported(ExpertFormat format, ExpertActivation activation) {
    static const bool a4  = single_fits<false, false>() && single_fits<false, false, true>();
    static const bool a8  = single_fits<true, false>();
    static const bool a16 = single_fits<false, true>();
    static const bool i8  = single_fits<false, false, false, true>();
    if (format == ExpertFormat::Mxfp8) { return activation == ExpertActivation::Checkpoint && a8; }
    if (activation == ExpertActivation::A8) { return i8; }
    return activation == ExpertActivation::A16 ? a16 : a4;
}

void moe_single(const std::uint16_t* x, const Activations& rows, const std::uint16_t* router, const float* bias,
                float route_scale, const Bank& bank, const Workspace& workspace, std::uint16_t* outputs,
                cudaStream_t stream) {
    using G = Inkling;
    if (bank.hidden != G::kHidden || bank.intermediate != G::kInter || bank.experts != kSingleExperts ||
        rows.columns != 1 || workspace.capacity < 1) {
        throw std::invalid_argument("w4a4_sparse_moe single: one column of Inkling 2's [128, 2304, 1152] banks");
    }
    if (!moe_single_supported(bank.format, rows.activation)) {
        throw std::invalid_argument("w4a4_sparse_moe single: the device cannot hold its grid resident");
    }
    const bool a16 = rows.activation == ExpertActivation::A16;
    const bool i8  = rows.activation == ExpertActivation::A8;
    if (a16 && (rows.rows == nullptr || rows.emax == nullptr)) {
        throw std::invalid_argument("w4a4_sparse_moe single A16: the column's limb row and exponent");
    }
    if (i8 && (rows.rows == nullptr || bank.format != ExpertFormat::Nvfp4FourOverSix || workspace.h_a8 == nullptr)) {
        throw std::invalid_argument("w4a4_sparse_moe single A8: an NVFP4 layer, the column's A8 row and a workspace carved "
                                    "for A8");
    }
    SingleArgs args;
    args.e           = Args{rows, Routing{workspace.routed, workspace.gamma}, bank, workspace, outputs};
    args.x           = x;
    args.router      = router;
    args.bias        = bias;
    args.route_scale = route_scale;
    if (i8) {
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Programmatic,
                                    {dim3(kSingleCtas), dim3(kSingleThreads), sizeof(SingleSmem<G, false, false, true>),
                                     stream},
                                    single_kernel<G, false, false, false, true>, args));
    } else if (bank.format == ExpertFormat::Nvfp4FourOverSix && a16) {
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Programmatic,
                                    {dim3(kSingleCtas), dim3(kSingleThreads), sizeof(SingleSmem<G, false, true>), stream},
                                    single_kernel<G, false, true>, args));
    } else if (bank.format == ExpertFormat::Nvfp4FourOverSix) {
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Programmatic,
                                    {dim3(kSingleCtas), dim3(kSingleThreads), sizeof(SingleSmem<G, false>), stream},
                                    rows.core == Nvfp4Core::Exact ? single_kernel<G, false, false, false>
                                                                  : single_kernel<G, false, false, true>,
                                    args));
    } else {
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Programmatic,
                                    {dim3(kSingleCtas), dim3(kSingleThreads), sizeof(SingleSmem<G, true>), stream},
                                    single_kernel<G, true>, args));
    }
    check_launch("single");
}

void moe_clear_counters(const Workspace& workspace, cudaStream_t stream) {
    if (workspace.route_done == nullptr || workspace.single == nullptr) {
        throw std::invalid_argument("w4a4_sparse_moe clear: a carved workspace");
    }
    CUDA_CHECK(cudaMemsetAsync(workspace.route_done, 0, sizeof(std::uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(workspace.single, 0, kSingleCounters * sizeof(std::uint32_t), stream));
}

namespace testing {
void leave_stalled_single_counters(const Workspace& workspace, cudaStream_t stream) {
    std::array<std::uint32_t, kSingleCounters> counts{};
    counts[0] = kRouterCtas;
    counts[1] = static_cast<std::uint32_t>(kSingleCtas - kSingleCtasPerJob);
    counts[2] = static_cast<std::uint32_t>(kSingleRowGroups - 1);
    for (int job = 1; job < kTopK; ++job) { counts[2 + job] = static_cast<std::uint32_t>(kSingleRowGroups); }
    CUDA_CHECK(cudaMemcpyAsync(workspace.single, counts.data(), sizeof(counts), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
} // namespace testing

void moe_encode_a8(const std::uint16_t* x, std::int32_t columns, std::uint8_t* rows, cudaStream_t stream) {
    if (x == nullptr || rows == nullptr || columns <= 0 || reinterpret_cast<std::uintptr_t>(rows) % 16 != 0 ||
        ops::a8_linear_column_words(Inkling::kHidden, false) * 4 != kA8XRowBytes) {
        throw std::invalid_argument("w4a4_sparse_moe A8 encode: BF16 [T][2304] into 16-byte aligned A8 rows");
    }
    const Tensor in(const_cast<std::uint16_t*>(x), DType::BF16, {Inkling::kHidden, columns});
    Tensor out(rows, DType::I32, {kA8XRowBytes / 4, columns});
    ops::a8_encode_columns(in, out, stream);
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
    if (x.activation == ExpertActivation::A8) {
        if (bank.format != ExpertFormat::Nvfp4FourOverSix || x.rows == nullptr || workspace.h_a8 == nullptr) {
            throw std::invalid_argument("w4a4_sparse_moe A8: NVFP4 layers (MXFP8 keep their 8-bit arithmetic), the "
                                        "columns' A8 rows and a workspace carved for A8");
        }
        using S = Shape<G, false>;
        static const bool configured = [] {
            allow_shared(gate_up_a8i_kernel<G>, sizeof(GateUpA8iSmem<G>));
            allow_shared(down_a8i_kernel<G>, sizeof(DownA8iSmem<G>));
            return true;
        }();
        (void)configured;
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Serialized,
                                    {dim3(S::kGateUpRowGroups / kA8Warps, jobs), dim3(kA8Threads), sizeof(GateUpA8iSmem<G>),
                                     stream},
                                    gate_up_a8i_kernel<G>, args));
        check_launch("NVFP4 A8 gate/up");
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Programmatic,
                                    {dim3(S::kDownRowGroups / kA8DownWarps, jobs), dim3(kA8DownThreads), sizeof(DownA8iSmem<G>),
                                     stream},
                                    down_a8i_kernel<G>, args));
        check_launch("NVFP4 A8 down");
        return;
    }
    if (x.activation == ExpertActivation::A16 && bank.format == ExpertFormat::Mxfp8) {
        if (x.x16 == nullptr) { throw std::invalid_argument("w4a4_sparse_moe MXFP8 A16: the columns' BF16 rows"); }
        using S = Shape<G, true>;
        static const bool configured = [] {
            allow_shared(gate_up_mx_a16_kernel<G>, sizeof(GateUpMxA16Smem<G>));
            allow_shared(down_mx_a16_kernel<G>, sizeof(DownMxA16Smem<G>));
            return true;
        }();
        (void)configured;
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Serialized,
                                    {dim3(S::kGateUpCtas, jobs), dim3(kGuThreads), sizeof(GateUpMxA16Smem<G>), stream},
                                    gate_up_mx_a16_kernel<G>, args));
        check_launch("MXFP8 A16 gate/up");
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Programmatic,
                                    {dim3(S::kDownCtas, jobs), dim3(S::kDownThreads), sizeof(DownMxA16Smem<G>), stream},
                                    down_mx_a16_kernel<G>, args));
        check_launch("MXFP8 A16 down");
        return;
    }
    if (x.activation == ExpertActivation::A16) {
        if (x.rows == nullptr || x.emax == nullptr || workspace.h_limbs == nullptr) {
            throw std::invalid_argument("w4a4_sparse_moe A16: x limb rows and exponents, and a workspace carved for A16");
        }
        using S = Shape<G, false>;
        static const bool configured = [] {
            allow_shared(gate_up_a16_kernel<G>, sizeof(GateUpA16Smem<G>));
            allow_shared(down_a16_kernel<G>, sizeof(DownA16Smem<G>));
            return true;
        }();
        (void)configured;
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Serialized,
                                    {dim3(S::kGateUpCtas, jobs), dim3(kGuThreads), sizeof(GateUpA16Smem<G>), stream},
                                    gate_up_a16_kernel<G>, args));
        check_launch("NVFP4 A16 gate/up");
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Programmatic,
                                    {dim3(S::kDownCtas, jobs), dim3(S::kDownThreads), sizeof(DownA16Smem<G>), stream},
                                    down_a16_kernel<G>, args));
        check_launch("NVFP4 A16 down");
        return;
    }
    if (bank.format == ExpertFormat::Nvfp4FourOverSix) {
        using S = Shape<G, false>;
        static const bool configured = [] {
            allow_shared(gate_up_a4_kernel<G, false>, sizeof(GateUpA4Smem<G>));
            allow_shared(down_a4_kernel<G, false>, sizeof(DownA4Smem<G>));
            allow_shared(gate_up_a4_kernel<G, true>, sizeof(GateUpA4Smem<G>));
            allow_shared(down_a4_kernel<G, true>, sizeof(DownA4Smem<G>));
            return true;
        }();
        (void)configured;
        const bool exact = x.core == Nvfp4Core::Exact;
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Serialized,
                                    {dim3(S::kGateUpCtas, jobs), dim3(kGuThreads), sizeof(GateUpA4Smem<G>), stream},
                                    exact ? gate_up_a4_kernel<G, false> : gate_up_a4_kernel<G, true>, args));
        check_launch("NVFP4 gate/up");
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Programmatic,
                                    {dim3(S::kDownCtas, jobs), dim3(S::kDownThreads), sizeof(DownA4Smem<G>), stream},
                                    exact ? down_a4_kernel<G, false> : down_a4_kernel<G, true>, args));
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

namespace {
// moe_prefetch_l2's kernel: thread 0 gathers the distinct experts, then every thread issues 16 KB pieces of their
// records' first `bytes` (one piece a thread for 8 NVFP4 experts' gate/up: 8 x 183 pieces).
constexpr int kPrefetchThreads = 128, kPrefetchMax = 16, kPrefetchCtas = 12;
__global__ void __launch_bounds__(kPrefetchThreads) prefetch_l2_kernel(const std::int32_t* experts, int count,
                                                                         const std::uint8_t* records, std::uint64_t stride,
                                                                         std::uint64_t bytes, int bank_experts) {
    __shared__ int ids[kPrefetchMax];
    __shared__ int distinct;
    if (threadIdx.x == 0) {
        int m = 0;
        for (int i = 0; i < count; ++i) {
            int e; // relaxed: a route's record may be rewritten concurrently (a stale hint, never undefined)
            asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(e) : "l"(experts + i) : "memory");
            if (e < 0 || e >= bank_experts) { continue; }
            bool seen = false;
            for (int j = 0; j < m; ++j) { seen = seen || ids[j] == e; }
            if (!seen) { ids[m++] = e; }
        }
        distinct = m;
    }
    __syncthreads();
    constexpr std::uint64_t kPiece = 16384;
    const std::uint64_t pieces     = (bytes + kPiece - 1) / kPiece;
    const std::uint64_t total      = pieces * static_cast<std::uint64_t>(distinct);
    for (std::uint64_t i = blockIdx.x * std::uint64_t{kPrefetchThreads} + threadIdx.x; i < total;
         i += std::uint64_t{gridDim.x} * kPrefetchThreads) {
        const std::uint64_t at = (i % pieces) * kPiece;
        const auto piece       = static_cast<unsigned>(bytes - at < kPiece ? bytes - at : kPiece);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(records + static_cast<std::uint64_t>(ids[i / pieces]) * stride + at),
                     "r"(piece)
                     : "memory");
    }
}
} // namespace

void moe_prefetch_l2(const std::int32_t* experts, std::int32_t count, const Bank& bank, std::uint64_t bytes,
                     cudaStream_t stream) {
    if (experts == nullptr || count <= 0 || count > kPrefetchMax || bytes == 0 || bytes % 16 != 0 ||
        bytes > bank.record_stride || bank.record_stride % 16 != 0) {
        throw std::invalid_argument("w4a4_sparse_moe prefetch: 1-16 experts, a 16-byte multiple of their records");
    }
    prefetch_l2_kernel<<<kPrefetchCtas, kPrefetchThreads, 0, stream>>>(experts, count, bank.records, bank.record_stride,
                                                                        bytes, bank.experts);
    check_launch("expert prefetch");
}

} // namespace infernix::ops::w4a4_moe
