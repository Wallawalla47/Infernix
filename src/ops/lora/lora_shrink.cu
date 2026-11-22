// lora_shrink (include/infernix/ops/lora.h): U = A x under the canonical chunk / group arithmetic.
//
// Three schedules, the same bits (the call takes the one modelled fastest for its shape):
// - Split (decode, verification and short prompts): a cluster of CTAs splits K by whole groups for one 16-row tile of
//   A and up to 32 columns. Each warp chains one 256-value chunk, every chunk of the rank at once (A's fragments
//   loaded before the PDL wait); the rank adds its groups' chunk partials in order and stores the group sums into
//   rank 0's shared memory, and rank 0 adds them in order after one cluster barrier. Many small CTAs stream A at full
//   bandwidth where a whole-K CTA would be limited by one SM's.
// - Stream (long prompts): one CTA takes every row of A over 32 columns and all of K, staging A and x through a
//   cp.async ring, and keeps the chunk, group and total sums in registers. x is read once. A stack too tall for one
//   CTA's ring (a wide adapter's) streams in slices of rows, one slice a CTA, in one launch.
// - Tile (prompts): a cluster of CTAs splits K by groups in rounds for one tile of up to 128 columns by 128 rows, fed
//   by TMA, the device's cluster capacity choosing the tile and the cluster.
// All add in the same order: chunk partials from +0 (MMA chain), groups from their first chunk, the total from its
// first group.

#include "infernix/ops/lora.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "core/tma_descriptor_staging.cuh"
#include "ops/common/mbarrier.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace detail {
namespace {

namespace cg = cooperative_groups;

constexpr int kChunk       = kLoraShrinkChunk;           // 256
constexpr int kGroup       = kLoraShrinkGroup;           // 4 chunks
constexpr int kGroupValues = kChunk * kGroup;            // 1024
constexpr int kTokenTile   = 32;                         // four n8 tiles
constexpr int kMaxRanks    = 8;                          // the portable cluster size
constexpr int kMaxGroupsPerRank = 3;                     // 17 groups (K = 17,408) over 8 ranks
constexpr int kThreads     = 128;

__device__ __forceinline__ unsigned load_pair(const __nv_bfloat16* p) { return *reinterpret_cast<const unsigned*>(p); }

__device__ __forceinline__ int group_begin(int rank, int ranks, int groups) { return rank * groups / ranks; }

// ---------------------------------------------------------------------------------------------------------------
// Split schedule. grid (ranks, rows / 16, ceil(T / (8 NT))), cluster (ranks, 1, 1); 4 G warps, G the most groups a
// rank takes: warp w chains chunk w % 4 of the rank's group w / 4, so a rank's every chunk runs at once. Latency is
// the cost at decode widths (one memory round trip, one chain, one barrier), so:
// - A is an immutable weight: each warp loads its chunk's A fragments before the PDL wait, overlapping the producer's
//   tail; x is the producer's output (not __restrict__: its loads stay after the wait).
// - NT n8 tiles of tokens a CTA (1 for T <= 8, 2 for T <= 16, else 4): no MMA on a tile past the call.
// - Each rank adds its groups' chunk partials in order and stores the group sums into rank 0's shared memory
//   (distributed shared memory); after one cluster barrier rank 0 adds every group sum in order from its own memory.
// Dynamic shared memory: part [G][4][NT][32] float4, then group_sum [groups][NT][32] float4 (read on rank 0).
template <int NT, int G>
__global__ __launch_bounds__(kThreads * G) void lora_shrink_split_kernel(const __nv_bfloat16* x,
                                                                         const __nv_bfloat16* __restrict__ a,
                                                                         __nv_bfloat16* u, int rows, int k, int tokens) {
    extern __shared__ __align__(16) unsigned char split_shared[];
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = static_cast<int>(cluster.block_rank()), ranks = static_cast<int>(cluster.num_blocks());
    const int groups = k / kGroupValues;
    const int g_begin = group_begin(rank, ranks, groups), g_end = group_begin(rank + 1, ranks, groups);
    const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
    const int g = lane >> 2, q = lane & 3;
    const int local_group = warp >> 2, chunk = warp & 3;
    const int group       = g_begin + local_group;
    const bool active     = group < g_end;
    const int row_lo = static_cast<int>(blockIdx.y) * 16 + g, row_hi = row_lo + 8;
    const int token_base = static_cast<int>(blockIdx.z) * 8 * NT;
    auto* part      = reinterpret_cast<float4*>(split_shared);   // [G][4][NT][32]
    auto* group_sum = part + G * 4 * NT * 32;                     // [groups][NT][32]
    const int k0    = ((active ? group : g_begin) * kGroup + chunk) * kChunk + 2 * q;

    // The chunk's A fragments: 16 k-steps of rows g and g + 8 (before the wait: immutable).
    unsigned af[kChunk / 16][4];
    {
        const __nv_bfloat16* a_lo = a + static_cast<std::int64_t>(row_lo) * k + k0;
        const __nv_bfloat16* a_hi = a + static_cast<std::int64_t>(row_hi) * k + k0;
#pragma unroll
        for (int step = 0; step < kChunk / 16; ++step) {
            af[step][0] = active ? load_pair(a_lo + step * 16) : 0U;
            af[step][1] = active ? load_pair(a_hi + step * 16) : 0U;
            af[step][2] = active ? load_pair(a_lo + step * 16 + 8) : 0U;
            af[step][3] = active ? load_pair(a_hi + step * 16 + 8) : 0U;
        }
    }
    pdl::enter();

    float c[NT][4] = {};
    if (active) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int token = token_base + nt * 8 + g;
            if (token_base + nt * 8 >= tokens) { break; } // warp-uniform: no tile past the call
            const bool live            = token < tokens;
            const __nv_bfloat16* x_col = x + static_cast<std::int64_t>(live ? token : 0) * k + k0;
            unsigned bf[kChunk / 16][2];
#pragma unroll
            for (int step = 0; step < kChunk / 16; ++step) {
                bf[step][0] = live ? load_pair(x_col + step * 16) : 0U;
                bf[step][1] = live ? load_pair(x_col + step * 16 + 8) : 0U;
            }
#pragma unroll
            for (int step = 0; step < kChunk / 16; ++step) {
                mma_bf16(c[nt][0], c[nt][1], c[nt][2], c[nt][3], af[step][0], af[step][1], af[step][2], af[step][3],
                         bf[step][0], bf[step][1]);
            }
        }
    }
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        part[((local_group * 4 + chunk) * NT + nt) * 32 + lane] = make_float4(c[nt][0], c[nt][1], c[nt][2], c[nt][3]);
    }
    __syncthreads();
    // Each element of each of the rank's groups: its four chunk partials in order, stored at rank 0.
    for (int i = static_cast<int>(threadIdx.x); i < G * NT * 32; i += kThreads * G) {
        const int lg = i / (NT * 32), nt = (i / 32) % NT, l = i % 32;
        if (g_begin + lg >= g_end) { continue; }
        float4 s = part[((lg * 4 + 0) * NT + nt) * 32 + l];
#pragma unroll
        for (int w = 1; w < kGroup; ++w) {
            const float4 p = part[((lg * 4 + w) * NT + nt) * 32 + l];
            s.x            = __fadd_rn(s.x, p.x);
            s.y            = __fadd_rn(s.y, p.y);
            s.z            = __fadd_rn(s.z, p.z);
            s.w            = __fadd_rn(s.w, p.w);
        }
        *cluster.map_shared_rank(group_sum + ((g_begin + lg) * NT + nt) * 32 + l, 0) = s;
    }
    cluster.sync(); // every group sum is in rank 0's memory; rank 0 then reads only its own
    if (rank != 0) { return; }
    for (int i = static_cast<int>(threadIdx.x); i < NT * 32; i += kThreads * G) {
        const int nt = i / 32, l = i % 32;
        float4 total = group_sum[nt * 32 + l];
        for (int gi = 1; gi < groups; ++gi) {
            const float4 s = group_sum[(gi * NT + nt) * 32 + l];
            total.x        = __fadd_rn(total.x, s.x);
            total.y        = __fadd_rn(total.y, s.y);
            total.z        = __fadd_rn(total.z, s.z);
            total.w        = __fadd_rn(total.w, s.w);
        }
        // Lane l of n tile nt holds rows g, g + 8 (g = l / 4) at tokens 2q, 2q + 1 of the n tile.
        const int rg = static_cast<int>(blockIdx.y) * 16 + (l >> 2), t0 = token_base + nt * 8 + 2 * (l & 3);
        const float values[4] = {total.x, total.y, total.z, total.w};
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int row = rg + (v >= 2 ? 8 : 0), token = t0 + (v & 1);
            if (token < tokens) { u[static_cast<std::int64_t>(token) * rows + row] = __float2bfloat16_rn(values[v]); }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Stream schedule (prompts). grid (ranks, ceil(T / 32), slices), cluster (ranks, 1, 1): rank r streams its own groups
// of K (group_begin) for every row of A (of its slice) over 32 columns, so x is read once in all (once a slice) and the
// call has ranks x T / 32 CTAs (a slice).
// 4 warps, warp w takes n tile w (8 columns) over every 16-row tile of A. A and x pass through a ring of Stages
// stages of kStageK values; rows are 128-byte lines (8 segments of 16 B) with the segment index swizzled by row % 8,
// so ldmatrix's eight row addresses hit distinct banks. Each rank keeps its groups' sums in shared memory; after a
// cluster barrier rank 0 adds every group sum in order (its own, then the other ranks', distributed shared memory) and
// a second barrier keeps the ranks until it has. Dynamic shared memory: the ring, then the group sums
// [local group][MTiles][4][32] float4.
constexpr int kStageK  = 64;
constexpr int kLineB   = kStageK * 2; // 128 bytes

__device__ __forceinline__ int swizzled(int row, int segment) { return row * kLineB + ((segment ^ (row & 7)) << 4); }

// The ring's depth: 3 stages to 96 rows, 2 above, so two CTAs share an SM where they can. A deeper ring (8 stages at 32
// rows, 5 at 96) measured slower from 1,024 columns (one CTA an SM: down at 2,048 columns 85.9 against 64.1 us,
// 2026-11-21) and no faster below.
template <int MTiles>
inline constexpr int kStreamStages = MTiles <= 6 ? 3 : 2;

template <int MTiles>
constexpr int stream_ring_bytes() { return kStreamStages<MTiles> * (MTiles * 16 + kTokenTile) * kLineB; }

template <int MTiles>
__global__ __launch_bounds__(kThreads) void lora_shrink_stream_kernel(const __nv_bfloat16* x,
                                                                      const __nv_bfloat16* __restrict__ a,
                                                                      __nv_bfloat16* u, int k, int tokens) {
    constexpr int kRows       = MTiles * 16;
    constexpr int kStages     = kStreamStages<MTiles>;
    constexpr int kStageBytes = (kRows + kTokenTile) * kLineB;
    extern __shared__ __align__(16) unsigned char shared[];
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = static_cast<int>(cluster.block_rank()), ranks = static_cast<int>(cluster.num_blocks());
    const int groups  = k / kGroupValues;
    const int g_begin = group_begin(rank, ranks, groups), g_end = group_begin(rank + 1, ranks, groups);
    const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
    const int token_base   = static_cast<int>(blockIdx.y) * kTokenTile;
    const int k_begin      = g_begin * kGroupValues;
    const int stages_total = (g_end - g_begin) * (kGroupValues / kStageK);
    auto* group_sums       = reinterpret_cast<float4*>(shared + kStages * kStageBytes);
    const auto sum_at      = [&](int local_group, int m) { return ((local_group * MTiles + m) * 4 + warp) * 32 + lane; };

    const auto stage_load = [&](int stage, int kt) {
        unsigned char* a_s = shared + stage * kStageBytes;
        unsigned char* x_s = a_s + kRows * kLineB;
        const int k0       = k_begin + kt * kStageK;
        for (int i = static_cast<int>(threadIdx.x); i < kRows * 8; i += kThreads) {
            const int row = i >> 3, segment = i & 7;
            cp_async<16>(a_s + swizzled(row, segment), a + static_cast<std::int64_t>(row) * k + k0 + segment * 8);
        }
        for (int i = static_cast<int>(threadIdx.x); i < kTokenTile * 8; i += kThreads) {
            const int t = i >> 3, segment = i & 7, token = token_base + t;
            const bool valid = token < tokens;
            cp_async_zfill<16>(x_s + swizzled(t, segment),
                               x + static_cast<std::int64_t>(valid ? token : 0) * k + k0 + segment * 8, valid ? 16 : 0);
        }
    };

    pdl::enter();
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < stages_total) { stage_load(s, s); }
        cp_commit();
    }

    float chunk[MTiles][4] = {}, group[MTiles][4] = {};
    // ldmatrix lane roles: A x4 (rows 0-15 at k + 0, then k + 8), x x2 (8 tokens at k + 0, then k + 8).
    const int a_row = lane & 15, a_segment = lane >> 4;
    const int b_row = lane & 7, b_segment = (lane >> 3) & 1;

    for (int kt = 0; kt < stages_total; ++kt) {
        cp_wait<kStages - 2>();
        __syncthreads();
        {
            const int next = kt + kStages - 1;
            if (next < stages_total) { stage_load(next % kStages, next); }
            cp_commit();
        }
        const unsigned char* a_s = shared + (kt % kStages) * kStageBytes;
        const unsigned char* x_s = a_s + kRows * kLineB;
#pragma unroll
        for (int step = 0; step < kStageK / 16; ++step) {
            unsigned b0, b1;
            ldmatrix_x2(b0, b1, smem_addr(x_s + swizzled(warp * 8 + b_row, step * 2 + b_segment)));
#pragma unroll
            for (int m = 0; m < MTiles; ++m) {
                unsigned a0, a1, a2, a3;
                ldmatrix_x4(a0, a1, a2, a3, smem_addr(a_s + swizzled(m * 16 + a_row, step * 2 + a_segment)));
                mma_bf16(chunk[m][0], chunk[m][1], chunk[m][2], chunk[m][3], a0, a1, a2, a3, b0, b1);
            }
        }
        // A chunk ends every kChunk / kStageK stages: fold it into its group; a group's sum goes to shared memory.
        constexpr int kStagesPerChunk = kChunk / kStageK;
        if ((kt + 1) % kStagesPerChunk == 0) {
            const int chunk_index = kt / kStagesPerChunk; // within this rank's groups, which start at a group
            const bool first_in_group = chunk_index % kGroup == 0, last_in_group = chunk_index % kGroup == kGroup - 1;
#pragma unroll
            for (int m = 0; m < MTiles; ++m)
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    group[m][v] = first_in_group ? chunk[m][v] : __fadd_rn(group[m][v], chunk[m][v]);
                    chunk[m][v] = 0.0F;
                }
            if (last_in_group) {
#pragma unroll
                for (int m = 0; m < MTiles; ++m) {
                    group_sums[sum_at(chunk_index / kGroup, m)] = make_float4(group[m][0], group[m][1], group[m][2],
                                                                              group[m][3]);
                }
            }
        }
    }
    cp_wait<0>();
    cluster.sync(); // every rank's group sums
    if (rank == 0) {
        float total[MTiles][4];
#pragma unroll
        for (int m = 0; m < MTiles; ++m) {
            const float4 s0 = group_sums[sum_at(0, m)]; // group 0 is rank 0's first
            total[m][0] = s0.x;
            total[m][1] = s0.y;
            total[m][2] = s0.z;
            total[m][3] = s0.w;
        }
        int owner = 0;
        for (int gi = 1; gi < groups; ++gi) {
            while (gi >= group_begin(owner + 1, ranks, groups)) { ++owner; }
            const int local = gi - group_begin(owner, ranks, groups);
#pragma unroll
            for (int m = 0; m < MTiles; ++m) {
                const float4 s = *cluster.map_shared_rank(group_sums + sum_at(local, m), owner);
                total[m][0]    = __fadd_rn(total[m][0], s.x);
                total[m][1]    = __fadd_rn(total[m][1], s.y);
                total[m][2]    = __fadd_rn(total[m][2], s.z);
                total[m][3]    = __fadd_rn(total[m][3], s.w);
            }
        }
        const int g = lane >> 2, q = lane & 3;
        const int t0 = token_base + warp * 8 + 2 * q;
#pragma unroll
        for (int m = 0; m < MTiles; ++m) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int row = m * 16 + g + (v >= 2 ? 8 : 0), token = t0 + (v & 1);
                if (token < tokens) {
                    u[static_cast<std::int64_t>(token) * kRows + row] = __float2bfloat16_rn(total[m][v]);
                }
            }
        }
    }
    cluster.sync(); // no rank leaves while rank 0 reads its group sums
}

// The schedule over one slice of the call's rows: rows [slice * MTiles * 16, + MTiles * 16) of MTiles * 16 * Slices
// (slice = blockIdx.z), so a wide adapter's 16- to 24-tile stack, whose rows would not fit one CTA's ring, keeps the
// schedule: x is read once per slice, A still once per 32 columns. A row's sums do not depend on its slice (the same
// bits). The body is lora_shrink_stream_kernel's but for the slice's offsets and u's row stride; that kernel keeps
// its own copy, since sharing one inlined body moved its register allocation (the narrow adapters' kernels stay as
// they were).
template <int MTiles, int Slices>
__device__ __forceinline__ void lora_shrink_stream(const __nv_bfloat16* x, const __nv_bfloat16* __restrict__ a,
                                                   __nv_bfloat16* u, int k, int tokens) {
    constexpr int kRows       = MTiles * 16;
    constexpr int kOutRows    = kRows * Slices; // u's row stride
    static_assert(Slices > 1);
    a += static_cast<std::int64_t>(blockIdx.z) * kRows * k;
    u += static_cast<int>(blockIdx.z) * kRows;
    constexpr int kStages     = kStreamStages<MTiles>;
    constexpr int kStageBytes = (kRows + kTokenTile) * kLineB;
    extern __shared__ __align__(16) unsigned char shared[];
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = static_cast<int>(cluster.block_rank()), ranks = static_cast<int>(cluster.num_blocks());
    const int groups  = k / kGroupValues;
    const int g_begin = group_begin(rank, ranks, groups), g_end = group_begin(rank + 1, ranks, groups);
    const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
    const int token_base   = static_cast<int>(blockIdx.y) * kTokenTile;
    const int k_begin      = g_begin * kGroupValues;
    const int stages_total = (g_end - g_begin) * (kGroupValues / kStageK);
    auto* group_sums       = reinterpret_cast<float4*>(shared + kStages * kStageBytes);
    const auto sum_at      = [&](int local_group, int m) { return ((local_group * MTiles + m) * 4 + warp) * 32 + lane; };

    const auto stage_load = [&](int stage, int kt) {
        unsigned char* a_s = shared + stage * kStageBytes;
        unsigned char* x_s = a_s + kRows * kLineB;
        const int k0       = k_begin + kt * kStageK;
        for (int i = static_cast<int>(threadIdx.x); i < kRows * 8; i += kThreads) {
            const int row = i >> 3, segment = i & 7;
            cp_async<16>(a_s + swizzled(row, segment), a + static_cast<std::int64_t>(row) * k + k0 + segment * 8);
        }
        for (int i = static_cast<int>(threadIdx.x); i < kTokenTile * 8; i += kThreads) {
            const int t = i >> 3, segment = i & 7, token = token_base + t;
            const bool valid = token < tokens;
            cp_async_zfill<16>(x_s + swizzled(t, segment),
                               x + static_cast<std::int64_t>(valid ? token : 0) * k + k0 + segment * 8, valid ? 16 : 0);
        }
    };

    pdl::enter();
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < stages_total) { stage_load(s, s); }
        cp_commit();
    }

    float chunk[MTiles][4] = {}, group[MTiles][4] = {};
    // ldmatrix lane roles: A x4 (rows 0-15 at k + 0, then k + 8), x x2 (8 tokens at k + 0, then k + 8).
    const int a_row = lane & 15, a_segment = lane >> 4;
    const int b_row = lane & 7, b_segment = (lane >> 3) & 1;

    for (int kt = 0; kt < stages_total; ++kt) {
        cp_wait<kStages - 2>();
        __syncthreads();
        {
            const int next = kt + kStages - 1;
            if (next < stages_total) { stage_load(next % kStages, next); }
            cp_commit();
        }
        const unsigned char* a_s = shared + (kt % kStages) * kStageBytes;
        const unsigned char* x_s = a_s + kRows * kLineB;
#pragma unroll
        for (int step = 0; step < kStageK / 16; ++step) {
            unsigned b0, b1;
            ldmatrix_x2(b0, b1, smem_addr(x_s + swizzled(warp * 8 + b_row, step * 2 + b_segment)));
#pragma unroll
            for (int m = 0; m < MTiles; ++m) {
                unsigned a0, a1, a2, a3;
                ldmatrix_x4(a0, a1, a2, a3, smem_addr(a_s + swizzled(m * 16 + a_row, step * 2 + a_segment)));
                mma_bf16(chunk[m][0], chunk[m][1], chunk[m][2], chunk[m][3], a0, a1, a2, a3, b0, b1);
            }
        }
        // A chunk ends every kChunk / kStageK stages: fold it into its group; a group's sum goes to shared memory.
        constexpr int kStagesPerChunk = kChunk / kStageK;
        if ((kt + 1) % kStagesPerChunk == 0) {
            const int chunk_index = kt / kStagesPerChunk; // within this rank's groups, which start at a group
            const bool first_in_group = chunk_index % kGroup == 0, last_in_group = chunk_index % kGroup == kGroup - 1;
#pragma unroll
            for (int m = 0; m < MTiles; ++m)
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    group[m][v] = first_in_group ? chunk[m][v] : __fadd_rn(group[m][v], chunk[m][v]);
                    chunk[m][v] = 0.0F;
                }
            if (last_in_group) {
#pragma unroll
                for (int m = 0; m < MTiles; ++m) {
                    group_sums[sum_at(chunk_index / kGroup, m)] = make_float4(group[m][0], group[m][1], group[m][2],
                                                                              group[m][3]);
                }
            }
        }
    }
    cp_wait<0>();
    cluster.sync(); // every rank's group sums
    if (rank == 0) {
        float total[MTiles][4];
#pragma unroll
        for (int m = 0; m < MTiles; ++m) {
            const float4 s0 = group_sums[sum_at(0, m)]; // group 0 is rank 0's first
            total[m][0] = s0.x;
            total[m][1] = s0.y;
            total[m][2] = s0.z;
            total[m][3] = s0.w;
        }
        int owner = 0;
        for (int gi = 1; gi < groups; ++gi) {
            while (gi >= group_begin(owner + 1, ranks, groups)) { ++owner; }
            const int local = gi - group_begin(owner, ranks, groups);
#pragma unroll
            for (int m = 0; m < MTiles; ++m) {
                const float4 s = *cluster.map_shared_rank(group_sums + sum_at(local, m), owner);
                total[m][0]    = __fadd_rn(total[m][0], s.x);
                total[m][1]    = __fadd_rn(total[m][1], s.y);
                total[m][2]    = __fadd_rn(total[m][2], s.z);
                total[m][3]    = __fadd_rn(total[m][3], s.w);
            }
        }
        const int g = lane >> 2, q = lane & 3;
        const int t0 = token_base + warp * 8 + 2 * q;
#pragma unroll
        for (int m = 0; m < MTiles; ++m) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int row = m * 16 + g + (v >= 2 ? 8 : 0), token = t0 + (v & 1);
                if (token < tokens) {
                    u[static_cast<std::int64_t>(token) * kOutRows + row] = __float2bfloat16_rn(total[m][v]);
                }
            }
        }
    }
    cluster.sync(); // no rank leaves while rank 0 reads its group sums
}

// grid (ranks, ceil(T / 32), Slices).
template <int MTiles, int Slices>
__global__ __launch_bounds__(kThreads) void lora_shrink_stream_sliced_kernel(const __nv_bfloat16* x,
                                                                             const __nv_bfloat16* __restrict__ a,
                                                                             __nv_bfloat16* u, int k, int tokens) {
    lora_shrink_stream<MTiles, Slices>(x, a, u, k, tokens);
}

// ---------------------------------------------------------------------------------------------------------------
// Tile schedule (prompts). grid (ranks, ceil(T / BM), slices), cluster (ranks, 1, 1). A CTA computes one BM-column x
// BN-row tile of one slice of A's rows, a whole group of K a round: rank r takes groups r, r + ranks, r + 2 ranks, ...,
// so after each round the cluster holds consecutive group sums, which it adds to U's running sums in order (each rank
// its share of the tile's elements, read from every rank's shared memory). Consumer warps take 32 x 32 sub-tiles (two
// m16 row tiles of A by four n8 column tiles): two ldmatrix of A and two of x per eight MMAs, the chunk and group sums
// in registers. One producer warp fills a ring of stages of kTileStageK values (A's BN rows, then x's BM columns, in
// 64-byte lines) with TMA copies under full / empty mbarriers: a consumer warp waits only for its next stage, never for
// the CTA (a cp.async ring with a CTA barrier a stage ran 1.15-1.4x slower, RTX 5090, 2026-11-22). The copies' 64-byte
// swizzle is tile_swizzled's layout (ldmatrix's eight rows hit distinct banks); rows and columns past the call are
// zero-filled. The published group sums reuse the ring, so the producer refills it only after a round's reduction.
// Dynamic shared memory: the ring, its barriers, then the rank's share of the running sums.
constexpr int kTileStageK = 32;
constexpr int kTileLineB  = kTileStageK * 2; // 64 bytes

__device__ __forceinline__ int tile_swizzled(int row, int segment) {
    return row * kTileLineB + ((segment ^ ((row >> 1) & 3)) << 4);
}

int shared_optin_bytes();

template <int BM, int BN>
struct ShrinkTile {
    static_assert(BM % 32 == 0 && BN % 32 == 0);
    static constexpr int kWarpsM     = BM / 32; // column (token) warps
    static constexpr int kWarpsN     = BN / 32; // row warps
    static constexpr int kConsumers  = kWarpsM * kWarpsN * 32;
    static constexpr int kThreads    = kConsumers + 32; // the producer warp last
    static constexpr int kStageBytes = (BM + BN) * kTileLineB;
    static constexpr int kElements4  = BM * BN / 4; // float4 units of the tile
    // The ring: as many stages as about 48 KB holds, 4 to 12 (a round of 32 stages is latency-bound: a small tile
    // keeps more stages in flight), and at least the published sums' bytes.
    static constexpr int kStages = std::max(std::min(49152 / kStageBytes, 12),
                                            std::max(4, (BM * BN * 4 + kStageBytes - 1) / kStageBytes));
    static constexpr int kRingBytes   = kStages * kStageBytes;
    static constexpr int kShareOffset = (kRingBytes + 2 * kStages * 8 + 15) / 16 * 16;
    // The running sums of one rank's share.
    static constexpr std::size_t share_bytes(int ranks) {
        return static_cast<std::size_t>((kElements4 + ranks - 1) / ranks) * sizeof(float4);
    }
    static constexpr std::size_t bytes(int ranks) { return kShareOffset + share_bytes(ranks); }
};

// At most 8 consumer warps: the 12- and 16-warp tiles (128 x 96, 128 x 128) spilled at their register cap and ran at
// half the speed of 64-column tiles, which already fill an SM.
template <int BM, int BN>
inline constexpr bool kTileShape = BM * BN <= 64 * 128;

struct alignas(128) ShrinkTileDescriptors {
    CUtensorMap a; // A [rows, k]: boxes of BN rows x 32 values
    CUtensorMap x; // x [T, k]: boxes of BM columns x 32 values
};

#ifdef _WIN32
inline TmaDescriptorStaging<ShrinkTileDescriptors>& shrink_tile_staging() {
    static TmaDescriptorStaging<ShrinkTileDescriptors> staging;
    return staging;
}
#endif

__device__ __forceinline__ void shrink_tile_load(void* destination, const CUtensorMap* map, int k0, int outer,
                                                std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], "
                 "[%4];"
                 :
                 : "r"(smem_addr(destination)), "l"(map), "r"(k0), "r"(outer), "r"(smem_addr(barrier))
                 : "memory");
}

template <int BM, int BN, bool InPlace>
__global__ __launch_bounds__(ShrinkTile<BM, BN>::kThreads, 1) void lora_shrink_tile_kernel(
#ifdef _WIN32
    const __grid_constant__ TmaDescriptorParameter<ShrinkTileDescriptors, InPlace> descriptors_pointer,
#else
    const __grid_constant__ ShrinkTileDescriptors descriptors,
#endif
    __nv_bfloat16* u, int rows, int k, int tokens) {
#ifdef _WIN32
    const ShrinkTileDescriptors& descriptors = *descriptors_pointer;
#endif
    using S                   = ShrinkTile<BM, BN>;
    constexpr int kStages     = S::kStages;
    constexpr int kConsumerWarps = S::kConsumers / 32;
    extern __shared__ __align__(1024) unsigned char shared[];
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = static_cast<int>(cluster.block_rank()), ranks = static_cast<int>(cluster.num_blocks());
    const int groups = k / kGroupValues;
    const int rounds = (groups + ranks - 1) / ranks;
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread >> 5, lane = thread & 31;
    const int token_base  = static_cast<int>(blockIdx.y) * BM;
    const int row_base    = static_cast<int>(blockIdx.z) * BN;
    auto* full            = reinterpret_cast<std::uint64_t*>(shared + S::kRingBytes);
    auto* empty           = full + kStages;
    auto* published       = reinterpret_cast<float*>(shared); // [BM][BN] after a round's stages
    auto* share           = reinterpret_cast<float4*>(shared + S::kShareOffset);
    const int share_begin = rank * S::kElements4 / ranks, share_end = (rank + 1) * S::kElements4 / ranks;
    constexpr int kStagesPerChunk = kChunk / kTileStageK;      // 8
    constexpr int kStagesPerGroup = kGroupValues / kTileStageK; // 32
    if (thread == 0) {
#pragma unroll
        for (int s = 0; s < kStages; ++s) {
            cta_mbarrier_init(full + s, 1);
            cta_mbarrier_init(empty + s, kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();

    if (warp == kConsumerWarps) {
        // The producer: lane 0 issues every stage; the warp takes part in the cluster barriers.
        int issued = 0; // stages this CTA has issued (the ring's position across rounds)
        for (int round = 0; round < rounds; ++round) {
            const int group = round * ranks + rank;
            if (group < groups) {
                const int k_begin = group * kGroupValues;
                const auto issue  = [&](int kt, bool with_a, bool with_x) {
                    const int n = issued + kt, slot = n % kStages;
                    unsigned char* a_s = shared + slot * S::kStageBytes;
                    if (with_a) {
                        if (n >= kStages) { cta_mbarrier_wait(empty + slot, ((n / kStages) & 1U) ^ 1U); }
                        cta_mbarrier_arrive_expect_tx(full + slot, S::kStageBytes);
                        shrink_tile_load(a_s, &descriptors.a, k_begin + kt * kTileStageK, row_base, full + slot);
                    }
                    if (with_x) {
                        shrink_tile_load(a_s + BN * kTileLineB, &descriptors.x, k_begin + kt * kTileStageK,
                                        token_base, full + slot);
                    }
                };
                int kt = 0;
                if (round == 0) {
                    if (lane == 0) {
#ifdef _WIN32
                        acquire_staged_tensor_map<InPlace>(&descriptors.a);
                        acquire_staged_tensor_map<InPlace>(&descriptors.x);
#endif
                        // A is an immutable weight: the first stages' A loads issue before the PDL wait.
                        for (int s = 0; s < kStages; ++s) { issue(s, true, false); }
                    }
                    pdl::enter();
                    if (lane == 0) {
                        for (int s = 0; s < kStages; ++s) { issue(s, false, true); }
                    }
                    kt = kStages;
                }
                if (lane == 0) {
                    // After a round's reduction the consumers' generic accesses to the ring precede these copies.
                    if (round > 0) { fence_reads_before_async_refill(); }
                    for (; kt < kStagesPerGroup; ++kt) { issue(kt, true, true); }
                }
                issued += kStagesPerGroup;
            } else if (round == 0) {
                pdl::enter();
            }
            __syncwarp();
            cluster.sync(); // the round's group sums are published
            cluster.sync(); // every rank has read them
        }
        return;
    }

    pdl::enter(); // the consumers touch global memory only to store U
    const int warp_m = warp % S::kWarpsM, warp_n = warp / S::kWarpsM;
    const int a_row = lane & 15, a_segment = lane >> 4;
    const int x_row = ((lane >> 4) << 3) + (lane & 7), x_segment = (lane >> 3) & 1;
    const int g = lane >> 2, q = lane & 3;
    int consumed = 0;
    for (int round = 0; round < rounds; ++round) {
        const int group = round * ranks + rank;
        if (group < groups) {
            float chunk[2][4][4] = {}, sum[2][4][4] = {};
            for (int kt = 0; kt < kStagesPerGroup; ++kt) {
                const int n = consumed + kt, slot = n % kStages;
                cta_mbarrier_wait(full + slot, (n / kStages) & 1U);
                const unsigned char* a_s = shared + slot * S::kStageBytes;
                const unsigned char* x_s = a_s + BN * kTileLineB;
#pragma unroll
                for (int step = 0; step < kTileStageK / 16; ++step) {
                    unsigned af[2][4], xf[4][2];
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        ldmatrix_x4(af[mt][0], af[mt][1], af[mt][2], af[mt][3],
                                    smem_addr(a_s + tile_swizzled(warp_n * 32 + mt * 16 + a_row, step * 2 + a_segment)));
                    }
#pragma unroll
                    for (int np = 0; np < 2; ++np) {
                        ldmatrix_x4(xf[2 * np][0], xf[2 * np][1], xf[2 * np + 1][0], xf[2 * np + 1][1],
                                    smem_addr(x_s + tile_swizzled(warp_m * 32 + np * 16 + x_row, step * 2 + x_segment)));
                    }
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                        for (int nt = 0; nt < 4; ++nt) {
                            mma_bf16(chunk[mt][nt][0], chunk[mt][nt][1], chunk[mt][nt][2], chunk[mt][nt][3],
                                     af[mt][0], af[mt][1], af[mt][2], af[mt][3], xf[nt][0], xf[nt][1]);
                        }
                }
                // The stage's reads are done: release it to the producer.
                fence_reads_before_async_refill();
                __syncwarp();
                if (lane == 0) { cta_mbarrier_arrive(empty + slot); }
                // A chunk ends every kStagesPerChunk stages: fold it into the group's sum (from its first chunk).
                if ((kt + 1) % kStagesPerChunk == 0) {
                    const bool first = kt + 1 == kStagesPerChunk;
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                            for (int v = 0; v < 4; ++v) {
                                sum[mt][nt][v]   = first ? chunk[mt][nt][v] : __fadd_rn(sum[mt][nt][v], chunk[mt][nt][v]);
                                chunk[mt][nt][v] = 0.0F;
                            }
                }
            }
            consumed += kStagesPerGroup;
            // Every consumer warp is done with the ring (every stage it waited for has landed): it now holds the
            // group's sum, [column][row].
            asm volatile("bar.sync 1, %0;" : : "r"(S::kConsumers) : "memory");
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int r = warp_n * 32 + mt * 16 + g, t = warp_m * 32 + nt * 8 + 2 * q;
                    published[t * BN + r]           = sum[mt][nt][0];
                    published[(t + 1) * BN + r]     = sum[mt][nt][1];
                    published[t * BN + r + 8]       = sum[mt][nt][2];
                    published[(t + 1) * BN + r + 8] = sum[mt][nt][3];
                }
        }
        cluster.sync(); // the round's group sums are published
        for (int e = share_begin + thread; e < share_end; e += S::kConsumers) {
            float4 total = round == 0 ? float4{} : share[e - share_begin];
            for (int r = 0; r < ranks; ++r) {
                const int gi = round * ranks + r;
                if (gi >= groups) { break; }
                const float4 v = *cluster.map_shared_rank(reinterpret_cast<float4*>(published) + e, r);
                if (gi == 0) {
                    total = v;
                } else {
                    total.x = __fadd_rn(total.x, v.x);
                    total.y = __fadd_rn(total.y, v.y);
                    total.z = __fadd_rn(total.z, v.z);
                    total.w = __fadd_rn(total.w, v.w);
                }
            }
            share[e - share_begin] = total;
        }
        // The ring's generic writes (published) and the cluster's reads of it precede the producer's refill.
        asm volatile("fence.proxy.async;" : : : "memory");
        cluster.sync(); // every rank has read the published sums: the rings may be refilled
    }
    const int live_tokens = min(BM, tokens - token_base);
    const int live_rows   = min(BN, rows - row_base);
    for (int e = share_begin + thread; e < share_end; e += S::kConsumers) {
        const int t = (e * 4) / BN, r = (e * 4) % BN;
        if (t >= live_tokens || r >= live_rows) { continue; }
        const float4 total = share[e - share_begin];
        __nv_bfloat16* dst = u + static_cast<std::int64_t>(token_base + t) * rows + row_base + r;
        if (r + 4 <= live_rows) {
            const __nv_bfloat162 lo = __floats2bfloat162_rn(total.x, total.y);
            const __nv_bfloat162 hi = __floats2bfloat162_rn(total.z, total.w);
            uint2 packed;
            packed.x = *reinterpret_cast<const unsigned*>(&lo);
            packed.y = *reinterpret_cast<const unsigned*>(&hi);
            *reinterpret_cast<uint2*>(dst) = packed;
        } else {
            const float values[4] = {total.x, total.y, total.z, total.w};
            for (int i = 0; i < 4 && r + i < live_rows; ++i) { dst[i] = __float2bfloat16_rn(values[i]); }
        }
    }
}

CUtensorMap shrink_tile_map(const void* pointer, int outer, int k, int box_outer) {
    CUtensorMap result{};
    const std::uint64_t dimensions[]{static_cast<std::uint64_t>(k), static_cast<std::uint64_t>(outer)};
    const std::uint64_t strides[]{static_cast<std::uint64_t>(k) * 2};
    const std::uint32_t box[]{static_cast<std::uint32_t>(kTileStageK), static_cast<std::uint32_t>(box_outer)};
    const std::uint32_t steps[]{1, 1};
    const auto status = cuTensorMapEncodeTiled(&result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(pointer),
                                               dimensions, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                               CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("lora_shrink tile descriptor: ") + (name ? name : "CUDA error"));
    }
    return result;
}

// The tile schedule's shape for a call: its token tile, row tile and slices, the cluster's ranks, and its modelled time.
struct TilePlan {
    int bm     = 0;
    int bn     = 0;
    int slices = 0;
    int ranks  = 0;
    double us  = 0.0;
};

int sm_count() {
    static const int count = [] {
        int device = 0, value = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device));
        return value;
    }();
    return count;
}

// The clusters of `ranks` CTAs of a tile kernel the device runs at once (cudaOccupancyMaxActiveClusters: what the GPCs
// hold, at most two CTAs an SM; on the RTX 5090 22 clusters of 6 at one CTA an SM, 52 at two), once a shape and ranks.
// -1: the shape has no kernel or does not fit.
template <int BM, int BN>
int tile_capacity(int ranks) {
    if constexpr (!kTileShape<BM, BN>) {
        return -1;
    } else {
        using S        = ShrinkTile<BM, BN>;
        constexpr auto kernel = lora_shrink_tile_kernel<BM, BN, false>;
        static std::atomic<int> cache[kMaxRanks + 1] = {};
        int value = cache[ranks].load(std::memory_order_relaxed);
        if (value != 0) { return value; }
        const std::size_t smem = S::bytes(ranks);
        if (smem > static_cast<std::size_t>(shared_optin_bytes())) {
            value = -1;
        } else {
            // The attribute only bounds a launch: the instance's own launch sets its largest need.
            CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_optin_bytes()));
            cudaLaunchAttribute attribute{};
            attribute.id               = cudaLaunchAttributeClusterDimension;
            attribute.val.clusterDim.x = static_cast<unsigned>(ranks);
            attribute.val.clusterDim.y = 1;
            attribute.val.clusterDim.z = 1;
            cudaLaunchConfig_t config{};
            config.gridDim          = dim3(static_cast<unsigned>(ranks));
            config.blockDim         = dim3(static_cast<unsigned>(S::kThreads));
            config.dynamicSmemBytes = smem;
            config.attrs            = &attribute;
            config.numAttrs         = 1;
            int clusters            = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveClusters(&clusters, kernel, &config));
            value = clusters > 0 ? clusters : -1;
        }
        cache[ranks].store(value, std::memory_order_relaxed);
        return value;
    }
}

int tile_capacity(int bm, int bn, int ranks) {
    const auto by_bn = [&]<int BM>() {
        switch (bn) {
        case 32: return tile_capacity<BM, 32>(ranks);
        case 64: return tile_capacity<BM, 64>(ranks);
        case 96: return tile_capacity<BM, 96>(ranks);
        default: return tile_capacity<BM, 128>(ranks);
        }
    };
    if (bm == 128) { return by_bn.template operator()<128>(); }
    if (bm == 64) { return by_bn.template operator()<64>(); }
    return by_bn.template operator()<32>();
}

// The tile schedule's modelled time for a shape (us). A round of a CTA takes the longer of the pipeline's latency
// floor (kTileRoundFloorUs: 32 stages of 32 values, each waiting for copies a few stages old) and its share of the SM:
// k CTAs on an SM run 2 BM BN 1024 FLOP each at kSmRate (reached from one 4-warp CTA). Waves follow the device's
// cluster capacity, each a few us of ramp and tail. Fitted on RTX 5090 measurements (2026-11-22, lora_shrink bench,
// the 27B's five sites at ranks 32 and 128, 256-4,096 columns, every token tile and cluster: 11 % rms error); the
// plan's pick averaged 1.02x the best measured tile shape and cluster, 1.14x at worst.
constexpr double kSmRate           = 1.0e6; // FLOP per us
constexpr double kTileRoundFloorUs = 11.0;
constexpr double kWaveOverheadUs   = 3.0;
double tile_model_us(int bm, int bn, int slices, int ranks, int rounds, int tokens, int capacity) {
    const int clusters = (tokens + bm - 1) / bm * slices;
    const int sms      = sm_count();
    const double work  = 2.0 * bm * bn * kGroupValues;
    const auto wave    = [&](int wave_clusters) {
        const int on_sm = (wave_clusters * ranks + sms - 1) / sms;
        return rounds * std::max(kTileRoundFloorUs, on_sm * work / kSmRate);
    };
    const int full = clusters / capacity, rest = clusters % capacity;
    return full * wave(capacity) + (rest > 0 ? wave(rest) : 0.0) + (full + (rest > 0)) * kWaveOverheadUs;
}

// The modelled fastest token tile and cluster for a call: every token tile, and every distinct split of the groups into
// rounds (ranks = ceil(groups / rounds), at most kMaxRanks), over row slices of at most 128 rows.
TilePlan tile_plan(int rows, int k, int tokens) {
    TilePlan plan;
    const int groups = k / kGroupValues;
    plan.slices      = (rows + 127) / 128;
    plan.bn          = ((rows + plan.slices - 1) / plan.slices + 31) / 32 * 32;
    plan.slices      = (rows + plan.bn - 1) / plan.bn;
    for (const int bm : {32, 64, 128}) {
        int previous = 0;
        for (int rounds = 1; rounds <= groups; ++rounds) {
            const int ranks = (groups + rounds - 1) / rounds;
            if (ranks > kMaxRanks || ranks == previous) { continue; }
            previous           = ranks;
            const int capacity = tile_capacity(bm, plan.bn, ranks);
            if (capacity <= 0) { continue; }
            const double us =
                tile_model_us(bm, plan.bn, plan.slices, ranks, (groups + ranks - 1) / ranks, tokens, capacity);
            if (plan.us == 0.0 || us < plan.us) {
                plan.us    = us;
                plan.bm    = bm;
                plan.ranks = ranks;
            }
        }
    }
    return plan;
}

template <int BM, int BN>
void launch_tile_as(const Tensor& x, const LoraShrinkWeight& a, Tensor& u, cudaStream_t stream, const TilePlan& plan) {
    using S          = ShrinkTile<BM, BN>;
    const int tokens = x.ne[1];
    const ShrinkTileDescriptors descriptors{shrink_tile_map(a.a, a.rows, a.k, BN), shrink_tile_map(x.data, tokens, a.k, BM)};
    const auto launch = [&]<bool InPlace>(std::bool_constant<InPlace>, const auto& parameter) {
        constexpr auto kernel  = lora_shrink_tile_kernel<BM, BN, InPlace>;
        static bool configured = [] {
            const std::size_t most = std::min(S::bytes(1), static_cast<std::size_t>(shared_optin_bytes()));
            CUDA_CHECK(cudaFuncSetAttribute(lora_shrink_tile_kernel<BM, BN, InPlace>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(most)));
            return true;
        }();
        (void)configured;
        cudaLaunchAttribute attributes[2]{};
        attributes[0].id               = cudaLaunchAttributeClusterDimension;
        attributes[0].val.clusterDim.x = static_cast<unsigned>(plan.ranks);
        attributes[0].val.clusterDim.y = 1;
        attributes[0].val.clusterDim.z = 1;
        unsigned count                 = 1;
        if (pdl::graph_dependent(stream, kernel)) {
            attributes[1].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
            attributes[1].val.programmaticStreamSerializationAllowed = 1;
            count                                                    = 2;
        }
        cudaLaunchConfig_t config{};
        config.gridDim          = dim3(static_cast<unsigned>(plan.ranks), static_cast<unsigned>((tokens + BM - 1) / BM),
                                       static_cast<unsigned>(plan.slices));
        config.blockDim         = dim3(S::kThreads);
        config.dynamicSmemBytes = S::bytes(plan.ranks);
        config.stream           = stream;
        config.attrs            = attributes;
        config.numAttrs         = count;
        CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, parameter, static_cast<__nv_bfloat16*>(u.data), a.rows, a.k,
                                      tokens));
        CUDA_CHECK(cudaGetLastError());
    };
#ifdef _WIN32
    with_tma_descriptor_parameter(shrink_tile_staging(), descriptors, stream, launch);
#else
    launch(std::false_type{}, descriptors);
#endif
}

template <int BM>
void launch_tile_bn(const Tensor& x, const LoraShrinkWeight& a, Tensor& u, cudaStream_t stream, const TilePlan& plan) {
    switch (plan.bn) {
    case 32: return launch_tile_as<BM, 32>(x, a, u, stream, plan);
    case 64: return launch_tile_as<BM, 64>(x, a, u, stream, plan);
    case 96:
        if constexpr (kTileShape<BM, 96>) { return launch_tile_as<BM, 96>(x, a, u, stream, plan); }
        break;
    default:
        if constexpr (kTileShape<BM, 128>) { return launch_tile_as<BM, 128>(x, a, u, stream, plan); }
        break;
    }
    throw std::logic_error("lora_shrink: the plan chose a tile without a kernel");
}

void launch_tile(const Tensor& x, const LoraShrinkWeight& a, Tensor& u, cudaStream_t stream, const TilePlan& plan) {
    if (plan.bm == 128) { return launch_tile_bn<128>(x, a, u, stream, plan); }
    if (plan.bm == 64) { return launch_tile_bn<64>(x, a, u, stream, plan); }
    launch_tile_bn<32>(x, a, u, stream, plan);
}

// From this many columns a call streams x once instead of splitting K: the split schedule reads x once per 16-row tile
// of A (from L2 after the first), the stream schedule A once per 32 columns. Measured 2026-11-21 (RTX 5090, the 27B's
// four sites at rank 32): the split is faster to 256 columns (q/k/v 13.7 against 17.0 us, down 16.6 against 23.1),
// the stream from 512 (q/k/v 17.3 against 20.8, down 23.8 against 29.0) and by 1.1-1.7x at 1,024-4,096.
constexpr std::int32_t kStreamMinTokens = 512;
// From this many columns a call may take the tile schedule (below, the split schedule is faster at every height).
constexpr std::int32_t kTileMinTokens = 64;

// The heights (rows / 16) the stream schedule takes from kStreamMinTokens columns (lora_shrink); others split.
constexpr bool stream_height(int tiles) {
    switch (tiles) {
    case 2: case 4: case 5: case 6: case 7: case 8: case 10: case 12: case 14: case 15: case 16: case 18: case 21:
    case 24: return true;
    default: return false;
    }
}

// The base schedules' (split, stream) modelled time for a call (us): about 8.5 us at the narrowest calls (14.5 above 8
// groups, whose ranks take several groups each), else 1e-7 us a row x column x value of K, 6e-8 where the stream
// schedule runs. A call takes the tile schedule when its modelled time is below the base's by kTileMargin (the tile
// model errs a little low on the narrowest stacks). Fitted on the crossover and tile sweeps (RTX 5090, 2026-11-22, the
// 27B's five sites, ranks 16-128 at 64-4,096 columns, 290 cells): the pick averaged 1.006x the fastest measured
// schedule; one cell took a schedule slower than the base rule's (o_proj at rank 32 and 1,536 columns: the tile,
// 16.7 against the stream schedule's 14.7 us).
constexpr double kTileMargin = 1.1;
double base_model_us(int rows, int k, int tokens) {
    const bool streamed = tokens >= kStreamMinTokens && stream_height(rows / 16);
    const double floor  = k / kGroupValues <= kMaxRanks ? 8.5 : 14.5;
    return std::max(floor, (streamed ? 6e-8 : 1e-7) * rows * static_cast<double>(k) * tokens);
}

// The stream schedule's dynamic shared memory for a rank of `rank_groups` groups.
template <int MTiles>
constexpr std::size_t stream_bytes(int rank_groups) {
    return static_cast<std::size_t>(stream_ring_bytes<MTiles>()) +
           static_cast<std::size_t>(rank_groups) * MTiles * 4 * 32 * sizeof(float4);
}

int shared_optin_bytes() {
    static const int bytes = [] {
        int device = 0, value = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        return value;
    }();
    return bytes;
}

// Whether the stream schedule's shared memory fits one CTA for these rows and K (else the call splits).
template <int MTiles>
bool stream_fits(int k) {
    const int groups = k / kGroupValues, ranks = std::min(groups, kMaxRanks);
    return stream_bytes<MTiles>((groups + ranks - 1) / ranks) <= static_cast<std::size_t>(shared_optin_bytes());
}

template <int MTiles, int Slices>
constexpr auto stream_kernel() {
    if constexpr (Slices == 1) {
        return lora_shrink_stream_kernel<MTiles>;
    } else {
        return lora_shrink_stream_sliced_kernel<MTiles, Slices>;
    }
}

template <int MTiles, int Slices>
void launch_stream(const Tensor& x, const LoraShrinkWeight& a, Tensor& u, cudaStream_t stream) {
    constexpr auto kernel = stream_kernel<MTiles, Slices>();
    static bool configured = [] {
        const std::size_t most = std::min(stream_bytes<MTiles>(kMaxGroupsPerRank),
                                          static_cast<std::size_t>(shared_optin_bytes()));
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(most)));
        return true;
    }();
    (void)configured;
    const int tokens = x.ne[1];
    const int groups = a.k / kGroupValues;
    const int ranks  = std::min(groups, kMaxRanks);
    cudaLaunchAttribute attributes[2]{};
    attributes[0].id               = cudaLaunchAttributeClusterDimension;
    attributes[0].val.clusterDim.x = static_cast<unsigned>(ranks);
    attributes[0].val.clusterDim.y = 1;
    attributes[0].val.clusterDim.z = 1;
    unsigned count                 = 1;
    if (pdl::graph_dependent(stream, kernel)) {
        attributes[1].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
        attributes[1].val.programmaticStreamSerializationAllowed = 1;
        count                                                    = 2;
    }
    cudaLaunchConfig_t config{};
    config.gridDim          = dim3(static_cast<unsigned>(ranks),
                                   static_cast<unsigned>((tokens + kTokenTile - 1) / kTokenTile),
                                   static_cast<unsigned>(Slices));
    config.blockDim         = dim3(kThreads);
    config.dynamicSmemBytes = stream_bytes<MTiles>((groups + ranks - 1) / ranks);
    config.stream           = stream;
    config.attrs            = attributes;
    config.numAttrs         = count;
    CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, static_cast<const __nv_bfloat16*>(x.data),
                                  static_cast<const __nv_bfloat16*>(a.a), static_cast<__nv_bfloat16*>(u.data), a.k,
                                  tokens));
}

template <int NT, int G>
void launch_split_as(const Tensor& x, const LoraShrinkWeight& a, Tensor& u, cudaStream_t stream, int ranks) {
    constexpr auto kernel = lora_shrink_split_kernel<NT, G>;
    // The largest K (kMaxRanks * kMaxGroupsPerRank groups) sets the attribute once.
    constexpr int kMaxBytes = (G * 4 + kMaxRanks * kMaxGroupsPerRank) * NT * 32 * 16;
    static bool configured  = [] {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxBytes));
        return true;
    }();
    (void)configured;
    const int groups   = a.k / kGroupValues;
    const int tokens   = x.ne[1];
    const auto bytes   = static_cast<std::size_t>((G * 4 + groups) * NT * 32 * 16);
    cudaLaunchAttribute attributes[2]{};
    attributes[0].id               = cudaLaunchAttributeClusterDimension;
    attributes[0].val.clusterDim.x = static_cast<unsigned>(ranks);
    attributes[0].val.clusterDim.y = 1;
    attributes[0].val.clusterDim.z = 1;
    unsigned count                 = 1;
    if (pdl::graph_dependent(stream, kernel)) {
        attributes[1].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
        attributes[1].val.programmaticStreamSerializationAllowed = 1;
        count                                                    = 2;
    }
    cudaLaunchConfig_t config{};
    config.gridDim          = dim3(static_cast<unsigned>(ranks), static_cast<unsigned>(a.rows / 16),
                                   static_cast<unsigned>((tokens + 8 * NT - 1) / (8 * NT)));
    config.blockDim         = dim3(kThreads * G);
    config.dynamicSmemBytes = bytes;
    config.stream           = stream;
    config.attrs            = attributes;
    config.numAttrs         = count;
    CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, static_cast<const __nv_bfloat16*>(x.data),
                                  static_cast<const __nv_bfloat16*>(a.a), static_cast<__nv_bfloat16*>(u.data), a.rows,
                                  a.k, tokens));
    CUDA_CHECK(cudaGetLastError());
}

template <int NT>
void launch_split_tiles(const Tensor& x, const LoraShrinkWeight& a, Tensor& u, cudaStream_t stream) {
    const int groups = a.k / kGroupValues;
    const int ranks  = std::min(groups, kMaxRanks);
    switch ((groups + ranks - 1) / ranks) { // the most groups a rank takes
    case 1: return launch_split_as<NT, 1>(x, a, u, stream, ranks);
    case 2: return launch_split_as<NT, 2>(x, a, u, stream, ranks);
    default: return launch_split_as<NT, 3>(x, a, u, stream, ranks);
    }
}

void launch_split(const Tensor& x, const LoraShrinkWeight& a, Tensor& u, cudaStream_t stream) {
    // The fewest n8 tiles that cover the call's columns a CTA (decode and short verification), else 32 columns.
    const int tokens = x.ne[1];
    if (tokens <= 8) { return launch_split_tiles<1>(x, a, u, stream); }
    if (tokens <= 16) { return launch_split_tiles<2>(x, a, u, stream); }
    launch_split_tiles<4>(x, a, u, stream);
}

} // namespace
} // namespace detail

void lora_shrink(const Tensor& x, const LoraShrinkWeight& a, Tensor& u, cudaStream_t stream) {
    const auto fail = [](const std::string& what) { throw std::invalid_argument("lora_shrink: " + what); };
    if (a.a == nullptr || a.rows <= 0 || a.rows % 16 != 0 || a.k <= 0 || a.k % (kLoraShrinkChunk * kLoraShrinkGroup)) {
        fail("A must be BF16 [rows, k] with rows a multiple of 16 and k a multiple of 1024");
    }
    if (a.k / (kLoraShrinkChunk * kLoraShrinkGroup) > detail::kMaxRanks * detail::kMaxGroupsPerRank) {
        fail("k exceeds 24 groups of 1024");
    }
    if (x.data == nullptr || x.dtype != DType::BF16 || !x.is_contiguous() || x.ne[0] != a.k || x.ne[1] <= 0 ||
        x.ne[2] != 1 || x.ne[3] != 1) {
        fail("x must be contiguous BF16 [k, T]");
    }
    if (u.data == nullptr || u.dtype != DType::BF16 || !u.is_contiguous() || u.ne[0] != a.rows || u.ne[1] != x.ne[1] ||
        u.ne[2] != 1 || u.ne[3] != 1) {
        fail("u must be contiguous BF16 [rows, T]");
    }
    // The tile schedule where its modelled time is the lower, else the split schedule below kStreamMinTokens columns
    // and the stream schedule from there (where it has the height and fits). Every schedule gives the same bits.
    // A base call faster than any tile (a tile's round and wave) skips the plan's host work. A captured call (a decode or
    // verification round: at most the lanes times the draft window, 64 columns at 8 lanes of 8) keeps the base
    // schedules, the faster ones at those widths in every measured shape (the tile's plan would query occupancy and
    // stage TMA descriptors inside the capture).
    const std::int32_t tokens = x.ne[1];
    const double base_us      = detail::base_model_us(a.rows, a.k, tokens);
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture == cudaStreamCaptureStatusNone && tokens >= detail::kTileMinTokens &&
        a.k >= 2 * kLoraShrinkChunk * kLoraShrinkGroup &&
        base_us > (detail::kTileRoundFloorUs + detail::kWaveOverheadUs) * detail::kTileMargin) {
        const detail::TilePlan plan = detail::tile_plan(a.rows, a.k, tokens);
        if (plan.bm > 0 && plan.us * detail::kTileMargin < base_us) {
            detail::launch_tile(x, a, u, stream, plan);
            return;
        }
    }
    if (x.ne[1] < detail::kStreamMinTokens) {
        detail::launch_split(x, a, u, stream);
        return;
    }
    const auto stream_or_split = [&]<int MTiles, int Slices>() {
        if (detail::stream_fits<MTiles>(a.k)) {
            detail::launch_stream<MTiles, Slices>(x, a, u, stream);
        } else {
            detail::launch_split(x, a, u, stream);
        }
    };
    switch (a.rows / 16) {
    case 2: return stream_or_split.template operator()<2, 1>();
    case 4: return stream_or_split.template operator()<4, 1>();
    case 6: return stream_or_split.template operator()<6, 1>();
    case 8: return stream_or_split.template operator()<8, 1>();
    case 12: return stream_or_split.template operator()<12, 1>();
    // Wide adapters' heights (ranks 80 and 112: 5 / 10 / 15 and 7 / 14 tiles) that fit one CTA. At rank 80 the split
    // schedule took 24.1 ms of a 2,048-token prefill's shrinks against rank 96's 20.5 (RTX 5090, 2026-11-21).
    case 5: return stream_or_split.template operator()<5, 1>();
    case 7: return stream_or_split.template operator()<7, 1>();
    case 10: return stream_or_split.template operator()<10, 1>();
    case 14: return stream_or_split.template operator()<14, 1>();
    case 15: return stream_or_split.template operator()<15, 1>();
    // The 16-, 18-, 21- and 24-tile stacks (gate/up at rank 128, q/k/v at 96, 112 and 128) exceed a CTA's ring: one
    // launch streams them in slices. Split, they ran at about 0.56 us a row at 2,048 tokens, against 0.41 streamed.
    case 16: return stream_or_split.template operator()<8, 2>();
    case 18: return stream_or_split.template operator()<6, 3>();
    case 21: return stream_or_split.template operator()<7, 3>();
    case 24: return stream_or_split.template operator()<12, 2>();
    default: break;
    }
    // Other heights split K at every width (the same bits).
    detail::launch_split(x, a, u, stream);
}

LoraAddend lora_row_range(const LoraAddend& addend, std::int32_t first_row, std::int32_t rows) {
    LoraAddend out = addend;
    out.b          = static_cast<const std::uint16_t*>(addend.b) + static_cast<std::int64_t>(first_row) * addend.rank;
    out.segments   = 0;
    out.segment    = {};
    for (int s = 0; s < addend.segments; ++s) {
        const LoraSegment& seg = addend.segment[s];
        if (seg.row_end <= first_row) { continue; }
        const std::int32_t end = std::min(seg.row_end, first_row + rows) - first_row;
        out.segment[out.segments++] = LoraSegment{end, seg.u_offset, seg.scale};
        if (end == rows) { break; }
    }
    return out;
}

void validate_lora_addend(const LoraAddend& addend, std::int32_t parent_rows, std::int32_t tokens) {
    const auto fail = [](const std::string& what) { throw std::invalid_argument("LoRA addend: " + what); };
    if (addend.b == nullptr || addend.u == nullptr) { fail("B and U are required"); }
    // The terms copy B's rows and U's columns in 16-byte pieces (8 values).
    if (reinterpret_cast<std::uintptr_t>(addend.b) % 16 != 0 || reinterpret_cast<std::uintptr_t>(addend.u) % 16 != 0) {
        fail("B and U must be 16-byte aligned");
    }
    if (addend.rank <= 0 || addend.rank % 16 != 0 || addend.rank > kLoraMaxRank) {
        fail("the padded rank must be a positive multiple of 16 up to " + std::to_string(kLoraMaxRank));
    }
    if (addend.u_rows <= 0 || addend.u_rows % 8 != 0) { fail("U's rows must be a positive multiple of 8"); }
    if (tokens <= 0) { fail("no columns"); }
    if (addend.segments <= 0 || addend.segments > 4) { fail("one to four segments"); }
    std::int32_t begin = 0;
    for (int s = 0; s < addend.segments; ++s) {
        const LoraSegment& seg = addend.segment[s];
        if (seg.row_end <= begin || seg.row_end % 16 != 0) { fail("segment ends must increase in steps of 16 rows"); }
        if (seg.u_offset >= 0 && (seg.u_offset + addend.rank > addend.u_rows)) { fail("a segment's U rows overrun U"); }
        if (seg.u_offset >= 0 && seg.u_offset % 8 != 0) { fail("a segment's U offset must be a multiple of 8"); }
        begin = seg.row_end;
    }
    if (begin != parent_rows) { fail("the segments must cover the parent's " + std::to_string(parent_rows) + " rows"); }
}

} // namespace infernix::ops
