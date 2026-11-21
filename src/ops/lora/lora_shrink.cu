// lora_shrink (include/infernix/ops/lora.h): U = A x under the canonical chunk / group arithmetic.
//
// Two schedules, the same bits:
// - Split (decode, verification and short prompts): a cluster of CTAs splits K by whole groups for one 16-row tile of
//   A and up to 32 columns. Each warp chains one 256-value chunk, every chunk of the rank at once (A's fragments
//   loaded before the PDL wait); the rank adds its groups' chunk partials in order and stores the group sums into
//   rank 0's shared memory, and rank 0 adds them in order after one cluster barrier. Many small CTAs stream A at full
//   bandwidth where a whole-K CTA would be limited by one SM's.
// - Stream (long prompts): one CTA takes every row of A over 32 columns and all of K, staging A and x through a
//   cp.async ring, and keeps the chunk, group and total sums in registers. x is read once.
// Both add in the same order: chunk partials from +0 (MMA chain), groups from their first chunk, the total from its
// first group.

#include "infernix/ops/lora.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
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
// Stream schedule (prompts). grid (ranks, ceil(T / 32)), cluster (ranks, 1, 1): rank r streams its own groups of K
// (group_begin) for every row of A over 32 columns, so x is read once in all and the call has ranks x T / 32 CTAs.
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

// From this many columns a call streams x once instead of splitting K: the split schedule reads x once per 16-row tile
// of A (from L2 after the first), the stream schedule A once per 32 columns. Measured 2026-11-21 (RTX 5090, the 27B's
// four sites at rank 32): the split is faster to 256 columns (q/k/v 13.7 against 17.0 us, down 16.6 against 23.1),
// the stream from 512 (q/k/v 17.3 against 20.8, down 23.8 against 29.0) and by 1.1-1.7x at 1,024-4,096.
constexpr std::int32_t kStreamMinTokens = 512;

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

template <int MTiles>
void launch_stream(const Tensor& x, const LoraShrinkWeight& a, Tensor& u, cudaStream_t stream) {
    constexpr auto kernel = lora_shrink_stream_kernel<MTiles>;
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
    config.gridDim          = dim3(static_cast<unsigned>(ranks), static_cast<unsigned>((tokens + kTokenTile - 1) / kTokenTile));
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
    if (x.ne[1] < detail::kStreamMinTokens) {
        detail::launch_split(x, a, u, stream);
        return;
    }
    const auto stream_or_split = [&]<int MTiles>() {
        if (detail::stream_fits<MTiles>(a.k)) {
            detail::launch_stream<MTiles>(x, a, u, stream);
        } else {
            detail::launch_split(x, a, u, stream);
        }
    };
    switch (a.rows / 16) {
    case 2: return stream_or_split.template operator()<2>();
    case 4: return stream_or_split.template operator()<4>();
    case 6: return stream_or_split.template operator()<6>();
    case 8: return stream_or_split.template operator()<8>();
    case 12: return stream_or_split.template operator()<12>();
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
