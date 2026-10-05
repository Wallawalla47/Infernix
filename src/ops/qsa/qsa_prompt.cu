// QSA prompt attention on Tensor Cores (see qsa_prompt.h).

#include "ops/qsa/qsa_prompt.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/qsa/page_spaces.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kD      = 256; // head dimension
constexpr int kTile   = 16;  // tokens per warp tile
constexpr int kRows   = 16;  // MMA rows: one KV head's query heads, padded
constexpr int kGroups = 4;   // INT8 G64 scale groups per row
constexpr int kQRow   = 2 * kD; // bytes of one FP16/BF16 query row in shared memory

template <KvCacheStorage Storage>
struct PromptShape;

// INT8: 4 KiB of K codes and 4 KiB of V codes per tile, with their G64 scales.
template <>
struct PromptShape<KvCacheStorage::Int8Group64> {
    static constexpr int Warps      = 4;
    static constexpr int RowBytes   = kD;
    static constexpr int TileBytes  = kTile * RowBytes;
    static constexpr int ScaleBytes = kTile * kGroups * 2;
    static constexpr int StageBytes = 2 * TileBytes + 2 * ScaleBytes;
};

// BF16 keys and FP16 values: 8 KiB each per tile.
template <>
struct PromptShape<KvCacheStorage::BFloat16> {
    static constexpr int Warps      = 2;
    static constexpr int RowBytes   = 2 * kD;
    static constexpr int TileBytes  = kTile * RowBytes;
    static constexpr int ScaleBytes = 0;
    static constexpr int StageBytes = 2 * TileBytes;
};

template <KvCacheStorage Storage>
constexpr int prompt_smem_bytes() {
    using Shape          = PromptShape<Storage>;
    constexpr int stages = Shape::Warps * 2 * Shape::StageBytes;
    constexpr int merge  = Shape::Warps * (kRows * kD + kRows * 2) * 4;
    return kQRow * kRows + (stages > merge ? stages : merge);
}

// 16-byte chunk c of row r lives at chunk c ^ (r & 7): the eight rows an ldmatrix reads hit
// distinct banks.
__device__ __forceinline__ int swizzled(int row, int row_bytes, int chunk) {
    return row * row_bytes + ((chunk ^ (row & 7)) << 4);
}

// The attended token at index i of a column's list (the decode kernel's order: selected blocks in
// ascending order, then the incomplete block's positions; every token 0..p while dense).
__device__ __forceinline__ int attended_token(int i, int count, const std::int32_t* list, int ratio, int p) {
    if (count < 0) { return i; }
    const int block_tokens = count * ratio;
    if (i < block_tokens) { return list[i / ratio] * ratio + i % ratio; }
    return (p + 1) / ratio * ratio + (i - block_tokens);
}

// INT8 rows: the k positions {2t, 2t+1 | 2t+8, 2t+9} of a 16-wide MMA k step hold dimensions
// {4t, 4t+1 | 4t+2, 4t+3}, which is what one ldmatrix lane receives of a code row (four
// consecutive bytes). The query rows are stored in that order, so the dot product is unchanged.
__device__ __forceinline__ int int8_query_position(int d) {
    const int local = d & 15;
    const int p     = ((local & 2) != 0 ? 8 : 0) + 2 * (local >> 2) + (local & 1);
    return (d & ~15) + p;
}

// Four INT8 codes (dimensions d..d+3 of one key) as the two FP16 B registers of a k step:
// {c0, c1} and {c2, c3}, widened exactly.
__device__ __forceinline__ void widen_codes(unsigned codes, unsigned& lo, unsigned& hi) {
    // code ^ 0x80 is code + 128; 0x6400 | byte is the FP16 value 1024 + byte.
    const unsigned biased = codes ^ 0x80808080u;
    const unsigned e      = __byte_perm(biased, 0x64646464u, 0x5140);
    const unsigned o      = __byte_perm(biased, 0x64646464u, 0x7362);
    const __half2 offset  = __float2half2_rn(1152.0f);
    const __half2 l       = __hsub2(load_vec<__half2>(&e), offset);
    const __half2 h       = __hsub2(load_vec<__half2>(&o), offset);
    lo                    = load_vec<unsigned>(&l);
    hi                    = load_vec<unsigned>(&h);
}

// One ldmatrix.trans b16 lane of an INT8 [key][d] tile holds {V[k][d], V[k][d+1], V[k+1][d],
// V[k+1][d+1]}: the FP16 B halves {V[k][d], V[k+1][d]} and {V[k][d+1], V[k+1][d+1]}, each code
// widened exactly and multiplied once by its key's group scale (`scales` = {s_k, s_k+1}).
__device__ __forceinline__ void decode_v_pair(unsigned codes, __half2 scales, unsigned& even, unsigned& odd) {
    const unsigned biased = codes ^ 0x80808080u;
    const unsigned e      = __byte_perm(biased, 0x64646464u, 0x4240);
    const unsigned o      = __byte_perm(biased, 0x64646464u, 0x4341);
    const __half2 offset  = __float2half2_rn(1152.0f);
    const __half2 ve      = __hmul2(__hsub2(load_vec<__half2>(&e), offset), scales);
    const __half2 vo      = __hmul2(__hsub2(load_vec<__half2>(&o), offset), scales);
    even                  = load_vec<unsigned>(&ve);
    odd                   = load_vec<unsigned>(&vo);
}

template <KvCacheStorage Storage>
__global__ void __launch_bounds__(PromptShape<Storage>::Warps * 32, 1)
    qsa_prompt_kernel(const bf16* __restrict__ q, int heads, int kv_heads, const void* __restrict__ k_pages,
                      const void* __restrict__ v_pages, const __half* __restrict__ k_scales,
                      const __half* __restrict__ v_scales, const std::int32_t* __restrict__ tables,
                      int table_stride, const std::int32_t* __restrict__ table_rows,
                      const std::int32_t* __restrict__ positions, int width, int ratio,
                      const std::int32_t* __restrict__ selected, const std::int32_t* __restrict__ counts,
                      int top_blocks, float scale, bf16* __restrict__ out, QsaPageSpaces spaces) {
    using Shape                 = PromptShape<Storage>;
    constexpr bool Int8         = Storage == KvCacheStorage::Int8Group64;
    constexpr int Warps         = Shape::Warps;
    constexpr int Threads       = Warps * 32;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    extern __shared__ __align__(16) unsigned char smem[];

    const int t = static_cast<int>(blockIdx.x), kv_head = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, lid = lane & 3;
    const int group = heads / kv_heads;
    const int p     = positions[t];
    const int count = counts[t];
    const int total = count < 0 ? p + 1 : count * ratio + (p + 1) % ratio;
    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[t / width]) * table_stride;
    const std::int32_t* list  = selected + static_cast<std::size_t>(t) * top_blocks;

    // Query rows (INT8: rotated into the keys' Hadamard domain, then FP16 in the k-step order;
    // BF16: the stored values). Rows past the group are zero.
    unsigned char* q_s = smem;
    for (int r = warp; r < kRows; r += Warps) {
        float v[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            v[k] = r < group ? __bfloat162float(
                                   q[(static_cast<std::size_t>(t) * heads + kv_head * group + r) * kD + lane + 32 * k])
                             : 0.0f;
        }
        if constexpr (Int8) { normalized_hadamard_d256_inplace(v, lane); }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int d = lane + 32 * k;
            const int e = Int8 ? int8_query_position(d) : d;
            unsigned char* at = q_s + swizzled(r, kQRow, e >> 3) + 2 * (e & 7);
            if constexpr (Int8) {
                *reinterpret_cast<__half*>(at) = __float2half_rn(v[k]);
            } else {
                *reinterpret_cast<bf16*>(at) = __float2bfloat16_rn(v[k]);
            }
        }
    }
    __syncthreads();

    unsigned char* stages = smem + kQRow * kRows + warp * 2 * Shape::StageBytes;
    const int tiles       = (total + kTile - 1) / kTile;

    // Gathers tile i's keys and values (and INT8 scales) into stage s; tokens past the column's
    // list are zero-filled.
    const auto issue = [&](int i, int s) {
        unsigned char* k_s = stages + s * Shape::StageBytes;
        unsigned char* v_s = k_s + Shape::TileBytes;
        const int first    = i * kTile;
        constexpr int Chunks = Shape::RowBytes / 16;
#pragma unroll
        for (int j = 0; j < kTile * Chunks / 32; ++j) {
            const int chunk  = lane + 32 * j;
            const int row    = chunk / Chunks, c = chunk % Chunks;
            const int index  = first + row;
            const bool valid = index < total;
            const int token  = valid ? attended_token(index, count, list, ratio, p) : 0;
            const int id     = table[token >> kPagedKVPageShift];
            int page;
            const void* k_plane = qsa_space_plane(spaces, k_pages, spaces.host_k, spaces.lent_k, id, page);
            const void* v_plane = qsa_space_plane(spaces, v_pages, spaces.host_v, spaces.lent_v, id, page);
            const std::size_t at =
                (static_cast<std::size_t>(kv_head + kv_heads * page) * kPagedKVPageSize + (token & kPagedKVPageMask)) *
                    Shape::RowBytes +
                16 * c;
            const int dst = swizzled(row, Shape::RowBytes, c);
            cp_async_zfill<16, Cache::cg>(k_s + dst, static_cast<const unsigned char*>(k_plane) + at, valid ? 16 : 0);
            cp_async_zfill<16, Cache::cg>(v_s + dst, static_cast<const unsigned char*>(v_plane) + at, valid ? 16 : 0);
        }
        if constexpr (Int8) {
            const int row    = lane & 15;
            const int index  = first + row;
            const bool valid = index < total;
            const int token  = valid ? attended_token(index, count, list, ratio, p) : 0;
            const int id     = table[token >> kPagedKVPageShift];
            int page;
            const auto* plane = static_cast<const __half*>(
                lane < 16 ? qsa_space_plane(spaces, k_scales, spaces.host_k_scale, spaces.lent_k_scale, id, page)
                          : qsa_space_plane(spaces, v_scales, spaces.host_v_scale, spaces.lent_v_scale, id, page));
            const std::size_t at =
                (static_cast<std::size_t>(kv_head + kv_heads * page) * kPagedKVPageSize + (token & kPagedKVPageMask)) *
                kGroups;
            __half* dst = reinterpret_cast<__half*>(v_s + Shape::TileBytes) + (lane < 16 ? 0 : kTile * kGroups) +
                          row * kGroups;
            cp_async_zfill<8>(dst, plane + at, valid ? 8 : 0);
        }
        cp_commit();
    };

    // The A fragments of k step kb of the query rows.
    const int a_mat = lane >> 3, a_rin = lane & 7;
    const int a_row = a_rin + ((a_mat & 1) << 3);
    const auto query_fragment = [&](int kb, unsigned (&a)[4]) {
        ldmatrix_x4(a[0], a[1], a[2], a[3], smem_addr(q_s + swizzled(a_row, kQRow, 2 * kb + (a_mat >> 1))));
    };

    // Output rows g (registers 0/1) and g + 8 (2/3) of 32 n8 tiles: INT8 tiles [db][0] and
    // [db][1] hold the even and odd dimensions of block db; BF16 tile [db][h] dimensions
    // 16 db + 8 h .. + 7.
    float o[16][2][4];
#pragma unroll
    for (int b = 0; b < 16; ++b) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { o[b][h][i] = 0.0f; }
        }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;
    const float scale_l2 = scale * Log2E;

    const auto compute = [&](int i, int s) {
        const unsigned char* k_s = stages + s * Shape::StageBytes;
        const unsigned char* v_s = k_s + Shape::TileBytes;
        float score[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int j = 0; j < 4; ++j) { score[nt][j] = 0.0f; }
        }
        if constexpr (Int8) {
            const __half* ks = reinterpret_cast<const __half*>(v_s + Shape::TileBytes);
#pragma unroll
            for (int g = 0; g < kGroups; ++g) {
                float acc[2][4];
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) { acc[nt][j] = 0.0f; }
                }
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int kb = g * 4 + kk;
                    unsigned a[4];
                    query_fragment(kb, a);
                    unsigned r0, r1;
                    ldmatrix_x2(r0, r1, smem_addr(k_s + swizzled(lane & 15, Shape::RowBytes, kb)));
                    unsigned lo, hi;
                    widen_codes(r0, lo, hi);
                    mma_f16(acc[0][0], acc[0][1], acc[0][2], acc[0][3], a[0], a[1], a[2], a[3], lo, hi);
                    widen_codes(r1, lo, hi);
                    mma_f16(acc[1][0], acc[1][1], acc[1][2], acc[1][3], a[0], a[1], a[2], a[3], lo, hi);
                }
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int key   = nt * 8 + 2 * lid;
                    const float ks0 = __half2float(ks[key * kGroups + g]);
                    const float ks1 = __half2float(ks[(key + 1) * kGroups + g]);
                    score[nt][0]    = __fmaf_rn(ks0, acc[nt][0], score[nt][0]);
                    score[nt][1]    = __fmaf_rn(ks1, acc[nt][1], score[nt][1]);
                    score[nt][2]    = __fmaf_rn(ks0, acc[nt][2], score[nt][2]);
                    score[nt][3]    = __fmaf_rn(ks1, acc[nt][3], score[nt][3]);
                }
            }
        } else {
            const int b_mat = lane >> 3, b_rin = lane & 7;
            const int b_row = b_rin + ((b_mat >> 1) << 3);
#pragma unroll
            for (int kb = 0; kb < 16; ++kb) {
                unsigned a[4];
                query_fragment(kb, a);
                unsigned b[4];
                ldmatrix_x4(b[0], b[1], b[2], b[3],
                            smem_addr(k_s + swizzled(b_row, Shape::RowBytes, 2 * kb + (b_mat & 1))));
                mma_bf16(score[0][0], score[0][1], score[0][2], score[0][3], a[0], a[1], a[2], a[3], b[0], b[1]);
                mma_bf16(score[1][0], score[1][1], score[1][2], score[1][3], a[0], a[1], a[2], a[3], b[2], b[3]);
            }
        }

        // Online softmax over the tile's 16 tokens (columns past the list are masked).
        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int index = i * kTile + nt * 8 + 2 * lid;
            if (index >= total) { score[nt][0] = score[nt][2] = -CUDART_INF_F; }
            if (index + 1 >= total) { score[nt][1] = score[nt][3] = -CUDART_INF_F; }
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);
        const float nm0 = fmaxf(m0, bm0), nm1 = fmaxf(m1, bm1);
        const float alpha0 = m0 == -CUDART_INF_F ? 0.0f : exp2_approx((m0 - nm0) * scale_l2);
        const float alpha1 = m1 == -CUDART_INF_F ? 0.0f : exp2_approx((m1 - nm1) * scale_l2);
        m0 = nm0;
        m1 = nm1;
        const float base0 = nm0 * scale_l2, base1 = nm1 * scale_l2;
        float pr[2][4];
        float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            pr[nt][0] = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -base0));
            pr[nt][1] = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -base0));
            pr[nt][2] = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -base1));
            pr[nt][3] = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -base1));
            bl0 += pr[nt][0] + pr[nt][1];
            bl1 += pr[nt][2] + pr[nt][3];
        }
        l0 = __fmaf_rn(l0, alpha0, bl0);
        l1 = __fmaf_rn(l1, alpha1, bl1);
#pragma unroll
        for (int b = 0; b < 16; ++b) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                o[b][h][0] *= alpha0;
                o[b][h][1] *= alpha0;
                o[b][h][2] *= alpha1;
                o[b][h][3] *= alpha1;
            }
        }

        if constexpr (Int8) {
            // Decoded values stay finite in FP16: a tile whose largest scale exceeds 256 decodes
            // with its scales divided by 2^shift and multiplies the probabilities by 2^shift
            // (both exact, so every product is unchanged).
            const __half* vs = reinterpret_cast<const __half*>(v_s + Shape::TileBytes) + kTile * kGroups;
            const __half2 own = load_vec<__half2>(&vs[2 * lane]);
            const float vmax  = warp_max(fmaxf(fabsf(__low2float(own)), fabsf(__high2float(own))), FullMask);
            int shift         = 0;
            while (shift < 14 && ldexpf(vmax, -shift) > 256.0f) { ++shift; }
            const float up = ldexpf(1.0f, shift);
            unsigned pa[4];
            pa[0] = pack_f16x2(pr[0][0] * up, pr[0][1] * up);
            pa[1] = pack_f16x2(pr[0][2] * up, pr[0][3] * up);
            pa[2] = pack_f16x2(pr[1][0] * up, pr[1][1] * up);
            pa[3] = pack_f16x2(pr[1][2] * up, pr[1][3] * up);
            const __half2 down = __float2half2_rn(ldexpf(1.0f, -shift));
            const int v_mat = lane >> 3, v_rin = lane & 7;
            const int v_row = v_rin + ((v_mat & 1) << 3);
#pragma unroll
            for (int db = 0; db < 16; db += 2) {
                unsigned r[4];
                ldmatrix_x4_t(r[0], r[1], r[2], r[3],
                              smem_addr(v_s + swizzled(v_row, Shape::RowBytes, db + (v_mat >> 1))));
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int g       = (db + h) >> 2;
                    const int key     = 2 * lid;
                    const __half2 s01 = __hmul2(__halves2half2(vs[key * kGroups + g], vs[(key + 1) * kGroups + g]), down);
                    const __half2 s89 =
                        __hmul2(__halves2half2(vs[(key + 8) * kGroups + g], vs[(key + 9) * kGroups + g]), down);
                    unsigned e0, o0, e1, o1;
                    decode_v_pair(r[2 * h], s01, e0, o0);
                    decode_v_pair(r[2 * h + 1], s89, e1, o1);
                    float* even = o[db + h][0];
                    float* odd  = o[db + h][1];
                    mma_f16(even[0], even[1], even[2], even[3], pa[0], pa[1], pa[2], pa[3], e0, e1);
                    mma_f16(odd[0], odd[1], odd[2], odd[3], pa[0], pa[1], pa[2], pa[3], o0, o1);
                }
            }
        } else {
            unsigned pa[4];
            pa[0] = pack_f16x2(pr[0][0], pr[0][1]);
            pa[1] = pack_f16x2(pr[0][2], pr[0][3]);
            pa[2] = pack_f16x2(pr[1][0], pr[1][1]);
            pa[3] = pack_f16x2(pr[1][2], pr[1][3]);
            const int v_mat = lane >> 3, v_rin = lane & 7;
            const int v_row = v_rin + ((v_mat & 1) << 3);
#pragma unroll
            for (int c = 0; c < 32; c += 2) {
                unsigned r[4];
                ldmatrix_x4_t(r[0], r[1], r[2], r[3],
                              smem_addr(v_s + swizzled(v_row, Shape::RowBytes, c + (v_mat >> 1))));
                float* first  = o[c >> 1][0];
                float* second = o[c >> 1][1];
                mma_f16(first[0], first[1], first[2], first[3], pa[0], pa[1], pa[2], pa[3], r[0], r[1]);
                mma_f16(second[0], second[1], second[2], second[3], pa[0], pa[1], pa[2], pa[3], r[2], r[3]);
            }
        }
    };

    int s = 0;
    if (warp < tiles) { issue(warp, 0); }
    for (int i = warp; i < tiles; i += Warps) {
        if (i + Warps < tiles) {
            issue(i + Warps, s ^ 1);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncwarp();
        compute(i, s);
        __syncwarp();
        s ^= 1;
    }
    cp_wait<0>();

    // Merge the warps' rows: each publishes its unnormalized rows and (max, sum).
    l0 += __shfl_xor_sync(FullMask, l0, 1);
    l0 += __shfl_xor_sync(FullMask, l0, 2);
    l1 += __shfl_xor_sync(FullMask, l1, 1);
    l1 += __shfl_xor_sync(FullMask, l1, 2);
    __syncthreads();
    float* rows  = reinterpret_cast<float*>(smem + kQRow * kRows);
    float* stats = rows + Warps * kRows * kD;
    float* mine  = rows + warp * kRows * kD;
#pragma unroll
    for (int b = 0; b < 16; ++b) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int row = gid + 8 * (j >> 1);
                const int d   = Int8 ? 16 * b + 4 * lid + 2 * (j & 1) + h : 16 * b + 8 * h + 2 * lid + (j & 1);
                mine[row * kD + d] = o[b][h][j];
            }
        }
    }
    if (lid == 0) {
        stats[(warp * kRows + gid) * 2]         = m0;
        stats[(warp * kRows + gid) * 2 + 1]     = l0;
        stats[(warp * kRows + gid + 8) * 2]     = m1;
        stats[(warp * kRows + gid + 8) * 2 + 1] = l1;
    }
    __syncthreads();
    for (int e = tid; e < group * kD; e += Threads) {
        const int row = e / kD, d = e % kD;
        float m = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < Warps; ++w) { m = fmaxf(m, stats[(w * kRows + row) * 2]); }
        float l = 0.0f, acc = 0.0f;
#pragma unroll
        for (int w = 0; w < Warps; ++w) {
            const float mw = stats[(w * kRows + row) * 2];
            if (mw == -CUDART_INF_F) { continue; }
            const float f = exp2_approx((mw - m) * scale_l2);
            l             = __fmaf_rn(f, stats[(w * kRows + row) * 2 + 1], l);
            acc           = __fmaf_rn(f, rows[(w * kRows + row) * kD + d], acc);
        }
        out[(static_cast<std::size_t>(t) * heads + kv_head * group + row) * kD + d] =
            __float2bfloat16_rn(l > 0.0f ? acc / l : 0.0f);
    }
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("qsa prompt: ") + message); }
}

template <KvCacheStorage Storage>
void launch(const Tensor& q, const QsaKVLayer& layer, const QsaBatch& batch, const QsaGeometry& g, float scale,
            const std::int32_t* selected, const std::int32_t* counts, Tensor& out, cudaStream_t stream) {
    constexpr int bytes = prompt_smem_bytes<Storage>();
    static const cudaError_t configured =
        cudaFuncSetAttribute(qsa_prompt_kernel<Storage>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    require(configured == cudaSuccess, "cannot reserve the kernel's shared memory");
    const int columns = q.ne[2];
    const dim3 grid(static_cast<unsigned>(columns), static_cast<unsigned>(g.kv_heads));
    qsa_prompt_kernel<Storage><<<grid, PromptShape<Storage>::Warps * 32, bytes, stream>>>(
        static_cast<const bf16*>(q.data), g.heads, g.kv_heads, layer.kv.k_pages.data, layer.kv.v_pages.data,
        static_cast<const __half*>(layer.kv.k_scale_pages.data), static_cast<const __half*>(layer.kv.v_scale_pages.data),
        static_cast<const std::int32_t*>(batch.block_tables.data), batch.block_tables.ne[0],
        static_cast<const std::int32_t*>(batch.table_rows.data), static_cast<const std::int32_t*>(batch.positions.data),
        batch.width, g.ratio, selected, counts, g.budget / g.ratio, scale, static_cast<bf16*>(out.data), layer.spaces);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("qsa prompt attention: ") + cudaGetErrorString(error)); }
}

} // namespace

bool qsa_prompt_supported(const QsaKVLayer& layer, const QsaGeometry& geometry) {
    return geometry.head_dim == kD && geometry.kv_heads > 0 &&
           geometry.heads / geometry.kv_heads <= kRows &&
           (layer.kv.storage == KvCacheStorage::Int8Group64 || layer.kv.storage == KvCacheStorage::BFloat16);
}

void qsa_prompt_attention(const Tensor& q, const QsaKVLayer& layer, const QsaBatch& batch,
                          const QsaGeometry& geometry, float scale, const std::int32_t* selected,
                          const std::int32_t* counts, Tensor& out, cudaStream_t stream) {
    require(qsa_prompt_supported(layer, geometry), "the prompt route does not serve this layer");
    if (layer.kv.storage == KvCacheStorage::Int8Group64) {
        launch<KvCacheStorage::Int8Group64>(q, layer, batch, geometry, scale, selected, counts, out, stream);
    } else {
        launch<KvCacheStorage::BFloat16>(q, layer, batch, geometry, scale, selected, counts, out, stream);
    }
}

} // namespace ninfer::ops::detail
