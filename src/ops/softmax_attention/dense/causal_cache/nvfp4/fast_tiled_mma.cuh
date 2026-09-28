#pragma once

// Fast causal prompt kernel for the NVFP4-G16 cache. It serves prompt-route launches whose
// CausalAttentionExecutionEnvelope::fast_prompt_kernel is set over more than 2048 visible keys; the
// model sets it unless --use-original-nvfp4-prefill-kernel asks for the tiled kernel (tiled_mma.cuh).
//
//   * Each warp owns 16 query rows of one head for the whole key sweep, so scores, probabilities
//     and the D256 output accumulator never leave registers. The only CTA barrier is the one that
//     publishes a key tile. The wide CTA has eight warps; four-warp CTAs serve launches too narrow
//     for wide ones.
//   * Q receives the fixed D256 rotation. The stored K codes and group scales feed the
//     block-scaled FP4 m16n8k64 MMA unchanged, and Q enters as two NVFP4 terms (q_terms.cuh), one
//     MMA per term; an FP32 row factor restores the Q scale. QK accumulates in FP32.
//   * One tile is one 64-key page: K codes, packed NVFP4 V codes and both scale planes are
//     double-buffered raw. V is decoded in registers: ldmatrix.trans over the packed rows hands a
//     lane two adjacent keys by four adjacent dimensions, which after the exact E2M1 widening and
//     one multiply by the two keys' represented group scales is one FP16 B fragment for each of
//     four n8 tiles. Every E2M1 code times a legal UE4M3 scale is exact in FP16.
//   * PV runs FP16 Tensor Cores with FP16 accumulation over the 64 keys of a tile, promoted into
//     the FP32 output accumulator once per tile. Every probability is at most one, so a partial is
//     bounded by 64 * 6 * max_scale; a tile whose largest V scale exceeds 128 decodes V with its
//     scales divided by an exact power of two and multiplies the partial back.
//   * The register decode permutes the output dimensions across n8 tiles. The inverse rotation is
//     applied in that layout as one butterfly per index bit (two of them across the lane quad),
//     then each lane stores four contiguous BF16 dimensions.
//   * CTAs are issued longest-first so a causal prompt's heaviest row blocks do not form the tail.
//   * A split launch (gridDim.z > 1) gives each CTA a contiguous run of key pages for launches
//     whose row blocks alone would leave SMs idle. It publishes the normalized, inverse-rotated
//     FP32 row and its (max, sum) statistics; causal_attention_prompt_rotated_fast_merge_kernel
//     combines the splits.

#include "ops/common/math.cuh"
#include "ops/common/math.h"
#include "ops/common/mma.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/softmax_attention/common/causal_geometry.h"
#include "ops/softmax_attention/dense/causal_cache/fast_prompt_common.cuh"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/q_terms.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

inline constexpr int kCausalPromptRotatedBc          = 64;
inline constexpr int kCausalPromptRotatedVBytes      = kCausalPromptRotatedBc * 128;
inline constexpr int kCausalPromptRotatedVScaleBytes = kCausalPromptRotatedBc * 16;

// One key tile: K codes, V codes, K scales and V scales of one 64-key page, back to back.
struct CausalPromptRotatedStage {
    static constexpr int KRowBytes    = 128;
    static constexpr int KBytes       = kCausalPromptRotatedBc * KRowBytes;
    static constexpr int KScaleBytes  = kCausalPromptRotatedBc * 16;
    static constexpr int VOffset      = KBytes;
    static constexpr int KScaleOffset = VOffset + kCausalPromptRotatedVBytes;
    static constexpr int VScaleOffset = KScaleOffset + KScaleBytes;
    static constexpr int Bytes        = VScaleOffset + kCausalPromptRotatedVScaleBytes;
};

// Each warp owns 16 query rows: the codes of both Q terms (term t, row r at (t * Br + r) * 128)
// followed by their group scales ((t * Br + r) * 16).
template <int Warps>
struct CausalPromptRotatedShape {
    using Stage = CausalPromptRotatedStage;
    static_assert(Warps == 4 || Warps == 8);
    static constexpr int Threads     = Warps * 32;
    static constexpr int Br          = Warps * 16;
    static constexpr int QBytes      = Br * kCausalPromptHeadDim;
    static constexpr int QScaleBytes = 2 * Br * kKVCacheNvfp4Groups;
    static constexpr int SmemBytes   = QBytes + QScaleBytes + 2 * Stage::Bytes;
    static_assert(SmemBytes <= 101376);
};

// A 64-key FP16 partial is bounded by 64 * 6 * max_scale (every probability is at most one).
// Keeping max_scale at or below 128 leaves that bound, with FP16 rounding slack, under 65504.
// UE4M3 codes order like their values, and 0x70 encodes exactly 128.
inline constexpr std::uint32_t kCausalPromptRotatedScaleLimitCode = 0x70;

static_assert(kCausalPromptRotatedBc == kPagedKVPageSize);
static_assert(CausalPromptRotatedShape<8>::SmemBytes == 73728);

__device__ __forceinline__ void causal_prompt_rotated_fast_mma_f16_acc(unsigned& c0, unsigned& c1,
                                                                       unsigned a0, unsigned a1,
                                                                       unsigned a2, unsigned a3,
                                                                       unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Two E2M1 codes (low nibble first) widened exactly to FP16 and scaled by an exact FP16 pair.
__device__ __forceinline__ unsigned causal_prompt_rotated_fast_widen(unsigned byte, __half2 scale) {
    __nv_fp4x2_e2m1 encoded;
    encoded.__x         = static_cast<__nv_fp4x2_storage_t>(byte & 0xFFu);
    const __half2 value = __hmul2(static_cast<__half2>(encoded), scale);
    return *reinterpret_cast<const unsigned*>(&value);
}

// Two UE4M3 scale bytes (low byte first) as their exact FP16 pair.
__device__ __forceinline__ __half2 causal_prompt_rotated_fast_scale_pair(unsigned bytes) {
    __nv_fp8x2_e4m3 encoded;
    encoded.__x = static_cast<__nv_fp8x2_storage_t>(bytes & 0xFFFFu);
    return static_cast<__half2>(encoded);
}

// One ldmatrix.trans lane word: key a's four codes of one b16 unit (low half) and key b's
// (high half). Returns, for the unit's dimensions 0..3, the FP16 B-fragment pairs {a, b}.
__device__ __forceinline__ void causal_prompt_rotated_fast_decode_unit(unsigned word, __half2 scale,
                                                                       unsigned (&pairs)[4]) {
    const unsigned even = word & 0x0F0F0F0Fu;        // dims 0 and 2 of both keys
    const unsigned odd  = (word >> 4) & 0x0F0F0F0Fu; // dims 1 and 3 of both keys
    const unsigned e    = even | (even >> 12);       // byte 0: dim 0 pair, byte 1: dim 2 pair
    const unsigned o    = odd | (odd >> 12);         // byte 0: dim 1 pair, byte 1: dim 3 pair
    pairs[0]            = causal_prompt_rotated_fast_widen(e, scale);
    pairs[1]            = causal_prompt_rotated_fast_widen(o, scale);
    pairs[2]            = causal_prompt_rotated_fast_widen(e >> 8, scale);
    pairs[3]            = causal_prompt_rotated_fast_widen(o >> 8, scale);
}

// One butterfly of the Sylvester transform over an index bit held in the lane quad.
__device__ __forceinline__ float
causal_prompt_rotated_fast_quad_butterfly(float value, int lane_bit, int stride) {
    const float peer = __shfl_xor_sync(0xffffffffu, value, stride);
    return lane_bit == 0 ? __fadd_rn(value, peer) : __fsub_rn(peer, value);
}

// Split partial layout for key split s, column c and query head h: the D256 row at
// ((s * width + c) * QHeads + h) * 256 and its (max, sum) pair at twice that row index.
template <typename Geometry>
__host__ __device__ __forceinline__ std::int64_t
causal_prompt_rotated_fast_partial_row(int split, int column, int q_head, int width) {
    return (static_cast<std::int64_t>(split) * width + column) * Geometry::QHeads + q_head;
}

template <typename Geometry, typename Metadata, int Warps, bool Split>
__global__ __launch_bounds__(
    CausalPromptRotatedShape<Warps>::Threads,
    1) void causal_attention_prompt_rotated_fast_kernel(const __nv_bfloat16* __restrict__ q,
                                                        const std::uint8_t* __restrict__ cache_k,
                                                        const std::uint8_t* __restrict__ cache_v,
                                                        const std::
                                                            uint8_t* __restrict__ cache_k_scale,
                                                        const std::
                                                            uint8_t* __restrict__ cache_v_scale,
                                                        Metadata metadata,
                                                        const std::int32_t* __restrict__ positions,
                                                        float scale,
                                                        __nv_bfloat16* __restrict__ out,
                                                        std::int32_t width,
                                                        float* __restrict__ partial_rows,
                                                        float2* __restrict__ partial_stats) {
    constexpr int D             = kCausalPromptHeadDim;
    constexpr int DB16          = D / 2;
    using Shape                 = CausalPromptRotatedShape<Warps>;
    using Stage                 = typename Shape::Stage;
    constexpr int Threads       = Shape::Threads;
    constexpr int Br            = Shape::Br;
    constexpr int Bc            = kCausalPromptRotatedBc;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr int Units         = 4; // 64-dimension output blocks
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::uint8_t* q_fp8    = reinterpret_cast<std::uint8_t*>(smem_raw);
    std::uint8_t* q_scales = smem_raw + Shape::QBytes;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    // Longest-first issue order: the first QHeads CTAs take the last row block of every head.
    const int row_blocks = static_cast<int>(gridDim.x);
    const int linear     = static_cast<int>(blockIdx.x + blockIdx.y * gridDim.x);
    const int q_head     = linear % Geometry::QHeads;
    const int q0         = (row_blocks - 1 - linear / Geometry::QHeads) * Br;
    const int kv_head    = q_head / Geometry::GroupSize;
    const int tokens     = metadata.valid_tokens(width);

    // Rows past the valid token count are inactive columns and publish zeros.
    const auto store_row = [&](int row, int d0, float v0, float v1, float v2, float v3) {
        const int token = q0 + row;
        if (token >= width) { return; }
        const bool valid   = token < tokens;
        const uint2 packed = make_uint2(pack_bf16x2(valid ? v0 : 0.0f, valid ? v1 : 0.0f),
                                        pack_bf16x2(valid ? v2 : 0.0f, valid ? v3 : 0.0f));
        store_vec(&out[causal_prompt_q_index<Geometry>(q_head, d0, token)], packed);
    };

    if (q0 >= tokens) {
        // The merge publishes a split launch's inactive columns.
        if constexpr (!Split) {
            for (int element = tid; element < Br * (D / 4); element += Threads) {
                const int row = element / (D / 4);
                store_row(row, (element - row * (D / 4)) * 4, 0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        return;
    }

    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int rows                  = min(Br, tokens - q0);
    const int max_query_abs         = base_pos + q0 + rows - 1;
    const int key_blocks            = max_query_abs / Bc + 1;
    // Key pages [kb_begin, kb_end) of this split; an empty run publishes neutral statistics.
    const int split           = Split ? static_cast<int>(blockIdx.z) : 0;
    const int pages_per_split = Split ? div_up(key_blocks, static_cast<int>(gridDim.z)) : 0;
    const int kb_begin        = Split ? min(key_blocks, split * pages_per_split) : 0;
    const int kb_end          = Split ? min(key_blocks, kb_begin + pages_per_split) : key_blocks;

    // Each warp rotates and encodes its own 16 rows, keeping the scales of the two rows each lane
    // owns in the MMA C layout.
    float q_scale_r[2] = {0.0f, 0.0f};
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int row    = warp * 16 + r;
        const bool valid = row < rows;
        float values[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            values[k] =
                valid ? __bfloat162float(
                            q[causal_prompt_q_index<Geometry>(q_head, lane + 32 * k, q0 + row)])
                      : 0.0f;
        }
        normalized_hadamard_d256_inplace(values, lane);
        float local_absmax = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) local_absmax = fmaxf(local_absmax, fabsf(values[k]));
        const float absmax = warp_max(local_absmax, FullMask);
        const float factor  = absmax > 0.0f ? absmax / kCausalNvfp4QTop : 0.0f;
        const float inverse = absmax > 0.0f ? kCausalNvfp4QTop / absmax : 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float x     = values[k] * inverse;
            const float first = causal_nvfp4_q_term(x, lane, row, k, q_fp8 + row * (D / 2),
                                                    q_scales + row * kKVCacheNvfp4Groups);
            (void)causal_nvfp4_q_term(x - first, lane, row, k, q_fp8 + (Br + row) * (D / 2),
                                      q_scales + (Br + row) * kKVCacheNvfp4Groups);
        }
        if (gid == (r & 7)) { q_scale_r[r >> 3] = factor; }
    }

    const auto stage_base = [&](int stage) {
        return smem_raw + Shape::QBytes + Shape::QScaleBytes + stage * Stage::Bytes;
    };

    // One tile is one physical page of this KV head. Keys past the CTA's last visible key are
    // zero-filled so masked columns stay finite.
    const auto issue_tile = [&](int kb) {
        unsigned char* base       = stage_base(kb & 1);
        const int page            = block_table[kb];
        const int valid           = min(Bc, max_query_abs + 1 - kb * Bc);
        const std::int64_t k_base = kv_cache_nvfp4_code_index<Geometry>(page, kv_head, 0, 0);
        const std::int64_t v_base = kv_cache_nvfp4_code_index<Geometry>(page, kv_head, 0, 0);
        constexpr int KChunks     = Stage::KRowBytes / 16;
#pragma unroll
        for (int i = 0; i < div_up(Stage::KBytes / 16, Threads); ++i) {
            const int chunk = tid + i * Threads;
            if (chunk >= Stage::KBytes / 16) { break; }
            const int key = chunk / KChunks;
            const int c   = chunk - key * KChunks;
            cp_async_zfill<16, Cache::cg>(base + key * Stage::KRowBytes + ((c ^ (key & 7)) << 4),
                                          cache_k + k_base + key * Stage::KRowBytes + c * 16,
                                          key < valid ? 16 : 0);
        }
#pragma unroll
        for (int i = 0; i < div_up(kCausalPromptRotatedVBytes / 16, Threads); ++i) {
            const int chunk = tid + i * Threads;
            if (chunk >= kCausalPromptRotatedVBytes / 16) { break; }
            const int key = chunk >> 3;
            const int c   = chunk & 7;
            cp_async_zfill<16, Cache::cg>(
                base + Stage::VOffset + key * 128 + ((c ^ (key & 7)) << 4),
                cache_v + v_base + key * 128 + c * 16, key < valid ? 16 : 0);
        }
        constexpr int KScaleChunks = Stage::KScaleBytes / 16;
        for (int chunk = tid; chunk < KScaleChunks + Bc; chunk += Threads) {
            if (chunk < KScaleChunks) {
                cp_async_zfill<16, Cache::cg>(
                    base + Stage::KScaleOffset + chunk * 16,
                    cache_k_scale + kv_cache_nvfp4_scale_index<Geometry>(page, kv_head, 0, chunk),
                    chunk < valid ? 16 : 0);
            } else {
                const int key = chunk - KScaleChunks;
                cp_async_zfill<16, Cache::cg>(
                    base + Stage::VScaleOffset + key * 16,
                    cache_v_scale + kv_cache_nvfp4_scale_index<Geometry>(page, kv_head, 0, key),
                    key < valid ? 16 : 0);
            }
        }
        cp_commit();
    };

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int row_base = warp * 16;

    // Rows this warp owns, for warp-uniform tile skipping and mask elision.
    const bool warp_active  = row_base < rows;
    const int warp_min_qabs = base_pos + q0 + row_base;
    const int warp_max_qabs = base_pos + q0 + min(row_base + 15, rows - 1);
    const int row0          = row_base + gid;
    const int row1          = row0 + 8;
    const int qabs0         = row0 < rows ? base_pos + q0 + row0 : -1;
    const int qabs1         = row1 < rows ? base_pos + q0 + row1 : -1;

    // acc[u][h][i] is n8 tile (u, h, i): column n holds dimension 64u + 32h + 4n + i.
    float acc[Units][2][4][4];
#pragma unroll
    for (int u = 0; u < Units; ++u) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int e = 0; e < 4; ++e) { acc[u][h][i][e] = 0.0f; }
            }
        }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    const float scale_l2 = scale * Log2E;

    // What QK hands to PV for one tile: the probability A fragments, the row rescale factors, and
    // the exact power of two that keeps the tile's FP16 partials representable.
    unsigned pa[PVKs][4];
    float tile_alpha0 = 0.0f;
    float tile_alpha1 = 0.0f;
    int tile_shift    = 0;
    bool tile_live    = false;

    const auto qk_softmax = [&](int kb) {
        tile_live    = false;
        const int k0 = kb * Bc;
        if (!warp_active || k0 > warp_max_qabs) { return; }
        const unsigned char* base = stage_base(kb & 1);
        const std::uint8_t* vs_s  = base + Stage::VScaleOffset;

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
        // One block-scaled m16n8k64 MMA per 64-dimension slab, Q term and key tile. Lane rows
        // follow the A4 linear route; the scale words are the slab's four UE4M3 group scales
        // of the lane's A row ((lane & 1) * 8 + lane / 4) and B key (lane / 4).
        const std::uint8_t* k_s = base;
        const auto* k_scale_words =
            reinterpret_cast<const unsigned*>(base + Stage::KScaleOffset);
        const auto* q_scale_words = reinterpret_cast<const unsigned*>(q_scales);
        const int a_row           = row_base + a_rowoff;
        const int sfa_row         = row_base + (((lane & 1) << 3) | gid);
#pragma unroll
        for (int slab = 0; slab < D / 64; ++slab) {
            unsigned af[2][4];
            unsigned sfa[2];
#pragma unroll
            for (int term = 0; term < 2; ++term) {
                const int chunk = 2 * slab + (a_mat >> 1);
                ldmatrix_x4(af[term][0], af[term][1], af[term][2], af[term][3],
                            smem_addr(q_fp8 + (term * Br + a_row) * (D / 2) +
                                      ((chunk ^ (a_row & 7)) << 4)));
                sfa[term] = q_scale_words[(term * Br + sfa_row) * 4 + slab];
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; nt += 2) {
                const int key   = (nt + (lane >> 4)) * 8 + a_rin;
                const int chunk = 2 * slab + ((lane >> 3) & 1);
                unsigned bf[4];
                ldmatrix_x4(
                    bf[0], bf[1], bf[2], bf[3],
                    smem_addr(k_s + key * Stage::KRowBytes + ((chunk ^ (key & 7)) << 4)));
                const unsigned sfb0 = k_scale_words[(nt * 8 + gid) * 4 + slab];
                const unsigned sfb1 = k_scale_words[((nt + 1) * 8 + gid) * 4 + slab];
#pragma unroll
                for (int term = 0; term < 2; ++term) {
                    mma_nvfp4_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3],
                                   af[term][0], af[term][1], af[term][2], af[term][3], bf[0],
                                   bf[1], sfa[term], sfb0);
                    mma_nvfp4_e4m3(score[nt + 1][0], score[nt + 1][1], score[nt + 1][2],
                                   score[nt + 1][3], af[term][0], af[term][1], af[term][2],
                                   af[term][3], bf[2], bf[3], sfa[term], sfb1);
                }
            }
        }
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] *= q_scale_r[0];
            score[nt][1] *= q_scale_r[0];
            score[nt][2] *= q_scale_r[1];
            score[nt][3] *= q_scale_r[1];
        }

        const bool full_tile = k0 + Bc - 1 <= warp_min_qabs;
        float bm0            = -CUDART_INF_F;
        float bm1            = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            if (!full_tile) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                score[nt][0]   = key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                score[nt][2]   = key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                score[nt][3]   = key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
            }
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0        = fmaxf(running_m0, bm0);
        const float nm1        = fmaxf(running_m1, bm1);
        const float nm0_scaled = nm0 == -CUDART_INF_F ? 0.0f : nm0 * scale_l2;
        const float nm1_scaled = nm1 == -CUDART_INF_F ? 0.0f : nm1 * scale_l2;
        tile_alpha0            = running_m0 == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
        tile_alpha1            = running_m1 == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
        running_m0             = nm0;
        running_m1             = nm1;

        // Probabilities become the PV A fragments directly: k-step j covers keys 16j..16j+15,
        // which are score tiles 2j (keys 2t, 2t+1) and 2j+1 (keys 8+2t, 9+2t).
        float bl0 = 0.0f;
        float bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
            const float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
            const float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
            const float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            pa[nt >> 1][(nt & 1) * 2 + 0] = pack_f16x2(p00, p01);
            pa[nt >> 1][(nt & 1) * 2 + 1] = pack_f16x2(p10, p11);
        }
        // Row sums stay lane-partial; the quad is reduced once after the sweep.
        running_l0 = __fmaf_rn(running_l0, tile_alpha0, bl0);
        running_l1 = __fmaf_rn(running_l1, tile_alpha1, bl1);

        static_assert(kCausalPromptRotatedVScaleBytes == 32 * 32);
        const uint4 s0  = load_vec<uint4>(vs_s + 32 * lane);
        const uint4 s1  = load_vec<uint4>(vs_s + 32 * lane + 16);
        unsigned code   = 0;
        const auto fold = [&](unsigned w) {
            code =
                max(code, max(max(w & 0xFFu, (w >> 8) & 0xFFu), max((w >> 16) & 0xFFu, w >> 24)));
        };
        fold(s0.x);
        fold(s0.y);
        fold(s0.z);
        fold(s0.w);
        fold(s1.x);
        fold(s1.y);
        fold(s1.z);
        fold(s1.w);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            code = max(code, __shfl_xor_sync(FullMask, code, offset));
        }
        // Scales above 128 are (128, 256) or [256, 448]: one or two halvings bring them to 128.
        tile_shift = code <= kCausalPromptRotatedScaleLimitCode ? 0 : (code < 0x78u ? 1 : 2);
        tile_live  = true;
    };

    const auto pv = [&](int kb) {
        if (!tile_live) { return; }
        const unsigned char* base = stage_base(kb & 1);
        const std::uint8_t* v_s   = base + Stage::VOffset;
        const std::uint8_t* vs_s  = base + Stage::VScaleOffset;
        const __half2 mul         = __float2half2_rn(ldexpf(1.0f, -tile_shift));
        const float unscale       = ldexpf(1.0f, tile_shift);
        const int group_lane      = gid >> 2;
        // Scale bytes {key a, key b} of group `byte` (0..3 within the loaded word) as a prmt
        // selector: low byte from a's word, high byte from b's.
        const unsigned sel_h0 =
            static_cast<unsigned>(group_lane) | (static_cast<unsigned>(4 + group_lane) << 4);
        const unsigned sel_h1 =
            static_cast<unsigned>(2 + group_lane) | (static_cast<unsigned>(6 + group_lane) << 4);
        // One pass per 32-dimension half (u, hh): its FP16 partials stay in eight registers.
#pragma unroll
        for (int u = 0; u < Units; ++u) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const unsigned sel = hh == 0 ? sel_h0 : sel_h1;
                unsigned h[4][2];
#pragma unroll
                for (int i = 0; i < 4; ++i) { h[i][0] = h[i][1] = 0u; }
#pragma unroll
                for (int j = 0; j < PVKs; ++j) {
                    // Matrices: keys 16j+0..7 and 16j+8..15 of 16-byte chunk 2u+hh. Lane (g, t)
                    // receives keys (16j+2t, 16j+2t+1) for b0 and (16j+8+2t, 16j+9+2t) for b1 of
                    // unit 16u+8hh+g, whose group is 4u+2hh+g/4.
                    const int key   = j * 16 + (((lane >> 3) & 1) << 3) + a_rin;
                    const int chunk = 2 * u + hh;
                    unsigned r_lo   = 0;
                    unsigned r_hi   = 0;
                    ldmatrix_x2_t(r_lo, r_hi,
                                  smem_addr(&v_s[key * 128 + ((chunk ^ (key & 7)) << 4)]));
                    const int ka     = j * 16 + 2 * lid;
                    const unsigned a = load_vec<unsigned>(vs_s + ka * 16 + 4 * u);
                    const unsigned b = load_vec<unsigned>(vs_s + (ka + 1) * 16 + 4 * u);
                    const unsigned c = load_vec<unsigned>(vs_s + (ka + 8) * 16 + 4 * u);
                    const unsigned d = load_vec<unsigned>(vs_s + (ka + 9) * 16 + 4 * u);
                    __half2 scale_lo =
                        causal_prompt_rotated_fast_scale_pair(__byte_perm(a, b, sel));
                    __half2 scale_hi =
                        causal_prompt_rotated_fast_scale_pair(__byte_perm(c, d, sel));
                    if (tile_shift != 0) {
                        scale_lo = __hmul2(scale_lo, mul);
                        scale_hi = __hmul2(scale_hi, mul);
                    }
                    unsigned lo[4];
                    unsigned hi[4];
                    causal_prompt_rotated_fast_decode_unit(r_lo, scale_lo, lo);
                    causal_prompt_rotated_fast_decode_unit(r_hi, scale_hi, hi);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        causal_prompt_rotated_fast_mma_f16_acc(h[i][0], h[i][1], pa[j][0], pa[j][1],
                                                               pa[j][2], pa[j][3], lo[i], hi[i]);
                    }
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float (&a)[4] = acc[u][hh][i];
                    float2 r0     = __half22float2(load_vec<__half2>(&h[i][0]));
                    float2 r1     = __half22float2(load_vec<__half2>(&h[i][1]));
                    if (tile_shift != 0) {
                        r0.x *= unscale;
                        r0.y *= unscale;
                        r1.x *= unscale;
                        r1.y *= unscale;
                    }
                    a[0] = __fmaf_rn(a[0], tile_alpha0, r0.x);
                    a[1] = __fmaf_rn(a[1], tile_alpha0, r0.y);
                    a[2] = __fmaf_rn(a[2], tile_alpha1, r1.x);
                    a[3] = __fmaf_rn(a[3], tile_alpha1, r1.y);
                }
            }
        }
    };

    // Every warp publishes one tile per barrier; the next tile's copy overlaps this tile's math.
    if (kb_begin < kb_end) { issue_tile(kb_begin); }
#pragma unroll 1
    for (int kb = kb_begin; kb < kb_end; ++kb) {
        cp_wait<0>();
        __syncthreads();
        if (kb + 1 < kb_end) { issue_tile(kb + 1); }
        qk_softmax(kb);
        pv(kb);
    }

    running_l0         = warp_sum<4>(running_l0, FullMask);
    running_l1         = warp_sum<4>(running_l1, FullMask);
    const float inv_l0 = running_l0 > 0.0f ? __frcp_rn(running_l0) : 0.0f;
    const float inv_l1 = running_l1 > 0.0f ? __frcp_rn(running_l1) : 0.0f;

    // Lane (g, t) holds, for rows g and g+8, dimensions d = 64u + 32h + 8t + 4c + i in
    // acc[u][h][i][2 * row + c], c selecting column 2t + c. The normalized inverse rotation is one
    // butterfly on each of the eight index bits (i: 0-1, c: 2, t: 3-4, h: 5, u: 6-7) and a single
    // 2^-4 scale, applied in place.
#pragma unroll
    for (int u = 0; u < Units; ++u) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float (&a)[4] = acc[u][hh][i];
                a[0] *= inv_l0;
                a[1] *= inv_l0;
                a[2] *= inv_l1;
                a[3] *= inv_l1;
            }
        }
    }
    const auto butterfly = [](float& lo, float& hi) {
        const float sum  = __fadd_rn(lo, hi);
        const float diff = __fsub_rn(lo, hi);
        lo               = sum;
        hi               = diff;
    };
#pragma unroll
    for (int u = 0; u < Units; ++u) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                // i bits 0 and 1.
                butterfly(acc[u][hh][0][e], acc[u][hh][1][e]);
                butterfly(acc[u][hh][2][e], acc[u][hh][3][e]);
                butterfly(acc[u][hh][0][e], acc[u][hh][2][e]);
                butterfly(acc[u][hh][1][e], acc[u][hh][3][e]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // c bit: columns 2t and 2t+1 of each row.
                butterfly(acc[u][hh][i][0], acc[u][hh][i][1]);
                butterfly(acc[u][hh][i][2], acc[u][hh][i][3]);
            }
        }
    }
    // t bits across the lane quad.
#pragma unroll
    for (int u = 0; u < Units; ++u) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float value      = acc[u][hh][i][e];
                    value            = causal_prompt_rotated_fast_quad_butterfly(value, lid & 1, 1);
                    value            = causal_prompt_rotated_fast_quad_butterfly(value, lid & 2, 2);
                    acc[u][hh][i][e] = value;
                }
            }
        }
    }
    // h bit, then u bits 0 and 1.
#pragma unroll
    for (int u = 0; u < Units; ++u) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int e = 0; e < 4; ++e) { butterfly(acc[u][0][i][e], acc[u][1][i][e]); }
        }
    }
#pragma unroll
    for (int span = 1; span < Units; span <<= 1) {
#pragma unroll
        for (int u = 0; u < Units; ++u) {
            if ((u & span) != 0) { continue; }
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        butterfly(acc[u][hh][i][e], acc[u + span][hh][i][e]);
                    }
                }
            }
        }
    }
    if constexpr (Split) {
        const auto publish = [&](int row, int e0, float maximum, float sum) {
            const int token = q0 + row;
            if (row >= rows) { return; }
            const std::int64_t index =
                causal_prompt_rotated_fast_partial_row<Geometry>(split, token, q_head, width);
            if (lid == 0) { partial_stats[index] = make_float2(maximum, sum); }
            float* target = partial_rows + index * D;
#pragma unroll
            for (int u = 0; u < Units; ++u) {
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int d0 = 64 * u + 32 * hh + 8 * lid + 4 * c;
                        store_vec(target + d0, make_float4(acc[u][hh][0][e0 + c] * 0x1p-4f,
                                                           acc[u][hh][1][e0 + c] * 0x1p-4f,
                                                           acc[u][hh][2][e0 + c] * 0x1p-4f,
                                                           acc[u][hh][3][e0 + c] * 0x1p-4f));
                    }
                }
            }
        };
        publish(row0, 0, running_m0, running_l0);
        publish(row1, 2, running_m1, running_l1);
    } else {
#pragma unroll
        for (int u = 0; u < Units; ++u) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int d0 = 64 * u + 32 * hh + 8 * lid + 4 * c;
                    store_row(row0, d0, acc[u][hh][0][c] * 0x1p-4f, acc[u][hh][1][c] * 0x1p-4f,
                              acc[u][hh][2][c] * 0x1p-4f, acc[u][hh][3][c] * 0x1p-4f);
                    store_row(row1, d0, acc[u][hh][0][2 + c] * 0x1p-4f,
                              acc[u][hh][1][2 + c] * 0x1p-4f, acc[u][hh][2][2 + c] * 0x1p-4f,
                              acc[u][hh][3][2 + c] * 0x1p-4f);
                }
            }
        }
    }
}

// Combines the key splits of one column and query head: every split row is normalized by its own
// sum, so the merged row is sum_s w_s * row_s / sum_s w_s with w_s = sum_s * 2^((m_s - M) * c).
// Columns past the valid count publish zeros.
template <typename Geometry>
__global__
__launch_bounds__(kCausalPromptHeadDim) void causal_attention_prompt_rotated_fast_merge_kernel(
    const float* __restrict__ partial_rows, const float2* __restrict__ partial_stats,
    const std::int32_t* __restrict__ valid_columns, std::int32_t width, std::int32_t splits,
    float scale_l2, __nv_bfloat16* __restrict__ out) {
    const int column = static_cast<int>(blockIdx.x);
    const int q_head = static_cast<int>(blockIdx.y);
    const int d      = static_cast<int>(threadIdx.x);
    const int tokens = valid_columns == nullptr ? width : max(0, min(width, valid_columns[0]));
    const auto index = causal_prompt_q_index<Geometry>(q_head, d, column);
    if (column >= tokens) {
        out[index] = __float2bfloat16(0.0f);
        return;
    }
    float maximum = -CUDART_INF_F;
    for (int split = 0; split < splits; ++split) {
        const float2 stats = partial_stats[causal_prompt_rotated_fast_partial_row<Geometry>(
            split, column, q_head, width)];
        if (stats.y > 0.0f) { maximum = fmaxf(maximum, stats.x); }
    }
    float numerator   = 0.0f;
    float denominator = 0.0f;
    for (int split = 0; split < splits; ++split) {
        const std::int64_t row =
            causal_prompt_rotated_fast_partial_row<Geometry>(split, column, q_head, width);
        const float2 stats = partial_stats[row];
        if (stats.y > 0.0f) {
            const float weight = stats.y * exp2f((stats.x - maximum) * scale_l2);
            numerator = __fmaf_rn(weight, partial_rows[row * kCausalPromptHeadDim + d], numerator);
            denominator += weight;
        }
    }
    out[index] = __float2bfloat16(denominator > 0.0f ? numerator / denominator : 0.0f);
}

} // namespace ninfer::ops::detail
