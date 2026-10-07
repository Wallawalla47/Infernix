#pragma once

#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/softmax_attention/common/causal_epilogue.cuh"
#include "ops/softmax_attention/common/causal_operands.h"
#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/common/causal_softmax.cuh"
#include "ops/softmax_attention/common/causal_tile_io.cuh"
#include "ops/softmax_attention/common/mxfp8_tiled_plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/q_terms.cuh"

namespace infernix::ops::detail {

// E4M3 row-scaled keys (FP8 and K8V4 KV): an E4M3 Q row with an FP32 scale, MXFP8 QK.
struct Fp8TiledKeys {
    using Scale                      = __half;
    static constexpr bool kNvfp4     = false;
    static constexpr int kScaleBytes = 2; // per key
};

// NVFP4-G16 keys: a two-term NVFP4 Q (q_terms.cuh) with an FP32 row factor, block-scaled FP4 QK
// on the stored codes and UE4M3 group scales.
struct Nvfp4TiledKeys {
    using Scale                      = std::uint8_t;
    static constexpr bool kNvfp4     = true;
    static constexpr int kScaleBytes = 16; // per key
};

// Each warp retains 16 query rows through QK (Keys: FP8 or NVFP4), softmax, and FP16 PV.
// Values selects the stored V representation; split outputs remain FP32 with
// maxima in natural scaled-score units for the final merge.
//
// PV accumulates each 64-key tile on FP16 Tensor Cores (FP32 accumulation runs at half
// their rate on RTX 5090) and promotes the partial into the FP32 output once per tile.
// Every probability is at most one, so a partial is bounded by 64 * max |V|;
// Values::tile_shift picks the exact power of two that V is decoded down by to keep that
// bound inside FP16, and the partial is multiplied back in FP32.
//
// The 8-bit PV form (Pv8: NVFP4-G16 V, that is K8V4 and NVFP4 KV;
// CausalAttentionExecutionEnvelope::fast_prompt_pv8) decodes V once per tile to E4M3 under
// Values::tile_shift_e4m3, which brings the tile's largest scale into [32, 64), and runs PV on
// block-scaled E4M3 Tensor Cores with FP32 accumulation (four times the FP16 rate with FP32
// accumulation); the shift returns exactly through the UE8M0 B scale.
// Probabilities are taken against the tile's own row maximum (the softmax reference follows it,
// at most 2^40 below the running maximum) and enter as E4M3 codes of 256 p. The k32 MMA consumes
// each lane's own QK keys in a fixed permuted order (2t, 2t+1, 8+2t, 9+2t per half); transposed
// byte-pair loads of the arena split into an even- and an odd-dimension n8 tile that match it.
template <class Geometry, class Schedule, class Keys, class Values, class Metadata, bool Pv8>
__global__ __maxnreg__(Schedule::kMaxRegisters) void mxfp8_kv_tiled_mma_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v,
    const typename Keys::Scale* __restrict__ cache_k_scale,
    const typename Values::Scale* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, std::int32_t width,
    CausalKvPartition partition, CausalPartialView partial) {
    constexpr int D             = 256;
    constexpr int Br            = Schedule::kQueryRows;
    constexpr int Bc            = Schedule::kKeyRows;
    constexpr int DB16          = 128;
    constexpr int QKKs          = D / 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVNtPerWarp   = D / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr unsigned FullMask = 0xffffffffU;
    static_assert(QKKs == 8);
    static_assert(PVNtPerWarp == 32);
    static_assert(!Pv8 || Values::kE4m3Values);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    // Q codes, then (NVFP4 keys) both terms' group scales, then the FP32 row scales.
    std::uint8_t* q_codes        = reinterpret_cast<std::uint8_t*>(smem_raw);
    std::uint8_t* q_group_scales = q_codes + Schedule::kQBytes;
    float* q_scale = reinterpret_cast<float*>(q_group_scales +
                                              (Keys::kNvfp4 ? 2 * Br * kKVCacheNvfp4Groups : 0));
    std::uint8_t* k_codes = q_codes + Schedule::kQBytes + Schedule::kQScaleBytes;
    std::uint8_t* v_codes = k_codes + Schedule::kKBytes;
    __half* v_f16         = reinterpret_cast<__half*>(v_codes + Schedule::kVBytes);
    // Pv8: the E4M3 arena occupies the first half of the FP16 one, 256 bytes per key with
    // 16-byte chunk c of key k at ((c ^ (k & 7)) << 4).
    std::uint8_t* v_e4m3 = reinterpret_cast<std::uint8_t*>(v_f16);
    auto* k_scale_s      = reinterpret_cast<typename Keys::Scale*>(
        reinterpret_cast<unsigned char*>(v_f16) + Schedule::kVStageBytes);
    auto* v_scale_s = reinterpret_cast<typename Values::Scale*>(
        reinterpret_cast<unsigned char*>(k_scale_s) + Bc * Keys::kScaleBytes);

    // Longest-first issue order: the first QHeads CTAs of each split take the last row block of
    // every head, so a causal prompt's heaviest row blocks do not form the tail.
    const int row_blocks = static_cast<int>(gridDim.x);
    const int linear     = static_cast<int>(blockIdx.x + blockIdx.y * gridDim.x);
    const int q_head     = linear % Geometry::QHeads;
    const int q_block    = row_blocks - 1 - linear / Geometry::QHeads;
    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int q0         = q_block * Br;
    const int kv_head    = q_head / Geometry::GroupSize;
    const int tokens     = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) return;
    if (q0 >= tokens) return;
    const int split   = blockIdx.z;
    const int visible = positions[width - 1] + 1;
    const int active_splits =
        mxfp8_tiled_active_splits(partition, visible, width, Geometry::QHeads);
    if (split >= active_splits) return;
    const int logical_tiles         = div_up(visible, Bc);
    const int first_owned_tile      = split * logical_tiles / active_splits;
    const int end_owned_tile        = (split + 1) * logical_tiles / active_splits;
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int tile_rows             = min(Br, tokens - q0);
    const int max_query_abs         = base_pos + q0 + tile_rows - 1;
    const int key_blocks = max(0, min(end_owned_tile, max_query_abs / Bc + 1) - first_owned_tile);

    for (int row = warp; row < Br; row += Schedule::kWarps) {
        float values[8];
        float local_absmax = 0.0F;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            values[r]   = row < tile_rows
                              ? __bfloat162float(q[causal_q_index<Geometry>(q_head, d, q0 + row)])
                              : 0.0F;
        }
        normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) local_absmax = fmaxf(local_absmax, fabsf(values[r]));
        const float absmax = warp_max(local_absmax, FullMask);
        if constexpr (Keys::kNvfp4) {
            const float factor  = absmax > 0.0F ? absmax / kCausalNvfp4QTop : 0.0F;
            const float inverse = absmax > 0.0F ? kCausalNvfp4QTop / absmax : 0.0F;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const float x     = values[r] * inverse;
                const float first = causal_nvfp4_q_term(x, lane, row, r, q_codes + row * (D / 2),
                                                        q_group_scales + row * kKVCacheNvfp4Groups);
                (void)causal_nvfp4_q_term(x - first, lane, row, r, q_codes + (Br + row) * (D / 2),
                                          q_group_scales + (Br + row) * kKVCacheNvfp4Groups);
            }
            if (lane == 0) q_scale[row] = factor;
        } else {
            const float qs  = absmax > 0.0F ? absmax / kKVCacheFp8MaxFinite : 0.0F;
            const float inv = qs > 0.0F ? 1.0F / qs : 0.0F;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int d = lane + 32 * r;
                causal_store_query_code(q_codes, row, d, kv_cache_fp8_quant_code(values[r], inv));
            }
            if (lane == 0) q_scale[row] = qs;
        }
    }
    __syncthreads();

    const int gid              = lane >> 2;
    const int lid              = lane & 3;
    const int a_mat            = lane >> 3;
    const int a_rin            = lane & 7;
    const int a_rowoff         = a_rin + ((a_mat & 1) << 3);
    const int b_rin            = lane & 7;
    const int b_koff           = ((lane >> 3) & 1) << 3;
    const int warp_row0        = warp * 16;
    const float q_scale_r0     = q_scale[warp_row0 + gid];
    const float q_scale_r1     = q_scale[warp_row0 + gid + 8];
    const unsigned q_lane_base = smem_addr(q_codes) + (warp_row0 + a_rowoff) * D;
    const unsigned q_as        = (a_mat >> 1) << 4;
    const unsigned q_r         = a_rin << 4;
    const unsigned k_lane_base = smem_addr(k_codes) + b_rin * D + (lane >> 4) * (8 * D);
    const unsigned k_as        = (b_koff >> 3) << 4;
    const unsigned k_r         = b_rin << 4;
    const unsigned v_lane_base = smem_addr(v_f16) + (((lane >> 3) & 1) * 8 + b_rin) * D * 2;
    const unsigned v_as        = (lane >> 4) << 4;
    const unsigned v_r         = b_rin << 4;
    // Pv8 x4.trans rows: matrix lane / 8 holds keys 8 (lane / 8) .. +7 of the k32 step.
    const int v8_key            = ((lane >> 3) << 3) + (lane & 7);
    const unsigned v8_lane_base = smem_addr(v_e4m3) + v8_key * D;
    auto issue_kv_scales        = [&](int tile_k0, int cooperative_tid, int cooperative_threads) {
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        const int page_offset0  = tile_k0 & (kPagedKVPageSize - 1);
        for (int key_l = cooperative_tid; key_l < Bc; key_l += cooperative_threads) {
            const int key = tile_k0 + key_l;
            if (key <= max_query_abs) {
                const std::int64_t off = kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head,
                                                                            page_offset0 + key_l);
                if constexpr (Keys::kNvfp4)
                    cp_async<16>(k_scale_s + key_l * Keys::kScaleBytes,
                                 cache_k_scale +
                                     kv_cache_nvfp4_scale_index<Geometry>(physical_page, kv_head, 0,
                                                                          page_offset0 + key_l));
                else
                    k_scale_s[key_l] = cache_k_scale[off];
                if constexpr (Values::kScaleItems == 1) {
                    v_scale_s[key_l] = cache_v_scale[off];
                } else {
                    const auto v_off =
                        paged_kv_element_offset<Values::kScaleItems, Geometry::KVHeads>(
                            physical_page, kv_head, page_offset0 + key_l, 0);
                    cp_async<16>(v_scale_s + key_l * Values::kScaleItems, cache_v_scale + v_off);
                }
            } else {
                if constexpr (Keys::kNvfp4)
                    store_vec(k_scale_s + key_l * Keys::kScaleBytes, make_int4(0, 0, 0, 0));
                else
                    k_scale_s[key_l] = __float2half_rn(0.0F);
                if constexpr (Values::kScaleItems == 1)
                    v_scale_s[key_l] = __float2half_rn(0.0F);
                else
                    store_vec(v_scale_s + key_l * Values::kScaleItems, make_int4(0, 0, 0, 0));
            }
        }
    };

    auto issue_kv_codes = [&](int tile_k0, int cooperative_tid, int cooperative_threads) {
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        const int page_offset0  = tile_k0 & (kPagedKVPageSize - 1);
        if constexpr (Keys::kNvfp4) {
            // 128 code bytes per key, 16-byte chunk c of key k at ((c ^ (k & 7)) << 4).
#pragma unroll 1
            for (int chunk = cooperative_tid; chunk < Bc * 8; chunk += cooperative_threads) {
                const int key_l = chunk >> 3;
                const int c     = chunk & 7;
                auto* dst       = k_codes + key_l * 128 + ((c ^ (key_l & 7)) << 4);
                if (tile_k0 + key_l <= max_query_abs) {
                    cp_async<16, Cache::cg>(
                        dst, cache_k +
                                 kv_cache_nvfp4_code_index<Geometry>(physical_page, kv_head, 0,
                                                                     page_offset0 + key_l) +
                                 c * 16);
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
#pragma unroll 1
        for (int chunk = cooperative_tid; chunk < (Keys::kNvfp4 ? 0 : Bc * (D / 16));
             chunk += cooperative_threads) {
            const int key_l  = chunk / (D / 16);
            const int dc     = chunk - key_l * (D / 16);
            const int d      = dc * 16;
            const int key    = tile_k0 + key_l;
            std::uint8_t* kd = &k_codes[(key_l * DB16 + causal_swizzle(key_l, dc * 8)) * 2];
            if (key <= max_query_abs) {
                const std::int64_t off = kv_cache_fp8_code_index<Geometry>(physical_page, kv_head,
                                                                           d, page_offset0 + key_l);
                cp_async<16, Cache::cg>(kd, &cache_k[off]);
                if constexpr (Values::kCodeBytes == D)
                    cp_async<16, Cache::cg>(&v_codes[key_l * D + d], &cache_v[off]);
            } else {
                store_vec(kd, make_int4(0, 0, 0, 0));
                if constexpr (Values::kCodeBytes == D)
                    store_vec(&v_codes[key_l * D + d], make_int4(0, 0, 0, 0));
            }
        }
        if constexpr (Values::kCodeBytes != D) {
#pragma unroll 1
            for (int chunk = cooperative_tid; chunk < Bc * (Values::kCodeBytes / 16);
                 chunk += cooperative_threads) {
                const int row = chunk / (Values::kCodeBytes / 16);
                const int col = (chunk % (Values::kCodeBytes / 16)) * 16;
                auto* dst     = v_codes + row * Values::kCodeBytes + col;
                if (tile_k0 + row <= max_query_abs) {
                    const auto off = paged_kv_element_offset<Values::kCodeBytes, Geometry::KVHeads>(
                        physical_page, kv_head, page_offset0 + row, col);
                    cp_async<16, Cache::cg>(dst, cache_v + off);
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        infernix::ops::cp_commit();
    };

    auto issue_kv_tile = [&](int tile_k0, int cooperative_tid, int cooperative_threads) {
        issue_kv_scales(tile_k0, cooperative_tid, cooperative_threads);
        issue_kv_codes(tile_k0, cooperative_tid, cooperative_threads);
    };

    if (key_blocks > 0) issue_kv_tile(first_owned_tile * Bc, tid, Schedule::kThreads);
    infernix::ops::cp_wait<0>();
    __syncthreads();

    float acc[PVNtPerWarp][4]{};
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0F, l1 = 0.0F;
    const float scale_l2 = scale * kLog2E;
    // Pv8: m* is the latest tile's row maximum held within frame_gap of seen_m*, and
    // probabilities and row sums carry a factor of 256.
    float seen_m0 = -CUDART_INF_F, seen_m1 = -CUDART_INF_F;
    const float frame_gap     = 40.0F / scale_l2;
    constexpr float POffset   = Pv8 ? 8.0F : 0.0F;
    const auto update_softmax = [&](float& maximum, float tile_maximum) {
        const float previous = maximum;
        maximum              = fmaxf(maximum, tile_maximum);
        return previous == -CUDART_INF_F
                   ? 0.0F
                   : causal_exp_scaled(previous, maximum * scale_l2, scale_l2);
    };
    const auto step = [&]<bool FullTile>(int kb) {
        const int k0 = (first_owned_tile + kb) * Bc;
        // Every warp derives the tile's V shift from the staged scales on its own.
        int v_shift = 0;
        if constexpr (Pv8)
            v_shift = Values::template tile_shift_e4m3<Bc>(v_scale_s, lane);
        else
            v_shift = Values::template tile_shift<Bc>(v_scale_s, lane);
        const __half v_mul    = __float2half_rn(ldexpf(1.0F, -v_shift));
        const float v_unscale = ldexpf(1.0F, v_shift);
        // Conversion runs while all query warps retain their row state in registers.
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += Schedule::kThreads) {
            const int row = chunk / (D / 8);
            const int d   = (chunk % (D / 8)) * 8;
            const std::uint8_t* codes =
                v_codes + row * Values::kCodeBytes + d * Values::kCodeBytes / D;
            const auto v_scale =
                v_scale_s[row * Values::kScaleItems + d / (D / Values::kScaleItems)];
            if constexpr (Pv8) {
                store_vec(&v_e4m3[row * D + ((((d >> 4) ^ (row & 7)) << 4) | (d & 15))],
                          Values::expand_e4m3(codes, v_scale, v_mul));
            } else {
                store_vec(&v_f16[row * D + causal_swizzle(row, d)],
                          Values::expand(codes, v_scale, v_mul));
            }
        }
        // S = Q Kᵀ for this warp's 16 rows over all Bc keys, in registers.
        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
        if constexpr (Keys::kNvfp4) {
            // One block-scaled m16n8k64 MMA per 64-dimension slab, Q term and key tile. Lane rows
            // follow the A4 linear route; the scale words are the slab's four UE4M3 group scales
            // of the lane's A row ((lane & 1) * 8 + lane / 4) and B key (lane / 4).
            const auto* k_scale_words = reinterpret_cast<const unsigned*>(k_scale_s);
            const auto* q_scale_words = reinterpret_cast<const unsigned*>(q_group_scales);
            const int a_row           = warp_row0 + a_rowoff;
            const int sfa_row         = warp_row0 + (((lane & 1) << 3) | gid);
            // Rolled: an unrolled slab loop holds every slab's fragment addresses and loads at
            // once, which spilled 112-176 bytes per thread at 255 registers and took about 5 %
            // longer (3.7 % with 8-bit PV); each slab's 16 MMAs keep the Tensor Cores busy.
#pragma unroll 1
            for (int slab = 0; slab < D / 64; ++slab) {
                unsigned af[2][4];
                unsigned sfa[2];
#pragma unroll
                for (int term = 0; term < 2; ++term) {
                    const int chunk = 2 * slab + (a_mat >> 1);
                    ldmatrix_x4(af[term][0], af[term][1], af[term][2], af[term][3],
                                smem_addr(q_codes + (term * Br + a_row) * (D / 2) +
                                          ((chunk ^ (a_row & 7)) << 4)));
                    sfa[term] = q_scale_words[(term * Br + sfa_row) * 4 + slab];
                }
#pragma unroll
                for (int nt = 0; nt < QKNt; nt += 2) {
                    const int key   = (nt + (lane >> 4)) * 8 + a_rin;
                    const int chunk = 2 * slab + ((lane >> 3) & 1);
                    unsigned bf[4];
                    ldmatrix_x4(bf[0], bf[1], bf[2], bf[3],
                                smem_addr(k_codes + key * 128 + ((chunk ^ (key & 7)) << 4)));
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
                score[nt][0] *= q_scale_r0;
                score[nt][1] *= q_scale_r0;
                score[nt][2] *= q_scale_r1;
                score[nt][3] *= q_scale_r1;
            }
        } else {
            // Software-pipelined like cute's gemm: issue the ldmatrix for contraction
            // step k+1 while the m16n8k32 MMAs for step k run, so the LSU (ldmatrix)
            // and tensor pipes overlap instead of stalling on each other.
            // Swizzled ldmatrix addresses via precomputed per-lane bases + immediates.
            unsigned af[2][4];
            unsigned bf[2][QKNt][2];
            {
                ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                            causal_swizzle_address(q_lane_base, 0u, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(
                        bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                        causal_swizzle_address(k_lane_base + static_cast<unsigned>(nt2 * (8 * D)),
                                               0u, k_as, k_r));
                }
            }
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                const int cur = k & 1;
                const int nxt = cur ^ 1;
                if (k + 1 < QKKs) {
                    const unsigned ck = static_cast<unsigned>((k + 1) << 5);
                    ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                                causal_swizzle_address(q_lane_base, ck, q_as, q_r));
#pragma unroll
                    for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                        ldmatrix_x4(bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0],
                                    bf[nxt][nt2 + 1][1],
                                    causal_swizzle_address(k_lane_base +
                                                               static_cast<unsigned>(nt2 * (8 * D)),
                                                           ck, k_as, k_r));
                    }
                }
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    mma_fp8_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                                 af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0],
                                 bf[cur][nt][1]);
                }
            }

#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int keya  = nt * 8 + 2 * lid;
                const float ks0 = __half2float(k_scale_s[keya]);
                const float ks1 = __half2float(k_scale_s[keya + 1]);
                score[nt][0] *= q_scale_r0 * ks0;
                score[nt][1] *= q_scale_r0 * ks1;
                score[nt][2] *= q_scale_r1 * ks0;
                score[nt][3] *= q_scale_r1 * ks1;
            }
        }
        // QK is finished with the codes; V has been decoded into its own arena.
        __syncthreads();
        if (kb + 1 < key_blocks) issue_kv_tile(k0 + Bc, tid, Schedule::kThreads);
        const int row0  = warp_row0 + gid;
        const int row1  = warp_row0 + gid + 8;
        const int qrow0 = q0 + row0;
        const int qrow1 = q0 + row1;
        const int qabs0 = (qrow0 < tokens) ? base_pos + qrow0 : -1;
        const int qabs1 = (qrow1 < tokens) ? base_pos + qrow1 : -1;

        // Row maximum before attention scaling; scale is folded into exp2 below.
        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
        if constexpr (FullTile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                score[nt][0]   = (qrow0 < tokens && key0 <= qabs0) ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = (qrow0 < tokens && key1 <= qabs0) ? score[nt][1] : -CUDART_INF_F;
                score[nt][2]   = (qrow1 < tokens && key0 <= qabs1) ? score[nt][2] : -CUDART_INF_F;
                score[nt][3]   = (qrow1 < tokens && key1 <= qabs1) ? score[nt][3] : -CUDART_INF_F;
                bm0            = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1            = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        float alpha0 = 0.0F, alpha1 = 0.0F;
        if constexpr (Pv8) {
            seen_m0        = fmaxf(seen_m0, bm0);
            seen_m1        = fmaxf(seen_m1, bm1);
            const float f0 = bm0 == -CUDART_INF_F ? m0 : fmaxf(bm0, seen_m0 - frame_gap);
            const float f1 = bm1 == -CUDART_INF_F ? m1 : fmaxf(bm1, seen_m1 - frame_gap);
            alpha0 = m0 == -CUDART_INF_F ? 0.0F : causal_exp_scaled(m0, f0 * scale_l2, scale_l2);
            alpha1 = m1 == -CUDART_INF_F ? 0.0F : causal_exp_scaled(m1, f1 * scale_l2, scale_l2);
            m0     = f0;
            m1     = f1;
        } else {
            alpha0 = update_softmax(m0, bm0);
            alpha1 = update_softmax(m1, bm1);
        }
        const float nm0_scaled = m0 * scale_l2 - POffset;
        const float nm1_scaled = m1 * scale_l2 - POffset;

        // P = exp2(S - m), repacked into the PV A-fragment layout, plus local block row-sum.
        // The row-sum allreduce is deferred to the epilogue; only row max must be reduced per tile.
        float bl0 = 0.0f, bl1 = 0.0f;
        unsigned p_frag[PVKs][4];
        // Pv8: k32 step nt / 4 takes score tiles 4J (low half) and 4J+1 (high half) of rows g
        // and g+8 into p8[J][0..1], tiles 4J+2 and 4J+3 into p8[J][2..3].
        unsigned p8[PVKs / 2][4];
        const auto store_p = [&](int nt, float p00, float p01, float p10, float p11) {
            if constexpr (Pv8) {
                const unsigned r0 =
                    __nv_cvt_float2_to_fp8x2(make_float2(p00, p01), __NV_SATFINITE, __NV_E4M3);
                const unsigned r1 =
                    __nv_cvt_float2_to_fp8x2(make_float2(p10, p11), __NV_SATFINITE, __NV_E4M3);
                const int q      = nt & 3;
                unsigned (&a)[4] = p8[nt >> 2];
                if ((q & 1) == 0) {
                    a[(q >> 1) * 2 + 0] = r0;
                    a[(q >> 1) * 2 + 1] = r1;
                } else {
                    a[(q >> 1) * 2 + 0] |= r0 << 16;
                    a[(q >> 1) * 2 + 1] |= r1 << 16;
                }
            } else {
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = pack_f16x2(p00, p01);
                    p_frag[pk][1] = pack_f16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_f16x2(p00, p01);
                    p_frag[pk][3] = pack_f16x2(p10, p11);
                }
            }
        };
        if constexpr (FullTile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
                const float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
                const float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
                const float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                store_p(nt, p00, p01, p10, p11);
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = (score[nt][0] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p01 = (score[nt][1] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p10 = (score[nt][2] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                      : 0.0f;
                const float p11 = (score[nt][3] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                store_p(nt, p00, p01, p10, p11);
            }
        }

        l0 = __fmaf_rn(l0, alpha0, bl0);
        l1 = __fmaf_rn(l1, alpha1, bl1);

        if constexpr (Pv8) {
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                acc[n][0] *= alpha0;
                acc[n][1] *= alpha0;
                acc[n][2] *= alpha1;
                acc[n][3] *= alpha1;
            }
            // O += P V over (k32 step, 16-dimension chunk c); each x4.trans load gives keys
            // 2t, 2t+1 of the step's four 8-key quarters for dimensions 2g, 2g+1 of chunk c,
            // which byte permutes split into the even (acc[2c]) and odd (acc[2c+1]) n8 tiles.
            const unsigned sfb   = static_cast<unsigned>(127 + v_shift);
            constexpr int Chunks = D / 16;
            constexpr int Loads  = (PVKs / 2) * Chunks;
            unsigned vb[2][4];
            const auto load_v8 = [&](unsigned (&r)[4], int li) {
                const int step = li / Chunks;
                const int c    = li % Chunks;
                ldmatrix_x4_t(r[0], r[1], r[2], r[3],
                              v8_lane_base + static_cast<unsigned>(step * 32 * D) +
                                  static_cast<unsigned>((c ^ (v8_key & 7)) << 4));
            };
            load_v8(vb[0], 0);
#pragma unroll
            for (int li = 0; li < Loads; ++li) {
                const int step = li / Chunks;
                const int c    = li % Chunks;
                const int cur  = li & 1;
                if (li + 1 < Loads) load_v8(vb[cur ^ 1], li + 1);
                const unsigned even0 = __byte_perm(vb[cur][0], vb[cur][1], 0x6420);
                const unsigned even1 = __byte_perm(vb[cur][2], vb[cur][3], 0x6420);
                const unsigned odd0  = __byte_perm(vb[cur][0], vb[cur][1], 0x7531);
                const unsigned odd1  = __byte_perm(vb[cur][2], vb[cur][3], 0x7531);
                mma_fp8_e4m3_scaled(acc[2 * c][0], acc[2 * c][1], acc[2 * c][2], acc[2 * c][3],
                                    p8[step][0], p8[step][1], p8[step][2], p8[step][3], even0,
                                    even1, 0x7FU, sfb);
                mma_fp8_e4m3_scaled(acc[2 * c + 1][0], acc[2 * c + 1][1], acc[2 * c + 1][2],
                                    acc[2 * c + 1][3], p8[step][0], p8[step][1], p8[step][2],
                                    p8[step][3], odd0, odd1, 0x7FU, sfb);
            }
            if (kb + 1 < key_blocks) infernix::ops::cp_wait<0>();
            __syncthreads();
            return;
        }

        // O = alpha O + P V, contracting over the Bc keys. The (n, k) iteration space is
        // flattened n-major and software-pipelined: the transposed ldmatrix for the next
        // V fragment is issued while the current MMA runs. Each x4.trans load covers 2
        // output n-tiles (16 dims), whose FP16 partials over the tile's keys stay in four
        // registers until their FP32 promotion.
        constexpr int PVHalf  = PVNtPerWarp / 2; // 16 n-tile pairs
        constexpr int PVLoads = PVKs * PVHalf;   // 64 x4.trans loads
        // Swizzled V x4.trans addresses via precomputed per-lane base + immediates.
        unsigned vf[2][4];
        {
            ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                          causal_swizzle_address(v_lane_base, 0u, v_as, v_r));
        }
        unsigned partial_h[2][2];
#pragma unroll
        for (int li = 0; li < PVLoads; ++li) {
            const int n2  = (li / PVKs) * 2;
            const int k   = li % PVKs;
            const int cur = li & 1;
            const int nxt = cur ^ 1;
            if (li + 1 < PVLoads) {
                const int n2b      = ((li + 1) / PVKs) * 2;
                const int k2       = (li + 1) % PVKs;
                const unsigned ckv = static_cast<unsigned>(n2b << 4);
                ldmatrix_x4_t(
                    vf[nxt][0], vf[nxt][1], vf[nxt][2], vf[nxt][3],
                    causal_swizzle_address(v_lane_base + static_cast<unsigned>(k2 * (16 * D * 2)),
                                           ckv, v_as, v_r));
            }
            if (k == 0) {
                partial_h[0][0] = partial_h[0][1] = 0U;
                partial_h[1][0] = partial_h[1][1] = 0U;
            }
            mma_f16_f16acc(partial_h[0][0], partial_h[0][1], p_frag[k][0], p_frag[k][1],
                           p_frag[k][2], p_frag[k][3], vf[cur][0], vf[cur][1]);
            mma_f16_f16acc(partial_h[1][0], partial_h[1][1], p_frag[k][0], p_frag[k][1],
                           p_frag[k][2], p_frag[k][3], vf[cur][2], vf[cur][3]);
            if (k == PVKs - 1) {
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    float2 r0 = __half22float2(load_vec<__half2>(&partial_h[t][0]));
                    float2 r1 = __half22float2(load_vec<__half2>(&partial_h[t][1]));
                    if (v_shift != 0) {
                        r0.x *= v_unscale;
                        r0.y *= v_unscale;
                        r1.x *= v_unscale;
                        r1.y *= v_unscale;
                    }
                    float (&a)[4] = acc[n2 + t];
                    a[0]          = __fmaf_rn(a[0], alpha0, r0.x);
                    a[1]          = __fmaf_rn(a[1], alpha0, r0.y);
                    a[2]          = __fmaf_rn(a[2], alpha1, r1.x);
                    a[3]          = __fmaf_rn(a[3], alpha1, r1.y);
                }
            }
        }

        if (kb + 1 < key_blocks) infernix::ops::cp_wait<0>();
        __syncthreads();
    };
    const int full_blocks =
        q0 + Br <= tokens ? min(key_blocks, max(0, (base_pos + q0 + 1) / Bc - first_owned_tile))
                          : 0;
    for (int kb = 0; kb < full_blocks; ++kb) step.template operator()<true>(kb);
    for (int kb = full_blocks; kb < key_blocks; ++kb) step.template operator()<false>(kb);
    l0             = warp_sum<4>(l0, FullMask);
    l1             = warp_sum<4>(l1, FullMask);
    const int row0 = warp_row0 + gid;
    const int row1 = row0 + 8;
    if (lid == 0) {
        if (row0 < tile_rows) {
            const auto index       = causal_stat_index<Geometry>(q_head, q0 + row0, split, width);
            partial.maximum[index] = m0 * scale;
            partial.sum[index]     = l0;
        }
        if (row1 < tile_rows) {
            const auto index       = causal_stat_index<Geometry>(q_head, q0 + row1, split, width);
            partial.maximum[index] = m1 * scale;
            partial.sum[index]     = l1;
        }
    }
    if constexpr (Pv8) {
        // n8 tile 2c holds dimensions 16c + 2n and tile 2c+1 dimensions 16c + 2n + 1: lane t
        // owns 16c + 4t .. 16c + 4t + 3 of rows g and g+8.
#pragma unroll
        for (int c = 0; c < PVNtPerWarp / 2; ++c) {
            const int d0 = 16 * c + 4 * lid;
            if (row0 < tile_rows) {
                float* target = partial.acc +
                                causal_partial_index<Geometry>(q_head, d0, q0 + row0, split, width);
                causal_store_partial_pair(target, acc[2 * c][0], acc[2 * c + 1][0]);
                causal_store_partial_pair(target + 2, acc[2 * c][1], acc[2 * c + 1][1]);
            }
            if (row1 < tile_rows) {
                float* target = partial.acc +
                                causal_partial_index<Geometry>(q_head, d0, q0 + row1, split, width);
                causal_store_partial_pair(target, acc[2 * c][2], acc[2 * c + 1][2]);
                causal_store_partial_pair(target + 2, acc[2 * c][3], acc[2 * c + 1][3]);
            }
        }
        return;
    }
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int d0 = n * 8 + 2 * lid;
        if (row0 < tile_rows)
            causal_store_partial_pair(
                partial.acc + causal_partial_index<Geometry>(q_head, d0, q0 + row0, split, width),
                acc[n][0], acc[n][1]);
        if (row1 < tile_rows)
            causal_store_partial_pair(
                partial.acc + causal_partial_index<Geometry>(q_head, d0, q0 + row1, split, width),
                acc[n][2], acc[n][3]);
    }
}
} // namespace infernix::ops::detail
