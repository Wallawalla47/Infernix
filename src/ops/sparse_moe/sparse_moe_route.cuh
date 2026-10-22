#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace infernix::ops::detail {

inline constexpr int kSparseMoeExperts = 256;
inline constexpr int kSparseMoeTopK    = 8;

struct SparseMoeRankedValue {
    float value;
    int id;
};

__device__ __forceinline__ bool sparse_moe_ranked_better(const SparseMoeRankedValue& a,
                                                         const SparseMoeRankedValue& b) {
    return a.value > b.value || (a.value == b.value && a.id < b.id);
}

// Merges the descending run of each lane with the run of its xor partner and keeps the better
// half. The eight exchanges of a step do not depend on each other, so the warp reaches the
// warp-wide top-8 in five merge steps instead of eight dependent reduction rounds. Ranking is a
// total order over distinct expert ids, so the selected set and its order are the same as the
// ones any other correct selection produces.
__device__ __forceinline__ void
sparse_moe_merge_ranked_runs(SparseMoeRankedValue (&run)[kSparseMoeTopK]) {
#pragma unroll
    for (int partner = 1; partner < 32; partner <<= 1) {
        SparseMoeRankedValue merged[kSparseMoeTopK];
#pragma unroll
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
            const SparseMoeRankedValue mirror = run[kSparseMoeTopK - 1 - rank];
            SparseMoeRankedValue other;
            other.value  = __shfl_xor_sync(kFullWarpMask, mirror.value, partner);
            other.id     = __shfl_xor_sync(kFullWarpMask, mirror.id, partner);
            merged[rank] = sparse_moe_ranked_better(run[rank], other) ? run[rank] : other;
        }
        // The kept half is bitonic; three compare-exchange stages restore descending order.
#pragma unroll
        for (int stride = kSparseMoeTopK / 2; stride > 0; stride >>= 1) {
#pragma unroll
            for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
                if ((rank & stride) != 0) { continue; }
                const int partner_rank = rank | stride;
                if (!sparse_moe_ranked_better(merged[rank], merged[partner_rank])) {
                    const SparseMoeRankedValue swap = merged[rank];
                    merged[rank]                    = merged[partner_rank];
                    merged[partner_rank]            = swap;
                }
            }
        }
#pragma unroll
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) { run[rank] = merged[rank]; }
    }
}

// A router logit as the model computes it: the router projection's output is BF16 (Transformers'
// Qwen3_5MoeSparseMoeBlock, the lossless route), so selection, the routing weights and the shared
// expert's gate read the FP32 score rounded once to BF16. Ties at that precision go to the lower id.
__device__ __forceinline__ float sparse_moe_bf16_logit(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ void sparse_moe_select_top8_warp(const float* scores, int* ids,
                                                            float* alpha, float* shared_scale,
                                                            float* selected_logits) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    SparseMoeRankedValue local[kSparseMoeTopK];
#pragma unroll
    for (int item = 0; item < kSparseMoeTopK; ++item) {
        const int id = lane + item * 32;
        local[item]  = {sparse_moe_bf16_logit(scores[id]), id};
    }
#pragma unroll
    for (int i = 1; i < kSparseMoeTopK; ++i) {
        const SparseMoeRankedValue value = local[i];
        int position                     = i;
        while (position > 0 && sparse_moe_ranked_better(value, local[position - 1])) {
            local[position] = local[position - 1];
            --position;
        }
        local[position] = value;
    }
    sparse_moe_merge_ranked_runs(local);

    if (lane == 0) {
#pragma unroll
        for (int rank = 0; rank < kSparseMoeTopK; ++rank) {
            ids[rank]             = local[rank].id;
            selected_logits[rank] = local[rank].value;
        }
    }
    __syncwarp();

    float exponential = 0.0f;
    if (lane < kSparseMoeTopK) { exponential = expf(selected_logits[lane] - selected_logits[0]); }
    float denominator = warp_reduce_sum(exponential);
    denominator       = __shfl_sync(kFullWarpMask, denominator, 0);
    if (lane < kSparseMoeTopK) { alpha[lane] = exponential / denominator; }
    if (lane == 0) { *shared_scale = sigmoid(sparse_moe_bf16_logit(scores[kSparseMoeExperts])); }
}

// The router score of one row for one token: eight warps own 256-element K slices; a lane sums its
// eight products in K order from zero, the warp reduces with warp_reduce_sum and warp 0 then reduces
// the eight warp sums (sparse_moe_router_row_reduce). The decode router and the column-invariant
// small-T router share these two steps, so a token's scores do not depend on the call's width.
inline constexpr int kSparseMoeRouterHidden = 2048;
inline constexpr int kSparseMoeRouterWarps  = 8;

__device__ __forceinline__ float sparse_moe_dot_bf16_eight(const __nv_bfloat16* a,
                                                           const __nv_bfloat16* b) {
    const uint4 av  = load_vec<uint4>(a);
    const uint4 bv  = load_vec<uint4>(b);
    const float2 a0 = bf16x2_bits_to_float2(av.x);
    const float2 a1 = bf16x2_bits_to_float2(av.y);
    const float2 a2 = bf16x2_bits_to_float2(av.z);
    const float2 a3 = bf16x2_bits_to_float2(av.w);
    const float2 b0 = bf16x2_bits_to_float2(bv.x);
    const float2 b1 = bf16x2_bits_to_float2(bv.y);
    const float2 b2 = bf16x2_bits_to_float2(bv.z);
    const float2 b3 = bf16x2_bits_to_float2(bv.w);
    float sum       = 0.0f;
    sum             = fmaf(a0.x, b0.x, sum);
    sum             = fmaf(a0.y, b0.y, sum);
    sum             = fmaf(a1.x, b1.x, sum);
    sum             = fmaf(a1.y, b1.y, sum);
    sum             = fmaf(a2.x, b2.x, sum);
    sum             = fmaf(a2.y, b2.y, sum);
    sum             = fmaf(a3.x, b3.x, sum);
    sum             = fmaf(a3.y, b3.y, sum);
    return sum;
}

// This warp's sum of the row's products with x over its K slice (every lane holds it).
__device__ __forceinline__ float sparse_moe_router_row_dot(const __nv_bfloat16* x,
                                                           const __nv_bfloat16* row) {
    constexpr int kSlice = kSparseMoeRouterHidden / kSparseMoeRouterWarps;
    constexpr int kVecs  = kSlice / (32 * 8);
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    float sum            = 0.0f;
#pragma unroll
    for (int vector = 0; vector < kVecs; ++vector) {
        const int k = warp * kSlice + vector * 32 * 8 + lane * 8;
        sum += sparse_moe_dot_bf16_eight(row + k, x + k);
    }
    return warp_reduce_sum(sum);
}

// Warp 0's reduction of the eight warp sums (partial[w] from warp w's lane 0); lane 0 holds it.
__device__ __forceinline__ float sparse_moe_router_row_reduce(const float* partial) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float value    = lane < kSparseMoeRouterWarps ? partial[lane] : 0.0f;
    return warp_reduce_sum<kSparseMoeRouterWarps>(value);
}

} // namespace infernix::ops::detail
