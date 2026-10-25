#pragma once

// Implements: include/infernix/ops/argmax.h
// Match: contiguous FP32 or BF16 [physical_rows,C], valid_rows <= physical_rows, with
// tuned routes qualified for the 248077-row and 131072-row vocabularies.
// Algorithm assumptions: one CTA reduces each blockDim.x-row tile and an atomic
// winner selects the exact value/lower-id maximum per column; argmax_with_probability's
// tiles also publish their softmax partials, which the column's last tile combines.

#include <cuda_bf16.h>
#include "core/pdl.cuh"
#include <cstdint>
#include <climits>
#include <math_constants.h>

namespace infernix::ops {

// Small-column execution uses 512 threads to reduce atomic contenders. Aggregate
// execution dispatches registered vocab profiles to a smaller block so the much
// larger 2-D grid exposes enough resident CTAs without oversized reductions.
inline constexpr int kArgmaxBlock          = 512;
inline constexpr int kArgmaxItemsPerThread = 16; // rows per thread: two 8-row vector loads

__device__ __forceinline__ bool argmax_better(float value, std::int32_t index, float best_value,
                                              std::int32_t best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

__device__ __forceinline__ void argmax_warp_reduce(float& value, std::int32_t& index) {
    constexpr unsigned int kMask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value        = __shfl_down_sync(kMask, value, offset);
        const std::int32_t other_index = __shfl_down_sync(kMask, index, offset);
        if (argmax_better(other_value, other_index, value, index)) {
            value = other_value;
            index = other_index;
        }
    }
}

__device__ __forceinline__ void argmax_block_reduce(float& value, std::int32_t& index) {
    __shared__ float warp_values[16];
    __shared__ std::int32_t warp_indices[16];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    argmax_warp_reduce(value, index);
    if (lane == 0) {
        warp_values[warp]  = value;
        warp_indices[warp] = index;
    }
    __syncthreads();

    value = (lane < (blockDim.x >> 5)) ? warp_values[lane] : -CUDART_INF_F;
    index = (lane < (blockDim.x >> 5)) ? warp_indices[lane] : INT32_MAX;
    if (warp == 0) { argmax_warp_reduce(value, index); }
}

__device__ __forceinline__ float argmax_logit(__nv_bfloat16 value) { return __bfloat162float(value); }
__device__ __forceinline__ float argmax_logit(float value) { return value; }

// One CTA per column (BF16 or FP32 logits).
template <class Logit>
__launch_bounds__(kArgmaxBlock) __global__
    void argmax_kernel(const Logit* logits, std::int32_t* out, std::int32_t valid_rows,
                       std::int32_t physical_rows) {
    const std::int32_t t    = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;

    float best_value        = argmax_logit(logits[base]);
    std::int32_t best_index = 0;
    for (std::int32_t v = static_cast<std::int32_t>(threadIdx.x); v < valid_rows; v += blockDim.x) {
        const float value = argmax_logit(logits[base + v]);
        if (argmax_better(value, v, best_value, best_index)) {
            best_value = value;
            best_index = v;
        }
    }

    __shared__ float values[kArgmaxBlock];
    __shared__ std::int32_t indices[kArgmaxBlock];
    values[threadIdx.x]  = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_value        = values[threadIdx.x + stride];
            const std::int32_t other_index = indices[threadIdx.x + stride];
            if (argmax_better(other_value, other_index, values[threadIdx.x],
                              indices[threadIdx.x])) {
                values[threadIdx.x]  = other_value;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) { out[t] = indices[0]; }
}

// argmax_with_probability: the running argmax of the rows seen with its softmax denominator,
// sum = sum_v exp(logit_v - value). A larger logit rescales the sum to itself; equal maxima keep the
// lowest row (argmax_better's order).
struct ArgmaxSoftmax {
    float value;
    std::int32_t index;
    float sum;
};

// No row seen yet.
__device__ __forceinline__ ArgmaxSoftmax argmax_softmax_empty() { return {-CUDART_INF_F, INT32_MAX, 0.0F}; }

__device__ __forceinline__ void argmax_softmax_consider(ArgmaxSoftmax& s, float logit, std::int32_t row) {
    if (logit > s.value) {
        s.sum   = s.sum * __expf(s.value - logit) + 1.0F; // 0 * exp(-inf) = 0 for the first row
        s.value = logit;
        s.index = row;
    } else {
        if (logit == s.value && row < s.index) { s.index = row; }
        if (logit > -CUDART_INF_F) { s.sum += __expf(logit - s.value); }
    }
}

// a then b: the larger maximum, its lowest row, and both sums rescaled to it (an empty side adds 0).
__device__ __forceinline__ ArgmaxSoftmax argmax_softmax_combine(const ArgmaxSoftmax& a, const ArgmaxSoftmax& b) {
    ArgmaxSoftmax out = argmax_softmax_empty();
    out.value = fmaxf(a.value, b.value);
    out.index = argmax_better(b.value, b.index, a.value, a.index) ? b.index : a.index;
    out.sum   = (a.sum > 0.0F ? a.sum * __expf(a.value - out.value) : 0.0F) +
              (b.sum > 0.0F ? b.sum * __expf(b.value - out.value) : 0.0F);
    return out;
}

// Lane 0 receives the warp's combination (a fixed shuffle tree).
__device__ __forceinline__ void argmax_softmax_warp_reduce(ArgmaxSoftmax& s) {
    constexpr unsigned int kMask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        ArgmaxSoftmax other = argmax_softmax_empty();
        other.value = __shfl_down_sync(kMask, s.value, offset);
        other.index = __shfl_down_sync(kMask, s.index, offset);
        other.sum   = __shfl_down_sync(kMask, s.sum, offset);
        s           = argmax_softmax_combine(s, other);
    }
}

// Thread 0 receives the block's combination: each warp's, then the warps' in order.
__device__ __forceinline__ void argmax_softmax_block_reduce(ArgmaxSoftmax& s) {
    __shared__ float warp_values[16];
    __shared__ std::int32_t warp_indices[16];
    __shared__ float warp_sums[16];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    argmax_softmax_warp_reduce(s);
    if (lane == 0) {
        warp_values[warp]  = s.value;
        warp_indices[warp] = s.index;
        warp_sums[warp]    = s.sum;
    }
    __syncthreads();
    s = argmax_softmax_empty();
    if (lane < (blockDim.x >> 5)) {
        s.value = warp_values[lane];
        s.index = warp_indices[lane];
        s.sum   = warp_sums[lane];
    }
    if (warp == 0) { argmax_softmax_warp_reduce(s); }
}

// argmax_with_probability over one block of rows (FP32 logits, or at most kArgmaxBlock rows): one
// CTA per column, thread i over rows i, i + blockDim.x, ... in order.
template <class Logit>
__launch_bounds__(kArgmaxBlock) __global__
    void argmax_probability_column_kernel(const Logit* logits, std::int32_t* out, float* probability,
                                          std::int32_t valid_rows, std::int32_t physical_rows) {
    pdl::enter();
    const std::int32_t t    = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;
    ArgmaxSoftmax s         = argmax_softmax_empty();
    for (std::int32_t v = static_cast<std::int32_t>(threadIdx.x); v < valid_rows; v += blockDim.x) {
        argmax_softmax_consider(s, argmax_logit(logits[base + v]), v);
    }
    argmax_softmax_block_reduce(s);
    if (threadIdx.x == 0) {
        out[t]         = s.index;
        probability[t] = 1.0F / s.sum;
    }
}

// argmax_with_probability over BF16 row tiles (grid: tiles x columns): each CTA reads its tile once,
// publishes its winner as argmax_tiled_atomic_kernel does (out[t] zeroed before the launch) and its
// tile's maximum and rescaled sum to partial_value/partial_sum[t * tiles + tile]. The column's last
// CTA to arrive (arrivals[t], zeroed before the launch) combines the tiles in row order: the
// maximum is the argmax's logit, so probability[t] = 1 / (the combined sum).
__launch_bounds__(kArgmaxBlock) __global__
    void argmax_probability_tiled_kernel(const __nv_bfloat16* logits, std::int32_t* out, float* probability,
                                         float* partial_value, float* partial_sum, unsigned int* arrivals,
                                         std::int32_t valid_rows, std::int32_t physical_rows) {
    pdl::enter();
    const std::int32_t t     = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t tiles = static_cast<std::int32_t>(gridDim.x);
    const std::int64_t base  = static_cast<std::int64_t>(t) * physical_rows;
    const std::int32_t tile_start =
        static_cast<std::int32_t>(blockIdx.x) * blockDim.x * kArgmaxItemsPerThread;

    ArgmaxSoftmax s = argmax_softmax_empty();
    // The same row order per thread as argmax_tiled_atomic_kernel: 8-row vector loads when every
    // column starts 16-byte aligned, else strided rows.
    constexpr int kVectorRows = 8;
    if (physical_rows % kVectorRows == 0 && (reinterpret_cast<std::uintptr_t>(logits) & 15U) == 0) {
#pragma unroll
        for (int group = 0; group < kArgmaxItemsPerThread / kVectorRows; ++group) {
            const std::int32_t v =
                tile_start + (static_cast<std::int32_t>(threadIdx.x) + group * static_cast<std::int32_t>(blockDim.x)) *
                                 kVectorRows;
            if (v + kVectorRows <= valid_rows) {
                const uint4 packed = __ldg(reinterpret_cast<const uint4*>(logits + base + v));
                const auto* rows   = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
                for (int k = 0; k < kVectorRows; ++k) { argmax_softmax_consider(s, __bfloat162float(rows[k]), v + k); }
            } else {
                for (std::int32_t row = v; row < valid_rows; ++row) {
                    argmax_softmax_consider(s, __bfloat162float(logits[base + row]), row);
                }
            }
        }
    } else {
#pragma unroll
        for (int item = 0; item < kArgmaxItemsPerThread; ++item) {
            const std::int32_t v = tile_start + threadIdx.x + item * blockDim.x;
            if (v < valid_rows) { argmax_softmax_consider(s, __bfloat162float(logits[base + v]), v); }
        }
    }

    argmax_softmax_block_reduce(s);
    __shared__ bool last;
    if (threadIdx.x == 0) {
        if (s.index != INT32_MAX) {
            int current = out[t];
            while (true) {
                const float current_value = __bfloat162float(logits[base + current]);
                if (!argmax_better(s.value, s.index, current_value, current)) { break; }
                const int observed = atomicCAS(reinterpret_cast<int*>(out + t), current, s.index);
                if (observed == current) { break; }
                current = observed;
            }
        }
        const std::int64_t at = static_cast<std::int64_t>(t) * tiles + blockIdx.x;
        partial_value[at]     = s.value;
        partial_sum[at]       = s.sum;
        __threadfence(); // the partials before the arrival
        last = atomicAdd(arrivals + t, 1U) == static_cast<unsigned int>(tiles - 1);
    }
    __syncthreads();
    if (!last || threadIdx.x >= 32) { return; }
    __threadfence(); // every other tile's partials after its arrival
    ArgmaxSoftmax column = argmax_softmax_empty();
    for (std::int32_t tile = static_cast<std::int32_t>(threadIdx.x); tile < tiles; tile += 32) {
        const std::int64_t at = static_cast<std::int64_t>(t) * tiles + tile;
        ArgmaxSoftmax part    = argmax_softmax_empty();
        part.value = __ldcg(partial_value + at);
        part.index = tile; // unused: the winner is out[t]
        part.sum   = __ldcg(partial_sum + at);
        column     = argmax_softmax_combine(column, part);
    }
    argmax_softmax_warp_reduce(column);
    if (threadIdx.x == 0) { probability[t] = 1.0F / column.sum; }
}

template <class Logit>
__launch_bounds__(kArgmaxBlock) __global__
    void argmax_tiled_atomic_kernel(const Logit* logits, std::int32_t* out,
                                    std::int32_t valid_rows, std::int32_t physical_rows) {
    pdl::enter();
    const std::int32_t t    = static_cast<std::int32_t>(blockIdx.y);
    const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;
    const std::int32_t tile_start =
        static_cast<std::int32_t>(blockIdx.x) * blockDim.x * kArgmaxItemsPerThread;

    float best_value        = -CUDART_INF_F;
    std::int32_t best_index = INT32_MAX;
    const auto consider     = [&](std::int32_t row, float value) {
        if (argmax_better(value, row, best_value, best_index)) {
            best_value = value;
            best_index = row;
        }
    };
    // Each thread reads contiguous rows per 16-byte load (8 BF16 or 4 FP32) when every column starts
    // 16-byte aligned (giveen/ninfer-ext 98a5904cb); the winner is the same value/lower-id maximum
    // either way.
    constexpr int kVectorRows = 16 / static_cast<int>(sizeof(Logit));
    if (physical_rows % kVectorRows == 0 && (reinterpret_cast<std::uintptr_t>(logits) & 15U) == 0) {
#pragma unroll
        for (int group = 0; group < kArgmaxItemsPerThread / kVectorRows; ++group) {
            const std::int32_t v =
                tile_start + (static_cast<std::int32_t>(threadIdx.x) + group * static_cast<std::int32_t>(blockDim.x)) *
                                 kVectorRows;
            if (v + kVectorRows <= valid_rows) {
                const uint4 packed = __ldg(reinterpret_cast<const uint4*>(logits + base + v));
                const auto* rows   = reinterpret_cast<const Logit*>(&packed);
#pragma unroll
                for (int k = 0; k < kVectorRows; ++k) { consider(v + k, argmax_logit(rows[k])); }
            } else {
                for (std::int32_t row = v; row < valid_rows; ++row) {
                    consider(row, argmax_logit(logits[base + row]));
                }
            }
        }
    } else {
#pragma unroll
        for (int item = 0; item < kArgmaxItemsPerThread; ++item) {
            const std::int32_t v = tile_start + threadIdx.x + item * blockDim.x;
            if (v < valid_rows) { consider(v, argmax_logit(logits[base + v])); }
        }
    }

    argmax_block_reduce(best_value, best_index);
    if (threadIdx.x != 0 || best_index == INT32_MAX) { return; }

    int current = out[t];
    while (true) {
        const float current_value = argmax_logit(logits[base + current]);
        if (!argmax_better(best_value, best_index, current_value, current)) { break; }
        const int observed = atomicCAS(reinterpret_cast<int*>(out + t), current, best_index);
        if (observed == current) { break; }
        current = observed;
    }
}

} // namespace infernix::ops
