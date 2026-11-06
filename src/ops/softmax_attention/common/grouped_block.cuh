#pragma once

// The grid of the NVFP4-KV grouped decode/verify attention kernel: one CTA per (batch row, KV-head or
// column tile, key split).

#include <cuda_runtime.h>

namespace infernix::ops::detail {

// With several batch rows (MultiBatch) the batch row is the fastest grid axis, then the tile, then the
// key split, so the CTAs of every row for one key split are dispatched together; one row keeps the tile,
// split order. NVFP4 KV, one column (RTX 5090, candidates §59): two rows 43.6 -> 41.6 us at 16,384 keys and
// 123.7 -> 115.8 at 61,440, four rows 72.2 -> 68.2 and 230.2 -> 218.0; four columns, two rows at 16,384
// keys 55.8 -> 53.1; level at 4,096 keys and at eight columns. The other formats keep tile, split, batch:
// on them this order measured level or slower (INT8, one column, four rows on shared pages: 175 -> 187 us
// at 61,440 keys). A CTA's keys, arithmetic and partial slot do not depend on the order, so the bits are
// unchanged.
struct GroupedAttentionBlock {
    int tile;
    int split;
    int split_count;
    int batch;
};

template <bool MultiBatch>
__device__ __forceinline__ GroupedAttentionBlock grouped_attention_block() {
    if constexpr (MultiBatch) {
        return {static_cast<int>(blockIdx.y), static_cast<int>(blockIdx.z),
                static_cast<int>(gridDim.z), static_cast<int>(blockIdx.x)};
    } else {
        return {static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y),
                static_cast<int>(gridDim.y), 0};
    }
}

template <bool MultiBatch>
inline dim3 grouped_attention_grid(int tiles, int splits, int batch) {
    if constexpr (MultiBatch) {
        return dim3(static_cast<unsigned>(batch), static_cast<unsigned>(tiles),
                    static_cast<unsigned>(splits));
    } else {
        return dim3(static_cast<unsigned>(tiles), static_cast<unsigned>(splits), 1U);
    }
}

} // namespace infernix::ops::detail
