#pragma once

#include <cstdint>

namespace infernix::ops::detail {

// Speculative verification trees (infernix/ops/speculative_tree.h). Column a of a verification row
// appends its key at block_start + a, block_start being the position of the row's first column.
// Bit a of a query column's ancestor mask admits that key; keys before the block (the committed
// prefix) stay visible under the causal rule alone.
__device__ __forceinline__ bool causal_tree_visible(int key, int block_start, std::uint32_t mask) {
    const int offset = key - block_start;
    return offset < 0 || offset >= 32 || ((mask >> offset) & 1u) != 0u;
}

// Ancestor mask of column `column` of verification row `row` in a [B][W] mask matrix; every key
// is visible without a tree.
template <bool Tree>
__device__ __forceinline__ std::uint32_t causal_tree_mask(const std::uint32_t* masks, int row,
                                                          int width, int column) {
    if constexpr (!Tree) {
        return ~0u;
    } else {
        return masks[static_cast<std::int64_t>(row) * width + column];
    }
}

} // namespace infernix::ops::detail
