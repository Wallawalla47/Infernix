#pragma once

#include "core/layout.h"
#include "ops/common/math.h"

#include <cstdint>

namespace infernix::ops::detail {

inline constexpr std::int32_t kLinearTopK                = 16;
inline constexpr std::int32_t kLinearTopKMaxChunkColumns = 128;
inline constexpr std::int32_t kLinearTopKGroupedRows     = 128;
inline constexpr std::int32_t kLinearTopKDirectRows      = 16;
// The keys each selection CTA reduces per column (256 threads of 16).
inline constexpr std::int32_t kLinearTopKSliceKeys       = 4096;
inline constexpr std::int32_t kLinearTopKFullRows        = 248320;
inline constexpr std::int32_t kLinearTopKFullValidRows   = 248077;
inline constexpr std::int32_t kLinearTopKOptimizedRows   = 131072;
inline constexpr std::int32_t kLinearTopKHidden          = 5120;
inline constexpr std::int32_t kLinearTopKFlashNextHidden = 2560;
inline constexpr std::int32_t kLinearTopK122BHidden      = 3072;

struct LinearTopKWorkspace {
    std::int32_t tile_columns = 0;
    std::int32_t block_k      = 128;
    Tensor partial_keys; // [16, producer_groups, columns]: each producer group's keys
    Tensor slice_keys;   // [16, slices, columns]: each slice's top sixteen, descending
    std::int32_t producer_groups   = 0;
    std::int32_t slices            = 0;
    std::int32_t columns           = 0;
    std::int32_t rows_per_producer = 0;
};

template <class Allocator>
LinearTopKWorkspace
allocate_linear_topk_workspace(Allocator& allocator, std::int32_t head_rows, std::int32_t columns,
                               std::int32_t rows_per_producer, std::int32_t tile_columns = 0,
                               std::int32_t block_k = 128) {
    const std::int32_t producer_groups = div_up(head_rows, rows_per_producer);
    const std::int32_t slices          = div_up(producer_groups * kLinearTopK, kLinearTopKSliceKeys);
    LinearTopKWorkspace out;
    out.tile_columns      = tile_columns;
    out.block_k           = block_k;
    out.partial_keys      = allocator.alloc(DType::I64, {kLinearTopK, producer_groups, columns}, 256);
    out.slice_keys        = allocator.alloc(DType::I64, {kLinearTopK, slices, columns}, 256);
    out.producer_groups   = producer_groups;
    out.slices            = slices;
    out.columns           = columns;
    out.rows_per_producer = rows_per_producer;
    return out;
}

} // namespace infernix::ops::detail
