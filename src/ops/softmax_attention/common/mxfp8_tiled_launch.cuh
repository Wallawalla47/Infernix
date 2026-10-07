#pragma once

#include "core/device.h"
#include "ops/softmax_attention/common/mxfp8_tiled_mma.cuh"
#include "ops/softmax_attention/common/mxfp8_tiled_plan.h"
#include <stdexcept>

namespace infernix::ops::detail {

// pv8 selects the 8-bit PV form (Values with E4M3 decode only).
template <class G, class S, class Keys, class Values, class View>
void launch_mxfp8_kv_tiled_mma(const CausalAttentionOperands& p, View cache,
                               CausalKvPartition partition, CausalPartialView partial, bool pv8,
                               cudaStream_t stream) {
    validate_quantized_causal_operands<G>(p, cache);
    if (p.batch != 1 || partition.target < 1 || partition.target > kMxfp8TiledMaxSplits ||
        partition.capacity != partition.active(p.visible_capacity) || !partial.acc ||
        !partial.maximum || !partial.sum)
        throw std::invalid_argument("MXFP8 tiled attention: invalid batch or partial storage");
    if (pv8 && !Values::kE4m3Values)
        throw std::invalid_argument("MXFP8 tiled attention: no 8-bit PV form for this V storage");
    const auto launch = [&]<class Metadata, bool Pv8>(Metadata metadata) {
        constexpr auto kernel    = mxfp8_kv_tiled_mma_kernel<G, S, Keys, Values, Metadata, Pv8>;
        static const auto status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, S::kSharedBytes);
        CUDA_CHECK(status);
        const dim3 grid(div_up(p.width, S::kQueryRows), G::QHeads, partition.capacity);
        kernel<<<grid, S::kThreads, S::kSharedBytes, stream>>>(
            p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
            p.positions, p.scale, p.width, partition, partial);
        CUDA_CHECK(cudaGetLastError());
    };
    const auto invoke = [&]<class Metadata>(Metadata metadata) {
        if constexpr (Values::kE4m3Values) {
            if (pv8) {
                launch.template operator()<Metadata, true>(metadata);
                return;
            }
        }
        launch.template operator()<Metadata, false>(metadata);
    };
    if (!cache.table_rows)
        invoke(PagedKVDirectMetadata{cache.tables});
    else if (cache.valid_columns)
        invoke(PagedKVBatchMetadata<true>{cache.tables, cache.valid_columns, cache.table_rows,
                                          cache.table_stride});
    else
        invoke(PagedKVBatchMetadata<false>{cache.tables, nullptr, cache.table_rows,
                                           cache.table_stride});
}

} // namespace infernix::ops::detail
