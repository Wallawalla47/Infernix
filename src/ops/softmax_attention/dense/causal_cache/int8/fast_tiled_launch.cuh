#pragma once

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/int8/fast_tiled_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/operands.h"
#include <stdexcept>

namespace ninfer::ops::detail {

// causal_softmax_attention_prompt_wave_tokens() counts waves of these rows.
static_assert(CausalPromptI8FastShape<8>::Br == 128);

// Both fast-kernel CTA shapes run one CTA per SM and every CTA of a launch sweeps a similar key
// range, so a launch costs about (waves) x (one CTA's sweep). A four-warp CTA sweeps in about 0.72
// of an eight-warp CTA's time (measured on RTX 5090 at 64K context) but covers half the rows.
inline bool int8_kv_fast_tiled_prefers_narrow(int tokens, int q_heads) {
    static const int multiprocessors = [] {
        int device = 0;
        int count  = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
        return count;
    }();
    const auto waves = [&](int rows) {
        return div_up(div_up(tokens, rows) * q_heads, multiprocessors);
    };
    constexpr int NarrowCostPercent = 72;
    return waves(CausalPromptI8FastShape<4>::Br) * NarrowCostPercent <
           waves(CausalPromptI8FastShape<8>::Br) * 100;
}

// The fast prompt kernel takes the same operands as launch_int8_kv_tiled_mma: one complete query
// row whose K/V are already in the paged cache. pv8 selects its 8-bit PV form.
template <class G>
void launch_int8_kv_fast_tiled_mma(const CausalAttentionOperands& p, Int8KvReadView cache, bool pv8,
                                   cudaStream_t stream) {
    validate_quantized_causal_operands<G>(p, cache);
    if (p.batch != 1)
        throw std::invalid_argument(
            "INT8 fast prompt attention requires a complete single query row");
    const auto invoke = [&]<class Metadata>(Metadata metadata) {
        const auto launch = [&]<int Warps, bool Pv8>() {
            using Shape           = CausalPromptI8FastShape<Warps>;
            constexpr auto kernel = causal_attention_prompt_i8_fast_kernel<G, Metadata, Warps, Pv8>;
            static const auto status = cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Shape::SmemBytes);
            CUDA_CHECK(status);
            const dim3 grid(div_up(p.width, Shape::Br), G::QHeads);
            kernel<<<grid, Shape::Threads, Shape::SmemBytes, stream>>>(
                p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
                p.positions, p.scale, p.out, p.width);
            CUDA_CHECK(cudaGetLastError());
        };
        const bool narrow = int8_kv_fast_tiled_prefers_narrow(p.width, G::QHeads);
        if (pv8) {
            if (narrow)
                launch.template operator()<4, true>();
            else
                launch.template operator()<8, true>();
        } else if (narrow) {
            launch.template operator()<4, false>();
        } else {
            launch.template operator()<8, false>();
        }
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

} // namespace ninfer::ops::detail
