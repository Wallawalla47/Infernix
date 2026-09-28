#pragma once

// Launch of the fast NVFP4 prompt kernel: its split partials, the partial kernel and the split
// merge.

#include "core/arena.h"
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/operands.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/fast_tiled_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/fast_tiled_plan.h"

#include <stdexcept>

namespace ninfer::ops::detail {

// causal_softmax_attention_prompt_wave_tokens() counts waves of these rows.
static_assert(CausalPromptRotatedShape<8>::Br == 128);

// One complete query row whose K/V are already in the paged cache, as the tiled kernel.
template <class G>
void launch_nvfp4_kv_fast_tiled_mma(const CausalAttentionOperands& p, Nvfp4KvReadView cache,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    validate_quantized_causal_operands<G>(p, cache);
    if (p.batch != 1)
        throw std::invalid_argument("fast prompt attention requires a complete single query row");
    const RotatedFastPromptPlan plan =
        rotated_fast_prompt_plan(G::QHeads, p.width, p.visible_capacity);
    auto scope = workspace.scope();
    RotatedFastPromptPartials partials{};
    if (plan.splits > 1)
        partials = allocate_rotated_fast_prompt_partials(workspace, G::QHeads, p.width, plan.splits);
    const auto invoke = [&]<class Metadata>(Metadata metadata, const std::int32_t* valid) {
        const auto launch = [&]<int Warps, bool Split>() {
            using Shape = CausalPromptRotatedShape<Warps>;
            constexpr auto kernel =
                causal_attention_prompt_rotated_fast_kernel<G, Metadata, Warps, Split>;
            static const auto status = cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Shape::SmemBytes);
            CUDA_CHECK(status);
            const dim3 grid(div_up(p.width, Shape::Br), G::QHeads, plan.splits);
            kernel<<<grid, Shape::Threads, Shape::SmemBytes, stream>>>(
                p.q, cache.keys, cache.values, cache.key_scales, cache.value_scales, metadata,
                p.positions, p.scale, p.out, p.width, static_cast<float*>(partials.rows.data),
                static_cast<float2*>(partials.stats.data));
            CUDA_CHECK(cudaGetLastError());
        };
        const auto dispatch = [&]<bool Split>() {
            if (plan.warps == 4)
                launch.template operator()<4, Split>();
            else
                launch.template operator()<8, Split>();
        };
        if (plan.splits == 1) {
            dispatch.template operator()<false>();
            return;
        }
        dispatch.template operator()<true>();
        constexpr float Log2E = 1.4426950408889634074f;
        causal_attention_prompt_rotated_fast_merge_kernel<G>
            <<<dim3(p.width, G::QHeads), kCausalPromptHeadDim, 0, stream>>>(
                static_cast<const float*>(partials.rows.data),
                static_cast<const float2*>(partials.stats.data), valid, p.width, plan.splits,
                p.scale * Log2E, p.out);
        CUDA_CHECK(cudaGetLastError());
    };
    if (!cache.table_rows)
        invoke(PagedKVDirectMetadata{cache.tables}, nullptr);
    else if (cache.valid_columns)
        invoke(PagedKVBatchMetadata<true>{cache.tables, cache.valid_columns, cache.table_rows,
                                          cache.table_stride},
               cache.valid_columns);
    else
        invoke(PagedKVBatchMetadata<false>{cache.tables, nullptr, cache.table_rows,
                                           cache.table_stride},
               nullptr);
}

} // namespace ninfer::ops::detail
