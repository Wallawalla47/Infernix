#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/softmax_attention/dense/causal_cache/int8/fast_tiled_mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/int8/operands.h"
#include "ops/softmax_attention/dense/causal_cache/int8/fast_tiled_plan.h"
#include <stdexcept>

namespace infernix::ops::detail {

// causal_softmax_attention_prompt_wave_tokens() counts waves of these rows.
static_assert(CausalPromptI8FastShape<8>::Br == 128);

// One complete query row whose K/V are already in the paged cache, as launch_int8_kv_tiled_mma.
// envelope.fast_prompt_pv8 selects the 8-bit PV form. int8_fast_prompt_plan() picks the CTA shape
// and splits the keys of an eight-warp launch whose row blocks alone would leave SMs idle, within
// envelope.prompt_split_workspace_bytes; the split partials come from the workspace.
template <class G>
void launch_int8_kv_fast_tiled_mma(const CausalAttentionOperands& p, Int8KvReadView cache,
                                   CausalAttentionExecutionEnvelope envelope,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    const bool pv8 = envelope.fast_prompt_pv8;
    validate_quantized_causal_operands<G>(p, cache);
    if (p.batch != 1)
        throw std::invalid_argument(
            "INT8 fast prompt attention requires a complete single query row");
    const FastPromptPlan plan = int8_fast_prompt_plan(G::QHeads, p.width, p.visible_capacity,
                                                      envelope.prompt_split_workspace_bytes);
    auto scope                = workspace.scope();
    FastPromptPartials partials{};
    if (plan.splits > 1)
        partials = allocate_fast_prompt_partials(workspace, G::QHeads, p.width, plan.splits);
    const auto invoke = [&]<class Metadata>(Metadata metadata, const std::int32_t* valid) {
        const auto launch = [&]<int Warps, bool Pv8, bool Split>() {
            using Shape = CausalPromptI8FastShape<Warps>;
            constexpr auto kernel =
                causal_attention_prompt_i8_fast_kernel<G, Metadata, Warps, Pv8, Split>;
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
        if (plan.splits == 1) {
            if (pv8) {
                if (plan.warps == 4)
                    launch.template operator()<4, true, false>();
                else
                    launch.template operator()<8, true, false>();
            } else if (plan.warps == 4) {
                launch.template operator()<4, false, false>();
            } else {
                launch.template operator()<8, false, false>();
            }
            return;
        }
        // Only eight-warp CTAs split.
        if (pv8)
            launch.template operator()<8, true, true>();
        else
            launch.template operator()<8, false, true>();
        constexpr float Log2E = 1.4426950408889634074f;
        causal_attention_prompt_fast_merge_kernel<G>
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

} // namespace infernix::ops::detail
