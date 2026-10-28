#pragma once

// VQ2 attention's route templates (the grouped decode/verification route and the prompt route), shared by
// launch.cu and the four translation units that instantiate them, a route of a key codec each
// (grouped_vq2.cu, grouped_q4.cu, prompt_vq2.cu, prompt_q4.cu): one launch.cu compiled about 550 kernels
// (2 codecs x 3 head geometries x every width, batch, mask, tree and prompt shape) in one 619 s job, the
// build's longest (Ninja, 2026-10-28); split, they compile in parallel. The same kernels either way.

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/common/causal_merge.cuh"
#include "ops/softmax_attention/dense/causal_cache/fast_prompt_plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/fast_tiled_plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/schedule.cuh"
#include "ops/softmax_attention/dense/causal_cache/vq2/grouped.cuh"
#include "ops/softmax_attention/dense/causal_cache/vq2/prompt.cuh"

namespace infernix::ops::detail {

struct VqKvCausalPlan {
    Int8KvCausalPlan plan;
    int window_splits = 0;
    int parallel_tile = 8; // kCausalGroupedTokens or kVqParallelTile of the geometry
    // The prompt route's split plan and its FP32 partials must use the workspace bound the
    // planning call was given (CausalAttentionExecutionEnvelope::prompt_split_workspace_bytes),
    // so the plan carries it to the launch.
    std::size_t prompt_split_workspace_bytes = kCausalPromptSplitWorkspaceDefaultBytes;
    // CausalAttentionExecutionEnvelope::fast_prompt_pv8: the prompt route's PV runs on 8-bit
    // Tensor Cores over the tile's INT8 V rows. It changes no allocation, so the plan carries it.
    bool prompt_pv8 = false;
};

// The grouped route over the planned partials, and the prompt route (a wide call's tiles).
template <KVCacheVqKeyCodec Codec>
void vq_grouped_attention(const CausalAttentionOperands& p, const VqKvCacheView& cache, const VqKvCausalPlan& vq,
                          CausalPartialView partial, cudaStream_t stream);
template <KVCacheVqKeyCodec Codec>
void vq_prompt_attention(const CausalAttentionOperands& p, const VqKvCacheView& cache, const VqKvCausalPlan& vq,
                         const KVCacheVqStaging* staging, WorkspaceArena& workspace, cudaStream_t stream);

namespace vq_routes {

// The natural merge of the grouped kernels applies the inverse rotation over whole rows.
using VqMergeSchedule = Int8KvMergeSchedule<256>;

// Keys per tile of the grouped kernel. A 64-key tile halves the number of tiles and their barriers,
// which is worth 7-11 % on vq2 where a block already has the SM to itself (the sixteen-column CTA,
// measured at 2-8K keys: W=16 takes 1.71x INT8 against 1.89x at 32 keys, B=1). It is not worth it
// anywhere else: an eight-column block loses its second CTA per SM to the extra shared memory and
// slows by 13-44 %, and single-token decode would lose it too. So 64 keys need the sixteen-warp CTA
// (above 64 query rows), and must fit one block: k4v2's 130-byte K rows never do at that width, so
// it keeps 32 everywhere.
template <class Geometry, int Tokens, KVCacheVqKeyCodec Codec>
inline constexpr int kVqGroupedKeyRows =
    kVqGroupedWarps<Geometry, Tokens> == 16 &&
            vq_kv_grouped_smem_bytes<Codec, 64, Tokens, Geometry::GroupSize>() <=
                      kVqGroupedSmemLimit
        ? 64
        : 32;

template <class G, KVCacheVqKeyCodec Codec, int Tokens, bool MultiBatch, bool Masked, bool Parallel>
void grouped(const CausalAttentionOperands& p, const VqKvCacheView& cache,
             CausalKvPartition partition, int window_splits, CausalPartialView partial,
             cudaStream_t stream) {
    using Shape            = VqKvGroupedShape<Codec, kVqGroupedKeyRows<G, Tokens, Codec>>;
    constexpr int kThreads = kVqGroupedWarps<G, Tokens> * 32;
    validate_vq_causal_operands<G>(p, cache);
    if ((!Parallel && p.width != Tokens) || MultiBatch != (p.batch > 1) ||
        Masked != (cache.valid_columns != nullptr) || partition.capacity < 1 ||
        partition.target > CausalKvPartition::kMaxSplits || partition.target < 1 ||
        partition.key_shift < 6 || partition.key_shift > 12 ||
        partition.capacity != partition.bound(p.visible_capacity) || !partial.acc ||
        !partial.maximum || !partial.sum)
        throw std::invalid_argument("VQ grouped attention: invalid schedule/partials");
    if (window_splits < 1 || window_splits > kVqMaxWindowSplits ||
        window_splits + partition.capacity > CausalKvPartition::kMaxSplits)
        throw std::invalid_argument("VQ grouped attention: invalid split count");
    // Verification trees arrive only on masked rows of three or more columns.
    constexpr bool TreeCapable = Masked && Tokens >= 3;
    if (!TreeCapable && cache.tree_masks)
        throw std::invalid_argument("VQ grouped attention: this route has no tree");
    const auto launch = [&]<bool Tree>() {
        constexpr auto kernel =
            vq_kv_grouped_kernel<G, Tokens, MultiBatch, Masked, Codec,
                                 kVqGroupedKeyRows<G, Tokens, Codec>, Parallel, Tree>;
        static const auto status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Shape::kArenaBytes);
        CUDA_CHECK(status);
        const dim3 grid(G::KVHeads * (Parallel ? div_up(p.width, Tokens) : 1),
                        window_splits + partition.capacity, p.batch);
        CUDA_CHECK(pdl::launch_consumer({grid, dim3(kThreads), Shape::kArenaBytes, stream}, kernel,
                                        p.q, p.positions, cache, p.width, p.visible_capacity,
                                        partition, window_splits, p.scale, partial.acc,
                                        partial.maximum, partial.sum));
    };
    if constexpr (TreeCapable) {
        if (cache.tree_masks) {
            launch.template operator()<true>();
        } else {
            launch.template operator()<false>();
        }
    } else {
        launch.template operator()<false>();
    }
    launch_causal_natural_merge<G, VqMergeSchedule, MultiBatch, Masked, true>(
        p, cache.valid_columns, vq_merge_partition(window_splits + partition.capacity), partial,
        stream);
}

template <class G, KVCacheVqKeyCodec Codec>
void grouped_routes(const CausalAttentionOperands& p, const VqKvCacheView& cache,
                    const VqKvCausalPlan& vq, CausalPartialView partial, cudaStream_t stream) {
    const CausalKvPartition partition = vq.plan.partition;
    const int window_splits           = vq.window_splits;
    const auto batch = [&]<int Tokens, bool Parallel>() {
        const bool masked = cache.valid_columns != nullptr;
        if (p.batch == 1) {
            if (masked)
                grouped<G, Codec, Tokens, false, true, Parallel>(p, cache, partition, window_splits,
                                                                 partial, stream);
            else
                grouped<G, Codec, Tokens, false, false, Parallel>(p, cache, partition,
                                                                  window_splits, partial, stream);
        } else {
            if (masked)
                grouped<G, Codec, Tokens, true, true, Parallel>(p, cache, partition, window_splits,
                                                                partial, stream);
            else
                grouped<G, Codec, Tokens, true, false, Parallel>(p, cache, partition, window_splits,
                                                                 partial, stream);
        }
    };
    if (vq.plan.family == Int8KvFamily::ParallelGrouped) {
        if (vq.parallel_tile == kVqParallelTile<G>)
            batch.template operator()<kVqParallelTile<G>, true>();
        else
            batch.template operator()<kCausalGroupedTokens<G>, true>();
        return;
    }
    switch (p.width) {
#define INFERNIX_VQ_GROUPED(T)                                                                       \
    case T:                                                                                        \
        if constexpr (T <= kCausalGroupedTokens<G>) {                                          \
            return batch.template operator()<T, false>();                                      \
        }                                                                                      \
        break
        INFERNIX_VQ_GROUPED(1);
        INFERNIX_VQ_GROUPED(2);
        INFERNIX_VQ_GROUPED(3);
        INFERNIX_VQ_GROUPED(4);
        INFERNIX_VQ_GROUPED(5);
        INFERNIX_VQ_GROUPED(6);
        INFERNIX_VQ_GROUPED(7);
        INFERNIX_VQ_GROUPED(8);
#undef INFERNIX_VQ_GROUPED
    }
    throw std::logic_error("VQ grouped plan exceeds the selected token tile");
}

template <class G, KVCacheVqKeyCodec Codec>
void prompt(const CausalAttentionOperands& p, const VqKvCacheView& cache,
            const KVCacheVqStaging* staging, std::size_t split_budget, bool pv8,
            WorkspaceArena& workspace, cudaStream_t stream) {
    validate_vq_causal_operands<G>(p, cache);
    if (p.batch != 1)
        throw std::invalid_argument("VQ prompt attention requires a complete single query row");
    const FastPromptPlan plan =
        int8_fast_prompt_plan(G::QHeads, p.width, p.visible_capacity, split_budget);
    auto scope                = workspace.scope();
    FastPromptPartials partials{};
    if (plan.splits > 1)
        partials = allocate_fast_prompt_partials(workspace, G::QHeads, p.width, plan.splits);
    const VqPromptStaging stage_source =
        staging ? VqPromptStaging{staging->k_codes, staging->v_codes, staging->k_scales,
                                  staging->v_scales}
                : VqPromptStaging{nullptr, nullptr, nullptr, nullptr};
    const auto launch_prompt = [&]<int Warps, bool Split, bool Pv8, int Bc, class Metadata>(
                                   Metadata metadata) {
        using Shape           = VqPromptShape<Codec, Warps, Bc>;
        constexpr auto kernel = vq_prompt_kernel<G, Metadata, Codec, Warps, Split, Pv8, Bc>;
        static const auto status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Shape::SmemBytes);
        CUDA_CHECK(status);
        const dim3 grid(G::QHeads, div_up(p.width, Shape::Br), plan.splits);
        kernel<<<grid, Shape::Threads, Shape::SmemBytes, stream>>>(
            p.q, cache, metadata, stage_source, p.positions, p.scale, p.out, p.width,
            static_cast<float*>(partials.rows.data), static_cast<float2*>(partials.stats.data));
        CUDA_CHECK(cudaGetLastError());
    };
    const auto invoke = [&]<class Metadata>(Metadata metadata, const std::int32_t* valid) {
        // 64-key tiles once the call sees kVqPromptWideTileKeys keys (mostly code tiles, whose load
        // the prefetch hides); 32-key tiles with double-buffered INT8 tiles for shorter calls,
        // whose tiles are mostly exact rows loaded before use.
        const bool wide_tiles = p.visible_capacity >= kVqPromptWideTileKeys;
        const auto launch     = [&]<int Warps, bool Split>() {
            if (pv8) {
                if (wide_tiles)
                    launch_prompt.template operator()<Warps, Split, true, kVqPromptWideBc, Metadata>(
                        metadata);
                else
                    launch_prompt.template operator()<Warps, Split, true, kVqPromptBc, Metadata>(
                        metadata);
                return;
            }
            if (wide_tiles)
                launch_prompt.template operator()<Warps, Split, false, kVqPromptWideBc, Metadata>(
                    metadata);
            else
                launch_prompt.template operator()<Warps, Split, false, kVqPromptBc, Metadata>(
                    metadata);
        };
        if (plan.splits == 1) {
            if (plan.warps == 4)
                launch.template operator()<4, false>();
            else
                launch.template operator()<8, false>();
            return;
        }
        launch.template operator()<8, true>();
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

} // namespace vq_routes

template <KVCacheVqKeyCodec Codec>
void vq_grouped_attention(const CausalAttentionOperands& p, const VqKvCacheView& cache, const VqKvCausalPlan& vq,
                          CausalPartialView partial, cudaStream_t stream) {
    with_causal_geometry(p.query_heads, [&]<class G>() { vq_routes::grouped_routes<G, Codec>(p, cache, vq, partial, stream); });
}

template <KVCacheVqKeyCodec Codec>
void vq_prompt_attention(const CausalAttentionOperands& p, const VqKvCacheView& cache, const VqKvCausalPlan& vq,
                         const KVCacheVqStaging* staging, WorkspaceArena& workspace, cudaStream_t stream) {
    with_causal_geometry(p.query_heads, [&]<class G>() {
        vq_routes::prompt<G, Codec>(p, cache, staging, vq.prompt_split_workspace_bytes, vq.prompt_pv8, workspace, stream);
    });
}

// Instantiated in grouped_vq2.cu, grouped_q4.cu, prompt_vq2.cu and prompt_q4.cu.
extern template void vq_grouped_attention<KVCacheVqKeyCodec::Vq2>(const CausalAttentionOperands&, const VqKvCacheView&,
                                                                  const VqKvCausalPlan&, CausalPartialView, cudaStream_t);
extern template void vq_grouped_attention<KVCacheVqKeyCodec::Q4>(const CausalAttentionOperands&, const VqKvCacheView&,
                                                                 const VqKvCausalPlan&, CausalPartialView, cudaStream_t);
extern template void vq_prompt_attention<KVCacheVqKeyCodec::Vq2>(const CausalAttentionOperands&, const VqKvCacheView&,
                                                                 const VqKvCausalPlan&, const KVCacheVqStaging*,
                                                                 WorkspaceArena&, cudaStream_t);
extern template void vq_prompt_attention<KVCacheVqKeyCodec::Q4>(const CausalAttentionOperands&, const VqKvCacheView&,
                                                                const VqKvCausalPlan&, const KVCacheVqStaging*,
                                                                WorkspaceArena&, cudaStream_t);

} // namespace infernix::ops::detail
