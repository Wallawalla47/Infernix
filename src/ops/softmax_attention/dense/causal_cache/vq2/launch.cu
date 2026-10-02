#include "ops/softmax_attention/dense/causal_cache/vq2/launch.h"

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

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// The natural merge of the grouped kernels applies the inverse rotation over whole rows.
using VqMergeSchedule = Int8KvMergeSchedule<256>;

template <class Allocator>
KVCacheVqStaging allocate_vq_staging(Allocator& allocator, int width, int kv_heads) {
    return {
        static_cast<std::int8_t*>(allocator.alloc(DType::I8, {256, width, kv_heads}).data),
        static_cast<std::int8_t*>(allocator.alloc(DType::I8, {256, width, kv_heads}).data),
        static_cast<__half*>(allocator.alloc(DType::FP16, {4, width, kv_heads}).data),
        static_cast<__half*>(allocator.alloc(DType::FP16, {4, width, kv_heads}).data),
        static_cast<std::int32_t*>(allocator.alloc(DType::I32, {2, width, kv_heads}).data),
    };
}

// The grouped schedule: one wave of CTAs per independent tile (a row's kv head and column tile;
// two CTAs per SM at eight warps, one at sixteen), window_splits of them on the window and the rest
// on the codes under the INT8 rule's key minimums (plan.partition). Rows of 9-256 columns take
// 16-column CTAs up to kVqWideTileMaxKeys visible keys (one expansion of each tile for sixteen
// columns) and 8-column CTAs beyond, where the sixteen-warp CTA's single residency per SM costs
// more than the second expansion (W = 16 at 128K keys: 421 against 359 us).
inline constexpr std::uint32_t kVqWideTileMaxKeys = 32768;
// Keys per tile of the grouped kernel. A 64-key tile halves the number of tiles and their barriers,
// which is worth 7-11 % on vq2 where a block already has the SM to itself (the sixteen-column CTA,
// measured at 2-8K keys: W=16 takes 1.71x INT8 against 1.89x at 32 keys, B=1). It is not worth it
// anywhere else: an eight-column block loses its second CTA per SM to the extra shared memory and
// slows by 13-44 %, and single-token decode would lose it too. So 64 keys need Tokens > 8, and must
// fit one block: k4v2's 130-byte K rows never do at that width, so it keeps 32 everywhere.
template <class Geometry, int Tokens, KVCacheVqKeyCodec Codec>
inline constexpr int kVqGroupedKeyRows =
    Tokens > 8 && vq_kv_grouped_smem_bytes<Codec, 64, Tokens, Geometry::GroupSize>() <=
                      kVqGroupedSmemLimit
        ? 64
        : 32;

struct VqKvCausalPlan {
    Int8KvCausalPlan plan;
    int window_splits = 0;
    int parallel_tile = 8;
    // The prompt route's split plan and its FP32 partials must use the workspace bound the
    // planning call was given (CausalAttentionExecutionEnvelope::prompt_split_workspace_bytes),
    // so the plan carries it to the launch.
    std::size_t prompt_split_workspace_bytes = kCausalPromptSplitWorkspaceDefaultBytes;
};

VqKvCausalPlan make_vq_kv_causal_plan(int heads, int width, int batch,
                                      CausalAttentionExecutionEnvelope envelope) {
    VqKvCausalPlan out{
        make_int8_kv_causal_plan(heads, width, batch, envelope, fast_prompt_multiprocessors()), 0};
    out.prompt_split_workspace_bytes = envelope.prompt_split_workspace_bytes;
    if (out.plan.family == Int8KvFamily::Tiled) return out;
    const bool parallel = out.plan.family == Int8KvFamily::ParallelGrouped;
    out.parallel_tile   = envelope.max_visible_keys <= kVqWideTileMaxKeys ? kVqParallelTile : 8;
    const bool wide     = parallel && out.parallel_tile == kVqParallelTile;
    const int tiles       = parallel ? div_up(width, out.parallel_tile) : 1;
    const int independent = batch * (heads == 24 ? 4 : 2) * tiles;
    const int ctas =
        std::max(2, (wide ? 1 : 2) * fast_prompt_multiprocessors() / independent);
    out.window_splits     = std::clamp(ctas / 8, 1, kVqMaxWindowSplits);
    CausalKvPartition& partition = out.plan.partition;
    partition.target =
        std::clamp(ctas - out.window_splits, 1, CausalKvPartition::kMaxSplits - out.window_splits);
    if (partition.balance_shift != 0) partition.balance_limit = partition.target << 8;
    partition.capacity = partition.bound(envelope.max_visible_keys);
    return out;
}

template <class G, KVCacheVqKeyCodec Codec, int Tokens, bool MultiBatch, bool Masked, bool Parallel>
void grouped(const CausalAttentionOperands& p, const VqKvCacheView& cache,
             CausalKvPartition partition, int window_splits, CausalPartialView partial,
             cudaStream_t stream) {
    using Shape            = VqKvGroupedShape<Codec, kVqGroupedKeyRows<G, Tokens, Codec>>;
    constexpr int kThreads = kVqGroupedWarps<Tokens> * 32;
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
        if (vq.parallel_tile == kVqParallelTile)
            batch.template operator()<kVqParallelTile, true>();
        else
            batch.template operator()<8, true>();
        return;
    }
    switch (p.width) {
#define NINFER_VQ_GROUPED(T)                                                                       \
    case T:                                                                                        \
        return batch.template operator()<T, false>()
        NINFER_VQ_GROUPED(1);
        NINFER_VQ_GROUPED(2);
        NINFER_VQ_GROUPED(3);
        NINFER_VQ_GROUPED(4);
        NINFER_VQ_GROUPED(5);
        NINFER_VQ_GROUPED(6);
        NINFER_VQ_GROUPED(7);
        NINFER_VQ_GROUPED(8);
#undef NINFER_VQ_GROUPED
    }
    throw std::logic_error("VQ grouped plan exceeds the selected token tile");
}

template <class G, KVCacheVqKeyCodec Codec>
void prompt(const CausalAttentionOperands& p, const VqKvCacheView& cache,
            const KVCacheVqStaging* staging, std::size_t split_budget, WorkspaceArena& workspace,
            cudaStream_t stream) {
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
    const auto launch_prompt = [&]<int Warps, bool Split, int Bc, class Metadata>(Metadata metadata) {
        using Shape           = VqPromptShape<Codec, Warps, Bc>;
        constexpr auto kernel = vq_prompt_kernel<G, Metadata, Codec, Warps, Split, Bc>;
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
            if (wide_tiles)
                launch_prompt.template operator()<Warps, Split, kVqPromptWideBc, Metadata>(metadata);
            else
                launch_prompt.template operator()<Warps, Split, kVqPromptBc, Metadata>(metadata);
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

template <KVCacheVqKeyCodec Codec>
void attention(const CausalAttentionOperands& p, const VqKvCacheView& cache,
               const VqKvCausalPlan& vq, const KVCacheVqStaging* staging,
               WorkspaceArena& workspace, cudaStream_t stream) {
    const Int8KvCausalPlan& plan = vq.plan;
    if (plan.family == Int8KvFamily::Tiled) {
        if (p.query_heads == 24)
            prompt<CausalD256H24Kv4, Codec>(p, cache, staging, vq.prompt_split_workspace_bytes,
                                            workspace, stream);
        else
            prompt<CausalD256H16Kv2, Codec>(p, cache, staging, vq.prompt_split_workspace_bytes,
                                            workspace, stream);
        return;
    }
    auto scope         = workspace.scope();
    const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                                  vq.window_splits + plan.partition.capacity,
                                                  plan.batch);
    if (p.query_heads == 24)
        grouped_routes<CausalD256H24Kv4, Codec>(p, cache, vq, partial.view(), stream);
    else
        grouped_routes<CausalD256H16Kv2, Codec>(p, cache, vq, partial.view(), stream);
}

void dispatch(KvCacheStorage storage, const CausalAttentionOperands& p, const VqKvCacheView& cache,
              const VqKvCausalPlan& plan, const KVCacheVqStaging* staging,
              WorkspaceArena& workspace, cudaStream_t stream) {
    if (storage == KvCacheStorage::Vq2)
        attention<KVCacheVqKeyCodec::Vq2>(p, cache, plan, staging, workspace, stream);
    else if (storage == KvCacheStorage::Q4KeyVq2Value)
        attention<KVCacheVqKeyCodec::Q4>(p, cache, plan, staging, workspace, stream);
    else
        throw std::logic_error("VQ attention on another KV storage");
}

} // namespace

void vq_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& positions, const Tensor& valid, const Tensor& rows,
                            const Tensor& tree_masks, float scale, PagedKVBatchLayerView cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, cudaStream_t stream) {
    const auto plan  = make_vq_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], envelope);
    const auto* tree = static_cast<const std::uint32_t*>(tree_masks.data);
    if (tree != nullptr && plan.plan.family == Int8KvFamily::Tiled)
        throw std::invalid_argument("VQ attention: a verification tree needs the grouped route");
    const auto p = make_causal_operands(q, positions, out, scale, envelope.max_visible_keys);
    auto view    = make_vq_cache_view(cache, &valid, &rows);
    view.tree_masks = tree;
    if (plan.plan.family == Int8KvFamily::Tiled && cache.window.present()) {
        // Wide calls stage their own window rows: the prompt reads them as the exact call
        // columns, and only after attention do they replace the slots the call reads. Without a
        // window every key reads its codes, so nothing is staged.
        auto scope          = workspace.scope();
        const auto staging  = allocate_vq_staging(workspace, q.ne[2], cache.num_kv_heads);
        kv_cache_append_vq_batch_launch(k, v, positions, valid, rows, cache, &staging, stream);
        dispatch(cache.storage, p, view, plan, &staging, workspace, stream);
        kv_cache_vq_window_commit_launch(staging, positions, valid, cache, q.ne[2], stream);
        return;
    }
    kv_cache_append_vq_batch_launch(k, v, positions, valid, rows, cache, nullptr, stream);
    dispatch(cache.storage, p, view, plan, nullptr, workspace, stream);
}

void vq_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                            const PagedKVLayerView& cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, cudaStream_t stream) {
    const auto plan = make_vq_kv_causal_plan(q.ne[1], q.ne[2], 1, envelope);
    const auto view = make_vq_cache_view(single_row_paged_kv_batch_view(cache));
    dispatch(cache.storage, make_causal_operands(q, positions, out, scale, envelope.max_visible_keys),
             view, plan, nullptr, workspace, stream);
}

std::size_t vq_kv_workspace_bytes(int heads, int kv_heads, int batch, int min_width,
                                  int max_width, CausalAttentionExecutionEnvelope envelope) {
    constexpr int kGroupedMaxWidth = 256;
    std::size_t maximum            = 0;
    for (int width = min_width; width <= std::min(max_width, kGroupedMaxWidth); ++width) {
        const auto vq = make_vq_kv_causal_plan(heads, width, batch, envelope);
        if (vq.plan.family == Int8KvFamily::Tiled) continue;
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width,
                                       vq.window_splits + vq.plan.partition.capacity, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    if (batch == 1 && max_width > kGroupedMaxWidth) {
        // A wide call holds its staging while the prompt kernel allocates split partials. The
        // split plan at max_visible_keys bounds every launch (int8_fast_prompt_workspace_bytes).
        for (int width = std::max(min_width, kGroupedMaxWidth + 1); width <= max_width; ++width) {
            const FastPromptPlan plan =
                int8_fast_prompt_plan(heads, width, envelope.max_visible_keys,
                                      envelope.prompt_split_workspace_bytes);
            WorkspaceLayoutBuilder layout;
            (void)allocate_vq_staging(layout, width, kv_heads);
            if (plan.splits > 1)
                (void)allocate_fast_prompt_partials(layout, heads, width, plan.splits);
            maximum = std::max(maximum, layout.peak_bytes(1));
        }
    }
    return maximum;
}

} // namespace ninfer::ops::detail
