#include "ops/softmax_attention/dense/causal_cache/vq2/launch.h"

#include "ops/softmax_attention/dense/causal_cache/vq2/routes.cuh"

#include <algorithm>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

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

VqKvCausalPlan make_vq_kv_causal_plan(int heads, int width, int batch,
                                      CausalAttentionExecutionEnvelope envelope) {
    // VQ2 plans with the INT8 partition but runs none of its fast-path schedules.
    envelope.decode_split64 = false;
    VqKvCausalPlan out{
        make_int8_kv_causal_plan(heads, width, batch, envelope, fast_prompt_multiprocessors()), 0};
    out.prompt_split_workspace_bytes = envelope.prompt_split_workspace_bytes;
    out.prompt_pv8                   = envelope.fast_prompt_pv8;
    if (out.plan.family == Int8KvFamily::Tiled) return out;
    const bool parallel = out.plan.family == Int8KvFamily::ParallelGrouped;
    const int narrow    = causal_grouped_tokens(heads);
    out.parallel_tile   = envelope.max_visible_keys <= kVqWideTileMaxKeys ? 2 * narrow : narrow;
    const bool wide     = parallel && out.parallel_tile == 2 * narrow;
    const int tiles       = parallel ? div_up(width, out.parallel_tile) : 1;
    const int independent =
        (envelope.row_invariant_partition ? 1 : batch) * causal_kv_heads(heads) * tiles;
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

template <KVCacheVqKeyCodec Codec>
void attention(const CausalAttentionOperands& p, const VqKvCacheView& cache,
               const VqKvCausalPlan& vq, const KVCacheVqStaging* staging,
               WorkspaceArena& workspace, cudaStream_t stream) {
    const Int8KvCausalPlan& plan = vq.plan;
    if (plan.family == Int8KvFamily::Tiled) {
        vq_prompt_attention<Codec>(p, cache, vq, staging, workspace, stream);
        return;
    }
    auto scope         = workspace.scope();
    const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                                  vq.window_splits + plan.partition.capacity,
                                                  plan.batch);
    vq_grouped_attention<Codec>(p, cache, vq, partial.view(), stream);
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

} // namespace infernix::ops::detail
