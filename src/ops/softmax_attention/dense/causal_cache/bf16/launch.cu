#include "ops/softmax_attention/dense/causal_cache/bf16/launch.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/template_launch.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"
#include "ops/kv_cache/append/launch.h"

namespace infernix::ops::detail {
namespace {

template <class G, Bf16KvInstance Instance, bool Writable, class Input>
void grouped(const CausalAttentionOperands& p, Bf16KvCacheView<Writable> cache, Input input,
             const Bf16KvCausalPlan& plan, CausalPartialView partial, cudaStream_t stream) {
    using S           = typename Bf16KvInstanceTraits<Instance>::Schedule;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_bf16_kv_grouped_mma<G, S, MultiBatch, Masked>(p, cache, input, plan.partition,
                                                             partial, stream);
        launch_bf16_kv_merge<G, Bf16KvMergeSchedule<256>, MultiBatch, Masked>(
            p, cache, plan.partition, partial, stream);
    };
    if (p.batch == 1) {
        if (cache.valid_columns)
            invoke.template operator()<false, true>();
        else
            invoke.template operator()<false, false>();
    } else {
        if (cache.valid_columns)
            invoke.template operator()<true, true>();
        else
            invoke.template operator()<true, false>();
    }
}

template <class G, bool Writable, class Input>
void grouped_instance(const CausalAttentionOperands& p, Bf16KvCacheView<Writable> cache,
                      Input input, const Bf16KvCausalPlan& plan, CausalPartialView partial,
                      cudaStream_t stream) {
    switch (plan.instance) {
    case Bf16KvInstance::GroupedDecode:
        return grouped<G, Bf16KvInstance::GroupedDecode>(p, cache, input, plan, partial, stream);
    case Bf16KvInstance::Grouped32:
        return grouped<G, Bf16KvInstance::Grouped32>(p, cache, input, plan, partial, stream);
    case Bf16KvInstance::Grouped64:
        return grouped<G, Bf16KvInstance::Grouped64>(p, cache, input, plan, partial, stream);
    default:
        throw std::logic_error("BF16 grouped launch received a tiled plan");
    }
}

void launch_tiled(const CausalAttentionOperands& p, Bf16KvReadView cache, const Bf16KvCausalPlan& plan,
                  CausalPartialView partial, cudaStream_t stream, Bf16KvSegment segment = {},
                  bool last_segment = true) {
    with_causal_geometry(p.query_heads, [&]<class G>() {
        if (plan.instance == Bf16KvInstance::Tiled128)
            launch_bf16_kv_tiled_mma<G, typename Bf16KvInstanceTraits<Bf16KvInstance::Tiled128>::Schedule>(
                p, cache, stream, plan.prompt_splits, partial, segment, last_segment);
        else
            launch_bf16_kv_tiled_mma<G, typename Bf16KvInstanceTraits<Bf16KvInstance::Tiled64>::Schedule>(
                p, cache, stream, plan.prompt_splits, partial, segment, last_segment);
    });
}

void tiled(const CausalAttentionOperands& p, Bf16KvReadView cache, const Bf16KvCausalPlan& plan,
           WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    CausalPartialView partial{};
    if (plan.prompt_splits > 1) {
        partial = allocate_causal_partials(workspace, plan.query_heads, plan.width, plan.prompt_splits, 1).view();
    }
    launch_tiled(p, cache, plan, partial, stream);
}

// The packed storage's prompt route: its keys unpacked kBf16pSegmentKeys at a time into a scratch of the bf16
// storage's rows (bf16p_kv_unpack_kernel), the bf16 tiled kernel running each segment's key blocks over it, its CTAs'
// state suspended between segments: the bf16 storage's launch, bit for bit. One row (`rows`, `valid` as the Op's).
void tiled_packed(const CausalAttentionOperands& p, const PagedKVBatchLayerView& cache, const Tensor* valid,
                  const Tensor* rows, const Bf16KvCausalPlan& plan, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope              = workspace.scope();
    const auto description  = bf16_kv_instance_description(plan.instance);
    const int keys          = static_cast<int>(plan.envelope.max_visible_keys);
    const auto scratch      = allocate_bf16p_prompt_scratch(workspace, plan.query_heads, cache.num_kv_heads, plan.width,
                                                            keys, plan.prompt_splits, description.query_rows);
    CausalPartialView partial{};
    if (plan.prompt_splits > 1) {
        partial = allocate_causal_partials(workspace, plan.query_heads, plan.width, plan.prompt_splits, 1).view();
    }
    const auto* valid_columns = valid != nullptr ? static_cast<const std::int32_t*>(valid->data) : nullptr;
    const auto* table_rows    = rows != nullptr ? static_cast<const std::int32_t*>(rows->data) : nullptr;
    if (valid_columns != nullptr) { CUDA_CHECK(cudaMemsetAsync(scratch.zero, 0, sizeof(std::int32_t), stream)); }
    const Bf16KvReadView view{reinterpret_cast<const __nv_bfloat16*>(scratch.keys),
                              reinterpret_cast<const __half*>(scratch.values),
                              scratch.table,
                              valid_columns,
                              valid_columns != nullptr ? scratch.zero : nullptr,
                              scratch.table_pages,
                              cache.head_dim,
                              cache.num_kv_heads};
    for (int segment = 0; segment < scratch.segments; ++segment) {
        const int first = segment * kBf16pSegmentKeys, count = std::min(kBf16pSegmentKeys, keys - first);
        const int warps = count * cache.num_kv_heads * 2;
        const auto unpack = [&]<int KVHeads>() {
            bf16p_kv_unpack_kernel<KVHeads><<<div_up(warps, 8), 256, 0, stream>>>(
                static_cast<const std::uint8_t*>(cache.k_pages.data), static_cast<const std::uint8_t*>(cache.v_pages.data),
                static_cast<const std::uint8_t*>(cache.k_scale_pages.data),
                static_cast<const std::uint8_t*>(cache.v_scale_pages.data),
                static_cast<const std::int32_t*>(cache.block_tables.data), table_rows, cache.block_tables.ne[0], p.positions,
                valid_columns, p.width, first, count, scratch.keys, scratch.values, scratch.table);
        };
        if (cache.num_kv_heads == 4) {
            unpack.template operator()<4>();
        } else {
            unpack.template operator()<2>();
        }
        CUDA_CHECK(cudaGetLastError());
        const Bf16KvSegment range = scratch.segments > 1
                                        ? Bf16KvSegment{first / description.key_rows,
                                                        (first + kBf16pSegmentKeys) / description.key_rows,
                                                        scratch.state}
                                        : Bf16KvSegment{};
        launch_tiled(p, view, plan, partial, stream, range, segment + 1 == scratch.segments);
    }
}

template <class Input>
void execute_grouped(const Tensor& q, Input input, const Tensor& positions, float scale,
                     PagedKVBatchLayerView cache, const Tensor* valid, const Tensor* rows,
                     const std::uint32_t* tree_masks, const Bf16KvCausalPlan& plan,
                     WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    auto scope   = workspace.scope();
    auto storage = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                            plan.partition.capacity, plan.batch);
    const auto p = make_causal_operands(q, positions, out, scale, plan.envelope.max_visible_keys);
    auto view          = bf16_kv_cache_view<Input::writes_cache>(cache, valid, rows);
    view.tree_masks    = tree_masks;
    const auto partial = storage.view();
    with_causal_geometry(p.query_heads, [&]<class G>() {
        grouped_instance<G>(p, view, input, plan, partial, stream);
    });
}
} // namespace

void bf16_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& table_rows, const Tensor& tree_masks, float scale,
                              PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan           = make_bf16_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], envelope,
                                                         execution.multiprocessor_count);
    const auto* tree = static_cast<const std::uint32_t*>(tree_masks.data);
    if (tree != nullptr && !plan.grouped())
        throw std::invalid_argument("BF16 attention: a verification tree needs the grouped route");
    if (cache.storage == KvCacheStorage::Bf16Packed) {
        // The packed storage encodes each row as it appends it: the rows go in first, then attention reads them all
        // from the cache (the bf16 tiles, so the bits of the bf16 storage's fused route).
        kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
        if (!plan.grouped()) {
            tiled_packed(make_causal_operands(q, positions, out, scale, envelope.max_visible_keys), cache,
                         valid_columns.data != nullptr ? &valid_columns : nullptr, &table_rows, plan, workspace,
                         stream);
        } else {
            execute_grouped(q, CausalCachedInput{}, positions, scale, cache, &valid_columns, &table_rows, tree, plan,
                            workspace, out, stream);
        }
        return;
    }
    if (!plan.grouped()) {
        kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
        tiled(make_causal_operands(q, positions, out, scale, envelope.max_visible_keys),
              bf16_kv_cache_view<false>(cache, &valid_columns, &table_rows), plan, workspace, stream);
    } else {
        execute_grouped(q,
                        CausalAppendInput{static_cast<const __nv_bfloat16*>(k.data),
                                          static_cast<const __nv_bfloat16*>(v.data)},
                        positions, scale, cache, &valid_columns, &table_rows, tree, plan, workspace,
                        out, stream);
    }
}

void bf16_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                              const PagedKVLayerView& cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan =
        make_bf16_kv_causal_plan(q.ne[1], q.ne[2], 1, envelope, execution.multiprocessor_count);
    const auto view = single_row_paged_kv_batch_view(cache);
    if (!plan.grouped() && cache.storage == KvCacheStorage::Bf16Packed) {
        tiled_packed(make_causal_operands(q, positions, out, scale, envelope.max_visible_keys), view, nullptr,
                     nullptr, plan, workspace, stream);
    } else if (!plan.grouped()) {
        tiled(make_causal_operands(q, positions, out, scale, envelope.max_visible_keys),
              bf16_kv_cache_view<false>(view), plan, workspace, stream);
    } else {
        execute_grouped(q, CausalCachedInput{}, positions, scale, view, nullptr, nullptr, nullptr,
                        plan, workspace, out, stream);
    }
}
void bf16_kv_cached_rows_attention(const Tensor& q, const Tensor& positions, const Tensor& valid,
                                   const Tensor& rows, float scale, PagedKVBatchLayerView cache,
                                   CausalAttentionExecutionEnvelope envelope, std::int32_t call_batch,
                                   WorkspaceArena& workspace, Tensor& out, DeviceExecutionView execution) {
    auto plan = make_bf16_kv_causal_plan(q.ne[1], 1, call_batch, envelope,
                                         execution.multiprocessor_count);
    if (!plan.grouped() || q.ne[2] != 1)
        throw std::logic_error("BF16 attention: one-column rows take the grouped route");
    // The partition stays the call batch's (its share groups are the columns' lanes); the launch holds q.ne[3] rows.
    plan.batch = q.ne[3];
    execute_grouped(q, CausalCachedInput{}, positions, scale, cache, &valid, &rows, nullptr, plan,
                    workspace, out, execution.stream);
}

std::size_t bf16_kv_cached_rows_workspace_bytes(int heads, int call_batch, int rows,
                                                CausalAttentionExecutionEnvelope envelope,
                                                int multiprocessor_count) {
    const auto plan = make_bf16_kv_causal_plan(heads, 1, call_batch, envelope, multiprocessor_count);
    WorkspaceLayoutBuilder layout;
    (void)allocate_causal_partials(layout, heads, 1, plan.partition.capacity, rows);
    return layout.peak_bytes(1);
}

} // namespace infernix::ops::detail
