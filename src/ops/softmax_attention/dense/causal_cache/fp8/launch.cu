#include "ops/softmax_attention/dense/causal_cache/fp8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/instances.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/template_launch.cuh"
#include "ops/softmax_attention/dense/causal_cache/fp8/tiled_launch.h"
#include "ops/kv_cache/append/launch.h"

namespace infernix::ops::detail {
namespace {

template <class G, int Tokens, class Input, bool Writable>
void grouped(const CausalAttentionOperands& p, Fp8KvCacheView<Writable> cache, Input input,
             CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Fp8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_fp8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked>(
            p, cache, input, partition, partial, stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, false>(
            p, cache.valid_columns, partition, partial, stream);
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

template <class G, class Input, bool Writable>
void grouped_instance(const CausalAttentionOperands& p, Fp8KvCacheView<Writable> cache, Input input,
                      CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    switch (p.width) {
#define INFERNIX_FP8_GROUPED(T)                                                                      \
    case T:                                                                                        \
        if constexpr (T <= kCausalGroupedTokens<G>) {                                \
            return grouped<G, T>(p, cache, input, partition, partial, stream);         \
        }                                                                          \
        break
        INFERNIX_FP8_GROUPED(1);
        INFERNIX_FP8_GROUPED(2);
        INFERNIX_FP8_GROUPED(3);
        INFERNIX_FP8_GROUPED(4);
        INFERNIX_FP8_GROUPED(5);
        INFERNIX_FP8_GROUPED(6);
        INFERNIX_FP8_GROUPED(7);
        INFERNIX_FP8_GROUPED(8);
#undef INFERNIX_FP8_GROUPED
    }
    throw std::logic_error("FP8 grouped plan exceeds the selected token tile");
}

template <class Input>
void execute_grouped(const Tensor& q, const Tensor& positions, float scale,
                     PagedKVBatchLayerView cache, const Tensor* valid, const Tensor* rows,
                     const std::uint32_t* tree_masks, Input input, const Fp8KvCausalPlan& plan,
                     WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    auto view =
        make_quantized_causal_cache_view<Fp8KvCacheView<Input::writes_cache>>(cache, valid, rows);
    view.tree_masks    = tree_masks;
    auto scope         = workspace.scope();
    const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                                  plan.partition.capacity, plan.batch);
    const auto p = make_causal_operands(q, positions, out, scale, plan.envelope.max_visible_keys);
    with_causal_geometry(plan.query_heads, [&]<class G>() {
        grouped_instance<G>(p, view, input, plan.partition, partial.view(), stream);
    });
}

template <class G, int Tokens>
void parallel_grouped(const CausalAttentionOperands& p, Fp8KvReadView cache,
                      CausalKvPartition partition, CausalPartialView partial, cudaStream_t stream) {
    using Instance    = Fp8KvGroupedInstance<G, Tokens>;
    const auto invoke = [&]<bool MultiBatch, bool Masked>() {
        launch_fp8_kv_grouped_mma<G, typename Instance::Schedule, MultiBatch, Masked, false,
                                  CausalCachedInput, true>(p, cache, {}, partition, partial,
                                                           stream);
        launch_causal_natural_merge<G, typename Instance::Merge, MultiBatch, Masked, false>(
            p, cache.valid_columns, partition, partial, stream);
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

void execute_parallel(const CausalAttentionOperands& p, Fp8KvReadView cache,
                      const Fp8KvCausalPlan& plan, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto partial = allocate_causal_partials(workspace, plan.query_heads, plan.width,
                                                  plan.partition.capacity, plan.batch);
    with_causal_geometry(plan.query_heads, [&]<class G>() {
        parallel_grouped<G, kCausalGroupedTokens<G>>(p, cache, plan.partition, partial.view(),
                                                      stream);
    });
}

} // namespace

void fp8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& positions, const Tensor& valid, const Tensor& rows,
                             const Tensor& tree_masks, float scale, PagedKVBatchLayerView cache,
                             CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                             Tensor& out, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan           = make_fp8_kv_causal_plan(q.ne[1], q.ne[2], q.ne[3], envelope,
                                                        execution.multiprocessor_count);
    const auto* tree = static_cast<const std::uint32_t*>(tree_masks.data);
    if (tree != nullptr && plan.family == Fp8KvFamily::Tiled)
        throw std::invalid_argument("FP8 attention: a verification tree needs the "
                                    "grouped routes");
    if (plan.family != Fp8KvFamily::Grouped) {
        kv_cache_append_batch_launch(k, v, positions, valid, rows, cache, stream);
        const auto p = make_causal_operands(q, positions, out, scale, envelope.max_visible_keys);
        auto view = make_quantized_causal_cache_view<Fp8KvCacheView<false>>(cache, &valid, &rows);
        view.tree_masks = tree;
        if (plan.family == Fp8KvFamily::Tiled)
            fp8_kv_tiled_attention(p, view, plan.partition, workspace, stream);
        else
            execute_parallel(p, view, plan, workspace, stream);
    } else {
        execute_grouped(q, positions, scale, cache, &valid, &rows, tree,
                        CausalAppendInput{static_cast<const __nv_bfloat16*>(k.data),
                                          static_cast<const __nv_bfloat16*>(v.data)},
                        plan, workspace, out, stream);
    }
}

void fp8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                             const PagedKVLayerView& cache,
                             CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                             Tensor& out, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const auto plan =
        make_fp8_kv_causal_plan(q.ne[1], q.ne[2], 1, envelope, execution.multiprocessor_count);
    const auto view = single_row_paged_kv_batch_view(cache);
    if (plan.family == Fp8KvFamily::Tiled)
        fp8_kv_tiled_attention(
            make_causal_operands(q, positions, out, scale, envelope.max_visible_keys),
            make_quantized_causal_cache_view<Fp8KvCacheView<false>>(view), plan.partition,
            workspace, stream);
    else if (plan.family == Fp8KvFamily::ParallelGrouped)
        execute_parallel(make_causal_operands(q, positions, out, scale, envelope.max_visible_keys),
                         make_quantized_causal_cache_view<Fp8KvCacheView<false>>(view), plan,
                         workspace, stream);
    else
        execute_grouped(q, positions, scale, view, nullptr, nullptr, nullptr, CausalCachedInput{},
                        plan, workspace, out, stream);
}

void fp8_kv_cached_rows_attention(const Tensor& q, const Tensor& positions, const Tensor& valid,
                                  const Tensor& rows, float scale, PagedKVBatchLayerView cache,
                                  CausalAttentionExecutionEnvelope envelope, std::int32_t call_batch,
                                  WorkspaceArena& workspace, Tensor& out, DeviceExecutionView execution) {
    auto plan = make_fp8_kv_causal_plan(q.ne[1], 1, call_batch, envelope,
                                        execution.multiprocessor_count);
    if (plan.family != Fp8KvFamily::Grouped || q.ne[2] != 1)
        throw std::logic_error("FP8 attention: one-column rows take the grouped route");
    plan.batch = q.ne[3];
    execute_grouped(q, positions, scale, cache, &valid, &rows, nullptr, CausalCachedInput{}, plan,
                    workspace, out, execution.stream);
}

std::size_t fp8_kv_cached_rows_workspace_bytes(int heads, int call_batch, int rows,
                                               CausalAttentionExecutionEnvelope envelope,
                                               int multiprocessor_count) {
    const auto plan = make_fp8_kv_causal_plan(heads, 1, call_batch, envelope, multiprocessor_count);
    WorkspaceLayoutBuilder layout;
    (void)allocate_causal_partials(layout, heads, 1, plan.partition.capacity, rows);
    return layout.peak_bytes(1);
}

} // namespace infernix::ops::detail
