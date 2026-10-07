#include "ops/softmax_attention/dense/causal_cache/nvfp4/fast_prompt_launch.h"
#include "ops/softmax_attention/common/causal_tiled_merge.cuh"
#include "ops/softmax_attention/common/mxfp8_tiled_launch.cuh"
#include "ops/softmax_attention/common/nvfp4_tiled_values.cuh"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/schedule.cuh"

namespace infernix::ops::detail {
void nvfp4_kv_fast_prompt_attention(const CausalAttentionOperands& p, Nvfp4KvReadView cache,
                                    CausalAttentionExecutionEnvelope envelope,
                                    WorkspaceArena& workspace, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    auto scope                = workspace.scope();
    const auto partition =
        mxfp8_tiled_partition(p.query_heads, p.width, static_cast<int>(p.visible_capacity),
                              execution.multiprocessor_count, envelope.prompt_split_workspace_bytes);
    const auto partial =
        allocate_causal_partials(workspace, p.query_heads, p.width, partition.capacity, 1);
    const auto invoke = [&]<class G>() {
        launch_mxfp8_kv_tiled_mma<G, Nvfp4KvArenaTiledSchedule, Nvfp4TiledKeys,
                                  Nvfp4G16TiledValues>(p, cache, partition, partial.view(),
                                                       envelope.fast_prompt_pv8, stream);
        launch_causal_tiled_merge<G, true>(p, cache.valid_columns, partition, partial.view(),
                                           stream);
    };
    if (p.query_heads == 24)
        invoke.template operator()<CausalD256H24Kv4>();
    else
        invoke.template operator()<CausalD256H16Kv2>();
}
} // namespace infernix::ops::detail
