#include "ops/softmax_attention/dense/causal_cache/k8v4/tiled_launch.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/instances.h"
#include "ops/softmax_attention/common/nvfp4_tiled_values.cuh"
#include "ops/softmax_attention/common/mxfp8_tiled_launch.cuh"
#include "ops/softmax_attention/common/causal_tiled_merge.cuh"

namespace infernix::ops::detail {
void k8v4_kv_tiled_attention(const CausalAttentionOperands& p, K8V4KvReadView cache,
                             CausalKvPartition partition, bool pv8, WorkspaceArena& workspace,
                             cudaStream_t stream) {
    auto scope = workspace.scope();
    const auto partial =
        allocate_causal_partials(workspace, p.query_heads, p.width, partition.capacity, 1);
    const auto invoke = [&]<class G>() {
        launch_mxfp8_kv_tiled_mma<G, K8V4KvTiledInstance, Fp8TiledKeys, Nvfp4G16TiledValues>(
            p, cache, partition, partial.view(), pv8, stream);
        launch_causal_tiled_merge<G, true>(p, cache.valid_columns, partition, partial.view(),
                                           stream);
    };
    if (p.query_heads == 24)
        invoke.template operator()<CausalD256H24Kv4>();
    else
        invoke.template operator()<CausalD256H16Kv2>();
}
} // namespace infernix::ops::detail
