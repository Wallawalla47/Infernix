#include "ops/softmax_attention/dense/causal_cache/nvfp4/fast_tiled_launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/fast_tiled_launch.cuh"

namespace ninfer::ops::detail {

void nvfp4_kv_fast_tiled_attention(const CausalAttentionOperands& p, Nvfp4KvReadView cache,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    if (p.query_heads == 24)
        launch_nvfp4_kv_fast_tiled_mma<CausalD256H24Kv4>(p, cache, workspace, stream);
    else
        launch_nvfp4_kv_fast_tiled_mma<CausalD256H16Kv2>(p, cache, workspace, stream);
}

} // namespace ninfer::ops::detail
