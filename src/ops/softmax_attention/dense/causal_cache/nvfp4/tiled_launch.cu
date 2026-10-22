// Non-RDC compilation is required for the producer/consumer register redistribution.
#include "ops/softmax_attention/dense/causal_cache/nvfp4/tiled_launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/instances.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/tiled_launch.cuh"

namespace infernix::ops::detail {
void nvfp4_kv_tiled_attention(const CausalAttentionOperands& p, Nvfp4KvReadView cache,
                              cudaStream_t stream) {
    with_causal_geometry(p.query_heads, [&]<class G>() {
        launch_nvfp4_kv_tiled_mma<G, Nvfp4KvTiledInstance>(p, cache, stream);
    });
}
} // namespace infernix::ops::detail
