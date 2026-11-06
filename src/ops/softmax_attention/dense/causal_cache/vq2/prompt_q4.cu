// VQ2 attention: the Q4 key codec's (k4v2) prompt route (routes.cuh).
#include "ops/softmax_attention/dense/causal_cache/vq2/routes.cuh"

namespace infernix::ops::detail {

template void vq_prompt_attention<KVCacheVqKeyCodec::Q4>(const CausalAttentionOperands&, const VqKvCacheView&, const VqKvCausalPlan&, const KVCacheVqStaging*, WorkspaceArena&, cudaStream_t);

} // namespace infernix::ops::detail
