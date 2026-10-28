// VQ2 attention: the vq2 key codec's prompt route (routes.cuh).
#include "ops/softmax_attention/dense/causal_cache/vq2/routes.cuh"

namespace infernix::ops::detail {

template void vq_prompt_attention<KVCacheVqKeyCodec::Vq2>(const CausalAttentionOperands&, const VqKvCacheView&, const VqKvCausalPlan&, const KVCacheVqStaging*, WorkspaceArena&, cudaStream_t);

} // namespace infernix::ops::detail
