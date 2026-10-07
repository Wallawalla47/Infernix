#pragma once
#include "ops/softmax_attention/dense/causal_cache/nvfp4/operands.h"

namespace infernix::ops::detail {
void nvfp4_kv_tiled_attention(const CausalAttentionOperands&, Nvfp4KvReadView, cudaStream_t);
} // namespace infernix::ops::detail
