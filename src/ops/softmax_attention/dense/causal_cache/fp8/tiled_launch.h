#pragma once

#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/operands.h"

namespace infernix::ops::detail {
void fp8_kv_tiled_attention(const CausalAttentionOperands&, Fp8KvReadView, CausalKvPartition,
                            WorkspaceArena&, cudaStream_t);
} // namespace infernix::ops::detail
