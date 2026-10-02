#pragma once

#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/operands.h"

namespace ninfer::ops::detail {
// pv8 runs the kernel's 8-bit (E4M3) PV form.
void k8v4_kv_tiled_attention(const CausalAttentionOperands&, K8V4KvReadView, CausalKvPartition,
                             bool pv8, WorkspaceArena&, cudaStream_t);
} // namespace ninfer::ops::detail
