#pragma once

#include "infernix/ops/softmax_attention.h"

namespace infernix::ops::detail {

// tree_masks is empty for causal rows or I32 [W,B] per-row ancestor masks (see
// causal_softmax_attention).
void fp8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& positions, const Tensor& valid, const Tensor& rows,
                             const Tensor& tree_masks, float scale, PagedKVBatchLayerView cache,
                             CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                             Tensor& out, DeviceExecutionView execution);

void fp8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                             const PagedKVLayerView& cache,
                             CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                             Tensor& out, DeviceExecutionView execution);

} // namespace infernix::ops::detail
