#pragma once

#include "infernix/ops/softmax_attention.h"

#include <cstddef>

namespace infernix::ops::detail {

// Vector-quantized caches (Vq2, Q4KeyVq2Value). The append always runs as its own launch before
// attention; tree_masks is empty for causal rows or I32 [W,B] per-row ancestor masks.
void vq_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& positions, const Tensor& valid, const Tensor& rows,
                            const Tensor& tree_masks, float scale, PagedKVBatchLayerView cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, cudaStream_t stream);

void vq_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                            const PagedKVLayerView& cache,
                            CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                            Tensor& out, cudaStream_t stream);

std::size_t vq_kv_workspace_bytes(int heads, int kv_heads, int batch, int min_width, int max_width,
                                  CausalAttentionExecutionEnvelope envelope);

} // namespace infernix::ops::detail
