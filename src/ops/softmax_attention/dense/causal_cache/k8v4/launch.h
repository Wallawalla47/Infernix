#pragma once

#include "infernix/ops/softmax_attention.h"

namespace infernix::ops::detail {

// tree_masks is empty for causal rows or I32 [W,B] per-row ancestor masks (see
// causal_softmax_attention).
void k8v4_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid, const Tensor& rows,
                              const Tensor& tree_masks, float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution);

void k8v4_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                              const PagedKVLayerView& cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, DeviceExecutionView execution);

// Rows of one-column calls whose keys are already appended (the column-invariant verification
// round's rows variant, INFERNIX_TMP_VERIFY_ROWS_ATTENTION): q [256, heads, 1, rows], positions,
// valid and rows hold one entry per row (valid and rows may be empty). Each row's key partition is
// the one a one-column call of `call_batch` rows plans, so every row computes what that call
// computes for it.
void k8v4_kv_cached_rows_attention(const Tensor& q, const Tensor& positions, const Tensor& valid,
                                   const Tensor& rows, float scale, PagedKVBatchLayerView cache,
                                   CausalAttentionExecutionEnvelope envelope, std::int32_t call_batch,
                                   WorkspaceArena& workspace, Tensor& out, DeviceExecutionView execution);
// Its split partials for `rows` rows.
std::size_t k8v4_kv_cached_rows_workspace_bytes(int heads, int call_batch, int rows,
                                                CausalAttentionExecutionEnvelope envelope,
                                                int multiprocessor_count);

} // namespace infernix::ops::detail
