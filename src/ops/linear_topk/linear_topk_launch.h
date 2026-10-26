#pragma once

#include "core/arena.h"
#include "core/weight.h"
#include "core/tensor.h"
#include "ops/linear_topk/linear_topk_workspace.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

void linear_topk_q8_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                           const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_q8_m64_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                               const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_fp8_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                            const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_fp8_m64_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                                const LinearTopKWorkspace& workspace, cudaStream_t stream);
void linear_topk_q4_launch(const Tensor& hidden, const Weight& head,
                           const Tensor& row_to_global_ids, const LinearTopKWorkspace& workspace,
                           cudaStream_t stream);
void linear_topk_q4_m64_launch(const Tensor& hidden, const Weight& head,
                               const Tensor& row_to_global_ids,
                               const LinearTopKWorkspace& workspace, cudaStream_t stream);
// The NVFP4 head: its FP32 logits [head rows, columns] in `logits`, the projection's own workspace
// from `arena`.
void linear_topk_nvfp4_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                              const LinearTopKWorkspace& workspace, Tensor& logits, WorkspaceArena& arena,
                              cudaStream_t stream);
// Each 128-row producer group's top sixteen per column of materialized FP32 logits [head rows, columns]
// (the 122B heads: NVFP4 through rg16_linear, per-row FP8 through linear).
void linear_topk_logits_launch(const Tensor& logits, std::int32_t valid_rows, const LinearTopKWorkspace& workspace,
                               cudaStream_t stream);
void linear_topk_merge_launch(const LinearTopKWorkspace& workspace, Tensor& candidate_ids,
                              Tensor& candidate_scores, cudaStream_t stream);

} // namespace infernix::ops::detail
