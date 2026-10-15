#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/linear.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

[[nodiscard]] std::size_t fp8_attn_input_partial_capacity_bytes(std::int32_t max_tokens);

inline constexpr int kFp8AttnInputLastSmallMmaT = 32;

[[nodiscard]] std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

// A16ColumnInvariant: one sliced-K schedule (4 K-warps) for every T, so a column's arithmetic does
// not depend on the call's other columns.
void fp8_attn_input_invariant_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream);
void fp8_attn_input_a16_small_mma_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void fp8_attn_input_a16_gemm_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, Fp8A8Workspace workspace, cudaStream_t stream);
// AllowA8Residual: the workspace's two E4M3 planes into two accumulators (MMA tiles).
void fp8_attn_input_a8_residual_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& k, Tensor& v,
                                       Fp8A8Workspace workspace, cudaStream_t stream);

void fp8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                             Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                             cudaStream_t stream);

} // namespace infernix::ops::detail
