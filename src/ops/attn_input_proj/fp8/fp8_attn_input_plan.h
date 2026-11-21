#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/lora.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

[[nodiscard]] std::size_t fp8_attn_input_partial_capacity_bytes(std::int32_t max_tokens);

inline constexpr int kFp8AttnInputLastSmallMmaT = 32;

// The [14336,5120] parent: fp8_e4m3fn_row_bf16, or fp8_e4m3fn_row_f32 (its static input scale encodes
// the A8 route's activations; a static scale has no AllowA8Residual profile).
[[nodiscard]] std::size_t fp8_attn_input_workspace_capacity_bytes(QType qtype, LinearPolicy policy,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

// A16ColumnInvariant: one sliced-K schedule (4 K-warps) for every T, so a column's arithmetic does
// not depend on the call's other columns.
void fp8_attn_input_invariant_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream,
    const LoraAddend* lora = nullptr);
void fp8_attn_input_a16_small_mma_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream,
    const LoraAddend* lora = nullptr);
void fp8_attn_input_a16_gemm_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream,
    const LoraAddend* lora = nullptr);

void fp8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, cudaStream_t stream,
    const LoraAddend* lora = nullptr);

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, Fp8A8Workspace workspace, cudaStream_t stream,
    const LoraAddend* lora = nullptr);
// AllowA8Residual: the workspace's two E4M3 planes into two accumulators (MMA tiles).
void fp8_attn_input_a8_residual_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                       Tensor& gate, Tensor& k, Tensor& v,
                                       Fp8A8Workspace workspace, cudaStream_t stream,
    const LoraAddend* lora = nullptr);

void fp8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                             Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                             cudaStream_t stream,
    const LoraAddend* lora = nullptr);

// Qwen3.5-122B-A10B: one fp8_e4m3fn_row_f32 parent [17408,3072] (query 8192, key 512, gate 8192,
// value 512). AllowA8 encodes with the parent's static input scale from T = 17; a static scale has
// no A8 residual profile.
[[nodiscard]] std::size_t fp8_attn_input_n17408_partial_capacity_bytes(std::int32_t max_tokens);
[[nodiscard]] std::size_t fp8_attn_input_n17408_workspace_capacity_bytes(LinearPolicy policy,
                                                                         std::int32_t min_tokens,
                                                                         std::int32_t max_tokens);
void fp8_attn_input_n17408_invariant_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                            Tensor& gate, Tensor& k, Tensor& v,
                                            cudaStream_t stream);
void fp8_attn_input_n17408_a16_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                      Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void fp8_attn_input_n17408_a8_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                     Tensor& gate, Tensor& k, Tensor& v, Fp8A8Workspace workspace,
                                     cudaStream_t stream);
void fp8_attn_input_n17408_dispatch(const Tensor& x, const Weight& weight, Tensor& q,
                                    Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                                    WorkspaceArena* workspace, cudaStream_t stream);

} // namespace infernix::ops::detail
