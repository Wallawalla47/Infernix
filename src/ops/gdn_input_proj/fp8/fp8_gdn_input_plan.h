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

// Two registered parents, told apart by K: [16384,5120] (q/k/v 10240, z 6144) in fp8_e4m3fn_row_bf16, or in
// static-scale fp8_e4m3fn_row_f32 (NVIDIA's NVFP4-QAD 27B), and the Qwen3.5-122B-A10B's static-scale
// fp8_e4m3fn_row_f32 [20480,3072] (q/k/v 12288, z 8192). A static scale's A8 route encodes with the parent's input
// scale and has no residual profile.
[[nodiscard]] std::size_t fp8_gdn_input_partial_capacity_bytes(std::int32_t max_tokens);
[[nodiscard]] std::size_t fp8_gdn_input_n20480_partial_capacity_bytes(std::int32_t max_tokens);
// The A8 partials of the parent with this K.
[[nodiscard]] std::size_t fp8_gdn_input_partial_capacity_bytes(std::int32_t input_rows,
                                                               std::int32_t max_tokens);

// The workspace of a call of [min_tokens, max_tokens] columns. A static input scale (fp8_e4m3fn_row_f32, either
// parent) has no residual route: an interval that reaches it is refused here, as the dispatch refuses it.
[[nodiscard]] std::size_t fp8_gdn_input_workspace_capacity_bytes(QType qtype, LinearPolicy policy,
                                                                 std::int32_t input_rows,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens);

void fp8_gdn_input_n16384_invariant_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                           Tensor& z, cudaStream_t stream);
void fp8_gdn_input_n20480_invariant_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                           Tensor& z, cudaStream_t stream);
void fp8_gdn_input_n20480_a16_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                     cudaStream_t stream);
void fp8_gdn_input_n20480_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    Fp8A8Workspace workspace, cudaStream_t stream);

void fp8_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream);

void fp8_gdn_input_matrix_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream);

void fp8_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream);
// AllowA8Residual: the workspace's two E4M3 planes into two accumulators (MMA tiles).
void fp8_gdn_input_a8_residual_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                      Tensor& z, Fp8A8Workspace workspace, cudaStream_t stream);

// Exact contraction mechanisms shared by G1/G2/G3. Semantic Ops own their route frontier and
// call one of these launchers after resolving their complete-form plan.
// A16ColumnInvariant: one sliced-K schedule (4 K-warps) for every T, so a column's arithmetic does
// not depend on the call's other columns.
void fp8_gdn_input_invariant_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream);
void fp8_gdn_input_a16_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream);

void fp8_gdn_input_a8_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               WorkspaceArena& workspace, cudaStream_t stream);
void fp8_gdn_input_a8_residual_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                        Tensor& z, WorkspaceArena& workspace,
                                        cudaStream_t stream);

void fp8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream);

} // namespace infernix::ops::detail
