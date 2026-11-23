#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/lora.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

[[nodiscard]] std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                   std::int32_t min_tokens,
                                                                   std::int32_t max_tokens);

// `lora` below: the LoRA term (include/infernix/ops/lora.h) over the parent's rows [q 2048, k 2048, value 6144,
// z 6144] and the call's columns, added to each projected value before its BF16 store; null keeps every kernel as it
// was.

// A16ColumnInvariant: one sliced-K schedule family (8 K-warps) for every T, so a column's
// arithmetic does not depend on the call's other columns.
void nvfp4_gdn_input_invariant_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                      Tensor& z, cudaStream_t stream, const LoraAddend* lora = nullptr);
void nvfp4_gdn_input_a16_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream, const LoraAddend* lora = nullptr);
// A8Only: x encoded by ggml_a8.h's codec, then the 8-bit tiles (nvfp4_gdn_input_a8.cu).
void nvfp4_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               WorkspaceArena& workspace, cudaStream_t stream, const LoraAddend* lora = nullptr);

void nvfp4_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream, const LoraAddend* lora = nullptr);

void nvfp4_gdn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream, const LoraAddend* lora = nullptr);

void nvfp4_gdn_input_a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               Nvfp4A4Workspace workspace, cudaStream_t stream, const LoraAddend* lora = nullptr);

void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream,
                              const LoraAddend* lora = nullptr);

} // namespace infernix::ops::detail
