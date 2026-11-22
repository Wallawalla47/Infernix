#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/lora.h"
#include "infernix/ops/speculative_tree.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

// input_rows selects the parent: 5120 ([16384,5120]) or 3072 (Qwen3.5-122B-A10B's [20480,3072]); qtype its row
// word (fp8_e4m3fn_row_f32: a static input scale, no A8 residual profile). `lora`: the call takes a LoRA term (the
// [16384,5120] parent only), so no width runs the fused route.
[[nodiscard]] std::size_t fp8_gdn_snapshot_workspace_capacity_bytes(QType qtype, LinearPolicy policy,
                                                                    std::int32_t input_rows,
                                                                    std::int32_t batch_size,
                                                                    std::int32_t min_width,
                                                                    std::int32_t max_width, bool lora = false);

[[nodiscard]] std::size_t fp8_gdn_record_workspace_capacity_bytes(QType qtype, LinearPolicy policy,
                                                                  std::int32_t input_rows,
                                                                  std::int32_t batch_size,
                                                                  std::int32_t min_width,
                                                                  std::int32_t max_width, bool lora = false);

void fp8_gdn_snapshot_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                   Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                   Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                   cudaStream_t stream);

void fp8_gdn_record_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 const Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                                 Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream);

// With a LoRA term (`lora`, over the call's W*B columns) the fused route's widths materialize too: the projection
// with the term on the plan's materialized route, then the convolution over it.
void fp8_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                               Tensor& conv_states, const Tensor& valid_columns,
                               const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                               Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                               LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream,
                               const LoraAddend* lora = nullptr);

// tree_rows selects the verification-tree convolution over a materialized projection; nullptr is
// the chain record. `lora` as for the snapshot.
void fp8_gdn_record_dispatch(const Tensor* tree_rows, const Tensor& x, const Weight& weight,
                             const Tensor& conv_weight, const Tensor& conv_states,
                             const Tensor& valid_columns, const Tensor& initial_slot,
                             Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                             Tensor& z, LinearPolicy policy, WorkspaceArena& workspace,
                             cudaStream_t stream, const LoraAddend* lora = nullptr);

} // namespace infernix::ops::detail
