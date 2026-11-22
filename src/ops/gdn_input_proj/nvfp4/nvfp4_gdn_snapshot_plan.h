#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/lora.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

enum class Nvfp4GdnConvScheduleId {
    DecodeFusedA16,
    SmallTFusedA16,
    Materialized,
};

struct Nvfp4GdnConvPlan {
    Nvfp4GdnConvScheduleId schedule;
};

Nvfp4GdnConvPlan nvfp4_gdn_conv_resolve_plan(LinearPolicy policy, std::int32_t tokens,
                                             std::int32_t batch_size);

// B=1. `lora`: the call takes a LoRA term, so every width runs the Materialized schedule (nvfp4_gdn_snapshot_dispatch).
[[nodiscard]] std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy policy,
                                                                      std::int32_t min_tokens,
                                                                      std::int32_t max_tokens,
                                                                      bool lora = false);

void nvfp4_gdn_snapshot_decode_launch(const Tensor& x, const Weight& weight,
                                      const Tensor& conv_weight, Tensor& conv_states,
                                      const Tensor& valid_columns, const Tensor& initial_slot,
                                      const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                      Tensor& value, Tensor& z, cudaStream_t stream);

void nvfp4_gdn_snapshot_small_t_launch(const Tensor& x, const Weight& weight,
                                       const Tensor& conv_weight, Tensor& conv_states,
                                       const Tensor& valid_columns, const Tensor& initial_slot,
                                       const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                       Tensor& value, Tensor& z, cudaStream_t stream);

void nvfp4_gdn_record_small_t_launch(const Tensor& x, const Weight& weight,
                                     const Tensor& conv_weight, const Tensor& conv_states,
                                     const Tensor& valid_columns, const Tensor& initial_slot,
                                     Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                     Tensor& z, cudaStream_t stream);

void nvfp4_gdn_snapshot_post_launch(const Tensor& projected, const Tensor& conv_weight,
                                    Tensor& conv_states, const Tensor& valid_columns,
                                    const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                    Tensor& query, Tensor& key, Tensor& value, cudaStream_t stream);

void nvfp4_gdn_record_post_launch(const Tensor& conv_record, const Tensor& conv_weight,
                                  const Tensor& conv_states, const Tensor& valid_columns,
                                  const Tensor& initial_slot, Tensor& query, Tensor& key,
                                  Tensor& value, cudaStream_t stream);

// B=1. With a LoRA term (`lora`, over the call's columns) the Materialized schedule runs at every width: the
// projection with the term (nvfp4_gdn_input_dispatch under `policy`), then the post kernel.
void nvfp4_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                 Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                 LinearPolicy policy, WorkspaceArena& workspace,
                                 cudaStream_t stream, const LoraAddend* lora = nullptr);

} // namespace infernix::ops::detail
