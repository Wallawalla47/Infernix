#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "infernix/ops/lora.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

// `lora`: the LoRA term over the logical rows [q 2048, k 2048, value 6144, z 6144] (the single parent's order:
// the Q4 parent's 4096 rows, then the Q5 parent's 12288; include/infernix/ops/lora.h) and the call's columns, added
// to each projected value before its BF16 store; null keeps every kernel as it was.
void q4_q5_gdn_input_independent_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                        Tensor& z, cudaStream_t stream, const LoraAddend* lora = nullptr);
void q4_q5_gdn_input_column_invariant_launch(const Tensor& x, const Weight& qk_weight,
                                             const Weight& value_z_weight, Tensor& qk,
                                             Tensor& value, Tensor& z, cudaStream_t stream,
                                             const LoraAddend* lora = nullptr);

void q4_q5_gdn_input_grouped_mma_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qkv, Tensor& z,
                                        Q4Q5GdnInputScheduleId schedule, cudaStream_t stream,
                                        const LoraAddend* lora = nullptr);

void q4_q5_gdn_input_conv_snapshot_launch(const Tensor& x, const Weight& qk_weight,
                                          const Weight& value_z_weight, const Tensor& conv_weight,
                                          Tensor& conv_states, const Tensor& valid_columns,
                                          const Tensor& initial_slot,
                                          const Tensor& snapshot_base_slot, Tensor& query,
                                          Tensor& key, Tensor& value, Tensor& z,
                                          cudaStream_t stream);

void q4_q5_gdn_input_conv_record_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, const Tensor& conv_weight,
                                        const Tensor& conv_states, const Tensor& valid_columns,
                                        const Tensor& initial_slot, Tensor& conv_record,
                                        Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                        cudaStream_t stream);

} // namespace infernix::ops::detail
