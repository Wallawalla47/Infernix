#pragma once

#include "core/tensor.h"
#include "ninfer/ops/speculative_tree.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void gdn_projected_conv_snapshot_launch(const Tensor& projected, const Tensor& conv_weight,
                                        Tensor& conv_states, const Tensor& valid_columns,
                                        const Tensor& initial_state_slots,
                                        const Tensor& snapshot_base_slots, Tensor& query,
                                        Tensor& key, Tensor& value, cudaStream_t stream);

void gdn_projected_conv_record_launch(const Tensor& conv_record, const Tensor& conv_weight,
                                      const Tensor& conv_states, const Tensor& valid_columns,
                                      const Tensor& initial_state_slots, Tensor& query, Tensor& key,
                                      Tensor& value, cudaStream_t stream);

// Verification-tree form: column c of row b convolves the projected inputs of its three nearest
// ancestors in tree_rows[b] (history past the anchor) with the chain kernel's FMA order, so a chain
// row gets the chain result over its valid prefix.
void gdn_projected_conv_record_tree_launch(const Tensor& conv_record, const Tensor& conv_weight,
                                           const Tensor& conv_states, const Tensor& valid_columns,
                                           const Tensor& initial_state_slots,
                                           const Tensor& tree_rows, Tensor& query, Tensor& key,
                                           Tensor& value, cudaStream_t stream);

} // namespace ninfer::ops::detail
