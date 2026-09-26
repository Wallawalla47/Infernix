#pragma once

#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t
attention_projection_workspace_bytes(const AttentionParameters& parameters, std::int32_t first,
                                     std::int32_t last);
void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream);

void text_rope(const Tensor& positions, const RopeConfig& config, const ops::PreparedRope& prepared,
               Tensor& query, DeviceExecutionView execution);

// Normalize q and k and rotate them with the prepared coefficients. Where the fused Op covers the
// geometry this is one graph node instead of three; everywhere else it is the three calls it
// replaces, which are the same arithmetic bit for bit. The fused Op carries the native schedule,
// so a scaled YaRN coefficient takes the three-call form. It chooses a schedule, not a result.
void text_qk_norm_rope(const Tensor& positions, const RopeConfig& rope,
                       const AttentionConfig& attention, float rms_norm_eps,
                       const Tensor& q_norm_weight, const Tensor& k_norm_weight,
                       const ops::PreparedRope& prepared,
                       const Tensor& query, const Tensor& key, Tensor& normalized_query,
                       Tensor& normalized_key, DeviceExecutionView execution);

} // namespace ninfer::models::qwen3_5::execution
