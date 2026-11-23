#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "core/device.h"

#include <optional>

namespace infernix::models::qwen3_5::execution {

[[nodiscard]] std::size_t gdn_projection_workspace_bytes(const GdnParameters& parameters,
                                                         std::int32_t first, std::int32_t last);
[[nodiscard]] std::size_t gdn_snapshot_workspace_bytes(const GdnParameters& parameters,
                                                       const GdnConfig& config, std::int32_t batch,
                                                       std::int32_t first_width,
                                                       std::int32_t last_width);
// tree covers the verification-tree projection route as well as the chain record.
[[nodiscard]] std::size_t gdn_record_workspace_bytes(const GdnParameters& parameters,
                                                     const GdnConfig& config, std::int32_t batch,
                                                     std::int32_t first_width,
                                                     std::int32_t last_width, bool tree = false);
// `lora`: the projection's term (GdnParameters::lora_input's addend over the call's U), or none.
void gdn_projection(const Tensor& hidden, const GdnParameters& parameters, Tensor& qkv, Tensor& z,
                    WorkspaceArena& workspace, cudaStream_t stream, const ops::LoraAddend* lora = nullptr);
// The input norm into `hidden` and the control's g and beta. --lora: with GdnParameters::lora_input it also computes
// U = A hidden into `u` (BF16 [lora_input.a.rows, T], the caller's) and returns the projection's term over it (none
// when only the control is adapted); an adapted control (lora_control) takes its term on the norm alone (rmsnorm,
// the shrink, then gdn_gating_proj's LoRA form).
std::optional<ops::LoraAddend> gdn_norm_control(const Tensor& residual, const Tensor& norm, float epsilon,
                                                const GdnParameters& parameters, Tensor& hidden, Tensor& g,
                                                Tensor& beta, Tensor* u, WorkspaceArena& workspace,
                                                DeviceExecutionView execution);
void gdn_projection_snapshot(const Tensor& hidden, const GdnParameters& parameters,
                             const GdnConfig& config, Tensor& conv_states,
                             const Tensor& valid_columns, const Tensor& initial_slots,
                             const Tensor& destination_slots, Tensor& query, Tensor& key,
                             Tensor& value, Tensor& z, WorkspaceArena& workspace,
                             cudaStream_t stream, const ops::LoraAddend* lora = nullptr);
// tree_rows (I32 [words,B]) selects the verification-tree convolution (single FP8/NVFP4
// parents); nullptr is the chain record.
void gdn_projection_record(const Tensor& hidden, const GdnParameters& parameters,
                           const GdnConfig& config, const Tensor& conv_states,
                           const Tensor& valid_columns, const Tensor& initial_slots,
                           Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                           Tensor& z, WorkspaceArena& workspace, cudaStream_t stream,
                           const Tensor* tree_rows = nullptr, const ops::LoraAddend* lora = nullptr);
// --lora: out_proj's U over `input` (allocated in `workspace`) and its term; none without an adapted out_proj.
[[nodiscard]] std::optional<ops::LoraAddend> gdn_output_lora(const GdnParameters& parameters, const Tensor& input,
                                                             WorkspaceArena& workspace, cudaStream_t stream);

} // namespace infernix::models::qwen3_5::execution
