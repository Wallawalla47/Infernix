#pragma once

// The --lossless route's Qwen3.5 text forward (docs/maintainer/lossless-route.md §2): Hugging Face's
// decoder operation by operation, each correctly rounded, over one causal-scoring window in one pass
// (attention over the window's own keys, GDN and convolution from zero state).

#include "core/arena.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/execution/parameters.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::models::qwen3_5::execution {

// The window's rotary table (lossless_rope_table) from its positions (I32 [T] or [T, 3]), allocated
// from `workspace` for the window's layers.
Tensor lossless_rope_window(const TextConfig& config, const Tensor& positions, WorkspaceArena& workspace,
                            cudaStream_t stream);
[[nodiscard]] std::size_t lossless_rope_window_bytes(const TextConfig& config, std::int32_t tokens);

// One decoder layer over the window's residual x (BF16 [H, T]), in place.
void lossless_layer(const BlockParameters& block, const TextConfig& config, MixerKind kind, const Tensor& rope_table,
                    Tensor& x, WorkspaceArena& workspace, cudaStream_t stream);
[[nodiscard]] std::size_t lossless_layer_workspace_bytes(const BlockParameters& block, const TextConfig& config,
                                                         MixerKind kind, std::int32_t tokens);

// The final norm (Qwen3_5RMSNorm) into out (BF16 [H, T]).
void lossless_final_norm(const Tensor& x, const Tensor& weight, const TextConfig& config, Tensor& out,
                         cudaStream_t stream);

// The scoring logits: fp32(hidden W_head) (FP32 [vocab, T]); hidden BF16 [H, T].
void lossless_head(const Tensor& hidden, const Weight& head, Tensor& logits, WorkspaceArena& workspace,
                   cudaStream_t stream);
[[nodiscard]] std::size_t lossless_head_workspace_bytes(const Weight& head, std::int32_t tokens);

} // namespace infernix::models::qwen3_5::execution
