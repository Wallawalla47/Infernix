#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops {

/**
 * Per-layer n-gram embedding injection (PLE) into hyper-connection residual streams
 * (docs/maintainer/qwen3_8-flash-next-design.md §12). For each column t, with S streams of H
 * features (residual R [S*H, T], stream-major):
 *
 *   e        = bf16(fp8_row * scale) for each of the column's n-gram rows       -> ple_embed
 *   key      = W_key e  [S*H],  value = W_value e  [H]                          (caller GEMMs)
 *   g_s      = <norm(key_s; w_key), norm(R_s; w_query)> / sqrt(H)
 *   g_s      = sign(g_s) * sqrt(max(|g_s|, 1e-6))
 *   gated_s  = sigmoid(g_s) * value                                             -> ple_gate
 *   nrm      = norm(gated; w_conv)                                              -> ple_gate
 *   R       += gated + SiLU(sum_j w_conv1d[j] * nrm[t - (K-1-j) * dilation])     -> ple_conv_inject
 *
 * norm() is the per-stream RMSNorm with unit-offset weight. The convolution is depthwise over the
 * S*H channels and causal; its state holds the last (K-1)*dilation columns of nrm of each
 * sequence. Every Op evaluates in FP32 from represented inputs and rounds each output once;
 * the oracle is the formula in FP64.
 */

/// rows: U8 [W, heads, T] FP8 E4M3FN codes; scale: BF16 [1]. out: BF16 [heads*W, T],
/// out[h*W + j, t] = bf16(e4m3(rows[j, h, t]) * scale).
void ple_embed(const Tensor& rows, const Tensor& scale, Tensor& out, cudaStream_t stream);

/// key: BF16 [S*H, T]; value: BF16 [H, T]; residual: BF16 [S*H, T]; key_norm, query_norm,
/// conv_norm: BF16 [S*H]. gated and normalized: BF16 [S*H, T].
void ple_gate(const Tensor& key, const Tensor& value, const Tensor& residual,
              const Tensor& key_norm, const Tensor& query_norm, const Tensor& conv_norm,
              std::int32_t streams, float eps, Tensor& gated, Tensor& normalized,
              cudaStream_t stream);

/// The T columns form `sequences` runs of T / sequences consecutive columns. Sequence i reads its
/// convolution history from state slot source_slots[i] and writes the updated history to slot
/// destination_slots[i] (both I32 [sequences]; equal slots update in place). An empty
/// destination_slots leaves every state unchanged (speculative verification, committed later by
/// ple_conv_commit). weight: BF16 [C, K] (tap j of channel c at j*C + c); states: BF16
/// [C, span, slots] with span = (K-1) * dilation, oldest column first. residual is updated in
/// place.
void ple_conv_inject(const Tensor& gated, const Tensor& normalized, const Tensor& weight,
                     std::int32_t dilation, Tensor& states, const Tensor& source_slots,
                     const Tensor& destination_slots, Tensor& residual, cudaStream_t stream);

/// Commits the first commit_columns[b] (I32 [B], each in [0, W]) of row b's verified columns to
/// slot slots[b] (I32 [B]): the history becomes the trailing span columns of the old history
/// followed by normalized[:, 0:n, b] (normalized: BF16 [C, W, B], the conv inputs ple_gate
/// produced for those columns). Zero leaves the row's history unchanged. The result equals the
/// in-place ple_conv_inject state update over those n columns.
void ple_conv_commit(const Tensor& normalized, const Tensor& commit_columns, Tensor& states,
                     const Tensor& slots, cudaStream_t stream);

} // namespace infernix::ops
