#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <span>

namespace ninfer::ops {

/**
 * @brief FP32 projection of BF16 activations by BF16 weight rows, for outputs whose consumers
 * need more than BF16 resolution: MoE router logits (BF16 logits turn experts within one BF16
 * step of the top-k boundary into index-biased ties) and LM-head logits (BF16 logits quantize
 * token probabilities by up to 2^-8 of the logit, ~6 % at logit 16).
 *
 * out[n, t] = sum_k FP32(w[n, k]) * FP32(x[k, t]) for the rows n of `weights` concatenated in
 * order. `x` is contiguous BF16 [K, T]; each weight is contiguous BF16 [K, N_i] (one row per
 * column of the view); `out` is contiguous FP32 [sum N_i, T]. K is a multiple of 8 and at most
 * 3072; at most four weights; every operand is 16-byte aligned.
 *
 * Each output is one FP32 dot product whose summation order depends only on K, so a column's
 * results are bitwise identical in every batch shape and column position. Oracle: the FP64 dot
 * product, with the FP32 accumulation error bound sum_k |w x| * K * 2^-24.
 */
void projection_fp32(const Tensor& x, std::span<const Tensor* const> weights, Tensor& out,
                     cudaStream_t stream);

} // namespace ninfer::ops
