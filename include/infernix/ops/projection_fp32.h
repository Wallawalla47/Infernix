#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <span>

namespace infernix::ops {

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
 * results are bitwise identical in every batch shape and column position. The one exception: a
 * BF16 vocabulary head given as one weight of at least 32,768 rows sums in tensor-core order at 1..16
 * columns; its columns are identical across those widths but may differ in rounding from a call of
 * more than 16 columns. Oracle: the FP64 dot product, with the FP32 accumulation error bound
 * sum_k |w x| * K * 2^-24.
 */
void projection_fp32(const Tensor& x, std::span<const Tensor* const> weights, Tensor& out,
                     cudaStream_t stream);

/**
 * The same FP32 projection with one row-split `q8_g32_fp16` weight [N, K] (an 8-bit `lm_head`) or
 * `q4_g64_fp16` weight (a proposal head): out[n, t] = sum_k w_hat[n, k] * FP32(x[k, t]), with
 * w_hat[n, k] = fl32(code[n, k] * scale[n, k / G]) for the group size G (32 or 64), which is
 * exact. The summation order again depends only on K. K is a multiple of 8 and at most 3072; the
 * weight's padded K is a multiple of G.
 */
void projection_fp32(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

} // namespace infernix::ops
