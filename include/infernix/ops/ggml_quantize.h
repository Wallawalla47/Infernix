#pragma once

#include "core/tensor.h"
#include "infernix/ops/ggml_linear.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops {

/**
 * Op: GGUF Q6_K quantization (--a8-activations' load-time repack of a GGUF import's BF16 projections; LOSSY)
 *
 * Math / indexing:
 *   w is BF16 [k, n] (row r's k elements contiguous, k a positive multiple of 256). Each 256-element block of a row
 *   becomes one Q6_K block (ggml's block_q6_K: 128 bytes of low nibbles, 64 of high bit pairs, 16 int8 sub-scales,
 *   a binary16 scale d) by ggml's reference quantizer, quantize_row_q6_K_ref: per 16-element sub-block the scale of
 *   make_qx_quants(16, 32, x, L, rmse_type 1) (codes in [-32, 31], 19 candidate scales, weights x^2), d =
 *   fp16(max_scale / -128) from the sub-scale of largest magnitude (the first such), sub-scales min(127,
 *   rne(-128 / max_scale * scale)), then codes clamp(rne(x / (d * sub)), -32, 31) wherever d * sub is not zero; a
 *   block whose sub-scales are all below 1e-15 in magnitude is all zero. Every product, sum and quotient is one
 *   binary32 operation in ggml's order (no contraction); rne rounds half to even.
 *
 * Layout:
 *   out holds n * k / 256 blocks of 210 bytes: Rows (ggml_rows_v1) puts block b of row r at (r k / 256 + b) 210;
 *   Tiles16 (ggml_tiles16_v1, n a multiple of 16) at (16 (r / 16) k / 256 + 16 b + r % 16) 210.
 *
 * Effects:
 *   Writes out's n * k / 256 * 210 bytes; w and out must not overlap.
 *
 * Execution:
 *   One launch; deterministic.
 */
void ggml_quantize_q6_k(const Tensor& w, GgmlLayout layout, std::uint8_t* out, cudaStream_t stream);

} // namespace infernix::ops
