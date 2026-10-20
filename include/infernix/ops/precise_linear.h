#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "infernix/ops/ggml_linear.h"
#include "infernix/ops/sparse_moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace infernix::ops {

/**
 * @brief The reference route's projection (docs/maintainer/model-quality.md §6.2):
 *
 *   out[n, t] = round(sum over k = 0 .. K-1 of w[n, k] * x[k, t])
 *
 * with every weight its logical value decoded exactly from the stored representation (as the op
 * suites' oracles decode it), every product formed exactly in binary32, the products summed in
 * ascending k in double-binary32 arithmetic (within (4096 + 2 K) u^2 of the exact sum, u = 2^-24,
 * relative to the sum of the terms' magnitudes; ops/precise/precise_dot.cuh), and one
 * round-to-nearest-even conversion of that sum to the output tensor's dtype (BF16 or FP32): the
 * correctly rounded result except within that bound of a rounding boundary. An output's bits depend
 * only on its row, its column of x and K: they are identical at every call width, column position
 * and batch composition, and from run to run. It computes on CUDA cores, about eight binary32
 * operations per term: only the reference route runs it.
 *
 * `x` is contiguous BF16 [K, T]; outputs are contiguous [N_out, T] BF16 or FP32 tensors.
 */

/**
 * Dense weights [N, K]: BF16 (contiguous), Q2/Q4/Q5/Q6/Q8 row-split (code times its group's FP16
 * scale), row-scaled FP8 (code times its row's BF16 scale) and NVFP4 with 16-element E4M3 scales
 * (fl32(code * scale / divisor), the format's logical FP32 weight). out is [N, T].
 */
void precise_linear(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

/**
 * BF16 weight tensors [K, N_i] (projection_fp32's form, one row per column of the view), their rows
 * concatenated in order: out is [sum N_i, T].
 */
void precise_linear(const Tensor& x, std::span<const Tensor* const> weights, Tensor& out, cudaStream_t stream);

/**
 * GGUF block segments (ggml_linear's form, every GGUF block format and both layouts): each segment's
 * rows into rows [out_offset, out_offset + rows) of output `output`. With `input_perm`, element j of a
 * column is read at x[input_perm[j / perm_block] * perm_block + j % perm_block] (ggml_linear's input
 * permutation). The weight is kappa * (d * sub * code - dmin * min_sub): each sub-block's sums of
 * code * x and of x are scaled by kappa * d * sub and kappa * dmin * min_sub (exact in binary32), and
 * the bound above counts both terms' magnitudes.
 */
void precise_linear(const Tensor& x, std::span<const GgmlSegment> segments, std::span<Tensor* const> outs,
                    const std::int32_t* input_perm, std::int32_t perm_block, cudaStream_t stream);

/**
 * SwiGLU over a fused gate/up projection (linear_swiglu's form): rows [0, I) of `gate_up` are the
 * gate rows and [I, 2 I) the up rows; out[i, t] = round_bf16(silu(g) * u) with g and u the sums
 * above (as binary64 values) and silu evaluated in binary64. out is BF16 [I, T].
 */
void precise_swiglu(const Tensor& x, const Weight& gate_up, Tensor& out, cudaStream_t stream);

/**
 * Weight rows [row_begin, row_begin + rows) of a dense Weight (the forms above) into rows
 * [out_row, out_row + rows) of out (BF16 or FP32 [N_out, T]): a fused projection's outputs, each from
 * its own range of the parent's rows (attn_input_proj's and gdn_input_proj's row maps).
 */
void precise_linear_rows(const Tensor& x, const Weight& weight, std::int32_t row_begin, std::int32_t rows,
                         Tensor& out, std::int32_t out_row, cudaStream_t stream);

/**
 * linear_add's form: residual[n, t] = round_bf16(residual[n, t] + sum_k w[n, k] x[k, t]), the BF16
 * residual added to the sum (as binary64 values) before the one rounding. residual is BF16 [N, T].
 */
void precise_linear_add(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream);

/**
 * gdn_gating_proj's form (after its RMSNorm): with a[h] and b[h] the sums of rows h and H + h of the
 * contiguous BF16 weights `ab` stacked in order (one [2H, K] parent, or [H, K] a and b weights),
 * g[h, t] = -exp(A_log[h]) * softplus(a + dt_bias[h]) and beta[h, t] = sigmoid(b), evaluated in
 * binary64 and rounded once to FP32 [H, T].
 */
void precise_gdn_gating(const Tensor& x, std::span<const Weight* const> ab, const Tensor& a_log,
                        const Tensor& dt_bias, Tensor& g, Tensor& beta, cudaStream_t stream);

/**
 * sparse_moe's form and mathematics (the 35B-A3B geometry, include/infernix/ops/sparse_moe.h, with
 * AddResidual): FP32 router scores from the precise sums; each token's top 8 (the lower id at a tie),
 * their softmax and the shared expert's sigmoid gate in binary64; each selected expert's and the
 * shared expert's SwiGLU activation precise and rounded to BF16 (the A16 activation the exact route
 * feeds its down projections), their down projections' precise sums weighted and accumulated in FP32
 * (a token's experts in ascending id, the shared expert last), and residual = round_bf16(residual +
 * that sum). The routing is read on the host (one synchronization per call); every expert runs over
 * its own tokens, so a token's bits do not depend on the other tokens. Workspace:
 * precise_sparse_moe_workspace_bytes(T).
 */
[[nodiscard]] std::size_t precise_sparse_moe_workspace_bytes(std::int32_t tokens);
void precise_sparse_moe(const Tensor& x, const SparseMoeWeights& weights, Tensor& residual,
                        WorkspaceArena& workspace, cudaStream_t stream);

} // namespace infernix::ops
