#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

/**
 * The canonical W4A16 dense projection of an `nvfp4_mul` matrix in the `nvfp4_rg16_v1` layout
 * (docs/maintainer/qwen3_5-122b-offload-design.md §6.3, §8.6 "Head"; Qwen3.8-Flash-Next design
 * §6.1.2) at its registered [N, K]: Qwen3.5-122B-A10B's output head [248320, 3072], and
 * Qwen3.8-Flash-Next FP4-Local's vision MLP [4304, 1152] (fc1) and [1152, 4304] (fc2) and merger
 * [4608, 4608] (fc1) and [2560, 4608] (fc2).
 *
 *   emax_t = a16_column_exponent(x[:, t]),  X = the A16 encoding of x[:, t] under emax_t
 *   S[n, t] = sum over 16-element blocks b of (doubled E2M1 codes of row n, block b) . X_b
 *             * e4m3_scaled(scale of row n, block b)      (exact int64)
 *   out[n, t] = fl32(fl32(S) * 2^(emax_t - 157)) * m       (canon::a16_row_output_f32; BF16 out
 *                                                          rounds that once, canon::a16_row_output)
 *
 * with m the matrix's FP32 multiplier (Weight::weight_multiplier). The sum is an exact integer, so
 * every column's outputs are the same bits whatever T, the route (a dp4a GEMV through eight columns,
 * four at K 4304 and 4608; integer tensor-core limb MMAs beyond, in passes of at most 4,096
 * columns) and the other columns: the projection is column-invariant on every policy. A column
 * holding Inf or NaN gives NaN outputs.
 *
 * x: contiguous, 16-byte aligned BF16 [K, T], T >= 1. weight: QType::NVFP4_MUL, QuantLayout::Rg16
 * at a registered [N, K], its units 16-byte aligned. out: contiguous FP32 or BF16 [N, T]. The
 * workspace holds the encoded activations of one pass (rg16_linear_workspace_bytes; an unregistered
 * [N, K] throws).
 */
[[nodiscard]] std::size_t rg16_linear_workspace_bytes(std::int32_t rows, std::int32_t input_rows,
                                                      std::int32_t columns);

void rg16_linear(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
                 cudaStream_t stream);

} // namespace infernix::ops
