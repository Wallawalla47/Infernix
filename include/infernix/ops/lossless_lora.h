#pragma once

// infernix::ops - the --lossless route's LoRA adapter (docs/maintainer/lossless-route.md §2, LoRA adapter): PEFT's
// FP32 adapter path, each operation correctly rounded.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infernix::ops {

/**
 * An FP32 matrix split exactly into three BF16 addends a = a0 + a1 + a2 (each the next 8 significant bits; the
 * --lossless route's factors at load). Returns false when some value cannot be split exactly: a nonzero value below
 * 2^-110 in magnitude, whose low part falls under BF16's smallest subnormal, or a non-finite one. `out` receives
 * the three planes [rows][3 cols]: row r's a0 values, then its a1, then its a2.
 */
[[nodiscard]] bool lossless_lora_split3(const float* values, std::int64_t rows, std::int64_t cols,
                                        std::vector<std::uint16_t>& out);

/**
 * u = fp32(A x): for each of A's rows and x's columns the exact sum of the products, rounded once to binary32.
 * x is BF16 [K, T] (K a multiple of 64), `a3` A's three BF16 planes (lossless_lora_split3: [rows, 3 K], on the
 * device), u FP32 [rows, T]. Exactly lossless_linear's result over x stacked three times.
 */
[[nodiscard]] std::size_t lossless_lora_shrink_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                               std::int32_t columns);
void lossless_lora_shrink(const Tensor& x, const void* a3, std::int32_t rows, Tensor& u, WorkspaceArena& workspace,
                          cudaStream_t stream);

/**
 * The adapted output: for n in [0, rows) and every column t,
 *   d = fp32(sum_j B[n, j] u[u_row + j, t])       (the exact sum, rounded once),
 *   y[y_row + n, t] = bf16(fp32(y + fp32(d fp32(scale)))).
 * B is FP32 [rows, rank] on the device (each value exactly split by lossless_lora_split3), u FP32 [*, T], y BF16
 * [*, T] (updated in place). The sum is lossless_linear's over B's and u's exact BF16 parts (nine products a term);
 * a column of u holding a nonzero value below 2^-110 times its largest makes that column's outputs NaN (it cannot
 * be split exactly: never seen; a visible failure instead of a rounded one).
 */
[[nodiscard]] std::size_t lossless_lora_add_workspace_bytes(std::int32_t rows, std::int32_t rank,
                                                            std::int32_t columns);
void lossless_lora_add(Tensor& y, std::int32_t y_row, const float* b, std::int32_t rows, std::int32_t rank,
                       const Tensor& u, std::int32_t u_row, float scale, WorkspaceArena& workspace,
                       cudaStream_t stream);

} // namespace infernix::ops
