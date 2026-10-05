#pragma once

// QSA attention for prompt calls on Tensor Cores (docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.8 F5), the internal route qsa_attention takes for exactly the calls its FP32 kernel would
// run unsplit (columns x KV heads above 170: 86+ columns for Qwen4Exp). Within that class a
// column's FP32 result never depended on the call's width, and the prefix cache relies on it: a
// resumed prompt may compute a position in a call of another width than the run that captured the
// state. This kernel's result is width-independent too, so the class keeps that property; narrower
// calls (decode, verification, short suffixes) keep the FP32 kernel and its split classes.
//
// One CTA per (column, KV head) computes the column's query heads of that KV head (12 for Qwen4Exp,
// padded to the 16 MMA rows) over the column's attended tokens: its selected blocks and its
// incomplete tail block (every token while it is dense), the same list the decode kernel reads.
// Each warp sweeps its own 16-token tiles, gathered by cp.async from the paged planes and
// double-buffered, with its own online softmax; the CTA merges its warps' (max, sum, output) rows
// at the end. Scores are FP32-accumulated m16n8k16 products of FP16 query rows (INT8 storage: the
// Hadamard-rotated rows) and the stored keys widened exactly (INT8 codes, with their group scales
// applied to each 64-dimension group's FP32 partial, or BF16 keys on BF16 Tensor Cores); P x V
// multiplies FP16 probabilities by FP16 values (INT8 codes times their represented group scale,
// or the stored FP16 values) into FP32 accumulators. The FP64 oracle of qsa_attention qualifies
// it (tests/ops/test_qsa.cpp).

#include "ninfer/ops/qsa.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

/// Whether the prompt kernel serves this layer and geometry (qsa_attention takes it for the calls it
/// would otherwise run unsplit).
[[nodiscard]] bool qsa_prompt_supported(const QsaKVLayer& layer, const QsaGeometry& geometry);

/// q: BF16 [head_dim, heads, T]; selected/counts: qsa_select's output for the call's columns.
void qsa_prompt_attention(const Tensor& q, const QsaKVLayer& layer, const QsaBatch& batch,
                          const QsaGeometry& geometry, float scale, const std::int32_t* selected,
                          const std::int32_t* counts, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
