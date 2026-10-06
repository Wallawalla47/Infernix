#pragma once

// QSA attention for prompt calls on Tensor Cores (docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.8 F5), the internal route qsa_attention takes for exactly the calls its FP32 kernel would
// run unsplit (columns x KV heads of at least half the SMs: 43+ columns for Qwen4Exp on an RTX
// 5090's 170 SMs). Within that class a
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
// at the end. Scores are FP32-accumulated m16n8k16 products of FP16 query rows (rotated into the
// keys' Hadamard domain for every storage but BF16) and the stored keys widened exactly: INT8 and
// FP8 E4M3 codes, with their group (INT8, 64 dimensions) or row (FP8) scales applied to the FP32
// partials; NVFP4 codes times their E4M3 group scales, exact in FP16; BF16 keys on BF16 Tensor
// Cores. P x V multiplies FP16 probabilities by FP16 values: 8-bit codes times their represented
// scale, NVFP4 values decoded exactly into a per-warp FP16 tile, or the stored FP16 values, into
// FP32 accumulators; rotated values (NVFP4, K8V4) are rotated back after the merge. The FP64
// oracle of qsa_attention qualifies it (tests/ops/test_qsa.cpp).

#include "ninfer/ops/qsa.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

/// The exact window of a vector-quantized layer (vq2, k4v2) and a wide call's staged rows, as QSA's
/// kernels read them; null pointers when absent. Window planes as ops/kv_cache/kv_window.cuh:
/// [256, kKVWindowSlots, Hkv, rows] codes, [4, ...] FP16 group scales, [2, ...] tags; `slots` the
/// window row of each sequence. Staging: [256, width, Hkv] codes, [4, width, Hkv] scales.
struct QsaVqWindow {
    const std::int8_t* k_codes  = nullptr;
    const std::int8_t* v_codes  = nullptr;
    const __half* k_scales      = nullptr;
    const __half* v_scales      = nullptr;
    const std::int32_t* tags    = nullptr;
    const std::int32_t* slots   = nullptr;
    const std::int8_t* staged_k_codes = nullptr;
    const std::int8_t* staged_v_codes = nullptr;
    const __half* staged_k_scales     = nullptr;
    const __half* staged_v_scales     = nullptr;
    int staged_width                  = 0; // the staged call's width; 0 when nothing is staged
};

/// Whether the prompt kernel serves this layer and geometry (qsa_attention takes it for the calls it
/// would otherwise run unsplit).
[[nodiscard]] bool qsa_prompt_supported(const QsaKVLayer& layer, const QsaGeometry& geometry);

/// q: BF16 [head_dim, heads, T]; selected/counts: qsa_select's output for the call's columns;
/// `vq`: the vector-quantized storages' window and staging (unused otherwise).
void qsa_prompt_attention(const Tensor& q, const QsaKVLayer& layer, const QsaBatch& batch,
                          const QsaGeometry& geometry, float scale, const std::int32_t* selected,
                          const std::int32_t* counts, Tensor& out, cudaStream_t stream,
                          const QsaVqWindow& vq);

} // namespace ninfer::ops::detail
