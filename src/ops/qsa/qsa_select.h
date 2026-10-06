#pragma once

// QSA block selection (docs/maintainer/qwen3_8-flash-next-design.md §8.10, §19.3.6 item 2): the
// exact top-(budget / ratio) pooled blocks of every column, the internal stage of qsa_attention.
//
// Semantics (unchanged by the multi-CTA kernels): column t at absolute position p sees the
// n = (p + 1) / ratio complete blocks 0..n-1. When n <= top = budget / ratio it selects them all
// (count -1, "dense"). Otherwise
//
//   score_b  = (1 / sqrt(Di)) * sum_h relu(<index_q_h(t), pooled key b>)
//   selected = the top highest scores, equal scores taken in ascending block order,
//
// written in ascending block order, count = top. The FP32 arithmetic of each score is fixed:
// lane l (of 32) accumulates q[h][l + 32 j] * k[l + 32 j] over j = 0..3 in order from 0, the 32
// lane partials are summed in warp_sum's xor-butterfly order (offsets 16, 8, 4, 2, 1), the heads'
// ReLUs are added in head order to 0 and the sum is multiplied by rsqrtf(Di). Scores, ids and
// counts are therefore bitwise equal to the original one-CTA-per-column kernel's.
//
// Execution, per group of at most kQsaSelectGroupColumns columns:
//   1. qsa_score_kernel: CTAs split each row's blocks; a warp reads 8 pooled keys once and scores
//      them for every column of its row tile (up to 16), reduce-scattering the 32 (block, head)
//      lane partials of a column in the butterfly's pairing. FP32 scores go to scratch. Rows of at
//      least 32 columns (prompt chunks) take qsa_score_wide_kernel instead: register-tiled on the
//      FP32 pipes, a thread forms all 32 lane partials of its 16 dots and adds them in the
//      butterfly's tree (leaves in bit-reversed lane order), so its scores are the same bits.
//   2. the selection, an exact radix select over the 64-bit (score, lower-id-first) order key
//      (ops/common/score_id_order.cuh): a 2048-bin histogram of the scores' bits [30:20] finds the
//      bin holding the top-th score; scores above it are selected, the bin's blocks are
//      candidates, and an 8-bit radix select over the candidates' 64-bit keys gives the threshold
//      key (a fallback runs the same select over the column when the bin exceeds the candidate
//      buffer). CTAs split each column, add their histograms into a global one and the last arriving
//      CTA finishes the column (linear_topk's split-and-merge structure). An 8-CTA cluster per
//      column exchanging its state through distributed shared memory was measured and dropped
//      (design §19.3.6 item 2, step Q2: slower at every context).
// Every launch is stream-ordered (programmatic dependents when captured) and keeps no state
// between calls, so it captures into CUDA Graphs and replays with new positions.

#include "ninfer/ops/qsa.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

/// Columns whose scores share one scratch pass (prefill chunks run several groups).
inline constexpr std::int32_t kQsaSelectGroupColumns = 128;

/// Scratch for the selection of `columns` columns (monotone in columns and max_context).
[[nodiscard]] std::size_t qsa_select_scratch_bytes(const QsaGeometry& geometry, std::int32_t columns,
                                                   std::int32_t max_context);

struct QsaSelectOutput {
    std::int32_t* selected = nullptr; // [top, columns]: ascending block ids of each column
    std::int32_t* counts   = nullptr; // [columns]: top, or -1 when the column selects every block
};

/// index_q: BF16 [Di, index_heads, T] from qsa_index_query; pooled_pages: the layer's BF16 pooled
/// plane holding every block the columns see; `spaces`: the layer's pages outside the pool, whose
/// pooled keys live in spaces.pooled (design §19.3.11; none by default).
void qsa_select(const Tensor& index_q, const Tensor& pooled_pages, const QsaBatch& batch,
                const QsaGeometry& geometry, std::int32_t max_context, void* scratch,
                std::size_t scratch_bytes, const QsaSelectOutput& output, cudaStream_t stream,
                const QsaPageSpaces& spaces = {});

/// After qsa_select, the FP32 scores of its last group: column c of the group (c counted from
/// the group's first column) holds block b at scores[c * stride + b] for b < its block count.
struct QsaSelectScores {
    const float* scores = nullptr;
    std::int32_t stride = 0;
};
[[nodiscard]] QsaSelectScores qsa_select_scores(const void* scratch, const QsaGeometry& geometry,
                                                std::int32_t max_context);

} // namespace ninfer::ops::detail
