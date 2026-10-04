#pragma once

#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/types.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * QSA: block-sparse attention selected by a pooled-key indexer
 * (docs/maintainer/qwen3_8-flash-next-design.md §8.10; upstream Qwen4Exp QSA indexer).
 *
 * For a query at absolute position p of a sequence, with ratio R and budget B:
 *
 *   visible          = positions 0..p;  complete blocks c = floor((p + 1) / R)
 *   pooled key b     = RoPE(norm(bf16(mean(raw_key[R b .. R b + R - 1]))), position R b)
 *   score_b          = (1/sqrt(Di)) * sum_h relu(<index_q_h(p), pooled key b>)
 *   selected blocks  = top-(B/R) scores among the c complete blocks (all when c <= B/R; equal
 *                      scores select the lower block id)
 *   attended tokens  = tokens of the selected blocks, then the tail positions R c .. p
 *   out_h(p)         = softmax_j(scale <q_h, k_j>) v_j over the attended tokens (GQA)
 *
 * norm() is RMSNorm with unit-offset weight, RoPE the half-split rotation of the first
 * `rotary_dim` dimensions with theta. While p + 1 <= B + R - 1 every token is attended, so QSA
 * equals causal attention. Keys and values are read from the layer's paged planes in the
 * configured KV storage; pooled keys live in their own BF16 plane, [Di/R, 64, 1, pages], so the
 * pooled key of block b fills the R token slots of its block. Pooling and selection are FP32
 * evaluations of BF16 inputs; selection ties are broken by block id, so selection is
 * deterministic. The attention oracle is FP64 softmax attention over the decoded K/V of the
 * selected tokens.
 */

struct QsaGeometry {
    std::int32_t heads          = 0; // query heads
    std::int32_t kv_heads       = 0;
    std::int32_t head_dim       = 0; // 256
    std::int32_t index_heads    = 0;
    std::int32_t index_head_dim = 0; // Di
    std::int32_t rotary_dim     = 0;
    std::int32_t budget         = 0; // B
    std::int32_t ratio          = 0; // R
    float theta                 = 0;
    float eps                   = 0;
};

/// One attention layer's paged planes.
struct QsaKVLayer {
    PagedKVLayerView kv;  // K/V planes in the configured storage (block_table unused here)
    Tensor pooled_pages;  // BF16 [Di/R, 64, 1, pages]
};

/// The call's columns: `batch` sequences of `width` consecutive positions each.
struct QsaBatch {
    Tensor block_tables;  // I32 [pages per row, rows]
    Tensor table_rows;    // I32 [batch]
    Tensor positions;     // I32 [batch * width]: absolute position of each column
    Tensor tail_slots;    // I32 [batch]: raw-key tail state slot of each sequence
    std::int32_t batch = 0;
    std::int32_t width = 0;
    // False leaves the tails unchanged (speculative verification, committed later by
    // qsa_commit_tails); pooled keys of blocks completing in the call are still written.
    bool update_tails = true;
};

/// q: BF16 [Di, index_heads, T] in place: q = RoPE(norm(q; weight), positions).
void qsa_index_query(Tensor& q, const Tensor& norm_weight, const Tensor& positions,
                     const QsaGeometry& geometry, cudaStream_t stream);

/// raw_keys: BF16 [Di, T], the un-normalized index keys of the call's columns. Writes the pooled
/// key of every block that completes inside the call, reading the keys of its positions before
/// the call from the sequence's tail of raw keys (tails: BF16 [Di, R - 1, slots]), and updates the
/// tail: afterwards slot q % R holds the raw key of every position q of the sequence's latest
/// block with q % R < R - 1 up to its last position, also when that position completes the block
/// (a later call that rewrites the position alone re-pools the same key). Positions inside a
/// sequence are consecutive.
void qsa_pool_keys(const Tensor& raw_keys, const Tensor& norm_weight, Tensor& tails,
                   const QsaKVLayer& layer, const QsaBatch& batch, const QsaGeometry& geometry,
                   cudaStream_t stream);

/// Commits verified columns to the raw-key tails of L layers at once. raw_keys: BF16
/// [Di, W, B, L], the un-normalized index keys a verification call produced; positions: I32
/// [W, B], consecutive per row; commit_columns: I32 [B], each in [0, W]; tails: BF16
/// [Di, R - 1, slots, L]; tail_slots: I32 [B]. For row b and n = commit_columns[b], the tails
/// become what qsa_pool_keys would leave after a call over that row's first n columns. Pooled
/// keys are untouched: the verification call already wrote those of every block its committed
/// columns complete.
void qsa_commit_tails(const Tensor& raw_keys, const Tensor& positions, const Tensor& commit_columns,
                      Tensor& tails, const Tensor& tail_slots, const QsaGeometry& geometry,
                      cudaStream_t stream);

[[nodiscard]] std::size_t qsa_attention_workspace_bytes(const QsaGeometry& geometry,
                                                        std::int32_t columns,
                                                        std::int32_t max_context);

/// q: BF16 [head_dim, heads, T] (normalized, rotated); index_q: BF16 [Di, index_heads, T] from
/// qsa_index_query. Every column's keys and values (and the call's pooled keys) are already in
/// the planes. out: BF16 [head_dim, heads, T].
void qsa_attention(const Tensor& q, const Tensor& index_q, const QsaKVLayer& layer,
                   const QsaBatch& batch, const QsaGeometry& geometry, float scale,
                   std::int32_t max_context, void* workspace, std::size_t workspace_bytes,
                   Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
