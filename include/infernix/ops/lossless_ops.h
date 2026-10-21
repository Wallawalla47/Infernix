#pragma once

#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

// The --lossless route's elementwise and row operations (docs/maintainer/lossless-route.md §2): each the
// Hugging Face Qwen3.5 operation with every step correctly rounded to its dtype. bf16(v) and fp32(v) are
// the correctly rounded value of the exact v; transcendental functions are correctly rounded binary32
// (src/ops/lossless/cr_math.cuh). Tensors are contiguous; out may alias an input of the same shape.

/**
 * Qwen3_5RMSNorm over rows of x (BF16 [width, rows]) with weight w (BF16 or FP32 [width]):
 * out = bf16(fp32(y fp32(1 + w))), y = fp32(x r), r = fp32(rsqrt(fp32(m + eps))), m = fp32(mean(x^2))
 * (the sum of squares exact).
 */
void lossless_rmsnorm(const Tensor& x, const Tensor& weight, float eps, Tensor& out, cudaStream_t stream);

/**
 * The grouped RMSNorm (Qwen4ExpTextRMSNorm with group_size): x BF16 [groups * width, T] as rows of width,
 * row r normalized with weight[(r % groups) width, + width) (weight BF16 or FP32 [groups * width]); every
 * row as lossless_rmsnorm's.
 */
void lossless_rmsnorm_grouped(const Tensor& x, const Tensor& weight, int groups, float eps, Tensor& out,
                              cudaStream_t stream);

/**
 * Qwen3_5RMSNormGated over rows of o (BF16 [width, rows]) gated by z (BF16, same shape), weight w (BF16
 * or FP32 [width]): n = bf16(fp32(o r)) with r as above, y = bf16(w n) (fp32(w n) for an FP32 weight,
 * which promotes the product), out = bf16(fp32(y fp32(silu(z)))) (sigmoid_gate: fp32(sigmoid(z)), the
 * output_gate_type of Qwen3.8-Flash-Next).
 */
void lossless_rmsnorm_gated(const Tensor& o, const Tensor& z, const Tensor& weight, float eps, Tensor& out,
                            cudaStream_t stream, bool sigmoid_gate = false);

/** out = bf16(a + b) (BF16, same shape). */
void lossless_add(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream);

/** out = bf16(bf16(silu(gate)) up) (BF16, same shape): the MLP's act_fn(gate) * up. */
void lossless_silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);

/** out = bf16(x bf16(sigmoid(gate))) (BF16, same shape): the attention output gate. */
void lossless_sigmoid_mul(const Tensor& x, const Tensor& gate, Tensor& out, cudaStream_t stream);

/**
 * The Gated DeltaNet gates from a and b (BF16 [H, T]) and the per-head A_log, dt_bias (FP32 [H]):
 * beta = bf16(sigmoid(b)) (BF16 [H, T]) and g = -fp32(fp32(exp(A_log)) fp32(softplus(fp32(a + dt_bias))))
 * (FP32 [H, T]); with a_direct, a_log holds A = -exp(A_log) itself (a GGUF import's ssm_a) and
 * g = fp32(A fp32(softplus(fp32(a + dt_bias)))).
 */
void lossless_gdn_gates(const Tensor& a, const Tensor& b, const Tensor& a_log, const Tensor& dt_bias, Tensor& g,
                        Tensor& beta, cudaStream_t stream, bool a_direct = false);

/** out[j block + i, t] = x[perm[j] block + i, t] (an exact copy; perm device I32 [K / block]); x and out
 * BF16 [K, T]. */
void lossless_permute_blocks(const Tensor& x, const std::int32_t* perm, int block, Tensor& out, cudaStream_t stream);

/**
 * The Gated DeltaNet's causal depthwise convolution over one window (zeros before its first column):
 * out[c, t] = bf16(silu(fp32(sum_i w[i, c] x[c, t - K + 1 + i]))) for x BF16 [C, T], w BF16 [K, C]
 * (tap i at i C + c, the last tap on the current column), the K-term sum exact.
 */
void lossless_causal_conv(const Tensor& x, const Tensor& weight, Tensor& out, cudaStream_t stream);
/// The same convolution over `sequences` independent sequences, each a run of T / sequences columns whose
/// K - 1 columns before its first come from slot source_slots[b] of `states` (BF16 [C, K - 1, slots], oldest
/// first, the causal_conv1d_silu state layout); afterwards slot destination_slots[b] holds the run's last
/// K - 1 columns (of the source state followed by the run). Slots I32 [>= sequences], device; a destination
/// may equal its own source but no other sequence's source or destination.
void lossless_causal_conv_slots(const Tensor& x, const Tensor& weight, int sequences, Tensor& states,
                                const Tensor& source_slots, const Tensor& destination_slots, Tensor& out,
                                cudaStream_t stream);

/**
 * The gated delta rule over one sequence (FLA's kernel, HF's recurrence operation by operation; key and
 * value head dims 128): per value head, with its key head h_v / (Hv / Hk),
 * q̂ = fp32(fp32(q r_q) fp32(128^-1/2)), k̂ = fp32(k r_k), r = fp32(rsqrt(fp32(fp32(sum x^2) + 1e-6)));
 * S = fp32(S fp32(exp(g))), u_j = fp32(sum_i fp32(S_ij k̂_i)), δ_j = fp32(fp32(v_j - u_j) beta),
 * S_ij = fp32(S_ij + fp32(k̂_i δ_j)), out_j = bf16(fp32(sum_i fp32(S_ij q̂_i))), every sum exact.
 * query and key BF16 [128 Hk, T], value BF16 [128 Hv, T] (each may be a row range of a larger tensor: its
 * columns strided), out BF16 [128 Hv, T], g FP32 and beta BF16 [Hv, T] contiguous;
 * state FP32 [128 (k), 128 (v), Hv] (the LinearAttentionStatePool layout: S_ij at (h 128 + j) 128 + i,
 * 16-byte aligned) is the initial state and receives the final one (nullptr: zeros, not kept).
 */
void lossless_gated_delta_rule(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& g,
                               const Tensor& beta, float* state, Tensor& out, cudaStream_t stream);
/// The same rule over `sequences` independent sequences, each a run of T / sequences columns: sequence b
/// starts from slot source_slots[b] of `states` (FP32 [128, 128, Hv, slots]) and leaves its final state in
/// slot destination_slots[b] (I32 [>= sequences], device); a destination may equal its own source but no
/// other sequence's source or destination.
void lossless_gated_delta_rule_slots(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& g,
                                     const Tensor& beta, int sequences, Tensor& states, const Tensor& source_slots,
                                     const Tensor& destination_slots, Tensor& out, cudaStream_t stream);

/**
 * Causal softmax attention, one operation (flash attention's): for query column t at position P_t and head
 * h (key/value head h / (Hq / Hkv)), s_j = (q . k_j) 2^-4 exact for j <= P_t (head dim 256),
 * m = max_j s_j, p_j = fp32(exp(fp32(s_j - m))), out = bf16(sum_j p_j v_j / sum_j p_j) with both sums
 * exact. q and out BF16 [256, Hq, T].
 *
 * Over one window: k and v BF16 [256, Hkv, T], P_t = t; the workspace is
 * lossless_attention_workspace_bytes(Hq, Hkv, T). With `select` (QSA, docs/maintainer/lossless-route.md
 * §2a), query t's keys are those j <= t in its tail (j >= floor((t + 1) / ratio) ratio) or in a block
 * b = j / ratio whose bit b of select[t * select_words + b / 32] is set (U32 [T][select_words]).
 */
[[nodiscard]] std::size_t lossless_attention_workspace_bytes(int query_heads, int kv_heads, int columns);
void lossless_attention(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out, void* workspace,
                        cudaStream_t stream, const std::uint32_t* select = nullptr, int select_words = 0,
                        int ratio = 0);
/// Over the paged cache: the columns are `sequences` equal runs; column t's position is positions[t]
/// (I32 [T], increasing within a run) and its keys and values are rows 0..P_t of the cache (bf16 storage
/// with BF16 K and V pages, the lossless profile) through block table row table_rows[b] (I32 [B]) of its
/// run b. max_keys bounds every P_t + 1 and is at most key_capacity, which sizes the workspace:
/// lossless_paged_attention_workspace_bytes(Hq, Hkv, the widest run, B, key_capacity).
[[nodiscard]] std::size_t lossless_paged_attention_workspace_bytes(int query_heads, int kv_heads, int max_width,
                                                                   int sequences, int key_capacity);
/// With `select` (QSA), as the window form's with each query's own position: its tail is
/// j >= floor((P_t + 1) / ratio) ratio and its blocks are the bits of its row of select.
void lossless_paged_attention(const Tensor& q, const Tensor& positions, int sequences, const Tensor& table_rows,
                              const PagedKVBatchLayerView& cache, int max_keys, int key_capacity, Tensor& out,
                              void* workspace, cudaStream_t stream, const std::uint32_t* select = nullptr,
                              int select_words = 0, int ratio = 0);
/// Stores the call's keys and values (contiguous BF16 [256, Hkv, T], `sequences` equal runs) bit for bit
/// in the lossless profile's pages: column t at position positions[t] through table row table_rows[b] of
/// its run b.
void lossless_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions, int sequences,
                        const Tensor& table_rows, const PagedKVBatchLayerView& cache, cudaStream_t stream);

/**
 * RoPE (Hugging Face's rotary module and apply_rotary_pos_emb, rotary dims R of each head, pairs P = R/2):
 * lossless_rope_frequencies writes inv_freq_i = fp32(1 / fp32(theta^(2i / R))) (FP32 [P]);
 * lossless_rope_table writes, per column t and pair i, cos and sin of fp32(position inv_freq_i), each
 * correctly rounded to binary32 and then to BF16 (BF16 [P, T, 2]: the cosines, then the sines), from
 * positions I32 [T], or [T, 3] with pair i on axis i % 3 (interleaved MRoPE); lossless_rope_apply rotates
 * x (BF16 [D >= R, H, T]) in place: x_i = bf16(bf16(x_i c) + bf16(-x_(i+P) s)),
 * x_(i+P) = bf16(bf16(x_(i+P) c) + bf16(x_i s)), dims [R, D) unchanged. |positions inv_freq| < 2^20.
 */
void lossless_rope_frequencies(float theta, int rotary_dim, Tensor& inv_freq, cudaStream_t stream);
void lossless_rope_table(const Tensor& positions, const Tensor& inv_freq, Tensor& table, cudaStream_t stream);
void lossless_rope_apply(const Tensor& table, Tensor& x, cudaStream_t stream);

/**
 * The sparse MoE router (Qwen3_5MoeTopKRouter): from logits BF16 [>= E, T] (rows [0, E) the experts'),
 * p = fp32(softmax(l)) per element, the top_k largest p (ties: the lower expert first) in descending
 * order, weights = fp32(p / fp32(sum of the top_k p)) (FP32 [top_k, T]; bf16_weights: their BF16
 * rounding, Qwen4ExpTextTopKRouter's cast to the logits' dtype) and their ids (I32 [top_k, T]).
 * E <= 1024, top_k <= 16.
 */
void lossless_router(const Tensor& logits, int experts, int top_k, Tensor& weights, Tensor& ids, cudaStream_t stream,
                     bool bf16_weights = false);

/**
 * The routed experts' columns: C = lossless_moe_columns(T, top_k, E) gathered columns, each expert's
 * (slot, column) pairs in its own run of whole 64-column tiles. lossless_moe_route fills tile_experts
 * (I32 [C / 64]: the tile's expert, -1 unused), slot_column (I32 [top_k, T]: where each pair's column
 * went) and column_token (I32 [C]: the source column, -1 padding), with counts (I32 [E]) as scratch;
 * lossless_moe_gather copies x (BF16 [H, T]) into gathered (BF16 [H, C]), padding zero, for
 * lossless_linear_grouped. Placement within a run is arbitrary; no column's result depends on it.
 */
[[nodiscard]] std::int32_t lossless_moe_columns(std::int32_t columns, std::int32_t top_k, std::int32_t experts);
void lossless_moe_route(const Tensor& ids, int experts, Tensor& counts, Tensor& tile_experts, Tensor& slot_column,
                        Tensor& column_token, cudaStream_t stream);
void lossless_moe_gather(const Tensor& x, const Tensor& column_token, const Tensor& tile_experts, Tensor& gathered,
                         cudaStream_t stream);

/**
 * The block's output into the residual (Qwen3_5MoeExperts, the shared expert, the decoder's sum): per
 * column, over its experts in ascending id, s = bf16(s + bf16(fp32(y_e w_e))) from zero (y_e the expert's
 * output column of expert_out BF16 [H, C] at slot_column, or at column t top_k + slot when slot_column
 * has no data: the offloaded MoE's entries); with a shared expert (shared_out BF16 [H, T] and its gate
 * logit, row shared_row of shared_logits BF16 [R, T]) s = bf16(s + bf16(shared bf16(sigmoid(gate))));
 * then residual = bf16(residual + s) (BF16 [H, T]), or residual = s without add.
 */
void lossless_moe_combine(const Tensor& expert_out, const Tensor& weights, const Tensor& ids, const Tensor& slot_column,
                          const Tensor* shared_out, const Tensor* shared_logits, int shared_row, Tensor& residual,
                          cudaStream_t stream, bool add = true);

// ----------------------------------------------------------------------- Qwen3.8-Flash-Next (§2a)

/**
 * Hyper-connection gates (Qwen4ExpTextGatedResidual) from the down projection dj (BF16 [rank (+ S), T]):
 * mix = bf16(silu(bf16(d / S))) (BF16 [rank, T]) and, with inject (dj holding the injection rows),
 * inject = 2 bf16(sigmoid(bf16(j / S))) (BF16 [S, T]).
 */
void lossless_hc_gates(const Tensor& dj, int rank, int streams, Tensor& mix, Tensor* inject, cudaStream_t stream);

/** The mixed input: x_i = bf16(sum_c bf16(bf16(sigmoid(u_ci)) n_ci) / S) (the sum exact); up and normed
 * BF16 [S*H, T], x BF16 [H, T]. */
void lossless_hc_mix(const Tensor& up, const Tensor& normed, int streams, Tensor& x, cudaStream_t stream);

/** The block output's injection: R_ci = bf16(R_ci + bf16(y_i j_c)); y BF16 [H, T], inject BF16 [S, T],
 * residual BF16 [S*H, T]. */
void lossless_hc_inject(const Tensor& y, const Tensor& inject, Tensor& residual, cudaStream_t stream);

/**
 * The PLE gate (Qwen4ExpTextPLELayer) from the grouped-normed keys and queries (BF16 [S*H, T]) and the
 * value (BF16 [H, T]): g_c = bf16(sum_i bf16(k_ci q_ci)) (the sum exact), g_c = bf16(g_c / fp32(sqrt(H))),
 * s_c = bf16(sqrt(bf16(max(|g_c|, fp32(1e-6))))) sign(g_c) (scores BF16 [S, T]), gated_ci =
 * bf16(bf16(sigmoid(s_c)) v_i) (BF16 [S*H, T]).
 */
void lossless_ple_gate(const Tensor& key_normed, const Tensor& query_normed, const Tensor& value, int streams,
                       Tensor& scores, Tensor& gated, cudaStream_t stream);

/**
 * The PLE dilated depthwise convolution and injection over one window (zeros before it):
 * conv_ct = bf16(sum_j w[j C + c] n[c, t - (K - 1 - j) dilation]) (the sum exact; weight BF16 [C, K]),
 * out = bf16(gated + bf16(silu(conv))), residual = bf16(residual + out) (all BF16 [C, T]).
 */
void lossless_ple_conv_inject(const Tensor& gated, const Tensor& normalized, const Tensor& weight, int dilation,
                              Tensor& residual, cudaStream_t stream);

/// The same over `sequences` runs (equal widths) continuing their sequences, with dilated history
/// (BF16 [C, (K - 1) dilation, slots], oldest first) from slot source_slots[b], left in slot
/// destination_slots[b] (I32 [>= sequences]; a destination may equal its own source only).
void lossless_ple_conv_inject_slots(const Tensor& gated, const Tensor& normalized, const Tensor& weight, int dilation,
                                    int sequences, Tensor& history, const Tensor& source_slots,
                                    const Tensor& destination_slots, Tensor& residual, cudaStream_t stream);

/** The QSA indexer's pooled keys: pooled[:, b] = bf16(fp32(sum of keys[:, r b + i], i < r) / r)) (the sum
 * exact, then HF's binary32 mean and its BF16 cast); keys BF16 [D, T], pooled BF16 [D, floor(T / r)]. */
void lossless_qsa_pool(const Tensor& keys, int ratio, Tensor& pooled, cudaStream_t stream);

/**
 * The QSA indexer's cache form. The call's raw index keys (BF16 [D, T]) form `sequences` runs of consecutive
 * positions (I32 [T]); a sequence's raw keys of its open block before the run sit in its tail slot
 * tail_slots[b] (BF16 [D, r - 1, slots], position p at slot p mod r).
 * - lossless_qsa_pool_cache: for a column at p with p + 1 a multiple of r, pooled[:, t] = the mean (as
 *   lossless_qsa_pool) of the keys at p - r + 1..p (from the tail before the run); other columns zeros.
 *   block_rope (I32 [T, 3] axis-major) gets the RoPE position of the completed block's first token: the
 *   column's when it is in the run, else the run's block_start_rope (I32 [B, 3] axis-major).
 * - lossless_qsa_pooled_store: each completing column's (normed, rotated) key into the pooled plane (BF16
 *   [D / r, 64, 1, pages]: block b's D values from token slot (r b) mod 64 of page table[r b / 64] of its row).
 * - lossless_qsa_pooled_gather: blocks [0, n) of row table_rows[sequence] into out (BF16 [D, n]), zeros past
 *   the sequence's complete blocks (those its run's last position completes).
 * - lossless_qsa_tails_store: the run's keys from its open block (positions from r floor((p_last + 1) / r))
 *   into its tail slot (after pooling, which reads the old ones).
 */
void lossless_qsa_pool_cache(const Tensor& keys, const Tensor& positions, int sequences, const Tensor& tails,
                             const Tensor& tail_slots, const Tensor& rope_positions, const Tensor& block_start_rope,
                             int ratio, Tensor& pooled, Tensor& block_rope, cudaStream_t stream);
void lossless_qsa_pooled_store(const Tensor& pooled, const Tensor& positions, int sequences, const Tensor& block_tables,
                               const Tensor& table_rows, int ratio, Tensor& plane, cudaStream_t stream);
void lossless_qsa_pooled_gather(const Tensor& plane, const Tensor& block_tables, const Tensor& table_rows,
                                const Tensor& positions, int sequences, int sequence, int ratio, Tensor& out,
                                cudaStream_t stream);
void lossless_qsa_tails_store(const Tensor& keys, const Tensor& positions, int sequences, Tensor& tails,
                              const Tensor& tail_slots, int ratio, cudaStream_t stream);

/**
 * The QSA block selection: scores FP32 [B, heads * T] with B = floor(T / ratio) (column h + heads t:
 * fp32(q_h,t . key_b), lossless_linear of the pooled keys), per query t with n = floor((t + 1) / ratio)
 * complete blocks: every block below n when n <= budget, else the budget largest of
 * fp32(fp32(sum_h relu(score)) / fp32(sqrt(index_dim))) (the sum exact; NaN largest; ties: the lower
 * block), as bits of select (U32 [T][select_words], select_words <= 2048). The workspace is
 * lossless_qsa_select_workspace_bytes(T, B). With positions (I32 [T], the cache form) query t has
 * n = floor((positions[t] + 1) / ratio) complete blocks, and the scores' B rows cover every query's n.
 */
[[nodiscard]] std::size_t lossless_qsa_select_workspace_bytes(int columns, int blocks);
void lossless_qsa_select(const Tensor& scores, int heads, int ratio, int budget, int index_dim, std::uint32_t* select,
                         int select_words, void* workspace, cudaStream_t stream, const Tensor* positions = nullptr);

} // namespace infernix::ops
