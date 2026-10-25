// The --lossless route of Qwen3.8-Flash-Next (TextConfig::lossless; docs/maintainer/lossless-route.md §2a,
// §3a): every operation of Transformers' modeling_qwen4_exp.py correctly rounded: the exact projections of
// lossless_linear over each weight's exact value, the hyper-connection mixes and injections, PLE, the gated
// delta rule, QSA attention (the indexer's pooled-key scores selecting each query's blocks), the router with
// BF16 weights, the offloaded experts' lossless pass, and the logits (each exact sum correctly rounded to FP32,
// or to BF16 with BF16 logits). Two forms:
//  - a scored window (ForwardBatch::lossless_keys 0): one sequence in one call from zero state over its own
//    keys, writing nothing to the KV cache or the recurrent states;
//  - the cache form (generation): prefill chunks and decode rounds continue their sequences, K/V and pooled
//    keys through the paged planes (the lossless bf16 profile), the indexer's raw-key tails, the PLE history
//    and the GDN state through the lanes' slots, so a token's logits do not depend on how its sequence was
//    split into calls.

#include "models/qwen4_exp/execution/forward.h"

#include "infernix/ops/embedding.h"
#include "infernix/ops/hyper_connection.h"
#include "infernix/ops/lossless_linear.h"
#include "infernix/ops/lossless_ops.h"
#include "infernix/ops/lossless_quantize.h"
#include "infernix/ops/rows.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace infernix::models::qwen4_exp::execution {
namespace {

std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }

// The cache form's indexer scores (FP32 [blocks, heads * columns]) are formed this many query columns at a
// time, bounding them at long contexts.
constexpr std::int32_t kScoreColumns = 64;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("Qwen4Exp lossless: ") + message); }
}

// A BF16 tensor [K, N] of N rows (one per column: the router, the shared gate's row, the BF16 head, the
// pooled index keys) as a Weight [N, K].
Weight bf16_rows(const Tensor& t) {
    Weight w{};
    w.qtype    = QType::BF16;
    w.layout   = QuantLayout::Contiguous;
    w.ndim     = 2;
    w.n        = t.ne[1];
    w.k        = t.ne[0];
    w.shape[0] = w.padded_shape[0] = w.n;
    w.shape[1] = w.padded_shape[1] = w.k;
    w.qdata    = t.data;
    w.payload  = t.data;
    return w;
}

// A GGUF segment as a Weight [rows, K].
Weight segment_weight(const ops::GgmlSegment& s, std::int32_t k) {
    Weight w{};
    w.qtype    = s.format;
    w.layout   = s.layout == ops::GgmlLayout::Tiles16 ? QuantLayout::GgmlTiles16 : QuantLayout::GgmlRows;
    w.ndim     = 2;
    w.n        = s.rows;
    w.k        = k;
    w.shape[0] = w.padded_shape[0] = s.rows;
    w.shape[1] = w.padded_shape[1] = k;
    w.qdata    = s.data;
    w.payload  = s.data;
    return w;
}

} // namespace

// Rows [row_begin, row_begin + rows) of a projection into rows [out_row, ...) of out (BF16 or FP32).
void Forward::lossless_project(const Tensor& x, const LinearParameters& p, std::int32_t row_begin, std::int32_t rows,
                               Tensor& out, std::int32_t out_row) {
    const cudaStream_t s = device_.stream;
    auto scope           = work_.scope();
    Tensor input         = x;
    if (p.input_perm != nullptr) {
        input = work_.alloc(DType::BF16, {x.ne[0], x.ne[1]});
        ops::lossless_permute_blocks(x, p.input_perm, p.perm_block, input, s);
    }
    if (!p.is_ggml()) {
        ops::lossless_linear(input, p.weight, row_begin, rows, out, out_row, work_, s);
        return;
    }
    for (const ops::GgmlSegment& segment : p.ggml) {
        if (segment.output != 0) { throw std::invalid_argument("Qwen4Exp lossless: a one-output projection's segments"); }
        const std::int32_t first = std::max(row_begin, segment.out_offset);
        const std::int32_t last  = std::min(row_begin + rows, segment.out_offset + segment.rows);
        if (first >= last) { continue; }
        const Weight w = segment_weight(segment, input.ne[0]);
        ops::lossless_linear(input, w, first - segment.out_offset, last - first, out, out_row + (first - row_begin), work_, s);
    }
}

void Forward::lossless_project_shared(const Tensor& x, std::span<const LosslessPart> parts) {
    const cudaStream_t s = device_.stream;
    std::vector<bool> done(parts.size(), false);
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (done[i]) { continue; }
        const std::int32_t* perm = parts[i].p != nullptr ? parts[i].p->input_perm : nullptr;
        const std::int32_t block = parts[i].p != nullptr ? parts[i].p->perm_block : 0;
        auto scope   = work_.scope();
        Tensor input = x;
        if (perm != nullptr) {
            input = work_.alloc(DType::BF16, {x.ne[0], x.ne[1]});
            ops::lossless_permute_blocks(x, perm, block, input, s);
        }
        std::size_t segments = 0;
        for (const LosslessPart& part : parts) { segments += part.p != nullptr ? part.p->ggml.size() : 0; }
        std::vector<Weight> weights;
        weights.reserve(segments); // stable addresses for the projections below
        std::vector<ops::LosslessProjection> list;
        for (std::size_t j = i; j < parts.size(); ++j) {
            const LosslessPart& part = parts[j];
            const std::int32_t* part_perm = part.p != nullptr ? part.p->input_perm : nullptr;
            if (done[j] || part_perm != perm || (perm != nullptr && part.p->perm_block != block)) { continue; }
            done[j] = true;
            if (part.p == nullptr || !part.p->is_ggml()) {
                const Weight* w = part.p != nullptr ? &part.p->weight : part.weight;
                list.push_back({w, part.row_begin, part.rows, part.out, part.out_row});
                continue;
            }
            for (const ops::GgmlSegment& segment : part.p->ggml) {
                if (segment.output != 0) { throw std::invalid_argument("Qwen4Exp lossless: a one-output projection's segments"); }
                const std::int32_t first = std::max(part.row_begin, segment.out_offset);
                const std::int32_t last  = std::min(part.row_begin + part.rows, segment.out_offset + segment.rows);
                if (first >= last) { continue; }
                weights.push_back(segment_weight(segment, input.ne[0]));
                list.push_back({&weights.back(), first - segment.out_offset, last - first, part.out,
                                part.out_row + (first - part.row_begin)});
            }
        }
        ops::lossless_linear_shared(input, list, work_, s);
    }
}

Tensor Forward::lossless_mix(const HyperConnectionParameters& p, const Tensor& residual, Tensor* inject) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = residual.ne[1], H = dim(config_.hidden_size);
    const std::int32_t S = dim(config_.hc.streams), rank = dim(config_.hc.rank), W = S * H;
    Tensor x       = work_.alloc(DType::BF16, {H, T});
    auto scope     = work_.scope();
    Tensor normed  = work_.alloc(DType::BF16, {W, T});
    ops::lossless_rmsnorm_grouped(residual, p.norm, S, config_.rms_norm_eps, normed, s);
    const bool combine = p.combine && inject != nullptr;
    Tensor dj = work_.alloc(DType::BF16, {rank + (combine ? S : 0), T});
    lossless_project(normed, p.down, 0, dj.ne[0], dj, 0);
    Tensor mix = work_.alloc(DType::BF16, {rank, T});
    ops::lossless_hc_gates(dj, rank, S, mix, combine ? inject : nullptr, s);
    Tensor up = work_.alloc(DType::BF16, {W, T});
    lossless_project(mix, p.up, 0, W, up, 0);
    ops::lossless_hc_mix(up, normed, S, x, s);
    return x;
}

void Forward::lossless_ple(const PleParameters& p, Tensor& residual, const ForwardBatch& batch) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = residual.ne[1], W = residual.ne[0], H = dim(config_.hidden_size);
    const std::int32_t S = dim(config_.hc.streams);
    auto scope = work_.scope();
    const Tensor e = ple_embedding(p, batch); // the n-gram rows decoded exactly, rounded to BF16
    Tensor key = work_.alloc(DType::BF16, {W, T}), value = work_.alloc(DType::BF16, {H, T});
    const LosslessPart ple_parts[] = {{&p.projection, nullptr, 0, W, &key, 0}, {&p.projection, nullptr, W, H, &value, 0}};
    lossless_project_shared(e, ple_parts);
    Tensor key_normed = work_.alloc(DType::BF16, {W, T}), query_normed = work_.alloc(DType::BF16, {W, T});
    ops::lossless_rmsnorm_grouped(key, p.key_norm, S, config_.rms_norm_eps, key_normed, s);
    ops::lossless_rmsnorm_grouped(residual, p.query_norm, S, config_.rms_norm_eps, query_normed, s);
    Tensor scores = work_.alloc(DType::BF16, {S, T}), gated = work_.alloc(DType::BF16, {W, T});
    ops::lossless_ple_gate(key_normed, query_normed, value, S, scores, gated, s);
    Tensor normalized = key_normed; // reused
    ops::lossless_rmsnorm_grouped(gated, p.conv_norm, S, config_.rms_norm_eps, normalized, s);
    if (batch.lossless_keys != 0) {
        // The dilated history of each lane, continued in place.
        ops::lossless_ple_conv_inject_slots(gated, normalized, p.convolution, dim(config_.ple.conv_dilation),
                                            batch.batch, state_.ple_conv, batch.slots, batch.slots, residual, s);
    } else {
        ops::lossless_ple_conv_inject(gated, normalized, p.convolution, dim(config_.ple.conv_dilation), residual, s);
    }
}

Tensor Forward::lossless_gdn(const GdnParameters& p, const Tensor& x, std::uint32_t index, const ForwardBatch& batch) {
    const cudaStream_t s = device_.stream;
    const auto& g        = config_.gdn;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size);
    const std::int32_t C = dim(g.conv_channels()), KW = dim(g.key_width()), VW = dim(g.value_width());
    const std::int32_t NV = dim(g.value_heads), DV = dim(g.value_head_dim);
    Tensor y   = work_.alloc(DType::BF16, {H, T});
    auto scope = work_.scope();
    Tensor qkv = work_.alloc(DType::BF16, {C, T}), z = work_.alloc(DType::BF16, {VW, T});
    Tensor a = work_.alloc(DType::BF16, {NV, T}), b = work_.alloc(DType::BF16, {NV, T});
    const LosslessPart gdn_parts[] = {{&p.projection, nullptr, 0, C, &qkv, 0}, {&p.projection, nullptr, C, VW, &z, 0},
                                      {&p.control, nullptr, 0, NV, &a, 0}, {&p.control, nullptr, NV, NV, &b, 0}};
    lossless_project_shared(x, gdn_parts);
    Tensor gate = work_.alloc(DType::FP32, {NV, T}), beta = work_.alloc(DType::BF16, {NV, T});
    ops::lossless_gdn_gates(a, b, p.a_log, p.dt_bias, gate, beta, s, g.a_direct);
    Tensor conv = work_.alloc(DType::BF16, {C, T});
    const Tensor q = conv.slice(0, 0, KW), k = conv.slice(0, KW, KW), v = conv.slice(0, 2 * KW, VW);
    Tensor o = work_.alloc(DType::BF16, {VW, T});
    if (batch.lossless_keys != 0) {
        // Each lane's convolution window and recurrent state, continued in place.
        LinearAttentionStateLayerView states = state_.gdn->layer_view(index);
        ops::lossless_causal_conv_slots(qkv, p.convolution, batch.batch, states.conv, batch.slots, batch.slots, conv, s);
        ops::lossless_gated_delta_rule_slots(q, k, v, gate, beta, batch.batch, states.recurrent, batch.slots,
                                             batch.slots, o, s);
    } else {
        ops::lossless_causal_conv(qkv, p.convolution, conv, s);
        ops::lossless_gated_delta_rule(q, k, v, gate, beta, nullptr, o, s);
    }
    Tensor normed = work_.alloc(DType::BF16, {VW, T});
    Tensor rows = o.view({DV, NV * T}), gates = z.view({DV, NV * T}), out_rows = normed.view({DV, NV * T});
    ops::lossless_rmsnorm_gated(rows, gates, p.norm, config_.rms_norm_eps, out_rows, s,
                                g.output_gate == GateActivation::Sigmoid);
    lossless_project(normed, p.output, 0, H, y, 0);
    return y;
}

Tensor Forward::lossless_attention_block(const AttentionParameters& p, const Tensor& x, std::uint32_t index,
                                         const ForwardBatch& batch) {
    const cudaStream_t s = device_.stream;
    const auto& a        = config_.attention;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size);
    const std::int32_t QW = dim(a.query_width()), KW = dim(a.key_width()), D = dim(a.head_dim);
    const std::int32_t heads = dim(a.heads), kv_heads = dim(a.kv_heads);
    const std::int32_t IH = dim(config_.qsa.index_heads), ID = dim(config_.qsa.index_head_dim);
    const std::int32_t ratio = dim(config_.qsa.compress_ratio), budget = dim(config_.qsa.block_budget());
    Tensor y   = work_.alloc(DType::BF16, {H, T});
    auto scope = work_.scope();
    Tensor q = work_.alloc(DType::BF16, {D, heads, T}), gate = work_.alloc(DType::BF16, {QW, T});
    Tensor k = work_.alloc(DType::BF16, {D, kv_heads, T}), v = work_.alloc(DType::BF16, {D, kv_heads, T});
    Tensor iq = work_.alloc(DType::BF16, {ID, IH, T}), ik = work_.alloc(DType::BF16, {ID, T});
    {
        Tensor qf = q.view({QW, T}), kf = k.view({KW, T}), vf = v.view({KW, T}), iqf = iq.view({IH * ID, T});
        const LinearParameters& index = p.index_projection ? *p.index_projection : p.projection;
        const std::int32_t index_row  = p.index_projection ? 0 : 2 * QW + 2 * KW;
        const LosslessPart parts[] = {{&p.projection, nullptr, 0, QW, &qf, 0},
                                      {&p.projection, nullptr, QW, QW, &gate, 0},
                                      {&p.projection, nullptr, 2 * QW, KW, &kf, 0},
                                      {&p.projection, nullptr, 2 * QW + KW, KW, &vf, 0},
                                      {&index, nullptr, index_row, IH * ID, &iqf, 0},
                                      {&index, nullptr, index_row + IH * ID, ID, &ik, 0}};
        lossless_project_shared(x, parts);
    }
    // q_norm, k_norm and the index query's norm per head; RoPE at the call's positions.
    Tensor qn = work_.alloc(DType::BF16, {D, heads, T}), kn = work_.alloc(DType::BF16, {D, kv_heads, T});
    Tensor iqn = work_.alloc(DType::BF16, {ID, IH, T});
    {
        Tensor qr = q.view({D, heads * T}), qo = qn.view({D, heads * T});
        Tensor kr = k.view({D, kv_heads * T}), ko = kn.view({D, kv_heads * T});
        Tensor ir = iq.view({ID, IH * T}), io = iqn.view({ID, IH * T});
        ops::lossless_rmsnorm(qr, p.query_norm, config_.rms_norm_eps, qo, s);
        ops::lossless_rmsnorm(kr, p.key_norm, config_.rms_norm_eps, ko, s);
        ops::lossless_rmsnorm(ir, p.index_query_norm, config_.rms_norm_eps, io, s);
    }
    const std::int32_t pairs = dim(config_.rope.rotary_dim) / 2;
    Tensor inv_freq = work_.alloc(DType::FP32, {pairs});
    ops::lossless_rope_frequencies(config_.rope.theta, dim(config_.rope.rotary_dim), inv_freq, s);
    Tensor table = work_.alloc(DType::BF16, {pairs, T, 2});
    ops::lossless_rope_table(batch.rope_positions, inv_freq, table, s);
    ops::lossless_rope_apply(table, qn, s);
    ops::lossless_rope_apply(table, kn, s);
    ops::lossless_rope_apply(table, iqn, s);
    if (batch.lossless_keys != 0) {
        Tensor out = lossless_cached_attention(p, qn, kn, v, iqn, ik, inv_freq, index, batch);
        Tensor gated = work_.alloc(DType::BF16, {QW, T});
        ops::lossless_sigmoid_mul(out.view({QW, T}), gate, gated, s);
        lossless_project(gated, p.output, 0, H, y, 0);
        return y;
    }
    // The indexer: pooled block keys (normed, rotated at their first token's position), each query's
    // scores against them, and its selection when it has more complete blocks than the budget.
    const std::int32_t blocks = T / ratio;
    const std::int32_t words  = std::max<std::int32_t>(1, (blocks + 31) / 32);
    const std::uint32_t* select = nullptr;
    if (blocks > budget) {
        Tensor pooled = work_.alloc(DType::BF16, {ID, blocks}), pooled_n = work_.alloc(DType::BF16, {ID, blocks});
        ops::lossless_qsa_pool(ik, ratio, pooled, s);
        ops::lossless_rmsnorm(pooled, p.index_key_norm, config_.rms_norm_eps, pooled_n, s);
        Tensor block_positions = work_.alloc(DType::I32, {blocks, 3});
        for (int axis = 0; axis < 3; ++axis) {
            CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::int32_t*>(block_positions.data) + static_cast<std::size_t>(axis) * blocks,
                                         sizeof(std::int32_t),
                                         static_cast<const std::int32_t*>(batch.rope_positions.data) +
                                             static_cast<std::size_t>(axis) * T,
                                         sizeof(std::int32_t) * ratio, sizeof(std::int32_t), static_cast<std::size_t>(blocks),
                                         cudaMemcpyDeviceToDevice, s));
        }
        Tensor block_table = work_.alloc(DType::BF16, {pairs, blocks, 2});
        ops::lossless_rope_table(block_positions, inv_freq, block_table, s);
        Tensor pooled_heads = pooled_n.view({ID, 1, blocks});
        ops::lossless_rope_apply(block_table, pooled_heads, s);
        Tensor scores = work_.alloc(DType::FP32, {blocks, IH * T});
        const Tensor index_queries = iqn.view({ID, IH * T});
        ops::lossless_linear(index_queries, bf16_rows(pooled_n), 0, blocks, scores, 0, work_, s);
        Tensor bits = work_.alloc(DType::I32, {words, T});
        const DeviceSpan scratch = work_.alloc_bytes(ops::lossless_qsa_select_workspace_bytes(T, blocks));
        ops::lossless_qsa_select(scores, IH, ratio, budget, ID, static_cast<std::uint32_t*>(bits.data), words,
                                 scratch.data, s);
        select = static_cast<const std::uint32_t*>(bits.data);
    }
    Tensor out = work_.alloc(DType::BF16, {D, heads, T});
    if (config_.design_reference && config_.design_kv != KvCacheStorage::BFloat16) {
        // The design reference with a coded KV format: K and V through its production encoder, attention over the
        // stored rows (lossless-route.md §3b).
        const DeviceSpan stored_rows  = work_.alloc_bytes(ops::lossless_kv_encode_bytes(config_.design_kv, kv_heads, T));
        const DeviceSpan attention_ws = work_.alloc_bytes(ops::lossless_codec_attention_workspace_bytes(heads, kv_heads, T));
        const ops::LosslessKvCodes stored = ops::lossless_kv_encode(kn, v, config_.design_kv, stored_rows.data, s);
        ops::lossless_codec_attention(qn, stored, kv_heads, out, attention_ws.data, s, select, words, ratio);
    } else {
        const DeviceSpan attention_ws = work_.alloc_bytes(ops::lossless_attention_workspace_bytes(heads, kv_heads, T));
        // The design reference's bf16 KV stores V as FP16 (exact in BF16 again).
        if (config_.design_reference) { ops::lossless_round_fp16(v, s); }
        ops::lossless_attention(qn, kn, v, out, attention_ws.data, s, select, words, ratio);
    }
    Tensor gated = work_.alloc(DType::BF16, {QW, T});
    ops::lossless_sigmoid_mul(out.view({QW, T}), gate, gated, s);
    lossless_project(gated, p.output, 0, H, y, 0);
    return y;
}

// The cache form's QSA attention (rotated q, k and index query; raw index keys ik): the call's K/V into the
// pages; the pooled key of every block the call completes (its keys from the tail before the run), normed
// and rotated at its first token, into the pooled plane; the open block's raw keys into the tail; each
// query's block selection from its scores against its sequence's pooled keys when it may have more complete
// blocks than the budget; attention over the cache.
Tensor Forward::lossless_cached_attention(const AttentionParameters& p, const Tensor& qn, const Tensor& kn, const Tensor& v,
                                          const Tensor& iqn, const Tensor& ik, const Tensor& inv_freq, std::uint32_t index,
                                          const ForwardBatch& batch) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = qn.ne[2], D = qn.ne[0], heads = qn.ne[1];
    const std::int32_t IH = dim(config_.qsa.index_heads), ID = dim(config_.qsa.index_head_dim);
    const std::int32_t ratio = dim(config_.qsa.compress_ratio), budget = dim(config_.qsa.block_budget());
    const std::int32_t B = batch.batch, W = batch.width, keys = batch.lossless_keys;
    const ops::QsaKVLayer& layer = kv_.layers.at(index);
    Tensor& tails = state_.qsa_tails.at(index);
    const PagedKVBatchLayerView view{.k_pages      = layer.kv.k_pages,
                                     .v_pages      = layer.kv.v_pages,
                                     .block_tables = kv_.block_tables,
                                     .head_dim     = layer.kv.head_dim,
                                     .num_kv_heads = layer.kv.num_kv_heads,
                                     .storage      = layer.kv.storage};
    Tensor out = work_.alloc(DType::BF16, {D, heads, T});
    auto scope = work_.scope();
    ops::lossless_kv_append(kn, v, batch.positions, B, batch.table_rows, view, s);
    {
        auto pool_scope = work_.scope();
        Tensor pooled = work_.alloc(DType::BF16, {ID, T}), pooled_n = work_.alloc(DType::BF16, {ID, T});
        Tensor block_rope = work_.alloc(DType::I32, {T, 3});
        ops::lossless_qsa_pool_cache(ik, batch.positions, B, tails, batch.slots, batch.rope_positions,
                                     batch.block_start_rope, ratio, pooled, block_rope, s);
        ops::lossless_rmsnorm(pooled, p.index_key_norm, config_.rms_norm_eps, pooled_n, s);
        Tensor block_table = work_.alloc(DType::BF16, {dim(config_.rope.rotary_dim) / 2, T, 2});
        ops::lossless_rope_table(block_rope, inv_freq, block_table, s);
        Tensor pooled_heads = pooled_n.view({ID, 1, T});
        ops::lossless_rope_apply(block_table, pooled_heads, s);
        Tensor plane = layer.pooled_pages;
        ops::lossless_qsa_pooled_store(pooled_n, batch.positions, B, kv_.block_tables, batch.table_rows, ratio, plane, s);
        ops::lossless_qsa_tails_store(ik, batch.positions, B, tails, batch.slots, ratio, s);
    }
    const std::int32_t blocks = keys / ratio;
    const std::int32_t words  = std::max<std::int32_t>(1, (blocks + 31) / 32);
    const std::uint32_t* select = nullptr;
    Tensor bits;
    if (blocks > budget) {
        bits = work_.alloc(DType::I32, {words, T});
        auto score_scope = work_.scope();
        Tensor gathered = work_.alloc(DType::BF16, {ID, blocks});
        Tensor scores   = work_.alloc(DType::FP32, {blocks, IH * kScoreColumns});
        const DeviceSpan scratch = work_.alloc_bytes(ops::lossless_qsa_select_workspace_bytes(kScoreColumns, blocks));
        for (std::int32_t b = 0; b < B; ++b) {
            ops::lossless_qsa_pooled_gather(layer.pooled_pages, kv_.block_tables, batch.table_rows, batch.positions, B,
                                            b, ratio, gathered, s);
            const Weight pooled_keys = bf16_rows(gathered);
            for (std::int32_t t0 = 0; t0 < W; t0 += kScoreColumns) {
                const std::int32_t n = std::min(kScoreColumns, W - t0), column = b * W + t0;
                const Tensor queries = iqn.slice(2, column, n).view({ID, IH * n});
                Tensor tile_scores   = scores.slice(1, 0, IH * n);
                ops::lossless_linear(queries, pooled_keys, 0, blocks, tile_scores, 0, work_, s);
                const Tensor tile_positions = batch.positions.slice(0, column, n);
                ops::lossless_qsa_select(tile_scores, IH, ratio, budget, ID,
                                         static_cast<std::uint32_t*>(bits.data) + static_cast<std::size_t>(column) * words,
                                         words, scratch.data, s, &tile_positions);
            }
        }
        select = static_cast<const std::uint32_t*>(bits.data);
    }
    const DeviceSpan ws = work_.alloc_bytes(
        ops::lossless_paged_attention_workspace_bytes(heads, view.num_kv_heads, W, B, max_context_));
    ops::lossless_paged_attention(qn, batch.positions, B, batch.table_rows, view, keys, max_context_, out, ws.data, s,
                                  select, words, ratio);
    return out;
}

Tensor Forward::lossless_moe(const MoeParameters& p, const Tensor& x, std::uint32_t layer) {
    const cudaStream_t s = device_.stream;
    const auto& m        = config_.moe;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size), E = dim(m.experts), K = dim(m.top_k);
    const std::int32_t I = dim(m.shared_intermediate);
    Tensor ym  = work_.alloc(DType::BF16, {H, T});
    auto scope = work_.scope();
    // Router: l = bf16(x W_router), w = bf16(fp32(p / fp32(sum of the top k))); the shared gate's logit.
    Tensor logits = work_.alloc(DType::BF16, {E, T});
    Tensor shared_logit = work_.alloc(DType::BF16, {1, T});
    // The router, the shared gate's logit and the shared expert's gate and up: one encode of x.
    Tensor gate = work_.alloc(DType::BF16, {I, T}), up = work_.alloc(DType::BF16, {I, T});
    {
        const Weight router = bf16_rows(p.router), shared_score = bf16_rows(p.shared_score);
        const LosslessPart parts[] = {{nullptr, &router, 0, E, &logits, 0},
                                      {nullptr, &shared_score, 0, 1, &shared_logit, 0},
                                      {&p.shared_gate_up, nullptr, 0, I, &gate, 0},
                                      {&p.shared_gate_up, nullptr, I, I, &up, 0}};
        lossless_project_shared(x, parts);
    }
    ops::MoeRouting routing{work_.alloc(DType::I32, {K, T}), work_.alloc(DType::FP32, {K, T}),
                            work_.alloc(DType::FP32, {T})};
    ops::lossless_router(logits, E, K, routing.weights, routing.ids, s, /*bf16_weights=*/true);
    std::int32_t* route_log = nullptr;
    if (experts_.route_log != nullptr && static_cast<std::size_t>(K) * T <= experts_.route_stride) {
        route_log = experts_.route_log + layer * experts_.route_stride;
    }
    const DeviceSpan dispatch_bytes = work_.alloc_bytes(ops::moe_dispatch_bytes(E, K * T));
    ops::MoeDispatch dispatch       = ops::carve_moe_dispatch(dispatch_bytes.data, E, K * T);
    ops::moe_dispatch(routing, E, dispatch, route_log, s);
    // The routed experts: every bank's lossless pass, GPU only, serial staging passes.
    ops::MoeExpertSource source = expert_source(p, layer);
    source.cpu       = ops::MoeCpuChannel{};
    source.scales    = p.lossless_scales;
    source.wide_a16  = ops::offloaded_moe::WideA16::kLossless;
    // The design reference: a W4A4 bank's experts quantize their inputs (lossless_scales then carry the
    // input scales beside the weights' multipliers).
    source.lossless_design = p.lossless_design;
    Tensor outputs   = work_.alloc(DType::BF16, {H, K * T});
    const std::int32_t max_jobs = std::min(E, K * T);
    const DeviceSpan expert_ws  = work_.alloc_bytes(ops::moe_experts_workspace_bytes(
        source.shape, max_jobs, K * T, T, source.activation, ops::offloaded_moe::WideA16::kLossless));
    ops::moe_experts(x, dispatch, source, K, max_jobs, expert_ws.data, outputs, s, /*wait_for_cpu=*/false, false);
    // The shared expert (its gate and up projected with the router above) and its gate.
    Tensor product = work_.alloc(DType::BF16, {I, T});
    ops::lossless_silu_mul(gate, up, product, s);
    Tensor shared = work_.alloc(DType::BF16, {H, T});
    lossless_project(product, p.shared_down, 0, H, shared, 0);
    // y = bf16(sum over the experts in ascending id of bf16(y_e w_e), then + the gated shared output).
    ops::lossless_moe_combine(outputs, routing.weights, routing.ids, Tensor{}, &shared, &shared_logit, 0, ym, s,
                              /*add=*/false);
    return ym;
}

void Forward::lossless_layer(std::uint32_t layer, const ForwardBatch& batch, Tensor& residual) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = residual.ne[1];
    const auto& block    = parameters_.layers[layer];
    const std::uint32_t compact = config_.compact_layer_indices[layer];
    auto scope           = work_.scope();
    if (block.ple) { lossless_ple(*block.ple, residual, batch); }
    Tensor inject = work_.alloc(DType::BF16, {dim(config_.hc.streams), T});
    {
        auto mixer_scope = work_.scope();
        const Tensor xa  = lossless_mix(block.attn_hc, residual, &inject);
        const Tensor y   = config_.layer_types[layer] == MixerKind::Gdn
                               ? lossless_gdn(std::get<GdnParameters>(block.mixer), xa, compact, batch)
                               : lossless_attention_block(std::get<AttentionParameters>(block.mixer), xa, compact, batch);
        ops::lossless_hc_inject(y, inject, residual, s);
    }
    const Tensor xm = lossless_mix(block.mlp_hc, residual, &inject);
    const Tensor ym = lossless_moe(block.moe, xm, layer);
    ops::lossless_hc_inject(ym, inject, residual, s);
}

void Forward::lossless_head(const Tensor& residuals, Tensor& logits) {
    const cudaStream_t s = device_.stream;
    auto scope           = work_.scope();
    const Tensor x       = lossless_mix(parameters_.final_mixer, residuals, nullptr);
    if (parameters_.output_head_q8) {
        lossless_project(x, *parameters_.output_head_q8, 0, logits.ne[0], logits, 0);
    } else {
        ops::lossless_linear(x, bf16_rows(parameters_.output_head), 0, logits.ne[0], logits, 0, work_, s);
    }
}

void Forward::check_lossless_call(const ForwardBatch& batch) const {
    require(batch.verify == nullptr && batch.mtp_chunk == nullptr && batch.vision == nullptr &&
                batch.draft_features.data == nullptr,
            "a call is text without verification, MTP, vision or drafter features");
    require(batch.lossless_keys != 0 ? batch.lossless_keys <= max_context_ : batch.batch == 1,
            "a scored window is one sequence; the cache form's keys stay within the context");
    require(config_.rope.yarn_factor == 1.0F, "YaRN-scaled rotations are not part of the route");
    for (std::size_t i = 0; i < config_.rope.pair_axes.size(); ++i) {
        require(config_.rope.pair_axes[i] == i % 3, "the rotation's pairs must be interleaved over the three axes");
    }
}

std::size_t Forward::lossless_workspace_bytes(const TextConfig& c, std::int32_t columns, std::int32_t key_capacity,
                                              std::int32_t sequences) {
    const std::size_t t = static_cast<std::size_t>(std::max(columns, 1));
    const std::size_t h = c.hidden_size, w = c.residual_width(), k = c.moe.top_k, bf = 2;
    const std::size_t S = c.hc.streams, rank = c.hc.rank;
    // Projections: the largest lossless_linear layout (BF16 weights: every row's block exponents too).
    const auto linear = [&](std::size_t rows, std::size_t kk) {
        Weight weight{};
        weight.qtype  = QType::BF16;
        weight.layout = QuantLayout::Contiguous;
        weight.n      = static_cast<std::int32_t>(rows);
        weight.k      = static_cast<std::int32_t>(kk);
        return ops::lossless_linear_workspace_bytes(weight, static_cast<std::int32_t>(rows), static_cast<std::int32_t>(t));
    };
    std::size_t projection = std::max({linear(w, w), linear(w, c.ple.embed_dim), linear(c.vocab_size, h),
                                       linear(c.moe.experts, h)});
    projection += w * t * bf; // a permuted input
    const std::size_t mix  = h * t * bf + w * t * bf * 2 + (rank + S) * t * bf + rank * t * bf;
    const std::size_t ple  = std::size_t(c.ple.embed_dim) * t * bf + w * t * bf * 5 + h * t * bf + S * t * bf;
    const std::size_t gdn  = h * t * bf + std::size_t(c.gdn.conv_channels()) * t * bf * 2 +
                            std::size_t(c.gdn.value_width()) * t * bf * 3 + std::size_t(c.gdn.value_heads) * t * (bf * 3 + 4);
    const std::size_t blocks = t / std::max<std::uint32_t>(c.qsa.compress_ratio, 1);
    const std::size_t attn = h * t * bf + std::size_t(c.attention.query_width()) * t * bf * 5 +
                             std::size_t(c.attention.key_width()) * t * bf * 3 +
                             std::size_t(c.qsa.index_heads + 1) * c.qsa.index_head_dim * t * bf * 2 +
                             std::size_t(c.qsa.index_head_dim) * blocks * bf * 2 + blocks * 12 +
                             std::size_t(c.rope.rotary_dim) * (t + blocks) * bf + blocks * c.qsa.index_heads * t * 4 +
                             ((blocks + 31) / 32 + 1) * t * 4 + ops::lossless_qsa_select_workspace_bytes(dim(t), dim(blocks)) +
                             (c.design_reference && c.design_kv != KvCacheStorage::BFloat16
                                  ? ops::lossless_codec_attention_workspace_bytes(dim(c.attention.heads),
                                                                                  dim(c.attention.kv_heads), dim(t)) +
                                        ops::lossless_kv_encode_bytes(c.design_kv, dim(c.attention.kv_heads), dim(t)) + 512
                                  : ops::lossless_attention_workspace_bytes(dim(c.attention.heads),
                                                                            dim(c.attention.kv_heads), dim(t)));
    std::size_t cached = 0;
    if (key_capacity > 0) {
        // The cache form: the window's projections and rotations as above, the pooled-key steps, each
        // sequence's gathered pooled keys and one tile of scores and selection, the selection bits and the
        // paged attention.
        const std::size_t kb = static_cast<std::size_t>(key_capacity) / std::max<std::uint32_t>(c.qsa.compress_ratio, 1);
        const std::size_t ih = c.qsa.index_heads, id = c.qsa.index_head_dim;
        Weight pooled{};
        pooled.qtype  = QType::BF16;
        pooled.layout = QuantLayout::Contiguous;
        pooled.n      = dim(std::max<std::size_t>(kb, 1));
        pooled.k      = dim(id);
        cached = h * t * bf + std::size_t(c.attention.query_width()) * t * bf * 5 +
                 std::size_t(c.attention.key_width()) * t * bf * 3 + (ih + 1) * id * t * bf * 2 +
                 std::size_t(c.rope.rotary_dim) * t * bf + 2 * id * t * bf + t * 12 + std::size_t(c.rope.rotary_dim) * t * bf +
                 ((kb + 31) / 32 + 1) * t * 4 + id * kb * bf + kb * ih * kScoreColumns * 4 +
                 ops::lossless_qsa_select_workspace_bytes(kScoreColumns, dim(kb)) +
                 ops::lossless_linear_workspace_bytes(pooled, pooled.n, dim(ih * kScoreColumns)) +
                 ops::lossless_paged_attention_workspace_bytes(dim(c.attention.heads), dim(c.attention.kv_heads), dim(t),
                                                               std::max(sequences, 1), key_capacity);
    }
    const std::size_t moe = h * t * bf + c.moe.experts * t * bf + t * bf + k * t * 8 + t * 4 +
                            ops::moe_dispatch_bytes(dim(c.moe.experts), dim(k * t)) + h * k * t * bf +
                            ops::moe_experts_workspace_bytes(ops::offloaded_moe::kFlashNextShape,
                                                             std::min<std::int32_t>(dim(c.moe.experts), dim(k * t)), dim(k * t),
                                                             dim(t), ops::offloaded_moe::ExpertActivation::kA16,
                                                             ops::offloaded_moe::WideA16::kLossless) +
                            std::size_t(c.moe.shared_intermediate) * t * bf * 3 + h * t * bf;
    const std::size_t layer = S * t * bf + std::max({ple, mix + std::max({gdn, attn, cached}), mix + moe});
    const std::size_t head  = mix + std::size_t(c.vocab_size) * 4; // per scored column, inside the score tiles
    return w * t * bf + std::max(layer, head) + projection + (256ULL << 20);
}

} // namespace infernix::models::qwen4_exp::execution
