#include "models/qwen4_exp/execution/forward.h"

#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/projection_fp32.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/resident_moe.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/rows.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen4_exp::execution {
namespace {

std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }

// Columns of one MTP call inside a prefill chunk (bounds the drafter's share of the workspace).
constexpr std::int32_t kMtpChunkColumns = 512;

void project(const Tensor& x, const LinearParameters& p, Tensor& out, WorkspaceArena& work, cudaStream_t s) {
    ops::linear(x, p.weight, out, p.policy, work, s);
}

} // namespace

ops::QsaGeometry qsa_geometry(const TextConfig& c) {
    return {.heads          = dim(c.attention.heads),
            .kv_heads       = dim(c.attention.kv_heads),
            .head_dim       = dim(c.attention.head_dim),
            .index_heads    = dim(c.qsa.index_heads),
            .index_head_dim = dim(c.qsa.index_head_dim),
            .rotary_dim     = dim(c.rope.rotary_dim),
            .budget         = dim(c.qsa.budget),
            .ratio          = dim(c.qsa.compress_ratio),
            .theta          = c.rope.theta,
            .eps            = c.rms_norm_eps};
}

Forward::Forward(const Parameters& parameters, DeviceContext& device, WorkspaceArena& work,
                 ForwardState state, ForwardKV kv, ForwardExperts experts, std::int32_t max_context)
    : parameters_(parameters), config_(parameters.model.config().text), device_(device), work_(work),
      state_(std::move(state)), kv_(std::move(kv)), experts_(std::move(experts)), max_context_(max_context) {
    if (state_.gdn == nullptr || state_.gdn->layer_count() != config_.gdn_layers ||
        state_.qsa_tails.size() != config_.attention_layers || kv_.layers.size() != config_.attention_layers ||
        experts_.frames.size() != config_.num_hidden_layers) {
        throw std::invalid_argument("Qwen4Exp forward: state, KV or expert residency is incomplete");
    }
}

std::size_t Forward::workspace_bytes(const TextConfig& c, std::int32_t columns, std::int32_t max_context) {
    const std::size_t t = static_cast<std::size_t>(columns);
    const std::size_t h = c.hidden_size, w = c.residual_width(), k = c.moe.top_k;
    const std::size_t bf = 2;
    std::size_t bytes = 0;
    bytes += w * t * bf * 3;                                  // residual, normalized, mix logits
    bytes += (c.hc.rank + c.hc.streams) * t * (bf + 4) + c.hc.rank * t * bf;
    bytes += h * t * bf * 4;                                  // x, y, embedding, final
    bytes += std::size_t(c.ple.embed_dim) * t * bf * 2 + (w + h) * t * bf + w * t * bf * 2;
    bytes += std::size_t(c.gdn.conv_channels()) * t * bf * 2 + std::size_t(c.gdn.value_width()) * t * bf * 4;
    bytes += std::size_t(c.gdn.value_heads) * t * (bf + 4) * 2;
    const std::size_t qkv = 2 * std::size_t(c.attention.query_width()) + 2 * c.attention.key_width() +
                            std::size_t(c.qsa.index_heads + 1) * c.qsa.index_head_dim;
    bytes += qkv * t * bf * 2 + std::size_t(c.attention.query_width()) * t * bf * 2;
    bytes += ops::qsa_attention_workspace_bytes(qsa_geometry(c), columns, max_context);
    bytes += (c.moe.experts + 1) * t * 4 + k * t * 8 + t * 4;           // FP32 router logits, routing
    bytes += ops::moe_dispatch_bytes(dim(c.moe.experts), dim(k * t));
    bytes += h * k * t * bf + ops::moe_experts_workspace_bytes(dim(c.moe.experts), dim(k * t));
    bytes += 2 * std::size_t(c.moe.shared_intermediate) * t * bf * 3;
    // Linear and GDN scratch (chunked recurrence), plus alignment slack per allocation.
    bytes += ops::gated_delta_net_workspace_capacity_bytes(dim(c.gdn.key_heads), dim(c.gdn.value_heads), 1,
                                                           std::max(columns, 1));
    bytes += 64ULL << 20;
    return bytes;
}

void Forward::run(const ForwardBatch& batch, Tensor& logits, const ForwardTap* tap) {
    const std::int32_t T = batch.ids.ne[0];
    if (batch.verify != nullptr && (batch.width < 2 || batch.verify->gdn.spec.width != batch.width ||
                                    batch.verify->gdn.spec.record_capacity < batch.batch)) {
        throw std::invalid_argument("Qwen4Exp forward: verification records do not match the call");
    }
    if (batch.batch <= 0 || batch.width <= 0 || batch.batch * batch.width != T ||
        logits.dtype != DType::FP32 || logits.ne[1] != batch.logit_columns.ne[0] ||
        logits.ne[0] != dim(config_.vocab_size)) {
        throw std::invalid_argument("Qwen4Exp forward: batch geometry is invalid");
    }
    const cudaStream_t s = device_.stream;
    const std::int32_t H = dim(config_.hidden_size), W = dim(config_.residual_width());
    // Prefill chunks and forced tokens run eagerly; decode and verification rounds may be captured.
    eager_chunk_ = batch.verify == nullptr && batch.width > 1;
    work_.reset();
    Tensor x0 = work_.alloc(DType::BF16, {H, T});
    ops::embedding(batch.ids, parameters_.token_embedding, x0, s);
    Tensor residual = work_.alloc(DType::BF16, {W, T});
    ops::hyper_connection_expand(x0, dim(config_.hc.streams), residual, s);

    const auto wait_layer = [&](std::size_t entry) {
        for (const auto& events : batch.layer_waits) {
            if (!events.empty()) { CUDA_CHECK(cudaStreamWaitEvent(s, events[entry], 0)); }
        }
    };
    for (const auto& events : batch.layer_waits) {
        if (!events.empty() && events.size() != config_.num_hidden_layers + 1U) {
            throw std::invalid_argument("Qwen4Exp forward: layer waits do not match the layers");
        }
    }
    for (std::uint32_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
        const auto& block = parameters_.layers[layer];
        wait_layer(layer);
        try {
            auto scope = work_.scope();
            if (block.ple) { ple(*block.ple, residual, batch); }
            Tensor inject = work_.alloc(DType::FP32, {dim(config_.hc.streams), T});
            Tensor xa     = mix(block.attn_hc, residual, &inject);
            const std::uint32_t compact = config_.compact_layer_indices[layer];
            Tensor y;
            if (config_.layer_types[layer] == MixerKind::Gdn) {
                y = gdn(std::get<GdnParameters>(block.mixer), xa, compact, batch);
            } else {
                // A verification call records the raw index keys for the tail commit.
                const AttentionCall call{
                    .layer        = kv_.layers.at(compact),
                    .tails        = state_.qsa_tails.at(compact),
                    .key_records  = batch.verify != nullptr
                                        ? batch.verify->qsa_keys.slice(3, static_cast<std::int32_t>(compact), 1)
                                              .view({dim(config_.qsa.index_head_dim), T})
                                        : Tensor{},
                    .update_tails = batch.verify == nullptr,
                    .positions    = batch.positions,
                    .slots        = batch.slots,
                    .table_rows   = batch.table_rows,
                    .batch        = batch.batch,
                    .width        = batch.width};
                y = attention(std::get<AttentionParameters>(block.mixer), xa, call);
            }
            ops::hyper_connection_inject(y, inject, residual, s);
            Tensor xm = mix(block.mlp_hc, residual, &inject);
            Tensor ym = moe(block.moe, xm, layer,
                            tap != nullptr && tap->routes != nullptr ? &tap->routes->at(layer) : nullptr);
            ops::hyper_connection_inject(ym, inject, residual, s);
            if (tap != nullptr) {
                const std::pair<std::vector<Tensor>*, const Tensor*> copies[] = {
                    {tap->mixer_inputs, &xa}, {tap->mixer_outputs, &y}, {tap->moe_inputs, &xm}, {tap->moe_outputs, &ym}};
                for (const auto& [targets, source] : copies) {
                    if (targets != nullptr &&
                        cudaMemcpyAsync(targets->at(layer).data, source->data, source->bytes(),
                                        cudaMemcpyDeviceToDevice, s) != cudaSuccess) {
                        throw std::runtime_error("Qwen4Exp forward: block tap copy failed");
                    }
                }
            }
        } catch (const std::exception& error) {
            throw std::runtime_error("qwen4_exp/layers/" + std::to_string(layer) + ": " + error.what());
        }
        if (tap != nullptr && tap->residuals != nullptr) {
            Tensor& copy = tap->residuals->at(layer);
            if (cudaMemcpyAsync(copy.data, residual.data, residual.bytes(), cudaMemcpyDeviceToDevice, s) !=
                cudaSuccess) {
                throw std::runtime_error("Qwen4Exp forward: residual tap copy failed");
            }
        }
    }
    if (batch.residual_out.data != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(batch.residual_out.data, residual.data, residual.bytes(), cudaMemcpyDeviceToDevice, s));
    }
    wait_layer(config_.num_hidden_layers);
    if (batch.mtp_chunk != nullptr) {
        // The chunk's MTP cells from its live residuals, then its last residual becomes pending.
        const MtpChunk& chunk = *batch.mtp_chunk;
        const std::int32_t cells = (chunk.prepend ? 1 : 0) + chunk.columns;
        const std::size_t column = static_cast<std::size_t>(W) * 2;
        if (chunk.columns < 0 || chunk.columns > T || batch.batch != 1 ||
            (cells > 0 && (chunk.ids.numel() != cells || chunk.positions.numel() != cells))) {
            throw std::invalid_argument("Qwen4Exp forward: MTP chunk does not match the call");
        }
        // Cell v is `saved` (v = 0 when prepending) or chunk column v - prepend; sub-chunks bound
        // the drafter's workspace.
        const std::int32_t first_column = chunk.prepend ? -1 : 0;
        for (std::int32_t begin = 0; begin < cells; begin += kMtpChunkColumns) {
            const std::int32_t n = std::min(kMtpChunkColumns, cells - begin);
            auto scope  = work_.scope();
            Tensor rows = work_.alloc(DType::BF16, {W, n});
            auto* out   = static_cast<std::byte*>(rows.data);
            std::int32_t from = first_column + begin;
            if (from < 0) {
                CUDA_CHECK(cudaMemcpyAsync(out, chunk.saved.data, column, cudaMemcpyDeviceToDevice, s));
                out += column;
                from = 0;
            }
            const std::int32_t rest = first_column + begin + n - from;
            if (rest > 0) {
                CUDA_CHECK(cudaMemcpyAsync(out, static_cast<const std::byte*>(residual.data) + column * from,
                                           column * rest, cudaMemcpyDeviceToDevice, s));
            }
            MtpCall call{.residuals  = rows,
                         .ids        = chunk.ids.slice(0, begin, n),
                         .positions  = chunk.positions.slice(0, begin, n),
                         .slots      = batch.slots,
                         .table_rows = batch.table_rows,
                         .batch      = 1,
                         .width      = n,
                         .kv_only    = true};
            mtp_block(call, rows);
        }
        CUDA_CHECK(cudaMemcpyAsync(chunk.saved.data, static_cast<const std::byte*>(residual.data) + column * (T - 1),
                                   column, cudaMemcpyDeviceToDevice, s));
    }
    Tensor final_x = mix(parameters_.final_mixer, residual, nullptr);
    Tensor last = work_.alloc(DType::BF16, {H, batch.logit_columns.ne[0]});
    ops::gather_columns(final_x, batch.logit_columns, last, s);
    if (parameters_.output_head_q8) {
        ops::projection_fp32(last, parameters_.output_head_q8->weight, logits, s);
    } else {
        const Tensor* head[] = {&parameters_.output_head};
        ops::projection_fp32(last, head, logits, s);
    }
}

Tensor Forward::mix(const HyperConnectionParameters& p, const Tensor& residual, Tensor* inject) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = residual.ne[1], W = residual.ne[0], H = dim(config_.hidden_size);
    const std::int32_t S = dim(config_.hc.streams), rank = dim(config_.hc.rank);
    Tensor normalized = work_.alloc(DType::BF16, {W, T});
    ops::hyper_connection_norm(residual, p.norm, S, config_.rms_norm_eps, normalized, s);
    Tensor z = work_.alloc(DType::BF16, {p.down.weight.n, T});
    project(normalized, p.down, z, work_, s);
    Tensor m = work_.alloc(DType::BF16, {rank, T});
    ops::hyper_connection_gates(z, rank, S, m, p.combine ? inject : nullptr, s);
    Tensor u = work_.alloc(DType::BF16, {W, T});
    project(m, p.up, u, work_, s);
    Tensor x = work_.alloc(DType::BF16, {H, T});
    ops::hyper_connection_collapse(u, normalized, S, x, s);
    return x;
}

void Forward::ple(const PleParameters& p, Tensor& residual, const ForwardBatch& batch) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = residual.ne[1], W = residual.ne[0], H = dim(config_.hidden_size);
    auto scope = work_.scope();
    Tensor e = work_.alloc(DType::BF16, {dim(config_.ple.embed_dim), T});
    ops::ple_embed(batch.ngram_rows, p.ngram_scale, e, s);
    Tensor kv = work_.alloc(DType::BF16, {W + H, T});
    project(e, p.projection, kv, work_, s);
    Tensor key = work_.alloc(DType::BF16, {W, T});
    Tensor value = work_.alloc(DType::BF16, {H, T});
    Tensor* parts[] = {&key, &value};
    ops::split_rows(kv, parts, s);
    Tensor gated      = work_.alloc(DType::BF16, {W, T});
    // A verification call records the convolution inputs for its commit and leaves the history.
    Tensor normalized = batch.verify != nullptr ? batch.verify->ple_inputs.view({W, T})
                                                : work_.alloc(DType::BF16, {W, T});
    ops::ple_gate(key, value, residual, p.key_norm, p.query_norm, p.conv_norm, dim(config_.hc.streams),
                  config_.rms_norm_eps, gated, normalized, s);
    ops::ple_conv_inject(gated, normalized, p.convolution, dim(config_.ple.conv_dilation), state_.ple_conv,
                         batch.slots, batch.verify != nullptr ? Tensor{} : batch.slots, residual, s);
}

Tensor Forward::gdn(const GdnParameters& p, const Tensor& x, std::uint32_t index, const ForwardBatch& batch) {
    const cudaStream_t s = device_.stream;
    const auto& g        = config_.gdn;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size);
    const std::int32_t C = dim(g.conv_channels()), KW = dim(g.key_width()), VW = dim(g.value_width());
    const std::int32_t NV = dim(g.value_heads), NK = dim(g.key_heads), DK = dim(g.key_head_dim),
                       DV = dim(g.value_head_dim);
    Tensor projected = work_.alloc(DType::BF16, {C + VW, T});
    project(x, p.projection, projected, work_, s);
    // A verification call projects its convolution inputs straight into the replay records.
    GdnReplayRecordLayer records;
    if (batch.verify != nullptr) { records = batch.verify->gdn.layer(static_cast<std::int32_t>(index), batch.batch); }
    Tensor qkv = batch.verify != nullptr ? records.conv.view({C, T}) : work_.alloc(DType::BF16, {C, T});
    Tensor z   = work_.alloc(DType::BF16, {VW, T});
    {
        Tensor* parts[] = {&qkv, &z};
        ops::split_rows(projected, parts, s);
    }
    Tensor ab = work_.alloc(DType::BF16, {2 * NV, T});
    project(x, p.control, ab, work_, s);
    Tensor a = work_.alloc(DType::BF16, {NV, T});
    Tensor b = work_.alloc(DType::BF16, {NV, T});
    {
        Tensor* parts[] = {&a, &b};
        ops::split_rows(ab, parts, s);
    }
    Tensor gate = work_.alloc(DType::FP32, {NV, T});
    Tensor beta = work_.alloc(DType::FP32, {NV, T});
    ops::gdn_gating(a, b, p.a_log, p.dt_bias, gate, beta, s);

    Tensor q = work_.alloc(DType::BF16, {KW, T});
    Tensor k = work_.alloc(DType::BF16, {KW, T});
    Tensor v = work_.alloc(DType::BF16, {VW, T});
    Tensor o = work_.alloc(DType::BF16, {DV, NV, T});
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(DK)));
    if (batch.verify != nullptr) {
        // Speculative verification: the convolution and recurrence run from the live states
        // without changing them, and the recurrence's inputs are recorded for the commit fold.
        const std::int32_t Wd = batch.width, B = batch.batch;
        auto layer = state_.gdn->layer_view(index);
        Tensor conv_out = work_.alloc(DType::BF16, {C, Wd, B});
        ops::causal_conv1d_silu_from_states(qkv.view({C, Wd, B}), p.convolution, layer.conv, batch.slots,
                                            conv_out, s);
        Tensor* parts[] = {&q, &k, &v};
        ops::split_rows(conv_out.view({C, T}), parts, s);
        Tensor o_rows = o.view({DV, NV, Wd, B});
        ops::gated_delta_net_replay_record(q.view({DK, NK, Wd, B}), k.view({DK, NK, Wd, B}), v.view({DV, NV, Wd, B}),
                                           gate.view({NV, Wd, B}), beta.view({NV, Wd, B}), scale, layer.recurrent,
                                           Tensor{}, batch.slots, records.key, records.value, records.gate,
                                           o_rows, s);
    } else if (batch.width > 1) {
        // One sequence of several positions (a prefill chunk): chunked recurrence.
        if (batch.batch != 1) { throw std::invalid_argument("Qwen4Exp GDN: multi-position calls hold one sequence"); }
        if (batch.host_slots.empty()) { throw std::invalid_argument("Qwen4Exp GDN: missing host slot"); }
        const std::int32_t slot = batch.host_slots[0];
        Tensor conv_state = state_.gdn->conv_slot(index, slot);
        ops::causal_conv1d_silu_split(qkv, p.convolution, conv_state, conv_state, q, k, v, s);
        Tensor state = state_.gdn->recurrent_slot(index, slot);
        ops::gated_delta_net(q.view({DK, NK, T}), k.view({DK, NK, T}), v.view({DV, NV, T}), gate, beta, scale,
                             /*normalize_qk=*/true, work_, state, state, o, device_.execution_view());
    } else {
        // One position per sequence (decode): the recurrent update with device-chosen slots.
        auto layer = state_.gdn->layer_view(index);
        Tensor conv_out = work_.alloc(DType::BF16, {C, 1, T});
        ops::causal_conv1d_silu_snapshot(qkv.view({C, 1, T}), p.convolution, layer.conv, Tensor{}, batch.slots,
                                         batch.slots, conv_out, s);
        Tensor* parts[] = {&q, &k, &v};
        ops::split_rows(conv_out.view({C, T}), parts, s);
        Tensor o_batch = o.view({DV, NV, 1, T});
        ops::gated_delta_net_batch_update(q.view({DK, NK, 1, T}), k.view({DK, NK, 1, T}), v.view({DV, NV, 1, T}),
                                          gate.view({NV, 1, T}), beta.view({NV, 1, T}), scale,
                                          /*normalize_qk=*/true, layer.recurrent, batch.slots, batch.slots,
                                          o_batch, s);
    }
    Tensor normed      = work_.alloc(DType::BF16, {DV, NV * T});
    const Tensor rows  = o.view({DV, NV * T});
    const Tensor gates = z.view({DV, NV * T});
    ops::gated_rmsnorm(rows, p.norm, gates,
                       g.output_gate == GateActivation::Sigmoid ? ops::RmsGate::Sigmoid : ops::RmsGate::Silu,
                       config_.rms_norm_eps, normed, s);
    Tensor y = work_.alloc(DType::BF16, {H, T});
    project(normed.view({VW, T}), p.output, y, work_, s);
    return y;
}

Tensor Forward::attention(const AttentionParameters& p, const Tensor& x, const AttentionCall& call) {
    const cudaStream_t s = device_.stream;
    const auto& a        = config_.attention;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size);
    const std::int32_t QW = dim(a.query_width()), KW = dim(a.key_width()), D = dim(a.head_dim);
    const std::int32_t IH = dim(config_.qsa.index_heads), ID = dim(config_.qsa.index_head_dim);
    Tensor projected = work_.alloc(DType::BF16, {2 * QW + 2 * KW + IH * ID + ID, T});
    project(x, p.projection, projected, work_, s);
    Tensor q = work_.alloc(DType::BF16, {D, dim(a.heads), T});
    Tensor gate = work_.alloc(DType::BF16, {QW, T});
    Tensor k = work_.alloc(DType::BF16, {D, dim(a.kv_heads), T});
    Tensor v = work_.alloc(DType::BF16, {D, dim(a.kv_heads), T});
    Tensor iq = work_.alloc(DType::BF16, {ID, IH, T});
    Tensor ik = call.key_records.data != nullptr ? call.key_records.view({ID, T}) : work_.alloc(DType::BF16, {ID, T});
    {
        Tensor qf = q.view({QW, T}), kf = k.view({KW, T}), vf = v.view({KW, T}), iqf = iq.view({IH * ID, T});
        Tensor* parts[] = {&qf, &gate, &kf, &vf, &iqf, &ik};
        ops::split_rows(projected, parts, s);
    }
    Tensor qn = work_.alloc(DType::BF16, {D, dim(a.heads), T});
    Tensor kn = work_.alloc(DType::BF16, {D, dim(a.kv_heads), T});
    {
        Tensor qn_rows = qn.view({D, dim(a.heads) * T});
        Tensor kn_rows = kn.view({D, dim(a.kv_heads) * T});
        ops::rmsnorm(q.view({D, dim(a.heads) * T}), p.query_norm, config_.rms_norm_eps, true, qn_rows, s);
        ops::rmsnorm(k.view({D, dim(a.kv_heads) * T}), p.key_norm, config_.rms_norm_eps, true, kn_rows, s);
    }
    ops::rope(call.positions, dim(config_.rope.rotary_dim), config_.rope.theta, qn, kn, s);

    // Append K/V of every sequence through its device-chosen table row, then the pooled index keys.
    const auto& layer = call.layer;
    {
        const PagedKVBatchLayerView view{.k_pages       = layer.kv.k_pages,
                                         .v_pages       = layer.kv.v_pages,
                                         .k_scale_pages = layer.kv.k_scale_pages,
                                         .v_scale_pages = layer.kv.v_scale_pages,
                                         .block_tables  = kv_.block_tables,
                                         .head_dim      = layer.kv.head_dim,
                                         .num_kv_heads  = layer.kv.num_kv_heads,
                                         .storage       = layer.kv.storage,
                                         .window        = layer.kv.window};
        const std::int32_t KVH = dim(a.kv_heads);
        ops::kv_cache_append_batch(kn.view({D, KVH, call.width, call.batch}), v.view({D, KVH, call.width, call.batch}),
                                   call.positions.view({call.width, call.batch}), call.table_rows, view, s);
    }
    const ops::QsaGeometry geometry = qsa_geometry(config_);
    const ops::QsaBatch qsa_batch{kv_.block_tables, call.table_rows, call.positions, call.slots, call.batch,
                                  call.width, call.update_tails};
    ops::qsa_pool_keys(ik, p.index_key_norm, call.tails, layer, qsa_batch, geometry, s);
    if (call.kv_only) { return Tensor{}; }
    ops::qsa_index_query(iq, p.index_query_norm, call.positions, geometry, s);

    Tensor out = work_.alloc(DType::BF16, {D, dim(a.heads), T});
    const std::size_t scratch = ops::qsa_attention_workspace_bytes(geometry, T, max_context_);
    const DeviceSpan span     = work_.alloc_bytes(scratch);
    ops::qsa_attention(qn, iq, layer, qsa_batch, geometry,
                       static_cast<float>(1.0 / std::sqrt(static_cast<double>(D))), max_context_, span.data,
                       span.bytes, out, s);
    // Output gate: out * sigmoid(gate), then o_proj.
    Tensor gated = out.view({QW, T});
    ops::sigmoid_mul(gate, gated, s);
    Tensor y = work_.alloc(DType::BF16, {H, T});
    project(gated, p.output, y, work_, s);
    return y;
}

Tensor Forward::moe(const MoeParameters& p, const Tensor& x, std::uint32_t layer, Tensor* route_tap) {
    const cudaStream_t s = device_.stream;
    const auto& m        = config_.moe;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size), E = dim(m.experts), K = dim(m.top_k);
    Tensor logits = work_.alloc(DType::FP32, {E + 1, T});
    const Tensor* rows[] = {&p.router, &p.shared_score};
    ops::projection_fp32(x, rows, logits, s);
    ops::MoeRouting routing{work_.alloc(DType::I32, {K, T}), work_.alloc(DType::FP32, {K, T}),
                            work_.alloc(DType::FP32, {T})};
    ops::moe_route(logits, K, routing, s);
    std::int32_t* route_log = nullptr;
    if (experts_.route_log != nullptr) {
        if (static_cast<std::size_t>(K) * T > experts_.route_stride) {
            throw std::invalid_argument("Qwen4Exp MoE: route log is too small for this call");
        }
        route_log = experts_.route_log + layer * experts_.route_stride; // written by moe_dispatch
    }
    if (route_tap != nullptr &&
        cudaMemcpyAsync(route_tap->data, routing.ids.data, routing.ids.bytes(), cudaMemcpyDeviceToDevice, s) !=
            cudaSuccess) {
        throw std::runtime_error("Qwen4Exp forward: route tap copy failed");
    }
    const DeviceSpan dispatch_bytes = work_.alloc_bytes(ops::moe_dispatch_bytes(E, K * T));
    ops::MoeDispatch dispatch       = ops::carve_moe_dispatch(dispatch_bytes.data, E, K * T);
    ops::moe_dispatch(routing, E, dispatch, route_log, s);
    ops::MoeExpertSource source{.frame_base   = experts_.frame_base,
                                .frames       = experts_.frames.at(layer),
                                .host_records = reinterpret_cast<const std::uint8_t*>(p.bank->planes.records),
                                .record_stride = p.bank->planes.record_stride,
                                .scales        = p.device_scales,
                                .staging_base  = experts_.staging_base,
                                .staging_slots = experts_.staging_slots,
                                .cpu           = experts_.cpu.empty() ? ops::MoeCpuChannel{} : experts_.cpu.at(layer)};
    // Prefill chunks (one sequence of many positions, run eagerly) overlap staging with compute.
    if (eager_chunk_ && experts_.overlap_stream != nullptr) {
        source.overlap_stream = experts_.overlap_stream;
        for (int i = 0; i < 5; ++i) { source.overlap_events[i] = experts_.overlap_events[static_cast<std::size_t>(i)]; }
    } else if (experts_.overlap_stream != nullptr) {
        // Decode and verification calls (one pass, possibly graph-captured): resident experts
        // compute while the misses stage on the same side stream.
        source.fork_stream    = experts_.overlap_stream;
        source.fork_events[0] = experts_.overlap_events[0];
        source.fork_events[1] = experts_.overlap_events[1];
    }
    if (experts_.frame_stride != 0 && experts_.frame_stride != source.record_stride) {
        throw std::invalid_argument("Qwen4Exp MoE: frame stride differs from the bank record stride");
    }
    Tensor outputs = work_.alloc(DType::BF16, {H, K * T});
    const std::int32_t max_jobs = std::min(E, K * T);
    const DeviceSpan expert_ws  = work_.alloc_bytes(ops::moe_experts_workspace_bytes(max_jobs, K * T));
    // The shared expert runs while the host computes the CPU-served misses.
    ops::moe_experts(x, dispatch, source, K, max_jobs, expert_ws.data, outputs, s, /*wait_for_cpu=*/false);

    const std::int32_t I = dim(m.shared_intermediate);
    Tensor gate_up = work_.alloc(DType::BF16, {2 * I, T});
    project(x, p.shared_gate_up, gate_up, work_, s);
    Tensor gate = work_.alloc(DType::BF16, {I, T});
    Tensor up   = work_.alloc(DType::BF16, {I, T});
    {
        Tensor* parts[] = {&gate, &up};
        ops::split_rows(gate_up, parts, s);
    }
    Tensor product = work_.alloc(DType::BF16, {I, T});
    ops::silu_mul(gate, up, product, s);
    Tensor shared = work_.alloc(DType::BF16, {H, T});
    project(product, p.shared_down, shared, work_, s);
    ops::moe_experts_cpu_wait(x, dispatch, source, max_jobs, expert_ws.data, outputs, s);
    Tensor y = work_.alloc(DType::BF16, {H, T});
    ops::moe_combine(outputs, routing, shared, y, s);
    return y;
}


void Forward::run_mtp(const MtpCall& call) {
    if (!parameters_.mtp || !kv_.mtp || state_.mtp_tails.data == nullptr) {
        throw std::logic_error("Qwen4Exp forward: the MTP drafter is not loaded");
    }
    if (call.batch <= 0 || call.width <= 0 || call.ids.numel() != call.batch * call.width ||
        call.residuals.ne[1] != call.batch * call.width || (!call.kv_only && call.width != 1)) {
        throw std::invalid_argument("Qwen4Exp forward: MTP call geometry is invalid");
    }
    eager_chunk_ = false;
    work_.reset();
    const std::int32_t W = dim(config_.residual_width()), T = call.batch * call.width;
    Tensor rows = work_.alloc(DType::BF16, {W, T});
    CUDA_CHECK(cudaMemcpyAsync(rows.data, call.residuals.data, rows.bytes(), cudaMemcpyDeviceToDevice, device_.stream));
    mtp_block(call, rows);
}

void Forward::mtp_block(const MtpCall& call, Tensor& rows) {
    const cudaStream_t s = device_.stream;
    const auto& p        = *parameters_.mtp;
    const std::int32_t H = dim(config_.hidden_size), W = dim(config_.residual_width()), S = dim(config_.hc.streams);
    const std::int32_t T = call.batch * call.width;
    try {
        auto scope = work_.scope();
        // Input fusion: R = fc_hidden(per stream of RMSNorm_{S*H}(rows)) + fc_embedding(RMSNorm_H(embed(id))).
        Tensor x0 = work_.alloc(DType::BF16, {H, T});
        ops::embedding(call.ids, parameters_.token_embedding, x0, s);
        Tensor e = work_.alloc(DType::BF16, {H, T});
        ops::rmsnorm(x0, p.embedding_norm, config_.rms_norm_eps, true, e, s);
        Tensor ep = work_.alloc(DType::BF16, {H, T});
        project(e, p.embedding_projection, ep, work_, s);
        Tensor hn = work_.alloc(DType::BF16, {W, T});
        ops::rmsnorm(rows, p.hidden_norm, config_.rms_norm_eps, true, hn, s);
        Tensor residual = work_.alloc(DType::BF16, {W, T});
        {
            Tensor per_stream = residual.view({H, S * T});
            project(hn.view({H, S * T}), p.hidden_projection, per_stream, work_, s);
        }
        ops::hyper_connection_inject(ep, state_.mtp_ones.slice(1, 0, T), residual, s);

        Tensor inject = work_.alloc(DType::FP32, {S, T});
        Tensor xa     = mix(p.attn_hc, residual, &inject);
        const AttentionCall attention_call{.layer        = *kv_.mtp,
                                           .tails        = state_.mtp_tails,
                                           .key_records  = call.key_records,
                                           .update_tails = call.kv_only && call.key_records.data == nullptr,
                                           .kv_only      = call.kv_only,
                                           .positions    = call.positions,
                                           .slots        = call.slots,
                                           .table_rows   = call.table_rows,
                                           .batch        = call.batch,
                                           .width        = call.width};
        Tensor y = attention(p.attention, xa, attention_call);
        if (call.kv_only) { return; }
        ops::hyper_connection_inject(y, inject, residual, s);

        // MoE over the device-resident experts, with the shared expert.
        Tensor xm = mix(p.mlp_hc, residual, &inject);
        const auto& m        = config_.moe;
        const std::int32_t E = dim(m.experts), K = dim(m.top_k), I = dim(m.intermediate);
        Tensor logits = work_.alloc(DType::FP32, {E + 1, T});
        const Tensor* router_rows[] = {&p.router, &p.shared_score};
        ops::projection_fp32(xm, router_rows, logits, s);
        ops::MoeRouting routing{work_.alloc(DType::I32, {K, T}), work_.alloc(DType::FP32, {K, T}),
                                work_.alloc(DType::FP32, {T})};
        ops::moe_route(logits, K, routing, s);
        Tensor outputs = work_.alloc(DType::BF16, {H, K * T});
        const DeviceSpan expert_ws = work_.alloc_bytes(ops::resident_moe_workspace_bytes(K * T, I));
        ops::resident_moe_experts(xm, routing.ids, p.experts_gate_up, p.experts_down, E, I, expert_ws.data,
                                  expert_ws.bytes, outputs, s);
        const std::int32_t SI = dim(m.shared_intermediate);
        Tensor gate_up = work_.alloc(DType::BF16, {2 * SI, T});
        project(xm, p.shared_gate_up, gate_up, work_, s);
        Tensor gate = work_.alloc(DType::BF16, {SI, T});
        Tensor up   = work_.alloc(DType::BF16, {SI, T});
        {
            Tensor* parts[] = {&gate, &up};
            ops::split_rows(gate_up, parts, s);
        }
        Tensor product = work_.alloc(DType::BF16, {SI, T});
        ops::silu_mul(gate, up, product, s);
        Tensor shared = work_.alloc(DType::BF16, {H, T});
        project(product, p.shared_down, shared, work_, s);
        Tensor ym = work_.alloc(DType::BF16, {H, T});
        ops::moe_combine(outputs, routing, shared, ym, s);
        ops::hyper_connection_inject(ym, inject, residual, s);
        CUDA_CHECK(cudaMemcpyAsync(call.residual_out.data, residual.data, residual.bytes(), cudaMemcpyDeviceToDevice, s));

        // The draft token: argmax of the draft head (the proposal head's rows mapped to token ids,
        // or the text head over the public tokens).
        Tensor final_x = mix(p.final_mixer, residual, nullptr);
        const auto& head = parameters_.draft_head;
        const std::int32_t rows_out = head.rows ? head.rows->weight.n : dim(config_.vocab_size);
        Tensor head_logits = work_.alloc(DType::FP32, {rows_out, T});
        if (head.rows) {
            ops::projection_fp32(final_x, head.rows->weight, head_logits, s);
        } else if (parameters_.output_head_q8) {
            ops::projection_fp32(final_x, parameters_.output_head_q8->weight, head_logits, s);
        } else {
            const Tensor* text_head[] = {&parameters_.output_head};
            ops::projection_fp32(final_x, text_head, head_logits, s);
        }
        Tensor narrow = work_.alloc(DType::BF16, {rows_out, T});
        ops::cast_fp32_to_bf16(head_logits, narrow, s);
        Tensor drafts = call.drafts;
        if (head.rows) {
            ops::argmax(narrow, head.token_ids, drafts, rows_out, s);
        } else {
            ops::argmax(narrow, drafts, dim(parameters_.model.resources().public_token_count), s);
        }
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("qwen4_exp/mtp: ") + error.what());
    }
}

std::size_t Forward::mtp_workspace_bytes(const TextConfig& c, std::int32_t kv_columns, std::int32_t draft_columns,
                                         std::int32_t vocabulary_rows, std::int32_t max_context) {
    const std::size_t t = static_cast<std::size_t>(std::max(std::min(kv_columns, kMtpChunkColumns), draft_columns));
    const std::size_t d = static_cast<std::size_t>(draft_columns);
    // One block at t columns (as the text forward sizes it), the input fusion's and the copied
    // residuals; the draft head's logits and the resident experts only at full steps, which run
    // one column per sequence.
    return workspace_bytes(c, static_cast<std::int32_t>(t), max_context) +
           std::size_t(c.residual_width()) * t * 2 * 4 + std::size_t(c.hidden_size) * t * 2 * 4 +
           std::size_t(vocabulary_rows) * d * 6 +
           ops::resident_moe_workspace_bytes(dim(c.moe.top_k * d), dim(c.moe.intermediate));
}

} // namespace ninfer::models::qwen4_exp::execution
