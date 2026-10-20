#include "models/qwen4_exp/execution/forward.h"

#include "infernix/ops/causal_conv1d_silu.h"
#include "infernix/ops/embedding.h"
#include "infernix/ops/gated_delta_net.h"
#include "infernix/ops/gated_rmsnorm.h"
#include "infernix/ops/gdn_gating.h"
#include "infernix/ops/ggml_linear.h"
#include "infernix/ops/ggml_rows.h"
#include "infernix/ops/gdn_input_proj.h"
#include "infernix/ops/hyper_connection.h"
#include "infernix/ops/kv_cache_append.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/linear_swiglu.h"
#include "infernix/ops/ple.h"
#include "infernix/ops/projection_fp32.h"
#include "infernix/ops/argmax.h"
#include "infernix/ops/top_logprobs.h"
#include "ops/offloaded_sparse_moe/cpu/miss_request.h"
#include "infernix/ops/cast.h"
#include "infernix/ops/resident_moe.h"
#include "infernix/ops/rmsnorm.h"
#include "infernix/ops/rope.h"
#include "infernix/ops/rows.h"
#include "infernix/ops/scatter.h"
#include "infernix/ops/sigmoid_mul.h"
#include "infernix/ops/silu_mul.h"
#include "infernix/ops/speculative_round.h"

#include <algorithm>
#include <array>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::models::qwen4_exp::execution {
namespace {

std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }

ops::offloaded_moe::ExpertActivation expert_activation(const TextConfig& c) {
    return c.moe.a16_experts ? ops::offloaded_moe::ExpertActivation::kA16 : ops::offloaded_moe::ExpertActivation::kA4;
}

// `routes` carries the selected fast paths' linear route flags (ForwardFastPaths::dense_routes) to the
// projections they serve; a problem without a route for a flag runs its exact route. A GGUF import's
// projection runs the ggml route (ggml-bf16-prefill's BF16 tiles at prefill widths, else exact).
void project(const Tensor& x, const LinearParameters& p, Tensor& out, WorkspaceArena& work, cudaStream_t s,
             ops::LinearRoutes routes = ops::LinearRoutes::Exact) {
    if (p.is_ggml()) {
        ops::ggml_linear(x, p.ggml, out, p.input_perm, p.perm_block, work, s, routes);
        return;
    }
    ops::linear(x, p.weight, out, p.policy, work, s, routes);
}

// Token rows: BF16, or a GGUF import's IQ4_XS rows decoded exactly.
void embed_tokens(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t s) {
    if (table.layout == QuantLayout::GgmlRows) {
        ops::ggml_embedding(ids, table, out, s);
    } else {
        ops::embedding(ids, table, out, s);
    }
}

// FP32 logits of the text head (BF16, Q8 or a GGUF import's ggml rows) for the columns of x.
void text_head_logits(const Parameters& parameters, const Tensor& x, Tensor& logits, WorkspaceArena& work, cudaStream_t s) {
    if (parameters.output_head_q8 && parameters.output_head_q8->is_ggml()) {
        ops::ggml_linear(x, parameters.output_head_q8->ggml, logits, nullptr, 0, work, s, ops::LinearRoutes::Exact);
    } else if (parameters.output_head_q8) {
        ops::projection_fp32(x, parameters.output_head_q8->weight, logits, s);
    } else {
        const Tensor* head[] = {&parameters.output_head};
        ops::projection_fp32(x, head, logits, s);
    }
}

// The shared expert's SiLU(gate) * up: one fused call for Q8 weights (gate and up stay FP32); BF16
// weights (recipe A) project, split and multiply.
void shared_swiglu(const Tensor& x, const LinearParameters& gate_up, Tensor& out, WorkspaceArena& work,
                   cudaStream_t s, ops::LinearRoutes routes = ops::LinearRoutes::Exact) {
    // Q8 (Dense8) at every width; BF16 (the bit-exact artifact) on its register-streamed route at decode
    // and verification widths, the tuned GEMM then SwiGLU beyond.
    if (gate_up.weight.qtype == QType::Q8_G32_FP16 ||
        (gate_up.weight.qtype == QType::BF16 && x.ne[1] <= 16)) {
        ops::linear_swiglu(x, gate_up.weight, out, gate_up.policy, work, s);
        return;
    }
    auto scope = work.scope();
    const std::int32_t I = out.ne[0], T = out.ne[1];
    if (gate_up.is_ggml() && gate_up.input_perm == nullptr) {
        // A GGUF import: the gate and up rows straight into their own tensors (no split copy).
        Tensor gate = work.alloc(DType::BF16, {I, T});
        Tensor up   = work.alloc(DType::BF16, {I, T});
        const std::int32_t bounds[] = {0, I, 2 * I};
        Tensor* parts[]             = {&gate, &up};
        ops::ggml_linear(x, ops::ggml_split_segments(gate_up.ggml, x.ne[0], bounds), parts, nullptr, 0, work, s, routes);
        ops::silu_mul(gate, up, out, s);
        return;
    }
    Tensor projected = work.alloc(DType::BF16, {2 * I, T});
    project(x, gate_up, projected, work, s, routes);
    Tensor gate = work.alloc(DType::BF16, {I, T});
    Tensor up   = work.alloc(DType::BF16, {I, T});
    {
        Tensor* parts[] = {&gate, &up};
        ops::split_rows(projected, parts, s);
    }
    ops::silu_mul(gate, up, out, s);
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
            .eps            = c.rms_norm_eps,
            .yarn_factor    = c.rope.yarn_factor,
            .original_positions = dim(c.max_position_embeddings)};
}

Forward::Forward(const Parameters& parameters, DeviceContext& device, WorkspaceArena& work,
                 ForwardState state, ForwardKV kv, ForwardExperts experts, std::int32_t max_context,
                 ForwardFastPaths fast_paths)
    : parameters_(parameters), config_(parameters.model.config().text), device_(device), work_(work),
      state_(std::move(state)), kv_(std::move(kv)), experts_(std::move(experts)), max_context_(max_context),
      fast_(fast_paths),
      rope_(ops::prepare_rope(dim(config_.rope.rotary_dim), config_.rope.theta,
                              {config_.rope.yarn_factor, config_.max_position_embeddings})) {
    if (state_.gdn == nullptr || state_.gdn->layer_count() != config_.gdn_layers ||
        state_.qsa_tails.size() != config_.attention_layers || kv_.layers.size() != config_.attention_layers ||
        experts_.frames.size() != config_.num_hidden_layers) {
        throw std::invalid_argument("Qwen4Exp forward: state, KV or expert residency is incomplete");
    }
    if (experts_.split != nullptr) {
        const auto E   = static_cast<std::size_t>(config_.moe.experts);
        split_offsets_ = PinnedHostBuffer((E + 1) * sizeof(std::int32_t));
        split_columns_.assign(E, 0);
        split_cpu_.assign(E, 0);
        CUDA_CHECK(cudaEventCreateWithFlags(&split_ready_, cudaEventDisableTiming | cudaEventBlockingSync));
    }
    if (const auto& dflash2 = parameters_.model.config().dflash2) { draft_taps_ = dflash2->target_layer_ids; }
}

Forward::~Forward() {
    if (split_ready_ != nullptr) { (void)cudaEventDestroy(split_ready_); }
}

ops::LinearRoutes Forward::dense_routes() const noexcept {
    if (eager_chunk_) { return fast_.dense_routes; }
    return static_cast<ops::LinearRoutes>(static_cast<std::uint8_t>(fast_.dense_routes) &
                                          ~static_cast<std::uint8_t>(ops::LinearRoutes::GgmlBf16Wide));
}

std::size_t Forward::workspace_bytes(const TextConfig& c, std::int32_t columns, std::int32_t max_context,
                                     KvCacheStorage storage, ops::offloaded_moe::WideA16 wide_a16) {
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
    bytes += ops::qsa_attention_workspace_bytes(qsa_geometry(c), columns, max_context, storage);
    bytes += (c.moe.experts + 1) * t * 4 + k * t * 8 + t * 4;           // FP32 router logits, routing
    bytes += ops::moe_dispatch_bytes(dim(c.moe.experts), dim(k * t));
    bytes += h * k * t * bf + ops::moe_experts_workspace_bytes(dim(c.moe.experts), dim(k * t), std::max(columns, 1),
                                                               expert_activation(c), wide_a16);
    bytes += 2 * std::size_t(c.moe.shared_intermediate) * t * bf * 3;
    // Linear and GDN scratch (chunked recurrence), plus alignment slack per allocation.
    bytes += ops::gated_delta_net_workspace_capacity_bytes(dim(c.gdn.key_heads), dim(c.gdn.value_heads), 1,
                                                           std::max(columns, 1));
    // A GGUF import's projections encode their input columns (K up to the GDN/QSA value width).
    if (c.ggml_weights) {
        const std::int32_t k = dim(std::max<std::size_t>({h, std::size_t(c.gdn.value_width()), std::size_t(c.attention.query_width())}));
        bytes += ops::ggml_linear_workspace_bytes(k, std::max(columns, 1));
    }
    bytes += 64ULL << 20;
    return bytes;
}

void Forward::begin_call(const ForwardBatch& batch, const Tensor* logits) {
    const std::int32_t T = batch.ids.ne[0];
    if (batch.verify != nullptr && (batch.width < 2 || batch.verify->gdn.spec.width != batch.width ||
                                    batch.verify->gdn.spec.record_capacity < batch.batch)) {
        throw std::invalid_argument("Qwen4Exp forward: verification records do not match the call");
    }
    if (batch.batch <= 0 || batch.width <= 0 || batch.batch * batch.width != T ||
        (logits != nullptr && (logits->dtype != DType::FP32 || logits->ne[1] != batch.logit_columns.ne[0] ||
                               logits->ne[0] != dim(config_.vocab_size)))) {
        throw std::invalid_argument("Qwen4Exp forward: batch geometry is invalid");
    }
    if (batch.rope_positions.ne[0] != T || batch.rope_positions.ne[1] != 3 ||
        batch.block_start_rope.ne[0] != batch.batch || batch.block_start_rope.ne[1] != 3) {
        throw std::invalid_argument("Qwen4Exp forward: RoPE positions must be [T, 3] and block starts [batch, 3]");
    }
    for (const auto& events : batch.layer_waits) {
        if (!events.empty() && events.size() != config_.num_hidden_layers + 1U) {
            throw std::invalid_argument("Qwen4Exp forward: layer waits do not match the layers");
        }
    }
    // Prefill chunks and forced tokens run eagerly; decode and verification rounds may be captured.
    eager_chunk_ = batch.verify == nullptr && batch.width > 1;
    work_.reset();
}

void Forward::wait_layer(const ForwardBatch& batch, std::size_t entry) {
    for (const auto& events : batch.layer_waits) {
        if (!events.empty()) { CUDA_CHECK(cudaStreamWaitEvent(device_.stream, events[entry], 0)); }
    }
}

void Forward::run(const ForwardBatch& batch, Tensor& logits, const ForwardTap* tap) {
    begin_call(batch, &logits);
    Tensor residual = work_.alloc(DType::BF16, {dim(config_.residual_width()), batch.ids.ne[0]});
    embed(batch, residual);
    for (std::uint32_t l = 0; l < config_.num_hidden_layers; ++l) {
        layer(l, batch, residual, tap);
    }
    finish(batch, residual, &logits);
}

void Forward::embed(const ForwardBatch& batch, Tensor& residual) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = batch.ids.ne[0], H = dim(config_.hidden_size);
    auto scope = work_.scope();
    Tensor x0  = work_.alloc(DType::BF16, {H, T});
    embed_tokens(batch.ids, parameters_.token_embedding, x0, s);
    if (batch.vision != nullptr && batch.vision->columns.data != nullptr) {
        ops::scatter(batch.vision->embeddings, batch.vision->columns, x0, s);
    }
    ops::hyper_connection_expand(x0, dim(config_.hc.streams), residual, s);
}

void Forward::layer(std::uint32_t layer, const ForwardBatch& batch, Tensor& residual, const ForwardTap* tap) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = batch.ids.ne[0];
    const auto& block    = parameters_.layers[layer];
    wait_layer(batch, layer);
    try {
        auto scope = work_.scope();
        if (block.ple) { ple(*block.ple, residual, batch); }
        Tensor inject = work_.alloc(DType::FP32, {dim(config_.hc.streams), T});
        Tensor xa     = mix(block.attn_hc, residual, &inject);
        Tensor y      = mixer(layer, batch, xa);
        ops::hyper_connection_inject(y, inject, residual, s);
        Tensor xm = mix(block.mlp_hc, residual, &inject);
        // The combine injects its output directly unless a tap records it.
        const bool fused_inject = tap == nullptr || tap->moe_outputs == nullptr;
        Tensor ym = moe(block.moe, xm, layer,
                        tap != nullptr && tap->routes != nullptr ? &tap->routes->at(layer) : nullptr, &batch, -1,
                        fused_inject ? &inject : nullptr, fused_inject ? &residual : nullptr);
        if (!fused_inject) { ops::hyper_connection_inject(ym, inject, residual, s); }
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
    if (batch.draft_features.data != nullptr) {
        const std::int32_t n = batch.draft_features.ne[1], T = residual.ne[1];
        if (n > T) { throw std::invalid_argument("Qwen4Exp forward: DFlash2 features exceed the call"); }
        capture_draft_features(layer, residual.slice(1, T - n, n), batch.draft_features);
    }
    if (tap != nullptr && tap->residuals != nullptr) {
        Tensor& copy = tap->residuals->at(layer);
        if (cudaMemcpyAsync(copy.data, residual.data, residual.bytes(), cudaMemcpyDeviceToDevice, s) !=
            cudaSuccess) {
            throw std::runtime_error("Qwen4Exp forward: residual tap copy failed");
        }
    }
}

void Forward::capture_draft_features(std::uint32_t layer, const Tensor& residual, const Tensor& features) {
    const auto& taps = draft_taps_;
    const auto it    = std::find(taps.begin(), taps.end(), layer);
    if (it == taps.end()) { return; }
    const std::int32_t H = dim(config_.hidden_size);
    if (features.ne[1] != residual.ne[1] || features.ne[0] != H * static_cast<std::int32_t>(taps.size())) {
        throw std::invalid_argument("Qwen4Exp forward: DFlash2 features do not match the residual");
    }
    Tensor out = features.slice(0, static_cast<std::int32_t>(it - taps.begin()) * H, H);
    ops::hyper_connection_stream_mean(residual, dim(config_.hc.streams), out, device_.stream);
}

Tensor Forward::mixer(std::uint32_t layer, const ForwardBatch& batch, const Tensor& xa) {
    const auto& block           = parameters_.layers[layer];
    const std::int32_t T        = batch.ids.ne[0];
    const std::uint32_t compact = config_.compact_layer_indices[layer];
    if (config_.layer_types[layer] == MixerKind::Gdn) {
        return gdn(std::get<GdnParameters>(block.mixer), xa, compact, batch);
    }
    // A verification call records the raw index keys for the tail commit.
    const AttentionCall call{.layer       = kv_.layers.at(compact),
                             .tails       = state_.qsa_tails.at(compact),
                             .key_records = batch.verify != nullptr
                                                ? batch.verify->qsa_keys.slice(3, static_cast<std::int32_t>(compact), 1)
                                                      .view({dim(config_.qsa.index_head_dim), T})
                                                : Tensor{},
                             .update_tails     = batch.verify == nullptr,
                             .positions        = batch.positions,
                             .rope_positions   = batch.rope_positions,
                             .block_start_rope = batch.block_start_rope,
                             .slots            = batch.slots,
                             .table_rows       = batch.table_rows,
                             .batch            = batch.batch,
                             .width            = batch.width};
    return attention(std::get<AttentionParameters>(block.mixer), xa, call);
}

void Forward::layer_mixer(std::uint32_t layer, const ForwardBatch& batch, Tensor& residual, Tensor& xm, Tensor& inject) {
    const cudaStream_t s = device_.stream;
    const auto& block    = parameters_.layers[layer];
    wait_layer(batch, layer);
    try {
        auto scope = work_.scope();
        if (block.ple) { ple(*block.ple, residual, batch); }
        Tensor xa = mix(block.attn_hc, residual, &inject);
        Tensor y  = mixer(layer, batch, xa);
        ops::hyper_connection_inject(y, inject, residual, s);
        (void)mix(block.mlp_hc, residual, &inject, &xm);
    } catch (const std::exception& error) {
        throw std::runtime_error("qwen4_exp/layers/" + std::to_string(layer) + ": " + error.what());
    }
}

void Forward::layer_moe(std::uint32_t layer, const ForwardBatch& batch, const Tensor& xm, const Tensor& inject,
                        Tensor& residual, std::int32_t route_columns) {
    const cudaStream_t s = device_.stream;
    try {
        auto scope = work_.scope();
        Tensor ym  = moe(parameters_.layers[layer].moe, xm, layer, nullptr, &batch, route_columns);
        ops::hyper_connection_inject(ym, inject, residual, s);
    } catch (const std::exception& error) {
        throw std::runtime_error("qwen4_exp/layers/" + std::to_string(layer) + ": " + error.what());
    }
}

std::size_t Forward::moe_workspace_bytes(const TextConfig& c, std::int32_t columns,
                                         ops::offloaded_moe::WideA16 wide_a16) {
    const std::size_t t = static_cast<std::size_t>(columns);
    const std::size_t h = c.hidden_size, k = c.moe.top_k;
    std::size_t bytes = (c.moe.experts + 1) * t * 4 + k * t * 8 + t * 4; // router logits, routing
    bytes += ops::moe_dispatch_bytes(dim(c.moe.experts), dim(k * t));
    bytes += h * k * t * 2 + ops::moe_experts_workspace_bytes(dim(c.moe.experts), dim(k * t), std::max(columns, 1),
                                                             expert_activation(c), wide_a16);
    bytes += 2 * std::size_t(c.moe.shared_intermediate) * t * 2 * 3 + h * t * 2 * 2; // shared expert, y
    if (c.ggml_weights) { // the shared expert's projections encode their inputs
        bytes += ops::ggml_linear_workspace_bytes(dim(h), std::max(columns, 1)) +
                 ops::ggml_linear_workspace_bytes(dim(c.moe.shared_intermediate), std::max(columns, 1));
    }
    return bytes + (16ULL << 20); // alignment slack
}

void Forward::finish(const ForwardBatch& batch, Tensor& residual, Tensor* logits) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = batch.ids.ne[0], H = dim(config_.hidden_size), W = dim(config_.residual_width());
    if (batch.residual_out.data != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(batch.residual_out.data, residual.data, residual.bytes(), cudaMemcpyDeviceToDevice, s));
    }
    wait_layer(batch, config_.num_hidden_layers);
    if (batch.mtp_chunk != nullptr) {
        // The chunk's MTP cells from its live residuals, then its last residual becomes pending.
        const MtpChunk& chunk = *batch.mtp_chunk;
        const std::int32_t cells = (chunk.prepend ? 1 : 0) + chunk.columns;
        const std::size_t column = static_cast<std::size_t>(W) * 2;
        if (chunk.columns < 0 || chunk.columns > T || batch.batch != 1 ||
            (cells > 0 && (chunk.ids.ne[0] != cells || chunk.positions.ne[0] != cells ||
                           chunk.rope_positions == nullptr || chunk.block_start_rope == nullptr))) {
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
            const std::int32_t k = begin / kMtpChunkColumns;
            MtpCall call{.residuals        = rows,
                         .ids              = chunk.ids.slice(0, begin, n),
                         .positions        = chunk.positions.slice(0, begin, n),
                         .rope_positions   = Tensor(chunk.rope_positions + 3 * static_cast<std::size_t>(begin),
                                                    DType::I32, {n, 3}),
                         .block_start_rope = Tensor(chunk.block_start_rope + 3 * static_cast<std::size_t>(k),
                                                    DType::I32, {1, 3}),
                         .slots            = batch.slots,
                         .table_rows       = batch.table_rows,
                         .batch            = 1,
                         .width            = n,
                         .kv_only          = true,
                         .vision           = chunk.vision != nullptr ? &chunk.vision[k] : nullptr};
            mtp_block(call, rows);
        }
        CUDA_CHECK(cudaMemcpyAsync(chunk.saved.data, static_cast<const std::byte*>(residual.data) + column * (T - 1),
                                   column, cudaMemcpyDeviceToDevice, s));
    }
    if (logits == nullptr) { return; }
    auto scope    = work_.scope();
    Tensor final_x = mix(parameters_.final_mixer, residual, nullptr);
    Tensor last    = work_.alloc(DType::BF16, {H, batch.logit_columns.ne[0]});
    ops::gather_columns(final_x, batch.logit_columns, last, s);
    text_head_logits(parameters_, last, *logits, work_, s);
}

void Forward::head_logits(const Tensor& residuals, Tensor& logits) {
    if (residuals.dtype != DType::BF16 || residuals.ne[0] != dim(config_.residual_width()) ||
        logits.dtype != DType::FP32 || logits.ne[0] != dim(config_.vocab_size) ||
        logits.ne[1] != residuals.ne[1]) {
        throw std::invalid_argument(
            "Qwen4Exp forward: head logits need BF16 [S*H, n] residuals and FP32 [V, n] logits");
    }
    work_.reset();
    auto scope           = work_.scope();
    const Tensor final_x = mix(parameters_.final_mixer, residuals, nullptr);
    text_head_logits(parameters_, final_x, logits, work_, device_.stream);
}

Tensor Forward::mix(const HyperConnectionParameters& p, const Tensor& residual, Tensor* inject, Tensor* into) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = residual.ne[1], H = dim(config_.hidden_size);
    const std::int32_t S = dim(config_.hc.streams), rank = dim(config_.hc.rank);
    Tensor x = into != nullptr ? *into : work_.alloc(DType::BF16, {H, T});
    ops::hyper_connection_mix(residual, p.norm, p.down.weight, p.up.weight, p.down.policy, S, rank, config_.rms_norm_eps,
                              x, p.combine ? inject : nullptr, work_, s);
    return x;
}

void Forward::ple(const PleParameters& p, Tensor& residual, const ForwardBatch& batch) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = residual.ne[1], W = residual.ne[0], H = dim(config_.hidden_size);
    auto scope = work_.scope();
    if (const NgramRowGate* gate = batch.ngram_gate) {
        // ple_embed is a plain launch (no PDL), so stream order holds it behind the copy.
        upload_pinned_when(batch.ngram_rows.data, gate->pinned_rows, batch.ngram_rows.bytes(), gate->ready,
                           gate->expected, gate->wait_stats, gate->wait_row, s);
    }
    const NgramRowFormat format = batch.ngram_format.value_or(config_.ple.table.format);
    Tensor e;
    if (format == NgramRowFormat::Bf16) {
        // BF16 rows are the embedding itself: rows [W, heads] of a column are e[h * W + j] (ple_embed's
        // order), so the rows are read in place.
        if (batch.ngram_rows.ne[0] != 2 * dim(config_.ple.head_width())) {
            throw std::invalid_argument("Qwen4Exp forward: BF16 n-gram rows must be 2 x head width bytes");
        }
        e = Tensor(batch.ngram_rows.data, DType::BF16, {dim(config_.ple.embed_dim), T});
    } else {
        e = work_.alloc(DType::BF16, {dim(config_.ple.embed_dim), T});
        if (format == NgramRowFormat::Iq4Nl) {
            ops::ple_embed_iq4_nl(batch.ngram_rows, e, s);
        } else {
            const Tensor& scale = batch.ngram_scale.data != nullptr ? batch.ngram_scale : p.ngram_scale;
            if (scale.data == nullptr) {
                throw std::invalid_argument("Qwen4Exp forward: FP8 n-gram rows need their table's scale");
            }
            ops::ple_embed(batch.ngram_rows, scale, e, s);
        }
    }
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
    // A verification call projects its convolution inputs straight into the replay records.
    GdnReplayRecordLayer records;
    if (batch.verify != nullptr) { records = batch.verify->gdn.layer(static_cast<std::int32_t>(index), batch.batch); }
    // gdn-conv-fused: a decode call's Q8 projection runs the causal convolution in its epilogue, from
    // the FP32 projection rather than its BF16 rounding. Verification and prefill stay composed.
    const bool decode     = batch.verify == nullptr && batch.width <= 1;
    const bool fused_conv = fast_.gdn_conv_fused && decode && p.projection.weight.qtype == QType::Q8_G32_FP16;
    Tensor qkv;
    if (!fused_conv) { qkv = batch.verify != nullptr ? records.conv.view({C, T}) : work_.alloc(DType::BF16, {C, T}); }
    Tensor z  = work_.alloc(DType::BF16, {VW, T});
    Tensor ab = work_.alloc(DType::BF16, {2 * NV, T}); // the control projection: a's rows, then b's
    Tensor q, k, v;
    const bool ggml = p.projection.is_ggml() && p.projection.input_perm == nullptr;
    if (fused_conv) {
        q = work_.alloc(DType::BF16, {KW, T});
        k = work_.alloc(DType::BF16, {KW, T});
        v = work_.alloc(DType::BF16, {VW, T});
        auto layer          = state_.gdn->layer_view(index);
        const Tensor x_rows = x.view({H, 1, T});
        Tensor q_rows = q.view({KW, 1, T}), k_rows = k.view({KW, 1, T}), v_rows = v.view({VW, 1, T}),
               z_rows = z.view({VW, 1, T});
        ops::gdn_input_proj_conv_snapshot(x_rows, p.projection.weight, p.convolution, layer.conv, batch.slots,
                                          batch.slots, q_rows, k_rows, v_rows, z_rows, dense_routes(), s);
        project(x, p.control, ab, work_, s);
    } else if (ggml && p.control.is_ggml() && p.control.input_perm == nullptr) {
        // A GGUF import's two projections of x in one call (one encoding of x), each part written to
        // its own tensor.
        std::vector<ops::GgmlSegment> segments = p.projection.ggml;
        for (ops::GgmlSegment segment : p.control.ggml) {
            segment.out_offset += C + VW;
            segments.push_back(segment);
        }
        const std::int32_t bounds[] = {0, C, C + VW, C + VW + 2 * NV};
        Tensor* parts[]             = {&qkv, &z, &ab};
        ops::ggml_linear(x, ops::ggml_split_segments(segments, x.ne[0], bounds), parts, nullptr, 0, work_, s,
                         dense_routes());
    } else {
        if (ggml) {
            // A GGUF import's projection (its control projection BF16): qkv and z in their own tensors.
            const std::int32_t bounds[] = {0, C, C + VW};
            Tensor* parts[]             = {&qkv, &z};
            ops::ggml_linear(x, ops::ggml_split_segments(p.projection.ggml, x.ne[0], bounds), parts, nullptr, 0, work_, s,
                             dense_routes());
        } else if (p.projection.weight.qtype == QType::Q8_G32_FP16) {
            // Q8 (Dense8): the projection writes q/k/v and z to their destinations itself.
            ops::gdn_input_proj(x, p.projection.weight, qkv, z, s, dense_routes());
        } else {
            Tensor projected = work_.alloc(DType::BF16, {C + VW, T});
            project(x, p.projection, projected, work_, s, dense_routes());
            Tensor* parts[] = {&qkv, &z};
            ops::split_rows(projected, parts, s);
        }
        project(x, p.control, ab, work_, s);
    }
    // Verification and prefill chunks evaluate the gates once for their recurrence; decode evaluates
    // them inside the recurrent update, except a GGUF import's (its third input holds A itself), whose
    // decode takes them from gdn_gating_direct too.
    Tensor gate, beta;
    if (batch.verify != nullptr || batch.width > 1 || g.a_direct) {
        gate = work_.alloc(DType::FP32, {NV, T});
        beta = work_.alloc(DType::FP32, {NV, T});
        if (g.a_direct) {
            ops::gdn_gating_direct(ab.slice(0, 0, NV), ab.slice(0, NV, NV), p.a_log, p.dt_bias, gate, beta, s);
        } else {
            ops::gdn_gating(ab.slice(0, 0, NV), ab.slice(0, NV, NV), p.a_log, p.dt_bias, gate, beta, s);
        }
    }

    if (!fused_conv) {
        q = work_.alloc(DType::BF16, {KW, T});
        k = work_.alloc(DType::BF16, {KW, T});
        v = work_.alloc(DType::BF16, {VW, T});
    }
    Tensor o = work_.alloc(DType::BF16, {DV, NV, T});
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(DK)));
    if (batch.verify != nullptr) {
        // Speculative verification: the convolution and recurrence run from the live states
        // without changing them, and the recurrence's inputs are recorded for the commit fold.
        const std::int32_t Wd = batch.width, B = batch.batch;
        auto layer = state_.gdn->layer_view(index);
        ops::causal_conv1d_silu_from_states_split(qkv.view({C, Wd, B}), p.convolution, layer.conv,
                                                  batch.slots, q, k, v, s);
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
                             /*normalize_qk=*/true, work_, state, state, o, device_.execution_view(),
                             fast_.gdn_arithmetic);
    } else {
        // One position per sequence (decode): the recurrent update with device-chosen slots.
        auto layer = state_.gdn->layer_view(index);
        if (!fused_conv) {
            ops::causal_conv1d_silu_snapshot_split(qkv.view({C, 1, T}), p.convolution, layer.conv, Tensor{},
                                                   batch.slots, batch.slots, q, k, v, s);
        }
        Tensor o_batch = o.view({DV, NV, 1, T});
        if (g.a_direct) {
            ops::gated_delta_net_batch_update(q.view({DK, NK, 1, T}), k.view({DK, NK, 1, T}), v.view({DV, NV, 1, T}),
                                              gate.view({NV, 1, T}), beta.view({NV, 1, T}), scale, /*normalize_qk=*/true,
                                              layer.recurrent, batch.slots, batch.slots, o_batch, s);
        } else {
            // The gates are evaluated inside the update from the control projection (no gdn_gating kernel).
            ops::gated_delta_net_batch_update_gating(q.view({DK, NK, 1, T}), k.view({DK, NK, 1, T}),
                                                     v.view({DV, NV, 1, T}), ab.slice(0, 0, NV), ab.slice(0, NV, NV),
                                                     p.a_log, p.dt_bias, scale, /*normalize_qk=*/true, layer.recurrent,
                                                     batch.slots, batch.slots, o_batch, s);
        }
    }
    Tensor normed      = work_.alloc(DType::BF16, {DV, NV * T});
    const Tensor rows  = o.view({DV, NV * T});
    const Tensor gates = z.view({DV, NV * T});
    ops::gated_rmsnorm(rows, p.norm, gates,
                       g.output_gate == GateActivation::Sigmoid ? ops::RmsGate::Sigmoid : ops::RmsGate::Silu,
                       config_.rms_norm_eps, normed, device_.execution_view().on_stream(s));
    Tensor y = work_.alloc(DType::BF16, {H, T});
    project(normed.view({VW, T}), p.output, y, work_, s, dense_routes());
    return y;
}

Tensor Forward::attention(const AttentionParameters& p, const Tensor& x, const AttentionCall& call) {
    const cudaStream_t s = device_.stream;
    const auto& a        = config_.attention;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size);
    const std::int32_t QW = dim(a.query_width()), KW = dim(a.key_width()), D = dim(a.head_dim);
    const std::int32_t IH = dim(config_.qsa.index_heads), ID = dim(config_.qsa.index_head_dim);
    const std::int32_t index_rows = IH * ID + ID;
    Tensor q = work_.alloc(DType::BF16, {D, dim(a.heads), T});
    Tensor gate = work_.alloc(DType::BF16, {QW, T});
    Tensor k = work_.alloc(DType::BF16, {D, dim(a.kv_heads), T});
    Tensor v = work_.alloc(DType::BF16, {D, dim(a.kv_heads), T});
    Tensor iq = work_.alloc(DType::BF16, {ID, IH, T});
    Tensor ik = call.key_records.data != nullptr ? call.key_records.view({ID, T}) : work_.alloc(DType::BF16, {ID, T});
    const bool ggml = p.projection.is_ggml() && p.projection.input_perm == nullptr &&
                      (!p.index_projection || (p.index_projection->is_ggml() && p.index_projection->input_perm == nullptr));
    if (ggml) {
        // A GGUF import: the projection and the index projection of x in one call (one encoding of
        // x), each part written to its own tensor.
        std::vector<ops::GgmlSegment> segments = p.projection.ggml;
        if (p.index_projection) {
            for (ops::GgmlSegment segment : p.index_projection->ggml) {
                segment.out_offset += 2 * QW + 2 * KW;
                segments.push_back(segment);
            }
        }
        Tensor qf = q.view({QW, T}), kf = k.view({KW, T}), vf = v.view({KW, T}), iqf = iq.view({IH * ID, T});
        const std::int32_t bounds[] = {0, QW, 2 * QW, 2 * QW + KW, 2 * QW + 2 * KW, 2 * QW + 2 * KW + IH * ID,
                                       2 * QW + 2 * KW + index_rows};
        Tensor* parts[] = {&qf, &gate, &kf, &vf, &iqf, &ik};
        ops::ggml_linear(x, ops::ggml_split_segments(segments, x.ne[0], bounds), parts, nullptr, 0, work_, s,
                         dense_routes());
    } else {
        Tensor projected = work_.alloc(DType::BF16, {2 * QW + 2 * KW + (p.index_projection ? 0 : index_rows), T});
        project(x, p.projection, projected, work_, s, dense_routes());
        Tensor qf = q.view({QW, T}), kf = k.view({KW, T}), vf = v.view({KW, T}), iqf = iq.view({IH * ID, T});
        if (p.index_projection) {
            Tensor indexed = work_.alloc(DType::BF16, {index_rows, T});
            project(x, *p.index_projection, indexed, work_, s);
            Tensor* parts[]       = {&qf, &gate, &kf, &vf};
            Tensor* index_parts[] = {&iqf, &ik};
            ops::split_rows(projected, parts, s);
            ops::split_rows(indexed, index_parts, s);
        } else {
            Tensor* parts[] = {&qf, &gate, &kf, &vf, &iqf, &ik};
            ops::split_rows(projected, parts, s);
        }
    }
    // The vector-quantized storages' exact window is addressed by the sequences' state slots.
    ops::QsaKVLayer layer = call.layer;
    if (layer.kv.window.present()) { layer.kv.window.slots = call.slots; }
    const bool vq = layer.kv.storage == KvCacheStorage::Vq2 || layer.kv.storage == KvCacheStorage::Q4KeyVq2Value;
    const std::int32_t KVH = dim(a.kv_heads);
    const ops::QsaGeometry geometry = qsa_geometry(config_);
    const ops::QsaBatch qsa_batch{.block_tables     = kv_.block_tables,
                                  .table_rows       = call.table_rows,
                                  .positions        = call.positions,
                                  .tail_slots       = call.slots,
                                  .rope_positions   = call.rope_positions,
                                  .block_start_rope = call.block_start_rope,
                                  .batch            = call.batch,
                                  .width            = call.width,
                                  .update_tails     = call.update_tails};
    // qsa-prepare-fused: every call (decode, verification, prefill and MTP chunks alike, so a position's
    // stored K and pooled keys do not depend on which route wrote them, as Replay and prefix-cache
    // equivalence require) normalizes and rotates Q, K, the index query and the pooled keys and appends
    // K/V and the tails in one kernel, FP32 from the projection rows into the cache codec. Q and the
    // index query are prepared in place.
    Tensor qn = q;
    ops::QsaAppend append{};
    if (fast_.qsa_prepare_fused && ops::qsa_prepare_supports(layer.kv.storage)) {
        const ops::QsaPrepareInputs inputs{.q = q, .k = k, .v = v, .index_q = iq, .index_k = ik};
        const ops::QsaPrepareNorms norms{.query       = p.query_norm,
                                         .key         = p.key_norm,
                                         .index_query = p.index_query_norm,
                                         .index_key   = p.index_key_norm};
        Tensor q_out  = call.kv_only ? Tensor{} : q;
        Tensor iq_out = call.kv_only ? Tensor{} : iq;
        ops::qsa_prepare(inputs, norms, call.tails, layer, qsa_batch, geometry, q_out, iq_out, s);
        if (call.kv_only) { return Tensor{}; }
    } else {
        qn        = work_.alloc(DType::BF16, {D, dim(a.heads), T});
        Tensor kn = work_.alloc(DType::BF16, {D, dim(a.kv_heads), T});
        {
            Tensor qn_rows = qn.view({D, dim(a.heads) * T});
            Tensor kn_rows = kn.view({D, dim(a.kv_heads) * T});
            ops::rmsnorm(q.view({D, dim(a.heads) * T}), p.query_norm, config_.rms_norm_eps, true, qn_rows, s);
            ops::rmsnorm(k.view({D, dim(a.kv_heads) * T}), p.key_norm, config_.rms_norm_eps, true, kn_rows, s);
        }
        ops::rope(call.rope_positions, rope_, qn, kn, device_.execution_view().on_stream(s));

        // Append K/V of every sequence through its device-chosen table row, then the pooled index
        // keys. qsa_attention appends the vector-quantized storages' K/V itself (a wide call stages
        // its window rows); a K/V-only call appends here.
        append = ops::QsaAppend{.k = kn.view({D, KVH, call.width, call.batch}),
                                .v = v.view({D, KVH, call.width, call.batch})};
        if (!vq || call.kv_only) {
            const PagedKVBatchLayerView view{.k_pages       = layer.kv.k_pages,
                                             .v_pages       = layer.kv.v_pages,
                                             .k_scale_pages = layer.kv.k_scale_pages,
                                             .v_scale_pages = layer.kv.v_scale_pages,
                                             .block_tables  = kv_.block_tables,
                                             .head_dim      = layer.kv.head_dim,
                                             .num_kv_heads  = layer.kv.num_kv_heads,
                                             .storage       = layer.kv.storage,
                                             .window        = layer.kv.window};
            ops::kv_cache_append_batch(append.k, append.v, call.positions.view({call.width, call.batch}),
                                       call.table_rows, view, s);
        }
        ops::qsa_pool_keys(ik, p.index_key_norm, call.tails, layer, qsa_batch, geometry, s);
        if (call.kv_only) { return Tensor{}; }
        ops::qsa_index_query(iq, p.index_query_norm, call.rope_positions, geometry, s);
    }

    Tensor out = work_.alloc(DType::BF16, {D, dim(a.heads), T});
    const std::size_t scratch = ops::qsa_attention_workspace_bytes(geometry, T, max_context_, layer.kv.storage);
    const DeviceSpan span     = work_.alloc_bytes(scratch);
    ops::qsa_attention(qn, iq, layer, qsa_batch, geometry,
                       static_cast<float>(1.0 / std::sqrt(static_cast<double>(D))), max_context_, span.data,
                       span.bytes, out, s, vq ? &append : nullptr);
    // Output gate: out * sigmoid(gate), then o_proj.
    Tensor gated = out.view({QW, T});
    ops::sigmoid_mul(gate, gated, s);
    Tensor y = work_.alloc(DType::BF16, {H, T});
    project(gated, p.output, y, work_, s, dense_routes());
    return y;
}

Tensor Forward::moe(const MoeParameters& p, const Tensor& x, std::uint32_t layer, Tensor* route_tap,
                    const ForwardBatch* batch, std::int32_t route_columns, const Tensor* inject, Tensor* residual) {
    const cudaStream_t s = device_.stream;
    const auto& m        = config_.moe;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size), E = dim(m.experts), K = dim(m.top_k);
    Tensor logits = work_.alloc(DType::FP32, {E + 1, T});
    const Tensor* rows[] = {&p.router, &p.shared_score};
    ops::projection_fp32(x, rows, logits, s);
    ops::MoeRouting routing{work_.alloc(DType::I32, {K, T}), work_.alloc(DType::FP32, {K, T}),
                            work_.alloc(DType::FP32, {T})};
    // The route log holds the call's last `logged` columns: all of them (written by the dispatch), or
    // the trailing ones (a layer walk's grouped call logs its last chunk), copied from the routing.
    const std::int32_t logged = route_columns < 0 ? T : std::min(route_columns, T);
    std::int32_t* route_log   = nullptr;
    std::int32_t* log_tail    = nullptr;
    if (experts_.route_log != nullptr) {
        if (static_cast<std::size_t>(K) * logged > experts_.route_stride) {
            throw std::invalid_argument("Qwen4Exp MoE: route log is too small for this call");
        }
        std::int32_t* log = experts_.route_log + layer * experts_.route_stride;
        if (logged == T) {
            route_log = log;
        } else {
            log_tail = log;
        }
    }
    const DeviceSpan dispatch_bytes = work_.alloc_bytes(ops::moe_dispatch_bytes(E, K * T));
    ops::MoeDispatch dispatch       = ops::carve_moe_dispatch(dispatch_bytes.data, E, K * T);
    // Without a route tap or a streamed prefill chunk (whose host-side split reads the dispatch), the
    // route waits until the experts' source is known, so its kernel also plans the CPU-served misses.
    const bool streamed_chunk = eager_chunk_ && batch != nullptr && batch->stream && experts_.stream != nullptr &&
                                experts_.stream->active();
    const bool route_plans = route_tap == nullptr && !streamed_chunk;
    if (!route_plans) { ops::moe_route_dispatch(logits, K, routing, dispatch, route_log, s); }
    if (route_tap != nullptr &&
        cudaMemcpyAsync(route_tap->data, routing.ids.data, routing.ids.bytes(), cudaMemcpyDeviceToDevice, s) !=
            cudaSuccess) {
        throw std::runtime_error("Qwen4Exp forward: route tap copy failed");
    }
    ops::MoeExpertSource source{.frame_base   = experts_.frame_bases.empty() ? nullptr : experts_.frame_bases.at(layer),
                                .frames       = experts_.frames.at(layer),
                                .host_records = reinterpret_cast<const std::uint8_t*>(p.bank->planes.device_records),
                                .record_stride = p.bank->planes.record_stride,
                                .scales        = p.device_scales,
                                .activation    = p.bank->activation,
                                .staging_base  = experts_.staging_base,
                                .staging_slots = experts_.staging_slots,
                                .cpu           = experts_.cpu.empty() ? ops::MoeCpuChannel{} : experts_.cpu.at(layer)};
    if (!experts_.host_tables.empty()) {
        source.host_records = nullptr; // the banks hold no records in tier mode
        source.host_table   = experts_.host_tables.at(layer);
        if (!experts_.fetch.empty()) { source.fetch = experts_.fetch.at(layer); }
    }
    source.error = experts_.error;
    // --expert-tail-skip acts in generation rounds only; prefill, forced and scored tokens stay exact.
    if (batch != nullptr && batch->decode_round) { source.tail_skip_rank = experts_.tail_skip_rank; }
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
        if (experts_.landing != nullptr) {
            source.landing       = experts_.landing + static_cast<std::size_t>(layer) * experts_.landing_slots;
            source.landed        = experts_.landed + static_cast<std::size_t>(layer) * experts_.landing_slots;
            source.landing_slots = experts_.landing_slots;
        }
    }
    // While the host computes this call's misses, the CPU wait warms the next layer's weights into L2
    // in the order its dense chain reads them. A one-column call (plain decode) warms the chain up to
    // 40 MiB: tg512 plain 121.8 -> 123.3 tok/s against the attention mixer and 16 MiB of input (32 MiB
    // +0.7 %, 48-56 MiB alike, 64-80 MiB less). A wider call keeps that smaller set: under MTP the
    // wait is shorter and 48-64 MiB measured 0.4-1.4 % slower (design §19.3.20b). A GGUF import's
    // projections warm their segments in order.
    if (source.cpu.max_jobs > 0 && layer + 1 < config_.num_hidden_layers) {
        const auto& next = parameters_.layers[layer + 1];
        const bool whole_chain = T == 1;
        constexpr std::size_t kChainWarm = 40ULL << 20, kInputWarm = 16ULL << 20;
        std::array<std::pair<const void*, std::size_t>, 16> chain{};
        std::size_t spans = 0;
        const auto add = [&](const void* data, std::size_t bytes) {
            if (spans < chain.size()) { chain[spans++] = {data, bytes}; }
        };
        // A linear's weights, at most `cap` bytes of them.
        const auto add_linear = [&](const LinearParameters& linear, std::int32_t k, std::size_t cap) {
            if (!linear.is_ggml()) {
                add(linear.weight.payload, std::min(linear.weight.payload_bytes, cap));
                return;
            }
            for (const auto& segment : linear.ggml) {
                const std::size_t bytes = static_cast<std::size_t>(segment.rows) * ops::ggml_row_bytes(segment.format, k);
                add(segment.data, std::min(bytes, cap));
                cap -= std::min(bytes, cap);
                if (cap == 0) { break; }
            }
        };
        add(next.attn_hc.down.weight.payload, next.attn_hc.down.weight.payload_bytes);
        add(next.attn_hc.up.weight.payload, next.attn_hc.up.weight.payload_bytes);
        const LinearParameters* input  = nullptr;
        const LinearParameters* output = nullptr;
        const LinearParameters* extra  = nullptr; // GDN control or the separate QSA index projection
        std::int32_t output_k          = 0;
        if (config_.layer_types[layer + 1] == MixerKind::Gdn) {
            const auto& g = std::get<GdnParameters>(next.mixer);
            input = &g.projection, extra = &g.control, output = &g.output;
            output_k = dim(config_.gdn.value_width());
        } else {
            const auto& a = std::get<AttentionParameters>(next.mixer);
            input = &a.projection, output = &a.output;
            output_k = dim(config_.attention.query_width());
            if (a.index_projection) { extra = &*a.index_projection; }
        }
        add_linear(*input, H, whole_chain ? ~std::size_t{0} : kInputWarm);
        if (whole_chain) {
            if (extra != nullptr) { add_linear(*extra, H, ~std::size_t{0}); }
            add_linear(*output, output_k, ~std::size_t{0});
            add(next.mlp_hc.down.weight.payload, next.mlp_hc.down.weight.payload_bytes);
            add(next.mlp_hc.up.weight.payload, next.mlp_hc.up.weight.payload_bytes);
            add(next.moe.router.data, next.moe.router.bytes());
            add_linear(next.moe.shared_gate_up, H, ~std::size_t{0});
            add_linear(next.moe.shared_down, dim(config_.moe.shared_intermediate), ~std::size_t{0});
        }
        std::size_t left = whole_chain ? kChainWarm : ~std::size_t{0};
        int i = 0;
        for (std::size_t n = 0; n < spans && i < ops::MoeL2Warm::kSpans && left >= 16; ++n) {
            const auto [data, bytes] = chain[n];
            if (data == nullptr || reinterpret_cast<std::uintptr_t>(data) % 16 != 0 || bytes < 16) { continue; }
            const std::size_t take  = std::min(bytes, left);
            source.l2_warm.ptr[i]   = data;
            source.l2_warm.bytes[i] = take;
            left -= take;
            ++i;
        }
    }
    // A streamed prefill chunk: the layer's copied-ahead experts are read from the ring.
    ExpertStream* const streaming = streamed_chunk ? experts_.stream : nullptr;
    if (streaming != nullptr && streaming->gated()) {
        // A gated call (design §19.3.12): the layer's routing decides which experts the link copies
        // and which the CPU serves. The CPU takes exactly the experts the stream leaves out: every
        // narrow miss is offered (no PCIe share), capped at the policy's count.
        CUDA_CHECK(cudaMemcpyAsync(split_offsets_.data(), dispatch.offsets, (static_cast<std::size_t>(E) + 1) * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaEventRecord(split_ready_, s));
        CUDA_CHECK(cudaEventSynchronize(split_ready_));
        const auto* offsets = static_cast<const std::int32_t*>(split_offsets_.data());
        for (std::int32_t e = 0; e < E; ++e) { split_columns_[static_cast<std::size_t>(e)] = offsets[e + 1] - offsets[e]; }
        std::fill(split_cpu_.begin(), split_cpu_.end(), std::uint8_t{0});
        const std::uint32_t cpu = batch->split && experts_.split != nullptr && source.cpu.max_jobs > 0
                                      ? experts_.split->choose(layer, split_columns_, split_cpu_)
                                      : 0U;
        split_streamed_experts_ += streaming->plan_layer(layer, split_columns_, split_cpu_);
        split_cpu_experts_ += cpu;
        if (cpu > 0) {
            source.cpu.max_columns        = ops::offloaded_moe::kMaxCpuCallColumns;
            source.cpu.pcie_divisor       = 0;
            source.cpu.one_column_divisor = 0;
            source.cpu.one_column_divisor_device = nullptr;
            source.cpu.wide_from          = 1;
            source.cpu.wide_jobs          = static_cast<std::int32_t>(cpu);
        } else {
            source.cpu = ops::MoeCpuChannel{};
        }
    }
    if (streaming != nullptr) {
        streaming->before_experts(layer, s);
        source.prefetched      = streaming->slots(layer);
        source.prefetch_base   = streaming->base();
        source.prefetch_stride = streaming->pitch(); // the ring's slots: the largest record
        // Every expert not resident when the chunk began has a ring slot and every record is in RAM:
        // the call stages nothing, so it runs as one pass over every job instead of staging passes of
        // half the slots (13 for a 16,384-column call of the layer walk). An expert a decode round
        // evicted since is read in place from its host record (the same bytes, so the same bits).
        if (streaming->complete(layer) && experts_.host_tables.empty()) {
            source.staging_slots  = 0;
            source.overlap_stream = nullptr;
        }
    }
    // The layer's frame pool: its pitch is the layer's own record stride.
    const std::uint64_t pitch = experts_.frame_pitches.empty() ? 0 : experts_.frame_pitches.at(layer);
    const auto& planes = p.bank->planes;
    if (planes.gate_up_format != QType::NVFP4_MUL) {
        // A GGUF import's or block-FP8 bank: its own record size in frames of its pool's pitch, the
        // exact A16 kernels on every column count, and CPU-served misses through the exact CPU
        // route; the SSD tier's fetch channel hands over its records like any other. Stages copy the
        // record rounded up to 16 bytes (the stride's zero padding).
        source.ggml = ops::MoeGgmlBank{.gate_up_format = planes.gate_up_format,
                                       .down_format    = planes.down_format,
                                       .gate_row_bytes = static_cast<std::uint32_t>(planes.gate_row_bytes),
                                       .down_row_bytes = static_cast<std::uint32_t>(planes.down_row_bytes),
                                       .record_bytes   = (planes.record_bytes + 15) / 16 * 16};
    }
    source.frame_stride = pitch;
    if (pitch != 0 && pitch != source.record_stride) {
        throw std::invalid_argument("Qwen4Exp MoE: the layer's frame pitch differs from its bank's record stride");
    }
    Tensor outputs = work_.alloc(DType::BF16, {H, K * T});
    const std::int32_t max_jobs = std::min(E, K * T);
    // The two-term route exists for NVFP4 W4A16 banks only; its layout carries no x limbs, which the
    // GGUF and block-FP8 wide routes read, so another bank keeps the exact layout whatever the flag.
    // ggml-bf16-prefill takes a GGUF bank's wide jobs in prefill calls onto the BF16 planes.
    const bool ggml_bank = source.ggml.gate_up_format != QType::NVFP4_MUL &&
                           source.ggml.gate_up_format != QType::FP8_E4M3FN_B128;
    source.wide_a16 = fast_.expert_a4_residual && source.ggml.gate_up_format == QType::NVFP4_MUL
                          ? ops::offloaded_moe::WideA16::kA4Residual
                      : ggml_bank && ops::has_route(dense_routes(), ops::LinearRoutes::GgmlBf16Wide)
                          ? ops::offloaded_moe::WideA16::kGgmlBf16
                          : ops::offloaded_moe::WideA16::kExact;
    const DeviceSpan expert_ws  = work_.alloc_bytes(ops::moe_experts_workspace_bytes(max_jobs, K * T, T, source.activation,
                                                                                 source.wide_a16));
    const bool planned = route_plans && ops::moe_route_dispatch_plan(logits, K, routing, dispatch, route_log, x, source,
                                                                     max_jobs, expert_ws.data, outputs, s);
    if (log_tail != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(log_tail, static_cast<const std::int32_t*>(routing.ids.data) + static_cast<std::size_t>(K) * (T - logged),
                                   static_cast<std::size_t>(K) * logged * sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
    }
    const std::int32_t I = dim(m.shared_intermediate);
    Tensor product = work_.alloc(DType::BF16, {I, T});
    Tensor shared  = work_.alloc(DType::BF16, {H, T});
    // The shared expert runs while the host computes the CPU-served misses, and in decode and
    // verification calls on its own stream while the routed experts compute too. It reads only x,
    // and its scoped workspace lies past every buffer the routed experts use; this stream allocates
    // nothing more until the join.
    const bool side = !eager_chunk_ && experts_.shared_stream != nullptr;
    cudaStream_t shared_s = s;
    if (side) {
        CUDA_CHECK(cudaEventRecord(experts_.shared_events[0], s));
        CUDA_CHECK(cudaStreamWaitEvent(experts_.shared_stream, experts_.shared_events[0], 0));
        shared_s = experts_.shared_stream;
    }
    ops::moe_experts(x, dispatch, source, K, max_jobs, expert_ws.data, outputs, s, /*wait_for_cpu=*/false, planned);
    if (streaming != nullptr && batch->release_stream) { streaming->after_experts(layer, s); }
    shared_swiglu(x, p.shared_gate_up, product, work_, shared_s, dense_routes());
    project(product, p.shared_down, shared, work_, shared_s, dense_routes());
    ops::moe_experts_cpu_wait(x, dispatch, source, max_jobs, expert_ws.data, outputs, s);
    if (side) {
        CUDA_CHECK(cudaEventRecord(experts_.shared_events[1], shared_s));
        CUDA_CHECK(cudaStreamWaitEvent(s, experts_.shared_events[1], 0));
    }
    if (inject != nullptr) {
        ops::moe_combine_inject(outputs, routing, shared, *inject, *residual, s);
        return Tensor{};
    }
    Tensor y = work_.alloc(DType::BF16, {H, T});
    ops::moe_combine(outputs, routing, shared, y, s);
    return y;
}


void Forward::run_mtp(const MtpCall& call) {
    if (!parameters_.mtp || !kv_.mtp || state_.mtp_tails.data == nullptr) {
        throw std::logic_error("Qwen4Exp forward: the MTP drafter is not loaded");
    }
    if (call.batch <= 0 || call.width <= 0 || call.ids.numel() != call.batch * call.width ||
        call.residuals.ne[1] != call.batch * call.width || (!call.kv_only && call.width != 1) ||
        call.rope_positions.ne[0] != call.batch * call.width || call.rope_positions.ne[1] != 3 ||
        call.block_start_rope.ne[0] != call.batch || call.block_start_rope.ne[1] != 3) {
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
        embed_tokens(call.ids, parameters_.token_embedding, x0, s);
        if (call.vision != nullptr && call.vision->columns.data != nullptr) {
            ops::scatter(call.vision->embeddings, call.vision->columns, x0, s);
        }
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
                                           .kv_only          = call.kv_only,
                                           .positions        = call.positions,
                                           .rope_positions   = call.rope_positions,
                                           .block_start_rope = call.block_start_rope,
                                           .slots            = call.slots,
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
        Tensor product = work_.alloc(DType::BF16, {SI, T});
        shared_swiglu(xm, p.shared_gate_up, product, work_, s);
        Tensor shared = work_.alloc(DType::BF16, {H, T});
        project(product, p.shared_down, shared, work_, s);
        ops::moe_combine_inject(outputs, routing, shared, inject, residual, s);
        CUDA_CHECK(cudaMemcpyAsync(call.residual_out.data, residual.data, residual.bytes(), cudaMemcpyDeviceToDevice, s));

        // The draft token: argmax of the draft head (the proposal head's rows mapped to token ids,
        // or the text head over the public tokens).
        Tensor final_x = mix(p.final_mixer, residual, nullptr);
        const auto& head            = parameters_.draft_head;
        const std::int32_t rows_out = head.rows ? head.rows->weight.n : dim(config_.vocab_size);
        Tensor head_logits = work_.alloc(DType::FP32, {rows_out, T});
        if (head.rows) {
            ops::projection_fp32(final_x, head.rows->weight, head_logits, s);
        } else {
            text_head_logits(parameters_, final_x, head_logits, work_, s);
        }
        Tensor drafts = call.drafts;
        Tensor narrow = work_.alloc(DType::BF16, {rows_out, T});
        ops::cast_fp32_to_bf16(head_logits, narrow, s);
        if (call.sample_configs != nullptr) {
            // Sampled drafts: the head's top rows, one draw each under the column's sampler; the top
            // row's log-probability is also the --spec-min-p confidence.
            const std::int32_t valid = head.rows ? rows_out : dim(parameters_.model.resources().public_token_count);
            Tensor sample_rows     = call.sample_rows;
            Tensor sample_logprobs = call.sample_logprobs;
            ops::top_logprobs(narrow, valid, sample_rows, sample_logprobs, s);
            Tensor candidates = call.draft_candidates;
            Tensor q          = call.draft_q;
            Tensor confidence = call.draft_logprobs.data != nullptr
                                    ? Tensor(call.draft_logprobs.data, DType::FP32, {T})
                                    : Tensor{};
            ops::speculative_sample_sparse_drafts(
                sample_rows, sample_logprobs, head.rows ? static_cast<const std::int32_t*>(head.token_ids.data) : nullptr,
                call.sample_configs, call.positions, call.draft_step, drafts, candidates, q, confidence, s);
        } else if (head.rows) {
            ops::argmax(narrow, head.token_ids, drafts, rows_out, s);
        } else {
            ops::argmax(narrow, drafts, dim(parameters_.model.resources().public_token_count), s);
        }
        if (call.draft_logprobs.data != nullptr && call.sample_configs == nullptr) {
            // The top row's log-probability over the head's rows: the draft's own (same rows, same
            // tie order as the argmax above).
            const std::int32_t valid = head.rows ? rows_out : dim(parameters_.model.resources().public_token_count);
            Tensor top_row = work_.alloc(DType::I32, {1, T});
            Tensor logprob(call.draft_logprobs.data, DType::FP32, {1, T});
            ops::top_logprobs(narrow, valid, top_row, logprob, s);
        }
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("qwen4_exp/mtp: ") + error.what());
    }
}

std::size_t Forward::mtp_workspace_bytes(const TextConfig& c, std::int32_t kv_columns, std::int32_t draft_columns,
                                         std::int32_t vocabulary_rows, std::int32_t max_context,
                                         KvCacheStorage storage) {
    const std::size_t t = static_cast<std::size_t>(std::max(std::min(kv_columns, kMtpChunkColumns), draft_columns));
    const std::size_t d = static_cast<std::size_t>(draft_columns);
    // One block at t columns (as the text forward sizes it), the input fusion's and the copied
    // residuals; the draft head's logits and the resident experts only at full steps, which run
    // one column per sequence.
    return workspace_bytes(c, static_cast<std::int32_t>(t), max_context, storage) +
           std::size_t(c.residual_width()) * t * 2 * 4 + std::size_t(c.hidden_size) * t * 2 * 4 +
           std::size_t(vocabulary_rows) * d * 6 +
           ops::resident_moe_workspace_bytes(dim(c.moe.top_k * d), dim(c.moe.intermediate));
}

} // namespace infernix::models::qwen4_exp::execution
