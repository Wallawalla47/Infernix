#include "models/qwen4_exp/execution/dflash2.h"

#include "core/nvtx.h"
#include "infernix/ops/attn_input_proj.h"
#include "infernix/ops/candidate_selector.h"
#include "infernix/ops/context_kv_materialize.h"
#include "infernix/ops/dynamic_grouped_conv.h"
#include "infernix/ops/embedding.h"
#include "infernix/ops/ggml_rows.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/linear_swiglu.h"
#include "infernix/ops/linear_topk.h"
#include "infernix/ops/ggml_linear.h"
#include "infernix/ops/prepare_masked_block.h"
#include "infernix/ops/projection_fp32.h"
#include "infernix/ops/rmsnorm.h"
#include "infernix/ops/rmsnorm_pack_tail.h"
#include "infernix/ops/rmsnorm_rope.h"
#include "infernix/ops/sliding_window_attention.h"
#include "infernix/ops/speculative_round.h"
#include "infernix/ops/top_logprobs.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace infernix::models::qwen4_exp::execution {
namespace {

std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }

// The drafter's top-16 candidates per position (the selector's lattice).
constexpr std::int32_t kCandidates = 16;

void project(const Tensor& x, const LinearParameters& p, Tensor& out, WorkspaceArena& work, cudaStream_t s) {
    ops::linear(x, p.weight, out, p.policy, work, s);
}

// Token rows: BF16, or a GGUF import's IQ4_XS rows decoded exactly (as Forward embeds them).
void embed_tokens(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t s) {
    if (table.layout == QuantLayout::GgmlRows) {
        ops::ggml_embedding(ids, table, out, s);
    } else {
        ops::embedding(ids, table, out, s);
    }
}

// The fused Ops take Q8_G32_FP16 RowSplit projections (the official companion recipe's).
void require_q8(const Weight& weight, const char* name) {
    if (weight.qtype != QType::Q8_G32_FP16 || weight.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument(std::string("Qwen4Exp DFlash2: ") + name +
                                    " must be q8_g32_fp16 (tools.convert recipe qwen3_8_flash_next_dflash2)");
    }
}

} // namespace

DFlash2::DFlash2(const Parameters& parameters, const DFlash2Config& config, ProposalHead head, DeviceContext& device,
                 WorkspaceArena& work, std::vector<CyclicKVCacheLayerView> ring, std::uint32_t max_context)
    : parameters_(parameters), p_(parameters.dflash2.value()), config_(config), text_(parameters.model.config().text),
      head_(head), device_(device), work_(work), ring_(std::move(ring)), max_context_(max_context) {
    if (ring_.size() != config_.layers || p_.layers.size() != config_.layers) {
        throw std::invalid_argument("Qwen4Exp DFlash2: one context ring layer per drafter layer");
    }
    if (head_ == ProposalHead::Optimized && !parameters_.draft_head.rows) {
        throw std::invalid_argument("Qwen4Exp DFlash2: --lm-head-draft needs the proposal head");
    }
    const std::int32_t rows = head_ == ProposalHead::Optimized ? parameters_.draft_head.rows->weight.n
                                                               : dim(text_.vocab_size);
    const bool fused = head_ == ProposalHead::Optimized
                           ? !parameters_.draft_head.rows->is_ggml() &&
                                 parameters_.draft_head.rows->weight.qtype == QType::Q4_G64_FP16 && rows == 131072
                           : parameters_.output_head_q8 && !parameters_.output_head_q8->is_ggml() &&
                                 parameters_.output_head_q8->weight.qtype == QType::Q8_G32_FP16;
    if (rows != dim(config_.head_rows) || fused != config_.fused_head) {
        throw std::logic_error("Qwen4Exp DFlash2: the bound head differs from the planned one");
    }
    require_q8(p_.feature_projection.weight, "the feature projection");
    for (const auto& layer : p_.layers) {
        require_q8(layer.query_key_value, "the query/key/value projections");
        require_q8(layer.context_key.weight, "the context key projection");
        require_q8(layer.context_value.weight, "the context value projection");
        require_q8(layer.output.weight, "the attention output projection");
        require_q8(layer.gate_up.weight, "the MLP gate/up projections");
        require_q8(layer.down.weight, "the MLP down projection");
    }
    // The fused Ops' contract (context_kv_materialize: five local layers, eps 1e-6; the selector's
    // 16 candidates of rank 256).
    if (config_.layers != ops::kContextKVMaterializeLayers || config_.rms_norm_eps != 1e-6F ||
        config_.selector_top_k != static_cast<std::uint32_t>(kCandidates) || config_.conv_kernel != 2 ||
        config_.conv_group != 16) {
        throw std::invalid_argument("Qwen4Exp DFlash2: the drafter is outside the fused Ops' registered geometry");
    }
}

void DFlash2::materialize(const DFlash2ContextCall& call) {
    const cudaStream_t s = device_.stream;
    const std::int32_t H = dim(text_.hidden_size), W = call.features.ne[1], B = call.features.ne[2];
    if (call.features.dtype != DType::BF16 || call.features.ne[0] != dim(config_.feature_width(text_.hidden_size)) ||
        W <= 0 || B <= 0 || call.max_count > static_cast<std::uint32_t>(W) || call.min_count > call.max_count) {
        throw std::invalid_argument("Qwen4Exp DFlash2: context call geometry is invalid");
    }
    const nvtx::ScopedRange range(nvtx::Name::DFlashContextAppend, nvtx::Category::DFlash, static_cast<std::uint64_t>(W * B));
    work_.reset();
    Tensor projected = work_.alloc(DType::BF16, {H, W * B});
    project(call.features.view({call.features.ne[0], W * B}), p_.feature_projection, projected, work_, s);
    Tensor context = work_.alloc(DType::BF16, {H, W, B});
    Tensor flat    = context.view({H, W * B});
    ops::rmsnorm(projected, p_.context_norm, config_.rms_norm_eps, false, flat, s);
    std::array<ops::ContextKVMaterializeLayerView, ops::kContextKVMaterializeLayers> layers;
    for (std::size_t l = 0; l < layers.size(); ++l) {
        const auto& layer = p_.layers[l];
        layers[l]         = {layer.context_key.weight, layer.context_value.weight, layer.key_norm, ring_[l]};
    }
    ops::context_kv_materialize(context, call.positions, call.counts, call.slots, layers, config_.rope_theta,
                                {call.min_count, call.max_count}, work_, s);
}

void DFlash2::head_candidates(const Tensor& hidden, Tensor& ids, Tensor& scores) {
    const cudaStream_t s = device_.stream;
    const std::int32_t columns = hidden.ne[1];
    const auto& head           = parameters_.draft_head;
    if (config_.fused_head) {
        if (head_ == ProposalHead::Optimized) {
            ops::linear_topk(hidden, head.rows->weight, head.token_ids, ids, scores, work_, s);
        } else {
            ops::linear_topk(hidden, parameters_.output_head_q8->weight,
                             dim(parameters_.model.resources().public_token_count), ids, scores, work_, s);
        }
        return;
    }
    // Any other head (a BF16 or GGUF text head, another proposal format): FP32 logits.
    auto scope              = work_.scope();
    const std::int32_t rows = dim(config_.head_rows);
    Tensor logits      = work_.alloc(DType::FP32, {rows, columns});
    if (head_ == ProposalHead::Optimized) {
        if (head.rows->is_ggml()) {
            ops::ggml_linear(hidden, head.rows->ggml, logits, nullptr, 0, work_, s, ops::LinearRoutes::Exact);
        } else {
            ops::projection_fp32(hidden, head.rows->weight, logits, s);
        }
    } else if (parameters_.output_head_q8 && parameters_.output_head_q8->is_ggml()) {
        ops::ggml_linear(hidden, parameters_.output_head_q8->ggml, logits, nullptr, 0, work_, s, ops::LinearRoutes::Exact);
    } else if (parameters_.output_head_q8) {
        ops::projection_fp32(hidden, parameters_.output_head_q8->weight, logits, s);
    } else {
        const Tensor* rows_tensor[] = {&parameters_.output_head};
        ops::projection_fp32(hidden, rows_tensor, logits, s);
    }
    const std::int32_t valid = head_ == ProposalHead::Optimized ? rows : dim(parameters_.model.resources().public_token_count);
    ops::top_logprobs(logits, valid, ids, scores, s);
    if (head_ == ProposalHead::Optimized) {
        Tensor flat = ids.view({kCandidates * columns});
        ops::proposal_remap_token_ids(flat, static_cast<const std::int32_t*>(head.token_ids.data), rows, s);
    }
}

void DFlash2::propose(const DFlash2ProposalCall& call) {
    const cudaStream_t s = device_.stream;
    const std::int32_t H = dim(text_.hidden_size), W = call.width, B = call.batch, K = W - 1;
    const std::int32_t D = dim(config_.attention.head_dim), Hq = dim(config_.attention.heads),
                       Hk = dim(config_.attention.kv_heads), I = dim(config_.intermediate);
    const std::int32_t G = H / dim(config_.conv_group), columns = W * B;
    if (W < 2 || W > 16 || B < 1 || B > static_cast<std::int32_t>(kMaximumConcurrency) || call.configs == nullptr) {
        throw std::invalid_argument("Qwen4Exp DFlash2: proposal geometry is invalid");
    }
    const nvtx::ScopedRange range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash, static_cast<std::uint64_t>(columns));
    work_.reset();
    Tensor ids       = call.ids;
    Tensor positions = call.positions;
    ops::prepare_masked_block(call.anchors, call.frontiers, call.valid_columns, dim(config_.mask_token_id), ids,
                              positions, s);
    Tensor residual = work_.alloc(DType::BF16, {H, W, B});
    {
        Tensor flat = residual.view({H, columns});
        embed_tokens(ids.view({columns}), parameters_.token_embedding, flat, s);
    }
    const ops::SlidingWindowAttentionExecutionEnvelope envelope{0, max_context_};
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));
    for (std::size_t l = 0; l < p_.layers.size(); ++l) {
        const auto& layer = p_.layers[l];
        const nvtx::ScopedRange layer_range(nvtx::Name::DFlashLayer, nvtx::Category::DFlash, l);
        {
            // Attention: prepare (norm, dynamic conv), Q/K/V, Q/K norm and RoPE, attention over the
            // ring and the block, then the output projection, conv finish and residual add.
            auto scope       = work_.scope();
            Tensor prepared  = work_.alloc(DType::BF16, {H, W, B});
            Tensor delta     = work_.alloc(DType::BF16, {G, 2, W, B});
            {
                WorkspaceArena scratch(work_.alloc_bytes(
                    ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(H, W, W, B, B)));
                ops::rmsnorm_dynamic_grouped_conv_prepare(residual, layer.input_norm, config_.rms_norm_eps,
                                                          layer.attention_base, layer.attention_projection.weight,
                                                          prepared, delta, scratch, s);
            }
            Tensor q = work_.alloc(DType::BF16, {D, Hq, W, B});
            Tensor k = work_.alloc(DType::BF16, {D, Hk, W, B});
            Tensor v = work_.alloc(DType::BF16, {D, Hk, W, B});
            {
                Tensor qf = q.view({D * Hq, columns}), kf = k.view({D * Hk, columns}), vf = v.view({D * Hk, columns});
                ops::attn_input_proj(prepared.view({H, columns}), layer.query_key_value, qf, kf, vf, s);
            }
            ops::rmsnorm_rope(positions, layer.query_norm, layer.key_norm, p_.rope, q, k, s);
            Tensor attention = work_.alloc(DType::BF16, {D, Hq, W, B});
            ops::sliding_window_attention(q, k, v, positions, call.valid_columns, call.slots, {D, Hq, Hk},
                                          config_.window, scale, ring_[l], envelope, work_, attention, s);
            WorkspaceArena scratch(work_.alloc_bytes(
                ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(H, D * Hq, W, W, B, B)));
            ops::linear_dynamic_grouped_conv_add(attention.view({D * Hq, W, B}), layer.output.weight,
                                                 layer.attention_base, delta, residual, scratch, s);
        }
        {
            // MLP: prepare, SwiGLU(gate, up), then down, conv finish and residual add.
            auto scope      = work_.scope();
            Tensor prepared = work_.alloc(DType::BF16, {H, W, B});
            Tensor delta    = work_.alloc(DType::BF16, {G, 2, W, B});
            {
                WorkspaceArena scratch(work_.alloc_bytes(
                    ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(H, W, W, B, B)));
                ops::rmsnorm_dynamic_grouped_conv_prepare(residual, layer.post_attention_norm, config_.rms_norm_eps,
                                                          layer.mlp_base, layer.mlp_projection.weight, prepared, delta,
                                                          scratch, s);
            }
            Tensor product = work_.alloc(DType::BF16, {I, W, B});
            {
                Tensor flat = product.view({I, columns});
                ops::linear_swiglu(prepared.view({H, columns}), layer.gate_up.weight, flat, layer.gate_up.policy, work_, s);
            }
            WorkspaceArena scratch(
                work_.alloc_bytes(ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(H, I, W, W, B, B)));
            ops::linear_dynamic_grouped_conv_add(product, layer.down.weight, layer.mlp_base, delta, residual, scratch, s);
        }
    }
    // The mask columns' final hidden, their head candidates and the selector's conditional path.
    Tensor hidden = work_.alloc(DType::BF16, {H, K * B});
    ops::rmsnorm_pack_tail(residual, p_.final_norm, hidden, s);
    Tensor candidates = call.candidates;
    Tensor ids_flat   = candidates.view({kCandidates, K * B});
    Tensor scores     = work_.alloc(DType::FP32, {kCandidates, K * B});
    head_candidates(hidden, ids_flat, scores);
    Tensor projected = work_.alloc(DType::BF16, {dim(config_.selector_rank), K * B});
    project(hidden, p_.selector_hidden, projected, work_, s);
    Tensor drafts     = call.drafts;
    Tensor q          = call.proposal_q;
    Tensor confidence = call.confidence;
    ops::candidate_selector_path(candidates, scores.view({kCandidates, K, B}),
                                 projected.view({dim(config_.selector_rank), K, B}), call.anchors,
                                 p_.predecessor_codebook, p_.successor_codebook, call.frontiers, call.configs, drafts, q,
                                 confidence.data != nullptr ? &confidence : nullptr, work_, s);
}

std::size_t DFlash2::workspace_bytes(const TextConfig& text, const DFlash2Config& config, std::int32_t max_batch,
                                     std::int32_t max_width, std::int32_t context_columns, std::uint32_t max_context) {
    constexpr auto kQ8           = QType::Q8_G32_FP16;
    constexpr auto kA16          = ops::LinearPolicy::A16Only;
    constexpr std::size_t kSlack = 256; // per allocation
    const std::size_t H = text.hidden_size, I = config.intermediate, D = config.attention.head_dim;
    const std::size_t q = config.attention.query_width(), kv = config.attention.key_width();
    const std::size_t G = H / config.conv_group, F = config.feature_width(text.hidden_size);
    // materialize: the projected and normalized context, the projection and the materializer, for
    // the batched profile (up to kBatchedContextColumns per row) and one row's widest call.
    std::size_t context = 0;
    for (const std::int32_t columns : {std::min(context_columns, kBatchedContextColumns), context_columns}) {
        const std::int32_t rows = columns <= kBatchedContextColumns ? max_batch : 1;
        const std::size_t cells = static_cast<std::size_t>(columns) * static_cast<std::size_t>(rows);
        context = std::max(context, 2 * (H * cells * 2 + kSlack) +
                                        ops::linear_workspace_capacity_bytes(kQ8, dim(H), dim(F), kA16, 1, dim(cells)) +
                                        ops::context_kv_materialize_workspace_capacity_bytes(rows, 1, columns));
    }
    // propose at the widest block (every buffer grows with W and B): the residual, one branch's
    // buffers and scratch, then the head's candidates and the selector.
    const std::size_t W = static_cast<std::size_t>(max_width), B = static_cast<std::size_t>(max_batch);
    const std::size_t cells = W * B, masks = (W - 1) * B;
    const std::size_t branch = (H * cells * 2 + kSlack) + (G * 2 * cells * 2 + kSlack);
    std::size_t prepare = 0, add_attention = 0, add_mlp = 0, window = 0;
    for (std::int32_t w = 2; w <= max_width; ++w) {
        for (std::int32_t b = 1; b <= max_batch; ++b) {
            prepare = std::max(prepare, ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(dim(H), w, w, b, b));
            add_attention =
                std::max(add_attention, ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(dim(H), dim(q), w, w, b, b));
            add_mlp = std::max(add_mlp, ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(dim(H), dim(I), w, w, b, b));
            window  = std::max(window, ops::sliding_window_attention_workspace_capacity_bytes(
                                           {dim(D), dim(config.attention.heads), dim(config.attention.kv_heads)},
                                           config.window, {0, max_context}, w, w, b));
        }
    }
    const std::size_t swiglu = ops::linear_swiglu_workspace_capacity_bytes(kQ8, dim(2 * I), dim(H), kA16, 1, dim(cells));
    const std::size_t attention = branch + (q + 2 * kv) * cells * 2 + 3 * kSlack + (q * cells * 2 + kSlack) +
                                  std::max(prepare, window) + add_attention + 2 * kSlack;
    const std::size_t mlp = branch + (I * cells * 2 + kSlack) + std::max(prepare, swiglu) + add_mlp + 2 * kSlack;
    // The head: linear_topk's scratch, or the FP32 logits of another head (and a GGUF head's
    // decode scratch). The selector's projection runs Q8 or BF16.
    std::size_t head = 0;
    if (config.fused_head) {
        head = ops::linear_topk_workspace_capacity_bytes(config.head_rows == 131072U ? QType::Q4_G64_FP16 : kQ8,
                                                         dim(config.head_rows), dim(H), 1, dim(masks));
    } else {
        head = static_cast<std::size_t>(config.head_rows) * masks * 4 + kSlack +
               (config.ggml_head ? ops::ggml_linear_workspace_bytes(dim(H), dim(masks)) : 0);
    }
    std::size_t selector = ops::candidate_selector_path_workspace_capacity_bytes(1, max_width - 1, 1, max_batch);
    for (const QType format : {kQ8, QType::BF16}) {
        try {
            selector = std::max(selector, ops::linear_workspace_capacity_bytes(format, dim(config.selector_rank), dim(H),
                                                                               kA16, 1, dim(masks)));
        } catch (const std::invalid_argument&) {}
    }
    const std::size_t tail = (H * masks * 2 + kSlack) + (kCandidates * masks * 4 + kSlack) +
                             (config.selector_rank * masks * 2 + kSlack) + std::max(head, selector);
    const std::size_t propose = (H * cells * 2 + kSlack) + std::max({attention, mlp, tail});
    return std::max(context, propose) + (1ULL << 20); // alignment slack
}

} // namespace infernix::models::qwen4_exp::execution
