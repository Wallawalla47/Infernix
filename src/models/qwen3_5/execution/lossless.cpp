#include "models/qwen3_5/execution/lossless.h"

#include "infernix/ops/linear.h"
#include "infernix/ops/lossless_linear.h"
#include "infernix/ops/lossless_ops.h"
#include "infernix/ops/lossless_quantize.h"
#include "infernix/ops/offloaded_sparse_moe.h"

#include <algorithm>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <variant>

// Every step below is one operation of Hugging Face's Qwen3.5 forward (modeling_qwen3_5.py and
// modeling_qwen3_5_moe.py, Transformers 5.5) as docs/maintainer/lossless-route.md §2 gives it; the
// row maps of the fused projections are the loader's (attention: query, key, output gate, value; GDN:
// q, k, v channels then z; its control: a then b; MLP and experts: gate then up). Each function runs
// with a WorkspaceArena, or sizes its allocations with a WorkspaceLayoutBuilder (no kernels): one
// sequence of allocations serves execution and startup planning.

namespace infernix::models::qwen3_5::execution {
namespace {

// Columns of an MLP or MoE pass (token-independent, so splitting them changes no result).
constexpr std::int32_t kFfnColumns = 4096;

template <class A>
constexpr bool kRuns = std::is_same_v<A, WorkspaceArena>;

ResidualTap g_residual_tap;

template <class A>
Tensor alloc(A& a, DType dtype, std::initializer_list<std::int32_t> shape) {
    return a.alloc(dtype, shape);
}

// Rows [row_begin, row_begin + rows) of w into rows [out_row, ...) of out.
template <class A>
void linear(A& a, const Tensor& x, const Weight& w, std::int32_t row_begin, std::int32_t rows, Tensor& out,
            std::int32_t out_row, cudaStream_t s, const ops::LosslessActivationScale& scale = {}) {
    if constexpr (kRuns<A>) {
        ops::lossless_linear(x, w, row_begin, rows, out, out_row, a, s, scale);
    } else {
        auto scope = a.scope();
        (void)a.alloc_bytes(ops::lossless_linear_workspace_bytes(w, rows, x.ne[1]));
    }
}

// Projections of one input sharing its encode (ops::lossless_linear_shared).
template <class A>
void shared(A& a, const Tensor& x, std::initializer_list<ops::LosslessProjection> projections, cudaStream_t s,
            const ops::LosslessActivationScale& scale = {}) {
    const std::span<const ops::LosslessProjection> list(projections.begin(), projections.size());
    if constexpr (kRuns<A>) {
        ops::lossless_linear_shared(x, list, a, s, scale);
    } else {
        auto scope = a.scope();
        (void)a.alloc_bytes(ops::lossless_linear_shared_workspace_bytes(list, x.ne[1]));
    }
}

// The design reference's quantization of one projection input (lossless-route.md §3b): the step its
// weight's Use declares (NVFP4 weights permitted A4: Qwen3.5's NVFP4 rule with the input's divisor; FP8 row
// weights permitted A8: the per-token E4M3 rule, or with a static input scale (fp8_e4m3fn_row_f32,
// ModelOpt's calibrated W8A8) the static rule), or none.
enum class Step { None, Nvfp4, Fp8Token, Fp8Static };

Step design_step(const LosslessDesign& design, const LinearParameters& p) {
    if (!design.active) { return Step::None; }
    if (p.weight.qtype == QType::NVFP4 && ops::allows_a4(p.policy)) { return Step::Nvfp4; }
    if (p.weight.qtype == QType::FP8_E4M3FN_ROW_BF16 && ops::allows_a8(p.policy)) { return Step::Fp8Token; }
    // A per-token fp8_e4m3fn_row_f32 parent (no calibrated scale: ModelOpt FP8_PER_CHANNEL_PER_TOKEN)
    // takes the per-token rule.
    if (p.weight.qtype == QType::FP8_E4M3FN_ROW_F32 && ops::allows_a8(p.policy)) {
        return p.weight.input_scale > 0.0F ? Step::Fp8Static : Step::Fp8Token;
    }
    return Step::None;
}

Step design_step(const LosslessDesign& design, const ops::ProjectionWeights& p) {
    const auto* single = std::get_if<LinearParameters>(&p);
    return single != nullptr ? design_step(design, *single) : Step::None;
}

// x quantized by `step` (x itself for none) and the activation scale lossless_linear applies to the sums.
struct Quantized {
    Tensor x;
    ops::LosslessActivationScale scale;
};

template <class A>
Quantized quantize(A& a, Step step, const Tensor& x, const Weight& w, cudaStream_t s) {
    if (step == Step::None) { return {x, {}}; }
    Tensor q = alloc(a, DType::BF16, {x.ne[0], x.ne[1]});
    if (step == Step::Nvfp4) {
        if constexpr (kRuns<A>) { ops::lossless_quantize_nvfp4(x, w.input_scale_divisor, q, s); }
        return {q, {w.input_scale_divisor, nullptr}};
    }
    Tensor scales = alloc(a, DType::FP32, {x.ne[1]});
    if constexpr (kRuns<A>) {
        if (step == Step::Fp8Static) {
            ops::lossless_quantize_fp8_static(x, w.input_scale, q, scales, s);
        } else {
            ops::lossless_quantize_fp8_token(x, q, scales, s);
        }
    }
    return {q, {0.0F, static_cast<const float*>(scales.data)}};
}

template <class A>
void grouped(A& a, const Tensor& x, const Weight& w, const Tensor& tiles, std::int32_t stride, std::int32_t offset,
             std::int32_t rows, Tensor& out, cudaStream_t s) {
    if constexpr (kRuns<A>) {
        ops::lossless_linear_grouped(x, w, tiles, stride, offset, rows, out, a, s);
    } else {
        auto scope = a.scope();
        (void)a.alloc_bytes(ops::lossless_linear_workspace_bytes(w, rows, x.ne[1]));
    }
}

template <class A>
void attention_layer(A& a, const BlockParameters& block, const TextConfig& config, const Tensor& rope_table,
                     Tensor& x, cudaStream_t s, const LosslessCache* cache, const LosslessDesign& design) {
    const auto& p    = std::get<AttentionParameters>(block.mixer);
    const auto& att  = *config.attention;
    const auto H     = dimension(config.hidden_size);
    const auto Q     = dimension(att.query_width());
    const auto K     = dimension(att.key_width());
    const auto D     = dimension(att.head_dim);
    const auto heads = dimension(att.num_attention_heads), kv_heads = dimension(att.num_key_value_heads);
    const std::int32_t T = x.ne[1];
    auto scope = a.scope();
    Tensor h = alloc(a, DType::BF16, {H, T});
    if constexpr (kRuns<A>) { ops::lossless_rmsnorm(x, block.input_norm, config.rms_norm_eps, h, s); }
    Tensor q = alloc(a, DType::BF16, {Q, T}), gate = alloc(a, DType::BF16, {Q, T});
    Tensor k = alloc(a, DType::BF16, {K, T}), v = alloc(a, DType::BF16, {K, T});
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&p.projection)) {
        // [query, key] and [output gate, value].
        shared(a, h,
               {{&pair->first, 0, Q, &q, 0}, {&pair->first, Q, K, &k, 0}, {&pair->second, 0, Q, &gate, 0},
                {&pair->second, Q, K, &v, 0}},
               s);
    } else {
        const LinearParameters& lp = std::get<LinearParameters>(p.projection);
        const Weight& w            = lp.weight;
        const Quantized hq         = quantize(a, design_step(design, lp), h, w, s);
        shared(a, hq.x, {{&w, 0, Q, &q, 0}, {&w, Q, K, &k, 0}, {&w, Q + K, Q, &gate, 0}, {&w, 2 * Q + K, K, &v, 0}}, s,
               hq.scale);
    }
    Tensor q3 = q.view({D, heads, T}), k3 = k.view({D, kv_heads, T}), v3 = v.view({D, kv_heads, T});
    Tensor o = alloc(a, DType::BF16, {D, heads, T});
    // The design reference with a coded KV format: K and V through its production encoder, attention over the
    // stored rows (lossless-route.md §3b).
    const bool codes    = design.active && design.kv != KvCacheStorage::BFloat16;
    const DeviceSpan ws = a.alloc_bytes(
        cache != nullptr ? ops::lossless_paged_attention_workspace_bytes(heads, kv_heads, T / cache->sequences,
                                                                        cache->sequences, cache->key_capacity)
        : codes          ? ops::lossless_codec_attention_workspace_bytes(heads, kv_heads, T)
                         : ops::lossless_attention_workspace_bytes(heads, kv_heads, T));
    const DeviceSpan stored_rows =
        codes ? a.alloc_bytes(ops::lossless_kv_encode_bytes(design.kv, kv_heads, T)) : DeviceSpan{};
    if constexpr (kRuns<A>) {
        Tensor qr = q.view({D, heads * T}), kr = k.view({D, kv_heads * T});
        ops::lossless_rmsnorm(qr, p.query_norm, config.rms_norm_eps, qr, s);
        ops::lossless_rmsnorm(kr, p.key_norm, config.rms_norm_eps, kr, s);
        ops::lossless_rope_apply(rope_table, q3, s);
        ops::lossless_rope_apply(rope_table, k3, s);
        if (cache != nullptr) {
            ops::lossless_kv_append(k3, v3, *cache->positions, cache->sequences, *cache->table_rows, cache->kv, s);
            ops::lossless_paged_attention(q3, *cache->positions, cache->sequences, *cache->table_rows, cache->kv,
                                          cache->max_keys, cache->key_capacity, o, ws.data, s);
        } else if (codes) {
            const ops::LosslessKvCodes stored = ops::lossless_kv_encode(k3, v3, design.kv, stored_rows.data, s);
            ops::lossless_codec_attention(q3, stored, kv_heads, o, ws.data, s);
        } else {
            // The design's bf16 KV stores V as FP16 (exact in BF16 again).
            if (design.active) { ops::lossless_round_fp16(v3, s); }
            ops::lossless_attention(q3, k3, v3, o, ws.data, s);
        }
        Tensor flat = o.view({Q, T});
        ops::lossless_sigmoid_mul(flat, gate, flat, s);
    }
    Tensor y = alloc(a, DType::BF16, {H, T});
    Tensor flat = o.view({Q, T});
    const Quantized oq = quantize(a, design_step(design, p.output), flat, p.output.weight, s);
    linear(a, oq.x, p.output.weight, 0, H, y, 0, s, oq.scale);
    if constexpr (kRuns<A>) { ops::lossless_add(x, y, x, s); }
}

template <class A>
void gdn_layer(A& a, const BlockParameters& block, const TextConfig& config, Tensor& x, cudaStream_t s,
               const LosslessCache* cache, const LosslessDesign& design) {
    const auto& p   = std::get<GdnParameters>(block.mixer);
    const auto& gdn = *config.gdn;
    const auto H    = dimension(config.hidden_size);
    const auto KW = dimension(gdn.key_width()), VW = dimension(gdn.value_width());
    const auto Hv = dimension(gdn.linear_num_value_heads), dv = dimension(gdn.linear_value_head_dim);
    const std::int32_t C = 2 * KW + VW;
    const std::int32_t T = x.ne[1];
    if (gdn.linear_key_head_dim != 128 || gdn.linear_value_head_dim != 128) {
        throw std::invalid_argument("the lossless gated delta rule takes head dims 128");
    }
    auto scope = a.scope();
    const auto tap = [&](const char* tag, const Tensor& t) {
        if constexpr (kRuns<A>) {
            if (g_residual_tap) { g_residual_tap(tag, t, s); }
        }
    };
    Tensor h = alloc(a, DType::BF16, {H, T});
    if constexpr (kRuns<A>) { ops::lossless_rmsnorm(x, block.input_norm, config.rms_norm_eps, h, s); }
    Tensor mixed = alloc(a, DType::BF16, {C, T}), z = alloc(a, DType::BF16, {VW, T});
    Tensor av = alloc(a, DType::BF16, {Hv, T}), bv = alloc(a, DType::BF16, {Hv, T});
    // qkv = [query-key rows; value rows], z = the value/z parent's z rows; the control's a then b rows; all
    // from h.
    const auto* proj_pair    = std::get_if<ops::PairedProjectionWeights>(&p.projection);
    const auto* control_pair = std::get_if<ops::PairedProjectionWeights>(&p.control);
    const Weight* qk_w   = proj_pair ? &proj_pair->first : &std::get<LinearParameters>(p.projection).weight;
    const Weight* vz_w   = proj_pair ? &proj_pair->second : qk_w;
    const std::int32_t qk = proj_pair ? proj_pair->first.n : C;
    const Weight* a_w    = control_pair ? &control_pair->first : &std::get<LinearParameters>(p.control).weight;
    const Weight* b_w    = control_pair ? &control_pair->second : a_w;
    const std::int32_t b_row = control_pair ? 0 : Hv;
    const Step projection_step = design_step(design, p.projection), control_step = design_step(design, p.control);
    if (projection_step == Step::None && control_step == Step::None) {
        if (proj_pair) {
            shared(a, h,
                   {{qk_w, 0, qk, &mixed, 0}, {vz_w, 0, C - qk, &mixed, qk}, {vz_w, C - qk, VW, &z, 0},
                    {a_w, 0, Hv, &av, 0}, {b_w, b_row, Hv, &bv, 0}},
                   s);
        } else {
            shared(a, h, {{qk_w, 0, C, &mixed, 0}, {qk_w, C, VW, &z, 0}, {a_w, 0, Hv, &av, 0}, {b_w, b_row, Hv, &bv, 0}},
                   s);
        }
    } else {
        // The design reference: the projection and the control each from h quantized as its own Use declares.
        const Quantized hp = quantize(a, projection_step, h, *qk_w, s);
        if (proj_pair) {
            shared(a, hp.x, {{qk_w, 0, qk, &mixed, 0}, {vz_w, 0, C - qk, &mixed, qk}, {vz_w, C - qk, VW, &z, 0}}, s,
                   hp.scale);
        } else {
            shared(a, hp.x, {{qk_w, 0, C, &mixed, 0}, {qk_w, C, VW, &z, 0}}, s, hp.scale);
        }
        const Quantized hc = quantize(a, control_step, h, *a_w, s);
        shared(a, hc.x, {{a_w, 0, Hv, &av, 0}, {b_w, b_row, Hv, &bv, 0}}, s, hc.scale);
    }
    tap("gdn_mixed", mixed);
    tap("gdn_z", z);
    tap("gdn_a", av);
    tap("gdn_b", bv);
    Tensor g = alloc(a, DType::FP32, {Hv, T}), beta = alloc(a, DType::BF16, {Hv, T});
    Tensor c = alloc(a, DType::BF16, {C, T});
    Tensor o = alloc(a, DType::BF16, {VW, T}), on = alloc(a, DType::BF16, {VW, T});
    if constexpr (kRuns<A>) {
        ops::lossless_gdn_gates(av, bv, p.a_log, p.dt_bias, g, beta, s);
        tap("gdn_g", g);
        tap("gdn_beta", beta);
        const Tensor qc = c.slice(0, 0, KW), kc = c.slice(0, KW, KW), vc = c.slice(0, 2 * KW, VW);
        if (cache != nullptr) {
            Tensor conv = cache->conv_states, recurrent = cache->recurrent_states;
            ops::lossless_causal_conv_slots(mixed, p.convolution, cache->sequences, conv, *cache->source_slots,
                                            *cache->destination_slots, c, s);
            ops::lossless_gated_delta_rule_slots(qc, kc, vc, g, beta, cache->sequences, recurrent,
                                                 *cache->source_slots, *cache->destination_slots, o, s);
        } else {
            ops::lossless_causal_conv(mixed, p.convolution, c, s);
            tap("gdn_conv", c);
            ops::lossless_gated_delta_rule(qc, kc, vc, g, beta, nullptr, o, s);
            tap("gdn_o", o);
        }
        Tensor o2 = o.view({dv, Hv * T}), z2 = z.view({dv, Hv * T}), on2 = on.view({dv, Hv * T});
        ops::lossless_rmsnorm_gated(o2, z2, p.norm, config.rms_norm_eps, on2, s);
        tap("gdn_on", on);
    }
    Tensor y = alloc(a, DType::BF16, {H, T});
    const Quantized oq = quantize(a, design_step(design, p.output), on, p.output.weight, s);
    linear(a, oq.x, p.output.weight, 0, H, y, 0, s, oq.scale);
    tap("gdn_y", y);
    if constexpr (kRuns<A>) { ops::lossless_add(x, y, x, s); }
}

template <class A>
void dense_mlp(A& a, const DenseParameters& p, const Tensor& h, Tensor& x, cudaStream_t s, const LosslessDesign& design) {
    const std::int32_t H = h.ne[0], T = h.ne[1], I = p.gate_up.weight.n / 2;
    for (std::int32_t c0 = 0; c0 < T; c0 += kFfnColumns) {
        const std::int32_t n = std::min(kFfnColumns, T - c0);
        auto scope = a.scope();
        const Tensor hc = h.slice(1, c0, n);
        Tensor g = alloc(a, DType::BF16, {I, n}), u = alloc(a, DType::BF16, {I, n});
        const Quantized hq = quantize(a, design_step(design, p.gate_up), hc, p.gate_up.weight, s);
        shared(a, hq.x, {{&p.gate_up.weight, 0, I, &g, 0}, {&p.gate_up.weight, I, I, &u, 0}}, s, hq.scale);
        if constexpr (kRuns<A>) { ops::lossless_silu_mul(g, u, g, s); }
        Tensor d = alloc(a, DType::BF16, {H, n});
        const Quantized gq = quantize(a, design_step(design, p.down), g, p.down.weight, s);
        linear(a, gq.x, p.down.weight, 0, H, d, 0, s, gq.scale);
        if constexpr (kRuns<A>) {
            Tensor xc = x.slice(1, c0, n);
            ops::lossless_add(xc, d, xc, s);
        }
        if constexpr (!kRuns<A>) { break; } // every pass has the first one's allocations or fewer
    }
}

template <class A>
void sparse_moe(A& a, const ops::SparseMoeWeights& p, const MoeConfig& moe, const Tensor& h, Tensor& x,
                cudaStream_t s) {
    const std::int32_t H = h.ne[0], T = h.ne[1];
    const std::int32_t E = p.router_shared_gate.n - 1, k = dimension(moe.num_experts_per_tok);
    const std::int32_t I = p.routed_down.k, Is = p.shared_down.k;
    if (E != dimension(moe.num_experts) || p.routed_gate_up.n != E * 2 * I || p.routed_down.n != E * H ||
        p.shared_gate_up.n != 2 * Is) {
        throw std::invalid_argument("the lossless MoE's banks do not match the configuration");
    }
    for (std::int32_t c0 = 0; c0 < T; c0 += kFfnColumns) {
        const std::int32_t n = std::min(kFfnColumns, T - c0);
        auto scope = a.scope();
        const Tensor hc = h.slice(1, c0, n);
        // Router logits and the shared expert's gate logit (row E).
        Tensor logits = alloc(a, DType::BF16, {E + 1, n});
        // The router's logits (and the shared expert's gate logit) and the shared expert's gate and up: one
        // encode of hc.
        Tensor sg = alloc(a, DType::BF16, {Is, n}), su = alloc(a, DType::BF16, {Is, n});
        shared(a, hc,
               {{&p.router_shared_gate, 0, E + 1, &logits, 0}, {&p.shared_gate_up, 0, Is, &sg, 0},
                {&p.shared_gate_up, Is, Is, &su, 0}},
               s);
        Tensor weights = alloc(a, DType::FP32, {k, n}), ids = alloc(a, DType::I32, {k, n});
        const std::int32_t C = ops::lossless_moe_columns(n, k, E);
        Tensor counts = alloc(a, DType::I32, {E}), tiles = alloc(a, DType::I32, {C / 64});
        Tensor slots = alloc(a, DType::I32, {k, n}), tokens = alloc(a, DType::I32, {C});
        Tensor gathered = alloc(a, DType::BF16, {H, C});
        if constexpr (kRuns<A>) {
            ops::lossless_router(logits, E, k, weights, ids, s);
            ops::lossless_moe_route(ids, E, counts, tiles, slots, tokens, s);
            ops::lossless_moe_gather(hc, tokens, tiles, gathered, s);
        }
        // Routed experts: expert e's gate rows at e 2I, up rows at e 2I + I, down rows at e H.
        Tensor eg = alloc(a, DType::BF16, {I, C}), eu = alloc(a, DType::BF16, {I, C});
        grouped(a, gathered, p.routed_gate_up, tiles, 2 * I, 0, I, eg, s);
        grouped(a, gathered, p.routed_gate_up, tiles, 2 * I, I, I, eu, s);
        if constexpr (kRuns<A>) { ops::lossless_silu_mul(eg, eu, eg, s); }
        Tensor ed = alloc(a, DType::BF16, {H, C});
        grouped(a, eg, p.routed_down, tiles, H, 0, H, ed, s);
        // The shared expert.
        if constexpr (kRuns<A>) { ops::lossless_silu_mul(sg, su, sg, s); }
        Tensor sd = alloc(a, DType::BF16, {H, n});
        linear(a, sg, p.shared_down, 0, H, sd, 0, s);
        if constexpr (kRuns<A>) {
            Tensor xc = x.slice(1, c0, n);
            ops::lossless_moe_combine(ed, weights, ids, slots, &sd, &logits, E, xc, s);
        }
        if constexpr (!kRuns<A>) { break; }
    }
}

// BF16 rows [K, N] as a contiguous BF16 Weight [N, K].
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

// An offloaded MoE layer (Qwen3_5MoeSparseMoeBlock over the 122B's offloaded banks): the router and the
// shared expert's gate logit in BF16, the router's top-k and weights, the routed experts and the shared
// expert (the dispatch's last job) through the offloaded op's lossless pass, which computes every expert
// on the GPU in serial staging passes (its records from frames, staging or the host), and the combine
// into the residual. The routes go to the Program's route log, so generation keeps the cache warm.
template <class A>
void offloaded_moe(A& a, const OffloadedMoeParameters& p, const Tensor& h, Tensor& x, OffloadedExperts* experts,
                   const OffloadedCall* call, cudaStream_t s) {
    const std::int32_t H = h.ne[0], T = h.ne[1];
    const auto E = static_cast<std::int32_t>(p.experts), k = static_cast<std::int32_t>(p.top_k);
    const ops::offloaded_moe::ExpertShape shape = p.bank->shape();
    if constexpr (kRuns<A>) {
        if (experts == nullptr) { throw std::logic_error("lossless: an offloaded MoE layer needs the Program's experts"); }
    }
    const Weight router = bf16_rows(p.router), score = bf16_rows(p.shared_score);
    // A streamed pass (ungated: no CPU split on this route): the stream copied every expert missing
    // when the pass began into its ring, so every column slice reads them there.
    runtime::expert_offload::ExpertStream* streaming = nullptr;
    if constexpr (kRuns<A>) {
        if (call != nullptr && call->streamed && experts->stream != nullptr && experts->stream->active() &&
            !experts->stream->gated()) {
            streaming = experts->stream;
            streaming->before_experts(p.layer, s);
        }
    }
    // A dense shared expert (an FP8 checkpoint's) is computed here; a shared expert bank is the
    // dispatch's last job.
    const std::int32_t sj = p.shared ? 0 : 1;
    const std::int32_t Is = p.shared ? p.shared->gate_up.weight.n / 2 : 0;
    for (std::int32_t c0 = 0; c0 < T; c0 += kFfnColumns) {
        const std::int32_t n = std::min(kFfnColumns, T - c0);
        auto scope = a.scope();
        const Tensor hc = h.slice(1, c0, n);
        // Router logits and the shared expert's gate logit (row E), and a dense shared expert's gate and
        // up: one encode of hc.
        Tensor logits = alloc(a, DType::BF16, {E + 1, n});
        Tensor sg, su, sd;
        if (p.shared) {
            sg = alloc(a, DType::BF16, {Is, n});
            su = alloc(a, DType::BF16, {Is, n});
            const Weight& gu = p.shared->gate_up.weight;
            shared(a, hc, {{&router, 0, E, &logits, 0}, {&score, 0, 1, &logits, E}, {&gu, 0, Is, &sg, 0},
                           {&gu, Is, Is, &su, 0}},
                   s);
            if constexpr (kRuns<A>) { ops::lossless_silu_mul(sg, su, sg, s); }
            sd = alloc(a, DType::BF16, {H, n});
            linear(a, sg, p.shared->down.weight, 0, H, sd, 0, s);
        } else {
            shared(a, hc, {{&router, 0, E, &logits, 0}, {&score, 0, 1, &logits, E}}, s);
        }
        Tensor weights = alloc(a, DType::FP32, {k, n}), ids = alloc(a, DType::I32, {k, n});
        Tensor gate    = alloc(a, DType::FP32, {n});
        const std::int32_t entries = (k + sj) * n, max_jobs = std::min(E, k * n) + sj;
        const DeviceSpan dispatch_bytes = a.alloc_bytes(ops::moe_dispatch_bytes(E + sj, entries));
        Tensor outputs = alloc(a, DType::BF16, {H, entries});
        const DeviceSpan expert_ws = a.alloc_bytes(ops::moe_experts_workspace_bytes(
            shape, max_jobs, entries, n, ops::offloaded_moe::ExpertActivation::kA16, ops::offloaded_moe::WideA16::kLossless));
        if constexpr (kRuns<A>) {
            ops::lossless_router(logits, E, k, weights, ids, s);
            ops::MoeRouting routing{ids, weights, gate};
            ops::MoeDispatch dispatch = ops::carve_moe_dispatch(dispatch_bytes.data, E + sj, entries);
            std::int32_t* route_log   = experts->route_log != nullptr && static_cast<std::size_t>(k) * n <= experts->route_stride
                                            ? experts->route_log + static_cast<std::size_t>(p.layer) * experts->route_stride
                                            : nullptr;
            ops::moe_dispatch(routing, E, dispatch, route_log, s, /*shared_job=*/sj == 1);
            ops::MoeExpertSource source = experts->sources.at(p.layer);
            source.cpu      = ops::MoeCpuChannel{}; // the lossless pass computes every expert on the GPU
            source.wide_a16 = ops::offloaded_moe::WideA16::kLossless;
            if (streaming != nullptr) {
                source.prefetched      = streaming->slots(p.layer);
                source.prefetch_base   = streaming->base();
                source.prefetch_stride = streaming->pitch();
            }
            ops::moe_experts(hc, dispatch, source, k, max_jobs, expert_ws.data, outputs, s, /*wait_for_cpu=*/false, false);
            // Over the experts in ascending id s = bf16(s + bf16(y_e w_e)), then the gated shared expert, then
            // the residual.
            const Tensor routed = outputs.slice(1, 0, k * n);
            const Tensor shared_out = p.shared ? sd : outputs.slice(1, k * n, n);
            Tensor xc = x.slice(1, c0, n);
            ops::lossless_moe_combine(routed, weights, ids, Tensor{}, &shared_out, &logits, E, xc, s);
        }
        if constexpr (!kRuns<A>) { break; } // every pass has the first one's allocations or fewer
    }
    if (streaming != nullptr && call->release_stream) { streaming->after_experts(p.layer, s); }
}

template <class A>
void layer(A& a, const BlockParameters& block, const TextConfig& config, MixerKind kind, const Tensor& rope_table,
           Tensor& x, cudaStream_t s, const LosslessCache* cache, const LosslessDesign& design,
           OffloadedExperts* experts, const OffloadedCall* call) {
    if (design.active && cache != nullptr) { throw std::invalid_argument("the design reference scores whole windows"); }
    if (kind == MixerKind::FullAttention) {
        attention_layer(a, block, config, rope_table, x, s, cache, design);
    } else {
        gdn_layer(a, block, config, x, s, cache, design);
    }
    if constexpr (kRuns<A>) {
        if (g_residual_tap) { g_residual_tap("mixer", x, s); }
    }
    auto scope = a.scope();
    Tensor h = alloc(a, DType::BF16, {x.ne[0], x.ne[1]});
    if constexpr (kRuns<A>) { ops::lossless_rmsnorm(x, block.post_attention_norm, config.rms_norm_eps, h, s); }
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&block.ffn)) {
        // The sparse MoE banks are 16-bit only (no declared quantization).
        sparse_moe(a, *moe, std::get<MoeConfig>(config.ffn), h, x, s);
    } else if (const auto* offloaded = std::get_if<OffloadedMoeParameters>(&block.ffn)) {
        // W4A16 experts (no declared activation step: the design reference computes them as --lossless).
        offloaded_moe(a, *offloaded, h, x, experts, call, s);
    } else {
        dense_mlp(a, std::get<DenseParameters>(block.ffn), h, x, s, design);
    }
}

std::int32_t rotary_pairs(const TextConfig& config) { return dimension(config.rope_parameters->rotary_dim) / 2; }

} // namespace

void set_lossless_residual_tap(ResidualTap tap) { g_residual_tap = std::move(tap); }

Tensor lossless_rope_window(const TextConfig& config, const Tensor& positions, WorkspaceArena& workspace,
                            cudaStream_t stream) {
    const auto& rope = *config.rope_parameters;
    if (rope.yarn_factor != 1.0F) { throw std::invalid_argument("the lossless route has no YaRN scaling"); }
    const std::int32_t pairs = rotary_pairs(config), tokens = positions.ne[0];
    Tensor inv_freq = workspace.alloc(DType::FP32, {pairs});
    Tensor table    = workspace.alloc(DType::BF16, {pairs, tokens, 2});
    ops::lossless_rope_frequencies(rope.rope_theta, dimension(rope.rotary_dim), inv_freq, stream);
    ops::lossless_rope_table(positions, inv_freq, table, stream);
    return table;
}

std::size_t lossless_rope_window_bytes(const TextConfig& config, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    const std::int32_t pairs = rotary_pairs(config);
    (void)layout.alloc(DType::FP32, {pairs});
    (void)layout.alloc(DType::BF16, {pairs, tokens, 2});
    return layout.peak_bytes(1);
}

void lossless_layer(const BlockParameters& block, const TextConfig& config, MixerKind kind, const Tensor& rope_table,
                    Tensor& x, WorkspaceArena& workspace, cudaStream_t stream, const LosslessCache* cache,
                    const LosslessDesign& design, OffloadedExperts* experts, const OffloadedCall* call) {
    layer(workspace, block, config, kind, rope_table, x, stream, cache, design, experts, call);
}

std::size_t lossless_layer_workspace_bytes(const BlockParameters& block, const TextConfig& config, MixerKind kind,
                                           std::int32_t tokens, const LosslessCache* cache,
                                           const LosslessDesign& design) {
    WorkspaceLayoutBuilder layout;
    Tensor x(nullptr, DType::BF16, {dimension(config.hidden_size), tokens});
    layer(layout, block, config, kind, Tensor{}, x, nullptr, cache, design, nullptr, nullptr);
    return layout.peak_bytes(1);
}

void lossless_final_norm(const Tensor& x, const Tensor& weight, const TextConfig& config, Tensor& out,
                         cudaStream_t stream) {
    ops::lossless_rmsnorm(x, weight, config.rms_norm_eps, out, stream);
}

void lossless_head(const Tensor& hidden, const Weight& head, Tensor& logits, WorkspaceArena& workspace,
                   cudaStream_t stream) {
    ops::lossless_linear(hidden, head, 0, head.n, logits, 0, workspace, stream);
}

std::size_t lossless_head_workspace_bytes(const Weight& head, std::int32_t tokens) {
    return ops::lossless_linear_workspace_bytes(head, head.n, tokens);
}

} // namespace infernix::models::qwen3_5::execution
