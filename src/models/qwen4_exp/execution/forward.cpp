#include "models/qwen4_exp/execution/forward.h"

#include "core/layout.h"
#include "infernix/ops/causal_conv1d_silu.h"
#include "infernix/ops/embedding.h"
#include "infernix/ops/gated_delta_net.h"
#include "infernix/ops/gated_rmsnorm.h"
#include "infernix/ops/gdn_gating.h"
#include "infernix/ops/ggml_encode.h"
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
#include "infernix/ops/screened_head.h"
#include "infernix/ops/sigmoid_mul.h"
#include "infernix/ops/silu_mul.h"
#include "infernix/ops/speculative_round.h"
#include "runtime/contract/execution.h"

#include <algorithm>
#include <array>
#include <vector>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace infernix::models::qwen4_exp::execution {
namespace {

std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }

// The screened heads' selection: the rows within 5 logits of a column's best screen score, at most 64,
// rescored exactly (Basalt's screened draft head: the exact head's pick in 99.93 % of 1,500 real inputs).
constexpr ops::ScreenedHeadPolicy kScreenedHeadPolicy{.margin = 5.0F, .rescored = 64};

ops::offloaded_moe::ExpertActivation expert_activation(const TextConfig& c) {
    return c.moe.a16_experts ? ops::offloaded_moe::ExpertActivation::kA16 : ops::offloaded_moe::ExpertActivation::kA4;
}

// The forward's steps run on a WorkspaceArena (kRuns: they launch) or plan their allocations on a
// WorkspaceLayoutBuilder (Forward::plan_run and its siblings): the same allocation code, nothing launched.
template <class A>
constexpr bool kRuns = std::is_same_v<A, WorkspaceArena>;

// An Op's scratch in a plan: its capacity query's bytes in a scope (every Op the forward calls with the arena
// returns its scratch before it returns).
template <class A>
void plan_scratch(A& a, std::size_t bytes) {
    auto scope = a.scope();
    (void)a.alloc_bytes(bytes);
}

// A planned tensor where an Op's predicate reads it: linear_qsa_group_supported requires its parts' data, which a
// plan's tensors do not have.
Tensor with_data(Tensor t) {
    if (t.data == nullptr) { t.data = reinterpret_cast<void*>(std::uintptr_t{256}); }
    return t;
}

// `routes` carries the selected fast paths' linear route flags (Forward::dense_routes) to the
// projections they serve; a problem without a route for a flag runs its exact route. A GGUF import's
// projection runs the ggml route (ggml-bf16-prefill's BF16 tiles at prefill widths, else exact).
template <class A>
void project(const Tensor& x, const LinearParameters& p, Tensor& out, A& work, cudaStream_t s,
             ops::LinearRoutes routes = ops::LinearRoutes::Exact) {
    if constexpr (kRuns<A>) {
        if (p.is_ggml()) {
            ops::ggml_linear(x, p.ggml, out, p.input_perm, p.perm_block, work, s, routes);
            return;
        }
        ops::linear(x, p.weight, out, p.policy, work, s, routes);
    } else {
        (void)out, (void)s, (void)routes;
        plan_scratch(work, p.is_ggml() ? ops::ggml_linear_workspace_bytes(x.ne[0], x.ne[1])
                                       : ops::linear_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k,
                                                                              p.policy, x.ne[1], x.ne[1]));
    }
}

// Whether a GGUF projection of `columns` columns of K = k takes its input encoded by the input's producer
// (infernix/ops/ggml_encode.h; design §19.3.39) instead of encoding it itself: the exact narrow route, and no
// activation scan to feed.
bool producer_encodes(const LinearParameters& p, std::int32_t k, std::int32_t columns, ops::LinearRoutes routes,
                      bool scanning) {
    return !scanning && p.is_ggml() && ops::ggml_encode_length(k) && (p.input_perm == nullptr || p.perm_block % 8 == 0) &&
           ops::ggml_linear_takes_encoded(k, columns, routes);
}

template <class A>
ops::GgmlEncodedColumns alloc_encoded(A& work, std::int32_t k, std::int32_t columns) {
    return {work.alloc(DType::I32, {ops::ggml_encoded_column_words(k), columns}), work.alloc(DType::I32, {columns}), k};
}

// A GGUF import's projection split among several outputs (one encoding of x).
template <class A>
void project_ggml_split(const Tensor& x, std::span<const ops::GgmlSegment> segments, std::span<const std::int32_t> bounds,
                        std::span<Tensor* const> parts, A& work, cudaStream_t s, ops::LinearRoutes routes) {
    if constexpr (kRuns<A>) {
        const auto split = ops::ggml_split_segments(segments, x.ne[0], bounds);
        ops::ggml_linear(x, split, parts, nullptr, 0, work, s, routes);
    } else {
        (void)segments, (void)bounds, (void)parts, (void)s, (void)routes;
        plan_scratch(work, ops::ggml_linear_workspace_bytes(x.ne[0], x.ne[1]));
    }
}

// Token rows: BF16 (packed at load or not), or a GGUF import's IQ4_XS rows decoded exactly (an id outside the vocabulary
// embeds zeros and writes kErrorTokenId to `error`).
void embed_tokens(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t s, std::uint32_t* error) {
    const DeviceFault fault{.word = error, .value = kErrorTokenId};
    if (table.layout == QuantLayout::GgmlRows) {
        ops::ggml_embedding(ids, table, out, s, fault);
    } else {
        ops::embedding(ids, table, out, s, fault);
    }
}

// FP32 logits of the text head (BF16, packed BF16, Q8 or a GGUF import's ggml rows) for the columns of x.
template <class A>
void text_head_logits(const Parameters& parameters, const Tensor& x, Tensor& logits, A& work, cudaStream_t s) {
    if constexpr (!kRuns<A>) {
        (void)logits, (void)s;
        if (parameters.output_head_weight && parameters.output_head_weight->is_ggml()) {
            plan_scratch(work, ops::ggml_linear_workspace_bytes(x.ne[0], x.ne[1]));
        }
    } else if (parameters.output_head_weight && parameters.output_head_weight->is_ggml()) {
        ops::ggml_linear(x, parameters.output_head_weight->ggml, logits, nullptr, 0, work, s, ops::LinearRoutes::Exact);
    } else if (parameters.output_head_weight) {
        ops::projection_fp32(x, parameters.output_head_weight->weight, logits, s);
    } else {
        const Tensor* head[] = {&parameters.output_head};
        ops::projection_fp32(x, head, logits, s);
    }
}

// The screened-lm-head fast path's logits for the columns of x (at most 16): every row from the head's
// 2-bit screen, and each column's near-best public rows with text_head_logits' bits (a BF16 or Q8 head).
void screened_text_head_logits(const Parameters& parameters, const Tensor& x, Tensor& logits, WorkspaceArena& work,
                               cudaStream_t s) {
    if (!parameters.output_head_screen || (parameters.output_head_weight && parameters.output_head_weight->is_ggml())) {
        throw std::logic_error("Qwen4Exp forward: a screened head round needs the head's screen and a BF16 or Q8 head");
    }
    const std::int32_t valid = dim(parameters.model.resources().public_token_count);
    if (parameters.output_head_weight) {
        ops::screened_head_logits(x, Tensor{}, parameters.output_head_weight->weight, *parameters.output_head_screen, valid,
                                  kScreenedHeadPolicy, logits, work, s);
    } else {
        ops::screened_head_logits(x, parameters.output_head, Weight{}, *parameters.output_head_screen, valid,
                                  kScreenedHeadPolicy, logits, work, s);
    }
}

// The shared expert's SiLU(gate) * up: one fused call for Q8 weights (gate and up stay FP32); BF16
// weights (recipe A) project, split and multiply.
template <class A>
void shared_swiglu(const Tensor& x, const LinearParameters& gate_up, Tensor& out, A& work, cudaStream_t s,
                   ops::LinearRoutes routes = ops::LinearRoutes::Exact) {
    // Q8 (Dense8) at every width; BF16 (the bit-exact artifact) on its register-streamed route at decode
    // and verification widths, the tuned GEMM then SwiGLU beyond.
    if (gate_up.weight.qtype == QType::Q8_G32_FP16 || (gate_up.weight.qtype == QType::BF16 && x.ne[1] <= 16)) {
        if constexpr (kRuns<A>) {
            ops::linear_swiglu(x, gate_up.weight, out, gate_up.policy, work, s);
        } else {
            plan_scratch(work, ops::linear_swiglu_workspace_capacity_bytes(gate_up.weight.qtype, gate_up.weight.n,
                                                                           gate_up.weight.k, gate_up.policy, x.ne[1],
                                                                           x.ne[1]));
        }
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
        project_ggml_split(x, gate_up.ggml, bounds, parts, work, s, routes);
        if constexpr (kRuns<A>) { ops::silu_mul(gate, up, out, s); }
        return;
    }
    // silu_mul reads the gate and up halves of each column in place (its strided-input route), so the
    // projection is not split into two tensors first; the arithmetic is the same.
    Tensor projected = work.alloc(DType::BF16, {2 * I, T});
    project(x, gate_up, projected, work, s, routes);
    if constexpr (kRuns<A>) { ops::silu_mul(projected.slice(0, 0, I), projected.slice(0, I, I), out, s); }
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
      fast_(fast_paths), storage_(kv_.layers.empty() ? KvCacheStorage{} : kv_.layers.front().kv.storage),
      rope_(ops::prepare_rope(dim(config_.rope.rotary_dim), config_.rope.theta,
                              {config_.rope.yarn_factor, config_.max_position_embeddings})) {
    if (state_.gdn == nullptr || state_.gdn->layer_count() != config_.gdn_layers ||
        state_.qsa_tails.size() != config_.attention_layers || kv_.layers.size() != config_.attention_layers ||
        experts_.frames.size() != config_.num_hidden_layers) {
        throw std::invalid_argument("Qwen4Exp forward: state, KV or expert residency is incomplete");
    }
    if (experts_.split != nullptr) {
        const auto E   = static_cast<std::size_t>(config_.moe.experts);
        split_offsets_ = PinnedHostBuffer(2 * (E + 1) * sizeof(std::int32_t));
        split_columns_.assign(E, 0);
        split_cpu_.assign(E, 0);
        split_priced_.assign(E, 0);
        guess_columns_.assign(E, 0);
        CUDA_CHECK(cudaEventCreateWithFlags(&split_ready_, cudaEventDisableTiming | cudaEventBlockingSync));
    }
    if (const auto& dflash2 = parameters_.model.config().dflash2) { draft_taps_ = dflash2->target_layer_ids; }
    scan_ = ActivationScan::from_environment(config_.num_hidden_layers);
}

Forward::Forward(PlanOnly, const Parameters& parameters, DeviceContext& device, WorkspaceArena& work,
                 std::int32_t max_context, ForwardFastPaths fast_paths, KvCacheStorage storage)
    : parameters_(parameters), config_(parameters.model.config().text), device_(device), work_(work),
      max_context_(max_context), fast_(fast_paths), storage_(storage),
      rope_(ops::prepare_rope(dim(config_.rope.rotary_dim), config_.rope.theta,
                              {config_.rope.yarn_factor, config_.max_position_embeddings})) {
    if (const auto& dflash2 = parameters_.model.config().dflash2) { draft_taps_ = dflash2->target_layer_ids; }
    scan_ = ActivationScan::from_environment(config_.num_hidden_layers);
}

std::unique_ptr<Forward> Forward::planner(const Parameters& parameters, DeviceContext& device, WorkspaceArena& work,
                                          std::int32_t max_context, ForwardFastPaths fast_paths, KvCacheStorage storage) {
    return std::unique_ptr<Forward>(new Forward(PlanOnly{}, parameters, device, work, max_context, fast_paths, storage));
}

Forward::~Forward() {
    if (split_ready_ != nullptr) { (void)cudaEventDestroy(split_ready_); }
}

Forward::CallShape Forward::call_shape(const ForwardBatch& batch) const {
    return {.batch           = batch.batch,
            .width           = batch.width,
            .verify          = batch.verify != nullptr,
            .bf16_rows       = batch.ngram_format.value_or(config_.ple.table.format) == NgramRowFormat::Bf16,
            .trim_last_layer = batch.trim_last_layer};
}

ops::offloaded_moe::WideA16 Forward::expert_wide_a16(const MoeParameters& p) const {
    // --a8-activations' wide route (GGUF and NVFP4 W4A16 banks, beside their 8-bit narrow route) takes a call of
    // more than 128 columns first. The A4 residual routes exist for NVFP4 W4A16 banks only; their layouts carry no
    // x limbs, which the GGUF and block-FP8 wide routes read, so another bank keeps the exact layout whatever the
    // flag. ggml-bf16-prefill takes a GGUF bank's wide jobs in prefill calls onto the BF16 stage-decode tiles.
    const QType format   = p.bank->planes.gate_up_format;
    const bool ggml_bank = format != QType::NVFP4_MUL && format != QType::FP8_E4M3FN_B128;
    const bool nvfp4_a16 = p.bank->activation == ops::offloaded_moe::ExpertActivation::kA16 && format == QType::NVFP4_MUL;
    return fast_.a8_experts_wide && (ggml_bank || nvfp4_a16) ? ops::offloaded_moe::WideA16::kA8
           : fast_.expert_a4_terms != 0 && format == QType::NVFP4_MUL
               ? (fast_.expert_a4_terms == 3 ? ops::offloaded_moe::WideA16::kA4Residual3
                                             : ops::offloaded_moe::WideA16::kA4Residual)
           : ggml_bank && ops::has_route(dense_routes(), ops::LinearRoutes::GgmlBf16Wide)
               ? ops::offloaded_moe::WideA16::kGgmlBf16
               : ops::offloaded_moe::WideA16::kExact;
}

ops::offloaded_moe::NarrowA16 Forward::expert_narrow_a16(const MoeParameters& p) const {
    // expert-a4-residual-decode, else --a8-activations' 8-bit route (calls of at most 128 columns): an NVFP4 W4A16
    // bank's narrow jobs (the CPU service was created with the same choice, which moe_experts checks).
    const bool nvfp4_a16 = p.bank->activation == ops::offloaded_moe::ExpertActivation::kA16 &&
                           p.bank->planes.gate_up_format == QType::NVFP4_MUL;
    return nvfp4_a16 && fast_.expert_a4_narrow ? ops::offloaded_moe::NarrowA16::kA4Residual
           : nvfp4_a16 && fast_.a8_experts     ? ops::offloaded_moe::NarrowA16::kA8
                                               : ops::offloaded_moe::NarrowA16::kExact;
}

bool Forward::trimmed_routes(const MoeParameters& p, std::int32_t columns) const {
    // The op decides from the bank's arithmetic in this call (expert_source's bank kind, expert_wide_a16,
    // expert_narrow_a16, the GGUF 8-bit flag).
    ops::MoeExpertSource arithmetic{};
    arithmetic.activation = p.bank->activation;
    const QType format    = p.bank->planes.gate_up_format;
    if (format != QType::NVFP4_MUL) {
        arithmetic.ggml.gate_up_format = format;
        arithmetic.ggml.a8             = fast_.a8_experts && format != QType::FP8_E4M3FN_B128;
    }
    arithmetic.wide_a16   = expert_wide_a16(p);
    arithmetic.narrow_a16 = expert_narrow_a16(p);
    return ops::moe_routes_by_width(arithmetic, columns);
}

template <class F>
std::size_t Forward::memoized(int kind, std::int32_t a, std::int32_t b, std::int32_t c, const F& plan) {
    const auto key = std::make_tuple(kind, a, b, c);
    if (const auto it = plans_.find(key); it != plans_.end()) { return it->second; }
    if (config_.lossless) { throw std::logic_error("Qwen4Exp forward: the lossless route keeps its workspace formula"); }
    // The plan places prefill steps (dense_routes adds the prefill routes); the forward's own flag comes back after it,
    // whatever the plan throws.
    struct EagerScope {
        bool& flag;
        bool saved;
        ~EagerScope() { flag = saved; }
    } eager{eager_chunk_, std::exchange(eager_chunk_, true)};
    WorkspaceLayoutBuilder layout;
    plan(layout);
    return plans_[key] = layout.peak_bytes();
}

std::size_t Forward::plan_run(const PrefillPlan& call) {
    return memoized(0, call.columns, call.mtp_cells, (call.logits ? 1 : 0) | (call.may_gate ? 2 : 0),
                    [&](WorkspaceLayoutBuilder& a) {
                        const CallShape shape{.width = call.columns, .may_gate = call.may_gate};
                        const FinishShape finish{.mtp_cells = call.mtp_cells, .logit_columns = call.logits ? 1 : 0};
                        Tensor logits(nullptr, DType::FP32, {dim(config_.vocab_size), 1});
                        run_impl(a, nullptr, shape, finish, call.logits ? &logits : nullptr, nullptr);
                    });
}

std::size_t Forward::plan_split(std::int32_t head, std::int32_t tail) {
    return memoized(1, head, tail, 0, [&](WorkspaceLayoutBuilder& a) {
        Tensor logits(nullptr, DType::FP32, {dim(config_.vocab_size), 1});
        run_split_impl(a, nullptr, nullptr, CallShape{.width = head}, CallShape{.width = tail}, logits, Tensor{}, nullptr);
    });
}

std::size_t Forward::plan_walk_mixer(std::int32_t columns) {
    return memoized(2, columns, 0, 0, [&](WorkspaceLayoutBuilder& a) {
        const CallShape shape{.width = columns};
        const std::int32_t H = dim(config_.hidden_size), S = dim(config_.hc.streams);
        Tensor residual(nullptr, DType::BF16, {dim(config_.residual_width()), columns});
        Tensor xm(nullptr, DType::BF16, {H, columns});
        Tensor inject(nullptr, DType::FP32, {S, columns});
        embed_impl(a, nullptr, residual);
        for (std::uint32_t l = 0; l < config_.num_hidden_layers; ++l) {
            layer_mixer_impl(a, l, nullptr, shape, residual, xm, inject);
        }
    });
}

std::size_t Forward::plan_walk_moe(std::int32_t columns) {
    return memoized(3, columns, 0, 0, [&](WorkspaceLayoutBuilder& a) {
        const CallShape shape{.width = columns};
        const std::int32_t H = dim(config_.hidden_size), S = dim(config_.hc.streams);
        Tensor residual(nullptr, DType::BF16, {dim(config_.residual_width()), columns});
        const Tensor xm(nullptr, DType::BF16, {H, columns});
        const Tensor inject(nullptr, DType::FP32, {S, columns});
        for (std::uint32_t l = 0; l < config_.num_hidden_layers; ++l) {
            layer_moe_impl(a, l, nullptr, shape, xm, inject, residual, columns, 0, 0);
        }
    });
}

std::size_t Forward::plan_walk_finish(std::int32_t columns, std::int32_t mtp_cells, bool logits) {
    return memoized(4, columns, mtp_cells, logits ? 1 : 0, [&](WorkspaceLayoutBuilder& a) {
        Tensor residual(nullptr, DType::BF16, {dim(config_.residual_width()), columns});
        Tensor out(nullptr, DType::FP32, {dim(config_.vocab_size), 1});
        finish_impl(a, nullptr, FinishShape{.mtp_cells = mtp_cells, .logit_columns = logits ? 1 : 0}, residual,
                    logits ? &out : nullptr);
    });
}

void Forward::scan_input(const Tensor& x, std::uint32_t layer, ActivationScan::Site site) {
    if (scan_ != nullptr && eager_chunk_) { scan_->scan(x, layer, site, device_.stream); }
}

ops::LinearRoutes Forward::dense_routes() const noexcept {
    return eager_chunk_ ? fast_.dense_routes | fast_.prefill_routes : fast_.dense_routes;
}

std::size_t Forward::workspace_bytes(const TextConfig& c, std::int32_t columns, std::int32_t max_context,
                                     KvCacheStorage storage, ops::offloaded_moe::WideA16 wide_a16) {
    const std::size_t t = static_cast<std::size_t>(columns);
    const std::size_t h = c.hidden_size, w = c.residual_width(), k = c.moe.top_k;
    const std::size_t bf = 2;
    std::size_t bytes = 0;
    bytes += w * t * bf * 3;                                  // residual, normalized, mix logits
    bytes += (c.hc.rank + c.hc.streams) * t * (bf + 4) + c.hc.rank * t * bf;
    // A packed mixer's matrices decoded for a call past its fused route (ops::hyper_connection_mix).
    if (columns > 16) { bytes += (2 * std::size_t(c.hc.rank) + c.hc.streams) * w * bf + 512; }
    bytes += h * t * bf * 4;                                  // x, y, embedding, final
    bytes += std::size_t(c.ple.embed_dim) * t * bf * 2 + (w + h) * t * bf + w * t * bf * 2;
    bytes += std::size_t(c.gdn.conv_channels()) * t * bf * 2 + std::size_t(c.gdn.value_width()) * t * bf * 4;
    bytes += std::size_t(c.gdn.value_heads) * t * (bf + 4) * 2;
    const std::size_t qkv = 2 * std::size_t(c.attention.query_width()) + 2 * c.attention.key_width() +
                            std::size_t(c.qsa.index_heads + 1) * c.qsa.index_head_dim;
    bytes += qkv * t * bf * 2 + std::size_t(c.attention.query_width()) * t * bf * 2;
    bytes += ops::qsa_attention_workspace_bytes(qsa_geometry(c), columns, max_context, storage);
    if (storage == KvCacheStorage::Bf16Packed && columns > kQsaUnpackColumns) {
        bytes += ops::qsa_attention_unpack_bytes(qsa_geometry(c), max_context);
    }
    bytes += (c.moe.experts + 1) * t * 4 + k * t * 8 + t * 4;           // FP32 router logits, routing
    // A shared expert bank is the call's last job: one more expert and T more entries and outputs.
    const std::size_t sj = c.moe.shared_bank ? 1 : 0;
    bytes += ops::moe_dispatch_bytes(dim(c.moe.experts + sj), dim((k + sj) * t));
    // A gated call's copy-ahead guess (its routing and dispatch; gated calls are at most the CPU channel's width).
    const std::size_t g = std::min<std::size_t>(t, ops::offloaded_moe::kMaxCpuCallColumns);
    bytes += k * g * 8 + g * 4 + ops::moe_dispatch_bytes(dim(c.moe.experts), dim(k * g));
    bytes += h * (k + sj) * t * bf +
             ops::moe_experts_workspace_bytes(ops::offloaded_moe::kFlashNextShape, dim(c.moe.experts + sj),
                                              dim((k + sj) * t), std::max(columns, 1), expert_activation(c), wide_a16);
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
    eager_chunk_ = check_call(batch, logits);
    work_.reset();
}

bool Forward::check_call(const ForwardBatch& batch, const Tensor* logits) const {
    const std::int32_t T = batch.ids.ne[0];
    if (batch.verify != nullptr && (batch.width < 2 || batch.verify->gdn.spec.width != batch.width ||
                                    batch.verify->gdn.spec.record_capacity < batch.batch)) {
        throw std::invalid_argument("Qwen4Exp forward: verification records do not match the call");
    }
    if (batch.batch <= 0 || batch.width <= 0 || batch.batch * batch.width != T ||
        (logits != nullptr && (!head_logits_dtype(*logits) || logits->ne[1] != batch.logit_columns.ne[0] ||
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
    // Prefill chunks and forced tokens run eagerly; decode and verification rounds may be captured. A
    // streamed call is a prefill chunk at any width: a layer walk's one-token chunk run as a decode call
    // took its experts beside the span's stream ring and read 2.6-12 nats off its cold prefill
    // (candidates SA-12; the walk keeps a one-token last call on its own for coalescing).
    return batch.verify == nullptr && (batch.width > 1 || batch.stream);
}

void Forward::wait_layer(const ForwardBatch& batch, std::size_t entry) {
    for (const auto& events : batch.layer_waits) {
        if (!events.empty()) { CUDA_CHECK(cudaStreamWaitEvent(device_.stream, events[entry], 0)); }
    }
}

Forward::FinishShape Forward::finish_shape(const ForwardBatch& batch, const Tensor* logits) {
    const MtpChunk* chunk = batch.mtp_chunk;
    return {.mtp_cells     = chunk != nullptr ? (chunk->prepend ? 1 : 0) + chunk->columns : 0,
            .logit_columns = logits != nullptr ? batch.logit_columns.ne[0] : 0,
            .screened_head = batch.screened_head};
}

Forward::AttentionShape Forward::attention_shape(const AttentionCall& call) {
    return {.batch       = call.batch,
            .width       = call.width,
            .kv_only     = call.kv_only,
            .key_records = call.key_records.data != nullptr,
            .storage     = call.layer.kv.storage};
}

void Forward::run(const ForwardBatch& batch, Tensor& logits, const ForwardTap* tap) {
    begin_call(batch, &logits);
    run_impl(work_, &batch, call_shape(batch), finish_shape(batch, &logits), &logits, tap);
}

template <class A>
void Forward::run_impl(A& a, const ForwardBatch* batch, const CallShape& shape, const FinishShape& finish,
                       Tensor* logits, const ForwardTap* tap) {
    Tensor residual = a.alloc(DType::BF16, {dim(config_.residual_width()), shape.batch * shape.width});
    embed_impl(a, batch, residual);
    for (std::uint32_t l = 0; l < config_.num_hidden_layers; ++l) {
        layer_impl(a, l, batch, shape, residual, tap);
    }
    finish_impl(a, batch, finish, residual, logits);
}

void Forward::run_split(const ForwardBatch& head, const ForwardBatch& tail, Tensor& logits,
                        const Tensor& draft_features, const std::function<void(std::uint32_t)>& at_split) {
    for (const ForwardBatch* chunk : {&head, &tail}) {
        if (config_.lossless || chunk->batch != 1 || chunk->verify != nullptr || chunk->mtp_chunk != nullptr ||
            chunk->stream || chunk->decode_round || chunk->residual_out.data != nullptr ||
            chunk->draft_features.data != nullptr) {
            throw std::invalid_argument("Qwen4Exp forward: a pair call takes two plain one-row prefill chunks");
        }
    }
    // Each chunk's steps run as its own call's would: eager or not by its own width (check_call).
    const bool head_eager = check_call(head, nullptr);
    const bool tail_eager = check_call(tail, &logits);
    if (!head_eager || !tail_eager) {
        throw std::invalid_argument("Qwen4Exp forward: a pair call's chunks are prefill chunks of two or more columns");
    }
    eager_chunk_ = true;
    work_.reset();
    run_split_impl(work_, &head, &tail, call_shape(head), call_shape(tail), logits, draft_features, &at_split);
}

// Both chunks are eager (run_split checks it), so eager_chunk_ stays set throughout.
template <class A>
void Forward::run_split_impl(A& a, const ForwardBatch* head, const ForwardBatch* tail, const CallShape& head_shape,
                             const CallShape& tail_shape, Tensor& logits, const Tensor& draft_features,
                             const std::function<void(std::uint32_t)>* at_split) {
    const std::int32_t T0 = head_shape.width, T1 = tail_shape.width, T = T0 + T1;
    const std::int32_t H = dim(config_.hidden_size), S = dim(config_.hc.streams);
    const std::uint32_t L = config_.num_hidden_layers;
    Tensor residual = a.alloc(DType::BF16, {dim(config_.residual_width()), T});
    Tensor xm       = a.alloc(DType::BF16, {H, T});
    Tensor inject   = a.alloc(DType::FP32, {S, T});
    Tensor residual_head = residual.slice(1, 0, T0), residual_tail = residual.slice(1, T0, T1);
    Tensor xm_head = xm.slice(1, 0, T0), xm_tail = xm.slice(1, T0, T1);
    Tensor inject_head = inject.slice(1, 0, T0), inject_tail = inject.slice(1, T0, T1);
    embed_impl(a, head, residual_head);
    embed_impl(a, tail, residual_tail);
    for (std::uint32_t l = 0; l < L; ++l) {
        layer_mixer_impl(a, l, head, head_shape, residual_head, xm_head, inject_head);
        if constexpr (kRuns<A>) { (*at_split)(l); }
        layer_mixer_impl(a, l, tail, tail_shape, residual_tail, xm_tail, inject_tail);
        // One MoE over both chunks: its routing, routed experts and combine act column by column, and its router and
        // shared expert, whose routes go by width (a Q8 projection is SIMT to 8 columns and tensor-core beyond), run
        // on each chunk on their own, so every column has its own call's bits. The last layer's runs over the tail's
        // trailing columns alone where the tail's own call would trim it (YC-01; its dense shared expert over the tail
        // chunk, as that call runs it): no other final residual is read.
        const std::int32_t kept = l + 1 == L && tail_shape.trim_last_layer ? last_layer_kept(T1) : 0;
        if (kept > 0) {
            const Tensor xm_kept     = xm.slice(1, T - kept, kept);
            const Tensor inject_kept = inject.slice(1, T - kept, kept);
            Tensor residual_kept     = residual.slice(1, T - kept, kept);
            layer_moe_impl(a, l, tail, tail_shape, xm_kept, inject_kept, residual_kept, -1, T - kept, 0,
                           TrimmedCall{.shared = &xm_tail, .routes = &xm, .routes_chunk = T0});
        } else {
            layer_moe_impl(a, l, tail, tail_shape, xm, inject, residual, -1, 0, T0);
        }
        if constexpr (kRuns<A>) {
            if (draft_features.data != nullptr) {
                const std::int32_t n = draft_features.ne[1];
                if (n > T) { throw std::invalid_argument("Qwen4Exp forward: DFlash2 features exceed the pair call"); }
                capture_draft_features(l, residual.slice(1, T - n, n), draft_features);
            }
        }
    }
    if constexpr (kRuns<A>) {
        finish_impl(a, head, finish_shape(*head, nullptr), residual_head, nullptr);
        (*at_split)(L);
        finish_impl(a, tail, finish_shape(*tail, &logits), residual_tail, &logits);
    } else {
        (void)draft_features, (void)at_split;
        finish_impl(a, head, FinishShape{}, residual_head, nullptr);
        finish_impl(a, tail, FinishShape{.logit_columns = 1}, residual_tail, &logits);
    }
}

void Forward::embed(const ForwardBatch& batch, Tensor& residual) { embed_impl(work_, &batch, residual); }

template <class A>
void Forward::embed_impl(A& a, const ForwardBatch* batch, Tensor& residual) {
    const std::int32_t T = residual.ne[1], H = dim(config_.hidden_size);
    auto scope = a.scope();
    Tensor x0  = a.alloc(DType::BF16, {H, T});
    if constexpr (kRuns<A>) {
        const cudaStream_t s = device_.stream;
        embed_tokens(batch->ids, parameters_.token_embedding, x0, s, experts_.error);
        if (batch->vision != nullptr && batch->vision->columns.data != nullptr) {
            ops::scatter(batch->vision->embeddings, batch->vision->columns, x0, s);
        }
        ops::hyper_connection_expand(x0, dim(config_.hc.streams), residual, s);
    } else {
        (void)batch;
    }
}

void Forward::layer(std::uint32_t layer, const ForwardBatch& batch, Tensor& residual, const ForwardTap* tap) {
    layer_impl(work_, layer, &batch, call_shape(batch), residual, tap);
}

template <class A>
void Forward::layer_impl(A& a, std::uint32_t layer, const ForwardBatch* batch, const CallShape& shape, Tensor& residual,
                         const ForwardTap* tap) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = residual.ne[1];
    const auto& block    = parameters_.layers[layer];
    if constexpr (kRuns<A>) {
        wait_layer(*batch, layer);
        if (config_.lossless) {
            if (tap != nullptr) { throw std::invalid_argument("Qwen4Exp forward: the lossless route takes no taps"); }
            check_lossless_call(*batch);
            try {
                lossless_layer(layer, *batch, residual);
            } catch (const std::exception&) {
                runtime::rethrow_with_context("qwen4_exp/layers/" + std::to_string(layer) + ": ");
            }
            return;
        }
    }
    try {
        auto scope = a.scope();
        if (block.ple) { ple_impl(a, *block.ple, residual, batch, shape); }
        Tensor inject = a.alloc(DType::FP32, {dim(config_.hc.streams), T});
        Tensor xa_a8;
        Tensor xa     = mix_mixer_input_impl(a, block, residual, inject, xa_a8);
        Tensor y      = mixer_impl(a, layer, batch, shape, xa, xa_a8.data != nullptr ? &xa_a8 : nullptr);
        // The attention half's inject, then the MLP half's mix. One fused Op folding the inject into the
        // mixer's reads (candidates §7.3 D6) measured flat (removed 2026-10-28).
        if constexpr (kRuns<A>) { ops::hyper_connection_inject(y, inject, residual, s); }
        Tensor xm = mix_impl(a, block.mlp_hc, residual, &inject);
        // YC-01: the last layer's MoE over the trailing columns whose final residuals are read (their logits). The
        // mix stays whole: a layer walk mixes each chunk whole too, and its MoE then takes the same columns.
        const std::int32_t kept =
            tap == nullptr && shape.trim_last_layer && layer + 1 == config_.num_hidden_layers ? last_layer_kept(T) : 0;
        Tensor ym;
        if (kept > 0) {
            Tensor xm_kept       = xm.slice(1, T - kept, kept);
            Tensor inject_kept   = inject.slice(1, T - kept, kept);
            Tensor residual_kept = residual.slice(1, T - kept, kept);
            // The dense shared expert over the whole call, as the untrimmed call runs it.
            (void)moe_impl(a, block.moe, xm_kept, layer, nullptr, batch, shape, -1, &inject_kept, &residual_kept, T - kept,
                           nullptr, 0, TrimmedCall{.shared = &xm, .routes = &xm});
        } else {
            // The combine injects its output directly unless a tap records it.
            const bool fused_inject = tap == nullptr || tap->moe_outputs == nullptr;
            ym = moe_impl(a, block.moe, xm, layer,
                          tap != nullptr && tap->routes != nullptr ? &tap->routes->at(layer) : nullptr, batch, shape, -1,
                          fused_inject ? &inject : nullptr, fused_inject ? &residual : nullptr, 0, &residual, 0);
            if constexpr (kRuns<A>) {
                if (!fused_inject) { ops::hyper_connection_inject(ym, inject, residual, s); }
            }
        }
        if constexpr (kRuns<A>) {
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
        }
    } catch (const std::exception&) {
        runtime::rethrow_with_context("qwen4_exp/layers/" + std::to_string(layer) + ": ");
    }
    if constexpr (kRuns<A>) {
        if (batch->draft_features.data != nullptr) {
            const std::int32_t n = batch->draft_features.ne[1];
            if (n > T) { throw std::invalid_argument("Qwen4Exp forward: DFlash2 features exceed the call"); }
            capture_draft_features(layer, residual.slice(1, T - n, n), batch->draft_features);
        }
        if (tap != nullptr && tap->residuals != nullptr) {
            Tensor& copy = tap->residuals->at(layer);
            if (cudaMemcpyAsync(copy.data, residual.data, residual.bytes(), cudaMemcpyDeviceToDevice, s) !=
                cudaSuccess) {
                throw std::runtime_error("Qwen4Exp forward: residual tap copy failed");
            }
        }
    } else {
        (void)s;
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

// The GGUF input projection call of a layer's mixer (gdn() and attention() make the same one): its segments, or none
// when the mixer projects otherwise (another format, an input permutation).
std::vector<ops::GgmlSegment> Forward::mixer_input_segments(const BlockParameters& block) const {
    std::vector<ops::GgmlSegment> segments;
    if (const auto* g = std::get_if<GdnParameters>(&block.mixer)) {
        if (!g->projection.is_ggml() || g->projection.input_perm != nullptr) { return segments; }
        segments = g->projection.ggml;
        if (g->control.is_ggml() && g->control.input_perm == nullptr) {
            segments.insert(segments.end(), g->control.ggml.begin(), g->control.ggml.end());
        }
        return segments;
    }
    const auto& a = std::get<AttentionParameters>(block.mixer);
    if (!a.projection.is_ggml() || a.projection.input_perm != nullptr ||
        (a.index_projection && (!a.index_projection->is_ggml() || a.index_projection->input_perm != nullptr))) {
        return segments;
    }
    segments = a.projection.ggml;
    if (a.index_projection) { segments.insert(segments.end(), a.index_projection->ggml.begin(), a.index_projection->ggml.end()); }
    return segments;
}

// The attention half's mix. Under --a8-activations, where the mixer's GGUF input projection runs on the 8-bit route and
// the fused mix serves the call, the mix also writes the projection's 8-bit columns into x_a8 (the same words the
// projection would encode, so the same bits; no encoding launch); x_a8 stays empty otherwise.
template <class A>
Tensor Forward::mix_mixer_input_impl(A& a, const BlockParameters& block, Tensor& residual, Tensor& inject, Tensor& x_a8) {
    const auto& p        = block.attn_hc;
    const std::int32_t T = residual.ne[1], H = dim(config_.hidden_size);
    if (scan_ == nullptr && !eager_chunk_) {
        if constexpr (kRuns<A>) {
            const std::vector<ops::GgmlSegment> segments = mixer_input_segments(block);
            if (!segments.empty() && ops::ggml_a8_takes_encoded(segments, H, T, dense_routes(), nullptr, 0)) {
                Tensor x     = a.alloc(DType::BF16, {H, T});
                Tensor words = a.alloc(DType::I32, {ops::ggml_a8_encoded_column_words(segments, H), T});
                if (ops::hyper_connection_mix_a8_encoded(residual, p.norm, p.down.weight, p.up.weight,
                                                         dim(config_.hc.streams), dim(config_.hc.rank),
                                                         config_.rms_norm_eps, x, p.combine ? &inject : nullptr, words,
                                                         ops::ggml_a8_sums(segments), a, device_.stream)) {
                    x_a8 = words;
                    return x;
                }
                return mix_impl(a, p, residual, &inject, &x);
            }
        } else {
            (void)x_a8, (void)T, (void)H;
            throw std::logic_error("Qwen4Exp forward: a plan covers prefill calls");
        }
    }
    return mix_impl(a, p, residual, &inject);
}

template <class A>
Tensor Forward::mixer_impl(A& a, std::uint32_t layer, const ForwardBatch* batch, const CallShape& shape, const Tensor& xa,
                           const Tensor* xa_a8) {
    const auto& block           = parameters_.layers[layer];
    const std::uint32_t compact = config_.compact_layer_indices[layer];
    scan_layer_                 = layer;
    if (config_.layer_types[layer] == MixerKind::Gdn) {
        return gdn_impl(a, std::get<GdnParameters>(block.mixer), xa, compact, batch, shape, xa_a8);
    }
    if constexpr (kRuns<A>) {
        const std::int32_t T = xa.ne[1];
        // A verification call records the raw index keys for the tail commit.
        const AttentionCall call{.layer       = kv_.layers.at(compact),
                                 .tails       = state_.qsa_tails.at(compact),
                                 .key_records = batch->verify != nullptr
                                                    ? batch->verify->qsa_keys.slice(3, static_cast<std::int32_t>(compact), 1)
                                                          .view({dim(config_.qsa.index_head_dim), T})
                                                    : Tensor{},
                                 .update_tails     = batch->verify == nullptr,
                                 .positions        = batch->positions,
                                 .rope_positions   = batch->rope_positions,
                                 .block_start_rope = batch->block_start_rope,
                                 .slots            = batch->slots,
                                 .table_rows       = batch->table_rows,
                                 .batch            = batch->batch,
                                 .width            = batch->width};
        return attention_impl(a, std::get<AttentionParameters>(block.mixer), xa, &call, attention_shape(call), xa_a8);
    } else {
        (void)batch;
        return attention_impl(a, std::get<AttentionParameters>(block.mixer), xa, nullptr,
                              AttentionShape{.batch = shape.batch, .width = shape.width, .storage = storage_}, xa_a8);
    }
}

void Forward::layer_mixer(std::uint32_t layer, const ForwardBatch& batch, Tensor& residual, Tensor& xm, Tensor& inject) {
    if (config_.lossless) { throw std::logic_error("Qwen4Exp forward: the lossless route runs no layer walks"); }
    layer_mixer_impl(work_, layer, &batch, call_shape(batch), residual, xm, inject);
}

template <class A>
void Forward::layer_mixer_impl(A& a, std::uint32_t layer, const ForwardBatch* batch, const CallShape& shape,
                               Tensor& residual, Tensor& xm, Tensor& inject) {
    const auto& block = parameters_.layers[layer];
    if constexpr (kRuns<A>) { wait_layer(*batch, layer); }
    try {
        auto scope = a.scope();
        if (block.ple) { ple_impl(a, *block.ple, residual, batch, shape); }
        Tensor xa = mix_impl(a, block.attn_hc, residual, &inject);
        Tensor y  = mixer_impl(a, layer, batch, shape, xa, nullptr);
        if constexpr (kRuns<A>) { ops::hyper_connection_inject(y, inject, residual, device_.stream); }
        (void)mix_impl(a, block.mlp_hc, residual, &inject, &xm);
    } catch (const std::exception&) {
        runtime::rethrow_with_context("qwen4_exp/layers/" + std::to_string(layer) + ": ");
    }
}

void Forward::layer_moe(std::uint32_t layer, const ForwardBatch& batch, const Tensor& xm, const Tensor& inject,
                        Tensor& residual, std::int32_t route_columns, std::int32_t route_skipped,
                        std::int32_t chunk_columns, TrimmedCall trimmed) {
    layer_moe_impl(work_, layer, &batch, call_shape(batch), xm, inject, residual, route_columns, route_skipped,
                   chunk_columns, trimmed);
}

template <class A>
void Forward::layer_moe_impl(A& a, std::uint32_t layer, const ForwardBatch* batch, const CallShape& shape,
                             const Tensor& xm, const Tensor& inject, Tensor& residual, std::int32_t route_columns,
                             std::int32_t route_skipped, std::int32_t chunk_columns, TrimmedCall trimmed) {
    try {
        auto scope = a.scope();
        Tensor ym  = moe_impl(a, parameters_.layers[layer].moe, xm, layer, nullptr, batch, shape, route_columns, nullptr,
                              nullptr, route_skipped, &residual, chunk_columns, trimmed);
        if constexpr (kRuns<A>) { ops::hyper_connection_inject(ym, inject, residual, device_.stream); }
    } catch (const std::exception&) {
        runtime::rethrow_with_context("qwen4_exp/layers/" + std::to_string(layer) + ": ");
    }
}

std::int32_t Forward::last_layer_kept(std::int32_t columns) const noexcept {
    const std::uint32_t last = config_.num_hidden_layers - 1;
    if (config_.lossless || scan_ != nullptr || std::find(draft_taps_.begin(), draft_taps_.end(), last) != draft_taps_.end()) {
        return 0;
    }
    // The router keeps its form: its MMA form needs kRouterMmaMinColumns columns. A dense shared expert runs over the
    // whole call (the callers' TrimmedCall::shared): keeping it to these columns moved the first token's logits by up
    // to 0.038 at 17 to 1,000 columns (NVIDIA Dense8 and IQ3_S's Q8 and GGUF, recipe A's BF16; 2026-11-18). The routed
    // experts follow the whole call's routes where they go by width (TrimmedCall::routes, trimmed_routes): without
    // that, ggml-bf16-prefill moved IQ3_S's logits by up to 3.1e-3 and expert-a4-residual FP4-Local's by up to 0.052
    // (2026-11-19).
    const std::int32_t kept = fast_.router_mma && columns >= kRouterMmaMinColumns ? kRouterMmaMinColumns : 1;
    return kept < columns ? kept : 0;
}

void Forward::finish(const ForwardBatch& batch, Tensor& residual, Tensor* logits) {
    finish_impl(work_, &batch, finish_shape(batch, logits), residual, logits);
}

template <class A>
void Forward::finish_impl(A& a, const ForwardBatch* batch, const FinishShape& shape, Tensor& residual, Tensor* logits) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = residual.ne[1], W = dim(config_.residual_width());
    bool mtp_cells = shape.mtp_cells > 0;
    if constexpr (kRuns<A>) {
        if (batch->residual_out.data != nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(batch->residual_out.data, residual.data, residual.bytes(), cudaMemcpyDeviceToDevice, s));
        }
        wait_layer(*batch, config_.num_hidden_layers);
        if (scan_ != nullptr && eager_chunk_) { scan_->write(s); }
        mtp_cells = batch->mtp_chunk != nullptr;
    }
    if (mtp_cells) {
        // The chunk's MTP cells from its live residuals, then its last residual becomes pending.
        const std::int32_t cells = shape.mtp_cells;
        const std::size_t column = static_cast<std::size_t>(W) * 2;
        std::int32_t first_column = 0;
        if constexpr (kRuns<A>) {
            const MtpChunk& chunk = *batch->mtp_chunk;
            if (chunk.columns < 0 || chunk.columns > T || batch->batch != 1 ||
                (cells > 0 && (chunk.ids.ne[0] != cells || chunk.positions.ne[0] != cells ||
                               chunk.rope_positions == nullptr || chunk.block_start_rope == nullptr))) {
                throw std::invalid_argument("Qwen4Exp forward: MTP chunk does not match the call");
            }
            // Cell v is `saved` (v = 0 when prepending) or chunk column v - prepend; sub-chunks bound
            // the drafter's workspace.
            first_column = chunk.prepend ? -1 : 0;
        }
        for (std::int32_t begin = 0; begin < cells; begin += kMtpChunkColumns) {
            const std::int32_t n = std::min(kMtpChunkColumns, cells - begin);
            auto scope  = a.scope();
            Tensor rows = a.alloc(DType::BF16, {W, n});
            if constexpr (kRuns<A>) {
                const MtpChunk& chunk = *batch->mtp_chunk;
                auto* out             = static_cast<std::byte*>(rows.data);
                std::int32_t from     = first_column + begin;
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
                             .slots            = batch->slots,
                             .table_rows       = batch->table_rows,
                             .batch            = 1,
                             .width            = n,
                             .kv_only          = true,
                             .vision           = chunk.vision != nullptr ? &chunk.vision[k] : nullptr};
                mtp_block_impl(a, &call, n, true, rows);
            } else {
                mtp_block_impl(a, nullptr, n, true, rows);
            }
        }
        if constexpr (kRuns<A>) {
            CUDA_CHECK(cudaMemcpyAsync(batch->mtp_chunk->saved.data,
                                       static_cast<const std::byte*>(residual.data) + column * (T - 1), column,
                                       cudaMemcpyDeviceToDevice, s));
        } else {
            (void)column, (void)first_column;
        }
    }
    if (logits == nullptr) { return; }
    auto scope = a.scope();
    // The head's columns are gathered before the final mix, as the lossless route does (candidates TM-06): the mix then
    // runs on those columns alone, so a prompt's first token no longer depends on the width of the call that ends the
    // prompt (the mix routes by width), and a prefill call mixes one column instead of all of them.
    Tensor last = a.alloc(DType::BF16, {W, shape.logit_columns});
    if constexpr (kRuns<A>) {
        ops::gather_columns(residual, batch->logit_columns, last, s);
        if (config_.lossless) {
            lossless_head(last, *logits);
            return;
        }
    }
    const Tensor final_x = mix_impl(a, parameters_.final_mixer, last, nullptr);
    if (shape.screened_head) {
        if constexpr (kRuns<A>) {
            screened_text_head_logits(parameters_, final_x, *logits, a, s);
        } else {
            throw std::logic_error("Qwen4Exp forward: a plan covers prefill calls");
        }
    } else {
        text_head_logits(parameters_, final_x, *logits, a, s);
    }
}

void Forward::head_logits(const Tensor& residuals, Tensor& logits) {
    if (residuals.dtype != DType::BF16 || residuals.ne[0] != dim(config_.residual_width()) ||
        !head_logits_dtype(logits) || logits.ne[0] != dim(config_.vocab_size) ||
        logits.ne[1] != residuals.ne[1]) {
        throw std::invalid_argument("Qwen4Exp forward: head logits need BF16 [S*H, n] residuals and FP32 [V, n] "
                                    "logits (or BF16 on the lossless route)");
    }
    work_.reset();
    auto scope           = work_.scope();
    if (config_.lossless) {
        lossless_head(residuals, logits);
        return;
    }
    const Tensor final_x = mix(parameters_.final_mixer, residuals, nullptr);
    text_head_logits(parameters_, final_x, logits, work_, device_.stream);
}

Tensor Forward::mix(const HyperConnectionParameters& p, const Tensor& residual, Tensor* inject, Tensor* into) {
    return mix_impl(work_, p, residual, inject, into);
}

template <class A>
Tensor Forward::mix_impl(A& a, const HyperConnectionParameters& p, const Tensor& residual, Tensor* inject, Tensor* into) {
    const std::int32_t T = residual.ne[1], H = dim(config_.hidden_size);
    const std::int32_t S = dim(config_.hc.streams), rank = dim(config_.hc.rank);
    Tensor x = into != nullptr ? *into : a.alloc(DType::BF16, {H, T});
    if constexpr (kRuns<A>) {
        ops::hyper_connection_mix(residual, p.norm, p.down.weight, p.up.weight, p.down.policy, S, rank,
                                  config_.rms_norm_eps, x, p.combine ? inject : nullptr, a, device_.stream);
    } else {
        (void)inject;
        plan_scratch(a, ops::hyper_connection_mix_workspace_capacity_bytes(p.down.weight, p.up.weight, p.down.policy, S,
                                                                           rank, T));
    }
    return x;
}

Tensor Forward::ple_embedding(const PleParameters& p, const ForwardBatch& batch) {
    return ple_embedding_impl(work_, p, &batch, call_shape(batch), batch.ids.ne[0]);
}

// A plan takes decoded rows (CallShape::bf16_rows is false): the embedding then lives in the workspace, which covers a
// call reading BF16 rows in place too.
template <class A>
Tensor Forward::ple_embedding_impl(A& a, const PleParameters& p, const ForwardBatch* batch, const CallShape& shape,
                                   std::int32_t columns) {
    const std::int32_t T = columns;
    if (shape.bf16_rows) {
        if constexpr (kRuns<A>) {
            // BF16 rows are the embedding itself: rows [W, heads] of a column are e[h * W + j] (ple_embed's
            // order), so the rows are read in place.
            // Checked before the gated upload is enqueued: a refused call leaves nothing on the stream.
            if (batch->ngram_rows.ne[0] != 2 * dim(config_.ple.head_width())) {
                throw std::invalid_argument("Qwen4Exp forward: BF16 n-gram rows must be 2 x head width bytes");
            }
            if (const NgramRowGate* gate = batch->ngram_gate) {
                const DeviceFault fault{.word = experts_.error, .value = config_.ple.layer << 16U | kErrorRowGateTimeout};
                upload_pinned_when(batch->ngram_rows.data, gate->pinned_rows, batch->ngram_rows.bytes(), gate->ready,
                                   gate->expected, gate->wait_stats, gate->wait_row, fault, device_.stream);
            }
            return Tensor(batch->ngram_rows.data, DType::BF16, {dim(config_.ple.embed_dim), T});
        } else {
            return Tensor(nullptr, DType::BF16, {dim(config_.ple.embed_dim), T});
        }
    }
    Tensor e = a.alloc(DType::BF16, {dim(config_.ple.embed_dim), T});
    if constexpr (!kRuns<A>) {
        (void)p, (void)batch;
        return e;
    } else {
        const cudaStream_t s = device_.stream;
        const ForwardBatch& b = *batch;
        if (const NgramRowGate* gate = b.ngram_gate) {
            // ple_embed is a plain launch (no PDL), so stream order holds it behind the copy. A host that
            // never publishes fails the round through the error word, and the stream drains.
            const DeviceFault fault{.word = experts_.error, .value = config_.ple.layer << 16U | kErrorRowGateTimeout};
            upload_pinned_when(b.ngram_rows.data, gate->pinned_rows, b.ngram_rows.bytes(), gate->ready, gate->expected,
                               gate->wait_stats, gate->wait_row, fault, s);
        }
        const NgramRowFormat format = b.ngram_format.value_or(config_.ple.table.format);
        if (format == NgramRowFormat::Iq4Nl) {
            ops::ple_embed_iq4_nl(b.ngram_rows, e, s);
        } else if (format == NgramRowFormat::Q8_0) {
            ops::ple_embed_q8_0(b.ngram_rows, e, s);
        } else {
            // FP8 rows take their table's BF16 scale, NVFP4 rows their table's FP32 multiplier.
            const Tensor& scale = b.ngram_scale.data != nullptr ? b.ngram_scale : p.ngram_scale;
            if (scale.data == nullptr) {
                throw std::invalid_argument("Qwen4Exp forward: FP8 and NVFP4 n-gram rows need their table's scale");
            }
            if (format == NgramRowFormat::Nvfp4) {
                ops::ple_embed_nvfp4(b.ngram_rows, scale, e, s);
            } else {
                ops::ple_embed(b.ngram_rows, scale, e, s);
            }
        }
        return e;
    }
}

template <class A>
void Forward::ple_impl(A& a, const PleParameters& p, Tensor& residual, const ForwardBatch* batch, const CallShape& shape) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = residual.ne[1], W = residual.ne[0], H = dim(config_.hidden_size);
    auto scope = a.scope();
    const Tensor e = ple_embedding_impl(a, p, batch, shape, T);
    Tensor kv = a.alloc(DType::BF16, {W + H, T});
    project(e, p.projection, kv, a, s);
    Tensor key = a.alloc(DType::BF16, {W, T});
    Tensor value = a.alloc(DType::BF16, {H, T});
    if constexpr (kRuns<A>) {
        Tensor* parts[] = {&key, &value};
        ops::split_rows(kv, parts, s);
    }
    Tensor gated = a.alloc(DType::BF16, {W, T});
    // A verification call records the convolution inputs for its commit and leaves the history.
    Tensor normalized;
    if (shape.verify) {
        if constexpr (kRuns<A>) {
            normalized = batch->verify->ple_inputs.view({W, T});
        } else {
            throw std::logic_error("Qwen4Exp forward: a plan covers prefill calls");
        }
    } else {
        normalized = a.alloc(DType::BF16, {W, T});
    }
    if constexpr (kRuns<A>) {
        ops::ple_gate(key, value, residual, p.key_norm, p.query_norm, p.conv_norm, dim(config_.hc.streams),
                      config_.rms_norm_eps, gated, normalized, s);
        ops::ple_conv_inject(gated, normalized, p.convolution, dim(config_.ple.conv_dilation), state_.ple_conv,
                             batch->slots, batch->verify != nullptr ? Tensor{} : batch->slots, residual, s);
    }
}

template <class A>
Tensor Forward::gdn_impl(A& a, const GdnParameters& p, const Tensor& x, std::uint32_t index, const ForwardBatch* batch,
                         const CallShape& shape, const Tensor* x_a8) {
    const cudaStream_t s = device_.stream;
    const auto& g        = config_.gdn;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size);
    const std::int32_t C = dim(g.conv_channels()), KW = dim(g.key_width()), VW = dim(g.value_width());
    const std::int32_t NV = dim(g.value_heads), NK = dim(g.key_heads), DK = dim(g.key_head_dim),
                       DV = dim(g.value_head_dim);
    // A verification call projects its convolution inputs straight into the replay records.
    GdnReplayRecordLayer records;
    if constexpr (kRuns<A>) {
        if (shape.verify) { records = batch->verify->gdn.layer(static_cast<std::int32_t>(index), batch->batch); }
        scan_input(x, scan_layer_, ActivationScan::Site::GdnInput);
    } else if (shape.verify || shape.width <= 1) {
        throw std::logic_error("Qwen4Exp forward: a plan covers prefill calls");
    }
    // gdn-conv-fused: a decode call's Q8 projection runs the causal convolution in its epilogue, from
    // the FP32 projection rather than its BF16 rounding. Verification and prefill stay composed.
    const bool decode     = !shape.verify && shape.width <= 1;
    const bool fused_conv = fast_.gdn_conv_fused && decode && p.projection.weight.qtype == QType::Q8_G32_FP16;
    Tensor qkv;
    if (!fused_conv) { qkv = shape.verify ? records.conv.view({C, T}) : a.alloc(DType::BF16, {C, T}); }
    Tensor z  = a.alloc(DType::BF16, {VW, T});
    Tensor ab = a.alloc(DType::BF16, {2 * NV, T}); // the control projection: a's rows, then b's
    Tensor q, k, v;
    const bool ggml = p.projection.is_ggml() && p.projection.input_perm == nullptr;
    if (fused_conv) {
        q = a.alloc(DType::BF16, {KW, T});
        k = a.alloc(DType::BF16, {KW, T});
        v = a.alloc(DType::BF16, {VW, T});
        if constexpr (kRuns<A>) {
            auto layer          = state_.gdn->layer_view(index);
            const Tensor x_rows = x.view({H, 1, T});
            Tensor q_rows = q.view({KW, 1, T}), k_rows = k.view({KW, 1, T}), v_rows = v.view({VW, 1, T}),
                   z_rows = z.view({VW, 1, T});
            ops::gdn_input_proj_conv_snapshot(x_rows, p.projection.weight, p.convolution, layer.conv, batch->slots,
                                              batch->slots, q_rows, k_rows, v_rows, z_rows, dense_routes(), s);
        }
        project(x, p.control, ab, a, s);
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
        if (x_a8 != nullptr) {
            if constexpr (kRuns<A>) {
                ops::ggml_linear_a8_encoded(*x_a8, H, ops::ggml_split_segments(segments, H, bounds), parts, s);
            }
        } else {
            project_ggml_split(x, segments, bounds, parts, a, s, dense_routes());
        }
    } else {
        if (ggml) {
            // A GGUF import's projection (its control projection BF16): qkv and z in their own tensors.
            const std::int32_t bounds[] = {0, C, C + VW};
            Tensor* parts[]             = {&qkv, &z};
            if (x_a8 != nullptr) {
                if constexpr (kRuns<A>) {
                    ops::ggml_linear_a8_encoded(*x_a8, H, ops::ggml_split_segments(p.projection.ggml, H, bounds), parts,
                                                s);
                }
            } else {
                project_ggml_split(x, p.projection.ggml, bounds, parts, a, s, dense_routes());
            }
        } else if (p.projection.weight.qtype == QType::Q8_G32_FP16 ||
                   p.projection.weight.qtype == QType::FP8_E4M3FN_G32_E8M0) {
            // Q8 (Dense8) and MXFP8 (FP4-Local): the projection writes q/k/v and z to their
            // destinations itself (MXFP8's f16-accumulate-prefill route encodes x in the workspace).
            if constexpr (kRuns<A>) {
                ops::gdn_input_proj(x, p.projection.weight, qkv, z, ops::LinearPolicy::A16Only, a, s, dense_routes());
            } else {
                plan_scratch(a, ops::gdn_input_proj_workspace_capacity_bytes(p.projection.weight.qtype,
                                                                             p.projection.weight.n, p.projection.weight.k,
                                                                             ops::LinearPolicy::A16Only, T, T));
            }
        } else {
            Tensor projected = a.alloc(DType::BF16, {C + VW, T});
            project(x, p.projection, projected, a, s, dense_routes());
            if constexpr (kRuns<A>) {
                Tensor* parts[] = {&qkv, &z};
                ops::split_rows(projected, parts, s);
            }
        }
        project(x, p.control, ab, a, s);
    }
    // A verification of at most kGdnReplayConvMaxWidth columns evaluates its convolution and gates inside the
    // recurrence (gated_delta_net_replay_record_conv, the same bits as the composed kernels below; design §19.3.38).
    const bool replay_conv = shape.verify && shape.width <= ops::kGdnReplayConvMaxWidth && NK == 16 && NV == 48 &&
                             DK == 128 && DV == 128;
    // Verification and prefill chunks evaluate the gates once for their recurrence; decode evaluates
    // them inside the recurrent update, except a GGUF import's (its third input holds A itself), whose
    // decode takes them from gdn_gating_direct too.
    Tensor gate, beta;
    if (!replay_conv && (shape.verify || shape.width > 1 || g.a_direct)) {
        gate = a.alloc(DType::FP32, {NV, T});
        beta = a.alloc(DType::FP32, {NV, T});
        if constexpr (kRuns<A>) {
            if (g.a_direct) {
                ops::gdn_gating_direct(ab.slice(0, 0, NV), ab.slice(0, NV, NV), p.a_log, p.dt_bias, gate, beta, s);
            } else {
                ops::gdn_gating(ab.slice(0, 0, NV), ab.slice(0, NV, NV), p.a_log, p.dt_bias, gate, beta, s);
            }
        }
    }

    if (!fused_conv && !replay_conv) {
        q = a.alloc(DType::BF16, {KW, T});
        k = a.alloc(DType::BF16, {KW, T});
        v = a.alloc(DType::BF16, {VW, T});
    }
    Tensor o = a.alloc(DType::BF16, {DV, NV, T});
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(DK)));
    if constexpr (!kRuns<A>) {
        (void)scale, (void)index, (void)x_a8;
        plan_scratch(a, ops::gated_delta_net_workspace_capacity_bytes(NK, NV, T, T)); // the chunked recurrence's
    } else if (replay_conv) {
        const std::int32_t Wd = batch->width, B = batch->batch;
        auto layer    = state_.gdn->layer_view(index);
        Tensor o_rows = o.view({DV, NV, Wd, B});
        ops::gated_delta_net_replay_record_conv(qkv.view({C, Wd, B}), p.convolution, layer.conv, ab.slice(0, 0, NV),
                                                ab.slice(0, NV, NV), p.a_log, p.dt_bias, g.a_direct, scale,
                                                layer.recurrent, batch->slots, records.key, records.value, records.gate,
                                                o_rows, s);
    } else if (shape.verify) {
        // Speculative verification: the convolution and recurrence run from the live states
        // without changing them, and the recurrence's inputs are recorded for the commit fold.
        const std::int32_t Wd = batch->width, B = batch->batch;
        auto layer = state_.gdn->layer_view(index);
        ops::causal_conv1d_silu_from_states_split(qkv.view({C, Wd, B}), p.convolution, layer.conv,
                                                  batch->slots, q, k, v, s);
        Tensor o_rows = o.view({DV, NV, Wd, B});
        ops::gated_delta_net_replay_record(q.view({DK, NK, Wd, B}), k.view({DK, NK, Wd, B}), v.view({DV, NV, Wd, B}),
                                           gate.view({NV, Wd, B}), beta.view({NV, Wd, B}), scale, layer.recurrent,
                                           Tensor{}, batch->slots, records.key, records.value, records.gate,
                                           o_rows, s);
    } else if (shape.width > 1) {
        // One sequence of several positions (a prefill chunk): chunked recurrence.
        if (batch->batch != 1) { throw std::invalid_argument("Qwen4Exp GDN: multi-position calls hold one sequence"); }
        if (batch->host_slots.empty()) { throw std::invalid_argument("Qwen4Exp GDN: missing host slot"); }
        const std::int32_t slot = batch->host_slots[0];
        Tensor conv_state = state_.gdn->conv_slot(index, slot);
        ops::causal_conv1d_silu_split(qkv, p.convolution, conv_state, conv_state, q, k, v, s);
        Tensor state = state_.gdn->recurrent_slot(index, slot);
        ops::gated_delta_net(q.view({DK, NK, T}), k.view({DK, NK, T}), v.view({DV, NV, T}), gate, beta, scale,
                             /*normalize_qk=*/true, a, state, state, o, device_.execution_view(),
                             fast_.gdn_arithmetic);
    } else {
        // One position per sequence (decode): the recurrent update with device-chosen slots.
        auto layer = state_.gdn->layer_view(index);
        if (!fused_conv) {
            ops::causal_conv1d_silu_snapshot_split(qkv.view({C, 1, T}), p.convolution, layer.conv, Tensor{},
                                                   batch->slots, batch->slots, q, k, v, s);
        }
        Tensor o_batch = o.view({DV, NV, 1, T});
        if (g.a_direct) {
            ops::gated_delta_net_batch_update(q.view({DK, NK, 1, T}), k.view({DK, NK, 1, T}), v.view({DV, NV, 1, T}),
                                              gate.view({NV, 1, T}), beta.view({NV, 1, T}), scale, /*normalize_qk=*/true,
                                              layer.recurrent, batch->slots, batch->slots, o_batch, s);
        } else {
            // The gates are evaluated inside the update from the control projection (no gdn_gating kernel).
            ops::gated_delta_net_batch_update_gating(q.view({DK, NK, 1, T}), k.view({DK, NK, 1, T}),
                                                     v.view({DV, NV, 1, T}), ab.slice(0, 0, NV), ab.slice(0, NV, NV),
                                                     p.a_log, p.dt_bias, scale, /*normalize_qk=*/true, layer.recurrent,
                                                     batch->slots, batch->slots, o_batch, s);
        }
    }
    const Tensor rows  = o.view({DV, NV * T});
    const Tensor gates = z.view({DV, NV * T});
    const ops::RmsGate gate_kind = g.output_gate == GateActivation::Sigmoid ? ops::RmsGate::Sigmoid : ops::RmsGate::Silu;
    Tensor y = a.alloc(DType::BF16, {H, T});
    if (!(scan_ != nullptr && eager_chunk_) && p.output.is_ggml() &&
        ops::ggml_a8_takes_encoded(p.output.ggml, VW, T, dense_routes(), p.output.input_perm, p.output.perm_block)) {
        // --a8-activations: the gated norm writes the output projection's 8-bit columns (no encoding launch; the
        // same words, so the same bits).
        Tensor encoded = a.alloc(DType::I32, {ops::ggml_a8_encoded_column_words(p.output.ggml, VW), T});
        if constexpr (kRuns<A>) {
            ops::gated_rmsnorm_ggml_a8_encoded(rows, p.norm, gates, gate_kind, config_.rms_norm_eps, p.output.input_perm,
                                               p.output.perm_block, p.output.ggml, encoded, s);
            Tensor* const outs[] = {&y};
            ops::ggml_linear_a8_encoded(encoded, VW, p.output.ggml, outs, s);
        }
        return y;
    }
    if (producer_encodes(p.output, VW, T, dense_routes(), scan_ != nullptr && eager_chunk_)) {
        // The gated norm writes the output projection's input encoded (no encoding launch).
        ops::GgmlEncodedColumns encoded = alloc_encoded(a, VW, T);
        if constexpr (kRuns<A>) {
            ops::gated_rmsnorm_ggml_encoded(rows, p.norm, gates, gate_kind, config_.rms_norm_eps, p.output.input_perm,
                                            p.output.perm_block, encoded, s);
            Tensor* const outs[] = {&y};
            ops::ggml_linear_encoded(encoded, p.output.ggml, outs, s);
        }
        return y;
    }
    Tensor normed = a.alloc(DType::BF16, {DV, NV * T});
    if constexpr (kRuns<A>) {
        ops::gated_rmsnorm(rows, p.norm, gates, gate_kind, config_.rms_norm_eps, normed,
                           device_.execution_view().on_stream(s));
        scan_input(normed.view({VW, T}), scan_layer_, ActivationScan::Site::GdnOutput);
    }
    project(normed.view({VW, T}), p.output, y, a, s, dense_routes());
    return y;
}

template <class A>
Tensor Forward::attention_impl(A& ws, const AttentionParameters& p, const Tensor& x, const AttentionCall* call,
                               const AttentionShape& shape, const Tensor* x_a8) {
    const cudaStream_t s = device_.stream;
    const auto& a        = config_.attention;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size);
    const std::int32_t QW = dim(a.query_width()), KW = dim(a.key_width()), D = dim(a.head_dim);
    const std::int32_t IH = dim(config_.qsa.index_heads), ID = dim(config_.qsa.index_head_dim);
    const std::int32_t index_rows = IH * ID + ID;
    // The MTP block's K/V-only chunk (in a prefill chunk, after the last layer) is no layer's attention input:
    // scan_layer_ still names the last layer.
    if constexpr (kRuns<A>) {
        if (!shape.kv_only) { scan_input(x, scan_layer_, ActivationScan::Site::QsaInput); }
    }
    Tensor q = ws.alloc(DType::BF16, {D, dim(a.heads), T});
    Tensor gate = ws.alloc(DType::BF16, {QW, T});
    Tensor k = ws.alloc(DType::BF16, {D, dim(a.kv_heads), T});
    Tensor v = ws.alloc(DType::BF16, {D, dim(a.kv_heads), T});
    Tensor iq = ws.alloc(DType::BF16, {ID, IH, T});
    Tensor ik;
    if (shape.key_records) {
        if constexpr (kRuns<A>) {
            ik = call->key_records.view({ID, T});
        } else {
            throw std::logic_error("Qwen4Exp forward: a plan covers prefill calls");
        }
    } else {
        ik = ws.alloc(DType::BF16, {ID, T});
    }
    const bool ggml = p.projection.is_ggml() && p.projection.input_perm == nullptr &&
                      (!p.index_projection || (p.index_projection->is_ggml() && p.index_projection->input_perm == nullptr));
    Tensor qf = q.view({QW, T}), kf = k.view({KW, T}), vf = v.view({KW, T}), iqf = iq.view({IH * ID, T});
    // linear_qsa_group_supported reads its parts' data, which a plan's tensors do not have (with_data).
    const std::array<Tensor, 6> group_parts{with_data(qf),  with_data(gate), with_data(kf),
                                            with_data(vf),  with_data(iqf),  with_data(ik)};
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
        const std::int32_t bounds[] = {0, QW, 2 * QW, 2 * QW + KW, 2 * QW + 2 * KW, 2 * QW + 2 * KW + IH * ID,
                                       2 * QW + 2 * KW + index_rows};
        Tensor* parts[] = {&qf, &gate, &kf, &vf, &iqf, &ik};
        if (x_a8 != nullptr) {
            if constexpr (kRuns<A>) {
                ops::ggml_linear_a8_encoded(*x_a8, H, ops::ggml_split_segments(segments, H, bounds), parts, s);
            }
        } else {
            project_ggml_split(x, segments, bounds, parts, ws, s, dense_routes());
        }
    } else if (!p.index_projection && !p.projection.is_ggml() &&
               ops::linear_qsa_group_supported(p.projection.weight,
                                               std::array<const Tensor*, 6>{&group_parts[0], &group_parts[1],
                                                                            &group_parts[2], &group_parts[3],
                                                                            &group_parts[4], &group_parts[5]}
                                                   .data(),
                                               p.projection.policy)) {
        // The [13952,2560] parent writes its six parts directly (linear's bits), with no [13952,T] intermediate
        // and no split_rows copy (candidates §7 D7: FP4-Local pp32768 / pp65536 +0.7 / +0.6 %, Dense8 flat, 2026-10-28).
        if constexpr (kRuns<A>) {
            Tensor* parts[] = {&qf, &gate, &kf, &vf, &iqf, &ik};
            ops::linear_qsa_group(x, p.projection.weight, parts, p.projection.policy, ws, s, dense_routes());
        } else {
            // linear_qsa_group's workspace is linear's for the parent problem.
            plan_scratch(ws, ops::linear_workspace_capacity_bytes(p.projection.weight.qtype, p.projection.weight.n,
                                                                  p.projection.weight.k, p.projection.policy, T, T));
        }
    } else {
        Tensor projected = ws.alloc(DType::BF16, {2 * QW + 2 * KW + (p.index_projection ? 0 : index_rows), T});
        project(x, p.projection, projected, ws, s, dense_routes());
        if (p.index_projection) {
            Tensor indexed = ws.alloc(DType::BF16, {index_rows, T});
            project(x, *p.index_projection, indexed, ws, s);
            if constexpr (kRuns<A>) {
                Tensor* parts[]       = {&qf, &gate, &kf, &vf};
                Tensor* index_parts[] = {&iqf, &ik};
                ops::split_rows(projected, parts, s);
                ops::split_rows(indexed, index_parts, s);
            }
        } else if constexpr (kRuns<A>) {
            Tensor* parts[] = {&qf, &gate, &kf, &vf, &iqf, &ik};
            ops::split_rows(projected, parts, s);
        }
    }
    // The vector-quantized storages' exact window is addressed by the sequences' state slots.
    ops::QsaKVLayer layer;
    ops::QsaBatch qsa_batch;
    bool vq = false;
    if constexpr (kRuns<A>) {
        layer = call->layer;
        if (layer.kv.window.present()) { layer.kv.window.slots = call->slots; }
        vq        = layer.kv.storage == KvCacheStorage::Vq2 || layer.kv.storage == KvCacheStorage::Q4KeyVq2Value;
        qsa_batch = ops::QsaBatch{.block_tables     = kv_.block_tables,
                                  .table_rows       = call->table_rows,
                                  .positions        = call->positions,
                                  .tail_slots       = call->slots,
                                  .rope_positions   = call->rope_positions,
                                  .block_start_rope = call->block_start_rope,
                                  .batch            = call->batch,
                                  .width            = call->width,
                                  .update_tails     = call->update_tails};
    }
    const std::int32_t KVH          = dim(a.kv_heads);
    const ops::QsaGeometry geometry = qsa_geometry(config_);
    // qsa-prepare-fused: every call (decode, verification, prefill and MTP chunks alike, so a position's
    // stored K and pooled keys do not depend on which route wrote them, as Replay and prefix-cache
    // equivalence require) normalizes and rotates Q, K, the index query and the pooled keys and appends
    // K/V and the tails in one kernel, FP32 from the projection rows into the cache codec. Q and the
    // index query are prepared in place.
    Tensor qn = q;
    ops::QsaAppend append{};
    if (fast_.qsa_prepare_fused && ops::qsa_prepare_supports(shape.storage)) {
        if constexpr (kRuns<A>) {
            const ops::QsaPrepareInputs inputs{.q = q, .k = k, .v = v, .index_q = iq, .index_k = ik};
            const ops::QsaPrepareNorms norms{.query       = p.query_norm,
                                             .key         = p.key_norm,
                                             .index_query = p.index_query_norm,
                                             .index_key   = p.index_key_norm};
            Tensor q_out  = shape.kv_only ? Tensor{} : q;
            Tensor iq_out = shape.kv_only ? Tensor{} : iq;
            ops::qsa_prepare(inputs, norms, call->tails, layer, qsa_batch, geometry, q_out, iq_out, s);
        }
        if (shape.kv_only) { return Tensor{}; }
    } else {
        qn        = ws.alloc(DType::BF16, {D, dim(a.heads), T});
        Tensor kn = ws.alloc(DType::BF16, {D, dim(a.kv_heads), T});
        if constexpr (kRuns<A>) {
            {
                Tensor qn_rows = qn.view({D, dim(a.heads) * T});
                Tensor kn_rows = kn.view({D, dim(a.kv_heads) * T});
                ops::rmsnorm(q.view({D, dim(a.heads) * T}), p.query_norm, config_.rms_norm_eps, true, qn_rows, s);
                ops::rmsnorm(k.view({D, dim(a.kv_heads) * T}), p.key_norm, config_.rms_norm_eps, true, kn_rows, s);
            }
            ops::rope(call->rope_positions, rope_, qn, kn, device_.execution_view().on_stream(s));

            // Append K/V of every sequence through its device-chosen table row, then the pooled index
            // keys. qsa_attention appends the vector-quantized storages' K/V itself (a wide call stages
            // its window rows); a K/V-only call appends here.
            append = ops::QsaAppend{.k = kn.view({D, KVH, call->width, call->batch}),
                                    .v = v.view({D, KVH, call->width, call->batch})};
            if (!vq || shape.kv_only) {
                const PagedKVBatchLayerView view{.k_pages       = layer.kv.k_pages,
                                                 .v_pages       = layer.kv.v_pages,
                                                 .k_scale_pages = layer.kv.k_scale_pages,
                                                 .v_scale_pages = layer.kv.v_scale_pages,
                                                 .block_tables  = kv_.block_tables,
                                                 .head_dim      = layer.kv.head_dim,
                                                 .num_kv_heads  = layer.kv.num_kv_heads,
                                                 .storage       = layer.kv.storage,
                                                 .window        = layer.kv.window,
                                                 .unpackable    = layer.kv.unpackable};
                ops::kv_cache_append_batch(append.k, append.v, call->positions.view({call->width, call->batch}),
                                           call->table_rows, view, s);
            }
            ops::qsa_pool_keys(ik, p.index_key_norm, call->tails, layer, qsa_batch, geometry, s);
        } else {
            (void)KVH;
        }
        if (shape.kv_only) { return Tensor{}; }
        if constexpr (kRuns<A>) { ops::qsa_index_query(iq, p.index_query_norm, call->rope_positions, geometry, s); }
    }

    Tensor out = ws.alloc(DType::BF16, {D, dim(a.heads), T});
    const std::size_t scratch = ops::qsa_attention_workspace_bytes(geometry, T, max_context_, shape.storage);
    const DeviceSpan span     = ws.alloc_bytes(scratch);
    void* unpack              = nullptr;
    if (shape.storage == KvCacheStorage::Bf16Packed && T > kQsaUnpackColumns && shape.batch == 1) {
        unpack = ws.alloc_bytes(ops::qsa_attention_unpack_bytes(geometry, max_context_)).data;
    }
    if constexpr (kRuns<A>) {
        ops::qsa_attention(qn, iq, layer, qsa_batch, geometry,
                           static_cast<float>(1.0 / std::sqrt(static_cast<double>(D))), max_context_, span.data,
                           span.bytes, out, s, vq ? &append : nullptr, unpack);
    } else {
        (void)span, (void)unpack, (void)vq, (void)call;
    }
    // Output gate: out * sigmoid(gate), then o_proj. Applying the gate in qsa_attention's split merge
    // (candidates §7.3 D7) measured flat (removed 2026-10-28).
    Tensor gated = out.view({QW, T});
    Tensor y     = ws.alloc(DType::BF16, {H, T});
    if (producer_encodes(p.output, QW, T, dense_routes(), scan_ != nullptr && eager_chunk_)) {
        // The gate writes the output projection's input encoded (no encoding launch).
        ops::GgmlEncodedColumns encoded = alloc_encoded(ws, QW, T);
        if constexpr (kRuns<A>) {
            ops::sigmoid_mul_ggml_encoded(gate, gated, p.output.input_perm, p.output.perm_block, encoded, s);
            Tensor* const outs[] = {&y};
            ops::ggml_linear_encoded(encoded, p.output.ggml, outs, s);
        }
        return y;
    }
    if constexpr (kRuns<A>) {
        ops::sigmoid_mul(gate, gated, s);
        scan_input(gated, scan_layer_, ActivationScan::Site::QsaOutput);
    }
    project(gated, p.output, y, ws, s, dense_routes());
    return y;
}

template <class A>
Tensor Forward::moe_impl(A& a, const MoeParameters& p, const Tensor& x, std::uint32_t layer, Tensor* route_tap,
                         const ForwardBatch* batch, const CallShape& shape, std::int32_t route_columns,
                         const Tensor* inject, Tensor* residual, std::int32_t route_skipped, const Tensor* streams,
                         std::int32_t chunk_columns, TrimmedCall trimmed) {
    const cudaStream_t s = device_.stream;
    const auto& m        = config_.moe;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size), E = dim(m.experts), K = dim(m.top_k);
    Tensor logits = a.alloc(DType::FP32, {E + 1, T});
    ops::MoeRouting routing{a.alloc(DType::I32, {K, T}), a.alloc(DType::FP32, {K, T}), a.alloc(DType::FP32, {T})};
    // The route log holds the call's last `logged` columns: all of them (written by the dispatch), or
    // the trailing ones (a layer walk's grouped call logs its last chunk), copied from the routing.
    const std::int32_t logged = route_columns < 0 ? T : std::min(route_columns, T);
    std::int32_t* route_log   = nullptr;
    std::int32_t* log_tail    = nullptr;
    if constexpr (kRuns<A>) {
        if (experts_.route_log != nullptr) {
            if (route_skipped < 0 || static_cast<std::size_t>(K) * (logged + route_skipped) > experts_.route_stride) {
                throw std::invalid_argument("Qwen4Exp MoE: route log is too small for this call");
            }
            std::int32_t* log = experts_.route_log + layer * experts_.route_stride;
            if (route_skipped > 0) {
                // A trimmed last layer's skipped columns route nowhere (the residency skips ids past its experts).
                CUDA_CHECK(
                    cudaMemsetAsync(log, 0xFF, static_cast<std::size_t>(K) * route_skipped * sizeof(std::int32_t), s));
                log += static_cast<std::size_t>(K) * route_skipped;
            }
            if (logged == T) {
                route_log = log;
            } else {
                log_tail = log;
            }
        }
    }
    // A shared expert bank (MoeConfig::shared_bank) is the call's last job: expert E with entries K*T + t, its outputs
    // the last T columns of `outputs` (design §8.6), in the routed bank's arithmetic.
    const bool shared_job = p.shared_bank != nullptr;
    const std::int32_t SJ = shared_job ? 1 : 0;
    const DeviceSpan dispatch_bytes = a.alloc_bytes(ops::moe_dispatch_bytes(E + SJ, (K + SJ) * T));
    ops::MoeDispatch dispatch{};
    // A trimmed last layer whose experts follow the untrimmed call's routes (TrimmedCall::routes, trimmed_routes): that
    // call's router and dispatch over its R columns (x is their trailing T), in plans and runs alike.
    const std::int32_t R = trimmed.routes != nullptr && trimmed_routes(p, trimmed.routes->ne[1]) ? trimmed.routes->ne[1] : 0;
    if (R > 0 && (R < T || trimmed.routes_chunk < 0 || trimmed.routes_chunk >= R ||
                  (kRuns<A> && static_cast<const std::byte*>(trimmed.routes->data) +
                                       static_cast<std::size_t>(R - T) * H * sizeof(std::uint16_t) !=
                                   static_cast<const std::byte*>(x.data)))) {
        throw std::invalid_argument("Qwen4Exp MoE: a trimmed call's columns must be its reference call's trailing ones");
    }
    Tensor route_logits;
    ops::MoeRouting route_routing{};
    DeviceSpan route_dispatch_bytes{};
    ops::MoeDispatch route_dispatch{};
    if (R > 0) {
        route_logits         = a.alloc(DType::FP32, {E + 1, R});
        route_routing        = {a.alloc(DType::I32, {K, R}), a.alloc(DType::FP32, {K, R}), a.alloc(DType::FP32, {R})};
        route_dispatch_bytes = a.alloc_bytes(ops::moe_dispatch_bytes(E + SJ, (K + SJ) * R));
    }
    bool streamed_chunk = false, route_plans = true;
    ops::MoeExpertSource source = {};
    if constexpr (!kRuns<A>) {
        (void)route_tap, (void)batch, (void)residual, (void)streams, (void)route_log, (void)dispatch_bytes;
        if (!eager_chunk_) { throw std::logic_error("Qwen4Exp forward: a plan covers prefill calls"); }
    } else {
    dispatch = ops::carve_moe_dispatch(dispatch_bytes.data, E + SJ, (K + SJ) * T);
    if (R > 0) {
        route_dispatch         = ops::carve_moe_dispatch(route_dispatch_bytes.data, E + SJ, (K + SJ) * R);
        dispatch.route_offsets = route_dispatch.offsets;
    }
    // Without a route tap or a streamed prefill chunk (whose host-side split reads the dispatch), the
    // route waits until the experts' source is known, so its kernel also plans the CPU-served misses.
    streamed_chunk = eager_chunk_ && batch != nullptr && batch->stream && experts_.stream != nullptr &&
                     experts_.stream->active();
    route_plans    = route_tap == nullptr && !streamed_chunk;
    source         = expert_source(p, layer);
    source.route_columns = R; // 0: the call's own (MoeExpertSource::route_columns)
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
        source.split_stream    = experts_.split_stream;
        source.split_events[0] = experts_.split_events[0];
        source.split_events[1] = experts_.split_events[1];
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
        std::size_t spans = 0, queued = 0;
        const std::size_t budget = whole_chain ? kChainWarm : ~std::size_t{0};
        const auto add = [&](const void* data, std::size_t bytes) {
            if (spans < chain.size()) {
                chain[spans++] = {data, bytes};
                queued += std::min(bytes, budget - queued);
            }
        };
        // A linear's weights, at most `cap` bytes of them.
        const auto add_linear = [&](const LinearParameters& linear, std::int32_t k, std::size_t cap) {
            if (linear.weight.qtype == QType::BF16_P12) {
                // bf16_p12: the same leading rows of the code and exponent planes (two code bytes to one), as many
                // as the cap and the chain's budget leave room for.
                const std::size_t room = std::min(cap, budget - queued);
                const std::size_t rows =
                    std::min(static_cast<std::size_t>(linear.weight.n), room / (static_cast<std::size_t>(k) * 3 / 2));
                add(linear.weight.qdata, rows * k);
                add(linear.weight.qhigh, rows * k / 2);
                return;
            }
            if (linear.weight.qtype == QType::FP8_E4M3FN_G32_E8M0) {
                // MXFP8: every stage of its route reads codes and group scales together, and the scale
                // plane follows the codes, so it warms first (1/32 of the code bytes).
                const std::size_t scale_bytes = static_cast<std::size_t>(linear.weight.n) * (linear.weight.k / 32);
                const std::size_t code_bytes  = static_cast<std::size_t>(linear.weight.n) * linear.weight.k;
                add(linear.weight.scales, std::min(scale_bytes, cap));
                add(linear.weight.qdata, std::min(code_bytes, cap - std::min(scale_bytes, cap)));
                return;
            }
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
            const auto& attn = std::get<AttentionParameters>(next.mixer);
            input = &attn.projection, output = &attn.output;
            output_k = dim(config_.attention.query_width());
            if (attn.index_projection) { extra = &*attn.index_projection; }
        }
        add_linear(*input, H, whole_chain ? ~std::size_t{0} : kInputWarm);
        if (whole_chain) {
            if (extra != nullptr) { add_linear(*extra, H, ~std::size_t{0}); }
            add_linear(*output, output_k, ~std::size_t{0});
            add(next.mlp_hc.down.weight.payload, next.mlp_hc.down.weight.payload_bytes);
            add(next.mlp_hc.up.weight.payload, next.mlp_hc.up.weight.payload_bytes);
            add(next.moe.router.data, next.moe.router.bytes());
            if (next.moe.shared_bank != nullptr) {
                // The shared expert's record: the next call's resident last job.
                add(next.moe.shared_bank->planes.device_records, next.moe.shared_bank->shape().record_bytes());
            } else {
                add_linear(next.moe.shared_gate_up, H, ~std::size_t{0});
                add_linear(next.moe.shared_down, dim(config_.moe.shared_intermediate), ~std::size_t{0});
            }
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
    }
    if (shared_job) {
        source.shared_expert = E;
        source.shared_record = reinterpret_cast<const std::uint8_t*>(p.shared_bank->planes.device_records);
    }
    Tensor outputs = a.alloc(DType::BF16, {H, (K + SJ) * T});
    const std::int32_t max_jobs = std::min(E, K * T) + SJ;
    // The bank's wide route (expert_wide_a16, the plan's too) and narrow route.
    source.wide_a16   = expert_wide_a16(p);
    source.narrow_a16 = expert_narrow_a16(p);
    // expert_source keeps MoeExpertSource's shape; a plan's source is the default too.
    const DeviceSpan expert_ws = a.alloc_bytes(ops::moe_experts_workspace_bytes(source.shape, max_jobs, (K + SJ) * T, T,
                                                                                p.bank->activation, source.wide_a16));
    const std::int32_t I = dim(m.shared_intermediate);
    // The dense shared expert's input and calls: x, as one call or a pair call's two chunks, else a trimmed last
    // layer's whole call (TrimmedCall::shared), whose trailing T columns are x.
    const Tensor& sx      = trimmed.shared != nullptr ? *trimmed.shared : x;
    const std::int32_t SW = sx.ne[1];
    std::vector<std::pair<std::int32_t, std::int32_t>> shared_chunks;
    if (trimmed.shared == nullptr && chunk_columns > 0) {
        shared_chunks = {{0, chunk_columns}, {chunk_columns, T - chunk_columns}};
    } else {
        shared_chunks = {{0, SW}};
    }
    if (SW < T || (trimmed.shared != nullptr && sx.dtype != DType::BF16)) {
        throw std::invalid_argument("Qwen4Exp MoE: the shared expert's input must hold the MoE's columns");
    }
    // The dense shared expert's product and output (over its input), and the trailing T columns the combine reads; a
    // shared job's output is the call's last T columns.
    Tensor product    = shared_job ? Tensor{} : a.alloc(DType::BF16, {I, SW});
    Tensor shared_all = shared_job ? Tensor{} : a.alloc(DType::BF16, {H, SW});
    Tensor shared     = shared_job ? outputs.slice(1, K * T, T) : shared_all.slice(1, SW - T, T);
    // Each call of the dense shared expert as that call alone runs it: every gate/up projection, then every down.
    const auto run_shared = [&](cudaStream_t stream) {
        for (const auto [from, n] : shared_chunks) {
            Tensor product_chunk = product.slice(1, from, n);
            shared_swiglu(sx.slice(1, from, n), p.shared_gate_up, product_chunk, a, stream, dense_routes());
        }
        if constexpr (kRuns<A>) { scan_input(product, layer, ActivationScan::Site::SharedDown); }
        for (const auto [from, n] : shared_chunks) {
            Tensor shared_chunk = shared_all.slice(1, from, n);
            project(product.slice(1, from, n), p.shared_down, shared_chunk, a, stream, dense_routes());
        }
    };
    // The shared expert runs while the host computes the CPU-served misses, and in decode and
    // verification calls on its own stream while the routed experts compute too. It reads only x,
    // and its scoped workspace lies past every buffer the routed experts use; this stream allocates
    // nothing more until the join. Eager prefill chunks keep the serial order: the side stream there
    // (candidates W18) measured flat twice (removed 2026-10-28).
    const bool side = !eager_chunk_ && experts_.shared_stream != nullptr;
    bool early = false, shared_early = false;
    if constexpr (kRuns<A>) {
    // The routed experts' activation encoding needs x alone, so it runs on the side stream beside the router and
    // the route (it followed the route on the call's critical path, ~3 us a layer); a call without one (an NVFP4
    // W4A16 bank) forks nothing: the empty fork and join alone cost Dense8 MTP decode 1.4 %. 16K decode, 3 runs each
    // (2026-10-29): IQ3_S 265.5 -> 267.7 tok/s with the fork, Dense8 243.9 either way.
    early = side && ops::moe_experts_prepares(x, source, max_jobs, expert_ws.data, outputs);
    // ForwardExperts::shared_early: the shared expert forks at the layer's input (after the encoding, when the call
    // has one) on the side stream at once, beside the router and the route's plan (a nearly idle device), instead of
    // after the plan beside the routed experts. Its scoped workspace lies past every buffer this call allocated, and
    // this stream allocates nothing until the join.
    shared_early = side && !shared_job && experts_.shared_early && scan_ == nullptr;
    if (early || shared_early) {
        CUDA_CHECK(cudaEventRecord(experts_.shared_events[2], s));
        CUDA_CHECK(cudaStreamWaitEvent(experts_.shared_stream, experts_.shared_events[2], 0));
        if (early) {
            ops::moe_experts_prepare(x, source, max_jobs, expert_ws.data, outputs, experts_.shared_stream);
            CUDA_CHECK(cudaEventRecord(experts_.shared_events[3], experts_.shared_stream));
        }
        if (shared_early) {
            run_shared(experts_.shared_stream);
            CUDA_CHECK(cudaEventRecord(experts_.shared_events[1], experts_.shared_stream));
        }
    }
    }
    // A pair call's two chunks (`chunk_columns`, the first one's): the router and the shared expert pick their
    // routes by the columns they are given, so each chunk's run on their own, as that chunk's own call runs them.
    if (chunk_columns < 0 || chunk_columns >= T || (chunk_columns > 0 && !eager_chunk_)) {
        throw std::invalid_argument("Qwen4Exp MoE: a pair call's first chunk must lie inside an eager call");
    }
    const std::array<std::pair<std::int32_t, std::int32_t>, 2> chunks{
        {{0, chunk_columns > 0 ? chunk_columns : T}, {chunk_columns, chunk_columns > 0 ? T - chunk_columns : 0}}};
    cudaStream_t shared_s = s;
    if constexpr (!kRuns<A>) {
        // A plan routes the copy-ahead's guess wherever the call may be gated (CallShape::may_gate): run mode
        // allocates its routing below when the stream's gated call reads it ahead.
        (void)log_tail, (void)logged, (void)route_plans, (void)streamed_chunk, (void)side;
        if (shape.may_gate && layer + 1 < config_.num_hidden_layers) {
            ops::MoeRouting guess{a.alloc(DType::I32, {K, T}), a.alloc(DType::FP32, {K, T}), a.alloc(DType::FP32, {T})};
            (void)guess;
            (void)a.alloc_bytes(ops::moe_dispatch_bytes(E, K * T));
        }
    } else {
    (void)shape;
    const Tensor* rows[] = {&p.router, &p.shared_score};
    const auto route = [&](const Tensor& input, Tensor& out, std::int32_t from, std::int32_t n) {
        const Tensor input_chunk = input.slice(1, from, n);
        Tensor out_chunk         = out.slice(1, from, n);
        if (fast_.router_mma && n >= kRouterMmaMinColumns) {
            ops::projection_fp32_router_mma(input_chunk, rows, out_chunk, s);
        } else {
            ops::projection_fp32(input_chunk, rows, out_chunk, s);
        }
    };
    if (R > 0) {
        // The reference call's router, per chunk as that call runs it, and its dispatch: each kept entry's route goes by
        // its expert's entry count there (MoeDispatch::route_offsets, the shared job's R included). The kept columns'
        // logits are its trailing T, so their routing is its own bit for bit.
        const std::int32_t first = trimmed.routes_chunk > 0 ? trimmed.routes_chunk : R;
        route(*trimmed.routes, route_logits, 0, first);
        if (first < R) { route(*trimmed.routes, route_logits, first, R - first); }
        ops::moe_route_dispatch(route_logits, K, route_routing, route_dispatch, nullptr, s, ops::RouterLogits::kFp32,
                                shared_job);
        CUDA_CHECK(cudaMemcpyAsync(logits.data, route_logits.slice(1, R - T, T).data, logits.bytes(),
                                   cudaMemcpyDeviceToDevice, s));
    } else {
        for (const auto [from, n] : chunks) {
            if (n > 0) { route(x, logits, from, n); }
        }
    }
    if (!route_plans) {
        ops::moe_route_dispatch(logits, K, routing, dispatch, route_log, s, ops::RouterLogits::kFp32, shared_job);
    }
    if (route_tap != nullptr &&
        cudaMemcpyAsync(route_tap->data, routing.ids.data, routing.ids.bytes(), cudaMemcpyDeviceToDevice, s) !=
            cudaSuccess) {
        throw std::runtime_error("Qwen4Exp forward: route tap copy failed");
    }
    // A streamed prefill chunk: the layer's copied-ahead experts are read from the ring.
    ExpertStream* const streaming = streamed_chunk ? experts_.stream : nullptr;
    // The copy-ahead (ExpertStream::plan_ahead, candidates BX-09): a gated call also routes a guess of the next
    // layer's MoE input through the next layer's router and copies the experts that guess makes heavy while this layer
    // ends, so the link does not wait idle for the next layer's routing. Placement only: the same record bytes are read.
    // A trimmed last layer (YC-01) routes only its kept columns: guessing it from the whole call would copy for nothing.
    const bool trims_next = batch != nullptr && batch->trim_last_layer && layer + 2 == config_.num_hidden_layers &&
                            last_layer_kept(T) > 0;
    const bool ahead = streaming != nullptr && streaming->gated() && batch->release_stream &&
                       layer + 1 < config_.num_hidden_layers && !trims_next && streams != nullptr &&
                       streams->ne[1] == T;
    if (streaming != nullptr && streaming->gated()) {
        // A gated call (design §19.3.12): the layer's routing decides which experts the link copies
        // and which the CPU serves. The CPU takes exactly the experts the stream leaves out: every
        // narrow miss is offered (no PCIe share), capped at the policy's count.
        auto* offsets = static_cast<std::int32_t*>(split_offsets_.data());
        CUDA_CHECK(cudaMemcpyAsync(offsets, dispatch.offsets, (static_cast<std::size_t>(E) + 1) * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, s));
        if (ahead) {
            // The route has read `logits`, so the guess reuses them. It decides placement only, so the
            // router's tensor-core form serves every model (a gated call has at least 256 columns). The guess
            // input (candidates YH-02, imp #3052): the streams normed with the next layer's own MLP norm weights and
            // averaged without the mixer's gates, closer to that layer's MoE input than this layer's x. It lives in
            // `shared` ([H, T]): a gated call is an eager chunk, whose shared expert writes it later on this stream,
            // after the router has read the guess.
            const auto& next          = parameters_.layers[layer + 1].moe;
            const Tensor* next_rows[] = {&next.router, &next.shared_score};
            Tensor& guess_input       = shared;
            ops::hyper_connection_normed_stream_mean(*streams, parameters_.layers[layer + 1].mlp_hc.norm,
                                                     dim(config_.hc.streams), config_.rms_norm_eps, guess_input, s);
            ops::projection_fp32_router_mma(guess_input, next_rows, logits, s);
            ops::MoeRouting guess{a.alloc(DType::I32, {K, T}), a.alloc(DType::FP32, {K, T}), a.alloc(DType::FP32, {T})};
            const DeviceSpan guess_bytes = a.alloc_bytes(ops::moe_dispatch_bytes(E, K * T));
            ops::MoeDispatch guessed     = ops::carve_moe_dispatch(guess_bytes.data, E, K * T);
            ops::moe_route_dispatch(logits, K, guess, guessed, nullptr, s);
            CUDA_CHECK(cudaMemcpyAsync(offsets + E + 1, guessed.offsets, (static_cast<std::size_t>(E) + 1) * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, s));
        }
        // A trimmed call following its reference's routes (R > 0, which copies nothing ahead): the reference's
        // offsets, after this call's.
        if (R > 0) {
            if (ahead) { throw std::logic_error("Qwen4Exp MoE: a trimmed last layer copies nothing ahead"); }
            CUDA_CHECK(cudaMemcpyAsync(offsets + E + 1, route_dispatch.offsets,
                                       (static_cast<std::size_t>(E) + 1) * sizeof(std::int32_t), cudaMemcpyDeviceToHost, s));
        }
        CUDA_CHECK(cudaEventRecord(split_ready_, s));
        CUDA_CHECK(cudaEventSynchronize(split_ready_));
        for (std::int32_t e = 0; e < E; ++e) { split_columns_[static_cast<std::size_t>(e)] = offsets[e + 1] - offsets[e]; }
        if (R > 0) {
            // An expert of this call that its reference makes wide is wide here too (the CPU computes narrow jobs only):
            // the split prices it at the reference's columns, and only the link takes it.
            const std::int32_t* reference = offsets + E + 1;
            for (std::int32_t e = 0; e < E; ++e) {
                const std::int32_t columns = reference[e + 1] - reference[e];
                auto& own                  = split_columns_[static_cast<std::size_t>(e)];
                if (own > 0 && columns > ops::offloaded_moe::kMaxColumns) { own = columns; }
            }
        }
        std::fill(split_cpu_.begin(), split_cpu_.end(), std::uint8_t{0});
        // Experts copied ahead for this layer are on their way: the CPU split neither takes nor prices them.
        std::copy(split_columns_.begin(), split_columns_.end(), split_priced_.begin());
        streaming->mask_ahead(layer, split_priced_);
        const std::uint32_t cpu = batch->split && experts_.split != nullptr && source.cpu.max_jobs > 0
                                      ? experts_.split->choose(layer, T, split_priced_, split_cpu_)
                                      : 0U;
        split_streamed_experts_ += streaming->plan_layer(layer, split_columns_, split_cpu_);
        split_cpu_experts_ += cpu;
        if (cpu > 0) {
            // The call takes the channel's connection and arithmetic and only the link share the policy names (a
            // share field added later stays at its "none"): below split_keep_columns the assisted calls' share of
            // the split's experts, below split_wide_keep_below the wide calls' share, else none, so they all go to
            // the CPU. Until 2026-11-04 the call reset the shares one by one and missed the wide calls' and the
            // assisted calls': from 11-02 every split call kept 51/256 or more of its CPU experts on the link
            // (Flash-Next design §19.3.50).
            const ops::MoeCpuChannel& channel = source.cpu;
            ops::MoeCpuChannel split;
            split.request         = channel.request;
            split.x               = channel.x;
            split.y               = channel.y;
            split.done            = channel.done;
            split.status          = channel.status;
            split.heartbeat       = channel.heartbeat;
            split.sequence        = channel.sequence;
            split.layer           = channel.layer;
            split.max_jobs        = channel.max_jobs;
            split.max_job_columns = channel.max_job_columns;
            split.narrow_a16      = channel.narrow_a16;
            split.ggml_a8         = channel.ggml_a8;
            split.max_columns     = ops::offloaded_moe::kMaxCpuCallColumns;
            split.pcie_divisor    = 0;
            split.wide_from       = 1;
            split.wide_jobs       = static_cast<std::int32_t>(cpu);
            if (T < experts_.split_keep_columns) {
                split.assist_share      = channel.assist_share;
                split.assist_share_wide = channel.assist_share_wide;
                split.assist_ramp_from  = channel.assist_ramp_from;
            } else if (T < experts_.split_wide_keep_below) {
                split.wide_share      = channel.wide_share;
                split.wide_share_from = channel.wide_share_from;
            }
            source.cpu            = split;
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
        // evicted since is read in place from its host record (the same bytes, so the same bits). With the
        // SSD tier, while no expert left both tiers since the span began (stream_whole), an expert evicted
        // since is still in its held frame or in RAM: IQ3_S's exclusive 16K prefill ran 13 passes a layer,
        // 6,500 against the full layout's 6,794 tok/s (2026-10-29).
        if (streaming->complete(layer) && (experts_.host_tables.empty() || batch->stream_whole)) {
            // CSF records cannot be read in place: the evicted ones are staged, still in one pass when they fit the
            // slots (else the call keeps its staging passes).
            const bool csf = source.host_csf != nullptr || source.host_csf_pitch != 0;
            if (!csf || static_cast<std::int32_t>(streaming->uncovered(layer)) <= source.staging_slots) {
                if (csf) {
                    source.one_pass = true;
                } else {
                    source.staging_slots = 0;
                }
                source.overlap_stream = nullptr;
                source.fetch          = {}; // nothing is SSD-only: an unserved record would fail the call
            }
        }
    }
    const bool planned = route_plans && ops::moe_route_dispatch_plan(logits, K, routing, dispatch, route_log, x, source,
                                                                     max_jobs, expert_ws.data, outputs, s,
                                                                     ops::RouterLogits::kFp32, shared_job);
    if (log_tail != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(log_tail, static_cast<const std::int32_t*>(routing.ids.data) + static_cast<std::size_t>(K) * (T - logged),
                                   static_cast<std::size_t>(K) * logged * sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
    }
    if (early) { CUDA_CHECK(cudaStreamWaitEvent(s, experts_.shared_events[3], 0)); }
    if (side && !shared_early && !shared_job) {
        CUDA_CHECK(cudaEventRecord(experts_.shared_events[0], s));
        CUDA_CHECK(cudaStreamWaitEvent(experts_.shared_stream, experts_.shared_events[0], 0));
        shared_s = experts_.shared_stream;
    }
    ops::moe_experts(x, dispatch, source, K, max_jobs, expert_ws.data, outputs, s, /*wait_for_cpu=*/false, planned,
                     /*x_prepared=*/early);
    if (streaming != nullptr && batch->release_stream) { streaming->after_experts(layer, s); }
    if (ahead) {
        // The next layer's guessed experts follow this layer's copies on the link.
        const auto* guessed = static_cast<const std::int32_t*>(split_offsets_.data()) + E + 1;
        for (std::int32_t e = 0; e < E; ++e) { guess_columns_[static_cast<std::size_t>(e)] = guessed[e + 1] - guessed[e]; }
        // Copied ahead: the guessed experts of more than ExpertStream::ahead_rows(T) columns (T / 64, at least 8;
        // design §19.3.12).
        (void)streaming->plan_ahead(layer + 1, guess_columns_, runtime::expert_offload::ExpertStream::ahead_rows(T), 0);
    }
    }
    if (!shared_early && !shared_job) {
        if constexpr (kRuns<A>) { scan_input(sx, layer, ActivationScan::Site::SharedInput); }
        run_shared(shared_s);
    }
    // y = bf16(sum_i w_i y_i + s y_shared): the routed entries (a shared job's T columns follow them).
    const Tensor routed = shared_job ? outputs.slice(1, 0, K * T) : outputs;
    if constexpr (kRuns<A>) {
        ops::moe_experts_cpu_wait(x, dispatch, source, max_jobs, expert_ws.data, outputs, s);
        if (side && !shared_job) {
            if (!shared_early) { CUDA_CHECK(cudaEventRecord(experts_.shared_events[1], shared_s)); }
            CUDA_CHECK(cudaStreamWaitEvent(s, experts_.shared_events[1], 0));
        }
        if (inject != nullptr) {
            ops::moe_combine_inject(routed, routing, shared, *inject, *residual, s);
            return Tensor{};
        }
    } else if (inject != nullptr) {
        (void)early, (void)expert_ws, (void)routed;
        return Tensor{};
    }
    Tensor y = a.alloc(DType::BF16, {H, T});
    if constexpr (kRuns<A>) { ops::moe_combine(routed, routing, shared, y, s); }
    return y;
}


ops::MoeExpertSource Forward::expert_source(const MoeParameters& p, std::uint32_t layer) const {
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
        source.host_records   = nullptr; // the banks hold no records in tier mode
        source.host_table     = experts_.host_tables.at(layer);
        source.host_csf_pitch = experts_.tier_csf_pitch; // compressed RAM slots: staged misses expand as they copy
        if (!experts_.fetch.empty()) { source.fetch = experts_.fetch.at(layer); }
    } else {
        source.host_csf = p.bank->csf.device; // the pinned bank's CSF records: staged misses expand as they copy
    }
    source.error   = experts_.error;
    source.wait_ns = experts_.wait_ns;
    source.a4_shared_input = p.a4_shared_input;
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
                                       .record_bytes   = (planes.record_bytes + 15) / 16 * 16,
                                       .a8            = fast_.a8_experts &&
                                                         planes.gate_up_format != QType::FP8_E4M3FN_B128};
    }
    source.frame_stride = pitch;
    if (pitch != 0 && pitch != source.record_stride) {
        throw std::invalid_argument("Qwen4Exp MoE: the layer's frame pitch differs from its bank's record stride");
    }
    return source;
}

void Forward::run_mtp(const MtpCall& call) {
    if (!parameters_.mtp || !kv_.mtp || state_.mtp_tails.data == nullptr) {
        throw std::logic_error("Qwen4Exp forward: the MTP drafter is not loaded");
    }
    if (call.batch <= 0 || call.width <= 0 || call.ids.numel() != call.batch * call.width ||
        call.residuals.ne[0] != dim(config_.residual_width()) || call.residuals.ne[1] != call.batch * call.width ||
        (!call.kv_only && call.width != 1) ||
        call.rope_positions.ne[0] != call.batch * call.width || call.rope_positions.ne[1] != 3 ||
        call.block_start_rope.ne[0] != call.batch || call.block_start_rope.ne[1] != 3) {
        throw std::invalid_argument("Qwen4Exp forward: MTP call geometry is invalid");
    }
    eager_chunk_ = false;
    work_.reset();
    const std::int32_t W = dim(config_.residual_width()), T = call.batch * call.width;
    Tensor rows = work_.alloc(DType::BF16, {W, T});
    CUDA_CHECK(cudaMemcpyAsync(rows.data, call.residuals.data, rows.bytes(), cudaMemcpyDeviceToDevice, device_.stream));
    mtp_block_impl(work_, &call, T, call.kv_only, rows);
}

template <class A>
void Forward::mtp_block_impl(A& a, const MtpCall* call_ptr, std::int32_t columns, bool kv_only, Tensor& rows) {
    const cudaStream_t s = device_.stream;
    const auto& p        = *parameters_.mtp;
    const std::int32_t H = dim(config_.hidden_size), W = dim(config_.residual_width()), S = dim(config_.hc.streams);
    const std::int32_t T = columns;
    try {
        auto scope = a.scope();
        // Input fusion: R = fc_hidden(per stream of RMSNorm_{S*H}(rows)) + fc_embedding(RMSNorm_H(embed(id))).
        Tensor x0 = a.alloc(DType::BF16, {H, T});
        if constexpr (kRuns<A>) {
            embed_tokens(call_ptr->ids, parameters_.token_embedding, x0, s, experts_.error);
            if (call_ptr->vision != nullptr && call_ptr->vision->columns.data != nullptr) {
                ops::scatter(call_ptr->vision->embeddings, call_ptr->vision->columns, x0, s);
            }
        }
        Tensor e = a.alloc(DType::BF16, {H, T});
        if constexpr (kRuns<A>) { ops::rmsnorm(x0, p.embedding_norm, config_.rms_norm_eps, true, e, s); }
        Tensor ep = a.alloc(DType::BF16, {H, T});
        project(e, p.embedding_projection, ep, a, s);
        Tensor hn = a.alloc(DType::BF16, {W, T});
        if constexpr (kRuns<A>) { ops::rmsnorm(rows, p.hidden_norm, config_.rms_norm_eps, true, hn, s); }
        Tensor residual = a.alloc(DType::BF16, {W, T});
        {
            Tensor per_stream = residual.view({H, S * T});
            project(hn.view({H, S * T}), p.hidden_projection, per_stream, a, s);
        }
        if constexpr (kRuns<A>) { ops::hyper_connection_inject(ep, state_.mtp_ones.slice(1, 0, T), residual, s); }

        Tensor inject = a.alloc(DType::FP32, {S, T});
        Tensor xa     = mix_impl(a, p.attn_hc, residual, &inject);
        Tensor y;
        if constexpr (kRuns<A>) {
            const MtpCall& call = *call_ptr;
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
            y = attention_impl(a, p.attention, xa, &attention_call, attention_shape(attention_call), nullptr);
        } else {
            (void)rows;
            y = attention_impl(a, p.attention, xa, nullptr,
                               AttentionShape{.batch = 1, .width = T, .kv_only = kv_only, .storage = storage_}, nullptr);
        }
        if (kv_only) { return; }
        if constexpr (!kRuns<A>) {
            throw std::logic_error("Qwen4Exp forward: a plan covers the MTP block's K/V-only form");
        } else {
        const MtpCall& call = *call_ptr;
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
        Tensor outputs = work_.alloc(DType::BF16, {H, K * T});
        const DeviceSpan expert_ws = work_.alloc_bytes(ops::resident_moe_workspace_bytes(K * T, I));
        if (experts_.drafter_slots > 0) { // the drafter pool: among the pooled experts, by their slots
            ops::moe_route_pooled(logits, K, experts_.drafter_slot_of, experts_.drafter_demand, routing, s);
            ops::resident_moe_experts(xm, routing.ids, experts_.drafter_gate_up, experts_.drafter_down,
                                      experts_.drafter_slots, I, expert_ws.data, expert_ws.bytes, outputs, s);
        } else {
            ops::moe_route(logits, K, routing, s);
            ops::resident_moe_experts(xm, routing.ids, p.experts_gate_up, p.experts_down, E, I, expert_ws.data,
                                      expert_ws.bytes, outputs, s);
        }
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
        const std::int32_t valid    = head.rows ? rows_out : dim(parameters_.model.resources().public_token_count);
        // With the proposal head's screen bound (--screen-heads artifacts): the rows scored from their 2-bit
        // copy and the near-best ones exactly (ops::screened_head_top), so the draft is the exact head's
        // argmax unless the screen misses it, and a sampled draft's ranks carry exact logits with an
        // approximate normalizer. Draft-only: verification is unchanged.
        const bool screened = head.rows && head.screen;
        Tensor drafts = call.drafts;
        Tensor narrow;
        if (!screened) {
            Tensor head_logits = work_.alloc(DType::FP32, {rows_out, T});
            if (head.rows) {
                ops::projection_fp32(final_x, head.rows->weight, head_logits, s);
            } else {
                text_head_logits(parameters_, final_x, head_logits, work_, s);
            }
            narrow = work_.alloc(DType::BF16, {rows_out, T});
            ops::cast_fp32_to_bf16(head_logits, narrow, s);
        }
        if (call.sample_configs != nullptr) {
            // Sampled drafts: the head's top rows, one draw each under the column's sampler; the top
            // row's log-probability is also the cumulative cut's confidence (a greedy row's).
            Tensor sample_rows     = call.sample_rows;
            Tensor sample_logprobs = call.sample_logprobs;
            if (screened) {
                Tensor unmapped;
                ops::screened_head_top(final_x, head.rows->weight, *head.screen, valid, kScreenedHeadPolicy, sample_rows,
                                       sample_logprobs, unmapped, unmapped, work_, s);
            } else {
                ops::top_logprobs(narrow, valid, sample_rows, sample_logprobs, s);
            }
            Tensor candidates = call.draft_candidates;
            Tensor q          = call.draft_q;
            Tensor confidence = call.draft_logprobs.data != nullptr
                                    ? Tensor(call.draft_logprobs.data, DType::FP32, {T})
                                    : Tensor{};
            ops::speculative_sample_sparse_drafts(
                sample_rows, sample_logprobs, head.rows ? static_cast<const std::int32_t*>(head.token_ids.data) : nullptr,
                call.sample_configs, call.positions, call.draft_step, drafts, candidates, q, confidence, s);
        } else if (screened) {
            // The top row, its token (the draft) and its log-probability (the confidence) in one call.
            Tensor top_row = work_.alloc(DType::I32, {1, T});
            Tensor logprob = call.draft_logprobs.data != nullptr ? Tensor(call.draft_logprobs.data, DType::FP32, {1, T})
                                                                 : work_.alloc(DType::FP32, {1, T});
            ops::screened_head_top(final_x, head.rows->weight, *head.screen, valid, kScreenedHeadPolicy, top_row, logprob,
                                   head.token_ids, drafts, work_, s);
        } else if (head.rows) {
            ops::argmax(narrow, head.token_ids, drafts, rows_out, s);
        } else {
            ops::argmax(narrow, drafts, dim(parameters_.model.resources().public_token_count), s);
        }
        if (!screened && call.draft_logprobs.data != nullptr && call.sample_configs == nullptr) {
            // The top row's log-probability over the head's rows: the draft's own (same rows, same
            // tie order as the argmax above).
            Tensor top_row = work_.alloc(DType::I32, {1, T});
            Tensor logprob(call.draft_logprobs.data, DType::FP32, {1, T});
            ops::top_logprobs(narrow, valid, top_row, logprob, s);
        }
        }
    } catch (const std::exception&) {
        runtime::rethrow_with_context("qwen4_exp/mtp: ");
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
