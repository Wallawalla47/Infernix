// Inkling 2's --lossless route (lossless.h; docs/maintainer/lossless-route.md §2b): one window, every operation of
// modeling_inkling2_loop.py on its exact operands, each rounded once to its dtype.
//
// Layouts: activations [width, T] with column t the window's position t. Attention runs a query head at a time over
// chunks of query columns: the scores S (BF16 [keys, chunk]) are lossless_linear's exact products of the head's
// queries against its KV head's keys (the keys as BF16 weight rows), lossless_inkling2_attention_probs turns them into
// the eager form's BF16 probabilities in place, and the head's output is their exact product with the values (the
// transposed value plane as BF16 weight rows), into its rows of o.
//
// Experts: x quantized four-over-six (or MXFP8) per product column (a column per routed entry, the entries grouped by
// expert into 64-column tiles), the exact gate/up products times amax_e outer / den (NVFP4) rounded to BF16, the
// softcapped SwiGLU times gamma, h quantized the same way, the exact down products rounded to BF16, and the 8 slots
// folded in slot order in binary32 and rounded to BF16 (the trained arithmetic; owner decision D4 pending).

#include "models/inkling2/execution/lossless.h"

#include "core/device.h"
#include "infernix/ops/lossless_inkling2.h"
#include "infernix/ops/lossless_linear.h"
#include "infernix/ops/lossless_quantize.h"
#include "infernix/ops/rows.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace infernix::models::inkling2 {
namespace {

constexpr std::int32_t kQueryChunk  = 256;  // attention's query columns a pass
constexpr std::int32_t kExpertChunk = 1024; // the experts' product columns a pass (whole tiles)
constexpr int kTile                 = 64;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument("Inkling 2 lossless: " + message); }
}

std::int32_t round_up(std::int32_t v, std::int32_t m) { return (v + m - 1) / m * m; }

Tensor matrix(const void* p, DType dtype, std::int32_t rows, std::int32_t columns) {
    return Tensor(const_cast<void*>(p), dtype, {rows, columns});
}

// Columns [first, first + count) of a contiguous [rows, *] tensor.
Tensor columns_of(const Tensor& t, std::int32_t first, std::int32_t count) {
    const std::size_t bytes = static_cast<std::size_t>(t.ne[0]) * dtype_size(t.dtype);
    return Tensor(static_cast<std::uint8_t*>(t.data) + bytes * first, t.dtype, {t.ne[0], count});
}

// A contiguous BF16 [K, N] matrix (N rows of K elements) as a Weight [N, K].
Weight bf16_rows(const void* data, std::int32_t k, std::int32_t n) {
    Weight w{};
    w.qtype    = QType::BF16;
    w.layout   = QuantLayout::Contiguous;
    w.ndim     = 2;
    w.n        = n;
    w.k        = k;
    w.shape[0] = w.padded_shape[0] = n;
    w.shape[1] = w.padded_shape[1] = k;
    w.qdata    = data;
    w.payload  = data;
    return w;
}

std::uint32_t float_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

} // namespace

// The scratch of one visit at window width T (bytes), with room for the projections' workspaces.
std::size_t LosslessForward::device_bytes(const TextConfig& c, std::int32_t max_columns) {
    const std::size_t T = static_cast<std::size_t>(round_up(max_columns, kTile)), H = c.hidden_size;
    const std::size_t Q = c.query_width(), K = c.key_width(), R = c.relative_width();
    const std::size_t I = c.intermediate, E = c.experts, k = c.top_k;
    std::size_t persistent = (9 + 3) * H * T * 2 + T * 8 + static_cast<std::size_t>(kHeadRows) * c.vocab_size * 4;
    std::size_t visit = 0;
    visit += 3 * H * T * 4 + 4 * H * T * 4;        // the attention mixes, the messages
    visit += 2 * H * T * 2 + 64 * T * 2 + 2 * H * T * 4; // a message gate's scratch
    visit += 3 * H * T * 2;                         // x_q, x_k, x_v
    visit += (2 * Q + R + c.heads + 4 * K + 2 * c.kv_heads * c.conv_kernel) * T * 2; // projections and convs
    visit += (Q + 2 * K) * T * 2;                   // the head planes
    visit += T * kQueryChunk * 2;                   // a head's scores
    visit += (2 * Q + 2 * H) * T * 2;               // o, the gated o, W_o's output, the conv
    visit += 3 * H * T * 4 + H * T * 2 + E * T * 4 + 2 * k * T * 4; // rho, the MLP mix, x, the router
    visit += H * (k * T + E * kTile) * 2;           // the experts' outputs by entry
    visit += (H + 3 * I) * kExpertChunk * 2 + 2 * kExpertChunk * 4 + (k * T / kTile + E + 1) * 64 * 16;
    visit += 2 * H * T * 2;                         // the combine, the post-MLP conv
    const std::size_t linear = 512ULL << 20;        // lossless_linear's workspaces (512 columns at a time)
    return persistent + visit + linear + (64ULL << 20);
}

LosslessForward::LosslessForward(const Parameters& parameters, std::int32_t max_columns, bool reference)
    : p_(parameters), c_(parameters.model.config().text), capacity_(round_up(max_columns, kTile)), reference_(reference) {
    require(c_.heads % c_.kv_heads == 0 && c_.head_dim == 128 && c_.relative_dim <= 32, "the attention geometry");
    require(p_.layers.size() == c_.num_hidden_layers, "every layer's operands");
    for (const LayerOperands& l : p_.layers) {
        const AttentionOperands& a = l.attention;
        for (const Weight* w : {&a.query, &a.relative, &a.gate, &a.key, &a.value, &a.sink, &a.output}) {
            require(ops::lossless_linear_supports(*w), "an attention projection's format");
        }
        if (l.full) {
            require(ops::lossless_linear_supports(a.key_taps) && ops::lossless_linear_supports(a.value_taps),
                    "the dynamic convolutions' tap projections");
        }
        if (l.messages) {
            for (const GateOperands& g : l.gates) {
                require(ops::lossless_linear_supports(g.down) && ops::lossless_linear_supports(g.forget_gain),
                        "a message gate's projections");
            }
        }
        require(l.bank.records != nullptr && l.bank.intermediate % kTile == 0 && l.bank.hidden % kTile == 0,
                "a resident expert bank");
        std::vector<float> amax, den;
        if (l.bank.format == ops::w4a4_moe::ExpertFormat::Nvfp4FourOverSix) {
            amax.resize(static_cast<std::size_t>(l.bank.experts) * 2);
            den.resize(2);
            CUDA_CHECK(cudaMemcpy(amax.data(), l.bank.amax, amax.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(den.data(), l.bank.denominators, 8, cudaMemcpyDeviceToHost));
        }
        amax_.push_back(std::move(amax));
        denominators_.push_back(std::move(den));
    }
    require(ops::lossless_linear_supports(p_.head), "the output head's format");
    const std::size_t H = c_.hidden_size, T = static_cast<std::size_t>(capacity_);
    std::size_t persistent = (9 + 3) * H * T * 2 + T * 8 + static_cast<std::size_t>(kHeadRows) * c_.vocab_size * 4 + 4096;
    const std::size_t total = device_bytes(c_, max_columns);
    memory_ = DeviceBuffer(total);
    auto* base = static_cast<std::uint8_t*>(memory_.p);
    std::size_t at = 0;
    const auto take = [&](std::size_t bytes) {
        std::uint8_t* q = base + at;
        at += (bytes + 255) / 256 * 256;
        return q;
    };
    tokens_    = reinterpret_cast<std::int32_t*>(take(T * 4));
    positions_ = reinterpret_cast<std::int32_t*>(take(T * 4));
    blocks_    = reinterpret_cast<std::uint16_t*>(take(9 * H * T * 2));
    hidden_[0] = reinterpret_cast<std::uint16_t*>(take(H * T * 2));
    hidden_[1] = reinterpret_cast<std::uint16_t*>(take(H * T * 2));
    final_     = reinterpret_cast<std::uint16_t*>(take(H * T * 2));
    logits_    = reinterpret_cast<float*>(take(static_cast<std::size_t>(kHeadRows) * c_.vocab_size * 4));
    (void)persistent;
    arena_ = std::make_unique<WorkspaceArena>(DeviceSpan{base + at, total - at});
    std::vector<std::int32_t> positions(T);
    for (std::size_t i = 0; i < T; ++i) { positions[i] = static_cast<std::int32_t>(i); }
    CUDA_CHECK(cudaMemcpy(positions_, positions.data(), T * 4, cudaMemcpyHostToDevice));
}

void LosslessForward::project(const Tensor& x, const Weight& w, std::int32_t row_begin, std::int32_t rows, Tensor& out,
                              cudaStream_t stream) {
    auto scope = arena_->scope();
    ops::lossless_linear(x, w, row_begin, rows, out, 0, *arena_, stream);
}

void LosslessForward::run(std::span<const std::int32_t> tokens, cudaStream_t stream) {
    const auto T = static_cast<std::int32_t>(tokens.size());
    require(T > 0 && T <= capacity_, "a window of 1 to " + std::to_string(capacity_) + " tokens");
    T_ = T;
    const std::int32_t H = static_cast<std::int32_t>(c_.hidden_size);
    CUDA_CHECK(cudaMemcpyAsync(tokens_, tokens.data(), tokens.size() * 4, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    (void)ops::lossless_inkling2_take_undecided(); // a failed window's
    arena_->reset();
    const auto block = [&](int i) { return matrix(blocks_ + static_cast<std::size_t>(i) * H * capacity_, DType::BF16, H, T); };
    Tensor e = block(0), h6 = block(1);
    Tensor h12[3] = {block(2), block(3), block(4)}, h18[3] = {block(5), block(6), block(7)};
    // The embedding: e = RMSNorm(E[t]) (the table read as BF16 [H, V], a column a token).
    {
        Tensor table = matrix(p_.embedding, DType::BF16, H, static_cast<std::int32_t>(c_.vocab_size));
        Tensor ids   = matrix(tokens_, DType::I32, T, 1);
        Tensor raw   = block(8);
        ops::gather_columns(table, ids, raw, stream);
        Tensor w = matrix(p_.embedding_norm, DType::BF16, H, 1);
        ops::lossless_inkling2_rmsnorm(raw, &w, c_.rms_norm_eps, e, stream);
    }
    Tensor hidden[2] = {matrix(hidden_[0], DType::BF16, H, T), matrix(hidden_[1], DType::BF16, H, T)};
    // The visits, with the block sources of reference forward(): blocks are each boundary layer's input.
    const std::uint32_t core_start = c_.loop.core_start, core_end = c_.loop.core_end;
    std::vector<const Tensor*> blocks;
    const Tensor* h = &e;
    int ping = 0;
    const auto run_layer = [&](std::uint32_t v, const Tensor* messages, const Tensor* boundary_copy) {
        const std::uint32_t layer = c_.visit_layer(v);
        const LayerOperands& l    = p_.layers[layer];
        std::vector<const Tensor*> mlp_blocks = blocks;
        if (l.boundary) { mlp_blocks.push_back(h); }
        Tensor& out = hidden[ping];
        visit(v, *h, blocks, mlp_blocks, messages, out, stream);
        if (l.boundary) {
            // The block keeps the layer's input (copied out of the ping-pong rows when it lives there).
            if (boundary_copy != nullptr) {
                CUDA_CHECK(cudaMemcpyAsync(boundary_copy->data, h->data, static_cast<std::size_t>(H) * T * 2,
                                           cudaMemcpyDeviceToDevice, stream));
                blocks.push_back(boundary_copy);
            } else {
                blocks.push_back(h);
            }
        }
        h    = &out;
        ping = 1 - ping;
    };
    // The core: its first layer and the one a block later open blocks (Inkling 2: 6 and 12, the core 6-17).
    const std::uint32_t mid = core_start + c_.residual_block;
    require(core_start % c_.residual_block == 0 && core_end == mid + c_.residual_block && c_.loop.passes <= 3,
            "a core of two attention-residual blocks and at most 3 passes");
    std::uint32_t v = 0;
    for (std::uint32_t layer = 0; layer < core_start; ++layer, ++v) { run_layer(v, nullptr, nullptr); }
    // h6: the prelude's output, the input of every pass.
    CUDA_CHECK(cudaMemcpyAsync(h6.data, h->data, static_cast<std::size_t>(H) * T * 2, cudaMemcpyDeviceToDevice, stream));
    const std::vector<const Tensor*> prelude_blocks = blocks;
    for (std::uint32_t pass = 0; pass < c_.loop.passes; ++pass) {
        blocks = prelude_blocks;
        h      = &h6;
        for (std::uint32_t layer = core_start; layer < core_end; ++layer, ++v) {
            // A boundary keeps its input: h6 itself at the core's start, layer `mid`'s input copied into h12[pass].
            const Tensor* boundary_copy = layer == mid ? &h12[pass] : nullptr;
            if (pass == 0 || !p_.layers[layer].messages) {
                run_layer(v, nullptr, boundary_copy);
                continue;
            }
            // The messages of the previous pass: its h12 and its output h18, mixed by this layer's four queries.
            auto scope = arena_->scope();
            Tensor out[4];
            Tensor* outs[4];
            for (int k = 0; k < 4; ++k) {
                out[k]  = arena_->alloc(DType::FP32, {H, T});
                outs[k] = &out[k];
            }
            const Tensor* sources[2] = {&h12[pass - 1], &h18[pass - 1]};
            Tensor queries = matrix(p_.layers[layer].message_queries, DType::FP32, H, 4);
            ops::lossless_inkling2_mix(sources, queries, outs, stream);
            run_layer(v, out, boundary_copy);
        }
        CUDA_CHECK(cudaMemcpyAsync(h18[pass].data, h->data, static_cast<std::size_t>(H) * T * 2, cudaMemcpyDeviceToDevice,
                                   stream));
        h = &h18[pass];
    }
    for (std::uint32_t layer = core_end; layer < c_.num_hidden_layers; ++layer, ++v) { run_layer(v, nullptr, nullptr); }
    // The output mix over the blocks and the last visit's output, then the final norm.
    {
        auto scope = arena_->scope();
        Tensor mixed = arena_->alloc(DType::FP32, {H, T});
        std::vector<const Tensor*> sources = blocks;
        sources.push_back(h);
        Tensor* outs[1] = {&mixed};
        Tensor query = matrix(p_.output_query, DType::FP32, H, 1);
        ops::lossless_inkling2_mix(sources, query, outs, stream);
        Tensor w = matrix(p_.final_norm, DType::BF16, H, 1);
        Tensor f = matrix(final_, DType::BF16, H, T);
        ops::lossless_inkling2_rmsnorm(mixed, &w, c_.rms_norm_eps, f, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (ops::lossless_inkling2_take_undecided()) {
        throw std::runtime_error("Inkling 2 lossless: a softmax element's rounding was left undecided by its "
                                 "double-double evaluation (no correctly rounded result was established)");
    }
}

void LosslessForward::head(std::int32_t first, std::int32_t count, cudaStream_t stream) {
    require(first >= 0 && count > 0 && count <= kHeadRows && first + count <= T_, "head rows of the window");
    const std::int32_t H = static_cast<std::int32_t>(c_.hidden_size), V = static_cast<std::int32_t>(c_.vocab_size);
    Tensor x = matrix(final_ + static_cast<std::size_t>(first) * H, DType::BF16, H, count);
    Tensor l = matrix(logits_, DType::FP32, V, count);
    project(x, p_.head, 0, V, l, stream);
    ops::lossless_inkling2_softcap(l, c_.final_logit_softcap, l, stream);
}

void LosslessForward::message_gate(const GateOperands& g, Tensor& x, const Tensor& m, cudaStream_t stream) {
    const std::int32_t H = x.ne[0], T = x.ne[1], rank = g.down.n;
    auto scope = arena_->scope();
    Tensor zin = arena_->alloc(DType::BF16, {2 * H, T});
    Tensor lo = zin, hi = zin;
    lo.ne[0] = H;
    hi.ne[0] = H;
    hi.data  = static_cast<std::uint16_t*>(zin.data) + H;
    ops::lossless_inkling2_rmsnorm(x, nullptr, c_.rms_norm_eps, lo, stream);
    ops::lossless_inkling2_rmsnorm(m, nullptr, c_.rms_norm_eps, hi, stream);
    Tensor z = arena_->alloc(DType::BF16, {rank, T});
    project(zin, g.down, 0, rank, z, stream);
    Tensor f = arena_->alloc(DType::FP32, {H, T}), gn = arena_->alloc(DType::FP32, {H, T});
    project(z, g.forget_gain, 0, H, f, stream);
    project(z, g.forget_gain, H, H, gn, stream);
    ops::lossless_inkling2_message_gate(f, gn, x, m, x, stream);
}

void LosslessForward::visit(std::uint32_t v, const Tensor& h, std::span<const Tensor* const> attention_blocks,
                            std::span<const Tensor* const> mlp_blocks, const Tensor* messages, Tensor& out,
                            cudaStream_t stream) {
    const std::uint32_t layer = c_.visit_layer(v);
    const LayerOperands& l    = p_.layers[layer];
    const std::int32_t H = static_cast<std::int32_t>(c_.hidden_size), T = h.ne[1];
    const float eps = c_.rms_norm_eps;
    auto scope = arena_->scope();
    // The attention inputs: the block mix (layer 0: the embedding itself), the messages' gates, the input norms.
    Tensor xq = arena_->alloc(DType::BF16, {H, T}), xk = arena_->alloc(DType::BF16, {H, T}), xv = arena_->alloc(DType::BF16, {H, T});
    {
        auto inner = arena_->scope();
        const int nq = l.full ? 3 : 1;
        if (l.attention_mix_query == nullptr) {
            Tensor w0 = matrix(l.norms[0], DType::BF16, H, 1), w1 = matrix(l.norms[1], DType::BF16, H, 1),
                   w2 = matrix(l.norms[2], DType::BF16, H, 1);
            ops::lossless_inkling2_rmsnorm(h, &w0, eps, xq, stream);
            ops::lossless_inkling2_rmsnorm(h, &w1, eps, xk, stream);
            ops::lossless_inkling2_rmsnorm(h, &w2, eps, xv, stream);
        } else {
            Tensor mixed[3];
            Tensor* outs[3];
            for (int k = 0; k < nq; ++k) {
                mixed[k] = arena_->alloc(DType::FP32, {H, T});
                outs[k]  = &mixed[k];
            }
            std::vector<const Tensor*> sources(attention_blocks.begin(), attention_blocks.end());
            sources.push_back(&h);
            Tensor queries = matrix(l.attention_mix_query, DType::FP32, H, nq);
            ops::lossless_inkling2_mix(sources, queries, std::span<Tensor* const>(outs, static_cast<std::size_t>(nq)), stream);
            // Sliding layers gate (and norm) the one mix output per q, k and v.
            Tensor gated[3];
            for (int k = 0; k < 3; ++k) {
                if (messages != nullptr) {
                    gated[k] = arena_->alloc(DType::FP32, {H, T});
                    CUDA_CHECK(cudaMemcpyAsync(gated[k].data, mixed[nq == 3 ? k : 0].data,
                                               static_cast<std::size_t>(H) * T * 4, cudaMemcpyDeviceToDevice, stream));
                    message_gate(l.gates[k], gated[k], messages[k], stream);
                } else {
                    gated[k] = mixed[nq == 3 ? k : 0];
                }
            }
            Tensor* x3[3] = {&xq, &xk, &xv};
            for (int k = 0; k < 3; ++k) {
                Tensor w = matrix(l.norms[k], DType::BF16, H, 1);
                ops::lossless_inkling2_rmsnorm(gated[k], &w, eps, *x3[k], stream);
            }
        }
    }
    Tensor c = arena_->alloc(DType::BF16, {H, T});
    attention(l, xq, xk, xv, c, stream);
    // The residual: rho = fp32(c) at a block boundary, else fp32(h + c); the MLP mix over the blocks and rho.
    Tensor rho = arena_->alloc(DType::FP32, {H, T});
    Tensor x_mlp = arena_->alloc(DType::BF16, {H, T});
    {
        auto inner = arena_->scope();
        if (l.boundary) {
            ops::lossless_inkling2_convert(c, rho, stream);
        } else {
            ops::lossless_inkling2_add(h, c, rho, stream);
        }
        Tensor mixed = arena_->alloc(DType::FP32, {H, T});
        std::vector<const Tensor*> sources(mlp_blocks.begin(), mlp_blocks.end());
        sources.push_back(&rho);
        Tensor* outs[1] = {&mixed};
        Tensor query = matrix(l.mlp_mix_query, DType::FP32, H, 1);
        ops::lossless_inkling2_mix(sources, query, outs, stream);
        if (messages != nullptr) { message_gate(l.gates[3], mixed, messages[3], stream); }
        Tensor w = matrix(l.mlp_norm, DType::BF16, H, 1);
        ops::lossless_inkling2_rmsnorm(mixed, &w, eps, x_mlp, stream);
    }
    Tensor moe = arena_->alloc(DType::BF16, {H, T});
    experts(l, layer, x_mlp, moe, stream);
    Tensor d = arena_->alloc(DType::BF16, {H, T});
    Tensor taps = matrix(l.mlp_conv, DType::BF16, H, static_cast<std::int32_t>(c_.conv_kernel));
    ops::lossless_inkling2_conv(moe, taps, d, stream);
    // h' = bf16(fp32(rho + d)).
    ops::lossless_inkling2_add(rho, d, out, stream, true);
}

void LosslessForward::attention(const LayerOperands& l, const Tensor& xq, const Tensor& xk, const Tensor& xv, Tensor& c,
                                cudaStream_t stream) {
    const AttentionOperands& a = l.attention;
    const std::int32_t H = static_cast<std::int32_t>(c_.hidden_size), T = xq.ne[1];
    const std::int32_t heads = static_cast<std::int32_t>(c_.heads), kv = static_cast<std::int32_t>(c_.kv_heads);
    const std::int32_t D = static_cast<std::int32_t>(c_.head_dim), rel = static_cast<std::int32_t>(c_.relative_dim);
    const std::int32_t Q = heads * D, KW = kv * D, R = heads * rel, ck = static_cast<std::int32_t>(c_.conv_kernel);
    const std::int32_t Tp = round_up(T, kTile);
    auto scope = arena_->scope();
    Tensor q = arena_->alloc(DType::BF16, {Q, T}), r = arena_->alloc(DType::BF16, {R, T});
    Tensor gate = arena_->alloc(DType::BF16, {Q, T}), sink = arena_->alloc(DType::BF16, {heads, T});
    {
        auto inner = arena_->scope();
        std::vector<ops::LosslessProjection> list = {{&a.query, 0, Q, &q, 0},
                                                     {&a.relative, 0, R, &r, 0},
                                                     {&a.gate, 0, Q, &gate, 0},
                                                     {&a.sink, 0, heads, &sink, 0}};
        ops::lossless_linear_shared(xq, list, *arena_, stream);
    }
    // q: the scaled clip (full layers: then the log scale); r: the same with its threshold.
    ops::LosslessLogScale log_scale;
    Tensor positions = matrix(positions_, DType::I32, T, 1);
    if (l.full) {
        log_scale.positions = &positions;
        log_scale.alpha     = c_.log_scaling_alpha;
        log_scale.floor     = static_cast<float>(c_.log_scaling_floor);
    }
    Tensor qs = matrix(a.layer.query_scale, DType::BF16, D, heads), rs = matrix(a.layer.relative_scale, DType::BF16, rel, heads);
    ops::lossless_inkling2_clip(q, &qs, heads, c_.qk_clip_threshold, log_scale, q, stream);
    ops::lossless_inkling2_clip(r, &rs, heads, c_.relative_clip_threshold, log_scale, r, stream);
    // k and v: the projection, its convolution (static; full layers dynamic), then k's clip and v's per-head norm.
    Tensor k = arena_->alloc(DType::BF16, {KW, T}), vv = arena_->alloc(DType::BF16, {KW, T});
    {
        auto inner = arena_->scope();
        Tensor kp = arena_->alloc(DType::BF16, {KW, T}), vp = arena_->alloc(DType::BF16, {KW, T});
        project(xk, a.key, 0, KW, kp, stream);
        project(xv, a.value, 0, KW, vp, stream);
        Tensor kt = matrix(a.layer.key_conv, DType::BF16, KW, ck), vt = matrix(a.layer.value_conv, DType::BF16, KW, ck);
        if (l.full) {
            Tensor kd = arena_->alloc(DType::BF16, {kv * ck, T}), vd = arena_->alloc(DType::BF16, {kv * ck, T});
            project(xk, a.key_taps, 0, kv * ck, kd, stream);
            project(xv, a.value_taps, 0, kv * ck, vd, stream);
            ops::lossless_inkling2_dynamic_conv(kp, kd, kt, kv, k, stream);
            ops::lossless_inkling2_dynamic_conv(vp, vd, vt, kv, vv, stream);
        } else {
            ops::lossless_inkling2_conv(kp, kt, k, stream);
            ops::lossless_inkling2_conv(vp, vt, vv, stream);
        }
    }
    ops::lossless_inkling2_clip(k, nullptr, kv, c_.qk_clip_threshold, {}, k, stream);
    {
        Tensor rows = matrix(vv.data, DType::BF16, D, kv * T);
        ops::lossless_inkling2_rmsnorm(rows, nullptr, c_.rms_norm_eps, rows, stream);
        // The design reference: V as the bf16 profile stores it (FP16 words, sliding rings and full pools alike).
        if (reference_) { ops::lossless_round_fp16(rows, stream); }
    }
    // Head planes: queries head-major [heads][T][D], keys head-major [kv][Tp][D] and values transposed [kv][D][Tp],
    // zero beyond the window.
    Tensor qh = arena_->alloc(DType::BF16, {D, T, heads});
    Tensor kh = arena_->alloc(DType::BF16, {D, Tp, kv}), vt = arena_->alloc(DType::BF16, {Tp, D, kv});
    CUDA_CHECK(cudaMemsetAsync(kh.data, 0, kh.bytes(), stream));
    CUDA_CHECK(cudaMemsetAsync(vt.data, 0, vt.bytes(), stream));
    ops::lossless_inkling2_heads(q, heads, false, T, qh, stream);
    ops::lossless_inkling2_heads(k, kv, false, Tp, kh, stream);
    ops::lossless_inkling2_heads(vv, kv, true, Tp, vt, stream);
    Tensor o = arena_->alloc(DType::BF16, {Q, T});
    Tensor table = matrix(a.layer.relative_table, DType::BF16, rel, static_cast<std::int32_t>(c_.relative_extent));
    const float scale = static_cast<float>(std::pow(static_cast<double>(D), -0.5));
    const int window = l.full ? 0 : static_cast<int>(c_.sliding_window);
    for (std::int32_t t0 = 0; t0 < T; t0 += kQueryChunk) {
        const std::int32_t n = std::min(kQueryChunk, T - t0);
        const std::int32_t keys = round_up(t0 + n, kTile); // the keys the chunk's queries can see
        auto inner = arena_->scope();
        Tensor s = arena_->alloc(DType::BF16, {Tp, n});
        Tensor o_chunk = columns_of(o, t0, n);
        Tensor rel_chunk = columns_of(r, t0, n), sink_chunk = columns_of(sink, t0, n);
        for (std::int32_t hq = 0; hq < heads; ++hq) {
            const std::int32_t g = hq / (heads / kv);
            Tensor x = matrix(static_cast<std::uint16_t*>(qh.data) + (static_cast<std::size_t>(hq) * T + t0) * D,
                              DType::BF16, D, n);
            const Weight kw = bf16_rows(static_cast<std::uint16_t*>(kh.data) + static_cast<std::size_t>(g) * Tp * D, D, keys);
            {
                auto ws = arena_->scope();
                ops::lossless_linear(x, kw, 0, keys, s, 0, *arena_, stream);
            }
            ops::LosslessInkling2Probs pa;
            pa.scores      = &s;
            pa.ld          = Tp;
            pa.keys        = Tp;
            pa.query_begin = t0;
            pa.head        = hq;
            pa.heads       = heads;
            pa.relative    = &rel_chunk;
            pa.table       = &table;
            pa.sinks       = &sink_chunk;
            pa.window      = window;
            pa.scale       = scale;
            ops::lossless_inkling2_attention_probs(pa, stream);
            const Weight vw = bf16_rows(static_cast<std::uint16_t*>(vt.data) + static_cast<std::size_t>(g) * D * Tp, Tp, D);
            {
                auto ws = arena_->scope();
                ops::lossless_linear(s, vw, 0, D, o_chunk, hq * D, *arena_, stream);
            }
        }
    }
    // The gated output projection and the post-attention convolution.
    Tensor gated = arena_->alloc(DType::BF16, {Q, T});
    ops::lossless_inkling2_sigmoid_gate(o, gate, gated, stream);
    Tensor y = arena_->alloc(DType::BF16, {H, T});
    project(gated, a.output, 0, H, y, stream);
    Tensor taps = matrix(l.attention_conv, DType::BF16, H, ck);
    ops::lossless_inkling2_conv(y, taps, c, stream);
}

void LosslessForward::experts(const LayerOperands& l, std::uint32_t layer, const Tensor& x, Tensor& out,
                              cudaStream_t stream) {
    const std::int32_t H = x.ne[0], T = x.ne[1], E = l.bank.experts, I = l.bank.intermediate;
    const std::int32_t top_k = static_cast<std::int32_t>(c_.top_k);
    const bool nvfp4 = l.bank.format == ops::w4a4_moe::ExpertFormat::Nvfp4FourOverSix;
    auto scope = arena_->scope();
    // The router: l = fp32(x W) exact, then the selection and the weights.
    Tensor logits = arena_->alloc(DType::FP32, {E, T});
    {
        auto inner = arena_->scope();
        ops::lossless_linear(x, bf16_rows(l.router, H, E), 0, E, logits, 0, *arena_, stream);
    }
    Tensor weights = arena_->alloc(DType::FP32, {top_k, T}), ids = arena_->alloc(DType::I32, {top_k, T});
    Tensor bias = matrix(l.router_bias, DType::FP32, E, 1);
    ops::lossless_inkling2_router(logits, bias, top_k, c_.route_scale, weights, ids, stream);
    std::vector<std::int32_t> host_ids(static_cast<std::size_t>(top_k) * T);
    std::vector<float> host_w(host_ids.size());
    CUDA_CHECK(cudaMemcpyAsync(host_ids.data(), ids.data, host_ids.size() * 4, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_w.data(), weights.data, host_w.size() * 4, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // Entries (t top_k + j) grouped by expert in ascending entry order, each expert's run cut into 64-column tiles.
    std::vector<std::vector<std::int32_t>> by_expert(static_cast<std::size_t>(E));
    for (std::size_t i = 0; i < host_ids.size(); ++i) {
        const std::int32_t e = host_ids[i];
        require(e >= 0 && e < E, "a routed expert id");
        by_expert[static_cast<std::size_t>(e)].push_back(static_cast<std::int32_t>(i));
    }
    std::vector<ops::LosslessTile> tiles_gu, tiles_down;
    std::vector<std::int32_t> in_columns, out_columns;
    std::vector<float> gamma;
    for (std::int32_t e = 0; e < E; ++e) {
        const auto& entries = by_expert[static_cast<std::size_t>(e)];
        for (std::size_t c0 = 0; c0 < entries.size(); c0 += kTile) {
            const std::uint8_t* record = l.bank.records + static_cast<std::size_t>(e) * l.bank.record_stride;
            const std::uint32_t a_gu   = nvfp4 ? float_bits(amax_[layer][2 * static_cast<std::size_t>(e)]) : 0U;
            const std::uint32_t a_down = nvfp4 ? float_bits(amax_[layer][2 * static_cast<std::size_t>(e) + 1]) : 0U;
            tiles_gu.push_back(ops::LosslessTile{record, {a_gu, a_gu}});
            tiles_down.push_back(ops::LosslessTile{record, {a_down, a_down}});
            for (std::size_t c = c0; c < c0 + kTile; ++c) {
                const bool real = c < entries.size();
                const std::int32_t entry = real ? entries[c] : -1;
                in_columns.push_back(real ? entry / top_k : -1);
                out_columns.push_back(entry);
                gamma.push_back(real ? host_w[static_cast<std::size_t>(entry)] : 0.0F);
            }
        }
    }
    const auto product_columns = static_cast<std::int32_t>(in_columns.size());
    // The record's matrices.
    ops::LosslessBankMatrix gu, down;
    gu.qtype = down.qtype = nvfp4 ? QType::NVFP4_MUL : QType::FP8_E4M3FN_G32_E8M0;
    gu.rows   = 2 * I;
    gu.k      = H;
    down.rows = H;
    down.k    = I;
    down.codes_offset = nvfp4 ? static_cast<std::uint64_t>(2 * I / 16) * (H / 16) * 144
                              : static_cast<std::uint64_t>(2 * I / 16) * (H / 32) * 528;
    // Device copies of the plan.
    Tensor d_in = arena_->alloc(DType::I32, {product_columns}), d_out = arena_->alloc(DType::I32, {product_columns});
    Tensor d_gamma = arena_->alloc(DType::FP32, {product_columns});
    const auto tiles = static_cast<std::int32_t>(tiles_gu.size());
    auto* d_tiles = static_cast<ops::LosslessTile*>(arena_->alloc_bytes(2 * static_cast<std::size_t>(tiles) * sizeof(ops::LosslessTile)).data);
    CUDA_CHECK(cudaMemcpyAsync(d_in.data, in_columns.data(), in_columns.size() * 4, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_out.data, out_columns.data(), out_columns.size() * 4, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_gamma.data, gamma.data(), gamma.size() * 4, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_tiles, tiles_gu.data(), tiles_gu.size() * sizeof(ops::LosslessTile), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_tiles + tiles, tiles_down.data(), tiles_down.size() * sizeof(ops::LosslessTile),
                               cudaMemcpyHostToDevice, stream));
    Tensor y = arena_->alloc(DType::BF16, {H, T * top_k});
    const float den_gu = nvfp4 ? denominators_[layer][0] : 0.0F, den_down = nvfp4 ? denominators_[layer][1] : 0.0F;
    for (std::int32_t c0 = 0; c0 < product_columns; c0 += kExpertChunk) {
        const std::int32_t n = std::min(kExpertChunk, product_columns - c0);
        auto inner = arena_->scope();
        const auto* in_map = static_cast<const std::int32_t*>(d_in.data) + c0;
        Tensor xq = arena_->alloc(DType::BF16, {H, n}), outer = arena_->alloc(DType::FP32, {n});
        if (nvfp4) {
            ops::lossless_inkling2_quantize_four_over_six(x, in_map, n, xq, outer, stream);
        } else {
            ops::lossless_inkling2_quantize_mxfp8(x, in_map, n, xq, stream);
        }
        Tensor gate_up = arena_->alloc(DType::BF16, {2 * I, n});
        {
            auto ws = arena_->scope();
            ops::LosslessActivationScale s;
            if (nvfp4) {
                s.divisor       = den_gu;
                s.column_scales = static_cast<const float*>(outer.data);
            }
            ops::lossless_linear_tiles(xq, nullptr, n, gu, d_tiles + c0 / kTile, gate_up, nullptr, *arena_, stream, s);
        }
        Tensor hcol = arena_->alloc(DType::BF16, {I, n});
        Tensor g = matrix(static_cast<const float*>(d_gamma.data) + c0, DType::FP32, n, 1);
        ops::lossless_inkling2_swiglu(gate_up, g, l.bank.gate_softcap, l.bank.up_softcap, hcol, stream);
        Tensor hq = arena_->alloc(DType::BF16, {I, n}), outer_h = arena_->alloc(DType::FP32, {n});
        if (nvfp4) {
            ops::lossless_inkling2_quantize_four_over_six(hcol, nullptr, n, hq, outer_h, stream);
        } else {
            ops::lossless_inkling2_quantize_mxfp8(hcol, nullptr, n, hq, stream);
        }
        {
            auto ws = arena_->scope();
            ops::LosslessActivationScale s;
            if (nvfp4) {
                s.divisor       = den_down;
                s.column_scales = static_cast<const float*>(outer_h.data);
            }
            ops::lossless_linear_tiles(hq, nullptr, n, down, d_tiles + tiles + c0 / kTile, y,
                                       static_cast<const std::int32_t*>(d_out.data) + c0, *arena_, stream, s);
        }
    }
    // The combine: entry t top_k + j is slot j of column t.
    std::vector<std::int32_t> identity(static_cast<std::size_t>(T) * top_k);
    for (std::size_t i = 0; i < identity.size(); ++i) { identity[i] = static_cast<std::int32_t>(i); }
    Tensor slots = arena_->alloc(DType::I32, {T * top_k});
    CUDA_CHECK(cudaMemcpyAsync(slots.data, identity.data(), identity.size() * 4, cudaMemcpyHostToDevice, stream));
    ops::lossless_inkling2_combine(y, slots, top_k, out, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream)); // the host vectors above outlive their copies
}

} // namespace infernix::models::inkling2
