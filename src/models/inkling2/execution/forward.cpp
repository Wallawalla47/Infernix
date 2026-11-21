#include "models/inkling2/execution/forward.h"

#include "core/device.h"
#include "core/tensor.h"
#include "infernix/ops/inkling2_projections.h"
#include "infernix/ops/inkling2_visit.h"
#include "infernix/ops/linear.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace infernix::models::inkling2 {
namespace {

namespace ik  = ops::inkling2;
namespace moe = ops::w4a4_moe;

constexpr std::int32_t kH        = ik::kHidden;
constexpr std::int32_t kQ        = ik::kHeads * ik::kHeadDim;
constexpr std::int32_t kR        = ik::kHeads * ik::kRelDim;
constexpr std::int32_t kKv       = ik::kKvHeads * ik::kHeadDim;
constexpr std::int32_t kTaps     = ik::kKvHeads * 4;
constexpr std::int32_t kSlice    = 8; // projection columns a call: the BF16 skinny route, whose column bits never
                                      // depend on the call's width
constexpr std::int64_t ring_words(std::int32_t width) { return std::int64_t{ik::kRingPositions} * width; }
constexpr std::int64_t kWindowWords = std::int64_t{ik::kKvHeads} * ik::kWindowRingPages * ik::kKeyUnit * ik::kHeadDim;

// The block layout this forward is written for: blocks of 6, the core at 6-17 run 3 times, 24 layers.
void check_schedule(const TextConfig& c) {
    if (c.num_hidden_layers != 24 || c.residual_block != 6 || c.loop.core_start != 6 || c.loop.core_end != 18 ||
        c.loop.passes != 3 || c.visits() != 48) {
        throw std::invalid_argument("inkling2 forward: Inkling 2 Mini's schedule (24 layers, blocks of 6, core 6-17 x 3)");
    }
}

std::size_t align(std::size_t v) { return (v + 255) / 256 * 256; }

// A BF16 or bf16_p12 matrix as one segment of a fused projection launch.
void add_segment(ik::ProjectionArgs& args, const Weight& w, std::int32_t input, void* out) {
    if (w.qdata == nullptr || args.count >= ik::kMaxProjectionSegments) {
        throw std::invalid_argument("inkling2 forward: a fused projection segment is missing");
    }
    ik::ProjectionSegment s;
    s.rows  = w.n;
    s.input = input;
    s.out   = out;
    if (w.qtype == QType::BF16 && w.layout == QuantLayout::Contiguous) {
        s.weight = static_cast<const std::uint16_t*>(w.qdata);
    } else if (w.qtype == QType::BF16_P12 && w.layout == QuantLayout::P12Rows) {
        s.codes   = w.qdata;
        s.nibbles = w.qhigh;
        s.words   = static_cast<const unsigned*>(w.scales);
        s.escapes = static_cast<const unsigned*>(w.escapes);
    } else {
        throw std::invalid_argument("inkling2 forward: the fused projections take BF16 or bf16_p12 matrices");
    }
    args.segments[args.count++] = s;
}

} // namespace

StateLayout state_layout(const TextConfig& c) {
    check_schedule(c);
    StateLayout out;
    std::int64_t at = 0;
    for (std::uint32_t v = 0; v < c.visits(); ++v) {
        VisitState s;
        s.key_ring       = at, at += ring_words(kKv);
        s.value_ring     = at, at += ring_words(kKv);
        s.attention_ring = at, at += ring_words(kH);
        s.mlp_ring       = at, at += ring_words(kH);
        if (c.layer_types[c.visit_layer(v)] == AttentionKind::Full) {
            s.full_slot = out.full_visits++;
        } else {
            s.window_k = at, at += kWindowWords;
            s.window_v = at, at += kWindowWords;
        }
        out.visits.push_back(s);
    }
    out.lane_words = at;
    return out;
}

Forward::Forward(const Parameters& parameters, std::int32_t max_lanes)
    : p_(parameters), c_(parameters.model.config().text), layout_(state_layout(c_)), max_lanes_(max_lanes) {
    if (max_lanes <= 0 || max_lanes > ik::kMaxAttentionLanes) {
        throw std::invalid_argument("inkling2 forward: 1-8 lanes");
    }
    constexpr std::size_t T = kMaxForwardColumns;
    const std::size_t V = c_.vocab_size, rank = c_.loop.message_rank;
    std::size_t at = 0;
    const auto take = [&at](std::size_t bytes) {
        const std::size_t here = at;
        at                     = align(at + bytes);
        return here;
    };
    std::vector<std::pair<void**, std::size_t>> slots;
    const auto add = [&](auto*& pointer, std::size_t bytes) {
        slots.emplace_back(reinterpret_cast<void**>(&pointer), take(bytes));
    };
    const std::size_t row16 = T * kH * 2;
    add(b_.tokens, T * 4), add(b_.lane, T * 4), add(b_.position, T * 4), add(b_.first, ik::kMaxAttentionLanes * 4);
    add(b_.lane_columns, 2 * ik::kMaxAttentionLanes * 4);
    add(b_.e, row16), add(b_.h6, row16), add(b_.hp, row16);
    for (auto*& h : b_.h12) { add(h, row16); }
    for (auto*& h : b_.h18) { add(h, row16); }
    add(b_.cand, T * kH * 4);
    for (auto*& x : b_.x) { add(x, row16); }
    add(b_.q, T * kQ * 2), add(b_.r, T * kR * 2), add(b_.gate, T * kQ * 2), add(b_.k, T * kKv * 2);
    add(b_.v, T * kKv * 2), add(b_.sink, T * ik::kHeads * 2), add(b_.key_taps, T * kTaps * 2);
    add(b_.value_taps, T * kTaps * 2), add(b_.attention, T * kQ * 2), add(b_.o, row16), add(b_.xm, row16);
    add(b_.rows, T * std::max(moe::row_bytes(moe::ExpertFormat::Nvfp4FourOverSix, kH),
                              moe::row_bytes(moe::ExpertFormat::Mxfp8, kH)));
    add(b_.amax, T * 4), add(b_.experts, T * moe::kTopK * kH * 2);
    add(b_.mixed, 3 * T * kH * 4), add(b_.message, 3 * T * kH * 4), add(b_.fg, 3 * T * 2 * kH * 4);
    add(b_.gate_input, 3 * T * 2 * kH * 2), add(b_.z, 3 * T * rank * 2);
    add(b_.final_hidden, row16), add(b_.logits, T * V * 4);
    std::uint8_t* moe_base = nullptr;
    std::uint8_t* attention_base = nullptr;
    add(moe_base, moe::workspace_bytes(static_cast<std::int32_t>(c_.experts), static_cast<std::int32_t>(c_.intermediate)));
    add(attention_base, ik::attention_workspace_bytes());
    memory_ = DeviceBuffer(at);
    memory_.fill(0); // the Ops' counters and the dispatch state start at zero once
    auto* base = static_cast<std::uint8_t*>(memory_.p);
    for (auto& [pointer, offset] : slots) { *pointer = base + offset; }
    b_.moe          = moe::carve_workspace(moe_base, static_cast<std::int32_t>(c_.experts),
                                           static_cast<std::int32_t>(c_.intermediate));
    b_.attention_ws = ik::carve_attention_workspace(attention_base);
}

std::vector<std::int32_t> Forward::layout_call(const CallPlan& call) {
    const std::int32_t T = call.columns(), B = call.lanes();
    if (T <= 0 || T > kMaxForwardColumns || B <= 0 || B > max_lanes_ || call.lane.size() != call.tokens.size() ||
        call.position.size() != call.tokens.size()) {
        throw std::invalid_argument("inkling2 forward: a call of 1-128 columns over the forward's lanes");
    }
    std::vector<std::int32_t> lane_columns(2 * ik::kMaxAttentionLanes, 0);
    std::int32_t widest = 1;
    for (std::int32_t t = 0; t < T; ++t) {
        const std::int32_t l = call.lane[t];
        if (l < -1 || l >= B) { throw std::invalid_argument("inkling2 forward: a column's lane is outside the call"); }
        if (l < 0) { continue; }
        if (lane_columns[2 * l + 1] == 0) {
            lane_columns[2 * l] = t;
        } else if (call.lane[t - 1] != l || call.position[t] != call.position[t - 1] + 1) {
            throw std::invalid_argument("inkling2 forward: a lane's columns are consecutive at consecutive positions");
        }
        widest = std::max(widest, ++lane_columns[2 * l + 1]);
    }
    if (widest > ik::kMaxAttentionLaneColumns) {
        throw std::invalid_argument("inkling2 forward: at most 16 columns a lane");
    }
    columns_          = T;
    lanes_            = B;
    max_lane_columns_ = widest;
    return lane_columns;
}

void Forward::prepare(const CallPlan& call) {
    const std::vector<std::int32_t> lane_columns = layout_call(call);
    const auto copy = [](void* dst, const std::vector<std::int32_t>& src) {
        CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * 4, cudaMemcpyHostToDevice));
    };
    copy(b_.tokens, call.tokens), copy(b_.lane, call.lane), copy(b_.position, call.position);
    copy(b_.first, call.first), copy(b_.lane_columns, lane_columns);
}

void Forward::upload(const CallPlan& call, std::int32_t* staging, cudaStream_t stream) {
    const std::vector<std::int32_t> lane_columns = layout_call(call);
    std::int32_t* at = staging;
    const auto copy = [&](void* dst, const std::vector<std::int32_t>& src) {
        std::copy(src.begin(), src.end(), at);
        CUDA_CHECK(cudaMemcpyAsync(dst, at, src.size() * 4, cudaMemcpyHostToDevice, stream));
        at += src.size();
    };
    copy(b_.tokens, call.tokens), copy(b_.lane, call.lane), copy(b_.position, call.position);
    copy(b_.first, call.first), copy(b_.lane_columns, lane_columns);
}

void Forward::embed(cudaStream_t stream) {
    ik::embed(b_.tokens, columns_, p_.embedding, p_.embedding_norm, b_.e, stream);
}

void Forward::project(const std::uint16_t* x, const Weight& w, void* out, bool fp32, cudaStream_t stream,
                      std::int32_t from, std::int32_t columns) const {
    const std::int32_t K = w.k, N = w.n;
    const std::size_t word = fp32 ? 4 : 2;
    const std::int32_t end = columns < 0 ? columns_ : from + columns;
    for (std::int32_t first = from; first < end; first += kSlice) {
        const std::int32_t count = std::min(kSlice, end - first);
        Tensor xt(const_cast<std::uint16_t*>(x) + static_cast<std::size_t>(first) * K, DType::BF16, {K, count});
        Tensor ot(static_cast<std::uint8_t*>(out) + static_cast<std::size_t>(first) * N * word,
                  fp32 ? DType::FP32 : DType::BF16, {N, count});
        ops::linear(xt, w, ot, stream);
    }
}

ik::Ring Forward::ring(const StateView& s, std::int64_t offset, std::int32_t width) const {
    return {s.images + offset, width, s.lane_words};
}

ik::Blocks Forward::blocks(std::uint32_t layer, std::uint32_t pass, bool include_current) const {
    ik::Blocks out;
    const auto take = [&](std::uint32_t boundary, const std::uint16_t* source) {
        if (boundary < layer || (boundary == layer && include_current)) { out.rows[out.count++] = source; }
    };
    const std::uint32_t last = c_.loop.passes - 1;
    take(0, b_.e);
    take(6, b_.h6);
    take(12, b_.h12[c_.in_core(layer) ? pass : last]);
    take(18, b_.h18[last]);
    return out;
}

std::uint16_t* Forward::tail_out(std::uint32_t visit) const {
    const std::uint32_t layer = c_.visit_layer(visit), pass = c_.visit_pass(visit);
    if (layer == 6) { return pass == 0 ? b_.h6 : b_.h18[pass - 1]; }
    if (layer == 12) { return b_.h12[pass]; }
    if (layer == 18) { return b_.h18[c_.loop.passes - 1]; }
    return b_.hp;
}

VisitInputs Forward::inputs(std::uint32_t visit) const {
    const std::uint32_t layer = c_.visit_layer(visit), pass = c_.visit_pass(visit);
    VisitInputs in;
    if (visit > 0) { in.tail_out = tail_out(visit); }
    in.input            = visit == 0 ? b_.e : (layer == 6 && pass > 0 ? b_.h6 : in.tail_out);
    in.attention_blocks = blocks(layer, pass, false);
    in.mlp_blocks       = blocks(layer, pass, p_.layers.at(layer).boundary);
    if (pass > 0) {
        in.message_sources[0] = b_.h12[pass - 1];
        in.message_sources[1] = b_.h18[pass - 1];
    }
    return in;
}

FinalInputs Forward::final_inputs() const {
    return {b_.hp, blocks(c_.num_hidden_layers, 0, false)};
}

void Forward::visit(std::uint32_t v, const StateView& s, const VisitInputs& in, cudaStream_t stream) {
    const std::uint32_t layer = c_.visit_layer(v), pass = c_.visit_pass(v);
    const LayerOperands& op   = p_.layers.at(layer);
    const VisitState& vs      = layout_.visits.at(v);
    const bool messages       = op.messages && pass > 0;
    const std::int32_t T      = columns_;
    const std::size_t TH      = static_cast<std::size_t>(T) * kH;
    const ik::ColumnMap map{b_.lane, b_.position, b_.first};
    const std::uint16_t* h = in.input;

    // K1 (K1a): the previous visit's tail, the attention mix, the norms or the messages.
    ik::EntryArgs e;
    e.columns = T;
    e.map     = map;
    if (v > 0) {
        const std::uint32_t prev = v - 1;
        e.tail     = {b_.experts, b_.cand, p_.layers[c_.visit_layer(prev)].mlp_conv,
                      ring(s, layout_.visits[prev].mlp_ring, kH)};
        e.tail_out = in.tail_out;
        e.input    = h == in.tail_out ? nullptr : h;
    } else {
        e.input = h;
    }
    if (layer > 0) {
        e.blocks  = in.attention_blocks;
        e.queries = op.attention_mix_query;
        e.nq      = op.full ? 3 : 1;
    }
    if (messages) {
        e.message_queries    = op.message_queries;
        e.message_sources[0] = in.message_sources[0];
        e.message_sources[1] = in.message_sources[1];
        e.mixed              = b_.mixed;
        e.message            = b_.message;
        e.gate_input         = b_.gate_input;
    } else {
        for (int i = 0; i < 3; ++i) { e.norms[i] = op.norms[i]; }
        e.x[0] = b_.x[0];
        if (op.full) { e.x[1] = b_.x[1], e.x[2] = b_.x[2]; }
    }
    ik::visit_entry(e, stream);
    if (messages) {
        // K1b: the three kinds' gate projections (one launch for the downs, one for forget and gain), then the gating
        // and norms.
        const std::size_t rank = c_.loop.message_rank;
        ik::ProjectionArgs down{}, fg{};
        down.columns = fg.columns = T;
        down.k                    = 2 * kH;
        fg.k                      = static_cast<std::int32_t>(rank);
        fg.fp32_out               = true;
        for (int k = 0; k < 3; ++k) {
            down.x[k] = b_.gate_input + k * TH * 2;
            fg.x[k]   = b_.z + k * T * rank;
            add_segment(down, op.gates[k].down, k, b_.z + k * T * rank);
            add_segment(fg, op.gates[k].forget_gain, k, b_.fg + k * TH * 2);
        }
        ik::projections(down, stream);
        ik::projections(fg, stream);
        ik::GateExitArgs g;
        g.columns  = T;
        g.kinds    = 3;
        g.mixed    = b_.mixed;
        g.shared_x = !op.full;
        g.message  = b_.message;
        g.fg       = b_.fg;
        for (int k = 0; k < 3; ++k) { g.norms[k] = op.norms[k], g.x[k] = b_.x[k]; }
        ik::gate_exit(g, stream);
    }
    const bool three   = op.full || messages;
    const std::uint16_t* xq = b_.x[0];
    const std::uint16_t* xk = three ? b_.x[1] : b_.x[0];
    const std::uint16_t* xv = three ? b_.x[2] : b_.x[0];

    // K2: every projection of the visit's inputs in one launch (the BF16 skinny route's arithmetic).
    const AttentionOperands& a = op.attention;
    ik::ProjectionArgs k2{};
    k2.columns = T;
    k2.k       = kH;
    k2.x[0] = xq, k2.x[1] = xk, k2.x[2] = xv;
    add_segment(k2, a.query, 0, b_.q);
    add_segment(k2, a.relative, 0, b_.r);
    add_segment(k2, a.gate, 0, b_.gate);
    add_segment(k2, a.sink, 0, b_.sink);
    add_segment(k2, a.key, 1, b_.k);
    add_segment(k2, a.value, 2, b_.v);
    if (op.full) {
        add_segment(k2, a.key_taps, 1, b_.key_taps);
        add_segment(k2, a.value_taps, 2, b_.value_taps);
    }
    ik::projections(k2, stream);

    // K3: attention.
    ik::AttentionArgs at;
    at.columns          = T;
    at.lanes            = lanes_;
    at.max_lane_columns = max_lane_columns_;
    at.map              = map;
    at.lane_columns     = b_.lane_columns;
    at.in               = {b_.q, b_.r, b_.k, b_.v, b_.sink, b_.gate, op.full ? b_.key_taps : nullptr,
                           op.full ? b_.value_taps : nullptr};
    at.layer            = a.layer;
    at.key_ring         = ring(s, vs.key_ring, kKv);
    at.value_ring       = ring(s, vs.value_ring, kKv);
    if (op.full) {
        at.kv = {s.pool_k.at(vs.full_slot), s.pool_v.at(vs.full_slot), 0, s.table, s.table_stride, s.pool_pages};
    } else {
        at.kv = {s.images + vs.window_k, s.images + vs.window_v, s.lane_words, nullptr, 0, 0};
    }
    at.ws  = b_.attention_ws;
    at.out = b_.attention;
    ik::attention(at, stream);

    // K4: W_o; K5 (K5a): its post-attention conv and residual, the MLP mix, norm and the experts' rows.
    project(b_.attention, a.output, b_.o, false, stream);
    const ik::ExpertRows rows{op.format, b_.rows, b_.amax};
    ik::MlpEntryArgs m;
    m.columns  = T;
    m.map      = map;
    m.o        = b_.o;
    m.conv     = op.attention_conv;
    m.ring     = ring(s, vs.attention_ring, kH);
    m.h        = h;
    m.boundary = op.boundary;
    m.cand     = b_.cand;
    m.blocks   = in.mlp_blocks;
    m.query    = op.mlp_mix_query;
    if (messages) {
        m.message_query      = op.message_queries + 3 * kH;
        m.message_sources[0] = in.message_sources[0];
        m.message_sources[1] = in.message_sources[1];
        m.mixed              = b_.mixed;
        m.message            = b_.message;
        m.gate_input         = b_.gate_input;
    } else {
        m.norm = op.mlp_norm;
        m.x    = b_.xm;
        m.rows = rows;
    }
    ik::mlp_entry(m, stream);
    if (messages) {
        ik::ProjectionArgs down{}, fg{};
        down.columns = fg.columns = T;
        down.k                    = 2 * kH;
        fg.k                      = static_cast<std::int32_t>(c_.loop.message_rank);
        fg.fp32_out               = true;
        down.x[0]                 = b_.gate_input;
        fg.x[0]                   = b_.z;
        add_segment(down, op.gates[3].down, 0, b_.z);
        add_segment(fg, op.gates[3].forget_gain, 0, b_.fg);
        ik::projections(down, stream);
        ik::projections(fg, stream);
        ik::GateExitArgs g;
        g.columns  = T;
        g.kinds    = 1;
        g.mixed    = b_.mixed;
        g.message  = b_.message;
        g.fg       = b_.fg;
        g.norms[0] = op.mlp_norm;
        g.x[0]     = b_.xm;
        g.rows     = rows;
        ik::gate_exit(g, stream);
    }

    // K6-K8: route and the experts (their outputs combine in the next visit's entry).
    moe::moe_route(b_.xm, T, op.router, op.router_bias, op.bank.experts, kH, c_.route_scale, b_.moe, stream);
    moe::moe_experts({b_.rows, b_.amax, T}, moe::routed_routing(b_.moe), op.bank, b_.moe, b_.experts, stream);
}

void Forward::finish(const StateView& s, const FinalInputs& in, cudaStream_t stream, std::int32_t head_first,
                     std::int32_t head_columns) {
    const std::uint32_t last = c_.visits() - 1;
    ik::FinalArgs f;
    f.columns  = columns_;
    f.map      = {b_.lane, b_.position, b_.first};
    f.tail     = {b_.experts, b_.cand, p_.layers[c_.visit_layer(last)].mlp_conv,
                  ring(s, layout_.visits[last].mlp_ring, kH)};
    f.tail_out = in.tail_out;
    f.blocks   = in.blocks;
    f.query    = p_.output_query;
    f.norm     = p_.final_norm;
    f.out      = b_.final_hidden;
    ik::final_mix(f, stream);
    // The head over the requested columns only (a prefill chunk needs none, or its last): their logits rows.
    const std::int32_t count = head_columns < 0 ? columns_ - head_first : head_columns;
    if (head_first < 0 || count < 0 || head_first + count > columns_) {
        throw std::invalid_argument("inkling2 forward: the head's columns are outside the call");
    }
    if (count == 0) { return; }
    // The rows go to the buffer's start (row r holds column head_first + r): a logits row is 8 bytes off 16-byte
    // alignment, so an odd column's own row could not take the head's output.
    const std::uint16_t* x = b_.final_hidden + static_cast<std::size_t>(head_first) * kH;
    project(x, p_.head, b_.logits, true, stream, 0, count);
    ik::softcap(b_.logits, static_cast<std::int64_t>(count) * c_.vocab_size, c_.final_logit_softcap, stream);
}

} // namespace infernix::models::inkling2
