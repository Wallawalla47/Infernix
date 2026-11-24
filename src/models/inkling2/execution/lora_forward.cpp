#include "models/inkling2/execution/lora_forward.h"

#include "infernix/ops/inkling2_lora.h"

#include <algorithm>
#include <stdexcept>

namespace infernix::models::inkling2 {
namespace {

namespace ik = ops::inkling2;

// A prompt call's terms on tensor cores (K a multiple of 64), a decode-route call's on projections_dmma: the same route
// the Forward's base projections take at these K (2304, 3072, 4608, and the gates' 64).
void shrink(const ik::ProjectionArgs& args, bool prompt, cudaStream_t stream) {
    if (prompt) {
        ik::projections_mma(args, stream);
    } else {
        ik::projections_dmma(args, stream);
    }
}

// A base launch that can carry BF16 A rows: BF16 storage on 16-bit inputs.
bool takes_bf16_rows(const ik::ProjectionArgs& args) {
    return args.count > 0 && args.segments[0].weight != nullptr && args.x8[0] == nullptr && args.x8[1] == nullptr &&
           args.x8[2] == nullptr;
}

void append_segment(ik::ProjectionArgs& args, const std::uint16_t* a, std::int32_t rows, int input, std::uint16_t* u) {
    ik::ProjectionSegment s;
    s.weight = a;
    s.rows   = rows;
    s.input  = input;
    s.out    = u;
    args.segments[args.count++] = s;
}

ik::LoraTarget target_of(const LoraTerm& t, const std::uint16_t* u, std::int32_t u_stride, void* y, std::int32_t y_stride) {
    ik::LoraTarget target;
    target.b        = t.b;
    target.u        = u;
    target.y        = y;
    target.rows     = t.rows;
    target.y_stride = y_stride;
    target.scale    = t.scale;
    target.u_stride = u_stride;
    return target;
}

// The rank of a group's present terms (one rank: LoraBinding::fast_layout).
template <class... Terms>
std::int32_t group_rank(const Terms&... terms) {
    return std::max({(terms.present() ? terms.rank : 0)...});
}

} // namespace

std::size_t LoraForward::device_bytes(std::int32_t rank, std::int32_t capacity) {
    return (static_cast<std::size_t>(kSlots) * capacity * rank * 2 + 255) / 256 * 256;
}

LoraForward::LoraForward(const LoraParameters& parameters, const TextConfig& config, std::int32_t capacity)
    : p_(parameters), hidden_(static_cast<std::int32_t>(config.hidden_size)),
      query_(static_cast<std::int32_t>(config.query_width())), gate_rank_(static_cast<std::int32_t>(config.loop.message_rank)),
      capacity_(capacity), rank_(parameters.rank), u_(device_bytes(parameters.rank, capacity)) {
    if (parameters.lossless) { throw std::logic_error("inkling2 lora: the lossless route has its own terms"); }
}

LoraForward::AttentionStack LoraForward::attention_stack(const LoraLayerTerms<LoraTerm>& l) {
    AttentionStack s;
    s.rank = group_rank(l.query, l.relative, l.gate, l.sink, l.key, l.key_taps, l.value, l.value_taps);
    const auto add = [&](int input, const LoraTerm& t) {
        if (!t.present()) { return; }
        if (s.a[input] == nullptr) { s.a[input] = t.a; }
        s.rows[input] += s.rank;
    };
    add(0, l.query), add(0, l.relative), add(0, l.gate), add(0, l.sink);
    add(1, l.key), add(1, l.key_taps);
    add(2, l.value), add(2, l.value_taps);
    return s;
}

void LoraForward::append_stack(const AttentionStack& s, bool aliased, ik::ProjectionArgs& args) const {
    if (aliased) { // one vector: the whole stack, contiguous from its first present input
        const int first = s.rows[0] != 0 ? 0 : (s.rows[1] != 0 ? 1 : 2);
        append_segment(args, s.a[first], s.total(), first, stack_rows(0));
        return;
    }
    std::int32_t before = 0;
    for (int input = 0; input < 3; ++input) {
        if (s.rows[input] == 0) { continue; }
        append_segment(args, s.a[input], s.rows[input], input, stack_rows(before));
        before += s.rows[input];
    }
}

bool LoraForward::fold_attention_inputs(std::uint32_t layer, ik::ProjectionArgs& k2, bool prompt) const {
    const LoraLayerTerms<LoraTerm>& l = p_.layers[layer];
    // A decode call's K2 splits K by its total rows: A's rows would take other runs inside it than alone.
    if (!prompt || !l.attention_input() || !takes_bf16_rows(k2)) { return false; }
    const AttentionStack s = attention_stack(l);
    const bool aliased     = k2.x[0] == k2.x[1] && k2.x[1] == k2.x[2];
    const int segments     = aliased ? 1 : (s.rows[0] != 0) + (s.rows[1] != 0) + (s.rows[2] != 0);
    if (k2.count + segments > ik::kMaxProjectionSegments) { return false; }
    append_stack(s, aliased, k2);
    return true;
}

bool LoraForward::fold_output(std::uint32_t layer, ik::ProjectionArgs& wo) const {
    const LoraTerm& t = p_.layers[layer].output;
    if (!t.present() || !takes_bf16_rows(wo) || wo.count >= ik::kMaxProjectionSegments) { return false; }
    append_segment(wo, t.a, t.rank, 0, slot(0));
    return true;
}

bool LoraForward::fold_gate_down(std::uint32_t layer, std::int32_t first_kind, std::int32_t kinds,
                                 ik::ProjectionArgs& down) const {
    const auto& gates = p_.layers[layer].gates;
    bool any          = false;
    int adapted       = 0;
    for (std::int32_t i = 0; i < kinds; ++i) {
        const bool present = gates[first_kind + i].down.present();
        any                = any || present;
        adapted += present ? 1 : 0;
    }
    if (!any || !takes_bf16_rows(down) || down.count + adapted > ik::kMaxProjectionSegments) { return false; }
    for (std::int32_t i = 0; i < kinds; ++i) {
        const LoraTerm& t = gates[first_kind + i].down;
        if (t.present()) { append_segment(down, t.a, t.rank, i, slot(i)); }
    }
    return true;
}

void LoraForward::attention_inputs(std::uint32_t layer, std::int32_t columns, const std::uint16_t* xq,
                                   const std::uint16_t* xk, const std::uint16_t* xv, const AttentionOutputs& out,
                                   bool prompt, bool folded, cudaStream_t stream) const {
    const LoraLayerTerms<LoraTerm>& l = p_.layers[layer];
    if (!l.attention_input()) { return; }
    const AttentionStack s = attention_stack(l);
    const bool aliased     = xq == xk && xk == xv;
    if (!folded) {
        ik::ProjectionArgs a{};
        a.columns = columns;
        a.k       = hidden_;
        a.x[0] = xq, a.x[1] = xk, a.x[2] = xv;
        append_stack(s, aliased, a);
        shrink(a, prompt, stream);
    }
    ik::LoraTermArgs term{};
    term.columns = columns;
    term.rank    = s.rank;
    // U of each input's stack: one block of [T][total] rows, or a block of its own rows per input.
    std::array<const std::uint16_t*, 3> base{};
    std::array<std::int32_t, 3> stride{};
    std::int32_t before = 0;
    for (int input = 0; input < 3; ++input) {
        base[input]   = aliased ? stack_rows(0) + before : stack_rows(before);
        stride[input] = aliased ? s.total() : s.rows[input];
        before += s.rows[input];
    }
    std::array<std::int32_t, 3> at{};
    const auto add = [&](const LoraTerm& t, int input, std::uint16_t* y) {
        if (!t.present()) { return; }
        if (y == nullptr) { throw std::logic_error("inkling2 lora: an adapted projection's output is missing"); }
        term.targets[term.count++] = target_of(t, base[input] + at[input], stride[input], y, t.rows);
        at[input] += s.rank;
    };
    // The stack's order (LoraBinding::fast_layout).
    add(l.query, 0, out.query);
    add(l.relative, 0, out.relative);
    add(l.gate, 0, out.gate);
    add(l.sink, 0, out.sink);
    add(l.key, 1, out.key);
    add(l.key_taps, 1, out.key_taps);
    add(l.value, 2, out.value);
    add(l.value_taps, 2, out.value_taps);
    ik::lora_term_add(term, stream);
}

void LoraForward::attention_output(std::uint32_t layer, std::int32_t columns, const std::uint16_t* attention,
                                   std::uint16_t* out, bool prompt, bool folded, cudaStream_t stream) const {
    const LoraTerm& t = p_.layers[layer].output;
    if (!t.present()) { return; }
    if (!folded) {
        ik::ProjectionArgs a{};
        a.columns = columns;
        a.k       = query_;
        a.x[0]    = attention;
        append_segment(a, t.a, t.rank, 0, slot(0));
        shrink(a, prompt, stream);
    }
    ik::LoraTermArgs term{};
    term.columns    = columns;
    term.rank       = t.rank;
    term.targets[0] = target_of(t, slot(0), t.rank, out, t.rows);
    term.count      = 1;
    ik::lora_term_add(term, stream);
}

void LoraForward::gate_down(std::uint32_t layer, std::int32_t first_kind, std::int32_t kinds, std::int32_t columns,
                            const std::uint16_t* gate_input, std::uint16_t* z, bool prompt, bool folded,
                            cudaStream_t stream) const {
    const auto& gates = p_.layers[layer].gates;
    ik::ProjectionArgs a{};
    a.columns = columns;
    a.k       = 2 * hidden_;
    ik::LoraTermArgs term{};
    term.columns = columns;
    for (std::int32_t i = 0; i < kinds; ++i) {
        const LoraTerm& t = gates[first_kind + i].down;
        if (!t.present()) { continue; }
        const std::size_t at = static_cast<std::size_t>(i) * columns;
        a.x[i] = gate_input + at * 2 * hidden_;
        append_segment(a, t.a, t.rank, i, slot(i));
        term.rank                  = t.rank;
        term.targets[term.count++] = target_of(t, slot(i), t.rank, z + at * gate_rank_, gate_rank_);
    }
    if (term.count == 0) { return; }
    if (!folded) { shrink(a, prompt, stream); }
    ik::lora_term_add(term, stream);
}

void LoraForward::gate_outputs(std::uint32_t layer, std::int32_t first_kind, std::int32_t kinds, std::int32_t columns,
                               const std::uint16_t* z, float* fg, bool prompt, cudaStream_t stream) const {
    const auto& gates = p_.layers[layer].gates;
    ik::ProjectionArgs a{};
    a.columns = columns;
    a.k       = gate_rank_;
    ik::LoraTermArgs term{};
    term.columns     = columns;
    term.fp32_out    = true;
    std::int32_t next = 0;
    const auto add = [&](const LoraTerm& t, int input, float* y) {
        if (!t.present()) { return; }
        std::uint16_t* u = slot(next++);
        append_segment(a, t.a, t.rank, input, u);
        term.rank                  = t.rank;
        term.targets[term.count++] = target_of(t, u, t.rank, y, 2 * hidden_);
    };
    for (std::int32_t i = 0; i < kinds; ++i) {
        const std::size_t at = static_cast<std::size_t>(i) * columns;
        a.x[i]               = z + at * gate_rank_;
        float* forget        = fg + at * 2 * hidden_;
        const LoraGateTerms<LoraTerm>& g = gates[first_kind + i];
        add(g.forget, i, forget);
        add(g.message, i, forget + hidden_);
    }
    if (term.count == 0) { return; }
    shrink(a, prompt, stream); // the base launch is FP32, U BF16: never folded
    ik::lora_term_add(term, stream);
}

} // namespace infernix::models::inkling2
