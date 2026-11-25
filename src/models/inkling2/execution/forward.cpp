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
constexpr std::int32_t kD        = ik::kHeadDim;
constexpr std::int64_t ring_words(std::int32_t width) { return std::int64_t{ik::kRingPositions} * width; }
constexpr std::int64_t kWindowWords = std::int64_t{ik::kKvHeads} * ik::kWindowRingPages * ik::kKeyUnit * ik::kHeadDim;

// A visit's conv rings by kind: 0 the K input, 1 the V input, 2 W_o's output, 3 the MLP output.
constexpr std::int64_t ring_width(int kind) { return kind < 2 ? kKv : kH; }
std::int64_t ring_offset(const VisitState& vs, int kind) {
    const std::int64_t offsets[4] = {vs.key_ring, vs.value_ring, vs.attention_ring, vs.mlp_ring};
    return offsets[kind];
}
// A ring's history within a lane's histories: visit-major, its kHistorySlots rows of each kind in order.
constexpr std::int64_t history_offset(std::uint32_t visit, int kind) {
    std::int64_t at = static_cast<std::int64_t>(visit) * (2 * kKv + 2 * kH) * ik::kHistorySlots;
    for (int k = 0; k < kind; ++k) { at += ring_width(k) * ik::kHistorySlots; }
    return at;
}

// The block layout this forward is written for: blocks of 6, the core at 6-17 run 3 times, 24 layers.
void check_schedule(const TextConfig& c) {
    if (c.num_hidden_layers != 24 || c.residual_block != 6 || c.loop.core_start != 6 || c.loop.core_end != 18 ||
        c.loop.passes != 3 || c.visits() != 48) {
        throw std::invalid_argument("inkling2 forward: Inkling 2 Mini's schedule (24 layers, blocks of 6, core 6-17 x 3)");
    }
}

std::size_t align(std::size_t v) { return (v + 255) / 256 * 256; }

// A matrix of 8-bit weights (Q8_G32_FP16 RowSplit, include/infernix/ops/q8_g32.h).
bool q8(const Weight& w) { return w.qtype == QType::Q8_G32_FP16 && w.layout == QuantLayout::RowSplit; }

// A BF16, bf16_p12 or Q8 matrix as one segment of a fused projection launch.
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
    } else if (q8(w)) {
        s.codes     = w.qdata;
        s.q8_scales = static_cast<const std::uint16_t*>(w.scales);
        s.q8_stride = static_cast<std::int32_t>(w.padded_shape[1]);
    } else {
        throw std::invalid_argument("inkling2 forward: the fused projections take BF16, bf16_p12 or Q8 matrices");
    }
    args.segments[args.count++] = s;
}

// The fused projections: a prompt call's on projections_mma, a decode-route call's (decode, verification, forced
// tokens; any width) on projections_dmma (BF16 or Q8 storage both: one route at every width, since a call's width
// depends on the other requests); bf16_p12 segments on the fold (projections).
bool use_mma(bool prompt, const ik::ProjectionArgs& a) {
    return prompt && (a.segments[0].weight != nullptr || a.segments[0].q8_scales != nullptr) && a.k % 64 == 0;
}
bool use_dmma(bool prompt, const ik::ProjectionArgs& a) {
    return !prompt && (a.segments[0].weight != nullptr || a.segments[0].q8_scales != nullptr) && a.k % 32 == 0;
}
void project_fused(bool prompt, const ik::ProjectionArgs& a, cudaStream_t stream) {
    if (a.x8[0] != nullptr) { // 8-bit inputs: projections_mma at every width (one arithmetic, width-invariant)
        ik::projections_mma(a, stream);
    } else if (use_mma(prompt, a)) {
        ik::projections_mma(a, stream);
    } else if (use_dmma(prompt, a)) {
        ik::projections_dmma(a, stream);
    } else {
        ik::projections(a, stream);
    }
}

std::int32_t max_groups(std::int32_t lanes, std::int32_t capacity) {
    return capacity / ik::kMaxAttentionLaneColumns + lanes;
}

// The forward's device buffers in order, each through add(pointer, bytes) (the constructor places them; device_bytes
// counts them).
// a8: the words of a prompt call's 8-bit inputs, a column: K2's three inputs, where W_o's one also fits.
std::size_t a8_column_words() {
    const std::size_t k2 = 3 * static_cast<std::size_t>(ik::projection_a8_column_words(kH));
    return std::max(k2, static_cast<std::size_t>(ik::projection_a8_column_words(kQ)));
}

template <class Add>
void walk_buffers(const TextConfig& c, std::int32_t max_lanes, std::int32_t capacity, bool features,
                  moe::ExpertActivation activation, bool a8, ForwardBuffers& b, std::uint8_t*& moe_base,
                  std::uint8_t*& attention_base, std::uint8_t*& guess_base, Add&& add) {
    const std::size_t T = static_cast<std::size_t>(capacity);
    const std::size_t V = c.vocab_size, rank = c.loop.message_rank, lanes = static_cast<std::size_t>(max_lanes);
    const std::size_t row16         = T * kH * 2;
    const std::size_t history_words = static_cast<std::size_t>(c.visits()) * history_offset(1, 0);
    const std::size_t staging_words =
        static_cast<std::size_t>(ik::kKvHeads) * ik::staging_ring_pages(capacity) * ik::kKeyUnit * kD;
    add(b.tokens, T * 4), add(b.lane, T * 4), add(b.position, T * 4), add(b.first, ik::kMaxAttentionLanes * 4);
    add(b.groups, 3 * static_cast<std::size_t>(max_groups(max_lanes, capacity)) * 4);
    add(b.history, lanes * history_words * 2);
    add(b.staging_k, lanes * staging_words * 2), add(b.staging_v, lanes * staging_words * 2);
    add(b.history_rings, static_cast<std::size_t>(c.visits()) * 4 * 3 * 8);
    add(b.e, row16), add(b.h6, row16), add(b.hp, row16);
    for (auto*& h : b.h12) { add(h, row16); }
    for (auto*& h : b.h18) { add(h, row16); }
    add(b.cand, T * kH * 4);
    for (auto*& x : b.x) { add(x, row16); }
    add(b.q, T * kQ * 2), add(b.r, T * kR * 2), add(b.gate, T * kQ * 2), add(b.k, T * kKv * 2);
    add(b.v, T * kKv * 2), add(b.sink, T * ik::kHeads * 2), add(b.key_taps, T * kTaps * 2);
    add(b.value_taps, T * kTaps * 2), add(b.attention, T * kQ * 2), add(b.o, row16), add(b.xm, row16);
    const bool a16 = activation == moe::ExpertActivation::A16;
    const bool a8x = activation == moe::ExpertActivation::A8;
    add(b.rows, T * std::max({moe::row_bytes(moe::ExpertFormat::Nvfp4FourOverSix, kH),
                              moe::row_bytes(moe::ExpertFormat::Mxfp8, kH), a16 ? moe::kA16XRowBytes : 0,
                              a8x ? moe::kA8XRowBytes : 0}));
    add(b.amax, T * 4), add(b.emax, T * 4), add(b.experts, T * moe::kTopK * kH * 2);
    add(b.mixed, 3 * T * kH * 4), add(b.message, 3 * T * kH * 4), add(b.fg, 3 * T * 2 * kH * 4);
    add(b.gate_input, 3 * T * 2 * kH * 2), add(b.z, 3 * T * rank * 2);
    if (a8) { add(b.x8, T * a8_column_words() * 4); }
    add(b.final_hidden, row16), add(b.logits, std::size_t{kMaxForwardColumns} * V * 4);
    if (features) { add(b.features, 4 * row16); }
    add(moe_base, moe::workspace_bytes(static_cast<std::int32_t>(c.experts), static_cast<std::int32_t>(c.intermediate),
                                       capacity, activation));
    add(attention_base, ik::attention_workspace_bytes(capacity));
    // The expert prefetch's: guessing calls' MLP inputs a visit, guesses, the last verification call's records, the
    // guesses' router workspace.
    const std::size_t visits = c.visits();
    add(b.guess_x, visits * Forward::kGuessColumns * kH * 2), add(b.guess_ids, visits * 16 * 4);
    add(b.expert_record, visits * moe::kTopK * 4);
    add(guess_base, moe::workspace_bytes(static_cast<std::int32_t>(c.experts), static_cast<std::int32_t>(c.intermediate),
                                         Forward::kGuessColumns));
}

} // namespace

std::size_t Forward::device_bytes(const TextConfig& config, std::int32_t max_lanes, std::int32_t capacity, bool features,
                                  moe::ExpertActivation activation, bool a8) {
    ForwardBuffers b;
    std::uint8_t *moe_base = nullptr, *attention_base = nullptr, *guess_base = nullptr;
    std::size_t at = 0;
    walk_buffers(config, max_lanes, capacity, features, activation, a8, b, moe_base, attention_base, guess_base,
                 [&](auto*&, std::size_t bytes) { at = align(at + bytes); });
    return at;
}

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

Forward::Forward(const Parameters& parameters, std::int32_t max_lanes, std::int32_t capacity,
                 moe::ExpertActivation activation, bool a8)
    : p_(parameters), c_(parameters.model.config().text), layout_(state_layout(c_)), max_lanes_(max_lanes),
      capacity_(capacity), activation_(activation), a8_(a8), max_groups_(max_groups(max_lanes, capacity)),
      staging_pages_(ik::staging_ring_pages(capacity)) {
    if (max_lanes <= 0 || max_lanes > ik::kMaxAttentionLanes) {
        throw std::invalid_argument("inkling2 forward: 1-8 lanes");
    }
    if (capacity < kMaxForwardColumns || capacity > moe::kMaxCapacity) {
        throw std::invalid_argument("inkling2 forward: a call capacity of 128-8191 columns");
    }
    history_words_ = static_cast<std::int64_t>(c_.visits()) * history_offset(1, 0);
    std::size_t at = 0;
    std::vector<std::pair<void**, std::size_t>> slots;
    std::uint8_t* moe_base       = nullptr;
    std::uint8_t* attention_base = nullptr;
    std::uint8_t* guess_base     = nullptr;
    walk_buffers(c_, max_lanes, capacity, p_.dspark.has_value(), activation, a8, b_, moe_base, attention_base, guess_base,
                 [&](auto*& pointer, std::size_t bytes) {
        slots.emplace_back(reinterpret_cast<void**>(&pointer), at);
        at = align(at + bytes);
    });
    memory_ = DeviceBuffer(at);
    memory_.fill(0); // the Ops' counters and the dispatch state start at zero once
    auto* base = static_cast<std::uint8_t*>(memory_.p);
    for (auto& [pointer, offset] : slots) { *pointer = base + offset; }
    b_.moe          = moe::carve_workspace(moe_base, static_cast<std::int32_t>(c_.experts),
                                           static_cast<std::int32_t>(c_.intermediate), capacity, activation);
    b_.attention_ws = ik::carve_attention_workspace(attention_base, capacity);
    b_.guess        = moe::carve_workspace(guess_base, static_cast<std::int32_t>(c_.experts),
                                           static_cast<std::int32_t>(c_.intermediate), kGuessColumns);
    side_           = std::make_unique<SideStreams>();
    if (p_.lora && !p_.lora->lossless) { lora_ = std::make_unique<LoraForward>(*p_.lora, c_, capacity); }
    // conv_history's descriptors: every visit's four rings.
    std::vector<std::int64_t> rings;
    for (std::uint32_t v = 0; v < c_.visits(); ++v) {
        for (int kind = 0; kind < 4; ++kind) {
            rings.insert(rings.end(), {ring_offset(layout_.visits[v], kind), history_offset(v, kind), ring_width(kind)});
        }
    }
    CUDA_CHECK(cudaMemcpy(b_.history_rings, rings.data(), rings.size() * 8, cudaMemcpyHostToDevice));
}

Forward::SideStreams::SideStreams() {
    try {
        CUDA_CHECK(cudaStreamCreateWithFlags(&guess, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&prefetch, cudaStreamNonBlocking));
        for (cudaEvent_t* e : {&x, &k2, &guessed, &join_guess, &join_prefetch}) {
            CUDA_CHECK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
        }
    } catch (...) {
        release();
        throw;
    }
}

Forward::SideStreams::~SideStreams() { release(); }

void Forward::SideStreams::release() noexcept {
    for (cudaEvent_t* e : {&x, &k2, &guessed, &join_guess, &join_prefetch}) {
        if (*e != nullptr) { (void)cudaEventDestroy(*e); }
        *e = nullptr;
    }
    for (cudaStream_t* s : {&guess, &prefetch}) {
        if (*s != nullptr) { (void)cudaStreamDestroy(*s); }
        *s = nullptr;
    }
}

std::vector<std::int32_t> Forward::layout_call(const CallPlan& call) {
    const std::int32_t T = call.columns(), B = call.lanes();
    if (T <= 0 || T > capacity_ || B <= 0 || B > max_lanes_ || call.lane.size() != call.tokens.size() ||
        call.position.size() != call.tokens.size()) {
        throw std::invalid_argument("inkling2 forward: a call of 1..capacity columns over the forward's lanes");
    }
    std::vector<CallLane> lanes;
    std::vector<std::int32_t> starts(static_cast<std::size_t>(B), -1);
    for (std::int32_t t = 0; t < T; ++t) {
        const std::int32_t l = call.lane[t];
        if (l < -1 || l >= B) { throw std::invalid_argument("inkling2 forward: a column's lane is outside the call"); }
        if (l < 0) { continue; }
        if (starts[l] < 0) {
            starts[l] = t;
            lanes.push_back({l, 0, call.position[t], call.first[l]});
        } else if (call.lane[t - 1] != l || call.position[t] != call.position[t - 1] + 1) {
            throw std::invalid_argument("inkling2 forward: a lane's columns are consecutive at consecutive positions");
        }
        ++lanes.back().count;
    }
    // Attention's groups: each lane's columns in runs of at most 16.
    std::vector<std::int32_t> groups;
    std::int32_t widest = 1;
    for (const CallLane& l : lanes) {
        if (l.first < 0 || l.position < l.first) {
            throw std::invalid_argument("inkling2 forward: a lane's columns start at or after its first position");
        }
        for (std::int32_t done = 0; done < l.count; done += ik::kMaxAttentionLaneColumns) {
            const std::int32_t count = std::min(ik::kMaxAttentionLaneColumns, l.count - done);
            groups.insert(groups.end(), {starts[l.lane] + done, count, l.lane});
            widest = std::max(widest, count);
        }
    }
    columns_           = T;
    lanes_             = B;
    groups_            = static_cast<std::int32_t>(groups.size() / 3);
    max_group_columns_ = widest;
    wide_conv_ = std::any_of(lanes.begin(), lanes.end(), [](const CallLane& l) { return l.count > ik::kMaxLaneColumns; });
    wide_window_ = std::any_of(lanes.begin(), lanes.end(), [](const CallLane& l) { return l.count > ik::kKeyUnit; });
    prompt_      = call.prompt;
    std::int32_t live = 0;
    one_group_lanes_  = true;
    for (const CallLane& l : lanes) {
        live += l.count;
        one_group_lanes_ = one_group_lanes_ && l.count <= ik::kMaxAttentionLaneColumns;
    }
    padding_ = live < T;
    call_lanes_  = std::move(lanes);
    host_groups_ = groups;
    return groups;
}

void Forward::prepare(const CallPlan& call) {
    const std::vector<std::int32_t> groups = layout_call(call);
    const auto copy = [](void* dst, const std::vector<std::int32_t>& src) {
        CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * 4, cudaMemcpyHostToDevice));
    };
    copy(b_.tokens, call.tokens), copy(b_.lane, call.lane), copy(b_.position, call.position);
    copy(b_.first, call.first), copy(b_.groups, groups);
}

void Forward::upload(const CallPlan& call, std::int32_t* staging, cudaStream_t stream) {
    const std::vector<std::int32_t> groups = layout_call(call);
    std::int32_t* at = staging;
    const auto copy = [&](void* dst, const std::vector<std::int32_t>& src) {
        std::copy(src.begin(), src.end(), at);
        CUDA_CHECK(cudaMemcpyAsync(dst, at, src.size() * 4, cudaMemcpyHostToDevice, stream));
        at += src.size();
    };
    copy(b_.tokens, call.tokens), copy(b_.lane, call.lane), copy(b_.position, call.position);
    copy(b_.first, call.first), copy(b_.groups, groups);
}

ik::ColumnMap Forward::column_map() const { return {b_.lane, b_.position, b_.first, columns_}; }

void Forward::snapshot_history(const StateView& s, cudaStream_t stream) const {
    for (const CallLane& l : call_lanes_) {
        ik::ConvHistoryArgs h;
        h.rings    = b_.history_rings;
        h.count    = static_cast<std::int32_t>(c_.visits()) * 4;
        h.image    = s.images + static_cast<std::int64_t>(l.lane) * s.lane_words;
        h.history  = b_.history + static_cast<std::int64_t>(l.lane) * history_words_;
        h.position = l.position;
        h.first    = l.first;
        ik::conv_history(h, stream);
    }
}

namespace {
ik::KvView staging_view(const ForwardBuffers& b, std::int32_t pages) {
    ik::KvView v;
    v.k           = b.staging_k;
    v.v           = b.staging_v;
    v.ring_pages  = pages;
    v.lane_stride = std::int64_t{ik::kKvHeads} * pages * ik::kKeyUnit * kD;
    return v;
}
} // namespace

void Forward::stage_window(const StateView& s, const VisitState& vs, bool in, cudaStream_t stream) const {
    ik::KvView state;
    state.k           = s.images + vs.window_k;
    state.v           = s.images + vs.window_v;
    state.lane_stride = s.lane_words;
    const ik::KvView staging = staging_view(b_, staging_pages_);
    for (const CallLane& l : call_lanes_) {
        ik::WindowCopyArgs w;
        w.src_lane = w.dst_lane = l.lane;
        if (in) { // the window before the call's first column
            w.src = state, w.dst = staging;
            w.from = std::max(l.first, l.position - (ik::kWindow - 1)), w.to = l.position;
        } else { // the call's own positions the state ring keeps (the window before them came from it unchanged)
            const std::int32_t end = l.position + l.count;
            w.src = staging, w.dst = state;
            w.from = std::max(l.position, end - ik::kWindowRingPages * ik::kKeyUnit), w.to = end;
        }
        ik::window_copy(w, stream);
    }
}

void Forward::tap(std::uint32_t visit, const std::uint16_t* rows, cudaStream_t stream) const {
    const std::vector<std::uint32_t>& targets = p_.model.config().dspark->target_layer_ids;
    for (std::size_t j = 0; j < targets.size(); ++j) {
        if (targets[j] != visit) { continue; }
        // The rows (the partial block sums entering the visit; the final-normed state for visit == visits()) into
        // feature slot j of every column.
        CUDA_CHECK(cudaMemcpy2DAsync(b_.features + j * kH, targets.size() * kH * 2, rows, kH * 2, kH * 2,
                                     static_cast<std::size_t>(columns_), cudaMemcpyDeviceToDevice, stream));
    }
}

void Forward::set_hidden_capture(std::uint16_t* rows, std::int32_t column) {
    if (rows == nullptr || column < 0) { throw std::invalid_argument("inkling2 forward: a hidden-state capture's rows"); }
    hidden_rows_   = rows;
    hidden_column_ = column;
    hidden_counts_.assign(c_.num_hidden_layers, 0);
}

void Forward::capture_hidden(std::uint32_t visit, const std::uint16_t* rows, cudaStream_t stream) {
    const std::uint32_t layer = c_.visit_layer(visit);
    if (c_.in_core(layer) && c_.visit_pass(visit) + 1 != c_.loop.passes) { return; } // the core's last pass only
    CUDA_CHECK(cudaMemcpyAsync(hidden_rows_ + static_cast<std::size_t>(layer) * kH,
                               rows + static_cast<std::size_t>(hidden_column_) * kH, static_cast<std::size_t>(kH) * 2,
                               cudaMemcpyDeviceToDevice, stream));
    ++hidden_counts_[layer];
}

void Forward::embed(cudaStream_t stream) {
    ik::embed(b_.tokens, columns_, p_.embedding, p_.embedding_norm, b_.e, stream);
    mark("embed");
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

ik::Ring Forward::ring(const StateView& s, std::uint32_t visit, int kind) const {
    ik::Ring r{s.images + ring_offset(layout_.visits.at(visit), kind), static_cast<std::int32_t>(ring_width(kind)),
               s.lane_words};
    if (wide_conv_) {
        r.history             = b_.history + history_offset(visit, kind);
        r.history_lane_stride = history_words_;
    }
    return r;
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

void Forward::clear_moe_counters(cudaStream_t stream) const {
    moe::moe_clear_counters(b_.moe, stream);
    moe::moe_clear_counters(b_.guess, stream);
}

const std::uint16_t* Forward::mlp_input(std::uint32_t visit) const {
    return !prompt_ && columns_ <= kGuessColumns ? b_.guess_x + static_cast<std::size_t>(visit) * kGuessColumns * kH : b_.xm;
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
    const ik::ColumnMap map = column_map();
    const std::uint16_t* h  = in.input;
    // The expert prefetch: a guessing call's MLP input lives in its visit's slot (the guess stream's router reads it
    // after the main stream has moved on); a reading call prefetches its visits' records.
    // The read-ahead is a verification call's (one lane's consecutive columns, whose last column predicts the lane's
    // next call); a decode round of 3-8 lanes prefetches nothing (8 lanes: 649.6 -> 646.4 tok/s with the last lane's).
    const bool guessing = !prompt_ && T <= kGuessColumns;
    const bool reading  = !prompt_ && T > kGuessColumns && T <= kReadAheadColumns && call_lanes_.size() == 1;
    auto* xm            = const_cast<std::uint16_t*>(mlp_input(v)); // b_'s own memory
    // The guess's size (2026-11-23, rounds at context 512): one column 4 whole records (8 gate/up parts were 2.0-3.4 %
    // slower a round); two columns 8 gate/up parts (6, or whole records, were not faster on every route).
    const std::int32_t guesses = T == 1 ? kGuessExperts1 : kGuessExperts;
    const std::uint64_t gate_up_bytes =
        moe::gate_up_record_bytes(op.bank.format, kH, static_cast<std::int32_t>(c_.intermediate));
    const std::uint64_t guess_bytes = T == 1 ? op.bank.record_stride : gate_up_bytes;
    if (v == 0 && wide_conv_) { // before the call's first ring write
        snapshot_history(s, stream);
        mark("conv_history");
    }

    // K1 (K1a): the previous visit's tail, the attention mix, the norms or the messages.
    ik::EntryArgs e;
    e.columns = T;
    e.map     = map;
    if (v > 0) {
        const std::uint32_t prev = v - 1;
        e.tail     = {b_.experts, b_.cand, p_.layers[c_.visit_layer(prev)].mlp_conv,
                      ring(s, prev, 3)};
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
    // a8: K2's 8-bit inputs written beside its BF16 inputs (by the entry, or the gates in a message visit).
    const std::int32_t k2_words = ik::projection_a8_column_words(kH);
    const auto k2_x8 = [&](int i) { return b_.x8 + static_cast<std::size_t>(i) * k2_words * T; };
    if (a8_ && !messages) {
        e.x8_stride = k2_words;
        for (int i = 0; i < 3; ++i) { e.x8[i] = e.x[i] != nullptr ? k2_x8(i) : nullptr; }
    }
    // --ablate: the tail's post-MLP conv output is the previous visit's layer's mlp lane.
    if (const ik::ResidualCut cut = v > 0 ? p_.ablate_cut(c_.visit_layer(v - 1), AblateParameters::Lane::Mlp)
                                          : ik::ResidualCut{};
        cut.active()) {
        ik::EntryCutArgs cut_args;
        static_cast<ik::EntryArgs&>(cut_args) = e;
        cut_args.tail_cut                     = cut;
        ik::visit_entry(cut_args, stream);
    } else {
        ik::visit_entry(e, stream);
    }
    mark("visit_entry");
    if (hidden_rows_ != nullptr && v > 0) { capture_hidden(v - 1, in.tail_out, stream); }
    if (b_.features != nullptr) { tap(v, h, stream); } // h: the partial block sum entering the visit
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
        const bool lora_down = lora_ && lora_->fold_gate_down(layer, 0, 3, down);
        project_fused(prompt_, down, stream);
        if (lora_) { lora_->gate_down(layer, 0, 3, T, b_.gate_input, b_.z, prompt_, lora_down, stream); }
        project_fused(prompt_, fg, stream);
        if (lora_) { lora_->gate_outputs(layer, 0, 3, T, b_.z, b_.fg, prompt_, stream); }
        ik::GateExitArgs g;
        g.columns  = T;
        g.kinds    = 3;
        g.mixed    = b_.mixed;
        g.shared_x = !op.full;
        g.message  = b_.message;
        g.fg       = b_.fg;
        for (int k = 0; k < 3; ++k) { g.norms[k] = op.norms[k], g.x[k] = b_.x[k]; }
        if (a8_) {
            g.x8_stride = k2_words;
            for (int k = 0; k < 3; ++k) { g.x8[k] = k2_x8(k); }
        }
        ik::gate_exit(g, stream);
        mark("message_gates");
    }
    const bool three   = op.full || messages;
    const std::uint16_t* xq = b_.x[0];
    const std::uint16_t* xk = three ? b_.x[1] : b_.x[0];
    const std::uint16_t* xv = three ? b_.x[2] : b_.x[0];

    // K2: every projection of the visit's inputs in one launch (project_fused: the call's route's arithmetic).
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
    if (a8_) { // the inputs' codes, written by their producers (aliased inputs share the first one's)
        for (int i = 0; i < 3; ++i) { k2.x8[i] = k2_x8(i); }
        k2.x8_encoded = true;
    }
    const bool lora_k2 = lora_ && lora_->fold_attention_inputs(layer, k2, prompt_); // U = A x inside this launch
    project_fused(prompt_, k2, stream);
    mark("k2_projections");
    // The expert prefetch, forked once K2 has streamed its weights: the visit's guessed experts (the guess the visit
    // before made) or the previous verification call's record, into L2 beside attention, W_o and the glue.
    if ((guessing && v > 0) || reading) {
        CUDA_CHECK(cudaEventRecord(side_->k2, stream));
        CUDA_CHECK(cudaStreamWaitEvent(side_->prefetch, side_->k2, 0));
        if (guessing) {
            CUDA_CHECK(cudaStreamWaitEvent(side_->prefetch, side_->guessed, 0));
            moe::moe_prefetch_l2(b_.guess_ids + static_cast<std::size_t>(v) * 16, guesses, op.bank, guess_bytes,
                                 side_->prefetch);
        } else {
            moe::moe_prefetch_l2(b_.expert_record + static_cast<std::size_t>(v) * moe::kTopK, kReadAheadExperts, op.bank,
                                 gate_up_bytes, side_->prefetch);
        }
        side_->forked_prefetch = true;
    }
    if (lora_) { // --lora: the term on each adapted output of K2, after its own rounding (docs/lora.md)
        lora_->attention_inputs(layer, T, xq, xk, xv,
                                {b_.q, b_.r, b_.gate, b_.sink, b_.k, b_.v, op.full ? b_.key_taps : nullptr,
                                 op.full ? b_.value_taps : nullptr},
                                prompt_, lora_k2, stream);
    }

    // K3: attention.
    ik::AttentionArgs at;
    at.columns          = T;
    at.groups            = groups_;
    at.max_group_columns = max_group_columns_;
    at.map               = map;
    at.groups_columns    = b_.groups;
    at.host_groups       = host_groups_.data();
    at.in               = {b_.q, b_.r, b_.k, b_.v, b_.sink, b_.gate, op.full ? b_.key_taps : nullptr,
                           op.full ? b_.value_taps : nullptr};
    at.layer            = a.layer;
    at.key_ring         = ring(s, v, 0);
    at.value_ring       = ring(s, v, 1);
    const bool staged   = !op.full && wide_window_;
    if (op.full) {
        at.kv.k            = s.pool_k.at(vs.full_slot);
        at.kv.v            = s.pool_v.at(vs.full_slot);
        at.kv.table        = s.table;
        at.kv.table_stride = s.table_stride;
        at.kv.pool_pages   = s.pool_pages;
        at.kv.codec        = s.codec;
        if (s.codec != ik::KvCodec::Bf16) {
            at.kv.k_side     = s.side_k.at(vs.full_slot);
            at.kv.v_side     = s.side_v.at(vs.full_slot);
            at.kv.unpackable = s.unpackable;
        }
    } else if (staged) {
        // A lane appending more than a page: its window runs over the staging ring, filled from the state's ring and
        // returned to it after.
        stage_window(s, vs, true, stream);
        mark("window_copy");
        at.kv = staging_view(b_, staging_pages_);
    } else {
        at.kv.k           = s.images + vs.window_k;
        at.kv.v           = s.images + vs.window_v;
        at.kv.lane_stride = s.lane_words;
    }
    at.ws  = b_.attention_ws;
    at.out = b_.attention;
    if (a8_ && !prompt_) { // the decode route's attention writes W_o's 8-bit inputs (over K2's, read by now)
        at.out8        = b_.x8;
        at.out8_stride = ik::projection_a8_column_words(kQ);
    }
    // The decode route's prologue inside attention's launch, for one lane of at most 4 columns (2026-11-21, against the
    // prologue kernel: the decode round at 4K -0.44 % at one column, -0.17 % at two, -0.09 % at four, +0.25 % at eight;
    // infernix_bench C = 4 -0.8 %, C = 8 -1.3 %). Padding columns need the prologue's zeros, and a lane in two groups
    // its keys before the second group reads them.
    at.fused = !padding_ && one_group_lanes_ && call_lanes_.size() == 1 && T <= 4;
    // W_o in Q8 storage loads all its weights before its PDL wait: attention lets its CTAs launch at once, beside it
    // (2026-11-23, rounds at context 512 / 4096: --q8-weights -0.3 to -1.6 %, --a8-activations -0.03 to -1.4 % at one to
    // eight columns; a BF16 W_o (the default route) was +0.07 / +0.25 % slower at one column, so it waits as before).
    at.trigger_at_entry = q8(a.output);
    prompt_ ? ik::attention_prompt(at, stream) : ik::attention(at, stream);
    mark("attention");
    if (staged) {
        stage_window(s, vs, false, stream);
        mark("window_copy");
    }

    // K4: W_o (project_fused: the call's route's arithmetic); K5 (K5a): its post-attention
    // conv and residual, the MLP mix, norm and the experts' rows.
    ik::ProjectionArgs wo{};
    wo.columns = T;
    wo.k       = kQ;
    wo.x[0]    = b_.attention;
    add_segment(wo, a.output, 0, b_.o);
    if (a8_) {
        wo.x8[0]      = b_.x8;
        wo.x8_encoded = !prompt_; // attention() wrote them; a prompt call encodes them here (the same bits)
    }
    const bool lora_wo = lora_ && lora_->fold_output(layer, wo);
    project_fused(prompt_, wo, stream);
    mark("w_o");
    if (lora_) { lora_->attention_output(layer, T, b_.attention, b_.o, prompt_, lora_wo, stream); }
    // The experts' x: the checkpoint's rows, or under A16 an NVFP4 layer's limb rows and an MXFP8 layer's BF16 x alone,
    // or under A8 an NVFP4 layer's A8 rows (written by the entry from the BF16 x; MXFP8 layers keep their rows).
    const bool a16   = activation_ == moe::ExpertActivation::A16;
    const bool nvfp4 = op.format == moe::ExpertFormat::Nvfp4FourOverSix;
    const bool a8x   = activation_ == moe::ExpertActivation::A8 && nvfp4;
    const ik::ExpertRows rows = a8x  ? ik::ExpertRows{op.format, b_.rows, nullptr, nullptr, true}
                              : !a16 ? ik::ExpertRows{op.format, b_.rows, b_.amax}
                              : nvfp4 ? ik::ExpertRows{op.format, b_.rows, nullptr, b_.emax}
                                      : ik::ExpertRows{op.format, nullptr, nullptr, nullptr};
    moe::Activations experts_x{b_.rows, b_.amax, T};
    if (a16) {
        experts_x.activation = moe::ExpertActivation::A16;
        experts_x.rows       = nvfp4 ? b_.rows : nullptr;
        experts_x.emax       = nvfp4 ? b_.emax : nullptr;
        experts_x.x16        = xm;
    }
    if (a8x) {
        experts_x.activation = moe::ExpertActivation::A8;
        experts_x.amax       = nullptr;
    }
    ik::MlpEntryArgs m;
    m.columns  = T;
    m.map      = map;
    m.o        = b_.o;
    m.conv     = op.attention_conv;
    m.ring     = ring(s, v, 2);
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
        m.x    = xm;
        m.rows = rows;
    }
    // --ablate: the post-attention conv's output is this layer's attn lane.
    if (const ik::ResidualCut cut = p_.ablate_cut(layer, AblateParameters::Lane::Attn); cut.active()) {
        ik::MlpEntryCutArgs cut_args;
        static_cast<ik::MlpEntryArgs&>(cut_args) = m;
        cut_args.cut                             = cut;
        ik::mlp_entry(cut_args, stream);
    } else {
        ik::mlp_entry(m, stream);
    }
    mark("mlp_entry");
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
        const bool lora_down = lora_ && lora_->fold_gate_down(layer, 3, 1, down);
        project_fused(prompt_, down, stream);
        if (lora_) { lora_->gate_down(layer, 3, 1, T, b_.gate_input, b_.z, prompt_, lora_down, stream); }
        project_fused(prompt_, fg, stream);
        if (lora_) { lora_->gate_outputs(layer, 3, 1, T, b_.z, b_.fg, prompt_, stream); }
        ik::GateExitArgs g;
        g.columns  = T;
        g.kinds    = 1;
        g.mixed    = b_.mixed;
        g.message  = b_.message;
        g.fg       = b_.fg;
        g.norms[0] = op.mlp_norm;
        g.x[0]     = xm;
        g.rows     = rows;
        ik::gate_exit(g, stream);
        mark("message_gates");
    }
    if (guessing && v + 1 < c_.visits()) { // the expert prefetch: the next visit's router on this MLP input
        const LayerOperands& next = p_.layers.at(c_.visit_layer(v + 1));
        CUDA_CHECK(cudaEventRecord(side_->x, stream));
        CUDA_CHECK(cudaStreamWaitEvent(side_->guess, side_->x, 0));
        moe::moe_route(xm, T, next.router, next.router_bias, next.bank.experts, kH, c_.route_scale, b_.guess, side_->guess);
        // Each column's first guesses / T slots (descending key: its likeliest experts).
        const std::size_t take = static_cast<std::size_t>(guesses / T) * 4;
        CUDA_CHECK(cudaMemcpy2DAsync(b_.guess_ids + static_cast<std::size_t>(v + 1) * 16, take, b_.guess.routed,
                                     static_cast<std::size_t>(moe::kTopK) * 4, take, static_cast<std::size_t>(T),
                                     cudaMemcpyDeviceToDevice, side_->guess));
        CUDA_CHECK(cudaEventRecord(side_->guessed, side_->guess));
        side_->forked_guess = true;
    }

    // K6-K8: route and the experts (their outputs combine in the next visit's entry). A prompt call: the router's logits
    // on tensor cores (FP32), the selection by column, the experts' grouped block-scaled GEMMs.
    if (prompt_) {
        ik::ProjectionArgs router{};
        router.columns  = T;
        router.k        = kH;
        router.x[0]     = xm;
        router.fp32_out = true;
        ik::ProjectionSegment s;
        s.weight = op.router;
        s.rows   = op.bank.experts;
        s.out    = b_.moe.logits;
        router.segments[router.count++] = s;
        ik::projections_mma(router, stream);
        moe::moe_select(b_.moe.logits, T, op.router_bias, op.bank.experts, c_.route_scale, b_.moe, stream);
        mark("router");
        // A16: the narrow kernels at the prompt's width too (exact sums; no A16 grouped GEMM). A8: the grouped 8-bit
        // GEMM, the narrow route's bits.
        a16 ? moe::moe_experts(experts_x, moe::routed_routing(b_.moe), op.bank, b_.moe, b_.experts, stream)
            : moe::moe_experts_prompt(experts_x, moe::routed_routing(b_.moe), op.bank, b_.moe, b_.experts, stream);
    } else if (T == 1 && moe::moe_single_supported(op.bank.format, experts_x.activation)) {
        // One column: the router and the experts in one launch (the same bits).
        moe::moe_single(xm, experts_x, op.router, op.router_bias, c_.route_scale, op.bank, b_.moe, b_.experts, stream);
    } else {
        moe::moe_route(xm, T, op.router, op.router_bias, op.bank.experts, kH, c_.route_scale, b_.moe, stream,
                       reading ? b_.expert_record + static_cast<std::size_t>(v) * moe::kTopK : nullptr);
        mark("router");
        moe::moe_experts(experts_x, moe::routed_routing(b_.moe), op.bank, b_.moe, b_.experts, stream);
    }
    mark("experts");
}

void Forward::finish(const StateView& s, const FinalInputs& in, cudaStream_t stream, std::int32_t head_first,
                     std::int32_t head_columns) {
    const std::uint32_t last = c_.visits() - 1;
    ik::FinalArgs f;
    f.columns  = columns_;
    f.map      = column_map();
    f.tail     = {b_.experts, b_.cand, p_.layers[c_.visit_layer(last)].mlp_conv,
                  ring(s, last, 3)};
    f.tail_out = in.tail_out;
    f.blocks   = in.blocks;
    f.query    = p_.output_query;
    f.norm     = p_.final_norm;
    f.out      = b_.final_hidden;
    if (const ik::ResidualCut cut = p_.ablate_cut(c_.visit_layer(last), AblateParameters::Lane::Mlp); cut.active()) {
        ik::final_mix(f, cut, stream);
    } else {
        ik::final_mix(f, stream);
    }
    mark("final_mix");
    if (hidden_rows_ != nullptr) { capture_hidden(last, in.tail_out, stream); }
    if (b_.features != nullptr) { tap(c_.visits(), b_.final_hidden, stream); }
    // The head over the requested columns only (a prefill chunk needs none, or its last).
    head(head_first, head_columns < 0 ? columns_ - head_first : head_columns, stream);
    { // the expert prefetch: each side stream the call forked joins the call's stream once
        if (side_->forked_guess) {
            CUDA_CHECK(cudaEventRecord(side_->join_guess, side_->guess));
            CUDA_CHECK(cudaStreamWaitEvent(stream, side_->join_guess, 0));
        }
        if (side_->forked_prefetch) {
            CUDA_CHECK(cudaEventRecord(side_->join_prefetch, side_->prefetch));
            CUDA_CHECK(cudaStreamWaitEvent(stream, side_->join_prefetch, 0));
        }
        side_->forked_guess = side_->forked_prefetch = false;
    }
}

void Forward::head(std::int32_t first, std::int32_t count, cudaStream_t stream) {
    if (first < 0 || count < 0 || first + count > columns_ || count > kMaxForwardColumns) {
        throw std::invalid_argument("inkling2 forward: the head takes at most 128 of the call's columns at a time");
    }
    if (count == 0) { return; }
    // The rows go to the buffer's start (row r holds column first + r): a logits row is 8 bytes off 16-byte alignment,
    // so an odd column's own row could not take the head's output.
    const std::uint16_t* x = b_.final_hidden + static_cast<std::size_t>(first) * kH;
    if (q8(p_.head)) { // ops::linear has no route for Q8 at the head's shape
        ik::ProjectionArgs h{};
        h.columns  = count;
        h.k        = kH;
        h.x[0]     = x;
        h.fp32_out = true;
        add_segment(h, p_.head, 0, b_.logits);
        if (a8_) { // 8-bit inputs (the head's rows are Q8 under a8)
            h.x8[0] = b_.x8;
            ik::projections_mma(h, stream);
        } else {
            ik::projections(h, stream);
        }
    } else {
        project(x, p_.head, b_.logits, true, stream, 0, count);
    }
    ik::softcap(b_.logits, static_cast<std::int64_t>(count) * c_.vocab_size, c_.final_logit_softcap, stream);
    mark("head");
}

void Forward::project_head(const std::uint16_t* x, std::int32_t count, float* out, cudaStream_t stream,
                           std::int32_t* x8, const Weight* head) const {
    if (count <= 0 || count > kMaxForwardColumns) { throw std::invalid_argument("inkling2 forward: 1-128 head rows"); }
    if (head != nullptr && !q8(*head)) { throw std::invalid_argument("inkling2 forward: a shortlist head is 8-bit"); }
    const Weight& matrix = head != nullptr ? *head : p_.head;
    if (q8(matrix)) {
        ik::ProjectionArgs h{};
        h.columns  = count;
        h.k        = kH;
        h.x[0]     = x;
        h.fp32_out = true;
        add_segment(h, matrix, 0, out);
        if (x8 != nullptr) {
            h.x8[0] = x8; // encoded here
            ik::projections_mma(h, stream);
        } else {
            ik::projections(h, stream);
        }
    } else {
        project(x, p_.head, out, true, stream, 0, count);
    }
}

} // namespace infernix::models::inkling2
