#include "models/inkling2/execution/dspark.h"

#include "core/device.h"
#include "infernix/ops/inkling2_visit.h"

#include <algorithm>
#include <stdexcept>

namespace infernix::models::inkling2 {
namespace {

namespace ik = ops::inkling2;

constexpr std::int32_t kH      = 2304;
constexpr std::int32_t kQ      = ik::kDSparkHeads * ik::kDSparkHeadDim;   // 3072
constexpr std::int32_t kKv     = ik::kDSparkKvHeads * ik::kDSparkHeadDim; // 1024
constexpr std::int32_t kI      = ik::kDSparkIntermediate;
constexpr std::int32_t kB      = ik::kDSparkBlock;
constexpr std::int32_t kFRing  = 16; // the Markov context ring's rows a lane (> a verify call's columns)
constexpr std::int32_t kSplitKeys = 128;

std::size_t align(std::size_t v) { return (v + 255) / 256 * 256; }

std::int32_t full_splits(std::uint32_t max_context) {
    return static_cast<std::int32_t>((static_cast<std::uint64_t>(max_context) + kB + kSplitKeys - 1) / kSplitKeys);
}
constexpr std::int32_t kSlidingSplits = (ik::kDSparkWindow + kB + kSplitKeys - 1) / kSplitKeys; // 9

// The drafter's device buffers in order, each through add(pointer, bytes).
struct Layout {
    std::uint16_t *cache_k[3] = {}, *cache_v[3] = {};
    std::uint16_t* f_ring = nullptr;
    std::uint16_t *mixed[3] = {}, *ctx_k[3] = {}, *ctx_v[3] = {};
    ik::DSparkRound* round = nullptr;
    std::uint16_t *block_in = nullptr, *residual = nullptr, *normed = nullptr, *q = nullptr, *k = nullptr, *v = nullptr;
    std::uint16_t *qr = nullptr, *attention = nullptr, *a = nullptr, *y = nullptr, *gate = nullptr, *up = nullptr;
    std::uint16_t *h = nullptr, *final_rows = nullptr;
    float *base = nullptr, *attention_scratch = nullptr, *markov_scratch = nullptr, *probabilities = nullptr;
    std::int32_t* tokens  = nullptr;
    float* inv_freq       = nullptr;
    std::int32_t* head_x8 = nullptr; // the block's final rows as 8-bit head inputs (Forward::project_head)
};

template <class Add>
void walk(std::int32_t lanes, std::uint32_t max_context, std::int32_t capacity, std::uint32_t vocab, Layout& b,
          Add&& add) {
    (void)max_context; // the full layer's context is the Program's pages
    const std::size_t L = static_cast<std::size_t>(lanes), T = static_cast<std::size_t>(capacity);
    for (int l = 0; l < 3; ++l) {
        if (l == 1) { continue; } // the full layer's context: the Program's pages (DSparkPagedContext)
        const std::size_t slots = ik::kDSparkRing;
        add(b.cache_k[l], L * ik::kDSparkKvHeads * slots * ik::kDSparkHeadDim * 2);
        add(b.cache_v[l], L * ik::kDSparkKvHeads * slots * ik::kDSparkHeadDim * 2);
    }
    add(b.f_ring, L * kFRing * kH * 2);
    for (int l = 0; l < 3; ++l) {
        add(b.mixed[l], T * kH * 2);
        add(b.ctx_k[l], T * kKv * 2);
        add(b.ctx_v[l], T * kKv * 2);
    }
    add(b.round, sizeof(ik::DSparkRound));
    add(b.block_in, kB * kH * 2), add(b.residual, kB * kH * 2), add(b.normed, kB * kH * 2);
    add(b.q, kB * kQ * 2), add(b.k, kB * kKv * 2), add(b.v, kB * kKv * 2), add(b.qr, kB * kQ * 2);
    add(b.attention, kB * kQ * 2), add(b.a, kB * kH * 2), add(b.y, kB * kH * 2);
    add(b.gate, kB * kI * 2), add(b.up, kB * kI * 2), add(b.h, kB * kI * 2), add(b.final_rows, kB * kH * 2);
    add(b.base, static_cast<std::size_t>(ik::kMarkovSteps) * vocab * 4);
    add(b.attention_scratch, ik::dspark_attention_scratch_bytes(std::max(full_splits(max_context), kSlidingSplits)));
    add(b.markov_scratch, ik::dspark_markov_scratch_bytes(static_cast<std::int32_t>(vocab)));
    add(b.tokens, kB * 4), add(b.probabilities, kB * 4), add(b.inv_freq, ik::kDSparkHeadDim / 2 * 4);
    add(b.head_x8, static_cast<std::size_t>(kB) * ik::projection_a8_column_words(kH) * 4);
}

// A BF16 or 8-bit (Q8_G32_FP16 RowSplit, converted at load: plan_load's drafter_q8_objects) matrix as one segment of
// a fused projection.
void add_segment(ik::ProjectionArgs& args, const Weight& w, std::int32_t input, void* out) {
    const bool bf16 = w.qtype == QType::BF16 && w.layout == QuantLayout::Contiguous;
    const bool q8   = w.qtype == QType::Q8_G32_FP16 && w.layout == QuantLayout::RowSplit;
    if (w.qdata == nullptr || !(bf16 || q8) || args.count >= ik::kMaxProjectionSegments) {
        throw std::invalid_argument("inkling2 dspark: a drafter projection is not a resident BF16 or Q8 matrix");
    }
    ik::ProjectionSegment s;
    if (bf16) {
        s.weight = static_cast<const std::uint16_t*>(w.qdata);
    } else {
        s.codes     = w.qdata;
        s.q8_scales = static_cast<const std::uint16_t*>(w.scales);
        s.q8_stride = static_cast<std::int32_t>(w.padded_shape[1]);
    }
    s.rows   = w.n;
    s.input  = input;
    s.out    = out;
    args.segments[args.count++] = s;
}

const std::uint16_t* bf16_words(const Weight& w) {
    if (w.qdata == nullptr || w.qtype != QType::BF16 || w.layout != QuantLayout::Contiguous) {
        throw std::invalid_argument("inkling2 dspark: a resident BF16 matrix is needed");
    }
    return static_cast<const std::uint16_t*>(w.qdata);
}

} // namespace

std::size_t Drafter::device_bytes(std::int32_t lanes, std::uint32_t max_context, std::int32_t capacity,
                                  std::uint32_t vocab) {
    Layout b;
    std::size_t at = 0;
    walk(lanes, max_context, capacity, vocab, b, [&](auto*&, std::size_t bytes) { at = align(at + bytes); });
    return at;
}

Drafter::Drafter(const Parameters& parameters, const Forward& forward, std::int32_t lanes, std::uint32_t max_context,
                 std::int32_t capacity, const DSparkPagedContext& full)
    : p_(parameters), d_(parameters.dspark.value()), forward_(forward), lanes_(lanes), capacity_(capacity),
      full_splits_(full_splits(max_context)),
      vocab_(parameters.model.config().text.vocab_size),
      pinned_(align(sizeof(ik::DSparkRound)) + align(kB * 4) + align(kB * 4)) {
    if (d_.layers.size() != 3 || !d_.layers[0].sliding || d_.layers[1].sliding || !d_.layers[2].sliding) {
        throw std::invalid_argument("inkling2 dspark: the drafter is 3 layers, sliding, full, sliding");
    }
    if (forward.buffers().features == nullptr) { throw std::invalid_argument("inkling2 dspark: the forward has no taps"); }
    Layout b;
    std::size_t at = 0;
    std::vector<std::pair<void**, std::size_t>> slots;
    walk(lanes, max_context, capacity, vocab_, b, [&](auto*& pointer, std::size_t bytes) {
        slots.emplace_back(reinterpret_cast<void**>(&pointer), at);
        at = align(at + bytes);
    });
    memory_ = DeviceBuffer(at);
    memory_.fill(0); // empty caches; the Markov counters start at zero
    auto* base = static_cast<std::uint8_t*>(memory_.p);
    for (auto& [pointer, offset] : slots) { *pointer = base + offset; }
    for (int l = 0; l < 3; ++l) {
        cache_k_[l]    = b.cache_k[l];
        cache_v_[l]    = b.cache_v[l];
        mixed_[l]      = b.mixed[l];
        ctx_k_[l]      = b.ctx_k[l];
        ctx_v_[l]      = b.ctx_v[l];
        lane_words_[l] = l == 1 ? 0 : static_cast<std::int64_t>(ik::kDSparkKvHeads) * ik::kDSparkRing *
                         ik::kDSparkHeadDim;
    }
    if (full.k == nullptr || full.v == nullptr || full.table == nullptr || full.table_stride <= 0) {
        throw std::invalid_argument("inkling2 dspark: the full layer's context needs the Program's pages");
    }
    full_       = full;
    cache_k_[1] = full.k; // the full layer's cache: the pool's planes, through the lanes' block-table rows
    cache_v_[1] = full.v;
    f_ring_ = b.f_ring, round_ = b.round, block_in_ = b.block_in, residual_ = b.residual, normed_ = b.normed;
    q_ = b.q, k_ = b.k, v_ = b.v, qr_ = b.qr, attention_ = b.attention, a_ = b.a, y_ = b.y;
    gate_ = b.gate, up_ = b.up, h_ = b.h, final_ = b.final_rows, base_ = b.base;
    attention_scratch_ = b.attention_scratch, markov_scratch_ = b.markov_scratch;
    tokens_ = b.tokens, probabilities_ = b.probabilities, inv_freq_ = b.inv_freq, head_x8_ = b.head_x8;
    {
        float table[ik::kDSparkHeadDim / 2];
        ik::dspark_inv_freq(static_cast<double>(parameters.model.config().dspark->rope_theta), table);
        CUDA_CHECK(cudaMemcpy(inv_freq_, table, sizeof(table), cudaMemcpyHostToDevice));
    }
    // The block's rows 1 .. 7: the mask embedding, once (row 0, the anchor's, is rewritten each round).
    for (int r = 1; r < kB; ++r) {
        CUDA_CHECK(cudaMemcpy(block_in_ + static_cast<std::size_t>(r) * kH, d_.mask_embedding, kH * 2,
                              cudaMemcpyDeviceToDevice));
    }
    auto* host          = static_cast<std::uint8_t*>(pinned_.data());
    host_round_         = reinterpret_cast<ik::DSparkRound*>(host);
    host_tokens_        = reinterpret_cast<std::int32_t*>(host + align(sizeof(ik::DSparkRound)));
    host_probabilities_ = reinterpret_cast<float*>(host + align(sizeof(ik::DSparkRound)) + align(kB * 4));
    *host_round_        = {};
}

void Drafter::project(ik::ProjectionArgs& args, cudaStream_t stream) const {
    if (args.columns <= 16) {
        ik::projections_dmma(args, stream);
    } else {
        ik::projections_mma(args, stream);
    }
}

void Drafter::append(std::int32_t first, std::int32_t count, cudaStream_t stream) {
    if (count <= 0) { return; }
    if (first < 0 || first + count > forward_.columns() || count > capacity_) {
        throw std::invalid_argument("inkling2 dspark: context rows outside the forward's last call");
    }
    const ForwardBuffers& fb   = forward_.buffers();
    const std::int32_t* position = fb.position + first;
    const std::int32_t* lane     = fb.lane + first;
    ik::DSparkMixArgs m;
    m.features    = fb.features + static_cast<std::size_t>(first) * ik::kDSparkFeatures * kH;
    m.columns     = count;
    m.fuse_weight = d_.feature_weights;
    m.count       = 4;
    for (int l = 0; l < 3; ++l) {
        m.rows[l]  = l;
        m.norms[l] = d_.layers[l].context_norm;
        m.out[l]   = mixed_[l];
    }
    m.rows[3]        = 3;
    m.norms[3]       = d_.markov_context_norm;
    m.out[3]         = f_ring_;
    m.ring[3]        = kFRing;
    m.positions      = position;
    m.lanes          = lane;
    m.lane_stride[3] = static_cast<std::int64_t>(kFRing) * kH;
    ik::dspark_mix(m, stream);
    // Every layer's context K and V in one launch.
    ik::ProjectionArgs kv{};
    kv.columns = count;
    kv.k       = kH;
    for (int l = 0; l < 3; ++l) {
        kv.x[l] = mixed_[l];
        add_segment(kv, d_.layers[l].context_key, l, ctx_k_[l]);
        add_segment(kv, d_.layers[l].context_value, l, ctx_v_[l]);
    }
    project(kv, stream);
    for (int l = 0; l < 3; ++l) {
        ik::DSparkRotateArgs r;
        r.x             = ctx_k_[l];
        r.x_stride      = kKv;
        r.values        = ctx_v_[l];
        r.values_stride = kKv;
        r.rows          = count;
        r.heads         = ik::kDSparkKvHeads;
        r.norm          = d_.layers[l].context_key_norm;
        r.positions     = position;
        r.lanes         = lane;
        r.cache_k       = cache_k_[l];
        r.cache_v       = cache_v_[l];
        r.lane_stride   = lane_words_[l];
        r.ring          = d_.layers[l].sliding ? ik::kDSparkRing : 0;
        r.table         = full_.table;
        r.table_stride  = full_.table_stride;
        r.inv_freq      = inv_freq_;
        ik::dspark_rotate(r, stream);
    }
}

void Drafter::draft(std::int32_t steps, cudaStream_t stream) {
    if (steps < 1 || steps > ik::kMarkovSteps) { throw std::invalid_argument("inkling2 dspark: 1-7 drafts a round"); }
    CUDA_CHECK(cudaMemcpyAsync(round_, host_round_, sizeof(ik::DSparkRound), cudaMemcpyHostToDevice, stream));
    // The block's input: the anchor's normed target embedding, then the mask rows; the residual stream starts there.
    ik::embed(&round_->anchor, 1, p_.embedding, p_.embedding_norm, block_in_, stream);
    CUDA_CHECK(cudaMemcpyAsync(residual_, block_in_, static_cast<std::size_t>(kB) * kH * 2, cudaMemcpyDeviceToDevice,
                               stream));
    for (int l = 0; l < 3; ++l) {
        const DSparkLayerOperands& o = d_.layers[l];
        ik::DSparkAddNormArgs n;
        n.residual = residual_;
        n.x        = l == 0 ? nullptr : y_;
        n.norm     = o.input_norm;
        n.normed   = normed_;
        n.rows     = kB;
        ik::dspark_add_norm(n, stream);
        ik::ProjectionArgs qkv{};
        qkv.columns = kB;
        qkv.k       = kH;
        qkv.x[0]    = normed_;
        add_segment(qkv, o.query, 0, q_);
        add_segment(qkv, o.key, 0, k_);
        add_segment(qkv, o.value, 0, v_);
        project(qkv, stream);
        ik::DSparkRotateArgs rq;
        rq.x        = q_;
        rq.x_stride = kQ;
        rq.rows     = kB;
        rq.heads    = ik::kDSparkHeads;
        rq.norm     = o.query_norm;
        rq.round    = round_;
        rq.dense    = qr_;
        rq.inv_freq = inv_freq_;
        ik::dspark_rotate(rq, stream);
        // The block's own keys and values into the cache at F .. F + 7 (transient: the next rows overwrite them).
        ik::DSparkRotateArgs rk;
        rk.x             = k_;
        rk.x_stride      = kKv;
        rk.values        = v_;
        rk.values_stride = kKv;
        rk.rows          = kB;
        rk.heads         = ik::kDSparkKvHeads;
        rk.norm          = o.key_norm;
        rk.round         = round_;
        rk.cache_k       = cache_k_[l];
        rk.cache_v       = cache_v_[l];
        rk.lane_stride   = lane_words_[l];
        rk.ring          = o.sliding ? ik::kDSparkRing : 0;
        rk.table         = full_.table;
        rk.table_stride  = full_.table_stride;
        rk.inv_freq      = inv_freq_;
        ik::dspark_rotate(rk, stream);
        ik::DSparkAttentionArgs at;
        at.q           = qr_;
        at.cache_k     = cache_k_[l];
        at.cache_v     = cache_v_[l];
        at.lane_stride = lane_words_[l];
        at.ring        = o.sliding ? ik::kDSparkRing : 0;
        at.table       = full_.table;
        at.table_stride = full_.table_stride;
        at.splits      = o.sliding ? kSlidingSplits : full_splits_;
        at.round       = round_;
        at.scratch     = attention_scratch_;
        at.out         = attention_;
        ik::dspark_attention(at, stream);
        ik::ProjectionArgs wo{};
        wo.columns = kB;
        wo.k       = kQ;
        wo.x[0]    = attention_;
        add_segment(wo, o.output, 0, a_);
        project(wo, stream);
        ik::DSparkAddNormArgs post;
        post.residual = residual_;
        post.x        = a_;
        post.norm     = o.post_attention_norm;
        post.normed   = normed_;
        post.rows     = kB;
        ik::dspark_add_norm(post, stream);
        ik::ProjectionArgs gu{};
        gu.columns = kB;
        gu.k       = kH;
        gu.x[0]    = normed_;
        add_segment(gu, o.gate, 0, gate_);
        add_segment(gu, o.up, 0, up_);
        project(gu, stream);
        ik::dspark_swiglu(gate_, up_, kB, kI, h_, stream);
        ik::ProjectionArgs down{};
        down.columns = kB;
        down.k       = kI;
        down.x[0]    = h_;
        add_segment(down, o.down, 0, y_);
        project(down, stream);
    }
    ik::DSparkAddNormArgs fin;
    fin.residual = residual_;
    fin.x        = y_;
    fin.norm     = d_.final_norm;
    fin.normed   = final_;
    fin.rows     = kB;
    ik::dspark_add_norm(fin, stream);
    // The target head over the block's rows 1 .. steps (the anchor's row predicts nothing); a Q8 head on 8-bit inputs.
    const std::optional<DraftHead>& shortlist = p_.model.draft_head(); // --lm-head-draft
    forward_.project_head(final_ + kH, steps, base_, stream, head_x8_, shortlist ? &shortlist->head : nullptr);
    ik::DSparkMarkovArgs mk;
    mk.projection    = bf16_words(d_.markov_projection);
    mk.bias          = d_.markov_projection_bias;
    mk.block         = final_;
    mk.f_ring        = f_ring_;
    mk.f_ring_rows   = kFRing;
    mk.f_lane_stride = static_cast<std::int64_t>(kFRing) * kH;
    mk.embedding     = d_.markov_embedding;
    if (d_.fold.qtype == QType::Q8_G32_FP16) { // converted at load (drafts only: plan_load's drafter_q8_objects)
        if (d_.fold.qdata == nullptr || d_.fold.scales == nullptr || d_.fold.padded_shape[1] != ik::kMarkovRank) {
            throw std::invalid_argument("inkling2 dspark: the 8-bit Fold needs its codes and scales at rank 256");
        }
        mk.fold_codes  = static_cast<const std::int8_t*>(d_.fold.qdata);
        mk.fold_scales = static_cast<const std::uint16_t*>(d_.fold.scales);
    } else {
        mk.fold = bf16_words(d_.fold);
    }
    mk.fold_bias     = d_.fold_bias;
    mk.base          = base_;
    mk.vocab         = static_cast<std::int32_t>(vocab_);
    if (shortlist) {
        mk.fold        = nullptr;
        mk.fold_codes  = shortlist->fold_codes;
        mk.fold_scales = shortlist->fold_scales;
        mk.fold_bias   = shortlist->fold_bias;
        mk.vocab       = shortlist->rows;
        mk.row_ids     = shortlist->token_ids;
    }
    mk.softcap       = p_.model.config().text.final_logit_softcap;
    mk.round         = round_;
    mk.tokens        = tokens_;
    mk.probabilities = probabilities_;
    mk.scratch       = markov_scratch_;
    ik::dspark_markov(mk, steps, stream);
}

void Drafter::download(std::int32_t steps, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(host_tokens_, tokens_, static_cast<std::size_t>(steps) * 4, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_probabilities_, probabilities_, static_cast<std::size_t>(steps) * 4,
                               cudaMemcpyDeviceToHost, stream));
}

} // namespace infernix::models::inkling2
