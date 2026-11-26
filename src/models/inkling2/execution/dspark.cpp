#include "models/inkling2/execution/dspark.h"

#include "core/device.h"
#include "infernix/ops/inkling2_visit.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infernix::models::inkling2 {
namespace {

namespace ik = ops::inkling2;

constexpr std::int32_t kH      = 2304;
constexpr std::int32_t kQ      = ik::kDSparkHeads * ik::kDSparkHeadDim;   // 3072
constexpr std::int32_t kKv     = ik::kDSparkKvHeads * ik::kDSparkHeadDim; // 1024
constexpr std::int32_t kI      = ik::kDSparkIntermediate;
constexpr std::int32_t kB      = ik::kDSparkBlock;
constexpr std::int32_t kFRing  = 16; // the Markov context ring's rows a lane (> a verify call's columns)

std::size_t align(std::size_t v) { return (v + 255) / 256 * 256; }

std::int32_t full_splits(std::uint32_t max_context) { return ik::dspark_attention_splits(false, max_context); }
const std::int32_t kSlidingSplits = ik::dspark_attention_splits(true, 0); // 9

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

// A row's draft workspace: each buffer and its bytes (a slice a row, in this order).
std::vector<std::pair<void**, std::size_t>> workspace_items(Layout& b, std::uint32_t max_context, std::uint32_t vocab) {
    const auto item = [](auto*& pointer, std::size_t bytes) {
        return std::pair<void**, std::size_t>{reinterpret_cast<void**>(&pointer), bytes};
    };
    return {item(b.block_in, kB * kH * 2),    item(b.residual, kB * kH * 2),   item(b.normed, kB * kH * 2),
            item(b.q, kB * kQ * 2),           item(b.k, kB * kKv * 2),         item(b.v, kB * kKv * 2),
            item(b.qr, kB * kQ * 2),          item(b.attention, kB * kQ * 2),  item(b.a, kB * kH * 2),
            item(b.y, kB * kH * 2),           item(b.gate, kB * kI * 2),       item(b.up, kB * kI * 2),
            item(b.h, kB * kI * 2),           item(b.final_rows, kB * kH * 2),
            item(b.base, static_cast<std::size_t>(ik::kMarkovSteps) * vocab * 4),
            item(b.attention_scratch,
                 ik::dspark_attention_scratch_bytes(std::max(full_splits(max_context), kSlidingSplits))),
            item(b.markov_scratch, ik::dspark_markov_scratch_bytes(static_cast<std::int32_t>(vocab))),
            item(b.head_x8, static_cast<std::size_t>(kB) * ik::projection_a8_column_words(kH) * 4)};
}

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
    add(b.round, L * sizeof(ik::DSparkRound));
    // A row's workspace, a lane's each (Layout's pointers are row 0's; row r's is r slices past: workspace_bytes).
    for (const auto& [pointer, bytes] : workspace_items(b, max_context, vocab)) { add(*pointer, L * align(bytes)); }
    add(b.tokens, L * kB * 4), add(b.probabilities, L * kB * 4), add(b.inv_freq, ik::kDSparkHeadDim / 2 * 4);
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
      pinned_(align(lanes * sizeof(ik::DSparkRound)) + 2 * align(static_cast<std::size_t>(lanes) * kB * 4)) {
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
    f_ring_ = b.f_ring, round_ = b.round;
    tokens_ = b.tokens, probabilities_ = b.probabilities, inv_freq_ = b.inv_freq;
    // Each row's workspace: its slice of every workspace buffer.
    rows_.resize(static_cast<std::size_t>(lanes));
    for (std::int32_t r = 0; r < lanes; ++r) {
        Layout row = b;
        for (const auto& [pointer, bytes] : workspace_items(row, max_context, vocab_)) {
            *pointer = static_cast<std::uint8_t*>(*pointer) + static_cast<std::size_t>(r) * align(bytes);
        }
        rows_[static_cast<std::size_t>(r)] = {row.block_in, row.residual,   row.normed, row.q,    row.k,  row.v,
                                              row.qr,       row.attention, row.a,      row.y,    row.gate, row.up,
                                              row.h,        row.final_rows, row.base,  row.attention_scratch,
                                              row.markov_scratch, row.head_x8};
        // The block's rows 1 .. 7: the mask embedding, once (row 0, the anchor's, is rewritten each round).
        for (int i = 1; i < kB; ++i) {
            CUDA_CHECK(cudaMemcpy(row.block_in + static_cast<std::size_t>(i) * kH, d_.mask_embedding, kH * 2,
                                  cudaMemcpyDeviceToDevice));
        }
    }
    {
        float table[ik::kDSparkHeadDim / 2];
        ik::dspark_inv_freq(static_cast<double>(parameters.model.config().dspark->rope_theta), table);
        CUDA_CHECK(cudaMemcpy(inv_freq_, table, sizeof(table), cudaMemcpyHostToDevice));
    }
    // Rows 1 .. lanes - 1 draft on streams of their own.
    for (std::int32_t r = 1; r < lanes; ++r) {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        streams_.push_back(stream);
        cudaEvent_t fork = nullptr, join = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
        fork_.push_back(fork);
        join_.push_back(join);
    }
    auto* host                = static_cast<std::uint8_t*>(pinned_.data());
    const std::size_t rounds  = align(static_cast<std::size_t>(lanes) * sizeof(ik::DSparkRound));
    const std::size_t drafts  = align(static_cast<std::size_t>(lanes) * kB * 4);
    host_round_               = reinterpret_cast<ik::DSparkRound*>(host);
    host_tokens_              = reinterpret_cast<std::int32_t*>(host + rounds);
    host_probabilities_       = reinterpret_cast<float*>(host + rounds + drafts);
    std::fill_n(host_round_, lanes, ik::DSparkRound{});
}

Drafter::~Drafter() {
    for (cudaStream_t stream : streams_) { (void)cudaStreamDestroy(stream); }
    for (cudaEvent_t event : fork_) { (void)cudaEventDestroy(event); }
    for (cudaEvent_t event : join_) { (void)cudaEventDestroy(event); }
}

void Drafter::stage(std::span<const ik::DSparkRound> rounds) {
    if (rounds.empty() || rounds.size() > static_cast<std::size_t>(lanes_)) {
        throw std::invalid_argument("inkling2 dspark: a round drafts 1 .. lanes rows");
    }
    std::copy(rounds.begin(), rounds.end(), host_round_);
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

void Drafter::draft(std::int32_t steps, cudaStream_t stream, std::int32_t rows) {
    if (steps < 1 || steps > ik::kMarkovSteps) { throw std::invalid_argument("inkling2 dspark: 1-7 drafts a round"); }
    if (rows < 1 || rows > lanes_) { throw std::invalid_argument("inkling2 dspark: a round drafts 1 .. lanes rows"); }
    CUDA_CHECK(cudaMemcpyAsync(round_, host_round_, static_cast<std::size_t>(rows) * sizeof(ik::DSparkRound),
                               cudaMemcpyHostToDevice, stream));
    if (rows == 1) {
        draft_row(0, steps, stream);
        return;
    }
    // Rows 1 .. on their own streams after the round words, row 0 on the caller's, then the caller's waits for all:
    // the rows' launch chains (each mostly small kernels in sequence: the Markov steps above all) overlap (2026-11-26,
    // four rows of the owner's configuration: a round's draft 2.7 -> 2.0 ms at 100K, +1.9 % tok/s, +2.1 % at 2K).
    for (std::int32_t row = 1; row < rows; ++row) {
        const auto r = static_cast<std::size_t>(row - 1);
        CUDA_CHECK(cudaEventRecord(fork_[r], stream));
        CUDA_CHECK(cudaStreamWaitEvent(streams_[r], fork_[r], 0));
        draft_row(row, steps, streams_[r]);
        CUDA_CHECK(cudaEventRecord(join_[r], streams_[r]));
    }
    draft_row(0, steps, stream);
    for (std::int32_t row = 1; row < rows; ++row) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, join_[static_cast<std::size_t>(row - 1)], 0));
    }
}

// One row's block (its round words at round_ + row; its lane's caches; its drafts into the row's slot) on the row's
// workspace.
void Drafter::draft_row(std::int32_t row, std::int32_t steps, cudaStream_t stream) {
    ik::DSparkRound* const round = round_ + row;
    const Workspace& w           = rows_[static_cast<std::size_t>(row)];
    // The block's input: the anchor's normed target embedding, then the mask rows; the residual stream starts there.
    ik::embed(&round->anchor, 1, p_.embedding, p_.embedding_norm, w.block_in, stream);
    CUDA_CHECK(cudaMemcpyAsync(w.residual, w.block_in, static_cast<std::size_t>(kB) * kH * 2, cudaMemcpyDeviceToDevice,
                               stream));
    for (int l = 0; l < 3; ++l) {
        const DSparkLayerOperands& o = d_.layers[l];
        ik::DSparkAddNormArgs n;
        n.residual = w.residual;
        n.x        = l == 0 ? nullptr : w.y;
        n.norm     = o.input_norm;
        n.normed   = w.normed;
        n.rows     = kB;
        ik::dspark_add_norm(n, stream);
        ik::ProjectionArgs qkv{};
        qkv.columns = kB;
        qkv.k       = kH;
        qkv.x[0]    = w.normed;
        add_segment(qkv, o.query, 0, w.q);
        add_segment(qkv, o.key, 0, w.k);
        add_segment(qkv, o.value, 0, w.v);
        project(qkv, stream);
        ik::DSparkRotateArgs rq;
        rq.x        = w.q;
        rq.x_stride = kQ;
        rq.rows     = kB;
        rq.heads    = ik::kDSparkHeads;
        rq.norm     = o.query_norm;
        rq.round    = round;
        rq.dense    = w.qr;
        rq.inv_freq = inv_freq_;
        ik::dspark_rotate(rq, stream);
        // The block's own keys and values into the cache at F .. F + 7 (transient: the next rows overwrite them).
        ik::DSparkRotateArgs rk;
        rk.x             = w.k;
        rk.x_stride      = kKv;
        rk.values        = w.v;
        rk.values_stride = kKv;
        rk.rows          = kB;
        rk.heads         = ik::kDSparkKvHeads;
        rk.norm          = o.key_norm;
        rk.round         = round;
        rk.cache_k       = cache_k_[l];
        rk.cache_v       = cache_v_[l];
        rk.lane_stride   = lane_words_[l];
        rk.ring          = o.sliding ? ik::kDSparkRing : 0;
        rk.table         = full_.table;
        rk.table_stride  = full_.table_stride;
        rk.inv_freq      = inv_freq_;
        ik::dspark_rotate(rk, stream);
        ik::DSparkAttentionArgs at;
        at.q            = w.qr;
        at.cache_k      = cache_k_[l];
        at.cache_v      = cache_v_[l];
        at.lane_stride  = lane_words_[l];
        at.ring         = o.sliding ? ik::kDSparkRing : 0;
        at.table        = full_.table;
        at.table_stride = full_.table_stride;
        at.splits       = o.sliding ? kSlidingSplits : full_splits_;
        at.round        = round;
        at.scratch      = w.attention_scratch;
        at.out          = w.attention;
        ik::dspark_attention(at, stream);
        ik::ProjectionArgs wo{};
        wo.columns = kB;
        wo.k       = kQ;
        wo.x[0]    = w.attention;
        add_segment(wo, o.output, 0, w.a);
        project(wo, stream);
        ik::DSparkAddNormArgs post;
        post.residual = w.residual;
        post.x        = w.a;
        post.norm     = o.post_attention_norm;
        post.normed   = w.normed;
        post.rows     = kB;
        ik::dspark_add_norm(post, stream);
        ik::ProjectionArgs gu{};
        gu.columns = kB;
        gu.k       = kH;
        gu.x[0]    = w.normed;
        add_segment(gu, o.gate, 0, w.gate);
        add_segment(gu, o.up, 0, w.up);
        project(gu, stream);
        ik::dspark_swiglu(w.gate, w.up, kB, kI, w.h, stream);
        ik::ProjectionArgs down{};
        down.columns = kB;
        down.k       = kI;
        down.x[0]    = w.h;
        add_segment(down, o.down, 0, w.y);
        project(down, stream);
    }
    ik::DSparkAddNormArgs fin;
    fin.residual = w.residual;
    fin.x        = w.y;
    fin.norm     = d_.final_norm;
    fin.normed   = w.final_rows;
    fin.rows     = kB;
    ik::dspark_add_norm(fin, stream);
    // The target head over the block's rows 1 .. steps (the anchor's row predicts nothing); a Q8 head on 8-bit inputs.
    const std::optional<DraftHead>& shortlist = p_.model.draft_head(); // --lm-head-draft
    forward_.project_head(w.final_rows + kH, steps, w.base, stream, w.head_x8, shortlist ? &shortlist->head : nullptr);
    ik::DSparkMarkovArgs mk;
    mk.projection    = bf16_words(d_.markov_projection);
    mk.bias          = d_.markov_projection_bias;
    mk.block         = w.final_rows;
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
    mk.base          = w.base;
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
    mk.round         = round;
    mk.tokens        = tokens_ + static_cast<std::size_t>(row) * kB;
    mk.probabilities = probabilities_ + static_cast<std::size_t>(row) * kB;
    mk.scratch       = w.markov_scratch;
    ik::dspark_markov(mk, steps, stream);
}

void Drafter::download(std::int32_t steps, cudaStream_t stream, std::int32_t rows) {
    if (rows == 1) { // one row: its steps only
        CUDA_CHECK(cudaMemcpyAsync(host_tokens_, tokens_, static_cast<std::size_t>(steps) * 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host_probabilities_, probabilities_, static_cast<std::size_t>(steps) * 4,
                                   cudaMemcpyDeviceToHost, stream));
        return;
    }
    // Several rows: their whole slots, in one copy each.
    const std::size_t bytes = static_cast<std::size_t>(rows) * kB * 4;
    CUDA_CHECK(cudaMemcpyAsync(host_tokens_, tokens_, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_probabilities_, probabilities_, bytes, cudaMemcpyDeviceToHost, stream));
}

} // namespace infernix::models::inkling2
