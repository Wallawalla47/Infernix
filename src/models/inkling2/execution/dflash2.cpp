#include "models/inkling2/execution/dflash2.h"

#include "core/device.h"
#include "core/layout.h"
#include "infernix/ops/attn_input_proj.h"
#include "infernix/ops/candidate_selector.h"
#include "infernix/ops/context_kv_materialize.h"
#include "infernix/ops/dynamic_grouped_conv.h"
#include "infernix/ops/inkling2_projections.h"
#include "infernix/ops/inkling2_visit.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/linear_swiglu.h"
#include "infernix/ops/linear_topk.h"
#include "infernix/ops/silu_mul.h"
#include "infernix/ops/prepare_masked_block.h"
#include "infernix/ops/rmsnorm.h"
#include "infernix/ops/rmsnorm_pack_tail.h"
#include "infernix/ops/rmsnorm_rope.h"
#include "infernix/ops/sliding_window_attention.h"
#include "infernix/ops/speculative_round.h"
#include "models/inkling2/program/test_taps.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace infernix::models::inkling2 {
namespace {

namespace ik = ops::inkling2;

constexpr std::int32_t kCandidates = 16; // the selector's lattice: 16 candidates a position
constexpr std::size_t kSlack       = 256; // an arena allocation's alignment

std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }
std::size_t align(std::size_t v) { return (v + 255) / 256 * 256; }

// The drafter's device regions in order: the rings, the io words, the head's 8-bit inputs, the commit's gathered
// features and the workspace.
struct Regions {
    std::size_t ring = 0, words = 0, x8 = 0, gathered = 0, workspace = 0;
    [[nodiscard]] std::size_t total() const noexcept { return ring + words + x8 + gathered + workspace; }
};

// The io words' layout (I32 offsets, each array on a 256-byte boundary, as the fused Ops' operands need): the round
// words (anchors, frontiers, valid columns, slots: P words each, P = the lanes rounded up to 64) and the selector
// configs [lanes] (one upload a round: `round` words), the commit's counts, firsts and slots (P each), ones (P), ids
// and positions [8][lanes], drafts [7][lanes], candidates and q [16][7][lanes], confidence [7][lanes], the iota
// [window + 1], the gathered positions [8][lanes].
struct IoLayout {
    std::size_t P = 0;
    std::size_t configs = 0, round = 0, commit = 0, ones = 0;
    std::size_t ids = 0, positions = 0, drafts = 0, candidates = 0, q = 0, confidence = 0, iota = 0, gathered = 0;
    std::size_t words = 0;
    IoLayout(std::int32_t lanes, std::uint32_t window) {
        const auto L  = static_cast<std::size_t>(lanes);
        const auto up = [](std::size_t n) { return (n + 63) / 64 * 64; };
        P             = up(L);
        std::size_t at = 4 * P;
        const auto take = [&](std::size_t n) {
            const std::size_t begin = at;
            at += up(n);
            return begin;
        };
        configs    = take((L * sizeof(ops::SamplingConfig) + 3) / 4);
        round      = at;
        commit     = take(3 * P);
        ones       = take(P);
        ids        = take(DFlash2Drafter::kWidth * L);
        positions  = take(DFlash2Drafter::kWidth * L);
        drafts     = take(DFlash2Drafter::kDrafts * L);
        candidates = take(kCandidates * DFlash2Drafter::kDrafts * L);
        q          = take(kCandidates * DFlash2Drafter::kDrafts * L);
        confidence = take(DFlash2Drafter::kDrafts * L);
        iota       = take(window + 1);
        gathered   = take(DFlash2Drafter::kWidth * L);
        words      = at;
    }
    [[nodiscard]] std::size_t bytes() const { return words * 4; }
};

// The workspace of a context call over `cells` = width x rows columns and of a draft of `lanes` rows, from the
// configuration alone (the projections are the companion recipe's Q8, which the binder checks; the selector's and
// the convolutions' kernel projections BF16).
std::size_t workspace_bytes(const TextConfig& text, const DFlash2Config& d, std::int32_t lanes, std::uint32_t max_context,
                            std::int32_t head_rows) {
    constexpr auto kQ8 = QType::Q8_G32_FP16;
    constexpr auto kA16 = ops::LinearPolicy::A16Only;
    const std::size_t H = text.hidden_size, I = d.intermediate, D = d.head_dim, F = d.feature_width(text.hidden_size);
    const std::size_t q = d.query_width(), kv = d.key_width(), G = H / d.conv_group;
    const auto W = static_cast<std::size_t>(DFlash2Drafter::kWidth), K = static_cast<std::size_t>(DFlash2Drafter::kDrafts);
    // Context: one lane's widest call (the window), a verify call of every lane (8 columns each) and a plain round.
    std::size_t context = 0;
    const std::array<std::pair<std::int32_t, std::int32_t>, 3> calls{
        {{dim(d.window), 1}, {DFlash2Drafter::kWidth, lanes}, {1, lanes}}};
    for (const auto& [columns, rows] : calls) {
        const std::size_t cells = static_cast<std::size_t>(columns) * static_cast<std::size_t>(rows);
        context = std::max(context, 2 * (H * cells * 2 + kSlack) +
                                        ops::linear_workspace_capacity_bytes(kQ8, dim(H), dim(F), kA16, 1, dim(cells)) +
                                        ops::context_kv_materialize_workspace_capacity_bytes(
                                            static_cast<std::int32_t>(d.layers), rows, 1, columns));
    }
    // The draft at every row count: the residual, one branch's buffers and scratch, then the tail.
    std::size_t propose = 0;
    for (std::int32_t b = 1; b <= lanes; ++b) {
        const std::size_t B = static_cast<std::size_t>(b), cells = W * B, masks = K * B;
        const std::int32_t w = DFlash2Drafter::kWidth;
        const std::size_t branch  = (H * cells * 2 + kSlack) + (G * 2 * cells * 2 + kSlack);
        const std::size_t prepare = ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(dim(H), w, w, b, b);
        const std::size_t window  = ops::sliding_window_attention_workspace_capacity_bytes(
            {dim(D), dim(d.heads), dim(d.kv_heads)}, d.window, {0, max_context}, w, w, b);
        const std::size_t add_attention =
            ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(dim(H), dim(q), w, w, b, b);
        // The MLP down Q8 or NVFP4 (the plan sees only the configuration): the larger.
        const std::size_t add_mlp =
            std::max(ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(dim(H), dim(I), w, w, b, b),
                     ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(dim(H), dim(I), w, w, b, b,
                                                                                   QType::NVFP4));
        // The MLP's gate/up: Q8's fused SwiGLU, or an NVFP4 parent's linear into a [2I] matrix then SwiGLU (recipe
        // inkling2_mini_dflash2_nvfp4_mlp): sized for the larger, as the plan sees only the configuration.
        const std::size_t swiglu = std::max(
            ops::linear_swiglu_workspace_capacity_bytes(kQ8, dim(2 * I), dim(H), kA16, 1, dim(cells)),
            (2 * I * cells * 2 + kSlack) +
                ops::linear_workspace_capacity_bytes(QType::NVFP4, dim(2 * I), dim(H), kA16, 1, dim(cells)));
        const std::size_t attention = branch + (q + 2 * kv) * cells * 2 + 3 * kSlack + (q * cells * 2 + kSlack) +
                                      std::max(prepare, window) + add_attention + 2 * kSlack;
        const std::size_t mlp = branch + (I * cells * 2 + kSlack) + std::max(prepare, swiglu) + add_mlp + 2 * kSlack;
        const std::size_t head =
            static_cast<std::size_t>(head_rows) * masks * 4 + kSlack +
            ops::logits_topk_workspace_capacity_bytes(head_rows, 1, dim(masks)) + kSlack;
        const std::size_t selector = ops::candidate_selector_path_workspace_capacity_bytes(1, dim(K), 1, b);
        const std::size_t tail = (H * masks * 2 + kSlack) + (kCandidates * masks * 4 + kSlack) +
                                 (d.selector_rank * masks * 2 + kSlack) + std::max(head, selector);
        propose = std::max(propose, (H * cells * 2 + kSlack) + std::max({attention, mlp, tail}));
    }
    return std::max(context, propose) + (1ULL << 20); // alignment slack
}

Regions regions(const Config& config, std::int32_t lanes, std::uint32_t max_context, std::int32_t head_rows,
                CyclicKVCacheLayout* ring_layout = nullptr) {
    const DFlash2Config& d = *config.dflash2;
    Regions out;
    LayoutBuilder ring;
    const CyclicKVCacheLayout layout =
        plan_cyclic_kv_cache(ring, d.layers, d.window, dim(d.kv_heads), dim(d.head_dim), lanes);
    out.ring = align(ring.finish(256));
    if (ring_layout != nullptr) { *ring_layout = layout; }
    out.words = align(IoLayout(lanes, d.window).bytes());
    out.x8    = align(static_cast<std::size_t>(lanes) * DFlash2Drafter::kDrafts *
                      ik::projection_a8_column_words(dim(config.text.hidden_size)) * 4);
    out.gathered  = align(static_cast<std::size_t>(d.feature_width(config.text.hidden_size)) * DFlash2Drafter::kWidth *
                          static_cast<std::size_t>(lanes) * 2);
    out.workspace = align(workspace_bytes(config.text, d, lanes, max_context, head_rows));
    return out;
}

std::int32_t shortlist_rows(const Model& model) {
    const std::optional<DraftHead>& head = model.draft_head();
    return head ? head->rows : static_cast<std::int32_t>(model.config().text.vocab_size);
}

// The norm and dynamic-conv prepare in an arena of its own carved from `work` (none when no route needs one).
void prepare(const Tensor& residual, const Tensor& norm, float eps, const Tensor& base, const Weight& projection,
             Tensor& prepared, Tensor& delta, WorkspaceArena& work, cudaStream_t s) {
    const std::size_t bytes = ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
        residual.ne[0], residual.ne[1], residual.ne[1], residual.ne[2], residual.ne[2]);
    if (bytes == 0) {
        ops::rmsnorm_dynamic_grouped_conv_prepare(residual, norm, eps, base, projection, prepared, delta, work, s);
        return;
    }
    WorkspaceArena scratch(work.alloc_bytes(bytes));
    ops::rmsnorm_dynamic_grouped_conv_prepare(residual, norm, eps, base, projection, prepared, delta, scratch, s);
}

// The fused Ops take Q8_G32_FP16 RowSplit projections (recipe inkling2_mini_dflash2's).
void require_q8(const Weight& weight, const char* name) {
    if (weight.qtype != QType::Q8_G32_FP16 || weight.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument(std::string("inkling2 dflash2: ") + name +
                                    " must be q8_g32_fp16 (tools.convert recipe inkling2_mini_dflash2)");
    }
}

// The MLP's projections: Q8 as above, or both NVFP4 (block_scale_k16_m128x4; recipe inkling2_mini_dflash2_nvfp4_mlp).
void require_mlp(const Weight& gate_up, const Weight& down) {
    const bool nvfp4 = gate_up.qtype == QType::NVFP4 && down.qtype == QType::NVFP4 &&
                       gate_up.layout == QuantLayout::BlockScaleK16M128x4 &&
                       down.layout == QuantLayout::BlockScaleK16M128x4;
    if (!nvfp4) {
        require_q8(gate_up, "the MLP gate/up projections");
        require_q8(down, "the MLP down projection");
    }
}

} // namespace

std::size_t DFlash2Drafter::device_bytes(const Config& config, std::int32_t lanes, std::uint32_t max_context,
                                         bool shortlist) {
    const std::int32_t rows = shortlist ? 65536 : static_cast<std::int32_t>(config.text.vocab_size);
    return regions(config, lanes, max_context, rows).total();
}

DFlash2Drafter::DFlash2Drafter(const Parameters& parameters, const Forward& forward, std::int32_t lanes,
                               std::uint32_t max_context)
    : p_(parameters), d_(parameters.dflash2.value()), config_(*parameters.model.config().dflash2), forward_(forward),
      lanes_(lanes), max_context_(max_context), head_rows_(shortlist_rows(parameters.model)),
      pinned_(4 * (IoLayout(lanes, 0).round + 3 * IoLayout(lanes, 0).P + 2 * kDrafts * static_cast<std::size_t>(lanes))) {
    const TextConfig& text = parameters.model.config().text;
    if (lanes < 1 || lanes > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("inkling2 dflash2: 1-8 lanes");
    }
    if (d_.layers.size() != config_.layers || forward.buffers().features == nullptr) {
        throw std::invalid_argument("inkling2 dflash2: the drafter's layers and the forward's taps");
    }
    require_q8(d_.feature_projection, "the feature projection");
    for (const DFlash2LayerOperands& l : d_.layers) {
        require_q8(l.query_key_value, "the query/key/value projections");
        require_q8(l.context_key, "the context key projection");
        require_q8(l.context_value, "the context value projection");
        require_q8(l.output, "the attention output projection");
        require_mlp(l.gate_up, l.down);
    }
    if (shortlist_rows(parameters.model) != (p_.model.draft_head() ? 65536 : static_cast<std::int32_t>(text.vocab_size))) {
        throw std::logic_error("inkling2 dflash2: the shortlist's rows differ from the planned head's");
    }
    CyclicKVCacheLayout layout;
    const Regions r = regions(parameters.model.config(), lanes, max_context, head_rows_, &layout);
    memory_         = DeviceBuffer(r.total());
    memory_.fill(0); // finite rings for every slot a block may read before its context lands
    auto* base = static_cast<std::uint8_t*>(memory_.p);
    ring_      = std::make_unique<CyclicKVCache>(DeviceSpan{base, r.ring}, layout);
    for (std::uint32_t l = 0; l < ring_->layer_count(); ++l) { layers_.push_back(ring_->layer_view(l)); }
    const IoLayout io(lanes, config_.window);
    const auto L    = static_cast<std::size_t>(lanes);
    auto* words     = reinterpret_cast<std::int32_t*>(base + r.ring);
    anchors_        = words;
    frontiers_      = words + io.P;
    valid_          = words + 2 * io.P;
    slots_          = words + 3 * io.P;
    configs_        = reinterpret_cast<ops::SamplingConfig*>(words + io.configs);
    commit_         = words + io.commit;
    ones_           = words + io.ones;
    ids_            = words + io.ids;
    positions_      = words + io.positions;
    drafts_         = words + io.drafts;
    candidates_     = words + io.candidates;
    q_              = reinterpret_cast<float*>(words + io.q);
    confidence_     = reinterpret_cast<float*>(words + io.confidence);
    iota_           = words + io.iota;
    gathered_positions_ = words + io.gathered;
    head_x8_       = reinterpret_cast<std::int32_t*>(base + r.ring + r.words);
    gathered_       = reinterpret_cast<std::uint16_t*>(base + r.ring + r.words + r.x8);
    work_ = std::make_unique<WorkspaceArena>(DeviceSpan{base + r.ring + r.words + r.x8 + r.gathered, r.workspace});
    {
        std::vector<std::int32_t> ones(L, 1), iota(config_.window + 1);
        for (std::size_t i = 0; i < iota.size(); ++i) { iota[i] = static_cast<std::int32_t>(i); }
        CUDA_CHECK(cudaMemcpy(ones_, ones.data(), ones.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(iota_, iota.data(), iota.size() * 4, cudaMemcpyHostToDevice));
    }
    // Pinned: the round words (4 P and the configs, the device order), the commit's words (3 P), the drafts and
    // confidence. The configs start greedy (temperature 0); stage() sets each round's.
    auto* host       = static_cast<std::int32_t*>(pinned_.data());
    host_words_      = host;
    host_configs_    = reinterpret_cast<ops::SamplingConfig*>(host + io.configs);
    host_commit_     = host + io.round;
    host_drafts_     = host_commit_ + 3 * io.P;
    host_confidence_ = reinterpret_cast<float*>(host_drafts_ + kDrafts * L);
    std::fill_n(host_words_, io.configs, 0);
    std::uninitialized_fill_n(host_configs_, L, ops::SamplingConfig{.temperature = 0.0F});
    std::fill_n(host_commit_, 3 * io.P, 0);
}

DFlash2Drafter::~DFlash2Drafter() = default;

std::vector<DSparkStatePart> DFlash2Drafter::snapshot_parts() const {
    std::vector<DSparkStatePart> out;
    for (const CyclicKVCacheLayerView& layer : layers_) {
        const std::int64_t words = static_cast<std::int64_t>(layer.head_dim) * layer.padded_capacity * layer.num_kv_heads;
        out.push_back({static_cast<std::uint16_t*>(layer.k.data), words});
        out.push_back({static_cast<std::uint16_t*>(layer.v.data), words});
    }
    return out;
}

void DFlash2Drafter::materialize(const std::uint16_t* source, std::int32_t width, std::int32_t rows,
                                 const std::int32_t* positions, const std::int32_t* counts, const std::int32_t* slots,
                                 std::uint32_t min_count, std::uint32_t max_count, cudaStream_t s) {
    const std::int32_t H = dim(p_.model.config().text.hidden_size), F = dim(config_.feature_width(H));
    const std::int32_t cells = width * rows;
    if (width < 1 || rows < 1 || rows > lanes_) { throw std::invalid_argument("inkling2 dflash2: a context call's shape"); }
    work_->reset();
    const Tensor features(const_cast<std::uint16_t*>(source), DType::BF16, {F, cells});
    Tensor projected = work_->alloc(DType::BF16, {H, cells});
    ops::linear(features, d_.feature_projection, projected, ops::LinearPolicy::A16Only, *work_, s);
    Tensor context = work_->alloc(DType::BF16, {H, width, rows});
    Tensor flat    = context.view({H, cells});
    ops::rmsnorm(projected, d_.context_norm, config_.rms_norm_eps, false, flat, s);
    std::array<ops::ContextKVMaterializeLayerView, ops::kContextKVMaterializeMaxLayers> layers;
    for (std::size_t l = 0; l < config_.layers; ++l) {
        const DFlash2LayerOperands& layer = d_.layers[l];
        layers[l] = {layer.context_key, layer.context_value, layer.key_norm, layers_[l]};
    }
    ops::context_kv_materialize(context, Tensor(const_cast<std::int32_t*>(positions), DType::I32, {width, rows}),
                                Tensor(const_cast<std::int32_t*>(counts), DType::I32, {rows}),
                                Tensor(const_cast<std::int32_t*>(slots), DType::I32, {rows}),
                                std::span(layers.data(), config_.layers), config_.rope_theta, {min_count, max_count},
                                *work_, s);
}

void DFlash2Drafter::context(std::int32_t first, std::int32_t count, cudaStream_t s) {
    if (count <= 0) { return; }
    const std::int32_t n      = std::min(count, dim(config_.window));
    const std::int32_t column = first + count - n;
    const ForwardBuffers& fb  = forward_.buffers();
    if (first < 0 || first + count > forward_.columns()) {
        throw std::invalid_argument("inkling2 dflash2: context columns outside the forward's last call");
    }
    const std::size_t F = config_.feature_width(p_.model.config().text.hidden_size);
    materialize(fb.features + static_cast<std::size_t>(column) * F, n, 1, fb.position + column, iota_ + n,
                fb.lane + column, static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(n), s);
    if (const testing::ContextObserver observer = testing::context_observer()) {
        observer(fb.features + static_cast<std::size_t>(column) * F, fb.position + column, n, s);
    }
}

void DFlash2Drafter::context(const std::uint16_t* features, const std::int32_t* positions, const std::int32_t* lanes,
                             std::int32_t count, cudaStream_t s) {
    if (count <= 0) { return; }
    const std::int32_t n      = std::min(count, dim(config_.window));
    const std::size_t skipped = static_cast<std::size_t>(count - n);
    const std::size_t F       = config_.feature_width(p_.model.config().text.hidden_size);
    materialize(features + skipped * F, n, 1, positions + skipped, iota_ + n, lanes + skipped,
                static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(n), s);
}

void DFlash2Drafter::context_columns(std::int32_t rows, cudaStream_t s) {
    const ForwardBuffers& fb = forward_.buffers();
    if (rows > forward_.columns()) { throw std::invalid_argument("inkling2 dflash2: a plain round's context columns"); }
    materialize(fb.features, 1, rows, fb.position, ones_, fb.lane, 1, 1, s);
}

void DFlash2Drafter::context_rows(std::span<const std::int32_t> counts, std::span<const std::int32_t> firsts,
                                  std::span<const std::int32_t> lanes, cudaStream_t s) {
    stage_commit(counts, firsts, lanes);
    commit_rows(static_cast<std::int32_t>(counts.size()), s);
}

void DFlash2Drafter::stage_commit(std::span<const std::int32_t> counts, std::span<const std::int32_t> firsts,
                                  std::span<const std::int32_t> lanes) {
    const auto rows = static_cast<std::int32_t>(counts.size());
    if (rows < 1 || rows > lanes_ || firsts.size() != counts.size() || lanes.size() != counts.size()) {
        throw std::invalid_argument("inkling2 dflash2: a commit's context of 1 .. lanes rows");
    }
    for (std::size_t r = 0; r < counts.size(); ++r) {
        if (counts[r] < 0 || counts[r] > kWidth || firsts[r] < 0 || firsts[r] + counts[r] > forward_.columns()) {
            throw std::invalid_argument("inkling2 dflash2: a commit's context columns outside the forward's last call");
        }
    }
    const std::size_t P = IoLayout(lanes_, 0).P;
    std::copy(counts.begin(), counts.end(), host_commit_);
    std::copy(firsts.begin(), firsts.end(), host_commit_ + P);
    std::copy(lanes.begin(), lanes.end(), host_commit_ + 2 * P);
}

void DFlash2Drafter::commit_rows(std::int32_t rows, cudaStream_t s) {
    if (rows < 1 || rows > lanes_) { throw std::invalid_argument("inkling2 dflash2: a commit's context of 1 .. lanes rows"); }
    const std::size_t P = IoLayout(lanes_, 0).P;
    CUDA_CHECK(cudaMemcpyAsync(commit_, host_commit_, 3 * P * 4, cudaMemcpyHostToDevice, s));
    const ForwardBuffers& fb = forward_.buffers();
    const auto F             = static_cast<std::int32_t>(config_.feature_width(p_.model.config().text.hidden_size));
    ik::dflash2_gather_columns(fb.features, F, fb.position, commit_, static_cast<std::int32_t>(P), rows, kWidth, gathered_,
                               gathered_positions_, s);
    materialize(gathered_, kWidth, rows, gathered_positions_, commit_, commit_ + 2 * P, 0,
                static_cast<std::uint32_t>(kWidth), s);
}

void DFlash2Drafter::stage(std::span<const ik::DSparkRound> rounds, std::span<const ops::SamplingConfig> draws) noexcept {
    const std::size_t P = IoLayout(lanes_, 0).P;
    for (std::size_t b = 0; b < rounds.size() && b < static_cast<std::size_t>(lanes_); ++b) {
        host_words_[b]         = rounds[b].anchor;
        host_words_[P + b]     = rounds[b].first;
        host_words_[2 * P + b] = kWidth;
        host_words_[3 * P + b] = rounds[b].lane;
        // A whole-vocabulary row's drafts coupled with its draws; every other row's greedy (an uncoupled sampled draft
        // would only lower its acceptance).
        ops::SamplingConfig draw;
        draw.temperature = 0.0F;
        if (b < draws.size() && draws[b].temperature > 0.0F && draws[b].full_vocabulary) {
            draw.temperature     = draws[b].temperature;
            draw.seed            = draws[b].seed;
            draw.full_vocabulary = true;
        }
        host_configs_[b] = draw;
    }
}

void DFlash2Drafter::head_candidates(const Tensor& hidden, Tensor& ids, Tensor& scores, cudaStream_t s) {
    const std::optional<DraftHead>& shortlist = p_.model.draft_head(); // --lm-head-draft
    const std::int32_t count                  = hidden.ne[1];
    auto scope                                = work_->scope();
    Tensor logits = work_->alloc(DType::FP32, {head_rows_, count});
    // The head's raw products (no softcap: the selector's own comes after its correction), on 8-bit inputs where the
    // head is 8-bit (drafts only).
    forward_.project_head(static_cast<const std::uint16_t*>(hidden.data), count, static_cast<float*>(logits.data), s,
                          head_x8_, shortlist ? &shortlist->head : nullptr);
    ops::logits_topk(logits, head_rows_, ids, scores, *work_, s);
    if (shortlist) {
        Tensor flat = ids.view({kCandidates * count});
        ops::proposal_remap_token_ids(flat, shortlist->token_ids, head_rows_, s);
    }
}

void DFlash2Drafter::draft(cudaStream_t s, std::int32_t rows) {
    if (rows < 1 || rows > lanes_) { throw std::invalid_argument("inkling2 dflash2: a round drafts 1 .. lanes rows"); }
    const TextConfig& text = p_.model.config().text;
    const std::int32_t H = dim(text.hidden_size), W = kWidth, B = rows, K = kDrafts, columns = W * B;
    const std::int32_t D = dim(config_.head_dim), Hq = dim(config_.heads), Hk = dim(config_.kv_heads);
    const std::int32_t I = dim(config_.intermediate), G = H / dim(config_.conv_group);
    // The round words (anchors, frontiers, valid columns and slots of every lane) and the selector configs from pinned
    // memory.
    CUDA_CHECK(cudaMemcpyAsync(anchors_, host_words_, 4 * IoLayout(lanes_, 0).round, cudaMemcpyHostToDevice, s));
    work_->reset();
    const Tensor anchors(anchors_, DType::I32, {B}), frontiers(frontiers_, DType::I32, {B});
    const Tensor valid(valid_, DType::I32, {B}), slots(slots_, DType::I32, {B});
    Tensor ids(ids_, DType::I32, {W, B}), positions(positions_, DType::I32, {W, B});
    ops::prepare_masked_block(anchors, frontiers, valid, dim(config_.mask_token_id), ids, positions, s);
    // The block's input: the target's normed embeddings of the anchor and the mask token.
    Tensor residual = work_->alloc(DType::BF16, {H, W, B});
    ik::embed(ids_, columns, p_.embedding, p_.embedding_norm, static_cast<std::uint16_t*>(residual.data), s);
    const ops::SlidingWindowAttentionExecutionEnvelope envelope{0, max_context_};
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));
    for (std::size_t l = 0; l < d_.layers.size(); ++l) {
        const DFlash2LayerOperands& layer = d_.layers[l];
        {
            // Attention: prepare (norm, dynamic conv), Q/K/V, Q/K norm and RoPE, attention over the ring and the
            // block, then the output projection, conv finish and residual add.
            auto scope      = work_->scope();
            Tensor prepared = work_->alloc(DType::BF16, {H, W, B});
            Tensor delta    = work_->alloc(DType::BF16, {G, 2, W, B});
            prepare(residual, layer.input_norm, config_.rms_norm_eps, layer.attention_base, layer.attention_projection,
                    prepared, delta, *work_, s);
            Tensor q = work_->alloc(DType::BF16, {D, Hq, W, B});
            Tensor k = work_->alloc(DType::BF16, {D, Hk, W, B});
            Tensor v = work_->alloc(DType::BF16, {D, Hk, W, B});
            {
                Tensor qf = q.view({D * Hq, columns}), kf = k.view({D * Hk, columns}), vf = v.view({D * Hk, columns});
                ops::attn_input_proj(prepared.view({H, columns}), layer.query_key_value, qf, kf, vf, s);
            }
            ops::rmsnorm_rope(positions, layer.query_norm, layer.key_norm, d_.rope, q, k, s);
            Tensor attention = work_->alloc(DType::BF16, {D, Hq, W, B});
            ops::sliding_window_attention(q, k, v, positions, valid, slots, {D, Hq, Hk}, config_.window,
                                          ops::SlidingWindowMask::Symmetric, scale, layers_[l], envelope, *work_,
                                          attention, s);
            WorkspaceArena scratch(work_->alloc_bytes(
                ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(H, D * Hq, W, W, B, B)));
            ops::linear_dynamic_grouped_conv_add(attention.view({D * Hq, W, B}), layer.output, layer.attention_base,
                                                 delta, residual, scratch, s);
        }
        {
            // MLP: prepare, SwiGLU(gate, up), then down, conv finish and residual add.
            auto scope      = work_->scope();
            Tensor prepared = work_->alloc(DType::BF16, {H, W, B});
            Tensor delta    = work_->alloc(DType::BF16, {G, 2, W, B});
            prepare(residual, layer.post_attention_norm, config_.rms_norm_eps, layer.mlp_base, layer.mlp_projection,
                    prepared, delta, *work_, s);
            Tensor product = work_->alloc(DType::BF16, {I, W, B});
            {
                Tensor flat = product.view({I, columns});
                if (layer.gate_up.qtype == QType::NVFP4) {
                    // The NVFP4 linear's A16 route over the stacked parent, then SwiGLU of its gate rows [0, I) and up
                    // rows [I, 2I).
                    auto inner     = work_->scope();
                    Tensor gate_up = work_->alloc(DType::BF16, {2 * I, columns});
                    ops::linear(prepared.view({H, columns}), layer.gate_up, gate_up, ops::LinearPolicy::A16Only, *work_,
                                s);
                    ops::silu_mul(gate_up.slice(0, 0, I), gate_up.slice(0, I, I), flat, s);
                } else {
                    ops::linear_swiglu(prepared.view({H, columns}), layer.gate_up, flat, ops::LinearPolicy::A16Only,
                                       *work_, s);
                }
            }
            WorkspaceArena scratch(
                work_->alloc_bytes(ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(H, I, W, W, B, B,
                                                                                                  layer.down.qtype)));
            ops::linear_dynamic_grouped_conv_add(product, layer.down, layer.mlp_base, delta, residual, scratch, s);
        }
    }
    // The mask columns' final hidden, their head candidates and the selector's conditional path.
    Tensor hidden = work_->alloc(DType::BF16, {H, K * B});
    ops::rmsnorm_pack_tail(residual, d_.final_norm, hidden, s);
    Tensor candidates(candidates_, DType::I32, {kCandidates, K * B});
    Tensor scores = work_->alloc(DType::FP32, {kCandidates, K * B});
    head_candidates(hidden, candidates, scores, s);
    Tensor projected = work_->alloc(DType::BF16, {dim(config_.selector_rank), K * B});
    {
        ik::ProjectionArgs args{};
        args.columns = K * B;
        args.k       = H;
        args.x[0]    = static_cast<const std::uint16_t*>(hidden.data);
        ik::ProjectionSegment segment;
        segment.weight     = static_cast<const std::uint16_t*>(d_.selector_hidden.qdata);
        segment.rows       = dim(config_.selector_rank);
        segment.input      = 0;
        segment.out        = projected.data;
        args.segments[0]   = segment;
        args.count         = 1;
        ik::projections_dmma(args, s);
    }
    Tensor drafts(drafts_, DType::I32, {K, B});
    Tensor q(q_, DType::FP32, {kCandidates, K, B});
    Tensor confidence(confidence_, DType::FP32, {K, B});
    ops::candidate_selector_path(candidates.view({kCandidates, K, B}), scores.view({kCandidates, K, B}),
                                 projected.view({dim(config_.selector_rank), K, B}), anchors, d_.predecessor_codebook,
                                 d_.successor_codebook, frontiers, configs_, drafts, q, &confidence, *work_, s,
                                 config_.softcap_after_correction ? config_.softcap : 0.0F,
                                 /*couple_draws=*/true); // verification is equality with the target's own draws
}

void DFlash2Drafter::download(cudaStream_t s, std::int32_t rows) {
    const std::size_t bytes = static_cast<std::size_t>(rows) * kDrafts * 4;
    CUDA_CHECK(cudaMemcpyAsync(host_drafts_, drafts_, bytes, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(host_confidence_, confidence_, bytes, cudaMemcpyDeviceToHost, s));
}

} // namespace infernix::models::inkling2
