#include "models/inkling2/program/program.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "infernix/ops/argmax.h"
#include "infernix/ops/inkling2_attention.h"
#include "infernix/ops/sampling.h"
#include "infernix/ops/token_constraint.h"
#include "models/inkling2/execution/dspark.h"
#include "models/inkling2/execution/forward.h"
#include "models/inkling2/execution/lossless.h"
#include "models/inkling2/execution/vision.h"
#include "models/inkling2/program/prefix_cache.h"
#include "models/inkling2/program/test_taps.h"
#include "runtime/expert_offload/ram_monitor.h"
#include "runtime/prefix_cache/block_hash.h"
#include "runtime/prefix_cache/prefix_holds.h"
#include "runtime/prefix_cache/source_policy.h"
#include "runtime/prefix_cache/tap_planner.h"
#include "runtime/speculative/draft_length.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace infernix::models::inkling2 {
namespace detail {

using Clock = std::chrono::steady_clock;

// A request's token constraint (TokenConstraint; /v1/decide's generated primitives), compiled at admission: step i
// permits set steps[i] of `sets`, or forces token -2 - steps[i]; `observe` and `capture_logits` as the request's.
struct ConstraintPlan {
    std::vector<std::int32_t> steps;
    std::vector<std::vector<TokenId>> sets;
    std::vector<std::vector<TokenId>> observe;
    bool capture_logits = false;
    std::uint64_t serial = 0;
};

struct BasePlanImpl {
    std::shared_ptr<const ConstraintPlan> constraint; // null: unconstrained
    std::vector<TokenId> tokens;
    std::shared_ptr<const std::vector<PromptImage>> images; // null: a text prompt
    runtime::RequestPlanSummary summary;
    runtime::ResolvedExecutionOptions options;
    // The prefix cache's keys (frontend PreparedPromptData) and whether the request reuses and publishes.
    bool reuse = false;
    std::vector<std::uint64_t> block_hashes, block_extras;
    std::vector<runtime::prefix_cache::TapHint> tap_hints;
};

// A snapshot source quoted for an admission (Program::hybrid_sources): its snapshot, frontier and anchor path.
struct QuoteImpl {
    runtime::prefix_cache::SnapshotRef snapshot;
    std::uint32_t frontier = 0;
    std::vector<runtime::prefix_cache::NodeRef> path; // the full blocks below the frontier, root first
};

struct ContractAccess {
    static SequenceHandle handle(const void* owner, std::uint32_t lane, std::uint64_t epoch) {
        SequenceHandle s;
        s.owner_ = owner;
        s.lane_  = lane;
        s.epoch_ = epoch;
        return s;
    }
    static const void* owner(const SequenceHandle& s) noexcept { return s.owner_; }
    static std::uint32_t lane(const SequenceHandle& s) noexcept { return s.lane_; }
    static std::uint64_t epoch(const SequenceHandle& s) noexcept { return s.epoch_; }

    // `counts` empty: one token a row (stride 1); else row r's counts[r] tokens at tokens[r * stride ..].
    static PendingBatch batch(const void* owner, std::uint64_t transaction, std::span<const SequenceHandle> rows,
                              std::span<const TokenId> tokens, runtime::ExecutionTiming timing,
                              std::span<const runtime::PendingRowFailure> failures,
                              std::span<const std::int32_t> counts = {}, std::uint32_t stride = 1) {
        PendingBatch b;
        b.owner_       = owner;
        b.transaction_ = transaction;
        b.row_count_   = rows.size();
        for (std::size_t i = 0; i < rows.size(); ++i) {
            b.rows_[i]         = rows[i];
            b.row_failures_[i] = failures[i];
        }
        b.tokens_     = tokens;
        b.row_counts_ = counts;
        b.row_stride_ = stride;
        b.timing_     = timing;
        return b;
    }
    static const void* owner(const PendingBatch& b) noexcept { return b.owner_; }
    static std::uint64_t transaction(const PendingBatch& b) noexcept { return b.transaction_; }
    static std::span<const SequenceHandle> rows(const PendingBatch& b) noexcept { return {b.rows_.data(), b.row_count_}; }
    static void consume(PendingBatch& b) noexcept {
        b.owner_     = nullptr;
        b.row_count_ = 0;
        b.tokens_    = {};
    }

    static ResumeState resume(std::unique_ptr<ResumeStateImpl> impl, std::uint32_t frontier) noexcept {
        return ResumeState(std::move(impl), frontier);
    }
    static ResumeStateImpl* resume_impl(const ResumeState* r) noexcept { return r != nullptr ? r->impl_.get() : nullptr; }
};

namespace {

std::uint64_t elapsed_ns(Clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

// The forward's call capacity: the prefill chunk (no wider than the context: a call never holds more prompt
// positions), and at least the decode and verify calls' width. The lossless route runs no call through it.
std::int32_t forward_capacity(const ProgramOptions& o) {
    if (o.lossless) { return kMaxForwardColumns; }
    return std::max(kMaxForwardColumns, static_cast<std::int32_t>(std::min(o.prefill_chunk, o.max_context)));
}

// LosslessForward's calls: a scored window (the context), or a generation call of at most a prefill chunk whose keys
// reach the context (the cache form).
std::int32_t lossless_columns(const ProgramOptions& o) {
    return static_cast<std::int32_t>(o.causal_scoring ? o.max_context : std::min(o.prefill_chunk, o.max_context));
}
std::int32_t lossless_keys(const ProgramOptions& o) {
    return o.causal_scoring ? 0 : static_cast<std::int32_t>(o.max_context);
}

ops::w4a4_moe::ExpertActivation expert_activation(const ProgramOptions& o) {
    return o.a8           ? ops::w4a4_moe::ExpertActivation::A8
           : o.expert_a16 ? ops::w4a4_moe::ExpertActivation::A16
                          : ops::w4a4_moe::ExpertActivation::Checkpoint;
}

// The planes of one full visit's page: K and V (16-bit words, BF16 storage), and a quantized form's side records.
std::size_t kv_planes_per_visit(ops::inkling2::KvCodec codec) { return codec == ops::inkling2::KvCodec::Bf16 ? 2 : 4; }

// The full visits' page geometry (core/paged_kv_cache.h): per full visit its K and V planes, then a quantized form's
// K and V side planes, each page-major [pages][kKvHeads][64][a row] (attention_common.cuh kv_row): a row of 128 BF16
// (K) or FP16 (V; BF16 on the lossless route, `lossless`) words, or a quantized form's codes and side record
// (inkling2_attention.h KvCodec); then, with DSpark, the drafter's full layer's K and V (BF16, inkling2_dspark.h), so a
// page carries the drafter's context with the target's.
KVPageGeometry kv_page_geometry(std::int32_t full_visits, ops::inkling2::KvCodec codec, bool dspark, bool lossless) {
    const ops::inkling2::KvRowBytes b = ops::inkling2::kv_row_bytes(codec);
    const auto plane                  = [](DType dtype, std::int32_t row) {
        return KVPlaneGeometry{.dtype = dtype, .leading_extent = row, .head_extent = ops::inkling2::kKvHeads};
    };
    KVPageGeometry g;
    g.device_plane_order = PagedKVPlaneOrder::PageMajor;
    for (std::int32_t s = 0; s < full_visits; ++s) {
        if (codec == ops::inkling2::KvCodec::Bf16) {
            g.planes.push_back(plane(DType::BF16, ops::inkling2::kHeadDim));
            g.planes.push_back(plane(lossless ? DType::BF16 : DType::FP16, ops::inkling2::kHeadDim));
        } else {
            g.planes.push_back(plane(DType::U8, b.key_code));
            g.planes.push_back(plane(DType::U8, b.value_code));
            g.planes.push_back(plane(DType::U8, b.key_side));
            g.planes.push_back(plane(DType::U8, b.value_side));
        }
    }
    if (dspark) {
        static_assert(ops::inkling2::kDSparkKvHeads == ops::inkling2::kKvHeads &&
                      ops::inkling2::kDSparkHeadDim == ops::inkling2::kHeadDim);
        g.planes.push_back(plane(DType::BF16, ops::inkling2::kDSparkHeadDim));
        g.planes.push_back(plane(DType::BF16, ops::inkling2::kDSparkHeadDim));
    }
    return g;
}

// Whether the Program runs the drafter (its two planes in every page).
bool dspark_planes(const ProgramOptions& o) { return o.dspark && !o.causal_scoring; }

// The full visits' page pool (its scratch page included), shared by the lanes, and their block-table rows in one
// backing allocation. The pool holds at least one lane's whole context, so a request alone always completes.
struct KvLayout {
    DeviceKVPagePoolLayout pool;
    KVExecutionTableLayout tables;
    std::uint64_t bytes      = 0;
    std::uint64_t page_bytes = 0; // one page of every plane
};
KvLayout kv_layout(const ProgramOptions& o, std::int32_t full_visits) {
    const std::uint32_t row   = pages_per_lane(o.max_context);
    const std::uint64_t pages = std::max<std::uint64_t>(row, o.kv_pages) + 1;
    if (pages > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Inkling 2: the KV pool has more pages than a block table names");
    }
    KvLayout out;
    LayoutBuilder builder;
    out.pool   = plan_device_kv_page_pool(builder, {.page_group_count = static_cast<std::uint32_t>(pages),
                                                    .geometry = kv_page_geometry(full_visits, o.kv_codec, dspark_planes(o),
                                                                                 o.lossless)});
    out.tables = plan_kv_execution_tables(builder, {.logical_page_capacity = row,
                                                    .table_rows = static_cast<std::int32_t>(o.max_concurrency)});
    out.bytes  = builder.finish(256);
    for (const KVPlaneGeometry& plane : out.pool.spec.geometry.planes) {
        out.page_bytes += dtype_size(plane.dtype) * static_cast<std::uint64_t>(plane.leading_extent) *
                          ops::inkling2::kKeyUnit * static_cast<std::uint64_t>(plane.head_extent);
    }
    return out;
}

std::uint64_t align_256(std::uint64_t bytes) { return (bytes + 255) / 256 * 256; }

// The sampled rows of one call at most: a DSpark round of several rows verifies up to a block of columns each.
constexpr std::int32_t kMaxSampleRows = kMaximumConcurrency * ops::inkling2::kDSparkBlock;
// The graphs of DSpark rounds of several rows a Program keeps (the most recently used): one batch size's take 14 (the
// verify forward at 7 widths, the drafts at 7 lengths), so three batch sizes alternate without a recapture.
constexpr std::int32_t kRowsGraphs = 42;

// Constrained rows' token masks (constrained-decoding.md §7.1): one packed vocabulary mask a position. Slot r is a plain
// round's row r, slot i a DSpark verify call's column i (a row's columns in a round of several rows), so the slots
// cover both: a block a lane with the drafter, a row a lane without.
std::int32_t mask_slots(const ProgramOptions& options) {
    return options.dspark ? std::max<std::int32_t>(static_cast<std::int32_t>(options.max_concurrency), 1) *
                                ops::inkling2::kDSparkBlock
                          : kMaximumConcurrency;
}
std::uint64_t mask_words(std::uint64_t vocab) { return (vocab + 31) / 32; }
// Device and pinned bytes of the masks (generation only: scoring samples nothing).
std::uint64_t mask_bytes(const ProgramOptions& options, std::uint64_t vocab) {
    return options.causal_scoring ? 0 : align_256(4 * mask_words(vocab) * static_cast<std::uint64_t>(mask_slots(options)));
}

// Token constraints' operands for one sample call of at most kMaximumConcurrency rows (ops::constrain_logits), in 4-byte
// words: every lane's sets [kChoices][lanes x kSets] and their counts, then per row its descriptor, its record, its
// observe list and its observed record (a constrained row samples on the plain route, never in a DSpark verify call).
struct ConstraintWords {
    std::size_t sets = 0, choices = 0, counts = 0, descriptors = 0, records = 0, observe = 0, observed = 0, total = 0;
};
ConstraintWords constraint_words(std::uint32_t lanes) {
    constexpr std::size_t rows = kMaximumConcurrency;
    ConstraintWords w;
    w.sets        = static_cast<std::size_t>(lanes) * kMaximumConstraintSets;
    w.choices     = 0;
    w.counts      = w.choices + w.sets * ops::kTokenConstraintChoices;
    w.descriptors = w.counts + w.sets;
    w.records     = w.descriptors + rows;
    w.observe     = w.records + rows * ops::kTokenConstraintRecord;
    w.observed    = w.observe + rows * ops::kTokenObserveTokens;
    w.total       = w.observed + rows * ops::kTokenObserveRecord;
    return w;
}
std::uint64_t constraint_bytes(const ProgramOptions& options) {
    return options.causal_scoring ? 0 : align_256(4 * constraint_words(options.max_concurrency).total);
}

} // namespace

// A lane's request (ProgramImpl::lanes_): its ledger, sampling and speculative state and its prefix-cache references.
// A pause keeps it in its ResumeState with its device references dropped (ResumeStateImpl).
struct Lane {
    bool bound          = false;
    std::uint64_t epoch = 0;
    std::vector<TokenId> tokens;       // the prompt, then the committed output and forced tokens
    std::vector<std::uint32_t> forced; // the positions in `tokens` of forced tokens (the penalties count sampled ones)
    std::shared_ptr<const std::vector<PromptImage>> images; // the prompt's images (null: none)
    std::uint32_t prompt_tokens = 0;
    std::uint32_t processed     = 0; // positions whose state the lane holds
    // A resumed request rebuilding its state toward `replay_target`, its paused frontier (advance_replay).
    bool replay                 = false;
    std::uint32_t replay_target = 0;
    // The admission's terms its first binding reported (a resume keeps them), and the prompt positions the Engine was
    // told were prefilled (a prompt resumed below its paused frontier reports nothing until it passes it again).
    std::uint32_t reused_prompt = 0;
    std::uint32_t counted       = 0;
    runtime::ResolvedExecutionOptions options;
    bool penalties = false;
    // A token constraint: its plan, the round's draw (committed with its token) and the request's draws.
    std::shared_ptr<const ConstraintPlan> constraint;
    std::vector<ConstrainedDraw> constraint_round, constraint_trace;
    std::uint64_t prefill_ns = 0, decode_ns = 0, decode_share_ns = 0;
    // The prompt calls' time on the device (prefill_seconds) and the timing events of those not yet read (settled
    // before a pause copies the lane, so a ResumeState holds none).
    std::uint64_t prefill_device_ns = 0;
    std::vector<std::array<cudaEvent_t, 2>> prefill_events;
    // DSpark: the drafts' acceptance by probability decile (draft_length.h) and the request's counters.
    runtime::speculative::DraftCalibration calibration;
    std::uint64_t policy_rounds = 0;
    std::int32_t step_cap = ops::inkling2::kMarkovSteps; // the calibrated length's Markov steps a round drafts
    std::uint64_t rounds = 0, drafted = 0, accepted = 0, fallback = 0;
    double draft_ms = 0, gap_ms = 0, verify_ms = 0; // device-side sums over the rounds (CUDA events)
    std::array<std::uint64_t, ops::inkling2::kMarkovSteps> accepted_at{}; // rounds accepting draft i
    // The prefix cache: whether the request reuses and publishes, its keys (a block past the prompt's full blocks under
    // `salt`, a decode-route key only resumes ask for: pause_keys), the blocks its row maps from the cache (path, root
    // first: pinned for the lane), the taps still to take, the snapshot frontiers on its path, what it published
    // (withdrawn when its logits go non-finite), the restore its first call waits for and the capture its next call's
    // image writes wait for.
    struct Prefix {
        bool reuse = false;
        std::vector<std::uint64_t> hashes, extras;
        std::uint64_t salt = 0; // 0: no decode-route key yet (pause_keys)
        std::vector<runtime::prefix_cache::NodeRef> path;
        std::vector<runtime::prefix_cache::PlannedTap> taps;
        std::vector<std::uint32_t> frontiers;
        std::uint32_t base = 0; // the frontier the lane was bound at (0: root)
        std::vector<runtime::prefix_cache::NodeRef> published;
        std::vector<std::uint64_t> captures;
        std::optional<prefix::RestoreTicket> restore;
        std::uint64_t capture = 0;
        bool withdraw = false;
        std::uint64_t saves = 0; // the cache file's saves at the binding (prefix_saves_)
        std::uint32_t deepest_capture = 0; // the deepest frontier the lane's own accepted captures took
    } prefix;
};

// A paused request (Program::start_pause): the lane's request as the pause left it (its pins and pages released), the
// frontier its state held, whether it was past its prompt, and the pause's endpoint capture (0: none), which its
// resume's quote waits for.
struct ResumeStateImpl {
    Lane lane;
    std::uint32_t target  = 0;
    bool generated        = false;
    std::uint64_t capture = 0;
};

class ProgramImpl {
public:
    ProgramImpl(const Parameters& parameters, DeviceContext& device, ProgramOptions options)
        : p_(parameters), c_(parameters.model.config().text), device_(device), o_(std::move(options)),
          plan_(Program::plan_device(o_, parameters.model.config())),
          forward_(parameters, static_cast<std::int32_t>(o_.max_concurrency), forward_capacity(o_), expert_activation(o_),
                   o_.a8, o_.window_codec),
          vocab_(static_cast<std::int32_t>(c_.vocab_size)), lanes_(o_.max_concurrency),
          sampling_(ops::sampling_workspace_capacity_bytes(vocab_, 1, static_cast<std::int32_t>(kMaximumConcurrency)) +
                    4096),
          host_(align_256(forward_.upload_words() * 4 * kStagingSlots) +
                align_256(sizeof(ops::SamplingConfig) * kMaxSampleRows) + 3 * align_256(4 * kMaxSampleRows) +
                align_256(4)) {
        const StateLayout& layout = forward_.layout();
        const std::uint32_t pages = pages_per_lane(o_.max_context);
        if (o_.lossless) {
            lossless_ = std::make_unique<LosslessForward>(p_, lossless_columns(o_), o_.reference, lossless_keys(o_));
        } else if (o_.causal_scoring) { // ScoreOptions::prompt_hidden_states: BF16 [layers][H]
            score_hidden_ = DeviceBuffer(static_cast<std::size_t>(c_.num_hidden_layers) * c_.hidden_size * 2);
        }
        images_ = DeviceBuffer(static_cast<std::size_t>(layout.lane_words) * 2 * lanes_.size());
        if (o_.window_codec != ops::inkling2::KvCodec::Bf16) {
            if (o_.lossless) { throw std::invalid_argument("Inkling 2: the lossless route keeps BF16 windows"); }
            // The quantized windows' codes and side records, zeroed (zero scales decode zero codes to zero).
            window_codes_ = DeviceBuffer(static_cast<std::size_t>(window_code_bytes(layout, o_.window_codec)) * lanes_.size());
            window_side_  = DeviceBuffer(static_cast<std::size_t>(window_side_bytes(layout, o_.window_codec)) * lanes_.size());
            window_codes_.fill(0);
            window_side_.fill(0);
            view_.window_codes  = static_cast<std::uint8_t*>(window_codes_.p);
            view_.window_side   = static_cast<std::uint8_t*>(window_side_.p);
            view_.window_planes = 2 * layout.sliding_visits;
        }
        view_.window_codec = o_.window_codec;
        // The full visits' page pool and the lanes' block-table rows (kv_layout), zeroed: a key beyond a lane's frontier
        // is masked, never NaN, and zero scales decode zero codes to zero.
        const KvLayout kv = kv_layout(o_, layout.full_visits);
        kv_backing_       = DeviceBuffer(kv.bytes);
        kv_backing_.fill(0);
        const DeviceSpan span{kv_backing_.p, kv_backing_.bytes};
        pool_   = std::make_unique<DeviceKVPagePool>(span, kv.pool);
        tables_ = std::make_unique<KVExecutionTablePool>(span, kv.tables, *pool_);
        // The scratch page (the lowest) every unmapped table entry names: a row past its lane's frontier is never read.
        std::optional<DeviceKVPageReservation> scratch = pool_->reserve(1);
        if (!scratch) { throw std::logic_error("Inkling 2: the KV pool has no scratch page"); }
        scratch_ = pool_->materialize_one(*scratch);
        for (std::uint32_t i = 0; i < lanes_.size(); ++i) {
            kv_rows_.push_back(tables_->acquire(static_cast<std::int32_t>(i)));
            tables_->publish_repeated(kv_rows_.back().handle(), scratch_.handle(), pages, device_.stream);
        }
        kv_lanes_.resize(lanes_.size());
        for (LaneKv& kv : kv_lanes_) { kv.pages.resize(pages); }
        view_.images     = static_cast<std::uint16_t*>(images_.p);
        view_.lane_words = layout.lane_words;
        const std::size_t per_visit = kv_planes_per_visit(o_.kv_codec);
        for (std::int32_t s = 0; s < layout.full_visits; ++s) {
            const std::size_t at = static_cast<std::size_t>(s) * per_visit;
            view_.pool_k.push_back(static_cast<std::uint16_t*>(pool_->plane(at).data));
            view_.pool_v.push_back(static_cast<std::uint16_t*>(pool_->plane(at + 1).data));
            if (per_visit == 4) {
                view_.side_k.push_back(static_cast<std::uint8_t*>(pool_->plane(at + 2).data));
                view_.side_v.push_back(static_cast<std::uint8_t*>(pool_->plane(at + 3).data));
            }
        }
        view_.codec        = o_.kv_codec;
        view_.pool_pages   = static_cast<std::int64_t>(pool_->capacity_pages());
        view_.table        = static_cast<const std::int32_t*>(tables_->matrix().data);
        view_.table_stride = static_cast<std::int32_t>(tables_->logical_page_capacity());
        counts_            = DeviceBuffer(static_cast<std::size_t>(vocab_) * 4 * lanes_.size());
        configs_           = DeviceBuffer(sizeof(ops::SamplingConfig) * kMaxSampleRows);
        positions_         = DeviceBuffer(4 * kMaxSampleRows);
        sampled_           = DeviceBuffer(4 * kMaxSampleRows);
        if (const std::uint64_t bytes = mask_bytes(o_, static_cast<std::uint64_t>(vocab_)); bytes != 0) {
            masks_ = DeviceBuffer(bytes);
            host_masks_.emplace(bytes);
        }
        if (const std::uint64_t bytes = constraint_bytes(o_); bytes != 0) {
            constraint_ = DeviceBuffer(bytes);
            constraint_host_.emplace(bytes);
        }
        auto* base         = static_cast<std::byte*>(host_.data());
        staging_           = reinterpret_cast<std::int32_t*>(base);
        base += align_256(forward_.upload_words() * 4 * kStagingSlots);
        host_configs_ = reinterpret_cast<ops::SamplingConfig*>(base);
        base += align_256(sizeof(ops::SamplingConfig) * kMaxSampleRows);
        host_positions_ = reinterpret_cast<std::int32_t*>(base);
        base += align_256(4 * kMaxSampleRows);
        host_sampled_ = reinterpret_cast<std::int32_t*>(base);
        base += align_256(4 * kMaxSampleRows);
        nonfinite_ = reinterpret_cast<std::int32_t*>(base);
        base += align_256(4 * kMaxSampleRows);
        // The MoE's error word, mapped: a down CTA that waited past its bound sets it, and finish_sample reads it after
        // the round's own synchronization (no copy).
        moe_error_  = reinterpret_cast<std::uint32_t*>(base);
        *moe_error_ = 0;
        forward_.set_moe_error_word(moe_error_);
        if (o_.kv_codec == ops::inkling2::KvCodec::Bf16Packed || o_.window_codec == ops::inkling2::KvCodec::Bf16Packed) {
            // bf16p pools or rings: each lane's lowest appended position with no packed form (atomicMin; 0xFFFFFFFF:
            // none).
            unpackable_ = DeviceBuffer(4 * kMaximumConcurrency);
            CUDA_CHECK(cudaMemset(unpackable_.p, 0xFF, 4 * kMaximumConcurrency));
            view_.unpackable = static_cast<std::uint32_t*>(unpackable_.p);
        }
        if (o_.dspark && !o_.causal_scoring) {
            if (!p_.dspark) { throw std::invalid_argument("Inkling 2: --spec dspark needs the drafter's parameters"); }
            if (o_.draft_tokens < 1 || o_.draft_tokens > static_cast<std::uint32_t>(ops::inkling2::kMarkovSteps)) {
                throw std::invalid_argument("Inkling 2: DSpark drafts 1-7 tokens a round");
            }
            // The full layer's context: the pool's last two planes, through the lanes' block-table rows.
            const std::size_t planes = pool_->plane_count();
            const DSparkPagedContext full{.k            = static_cast<std::uint16_t*>(pool_->plane(planes - 2).data),
                                          .v            = static_cast<std::uint16_t*>(pool_->plane(planes - 1).data),
                                          .table        = view_.table,
                                          .table_stride = view_.table_stride};
            drafter_ = std::make_unique<Drafter>(p_, forward_, static_cast<std::int32_t>(lanes_.size()), o_.max_context,
                                                 forward_capacity(o_), full);
        }
        if (o_.vision) {
            if (!p_.vision || !parameters.model.config().vision) {
                throw std::invalid_argument("Inkling 2: --vision needs the tower's parameters");
            }
            // --vision-offload on: the model's pinned tower, streamed through the tower's slots on the transfer stream.
            const auto& host = parameters.model.vision_host();
            vision_ = std::make_unique<VisionTower>(*parameters.model.config().vision, *p_.vision, o_.vision_max_cells,
                                                    host ? &*host : nullptr, device_.transfer_stream);
        }
        if (o_.prefix_cache && !o_.causal_scoring) {
            // A snapshot's image: the lane's state image and, with DSpark, the drafter's sliding rings.
            prefix::PrefixCacheConfig config;
            config.taps     = {.max_new_taps   = o_.prefix_max_new_taps,
                               .ladder_tokens  = o_.prefix_ladder_tokens,
                               .min_gap_tokens = o_.prefix_min_gap_tokens};
            config.kv_pages = pool_->capacity_pages();
            // The source choice's machine model (runtime/prefix_cache/cost.h), a linear fit of the prompt route's measured
            // rates (RTX 5090, infernix_bench C = 1, pp2048, 2026-11-21: 9,514 tokens a second in chunks of 512, 16,400
            // in chunks of 2,048): ~30 ms a call and 1 / 21,600 s a token. Host-to-Device at 22 GB/s, an estimate for
            // the PCIe x8 link (restores are not measured yet). It only ranks a snapshot against a root start.
            config.index.cost.chunk_tokens           = o_.prefill_chunk;
            config.index.cost.chunk_seconds          = 30.0e-3;
            config.index.cost.token_seconds          = 1.0 / 21600.0;
            config.index.cost.attention_pair_seconds = 0.0;
            if (o_.lossless) {
                // The lossless route's calls (lossless-route.md §3a; inkling2-mini, RTX 5090, 2026-11-23): about 0.27 s a
                // call (infernix --lossless, one-column decode calls at a 20-token context, 3.6 tok/s), then 3.1 ms a
                // token and 0.89 us a (query, key) pair, fitted to its scoring runs (infernix-perplexity: 11 windows of
                // 2,048 tokens in 93.7 s, 2 of 9,216 in 133.5 s).
                config.index.cost.chunk_seconds          = 0.27;
                config.index.cost.token_seconds          = 3.1e-3;
                config.index.cost.attention_pair_seconds = 0.89e-6;
            }
            config.index.cost.h2d_bytes_per_second   = 22.0e9;
            config.ssd_bytes                         = o_.prefix_ssd_bytes;
            config.ssd_directory                     = o_.prefix_ssd_directory;
            config.ssd_write_bytes_per_hour          = o_.prefix_ssd_write_bytes_per_hour;
            config.ssd_kept_file                     = o_.prefix_ssd_kept_file;
            config.host_ceiling_bytes                = o_.prefix_host_ceiling_bytes;
            prefix::LaneImages images;
            const std::size_t lane_bytes = static_cast<std::size_t>(forward_.layout().lane_words) * 2;
            images.parts.push_back({static_cast<std::byte*>(images_.p), lane_bytes, lane_bytes});
            for (const DeviceBuffer* window : {&window_codes_, &window_side_}) { // quantized windows (design §7.6)
                if (window->bytes == 0) { continue; }
                const std::size_t bytes = window->bytes / lanes_.size();
                images.parts.push_back({static_cast<std::byte*>(window->p), bytes, bytes});
            }
            if (drafter_) {
                for (const DSparkStatePart& part : drafter_->snapshot_parts()) {
                    const std::size_t bytes = static_cast<std::size_t>(part.lane_words) * 2;
                    images.parts.push_back({reinterpret_cast<std::byte*>(part.base), bytes, bytes});
                }
            }
            prefix_ = std::make_unique<prefix::PrefixCache>(config, *pool_, images, o_.prefix_host_bytes, device_.stream);
        }
        if (o_.use_cuda_graph && !o_.causal_scoring && !o_.lossless) { capture_decode_graphs(); }
    }

    ~ProgramImpl() {
        // The cache's transfers and the compute stream's last calls land before any buffer they read is freed.
        (void)cudaStreamSynchronize(device_.stream);
        ram_.reset(); // the RAM monitor's thread and worker stop (finishing their jobs) before the cache they resize goes
        prefix_.reset();
        for (Lane& lane : lanes_) { drop_prefill_events(lane); }
        for (const cudaEvent_t event : timing_events_) { (void)cudaEventDestroy(event); }
    }

    // ---------------------------------------------------------------- admission
    RequestBasePlan plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options) {
        if (!prompt) { throw std::invalid_argument("Inkling 2: an empty prepared prompt"); }
        std::unique_ptr<PreparedPromptData> data = std::move(prompt).take();
        const auto n                             = static_cast<std::uint32_t>(data->token_ids.size());
        if (n == 0 || n > o_.max_context) { throw std::invalid_argument("Inkling 2: the prompt is empty or exceeds max_context"); }
        for (const TokenId id : options.readout_tokens) {
            if (id < 0 || id >= vocab_) { throw std::invalid_argument("readout token is outside the public token domain"); }
        }
        auto base                             = std::make_unique<BasePlanImpl>();
        base->constraint                      = compile_constraint(options.constraint);
        if (!data->images.empty()) {
            check_images(data->images, n);
            base->images = std::make_shared<const std::vector<PromptImage>>(std::move(data->images));
        }
        base->tokens                          = std::move(data->token_ids);
        base->options                         = options;
        // Background requests (next-turn prefill) are not built for this model; a request that may not reuse history
        // also publishes none.
        base->reuse        = prefix_ != nullptr && options.allow_prefix_reuse && !options.background && data->reusable;
        base->block_hashes = std::move(data->block_hashes);
        base->block_extras = std::move(data->block_extras);
        base->tap_hints    = std::move(data->tap_hints);
        base->summary.prompt_tokens           = n;
        base->summary.requested_output_tokens = options.requested_output_tokens;
        const std::uint32_t capacity_output   = o_.max_context - n + 1U;
        base->summary.effective_output_tokens = std::min(options.requested_output_tokens, capacity_output);
        base->summary.effective_limit_reason  = options.requested_output_tokens <= capacity_output
                                                    ? FinishReason::OutputLimit
                                                    : FinishReason::ContextCapacity;
        base->summary.publish_continuation    = false;
        return RequestBasePlan(std::move(base));
    }

    // The prompt's images as spans no resume point may lie in (tap_planner.h media_exclusion).
    static std::vector<runtime::prefix_cache::TapExclusion>
    image_spans(const std::shared_ptr<const std::vector<PromptImage>>& images) {
        std::vector<runtime::prefix_cache::TapExclusion> out;
        if (images) {
            for (const PromptImage& image : *images) {
                out.push_back(runtime::prefix_cache::media_exclusion(
                    image.first_token, image.first_token + static_cast<std::uint32_t>(image.image.cells())));
            }
        }
        return out;
    }
    static std::vector<runtime::prefix_cache::TapExclusion> exclusions(const BasePlanImpl& b) { return image_spans(b.images); }
    static bool excluded(std::span<const runtime::prefix_cache::TapExclusion> spans, std::uint32_t frontier) {
        return std::any_of(spans.begin(), spans.end(), [&](const auto& s) { return frontier > s.begin && frontier < s.end; });
    }

    // Admission's sources: the snapshot the cost model chooses (the deepest worth restoring, outside every image), then
    // a root start. A paused request matches its ledger (its prompt's keys, then its private ones) up to its paused
    // frontier and takes the deepest state cached there: its pause's endpoint while the cache still holds it.
    std::vector<SourceCandidate> sources(const RequestBasePlan& base, const ResumeState* resume) {
        SourceCandidate root;
        const BasePlanImpl& b = *base.impl_;
        prefix_ssd_wait_ = false;
        if (!prefix_ || !b.reuse || prefix_failed_) { return {root}; }
        namespace pc = runtime::prefix_cache;
        const ResumeStateImpl* r = ContractAccess::resume_impl(resume);
        const std::vector<TokenId>& tokens       = r != nullptr ? r->lane.tokens : b.tokens;
        const std::vector<std::uint64_t>& hashes = r != nullptr ? r->lane.prefix.hashes : b.block_hashes;
        const std::vector<std::uint64_t>& extras = r != nullptr ? r->lane.prefix.extras : b.block_extras;
        // match returns the snapshots below its count: a resume's at or below its paused frontier.
        const auto n = r != nullptr ? std::min<std::uint32_t>(static_cast<std::uint32_t>(tokens.size()), r->target + 1U)
                                    : static_cast<std::uint32_t>(tokens.size());
        // The pause's endpoint lands first, so the match sees it: that copy alone (and what the transfer stream queued
        // before it). A resume that quoted before it landed would replay its reply at 16 positions a call.
        if (r != nullptr && r->capture != 0) {
            for (const cudaEvent_t event : prefix_->capture_events(r->capture)) { CUDA_CHECK(cudaEventSynchronize(event)); }
        }
        prefix_->poll();
        pc::MatchResult match = prefix_->index().match(tokens, hashes, extras, n);
        // A tap another lane is still copying out that this prompt extends: it publishes first.
        if (r == nullptr && prefix_->await_extended_capture(match.path, tokens)) {
            match = prefix_->index().match(tokens, hashes, extras, n);
        }
        root.cached_block_tokens = static_cast<std::uint32_t>(match.path.size()) * pc::kBlockTokens;
        const auto spans = image_spans(r != nullptr ? r->lane.images : b.images);
        std::erase_if(match.candidates, [&](const pc::MatchCandidate& c) {
            return excluded(spans, c.frontier) || c.filling_blocks != 0;
        });
        if (match.candidates.empty()) { return {root}; }
        std::optional<std::size_t> picked;
        if (r != nullptr) {
            for (std::size_t i = 0; i < match.candidates.size(); ++i) {
                if (!picked || match.candidates[i].frontier > match.candidates[*picked].frontier) { picked = i; }
            }
        } else {
            picked = prefix_->index().choose(match, n).candidate;
        }
        if (!picked) { return {root}; }
        const pc::MatchCandidate& c = match.candidates[*picked];
        if (c.ssd_bytes != 0) {
            // Part of it is only on the SSD tier (hybrid-prefix-cache-spec §5.7): the quote starts the read and the
            // request waits for it (no sources: hybrid_source_pending), or starts at the root when the Host tier cannot
            // take it. The cost model already weighed the read.
            const pc::SsdTier::Promotion promotion = prefix_->promote(c.snapshot);
            if (promotion == pc::SsdTier::Promotion::Pending) {
                prefix_ssd_wait_ = true;
                return {};
            }
            if (promotion == pc::SsdTier::Promotion::Unavailable) { return {root}; }
        }
        auto quote      = std::make_shared<QuoteImpl>();
        quote->snapshot = c.snapshot;
        quote->frontier = c.frontier;
        quote->path.assign(match.path.begin(), match.path.begin() + c.path_blocks);
        SourceCandidate chosen;
        chosen.reused_tokens       = c.frontier;
        chosen.quote               = std::move(quote);
        chosen.reuse_path          = PrefixReusePath::HybridSnapshot;
        chosen.cached_block_tokens = root.cached_block_tokens;
        return {chosen, root};
    }

    // Makes `pages` free in the pool: evicts cached blocks (none a lane maps), waiting for a transfer when only
    // transfers hold the rest. False when the other lanes hold the pages.
    [[nodiscard]] bool make_room(std::uint32_t pages) {
        while (pool_->available_pages() < pages) {
            if (prefix_ && prefix_->index().evict_device_blocks(pages - pool_->available_pages()) != 0) { continue; }
            if (!prefix_ || !prefix_->settle_oldest_transfer()) { return false; }
        }
        return true;
    }

    // Whether the cache can give `pages` up (every free page, every evictable cached block and those only in-flight Host
    // writes hold, once they land): always while no other lane holds pages, since every cached block can then go.
    // Otherwise a binding makes no room and reports its shortage with reclaim_cache false, and the Engine pauses a
    // resident for a resume or defers a fresh request, instead of evicting the request's own cached history for nothing.
    [[nodiscard]] bool room_possible(std::uint32_t lane, std::uint32_t pages) const noexcept {
        if (!prefix_) { return pool_->available_pages() >= pages; }
        const bool resident = std::any_of(lanes_.begin(), lanes_.end(), [&](const Lane& other) {
            return other.bound && &other != &lanes_[lane];
        });
        const runtime::prefix_cache::PrefixCacheIndex& cache = prefix_->index();
        return !resident || std::uint64_t{pages} <= std::uint64_t{pool_->available_pages()} +
                                                        cache.device_evictable_blocks() +
                                                        cache.device_transfer_held_blocks();
    }

public:
    bool reclaim(runtime::ContextResourceUsage shortage) {
        if (shortage.main_kv_pages == 0 || !prefix_) { return false; }
        const std::uint32_t before = pool_->available_pages();
        (void)make_room(before + shortage.main_kv_pages);
        return pool_->available_pages() > before;
    }

private:
    // The positions a unit of the lane may write past its frontier: a DSpark round's block rows (F .. F + 7, the verify
    // call's columns among them), else one.
    [[nodiscard]] std::uint32_t round_positions() const noexcept {
        return drafter_ ? static_cast<std::uint32_t>(ops::inkling2::kDSparkBlock) : 1U;
    }
    // The lane's block-table entries that positions [0, end) need (no more than its row).
    [[nodiscard]] std::uint32_t pages_through(std::uint64_t end) const noexcept {
        const std::uint64_t pages = (end + ops::inkling2::kKeyUnit - 1) / ops::inkling2::kKeyUnit;
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(pages, lane_pages()));
    }

public:
    BindingReservation start_binding(const RequestBasePlan& base, runtime::LaneId lane_id, const SourceCandidate& source,
                                     ResumeState* resume) {
        if (binding_ || pause_) { throw std::logic_error("Inkling 2: a context transaction is already open"); }
        if (lane_id.value >= lanes_.size() || lanes_[lane_id.value].bound) {
            throw std::logic_error("Inkling 2: the binding's lane is not free");
        }
        ResumeStateImpl* r = ContractAccess::resume_impl(resume);
        if (resume != nullptr && r == nullptr) { throw std::logic_error("Inkling 2: a resume carries no state"); }
        namespace pc            = runtime::prefix_cache;
        const BasePlanImpl& b   = *base.impl_;
        const std::uint32_t idx = lane_id.value;
        // A snapshot source still as quoted: its snapshot, path and image on the Host, nothing of it being filled.
        const QuoteImpl* quote = source.quote.get();
        if (quote != nullptr) {
            const auto bad = [&] {
                BindingReservation out;
                out.source_valid = false;
                return out;
            };
            if (!prefix_ || !b.reuse) { return bad(); }
            prefix_->poll();
            pc::PrefixCacheIndex& index = prefix_->index();
            if (!index.valid(quote->snapshot)) { return bad(); }
            const pc::SnapshotView view = index.snapshot(quote->snapshot);
            const std::uint32_t k       = quote->frontier / pc::kBlockTokens;
            if (view.frontier != quote->frontier || quote->path.size() != k || view.host != pc::CopyState::Resident ||
                (k > 0 && view.anchor != quote->path.back()) || (k == 0 && view.anchor.valid())) {
                return bad();
            }
            for (const pc::NodeRef node : quote->path) {
                if (!index.valid(node)) { return bad(); }
                const pc::NodeView n = index.node(node);
                if (n.device == pc::CopyState::Filling || (n.device != pc::CopyState::Resident && n.host != pc::CopyState::Resident)) {
                    return bad();
                }
            }
        }
        const std::uint32_t k = quote != nullptr ? quote->frontier / runtime::prefix_cache::kBlockTokens : 0U;
        // The path's pins hold its blocks through the room-making below (a binding never evicts what it maps).
        pc::PathPin pins;
        if (quote != nullptr && k > 0) { pins = pc::PathPin(prefix_->index(), quote->path); }
        std::uint32_t host_only = 0;
        for (std::uint32_t i = 0; i < k; ++i) {
            host_only += prefix_->index().node(quote->path[i]).device != runtime::prefix_cache::CopyState::Resident ? 1U : 0U;
        }
        // The lane's pages past the path's blocks (which it maps from the cache): through its first prompt chunk, or a
        // resume's through its paused frontier and one round, which its replay never outgrows (so a resumed request
        // reaches new progress before it can be short again). The path's Host-only blocks take pages of their own,
        // which the cache owns once restored.
        const std::uint32_t frontier = quote != nullptr ? quote->frontier : 0U;
        const std::uint32_t prompt   = r != nullptr ? r->lane.prompt_tokens : b.summary.prompt_tokens;
        std::uint64_t end            = std::min<std::uint64_t>(prompt, std::uint64_t{frontier} + o_.prefill_chunk);
        if (r != nullptr) {
            end = r->generated ? std::uint64_t{r->target} + round_positions() : std::max<std::uint64_t>(end, r->target);
        }
        const std::uint32_t own  = pages_through(end) > k ? pages_through(end) - k : 0U;
        const std::uint32_t need = own + host_only;
        if (!room_possible(idx, need) || !make_room(need)) {
            // make_room took what the cache could give: the Engine pauses a resident for a resume, or defers.
            BindingReservation out;
            out.shortage.main_kv_pages = need > pool_->available_pages() ? need - pool_->available_pages() : 1U;
            out.reclaim_cache          = false;
            out.capacity_possible      = std::uint64_t{pages_through(end)} + host_only < pool_->capacity_pages();
            return out;
        }
        reserve_pages(idx, own);
        Lane& lane = lanes_[lane_id.value];
        if (r != nullptr) {
            // The request as its pause left it: its ledger, sampling, constraint, timings and speculative counters.
            lane                = r->lane;
            lane.prefix.path    = {};
            lane.prefix.taps    = {};
            lane.prefix.restore.reset();
            lane.prefix.capture = 0;
        } else {
            lane               = Lane{};
            lane.tokens        = base.impl_->tokens;
            lane.images        = base.impl_->images;
            lane.prompt_tokens = base.impl_->summary.prompt_tokens;
            lane.options       = base.impl_->options;
            lane.constraint    = base.impl_->constraint;
            lane.penalties = lane.options.sampling.presence_penalty != 0.0F || lane.options.sampling.frequency_penalty != 0.0F;
            lane.calibration.reset();
            lane.reused_prompt = frontier;
            lane.counted       = frontier;
            lane.prefix.reuse  = b.reuse;
            lane.prefix.saves  = prefix_saves_;
            lane.prefix.hashes = b.block_hashes;
            lane.prefix.extras = b.block_extras;
        }
        lane.bound     = true;
        lane.epoch     = ++epoch_;
        lane.processed = 0;
        lane.replay    = false;
        clear_unpackable(lane_id.value);
        if (lane.penalties && r != nullptr) {
            rebuild_counts(idx);
        } else if (lane.penalties) {
            CUDA_CHECK(cudaMemsetAsync(lane_counts(idx), 0, static_cast<std::size_t>(vocab_) * 4, device_.stream));
        }
        if (quote == nullptr) {
            // A fresh state: the conv rings' history before position 0 is zero, and the window's rows are rewritten.
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::uint16_t*>(images_.p) +
                                           static_cast<std::size_t>(lane_id.value) * forward_.layout().lane_words,
                                       0, static_cast<std::size_t>(forward_.layout().lane_words) * 2, device_.stream));
            clear_windows(static_cast<std::size_t>(lane_id.value));
        } else {
            bind_snapshot(idx, *quote, k);
            (void)pins.release(); // the lane's now (released with it)
        }
        lane.prefix.base = frontier;
        if (r != nullptr && r->generated) {
            // Past its prompt: the lane replays toward its paused frontier (none to replay when the cache held it), and
            // publishes the replay's blocks but takes no taps.
            lane.replay        = lane.processed < r->target;
            lane.replay_target = r->target;
            // A lane that decodes at once: a DSpark round drafts before its verify call's upload orders the stream
            // after the cache, so the compute stream waits for the whole restore here.
            if (!lane.replay && lane.prefix.restore) {
                prefix_->order_after_restore(*lane.prefix.restore, device_.stream);
                lane.prefix.restore.reset();
            }
            lane.prefix.frontiers.clear();
            if (lane.prefix.reuse) {
                const pc::MatchResult match = prefix_->index().match(lane.tokens, lane.prefix.hashes, lane.prefix.extras,
                                                                     std::min<std::uint32_t>(static_cast<std::uint32_t>(
                                                                                                 lane.tokens.size()),
                                                                                             r->target + 1U));
                for (const pc::MatchCandidate& c : match.candidates) { lane.prefix.frontiers.push_back(c.frontier); }
            }
        } else if (lane.prefix.reuse) {
            plan_lane_taps(idx, b);
        }
        binding_ = lane_id.value;
        BindingReservation out;
        out.reserved = true;
        return out;
    }

    // A resumed lane's penalty counts (the sampler counts every token it draws, so the forced tokens are left out): the
    // committed output's sampled tokens, counted on the host and copied in on the compute stream.
    void rebuild_counts(std::uint32_t idx) {
        const Lane& lane = lanes_[idx];
        penalty_host_.assign(static_cast<std::size_t>(vocab_), 0);
        std::size_t next_forced = 0;
        for (std::size_t p = lane.prompt_tokens; p < lane.tokens.size(); ++p) {
            while (next_forced < lane.forced.size() && lane.forced[next_forced] < p) { ++next_forced; }
            if (next_forced < lane.forced.size() && lane.forced[next_forced] == p) { continue; }
            const TokenId token = lane.tokens[p];
            if (token >= 0 && token < vocab_) { ++penalty_host_[static_cast<std::size_t>(token)]; }
        }
        // A pageable source: the call returns once the counts are staged, in the compute stream's order.
        CUDA_CHECK(cudaMemcpyAsync(lane_counts(idx), penalty_host_.data(), penalty_host_.size() * 4, cudaMemcpyHostToDevice,
                                   device_.stream));
    }

    // A binding from snapshot `quote` (its path pinned by the caller, `k` full blocks): the path's Device blocks mapped,
    // its Host-only blocks restored into new cache pages, the tail page (a frontier inside a block) restored into the
    // lane's first own page, and the image into the lane; the lane's first call waits for the restore.
    void bind_snapshot(std::uint32_t idx, const QuoteImpl& quote, std::uint32_t k) {
        Lane& lane = lanes_[idx];
        namespace pc                 = runtime::prefix_cache;
        pc::PrefixCacheIndex& index  = prefix_->index();
        LaneKv& kv                   = kv_lanes_[idx];
        const std::uint32_t tail_len = quote.frontier - k * pc::kBlockTokens;
        std::optional<DeviceKVPageReservation> restored;
        std::uint32_t host_only = 0;
        for (const pc::NodeRef node : quote.path) { host_only += index.node(node).device != pc::CopyState::Resident ? 1U : 0U; }
        if (host_only != 0) {
            restored = pool_->reserve(host_only); // make_room left them free
            if (!restored) { throw std::logic_error("Inkling 2: the KV pool has no room for a restored path"); }
        }
        prefix_->open_restore(device_.stream);
        try {
            for (const pc::NodeRef node : quote.path) {
                if (index.node(node).device != pc::CopyState::Resident) {
                    prefix_->restore_block(node, pool_->materialize_one(*restored));
                }
            }
            if (tail_len != 0) {
                kv.pages[k] = pool_->materialize_one(kv.reservation);
                prefix_->restore_tail_page(quote.snapshot, kv.pages[k].handle());
            }
            prefix_->restore_image(quote.snapshot, idx);
            lane.prefix.restore = prefix_->submit_restore();
        } catch (...) {
            prefix_->fail_restore();
            throw;
        }
        // The row: the path's cache pages, then the tail's own page.
        std::vector<DeviceKVPageHandle> row;
        row.reserve(k + 1U);
        for (const pc::NodeRef node : quote.path) { row.push_back(prefix_->page(index.node(node).device_id)); }
        if (tail_len != 0) { row.push_back(kv.pages[k].handle()); }
        if (!row.empty()) {
            tables_->publish(kv_rows_[idx].handle(), 0, std::span<const DeviceKVPageHandle>(row), device_.stream);
        }
        kv.mapped           = k + (tail_len != 0 ? 1U : 0U);
        lane.prefix.path    = quote.path;
        lane.prefix.base    = quote.frontier;
        lane.processed      = quote.frontier;
        index.note_hit(quote.snapshot);
        ++prefix_->counters().admissions;
        prefix_->counters().reused_tokens += quote.frontier;
    }

    // The lane's taps (tap_planner.h): past its base, outside its images, away from the snapshots on its path.
    void plan_lane_taps(std::uint32_t idx, const BasePlanImpl& b) {
        Lane& lane = lanes_[idx];
        namespace pc = runtime::prefix_cache;
        const auto n = static_cast<std::uint32_t>(b.tokens.size());
        const pc::MatchResult match = prefix_->index().match(b.tokens, b.block_hashes, b.block_extras, n);
        lane.prefix.frontiers.clear();
        for (const pc::MatchCandidate& c : match.candidates) { lane.prefix.frontiers.push_back(c.frontier); }
        const auto spans = exclusions(b);
        lane.prefix.taps = pc::plan_taps(n, lane.prefix.base, b.tap_hints, lane.prefix.frontiers, spans, prefix_->taps());
    }

    ContextProgress poll_context() {
        if (pause_) {
            ContextProgress out;
            out.kind      = ContextOperationKind::Pause;
            out.advanced  = true;
            out.complete  = true;
            out.published = true;
            out.paused.emplace(std::move(*pause_));
            pause_.reset();
            return out;
        }
        if (!binding_) { throw std::logic_error("Inkling 2: no binding is open"); }
        const std::uint32_t index = *binding_;
        binding_.reset();
        ContextProgress out;
        out.kind      = ContextOperationKind::Bind;
        out.advanced  = true;
        out.complete  = true;
        out.published = true; // the sequence is live (the Engine cancels a binding that publishes none)
        out.replaying = lanes_[index].replay;
        out.sequence  = handle(index);
        return out;
    }

    [[nodiscard]] bool has_context_transaction() const noexcept { return binding_.has_value() || pause_.has_value(); }

    // ---------------------------------------------------------------- pause and replay
    // Pauses a resident at its committed boundary. Its prompt-computed blocks are the cache's as at any release; past the
    // prompt's full blocks its blocks, and its state at the frontier, go in under the request's private key
    // (pause_keys), so only its resume finds them: decode-computed state is never another request's prefix (a resumed
    // request's output equals a cold one's). The lane is then released and the Engine receives the ResumeState.
    bool start_pause(SequenceHandle sequence) {
        if (binding_ || pause_ || pending_) { return false; }
        const std::uint32_t index = lane_of(sequence);
        Lane& lane                = lanes_[index];
        {
            const auto wait = Clock::now();
            CUDA_CHECK(cudaStreamSynchronize(device_.stream));
            wait_ns_ += elapsed_ns(wait);
        }
        settle_prefill_events(lane, true); // synchronized above: the copy below holds no events
        auto saved       = std::make_unique<ResumeStateImpl>();
        saved->target    = lane.replay ? lane.replay_target : lane.processed;
        saved->generated = lane.replay || lane.processed >= lane.prompt_tokens;
        // The lane's state at its frontier (a replay short of its target publishes what it has rebuilt).
        saved->capture = pause_publish(index);
        // What the resume binds from: the request, without its device references.
        saved->lane                = lane;
        saved->lane.bound          = false;
        saved->lane.replay         = false;
        saved->lane.processed      = 0;
        saved->lane.prefix.path    = {};
        saved->lane.prefix.taps    = {};
        saved->lane.prefix.restore.reset();
        saved->lane.prefix.capture = 0;
        saved->lane.constraint_round.clear();
        release(index, true);
        const std::uint32_t frontier = saved->target;
        pause_.emplace(ContractAccess::resume(std::move(saved), frontier));
        return true;
    }

    // The keys of the lane's ledger: its prompt's full blocks as the frontend keyed them (extras zero without images),
    // then every further full block under `salt`, a key of the prompt's length alone. Those blocks hold decode-route
    // state (from the prompt's end on), which a request whose prompt covers them computes on the prompt route: only a
    // resume asks for them (a fresh prompt's keys carry no salt). Their bits are those of any request with the same
    // ledger and prompt length (decode is lane- and width-invariant), so the key need not name the request.
    void pause_keys(Lane& lane) {
        namespace pc    = runtime::prefix_cache;
        Lane::Prefix& p = lane.prefix;
        // Never 0 (no key yet) and never an image key: the high bit set, the prompt's length mixed in below it.
        p.salt = (std::uint64_t{1} << 63U) | ((0x6465636f64650000ULL ^ (std::uint64_t{lane.prompt_tokens} * 0x9E3779B97F4A7C15ULL)) >> 1U);
        p.extras.resize(p.hashes.size(), 0U);
        const std::size_t blocks = lane.tokens.size() / pc::kBlockTokens;
        while (p.hashes.size() < blocks) {
            const std::size_t bi           = p.hashes.size();
            const std::uint64_t parent     = bi == 0 ? pc::kRootLookupHash : p.hashes[bi - 1];
            const std::span<const TokenId> tokens(lane.tokens.data() + bi * pc::kBlockTokens, pc::kBlockTokens);
            p.hashes.push_back(pc::block_lookup_hash(parent, tokens, p.salt));
            p.extras.push_back(p.salt);
        }
    }

    // A pause's publication: the lane's full blocks through its frontier F (pause_keys) and its endpoint at F when the
    // snapshot's key is private or prompt-computed (a frontier past the prompt inside the prompt's last block would be
    // keyed by the shared anchor: no endpoint then, and the resume replays from the deepest snapshot). The capture id,
    // or 0.
    std::uint64_t pause_publish(std::uint32_t idx) {
        Lane& lane = lanes_[idx];
        namespace pc     = runtime::prefix_cache;
        Lane::Prefix& p  = lane.prefix;
        const std::uint32_t F = lane.processed;
        if (!prefix_ || !p.reuse || prefix_failed_ || F == 0) { return 0; }
        pause_keys(lane);
        publish_blocks(idx, F);
        if (!p.reuse) { return 0; } // bf16p: a row with no packed form
        const std::uint32_t k = F / pc::kBlockTokens;
        const bool shared     = F <= lane.prompt_tokens;
        const bool keyed      = shared || k > lane.prompt_tokens / pc::kBlockTokens;
        if (!keyed || p.path.size() < k || std::find(p.frontiers.begin(), p.frontiers.end(), F) != p.frontiers.end()) {
            return 0;
        }
        const std::size_t before = p.captures.size();
        capture_tap(idx, F, pc::SnapshotKind::Endpoint, true);
        return p.captures.size() != before ? p.captures.back() : 0U;
    }

    // The tokens a pause now would recompute on its resume.
    [[nodiscard]] std::uint32_t pause_replay_tokens(SequenceHandle sequence) const noexcept {
        if (!valid(sequence)) { return 0; }
        const Lane& lane           = lanes_[ContractAccess::lane(sequence)];
        const std::uint32_t target = lane.replay ? lane.replay_target : lane.processed;
        if (!prefix_ || !lane.prefix.reuse || prefix_failed_) { return target; }
        namespace pc          = runtime::prefix_cache;
        const std::uint32_t F = lane.processed;
        const bool keyed = F <= lane.prompt_tokens || F / pc::kBlockTokens > lane.prompt_tokens / pc::kBlockTokens;
        std::uint32_t cached = lane.prefix.base;
        for (const std::uint32_t f : lane.prefix.frontiers) {
            if (f <= F) { cached = std::max(cached, f); }
        }
        if (F != 0 && keyed) { cached = F; }
        return target > cached ? target - cached : 0U;
    }

    // One call of the resumed lane's ledger toward its paused frontier, without sampling (see Program::advance_replay).
    ReplayProgress advance_replay(SequenceHandle sequence) {
        const auto start = Clock::now();
        if (pending_) { throw std::logic_error("Inkling 2: a pending batch awaits its commit"); }
        const std::uint32_t index = lane_of(sequence);
        Lane& lane                = lanes_[index];
        if (!lane.replay || lane.processed >= lane.replay_target) {
            throw std::logic_error("Inkling 2: replay on a lane that is not replaying");
        }
        const bool prompt = lane.processed < lane.prompt_tokens;
        const std::uint32_t end =
            prompt ? std::min({lane.prompt_tokens, lane.processed + o_.prefill_chunk, lane.replay_target})
                   : std::min(lane.replay_target,
                              lane.processed + static_cast<std::uint32_t>(ops::inkling2::kMaxAttentionLaneColumns));
        const std::uint32_t width = end - lane.processed;
        CallPlan call;
        call.first.assign(index + 1, 0);
        call.prompt = prompt;
        for (std::uint32_t p = lane.processed; p < end; ++p) {
            call.tokens.push_back(lane.tokens[p]);
            call.lane.push_back(static_cast<std::int32_t>(index));
            call.position.push_back(static_cast<std::int32_t>(p));
        }
        if (lossless_) {
            run_lossless(index, call, false);
        } else {
            upload(call);
            if (prompt) {
                const std::uint32_t at = lane.processed;
                run_forward(static_cast<std::int32_t>(width) - 1, 0, [&] { overlay_images(lane.images, at, width); });
            } else {
                run_forward(static_cast<std::int32_t>(width) - 1, 0);
            }
            if (drafter_) { drafter_->append(0, static_cast<std::int32_t>(width), device_.stream); } // its context rows
        }
        lane.processed = end;
        publish_prefill(index);
        ReplayProgress out;
        out.processed_tokens = width;
        out.complete         = end == lane.replay_target;
        if (out.complete) { lane.replay = false; }
        out.timing = timing_since(start); // the Engine counts a replay's time and tokens apart from the prefill's
        return out;
    }

    // ---------------------------------------------------------------- prefill timing
    // A request's prefill time is its prompt calls' time on the device: each call is bracketed by a pair of timing events
    // on the compute stream, where the lanes' calls run one at a time. Host time misses it: a prompt call returns once
    // enqueued, and the rounds of other lanes that wait on the stream absorb the chunk's work (2026-11-25: 8,251 prompt
    // tokens in chunks of 4,096, 4,096 and 59 beside decoding lanes timed 99 ms, the 59-token call's wait, for 84K tok/s).
    std::array<cudaEvent_t, 2> take_timing_pair() {
        std::array<cudaEvent_t, 2> pair{};
        for (cudaEvent_t& event : pair) {
            if (!timing_events_.empty()) {
                event = timing_events_.back();
                timing_events_.pop_back();
            } else {
                CUDA_CHECK(cudaEventCreate(&event));
            }
        }
        return pair;
    }

    // Adds the durations of the lane's calls whose end has been reached to its device prefill time and returns their
    // events to the pool; `all`: every call (the caller synchronized the stream).
    void settle_prefill_events(Lane& lane, bool all) {
        std::size_t kept = 0;
        for (std::size_t i = 0; i < lane.prefill_events.size(); ++i) {
            const std::array<cudaEvent_t, 2>& pair = lane.prefill_events[i];
            if (!all && cudaEventQuery(pair[1]) == cudaErrorNotReady) {
                lane.prefill_events[kept++] = pair;
                continue;
            }
            float ms = 0.0F;
            if (cudaEventElapsedTime(&ms, pair[0], pair[1]) == cudaSuccess && ms > 0.0F) {
                lane.prefill_device_ns += static_cast<std::uint64_t>(static_cast<double>(ms) * 1.0e6);
            }
            (void)cudaGetLastError(); // an unrecorded pair (a failed call) reports an error that is not sticky
            timing_events_.push_back(pair[0]);
            timing_events_.push_back(pair[1]);
        }
        lane.prefill_events.resize(kept);
    }

    // A released lane's pairs go back unread (re-recording an event is allowed while its last record is pending).
    void drop_prefill_events(Lane& lane) noexcept {
        for (const std::array<cudaEvent_t, 2>& pair : lane.prefill_events) {
            try {
                timing_events_.push_back(pair[0]);
                timing_events_.push_back(pair[1]);
            } catch (...) {}
        }
        lane.prefill_events.clear();
    }

    // ---------------------------------------------------------------- execution
    PrefillProgress advance_prefill(SequenceHandle sequence, runtime::TokenMaskProvider* masks) {
        const auto start = Clock::now();
        if (pending_) { throw std::logic_error("Inkling 2: a pending batch awaits its commit"); }
        const std::uint32_t index = lane_of(sequence);
        Lane& lane                = lanes_[index];
        if (lane.processed >= lane.prompt_tokens) { throw std::logic_error("Inkling 2: the prompt is already prefilled"); }
        std::uint32_t chunk = std::min<std::uint32_t>(o_.prefill_chunk, lane.prompt_tokens - lane.processed);
        // An exact tap the image after this call would not reach ends the call there (every chunk gives the same bits).
        for (const runtime::prefix_cache::PlannedTap& tap : lane.prefix.taps) {
            if (tap.placement != runtime::prefix_cache::TapPlacement::Exact || tap.position <= lane.processed) { continue; }
            if (tap.position + capture_reach() < lane.processed + chunk) { chunk = tap.position - lane.processed; }
            break;
        }
        const bool last = lane.processed + chunk == lane.prompt_tokens;
        CallPlan call;
        // first: the sequence's first position (every lane's sequence starts at 0), not the call's first column.
        call.first.assign(index + 1, 0);
        call.prompt = true;
        for (std::uint32_t i = 0; i < chunk; ++i) {
            call.tokens.push_back(lane.tokens[lane.processed + i]);
            call.lane.push_back(static_cast<std::int32_t>(index));
            call.position.push_back(static_cast<std::int32_t>(lane.processed + i));
        }
        // The call is timed on the device (prefill_events) unless it only rebuilds positions the Engine was already told
        // of (a prompt resumed below its paused frontier), as its tokens are not counted again.
        settle_prefill_events(lane, false);
        const bool timed = lane.processed + chunk > lane.counted;
        // Held by the lane at once, so a call that throws returns them with it (an unrecorded end reads no time).
        std::array<cudaEvent_t, 2> events{};
        if (timed) {
            events = take_timing_pair();
            lane.prefill_events.push_back(events);
        }
        if (lossless_) {
            if (timed) { CUDA_CHECK(cudaEventRecord(events[0], device_.stream)); }
            run_lossless(index, call, last);
        } else {
            upload(call);
            if (timed) { CUDA_CHECK(cudaEventRecord(events[0], device_.stream)); }
            run_forward(static_cast<std::int32_t>(chunk) - 1, last ? 1 : 0,
                        [&] { overlay_images(lane.images, lane.processed, chunk); });
            if (drafter_) { drafter_->append(0, static_cast<std::int32_t>(chunk), device_.stream); } // its context rows
        }
        if (timed) { CUDA_CHECK(cudaEventRecord(events[1], device_.stream)); }
        lane.processed += chunk;
        const std::size_t captured = lane.prefix.captures.size();
        publish_prefill(index);
        PrefillProgress out;
        out.snapshot_published = lane.prefix.captures.size() != captured;
        // The admission's terms: the tokens the first binding restored from a snapshot (none at a root start).
        out.summary = {.prompt_tokens        = lane.prompt_tokens,
                       .reused_prompt_tokens = lane.reused_prompt,
                       .prefix_reuse_path = lane.reused_prompt != 0 ? PrefixReusePath::HybridSnapshot : PrefixReusePath::Root};
        // The prompt positions past those the Engine was told of (the Engine sums them): a prompt resumed below its
        // paused frontier reports nothing until it passes it again.
        out.processed_prompt_tokens = lane.processed > lane.counted ? lane.processed - lane.counted : 0U;
        lane.counted                = std::max(lane.counted, lane.processed);
        out.complete                = last;
        if (last) {
            const std::uint64_t transaction = ++transaction_;
            const SequenceHandle row        = handle(index);
            // The head ran over the chunk's last column only: its logits are row 0. A readout reads them before a
            // token constraint masks them in place for the sampler.
            if (!lane.options.readout_tokens.empty()) { out.readout = readout(lane, 0); }
            sample({&row, 1}, {&index, 1}, 0, ops::kSamplePurposePrefill, masks);
            settle_prefill_events(lane, true); // the sample synchronized the stream
            out.timing = timing_since(start);
            out.pending.emplace(ContractAccess::batch(this, transaction, {&row, 1}, {sampled_tokens_.data(), 1},
                                                      out.timing, {failures_.data(), 1}));
            pending_      = transaction;
            pending_kind_ = PendingKind::Prefill;
        } else {
            out.timing = timing_since(start);
        }
        lane.prefill_ns += elapsed_ns(start);
        return out;
    }

    PendingBatch decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
                        runtime::TokenMaskProvider* masks) {
        const auto start = Clock::now();
        if (pending_) { throw std::logic_error("Inkling 2: a pending batch awaits its commit"); }
        if (sequences.empty() || sequences.size() > lanes_.size()) { throw std::invalid_argument("Inkling 2: a decode of 1-8 rows"); }
        round_captured_ = false; // instantiate_graph and rows_graph's evictions set it
        // A row's round licenses at most its remaining generated tokens: a DSpark round needs two of them.
        const std::uint32_t remaining = budgets.empty() ? 0U : budgets[0].generated_tokens_remaining;
        if (remaining >= 2 && dspark_round(sequences)) { return decode_dspark(sequences[0], remaining, masks, start); }
        if (lossless_) { return decode_lossless(sequences, masks, start); }
        // Several rows draft together (decode_dspark_rows), unless the round's width verifies no draft: it then
        // decodes plainly below.
        bool drafted = false;
        if (dspark_rows_round(sequences, budgets)) {
            if (std::optional<PendingBatch> batch = decode_dspark_rows(sequences, budgets, masks, start)) {
                return std::move(*batch);
            }
            drafted = true;
        }
        if (drafter_ && (sequences.size() == 1 || drafted)) { // a plain round of drafting rows
            for (const SequenceHandle& s : sequences) { ++lanes_[lane_of(s)].fallback; }
        }
        std::vector<std::uint32_t> rows;
        for (const SequenceHandle& s : sequences) { rows.push_back(lane_of(s)); }
        // A lane's columns are consecutive and lanes ascend in the call; rows keep the Engine's order.
        std::vector<std::uint32_t> order = rows;
        std::sort(order.begin(), order.end());
        CallPlan call;
        call.first.assign(order.back() + 1, 0);
        bool uniform = true;
        std::map<std::uint32_t, std::int32_t> last_column;
        for (const std::uint32_t index : order) {
            Lane& lane = lanes_[index];
            if (lane.processed >= lane.tokens.size()) { throw std::logic_error("Inkling 2: a decode row has no input"); }
            const auto inputs = static_cast<std::uint32_t>(lane.tokens.size() - lane.processed);
            if (inputs > static_cast<std::uint32_t>(ops::inkling2::kMaxAttentionLaneColumns)) {
                throw std::logic_error("Inkling 2: more than 16 unprocessed tokens in a lane");
            }
            uniform = uniform && inputs == 1;
            for (std::uint32_t i = 0; i < inputs; ++i) {
                call.tokens.push_back(lane.tokens[lane.processed + i]);
                call.lane.push_back(static_cast<std::int32_t>(index));
                call.position.push_back(static_cast<std::int32_t>(lane.processed + i));
            }
            last_column[index] = static_cast<std::int32_t>(call.tokens.size()) - 1;
        }
        upload(call);
        const auto columns = static_cast<std::int32_t>(call.tokens.size());
        std::vector<std::int32_t> logit_rows;
        for (const std::uint32_t index : rows) { logit_rows.push_back(last_column[index]); }
        // Rows sample in the Engine's order; their logits rows in the call's column order.
        const std::uint64_t transaction = ++transaction_;
        // A plain round of rows that could draft together measures the cost line rounds of drafts are weighed against.
        const bool timed = rows.size() >= 2 && rows_eligible(sequences, budgets);
        float device_ms  = 0.0F;
        if (uniform && o_.use_cuda_graph) {
            decode_graph(static_cast<std::int32_t>(order.back()) + 1, columns, rows, logit_rows, masks,
                         timed ? &device_ms : nullptr);
        } else {
            run_decode_call(columns);
            sample(sequences, rows, logit_rows, ops::kSamplePurposeDecode, masks);
        }
        for (const std::uint32_t index : order) {
            Lane& lane     = lanes_[index];
            lane.processed = static_cast<std::uint32_t>(lane.tokens.size());
        }
        const std::uint64_t ns = elapsed_ns(start);
        if (timed && !round_captured_) {
            RowsCost& cost = rows_cost(sequences);
            if (drafted) {
                // A drafted round whose width verified nothing: its drafts were spent for the rows' plain tokens.
                ewma(cost.draft_wall, static_cast<double>(ns) * 1e-6);
                ewma(cost.draft_tokens, static_cast<double>(rows.size()));
            } else if (device_ms > 0.0F) {
                ewma(cost.plain_ms, device_ms);
                ewma(cost.plain_wall, static_cast<double>(ns) * 1e-6);
            }
        }
        for (const std::uint32_t index : rows) {
            lanes_[index].decode_ns += ns;
            lanes_[index].decode_share_ns += ns / rows.size();
        }
        pending_      = transaction;
        pending_kind_ = PendingKind::Decode;
        return ContractAccess::batch(this, transaction, sequences, {sampled_tokens_.data(), sequences.size()},
                                     timing_since(start), {failures_.data(), sequences.size()});
    }

    runtime::ExecutionTiming append_forced(std::span<const SequenceHandle> sequences, std::span<const TokenId> tokens,
                                           std::uint32_t stride) {
        const auto start = Clock::now();
        if (pending_) { throw std::logic_error("Inkling 2: a pending batch awaits its commit"); }
        if (stride == 0 || tokens.size() != sequences.size() * stride) {
            throw std::invalid_argument("Inkling 2: forced tokens are row-major, `stride` a row");
        }
        for (std::size_t row = 0; row < sequences.size(); ++row) {
            Lane& lane = lanes_[lane_of(sequences[row])];
            for (std::uint32_t i = 0; i < stride; ++i) {
                const TokenId id = tokens[row * stride + i];
                if (id < 0 || id >= vocab_) { throw std::invalid_argument("Inkling 2: a forced token is outside the vocabulary"); }
                lane.forced.push_back(static_cast<std::uint32_t>(lane.tokens.size()));
                lane.tokens.push_back(id); // processed by the lane's next decode call, with its pending input
            }
        }
        return timing_since(start);
    }

    CommitResult commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions) {
        const auto start = Clock::now();
        if (ContractAccess::owner(pending) != this || !pending_ || ContractAccess::transaction(pending) != *pending_) {
            throw std::logic_error("Inkling 2: commit of a foreign or stale pending batch");
        }
        const auto rows = ContractAccess::rows(pending);
        if (decisions.size() != rows.size()) { throw std::logic_error("Inkling 2: commit decisions are not row aligned"); }
        CommitResult out;
        out.row_count = rows.size();
        for (std::size_t row = 0; row < rows.size(); ++row) {
            const std::uint32_t index = ContractAccess::lane(rows[row]);
            Lane& lane                = lanes_[index];
            const auto& d             = decisions[row];
            if ((d.cancelled && d.failed) || (d.failed && (d.accepted_tokens != 0 || !d.terminal)) ||
                (!d.cancelled && d.failed != (failures_[row] != runtime::PendingRowFailure::None))) {
                throw std::logic_error("Inkling 2: a commit decision is invalid");
            }
            if (d.cancelled || d.failed) {
                lane.constraint_round.clear();
                out.rows[row].disposition = d.failed ? runtime::CommitDisposition::FailedReleased
                                                     : runtime::CommitDisposition::CancelledReleased;
                continue;
            }
            if (pending_draft_.active) {
                // A DSpark row: its accepted prefix; the lane holds positions F .. F + accepted - 1, its last token
                // pending.
                if (d.accepted_tokens == 0 || d.accepted_tokens > static_cast<std::uint32_t>(row_counts_[row])) {
                    throw std::logic_error("Inkling 2: a DSpark row commits 1 .. its licensed tokens");
                }
                for (std::uint32_t i = 0; i < d.accepted_tokens; ++i) {
                    lane.tokens.push_back(round_tokens_[row * kRowStride + i]);
                }
                lane.processed = pending_firsts_[row] + d.accepted_tokens;
            } else {
                if (d.accepted_tokens != 1) { throw std::logic_error("Inkling 2: a row commits exactly one token"); }
                lane.tokens.push_back(sampled_tokens_[row]);
                // A token-constrained row's draw is the request's with its token.
                for (ConstrainedDraw& draw : lane.constraint_round) { lane.constraint_trace.push_back(std::move(draw)); }
                lane.constraint_round.clear();
            }
            out.rows[row].disposition =
                d.terminal ? runtime::CommitDisposition::Finishable : runtime::CommitDisposition::Active;
            speculative_of(lane, out.rows[row]);
        }
        pending_draft_ = {};
        const std::uint64_t ns = elapsed_ns(start);
        for (std::size_t row = 0; row < rows.size(); ++row) {
            Lane& lane = lanes_[ContractAccess::lane(rows[row])];
            if (pending_kind_ == PendingKind::Prefill) {
                lane.prefill_ns += ns;
            } else {
                lane.decode_ns += ns;
                lane.decode_share_ns += ns / rows.size();
            }
        }
        for (std::size_t row = 0; row < rows.size(); ++row) {
            if (decisions[row].cancelled || decisions[row].failed) {
                const std::uint32_t index = ContractAccess::lane(rows[row]);
                out.rows[row].timings     = timings(lanes_[index]);
                release(index);
            }
        }
        out.timing = timing_since(start);
        ContractAccess::consume(pending);
        pending_.reset();
        return out;
    }

    DiscardResult discard(PendingBatch&& pending) noexcept {
        DiscardResult out;
        if (ContractAccess::owner(pending) != this || !pending_ || ContractAccess::transaction(pending) != *pending_) {
            return out;
        }
        out.row_count = ContractAccess::rows(pending).size();
        out.status    = runtime::ConsumeStatus::Consumed;
        for (const SequenceHandle& row : ContractAccess::rows(pending)) {
            const std::uint32_t index = ContractAccess::lane(row);
            if (index < lanes_.size()) { lanes_[index].constraint_round.clear(); }
        }
        ContractAccess::consume(pending);
        pending_.reset();
        pending_draft_ = {};
        return out;
    }

    FinishResult finish(SequenceHandle sequence) noexcept {
        FinishResult out;
        if (!valid(sequence)) { return out; }
        const std::uint32_t index = ContractAccess::lane(sequence);
        out.timings               = timings(lanes_[index]);
        out.status                = runtime::ConsumeStatus::Consumed;
        out.constrained_draws     = std::move(lanes_[index].constraint_trace);
        CommitRowResult stats;
        speculative_of(lanes_[index], stats);
        out.speculative = std::move(stats.speculative);
        report_rounds(lanes_[index]);
        // The prompt tokens past the reused prefix that this request's accepted captures made reusable for a later
        // request (Anthropic's cache_creation_input_tokens), as the other families report them.
        if (const Lane& lane = lanes_[index]; prefix_ && lane.prefix.reuse) {
            const std::uint32_t deepest =
                std::min<std::uint32_t>(std::max(lane.reused_prompt, lane.prefix.deepest_capture),
                                        static_cast<std::uint32_t>(lane.prompt_tokens));
            out.cache_created_prompt_tokens = deepest > lane.reused_prompt ? deepest - lane.reused_prompt : 0U;
        }
        release(index);
        return out;
    }

    AbortResult abort(SequenceHandle sequence) noexcept {
        AbortResult out;
        if (!valid(sequence)) { return out; }
        const std::uint32_t index = ContractAccess::lane(sequence);
        out.timings               = timings(lanes_[index]);
        out.status                = runtime::ConsumeStatus::Consumed;
        CommitRowResult stats;
        speculative_of(lanes_[index], stats);
        out.speculative = std::move(stats.speculative);
        release(index);
        return out;
    }

    void release_all() noexcept {
        for (std::uint32_t i = 0; i < lanes_.size(); ++i) { release(i); }
        binding_.reset();
        pending_.reset();
    }

    bool recover() noexcept {
        try {
            release_all();
            CUDA_CHECK(cudaStreamSynchronize(device_.stream));
            return true;
        } catch (...) {
            return false;
        }
    }

    [[nodiscard]] bool context_blocks(SequenceHandle sequence) const noexcept {
        return binding_.has_value() && valid(sequence) && *binding_ == ContractAccess::lane(sequence);
    }

    // ---------------------------------------------------------------- scoring
    ScoreResult causal_score(PreparedPrompt&& prompt, std::uint32_t first_target, const ScoreOptions& options) {
        if (!o_.causal_scoring) { throw std::logic_error("Inkling 2: causal scoring needs a scoring Program"); }
        if (binding_ || pending_ || std::any_of(lanes_.begin(), lanes_.end(), [](const Lane& l) { return l.bound; })) {
            throw std::logic_error("Inkling 2: causal scoring needs an idle Program");
        }
        std::unique_ptr<PreparedPromptData> data = std::move(prompt).take();
        const std::vector<TokenId>& tokens       = data->token_ids;
        const auto n                             = static_cast<std::uint32_t>(tokens.size());
        if (n < 2 || first_target == 0 || first_target >= n || n > o_.max_context) {
            throw std::invalid_argument("Inkling 2: scoring needs 2..max_context tokens and 0 < first_target < count");
        }
        if (options.candidates.size() != static_cast<std::size_t>(options.candidates_per_position) * (n - first_target)) {
            throw std::invalid_argument("Inkling 2: candidates must hold candidates_per_position for each scored position");
        }
        std::shared_ptr<const std::vector<PromptImage>> images;
        if (!data->images.empty()) {
            check_images(data->images, n);
            images = std::make_shared<const std::vector<PromptImage>>(std::move(data->images));
        }
        ScoreResult out;
        std::vector<float> row(static_cast<std::size_t>(vocab_));
        std::size_t next_logits = 0;
        // Position `position`'s logits (a device row of vocab_ FP32), predicting token position + 1.
        const auto consume = [&](std::uint32_t position, const float* device_row) {
            const std::uint32_t scored = position + 1 - first_target;
            CUDA_CHECK(cudaMemcpy(row.data(), device_row, row.size() * 4, cudaMemcpyDeviceToHost));
            double max = -std::numeric_limits<double>::infinity();
            for (const float v : row) { max = std::max(max, static_cast<double>(v)); }
            double sum = 0;
            for (const float v : row) { sum += std::exp(static_cast<double>(v) - max); }
            const double lse = max + std::log(sum);
            out.logprobs.push_back(static_cast<float>(row[static_cast<std::size_t>(tokens[position + 1])] - lse));
            if (options.top_k != 0) {
                std::vector<std::int32_t> ids(row.size());
                for (std::size_t i = 0; i < ids.size(); ++i) { ids[i] = static_cast<std::int32_t>(i); }
                std::partial_sort(ids.begin(), ids.begin() + options.top_k, ids.end(), [&](std::int32_t a, std::int32_t b) {
                    return row[a] > row[b] || (row[a] == row[b] && a < b);
                });
                for (std::uint32_t k = 0; k < options.top_k; ++k) {
                    out.top_ids.push_back(ids[k]);
                    out.top_logprobs.push_back(static_cast<float>(row[ids[k]] - lse));
                }
            }
            for (std::uint32_t k = 0; k < options.candidates_per_position; ++k) {
                const TokenId id = options.candidates[static_cast<std::size_t>(scored) * options.candidates_per_position + k];
                out.candidate_logprobs.push_back(id >= 0 && id < vocab_ ? static_cast<float>(row[id] - lse)
                                                                        : -std::numeric_limits<float>::infinity());
            }
            if (next_logits < options.logits_positions.size() && options.logits_positions[next_logits] == scored) {
                out.logits.insert(out.logits.end(), row.begin(), row.end());
                ++next_logits;
            }
        };
        if (lossless_) {
            // The lossless route: the window's positions 0 .. n - 2 in one pass, the head a slice of rows at a time.
            if (images) { throw std::invalid_argument("Inkling 2: the lossless route scores text windows"); }
            if (options.argmax_only) { throw std::invalid_argument("Inkling 2: argmax_only is not on the lossless route"); }
            lossless_->run(std::span<const std::int32_t>(tokens.data(), n - 1), device_.stream);
            for (std::uint32_t c = first_target - 1; c < n - 1; c += LosslessForward::kHeadRows) {
                const auto count = static_cast<std::int32_t>(std::min<std::uint32_t>(LosslessForward::kHeadRows, n - 1 - c));
                lossless_->head(static_cast<std::int32_t>(c), count, device_.stream);
                CUDA_CHECK(cudaStreamSynchronize(device_.stream));
                for (std::int32_t i = 0; i < count; ++i) {
                    consume(c + static_cast<std::uint32_t>(i), lossless_->logits() + static_cast<std::size_t>(i) * vocab_);
                }
            }
            return out;
        }
        CUDA_CHECK(cudaMemsetAsync(images_.p, 0, static_cast<std::size_t>(forward_.layout().lane_words) * 2, device_.stream));
        clear_windows(0);
        clear_unpackable(0);
        release_pages(0); // a failed window's pages
        reserve_pages(0, lane_pages()); // lane 0 maps its pages as the window's calls reach them
        // ScoreOptions::prompt_hidden_states: the call holding the last prompt position captures its column.
        if (options.prompt_hidden_states && score_hidden_.p == nullptr) {
            throw std::invalid_argument("Inkling 2: hidden states need a scoring Program off the lossless route");
        }
        bool hidden_captured = false;
        // A failed forward leaves no capture armed for the calls that follow (as Flash-Next's scoring).
        struct Disarm {
            Forward& forward;
            ~Disarm() { forward.clear_hidden_capture(); }
        } disarm{forward_};
        for (std::uint32_t at = 0; at < n - 1;) {
            const std::uint32_t chunk = std::min<std::uint32_t>(o_.prefill_chunk, n - 1 - at);
            CallPlan call;
            call.first  = {0};
            call.prompt = true;
            for (std::uint32_t i = 0; i < chunk; ++i) {
                call.tokens.push_back(tokens[at + i]);
                call.lane.push_back(0);
                call.position.push_back(static_cast<std::int32_t>(at + i));
            }
            upload(call);
            // Positions at + i predict token at + i + 1; scored from first_target - 1 on.
            const std::int32_t from = at + chunk <= first_target - 1
                                          ? static_cast<std::int32_t>(chunk)
                                          : static_cast<std::int32_t>(std::max<std::int64_t>(0, std::int64_t{first_target} - 1 - at));
            const bool capture_hidden = options.prompt_hidden_states && first_target - 1 >= at && first_target - 1 < at + chunk;
            if (capture_hidden) {
                forward_.set_hidden_capture(static_cast<std::uint16_t*>(score_hidden_.p),
                                            static_cast<std::int32_t>(first_target - 1 - at));
            }
            // The final mix; the head runs below a slice of logits rows at a time.
            run_forward(from, 0, [&] { overlay_images(images, at, chunk); });
            if (capture_hidden) {
                forward_.clear_hidden_capture();
                const auto& counts = forward_.hidden_counts();
                if (std::any_of(counts.begin(), counts.end(), [](std::uint8_t c) { return c != 1; })) {
                    throw std::logic_error("Inkling 2: the scored prompt's hidden states were not captured once per layer");
                }
                hidden_captured = true;
            }
            if (options.argmax_only) {
                // Each slice's argmax on the device; the host reads only the ids (out.top_ids never reallocates).
                if (score_ids_.bytes == 0) { score_ids_ = DeviceBuffer(static_cast<std::size_t>(kMaxForwardColumns) * 4); }
                out.top_ids.resize(n - first_target);
                for (std::int32_t c = from; c < static_cast<std::int32_t>(chunk); c += kMaxForwardColumns) {
                    const std::int32_t count = std::min<std::int32_t>(kMaxForwardColumns, static_cast<std::int32_t>(chunk) - c);
                    forward_.head(c, count, device_.stream);
                    Tensor logits(forward_.buffers().logits, DType::FP32, {static_cast<std::int32_t>(vocab_), count});
                    Tensor ids(score_ids_.p, DType::I32, {count});
                    ops::argmax(logits, ids, static_cast<std::int32_t>(vocab_), device_.stream);
                    CUDA_CHECK(cudaMemcpyAsync(out.top_ids.data() + (at + static_cast<std::uint32_t>(c) + 1 - first_target),
                                               score_ids_.p, static_cast<std::size_t>(count) * 4, cudaMemcpyDeviceToHost,
                                               device_.stream));
                }
                at += chunk;
                continue;
            }
            for (std::int32_t c = from; c < static_cast<std::int32_t>(chunk); ++c) {
                const std::int32_t slice = (c - from) % kMaxForwardColumns;
                if (slice == 0) {
                    forward_.head(c, std::min<std::int32_t>(kMaxForwardColumns, static_cast<std::int32_t>(chunk) - c),
                                  device_.stream);
                    CUDA_CHECK(cudaStreamSynchronize(device_.stream));
                }
                consume(at + static_cast<std::uint32_t>(c), forward_.buffers().logits + static_cast<std::size_t>(slice) * vocab_);
            }
            at += chunk;
        }
        CUDA_CHECK(cudaStreamSynchronize(device_.stream));
        release_pages(0);
        // A stalled MoE call (under --a16-activations the prompt route runs the waiting kernels) left wrong expert
        // output in the window: the scores fail, as a sampled call does (finish_sample).
        check_moe_error();
        if (options.prompt_hidden_states) {
            if (!hidden_captured) { throw std::logic_error("Inkling 2: the scored prompt's last position was not captured"); }
            // [layers][H], the BF16 values widened exactly.
            const std::size_t values = static_cast<std::size_t>(c_.num_hidden_layers) * c_.hidden_size;
            std::vector<std::uint16_t> raw(values);
            CUDA_CHECK(cudaMemcpy(raw.data(), score_hidden_.p, values * 2, cudaMemcpyDeviceToHost));
            out.hidden_states.resize(values);
            out.hidden_state_layers = c_.num_hidden_layers;
            for (std::size_t i = 0; i < values; ++i) {
                const std::uint32_t bits = std::uint32_t{raw[i]} << 16U;
                std::memcpy(&out.hidden_states[i], &bits, sizeof(bits));
            }
        }
        if (unpackable_.p != nullptr) {
            CUDA_CHECK(cudaMemcpy(unpackable_seen_.data(), unpackable_.p, 4, cudaMemcpyDeviceToHost));
            if (unpackable_seen_[0] != kNoPosition) {
                clear_unpackable(0);
                throw std::runtime_error("Inkling 2: --kv-dtype bf16p: the prompt has a K or V row with no packed form "
                                         "(more than 14 exponents outside its window), so its scores would not be bf16's");
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- diagnostics
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept {
        PhysicalUsageSnapshot out;
        for (const Lane& lane : lanes_) {
            if (lane.bound) { ++out.occupied.state_slots; }
        }
        out.occupied.main_kv_pages = request_pages();
        out.capacity.state_slots   = static_cast<std::uint32_t>(lanes_.size());
        out.capacity.main_kv_pages = plan_.kv_pages - 1; // the scratch page holds nothing
        return out;
    }

    [[nodiscard]] MemoryUsage memory_usage() const noexcept {
        MemoryUsage out;
        out.device.weights   = p_.model.device_weight_bytes();
        // As the Qwen3.5 family's: the pool less the pages only the prefix cache holds (its evictable blocks, `prefix`),
        // and of the rest the pages requests hold (every allocated page but the scratch page and those blocks: a lane's
        // own, and the cached blocks its row maps).
        const std::uint64_t cached = cache_only_pages();
        out.device.prefix    = cached * plan_.page_bytes;
        out.device.kv        = plan_.kv > out.device.prefix ? plan_.kv - out.device.prefix : 0U;
        out.device.kv_used   = std::uint64_t{request_pages()} * plan_.page_bytes;
        out.device.state     = plan_.state + plan_.io;
        out.device.workspace = plan_.workspace;
        out.device.adapter   = p_.adapter_bytes(); // --lora's factors and --ablate's direction (0 without them)
        out.device.cuda_graphs = graph_bytes_.load(std::memory_order_relaxed);
        // The plan's untaken share less what the graphs instantiated since took from it.
        const std::uint64_t since = out.device.cuda_graphs > reserved_graphs_ ? out.device.cuda_graphs - reserved_graphs_ : 0U;
        out.device.reserved       = reserved_bytes_ > since ? reserved_bytes_ - since : 0U;
        out.host.staging   = host_.size();
        return out;
    }

    [[nodiscard]] MemorySummary memory_summary() const noexcept {
        MemorySummary out;
        out.max_context      = o_.max_context;
        out.kv_capacity      = (plan_.kv_pages - 1U) * static_cast<std::uint32_t>(ops::inkling2::kKeyUnit); // the pool
                                                                                                          // the lanes share
        out.kv_cache         = kv_storage(o_.kv_codec);
        out.kv_payload_bytes = plan_.kv;
        out.cuda_graph_allowance_bytes = plan_.graphs;
        out.cuda_graph_measured_bytes  = graph_bytes_.load(std::memory_order_relaxed);
        out.adapter_bytes    = p_.adapter_bytes();
        return out;
    }

    const ProgramDevicePlan& device_plan() const noexcept { return plan_; }

private:
    // A DSpark round awaiting its commit: each row's anchor position (pending_firsts_), tokens (round_tokens_, a row
    // each kRowStride) and the count it licenses (row_counts_).
    struct PendingDraft {
        bool active = false;
    };
    static constexpr std::uint32_t kRowStride = ops::inkling2::kDSparkBlock;
    enum class PendingKind : std::uint8_t { Prefill, Decode };

    SequenceHandle handle(std::uint32_t index) const {
        return ContractAccess::handle(this, index, lanes_[index].epoch);
    }
    [[nodiscard]] bool valid(SequenceHandle s) const noexcept {
        const std::uint32_t index = ContractAccess::lane(s);
        return ContractAccess::owner(s) == this && index < lanes_.size() && lanes_[index].bound &&
               lanes_[index].epoch == ContractAccess::epoch(s);
    }
    std::uint32_t lane_of(SequenceHandle s) const {
        if (!valid(s)) { throw std::logic_error("Inkling 2: a stale or foreign sequence"); }
        return ContractAccess::lane(s);
    }
    std::int32_t* lane_counts(std::uint32_t index) {
        return static_cast<std::int32_t*>(counts_.p) + static_cast<std::size_t>(index) * vocab_;
    }
    // `pausing`: the request's captures stay known (its ResumeState keeps their ids for a later withdrawal).
    void release(std::uint32_t index, bool pausing = false) noexcept {
        if (index < lanes_.size()) {
            release_prefix(index, pausing);
            release_pages(index);
            drop_prefill_events(lanes_[index]);
            lanes_[index] = Lane{};
            clear_failed_prefix();
        }
    }
    // bf16p: forgets a lane's rows with no packed form (a new request, or rejected draft rows the next calls overwrite).
    // A fresh lane's quantized windows (BF16 windows are zeroed with its image).
    void clear_windows(std::size_t index) {
        for (const DeviceBuffer* window : {&window_codes_, &window_side_}) {
            if (window->bytes == 0) { continue; }
            const std::size_t bytes = window->bytes / lanes_.size();
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(window->p) + index * bytes, 0, bytes, device_.stream));
        }
    }
    void clear_unpackable(std::uint32_t index) {
        if (unpackable_.p != nullptr) {
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::uint32_t*>(unpackable_.p) + index, 0xFF, 4, device_.stream));
            unpackable_seen_[index] = kNoPosition;
        }
    }
    static GenerationTimings timings(const Lane& lane) {
        GenerationTimings out;
        // The prompt calls' device time (settle_prefill_events); host time only for a request with none timed (one
        // resumed past its prompt, whose first binding's time the ResumeState kept).
        out.prefill_seconds = static_cast<double>(lane.prefill_device_ns != 0 ? lane.prefill_device_ns : lane.prefill_ns) * 1e-9;
        out.decode_seconds       = static_cast<double>(lane.decode_ns) * 1e-9;
        out.decode_share_seconds = static_cast<double>(lane.decode_share_ns) * 1e-9;
        return out;
    }
    // The unit's time since `start`: the device waits it made (wait_ns_, taken) and the rest as host time.
    runtime::ExecutionTiming timing_since(Clock::time_point start) {
        runtime::ExecutionTiming t;
        const std::uint64_t total = elapsed_ns(start);
        t.device_wait_ns          = std::min(wait_ns_, total);
        t.submit_host_ns          = total - t.device_wait_ns;
        wait_ns_                  = 0;
        return t;
    }

    // The call plan through the next staging slot: a slot is rewritten only once the copies that read it ran (calls
    // queue without a host wait between them). `map` (every call but a graph's capture, whose lanes are free) first maps
    // the pages the call's columns write (map_call_pages).
    void upload(const CallPlan& call, bool map = true) {
        if (map) {
            map_call_pages(call);
            if (prefix_) { order_after_cache(call); }
        }
        slot_ = (slot_ + 1) % kStagingSlots;
        if (slot_used_[slot_]) { CUDA_CHECK(cudaEventSynchronize(slot_events_[slot_])); }
        forward_.upload(call, staging_ + slot_ * forward_.upload_words(), device_.stream);
        CUDA_CHECK(cudaEventRecord(slot_events_[slot_], device_.stream));
        slot_used_[slot_] = true;
    }

    // ---------------------------------------------------------------- KV pages
    // The full visits' pages (core/paged_kv_cache.h), one pool the lanes share: a lane reserves the pages of its next unit
    // (a binding its first, reserve_units each one after) and maps its leading pages from that reservation as its calls
    // first write a position in them; its block-table row names the scratch page past them (a row past a lane's frontier
    // is never read). The row's publications are ordered on the compute stream ahead of the call that reads them.
    [[nodiscard]] std::uint32_t lane_pages() const noexcept { return tables_->logical_page_capacity(); }

    // Reserves `pages` beyond the path's blocks for a lane that holds none (a binding, whose make_room left them free;
    // lane 0's whole context for a scoring window).
    void reserve_pages(std::uint32_t lane, std::uint32_t pages) {
        LaneKv& kv = kv_lanes_[lane];
        if (kv.reservation.valid() || kv.mapped != 0) { throw std::logic_error("Inkling 2: the lane holds KV pages"); }
        if (pages == 0) {
            kv.reservation = pool_->make_empty_reservation();
            return;
        }
        if (!make_room(pages)) { throw std::logic_error("Inkling 2: the KV pool cannot hold the lane's pages"); }
        std::optional<DeviceKVPageReservation> reserved = pool_->reserve(pages);
        if (!reserved) { throw std::logic_error("Inkling 2: the KV pool cannot hold the lane's pages"); }
        kv.reservation = std::move(*reserved);
    }

    // Grows a lane's reservation so its mapped and reserved pages cover positions [0, end): from free pages, or from
    // what the cache can give up. False when the other lanes hold the rest (`missing`: the pages short).
    [[nodiscard]] bool grow_lane(std::uint32_t lane, std::uint64_t end, std::uint32_t& missing) {
        LaneKv& kv               = kv_lanes_[lane];
        const std::uint32_t need = pages_through(end);
        const std::uint32_t have = kv.mapped + (kv.reservation.valid() ? kv.reservation.pages() : 0U);
        if (need <= have) { return true; }
        const std::uint32_t more = need - have;
        if (pool_->available_pages() < more) { (void)make_room(more); }
        if (pool_->available_pages() < more) {
            missing = more - pool_->available_pages();
            return false;
        }
        if (!kv.reservation.valid()) {
            std::optional<DeviceKVPageReservation> reserved = pool_->reserve(more);
            if (!reserved) { throw std::logic_error("Inkling 2: the KV pool refused free pages"); }
            kv.reservation = std::move(*reserved);
        } else {
            pool_->resize_reservation(kv.reservation, kv.reservation.pages() + more);
        }
        return true;
    }

public:
    // Each unit's lane takes the pages its next call writes: a prompt chunk, a replay call, a decode round (a DSpark
    // round's block rows) or forced tokens and the round after them.
    runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit> units) {
        runtime::ResourceReservation out;
        for (const ExecutionUnit& unit : units) {
            const std::uint32_t index = lane_of(unit.sequence);
            const Lane& lane          = lanes_[index];
            const std::uint64_t pending = lane.tokens.size();
            std::uint64_t end           = lane.processed;
            switch (unit.kind) {
            case ExecutionUnitKind::Prefill:
                end = std::min<std::uint64_t>(lane.prompt_tokens, std::uint64_t{lane.processed} + o_.prefill_chunk);
                break;
            case ExecutionUnitKind::Replay:
                end = lane.replay_target;
                break;
            case ExecutionUnitKind::Decode:
                end = std::max<std::uint64_t>(pending, std::uint64_t{lane.processed} + round_positions());
                break;
            case ExecutionUnitKind::Control:
                end = std::max<std::uint64_t>(pending + unit.tokens, std::uint64_t{lane.processed} + round_positions());
                break;
            case ExecutionUnitKind::Normalize: continue;
            }
            std::uint32_t missing = 0;
            if (!grow_lane(index, end, missing)) {
                out.shortage.main_kv_pages = std::max(1U, missing);
                return out;
            }
        }
        out.reserved = true;
        return out;
    }

private:

    // Maps every page the call's columns write (each lane's positions through its last column).
    void map_call_pages(const CallPlan& call) {
        std::array<std::int32_t, kMaximumConcurrency> end{};
        end.fill(0);
        for (std::size_t i = 0; i < call.lane.size(); ++i) {
            const auto lane = static_cast<std::size_t>(call.lane[i]);
            end[lane]       = std::max(end[lane], call.position[i] + 1);
        }
        for (std::uint32_t lane = 0; lane < lanes_.size(); ++lane) { map_lane_pages(lane, end[lane]); }
    }

    // Maps a lane's pages through position `end` (exclusive): a call's writes, or a draft's block rows.
    void map_lane_pages(std::uint32_t lane, std::int32_t end) {
        LaneKv& kv      = kv_lanes_[lane];
        const auto need = static_cast<std::uint32_t>((end + ops::inkling2::kKeyUnit - 1) / ops::inkling2::kKeyUnit);
        if (need <= kv.mapped) { return; }
        // A page past the reservation would leave the row naming the scratch page: the unit reserved too little.
        if (need > lane_pages() || !kv.reservation.valid() || need - kv.mapped > kv.reservation.pages()) {
            throw std::logic_error("Inkling 2: a call writes past its lane's reserved pages");
        }
        std::vector<DeviceKVPageHandle> fresh;
        fresh.reserve(need - kv.mapped);
        for (std::uint32_t p = kv.mapped; p < need; ++p) {
            kv.pages[p] = pool_->materialize_one(kv.reservation);
            fresh.push_back(kv.pages[p].handle());
        }
        tables_->publish(kv_rows_[lane].handle(), kv.mapped, std::span<const DeviceKVPageHandle>(fresh), device_.stream);
        kv.mapped = need;
    }

    // Returns a lane's pages and reservation to the pool, after any capture still reading one of them (a tap's tail).
    // Its row keeps naming the pages until its next binding's calls publish theirs: only a lane's own calls read its row,
    // and the compute stream runs its last call before any writer of a page it held.
    void release_pages(std::uint32_t lane) noexcept {
        if (lane >= kv_lanes_.size()) { return; }
        LaneKv& kv = kv_lanes_[lane];
        if (prefix_) {
            try {
                prefix_->order_page_reuse(std::span<const DeviceKVPageLease>(kv.pages).first(kv.mapped));
            } catch (...) {}
        }
        // The row names the scratch page again before its pages go: no later write through it (a draft's block rows,
        // a stale entry) can reach a page another lane or the cache holds.
        if (kv.mapped != 0) {
            try {
                tables_->publish_repeated(kv_rows_[lane].handle(), scratch_.handle(), kv.mapped, device_.stream);
            } catch (...) {}
        }
        for (std::uint32_t p = 0; p < kv.mapped; ++p) { (void)kv.pages[p].release(); }
        kv.mapped = 0;
        kv.reservation.release();
    }

    // The pool's pages only the prefix cache holds: its blocks no lane maps (a lane pins the blocks its row maps), and
    // those only an in-flight Host write holds (a finished prompt's blocks while their copies land), as room_possible
    // counts them.
    [[nodiscard]] std::uint32_t cache_only_pages() const noexcept {
        if (!prefix_) { return 0; }
        const runtime::prefix_cache::PrefixCacheIndex& cache = prefix_->index();
        return std::min(cache.device_evictable_blocks() + cache.device_transfer_held_blocks(), pool_->allocated_pages());
    }
    // The pool's pages requests hold: every allocated page (a lane's own, a cached block, a snapshot's tail) but the
    // scratch page and the cache's own. A published block stays a request's while its lane maps it.
    [[nodiscard]] std::uint32_t request_pages() const noexcept {
        const std::uint32_t allocated = pool_->allocated_pages();
        const std::uint32_t other     = 1U + cache_only_pages(); // the scratch page
        return allocated > other ? allocated - other : 0U;
    }

public:
    // What the start planned for the Program and the Program did not allocate (MemoryUsage::Device::reserved): set once
    // the Program is built, from the free device memory before and after.
    void set_reserved_bytes(std::uint64_t bytes) noexcept {
        reserved_bytes_  = bytes;
        reserved_graphs_ = graph_bytes_.load(std::memory_order_relaxed); // the startup graphs are in what was taken
    }

    // ---------------------------------------------------------------- prefix cache (prefix_cache.h)
public:
    runtime::prefix_cache::PersistResult attach_prefix_cache_file(const std::filesystem::path& path, std::string fingerprint,
                                                                  const StartupObserver& observer) {
        if (!prefix_) { return {.message = "the prefix cache is not enabled"}; }
        if (path.empty()) { throw std::invalid_argument("prefix cache file path is empty"); }
        runtime::prefix_cache::PersistResult loaded = prefix_->load(path, fingerprint, observer);
        prefix_file_         = path;
        prefix_fingerprint_  = std::move(fingerprint);
        prefix_saved_writes_ = prefix_->counters().host_write_bytes;
        return loaded;
    }

    // The periodic save: only when the Host tier was written since the last one. Copies land first (the save reads
    // the slabs; lanes mapping a path do not matter).
    std::optional<runtime::prefix_cache::PersistResult> save_prefix_cache_now(const CancellationView& abandoned) {
        if (!prefix_ || prefix_file_.empty() || prefix_->counters().host_write_bytes == prefix_saved_writes_) {
            return std::nullopt;
        }
        CUDA_CHECK(cudaStreamSynchronize(device_.stream));
        prefix_->drain();
        // Withdrawn entries another lane pinned until now never reach the file (hybrid-prefix-cache-spec §7.8).
        if (!prefix_withdrawal_.empty()) { (void)prefix_->index().withdraw(prefix_withdrawal_); }
        const std::uint64_t written                 = prefix_->counters().host_write_bytes;
        runtime::prefix_cache::PersistResult saved = prefix_->save(prefix_file_, prefix_fingerprint_, abandoned);
        if (saved.ok) {
            prefix_saved_writes_ = written;
            ++prefix_saves_;
        }
        return saved;
    }

    // After every lane is released (shutdown_cleanup): copies land, then the save reads the slabs.
    void save_prefix_cache_for_shutdown() noexcept {
        if (!prefix_ || prefix_file_.empty()) { return; }
        try {
            CUDA_CHECK(cudaStreamSynchronize(device_.stream));
            prefix_->drain();
            if (!prefix_withdrawal_.empty()) { (void)prefix_->index().withdraw(prefix_withdrawal_); }
            const PrefixCacheSaveControl& control = o_.prefix_save;
            if (!control.begin()) {
                prefix_shutdown_save_ = runtime::prefix_cache::PersistResult{.message = "abandoned before it began"};
                return;
            }
            struct Ended {
                const PrefixCacheSaveControl& control;
                bool saved = false;
                ~Ended() { control.end(saved); }
            } ended{control};
            prefix_shutdown_save_ =
                prefix_->save(prefix_file_, prefix_fingerprint_, CancellationView([&control] { return control.abandoned(); }));
            ended.saved = prefix_shutdown_save_->ok;
        } catch (const std::exception& error) {
            prefix_shutdown_save_ = runtime::prefix_cache::PersistResult{.message = error.what()};
        } catch (...) { prefix_shutdown_save_ = runtime::prefix_cache::PersistResult{.message = "unknown error"}; }
    }

    [[nodiscard]] std::optional<runtime::prefix_cache::PersistResult> prefix_shutdown_save() const {
        return prefix_shutdown_save_;
    }

    [[nodiscard]] std::uint64_t prefix_epoch() const noexcept { return prefix_ ? prefix_->index().snapshot_epoch() : 0U; }

    // The snapshot each queued request would resume from now, in queue order (paused requests first): a paused one's
    // deepest state of its ledger up to its frontier, as its resume binds it (its pause's endpoint while cached), a fresh
    // one's as its admission chooses.
    void hold_queue(std::span<const RequestBasePlan* const> queue, std::span<const ResumeState* const> resumes) {
        if (!prefix_ || prefix_failed_) { return; }
        if (resumes.size() != queue.size()) { throw std::logic_error("Inkling 2: queue holds need one resume entry a request"); }
        namespace pc                = runtime::prefix_cache;
        pc::PrefixCacheIndex& index = prefix_->index();
        std::vector<pc::SnapshotRef> holds;
        for (std::size_t entry = 0; entry < queue.size() && holds.size() < pc::kMaxQueueHolds; ++entry) {
            const RequestBasePlan* base = queue[entry];
            if (base == nullptr || base->impl_ == nullptr || !base->impl_->reuse) { continue; }
            const BasePlanImpl& b    = *base->impl_;
            const ResumeStateImpl* r = ContractAccess::resume_impl(resumes[entry]);
            const std::vector<TokenId>& tokens = r != nullptr ? r->lane.tokens : b.tokens;
            const auto n = r != nullptr ? std::min<std::uint32_t>(static_cast<std::uint32_t>(tokens.size()), r->target + 1U)
                                        : static_cast<std::uint32_t>(tokens.size());
            pc::MatchResult match = index.match(tokens, r != nullptr ? r->lane.prefix.hashes : b.block_hashes,
                                                r != nullptr ? r->lane.prefix.extras : b.block_extras, n);
            const auto spans = image_spans(r != nullptr ? r->lane.images : b.images);
            std::erase_if(match.candidates, [&](const pc::MatchCandidate& c) { return excluded(spans, c.frontier); });
            if (match.candidates.empty()) { continue; }
            if (r != nullptr) {
                const auto deepest = std::max_element(match.candidates.begin(), match.candidates.end(),
                                                      [](const pc::MatchCandidate& a, const pc::MatchCandidate& c) {
                                                          return a.frontier < c.frontier;
                                                      });
                holds.push_back(deepest->snapshot);
            } else if (const pc::AdmissionChoice choice = index.choose(match, n); choice.candidate) {
                holds.push_back(match.candidates[*choice.candidate].snapshot);
            }
        }
        index.set_queue_holds(holds);
    }
    bool prefix_poll_storage() { return prefix_ && prefix_->poll_ssd(); }
    [[nodiscard]] bool prefix_ssd_wait() const noexcept { return prefix_ssd_wait_; }
    [[nodiscard]] bool prefix_storage_due() const noexcept { return prefix_ && prefix_->ssd_due(); }
    [[nodiscard]] std::optional<std::chrono::steady_clock::time_point> prefix_storage_retry_at() const noexcept {
        return prefix_ ? prefix_->ssd_retry_at() : std::nullopt;
    }

    // ---- RAM pressure (Program::start_ram_resize, runtime::expert_offload::RamMonitor) ----
    // As Qwen3.8-Flash-Next's (qwen4_exp program_impl.h maintain_ram), for the one RAM tier this Program has: the
    // prefix cache's Host tier.
    void set_maintenance_waker(std::function<void()> waker) {
        const std::lock_guard<std::mutex> lock(waker_mutex_);
        if (ram_) { ram_->set_waker(waker); }
        waker_ = std::move(waker);
    }

    void start_ram_resize(std::uint64_t reserve_bytes, bool trim_host_tier) {
        if (!prefix_) { return; }
        ram_ = std::make_unique<runtime::expert_offload::RamMonitor>(reserve_bytes, device_.device,
                                                                     inkling2::testing::ram_available());
        ram_->set_room(prefix_->host_tier_shrinkable_bytes(), prefix_->host_tier_growable_bytes());
        ram_reserve_ = reserve_bytes;
        ram_trim_    = trim_host_tier;
        ram_->set_busy(ram_trim_); // the first maintenance trims
        const std::lock_guard<std::mutex> lock(waker_mutex_);
        if (waker_) { ram_->set_waker(waker_); }
    }

    // The Host tier's resize, idle (maintain) and between rounds while requests run (maintain_busy). It advances the
    // resize under way (the tier commits its moves and links or releases its chunks as the monitor's worker finishes
    // them) and, with none under way, starts the one the monitor finds due: a release of whole chunks (the lowest-valued
    // entries go, the rest are copied below the cut), or a growth of one step. A release is logged once complete, the
    // growth steps together once they stop.
    void maintain_ram(bool idle) {
        if (!ram_ || !prefix_) { return; }
        HostMemoryWorker& worker = ram_->worker();
        if (prefix_->host_resizing()) {
            const runtime::prefix_cache::HostSlabStore::Progress step = prefix_->step_host_resize(worker);
            if (ram_resize_) { ram_resize_->prefix += step.released + step.grown; }
            // Landed moves wait only for the restores in flight, which this round finishes: the next step before the
            // next round.
            if (step.waiting && !idle) { ram_->poke(); }
        }
        const bool busy = prefix_->host_resizing();
        ram_->set_busy(busy);
        if (busy) { return; }
        if (ram_resize_) {
            const RamResize done = *std::exchange(ram_resize_, std::nullopt);
            const bool more      = ram_->resized(done.release || done.prefix == 0, prefix_->host_tier_shrinkable_bytes(),
                                                 prefix_->host_tier_growable_bytes());
            if (done.trim) {
                report_trim(done.asked_at, done.prefix);
            } else if (done.release) {
                report_ram(true, done.asked_at, done.prefix);
            } else {
                ram_grown_ += done.prefix;
                if (!more) { report_ram(false, ram_->available(), std::exchange(ram_grown_, 0)); }
            }
            return; // the next step, if due, at the wake resized() asked for
        }
        if (ram_trim_ && trim_host_tier(worker)) { return; }
        if (!idle && !ram_->due()) { return; }
        const runtime::expert_offload::RamDecision decision =
            ram_->decide(prefix_->host_tier_shrinkable_bytes(), prefix_->host_tier_growable_bytes());
        if (decision.grow == 0) { report_ram(false, ram_->available(), std::exchange(ram_grown_, 0)); }
        if (decision.release == 0 && decision.grow == 0) { return; }
        // The monitor hears back (resized) after every decision, one the tier could not start included.
        const RamResize resize{.release = decision.release != 0, .asked_at = ram_->available()};
        if (decision.release != 0) {
            (void)prefix_->begin_host_release(decision.release, worker);
        } else {
            (void)prefix_->begin_host_growth(std::min(decision.grow, runtime::expert_offload::RamControl::kStepBytes),
                                             worker);
        }
        ram_resize_ = resize;
        ram_->set_busy(true);
    }

    // Once, at the first maintenance after the start (the Engine is ready): an automatic Host tier was sized from the
    // start's plan, and the process holds more than the plan counts once it runs (its CUDA context, the driver's copy
    // of its code, runtime buffers), so it gives back what keeps less than --ram-free-mib available, whole chunks, to
    // leave the reserve free as auto promises. Whether a release started.
    bool trim_host_tier(HostMemoryWorker& worker) {
        ram_trim_                     = false;
        const std::uint64_t available = ram_->measure();
        if (available >= ram_reserve_) { return false; }
        if (prefix_->begin_host_release(ram_reserve_ - available, worker) == 0) { return false; }
        ram_resize_ = RamResize{.release = true, .trim = true, .asked_at = available};
        ram_->set_busy(true);
        return true;
    }
    void report_trim(std::uint64_t available, std::uint64_t prefix) const {
        if (prefix == 0) { return; }
        constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
        char text[256];
        std::snprintf(text, sizeof(text),
                      "RAM after the start: %.1f GiB available, under --ram-free-mib %.1f GiB: the automatic prefix "
                      "cache gave back %.2f GiB",
                      static_cast<double>(available) / kGiB, static_cast<double>(ram_reserve_) / kGiB,
                      static_cast<double>(prefix) / kGiB);
        report(DiagnosticLevel::Info, text);
    }
    // A release with the reading that asked for it; a growth with what is available after it.
    void report_ram(bool release, std::uint64_t available, std::uint64_t prefix) const {
        if (prefix == 0) { return; }
        constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
        char text[256];
        if (release) {
            std::snprintf(text, sizeof(text), "RAM pressure: %.1f GiB available; the prefix cache gave back %.1f GiB",
                          static_cast<double>(available) / kGiB, static_cast<double>(prefix) / kGiB);
        } else {
            std::snprintf(text, sizeof(text),
                          "RAM freed: the prefix cache took back %.1f GiB (%.1f GiB in RAM now); %.1f GiB available",
                          static_cast<double>(prefix) / kGiB, static_cast<double>(prefix_->pinned_bytes()) / kGiB,
                          static_cast<double>(available) / kGiB);
        }
        report(DiagnosticLevel::Info, text);
    }
    void report(DiagnosticLevel level, const char* text) const noexcept {
        if (!o_.diagnostics.callback) { return; }
        try {
            o_.diagnostics.callback(Diagnostic{.level = level, .message = text});
        } catch (...) {}
    }

    [[nodiscard]] HybridPrefixCacheStats prefix_stats() const noexcept {
        HybridPrefixCacheStats out;
        if (!prefix_) { return out; }
        const runtime::prefix_cache::PrefixIndexStats s = prefix_->index().stats();
        const prefix::PrefixCacheCounters& c          = prefix_->counters();
        out.nodes                   = s.nodes;
        out.snapshots               = s.snapshots;
        out.device_resident_blocks  = s.device_resident_blocks;
        out.device_evictable_blocks = s.device_evictable_blocks;
        out.host_slabs              = s.host_slab_limit; // in service now (host_free_slabs' bound): a resize moves it
        out.host_free_slabs         = s.host_free_slabs;
        out.host_slab_bytes         = prefix_->host_layout().slab_bytes;
        out.snapshot_hits           = s.snapshot_hits;
        out.reused_tokens           = c.reused_tokens;
        out.blocks_inserted         = c.blocks_inserted;
        out.blocks_reattached       = c.blocks_reattached;
        out.blocks_duplicate        = c.blocks_duplicate;
        out.taps_created            = c.taps_created;
        out.taps_skipped            = c.taps_skipped;
        out.host_image_writes       = c.host_image_writes;
        out.host_block_writes       = c.host_block_writes;
        out.host_image_restores     = c.host_image_restores;
        out.host_block_restores     = c.host_block_restores;
        out.host_write_bytes        = c.host_write_bytes;
        out.host_restore_bytes      = c.host_restore_bytes;
        out.evicted_blocks          = s.device_block_evictions;
        out.host_snapshot_evictions = s.host_snapshot_evictions;
        out.host_dead_reclaims      = s.host_dead_reclaims;
        out.unbacked_node_losses    = s.unbacked_node_losses;
        out.ssd = runtime::prefix_cache::ssd_tier_stats(prefix_->ssd(), s, prefix_->host_layout().slab_bytes,
                                                        prefix_->index().config().ssd_slabs);
        return out;
    }

private:

    // A call of lanes whose image a restore is still writing, or a capture still reading, runs after it (the compute
    // stream waits: the call writes the image's rings).
    void order_after_cache(const CallPlan& call) {
        for (std::uint32_t lane = 0; lane < lanes_.size(); ++lane) {
            if (std::find(call.lane.begin(), call.lane.end(), static_cast<std::int32_t>(lane)) == call.lane.end()) { continue; }
            Lane::Prefix& p = lanes_[lane].prefix;
            if (p.restore) {
                prefix_->order_after_restore(*p.restore, device_.stream);
                p.restore.reset();
            }
            if (p.capture != 0) {
                for (const cudaEvent_t event : prefix_->capture_events(p.capture)) {
                    CUDA_CHECK(cudaStreamWaitEvent(device_.stream, event, 0));
                }
                p.capture = 0;
            }
        }
    }

    // The positions an image captured after a call ending at `end` serves (state_image's rings: the conv rings hold 32
    // positions, of which a frontier reads its 3 before; the drafter's Markov ring 16, of which it reads 1).
    [[nodiscard]] std::uint32_t capture_reach() const noexcept { return drafter_ ? 15U : 29U; }

    // After a prompt (or replay) call: the lane's full blocks the call completed are published, then each tap the call
    // reached is captured. Shared keys hold prompt-computed state only (a resumed request's output equals a cold one's);
    // a replay's decode-route blocks take their private keys (pause_keys).
    void publish_prefill(std::uint32_t idx) {
        Lane& lane = lanes_[idx];
        namespace pc     = runtime::prefix_cache;
        Lane::Prefix& p  = lane.prefix;
        if (!prefix_ || !p.reuse || prefix_failed_) { return; }
        const std::uint32_t end = lane.processed;
        publish_blocks(idx, end);
        if (!p.reuse) { return; }
        // Each tap the call reached: at its own position when the image serves it, else (a flexible tap the call passed)
        // at the call's end; never the prompt's last position (one token is always prefilled).
        const auto n = static_cast<std::uint32_t>(lane.prompt_tokens);
        while (!p.taps.empty() && p.taps.front().position <= end) {
            const pc::PlannedTap tap = p.taps.front();
            p.taps.erase(p.taps.begin());
            std::uint32_t F = tap.position + capture_reach() >= end ? tap.position : end;
            F               = std::min(F, n - 1U);
            if (F <= p.base || std::find(p.frontiers.begin(), p.frontiers.end(), F) != p.frontiers.end()) { continue; }
            if (F != tap.position && excluded(image_spans(lane.images), F)) { continue; } // moved into an image
            capture_tap(idx, F, tap.boundary ? pc::SnapshotKind::Boundary : pc::SnapshotKind::Tap);
        }
    }

    // The lane's full blocks below `end` not yet on its path, each keyed by its block hash (the cache takes the lane's
    // page unless an identical block is mapped elsewhere). bf16p: a row with no packed form fails the request, and
    // nothing it reached is published (reuse goes off).
    void publish_blocks(std::uint32_t idx, std::uint32_t end) {
        Lane& lane = lanes_[idx];
        namespace pc     = runtime::prefix_cache;
        Lane::Prefix& p  = lane.prefix;
        if (unpackable_.p != nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(&unpackable_seen_[idx], static_cast<std::uint32_t*>(unpackable_.p) + idx, 4,
                                       cudaMemcpyDeviceToHost, device_.stream));
            CUDA_CHECK(cudaStreamSynchronize(device_.stream));
            if (unpackable_seen_[idx] != kNoPosition) {
                p.reuse = false;
                return;
            }
        }
        LaneKv& kv        = kv_lanes_[idx];
        const auto blocks = std::min<std::size_t>(end / pc::kBlockTokens, p.hashes.size());
        while (p.path.size() < blocks) {
            const std::size_t bi     = p.path.size();
            const pc::NodeRef parent = p.path.empty() ? pc::NodeRef{} : p.path.back();
            const std::span<const TokenId> tokens(lane.tokens.data() + bi * pc::kBlockTokens, pc::kBlockTokens);
            const pc::InsertResult r = prefix_->insert_block(parent, p.hashes[bi], tokens,
                                                             p.extras.empty() ? 0U : p.extras[bi], kv.pages[bi]);
            p.path.push_back(r.node); // pinned for the lane (insert_block), its Device copy held
            if (r.device_attached) {
                p.published.push_back(r.node);
            } else if (const pc::NodeView view = prefix_->index().node(r.node); view.device == pc::CopyState::Resident) {
                // An identical block another lane maps (the same key: the same bits): the row names its page and the
                // lane's own goes back to the pool, so one block never holds two pages. The calls before read the lane's
                // page; the row's publication runs after them on the compute stream.
                const DeviceKVPageHandle shared = prefix_->page(view.device_id);
                tables_->publish(kv_rows_[idx].handle(), static_cast<std::uint32_t>(bi),
                                 std::span<const DeviceKVPageHandle>(&shared, 1), device_.stream);
                prefix_->order_page_reuse(std::span<const DeviceKVPageLease>(&kv.pages[bi], 1));
                (void)kv.pages[bi].release();
            }
        }
    }

    // `resume_point`: a pause's endpoint, admitted over any retained snapshot that is not pinned (its claim only admits
    // it; a refused capture makes the resume replay).
    void capture_tap(std::uint32_t idx, std::uint32_t F, runtime::prefix_cache::SnapshotKind kind, bool resume_point = false) {
        Lane& lane = lanes_[idx];
        namespace pc     = runtime::prefix_cache;
        Lane::Prefix& p  = lane.prefix;
        const std::uint32_t k = F / pc::kBlockTokens;
        if (p.path.size() < k) { return; } // its blocks are not all published (cannot happen past publish_prefill)
        prefix::PrefixCache::Capture c;
        c.id       = ++capture_ids_;
        c.anchor   = k > 0 ? p.path[k - 1] : pc::NodeRef{};
        c.frontier = F;
        c.tail.assign(lane.tokens.begin() + static_cast<std::ptrdiff_t>(k) * pc::kBlockTokens,
                      lane.tokens.begin() + F);
        if (!c.tail.empty()) {
            // The lane's page of block k: its own, or the cache's once the block was published.
            const DeviceKVPageLease& own = kv_lanes_[idx].pages[k];
            c.tail_page = own.valid() ? own.handle()
                                      : prefix_->page(prefix_->index().node(p.path[k]).device_id);
        }
        c.kind = kind;
        // Valued against the nearest snapshot below it on the path (its base, or an earlier tap or cached snapshot).
        std::uint32_t below = p.base;
        for (const std::uint32_t f : p.frontiers) {
            if (f < F) { below = std::max(below, f); }
        }
        c.claim = resume_point ? pc::kResumePointClaim : prefix_->index().estimate_priority(below, F, !c.tail.empty());
        const std::uint64_t id = c.id;
        // The path's Device-only blocks go to the Host tier behind the image (the lane's next call waits only for the
        // image).
        if (prefix_->capture(idx, std::move(c), std::span<const pc::NodeRef>(p.path).first(k), device_.stream)) {
            p.capture = id;
            p.captures.push_back(id);
            p.frontiers.push_back(F);
            p.deepest_capture = std::max(p.deepest_capture, F);
        }
    }

    // The lane's cache references at its release: its path's pins, and (a request whose logits went non-finite) what it
    // published withdrawn (hybrid-prefix-cache-spec §7.8). A failure empties the cache.
    void release_prefix(std::uint32_t idx, bool pausing = false) noexcept {
        Lane& lane = lanes_[idx];
        if (!prefix_) { return; }
        Lane::Prefix& p = lane.prefix;
        try {
            // Only the lane's next call waits for its copies (order_after_cache), and a lane released before one (a
            // request cancelled or failed after its binding, or one that ends at its prompt call) may still have a
            // restore landing in its image and tail page or a capture reading its image: the next binding of the lane
            // writes the image on the compute stream (the cold start's zeroing, a restore that follows the compute
            // stream, its first call) and its pages go back to the pool below, so the compute stream waits for both
            // here (free once they have landed), as Qwen3.5's and Flash-Next's lane releases do.
            if (p.restore) { prefix_->order_after_restore(*p.restore, device_.stream); }
            for (const cudaEvent_t event : prefix_->capture_events(p.capture)) {
                CUDA_CHECK(cudaStreamWaitEvent(device_.stream, event, 0));
            }
            if (!p.path.empty()) { prefix_->index().release_path(p.path); }
            // A lane released (cancelled) after a stalled MoE call and before the next check: what it published may
            // hold the incomplete expert output, so it goes too (the word stays set for check_moe_error, which fails
            // the other lanes the call may have reached).
            if (moe_error_ != nullptr && *static_cast<volatile std::uint32_t*>(moe_error_) != 0) { p.withdraw = true; }
            if (p.withdraw && (!p.published.empty() || !p.captures.empty())) {
                // What another lane still maps stays in prefix_withdrawal_ and goes at a later release or save; the
                // cache file is rewritten at its next save, and deleted now when a save since the binding may hold
                // what the request published (as Flash-Next's and Qwen3.5's withdrawals).
                prefix_->drain();
                prefix_withdrawal_.nodes.insert(prefix_withdrawal_.nodes.end(), p.published.begin(), p.published.end());
                for (const std::uint64_t id : p.captures) {
                    if (const auto r = prefix_->capture_result(id); r && r->created) {
                        prefix_withdrawal_.snapshots.push_back(r->snapshot);
                    }
                }
                if (!prefix_file_.empty()) {
                    prefix_saved_writes_ = ~std::uint64_t{0};
                    if (prefix_saves_ != p.saves) {
                        std::error_code ignored;
                        std::filesystem::remove(prefix_file_, ignored);
                    }
                }
            }
            const bool withdrew = p.withdraw && (!p.published.empty() || !p.captures.empty());
            const std::uint32_t removed =
                prefix_withdrawal_.empty() ? 0U : prefix_->index().withdraw(prefix_withdrawal_);
            if (withdrew && o_.diagnostics.callback) {
                const std::size_t waiting = prefix_withdrawal_.nodes.size() + prefix_withdrawal_.snapshots.size();
                o_.diagnostics.callback(Diagnostic{
                    .level   = DiagnosticLevel::Warning,
                    .message = "Inkling 2: withdrew " + std::to_string(removed) +
                               " prefix-cache entries a failed request published" +
                               (waiting != 0 ? "; " + std::to_string(waiting) + " go once other requests release them"
                                             : std::string())});
            }
            if (!pausing) {
                for (const std::uint64_t id : p.captures) { prefix_->forget_capture(id); }
            }
        } catch (...) {
            // Emptied once no lane maps it (release); until then nothing is reused or published.
            prefix_failed_ = true;
        }
        p = Lane::Prefix{};
    }

    // A cache whose bookkeeping failed is rebuilt empty once the last lane that maps it is released.
    void clear_failed_prefix() noexcept {
        if (!prefix_ || !prefix_failed_ ||
            std::any_of(lanes_.begin(), lanes_.end(), [](const Lane& l) { return l.bound; })) {
            return;
        }
        prefix_->clear();
        prefix_withdrawal_ = {}; // its references would name the rebuilt index's entries
        prefix_failed_     = false;
    }

    // The Forward over the uploaded call, its head over columns [head_first, head_first + head_columns); `after_embed`
    // (a prompt chunk's images) runs between the embedding and the first visit.
    void run_forward(std::int32_t head_first, std::int32_t head_columns, const std::function<void()>& after_embed = {}) {
        forward_.embed(device_.stream);
        if (after_embed) { after_embed(); }
        for (std::uint32_t v = 0; v < c_.visits(); ++v) { forward_.visit(v, view_, device_.stream); }
        forward_.finish(view_, forward_.final_inputs(), device_.stream, head_first, head_columns);
    }

    // A prompt's images: inside the prompt, in order, apart, each within the tower's budget (the frontend made them so).
    void check_images(const std::vector<PromptImage>& images, std::uint32_t tokens) const {
        if (!vision_) { throw std::invalid_argument("Inkling 2: image input needs --vision"); }
        std::uint64_t next = 0;
        for (const PromptImage& image : images) {
            const std::int32_t cells = image.image.cells();
            if (cells <= 0 || static_cast<std::uint32_t>(cells) > vision_->max_cells() || image.image.patches == nullptr ||
                image.first_token < next || std::uint64_t{image.first_token} + cells > tokens) {
                throw std::invalid_argument("Inkling 2: a prompt image's tokens or size disagree with the prompt");
            }
            next = std::uint64_t{image.first_token} + cells;
        }
    }

    // The image rows of a prompt chunk [first, first + count) over its embedding columns (after the embedding norm,
    // as the reference scatters them): each overlapping image encoded, unless the tower still holds its rows (an
    // image across chunks encodes once), and its overlapping rows copied. `encoded_set_` keeps the encoded image's
    // prompt alive, so its address names it until another image is encoded.
    void overlay_images(const std::shared_ptr<const std::vector<PromptImage>>& images, std::uint32_t first,
                        std::uint32_t count) {
        if (!images) { return; }
        const std::size_t width = c_.hidden_size;
        for (std::size_t i = 0; i < images->size(); ++i) {
            const PromptImage& image  = (*images)[i];
            const std::uint32_t begin = image.first_token;
            const std::uint32_t end   = begin + static_cast<std::uint32_t>(image.image.cells());
            if (end <= first || begin >= first + count) { continue; }
            if (encoded_set_ != images || encoded_index_ != i) {
                encoded_rows_ = vision_->encode(
                    {image.image.rows, image.image.cols, std::span<const std::uint16_t>(*image.image.patches)},
                    device_.stream);
                encoded_set_   = images;
                encoded_index_ = i;
            }
            const std::uint32_t lo = std::max(begin, first), hi = std::min(end, first + count);
            CUDA_CHECK(cudaMemcpyAsync(forward_.buffers().e + (lo - first) * width, encoded_rows_ + (lo - begin) * width,
                                       (hi - lo) * width * 2, cudaMemcpyDeviceToDevice, device_.stream));
        }
    }

    // The lossless route's call (lossless-route.md §3a): one lane's columns through LosslessForward's cache form, which
    // continues the lane's state (its conv rings, window rings and pages) and leaves the call's in it; with `head`,
    // the last column's logits (row 0 of lossless_->logits()).
    void run_lossless(std::uint32_t index, const CallPlan& call, bool head) {
        map_call_pages(call);
        if (prefix_) { order_after_cache(call); }
        LosslessLaneState state;
        state.image  = static_cast<std::uint16_t*>(images_.p) + static_cast<std::size_t>(index) * forward_.layout().lane_words;
        state.layout = &forward_.layout();
        state.pool_k = view_.pool_k;
        state.pool_v = view_.pool_v;
        state.table  = view_.table + static_cast<std::size_t>(index) * view_.table_stride;
        state.first  = call.position.front();
        lossless_->run(call.tokens, state, device_.stream);
        if (head) { lossless_->head(call.columns() - 1, 1, device_.stream); }
    }

    // A decode round on the lossless route: each row's lane runs its pending inputs (its last token and any forced
    // ones) as one call, a lane at a time (the route's calls are eager), and the rows' logits are gathered in the
    // Engine's order for one sample.
    PendingBatch decode_lossless(std::span<const SequenceHandle> sequences, runtime::TokenMaskProvider* masks,
                                 Clock::time_point start) {
        if (gathered_.bytes == 0) { gathered_ = DeviceBuffer(static_cast<std::size_t>(vocab_) * 4 * kMaximumConcurrency); }
        auto* gathered = static_cast<float*>(gathered_.p);
        std::vector<std::uint32_t> rows;
        std::vector<std::int32_t> identity;
        for (const SequenceHandle& s : sequences) { rows.push_back(lane_of(s)); }
        for (std::size_t r = 0; r < rows.size(); ++r) {
            const std::uint32_t index = rows[r];
            Lane& lane                = lanes_[index];
            if (lane.processed >= lane.tokens.size()) { throw std::logic_error("Inkling 2: a decode row has no input"); }
            CallPlan call;
            call.first.assign(index + 1, 0);
            for (std::uint32_t p = lane.processed; p < lane.tokens.size(); ++p) {
                call.tokens.push_back(lane.tokens[p]);
                call.lane.push_back(static_cast<std::int32_t>(index));
                call.position.push_back(static_cast<std::int32_t>(p));
            }
            run_lossless(index, call, true);
            CUDA_CHECK(cudaMemcpyAsync(gathered + r * static_cast<std::size_t>(vocab_), lossless_->logits(),
                                       static_cast<std::size_t>(vocab_) * 4, cudaMemcpyDeviceToDevice, device_.stream));
            identity.push_back(static_cast<std::int32_t>(r));
        }
        stage_sample(rows, identity, ops::kSamplePurposeDecode, masks);
        fill_staged_masks(masks);
        launch_sample(rows.size(), identity, ops::kSamplePurposeDecode, gathered);
        finish_sample(rows.size());
        for (const std::uint32_t index : rows) { lanes_[index].processed = static_cast<std::uint32_t>(lanes_[index].tokens.size()); }
        const std::uint64_t ns = elapsed_ns(start);
        for (const std::uint32_t index : rows) {
            lanes_[index].decode_ns += ns;
            lanes_[index].decode_share_ns += ns / rows.size();
        }
        const std::uint64_t transaction = ++transaction_;
        pending_      = transaction;
        pending_kind_ = PendingKind::Decode;
        return ContractAccess::batch(this, transaction, sequences, {sampled_tokens_.data(), sequences.size()},
                                     timing_since(start), {failures_.data(), sequences.size()});
    }

    // A decode or verify call: the Forward with the head over every column, then (DSpark) its columns' context rows
    // (a rejected verify column's rows are overwritten before anything reads them).
    void run_decode_call(std::int32_t columns) {
        run_forward(0, columns);
        if (drafter_) { drafter_->append(0, columns, device_.stream); }
    }

    // ---------------------------------------------------------------- DSpark (design §13)
    static constexpr double kAcceptanceWeight     = 0.1;  // the calibration's EWMA weight
    static constexpr std::uint64_t kProbeInterval = 8;    // one draft more every 8 rounds keeps longer deciles measured
    // Costs in plain one-row rounds (2026-11-21, the forward test's decode round at 4K: +0.11-0.13 a verify column;
    // the draft's ~1 ms of a ~6 ms round).
    static constexpr double kDraftCost    = 0.12;
    static constexpr double kProposalCost = 0.18;

    // A round drafts when one row decodes with its one pending input, the drafter has its context (F >= 1), the
    // position after the anchor fits the context, and no penalty counts the draws: verification samples each column as
    // plain decoding would. A reasoning ban is fine (every column samples under it, and the round ends at the first close
    // it draws), and so is a grammar (each column samples under the mask of its position after the drafts before it,
    // constrained-decoding.md §8.1).
    [[nodiscard]] bool dspark_round(std::span<const SequenceHandle> sequences) const {
        if (!drafter_ || sequences.size() != 1) { return false; }
        const Lane& lane = lanes_[lane_of(sequences[0])];
        // A token-constrained row decodes plainly: its steps' records come from its one column a round.
        return !lane.penalties && !lane.constraint && lane.processed != 0 && lane.tokens.size() == lane.processed + 1U &&
               lane.processed + 1U < o_.max_context;
    }

    PendingBatch decode_dspark(SequenceHandle sequence, std::uint32_t remaining, runtime::TokenMaskProvider* masks,
                               Clock::time_point start) {
        const std::uint32_t index = lane_of(sequence);
        Lane& lane                = lanes_[index];
        const auto F              = static_cast<std::int32_t>(lane.processed);
        // Drafts whose positions fit the context and whose round (drafts + 1 tokens) fits the row's budget.
        // With the calibrated length the Markov head drafts at most two more steps than the last round verified (each
        // step reads the whole folded vocabulary): room for the policy's probe of one more.
        const std::int32_t most = o_.draft_tokens_auto ? lane.step_cap : static_cast<std::int32_t>(o_.draft_tokens);
        const std::int32_t steps =
            std::min({most, static_cast<std::int32_t>(o_.max_context) - 1 - F,
                      static_cast<std::int32_t>(std::min<std::uint32_t>(remaining - 1U, ops::inkling2::kMarkovSteps))});
        // The drafts: the block, the head over its rows and the Markov steps, read back before the verify call. The
        // block's rows F .. F + 7 go to the full layer's pages first (mapped here: the verify call maps no more).
        map_lane_pages(index, F + ops::inkling2::kDSparkBlock);
        drafter_->stage({F, lane.tokens[static_cast<std::size_t>(F)], static_cast<std::int32_t>(index)});
        CUDA_CHECK(cudaEventRecord(round_events_[0], device_.stream));
        if (o_.use_cuda_graph) {
            draft_graph(steps).launch(device_.stream);
        } else {
            drafter_->draft(steps, device_.stream);
            drafter_->download(steps, device_.stream);
        }
        CUDA_CHECK(cudaEventRecord(round_events_[1], device_.stream));
        {
            const auto wait = Clock::now();
            CUDA_CHECK(cudaStreamSynchronize(device_.stream));
            wait_ns_ += elapsed_ns(wait);
        }
        std::array<TokenId, ops::inkling2::kMarkovSteps> drafts{};
        std::array<float, ops::inkling2::kMarkovSteps> probabilities{};
        for (std::int32_t i = 0; i < steps; ++i) {
            drafts[static_cast<std::size_t>(i)]        = drafter_->host_tokens()[i];
            probabilities[static_cast<std::size_t>(i)] = drafter_->host_probabilities()[i];
        }
        std::int32_t k = steps;
        if (o_.draft_tokens_auto) {
            k = runtime::speculative::calibrated_draft_length(
                lane.calibration, std::span<const float>(probabilities.data(), static_cast<std::size_t>(steps)), steps,
                kDraftCost, kProposalCost);
            if (++lane.policy_rounds % kProbeInterval == 0) { k = std::min(k + 1, steps); }
            const auto window = static_cast<std::int32_t>(o_.draft_tokens);
            lane.step_cap     = std::clamp(k + 2, std::min(2, window), window);
        }
        // The verify call: the pending input and the k drafts at F .. F + k, every column sampled as decoding would.
        CallPlan call;
        call.first.assign(index + 1, 0);
        const std::int32_t columns = k + 1;
        std::array<std::uint32_t, kMaximumConcurrency> rows{};
        std::array<std::int32_t, kMaximumConcurrency> logit_rows{}, positions{};
        for (std::int32_t i = 0; i < columns; ++i) {
            call.tokens.push_back(i == 0 ? lane.tokens[static_cast<std::size_t>(F)] : drafts[static_cast<std::size_t>(i - 1)]);
            call.lane.push_back(static_cast<std::int32_t>(index));
            call.position.push_back(F + i);
            rows[static_cast<std::size_t>(i)]       = index;
            logit_rows[static_cast<std::size_t>(i)] = i;
            positions[static_cast<std::size_t>(i)]  = F + i;
        }
        upload(call);
        const auto n = static_cast<std::size_t>(columns);
        // Every column samples under the row's reasoning ban (an open reasoning excludes its end tokens).
        const runtime::ReasoningBan ban = masks != nullptr ? masks->reasoning_ban(0) : runtime::ReasoningBan{};
        stage_sample({rows.data(), n}, {logit_rows.data(), n}, ops::kSamplePurposeDecode, masks, {positions.data(), n},
                     true);
        CUDA_CHECK(cudaEventRecord(round_events_[2], device_.stream));
        // A constrained row: column i samples under the mask of its position after drafts 0 .. i - 1 (an illegal draft
        // leaves the columns after it a filler: its own column's draw differs from it, so none of them is accepted). The
        // matcher walks the drafts while the verify forward runs; the sample follows the masks' upload.
        std::uint64_t dead = 0;
        const bool constrained = masks != nullptr && masks->constrained(0);
        if (constrained) {
            if (o_.use_cuda_graph) {
                verify_graph(index, columns, false).launch(device_.stream);
            } else {
                run_decode_call(columns);
            }
            fill_masks(*masks, 0, 0, std::span<const TokenId>(drafts.data(), static_cast<std::size_t>(k)));
            dead = mask_dead_[0];
            launch_sample(n, {logit_rows.data(), n}, ops::kSamplePurposeDecode);
        } else if (o_.use_cuda_graph) {
            verify_graph(index, columns).launch(device_.stream);
        } else {
            run_decode_call(columns);
            launch_sample(n, {logit_rows.data(), n}, ops::kSamplePurposeDecode);
        }
        CUDA_CHECK(cudaEventRecord(round_events_[3], device_.stream));
        finish_sample(n, true);
        {
            float draft = 0, gap = 0, verify = 0;
            CUDA_CHECK(cudaEventElapsedTime(&draft, round_events_[0], round_events_[1]));
            CUDA_CHECK(cudaEventElapsedTime(&gap, round_events_[1], round_events_[2]));
            CUDA_CHECK(cudaEventElapsedTime(&verify, round_events_[2], round_events_[3]));
            lane.draft_ms += draft, lane.gap_ms += gap, lane.verify_ms += verify;
        }
        // The accepted prefix: draft i stands while it equals the target's own draw at its position.
        std::int32_t a = 0;
        while (a < k && drafts[static_cast<std::size_t>(a)] == sampled_tokens_[static_cast<std::size_t>(a)] &&
               failures_[static_cast<std::size_t>(a)] == runtime::PendingRowFailure::None) {
            ++a;
        }
        lane.calibration.observe(std::span<const float>(probabilities.data(), static_cast<std::size_t>(steps)),
                                 static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(a), kAcceptanceWeight);
        // The round ends at the first reasoning close it draws: later columns sampled under the ban the close lifts.
        if (ban.active()) {
            for (std::int32_t i = 0; i < a; ++i) {
                if (sampled_tokens_[static_cast<std::size_t>(i)] == ban.close) {
                    a = i;
                    break;
                }
            }
        }
        // A kept column's failure (0 .. a) fails the request; a cut column's does not count. Under a grammar a column
        // past a drafted close can have no legal token outside the ban it still samples under (the tool-call opener the
        // ban withholds while thinking is open, after a drafted close and a tool's name): its draw is discarded.
        runtime::PendingRowFailure failure = runtime::PendingRowFailure::None;
        for (std::int32_t i = 0; i <= a; ++i) {
            if (failures_[static_cast<std::size_t>(i)] != runtime::PendingRowFailure::None) {
                failure = failures_[static_cast<std::size_t>(i)];
                break;
            }
        }
        // A dead position among the kept ones (0 .. a) fails the request as plain decoding would.
        if (failure == runtime::PendingRowFailure::None && (dead & ((std::uint64_t{2} << a) - 1U)) != 0) {
            failure = runtime::PendingRowFailure::ConstraintDeadEnd;
        }
        // bf16p: a row with no packed form at a kept position (the anchor's F .. F + a) fails the request as plain
        // decoding would; one at a rejected draft's position is forgotten (the next calls rewrite that row).
        if (const std::uint32_t at = unpackable_seen_[index]; at != kNoPosition) {
            if (at <= static_cast<std::uint32_t>(F + a)) {
                if (failure == runtime::PendingRowFailure::None) { failure = runtime::PendingRowFailure::UnpackableKv; }
            } else {
                clear_unpackable(index);
            }
        }
        failures_[0] = failure;
        if (failure != runtime::PendingRowFailure::None && failure != runtime::PendingRowFailure::ConstraintDeadEnd) {
            lane.prefix.withdraw = true; // as finish_sample does for a plain row
        }
        ++lane.rounds;
        lane.drafted += static_cast<std::uint64_t>(k);
        lane.accepted += static_cast<std::uint64_t>(a);
        for (std::int32_t i = 0; i < a; ++i) { ++lane.accepted_at[static_cast<std::size_t>(i)]; }
        row_counts_[0] = a + 1;
        std::copy_n(sampled_tokens_.begin(), a + 1, round_tokens_.begin());
        pending_draft_     = {true};
        pending_firsts_[0] = static_cast<std::uint32_t>(F);
        const std::uint64_t transaction = ++transaction_;
        const std::uint64_t ns          = elapsed_ns(start);
        lane.decode_ns += ns;
        lane.decode_share_ns += ns;
        pending_      = transaction;
        pending_kind_ = PendingKind::Decode;
        return ContractAccess::batch(this, transaction, {&sequence, 1}, {round_tokens_.data(), kRowStride},
                                     timing_since(start), {failures_.data(), 1}, {row_counts_.data(), 1}, kRowStride);
    }

    // ---------------------------------------------------------------- DSpark, several rows (design §13.7)
    // Rows decoding together draft together: every row's block is drafted (the rows one after another), and one verify
    // call takes 1 + K columns of each row (K the round's one width: a graph per (rows, K) instead of one per mix of
    // widths), each column sampled as decoding would. The decode route is width-invariant (a lane's columns'
    // bits do not depend on the other lanes' columns, the forward test's wide_lanes), so a round of several rows
    // commits the tokens their plain decoding would, as a one-row round does.
    //
    // The width is the one that makes the most of the rows' calibrated drafts per verify time: the rows' expected
    // tokens at K (draft_length.h, each row capped by its budget) over the verify call's device time at K. The rounds
    // measure that cost line themselves (rows_cost_, per row count: the plain round's and each width's device time, as
    // EWMAs), since it moves with the context (attention is per lane, the experts' union per column). A batch size
    // whose drafted rounds commit fewer tokens per host millisecond than its plain rounds decodes plainly, drafting
    // again every kRowsProbe rounds to re-measure; a plain round every kRowsPlainProbe drafted rounds keeps the plain
    // rate measured.
    struct RowsCost {
        double plain_ms = 0;                                         // a plain round's forward and sample (device)
        std::array<double, ops::inkling2::kDSparkBlock> verify_ms{}; // [k]: the verify call of 1 + k columns a row
        double plain_wall = 0, draft_wall = 0, draft_tokens = 0;     // host ms of a round; a drafted round's tokens
        std::uint64_t rounds = 0, skipped = 0;                       // drafted rounds; plain ones since the last draft
        // What the policy did since the last report (report_rows): the widths it chose, the rounds it left plain, the
        // dealt rounds' expected tokens over K each's at the same columns, and the rounds that dealt a row every draft
        // it had while it could have drafted more.
        std::array<std::uint64_t, ops::inkling2::kDSparkBlock> widths{};
        std::uint64_t plain_rounds = 0;
        double own_ratio = 0;
        std::uint64_t own_rounds = 0, saturated = 0;
    };
    static constexpr std::uint64_t kRowsReport = 256; // drafted rounds between a batch size's Debug reports
    // The cost line moves with the context (attention reads every key of each lane, the drafter's full layer too; the
    // acceptance moves with the text): rounds are measured per row count and per bucket of the rows' longest context,
    // up to 2K, 4K, ... 128K and beyond (2026-11-26: costs measured at 100K and taken for 2K rounds kept drafting on
    // there at 18 % acceptance, 17 % slower than plain).
    static constexpr std::size_t kContextBuckets = 8;
    [[nodiscard]] static std::size_t context_bucket(std::uint32_t longest) noexcept {
        std::size_t bucket = 0;
        while (bucket + 1 < kContextBuckets && (std::uint64_t{2048} << bucket) < longest) { ++bucket; }
        return bucket;
    }
    [[nodiscard]] std::uint32_t longest_context(std::span<const SequenceHandle> sequences) const noexcept {
        std::uint32_t longest = 0;
        for (const SequenceHandle& s : sequences) { longest = std::max(longest, lanes_[lane_of(s)].processed); }
        return longest;
    }
    [[nodiscard]] RowsCost& rows_cost(std::span<const SequenceHandle> sequences) noexcept {
        return rows_cost_[sequences.size()][context_bucket(longest_context(sequences))];
    }
    static constexpr double kCostWeight            = 0.1;
    static constexpr double kColumnPrior           = 0.1; // a width's cost in plain rounds a draft, before any is measured
    static constexpr std::uint64_t kRowsProbe      = 16;
    static constexpr std::uint64_t kRowsPlainProbe = 64;
    static void ewma(double& value, double sample) noexcept {
        value = value == 0.0 ? sample : (1.0 - kCostWeight) * value + kCostWeight * sample;
    }

    // Whether every row could draft: its one pending input (no forced tokens), its drafter context (F >= 1), a whole
    // block of positions left in the context, no penalty counting its draws and no token constraint. A grammar and a
    // reasoning ban are fine, as in a one-row round.
    [[nodiscard]] bool rows_eligible(std::span<const SequenceHandle> sequences,
                                     std::span<const runtime::RoundBudget> budgets) const {
        if (!drafter_ || lossless_ || sequences.size() < 2 || budgets.size() != sequences.size()) { return false; }
        for (std::size_t r = 0; r < sequences.size(); ++r) {
            const Lane& lane = lanes_[lane_of(sequences[r])];
            if (lane.penalties || lane.constraint || lane.processed == 0 || lane.tokens.size() != lane.processed + 1U ||
                std::uint64_t{lane.processed} + ops::inkling2::kDSparkBlock > o_.max_context ||
                budgets[r].generated_tokens_remaining == 0) {
                return false;
            }
        }
        return true;
    }

    // A round of several rows drafts when they are eligible and their batch size's measured rates favour drafting.
    [[nodiscard]] bool dspark_rows_round(std::span<const SequenceHandle> sequences,
                                         std::span<const runtime::RoundBudget> budgets) {
        if (!rows_eligible(sequences, budgets) || !o_.use_cuda_graph) { return false; }
        RowsCost& cost = rows_cost(sequences);
        if (!o_.draft_tokens_auto) { return true; } // a fixed --draft-tokens drafts every round
        // The plain rate is measured after the first drafted round, and kept measured by a plain round every
        // kRowsPlainProbe drafted rounds.
        if (cost.rounds != 0 && cost.skipped == 0 && (cost.plain_wall == 0.0 || cost.rounds % kRowsPlainProbe == 0)) {
            ++cost.skipped;
            return false;
        }
        if (cost.plain_wall == 0.0 || cost.draft_wall == 0.0) {
            cost.skipped = 0;
            return true;
        }
        const double drafted = cost.draft_tokens / cost.draft_wall;
        const double plain   = static_cast<double>(sequences.size()) / cost.plain_wall;
        if (drafted >= plain || ++cost.skipped >= kRowsProbe) {
            cost.skipped = 0;
            return true;
        }
        ++cost.plain_rounds;
        return false;
    }

    // A batch size's Debug report every kRowsReport drafted rounds: the widths chosen, the rounds the rates left plain,
    // the cost line and both rates, and the graphs captured and evicted so far.
    void report_rows(std::size_t rows, std::size_t bucket, RowsCost& cost) {
        if (!o_.diagnostics.callback || cost.rounds % kRowsReport != 0) { return; }
        std::string widths;
        for (std::size_t k = 0; k < cost.widths.size(); ++k) {
            if (cost.widths[k] != 0) { widths += " K" + std::to_string(k) + ":" + std::to_string(cost.widths[k]); }
        }
        char text[768];
        std::snprintf(text, sizeof(text),
                      "Inkling 2 DSpark rounds of %zu rows, contexts up to %lluK: widths%s; %llu rounds plain by the "
                      "rates; plain round %.2f "
                      "ms (device), %.2f ms (host), %.3f tok/ms; drafted round %.2f ms (host), %.2f tokens, %.3f tok/ms; "
                      "verify ms by width %.2f %.2f %.2f %.2f %.2f %.2f %.2f; graphs of these rounds %llu captured, "
                      "%llu evicted; the drafts dealt by row expect %+.1f %% more tokens than K each at the same "
                      "columns; %llu rounds dealt a row every draft it had",
                      rows, static_cast<unsigned long long>(2ULL << bucket), widths.c_str(),
                      static_cast<unsigned long long>(cost.plain_rounds), cost.plain_ms,
                      cost.plain_wall, cost.plain_wall > 0.0 ? static_cast<double>(rows) / cost.plain_wall : 0.0,
                      cost.draft_wall, cost.draft_tokens, cost.draft_wall > 0.0 ? cost.draft_tokens / cost.draft_wall : 0.0,
                      cost.verify_ms[1], cost.verify_ms[2], cost.verify_ms[3], cost.verify_ms[4], cost.verify_ms[5],
                      cost.verify_ms[6], cost.verify_ms[7], static_cast<unsigned long long>(rows_captures_),
                      static_cast<unsigned long long>(rows_evictions_),
                      cost.own_rounds != 0 ? 100.0 * (cost.own_ratio / static_cast<double>(cost.own_rounds) - 1.0) : 0.0,
                      static_cast<unsigned long long>(cost.saturated));
        report(DiagnosticLevel::Debug, text);
        cost.widths       = {};
        cost.plain_rounds = 0;
        cost.own_ratio    = 0.0;
        cost.own_rounds   = 0;
        cost.saturated    = 0;
    }

    // The verify call's device time at width k for rows of `cost`: measured, else interpolated between measured widths
    // (the plain round is width 0), else extrapolated by the last measured slope or the prior. An unmeasured plain
    // round is the narrowest measured width less the prior's cost; with nothing measured the times are in plain rounds.
    [[nodiscard]] static double verify_estimate(const RowsCost& cost, std::int32_t k) noexcept {
        double plain = cost.plain_ms;
        for (std::int32_t w = 1; plain == 0.0 && w < ops::inkling2::kDSparkBlock; ++w) {
            if (const double v = cost.verify_ms[static_cast<std::size_t>(w)]; v > 0.0) { plain = v / (1.0 + kColumnPrior * w); }
        }
        if (plain == 0.0) { plain = 1.0; }
        if (k == 0) { return plain; }
        if (cost.verify_ms[static_cast<std::size_t>(k)] > 0.0) { return cost.verify_ms[static_cast<std::size_t>(k)]; }
        const auto at = [&](std::int32_t w) { return w == 0 ? plain : cost.verify_ms[static_cast<std::size_t>(w)]; };
        std::int32_t lo = k - 1;
        while (lo > 0 && at(lo) == 0.0) { --lo; }
        std::int32_t hi = k + 1;
        while (hi < ops::inkling2::kDSparkBlock && at(hi) == 0.0) { ++hi; }
        if (hi < ops::inkling2::kDSparkBlock) { return at(lo) + (at(hi) - at(lo)) * (k - lo) / (hi - lo); }
        double slope = kColumnPrior * plain;
        if (lo > 0) {
            std::int32_t below = lo - 1;
            while (below > 0 && at(below) == 0.0) { --below; }
            slope = std::max(0.0, (at(lo) - at(below)) / (lo - below));
        }
        return at(lo) + slope * (k - lo);
    }

    // A round of several rows (rows_eligible): their drafts, the round's width and one verify call. Empty when the
    // width verifies no draft (the caller decodes plainly; the drafts are spent).
    std::optional<PendingBatch> decode_dspark_rows(std::span<const SequenceHandle> sequences,
                                                   std::span<const runtime::RoundBudget> budgets,
                                                   runtime::TokenMaskProvider* masks, Clock::time_point start) {
        namespace ik            = ops::inkling2;
        const std::size_t n     = sequences.size();
        RowsCost& cost          = rows_cost(sequences);
        const auto window       = static_cast<std::int32_t>(o_.draft_tokens);
        std::array<std::uint32_t, kMaximumConcurrency> index{}, order{};
        std::int32_t steps = 0, room = 0;
        for (std::size_t r = 0; r < n; ++r) {
            index[r] = lane_of(sequences[r]);
            order[r] = static_cast<std::uint32_t>(r);
            const Lane& lane = lanes_[index[r]];
            steps = std::max(steps, o_.draft_tokens_auto ? lane.step_cap : window);
            room  = std::max(room, static_cast<std::int32_t>(std::min<std::uint32_t>(budgets[r].generated_tokens_remaining,
                                                                                      ik::kDSparkBlock)) - 1);
        }
        // The rows in the call's order: lanes ascending, each lane's columns consecutive.
        std::sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(n),
                  [&](std::uint32_t a, std::uint32_t b) { return index[a] < index[b]; });
        // Drafts no row's budget can take are not drafted.
        steps = std::min({steps, room, ik::kMarkovSteps});
        if (steps < 1) { return std::nullopt; }
        // Each row's block rows F .. F + 7 go to the full layer's pages first; the drafts are rows in the call's order.
        std::array<ik::DSparkRound, kMaximumConcurrency> rounds{};
        for (std::size_t b = 0; b < n; ++b) {
            const std::uint32_t lane = index[order[b]];
            const auto F             = static_cast<std::int32_t>(lanes_[lane].processed);
            map_lane_pages(lane, F + ik::kDSparkBlock);
            rounds[b] = {F, lanes_[lane].tokens[static_cast<std::size_t>(F)], static_cast<std::int32_t>(lane)};
        }
        drafter_->stage(std::span<const ik::DSparkRound>(rounds.data(), n));
        // The graph first (a first use captures it), so the events time the launch alone.
        DecodeGraphExecutable& draft = rows_draft_graph(static_cast<std::int32_t>(n), steps);
        CUDA_CHECK(cudaEventRecord(round_events_[0], device_.stream));
        draft.launch(device_.stream);
        CUDA_CHECK(cudaEventRecord(round_events_[1], device_.stream));
        {
            const auto wait = Clock::now();
            CUDA_CHECK(cudaStreamSynchronize(device_.stream));
            wait_ns_ += elapsed_ns(wait);
        }
        // Each row's drafts and confidences (draft row b is the call's b-th row).
        std::array<std::array<TokenId, ik::kMarkovSteps>, kMaximumConcurrency> drafts{};
        std::array<std::array<float, ik::kMarkovSteps>, kMaximumConcurrency> probabilities{};
        for (std::size_t b = 0; b < n; ++b) {
            for (std::int32_t i = 0; i < steps; ++i) {
                drafts[b][static_cast<std::size_t>(i)]        = drafter_->host_tokens(static_cast<std::int32_t>(b))[i];
                probabilities[b][static_cast<std::size_t>(i)] = drafter_->host_probabilities(static_cast<std::int32_t>(b))[i];
            }
        }
        // The round's columns, n (K + 1), and each row's share of its n x K drafts (a fixed --draft-tokens: K = all it
        // drafted, each row).
        //
        // The n x K drafts are dealt one at a time to the row whose expected tokens the next one raises most
        // (draft_length.h, each row at most the drafts it has, capped by its budget), not K each: the same columns, so
        // the same verify cost and graph (CallPlan::group_width), more tokens where the rows' confidences differ
        // (2026-11-26, the owner's configuration: +3.5 to +6.6 % expected tokens at 2K, +7.6 to +12.3 % at 100K). Every
        // row's expected gain falls draft by draft (each draft's reach is the one before it times an acceptance), so a
        // greedy deal is the best at each count, and one pass to n x steps gives them all: K maximizes the dealt tokens
        // at n x K over the verify call's device time at K.
        std::int32_t K = steps;
        std::array<std::int32_t, kMaximumConcurrency> own{};
        for (std::size_t b = 0; b < n; ++b) { own[b] = K; }
        if (o_.draft_tokens_auto) {
            const auto row_tokens = [&](std::size_t b, std::int32_t k) {
                const std::size_t r = order[b];
                const auto licensed = static_cast<std::int32_t>(budgets[r].generated_tokens_remaining);
                return runtime::speculative::expected_draft_tokens(
                    lanes_[index[r]].calibration,
                    std::span<const float>(probabilities[b].data(), static_cast<std::size_t>(steps)),
                    std::min(k, licensed - 1));
            };
            // [k]: the expected tokens at n x k drafts and each row's share of them.
            std::array<double, ik::kDSparkBlock> tokens{};
            std::array<std::array<std::int32_t, kMaximumConcurrency>, ik::kDSparkBlock> share{};
            std::array<std::int32_t, kMaximumConcurrency> held{};
            for (std::size_t b = 0; b < n; ++b) { tokens[0] += row_tokens(b, 0); }
            for (std::int32_t k = 1; k <= steps; ++k) {
                tokens[static_cast<std::size_t>(k)] = tokens[static_cast<std::size_t>(k) - 1];
                for (std::size_t d = 0; d < n; ++d) {
                    double gain      = -1.0;
                    std::size_t pick = 0;
                    for (std::size_t b = 0; b < n; ++b) {
                        if (held[b] >= steps) { continue; }
                        if (const double g = row_tokens(b, held[b] + 1) - row_tokens(b, held[b]); g > gain) {
                            gain = g;
                            pick = b;
                        }
                    }
                    ++held[pick]; // n x k <= n x steps: a row always has room
                    tokens[static_cast<std::size_t>(k)] += gain;
                }
                share[static_cast<std::size_t>(k)] = held;
            }
            double best = -1.0;
            for (std::int32_t k = 0; k <= steps; ++k) {
                if (const double score = tokens[static_cast<std::size_t>(k)] / verify_estimate(cost, k); score > best) {
                    best = score;
                    K    = k;
                }
            }
            // One width more every kProbeInterval rounds, K each, keeps the longer widths' cost and acceptance measured
            // and every row's acceptance (a row dealt no drafts observes none).
            if (++cost.rounds % kProbeInterval == 0) {
                K = std::min(K + 1, steps);
                for (std::size_t b = 0; b < n; ++b) { own[b] = K; }
            } else {
                own = share[static_cast<std::size_t>(K)];
                double uniform = 0.0;
                for (std::size_t b = 0; b < n; ++b) { uniform += row_tokens(b, K); }
                if (K != 0 && uniform > 0.0) {
                    cost.own_ratio += tokens[static_cast<std::size_t>(K)] / uniform;
                    ++cost.own_rounds;
                }
            }
            if (K != 0 && inkling2::testing::skewed_dealing()) {
                // The test seam: the n x K drafts from a rotating row on, each row all it drafted until none are left.
                std::int32_t left = static_cast<std::int32_t>(n) * K;
                for (std::size_t i = 0; i < n; ++i) {
                    const std::size_t b = (cost.rounds + i) % n;
                    own[b]              = std::min(steps, left);
                    left -= own[b];
                }
            }
            // A row's next draft: 2 past what it verified, as a one-row round's.
            std::int32_t widest = 0;
            for (std::size_t b = 0; b < n; ++b) {
                Lane& lane    = lanes_[index[order[b]]];
                lane.step_cap = std::clamp(own[b] + 2, std::min(2, window), window);
                ++lane.policy_rounds;
                widest = std::max(widest, own[b]);
            }
            if (K != 0 && widest == steps && steps < window) { ++cost.saturated; }
        } else {
            ++cost.rounds;
        }
        ++cost.widths[static_cast<std::size_t>(K)];
        report_rows(n, context_bucket(longest_context(sequences)), cost);
        if (K == 0) { return std::nullopt; }
        inkling2::testing::count_dealing(std::span<const std::int32_t>(own.data(), n), steps);
        // The verify call: each row's pending input and its drafts at F .. F + own, in lane order, n (K + 1) columns.
        CallPlan call;
        call.first.assign(lanes_.size(), 0); // every lane's entry: the call's graph is the same whichever lanes it has
        call.group_width = ik::kDSparkBlock; // a row's widest: one graph for every dealing of the round's columns
        std::array<std::uint32_t, kMaxSampleRows> sample_lanes{}, providers{};
        std::array<std::int32_t, kMaxSampleRows> logit_rows{}, positions{};
        std::array<std::int32_t, kMaximumConcurrency> offset{};
        for (std::size_t b = 0; b < n; ++b) {
            const std::size_t r      = order[b];
            const std::uint32_t lane = index[r];
            const auto F             = static_cast<std::int32_t>(lanes_[lane].processed);
            offset[r]                = static_cast<std::int32_t>(call.tokens.size());
            for (std::int32_t i = 0; i < own[b] + 1; ++i) {
                const auto column = static_cast<std::size_t>(offset[r] + i);
                call.tokens.push_back(i == 0 ? lanes_[lane].tokens[static_cast<std::size_t>(F)]
                                             : drafts[b][static_cast<std::size_t>(i - 1)]);
                call.lane.push_back(static_cast<std::int32_t>(lane));
                call.position.push_back(F + i);
                sample_lanes[column] = lane;
                providers[column]    = static_cast<std::uint32_t>(r);
                logit_rows[column]   = static_cast<std::int32_t>(column);
                positions[column]    = F + i;
            }
        }
        upload(call);
        const auto T = static_cast<std::size_t>(call.columns());
        // Every column samples under its row's reasoning ban (read before the masks are filled, as a one-row round
        // reads it) and mask.
        std::array<runtime::ReasoningBan, kMaximumConcurrency> bans{};
        bool constrained = false;
        for (std::size_t r = 0; r < n && masks != nullptr; ++r) {
            bans[r]     = masks->reasoning_ban(r);
            constrained = constrained || masks->constrained(r);
        }
        stage_sample({sample_lanes.data(), T}, {logit_rows.data(), T}, ops::kSamplePurposeDecode, masks,
                     {positions.data(), T}, false, {providers.data(), T});
        DecodeGraphExecutable& verify = rows_verify_graph(static_cast<std::int32_t>(n), K + 1); // n (K + 1) columns
        CUDA_CHECK(cudaEventRecord(round_events_[2], device_.stream));
        // The forward's graph, then (a constrained row's masks filled while it runs: column i samples under the mask of
        // its position after its drafts 0 .. i - 1) the sample of every column.
        verify.launch(device_.stream);
        std::array<std::uint64_t, kMaximumConcurrency> dead{};
        for (std::size_t b = 0; b < n && constrained; ++b) {
            const std::size_t r = order[b];
            if (!masks->constrained(r)) { continue; }
            fill_masks(*masks, r, static_cast<std::size_t>(offset[r]),
                       std::span<const TokenId>(drafts[b].data(), static_cast<std::size_t>(own[b])));
            dead[r] = mask_dead_[r];
        }
        launch_sample(T, {logit_rows.data(), T}, ops::kSamplePurposeDecode);
        CUDA_CHECK(cudaEventRecord(round_events_[3], device_.stream));
        finish_sample(T, true);
        float draft_ms = 0, gap_ms = 0, verify_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&draft_ms, round_events_[0], round_events_[1]));
        CUDA_CHECK(cudaEventElapsedTime(&gap_ms, round_events_[1], round_events_[2]));
        CUDA_CHECK(cudaEventElapsedTime(&verify_ms, round_events_[2], round_events_[3]));
        // A round that captured a graph (or evicted one) is no sample of the cost line.
        if (!round_captured_) { ewma(cost.verify_ms[static_cast<std::size_t>(K)], verify_ms); }
        // Each row: its accepted prefix, as a one-row round decides it.
        std::array<runtime::PendingRowFailure, kMaximumConcurrency> row_failures{};
        std::int32_t committed = 0;
        for (std::size_t b = 0; b < n; ++b) {
            const std::size_t r       = order[b];
            const std::uint32_t idx   = index[r];
            Lane& lane                = lanes_[idx];
            const auto F              = static_cast<std::int32_t>(lane.processed);
            const auto o              = static_cast<std::size_t>(offset[r]);
            const auto licensed       = static_cast<std::int32_t>(budgets[r].generated_tokens_remaining);
            std::int32_t a            = 0;
            while (a < own[b] && drafts[b][static_cast<std::size_t>(a)] == sampled_tokens_[o + static_cast<std::size_t>(a)] &&
                   failures_[o + static_cast<std::size_t>(a)] == runtime::PendingRowFailure::None) {
                ++a;
            }
            lane.calibration.observe(std::span<const float>(probabilities[b].data(), static_cast<std::size_t>(steps)),
                                     static_cast<std::uint32_t>(own[b]), static_cast<std::uint32_t>(a), kAcceptanceWeight);
            // The row's round ends at the first reasoning close it draws, and at its budget.
            if (const runtime::ReasoningBan& ban = bans[r]; ban.active()) {
                for (std::int32_t i = 0; i < a; ++i) {
                    if (sampled_tokens_[o + static_cast<std::size_t>(i)] == ban.close) {
                        a = i;
                        break;
                    }
                }
            }
            const std::int32_t kept = std::min(a, licensed - 1);
            runtime::PendingRowFailure failure = runtime::PendingRowFailure::None;
            for (std::int32_t i = 0; i <= kept; ++i) {
                if (failures_[o + static_cast<std::size_t>(i)] != runtime::PendingRowFailure::None) {
                    failure = failures_[o + static_cast<std::size_t>(i)];
                    break;
                }
            }
            if (failure == runtime::PendingRowFailure::None && (dead[r] & ((std::uint64_t{2} << kept) - 1U)) != 0) {
                failure = runtime::PendingRowFailure::ConstraintDeadEnd;
            }
            if (const std::uint32_t at = unpackable_seen_[idx]; at != kNoPosition) {
                if (at <= static_cast<std::uint32_t>(F + kept)) {
                    if (failure == runtime::PendingRowFailure::None) { failure = runtime::PendingRowFailure::UnpackableKv; }
                } else {
                    clear_unpackable(idx);
                }
            }
            if (failure != runtime::PendingRowFailure::None && failure != runtime::PendingRowFailure::ConstraintDeadEnd) {
                lane.prefix.withdraw = true; // as finish_sample does for a plain row
            }
            row_failures[r] = failure;
            ++lane.rounds;
            lane.drafted += static_cast<std::uint64_t>(own[b]);
            lane.accepted += static_cast<std::uint64_t>(kept);
            for (std::int32_t i = 0; i < kept; ++i) { ++lane.accepted_at[static_cast<std::size_t>(i)]; }
            lane.draft_ms += draft_ms, lane.gap_ms += gap_ms, lane.verify_ms += verify_ms;
            row_counts_[r]     = kept + 1;
            pending_firsts_[r] = static_cast<std::uint32_t>(F);
            std::copy_n(sampled_tokens_.begin() + static_cast<std::ptrdiff_t>(o), kept + 1,
                        round_tokens_.begin() + static_cast<std::ptrdiff_t>(r * kRowStride));
            committed += kept + 1;
        }
        std::copy_n(row_failures.begin(), n, failures_.begin());
        inkling2::testing::count_rows_drafted(n);
        pending_draft_ = {true};
        const std::uint64_t transaction = ++transaction_;
        const std::uint64_t ns          = elapsed_ns(start);
        if (!round_captured_) {
            ewma(cost.draft_wall, static_cast<double>(ns) * 1e-6);
            ewma(cost.draft_tokens, static_cast<double>(committed));
        }
        for (std::size_t r = 0; r < n; ++r) {
            lanes_[index[r]].decode_ns += ns;
            lanes_[index[r]].decode_share_ns += ns / n;
        }
        pending_      = transaction;
        pending_kind_ = PendingKind::Decode;
        return ContractAccess::batch(this, transaction, sequences, {round_tokens_.data(), n * kRowStride},
                                     timing_since(start), {failures_.data(), n}, {row_counts_.data(), n}, kRowStride);
    }

    // The captured drafts of `rows` rows and `steps` steps (a round of several rows), the most recently used
    // kRowsGraphs of them kept.
    DecodeGraphExecutable& rows_draft_graph(std::int32_t rows, std::int32_t steps) {
        const std::uint64_t key = (1ULL << 63) | (1ULL << 60) | static_cast<std::uint64_t>(rows) << 8 |
                                  static_cast<std::uint32_t>(steps);
        return rows_graph(key, [&] {
            drafter_->draft(steps, device_.stream, rows);
            drafter_->download(steps, device_.stream, rows);
        });
    }

    // The captured forward of a verify call of `rows` rows of `width` columns each (its call plan uploaded before the
    // launch; the sample follows it, eagerly, so a constrained row's masks can be filled meanwhile). Its topology
    // depends on those counts only: which lanes the rows are, their tokens and positions are device data (each lane's
    // columns stay one attention group, below the conv ring's and window's wide paths, so no launch names a lane).
    DecodeGraphExecutable& rows_verify_graph(std::int32_t rows, std::int32_t width) {
        const std::uint64_t key = (1ULL << 63) | (1ULL << 61) | kForwardOnlyGraph | static_cast<std::uint64_t>(rows) << 8 |
                                  static_cast<std::uint32_t>(width);
        return rows_graph(key, [&] { run_decode_call(rows * width); });
    }

    // A graph of rounds of several rows by key, captured on first use; the least recently used is destroyed once
    // kRowsGraphs are held (the graphs' allowance counts kRowsGraphs).
    template <class Body>
    DecodeGraphExecutable& rows_graph(std::uint64_t key, Body&& body) {
        if (const auto used = std::find(rows_graph_order_.begin(), rows_graph_order_.end(), key);
            used != rows_graph_order_.end()) {
            rows_graph_order_.erase(used);
            rows_graph_order_.push_back(key);
            return graphs_.at(key);
        }
        while (rows_graph_order_.size() >= static_cast<std::size_t>(kRowsGraphs)) {
            graphs_.erase(rows_graph_order_.front());
            rows_graph_order_.pop_front();
            graph_executables_.fetch_sub(1, std::memory_order_relaxed);
            round_captured_ = true;
            ++rows_evictions_;
        }
        ++rows_captures_;
        DecodeGraphDefinition definition;
        definition.capture(device_.stream, body);
        DecodeGraphExecutable exec;
        instantiate_graph(exec, definition);
        rows_graph_order_.push_back(key);
        return graphs_.emplace(key, std::move(exec)).first->second;
    }

    // The captured draft of `steps` drafts (its round words copied in from pinned memory, the drafts copied out).
    DecodeGraphExecutable& draft_graph(std::int32_t steps) {
        auto found = draft_graphs_.find(steps);
        if (found == draft_graphs_.end()) {
            DecodeGraphDefinition definition;
            definition.capture(device_.stream, [&] {
                drafter_->draft(steps, device_.stream);
                drafter_->download(steps, device_.stream);
            });
            DecodeGraphExecutable exec;
            instantiate_graph(exec, definition);
            found = draft_graphs_.emplace(steps, std::move(exec)).first;
        }
        return found->second;
    }

    // The captured verify call of one lane and `columns` columns (its call plan, configs and positions uploaded
    // before the launch): the forward, the context rows and, with `sample`, the sample of every column (a constrained
    // row samples after its masks, filled while this runs).
    DecodeGraphExecutable& verify_graph(std::uint32_t lane, std::int32_t columns, bool sample = true) {
        const std::uint64_t key = (1ULL << 63) | (sample ? 0ULL : kForwardOnlyGraph) |
                                  static_cast<std::uint64_t>(lane) << 32 | static_cast<std::uint32_t>(columns);
        auto found = graphs_.find(key);
        if (found == graphs_.end()) {
            std::vector<std::int32_t> identity(static_cast<std::size_t>(columns));
            for (std::int32_t i = 0; i < columns; ++i) { identity[static_cast<std::size_t>(i)] = i; }
            DecodeGraphDefinition definition;
            definition.capture(device_.stream, [&] {
                run_decode_call(columns);
                if (sample) { launch_sample(static_cast<std::size_t>(columns), identity, ops::kSamplePurposeDecode); }
            });
            DecodeGraphExecutable exec;
            instantiate_graph(exec, definition);
            found = graphs_.emplace(key, std::move(exec)).first;
        }
        return found->second;
    }

    // A request's DSpark rounds at its end (Debug): their counts and the mean device time of a round's parts.
    void report_rounds(const Lane& lane) const noexcept {
        if (!drafter_ || lane.rounds == 0 || !o_.diagnostics.callback) { return; }
        try {
            const double n = static_cast<double>(lane.rounds);
            char text[256];
            std::snprintf(text, sizeof(text),
                          "Inkling 2 DSpark: %llu rounds (%llu plain), %llu of %llu drafts accepted; a round's draft "
                          "%.3f ms, host gap %.3f ms, verify call %.3f ms (device means)",
                          static_cast<unsigned long long>(lane.rounds), static_cast<unsigned long long>(lane.fallback),
                          static_cast<unsigned long long>(lane.accepted), static_cast<unsigned long long>(lane.drafted),
                          lane.draft_ms / n, lane.gap_ms / n, lane.verify_ms / n);
            o_.diagnostics.callback(Diagnostic{.level = DiagnosticLevel::Debug, .message = text});
        } catch (...) {}
    }

    // A lane's speculative counters and statistics for the Engine.
    void speculative_of(const Lane& lane, CommitRowResult& row) const {
        if (!drafter_) { return; }
        row.speculative_counters = {lane.rounds, lane.drafted, lane.accepted, lane.fallback};
        SpeculativeStats& s      = row.speculative;
        s.backend                = SpeculativeBackend::DSpark;
        s.enabled                = true;
        s.draft_window           = o_.draft_tokens;
        s.rounds                 = lane.rounds;
        s.drafted_tokens         = lane.drafted;
        s.accepted_tokens        = lane.accepted;
        s.fallback_steps         = lane.fallback;
        s.accepted_per_position.assign(lane.accepted_at.begin(), lane.accepted_at.begin() + o_.draft_tokens);
    }

    // Samples one token a row from the logits rows `logit_rows` (each row's lane `lanes[row]`), into
    // sampled_tokens_ and failures_ (synchronous).
    void sample(std::span<const SequenceHandle> rows, std::span<const std::uint32_t> lanes, std::span<const std::int32_t> logit_rows,
                std::int32_t purpose, runtime::TokenMaskProvider* masks) {
        enqueue_sample(lanes, logit_rows, purpose, masks);
        finish_sample(rows.size());
    }
    void sample(std::span<const SequenceHandle> rows, std::span<const std::uint32_t> lanes, std::int32_t logit_row,
                std::int32_t purpose, runtime::TokenMaskProvider* masks) {
        const std::int32_t r = logit_row;
        sample(rows, lanes, std::span<const std::int32_t>(&r, 1), purpose, masks);
    }

    // A constrained row's masks: its matcher's legal tokens at its committed position and at each draft's position
    // after it (the provider's row `row`), into mask slots `slot`.. and uploaded; its dead positions into
    // mask_dead_[row].
    void fill_masks(runtime::TokenMaskProvider& masks, std::size_t row, std::size_t slot, std::span<const TokenId> drafts) {
        const auto words = static_cast<std::size_t>(mask_words(static_cast<std::uint64_t>(vocab_)));
        if (slot + drafts.size() + 1 > static_cast<std::size_t>(mask_slots(o_))) {
            throw std::logic_error("Inkling 2: a constrained round needs more mask slots than it has");
        }
        auto* host = static_cast<std::uint32_t*>(host_masks_->data()) + slot * words;
        const std::span<std::uint32_t> span(host, (drafts.size() + 1) * words);
        mask_dead_[row] = masks.fill(row, drafts, span);
        upload_pinned(static_cast<std::uint32_t*>(masks_.p) + slot * words, host, span.size_bytes(), device_.stream);
        masks.uploaded(row, span.size_bytes());
    }

    [[nodiscard]] ops::SamplingMask mask_slot(std::size_t slot) const noexcept {
        const auto words = static_cast<std::size_t>(mask_words(static_cast<std::uint64_t>(vocab_)));
        return {static_cast<const std::uint32_t*>(masks_.p) + slot * words, static_cast<std::int32_t>(words)};
    }

    // ---------------------------------------------------------------- token constraints (TokenConstraint)
    // A request's plan: each step's tokens in the public domain, at most kMaximumConstraintChoices; a one-token step is
    // forced; at most kMaximumConstraintSets distinct multi-token sets (constraint_words' columns).
    std::shared_ptr<const ConstraintPlan> compile_constraint(const TokenConstraint& constraint) {
        if (constraint.empty()) { return nullptr; }
        auto plan = std::make_shared<ConstraintPlan>();
        for (const std::vector<TokenId>& step : constraint.steps) {
            if (step.empty() || step.size() > kMaximumConstraintChoices) {
                throw std::invalid_argument("constraint step has an invalid token count");
            }
            for (const TokenId id : step) {
                if (id < 0 || id >= vocab_) { throw std::invalid_argument("constraint token is outside the public token domain"); }
            }
            if (step.size() == 1) {
                plan->steps.push_back(-2 - step.front());
                continue;
            }
            auto set = std::find(plan->sets.begin(), plan->sets.end(), step);
            if (set == plan->sets.end()) {
                if (plan->sets.size() == kMaximumConstraintSets) {
                    throw std::invalid_argument("constraint uses too many distinct token sets");
                }
                set = plan->sets.insert(plan->sets.end(), step);
            }
            plan->steps.push_back(static_cast<std::int32_t>(set - plan->sets.begin()));
        }
        for (const std::vector<TokenId>& observed : constraint.observe) {
            if (observed.size() > static_cast<std::size_t>(ops::kTokenObserveTokens)) {
                throw std::invalid_argument("constraint step observes too many tokens");
            }
            for (const TokenId id : observed) {
                if (id < 0 || id >= vocab_) { throw std::invalid_argument("observed token is outside the public token domain"); }
            }
        }
        plan->observe        = constraint.observe;
        plan->capture_logits = constraint.capture_logits;
        plan->serial         = next_constraint_serial_++;
        return plan;
    }

    // The staged rows' descriptors, a constrained lane's sets (once a request) and the observe lists, uploaded before
    // the sample (stage_sample's last step). A row samples output step (its generated tokens).
    void stage_constraints(std::span<const std::uint32_t> lanes) {
        staged_constraint_      = false;
        staged_constraint_rows_ = lanes.size();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            staged_descriptors_[row] = -1;
            const Lane& lane         = lanes_[lanes[row]];
            if (!lane.constraint) { continue; }
            const std::size_t step = lane.tokens.size() - lane.prompt_tokens;
            if (step >= lane.constraint->steps.size()) {
                throw std::logic_error("Inkling 2: a constrained row samples past its last step");
            }
            const std::int32_t d     = lane.constraint->steps[step];
            staged_descriptors_[row] = d >= 0 ? static_cast<std::int32_t>(lanes[row] * kMaximumConstraintSets) + d : d;
            staged_steps_[row]       = step;
            staged_constraint_       = true;
        }
        if (!staged_constraint_) { return; }
        const ConstraintWords w = constraint_words(static_cast<std::uint32_t>(lanes_.size()));
        auto* host              = static_cast<std::int32_t*>(constraint_host_->data());
        auto* device            = static_cast<std::int32_t*>(constraint_.p);
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t index = lanes[row];
            const Lane& lane          = lanes_[index];
            if (!lane.constraint || constraint_serials_[index] == lane.constraint->serial) { continue; }
            const std::size_t first = static_cast<std::size_t>(index) * kMaximumConstraintSets;
            std::int32_t* choices   = host + w.choices + first * ops::kTokenConstraintChoices;
            std::int32_t* counts    = host + w.counts + first;
            std::fill(choices, choices + kMaximumConstraintSets * ops::kTokenConstraintChoices, 0);
            std::fill(counts, counts + kMaximumConstraintSets, 0);
            for (std::size_t set = 0; set < lane.constraint->sets.size(); ++set) {
                std::copy(lane.constraint->sets[set].begin(), lane.constraint->sets[set].end(),
                          choices + set * ops::kTokenConstraintChoices);
                counts[set] = static_cast<std::int32_t>(lane.constraint->sets[set].size());
            }
            upload_pinned(device + w.choices + first * ops::kTokenConstraintChoices, choices,
                          4ULL * kMaximumConstraintSets * ops::kTokenConstraintChoices, device_.stream);
            upload_pinned(device + w.counts + first, counts, 4ULL * kMaximumConstraintSets, device_.stream);
            constraint_serials_[index] = lane.constraint->serial;
        }
        std::copy_n(staged_descriptors_.begin(), lanes.size(), host + w.descriptors);
        upload_pinned(device + w.descriptors, host + w.descriptors, 4ULL * lanes.size(), device_.stream);
        std::int32_t* observe = host + w.observe;
        std::fill(observe, observe + lanes.size() * ops::kTokenObserveTokens, -1);
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const Lane& lane = lanes_[lanes[row]];
            if (staged_descriptors_[row] == -1 || staged_steps_[row] >= lane.constraint->observe.size()) { continue; }
            const std::vector<TokenId>& tokens = lane.constraint->observe[staged_steps_[row]];
            std::copy(tokens.begin(), tokens.end(), observe + row * ops::kTokenObserveTokens);
        }
        upload_pinned(device + w.observe, observe, 4ULL * lanes.size() * ops::kTokenObserveTokens, device_.stream);
    }

    // The staged rows' constraint on `logits` (the sample's operand, rows in staged order), before the sampler reads it:
    // a capturing row's logits copied out first, then each constrained column masked to its step's tokens and its
    // records written (ops::constrain_logits). Never inside a captured graph (a constrained round samples eagerly).
    void constrain_staged(float* logits, std::size_t rows) {
        const ConstraintWords w = constraint_words(static_cast<std::uint32_t>(lanes_.size()));
        auto* device            = static_cast<std::int32_t*>(constraint_.p);
        const auto columns      = static_cast<std::int32_t>(rows);
        captured_rows_.clear();
        for (std::size_t row = 0; row < rows; ++row) {
            const Lane& lane = lanes_[sample_lanes_[row]];
            if (staged_descriptors_[row] == -1 || !lane.constraint->capture_logits) { continue; }
            captured_rows_.resize(rows * static_cast<std::size_t>(vocab_));
            CUDA_CHECK(cudaMemcpyAsync(captured_rows_.data() + row * vocab_, logits + row * vocab_,
                                       static_cast<std::size_t>(vocab_) * 4, cudaMemcpyDeviceToHost, device_.stream));
        }
        const auto sets = static_cast<std::int32_t>(w.sets);
        Tensor in(logits, DType::FP32, {vocab_, columns});
        const Tensor choices(device + w.choices, DType::I32, {ops::kTokenConstraintChoices, sets});
        const Tensor counts(device + w.counts, DType::I32, {sets});
        const Tensor descriptors(device + w.descriptors, DType::I32, {columns});
        Tensor records(device + w.records, DType::FP32, {ops::kTokenConstraintRecord, columns});
        const Tensor observe(device + w.observe, DType::I32, {ops::kTokenObserveTokens, columns});
        Tensor observed(device + w.observed, DType::FP32, {ops::kTokenObserveRecord, columns});
        ops::constrain_logits(in, nullptr, descriptors, choices, counts, vocab_, records, &observe, &observed,
                              device_.stream);
    }

    // After the call's synchronize: each constrained row's draw (its step's restricted distribution, mass, observed
    // log-probabilities and captured row) as its lane's round draw, committed with its token.
    void take_constraint_draws(std::size_t rows) {
        const ConstraintWords w = constraint_words(static_cast<std::uint32_t>(lanes_.size()));
        std::vector<float> records(rows * ops::kTokenConstraintRecord), observed(rows * ops::kTokenObserveRecord);
        CUDA_CHECK(cudaMemcpy(records.data(), static_cast<const std::int32_t*>(constraint_.p) + w.records,
                              4ULL * records.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(observed.data(), static_cast<const std::int32_t*>(constraint_.p) + w.observed,
                              4ULL * observed.size(), cudaMemcpyDeviceToHost));
        for (std::size_t row = 0; row < rows; ++row) {
            const std::int32_t d = staged_descriptors_[row];
            if (d == -1) { continue; }
            Lane& lane              = lanes_[sample_lanes_[row]];
            const ConstraintPlan& p = *lane.constraint;
            const std::int32_t set  = d >= 0 ? d - static_cast<std::int32_t>(sample_lanes_[row] * kMaximumConstraintSets) : -1;
            const std::size_t count = set >= 0 ? p.sets[static_cast<std::size_t>(set)].size() : 1U;
            const float* record     = records.data() + row * ops::kTokenConstraintRecord;
            ConstrainedDraw draw{.token         = sampled_tokens_[row],
                                 .probabilities = std::vector<float>(record, record + count),
                                 .mass          = record[ops::kTokenConstraintChoices]};
            if (staged_steps_[row] < p.observe.size()) {
                const float* values = observed.data() + row * ops::kTokenObserveRecord;
                draw.observed_logprobs.assign(values, values + p.observe[staged_steps_[row]].size());
                draw.top_logprob = values[ops::kTokenObserveTokens];
            }
            if (p.capture_logits) {
                draw.logits.assign(captured_rows_.begin() + static_cast<std::ptrdiff_t>(row * vocab_),
                                   captured_rows_.begin() + static_cast<std::ptrdiff_t>((row + 1) * vocab_));
            }
            lane.constraint_round = {std::move(draw)};
        }
    }

    // The staged constrained rows' masks (stage_sample), filled while the call's forward runs: every caller launches
    // the forward first, then this, then the sample.
    void fill_staged_masks(runtime::TokenMaskProvider* masks) {
        for (std::size_t i = 0; i < staged_masks_; ++i) { fill_masks(*masks, staged_rows_[i], staged_rows_[i], {}); }
        staged_masks_ = 0;
    }

    // The rows' configs and positions to pinned memory, and the logits rows gathered: everything a sample needs
    // before its launch (outside a graph). A constrained row's config names its mask slot; fill_staged_masks fills it.
    // `positions`: each row's draw key (a verify call's columns); empty: the lane's. `mask_row_zero`: every row takes the
    // masks' row 0 (a verify call's columns of one row), whose masks decode_dspark fills into the columns' slots.
    // `providers`: each row's row of the masks (a verify call's columns of several rows, whose masks
    // decode_dspark_rows fills); empty: the row itself.
    void stage_sample(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> logit_rows, std::int32_t purpose,
                      runtime::TokenMaskProvider* masks, std::span<const std::int32_t> positions = {},
                      bool mask_row_zero = false, std::span<const std::uint32_t> providers = {}) {
        if (lanes.size() > static_cast<std::size_t>(kMaxSampleRows)) { throw std::logic_error("Inkling 2: too many sample rows"); }
        const bool verify = mask_row_zero || !providers.empty();
        staged_masks_     = 0;
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (!verify) { mask_dead_[row] = 0; }
            const Lane& lane                       = lanes_[lanes[row]];
            const ResolvedSamplingParameters& s    = lane.options.sampling;
            ops::SamplingConfig config;
            config.temperature       = s.temperature;
            config.top_k             = s.top_k;
            config.top_p             = s.top_p;
            config.min_p             = s.min_p;
            config.presence_penalty  = s.presence_penalty;
            config.frequency_penalty = s.frequency_penalty;
            config.seed              = s.seed;
            // The lossless route draws by Gumbel-max (lossless-route.md §3a).
            config.gumbel_max        = o_.lossless;
            // top_k 0 (no limit) with top_p 1 and min_p 0: the whole vocabulary (JM-40), except on the lossless route,
            // which keeps the 20-candidate cap and its bits.
            config.full_vocabulary = s.top_k == 0 && s.top_p >= 1.0F && s.min_p <= 0.0F && !o_.lossless;
            config.token_counts      = lane.penalties ? lane_counts(lanes[row]) : nullptr;
            nonfinite_[row]          = 0;
            config.nonfinite         = &nonfinite_[row];
            sample_lanes_[row]       = lanes[row];
            if (masks != nullptr) {
                const std::size_t provider_row  = mask_row_zero ? 0 : !providers.empty() ? providers[row] : row;
                const runtime::ReasoningBan ban = masks->reasoning_ban(provider_row);
                config.reasoning_close          = ban.close;
                config.reasoning_ends[0]        = ban.ends[0];
                config.reasoning_ends[1]        = ban.ends[1];
                config.reasoning_withheld       = ban.withheld;
                // A constrained row samples among its grammar's legal tokens (constrained-decoding.md §7.3).
                if (masks->constrained(provider_row)) {
                    if (!verify) { staged_rows_[staged_masks_++] = row; }
                    config.mask = mask_slot(row);
                }
            }
            host_configs_[row] = config;
            // Prefill keys the draw by the prompt's next position (the lane has processed the prompt), decode by its
            // last input token's (the lane's last token, processed by this call).
            host_positions_[row] = !positions.empty() ? positions[row]
                                   : static_cast<std::int32_t>(purpose == ops::kSamplePurposePrefill
                                                                   ? lane.processed
                                                                   : static_cast<std::uint32_t>(lane.tokens.size()) - 1U);
        }
        CUDA_CHECK(cudaMemcpyAsync(configs_.p, host_configs_, sizeof(ops::SamplingConfig) * lanes.size(),
                                   cudaMemcpyHostToDevice, device_.stream));
        CUDA_CHECK(cudaMemcpyAsync(positions_.p, host_positions_, 4 * lanes.size(), cudaMemcpyHostToDevice,
                                   device_.stream));
        // A verify call's columns are drafting rows', and a token-constrained row never drafts (dspark_round,
        // dspark_rows_round).
        if (verify) {
            staged_constraint_ = false;
        } else {
            stage_constraints(lanes);
        }
        (void)logit_rows;
    }

    // The last call's logits rows: the Forward's, or LosslessForward's on the lossless route.
    [[nodiscard]] const float* call_logits() const noexcept {
        return lossless_ ? lossless_->logits() : forward_.buffers().logits;
    }

    // The sample of rows whose logits are rows `logit_rows` of `logits` (null: the call's, call_logits()): one per row,
    // gathered when the rows are not the leading rows in order.
    void launch_sample(std::size_t rows, std::span<const std::int32_t> logit_rows, std::int32_t purpose,
                       const float* logits = nullptr) {
        if (logits == nullptr) { logits = call_logits(); }
        bool identity = true;
        for (std::size_t i = 0; i < rows; ++i) { identity = identity && logit_rows[i] == static_cast<std::int32_t>(i); }
        if (!identity) {
            // Plain rounds only (at most a row a lane): a verify call samples its columns in order.
            if (rows > kMaximumConcurrency) { throw std::logic_error("Inkling 2: a gathered sample of more than 8 rows"); }
            if (gathered_.bytes == 0) { gathered_ = DeviceBuffer(static_cast<std::size_t>(vocab_) * 4 * kMaximumConcurrency); }
            for (std::size_t i = 0; i < rows; ++i) {
                CUDA_CHECK(cudaMemcpyAsync(static_cast<float*>(gathered_.p) + i * vocab_,
                                           logits + static_cast<std::size_t>(logit_rows[i]) * vocab_,
                                           static_cast<std::size_t>(vocab_) * 4, cudaMemcpyDeviceToDevice, device_.stream));
            }
            logits = static_cast<const float*>(gathered_.p);
        }
        // A token-constrained row's logits are masked to its step's tokens (records written) before the sampler reads them.
        if (staged_constraint_) { constrain_staged(const_cast<float*>(logits), rows); }
        // Any lane may sample the whole vocabulary (top_k 0, JM-40) and the call is in captured graphs, so every call
        // but the lossless route's (which never does) declares such rows. A DSpark round of several rows samples its
        // columns 8 at a time: the sampler's own route for the widths plain decoding and a one-row round use, so each
        // column draws as it would there (and the workspace stays sized for 8 rows).
        for (std::size_t first = 0; first < rows; first += kMaximumConcurrency) {
            const auto count = static_cast<std::int32_t>(std::min<std::size_t>(kMaximumConcurrency, rows - first));
            const Tensor in(const_cast<float*>(logits) + first * static_cast<std::size_t>(vocab_), DType::FP32,
                            {vocab_, count});
            Tensor out(static_cast<std::int32_t*>(sampled_.p) + first, DType::I32, {count});
            const Tensor positions(static_cast<std::int32_t*>(positions_.p) + first, DType::I32, {count});
            ops::sample(in, out, vocab_, static_cast<const ops::SamplingConfig*>(configs_.p) + first, positions, purpose,
                        sampling_, device_.stream, !o_.lossless);
        }
        CUDA_CHECK(cudaMemcpyAsync(host_sampled_, sampled_.p, 4 * rows, cudaMemcpyDeviceToHost, device_.stream));
    }

    // After the call's forward was launched: its constrained rows' masks are filled while it runs.
    void enqueue_sample(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> logit_rows,
                        std::int32_t purpose, runtime::TokenMaskProvider* masks) {
        stage_sample(lanes, logit_rows, purpose, masks);
        fill_staged_masks(masks);
        launch_sample(lanes.size(), logit_rows, purpose);
    }

    // A down CTA of the MoE that waited past its bound for its job's h set the error word and wrote nothing, so a call
    // since the last check (this round's, or a prefill chunk that sampled nothing) may have left wrong expert output
    // in any bound lane's state: the requests holding lanes fail with a RecoverableExecutionError (the Engine keeps
    // serving: recover() releases them, queued requests wait) and what every bound lane published is withdrawn. The
    // CTA that gave up never counted itself, so the workspaces' cross-call counters are cleared before any later call
    // (the caller has synchronized the stream: every CTA of the stalled call has returned).
    void check_moe_error() {
        if (inkling2::testing::take_forced_stall()) {
            // A real stall's aftermath: the error word and the counters the stalled call left.
            ops::w4a4_moe::testing::leave_stalled_single_counters(forward_.buffers().moe, device_.stream);
            *forward_.buffers().moe.error |= ops::w4a4_moe::kErrorDownStalled; // through the kernels' own pointer
        }
        const std::uint32_t word = *static_cast<volatile std::uint32_t*>(moe_error_);
        if (word == 0) { return; }
        *moe_error_ = 0;
        forward_.clear_moe_counters(device_.stream);
        CUDA_CHECK(cudaStreamSynchronize(device_.stream));
        for (Lane& lane : lanes_) {
            if (lane.bound) { lane.prefix.withdraw = true; }
        }
        throw runtime::RecoverableExecutionError(
            "Inkling 2: an expert's down pass waited past its bound for its h (error word " + std::to_string(word) +
            "); the call's expert output is incomplete");
    }

    // `verify`: a DSpark verify call, whose rows with no packed form fail the request only at accepted positions
    // (decode_dspark decides once the accepted prefix is known).
    void finish_sample(std::size_t rows, bool verify = false) {
        const auto wait = Clock::now();
        CUDA_CHECK(cudaStreamSynchronize(device_.stream));
        wait_ns_ += elapsed_ns(wait);
        check_moe_error();
        if (unpackable_.p != nullptr) {
            CUDA_CHECK(cudaMemcpy(unpackable_seen_.data(), unpackable_.p, 4 * kMaximumConcurrency, cudaMemcpyDeviceToHost));
        }
        for (std::size_t row = 0; row < rows; ++row) {
            sampled_tokens_[row] = host_sampled_[row];
            const bool unpackable = !verify && unpackable_seen_[sample_lanes_[row]] != kNoPosition;
            // A plain row's one position with no legal token (its sampler drew the filler) fails it; a verify call's
            // dead positions count only where they are accepted (decode_dspark).
            const bool dead       = !verify && (mask_dead_[row] & 1U) != 0;
            // runtime::testing's fault injection reports a plain row as non-finite, as the other families do.
            const bool nonfinite  = nonfinite_[row] != 0 || (!verify && runtime::testing::take_nonfinite_fault());
            failures_[row]        = nonfinite            ? runtime::PendingRowFailure::NonFiniteLogits
                                    : unpackable         ? runtime::PendingRowFailure::UnpackableKv
                                    : dead               ? runtime::PendingRowFailure::ConstraintDeadEnd
                                                         : runtime::PendingRowFailure::None;
            // A request whose state may be wrong takes back what it published (release_prefix); a dead end's state is
            // right. A verify call's columns count only where kept (decode_dspark decides).
            if (!verify && failures_[row] != runtime::PendingRowFailure::None &&
                failures_[row] != runtime::PendingRowFailure::ConstraintDeadEnd && sample_lanes_[row] < lanes_.size()) {
                lanes_[sample_lanes_[row]].prefix.withdraw = true;
            }
        }
        if (!verify && staged_constraint_) { take_constraint_draws(rows); }
    }

    // A uniform decode round (every lane one input) as the graph of its (lanes, columns), captured on first use; the
    // call plan, configs and positions are uploaded before the launch.
    // `device_ms`: the round's forward and sample timed on the device (round_events_ 2 and 3), read after its sync.
    void decode_graph(std::int32_t lanes, std::int32_t columns, std::span<const std::uint32_t> rows,
                      std::span<const std::int32_t> logit_rows, runtime::TokenMaskProvider* masks,
                      float* device_ms = nullptr) {
        stage_sample(rows, logit_rows, ops::kSamplePurposeDecode, masks);
        // A round with a constrained row (a grammar's masks, a token constraint) runs the forward's graph alone, fills
        // the masks while it runs, then constrains and samples.
        const bool sample        = staged_masks_ == 0 && !staged_constraint_;
        const std::uint64_t key  = (sample ? 0ULL : kForwardOnlyGraph) | static_cast<std::uint64_t>(lanes) << 32 |
                                  static_cast<std::uint32_t>(columns);
        // A graph samples the call's rows in column order; rows given in another order are gathered eagerly.
        bool identity = true;
        for (std::size_t i = 0; i < rows.size(); ++i) { identity = identity && logit_rows[i] == static_cast<std::int32_t>(i); }
        if (!identity) {
            run_decode_call(columns);
            fill_staged_masks(masks);
            launch_sample(rows.size(), logit_rows, ops::kSamplePurposeDecode);
            finish_sample(rows.size());
            return;
        }
        auto found = graphs_.find(key);
        if (found == graphs_.end()) { found = capture_decode(lanes, columns, sample); }
        if (device_ms != nullptr) { CUDA_CHECK(cudaEventRecord(round_events_[2], device_.stream)); }
        found->second.launch(device_.stream);
        if (!sample) {
            fill_staged_masks(masks);
            launch_sample(rows.size(), logit_rows, ops::kSamplePurposeDecode);
        }
        if (device_ms != nullptr) { CUDA_CHECK(cudaEventRecord(round_events_[3], device_.stream)); }
        finish_sample(rows.size());
        if (device_ms != nullptr) { CUDA_CHECK(cudaEventElapsedTime(device_ms, round_events_[2], round_events_[3])); }
    }

    // The graph of a uniform decode call over `lanes` lanes with `columns` of them active (its rows the call's columns
    // in order). Its topology depends on those two counts only: tokens, lanes, positions, configs and the state are
    // device data the launch reads.
    // `sample` false: the forward alone (a constrained round samples after its masks, captured at its first use).
    std::map<std::uint64_t, DecodeGraphExecutable>::iterator capture_decode(std::int32_t lanes, std::int32_t columns,
                                                                            bool sample = true) {
        std::vector<std::int32_t> identity(static_cast<std::size_t>(columns));
        for (std::int32_t i = 0; i < columns; ++i) { identity[static_cast<std::size_t>(i)] = i; }
        DecodeGraphDefinition definition;
        definition.capture(device_.stream, [&] {
            run_decode_call(columns);
            if (sample) { launch_sample(static_cast<std::size_t>(columns), identity, ops::kSamplePurposeDecode); }
        });
        DecodeGraphExecutable exec;
        instantiate_graph(exec, definition);
        const std::uint64_t key = (sample ? 0ULL : kForwardOnlyGraph) | static_cast<std::uint64_t>(lanes) << 32 |
                                  static_cast<std::uint32_t>(columns);
        return graphs_.emplace(key, std::move(exec)).first;
    }

    // Instantiates and uploads a graph, adding the device memory the two took (free memory before and after, in the
    // driver's granularity) to the measured total (MemorySummary::cuda_graph_measured_bytes, beside the plan's
    // allowance, ProgramDevicePlan::graphs). The upload is the one its first launch would make.
    void instantiate_graph(DecodeGraphExecutable& exec, const DecodeGraphDefinition& definition) {
        std::size_t before = 0, after = 0, total = 0;
        round_captured_ = true; // the round's times are no sample of the cost line (decode_dspark_rows)
        CUDA_CHECK(cudaMemGetInfo(&before, &total));
        exec.instantiate(definition);
        exec.upload(device_.stream);
        CUDA_CHECK(cudaMemGetInfo(&after, &total));
        const std::uint64_t taken = before > after ? before - after : 0U;
        graph_bytes_.fetch_add(taken, std::memory_order_relaxed);
        const std::uint32_t count = graph_executables_.fetch_add(1, std::memory_order_relaxed) + 1U;
        if (o_.diagnostics.callback) {
            constexpr double kMiB = 1024.0 * 1024.0;
            char text[200];
            std::snprintf(text, sizeof(text),
                          "Inkling 2: CUDA graph %u took %.1f MiB; %u executables %.1f MiB of the %.0f MiB allowance",
                          count, static_cast<double>(taken) / kMiB, count,
                          static_cast<double>(graph_bytes_.load(std::memory_order_relaxed)) / kMiB,
                          static_cast<double>(plan_.graphs) / kMiB);
            o_.diagnostics.callback(Diagnostic{.level = DiagnosticLevel::Debug, .message = text});
        }
    }

    // Every uniform decode call's graph, at startup (design §8): (lanes, columns) for 1 <= columns <= lanes <= the
    // concurrency. A placeholder call fixes the Forward's widths for each capture; nothing runs.
    void capture_decode_graphs() {
        const auto max = static_cast<std::int32_t>(lanes_.size());
        for (std::int32_t lanes = 1; lanes <= max; ++lanes) {
            for (std::int32_t columns = 1; columns <= lanes; ++columns) {
                CallPlan call;
                call.first.assign(static_cast<std::size_t>(lanes), 0);
                for (std::int32_t c = 0; c < columns; ++c) {
                    call.tokens.push_back(0);
                    call.lane.push_back(lanes - columns + c);
                    call.position.push_back(0);
                }
                upload(call, false); // free lanes: their rows name the scratch page
                (void)capture_decode(lanes, columns);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(device_.stream));
    }

    // On the compute stream: the call's forward may still be running (the readout precedes the sample, which a token
    // constraint's mask would change).
    PromptReadout readout(const Lane& lane, std::int32_t column) {
        std::vector<float> row(static_cast<std::size_t>(vocab_));
        CUDA_CHECK(cudaMemcpyAsync(row.data(), call_logits() + static_cast<std::size_t>(column) * vocab_, row.size() * 4,
                                   cudaMemcpyDeviceToHost, device_.stream));
        CUDA_CHECK(cudaStreamSynchronize(device_.stream));
        double max = -std::numeric_limits<double>::infinity();
        std::int32_t top = 0;
        for (std::int32_t v = 0; v < vocab_; ++v) {
            if (row[v] > max) {
                max = row[v];
                top = v;
            }
        }
        double sum = 0;
        for (const float v : row) { sum += std::exp(static_cast<double>(v) - max); }
        const double lse = max + std::log(sum);
        PromptReadout out;
        for (const TokenId id : lane.options.readout_tokens) { out.logprobs.push_back(static_cast<float>(row[id] - lse)); }
        out.top_token   = top;
        out.top_logprob = static_cast<float>(max - lse);
        return out;
    }

    const Parameters& p_;
    const TextConfig& c_;
    DeviceContext& device_;
    ProgramOptions o_;
    ProgramDevicePlan plan_;
    Forward forward_;
    std::unique_ptr<LosslessForward> lossless_; // --lossless (scored windows; generation's cache form)
    std::unique_ptr<Drafter> drafter_; // --spec dspark
    std::unique_ptr<VisionTower> vision_; // --vision
    std::shared_ptr<const std::vector<PromptImage>> encoded_set_; // the prompt whose image the tower last encoded
    std::size_t encoded_index_          = 0;
    const std::uint16_t* encoded_rows_  = nullptr;
    std::map<std::int32_t, DecodeGraphExecutable> draft_graphs_;
    PendingDraft pending_draft_;
    std::array<std::int32_t, kMaximumConcurrency> row_counts_{};
    std::array<std::uint32_t, kMaximumConcurrency> pending_firsts_{};
    // By row count and context bucket (decode_dspark_rows, rows_cost).
    std::array<std::array<RowsCost, kContextBuckets>, kMaximumConcurrency + 1> rows_cost_{};
    std::deque<std::uint64_t> rows_graph_order_;               // rows_graph's keys, least recently used first
    bool round_captured_ = false; // the decode round captured or evicted a graph
    std::uint64_t rows_captures_ = 0, rows_evictions_ = 0; // rows_graph's (report_rows)
    // RAM pressure (start_ram_resize): the monitor, the reserve, the start's trim still to make, the resize under way,
    // the growth not yet reported, and the Engine's waker.
    struct RamResize {
        bool release            = false;
        bool trim               = false;
        std::uint64_t asked_at  = 0;
        std::uint64_t prefix    = 0;
    };
    std::unique_ptr<runtime::expert_offload::RamMonitor> ram_;
    std::uint64_t ram_reserve_ = 0;
    bool ram_trim_             = false;
    std::optional<RamResize> ram_resize_;
    std::uint64_t ram_grown_ = 0;
    std::mutex waker_mutex_;
    std::function<void()> waker_;
    std::int32_t vocab_ = 0;
    std::vector<Lane> lanes_;
    DeviceBuffer images_, counts_, configs_, positions_, sampled_, gathered_;
    DeviceBuffer score_hidden_; // ScoreOptions::prompt_hidden_states: BF16 [layers][H]
    StateView view_;
    WorkspaceArena sampling_;
    PinnedHostBuffer host_;
    std::int32_t* staging_                = nullptr;
    ops::SamplingConfig* host_configs_    = nullptr;
    std::int32_t* host_positions_         = nullptr;
    std::int32_t* host_sampled_           = nullptr;
    std::int32_t* nonfinite_              = nullptr;
    std::uint32_t* moe_error_             = nullptr;
    // The full visits' KV (map_call_pages): the pool and the lanes' block-table rows over kv_backing_, the scratch page,
    // and each lane's reservation and mapped pages. Declared in teardown order: leases before their pool.
    // A lane's pages: block b is mapped for b < mapped, its page pages[b] while the lane owns it (an invalid lease
    // where the row names a cache page: a block the binding mapped from its path, or one it published).
    struct LaneKv {
        DeviceKVPageReservation reservation;
        std::vector<DeviceKVPageLease> pages; // lane_pages() entries
        std::uint32_t mapped = 0;
    };
    DeviceBuffer kv_backing_;
    std::unique_ptr<DeviceKVPagePool> pool_;
    std::unique_ptr<KVExecutionTablePool> tables_;
    std::vector<KVExecutionRowLease> kv_rows_;
    DeviceKVPageLease scratch_;
    std::vector<LaneKv> kv_lanes_;
    // The prefix cache (ProgramOptions::prefix_cache): destroyed before the pool its blocks' leases return to.
    std::unique_ptr<prefix::PrefixCache> prefix_;
    std::uint64_t capture_ids_ = 0;
    bool prefix_failed_        = false; // clear_failed_prefix
    bool prefix_ssd_wait_      = false; // a quote is reading its source from the SSD tier (hybrid_source_pending)
    // --prefix-cache-file (attach_prefix_cache_file): the file, its fingerprint, the Host bytes written at the last
    // save, and the shutdown save's result.
    std::filesystem::path prefix_file_;
    std::string prefix_fingerprint_;
    std::uint64_t prefix_saved_writes_ = 0;
    std::uint64_t prefix_saves_        = 0; // saves of the cache file that succeeded
    // Entries a withdrawal found pinned by another lane: withdrawn at a later release or before a save.
    runtime::prefix_cache::Withdrawal prefix_withdrawal_;
    std::optional<runtime::prefix_cache::PersistResult> prefix_shutdown_save_;
    // bf16p: device [kMaximumConcurrency], each lane's lowest appended position with no packed form, and its last
    // reading.
    static constexpr std::uint32_t kNoPosition = 0xFFFFFFFFU;
    DeviceBuffer unpackable_;
    DeviceBuffer window_codes_, window_side_; // quantized windows: [lanes][planes][kWindowRows][bytes] (StateView)
    DeviceBuffer score_ids_; // ScoreOptions::argmax_only: a head slice's top token ids
    std::array<std::uint32_t, kMaximumConcurrency> unpackable_seen_ = [] {
        std::array<std::uint32_t, kMaximumConcurrency> a{};
        a.fill(kNoPosition);
        return a;
    }();
    std::array<std::uint32_t, kMaxSampleRows> sample_lanes_{}; // the staged sample rows' lanes
    // Constrained rows' masks (mask_bytes, fill_masks): device and pinned [mask_slots][words], and each staged row's
    // dead positions (bit j: its j-th position has no legal token).
    DeviceBuffer masks_;
    std::optional<PinnedHostBuffer> host_masks_;
    std::array<std::uint64_t, kMaxSampleRows> mask_dead_{};
    // The rows stage_sample left for fill_staged_masks.
    std::array<std::size_t, kMaxSampleRows> staged_rows_{};
    std::size_t staged_masks_ = 0;
    // Token constraints (constraint_words): device and pinned operands, the plan whose sets each lane's columns hold,
    // and the staged call's rows: each one's descriptor (-1 unconstrained) and step.
    DeviceBuffer constraint_;
    std::optional<PinnedHostBuffer> constraint_host_;
    std::array<std::uint64_t, kMaximumConcurrency> constraint_serials_{};
    std::uint64_t next_constraint_serial_ = 1;
    bool staged_constraint_               = false;
    std::size_t staged_constraint_rows_   = 0;
    std::array<std::int32_t, kMaximumConcurrency> staged_descriptors_{};
    std::array<std::size_t, kMaximumConcurrency> staged_steps_{};
    std::vector<float> captured_rows_; // TokenConstraint::capture_logits: the staged rows' logits, before the mask
    // A graph key's bit for a capture without the sample (a constrained round samples after its masks).
    static constexpr std::uint64_t kForwardOnlyGraph = 1ULL << 62;
    std::array<TokenId, kMaxSampleRows> sampled_tokens_{};
    std::array<runtime::PendingRowFailure, kMaxSampleRows> failures_{};
    // A DSpark round's rows' tokens, row-major (kDSparkBlock a row): what each row licenses (row_counts_), committed from
    // here.
    std::array<TokenId, kMaxSampleRows> round_tokens_{};
    std::map<std::uint64_t, DecodeGraphExecutable> graphs_;
    static constexpr int kStagingSlots = 4;
    struct Events {
        std::array<cudaEvent_t, kStagingSlots> events{};
        Events() {
            for (auto& e : events) { CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); }
        }
        ~Events() {
            for (auto e : events) { cudaEventDestroy(e); }
        }
        Events(const Events&)            = delete;
        Events& operator=(const Events&) = delete;
        cudaEvent_t operator[](int i) const { return events[static_cast<std::size_t>(i)]; }
    } slot_events_;
    std::array<bool, kStagingSlots> slot_used_{};
    // A DSpark round's timing: before the draft, after it, before the verify call, after it (timing events).
    struct RoundEvents {
        std::array<cudaEvent_t, 4> events{};
        RoundEvents() {
            for (auto& e : events) { CUDA_CHECK(cudaEventCreate(&e)); }
        }
        ~RoundEvents() {
            for (auto e : events) { cudaEventDestroy(e); }
        }
        RoundEvents(const RoundEvents&)            = delete;
        RoundEvents& operator=(const RoundEvents&) = delete;
        cudaEvent_t operator[](int i) const { return events[static_cast<std::size_t>(i)]; }
    } round_events_;
    int slot_ = 0;
    std::optional<std::uint32_t> binding_;
    std::optional<ResumeState> pause_;   // a pause the Engine has not polled yet (start_pause)
    // The CUDA graphs instantiated so far and the device memory they took (instantiate_graph).
    std::atomic<std::uint64_t> graph_bytes_{0};
    std::atomic<std::uint32_t> graph_executables_{0};
    std::vector<std::int32_t> penalty_host_; // a resumed lane's penalty counts on the host (rebuild_counts)
    std::vector<cudaEvent_t> timing_events_; // the prompt calls' timing events not in use (take_timing_pair)
    std::uint64_t reserved_bytes_ = 0;      // set_reserved_bytes
    std::uint64_t reserved_graphs_ = 0;     // graph_bytes_ when reserved_bytes_ was set
    std::optional<std::uint64_t> pending_;
    PendingKind pending_kind_ = PendingKind::Decode;
    std::uint64_t transaction_ = 0;
    std::uint64_t epoch_       = 0;
    std::uint64_t wait_ns_     = 0; // device waits of the unit in progress
};

} // namespace detail

std::optional<ops::inkling2::KvCodec> kv_codec(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16: return ops::inkling2::KvCodec::Bf16;
    case KvCacheStorage::Int8Group64: return ops::inkling2::KvCodec::Int8G64;
    case KvCacheStorage::Fp8E4M3Row256: return ops::inkling2::KvCodec::Fp8Row;
    case KvCacheStorage::Nvfp4Group16: return ops::inkling2::KvCodec::Nvfp4G16;
    case KvCacheStorage::Fp8KeyNvfp4Value: return ops::inkling2::KvCodec::Fp8KeyNvfp4V;
    case KvCacheStorage::Bf16Packed: return ops::inkling2::KvCodec::Bf16Packed;
    case KvCacheStorage::Vq2: return ops::inkling2::KvCodec::Vq2;
    case KvCacheStorage::Q4KeyVq2Value: return ops::inkling2::KvCodec::Q4KeyVq2V;
    }
    return std::nullopt;
}

std::optional<ops::inkling2::KvCodec> window_codec(ops::inkling2::KvCodec pool, std::optional<KvCacheStorage> window) {
    using ops::inkling2::KvCodec;
    if (window) {
        switch (*window) {
        case KvCacheStorage::BFloat16: return KvCodec::Bf16;
        case KvCacheStorage::Bf16Packed: return KvCodec::Bf16Packed;
        case KvCacheStorage::Int8Group64: return KvCodec::Int8G64;
        default: return std::nullopt;
        }
    }
    switch (pool) {
    case KvCodec::Bf16: return KvCodec::Bf16;
    case KvCodec::Bf16Packed: return KvCodec::Bf16Packed;
    default: return KvCodec::Int8G64;
    }
}

KvCacheStorage kv_storage(ops::inkling2::KvCodec codec) {
    switch (codec) {
    case ops::inkling2::KvCodec::Bf16: return KvCacheStorage::BFloat16;
    case ops::inkling2::KvCodec::Int8G64: return KvCacheStorage::Int8Group64;
    case ops::inkling2::KvCodec::Fp8Row: return KvCacheStorage::Fp8E4M3Row256;
    case ops::inkling2::KvCodec::Nvfp4G16: return KvCacheStorage::Nvfp4Group16;
    case ops::inkling2::KvCodec::Fp8KeyNvfp4V: return KvCacheStorage::Fp8KeyNvfp4Value;
    case ops::inkling2::KvCodec::Bf16Packed: return KvCacheStorage::Bf16Packed;
    case ops::inkling2::KvCodec::Vq2: return KvCacheStorage::Vq2;
    case ops::inkling2::KvCodec::Q4KeyVq2V: return KvCacheStorage::Q4KeyVq2Value;
    }
    return KvCacheStorage::BFloat16;
}

using detail::ContractAccess;

ResumeState::ResumeState(std::unique_ptr<detail::ResumeStateImpl> impl, std::uint32_t frontier) noexcept
    : impl_(std::move(impl)), frontier_(frontier) {}
ResumeState::ResumeState(ResumeState&&) noexcept            = default;
ResumeState& ResumeState::operator=(ResumeState&&) noexcept = default;
ResumeState::~ResumeState()                                 = default;

RequestBasePlan::RequestBasePlan(std::unique_ptr<detail::BasePlanImpl> impl) noexcept : impl_(std::move(impl)) {}
RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept            = default;
RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;
RequestBasePlan::~RequestBasePlan()                                     = default;
const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept { return impl_->summary; }

std::uint32_t pages_per_lane(std::uint32_t max_context) noexcept {
    return (max_context + ops::inkling2::kKeyUnit - 1) / ops::inkling2::kKeyUnit + 1;
}

ProgramDevicePlan Program::plan_device(const ProgramOptions& options, const Config& config) {
    if (options.max_context == 0) { throw std::invalid_argument("Inkling 2: max_context must be positive"); }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Inkling 2: max_concurrency must be 1-8");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk > static_cast<std::uint32_t>(kMaxPrefillChunk)) {
        throw std::invalid_argument("Inkling 2: prefill chunks are 1-8191 columns");
    }
    const StateLayout layout   = state_layout(config.text, options.window_codec);
    const std::uint64_t lanes  = options.max_concurrency;
    const detail::KvLayout kv  = detail::kv_layout(options, layout.full_visits);
    ProgramDevicePlan out;
    out.kv_pages   = kv.pool.spec.page_group_count;
    out.kv         = kv.bytes;
    out.page_bytes = kv.page_bytes;
    out.state     = (static_cast<std::uint64_t>(layout.lane_words) * 2 +
                     static_cast<std::uint64_t>(window_code_bytes(layout, options.window_codec) +
                                                window_side_bytes(layout, options.window_codec))) *
                    lanes;
    out.io        = static_cast<std::uint64_t>(config.text.vocab_size) * 4 * lanes * 2 +
             detail::mask_bytes(options, config.text.vocab_size) + detail::constraint_bytes(options);
    const bool dspark = options.dspark && !options.causal_scoring;
    out.workspace = Forward::device_bytes(config.text, static_cast<std::int32_t>(lanes), detail::forward_capacity(options),
                                          dspark, detail::expert_activation(options), options.a8, options.window_codec) +
                    ops::sampling_workspace_capacity_bytes(static_cast<std::int32_t>(config.text.vocab_size), 1,
                                                           static_cast<std::int32_t>(kMaximumConcurrency));
    if (dspark) {
        out.workspace += Drafter::device_bytes(static_cast<std::int32_t>(lanes), options.max_context,
                                               detail::forward_capacity(options), config.text.vocab_size);
    }
    if (options.vision) {
        if (!config.vision || options.vision_max_cells == 0) {
            throw std::invalid_argument("Inkling 2: --vision needs the vision config and a positive image budget");
        }
        out.workspace += VisionTower::device_bytes(*config.vision, options.vision_max_cells, options.vision_host);
    }
    if (options.lossless) {
        out.workspace +=
            LosslessForward::device_bytes(config.text, detail::lossless_columns(options), detail::lossless_keys(options));
        if (options.lora_rank != 0) {
            out.workspace += LosslessForward::lora_workspace_bytes(config.text, static_cast<std::int32_t>(options.lora_rank),
                                                                  detail::lossless_columns(options));
        }
    } else if (options.lora_rank != 0) {
        out.workspace += LoraForward::device_bytes(static_cast<std::int32_t>(options.lora_rank), detail::forward_capacity(options));
    }
    // ScoreOptions::prompt_hidden_states: every layer's output at one column.
    if (options.causal_scoring && !options.lossless) {
        out.workspace += detail::align_256(std::uint64_t{config.text.num_hidden_layers} * config.text.hidden_size * 2);
    }
    out.graph_bound = static_cast<std::uint32_t>(lanes * (lanes + 1) / 2);
    if (options.use_cuda_graph && !options.causal_scoring && !options.lossless) {
        // The decode graphs and their forward-only forms; with DSpark a verify graph per (lane, 2-8 columns, sampled or
        // forward-only) and a draft graph per step count, and with several lanes the rounds of several rows' graphs
        // (rows_graph keeps kRowsGraphs).
        std::uint64_t executables = 2ULL * out.graph_bound;
        if (dspark) {
            executables += lanes * static_cast<std::uint64_t>(ops::inkling2::kDSparkBlock - 1) * 2 +
                           static_cast<std::uint64_t>(ops::inkling2::kMarkovSteps);
            if (lanes >= 2) { executables += static_cast<std::uint64_t>(detail::kRowsGraphs); }
        }
        out.graphs = kGraphDriverBytes + executables * kGraphExecutableBytes;
    }
    return out;
}

Program::Program(const Parameters& parameters, DeviceContext& device, ProgramOptions options)
    : impl_(std::make_unique<detail::ProgramImpl>(parameters, device, std::move(options))) {}
Program::~Program() noexcept = default;

const ProgramDevicePlan& Program::device_plan() const noexcept { return impl_->device_plan(); }

RequestBasePlan Program::plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(std::move(prompt), options);
}

std::vector<SourceCandidate> Program::hybrid_sources(const RequestBasePlan& base, const ResumeState* resume) {
    return impl_->sources(base, resume);
}

BindingReservation Program::start_binding(const RequestBasePlan& base, runtime::LaneId lane, const SourceCandidate& source,
                                          ResumeState* resume, ExecutionUnitKind, std::uint32_t) {
    return impl_->start_binding(base, lane, source, resume);
}

bool Program::hybrid_reclaim(runtime::ContextResourceUsage shortage) { return impl_->reclaim(shortage); }
void Program::hybrid_hold_queue(std::span<const RequestBasePlan* const> queue, std::span<const ResumeState* const> resumes) {
    impl_->hold_queue(queue, resumes);
}
std::uint32_t Program::pause_replay_tokens(SequenceHandle sequence) const noexcept {
    return impl_->pause_replay_tokens(sequence);
}
bool Program::start_pause(SequenceHandle sequence, runtime::ExecutionTiming*) { return impl_->start_pause(sequence); }
ReplayProgress Program::advance_replay(SequenceHandle sequence, runtime::ExecutionTiming*) {
    return impl_->advance_replay(sequence);
}

HybridPrefixCacheStats Program::hybrid_stats() const noexcept { return impl_->prefix_stats(); }
bool Program::hybrid_poll_storage() { return impl_->prefix_poll_storage(); }
void Program::maintain() {
    (void)impl_->prefix_poll_storage();
    impl_->maintain_ram(true);
}
void Program::maintain_busy() { impl_->maintain_ram(false); }
void Program::set_maintenance_waker(std::function<void()> waker) { impl_->set_maintenance_waker(std::move(waker)); }
void Program::start_ram_resize(std::uint64_t reserve_bytes, bool trim_host_tier) {
    impl_->start_ram_resize(reserve_bytes, trim_host_tier);
}
bool Program::hybrid_source_pending() const noexcept { return impl_->prefix_ssd_wait(); }
bool Program::hybrid_storage_due() const noexcept { return impl_->prefix_storage_due(); }
std::optional<std::chrono::steady_clock::time_point> Program::hybrid_storage_retry_at() const noexcept {
    return impl_->prefix_storage_retry_at();
}
std::uint64_t Program::hybrid_cache_epoch() const noexcept { return impl_->prefix_epoch(); }

ContextProgress Program::poll_context(runtime::CancellationFlagView) { return impl_->poll_context(); }
bool Program::has_context_transaction() const noexcept { return impl_->has_context_transaction(); }
bool Program::context_blocks(SequenceHandle sequence) const noexcept { return impl_->context_blocks(sequence); }

runtime::ResourceReservation Program::reserve_units(std::span<const ExecutionUnit> units) {
    return impl_->reserve_units(units);
}

PrefillProgress Program::advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming*, runtime::TokenMaskProvider* masks) {
    return impl_->advance_prefill(sequence, masks);
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming*, runtime::TokenMaskProvider* masks) {
    return impl_->decode(sequences, budgets, masks);
}

runtime::ExecutionTiming Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
                                                       std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                                                       std::span<const std::optional<std::uint32_t>>,
                                                       runtime::ExecutionTiming*) {
    return impl_->append_forced(sequences, row_major_tokens, row_stride);
}

CommitResult Program::commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation, runtime::ExecutionTiming*) {
    return impl_->commit(std::move(pending), decisions);
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept { return impl_->discard(std::move(pending)); }
FinishResult Program::finish(SequenceHandle sequence) noexcept { return impl_->finish(sequence); }
AbortResult Program::abort(SequenceHandle sequence) noexcept { return impl_->abort(sequence); }
void Program::fail_all_cleanup() noexcept { impl_->release_all(); }
bool Program::recover_after_failure() noexcept { return impl_->recover(); }
void Program::shutdown_cleanup() noexcept {
    impl_->release_all();
    impl_->save_prefix_cache_for_shutdown();
}

runtime::prefix_cache::PersistResult Program::attach_prefix_cache_file(const std::filesystem::path& path,
                                                                       std::string fingerprint,
                                                                       const StartupObserver& observer) {
    return impl_->attach_prefix_cache_file(path, std::move(fingerprint), observer);
}
std::optional<runtime::prefix_cache::PersistResult> Program::save_prefix_cache_now(const CancellationView& abandoned) {
    return impl_->save_prefix_cache_now(abandoned);
}
std::optional<runtime::prefix_cache::PersistResult> Program::prefix_shutdown_save() const {
    return impl_->prefix_shutdown_save();
}

ScoreResult Program::causal_score(PreparedPrompt&& prompt, std::uint32_t first_target, const ScoreOptions& options) {
    return impl_->causal_score(std::move(prompt), first_target, options);
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->physical_usage(); }
MemoryUsage Program::memory_usage() const noexcept { return impl_->memory_usage(); }
void Program::set_reserved_bytes(std::uint64_t bytes) noexcept { impl_->set_reserved_bytes(bytes); }
MemorySummary Program::memory_summary() const noexcept { return impl_->memory_summary(); }

} // namespace infernix::models::inkling2
