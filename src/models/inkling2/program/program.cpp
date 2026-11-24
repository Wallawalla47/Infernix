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
#include "runtime/prefix_cache/prefix_holds.h"
#include "runtime/prefix_cache/tap_planner.h"
#include "runtime/speculative/draft_length.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
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

// The full visits' page pool (its scratch page included) and the lanes' block-table rows in one backing allocation.
struct KvLayout {
    DeviceKVPagePoolLayout pool;
    KVExecutionTableLayout tables;
    std::uint64_t bytes      = 0;
    std::uint64_t page_bytes = 0; // one page of every plane
};
KvLayout kv_layout(const ProgramOptions& o, std::int32_t full_visits) {
    const std::uint32_t row   = pages_per_lane(o.max_context);
    const std::uint64_t rows  = static_cast<std::uint64_t>(row) * o.max_concurrency;
    const std::uint64_t pages = std::max<std::uint64_t>(rows, o.kv_pages) + 1;
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

// Constrained rows' token masks (constrained-decoding.md §7.1): one packed vocabulary mask a position. Slot r is a plain
// round's row r, slot i a DSpark verify call's column i (one row drafts at a time), so the slots cover both.
constexpr std::int32_t kMaskSlots = std::max<std::int32_t>(kMaximumConcurrency, ops::inkling2::kDSparkBlock);
std::uint64_t mask_words(std::uint64_t vocab) { return (vocab + 31) / 32; }
// Device and pinned bytes of the masks (generation only: scoring samples nothing).
std::uint64_t mask_bytes(const ProgramOptions& options, std::uint64_t vocab) {
    return options.causal_scoring ? 0 : align_256(4 * mask_words(vocab) * kMaskSlots);
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

class ProgramImpl {
public:
    ProgramImpl(const Parameters& parameters, DeviceContext& device, ProgramOptions options)
        : p_(parameters), c_(parameters.model.config().text), device_(device), o_(std::move(options)),
          plan_(Program::plan_device(o_, parameters.model.config())),
          forward_(parameters, static_cast<std::int32_t>(o_.max_concurrency), forward_capacity(o_), expert_activation(o_),
                   o_.a8),
          vocab_(static_cast<std::int32_t>(c_.vocab_size)), lanes_(o_.max_concurrency),
          sampling_(ops::sampling_workspace_capacity_bytes(vocab_, 1, static_cast<std::int32_t>(kMaximumConcurrency)) +
                    4096),
          host_(align_256(forward_.upload_words() * 4 * kStagingSlots) +
                align_256(sizeof(ops::SamplingConfig) * kMaximumConcurrency) + 3 * align_256(4 * kMaximumConcurrency) +
                align_256(4)) {
        const StateLayout& layout = forward_.layout();
        const std::uint32_t pages = pages_per_lane(o_.max_context);
        if (o_.lossless) {
            lossless_ = std::make_unique<LosslessForward>(p_, lossless_columns(o_), o_.reference, lossless_keys(o_));
        } else if (o_.causal_scoring) { // ScoreOptions::prompt_hidden_states: BF16 [layers][H]
            score_hidden_ = DeviceBuffer(static_cast<std::size_t>(c_.num_hidden_layers) * c_.hidden_size * 2);
        }
        images_ = DeviceBuffer(static_cast<std::size_t>(layout.lane_words) * 2 * lanes_.size());
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
        configs_           = DeviceBuffer(sizeof(ops::SamplingConfig) * kMaximumConcurrency);
        positions_         = DeviceBuffer(4 * kMaximumConcurrency);
        sampled_           = DeviceBuffer(4 * kMaximumConcurrency);
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
        base += align_256(sizeof(ops::SamplingConfig) * kMaximumConcurrency);
        host_positions_ = reinterpret_cast<std::int32_t*>(base);
        base += align_256(4 * kMaximumConcurrency);
        host_sampled_ = reinterpret_cast<std::int32_t*>(base);
        base += align_256(4 * kMaximumConcurrency);
        nonfinite_ = reinterpret_cast<std::int32_t*>(base);
        base += align_256(4 * kMaximumConcurrency);
        // The MoE's error word, mapped: a down CTA that waited past its bound sets it, and finish_sample reads it after
        // the round's own synchronization (no copy).
        moe_error_  = reinterpret_cast<std::uint32_t*>(base);
        *moe_error_ = 0;
        forward_.set_moe_error_word(moe_error_);
        if (o_.kv_codec == ops::inkling2::KvCodec::Bf16Packed) {
            // bf16p: each lane's lowest appended position with no packed form (atomicMin; 0xFFFFFFFF: none).
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
            prefix::LaneImages images;
            const std::size_t lane_bytes = static_cast<std::size_t>(forward_.layout().lane_words) * 2;
            images.parts.push_back({static_cast<std::byte*>(images_.p), lane_bytes, lane_bytes});
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
        prefix_.reset();
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
    // a root start.
    std::vector<SourceCandidate> sources(const RequestBasePlan& base, const ResumeState* resume) {
        if (resume != nullptr) { throw std::logic_error("Inkling 2: no request is paused"); }
        SourceCandidate root;
        const BasePlanImpl& b = *base.impl_;
        prefix_ssd_wait_ = false;
        if (!prefix_ || !b.reuse || prefix_failed_) { return {root}; }
        namespace pc = runtime::prefix_cache;
        const auto n = static_cast<std::uint32_t>(b.tokens.size());
        prefix_->poll();
        pc::MatchResult match = prefix_->index().match(b.tokens, b.block_hashes, b.block_extras, n);
        // A tap another lane is still copying out that this prompt extends: it publishes first.
        if (prefix_->await_extended_capture(match.path, b.tokens)) {
            match = prefix_->index().match(b.tokens, b.block_hashes, b.block_extras, n);
        }
        root.cached_block_tokens = static_cast<std::uint32_t>(match.path.size()) * pc::kBlockTokens;
        const auto spans = exclusions(b);
        std::erase_if(match.candidates, [&](const pc::MatchCandidate& c) {
            return excluded(spans, c.frontier) || c.filling_blocks != 0;
        });
        if (match.candidates.empty()) { return {root}; }
        const pc::AdmissionChoice choice = prefix_->index().choose(match, n);
        if (!choice.candidate) { return {root}; }
        const pc::MatchCandidate& c = match.candidates[*choice.candidate];
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
    // transfers hold the rest. Every lane reserves at most its whole context and the pool holds every lane's, so the
    // pages exist once the cache gives them up.
    void make_room(std::uint32_t pages) {
        while (pool_->available_pages() < pages) {
            if (prefix_ && prefix_->index().evict_device_blocks(pages - pool_->available_pages()) != 0) { continue; }
            if (!prefix_ || !prefix_->settle_oldest_transfer()) {
                throw std::logic_error("Inkling 2: the KV pool cannot give a lane its context");
            }
        }
    }

    BindingReservation start_binding(const RequestBasePlan& base, runtime::LaneId lane_id, const SourceCandidate& source,
                                     const ResumeState* resume) {
        if (resume != nullptr) { throw std::logic_error("Inkling 2: no request is paused"); }
        if (binding_) { throw std::logic_error("Inkling 2: a binding is already open"); }
        if (lane_id.value >= lanes_.size() || lanes_[lane_id.value].bound) {
            throw std::logic_error("Inkling 2: the binding's lane is not free");
        }
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
        // The lane reserves its context less the path's blocks, which it maps from the cache; the Host-only ones take
        // pages of their own, which the cache owns once restored.
        make_room(lane_pages() - k + host_only);
        reserve_pages(idx, lane_pages() - k);
        Lane& lane         = lanes_[lane_id.value];
        lane               = Lane{};
        lane.bound         = true;
        lane.epoch         = ++epoch_;
        lane.tokens        = base.impl_->tokens;
        lane.images        = base.impl_->images;
        lane.prompt_tokens = base.impl_->summary.prompt_tokens;
        lane.options       = base.impl_->options;
        lane.constraint    = base.impl_->constraint;
        lane.penalties     = lane.options.sampling.presence_penalty != 0.0F || lane.options.sampling.frequency_penalty != 0.0F;
        lane.calibration.reset();
        clear_unpackable(lane_id.value);
        if (lane.penalties) {
            CUDA_CHECK(cudaMemsetAsync(lane_counts(lane_id.value), 0, static_cast<std::size_t>(vocab_) * 4,
                                       device_.stream));
        }
        lane.prefix.reuse  = b.reuse;
        lane.prefix.hashes = b.block_hashes;
        lane.prefix.extras = b.block_extras;
        if (quote == nullptr) {
            // A fresh state: the conv rings' history before position 0 is zero, and the window's rows are rewritten.
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::uint16_t*>(images_.p) +
                                           static_cast<std::size_t>(lane_id.value) * forward_.layout().lane_words,
                                       0, static_cast<std::size_t>(forward_.layout().lane_words) * 2, device_.stream));
        } else {
            bind_snapshot(idx, *quote, k);
            (void)pins.release(); // the lane's now (released with it)
        }
        if (lane.prefix.reuse) { plan_lane_taps(idx, b); }
        binding_ = lane_id.value;
        BindingReservation out;
        out.reserved = true;
        return out;
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
        if (!binding_) { throw std::logic_error("Inkling 2: no binding is open"); }
        const std::uint32_t index = *binding_;
        binding_.reset();
        ContextProgress out;
        out.kind      = ContextOperationKind::Bind;
        out.advanced  = true;
        out.complete  = true;
        out.published = true; // the sequence is live (the Engine cancels a binding that publishes none)
        out.sequence  = handle(index);
        return out;
    }

    [[nodiscard]] bool has_context_transaction() const noexcept { return binding_.has_value(); }

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
        if (lossless_) {
            run_lossless(index, call, last);
        } else {
            upload(call);
            run_forward(static_cast<std::int32_t>(chunk) - 1, last ? 1 : 0,
                        [&] { overlay_images(lane.images, lane.processed, chunk); });
            if (drafter_) { drafter_->append(0, static_cast<std::int32_t>(chunk), device_.stream); } // its context rows
        }
        lane.processed += chunk;
        const std::size_t captured = lane.prefix.captures.size();
        publish_prefill(index);
        PrefillProgress out;
        out.snapshot_published = lane.prefix.captures.size() != captured;
        // The admission's terms: the tokens the binding restored from a snapshot (none at a root start).
        out.summary = {.prompt_tokens        = lane.prompt_tokens,
                       .reused_prompt_tokens = lane.prefix.base,
                       .prefix_reuse_path    = lane.prefix.base != 0 ? PrefixReusePath::HybridSnapshot : PrefixReusePath::Root};
        out.processed_prompt_tokens = chunk; // this unit's tokens (the Engine sums them)
        out.complete                = last;
        if (last) {
            const std::uint64_t transaction = ++transaction_;
            const SequenceHandle row        = handle(index);
            // The head ran over the chunk's last column only: its logits are row 0. A readout reads them before a
            // token constraint masks them in place for the sampler.
            if (!lane.options.readout_tokens.empty()) { out.readout = readout(lane, 0); }
            sample({&row, 1}, {&index, 1}, 0, ops::kSamplePurposePrefill, masks);
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
        // A row's round licenses at most its remaining generated tokens: a DSpark round needs two of them.
        const std::uint32_t remaining = budgets.empty() ? 0U : budgets[0].generated_tokens_remaining;
        if (remaining >= 2 && dspark_round(sequences)) { return decode_dspark(sequences[0], remaining, masks, start); }
        if (lossless_) { return decode_lossless(sequences, masks, start); }
        if (drafter_ && sequences.size() == 1) { ++lanes_[lane_of(sequences[0])].fallback; } // a plain round of a drafting row
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
        if (uniform && o_.use_cuda_graph) {
            decode_graph(static_cast<std::int32_t>(order.back()) + 1, columns, rows, logit_rows, masks);
        } else {
            run_decode_call(columns);
            sample(sequences, rows, logit_rows, ops::kSamplePurposeDecode, masks);
        }
        for (const std::uint32_t index : order) {
            Lane& lane     = lanes_[index];
            lane.processed = static_cast<std::uint32_t>(lane.tokens.size());
        }
        const std::uint64_t ns = elapsed_ns(start);
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
                if (d.accepted_tokens == 0 || d.accepted_tokens > static_cast<std::uint32_t>(row_counts_[0])) {
                    throw std::logic_error("Inkling 2: a DSpark row commits 1 .. its licensed tokens");
                }
                for (std::uint32_t i = 0; i < d.accepted_tokens; ++i) { lane.tokens.push_back(sampled_tokens_[i]); }
                lane.processed = pending_draft_.first + d.accepted_tokens;
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
        clear_unpackable(0);
        release_pages(0); // a failed window's pages
        reserve_pages(0, lane_pages()); // lane 0 maps its pages as the window's calls reach them
        // ScoreOptions::prompt_hidden_states: the call holding the last prompt position captures its column.
        if (options.prompt_hidden_states && score_hidden_.p == nullptr) {
            throw std::invalid_argument("Inkling 2: hidden states need a scoring Program off the lossless route");
        }
        bool hidden_captured = false;
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
        out.occupied.main_kv_pages = static_cast<std::uint32_t>(mapped_pages());
        out.capacity.state_slots   = static_cast<std::uint32_t>(lanes_.size());
        out.capacity.main_kv_pages = plan_.kv_pages - 1; // the scratch page holds nothing
        return out;
    }

    [[nodiscard]] MemoryUsage memory_usage() const noexcept {
        MemoryUsage out;
        out.device.weights   = p_.model.device_weight_bytes();
        out.device.kv        = plan_.kv;
        out.device.state     = plan_.state + plan_.io;
        out.device.workspace = plan_.workspace;
        out.device.adapter   = p_.adapter_bytes(); // --lora's factors and --ablate's direction (0 without them)
        out.device.kv_used   = mapped_pages() * plan_.page_bytes;
        out.host.staging   = host_.size();
        return out;
    }

    [[nodiscard]] MemorySummary memory_summary() const noexcept {
        MemorySummary out;
        out.max_context      = o_.max_context;
        out.kv_capacity      = o_.max_context * o_.max_concurrency;
        out.kv_cache         = kv_storage(o_.kv_codec);
        out.kv_payload_bytes = plan_.kv;
        out.adapter_bytes    = p_.adapter_bytes();
        return out;
    }

    const ProgramDevicePlan& device_plan() const noexcept { return plan_; }

private:
    struct Lane {
        bool bound          = false;
        std::uint64_t epoch = 0;
        std::vector<TokenId> tokens; // the prompt, then the committed output and forced tokens
        std::shared_ptr<const std::vector<PromptImage>> images; // the prompt's images (null: none)
        std::uint32_t prompt_tokens = 0;
        std::uint32_t processed     = 0; // positions whose state the lane holds
        runtime::ResolvedExecutionOptions options;
        bool penalties = false;
        // A token constraint: its plan, the round's draw (committed with its token) and the request's draws.
        std::shared_ptr<const ConstraintPlan> constraint;
        std::vector<ConstrainedDraw> constraint_round, constraint_trace;
        std::uint64_t prefill_ns = 0, decode_ns = 0, decode_share_ns = 0;
        // DSpark: the drafts' acceptance by probability decile (draft_length.h) and the request's counters.
        runtime::speculative::DraftCalibration calibration;
        std::uint64_t policy_rounds = 0;
        std::int32_t step_cap = ops::inkling2::kMarkovSteps; // the calibrated length's Markov steps a round drafts
        std::uint64_t rounds = 0, drafted = 0, accepted = 0, fallback = 0;
        double draft_ms = 0, gap_ms = 0, verify_ms = 0; // device-side sums over the rounds (CUDA events)
        std::array<std::uint64_t, ops::inkling2::kMarkovSteps> accepted_at{}; // rounds accepting draft i
        // The prefix cache: whether the request reuses and publishes, its keys, the blocks its row maps from the
        // cache (path, root first: pinned for the lane), the taps still to take, the snapshot frontiers on its path,
        // what it published (withdrawn when its logits go non-finite), the restore its first call waits for and the
        // capture its next call's image writes wait for.
        struct Prefix {
            bool reuse = false;
            std::vector<std::uint64_t> hashes, extras;
            std::vector<runtime::prefix_cache::NodeRef> path;
            std::vector<runtime::prefix_cache::PlannedTap> taps;
            std::vector<std::uint32_t> frontiers;
            std::uint32_t base = 0; // the frontier the lane resumed at (0: root)
            std::vector<runtime::prefix_cache::NodeRef> published;
            std::vector<std::uint64_t> captures;
            std::optional<prefix::RestoreTicket> restore;
            std::uint64_t capture = 0;
            bool withdraw = false;
        } prefix;
    };
    // A DSpark round's row awaiting its commit: the lane, the anchor's position and the tokens it licenses.
    struct PendingDraft {
        bool active        = false;
        std::uint32_t lane = 0, first = 0;
    };
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
    void release(std::uint32_t index) noexcept {
        if (index < lanes_.size()) {
            release_prefix(index);
            release_pages(index);
            lanes_[index] = Lane{};
            clear_failed_prefix();
        }
    }
    // bf16p: forgets a lane's rows with no packed form (a new request, or rejected draft rows the next calls overwrite).
    void clear_unpackable(std::uint32_t index) {
        if (unpackable_.p != nullptr) {
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::uint32_t*>(unpackable_.p) + index, 0xFF, 4, device_.stream));
            unpackable_seen_[index] = kNoPosition;
        }
    }
    static GenerationTimings timings(const Lane& lane) {
        GenerationTimings out;
        out.prefill_seconds      = static_cast<double>(lane.prefill_ns) * 1e-9;
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
    // The full visits' pages (core/paged_kv_cache.h): a lane reserves its whole context when it binds (so no request ever
    // waits for pages) and maps its leading pages from that reservation as its calls first write a position in them; its
    // block-table row names the scratch page past them (a row past a lane's frontier is never read). The row's
    // publications are ordered on the compute stream ahead of the call that reads them.
    [[nodiscard]] std::uint32_t lane_pages() const noexcept { return tables_->logical_page_capacity(); }

    // Reserves `pages` of a lane's context (all of it at a root start, less its path's blocks at a restore): at a
    // binding, and lane 0's for a scoring window.
    void reserve_pages(std::uint32_t lane, std::uint32_t pages) {
        LaneKv& kv = kv_lanes_[lane];
        if (kv.reservation.valid() || kv.mapped != 0) { throw std::logic_error("Inkling 2: the lane holds KV pages"); }
        make_room(pages);
        std::optional<DeviceKVPageReservation> reserved = pool_->reserve(pages);
        if (!reserved) { throw std::logic_error("Inkling 2: the KV pool cannot hold the lane's context"); }
        kv.reservation = std::move(*reserved);
    }

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
        if (need > lane_pages() || !kv.reservation.valid()) {
            throw std::logic_error("Inkling 2: a call writes past its lane's reserved context");
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

    [[nodiscard]] std::uint64_t mapped_pages() const noexcept {
        std::uint64_t out = 0;
        for (const LaneKv& kv : kv_lanes_) {
            for (std::uint32_t p = 0; p < kv.mapped; ++p) { out += kv.pages[p].valid() ? 1U : 0U; }
        }
        return out;
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
        const std::uint64_t written                 = prefix_->counters().host_write_bytes;
        runtime::prefix_cache::PersistResult saved = prefix_->save(prefix_file_, prefix_fingerprint_, abandoned);
        if (saved.ok) { prefix_saved_writes_ = written; }
        return saved;
    }

    // After every lane is released (shutdown_cleanup): copies land, then the save reads the slabs.
    void save_prefix_cache_for_shutdown() noexcept {
        if (!prefix_ || prefix_file_.empty()) { return; }
        try {
            CUDA_CHECK(cudaStreamSynchronize(device_.stream));
            prefix_->drain();
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
    bool prefix_poll_storage() { return prefix_ && prefix_->poll_ssd(); }
    [[nodiscard]] bool prefix_ssd_wait() const noexcept { return prefix_ssd_wait_; }
    [[nodiscard]] bool prefix_storage_due() const noexcept { return prefix_ && prefix_->ssd_due(); }
    [[nodiscard]] std::optional<std::chrono::steady_clock::time_point> prefix_storage_retry_at() const noexcept {
        return prefix_ ? prefix_->ssd_retry_at() : std::nullopt;
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
        out.host_slabs              = prefix_->host_layout().slabs;
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

    // After a prompt call: the lane's full blocks the call completed are published (each keyed by the prompt's block
    // hash; the cache takes the lane's page unless an identical block is mapped elsewhere), then each tap the call
    // reached is captured. Only prompt-computed state is published: a resumed request's output equals a cold one's.
    void publish_prefill(std::uint32_t idx) {
        Lane& lane = lanes_[idx];
        namespace pc     = runtime::prefix_cache;
        Lane::Prefix& p  = lane.prefix;
        if (!prefix_ || !p.reuse || prefix_failed_) { return; }
        if (unpackable_.p != nullptr) {
            // bf16p: a row with no packed form fails the request; nothing it reached is published.
            CUDA_CHECK(cudaMemcpyAsync(&unpackable_seen_[idx], static_cast<std::uint32_t*>(unpackable_.p) + idx, 4,
                                       cudaMemcpyDeviceToHost, device_.stream));
            CUDA_CHECK(cudaStreamSynchronize(device_.stream));
            if (unpackable_seen_[idx] != kNoPosition) {
                p.reuse = false;
                return;
            }
        }
        const std::uint32_t end = lane.processed;
        LaneKv& kv              = kv_lanes_[idx];
        const auto blocks       = std::min<std::size_t>(end / pc::kBlockTokens, p.hashes.size());
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
                // An identical block another lane maps (the same bits: only prompt-computed blocks are published): the
                // row names its page and the lane's own goes back to the pool, so one block never holds two pages.
                // The calls before read the lane's page; the row's publication runs after them on the compute stream.
                const DeviceKVPageHandle shared = prefix_->page(view.device_id);
                tables_->publish(kv_rows_[idx].handle(), static_cast<std::uint32_t>(bi),
                                 std::span<const DeviceKVPageHandle>(&shared, 1), device_.stream);
                prefix_->order_page_reuse(std::span<const DeviceKVPageLease>(&kv.pages[bi], 1));
                (void)kv.pages[bi].release();
            }
        }
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

    void capture_tap(std::uint32_t idx, std::uint32_t F, runtime::prefix_cache::SnapshotKind kind) {
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
        c.claim                = prefix_->index().estimate_priority(below, F, !c.tail.empty());
        const std::uint64_t id = c.id;
        // The path's Device-only blocks go to the Host tier behind the image (the lane's next call waits only for the
        // image).
        if (prefix_->capture(idx, std::move(c), std::span<const pc::NodeRef>(p.path).first(k), device_.stream)) {
            p.capture = id;
            p.captures.push_back(id);
            p.frontiers.push_back(F);
        }
    }

    // The lane's cache references at its release: its path's pins, and (a request whose logits went non-finite) what it
    // published withdrawn (hybrid-prefix-cache-spec §7.8). A failure empties the cache.
    void release_prefix(std::uint32_t idx) noexcept {
        Lane& lane = lanes_[idx];
        if (!prefix_) { return; }
        namespace pc    = runtime::prefix_cache;
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
            if (p.withdraw && (!p.published.empty() || !p.captures.empty())) {
                prefix_->drain();
                pc::Withdrawal w;
                w.nodes = p.published;
                for (const std::uint64_t id : p.captures) {
                    if (const auto r = prefix_->capture_result(id); r && r->created) { w.snapshots.push_back(r->snapshot); }
                }
                (void)prefix_->index().withdraw(w);
            }
            for (const std::uint64_t id : p.captures) { prefix_->forget_capture(id); }
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
        prefix_failed_ = false;
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
        row_counts_[0]   = a + 1;
        pending_draft_   = {true, index, static_cast<std::uint32_t>(F)};
        const std::uint64_t transaction = ++transaction_;
        const std::uint64_t ns          = elapsed_ns(start);
        lane.decode_ns += ns;
        lane.decode_share_ns += ns;
        pending_      = transaction;
        pending_kind_ = PendingKind::Decode;
        return ContractAccess::batch(this, transaction, {&sequence, 1}, {sampled_tokens_.data(), kMaximumConcurrency},
                                     timing_since(start), {failures_.data(), 1}, {row_counts_.data(), 1},
                                     static_cast<std::uint32_t>(kMaximumConcurrency));
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
            exec.instantiate(definition);
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
            exec.instantiate(definition);
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
        if (slot + drafts.size() + 1 > static_cast<std::size_t>(kMaskSlots)) {
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
    void stage_sample(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> logit_rows, std::int32_t purpose,
                      runtime::TokenMaskProvider* masks, std::span<const std::int32_t> positions = {},
                      bool mask_row_zero = false) {
        staged_masks_ = 0;
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (!mask_row_zero) { mask_dead_[row] = 0; }
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
            config.token_counts      = lane.penalties ? lane_counts(lanes[row]) : nullptr;
            nonfinite_[row]          = 0;
            config.nonfinite         = &nonfinite_[row];
            sample_lanes_[row]       = lanes[row];
            if (masks != nullptr) {
                const std::size_t provider_row  = mask_row_zero ? 0 : row;
                const runtime::ReasoningBan ban = masks->reasoning_ban(provider_row);
                config.reasoning_close          = ban.close;
                config.reasoning_ends[0]        = ban.ends[0];
                config.reasoning_ends[1]        = ban.ends[1];
                config.reasoning_withheld       = ban.withheld;
                // A constrained row samples among its grammar's legal tokens (constrained-decoding.md §7.3).
                if (masks->constrained(provider_row)) {
                    if (!mask_row_zero) { staged_rows_[staged_masks_++] = row; }
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
        // A verify call's columns are one drafting row's, and a token-constrained row never drafts (dspark_round).
        if (mask_row_zero) {
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
        const Tensor in(const_cast<float*>(logits), DType::FP32, {vocab_, static_cast<std::int32_t>(rows)});
        Tensor out(sampled_.p, DType::I32, {static_cast<std::int32_t>(rows)});
        const Tensor positions(positions_.p, DType::I32, {static_cast<std::int32_t>(rows)});
        ops::sample(in, out, vocab_, static_cast<const ops::SamplingConfig*>(configs_.p), positions, purpose, sampling_,
                    device_.stream);
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
    // serving: recover() releases them, queued requests wait) and what every bound lane published is withdrawn.
    void check_moe_error() {
        if (inkling2::testing::take_forced_stall()) {
            *forward_.buffers().moe.error |= ops::w4a4_moe::kErrorDownStalled; // through the kernels' own pointer
        }
        const std::uint32_t word = *static_cast<volatile std::uint32_t*>(moe_error_);
        if (word == 0) { return; }
        *moe_error_ = 0;
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
            failures_[row]        = nonfinite_[row] != 0 ? runtime::PendingRowFailure::NonFiniteLogits
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
    void decode_graph(std::int32_t lanes, std::int32_t columns, std::span<const std::uint32_t> rows,
                      std::span<const std::int32_t> logit_rows, runtime::TokenMaskProvider* masks) {
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
        found->second.launch(device_.stream);
        if (!sample) {
            fill_staged_masks(masks);
            launch_sample(rows.size(), logit_rows, ops::kSamplePurposeDecode);
        }
        finish_sample(rows.size());
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
        exec.instantiate(definition);
        const std::uint64_t key = (sample ? 0ULL : kForwardOnlyGraph) | static_cast<std::uint64_t>(lanes) << 32 |
                                  static_cast<std::uint32_t>(columns);
        return graphs_.emplace(key, std::move(exec)).first;
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
    std::optional<runtime::prefix_cache::PersistResult> prefix_shutdown_save_;
    // bf16p: device [kMaximumConcurrency], each lane's lowest appended position with no packed form, and its last
    // reading.
    static constexpr std::uint32_t kNoPosition = 0xFFFFFFFFU;
    DeviceBuffer unpackable_;
    DeviceBuffer score_ids_; // ScoreOptions::argmax_only: a head slice's top token ids
    std::array<std::uint32_t, kMaximumConcurrency> unpackable_seen_ = [] {
        std::array<std::uint32_t, kMaximumConcurrency> a{};
        a.fill(kNoPosition);
        return a;
    }();
    std::array<std::uint32_t, kMaximumConcurrency> sample_lanes_{}; // the staged sample rows' lanes
    // Constrained rows' masks (mask_bytes, fill_masks): device and pinned [kMaskSlots][words], and each staged row's
    // dead positions (bit j: its j-th position has no legal token).
    DeviceBuffer masks_;
    std::optional<PinnedHostBuffer> host_masks_;
    std::array<std::uint64_t, kMaximumConcurrency> mask_dead_{};
    // The rows stage_sample left for fill_staged_masks.
    std::array<std::size_t, kMaximumConcurrency> staged_rows_{};
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
    std::array<TokenId, kMaximumConcurrency> sampled_tokens_{};
    std::array<runtime::PendingRowFailure, kMaximumConcurrency> failures_{};
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
    const StateLayout layout   = state_layout(config.text);
    const std::uint64_t lanes  = options.max_concurrency;
    const detail::KvLayout kv  = detail::kv_layout(options, layout.full_visits);
    ProgramDevicePlan out;
    out.kv_pages   = kv.pool.spec.page_group_count;
    out.kv         = kv.bytes;
    out.page_bytes = kv.page_bytes;
    out.state     = static_cast<std::uint64_t>(layout.lane_words) * 2 * lanes;
    out.io        = static_cast<std::uint64_t>(config.text.vocab_size) * 4 * lanes * 2 +
             detail::mask_bytes(options, config.text.vocab_size) + detail::constraint_bytes(options);
    const bool dspark = options.dspark && !options.causal_scoring;
    out.workspace = Forward::device_bytes(config.text, static_cast<std::int32_t>(lanes), detail::forward_capacity(options),
                                          dspark, detail::expert_activation(options), options.a8) +
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

HybridPrefixCacheStats Program::hybrid_stats() const noexcept { return impl_->prefix_stats(); }
bool Program::hybrid_poll_storage() { return impl_->prefix_poll_storage(); }
void Program::maintain() { (void)impl_->prefix_poll_storage(); }
bool Program::hybrid_source_pending() const noexcept { return impl_->prefix_ssd_wait(); }
bool Program::hybrid_storage_due() const noexcept { return impl_->prefix_storage_due(); }
std::optional<std::chrono::steady_clock::time_point> Program::hybrid_storage_retry_at() const noexcept {
    return impl_->prefix_storage_retry_at();
}
std::uint64_t Program::hybrid_cache_epoch() const noexcept { return impl_->prefix_epoch(); }

ContextProgress Program::poll_context(runtime::CancellationFlagView) { return impl_->poll_context(); }
bool Program::has_context_transaction() const noexcept { return impl_->has_context_transaction(); }
bool Program::context_blocks(SequenceHandle sequence) const noexcept { return impl_->context_blocks(sequence); }

runtime::ResourceReservation Program::reserve_units(std::span<const ExecutionUnit>) {
    return {.reserved = true}; // every lane's KV is reserved at startup
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
MemorySummary Program::memory_summary() const noexcept { return impl_->memory_summary(); }

} // namespace infernix::models::inkling2
