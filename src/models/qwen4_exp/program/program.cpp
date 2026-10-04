#include "models/qwen4_exp/program/program.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "core/tensor.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/ngram_proposer.h"
#include "models/qwen4_exp/execution/forward.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/frontend/ngram_hash.h"
#include "models/qwen4_exp/program/expert_residency.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen4_exp {
namespace detail {

struct BasePlanImpl {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::uint32_t pages = 0; // KV page groups the request needs
};

struct QuoteImpl {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::uint32_t pages = 0;
    std::uint32_t lane  = 0;
};

struct ContractAccess {
    static SequenceHandle make_sequence(const void* owner, std::uint32_t lane, std::uint64_t epoch) noexcept {
        SequenceHandle out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        return out;
    }
    static const void* owner(const SequenceHandle& h) noexcept { return h.owner_; }
    static std::uint32_t lane(const SequenceHandle& h) noexcept { return h.lane_; }
    static std::uint64_t epoch(const SequenceHandle& h) noexcept { return h.epoch_; }

    static PendingBatch make_pending(const void* owner, std::uint64_t transaction,
                                     std::span<const SequenceHandle> rows, std::span<const TokenId> tokens,
                                     runtime::ExecutionTiming timing, std::span<const std::int32_t> counts = {},
                                     std::uint32_t stride = 1) {
        PendingBatch out;
        out.owner_       = owner;
        out.transaction_ = transaction;
        out.row_count_   = rows.size();
        for (std::size_t i = 0; i < rows.size(); ++i) { out.rows_[i] = rows[i]; }
        out.tokens_     = tokens;
        out.row_counts_ = counts;
        out.row_stride_ = stride;
        out.timing_     = timing;
        return out;
    }
    static const void* owner(const PendingBatch& p) noexcept { return p.owner_; }
    static std::uint64_t transaction(const PendingBatch& p) noexcept { return p.transaction_; }
    static std::span<const SequenceHandle> rows(const PendingBatch& p) noexcept {
        return {p.rows_.data(), p.row_count_};
    }
    static void consume(PendingBatch& p) noexcept {
        p.owner_       = nullptr;
        p.transaction_ = 0;
        p.row_count_   = 0;
        p.tokens_      = {};
        p.row_counts_  = {};
        p.row_stride_  = 0;
    }
};

namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }

ops::SamplingConfig translate(const ResolvedSamplingParameters& s) {
    if (!(s.temperature >= 0.0F) || !(s.top_p > 0.0F && s.top_p <= 1.0F) || s.min_p < 0.0F || s.min_p > 1.0F) {
        throw std::invalid_argument("Qwen4Exp: sampling parameters are out of range");
    }
    ops::SamplingConfig out;
    out.temperature       = s.temperature;
    out.top_k             = s.top_k;
    out.top_p             = s.top_p;
    out.min_p             = s.min_p;
    out.presence_penalty  = s.presence_penalty;
    out.frequency_penalty = s.frequency_penalty;
    out.seed              = s.seed;
    return out;
}

} // namespace

class ProgramImpl {
public:
    enum class Phase : std::uint8_t { Free, Prefill, Decode, Finishable };

    struct Lane {
        Phase phase          = Phase::Free;
        std::uint64_t epoch  = 0;
        std::vector<std::int32_t> history; // prompt, then committed output
        std::uint32_t prompt_tokens = 0;
        std::uint32_t state_tokens  = 0; // positions already in the model state
        ops::SamplingConfig sampling;
        std::optional<DeviceKVPageReservation> reservation;
        std::vector<DeviceKVPageLease> pages;
        KVExecutionRowLease row;
        std::uint64_t prefill_ns      = 0;
        std::uint64_t decode_ns       = 0;
        std::uint64_t decode_share_ns = 0;
        ExpertResidency::Stats cache_at_admission;
        std::uint64_t cpu_served_at_admission = 0;
        std::unique_ptr<qwen3_5::detail::NgramProposer> proposer; // copy proposals over history
        SpeculativeStats speculative;
        // MTP drafter: the residual of the last processed position (its pending cell) is in the
        // lane's saved column; mtp_written says whether that cell's K/V is already written.
        bool mtp_written = false;
        bool mtp_live    = false; // false after a plain round left its cells behind
    };

    ProgramImpl(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options)
        : parameters_(parameters), device_(device), c_(parameters.model.config().text), options_(std::move(options)),
          volume_(options_.ngram_volume, c_.ple.table), hash_(c_.ple.ngram, 0) {
        if (options_.max_concurrency == 0 || options_.max_concurrency > kMaximumConcurrency) {
            throw std::invalid_argument("Qwen4Exp: max_concurrency must be in [1,8]");
        }
        if (options_.kv_cache != KvCacheStorage::BFloat16 && options_.kv_cache != KvCacheStorage::Int8Group64) {
            throw std::invalid_argument("Qwen4Exp supports --kv-dtype bf16 or int8");
        }
        if (options_.max_context == 0 || options_.prefill_chunk == 0) {
            throw std::invalid_argument("Qwen4Exp: max_context and prefill_chunk must be nonzero");
        }
        const auto lanes   = static_cast<std::int32_t>(options_.max_concurrency);
        vocab_             = dim(c_.vocab_size);
        token_domain_      = dim(parameters.model.resources().public_token_count);
        chunk_             = static_cast<std::int32_t>(std::min(options_.prefill_chunk, options_.max_context));
        if (options_.ngram_draft_tokens > 15 ||
            (options_.ngram_draft_tokens > 0 && (options_.ngram_min_match < 4 || options_.ngram_min_match > 64))) {
            throw std::invalid_argument("Qwen4Exp: n-gram drafts must be 0..15 with a minimum match of 4..64");
        }
        mtp_ = parameters_.mtp.has_value();
        if (mtp_ && (options_.mtp_draft_tokens == 0 || options_.mtp_draft_tokens > 7)) {
            throw std::invalid_argument("Qwen4Exp: MTP needs 1..7 draft tokens");
        }
        mtp_k_     = mtp_ ? static_cast<std::int32_t>(options_.mtp_draft_tokens) : 0;
        max_width_ = 1 + std::max(static_cast<std::int32_t>(options_.ngram_draft_tokens), mtp_k_);
        columns_           = std::max(chunk_, lanes * max_width_);
        pages_per_row_     = (dim(options_.max_context) + kPagedKVPageSize - 1) / kPagedKVPageSize;
        const auto kv_tokens = options_.kv_capacity_tokens != 0 ? options_.kv_capacity_tokens
                                                                 : options_.max_context * options_.max_concurrency;
        kv_pages_ = (kv_tokens + kPagedKVPageSize - 1) / kPagedKVPageSize;

        // Recurrent state: one slot per lane.
        LayoutBuilder state_builder;
        const LinearAttentionStatePoolSpec gdn_spec{
            .layers         = c_.gdn_layers,
            .conv_channels  = dim(c_.gdn.conv_channels()),
            .conv_width     = dim(c_.gdn.conv_kernel - 1),
            .value_heads    = dim(c_.gdn.value_heads),
            .value_head_dim = dim(c_.gdn.value_head_dim),
            .key_head_dim   = dim(c_.gdn.key_head_dim),
            .slot_count     = lanes,
        };
        const auto gdn_layout = plan_linear_attention_state_pool(state_builder, gdn_spec);
        state_backing_        = DeviceBuffer(state_builder.finish(256));
        state_backing_.fill(0);
        gdn_ = std::make_unique<LinearAttentionStatePool>(DeviceSpan{state_backing_.p, state_backing_.bytes}, gdn_layout);

        width_   = dim(c_.residual_width());
        span_    = dim(c_.ple.conv_span());
        ple_backing_ = DeviceBuffer(static_cast<std::size_t>(width_) * span_ * lanes * 2);
        ple_backing_.fill(0);
        di_ = dim(c_.qsa.index_head_dim);
        r_  = dim(c_.qsa.compress_ratio);
        // One more tail slab and KV layer for the MTP block (after the text layers).
        const std::uint32_t kv_layers = c_.attention_layers + (mtp_ ? 1U : 0U);
        tails_backing_ = DeviceBuffer(static_cast<std::size_t>(kv_layers) * di_ * (r_ - 1) * lanes * 2);
        tails_backing_.fill(0);

        // Paged KV: one page group holds 64 positions of every attention layer.
        const auto layout = paged_kv_storage_layout(options_.kv_cache, dim(c_.attention.head_dim));
        KVPageGeometry geometry;
        const auto kv_heads = dim(c_.attention.kv_heads);
        for (std::uint32_t l = 0; l < kv_layers; ++l) {
            geometry.planes.push_back({layout.key.data_dtype, layout.key.data_leading_extent, kv_heads});
            if (layout.key.has_scale()) {
                geometry.planes.push_back({layout.key.scale_dtype, layout.key.scale_leading_extent, kv_heads});
            }
            geometry.planes.push_back({layout.value.data_dtype, layout.value.data_leading_extent, kv_heads});
            if (layout.value.has_scale()) {
                geometry.planes.push_back({layout.value.scale_dtype, layout.value.scale_leading_extent, kv_heads});
            }
            geometry.planes.push_back({DType::BF16, di_ / r_, 1});
        }
        LayoutBuilder kv_builder;
        const auto pool_layout =
            plan_device_kv_page_pool(kv_builder, {.page_group_count = kv_pages_, .geometry = geometry});
        const auto table_layout = plan_kv_execution_tables(
            kv_builder, {.logical_page_capacity = static_cast<std::uint32_t>(pages_per_row_), .table_rows = lanes});
        kv_backing_ = DeviceBuffer(kv_builder.finish(256));
        kv_backing_.fill(0);
        pool_   = std::make_unique<DeviceKVPagePool>(DeviceSpan{kv_backing_.p, kv_backing_.bytes}, pool_layout);
        tables_ = std::make_unique<KVExecutionTablePool>(DeviceSpan{kv_backing_.p, kv_backing_.bytes}, table_layout, *pool_);

        // Per-call inputs, staged through pinned memory.
        const std::size_t heads     = hash_.heads();
        const std::size_t row_bytes = c_.ple.table.row_bytes;
        io_layout_.ids       = 0;
        io_layout_.positions = align(io_layout_.ids + 4ULL * columns_);
        io_layout_.slots     = align(io_layout_.positions + 4ULL * columns_);
        io_layout_.rows      = align(io_layout_.slots + 4ULL * lanes);
        io_layout_.columns   = align(io_layout_.rows + 4ULL * lanes);
        io_layout_.ngram     = align(io_layout_.columns + 4ULL * columns_);
        io_layout_.mtp_ids   = align(io_layout_.ngram + row_bytes * heads * columns_);
        io_layout_.mtp_cells = align(io_layout_.mtp_ids + 4ULL * (columns_ + 1));
        io_layout_.bytes     = align(io_layout_.mtp_cells + 4ULL * (columns_ + 1));
        io_device_ = DeviceBuffer(io_layout_.bytes);
        io_host_   = PinnedHostBuffer(io_layout_.bytes);

        logits32_     = DeviceBuffer(sizeof(float) * vocab_ * lanes * max_width_);
        logits16_     = DeviceBuffer(2ULL * vocab_ * lanes * max_width_);
        sampled_      = DeviceBuffer(4ULL * lanes);
        sample_pos_   = DeviceBuffer(4ULL * lanes);
        configs_      = DeviceBuffer(sizeof(ops::SamplingConfig) * lanes);
        token_counts_ = DeviceBuffer(4ULL * token_domain_ * lanes);
        token_counts_.fill(0);
        host_sampled_ = PinnedHostBuffer(4ULL * lanes);
        host_configs_ = PinnedHostBuffer(sizeof(ops::SamplingConfig) * lanes);

        execution::ForwardState state;
        state.gdn      = gdn_.get();
        state.ple_conv = Tensor(ple_backing_.p, DType::BF16, {width_, span_, lanes});
        for (std::uint32_t l = 0; l < c_.attention_layers; ++l) {
            state.qsa_tails.push_back(Tensor(static_cast<std::byte*>(tails_backing_.p) +
                                                 static_cast<std::size_t>(l) * di_ * (r_ - 1) * lanes * 2,
                                             DType::BF16, {di_, r_ - 1, lanes}));
        }
        execution::ForwardKV kv;
        kv.block_tables  = tables_->matrix();
        std::size_t plane = 0;
        for (std::uint32_t l = 0; l < kv_layers; ++l) {
            ops::QsaKVLayer layer;
            layer.kv.storage      = options_.kv_cache;
            layer.kv.head_dim     = dim(c_.attention.head_dim);
            layer.kv.num_kv_heads = kv_heads;
            layer.kv.k_pages      = pool_->plane(plane++);
            if (layout.key.has_scale()) { layer.kv.k_scale_pages = pool_->plane(plane++); }
            layer.kv.v_pages = pool_->plane(plane++);
            if (layout.value.has_scale()) { layer.kv.v_scale_pages = pool_->plane(plane++); }
            layer.pooled_pages = pool_->plane(plane++);
            if (l < c_.attention_layers) {
                kv.layers.push_back(layer);
            } else {
                kv.mtp = layer;
            }
        }
        if (mtp_) {
            state.mtp_tails = Tensor(static_cast<std::byte*>(tails_backing_.p) +
                                         static_cast<std::size_t>(c_.attention_layers) * di_ * (r_ - 1) * lanes * 2,
                                     DType::BF16, {di_, r_ - 1, lanes});
            allocate_mtp(lanes);
            state.mtp_ones = Tensor(mtp_ones_.p, DType::FP32, {dim(c_.hc.streams), mtp_columns_});
        }
        execution::ForwardExperts experts;
        experts.frames.assign(c_.num_hidden_layers, nullptr);
        work_capacity_ = execution::Forward::workspace_bytes(c_, columns_, dim(options_.max_context)) +
                         ops::sampling_workspace_capacity_bytes(token_domain_, 1, lanes);
        if (max_width_ > 1) {
            work_capacity_ += ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                token_domain_, 1, max_width_ - 1, 1, lanes);
            allocate_verification(lanes);
        }
        if (mtp_) {
            const auto& head = parameters_.draft_head;
            work_capacity_ += execution::Forward::mtp_workspace_bytes(
                c_, std::max(lanes * max_width_, std::min(chunk_, 512)), lanes,
                head.rows ? head.rows->weight.n : vocab_, dim(options_.max_context));
        }
        work_    = std::make_unique<WorkspaceArena>(work_capacity_);

        // The VRAM expert cache takes the device memory left over, less a reserve.
        std::vector<const std::uint8_t*> banks;
        std::uint64_t record_stride = 0;
        for (const auto& layer : parameters_.layers) {
            banks.push_back(reinterpret_cast<const std::uint8_t*>(layer.moe.bank->planes.records));
            record_stride = layer.moe.bank->planes.record_stride;
        }
        // Staging slots for each layer call's misses (copied with a compact read window, design
        // 8.6), allocated before the frames take the remaining memory.
        staging_ = DeviceBuffer(static_cast<std::size_t>(kStagingSlots) * record_stride);
        experts.staging_base  = static_cast<std::uint8_t*>(staging_.p);
        experts.staging_slots = kStagingSlots;
        CUDA_CHECK(cudaStreamCreateWithFlags(&overlap_.stream, cudaStreamNonBlocking));
        for (auto& event : overlap_.events) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
        experts.overlap_stream = overlap_.stream;
        experts.overlap_events = overlap_.events;
        // CPU-served misses for decode and verify calls (prefill chunks stay on the GPU).
        const std::uint32_t cpu_workers = options_.cpu_expert_workers;
        const std::uint32_t cpu_jobs    = options_.cpu_expert_jobs;
        if (cpu_workers > 0 && cpu_jobs > 0) {
            std::vector<ops::offloaded_moe::CpuMissService::Layer> service_layers;
            for (const auto& layer : parameters_.layers) {
                service_layers.push_back({.records       = reinterpret_cast<const std::uint8_t*>(layer.moe.bank->planes.records),
                                          .record_stride = layer.moe.bank->planes.record_stride,
                                          .scales        = layer.moe.bank->scales.data()});
            }
            cpu_service_ = std::make_unique<ops::offloaded_moe::CpuMissService>(
                std::move(service_layers),
                ops::offloaded_moe::CpuMissService::Options{
                    .workers     = static_cast<int>(cpu_workers),
                    .max_jobs    = static_cast<int>(std::min<std::uint32_t>(cpu_jobs, ops::offloaded_moe::kMaxCpuJobs)),
                    .max_columns = lanes * max_width_,
                    .pcie_divisor = options_.cpu_pcie_divisor,
                    .cpus        = {}});
            for (std::uint32_t l = 0; l < c_.num_hidden_layers; ++l) {
                experts.cpu.push_back(cpu_service_->channel(static_cast<int>(l)));
            }
        }
        std::uint32_t frames = 0;
        if (options_.expert_cache) {
            std::size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            const std::size_t reserve = options_.expert_cache_reserve_bytes;
            frames = free_bytes > reserve ? static_cast<std::uint32_t>((free_bytes - reserve) / record_stride) : 0U;
        }
        residency_ = std::make_unique<ExpertResidency>(c_, std::move(banks), record_stride, frames, columns_);
        experts.frame_base   = residency_->frame_base();
        experts.frame_stride = residency_->frame_stride();
        experts.route_log    = residency_->route_log();
        experts.route_stride = residency_->route_stride();
        for (std::uint32_t l = 0; l < c_.num_hidden_layers; ++l) { experts.frames[l] = residency_->table(l); }
        forward_ = std::make_unique<execution::Forward>(parameters_, device_, *work_, std::move(state), std::move(kv),
                                                        std::move(experts), dim(options_.max_context));
        device_.synchronize();
    }

    // ---------------------------------------------------------------- admission
    RequestBasePlan plan_request(const PreparedPrompt& prompt, const runtime::ResolvedExecutionOptions& options) {
        const auto& data = qwen3_5::PreparedPromptAccess::view(prompt);
        if (data.has_media()) { throw std::invalid_argument("Qwen4Exp does not serve images or video yet"); }
        const auto n = static_cast<std::uint32_t>(data.token_ids.size());
        if (n == 0 || n > options_.max_context) {
            throw std::invalid_argument("Qwen4Exp: the prompt is empty or exceeds max_context");
        }
        auto base                       = std::make_unique<BasePlanImpl>();
        base->summary.prompt_tokens     = n;
        base->summary.requested_output_tokens = options.requested_output_tokens;
        const std::uint32_t capacity_output   = options_.max_context - n + 1U;
        base->summary.effective_output_tokens = std::min(options.requested_output_tokens, capacity_output);
        base->summary.effective_limit_reason  = options.requested_output_tokens <= capacity_output
                                                    ? FinishReason::OutputLimit
                                                    : FinishReason::ContextCapacity;
        base->summary.prefix_reuse_path       = PrefixReusePath::Root;
        base->summary.publish_continuation    = false;
        const std::uint64_t chunks = 1ULL + (n - 1ULL) / static_cast<std::uint64_t>(chunk_);
        base->summary.service_work_quanta =
            chunks + (base->summary.effective_output_tokens == 0 ? 0ULL : base->summary.effective_output_tokens - 1ULL);
        base->sampling = translate(options.sampling);
        const std::uint32_t positions = std::min(options_.max_context, n + base->summary.effective_output_tokens);
        base->pages = (positions + kPagedKVPageSize - 1) / kPagedKVPageSize;
        return RequestBasePlan(std::move(base));
    }

    bool feasible(const RequestBasePlan& base) const noexcept {
        return base.impl_ != nullptr && base.impl_->pages <= pool_->capacity_pages() &&
               base.impl_->pages <= static_cast<std::uint32_t>(pages_per_row_);
    }

    HybridAdmissionQuote quote(const RequestBasePlan& base, runtime::LaneId destination) {
        HybridAdmissionQuote out;
        out.destination = destination;
        out.summary     = base.summary();
        if (destination.value >= options_.max_concurrency || lanes_[destination.value].phase != Phase::Free ||
            transaction_lane_ || pool_->available_pages() < base.impl_->pages) {
            out.readiness = runtime::Readiness::TemporarilyBlocked;
            return out;
        }
        auto impl      = std::make_shared<QuoteImpl>();
        impl->summary  = base.summary();
        impl->sampling = base.impl_->sampling;
        impl->pages    = base.impl_->pages;
        impl->lane     = destination.value;
        out.impl       = std::move(impl);
        out.readiness  = runtime::Readiness::Ready;
        return out;
    }

    runtime::ContextTransactionReserveStatus reserve(HybridAdmissionQuote&& quote, PreparedPrompt&& prompt,
                                                     runtime::CancellationFlagView cancellation) {
        if (cancellation.requested() || quote.impl == nullptr) { return runtime::ContextTransactionReserveStatus::Aborted; }
        const QuoteImpl& q = *quote.impl;
        Lane& lane         = lanes_.at(q.lane);
        if (lane.phase != Phase::Free || transaction_lane_) {
            throw std::logic_error("Qwen4Exp: admission destination is not free");
        }
        auto reservation = pool_->reserve(q.pages);
        if (!reservation) { return runtime::ContextTransactionReserveStatus::Aborted; }
        lane.pages.clear();
        lane.pages.reserve(q.pages);
        pool_->materialize(*reservation, q.pages, lane.pages);
        lane.reservation = std::move(*reservation);
        lane.row         = tables_->acquire(static_cast<std::int32_t>(q.lane));
        tables_->publish(lane.row.handle(), 0, std::span<const DeviceKVPageLease>(lane.pages), device_.stream);
        reset_slot(q.lane);

        const auto& data    = qwen3_5::PreparedPromptAccess::view(prompt);
        lane.history.assign(data.token_ids.begin(), data.token_ids.end());
        lane.history.reserve(lane.history.size() + q.summary.effective_output_tokens + 1);
        lane.prompt_tokens  = static_cast<std::uint32_t>(lane.history.size());
        lane.state_tokens   = 0;
        lane.speculative    = {};
        lane.mtp_written = false;
        lane.mtp_live    = mtp_;
        if (max_width_ > 1) {
            if (options_.ngram_draft_tokens > 0) {
                lane.proposer =
                    std::make_unique<qwen3_5::detail::NgramProposer>(proposer_tokens_, proposer_tokens_ / 2);
                for (const auto token : lane.history) { lane.proposer->append(token); }
            }
            lane.speculative.enabled      = true;
            lane.speculative.backend      = mtp_ ? SpeculativeBackend::Mtp : SpeculativeBackend::None;
            lane.speculative.draft_window = static_cast<std::uint32_t>(max_width_ - 1);
            lane.speculative.accepted_per_position.assign(static_cast<std::size_t>(max_width_ - 1), 0);
        }
        lane.sampling       = q.sampling;
        lane.sampling.token_counts =
            static_cast<std::int32_t*>(token_counts_.p) + static_cast<std::size_t>(q.lane) * token_domain_;
        lane.epoch           = ++next_epoch_;
        lane.phase           = Phase::Prefill;
        lane.prefill_ns      = 0;
        lane.decode_ns       = 0;
        lane.decode_share_ns = 0;
        lane.cache_at_admission = residency_->stats();
        lane.cpu_served_at_admission = cpu_service_ ? cpu_service_->served_experts() : 0;
        transaction_lane_ = q.lane;
        published_        = false;
        ++revision_;
        return runtime::ContextTransactionReserveStatus::Reserved;
    }

    ContextTransactionProgress progress(runtime::CancellationFlagView) {
        if (!transaction_lane_ || published_) { throw std::logic_error("Qwen4Exp: no admission to progress"); }
        published_ = true;
        MaterializationResult out;
        out.status    = runtime::ContextTransactionStatus::Published;
        out.published = StartResult{handle(*transaction_lane_)};
        return out;
    }

    void finalize() noexcept {
        transaction_lane_.reset();
        published_ = false;
    }

    [[nodiscard]] bool in_transaction() const noexcept { return transaction_lane_.has_value(); }

    // ---------------------------------------------------------------- execution
    PrefillProgress advance_prefill(SequenceHandle sequence) {
        const auto start  = Clock::now();
        const auto index  = lane_of(sequence);
        Lane& lane        = lanes_[index];
        if (lane.phase != Phase::Prefill) { throw std::logic_error("Qwen4Exp: prefill on a lane that is not prefilling"); }
        const std::int32_t begin = static_cast<std::int32_t>(lane.state_tokens);
        const std::int32_t width =
            std::min<std::int32_t>(chunk_, static_cast<std::int32_t>(lane.prompt_tokens) - begin);
        const bool last = begin + width == static_cast<std::int32_t>(lane.prompt_tokens);
        stage_sequence(index, begin, width, lane.history);
        const auto chunk = stage_mtp_chunk(lane, index, begin, width);
        run(1, width, 1, chunk ? &*chunk : nullptr);
        lane.state_tokens += static_cast<std::uint32_t>(width);
        if (!last) {
            device_.synchronize();
            residency_->after_round(device_.stream, width, kPrefillPromotionsPerLayer);
        }

        PrefillProgress out;
        out.summary                 = runtime::BeginSummary{.prompt_tokens        = lane.prompt_tokens,
                                                            .reused_prompt_tokens = 0,
                                                            .prefix_reuse_path    = PrefixReusePath::Root};
        out.processed_prompt_tokens = static_cast<std::uint32_t>(width);
        out.complete                = last;
        if (last) {
            const std::uint32_t lanes[] = {index};
            const std::int32_t positions[] = {begin + width - 1};
            sample(lanes, positions);
            residency_->after_round(device_.stream, width, kPrefillPromotionsPerLayer);
            const SequenceHandle rows[] = {sequence};
            out.timing.submit_host_ns = elapsed_ns(start);
            lane.prefill_ns += out.timing.submit_host_ns;
            out.pending.emplace(ContractAccess::make_pending(this, ++next_transaction_, rows,
                                                             {pending_tokens_.data(), 1}, out.timing));
            pending_transaction_ = next_transaction_;
        } else {
            out.timing.submit_host_ns = elapsed_ns(start);
            lane.prefill_ns += out.timing.submit_host_ns;
        }
        return out;
    }

    PendingBatch decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets) {
        const auto start = Clock::now();
        const auto batch = static_cast<std::int32_t>(sequences.size());
        if (batch <= 0 || batch > static_cast<std::int32_t>(options_.max_concurrency)) {
            throw std::logic_error("Qwen4Exp: decode batch is empty or too large");
        }
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<std::int32_t, kMaximumConcurrency> positions{};
        for (std::int32_t b = 0; b < batch; ++b) {
            lanes[b]   = lane_of(sequences[b]);
            Lane& lane = lanes_[lanes[b]];
            if (lane.phase != Phase::Decode || lane.history.size() != lane.state_tokens + 1ULL) {
                throw std::logic_error("Qwen4Exp: decode row is not at a committed frontier");
            }
            positions[b] = static_cast<std::int32_t>(lane.state_tokens);
        }
        // Proposals: a round verifies 1 + the longest row's drafts; a row may draft at most one token
        // fewer than it may still emit, so every verified position lies in its reservation. MTP
        // drafts come first; a longer n-gram copy proposal replaces them.
        if (mtp_) {
            mtp_draft(std::span<const std::uint32_t>(lanes.data(), batch), budgets);
        }
        std::int32_t width = 1;
        for (std::int32_t b = 0; b < batch && max_width_ > 1; ++b) {
            Lane& lane = lanes_[lanes[b]];
            drafts_[b].clear();
            from_ngram_[b] = false;
            const std::uint32_t remaining =
                static_cast<std::size_t>(b) < budgets.size() ? budgets[b].generated_tokens_remaining : 0U;
            if (remaining < 2) { continue; }
            const auto limit = std::min<std::uint32_t>(static_cast<std::uint32_t>(max_width_ - 1), remaining - 1U);
            if (mtp_ && lane.mtp_live) {
                const auto n = std::min<std::uint32_t>(limit, static_cast<std::uint32_t>(mtp_k_));
                drafts_[b].assign(mtp_drafts_[b].begin(), mtp_drafts_[b].begin() + n);
            }
            if (lane.proposer) {
                auto copy = lane.proposer->propose(lane.history, limit, options_.ngram_min_match).tokens;
                if (copy.size() > drafts_[b].size()) {
                    drafts_[b]     = std::move(copy);
                    from_ngram_[b] = true;
                }
            }
            width = std::max(width, 1 + static_cast<std::int32_t>(drafts_[b].size()));
        }
        if (width > 1) {
            return verify(sequences, std::span<const std::uint32_t>(lanes.data(), batch),
                          std::span<const std::int32_t>(positions.data(), batch), width, start);
        }
        for (std::int32_t b = 0; b < batch; ++b) {
            ++lanes_[lanes[b]].speculative.fallback_steps;
            lanes_[lanes[b]].mtp_live = false; // a plain round leaves the drafter's cells behind
        }
        round_width_ = 1;
        stage_decode(std::span<const std::uint32_t>(lanes.data(), batch), std::span<const std::int32_t>(positions.data(), batch));
        run_decode(batch);
        for (std::int32_t b = 0; b < batch; ++b) { ++lanes_[lanes[b]].state_tokens; }
        sample(std::span<const std::uint32_t>(lanes.data(), batch), std::span<const std::int32_t>(positions.data(), batch));
        residency_->after_round(device_.stream, batch, decode_budget(1));
        runtime::ExecutionTiming timing;
        timing.submit_host_ns = elapsed_ns(start);
        for (std::int32_t b = 0; b < batch; ++b) {
            lanes_[lanes[b]].decode_ns += timing.submit_host_ns;
            lanes_[lanes[b]].decode_share_ns += timing.submit_host_ns / static_cast<std::uint64_t>(batch);
        }
        pending_transaction_  = ++next_transaction_;
        return ContractAccess::make_pending(this, pending_transaction_, sequences,
                                            {pending_tokens_.data(), static_cast<std::size_t>(batch)}, timing);
    }

    runtime::ExecutionTiming append_forced(std::span<const SequenceHandle> sequences, std::span<const TokenId> tokens,
                                           std::uint32_t stride) {
        const auto start = Clock::now();
        if (stride == 0 || tokens.size() != static_cast<std::size_t>(stride) * sequences.size()) {
            throw std::invalid_argument("Qwen4Exp: forced tokens do not match their rows");
        }
        for (std::size_t row = 0; row < sequences.size(); ++row) {
            const auto index = lane_of(sequences[row]);
            Lane& lane       = lanes_[index];
            if (lane.phase != Phase::Decode || lane.history.size() != lane.state_tokens + 1ULL) {
                throw std::logic_error("Qwen4Exp: forced tokens on a lane that is not at a committed frontier");
            }
            const auto forced = tokens.subspan(row * stride, stride);
            // The model reads the pending input token and every forced token but the last, which
            // becomes the next decode input.
            lane.history.insert(lane.history.end(), forced.begin(), forced.end());
            if (lane.proposer) {
                for (const auto token : forced) { lane.proposer->append(token); }
            }
            const auto begin = static_cast<std::int32_t>(lane.state_tokens);
            stage_sequence(index, begin, static_cast<std::int32_t>(stride), lane.history);
            const auto chunk = stage_mtp_chunk(lane, index, begin, static_cast<std::int32_t>(stride));
            run(1, static_cast<std::int32_t>(stride), 1, chunk ? &*chunk : nullptr);
            device_.synchronize();
            residency_->after_round(device_.stream, static_cast<std::int32_t>(stride), kDecodePromotionsPerLayer);
            lane.state_tokens += stride;
        }
        runtime::ExecutionTiming timing;
        timing.submit_host_ns = elapsed_ns(start);
        return timing;
    }

    CommitResult commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions) {
        if (ContractAccess::owner(pending) != this || ContractAccess::transaction(pending) != pending_transaction_ ||
            pending_transaction_ == 0) {
            throw std::logic_error("Qwen4Exp: commit of a foreign or stale pending batch");
        }
        const auto rows = ContractAccess::rows(pending);
        if (decisions.size() != rows.size()) { throw std::logic_error("Qwen4Exp: commit decisions are not row aligned"); }
        if (round_width_ > 1) { return commit_verified(std::move(pending), decisions); }
        CommitResult out;
        out.row_count = rows.size();
        for (std::size_t row = 0; row < rows.size(); ++row) {
            const auto index = lane_of(rows[row]);
            Lane& lane       = lanes_[index];
            const auto& d    = decisions[row];
            if (d.cancelled) {
                out.rows[row].timings     = timings(lane);
                out.rows[row].speculative = lane.speculative;
                out.rows[row].disposition = runtime::CommitDisposition::CancelledReleased;
                release(index);
                continue;
            }
            if (d.accepted_tokens != 1) { throw std::logic_error("Qwen4Exp: a row commits exactly one token"); }
            lane.history.push_back(pending_tokens_[row]);
            // After a plain decode round the drafter's pending cell is two positions back; it
            // resumes only from a chunk (the prefill's last) whose pending cell is this one.
            if (lane.phase != Phase::Prefill && mtp_) { lane.mtp_live = false; }
            if (lane.proposer) { lane.proposer->append(pending_tokens_[row]); }
            if (lane.phase == Phase::Prefill) { lane.phase = Phase::Decode; }
            if (d.terminal) {
                lane.phase                = Phase::Finishable;
                out.rows[row].disposition = runtime::CommitDisposition::Finishable;
            } else {
                out.rows[row].disposition = runtime::CommitDisposition::Active;
            }
        }
        ContractAccess::consume(pending);
        pending_transaction_ = 0;
        return out;
    }

    DiscardResult discard(PendingBatch&& pending) noexcept {
        DiscardResult out;
        if (ContractAccess::owner(pending) != this || ContractAccess::transaction(pending) != pending_transaction_) {
            return out;
        }
        const auto rows = ContractAccess::rows(pending);
        for (const auto& row : rows) {
            const auto index = ContractAccess::lane(row);
            if (index < options_.max_concurrency && lanes_[index].phase != Phase::Free) { release(index); }
        }
        out.status    = runtime::ConsumeStatus::Consumed;
        out.row_count = rows.size();
        ContractAccess::consume(pending);
        pending_transaction_ = 0;
        return out;
    }

    FinishResult finish(SequenceHandle sequence) noexcept {
        FinishResult out;
        try {
            const auto index = lane_of(sequence);
            if (lanes_[index].phase != Phase::Finishable) { return out; }
            out.timings     = timings(lanes_[index]);
            out.speculative = lanes_[index].speculative;
            report_cache(lanes_[index]);
            release(index);
            out.status      = runtime::ConsumeStatus::Consumed;
            out.disposition = runtime::FinishDisposition::Released;
        } catch (...) {}
        return out;
    }

    AbortResult abort(SequenceHandle sequence) noexcept {
        AbortResult out;
        try {
            const auto index = lane_of(sequence);
            out.timings      = timings(lanes_[index]);
            out.speculative  = lanes_[index].speculative;
            release(index);
            out.status = runtime::ConsumeStatus::Consumed;
        } catch (...) {}
        return out;
    }

    void release_all() noexcept {
        for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
            if (lanes_[i].phase != Phase::Free) { release(i); }
        }
        transaction_lane_.reset();
        published_           = false;
        pending_transaction_ = 0;
        try { device_.synchronize(); } catch (...) {}
    }

    PhysicalUsageSnapshot usage() const noexcept {
        PhysicalUsageSnapshot out;
        out.resource_revision.value = revision_;
        for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
            if (lanes_[i].phase != Phase::Free) { ++out.device_state_slots; }
        }
        out.device_main_kv_pages = pool_->allocated_pages();
        return out;
    }

    MemorySummary memory() const noexcept {
        MemorySummary out;
        out.max_context             = options_.max_context;
        out.kv_capacity             = kv_pages_ * static_cast<std::uint32_t>(kPagedKVPageSize);
        out.kv_capacity_page_groups = kv_pages_;
        out.kv_capacity_max_page_groups = kv_pages_;
        out.kv_cache                = options_.kv_cache;
        out.kv_payload_bytes        = kv_backing_.bytes;
        out.workspace_logical_peak_bytes = work_capacity_;
        return out;
    }

    std::uint64_t revision() const noexcept { return revision_; }

private:
    // Promotions per layer call (design section 19.2): one per decode round measured fastest; a
    // promotion moves as many PCIe bytes as serving the expert once zero-copy.
    static constexpr std::size_t kDecodePromotionsPerLayer  = 1;
    // Once the frames are full, decode promotes only every kDecodePromotionInterval-th round: a
    // promotion moves as many PCIe bytes as serving its expert once, and fewer of them measured
    // faster at 512 tokens (design section 19.2). Until then every round promotes, so a cold cache
    // fills at the full rate.
    static constexpr std::uint64_t kDecodePromotionInterval = 4;
    // Misses staged per pass of a layer's experts: every decode and verify call's misses in one
    // pass; a prefill chunk's in several.
    static constexpr std::int32_t kStagingSlots = 64;
    // Promotions follow the tokens a round advances (the longest row's), not rounds: a verification
    // round that advances four tokens promotes what four decode rounds would, so a speculative
    // round's cache warms per token as plain decode does.
    std::size_t decode_budget(std::uint32_t tokens) {
        if (residency_->stats().promotions < residency_->frames()) { return kDecodePromotionsPerLayer * tokens; }
        budget_tokens_ += tokens;
        const std::uint64_t due = budget_tokens_ / kDecodePromotionInterval;
        budget_tokens_ %= kDecodePromotionInterval;
        return static_cast<std::size_t>(due) * kDecodePromotionsPerLayer;
    }
    std::uint64_t budget_tokens_ = 0;
    static constexpr std::size_t kPrefillPromotionsPerLayer = 16;

    // Reports the expert cache over the finished request (and since start) as an Engine diagnostic,
    // or on stderr without an observer.
    void report_cache(const Lane& lane) noexcept {
        try {
            const auto& at_admission = lane.cache_at_admission;
            const auto& s       = residency_->stats();
            const auto routed   = s.routed - at_admission.routed;
            const auto hits     = s.hits - at_admission.hits;
            const auto promoted = s.promotions - at_admission.promotions;
            const auto cpu      = cpu_service_ ? cpu_service_->served_experts() - lane.cpu_served_at_admission : 0;
            std::size_t free_vram = 0, total_vram = 0;
            if (cudaMemGetInfo(&free_vram, &total_vram) != cudaSuccess) { free_vram = 0; }
            char text[384];
            std::snprintf(text, sizeof(text),
                          "expert cache: %u frames; request %.1f%% of %llu routed experts hit, %llu promotions, "
                          "%llu misses CPU-served; since start %.1f%%; VRAM free %zu MiB",
                          residency_->frames(), routed ? 100.0 * static_cast<double>(hits) / static_cast<double>(routed) : 0.0,
                          static_cast<unsigned long long>(routed), static_cast<unsigned long long>(promoted),
                          static_cast<unsigned long long>(cpu),
                          s.routed ? 100.0 * static_cast<double>(s.hits) / static_cast<double>(s.routed) : 0.0,
                          free_vram >> 20);
            if (options_.diagnostics.callback) {
                options_.diagnostics.callback(Diagnostic{.level = DiagnosticLevel::Info, .message = text});
            } else {
                std::fprintf(stderr, "[engine] %s\n", text);
            }
        } catch (...) {}
    }

    struct IoLayout {
        std::size_t ids = 0, positions = 0, slots = 0, rows = 0, columns = 0, ngram = 0, mtp_ids = 0, mtp_cells = 0,
                    bytes = 0;
    };

    static std::size_t align(std::size_t v) { return (v + 255) / 256 * 256; }

    // The io bytes a decode or verification round of `columns` columns reads: everything before
    // the n-gram rows (ids, positions, slots, table rows, logit columns) and its columns' rows.
    // Prefill and forced-token calls copy the whole layout (their MTP cells follow the rows).
    std::size_t io_prefix(std::int32_t columns) const {
        return io_layout_.ngram + static_cast<std::size_t>(columns) * hash_.heads() * c_.ple.table.row_bytes;
    }

    SequenceHandle handle(std::uint32_t lane) const noexcept {
        return ContractAccess::make_sequence(this, lane, lanes_[lane].epoch);
    }

    std::uint32_t lane_of(const SequenceHandle& sequence) const {
        const auto lane = ContractAccess::lane(sequence);
        if (ContractAccess::owner(sequence) != this || lane >= options_.max_concurrency ||
            lanes_[lane].phase == Phase::Free || lanes_[lane].epoch != ContractAccess::epoch(sequence)) {
            throw std::logic_error("Qwen4Exp: stale or foreign sequence handle");
        }
        return lane;
    }

    GenerationTimings timings(const Lane& lane) const {
        GenerationTimings out;
        out.prefill_seconds      = static_cast<double>(lane.prefill_ns) * 1e-9;
        out.decode_seconds       = static_cast<double>(lane.decode_ns) * 1e-9;
        out.decode_share_seconds = static_cast<double>(lane.decode_share_ns) * 1e-9;
        return out;
    }

    void release(std::uint32_t index) noexcept {
        Lane& lane = lanes_[index];
        lane.pages.clear();
        lane.reservation.reset();
        lane.row = KVExecutionRowLease{};
        lane.history.clear();
        lane.history.shrink_to_fit();
        lane.proposer.reset();
        lane.phase        = Phase::Free;
        lane.state_tokens = 0;
        ++revision_;
    }

    // Zeroes a slot's recurrent state, PLE convolution history, QSA tails and penalty counts.
    void reset_slot(std::uint32_t slot) {
        const cudaStream_t s = device_.stream;
        for (std::uint32_t l = 0; l < c_.gdn_layers; ++l) {
            Tensor conv      = gdn_->conv_slot(l, static_cast<std::int32_t>(slot));
            Tensor recurrent = gdn_->recurrent_slot(l, static_cast<std::int32_t>(slot));
            CUDA_CHECK(cudaMemsetAsync(conv.data, 0, conv.bytes(), s));
            CUDA_CHECK(cudaMemsetAsync(recurrent.data, 0, recurrent.bytes(), s));
        }
        const std::size_t ple_bytes = static_cast<std::size_t>(width_) * span_ * 2;
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(ple_backing_.p) + slot * ple_bytes, 0, ple_bytes, s));
        const auto lanes            = static_cast<std::size_t>(options_.max_concurrency);
        const std::size_t tail_bytes = static_cast<std::size_t>(di_) * (r_ - 1) * 2;
        for (std::uint32_t l = 0; l < c_.attention_layers + (mtp_ ? 1U : 0U); ++l) {
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(tails_backing_.p) + (l * lanes + slot) * tail_bytes, 0,
                                       tail_bytes, s));
        }
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::int32_t*>(token_counts_.p) + slot * static_cast<std::size_t>(token_domain_),
                                   0, 4ULL * token_domain_, s));
    }

    std::byte* host_io() const { return static_cast<std::byte*>(io_host_.data()); }

    // The n-gram rows of `count` positions starting at `begin`, from the tokens before them.
    void stage_ngram(const std::vector<std::int32_t>& history, std::int32_t begin, std::int32_t count,
                     std::size_t column) {
        const std::int32_t context = static_cast<std::int32_t>(c_.ple.ngram.ngram_size) - 1;
        window_.clear();
        for (std::int32_t p = begin - context; p < begin + count; ++p) {
            window_.push_back(p < 0 ? static_cast<std::int32_t>(c_.eos_token_id) : history[static_cast<std::size_t>(p)]);
        }
        const std::size_t heads = hash_.heads();
        row_ids_.resize(static_cast<std::size_t>(count) * heads);
        hash_.row_ids(window_, static_cast<std::size_t>(count), row_ids_.data());
        const std::size_t row_bytes = c_.ple.table.row_bytes;
        volume_.read_rows(row_ids_, std::span<std::byte>(host_io() + io_layout_.ngram + column * heads * row_bytes,
                                                         static_cast<std::size_t>(count) * heads * row_bytes));
    }

    void stage_sequence(std::uint32_t lane, std::int32_t begin, std::int32_t width,
                        const std::vector<std::int32_t>& history) {
        auto* ids       = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.ids);
        auto* positions = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.positions);
        auto* columns   = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.columns);
        for (std::int32_t i = 0; i < width; ++i) {
            ids[i]       = history[static_cast<std::size_t>(begin + i)];
            positions[i] = begin + i;
        }
        columns[0] = width - 1;
        reinterpret_cast<std::int32_t*>(host_io() + io_layout_.slots)[0] = static_cast<std::int32_t>(lane);
        reinterpret_cast<std::int32_t*>(host_io() + io_layout_.rows)[0]  = static_cast<std::int32_t>(lane);
        stage_ngram(history, begin, width, 0);
        host_lanes_[0] = static_cast<std::int32_t>(lane);
    }

    void stage_decode(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions) {
        auto* ids     = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.ids);
        auto* pos     = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.positions);
        auto* columns = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.columns);
        auto* slots   = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.slots);
        auto* rows    = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.rows);
        for (std::size_t b = 0; b < lanes.size(); ++b) {
            const Lane& lane = lanes_[lanes[b]];
            ids[b]           = lane.history[static_cast<std::size_t>(positions[b])];
            pos[b]           = positions[b];
            columns[b]       = static_cast<std::int32_t>(b);
            slots[b]         = static_cast<std::int32_t>(lanes[b]);
            rows[b]          = static_cast<std::int32_t>(lanes[b]);
            host_lanes_[b]   = static_cast<std::int32_t>(lanes[b]);
            stage_ngram(lane.history, positions[b], 1, b);
        }
    }

    // Runs one eager Forward call over the staged inputs (prefill chunks, forced tokens); logits of
    // `logit_columns` columns land in logits32_.
    void run(std::int32_t batch, std::int32_t width, std::int32_t logit_columns,
             const execution::MtpChunk* chunk = nullptr) {
        const cudaStream_t s = device_.stream;
        CUDA_CHECK(cudaMemcpyAsync(io_device_.p, io_host_.data(), io_layout_.bytes, cudaMemcpyHostToDevice, s));
        residency_->before_round(s);
        forward_call(batch, width, logit_columns, nullptr, chunk);
        residency_->enqueue_route_download(s, batch * width);
    }

    // One decode round of `batch` sequences: its inputs are staged at fixed device addresses and
    // every row is chosen on the device, so one CUDA graph per batch size serves every round. The
    // first round of a size runs eagerly (it also performs the Ops' one-time setup); the next is
    // captured and later rounds replay it.
    void run_decode(std::int32_t batch) {
        const cudaStream_t s = device_.stream;
        CUDA_CHECK(cudaMemcpyAsync(io_device_.p, io_host_.data(), io_prefix(batch), cudaMemcpyHostToDevice, s));
        residency_->before_round(s);
        replay(graphs_[static_cast<std::size_t>(batch - 1)], [&] { forward_call(batch, 1, batch); });
        residency_->enqueue_route_download(s, batch);
    }

    struct DecodeGraph {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        bool warmed = false;
    };

    // Runs a fixed-address round: the first time eagerly (the Ops' one-time setup), the second
    // captured, then by replay.
    void replay(DecodeGraph& graph, const std::function<void()>& body) {
        const cudaStream_t s = device_.stream;
        if (graph.executable.ready()) {
            graph.executable.launch(s);
        } else if (!graph.warmed) {
            body();
            graph.warmed = true;
        } else {
            graph.definition.capture(s, body);
            graph.executable.instantiate(graph.definition);
            graph.executable.launch(s);
        }
    }

    void forward_call(std::int32_t batch, std::int32_t width, std::int32_t logit_columns,
                      const execution::ForwardVerify* verify = nullptr, const execution::MtpChunk* chunk = nullptr) {
        const std::int32_t cols = batch * width;
        auto* base = static_cast<std::byte*>(io_device_.p);
        execution::ForwardBatch fb;
        fb.ids           = Tensor(base + io_layout_.ids, DType::I32, {cols});
        fb.positions     = Tensor(base + io_layout_.positions, DType::I32, {cols});
        fb.slots         = Tensor(base + io_layout_.slots, DType::I32, {batch});
        fb.table_rows    = Tensor(base + io_layout_.rows, DType::I32, {batch});
        fb.logit_columns = Tensor(base + io_layout_.columns, DType::I32, {logit_columns});
        fb.ngram_rows    = Tensor(base + io_layout_.ngram, DType::U8,
                                  {dim(c_.ple.table.row_bytes), dim(hash_.heads()), cols});
        fb.host_slots      = std::span<const std::int32_t>(host_lanes_.data(), static_cast<std::size_t>(batch));
        fb.host_table_rows = fb.host_slots;
        fb.batch           = batch;
        fb.width           = width;
        fb.verify          = verify;
        fb.mtp_chunk       = chunk;
        // Decode and verification rounds export their final residuals for the MTP catch-up.
        if (mtp_ && chunk == nullptr) { fb.residual_out = Tensor(mtp_residuals_.p, DType::BF16, {width_, cols}); }
        Tensor logits(logits32_.p, DType::FP32, {vocab_, logit_columns});
        forward_->run(fb, logits);
    }

    // ---------------------------------------------------------------- speculative verification
    // Device I32 arrays of a verification round, each with room for every lane and width.
    struct SpecLayout {
        std::size_t target = 0, drafts = 0, extents = 0, lengths = 0, anchors = 0, licensed = 0, counts = 0,
                    accepted = 0, commit = 0, words = 0;
    };

    void allocate_verification(std::int32_t lanes) {
        const GdnReplayRecordSpec spec{.layers          = dim(c_.gdn_layers),
                                       .record_capacity = lanes,
                                       .width           = max_width_,
                                       .conv_channels   = dim(c_.gdn.conv_channels()),
                                       .qk_heads        = dim(c_.gdn.key_heads),
                                       .value_heads     = dim(c_.gdn.value_heads),
                                       .key_dim         = dim(c_.gdn.key_head_dim),
                                       .value_dim       = dim(c_.gdn.value_head_dim)};
        LayoutBuilder builder;
        const auto layout = plan_gdn_replay_records(builder, spec);
        records_backing_  = DeviceBuffer(builder.finish(256));
        records_          = GdnReplayRecords(DeviceSpan{records_backing_.p, records_backing_.bytes}, layout);
        ple_records_      = DeviceBuffer(2ULL * width_ * max_width_ * lanes);
        qsa_records_      = DeviceBuffer(2ULL * di_ * max_width_ * lanes * c_.attention_layers);
        folds_.resize(static_cast<std::size_t>(max_width_) + 1);
        verify_graphs_.resize(static_cast<std::size_t>(lanes) * max_width_);
        const std::size_t cells = static_cast<std::size_t>(lanes) * max_width_;
        std::size_t at          = 0;
        const auto take = [&](std::size_t words) {
            const std::size_t begin = at;
            at += (words + 63) / 64 * 64;
            return begin;
        };
        spec_layout_.target   = take(cells);
        spec_layout_.drafts   = take(cells);
        spec_layout_.extents  = take(lanes);
        spec_layout_.lengths  = take(lanes);
        spec_layout_.anchors  = take(lanes);
        spec_layout_.licensed = take(cells);
        spec_layout_.counts   = take(lanes);
        spec_layout_.accepted = take(lanes);
        spec_layout_.commit   = take(lanes);
        spec_layout_.words    = at;
        spec_device_          = DeviceBuffer(4ULL * at);
        spec_host_            = PinnedHostBuffer(4ULL * at);
        pending_tokens_.assign(cells, 0);
        live_.assign(cells, 0);
        proposer_tokens_ = std::bit_ceil(std::max<std::size_t>(64, options_.max_context + 64ULL));
    }

    std::int32_t* spec_host(std::size_t offset) const { return static_cast<std::int32_t*>(spec_host_.data()) + offset; }
    std::int32_t* spec_device(std::size_t offset) const { return static_cast<std::int32_t*>(spec_device_.p) + offset; }

    execution::ForwardVerify verify_view(std::int32_t batch, std::int32_t width) const {
        execution::ForwardVerify v;
        v.gdn        = records_.narrowed(width);
        v.ple_inputs = Tensor(ple_records_.p, DType::BF16, {width_, width, batch});
        v.qsa_keys   = Tensor(qsa_records_.p, DType::BF16, {di_, width, batch, dim(c_.attention_layers)});
        return v;
    }

    const ops::GdnReplayFoldPlan& fold(std::int32_t width) {
        auto& plan = folds_.at(static_cast<std::size_t>(width));
        if (!plan) { plan.emplace(records_.narrowed(width), gdn_->all_layers_view()); }
        return *plan;
    }

    // One verification round: every row's anchor and drafts (rows with fewer drafts are padded with
    // filler drafts past their extent), one forward over batch x width columns that leaves all
    // recurrent state in place, then on-device acceptance. The pending batch licenses each row's
    // accepted drafts and its correction or bonus token; commit folds the accepted prefix in.
    PendingBatch verify(std::span<const SequenceHandle> sequences, std::span<const std::uint32_t> lanes,
                        std::span<const std::int32_t> positions, std::int32_t width, Clock::time_point start) {
        const cudaStream_t s = device_.stream;
        const auto batch     = static_cast<std::int32_t>(lanes.size());
        const std::int32_t W = width, K = width - 1;
        auto* ids     = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.ids);
        auto* pos     = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.positions);
        auto* columns = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.columns);
        auto* slots   = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.slots);
        auto* rows    = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.rows);
        for (std::int32_t b = 0; b < batch; ++b) {
            const Lane& lane          = lanes_[lanes[b]];
            const auto& d             = drafts_[b];
            const auto n              = static_cast<std::int32_t>(d.size());
            const std::int32_t anchor = lane.history.back();
            sequence_.assign(lane.history.begin(), lane.history.end());
            for (std::int32_t j = 0; j < K; ++j) {
                const std::int32_t token = j < n ? d[static_cast<std::size_t>(j)] : (n > 0 ? d.back() : anchor);
                sequence_.push_back(token);
                spec_host(spec_layout_.drafts)[b * K + j] = token;
            }
            for (std::int32_t j = 0; j < W; ++j) {
                ids[b * W + j]     = sequence_[static_cast<std::size_t>(positions[b] + j)];
                pos[b * W + j]     = positions[b] + j;
                columns[b * W + j] = b * W + j;
            }
            slots[b]       = static_cast<std::int32_t>(lanes[b]);
            rows[b]        = static_cast<std::int32_t>(lanes[b]);
            host_lanes_[b] = static_cast<std::int32_t>(lanes[b]);
            stage_ngram(sequence_, positions[b], W, static_cast<std::size_t>(b * W));
            spec_host(spec_layout_.extents)[b] = n;
            spec_host(spec_layout_.lengths)[b] = positions[b];
            spec_host(spec_layout_.anchors)[b] = anchor;
        }
        CUDA_CHECK(cudaMemcpyAsync(io_device_.p, io_host_.data(), io_prefix(batch * W), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(spec_device_.p, spec_host_.data(), 4ULL * spec_layout_.licensed,
                                   cudaMemcpyHostToDevice, s));
        residency_->before_round(s);
        replay(verify_graphs_[static_cast<std::size_t>(batch - 1) * max_width_ + (W - 1)], [&] {
            const execution::ForwardVerify view = verify_view(batch, W);
            forward_call(batch, W, batch * W, &view);
        });
        residency_->enqueue_route_download(s, batch * W);

        // Acceptance on the device, from BF16-rounded logits as in plain decode (design 16.5).
        Tensor wide(logits32_.p, DType::FP32, {vocab_, batch * W});
        Tensor narrow(logits16_.p, DType::BF16, {vocab_, batch * W});
        ops::cast_fp32_to_bf16(wide, narrow, s);
        Tensor target(spec_device(spec_layout_.target), DType::I32, {batch * W});
        ops::argmax(narrow, target, token_domain_, s);
        auto* configs = static_cast<ops::SamplingConfig*>(host_configs_.data());
        for (std::int32_t b = 0; b < batch; ++b) { configs[b] = lanes_[lanes[b]].sampling; }
        CUDA_CHECK(cudaMemcpyAsync(configs_.p, configs, sizeof(ops::SamplingConfig) * batch, cudaMemcpyHostToDevice, s));
        Tensor drafts(spec_device(spec_layout_.drafts), DType::I32, {K, batch});
        Tensor extents(spec_device(spec_layout_.extents), DType::I32, {batch});
        Tensor lengths(spec_device(spec_layout_.lengths), DType::I32, {batch});
        Tensor anchors(spec_device(spec_layout_.anchors), DType::I32, {batch});
        Tensor licensed(spec_device(spec_layout_.licensed), DType::I32, {W, batch});
        Tensor counts(spec_device(spec_layout_.counts), DType::I32, {batch});
        Tensor accepted(spec_device(spec_layout_.accepted), DType::I32, {batch});
        {
            auto scope = work_->scope();
            ops::speculative_accept_greedy_drafts(target.view({W, batch}), narrow.view({vocab_, W, batch}), drafts,
                                                  extents, lengths, anchors, licensed, counts, accepted,
                                                  token_domain_, static_cast<const ops::SamplingConfig*>(configs_.p),
                                                  *work_, s);
        }
        CUDA_CHECK(cudaMemcpyAsync(spec_host(spec_layout_.licensed), licensed.data,
                                   4ULL * (spec_layout_.commit - spec_layout_.licensed), cudaMemcpyDeviceToHost, s));
        device_.synchronize();

        const std::int32_t* licensed_host = spec_host(spec_layout_.licensed);
        for (std::int32_t b = 0; b < batch; ++b) {
            Lane& lane     = lanes_[lanes[b]];
            const auto n   = static_cast<std::uint64_t>(drafts_[b].size());
            const auto L   = spec_host(spec_layout_.counts)[b];
            const auto A   = spec_host(spec_layout_.accepted)[b];
            if (L < 1 || L > W || A != L - 1) { throw std::logic_error("Qwen4Exp: acceptance returned an invalid extent"); }
            for (std::int32_t j = 0; j < W; ++j) {
                pending_tokens_[static_cast<std::size_t>(b * W + j)] = j < L ? licensed_host[b * W + j] : 0;
                live_[static_cast<std::size_t>(b * W + j)]           = j <= A ? 1 : 0;
            }
            pending_counts_[b] = L;
            auto& stats        = lane.speculative;
            if (n == 0) {
                ++stats.fallback_steps;
                continue;
            }
            ++stats.rounds;
            stats.drafted_tokens += n;
            stats.accepted_tokens += static_cast<std::uint64_t>(A);
            if (from_ngram_[b]) {
                ++stats.ngram_rounds;
                stats.ngram_drafted_tokens += n;
                stats.ngram_accepted_tokens += static_cast<std::uint64_t>(A);
            }
            for (std::int32_t j = 0; j < A; ++j) { ++stats.accepted_per_position[static_cast<std::size_t>(j)]; }
        }
        std::uint32_t advanced = 1;
        for (std::int32_t b = 0; b < batch; ++b) {
            advanced = std::max<std::uint32_t>(advanced, static_cast<std::uint32_t>(pending_counts_[b]));
        }
        residency_->after_round(s, batch * W, decode_budget(advanced),
                                std::span<const std::uint8_t>(live_.data(), static_cast<std::size_t>(batch * W)));
        round_width_ = W;
        runtime::ExecutionTiming timing;
        timing.submit_host_ns = elapsed_ns(start);
        for (std::int32_t b = 0; b < batch; ++b) {
            lanes_[lanes[b]].decode_ns += timing.submit_host_ns;
            lanes_[lanes[b]].decode_share_ns += timing.submit_host_ns / static_cast<std::uint64_t>(batch);
        }
        pending_transaction_ = ++next_transaction_;
        return ContractAccess::make_pending(this, pending_transaction_, sequences,
                                            {pending_tokens_.data(), static_cast<std::size_t>(batch * W)}, timing,
                                            {pending_counts_.data(), static_cast<std::size_t>(batch)},
                                            static_cast<std::uint32_t>(W));
    }

    // Commits a verification round: each row keeps its first accepted_tokens licensed tokens, so
    // the model state advances by that many columns (the anchor and the accepted drafts before the
    // last kept token). The GDN fold, the QSA tails and the PLE history replay those columns' records.
    CommitResult commit_verified(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions) {
        const cudaStream_t s = device_.stream;
        const auto rows      = ContractAccess::rows(pending);
        const auto batch     = static_cast<std::int32_t>(rows.size());
        const std::int32_t W = round_width_;
        CommitResult out;
        out.row_count = rows.size();
        std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> fold_rows{};
        std::int32_t* commit = spec_host(spec_layout_.commit);
        for (std::int32_t row = 0; row < batch; ++row) {
            const auto index = ContractAccess::lane(rows[row]);
            Lane& lane       = lanes_.at(index);
            const auto& d    = decisions[static_cast<std::size_t>(row)];
            fold_rows[row]   = {static_cast<std::int32_t>(index), static_cast<std::int32_t>(index), 0};
            commit[row]      = 0;
            if (d.cancelled) {
                out.rows[row].timings     = timings(lane);
                out.rows[row].speculative = lane.speculative;
                out.rows[row].disposition = runtime::CommitDisposition::CancelledReleased;
                continue;
            }
            const auto k = static_cast<std::int32_t>(d.accepted_tokens);
            if (k < 1 || k > pending_counts_[row]) { throw std::logic_error("Qwen4Exp: commit exceeds the licensed tokens"); }
            for (std::int32_t j = 0; j < k; ++j) {
                const std::int32_t token = pending_tokens_[static_cast<std::size_t>(row * W + j)];
                lane.history.push_back(token);
                if (lane.proposer) { lane.proposer->append(token); }
            }
            fold_rows[row].commit_columns = k;
            commit[row]                   = k;
            lane.state_tokens += static_cast<std::uint32_t>(k);
            if (d.terminal) {
                lane.phase                = Phase::Finishable;
                out.rows[row].disposition = runtime::CommitDisposition::Finishable;
            } else {
                out.rows[row].disposition = runtime::CommitDisposition::Active;
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(spec_device(spec_layout_.commit), commit, 4ULL * batch, cudaMemcpyHostToDevice, s));
        fold(W).execute(std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), static_cast<std::size_t>(batch)), s);
        auto* io = static_cast<std::byte*>(io_device_.p);
        const Tensor commit_columns(spec_device(spec_layout_.commit), DType::I32, {batch});
        const Tensor slots(io + io_layout_.slots, DType::I32, {batch});
        const Tensor positions(io + io_layout_.positions, DType::I32, {W, batch});
        const auto layers = dim(c_.attention_layers);
        Tensor tails(tails_backing_.p, DType::BF16, {di_, r_ - 1, dim(options_.max_concurrency), layers});
        ops::qsa_commit_tails(Tensor(qsa_records_.p, DType::BF16, {di_, W, batch, layers}), positions, commit_columns,
                              tails, slots, execution::qsa_geometry(c_), s);
        Tensor ple_states(ple_backing_.p, DType::BF16, {width_, span_, dim(options_.max_concurrency)});
        ops::ple_conv_commit(Tensor(ple_records_.p, DType::BF16, {width_, W, batch}), commit_columns, ple_states,
                             slots, s);
        if (mtp_) { mtp_catch_up(rows, W, commit); }
        for (std::int32_t row = 0; row < batch; ++row) {
            if (decisions[static_cast<std::size_t>(row)].cancelled) { release(ContractAccess::lane(rows[row])); }
        }
        ContractAccess::consume(pending);
        pending_transaction_ = 0;
        round_width_         = 1;
        return out;
    }

    // ---------------------------------------------------------------- MTP drafter (design 11.2)
    void allocate_mtp(std::int32_t lanes) {
        const std::size_t column = 2ULL * width_;
        mtp_columns_   = std::max(lanes * max_width_, std::min(chunk_, 512));
        mtp_residuals_ = DeviceBuffer(column * lanes * max_width_);
        mtp_saved_     = DeviceBuffer(column * lanes);
        mtp_chain_     = DeviceBuffer(column * lanes);
        mtp_records_   = DeviceBuffer(2ULL * di_ * max_width_ * lanes);
        mtp_saved_.fill(0);
        std::vector<float> ones(static_cast<std::size_t>(c_.hc.streams) * mtp_columns_, 1.0F);
        mtp_ones_ = DeviceBuffer(ones.size() * sizeof(float));
        mtp_ones_.copy_from_host(ones.data(), ones.size() * sizeof(float));
        // I32: chain ids [B], cells [K, B], drafts [K, B], catch-up ids [W, B], gather columns [B],
        // catch-up cells [W, B].
        mtp_io_.ids      = 0;
        mtp_io_.cells    = mtp_io_.ids + lanes;
        mtp_io_.drafts   = mtp_io_.cells + lanes * mtp_k_;
        mtp_io_.up_ids   = mtp_io_.drafts + lanes * mtp_k_;
        mtp_io_.gather   = mtp_io_.up_ids + lanes * max_width_;
        mtp_io_.up_cells = mtp_io_.gather + lanes;
        mtp_io_.words    = mtp_io_.up_cells + lanes * max_width_;
        mtp_device_      = DeviceBuffer(4ULL * mtp_io_.words);
        mtp_host_        = PinnedHostBuffer(4ULL * mtp_io_.words);
        for (auto& d : mtp_drafts_) { d.assign(static_cast<std::size_t>(mtp_k_), 0); }
        mtp_catch_graphs_.resize(static_cast<std::size_t>(lanes) * max_width_);
    }

    std::int32_t* mtp_host(std::size_t offset) const { return static_cast<std::int32_t*>(mtp_host_.data()) + offset; }
    std::int32_t* mtp_device(std::size_t offset) const { return static_cast<std::int32_t*>(mtp_device_.p) + offset; }
    Tensor saved_column(std::uint32_t lane) const {
        return Tensor(static_cast<std::byte*>(mtp_saved_.p) + 2ULL * width_ * lane, DType::BF16, {width_});
    }

    // The MTP cells of a chunk (prefill or forced tokens) of one lane. The pending cell (the
    // position before the chunk) is prepended unless written; the chunk's last cell is included
    // only when the token after it is known (forced tokens). Stages ids and cells into the io.
    std::optional<execution::MtpChunk> stage_mtp_chunk(Lane& lane, std::uint32_t index, std::int32_t begin,
                                                       std::int32_t width) {
        if (!mtp_) { return std::nullopt; }
        const bool prepend = begin > 0 && !lane.mtp_written;
        const bool known   = static_cast<std::int32_t>(lane.history.size()) > begin + width;
        const std::int32_t columns = known ? width : width - 1;
        const std::int32_t first   = prepend ? begin - 1 : begin;
        const std::int32_t cells   = (prepend ? 1 : 0) + columns;
        auto* ids   = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.mtp_ids);
        auto* cells_host = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.mtp_cells);
        for (std::int32_t i = 0; i < cells; ++i) {
            cells_host[i] = first + i;
            ids[i]        = lane.history[static_cast<std::size_t>(first + i + 1)];
        }
        lane.mtp_written = known;
        lane.mtp_live    = true;
        auto* base = static_cast<std::byte*>(io_device_.p);
        execution::MtpChunk chunk;
        chunk.saved     = saved_column(index);
        chunk.prepend   = prepend;
        chunk.columns   = columns;
        // A one-token prompt has no cell yet (its only position is the pending cell).
        if (cells > 0) {
            chunk.ids       = Tensor(base + io_layout_.mtp_ids, DType::I32, {cells});
            chunk.positions = Tensor(base + io_layout_.mtp_cells, DType::I32, {cells});
        }
        return chunk;
    }

    // MTP drafts for every live lane of a decode round: the pending cell's K/V when unwritten,
    // then mtp_k_ chained full steps from the saved residual and the anchor token.
    void mtp_draft(std::span<const std::uint32_t> lanes, std::span<const runtime::RoundBudget>) {
        const cudaStream_t s = device_.stream;
        const auto batch     = static_cast<std::int32_t>(lanes.size());
        const std::size_t column = 2ULL * width_;
        bool any = false, unwritten = false;
        auto* slots = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.slots);
        auto* rows  = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.rows);
        for (std::int32_t b = 0; b < batch; ++b) {
            const Lane& lane = lanes_[lanes[b]];
            any |= lane.mtp_live;
            unwritten |= lane.mtp_live && !lane.mtp_written;
            const std::int32_t cell  = static_cast<std::int32_t>(lane.state_tokens) - 1;
            const std::int32_t limit = static_cast<std::int32_t>(lane.pages.size()) * kPagedKVPageSize - 1;
            mtp_host(mtp_io_.ids)[b] = lane.history.back();
            for (std::int32_t j = 0; j < mtp_k_; ++j) {
                mtp_host(mtp_io_.cells)[j * batch + b] = std::min(cell + j, limit);
            }
            slots[b] = static_cast<std::int32_t>(lanes[b]);
            rows[b]  = static_cast<std::int32_t>(lanes[b]);
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(mtp_chain_.p) + column * b, saved_column(lanes[b]).data,
                                       column, cudaMemcpyDeviceToDevice, s));
        }
        if (!any) { return; }
        auto* io = static_cast<std::byte*>(io_device_.p);
        CUDA_CHECK(cudaMemcpyAsync(io + io_layout_.slots, slots, 4ULL * batch, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(io + io_layout_.rows, rows, 4ULL * batch, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(mtp_device_.p, mtp_host_.data(), 4ULL * mtp_io_.drafts, cudaMemcpyHostToDevice, s));
        const Tensor slot_tensor(io + io_layout_.slots, DType::I32, {batch});
        const Tensor row_tensor(io + io_layout_.rows, DType::I32, {batch});
        Tensor chain(mtp_chain_.p, DType::BF16, {width_, batch});
        Tensor anchors(mtp_device(mtp_io_.ids), DType::I32, {batch});
        if (unwritten) {
            // The pending cells' K/V (rewriting a written cell is idempotent).
            forward_->run_mtp({.residuals  = chain,
                               .ids        = anchors,
                               .positions  = Tensor(mtp_device(mtp_io_.cells), DType::I32, {batch}),
                               .slots      = slot_tensor,
                               .table_rows = row_tensor,
                               .batch      = batch,
                               .width      = 1,
                               .kv_only    = true});
        }
        replay(mtp_draft_graphs_[static_cast<std::size_t>(batch - 1)], [&] {
        for (std::int32_t j = 0; j < mtp_k_; ++j) {
            Tensor drafts(mtp_device(mtp_io_.drafts + static_cast<std::size_t>(j) * batch), DType::I32, {batch});
            forward_->run_mtp({.residuals    = chain,
                               .ids          = j == 0 ? anchors
                                                      : Tensor(mtp_device(mtp_io_.drafts + static_cast<std::size_t>(j - 1) * batch),
                                                               DType::I32, {batch}),
                               .positions    = Tensor(mtp_device(mtp_io_.cells + static_cast<std::size_t>(j) * batch),
                                                      DType::I32, {batch}),
                               .slots        = slot_tensor,
                               .table_rows   = row_tensor,
                               .batch        = batch,
                               .width        = 1,
                               .kv_only      = false,
                               .residual_out = chain,
                               .drafts       = drafts});
        }
        });
        CUDA_CHECK(cudaMemcpyAsync(mtp_host(mtp_io_.drafts), mtp_device(mtp_io_.drafts), 4ULL * mtp_k_ * batch,
                                   cudaMemcpyDeviceToHost, s));
        device_.synchronize();
        for (std::int32_t b = 0; b < batch; ++b) {
            Lane& lane = lanes_[lanes[b]];
            if (unwritten && lane.mtp_live) { lane.mtp_written = true; }
            for (std::int32_t j = 0; j < mtp_k_; ++j) {
                mtp_drafts_[b][static_cast<std::size_t>(j)] = mtp_host(mtp_io_.drafts)[j * batch + b];
            }
        }
    }

    // After a verification round's commit: the drafter's cells of the verified window from its
    // residuals and committed tokens (index-key tails advance by the committed count), and each
    // row's last committed residual becomes its pending cell, already written.
    void mtp_catch_up(std::span<const SequenceHandle> rows, std::int32_t W, const std::int32_t* commit) {
        const cudaStream_t s = device_.stream;
        const auto batch     = static_cast<std::int32_t>(rows.size());
        const std::size_t column = 2ULL * width_;
        auto* io = static_cast<std::byte*>(io_device_.p);
        for (std::int32_t b = 0; b < batch; ++b) {
            const std::int32_t k = commit[b];
            for (std::int32_t j = 0; j < W; ++j) {
                mtp_host(mtp_io_.up_ids)[b * W + j] =
                    j < k ? pending_tokens_[static_cast<std::size_t>(b * W + j)] : pending_tokens_[static_cast<std::size_t>(b * W)];
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(mtp_device(mtp_io_.up_ids), mtp_host(mtp_io_.up_ids), 4ULL * batch * W,
                                   cudaMemcpyHostToDevice, s));
        const Tensor positions(io + io_layout_.positions, DType::I32, {W * batch});
        const Tensor slots(io + io_layout_.slots, DType::I32, {batch});
        replay(mtp_catch_graphs_[static_cast<std::size_t>(batch - 1) * max_width_ + (W - 1)], [&] {
        forward_->run_mtp({.residuals   = Tensor(mtp_residuals_.p, DType::BF16, {width_, W * batch}),
                           .ids         = Tensor(mtp_device(mtp_io_.up_ids), DType::I32, {W * batch}),
                           .positions   = positions,
                           .slots       = slots,
                           .table_rows  = Tensor(io + io_layout_.rows, DType::I32, {batch}),
                           .batch       = batch,
                           .width       = W,
                           .kv_only     = true,
                           .key_records = Tensor(mtp_records_.p, DType::BF16, {di_, W, batch})});
        Tensor tails(static_cast<std::byte*>(tails_backing_.p) +
                         static_cast<std::size_t>(c_.attention_layers) * di_ * (r_ - 1) * options_.max_concurrency * 2,
                     DType::BF16, {di_, r_ - 1, dim(options_.max_concurrency), 1});
        ops::qsa_commit_tails(Tensor(mtp_records_.p, DType::BF16, {di_, W, batch, 1}), positions.view({W, batch}),
                              Tensor(spec_device(spec_layout_.commit), DType::I32, {batch}), tails, slots,
                              execution::qsa_geometry(c_), s);
        });
        for (std::int32_t b = 0; b < batch; ++b) {
            const std::int32_t k = commit[b];
            const auto lane      = ContractAccess::lane(rows[static_cast<std::size_t>(b)]);
            if (k < 1) { continue; }
            CUDA_CHECK(cudaMemcpyAsync(saved_column(lane).data,
                                       static_cast<const std::byte*>(mtp_residuals_.p) + column * (b * W + k - 1),
                                       column, cudaMemcpyDeviceToDevice, s));
            lanes_[lane].mtp_written = true;
        }
    }

    // Samples the next token of each row from logits32_ (row b = column b) into pending_tokens_.
    void sample(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions) {
        const cudaStream_t s = device_.stream;
        const auto batch     = static_cast<std::int32_t>(lanes.size());
        Tensor wide(logits32_.p, DType::FP32, {vocab_, batch});
        Tensor narrow(logits16_.p, DType::BF16, {vocab_, batch});
        // The sampler reads BF16 logits; BF16 and FP32 logits measured equal in perplexity
        // (design §16.5).
        ops::cast_fp32_to_bf16(wide, narrow, s);
        auto* configs = static_cast<ops::SamplingConfig*>(host_configs_.data());
        auto* sample_positions = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.positions);
        for (std::int32_t b = 0; b < batch; ++b) {
            configs[b]          = lanes_[lanes[b]].sampling;
            sample_positions[b] = positions[b];
        }
        CUDA_CHECK(cudaMemcpyAsync(configs_.p, configs, sizeof(ops::SamplingConfig) * batch, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(sample_pos_.p, sample_positions, 4ULL * batch, cudaMemcpyHostToDevice, s));
        Tensor out(sampled_.p, DType::I32, {batch});
        Tensor logical(sample_pos_.p, DType::I32, {batch});
        {
            auto scope = work_->scope();
            ops::sample(narrow, out, token_domain_, static_cast<const ops::SamplingConfig*>(configs_.p), logical,
                        ops::kSamplePurposeDecode, *work_, s);
        }
        CUDA_CHECK(cudaMemcpyAsync(host_sampled_.data(), sampled_.p, 4ULL * batch, cudaMemcpyDeviceToHost, s));
        device_.synchronize();
        const auto* tokens = static_cast<const std::int32_t*>(host_sampled_.data());
        for (std::int32_t b = 0; b < batch; ++b) { pending_tokens_[b] = tokens[b]; }
    }

    const execution::Parameters& parameters_;
    DeviceContext& device_;
    const TextConfig& c_;
    ProgramOptions options_;
    NgramVolume volume_;
    NgramHash hash_;

    std::int32_t vocab_ = 0, token_domain_ = 0, chunk_ = 0, columns_ = 0, pages_per_row_ = 0;
    std::int32_t width_ = 0, span_ = 0, di_ = 0, r_ = 0;
    std::uint32_t kv_pages_ = 0;

    DeviceBuffer state_backing_, ple_backing_, tails_backing_, kv_backing_, staging_;
    DeviceBuffer io_device_, logits32_, logits16_, sampled_, sample_pos_, configs_, token_counts_;
    PinnedHostBuffer io_host_{1}, host_sampled_{1}, host_configs_{1};
    IoLayout io_layout_;
    std::unique_ptr<LinearAttentionStatePool> gdn_;
    std::unique_ptr<DeviceKVPagePool> pool_;
    std::unique_ptr<KVExecutionTablePool> tables_;
    std::size_t work_capacity_ = 0;
    std::unique_ptr<WorkspaceArena> work_;
    std::unique_ptr<ExpertResidency> residency_;
    std::unique_ptr<ops::offloaded_moe::CpuMissService> cpu_service_;
    // Prefill staging overlap (see ForwardExperts); destroyed after every call has completed.
    struct OverlapResources {
        ~OverlapResources() {
            for (auto event : events) {
                if (event != nullptr) { cudaEventDestroy(event); }
            }
            if (stream != nullptr) { cudaStreamDestroy(stream); }
        }
        cudaStream_t stream = nullptr;
        std::array<cudaEvent_t, 5> events{};
    };
    OverlapResources overlap_;
    std::unique_ptr<execution::Forward> forward_;
    std::array<DecodeGraph, kMaximumConcurrency> graphs_;
    std::array<DecodeGraph, kMaximumConcurrency> mtp_draft_graphs_;
    std::vector<DecodeGraph> mtp_catch_graphs_; // (batch - 1) * max_width_ + width - 1
    std::vector<DecodeGraph> verify_graphs_; // (batch - 1) * max_width_ + width - 1

    std::array<Lane, kMaximumConcurrency> lanes_{};
    std::array<std::int32_t, kMaximumConcurrency> host_lanes_{};
    std::vector<TokenId> pending_tokens_ = std::vector<TokenId>(kMaximumConcurrency, 0);
    std::array<std::int32_t, kMaximumConcurrency> pending_counts_{};

    // MTP drafter.
    struct MtpIo {
        std::size_t ids = 0, cells = 0, drafts = 0, up_ids = 0, gather = 0, up_cells = 0, words = 0;
    };
    bool mtp_            = false;
    std::int32_t mtp_k_  = 0;
    std::int32_t mtp_columns_ = 0;
    DeviceBuffer mtp_residuals_, mtp_saved_, mtp_chain_, mtp_records_, mtp_ones_, mtp_device_;
    PinnedHostBuffer mtp_host_{1};
    MtpIo mtp_io_;
    std::array<std::vector<std::int32_t>, kMaximumConcurrency> mtp_drafts_;
    std::array<bool, kMaximumConcurrency> from_ngram_{};

    // Speculative verification (max_width_ > 1).
    std::int32_t max_width_   = 1;
    std::int32_t round_width_ = 1;
    std::size_t proposer_tokens_ = 0;
    DeviceBuffer records_backing_, ple_records_, qsa_records_, spec_device_;
    PinnedHostBuffer spec_host_{1};
    SpecLayout spec_layout_;
    GdnReplayRecords records_;
    std::vector<std::optional<ops::GdnReplayFoldPlan>> folds_;
    std::array<std::vector<std::int32_t>, kMaximumConcurrency> drafts_;
    std::vector<std::int32_t> sequence_;
    std::vector<std::uint8_t> live_;
    std::vector<std::int32_t> window_;
    std::vector<std::uint32_t> row_ids_;
    std::optional<std::uint32_t> transaction_lane_;
    bool published_                    = false;
    std::uint64_t next_epoch_          = 0;
    std::uint64_t next_transaction_    = 0;
    std::uint64_t pending_transaction_ = 0;
    std::uint64_t revision_            = 1;
};

} // namespace detail

RequestBasePlan::RequestBasePlan(std::unique_ptr<detail::BasePlanImpl> impl) noexcept : impl_(std::move(impl)) {}
RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept            = default;
RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;
RequestBasePlan::~RequestBasePlan()                                     = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept { return impl_->summary; }

Program::Program(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options)
    : impl_(std::make_unique<detail::ProgramImpl>(parameters, device, std::move(options))) {}

Program::~Program() noexcept = default;

RequestBasePlan Program::plan_request(const PreparedPrompt& prompt, const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(prompt, options);
}

bool Program::isolated_request_feasible(const RequestBasePlan& base) const noexcept { return impl_->feasible(base); }

HybridAdmissionQuote Program::hybrid_quote(const PreparedPrompt&, const RequestBasePlan& base,
                                           runtime::LaneId destination) {
    return impl_->quote(base, destination);
}

runtime::ContextTransactionReserveStatus Program::hybrid_reserve_materialization(
    HybridAdmissionQuote&& quote, PreparedPrompt&& prompt, runtime::CancellationFlagView cancellation) {
    return impl_->reserve(std::move(quote), std::move(prompt), cancellation);
}

ContextTransactionProgress Program::progress_context_transaction(runtime::CancellationFlagView cancellation) {
    return impl_->progress(cancellation);
}

void Program::finalize_context_transaction() noexcept { impl_->finalize(); }

bool Program::has_context_transaction() const noexcept { return impl_->in_transaction(); }

PrefillProgress Program::advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming*,
                                         runtime::PrefillStepWidth) {
    return impl_->advance_prefill(sequence);
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming*) {
    return impl_->decode(sequences, budgets);
}

runtime::ExecutionTiming Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
                                                       std::span<const TokenId> row_major_tokens,
                                                       std::uint32_t row_stride,
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

std::optional<PhysicalUsageSnapshot> Program::fail_all_cleanup() noexcept {
    impl_->release_all();
    return std::nullopt;
}

runtime::ProgramResourceRevision Program::resource_revision() const noexcept {
    return runtime::ProgramResourceRevision{impl_->revision()};
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->usage(); }

MemorySummary Program::memory_summary() const noexcept { return impl_->memory(); }

} // namespace ninfer::models::qwen4_exp
