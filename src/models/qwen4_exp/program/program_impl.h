#pragma once

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
#include "core/vram_budget.h"
#include "core/weight_view.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/ngram_proposer.h"
#include "models/qwen4_exp/execution/forward.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/frontend/ngram_hash.h"
#include "models/qwen4_exp/memory_plan.h"
#include "models/qwen4_exp/program/expert_residency.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "models/qwen4_exp/program/route_trace.h"
#include "models/qwen4_exp/program/vram_monitor.h"
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
#include <mutex>
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

using Clock = std::chrono::steady_clock;

inline std::size_t align(std::size_t v) { return (v + 255) / 256 * 256; }

// Per-call inputs staged through pinned memory, one device copy.
struct IoLayout {
    std::size_t ids = 0, positions = 0, slots = 0, rows = 0, columns = 0, ngram = 0, mtp_ids = 0, mtp_cells = 0,
                bytes = 0;
};

// Device I32 arrays of a verification round, each with room for every lane and width.
struct SpecLayout {
    std::size_t target = 0, drafts = 0, extents = 0, lengths = 0, anchors = 0, licensed = 0, counts = 0,
                accepted = 0, commit = 0, words = 0;
};

// I32 MTP drafter io: chain ids [B], cells [K, B], drafts [K, B], catch-up ids [W, B], gather
// columns [B], catch-up cells [W, B].
struct MtpIo {
    std::size_t ids = 0, cells = 0, drafts = 0, up_ids = 0, gather = 0, up_cells = 0, words = 0;
};

// Every fixed device allocation of a Program and the layouts in it (design §19.3.7). One function
// (ProgramImpl::plan_device) computes it from the options and the configuration alone: before the
// weights are read, for the startup check, and in the constructor, which allocates exactly it.
struct DeviceLayout {
    std::int32_t lanes = 0, token_domain = 0, head_rows = 0, chunk = 0, mtp_k = 0, max_width = 1, columns = 0,
                 pages_per_row = 0, mtp_columns = 0;
    bool mtp = false;
    std::uint32_t kv_pages = 0, kv_layers = 0;
    std::uint64_t record_stride = 0;
    LinearAttentionStatePoolLayout gdn;
    DeviceKVPagePoolLayout pool;
    KVExecutionTableLayout tables;
    GdnReplayRecordLayout records;
    IoLayout io;
    SpecLayout spec;
    MtpIo mtp_io;
    std::size_t gdn_bytes = 0, ple = 0, tails = 0, kv = 0, records_bytes = 0, ple_records = 0, qsa_records = 0;
    std::size_t logits32 = 0, logits16 = 0, token_counts = 0, work = 0, staging = 0;
    std::size_t mtp_column_bytes = 0, mtp_ones = 0;
    ProgramDevicePlan bytes;
};

inline std::uint64_t elapsed_ns(Clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

inline std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }

inline ops::SamplingConfig translate(const ResolvedSamplingParameters& s) {
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
        NgramVolume::Counters ngram; // this request's n-gram row traffic
        std::unique_ptr<qwen3_5::detail::NgramProposer> proposer; // copy proposals over history
        SpeculativeStats speculative;
        // MTP drafter (design 11.2; rule MR1 of 19.3.1): its KV cells [0, mtp_cells) are final. At
        // every settled point the lane's saved column holds the residual of position
        // state_tokens - 1, whose cell is written (mtp_cells == state_tokens) or pending
        // (state_tokens - 1: the next call or round writes it).
        std::uint32_t mtp_cells = 0;
        bool mtp_live           = false;
        // Draft-length policy (design 11.3): conditional acceptance of each draft position.
        std::array<double, 8> mtp_accept{};
        std::uint64_t mtp_policy_rounds = 0;
    };

    // Everything the Program allocates on the device besides the expert frames; validates the
    // options.
    static DeviceLayout plan_device(const ProgramOptions& o, const Config& config, std::int32_t public_tokens) {
        const TextConfig& c = config.text;
        if (o.max_concurrency == 0 || o.max_concurrency > kMaximumConcurrency) {
            throw std::invalid_argument("Qwen4Exp: max_concurrency must be in [1,8]");
        }
        if (o.kv_cache != KvCacheStorage::BFloat16 && o.kv_cache != KvCacheStorage::Int8Group64) {
            throw std::invalid_argument("Qwen4Exp supports --kv-dtype bf16 or int8");
        }
        if (o.max_context == 0 || o.prefill_chunk == 0) {
            throw std::invalid_argument("Qwen4Exp: max_context and prefill_chunk must be nonzero");
        }
        if (o.ngram_draft_tokens > 15 ||
            (o.ngram_draft_tokens > 0 && (o.ngram_min_match < 4 || o.ngram_min_match > 64))) {
            throw std::invalid_argument("Qwen4Exp: n-gram drafts must be 0..15 with a minimum match of 4..64");
        }
        if (config.mtp && (o.mtp_draft_tokens == 0 || o.mtp_draft_tokens > 7)) {
            throw std::invalid_argument("Qwen4Exp: MTP needs 1..7 draft tokens");
        }
        DeviceLayout d;
        d.lanes         = static_cast<std::int32_t>(o.max_concurrency);
        d.token_domain  = public_tokens;
        d.mtp           = config.mtp;
        d.head_rows     = config.proposal_rows != 0 ? dim(config.proposal_rows) : dim(c.vocab_size);
        d.chunk         = static_cast<std::int32_t>(std::min(o.prefill_chunk, o.max_context));
        d.mtp_k         = d.mtp ? static_cast<std::int32_t>(o.mtp_draft_tokens) : 0;
        d.max_width     = 1 + std::max(static_cast<std::int32_t>(o.ngram_draft_tokens), d.mtp_k);
        d.columns       = std::max(d.chunk, d.lanes * d.max_width);
        d.pages_per_row = (dim(o.max_context) + kPagedKVPageSize - 1) / kPagedKVPageSize;
        const auto kv_tokens = o.kv_capacity_tokens != 0 ? o.kv_capacity_tokens : o.max_context * o.max_concurrency;
        d.kv_pages      = (kv_tokens + kPagedKVPageSize - 1) / kPagedKVPageSize;
        const std::int32_t lanes = d.lanes, W = d.max_width;
        const std::size_t width = c.residual_width(), di = c.qsa.index_head_dim, r = c.qsa.compress_ratio;

        // Recurrent state: one slot per lane; the PLE history; one QSA tail slab per attention
        // layer, and one more for the MTP block.
        LayoutBuilder state_builder;
        d.gdn       = plan_linear_attention_state_pool(state_builder, {
                                                                    .layers         = c.gdn_layers,
                                                                    .conv_channels  = dim(c.gdn.conv_channels()),
                                                                    .conv_width     = dim(c.gdn.conv_kernel - 1),
                                                                    .value_heads    = dim(c.gdn.value_heads),
                                                                    .value_head_dim = dim(c.gdn.value_head_dim),
                                                                    .key_head_dim   = dim(c.gdn.key_head_dim),
                                                                    .slot_count     = lanes,
                                                                });
        d.gdn_bytes = state_builder.finish(256);
        d.ple       = width * c.ple.conv_span() * lanes * 2;
        d.kv_layers = c.attention_layers + (d.mtp ? 1U : 0U);
        d.tails     = d.kv_layers * di * (r - 1) * lanes * 2;

        // Paged KV: one page group holds 64 positions of every attention layer.
        const auto storage = paged_kv_storage_layout(o.kv_cache, dim(c.attention.head_dim));
        KVPageGeometry geometry;
        const auto kv_heads = dim(c.attention.kv_heads);
        for (std::uint32_t l = 0; l < d.kv_layers; ++l) {
            geometry.planes.push_back({storage.key.data_dtype, storage.key.data_leading_extent, kv_heads});
            if (storage.key.has_scale()) {
                geometry.planes.push_back({storage.key.scale_dtype, storage.key.scale_leading_extent, kv_heads});
            }
            geometry.planes.push_back({storage.value.data_dtype, storage.value.data_leading_extent, kv_heads});
            if (storage.value.has_scale()) {
                geometry.planes.push_back({storage.value.scale_dtype, storage.value.scale_leading_extent, kv_heads});
            }
            geometry.planes.push_back({DType::BF16, dim(di / r), 1});
        }
        LayoutBuilder kv_builder;
        d.pool   = plan_device_kv_page_pool(kv_builder, {.page_group_count = d.kv_pages, .geometry = geometry});
        d.tables = plan_kv_execution_tables(
            kv_builder, {.logical_page_capacity = static_cast<std::uint32_t>(d.pages_per_row), .table_rows = lanes});
        d.kv = kv_builder.finish(256);

        // Per-call inputs, logits and sampling counts.
        const std::size_t heads     = NgramHash(c.ple.ngram, 0).heads();
        const std::size_t row_bytes = c.ple.table.row_bytes;
        const std::size_t columns   = static_cast<std::size_t>(d.columns);
        d.io.ids       = 0;
        d.io.positions = align(d.io.ids + 4ULL * columns);
        d.io.slots     = align(d.io.positions + 4ULL * columns);
        d.io.rows      = align(d.io.slots + 4ULL * lanes);
        d.io.columns   = align(d.io.rows + 4ULL * lanes);
        d.io.ngram     = align(d.io.columns + 4ULL * columns);
        d.io.mtp_ids   = align(d.io.ngram + row_bytes * heads * columns);
        d.io.mtp_cells = align(d.io.mtp_ids + 4ULL * (columns + 1));
        d.io.bytes     = align(d.io.mtp_cells + 4ULL * (columns + 1));
        d.logits32     = sizeof(float) * c.vocab_size * lanes * W;
        d.logits16     = 2ULL * c.vocab_size * lanes * W;
        d.token_counts = 4ULL * static_cast<std::size_t>(d.token_domain) * lanes;

        // One workspace arena: the forward pass, sampling, draft acceptance and the drafter.
        d.work = workspace_bytes(c, o, d, lanes);

        // Verification (W > 1): GDN replay records, PLE and QSA records, the round's I32 arrays.
        if (W > 1) {
            LayoutBuilder records_builder;
            d.records       = plan_gdn_replay_records(records_builder, {.layers          = dim(c.gdn_layers),
                                                                        .record_capacity = lanes,
                                                                        .width           = W,
                                                                        .conv_channels   = dim(c.gdn.conv_channels()),
                                                                        .qk_heads        = dim(c.gdn.key_heads),
                                                                        .value_heads     = dim(c.gdn.value_heads),
                                                                        .key_dim         = dim(c.gdn.key_head_dim),
                                                                        .value_dim       = dim(c.gdn.value_head_dim)});
            d.records_bytes = records_builder.finish(256);
            d.ple_records   = 2ULL * width * W * lanes;
            d.qsa_records   = 2ULL * di * W * lanes * c.attention_layers;
            const std::size_t cells = static_cast<std::size_t>(lanes) * W;
            std::size_t at          = 0;
            const auto take         = [&](std::size_t words) {
                const std::size_t begin = at;
                at += (words + 63) / 64 * 64;
                return begin;
            };
            d.spec.target   = take(cells);
            d.spec.drafts   = take(cells);
            d.spec.extents  = take(lanes);
            d.spec.lengths  = take(lanes);
            d.spec.anchors  = take(lanes);
            d.spec.licensed = take(cells);
            d.spec.counts   = take(lanes);
            d.spec.accepted = take(lanes);
            d.spec.commit   = take(lanes);
            d.spec.words    = at;
        }

        // The MTP drafter (design 11.2): residual columns, the saved and chain columns, its QSA
        // records, the HC ones and its io.
        if (d.mtp) {
            d.mtp_columns      = std::max(lanes * W, std::min(d.chunk, 512));
            d.mtp_column_bytes = 2ULL * width;
            d.mtp_ones         = sizeof(float) * c.hc.streams * static_cast<std::size_t>(d.mtp_columns);
            const auto k       = static_cast<std::size_t>(d.mtp_k);
            d.mtp_io.ids       = 0;
            d.mtp_io.cells     = d.mtp_io.ids + lanes;
            d.mtp_io.drafts    = d.mtp_io.cells + lanes * k;
            d.mtp_io.up_ids    = d.mtp_io.drafts + lanes * k;
            d.mtp_io.gather    = d.mtp_io.up_ids + static_cast<std::size_t>(lanes) * W;
            d.mtp_io.up_cells  = d.mtp_io.gather + lanes;
            d.mtp_io.words     = d.mtp_io.up_cells + static_cast<std::size_t>(lanes) * W;
        }

        // Staging slots for each layer call's misses (design 8.6), one expert record each.
        const std::uint64_t shape[] = {c.moe.experts, c.hidden_size, c.moe.intermediate};
        d.record_stride = weight_geometry(QType::NVFP4_MUL, QuantLayout::ExpertRg16, shape).record_stride;
        d.staging       = static_cast<std::size_t>(kStagingSlots) * d.record_stride;

        auto& b     = d.bytes;
        b.kv_pages  = d.kv_pages;
        b.kv        = d.kv;
        b.workspace = d.work;
        b.staging   = d.staging;
        b.state     = d.gdn_bytes + d.ple + d.tails + d.records_bytes + d.ple_records + d.qsa_records +
                  (d.mtp ? d.mtp_column_bytes * lanes * (W + 2) + 2ULL * di * W * lanes : 0);
        b.io = d.io.bytes + d.logits32 + d.logits16 + 4ULL * lanes + 4ULL * lanes +
               sizeof(ops::SamplingConfig) * lanes + d.token_counts + 4ULL * d.spec.words + d.mtp_ones +
               4ULL * d.mtp_io.words;
        b.residency            = ExpertResidency::table_bytes(c, d.columns);
        b.expert_record_stride = d.record_stride;
        b.max_frames           = ExpertResidency::max_frames(c);
        b.graph_bound          = graph_bound(o.max_concurrency, static_cast<std::uint32_t>(W),
                                             static_cast<std::uint32_t>(d.mtp_k));
        return d;
    }

    // The workspace arena of a Program with `lanes` lanes and the plan's widths: the forward at its
    // widest call (a prefill chunk or every lane's widest round), sampling, acceptance and the
    // drafter. The plan sizes the arena with the Program's lane count; the per-lane report compares
    // lane counts.
    static std::size_t workspace_bytes(const TextConfig& c, const ProgramOptions& o, const DeviceLayout& d,
                                       std::int32_t lanes) {
        const std::int32_t W = d.max_width;
        std::size_t bytes =
            execution::Forward::workspace_bytes(c, std::max(d.chunk, lanes * W), dim(o.max_context)) +
            ops::sampling_workspace_capacity_bytes(d.token_domain, 1, lanes);
        if (W > 1) {
            bytes += ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(d.token_domain, 1, W - 1, 1, lanes);
        }
        if (d.mtp) {
            bytes += execution::Forward::mtp_workspace_bytes(c, std::max(lanes * W, std::min(d.chunk, 512)), lanes,
                                                             d.head_rows, dim(o.max_context));
        }
        return bytes;
    }

    static void allocate(DeviceBuffer& buffer, std::size_t bytes, std::uint64_t& tally) {
        buffer = DeviceBuffer(bytes);
        tally += bytes;
    }

    // The startup failure for device allocations the driver placed in system memory.
    static std::string spill_message(std::uint64_t bytes) {
        return std::to_string(bytes >> 20) +
               " MiB of NInfer's VRAM allocations were placed in shared system memory by the driver's sysmem "
               "fallback: free VRAM, or lower --max-context, --max-concurrency or --prefill-chunk. Setting the NVIDIA "
               "Control Panel 'CUDA - Sysmem Fallback Policy' to 'Prefer No Sysmem Fallback' for this program makes "
               "this fail at allocation instead.";
    }

    ProgramImpl(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options)
        : parameters_(parameters), device_(device), c_(parameters.model.config().text), options_(std::move(options)),
          volume_(options_.ngram_volume, c_.ple.table), hash_(c_.ple.ngram, 0),
          plan_(plan_device(options_, parameters.model.config(), dim(parameters.model.resources().public_token_count))) {
        const auto& head = parameters_.draft_head;
        mtp_             = parameters_.mtp.has_value();
        if (mtp_ != plan_.mtp || (mtp_ && (head.rows ? dim(head.rows->weight.n) : dim(c_.vocab_size)) != plan_.head_rows)) {
            throw std::logic_error("Qwen4Exp: the bound drafter differs from its configuration");
        }
        const std::int32_t lanes = plan_.lanes;
        vocab_         = dim(c_.vocab_size);
        token_domain_  = plan_.token_domain;
        chunk_         = plan_.chunk;
        mtp_k_         = plan_.mtp_k;
        max_width_     = plan_.max_width;
        columns_       = plan_.columns;
        pages_per_row_ = plan_.pages_per_row;
        kv_pages_      = plan_.kv_pages;
        width_         = dim(c_.residual_width());
        span_          = dim(c_.ple.conv_span());
        di_            = dim(c_.qsa.index_head_dim);
        r_             = dim(c_.qsa.compress_ratio);

        // Every fixed allocation below is checked against the plan and against device free memory
        // (an allocation the driver placed in system memory fails startup, design §19.3.7).
        vram_ = open_vram_budget_source(device_.device);
        SpillGuard guard(*vram_);
        guard.begin();
        std::uint64_t allocated = 0;

        allocate(state_backing_, plan_.gdn_bytes, allocated);
        state_backing_.fill(0);
        gdn_ = std::make_unique<LinearAttentionStatePool>(DeviceSpan{state_backing_.p, state_backing_.bytes}, plan_.gdn);
        allocate(ple_backing_, plan_.ple, allocated);
        ple_backing_.fill(0);
        allocate(tails_backing_, plan_.tails, allocated);
        tails_backing_.fill(0);

        allocate(kv_backing_, plan_.kv, allocated);
        kv_backing_.fill(0);
        pool_   = std::make_unique<DeviceKVPagePool>(DeviceSpan{kv_backing_.p, kv_backing_.bytes}, plan_.pool);
        tables_ = std::make_unique<KVExecutionTablePool>(DeviceSpan{kv_backing_.p, kv_backing_.bytes}, plan_.tables, *pool_);

        io_layout_ = plan_.io;
        allocate(io_device_, io_layout_.bytes, allocated);
        io_host_ = PinnedHostBuffer(io_layout_.bytes);
        allocate(logits32_, plan_.logits32, allocated);
        allocate(logits16_, plan_.logits16, allocated);
        allocate(sampled_, 4ULL * lanes, allocated);
        allocate(sample_pos_, 4ULL * lanes, allocated);
        allocate(configs_, sizeof(ops::SamplingConfig) * lanes, allocated);
        allocate(token_counts_, plan_.token_counts, allocated);
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
        const auto layout   = paged_kv_storage_layout(options_.kv_cache, dim(c_.attention.head_dim));
        const auto kv_heads = dim(c_.attention.kv_heads);
        execution::ForwardKV kv;
        kv.block_tables  = tables_->matrix();
        std::size_t plane = 0;
        for (std::uint32_t l = 0; l < plan_.kv_layers; ++l) {
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
            allocate_mtp(lanes, allocated);
            state.mtp_ones = Tensor(mtp_ones_.p, DType::FP32, {dim(c_.hc.streams), mtp_columns_});
        }
        execution::ForwardExperts experts;
        experts.frames.assign(c_.num_hidden_layers, nullptr);
        if (max_width_ > 1) { allocate_verification(lanes, allocated); }
        work_capacity_ = plan_.work;
        work_          = std::make_unique<WorkspaceArena>(work_capacity_);
        allocated += work_capacity_;

        std::vector<const std::uint8_t*> banks;
        std::uint64_t record_stride = 0;
        for (const auto& layer : parameters_.layers) {
            banks.push_back(reinterpret_cast<const std::uint8_t*>(layer.moe.bank->planes.records));
            record_stride = layer.moe.bank->planes.record_stride;
        }
        if (record_stride != plan_.record_stride) {
            throw std::logic_error("Qwen4Exp: the expert banks' record stride differs from their format's");
        }
        // Staging slots for each layer call's misses (copied with a compact read window, design
        // 8.6), allocated before the frames take the remaining memory.
        allocate(staging_, plan_.staging, allocated);
        experts.staging_base  = static_cast<std::uint8_t*>(staging_.p);
        experts.staging_slots = kStagingSlots;
        // The expert cache's tables and route log; its frames come last (below).
        residency_ = std::make_unique<ExpertResidency>(c_, std::move(banks), record_stride, columns_, device_.device);
        allocated += plan_.bytes.residency;
        if (allocated != plan_.bytes.fixed_bytes()) {
            throw std::logic_error("Qwen4Exp: the Program's device allocations differ from its device plan");
        }
        if (const std::uint64_t spilled = guard.end(allocated); spilled != 0) {
            throw std::runtime_error(spill_message(spilled));
        }
        CUDA_CHECK(cudaStreamCreateWithFlags(&overlap_.stream, cudaStreamNonBlocking));
        for (auto& event : overlap_.events) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
        experts.overlap_stream = overlap_.stream;
        experts.overlap_events = overlap_.events;
        // CPU-served misses for decode and verify calls, and for every call of at most
        // kMaxCpuColumns columns (short prompts, chunk remainders, forced tokens), whose experts are
        // all thin and so give the same bits on either device (design 16.2); wider prefill chunks
        // stay on the GPU.
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
                    .max_columns = std::max(ops::offloaded_moe::kMaxCpuColumns, lanes * max_width_),
                    .pcie_divisor = options_.cpu_pcie_divisor,
                    .cpus        = {}});
            for (std::uint32_t l = 0; l < c_.num_hidden_layers; ++l) {
                experts.cpu.push_back(cpu_service_->channel(static_cast<int>(l)));
            }
        }

        // The VRAM expert cache takes what the fixed allocations, the reserve for graph
        // executables and the display headroom leave (design §19.3.7). Its frames are the one
        // elastic allocation: a chunk the driver places in system memory (another program took
        // memory since the reading) is given back and the cache stops growing there.
        if (options_.expert_cache) {
            VramDemand demand;
            demand.headroom    = options_.vram_headroom;
            demand.elastic     = residency_->elastic() && options_.vram_monitor;
            demand.graphs      = plan_.bytes.graph_bound;
            demand.frame_bytes = record_stride;
            demand.max_frames  = residency_->frame_limit();
            sizing_            = size_expert_frames(vram_->query(), demand);
            const auto grown   = residency_->resize(sizing_.frames, device_.stream, *vram_);
            if (grown.spilled != 0) {
                diagnostic(std::to_string(grown.spilled >> 20) + " MiB of the expert frames were placed in system memory "
                       "and given back; the expert cache starts with " + std::to_string(grown.frames) + " of " +
                       std::to_string(sizing_.frames) + " frames", DiagnosticLevel::Warning);
            } else if (grown.refused) {
                diagnostic("the GPU refused expert-cache memory; the cache starts with " +
                                                     std::to_string(grown.frames) + " of " +
                                                     std::to_string(sizing_.frames) + " frames", DiagnosticLevel::Warning);
            }
            sizing_.frames = grown.frames;
            start_vram_monitor(demand);
        }
        experts.frame_base   = residency_->frame_base();
        experts.frame_stride = residency_->frame_stride();
        experts.route_log    = residency_->route_log();
        experts.route_stride = residency_->route_stride();
        for (std::uint32_t l = 0; l < c_.num_hidden_layers; ++l) { experts.frames[l] = residency_->table(l); }
        if (!options_.route_trace.empty()) {
            trace_ = std::make_unique<RouteTrace>(
                options_.route_trace,
                RouteTraceSetup{.layers             = c_.num_hidden_layers,
                                .experts            = c_.moe.experts,
                                .top_k              = c_.moe.top_k,
                                .frames             = residency_->frames(),
                                .max_columns        = static_cast<std::uint32_t>(columns_),
                                .lanes              = options_.max_concurrency,
                                .max_width          = static_cast<std::uint32_t>(max_width_),
                                .mtp_draft_tokens   = static_cast<std::uint32_t>(mtp_k_),
                                .ngram_draft_tokens = options_.ngram_draft_tokens,
                                .prefill_chunk      = static_cast<std::uint32_t>(chunk_)});
        }
        forward_ = std::make_unique<execution::Forward>(parameters_, device_, *work_, std::move(state), std::move(kv),
                                                        std::move(experts), dim(options_.max_context));
        device_.synchronize();
        report_lane_cost(lanes, record_stride);
    }

    const DeviceLayout& device_layout() const noexcept { return plan_; }
    const VramSizing& vram_sizing() const noexcept { return sizing_; }

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
        lane.mtp_cells   = 0;
        lane.mtp_live    = mtp_;
        lane.mtp_accept.fill(kAcceptancePrior);
        lane.mtp_policy_rounds = 0;
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
        lane.ngram                   = {};
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
            trace_round(RouteTraceKind::PrefillChunk, 1, width, static_cast<std::uint32_t>(width),
                        kPrefillPromotionsPerLayer, static_cast<std::uint32_t>(begin));
            residency_->after_round(device_.stream, width, kPrefillPromotionsPerLayer);
            apply_vram_target(false);
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
            trace_round(RouteTraceKind::PrefillChunk, 1, width, static_cast<std::uint32_t>(width),
                        kPrefillPromotionsPerLayer, static_cast<std::uint32_t>(begin));
            residency_->after_round(device_.stream, width, kPrefillPromotionsPerLayer);
            apply_vram_target(false);
            const SequenceHandle rows[] = {sequence};
            out.timing.submit_host_ns = elapsed_ns(start);
            lane.prefill_ns += out.timing.submit_host_ns;
            out.pending.emplace(ContractAccess::make_pending(this, ++next_transaction_, rows,
                                                             {pending_tokens_.data(), 1}, out.timing));
            plain_round_ = false;
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
        // fewer than it may still emit, so every verified position lies in its reservation. Each
        // row's MTP draft length maximizes its expected tokens per round time; a longer n-gram copy
        // proposal replaces the drafts. Only one-row rounds speculate (design 11.3, 19.3.5 S1): at
        // B >= 2 every row decodes plain, which shares the dense reads without padding rows to the
        // longest draft and keeps the round's B <= 8 columns on the column-invariant dense routes,
        // so greedy output at C > 1 is meant to equal C = 1. The plain round's W = 1 catch-up keeps
        // each drafter current, and a row's draft-length policy advances only in its one-row rounds.
        const bool speculate = batch == 1;
        std::array<std::int32_t, kMaximumConcurrency> wanted{};
        if (mtp_) {
            std::int32_t steps = 0;
            for (std::int32_t b = 0; b < batch; ++b) {
                Lane& lane = lanes_[lanes[b]];
                wanted[b]  = speculate && lane.mtp_live ? choose_draft_length(lane) : 0;
                steps      = std::max(steps, wanted[b]);
            }
            mtp_draft(std::span<const std::uint32_t>(lanes.data(), batch), steps);
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
            if (mtp_ && lane.mtp_live && wanted[b] > 0) {
                const auto n = std::min<std::uint32_t>(limit, static_cast<std::uint32_t>(wanted[b]));
                drafts_[b].assign(mtp_drafts_[b].begin(), mtp_drafts_[b].begin() + n);
            }
            if (speculate && lane.proposer) {
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
        for (std::int32_t b = 0; b < batch; ++b) { ++lanes_[lanes[b]].speculative.fallback_steps; }
        round_width_ = 1;
        plain_round_ = true;
        stage_decode(std::span<const std::uint32_t>(lanes.data(), batch), std::span<const std::int32_t>(positions.data(), batch));
        run_decode(batch);
        for (std::int32_t b = 0; b < batch; ++b) { ++lanes_[lanes[b]].state_tokens; }
        sample(std::span<const std::uint32_t>(lanes.data(), batch), std::span<const std::int32_t>(positions.data(), batch));
        deferred_ = {.columns = batch, .tokens = 1, .live = false, .pending = true};
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
            trace_round(RouteTraceKind::ForcedTokens, 1, static_cast<std::int32_t>(stride), stride,
                        kDecodePromotionsPerLayer, static_cast<std::uint32_t>(begin));
            residency_->after_round(device_.stream, static_cast<std::int32_t>(stride), kDecodePromotionsPerLayer);
            apply_vram_target(false);
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
        // After a plain decode round the drafter writes the round's cell (its pending cell moves
        // one position); after the prefill's first token the pending cell stays the prompt's last.
        const bool catch_up = mtp_ && plain_round_;
        std::int32_t* commit = catch_up ? spec_host(spec_layout_.commit) : nullptr;
        bool any_cell        = false;
        for (std::size_t row = 0; row < rows.size(); ++row) {
            const auto index = lane_of(rows[row]);
            Lane& lane       = lanes_[index];
            const auto& d    = decisions[row];
            if (commit != nullptr) { commit[row] = 0; }
            if (d.cancelled) {
                if (catch_up && lane.mtp_live) {
                    // MR2: decode() advanced the state past the round's position at submission and
                    // a cancelled row skips the catch-up, so that position's cell becomes the
                    // pending one: its residual, which the round exported, becomes the saved one.
                    CUDA_CHECK(cudaMemcpyAsync(saved_column(index).data,
                                               static_cast<const std::byte*>(mtp_residuals_.p) + 2ULL * width_ * row,
                                               2ULL * width_, cudaMemcpyDeviceToDevice, device_.stream));
                    lane.mtp_cells = lane.state_tokens - 1;
                }
                out.rows[row].timings     = timings(lane);
                out.rows[row].speculative = lane.speculative;
                out.rows[row].disposition = runtime::CommitDisposition::CancelledReleased;
                continue;
            }
            if (d.accepted_tokens != 1) { throw std::logic_error("Qwen4Exp: a row commits exactly one token"); }
            lane.history.push_back(pending_tokens_[row]);
            if (commit != nullptr && lane.mtp_live) {
                commit[row] = 1;
                any_cell    = true;
            }
            if (lane.proposer) { lane.proposer->append(pending_tokens_[row]); }
            if (lane.phase == Phase::Prefill) { lane.phase = Phase::Decode; }
            if (d.terminal) {
                lane.phase                = Phase::Finishable;
                out.rows[row].disposition = runtime::CommitDisposition::Finishable;
            } else {
                out.rows[row].disposition = runtime::CommitDisposition::Active;
            }
        }
        if (any_cell) {
            upload_pinned(spec_device(spec_layout_.commit), commit, 4ULL * rows.size(), device_.stream);
            mtp_catch_up(rows, 1, commit);
        }
        for (const auto& row : rows) { require_mtp_settled(lanes_[ContractAccess::lane(row)]); }
        settle_round();
        for (std::size_t row = 0; row < rows.size(); ++row) {
            if (decisions[row].cancelled) { release(ContractAccess::lane(rows[row])); }
        }
        plain_round_ = false;
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
        deferred_.pending    = false;
        return out;
    }

    FinishResult finish(SequenceHandle sequence) noexcept {
        FinishResult out;
        try {
            const auto index = lane_of(sequence);
            if (lanes_[index].phase != Phase::Finishable) { return out; }
            if (!mtp_settled(lanes_[index])) {
                diagnostic("Qwen4Exp: a finished request's MTP cells are out of step with its state", DiagnosticLevel::Warning);
            }
            out.timings     = timings(lanes_[index]);
            out.speculative = lanes_[index].speculative;
            report_cache(lanes_[index]);
            report_ngram(lanes_[index]);
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

    // A decode or verification round's cache update (routes, LFRU, promotions) runs on the host
    // after its commit has enqueued its GPU work (the GDN fold, tail commits, the drafter's
    // catch-up), so the policy overlaps that work instead of idling the GPU after the round.
    struct DeferredRound {
        std::int32_t columns  = 0;
        std::uint32_t tokens  = 0;
        bool live             = false; // only the verification's accepted columns count
        bool pending          = false;
    };
    DeferredRound deferred_;

    void settle_round() {
        if (!deferred_.pending) { return; }
        deferred_.pending = false;
        const std::size_t budget = decode_budget(deferred_.tokens);
        if (trace_) {
            const std::int32_t width = deferred_.live ? round_width_ : 1;
            trace_round(deferred_.live ? RouteTraceKind::Verify : RouteTraceKind::Decode, deferred_.columns / width,
                        width, deferred_.tokens, budget, RouteTrace::kUnknownPosition,
                        deferred_.live ? std::span<const std::uint8_t>(live_.data(),
                                                                       static_cast<std::size_t>(deferred_.columns))
                                       : std::span<const std::uint8_t>{});
        }
        residency_->after_round(device_.stream, deferred_.columns, budget,
                                deferred_.live ? std::span<const std::uint8_t>(live_.data(),
                                                                               static_cast<std::size_t>(deferred_.columns))
                                               : std::span<const std::uint8_t>{});
        apply_vram_target(false);
    }

    // ---------------------------------------------------------------- VRAM monitor (design §19.3.7)
    void start_vram_monitor(const VramDemand& demand) {
        if (!options_.vram_monitor) { return; }
        const VramSnapshot now = vram_->query();
        if (residency_->elastic()) {
            control_.emplace(demand, ExpertResidency::kChunkBytes, now.local_usage, residency_->pool_bytes(),
                             options_.vram_grow_delay_seconds);
        }
        frames_now_     = residency_->frames();
        pool_now_       = residency_->pool_bytes();
        startup_frames_ = frames_now_;
        vram_epoch_     = Clock::now();
        monitor_ = std::make_unique<VramMonitor>(device_.device, *vram_,
                                                 [this](const VramSnapshot& snapshot) { on_vram_snapshot(snapshot); });
    }

    double vram_seconds() const {
        return std::chrono::duration<double>(Clock::now() - vram_epoch_).count();
    }

    // The monitor thread: wakes an idle engine when the cache should resize; without an elastic
    // pool, warns once per pressure episode that nothing absorbs it.
    void on_vram_snapshot(const VramSnapshot& snapshot) {
        const std::lock_guard<std::mutex> lock(vram_mutex_);
        if (control_) {
            const auto d = control_->decide(snapshot, pool_now_, frames_now_, vram_seconds(), true);
            if ((d.shrink || d.grow) && waker_) { waker_(); }
            return;
        }
        const std::uint64_t headroom = display_headroom(snapshot.display, false, options_.vram_headroom);
        const bool pressure = snapshot.device_free < headroom / 2 ||
                              (snapshot.has_budget && snapshot.local_usage > snapshot.local_budget);
        if (pressure && !pressure_warned_) {
            const std::uint64_t short_by = headroom > snapshot.device_free ? headroom - snapshot.device_free : 0;
            diagnostic("free VRAM is " + std::to_string(snapshot.device_free >> 20) + " MiB, below the " +
                       std::to_string(headroom >> 20) + " MiB headroom by " + std::to_string(short_by >> 20) +
                       " MiB; the expert cache cannot shrink on this system, so NInfer memory may move to system "
                       "memory and slow down. Raise --vram-headroom-mib to " +
                       std::to_string((headroom + short_by + (256ULL << 20)) >> 20), DiagnosticLevel::Warning);
        }
        pressure_warned_ = pressure;
    }

public:
    void set_maintenance_waker(std::function<void()> waker) {
        const std::lock_guard<std::mutex> lock(vram_mutex_);
        waker_ = std::move(waker);
    }

    // At a boundary (no round's kernels in flight; `idle`: no request active): resizes the expert
    // cache to the control law's decision.
    void apply_vram_target(bool idle) {
        if (!control_) { return; }
        const std::lock_guard<std::mutex> lock(vram_mutex_);
        const auto d = control_->decide(monitor_->latest(), pool_now_, frames_now_, vram_seconds(), idle);
        if (d.frames == frames_now_) { return; }
        const std::uint32_t before = frames_now_;
        const auto resized         = residency_->resize(d.frames, device_.stream, *vram_);
        frames_now_                = resized.frames;
        pool_now_                  = residency_->pool_bytes();
        const VramSnapshot after   = monitor_->refresh();
        diagnostic(std::string(d.shrink ? "free VRAM fell to " : "free VRAM rose to ") +
                   std::to_string(after.device_free >> 20) + " MiB: expert cache " + std::to_string(before) +
                   " -> " + std::to_string(frames_now_) + " frames" +
                   (resized.spilled != 0 ? " (a chunk placed in system memory was given back)" : ""), DiagnosticLevel::Info);
        if (d.shrink && frames_now_ * 4 < startup_frames_ && !low_frames_warned_) {
            diagnostic("another program holds VRAM: the expert cache is down to " + std::to_string(frames_now_) + " of " +
                       std::to_string(startup_frames_) +
                       " frames, so decode is slower until that memory is released", DiagnosticLevel::Warning);
            low_frames_warned_ = true;
        }
        if (frames_now_ * 4 >= startup_frames_) { low_frames_warned_ = false; }
    }

private:
    static constexpr std::size_t kPrefillPromotionsPerLayer = 16;

    // Appends the round whose routes after_round is about to apply to the internal route trace
    // (ProgramOptions::route_trace); rows' lanes are host_lanes_ as staged for the round.
    void trace_round(RouteTraceKind kind, std::int32_t rows, std::int32_t width, std::uint32_t tokens,
                     std::size_t budget, std::uint32_t position, std::span<const std::uint8_t> live = {}) {
        if (!trace_) { return; }
        trace_->append({.kind     = kind,
                        .rows     = static_cast<std::uint32_t>(rows),
                        .width    = static_cast<std::uint32_t>(width),
                        .tokens   = tokens,
                        .budget   = static_cast<std::uint32_t>(std::min<std::size_t>(budget, 0xFFFFFFFFU)),
                        .position = position},
                       std::span<const std::int32_t>(host_lanes_.data(), static_cast<std::size_t>(rows)), live,
                       residency_->route_host(), residency_->route_stride());
    }

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
            diagnostic(text);
        } catch (...) {}
    }

    // The workspace arena of a Program with `lanes` lanes: the forward at its widest call (a prefill
    // chunk or every lane's widest round), sampling, acceptance and the drafter.
    std::size_t workspace_capacity(std::int32_t lanes) const { return workspace_bytes(c_, options_, plan_, lanes); }

    // Reports at startup what each lane of --max-concurrency takes from the expert cache, in frames
    // (design 19.3.5): its KV extent unless --kv-capacity fixes the pool; its recurrent, convolution
    // and QSA-tail state; its verification and drafter records; its logit columns and penalty
    // counts; and its share of the workspace. Every other allocation is independent of the lane
    // count, so one more lane costs these frames.
    void report_lane_cost(std::int32_t lanes, std::uint64_t record_stride) noexcept {
        try {
            const auto n         = static_cast<std::size_t>(lanes);
            const bool fixed_kv  = options_.kv_capacity_tokens != 0;
            const std::size_t kv = fixed_kv ? 0 : kv_backing_.bytes / n;
            const std::size_t state =
                (state_backing_.bytes + ple_backing_.bytes + tails_backing_.bytes + logits32_.bytes + logits16_.bytes +
                 token_counts_.bytes + records_backing_.bytes + ple_records_.bytes + qsa_records_.bytes +
                 mtp_residuals_.bytes + mtp_saved_.bytes + mtp_chain_.bytes + mtp_records_.bytes) /
                n;
            // The workspace grows with the lane count by its sampling, acceptance and drafter rows:
            // the mean step from one lane to this count, or the second lane's at one lane.
            const std::size_t wide   = workspace_capacity(lanes > 1 ? lanes : 2);
            const std::size_t narrow = workspace_capacity(1);
            const std::size_t workspace = wide > narrow ? (wide - narrow) / (lanes > 1 ? n - 1 : 1) : 0;
            const double lane_frames = static_cast<double>(kv + state + workspace) /
                                       static_cast<double>(std::max<std::uint64_t>(record_stride, 1));
            char kv_text[64];
            if (fixed_kv) {
                std::snprintf(kv_text, sizeof(kv_text), "KV in the fixed --kv-capacity pool");
            } else {
                std::snprintf(kv_text, sizeof(kv_text), "KV %zu MiB", kv >> 20);
            }
            char text[320];
            std::snprintf(text, sizeof(text),
                          "expert cache: %u frames of %.2f MiB at %d lane%s; each lane holds %.1f frames (%s; "
                          "recurrent state, records and workspace %zu MiB)",
                          residency_->frames(), static_cast<double>(record_stride) / 1048576.0, lanes,
                          lanes == 1 ? "" : "s", lane_frames, kv_text, (state + workspace) >> 20);
            diagnostic(text);
        } catch (...) {}
    }

    // Reports the finished request's n-gram row traffic (design 12.4). Every row is read on the host
    // before the call that consumes it is launched.
    void report_ngram(const Lane& lane) noexcept {
        try {
            const auto& n = lane.ngram;
            char text[192];
            std::snprintf(text, sizeof(text),
                          "n-gram rows: %llu requested, %.1f%% host-cache hits, %llu NVMe reads, %.1f ms of reads",
                          static_cast<unsigned long long>(n.rows),
                          n.rows ? 100.0 * static_cast<double>(n.hits) / static_cast<double>(n.rows) : 0.0,
                          static_cast<unsigned long long>(n.reads), static_cast<double>(n.read_ns) * 1e-6);
            diagnostic(text);
        } catch (...) {}
    }

    // A diagnostic (Info unless stated) for the Engine's observer, or stderr without one.
    void diagnostic(const std::string& text, DiagnosticLevel level = DiagnosticLevel::Info) const noexcept {
        try {
            if (options_.diagnostics.callback) {
                options_.diagnostics.callback(Diagnostic{.level = level, .message = text});
            } else {
                std::fprintf(stderr, "[engine] %s\n", text.c_str());
            }
        } catch (...) {}
    }


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

    // The n-gram rows of `count` positions starting at `begin`, from the tokens before them; the
    // traffic counts toward `lane`'s request.
    void stage_ngram(Lane& lane, const std::vector<std::int32_t>& history, std::int32_t begin, std::int32_t count,
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
        const NgramVolume::Counters before = volume_.counters();
        volume_.read_rows(row_ids_, std::span<std::byte>(host_io() + io_layout_.ngram + column * heads * row_bytes,
                                                         static_cast<std::size_t>(count) * heads * row_bytes));
        const NgramVolume::Counters& after = volume_.counters();
        lane.ngram.rows += after.rows - before.rows;
        lane.ngram.hits += after.hits - before.hits;
        lane.ngram.reads += after.reads - before.reads;
        lane.ngram.read_ns += after.read_ns - before.read_ns;
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
        stage_ngram(lanes_[lane], history, begin, width, 0);
        host_lanes_[0] = static_cast<std::int32_t>(lane);
    }

    void stage_decode(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions) {
        auto* ids     = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.ids);
        auto* pos     = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.positions);
        auto* columns = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.columns);
        auto* slots   = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.slots);
        auto* rows    = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.rows);
        for (std::size_t b = 0; b < lanes.size(); ++b) {
            Lane& lane       = lanes_[lanes[b]];
            ids[b]           = lane.history[static_cast<std::size_t>(positions[b])];
            pos[b]           = positions[b];
            columns[b]       = static_cast<std::int32_t>(b);
            slots[b]         = static_cast<std::int32_t>(lanes[b]);
            rows[b]          = static_cast<std::int32_t>(lanes[b]);
            host_lanes_[b]   = static_cast<std::int32_t>(lanes[b]);
            stage_ngram(lane, lane.history, positions[b], 1, b);
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
        upload_pinned(io_device_.p, io_host_.data(), io_prefix(batch), s);
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
            const std::size_t free_before = trace_ ? trace_free() : 0;
            graph.definition.capture(s, body);
            graph.executable.instantiate(graph.definition);
            graph.executable.launch(s);
            if (trace_) { trace_graph(graph, free_before); }
        }
    }

    // Device free bytes, and the route trace's record of a graph executable just instantiated
    // (its family and shape follow from the slot it occupies).
    static std::size_t trace_free() {
        std::size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        return free_bytes;
    }
    void trace_graph(const DecodeGraph& graph, std::size_t free_before) {
        RouteTraceGraph family = RouteTraceGraph::Decode;
        std::ptrdiff_t at      = route_trace_index(graphs_, &graph);
        std::ptrdiff_t stride  = 1;
        if (at < 0 && (at = route_trace_index(verify_graphs_, &graph)) >= 0) {
            family = RouteTraceGraph::Verify;
            stride = max_width_;
        } else if (at < 0 && (at = route_trace_index(mtp_draft_graphs_, &graph)) >= 0) {
            family = RouteTraceGraph::MtpDraft;
            stride = mtp_k_;
        } else if (at < 0 && (at = route_trace_index(mtp_catch_graphs_, &graph)) >= 0) {
            family = RouteTraceGraph::MtpCatchUp;
            stride = max_width_;
        }
        if (at < 0) { return; }
        trace_->graph(family, static_cast<std::uint32_t>(at / stride + 1), static_cast<std::uint32_t>(at % stride + 1),
                      free_before, trace_free());
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
    void allocate_verification(std::int32_t lanes, std::uint64_t& allocated) {
        allocate(records_backing_, plan_.records_bytes, allocated);
        records_ = GdnReplayRecords(DeviceSpan{records_backing_.p, records_backing_.bytes}, plan_.records);
        allocate(ple_records_, plan_.ple_records, allocated);
        allocate(qsa_records_, plan_.qsa_records, allocated);
        folds_.resize(static_cast<std::size_t>(max_width_) + 1);
        verify_graphs_.resize(static_cast<std::size_t>(lanes) * max_width_);
        const std::size_t cells = static_cast<std::size_t>(lanes) * max_width_;
        spec_layout_ = plan_.spec;
        allocate(spec_device_, 4ULL * spec_layout_.words, allocated);
        spec_host_ = PinnedHostBuffer(4ULL * spec_layout_.words);
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
            Lane& lane                = lanes_[lanes[b]];
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
            stage_ngram(lane, sequence_, positions[b], W, static_cast<std::size_t>(b * W));
            spec_host(spec_layout_.extents)[b] = n;
            spec_host(spec_layout_.lengths)[b] = positions[b];
            spec_host(spec_layout_.anchors)[b] = anchor;
        }
        upload_pinned(io_device_.p, io_host_.data(), io_prefix(batch * W), s);
        upload_pinned(spec_device_.p, spec_host_.data(), 4ULL * spec_layout_.licensed, s);
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
        upload_pinned(configs_.p, configs, sizeof(ops::SamplingConfig) * batch, s);
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
            if (!from_ngram_[b]) {
                // Positions 1..A were accepted; position A + 1 (when drafted) was the first rejected.
                for (std::uint64_t j = 0; j < n && j <= static_cast<std::uint64_t>(A); ++j) {
                    double& a = lane.mtp_accept[static_cast<std::size_t>(j)];
                    a = (1.0 - kAcceptanceWeight) * a + kAcceptanceWeight * (j < static_cast<std::uint64_t>(A) ? 1.0 : 0.0);
                }
            }
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
        deferred_ = {.columns = batch * W, .tokens = advanced, .live = true, .pending = true};
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
        upload_pinned(spec_device(spec_layout_.commit), commit, 4ULL * batch, s);
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
        for (const auto& row : rows) { require_mtp_settled(lanes_[ContractAccess::lane(row)]); }
        settle_round();
        for (std::int32_t row = 0; row < batch; ++row) {
            if (decisions[static_cast<std::size_t>(row)].cancelled) { release(ContractAccess::lane(rows[row])); }
        }
        ContractAccess::consume(pending);
        pending_transaction_ = 0;
        round_width_         = 1;
        return out;
    }

    // ---------------------------------------------------------------- MTP drafter (design 11.2)
    void allocate_mtp(std::int32_t lanes, std::uint64_t& allocated) {
        const std::size_t column = plan_.mtp_column_bytes;
        mtp_columns_ = plan_.mtp_columns;
        allocate(mtp_residuals_, column * lanes * max_width_, allocated);
        allocate(mtp_saved_, column * lanes, allocated);
        allocate(mtp_chain_, column * lanes, allocated);
        allocate(mtp_records_, 2ULL * di_ * max_width_ * lanes, allocated);
        mtp_saved_.fill(0);
        std::vector<float> ones(static_cast<std::size_t>(c_.hc.streams) * mtp_columns_, 1.0F);
        allocate(mtp_ones_, plan_.mtp_ones, allocated);
        mtp_ones_.copy_from_host(ones.data(), ones.size() * sizeof(float));
        mtp_io_ = plan_.mtp_io;
        allocate(mtp_device_, 4ULL * mtp_io_.words, allocated);
        mtp_host_ = PinnedHostBuffer(4ULL * mtp_io_.words);
        for (auto& d : mtp_drafts_) { d.assign(static_cast<std::size_t>(mtp_k_), 0); }
        mtp_catch_graphs_.resize(static_cast<std::size_t>(lanes) * max_width_);
        mtp_draft_graphs_.resize(static_cast<std::size_t>(lanes) * mtp_k_);
    }

    std::int32_t* mtp_host(std::size_t offset) const { return static_cast<std::int32_t*>(mtp_host_.data()) + offset; }
    std::int32_t* mtp_device(std::size_t offset) const { return static_cast<std::int32_t*>(mtp_device_.p) + offset; }
    Tensor saved_column(std::uint32_t lane) const {
        return Tensor(static_cast<std::byte*>(mtp_saved_.p) + 2ULL * width_ * lane, DType::BF16, {width_});
    }

    // MR1's settled invariant (see Lane): the pending cell is the last processed position's.
    bool mtp_settled(const Lane& lane) const noexcept {
        return !mtp_ || !lane.mtp_live || lane.mtp_cells == lane.state_tokens ||
               lane.mtp_cells + 1U == lane.state_tokens;
    }
    // A drafter whose cells fall out of step with the model state would only propose worse drafts,
    // which verification rejects: output is unaffected. The request keeps decoding without drafts
    // and the defect is reported, rather than failing the request.
    void require_mtp_settled(Lane& lane) const noexcept {
        if (mtp_settled(lane)) { return; }
        lane.mtp_live = false;
        diagnostic("Qwen4Exp: the MTP drafter's cells are out of step with the model state; drafting is "
                 "off for the rest of this request", DiagnosticLevel::Warning);
    }

    // The MTP cells of a chunk (prefill or forced tokens) of one lane. The pending cell (the
    // position before the chunk) is prepended unless written; the chunk's last cell is included
    // only when the token after it is known (forced tokens). Stages ids and cells into the io.
    std::optional<execution::MtpChunk> stage_mtp_chunk(Lane& lane, std::uint32_t index, std::int32_t begin,
                                                       std::int32_t width) {
        if (!mtp_) { return std::nullopt; }
        const bool prepend = begin > 0 && lane.mtp_cells < static_cast<std::uint32_t>(begin);
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
        lane.mtp_cells = static_cast<std::uint32_t>(begin + columns);
        lane.mtp_live  = true;
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
    // Draft-length policy (design 11.3): the K in [0, mtp_k_] that maximizes expected tokens per
    // round time. Expected tokens are 1 + the sum of the products of the lane's conditional
    // position acceptances (learned online); a round of K drafts costs 1 + kWidthCost * K plain
    // rounds, the slope measured on the 5090 (design 19.2), so the plain round time cancels. Every
    // kProbeInterval rounds a row drafts one token more than its choice, so the estimate of the
    // next position stays current (K = 0 then probes one draft).
    std::int32_t choose_draft_length(Lane& lane) const {
        std::int32_t best = 0;
        double best_score = 1.0;
        double reach = 1.0, tokens = 1.0;
        for (std::int32_t k = 1; k <= mtp_k_; ++k) {
            reach *= lane.mtp_accept[static_cast<std::size_t>(k - 1)];
            tokens += reach;
            const double score = tokens / (1.0 + kWidthCost * k);
            if (score > best_score) {
                best_score = score;
                best       = k;
            }
        }
        if (++lane.mtp_policy_rounds % kProbeInterval == 0) { best = std::min(best + 1, mtp_k_); }
        return best;
    }

    void mtp_draft(std::span<const std::uint32_t> lanes, std::int32_t steps) {
        const cudaStream_t s = device_.stream;
        const auto batch     = static_cast<std::int32_t>(lanes.size());
        const std::size_t column = 2ULL * width_;
        bool any = false, unwritten = false;
        for (std::int32_t b = 0; b < batch; ++b) {
            any |= lanes_[lanes[b]].mtp_live;
            unwritten |= lanes_[lanes[b]].mtp_live && lanes_[lanes[b]].mtp_cells < lanes_[lanes[b]].state_tokens;
        }
        // Without drafts (a gated or zero-length round) only unwritten pending cells need the drafter.
        if (!any || (steps == 0 && !unwritten)) { return; }
        auto* slots = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.slots);
        auto* rows  = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.rows);
        for (std::int32_t b = 0; b < batch; ++b) {
            const Lane& lane = lanes_[lanes[b]];
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
        auto* io = static_cast<std::byte*>(io_device_.p);
        upload_pinned(io + io_layout_.slots, slots, 4ULL * batch, s);
        upload_pinned(io + io_layout_.rows, rows, 4ULL * batch, s);
        upload_pinned(mtp_device_.p, mtp_host_.data(), 4ULL * mtp_io_.drafts, s);
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
        if (steps == 0) {
            for (std::int32_t b = 0; b < batch; ++b) {
                Lane& lane = lanes_[lanes[b]];
                if (lane.mtp_live) { lane.mtp_cells = lane.state_tokens; }
            }
            return;
        }
        replay(mtp_draft_graphs_[static_cast<std::size_t>(batch - 1) * mtp_k_ + (steps - 1)], [&] {
        for (std::int32_t j = 0; j < steps; ++j) {
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
        CUDA_CHECK(cudaMemcpyAsync(mtp_host(mtp_io_.drafts), mtp_device(mtp_io_.drafts), 4ULL * steps * batch,
                                   cudaMemcpyDeviceToHost, s));
        device_.synchronize();
        for (std::int32_t b = 0; b < batch; ++b) {
            Lane& lane = lanes_[lanes[b]];
            if (unwritten && lane.mtp_live) { lane.mtp_cells = lane.state_tokens; }
            for (std::int32_t j = 0; j < steps; ++j) {
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
        upload_pinned(mtp_device(mtp_io_.up_ids), mtp_host(mtp_io_.up_ids), 4ULL * batch * W, s);
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
            lanes_[lane].mtp_cells = lanes_[lane].state_tokens;
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
        upload_pinned(configs_.p, configs, sizeof(ops::SamplingConfig) * batch, s);
        upload_pinned(sample_pos_.p, sample_positions, 4ULL * batch, s);
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
    DeviceLayout plan_;
    std::unique_ptr<VramBudgetSource> vram_;
    VramSizing sizing_;
    std::mutex vram_mutex_; // the control law, the waker and the cache size the monitor reads
    std::optional<VramControl> control_;
    std::function<void()> waker_;
    std::uint32_t frames_now_ = 0, startup_frames_ = 0;
    std::uint64_t pool_now_   = 0;
    bool low_frames_warned_   = false;
    bool pressure_warned_     = false;
    Clock::time_point vram_epoch_{};

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
    std::unique_ptr<RouteTrace> trace_;
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
    std::vector<DecodeGraph> mtp_draft_graphs_; // (batch - 1) * mtp_k_ + steps - 1
    std::vector<DecodeGraph> mtp_catch_graphs_; // (batch - 1) * max_width_ + width - 1
    std::vector<DecodeGraph> verify_graphs_; // (batch - 1) * max_width_ + width - 1

    std::array<Lane, kMaximumConcurrency> lanes_{};
    std::array<std::int32_t, kMaximumConcurrency> host_lanes_{};
    std::vector<TokenId> pending_tokens_ = std::vector<TokenId>(kMaximumConcurrency, 0);
    std::array<std::int32_t, kMaximumConcurrency> pending_counts_{};

    // MTP drafter.
    bool mtp_            = false;
    std::int32_t mtp_k_  = 0;
    std::int32_t mtp_columns_ = 0;
    DeviceBuffer mtp_residuals_, mtp_saved_, mtp_chain_, mtp_records_, mtp_ones_, mtp_device_;
    PinnedHostBuffer mtp_host_{1};
    MtpIo mtp_io_;
    std::array<std::vector<std::int32_t>, kMaximumConcurrency> mtp_drafts_;
    std::array<bool, kMaximumConcurrency> from_ngram_{};
    // Draft-length policy: acceptance prior and EWMA weight, and the cost of each draft column in
    // plain rounds (warm tg512 on the 5090: W = 3, 4, 5 rounds took 1.84, 2.13 and 2.54 plain
    // rounds, design 19.2).
    static constexpr double kAcceptancePrior      = 0.75;
    static constexpr double kAcceptanceWeight     = 0.1;
    static constexpr double kWidthCost            = 0.38;
    static constexpr std::uint64_t kProbeInterval = 8;
    bool plain_round_ = false;

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
    // Last: its thread reads the source, the control law and the residency, so it stops first.
    std::unique_ptr<VramMonitor> monitor_;
};

} // namespace detail
} // namespace ninfer::models::qwen4_exp
