#pragma once

// Inkling 2's Program (design docs/maintainer/inkling2_mini-design.md §7-§8): the lanes, their state and KV, the
// calls of the Forward (execution/forward.h), sampling and the commit of each round, behind the common EngineCore.
//
// First slice (.local tracker "F3 Program, first slice"): every lane's KV is reserved to the context limit at startup,
// so a binding never waits for pages and no request is paused; the prefix cache, speculation, the n-gram archive,
// next-turn prefill and constrained output are off for this architecture (normalize_inkling2_options refuses them).
// Prompts prefill in chunks of up to 128 columns of one lane through the decode and verify kernels' arithmetic (a wide
// call runs the same per-column bits; the wide-route milestone moves each Op's pass loop inside its kernel), decode
// rounds are B lanes x 1 column as CUDA graphs per (B, T) captured at startup.

#include "infernix/types.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/frontend/frontend.h"
#include "models/inkling2/ngram.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/request.h"
#include "runtime/contract/resources.h"
#include "runtime/contract/timing.h"
#include "runtime/prefix_cache/persist_result.h"
#include "runtime/prefix_cache/ssd_stats.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace infernix {
struct DeviceContext;
}

namespace infernix::models::inkling2 {

namespace detail {
class ProgramImpl;
struct BasePlanImpl;
struct ContractAccess;
} // namespace detail

inline constexpr std::size_t kMaximumConcurrency = 8;
// The widest prefill chunk: the forward's call capacity, bounded by the routed experts' dispatch.
inline constexpr std::int32_t kMaxPrefillChunk = 8191;
// The default chunk: the fastest measured (every chunk gives the same bits).
inline constexpr std::uint32_t kDefaultPrefillChunk = 512;

// Read-only diagnostics of the Program's physical stores.
struct PhysicalUsageSnapshot {
    runtime::ContextResourceUsage occupied;
    runtime::ContextResourceUsage capacity;
};

// The prefix cache's statistics, as the Engine reads them (the common hybrid resource manager); all zero here: this
// slice has no prefix cache.
struct HybridPrefixCacheStats {
    std::uint32_t nodes                   = 0;
    std::uint32_t snapshots               = 0;
    std::uint32_t device_resident_blocks  = 0;
    std::uint32_t device_evictable_blocks = 0;
    std::uint32_t host_slabs              = 0;
    std::uint32_t host_free_slabs         = 0;
    std::uint64_t host_slab_bytes         = 0;
    std::uint64_t snapshot_hits           = 0;
    std::uint64_t reused_tokens           = 0;
    std::uint64_t blocks_inserted         = 0;
    std::uint64_t blocks_reattached       = 0;
    std::uint64_t blocks_duplicate        = 0;
    std::uint64_t taps_created            = 0;
    std::uint64_t taps_skipped            = 0;
    std::uint64_t endpoints_created       = 0;
    std::uint64_t host_image_writes       = 0;
    std::uint64_t host_block_writes       = 0;
    std::uint64_t host_image_restores     = 0;
    std::uint64_t host_block_restores     = 0;
    std::uint64_t prefetched_blocks       = 0;
    std::uint64_t host_write_bytes        = 0;
    std::uint64_t host_restore_bytes      = 0;
    std::uint64_t evicted_blocks          = 0;
    std::uint64_t host_snapshot_evictions = 0;
    std::uint64_t host_dead_reclaims      = 0;
    std::uint64_t unbacked_node_losses    = 0;
    std::uint32_t held_snapshots          = 0;
    std::uint64_t held_device_evictions   = 0;
    std::uint64_t held_snapshot_losses    = 0;
    std::uint64_t held_host_refusals      = 0;
    std::uint64_t tool_call_holds         = 0;
    runtime::prefix_cache::SsdTierStats ssd;
};

class SequenceHandle {
public:
    SequenceHandle() noexcept                                       = default;
    SequenceHandle(const SequenceHandle&) noexcept                  = default;
    SequenceHandle& operator=(const SequenceHandle&) noexcept       = default;
    friend bool operator==(SequenceHandle, SequenceHandle) noexcept = default;

private:
    const void* owner_   = nullptr;
    std::uint32_t lane_  = 0;
    std::uint64_t epoch_ = 0;

    friend struct detail::ContractAccess;
};

// A root start (the only source without a prefix cache).
struct SourceCandidate {
    std::uint32_t reused_tokens    = 0;
    PrefixReusePath reuse_path     = PrefixReusePath::Root;
    std::uint32_t cached_block_tokens = 0;
};

struct BindingReservation {
    bool reserved          = false;
    bool source_valid      = true;
    bool capacity_possible = true;
    runtime::ContextResourceUsage shortage;
    bool reclaim_cache = true;
    std::exception_ptr refusal;

    explicit operator bool() const noexcept { return reserved; }
};

// No request is paused in this slice; the type exists for the EngineCore's signatures.
class ResumeState {
public:
    ResumeState(ResumeState&&) noexcept            = default;
    ResumeState& operator=(ResumeState&&) noexcept = default;
    ResumeState(const ResumeState&)                = delete;
    ResumeState& operator=(const ResumeState&)     = delete;
    [[nodiscard]] std::uint32_t frontier() const noexcept { return frontier_; }

private:
    ResumeState() = default;
    std::uint32_t frontier_ = 0;
};

enum class ExecutionUnitKind : std::uint8_t { Prefill, Replay, Decode, Control, Normalize };

struct ExecutionUnit {
    SequenceHandle sequence;
    ExecutionUnitKind kind = ExecutionUnitKind::Decode;
    // Decode: remaining output budget. Control: exact forced span. Other kinds: zero.
    std::uint32_t tokens = 0;
};

// The sampled tokens of one model unit, awaiting the Engine's commit decisions. Its token span points into Program
// storage that stays valid until commit or abort_pending.
class PendingBatch {
public:
    PendingBatch() noexcept = default;
    PendingBatch(PendingBatch&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), transaction_(std::exchange(other.transaction_, 0)),
          rows_(other.rows_), row_count_(std::exchange(other.row_count_, 0)),
          tokens_(std::exchange(other.tokens_, {})), row_counts_(std::exchange(other.row_counts_, {})),
          row_stride_(std::exchange(other.row_stride_, 0)), timing_(std::exchange(other.timing_, {})),
          row_failures_(other.row_failures_) {}
    PendingBatch& operator=(PendingBatch&&)      = delete;
    PendingBatch(const PendingBatch&)            = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }
    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }
    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return row_counts_; }
    [[nodiscard]] std::uint32_t row_stride() const noexcept { return row_stride_; }
    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept { return timing_; }
    [[nodiscard]] runtime::PendingRowFailure row_failure(std::size_t row) const { return row_failures_.at(row); }

private:
    const void* owner_         = nullptr;
    std::uint64_t transaction_ = 0;
    std::array<SequenceHandle, kMaximumConcurrency> rows_{};
    std::size_t row_count_ = 0;
    std::span<const TokenId> tokens_;
    std::span<const std::int32_t> row_counts_;
    std::uint32_t row_stride_ = 0;
    runtime::ExecutionTiming timing_;
    std::array<runtime::PendingRowFailure, kMaximumConcurrency> row_failures_{};

    friend struct detail::ContractAccess;
};

class RequestBasePlan {
public:
    explicit RequestBasePlan(std::unique_ptr<detail::BasePlanImpl> impl) noexcept;
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();
    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

    std::unique_ptr<detail::BasePlanImpl> impl_;
};

struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    bool capacity_released                = false;
    bool snapshot_published               = false;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch> pending;
    std::optional<PromptReadout> readout;
};

struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    GenerationTimings timings;
    SpeculativeStats speculative;
    struct SpeculativeCounters {
        std::uint64_t rounds          = 0;
        std::uint64_t drafted_tokens  = 0;
        std::uint64_t accepted_tokens = 0;
        std::uint64_t fallback_steps  = 0;
    } speculative_counters;
};

struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    std::size_t row_count = 0;
    runtime::ExecutionTiming timing;
};

struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::size_t row_count         = 0;
};

struct FinishResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
    std::vector<ConstrainedDraw> constrained_draws;
    std::optional<ExpertCacheStats> expert_cache;
    std::optional<std::uint32_t> cache_created_prompt_tokens;
    std::exception_ptr device_error;
};

struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
    std::exception_ptr device_error;
};

struct ReplayProgress {
    std::uint32_t processed_tokens = 0;
    bool complete                  = false;
    runtime::ExecutionTiming timing;
};

enum class ContextOperationKind : std::uint8_t { Bind, Pause };

struct ContextProgress {
    ContextOperationKind kind = ContextOperationKind::Bind;
    bool advanced             = false;
    bool complete             = false;
    bool published            = false;
    std::optional<SequenceHandle> sequence;
    std::optional<ResumeState> paused;
    bool replaying = false;
    runtime::ContextOperationCounts operations;
    std::optional<GenerationTimings> request_timings;
    SpeculativeStats request_speculative;
};

struct ProgramOptions {
    std::uint32_t max_context     = 0;
    std::uint32_t max_concurrency = 1;
    bool causal_scoring           = false;
    // Columns a prefill call takes of one lane (1..kMaxPrefillChunk); the forward's call capacity follows it.
    std::uint32_t prefill_chunk = kDefaultPrefillChunk;
    bool use_cuda_graph         = true;
    DiagnosticObserver diagnostics;
};

// Device memory the Program allocates, from the options and the configuration alone.
struct ProgramDevicePlan {
    std::uint32_t kv_pages      = 0; // every lane's pages: lanes x ceil((max_context + 16) / 64)
    std::uint64_t kv            = 0; // the full visits' K and V pools and the block table
    std::uint64_t state         = 0; // the lanes' state images
    std::uint64_t workspace     = 0; // the Forward's call buffers and the sampling workspace
    std::uint64_t io            = 0; // per-call inputs, logits rows and sampling arrays
    std::uint32_t graph_bound   = 0;
    [[nodiscard]] std::uint64_t fixed_bytes() const noexcept { return kv + state + workspace + io; }
};

class Program {
public:
    Program(const Parameters& parameters, DeviceContext& device, ProgramOptions options);
    ~Program() noexcept;
    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;

    [[nodiscard]] static ProgramDevicePlan plan_device(const ProgramOptions& options, const Config& config);
    [[nodiscard]] const ProgramDevicePlan& device_plan() const noexcept;

    [[nodiscard]] RequestBasePlan plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options);

    // ---- admission (a root start only) ----
    [[nodiscard]] std::vector<SourceCandidate> hybrid_sources(const RequestBasePlan& base, const ResumeState* resume);
    [[nodiscard]] BindingReservation start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                                   const SourceCandidate& source, ResumeState* resume = nullptr,
                                                   ExecutionUnitKind resume_kind = ExecutionUnitKind::Decode,
                                                   std::uint32_t resume_tokens   = 1);
    [[nodiscard]] ContextProgress poll_context(runtime::CancellationFlagView cancellation);
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] bool context_blocks(SequenceHandle sequence) const noexcept;

    // ---- execution units ----
    [[nodiscard]] runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit> units);
    void release_units(std::span<const SequenceHandle>) noexcept {}
    [[nodiscard]] bool recovery_pending(SequenceHandle) const noexcept { return false; }
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* failed_timing = nullptr,
                                                  runtime::TokenMaskProvider* masks = nullptr);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing = nullptr,
                                      runtime::TokenMaskProvider* masks       = nullptr);
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences, std::span<const TokenId> row_major_tokens,
                         std::uint32_t row_stride, std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                         runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CommitResult commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
                                      runtime::CommitObservation observation  = runtime::CommitObservation::AllRows,
                                      runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;

    void fail_all_cleanup() noexcept;
    [[nodiscard]] bool recover_after_failure() noexcept;
    void shutdown_cleanup() noexcept;

    // Teacher-forced scoring (ProgramOptions::causal_scoring; an idle Program).
    [[nodiscard]] ScoreResult causal_score(PreparedPrompt&& prompt, std::uint32_t first_target, const ScoreOptions& options);

    [[nodiscard]] MemoryUsage memory_usage() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept {}

    // ---- the common contract's hooks this slice has nothing behind (runtime/contract/program.h) ----
    void maintain() {}
    void maintain_busy() {}
    void set_maintenance_waker(std::function<void()>) {}
    [[nodiscard]] std::uint64_t engine_thread_cpus() const noexcept { return 0; }
    [[nodiscard]] bool hybrid_poll_storage() { return false; }
    [[nodiscard]] bool hybrid_source_pending() const noexcept { return false; }
    [[nodiscard]] bool hybrid_storage_due() const noexcept { return false; }
    [[nodiscard]] std::optional<std::chrono::steady_clock::time_point> hybrid_storage_retry_at() const noexcept {
        return std::nullopt;
    }
    [[nodiscard]] std::uint32_t hybrid_resume_grid() const noexcept { return 16; }
    [[nodiscard]] std::optional<runtime::prefix_cache::PersistResult> save_prefix_cache_now(const CancellationView&) {
        return std::nullopt;
    }
    [[nodiscard]] std::optional<ExpertCacheTotals> expert_cache_totals() const noexcept { return std::nullopt; }
    [[nodiscard]] HybridPrefixCacheStats hybrid_stats() const noexcept { return {}; }
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    // No prefix cache: nothing to prefetch, reclaim or hold.
    [[nodiscard]] std::optional<std::uint32_t> hybrid_prefetch(const RequestBasePlan&, bool = false) {
        return std::nullopt;
    }
    [[nodiscard]] std::uint32_t hybrid_prefetch_room() const noexcept { return 0; }
    [[nodiscard]] bool hybrid_reclaim(runtime::ContextResourceUsage) { return false; }
    [[nodiscard]] std::uint64_t hybrid_cache_epoch() const noexcept { return 0; }
    [[nodiscard]] bool hybrid_holds_stale() const noexcept { return false; }
    void hybrid_hold_queue(std::span<const RequestBasePlan* const>, std::span<const ResumeState* const>) {}
    void hybrid_hold_continuation(std::chrono::steady_clock::time_point) {
        throw std::logic_error("Inkling 2: the prefix cache has no tool-call holds");
    }
    // Every lane's KV is reserved at startup, so no shortage ever pauses a request.
    [[nodiscard]] std::uint32_t pause_replay_tokens(SequenceHandle) const noexcept { return 0; }
    [[nodiscard]] bool start_pause(SequenceHandle, runtime::ExecutionTiming* = nullptr) {
        throw std::logic_error("Inkling 2: no request is paused (every lane's KV is reserved at startup)");
    }
    [[nodiscard]] ReplayProgress advance_replay(SequenceHandle, runtime::ExecutionTiming* = nullptr) {
        throw std::logic_error("Inkling 2: no request is replayed (every lane's KV is reserved at startup)");
    }

private:
    std::unique_ptr<detail::ProgramImpl> impl_;
};

// The common EngineCore is instantiated once for this model.
struct RuntimeTypes {
    using Frontend           = inkling2::Frontend;
    using PreparedPrompt     = inkling2::PreparedPrompt;
    using NgramArchive       = inkling2::NgramArchive;
    using OutputSession      = inkling2::OutputSession;
    using PublishedOutput    = inkling2::PublishedOutput;
    using RequestBasePlan    = inkling2::RequestBasePlan;
    using SequenceHandle     = inkling2::SequenceHandle;
    using SourceCandidate    = inkling2::SourceCandidate;
    using ResumeState        = inkling2::ResumeState;
    using ExecutionUnit      = inkling2::ExecutionUnit;
    using ExecutionUnitKind  = inkling2::ExecutionUnitKind;
    using ContextProgress    = inkling2::ContextProgress;
    using PendingBatch       = inkling2::PendingBatch;
    using PrefillProgress    = inkling2::PrefillProgress;
    using ReplayProgress     = inkling2::ReplayProgress;
    using CommitResult       = inkling2::CommitResult;
    using DiscardResult      = inkling2::DiscardResult;
    using FinishResult       = inkling2::FinishResult;
    using AbortResult        = inkling2::AbortResult;
    using BindingReservation = inkling2::BindingReservation;
    using Program            = inkling2::Program;
};

} // namespace infernix::models::inkling2
