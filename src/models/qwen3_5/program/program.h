#pragma once
#include "infernix/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "runtime/prefix_cache/cost.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace infernix {
struct DeviceContext;
}

namespace infernix::models::qwen3_5 {
namespace execution {
class Parameters;
}

namespace detail {
struct SequencePlanImpl;
struct SequencePlannerImpl;
struct RequestBasePlanImpl;
struct ResumeStateImpl;
class ProgramImpl;
struct ContractAccess;
struct HybridQuoteImpl;
} // namespace detail
class SequencePlanner;
class Program;

struct GraphExecutionProfile {
    std::uint32_t min            = 0;
    std::uint32_t max            = 0;
    std::uint32_t topology_class = 0;
};
enum class TextPhase { Prefill, Verify };

class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t host_capacity_bytes() const noexcept;
    // Whether automatic DFlash2 tree widths are active: requested and verifiable by the target.
    [[nodiscard]] bool draft_tree_auto() const noexcept;

private:
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl> impl_;

    friend class SequencePlanner;

    friend std::unique_ptr<Program> create_program(const execution::Parameters&, SequencePlan&&,
                                                   DeviceContext&, const StartupObserver&);
};

class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept;
    [[nodiscard]] SequencePlan finalize(std::uint32_t main_page_groups) &&;

private:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl> impl_;

    friend SequencePlanner make_sequence_planner(const execution::Parameters&, DeviceContext&,
                                                 const EngineOptions&);
};

class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

private:
    explicit RequestBasePlan(std::shared_ptr<detail::RequestBasePlanImpl> impl) noexcept;
    std::shared_ptr<detail::RequestBasePlanImpl> impl_;

    friend class detail::ProgramImpl;
};

class SequenceHandle {
public:
    SequenceHandle() noexcept                                       = default;
    SequenceHandle(const SequenceHandle&) noexcept                  = default;
    SequenceHandle& operator=(const SequenceHandle&) noexcept       = default;
    friend bool operator==(SequenceHandle, SequenceHandle) noexcept = default;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;

    friend struct detail::ContractAccess;
};

// One admission source (docs/maintainer/hybrid-prefix-cache-spec.md §6): a quote of the
// block-tree path and snapshot to resume from, or of a root start.
struct SourceCandidate {
    std::uint32_t reused_tokens = 0;
    std::shared_ptr<const detail::HybridQuoteImpl> hybrid;
    PrefixReusePath reuse_path = PrefixReusePath::Root;
};

struct BindingReservation {
    bool reserved          = false;
    bool source_valid      = true;
    bool capacity_possible = true;
    runtime::ContextResourceUsage shortage;

    explicit operator bool() const noexcept { return reserved; }
};

// Outcome of saving or restoring the hybrid prefix cache's Host tier.
struct HybridCachePersistence {
    bool ok = false;
    std::string message;
    std::uint64_t blocks    = 0;
    std::uint64_t snapshots = 0;
    std::uint64_t bytes     = 0;
    double seconds          = 0.0;
    // Load only: what the file holds and needs, against this Host tier.
    std::uint64_t saved_blocks        = 0;
    std::uint64_t saved_snapshots     = 0;
    std::uint64_t required_host_bytes = 0;
    std::uint64_t host_bytes          = 0;
};

struct HybridPrefixCacheStats {
    std::uint32_t nodes                      = 0;
    std::uint32_t snapshots                  = 0;
    std::uint32_t device_resident_blocks     = 0;
    std::uint32_t device_evictable_blocks    = 0;
    std::uint32_t host_slabs                 = 0;
    std::uint32_t host_free_slabs            = 0;
    std::uint64_t host_slab_bytes            = 0;
    std::uint32_t free_device_snapshot_slots = 0;
    std::uint64_t admissions                 = 0;
    std::uint64_t snapshot_hits              = 0;
    std::uint64_t reused_tokens              = 0;
    std::uint64_t blocks_inserted            = 0;
    std::uint64_t blocks_reattached          = 0;
    std::uint64_t blocks_duplicate           = 0;
    std::uint64_t taps_created               = 0;
    std::uint64_t taps_skipped               = 0;
    std::uint64_t endpoints_created          = 0;
    std::uint64_t host_image_writes          = 0;
    std::uint64_t host_block_writes          = 0;
    std::uint64_t host_image_restores        = 0;
    std::uint64_t host_block_restores        = 0;
    std::uint64_t prefetched_blocks          = 0;
    std::uint64_t host_tail_restores         = 0;
    std::uint64_t host_write_bytes           = 0;
    std::uint64_t host_restore_bytes         = 0;
    std::uint64_t evicted_blocks             = 0;
    std::uint64_t host_snapshot_evictions    = 0;
    std::uint64_t host_dead_reclaims         = 0;
    std::uint64_t unbacked_node_losses       = 0;
    std::uint32_t held_snapshots             = 0;
    std::uint64_t held_device_evictions      = 0;
    std::uint64_t held_snapshot_losses       = 0;
    std::uint64_t held_host_refusals         = 0;
};

class ResumeState {
public:
    ResumeState(ResumeState&&) noexcept;
    ResumeState& operator=(ResumeState&&) noexcept;
    ~ResumeState();
    ResumeState(const ResumeState&)            = delete;
    ResumeState& operator=(const ResumeState&) = delete;
    [[nodiscard]] std::uint32_t frontier() const noexcept;

private:
    explicit ResumeState(std::unique_ptr<detail::ResumeStateImpl>) noexcept;
    std::unique_ptr<detail::ResumeStateImpl> impl_;

    friend class detail::ProgramImpl;
};
enum class ExecutionUnitKind : std::uint8_t { Prefill, Replay, Decode, Control, Normalize };

struct ExecutionUnit {
    SequenceHandle sequence;
    ExecutionUnitKind kind = ExecutionUnitKind::Decode;
    // Decode: remaining output budget. Control: exact forced span. Other kinds: zero.
    std::uint32_t tokens = 0;
};

class PendingBatch {
public:
    PendingBatch() noexcept = default;
    ~PendingBatch()         = default;

    PendingBatch(PendingBatch&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          transaction_(std::exchange(other.transaction_, 0)), rows_(other.rows_),
          row_count_(std::exchange(other.row_count_, 0)), tokens_(other.tokens_),
          row_counts_(other.row_counts_), row_stride_(other.row_stride_), timing_(other.timing_),
          constraint_failed_(other.constraint_failed_) {
        other.tokens_     = {};
        other.row_counts_ = {};
        other.row_stride_ = 0;
        other.timing_     = {};
    }

    PendingBatch& operator=(PendingBatch&&)      = delete;
    PendingBatch(const PendingBatch&)            = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }

    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }

    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return row_counts_; }

    [[nodiscard]] std::uint32_t row_stride() const noexcept { return row_stride_; }

    [[nodiscard]] bool constraint_failed(std::size_t row) const {
        return constraint_failed_.at(row);
    }

    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept { return timing_; }

private:
    const void* owner_         = nullptr;
    std::uint64_t transaction_ = 0;
    std::array<SequenceHandle, kMaximumConcurrency> rows_{};
    std::size_t row_count_ = 0;
    std::span<const TokenId> tokens_;
    std::span<const std::int32_t> row_counts_;
    std::uint32_t row_stride_ = 0;
    runtime::ExecutionTiming timing_;
    std::array<bool, kMaximumConcurrency> constraint_failed_{};

    friend struct detail::ContractAccess;
};

struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch> pending;
    std::optional<PromptReadout> readout;
};

struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    GenerationTimings timings;
    SpeculativeStats speculative;

    // Fixed-size cumulative observation, including active rows without copying per-position data.
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
};

struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
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

struct PhysicalUsageSnapshot {
    runtime::ContextResourceUsage occupied;
    runtime::ContextResourceUsage capacity;
};

class Program {
public:
    ~Program() noexcept;
    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;
    [[nodiscard]] RequestBasePlan plan_request(PreparedPrompt&& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] ScoreResult causal_score(PreparedPrompt&& prompt, std::uint32_t first_target,
                                           const ScoreOptions& options);
    // Reservations stay owned by their lane through commit. Failure leaves prior permits intact.
    [[nodiscard]] runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit> units);
    void release_units(std::span<const SequenceHandle> sequences) noexcept;
    [[nodiscard]] BindingReservation
    start_binding(const RequestBasePlan& base, runtime::LaneId lane, const SourceCandidate& source,
                  ResumeState* resume           = nullptr,
                  ExecutionUnitKind resume_kind = ExecutionUnitKind::Decode,
                  std::uint32_t resume_tokens   = 1);
    // Releases the lane and opens a Pause transaction that returns the request's ResumeState: its
    // ledger, frontier and request-level state. The resume binding restores the deepest cached
    // state up to the frontier and replays the rest.
    [[nodiscard]] bool start_pause(SequenceHandle sequence,
                                   runtime::ExecutionTiming* timing = nullptr);
    [[nodiscard]] ContextProgress poll_context(runtime::CancellationFlagView cancellation);
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] bool context_blocks(SequenceHandle sequence) const noexcept;
    // A resumed binding retains its complete recovery capacity until committed new progress.
    [[nodiscard]] bool recovery_pending(SequenceHandle sequence) const noexcept;
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence,
                                                  runtime::ExecutionTiming* failed_timing = nullptr,
                                                  runtime::TokenMaskProvider* masks = nullptr);
    [[nodiscard]] ReplayProgress advance_replay(SequenceHandle sequence,
                                                runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing = nullptr,
                                      runtime::TokenMaskProvider* masks       = nullptr);
    // Forced control contributes to counts once. Replay does not call this operation.
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                         runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CommitResult
    commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
           runtime::CommitObservation observation  = runtime::CommitObservation::AllRows,
           runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    void fail_all_cleanup() noexcept;
    // fail_all_cleanup for the Engine's orderly stop. With a hybrid cache file attached, the Host
    // tier is saved once every lane has written its blocks through and before the cleanup drops
    // it; hybrid_shutdown_save() reports the result.
    void shutdown_cleanup() noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;

    // The prefix cache, when the context cache is enabled. Admission binds a lane through a
    // context transaction: hybrid_sources() quotes, start_binding() stages and poll_context()
    // activates.
    // The quoted sources for a request in preference order: the chosen cached path, if any, then
    // a root start. A resumed request's source never passes `maximum_frontier`. Empty while a
    // prefilling sibling is about to publish the snapshot this request should resume from.
    [[nodiscard]] std::vector<SourceCandidate> hybrid_sources(const RequestBasePlan& base,
                                                              std::uint32_t maximum_frontier);
    // Evicts unpinned cached Device blocks, and the Device copies of cached snapshot images, until
    // the shortage is covered or nothing evictable remains. Returns whether anything was freed.
    [[nodiscard]] bool hybrid_reclaim(runtime::ContextResourceUsage shortage);
    // Copies Host-only blocks a waiting request resumes from into Device pages the pools can
    // spare as cache (hybrid-prefix-cache-spec §6.6), so its admission restores less. Returns the
    // blocks whose copy started; absent while a prefetch or an admission is still in flight.
    [[nodiscard]] std::optional<std::uint32_t> hybrid_prefetch(const RequestBasePlan& base);
    // Device pages a prefetch could fill now: free ones and host-backed cached ones.
    [[nodiscard]] std::uint32_t hybrid_prefetch_room() const noexcept;
    // Holds the snapshot each queued request would resume from now, in admission order
    // (hybrid-prefix-cache-spec §9.6); an empty queue releases every hold.
    void hybrid_hold_queue(std::span<const RequestBasePlan* const> queue);
    // Changes when a queued request's choice of snapshot may have changed.
    [[nodiscard]] std::uint64_t hybrid_cache_epoch() const noexcept;
    [[nodiscard]] HybridPrefixCacheStats hybrid_stats() const noexcept;
    // Installs the Engine's calibrated machine model for hybrid admission choice and eviction.
    void set_hybrid_cost(const runtime::prefix_cache::CacheCostModel& cost);
    // Longest predicted wait for a prefilling sibling's snapshot that admission may choose over
    // prefilling the shared prefix again. Zero disables coalescing.
    void set_hybrid_coalesce_wait_limit(double seconds);
    // Restores a saved Host tier before the first request and attaches the file, so
    // shutdown_cleanup saves the tier back to it. `fingerprint` names everything the saved bytes
    // depend on; `observer` receives the file read as StartupPhase::PrefixCacheLoad.
    [[nodiscard]] HybridCachePersistence attach_hybrid_cache_file(const std::filesystem::path& path,
                                                                  std::string fingerprint,
                                                                  const StartupObserver& observer);
    [[nodiscard]] std::optional<HybridCachePersistence> hybrid_shutdown_save() const;
    // Between units (no unit's Device work in flight): writes the Host tier to the attached file
    // when it has been written since the last save or load (Engine periodic save,
    // HybridPrefixCacheOptions::persistent_save_interval). Absent without a file or a change.
    [[nodiscard]] std::optional<HybridCachePersistence> save_prefix_cache_now(const CancellationView& abandoned);
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;
private:
    explicit Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl> impl_;
    friend std::unique_ptr<Program> create_program(const execution::Parameters&, SequencePlan&&,
                                                   DeviceContext&, const StartupObserver&);
};

[[nodiscard]] SequencePlanner make_sequence_planner(const execution::Parameters&, DeviceContext&,
                                                    const EngineOptions&);
[[nodiscard]] std::unique_ptr<Program> create_program(const execution::Parameters&, SequencePlan&&,
                                                      DeviceContext&, const StartupObserver&);
} // namespace infernix::models::qwen3_5
