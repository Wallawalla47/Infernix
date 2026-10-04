#pragma once

// The Qwen4Exp Program: the Engine-facing execution contract of the common EngineCore over the
// HybridResourceManager surface (docs/maintainer/engine-architecture.md §2.4, §3.3). It owns the
// physical lanes (one per concurrency slot), their KV pages, recurrent state, sampling state and
// the one pending model unit. Prompts and outputs use the Qwen3.5 frontend (same tokenizer and
// chat template, design §7).
//
// Admission reserves a request's whole KV extent (prompt plus its effective output ceiling) when
// it is admitted, so an admitted request always completes; there is no prefix cache yet, so every
// admission reuses nothing.

#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/output_session.h"
#include "models/qwen3_5/ngram.h"
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/request.h"
#include "runtime/contract/resources.h"
#include "runtime/contract/timing.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::models::qwen4_exp {

namespace execution {
class Parameters;
}

namespace detail {
class ProgramImpl;
struct BasePlanImpl;
struct QuoteImpl;
struct ContractAccess;
} // namespace detail

using PreparedPrompt  = qwen3_5::PreparedPrompt;
using Frontend        = qwen3_5::Frontend;
using OutputSession   = qwen3_5::OutputSession;
using PublishedOutput = qwen3_5::PublishedOutput;
using NgramArchive    = qwen3_5::NgramArchive;

// Read-only diagnostics of the Program's physical stores.
struct PhysicalUsageSnapshot {
    runtime::ProgramResourceRevision resource_revision;
    std::uint32_t device_state_slots            = 0;
    std::uint32_t host_state_slots              = 0;
    std::uint32_t device_main_kv_pages          = 0;
    std::uint32_t device_backend_kv_pages       = 0;
    std::uint32_t device_main_kv_lease_pages    = 0;
    std::uint32_t device_backend_kv_lease_pages = 0;
    std::size_t host_kv_bytes                   = 0;
};

struct DeviceKVLeaseShortfall {
    std::uint32_t main_pages    = 0;
    std::uint32_t backend_pages = 0;
};

// The hybrid prefix cache's statistics; all zero while Qwen4Exp has no prefix cache.
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
    std::uint64_t host_write_bytes        = 0;
    std::uint64_t host_restore_bytes      = 0;
    std::uint64_t evicted_blocks          = 0;
    std::uint64_t host_snapshot_evictions = 0;
    std::uint64_t host_dead_reclaims      = 0;
    std::uint64_t unbacked_node_losses    = 0;
};

class SequenceHandle {
public:
    SequenceHandle() noexcept                                 = default;
    SequenceHandle(const SequenceHandle&) noexcept            = default;
    SequenceHandle& operator=(const SequenceHandle&) noexcept = default;

private:
    const void* owner_   = nullptr;
    std::uint32_t lane_  = 0;
    std::uint64_t epoch_ = 0;

    friend struct detail::ContractAccess;
};

// Qwen4Exp offers no prefix captures; the type exists for the common Engine surface.
class CaptureOffer {
public:
    CaptureOffer() noexcept                          = default;
    CaptureOffer(CaptureOffer&&) noexcept            = default;
    CaptureOffer& operator=(CaptureOffer&&)          = delete;
    CaptureOffer(const CaptureOffer&)                = delete;
    CaptureOffer& operator=(const CaptureOffer&)     = delete;
};

// The sampled tokens of one model unit, awaiting the Engine's commit decisions. Its token span
// points into Program storage that stays valid until commit or abort_pending.
class PendingBatch {
public:
    PendingBatch() noexcept = default;
    PendingBatch(PendingBatch&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), transaction_(std::exchange(other.transaction_, 0)),
          rows_(other.rows_), row_count_(std::exchange(other.row_count_, 0)),
          tokens_(std::exchange(other.tokens_, {})), row_counts_(std::exchange(other.row_counts_, {})),
          row_stride_(std::exchange(other.row_stride_, 0)),
          timing_(std::exchange(other.timing_, {})) {}
    PendingBatch& operator=(PendingBatch&&)      = delete;
    PendingBatch(const PendingBatch&)            = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }
    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }
    // Licensed tokens per row; empty when every row licenses one.
    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return row_counts_; }
    [[nodiscard]] std::uint32_t row_stride() const noexcept { return row_stride_; }
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

// Never minted: the hybrid manager never overtakes a blocked FIFO head.
class PersistentBackfillProof {
public:
    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept { return {}; }
};

struct AdmissionCandidate {};

struct HybridAdmissionQuote {
    runtime::Readiness readiness = runtime::Readiness::TemporarilyBlocked;
    runtime::LaneId destination{};
    runtime::RequestPlanSummary summary;
    std::shared_ptr<detail::QuoteImpl> impl;
};

struct StartResult {
    SequenceHandle sequence;
};

struct MaterializationResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    std::optional<StartResult> published;
    MaterializationDiagnostics diagnostics;
};

using ContextTransactionProgress = std::variant<runtime::ContextTransactionInProgress, MaterializationResult>;

struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    bool completes_service_unit           = true;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch> pending;
    std::optional<CaptureOffer> capture;
};

struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    std::array<std::optional<CaptureOffer>, kMaximumConcurrency> captures{};
    std::size_t row_count = 0;
    runtime::ExecutionTiming timing;
};

struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::size_t row_count         = 0;
};

struct FinishResult {
    runtime::ConsumeStatus status          = runtime::ConsumeStatus::InvariantMismatch;
    runtime::FinishDisposition disposition = runtime::FinishDisposition::Released;
    GenerationTimings timings;
    SpeculativeStats speculative;
    bool salvaged = false;
};

struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
    bool salvaged = false;
};

struct ProgramOptions {
    std::uint32_t max_context     = 0;
    std::uint32_t max_concurrency = 1;
    std::uint32_t prefill_chunk   = 1024;
    // Main KV capacity in tokens, shared by every lane.
    std::uint32_t kv_capacity_tokens = 0;
    KvCacheStorage kv_cache          = KvCacheStorage::Int8Group64;
    std::filesystem::path ngram_volume;
    // Device memory left free after the expert frames take the rest. 384 MiB (98.8 % of the 5090
    // in use) measured safe through decode, 4K-token prefill chunks and verification graphs
    // (design section 19.2).
    std::size_t expert_cache_reserve_bytes = std::size_t{384} << 20;
    bool expert_cache                      = true;
    // Speculative decoding with n-gram copy proposals (design section 11.3): at most this many
    // draft tokens per round (0 disables, at most 15), proposed only from a match of at least
    // ngram_min_match tokens (4..64) earlier in the request.
    std::uint32_t ngram_draft_tokens = 0;
    std::uint32_t ngram_min_match    = 12;
    // MTP drafts per round when the drafter is loaded (design section 11, at most 7). A longer
    // n-gram proposal, when enabled, replaces a round's MTP drafts.
    std::uint32_t mtp_draft_tokens = 0;
    // CPU-served misses (design section 10): host expert-engine workers (0 disables), the most
    // experts one layer call hands to them, and the share kept on the PCIe stage (misses / divisor).
    // Defaults measured fastest on the i9-13900K (design section 19.2); 8 jobs pay under MTP's
    // wider verification calls and are neutral for plain decode.
    std::uint32_t cpu_expert_workers = 6;
    std::uint32_t cpu_expert_jobs    = 8;
    std::int32_t cpu_pcie_divisor    = 3;
    DiagnosticObserver diagnostics;
};

class Program {
public:
    Program(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options);
    ~Program() noexcept;
    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;

    [[nodiscard]] RequestBasePlan plan_request(const PreparedPrompt& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] bool isolated_request_feasible(const RequestBasePlan& base) const noexcept;

    [[nodiscard]] HybridAdmissionQuote hybrid_quote(const PreparedPrompt& prompt, const RequestBasePlan& base,
                                                    runtime::LaneId destination);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    hybrid_reserve_materialization(HybridAdmissionQuote&& quote, PreparedPrompt&& prompt,
                                   runtime::CancellationFlagView cancellation);
    [[nodiscard]] ContextTransactionProgress progress_context_transaction(runtime::CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] bool wait_context_transfer() noexcept { return false; }
    [[nodiscard]] std::uint32_t hybrid_reclaim_device_kv(std::uint32_t, std::uint32_t) { return 0; }
    [[nodiscard]] std::optional<std::uint32_t> hybrid_prefetch(const PreparedPrompt&, const RequestBasePlan&) {
        return 0U;
    }
    [[nodiscard]] std::uint32_t hybrid_prefetch_room() const noexcept { return 0; }
    [[nodiscard]] HybridPrefixCacheStats hybrid_stats() const noexcept { return {}; }
    void skip_capture(CaptureOffer&&) noexcept {}

    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* failed_timing,
                                                  runtime::PrefillStepWidth width);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences, std::span<const TokenId> row_major_tokens,
                         std::uint32_t row_stride,
                         std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                         runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] CommitResult commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
                                      runtime::CommitObservation observation,
                                      runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;

    // The whole KV extent is reserved at admission, so leases never need to grow.
    [[nodiscard]] std::optional<std::uint32_t> device_kv_lease_settlement_tokens(SequenceHandle,
                                                                                 std::uint32_t) const noexcept {
        return std::nullopt;
    }
    [[nodiscard]] std::optional<DeviceKVLeaseShortfall> device_kv_lease_shortfall(SequenceHandle) const noexcept {
        return std::nullopt;
    }
    [[nodiscard]] bool resume_device_kv_lease(SequenceHandle) noexcept { return true; }

    [[nodiscard]] std::optional<PhysicalUsageSnapshot> fail_all_cleanup() noexcept;
    [[nodiscard]] std::optional<PhysicalUsageSnapshot> shutdown_cleanup() noexcept { return fail_all_cleanup(); }

    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept {}

private:
    std::unique_ptr<detail::ProgramImpl> impl_;
};

// The common EngineCore is instantiated once for this model.
struct RuntimeTypes {
    using Frontend                = qwen4_exp::Frontend;
    using PreparedPrompt          = qwen4_exp::PreparedPrompt;
    using NgramArchive            = qwen4_exp::NgramArchive;
    using OutputSession           = qwen4_exp::OutputSession;
    using PublishedOutput         = qwen4_exp::PublishedOutput;
    using RequestBasePlan         = qwen4_exp::RequestBasePlan;
    using AdmissionCandidate      = qwen4_exp::AdmissionCandidate;
    using PersistentBackfillProof = qwen4_exp::PersistentBackfillProof;
    using SequenceHandle          = qwen4_exp::SequenceHandle;
    using CaptureOffer            = qwen4_exp::CaptureOffer;
    using HybridAdmissionQuote    = qwen4_exp::HybridAdmissionQuote;
    using MaterializationResult   = qwen4_exp::MaterializationResult;
    using StartResult             = qwen4_exp::StartResult;
    using PendingBatch            = qwen4_exp::PendingBatch;
    using PrefillProgress         = qwen4_exp::PrefillProgress;
    using CommitResult            = qwen4_exp::CommitResult;
    using DiscardResult           = qwen4_exp::DiscardResult;
    using FinishResult            = qwen4_exp::FinishResult;
    using AbortResult             = qwen4_exp::AbortResult;
    using Program                 = qwen4_exp::Program;
};

} // namespace ninfer::models::qwen4_exp
