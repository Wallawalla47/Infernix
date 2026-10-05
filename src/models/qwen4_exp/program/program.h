#pragma once

// The Qwen4Exp Program: the Engine-facing execution contract of the common EngineCore over the
// HybridResourceManager surface (docs/maintainer/engine-architecture.md §2.4, §3.3). It owns the
// physical lanes (one per concurrency slot), their KV pages, recurrent state, sampling state and
// the one pending model unit. Prompts and outputs use the Qwen3.5 frontend (same tokenizer and
// chat template, design §7).
//
// Admission reserves a request's whole KV extent (prompt plus its effective output ceiling) when
// it is admitted, so an admitted request always completes. With the prefix cache, an admission
// resumes from the deepest affordable snapshot and maps the cached blocks below it.

#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/output_session.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/ngram.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/memory_plan.h"
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/request.h"
#include "runtime/contract/resources.h"
#include "runtime/contract/timing.h"
#include "runtime/prefix_cache/cost.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>
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
    runtime::ContextResourceUsage occupied;
    runtime::ContextResourceUsage capacity;
    std::size_t host_reserved_bytes      = 0;
    std::size_t host_peak_occupied_bytes = 0;
    std::uint32_t host_state_slots       = 0;
    std::size_t host_kv_bytes            = 0;
};

// The hybrid prefix cache's statistics; all zero without a prefix cache.
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
    std::uint64_t host_block_restores     = 0; // prefetched blocks included
    std::uint64_t prefetched_blocks       = 0;
    std::uint64_t host_write_bytes        = 0;
    std::uint64_t host_restore_bytes      = 0;
    std::uint64_t evicted_blocks          = 0;
    std::uint64_t host_snapshot_evictions = 0;
    std::uint64_t host_dead_reclaims      = 0;
    std::uint64_t unbacked_node_losses    = 0;
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

// The key of an original-cache demand record. Qwen4Exp has no original cache, so none is ever made.
struct PrefixShortlistKey {
    std::array<std::uint64_t, 2> digests{};
    std::uint32_t frontier                                                  = 0;
    std::uint32_t identity_tag                                              = 0;
    friend bool operator==(PrefixShortlistKey, PrefixShortlistKey) noexcept = default;
};

// A reclaim plan of the original cache's checkpoints; Qwen4Exp has none.
struct ContextReclaimPlan {};

// Qwen4Exp keeps no checkpoints (its prefix cache's snapshots live in the block tree, and a request
// is never paused); the handle exists for the common Engine surface and is never minted.
struct CheckpointHandle {
    const void* owner                                                   = nullptr;
    std::uint32_t index                                                 = 0;
    std::uint64_t generation                                            = 0;
    friend bool operator==(CheckpointHandle, CheckpointHandle) noexcept = default;
};

struct CheckpointSummary {
    std::uint32_t frontier       = 0;
    runtime::CheckpointRole role = runtime::CheckpointRole::Continuation;
    bool leased                  = false;
    runtime::ContextResourceUsage evictable_resources;
};

// One admission source: a quote of the prefix cache's chosen path and snapshot to resume from, or
// of a root start (docs/maintainer/hybrid-prefix-cache-spec.md §17).
struct SourceCandidate {
    std::optional<CheckpointHandle> checkpoint; // never set
    std::uint32_t reused_tokens = 0;
    runtime::PrefillWork remaining_work;
    std::vector<runtime::ContextTransferRequirement> transfers;
    std::vector<CheckpointHandle> private_points; // never set
    std::shared_ptr<const detail::QuoteImpl> hybrid;
    PrefixReusePath reuse_path = PrefixReusePath::Root;
};

// A paused request's recovery state. Qwen4Exp reserves a request's whole KV extent at admission and
// never pauses one, so no ResumeState is ever produced; the type completes the Engine surface.
class ResumeState {
public:
    ResumeState(ResumeState&&) noexcept            = default;
    ResumeState& operator=(ResumeState&&) noexcept = default;
    ResumeState(const ResumeState&)                = delete;
    ResumeState& operator=(const ResumeState&)     = delete;
    [[nodiscard]] bool has_snapshot() const noexcept { return false; }
    [[nodiscard]] std::optional<CheckpointHandle> snapshot_handle() const noexcept { return std::nullopt; }
    [[nodiscard]] std::uint32_t frontier() const noexcept { return 0; }

private:
    ResumeState() noexcept = default;
};

enum class ExecutionUnitKind : std::uint8_t { Prefill, Replay, Decode, Control, Normalize };

struct ExecutionUnit {
    SequenceHandle sequence;
    ExecutionUnitKind kind = ExecutionUnitKind::Decode;
    // Decode: remaining output budget. Control: exact forced span. Other kinds: zero.
    std::uint32_t tokens = 0;
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
    [[nodiscard]] std::optional<PrefixShortlistKey> prefix_shortlist_key(std::uint32_t) const noexcept {
        return std::nullopt;
    }

    std::unique_ptr<detail::BasePlanImpl> impl_;
};

struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch> pending;
    bool capture_ready = false; // never: Qwen4Exp captures through its own prefix cache
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
    std::array<bool, kMaximumConcurrency> capture_ready{};
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
    std::optional<CheckpointHandle> checkpoint; // never set
};

struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct ReplayProgress {
    bool capture_ready             = false;
    std::uint32_t processed_tokens = 0;
    bool complete                  = false;
    runtime::ExecutionTiming timing;
};

struct CapturePreparation {
    std::uint32_t frontier = 0;
    bool reserved          = false;
    runtime::ContextResourceUsage shortage;
    std::size_t host_bytes = 0;
};

enum class ContextOperationKind : std::uint8_t { Bind, Capture, Demote, Pause };

struct ContextProgress {
    ContextOperationKind kind = ContextOperationKind::Bind;
    bool advanced             = false;
    bool complete             = false;
    bool published            = false;
    std::optional<SequenceHandle> sequence;
    std::optional<ResumeState> paused;
    bool replaying = false;
    std::vector<CheckpointHandle> private_points;
    std::vector<CheckpointHandle> retired_checkpoints;
    std::vector<CheckpointHandle> captured_checkpoints;
    std::vector<runtime::ContextTransferObservation> transfers;
    runtime::ContextOperationCounts operations;
    std::optional<GenerationTimings> request_timings;
    SpeculativeStats request_speculative;
};

// Device memory a Program allocates besides its expert frames (design §19.3.7), from the options
// and the model's configuration alone, so startup checks it before the weights are read. The
// constructor allocates exactly these sizes.
struct ProgramDevicePlan {
    std::uint32_t kv_pages = 0;
    std::uint64_t kv        = 0; // KV pages and execution tables (an elastic pool: its base, mapped at startup)
    std::uint64_t kv_max    = 0; // the KV at --max-context; above `kv`, an elastic pool takes it from the frames
    std::uint64_t workspace = 0; // the forward, sampling, acceptance and MTP workspace arena
    std::uint64_t staging   = 0; // the expert miss staging slots
    std::uint64_t state     = 0; // GDN, PLE and QSA tail state, verification records, MTP columns
    std::uint64_t io        = 0; // per-call inputs, logits, sampling and round arrays
    std::uint64_t residency = 0; // the expert frame tables and route log
    std::uint64_t expert_record_stride = 0; // one expert frame
    std::uint32_t max_frames           = 0; // the most frames the expert cache uses
    std::uint32_t graph_bound          = 0; // CUDA graph executables the Program may instantiate

    [[nodiscard]] std::uint64_t fixed_bytes() const noexcept {
        return kv + workspace + staging + state + io + residency;
    }
};

// The prefix cache's Host tier saved at shutdown or loaded at startup (--prefix-cache-file).
struct PrefixCachePersistence {
    bool ok = false;
    std::string message;
    std::uint64_t blocks    = 0;
    std::uint64_t snapshots = 0;
    std::uint64_t bytes     = 0;
    double seconds          = 0.0;
    std::uint64_t saved_blocks        = 0;
    std::uint64_t saved_snapshots     = 0;
    std::uint64_t required_host_bytes = 0;
    std::uint64_t host_bytes          = 0;
};

struct ProgramOptions {
    std::uint32_t max_context     = 0;
    std::uint32_t max_concurrency = 1;
    std::uint32_t prefill_chunk   = 1024;
    // Main KV capacity in tokens, shared by every lane.
    std::uint32_t kv_capacity_tokens = 0;
    KvCacheStorage kv_cache          = KvCacheStorage::Int8Group64;
    std::filesystem::path ngram_volume;
    // The expert cache's saved state (design §19.3.5 S4b): read at startup to fill the frames, written
    // at stop and every 10 minutes between requests. Empty: no warm start. The identity names the
    // artifact the state belongs to.
    std::filesystem::path expert_state;
    std::string expert_state_identity;
    // Device memory left free for the display and other programs once the fixed allocations and
    // the expert frames are made (design §19.3.7); empty selects it from the display state.
    std::optional<std::uint64_t> vram_headroom;
    bool vram_past_budget = false; // EngineOptions::vram_past_budget
    bool expert_cache = true;
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
    // Prefill CPU assist (design §19.3.1 P7): prefill calls wider than a decode round and narrower
    // than kAssistMaxColumns hand up to this many of each layer's thinnest misses to the CPU (same
    // divisor). 0 disables.
    std::uint32_t cpu_assist_jobs    = 256;
    // Prefill expert streaming (design §19.3.8 F2, F3): chunks of at least 256 columns copy each
    // layer's non-resident experts ahead by DMA into frames lent for the prompt.
    bool prefill_stream              = true;
    // Prefix cache (design §19.3.1): requests resume from and publish to a shared block tree with
    // Host-born state snapshots in a pinned slab pool of prefix_host_bytes.
    bool prefix_cache                = false;
    std::uint64_t prefix_host_bytes  = 0;
    runtime::prefix_cache::TapPlannerConfig prefix_taps;
    runtime::prefix_cache::CacheCostModel prefix_cost;
    // What a cut that ends a layer walk's span costs (the next span streams the experts again).
    double prefix_span_seconds = 0.0;
    // The longest predicted wait for a prefilling sibling's snapshot a fresh request accepts instead
    // of prefilling the shared prefix itself (0: no coalescing).
    double prefix_coalesce_wait_seconds = 0.0;
    // The product's control over the shutdown save of a persistent prefix cache.
    PrefixCacheSaveControl prefix_save;
    DiagnosticObserver diagnostics;
    // Internal (A/B measurement and tests): the runtime VRAM monitor that shrinks and grows the
    // expert cache as other programs take and release device memory (design §19.3.7), and how long
    // free memory must stay above the cache's target before it grows.
    bool vram_monitor              = true;
    double vram_grow_delay_seconds = 30.0;
    // Internal (measurement tools and tests): when set, every round's routed experts and every CUDA
    // graph executable's device memory are appended to this file for the expert-cache replay and
    // the graph-memory measurement (program/route_trace.h). Changes no result.
    std::filesystem::path route_trace;
};

class Program {
public:
    Program(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options);
    ~Program() noexcept;

    // The device plan of a Program with these options over a model with this configuration and
    // public token count; validates the options.
    [[nodiscard]] static ProgramDevicePlan plan_device(const ProgramOptions& options, const Config& config,
                                                       std::uint32_t public_tokens);
    [[nodiscard]] const ProgramDevicePlan& device_plan() const noexcept;
    // How the expert frames were sized at construction.
    [[nodiscard]] const VramSizing& vram_sizing() const noexcept;

    // Expert-cache resizing outside rounds (design §19.3.7). The VRAM monitor calls the waker from
    // its own thread when the cache should resize; the engine then calls maintain() from its
    // worker, between rounds and under its execution lock. An empty waker detaches the engine.
    void set_maintenance_waker(std::function<void()> waker);
    void maintain();
    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;

    [[nodiscard]] RequestBasePlan plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] bool isolated_request_feasible(const RequestBasePlan& base) const noexcept;

    // ---- admission: hybrid_sources quotes, start_binding stages, poll_context activates ----
    // The prefix cache's quotes in preference order: its chosen path and snapshot, then a root start
    // (a root start only, without a prefix cache).
    [[nodiscard]] std::vector<SourceCandidate> hybrid_sources(const RequestBasePlan& base,
                                                              std::uint32_t maximum_frontier);
    // Reserves the request's whole KV extent on `lane` and stages its prefix restore; a shortage
    // reports the pages missing. Qwen4Exp never pauses, so `resume` must be null.
    [[nodiscard]] runtime::ResourceReservation start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                                             const SourceCandidate& source,
                                                             ResumeState* resume           = nullptr,
                                                             ExecutionUnitKind resume_kind = ExecutionUnitKind::Decode,
                                                             std::uint32_t resume_tokens   = 1);
    [[nodiscard]] ContextProgress poll_context(runtime::CancellationFlagView cancellation);
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] bool context_blocks(SequenceHandle sequence) const noexcept;
    // Evicts unpinned cached Device blocks until the shortage is covered; whether anything was freed.
    [[nodiscard]] bool hybrid_reclaim(runtime::ContextResourceUsage shortage);
    // Copies the blocked FIFO head's Host-only blocks into spare Device cache while it waits: the
    // blocks started (nullopt while a transaction or an earlier prefetch is in flight).
    [[nodiscard]] std::optional<std::uint32_t> hybrid_prefetch(const RequestBasePlan& base);
    // The Device pages a prefetch could fill now (free plus Host-backed cached).
    [[nodiscard]] std::uint32_t hybrid_prefetch_room() const noexcept;
    [[nodiscard]] HybridPrefixCacheStats hybrid_stats() const noexcept;

    // ---- execution units: the whole extent is reserved at admission, so every unit fits ----
    [[nodiscard]] runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit>) {
        return {.reserved = true};
    }
    void release_units(std::span<const SequenceHandle>) noexcept {}
    [[nodiscard]] bool recovery_pending(SequenceHandle) const noexcept { return false; }

    // ---- pause, replay, checkpoints and captures: never offered by Qwen4Exp ----
    [[nodiscard]] bool start_pause(SequenceHandle, bool, runtime::ExecutionTiming* = nullptr) { return false; }
    [[nodiscard]] ReplayProgress advance_replay(SequenceHandle, runtime::ExecutionTiming* = nullptr);
    [[nodiscard]] bool revoke_snapshot(ResumeState&) noexcept { return false; }
    [[nodiscard]] runtime::ContextResourceUsage snapshot_resources(const ResumeState&) const { return {}; }
    [[nodiscard]] std::optional<std::size_t> pause_host_bytes(SequenceHandle) const { return std::nullopt; }
    [[nodiscard]] std::size_t release_redundant_host(std::span<const CheckpointHandle>,
                                                     std::optional<SequenceHandle> = std::nullopt) {
        return 0;
    }
    [[nodiscard]] bool valid_checkpoint(CheckpointHandle) const noexcept { return false; }
    [[nodiscard]] ContextReclaimPlan plan_reclaim(std::span<const CheckpointHandle>, std::span<const CheckpointHandle>,
                                                  runtime::ContextResourceUsage) const {
        return {};
    }
    [[nodiscard]] bool reclaim_capture_reservation(runtime::ContextResourceUsage) { return false; }
    [[nodiscard]] bool start_capture(SequenceHandle) { return false; }
    [[nodiscard]] bool capture_is_input(SequenceHandle) const { return false; }
    [[nodiscard]] std::optional<CapturePreparation> prepare_capture(SequenceHandle) { return std::nullopt; }
    void skip_capture(SequenceHandle) {}

    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences, std::span<const TokenId> row_major_tokens,
                         std::uint32_t row_stride,
                         std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                         runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CommitResult commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
                                      runtime::CommitObservation observation  = runtime::CommitObservation::AllRows,
                                      runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;

    void fail_all_cleanup() noexcept;
    // Releases every lane, saves the expert cache's state and then the prefix cache's Host tier
    // when a file is attached.
    void shutdown_cleanup() noexcept;
    // Loads the Host tier from `path` into the empty prefix cache and keeps `path` for the shutdown
    // save. `fingerprint` names everything the saved bytes depend on besides their geometry.
    [[nodiscard]] PrefixCachePersistence attach_prefix_cache_file(const std::filesystem::path& path,
                                                                  std::string fingerprint,
                                                                  const StartupObserver& observer);
    // The shutdown save's result; absent before shutdown or without a file.
    [[nodiscard]] std::optional<PrefixCachePersistence> prefix_shutdown_save() const;

    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept {}

private:
    std::unique_ptr<detail::ProgramImpl> impl_;
};

// The common EngineCore is instantiated once for this model.
struct RuntimeTypes {
    using Frontend          = qwen4_exp::Frontend;
    using PreparedPrompt    = qwen4_exp::PreparedPrompt;
    using NgramArchive      = qwen4_exp::NgramArchive;
    using OutputSession     = qwen4_exp::OutputSession;
    using PublishedOutput   = qwen4_exp::PublishedOutput;
    using RequestBasePlan   = qwen4_exp::RequestBasePlan;
    using SequenceHandle    = qwen4_exp::SequenceHandle;
    using CheckpointHandle  = qwen4_exp::CheckpointHandle;
    using CheckpointSummary = qwen4_exp::CheckpointSummary;
    using SourceCandidate   = qwen4_exp::SourceCandidate;
    using ResumeState       = qwen4_exp::ResumeState;
    using ExecutionUnit     = qwen4_exp::ExecutionUnit;
    using ExecutionUnitKind = qwen4_exp::ExecutionUnitKind;
    using ContextProgress   = qwen4_exp::ContextProgress;
    using PendingBatch      = qwen4_exp::PendingBatch;
    using PrefillProgress   = qwen4_exp::PrefillProgress;
    using ReplayProgress    = qwen4_exp::ReplayProgress;
    using CommitResult      = qwen4_exp::CommitResult;
    using DiscardResult     = qwen4_exp::DiscardResult;
    using FinishResult      = qwen4_exp::FinishResult;
    using AbortResult       = qwen4_exp::AbortResult;
    using Program           = qwen4_exp::Program;
    using CacheSessionKey   = qwen3_5::PreparedSessionKey;
};

} // namespace ninfer::models::qwen4_exp
