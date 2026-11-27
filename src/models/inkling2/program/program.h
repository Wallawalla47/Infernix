#pragma once

// Inkling 2's Program (design docs/maintainer/inkling2_mini-design.md §7-§8): the lanes, their state and KV, the
// calls of the Forward (execution/forward.h), sampling and the commit of each round, behind the common EngineCore.
//
// The lanes share one KV page pool (--kv-capacity), from which each takes the pages of its next execution unit
// (reserve_units); a shortage the prefix cache cannot cover pauses a resident (start_pause), whose resume binds the
// deepest cached state of its ledger and replays the rest exactly (advance_replay: the prompt route below the prompt's
// end, the decode route past it, as they first ran). The n-gram archive and next-turn prefill are off for this
// architecture. Prompts prefill in chunks of one lane; decode rounds are B lanes x 1 column as CUDA graphs per (B, T)
// captured at startup, and decoding lanes draft with DSpark or a DFlash2 companion. A constrained row (constrained-decoding.md §4.4) samples under its matcher's masks, filled while its
// call's forward runs (a forward-only graph, then the sample); a token-constrained row (TokenConstraint, /v1/decide)
// decodes plainly, its logits masked to its step's tokens before the sample (ops::constrain_logits).

#include "core/startup.h"
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
struct QuoteImpl;
struct ResumeStateImpl;
} // namespace detail

inline constexpr std::size_t kMaximumConcurrency = 8;
// The CUDA graphs' allowance, the Qwen3.5 family's for its column-invariant decode (planning/startup.cpp
// kGraphDriverAllowance, kGraphExecutableAllowanceColumnInvariant): 16 MiB for the driver and 3 MiB an executable.
// Measured on Inkling 2 (RTX 5090, 2026-11-25, 240,000 tokens x 4 lanes, INT8 KV, DSpark, vision offloaded): each of 40
// executables took no device memory the driver did not already hold, and the whole Program about 30 MiB more than its
// plan without the allowance; with --vram-past-budget a pool sized without it ran out of memory at startup.
inline constexpr std::uint64_t kGraphDriverBytes     = 16ULL << 20;
inline constexpr std::uint64_t kGraphExecutableBytes = 3ULL << 20;
// The widest prefill chunk: the forward's call capacity, bounded by the routed experts' dispatch.
inline constexpr std::int32_t kMaxPrefillChunk = 8191;
// The default chunk: the fastest measured (every chunk gives the same bits).
inline constexpr std::uint32_t kDefaultPrefillChunk = 4096;

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

// A root start, or (quote set) a prefix-cache snapshot the binding restores.
struct SourceCandidate {
    std::uint32_t reused_tokens    = 0;
    std::shared_ptr<const detail::QuoteImpl> quote;
    PrefixReusePath reuse_path     = PrefixReusePath::Root;
    // The prompt prefix held as cached full blocks, reusable or not (diagnostics).
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

// A paused request (Program::start_pause): its ledger and request state; its device state went to the prefix cache.
class ResumeState {
public:
    ResumeState(ResumeState&&) noexcept;
    ResumeState& operator=(ResumeState&&) noexcept;
    ResumeState(const ResumeState&)            = delete;
    ResumeState& operator=(const ResumeState&) = delete;
    ~ResumeState();
    // The paused frontier: the positions whose state the lane held.
    [[nodiscard]] std::uint32_t frontier() const noexcept { return frontier_; }

private:
    ResumeState(std::unique_ptr<detail::ResumeStateImpl> impl, std::uint32_t frontier) noexcept;
    std::unique_ptr<detail::ResumeStateImpl> impl_;
    std::uint32_t frontier_ = 0;

    friend struct detail::ContractAccess;
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

// Several lanes' prefill units in one prompt call (Program::advance_prefill_group): each lane's progress in the call's
// order, none with a pending batch of its own; the lanes that completed their prompt sample together, `pending`'s rows
// those lanes in the call's order (`completed`: their indices in it).
struct PrefillGroupProgress {
    std::vector<PrefillProgress> lanes;
    std::optional<PendingBatch> pending;
    std::vector<std::uint32_t> completed;
    runtime::ExecutionTiming timing;
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
    // --lossless (LosslessForward, lossless.h): causal scoring scores each window in one pass; generation runs every
    // prefill chunk and decode call of a lane through the cache form (eager calls, one lane at a time; the K and V
    // pages and rings hold BF16 values, V too). `reference` (causal scoring only): the design reference
    // (--reference: V stored as the bf16 profile's FP16).
    bool lossless  = false;
    bool reference = false;
    // Columns a prefill call takes of one lane (1..kMaxPrefillChunk); the forward's call capacity follows it.
    std::uint32_t prefill_chunk = kDefaultPrefillChunk;
    bool use_cuda_graph         = true;
    // --spec dspark (the drafter bound in the Parameters): a decode round of one row drafts draft_tokens (1-7) and
    // verifies all of them, or (draft_tokens_auto) the count their calibrated probabilities make worth it.
    bool dspark                 = false;
    // --spec dflash2 --draft-model (the companion's drafter bound in the Parameters; execution/dflash2.h): the same
    // rounds, each drafting its whole block of 7 (DSpark's KV-page planes and tables are not built).
    bool dflash2                = false;
    std::uint32_t draft_tokens  = 7;
    bool draft_tokens_auto      = true;
    // --a16-activations: the routed experts on 16-bit activations (w4a4_sparse_moe.h's ExpertActivation::A16).
    // --lm-head-draft: the drafts read the shortlist head (DFlash2's workspace holds its logits, not the vocabulary's).
    bool shortlist_head         = false;
    bool expert_a16             = false;
    // --a8-activations (with the 8-bit weights it implies): 8-bit activations in the attention projections, the head
    // and the NVFP4 layers' routed experts, in every call (Forward's a8; the experts' ExpertActivation::A8).
    bool a8                     = false;
    // --lora: the adapter's padded rank (0: none), which sizes its per-call workspace on the route the Program runs.
    std::uint32_t lora_rank     = 0;
    // --vision (the tower bound in the Parameters): the tower's buffers for images of up to vision_max_cells cells.
    bool vision                    = false;
    std::uint32_t vision_max_cells = 0;
    // --vision-offload on: the tower's pinned groups (the model's vision_host(), or the load plan's when planning),
    // whose device staging the plan counts; the Program's tower streams through it.
    const VisionHostWeights* vision_host = nullptr;
    // --kv-dtype: the full visits' pools; --kv-window-dtype (window_codec): the sliding windows' rings.
    ops::inkling2::KvCodec kv_codec     = ops::inkling2::KvCodec::Bf16;
    ops::inkling2::KvCodec window_codec = ops::inkling2::KvCodec::Bf16;
    // The full visits' pool (--kv-capacity), shared by the lanes: at least one lane's whole context
    // (pages_per_lane(max_context)); pages no lane holds are the prefix cache's blocks. 0: one lane's context.
    std::uint32_t kv_pages = 0;
    // The prefix cache (generation only; prefix_cache.h): its pinned Host tier, which holds every snapshot, and the
    // tap planner's limits. Off: no request reuses or publishes anything.
    bool prefix_cache                = false;
    std::uint64_t prefix_host_bytes  = 0;
    // An automatic Host tier's growth room (--ram-resize auto): the most it may grow to once RAM frees (0: none; the
    // start page-locks prefix_host_bytes of it).
    std::uint64_t prefix_host_ceiling_bytes = 0;
    std::uint32_t prefix_max_new_taps   = 8;
    std::uint32_t prefix_ladder_tokens  = 4096;
    std::uint32_t prefix_min_gap_tokens = 1024;
    // In-flight coalescing (hybrid-prefix-cache-spec §12.2): the longest predicted wait for a prefilling sibling's
    // snapshot a fresh request takes instead of prefilling the shared prefix again (0: never waits).
    double prefix_coalesce_wait_seconds = 0.0;
    // The product's control over the shutdown save (--prefix-cache-file; a second Ctrl+C abandons it).
    PrefixCacheSaveControl prefix_save;
    // The SSD tier below the Host tier (0: none): its size, directory, write budget (0: none) and kept file.
    std::uint64_t prefix_ssd_bytes = 0;
    std::filesystem::path prefix_ssd_directory;
    std::uint64_t prefix_ssd_write_bytes_per_hour = 0;
    std::filesystem::path prefix_ssd_kept_file;
    DiagnosticObserver diagnostics;
};

// A lane's block-table row: the pages of max_context positions and of a call's write lead past them.
[[nodiscard]] std::uint32_t pages_per_lane(std::uint32_t max_context) noexcept;

// --kv-dtype's forms of the full visits' pools: the KvCodec of a storage (none: not built for Inkling 2), and back.
[[nodiscard]] std::optional<ops::inkling2::KvCodec> kv_codec(KvCacheStorage storage);
// The sliding windows' storage (design §7.6): --kv-window-dtype's `window` (bf16, bf16p or int8; none for a storage that
// is not a window codec), or by default (owner 2026-11-24) BF16 words under a bf16 pool, bf16p under bf16p and int8
// under every other codec.
[[nodiscard]] std::optional<ops::inkling2::KvCodec> window_codec(ops::inkling2::KvCodec pool,
                                                                 std::optional<KvCacheStorage> window);
[[nodiscard]] KvCacheStorage kv_storage(ops::inkling2::KvCodec codec);

// Device memory the Program allocates, from the options and the configuration alone.
struct ProgramDevicePlan {
    std::uint32_t kv_pages      = 0; // the pool's pages (scratch page included): max(options.kv_pages,
                                     // pages_per_lane(max_context)) + 1
    std::uint64_t page_bytes    = 0; // one page of every full visit's K and V planes (side records included)
    std::uint64_t kv            = 0; // the full visits' K and V pools and the block table
    std::uint64_t state         = 0; // the lanes' state images
    std::uint64_t workspace     = 0; // the Forward's call buffers and the sampling workspace
    std::uint64_t io            = 0; // per-call inputs, logits rows and sampling arrays
    std::uint32_t graph_bound   = 0;
    // The CUDA graph executables the Program may instantiate (the decode graphs at startup; their forward-only forms,
    // DSpark's verify and draft graphs as rounds first need them), at the Qwen3.5 family's allowance (kGraphDriverBytes
    // and kGraphExecutableBytes each).
    std::uint64_t graphs        = 0;
    [[nodiscard]] std::uint64_t fixed_bytes() const noexcept { return kv + state + workspace + io + graphs; }
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

    // ---- execution units: each lane takes the pages its next unit writes; a shortage the prefix cache cannot cover
    // is reported, and the Engine reclaims (hybrid_reclaim) or pauses a resident (Scheduler::preemption_victim) ----
    [[nodiscard]] runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit> units);
    void release_units(std::span<const SequenceHandle>) noexcept {}
    [[nodiscard]] bool recovery_pending(SequenceHandle) const noexcept { return false; }
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* failed_timing = nullptr,
                                                  runtime::TokenMaskProvider* masks = nullptr);
    // The Engine's prefill companions (engine-architecture "Prefill order"): whether `companion`'s next prefill unit may
    // join the prompt call of `units` (the lanes already in it, the first the primary, each prefilling): the unit
    // completes its prompt within kPrefillCompanionTokens, and the call stays within the prefill chunk. Never on
    // --lossless (its calls are one lane's).
    static constexpr std::uint32_t kPrefillCompanionTokens = 64;
    [[nodiscard]] bool prefill_joins(std::span<const SequenceHandle> units, SequenceHandle companion) const;
    // Each sequence's next prefill unit (as advance_prefill would run it alone) in one prompt call, `sequences[0]` the
    // primary and the rest admitted by prefill_joins: every column is computed from its own inputs on the prompt
    // route, so each lane's bits are its unit alone's. `masks` rows are `sequences` positions.
    [[nodiscard]] PrefillGroupProgress advance_prefill_group(std::span<const SequenceHandle> sequences,
                                                             runtime::ExecutionTiming* failed_timing = nullptr,
                                                             runtime::TokenMaskProvider* masks       = nullptr);
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
    // What the start planned for the Program and the Program did not allocate (MemoryUsage::Device::reserved).
    void set_reserved_bytes(std::uint64_t bytes) noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept {}

    // The idle Engine's maintenance (hybrid_storage_due, or the RAM monitor's waker): the SSD tier's write-backs of what
    // the last requests left in the Host tier, and the Host tier's resize under RAM pressure (start_ram_resize).
    void maintain();
    // Between rounds while requests run, when the waker asked (runtime/contract/program.h): the Host tier's resize
    // advances or starts.
    void maintain_busy();
    // The RAM monitor's waker (start_ram_resize): the Engine then calls maintain() or maintain_busy() on its worker.
    void set_maintenance_waker(std::function<void()> waker);
    // RAM pressure (EngineOptions::ram_resize, runtime::expert_offload::RamMonitor): starts the monitor under which
    // maintain() and maintain_busy() shrink the prefix cache's Host tier while the OS runs short (less than
    // max(reserve_bytes / 2, 2 GiB) available) and grow it back, an automatic tier up to its growth room
    // (ProgramOptions::prefix_host_ceiling_bytes), once more than the reserve plus a step stays free. `trim_host_tier`
    // (an automatic Host tier): the first maintenance after the start gives back what keeps less than the reserve
    // available. Without a Host tier, nothing.
    void start_ram_resize(std::uint64_t reserve_bytes, bool trim_host_tier);
    [[nodiscard]] std::uint64_t engine_thread_cpus() const noexcept { return 0; }
    // The SSD tier (ProgramOptions::prefix_ssd_bytes): landed transfers applied and write-backs started; a source a quote
    // is reading from the SSD; whether a write-back is due, or when one may be.
    [[nodiscard]] bool hybrid_poll_storage();
    [[nodiscard]] bool hybrid_source_pending() const noexcept;
    [[nodiscard]] bool hybrid_storage_due() const noexcept;
    [[nodiscard]] std::optional<std::chrono::steady_clock::time_point> hybrid_storage_retry_at() const noexcept;
    [[nodiscard]] std::uint32_t hybrid_resume_grid() const noexcept { return 16; }
    // --prefix-cache-file: restores the Host tier from `path` (a file written for this fingerprint), and saves it there
    // at shutdown (shutdown_cleanup) and whenever save_prefix_cache_now runs (the periodic save).
    [[nodiscard]] runtime::prefix_cache::PersistResult attach_prefix_cache_file(const std::filesystem::path& path,
                                                                                std::string fingerprint,
                                                                                const StartupObserver& observer);
    [[nodiscard]] std::optional<runtime::prefix_cache::PersistResult> save_prefix_cache_now(const CancellationView& abandoned);
    // The shutdown save's result (empty: none ran).
    [[nodiscard]] std::optional<runtime::prefix_cache::PersistResult> prefix_shutdown_save() const;
    [[nodiscard]] std::optional<ExpertCacheTotals> expert_cache_totals() const noexcept { return std::nullopt; }
    [[nodiscard]] HybridPrefixCacheStats hybrid_stats() const noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    // No prefetch or tool-call hold is built.
    [[nodiscard]] std::optional<std::uint32_t> hybrid_prefetch(const RequestBasePlan&, bool = false) {
        return std::nullopt;
    }
    [[nodiscard]] std::uint32_t hybrid_prefetch_room() const noexcept { return 0; }
    // The Engine's reclaim for a shortage: the cache's unpinned Device blocks go (and those only in-flight Host writes
    // hold, once they land) until it is covered; whether any page was freed.
    [[nodiscard]] bool hybrid_reclaim(runtime::ContextResourceUsage shortage);
    [[nodiscard]] std::uint64_t hybrid_cache_epoch() const noexcept;
    [[nodiscard]] bool hybrid_holds_stale() const noexcept { return false; }
    // The snapshot each queued request would resume from now is held in queue order (hybrid-prefix-cache-spec §9.6): a
    // paused request's over its ledger (its pause's endpoint), a fresh one's as its admission would choose it.
    void hybrid_hold_queue(std::span<const RequestBasePlan* const> queue, std::span<const ResumeState* const> resumes);
    void hybrid_hold_continuation(std::chrono::steady_clock::time_point) {}
    // ---- pause and replay ----
    // The tokens a pause of `sequence` now would recompute on its resume: none where the pause captures the lane's
    // state at its frontier, else the frontier past the deepest cached state of its ledger.
    [[nodiscard]] std::uint32_t pause_replay_tokens(SequenceHandle sequence) const noexcept;
    // Pauses a resident at its committed boundary: its blocks and its state at the frontier go to the prefix cache
    // (decode-computed ones under a key only its resume names), the lane is released, and a Pause transaction hands the
    // Engine the ResumeState. False while a binding or a pending batch is open.
    [[nodiscard]] bool start_pause(SequenceHandle sequence, runtime::ExecutionTiming* = nullptr);
    // One call of a resumed lane's ledger toward its paused frontier, without sampling: the prompt route below the
    // prompt's end, decode-route calls of up to 16 columns past it (the bits the positions were first computed with).
    [[nodiscard]] ReplayProgress advance_replay(SequenceHandle sequence, runtime::ExecutionTiming* = nullptr);

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
