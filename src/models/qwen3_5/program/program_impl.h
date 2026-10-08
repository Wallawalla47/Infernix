#pragma once
#include "models/qwen3_5/program/internal.h"

#include "core/arena.h"
#include "core/gdn_replay_records.h"
#include "core/host_kv_arena.h"
#include "infernix/ops/gdn_replay.h"
#include "infernix/ops/sampling.h"
#include "infernix/ops/speculative_tree.h"
#include "core/decode_graph.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"

#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/speculative/tree_width_controller.h"
#include "models/qwen3_5/program/storage/draft_context.h"
#include "models/qwen3_5/program/storage/host_kv_store.h"
#include "models/qwen3_5/program/storage/kv_address_space.h"
#include "models/qwen3_5/program/storage/state_store.h"
#include "models/qwen3_5/program/prefix_identity.h"
#include "core/host_context_arena.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/program/vision_prefill.h"
#include "models/qwen3_5/program/ngram_proposer.h"
#include "models/qwen3_5/program/prefix/hybrid_cache.h"
#include "runtime/prefix_cache/cost.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <algorithm>
#include <cstdint>
#include <array>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace infernix::models::qwen3_5::detail {
using execution::dimension;
using PreparedPromptData = qwen3_5::PreparedPromptData;

struct PreparedCaptureBacking {
    PrefixShortlistDigests digests;
    std::vector<TokenId> ledger;
    qwen3_5::detail::ResidentPrefixIdentity prefix_identity;
};

struct CaptureGroup {
    std::shared_ptr<const PreparedCaptureBacking> identity;
    PrefixShortlistKey key;
    std::uint32_t frontier = 0;
    std::vector<runtime::CheckpointRole> roles;
};

enum class MtpBridgeMode : std::uint8_t {
    None,
    BeforeSuffix,
    AfterExactHit,
};

struct PreparedNgramIndex {
    std::unique_ptr<NgramProposer> index;
};

// A TokenConstraint compiled for ops::constrain_logits: its distinct multi-token sets (written
// into the lane's set columns when the request binds) and each step's descriptor relative to
// the lane (a set index, or the forced-token encoding -2 - token).
struct RequestConstraintPlan {
    std::vector<std::vector<TokenId>> sets;
    std::vector<std::int32_t> step_descriptors;
    // Unique per planned request, so a lane's uploaded sets are never mistaken for a later
    // request's.
    std::uint64_t serial = 0;

    [[nodiscard]] bool empty() const noexcept { return step_descriptors.empty(); }
};

struct RequestBasePlanImpl {
    std::shared_ptr<const PreparedPromptData> prompt;
    runtime::RequestPlanSummary summary;
    qwen3_5::PreparedContextCache context_cache;
    ops::SamplingConfig sampling;
    // Tokens whose prompt-frontier log-probabilities complete the request's prefill.
    std::vector<TokenId> readout_tokens;
    RequestConstraintPlan constraint;
    std::shared_ptr<const VisionControlPlan> vision_control_plan;
    std::vector<CaptureGroup> capture_groups;
    std::shared_ptr<const PreparedCaptureBacking> capture_backing;
    PrefixShortlistDigests prefix_digests;
    std::uint32_t prefix_identity_tag = 0;
    bool allow_prefix_reuse           = false;
    // The prepared ngram index moves out of the immutable prompt into this one-shot slot; the
    // request's single fresh binding takes it (a resumed binding keeps its saved proposer).
    std::shared_ptr<PreparedNgramIndex> ngram_index;

    [[nodiscard]] bool accepts_capture(std::uint32_t frontier) const noexcept;
    [[nodiscard]] CaptureGroup capture_group(std::uint32_t frontier) const;
};
enum class PendingKind : std::uint8_t { None, Begin, Ordinary, Speculative };

struct PendingCandidate {
    PendingKind kind            = PendingKind::None;
    std::uint32_t base_E        = 0;
    std::uint32_t base_S        = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t produced      = 0;
    // Draft columns the speculative round verified: its egress/frame row stride is this + 1 and
    // its ReplaySSM records use the matching record view.
    std::uint32_t verify_drafts = 0;
};
enum class Lifecycle : std::uint8_t {
    Empty,
    Binding,
    Prefilling,
    Replaying,
    Active,
    Pending,
    Finishable,
    Pausing
};

struct UnitDemand {
    ExecutionUnitKind kind         = ExecutionUnitKind::Decode;
    std::uint32_t tokens           = 0;
    std::uint32_t main_frontier    = 0;
    std::uint32_t backend_frontier = 0;
};

struct RecoveryPermit {
    // The active KV addresses hold the physical reservation through this coverage.
    // These frontiers distinguish reconstructed work from committed new progress.
    UnitDemand coverage;
    std::uint32_t frontier    = 0;
    std::size_t ledger_tokens = 0;
};

struct SequenceKVBundle {
    KVAddressSpaceHandle text;
    std::optional<KVAddressSpaceHandle> backend;
};

// One mutable KV directory shared by the internal recovery points of a private history.
// Independent public/branch views own a different history and share only physical prefix pages.
struct KVHistory {
    ProgramImpl* owner = nullptr;
    KVAddressSpaceHandle text;
    std::optional<KVAddressSpaceHandle> backend;
    ~KVHistory();
};

struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0;
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0;
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;
    DecodeGraphProfile& select(std::uint32_t batch_size, std::uint32_t frontier);
    DecodeGraphExecutable& install(DecodeGraphProfile& profile);
};

// A Forward stage publishes these IDs before target execution finishes. Program owns the
// pinned storage and external event together; both outlive the graphs that reference them.
struct DFlashDraftHandoff {
    PinnedHostBuffer ids;
    CudaCompletionEvent ready;

    DFlashDraftHandoff(const DeviceContext& device, std::size_t count)
        : ids(count * sizeof(TokenId)), ready(device) {}

    [[nodiscard]] std::span<TokenId> tokens() const noexcept {
        return {static_cast<TokenId*>(ids.data()), ids.size() / sizeof(TokenId)};
    }
};

// A speculative round family (see SpeculativeRoundShape) and its captured Forward/Finish graphs.
struct SpeculativeRoundFamily {
    SpeculativeRoundShape shape;
    DecodeGraphFamily forward;
    DecodeGraphFamily finish;
};

// ReplaySSM records viewed densely at a width narrower than the frame's, with the fold bound to
// that view. Rounds verified at the narrower width record and fold only through it.
struct NarrowReplayView {
    GdnReplayRecords records;
    ops::GdnReplayFoldPlan fold;
};
struct SequenceState {
    std::shared_ptr<KVHistory> kv;
    ActiveStateBinding state;
    Tensor tail_hidden;
    std::uint32_t lane = 0;

    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::vector<TokenId> ledger;
    qwen3_5::detail::ResidentPrefixIdentity prefix_identity;
    qwen3_5::detail::PrefixShortlistDigests prefix_digests;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, qwen3_5::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid        = false;
    bool endpoint_valid           = false;
};

struct CaptureReservation {
    ProgramImpl* owner = nullptr;
    std::vector<std::pair<CheckpointHandle, runtime::CheckpointRole>> points;
    StateImageHandle state;
    std::optional<HostContextAllocation> host;
    std::optional<DeviceKVPageReservation> main_tail;
    std::optional<DeviceKVPageReservation> backend_tail;
    std::uint32_t frontier = 0;
    ~CaptureReservation();
};

struct RequestControl {
    std::unique_ptr<NgramProposer> ngram;
    std::shared_ptr<const NgramSnapshot> ngram_snapshot;
    std::uint64_t ngram_copy_source = 0;
    std::uint32_t ngram_copy_offset = 0;
    std::size_t ngram_copy_ledger   = 0;
    std::size_t ngram_indexed       = 0;
    Lifecycle lifecycle             = Lifecycle::Empty;
    PendingCandidate pending;
    ops::SamplingConfig sampling_host;
    GenerationTimings timings;
    SpeculativeStats speculative_stats;
    std::shared_ptr<const RequestBasePlanImpl> base;
    std::optional<UnitDemand> permit;
    std::optional<RecoveryPermit> recovery;
    std::unique_ptr<CaptureReservation> capture_reservation;
    std::uint32_t replay_target = 0;
    std::uint32_t replay_cursor = 0;
    Lifecycle resume_lifecycle  = Lifecycle::Empty;
    bool publish_continuation   = true;
    std::vector<CaptureGroup> capture_groups;
    std::size_t next_capture = 0;
    bool capture_pending     = false;
    // Constrained requests: the draws of the pending candidate (one per produced token, in
    // order) and the committed trace they move into.
    std::vector<ConstrainedDraw> constraint_round;
    std::vector<ConstrainedDraw> constraint_trace;

    struct Prefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<execution::VisionPrefillSession> vision;
        std::uint32_t base               = 0;
        std::uint32_t cursor             = 0;
        std::uint32_t prompt_tokens      = 0;
        std::uint32_t initial_mtp_extent = 0;
        double elapsed_seconds           = 0.0;
        double retired_vision_seconds    = 0.0;
        bool prepare_mtp                 = false;
        PrefixReusePath reuse            = PrefixReusePath::Root;
        MtpBridgeMode mtp_bridge         = MtpBridgeMode::None;
    };

    std::optional<Prefill> prefill;
    std::optional<Prefill> replay;
};

struct CheckpointState {
    std::shared_ptr<KVHistory> kv;
    runtime::CheckpointRole role = runtime::CheckpointRole::Continuation;
    StateImageHandle state;
    std::shared_ptr<const PreparedCaptureBacking> identity;
    PrefixShortlistKey key;
    std::uint32_t frontier         = 0;
    std::uint32_t backend_frontier = 0;
    std::int32_t rope_delta        = 0;
    bool tail_hidden_valid         = false;
};

struct CheckpointSlot {
    std::optional<CheckpointState> value;
    std::uint64_t generation = 1;
    std::uint32_t pins       = 0;
    bool reserved            = false;
};

struct ResumeStateImpl {
    ProgramImpl* owner = nullptr;
    std::optional<CheckpointHandle> snapshot;
    SequenceState sequence;
    RequestControl control;
    std::uint32_t frontier = 0;
    ~ResumeStateImpl();
};

// Hybrid prefix cache admission decision (docs/maintainer/hybrid-prefix-cache-spec.md §6): the
// snapshot to resume from (invalid: a root start) and the facts the binding logs.
struct HybridQuoteImpl {
    runtime::prefix_cache::SnapshotRef snapshot;
    std::uint32_t reuse_frontier = 0;
    // Longest prompt prefix held as cached full blocks, reusable or not.
    std::uint32_t cached_prefix_tokens = 0;
};

// A prefill tap whose StateImage is captured but whose snapshot waits for the blocks it anchors
// on: a frontier inside a block needs that block complete, and an MTP backend trails the text
// frontier by one token, so even a page-aligned frontier waits for its last block's backend page.
struct HybridPendingTap {
    std::uint32_t frontier = 0;
    StateImageHandle image;
    std::uint32_t slot = 0; // staging device snapshot slot
    bool boundary      = false;
};

// Per-lane hybrid bookkeeping for the active sequence.
struct HybridLaneState {
    bool active  = false;
    bool publish = false;
    // Root path of full blocks this sequence pins, in prompt order. Blocks past the reuse
    // frontier are appended as the sequence commits them.
    std::vector<runtime::prefix_cache::NodeRef> path;
    std::uint64_t path_hash = runtime::prefix_cache::kRootLookupHash;
    // Extra key of every full prompt block (Vision identity), empty for text-only prompts.
    std::vector<std::uint64_t> prompt_extras;
    // Extra key of blocks after the last full prompt block: every Vision item precedes them.
    std::uint64_t trailing_extra = 0;
    std::vector<runtime::prefix_cache::PlannedTap> taps;
    std::size_t next_tap = 0;
    std::vector<runtime::prefix_cache::TapExclusion> exclusions;
    std::vector<HybridPendingTap> pending;
    // Most recent snapshot frontier this sequence reused or captured.
    std::uint32_t last_capture = 0;
    // Deepest snapshot frontier known on this path (reused or created by this sequence).
    std::uint32_t deepest_snapshot = 0;
    // The snapshot this sequence resumed from (invalid: root). Once the sequence publishes a
    // deeper snapshot, its lineage resumes from that one and this one is superseded.
    runtime::prefix_cache::SnapshotRef resume_snapshot;
    std::uint32_t resume_frontier = 0;
    // The newest Tap (not Boundary) this sequence published. A deeper tap of the same prompt
    // supersedes it: the lineage resumes from the deeper one, and it only serves a request
    // diverging between them. The endpoint does not, since a next turn whose template re-renders
    // the reply resumes from the prompt-end tap.
    runtime::prefix_cache::SnapshotRef tap_snapshot;
    std::uint32_t tap_frontier = 0;
    // The Host restore this sequence was admitted from (0 without one). Its first prefill pass
    // queues behind the restore's per-layer events; releasing the lane queues behind the whole
    // restore if it may still be landing.
    std::uint64_t restore_ticket = 0;
    bool restore_layers_pending  = false;
};

class ProgramImpl {
public:
    ProgramImpl(const execution::Parameters&, const SequencePlanImpl&, DeviceContext&,
                const StartupObserver&);
    ~ProgramImpl() noexcept;
    [[nodiscard]] RequestBasePlan plan_request(PreparedPromptData&&,
                                               const runtime::ResolvedExecutionOptions&);
    [[nodiscard]] ScoreResult causal_score(PreparedPromptData&&, std::uint32_t first_target,
                                           const ScoreOptions& options);
    [[nodiscard]] std::optional<SourceCandidate>
    inspect_source(const RequestBasePlan&, std::optional<CheckpointHandle>, bool = false,
                   std::span<const CheckpointHandle> = {},
                   std::span<const CheckpointHandle> = {}) const;
    [[nodiscard]] PrefixShortlistKey checkpoint_key(CheckpointHandle, std::uint32_t frontier) const;
    [[nodiscard]] runtime::ContextResourceUsage
        checkpoint_footprint(std::span<const CheckpointHandle>) const;
    [[nodiscard]] CheckpointSummary checkpoint_summary(CheckpointHandle) const;
    [[nodiscard]] CheckpointMetadata checkpoint_metadata(CheckpointHandle) const;
    [[nodiscard]] bool checkpoint_matches(CheckpointHandle, const RequestBasePlan&) const;
    [[nodiscard]] std::uint32_t checkpoint_recovery_frontier(CheckpointHandle,
                                                             const RequestBasePlan&,
                                                             std::uint32_t target) const;
    [[nodiscard]] std::uint64_t checkpoint_recovery_loss(std::span<const CheckpointHandle>,
                                                         std::span<const CheckpointHandle>) const;
    [[nodiscard]] bool can_release_checkpoint(CheckpointHandle) const noexcept;
    [[nodiscard]] std::optional<runtime::ContextResourceUsage>
        checkpoint_release_resources(std::span<const CheckpointHandle>,
                                     runtime::ContextResourceUsage) const;
    [[nodiscard]] bool release_checkpoint(CheckpointHandle) noexcept;
    void refresh_history_requirements(const std::shared_ptr<KVHistory>&, bool trim_unused = false);
    [[nodiscard]] bool revoke_snapshot(ResumeState&) noexcept;
    [[nodiscard]] runtime::ContextResourceUsage snapshot_resources(const ResumeState& paused) const;
    // Exact Host bytes freed by deleting this fixed set, with physical sharing counted once.
    [[nodiscard]] std::size_t
    host_bytes_released(std::span<const CheckpointHandle> checkpoints) const;
    [[nodiscard]] std::optional<std::size_t> pause_host_bytes(SequenceHandle sequence) const;
    [[nodiscard]] std::size_t
    release_redundant_host(std::span<const CheckpointHandle> excluded,
                           std::optional<SequenceHandle> pending_backup = std::nullopt);
    [[nodiscard]] runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit>);
    [[nodiscard]] bool reclaim_capture_reservation(runtime::ContextResourceUsage shortage);
    void release_units(std::span<const SequenceHandle>) noexcept;
    [[nodiscard]] BindingReservation start_binding(const RequestBasePlan&, runtime::LaneId,
                                                   const SourceCandidate&, ResumeState*,
                                                   ExecutionUnitKind, std::uint32_t);
    [[nodiscard]] bool start_capture(SequenceHandle);
    [[nodiscard]] bool capture_is_input(SequenceHandle) const;
    [[nodiscard]] std::optional<CapturePreparation> prepare_capture(SequenceHandle);
    void skip_capture(SequenceHandle);
    [[nodiscard]] ContextReclaimPlan plan_reclaim(std::span<const CheckpointHandle>,
                                                  std::span<const CheckpointHandle>,
                                                  runtime::ContextResourceUsage) const;
    [[nodiscard]] std::vector<ContextRelease> plan_releases(std::span<const CheckpointHandle>,
                                                            std::span<const CheckpointHandle>,
                                                            runtime::ContextResourceUsage) const;
    [[nodiscard]] bool start_demote(const ContextDemotion&);
    [[nodiscard]] bool start_pause(SequenceHandle, bool save_snapshot, runtime::ExecutionTiming*);
    [[nodiscard]] ContextProgress poll_context(runtime::CancellationFlagView);

    [[nodiscard]] bool has_context_transaction() const noexcept {
        return context_transaction_.has_value();
    }

    [[nodiscard]] bool context_blocks(SequenceHandle) const noexcept;
    [[nodiscard]] bool recovery_pending(SequenceHandle) const noexcept;

    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle, runtime::ExecutionTiming*,
                                                  runtime::TokenMaskProvider*);
    [[nodiscard]] ReplayProgress advance_replay(SequenceHandle, runtime::ExecutionTiming*);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle>,
                                      std::span<const runtime::RoundBudget>,
                                      runtime::ExecutionTiming*, runtime::TokenMaskProvider*);
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle>, std::span<const TokenId>, std::uint32_t,
                         std::span<const std::optional<std::uint32_t>>, runtime::ExecutionTiming*);
    [[nodiscard]] CommitResult commit(PendingBatch&&, std::span<const runtime::CommitDecision>,
                                      runtime::CommitObservation, runtime::ExecutionTiming*);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&&) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle) noexcept;
    void fail_all_cleanup() noexcept;
    void shutdown_cleanup() noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;

    [[nodiscard]] std::vector<SourceCandidate> hybrid_sources(const RequestBasePlan& base,
                                                              std::uint32_t maximum_frontier);
    [[nodiscard]] bool hybrid_reclaim(runtime::ContextResourceUsage shortage);
    [[nodiscard]] std::optional<std::uint32_t> hybrid_prefetch(const RequestBasePlan& base);
    [[nodiscard]] std::uint32_t hybrid_prefetch_room() const noexcept;
    void hybrid_hold_queue(std::span<const RequestBasePlan* const> queue);
    [[nodiscard]] std::uint64_t hybrid_cache_epoch() const noexcept;
    [[nodiscard]] HybridPrefixCacheStats hybrid_stats() const noexcept;
    void set_hybrid_cost(const runtime::prefix_cache::CacheCostModel& cost);

    void set_hybrid_coalesce_wait_limit(double seconds) noexcept {
        hybrid_coalesce_wait_seconds_ = seconds > 0.0 ? seconds : 0.0;
    }

    [[nodiscard]] HybridCachePersistence attach_hybrid_cache_file(const std::filesystem::path& path,
                                                                  std::string fingerprint,
                                                                  const StartupObserver& observer);

    [[nodiscard]] std::optional<HybridCachePersistence> hybrid_shutdown_save() const {
        return hybrid_shutdown_save_;
    }
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

    const execution::Parameters& parameters;
    DeviceContext& device;
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const ContextCacheOptions context_cache;
    const std::uint32_t prefill_chunk;
    const PromptAttention prompt_attention;
    const std::uint32_t draft_window;
    const std::uint32_t neural_draft_window;
    // DFlash2 tree verification: the tree widths each batch size may verify and the most
    // root-to-leaf paths a tree row holds. In automatic mode tree_controller picks each
    // all-neural round's width.
    const TreeWidthPlan tree_widths;
    const std::uint32_t draft_tree_paths;
    std::optional<TreeWidthController> tree_controller;
    const std::uint32_t ngram_draft_window;
    const std::uint32_t ngram_min_match;
    const SpeculativeBackend speculative_backend;
    const KvCacheStorage kv_storage;
    const ProposalHead proposal_head;
    const bool vision_enabled;
    const bool use_cuda_graph;
    const bool causal_scoring;
    const std::size_t kv_payload_bytes;
    const std::size_t graph_allowance_bytes;
    // Free Device memory CUDA Graph preparation consumed at startup (instantiate, upload and one
    // launch of every executable), for comparison with graph_allowance_bytes.
    std::size_t graph_measured_bytes = 0;
    const WorkspacePlan workspace_plan;

    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_5::DecoderState> decoder;
    std::unique_ptr<HostContextArena> host_context_arena;
    std::unique_ptr<HostKVArena> host_kv_arena;
    std::unique_ptr<LogicalKVPageStore> text_kv_pages;
    std::unique_ptr<KVAddressSpaceStore> text_kv_addresses;
    std::unique_ptr<LogicalKVPageStore> backend_kv_pages;
    std::unique_ptr<KVAddressSpaceStore> backend_kv_addresses;
    std::unique_ptr<HostKVExtentStore> host_kv_extents;
    std::size_t text_host_kv_page_stride    = 0;
    std::size_t backend_host_kv_page_stride = 0;
    std::unique_ptr<qwen3_5::StateImageDevicePool> state_images;
    std::unique_ptr<qwen3_5::HostStatePool> host_state_images;
    std::unique_ptr<StateImageStore> state_store;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<ops::GdnReplayFoldPlan> replay_fold;
    // One view per round-family width below draft_window: rounds of a narrower family verify at
    // their own width for every batch size and record/fold through the dense view of that width.
    std::vector<NarrowReplayView> narrow_replay_views;
    std::optional<DFlashPersistentState> dflash;
    qwen3_5::RoundState io;
    Tensor prefill_hidden;
    std::optional<Tensor> score_hidden;
    Tensor sampling_config;
    Tensor grammar_masks_device;
    std::optional<PinnedHostBuffer> grammar_masks_host;
    std::optional<DFlashDraftHandoff> dflash_draft_handoff;
    std::array<std::uint64_t, kMaximumConcurrency> grammar_dead_positions{};
    ops::SamplingMask bind_grammar_mask(runtime::TokenMaskProvider*, std::size_t row);
    ops::SamplingMask fill_grammar_mask(runtime::TokenMaskProvider*, std::size_t row,
                                        std::span<const TokenId> drafts);
    Tensor token_counts;

    VisionHandoffState vision_handoff;
    std::array<SequenceState, kMaximumConcurrency> sequences;
    // Constrained generation: the host mirror of the device descriptors, how many leading device
    // columns may hold a constraint, and which plan each lane's set columns were written for.
    std::vector<std::int32_t> constraint_descriptors_host_;
    std::size_t constraint_device_columns_ = 0;
    std::array<std::uint64_t, kMaximumConcurrency> constraint_bound_serials_{};
    std::uint64_t next_constraint_serial_ = 1;
    std::array<RequestControl, kMaximumConcurrency> requests;
    std::array<std::uint64_t, kMaximumConcurrency> lane_epochs{};
    std::vector<CheckpointSlot> checkpoints;

    std::optional<PinnedHostBuffer> round_host;
    std::optional<PinnedHostBuffer> score_logprobs_host;
    TokenId* host_tokens = nullptr;
    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3_5::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3_5::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_5::MtpDecodeIngress* mtp_host_ingress = nullptr;
    qwen3_5::MtpDecodeEgress* mtp_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> dflash_host;
    qwen3_5::DFlashDecodeIngress* dflash_host_ingress          = nullptr;
    qwen3_5::DFlashDecodeEgress* dflash_host_egress            = nullptr;
    qwen3_5::DFlashPrefillIngress* dflash_prefill_host_ingress = nullptr;

    std::size_t workspace_logical_peak_bytes = 0;
    std::size_t vision_handoff_peak_bytes    = 0;

    struct PendingTransaction {
        std::uint64_t id = 0;
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<std::uint64_t, kMaximumConcurrency> epochs{};
        std::size_t size = 0;
    };

    std::optional<PendingTransaction> pending_transaction_;
    std::uint64_t next_transaction_id_ = 1;

    struct KVTransfer {
        LogicalKVPageStore* pages              = nullptr;
        runtime::ContextResourceClass resource = runtime::ContextResourceClass::MainKV;
        std::vector<LogicalKVPageHandle> logical;
        std::vector<DeviceKVPageHandle> physical;
        std::optional<HostKVExtentReservation> host_destination;
        std::optional<DeviceKVPageReservation> device_reservation;
        bool restore     = false;
        bool drop_device = false;
    };

    struct ContextTransaction {
        ContextOperationKind kind = ContextOperationKind::Bind;
        std::uint32_t lane        = 0;
        std::uint64_t epoch       = 0;
        std::optional<CheckpointHandle> source;
        std::vector<std::pair<CheckpointHandle, runtime::CheckpointRole>> capture_points;
        std::vector<CheckpointHandle> carried_points;
        std::vector<CheckpointHandle> carried_pins;
        std::vector<std::pair<CheckpointHandle, CheckpointHandle>> carried_clones;
        std::vector<CheckpointHandle> retired_points;
        std::shared_ptr<const RequestBasePlanImpl> base;
        ResumeState* resume  = nullptr;
        bool resume_snapshot = false;
        std::optional<ResumeState> paused;
        std::optional<StateImageTransfer> state_transfer;
        std::vector<KVTransfer> kv_transfers;
        std::optional<KVPrefixForkReservation> text_fork;
        std::optional<KVPrefixForkReservation> backend_fork;
        std::optional<KVActivationReservation> text_activation;
        std::optional<KVActivationReservation> backend_activation;
        std::optional<KVActivePrefixViewReservation> text_view;
        std::optional<KVActivePrefixViewReservation> backend_view;
        std::optional<StateImageHandle> reserved_state;
        std::optional<DeviceKVPageReservation> main_growth;
        std::optional<DeviceKVPageReservation> backend_growth;
        bool binding_prepared          = false;
        std::uint32_t reuse_frontier   = 0;
        std::uint32_t backend_frontier = 0;
        UnitDemand first_unit;
        UnitDemand reservation_demand;
        std::optional<RecoveryPermit> recovery;
        std::optional<SequenceKVBundle> reserved_kv;
        std::shared_ptr<KVHistory> binding_history;
        std::shared_ptr<KVHistory> source_history;
        std::vector<runtime::ContextTransferRequirement> transfers;
        std::vector<runtime::ContextTransferObservation> observations;
        runtime::ContextOperationCounts operations;
        bool submitted             = false;
        bool preserve_state_device = true;
        bool consume_source        = false;
        bool take_private          = false;
        bool split_state           = false;
        bool backup_state          = false;
        bool adopted               = false;
        bool borrow_text           = false;
        bool borrow_backend        = false;
        bool borrow_state          = false;
        bool source_tail_hidden    = false;

        // A binding from a hybrid prefix cache source: the pinned tree path and snapshot, the
        // forks that share its pages, and the Host restore the source needs.
        struct HybridBinding {
            std::shared_ptr<const HybridQuoteImpl> quote;
            std::vector<runtime::prefix_cache::NodeRef> path;
            std::vector<std::uint64_t> extras;
            bool snapshot_pinned = false;
            bool state_restored  = false;
            // This binding opened the cache's staged restore batch.
            bool restore_staged = false;
            std::optional<KVPagePrefixForkReservation> text_fork;
            std::optional<KVPagePrefixForkReservation> backend_fork;
            std::uint64_t restore_bytes = 0;
        };

        std::optional<HybridBinding> hybrid;
    };

    std::optional<ContextTransaction> context_transaction_;
    CudaCompletionEvent context_source_ready_;
    CudaCompletionEvent context_completion_;
    std::array<CudaEventTimer, 3> context_transfer_timers_;
    CudaEventTimer prefill_gpu_timer_;

    // Captured transfers and external events reference the buffers and events declared above.
    // Families are destroyed first, including when startup throws.
    DecodeGraphFamily ordinary_graphs;
    // Speculative round families in plan order; fixed after construction, so references into it
    // stay valid.
    std::vector<SpeculativeRoundFamily> round_families;

    [[nodiscard]] std::uint32_t initial_mtp_extent(const RequestBasePlanImpl&) const;
    [[nodiscard]] UnitDemand prefill_unit(std::uint32_t prompt, std::uint32_t cursor,
                                          std::uint32_t mtp_extent) const;
    [[nodiscard]] UnitDemand next_unit(const SequenceState&, const RequestControl&,
                                       ExecutionUnitKind, std::uint32_t tokens) const;
    void require_unit(std::uint32_t lane, ExecutionUnitKind, std::uint32_t tokens = 0) const;
    void settle_unit(std::uint32_t lane) noexcept;
    [[nodiscard]] bool valid_sequence(SequenceHandle) const noexcept;
    [[nodiscard]] bool valid_pending(const PendingBatch&) const noexcept;
    [[nodiscard]] bool valid_checkpoint(CheckpointHandle) const noexcept;
    [[nodiscard]] CheckpointState& checkpoint(CheckpointHandle);
    [[nodiscard]] const CheckpointState& checkpoint(CheckpointHandle) const;
    [[nodiscard]] std::optional<CheckpointHandle> reserve_checkpoint();
    [[nodiscard]] std::optional<CheckpointHandle> detach_checkpoint(SequenceState&);
    void abort_context() noexcept;
    void enqueue_context_transfers(ContextTransaction&);
    void enqueue_state_backup(ContextTransaction&);
    void copy_local_for_context(ContextTransaction&, std::int32_t source, std::int32_t destination);
    void copy_context_tail(ContextTransaction&, LogicalKVPageStore&, DeviceKVPageHandle source,
                           DeviceKVPageHandle destination, runtime::ContextResourceClass);
    void publish_context_transfers(ContextTransaction&);
    [[nodiscard]] bool prepare_backup(ContextTransaction&, CheckpointState&, bool shared_device);
    void plan_binding_units(ContextTransaction&, const RequestBasePlan&, ResumeState*,
                            ExecutionUnitKind, std::uint32_t);
    void install_binding(ContextTransaction&);
    void prepare_binding(ContextTransaction&);
    void complete_binding(ContextTransaction&, ContextProgress&);
    void publish_capture(ContextTransaction&);
    [[nodiscard]] bool reserve_capture_destination(std::uint32_t lane, std::uint32_t frontier);
    void prepare_capture_boundary(std::uint32_t lane);
    [[nodiscard]] ResumeState complete_pause(ContextTransaction&);
    void install_resume_sampling(SequenceState&, RequestControl&);
    void initialize_prefill(std::uint32_t lane, std::uint32_t base);
    void initialize_captures(std::uint32_t lane, std::uint32_t from, std::uint32_t through);
    void preencode_overlay_vision(execution::VisionPrefillSession&, const PreparedPromptData&,
                                  const VisionPrefillPlan&, std::uint32_t base);
    [[nodiscard]] SequenceHandle sequence_handle(std::uint32_t lane) const noexcept;
    void invalidate_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] SequenceState& active_sequence(std::uint32_t lane);
    [[nodiscard]] const SequenceState& active_sequence(std::uint32_t lane) const;
    void clear_lane(SequenceState&, RequestControl&) noexcept;
    void clear_execution_failure_lanes(std::span<const std::uint32_t>) noexcept;
    void ordered_reset(SequenceState&);
    void refresh_state_views(SequenceState&);
    [[nodiscard]] StateImageSelectors state_selectors(const SequenceState&) const;
    void settle_state_fork(SequenceState&);
    void release_sequence_state(SequenceState&) noexcept;
    void release_sequence_kv(SequenceState&) noexcept;
    void ensure_sequence_kv_mapped(SequenceState&, std::uint32_t main, std::uint32_t backend = 0);
    void trim_sequence_kv(SequenceState&, std::uint32_t main, std::uint32_t backend = 0);
    void commit_sequence_kv(SequenceState&, std::uint32_t main, std::uint32_t backend = 0);
    [[nodiscard]] PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState&) const noexcept;
    [[nodiscard]] PagedKVCacheView text_kv_view(const SequenceState&) const;
    [[nodiscard]] PagedKVCacheView mtp_kv_view(const SequenceState&) const;
    [[nodiscard]] PrefillProgress wrap_prefill(std::uint32_t lane, runtime::PrefillStepResult);
    [[nodiscard]] PendingBatch wrap_pending(std::span<const std::uint32_t>,
                                            const runtime::BatchedGeneratedRound&);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_raw(std::uint32_t,
                                                                 runtime::ExecutionTiming*);
    [[nodiscard]] runtime::ExecutionTiming resolve_prefill_raw(std::uint32_t, bool,
                                                               runtime::ExecutionTiming*);
    [[nodiscard]] runtime::ExecutionTiming
    resolve_pending_raw(std::span<const std::uint32_t>, std::span<const std::uint32_t>,
                        std::span<const std::uint8_t>, std::span<const std::uint8_t>,
                        std::span<const std::optional<std::uint32_t>>, runtime::ExecutionTiming*);
    void start_context_transfer_timer(runtime::ContextResourceClass);
    void stop_context_transfer_timer(runtime::ContextResourceClass);
    [[nodiscard]] runtime::ContextTransferObservation
        context_transfer_observation(runtime::ContextResourceClass,
                                     runtime::ContextTransferDirection, TransferWork, std::uint32_t,
                                     std::uint64_t) const;
    [[nodiscard]] runtime::BatchedGeneratedRound decode_raw(std::span<const std::uint32_t>,
                                                            std::span<const runtime::RoundBudget>,
                                                            runtime::ExecutionTiming*,
                                                            runtime::TokenMaskProvider*);
    void prepare_graphs();
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void copy_tail(SequenceState& sequence, const Tensor& source);
    void copy_round_token();
    void
    commit_generated_prefix_identity(SequenceState& sequence, std::uint32_t base_ledger_frontier,
                                     std::span<const TokenId> accepted_tokens,
                                     std::optional<std::uint32_t> prefix_execution_split_after);
    [[nodiscard]] runtime::ExecutionTiming
    resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                    std::uint32_t accepted_tokens, bool terminal,
                                    std::optional<std::uint32_t> prefix_execution_split_after,
                                    runtime::ExecutionTiming* failed_timing);
    [[nodiscard]] runtime::PrefillStepResult
    advance_prefill(SequenceState& sequence, RequestControl& request,
                    runtime::ExecutionTiming* failed_timing);
    // Log-probabilities of `tokens` and the unrestricted top token at the position after the
    // final-normalized `hidden`, from the same output-head projection the first token samples.
    [[nodiscard]] PromptReadout read_prompt_frontier(const Tensor& hidden,
                                                     std::span<const TokenId> tokens);

    // ---- constrained generation (constraint.cpp) ----
    // One row of a round: the lane, its request's compiled constraint (null when
    // unconstrained) and the output step its first logits column samples.
    struct ConstraintRow {
        std::uint32_t lane                 = 0;
        const RequestConstraintPlan* plan  = nullptr;
        std::uint32_t first_step           = 0;
    };
    // Writes the round's descriptors (row-major, `width` columns per row) for every
    // constrain_logits call the round makes, and resets columns an earlier round constrained.
    void stage_constraint_columns(std::span<const ConstraintRow> rows, std::uint32_t width);
    void bind_constraint_sets(std::uint32_t lane, const RequestConstraintPlan* plan);
    // After the round completed: the draws of `tokens`, produced by consecutive columns from
    // `first_column` for output steps from `first_step`.
    [[nodiscard]] std::vector<ConstrainedDraw>
    read_constraint_draws(const RequestConstraintPlan& plan, std::uint32_t first_step,
                          std::size_t first_column, std::span<const TokenId> tokens);
    [[nodiscard]] const RequestConstraintPlan* constraint_plan(std::uint32_t lane) const noexcept;
    // The output step the lane's next sampled token is: its generated-token count.
    [[nodiscard]] std::uint32_t constraint_first_step(std::uint32_t lane) const;
    [[nodiscard]] ConstraintRow constraint_row(std::uint32_t lane) const {
        const RequestConstraintPlan* plan = constraint_plan(lane);
        return ConstraintRow{.lane       = lane,
                             .plan       = plan,
                             .first_step = plan != nullptr ? constraint_first_step(lane) : 0U};
    }
    void enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                       std::span<const std::uint32_t> starts,
                                       std::span<const std::uint32_t> counts);
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
    void mark_workspace_usage(std::size_t phase_bytes) noexcept;
    // ReplaySSM record view and fold plan for a speculative round verified at verify_drafts.
    [[nodiscard]] const GdnReplayRecords* round_replay_records(std::uint32_t verify_drafts) const;
    [[nodiscard]] const ops::GdnReplayFoldPlan& round_replay_fold(std::uint32_t verify_drafts) const;
    // The narrowest family of `kind` that verifies at least `drafts` proposals per row.
    [[nodiscard]] SpeculativeRoundFamily& round_family(SpeculativeRoundKind kind,
                                                       std::uint32_t drafts = 0);
    [[nodiscard]] std::vector<NgramProposer::Match>
    propose_ngram(std::span<const std::uint32_t> lanes,
                  std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] NgramProposer::Match propose_ngram_one(std::uint32_t lane,
                                                         const runtime::RoundBudget& budget);
    // Moves the prepared prompt's ngram index and archive snapshot into an admitted request.
    static void take_ngram_index(RequestControl& request, const RequestBasePlanImpl& base);
    [[nodiscard]] runtime::BatchedGeneratedRound decode_ordinary_batch(
        std::span<const std::uint32_t> lanes, std::span<const runtime::RoundBudget> budgets,
        runtime::ExecutionTiming* failed_timing, runtime::TokenMaskProvider* masks);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_mtp_batch(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets,
                     runtime::ExecutionTiming* failed_timing, runtime::TokenMaskProvider* masks);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets,
                        runtime::ExecutionTiming* failed_timing, runtime::TokenMaskProvider* masks);

    // ---- hybrid prefix cache (null unless ContextCacheMode::Hybrid) ----------------------------
    std::unique_ptr<HybridPrefixCache> hybrid_;
    std::array<HybridLaneState, kMaximumConcurrency> hybrid_lanes_;
    runtime::prefix_cache::CacheCostModel hybrid_cost_;
    double hybrid_coalesce_wait_seconds_ = 0.0;
    std::filesystem::path hybrid_file_;
    std::string hybrid_fingerprint_;
    std::optional<HybridCachePersistence> hybrid_shutdown_save_;

    void create_hybrid_prefix_cache(const StartupObserver& observer);
    // Stages a binding from a hybrid source: pins the quoted path and snapshot, makes room by
    // evicting unpinned cached blocks, submits the Host restores the source needs and reserves
    // the forks. Returns the remaining shortage when room cannot be made from the cache.
    [[nodiscard]] BindingReservation
    start_hybrid_binding(const RequestBasePlan& base, std::uint32_t lane,
                         const SourceCandidate& candidate, ResumeState* resume,
                         ExecutionUnitKind resume_kind, std::uint32_t resume_tokens);
    // Activates the staged binding: the lane's KV shares the tree's pages, its state is the
    // snapshot image (or a reset), and the lane starts publishing its own blocks and taps.
    void complete_hybrid_binding(ContextTransaction& tx, ContextProgress& out);
    // Returns everything a staged hybrid binding holds. Safe on a partially staged binding.
    void abort_hybrid_binding(ContextTransaction& tx) noexcept;
    // The per-layer events the lane's first prefill pass waits on, consumed by this call; empty
    // once the batch has landed. The view is valid only until the cache's next poll(), which every
    // KV commit runs, so only PrefillContext::take_layer_ready calls it, inside the chunk function.
    [[nodiscard]] std::span<const cudaEvent_t> hybrid_take_restore_layers(std::uint32_t lane);
    // True when a sibling lane still prefilling a prompt that shares more with this one than the
    // cache offers will publish a snapshot where they diverge soon enough to wait for. Plans that
    // snapshot as an exact tap of the sibling when none is planned near the divergence.
    [[nodiscard]] bool hybrid_await_sibling(const PreparedPromptData& prompt, std::uint32_t reuse);
    // Writes the Host tier to the attached file once every Host write has landed. Called by the
    // shutdown cleanup after the lanes wrote their blocks through.
    void save_hybrid_cache_for_shutdown() noexcept;
    [[nodiscard]] bool hybrid_make_room(std::uint32_t text_pages, std::uint32_t backend_pages);
    // The backend KV frontier restored with a snapshot at `frontier` (MTP trails by one token).
    [[nodiscard]] std::uint32_t hybrid_backend_frontier(std::uint32_t frontier) const noexcept;
    // Inserts every newly committed full block of the lane's sequence into the tree, then
    // publishes the pending taps those blocks complete.
    void hybrid_publish_blocks(SequenceState& sequence);
    // Snapshots the lane's committed state at the prefill frontier `frontier`; a boundary tap is
    // published as SnapshotKind::Boundary.
    void hybrid_capture_tap(SequenceState& sequence, std::uint32_t frontier, bool boundary);
    // Realizes the planned taps a completed prefill chunk reached.
    void hybrid_after_prefill_chunk(SequenceState& sequence, std::uint32_t cursor,
                                    std::uint32_t prompt_tokens);
    // Publishes pending taps whose blocks are committed; a finishing lane hands its own last
    // pages to the remaining ones or drops them.
    void hybrid_publish_pending(SequenceState& sequence, bool finishing);
    // Copies a tail bundle into cache-owned pages; absent when no Device page can be freed.
    [[nodiscard]] std::optional<std::uint32_t> hybrid_copy_tail(const HybridBlockPages& source,
                                                                std::uint32_t columns);
    // Terminal publication: committed blocks and, when useful, an endpoint snapshot whose image
    // moves out of the lane; the snapshot the lane resumed from is superseded once a deeper one
    // exists. The caller then clears the lane.
    void hybrid_finish_lane(SequenceState& sequence, bool endpoint) noexcept;
    // Drops the lane's index pins and pending taps. Safe on any lane state.
    void hybrid_release_lane(std::uint32_t lane) noexcept;
    // Supersedes the snapshot the sequence resumed from once it snapshots past it at
    // `frontier`, before the new snapshot takes a slot or slabs (spec §9.2, §9.3).
    void hybrid_supersede_resume(HybridLaneState& lane, std::uint32_t frontier);
    void hybrid_supersede_tap(HybridLaneState& lane, std::uint32_t frontier);
};
} // namespace infernix::models::qwen3_5::detail
