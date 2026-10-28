#pragma once

// The Qwen4Exp Program: the Engine-facing execution contract of the common EngineCore over the
// HybridResourceManager surface (docs/maintainer/engine-architecture.md §2.4, §3.3). It owns the
// physical lanes (one per concurrency slot), their KV pages, recurrent state, sampling state and
// the one pending model unit. Prompts and outputs use the Qwen3.5 frontend (same tokenizer and
// chat template, design §7).
//
// Admission reserves a request's prompt pages plus one round, later rounds grow the lane, and a
// shortage pauses a younger request (design §19.3.13). With the prefix cache, an admission resumes
// from the deepest affordable snapshot and maps the cached blocks below it.

#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/output_session.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/ngram.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/expert_offload.h"
#include "models/qwen4_exp/execution/fast_path_flags.h"
#include "infernix/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/request.h"
#include "runtime/contract/resources.h"
#include "runtime/contract/timing.h"
#include "runtime/prefix_cache/cost.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace infernix {
struct DeviceContext;
}

namespace infernix::models::qwen4_exp {

namespace execution {
class Parameters;
}

namespace detail {
class ProgramImpl;
struct BasePlanImpl;
struct QuoteImpl;
struct ResumeStateImpl;
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
    std::uint32_t held_snapshots          = 0;
    std::uint64_t held_device_evictions   = 0;
    std::uint64_t held_snapshot_losses    = 0;
    std::uint64_t held_host_refusals      = 0;
    std::uint64_t tool_call_holds         = 0;
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

// One admission source: a quote of the prefix cache's chosen path and snapshot to resume from, or
// of a root start (docs/maintainer/hybrid-prefix-cache-spec.md §17).
struct SourceCandidate {
    std::uint32_t reused_tokens = 0;
    std::shared_ptr<const detail::QuoteImpl> hybrid;
    PrefixReusePath reuse_path = PrefixReusePath::Root;
};

// A binding's outcome for the Engine's admission (the shape Qwen3.5's binding returns). The prefix
// cache quotes from its own tree, so the source is always valid. A shortage is KV pages, or (no
// pages short) a Vision window the expert cache cannot lend while other lanes hold its frames
// (design §19.3.2): the Engine waits for them.
struct BindingReservation {
    bool reserved          = false;
    bool source_valid      = true;
    bool capacity_possible = true;
    runtime::ContextResourceUsage shortage;
    // With capacity_possible false: why no state of the Engine admits the request, a RequestError
    // the Engine fails it with at once (a Vision window the expert cache cannot lend with every
    // other lane idle and this request's context reserved).
    std::exception_ptr refusal;

    explicit operator bool() const noexcept { return reserved; }
};

// A paused request (design §19.3.13): its ledger, frontier and request-level state. The state at
// the frontier is not owned: the pause published it to the prefix cache as an evictable snapshot,
// and the resume binding restores from that snapshot when it is still cached, replaying the ledger
// from the deepest one otherwise.
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
          timing_(std::exchange(other.timing_, {})), row_failures_(other.row_failures_) {}
    PendingBatch& operator=(PendingBatch&&)      = delete;
    PendingBatch(const PendingBatch&)            = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }
    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }
    // Licensed tokens per row; empty when every row licenses one.
    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return row_counts_; }
    [[nodiscard]] std::uint32_t row_stride() const noexcept { return row_stride_; }
    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept { return timing_; }
    // Why the row cannot commit (None: it can).
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
    // The expert frames lent to this prompt's calls are returnable now (between calls, and after
    // each layer-walk span), so a request waiting for KV growth may be admitted: the Engine re-runs
    // admission (GitHub #24).
    bool capacity_released = false;
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
    std::optional<ExpertCacheStats> expert_cache;
    // With the prefix cache: the prompt tokens past the reused prefix whose state this request's
    // snapshots made reusable (its deepest accepted capture within the prompt).
    std::optional<std::uint32_t> cache_created_prompt_tokens;
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
    std::uint64_t expert_record_stride = 0; // the largest expert record (staging and stream-ring slots)
    std::uint32_t expert_tier_list_slots = 0; // the SSD tier's slots that hold no expert (ring, prefetch, demotion)
    std::uint64_t expert_frame_bytes   = 0; // one average expert frame (the cache's sizing unit)
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
    // Teacher-forced scoring (EnginePurpose::CausalScoring, causal_score): the device plan adds the
    // final-mixer output of a max_context window and one tile of FP32 logits.
    bool causal_scoring = false;
    std::uint32_t prefill_chunk   = 1024;
    // Main KV capacity in tokens, shared by every lane.
    std::uint32_t kv_capacity_tokens = 0;
    KvCacheStorage kv_cache          = KvCacheStorage::Int8Group64;
    // The fast paths the Forward runs (fast_paths.h); all off on the exact route.
    execution::ForwardFastPaths fast_paths;
    std::filesystem::path ngram_volume;
    // The table that volume holds (resolve_ngram_volume): another volume, in any format, replaces the
    // artifact's table. Empty: the artifact's table.
    std::optional<NgramTableConfig> ngram_table;
    // The whole n-gram table read into RAM locked for the CPU at startup (EngineOptions::ngram_residency,
    // resolved by the RAM ledger); rows are then never read from the volume.
    bool ngram_in_ram = false;
    // The expert cache's saved state (design §19.3.5 S4b): read at startup to fill the frames, written
    // at stop and every 10 minutes between requests. Empty: no file. The identity names the artifact the
    // state belongs to.
    std::filesystem::path expert_state;
    std::string expert_state_identity;
    // Without a usable saved state, seed the frames from the calibrated expert order (false: --expert-state
    // off, an empty cache).
    bool expert_seed = true;
    // The SSD tier reads each record's tail from this expert stripe (EngineOptions::expert_stripe_path).
    std::filesystem::path expert_stripe;
    // Device memory left free for the display and other programs once the fixed allocations and
    // the expert frames are made (design §19.3.7); empty selects it from the display state.
    std::optional<std::uint64_t> vram_headroom;
    bool vram_past_budget = false; // EngineOptions::vram_past_budget
    // EngineOptions::expert_tail_skip (approximate, 0: off): execution::ForwardExperts::tail_skip_rank.
    std::uint32_t expert_tail_skip = 0;
    bool expert_cache = true;
    // SSD expert tier (design §19.3.7): pinned RAM for expert slots when the model's banks are
    // streamed from the artifact (LoadOptions::stream_experts; the RAM ledger's pinned expert
    // share), and resident pageable RAM for unpinned records past the page-locking limit (its
    // pageable share). Ignored in full mode.
    std::uint64_t expert_ram_bytes      = 0;
    std::uint64_t expert_unpinned_bytes = 0;
    // The share of the frames a warm start fills with the best-ranked experts (the rest stay free for the
    // fill phase's landings). The exclusive RAM layout (RAM holds the experts VRAM does not, plus the
    // shadows a prefill's loan needs) seeds more, so fewer experts need RAM at startup.
    double expert_seed_share = 0.5;
    // The exclusive RAM layout's tier (EngineOptions::expert_ram_layout, when the plan chose it): a loan's returned
    // frames stay free for decode's landings (ExpertResidency::give_back).
    bool expert_ram_exclusive = false;
    // Pinned allocations the Program makes after the tier (the prefix cache's Host tier and an
    // allowance): when the OS refuses a slot chunk, the tier leaves this much below the limit.
    std::uint64_t later_pinned_bytes = 0;
    // Speculative decoding with n-gram copy proposals (design section 11.3): at most this many
    // draft tokens per round (0 disables, at most 15), proposed only from a match of at least
    // ngram_min_match tokens (4..64) earlier in the request.
    std::uint32_t ngram_draft_tokens = 0;
    std::uint32_t ngram_min_match    = 12;
    // MTP drafts per round when the drafter is loaded (design section 11, at most 7). A longer
    // n-gram proposal, when enabled, replaces a round's MTP drafts.
    std::uint32_t mtp_draft_tokens = 0;
    // DFlash2 (--spec dflash2 --draft-model, docs/maintainer/dflash.md): the drafter proposes this
    // many tokens per round (1..7) in one block of dflash2_draft_tokens + 1 columns; a round verifies
    // them all, or with auto_draft_length the prefix its draft-length policy chooses. A longer n-gram
    // proposal, when enabled, replaces a round's drafts as it replaces MTP's.
    std::uint32_t dflash2_draft_tokens = 0;
    // The automatic draft length (--draft-tokens auto, design 11.3): each one-row round drafts the
    // length in 0..mtp_draft_tokens that maximizes its expected tokens per round time, and a pair
    // round one length in 0..min(mtp_draft_tokens, 2) from both rows' acceptance (DFlash2: its
    // calibrated length, and a pair round's one width from both rows' calibrated drafts). False:
    // one-row rounds draft mtp_draft_tokens and pair rounds min(mtp_draft_tokens, 2) (DFlash2:
    // dflash2_draft_tokens and min(dflash2_draft_tokens, 2)).
    bool auto_draft_length = false;
    // A one-row round verifies its neural drafts up to the first whose drafter probability is below
    // this value (design §19.3.5; --spec-min-p): MTP's draft-head probability, DFlash2's
    // candidate-selector probability (temperature 1); 0 verifies every draft. 0.5 is the measured MTP
    // value the Engine passes with the automatic draft length, 0 with a fixed one.
    double draft_min_p = 0.5;
    // The request-coupling fast paths (FastPath; docs/maintainer/model-quality.md section 6.7). Each
    // is off unless selected, and with all three off a request's output does not depend on the other
    // requests (concurrency, cache contents, KV pressure):
    // - pair_speculation (adaptive-speculation with MTP, dflash2-pairs with DFlash2): two-row rounds
    //   draft one length chosen from both rows' acceptance (DFlash2: after the proposal, from both
    //   rows' calibrated draft probabilities); without it they decode plain, as rounds of three or
    //   more rows do;
    // - free_cache_resume (free-cache-resume): prefill calls, taps and resumes at any position, sibling
    //   coalescing, and endpoints and blocks holding decode-computed state; without it every call
    //   boundary, tap and resume is a multiple of 16 tokens (the GDN prefill's sub-chunk) and only
    //   prompt blocks and taps are published;
    // - preemption (preemption): KV pages grow per unit and a short pool pauses residents (replay
    //   recomputes generated tokens as prefill); without it a request holds its whole extent from
    //   admission.
    bool pair_speculation  = false;
    bool free_cache_resume = false;
    bool preemption        = false;
    // sampled-mtp-drafts (FastPath::SampledMtpDrafts): a sampled row's MTP drafts are drawn from the
    // draft head's top tokens under its sampler and verified by sparse rejection sampling, and its
    // one-row rounds verify every draft (no draft_min_p cut). Greedy rows are unchanged.
    bool sampled_mtp_drafts = false;
    // The reproducible route (EngineOptions::fast_paths off): sampled requests draw by Gumbel-max
    // (ops::SamplingConfig::gumbel_max), so speculation returns plain sampling's text.
    bool gumbel_sampling = false;
    // EngineOptions::logits_dtype BF16: the sampler, verification, constraints, the prompt readout and scoring
    // read the head's logits rounded once to BF16 (the lossless route: each exact sum correctly rounded to BF16).
    bool bf16_logits = false;
    // screened-lm-head (FastPath::ScreenedLmHead): decode and verification rounds of at most 16 columns
    // whose every row is greedy, without penalties, grammar or token constraint take the head's logits
    // from its 2-bit screen (Parameters::output_head_screen); other rounds run the exact head. Each
    // round kind keeps its own CUDA graphs.
    bool screened_lm_head = false;
    // fp16-gdn-state (FastPath::Fp16GdnState): the GDN recurrent state pool (and so its prefix-cache
    // images) stores FP16; every GDN call keeps an FP32 master state and rounds the state it publishes.
    bool fp16_gdn_state = false;
    // block-verification (FastPath::BlockVerification): sampled rows of DFlash2 and sampled-MTP chain
    // rounds accept by block verification (ops::speculative_accept_sparse_drafts).
    bool block_verification = false;
    // CPU-served misses (design section 10): host expert-engine workers (0 disables), the most
    // experts one layer call hands to them, and the share kept on the PCIe stage (misses / divisor;
    // 0 derives it from the measured link and CPU rates, design §19.3.12).
    // Defaults measured fastest on the i9-13900K (design sections 19.2 and 19.3.16): 16 jobs pay in
    // calls with many misses (C = 4 fill +7 %, cold MTP +3 %) and are neutral elsewhere; 32 is no faster.
    std::uint32_t cpu_expert_workers = 6;
    // The workers for a GGUF import's banks (the ggml CPU route, compute-bound where the NVFP4
    // route is DRAM-bound): on the i9-13900K, tg512 decoded 100.5 tok/s without CPU service, 98.4
    // with 6 workers, 103.7 with 16 and 103.6 with 20 (2026-10-11; 16 workers with divisor 3: 101.1).
    std::uint32_t cpu_ggml_expert_workers = 16;
    std::uint32_t cpu_expert_jobs    = 16;
    std::int32_t cpu_pcie_divisor    = 0;
    // Prefill CPU assist (design §19.3.1 P7): prefill calls wider than a decode round and narrower
    // than kAssistMaxColumns hand up to this many of each layer's thinnest misses to the CPU (same
    // divisor). 0 disables.
    std::uint32_t cpu_assist_jobs    = 256;
    // Prefill CPU split (design §19.3.12): a narrow streamed call outside a layer walk copies each
    // layer's experts only once its routing is known, leaving the thinnest non-resident ones (at
    // most 8 columns, the placement-invariant narrow route) to the CPU while the link carries the
    // rest. The split balances the measured link against the CPU team's rates, measured at startup
    // with the link busy (cpu_rates.cpp): experts per second on one-column jobs and expert columns
    // per second on 8-column jobs, planned at a share of them (ProgramImpl::kSplitRateShare, design
    // §19.3.12). cpu_split_*_rate are the reference rates used only when the team
    // cannot be timed (the SSD tier: no records in host memory; i9-13900K, 6 workers, from
    // host_probe with the link idle). Gating would leave the link idle between a layer's routing and
    // its copies; each layer's routing guess fills the next layer's ring half meanwhile (copy-ahead,
    // ExpertStream::plan_ahead), so the gated stream pays up to cpu_split_columns columns on the
    // reference link (PCIe 5.0 x8, 27.5 GB/s), a limit that scales inversely with the measured link
    // (runtime::expert_offload::split_width). Dense8 prefill against the earlier route (2026-10-26, ABBA:
    // pp1024 gated without the copy-ahead, pp2048 and pp4096 blind; the row threshold the copy-ahead uses,
    // ExpertStream::ahead_rows): +10.2 %, +14.4 %, +8.9 %; without
    // the copy-ahead the gated stream lost from 2,048 columns (-2.3 % / -9.6 %).
    bool prefill_cpu_split           = true;
    double cpu_split_expert_rate     = 21000.0;
    double cpu_split_column_rate     = 66000.0;
    std::uint32_t cpu_split_columns  = 4096;
    // Prefill expert streaming (design §19.3.8 F2, F3): chunks of at least 256 columns copy each
    // layer's non-resident experts ahead by DMA into frames lent for the prompt.
    bool prefill_stream              = true;
    // Prefix cache (design §19.3.1): requests resume from and publish to a shared block tree with
    // Host-born state snapshots in a pinned slab pool of prefix_host_bytes.
    bool prefix_cache                = false;
    std::uint64_t prefix_host_bytes  = 0;
    runtime::prefix_cache::TapPlannerConfig prefix_taps;
    runtime::prefix_cache::CacheCostModel prefix_cost;
    // What a cut that ends a layer walk's span costs (the next span streams the experts again), at
    // the reference link rate kReferenceLinkBytesPerSecond; the Program rescales it to its link.
    double prefix_span_seconds = 0.0;
    // The host-to-device link rate in bytes per second; 0 measures it at startup (a fixed rate is
    // for tests of the bandwidth-dependent choices).
    double link_bytes_per_second = 0.0;
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
    // The expert cache's cumulative counters and current size (RuntimeStats::expert_cache); nothing
    // without an expert cache. Engine thread, between rounds.
    [[nodiscard]] std::optional<ExpertCacheTotals> expert_cache_totals() const noexcept;

    // Expert-cache resizing outside rounds (design §19.3.7). The VRAM monitor calls the waker from
    // its own thread when the cache should resize; the engine then calls maintain() from its
    // worker, between rounds and under its execution lock. An empty waker detaches the engine.
    void set_maintenance_waker(std::function<void()> waker);
    void maintain();
    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;

    [[nodiscard]] RequestBasePlan plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options);

    // ---- admission: hybrid_sources quotes, start_binding stages, poll_context activates ----
    // The prefix cache's quotes in preference order: its chosen path and snapshot, then a root start
    // (a root start only, without a prefix cache).
    [[nodiscard]] std::vector<SourceCandidate> hybrid_sources(const RequestBasePlan& base,
                                                              std::uint32_t maximum_frontier);
    // Reserves the KV pages of the request's prompt plus one round on `lane`, and the frames of its
    // Vision window, and stages its prefix restore; a shortage reports the pages missing (none for
    // a window that waits for frames), a window that cannot fit even so a refusal. With `resume`,
    // binds the paused request's ledger instead: the deepest cached state up to its paused frontier,
    // then replay to it.
    [[nodiscard]] BindingReservation start_binding(const RequestBasePlan& base, runtime::LaneId lane,
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
    // Holds the snapshot each queued request would resume from now, in admission order
    // (hybrid-prefix-cache-spec §9.6); an empty queue releases every hold.
    void hybrid_hold_queue(std::span<const RequestBasePlan* const> queue);
    // No tool-call holds (HybridPrefixCacheOptions::tool_call_hold is zero for this architecture
    // and normalize_engine_options refuses another value), so the queue holds go stale only with
    // the queue or the snapshots.
    void hybrid_hold_continuation(std::chrono::steady_clock::time_point) {
        throw std::logic_error("Qwen4Exp: the prefix cache has no tool-call holds");
    }
    [[nodiscard]] bool hybrid_holds_stale() const noexcept { return false; }
    // Changes when a queued request's choice of snapshot may have changed.
    [[nodiscard]] std::uint64_t hybrid_cache_epoch() const noexcept;
    // The positions a request can resume from: every position with free-cache-resume, else the
    // 16-token grid (the reproducible route).
    [[nodiscard]] std::uint32_t hybrid_resume_grid() const noexcept;
    [[nodiscard]] HybridPrefixCacheStats hybrid_stats() const noexcept;

    // ---- execution units (design §19.3.13): a binding reserves the KV pages of its prompt (or
    // replay) plus one round; a decode or control unit grows the lane's pages to its next round, and
    // a shortage the Engine cannot reclaim pauses the youngest resident ----
    [[nodiscard]] runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit> units);
    void release_units(std::span<const SequenceHandle>) noexcept {}
    [[nodiscard]] bool recovery_pending(SequenceHandle) const noexcept { return false; }

    // ---- pause and replay (design §19.3.13) ----
    // Publishes the lane's state at its frontier to the prefix cache, releases the lane and opens a
    // Pause transaction that returns the ResumeState (a lane inside a layer walk's span publishes
    // nothing past the span's start). False while another transaction or a round is open.
    [[nodiscard]] bool start_pause(SequenceHandle sequence, runtime::ExecutionTiming* = nullptr);
    // One prefill call of a resumed lane's ledger, without sampling, toward its paused frontier.
    [[nodiscard]] ReplayProgress advance_replay(SequenceHandle sequence, runtime::ExecutionTiming* = nullptr);

    // `masks` (null when no row is constrained) supplies the rows' grammar masks for this call.
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* failed_timing = nullptr,
                                                  runtime::TokenMaskProvider* masks = nullptr);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing = nullptr,
                                      runtime::TokenMaskProvider* masks       = nullptr);
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
    // After a runtime::RecoverableExecutionError from a call: releases every lane without
    // publishing, lets the prefix cache's transfers land and empties the cache when the failed
    // round may have published into it. False when that fails (the Engine fails everything).
    [[nodiscard]] bool recover_after_failure() noexcept;
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
    // Between units (no unit's Device work in flight): writes the Host tier to the attached file
    // when it has been written since the last save or load (Engine periodic save,
    // HybridPrefixCacheOptions::persistent_save_interval). Absent without a file or a change.
    [[nodiscard]] std::optional<PrefixCachePersistence> save_prefix_cache_now(const CancellationView& abandoned);

    // Teacher-forced scoring of a text prompt (ProgramOptions::causal_scoring; an idle Program): the
    // log-probability of each token from first_target on given the tokens before it, with the
    // top_k distribution and the candidates' log-probabilities (ScoreOptions).
    [[nodiscard]] ScoreResult causal_score(PreparedPrompt&& prompt, std::uint32_t first_target,
                                           const ScoreOptions& options);

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
};

} // namespace infernix::models::qwen4_exp
