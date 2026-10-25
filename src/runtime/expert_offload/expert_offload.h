#pragma once

// One Program's offloaded routed experts (docs/maintainer/qwen3_5-122b-offload-design.md §7, §9.2,
// §10, §11): the VRAM expert cache (frames, LFRU promotions, landings), the staging slots, the SSD
// tier when the banks stay in the artifact, the measured host-to-device link, the CPU miss service
// and its measured rates, the prefill expert stream with its frame leases, the prefill CPU split, the
// VRAM monitor and its control law, the warm start and saved state, the route trace and the error
// word. The mechanisms and constants are Qwen3.8-Flash-Next's, each measured there
// (docs/maintainer/qwen3_8-flash-next-design.md §19.2-19.3); nothing here knows a model. A Program
// owns one and shares no mutable state with another Program.
//
// Order of use. The Program plans fixed_bytes() into its device plan and constructs the engine among
// its fixed allocations (under the engine's VRAM source, vram()). After them, start() builds the SSD
// tier, measures the link, creates the prefill stream and the CPU service, sizes the frames from
// free memory, starts the monitor and warm-starts the frames. source(layer) is a layer's
// MoeExpertSource for its MoE calls; the Program sets the per-call fields (a graphed round's fork
// and landing, an eager call's overlap, a streamed chunk's prefetch ring). Around each round:
// before_round, then the calls, enqueue_route_download, the Program's synchronization,
// check_error, and settle (decode and verification, deferred past the commit) or after_call
// (prefill and forced tokens), each of which appends its round to the route trace; prefill calls
// credit their lane's prompt routes (note_prompt_routes, seed_prompt at the prompt's end);
// apply_vram_target at boundaries.

#include "core/device.h"
#include "core/host_memory.h"
#include "core/vram_budget.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "infernix/types.h"
#include "ops/offloaded_sparse_moe/cpu/fetch_channel.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "runtime/expert_offload/expert_bank.h"
#include "runtime/expert_offload/expert_residency.h"
#include "runtime/expert_offload/expert_state.h"
#include "runtime/expert_offload/expert_store.h"
#include "runtime/expert_offload/expert_stream.h"
#include "runtime/expert_offload/host_expert_tier.h"
#include "runtime/expert_offload/memory_plan.h"
#include "runtime/expert_offload/placement.h"
#include "runtime/expert_offload/route_trace.h"
#include "runtime/expert_offload/vram_monitor.h"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace infernix::runtime::expert_offload {

// The prefill CPU split's choice (Flash-Next design §19.3.12): for one layer of a gated streamed
// chunk, marks cpu[e] = 1 for the non-resident experts the CPU computes (columns[e]: the chunk's
// columns routed to expert e) and returns how many it marked.
class CpuSplitPolicy {
public:
    virtual ~CpuSplitPolicy() = default;
    virtual std::uint32_t choose(std::uint32_t layer, std::span<const std::int32_t> columns,
                                 std::span<std::uint8_t> cpu) const = 0;
};

// The model-independent subset of a Program's options the engine follows (the Program's own
// ProgramOptions carry them from the EngineOptions).
struct OffloadOptions {
    bool expert_cache = true;  // VRAM frames (false: every expert is read zero-copy, staged or CPU-served)
    bool vram_monitor = true;  // the elastic frame pool follows free VRAM (VramControl)
    bool vram_past_budget = false;
    std::optional<std::uint64_t> vram_headroom;
    double vram_grow_delay_seconds = 30.0;
    bool prefill_stream    = true; // ring streaming of wide prefill calls (design §10)
    bool prefill_cpu_split = true; // the CPU computes a gated streamed chunk's thinnest misses
    std::uint32_t cpu_expert_workers = 6;
    std::uint32_t cpu_expert_jobs    = 16;  // CPU-served misses per decode or verification call
    std::uint32_t cpu_assist_jobs    = 256; // CPU-served misses per assisted prefill call (0: none)
    int cpu_pcie_divisor             = 0;  // > 0 fixes the decode miss divisor
    // A GGUF import's one-column calls take the context's divisor (plain_divisor) without speculation
    // (INFERNIX_TMP_PLAIN_DIVISOR_ALL: every bank's).
    bool plain_divisor_by_context    = false;
    double link_bytes_per_second     = 0.0; // > 0 fixes the link rate (else measured)
    // The CPU team's reference rates (i9-13900K, 6 workers, NVFP4 records) until measured, and the
    // widest call the prefill split takes on the reference link.
    double cpu_split_expert_rate = 21000.0;
    double cpu_split_column_rate = 66000.0;
    double cpu_split_columns     = 1536.0;
    // The SSD tier's RAM (the RAM ledger's Tier placement): pinned and unpinned record bytes, and
    // the pinned bytes Infernix's later allocations need.
    std::uint64_t expert_ram_bytes      = 0;
    std::uint64_t expert_unpinned_bytes = 0;
    std::uint64_t later_pinned_bytes    = 0;
    std::filesystem::path expert_state; // the saved expert state (warm start); empty: none
    std::string expert_state_identity;
    std::filesystem::path expert_stripe; // an expert stripe on a second drive (SSD tier)
    std::filesystem::path route_trace;   // the internal route trace; empty: none
    std::uint32_t expert_tail_skip = 0;  // approximate tail skip rank (0: off)
};

// What the engine serves: the offloaded layers' routed banks and the Program's round geometry.
struct OffloadSetup {
    ExpertLayout layout;
    // Per offloaded layer: its routed bank (pinned host planes, or views into the artifact with the
    // SSD tier) and its device ExpertScales (scales holds `experts` entries; with a shared job the
    // Program's array holds one more).
    std::vector<const ExpertBank*> banks;
    std::vector<const ops::offloaded_moe::ExpertScales*> device_scales;
    // The SSD tier's records (the Model's store; null in full mode) and the artifact's id (an
    // expert stripe names it).
    const ExpertStore* store = nullptr;
    artifact::ArtifactId artifact_id{};
    std::int32_t lanes     = 1;
    std::int32_t max_width = 1; // a verification round's columns per row (1: plain decode only)
    std::int32_t columns   = 1; // the most columns one round routes (the route log)
    // Prefill: the chunk, the layer walk's longest span (tokens), and whether rounds speculate (MTP or
    // n-gram drafts: the one-column divisor rule is plain decode's only).
    std::int32_t chunk           = 1024;
    std::uint32_t walk_max_tokens = 65536;
    bool speculative             = false;
    // The route trace's header: the most drafts an MTP or n-gram round verifies.
    std::uint32_t mtp_draft_tokens   = 0;
    std::uint32_t ngram_draft_tokens = 0;
    // Graph executables the VRAM sizing reserves room for (memory_plan graph_bound), and whether the
    // Program captures them all right after the frames are sized (VramDemand::eager_graphs).
    std::uint32_t graph_bound = 0;
    bool eager_graphs         = false;
    // A model's calibrated expert order (Qwen3.8-Flash-Next's): the warm start's seed when the saved
    // state cannot be used; null: none.
    const expert_cache::SavedState* calibrated_state = nullptr;
    int device = 0;
};

class ExpertOffload {
public:
    using Diagnostic = std::function<void(const std::string&, DiagnosticLevel)>;

    // Misses staged per pass of a layer's experts: every decode and verification call's misses in
    // one pass, a prefill chunk's in several; at least kStagingSlots, with the SSD tier exactly that
    // (Flash-Next: 80 slots +7.2 % MTP on IQ3_S; with the tier the larger staging cost frames and RAM).
    static constexpr std::int32_t kStagingSlots = 64, kMaxStagingSlots = 512;
    // Fewer frames than this make decode slow (runtime/engine's kFewExpertFrames for Flash-Next).
    static constexpr std::uint32_t kFewFrames = 2048;
    // Calls of at most this many columns are CPU-served (design §16.2); prefill calls up to
    // kAssistMaxColumns are CPU-assisted (P7).
    static constexpr std::int32_t kServedCallColumns = 8;
    static constexpr std::int32_t kAssistMaxColumns  = 255;
    // Chunks narrower than this keep the MoE's staging; wider ones stream (F2), the ring halves at
    // least kStreamHalfMin records.
    static constexpr std::int32_t kStreamMinColumns = 256;
    static constexpr std::uint32_t kStreamHalfMin   = 16;
    // RAM slots a layer walk's span lends for its stream's reads of SSD-only experts.
    static constexpr std::uint32_t kStreamReadSlots = 1280;
    // The rate the link-bound constants were fitted at (RTX 5090 on PCIe 5.0 x8).
    static constexpr double kReferenceLinkBytesPerSecond = 27.5e9;
    // Promotions per layer and token at the decode rate (a forced-token call takes one).
    static constexpr std::size_t kDecodePromotionsPerLayer = 1;

    // The staging slots and the device bytes the constructor allocates (staging and the residency's
    // tables), for the Program's device plan.
    [[nodiscard]] static std::int32_t staging_slots(const OffloadSetup& setup, const OffloadOptions& options) noexcept;
    [[nodiscard]] static std::uint64_t record_stride(const OffloadSetup& setup);
    [[nodiscard]] static std::uint64_t fixed_bytes(const OffloadSetup& setup, const OffloadOptions& options);

    // Opens the VRAM source, allocates the staging slots and creates the residency (no frames yet).
    ExpertOffload(OffloadSetup setup, OffloadOptions options, Diagnostic diagnostic);
    ~ExpertOffload();
    ExpertOffload(const ExpertOffload&)            = delete;
    ExpertOffload& operator=(const ExpertOffload&) = delete;

    // The VRAM source the Program's SpillGuard checks its own fixed allocations with.
    [[nodiscard]] VramBudgetSource& vram() noexcept { return *vram_; }

    // After the Program's fixed allocations: the SSD tier, the link, the stream, the CPU service and
    // its rates, the frames (sized from free memory for `demand`'s graphs and headroom), the VRAM
    // monitor, the warm start and the route trace. `lendable_frames` (optional) tells the Program how
    // many frames it may borrow for a wide workspace (fit_wide_work_to_cache), before and after sizing.
    void start(cudaStream_t compute, const std::function<void(std::uint64_t frames, std::uint64_t pitch)>& lendable_frames = {});

    // A layer's MoE source: its frames, host records (or tier tables), scales, geometry, staging, CPU
    // channel, fetch channel and error word. The per-call fields stay unset: landing (landing_slots(),
    // landing_table(), landed_log() for a graphed round's forked calls), fork and overlap streams
    // (fork_stream()/overlap_stream()), the stream ring (prefetched) and tail skip (tail_skip_rank()).
    [[nodiscard]] ops::MoeExpertSource source(std::uint32_t layer) const;
    [[nodiscard]] std::int32_t* route_log() noexcept { return residency_->route_log(); }
    [[nodiscard]] std::size_t route_stride() const noexcept { return residency_->route_stride(); }
    [[nodiscard]] const std::int32_t* landing_table() const noexcept { return residency_->landing_table(); }
    [[nodiscard]] std::int32_t* landed_log() noexcept { return residency_->landed_log(); }
    [[nodiscard]] static constexpr std::int32_t landing_slots() noexcept {
        return static_cast<std::int32_t>(ExpertResidency::kLandingSlots);
    }
    // The fork route's stream and events (graphed decode and verification calls) and the overlap
    // route's (eager prefill calls), owned here.
    [[nodiscard]] cudaStream_t fork_stream() const noexcept { return overlap_.stream; }
    [[nodiscard]] std::span<const cudaEvent_t, 2> fork_events() const noexcept {
        return std::span<const cudaEvent_t, 2>(overlap_.events.data(), 2);
    }
    [[nodiscard]] cudaStream_t overlap_stream() const noexcept { return overlap_.stream; }
    [[nodiscard]] std::span<const cudaEvent_t, 5> overlap_events() const noexcept { return overlap_.events; }
    [[nodiscard]] std::int32_t tail_skip_rank() const noexcept { return static_cast<std::int32_t>(options_.expert_tail_skip); }
    [[nodiscard]] std::int32_t staging_slots() const noexcept { return staging_slots_; }
    [[nodiscard]] std::uint64_t record_stride() const noexcept { return record_stride_; }

    [[nodiscard]] ExpertResidency& residency() noexcept { return *residency_; }
    [[nodiscard]] const ExpertResidency& residency() const noexcept { return *residency_; }
    [[nodiscard]] HostExpertTier* tier() noexcept { return tier_.get(); }
    [[nodiscard]] ExpertStream* stream() noexcept { return stream_.get(); }
    [[nodiscard]] ops::offloaded_moe::CpuMissService* cpu_service() noexcept { return cpu_service_.get(); }
    [[nodiscard]] const CpuSplitPolicy* split_policy() const noexcept { return split_.get(); }
    [[nodiscard]] double link_bytes_per_second() const noexcept { return link_bytes_per_second_; }
    [[nodiscard]] std::uint32_t frames() const noexcept { return residency_->frames(); }
    // The memory summary's expert-cache facts (MemorySummary::expert_cache).
    [[nodiscard]] ExpertCacheMemorySummary memory_summary() const noexcept;

    // ---- round hooks ----
    // Before a round's calls: wakes the CPU service and opens the residency's round (landing for
    // decode and verification rounds; `walking` while a layer walk spans rounds).
    void before_round(cudaStream_t compute, bool landing, bool walking = false, std::uint32_t stream_slots = 0);
    void enqueue_route_download(cudaStream_t compute, std::int32_t columns) {
        residency_->enqueue_route_download(compute, columns);
    }
    // After every synchronization of a round with offloaded layers: a call that could not serve an
    // expert left undefined outputs; throws RecoverableExecutionError naming the layer (the round's
    // requests fail, the Engine keeps serving), or runtime_error when the CPU service or the tier's
    // read agent stopped. `model` names the model in the message. Returns normally when clean.
    void check_error(const std::string& model);
    // Whether the last check_error failed a round (the Program drops the prefix entries it published).
    [[nodiscard]] bool take_failed_round() noexcept {
        const bool failed = failed_round_;
        failed_round_     = false;
        return failed;
    }

    // A round as the route trace records it (ignored without a trace): its kind, `rows` rows of
    // `width` columns from `lanes` (one per row), and a prefill call's first position.
    struct TracedRound {
        RouteTraceKind kind   = RouteTraceKind::Decode;
        std::int32_t rows     = 1;
        std::int32_t width    = 1;
        std::uint32_t position = RouteTrace::kUnknownPosition;
        std::span<const std::int32_t> lanes;
    };
    // A decode or verification round's cache update, deferred past its commit's enqueue so the policy
    // overlaps that GPU work (Flash-Next §19.3.5): defer() after the round's synchronization, settle()
    // once the commit is enqueued. `tokens` is the longest row's advance; with `live` only its
    // flagged columns count as uses (a verification's accepted columns).
    void defer(std::int32_t columns, std::uint32_t tokens, const TracedRound& round,
               std::span<const std::uint8_t> live = {});
    void settle(cudaStream_t compute);
    void drop_deferred() noexcept { deferred_.pending = false; }
    // A prefill or forced call's update (after its synchronization): `promotions` per layer, the call
    // advancing `tokens` positions; `live` (one flag per column, empty: every column) as defer's.
    void after_call(cudaStream_t compute, std::int32_t columns, std::uint32_t tokens, std::size_t promotions,
                    const TracedRound& round, std::span<const std::uint8_t> live = {});
    // Promotions per layer for a round advancing `tokens` tokens: one per token while the frames fill,
    // then every promotion_interval()-th token.
    [[nodiscard]] std::size_t decode_budget(std::uint32_t tokens);
    [[nodiscard]] std::uint64_t promotion_interval() const noexcept;

    // A prompt's routes credit the cache at its end (design §19.3.27): a prefill call's per-token
    // routes (read from the route log its after_call is about to apply; a layer walk's span counts
    // its last chunk) add to its lane's uses, and at the prompt's end, after its last after_call, an
    // expert every counted token routed to gains kPromptSeedCredit of count, the others in
    // proportion (INFERNIX_TMP_PROMPT_SEED: another credit, and promotions on `compute`). A lane's
    // uses clear when it binds a new request.
    void note_prompt_routes(std::uint32_t lane, std::int32_t columns);
    void seed_prompt(std::uint32_t lane, cudaStream_t compute);
    void clear_prompt_uses(std::uint32_t lane);

    // ---- frames and memory ----
    // At a boundary (nothing in flight): resizes the frames to the control law's target. `idle`: no
    // request is active.
    void apply_vram_target(cudaStream_t compute, bool idle);
    // Device memory moves between the frames and another elastic pool (the Program's KV): take()
    // shrinks the frames by `bytes` (false: a lease holds them or the frames do not cover it), give()
    // grows them by `bytes`; both account the other pool to the control law.
    [[nodiscard]] bool take_frames(std::uint64_t bytes, cudaStream_t compute);
    void give_frames(std::uint64_t bytes, cudaStream_t compute);
    // Frames the sizing would give the cache now beyond its current count (memory the other pool may
    // take first).
    [[nodiscard]] std::uint32_t spare_frames() const;
    void set_maintenance_waker(std::function<void()> waker);
    // A Program-side condition the monitor thread also wakes the idle engine for (the elastic KV's idle
    // shrink); it must be cheap and thread-safe.
    void set_maintenance_due(std::function<bool()> due);
    // Between requests: the VRAM target, and the saved state every ten minutes of rounds.
    void maintain(cudaStream_t compute);
    void save_state() noexcept;
    void report_layers() noexcept;
    [[nodiscard]] std::string request_report(const ExpertResidency::Stats& at_admission, std::uint64_t cpu_at_admission,
                                             const HostExpertTier::Stats* tier_at_admission) const;
    // A finished request's request_report as a Debug diagnostic.
    void report_request(const ExpertResidency::Stats& at_admission, std::uint64_t cpu_at_admission,
                        const HostExpertTier::Stats* tier_at_admission) const noexcept;
    // Misses the CPU service has served since start (0 without one).
    [[nodiscard]] std::uint64_t cpu_served() const noexcept;

    // ---- prefill stream and frame leases ----
    // Whether a call of `width` columns streams: lends the ring (and `extra` leases' bytes, each one
    // contiguous run: the Program's wide workspace and a layer walk's area) when none is held.
    struct StreamLeases {
        ExpertResidency::FrameLease ring, wide, walk;
    };
    [[nodiscard]] bool lend_stream(std::int32_t width, std::uint64_t wide_bytes, std::uint64_t walk_bytes,
                                   cudaStream_t compute);
    [[nodiscard]] bool stream_lent() const noexcept { return leases_.ring.valid(); }
    [[nodiscard]] const StreamLeases& stream_leases() const noexcept { return leases_; }
    [[nodiscard]] DeviceSpan ring_span() const noexcept { return leases_.ring.memory; }
    // Returns the stream's leases (after the prompt's last streamed call is enqueued).
    void return_stream(cudaStream_t compute);
    // A lease for another use of device memory (a wide workspace without the stream, a vision window).
    [[nodiscard]] ExpertResidency::FrameLease lend(std::uint64_t bytes, cudaStream_t compute,
                                                   std::span<const cudaStream_t> writers);
    void give_back(ExpertResidency::FrameLease& lease, cudaStream_t compute) { residency_->give_back(lease, compute); }

    // ---- CPU ----
    [[nodiscard]] double cpu_expert_rate() const noexcept;
    [[nodiscard]] double cpu_column_rate() const noexcept;
    [[nodiscard]] int pcie_divisor() const noexcept;
    // INFERNIX_TMP_LINK_SHARE: the link's share of a call's misses in 256ths (0: pcie_divisor).
    [[nodiscard]] int pcie_share() const noexcept;
    // Plain decode's one-column divisor by context (placement.h plain_divisor; Flash-Next: d = 1
    // below 1,536 tokens, 2 from there): its startup value, 0 when the rule does not apply.
    [[nodiscard]] int one_column_divisor() const noexcept;
    // Before a one-column decode round at `position` context tokens: the divisor word the calls read
    // follows the context (uploaded on `compute` when it changes). A no-op without the rule.
    void set_plain_position(std::int32_t position, cudaStream_t compute);
    // The widest call the prefill CPU split takes at the measured link.
    [[nodiscard]] std::int32_t split_columns() const noexcept;

    // The route trace (optional; settle and after_call append their rounds).
    [[nodiscard]] RouteTrace* trace() noexcept { return trace_.get(); }

    // The engine's diagnostic sink, for the Program's memory events beside the cache's (elastic KV).
    void report(const std::string& text, DiagnosticLevel level = DiagnosticLevel::Info) const noexcept {
        diagnostic(text, level);
    }

private:
    void measure_link();
    void measure_cpu_rate();
    // INFERNIX_TMP_CPU_PROBE_DETAIL=1 (candidates §11.2 OR-08): the team's rates on the same 32 records
    // where they live (the tier's slots or the registered banks), in registered VirtualAlloc copies and
    // in cudaHostAlloc copies, logged once at startup.
    void cpu_probe_detail();
    void create_tier();
    void create_stream();
    void create_cpu_service();
    void size_frames(cudaStream_t compute, const std::function<void(std::uint64_t, std::uint64_t)>& lendable_frames);
    void warm_start(cudaStream_t compute);
    [[nodiscard]] VramDemand frame_demand(std::uint64_t frame_bytes) const;
    void on_vram_snapshot(const VramSnapshot& snapshot);
    [[nodiscard]] double vram_seconds() const;
    void diagnostic(const std::string& text, DiagnosticLevel level = DiagnosticLevel::Info) const noexcept;
    // A record's host address for the CPU and the rate probe: the pinned bank, or the tier's RAM copy.
    [[nodiscard]] const std::uint8_t* host_record(std::uint32_t layer, std::uint32_t expert) const;

    OffloadSetup setup_;
    OffloadOptions options_;
    Diagnostic diagnostic_;
    std::unique_ptr<VramBudgetSource> vram_;
    std::uint64_t record_stride_ = 0;
    std::vector<std::uint64_t> host_strides_;
    std::int32_t staging_slots_  = 0;
    DeviceBuffer staging_;
    // Plain decode's divisor word (one_column_divisor() != 0): the device I32 the one-column calls read,
    // its pinned source and the value last uploaded.
    DeviceBuffer divisor_word_;
    PinnedHostBuffer divisor_host_{16};
    std::int32_t divisor_now_ = 0;
    // The CPU expert team's worker count and CPUs (INFERNIX_TMP_CPU_PLACEMENT / _ENGINE_ISOLATE).
    CpuPlacement placement_;
    std::unique_ptr<ExpertResidency> residency_;
    std::optional<ExpertStore> striped_store_;
    std::unique_ptr<ops::offloaded_moe::FetchChannel> fetch_channel_;
    std::unique_ptr<HostExpertTier> tier_;
    PinnedHostBuffer error_{64};
    std::unique_ptr<ExpertStream> stream_;
    std::unique_ptr<ops::offloaded_moe::CpuMissService> cpu_service_;
    std::unique_ptr<CpuSplitPolicy> split_;
    double link_bytes_per_second_ = kReferenceLinkBytesPerSecond;
    double cpu_expert_rate_ = 0.0, cpu_column_rate_ = 0.0; // 0 until measured
    std::string calibration_note_; // INFERNIX_TMP_MACHINE_CALIBRATION's note on the CPU rates (the startup line)
    bool failed_round_ = false;

    // The fork / overlap stream and its events (start, staged[2], consumed[2]; the fork uses the first two).
    struct Overlap {
        ~Overlap();
        cudaStream_t stream = nullptr;
        std::array<cudaEvent_t, 5> events{};
    };
    Overlap overlap_;

    void trace_round(const TracedRound& round, std::uint32_t tokens, std::size_t budget,
                     std::span<const std::uint8_t> live);

    struct Deferred {
        std::int32_t columns = 0;
        std::uint32_t tokens = 0;
        std::vector<std::uint8_t> live;
        TracedRound round;
        std::vector<std::int32_t> lanes; // round.lanes' copy
        bool pending = false;
    };
    Deferred deferred_;
    std::uint64_t budget_tokens_ = 0;

    // Prompt credit (design §19.3.27), per lane: per key, the prompt tokens routed to it.
    struct PromptUses {
        std::vector<std::uint32_t> uses;
        std::uint32_t counted = 0;
    };
    std::vector<PromptUses> prompt_;

    StreamLeases leases_;

    // VRAM monitor (design §19.3.7).
    using Clock = std::chrono::steady_clock;
    VramSizing sizing_;
    std::optional<VramControl> control_;
    std::unique_ptr<VramMonitor> monitor_;
    std::function<void()> waker_;
    std::function<bool()> maintenance_due_; // guarded by vram_mutex_, as waker_
    mutable std::mutex vram_mutex_;
    std::uint32_t frames_now_ = 0, startup_frames_ = 0;
    std::uint64_t pool_now_   = 0;
    Clock::time_point vram_epoch_{};
    bool pressure_warned_ = false, low_frames_warned_ = false;

    // Saved state.
    Clock::time_point state_saved_{};
    std::uint64_t state_routed_ = 0;

    std::unique_ptr<RouteTrace> trace_;
};

namespace testing {
// Fault injection for engine-level tests: the `checks`-th later check_error (one per prefill call or
// decode round, process-wide) reports a failed expert read of layer 0, as a real one would. 0 disarms.
void set_expert_fault(std::uint32_t checks) noexcept;
// Consumes one check; true on the armed one.
[[nodiscard]] bool take_expert_fault() noexcept;
} // namespace testing

} // namespace infernix::runtime::expert_offload
