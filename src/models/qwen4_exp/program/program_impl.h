#pragma once

#include "models/qwen4_exp/program/program.h"

#include "core/arena.h"
#include "core/cuda_vmm.h"
#include "core/cyclic_kv_cache.h"
#include "core/vmm_arena.h"
#include "core/decode_graph.h"
#include "core/round_clock.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/link_probe.h"
#include "core/linear_attention_state.h"
#include "core/nvtx.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "core/tensor.h"
#include "core/vram_budget.h"
#include "core/weight_view.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/ngram_proposer.h"
#include "models/qwen4_exp/execution/dflash2.h"
#include "models/qwen4_exp/execution/forward.h"
#include "runtime/speculative/draft_length.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/frontend/ngram_hash.h"
#include "models/qwen4_exp/expert_offload.h"
#include "runtime/expert_offload/placement.h"
#include "models/qwen4_exp/program/calibrated_order.h"
#include "runtime/expert_offload/drafter_pool.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "models/qwen4_exp/program/prefix/call_plan.h"
#include "models/qwen4_exp/program/prefix/prefix_cache.h"
#include "models/qwen4_exp/program/prefix/state_image.h"
#include "models/qwen4_exp/program/rope_positions.h"
#include "models/qwen4_exp/program/vision_window.h"
#include "models/qwen3_5/execution/vision_tower.h"
#include "models/qwen3_5/execution/vision_weight_stream.h"
#include "models/qwen3_5/program/vision_control.h"
#include "ops/offloaded_sparse_moe/cpu/fetch_channel.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "infernix/ops/argmax.h"
#include "infernix/ops/cast.h"
#include "infernix/ops/gdn_replay.h"
#include "infernix/ops/ple.h"
#include "infernix/ops/qsa.h"
#include "infernix/ops/sampling.h"
#include "infernix/ops/screened_head.h"
#include "infernix/ops/speculative_round.h"
#include "infernix/ops/token_constraint.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

namespace infernix::models::qwen4_exp {
namespace detail {

// A prompt with media: its items' control plan and the encode window it needs (design §19.3.2).
struct VisionAdmission {
    qwen3_5::VisionControlPlan control;
    VisionWindowPlan window;
};

// A TokenConstraint compiled for ops::constrain_logits: its distinct multi-token sets (written into
// the lane's set columns when it is first constrained) and each output step's descriptor relative to
// the lane (a set index, or the forced-token encoding -2 - token), and each step's observed tokens
// (empty when the request observes nothing).
struct ConstraintPlan {
    std::vector<std::vector<TokenId>> sets;
    std::vector<std::int32_t> steps;
    std::vector<std::vector<TokenId>> observe;
    std::uint64_t serial = 0; // unique per planned request: a lane's uploaded sets name their plan
};

struct BasePlanImpl {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::vector<TokenId> readout_tokens;               // read at the prompt's frontier (/v1/decide)
    std::shared_ptr<const ConstraintPlan> constraint;  // null: unconstrained
    std::uint32_t pages     = 0; // KV page groups the request needs at most (its extent)
    std::uint32_t positions = 0; // the extent: prompt + effective output, at most max_context
    bool reuse              = false; // the prefix cache may resume and publish this request
    std::shared_ptr<const VisionAdmission> vision;
    std::shared_ptr<const PreparedPrompt> prompt; // the request's prompt, moved in by plan_request
};

// The copy-on-write source of a resume (design §19.3.1, rule MR6): the snapshot's tail page (the
// frontier inside a block), the anchor block when its last MTP cell encodes another continuation,
// or none.
enum class PrefixCow : std::uint8_t { None, Tail, Anchor };

// A resume source chosen by the quote: a snapshot at `frontier` over `path`, or the root.
struct PrefixSelection {
    std::optional<runtime::prefix_cache::SnapshotRef> snapshot;
    std::vector<runtime::prefix_cache::NodeRef> path; // matched blocks up to the snapshot's anchor
    std::uint32_t frontier = 0;
    std::uint32_t shared   = 0; // leading blocks mapped from the cache
    PrefixCow cow          = PrefixCow::None;
    std::uint32_t need     = 0; // pages the admission newly takes (private + restored)
    bool image_restore     = false;
    std::vector<std::uint32_t> existing; // frontiers of every matched snapshot (tap spacing)
    std::uint32_t cached_tokens = 0;     // the prompt prefix held as cached blocks (diagnostics)
    prefix::CallPlan plan;               // the suffix's prefill calls and taps from `frontier`
};

// One lane's prefix-cache state.
struct LanePrefix {
    bool reuse = false;
    std::vector<runtime::prefix_cache::NodeRef> path; // pinned once each: mapped, then published blocks
    std::uint32_t page_base = 0;                       // logical block of lane.pages[0]
    std::uint32_t deepest   = 0;                       // frontier of the lineage's deepest snapshot
    std::uint32_t reused    = 0;
    std::optional<runtime::prefix_cache::SnapshotRef> resume;
    std::optional<prefix::RestoreTicket> restore;     // the first call waits on it per layer
    std::uint64_t capture          = 0;               // its copy-out gates the next call per layer
    // The current request has captured (its first capture superseded the resume snapshot). `capture`
    // may still name an earlier request's capture on this slot (its copy-out must land before the slot
    // is rewritten).
    bool captured                  = false;
    std::uint64_t resident_capture = 0;               // the endpoint the slot still holds
    std::vector<runtime::prefix_cache::PlannedTap> taps;
    std::vector<runtime::prefix_cache::TapHint> hints;
    std::vector<runtime::prefix_cache::TapExclusion> exclusions; // Vision spans: no snapshot inside
    // Block lookup hashes: the prompt's, extended by publication over the generated blocks; the
    // prompt's cumulative Vision keys, and the key every block after the prompt carries.
    std::vector<std::uint64_t> hashes, extras;
    std::uint64_t trailing_extra = 0;
    // The admission's diagnostics: the prompt prefix held as cached blocks, and the bytes its
    // restore copied back from the Host tier.
    std::uint32_t cached_tokens  = 0;
    std::uint64_t restored_bytes = 0;
    // What the current request published that a non-finite failure withdraws (prefix_withdraw;
    // hybrid-prefix-cache-spec §7.8): the blocks whose Device copy is its page, and its captures
    // before the latest (`capture`). `withdraw` marks that failure (set by the commit that releases
    // its row); `saves` counts the cache file's saves at admission.
    std::vector<runtime::prefix_cache::NodeRef> owned;
    std::vector<std::uint64_t> captures;
    // The resume snapshot its first capture superseded: the withdrawal restores it (superseded snapshots are
    // evicted first), so a resend resumes from it.
    std::optional<runtime::prefix_cache::SnapshotRef> superseded;
    bool withdraw       = false;
    std::uint64_t saves = 0;
};

struct QuoteImpl {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::uint32_t pages = 0;
    std::uint32_t lane  = 0;
    std::optional<PrefixSelection> prefix;
    std::shared_ptr<const VisionAdmission> vision;
};

// A media prompt's encode window and handoff (design §19.3.2): filled at reserve, encoded over
// admission steps, released once the prompt's last call is enqueued or the lane is released.
struct LaneVision {
    std::shared_ptr<const VisionAdmission> admission;
    qwen3_5::VisionControl control;            // the encoded items' controls
    std::vector<std::shared_ptr<const qwen3_5::PreparedMediaPayload>> payloads; // per encoded item
    std::vector<std::uint32_t> visual;         // the encoded visual tokens' positions, ascending
    ExpertResidency::FrameLease lease;         // the handoff (W), or handoff and window (L)
    Tensor handoff;                            // BF16 [out, V_encoded]
    std::unique_ptr<qwen3_5::execution::VisionWeightStream> stream;
    std::optional<qwen3_5::execution::VisionParameters> weights; // the stream's rebased view
    std::unique_ptr<qwen3_5::execution::VisionPassLayout> layout;
    std::unique_ptr<qwen3_5::execution::VisionTowerPass> pass;
    std::optional<CudaEventTimer> timer;
    bool encoded = false;
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
    static void set_row_failure(PendingBatch& p, std::size_t row, runtime::PendingRowFailure failure) noexcept {
        p.row_failures_[row] = failure;
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

// Per-call inputs staged through pinned memory, one device copy. Decode and verification rounds
// upload only the prefix before `ngram` plus their rows (io_prefix), so every array they read,
// the RoPE positions and pooled-block starts included, precedes `ngram`. `rope` holds a call's
// [columns, 3] axis-major RoPE positions, `block_rope` its [batch, 3] block starts; `mtp_rope` and
// `mtp_block_rope` the MTP chunk's per sub-chunk (execution::MtpChunk). `gate` is the U32 a gated
// verification round's rows wait for (design §12.3): it precedes `ngram`, so the round uploads it
// with the prefix and its rows go through the gate instead.
struct IoLayout {
    std::size_t ids = 0, positions = 0, rope = 0, block_rope = 0, slots = 0, rows = 0, columns = 0, gate = 0, divisor = 0,
                ngram = 0,
                mtp_ids = 0, mtp_cells = 0, mtp_rope = 0, mtp_block_rope = 0, vision_columns = 0,
                mtp_vision_columns = 0, bytes = 0;
};

// A request's verification rounds whose rows were read behind the n-gram gate, and those reads.
struct GateTraffic {
    std::uint64_t rounds = 0, reads = 0, read_ns = 0;
};
inline constexpr std::size_t kGateStatsBytes     = 16;          // per lane: U64 wait ns, U64 waits
inline constexpr std::uint64_t kSlowGatedReadNs  = 100000000;   // a gated round's reads warn past 100 ms

// Device I32 arrays of a verification round, each with room for every lane and width.
struct SpecLayout {
    std::size_t target = 0, drafts = 0, extents = 0, lengths = 0, anchors = 0, licensed = 0, counts = 0,
                accepted = 0, commit = 0, words = 0;
};

// I32 MTP drafter io: chain ids [B], cells [K, B], per draft step the cells' RoPE positions and
// pooled-block starts [K][3][B] (axis-major per step), the draft rows' state slots and table rows
// [B] each, the one-row round's FP32 cumulative threshold [1] (MtpCumulative),
// drafts [K, B], catch-up ids [W, B], gather columns [B], catch-up cells [W, B], the
// drafts' FP32 log-probabilities [K, B]. Everything before `drafts` is uploaded per draft round.
// Sampled drafts add, per draft step, the head's top ranks and their FP32 log-probabilities
// [kSampledDraftRanks, B] and the proposal's candidates and FP32 q [16, B] (step j at j * rows * B),
// then the verify layout of candidates and q [16, W - 1, B], then the draft rows' samplers
// (ops::SamplingConfig [B], 8-byte aligned). The draft's slots, rows and samplers are its own, so a
// draft never shares an input with the target's io.
struct MtpIo {
    std::size_t ids = 0, cells = 0, rope_cells = 0, block_rope_cells = 0, slots = 0, rows = 0, cumulative = 0,
                drafts = 0, up_ids = 0, gather = 0, up_cells = 0, logprobs = 0, sample_rows = 0, sample_logprobs = 0,
                draft_candidates = 0, draft_q = 0, verify_candidates = 0, verify_q = 0, configs = 0, words = 0;
};

// I32 DFlash2 drafter io (FP32 words where noted): per row its anchor, frontier, live block columns
// and ring lane [L] (L lanes; a round of B rows uses the first B of each); the block's ids and
// positions [W, L] (W = drafts + 1); the drafts [K, L] and their FP32 selector probabilities [K, L],
// their candidates and FP32 laws [16, K, L];
// the verification layout of candidates and laws [16, V - 1, L] (V the widest round: a round
// verifying fewer drafts than proposed, n-gram rows' one-hot laws); then a prefill call's, forced
// tokens' or layer walk's context positions [window], its count and its lane. Everything before `ids`
// is uploaded per proposal.
struct DFlash2Io {
    std::size_t anchors = 0, frontiers = 0, valid = 0, slots = 0, ids = 0, positions = 0, drafts = 0, confidence = 0,
                candidates = 0,
                q = 0, verify_candidates = 0, verify_q = 0, context_positions = 0, context_count = 0, context_slot = 0,
                words = 0;
};

// Sampled MTP drafts (sampled-mtp-drafts) draw from the draft head's four most probable tokens:
// measured against 8 and 16 at T 1.0 / top-p 0.95 / top-k 20 (design §19.3.24).
inline constexpr std::int32_t kSampledDraftRanks = 4;

// Every fixed device allocation of a Program and the layouts in it (design §19.3.7). One function
// (ProgramImpl::plan_device) computes it from the options and the configuration alone: before the
// weights are read, for the startup check, and in the constructor, which allocates exactly it.
struct DeviceLayout {
    std::int32_t lanes = 0, token_domain = 0, head_rows = 0, chunk = 0, mtp_k = 0, max_width = 1, columns = 0,
                 pages_per_row = 0, mtp_columns = 0;
    bool mtp = false;
    std::uint32_t kv_pages = 0, kv_layers = 0;
    std::uint64_t record_stride = 0; // the largest expert record
    std::uint64_t frame_bytes   = 0; // the expert cache's average frame
    LinearAttentionStatePoolLayout gdn;
    DeviceKVPagePoolLayout pool;
    KVExecutionTableLayout tables;
    GdnReplayRecordLayout records;
    IoLayout io;
    SpecLayout spec;
    MtpIo mtp_io;
    std::size_t gdn_bytes = 0, ple = 0, tails = 0, kv = 0, records_bytes = 0, ple_records = 0, qsa_records = 0;
    // The vq2/k4v2 exact window of every KV layer and lane (prefix::KvWindowGeometry; zero otherwise).
    std::size_t window = 0;
    std::size_t logits32 = 0, logits16 = 0, token_counts = 0, work = 0, staging = 0;
    std::int32_t staging_slots = 0; // misses staged per pass of a layer's experts
    // The arena wide prefill calls swap in (columns above static_columns), lent from expert frames
    // while a prompt runs such calls; zero when the static arena covers every call.
    std::size_t wide_work = 0;
    std::int32_t static_columns = 0;
    // Elastic KV (design §19.3.11): the backing reserves `kv` bytes of address space and maps the
    // chunks pages [0, n) of every plane (and the block tables) need; `kv_base_pages` are mapped at
    // startup and counted as fixed, the rest grows from the expert frames when admissions need it.
    bool kv_elastic             = false;
    std::uint32_t kv_base_pages = 0;
    std::size_t mtp_column_bytes = 0, mtp_ones = 0;
    // The drafter pool: the drafter pool's device bytes (runtime::expert_offload::DrafterExpertPool::device_bytes; 0: none).
    std::size_t drafter_pool = 0;
    // DFlash2 (docs/maintainer/dflash.md): drafts per proposal; the feature columns, a round region
    // (every column of a decode or verification round) then a call region (a prefill call's or layer
    // walk's last window), and their bytes; the context ring of every lane, the drafter's io and its
    // configuration.
    bool dflash2 = false;
    std::int32_t dflash_k = 0, dflash_round_columns = 0, dflash_columns = 0;
    std::size_t dflash_features = 0, dflash_ring = 0;
    CyclicKVCacheLayout dflash_ring_layout;
    DFlash2Io dflash_io;
    std::optional<DFlash2Config> dflash2_config;
    // Causal scoring: the window's final residuals (BF16 [S*H, max_context]), one tile's FP32 logits
    // and its targets, log-probabilities and distributions (ProgramImpl::causal_score).
    std::size_t score_residuals = 0, score_logits = 0, score_logits16 = 0, score_io = 0;
    ProgramDevicePlan bytes;
};

// Causal scoring's logits tile: 64 columns of FP32 logits (64 MB at 248,320 rows).
inline constexpr std::int32_t kScoreTileColumns = 64;
// The tile's I32/FP32 words: targets and their log-probabilities, then top ids and log-probabilities
// and candidate ids and log-probabilities of up to kMaximumScoreTopTokens per column.
inline constexpr std::size_t kScoreIoWords =
    static_cast<std::size_t>(kScoreTileColumns) * (2 + 4 * kMaximumScoreTopTokens);

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

// The cumulative cut of an argmax MTP chain (candidates §7 D8; Basalt draft_cum.hpp): the chain keeps
// draft j while the product over i <= j of the observed acceptance rate of draft i's probability bin
// stays at or above the request's threshold. The bins are Basalt's, as log-probability edges (the
// drafter's stored values, compared without exp, so host and device classify alike:
// core/decode_graph.h). The threshold starts at kMtpCumulativeBase and falls toward 0.25 as the
// request's drafts are accepted more often than the table predicts (observed / expected accepted,
// decayed 0.97 per round, mapped linearly over 0.9..1.1). Draft-only: verification and outputs are
// unchanged. It is the cut of `--draft-tokens auto` (Dense8 MTP auto, ABBA x5: conv 1.0529x, 5/5, worst 1.0296;
// struct 1.0009x); a fixed draft length keeps the --spec-min-p cut (MTP-3 under this cut: conv 1.0423x, struct
// 0.9919x, worse in every pair; 2026-10-28).
inline constexpr GraphGateBins kMtpCumulativeBins{
    {-1.20397280F, -0.693147181F, -0.356674944F, -0.223143551F, -0.105360516F, -0.0512932944F, -0.0100503359F},
    {0.12F, 0.26F, 0.43F, 0.51F, 0.59F, 0.66F, 0.79F, 0.96F}};
inline constexpr float kMtpCumulativeBase = 0.35F;

struct MtpCumulative {
    float threshold = kMtpCumulativeBase;
    double observed = 0.0, expected = 0.0;

    void reset() { *this = MtpCumulative{}; }
    // One verified chain: `offered` drafts with these log-probabilities, `accepted` of them taken.
    void observe(const float* logprobs, std::int32_t offered, std::int32_t accepted) {
        if (offered <= 0) { return; }
        constexpr double kDecay = 0.97, kLowRatio = 0.9, kHighRatio = 1.1;
        constexpr float kFloor  = 0.25F;
        double chance = 1.0, sum = 0.0;
        for (std::int32_t j = 0; j < offered; ++j) {
            chance *= static_cast<double>(graph_gate_rate(logprobs[j], kMtpCumulativeBins));
            sum += chance;
        }
        observed = observed * kDecay + static_cast<double>(accepted);
        expected = expected * kDecay + sum;
        if (expected < 0.5) { return; }
        const double t = std::clamp((observed / expected - kLowRatio) / (kHighRatio - kLowRatio), 0.0, 1.0);
        threshold      = kMtpCumulativeBase + static_cast<float>(t) * (kFloor - kMtpCumulativeBase);
    }
};

// A paused request (design §19.3.13): what its lane held besides model state. The pause published
// the state at `target` to the prefix cache (a request the cache publishes); a resume binds the
// ledger from the deepest cached state up to `target` and replays the rest without sampling.
struct ResumeStateImpl {
    std::uint32_t target = 0;   // the paused frontier: the positions the lane's state held
    bool generated       = false; // paused after its first token (the ledger runs past the prompt)
    std::vector<std::int32_t> history; // prompt, then committed output
    std::uint32_t prompt_tokens = 0;
    // Prefix-cache lookup keys over the ledger (the prompt's, extended over published output
    // blocks), and the capture the pause published.
    std::vector<std::uint64_t> hashes;
    std::uint64_t capture = 0;
    runtime::BeginSummary begin;
    std::uint64_t prefill_ns = 0, decode_ns = 0, decode_share_ns = 0;
    double vision_seconds = 0;
    SpeculativeStats speculative;
    std::array<double, 8> mtp_accept{};
    std::uint64_t mtp_policy_rounds = 0;
    MtpCumulative mtp_cumulative;
    runtime::speculative::DraftCalibration dflash_calibration; // DFlash2: acceptance by selector-probability decile
    std::array<double, 4> ngram_accept{}; // the n-gram copies' acceptance per match-length bucket
    std::vector<ConstrainedDraw> constraint_trace;
    ExpertResidency::Stats cache_at_admission;
    std::uint64_t cpu_served_at_admission = 0;
    NgramVolume::Counters ngram;
    GateTraffic gate;
};

class ProgramImpl {
public:
    enum class Phase : std::uint8_t { Free, Prefill, Decode, Finishable };

    struct Lane {
        Phase phase          = Phase::Free;
        std::uint64_t epoch  = 0;
        runtime::BeginSummary begin; // the Begin of the binding
        std::vector<std::int32_t> history; // prompt, then committed output
        std::uint32_t prompt_tokens = 0;
        std::uint32_t state_tokens  = 0; // positions already in the model state
        // Where the prefill phase ends: the prompt, or a resumed lane's paused frontier, which its
        // replay reaches without sampling (design §19.3.13).
        std::uint32_t prefill_end = 0;
        bool replay               = false;
        std::uint32_t extent      = 0; // the positions the request may ever hold (prompt + output)
        // The prompt's planned prefill calls (design §19.3.1): exclusive ends, the last prompt_tokens.
        std::vector<std::uint32_t> calls;
        std::size_t next_call = 0;
        LaneRope rope;                    // RoPE positions of the lane's tokens (design §19.3.2)
        std::unique_ptr<LaneVision> vision; // the encode window and handoff of a media prompt
        double vision_seconds = 0;          // the tower's GPU time
        // The tower's timer when its stop had not completed at release: read later
        // (vision_seconds_of), never waited for.
        std::optional<CudaEventTimer> vision_timer;
        ops::SamplingConfig sampling;
        // /v1/decide: the tokens the prompt's readout names, the constraint, the draws of the pending
        // round's tokens (one per licensed token) and the committed trace they move into.
        std::vector<TokenId> readout;
        std::shared_ptr<const ConstraintPlan> constraint;
        std::shared_ptr<const PreparedPrompt> prompt; // a waiting sibling compares its media
        std::vector<ConstrainedDraw> constraint_round, constraint_trace;
        std::optional<DeviceKVPageReservation> reservation;
        std::vector<DeviceKVPageLease> pages;
        KVExecutionRowLease row;
        std::uint64_t prefill_ns      = 0;
        std::uint64_t decode_ns       = 0;
        std::uint64_t decode_share_ns = 0;
        ExpertResidency::Stats cache_at_admission;
        std::uint64_t cpu_served_at_admission = 0;
        HostExpertTier::Stats tier_at_admission{}; // SSD tier mode
        expert_cache::HostTier::Losses losses_at_admission{};
        NgramVolume::Counters ngram; // this request's n-gram row traffic
        GateTraffic gate;            // its verification rounds' rows read behind the n-gram gate
        std::unique_ptr<qwen3_5::detail::NgramProposer> proposer; // copy proposals over history
        SpeculativeStats speculative;
        // MTP drafter (design 11.2; rule MR1 of 19.3.1): its KV cells [0, mtp_cells) are final. At
        // every settled point the lane's saved column holds the residual of position
        // state_tokens - 1, whose cell is written (mtp_cells == state_tokens) or pending
        // (state_tokens - 1: the next call or round writes it).
        std::uint32_t mtp_cells = 0;
        bool mtp_live           = false;
        // DFlash2: the context ring holds every position below this one (state_tokens at every
        // settled point, except after a resume from a tap inside a prefill layer walk).
        std::uint32_t dflash_frontier = 0;
        // Draft-length policy (design 11.3): conditional acceptance of each draft position.
        std::array<double, 8> mtp_accept{};
        std::uint64_t mtp_policy_rounds = 0;
        MtpCumulative mtp_cumulative; // the argmax chains' cumulative cut
        // DFlash2 (--draft-tokens auto): a draft's acceptance (given the earlier ones') by its
        // selector probability's decile.
        runtime::speculative::DraftCalibration dflash_calibration;
        // Conditional acceptance of an n-gram copy
        // position by the proposal's match length (kNgramBuckets).
        std::array<double, 4> ngram_accept{};
        LanePrefix prefix;
        // Prompt seeding (design §19.3.27): of the prompt's prefilled tokens counted, how many
        // routed to each (layer, expert); [layers][experts], allocated on first use.
        std::vector<std::uint32_t> prompt_uses;
        std::uint32_t prompt_counted = 0;
    };

    // Everything the Program allocates on the device besides the expert frames; validates the
    // options.
    static DeviceLayout plan_device(const ProgramOptions& o, const Config& config, std::int32_t public_tokens) {
        const TextConfig& c = config.text;
        if (o.max_concurrency == 0 || o.max_concurrency > kMaximumConcurrency) {
            throw std::invalid_argument("Qwen4Exp: max_concurrency must be in [1,8]");
        }
        if (o.max_context == 0 || o.prefill_chunk == 0) {
            throw std::invalid_argument("Qwen4Exp: max_context and prefill_chunk must be nonzero");
        }
        if (o.max_context > c.context_ceiling()) {
            throw std::invalid_argument("Qwen4Exp: max_context " + std::to_string(o.max_context) + " exceeds the " +
                                        std::to_string(c.context_ceiling()) +
                                        " positions of the model (native positions x the YaRN factor)");
        }
        if (o.ngram_draft_tokens > 15 ||
            (o.ngram_draft_tokens > 0 && (o.ngram_min_match < 4 || o.ngram_min_match > 64))) {
            throw std::invalid_argument("Qwen4Exp: n-gram drafts must be 0..15 with a minimum match of 4..64");
        }
        if (config.mtp && (o.mtp_draft_tokens == 0 || o.mtp_draft_tokens > 7)) {
            throw std::invalid_argument("Qwen4Exp: MTP needs 1..7 draft tokens");
        }
        // Verification widths up to 8 keep the dense routes column-invariant (design 11.3).
        if (config.dflash2 && (o.dflash2_draft_tokens == 0 || o.dflash2_draft_tokens > 7 || config.mtp)) {
            throw std::invalid_argument("Qwen4Exp: DFlash2 needs 1..7 draft tokens and no MTP drafter");
        }
        // Its drafts are verified with their laws (sparse acceptance, which serves one token domain).
        if (config.dflash2 &&
            !sparse_acceptance(public_tokens, 1 + std::max(static_cast<std::int32_t>(o.ngram_draft_tokens),
                                                          static_cast<std::int32_t>(o.dflash2_draft_tokens)),
                               static_cast<std::int32_t>(o.max_concurrency))) {
            throw std::invalid_argument("Qwen4Exp: DFlash2 needs the sparse acceptance's token domain");
        }
        DeviceLayout d;
        d.lanes         = static_cast<std::int32_t>(o.max_concurrency);
        d.token_domain  = public_tokens;
        d.mtp           = config.mtp;
        d.head_rows     = config.proposal_rows != 0 ? dim(config.proposal_rows) : dim(c.vocab_size);
        d.chunk         = static_cast<std::int32_t>(std::min(o.prefill_chunk, o.max_context));
        // Without free cache resume every prefill call starts on the GDN prefill's 16-token grid.
        if (!o.free_cache_resume) {
            constexpr auto grid = static_cast<std::int32_t>(kGdnGrid);
            d.chunk             = std::max(grid, d.chunk / grid * grid);
        }
        d.mtp_k         = d.mtp ? static_cast<std::int32_t>(o.mtp_draft_tokens) : 0;
        d.dflash2       = config.dflash2.has_value();
        d.dflash_k      = d.dflash2 ? static_cast<std::int32_t>(o.dflash2_draft_tokens) : 0;
        d.dflash2_config = config.dflash2;
        d.max_width     = 1 + std::max({static_cast<std::int32_t>(o.ngram_draft_tokens), d.mtp_k, d.dflash_k});
        d.columns       = std::max(d.chunk, d.lanes * d.max_width);
        d.pages_per_row = (dim(o.max_context) + kPagedKVPageSize - 1) / kPagedKVPageSize;
        const auto kv_tokens = o.kv_capacity_tokens != 0 ? o.kv_capacity_tokens : o.max_context * o.max_concurrency;
        d.kv_pages      = (kv_tokens + kPagedKVPageSize - 1) / kPagedKVPageSize;
        const std::int32_t lanes = d.lanes, W = d.max_width;
        // One copy-on-write page per lane, so a whole-extent resume always fits (design §19.3.1).
        if (o.prefix_cache) { d.kv_pages += static_cast<std::uint32_t>(lanes); }
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
                                                                    .recurrent_dtype = o.fp16_gdn_state ? DType::FP16
                                                                                                        : DType::FP32,
                                                                });
        d.gdn_bytes = state_builder.finish(256);
        d.ple       = width * c.ple.conv_span() * lanes * 2;
        d.kv_layers = c.attention_layers + (d.mtp ? 1U : 0U);
        d.tails     = d.kv_layers * di * (r - 1) * lanes * 2;
        d.window    = prefix::kv_window_geometry(c, o.kv_cache, d.mtp, static_cast<std::uint32_t>(lanes)).bytes();

        // Paged KV: one page group holds 64 positions of every attention layer.
        // The page group's planes, in the order the prefix cache's records also use.
        const prefix::KvPageGeometry kv_geometry = prefix::kv_page_geometry(c, o.kv_cache, d.mtp);
        LayoutBuilder kv_builder;
        d.pool   = plan_device_kv_page_pool(kv_builder, {.page_group_count = d.kv_pages, .geometry = kv_geometry.geometry});
        d.tables = plan_kv_execution_tables(
            kv_builder, {.logical_page_capacity = static_cast<std::uint32_t>(d.pages_per_row), .table_rows = lanes});
        d.kv = kv_builder.finish(256);
        // Pages [0, n) are a prefix of each page-major plane, so growth maps whole chunks at each plane's end.
        d.kv_elastic = d.pool.spec.geometry.device_plane_order == PagedKVPlaneOrder::PageMajor;
        d.kv_base_pages = std::min<std::uint32_t>(d.kv_pages, kKvBaseTokens / kPagedKVPageSize);

        // Per-call inputs, logits and sampling counts.
        const std::size_t heads     = NgramHash(c.ple.ngram, 0).heads();
        const std::size_t row_bytes = (o.ngram_table ? *o.ngram_table : c.ple.table).row_bytes;
        const std::size_t columns   = static_cast<std::size_t>(d.columns);
        const std::size_t mtp_subchunks = (columns + 1 + execution::kMtpChunkColumns - 1) / execution::kMtpChunkColumns;
        d.io.ids            = 0;
        d.io.positions      = align(d.io.ids + 4ULL * columns);
        d.io.rope           = align(d.io.positions + 4ULL * columns);
        d.io.block_rope     = align(d.io.rope + 12ULL * columns);
        d.io.slots          = align(d.io.block_rope + 12ULL * lanes);
        d.io.rows           = align(d.io.slots + 4ULL * lanes);
        d.io.columns        = align(d.io.rows + 4ULL * lanes);
        d.io.gate           = align(d.io.columns + 4ULL * columns);
        d.io.divisor        = d.io.gate + 4; // I32: plain decode's one-column CPU divisor this round
        d.io.ngram          = align(d.io.divisor + 4);
        d.io.mtp_ids        = align(d.io.ngram + row_bytes * heads * columns);
        d.io.mtp_cells      = align(d.io.mtp_ids + 4ULL * (columns + 1));
        d.io.mtp_rope       = align(d.io.mtp_cells + 4ULL * (columns + 1));
        d.io.mtp_block_rope = align(d.io.mtp_rope + 12ULL * (columns + 1));
        d.io.vision_columns     = align(d.io.mtp_block_rope + 12ULL * mtp_subchunks);
        d.io.mtp_vision_columns = align(d.io.vision_columns + 4ULL * columns);
        d.io.bytes              = align(d.io.mtp_vision_columns + 4ULL * (columns + 1));
        if (d.io.block_rope + 12ULL * lanes > d.io.ngram || d.io.columns + 4ULL * columns > d.io.ngram) {
            throw std::logic_error("Qwen4Exp: decode io must precede the n-gram rows");
        }
        // The head's FP32 logits, which the sampler, verification, constraints and the prompt readout read; with
        // BF16 logits their BF16 rounding beside them (the lossless head writes only that).
        const bool bf16_head = o.bf16_logits && c.lossless;
        d.logits32     = bf16_head ? 0 : sizeof(float) * c.vocab_size * lanes * W;
        d.logits16     = o.bf16_logits ? 2ULL * c.vocab_size * lanes * W : 0;
        d.token_counts = 4ULL * static_cast<std::size_t>(d.token_domain) * lanes;
        if (o.causal_scoring) {
            d.score_residuals = align(2ULL * width * o.max_context);
            d.score_logits    = bf16_head ? 0 : align(sizeof(float) * c.vocab_size * kScoreTileColumns);
            d.score_logits16  = o.bf16_logits ? align(2ULL * c.vocab_size * kScoreTileColumns) : 0;
            d.score_io        = align(4ULL * kScoreIoWords);
        }

        // The workspace arena (the forward pass, sampling, draft acceptance and the drafter) covers
        // decode, verification and calls of up to kStaticWorkColumns columns; a wider prefill call
        // swaps in an arena of the chunk's size lent from expert frames (design §19.3.7, VRAM item 1).
        d.static_columns = std::max(lanes * W, std::min(d.chunk, kStaticWorkColumns));
        d.work           = workspace_bytes(c, o, d, lanes, d.static_columns);
        d.wide_work      = d.chunk > d.static_columns ? workspace_bytes(c, o, d, lanes, d.chunk) : 0;

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
            d.mtp_io.ids              = 0;
            d.mtp_io.cells            = d.mtp_io.ids + lanes;
            d.mtp_io.rope_cells       = d.mtp_io.cells + lanes * k;
            d.mtp_io.block_rope_cells = d.mtp_io.rope_cells + 3 * lanes * k;
            d.mtp_io.slots            = d.mtp_io.block_rope_cells + 3 * lanes * k;
            d.mtp_io.rows             = d.mtp_io.slots + lanes;
            // The cumulative cut: the one-row round's FP32 threshold (uploaded with the round).
            d.mtp_io.cumulative       = d.mtp_io.rows + lanes;
            d.mtp_io.drafts           = d.mtp_io.cumulative + 1;
            d.mtp_io.up_ids    = d.mtp_io.drafts + lanes * k;
            d.mtp_io.gather    = d.mtp_io.up_ids + static_cast<std::size_t>(lanes) * W;
            d.mtp_io.up_cells  = d.mtp_io.gather + lanes;
            d.mtp_io.logprobs  = d.mtp_io.up_cells + static_cast<std::size_t>(lanes) * W;
            // Sampled drafts (sampled-mtp-drafts; planned whether or not it runs, a few KiB).
            const auto ranks      = static_cast<std::size_t>(kSampledDraftRanks);
            const auto candidates = static_cast<std::size_t>(ops::kSparseSpeculativeCandidates);
            const auto verified   = static_cast<std::size_t>(lanes) * static_cast<std::size_t>(W - 1);
            d.mtp_io.sample_rows       = d.mtp_io.logprobs + lanes * k;
            d.mtp_io.sample_logprobs   = d.mtp_io.sample_rows + ranks * lanes * k;
            d.mtp_io.draft_candidates  = d.mtp_io.sample_logprobs + ranks * lanes * k;
            d.mtp_io.draft_q           = d.mtp_io.draft_candidates + candidates * lanes * k;
            d.mtp_io.verify_candidates = d.mtp_io.draft_q + candidates * lanes * k;
            d.mtp_io.verify_q          = d.mtp_io.verify_candidates + candidates * verified;
            static_assert(alignof(ops::SamplingConfig) <= 8 && sizeof(ops::SamplingConfig) % 8 == 0);
            d.mtp_io.configs = (d.mtp_io.verify_q + candidates * verified + 1) / 2 * 2;
            d.mtp_io.words   = d.mtp_io.configs + sizeof(ops::SamplingConfig) / 4 * lanes;
            // The drafter pool: the drafter's pooled experts (their banks are pinned, not in the
            // model's device bytes).
            if (config.mtp_expert_slots > 0) {
                d.drafter_pool = runtime::expert_offload::DrafterExpertPool::device_bytes(config.mtp_expert_bytes, dim(config.mtp_expert_slots),
                                                                 dim(c.moe.experts));
            }
        }

        // The DFlash2 drafter: the target features of every verification column or a call's last
        // window, every lane's context ring and the drafter's io.
        if (d.dflash2) {
            const DFlash2Config& f = *config.dflash2;
            d.dflash_round_columns = lanes * W;
            d.dflash_columns       = d.dflash_round_columns + dim(f.window);
            d.dflash_features      = align(2ULL * f.feature_width(c.hidden_size) * static_cast<std::size_t>(d.dflash_columns));
            LayoutBuilder ring_builder;
            d.dflash_ring_layout = plan_cyclic_kv_cache(ring_builder, f.layers, f.window, dim(f.attention.kv_heads),
                                                        dim(f.attention.head_dim), lanes);
            d.dflash_ring        = ring_builder.finish(256);
            const auto L = static_cast<std::size_t>(lanes), K = static_cast<std::size_t>(d.dflash_k);
            const auto C = static_cast<std::size_t>(ops::kSparseSpeculativeCandidates);
            auto take = [at = std::size_t{0}](std::size_t words) mutable {
                const std::size_t begin = at;
                at += (words + 63) / 64 * 64;
                return begin;
            };
            d.dflash_io.anchors           = take(L);
            d.dflash_io.frontiers         = take(L);
            d.dflash_io.valid             = take(L);
            d.dflash_io.slots             = take(L);
            d.dflash_io.ids               = take((K + 1) * L);
            d.dflash_io.positions         = take((K + 1) * L);
            d.dflash_io.drafts            = take(K * L);
            d.dflash_io.confidence        = take(K * L);
            d.dflash_io.candidates        = take(C * K * L);
            d.dflash_io.q                 = take(C * K * L);
            d.dflash_io.verify_candidates = take(C * static_cast<std::size_t>(W - 1) * L);
            d.dflash_io.verify_q          = take(C * static_cast<std::size_t>(W - 1) * L);
            d.dflash_io.context_positions = take(f.window);
            d.dflash_io.context_count     = take(1);
            d.dflash_io.context_slot      = take(1);
            d.dflash_io.words             = take(0);
        }

        // Staging slots for each layer call's misses (design 8.6), one expert record each.
        const std::uint64_t shape[] = {c.moe.experts, c.hidden_size, c.moe.intermediate};
        // The largest record (staging, stream-ring and RAM slots): the NVFP4 record's, or an exact
        // bank's largest per-layer record; and the expert cache's average frame (its sizing unit).
        d.record_stride = c.expert_frame_stride != 0
                              ? c.expert_frame_stride
                              : weight_geometry(QType::NVFP4_MUL, QuantLayout::ExpertRg16, shape).record_stride;
        d.frame_bytes   = c.expert_frame_bytes != 0 ? c.expert_frame_bytes : d.record_stride;
        // Without the SSD tier, a verification call's misses (up to top_k per column) stage in one
        // pass, so it keeps the fork route (ops::moe_experts): eight columns (seven drafts) route up
        // to 80 experts, and 80 slots ran MTP 4 + n-gram 7 on the IQ3_S import at 204.0 tok/s
        // against 190.3 with 64 (+7.2 %, 3 runs, 2026-10-16). With the tier the larger staging and
        // its ring took RAM and frames from the cache: the FP8 import with n-gram 7 ran 48.6-49.8
        // against 51.2 with 64 (SSD reads 59.0 against 35.2 GiB), so a tier short of RAM keeps 64. The
        // exclusive layout's tier holds every expert VRAM does not, and with 64 slots its eight-column
        // verification calls lost the fork: IQ3_S struct decode 363.7 -> 419.4 tok/s with 80 (2026-10-28).
        // INFERNIX_TMP_STAGING_SLOTS fixes the count (a temporary A/B: n-gram 15 asks for 160 slots).
        const std::int32_t fixed_slots = temporary_switches::offload().staging_slots;
        d.staging_slots = fixed_slots > 0                                     ? fixed_slots
                          : o.expert_ram_bytes != 0 && !o.expert_ram_exclusive ? kStagingSlots
                                                                               : std::clamp(static_cast<std::int32_t>(c.moe.top_k) * d.max_width,
                                                                                            kStagingSlots, kMaxStagingSlots);
        d.staging       = static_cast<std::size_t>(d.staging_slots) * d.record_stride;

        auto& b     = d.bytes;
        b.kv_pages  = d.kv_pages;
        b.kv        = d.kv_elastic ? kv_mapped_bytes(d, d.kv_base_pages) : d.kv;
        b.kv_max    = d.kv;
        b.workspace = d.work;
        b.staging   = d.staging;
        b.state     = d.gdn_bytes + d.ple + d.tails + d.window + d.records_bytes + d.ple_records + d.qsa_records +
                  (d.mtp ? d.mtp_column_bytes * lanes * (W + 2) + 2ULL * di * W * lanes : 0) + d.dflash_features +
                  d.dflash_ring + d.drafter_pool;
        b.io = d.io.bytes + d.logits32 + d.logits16 + 4ULL * lanes + 4ULL * lanes +
               sizeof(ops::SamplingConfig) * lanes + d.token_counts + 4ULL * d.spec.words + d.mtp_ones +
               4ULL * d.mtp_io.words + 4ULL * d.dflash_io.words + kGateStatsBytes * lanes +
               (o.ngram_table && o.ngram_table->fp8_scale ? sizeof(std::uint16_t) : 0) +
               (o.ngram_table && o.ngram_table->nvfp4_scale ? sizeof(std::uint32_t) : 0) + constraint_bytes(lanes, W) +
               grammar_mask_bytes(d.token_domain, lanes, W) + d.score_residuals + d.score_logits + d.score_logits16 +
               d.score_io;
        b.residency            = ExpertResidency::table_bytes(expert_layout(c), d.columns);
        b.expert_record_stride = d.record_stride;
        b.expert_tier_list_slots =
            tier_ring_slots(o, d.staging_slots) + HostExpertTier::Options{}.prefetch + HostExpertTier::Options{}.demotion;
        b.expert_tier_stream_slots = o.prefill_stream && o.expert_cache ? kStreamReadSlots : 0;
        b.expert_frame_bytes   = d.frame_bytes;
        b.max_frames           = ExpertResidency::max_frames(expert_layout(c));
        b.graph_bound          = graph_bound(o.max_concurrency, static_cast<std::uint32_t>(W),
                                             static_cast<std::uint32_t>(d.mtp_k), o.sampled_mtp_drafts && d.mtp,
                                             static_cast<std::uint32_t>(d.dflash_k));
        // screened-lm-head's rounds keep their own decode and verification graphs.
        if (o.screened_lm_head) { b.graph_bound += o.max_concurrency * static_cast<std::uint32_t>(W + 1); }
        // A commit graph per batch size and width of the speculative rounds (commit_verified).
        if (W > 1) { b.graph_bound += o.max_concurrency * static_cast<std::uint32_t>(W); }
        return d;
    }

    // The SSD tier's ring: a call's CPU jobs hold their ring slots until the call's answer, and a pass's
    // fetched records until the pass has copied them, so it holds both with a pass to spare.
    static std::uint32_t tier_ring_slots(const ProgramOptions& o, std::int32_t staging_slots) {
        const std::uint32_t cpu_cap =
            o.cpu_expert_workers > 0 && o.cpu_expert_jobs > 0
                ? std::min<std::uint32_t>(std::max(o.cpu_expert_jobs, o.cpu_assist_jobs),
                                          static_cast<std::uint32_t>(ops::offloaded_moe::kMaxCpuJobs))
                : 0U;
        return std::max<std::uint32_t>(128, cpu_cap + 2U * static_cast<std::uint32_t>(staging_slots));
    }

    // The chunks of the KV backing that pages [0, pages) of every plane and the block tables need.
    static std::vector<std::uint8_t> kv_chunks(const DeviceLayout& d, std::uint32_t pages) {
        std::vector<std::uint8_t> need((d.kv + kKvChunkBytes - 1) / kKvChunkBytes, 0);
        const auto mark = [&](std::size_t offset, std::size_t bytes) {
            if (bytes == 0) { return; }
            for (std::size_t c = offset / kKvChunkBytes; c <= (offset + bytes - 1) / kKvChunkBytes; ++c) { need[c] = 1; }
        };
        for (const auto& plane : d.pool.planes) { // page-major: the page is the outermost dimension
            const auto total = static_cast<std::size_t>(d.pool.spec.page_group_count);
            mark(plane.storage.region.offset, plane.storage.region.bytes / total * pages);
        }
        mark(d.tables.block_tables.region.offset, d.tables.block_tables.region.bytes);
        return need;
    }
    static std::size_t kv_mapped_bytes(const DeviceLayout& d, std::uint32_t pages) {
        const auto need = kv_chunks(d, pages);
        return static_cast<std::size_t>(std::count(need.begin(), need.end(), std::uint8_t{1})) * kKvChunkBytes;
    }

    // 16-bit-activation experts' wide route under these options, which sizes the workspace: --a8-activations'
    // (GGUF and NVFP4 W4A16 banks; the exact layout), the expert-a4-residual fast paths' (NVFP4 W4A16 banks, two
    // or three terms) or ggml-bf16-prefill's (GGUF banks' BF16 stage-decode tiles; the exact layout), in the
    // Forward's order.
    static ops::offloaded_moe::WideA16 wide_a16(const ProgramOptions& o) noexcept {
        if (o.fast_paths.a8_experts_wide) { return ops::offloaded_moe::WideA16::kA8; }
        if (o.fast_paths.expert_a4_terms == 3) { return ops::offloaded_moe::WideA16::kA4Residual3; }
        if (o.fast_paths.expert_a4_terms == 2) { return ops::offloaded_moe::WideA16::kA4Residual; }
        return ops::has_route(o.fast_paths.prefill_routes, ops::LinearRoutes::GgmlBf16Wide) ? ops::offloaded_moe::WideA16::kGgmlBf16
                                                                                            : ops::offloaded_moe::WideA16::kExact;
    }

    // The CPU expert engine's W4A16 arithmetic: expert-a4-residual-decode's two terms on NVFP4 W4A16
    // banks when it is selected, else --a8-activations' 8-bit route (the Forward gives those banks' GPU narrow
    // calls the same), else exact.
    ops::offloaded_moe::NarrowA16 cpu_narrow_a16() const noexcept {
        const auto& bank     = *parameters_.layers.front().moe.bank;
        const bool nvfp4_a16 = bank.activation == ops::offloaded_moe::ExpertActivation::kA16 &&
                               bank.planes.gate_up_format == QType::NVFP4_MUL;
        return nvfp4_a16 && options_.fast_paths.expert_a4_narrow ? ops::offloaded_moe::NarrowA16::kA4Residual
               : nvfp4_a16 && options_.fast_paths.a8_experts ? ops::offloaded_moe::NarrowA16::kA8
                                                             : ops::offloaded_moe::NarrowA16::kExact;
    }

    // --a8-activations on a GGUF import: the CPU expert engine computes the 8-bit route's requests (the Forward
    // gives the banks' calls MoeGgmlBank::a8 with the same choice).
    bool cpu_ggml_a8() const noexcept {
        const auto& bank = *parameters_.layers.front().moe.bank;
        return options_.fast_paths.a8_experts && bank.planes.gate_up_format != QType::NVFP4_MUL &&
               bank.planes.gate_up_format != QType::FP8_E4M3FN_B128;
    }

    // The workspace arena of a Program with `lanes` lanes and the plan's widths: the forward at its
    // widest call (a prefill chunk or every lane's widest round), sampling, acceptance and the
    // drafter. The plan sizes the arena with the Program's lane count; the per-lane report compares
    // lane counts.
    static std::size_t workspace_bytes(const TextConfig& c, const ProgramOptions& o, const DeviceLayout& d,
                                       std::int32_t lanes, std::int32_t columns) {
        const std::int32_t W = d.max_width;
        std::size_t bytes =
            execution::Forward::workspace_bytes(c, std::max(columns, lanes * W), dim(o.max_context), o.kv_cache,
                                                wide_a16(o)) +
            ops::sampling_workspace_capacity_bytes(d.token_domain, 1, lanes);
        // A prefill projection's input encoded for its prefill route, K up to the GDN/QSA value width (the output
        // projections' input): f16-accumulate-prefill's FP16 columns, or the two E4M3 planes of --a8-activations'
        // block-scaled pair (MXFP8 dense, in place of f16-accumulate-prefill).
        {
            const std::size_t k = std::max<std::size_t>({std::size_t(c.hidden_size), std::size_t(c.gdn.value_width()),
                                                         std::size_t(c.attention.query_width())});
            if (ops::has_route(o.fast_paths.prefill_routes, ops::LinearRoutes::F16Accumulate)) {
                bytes += ops::linear_f16_accumulate_workspace_bytes(dim(k), std::max(columns, lanes * W));
            }
            if (ops::has_route(o.fast_paths.prefill_routes, ops::LinearRoutes::Fp8A8PairBlock32)) {
                bytes += ops::linear_fp8_a8_pair_block32_workspace_bytes(dim(k), std::max(columns, lanes * W));
            }
        }
        if (W > 1) {
            std::size_t accept =
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(d.token_domain, 1, W - 1, 1, lanes);
            if (((o.sampled_mtp_drafts && d.mtp) || d.dflash2) && sparse_acceptance(d.token_domain, W, lanes)) {
                accept = std::max(accept, ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                                              d.token_domain, ops::SpeculativeAcceptExecutionEnvelope{}, 1, W - 1, 1,
                                              lanes));
            }
            bytes += accept;
        }
        if (d.mtp) {
            bytes += execution::Forward::mtp_workspace_bytes(c, std::max(lanes * W, std::min(d.chunk, 512)), lanes,
                                                             d.head_rows, dim(o.max_context), o.kv_cache);
        }
        // The lossless route's calls: a scored window, or (generation) the cache form over the lanes.
        if (c.lossless) {
            bytes = std::max(bytes, execution::Forward::lossless_workspace_bytes(
                                        c, std::max(columns, lanes * W), o.causal_scoring ? 0 : dim(o.max_context), lanes) +
                                        ops::sampling_workspace_capacity_bytes(d.token_domain, 1, lanes));
        }
        // The DFlash2 drafter runs between Forward calls, in the same arena.
        if (d.dflash2) {
            bytes = std::max(bytes, execution::DFlash2::workspace_bytes(c, *d.dflash2_config, lanes, d.dflash_k + 1,
                                                                        dim(d.dflash2_config->window), o.max_context));
        }
        return bytes;
    }

    // Whether the sparse acceptance Op (sampled drafts' verifier) serves this token domain and these
    // verify widths and row counts.
    static bool sparse_acceptance(std::int32_t token_domain, std::int32_t W, std::int32_t lanes) {
        try {
            (void)ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                token_domain, ops::SpeculativeAcceptExecutionEnvelope{}, 1, W - 1, 1, lanes);
            return true;
        } catch (const std::invalid_argument&) {
            return false;
        }
    }

    static void allocate(DeviceBuffer& buffer, std::size_t bytes, std::uint64_t& tally) {
        buffer = DeviceBuffer(bytes);
        tally += bytes;
    }

    // The startup failure for device allocations the driver placed in system memory.
    static std::string spill_message(std::uint64_t bytes) {
        return std::to_string(bytes >> 20) +
               " MiB of Infernix's VRAM allocations were placed in shared system memory by the driver's sysmem "
               "fallback: free VRAM, or lower --max-context, --max-concurrency or --prefill-chunk. Setting the NVIDIA "
               "Control Panel 'CUDA - Sysmem Fallback Policy' to 'Prefer No Sysmem Fallback' for this program makes "
               "this fail at allocation instead.";
    }

    ProgramImpl(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options)
        : parameters_(parameters), device_(device), c_(parameters.model.config().text), options_(std::move(options)),
          volume_(options_.ngram_volume, ngram_table(), options_.ngram_in_ram), hash_(c_.ple.ngram, 0),
          plan_(plan_device(options_, parameters.model.config(), dim(parameters.model.resources().public_token_count))) {
        const auto& head = parameters_.draft_head;
        mtp_             = parameters_.mtp.has_value();
        dflash2_         = parameters_.dflash2.has_value();
        if (dflash2_ != plan_.dflash2) { throw std::logic_error("Qwen4Exp: the bound DFlash2 drafter differs from its configuration"); }
        if (mtp_ != plan_.mtp || (mtp_ && (head.rows ? dim(head.rows->weight.n) : dim(c_.vocab_size)) != plan_.head_rows)) {
            throw std::logic_error("Qwen4Exp: the bound drafter differs from its configuration");
        }
        const std::int32_t lanes = plan_.lanes;
        vocab_         = dim(c_.vocab_size);
        token_domain_  = plan_.token_domain;
        chunk_         = plan_.chunk;
        mtp_k_         = plan_.mtp_k;
        dflash_k_      = plan_.dflash_k;
        max_width_     = plan_.max_width;
        columns_       = plan_.columns;
        kv_pages_      = plan_.kv_pages;
        width_         = dim(c_.residual_width());
        span_          = dim(c_.ple.conv_span());
        di_            = dim(c_.qsa.index_head_dim);
        r_             = dim(c_.qsa.compress_ratio);
        draft_min_logprob_ = options_.draft_min_p > 0.0 ? std::log(options_.draft_min_p)
                                                        : -std::numeric_limits<double>::infinity();
        sampled_drafts_ = options_.sampled_mtp_drafts && plan_.mtp && max_width_ > 1 &&
                          sparse_acceptance(plan_.token_domain, max_width_, plan_.lanes);
        if (options_.sampled_mtp_drafts && plan_.mtp && !sampled_drafts_) {
            diagnostic("sampled MTP drafts need the sparse acceptance's token domain; drafts stay one-hot",
                       DiagnosticLevel::Warning);
        }
        walk_step_tokens_ = walk_step_tokens_from_environment();
        dflash_process_calibration_.reset();
        screened_head_ = options_.screened_lm_head;
        if (screened_head_ && !parameters_.output_head_screen) {
            throw std::logic_error("Qwen4Exp: screened-lm-head needs the LM head's bound screen");
        }
        probe_ngram_volume();

        // Every fixed allocation below is checked against the plan and against device free memory
        // (an allocation the driver placed in system memory fails startup, design §19.3.7).
        vram_ = open_vram_budget_source(device_.device, options_.vram_past_budget);
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
        window_geometry_ = prefix::kv_window_geometry(c_, options_.kv_cache, mtp_,
                                                      static_cast<std::uint32_t>(options_.max_concurrency));
        if (window_geometry_.present()) {
            allocate(window_backing_, plan_.window, allocated);
            window_backing_.fill(0); // zero tags never match a row's (tags are odd)
        }

        kv_geometry_ = prefix::kv_page_geometry(c_, options_.kv_cache, mtp_);
        if (plan_.kv_elastic) {
            kv_vmm_  = std::make_unique<VmmRange>(device_.device, plan_.kv, kKvChunkBytes);
            kv_span_ = DeviceSpan{kv_vmm_->base(), plan_.kv};
            if (!map_kv(plan_.kv_base_pages)) { throw std::runtime_error("Qwen4Exp: the KV pool's base pages do not fit"); }
            allocated += kv_vmm_->mapped_bytes();
        } else {
            allocate(kv_backing_, plan_.kv, allocated);
            kv_backing_.fill(0);
            kv_span_ = DeviceSpan{kv_backing_.p, kv_backing_.bytes};
        }
        pool_   = std::make_unique<DeviceKVPagePool>(kv_span_, plan_.pool);
        tables_ = std::make_unique<KVExecutionTablePool>(kv_span_, plan_.tables, *pool_);
        if (plan_.kv_elastic) { pool_->set_backed_pages(plan_.kv_base_pages); }

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
        allocate(gate_stats_, kGateStatsBytes * lanes, allocated);
        gate_stats_.fill(0);
        // Another FP8 volume's per-tensor scale or NVFP4 volume's multiplier (resolve_ngram_volume): the
        // artifact's ngram_scale serves only its own table.
        if (const auto scale = ngram_table().fp8_scale) {
            allocate(ngram_scale_, sizeof(*scale), allocated);
            ngram_scale_.copy_from_host(&*scale, sizeof(*scale), 0);
        }
        if (const auto scale = ngram_table().nvfp4_scale) {
            allocate(ngram_scale_, sizeof(*scale), allocated);
            ngram_scale_.copy_from_host(&*scale, sizeof(*scale), 0);
        }
        allocate(constraint_, constraint_bytes(lanes, max_width_), allocated);
        constraint_pinned_ = PinnedHostBuffer(constraint_bytes(lanes, max_width_));
        allocate(grammar_masks_, grammar_mask_bytes(token_domain_, lanes, max_width_), allocated);
        grammar_host_ = PinnedHostBuffer(grammar_mask_bytes(token_domain_, lanes, max_width_));
        if (options_.causal_scoring) {
            allocate(score_residuals_, plan_.score_residuals, allocated);
            allocate(score_logits_, plan_.score_logits, allocated);
            allocate(score_logits16_, plan_.score_logits16, allocated);
            allocate(score_io_, plan_.score_io, allocated);
            score_host_ = PinnedHostBuffer(plan_.score_io);
        }
        constraint_host_.assign(static_cast<std::size_t>(lanes * max_width_), -1);
        publish_pinned_word(gate_ready(), 0); // pinned memory starts undefined; 0 is never a round's word

        execution::ForwardState state;
        state.gdn      = gdn_.get();
        state.ple_conv = Tensor(ple_backing_.p, DType::BF16, {width_, span_, lanes});
        for (std::uint32_t l = 0; l < c_.attention_layers; ++l) {
            state.qsa_tails.push_back(Tensor(static_cast<std::byte*>(tails_backing_.p) +
                                                 static_cast<std::size_t>(l) * di_ * (r_ - 1) * lanes * 2,
                                             DType::BF16, {di_, r_ - 1, lanes}));
        }
        const auto layout   = c_.lossless ? paged_kv_lossless_storage_layout(dim(c_.attention.head_dim))
                                          : paged_kv_storage_layout(options_.kv_cache, dim(c_.attention.head_dim));
        const auto kv_heads = dim(c_.attention.kv_heads);
        execution::ForwardKV kv;
        kv.block_tables  = tables_->matrix();
        for (std::uint32_t l = 0; l < plan_.kv_layers; ++l) {
            const prefix::PlaneRange planes = prefix::kv_layer_planes(kv_geometry_, l);
            std::size_t plane               = planes.begin;
            ops::QsaKVLayer layer;
            layer.kv.storage      = options_.kv_cache;
            layer.kv.head_dim     = dim(c_.attention.head_dim);
            layer.kv.num_kv_heads = kv_heads;
            layer.kv.k_pages      = pool_->plane(plane++);
            if (layout.key.has_scale()) { layer.kv.k_scale_pages = pool_->plane(plane++); }
            layer.kv.v_pages = pool_->plane(plane++);
            if (layout.value.has_scale()) { layer.kv.v_scale_pages = pool_->plane(plane++); }
            layer.pooled_pages = pool_->plane(plane++);
            if (window_geometry_.present()) { layer.kv.window = prefix::kv_window_view(window_geometry_, window_backing_.p, l); }
            if (plane != planes.end) { throw std::logic_error("Qwen4Exp: KV planes do not match the page geometry"); }
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
        if (plan_.dflash2) { allocate_dflash2(lanes, allocated); }
        if (plan_.drafter_pool > 0) { create_drafter_pool(experts, allocated); }
        work_capacity_ = plan_.work;
        wide_work_     = plan_.wide_work;
        work_          = std::make_unique<WorkspaceArena>(work_capacity_);
        allocated += work_capacity_;

        // The promotions' copy sources: the banks where the device reads them (cudaMemcpyBatchAsync
        // faults on a registered block's host address where the device maps it elsewhere: WDDM).
        std::vector<const std::uint8_t*> banks;
        std::vector<std::uint64_t> host_strides; // each bank's own record stride (a GGUF import's vary)
        std::uint64_t record_stride = 0;
        for (const auto& layer : parameters_.layers) {
            banks.push_back(reinterpret_cast<const std::uint8_t*>(layer.moe.bank->planes.device_records));
            host_strides.push_back(layer.moe.bank->planes.record_stride);
            record_stride = std::max(record_stride, layer.moe.bank->planes.record_stride);
        }
        if (record_stride != plan_.record_stride ||
            (!c_.exact_experts && std::any_of(host_strides.begin(), host_strides.end(),
                                             [&](std::uint64_t s) { return s != record_stride; }))) {
            throw std::logic_error("Qwen4Exp: the expert banks' record stride differs from their format's");
        }
        // Staging slots for each layer call's misses (copied with a compact read window, design
        // 8.6), allocated before the frames take the remaining memory.
        allocate(staging_, plan_.staging, allocated);
        experts.staging_base  = static_cast<std::uint8_t*>(staging_.p);
        experts.staging_slots = plan_.staging_slots;
        // The expert cache's tables and route log; its frames come last (below).
        // A GGUF import's experts take Basalt's adaptive tier (IQ3_S decode 1.070-1.076x the LFRU in four A/Bs,
        // 2026-10-27/28, hit rate 84.6 against 85.1 % with half the promotions); NVFP4 and block-FP8 banks keep the LFRU
        // (Dense8 0.971x adaptive).
        const QType experts_format = parameters_.layers.front().moe.bank->planes.gate_up_format;
        const auto policy = experts_format != QType::NVFP4_MUL && experts_format != QType::FP8_E4M3FN_B128
                                ? ExpertResidency::VramPolicy::Adaptive
                                : ExpertResidency::VramPolicy::Lfru;
        residency_ = std::make_unique<ExpertResidency>(expert_layout(c_), std::move(banks), record_stride, columns_,
                                                       device_.device, host_strides, policy);
        allocated += plan_.bytes.residency;
        if (allocated != plan_.bytes.fixed_bytes()) {
            throw std::logic_error("Qwen4Exp: the Program's device allocations differ from its device plan");
        }
        if (const std::uint64_t spilled = guard.end(allocated); spilled != 0) {
            throw std::runtime_error(spill_message(spilled));
        }
        // The staging, shared-expert and split streams at the device's greatest priority (the decode graphs keep
        // it as node priorities, core/decode_graph.cpp), so their short kernels take an SM as the routed experts'
        // CTAs retire: the tensor-core gate/up holds ~90 % of an SM's registers and shared memory, and the shared
        // expert's SwiGLU waited ~30 us a layer for an SM. IQ3_S full layout, 2 runs each (2026-10-29): 16K
        // 263.6 -> 265.3 tok/s, struct 456.5 -> 463.3; the shared stream alone 265.8 / 455.4. (With the dp4a
        // kernels' smaller CTAs this measured 0.999x and was removed on 2026-10-28.)
        int least_priority = 0, side_priority = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &side_priority));
        CUDA_CHECK(cudaStreamCreateWithPriority(&overlap_.stream, cudaStreamNonBlocking, side_priority));
        for (auto& event : overlap_.events) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
        experts.overlap_stream = overlap_.stream;
        experts.overlap_events = overlap_.events;
        CUDA_CHECK(cudaStreamCreateWithPriority(&overlap_.shared_stream, cudaStreamNonBlocking, side_priority));
        for (auto& event : overlap_.shared_events) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
        experts.shared_stream = overlap_.shared_stream;
        experts.shared_events = overlap_.shared_events;
        // The stream on which a forked call's staged misses compute (their down rows stage beside the gate/up pass;
        // Dense8 MTP decode +6.7 / +1.3 / +10.3 %, Ornith +1.6 / +8.0 / +4.6 %, IQ3_S within spread, 2026-10-27).
        {
            CUDA_CHECK(cudaStreamCreateWithPriority(&overlap_.split_stream, cudaStreamNonBlocking, side_priority));
            for (auto& event : overlap_.split_events) {
                CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            }
            experts.split_stream = overlap_.split_stream;
            experts.split_events = overlap_.split_events;
        }
        // The prefill chunk the expert cache can lend a wide workspace for, from the frames the
        // cache will start with (sized below with the same demand); before the SSD tier, whose
        // tickets are sized by the chunk. A cache of several frame pools lends from one, which the
        // check after the cache is sized accounts for.
        if (options_.expert_cache) {
            const std::uint64_t frames = size_expert_frames(vram_->query(), expert_frame_demand(record_stride)).frames;
            fit_wide_work_to_cache(frames - frames / 4, record_stride);
        }
        // The SSD expert tier (design §19.3.7): the banks were not pinned; RAM slots hold the
        // records the tier keeps and the rest are read from the artifact when a round needs them.
        if (const auto& store = parameters_.model.expert_store(); store.has_value()) {
            for (std::uint32_t l = 0; l < store->layers(); ++l) {
                if (store->record_bytes(l) != host_strides[l]) {
                    throw std::logic_error("Qwen4Exp: the expert store's records differ from the banks' strides");
                }
            }
            // Slots pooled by record size (a GGUF import's layers differ; HostExpertTier): a VRAM victim's
            // demotion copies its frame (its layer's pitch) into a slot of that size.
            HostExpertTier::Options tier;
            tier.ring = tier_ring_slots(options_, plan_.staging_slots);
            const std::uint32_t lists = tier.ring + tier.prefetch + tier.demotion;
            const std::uint64_t keys  = static_cast<std::uint64_t>(store->layers()) * store->experts();
            // Pinned slots up to every key besides the lists; unpinned records for the keys they leave.
            // With unpinned records, tier.bounce of the pinned slots stage their promotions and
            // tier.stream_bounce the prefill stream's copies of them.
            if (options_.prefill_stream && options_.expert_cache) {
                // A wide chunk streams about a layer's unpinned share of its experts.
                const double share = std::min(1.0, static_cast<double>(options_.expert_unpinned_bytes / record_stride) /
                                                       static_cast<double>(keys));
                const auto per_part = std::clamp<std::uint32_t>(
                    (static_cast<std::uint32_t>(std::ceil(share * store->experts())) + 15U) / 16U * 16U,
                    kStreamBounceMin, kStreamBounceMax);
                tier.stream_bounce = ExpertStream::kBounceParts * per_part;
            }
            tier.slots    = HostExpertTier::slots_for_bytes(*store, options_.expert_ram_bytes, tier);
            // The prefill stream's reads of SSD-only experts (one per layer walk span instead of one
            // per call) borrow RAM slots for the span (the lowest-ranked residents leave RAM).
            if (options_.prefill_stream && options_.expert_cache) { tier.stream = std::min(kStreamReadSlots, tier.slots / 8); }
            // The exclusive layout keeps RAM copies of as many VRAM experts as the RAM plan's reserve for a prefill's
            // loan and the KV growth (a loan takes shadowed experts first): without a floor the
            // shadows were RAM's first victims, a loan found 804 of its ~3,250 and the experts it evicted without a copy
            // came back from the SSD (2026-10-28, IQ3_S at 16K).
            if (options_.expert_ram_exclusive && options_.expert_cache) {
                const std::uint64_t frames = size_expert_frames(vram_->query(), expert_frame_demand(record_stride)).frames;
                tier.shadow_floor =
                    static_cast<std::uint32_t>(std::llround(static_cast<double>(frames) * options_.expert_shadow_share));
            }
            const std::uint32_t reserved =
                options_.expert_unpinned_bytes >= record_stride ? lists + tier.bounce + tier.stream_bounce : lists;
            tier.unpinned = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                options_.expert_unpinned_bytes / record_stride, keys - std::min<std::uint64_t>(keys, tier.slots - std::min(tier.slots, reserved))));
            tier.min_slots    = std::max<std::uint32_t>(kMinTierSlots, lists + kMinTierSlots / 2);
            tier.pin_reserve  = options_.later_pinned_bytes;
            const std::uint32_t needed = tier.min_slots + (tier.unpinned > 0 ? tier.bounce + tier.stream_bounce : 0U);
            if (tier.slots < needed) {
                throw std::runtime_error(
                    "Qwen3.8-Flash-Next: the SSD expert tier needs RAM for at least " + std::to_string(needed) +
                    " pinned expert slots (" + std::to_string(tier.slots) +
                    " fit in the RAM ledger's page-locked expert share; on Windows page-locked memory is limited to "
                    "the GPU's shared memory, so lower --host-context-mib or close programs that use it)");
            }
            fetch_channel_ = std::make_unique<ops::offloaded_moe::FetchChannel>();
            tier.fetch     = fetch_channel_.get();
            // A layer walk's step calls ceil(layers / n) layers once per chunk of its n-chunk span (one
            // tier round per step): at most layers + n - 1 layer calls.
            tier.layer_calls = store->layers() + kWalkMaxTokens / static_cast<std::uint32_t>(std::max(chunk_, 1)) + 1U;
            // An expert stripe on a second drive: each record's tail is read from it while its head
            // is read from the artifact, at the same depth per drive.
            const ExpertStore* records = &*store;
            if (!options_.expert_stripe.empty()) {
                striped_store_.emplace(with_stripe(*store, read_expert_stripe(options_.expert_stripe),
                                                   parameters_.model.info().artifact_id));
                records = &*striped_store_;
                tier.io.demand_depth *= 2;
                tier.io.total_depth = tier.io.demand_depth + tier.io.restore_depth + tier.io.prefetch_depth;
                diagnostic("SSD tier: expert record tails from " + options_.expert_stripe.string(),
                           DiagnosticLevel::Info);
            }
            tier_ = std::make_unique<HostExpertTier>(*records, std::move(tier));
            residency_->attach_tier(tier_.get(), options_.expert_ram_exclusive);
            if (tier_->refused_slots() > 0) {
                diagnostic("SSD expert tier: Windows refused to page-lock " + std::to_string(tier_->refused_slots()) +
                               " of the planned expert slots (its limit is the GPU's shared memory); they are kept as "
                               "unpinned records the CPU serves",
                           DiagnosticLevel::Warning);
            }
            if (!tier_->unpinned_lock_failure().empty()) {
                diagnostic("SSD expert tier: the OS would not lock the " + std::to_string(tier_->unpinned_records()) +
                               " unpinned expert records in RAM (" + tier_->unpinned_lock_failure() +
                               "); they still work, but under memory pressure the OS may page them out and responses slow down",
                           DiagnosticLevel::Warning);
            }
        }
        // After the tier (whose pinned slots are the probe's source when the banks stay in the
        // artifact), before the CPU service and the split take the measured rate.
        measure_link();
        expert_error_ = PinnedHostBuffer(64);
        std::memset(expert_error_.data(), 0, 64);
        experts.error = static_cast<std::uint32_t*>(expert_error_.data());
        // TEMPORARY (INFERNIX_TMP_REPLAY_BENCH): the replay bench's round clock and CPU-wait word.
        if (temporary_switches::integer("INFERNIX_TMP_REPLAY_BENCH", -1, -1, std::int64_t{1} << 40) >= 0) {
            replay_clock_ = std::make_unique<RoundClock>();
            replay_wait_  = DeviceBuffer(256); // outside the planned ledger: one word, bench runs only
            replay_wait_.fill(0);
            experts.wait_ns = static_cast<unsigned long long*>(replay_wait_.p);
        }
        // The prefill stream copies records into ring slots of the frame pitch; a GGUF import's
        // banks keep their own (smaller) strides.
        if (options_.prefill_stream && options_.expert_cache) {
            ExpertStream::RecordOf record_of;
            if (tier_) {
                record_of = [tier = tier_.get()](std::uint32_t layer, std::uint32_t expert) {
                    return tier->record(tier->key(layer, expert));
                };
            } else {
                std::vector<const std::uint8_t*> banks; // device addresses: the stream batches its copies
                for (const auto& layer : parameters_.layers) {
                    banks.push_back(reinterpret_cast<const std::uint8_t*>(layer.moe.bank->planes.device_records));
                }
                record_of = [banks = std::move(banks), host_strides](std::uint32_t layer, std::uint32_t expert) {
                    return banks[layer] + static_cast<std::size_t>(expert) * host_strides[layer];
                };
            }
            ExpertStream::Hold hold;
            ExpertStream::SsdReads ssd;
            if (tier_) {
                hold = [tier = tier_.get()](std::uint32_t layer, std::uint32_t expert) {
                    tier->stream_pin(tier->key(layer, expert));
                };
                ssd.read = [tier = tier_.get()](std::uint32_t layer, std::uint32_t expert) {
                    return tier->stream_read(tier->key(layer, expert));
                };
                ssd.wait    = [tier = tier_.get()](std::int32_t slot) { return tier->stream_wait(slot); };
                ssd.release = [tier = tier_.get()](std::int32_t slot) { tier->stream_release(slot); };
                ssd.layer_cap = [tier = tier_.get()] { return tier->stream_layer_cap(); };
            }
            expert_stream_ = std::make_unique<ExpertStream>(c_.num_hidden_layers, c_.moe.experts, record_stride,
                                                                       std::move(record_of), !tier_, std::move(hold),
                                                                       host_strides, std::move(ssd));
            if (tier_ && tier_->stream_bounce_slots() > 0) {
                // Unpinned records stream through the tier's bounce slots, still two layers ahead.
                HostExpertTier* tier = tier_.get();
                expert_stream_->set_staging(ExpertStream::Staging{
                    .unpinned = [tier](std::uint32_t layer, std::uint32_t expert) {
                        return tier->unpinned_record(tier->key(layer, expert));
                    },
                    .slot     = [tier](std::uint32_t i) { return tier->stream_bounce_slot(i); },
                    .copy     = [tier](std::span<const std::pair<std::uint8_t*, const std::uint8_t*>> copies) {
                        tier->copy_records(copies);
                    },
                    .per_part = tier->stream_bounce_slots() / ExpertStream::kBounceParts});
            }
            experts.stream = expert_stream_.get();
        }
        // CPU-served misses for decode and verify calls, and for every call of at most
        // kMaxCpuColumns columns (short prompts, chunk remainders, forced tokens), whose experts are
        // all thin and so give the same bits on either device (design 16.2); wider prefill chunks
        // stay on the GPU.
        // A GGUF import's records are computed by the ggml CPU route (ops/offloaded_sparse_moe/cpu/ggml_expert.h).
        // What the offload switches resolved to (their values are in the startup log's temporary switches line).
        {
            std::string line = "expert offload (temporary switches resolved): staging slots " +
                               std::to_string(plan_.staging_slots);
            if (one_column_divisor() != 0) {
                line += "; plain decode one-column divisor " + std::to_string(plain_divisor(0)) + " below " +
                        std::to_string(runtime::expert_offload::kPlainDivisorContext) + " context tokens, " +
                        std::to_string(plain_divisor(runtime::expert_offload::kPlainDivisorContext)) + " from there";
            }
            diagnostic(line, DiagnosticLevel::Info);
        }
        // The team's size and CPUs (INFERNIX_TMP_CPU_PLACEMENT; today unpinned). The engine thread is not
        // isolated: Dense8 +0.15 %, IQ3_S -0.14 % (2026-10-28).
        placement_ = runtime::expert_offload::cpu_placement(
            temporary_switches::offload(),
            c_.exact_experts ? options_.cpu_ggml_expert_workers : options_.cpu_expert_workers);
        const std::uint32_t cpu_workers = this->cpu_workers();
        const std::uint32_t cpu_jobs    = options_.cpu_expert_jobs;
        const std::uint32_t assist_jobs = options_.cpu_assist_jobs;
        if (cpu_workers > 0 && cpu_jobs > 0) {
            if (!placement_.text.empty()) { diagnostic(placement_.text, DiagnosticLevel::Info); }
            // The team's rate sets the decode miss split (design §19.3.12); the SSD tier's is measured
            // once its RAM is filled (below).
            if (!tier_) { measure_cpu_rate(record_stride); }
            // Decode and verification calls are CPU-served up to decode_columns; prefill calls up to
            // kAssistMaxColumns take the assist cap (wider chunks stream or stage on the GPU).
            const int decode_columns = std::max(ops::offloaded_moe::kMaxCpuColumns, lanes * max_width_);
            const bool assist        = assist_jobs > 0 && decode_columns < kAssistMaxColumns;
            std::vector<ops::offloaded_moe::CpuMissService::Layer> service_layers;
            for (const auto& layer : parameters_.layers) {
                service_layers.push_back({.records       = reinterpret_cast<const std::uint8_t*>(layer.moe.bank->planes.records),
                                          .record_stride = layer.moe.bank->planes.record_stride,
                                          .scales        = layer.moe.bank->scales.data(),
                                          .ggml          = ggml_format(*layer.moe.bank)});
            }
            cpu_service_ = std::make_unique<ops::offloaded_moe::CpuMissService>(
                std::move(service_layers),
                ops::offloaded_moe::CpuMissService::Options{
                    .workers     = static_cast<int>(cpu_workers),
                    .max_jobs    = static_cast<int>(std::min<std::uint32_t>(cpu_jobs, ops::offloaded_moe::kMaxCpuJobs)),
                    .max_columns = assist ? kAssistMaxColumns : decode_columns,
                    .pcie_divisor = pcie_divisor(),
                    .one_column_divisor = one_column_divisor(),
                    .pcie_share  = pcie_share(),
                    .wide_from   = assist ? decode_columns + 1 : 0,
                    .wide_jobs   = assist ? static_cast<int>(std::min<std::uint32_t>(assist_jobs, ops::offloaded_moe::kMaxCpuJobs)) : 0,
                    .cpus        = placement_.cpus,
                    .activation  = parameters_.layers.front().moe.bank->activation,
                    .narrow_a16  = cpu_narrow_a16(),
                    .ggml_a8     = cpu_ggml_a8(),
                    .nvfp4_a8    = cpu_narrow_a16() == ops::offloaded_moe::NarrowA16::kA8,
                    .records     = tier_.get(), // tier mode: SSD-only jobs read through the tier
                    // Reported when it happens; check_expert_error then stops the Engine with it.
                    .on_failure  = [this](const std::string& why) {
                        diagnostic("Qwen3.8-Flash-Next: the CPU expert service stopped serving misses: " + why,
                                   DiagnosticLevel::Error);
                    }});
            for (std::uint32_t l = 0; l < c_.num_hidden_layers; ++l) {
                experts.cpu.push_back(cpu_service_->channel(static_cast<int>(l)));
                // Plain decode's divisor follows the context (plain_divisor): the calls read it from
                // the io word each round uploads.
                if (one_column_divisor() != 0) {
                    experts.cpu.back().one_column_divisor_device =
                        reinterpret_cast<const std::int32_t*>(static_cast<std::byte*>(io_device_.p) + io_layout_.divisor);
                }
            }
            if (assist && expert_stream_ && options_.prefill_cpu_split) {
                split_policy_ = std::make_unique<SplitPolicy>(*this);
                experts.split = split_policy_.get();
            }
        }
        create_prefix_cache();

        // The VRAM expert cache takes what the fixed allocations, the reserve for graph
        // executables and the display headroom leave (design §19.3.7). Its frames are the one
        // elastic allocation: a chunk the driver places in system memory (another program took
        // memory since the reading) is given back and the cache stops growing there.
        if (options_.expert_cache) {
            const VramDemand demand = expert_frame_demand(residency_->frame_bytes());
            sizing_                 = size_expert_frames(vram_->query(), demand);
            // Each pool maps whole chunks: one chunk of slack per pool beyond the first.
            const auto slack = static_cast<std::uint32_t>((residency_->pools() - 1) * ExpertResidency::kChunkBytes /
                                                          residency_->frame_bytes());
            sizing_.frames   = sizing_.frames > slack ? sizing_.frames - slack : 0U;
            const auto grown = residency_->resize(sizing_.frames, device_.stream, *vram_);
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
            // The pool the workspace is lent from (one of several for a GGUF import's record sizes).
            if (grown.frames > 0) {
                fit_wide_work_to_cache(residency_->lend_capacity_bytes() / residency_->lend_pitch(), residency_->lend_pitch());
            }
        }
        for (std::uint32_t l = 0; l < c_.num_hidden_layers; ++l) {
            experts.frame_bases.push_back(residency_->frame_base(l));
            experts.frame_pitches.push_back(residency_->frame_pitch(l));
        }
        experts.route_log    = residency_->route_log();
        experts.route_stride = residency_->route_stride();
        for (std::uint32_t l = 0; l < c_.num_hidden_layers; ++l) { experts.frames[l] = residency_->table(l); }
        experts.landing       = residency_->landing_table();
        experts.landed        = residency_->landed_log();
        experts.landing_slots = static_cast<std::int32_t>(ExpertResidency::kLandingSlots);
        experts.tail_skip_rank = static_cast<std::int32_t>(options_.expert_tail_skip);
        if (tier_) {
            for (std::uint32_t l = 0; l < c_.num_hidden_layers; ++l) {
                experts.host_tables.push_back(residency_->record_table(l));
                experts.fetch.push_back(fetch_channel_->channel(static_cast<int>(l)));
            }
        }
        warm_start_experts();
        // The SSD tier has no records in RAM before its fill: the CPU team's rates are measured on its
        // RAM copies now, so the decode miss divisor follows this record size and CPU, not the reference.
        if (tier_ && cpu_service_) {
            // Right after the tier's fill (tens of GiB of direct reads) the machine's memory is slow for a few seconds:
            // Dense8's probe read 9,377 experts/s there and 18,942 5 s later in the same process, on the same records
            // (2026-10-27), and the low reading kept half the decode misses on the link instead of a third for the
            // whole session (decode -26 %). The rates are measured again after a pause; the better reading stands.
            measure_cpu_rate(record_stride);
            const double first_experts = cpu_expert_rate_, first_columns = cpu_column_rate_;
            std::this_thread::sleep_for(kTierRateSettle);
            measure_cpu_rate(record_stride);
            cpu_expert_rate_ = std::max(cpu_expert_rate_, first_experts);
            cpu_column_rate_ = std::max(cpu_column_rate_, first_columns);
            for (auto& channel : experts.cpu) {
                channel.pcie_divisor = pcie_divisor();
                channel.pcie_share   = pcie_share();
            }
        }
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
                                                        std::move(experts), dim(options_.max_context), options_.fast_paths);
        if (dflash2_) { create_dflash2_drafter(); }
        capture_graph_set();
        device_.synchronize();
        report_lane_cost(lanes, residency_->frame_bytes());
    }

    const DeviceLayout& device_layout() const noexcept { return plan_; }
    const VramSizing& vram_sizing() const noexcept { return sizing_; }
    // The expert cache's counters since startup and its size now (the per-request figures in
    // finish() are deltas of the same counters); nothing when the cache is off.
    [[nodiscard]] std::optional<ExpertCacheTotals> expert_cache_totals() const noexcept {
        if (!residency_) { return std::nullopt; }
        const ExpertResidency::Stats& s = residency_->stats();
        return ExpertCacheTotals{.frames     = residency_->frames(),
                                 .routed     = s.routed,
                                 .hits       = s.hits,
                                 .promotions = s.promotions,
                                 .landed     = s.landed,
                                 .cpu_served = cpu_service_ ? cpu_service_->served_experts() : 0};
    }

    // ---------------------------------------------------------------- admission
    RequestBasePlan plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options) {
        const auto& data = qwen3_5::PreparedPromptAccess::view(prompt);
        std::shared_ptr<const VisionAdmission> vision;
        if (data.has_media()) { vision = plan_vision(data); }
        const auto n = static_cast<std::uint32_t>(data.token_ids.size());
        if (n == 0 || n > options_.max_context) {
            throw std::invalid_argument("Qwen4Exp: the prompt is empty or exceeds max_context");
        }
        for (const TokenId id : options.readout_tokens) {
            if (id < 0 || id >= token_domain_) {
                throw std::invalid_argument("readout token is outside the public token domain");
            }
        }
        auto base                       = std::make_unique<BasePlanImpl>();
        base->readout_tokens            = options.readout_tokens;
        base->constraint                = compile_constraint(options.constraint);
        base->summary.prompt_tokens     = n;
        base->summary.requested_output_tokens = options.requested_output_tokens;
        const std::uint32_t capacity_output   = options_.max_context - n + 1U;
        base->summary.effective_output_tokens = std::min(options.requested_output_tokens, capacity_output);
        base->summary.effective_limit_reason  = options.requested_output_tokens <= capacity_output
                                                    ? FinishReason::OutputLimit
                                                    : FinishReason::ContextCapacity;
        base->reuse                           = prefix_ != nullptr && options.allow_prefix_reuse && data.identity.reusable;
        base->summary.publish_continuation    = base->reuse;
        base->vision   = std::move(vision);
        base->sampling            = translate(options.sampling);
        base->sampling.gumbel_max = options_.gumbel_sampling;
        const std::uint32_t positions = std::min(options_.max_context, n + base->summary.effective_output_tokens);
        base->positions = positions;
        base->pages     = (positions + kPagedKVPageSize - 1) / kPagedKVPageSize;
        base->prompt = std::make_shared<const PreparedPrompt>(std::move(prompt));
        return RequestBasePlan(std::move(base));
    }

    // The positions one round may write past the lane's frontier: a verification's columns and the
    // MTP drafter's cells (design §19.3.13).
    std::uint32_t round_positions() const noexcept {
        return static_cast<std::uint32_t>(std::max(max_width_, mtp_ ? mtp_k_ + 1 : 1)) + 1U;
    }

    // The KV pages a binding of `base` whose prefill ends at `end` reserves: through one round past
    // it, within the request's extent.
    std::uint32_t binding_pages(const BasePlanImpl& base, std::uint32_t end) const noexcept {
        std::uint32_t positions = std::min(base.positions, end + round_positions());
        // Without preemption the request holds its whole extent from admission (less a page of the
        // pool), so no unit of it grows and nothing pauses it.
        if (!options_.preemption) {
            const std::uint32_t pool = pool_->capacity_pages() > 1 ? pool_->capacity_pages() - 1U : 1U;
            positions = std::max(positions, std::min(base.positions, pool * static_cast<std::uint32_t>(kPagedKVPageSize)));
        }
        return (positions + kPagedKVPageSize - 1) / kPagedKVPageSize;
    }

    static PrefixReusePath reuse_path_for(runtime::prefix_cache::SnapshotKind kind) noexcept {
        return kind == runtime::prefix_cache::SnapshotKind::Endpoint ? PrefixReusePath::HybridEndpoint
                                                                     : PrefixReusePath::HybridSnapshot;
    }

    // A source over `selection` (absent: a root start without a prefix cache).
    SourceCandidate make_source(const BasePlanImpl& base, const qwen3_5::PreparedPromptData& data,
                                std::optional<PrefixSelection> selection) const {
        auto impl      = std::make_shared<QuoteImpl>();
        impl->summary  = base.summary;
        impl->sampling = base.sampling;
        impl->pages    = binding_pages(base, base.summary.prompt_tokens);
        impl->vision   = base.vision;
        SourceCandidate out;
        if (selection) {
            if (selection->snapshot) {
                // Items inside the reused prefix are not encoded again (design §19.3.2).
                if (impl->vision && selection->frontier > 0) { impl->vision = plan_vision(data, selection->frontier); }
                out.reused_tokens = selection->frontier;
                out.reuse_path    = reuse_path_for(prefix_->index().snapshot(*selection->snapshot).kind);
            }
            impl->prefix = std::move(selection);
        }
        out.hybrid = std::move(impl);
        return out;
    }

    // The prefix cache's choice for the request (without a lane: a lane-resident snapshot is credited
    // at start_binding), then a root start.
    std::vector<SourceCandidate> hybrid_sources(const RequestBasePlan& base, std::uint32_t maximum_frontier) {
        const BasePlanImpl& b = *base.impl_;
        const auto& data      = qwen3_5::PreparedPromptAccess::view(*b.prompt);
        std::vector<SourceCandidate> out;
        if (prefix_) {
            std::optional<PrefixSelection> chosen = prefix_select(data, b, kNoLane);
            const std::uint32_t reuse = chosen && chosen->snapshot ? chosen->frontier : 0U;
            // Coalescing only defers a fresh request (an empty list: the Engine asks again).
            if (b.reuse && maximum_frontier == UINT32_MAX && prefix_await_sibling(data, reuse)) { return {}; }
            if (chosen && chosen->snapshot && chosen->frontier > 0 && chosen->frontier <= maximum_frontier) {
                out.push_back(make_source(b, data, std::move(chosen)));
            }
            out.push_back(make_source(b, data, prefix_root(data, b)));
        } else {
            out.push_back(make_source(b, data, std::nullopt));
        }
        return out;
    }

    // Stages the binding of `source` on `lane`: re-selects against the cache as it is now (crediting
    // a lane-resident snapshot), reserves the prompt plus one round, lends the Vision window
    // (vision_lend) and maps the cached blocks. The Begin the Engine published names the source's
    // frontier, so a changed choice is refused and the Engine falls back to the next source (the root).
    BindingReservation start_binding(const RequestBasePlan& base, runtime::LaneId destination,
                                     const SourceCandidate& source, ResumeState* resume) {
        if (source.hybrid == nullptr) { throw std::logic_error("Qwen4Exp: a binding source carries no quote"); }
        if (resume != nullptr && resume->impl_ == nullptr) { throw std::logic_error("Qwen4Exp: a resume carries no state"); }
        const QuoteImpl& q = *source.hybrid;
        const std::uint32_t index = destination.value;
        if (index >= options_.max_concurrency || lanes_[index].phase != Phase::Free || in_transaction()) {
            throw std::logic_error("Qwen4Exp: the binding destination is not free");
        }
        // A request paused after its first token binds its ledger; one paused inside its prompt binds
        // the prompt again, from the deepest cached state (its pause published the frontier).
        if (resume != nullptr && resume->impl_->generated) { return bind_replay(*base.impl_, index, *resume->impl_); }
        Lane& lane = lanes_[index];
        kv_busy();
        const BasePlanImpl& b = *base.impl_;
        const auto& data      = qwen3_5::PreparedPromptAccess::view(*b.prompt);
        std::optional<PrefixSelection> selection;
        if (prefix_) {
            if (q.prefix && q.prefix->snapshot) {
                selection = prefix_select(data, b, index);
                if (!selection || !selection->snapshot || selection->frontier != source.reused_tokens) {
                    return {}; // the quoted snapshot changed: no shortage, the Engine moves to the root source
                }
            } else {
                selection = prefix_root(data, b);
            }
            prefix_pin(*selection);
        }
        lane.history.assign(data.token_ids.begin(), data.token_ids.end());
        lane.history.reserve(lane.history.size() + q.summary.effective_output_tokens + 1);
        lane.prompt_tokens  = static_cast<std::uint32_t>(lane.history.size());
        // A prompt with media keeps its M-RoPE positions; every other token is its index + delta.
        lane.rope.prompt_tokens = lane.prompt_tokens;
        lane.rope.delta         = data.rope_delta;
        if (data.has_media()) {
            lane.rope.prompt.assign(data.positions.begin(), data.positions.end());
        } else {
            lane.rope.prompt.clear();
        }
        // Reset before activation: a resume's activation sets the lane's frontier and state.
        lane.state_tokens   = 0;
        lane.vision_seconds = 0;
        lane.vision_timer.reset();
        lane.speculative    = {};
        lane.mtp_cells      = 0;
        lane.mtp_live       = mtp_;
        lane.dflash_frontier = 0;
        lane.row = tables_->acquire(static_cast<std::int32_t>(index));
        const auto shortage_of = [&](std::uint32_t need) {
            BindingReservation out;
            out.shortage.main_kv_pages = need > pool_->available_pages() ? need - pool_->available_pages() : 0;
            return out;
        };
        vision_make_room(q.vision.get());
        if (selection) {
            lane.prefix.reuse  = q.summary.publish_continuation;
            lane.prefix.hashes = data.block_hashes;
            lane.prefix.extras = data.block_extras;
            lane.prefix.trailing_extra = prefix_trailing_extra(data);
            lane.prefix.hints  = data.tap_hints.hints;
            lane.prefix.exclusions = prefix_exclusions(data);
            // Make room before reserving: an elastic pool grows only for what eviction cannot free, so
            // idle cache blocks never hold memory the expert frames could use (design §19.3.11).
            pc_make_room(selection->need);
            if (pool_->available_pages() < selection->need) { (void)grow_kv(selection->need); }
            auto reservation = pool_->reserve(selection->need);
            if (!reservation) {
                const auto out = shortage_of(selection->need);
                prefix_unpin(*selection);
                lane.row = KVExecutionRowLease{};
                return out;
            }
            if (q.vision) {
                vision_reserve(lane, q.vision, data);
                if (auto refused = vision_lend(lane, index)) {
                    lane.vision.reset(); // nothing was lent
                    prefix_unpin(*selection);
                    lane.row = KVExecutionRowLease{};
                    return std::move(*refused);
                }
            }
            const std::uint64_t restored = prefix_->counters().host_restore_bytes;
            prefix_activate(lane, index, *selection, *reservation, q.pages);
            lane.prefix.cached_tokens  = selection->cached_tokens;
            lane.prefix.restored_bytes = prefix_->counters().host_restore_bytes - restored;
            lane.reservation    = std::move(*reservation);
            lane.prefix.reused  = selection->frontier;
            lane.calls          = selection->plan.ends;
        } else {
            if (pool_->available_pages() < q.pages) { (void)grow_kv(q.pages); }
            auto reservation = pool_->reserve(q.pages);
            if (!reservation) {
                const auto out = shortage_of(q.pages);
                lane.row = KVExecutionRowLease{};
                return out;
            }
            if (q.vision) {
                vision_reserve(lane, q.vision, data);
                if (auto refused = vision_lend(lane, index)) {
                    lane.vision.reset(); // nothing was lent
                    lane.row = KVExecutionRowLease{};
                    return std::move(*refused);
                }
            }
            lane.pages.clear();
            lane.pages.reserve(q.pages);
            pool_->materialize(*reservation, q.pages, lane.pages);
            lane.reservation = std::move(*reservation);
            tables_->publish(lane.row.handle(), 0, std::span<const DeviceKVPageLease>(lane.pages), device_.stream);
            reset_slot(index);
            lane.calls = plan_prefill(data, 0, {}, false).ends;
        }
        lane.next_call = 0;
        clear_prompt_uses(lane);
        lane.mtp_accept.fill(kAcceptancePrior);
        lane.mtp_policy_rounds = 0;
        lane.mtp_cumulative.reset();
        lane.dflash_calibration = dflash_process_calibration_;
        lane.ngram_accept       = kNgramAcceptancePriors;
        if (max_width_ > 1) {
            if (options_.ngram_draft_tokens > 0) {
                lane.proposer =
                    std::make_unique<qwen3_5::detail::NgramProposer>(proposer_tokens_, proposer_tokens_ / 2);
                for (const auto token : lane.history) { lane.proposer->append(token); }
            }
            lane.speculative.enabled      = true;
            lane.speculative.backend      = mtp_       ? SpeculativeBackend::Mtp
                                            : dflash2_ ? SpeculativeBackend::DFlash2
                                                       : SpeculativeBackend::None;
            lane.speculative.draft_window = static_cast<std::uint32_t>(max_width_ - 1);
            lane.speculative.accepted_per_position.assign(static_cast<std::size_t>(max_width_ - 1), 0);
        }
        lane.sampling       = q.sampling;
        lane.readout        = b.readout_tokens;
        lane.prompt         = b.prompt;
        lane.constraint     = b.constraint;
        lane.constraint_round.clear();
        lane.constraint_trace.clear();
        lane.sampling.token_counts =
            static_cast<std::int32_t*>(token_counts_.p) + static_cast<std::size_t>(index) * token_domain_;
        *nonfinite_flag(index)  = 0; // the lane's earlier rounds have completed
        lane.sampling.nonfinite = nonfinite_flag(index);
        lane.epoch           = ++next_epoch_;
        lane.phase           = Phase::Prefill;
        lane.prefill_ns      = 0;
        lane.decode_ns       = 0;
        lane.decode_share_ns = 0;
        lane.cache_at_admission = residency_->stats();
        lane.cpu_served_at_admission = cpu_service_ ? cpu_service_->served_experts() : 0;
        if (tier_) {
            lane.tier_at_admission   = tier_->stats();
            lane.losses_at_admission = tier_->controller().losses();
        }
        lane.ngram                   = {};
        lane.gate                    = {};
        lane.begin = runtime::BeginSummary{.prompt_tokens        = lane.prompt_tokens,
                                           .reused_prompt_tokens = source.reused_tokens,
                                           .prefix_reuse_path =
                                               source.reused_tokens ? source.reuse_path : PrefixReusePath::Root};
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(gate_stats_.p) + kGateStatsBytes * index, 0, kGateStatsBytes,
                                   device_.stream));
        lane.prefill_end = lane.prompt_tokens;
        lane.replay      = false;
        lane.extent      = b.positions;
        // A request paused inside its prompt keeps what it had measured and decided.
        if (resume != nullptr) { restore_request(lane, *resume->impl_); }
        transaction_lane_ = index;
        replaying_        = false;
        ++revision_;
        BindingReservation out;
        out.reserved = true;
        return out;
    }

    // ---------------------------------------------------------------- pause and replay (design §19.3.13)

    // The lane's request-level state a pause keeps (everything but the model state and the pages).
    void save_request(const Lane& lane, ResumeStateImpl& r) const {
        r.begin                   = lane.begin;
        r.prefill_ns              = lane.prefill_ns;
        r.decode_ns               = lane.decode_ns;
        r.decode_share_ns         = lane.decode_share_ns;
        r.vision_seconds          = vision_seconds_of(lane);
        r.speculative             = lane.speculative;
        r.mtp_accept              = lane.mtp_accept;
        r.mtp_policy_rounds       = lane.mtp_policy_rounds;
        r.mtp_cumulative          = lane.mtp_cumulative;
        r.dflash_calibration      = lane.dflash_calibration;
        r.ngram_accept            = lane.ngram_accept;
        r.constraint_trace        = lane.constraint_trace;
        r.cache_at_admission      = lane.cache_at_admission;
        r.cpu_served_at_admission = lane.cpu_served_at_admission;
        r.ngram                   = lane.ngram;
        r.gate                    = lane.gate;
    }

    void restore_request(Lane& lane, const ResumeStateImpl& r) {
        lane.begin                   = r.begin;
        lane.prefill_ns              = r.prefill_ns;
        lane.decode_ns               = r.decode_ns;
        lane.decode_share_ns         = r.decode_share_ns;
        lane.vision_seconds          = r.vision_seconds;
        lane.speculative             = r.speculative;
        lane.mtp_accept              = r.mtp_accept;
        lane.mtp_policy_rounds       = r.mtp_policy_rounds;
        lane.mtp_cumulative          = r.mtp_cumulative;
        lane.dflash_calibration      = r.dflash_calibration;
        lane.ngram_accept            = r.ngram_accept;
        lane.constraint_trace        = r.constraint_trace;
        lane.cache_at_admission      = r.cache_at_admission;
        lane.cpu_served_at_admission = r.cpu_served_at_admission;
        lane.ngram                   = r.ngram;
        lane.gate                    = r.gate;
    }

    // The resumed request's ledger as a prompt: its tokens through the next input, their RoPE
    // positions, and its lookup keys (the prompt's, chained over the output blocks as
    // prefix_publish chains them); no tap hints (a replay publishes blocks, not taps).
    qwen3_5::PreparedPromptData ledger_prompt(const qwen3_5::PreparedPromptData& prompt,
                                              const ResumeStateImpl& r) const;

    // Binds a request paused after its first token: the deepest cached state of its ledger up to
    // the paused frontier (the pause's own capture when it is still cached), then a replay of the
    // ledger to that frontier, without sampling. The pages cover the replay and one round.
    BindingReservation bind_replay(const BasePlanImpl& b, std::uint32_t index, ResumeStateImpl& r) {
        Lane& lane = lanes_[index];
        kv_busy();
        const auto& prompt = qwen3_5::PreparedPromptAccess::view(*b.prompt);
        const qwen3_5::PreparedPromptData ledger = ledger_prompt(prompt, r);
        const std::uint32_t pages = binding_pages(b, r.target + 1U);
        std::optional<PrefixSelection> selection;
        if (prefix_) {
            // The pause's capture lands first, so the selection can see it.
            prefix_->drain();
            selection = b.reuse ? prefix_select(ledger, b, index) : prefix_root(ledger, b);
            if (!selection) {
                BindingReservation out;
                out.shortage.main_kv_pages = pages > pool_->available_pages() ? pages - pool_->available_pages() : 1U;
                return out;
            }
            // A state cached at the paused frontier itself leaves nothing to replay.
            selection->plan = selection->frontier < r.target
                                  ? prefix::plan_calls(selection->frontier, r.target, static_cast<std::uint32_t>(chunk_), {},
                                                       call_cost())
                                  : prefix::CallPlan{};
            prefix_pin(*selection);
        }
        lane.history       = r.history;
        lane.prompt_tokens = r.prompt_tokens;
        lane.rope.prompt_tokens = r.prompt_tokens;
        lane.rope.delta         = prompt.rope_delta;
        if (prompt.has_media()) {
            lane.rope.prompt.assign(prompt.positions.begin(), prompt.positions.end());
        } else {
            lane.rope.prompt.clear();
        }
        lane.state_tokens   = 0;
        lane.vision_seconds = 0;
        lane.vision_timer.reset();
        lane.speculative    = {};
        lane.mtp_cells      = 0;
        lane.mtp_live       = mtp_;
        lane.dflash_frontier = 0;
        lane.row            = tables_->acquire(static_cast<std::int32_t>(index));
        // The items past the resume frontier are encoded again.
        std::shared_ptr<const VisionAdmission> vision;
        if (b.vision) { vision = plan_vision(ledger, selection ? selection->frontier : 0U); }
        vision_make_room(vision.get());
        if (selection) {
            lane.prefix.reuse          = b.reuse;
            lane.prefix.hashes         = ledger.block_hashes;
            lane.prefix.extras         = prompt.block_extras;
            lane.prefix.trailing_extra = prefix_trailing_extra(prompt);
            lane.prefix.hints.clear();
            lane.prefix.exclusions = prefix_exclusions(prompt);
            pc_make_room(selection->need);
            if (pool_->available_pages() < selection->need) { (void)grow_kv(selection->need); }
            auto reservation = pool_->reserve(selection->need);
            if (!reservation) {
                BindingReservation out;
                out.shortage.main_kv_pages = selection->need - std::min(selection->need, pool_->available_pages());
                prefix_unpin(*selection);
                lane.row = KVExecutionRowLease{};
                return out;
            }
            if (vision) {
                vision_reserve(lane, vision, ledger);
                if (auto refused = vision_lend(lane, index)) {
                    lane.vision.reset(); // nothing was lent
                    prefix_unpin(*selection);
                    lane.row = KVExecutionRowLease{};
                    return std::move(*refused);
                }
            }
            const std::uint64_t restored = prefix_->counters().host_restore_bytes;
            prefix_activate(lane, index, *selection, *reservation, pages);
            lane.prefix.cached_tokens  = selection->cached_tokens;
            lane.prefix.restored_bytes = prefix_->counters().host_restore_bytes - restored;
            lane.prefix.taps.clear();
            lane.reservation   = std::move(*reservation);
            lane.prefix.reused = selection->frontier;
            lane.calls         = selection->plan.ends;
        } else {
            if (pool_->available_pages() < pages) { (void)grow_kv(pages); }
            auto reservation = pool_->reserve(pages);
            if (!reservation) {
                BindingReservation out;
                out.shortage.main_kv_pages = pages - std::min(pages, pool_->available_pages());
                lane.row = KVExecutionRowLease{};
                return out;
            }
            if (vision) {
                vision_reserve(lane, vision, ledger);
                if (auto refused = vision_lend(lane, index)) {
                    lane.vision.reset(); // nothing was lent
                    lane.row = KVExecutionRowLease{};
                    return std::move(*refused);
                }
            }
            lane.pages.clear();
            lane.pages.reserve(pages);
            pool_->materialize(*reservation, pages, lane.pages);
            lane.reservation = std::move(*reservation);
            tables_->publish(lane.row.handle(), 0, std::span<const DeviceKVPageLease>(lane.pages), device_.stream);
            reset_slot(index);
            lane.calls = prefix::plan_calls(0, r.target, static_cast<std::uint32_t>(chunk_), {}, call_cost()).ends;
        }
        lane.next_call = 0;
        clear_prompt_uses(lane);
        if (max_width_ > 1 && options_.ngram_draft_tokens > 0) {
            lane.proposer = std::make_unique<qwen3_5::detail::NgramProposer>(proposer_tokens_, proposer_tokens_ / 2);
            for (const auto token : lane.history) { lane.proposer->append(token); }
        }
        lane.sampling = b.sampling;
        lane.sampling.token_counts =
            static_cast<std::int32_t*>(token_counts_.p) + static_cast<std::size_t>(index) * token_domain_;
        *nonfinite_flag(index)  = 0;
        lane.sampling.nonfinite = nonfinite_flag(index);
        rebuild_token_counts(lane);
        lane.readout.clear(); // the prompt's readout was delivered before the pause
        lane.prompt     = b.prompt;
        lane.constraint = b.constraint;
        lane.constraint_round.clear();
        lane.epoch = ++next_epoch_;
        restore_request(lane, r);
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(gate_stats_.p) + kGateStatsBytes * index, 0, kGateStatsBytes,
                                   device_.stream));
        lane.extent      = b.positions;
        lane.prefill_end = r.target;
        // The cached state may reach the paused frontier itself: then nothing replays.
        lane.replay = lane.state_tokens < r.target;
        lane.phase  = lane.replay ? Phase::Prefill : Phase::Decode;
        if (!lane.replay && lane.vision) { vision_release(lane); }
        transaction_lane_ = index;
        replaying_        = lane.replay;
        ++revision_;
        BindingReservation out;
        out.reserved = true;
        return out;
    }

    // Penalty counts over the request's output so far (the sampler counts output tokens only),
    // counted on the host: a binding may overlap a round that still uses the workspace.
    void rebuild_token_counts(const Lane& lane) {
        const cudaStream_t s = device_.stream;
        auto* counts         = lane.sampling.token_counts;
        if (lane.sampling.presence_penalty == 0.0F && lane.sampling.frequency_penalty == 0.0F) {
            CUDA_CHECK(cudaMemsetAsync(counts, 0, 4ULL * token_domain_, s));
            return;
        }
        // A pinned table read by the SMs (upload_pinned), so the upload never waits on the copy engine.
        const std::size_t bytes = 4ULL * static_cast<std::size_t>(token_domain_);
        if (penalty_host_.size() < bytes) { penalty_host_ = PinnedHostBuffer(bytes); }
        auto* host = static_cast<std::int32_t*>(penalty_host_.data());
        std::fill(host, host + token_domain_, 0);
        for (std::size_t i = lane.prompt_tokens; i < lane.history.size(); ++i) {
            const std::int32_t token = lane.history[i];
            if (token >= 0 && token < token_domain_) { ++host[static_cast<std::size_t>(token)]; }
        }
        upload_pinned(counts, host, bytes, s);
        CUDA_CHECK(cudaStreamSynchronize(s)); // the table is read before the next lane's rewrites it
    }

    // Pauses a resident at its committed boundary (design §19.3.13): the prefix cache receives the
    // lane's blocks and its state at the frontier, exactly as a consistent abort leaves them, the
    // lane is released and a Pause transaction hands the Engine the ResumeState.
    bool start_pause(SequenceHandle sequence) {
        if (in_transaction() || pending_transaction_ != 0) {
            diagnostic(std::string("Qwen4Exp: pause refused: ") +
                           (in_transaction() ? "a context transaction is open" : "a round awaits its commit"),
                       DiagnosticLevel::Warning);
            return false;
        }
        const std::uint32_t index = lane_of(sequence);
        Lane& lane                = lanes_[index];
        if (lane.phase != Phase::Prefill && lane.phase != Phase::Decode) {
            diagnostic("Qwen4Exp: pause refused: lane " + std::to_string(index) + " is in phase " +
                           std::to_string(static_cast<int>(lane.phase)),
                       DiagnosticLevel::Warning);
            return false;
        }
        // Inside a layer walk's span the layers are at different positions: the pause publishes
        // nothing past the span's start (its taps did), and the walk is abandoned with the lane.
        const bool walking = walk_.active && walk_.lane == index;
        device_.synchronize();
        auto saved           = std::make_unique<ResumeStateImpl>();
        saved->generated     = lane.replay || lane.phase == Phase::Decode;
        saved->target        = lane.replay ? lane.prefill_end : lane.state_tokens;
        saved->history       = lane.history;
        saved->prompt_tokens = lane.prompt_tokens;
        save_request(lane, *saved);
        // A replaying lane short of its frontier publishes what it has rebuilt so far.
        if (!walking) { prefix_finish(lane, index); }
        saved->capture = lane.prefix.capture;
        release(index);
        pause_.emplace(ResumeState(std::move(saved)));
        ++revision_;
        return true;
    }

    // One prefill call of a resumed lane's ledger toward its paused frontier, without sampling.
    ReplayProgress advance_replay(SequenceHandle sequence) {
        const std::uint32_t index = lane_of(sequence);
        if (!lanes_[index].replay || lanes_[index].phase != Phase::Prefill) {
            throw std::logic_error("Qwen4Exp: replay on a lane that is not replaying");
        }
        const PrefillProgress progress = advance_prefill(sequence, nullptr); // replay samples nothing
        ReplayProgress out;
        out.processed_tokens = progress.processed_prompt_tokens;
        out.complete         = progress.complete;
        out.timing           = progress.timing;
        return out;
    }

    // Grows each unit's lane to its next round (design §19.3.13): a decode or control unit needs the
    // positions it writes, a verification's columns and the drafter's cells, within the request's
    // extent. A shortage cached blocks and the elastic pool cannot cover is reported, and the Engine
    // reclaims or pauses the youngest resident.
    runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit> units) {
        runtime::ResourceReservation out;
        for (const ExecutionUnit& unit : units) {
            const std::uint32_t index = lane_of(unit.sequence);
            Lane& lane                = lanes_[index];
            std::uint32_t end         = lane.state_tokens + round_positions();
            if (unit.kind == ExecutionUnitKind::Control) { end += unit.tokens; }
            if (unit.kind == ExecutionUnitKind::Prefill || unit.kind == ExecutionUnitKind::Replay) {
                end = lane.prefill_end + round_positions();
            }
            const std::uint32_t needed = (std::min(end, lane.extent) + kPagedKVPageSize - 1) / kPagedKVPageSize;
            const std::uint32_t have   = lane.prefix.page_base + static_cast<std::uint32_t>(lane.pages.size());
            if (needed <= have) { continue; }
            const std::uint32_t missing = needed - have;
            kv_busy();
            if (prefix_) { pc_make_room(missing); }
            if (pool_->available_pages() < missing) { (void)grow_kv(missing); }
            if (pool_->available_pages() < missing && prefix_ && prefix_->transfers_pending()) {
                // Blocks pinned only by their in-flight Host writes (a lane just paused or finished)
                // become evictable once those land: wait for them before reporting a shortage.
                prefix_->drain();
                pc_make_room(missing);
            }
            auto reservation = pool_->reserve(missing);
            if (!reservation) {
                out.shortage.main_kv_pages += std::max(1U, missing - std::min(missing, pool_->available_pages()));
                return out;
            }
            std::vector<DeviceKVPageLease> grown;
            grown.reserve(missing);
            pool_->materialize(*reservation, missing, grown);
            tables_->publish(lane.row.handle(), static_cast<std::int32_t>(have), std::span<const DeviceKVPageLease>(grown),
                             device_.stream);
            for (DeviceKVPageLease& page : grown) { lane.pages.push_back(std::move(page)); }
            ++revision_;
        }
        out.reserved = true;
        return out;
    }

    // Advances the binding: a media prompt's encode window one step per call (design §19.3.2; other
    // lanes run between steps), then the lane is published.
    ContextProgress poll_context(runtime::CancellationFlagView cancellation) {
        if (pause_) {
            ContextProgress out{.kind = ContextOperationKind::Pause};
            out.advanced  = true;
            out.complete  = true;
            out.published = true;
            out.paused.emplace(std::move(*pause_));
            pause_.reset();
            return out;
        }
        if (!transaction_lane_) { throw std::logic_error("Qwen4Exp: there is no context operation"); }
        ContextProgress out{.kind = ContextOperationKind::Bind};
        Lane& admitted = lanes_[*transaction_lane_];
        if (cancellation.requested()) {
            release(*transaction_lane_);
            transaction_lane_.reset();
            out.complete = true;
            return out;
        }
        if (admitted.vision && !vision_encoded(admitted)) {
            try {
                vision_step(admitted);
            } catch (...) {
                release(*transaction_lane_);
                transaction_lane_.reset();
                throw;
            }
            out.advanced = true;
            if (!vision_encoded(admitted)) { return out; }
        }
        out.advanced  = true;
        out.complete  = true;
        out.published = true;
        out.replaying = replaying_;
        out.sequence  = handle(*transaction_lane_);
        transaction_lane_.reset();
        return out;
    }

    [[nodiscard]] bool in_transaction() const noexcept { return transaction_lane_.has_value() || pause_.has_value(); }

    // Whether the open binding holds `sequence` back (its lane is still being published).
    // A lane's context is blocked while it is being bound, and a prefilling lane while another
    // lane's layer walk holds the expert stream and the lent workspace (the walk's layers are at
    // different positions until its span ends, so no other prefill call may run in between).
    [[nodiscard]] bool context_blocks(SequenceHandle sequence) const noexcept {
        const std::uint32_t index = ContractAccess::lane(sequence);
        if (transaction_lane_ && *transaction_lane_ == index) { return true; }
        return walk_.active && walk_.lane != index && index < lanes_.size() && lanes_[index].phase == Phase::Prefill;
    }

    // The Engine's reclaim for a binding shortage: evict cached Device blocks until it is covered,
    // then grow the elastic pool, taking back a prefill's loans first if only they block growth.
    bool hybrid_reclaim(runtime::ContextResourceUsage shortage) {
        if (shortage.main_kv_pages == 0) { return false; }
        const std::uint32_t before = pool_->available_pages();
        const std::uint32_t target = before + shortage.main_kv_pages;
        if (prefix_) {
            pc_make_room(target);
            if (pool_->available_pages() == before && prefix_->transfers_pending()) {
                // Blocks pinned only by in-flight Host writes become evictable once those land.
                prefix_->drain();
                pc_make_room(target);
            }
        }
        // A long prompt's stream ring and wide arena stay lent until its last call, and growth is
        // refused while frames are lent, so a request queued behind it waited out its prefill (issue
        // #24). Between calls they can go back: the prompt's next streamed call lends again from
        // the smaller cache, or stages its experts (the same bytes either way, design §19.3.8 F3).
        // Only when the pool's capacity leaves room for the growth: a return that cannot grow the pool
        // only makes the prompt's next walk span lend again (a full --max-context returned 4,221
        // frames for nothing, 2026-10-16).
        const bool room = pool_->capacity_pages() - pool_->backed_pages() >=
                          target - std::min(target, pool_->available_pages());
        if (pool_->available_pages() < target && room && leases_returnable()) {
            return_stream_lease();
            return_wide_work();
            (void)grow_kv(target);
        }
        return pool_->available_pages() > before;
    }

    // Whether every lent frame is held by a lease a round boundary can return (the prefill stream
    // ring and the wide arena; not a layer walk's area or a vision window), so growth needs only
    // their return (hybrid_reclaim).
    [[nodiscard]] bool leases_returnable() const noexcept {
        if (!kv_vmm_ || !residency_ || walk_.active || wide_active_) { return false; }
        const std::uint64_t lent = residency_->stats().lent_frames;
        const std::uint64_t returnable = (stream_lease_.valid() ? stream_lease_.count : 0U) +
                                         (work_lease_.valid() ? work_lease_.count : 0U);
        return lent != 0 && lent == returnable;
    }

    // ---------------------------------------------------------------- execution
    PrefillProgress advance_prefill(SequenceHandle sequence, runtime::TokenMaskProvider* masks) {
        const GrammarScope grammar(*this, masks);
        const auto start  = Clock::now();
        const auto index  = lane_of(sequence);
        Lane& lane        = lanes_[index];
        if (lane.phase != Phase::Prefill) { throw std::logic_error("Qwen4Exp: prefill on a lane that is not prefilling"); }
        if (walk_.active) {
            if (walk_.lane != index) { throw std::logic_error("Qwen4Exp: another lane's prefill holds the layer walk"); }
            return walk_step(lane, index, start);
        }
        const std::int32_t begin = static_cast<std::int32_t>(lane.state_tokens);
        if (lane.next_call >= lane.calls.size() || lane.calls[lane.next_call] <= lane.state_tokens) {
            throw std::logic_error(
                "Qwen4Exp: the lane's prefill plan does not continue its state (lane " + std::to_string(index) +
                ", call " + std::to_string(lane.next_call) + " of " + std::to_string(lane.calls.size()) +
                (lane.next_call < lane.calls.size() ? ", ending at " + std::to_string(lane.calls[lane.next_call]) : "") +
                ", state " + std::to_string(lane.state_tokens) + ", prompt end " + std::to_string(lane.prefill_end) + ")");
        }
        if (const std::size_t end = walk_span_end(lane, index); end > lane.next_call) {
            walk_begin(lane, index, end);
            return walk_step(lane, index, start);
        }
        const std::int32_t width = static_cast<std::int32_t>(lane.calls[lane.next_call++] - lane.state_tokens);
        const bool last          = begin + width == static_cast<std::int32_t>(lane.prefill_end);
        // The lossless route's call reads only its own keys and starts from zero state: it is the whole
        // window (TextConfig::lossless).
        if (c_.lossless && options_.causal_scoring && (begin != 0 || !last)) {
            throw std::logic_error("Qwen4Exp: the lossless route scores a window in one call from its first token");
        }
        // A call the CPU serves (an opener tail, a tiny suffix) promotes at the decode rate (design
        // §19.3.1). Wider prefill calls promote nothing: the answer's staged misses land in free
        // frames and pick its experts, while the prompt's would displace them (16 per layer per
        // call measured slower than none in every regime: serve C = 4 fill -4.3 %, warm -5.6 %;
        // design §19.3.17). Their routes still count toward the cache's scores.
        const std::size_t promotions =
            width <= kServedCallColumns ? decode_budget(static_cast<std::uint32_t>(width)) : 0;
        // Host phases of a chunk for nsys attribution (the GPU side is in the CUDA trace).
        const nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                            static_cast<std::uint64_t>(width));
        {
            const nvtx::ScopedRange rows_range(nvtx::Name::PrefillPleRows, nvtx::Category::Prefill,
                                               static_cast<std::uint64_t>(width));
            stage_sequence(index, begin, width, lane.history);
        }
        const auto chunk = stage_mtp_chunk(lane, index, begin, width);
        // A resumed lane's first call waits per layer for its restore; any call waits per layer for
        // a copy-out of this lane's state still in flight (design §19.3.1).
        next_waits_ = prefix_waits(lane);
        const bool streamed = !walk_.active && stream_chunk(width);
        score_column_       = begin;
        run(1, width, 1, chunk ? &*chunk : nullptr, stage_vision(lane, static_cast<std::uint32_t>(begin), width), streamed);
        if (last && !prefilling_besides(index)) {
            return_stream_lease();
            return_wide_work();
        }
        lane.state_tokens += static_cast<std::uint32_t>(width);
        if (dflash2_) { lane.dflash_frontier = lane.state_tokens; }
        prefix_after_prefill_call(lane, index, last);
        // The handoff is read only by the prompt's calls: it returns once the last is enqueued.
        if (last && lane.vision) { vision_release(lane); }
        if (!last) {
            {
                const nvtx::ScopedRange wait_range(nvtx::Name::DeviceWait, nvtx::Category::Prefill);
                device_.synchronize();
            }
            check_expert_error();
            const nvtx::ScopedRange residency_range(nvtx::Name::PrefillResidency, nvtx::Category::Moe,
                                                    static_cast<std::uint64_t>(width));
            trace_round(RouteTraceKind::PrefillChunk, 1, width, static_cast<std::uint32_t>(width), promotions,
                        static_cast<std::uint32_t>(begin));
            apply_prompt_routes(lane, width, promotions);
            apply_vram_target(false);
        }

        return prefill_progress(lane, index, begin, width, static_cast<std::uint32_t>(width), last, promotions, start);
    }

    // A prefill call's routes into the residency: the lane's prompt uses (note_prompt_routes), then the
    // round's policies. TEMPORARY (INFERNIX_TMP_CONTINUING_PROMPT=<fraction>, candidates CF-01): a prompt
    // whose reused prefix covers at least that fraction of it is a continuing conversation turn. Its
    // calls count toward neither the cache's scores nor the prompt seed and start no promotions (every
    // column passes as not live), so the conversation's decode experts keep their ranks against the
    // turn's new tokens (a tool result, a pasted file).
    void apply_prompt_routes(Lane& lane, std::int32_t width, std::size_t promotions) {
        static const double continuing = [] {
            const auto value = temporary_switches::value("INFERNIX_TMP_CONTINUING_PROMPT");
            const double fraction = value ? std::stod(*value) : 0.0;
            if (value && !(fraction > 0.0 && fraction <= 1.0)) {
                throw std::invalid_argument("INFERNIX_TMP_CONTINUING_PROMPT must be a fraction in (0, 1]");
            }
            return fraction;
        }();
        if (continuing > 0.0 && lane.prompt_tokens != 0 &&
            static_cast<double>(lane.prefix.reused) >= continuing * static_cast<double>(lane.prompt_tokens)) {
            prompt_silent_.assign(static_cast<std::size_t>(width), 0);
            residency_->after_round(device_.stream, width, 0, prompt_silent_, staged_lane());
            ++continuing_calls_;
            return;
        }
        note_prompt_routes(lane, width);
        residency_->after_round(device_.stream, width, promotions, {}, staged_lane());
    }

    // Prompt seeding (design §19.3.27): a prefill call's per-token routes, read from the route log
    // its after_round is about to apply, add to the lane's prompt uses. A layer walk's span counts
    // its last chunk, the one its route log holds.
    void note_prompt_routes(Lane& lane, std::int32_t width) {
        if (width <= 0) { return; }
        const std::uint32_t L = c_.num_hidden_layers, E = c_.moe.experts, K = c_.moe.top_k;
        if (lane.prompt_uses.size() != static_cast<std::size_t>(L) * E) {
            lane.prompt_uses.assign(static_cast<std::size_t>(L) * E, 0);
            lane.prompt_counted = 0;
        }
        const std::int32_t* routes = residency_->route_host();
        const std::size_t stride   = residency_->route_stride();
        for (std::uint32_t l = 0; l < L; ++l) {
            const std::int32_t* ids = routes + static_cast<std::size_t>(l) * stride;
            std::uint32_t* uses     = lane.prompt_uses.data() + static_cast<std::size_t>(l) * E;
            for (std::size_t i = 0; i < static_cast<std::size_t>(width) * K; ++i) {
                const std::int32_t expert = ids[i];
                if (expert >= 0 && static_cast<std::uint32_t>(expert) < E) { ++uses[expert]; }
            }
        }
        lane.prompt_counted += static_cast<std::uint32_t>(width);
    }

    // At the prompt's end, after its last after_round: an expert that every counted token routed to
    // gains kPromptSeedCredit of LFRU count, the others in proportion.
    void seed_prompt_experts(Lane& lane) {
        if (lane.prompt_counted == 0) { return; }
        residency_->seed_prompt(lane.prompt_uses, kPromptSeedCredit / lane.prompt_counted);
        clear_prompt_uses(lane);
    }
    static void clear_prompt_uses(Lane& lane) {
        std::fill(lane.prompt_uses.begin(), lane.prompt_uses.end(), 0U);
        lane.prompt_counted = 0;
    }

    // The progress of a prefill unit whose calls were enqueued (the last ends at `begin + width`,
    // `tokens` prompt tokens in all): for the prompt's last call the first token is sampled and the
    // pending batch returned. Non-last units have already synchronized and applied their routes.
    PrefillProgress prefill_progress(Lane& lane, std::uint32_t index, std::int32_t begin, std::int32_t width,
                                     std::uint32_t tokens, bool last, std::size_t promotions,
                                     Clock::time_point start) {
        PrefillProgress out;
        out.summary                 = prefill_summary(lane);
        out.processed_prompt_tokens = tokens;
        out.complete                = last;
        out.capacity_released       = !last && leases_returnable();
        if (last && lane.replay) {
            // A replay ends at its paused frontier, whose next input the ledger already holds: the
            // lane decodes from there (design §19.3.13).
            {
                const nvtx::ScopedRange wait_range(nvtx::Name::DeviceWait, nvtx::Category::Prefill);
                device_.synchronize();
            }
            trace_round(RouteTraceKind::PrefillChunk, 1, width, static_cast<std::uint32_t>(width), promotions,
                        static_cast<std::uint32_t>(begin));
            apply_prompt_routes(lane, width, promotions);
            seed_prompt_experts(lane);
            apply_vram_target(false);
            lane.replay = false;
            lane.phase  = Phase::Decode;
            out.timing.submit_host_ns = elapsed_ns(start);
            lane.prefill_ns += out.timing.submit_host_ns;
            return out;
        }
        if (last) {
            const std::uint32_t lanes[] = {index};
            const std::int32_t positions[] = {begin + width - 1};
            // The readout reads the logits before the sampler's constraints mask them.
            if (!lane.readout.empty()) { out.readout = read_prompt_frontier(lane); }
            sample(lanes, positions);
            {
                const nvtx::ScopedRange residency_range(nvtx::Name::PrefillResidency, nvtx::Category::Moe,
                                                        static_cast<std::uint64_t>(width));
                trace_round(RouteTraceKind::PrefillChunk, 1, width, static_cast<std::uint32_t>(width), promotions,
                            static_cast<std::uint32_t>(begin));
                apply_prompt_routes(lane, width, promotions);
                seed_prompt_experts(lane);
                apply_vram_target(false);
            }
            const SequenceHandle rows[] = {handle(index)};
            out.timing.submit_host_ns = elapsed_ns(start);
            lane.prefill_ns += out.timing.submit_host_ns;
            out.pending.emplace(ContractAccess::make_pending(this, ++next_transaction_, rows,
                                                             {pending_tokens_.data(), 1}, out.timing));
            mark_row_failures(*out.pending, {});
            plain_round_ = false;
            pending_transaction_ = next_transaction_;
        } else {
            out.timing.submit_host_ns = elapsed_ns(start);
            lane.prefill_ns += out.timing.submit_host_ns;
        }
        return out;
    }

    // The Begin the Engine admitted the request with, fixed at binding (a snapshot evicted since
    // must not change it).
    const runtime::BeginSummary& prefill_summary(const Lane& lane) const noexcept { return lane.begin; }

    PendingBatch decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
                        runtime::TokenMaskProvider* masks) {
        const GrammarScope grammar(*this, masks);
        const auto start = Clock::now();
        const auto batch = static_cast<std::int32_t>(sequences.size());
        if (batch <= 0 || batch > static_cast<std::int32_t>(options_.max_concurrency)) {
            throw std::logic_error("Qwen4Exp: decode batch is empty or too large");
        }
        if (replay_clock_) { replay_clock_->begin(device_.stream); } // ends after its commit (end_round_clock)
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
        // fewer than it may still emit, so every verified position lies in its reservation. A longer
        // n-gram copy proposal replaces a row's MTP drafts. By round size (design 11.3, 19.3.5):
        // - one row: its own draft length, maximizing its expected tokens per round time;
        // - two rows, with the adaptive-speculation fast path: one draft length for both
        //   (choose_pair_draft_length), so no row is padded to the other's; its at most 2 x 3 = 6
        //   columns stay on the column-invariant dense routes, so greedy output equals C = 1, and
        //   below the 7-column fork and CPU-job limits; the shared length couples the rows, and a
        //   pair round's output differed from the rows alone in 1-4 of 96 segments (2026-10-14);
        //   with DFlash2 (dflash2-pairs) both rows take the proposal's first drafts and, with
        //   --draft-tokens auto, the round's one width is chosen after it (dflash2_pair_width);
        // - otherwise two or more rows decode plain (one draft each measured slower at four rows).
        // A plain round's W = 1 catch-up keeps each drafter current.
        // DFlash2 rounds of three or four rows at one shared length lost (4 rows 0.78x, removed 2026-10-28).
        const bool pair_round = options_.pair_speculation && batch == 2 && (mtp_ || dflash2_);
        const std::int32_t pair_k = pair_round ? choose_pair_draft_length(lanes_[lanes[0]], lanes_[lanes[1]]) : 0;
        const bool speculate = batch == 1 || pair_k > 0;
        std::array<std::int32_t, kMaximumConcurrency> wanted{};
        std::int32_t steps = 0;
        if (mtp_ || dflash2_) {
            for (std::int32_t b = 0; b < batch; ++b) {
                Lane& lane = lanes_[lanes[b]];
                wanted[b]  = speculate && drafter_live(lane) ? (batch == 1 ? choose_draft_length(lane) : pair_k) : 0;
                steps      = std::max(steps, wanted[b]);
            }
        }
        observe_round(start, batch);
        // Each row's draft count and source are fixed before the drafter runs (an MTP row's count
        // is its draft length; a longer n-gram copy proposal replaces its drafts), so the round's
        // width, and with it every row's n-gram landing columns, is known while it drafts.
        std::array<std::int32_t, kMaximumConcurrency> extents{};
        std::int32_t width = 1;
        for (std::int32_t b = 0; b < batch && max_width_ > 1; ++b) {
            Lane& lane = lanes_[lanes[b]];
            drafts_[b].clear();
            from_ngram_[b] = false;
            const std::uint32_t remaining =
                static_cast<std::size_t>(b) < budgets.size() ? budgets[b].generated_tokens_remaining : 0U;
            if (remaining < 2) { continue; }
            auto limit = std::min<std::uint32_t>(static_cast<std::uint32_t>(max_width_ - 1), remaining - 1U);
            if (batch > 1) { limit = std::min<std::uint32_t>(limit, static_cast<std::uint32_t>(pair_k)); }
            if (drafter_live(lane) && wanted[b] > 0) {
                extents[b] = static_cast<std::int32_t>(std::min<std::uint32_t>(limit, static_cast<std::uint32_t>(wanted[b])));
            }
            ngram_bucket_[b] = -1;
            if (speculate && lane.proposer) {
                auto match = lane.proposer->propose(lane.history, limit, options_.ngram_min_match);
                const int bucket = ngram_bucket(match.matched);
                // A longer copy proposal replaces the row's MTP drafts; a one-row round takes it only when
                // its expected tokens per round time (its bucket's learned acceptance per position) beat the
                // MTP drafts' (Strata's DraftPolicy).
                bool take = match.tokens.size() > static_cast<std::size_t>(extents[b]);
                if (take && batch == 1) {
                    const double cost = width_cost();
                    double reach = 1.0, mtp_tokens = 1.0;
                    for (std::int32_t k = 0; k < extents[b]; ++k) {
                        reach *= lane.mtp_accept[static_cast<std::size_t>(k)];
                        mtp_tokens += reach;
                    }
                    const double p = lane.ngram_accept[static_cast<std::size_t>(bucket)];
                    double copy_reach = 1.0, copy_tokens = 1.0;
                    for (std::size_t k = 0; k < match.tokens.size(); ++k) {
                        copy_reach *= p;
                        copy_tokens += copy_reach;
                    }
                    take = copy_tokens / (1.0 + cost * static_cast<double>(match.tokens.size())) >
                           mtp_tokens / (1.0 + cost * extents[b]);
                }
                if (take) {
                    drafts_[b]       = std::move(match.tokens);
                    from_ngram_[b]   = true;
                    ngram_bucket_[b] = bucket;
                    extents[b]       = static_cast<std::int32_t>(drafts_[b].size());
                }
            }
            width = std::max(width, 1 + extents[b]);
        }
        if (batch == 1 && mtp_ && !from_ngram_[0]) { last_round_steps_ = steps; }
        // Columns of each verification row whose n-gram rows are already staged: the drafter's
        // wait reads every committed-token column (the anchor's) and all of a copy row's.
        std::array<std::int32_t, kMaximumConcurrency> staged{};
        if (mtp_) {
            mtp_draft(std::span<const std::uint32_t>(lanes.data(), batch), steps, [&] {
                if (width == 1) { return; }
                for (std::int32_t b = 0; b < batch; ++b) {
                    staged[b] = from_ngram_[b] ? width : 1;
                    stage_verify_ngram(lanes_[lanes[b]], static_cast<std::size_t>(b), positions[b], width, 0, staged[b]);
                }
            });
            for (std::int32_t b = 0; b < batch && max_width_ > 1; ++b) {
                if (!from_ngram_[b] && extents[b] > 0) {
                    drafts_[b].assign(mtp_drafts_[b].begin(), mtp_drafts_[b].begin() + extents[b]);
                }
            }
            // A one-row round verifies only its confident drafts: those its cumulative cut keeps
            // (read_mtp_drafts; every draft of a sampled row with sampled-mtp-drafts). Its width may
            // still shrink, as the drafter staged only column 0's n-gram rows; a pair round's
            // columns are staged at their offsets, so it keeps its length.
            if (batch == 1 && !from_ngram_[0] && extents[0] > 0) {
                extents[0] = std::min(extents[0], mtp_confident_[0]);
                drafts_[0].resize(static_cast<std::size_t>(extents[0]));
                width = 1 + extents[0];
            }
        } else if (dflash2_ && width > 1) {
            // One proposal for every row that takes the drafter's tokens (none when every speculating
            // row copies an n-gram), while the host stages the verification's n-gram rows.
            bool neural = false;
            for (std::int32_t b = 0; b < batch; ++b) { neural = neural || (!from_ngram_[b] && extents[b] > 0); }
            const auto stage_rows = [&] {
                for (std::int32_t b = 0; b < batch; ++b) {
                    staged[b] = from_ngram_[b] ? width : 1;
                    stage_verify_ngram(lanes_[lanes[b]], static_cast<std::size_t>(b), positions[b], width, 0, staged[b]);
                }
            };
            if (neural) {
                dflash2_propose(std::span<const std::uint32_t>(lanes.data(), batch), stage_rows);
                for (std::int32_t b = 0; b < batch; ++b) {
                    if (!from_ngram_[b] && extents[b] > 0) {
                        drafts_[b].assign(dflash_drafts_[b].begin(), dflash_drafts_[b].begin() + extents[b]);
                    }
                }
                // A one-row round verifies its drafts up to the first whose selector probability is
                // below --spec-min-p, and with --draft-tokens auto only the calibrated length of them
                // (draft_length.h). The proposal staged only column 0's n-gram rows.
                if (batch == 1 && !from_ngram_[0] && extents[0] > 0) {
                    extents[0] = std::min(extents[0], dflash_confident_[0]);
                    if (options_.auto_draft_length) { extents[0] = dflash2_auto_length(lanes_[lanes[0]], extents[0]); }
                    drafts_[0].resize(static_cast<std::size_t>(extents[0]));
                    width = 1 + extents[0];
                } else if (batch == 2 && options_.auto_draft_length) {
                    // A pair round (dflash2-pairs) verifies one width for both rows, chosen from both rows'
                    // drafts (dflash2_pair_width); its drafter rows keep that many.
                    const auto rows = static_cast<std::size_t>(batch);
                    const std::int32_t k = dflash2_pair_width(std::span<const std::uint32_t>(lanes.data(), rows),
                                                              std::span<const std::int32_t>(extents.data(), rows));
                    std::int32_t pair_width = 1;
                    for (std::int32_t b = 0; b < batch; ++b) {
                        if (!from_ngram_[b] && extents[b] > k) {
                            extents[b] = k;
                            drafts_[b].resize(static_cast<std::size_t>(k));
                        }
                        pair_width = std::max(pair_width, 1 + extents[b]);
                    }
                    // The proposal staged row b's columns at b x the earlier width: row 0's stay in
                    // place, row 1's are read again at the round's width.
                    if (pair_width != width) {
                        for (std::int32_t b = 1; b < batch; ++b) { staged[b] = 0; }
                        width = pair_width;
                    }
                }
                neural = false;
                for (std::int32_t b = 0; b < batch; ++b) { neural = neural || (!from_ngram_[b] && extents[b] > 0); }
            }
            dflash_round_ = neural && width > 1;
        }
        if (width > 1) {
            return verify(sequences, std::span<const std::uint32_t>(lanes.data(), batch),
                          std::span<const std::int32_t>(positions.data(), batch), width,
                          std::span<const std::int32_t>(staged.data(), batch), start);
        }
        for (std::int32_t b = 0; b < batch; ++b) { ++lanes_[lanes[b]].speculative.fallback_steps; }
        round_width_ = 1;
        plain_round_ = true;
        const std::span<const std::uint32_t> round_lanes(lanes.data(), static_cast<std::size_t>(batch));
        const std::span<const std::int32_t> round_positions(positions.data(), static_cast<std::size_t>(batch));
        // A replayed round reads its rows after the launch, while the GPU embeds and runs layer 0,
        // and releases them to the gate before ple_embed (design §12.3, n-gram S4b); a round that
        // warms or captures its graph reads them first.
        // A round whose rows are all in the host cache reads nothing, so it stages them first.
        const bool screened = screened_round(round_lanes, batch);
        DecodeGraph& graph  = decode_graph(batch, screened);
        const bool gated    = graph.executable.ready() && !decode_rows_cached(round_lanes, round_positions);
        stage_decode(round_lanes, round_positions);
        const std::uint32_t word = next_gate_word();
        if (!gated) {
            stage_decode_rows(round_lanes, round_positions);
            publish_pinned_word(gate_ready(), word);
        }
        run_decode(batch, round_positions[0], graph, screened);
        GateRelease release_gate{gated ? this : nullptr, word, io_prefix(batch) - io_layout_.ngram};
        for (std::int32_t b = 0; b < batch; ++b) { ++lanes_[lanes[b]].state_tokens; }
        sample(round_lanes, round_positions, [&] {
            if (!gated) { return; }
            read_behind_gate(lanes_[lanes[0]], word, [&] { stage_decode_rows(round_lanes, round_positions); });
            release_gate.owner = nullptr;
        });
        deferred_ = {.columns = batch, .tokens = 1, .live = false, .pending = true};
        runtime::ExecutionTiming timing;
        timing.submit_host_ns = elapsed_ns(start);
        for (std::int32_t b = 0; b < batch; ++b) {
            lanes_[lanes[b]].decode_ns += timing.submit_host_ns;
            lanes_[lanes[b]].decode_share_ns += timing.submit_host_ns / static_cast<std::uint64_t>(batch);
        }
        pending_transaction_  = ++next_transaction_;
        auto pending = ContractAccess::make_pending(this, pending_transaction_, sequences,
                                                    {pending_tokens_.data(), static_cast<std::size_t>(batch)}, timing);
        mark_row_failures(pending, {});
        return pending;
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
            check_expert_error();
            trace_round(RouteTraceKind::ForcedTokens, 1, static_cast<std::int32_t>(stride), stride,
                        kDecodePromotionsPerLayer, static_cast<std::uint32_t>(begin));
            residency_->after_round(device_.stream, static_cast<std::int32_t>(stride), kDecodePromotionsPerLayer, {},
                                    staged_lane());
            apply_vram_target(false);
            lane.state_tokens += stride;
            if (dflash2_) { lane.dflash_frontier = lane.state_tokens; }
        }
        runtime::ExecutionTiming timing;
        timing.submit_host_ns = elapsed_ns(start);
        return timing;
    }

    // The request's cumulative speculative work, which the Engine differences per commit into its
    // live counters. The round's drafts and acceptance were counted when it was submitted. Without
    // speculation (max_width_ == 1) every plain round still counts a fallback step in the request's
    // stats; that is no speculative work, so the live counters stay at zero.
    CommitRowResult::SpeculativeCounters speculative_counters(const SpeculativeStats& stats) const noexcept {
        if (max_width_ <= 1) { return {}; }
        return {.rounds          = stats.rounds,
                .drafted_tokens  = stats.drafted_tokens,
                .accepted_tokens = stats.accepted_tokens,
                .fallback_steps  = stats.fallback_steps};
    }

    CommitResult commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions) {
        if (ContractAccess::owner(pending) != this || ContractAccess::transaction(pending) != pending_transaction_ ||
            pending_transaction_ == 0) {
            throw std::logic_error("Qwen4Exp: commit of a foreign or stale pending batch");
        }
        const auto rows = ContractAccess::rows(pending);
        if (decisions.size() != rows.size()) { throw std::logic_error("Qwen4Exp: commit decisions are not row aligned"); }
        // A failed row (its PendingRowFailure) commits nothing and is released like a cancelled one.
        for (std::size_t row = 0; row < rows.size(); ++row) {
            const auto& d = decisions[row];
            if ((d.cancelled && d.failed) ||
                (!d.cancelled && d.failed != (pending.row_failure(row) != runtime::PendingRowFailure::None)) ||
                (d.failed && (d.accepted_tokens != 0 || !d.terminal))) {
                throw std::logic_error("Qwen4Exp: a commit decision is invalid");
            }
        }
        if (round_width_ > 1) { return commit_verified(std::move(pending), decisions); }
        CommitResult out;
        out.row_count = rows.size();
        // After a plain decode round the drafter writes the round's cell (its pending cell moves
        // one position); after the prefill's first token the pending cell stays the prompt's last.
        const bool catch_up = (mtp_ || dflash2_) && plain_round_;
        std::int32_t* commit = catch_up ? spec_host(spec_layout_.commit) : nullptr;
        bool any_cell        = false;
        for (std::size_t row = 0; row < rows.size(); ++row) {
            const auto index = lane_of(rows[row]);
            Lane& lane       = lanes_[index];
            const auto& d    = decisions[row];
            if (commit != nullptr) { commit[row] = 0; }
            out.rows[row].speculative_counters = speculative_counters(lane.speculative);
            if (d.cancelled || d.failed) {
                if (catch_up && lane.mtp_live) {
                    // MR2: decode() advanced the state past the round's position at submission and
                    // a cancelled row skips the catch-up, so that position's cell becomes the
                    // pending one: its residual, which the round exported, becomes the saved one.
                    CUDA_CHECK(cudaMemcpyAsync(saved_column(index).data,
                                               static_cast<const std::byte*>(mtp_residuals_.p) + 2ULL * width_ * row,
                                               2ULL * width_, cudaMemcpyDeviceToDevice, device_.stream));
                    lane.mtp_cells = lane.state_tokens - 1;
                }
                // Non-finite logits: what the request published may hold that state, so its release
                // withdraws it (prefix_withdraw).
                lane.prefix.withdraw =
                    pending.row_failure(row) == runtime::PendingRowFailure::NonFiniteLogits;
                lane.constraint_round.clear();
                out.rows[row].timings     = timings(lane);
                out.rows[row].speculative = lane.speculative;
                out.rows[row].disposition = d.failed ? runtime::CommitDisposition::FailedReleased
                                                     : runtime::CommitDisposition::CancelledReleased;
                continue;
            }
            if (d.accepted_tokens != 1) { throw std::logic_error("Qwen4Exp: a row commits exactly one token"); }
            commit_constraint_draws(lane, 1);
            lane.history.push_back(pending_tokens_[row]);
            if (commit != nullptr && (lane.mtp_live || dflash2_)) {
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
            if (mtp_) {
                mtp_catch_up(rows, 1, commit);
            } else {
                dflash2_catch_up(rows, 1, commit);
            }
        }
        for (const auto& row : rows) { require_mtp_settled(lanes_[ContractAccess::lane(row)]); }
        for (std::size_t row = 0; row < rows.size(); ++row) {
            if (!decisions[row].cancelled && !decisions[row].failed) {
                Lane& lane = lanes_[ContractAccess::lane(rows[row])];
                prefix_publish(lane, prefix_frontier(lane, false));
            }
        }
        settle_round();
        if (replay_clock_) { replay_clock_->end(device_.stream); }
        for (std::size_t row = 0; row < rows.size(); ++row) {
            if (decisions[row].cancelled || decisions[row].failed) { release(ContractAccess::lane(rows[row])); }
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
            if (lanes_[index].phase != Phase::Finishable) {
                diagnostic("Qwen4Exp: finish of a lane that is not finishable", DiagnosticLevel::Error);
                return out;
            }
            if (!mtp_settled(lanes_[index])) {
                diagnostic("Qwen4Exp: a finished request's MTP cells are out of step with its state", DiagnosticLevel::Warning);
            }
            out.timings           = timings(lanes_[index]);
            out.speculative       = lanes_[index].speculative;
            out.constrained_draws = std::move(lanes_[index].constraint_trace);
            if (residency_) {
                const auto& s = residency_->stats();
                const auto& a = lanes_[index].cache_at_admission;
                out.expert_cache = ExpertCacheStats{.routed        = s.routed - a.routed,
                                                    .hits          = s.hits - a.hits,
                                                    .decode_routed = s.decode_routed - a.decode_routed,
                                                    .decode_hits   = s.decode_hits - a.decode_hits};
            }
            report_cache(lanes_[index]);
            report_ngram(lanes_[index]);
            prefix_finish(lanes_[index], index);
            // The prompt tokens past the reused prefix that this request's accepted captures made
            // reusable for a later request (Anthropic's cache_creation_input_tokens).
            if (const Lane& lane = lanes_[index]; prefix_ && lane.prefix.reuse) {
                const std::uint32_t deepest     = std::min(lane.prefix.deepest, lane.prompt_tokens);
                out.cache_created_prompt_tokens = deepest > lane.prefix.reused ? deepest - lane.prefix.reused : 0U;
            }
            release(index);
            out.status = runtime::ConsumeStatus::Consumed;
        } catch (const std::exception& error) {
            // The Engine fails a terminal sequence that does not finish; say why.
            diagnostic(std::string("Qwen4Exp: finish failed: ") + error.what(), DiagnosticLevel::Error);
        } catch (...) {
            diagnostic("Qwen4Exp: finish failed", DiagnosticLevel::Error);
        }
        return out;
    }

    AbortResult abort(SequenceHandle sequence) noexcept {
        AbortResult out;
        try {
            const auto index = lane_of(sequence);
            out.timings      = timings(lanes_[index]);
            out.speculative  = lanes_[index].speculative;
            // A consistent abort (no unit in flight) leaves an endpoint a resend resumes from; inside
            // a layer walk the layers are at different positions, so it leaves none.
            if (pending_transaction_ == 0 && lanes_[index].phase != Phase::Free && !(walk_.active && walk_.lane == index)) {
                prefix_finish(lanes_[index], index);
            }
            release(index);
            out.status = runtime::ConsumeStatus::Consumed;
        } catch (...) {}
        return out;
    }

    // After a RecoverableExecutionError: every lane is released without publishing anything (the
    // failed round's state is unknown), the cache's transfers in flight land, and a cache the
    // failure may have published into is emptied. False when that fails (the device itself failed;
    // the Engine then fails everything).
    bool recover_after_failure() noexcept {
        try {
            for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
                if (lanes_[i].phase != Phase::Free) { release(i); }
            }
            transaction_lane_.reset();
            pending_transaction_ = 0;
            deferred_.pending    = false; // the failed round's cache update
            device_.synchronize();
            if (prefix_) {
                prefix_->drain();
                if (clear_prefix_on_release_) {
                    prefix_->clear();
                    prefix_withdrawal_ = {}; // its references would name the rebuilt index's entries
                }
            }
            clear_prefix_on_release_ = false;
            return true;
        } catch (...) {
            return false;
        }
    }

    void release_all() noexcept {
        for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
            if (lanes_[i].phase != Phase::Free) { release(i); }
        }
        transaction_lane_.reset();
        pending_transaction_ = 0;
        try { device_.synchronize(); } catch (...) {}
        if (clear_prefix_on_release_) { // a failed round may have published state it never computed
            if (prefix_) { prefix_->clear(); }
            prefix_withdrawal_       = {};
            clear_prefix_on_release_ = false;
        }
    }

    PhysicalUsageSnapshot usage() const noexcept {
        PhysicalUsageSnapshot out;
        for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
            if (lanes_[i].phase != Phase::Free) { ++out.occupied.state_slots; }
        }
        out.capacity.state_slots   = options_.max_concurrency;
        out.occupied.main_kv_pages = pool_->allocated_pages();
        out.capacity.main_kv_pages = pool_->capacity_pages();
        return out;
    }

    MemorySummary memory() const noexcept {
        MemorySummary out;
        out.max_context             = options_.max_context;
        out.kv_capacity             = kv_pages_ * static_cast<std::uint32_t>(kPagedKVPageSize);
        out.kv_capacity_page_groups = kv_pages_;
        out.kv_capacity_max_page_groups = kv_pages_;
        out.kv_cache                = options_.kv_cache;
        out.kv_payload_bytes        = kv_vmm_ ? kv_vmm_->mapped_bytes() : kv_backing_.bytes;
        out.workspace_logical_peak_bytes = work_capacity_;
        return out;
    }

    std::uint64_t revision() const noexcept { return revision_; }

private:
    // Promotions per layer call (design section 19.2): one per decode round measured fastest; a
    // promotion moves as many PCIe bytes as serving the expert once zero-copy.
    static constexpr std::size_t kDecodePromotionsPerLayer  = 1;
    // Once the frames are full, decode promotes only every kDecodePromotionInterval-th round on the
    // reference link (promotion_interval scales it to the measured one): a promotion moves as many
    // PCIe bytes as serving its expert once, and fewer of them measured faster at 512 tokens (design
    // section 19.2). Until then every round promotes, so a cold cache fills at the full rate.
    static constexpr std::uint64_t kDecodePromotionInterval = 4;
    // Misses staged per pass of a layer's experts: every decode and verify call's misses in one
    // pass; a prefill chunk's in several. The least, and with the SSD tier the count (plan_device);
    // the fork route takes at most 512 (ops::moe_experts).
    static constexpr std::int32_t kStagingSlots = 64, kMaxStagingSlots = 512;
    // The SSD tier's smallest RAM (design §19.3.7 H_min: 1,024 slots, 2.6 GiB).
    static constexpr std::uint32_t kMinTierSlots = 1024;
    // RAM slots a layer walk's span borrows for its stream reads (at most 1/8 of the SSD tier's): with
    // reads kReadAhead layers ahead, each layer streams up to 1,280 / 4 = 320 SSD-only experts. 640
    // left about a quarter of the FP8 artifact's SSD-only experts to the tier's demand reads in a 32K
    // prefill (7,680 of 10,156 streamed); 1,280 streams them all: 32K prefill 2,850-2,867 -> 3,945-3,986
    // tok/s, and the decode after it no slower (2026-10-16). A fixed pool measured -8 % F-A decode
    // (fewer RAM residents); lent per span it costs decode nothing.
    static constexpr std::uint32_t kStreamReadSlots = 1280;
    // Unpinned records a layer may stream through the bounce, per part: the layer's unpinned share
    // of its experts, within these bounds (the bounce comes out of the pinned share). RTX 5090 at
    // x8, 8K-token prefill: 25.4 GiB of 63.3 unpinned ran at 2,350 tok/s with 160 and 2,673 with 256;
    // 18.4 of 42.8 GiB unpinned (24.4 pinned) at 1,144 with 160 and 1,082 with 256.
    static constexpr std::uint32_t kStreamBounceMin = 64, kStreamBounceMax = 256;
    // Promotions follow the tokens a round advances (the longest row's), not rounds: a verification
    // round that advances four tokens promotes what four decode rounds would, so a speculative
    // round's cache warms per token as plain decode does.
    std::size_t decode_budget(std::uint32_t tokens) {
        // The fill phase: until this session promoted or landed as many experts as there are frames.
        // Seeded frames (a warm start) do not end it: promotions only follow misses, so a workload
        // the saved state matches promotes little anyway, while one it does not match replaces the
        // seeded experts at the fill rate instead of the steady-state rate.
        const ExpertResidency::Stats& stats = residency_->stats();
        if (stats.promotions + stats.landed < residency_->frames()) {
            return kDecodePromotionsPerLayer * tokens;
        }
        const std::uint64_t interval = promotion_interval();
        budget_tokens_ += tokens;
        const std::uint64_t due = budget_tokens_ / interval;
        budget_tokens_ %= interval;
        return static_cast<std::size_t>(due) * kDecodePromotionsPerLayer;
    }
    // The interval at the measured link (design §19.3.12): a promotion's bytes cost less wall time
    // on a faster link, whose rounds leave it more idle, so it promotes proportionally more often
    // (x8 4, x16 2; the x16 value is derived, not measured).
    std::uint64_t promotion_interval() const noexcept {
        const double scaled = static_cast<double>(kDecodePromotionInterval) * kReferenceLinkBytesPerSecond / link_bytes_per_second_;
        return static_cast<std::uint64_t>(std::clamp(std::lround(scaled), 1L, 8L));
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
        const std::int32_t rows = deferred_.live ? deferred_.columns / round_width_ : deferred_.columns;
        residency_->after_round(device_.stream, deferred_.columns, budget,
                                deferred_.live ? std::span<const std::uint8_t>(live_.data(),
                                                                               static_cast<std::size_t>(deferred_.columns))
                                               : std::span<const std::uint8_t>{},
                                std::span<const std::int32_t>(host_lanes_.data(), static_cast<std::size_t>(rows)));
        if (drafter_pool_) { // the drafter pool: before the next round's drafts
            drafter_pool_->step(device_.stream);
            report_drafter_pool();
        }
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
        if (kv_idle_due() && waker_) { waker_(); }
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
                       " MiB; the expert cache cannot shrink on this system, so Infernix memory may move to system "
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

    // ---- warm start (design §19.3.5 S4b) ----
    // Fills the frames from the saved state, before the first request; without one (a first start, a
    // state of another artifact, or no file given: the bench and the tools), from the calibrated expert
    // order when the model has its shape, unless --expert-state off asked for an empty cache. With the
    // SSD tier, RAM is filled first (the frames load from it), then the seeds' RAM copies are released
    // and their slots refilled with the next keys of the fill order, so the two levels hold different
    // experts. Only a given file is read or saved.
    void warm_start_experts() {
        const auto keys = c_.num_hidden_layers * c_.moe.experts;
        expert_cache::ExpertStateLoad load;
        SeedSource source = SeedSource::None;
        if (options_.expert_cache) {
            if (!options_.expert_state.empty()) {
                load = expert_cache::load_expert_state(options_.expert_state, options_.expert_state_identity, keys);
            }
            if (load.state) {
                source = SeedSource::Saved;
            } else if (options_.expert_seed && has_calibrated_order()) {
                if (!options_.expert_state.empty()) {
                    diagnostic("expert cache: no saved state used: " + load.message, DiagnosticLevel::Debug);
                }
                load.state = calibrated_expert_order();
                source     = SeedSource::Calibrated;
            }
        }
        const expert_cache::SavedState* state = load.state ? &*load.state : nullptr;
        TierFill fill;
        if (tier_) {
            fill = prefill_tier(state, source);
            // The frames seed along the whole fill order, which starts with the saved ranking: when that ranking
            // holds fewer experts than the frames' share (a session of the full layout, whose warm start seeds half
            // the frames), the next keys of the order take the rest, so the frames and RAM together hold every expert
            // as the exclusive layout sizes RAM for (2026-10-27: Dense8 seeded 5,188 of its 6,306 and left 796 experts
            // on the SSD).
            if (load.state && load.state->ranked.size() < fill.order.size()) { load.state->ranked = fill.order; }
        }
        const std::uint32_t seeded = warm_start_frames(load, source);
        if (tier_) { finish_tier_fill(fill, seeded); }
    }

private:
    // Where the warm start's ranking and counts came from.
    enum class SeedSource { None, Saved, Calibrated };

    // The calibrated order (program/calibrated_order.h) ranks Qwen3.8-Flash-Next's experts: it
    // applies to every model of that shape (NInfer's guard), whatever the artifact's formats.
    [[nodiscard]] bool has_calibrated_order() const noexcept {
        return c_.hidden_size == 2560 && c_.num_hidden_layers == kCalibratedLayers &&
               c_.moe.experts == kCalibratedExperts && c_.moe.top_k == 10 &&
               c_.moe.intermediate == 640 && c_.moe.shared_intermediate == 640;
    }

    // The SSD tier's startup fill: its key order, how many keys the first pass read, and its time.
    struct TierFill {
        std::vector<std::uint32_t> order;
        std::uint32_t filled = 0;
        double seconds       = 0;
        SeedSource source    = SeedSource::None;
    };

    // The SSD tier's RAM fill order (design §19.3.7 §4.9): the saved state's ranking (the last
    // session's VRAM residents, best first), then every other key that session used, most used
    // first, then the calibrated order's ranking of the rest (unless --expert-state off), then the keys
    // left one expert of every layer in turn, so that no layer is left wholly on the SSD (file order
    // filled the first layers and left the last ones on disk).
    [[nodiscard]] std::vector<std::uint32_t> tier_fill_order(const expert_cache::SavedState* state) const {
        const std::uint32_t layers = c_.num_hidden_layers, experts = c_.moe.experts, keys = layers * experts;
        if (tier_->keys() != keys) { throw std::logic_error("Qwen3.8-Flash-Next: the SSD tier has another key count"); }
        // A key's place in the layer-interleaved order: expert e of every layer before expert e + 1.
        const auto spread = [&](std::uint32_t key) { return (key % experts) * layers + key / experts; };
        std::vector<std::uint32_t> order;
        order.reserve(keys);
        std::vector<std::uint8_t> listed(keys, 0);
        if (state != nullptr) {
            for (const std::uint32_t key : state->ranked) {
                if (key < keys && listed[key] == 0) {
                    order.push_back(key);
                    listed[key] = 1;
                }
            }
            if (state->counts.size() == keys) {
                std::vector<std::uint32_t> used;
                for (std::uint32_t key = 0; key < keys; ++key) {
                    if (listed[key] == 0 && state->counts[key] != 0) { used.push_back(key); }
                }
                std::sort(used.begin(), used.end(), [&](std::uint32_t a, std::uint32_t b) {
                    const std::uint32_t ca = state->counts[a], cb = state->counts[b];
                    return ca != cb ? ca > cb : spread(a) < spread(b);
                });
                for (const std::uint32_t key : used) {
                    order.push_back(key);
                    listed[key] = 1;
                }
            }
        }
        // Then the calibrated order (unless --expert-state off): it ranks every key, so the keys a saved session
        // never used follow its ranking rather than the file's.
        if (options_.expert_seed && has_calibrated_order()) {
            for (const std::uint32_t key : calibrated_expert_order().ranked) {
                if (key < keys && listed[key] == 0) {
                    order.push_back(key);
                    listed[key] = 1;
                }
            }
        }
        for (std::uint32_t i = 0; i < keys; ++i) {
            const std::uint32_t key = (i % layers) * experts + i / layers; // spread(key) == i
            if (listed[key] == 0) { order.push_back(key); }
        }
        return order;
    }

    // The SSD tier's first RAM pass, before the frames' warm start reads from it.
    TierFill prefill_tier(const expert_cache::SavedState* state, SeedSource source) {
        const auto start = std::chrono::steady_clock::now();
        TierFill fill;
        fill.source = source;
        fill.order = tier_fill_order(state);
        if (state != nullptr && state->counts.size() == tier_->keys()) {
            tier_->controller().seed_uses(state->counts, kSeedCountCap);
        }
        fill.filled  = tier_->prefill(fill.order);
        fill.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return fill;
    }

    // After the frames' warm start: the seeds are in VRAM, so their RAM copies (shadows) are
    // released and their slots take the next keys of the fill order. A seed VRAM later evicts is
    // demoted into RAM when it outranks the coldest resident (T4), as any key without a host copy.
    // Slots still free once every other key has a host copy (the exclusive layout's reserve) hold
    // shadows of the least-used seeds, the first a prefill's loan or decode evicts, so those
    // evictions keep a RAM copy instead of leaving the expert to the SSD.
    void finish_tier_fill(const TierFill& fill, std::uint32_t seeded) {
        const auto start        = std::chrono::steady_clock::now();
        std::uint32_t released  = 0, refilled = 0;
        if (seeded > 0) {
            released = tier_->controller().release_shadows();
            if (released > 0) {
                refilled = tier_->prefill(std::span<const std::uint32_t>(fill.order).subspan(fill.filled));
            }
        }
        // The unpinned records take the next keys of the order that neither the slots nor the frames
        // hold: they keep them for good, so a copy of a VRAM seed would be wasted.
        const std::int32_t* in_vram = residency_->host_table();
        const std::uint32_t unpinned =
            tier_->fill_unpinned(fill.order, [in_vram](std::uint32_t key) { return in_vram != nullptr && in_vram[key] >= 0; });
        std::uint32_t shadows = 0;
        if (seeded > 0 && in_vram != nullptr) {
            std::vector<std::uint32_t> least_used_seeds;
            for (auto it = fill.order.rbegin(); it != fill.order.rend(); ++it) {
                if (in_vram[*it] >= 0) { least_used_seeds.push_back(*it); }
            }
            shadows = tier_->prefill_shadows(least_used_seeds);
        }
        const double seconds =
            fill.seconds + std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const std::uint32_t keys = tier_->keys(), in_ram = fill.filled - released + refilled;
        const std::uint32_t slots = tier_->pinned_slots();
        const double gib          = static_cast<double>(tier_->record_bytes()) / (1ULL << 30); // an unpinned record: the largest
        char split[160] = "";
        if (tier_->unpinned_records() > 0) {
            std::snprintf(split, sizeof(split), " and %u unpinned records (%.1f GiB, served by the CPU) hold %u more",
                          tier_->unpinned_records(), gib * tier_->unpinned_records(), unpinned);
        }
        char shadow_text[96] = "";
        if (shadows > 0) {
            std::snprintf(shadow_text, sizeof(shadow_text), " and copies of the %u least-used in VRAM", shadows);
        }
        char line[640];
        std::snprintf(line, sizeof(line),
                      "SSD expert tier: %u RAM slots (%.1f GiB) hold %u experts (%s)%s%s%s, read in %.1f s; %u of %u "
                      "experts are read from the artifact when used",
                      slots, static_cast<double>(tier_->pinned_bytes()) / (1ULL << 30), in_ram,
                      fill.source == SeedSource::Saved        ? "the saved ranking, then the last session's use counts"
                      : fill.source == SeedSource::Calibrated ? "the calibrated expert order"
                                                              : "spread across the layers",
                      released > 0 ? (", none of them among the " + std::to_string(released) + " in VRAM").c_str() : "",
                      shadow_text, split, seconds, keys - in_ram - released - unpinned, keys);
        diagnostic(line);
    }

    // The frames' warm start from the saved state or the calibrated order; returns how many experts
    // it loaded.
    std::uint32_t warm_start_frames(const expert_cache::ExpertStateLoad& load, SeedSource source) {
        if (!options_.expert_cache) { return 0; }
        if (source == SeedSource::None) {
            diagnostic(!options_.expert_seed ? "expert cache: --expert-state off, so it starts empty and fills as requests arrive"
                                             : "expert cache: no saved state yet, so it starts empty and fills as requests arrive");
            if (!load.message.empty()) { diagnostic("expert cache starts empty: " + load.message, DiagnosticLevel::Debug); }
            expert_state_saved_ = std::chrono::steady_clock::now();
            return 0;
        }
        const auto start           = std::chrono::steady_clock::now();
        const auto max_keys = static_cast<std::uint32_t>(options_.expert_seed_share * residency_->frames());
        const std::uint32_t loaded = residency_->warm_start(*load.state, kSeedCountCap, max_keys, device_.stream);
        device_.synchronize();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const double loaded_gib =
            static_cast<double>(loaded) * static_cast<double>(residency_->frame_bytes()) / (1ULL << 30);
        char line[256];
        if (source == SeedSource::Saved) {
            std::snprintf(line, sizeof(line),
                          "expert cache: restored the %u most-used experts from the last session (%.1f GiB in %.2f s)",
                          loaded, loaded_gib, seconds);
        } else {
            std::snprintf(line, sizeof(line),
                          "expert cache: no saved state yet, so it starts with the %u highest-ranked experts of "
                          "the calibrated order (%.1f GiB in %.2f s)",
                          loaded, loaded_gib, seconds);
        }
        diagnostic(line);
        std::snprintf(line, sizeof(line), "expert cache warm start: %u of %u frames from %s", loaded,
                      residency_->frames(),
                      source == SeedSource::Saved ? options_.expert_state.string().c_str() : "the calibrated order");
        diagnostic(line, DiagnosticLevel::Debug);
        expert_state_saved_  = std::chrono::steady_clock::now();
        expert_state_routed_ = residency_->stats().routed;
        return loaded;
    }

public:
    // After every synchronization of a round with MoE layers: a call that could not serve an expert
    // (design §19.3.7: an unreadable record, a host service that stopped answering) left undefined
    // outputs, so the round's requests fail and the engine keeps serving. Prefix-cache entries the
    // round may have published are dropped when the lanes are released.
    void check_expert_error() {
        auto* word           = static_cast<volatile std::uint32_t*>(expert_error_.data());
        std::uint32_t value  = *word;
        if (value == 0 && runtime::expert_offload::testing::take_expert_fault()) { value = EIO; } // layer 0
        if (value == 0) { return; }
        *word                    = 0;
        clear_prefix_on_release_ = true;
        // A stopped service or read agent fails every later call too: stop the Engine with its cause.
        if (cpu_service_) {
            if (const std::string why = cpu_service_->failure(); !why.empty()) {
                throw std::runtime_error("Qwen3.8-Flash-Next: the CPU expert service stopped (" + why + ")");
            }
        }
        if (tier_) {
            if (const std::string why = tier_->failure(); !why.empty()) {
                throw std::runtime_error("Qwen3.8-Flash-Next: the SSD expert tier's read agent stopped (" + why + ")");
            }
        }
        const std::uint32_t code = value & 0xFFFFU;
        std::string what         = code == ops::offloaded_moe::kErrorUnservedRecord ? "an SSD-only expert had no path to it"
                                   : code == ops::offloaded_moe::kErrorHostSilent   ? "the CPU expert service stopped answering"
                                   : code == ops::offloaded_moe::kErrorFetchSilent  ? "the SSD expert tier's read agent stopped answering"
                                                                                     : "an expert record could not be read (" +
                                                                                       std::generic_category().message(static_cast<int>(code)) + ")";
        throw runtime::RecoverableExecutionError("Qwen3.8-Flash-Next layer " + std::to_string(value >> 16U) + ": " + what);
    }

public:
    // Writes the saved state (at stop, and from maintain_expert_state). Never throws.
    void save_expert_state() noexcept {
        if (options_.expert_state.empty() || !residency_ || residency_->frames() == 0) { return; }
        try {
            const std::string error =
                expert_cache::save_expert_state(options_.expert_state, options_.expert_state_identity, residency_->saved_state());
            if (!error.empty()) { diagnostic("expert state not saved: " + error, DiagnosticLevel::Warning); }
            expert_state_saved_  = std::chrono::steady_clock::now();
            expert_state_routed_ = residency_->stats().routed;
        } catch (const std::exception& error) {
            diagnostic(std::string("expert state not saved: ") + error.what(), DiagnosticLevel::Warning);
        } catch (...) {}
    }
    // Between requests: saves when rounds ran since the last save and 10 minutes have passed.
    void maintain_expert_state() noexcept {
        if (options_.expert_state.empty() || !residency_ || residency_->stats().routed == expert_state_routed_) { return; }
        if (std::chrono::steady_clock::now() - expert_state_saved_ < kExpertStateInterval) { return; }
        save_expert_state();
    }

    // At the end of a run: reports the expert cache's routed uses and hits per layer (the whole
    // run, not one request) as one Engine diagnostic, beside report_cache's per-request line.
    void report_cache_layers() noexcept {
        if (!residency_) { return; }
        try {
            const std::span<const std::uint64_t> routed = residency_->routed_by_layer();
            const std::span<const std::uint64_t> hits   = residency_->hits_by_layer();
            if (routed.empty()) { return; }
            std::string text = "expert cache per layer (hit% of routed uses):";
            char part[40];
            std::uint64_t total_routed = 0, total_hits = 0;
            for (std::size_t layer = 0; layer < routed.size(); ++layer) {
                total_routed += routed[layer];
                total_hits += hits[layer];
                std::snprintf(part, sizeof(part), " %zu:%.1f%%(%llu)", layer,
                              routed[layer] ? 100.0 * static_cast<double>(hits[layer]) / static_cast<double>(routed[layer]) : 0.0,
                              static_cast<unsigned long long>(routed[layer]));
                text += part;
            }
            std::snprintf(part, sizeof(part), "; run %.1f%% of %llu",
                          total_routed ? 100.0 * static_cast<double>(total_hits) / static_cast<double>(total_routed) : 0.0,
                          static_cast<unsigned long long>(total_routed));
            text += part;
            diagnostic(text, DiagnosticLevel::Debug);
        } catch (...) {}
    }

private:
    // A seeded count is capped, so an expert used heavily in an old session yields to new uses.
    static constexpr std::uint32_t kSeedCountCap = 16;
    // ProgramOptions::expert_seed_share: the share of the frames a warm start fills (the best-ranked
    // experts); the rest stay free for the fill phase's landing (S4), so a workload unlike the saved state
    // still fills quickly.
    static constexpr std::chrono::minutes kExpertStateInterval{10};
    // The pause before the SSD tier's second CPU rate reading (the first follows the tier's fill).
    static constexpr std::chrono::seconds kTierRateSettle{4};
    std::chrono::steady_clock::time_point expert_state_saved_{};
    std::uint64_t expert_state_routed_ = 0;

public:
    // Prefill calls up to this width are CPU-assisted (design §19.3.1 P7); wider chunks route nearly
    // every expert, and their misses stream or stage on the GPU.
    static constexpr int kAssistMaxColumns = 255;

    // At a boundary (no round's kernels in flight; `idle`: no request active): resizes the expert
    // cache to the control law's decision.
    void apply_vram_target(bool idle) {
        if (idle && kv_idle_due() &&
            std::all_of(lanes_.begin(), lanes_.end(), [](const Lane& l) { return l.phase == Phase::Free; })) {
            kv_busy(); // one attempt per idle period: blocks without host copies may still hold the top
            shrink_kv(true);
        } else if (kv_shrink_busy_ && !idle && kv_idle_due()) {
            // TEMPORARY (INFERNIX_TMP_KV_SHRINK_BUSY, candidates §8.3 CF-06): while requests decode, a pool
            // grown past its base with no admission or release for kKvIdleShrink gives back what other
            // conversations' idle blocks hold: their host-backed device copies (unpinned: no lane maps
            // them) are released and the free top returns to the expert frames. Once per idle period.
            kv_busy();
            shrink_kv(true);
        }
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
    // Calls this narrow are CPU-served (design §16.2, prefix P0 step 3).
    static constexpr std::int32_t kServedCallColumns = 8;
    // Prefill calls up to this many columns run in the static workspace; wider ones in the wide arena.
    static constexpr std::int32_t kStaticWorkColumns = 512;

    // A one-row call's lane as staged (for the residency's route predictor, INFERNIX_TMP_ROUTE_PROTECT).
    [[nodiscard]] std::span<const std::int32_t> staged_lane() const noexcept { return {host_lanes_.data(), 1}; }

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
                        .position = position,
                        .frames   = residency_->frames()},
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
            const auto landed   = s.landed - at_admission.landed;
            const auto cpu      = cpu_service_ ? cpu_service_->served_experts() - lane.cpu_served_at_admission : 0;
            std::size_t free_vram = 0, total_vram = 0;
            if (cudaMemGetInfo(&free_vram, &total_vram) != cudaSuccess) { free_vram = 0; }
            // The process's RAM beside the plan (candidates §22 RAM comparison: the engines' totals were only derived).
            const HostMemorySnapshot ram = host_memory_snapshot();
            constexpr double kGiB        = 1024.0 * 1024.0 * 1024.0;
            char text[640];
            std::snprintf(text, sizeof(text),
                          "expert cache: %u frames; request %.1f%% of %llu routed experts hit, %llu promotions, "
                          "%llu landed, %llu misses CPU-served; since start %.1f%%; VRAM free %zu MiB; RAM: process "
                          "%.1f GiB private, %.1f GiB resident, %.1f of %.1f GiB free",
                          residency_->frames(), routed ? 100.0 * static_cast<double>(hits) / static_cast<double>(routed) : 0.0,
                          static_cast<unsigned long long>(routed), static_cast<unsigned long long>(promoted),
                          static_cast<unsigned long long>(landed),
                          static_cast<unsigned long long>(cpu),
                          s.routed ? 100.0 * static_cast<double>(s.hits) / static_cast<double>(s.routed) : 0.0,
                          free_vram >> 20, static_cast<double>(ram.process_private) / kGiB,
                          static_cast<double>(ram.process_resident) / kGiB, static_cast<double>(ram.available_physical) / kGiB,
                          static_cast<double>(ram.total_physical) / kGiB);
            diagnostic(text, DiagnosticLevel::Debug);
            if (tier_) {
                // The tier's counters as of the last round boundary.
                const auto& t = tier_->stats();
                const auto& a = lane.tier_at_admission;
                const auto reads = t.demand_reads - a.demand_reads;
                std::snprintf(text, sizeof(text),
                              "SSD expert tier: request %llu expert reads (%.2f GiB, mean %.2f ms), %llu failed; %llu "
                              "read for the prefill stream (%.2f GiB); %llu "
                              "fetched (%llu from RAM slots, %llu copied from unpinned records); %llu served in place "
                              "from unpinned records; %llu admitted to RAM, %llu demoted from VRAM, %llu promoted from unpinned records",
                              static_cast<unsigned long long>(reads),
                              static_cast<double>(reads) * static_cast<double>(residency_->frame_bytes()) / (1ULL << 30),
                              reads ? 1e-6 * static_cast<double>(t.read_ns - a.read_ns) / static_cast<double>(reads) : 0.0,
                              static_cast<unsigned long long>(t.demand_failures - a.demand_failures),
                              static_cast<unsigned long long>(t.stream_reads - a.stream_reads),
                              static_cast<double>(t.stream_reads - a.stream_reads) *
                                  static_cast<double>(residency_->frame_bytes()) / (1ULL << 30),
                              static_cast<unsigned long long>(t.fetch_records - a.fetch_records),
                              static_cast<unsigned long long>(t.fetch_from_ram - a.fetch_from_ram),
                              static_cast<unsigned long long>(t.unpinned_copies - a.unpinned_copies),
                              static_cast<unsigned long long>(t.unpinned_served - a.unpinned_served),
                              static_cast<unsigned long long>(t.admitted - a.admitted),
                              static_cast<unsigned long long>(s.demotions - at_admission.demotions),
                              static_cast<unsigned long long>(s.unpinned_promotions - at_admission.unpinned_promotions));
                diagnostic(text, DiagnosticLevel::Debug);
                // Where the reads came from: the keys that left both tiers this request, by cause.
                const auto& l  = tier_->controller().losses();
                const auto& la = lane.losses_at_admission;
                std::snprintf(text, sizeof(text),
                              "SSD expert tier: request lost %llu experts to the SSD: %llu VRAM victims unpublished "
                              "(loans, shrinks, evicted loading), %llu without a demotion slot, %llu not worth one; %llu "
                              "RAM-only experts evicted",
                              static_cast<unsigned long long>((l.unpublished - la.unpublished) + (l.no_slot - la.no_slot) +
                                                              (l.unworthy - la.unworthy) +
                                                              (l.residents_evicted - la.residents_evicted)),
                              static_cast<unsigned long long>(l.unpublished - la.unpublished),
                              static_cast<unsigned long long>(l.no_slot - la.no_slot),
                              static_cast<unsigned long long>(l.unworthy - la.unworthy),
                              static_cast<unsigned long long>(l.residents_evicted - la.residents_evicted));
                diagnostic(text, DiagnosticLevel::Debug);
                std::snprintf(text, sizeof(text),
                              "SSD expert tier: request's loans and shrinks evicted %llu experts with a RAM copy and %llu "
                              "without; %llu shadows before the last loan, %zu now, %zu RAM-only; %llu refill reads failed "
                              "(copied over the link)",
                              static_cast<unsigned long long>(s.loan_victims_kept - at_admission.loan_victims_kept),
                              static_cast<unsigned long long>(s.loan_victims_lost - at_admission.loan_victims_lost),
                              static_cast<unsigned long long>(s.shadows_at_loan), tier_->controller().shadow_count(),
                              tier_->controller().resident_count() - tier_->controller().shadow_count(),
                              static_cast<unsigned long long>(s.refill_fallbacks));
                diagnostic(text, DiagnosticLevel::Debug);
            }
        } catch (...) {}
    }

    // The workspace arena of a Program with `lanes` lanes: the forward at its widest call (a prefill
    // chunk or every lane's widest round), sampling, acceptance and the drafter.
    std::size_t workspace_capacity(std::int32_t lanes) const {
        return workspace_bytes(c_, options_, plan_, lanes, plan_.static_columns);
    }

    // Reports at startup what each lane of --max-concurrency takes from the expert cache, in frames
    // (design 19.3.5): its KV extent unless --kv-capacity fixes the pool; its recurrent, convolution
    // and QSA-tail state; its verification and drafter records (DFlash2: its context ring and round
    // feature columns); its logit columns and penalty counts; and its share of the workspace. Every
    // other allocation is independent of the lane count, so one more lane costs these frames.
    void report_lane_cost(std::int32_t lanes, std::uint64_t record_stride) noexcept {
        try {
            const auto n         = static_cast<std::size_t>(lanes);
            const bool fixed_kv  = options_.kv_capacity_tokens != 0;
            const std::size_t kv = fixed_kv ? 0 : kv_span_.bytes / n;
            // The feature buffer's call region (one window) is shared; its round region is per lane.
            const std::size_t dflash_round =
                plan_.dflash2 ? 2ULL * plan_.dflash2_config->feature_width(c_.hidden_size) *
                                    static_cast<std::size_t>(plan_.dflash_round_columns)
                              : 0;
            const std::size_t state =
                (state_backing_.bytes + ple_backing_.bytes + tails_backing_.bytes + logits32_.bytes + logits16_.bytes +
                 token_counts_.bytes + records_backing_.bytes + ple_records_.bytes + qsa_records_.bytes +
                 mtp_residuals_.bytes + mtp_saved_.bytes + mtp_chain_.bytes + mtp_records_.bytes +
                 dflash_ring_backing_.bytes + dflash_round) /
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
            diagnostic(text, DiagnosticLevel::Debug);
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
            std::string line = text;
            if (const auto& g = lane.gate; g.rounds > 0) {
                std::uint64_t waits[2] = {};
                CUDA_CHECK(cudaMemcpyAsync(waits, static_cast<const std::byte*>(gate_stats_.p) + kGateStatsBytes * lane_index(lane),
                                           sizeof(waits), cudaMemcpyDeviceToHost, device_.stream));
                device_.synchronize();
                std::snprintf(text, sizeof(text),
                              "; %llu gated rounds: %llu NVMe reads (%.1f ms) behind the gate, the GPU waited in %llu "
                              "(mean %.1f us)",
                              static_cast<unsigned long long>(g.rounds), static_cast<unsigned long long>(g.reads),
                              static_cast<double>(g.read_ns) * 1e-6, static_cast<unsigned long long>(waits[1]),
                              waits[1] ? static_cast<double>(waits[0]) * 1e-3 / static_cast<double>(waits[1]) : 0.0);
                line += text;
            }
            diagnostic(line, DiagnosticLevel::Debug);
        } catch (...) {}
    }

    std::uint32_t* gate_ready() const { return static_cast<std::uint32_t*>(gate_host_.data()); }

    // The n-gram gate of a decode or verification graph: the pinned rows behind the round's word.
    execution::NgramRowGate row_gate() const {
        auto* io_device = static_cast<std::byte*>(io_device_.p);
        return {.pinned_rows = host_io() + io_layout_.ngram,
                .ready       = gate_ready(),
                .expected    = reinterpret_cast<const std::uint32_t*>(io_device + io_layout_.gate),
                .wait_stats  = static_cast<std::uint64_t*>(gate_stats_.p),
                .wait_row    = reinterpret_cast<const std::int32_t*>(io_device + io_layout_.slots)};
    }

    std::uint32_t lane_index(const Lane& lane) const noexcept {
        return static_cast<std::uint32_t>(&lane - lanes_.data());
    }

    // Publishes a gated round's word if the round ends before its rows are read: the rows are
    // zeroed (the round's results are discarded with the error), so the enqueued gate passes.
    struct GateRelease {
        ProgramImpl* owner = nullptr;
        std::uint32_t word = 0;
        std::size_t bytes  = 0;
        GateRelease(ProgramImpl* o, std::uint32_t w, std::size_t b) : owner(o), word(w), bytes(b) {}
        GateRelease(const GateRelease&)            = delete;
        GateRelease& operator=(const GateRelease&) = delete;
        ~GateRelease() {
            if (owner == nullptr) { return; }
            std::memset(owner->host_io() + owner->io_layout_.ngram, 0, bytes);
            publish_pinned_word(owner->gate_ready(), word);
        }
    };

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

    // Times a few hundred random 4 KiB reads of a volume that is read on use and says so: a drive
    // that serves them slowly (a DRAM-less SSD at 5,000 reads/s against 17,000-80,000 on the owner's
    // drives and 230,000 on a Samsung 9100 PRO) makes a long prompt spend minutes reading n-gram rows.
    // It never changes placement; the table stays where the ledger put it.
    void probe_ngram_volume() const noexcept {
        constexpr double kSlowReadsPerSecond = 15000.0;
        if (options_.ngram_in_ram) {
            try {
                if (const std::string why = volume_.resident_lock_failure(); !why.empty()) {
                    diagnostic("the n-gram table is in RAM but Windows would not lock it there (" + why +
                                   "); it still serves rows, but under memory pressure Windows may page part of it out "
                                   "and long prompts slow down",
                               DiagnosticLevel::Warning);
                }
            } catch (...) {}
            return;
        }
        try {
            const NgramVolume::ReadProbe probe = volume_.probe_reads();
            if (probe.reads == 0) { return; }
            const double rate = probe.reads_per_second();
            char text[320];
            if (rate >= kSlowReadsPerSecond) {
                std::snprintf(text, sizeof text, "n-gram volume read probe: %.0f reads/s (%.0f us per 4 KiB read, %zu reads)",
                              rate, 1e6 / rate, probe.reads);
                diagnostic(text);
            } else {
                std::snprintf(text, sizeof text,
                              "n-gram volume reads are slow: %.0f reads/s (%zu reads; 17,000 and more is healthy), so a long "
                              "prompt can spend minutes reading n-gram rows; keep the table in RAM with --ngram-residency ram "
                              "if the experts fit beside it, or move the volume to a faster drive",
                              rate, probe.reads);
                diagnostic(text, DiagnosticLevel::Warning);
            }
        } catch (...) {}
    }

    // The io bytes a decode or verification round of `columns` columns reads: everything before
    // the n-gram rows (ids, positions, slots, table rows, logit columns) and its columns' rows.
    // Prefill and forced-token calls copy the whole layout (their MTP cells follow the rows).
    std::size_t io_prefix(std::int32_t columns) const {
        return io_layout_.ngram + static_cast<std::size_t>(columns) * hash_.heads() * ngram_table().row_bytes;
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

    static double vision_seconds_of(const Lane& lane) {
        if (lane.vision_timer) {
            if (const auto ms = lane.vision_timer->try_elapsed_ms()) { return *ms * 1e-3; }
        }
        return lane.vision_seconds;
    }

    GenerationTimings timings(const Lane& lane) const {
        GenerationTimings out;
        out.prefill_seconds      = static_cast<double>(lane.prefill_ns) * 1e-9;
        out.decode_seconds       = static_cast<double>(lane.decode_ns) * 1e-9;
        out.decode_share_seconds = static_cast<double>(lane.decode_share_ns) * 1e-9;
        out.vision_seconds       = vision_seconds_of(lane);
        return out;
    }

    void release(std::uint32_t index) noexcept {
        Lane& lane = lanes_[index];
        if (continuing_calls_ != 0) {
            // TEMPORARY (INFERNIX_TMP_CONTINUING_PROMPT, CF-01): the positive control of the arm.
            try {
                diagnostic("continuing prompts: " + std::to_string(continuing_calls_) +
                           " prefill calls left out of the expert cache's scores");
            } catch (...) {}
        }
        if (replay_clock_) {
            // TEMPORARY (INFERNIX_TMP_REPLAY_BENCH): the decode rounds' device time since startup, the
            // CPU waits inside it, and the residency's state (frozen once its warm rounds passed).
            try {
                replay_clock_->drain();
                unsigned long long wait_ns = 0;
                CUDA_CHECK(cudaMemcpy(&wait_ns, replay_wait_.p, sizeof(wait_ns), cudaMemcpyDeviceToHost));
                const double round_ms = replay_clock_->milliseconds(), wait_ms = static_cast<double>(wait_ns) * 1e-6;
                const auto rounds     = replay_clock_->rounds();
                diagnostic("replay bench: " + std::to_string(rounds) + " rounds, " + std::to_string(round_ms) +
                           " ms device time, " + std::to_string(wait_ms) + " ms CPU waits, " +
                           std::to_string(round_ms - wait_ms) + " ms less waits (" +
                           std::to_string(rounds != 0 ? (round_ms - wait_ms) / static_cast<double>(rounds) : 0.0) +
                           " ms a round); residency " + (residency_ && residency_->frozen() ? "frozen" : "not frozen"));
            } catch (...) {}
        }
        if (walk_.active && walk_.lane == index) { walk_abandon(); }
        if (stream_lease_.valid() && !prefilling_besides(index)) {
            try { return_stream_lease(); } catch (...) {}
        }
        if (!prefilling_besides(index)) {
            try { return_wide_work(); } catch (...) {}
        }
        prefix_release(lane);
        lane.pages.clear();
        lane.reservation.reset();
        lane.row = KVExecutionRowLease{};
        lane.history.clear();
        lane.history.shrink_to_fit();
        lane.rope = {};
        if (lane.vision) { vision_release(lane); }
        lane.proposer.reset();
        lane.readout.clear();
        lane.constraint.reset();
        lane.prompt.reset();
        lane.constraint_round.clear();
        lane.constraint_trace.clear();
        lane.phase        = Phase::Free;
        lane.state_tokens = 0;
        lane.prefill_end  = 0;
        lane.replay       = false;
        lane.extent       = 0;
        ++revision_;
        kv_busy();
        try { shrink_kv(); } catch (...) {}
        // A cache a failure may have left wrong state in (prefix_withdraw could not tell its entries
        // apart) is emptied once no lane maps it.
        if (clear_prefix_on_release_ && prefix_ && !in_transaction() &&
            std::all_of(lanes_.begin(), lanes_.begin() + static_cast<std::ptrdiff_t>(options_.max_concurrency),
                        [](const Lane& other) { return other.phase == Phase::Free; })) {
            try { device_.synchronize(); } catch (...) {}
            prefix_->clear();
            prefix_withdrawal_       = {};
            clear_prefix_on_release_ = false;
        }
    }

    // Whether the elastic pool could grow to make `need` pages available (the quote's check).
    bool kv_growable(std::uint32_t need) const noexcept {
        return static_cast<std::uint64_t>(pool_->available_pages()) + kv_grow_pages() >= need;
    }

    // The pages one growth could add at most: grow_kv takes up to every frame, and none while frames
    // are lent, unless only leases hybrid_reclaim can return hold them (an estimate; grow_kv
    // decides).
    std::uint32_t kv_grow_pages() const noexcept {
        if (!kv_vmm_ || !residency_ || !residency_->elastic() || residency_->frames() == 0 ||
            (residency_->stats().lent_frames != 0 && !leases_returnable())) {
            return 0;
        }
        const std::uint32_t room = pool_->capacity_pages() - pool_->backed_pages();
        const std::uint64_t page_bytes =
            std::max<std::uint64_t>(1, kv_span_.bytes / std::max<std::uint32_t>(1, pool_->capacity_pages()));
        const std::uint64_t frame_room = static_cast<std::uint64_t>(residency_->frames()) * residency_->frame_bytes();
        const std::uint64_t slack      = residency_->pools() * ExpertResidency::kChunkBytes + 2 * kKvChunkBytes * plan_.pool.planes.size();
        const std::uint64_t usable     = frame_room > slack ? frame_room - slack : 0;
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(room, usable / page_bytes));
    }

    // Zeroes a slot's recurrent state, PLE convolution history, QSA tails (and exact-window tags) and
    // penalty counts.
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
            if (window_geometry_.present()) { // the window's tags: no earlier history's row is read
                const std::size_t tags = window_geometry_.lane_bytes(4);
                CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(window_backing_.p) +
                                               window_geometry_.plane_offset(l, 4) + slot * tags,
                                           0, tags, s));
            }
        }
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::int32_t*>(token_counts_.p) + slot * static_cast<std::size_t>(token_domain_),
                                   0, 4ULL * token_domain_, s));
    }

    std::byte* host_io() const { return static_cast<std::byte*>(io_host_.data()); }

    // `count` I32 words of the pinned io at byte offset `offset`.
    std::span<std::int32_t> io_words(std::size_t offset, std::size_t count) const {
        return {reinterpret_cast<std::int32_t*>(host_io() + offset), count};
    }

    // RoPE positions of `count` consecutive tokens of `lane` from `first`, as columns column ..
    // column + count - 1 of a call of `columns` columns over `batch` sequences, and the pooled-block
    // start of the lane's sequence b.
    void stage_call_rope(const Lane& lane, std::uint32_t first, std::int32_t count, std::int32_t columns,
                         std::int32_t column, std::int32_t batch, std::int32_t b) const {
        stage_rope(lane.rope, first, count, io_words(io_layout_.rope, 3ULL * columns), columns, column);
        stage_rope_value(block_start_rope(lane.rope, first, static_cast<std::uint32_t>(r_)),
                         io_words(io_layout_.block_rope, 3ULL * batch), batch, b);
    }

    // The n-gram rows of `count` positions starting at `begin`, from the tokens before them; the
    // traffic counts toward `lane`'s request.
    void stage_ngram(Lane& lane, const std::vector<std::int32_t>& history, std::int32_t begin, std::int32_t count,
                     std::size_t column) {
        ngram_window(history, begin, count);
        read_ngram(lane, count, column);
    }

    // window_ = the tokens of positions [begin - (n - 1), begin + count), EOS before the sequence.
    void ngram_window(const std::vector<std::int32_t>& history, std::int32_t begin, std::int32_t count) {
        const std::int32_t context = static_cast<std::int32_t>(c_.ple.ngram.ngram_size) - 1;
        window_.clear();
        for (std::int32_t p = begin - context; p < begin + count; ++p) {
            window_.push_back(p < 0 ? static_cast<std::int32_t>(c_.eos_token_id) : history[static_cast<std::size_t>(p)]);
        }
    }

    // Verification row b's token at position q: its history up to the anchor (position p), then its
    // drafts, padded past their extent with the last draft (the anchor without drafts).
    std::int32_t verify_token(const Lane& lane, std::size_t b, std::int32_t p, std::int32_t q) const {
        if (q < 0) { return static_cast<std::int32_t>(c_.eos_token_id); }
        if (q <= p) { return lane.history[static_cast<std::size_t>(q)]; }
        const auto& d = drafts_[b];
        const auto j  = static_cast<std::size_t>(q - p - 1);
        return j < d.size() ? d[j] : (d.empty() ? lane.history.back() : d.back());
    }

    // The n-gram rows of verification row b's columns [from, to) (W columns from position p).
    void stage_verify_ngram(Lane& lane, std::size_t b, std::int32_t p, std::int32_t W, std::int32_t from, std::int32_t to) {
        if (from >= to) { return; }
        const std::int32_t context = static_cast<std::int32_t>(c_.ple.ngram.ngram_size) - 1;
        window_.clear();
        for (std::int32_t q = p + from - context; q < p + to; ++q) { window_.push_back(verify_token(lane, b, p, q)); }
        read_ngram(lane, to - from, b * static_cast<std::size_t>(W) + static_cast<std::size_t>(from));
    }

    // Reads the rows of the `count` positions whose tokens (with their context) are in window_ into
    // io columns from `column`.
    void read_ngram(Lane& lane, std::int32_t count, std::size_t column) {
        read_ngram_to(lane, count, host_io() + io_layout_.ngram + column * hash_.heads() * ngram_table().row_bytes);
    }

    // As read_ngram, into `out` (count * heads rows).
    void read_ngram_to(Lane& lane, std::int32_t count, std::byte* out) {
        const nvtx::ScopedRange rows_range(nvtx::Name::PleRows, nvtx::Category::Control, static_cast<std::uint64_t>(count));
        const std::size_t heads = hash_.heads();
        row_ids_.resize(static_cast<std::size_t>(count) * heads);
        hash_.row_ids(window_, static_cast<std::size_t>(count), row_ids_.data());
        const std::size_t row_bytes = ngram_table().row_bytes;
        const NgramVolume::Counters before = volume_.counters();
        volume_.read_rows(row_ids_, std::span<std::byte>(out, static_cast<std::size_t>(count) * heads * row_bytes));
        const NgramVolume::Counters& after = volume_.counters();
        lane.ngram.rows += after.rows - before.rows;
        lane.ngram.hits += after.hits - before.hits;
        lane.ngram.reads += after.reads - before.reads;
        lane.ngram.read_ns += after.read_ns - before.read_ns;
    }

    // `rows`: also read the n-gram rows (a walk chunk whose rows were prefetched skips them).
    void stage_sequence(std::uint32_t lane, std::int32_t begin, std::int32_t width,
                        const std::vector<std::int32_t>& history, bool rows = true) {
        auto* ids       = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.ids);
        auto* positions = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.positions);
        auto* columns   = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.columns);
        for (std::int32_t i = 0; i < width; ++i) {
            ids[i]       = history[static_cast<std::size_t>(begin + i)];
            positions[i] = begin + i;
        }
        columns[0] = width - 1;
        stage_call_rope(lanes_[lane], static_cast<std::uint32_t>(begin), width, width, 0, 1, 0);
        reinterpret_cast<std::int32_t*>(host_io() + io_layout_.slots)[0] = static_cast<std::int32_t>(lane);
        reinterpret_cast<std::int32_t*>(host_io() + io_layout_.rows)[0]  = static_cast<std::int32_t>(lane);
        if (rows) { stage_ngram(lanes_[lane], history, begin, width, 0); }
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
            stage_call_rope(lane, static_cast<std::uint32_t>(positions[b]), 1, static_cast<std::int32_t>(lanes.size()),
                            static_cast<std::int32_t>(b), static_cast<std::int32_t>(lanes.size()), static_cast<std::int32_t>(b));
        }
    }

    // The n-gram rows of a plain decode round (column b: lane b's input token).
    void stage_decode_rows(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions) {
        for (std::size_t b = 0; b < lanes.size(); ++b) {
            Lane& lane = lanes_[lanes[b]];
            stage_ngram(lane, lane.history, positions[b], 1, b);
        }
    }

    // Whether every n-gram row of a plain decode round is in the host row cache.
    bool decode_rows_cached(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions) {
        const std::int32_t context = static_cast<std::int32_t>(c_.ple.ngram.ngram_size) - 1;
        for (std::size_t b = 0; b < lanes.size(); ++b) {
            const Lane& lane = lanes_[lanes[b]];
            window_.clear();
            for (std::int32_t p = positions[b] - context; p <= positions[b]; ++p) {
                window_.push_back(p < 0 ? static_cast<std::int32_t>(c_.eos_token_id) : lane.history[static_cast<std::size_t>(p)]);
            }
            row_ids_.resize(hash_.heads());
            hash_.row_ids(window_, 1, row_ids_.data());
            if (!volume_.cached(row_ids_)) { return false; }
        }
        return true;
    }

    std::uint32_t next_gate_word() {
        if (++gate_sequence_ == 0) { ++gate_sequence_; }
        const std::uint32_t word = gate_sequence_;
        *reinterpret_cast<std::uint32_t*>(host_io() + io_layout_.gate) = word;
        return word;
    }

    // After a gated launch: submits the queued work, reads the rows (`read`), releases the gate and
    // counts the reads toward `lane` (the round's first row).
    void read_behind_gate(Lane& lane, std::uint32_t word, const std::function<void()>& read) {
        device_.flush();
        const NgramVolume::Counters before = volume_.counters();
        read();
        const NgramVolume::Counters& after = volume_.counters();
        publish_pinned_word(gate_ready(), word);
        ++lane.gate.rounds;
        lane.gate.reads += after.reads - before.reads;
        lane.gate.read_ns += after.read_ns - before.read_ns;
        if (after.read_ns - before.read_ns > kSlowGatedReadNs) {
            diagnostic("n-gram rows behind the gate took " + std::to_string((after.read_ns - before.read_ns) / 1000000) +
                           " ms (the GPU waited at layer 1)",
                       DiagnosticLevel::Warning);
        }
    }

    // Runs one eager Forward call over the staged inputs (prefill chunks, forced tokens); logits of
    // `logit_columns` columns land in logits32_.
    void run(std::int32_t batch, std::int32_t width, std::int32_t logit_columns,
             const execution::MtpChunk* chunk = nullptr, const execution::VisionInput* vision = nullptr,
             bool streamed = false) {
        const cudaStream_t s = device_.stream;
        // Read by the SMs, so it never queues on the copy engine behind expert promotions (W7).
        upload_pinned(io_device_.p, io_host_.data(), io_layout_.bytes, s);
        if (cpu_service_) { cpu_service_->wake(); } // its first request comes within this call
        residency_->before_round(s);
        // A call the CPU can share (design §19.3.12) gates the stream on each layer's routing.
        split_call_ = streamed && split_policy_ && batch * width <= split_columns();
        if (streamed) { expert_stream_->begin(ring_span(), residency_->host_table(), s, split_call_); }
        const WideWork wide(*this, batch * width);
        try {
            forward_call(batch, width, logit_columns, nullptr, chunk, vision, nullptr, streamed);
            if (dflash2_) { dflash2_call_context(batch, width); }
        } catch (...) {
            // A failed call (an unreadable record, say) still returns the stream's slots: left
            // active, the next streamed chunk could never begin and the release would refuse.
            if (streamed) { expert_stream_->end(); }
            split_call_ = false;
            throw;
        }
        if (streamed) { expert_stream_->end(); }
        split_call_ = false;
        residency_->enqueue_route_download(s, batch * width);
    }

    // ---- prefill expert streaming (design §19.3.8 F2, F3) ----
    // Chunks of at least kStreamMinColumns route nearly every expert of a layer, so all of a layer's
    // non-resident experts are copied ahead; narrower chunks keep the MoE's staging. The ring holds
    // two halves of as many records as the layer with the most non-resident experts has when it is
    // lent (+ one frame for the slot tables), so no streamed layer leaves misses to staging; it is
    // lent by the expert cache for the prompt and returned when no lane is prefilling. Halves of
    // 384 records (a cold cache misses all 512 of a layer) were 9 % slower at pp4096 (design
    // §19.3.8, F2 ring size).
    static constexpr std::int32_t kStreamMinColumns = 256;
    static constexpr std::uint32_t kStreamHalfMin   = 16;

    // The call planner's costs: the fitted call costs, and the walk's spans as walk_span_end forms them.
    [[nodiscard]] prefix::CallCost call_cost() const {
        return prefix::CallCost{.model            = options_.prefix_cost,
                                .span_seconds     = options_.prefix_span_seconds,
                                .walk_min_columns = static_cast<std::uint32_t>(kStreamMinColumns),
                                .walk_max_tokens  = kWalkMaxTokens};
    }

    // Whether this chunk streams; lends the ring first when none is held. With `walk_bytes`, the
    // lease also holds a layer walk's area after the ring (stream_walk_bytes_; zero when the
    // lendable frames do not cover it): the experts its frames evict widen each half by their
    // per-layer share.
    bool stream_chunk(std::int32_t width, std::size_t walk_bytes = 0) {
        if (!expert_stream_ || width < kStreamMinColumns || residency_->frames() == 0) { return false; }
        if (!stream_lease_.valid()) {
            const std::int32_t* table = residency_->host_table();
            const std::uint32_t E     = c_.moe.experts;
            std::uint32_t most        = 0;
            for (std::uint32_t l = 0; l < c_.num_hidden_layers; ++l) {
                std::uint32_t absent = 0;
                for (std::uint32_t e = 0; e < E; ++e) { absent += table[static_cast<std::size_t>(l) * E + e] < 0 ? 1U : 0U; }
                most = std::max(most, absent);
            }
            // Three leases, each one contiguous run of one pool's frames (the pool with the most room
            // when it is taken): the wide arena whenever the plan has one (a prompt with a streamed
            // call runs wide calls), a walk's area when it fits, then the ring, whose slots hold the
            // largest record (plan_.record_stride) with one more slot's room for the slot tables.
            // The frames a lease takes evict experts of its pool's layers only: each layer's half
            // widens by the most any lease takes per layer of its pool.
            const std::uint64_t slot     = plan_.record_stride;
            const cudaStream_t writers[] = {expert_stream_->stream()};
            const auto per_layer         = [&](const ExpertResidency::FrameLease& lease) {
                const std::uint32_t layers = std::max(1U, residency_->lease_layers(lease));
                return (lease.count + layers - 1) / layers;
            };
            ExpertResidency::FrameLease wide, walk;
            if (const std::size_t wide_bytes = pending_wide_bytes(width); wide_bytes > 0) {
                if (wide_bytes + residency_->lend_pitch() > residency_->lendable_bytes()) { return false; }
                wide = residency_->lend(wide_bytes, device_.stream, writers);
            }
            if (walk_bytes > 0 && walk_bytes + 2 * (kStreamHalfMin + 1) * slot <= residency_->lendable_bytes()) {
                walk = residency_->lend(walk_bytes, device_.stream, writers);
            }
            const std::uint32_t widened = std::min(E, most + std::max(per_layer(wide), per_layer(walk)));
            // The ring as wide as the room left allows (at most 2 * half + 1 slots).
            const auto room_slots = static_cast<std::uint32_t>(residency_->lendable_bytes() / slot);
            const std::uint32_t half = std::min(widened, room_slots > 1 ? (room_slots - 1) / 2 : 0U);
            if (half < kStreamHalfMin) {
                if (walk.valid()) { residency_->give_back(walk, device_.stream); }
                if (wide.valid()) { residency_->give_back(wide, device_.stream); }
                return false;
            }
            stream_lease_      = residency_->lend((2ULL * half + 1) * slot, device_.stream, writers);
            stream_wide_lease_ = wide;
            stream_walk_lease_ = walk;
            stream_ring_bytes_ = stream_lease_.memory.bytes;
            stream_wide_bytes_ = wide.memory.bytes;
            stream_walk_bytes_ = walk.memory.bytes;
            stream_at_lease_   = expert_stream_->stats();
            split_at_lease_    = forward_->split_stats();
            prefetched_at_lease_ = walk_prefetched_chunks_;
        }
        return true;
    }

    // The wide arena a call of `columns` needs (candidates BX-01), and the one the prefilling lanes' calls
    // need from this call (`width` columns) on: the widest of it and every prefilling lane's remaining calls,
    // so a short prompt no longer evicts the experts of a whole chunk's arena (the arena holds intermediates
    // only, so no bit changes). Against a whole chunk's arena (Dense8, MTP 3, warm cache, 2026-10-24): a
    // 1,024-token prompt then 128 tokens prefilled 1.038x and decoded 1.040x as fast (6 of 6 pairs); cold
    // 512- and 4,096-token prompts unchanged.
    [[nodiscard]] std::size_t wide_bytes_for(std::int32_t columns) const {
        if (wide_work_ == 0 || columns >= chunk_) { return wide_work_; }
        return std::min(wide_work_, workspace_bytes(c_, options_, plan_, plan_.lanes, columns));
    }
    [[nodiscard]] std::size_t pending_wide_bytes(std::int32_t width) const {
        if (wide_work_ == 0) { return wide_work_; }
        std::uint32_t widest = static_cast<std::uint32_t>(std::max(width, 0));
        for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
            const Lane& lane = lanes_[i];
            if (lane.phase != Phase::Prefill) { continue; }
            // A call in flight has advanced next_call; its end is where the next call starts.
            std::uint32_t from = lane.next_call > 0 ? lane.calls[lane.next_call - 1] : lane.state_tokens;
            for (std::size_t k = lane.next_call; k < lane.calls.size(); ++k) {
                widest = std::max(widest, lane.calls[k] - from);
                from   = lane.calls[k];
            }
        }
        return widest > static_cast<std::uint32_t>(plan_.static_columns) ? wide_bytes_for(static_cast<std::int32_t>(widest))
                                                                         : 0;
    }

    // The ring (the stream lease) and the walk's area (its own lease).
    DeviceSpan ring_span() const noexcept { return DeviceSpan{stream_lease_.memory.data, stream_ring_bytes_}; }
    std::byte* walk_area() const noexcept { return static_cast<std::byte*>(stream_walk_lease_.memory.data); }

    // ---- the wide workspace (design §19.3.7, VRAM item 1) ----
    // A call wider than the static arena's columns runs in the wide arena: the stream lease's part
    // when it has one, else frames lent for it alone, else (no lendable frames: the VRAM monitor
    // shrank the cache, or there are no frames) a temporary device allocation, with a warning. The
    // Forward keeps its arena reference; the arenas' contents are swapped for the call.
    class WideWork {
    public:
        WideWork(ProgramImpl& program, std::int32_t columns) : program_(program) {
            if (columns > program.plan_.static_columns && program.wide_work_ > 0) {
                program.enter_wide(columns);
                active_ = true;
            }
        }
        ~WideWork() {
            if (active_) { program_.leave_wide(); }
        }
        WideWork(const WideWork&)            = delete;
        WideWork& operator=(const WideWork&) = delete;

    private:
        ProgramImpl& program_;
        bool active_ = false;
    };

    // The expert cache's VRAM demand: what the fixed allocations, the graph executables' reserve and
    // the display headroom leave becomes frames of `frame_bytes`.
    VramDemand expert_frame_demand(std::uint64_t frame_bytes) const {
        VramDemand demand;
        demand.headroom    = options_.vram_headroom;
        demand.elastic     = residency_->elastic() && options_.vram_monitor;
        demand.graphs      = plan_.bytes.graph_bound;
        demand.frame_bytes = frame_bytes;
        demand.max_frames  = residency_->frame_limit();
        return demand;
    }

    // A prefill chunk whose wide workspace the expert cache could never lend (more than the three
    // quarters of its frames serving may give up) would allocate it for every long prompt, which
    // fails on a full card after admission (a 65,536-token chunk on the SSD tier wanted 27 GiB).
    // Prefill then runs at the widest halving of the chunk that can be lent (or the static arena's
    // width): its exact route is chunk-invariant, so only speed changes. The buffers planned for the
    // wider chunk stay as they are. `lendable` counts frames of `stride` bytes the cache can lend (a
    // cache of several frame pools lends from one: ExpertResidency::lend_capacity_bytes).
    void fit_wide_work_to_cache(std::uint64_t lendable, std::uint64_t stride) {
        if (wide_work_ == 0 || lendable == 0 || stride == 0) { return; }
        const auto frames_for        = [&](std::size_t bytes) { return (bytes + stride - 1) / stride; };
        if (frames_for(wide_work_) < lendable) { return; }
        const std::int32_t requested = chunk_;
        const std::size_t needed     = wide_work_;
        std::int32_t chunk           = chunk_ / 2;
        while (chunk > plan_.static_columns &&
               frames_for(workspace_bytes(c_, options_, plan_, plan_.lanes, chunk)) >= lendable) {
            chunk /= 2;
        }
        chunk_     = std::max(chunk, plan_.static_columns);
        wide_work_ = chunk_ > plan_.static_columns ? workspace_bytes(c_, options_, plan_, plan_.lanes, chunk_) : 0;
        diagnostic("prefill chunk " + std::to_string(requested) + " needs " + std::to_string(needed >> 20) +
                       " MiB of workspace, more than the expert cache can lend (" +
                       std::to_string((lendable * stride) >> 20) + " MiB); prefilling at " + std::to_string(chunk_) +
                       " tokens (the output is the same)",
                   DiagnosticLevel::Warning);
    }

    void enter_wide(std::int32_t columns) {
        DeviceSpan span{};
        // This call's need, and a lease for the prefilling lanes' remaining calls (one too small for this
        // call is lent again).
        const std::size_t need = wide_bytes_for(columns);
        if (stream_lease_.valid() && stream_wide_bytes_ >= need) {
            span = {stream_wide_lease_.memory.data, need};
        } else {
            if (work_lease_.valid() && work_lease_.memory.bytes < need) {
                residency_->give_back(work_lease_, device_.stream);
                work_lease_ = {};
            }
            if (!work_lease_.valid() && wide_fallback_.p == nullptr) {
                const std::size_t lend = std::max(need, pending_wide_bytes(columns));
                if (residency_->frames() > 0 && lend + residency_->lend_pitch() <= residency_->lendable_bytes()) {
                    const cudaStream_t writers[] = {device_.stream};
                    work_lease_ = residency_->lend(lend, device_.stream, writers);
                } else {
                    diagnostic("the prefill workspace could not be lent from the expert cache; " +
                                   std::to_string(wide_work_ >> 20) + " MiB are allocated for the prompt",
                               DiagnosticLevel::Warning);
                    // The cache shrank under VRAM pressure: a failed allocation fails this prompt's
                    // requests, not the Engine.
                    try {
                        wide_fallback_ = DeviceBuffer(wide_work_);
                    } catch (const std::exception& error) {
                        throw runtime::RecoverableExecutionError(
                            "the prefill workspace (" + std::to_string(wide_work_ >> 20) +
                            " MiB) could neither be lent from the expert cache nor allocated (" + error.what() +
                            "); a smaller --prefill-chunk needs less");
                    }
                }
            }
            span = work_lease_.valid() ? DeviceSpan{work_lease_.memory.data, need}
                                       : DeviceSpan{wide_fallback_.p, wide_work_};
        }
        wide_arena_ = std::make_unique<WorkspaceArena>(span);
        std::swap(*work_, *wide_arena_);
        wide_active_ = true;
    }

    void leave_wide() noexcept {
        std::swap(*work_, *wide_arena_);
        wide_active_ = false;
    }
    bool wide_active_ = false; // *work_ is the wide arena (enter_wide .. leave_wide)

    // After a prompt's last wide call is enqueued: the wide arena's own lease or allocation goes back
    // (the stream lease's part goes back with it).
    void return_wide_work() {
        if (work_lease_.valid()) { residency_->give_back(work_lease_, device_.stream); }
        work_lease_ = {};
        if (wide_fallback_.p != nullptr) {
            device_.synchronize();
            wide_fallback_ = DeviceBuffer{};
        }
    }

    // Returns the ring (after the last streamed chunk is enqueued: compute waited for every copy).
    void return_stream_lease() {
        if (!stream_lease_.valid() || walk_.active) { return; }
        const auto stream_ring_frames = stream_lease_.count;
        const auto stream_wide_frames = stream_wide_lease_.count;
        const auto stream_walk_frames = stream_walk_lease_.count;
        const auto frames             = stream_ring_frames + stream_wide_frames + stream_walk_frames;
        residency_->give_back(stream_lease_, device_.stream);
        residency_->give_back(stream_wide_lease_, device_.stream);
        residency_->give_back(stream_walk_lease_, device_.stream);
        stream_lease_         = {};
        stream_ring_bytes_    = 0;
        stream_wide_bytes_    = 0;
        stream_walk_bytes_    = 0;
        const auto& now       = expert_stream_->stats();
        const auto streamed   = now.streamed - stream_at_lease_.streamed;
        const auto copies     = now.copies - stream_at_lease_.copies;
        const auto split      = forward_->split_stats();
        char line[480];
        std::snprintf(line, sizeof(line),
                      "prefill expert stream: %llu experts (%.2f GiB, %llu read from the SSD) in %llu copies through "
                      "%u lent frames (wide arena %u, ring %u, walk %u); CPU split: %llu experts on the CPU, %llu "
                      "gated copies; n-gram rows of %llu walk chunks read during the previous span",
                      static_cast<unsigned long long>(streamed),
                      static_cast<double>(now.bytes - stream_at_lease_.bytes) / (1ULL << 30),
                      static_cast<unsigned long long>(now.ssd - stream_at_lease_.ssd),
                      static_cast<unsigned long long>(copies), frames, stream_wide_frames, stream_ring_frames,
                      stream_walk_frames,
                      static_cast<unsigned long long>(split.cpu_experts - split_at_lease_.cpu_experts),
                      static_cast<unsigned long long>(split.streamed_experts - split_at_lease_.streamed_experts),
                      static_cast<unsigned long long>(walk_prefetched_chunks_ - prefetched_at_lease_));
        diagnostic(line, DiagnosticLevel::Debug);
        // The copy-ahead (ExpertStream::plan_ahead): how well the guess served the gated layers.
        if (const auto ahead = now.ahead - stream_at_lease_.ahead; ahead > 0) {
            const auto heavy = now.heavy - stream_at_lease_.heavy;
            const auto used  = now.ahead_used - stream_at_lease_.ahead_used;
            const auto late  = now.late - stream_at_lease_.late;
            std::snprintf(line, sizeof(line),
                          "prefill copy-ahead: %llu experts ahead, %llu of them routed (precision %.3f); %llu of %llu heavy "
                          "experts ahead (recall %.3f); %llu copied after their layer's routing",
                          static_cast<unsigned long long>(ahead), static_cast<unsigned long long>(used),
                          static_cast<double>(used) / static_cast<double>(ahead),
                          static_cast<unsigned long long>(now.heavy_ahead - stream_at_lease_.heavy_ahead),
                          static_cast<unsigned long long>(heavy),
                          heavy > 0 ? static_cast<double>(now.heavy_ahead - stream_at_lease_.heavy_ahead) / static_cast<double>(heavy) : 0.0,
                          static_cast<unsigned long long>(late));
            diagnostic(line, DiagnosticLevel::Debug);
        }
    }

    // Whether a lane other than `index` is prefilling.
    [[nodiscard]] bool prefilling_besides(std::uint32_t index) const noexcept {
        for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
            if (i != index && lanes_[i].phase == Phase::Prefill) { return true; }
        }
        return false;
    }

    // One decode round of `batch` sequences: its inputs are staged at fixed device addresses and
    // every row is chosen on the device, so one CUDA graph per batch size serves every round. The
    // first round of a size captures it (replay; startup normally did, capture_graph_set) and
    // later rounds replay it.
    // `position`: the first sequence's (a one-column round's context, for the plain divisor). `graph`: the
    // round kind's graph for this batch (graphs_, or screened_graphs_ for a screened round).
    struct DecodeGraph; // below, with the graphs
    void run_decode(std::int32_t batch, std::int32_t position, DecodeGraph& graph, bool screened) {
        const cudaStream_t s = device_.stream;
        if (one_column_divisor() != 0) { *reinterpret_cast<std::int32_t*>(host_io() + io_layout_.divisor) = plain_divisor(position); }
        upload_pinned(io_device_.p, io_host_.data(), io_layout_.ngram, s);
        if (cpu_service_) { cpu_service_->wake(); } // its first request comes within this round
        residency_->before_round(s, /*landing=*/true, /*walking=*/walk_.active);
        // The lossless route runs eagerly (its experts' pass reads job counts on the host).
        if (c_.lossless) {
            decode_graph_body(batch, screened);
        } else {
            replay(graph, [&] { decode_graph_body(batch, screened); });
        }
        residency_->enqueue_route_download(s, batch);
    }

    // screened-lm-head: whether a round of these rows and `columns` logit columns takes the screened head.
    // Every row must be greedy (only the argmax of its logits is read), without presence or frequency
    // penalties (they move rows that the screen ranked below its margin), a grammar or a token
    // constraint (masked or observed logits); and the round at most ops::kScreenedHeadMaxColumns wide.
    [[nodiscard]] bool screened_round(std::span<const std::uint32_t> lanes, std::int32_t columns) const {
        if (!screened_head_ || c_.lossless || columns > ops::kScreenedHeadMaxColumns) { return false; }
        for (std::size_t b = 0; b < lanes.size(); ++b) {
            const Lane& lane              = lanes_[lanes[b]];
            const ops::SamplingConfig& sc = lane.sampling;
            if (sc.temperature > 0.0F || sc.presence_penalty != 0.0F || sc.frequency_penalty != 0.0F ||
                lane.constraint != nullptr || (grammar_ != nullptr && grammar_->constrained(b))) {
                return false;
            }
        }
        return true;
    }

    struct DecodeGraph {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
    };

    // ---- the graphed rounds by shape: each round and the startup capture (capture_graph_set) take a
    // shape's graph slot and record its body through these, so both capture the same kernels ----
    DecodeGraph& decode_graph(std::int32_t batch, bool screened) {
        return (screened ? screened_graphs_ : graphs_)[static_cast<std::size_t>(batch - 1)];
    }
    // A decode round of `batch` rows (also the lossless route's eager round).
    void decode_graph_body(std::int32_t batch, bool screened) {
        const execution::NgramRowGate gate = row_gate();
        forward_call(batch, 1, batch, nullptr, nullptr, nullptr, &gate, false, /*decode_round=*/true, screened);
    }
    DecodeGraph& verify_graph(std::int32_t batch, std::int32_t W, bool screened) {
        return (screened ? screened_verify_graphs_ : verify_graphs_)[static_cast<std::size_t>(batch - 1) * max_width_ + (W - 1)];
    }
    // A verification round's forward over batch x W columns.
    void verify_graph_body(std::int32_t batch, std::int32_t W, bool screened) {
        const execution::NgramRowGate gate  = row_gate();
        const execution::ForwardVerify view = verify_view(batch, W);
        forward_call(batch, W, batch * W, &view, nullptr, nullptr, &gate, false, /*decode_round=*/true, screened);
    }
    DecodeGraph& commit_graph(std::int32_t batch, std::int32_t W) {
        return commit_graphs_[static_cast<std::size_t>(batch - 1) * max_width_ + (W - 1)];
    }
    // A verification round's commit (commit_verified): the GDN fold, the QSA tails and the PLE convolution
    // commit of each row's committed columns (and on DFlash2 without MTP the drafter's context), every row's
    // count and slot read on the device; `rows` (an eager commit) gives the fold its rows on the host instead.
    void commit_body(std::int32_t batch, std::int32_t W, std::span<const ops::GdnReplayFoldRow> rows = {}) {
        const cudaStream_t s = device_.stream;
        auto* io             = static_cast<std::byte*>(io_device_.p);
        const Tensor commit_columns(spec_device(spec_layout_.commit), DType::I32, {batch});
        const Tensor slots(io + io_layout_.slots, DType::I32, {batch});
        const Tensor positions(io + io_layout_.positions, DType::I32, {W, batch});
        const auto layers = dim(c_.attention_layers);
        Tensor tails(tails_backing_.p, DType::BF16, {di_, r_ - 1, dim(options_.max_concurrency), layers});
        Tensor ple_states(ple_backing_.p, DType::BF16, {width_, span_, dim(options_.max_concurrency)});
        if (rows.empty()) {
            fold(W).execute_device(slots, commit_columns, batch, s);
        } else {
            fold(W).execute(rows, s);
        }
        ops::qsa_commit_tails(Tensor(qsa_records_.p, DType::BF16, {di_, W, batch, layers}), positions, commit_columns,
                              tails, slots, execution::qsa_geometry(c_), s);
        ops::ple_conv_commit(Tensor(ple_records_.p, DType::BF16, {width_, W, batch}), commit_columns, ple_states, slots, s);
        if (rows.empty() && dflash2_ && !mtp_) { drafter_->materialize(dflash2_context_call(batch, W)); }
    }
    DecodeGraph& mtp_catch_graph(std::int32_t batch, std::int32_t W) {
        return mtp_catch_graphs_[static_cast<std::size_t>(batch - 1) * max_width_ + (W - 1)];
    }
    // The MTP drafter's catch-up over a round of batch x W columns (mtp_catch_up): its cells from the
    // round's residuals and the staged ids, and its index-key tails by each row's committed count.
    void mtp_catch_body(std::int32_t batch, std::int32_t W) {
        const cudaStream_t s = device_.stream;
        auto* io             = static_cast<std::byte*>(io_device_.p);
        const Tensor positions(io + io_layout_.positions, DType::I32, {W * batch});
        const Tensor slots(io + io_layout_.slots, DType::I32, {batch});
        forward_->run_mtp({.residuals        = Tensor(mtp_residuals_.p, DType::BF16, {width_, W * batch}),
                           .ids              = Tensor(mtp_device(mtp_io_.up_ids), DType::I32, {W * batch}),
                           .positions        = positions,
                           .rope_positions   = Tensor(io + io_layout_.rope, DType::I32, {W * batch, 3}),
                           .block_start_rope = Tensor(io + io_layout_.block_rope, DType::I32, {batch, 3}),
                           .slots            = slots,
                           .table_rows       = Tensor(io + io_layout_.rows, DType::I32, {batch}),
                           .batch            = batch,
                           .width            = W,
                           .kv_only          = true,
                           .key_records      = Tensor(mtp_records_.p, DType::BF16, {di_, W, batch})});
        Tensor tails(static_cast<std::byte*>(tails_backing_.p) +
                         static_cast<std::size_t>(c_.attention_layers) * di_ * (r_ - 1) * options_.max_concurrency * 2,
                     DType::BF16, {di_, r_ - 1, dim(options_.max_concurrency), 1});
        ops::qsa_commit_tails(Tensor(mtp_records_.p, DType::BF16, {di_, W, batch, 1}), positions.view({W, batch}),
                              Tensor(spec_device(spec_layout_.commit), DType::I32, {batch}), tails, slots,
                              execution::qsa_geometry(c_), s);
    }

    // Runs a fixed-address round: captured on its first use (the Ops' first-use setup is capture-safe:
    // function attributes, the TMA descriptors' relaxed-mode staging; design §19.3.34), then by replay.
    void replay(DecodeGraph& graph, const std::function<void()>& body) {
        const cudaStream_t s = device_.stream;
        if (graph.executable.ready()) {
            if (!graph_set_.capturing) { graph.executable.launch(s); }
            return;
        }
        prepare_graph(graph, [&] { graph.definition.capture(s, body); });
    }

    // replay for a chain of `steps` bodies whose body j > 0 runs only while the gate after body
    // j - 1 passes (DecodeGraphDefinition::capture_gated_chain).
    void replay_gated(DecodeGraph& graph, std::int32_t steps, const std::function<void(std::int32_t)>& body,
                      const std::function<void(std::int32_t, cudaGraphConditionalHandle)>& gate) {
        const cudaStream_t s = device_.stream;
        if (graph.executable.ready()) {
            if (!graph_set_.capturing) { graph.executable.launch(s); }
            return;
        }
        prepare_graph(graph, [&] { graph.definition.capture_gated_chain(s, steps, body, gate); });
    }

    // A graph's capture (`capture` records its body on the device stream) and instantiation, then its first
    // launch; while capture_graph_set captures the startup set, its upload instead (nothing runs: a graph
    // reads its inputs by address when it is launched).
    void prepare_graph(DecodeGraph& graph, const std::function<void()>& capture) {
        const cudaStream_t s = device_.stream;
        // A captured body bakes in whatever it reads by address: the arena *work_ points at and
        // the per-layer waits forward_call consumes. Neither may be a call's temporary.
        if (wide_active_ || !next_waits_[0].empty() || !next_waits_[1].empty()) {
            throw std::logic_error("Qwen4Exp: a graph capture inside a wide call or with pending layer waits");
        }
        const std::size_t free_before = trace_ ? trace_free() : 0;
        capture();
        graph.executable.instantiate(graph.definition);
        if (graph_set_.capturing) {
            graph.executable.upload(s);
            ++graph_set_.captured;
        } else {
            graph.executable.launch(s);
            if (graph_set_.at_startup) {
                diagnostic("CUDA graph " + graph_label(graph) +
                               " captured at its first use: the startup capture did not "
                               "include its shape",
                           DiagnosticLevel::Warning);
            }
        }
        if (trace_) { trace_graph(graph, free_before); }
    }

    // Candidates §23 NU-04 (program/graph_set.cpp): at the end of startup, every graph a round may replay (the
    // shapes decode() and commit() can reach) is captured, instantiated and uploaded without running, so no
    // request pays a capture. Same kernels as a first-use capture. The lossless route and scoring skip it.
    void capture_graph_set();
    // A graph slot's family and shape ("verify 1x5"), for the startup capture's lines.
    [[nodiscard]] std::string graph_label(const DecodeGraph& graph) const;

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
        if (at < 0) { at = route_trace_index(screened_graphs_, &graph); }
        if (at < 0 && ((at = route_trace_index(verify_graphs_, &graph)) >= 0 ||
                       (at = route_trace_index(screened_verify_graphs_, &graph)) >= 0)) {
            family = RouteTraceGraph::Verify;
            stride = max_width_;
        } else if (at < 0 && ((at = route_trace_index(mtp_draft_graphs_, &graph)) >= 0 ||
                              (at = route_trace_index(mtp_sampled_graphs_, &graph)) >= 0 ||
                              (at = route_trace_index(mtp_gated_graphs_, &graph)) >= 0)) {
            family = RouteTraceGraph::MtpDraft;
            stride = mtp_k_;
        } else if (at < 0 && (at = route_trace_index(mtp_catch_graphs_, &graph)) >= 0) {
            family = RouteTraceGraph::MtpCatchUp;
            stride = max_width_;
        } else if (at < 0 && (at = route_trace_index(dflash_propose_graphs_, &graph)) >= 0) {
            family = RouteTraceGraph::DFlashPropose;
        } else if (at < 0 && (at = route_trace_index(dflash_context_graphs_, &graph)) >= 0) {
            family = RouteTraceGraph::DFlashCatchUp;
            stride = max_width_;
        }
        if (at < 0) { return; }
        trace_->graph(family, static_cast<std::uint32_t>(at / stride + 1), static_cast<std::uint32_t>(at % stride + 1),
                      free_before, trace_free());
    }

    // A call whose MoE layers may publish fetch requests runs inside an open tier round: the read
    // agent answers them only then, and a closed round would read as a silent host after a second
    // per layer. The startup capture (capture_graph_set) enqueues nothing, so it needs no round.
    void require_tier_round() const {
        if (fetch_channel_ && !graph_set_.capturing && !tier_->round_open()) {
            throw std::logic_error("Qwen4Exp: a call reading SSD-only experts was enqueued outside the tier's round");
        }
    }

    void forward_call(std::int32_t batch, std::int32_t width, std::int32_t logit_columns,
                      const execution::ForwardVerify* verify = nullptr, const execution::MtpChunk* chunk = nullptr,
                      const execution::VisionInput* vision = nullptr, const execution::NgramRowGate* gate = nullptr,
                      bool streamed = false, bool decode_round = false, bool screened_head = false) {
        require_tier_round();
        execution::ForwardBatch fb = io_batch(static_cast<std::byte*>(io_device_.p), batch, width, logit_columns);
        fb.decode_round  = decode_round;
        fb.screened_head = screened_head;
        fb.ngram_gate    = gate;
        fb.verify        = verify;
        fb.mtp_chunk     = chunk;
        fb.layer_waits   = next_waits_;
        next_waits_      = {};
        fb.vision        = vision;
        fb.stream        = streamed;
        fb.split         = streamed && split_call_;
        // Decode and verification rounds export their final residuals for the MTP catch-up.
        const std::int32_t cols = batch * width;
        if (mtp_ && chunk == nullptr) { fb.residual_out = Tensor(mtp_residuals_.p, DType::BF16, {width_, cols}); }
        // DFlash2's target features: every column of a decode or verification round, a prefill call's
        // or forced tokens' last window (the ring keeps no earlier position).
        if (dflash2_) {
            const auto [column, n] = decode_round ? std::pair<std::int32_t, std::int32_t>{0, cols} : dflash_call_region(cols);
            fb.draft_features      = dflash_feature_columns(column, n);
        }
        // A scored prompt's calls keep their final residuals for the LM head (causal_score).
        if (scoring_) { fb.residual_out = score_residuals(score_column_, cols); }
        // Lossless generation runs the cache form, bounded by the call's largest position (the staged io).
        if (c_.lossless && !options_.causal_scoring) {
            const auto* positions = reinterpret_cast<const std::int32_t*>(host_io() + io_layout_.positions);
            fb.lossless_keys      = *std::max_element(positions, positions + cols) + 1;
        }
        Tensor logits = head_logits(logit_columns);
        forward_->run(fb, logits);
        round_logits(logit_columns, device_.stream);
    }

    // The head's logits of `columns` columns as the Forward writes them: FP32 (logits32_), or with BF16 logits on
    // the lossless route BF16 (logits16_, each exact sum correctly rounded).
    Tensor head_logits(std::int32_t columns) {
        if (options_.bf16_logits && c_.lossless) { return Tensor(logits16_.p, DType::BF16, {vocab_, columns}); }
        return Tensor(logits32_.p, DType::FP32, {vocab_, columns});
    }
    // The logits the sampler, verification, constraints and the prompt readout read, `columns` columns: FP32, or
    // BF16 (ProgramOptions::bf16_logits).
    Tensor read_logits(std::int32_t columns) {
        if (options_.bf16_logits) { return Tensor(logits16_.p, DType::BF16, {vocab_, columns}); }
        return Tensor(logits32_.p, DType::FP32, {vocab_, columns});
    }
    // BF16 logits off the lossless route: the head's FP32 sums of `columns` columns rounded once to BF16.
    void round_logits(std::int32_t columns, cudaStream_t s) {
        if (!options_.bf16_logits || c_.lossless || columns == 0) { return; }
        const Tensor wide(logits32_.p, DType::FP32, {vocab_, columns});
        Tensor narrow(logits16_.p, DType::BF16, {vocab_, columns});
        ops::cast_fp32_to_bf16(wide, narrow, s);
    }

    // A call's inputs as the Forward reads them from a staged io image at `base` (the io, or one
    // chunk's copy in a layer walk's area).
    execution::ForwardBatch io_batch(std::byte* base, std::int32_t batch, std::int32_t width,
                                     std::int32_t logit_columns) const {
        const std::int32_t cols = batch * width;
        execution::ForwardBatch fb;
        fb.ids           = Tensor(base + io_layout_.ids, DType::I32, {cols});
        fb.positions     = Tensor(base + io_layout_.positions, DType::I32, {cols});
        fb.rope_positions   = Tensor(base + io_layout_.rope, DType::I32, {cols, 3});
        fb.block_start_rope = Tensor(base + io_layout_.block_rope, DType::I32, {batch, 3});
        fb.slots         = Tensor(base + io_layout_.slots, DType::I32, {batch});
        fb.table_rows    = Tensor(base + io_layout_.rows, DType::I32, {batch});
        fb.logit_columns = Tensor(base + io_layout_.columns, DType::I32, {logit_columns});
        fb.ngram_rows    = Tensor(base + io_layout_.ngram, DType::U8,
                                  {dim(ngram_table().row_bytes), dim(hash_.heads()), cols});
        fb.ngram_format  = ngram_table().format;
        if (ngram_scale_.p != nullptr) {
            fb.ngram_scale = Tensor(ngram_scale_.p, ngram_table().nvfp4_scale ? DType::FP32 : DType::BF16, {1});
        }
        fb.host_slots      = std::span<const std::int32_t>(host_lanes_.data(), static_cast<std::size_t>(batch));
        fb.host_table_rows = fb.host_slots;
        fb.batch           = batch;
        fb.width           = width;
        return fb;
    }

    // ---------------------------------------------------------------- speculative verification
    void allocate_verification(std::int32_t lanes, std::uint64_t& allocated) {
        allocate(records_backing_, plan_.records_bytes, allocated);
        records_ = GdnReplayRecords(DeviceSpan{records_backing_.p, records_backing_.bytes}, plan_.records);
        allocate(ple_records_, plan_.ple_records, allocated);
        allocate(qsa_records_, plan_.qsa_records, allocated);
        folds_.resize(static_cast<std::size_t>(max_width_) + 1);
        verify_graphs_.resize(static_cast<std::size_t>(lanes) * max_width_);
        commit_graphs_.resize(static_cast<std::size_t>(lanes) * max_width_);
        if (screened_head_) { screened_verify_graphs_.resize(static_cast<std::size_t>(lanes) * max_width_); }
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
    //
    // The n-gram rows not yet staged (`staged[b]` columns of row b are) are read on the host after
    // the launch, while the GPU embeds and runs layer 0, and released to the round's gate before
    // ple_embed (design §12.3, n-gram S2). A round that captures or warms its graph reads them
    // first and releases the gate before the launch. Every call that writes the landing area
    // ends in a stream sync before the next round stages, so a gate never reads it across calls.
    PendingBatch verify(std::span<const SequenceHandle> sequences, std::span<const std::uint32_t> lanes,
                        std::span<const std::int32_t> positions, std::int32_t width,
                        std::span<const std::int32_t> staged, Clock::time_point start) {
        const cudaStream_t s = device_.stream;
        const auto batch     = static_cast<std::int32_t>(lanes.size());
        const std::int32_t W = width, K = width - 1;
        auto* ids     = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.ids);
        auto* pos     = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.positions);
        auto* columns = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.columns);
        auto* slots   = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.slots);
        auto* rows    = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.rows);
        for (std::int32_t b = 0; b < batch; ++b) {
            Lane& lane = lanes_[lanes[b]];
            const auto n = static_cast<std::int32_t>(drafts_[b].size());
            const std::int32_t p = positions[b];
            for (std::int32_t j = 0; j < W; ++j) {
                ids[b * W + j]     = verify_token(lane, static_cast<std::size_t>(b), p, p + j);
                pos[b * W + j]     = p + j;
                columns[b * W + j] = b * W + j;
            }
            for (std::int32_t j = 0; j < K; ++j) { spec_host(spec_layout_.drafts)[b * K + j] = ids[b * W + j + 1]; }
            slots[b]       = static_cast<std::int32_t>(lanes[b]);
            rows[b]        = static_cast<std::int32_t>(lanes[b]);
            host_lanes_[b] = static_cast<std::int32_t>(lanes[b]);
            stage_call_rope(lane, static_cast<std::uint32_t>(p), W, batch * W, b * W, batch, b);
            spec_host(spec_layout_.extents)[b] = n;
            spec_host(spec_layout_.lengths)[b] = p;
            spec_host(spec_layout_.anchors)[b] = lane.history.back();
        }
        const auto read_rows = [&] {
            for (std::int32_t b = 0; b < batch; ++b) {
                stage_verify_ngram(lanes_[lanes[b]], static_cast<std::size_t>(b), positions[b], W, staged[b], W);
            }
        };
        const std::uint32_t word = next_gate_word();
        const bool screened = screened_round(lanes, batch * W);
        DecodeGraph& graph  = verify_graph(batch, W, screened);
        const bool gated    = graph.executable.ready();
        if (!gated) {
            read_rows();
            publish_pinned_word(gate_ready(), word);
        }
        upload_pinned(io_device_.p, io_host_.data(), io_layout_.ngram, s);
        upload_pinned(spec_device_.p, spec_host_.data(), 4ULL * spec_layout_.licensed, s);
        if (cpu_service_) { cpu_service_->wake(); } // its first request comes within this round
        residency_->before_round(s, /*landing=*/true, /*walking=*/walk_.active);
        replay(graph, [&] { verify_graph_body(batch, W, screened); });
        // From here an enqueued gate waits for `word`: an error before it is published still
        // publishes it (over zeroed rows), so the stream drains and the error keeps its class.
        GateRelease release_gate{gated ? this : nullptr, word, io_prefix(batch * W) - io_layout_.ngram};
        residency_->enqueue_route_download(s, batch * W);

        // Acceptance on the device, from the logits plain decode samples (FP32, or BF16: design 16.5); the
        // constraints mask them in place.
        Tensor logits = read_logits(batch * W);
        Tensor target(spec_device(spec_layout_.target), DType::I32, {batch * W});
        ops::argmax(logits, target, token_domain_, s);
        const bool constrained = stage_constraints(lanes, W);
        if (constrained) { constrain(logits, &target); }
        auto* configs = static_cast<ops::SamplingConfig*>(host_configs_.data());
        for (std::int32_t b = 0; b < batch; ++b) {
            configs[b]      = lanes_[lanes[b]].sampling;
            configs[b].mask = grammar_mask(static_cast<std::size_t>(b), drafts_[b]);
        }
        upload_pinned(configs_.p, configs, sizeof(ops::SamplingConfig) * batch, s);
        Tensor drafts(spec_device(spec_layout_.drafts), DType::I32, {K, batch});
        Tensor extents(spec_device(spec_layout_.extents), DType::I32, {batch});
        Tensor lengths(spec_device(spec_layout_.lengths), DType::I32, {batch});
        Tensor anchors(spec_device(spec_layout_.anchors), DType::I32, {batch});
        Tensor licensed(spec_device(spec_layout_.licensed), DType::I32, {W, batch});
        Tensor counts(spec_device(spec_layout_.counts), DType::I32, {batch});
        Tensor accepted(spec_device(spec_layout_.accepted), DType::I32, {batch});
        // A sampled row's DFlash2 drafts were drawn from the selector's laws: such a round verifies
        // by sparse rejection sampling with them (greedy rows' laws are one-hot, n-gram rows' copies
        // one-hot too). A round of greedy rows drafted deterministically and takes the greedy Op.
        bool dflash_sampled = false;
        if (std::exchange(dflash_round_, false)) {
            for (std::int32_t b = 0; b < batch; ++b) {
                dflash_sampled = dflash_sampled || lanes_[lanes[b]].sampling.temperature > 0.0F;
            }
        }
        const bool sampled = std::exchange(mtp_sampled_round_, false);
        const bool sparse  = sampled || dflash_sampled;
        if (dflash_sampled) {
            constexpr std::int32_t C = ops::kSparseSpeculativeCandidates;
            const auto [at_candidates, at_q] = dflash2_verify_planes(batch, K);
            Tensor candidates(dflash_device(at_candidates), DType::I32, {C, K, batch});
            Tensor q(dflash_device(at_q), DType::FP32, {C, K, batch});
            auto scope = work_->scope();
            ops::speculative_accept_sparse_drafts(target.view({W, batch}), logits.view({vocab_, W, batch}), drafts,
                                                  candidates, q, extents, lengths, anchors, licensed, counts, accepted,
                                                  token_domain_, static_cast<const ops::SamplingConfig*>(configs_.p),
                                                  ops::SpeculativeAcceptExecutionEnvelope{.block_verification = options_.block_verification},
                                                  *work_, s);
        } else if (sampled) {
            // Sampled drafts: each row's 16 candidates and q per verified position, [16, K, batch]: an
            // MTP row's from its draft steps, an n-gram row's one-hot at its copied token.
            constexpr std::int32_t C = ops::kSparseSpeculativeCandidates;
            std::int32_t* host_candidates = mtp_host(mtp_io_.verify_candidates);
            auto* host_q                  = reinterpret_cast<float*>(mtp_host(mtp_io_.verify_q));
            for (std::int32_t b = 0; b < batch; ++b) {
                const auto n = static_cast<std::int32_t>(drafts_[b].size());
                if (n == 0) { continue; }
                const std::size_t row = static_cast<std::size_t>(b) * K * C;
                if (from_ngram_[b]) {
                    for (std::int32_t j = 0; j < n; ++j) {
                        const std::int32_t token = drafts_[b][static_cast<std::size_t>(j)];
                        for (std::int32_t c = 0; c < C; ++c) {
                            host_candidates[row + j * C + c] = (token + c) % token_domain_;
                            host_q[row + j * C + c]          = c == 0 ? 1.0F : 0.0F;
                        }
                    }
                    upload_pinned(mtp_device(mtp_io_.verify_candidates + row), host_candidates + row, 4ULL * C * n, s);
                    upload_pinned(mtp_device(mtp_io_.verify_q + row), host_q + row, 4ULL * C * n, s);
                    continue;
                }
                // Draft step j's plane of row b is at (j * batch + b) * C.
                for (const auto [from, to] : {std::pair{mtp_io_.draft_candidates, mtp_io_.verify_candidates},
                                              std::pair{mtp_io_.draft_q, mtp_io_.verify_q}}) {
                    CUDA_CHECK(cudaMemcpy2DAsync(mtp_device(to + row), 4ULL * C,
                                                 mtp_device(from + static_cast<std::size_t>(C) * b), 4ULL * C * batch,
                                                 4ULL * C, static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice, s));
                }
            }
            Tensor candidates(mtp_device(mtp_io_.verify_candidates), DType::I32, {C, K, batch});
            Tensor q(mtp_device(mtp_io_.verify_q), DType::FP32, {C, K, batch});
            auto scope = work_->scope();
            ops::speculative_accept_sparse_drafts(target.view({W, batch}), logits.view({vocab_, W, batch}), drafts,
                                                  candidates, q, extents, lengths, anchors, licensed, counts, accepted,
                                                  token_domain_, static_cast<const ops::SamplingConfig*>(configs_.p),
                                                  ops::SpeculativeAcceptExecutionEnvelope{.block_verification = options_.block_verification},
                                                  *work_, s);
        } else {
            auto scope = work_->scope();
            ops::speculative_accept_greedy_drafts(target.view({W, batch}), logits.view({vocab_, W, batch}), drafts,
                                                  extents, lengths, anchors, licensed, counts, accepted,
                                                  token_domain_, static_cast<const ops::SamplingConfig*>(configs_.p),
                                                  *work_, s);
        }
        CUDA_CHECK(cudaMemcpyAsync(spec_host(spec_layout_.licensed), licensed.data,
                                   4ULL * (spec_layout_.commit - spec_layout_.licensed), cudaMemcpyDeviceToHost, s));
        if (gated) {
            read_behind_gate(lanes_[lanes[0]], word, read_rows);
            release_gate.owner = nullptr;
        }
        device_.synchronize();
        check_expert_error();

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
            if (sparse && lane.sampling.token_counts != nullptr) {
                // The sparse acceptance leaves token counts to the caller (the greedy-draft Op counts
                // each licensed token itself): count them before the next round's penalties.
                Tensor ids(spec_device(spec_layout_.licensed) + static_cast<std::size_t>(b) * W, DType::I32, {L});
                Tensor token_counts(lane.sampling.token_counts, DType::I32, {token_domain_});
                ops::increment_token_counts(ids, token_counts, s);
            }
            if (constrained) {
                take_constraint_draws(lane, static_cast<std::size_t>(b * W),
                                      std::span<const std::int32_t>(&pending_tokens_[static_cast<std::size_t>(b * W)],
                                                                    static_cast<std::size_t>(L)));
            }
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
                    const double hit = j < static_cast<std::uint64_t>(A) ? 1.0 : 0.0;
                    double& a        = lane.mtp_accept[static_cast<std::size_t>(j)];
                    a                = (1.0 - kAcceptanceWeight) * a + kAcceptanceWeight * hit;
                }
                // The chain's verified drafts against the cumulative cut's bin table (rows whose drafts
                // read_mtp_drafts kept: argmax drafts).
                if (mtp_ && !dflash2_ && !(sparse && lane.sampling.temperature > 0.0F)) {
                    lane.mtp_cumulative.observe(mtp_draft_logprobs_[static_cast<std::size_t>(b)].data(),
                                                static_cast<std::int32_t>(std::min<std::uint64_t>(n, 8)),
                                                static_cast<std::int32_t>(std::min<std::uint64_t>(A, n)));
                }
                if (dflash2_) {
                    const std::span<const float> confidences(dflash_confidence_[b].data(), n);
                    const auto verified = static_cast<std::uint32_t>(n);
                    const auto taken    = static_cast<std::uint32_t>(std::min<std::uint64_t>(A, n));
                    lane.dflash_calibration.observe(confidences, verified, taken, kAcceptanceWeight);
                    dflash_process_calibration_.observe(confidences, verified, taken, kAcceptanceWeight);
                }
            }
            if (from_ngram_[b]) {
                ++stats.ngram_rounds;
                stats.ngram_drafted_tokens += n;
                stats.ngram_accepted_tokens += static_cast<std::uint64_t>(A);
                // The bucket's acceptance per position.
                if (ngram_bucket_[b] >= 0) {
                    double& a = lane.ngram_accept[static_cast<std::size_t>(ngram_bucket_[b])];
                    for (std::uint64_t j = 0; j < n && j <= static_cast<std::uint64_t>(A); ++j) {
                        const double hit = j < static_cast<std::uint64_t>(A) ? 1.0 : 0.0;
                        a = (1.0 - kAcceptanceWeight) * a + kAcceptanceWeight * hit;
                    }
                }
            }
            for (std::int32_t j = 0; j < A; ++j) { ++stats.accepted_per_position[static_cast<std::size_t>(j)]; }
            stats.round_steps.push_back({.drafted  = static_cast<std::uint16_t>(n),
                                         .accepted = static_cast<std::uint16_t>(A),
                                         .ngram    = from_ngram_[b]});
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
        auto pending = ContractAccess::make_pending(this, pending_transaction_, sequences,
                                                    {pending_tokens_.data(), static_cast<std::size_t>(batch * W)},
                                                    timing, {pending_counts_.data(), static_cast<std::size_t>(batch)},
                                                    static_cast<std::uint32_t>(W));
        mark_row_failures(pending, {pending_counts_.data(), static_cast<std::size_t>(batch)});
        return pending;
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
            out.rows[row].speculative_counters = speculative_counters(lane.speculative);
            if (d.cancelled || d.failed) {
                lane.prefix.withdraw = pending.row_failure(static_cast<std::size_t>(row)) ==
                                       runtime::PendingRowFailure::NonFiniteLogits;
                lane.constraint_round.clear();
                out.rows[row].timings     = timings(lane);
                out.rows[row].speculative = lane.speculative;
                out.rows[row].disposition = d.failed ? runtime::CommitDisposition::FailedReleased
                                                     : runtime::CommitDisposition::CancelledReleased;
                continue;
            }
            const auto k = static_cast<std::int32_t>(d.accepted_tokens);
            if (k < 1 || k > pending_counts_[row]) { throw std::logic_error("Qwen4Exp: commit exceeds the licensed tokens"); }
            commit_constraint_draws(lane, static_cast<std::size_t>(k));
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
        // The fold, QSA tails and PLE commit (and on DFlash2 without MTP the drafter's context too) as one graph per
        // batch size and width, which reads every row's count and slot from the device, instead of three launches
        // (four with the context's graph) on the host's turnaround each round (design §19.3.34). Same bits.
        const bool graphed          = !c_.lossless;
        const bool context_in_graph = graphed && dflash2_ && !mtp_;
        if (graphed) {
            replay(commit_graph(batch, W), [&] { commit_body(batch, W); });
        } else {
            commit_body(batch, W, std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), static_cast<std::size_t>(batch)));
        }
        // The drafter's catch-up on a stream of greater priority beside these commit kernels, with the next
        // round's drafts pre-drafted there (candidates §13 SR-01), measured within chance (removed 2026-10-28).
        if (mtp_) { mtp_catch_up(rows, W, commit); }
        if (context_in_graph) {
            dflash2_settle_frontiers(rows, commit);
        } else if (dflash2_) {
            dflash2_catch_up(rows, W, commit);
        }
        for (const auto& row : rows) { require_mtp_settled(lanes_[ContractAccess::lane(row)]); }
        for (std::int32_t row = 0; row < batch; ++row) {
            const auto& d = decisions[static_cast<std::size_t>(row)];
            if (!d.cancelled && !d.failed) {
                Lane& lane = lanes_[ContractAccess::lane(rows[row])];
                prefix_publish(lane, prefix_frontier(lane, false));
            }
        }
        settle_round();
        if (replay_clock_) { replay_clock_->end(device_.stream); }
        for (std::int32_t row = 0; row < batch; ++row) {
            const auto& d = decisions[static_cast<std::size_t>(row)];
            if (d.cancelled || d.failed) { release(ContractAccess::lane(rows[row])); }
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
        mtp_gated_graphs_.resize(static_cast<std::size_t>(mtp_k_));
        if (sampled_drafts_) { mtp_sampled_graphs_.resize(static_cast<std::size_t>(lanes) * mtp_k_); }
    }

    // The drafter pool (candidates §22 VM-01 variant B): the drafter's routed experts behind a
    // pool of device slots (their banks were bound pinned, Config::mtp_expert_slots); the draft steps route among
    // the pooled experts and settle_round moves the pool to their demand.
    void create_drafter_pool(execution::ForwardExperts& experts, std::uint64_t& allocated) {
        const auto& m        = *parameters_.mtp;
        const Config& config = parameters_.model.config();
        const std::int32_t E = dim(c_.moe.experts), I = dim(c_.moe.intermediate), H = dim(c_.hidden_size);
        const auto slots     = dim(config.mtp_expert_slots);
        if (runtime::expert_offload::DrafterExpertPool::expert_bytes(m.experts_gate_up, m.experts_down, I, H) != config.mtp_expert_bytes) {
            throw std::logic_error("Qwen4Exp: the drafter's bound experts differ from the drafter pool's plan");
        }
        allocate(drafter_pool_memory_, plan_.drafter_pool, allocated);
        drafter_pool_ = std::make_unique<runtime::expert_offload::DrafterExpertPool>(m.experts_gate_up, m.experts_down, E, I, H, slots,
                                                            drafter_pool_memory_.p, plan_.drafter_pool, device_.stream);
        experts.drafter_slots   = slots;
        experts.drafter_gate_up = drafter_pool_->gate_up();
        experts.drafter_down    = drafter_pool_->down();
        experts.drafter_slot_of = drafter_pool_->slot_of();
        experts.drafter_demand  = drafter_pool_->demand();
        drafter_reported_       = Clock::now();
        diagnostic("MTP drafter pool: " + std::to_string(slots) + " of its " + std::to_string(E) +
                       " routed experts in a VRAM pool (" + std::to_string(plan_.drafter_pool >> 20) +
                       " MiB), the rest pinned in RAM; draft steps route among the pooled experts",
                   DiagnosticLevel::Info);
    }
    // The pool's share of draft picks it held and its swaps since startup, at most every 10 seconds.
    void report_drafter_pool() {
        const auto now = Clock::now();
        if (now - drafter_reported_ < std::chrono::seconds(10)) { return; }
        drafter_reported_                    = now;
        const runtime::expert_offload::DrafterExpertPool::Stats& s = drafter_pool_->stats();
        if (s.picks == 0) { return; }
        char line[192];
        std::snprintf(line, sizeof(line), "MTP drafter pool: %.1f %% of %llu draft picks pooled, %llu swaps",
                      100.0 * static_cast<double>(s.picks - s.misses) / static_cast<double>(s.picks),
                      static_cast<unsigned long long>(s.picks), static_cast<unsigned long long>(s.swaps));
        diagnostic(line, DiagnosticLevel::Info);
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
        // Cell c takes the target's RoPE position of c (design §19.3.2), per sub-chunk.
        const std::int32_t subchunks = (cells + execution::kMtpChunkColumns - 1) / execution::kMtpChunkColumns;
        stage_mtp_chunk_rope(lane.rope, static_cast<std::uint32_t>(first), cells, execution::kMtpChunkColumns,
                             static_cast<std::uint32_t>(r_), io_words(io_layout_.mtp_rope, 3ULL * cells),
                             io_words(io_layout_.mtp_block_rope, 3ULL * subchunks));
        lane.mtp_cells = static_cast<std::uint32_t>(begin + columns);
        lane.mtp_live  = true;
        auto* base = static_cast<std::byte*>(io_device_.p);
        execution::MtpChunk chunk;
        chunk.saved     = saved_column(index);
        chunk.prepend   = prepend;
        chunk.columns   = columns;
        // A one-token prompt has no cell yet (its only position is the pending cell).
        if (cells > 0) {
            chunk.ids              = Tensor(base + io_layout_.mtp_ids, DType::I32, {cells});
            chunk.positions        = Tensor(base + io_layout_.mtp_cells, DType::I32, {cells});
            chunk.rope_positions   = reinterpret_cast<std::int32_t*>(base + io_layout_.mtp_rope);
            chunk.block_start_rope = reinterpret_cast<std::int32_t*>(base + io_layout_.mtp_block_rope);
            chunk.vision           = stage_mtp_vision(lane, static_cast<std::uint32_t>(first), cells, subchunks);
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
    // next position stays current (K = 0 then probes one draft). A fixed draft length (no
    // auto_draft_length) drafts mtp_k_ every round.
    std::int32_t choose_draft_length(Lane& lane) const {
        const std::int32_t k = draft_length_at(lane, lane.mtp_policy_rounds + 1);
        if (options_.auto_draft_length && !dflash2_) { ++lane.mtp_policy_rounds; }
        return k;
    }

    // choose_draft_length's choice as the lane's policy round `round` (1-based), without counting it.
    std::int32_t draft_length_at(const Lane& lane, std::uint64_t round) const {
        const std::int32_t most = neural_k();
        // DFlash2 proposes its whole block; its length is chosen after the proposal (dflash2_auto_length).
        if (!options_.auto_draft_length || dflash2_) { return most; }
        std::int32_t best = 0;
        double best_score = 1.0;
        double reach = 1.0, tokens = 1.0;
        const double cost = width_cost();
        for (std::int32_t k = 1; k <= most; ++k) {
            reach *= lane.mtp_accept[static_cast<std::size_t>(k - 1)];
            tokens += reach;
            const double score = tokens / (1.0 + cost * k);
            if (score > best_score) {
                best_score = score;
                best       = k;
            }
        }
        if (round % kProbeInterval == 0) { best = std::min(best + 1, most); }
        return best;
    }

    // The cost of one draft in K = 0 rounds (design 11.3), learned online: consecutive one-row MTP
    // rounds' start-to-start times (drafting, verification, commit and host work) against their
    // drafter steps K, fit as t = a + b K by exponentially weighted least squares; a draft then costs
    // b / a rounds. The probes (one draft more every kProbeInterval rounds) keep K varied. Until
    // the fit has kCostMinRounds rounds of spread K (or when it is not positive), the measured prior
    // kWidthCost stands. A cycle past kCostGapRatio predicted rounds held an idle gap and is skipped.
    struct RoundCostFit {
        double n = 0.0, k = 0.0, kk = 0.0, t = 0.0, kt = 0.0;
    };
    RoundCostFit round_cost_;
    Clock::time_point last_round_start_{};
    std::int32_t last_round_steps_ = -1; // -1: the previous round was not a one-row MTP round

    // The fit's a and b (seconds, seconds per draft); false while it is not determined or positive.
    [[nodiscard]] bool round_cost_fit(double& a, double& b) const noexcept {
        const RoundCostFit& f = round_cost_;
        const double spread   = f.n * f.kk - f.k * f.k;
        if (f.n < kCostMinRounds || spread < kCostMinSpread * f.n * f.n) { return false; }
        b = (f.n * f.kt - f.k * f.t) / spread;
        a = (f.t - b * f.k) / f.n;
        return a > 0.0 && b > 0.0;
    }

    [[nodiscard]] double width_cost() const noexcept {
        // The replay bench (INFERNIX_TMP_REPLAY_BENCH) keeps the prior: draft lengths then follow
        // acceptance only, so A/B arms replaying one greedy request run the same rounds.
        if (replay_clock_) { return kWidthCost; }
        double a = 0.0, b = 0.0;
        return round_cost_fit(a, b) ? std::min(b / a, kCostMax) : kWidthCost;
    }

    // At a decode round's start: the previous round's cycle when both are one-row MTP rounds (an
    // n-gram round's width does not follow its drafter steps: not sampled).
    void observe_round(Clock::time_point start, std::int32_t batch) {
        if (last_round_steps_ >= 0 && batch == 1) {
            const double seconds = std::chrono::duration<double>(start - last_round_start_).count();
            RoundCostFit& f      = round_cost_;
            const double k       = last_round_steps_;
            double a = 0.0, b = 0.0;
            // Before the fit, a cycle past kCostGapRatio mean cycles is a gap.
            const double predict = round_cost_fit(a, b) ? a + b * k : (f.n > 0.0 ? f.t / f.n : seconds);
            if (seconds < kCostGapRatio * predict) {
                f.n  = kCostDecay * f.n + 1.0;
                f.k  = kCostDecay * f.k + k;
                f.kk = kCostDecay * f.kk + k * k;
                f.t  = kCostDecay * f.t + seconds;
                f.kt = kCostDecay * f.kt + k * seconds;
            }
        }
        last_round_start_ = start;
        last_round_steps_ = -1; // set by the round once its drafts are known
    }

    // Two-row rounds (design 19.3.5): the K in [0, kPairMaxDrafts] that maximizes both rows' expected
    // tokens per round time, where a round of K drafts per row costs 1 + kPairWidthCost * K plain
    // two-row rounds. Every kProbeInterval pair rounds it drafts one token more than its choice, so
    // the rows' estimates of the next position stay current. A fixed draft length drafts the most a
    // pair round allows. DFlash2 proposes its whole block for both rows; the width is chosen after
    // the proposal (dflash2_pair_width).
    std::int32_t choose_pair_draft_length(const Lane& a, const Lane& b) {
        const std::int32_t k = pair_draft_length_at(a, b, pair_policy_rounds_ + 1);
        if (drafter_live(a) && drafter_live(b) && options_.auto_draft_length && !dflash2_) { ++pair_policy_rounds_; }
        return k;
    }

    // choose_pair_draft_length's choice as pair policy round `round` (1-based), without counting it.
    std::int32_t pair_draft_length_at(const Lane& a, const Lane& b, std::uint64_t round) const {
        if (!drafter_live(a) || !drafter_live(b)) { return 0; }
        const std::int32_t most = std::min(kPairMaxDrafts, neural_k());
        if (!options_.auto_draft_length || dflash2_) { return most; }
        std::int32_t best = 0;
        double best_score = 2.0;
        double reach_a = 1.0, reach_b = 1.0, tokens = 2.0;
        for (std::int32_t k = 1; k <= most; ++k) {
            reach_a *= a.mtp_accept[static_cast<std::size_t>(k - 1)];
            reach_b *= b.mtp_accept[static_cast<std::size_t>(k - 1)];
            tokens += reach_a + reach_b;
            const double score = tokens / (1.0 + kPairWidthCost * k);
            if (score > best_score) {
                best_score = score;
                best       = k;
            }
        }
        if (round % kProbeInterval == 0) { best = std::min(best + 1, most); }
        return best;
    }

    // `while_drafting` runs on the host after the draft steps are submitted, before their wait
    // (only when draft steps run).
    void mtp_draft(std::span<const std::uint32_t> lanes, std::int32_t steps,
                   const std::function<void()>& while_drafting = {}) {
        const cudaStream_t s = device_.stream;
        const auto batch     = static_cast<std::int32_t>(lanes.size());
        const std::size_t column = 2ULL * width_;
        mtp_sampled_round_       = false;
        bool any = false, unwritten = false;
        for (std::int32_t b = 0; b < batch; ++b) {
            any |= lanes_[lanes[b]].mtp_live;
            unwritten |= lanes_[lanes[b]].mtp_live && lanes_[lanes[b]].mtp_cells < lanes_[lanes[b]].state_tokens;
        }
        // Without drafts (a gated or zero-length round) only unwritten pending cells need the drafter.
        if (!any || (steps == 0 && !unwritten)) { return; }
        const bool sampled = mtp_sampled(lanes);
        stage_mtp_draft(lanes, s);
        for (std::int32_t b = 0; b < batch; ++b) {
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(mtp_chain_.p) + column * b, saved_column(lanes[b]).data,
                                       column, cudaMemcpyDeviceToDevice, s));
        }
        if (unwritten) {
            // The pending cells' K/V (rewriting a written cell is idempotent).
            forward_->run_mtp({.residuals        = Tensor(mtp_chain_.p, DType::BF16, {width_, batch}),
                               .ids              = Tensor(mtp_device(mtp_io_.ids), DType::I32, {batch}),
                               .positions        = Tensor(mtp_device(mtp_io_.cells), DType::I32, {batch}),
                               .rope_positions   = Tensor(mtp_device(mtp_io_.rope_cells), DType::I32, {batch, 3}),
                               .block_start_rope = Tensor(mtp_device(mtp_io_.block_rope_cells), DType::I32, {batch, 3}),
                               .slots            = Tensor(mtp_device(mtp_io_.slots), DType::I32, {batch}),
                               .table_rows       = Tensor(mtp_device(mtp_io_.rows), DType::I32, {batch}),
                               .batch            = batch,
                               .width            = 1,
                               .kv_only          = true});
        }
        if (steps == 0) {
            // No synchronize here: the uploads above read mtp_host_, which is next written after the
            // round's synchronize.
            for (std::int32_t b = 0; b < batch; ++b) {
                Lane& lane = lanes_[lanes[b]];
                if (lane.mtp_live) { lane.mtp_cells = lane.state_tokens; }
            }
            return;
        }
        mtp_sampled_round_ = sampled;
        if (sampled) { stage_mtp_samplers(lanes, s); }
        run_mtp_draft_steps(batch, steps, sampled);
        CUDA_CHECK(cudaMemcpyAsync(mtp_host(mtp_io_.drafts), mtp_device(mtp_io_.drafts), 4ULL * steps * batch,
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(mtp_host(mtp_io_.logprobs), mtp_device(mtp_io_.logprobs), 4ULL * steps * batch,
                                   cudaMemcpyDeviceToHost, s));
        if (while_drafting) {
            device_.flush();
            while_drafting();
        }
        device_.synchronize();
        read_mtp_drafts(lanes, steps, unwritten, sampled);
    }

    // Sampled drafts when a row samples (greedy rows draft the argmax either way).
    [[nodiscard]] bool mtp_sampled(std::span<const std::uint32_t> lanes) const {
        return sampled_drafts_ && std::any_of(lanes.begin(), lanes.end(), [&](std::uint32_t lane) {
                   return lanes_[lane].sampling.temperature > 0.0F;
               });
    }

    // A draft round's io (everything before mtp_io_.drafts), staged and uploaded on `stream`: each
    // row's anchor (its last committed token), the cells and RoPE positions of its mtp_k_ steps from
    // its pending cell (clamped to its last mapped position), its state slot and table row.
    void stage_mtp_draft(std::span<const std::uint32_t> lanes, cudaStream_t stream) {
        const auto batch = static_cast<std::int32_t>(lanes.size());
        for (std::int32_t b = 0; b < batch; ++b) {
            const Lane& lane = lanes_[lanes[b]];
            const std::int32_t cell  = static_cast<std::int32_t>(lane.state_tokens) - 1;
            const std::int32_t limit = mtp_last_cell(lane);
            mtp_host(mtp_io_.ids)[b] = lane.history.back();
            for (std::int32_t j = 0; j < mtp_k_; ++j) {
                const std::int32_t at = std::min(cell + j, limit);
                mtp_host(mtp_io_.cells)[j * batch + b] = at;
                const std::size_t step = 3ULL * static_cast<std::size_t>(batch) * j;
                stage_rope_value(rope_of(lane.rope, static_cast<std::uint32_t>(at)),
                                 std::span<std::int32_t>(mtp_host(mtp_io_.rope_cells + step), 3ULL * batch), batch, b);
                stage_rope_value(block_start_rope(lane.rope, static_cast<std::uint32_t>(at), static_cast<std::uint32_t>(r_)),
                                 std::span<std::int32_t>(mtp_host(mtp_io_.block_rope_cells + step), 3ULL * batch), batch, b);
            }
            mtp_host(mtp_io_.slots)[b] = static_cast<std::int32_t>(lanes[b]);
            mtp_host(mtp_io_.rows)[b]  = static_cast<std::int32_t>(lanes[b]);
            mtp_cumulative_threshold_[static_cast<std::size_t>(b)] = lane.mtp_cumulative.threshold;
        }
        // The threshold the gated one-row graph compares with (read_mtp_drafts cuts with the same).
        std::memcpy(mtp_host(mtp_io_.cumulative), &mtp_cumulative_threshold_[0], sizeof(float));
        upload_pinned(mtp_device_.p, mtp_host_.data(), 4ULL * mtp_io_.drafts, stream);
    }

    // The lane's last mapped position: its shared prefix pages, then its private ones.
    [[nodiscard]] static std::int32_t mtp_last_cell(const Lane& lane) {
        return static_cast<std::int32_t>(lane.prefix.page_base + lane.pages.size()) * kPagedKVPageSize - 1;
    }

    // The rows' samplers for the draws inside the draft graph, uploaded on `stream` (verify uploads
    // its own, with grammar masks).
    void stage_mtp_samplers(std::span<const std::uint32_t> lanes, cudaStream_t stream) {
        auto* configs = reinterpret_cast<ops::SamplingConfig*>(mtp_host(mtp_io_.configs));
        for (std::size_t b = 0; b < lanes.size(); ++b) {
            configs[b]           = lanes_[lanes[b]].sampling;
            configs[b].mask      = {};
            configs[b].nonfinite = nullptr; // a draft's selection never fails the request
        }
        upload_pinned(mtp_device(mtp_io_.configs), configs, sizeof(ops::SamplingConfig) * lanes.size(), stream);
    }

    // A one-row argmax round verifies its drafts only up to its cut (with --draft-tokens auto the cumulative
    // cut, MtpCumulative; with a fixed length the first draft below --spec-min-p), so a draft step runs only
    // while every earlier draft cleared it: the device applies the host's cut (the same comparisons) and
    // skips the steps whose drafts it would discard. A skipped step leaves its draft and confidence words
    // stale, beyond the cut that reads them. (Design §19.3.27: at --spec-min-p, 34 % of the steps of
    // mixed-topic MTP 4 rounds, decode +2.2 %.)
    [[nodiscard]] bool mtp_draft_gated(std::int32_t batch, std::int32_t steps, bool sampled) const noexcept {
        return !sampled && batch == 1 && steps > 1 && (options_.auto_draft_length || std::isfinite(draft_min_logprob_));
    }

    DecodeGraph& mtp_draft_graph(std::int32_t batch, std::int32_t steps, bool sampled) {
        if (mtp_draft_gated(batch, steps, sampled)) { return mtp_gated_graphs_[static_cast<std::size_t>(steps - 1)]; }
        auto& graphs = sampled ? mtp_sampled_graphs_ : mtp_draft_graphs_;
        return graphs[static_cast<std::size_t>(batch - 1) * mtp_k_ + (steps - 1)];
    }

    // The draft round's `steps` chained steps over its staged io and chain columns.
    void run_mtp_draft_steps(std::int32_t batch, std::int32_t steps, bool sampled) {
        const cudaStream_t s = device_.stream;
        const Tensor slot_tensor(mtp_device(mtp_io_.slots), DType::I32, {batch});
        const Tensor row_tensor(mtp_device(mtp_io_.rows), DType::I32, {batch});
        Tensor chain(mtp_chain_.p, DType::BF16, {width_, batch});
        Tensor anchors(mtp_device(mtp_io_.ids), DType::I32, {batch});
        const auto* samplers = reinterpret_cast<const ops::SamplingConfig*>(mtp_device(mtp_io_.configs));
        const auto draft_step = [&](std::int32_t j) {
            Tensor drafts(mtp_device(mtp_io_.drafts + static_cast<std::size_t>(j) * batch), DType::I32, {batch});
            const std::size_t step = 3ULL * static_cast<std::size_t>(batch) * j;
            // Sampled drafts: step j's top-rank scratch and proposal planes.
            const auto plane = [&](std::size_t offset, std::int32_t rows, DType type) {
                return sampled
                           ? Tensor(mtp_device(offset + static_cast<std::size_t>(rows) * batch * j), type, {rows, batch})
                           : Tensor{};
            };
            constexpr std::int32_t C = ops::kSparseSpeculativeCandidates;
            forward_->run_mtp({.residuals    = chain,
                               .ids          = j == 0 ? anchors
                                                      : Tensor(mtp_device(mtp_io_.drafts + static_cast<std::size_t>(j - 1) * batch),
                                                               DType::I32, {batch}),
                               .positions    = Tensor(mtp_device(mtp_io_.cells + static_cast<std::size_t>(j) * batch),
                                                      DType::I32, {batch}),
                               .rope_positions   = Tensor(mtp_device(mtp_io_.rope_cells + step), DType::I32, {batch, 3}),
                               .block_start_rope = Tensor(mtp_device(mtp_io_.block_rope_cells + step), DType::I32, {batch, 3}),
                               .slots        = slot_tensor,
                               .table_rows   = row_tensor,
                               .batch        = batch,
                               .width        = 1,
                               .kv_only      = false,
                               .residual_out = chain,
                               .drafts       = drafts,
                               .draft_logprobs =
                                   Tensor(mtp_device(mtp_io_.logprobs + static_cast<std::size_t>(j) * batch), DType::FP32,
                                          {1, batch}),
                               .sample_configs   = sampled ? samplers : nullptr,
                               .draft_step       = static_cast<std::uint32_t>(j),
                               .sample_rows      = plane(mtp_io_.sample_rows, kSampledDraftRanks, DType::I32),
                               .sample_logprobs  = plane(mtp_io_.sample_logprobs, kSampledDraftRanks, DType::FP32),
                               .draft_candidates = plane(mtp_io_.draft_candidates, C, DType::I32),
                               .draft_q          = plane(mtp_io_.draft_q, C, DType::FP32)});
        };
        DecodeGraph& graph = mtp_draft_graph(batch, steps, sampled);
        if (mtp_draft_gated(batch, steps, sampled)) {
            replay_gated(graph, steps, draft_step,
                         [&](std::int32_t j, cudaGraphConditionalHandle next) {
                             if (options_.auto_draft_length) {
                                 // Steps 0..j's product against the round's threshold word (one row).
                                 launch_graph_gate_product(reinterpret_cast<const float*>(mtp_device(mtp_io_.logprobs)),
                                                           1, j, kMtpCumulativeBins,
                                                           reinterpret_cast<const float*>(mtp_device(mtp_io_.cumulative)),
                                                           next, s);
                                 return;
                             }
                             const auto* confidence =
                                 reinterpret_cast<const float*>(mtp_device(mtp_io_.logprobs + static_cast<std::size_t>(j)));
                             launch_graph_gate(confidence, draft_min_logprob_, next, s);
                         });
        } else {
            replay(graph, [&] {
                for (std::int32_t j = 0; j < steps; ++j) { draft_step(j); }
            });
        }
    }

    // A draft round's readout once its copies have landed: each row's first `steps` drafts and its
    // confident count, and with `unwritten` its pending cell written.
    void read_mtp_drafts(std::span<const std::uint32_t> lanes, std::int32_t steps, bool unwritten, bool sampled) {
        const auto batch = static_cast<std::int32_t>(lanes.size());
        for (std::int32_t b = 0; b < batch; ++b) {
            Lane& lane = lanes_[lanes[b]];
            if (unwritten && lane.mtp_live) { lane.mtp_cells = lane.state_tokens; }
            for (std::int32_t j = 0; j < steps; ++j) {
                mtp_drafts_[b][static_cast<std::size_t>(j)] = mtp_host(mtp_io_.drafts)[j * batch + b];
            }
            // The row's confident drafts: those before the first chain product below the staged
            // threshold with --draft-tokens auto, else before the first below draft_min_logprob_ (the
            // device gate's comparison for a gated one-row round); every sampled draft
            // with sampled-mtp-drafts (a sampled proposal's acceptance does not follow the head's top
            // probability, and its rounds were measured faster without a cut).
            mtp_confident_[b] = steps;
            if (sampled && lane.sampling.temperature > 0.0F) { continue; }
            const auto* logprobs = reinterpret_cast<const float*>(mtp_host(mtp_io_.logprobs));
            auto& kept           = mtp_draft_logprobs_[static_cast<std::size_t>(b)];
            for (std::int32_t j = 0; j < steps; ++j) { kept[static_cast<std::size_t>(j)] = logprobs[j * batch + b]; }
            for (std::int32_t j = 0; j < steps; ++j) {
                const bool cut = options_.auto_draft_length
                                     ? graph_gate_product(logprobs + b, batch, j, kMtpCumulativeBins) <
                                           mtp_cumulative_threshold_[static_cast<std::size_t>(b)]
                                     : logprobs[j * batch + b] < draft_min_logprob_;
                if (cut) {
                    mtp_confident_[b] = j;
                    break;
                }
            }
        }
    }

    // TEMPORARY (INFERNIX_TMP_CONTINUING_PROMPT, CF-01): the not-live column flags of a continuing turn's
    // prefill call, and how many such calls skipped the cache's scoring (reported at each release).
    std::vector<std::uint8_t> prompt_silent_;
    std::uint64_t continuing_calls_ = 0;

    // After a verification round's commit: the drafter's cells of the verified window from its
    // residuals and committed tokens (index-key tails advance by the committed count), and each
    // row's last committed residual becomes its pending cell, already written.
    void mtp_catch_up(std::span<const SequenceHandle> rows, std::int32_t W, const std::int32_t* commit) {
        const cudaStream_t s     = device_.stream;
        const auto batch         = static_cast<std::int32_t>(rows.size());
        const std::size_t column = 2ULL * width_;
        for (std::int32_t b = 0; b < batch; ++b) {
            const std::int32_t k = commit[b];
            for (std::int32_t j = 0; j < W; ++j) {
                mtp_host(mtp_io_.up_ids)[b * W + j] =
                    j < k ? pending_tokens_[static_cast<std::size_t>(b * W + j)] : pending_tokens_[static_cast<std::size_t>(b * W)];
            }
        }
        upload_pinned(mtp_device(mtp_io_.up_ids), mtp_host(mtp_io_.up_ids), 4ULL * batch * W, s);
        replay(mtp_catch_graph(batch, W), [&] { mtp_catch_body(batch, W); });
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

    // Samples the next token of each row from read_logits (row b = column b) into pending_tokens_.
    // `before_wait` runs on the host after the sampling work is queued, before its wait.
    void sample(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions,
                const std::function<void()>& before_wait = {}) {
        const cudaStream_t s = device_.stream;
        const auto batch     = static_cast<std::int32_t>(lanes.size());
        // The sampler reads the FP32 logits, or BF16 (design §16.5); a constraint masks them in place (the prompt
        // readout has read them first).
        Tensor logits = read_logits(batch);
        auto* configs = static_cast<ops::SamplingConfig*>(host_configs_.data());
        auto* sample_positions = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.positions);
        for (std::int32_t b = 0; b < batch; ++b) {
            configs[b]          = lanes_[lanes[b]].sampling;
            configs[b].mask     = grammar_mask(static_cast<std::size_t>(b), {});
            sample_positions[b] = positions[b];
        }
        upload_pinned(configs_.p, configs, sizeof(ops::SamplingConfig) * batch, s);
        upload_pinned(sample_pos_.p, sample_positions, 4ULL * batch, s);
        const bool constrained = stage_constraints(lanes, 1);
        if (constrained) { constrain(logits, nullptr); }
        Tensor out(sampled_.p, DType::I32, {batch});
        Tensor logical(sample_pos_.p, DType::I32, {batch});
        {
            auto scope = work_->scope();
            ops::sample(logits, out, token_domain_, static_cast<const ops::SamplingConfig*>(configs_.p), logical,
                        ops::kSamplePurposeDecode, *work_, s);
        }
        CUDA_CHECK(cudaMemcpyAsync(host_sampled_.data(), sampled_.p, 4ULL * batch, cudaMemcpyDeviceToHost, s));
        if (before_wait) { before_wait(); }
        device_.synchronize();
        check_expert_error();
        const auto* tokens = static_cast<const std::int32_t*>(host_sampled_.data());
        for (std::int32_t b = 0; b < batch; ++b) { pending_tokens_[b] = tokens[b]; }
        if (constrained) {
            for (std::int32_t b = 0; b < batch; ++b) {
                take_constraint_draws(lanes_[lanes[b]], static_cast<std::size_t>(b),
                                      std::span<const std::int32_t>(&pending_tokens_[b], 1));
            }
        }
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

    std::int32_t vocab_ = 0, token_domain_ = 0, chunk_ = 0, columns_ = 0;
    std::int32_t width_ = 0, span_ = 0, di_ = 0, r_ = 0;
    std::uint32_t kv_pages_ = 0;

    // ---------------------------------------------------------------- prefix cache (prefix_program.cpp)
    void create_prefix_cache();
    // A lane index meaning "no lane yet" (a quote before the Engine picks the destination).
    static constexpr std::uint32_t kNoLane = 0xFFFFFFFFU;
    // The root start of a request with a prefix cache: every page reserved, the cold call grid.
    PrefixSelection prefix_root(const qwen3_5::PreparedPromptData& prompt, const BasePlanImpl& base) {
        PrefixSelection root;
        root.need = binding_pages(base, static_cast<std::uint32_t>(prompt.token_ids.size()));
        root.plan = plan_prefill(prompt, 0, {}, base.reuse);
        return root;
    }
    std::optional<PrefixSelection> prefix_select(const qwen3_5::PreparedPromptData& prompt, const BasePlanImpl& base,
                                                 std::uint32_t lane);
    void pc_make_room(std::uint32_t pages);
    // Whether a fresh prompt reusing `reuse` tokens should wait for a prefilling sibling's snapshot
    // inside their shared prefix; plans that snapshot as a boundary tap of the sibling when so.
    bool prefix_await_sibling(const qwen3_5::PreparedPromptData& prompt, std::uint32_t reuse);
    // A restore batch prefetching a blocked head's blocks (none once it has landed).
    std::optional<prefix::RestoreTicket> prefetch_ticket_;
    [[nodiscard]] bool prefetch_landing() const noexcept {
        return prefetch_ticket_ && !prefix_->restore_events(*prefetch_ticket_).empty();
    }

public:
    std::optional<std::uint32_t> hybrid_prefetch(const RequestBasePlan& base);
    [[nodiscard]] std::uint32_t hybrid_prefetch_room() const noexcept;
    void hybrid_hold_queue(std::span<const RequestBasePlan* const> queue);
    [[nodiscard]] std::uint64_t hybrid_cache_epoch() const noexcept;
    // Without free cache resume a request resumes only from prefill state on the prefill-chunk grid.
    [[nodiscard]] std::uint32_t hybrid_resume_grid() const noexcept { return resume_grid(); }

private:
    void prefix_pin(const PrefixSelection& selection);
    void prefix_unpin(const PrefixSelection& selection) noexcept;
    void prefix_activate(Lane& lane, std::uint32_t index, const PrefixSelection& selection,
                         DeviceKVPageReservation& reservation, std::uint32_t pages);
    [[nodiscard]] std::uint32_t prefix_frontier(const Lane& lane, bool finishing) const noexcept;
    void prefix_publish(Lane& lane, std::uint32_t frontier);
    void prefix_capture(Lane& lane, std::uint32_t index, runtime::prefix_cache::SnapshotKind kind, bool hand_over_tail);
    // The capture of the lane's state at frontier F (the lane's published path reaches F's anchor);
    // `mtp_written`: the MTP cell F - 1 is final.
    [[nodiscard]] prefix::PrefixCache::Capture prefix_capture_at(Lane& lane, std::uint32_t F,
                                                                 runtime::prefix_cache::SnapshotKind kind,
                                                                 bool hand_over_tail, bool mtp_written);
    // Records a capture the cache accepted as the lane's latest.
    void prefix_captured(Lane& lane, std::uint64_t id, std::uint32_t F);
    // Removes the lane's taps realized at the call boundary B before calls[next_call]; returns whether
    // any was, and sets `boundary` when one is a boundary tap.
    bool prefix_take_taps(Lane& lane, std::uint32_t B, std::size_t next_call, bool& boundary);
    void prefix_after_prefill_call(Lane& lane, std::uint32_t index, bool last);
    void prefix_finish(Lane& lane, std::uint32_t index);
    // Releases the lane's pins; a lane marked `withdraw` withdraws what it published (prefix_withdraw).
    void prefix_release(Lane& lane) noexcept;
    void prefix_withdraw(Lane& lane) noexcept;
    // Entries withdrawn after a non-finite failure that another lane or a transfer still pinned
    // (hybrid-prefix-cache-spec §7.8): every later lane release retries them.
    runtime::prefix_cache::Withdrawal prefix_withdrawal_;
    // The cumulative Vision key of every item of the prompt: the key of each block after it.
    [[nodiscard]] static std::uint64_t prefix_trailing_extra(const qwen3_5::PreparedPromptData& prompt);
    // Every Vision item's token span: no tap and no resume lies strictly inside one.
    [[nodiscard]] static std::vector<runtime::prefix_cache::TapExclusion>
    prefix_exclusions(const qwen3_5::PreparedPromptData& prompt);
    [[nodiscard]] std::optional<runtime::prefix_cache::SnapshotRef> prefix_resident(const Lane& lane) const;
    [[nodiscard]] std::array<std::span<const cudaEvent_t>, 2> prefix_waits(const Lane& lane) const noexcept;
    void mtp_flush_cell(Lane& lane, std::uint32_t index);

public:
    // Persistence of the Host tier (--prefix-cache-file).
    PrefixCachePersistence attach_prefix_cache_file(const std::filesystem::path& path, std::string fingerprint,
                                                    const StartupObserver& observer);
    void save_prefix_cache_for_shutdown() noexcept;
    [[nodiscard]] std::optional<PrefixCachePersistence> prefix_shutdown_save() const { return prefix_shutdown_save_; }
    [[nodiscard]] std::optional<PrefixCachePersistence> save_prefix_cache_now(const CancellationView& abandoned);

private:
    std::filesystem::path prefix_file_;
    std::string prefix_fingerprint_;
    std::optional<PrefixCachePersistence> prefix_shutdown_save_;
    std::uint64_t prefix_saved_writes_ = 0; // host_write_bytes at the last save or load
    std::uint64_t prefix_saves_        = 0; // periodic saves of the attached file that succeeded

public:
    [[nodiscard]] HybridPrefixCacheStats prefix_stats() const noexcept;

private:
    // The prompt's prefill calls from `frontier`, and with `taps` the prefix taps they realize. A
    // nonzero `branch` (past the frontier) adds an exact boundary tap there: where the prompt leaves
    // a cached path (prefix_select).
    [[nodiscard]] prefix::CallPlan plan_prefill(const qwen3_5::PreparedPromptData& prompt, std::uint32_t frontier,
                                                std::span<const std::uint32_t> existing, bool taps,
                                                std::uint32_t branch = 0) const;
    // A branch tap (prefix_select) needs the prompt to share at least this many cached tokens past
    // its resume frontier: what a later prompt diverging further along re-prefills without it.
    static constexpr std::uint32_t kBranchTapMinTokens = 1024;
    DeviceBuffer state_backing_, ple_backing_, tails_backing_, kv_backing_, staging_;
    // The vq2/k4v2 exact window (design §19.3.15), every KV layer and lane.
    DeviceBuffer window_backing_;
    prefix::KvWindowGeometry window_geometry_;

    // ---- the host-to-device link (design §19.3.12: bandwidth-dependent choices follow the machine) ----
    // The rate at which the prefill and prefix cost constants were fitted (RTX 5090 at PCIe 5.0 x8).
    static constexpr double kReferenceLinkBytesPerSecond = 27.5e9;
    double link_bytes_per_second_ = kReferenceLinkBytesPerSecond;
    // Measures the link once, by copying expert records from the pinned banks (the SSD tier's pinned
    // slots when the banks stay in the artifact) into the staging slots before any call uses them
    // (or takes ProgramOptions::link_bytes_per_second), and
    // rescales the link-bound costs: the prefix cost's restore rate and the walk-span cost (one pass
    // of the experts over the link).
    void measure_link() {
        double rate = options_.link_bytes_per_second;
        const void* source = parameters_.layers.front().moe.bank->planes.records;
        if (source == nullptr && tier_) { source = tier_->slot_bytes(0); }
        if (rate <= 0.0 && source != nullptr && staging_.p != nullptr && plan_.staging > 0) {
            rate = measure_h2d_bytes_per_second(source, staging_.p, plan_.staging, 3);
        }
        if (rate <= 0.0) { return; } // no frames to probe with: keep the reference rate
        link_bytes_per_second_ = rate;
        options_.prefix_cost.h2d_bytes_per_second = rate;
        options_.prefix_span_seconds *= kReferenceLinkBytesPerSecond / rate;
        if (prefix_) { prefix_->index().set_cost(options_.prefix_cost); }
        char text[256];
        std::snprintf(text, sizeof(text),
                      "host-to-device link: %.1f GB/s%s; decode promotes every %llu tokens; prefill CPU split up to "
                      "%d columns",
                      rate / 1e9, options_.link_bytes_per_second > 0.0 ? " (set)" : " measured",
                      static_cast<unsigned long long>(promotion_interval()),
                      options_.prefill_cpu_split ? split_columns() : 0);
        diagnostic(text, DiagnosticLevel::Debug);
        std::snprintf(text, sizeof(text), "PCIe link to the GPU: %.1f GB/s%s", rate / 1e9,
                      options_.link_bytes_per_second > 0.0 ? " (fixed)" : " measured");
        diagnostic(text);
    }

    // ---- the CPU expert team's rates (design §19.3.12) ----
    // For cpu_expert_workers workers, measured at startup while the link copies records (the staging's
    // DRAM reads compete with the CPU's): experts per second on one-column jobs (memory-bound, as
    // decode's misses are) and expert columns per second on 8-column jobs (the narrow route's widest).
    // The decode miss divisor and the prefill CPU split both use them. The SSD tier measures them on
    // its RAM copies once filled; until then (and without any host record) the reference rates stand
    // (ProgramOptions::cpu_split_*_rate, the i9-13900K at 6 workers on NVFP4 records).
    double cpu_expert_rate_ = 0.0; // 0 until measured
    double cpu_column_rate_ = 0.0;
    void measure_cpu_rate(std::uint64_t record_stride);
    [[nodiscard]] double cpu_expert_rate() const noexcept {
        return cpu_expert_rate_ > 0.0 ? cpu_expert_rate_ : options_.cpu_split_expert_rate;
    }
    [[nodiscard]] double cpu_column_rate() const noexcept {
        return cpu_column_rate_ > 0.0 ? cpu_column_rate_ : options_.cpu_split_column_rate;
    }

    // The share of a decode call's misses the PCIe stage keeps, misses / d, from C t (C experts per
    // second on the CPU, t seconds per record on the link): d = floor(1.65 + C t). The balance point
    // of the two parts is d = 1 + C t; the measured fastest divisors (2026-10-09, x8) put the
    // switches between them: C t = 1.5-1.7 at 6 workers (d = 3; d = 2 loses 9 % MTP), 1.0 at 3 and
    // 0.7-0.8 at 2 (d = 2; d = 3 loses 4-12 %), so d steps from 2 to 3 at C t = 1.35, midway, and
    // the startup measurement's noise (± 7 %) stays clear of it; x16 gives 2. A whole divisor, as a
    // fractional share flips calls' splits at floor(M share) between startups (3 workers: 100.8
    // against 107.5 tok/s). A positive cpu_pcie_divisor fixes d (one_column_divisor below
    // overrides d in a GGUF import's one-column calls).
    // The CPU expert team's workers: a GGUF import's banks take their own count (ProgramOptions), and
    // their CPUs (both from INFERNIX_TMP_CPU_PLACEMENT when set; the engine thread stays unpinned).
    runtime::expert_offload::CpuPlacement placement_;
    [[nodiscard]] std::uint32_t cpu_workers() const noexcept { return placement_.workers; }

    // INFERNIX_TMP_LINK_SHARE (Strata's L / (L + 1.85 R), R = C x the record's bytes): the link's share
    // of a call's misses in 256ths instead of misses / d (0: the divisor rule; a fixed d wins).
    [[nodiscard]] int pcie_share() const noexcept {
        if (options_.cpu_pcie_divisor > 0) { return 0; }
        return runtime::expert_offload::link_share_256(
            temporary_switches::offload(), link_bytes_per_second_, cpu_expert_rate(),
            static_cast<double>(parameters_.layers.front().moe.bank->planes.record_stride));
    }

    [[nodiscard]] int pcie_divisor() const noexcept {
        if (options_.cpu_pcie_divisor > 0) { return options_.cpu_pcie_divisor; }
        const double record_seconds =
            static_cast<double>(parameters_.layers.front().moe.bank->planes.record_stride) / link_bytes_per_second_;
        return std::max(1, static_cast<int>(std::floor(1.65 + cpu_expert_rate() * record_seconds)));
    }

    // Without speculation, a GGUF import's one-column calls (plain decode) take the round's divisor
    // by context (plain_divisor; 1 until the first round sets it). 0 (pcie_divisor) for other banks,
    // with speculation and when cpu_pcie_divisor fixes d: MTP keeps d everywhere, as d = 1 in its
    // rounds' one-column calls lost 216.2 -> 212.9 tok/s (2026-10-13, IQ3_S, 16 workers, derived d = 2).
    // The rule on every bank lost on Dense8 (plain decode 0.731x / 0.933x, removed 2026-10-28).
    [[nodiscard]] int one_column_divisor() const noexcept {
        const bool speculative = mtp_k_ > 0 || options_.ngram_draft_tokens > 0;
        return c_.ggml_weights && !speculative && options_.cpu_pcie_divisor == 0 ? plain_divisor(0) : 0;
    }

    // Plain decode's one-column divisor (runtime/expert_offload/placement.h plain_divisor): every
    // host-memory miss on the link (d = 1) below
    // kPlainDivisorContext context tokens, half of them on the CPU (d = 2) from there; each round
    // writes it into the device io word its graph reads. Measured 2026-10-16 (Q2 IQ3_S artifact, RTX
    // 5090 on PCIe 5.0 x8, plain decode, tok/s at d = 1 / d = 2; prompt + output tokens): 512 + 256
    // 151.5 / 139-141, 1K + 256 152.4 / 143.8, 2K + 256 124.9 / 129.9, 4K + 512 99.5 / 114.4, 32K +
    // 256 119.5 / 126.3. This rule: 151.4 at 512 + 256, 115.2 at 4K + 512, and 114.4 at 1K + 1,024
    // (d = 1 105.5, d = 2 121.0: a long output from a short prompt would gain from d = 2 before the
    // switch). A divisor following the hit rate over eight-token windows trailed the better fixed
    // divisor at every length (its windows swing around any threshold: 47 switches in a 4K run).
    [[nodiscard]] static std::int32_t plain_divisor(std::int32_t position) noexcept {
        return runtime::expert_offload::plain_divisor(position);
    }

    // The widest call the prefill CPU split takes (ProgramOptions::cpu_split_columns on the reference
    // link, inversely to the measured one; at most the CPU channel's call width): runtime::expert_offload::split_width.
    [[nodiscard]] std::int32_t split_columns() const noexcept {
        return runtime::expert_offload::split_width(options_.cpu_split_columns, kReferenceLinkBytesPerSecond,
                                                    link_bytes_per_second_);
    }

public:
    [[nodiscard]] double link_bytes_per_second() const noexcept { return link_bytes_per_second_; }

private:
    // ---- elastic KV (design §19.3.11) ----
    static constexpr std::size_t kKvChunkBytes   = 2ULL << 20;
    // Backed at startup: 4,096 tokens (one growth unit; candidates §22 VM-08, 2026-10-27), not 32,768. The 28K
    // tokens less backed are ~400 MiB of frames on Dense8 while contexts stay short (the VRAM defaults measured
    // Dense8 MTP decode +8.0 / +3.4 / +2.4 % together); longer prompts grow the pool by kKvGrowPages at a time.
    static constexpr std::uint32_t kKvBaseTokens = 4096;
    static constexpr std::uint32_t kKvGrowPages  = 64;    // growth unit (4,096 tokens)
    static constexpr std::chrono::seconds kKvIdleShrink{60};
    // The last admission, release or idle shrink (steady-clock ticks), and whether the pool backs more
    // than its base: the monitor thread reads both to wake an idle engine for the shrink.
    std::atomic<std::int64_t> kv_busy_at_{0};
    std::atomic<bool> kv_grown_{false};
    const bool kv_shrink_busy_ = temporary_switches::flag("INFERNIX_TMP_KV_SHRINK_BUSY");
    void kv_busy() noexcept { kv_busy_at_.store(std::chrono::steady_clock::now().time_since_epoch().count()); }
    bool kv_idle_due() const noexcept {
        return kv_grown_.load() && std::chrono::steady_clock::now().time_since_epoch().count() - kv_busy_at_.load() >=
                                       std::chrono::duration_cast<std::chrono::steady_clock::duration>(kKvIdleShrink).count();
    }
    std::unique_ptr<VmmRange> kv_vmm_;
    DeviceSpan kv_span_{};

    // Maps (and zeroes) the chunks pages [0, pages) need; false when the device has no memory.
    bool map_kv(std::uint32_t pages) {
        const auto need = kv_chunks(plan_, pages);
        for (std::size_t c = 0; c < need.size(); ++c) {
            if (!need[c] || kv_vmm_->mapped(c)) { continue; }
            if (!kv_vmm_->map(c)) { return false; }
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(kv_vmm_->base()) + c * kKvChunkBytes, 0, kKvChunkBytes,
                                       device_.stream));
        }
        return true;
    }

    // Raises the pool's backed pages so `need` pages are available, taking the memory from the expert
    // frames, as many as the growth needs: a fixed pool at the capacity would take them all at startup,
    // and growing in steps reaches the same pool. False when it cannot: a lease holds the frames, the
    // frames do not cover it, or the pool is at its capacity.
    bool grow_kv(std::uint32_t need) {
        if (!kv_vmm_ || pool_->available_pages() >= need) { return pool_->available_pages() >= need; }
        const std::uint32_t backed = pool_->backed_pages();
        std::uint32_t target       = backed + (need - pool_->available_pages());
        target = std::min(pool_->capacity_pages(), (target + kKvGrowPages - 1) / kKvGrowPages * kKvGrowPages);
        if (target - backed < need - pool_->available_pages()) { return false; }
        const std::lock_guard<std::mutex> lock(vram_mutex_);
        const std::size_t added    = kv_mapped_bytes(plan_, target) - kv_vmm_->mapped_bytes();
        const std::uint64_t stride = residency_->frame_bytes();
        const std::uint32_t frames = residency_->frames();
        // Memory the sizing function would give the frames anyway comes first; the rest is taken from
        // the frames, one chunk per pool over so the pools unmap at least that much.
        const std::uint32_t spare =
            control_ ? std::max<std::uint32_t>(control_->target(vram_->query(), pool_now_), frames) - frames : 0;
        const std::uint64_t spare_bytes = static_cast<std::uint64_t>(spare) * stride;
        std::uint32_t give              = 0;
        if (added > spare_bytes) {
            give = static_cast<std::uint32_t>((added - spare_bytes + residency_->pools() * ExpertResidency::kChunkBytes + stride - 1) / stride);
        }
        if (give > frames) { return false; }
        device_.synchronize();
        if (give != 0) {
            const auto resized = residency_->resize(frames - give, device_.stream, *vram_);
            if (resized.frames > frames - give) { return false; } // a lease holds the frames
        }
        const std::size_t mapped_before = kv_vmm_->mapped_bytes();
        if (!map_kv(target)) {
            // The chunks mapped before the failure stay with the region (the next growth uses them) and count as
            // fixed bytes; the frames get back only what was not mapped (as Qwen3.5's elastic_kv.cpp), so frames and
            // KV never claim the same memory.
            const std::size_t kept = kv_vmm_->mapped_bytes() - mapped_before;
            if (control_ && kept != 0) { control_->account_fixed(static_cast<std::int64_t>(kept)); }
            const auto held = static_cast<std::uint32_t>(std::min<std::uint64_t>(give, (kept + stride - 1) / stride));
            if (give > held) { (void)residency_->resize(frames - held, device_.stream, *vram_); }
            frames_now_ = residency_->frames();
            pool_now_   = residency_->pool_bytes();
            return false;
        }
        pool_->set_backed_pages(target);
        kv_grown_.store(true);
        if (control_) { control_->account_fixed(static_cast<std::int64_t>(added)); }
        frames_now_ = residency_->frames();
        pool_now_   = residency_->pool_bytes();
        diagnostic("KV pool grew to " + std::to_string(target) + " pages (" +
                   std::to_string(kv_vmm_->mapped_bytes() >> 20) + " MiB); expert cache " + std::to_string(frames) +
                   " -> " + std::to_string(residency_->frames()) + " frames");
        return true;
    }

    // Lowers the pool's backed pages to the free top (not below the base), and returns the memory to the
    // expert frames. Between rounds: nothing reads free pages. `evict` (every lane free for
    // kKvIdleShrink): the cache's idle device blocks that have host copies are released first, in
    // the index's order (nothing is lost: a later match restores them from the Host tier), so a long
    // conversation that has ended gives its memory back to the frames. The delay keeps a follow-up
    // turn from paying a restore and an expert reload for memory the idle gap never used.
    void shrink_kv(bool evict = false) {
        if (!kv_vmm_ || pool_->backed_pages() <= plan_.kv_base_pages) { return; }
        if (evict && prefix_) {
            runtime::prefix_cache::PrefixCacheIndex& index = prefix_->index();
            for (std::uint32_t budget = pool_->backed_pages() - plan_.kv_base_pages;
                 budget != 0 && !pool_->can_back(plan_.kv_base_pages) && index.evict_backed_device_blocks(1) != 0; --budget) {}
        }
        // Down in growth steps, the last one to the base (Qwen3.5's elastic_kv.cpp: a pool grown to a capacity
        // off the step grid stopped above the base when a page below the base was in use, and then woke the
        // engine every kKvIdleShrink for a shrink that did nothing).
        std::uint32_t target = pool_->backed_pages();
        while (target > plan_.kv_base_pages) {
            const std::uint32_t next = std::max(plan_.kv_base_pages, target - std::min(target, kKvGrowPages));
            if (!pool_->can_back(next)) { break; }
            target = next;
        }
        if (target == pool_->backed_pages()) { return; }
        const std::lock_guard<std::mutex> lock(vram_mutex_);
        device_.synchronize();
        const std::size_t before = kv_vmm_->mapped_bytes();
        pool_->set_backed_pages(target);
        const auto need = kv_chunks(plan_, target);
        for (std::size_t c = 0; c < need.size(); ++c) {
            if (!need[c] && kv_vmm_->mapped(c)) { kv_vmm_->unmap(c); }
        }
        kv_grown_.store(target > plan_.kv_base_pages);
        const std::size_t freed   = before - kv_vmm_->mapped_bytes();
        const std::uint64_t stride = residency_->frame_bytes();
        const std::uint32_t frames = residency_->frames();
        (void)residency_->resize(frames + static_cast<std::uint32_t>(freed / stride), device_.stream, *vram_);
        if (control_) { control_->account_fixed(-static_cast<std::int64_t>(freed)); }
        frames_now_ = residency_->frames();
        pool_now_   = residency_->pool_bytes();
        diagnostic("KV pool shrank to " + std::to_string(target) + " pages; expert cache " + std::to_string(frames) +
                   " -> " + std::to_string(residency_->frames()) + " frames");
    }
    DeviceBuffer io_device_, logits32_, logits16_, sampled_, sample_pos_, configs_, token_counts_;

    // The n-gram table the volume holds: the artifact's, or another volume's (ProgramOptions::ngram_table).
    [[nodiscard]] const NgramTableConfig& ngram_table() const noexcept {
        return options_.ngram_table ? *options_.ngram_table : c_.ple.table;
    }

    // ---- causal scoring (score.cpp) ----
    // While a scored prompt prefills, each call's final residuals land in score_residuals_ at its
    // first position (score_column_: forward_call's; a layer walk's chunks carry their own).
    DeviceBuffer score_residuals_, score_logits_, score_logits16_, score_io_;
    PinnedHostBuffer score_host_{1};
    bool scoring_                = false;
    std::int32_t score_column_   = 0;
    // Columns [begin, begin + columns) of score_residuals_.
    [[nodiscard]] Tensor score_residuals(std::int32_t begin, std::int32_t columns) const {
        return Tensor(static_cast<std::byte*>(score_residuals_.p) + 2ULL * width_ * static_cast<std::size_t>(begin),
                      DType::BF16, {width_, columns});
    }

public:
    ScoreResult causal_score(PreparedPrompt&& prompt, std::uint32_t first_target, const ScoreOptions& options);

private:
    // ---- /v1/decide (constraint.cpp) ----
    // The constraint controls: each lane's kMaximumConstraintSets sets of up to
    // ops::kTokenConstraintChoices tokens and their counts, then one descriptor, one probability
    // record, one observe list and one observation record per round column (lanes x max width).
    static std::size_t constraint_bytes(std::int32_t lanes, std::int32_t width) noexcept {
        const auto sets    = static_cast<std::size_t>(lanes) * kMaximumConstraintSets;
        const auto columns = static_cast<std::size_t>(lanes) * static_cast<std::size_t>(width);
        return 4ULL * (sets * ops::kTokenConstraintChoices + sets +
                       columns * (1 + ops::kTokenConstraintRecord + ops::kTokenObserveTokens +
                                  ops::kTokenObserveRecord));
    }
    [[nodiscard]] std::shared_ptr<const ConstraintPlan> compile_constraint(const TokenConstraint& constraint);
    // Writes the round's descriptors (`width` columns per row, rows in `lanes` order) and the
    // constrained lanes' sets; false (and nothing written) when no row is constrained.
    bool stage_constraints(std::span<const std::uint32_t> lanes, std::int32_t width);
    // The word offset of the observe lists in constraint_ (the observation records follow them).
    [[nodiscard]] std::size_t observe_offset() const noexcept;
    // Masks the staged round's columns of `logits` [vocab, columns] and rewrites `argmax` for them.
    void constrain(Tensor& logits, Tensor* argmax);
    // After the round synchronized: the draws of `tokens`, chosen by consecutive columns from
    // `first_column` for the lane's next output steps (none for an unconstrained lane).
    void take_constraint_draws(Lane& lane, std::size_t first_column, std::span<const std::int32_t> tokens);
    // Moves the first `accepted` draws of the pending round into the lane's trace.
    static void commit_constraint_draws(Lane& lane, std::size_t accepted);
    // The readout of the lane's prompt from its last prefill call's logits (column 0).
    [[nodiscard]] PromptReadout read_prompt_frontier(const Lane& lane);
    // ---- Constrained decoding (GBNF, JSON, JSON Schema, choice, regex, tool calls) ----
    // The Engine's grammar matchers fill packed vocabulary masks for a row's positions 0..drafts
    // (TokenMaskProvider); the sampler and draft acceptance read them through the row's sampling
    // config. Sampling and acceptance run on the host-launched path after each forward, where every
    // row's drafts are known, so no graph split is needed.
    static std::size_t grammar_mask_bytes(std::int32_t token_domain, std::int32_t lanes, std::int32_t width) noexcept {
        return 4ULL * ((static_cast<std::size_t>(token_domain) + 31) / 32) * static_cast<std::size_t>(width) *
               static_cast<std::size_t>(lanes);
    }
    // Row `row`'s masks for positions 0..drafts.size(), uploaded; an empty mask when the row is
    // unconstrained. Records which positions are dead ends (no legal token).
    ops::SamplingMask grammar_mask(std::size_t row, std::span<const TokenId> drafts) {
        grammar_dead_[row] = 0;
        if (grammar_ == nullptr || !grammar_->constrained(row)) { return {}; }
        if (drafts.size() + 1 > static_cast<std::size_t>(max_width_)) {
            throw std::logic_error("Qwen4Exp: a constrained row verifies more positions than its mask holds");
        }
        const std::size_t words  = (static_cast<std::size_t>(token_domain_) + 31) / 32;
        const std::size_t offset = row * static_cast<std::size_t>(max_width_) * words;
        std::span<std::uint32_t> host(static_cast<std::uint32_t*>(grammar_host_.data()) + offset,
                                      (drafts.size() + 1) * words);
        grammar_dead_[row] = grammar_->fill(row, drafts, host);
        auto* device_words = static_cast<std::uint32_t*>(grammar_masks_.p) + offset;
        // Read through the host mapping: a copy-engine copy would wait behind expert promotions.
        upload_pinned(device_words, host.data(), host.size_bytes(), device_.stream);
        grammar_->uploaded(row, host.size_bytes());
        return {device_words, static_cast<std::int32_t>(words)};
    }
    // After the round's synchronize. A row fails when its sampler found no finite logit (the lane's
    // flag, cleared here for its next round; its tokens are placeholders), or when the tokens it
    // emits reach a dead-end position (one mask position per token).
    void mark_row_failures(PendingBatch& pending, std::span<const std::int32_t> counts) noexcept {
        const auto rows = ContractAccess::rows(pending);
        for (std::size_t row = 0; row < rows.size(); ++row) {
            const auto count = counts.empty() ? 1 : counts[row];
            const std::uint64_t reached = count >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << count) - 1U;
            std::int32_t* const nonfinite = nonfinite_flag(ContractAccess::lane(rows[row]));
            const bool nonfinite_row      = std::exchange(*nonfinite, 0) != 0;
            ContractAccess::set_row_failure(pending, row,
                                            nonfinite_row || runtime::testing::take_nonfinite_fault()
                                                ? runtime::PendingRowFailure::NonFiniteLogits
                                            : (grammar_dead_[row] & reached) != 0
                                                ? runtime::PendingRowFailure::ConstraintDeadEnd
                                                : runtime::PendingRowFailure::None);
        }
    }
    // Per lane: the sampler's non-finite flag (ops::SamplingConfig::nonfinite), written by the
    // round's kernels through the host mapping.
    PinnedHostBuffer nonfinite_flags_{sizeof(std::int32_t) * kMaximumConcurrency};
    [[nodiscard]] std::int32_t* nonfinite_flag(std::uint32_t lane) const noexcept {
        return static_cast<std::int32_t*>(nonfinite_flags_.data()) + lane;
    }
    // Set for the duration of one decode or prefill call.
    struct GrammarScope {
        ProgramImpl& impl;
        GrammarScope(ProgramImpl& owner, runtime::TokenMaskProvider* masks) : impl(owner) { impl.grammar_ = masks; }
        ~GrammarScope() { impl.grammar_ = nullptr; }
    };
    DeviceBuffer grammar_masks_;
    PinnedHostBuffer grammar_host_{1};
    runtime::TokenMaskProvider* grammar_ = nullptr;
    std::array<std::uint64_t, kMaximumConcurrency> grammar_dead_{};

    DeviceBuffer constraint_;
    std::vector<std::int32_t> constraint_host_;
    std::int32_t constraint_columns_ = 0; // the staged round's columns
    std::array<std::uint64_t, kMaximumConcurrency> constraint_serials_{};
    std::uint64_t next_constraint_serial_ = 1;
    PinnedHostBuffer io_host_{1}, host_sampled_{1}, host_configs_{1};
    PinnedHostBuffer penalty_host_{1};    // a lane's penalty counts, staged for upload_pinned
    PinnedHostBuffer constraint_pinned_{1}; // constraint_'s image: the staged parts, for upload_pinned
    IoLayout io_layout_;
    std::unique_ptr<LinearAttentionStatePool> gdn_;
    prefix::KvPageGeometry kv_geometry_;
    std::unique_ptr<DeviceKVPagePool> pool_;
    std::unique_ptr<KVExecutionTablePool> tables_;
    std::size_t work_capacity_ = 0;
    std::unique_ptr<WorkspaceArena> work_;
    // SSD tier mode (design §19.3.7): the fetch channel and the tier outlive everything that reads
    // their memory (the residency's promotion copies, the stream, the CPU service), declared later.
    // With an expert stripe the tier reads through this store, the Model's with each record's tail
    // in the stripe file.
    std::optional<ExpertStore> striped_store_;
    std::unique_ptr<ops::offloaded_moe::FetchChannel> fetch_channel_;
    std::unique_ptr<HostExpertTier> tier_;
    PinnedHostBuffer expert_error_{1}; // mapped u32: ops::MoeExpertSource::error of every call
    // TEMPORARY (INFERNIX_TMP_REPLAY_BENCH, candidates §8.5 CF-03): decode rounds' device time and the
    // CPU waits' nanoseconds (ops::MoeExpertSource::wait_ns), reported at each lane's release.
    std::unique_ptr<RoundClock> replay_clock_;
    DeviceBuffer replay_wait_;
    bool clear_prefix_on_release_ = false;
    std::unique_ptr<ExpertResidency> residency_;
    std::unique_ptr<ExpertStream> expert_stream_;
    // The drafter pool (candidates §22 VM-01): the MTP drafter's pooled experts (their memory is
    // declared first, so the pool, whose copies write it, goes first) and when it last reported.
    DeviceBuffer drafter_pool_memory_;
    std::unique_ptr<runtime::expert_offload::DrafterExpertPool> drafter_pool_;
    Clock::time_point drafter_reported_{};
    // The stream's leases: the ring, the wide arena and a walk's area (each a run of one pool).
    ExpertResidency::FrameLease stream_lease_, stream_wide_lease_, stream_walk_lease_;
    std::size_t stream_ring_bytes_ = 0, stream_wide_bytes_ = 0, stream_walk_bytes_ = 0;
    std::size_t wide_work_ = 0;              // the plan's wide arena bytes (0: none)
    ExpertResidency::FrameLease work_lease_; // the wide arena lent alone (no stream lease)
    DeviceBuffer wide_fallback_;             // the wide arena when nothing could be lent
    std::unique_ptr<WorkspaceArena> wide_arena_; // the arena swapped with *work_ around a wide call
    ExpertStream::Stats stream_at_lease_;
    execution::Forward::SplitStats split_at_lease_;

    // ---------------------------------------------------------------- layer walk (prefill_walk.cpp)
    // Design §19.3.8 F4: a span of consecutive streamed calls of one prompt runs layer-major (every
    // decoder layer over all of the span's chunks, then the next), so each layer's non-resident
    // experts are streamed once per span instead of once per call. The chunk grid, and with it
    // every output bit, is the chunk-major one. The span's residual stream [S*H, tokens] and each
    // chunk's staged io image live in the stream lease after the ring.
    struct LayerWalk {
        bool active = false;
        std::uint32_t lane = 0;
        std::size_t first_call = 0, end_call = 0;
        std::vector<std::int32_t> begins, widths, offsets; // per chunk: position, width, residual column
        std::uint32_t next_layer      = 0;                // the next decoder layer to enqueue
        std::uint32_t layers_per_step = 1;
        std::size_t io_stride = 0, residual_bytes = 0;
        std::int32_t host_slot = 0; // the lane's state slot as the Forward reads it on the host
        std::vector<std::optional<execution::MtpChunk>> mtp;
        std::vector<std::optional<execution::VisionInput>> vision;
        std::vector<std::vector<execution::VisionInput>> mtp_vision;
        std::uint64_t steps = 0;
        // MoE groups: consecutive chunks [first, end) whose MoE runs as one call over their columns
        // (at most kWalkMoeColumns, and what the workspace holds), its input in the area at moe_offset.
        std::vector<std::pair<std::size_t, std::size_t>> groups;
        std::size_t moe_offset = 0;
        // Prefix taps realized at a chunk end inside the span (walk_plan_taps): each chunk's staged
        // capture (0: none), copied group by group as the walk enqueues that layer of the chunk and
        // committed after the span; the next chunk's layers wait on its events.
        struct Tap {
            std::uint32_t frontier = 0;
            runtime::prefix_cache::SnapshotKind kind = runtime::prefix_cache::SnapshotKind::Tap;
            std::uint64_t staged = 0;
        };
        std::vector<Tap> taps; // per chunk
    } walk_;
    PinnedHostBuffer walk_host_{1}; // every chunk's io image, uploaded once per span
    // The next span's n-gram rows, read into walk_host_'s chunk slots while the current span's steps
    // run on the device (prefill_walk.cpp): chunk k is call first_call + k, its rows stand at slot k's
    // n-gram columns, and key is a hash of the tokens its rows hash.
    struct WalkPrefetch {
        bool valid = false;
        std::uint32_t lane = 0;
        std::size_t first_call = 0;
        std::vector<std::int32_t> begins, widths;
        std::vector<std::uint64_t> keys;
        bool done = false; // no further chunk can join the next span
    } walk_prefetch_;
    void walk_prefetch_rows(Lane& lane, std::uint32_t index, std::size_t chunks);
    [[nodiscard]] bool walk_prefetched(std::uint32_t index, std::size_t first_call, std::size_t c, std::int32_t begin,
                                       std::int32_t width);
    [[nodiscard]] std::uint64_t window_key() const noexcept;
    std::uint64_t walk_spans_ = 0, walk_calls_ = 0;
    std::uint64_t walk_prefetched_chunks_ = 0, prefetched_at_lease_ = 0; // chunks whose rows were prefetched
    // Temporary switch INFERNIX_TMP_DECODE_SHARE_WALK_TOKENS (0, unset: off): while another lane
    // decodes, a walk step enqueues about this many tokens of one-call work (layers * span tokens /
    // L, at least one layer) instead of one call's, so the Engine's decode share interleaves with the
    // span at finer steps. Step boundaries change only when work is enqueued, never a bit.
    std::uint32_t walk_step_tokens_ = 0;
    [[nodiscard]] static std::uint32_t walk_step_tokens_from_environment();
    [[nodiscard]] std::uint32_t walk_step_layers(std::uint32_t index) const noexcept;
    static constexpr std::uint32_t kWalkMaxTokens = 65536;
    // A walk's MoE calls take up to this many columns (several chunks'): at 16,384 tokens the
    // experts run 33-50 % more effective TOPS than at 4,096, and at 32,768 a further 7-14 %
    // (infernix_offloaded_moe_ggml_test --bench; IQ3_S prefill +1.4 % at 32K tokens and +2.5 % at
    // 64K end to end, design §19.3.29).
    static constexpr std::int32_t kWalkMoeColumns = 32768;

    [[nodiscard]] std::size_t walk_bytes(std::uint32_t tokens, std::size_t chunks) const noexcept;
    std::size_t walk_span_end(const Lane& lane, std::uint32_t index);
    void walk_begin(Lane& lane, std::uint32_t index, std::size_t end);
    PrefillProgress walk_step(Lane& lane, std::uint32_t index, Clock::time_point start);
    [[nodiscard]] execution::ForwardBatch walk_batch(std::size_t chunk);
    [[nodiscard]] Tensor walk_residual(std::size_t first, std::size_t end);
    void walk_mixer(std::size_t chunk, std::uint32_t layer);
    void walk_moe(std::size_t group, std::uint32_t layer);
    // The span's taps: realizes the lane's taps due at its inner chunk ends (as prefix_after_prefill_call
    // would after chunk-major calls) by opening a staged capture each; committed by walk_commit_taps.
    void walk_plan_taps(Lane& lane);
    void walk_capture(std::size_t chunk, std::uint32_t layer);
    void walk_commit_taps(Lane& lane);
    void walk_enqueue(std::uint32_t to_layer);
    void walk_abandon() noexcept;
    std::unique_ptr<RouteTrace> trace_;
    std::unique_ptr<ops::offloaded_moe::CpuMissService> cpu_service_;

    // ---- prefill CPU split (design §19.3.12) ----
    // Gives the CPU the thinnest non-resident experts of a layer while the link carries the rest:
    // narrow ones (n <= 8) in order of n, then expert id, as many as minimise the later of the
    // CPU's predicted time and the link's time for the experts left to it.
    // The model charges the link a record's copy per kept expert, but the stream copies most of a
    // layer's experts ahead, so the GPU side finishes far sooner than modelled and full rates overload
    // the CPU (traced 2026-10-10: the GPU waited ~the whole CPU time). Planning at kSplitRateShare of
    // the measured rates was fastest on a 13900K at 6 and at 3 workers (Dense8, pp256-1024; 6 workers
    // -4 to -6 % against the fitted rates, 3 workers -11 to -19 % against the measured ones). A line
    // fitted to measured layer times starved the CPU instead (the kept experts' cost is not linear).
    static constexpr double kSplitRateShare = 0.3;
    class SplitPolicy final : public CpuSplitPolicy {
    public:
        explicit SplitPolicy(const ProgramImpl& program) : program_(program) {}
        std::uint32_t choose(std::uint32_t layer, std::span<const std::int32_t> columns,
                             std::span<std::uint8_t> cpu) const override {
            const auto& p              = program_;
            const std::uint32_t E      = p.c_.moe.experts;
            const std::int32_t* frames = p.residency_->host_table() + static_cast<std::size_t>(layer) * E;
            const double link          = static_cast<double>(p.residency_->frame_pitch(layer)) / p.link_bytes_per_second_;
            // The team's measured rates (cpu_rates.cpp: one-column jobs fix the per-expert cost, 8-column
            // jobs the per-column cost) at kSplitRateShare, the share of them a gated layer realises.
            const double per_expert    = 1.0 / (p.cpu_expert_rate() * kSplitRateShare);
            const double per_column    = 1.0 / (p.cpu_column_rate() * kSplitRateShare);
            constexpr int kNarrow      = ops::offloaded_moe::kMaxCpuColumns;
            std::array<std::uint32_t, kNarrow + 1> width_count{};
            std::uint32_t absent = 0;
            for (std::uint32_t e = 0; e < E; ++e) {
                if (columns[e] <= 0 || frames[e] >= 0) { continue; }
                ++absent;
                if (columns[e] <= kNarrow) { ++width_count[static_cast<std::size_t>(columns[e])]; }
            }
            // The best count: the CPU's time grows and the link's shrinks with each expert moved.
            double best_time = absent * link, cpu_time = 0.0;
            std::uint32_t best = 0, k = 0;
            for (int n = 1; n <= kNarrow; ++n) {
                for (std::uint32_t i = 0; i < width_count[static_cast<std::size_t>(n)]; ++i) {
                    if (k == static_cast<std::uint32_t>(ops::offloaded_moe::kMaxCpuJobs)) { break; }
                    cpu_time += std::max(per_expert, n * per_column);
                    ++k;
                    const double time = std::max(cpu_time, (absent - k) * link);
                    if (time < best_time) {
                        best_time = time;
                        best      = k;
                    }
                }
            }
            // Mark the first `best` in that order.
            std::uint32_t marked = 0;
            for (int n = 1; n <= kNarrow && marked < best; ++n) {
                for (std::uint32_t e = 0; e < E && marked < best; ++e) {
                    if (columns[e] == n && frames[e] < 0) {
                        cpu[e] = 1;
                        ++marked;
                    }
                }
            }
            return marked;
        }

    private:
        const ProgramImpl& program_;
    };
    std::unique_ptr<SplitPolicy> split_policy_;
    bool split_call_ = false;
    // Prefill staging overlap and the shared expert's stream (see ForwardExperts); destroyed after every
    // call has completed.
    struct OverlapResources {
        ~OverlapResources() {
            for (auto event : events) {
                if (event != nullptr) { cudaEventDestroy(event); }
            }
            for (auto event : shared_events) {
                if (event != nullptr) { cudaEventDestroy(event); }
            }
            for (auto event : split_events) {
                if (event != nullptr) { cudaEventDestroy(event); }
            }
            if (stream != nullptr) { cudaStreamDestroy(stream); }
            if (shared_stream != nullptr) { cudaStreamDestroy(shared_stream); }
            if (split_stream != nullptr) { cudaStreamDestroy(split_stream); }
        }
        cudaStream_t stream = nullptr;
        std::array<cudaEvent_t, 5> events{};
        cudaStream_t shared_stream = nullptr;
        std::array<cudaEvent_t, 4> shared_events{};
        cudaStream_t split_stream = nullptr;         // forked calls' staged misses
        std::array<cudaEvent_t, 2> split_events{};
    };
    OverlapResources overlap_;
    std::unique_ptr<execution::Forward> forward_;
    std::array<DecodeGraph, kMaximumConcurrency> graphs_;
    std::vector<DecodeGraph> mtp_draft_graphs_; // (batch - 1) * mtp_k_ + steps - 1
    // With sampled drafts, the same for rounds with a sampled row (greedy rounds keep the argmax).
    std::vector<DecodeGraph> mtp_sampled_graphs_;
    std::vector<DecodeGraph> mtp_gated_graphs_; // one-row argmax rounds: steps - 1
    // Whether this round's MTP drafts were sampled (their proposals wait in the drafter's io).
    bool mtp_sampled_round_ = false;
    std::vector<DecodeGraph> mtp_catch_graphs_; // (batch - 1) * max_width_ + width - 1
    std::vector<DecodeGraph> verify_graphs_; // (batch - 1) * max_width_ + width - 1
    // A verification round's commit (commit_verified), indexed as verify_graphs_.
    std::vector<DecodeGraph> commit_graphs_;
    // screened-lm-head (screened_round): the screened rounds' decode and verification graphs, indexed as
    // graphs_ and verify_graphs_.
    bool screened_head_ = false;
    std::array<DecodeGraph, kMaximumConcurrency> screened_graphs_;
    std::vector<DecodeGraph> screened_verify_graphs_;
    // The startup capture (capture_graph_set, candidates §23 NU-04):
    // whether it is capturing now (graphs upload instead of launching), whether it ran (a later capture is
    // reported), how many graphs it captured and the device memory they took.
    struct GraphSet {
        bool capturing         = false;
        bool at_startup        = false;
        std::uint32_t captured = 0;
        std::int64_t bytes     = 0;
    };
    GraphSet graph_set_;

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
    std::array<std::int32_t, kMaximumConcurrency> mtp_confident_{}; // per row: drafts kept by the cumulative cut
    std::array<bool, kMaximumConcurrency> from_ngram_{};
    // Copy proposals are chosen against the MTP drafts by expected tokens per round time, their
    // acceptance learned per match-length bucket (4-7, 8-11, 12-15, 16+ tokens; Strata's priors).
    // IQ3_S import, 4K + 512 tokens, MTP 4 and 7 n-gram drafts: 190.2 -> 194.2 tok/s (2026-10-15).
    static int ngram_bucket(std::uint32_t matched) { return matched < 8 ? 0 : matched < 12 ? 1 : matched < 16 ? 2 : 3; }
    static constexpr std::array<double, 4> kNgramAcceptancePriors{0.75, 0.88, 0.93, 0.96};
    std::array<int, kMaximumConcurrency> ngram_bucket_{};
    // DFlash2's acceptance per probability decile learned across the process's requests with the
    // requests' own EWMA (candidates VS-2 / D11's draft half): a new request starts from it rather than
    // from the deciles' face values, which a short request otherwise spends its rounds unlearning.
    // Draft-only. Dense8 DFlash2 auto, ABBA x5: struct 1.0168x (4/5), conv 1.0089x (4/5); MTP's
    // per-position acceptance learned the same way was flat (0.9985x / 0.9961x) and starts from its prior.
    runtime::speculative::DraftCalibration dflash_process_calibration_;
    // Draft-length policy: acceptance prior and EWMA weight, and the cost of each draft column in
    // plain rounds (warm tg512 on the 5090: W = 3, 4, 5 rounds took 1.84, 2.13 and 2.54 plain
    // rounds, design 19.2; re-measured 2026-10-10 at 1.61-1.79, 1.89-2.11 and 2.42-2.53 between 8 %
    // and 20 % expert misses, where a miss-rate cost gained no more than it lost; W = 5 and 6 then
    // became 9-17 % cheaper with the one-column expert kernels up to 8 columns, not re-fitted here).
    static constexpr double kAcceptancePrior      = 0.75;
    static constexpr double kAcceptanceWeight     = 0.1;
    static constexpr double kWidthCost            = 0.38;
    static constexpr std::uint64_t kProbeInterval = 8;
    // DFlash2's calibrated length (--draft-tokens auto): a verified draft's column and a proposal in
    // plain one-row rounds. Measured on the 5090 with the NVFP4 Dense8 import, greedy code, INT8 KV
    // (2026-10-20): fixed lengths 2-7 took 14.2-27.7 ms rounds against 8.4 ms plain ones, about 2.6 ms
    // a draft and 0.6-1 ms of proposal. The online fit (width_cost) chose shorter lengths and was
    // 6 % slower, so the priors stand. Re-checked on the GSQ-RCO IQ3_S import with the tensor-core expert
    // kernels (2026-10-29, full layout, 2 runs each): 16K steady state 252.3 / 260.4 / 249.5 tok/s at costs
    // 0.15 / 0.31 / 0.5 (2.61 drafts, 2.01 accepted a round at 0.31), the struct fixture flat within its noise.
    static constexpr double kDFlashDraftCost    = 0.31;
    static constexpr double kDFlashProposalCost = 0.1;
    // The online draft cost (width_cost): sample decay (about 50 rounds), the rounds and the share of
    // K's variance (of the mean square) it needs, its ceiling, and the cycle that counts as a gap.
    static constexpr double kCostDecay     = 0.98;
    static constexpr double kCostMinRounds = 16.0;
    static constexpr double kCostMinSpread = 0.02;
    static constexpr double kCostMax       = 2.0;
    static constexpr double kCostGapRatio  = 3.0;
    // Two-row rounds: at most 2 drafts per row, each costing kPairWidthCost plain two-row rounds
    // (C = 2 decode on the 5090, AIME prompts sampled: K = 1 and 2 rounds took 1.44 and 1.95 plain
    // two-row rounds; K = 3, 8 columns, took 3.1 and lost to plain decode; design 19.3.5).
    static constexpr std::int32_t kPairMaxDrafts = 2;
    static constexpr double kPairWidthCost       = 0.47;
    // The GDN prefill's sub-chunk: every prefill call starts on a multiple of it.
    static constexpr std::uint32_t kGdnGrid = 16;
    // Without free cache resume (the reproducible route) taps, coalescing and resumes sit on the prefill-chunk
    // grid, as Qwen3.5's do: a resumed prefill then runs the cold prefill's calls. Several prefill routes change
    // their arithmetic with the call's width (GDN: decode update at 1 column, recurrent kernel to 15, chunked from
    // 16; Q8 dense SIMT to 8 columns, MMA tiles beyond; BF16 dense by width; QSA attention split or unsplit; the
    // hyper-connection mix fused at decode widths), so a resume on the 16-token grid gave a position other bits
    // than the cold prefill: on Dense8 every resume of 1-17 tokens at 1040 against the cold [1024, L) call differed
    // (|dlogprob| up to 0.078; infernix_qwen4_exp_resume_width_real_test, 2026-10-27).
    [[nodiscard]] std::uint32_t resume_grid() const noexcept {
        if (options_.free_cache_resume) { return 1U; }
        return static_cast<std::uint32_t>(std::max(chunk_, static_cast<std::int32_t>(kGdnGrid)));
    }
    // dflash2-pairs' width with --draft-tokens auto (dflash2_pair_width): a draft per row and the
    // two-row proposal in plain two-row rounds, fixed priors as for one row (an online fit chose
    // shorter lengths there and was slower). A pair round verifies MTP pair rounds' widths with the
    // same kernels, so a draft takes their kPairWidthCost (which also paid MTP's draft steps: an upper
    // bound here); the block proposal for two rows is taken as one row's share (kDFlashProposalCost) of
    // the longer two-row round. Neither is measured on DFlash2 pair rounds yet.
    static constexpr double kDFlashPairDraftCost    = kPairWidthCost;
    static constexpr double kDFlashPairProposalCost = 0.1;
    // The cumulative cut (MtpCumulative): each staged row's threshold (the device gate's for row 0 of
    // a one-row round, the host cut's for every row), and each row's draft log-probabilities, kept for
    // the verified chain's observe.
    std::array<float, kMaximumConcurrency> mtp_cumulative_threshold_{};
    // A fixed draft length's cut: log(--spec-min-p), -inf without one (every draft verified).
    double draft_min_logprob_ = 0.0;
    std::array<std::array<float, 8>, kMaximumConcurrency> mtp_draft_logprobs_{};
    // Prompt seeding (seed_prompt_experts): the LFRU count an expert gains when every token of the
    // prompt routed to it. Mixed-topic Dense8 decode, credit only: 64 -0.9 %, 256 -1.8 % time, and
    // replayed decode hits +0.34 / +0.59 / +0.54 points at 64 / 256 / 1,024 (design §19.3.27).
    static constexpr double kPromptSeedCredit = 256.0;
    // sampled-mtp-drafts (ProgramOptions::sampled_mtp_drafts) on a token domain the sparse
    // acceptance serves: sampled rows draw their drafts and skip the confidence cut.
    bool sampled_drafts_ = false;
    std::uint64_t pair_policy_rounds_            = 0;
    bool plain_round_ = false;

    // The neural drafter's most drafts per round (MTP's or DFlash2's), and whether a lane drafts.
    [[nodiscard]] std::int32_t neural_k() const noexcept { return mtp_ ? mtp_k_ : dflash_k_; }
    [[nodiscard]] bool drafter_live(const Lane& lane) const noexcept { return mtp_ ? lane.mtp_live : dflash2_; }

    // ---------------------------------------------------------------- DFlash2 drafter (dflash2_program.cpp)
    void allocate_dflash2(std::int32_t lanes, std::uint64_t& allocated);
    void create_dflash2_drafter();
    [[nodiscard]] std::int32_t dflash_feature_rows() const noexcept {
        return dim(plan_.dflash2_config->feature_width(c_.hidden_size));
    }
    // Feature columns [column, column + n) (BF16 [taps * H, n]).
    [[nodiscard]] Tensor dflash_feature_columns(std::int32_t column, std::int32_t n) const {
        return Tensor(static_cast<std::byte*>(dflash_features_.p) +
                          2ULL * static_cast<std::size_t>(dflash_feature_rows()) * static_cast<std::size_t>(column),
                      DType::BF16, {dflash_feature_rows(), n});
    }
    // Where a prefill call or forced tokens of `cols` columns leave their features: the call region's
    // last window. During another lane's layer walk, whose span holds the call region, forced tokens
    // use the round region (no round is pending between rounds) and keep its columns.
    [[nodiscard]] std::pair<std::int32_t, std::int32_t> dflash_call_region(std::int32_t cols) const noexcept {
        if (walk_.active) { return {0, std::min(cols, plan_.dflash_round_columns)}; }
        return {plan_.dflash_round_columns, std::min(cols, dim(plan_.dflash2_config->window))};
    }
    std::int32_t* dflash_host(std::size_t offset) const { return static_cast<std::int32_t*>(dflash_host_.data()) + offset; }
    std::int32_t* dflash_device(std::size_t offset) const { return static_cast<std::int32_t*>(dflash_device_.p) + offset; }
    // One proposal of dflash_k_ drafts for every row of a decode round (the rows' drafts land in
    // dflash_drafts_, their candidates and laws in the io); `while_drafting` runs on the host after
    // it is submitted, before its wait.
    void dflash2_propose(std::span<const std::uint32_t> lanes, const std::function<void()>& while_drafting);
    // dflash2_propose's call over `batch` rows of the staged io (the proposal graph's body and the startup capture's).
    [[nodiscard]] execution::DFlash2ProposalCall dflash2_proposal_call(std::int32_t batch);
    // After a round's commit: each row's committed columns' features into its ring (`commit`: the
    // committed columns per row, already on the device at spec_layout_.commit).
    void dflash2_catch_up(std::span<const SequenceHandle> rows, std::int32_t W, const std::int32_t* commit);
    // dflash2_catch_up's two halves, for a commit graph that holds the materialization (commit_verified): the call
    // over a round of `batch` rows of width W, whose counts the device reads
    // at spec_layout_.commit, and each committing row's new frontier.
    [[nodiscard]] execution::DFlash2ContextCall dflash2_context_call(std::int32_t batch, std::int32_t W);
    void dflash2_settle_frontiers(std::span<const SequenceHandle> rows, const std::int32_t* commit);
    // After a prefill call or forced tokens (one row): its last window's features into the ring.
    void dflash2_call_context(std::int32_t batch, std::int32_t width);
    // A layer walk: a tapped layer's completed residual for the span's last window, and after the
    // span the window into the ring.
    void dflash2_walk_features(std::size_t first, std::size_t end, std::uint32_t layer, const Tensor& residual);
    void dflash2_walk_context(Lane& lane, std::uint32_t index);
    // n consecutive positions from `first` of lane `slot`, whose features stand at `column`, into the ring.
    void dflash2_context(std::int32_t column, std::int32_t n, std::int32_t first, std::int32_t slot);
    // The stream waits for the prefix cache's copies of the lane's ring in flight (a restore, a copy-out).
    void dflash2_wait_copies(const Lane& lane);
    // A DFlash2 round's verification candidates and laws [16, K, batch] (their io offsets): a DFlash2
    // row's first drafts' from the proposal, an n-gram row's one-hot at its copied tokens. The
    // proposal's own planes when every drafting row is DFlash2's and K is the proposal's.
    std::pair<std::size_t, std::size_t> dflash2_verify_planes(std::int32_t batch, std::int32_t K);
    bool dflash2_          = false;
    bool dflash_round_     = false; // this verification's neural drafts are DFlash2's
    std::int32_t dflash_k_ = 0;
    DeviceBuffer dflash_ring_backing_, dflash_features_, dflash_device_;
    PinnedHostBuffer dflash_host_{1};
    DFlash2Io dflash_io_;
    std::unique_ptr<CyclicKVCache> dflash_ring_;
    std::unique_ptr<execution::DFlash2> drafter_;
    std::array<std::vector<std::int32_t>, kMaximumConcurrency> dflash_drafts_;
    std::array<std::int32_t, kMaximumConcurrency> dflash_confident_{}; // per row: drafts above draft_min_p
    std::array<std::array<float, 8>, kMaximumConcurrency> dflash_confidence_{}; // per row: each draft's
    // --draft-tokens auto: the calibrated length of row 0's proposal (draft_length.h), one draft more
    // every kProbeInterval rounds, so the deciles of longer drafts keep being measured.
    std::int32_t dflash2_auto_length(Lane& lane, std::int32_t most) {
        std::int32_t best = runtime::speculative::calibrated_draft_length(
            lane.dflash_calibration, std::span<const float>(dflash_confidence_[0].data(), static_cast<std::size_t>(most)),
            most, kDFlashDraftCost, kDFlashProposalCost);
        if (++lane.mtp_policy_rounds % kProbeInterval == 0) { best = std::min(best + 1, most); }
        return best;
    }
    // dflash2-pairs with --draft-tokens auto: the drafts per row a two-row round verifies after its
    // proposal (draft_length.h's pair_draft_width), from each row's expected tokens: a drafter row's
    // from its drafts' calibrated probabilities, a copy row's from its bucket's acceptance (its whole
    // copy is verified, so it bounds the width from below). `extents`: each row's drafts so far (at
    // most kPairMaxDrafts). One draft more every kProbeInterval pair rounds, as for one row.
    std::int32_t dflash2_pair_width(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> extents) {
        std::array<std::array<double, kPairMaxDrafts + 1>, 2> tokens{};
        std::int32_t least = 0, most = 0;
        for (std::size_t b = 0; b < 2; ++b) {
            const Lane& lane = lanes_[lanes[b]];
            const std::int32_t n = extents[b];
            most                 = std::max(most, n);
            const double p       = from_ngram_[b] && ngram_bucket_[b] >= 0
                                       ? lane.ngram_accept[static_cast<std::size_t>(ngram_bucket_[b])]
                                       : 0.0;
            if (from_ngram_[b]) { least = std::max(least, n); }
            double reach = 1.0;
            tokens[b][0] = 1.0;
            for (std::int32_t k = 1; k <= kPairMaxDrafts; ++k) {
                const auto at = static_cast<std::size_t>(k);
                if (k > n) {
                    tokens[b][at] = tokens[b][at - 1];
                } else if (from_ngram_[b]) {
                    reach *= p;
                    tokens[b][at] = tokens[b][at - 1] + reach;
                } else {
                    tokens[b][at] = runtime::speculative::expected_draft_tokens(
                        lane.dflash_calibration,
                        std::span<const float>(dflash_confidence_[b].data(), static_cast<std::size_t>(n)), k);
                }
            }
        }
        std::int32_t best = runtime::speculative::pair_draft_width(tokens[0], tokens[1], least, most,
                                                                   kDFlashPairDraftCost, kDFlashPairProposalCost);
        if (++pair_policy_rounds_ % kProbeInterval == 0) { best = std::min(best + 1, most); }
        return best;
    }
    std::vector<DecodeGraph> dflash_propose_graphs_; // batch - 1
    std::vector<DecodeGraph> dflash_context_graphs_; // (batch - 1) * max_width_ + width - 1
    // The DFlash2 frontier of the lane a layer walk started (its inner taps capture that ring).
    std::uint32_t walk_dflash_frontier_ = 0;

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
    // The n-gram gate of verification rounds (design §12.3): the pinned ready word, its last
    // published value (strictly increasing, never 0) and per-lane device wait statistics.
    PinnedHostBuffer gate_host_{64};
    std::uint32_t gate_sequence_ = 0;
    DeviceBuffer gate_stats_;
    // BF16 [1]: NgramTableConfig::fp8_scale, or FP32 [1]: NgramTableConfig::nvfp4_scale, when another
    // volume records it
    DeviceBuffer ngram_scale_;
    std::vector<std::uint8_t> live_;
    std::vector<std::int32_t> window_;
    std::vector<std::uint32_t> row_ids_;
    std::optional<std::uint32_t> transaction_lane_;
    std::optional<ResumeState> pause_; // the open Pause transaction's result
    bool replaying_ = false;           // the open binding replays toward a paused frontier
    std::uint64_t next_epoch_          = 0;
    std::uint64_t next_transaction_    = 0;
    std::uint64_t pending_transaction_ = 0;
    std::uint64_t revision_            = 1;
    // ---------------------------------------------------------------- vision (vision_program.cpp)
    // The encode window of the items ending past the reused prefix [0, reused).
    [[nodiscard]] std::shared_ptr<const VisionAdmission> plan_vision(const qwen3_5::PreparedPromptData& prompt,
                                                                     std::uint32_t reused = 0) const;
    void vision_reserve(Lane& lane, std::shared_ptr<const VisionAdmission> admission,
                        const qwen3_5::PreparedPromptData& prompt);
    // The window's frames are part of its binding's reservation: vision_lend lends them right after
    // the binding reserved its KV (the KV may grow into the frames first, never after a lease), or
    // returns the unreserved outcome when it cannot. Another lane holding the frames defers the
    // admission (no pages short); with every other lane idle the request is refused
    // (media_budget_exceeded). Failing a binding step instead would fail the Engine.
    [[nodiscard]] std::optional<BindingReservation> vision_lend(Lane& lane, std::uint32_t index);
    // Before such a binding reserves its KV, on an otherwise idle Program whose frames cannot hold
    // the window: the elastic KV pool's idle top (Host-backed cached blocks first) returns to them.
    void vision_make_room(const VisionAdmission* vision);
    // What the window may take now, and with no lease out (testing::vision_lend_limit caps both).
    [[nodiscard]] std::uint64_t vision_lendable() const noexcept;
    [[nodiscard]] std::uint64_t vision_lend_capacity() const noexcept;
    [[nodiscard]] static bool vision_encoded(const Lane& lane) noexcept { return !lane.vision || lane.vision->encoded; }
    void vision_step(Lane& lane);
    void vision_release(Lane& lane) noexcept;
    // The image columns of a prefill call [begin, begin + width), staged into the io; null without any.
    const execution::VisionInput* stage_vision(Lane& lane, std::uint32_t begin, std::int32_t width);
    // Per MTP sub-chunk: the cells whose next token is an image token; null without media.
    const execution::VisionInput* stage_mtp_vision(Lane& lane, std::uint32_t first, std::int32_t cells,
                                                   std::int32_t subchunks);
    // The call inputs that point into the io.
    execution::VisionInput vision_input_;
    std::vector<execution::VisionInput> mtp_vision_inputs_;

    // The prefix cache, declared last so it is destroyed first: it returns its page leases to pool_
    // and its lane image reads gdn_ and the lane buffers.
    prefix::StateImageLayout image_layout_;
    std::unique_ptr<prefix::LaneStateImage> lane_image_;
    std::unique_ptr<prefix::PrefixCache> prefix_;
    std::uint64_t next_capture_ = 0;
    std::array<std::span<const cudaEvent_t>, 2> next_waits_{};
    PinnedHostBuffer flush_host_{1};
    DeviceBuffer flush_device_;
    cudaEvent_t flush_uploaded_ = nullptr;

    // Last: its thread reads the source, the control law and the residency, so it stops first.
    std::unique_ptr<VramMonitor> monitor_;
};

} // namespace detail
} // namespace infernix::models::qwen4_exp
