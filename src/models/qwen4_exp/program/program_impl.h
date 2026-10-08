#pragma once

#include "models/qwen4_exp/program/program.h"

#include "core/arena.h"
#include "core/cuda_vmm.h"
#include "core/vmm_arena.h"
#include "core/decode_graph.h"
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
#include "models/qwen4_exp/execution/forward.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/frontend/ngram_hash.h"
#include "models/qwen4_exp/memory_plan.h"
#include "models/qwen4_exp/program/expert_residency.h"
#include "models/qwen4_exp/program/host_expert_tier.h"
#include "models/qwen4_exp/program/expert_cache/expert_state.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "models/qwen4_exp/program/prefix/call_plan.h"
#include "models/qwen4_exp/program/prefix/prefix_cache.h"
#include "models/qwen4_exp/program/prefix/state_image.h"
#include "models/qwen4_exp/program/rope_positions.h"
#include "models/qwen4_exp/program/vision_window.h"
#include "models/qwen3_5/execution/vision_tower.h"
#include "models/qwen3_5/execution/vision_weight_stream.h"
#include "models/qwen3_5/program/vision_control.h"
#include "models/qwen4_exp/program/route_trace.h"
#include "models/qwen4_exp/program/vram_monitor.h"
#include "ops/offloaded_sparse_moe/cpu/fetch_channel.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "infernix/ops/argmax.h"
#include "infernix/ops/cast.h"
#include "infernix/ops/gdn_replay.h"
#include "infernix/ops/ple.h"
#include "infernix/ops/qsa.h"
#include "infernix/ops/sampling.h"
#include "infernix/ops/speculative_round.h"
#include "infernix/ops/token_constraint.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>

namespace infernix::models::qwen4_exp {
namespace detail {

// A prompt with media: its items' control plan and the encode window it needs (design §19.3.2).
struct VisionAdmission {
    qwen3_5::VisionControlPlan control;
    VisionWindowPlan window;
};

// A TokenConstraint compiled for ops::constrain_logits: its distinct multi-token sets (written into
// the lane's set columns when it is first constrained) and each output step's descriptor relative to
// the lane (a set index, or the forced-token encoding -2 - token).
struct ConstraintPlan {
    std::vector<std::vector<TokenId>> sets;
    std::vector<std::int32_t> steps;
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
    // The current request's latest capture: its snapshot is the one the next capture supersedes.
    // `capture` may still name an earlier request's capture on this slot (its copy-out must land
    // before the slot is rewritten), which belongs to another lineage.
    std::uint64_t lineage_capture  = 0;
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
    static void set_constraint_failed(PendingBatch& p, std::size_t row, bool failed) noexcept {
        p.constraint_failed_[row] = failed;
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
    std::size_t ids = 0, positions = 0, rope = 0, block_rope = 0, slots = 0, rows = 0, columns = 0, gate = 0, ngram = 0,
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
// pooled-block starts [K][3][B] (axis-major per step), drafts [K, B], catch-up ids [W, B], gather
// columns [B], catch-up cells [W, B]. Everything before `drafts` is uploaded per draft round.
struct MtpIo {
    std::size_t ids = 0, cells = 0, rope_cells = 0, block_rope_cells = 0, drafts = 0, up_ids = 0, gather = 0,
                up_cells = 0, logprobs = 0, words = 0;
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
    // The vq2/k4v2 exact window of every KV layer and lane (prefix::KvWindowGeometry; zero otherwise).
    std::size_t window = 0;
    std::size_t logits32 = 0, logits16 = 0, token_counts = 0, work = 0, staging = 0;
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
        // Draft-length policy (design 11.3): conditional acceptance of each draft position.
        std::array<double, 8> mtp_accept{};
        std::uint64_t mtp_policy_rounds = 0;
        LanePrefix prefix;
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
        const std::size_t row_bytes = c.ple.table.row_bytes;
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
        d.io.ngram          = align(d.io.gate + 4);
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
        d.logits32     = sizeof(float) * c.vocab_size * lanes * W;
        d.logits16     = 2ULL * c.vocab_size * lanes * W;
        d.token_counts = 4ULL * static_cast<std::size_t>(d.token_domain) * lanes;

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
            d.mtp_io.drafts           = d.mtp_io.block_rope_cells + 3 * lanes * k;
            d.mtp_io.up_ids    = d.mtp_io.drafts + lanes * k;
            d.mtp_io.gather    = d.mtp_io.up_ids + static_cast<std::size_t>(lanes) * W;
            d.mtp_io.up_cells  = d.mtp_io.gather + lanes;
            d.mtp_io.logprobs  = d.mtp_io.up_cells + static_cast<std::size_t>(lanes) * W;
            d.mtp_io.words     = d.mtp_io.logprobs + lanes * k;
        }

        // Staging slots for each layer call's misses (design 8.6), one expert record each.
        const std::uint64_t shape[] = {c.moe.experts, c.hidden_size, c.moe.intermediate};
        d.record_stride = weight_geometry(QType::NVFP4_MUL, QuantLayout::ExpertRg16, shape).record_stride;
        d.staging       = static_cast<std::size_t>(kStagingSlots) * d.record_stride;

        auto& b     = d.bytes;
        b.kv_pages  = d.kv_pages;
        b.kv        = d.kv_elastic ? kv_mapped_bytes(d, d.kv_base_pages) : d.kv;
        b.kv_max    = d.kv;
        b.workspace = d.work;
        b.staging   = d.staging;
        b.state     = d.gdn_bytes + d.ple + d.tails + d.window + d.records_bytes + d.ple_records + d.qsa_records +
                  (d.mtp ? d.mtp_column_bytes * lanes * (W + 2) + 2ULL * di * W * lanes : 0);
        b.io = d.io.bytes + d.logits32 + d.logits16 + 4ULL * lanes + 4ULL * lanes +
               sizeof(ops::SamplingConfig) * lanes + d.token_counts + 4ULL * d.spec.words + d.mtp_ones +
               4ULL * d.mtp_io.words + kGateStatsBytes * lanes + constraint_bytes(lanes, W) +
               grammar_mask_bytes(d.token_domain, lanes, W);
        b.residency            = ExpertResidency::table_bytes(c, d.columns);
        b.expert_record_stride = d.record_stride;
        b.max_frames           = ExpertResidency::max_frames(c);
        b.graph_bound          = graph_bound(o.max_concurrency, static_cast<std::uint32_t>(W),
                                             static_cast<std::uint32_t>(d.mtp_k));
        return d;
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

    // The workspace arena of a Program with `lanes` lanes and the plan's widths: the forward at its
    // widest call (a prefill chunk or every lane's widest round), sampling, acceptance and the
    // drafter. The plan sizes the arena with the Program's lane count; the per-lane report compares
    // lane counts.
    static std::size_t workspace_bytes(const TextConfig& c, const ProgramOptions& o, const DeviceLayout& d,
                                       std::int32_t lanes, std::int32_t columns) {
        const std::int32_t W = d.max_width;
        std::size_t bytes =
            execution::Forward::workspace_bytes(c, std::max(columns, lanes * W), dim(o.max_context), o.kv_cache) +
            ops::sampling_workspace_capacity_bytes(d.token_domain, 1, lanes);
        if (W > 1) {
            bytes += ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(d.token_domain, 1, W - 1, 1, lanes);
        }
        if (d.mtp) {
            bytes += execution::Forward::mtp_workspace_bytes(c, std::max(lanes * W, std::min(d.chunk, 512)), lanes,
                                                             d.head_rows, dim(o.max_context), o.kv_cache);
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
               " MiB of Infernix's VRAM allocations were placed in shared system memory by the driver's sysmem "
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
        allocate(constraint_, constraint_bytes(lanes, max_width_), allocated);
        allocate(grammar_masks_, grammar_mask_bytes(token_domain_, lanes, max_width_), allocated);
        grammar_host_ = PinnedHostBuffer(grammar_mask_bytes(token_domain_, lanes, max_width_));
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
        const auto layout   = paged_kv_storage_layout(options_.kv_cache, dim(c_.attention.head_dim));
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
        work_capacity_ = plan_.work;
        wide_work_     = plan_.wide_work;
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
        // The SSD expert tier (design §19.3.7): the banks were not pinned; RAM slots hold the
        // records the tier keeps and the rest are read from the artifact when a round needs them.
        if (const auto& store = parameters_.model.expert_store(); store.has_value()) {
            if (store->record_bytes() != record_stride) {
                throw std::logic_error("Qwen4Exp: the expert store's records differ from the banks' stride");
            }
            // A call's CPU jobs hold their ring slots until the call's answer, and a pass's fetched
            // records until the pass has copied them: the ring holds both with a pass to spare.
            const std::uint32_t cpu_cap =
                options_.cpu_expert_workers > 0 && options_.cpu_expert_jobs > 0
                    ? std::min<std::uint32_t>(std::max(options_.cpu_expert_jobs, options_.cpu_assist_jobs),
                                              static_cast<std::uint32_t>(ops::offloaded_moe::kMaxCpuJobs))
                    : 0U;
            HostExpertTier::Options tier;
            tier.ring  = std::max<std::uint32_t>(128, cpu_cap + 2U * kStagingSlots);
            tier.slots = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                options_.expert_ram_bytes / record_stride, static_cast<std::uint64_t>(store->layers()) * store->experts()));
            const std::uint32_t lists = tier.ring + tier.prefetch + tier.demotion;
            if (tier.slots < std::max<std::uint32_t>(kMinTierSlots, lists + kMinTierSlots / 2)) {
                throw std::runtime_error("Qwen3.8-Flash-Next: the SSD expert tier needs RAM for at least " +
                                         std::to_string(std::max<std::uint32_t>(kMinTierSlots, lists + kMinTierSlots / 2)) +
                                         " expert slots (" + std::to_string(tier.slots) + " fit in the RAM ledger's expert share)");
            }
            fetch_channel_ = std::make_unique<ops::offloaded_moe::FetchChannel>();
            tier.fetch     = fetch_channel_.get();
            // A layer walk's step calls ceil(layers / n) layers once per chunk of its n-chunk span (one
            // tier round per step): at most layers + n - 1 layer calls.
            tier.layer_calls = store->layers() + kWalkMaxTokens / static_cast<std::uint32_t>(std::max(chunk_, 1)) + 1U;
            tier_          = std::make_unique<HostExpertTier>(*store, std::move(tier));
            residency_->attach_tier(tier_.get());
        }
        // After the tier (whose pinned slots are the probe's source when the banks stay in the
        // artifact), before the CPU service and the split take the measured rate.
        measure_link();
        expert_error_ = PinnedHostBuffer(64);
        std::memset(expert_error_.data(), 0, 64);
        experts.error = static_cast<std::uint32_t*>(expert_error_.data());
        if (options_.prefill_stream && options_.expert_cache) {
            execution::ExpertStream::RecordOf record_of;
            if (tier_) {
                record_of = [tier = tier_.get()](std::uint32_t layer, std::uint32_t expert) {
                    return tier->record(tier->key(layer, expert));
                };
            } else {
                std::vector<const std::uint8_t*> banks;
                for (const auto& layer : parameters_.layers) {
                    banks.push_back(reinterpret_cast<const std::uint8_t*>(layer.moe.bank->planes.records));
                }
                record_of = [banks = std::move(banks), record_stride](std::uint32_t layer, std::uint32_t expert) {
                    return banks[layer] + static_cast<std::size_t>(expert) * record_stride;
                };
            }
            execution::ExpertStream::Hold hold;
            if (tier_) {
                hold = [tier = tier_.get()](std::uint32_t layer, std::uint32_t expert) {
                    tier->stream_pin(tier->key(layer, expert));
                };
            }
            expert_stream_ = std::make_unique<execution::ExpertStream>(
                c_.num_hidden_layers, c_.moe.experts, record_stride, std::move(record_of), !tier_, std::move(hold));
            experts.stream = expert_stream_.get();
        }
        // CPU-served misses for decode and verify calls, and for every call of at most
        // kMaxCpuColumns columns (short prompts, chunk remainders, forced tokens), whose experts are
        // all thin and so give the same bits on either device (design 16.2); wider prefill chunks
        // stay on the GPU.
        const std::uint32_t cpu_workers = options_.cpu_expert_workers;
        const std::uint32_t cpu_jobs    = options_.cpu_expert_jobs;
        const std::uint32_t assist_jobs = options_.cpu_assist_jobs;
        if (cpu_workers > 0 && cpu_jobs > 0) {
            // Decode and verification calls are CPU-served up to decode_columns; prefill calls up to
            // kAssistMaxColumns take the assist cap (wider chunks stream or stage on the GPU).
            const int decode_columns = std::max(ops::offloaded_moe::kMaxCpuColumns, lanes * max_width_);
            const bool assist        = assist_jobs > 0 && decode_columns < kAssistMaxColumns;
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
                    .max_columns = assist ? kAssistMaxColumns : decode_columns,
                    .pcie_divisor = pcie_divisor(),
                    .wide_from   = assist ? decode_columns + 1 : 0,
                    .wide_jobs   = assist ? static_cast<int>(std::min<std::uint32_t>(assist_jobs, ops::offloaded_moe::kMaxCpuJobs)) : 0,
                    .cpus        = {},
                    .records     = tier_.get()}); // tier mode: SSD-only jobs read through the tier
            for (std::uint32_t l = 0; l < c_.num_hidden_layers; ++l) {
                experts.cpu.push_back(cpu_service_->channel(static_cast<int>(l)));
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
        experts.landing       = residency_->landing_table();
        experts.landed        = residency_->landed_log();
        experts.landing_slots = static_cast<std::int32_t>(ExpertResidency::kLandingSlots);
        if (tier_) {
            for (std::uint32_t l = 0; l < c_.num_hidden_layers; ++l) {
                experts.host_tables.push_back(residency_->record_table(l));
                experts.fetch.push_back(fetch_channel_->channel(static_cast<int>(l)));
            }
        }
        warm_start_experts();
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
        base->sampling = translate(options.sampling);
        const std::uint32_t positions = std::min(options_.max_context, n + base->summary.effective_output_tokens);
        base->positions = positions;
        base->pages     = (positions + kPagedKVPageSize - 1) / kPagedKVPageSize;
        base->prompt = std::make_shared<const PreparedPrompt>(std::move(prompt));
        return RequestBasePlan(std::move(base));
    }

    bool feasible(const RequestBasePlan& base) const noexcept {
        // A window lends at most what serving never gives up (a quarter of the frames stays).
        const bool frames = !base.impl_ || !base.impl_->vision ||
                            base.impl_->vision->window.frames <= residency_->frames() - residency_->frames() / 4;
        return base.impl_ != nullptr && frames && base.impl_->pages <= pool_->capacity_pages() &&
               base.impl_->pages <= static_cast<std::uint32_t>(pages_per_row_);
    }

    // The positions one round may write past the lane's frontier: a verification's columns and the
    // MTP drafter's cells (design §19.3.13).
    std::uint32_t round_positions() const noexcept {
        return static_cast<std::uint32_t>(std::max(max_width_, mtp_ ? mtp_k_ + 1 : 1)) + 1U;
    }

    // The KV pages a binding of `base` whose prefill ends at `end` reserves: through one round past
    // it, within the request's extent.
    std::uint32_t binding_pages(const BasePlanImpl& base, std::uint32_t end) const noexcept {
        const std::uint32_t positions = std::min(base.positions, end + round_positions());
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
        std::size_t calls = plan_prefill(data, 0, {}, base.reuse).ends.size();
        if (selection) {
            if (selection->snapshot) {
                // Items inside the reused prefix are not encoded again (design §19.3.2).
                if (impl->vision && selection->frontier > 0) { impl->vision = plan_vision(data, selection->frontier); }
                out.reused_tokens = selection->frontier;
                out.reuse_path    = reuse_path_for(prefix_->index().snapshot(*selection->snapshot).kind);
            }
            calls        = selection->plan.ends.size();
            impl->prefix = std::move(selection);
        }
        out.remaining_work.chunks = calls;
        out.remaining_work.tokens = base.summary.prompt_tokens - out.reused_tokens;
        out.hybrid                = std::move(impl);
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
    // a lane-resident snapshot), reserves the whole KV extent and maps the cached blocks. The Begin
    // the Engine published names the source's frontier, so a changed choice is refused and the Engine
    // falls back to the next source (the root).
    runtime::ResourceReservation start_binding(const RequestBasePlan& base, runtime::LaneId destination,
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
        lane.speculative    = {};
        lane.mtp_cells      = 0;
        lane.mtp_live       = mtp_;
        lane.row = tables_->acquire(static_cast<std::int32_t>(index));
        const auto shortage_of = [&](std::uint32_t need) {
            runtime::ResourceReservation out;
            out.shortage.main_kv_pages = need > pool_->available_pages() ? need - pool_->available_pages() : 0;
            return out;
        };
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
            if (q.vision) { vision_reserve(lane, q.vision, data); }
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
            if (q.vision) { vision_reserve(lane, q.vision, data); }
            lane.pages.clear();
            lane.pages.reserve(q.pages);
            pool_->materialize(*reservation, q.pages, lane.pages);
            lane.reservation = std::move(*reservation);
            tables_->publish(lane.row.handle(), 0, std::span<const DeviceKVPageLease>(lane.pages), device_.stream);
            reset_slot(index);
            lane.calls = plan_prefill(data, 0, {}, false).ends;
        }
        lane.next_call = 0;
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
        lane.readout        = b.readout_tokens;
        lane.prompt         = b.prompt;
        lane.constraint     = b.constraint;
        lane.constraint_round.clear();
        lane.constraint_trace.clear();
        lane.sampling.token_counts =
            static_cast<std::int32_t*>(token_counts_.p) + static_cast<std::size_t>(index) * token_domain_;
        lane.epoch           = ++next_epoch_;
        lane.phase           = Phase::Prefill;
        lane.prefill_ns      = 0;
        lane.decode_ns       = 0;
        lane.decode_share_ns = 0;
        lane.cache_at_admission = residency_->stats();
        lane.cpu_served_at_admission = cpu_service_ ? cpu_service_->served_experts() : 0;
        if (tier_) { lane.tier_at_admission = tier_->stats(); }
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
        runtime::ResourceReservation out;
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
        r.vision_seconds          = lane.vision_seconds;
        r.speculative             = lane.speculative;
        r.mtp_accept              = lane.mtp_accept;
        r.mtp_policy_rounds       = lane.mtp_policy_rounds;
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
    runtime::ResourceReservation bind_replay(const BasePlanImpl& b, std::uint32_t index, ResumeStateImpl& r) {
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
                runtime::ResourceReservation out;
                out.shortage.main_kv_pages = pages > pool_->available_pages() ? pages - pool_->available_pages() : 1U;
                return out;
            }
            // A state cached at the paused frontier itself leaves nothing to replay.
            selection->plan = selection->frontier < r.target
                                  ? prefix::plan_calls(selection->frontier, r.target, static_cast<std::uint32_t>(chunk_), {},
                                                       prefix::CallCost{.model = options_.prefix_cost,
                                                                        .span_seconds = options_.prefix_span_seconds})
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
        lane.speculative    = {};
        lane.mtp_cells      = 0;
        lane.mtp_live       = mtp_;
        lane.row            = tables_->acquire(static_cast<std::int32_t>(index));
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
                runtime::ResourceReservation out;
                out.shortage.main_kv_pages = selection->need - std::min(selection->need, pool_->available_pages());
                prefix_unpin(*selection);
                lane.row = KVExecutionRowLease{};
                return out;
            }
            if (b.vision) { vision_reserve(lane, plan_vision(ledger, selection->frontier), ledger); }
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
                runtime::ResourceReservation out;
                out.shortage.main_kv_pages = pages - std::min(pages, pool_->available_pages());
                lane.row = KVExecutionRowLease{};
                return out;
            }
            if (b.vision) { vision_reserve(lane, plan_vision(ledger, 0), ledger); }
            lane.pages.clear();
            lane.pages.reserve(pages);
            pool_->materialize(*reservation, pages, lane.pages);
            lane.reservation = std::move(*reservation);
            tables_->publish(lane.row.handle(), 0, std::span<const DeviceKVPageLease>(lane.pages), device_.stream);
            reset_slot(index);
            lane.calls = prefix::plan_calls(0, r.target, static_cast<std::uint32_t>(chunk_), {},
                                            prefix::CallCost{.model = options_.prefix_cost,
                                                             .span_seconds = options_.prefix_span_seconds})
                             .ends;
        }
        lane.next_call = 0;
        if (max_width_ > 1 && options_.ngram_draft_tokens > 0) {
            lane.proposer = std::make_unique<qwen3_5::detail::NgramProposer>(proposer_tokens_, proposer_tokens_ / 2);
            for (const auto token : lane.history) { lane.proposer->append(token); }
        }
        lane.sampling = b.sampling;
        lane.sampling.token_counts =
            static_cast<std::int32_t*>(token_counts_.p) + static_cast<std::size_t>(index) * token_domain_;
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
        runtime::ResourceReservation out;
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
        std::vector<std::int32_t> host(static_cast<std::size_t>(token_domain_), 0);
        for (std::size_t i = lane.prompt_tokens; i < lane.history.size(); ++i) {
            const std::int32_t token = lane.history[i];
            if (token >= 0 && token < token_domain_) { ++host[static_cast<std::size_t>(token)]; }
        }
        CUDA_CHECK(cudaMemcpyAsync(counts, host.data(), 4ULL * token_domain_, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaStreamSynchronize(s)); // the host table is read before it goes out of scope
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

    // The Engine's reclaim for a binding shortage: evict cached Device blocks until it is covered.
    bool hybrid_reclaim(runtime::ContextResourceUsage shortage) {
        if (!prefix_ || shortage.main_kv_pages == 0) { return false; }
        const std::uint32_t before = pool_->available_pages();
        pc_make_room(before + shortage.main_kv_pages);
        if (pool_->available_pages() == before && prefix_->transfers_pending()) {
            // Blocks pinned only by in-flight Host writes become evictable once those land.
            prefix_->drain();
            pc_make_room(before + shortage.main_kv_pages);
        }
        return pool_->available_pages() > before;
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
        run(1, width, 1, chunk ? &*chunk : nullptr, stage_vision(lane, static_cast<std::uint32_t>(begin), width), streamed);
        if (last && !prefilling_besides(index)) {
            return_stream_lease();
            return_wide_work();
        }
        lane.state_tokens += static_cast<std::uint32_t>(width);
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
            residency_->after_round(device_.stream, width, promotions);
            apply_vram_target(false);
        }

        return prefill_progress(lane, index, begin, width, static_cast<std::uint32_t>(width), last, promotions, start);
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
        if (last && lane.replay) {
            // A replay ends at its paused frontier, whose next input the ledger already holds: the
            // lane decodes from there (design §19.3.13).
            {
                const nvtx::ScopedRange wait_range(nvtx::Name::DeviceWait, nvtx::Category::Prefill);
                device_.synchronize();
            }
            trace_round(RouteTraceKind::PrefillChunk, 1, width, static_cast<std::uint32_t>(width), promotions,
                        static_cast<std::uint32_t>(begin));
            residency_->after_round(device_.stream, width, promotions);
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
            sample(lanes, positions);
            if (!lane.readout.empty()) { out.readout = read_prompt_frontier(lane); }
            {
                const nvtx::ScopedRange residency_range(nvtx::Name::PrefillResidency, nvtx::Category::Moe,
                                                        static_cast<std::uint64_t>(width));
                trace_round(RouteTraceKind::PrefillChunk, 1, width, static_cast<std::uint32_t>(width), promotions,
                            static_cast<std::uint32_t>(begin));
                residency_->after_round(device_.stream, width, promotions);
                apply_vram_target(false);
            }
            const SequenceHandle rows[] = {handle(index)};
            out.timing.submit_host_ns = elapsed_ns(start);
            lane.prefill_ns += out.timing.submit_host_ns;
            out.pending.emplace(ContractAccess::make_pending(this, ++next_transaction_, rows,
                                                             {pending_tokens_.data(), 1}, out.timing));
            mark_grammar_failures(*out.pending, {});
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
        // - two rows: one draft length for both (choose_pair_draft_length), so no row is padded to the
        //   other's; its at most 2 x 3 = 6 columns stay on the column-invariant dense routes, so greedy
        //   output equals C = 1, and below the 7-column fork and CPU-job limits;
        // - three or more rows decode plain (one draft each measured slower at four rows).
        // A plain round's W = 1 catch-up keeps each drafter current.
        const std::int32_t pair_k =
            batch == 2 && mtp_ ? choose_pair_draft_length(lanes_[lanes[0]], lanes_[lanes[1]]) : 0;
        const bool speculate = batch == 1 || pair_k > 0;
        std::array<std::int32_t, kMaximumConcurrency> wanted{};
        std::int32_t steps = 0;
        if (mtp_) {
            for (std::int32_t b = 0; b < batch; ++b) {
                Lane& lane = lanes_[lanes[b]];
                wanted[b]  = speculate && lane.mtp_live ? (batch == 1 ? choose_draft_length(lane) : pair_k) : 0;
                steps      = std::max(steps, wanted[b]);
            }
        }
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
            if (mtp_ && lane.mtp_live && wanted[b] > 0) {
                extents[b] = static_cast<std::int32_t>(std::min<std::uint32_t>(limit, static_cast<std::uint32_t>(wanted[b])));
            }
            if (speculate && lane.proposer) {
                auto copy = lane.proposer->propose(lane.history, limit, options_.ngram_min_match).tokens;
                if (copy.size() > static_cast<std::size_t>(extents[b])) {
                    drafts_[b]     = std::move(copy);
                    from_ngram_[b] = true;
                    extents[b]     = static_cast<std::int32_t>(drafts_[b].size());
                }
            }
            width = std::max(width, 1 + extents[b]);
        }
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
            // A one-row round verifies only its confident drafts: those before the first whose
            // draft-head probability is below 0.5 (kDraftMinLogprob). Its width may still shrink, as
            // the drafter staged only column 0's n-gram rows; a pair round's columns are staged at
            // their offsets, so it keeps its length.
            if (batch == 1 && !from_ngram_[0] && extents[0] > 0) {
                extents[0] = std::min(extents[0], mtp_confident_[0]);
                drafts_[0].resize(static_cast<std::size_t>(extents[0]));
                width = 1 + extents[0];
            }
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
        const bool gated = graphs_[static_cast<std::size_t>(batch - 1)].executable.ready() &&
                           !decode_rows_cached(round_lanes, round_positions);
        stage_decode(round_lanes, round_positions);
        const std::uint32_t word = next_gate_word();
        if (!gated) {
            stage_decode_rows(round_lanes, round_positions);
            publish_pinned_word(gate_ready(), word);
        }
        run_decode(batch);
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
        mark_grammar_failures(pending, {});
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
                lane.constraint_round.clear();
                out.rows[row].timings     = timings(lane);
                out.rows[row].speculative = lane.speculative;
                out.rows[row].disposition = runtime::CommitDisposition::CancelledReleased;
                continue;
            }
            if (d.accepted_tokens != 1) { throw std::logic_error("Qwen4Exp: a row commits exactly one token"); }
            commit_constraint_draws(lane, 1);
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
        for (std::size_t row = 0; row < rows.size(); ++row) {
            if (!decisions[row].cancelled) {
                Lane& lane = lanes_[ContractAccess::lane(rows[row])];
                prefix_publish(lane, prefix_frontier(lane, false));
            }
        }
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
                out.expert_cache = ExpertCacheStats{.routed = s.routed - a.routed, .hits = s.hits - a.hits};
            }
            report_cache(lanes_[index]);
            report_ngram(lanes_[index]);
            prefix_finish(lanes_[index], index);
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
                if (clear_prefix_on_release_) { prefix_->clear(); }
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
    // pass; a prefill chunk's in several.
    static constexpr std::int32_t kStagingSlots = 64;
    // The SSD tier's smallest RAM (design §19.3.7 H_min: 1,024 slots, 2.6 GiB).
    static constexpr std::uint32_t kMinTierSlots = 1024;
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
    // Fills the frames from the saved state, before the first request. With the SSD tier, RAM is
    // filled first (the frames load from it), then the seeds' RAM copies are released and their slots
    // refilled with the next keys of the fill order, so the two levels hold different experts.
    void warm_start_experts() {
        const auto keys = c_.num_hidden_layers * c_.moe.experts;
        expert_cache::ExpertStateLoad load;
        if (!options_.expert_state.empty() && options_.expert_cache) {
            load = expert_cache::load_expert_state(options_.expert_state, options_.expert_state_identity, keys);
        }
        const expert_cache::SavedState* state = load.state ? &*load.state : nullptr;
        TierFill fill;
        if (tier_) { fill = prefill_tier(state); }
        const std::uint32_t seeded = warm_start_frames(load);
        if (tier_) { finish_tier_fill(fill, seeded); }
    }

private:
    // The SSD tier's startup fill: its key order, how many keys the first pass read, and its time.
    struct TierFill {
        std::vector<std::uint32_t> order;
        std::uint32_t filled = 0;
        double seconds       = 0;
        bool saved           = false;
    };

    // The SSD tier's RAM fill order (design §19.3.7 §4.9): the saved state's ranking (the last
    // session's VRAM residents, best first), then every other key that session used, most used
    // first, then the keys it never used one expert of every layer in turn, so that no layer is left
    // wholly on the SSD (file order filled the first layers and left the last ones on disk).
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
        for (std::uint32_t i = 0; i < keys; ++i) {
            const std::uint32_t key = (i % layers) * experts + i / layers; // spread(key) == i
            if (listed[key] == 0) { order.push_back(key); }
        }
        return order;
    }

    // The SSD tier's first RAM pass, before the frames' warm start reads from it.
    TierFill prefill_tier(const expert_cache::SavedState* state) {
        const auto start = std::chrono::steady_clock::now();
        TierFill fill;
        fill.saved = state != nullptr;
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
    void finish_tier_fill(const TierFill& fill, std::uint32_t seeded) {
        const auto start        = std::chrono::steady_clock::now();
        std::uint32_t released  = 0, refilled = 0;
        if (seeded > 0) {
            released = tier_->controller().release_shadows();
            if (released > 0) {
                refilled = tier_->prefill(std::span<const std::uint32_t>(fill.order).subspan(fill.filled));
            }
        }
        const double seconds =
            fill.seconds + std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const std::uint32_t keys = tier_->keys(), in_ram = fill.filled - released + refilled;
        const std::uint32_t slots = tier_->controller().config().slots;
        const double gib          = static_cast<double>(residency_->frame_stride()) / (1ULL << 30);
        char line[400];
        std::snprintf(line, sizeof(line),
                      "SSD expert tier: %u RAM slots (%.1f GiB) hold %u experts (%s)%s, read in %.1f s; %u of %u "
                      "experts are read from the artifact when used",
                      slots, gib * slots, in_ram,
                      fill.saved ? "the saved ranking, then the last session's use counts" : "spread across the layers",
                      released > 0 ? (", none of them among the " + std::to_string(released) + " in VRAM").c_str() : "",
                      seconds, keys - in_ram - released, keys);
        diagnostic(line);
    }

    // The frames' warm start from the saved state; returns how many experts it loaded.
    std::uint32_t warm_start_frames(const expert_cache::ExpertStateLoad& load) {
        if (options_.expert_state.empty() || !options_.expert_cache) { return 0; }
        if (!load.state) {
            diagnostic("expert cache: no saved state yet, so it starts empty and fills as requests arrive");
            diagnostic("expert cache starts empty: " + load.message, DiagnosticLevel::Debug);
            expert_state_saved_ = std::chrono::steady_clock::now();
            return 0;
        }
        const auto start           = std::chrono::steady_clock::now();
        const auto max_keys = static_cast<std::uint32_t>(kSeedShare * residency_->frames());
        const std::uint32_t loaded = residency_->warm_start(*load.state, kSeedCountCap, max_keys, device_.stream);
        device_.synchronize();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const double loaded_gib =
            static_cast<double>(loaded) * static_cast<double>(residency_->frame_stride()) / (1ULL << 30);
        char line[256];
        std::snprintf(line, sizeof(line),
                      "expert cache: restored the %u most-used experts from the last session (%.1f GiB in %.2f s)",
                      loaded, loaded_gib, seconds);
        diagnostic(line);
        std::snprintf(line, sizeof(line), "expert cache warm start: %u of %u frames from %s", loaded,
                      residency_->frames(), options_.expert_state.string().c_str());
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
        if (value == 0 && testing::take_expert_fault()) { value = EIO; } // layer 0
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
                                   : code == ops::offloaded_moe::kErrorHostSilent   ? "the host expert service stopped answering"
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

private:
    // A seeded count is capped, so an expert used heavily in an old session yields to new uses.
    static constexpr std::uint32_t kSeedCountCap = 16;
    // The share of the frames a warm start fills (the best-ranked experts); the rest stay free for
    // the fill phase's landing (S4), so a workload unlike the saved state still fills quickly.
    static constexpr double kSeedShare = 0.5;
    static constexpr std::chrono::minutes kExpertStateInterval{10};
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
            const auto landed   = s.landed - at_admission.landed;
            const auto cpu      = cpu_service_ ? cpu_service_->served_experts() - lane.cpu_served_at_admission : 0;
            std::size_t free_vram = 0, total_vram = 0;
            if (cudaMemGetInfo(&free_vram, &total_vram) != cudaSuccess) { free_vram = 0; }
            char text[384];
            std::snprintf(text, sizeof(text),
                          "expert cache: %u frames; request %.1f%% of %llu routed experts hit, %llu promotions, "
                          "%llu landed, %llu misses CPU-served; since start %.1f%%; VRAM free %zu MiB",
                          residency_->frames(), routed ? 100.0 * static_cast<double>(hits) / static_cast<double>(routed) : 0.0,
                          static_cast<unsigned long long>(routed), static_cast<unsigned long long>(promoted),
                          static_cast<unsigned long long>(landed),
                          static_cast<unsigned long long>(cpu),
                          s.routed ? 100.0 * static_cast<double>(s.hits) / static_cast<double>(s.routed) : 0.0,
                          free_vram >> 20);
            diagnostic(text, DiagnosticLevel::Debug);
            if (tier_) {
                // The tier's counters as of the last round boundary.
                const auto& t = tier_->stats();
                const auto& a = lane.tier_at_admission;
                const auto reads = t.demand_reads - a.demand_reads;
                std::snprintf(text, sizeof(text),
                              "SSD expert tier: request %llu expert reads (%.2f GiB, mean %.2f ms), %llu failed; %llu "
                              "fetched (%llu from RAM); %llu admitted to RAM, %llu demoted from VRAM",
                              static_cast<unsigned long long>(reads),
                              static_cast<double>(reads) * static_cast<double>(residency_->frame_stride()) / (1ULL << 30),
                              reads ? 1e-6 * static_cast<double>(t.read_ns - a.read_ns) / static_cast<double>(reads) : 0.0,
                              static_cast<unsigned long long>(t.demand_failures - a.demand_failures),
                              static_cast<unsigned long long>(t.fetch_records - a.fetch_records),
                              static_cast<unsigned long long>(t.fetch_from_ram - a.fetch_from_ram),
                              static_cast<unsigned long long>(t.admitted - a.admitted),
                              static_cast<unsigned long long>(s.demotions - at_admission.demotions));
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
    // and QSA-tail state; its verification and drafter records; its logit columns and penalty
    // counts; and its share of the workspace. Every other allocation is independent of the lane
    // count, so one more lane costs these frames.
    void report_lane_cost(std::int32_t lanes, std::uint64_t record_stride) noexcept {
        try {
            const auto n         = static_cast<std::size_t>(lanes);
            const bool fixed_kv  = options_.kv_capacity_tokens != 0;
            const std::size_t kv = fixed_kv ? 0 : kv_span_.bytes / n;
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
        out.vision_seconds       = lane.vision_seconds;
        return out;
    }

    void release(std::uint32_t index) noexcept {
        Lane& lane = lanes_[index];
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
    }

    // Whether the elastic pool could grow to make `need` pages available (the quote's check).
    bool kv_growable(std::uint32_t need) const noexcept {
        return static_cast<std::uint64_t>(pool_->available_pages()) + kv_grow_pages() >= need;
    }

    // The pages one growth could add at most: grow_kv takes up to a quarter of the frames, and none
    // while frames are lent (an estimate; grow_kv decides).
    std::uint32_t kv_grow_pages() const noexcept {
        if (!kv_vmm_ || !residency_ || !residency_->elastic() || residency_->frames() == 0 ||
            residency_->stats().lent_frames != 0) {
            return 0;
        }
        const std::uint32_t room = pool_->capacity_pages() - pool_->backed_pages();
        const std::uint64_t page_bytes =
            std::max<std::uint64_t>(1, kv_span_.bytes / std::max<std::uint32_t>(1, pool_->capacity_pages()));
        const std::uint64_t frame_room = static_cast<std::uint64_t>(residency_->frames() / 4) * residency_->frame_stride();
        const std::uint64_t slack      = ExpertResidency::kChunkBytes + 2 * kKvChunkBytes * plan_.pool.planes.size();
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
        read_ngram_to(lane, count, host_io() + io_layout_.ngram + column * hash_.heads() * c_.ple.table.row_bytes);
    }

    // As read_ngram, into `out` (count * heads rows).
    void read_ngram_to(Lane& lane, std::int32_t count, std::byte* out) {
        const std::size_t heads = hash_.heads();
        row_ids_.resize(static_cast<std::size_t>(count) * heads);
        hash_.row_ids(window_, static_cast<std::size_t>(count), row_ids_.data());
        const std::size_t row_bytes = c_.ple.table.row_bytes;
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
        CUDA_CHECK(cudaMemcpyAsync(io_device_.p, io_host_.data(), io_layout_.bytes, cudaMemcpyHostToDevice, s));
        residency_->before_round(s);
        // The ring's copies queue on the copy engine behind the io upload above. A call the CPU can
        // share (design §19.3.12) gates the stream on each layer's routing.
        split_call_ = streamed && split_policy_ && batch * width <= split_columns();
        if (streamed) { expert_stream_->begin(ring_span(), residency_->host_table(), s, split_call_); }
        const WideWork wide(*this, batch * width);
        forward_call(batch, width, logit_columns, nullptr, chunk, vision, nullptr, streamed);
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
            const std::uint64_t stride  = residency_->frame_stride();
            const auto frames_of        = [&](std::size_t bytes) {
                return static_cast<std::uint32_t>((bytes + stride - 1) / stride);
            };
            // The wide arena rides in the lease whenever the plan has one: a prompt with a streamed
            // call runs wide calls.
            const std::uint32_t wide_frames = frames_of(wide_work_);
            const auto walk_frames          = frames_of(walk_bytes);
            const std::uint32_t lendable    = residency_->lendable();
            if (wide_frames >= lendable) { return false; }
            const std::uint32_t evicted = wide_frames + walk_frames;
            const std::uint32_t widened = std::min(E, most + (evicted + c_.num_hidden_layers - 1) / c_.num_hidden_layers);
            const bool walk = walk_frames > 0 && 2 * widened + 1 + wide_frames + walk_frames <= lendable;
            const std::uint32_t half = walk ? widened : std::min(E, most + (wide_frames + c_.num_hidden_layers - 1) /
                                                                              c_.num_hidden_layers);
            const std::uint32_t ring = std::min(2 * half + 1, lendable - wide_frames);
            if (half == 0 || expert_stream_->slots_in(static_cast<std::size_t>(ring) * stride) < 2 * kStreamHalfMin) {
                return false;
            }
            const cudaStream_t writers[] = {expert_stream_->stream()};
            stream_lease_      = residency_->lend(ring + wide_frames + (walk ? walk_frames : 0U), device_.stream, writers);
            stream_ring_bytes_ = static_cast<std::size_t>(ring) * stride;
            stream_wide_bytes_ = static_cast<std::size_t>(wide_frames) * stride;
            stream_walk_bytes_ = walk ? static_cast<std::size_t>(walk_frames) * stride : 0;
            stream_at_lease_   = expert_stream_->stats();
            split_at_lease_    = forward_->split_stats();
            prefetched_at_lease_ = walk_prefetched_chunks_;
        }
        return true;
    }

    // The ring's part of the stream lease.
    DeviceSpan ring_span() const noexcept { return DeviceSpan{stream_lease_.memory.data, stream_ring_bytes_}; }
    // The walk's area: after the ring and the wide arena.
    std::byte* walk_area() const noexcept {
        return static_cast<std::byte*>(stream_lease_.memory.data) + stream_ring_bytes_ + stream_wide_bytes_;
    }

    // ---- the wide workspace (design §19.3.7, VRAM item 1) ----
    // A call wider than the static arena's columns runs in the wide arena: the stream lease's part
    // when it has one, else frames lent for it alone, else (no lendable frames: the VRAM monitor
    // shrank the cache, or there are no frames) a temporary device allocation, with a warning. The
    // Forward keeps its arena reference; the arenas' contents are swapped for the call.
    class WideWork {
    public:
        WideWork(ProgramImpl& program, std::int32_t columns) : program_(program) {
            if (columns > program.plan_.static_columns && program.wide_work_ > 0) {
                program.enter_wide();
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

    void enter_wide() {
        DeviceSpan span{};
        if (stream_lease_.valid() && stream_wide_bytes_ >= wide_work_) {
            span = {static_cast<std::byte*>(stream_lease_.memory.data) + stream_ring_bytes_, wide_work_};
        } else {
            if (!work_lease_.valid() && wide_fallback_.p == nullptr) {
                const std::uint64_t stride = residency_->frame_stride();
                const auto frames          = static_cast<std::uint32_t>((wide_work_ + stride - 1) / stride);
                if (residency_->frames() > 0 && frames < residency_->lendable()) {
                    const cudaStream_t writers[] = {device_.stream};
                    work_lease_ = residency_->lend(frames, device_.stream, writers);
                } else {
                    diagnostic("the prefill workspace could not be lent from the expert cache; " +
                                   std::to_string(wide_work_ >> 20) + " MiB are allocated for the prompt",
                               DiagnosticLevel::Warning);
                    wide_fallback_ = DeviceBuffer(wide_work_);
                }
            }
            span = work_lease_.valid() ? DeviceSpan{work_lease_.memory.data, wide_work_}
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
        const auto frames = stream_lease_.count;
        residency_->give_back(stream_lease_, device_.stream);
        stream_lease_         = {};
        stream_ring_bytes_    = 0;
        stream_wide_bytes_    = 0;
        stream_walk_bytes_    = 0;
        const auto& now       = expert_stream_->stats();
        const auto streamed   = now.streamed - stream_at_lease_.streamed;
        const auto copies     = now.copies - stream_at_lease_.copies;
        const auto split      = forward_->split_stats();
        char line[320];
        std::snprintf(line, sizeof(line),
                      "prefill expert stream: %llu experts (%.2f GiB) in %llu copies through %u lent frames; "
                      "CPU split: %llu experts on the CPU, %llu gated copies; n-gram rows of %llu walk chunks read "
                      "during the previous span",
                      static_cast<unsigned long long>(streamed),
                      static_cast<double>(streamed) * static_cast<double>(residency_->frame_stride()) / (1ULL << 30),
                      static_cast<unsigned long long>(copies), frames,
                      static_cast<unsigned long long>(split.cpu_experts - split_at_lease_.cpu_experts),
                      static_cast<unsigned long long>(split.streamed_experts - split_at_lease_.streamed_experts),
                      static_cast<unsigned long long>(walk_prefetched_chunks_ - prefetched_at_lease_));
        diagnostic(line, DiagnosticLevel::Debug);
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
    // first round of a size runs eagerly (it also performs the Ops' one-time setup); the next is
    // captured and later rounds replay it.
    void run_decode(std::int32_t batch) {
        const cudaStream_t s = device_.stream;
        upload_pinned(io_device_.p, io_host_.data(), io_layout_.ngram, s);
        residency_->before_round(s, /*landing=*/true);
        const execution::NgramRowGate gate = row_gate();
        replay(graphs_[static_cast<std::size_t>(batch - 1)],
               [&] { forward_call(batch, 1, batch, nullptr, nullptr, nullptr, &gate); });
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
            // A captured body bakes in whatever it reads by address: the arena *work_ points at and
            // the per-layer waits forward_call consumes. Neither may be a call's temporary.
            if (wide_active_ || !next_waits_[0].empty() || !next_waits_[1].empty()) {
                throw std::logic_error("Qwen4Exp: a graph capture inside a wide call or with pending layer waits");
            }
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
                      const execution::ForwardVerify* verify = nullptr, const execution::MtpChunk* chunk = nullptr,
                      const execution::VisionInput* vision = nullptr, const execution::NgramRowGate* gate = nullptr,
                      bool streamed = false) {
        execution::ForwardBatch fb = io_batch(static_cast<std::byte*>(io_device_.p), batch, width, logit_columns);
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
        Tensor logits(logits32_.p, DType::FP32, {vocab_, logit_columns});
        forward_->run(fb, logits);
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
                                  {dim(c_.ple.table.row_bytes), dim(hash_.heads()), cols});
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
        DecodeGraph& graph = verify_graphs_[static_cast<std::size_t>(batch - 1) * max_width_ + (W - 1)];
        const bool gated   = graph.executable.ready();
        if (!gated) {
            read_rows();
            publish_pinned_word(gate_ready(), word);
        }
        upload_pinned(io_device_.p, io_host_.data(), io_layout_.ngram, s);
        upload_pinned(spec_device_.p, spec_host_.data(), 4ULL * spec_layout_.licensed, s);
        residency_->before_round(s, /*landing=*/true);
        const execution::NgramRowGate gate = row_gate();
        replay(graph, [&] {
            const execution::ForwardVerify view = verify_view(batch, W);
            forward_call(batch, W, batch * W, &view, nullptr, nullptr, &gate);
        });
        // From here an enqueued gate waits for `word`: an error before it is published still
        // publishes it (over zeroed rows), so the stream drains and the error keeps its class.
        GateRelease release_gate{gated ? this : nullptr, word, io_prefix(batch * W) - io_layout_.ngram};
        residency_->enqueue_route_download(s, batch * W);

        // Acceptance on the device, from BF16-rounded logits as in plain decode (design 16.5).
        Tensor wide(logits32_.p, DType::FP32, {vocab_, batch * W});
        Tensor narrow(logits16_.p, DType::BF16, {vocab_, batch * W});
        ops::cast_fp32_to_bf16(wide, narrow, s);
        Tensor target(spec_device(spec_layout_.target), DType::I32, {batch * W});
        ops::argmax(narrow, target, token_domain_, s);
        const bool constrained = stage_constraints(lanes, W);
        if (constrained) { constrain(narrow, &target); }
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
        {
            auto scope = work_->scope();
            ops::speculative_accept_greedy_drafts(target.view({W, batch}), narrow.view({vocab_, W, batch}), drafts,
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
        auto pending = ContractAccess::make_pending(this, pending_transaction_, sequences,
                                                    {pending_tokens_.data(), static_cast<std::size_t>(batch * W)},
                                                    timing, {pending_counts_.data(), static_cast<std::size_t>(batch)},
                                                    static_cast<std::uint32_t>(W));
        mark_grammar_failures(pending, {pending_counts_.data(), static_cast<std::size_t>(batch)});
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
            if (d.cancelled) {
                lane.constraint_round.clear();
                out.rows[row].timings     = timings(lane);
                out.rows[row].speculative = lane.speculative;
                out.rows[row].disposition = runtime::CommitDisposition::CancelledReleased;
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
        for (std::int32_t row = 0; row < batch; ++row) {
            if (!decisions[static_cast<std::size_t>(row)].cancelled) {
                Lane& lane = lanes_[ContractAccess::lane(rows[row])];
                prefix_publish(lane, prefix_frontier(lane, false));
            }
        }
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

    // Two-row rounds (design 19.3.5): the K in [0, kPairMaxDrafts] that maximizes both rows' expected
    // tokens per round time, where a round of K drafts per row costs 1 + kPairWidthCost * K plain
    // two-row rounds. Every kProbeInterval pair rounds it drafts one token more than its choice, so
    // the rows' estimates of the next position stay current.
    std::int32_t choose_pair_draft_length(const Lane& a, const Lane& b) {
        if (!a.mtp_live || !b.mtp_live) { return 0; }
        const std::int32_t most = std::min(kPairMaxDrafts, mtp_k_);
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
        if (++pair_policy_rounds_ % kProbeInterval == 0) { best = std::min(best + 1, most); }
        return best;
    }

    // `while_drafting` runs on the host after the draft steps are submitted, before their wait
    // (only when draft steps run).
    void mtp_draft(std::span<const std::uint32_t> lanes, std::int32_t steps,
                   const std::function<void()>& while_drafting = {}) {
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
            // The lane's last mapped position: its shared prefix pages, then its private ones.
            const std::int32_t limit =
                static_cast<std::int32_t>(lane.prefix.page_base + lane.pages.size()) * kPagedKVPageSize - 1;
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
            forward_->run_mtp({.residuals        = chain,
                               .ids              = anchors,
                               .positions        = Tensor(mtp_device(mtp_io_.cells), DType::I32, {batch}),
                               .rope_positions   = Tensor(mtp_device(mtp_io_.rope_cells), DType::I32, {batch, 3}),
                               .block_start_rope = Tensor(mtp_device(mtp_io_.block_rope_cells), DType::I32, {batch, 3}),
                               .slots            = slot_tensor,
                               .table_rows       = row_tensor,
                               .batch            = batch,
                               .width            = 1,
                               .kv_only          = true});
        }
        if (steps == 0) {
            // No synchronize here: the uploads above may still be queued when the caller stages the
            // verification round into the same pinned slots and rows, which is safe only because it
            // stages the same lanes in the same order (and mtp_host_ is next written after that
            // round's synchronize). A caller that stages other lanes there must synchronize first.
            for (std::int32_t b = 0; b < batch; ++b) {
                Lane& lane = lanes_[lanes[b]];
                if (lane.mtp_live) { lane.mtp_cells = lane.state_tokens; }
            }
            return;
        }
        replay(mtp_draft_graphs_[static_cast<std::size_t>(batch - 1) * mtp_k_ + (steps - 1)], [&] {
        for (std::int32_t j = 0; j < steps; ++j) {
            Tensor drafts(mtp_device(mtp_io_.drafts + static_cast<std::size_t>(j) * batch), DType::I32, {batch});
            const std::size_t step = 3ULL * static_cast<std::size_t>(batch) * j;
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
                                          {1, batch})});
        }
        });
        CUDA_CHECK(cudaMemcpyAsync(mtp_host(mtp_io_.drafts), mtp_device(mtp_io_.drafts), 4ULL * steps * batch,
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(mtp_host(mtp_io_.logprobs), mtp_device(mtp_io_.logprobs), 4ULL * steps * batch,
                                   cudaMemcpyDeviceToHost, s));
        if (while_drafting) {
            device_.flush();
            while_drafting();
        }
        device_.synchronize();
        for (std::int32_t b = 0; b < batch; ++b) {
            Lane& lane = lanes_[lanes[b]];
            if (unwritten && lane.mtp_live) { lane.mtp_cells = lane.state_tokens; }
            for (std::int32_t j = 0; j < steps; ++j) {
                mtp_drafts_[b][static_cast<std::size_t>(j)] = mtp_host(mtp_io_.drafts)[j * batch + b];
            }
            // The row's confident drafts: those before the first below kDraftMinLogprob.
            mtp_confident_[b] = steps;
            const auto* logprobs = reinterpret_cast<const float*>(mtp_host(mtp_io_.logprobs));
            for (std::int32_t j = 0; j < steps; ++j) {
                if (logprobs[j * batch + b] < kDraftMinLogprob) {
                    mtp_confident_[b] = j;
                    break;
                }
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
        const Tensor rope(io + io_layout_.rope, DType::I32, {W * batch, 3});
        const Tensor block_rope(io + io_layout_.block_rope, DType::I32, {batch, 3});
        const Tensor slots(io + io_layout_.slots, DType::I32, {batch});
        replay(mtp_catch_graphs_[static_cast<std::size_t>(batch - 1) * max_width_ + (W - 1)], [&] {
        forward_->run_mtp({.residuals   = Tensor(mtp_residuals_.p, DType::BF16, {width_, W * batch}),
                           .ids         = Tensor(mtp_device(mtp_io_.up_ids), DType::I32, {W * batch}),
                           .positions   = positions,
                           .rope_positions   = rope,
                           .block_start_rope = block_rope,
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
    // `before_wait` runs on the host after the sampling work is queued, before its wait.
    void sample(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions,
                const std::function<void()>& before_wait = {}) {
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
            configs[b].mask     = grammar_mask(static_cast<std::size_t>(b), {});
            sample_positions[b] = positions[b];
        }
        upload_pinned(configs_.p, configs, sizeof(ops::SamplingConfig) * batch, s);
        upload_pinned(sample_pos_.p, sample_positions, 4ULL * batch, s);
        const bool constrained = stage_constraints(lanes, 1);
        if (constrained) { constrain(narrow, nullptr); }
        Tensor out(sampled_.p, DType::I32, {batch});
        Tensor logical(sample_pos_.p, DType::I32, {batch});
        {
            auto scope = work_->scope();
            ops::sample(narrow, out, token_domain_, static_cast<const ops::SamplingConfig*>(configs_.p), logical,
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

    std::int32_t vocab_ = 0, token_domain_ = 0, chunk_ = 0, columns_ = 0, pages_per_row_ = 0;
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

private:
    void prefix_pin(const PrefixSelection& selection);
    void prefix_unpin(const PrefixSelection& selection) noexcept;
    void prefix_activate(Lane& lane, std::uint32_t index, const PrefixSelection& selection,
                         DeviceKVPageReservation& reservation, std::uint32_t pages);
    [[nodiscard]] std::uint32_t prefix_frontier(const Lane& lane, bool finishing) const noexcept;
    void prefix_publish(Lane& lane, std::uint32_t frontier);
    void prefix_capture(Lane& lane, std::uint32_t index, runtime::prefix_cache::SnapshotKind kind, bool hand_over_tail);
    void prefix_after_prefill_call(Lane& lane, std::uint32_t index, bool last);
    void prefix_finish(Lane& lane, std::uint32_t index);
    void prefix_release(Lane& lane) noexcept;
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

public:
    [[nodiscard]] HybridPrefixCacheStats prefix_stats() const noexcept;

private:
    // The prompt's prefill calls from `frontier`, and with `taps` the prefix taps they realize.
    [[nodiscard]] prefix::CallPlan plan_prefill(const qwen3_5::PreparedPromptData& prompt, std::uint32_t frontier,
                                                std::span<const std::uint32_t> existing, bool taps) const;
    DeviceBuffer state_backing_, ple_backing_, tails_backing_, kv_backing_, staging_;
    // The vq2/k4v2 exact window (design §19.3.15), every KV layer and lane.
    DeviceBuffer window_backing_;
    prefix::KvWindowGeometry window_geometry_;

    // ---- the host-to-device link (design §19.4: bandwidth-dependent choices follow the machine) ----
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
                      "host-to-device link: %.1f GB/s%s; decode misses kept on the link: 1/%d; decode promotes every "
                      "%llu tokens; prefill CPU split up to %d columns",
                      rate / 1e9, options_.link_bytes_per_second > 0.0 ? " (set)" : " measured", pcie_divisor(),
                      static_cast<unsigned long long>(promotion_interval()),
                      options_.prefill_cpu_split ? split_columns() : 0);
        diagnostic(text, DiagnosticLevel::Debug);
        std::snprintf(text, sizeof(text), "PCIe link to the GPU: %.1f GB/s%s", rate / 1e9,
                      options_.link_bytes_per_second > 0.0 ? " (fixed)" : " measured");
        diagnostic(text);
    }

    // The share of a decode call's misses the PCIe stage keeps (misses / divisor): 3 measured
    // fastest on the reference link; a faster link carries more of them (x16: 2), a slower one fewer.
    [[nodiscard]] int pcie_divisor() const noexcept {
        if (options_.cpu_pcie_divisor > 0) { return options_.cpu_pcie_divisor; }
        return std::max(2, static_cast<int>(std::lround(1.0 + 2.0 * kReferenceLinkBytesPerSecond / link_bytes_per_second_)));
    }

    // The widest call the prefill CPU split takes (ProgramOptions::cpu_split_columns on the reference
    // link, inversely to the measured one; at most the CPU channel's call width).
    [[nodiscard]] std::int32_t split_columns() const noexcept {
        const double scaled = options_.cpu_split_columns * kReferenceLinkBytesPerSecond / link_bytes_per_second_;
        return static_cast<std::int32_t>(std::min<double>(scaled, ops::offloaded_moe::kMaxCpuCallColumns));
    }

public:
    [[nodiscard]] double link_bytes_per_second() const noexcept { return link_bytes_per_second_; }

private:
    // ---- elastic KV (design §19.3.11) ----
    static constexpr std::size_t kKvChunkBytes   = 2ULL << 20;
    static constexpr std::uint32_t kKvBaseTokens = 32768; // backed at startup
    static constexpr std::uint32_t kKvGrowPages  = 64;    // growth unit (4,096 tokens)
    static constexpr std::chrono::seconds kKvIdleShrink{60};
    // The last admission, release or idle shrink (steady-clock ticks), and whether the pool backs more
    // than its base: the monitor thread reads both to wake an idle engine for the shrink.
    std::atomic<std::int64_t> kv_busy_at_{0};
    std::atomic<bool> kv_grown_{false};
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
    // frames (never below three quarters of them). False when it cannot: a lease holds the frames, or
    // the pool is at its capacity.
    bool grow_kv(std::uint32_t need) {
        if (!kv_vmm_ || pool_->available_pages() >= need) { return pool_->available_pages() >= need; }
        const std::uint32_t backed = pool_->backed_pages();
        std::uint32_t target       = backed + (need - pool_->available_pages());
        target = std::min(pool_->capacity_pages(), (target + kKvGrowPages - 1) / kKvGrowPages * kKvGrowPages);
        if (target - backed < need - pool_->available_pages()) { return false; }
        const std::lock_guard<std::mutex> lock(vram_mutex_);
        const std::size_t added    = kv_mapped_bytes(plan_, target) - kv_vmm_->mapped_bytes();
        const std::uint64_t stride = residency_->frame_stride();
        const std::uint32_t frames = residency_->frames();
        // Memory the sizing function would give the frames anyway comes first; the rest is taken from
        // the frames, one frame chunk over so the pool unmaps at least that much.
        const std::uint32_t spare =
            control_ ? std::max<std::uint32_t>(control_->target(vram_->query(), pool_now_), frames) - frames : 0;
        const std::uint64_t spare_bytes = static_cast<std::uint64_t>(spare) * stride;
        std::uint32_t give              = 0;
        if (added > spare_bytes) {
            give = static_cast<std::uint32_t>((added - spare_bytes + ExpertResidency::kChunkBytes + stride - 1) / stride);
        }
        if (give > frames / 4) { return false; } // grow by at most a quarter of the cache at once
        device_.synchronize();
        if (give != 0) {
            const auto resized = residency_->resize(frames - give, device_.stream, *vram_);
            if (resized.frames > frames - give) { return false; } // a lease holds the frames
        }
        if (!map_kv(target)) {
            if (give != 0) { (void)residency_->resize(frames, device_.stream, *vram_); }
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
        std::uint32_t target = pool_->backed_pages();
        while (target > plan_.kv_base_pages && pool_->can_back(target - std::min(target, kKvGrowPages))) {
            target -= std::min(target, kKvGrowPages);
        }
        target = std::max(target, plan_.kv_base_pages);
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
        const std::uint64_t stride = residency_->frame_stride();
        const std::uint32_t frames = residency_->frames();
        (void)residency_->resize(frames + static_cast<std::uint32_t>(freed / stride), device_.stream, *vram_);
        if (control_) { control_->account_fixed(-static_cast<std::int64_t>(freed)); }
        frames_now_ = residency_->frames();
        pool_now_   = residency_->pool_bytes();
        diagnostic("KV pool shrank to " + std::to_string(target) + " pages; expert cache " + std::to_string(frames) +
                   " -> " + std::to_string(residency_->frames()) + " frames");
    }
    DeviceBuffer io_device_, logits32_, logits16_, sampled_, sample_pos_, configs_, token_counts_;

    // ---- /v1/decide (constraint.cpp) ----
    // The constraint controls: each lane's kMaximumConstraintSets sets of up to
    // ops::kTokenConstraintChoices tokens and their counts, then one descriptor and one probability
    // record per round column (lanes x max width).
    static std::size_t constraint_bytes(std::int32_t lanes, std::int32_t width) noexcept {
        const auto sets    = static_cast<std::size_t>(lanes) * kMaximumConstraintSets;
        const auto columns = static_cast<std::size_t>(lanes) * static_cast<std::size_t>(width);
        return 4ULL * (sets * ops::kTokenConstraintChoices + sets + columns + columns * ops::kTokenConstraintRecord);
    }
    [[nodiscard]] std::shared_ptr<const ConstraintPlan> compile_constraint(const TokenConstraint& constraint);
    // Writes the round's descriptors (`width` columns per row, rows in `lanes` order) and the
    // constrained lanes' sets; false (and nothing written) when no row is constrained.
    bool stage_constraints(std::span<const std::uint32_t> lanes, std::int32_t width);
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
        CUDA_CHECK(cudaMemcpyAsync(device_words, host.data(), host.size_bytes(), cudaMemcpyHostToDevice,
                                   device_.stream));
        grammar_->uploaded(row, host.size_bytes());
        return {device_words, static_cast<std::int32_t>(words)};
    }
    // A row fails when the tokens it emits reach a dead-end position (one mask position per token).
    void mark_grammar_failures(PendingBatch& pending, std::span<const std::int32_t> counts) const noexcept {
        for (std::size_t row = 0; row < pending.row_count(); ++row) {
            const auto count = counts.empty() ? 1 : counts[row];
            const std::uint64_t reached = count >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << count) - 1U;
            ContractAccess::set_constraint_failed(pending, row, (grammar_dead_[row] & reached) != 0);
        }
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
    IoLayout io_layout_;
    std::unique_ptr<LinearAttentionStatePool> gdn_;
    prefix::KvPageGeometry kv_geometry_;
    std::unique_ptr<DeviceKVPagePool> pool_;
    std::unique_ptr<KVExecutionTablePool> tables_;
    std::size_t work_capacity_ = 0;
    std::unique_ptr<WorkspaceArena> work_;
    // SSD tier mode (design §19.3.7): the fetch channel and the tier outlive everything that reads
    // their memory (the residency's promotion copies, the stream, the CPU service), declared later.
    std::unique_ptr<ops::offloaded_moe::FetchChannel> fetch_channel_;
    std::unique_ptr<HostExpertTier> tier_;
    PinnedHostBuffer expert_error_{1}; // mapped u32: ops::MoeExpertSource::error of every call
    bool clear_prefix_on_release_ = false;
    std::unique_ptr<ExpertResidency> residency_;
    std::unique_ptr<execution::ExpertStream> expert_stream_;
    ExpertResidency::FrameLease stream_lease_;
    // The stream lease: the ring, the wide arena, then a walk area.
    std::size_t stream_ring_bytes_ = 0, stream_wide_bytes_ = 0, stream_walk_bytes_ = 0;
    std::size_t wide_work_ = 0;              // the plan's wide arena bytes (0: none)
    ExpertResidency::FrameLease work_lease_; // the wide arena lent alone (no stream lease)
    DeviceBuffer wide_fallback_;             // the wide arena when nothing could be lent
    std::unique_ptr<WorkspaceArena> wide_arena_; // the arena swapped with *work_ around a wide call
    execution::ExpertStream::Stats stream_at_lease_;
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
    static constexpr std::uint32_t kWalkMaxTokens = 65536;

    [[nodiscard]] std::size_t walk_bytes(std::uint32_t tokens, std::size_t chunks) const noexcept;
    std::size_t walk_span_end(const Lane& lane, std::uint32_t index);
    void walk_begin(Lane& lane, std::uint32_t index, std::size_t end);
    PrefillProgress walk_step(Lane& lane, std::uint32_t index, Clock::time_point start);
    void walk_pass(std::size_t chunk, std::uint32_t layer);
    void walk_enqueue(std::uint32_t to_layer);
    void walk_abandon() noexcept;
    [[nodiscard]] bool prefix_tap_due(const Lane& lane, std::size_t next_call) const;
    std::unique_ptr<RouteTrace> trace_;
    std::unique_ptr<ops::offloaded_moe::CpuMissService> cpu_service_;

    // ---- prefill CPU split (design §19.3.12) ----
    // Gives the CPU the thinnest non-resident experts of a layer while the link carries the rest:
    // narrow ones (n <= 8) in order of n, then expert id, as many as minimise the later of the
    // CPU's predicted time and the link's time for the experts left to it.
    class SplitPolicy final : public execution::CpuSplitPolicy {
    public:
        explicit SplitPolicy(const ProgramImpl& program) : program_(program) {}
        std::uint32_t choose(std::uint32_t layer, std::span<const std::int32_t> columns,
                             std::span<std::uint8_t> cpu) const override {
            const auto& p              = program_;
            const std::uint32_t E      = p.c_.moe.experts;
            const std::int32_t* frames = p.residency_->host_table() + static_cast<std::size_t>(layer) * E;
            const double link          = static_cast<double>(p.residency_->frame_stride()) / p.link_bytes_per_second_;
            const double per_expert    = 1.0 / p.options_.cpu_split_expert_rate;
            const double per_column    = 1.0 / p.options_.cpu_split_column_rate;
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
    std::array<std::int32_t, kMaximumConcurrency> mtp_confident_{}; // per row: drafts above kDraftMinLogprob
    std::array<bool, kMaximumConcurrency> from_ngram_{};
    // Draft-length policy: acceptance prior and EWMA weight, and the cost of each draft column in
    // plain rounds (warm tg512 on the 5090: W = 3, 4, 5 rounds took 1.84, 2.13 and 2.54 plain
    // rounds, design 19.2).
    static constexpr double kAcceptancePrior      = 0.75;
    static constexpr double kAcceptanceWeight     = 0.1;
    static constexpr double kWidthCost            = 0.38;
    static constexpr std::uint64_t kProbeInterval = 8;
    // Two-row rounds: at most 2 drafts per row, each costing kPairWidthCost plain two-row rounds
    // (C = 2 decode on the 5090, AIME prompts sampled: K = 1 and 2 rounds took 1.44 and 1.95 plain
    // two-row rounds; K = 3, 8 columns, took 3.1 and lost to plain decode; design 19.3.5).
    static constexpr std::int32_t kPairMaxDrafts = 2;
    static constexpr double kPairWidthCost       = 0.47;
    // A one-row round verifies its drafts up to the first whose draft-head probability is below 0.5
    // (Strata's --spec-min-p rule; C = 1 MTP, AIME prompts sampled on the 5090: 138.8 -> 147.2 tok/s,
    // acceptance 75.6 -> 81.8 %, repeatable to the decimal; 0.7 gave 142.6; design 19.3.5).
    static constexpr double kDraftMinLogprob = -0.6931471805599453; // log(0.5)
    std::uint64_t pair_policy_rounds_            = 0;
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
    // The n-gram gate of verification rounds (design §12.3): the pinned ready word, its last
    // published value (strictly increasing, never 0) and per-lane device wait statistics.
    PinnedHostBuffer gate_host_{64};
    std::uint32_t gate_sequence_ = 0;
    DeviceBuffer gate_stats_;
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
