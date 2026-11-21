#include "models/inkling2/program/program.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "infernix/ops/inkling2_attention.h"
#include "infernix/ops/sampling.h"
#include "models/inkling2/execution/dspark.h"
#include "models/inkling2/execution/forward.h"
#include "runtime/speculative/draft_length.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::models::inkling2 {
namespace detail {

using Clock = std::chrono::steady_clock;

struct BasePlanImpl {
    std::vector<TokenId> tokens;
    runtime::RequestPlanSummary summary;
    runtime::ResolvedExecutionOptions options;
};

struct ContractAccess {
    static SequenceHandle handle(const void* owner, std::uint32_t lane, std::uint64_t epoch) {
        SequenceHandle s;
        s.owner_ = owner;
        s.lane_  = lane;
        s.epoch_ = epoch;
        return s;
    }
    static const void* owner(const SequenceHandle& s) noexcept { return s.owner_; }
    static std::uint32_t lane(const SequenceHandle& s) noexcept { return s.lane_; }
    static std::uint64_t epoch(const SequenceHandle& s) noexcept { return s.epoch_; }

    // `counts` empty: one token a row (stride 1); else row r's counts[r] tokens at tokens[r * stride ..].
    static PendingBatch batch(const void* owner, std::uint64_t transaction, std::span<const SequenceHandle> rows,
                              std::span<const TokenId> tokens, runtime::ExecutionTiming timing,
                              std::span<const runtime::PendingRowFailure> failures,
                              std::span<const std::int32_t> counts = {}, std::uint32_t stride = 1) {
        PendingBatch b;
        b.owner_       = owner;
        b.transaction_ = transaction;
        b.row_count_   = rows.size();
        for (std::size_t i = 0; i < rows.size(); ++i) {
            b.rows_[i]         = rows[i];
            b.row_failures_[i] = failures[i];
        }
        b.tokens_     = tokens;
        b.row_counts_ = counts;
        b.row_stride_ = stride;
        b.timing_     = timing;
        return b;
    }
    static const void* owner(const PendingBatch& b) noexcept { return b.owner_; }
    static std::uint64_t transaction(const PendingBatch& b) noexcept { return b.transaction_; }
    static std::span<const SequenceHandle> rows(const PendingBatch& b) noexcept { return {b.rows_.data(), b.row_count_}; }
    static void consume(PendingBatch& b) noexcept {
        b.owner_     = nullptr;
        b.row_count_ = 0;
        b.tokens_    = {};
    }
};

namespace {

std::uint64_t elapsed_ns(Clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

// The forward's call capacity: the prefill chunk (no wider than the context: a call never holds more prompt
// positions), and at least the decode and verify calls' width.
std::int32_t forward_capacity(const ProgramOptions& o) {
    return std::max(kMaxForwardColumns, static_cast<std::int32_t>(std::min(o.prefill_chunk, o.max_context)));
}

std::uint32_t pages_per_lane(std::uint32_t max_context) {
    return (max_context + ops::inkling2::kKeyUnit - 1) / ops::inkling2::kKeyUnit + 1;
}

// One full slot's K or V pool for `pages` pages: [kKvHeads][pages][64][128] 16-bit words.
std::uint64_t pool_bytes(std::uint64_t pages) {
    return std::uint64_t{ops::inkling2::kKvHeads} * pages * ops::inkling2::kKeyUnit * ops::inkling2::kHeadDim * 2;
}

std::uint64_t align_256(std::uint64_t bytes) { return (bytes + 255) / 256 * 256; }

} // namespace

class ProgramImpl {
public:
    ProgramImpl(const Parameters& parameters, DeviceContext& device, ProgramOptions options)
        : p_(parameters), c_(parameters.model.config().text), device_(device), o_(std::move(options)),
          plan_(Program::plan_device(o_, parameters.model.config())),
          forward_(parameters, static_cast<std::int32_t>(o_.max_concurrency), forward_capacity(o_)),
          vocab_(static_cast<std::int32_t>(c_.vocab_size)), lanes_(o_.max_concurrency),
          sampling_(ops::sampling_workspace_capacity_bytes(vocab_, 1, static_cast<std::int32_t>(kMaximumConcurrency)) +
                    4096),
          host_(align_256(forward_.upload_words() * 4 * kStagingSlots) +
                align_256(sizeof(ops::SamplingConfig) * kMaximumConcurrency) + 3 * align_256(4 * kMaximumConcurrency)) {
        const StateLayout& layout = forward_.layout();
        const std::uint32_t pages = pages_per_lane(o_.max_context);
        images_ = DeviceBuffer(static_cast<std::size_t>(layout.lane_words) * 2 * lanes_.size());
        for (std::int32_t s = 0; s < 2 * layout.full_visits; ++s) {
            pools_.emplace_back(pool_bytes(static_cast<std::uint64_t>(pages) * lanes_.size()));
            pools_.back().fill(0); // a key beyond a lane's frontier is masked; never NaN
        }
        std::vector<std::int32_t> rows(static_cast<std::size_t>(pages) * lanes_.size());
        for (std::size_t i = 0; i < rows.size(); ++i) { rows[i] = static_cast<std::int32_t>(i); }
        table_ = DeviceBuffer(rows.size() * 4);
        table_.copy_from_host(rows.data(), rows.size() * 4);
        view_.images     = static_cast<std::uint16_t*>(images_.p);
        view_.lane_words = layout.lane_words;
        for (std::int32_t s = 0; s < layout.full_visits; ++s) {
            view_.pool_k.push_back(static_cast<std::uint16_t*>(pools_[2 * s].p));
            view_.pool_v.push_back(static_cast<std::uint16_t*>(pools_[2 * s + 1].p));
        }
        view_.pool_pages   = static_cast<std::int64_t>(pages) * lanes_.size();
        view_.table        = static_cast<const std::int32_t*>(table_.p);
        view_.table_stride = static_cast<std::int32_t>(pages);
        counts_            = DeviceBuffer(static_cast<std::size_t>(vocab_) * 4 * lanes_.size());
        configs_           = DeviceBuffer(sizeof(ops::SamplingConfig) * kMaximumConcurrency);
        positions_         = DeviceBuffer(4 * kMaximumConcurrency);
        sampled_           = DeviceBuffer(4 * kMaximumConcurrency);
        auto* base         = static_cast<std::byte*>(host_.data());
        staging_           = reinterpret_cast<std::int32_t*>(base);
        base += align_256(forward_.upload_words() * 4 * kStagingSlots);
        host_configs_ = reinterpret_cast<ops::SamplingConfig*>(base);
        base += align_256(sizeof(ops::SamplingConfig) * kMaximumConcurrency);
        host_positions_ = reinterpret_cast<std::int32_t*>(base);
        base += align_256(4 * kMaximumConcurrency);
        host_sampled_ = reinterpret_cast<std::int32_t*>(base);
        base += align_256(4 * kMaximumConcurrency);
        nonfinite_ = reinterpret_cast<std::int32_t*>(base);
        if (o_.dspark && !o_.causal_scoring) {
            if (!p_.dspark) { throw std::invalid_argument("Inkling 2: --spec dspark needs the drafter's parameters"); }
            if (o_.draft_tokens < 1 || o_.draft_tokens > static_cast<std::uint32_t>(ops::inkling2::kMarkovSteps)) {
                throw std::invalid_argument("Inkling 2: DSpark drafts 1-7 tokens a round");
            }
            drafter_ = std::make_unique<Drafter>(p_, forward_, static_cast<std::int32_t>(lanes_.size()), o_.max_context,
                                                 forward_capacity(o_));
        }
        if (o_.use_cuda_graph && !o_.causal_scoring) { capture_decode_graphs(); }
    }

    // ---------------------------------------------------------------- admission
    RequestBasePlan plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options) {
        if (!prompt) { throw std::invalid_argument("Inkling 2: an empty prepared prompt"); }
        std::unique_ptr<PreparedPromptData> data = std::move(prompt).take();
        const auto n                             = static_cast<std::uint32_t>(data->token_ids.size());
        if (n == 0 || n > o_.max_context) { throw std::invalid_argument("Inkling 2: the prompt is empty or exceeds max_context"); }
        if (!options.constraint.empty()) {
            throw std::invalid_argument("Inkling 2: token constraints are not available yet");
        }
        for (const TokenId id : options.readout_tokens) {
            if (id < 0 || id >= vocab_) { throw std::invalid_argument("readout token is outside the public token domain"); }
        }
        auto base                             = std::make_unique<BasePlanImpl>();
        base->tokens                          = std::move(data->token_ids);
        base->options                         = options;
        base->summary.prompt_tokens           = n;
        base->summary.requested_output_tokens = options.requested_output_tokens;
        const std::uint32_t capacity_output   = o_.max_context - n + 1U;
        base->summary.effective_output_tokens = std::min(options.requested_output_tokens, capacity_output);
        base->summary.effective_limit_reason  = options.requested_output_tokens <= capacity_output
                                                    ? FinishReason::OutputLimit
                                                    : FinishReason::ContextCapacity;
        base->summary.publish_continuation    = false;
        return RequestBasePlan(std::move(base));
    }

    std::vector<SourceCandidate> sources(const ResumeState* resume) const {
        if (resume != nullptr) { throw std::logic_error("Inkling 2: no request is paused"); }
        return {SourceCandidate{}};
    }

    BindingReservation start_binding(const RequestBasePlan& base, runtime::LaneId lane_id, const ResumeState* resume) {
        if (resume != nullptr) { throw std::logic_error("Inkling 2: no request is paused"); }
        if (binding_) { throw std::logic_error("Inkling 2: a binding is already open"); }
        if (lane_id.value >= lanes_.size() || lanes_[lane_id.value].bound) {
            throw std::logic_error("Inkling 2: the binding's lane is not free");
        }
        Lane& lane         = lanes_[lane_id.value];
        lane               = Lane{};
        lane.bound         = true;
        lane.epoch         = ++epoch_;
        lane.tokens        = base.impl_->tokens;
        lane.prompt_tokens = base.impl_->summary.prompt_tokens;
        lane.options       = base.impl_->options;
        lane.penalties     = lane.options.sampling.presence_penalty != 0.0F || lane.options.sampling.frequency_penalty != 0.0F;
        lane.calibration.reset();
        // A fresh state: the conv rings' history before position 0 is zero, and the window's rows are rewritten.
        CUDA_CHECK(cudaMemsetAsync(static_cast<std::uint16_t*>(images_.p) +
                                       static_cast<std::size_t>(lane_id.value) * forward_.layout().lane_words,
                                   0, static_cast<std::size_t>(forward_.layout().lane_words) * 2, device_.stream));
        if (lane.penalties) {
            CUDA_CHECK(cudaMemsetAsync(lane_counts(lane_id.value), 0, static_cast<std::size_t>(vocab_) * 4,
                                       device_.stream));
        }
        binding_ = lane_id.value;
        BindingReservation out;
        out.reserved = true;
        return out;
    }

    ContextProgress poll_context() {
        if (!binding_) { throw std::logic_error("Inkling 2: no binding is open"); }
        const std::uint32_t index = *binding_;
        binding_.reset();
        ContextProgress out;
        out.kind      = ContextOperationKind::Bind;
        out.advanced  = true;
        out.complete  = true;
        out.published = true; // the sequence is live (the Engine cancels a binding that publishes none)
        out.sequence  = handle(index);
        return out;
    }

    [[nodiscard]] bool has_context_transaction() const noexcept { return binding_.has_value(); }

    // ---------------------------------------------------------------- execution
    PrefillProgress advance_prefill(SequenceHandle sequence, runtime::TokenMaskProvider* masks) {
        const auto start = Clock::now();
        if (pending_) { throw std::logic_error("Inkling 2: a pending batch awaits its commit"); }
        const std::uint32_t index = lane_of(sequence);
        Lane& lane                = lanes_[index];
        if (lane.processed >= lane.prompt_tokens) { throw std::logic_error("Inkling 2: the prompt is already prefilled"); }
        const std::uint32_t chunk = std::min<std::uint32_t>(o_.prefill_chunk, lane.prompt_tokens - lane.processed);
        const bool last           = lane.processed + chunk == lane.prompt_tokens;
        CallPlan call;
        // first: the sequence's first position (every lane's sequence starts at 0), not the call's first column.
        call.first.assign(index + 1, 0);
        call.prompt = true;
        for (std::uint32_t i = 0; i < chunk; ++i) {
            call.tokens.push_back(lane.tokens[lane.processed + i]);
            call.lane.push_back(static_cast<std::int32_t>(index));
            call.position.push_back(static_cast<std::int32_t>(lane.processed + i));
        }
        upload(call);
        run_forward(static_cast<std::int32_t>(chunk) - 1, last ? 1 : 0);
        if (drafter_) { drafter_->append(0, static_cast<std::int32_t>(chunk), device_.stream); } // the chunk's context rows
        lane.processed += chunk;
        PrefillProgress out;
        out.summary                 = {.prompt_tokens = lane.prompt_tokens};
        out.processed_prompt_tokens = chunk; // this unit's tokens (the Engine sums them)
        out.complete                = last;
        if (last) {
            const std::uint64_t transaction = ++transaction_;
            const SequenceHandle row        = handle(index);
            // The head ran over the chunk's last column only: its logits are row 0.
            sample({&row, 1}, {&index, 1}, 0, ops::kSamplePurposePrefill, masks);
            if (!lane.options.readout_tokens.empty()) { out.readout = readout(lane, 0); }
            out.timing = timing_since(start);
            out.pending.emplace(ContractAccess::batch(this, transaction, {&row, 1}, {sampled_tokens_.data(), 1},
                                                      out.timing, {failures_.data(), 1}));
            pending_      = transaction;
            pending_kind_ = PendingKind::Prefill;
        } else {
            out.timing = timing_since(start);
        }
        lane.prefill_ns += elapsed_ns(start);
        return out;
    }

    PendingBatch decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
                        runtime::TokenMaskProvider* masks) {
        const auto start = Clock::now();
        if (pending_) { throw std::logic_error("Inkling 2: a pending batch awaits its commit"); }
        if (sequences.empty() || sequences.size() > lanes_.size()) { throw std::invalid_argument("Inkling 2: a decode of 1-8 rows"); }
        // A row's round licenses at most its remaining generated tokens: a DSpark round needs two of them.
        const std::uint32_t remaining = budgets.empty() ? 0U : budgets[0].generated_tokens_remaining;
        if (remaining >= 2 && dspark_round(sequences, masks)) { return decode_dspark(sequences[0], remaining, masks, start); }
        if (drafter_ && sequences.size() == 1) { ++lanes_[lane_of(sequences[0])].fallback; } // a plain round of a drafting row
        std::vector<std::uint32_t> rows;
        for (const SequenceHandle& s : sequences) { rows.push_back(lane_of(s)); }
        // A lane's columns are consecutive and lanes ascend in the call; rows keep the Engine's order.
        std::vector<std::uint32_t> order = rows;
        std::sort(order.begin(), order.end());
        CallPlan call;
        call.first.assign(order.back() + 1, 0);
        bool uniform = true;
        std::map<std::uint32_t, std::int32_t> last_column;
        for (const std::uint32_t index : order) {
            Lane& lane = lanes_[index];
            if (lane.processed >= lane.tokens.size()) { throw std::logic_error("Inkling 2: a decode row has no input"); }
            const auto inputs = static_cast<std::uint32_t>(lane.tokens.size() - lane.processed);
            if (inputs > static_cast<std::uint32_t>(ops::inkling2::kMaxAttentionLaneColumns)) {
                throw std::logic_error("Inkling 2: more than 16 unprocessed tokens in a lane");
            }
            uniform = uniform && inputs == 1;
            for (std::uint32_t i = 0; i < inputs; ++i) {
                call.tokens.push_back(lane.tokens[lane.processed + i]);
                call.lane.push_back(static_cast<std::int32_t>(index));
                call.position.push_back(static_cast<std::int32_t>(lane.processed + i));
            }
            last_column[index] = static_cast<std::int32_t>(call.tokens.size()) - 1;
        }
        upload(call);
        const auto columns = static_cast<std::int32_t>(call.tokens.size());
        std::vector<std::int32_t> logit_rows;
        for (const std::uint32_t index : rows) { logit_rows.push_back(last_column[index]); }
        // Rows sample in the Engine's order; their logits rows in the call's column order.
        const std::uint64_t transaction = ++transaction_;
        if (uniform && o_.use_cuda_graph) {
            decode_graph(static_cast<std::int32_t>(order.back()) + 1, columns, rows, logit_rows, masks);
        } else {
            run_decode_call(columns);
            sample(sequences, rows, logit_rows, ops::kSamplePurposeDecode, masks);
        }
        for (const std::uint32_t index : order) {
            Lane& lane     = lanes_[index];
            lane.processed = static_cast<std::uint32_t>(lane.tokens.size());
        }
        const std::uint64_t ns = elapsed_ns(start);
        for (const std::uint32_t index : rows) {
            lanes_[index].decode_ns += ns;
            lanes_[index].decode_share_ns += ns / rows.size();
        }
        pending_      = transaction;
        pending_kind_ = PendingKind::Decode;
        return ContractAccess::batch(this, transaction, sequences, {sampled_tokens_.data(), sequences.size()},
                                     timing_since(start), {failures_.data(), sequences.size()});
    }

    runtime::ExecutionTiming append_forced(std::span<const SequenceHandle> sequences, std::span<const TokenId> tokens,
                                           std::uint32_t stride) {
        const auto start = Clock::now();
        if (pending_) { throw std::logic_error("Inkling 2: a pending batch awaits its commit"); }
        if (stride == 0 || tokens.size() != sequences.size() * stride) {
            throw std::invalid_argument("Inkling 2: forced tokens are row-major, `stride` a row");
        }
        for (std::size_t row = 0; row < sequences.size(); ++row) {
            Lane& lane = lanes_[lane_of(sequences[row])];
            for (std::uint32_t i = 0; i < stride; ++i) {
                const TokenId id = tokens[row * stride + i];
                if (id < 0 || id >= vocab_) { throw std::invalid_argument("Inkling 2: a forced token is outside the vocabulary"); }
                lane.tokens.push_back(id); // processed by the lane's next decode call, with its pending input
            }
        }
        return timing_since(start);
    }

    CommitResult commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions) {
        const auto start = Clock::now();
        if (ContractAccess::owner(pending) != this || !pending_ || ContractAccess::transaction(pending) != *pending_) {
            throw std::logic_error("Inkling 2: commit of a foreign or stale pending batch");
        }
        const auto rows = ContractAccess::rows(pending);
        if (decisions.size() != rows.size()) { throw std::logic_error("Inkling 2: commit decisions are not row aligned"); }
        CommitResult out;
        out.row_count = rows.size();
        for (std::size_t row = 0; row < rows.size(); ++row) {
            const std::uint32_t index = ContractAccess::lane(rows[row]);
            Lane& lane                = lanes_[index];
            const auto& d             = decisions[row];
            if ((d.cancelled && d.failed) || (d.failed && (d.accepted_tokens != 0 || !d.terminal)) ||
                (!d.cancelled && d.failed != (failures_[row] != runtime::PendingRowFailure::None))) {
                throw std::logic_error("Inkling 2: a commit decision is invalid");
            }
            if (d.cancelled || d.failed) {
                out.rows[row].disposition = d.failed ? runtime::CommitDisposition::FailedReleased
                                                     : runtime::CommitDisposition::CancelledReleased;
                continue;
            }
            if (pending_draft_.active) {
                // A DSpark row: its accepted prefix; the lane holds positions F .. F + accepted - 1, its last token
                // pending.
                if (d.accepted_tokens == 0 || d.accepted_tokens > static_cast<std::uint32_t>(row_counts_[0])) {
                    throw std::logic_error("Inkling 2: a DSpark row commits 1 .. its licensed tokens");
                }
                for (std::uint32_t i = 0; i < d.accepted_tokens; ++i) { lane.tokens.push_back(sampled_tokens_[i]); }
                lane.processed = pending_draft_.first + d.accepted_tokens;
            } else {
                if (d.accepted_tokens != 1) { throw std::logic_error("Inkling 2: a row commits exactly one token"); }
                lane.tokens.push_back(sampled_tokens_[row]);
            }
            out.rows[row].disposition =
                d.terminal ? runtime::CommitDisposition::Finishable : runtime::CommitDisposition::Active;
            speculative_of(lane, out.rows[row]);
        }
        pending_draft_ = {};
        const std::uint64_t ns = elapsed_ns(start);
        for (std::size_t row = 0; row < rows.size(); ++row) {
            Lane& lane = lanes_[ContractAccess::lane(rows[row])];
            if (pending_kind_ == PendingKind::Prefill) {
                lane.prefill_ns += ns;
            } else {
                lane.decode_ns += ns;
                lane.decode_share_ns += ns / rows.size();
            }
        }
        for (std::size_t row = 0; row < rows.size(); ++row) {
            if (decisions[row].cancelled || decisions[row].failed) {
                const std::uint32_t index = ContractAccess::lane(rows[row]);
                out.rows[row].timings     = timings(lanes_[index]);
                release(index);
            }
        }
        out.timing = timing_since(start);
        ContractAccess::consume(pending);
        pending_.reset();
        return out;
    }

    DiscardResult discard(PendingBatch&& pending) noexcept {
        DiscardResult out;
        if (ContractAccess::owner(pending) != this || !pending_ || ContractAccess::transaction(pending) != *pending_) {
            return out;
        }
        out.row_count = ContractAccess::rows(pending).size();
        out.status    = runtime::ConsumeStatus::Consumed;
        ContractAccess::consume(pending);
        pending_.reset();
        pending_draft_ = {};
        return out;
    }

    FinishResult finish(SequenceHandle sequence) noexcept {
        FinishResult out;
        if (!valid(sequence)) { return out; }
        const std::uint32_t index = ContractAccess::lane(sequence);
        out.timings               = timings(lanes_[index]);
        out.status                = runtime::ConsumeStatus::Consumed;
        CommitRowResult stats;
        speculative_of(lanes_[index], stats);
        out.speculative = std::move(stats.speculative);
        release(index);
        return out;
    }

    AbortResult abort(SequenceHandle sequence) noexcept {
        AbortResult out;
        if (!valid(sequence)) { return out; }
        const std::uint32_t index = ContractAccess::lane(sequence);
        out.timings               = timings(lanes_[index]);
        out.status                = runtime::ConsumeStatus::Consumed;
        CommitRowResult stats;
        speculative_of(lanes_[index], stats);
        out.speculative = std::move(stats.speculative);
        release(index);
        return out;
    }

    void release_all() noexcept {
        for (std::uint32_t i = 0; i < lanes_.size(); ++i) { release(i); }
        binding_.reset();
        pending_.reset();
    }

    bool recover() noexcept {
        try {
            release_all();
            CUDA_CHECK(cudaStreamSynchronize(device_.stream));
            return true;
        } catch (...) {
            return false;
        }
    }

    [[nodiscard]] bool context_blocks(SequenceHandle sequence) const noexcept {
        return binding_.has_value() && valid(sequence) && *binding_ == ContractAccess::lane(sequence);
    }

    // ---------------------------------------------------------------- scoring
    ScoreResult causal_score(PreparedPrompt&& prompt, std::uint32_t first_target, const ScoreOptions& options) {
        if (!o_.causal_scoring) { throw std::logic_error("Inkling 2: causal scoring needs a scoring Program"); }
        if (binding_ || pending_ || std::any_of(lanes_.begin(), lanes_.end(), [](const Lane& l) { return l.bound; })) {
            throw std::logic_error("Inkling 2: causal scoring needs an idle Program");
        }
        std::unique_ptr<PreparedPromptData> data = std::move(prompt).take();
        const std::vector<TokenId>& tokens       = data->token_ids;
        const auto n                             = static_cast<std::uint32_t>(tokens.size());
        if (n < 2 || first_target == 0 || first_target >= n || n > o_.max_context) {
            throw std::invalid_argument("Inkling 2: scoring needs 2..max_context tokens and 0 < first_target < count");
        }
        if (options.candidates.size() != static_cast<std::size_t>(options.candidates_per_position) * (n - first_target)) {
            throw std::invalid_argument("Inkling 2: candidates must hold candidates_per_position for each scored position");
        }
        CUDA_CHECK(cudaMemsetAsync(images_.p, 0, static_cast<std::size_t>(forward_.layout().lane_words) * 2, device_.stream));
        ScoreResult out;
        std::vector<float> row(static_cast<std::size_t>(vocab_));
        std::size_t next_logits = 0;
        for (std::uint32_t at = 0; at < n - 1;) {
            const std::uint32_t chunk = std::min<std::uint32_t>(o_.prefill_chunk, n - 1 - at);
            CallPlan call;
            call.first  = {0};
            call.prompt = true;
            for (std::uint32_t i = 0; i < chunk; ++i) {
                call.tokens.push_back(tokens[at + i]);
                call.lane.push_back(0);
                call.position.push_back(static_cast<std::int32_t>(at + i));
            }
            upload(call);
            // Positions at + i predict token at + i + 1; scored from first_target - 1 on.
            const std::int32_t from = at + chunk <= first_target - 1
                                          ? static_cast<std::int32_t>(chunk)
                                          : static_cast<std::int32_t>(std::max<std::int64_t>(0, std::int64_t{first_target} - 1 - at));
            run_forward(from, 0); // the final mix; the head runs below a slice of logits rows at a time
            for (std::int32_t c = from; c < static_cast<std::int32_t>(chunk); ++c) {
                const std::int32_t slice = (c - from) % kMaxForwardColumns;
                if (slice == 0) {
                    forward_.head(c, std::min<std::int32_t>(kMaxForwardColumns, static_cast<std::int32_t>(chunk) - c),
                                  device_.stream);
                    CUDA_CHECK(cudaStreamSynchronize(device_.stream));
                }
                const std::uint32_t position = at + static_cast<std::uint32_t>(c);
                const std::uint32_t scored   = position + 1 - first_target;
                CUDA_CHECK(cudaMemcpy(row.data(), forward_.buffers().logits + static_cast<std::size_t>(slice) * vocab_,
                                      row.size() * 4, cudaMemcpyDeviceToHost));
                double max = -std::numeric_limits<double>::infinity();
                for (const float v : row) { max = std::max(max, static_cast<double>(v)); }
                double sum = 0;
                for (const float v : row) { sum += std::exp(static_cast<double>(v) - max); }
                const double lse = max + std::log(sum);
                out.logprobs.push_back(static_cast<float>(row[static_cast<std::size_t>(tokens[position + 1])] - lse));
                if (options.top_k != 0) {
                    std::vector<std::int32_t> ids(row.size());
                    for (std::size_t i = 0; i < ids.size(); ++i) { ids[i] = static_cast<std::int32_t>(i); }
                    std::partial_sort(ids.begin(), ids.begin() + options.top_k, ids.end(), [&](std::int32_t a, std::int32_t b) {
                        return row[a] > row[b] || (row[a] == row[b] && a < b);
                    });
                    for (std::uint32_t k = 0; k < options.top_k; ++k) {
                        out.top_ids.push_back(ids[k]);
                        out.top_logprobs.push_back(static_cast<float>(row[ids[k]] - lse));
                    }
                }
                for (std::uint32_t k = 0; k < options.candidates_per_position; ++k) {
                    const TokenId id = options.candidates[static_cast<std::size_t>(scored) * options.candidates_per_position + k];
                    out.candidate_logprobs.push_back(id >= 0 && id < vocab_ ? static_cast<float>(row[id] - lse)
                                                                            : -std::numeric_limits<float>::infinity());
                }
                if (next_logits < options.logits_positions.size() && options.logits_positions[next_logits] == scored) {
                    out.logits.insert(out.logits.end(), row.begin(), row.end());
                    ++next_logits;
                }
            }
            at += chunk;
        }
        return out;
    }

    // ---------------------------------------------------------------- diagnostics
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept {
        PhysicalUsageSnapshot out;
        const std::uint32_t pages = pages_per_lane(o_.max_context);
        for (const Lane& lane : lanes_) {
            if (lane.bound) {
                ++out.occupied.state_slots;
                out.occupied.main_kv_pages += pages;
            }
        }
        out.capacity.state_slots   = static_cast<std::uint32_t>(lanes_.size());
        out.capacity.main_kv_pages = pages * static_cast<std::uint32_t>(lanes_.size());
        return out;
    }

    [[nodiscard]] MemoryUsage memory_usage() const noexcept {
        MemoryUsage out;
        out.device.weights   = p_.model.device_weight_bytes();
        out.device.kv        = plan_.kv;
        out.device.state     = plan_.state + plan_.io;
        out.device.workspace = plan_.workspace;
        std::uint64_t used   = 0;
        for (const Lane& lane : lanes_) {
            if (lane.bound) { used += plan_.kv / lanes_.size(); }
        }
        out.device.kv_used = used;
        out.host.staging   = host_.size();
        return out;
    }

    [[nodiscard]] MemorySummary memory_summary() const noexcept {
        MemorySummary out;
        out.max_context      = o_.max_context;
        out.kv_capacity      = o_.max_context * o_.max_concurrency;
        out.kv_cache         = KvCacheStorage::BFloat16;
        out.kv_payload_bytes = plan_.kv;
        return out;
    }

    const ProgramDevicePlan& device_plan() const noexcept { return plan_; }

private:
    struct Lane {
        bool bound          = false;
        std::uint64_t epoch = 0;
        std::vector<TokenId> tokens; // the prompt, then the committed output and forced tokens
        std::uint32_t prompt_tokens = 0;
        std::uint32_t processed     = 0; // positions whose state the lane holds
        runtime::ResolvedExecutionOptions options;
        bool penalties = false;
        std::uint64_t prefill_ns = 0, decode_ns = 0, decode_share_ns = 0;
        // DSpark: the drafts' acceptance by probability decile (draft_length.h) and the request's counters.
        runtime::speculative::DraftCalibration calibration;
        std::uint64_t policy_rounds = 0;
        std::uint64_t rounds = 0, drafted = 0, accepted = 0, fallback = 0;
        std::array<std::uint64_t, ops::inkling2::kMarkovSteps> accepted_at{}; // rounds accepting draft i
    };
    // A DSpark round's row awaiting its commit: the lane, the anchor's position and the tokens it licenses.
    struct PendingDraft {
        bool active        = false;
        std::uint32_t lane = 0, first = 0;
    };
    enum class PendingKind : std::uint8_t { Prefill, Decode };

    SequenceHandle handle(std::uint32_t index) const {
        return ContractAccess::handle(this, index, lanes_[index].epoch);
    }
    [[nodiscard]] bool valid(SequenceHandle s) const noexcept {
        const std::uint32_t index = ContractAccess::lane(s);
        return ContractAccess::owner(s) == this && index < lanes_.size() && lanes_[index].bound &&
               lanes_[index].epoch == ContractAccess::epoch(s);
    }
    std::uint32_t lane_of(SequenceHandle s) const {
        if (!valid(s)) { throw std::logic_error("Inkling 2: a stale or foreign sequence"); }
        return ContractAccess::lane(s);
    }
    std::int32_t* lane_counts(std::uint32_t index) {
        return static_cast<std::int32_t*>(counts_.p) + static_cast<std::size_t>(index) * vocab_;
    }
    void release(std::uint32_t index) noexcept {
        if (index < lanes_.size()) { lanes_[index] = Lane{}; }
    }
    static GenerationTimings timings(const Lane& lane) {
        GenerationTimings out;
        out.prefill_seconds      = static_cast<double>(lane.prefill_ns) * 1e-9;
        out.decode_seconds       = static_cast<double>(lane.decode_ns) * 1e-9;
        out.decode_share_seconds = static_cast<double>(lane.decode_share_ns) * 1e-9;
        return out;
    }
    // The unit's time since `start`: the device waits it made (wait_ns_, taken) and the rest as host time.
    runtime::ExecutionTiming timing_since(Clock::time_point start) {
        runtime::ExecutionTiming t;
        const std::uint64_t total = elapsed_ns(start);
        t.device_wait_ns          = std::min(wait_ns_, total);
        t.submit_host_ns          = total - t.device_wait_ns;
        wait_ns_                  = 0;
        return t;
    }

    // The call plan through the next staging slot: a slot is rewritten only once the copies that read it ran (calls
    // queue without a host wait between them).
    void upload(const CallPlan& call) {
        slot_ = (slot_ + 1) % kStagingSlots;
        if (slot_used_[slot_]) { CUDA_CHECK(cudaEventSynchronize(slot_events_[slot_])); }
        forward_.upload(call, staging_ + slot_ * forward_.upload_words(), device_.stream);
        CUDA_CHECK(cudaEventRecord(slot_events_[slot_], device_.stream));
        slot_used_[slot_] = true;
    }

    // The Forward over the uploaded call, its head over columns [head_first, head_first + head_columns).
    void run_forward(std::int32_t head_first, std::int32_t head_columns) {
        forward_.embed(device_.stream);
        for (std::uint32_t v = 0; v < c_.visits(); ++v) { forward_.visit(v, view_, device_.stream); }
        forward_.finish(view_, forward_.final_inputs(), device_.stream, head_first, head_columns);
    }

    // A decode or verify call: the Forward with the head over every column, then (DSpark) its columns' context rows
    // (a rejected verify column's rows are overwritten before anything reads them).
    void run_decode_call(std::int32_t columns) {
        run_forward(0, columns);
        if (drafter_) { drafter_->append(0, columns, device_.stream); }
    }

    // ---------------------------------------------------------------- DSpark (design §13)
    static constexpr double kAcceptanceWeight     = 0.1;  // the calibration's EWMA weight
    static constexpr std::uint64_t kProbeInterval = 8;    // one draft more every 8 rounds keeps longer deciles measured
    // Costs in plain one-row rounds (2026-11-21, the forward test's decode round at 4K: +0.11-0.13 a verify column;
    // the draft's ~1 ms of a ~6 ms round).
    static constexpr double kDraftCost    = 0.12;
    static constexpr double kProposalCost = 0.18;

    // A round drafts when one row decodes with its one pending input, the drafter has its context (F >= 1), the
    // position after the anchor fits the context, and nothing per position constrains the draw (penalties, a
    // grammar, a reasoning ban): verification samples each column as plain decoding would.
    [[nodiscard]] bool dspark_round(std::span<const SequenceHandle> sequences, runtime::TokenMaskProvider* masks) const {
        if (!drafter_ || sequences.size() != 1) { return false; }
        const Lane& lane = lanes_[lane_of(sequences[0])];
        if (lane.penalties || lane.processed == 0 || lane.tokens.size() != lane.processed + 1U ||
            lane.processed + 1U >= o_.max_context) {
            return false;
        }
        // A reasoning ban is fine: every column samples under it, and the round ends at the first close it draws.
        return masks == nullptr || !masks->constrained(0);
    }

    PendingBatch decode_dspark(SequenceHandle sequence, std::uint32_t remaining, runtime::TokenMaskProvider* masks,
                               Clock::time_point start) {
        const std::uint32_t index = lane_of(sequence);
        Lane& lane                = lanes_[index];
        const auto F              = static_cast<std::int32_t>(lane.processed);
        // Drafts whose positions fit the context and whose round (drafts + 1 tokens) fits the row's budget.
        const std::int32_t steps =
            std::min({static_cast<std::int32_t>(o_.draft_tokens), static_cast<std::int32_t>(o_.max_context) - 1 - F,
                      static_cast<std::int32_t>(std::min<std::uint32_t>(remaining - 1U, ops::inkling2::kMarkovSteps))});
        // The drafts: the block, the head over its rows and the Markov steps, read back before the verify call.
        drafter_->stage({F, lane.tokens[static_cast<std::size_t>(F)], static_cast<std::int32_t>(index)});
        if (o_.use_cuda_graph) {
            draft_graph(steps).launch(device_.stream);
        } else {
            drafter_->draft(steps, device_.stream);
            drafter_->download(steps, device_.stream);
        }
        {
            const auto wait = Clock::now();
            CUDA_CHECK(cudaStreamSynchronize(device_.stream));
            wait_ns_ += elapsed_ns(wait);
        }
        std::array<TokenId, ops::inkling2::kMarkovSteps> drafts{};
        std::array<float, ops::inkling2::kMarkovSteps> probabilities{};
        for (std::int32_t i = 0; i < steps; ++i) {
            drafts[static_cast<std::size_t>(i)]        = drafter_->host_tokens()[i];
            probabilities[static_cast<std::size_t>(i)] = drafter_->host_probabilities()[i];
        }
        std::int32_t k = steps;
        if (o_.draft_tokens_auto) {
            k = runtime::speculative::calibrated_draft_length(
                lane.calibration, std::span<const float>(probabilities.data(), static_cast<std::size_t>(steps)), steps,
                kDraftCost, kProposalCost);
            if (++lane.policy_rounds % kProbeInterval == 0) { k = std::min(k + 1, steps); }
        }
        // The verify call: the pending input and the k drafts at F .. F + k, every column sampled as decoding would.
        CallPlan call;
        call.first.assign(index + 1, 0);
        const std::int32_t columns = k + 1;
        std::array<std::uint32_t, kMaximumConcurrency> rows{};
        std::array<std::int32_t, kMaximumConcurrency> logit_rows{}, positions{};
        for (std::int32_t i = 0; i < columns; ++i) {
            call.tokens.push_back(i == 0 ? lane.tokens[static_cast<std::size_t>(F)] : drafts[static_cast<std::size_t>(i - 1)]);
            call.lane.push_back(static_cast<std::int32_t>(index));
            call.position.push_back(F + i);
            rows[static_cast<std::size_t>(i)]       = index;
            logit_rows[static_cast<std::size_t>(i)] = i;
            positions[static_cast<std::size_t>(i)]  = F + i;
        }
        upload(call);
        const auto n = static_cast<std::size_t>(columns);
        // Every column samples under the row's reasoning ban (an open reasoning excludes its end tokens).
        const runtime::ReasoningBan ban = masks != nullptr ? masks->reasoning_ban(0) : runtime::ReasoningBan{};
        stage_sample({rows.data(), n}, {logit_rows.data(), n}, ops::kSamplePurposeDecode, masks, {positions.data(), n},
                     true);
        if (o_.use_cuda_graph) {
            verify_graph(index, columns).launch(device_.stream);
        } else {
            run_decode_call(columns);
            launch_sample(n, {logit_rows.data(), n}, ops::kSamplePurposeDecode);
        }
        finish_sample(n);
        // The accepted prefix: draft i stands while it equals the target's own draw at its position.
        std::int32_t a = 0;
        while (a < k && drafts[static_cast<std::size_t>(a)] == sampled_tokens_[static_cast<std::size_t>(a)] &&
               failures_[static_cast<std::size_t>(a)] == runtime::PendingRowFailure::None) {
            ++a;
        }
        runtime::PendingRowFailure failure = runtime::PendingRowFailure::None;
        for (std::int32_t i = 0; i <= a; ++i) {
            if (failures_[static_cast<std::size_t>(i)] != runtime::PendingRowFailure::None) {
                failure = failures_[static_cast<std::size_t>(i)];
                break;
            }
        }
        failures_[0] = failure;
        lane.calibration.observe(std::span<const float>(probabilities.data(), static_cast<std::size_t>(steps)),
                                 static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(a), kAcceptanceWeight);
        // The round ends at the first reasoning close it draws: later columns sampled under the ban the close lifts.
        if (ban.active()) {
            for (std::int32_t i = 0; i < a; ++i) {
                if (sampled_tokens_[static_cast<std::size_t>(i)] == ban.close) {
                    a = i;
                    break;
                }
            }
        }
        ++lane.rounds;
        lane.drafted += static_cast<std::uint64_t>(k);
        lane.accepted += static_cast<std::uint64_t>(a);
        for (std::int32_t i = 0; i < a; ++i) { ++lane.accepted_at[static_cast<std::size_t>(i)]; }
        row_counts_[0]   = a + 1;
        pending_draft_   = {true, index, static_cast<std::uint32_t>(F)};
        const std::uint64_t transaction = ++transaction_;
        const std::uint64_t ns          = elapsed_ns(start);
        lane.decode_ns += ns;
        lane.decode_share_ns += ns;
        pending_      = transaction;
        pending_kind_ = PendingKind::Decode;
        return ContractAccess::batch(this, transaction, {&sequence, 1}, {sampled_tokens_.data(), kMaximumConcurrency},
                                     timing_since(start), {failures_.data(), 1}, {row_counts_.data(), 1},
                                     static_cast<std::uint32_t>(kMaximumConcurrency));
    }

    // The captured draft of `steps` drafts (its round words copied in from pinned memory, the drafts copied out).
    DecodeGraphExecutable& draft_graph(std::int32_t steps) {
        auto found = draft_graphs_.find(steps);
        if (found == draft_graphs_.end()) {
            DecodeGraphDefinition definition;
            definition.capture(device_.stream, [&] {
                drafter_->draft(steps, device_.stream);
                drafter_->download(steps, device_.stream);
            });
            DecodeGraphExecutable exec;
            exec.instantiate(definition);
            found = draft_graphs_.emplace(steps, std::move(exec)).first;
        }
        return found->second;
    }

    // The captured verify call of one lane and `columns` columns (its call plan, configs and positions uploaded
    // before the launch): the forward, the context rows and the sample of every column.
    DecodeGraphExecutable& verify_graph(std::uint32_t lane, std::int32_t columns) {
        const std::uint64_t key = (1ULL << 63) | static_cast<std::uint64_t>(lane) << 32 | static_cast<std::uint32_t>(columns);
        auto found = graphs_.find(key);
        if (found == graphs_.end()) {
            std::vector<std::int32_t> identity(static_cast<std::size_t>(columns));
            for (std::int32_t i = 0; i < columns; ++i) { identity[static_cast<std::size_t>(i)] = i; }
            DecodeGraphDefinition definition;
            definition.capture(device_.stream, [&] {
                run_decode_call(columns);
                launch_sample(static_cast<std::size_t>(columns), identity, ops::kSamplePurposeDecode);
            });
            DecodeGraphExecutable exec;
            exec.instantiate(definition);
            found = graphs_.emplace(key, std::move(exec)).first;
        }
        return found->second;
    }

    // A lane's speculative counters and statistics for the Engine.
    void speculative_of(const Lane& lane, CommitRowResult& row) const {
        if (!drafter_) { return; }
        row.speculative_counters = {lane.rounds, lane.drafted, lane.accepted, lane.fallback};
        SpeculativeStats& s      = row.speculative;
        s.backend                = SpeculativeBackend::DSpark;
        s.enabled                = true;
        s.draft_window           = o_.draft_tokens;
        s.rounds                 = lane.rounds;
        s.drafted_tokens         = lane.drafted;
        s.accepted_tokens        = lane.accepted;
        s.fallback_steps         = lane.fallback;
        s.accepted_per_position.assign(lane.accepted_at.begin(), lane.accepted_at.begin() + o_.draft_tokens);
    }

    // Samples one token a row from the logits rows `logit_rows` (each row's lane `lanes[row]`), into
    // sampled_tokens_ and failures_ (synchronous).
    void sample(std::span<const SequenceHandle> rows, std::span<const std::uint32_t> lanes, std::span<const std::int32_t> logit_rows,
                std::int32_t purpose, runtime::TokenMaskProvider* masks) {
        enqueue_sample(lanes, logit_rows, purpose, masks);
        finish_sample(rows.size());
    }
    void sample(std::span<const SequenceHandle> rows, std::span<const std::uint32_t> lanes, std::int32_t logit_row,
                std::int32_t purpose, runtime::TokenMaskProvider* masks) {
        const std::int32_t r = logit_row;
        sample(rows, lanes, std::span<const std::int32_t>(&r, 1), purpose, masks);
    }

    // The rows' configs and positions to pinned memory, and the logits rows gathered: everything a sample needs
    // before its launch (outside a graph).
    // `positions`: each row's draw key (a verify call's columns); empty: the lane's. `mask_row_zero`: every row takes the
    // masks' row 0 (a verify call's columns of one row).
    void stage_sample(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> logit_rows, std::int32_t purpose,
                      runtime::TokenMaskProvider* masks, std::span<const std::int32_t> positions = {},
                      bool mask_row_zero = false) {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const Lane& lane                       = lanes_[lanes[row]];
            const ResolvedSamplingParameters& s    = lane.options.sampling;
            ops::SamplingConfig config;
            config.temperature       = s.temperature;
            config.top_k             = s.top_k;
            config.top_p             = s.top_p;
            config.min_p             = s.min_p;
            config.presence_penalty  = s.presence_penalty;
            config.frequency_penalty = s.frequency_penalty;
            config.seed              = s.seed;
            config.token_counts      = lane.penalties ? lane_counts(lanes[row]) : nullptr;
            nonfinite_[row]          = 0;
            config.nonfinite         = &nonfinite_[row];
            if (masks != nullptr) {
                const runtime::ReasoningBan ban = masks->reasoning_ban(mask_row_zero ? 0 : row);
                config.reasoning_close          = ban.close;
                config.reasoning_ends[0]        = ban.ends[0];
                config.reasoning_ends[1]        = ban.ends[1];
                config.reasoning_withheld       = ban.withheld;
            }
            host_configs_[row] = config;
            // Prefill keys the draw by the prompt's next position (the lane has processed the prompt), decode by its
            // last input token's (the lane's last token, processed by this call).
            host_positions_[row] = !positions.empty() ? positions[row]
                                   : static_cast<std::int32_t>(purpose == ops::kSamplePurposePrefill
                                                                   ? lane.processed
                                                                   : static_cast<std::uint32_t>(lane.tokens.size()) - 1U);
        }
        CUDA_CHECK(cudaMemcpyAsync(configs_.p, host_configs_, sizeof(ops::SamplingConfig) * lanes.size(),
                                   cudaMemcpyHostToDevice, device_.stream));
        CUDA_CHECK(cudaMemcpyAsync(positions_.p, host_positions_, 4 * lanes.size(), cudaMemcpyHostToDevice,
                                   device_.stream));
        (void)logit_rows;
    }

    // The sample of rows whose logits are rows `logit_rows` of the call's logits: one per row, gathered when
    // the rows are not the call's leading rows in order.
    void launch_sample(std::size_t rows, std::span<const std::int32_t> logit_rows, std::int32_t purpose) {
        const float* logits = forward_.buffers().logits;
        bool identity       = true;
        for (std::size_t i = 0; i < rows; ++i) { identity = identity && logit_rows[i] == static_cast<std::int32_t>(i); }
        if (!identity) {
            if (gathered_.bytes == 0) { gathered_ = DeviceBuffer(static_cast<std::size_t>(vocab_) * 4 * kMaximumConcurrency); }
            for (std::size_t i = 0; i < rows; ++i) {
                CUDA_CHECK(cudaMemcpyAsync(static_cast<float*>(gathered_.p) + i * vocab_,
                                           logits + static_cast<std::size_t>(logit_rows[i]) * vocab_,
                                           static_cast<std::size_t>(vocab_) * 4, cudaMemcpyDeviceToDevice, device_.stream));
            }
            logits = static_cast<const float*>(gathered_.p);
        }
        const Tensor in(const_cast<float*>(logits), DType::FP32, {vocab_, static_cast<std::int32_t>(rows)});
        Tensor out(sampled_.p, DType::I32, {static_cast<std::int32_t>(rows)});
        const Tensor positions(positions_.p, DType::I32, {static_cast<std::int32_t>(rows)});
        ops::sample(in, out, vocab_, static_cast<const ops::SamplingConfig*>(configs_.p), positions, purpose, sampling_,
                    device_.stream);
        CUDA_CHECK(cudaMemcpyAsync(host_sampled_, sampled_.p, 4 * rows, cudaMemcpyDeviceToHost, device_.stream));
    }

    void enqueue_sample(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> logit_rows,
                        std::int32_t purpose, runtime::TokenMaskProvider* masks) {
        stage_sample(lanes, logit_rows, purpose, masks);
        launch_sample(lanes.size(), logit_rows, purpose);
    }

    void finish_sample(std::size_t rows) {
        const auto wait = Clock::now();
        CUDA_CHECK(cudaStreamSynchronize(device_.stream));
        wait_ns_ += elapsed_ns(wait);
        for (std::size_t row = 0; row < rows; ++row) {
            sampled_tokens_[row] = host_sampled_[row];
            failures_[row] = nonfinite_[row] != 0 ? runtime::PendingRowFailure::NonFiniteLogits : runtime::PendingRowFailure::None;
        }
    }

    // A uniform decode round (every lane one input) as the graph of its (lanes, columns), captured on first use; the
    // call plan, configs and positions are uploaded before the launch.
    void decode_graph(std::int32_t lanes, std::int32_t columns, std::span<const std::uint32_t> rows,
                      std::span<const std::int32_t> logit_rows, runtime::TokenMaskProvider* masks) {
        stage_sample(rows, logit_rows, ops::kSamplePurposeDecode, masks);
        const std::uint64_t key = static_cast<std::uint64_t>(lanes) << 32 | static_cast<std::uint32_t>(columns);
        // A graph samples the call's rows in column order; rows given in another order are gathered eagerly.
        bool identity = true;
        for (std::size_t i = 0; i < rows.size(); ++i) { identity = identity && logit_rows[i] == static_cast<std::int32_t>(i); }
        if (!identity) {
            run_decode_call(columns);
            launch_sample(rows.size(), logit_rows, ops::kSamplePurposeDecode);
            finish_sample(rows.size());
            return;
        }
        auto found = graphs_.find(key);
        if (found == graphs_.end()) { found = capture_decode(lanes, columns); }
        found->second.launch(device_.stream);
        finish_sample(rows.size());
    }

    // The graph of a uniform decode call over `lanes` lanes with `columns` of them active (its rows the call's columns
    // in order). Its topology depends on those two counts only: tokens, lanes, positions, configs and the state are
    // device data the launch reads.
    std::map<std::uint64_t, DecodeGraphExecutable>::iterator capture_decode(std::int32_t lanes, std::int32_t columns) {
        std::vector<std::int32_t> identity(static_cast<std::size_t>(columns));
        for (std::int32_t i = 0; i < columns; ++i) { identity[static_cast<std::size_t>(i)] = i; }
        DecodeGraphDefinition definition;
        definition.capture(device_.stream, [&] {
            run_decode_call(columns);
            launch_sample(static_cast<std::size_t>(columns), identity, ops::kSamplePurposeDecode);
        });
        DecodeGraphExecutable exec;
        exec.instantiate(definition);
        const std::uint64_t key = static_cast<std::uint64_t>(lanes) << 32 | static_cast<std::uint32_t>(columns);
        return graphs_.emplace(key, std::move(exec)).first;
    }

    // Every uniform decode call's graph, at startup (design §8): (lanes, columns) for 1 <= columns <= lanes <= the
    // concurrency. A placeholder call fixes the Forward's widths for each capture; nothing runs.
    void capture_decode_graphs() {
        const auto max = static_cast<std::int32_t>(lanes_.size());
        for (std::int32_t lanes = 1; lanes <= max; ++lanes) {
            for (std::int32_t columns = 1; columns <= lanes; ++columns) {
                CallPlan call;
                call.first.assign(static_cast<std::size_t>(lanes), 0);
                for (std::int32_t c = 0; c < columns; ++c) {
                    call.tokens.push_back(0);
                    call.lane.push_back(lanes - columns + c);
                    call.position.push_back(0);
                }
                upload(call);
                (void)capture_decode(lanes, columns);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(device_.stream));
    }

    PromptReadout readout(const Lane& lane, std::int32_t column) {
        std::vector<float> row(static_cast<std::size_t>(vocab_));
        CUDA_CHECK(cudaMemcpy(row.data(), forward_.buffers().logits + static_cast<std::size_t>(column) * vocab_,
                              row.size() * 4, cudaMemcpyDeviceToHost));
        double max = -std::numeric_limits<double>::infinity();
        std::int32_t top = 0;
        for (std::int32_t v = 0; v < vocab_; ++v) {
            if (row[v] > max) {
                max = row[v];
                top = v;
            }
        }
        double sum = 0;
        for (const float v : row) { sum += std::exp(static_cast<double>(v) - max); }
        const double lse = max + std::log(sum);
        PromptReadout out;
        for (const TokenId id : lane.options.readout_tokens) { out.logprobs.push_back(static_cast<float>(row[id] - lse)); }
        out.top_token   = top;
        out.top_logprob = static_cast<float>(max - lse);
        return out;
    }

    const Parameters& p_;
    const TextConfig& c_;
    DeviceContext& device_;
    ProgramOptions o_;
    ProgramDevicePlan plan_;
    Forward forward_;
    std::unique_ptr<Drafter> drafter_; // --spec dspark
    std::map<std::int32_t, DecodeGraphExecutable> draft_graphs_;
    PendingDraft pending_draft_;
    std::array<std::int32_t, kMaximumConcurrency> row_counts_{};
    std::int32_t vocab_ = 0;
    std::vector<Lane> lanes_;
    DeviceBuffer images_, table_, counts_, configs_, positions_, sampled_, gathered_;
    std::vector<DeviceBuffer> pools_;
    StateView view_;
    WorkspaceArena sampling_;
    PinnedHostBuffer host_;
    std::int32_t* staging_                = nullptr;
    ops::SamplingConfig* host_configs_    = nullptr;
    std::int32_t* host_positions_         = nullptr;
    std::int32_t* host_sampled_           = nullptr;
    std::int32_t* nonfinite_              = nullptr;
    std::array<TokenId, kMaximumConcurrency> sampled_tokens_{};
    std::array<runtime::PendingRowFailure, kMaximumConcurrency> failures_{};
    std::map<std::uint64_t, DecodeGraphExecutable> graphs_;
    static constexpr int kStagingSlots = 4;
    struct Events {
        std::array<cudaEvent_t, kStagingSlots> events{};
        Events() {
            for (auto& e : events) { CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); }
        }
        ~Events() {
            for (auto e : events) { cudaEventDestroy(e); }
        }
        Events(const Events&)            = delete;
        Events& operator=(const Events&) = delete;
        cudaEvent_t operator[](int i) const { return events[static_cast<std::size_t>(i)]; }
    } slot_events_;
    std::array<bool, kStagingSlots> slot_used_{};
    int slot_ = 0;
    std::optional<std::uint32_t> binding_;
    std::optional<std::uint64_t> pending_;
    PendingKind pending_kind_ = PendingKind::Decode;
    std::uint64_t transaction_ = 0;
    std::uint64_t epoch_       = 0;
    std::uint64_t wait_ns_     = 0; // device waits of the unit in progress
};

} // namespace detail

using detail::ContractAccess;

RequestBasePlan::RequestBasePlan(std::unique_ptr<detail::BasePlanImpl> impl) noexcept : impl_(std::move(impl)) {}
RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept            = default;
RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;
RequestBasePlan::~RequestBasePlan()                                     = default;
const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept { return impl_->summary; }

ProgramDevicePlan Program::plan_device(const ProgramOptions& options, const Config& config) {
    if (options.max_context == 0) { throw std::invalid_argument("Inkling 2: max_context must be positive"); }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Inkling 2: max_concurrency must be 1-8");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk > static_cast<std::uint32_t>(kMaxPrefillChunk)) {
        throw std::invalid_argument("Inkling 2: prefill chunks are 1-8191 columns");
    }
    const StateLayout layout   = state_layout(config.text);
    const std::uint64_t lanes  = options.max_concurrency;
    const std::uint64_t pages  = static_cast<std::uint64_t>(detail::pages_per_lane(options.max_context)) * lanes;
    ProgramDevicePlan out;
    out.kv_pages  = static_cast<std::uint32_t>(pages);
    out.kv        = 2ULL * layout.full_visits * detail::pool_bytes(pages) + pages * 4;
    out.state     = static_cast<std::uint64_t>(layout.lane_words) * 2 * lanes;
    out.io        = static_cast<std::uint64_t>(config.text.vocab_size) * 4 * lanes * 2;
    const bool dspark = options.dspark && !options.causal_scoring;
    out.workspace = Forward::device_bytes(config.text, static_cast<std::int32_t>(lanes), detail::forward_capacity(options),
                                          dspark) +
                    ops::sampling_workspace_capacity_bytes(static_cast<std::int32_t>(config.text.vocab_size), 1,
                                                           static_cast<std::int32_t>(kMaximumConcurrency));
    if (dspark) {
        out.workspace += Drafter::device_bytes(static_cast<std::int32_t>(lanes), options.max_context,
                                               detail::forward_capacity(options), config.text.vocab_size);
    }
    out.graph_bound = static_cast<std::uint32_t>(lanes * (lanes + 1) / 2);
    return out;
}

Program::Program(const Parameters& parameters, DeviceContext& device, ProgramOptions options)
    : impl_(std::make_unique<detail::ProgramImpl>(parameters, device, std::move(options))) {}
Program::~Program() noexcept = default;

const ProgramDevicePlan& Program::device_plan() const noexcept { return impl_->device_plan(); }

RequestBasePlan Program::plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options) {
    return impl_->plan_request(std::move(prompt), options);
}

std::vector<SourceCandidate> Program::hybrid_sources(const RequestBasePlan&, const ResumeState* resume) {
    return impl_->sources(resume);
}

BindingReservation Program::start_binding(const RequestBasePlan& base, runtime::LaneId lane, const SourceCandidate&,
                                          ResumeState* resume, ExecutionUnitKind, std::uint32_t) {
    return impl_->start_binding(base, lane, resume);
}

ContextProgress Program::poll_context(runtime::CancellationFlagView) { return impl_->poll_context(); }
bool Program::has_context_transaction() const noexcept { return impl_->has_context_transaction(); }
bool Program::context_blocks(SequenceHandle sequence) const noexcept { return impl_->context_blocks(sequence); }

runtime::ResourceReservation Program::reserve_units(std::span<const ExecutionUnit>) {
    return {.reserved = true}; // every lane's KV is reserved at startup
}

PrefillProgress Program::advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming*, runtime::TokenMaskProvider* masks) {
    return impl_->advance_prefill(sequence, masks);
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming*, runtime::TokenMaskProvider* masks) {
    return impl_->decode(sequences, budgets, masks);
}

runtime::ExecutionTiming Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
                                                       std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                                                       std::span<const std::optional<std::uint32_t>>,
                                                       runtime::ExecutionTiming*) {
    return impl_->append_forced(sequences, row_major_tokens, row_stride);
}

CommitResult Program::commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation, runtime::ExecutionTiming*) {
    return impl_->commit(std::move(pending), decisions);
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept { return impl_->discard(std::move(pending)); }
FinishResult Program::finish(SequenceHandle sequence) noexcept { return impl_->finish(sequence); }
AbortResult Program::abort(SequenceHandle sequence) noexcept { return impl_->abort(sequence); }
void Program::fail_all_cleanup() noexcept { impl_->release_all(); }
bool Program::recover_after_failure() noexcept { return impl_->recover(); }
void Program::shutdown_cleanup() noexcept { impl_->release_all(); }

ScoreResult Program::causal_score(PreparedPrompt&& prompt, std::uint32_t first_target, const ScoreOptions& options) {
    return impl_->causal_score(std::move(prompt), first_target, options);
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept { return impl_->physical_usage(); }
MemoryUsage Program::memory_usage() const noexcept { return impl_->memory_usage(); }
MemorySummary Program::memory_summary() const noexcept { return impl_->memory_summary(); }

} // namespace infernix::models::inkling2
