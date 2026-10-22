#pragma once

#include "models/qwen3_5/execution/lossless.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/program.h"
#include "infernix/ops/gated_delta_net.h"
#include "infernix/ops/softmax_attention.h"

#include <cstddef>
#include <cstdint>

namespace infernix::models::qwen3_5 {

inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 5;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 15;

// Execution settings of a Program that select among equivalent arithmetic. Causal attention
// (CausalAttentionExecutionEnvelope), for prefill chunks: fast selects the storage's fast prompt
// kernel (INT8 or NVFP4 KV); pv8 runs the prompt kernel's PV in 8 bits (the prompt-pv8 fast path);
// and split_workspace_bytes bounds the FP32 partials of a launch's key splits. For every launch
// (prefill, decode, verification, MTP): decode_split64 is the decode-split64 fast path.
//
// The rest are request invariance, each set where the Model supports it and its request-coupling
// fast path (FastPath::BatchedDecode and those after it) is not selected:
// column_invariant_decode runs decode and verification projections under
// LinearPolicy::A16ColumnInvariant (without it, a8_decode and a4_decode let wide rounds of FP8 and
// NVFP4 projections take the artifact's 8- or 4-bit activations, a8_residual_decode the FP8
// projections the A8 residual profile and a4_residual_decode the NVFP4 MLP projections the A4
// residual profile, else they stay at 16 bits), except calls of one column under
// single_column_decode (batched_columns);
// row_invariant_partition sizes
// every attention row's key partition by that row alone; fixed_speculation verifies the
// configured chain at a fixed width;
// grid_cache splits prefill, publishes and resumes only on the prefill chunk grid; held_extent
// holds each request's whole KV extent from admission, so nothing preempts it. With all of them a
// request's output does not depend on the other requests of the Engine. linear_routes carries
// the LinearRoutes flags of every Text-layer MLP gate/up projection (q4-w4a8); under
// column_invariant_decode the decode and verification projections ignore them. Workspace
// planning and every launch use the same settings.
struct ExecutionSchedule {
    bool fast                         = false;
    bool pv8                          = false;
    std::size_t split_workspace_bytes = ops::kCausalPromptSplitWorkspaceDefaultBytes;
    bool decode_split64               = false;
    bool column_invariant_decode      = false;
    // single-column-decode: decode and verification calls of one column run the batched route.
    bool single_column_decode         = false;
    bool a8_decode                    = true;
    bool a4_decode                    = true;
    bool a8_residual_decode           = false;
    bool a4_residual_decode           = false;
    bool row_invariant_partition      = false;
    bool fixed_speculation            = false;
    bool grid_cache                   = false;
    bool held_extent                  = false;
    // adaptive-mtp: MTP rounds choose their draft length (only where speculation is adaptive).
    bool adaptive_mtp                 = false;
    // adaptive-dflash: DFlash and DFlash2 all-neural rounds propose, then choose their verify
    // width (only where speculation is adaptive).
    bool adaptive_dflash              = false;
    // MTP rounds write each next draft's drafter probability (MtpConfidencePolicy's input; MTP with
    // offloaded experts).
    bool mtp_draft_confidence         = false;
    ops::LinearRoutes linear_routes   = ops::LinearRoutes::Exact;
    // gdn-f16-accumulate: the chunked GDN calls' arithmetic.
    ops::GdnChunkArithmetic gdn_arithmetic = ops::GdnChunkArithmetic::Exact;
    // The lossless route (EngineOptions::lossless, docs/maintainer/lossless-route.md): every layer
    // execution::lossless_layer, FP32 logits. A CausalScoring Program scores each window in one pass (the
    // window form, lossless_keys 0); a Generation Program runs prefill chunks and decode rounds through
    // the KV cache (the lossless bf16 profile) and the recurrent state slots (the cache form, keys up to
    // lossless_keys, max_context) without speculation; decode rounds run as the CUDA Graphs of the
    // ordinary rounds (their attention key bound the graph profile's).
    bool lossless              = false;
    std::int32_t lossless_keys = 0;
    // The design reference (EngineOptions::reference, with lossless; lossless-route.md §3b): the window form
    // with the declared activation quantization and the KV format's codec.
    execution::LosslessDesign lossless_design{};
    // The reproducible and lossless routes: sampled requests draw by Gumbel-max
    // (ops::SamplingConfig::gumbel_max), so speculation returns plain sampling's text.
    bool gumbel_sampling = false;

    // Whether a decode or verification call of `columns` columns runs batched-decode's route: its
    // projections' column-count routes (Parameters::decode_text without column invariance) and the
    // MLP down projections' tiles (ops::LinearRoutes::WideDownTile).
    [[nodiscard]] bool batched_columns(std::int64_t columns) const noexcept {
        return !column_invariant_decode || (single_column_decode && columns == 1);
    }

    // A prefill chunk's envelope.
    [[nodiscard]] ops::CausalAttentionExecutionEnvelope
    envelope(std::uint32_t min_visible_keys, std::uint32_t max_visible_keys) const noexcept {
        return {.min_visible_keys             = min_visible_keys,
                .max_visible_keys             = max_visible_keys,
                .fast_prompt_kernel           = fast,
                .fast_prompt_pv8              = pv8,
                .prompt_split_workspace_bytes = split_workspace_bytes,
                .decode_split64               = decode_split64,
                .row_invariant_partition      = row_invariant_partition};
    }

    // Any launch's envelope with this schedule's decode fast path and partition rule.
    [[nodiscard]] ops::CausalAttentionExecutionEnvelope
    apply(ops::CausalAttentionExecutionEnvelope envelope) const noexcept {
        envelope.decode_split64          = decode_split64;
        envelope.row_invariant_partition = row_invariant_partition;
        return envelope;
    }

    // A target decode or verification round's envelope: without adaptive speculation each verify
    // column computes as that token decoded alone, so the round boundaries the proposals set do
    // not change a token's bits.
    [[nodiscard]] ops::CausalAttentionExecutionEnvelope
    verify(ops::CausalAttentionExecutionEnvelope envelope) const noexcept {
        envelope                  = apply(envelope);
        envelope.column_invariant = fixed_speculation;
        return envelope;
    }
};

// Rows of 17..64 verification columns keep 16-bit activations in FP8 residual projections at any
// batch size; first/last bound the per-row width, not the round's aggregate columns.
[[nodiscard]] inline bool wide_residual_verification(TextPhase phase, std::int32_t first,
                                                     std::int32_t last) {
    return phase == TextPhase::Verify && first > 16 && first <= last && last <= 64;
}

} // namespace infernix::models::qwen3_5

namespace infernix::models::qwen3_5::detail {
struct ContractAccess {
    static void constraint_failed(PendingBatch& pending, std::size_t row, bool failed) {
        pending.constraint_failed_.at(row) = failed;
    }

    static SequenceHandle make_sequence(const void* owner, runtime::LaneId lane,
                                        std::uint64_t epoch) noexcept {
        SequenceHandle out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        return out;
    }

    static const void* owner(const SequenceHandle& h) noexcept { return h.owner_; }

    static runtime::LaneId lane(const SequenceHandle& h) noexcept { return h.lane_; }

    static std::uint64_t epoch(const SequenceHandle& h) noexcept { return h.epoch_; }

    [[nodiscard]] static PendingBatch
    make_pending(const void* owner, std::uint64_t transaction, std::span<const SequenceHandle> rows,
                 std::span<const TokenId> tokens, std::span<const std::int32_t> row_counts,
                 std::uint32_t row_stride, runtime::ExecutionTiming timing) {
        PendingBatch out;
        out.owner_       = owner;
        out.transaction_ = transaction;
        out.row_count_   = rows.size();
        for (std::size_t i = 0; i < rows.size(); ++i) { out.rows_[i] = rows[i]; }
        out.tokens_     = tokens;
        out.row_counts_ = row_counts;
        out.row_stride_ = row_stride;
        out.timing_     = timing;
        return out;
    }

    [[nodiscard]] static const void* owner(const PendingBatch& pending) noexcept {
        return pending.owner_;
    }

    [[nodiscard]] static std::uint64_t transaction(const PendingBatch& pending) noexcept {
        return pending.transaction_;
    }

    [[nodiscard]] static std::span<const SequenceHandle>
    rows(const PendingBatch& pending) noexcept {
        return {pending.rows_.data(), pending.row_count_};
    }

    static void consume(PendingBatch& pending) noexcept {
        pending.owner_       = nullptr;
        pending.transaction_ = 0;
        pending.row_count_   = 0;
        pending.tokens_      = {};
        pending.row_counts_  = {};
        pending.row_stride_  = 0;
        pending.timing_      = {};
    }
};

[[nodiscard]] inline std::uint32_t backend_frontier_at(SpeculativeBackend backend,
                                                       std::uint32_t main_frontier) noexcept {
    if (backend == SpeculativeBackend::Mtp) { return main_frontier == 0 ? 0U : main_frontier - 1U; }
    return backend == SpeculativeBackend::DFlash ? main_frontier : 0U;
}

} // namespace infernix::models::qwen3_5::detail
