#pragma once

#include "models/qwen3_5/execution/lossless.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/program.h"
#include "infernix/ops/gated_delta_net.h"
#include "infernix/ops/offloaded_sparse_moe.h"
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
// NVFP4 projections take the artifact's 8- or 4-bit activations, a8_subset_decode the 8-bit ones in
// rounds of A8Subset::min_columns or more for A8Subset's projections only, a8_residual_decode the
// FP8 projections the A8 residual profile and a4_residual_decode the NVFP4 MLP projections the A4
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
    bool bf16_prompt_split            = false;
    std::size_t split_workspace_bytes = ops::kCausalPromptSplitWorkspaceDefaultBytes;
    bool decode_split64               = false;
    bool column_invariant_decode      = false;
    // single-column-decode: decode and verification calls of one column run the batched route.
    bool single_column_decode         = false;
    bool a8_decode                    = true;
    bool a4_decode                    = true;
    bool a8_residual_decode           = false;
    bool a4_residual_decode           = false;
    // batched-a8-subset-decode: calls of a8_subset_min_columns or more columns
    // (temporary_switches::A8Subset, which also sets its projection classes and layers).
    bool a8_subset_decode             = false;
    std::int32_t a8_subset_min_columns = 0;
    bool row_invariant_partition      = false;
    bool fixed_speculation            = false;
    bool grid_cache                   = false;
    bool held_extent                  = false;
    // adaptive-mtp: MTP rounds choose their draft length (only where speculation is adaptive).
    bool adaptive_mtp                 = false;
    // adaptive-dflash: DFlash and DFlash2 all-neural rounds propose, then choose their verify
    // width (only where speculation is adaptive).
    bool adaptive_dflash              = false;
    // MTP rounds write each next draft's drafter probability (ops::argmax_with_probability):
    // MtpConfidencePolicy's input with offloaded experts, and on resident targets the input of the
    // cut, the gate and the calibrated extents below (or alone under INFERNIX_TMP_QWEN35_MTP_PROBS).
    bool mtp_draft_confidence         = false;
    // --spec-min-p on Qwen3.5 MTP: a one-row round verifies its drafts up to the first whose drafter
    // probability is below it (0: every draft). With adaptive-mtp the round runs the family of that
    // width.
    double mtp_min_p                  = 0.0;
    // One-row neural MTP rounds propose through the gated chain (mtp_chained_round): the verify width
    // (the family) comes from the drafts' cut or calibrated extent and the proposal depth from the
    // draft-length policy, which a device limit applies. adaptive-mtp with the cut or the
    // calibrated extents, argmax drafts.
    bool mtp_proposal_chain           = false;
    // The chain also stops on the device after the first draft below mtp_min_p, skipping the steps
    // whose drafts the cut discards (INFERNIX_TMP_QWEN35_MTP_GATE; B-01's draft gate).
    bool mtp_gate                     = false;
    // A one-row chained round verifies the extent its drafts' calibrated acceptance makes fastest
    // (MtpConfidencePolicy over the cut prefix; INFERNIX_TMP_QWEN35_MTP_CALIBRATED).
    bool mtp_calibrated_extent        = false;
    // TEMPORARY (INFERNIX_TMP_QWEN35_MTP_PAD_MASK, candidates §11 OR-09): a batched drafter call's
    // columns past each row's valid count route no resident expert (the 122B drafter's MoE).
    bool mtp_drafter_mask             = false;
    // TEMPORARY (INFERNIX_TMP_QWEN35_MTP_STEP_GATE): offloaded MTP rounds (MtpConfidencePolicy's
    // extents) propose through the step chain: the alignment, then each draft behind a device gate
    // that runs it while some row's proposal limit admits it and its previous draft clears the
    // policy's step threshold (MtpConfidencePolicy::step_threshold), past which no later draft
    // could be verified; a row's drafts end at its first one below it.
    bool mtp_step_chain               = false;
    // TEMPORARY (INFERNIX_TMP_QWEN35_MTP_SKIP_ROUNDS): with the step chain, a round may run its
    // alignment alone (proposal limit 0: the drafter's KV, no draft) while MtpDraftRegime measures
    // such rounds faster.
    bool mtp_draft_regime             = false;
    // sampled-mtp-drafts: MTP drafts are drawn from the draft head's top kMtpSampledDraftRanks
    // tokens under each row's sampler and verified by sparse rejection sampling against that
    // proposal (greedy rows draw the argmax with a one-hot proposal); its rounds verify every draft.
    bool sampled_mtp_drafts           = false;
    ops::LinearRoutes linear_routes   = ops::LinearRoutes::Exact;
    // q5-ksplit-mma: decode and verification calls of the Q5 output and MLP down projections take
    // ops::LinearRoutes::Q5KSplit (column-invariant where the policy is).
    bool q5_ksplit = false;
    // gdn-f16-accumulate: the chunked GDN calls' arithmetic.
    ops::GdnChunkArithmetic gdn_arithmetic = ops::GdnChunkArithmetic::Exact;
    // fp16-gdn-state: the state pool stores the GDN recurrent state as FP16 (half its traffic and
    // memory); every GDN call rounds the state it publishes once (ops::gated_delta_net).
    bool fp16_gdn_state = false;
    // block-verification: sampled rows of sparse-proposal chain rounds accept by block verification
    // (ops::speculative_accept_sparse_drafts).
    bool block_verification = false;
    // univer-trees: the drafter builds sampled tree rows for UniVer (SpeculativeTreeShape::transport),
    // which tree acceptance then applies (SpeculativeTreeRow::transport).
    bool univer_trees = false;
    // gdn-tree-closed-form: tree rounds' GDN records in closed form (ops::GdnTreeRecordRoute::ClosedForm).
    bool gdn_tree_closed_form = false;
    // TEMPORARY (INFERNIX_TMP_TREE_ROUND_WIDTH, candidates §13 PUB-03): with DFlash2 and
    // --draft-tree-nodes auto (not adaptive-dflash), each automatic round first proposes (the
    // proposal graph keeps the block's lattice), then the controller chooses its width from the
    // round's own draft probabilities (TreeWidthController::choose's rows) and a split family verifies
    // the chain or builds and verifies the tree.
    bool tree_round_width = false;
    // expert-a4-residual (-3): the offloaded experts' wide route (Qwen3.5-122B-A10B's W4A16 banks;
    // ExecutionSchedule::lossless runs its own). Its workspace is planned with the same route.
    ops::offloaded_moe::WideA16 expert_wide = ops::offloaded_moe::WideA16::kExact;
    // router-mma: the offloaded MoE layers' prefill calls (OffloadedExperts::router_mma) take the
    // tensor-core router (ops::projection_fp32_router_mma), as Flash-Next's do.
    bool router_mma = false;
    // The lossless route (EngineOptions::lossless, docs/maintainer/lossless-route.md): every layer
    // execution::lossless_layer, the head's exact sums correctly rounded to `logits`. A CausalScoring Program scores each window in one pass (the
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
    // Every main-head logits buffer's dtype (EngineOptions::logits_dtype): FP32, or BF16.
    DType logits = DType::FP32;

    // Whether a decode or verification call of `columns` columns runs batched-decode's route: its
    // projections' column-count routes (Parameters::decode_text without column invariance) and the
    // MLP down projections' tiles (ops::LinearRoutes::WideDownTile).
    [[nodiscard]] bool batched_columns(std::int64_t columns) const noexcept {
        return !column_invariant_decode || (single_column_decode && columns == 1);
    }

    // The routes of a decode or verification call's residual projections (output, MLP down):
    // batched-decode's down tiles where its route runs, q5-ksplit-mma's K-split tiles where selected.
    [[nodiscard]] ops::LinearRoutes residual_decode_routes(std::int64_t columns, bool down) const noexcept {
        ops::LinearRoutes routes = down && batched_columns(columns) ? ops::LinearRoutes::WideDownTile
                                                                    : ops::LinearRoutes::Exact;
        if (q5_ksplit) { routes = routes | ops::LinearRoutes::Q5KSplit; }
        return routes;
    }

    // The activation precisions a batched decode or verification call of `columns` columns gives its
    // projections (Parameters::decode_text).
    [[nodiscard]] execution::DecodePrecision decode_precision(std::int64_t columns) const noexcept {
        return {.a8          = a8_decode,
                .a4          = a4_decode,
                .a8_residual = a8_residual_decode,
                .a4_residual = a4_residual_decode,
                .a8_subset   = a8_subset_decode && columns >= a8_subset_min_columns};
    }

    // A prefill chunk's envelope.
    [[nodiscard]] ops::CausalAttentionExecutionEnvelope
    envelope(std::uint32_t min_visible_keys, std::uint32_t max_visible_keys) const noexcept {
        return {.min_visible_keys             = min_visible_keys,
                .max_visible_keys             = max_visible_keys,
                .fast_prompt_kernel           = fast,
                .fast_prompt_pv8              = pv8,
                .bf16_prompt_split            = bf16_prompt_split,
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
    static void row_failure(PendingBatch& pending, std::size_t row,
                            runtime::PendingRowFailure failure) {
        pending.row_failures_.at(row) = failure;
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
