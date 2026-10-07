#pragma once

#include "core/layout.h"
#include "core/tensor.h"
#include "infernix/ops/sampling.h"
#include "infernix/ops/speculative_round.h"
#include "infernix/ops/token_constraint.h"
#include "infernix/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace infernix::models::qwen3_5 {

inline constexpr std::uint32_t kMtpDecodeMaximumDrafts    = 5;
inline constexpr std::uint32_t kMtpDecodeMaximumWidth     = kMtpDecodeMaximumDrafts + 1;
inline constexpr std::uint32_t kMtpVerifyMaximumDrafts    = 63;
inline constexpr std::uint32_t kMtpVerifyMaximumWidth     = kMtpVerifyMaximumDrafts + 1;
inline constexpr std::uint32_t kDFlashDecodeMaximumDrafts = 15;
inline constexpr std::uint32_t kDFlashDecodeMaximumWidth  = kDFlashDecodeMaximumDrafts + 1;
inline constexpr std::uint32_t kDFlashVerifyMaximumDrafts = 63;
inline constexpr std::uint32_t kDFlashVerifyMaximumWidth  = kDFlashVerifyMaximumDrafts + 1;

struct RoundStateSpec {
    std::int32_t hidden          = 0;
    std::int32_t output_rows     = 0;
    std::uint32_t batch_capacity = 1;
    std::uint32_t draft_window   = 0;
    SpeculativeBackend backend   = SpeculativeBackend::None;
    bool causal_scoring          = false;
};

// Stable pinned/device transfer format for ordinary decode. The full fixed-size object is copied
// once per round; only its exact-B prefixes are consumed by the model schedule.
struct OrdinaryDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> tokens{};
    std::array<std::int32_t, kMaximumConcurrency> cache_positions{};
    std::array<std::int32_t, kMaximumConcurrency> rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

struct OrdinaryDecodeEgress {
    std::array<TokenId, kMaximumConcurrency> sampled_tokens{};
};

// Stable pinned/device transfer formats for concurrent MTP decode. The arrays use the maximum
// product domain; RoundState binds only the configured [K,C] and [K+1,C] prefixes.
struct MtpDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> anchors{};
    std::array<std::int32_t, kMaximumConcurrency> base_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> remaining_budgets{};
    std::array<std::int32_t, kMaximumConcurrency> current_extents{};
    std::array<std::int32_t, kMaximumConcurrency> target_valid_columns{};
    std::array<TokenId, kMaximumConcurrency * kMtpVerifyMaximumDrafts> current_drafts{};
    std::array<std::int32_t, kMaximumConcurrency * kMtpVerifyMaximumWidth> target_rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> mtp_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<std::int32_t, kMaximumConcurrency> rope_deltas{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

struct MtpDecodeEgress {
    std::array<TokenId, kMaximumConcurrency * kMtpVerifyMaximumWidth> licensed_tokens{};
    std::array<std::int32_t, kMaximumConcurrency> licensed_counts{};
    std::array<std::int32_t, kMaximumConcurrency> accepted_drafts{};
    // Step-major: all B rows for proposal step 0, followed by all B rows for step 1, etc.
    std::array<TokenId, kMaximumConcurrency * kMtpDecodeMaximumDrafts> next_drafts{};
    std::array<std::int32_t, kMaximumConcurrency> next_extents{};
};

// Prefill binds the current chunk independently of the compact decode batch.
struct DFlashPrefillIngress {
    std::int32_t append_count           = 0;
    std::int32_t state_destination_slot = 0;
    std::int32_t full_kv_table_row      = 0;
};

// Stable pinned/device transfer formats for one exact-B DFlash transaction. The proposal is
// produced and verified in the same round, so no draft state crosses the round boundary.
struct DFlashDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> anchors{};
    std::array<std::int32_t, kMaximumConcurrency> execution_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> context_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> proposal_extents{};
    std::array<std::int32_t, kMaximumConcurrency> target_valid_columns{};
    std::array<std::int32_t, kMaximumConcurrency> proposal_valid_columns{};
    // DFlash uses logical positions for its own attention. Target verification carries a separate
    // continuation RoPE position so multimodal rows retain their per-sequence rope_delta.
    std::array<std::int32_t, kMaximumConcurrency * kDFlashVerifyMaximumWidth>
        target_rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> dflash_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> active_lanes{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
    // Rows whose proposal is the ngram copy payload below; other rows keep the neural drafter's
    // proposal. Read on the device, so one captured ngram-family graph serves every mixture.
    std::array<std::int32_t, kMaximumConcurrency> copy_rows{};
    // Neural rounds copy only the prefix before ngram_tokens. Keep all controls above it;
    // proposal-only payloads below it preserve the existing vector-aligned offsets. The arrays
    // are column-major per row (row stride = the round's draft width k, <= verify maximum):
    //   ngram_tokens[row * k + step];
    //   candidates/q[row * k * slots + step * slots + slot].
    std::array<TokenId, kMaximumConcurrency * kDFlashVerifyMaximumDrafts> ngram_tokens{};
    std::array<TokenId,
               kMaximumConcurrency * ops::kSparseSpeculativeCandidates * kDFlashVerifyMaximumDrafts>
        ngram_candidates{};
    std::array<float,
               kMaximumConcurrency * ops::kSparseSpeculativeCandidates * kDFlashVerifyMaximumDrafts>
        ngram_q{};
};

struct DFlashDecodeEgress {
    std::array<TokenId, kMaximumConcurrency * kDFlashVerifyMaximumWidth> licensed_tokens{};
    std::array<std::int32_t, kMaximumConcurrency> licensed_counts{};
    std::array<std::int32_t, kMaximumConcurrency> accepted_drafts{};
    // Tree rounds: the accepted root path's columns (row stride = the round width, -1 beyond A)
    // and the depth at which it left the main chain (0 when it stayed on it, -1 for a chain row).
    std::array<std::int32_t, kMaximumConcurrency * kDFlashVerifyMaximumWidth> accepted_path{};
    std::array<std::int32_t, kMaximumConcurrency> accepted_branch{};
};

struct OrdinaryDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion logits;
    TensorRegion hidden;
};

struct MtpPrefillStateLayout {
    TensorRegion position;
    TensorRegion ar_hidden;
    TensorRegion draft_tokens;
    TensorRegion target_input_ids;
    TensorRegion target_positions;
};

struct DFlashPrefillStateLayout {
    LayoutRegion ingress;
    TensorRegion local_append_count;
};

struct MtpDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion verify_ids;
    TensorRegion target_positions;
    TensorRegion target_argmax;
    TensorRegion target_logits;
    TensorRegion target_hidden;
    TensorRegion target_continuation_hidden;
    TensorRegion proposal_logits;
    TensorRegion alignment_ids;
    TensorRegion alignment_hidden;
    TensorRegion ar_hidden;
    TensorRegion next_hidden;
    TensorRegion ar_positions;
    TensorRegion ar_rope_positions;
    TensorRegion ar_valid_columns;
};

struct DFlashDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion proposal_ids;
    TensorRegion proposal_positions;
    TensorRegion verify_positions;
    std::optional<TensorRegion> candidate_ids;
    std::optional<TensorRegion> proposal_q;
    TensorRegion append_positions;
    TensorRegion append_counts;
    TensorRegion draft_tokens;
    TensorRegion verify_ids;
    TensorRegion target_argmax;
    TensorRegion target_logits;
    TensorRegion target_hidden;
    TensorRegion target_continuation_hidden;
    // DFlash2 tree rounds: each row's device-built tree (ops::SpeculativeTreeRow) and its I32
    // [W,B] ancestor masks.
    std::optional<TensorRegion> tree_rows;
    std::optional<TensorRegion> tree_masks;
};

// Constrained-generation controls (TokenConstraint): each lane's permitted sets, the per-column
// descriptors of the round being executed and the probability records it produces.
struct TokenConstraintStateLayout {
    TensorRegion choices;
    TensorRegion counts;
    TensorRegion descriptors;
    TensorRegion records;
};

// Most logits columns one round constrains: every row at the widest verification.
inline constexpr std::int32_t kTokenConstraintColumns = static_cast<std::int32_t>(
    kMaximumConcurrency * std::max(kDFlashVerifyMaximumWidth, kMtpVerifyMaximumWidth));

struct RoundStateLayout {
    RoundStateSpec spec;
    std::optional<TokenConstraintStateLayout> constraint;
    std::optional<OrdinaryDecodeStateLayout> ordinary;
    TensorRegion token;
    TensorRegion pos;
    TensorRegion rope_pos;
    TensorRegion rope_delta;
    TensorRegion logits;
    TensorRegion text_kv_table_row;
    TensorRegion backend_kv_table_row;
    std::optional<MtpPrefillStateLayout> mtp;
    std::optional<DFlashPrefillStateLayout> dflash_prefill;
    std::optional<MtpDecodeStateLayout> mtp_decode;
    std::optional<DFlashDecodeStateLayout> dflash_decode;
    bool complete = false;
};

struct OrdinaryDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor tokens;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor text_kv_table_rows;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor sampled_tokens;
    Tensor logits;
    Tensor hidden;

    OrdinaryDecodeState() = default;
    OrdinaryDecodeState(DeviceSpan backing, const OrdinaryDecodeStateLayout& layout,
                        std::uint32_t batch_capacity);
};

// The two planning calls expose one deliberate exact-target extension seam after scalar logits.
// This lets a target retain its schedule-sized prefill activation at the established physical
// address without making that activation part of the family round contract.
[[nodiscard]] RoundStateLayout begin_round_state_layout(LayoutBuilder& builder,
                                                        const RoundStateSpec& spec);
void complete_round_state_layout(LayoutBuilder& builder, RoundStateLayout& layout);

struct MtpPrefillState {
    Tensor position;
    Tensor ar_hidden;
    Tensor draft_tokens;
    Tensor target_input_ids;
    Tensor target_positions;

    MtpPrefillState() = default;
    MtpPrefillState(DeviceSpan backing, const MtpPrefillStateLayout& layout);
};

struct DFlashPrefillState {
    DeviceSpan ingress;
    Tensor append_count;
    Tensor state_destination_slot;
    Tensor full_kv_table_row;
    // Oversized chunks retain only their tail locally; full KV still uses append_count.
    Tensor local_append_count;

    DFlashPrefillState() = default;
    DFlashPrefillState(DeviceSpan backing, const DFlashPrefillStateLayout& layout);
};

struct MtpDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor anchors;
    Tensor base_frontiers;
    Tensor remaining_budgets;
    Tensor current_extents;
    Tensor target_valid_columns;
    Tensor current_drafts;
    Tensor target_rope_positions;
    Tensor text_kv_table_rows;
    Tensor mtp_kv_table_rows;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    Tensor rope_deltas;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor next_drafts;
    Tensor next_extents;
    Tensor verify_ids;
    Tensor target_positions;
    Tensor target_argmax;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor target_continuation_hidden;
    Tensor proposal_logits;
    Tensor alignment_ids;
    Tensor alignment_hidden;
    Tensor ar_hidden;
    Tensor next_hidden;
    Tensor ar_positions;
    Tensor ar_rope_positions;
    Tensor ar_valid_columns;

    MtpDecodeState() = default;
    MtpDecodeState(DeviceSpan backing, const MtpDecodeStateLayout& layout,
                   std::uint32_t batch_capacity, std::uint32_t draft_window);
    // The frame viewed for a round that verifies k drafts per row and proposes next_k: every round
    // tensor becomes a dense [k+1,C] (or [k,C]) view of the native storage and the step-major AR
    // tensors keep their row stride.
    [[nodiscard]] MtpDecodeState narrowed(std::uint32_t k, std::uint32_t next_k) const;
};

struct DFlashDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor anchors;
    Tensor execution_frontiers;
    Tensor context_frontiers;
    Tensor proposal_extents;
    Tensor target_valid_columns;
    Tensor proposal_valid_columns;
    Tensor target_rope_positions;
    Tensor text_kv_table_rows;
    Tensor dflash_kv_table_rows;
    Tensor active_lanes;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor tree_rows;
    Tensor tree_masks;
    Tensor accepted_path;
    Tensor accepted_branch;
    Tensor proposal_ids;
    Tensor proposal_positions;
    Tensor verify_positions;
    Tensor candidate_ids;
    Tensor proposal_q;
    Tensor append_positions;
    Tensor append_counts;
    Tensor draft_tokens;
    Tensor verify_ids;
    Tensor target_argmax;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor target_continuation_hidden;

    DFlashDecodeState() = default;
    DFlashDecodeState(DeviceSpan backing, const DFlashDecodeStateLayout& layout,
                      std::uint32_t batch_capacity, std::uint32_t draft_window);
    // The same storage viewed as a dense round of k drafts per row, for every row capacity; the
    // append-context geometry keeps its native width. k equal to the native width is identity.
    [[nodiscard]] DFlashDecodeState narrowed(std::uint32_t k) const;
};

struct TokenConstraintState {
    // I32 [kTokenConstraintChoices, lanes * kMaximumConstraintSets]: set s of lane l is column
    // l * kMaximumConstraintSets + s; counts is I32 [lanes * kMaximumConstraintSets].
    Tensor choices;
    Tensor counts;
    // I32 [kTokenConstraintColumns], -1 where unconstrained (ops::constrain_logits encoding).
    Tensor descriptors;
    // FP32 [kTokenConstraintRecord, kTokenConstraintColumns].
    Tensor records;

    // The first `columns` descriptors and records, for a logits view of that many columns.
    [[nodiscard]] Tensor descriptor_view(std::int32_t columns) const {
        return descriptors.slice(0, 0, columns);
    }
    [[nodiscard]] Tensor record_view(std::int32_t columns) const {
        return records.slice(1, 0, columns);
    }
};

struct RoundState {
    std::optional<OrdinaryDecodeState> ordinary;
    std::optional<TokenConstraintState> constraint;
    Tensor token;
    Tensor pos;
    Tensor rope_pos;
    Tensor rope_delta;
    Tensor logits;
    Tensor text_kv_table_row;
    Tensor backend_kv_table_row;
    std::optional<MtpPrefillState> mtp;
    std::optional<DFlashPrefillState> dflash_prefill;
    std::optional<MtpDecodeState> mtp_decode;
    std::optional<DFlashDecodeState> dflash_decode;

    RoundState() = default;
    RoundState(DeviceSpan backing, const RoundStateLayout& layout);
};

// Applies the staged constraint descriptors to `logits` ([rows,C], or [rows,K+1,B] viewed as
// C = (K+1)*B columns) before a sampler or acceptance reads it; `argmax` (I32, C elements, or
// null) is a precomputed greedy choice to keep consistent. No-op without constraint controls.
void constrain_round_logits(const RoundState& io, const Tensor& logits, const Tensor* argmax,
                            std::int32_t token_domain, cudaStream_t stream);

} // namespace infernix::models::qwen3_5
