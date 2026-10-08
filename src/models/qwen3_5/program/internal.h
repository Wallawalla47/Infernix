#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/program.h"
#include "infernix/ops/softmax_attention.h"

#include <cstddef>
#include <cstdint>

namespace infernix::models::qwen3_5 {

inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 5;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 15;

// Prompt-attention settings of every prefill chunk (CausalAttentionExecutionEnvelope): fast selects
// the storage's fast prompt kernel (INT8 or NVFP4 KV); pv8 runs the prompt kernel's PV on 8-bit
// Tensor Cores (the fast INT8 or NVFP4 kernel, or K8V4's prompt kernel, which ignores fast); and
// split_workspace_bytes bounds the FP32 partials of a launch's key splits. Workspace planning and
// every chunk use the same settings.
struct PromptAttention {
    bool fast                         = false;
    bool pv8                          = false;
    std::size_t split_workspace_bytes = ops::kCausalPromptSplitWorkspaceDefaultBytes;

    [[nodiscard]] ops::CausalAttentionExecutionEnvelope
    envelope(std::uint32_t min_visible_keys, std::uint32_t max_visible_keys) const noexcept {
        return {.min_visible_keys             = min_visible_keys,
                .max_visible_keys             = max_visible_keys,
                .fast_prompt_kernel           = fast,
                .fast_prompt_pv8              = pv8,
                .prompt_split_workspace_bytes = split_workspace_bytes};
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
