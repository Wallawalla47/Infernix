#pragma once
#include "models/qwen3_5/program/internal.h"

#include "core/cyclic_kv_cache.h"
#include "core/dtype.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/program/speculative/dflash_width_policy.h"
#include "models/qwen3_5/state/state_image.h"
#include "models/load_options.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace infernix::models::qwen3_5::detail {

using TensorLayout                              = TensorRegion;

// Where a speculative round's proposals come from: the backend's neural drafter, n-gram copies
// (rows of a batched copy round without a copy keep their neural proposal), or a DFlash2 draft
// tree built from the drafter's lattice (verify_drafts = tree columns - 1).
enum class SpeculativeRoundKind : std::uint8_t { Neural, Ngram, Tree };

// One CUDA-Graph family of speculative decode rounds. Every round of the family verifies
// verify_drafts proposals per row on the decode frame viewed at that width and records ReplaySSM
// transitions through the record view of the same width.
struct SpeculativeRoundShape {
    SpeculativeRoundKind kind   = SpeculativeRoundKind::Neural;
    std::uint32_t verify_drafts = 0;
    // The family serves only rounds of one row (adaptive-mtp draft lengths, adaptive-dflash widths
    // outside the batch lengths: mtp_draft_policy.h, dflash_width_policy.h).
    bool single_row = false;
    // adaptive-dflash: the family's Forward only verifies; the round's proposal graph drafted the
    // whole block first (ProgramImpl::dflash_propose_graphs) and its width was chosen from it.
    bool split = false;
    // adaptive-dflash with a batch length (fast_paths.h, dflash_batch_drafts): the plain neural
    // family rounds of several rows run, drafting and verifying that length; it serves no one row.
    bool batch_only = false;
};

// The drafts an MTP round of `shape` proposes for the next round. A neural round proposes at its
// own draft length (the configured window, or with adaptive-mtp each length's own family); a copy
// round proposes at the configured window.
[[nodiscard]] inline std::uint32_t mtp_proposal_drafts(const SpeculativeRoundShape& shape,
                                                       std::uint32_t neural_draft_window) noexcept {
    return shape.kind == SpeculativeRoundKind::Neural ? shape.verify_drafts : neural_draft_window;
}

// Whether a round of the family with `batch_size` rows proposes through the gated proposal chain
// (ExecutionSchedule::mtp_proposal_chain): a one-row neural round then proposes up to the
// configured window, and the device stops it at the round's proposal limit (the draft-length
// policy's choice) and, with mtp_gate, after the first draft below --spec-min-p, so its verify width
// (the family) and its proposal depth are chosen apart.
[[nodiscard]] inline bool mtp_chained_round(const SpeculativeRoundShape& shape, std::uint32_t batch_size,
                                            bool chain) noexcept {
    return chain && shape.kind == SpeculativeRoundKind::Neural && batch_size == 1U;
}

// The drafts a round of the family with `batch_size` rows proposes (its graph's proposal steps).
[[nodiscard]] inline std::uint32_t mtp_proposal_drafts(const SpeculativeRoundShape& shape,
                                                       std::uint32_t neural_draft_window,
                                                       std::uint32_t batch_size, bool chain) noexcept {
    return mtp_chained_round(shape, batch_size, chain) ? neural_draft_window
                                                       : mtp_proposal_drafts(shape, neural_draft_window);
}

// The deepest proposal of any of the family's rounds, which its graph profiles and attention
// envelopes cover.
[[nodiscard]] inline std::uint32_t mtp_family_proposal_drafts(const SpeculativeRoundShape& shape,
                                                              std::uint32_t neural_draft_window,
                                                              bool chain) noexcept {
    return mtp_proposal_drafts(shape, neural_draft_window, 1U, chain);
}

// The tree widths (columns per row, anchor included) each batch size's all-neural rounds
// may verify. A fixed table gives a batch size one tree width or the chain; automatic mode lets
// rounds of up to automatic_max_batch rows choose between the chain and every automatic width.
struct TreeWidthPlan {
    std::array<std::uint32_t, kMaximumConcurrency> fixed{};
    std::vector<std::uint32_t> automatic; // ascending; empty outside automatic mode
    std::uint32_t automatic_max_batch = 0;

    [[nodiscard]] bool automatic_mode() const noexcept { return !automatic.empty(); }

    // Whether an all-neural round of batch_size rows may verify a tree of `columns` columns.
    [[nodiscard]] bool tree(std::uint32_t batch_size, std::uint32_t columns) const noexcept {
        if (automatic_mode()) {
            return batch_size <= automatic_max_batch &&
                   std::find(automatic.begin(), automatic.end(), columns) != automatic.end();
        }
        return columns != 0 && fixed[batch_size - 1U] == columns;
    }

    // Whether an all-neural round of batch_size rows may verify a chain.
    [[nodiscard]] bool chain(std::uint32_t batch_size) const noexcept {
        return automatic_mode() || fixed[batch_size - 1U] == 0;
    }

    // Every tree width some batch size may verify, ascending.
    [[nodiscard]] std::vector<std::uint32_t> widths() const {
        std::vector<std::uint32_t> out = automatic;
        for (const std::uint32_t columns : fixed) {
            if (columns != 0 && std::find(out.begin(), out.end(), columns) == out.end()) {
                out.push_back(columns);
            }
        }
        std::sort(out.begin(), out.end());
        return out;
    }
};

// Whether the round family of `shape` serves rounds of batch_size rows: a tree family only the batch
// sizes that may verify its width, a single-row family one row, a batch-only family every other batch
// size. Chain families serve every batch size: a round with a constrained row verifies the chain
// whatever the tree table selects.
[[nodiscard]] inline bool round_shape_serves(const SpeculativeRoundShape& shape, const TreeWidthPlan& tree_widths,
                                             std::uint32_t batch_size) noexcept {
    return (shape.kind != SpeculativeRoundKind::Tree || tree_widths.tree(batch_size, shape.verify_drafts + 1U)) &&
           (!shape.single_row || batch_size == 1U) && (!shape.batch_only || batch_size > 1U);
}

// The most target columns one round verifies (RoundStateSpec::verify_columns): each family's width
// times the most rows it serves; 0 without speculative rounds.
[[nodiscard]] inline std::uint32_t verify_column_capacity(const std::vector<SpeculativeRoundShape>& shapes,
                                                          const TreeWidthPlan& tree_widths,
                                                          std::uint32_t max_concurrency) noexcept {
    std::uint32_t columns = 0;
    for (const SpeculativeRoundShape& shape : shapes) {
        for (std::uint32_t batch = max_concurrency; batch != 0; --batch) {
            if (round_shape_serves(shape, tree_widths, batch)) {
                columns = std::max(columns, (shape.verify_drafts + 1U) * batch);
                break;
            }
        }
    }
    return columns;
}

// The columns one causal-scoring tile holds; its logits are [vocabulary, tile].
inline constexpr std::uint32_t kCausalScoreTile = 512;

struct DFlashPersistentLayout {
    std::optional<qwen3_5::PagedKVCacheLayout> full;
    TensorLayout prefill_features;
    TensorLayout prefill_positions;
    TensorLayout pending_features;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept {
        return full ? full->payload_bytes() : 0;
    }
};

struct PersistentLayout {
    qwen3_5::DecoderStateLayout decoder;
    qwen3_5::StateImageDeviceLayout state_images;
    std::optional<GdnReplayRecordLayout> replay_records;
    std::optional<DFlashPersistentLayout> dflash;
    qwen3_5::RoundStateLayout round;
    TensorLayout prefill_hidden;
    std::optional<TensorLayout> score_hidden;
    std::optional<TensorLayout> token_counts;
    std::optional<TensorLayout> sampling_config;
    std::optional<TensorLayout> grammar_masks;
    std::size_t bytes            = 0;
    std::size_t kv_payload_bytes = 0;
    // Elastic text KV (offloaded experts; Qwen3.5-122B design §11.1): the text KV page planes in their
    // own region of text_kv_region_bytes, reserved as virtual memory, its pages [0, text_kv_base_pages)
    // backed at startup and the rest taken from the expert frames as requests need them (ProgramImpl::
    // grow_kv). 0: the planes are in this layout (`bytes`). The plan's device reservation counts the
    // whole region, so the KV can always grow to its capacity by taking frames back.
    std::size_t text_kv_region_bytes = 0;
    std::uint32_t text_kv_base_pages = 0;
};

// The text KV pages an elastic pool backs at startup: 32,768 tokens (Qwen3.8-Flash-Next's base).
inline constexpr std::uint32_t kElasticKvBaseTokens = 32768;

struct VisionWorkspacePlan {
    std::int32_t output_hidden         = 0;
    std::uint32_t max_merged_tokens    = 0;
    std::size_t general_capacity_bytes = 0;
    std::size_t encode_peak_bytes      = 0;
    std::size_t handoff_offset_bytes   = 0;
    std::size_t handoff_capacity_bytes = 0;
    std::size_t capacity_bytes         = 0;
};

struct WorkspacePlan {
    std::size_t text_prefill     = 0;
    // With offloaded experts: a full-width text (MTP) prefill pass, lent by the expert cache for each
    // pass wider than kOffloadedMoeStaticColumns (text_prefill and mtp_prefill then plan that width);
    // 0 otherwise.
    std::size_t text_prefill_wide = 0;
    std::size_t mtp_prefill_wide  = 0;
    // With lent passes: the wider of the two plans at widths 512, 1,024, ... below the chunk, then the chunk
    // (ascending), so a pass can borrow what its width needs (INFERNIX_TMP_PROMPT_SIZED_LOAN).
    std::vector<std::pair<std::int32_t, std::size_t>> prefill_wide_ladder;
    std::size_t ordinary_round   = 0;
    std::size_t mtp_prefill      = 0;
    std::size_t mtp_round        = 0;
    std::size_t dflash_context   = 0;
    std::size_t dflash_round     = 0;
    std::size_t causal_score     = 0;
    std::size_t general_capacity = 0;
    std::optional<VisionWorkspacePlan> vision;
    std::size_t capacity = 0;
};

struct SequencePlanningInputs {
    const execution::Parameters* parameters = nullptr;
    std::uint32_t neural_draft_window       = 0;
    // Every speculative round family the engine captures; draft_window is their widest width.
    std::vector<SpeculativeRoundShape> round_shapes;
    // adaptive-dflash: the configuration's measured round costs (with split round shapes).
    DFlashWidthCosts dflash_width_costs;
    // Tree verification: the tree widths of each batch size and the most root-to-leaf
    // paths a tree row may hold.
    TreeWidthPlan tree_widths;
    std::uint32_t draft_tree_paths          = 0;
    std::uint32_t ngram_draft_window        = 0;
    // The copy window of rounds of several rows (0: ngram_draft_window).
    std::uint32_t ngram_batch_window        = 0;
    std::uint32_t ngram_min_match           = 12;
    std::uint32_t capacity                  = 0;
    std::uint32_t max_concurrency           = 1;
    std::uint32_t prefill_chunk             = 0;
    // Widest staged prefill step (idle-wide-prefill), a multiple of prefill_chunk; prefill_chunk
    // when the fast path is off. Prefill buffers and the prefill workspace are planned at it.
    std::uint32_t idle_prefill_width        = 0;
    ExecutionSchedule execution_schedule;
    std::uint32_t draft_window              = 0;
    SpeculativeBackend speculative_backend  = SpeculativeBackend::None;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    ProposalHead proposal_head              = ProposalHead::Full;
    models::LoadOptions features;
    bool use_cuda_graph               = true;
    bool causal_scoring               = false;
    int device                        = 0;
    std::int32_t multiprocessor_count = 0;
    ContextCacheOptions context_cache;
};

struct SequencePlanImpl {
    const execution::Parameters* parameters = nullptr;
    std::uint32_t neural_draft_window       = 0;
    std::vector<SpeculativeRoundShape> round_shapes;
    DFlashWidthCosts dflash_width_costs;
    // Tree verification: the tree widths of each batch size and the most root-to-leaf
    // paths a tree row may hold.
    TreeWidthPlan tree_widths;
    std::uint32_t draft_tree_paths          = 0;
    std::uint32_t ngram_draft_window        = 0;
    std::uint32_t ngram_batch_window        = 0;
    std::uint32_t ngram_min_match           = 12;
    std::uint32_t capacity                  = 0;
    std::uint32_t kv_capacity               = 0;
    std::uint32_t main_page_groups          = 0;
    std::uint32_t max_concurrency           = 1;
    std::uint32_t prefill_chunk             = 0;
    std::uint32_t idle_prefill_width        = 0;
    ExecutionSchedule execution_schedule;
    std::uint32_t draft_window              = 0;
    SpeculativeBackend speculative_backend  = SpeculativeBackend::None;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    ProposalHead proposal_head              = ProposalHead::Full;
    models::LoadOptions features;
    bool use_cuda_graph               = true;
    bool causal_scoring               = false;
    int device                        = 0;
    std::int32_t multiprocessor_count = 0;
    ContextCacheOptions context_cache;
    PersistentLayout persistent;
    WorkspacePlan workspace;
    std::size_t graph_allowance_bytes    = 0;
    std::size_t device_reservation_bytes = 0;
};

struct SequencePlannerImpl {
    SequencePlanningInputs inputs;
    runtime::SequenceCapacityCurve curve;
    std::unique_ptr<SequencePlanImpl> minimum;
};

// `idle_prefill_width` plans an idle-wide-prefill width (one of idle_prefill_widths_impl); zero
// plans prefill steps of the chunk only.
[[nodiscard]] std::unique_ptr<qwen3_5::detail::SequencePlannerImpl>
make_sequence_planner_impl(const execution::Parameters& parameters, DeviceContext& device,
                           const EngineOptions& options, std::uint32_t idle_prefill_width = 0);
// The idle-wide-prefill widths a plan may take, widest first (planning/idle_prefill.h); empty
// unless the fast path is selected and some width is wider than the chunk.
[[nodiscard]] std::vector<std::uint32_t>
idle_prefill_widths_impl(const execution::Parameters& parameters, const EngineOptions& options);
// qwen3_5::resolve_auto_draft_length (program.h).
[[nodiscard]] std::uint32_t resolve_auto_draft_length_impl(const execution::Parameters& parameters,
                                                           EngineOptions& options);
// qwen3_5::resolve_auto_ngram (program.h).
[[nodiscard]] std::uint32_t resolve_auto_ngram_impl(const Model& model, EngineOptions& options);

// TEMPORARY A/B switches of the Qwen3.5 MTP drafting package (rank 5 of the 2026-10-21 candidates),
// read once from the environment ("1" sets a flag); each defaults to today's behaviour:
//   INFERNIX_TMP_QWEN35_MTP_PROBS       resident targets' drafts carry their drafter probability
//                                       (ExecutionSchedule::mtp_draft_confidence) with nothing reading it,
//                                       which isolates its cost; the cut, gate and calibrated extents imply it
//   INFERNIX_TMP_QWEN35_MTP_GATE        the proposal chain stops at --spec-min-p on the device (mtp_gate)
//   INFERNIX_TMP_QWEN35_MTP_CALIBRATED  one-row rounds verify their calibrated extent (mtp_calibrated_extent)
//   INFERNIX_TMP_QWEN35_MTP_DRAFT_LOG   a file path: one line per verified MTP row (the offline cut replay)
// and of the offloaded-expert drafter's round cost (candidates §11 OR-09; output-neutral):
//   INFERNIX_TMP_QWEN35_MTP_PAD_MASK    the drafter's resident MoE skips each row's padding columns
//                                       (mtp_drafter_mask)
//   INFERNIX_TMP_QWEN35_MTP_STEP_GATE   offloaded MTP rounds propose through the step chain, whose device gate
//                                       stops where no later draft could be verified (mtp_step_chain)
//   INFERNIX_TMP_QWEN35_MTP_SKIP_ROUNDS with the step chain, rounds also run without drafting (the drafter's
//                                       KV only) while that measures faster (mtp_draft_regime)
struct MtpDraftSwitches {
    bool probabilities = false;
    bool gate          = false;
    bool calibrated    = false;
    bool pad_mask      = false;
    bool step_gate     = false;
    bool skip_rounds   = false;
    std::string draft_log;
};
[[nodiscard]] const MtpDraftSwitches& mtp_draft_switches();
// qwen3_5::mtp_draft_switch_report (program.h).
[[nodiscard]] std::string mtp_draft_switch_report_impl(const execution::Parameters& parameters,
                                                       const EngineOptions& options);
[[nodiscard]] std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_5::detail::SequencePlannerImpl> planner,
                            std::uint32_t main_page_groups);

} // namespace infernix::models::qwen3_5::detail
