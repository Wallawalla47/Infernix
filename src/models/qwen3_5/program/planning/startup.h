#pragma once
#include "models/qwen3_5/program/internal.h"

#include "core/cyclic_kv_cache.h"
#include "core/dtype.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/state/state_image.h"
#include "models/load_options.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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
};

// The DFlash2 tree widths (columns per row, anchor included) each batch size's all-neural rounds
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
inline constexpr std::uint32_t kCausalScoreTile = 1024;

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
    std::size_t bytes            = 0;
    std::size_t kv_payload_bytes = 0;
};

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
    // DFlash2 tree verification: the tree widths of each batch size and the most root-to-leaf
    // paths a tree row may hold.
    TreeWidthPlan tree_widths;
    std::uint32_t draft_tree_paths          = 0;
    std::uint32_t ngram_draft_window        = 0;
    std::uint32_t ngram_min_match           = 12;
    std::uint32_t capacity                  = 0;
    std::uint32_t max_concurrency           = 1;
    std::uint32_t prefill_chunk             = 0;
    PromptAttention prompt_attention;
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
    // DFlash2 tree verification: the tree widths of each batch size and the most root-to-leaf
    // paths a tree row may hold.
    TreeWidthPlan tree_widths;
    std::uint32_t draft_tree_paths          = 0;
    std::uint32_t ngram_draft_window        = 0;
    std::uint32_t ngram_min_match           = 12;
    std::uint32_t capacity                  = 0;
    std::uint32_t kv_capacity               = 0;
    std::uint32_t main_page_groups          = 0;
    std::uint32_t max_concurrency           = 1;
    std::uint32_t prefill_chunk             = 0;
    PromptAttention prompt_attention;
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

[[nodiscard]] std::unique_ptr<qwen3_5::detail::SequencePlannerImpl>
make_sequence_planner_impl(const execution::Parameters& parameters, DeviceContext& device,
                           const EngineOptions& options);
[[nodiscard]] std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_5::detail::SequencePlannerImpl> planner,
                            std::uint32_t main_page_groups);

} // namespace infernix::models::qwen3_5::detail
