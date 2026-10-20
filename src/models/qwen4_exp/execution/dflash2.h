#pragma once

// The DFlash2 drafter of Qwen3.8-Flash-Next (docs/maintainer/dflash.md): one masked block of K + 1
// columns (the anchor and K masks) through the drafter's sliding-window layers with dynamic
// convolutions, conditioned on a per-lane context ring of K/V projected from the target's features
// (the mean of the four hyper-connection streams of completed layers), then the target's head (or
// the proposal head) and the conditional candidate selector. The Program owns the ring, the
// features and every io word; this class owns no state.

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "core/tensor.h"
#include "infernix/ops/sampling.h"
#include "infernix/types.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/execution/parameters.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infernix::models::qwen4_exp::execution {

// Committed target features into the ring: row b's first counts[b] columns (positions[i, b], a
// sequential prefix) are stored in lane slots[b]; the other columns have no effect.
struct DFlash2ContextCall {
    Tensor features;  // BF16 [taps * H, W, B]
    Tensor positions; // I32 [W, B]
    Tensor counts;    // I32 [B]
    Tensor slots;     // I32 [B]
    // The host's promise on counts (min <= counts[b] <= max <= W), which sizes the launch.
    std::uint32_t min_count = 0, max_count = 0;
};

// One proposal per row: the block [anchor, mask x K] at positions frontier .. frontier + K, against
// the ring's committed context [frontier - window, frontier).
struct DFlash2ProposalCall {
    Tensor anchors;       // I32 [B]: each row's unprocessed anchor token
    Tensor frontiers;     // I32 [B]: the anchor's position (the ring holds every earlier one)
    Tensor valid_columns; // I32 [B]: live block columns 1..W (the tail past them is inert)
    Tensor slots;         // I32 [B]: ring lanes
    const ops::SamplingConfig* configs = nullptr; // device [B]: temperature and seed of each row
    std::int32_t batch = 0, width = 0;             // width W = K + 1, in [2, 16]
    Tensor ids, positions;                         // I32 [W, B] scratch
    Tensor drafts;                                 // I32 [K, B]
    Tensor candidates;                             // I32 [16, K, B]: each draft position's candidates
    Tensor proposal_q;                             // FP32 [16, K, B]: the law each draft was drawn from
    Tensor confidence;                             // FP32 [K, B]: each draft's selector probability (T 1)
};

class DFlash2 {
public:
    // `ring` holds one layer view per drafter layer (capacity = the window). `head` selects the
    // proposal head (--lm-head-draft) or the text head; max_context bounds every frontier.
    DFlash2(const Parameters& parameters, const DFlash2Config& config, ProposalHead head, DeviceContext& device,
            WorkspaceArena& work, std::vector<CyclicKVCacheLayerView> ring, std::uint32_t max_context);

    void materialize(const DFlash2ContextCall& call);
    void propose(const DFlash2ProposalCall& call);

    // Workspace of materialize over up to `context_columns` columns per row (rows of `max_batch`
    // up to the batched width, one row beyond) and of propose at widths [2, max_width] and up to
    // max_batch rows. From the configuration alone (the drafter's projections are Q8_G32_FP16,
    // which Parameters checks; the head's kind is DFlash2Config's).
    [[nodiscard]] static std::size_t workspace_bytes(const TextConfig& text, const DFlash2Config& config,
                                                     std::int32_t max_batch, std::int32_t max_width,
                                                     std::int32_t context_columns, std::uint32_t max_context);

    // The widest context call of several rows (ops::context_kv_materialize's batched profile).
    static constexpr std::int32_t kBatchedContextColumns = 64;

private:
    // The drafts' head top-16: the fused linear_topk of a Q4 proposal head or a Q8 text head, else
    // FP32 logits and their top log-probabilities (a per-column shift of the logits, to which the
    // selector's softmax and argmax are invariant).
    void head_candidates(const Tensor& hidden, Tensor& ids, Tensor& scores);

    const Parameters& parameters_;
    const DFlash2Parameters& p_;
    const DFlash2Config& config_;
    const TextConfig& text_;
    ProposalHead head_;
    DeviceContext& device_;
    WorkspaceArena& work_;
    std::vector<CyclicKVCacheLayerView> ring_;
    std::uint32_t max_context_ = 0;
};

} // namespace infernix::models::qwen4_exp::execution
