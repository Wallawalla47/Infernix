#pragma once

// Inkling 2's DFlash2 drafter (--spec dflash2 --draft-model; docs/maintainer/dflash.md, design §13.8): a companion
// artifact's DFlash2 backbone (five sliding-window layers with dynamic convolutions) conditioned on a per-lane context
// ring of K/V projected from the target's DSpark features (the inputs of visits 5, 23 and 41 and the final-normed
// state), drafting from the target's head (or the --lm-head-draft shortlist) with the conditional candidate selector.
// The fused DFlash2 Ops (the Flash-Next drafter's, registered at Inkling 2 Mini's widths) run every step; the block's
// embeddings are the target's normed embeddings (ops::inkling2::embed), as the reference's draft input.
//
// - Context: each committed target column's features through the feature projection and norm into every layer's ring
//   (BF16 K, FP16 V, 2,048 slots: the window). The ring holds committed positions only: a slot a rejected column
//   overwrote would be one the next block still reads (the ring has no slack past the window), so a verify call's
//   columns are written at the round's commit, its committed count each; a prefill call's and a plain round's at once.
// - Draft: one block a row, [anchor, mask x 7] at F .. F + 7, through the layers against the ring's [F - 2047, F), the
//   head's top 16 at each mask column and the selector's path: 7 drafts a row and their temperature-1 probabilities
//   (the selector's confidence), the rows' blocks in one call. A row's path is greedy, or (a staged whole-vocabulary
//   draw) coupled with the target's own draws: each draft the candidate the target's Gumbel noise at its verify column
//   ranks first. The drafter decides only proposals: verification keeps a draft only when it equals the target's
//   own draw.

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"
#include "infernix/ops/inkling2_dspark.h"
#include "infernix/ops/sampling.h"
#include "models/inkling2/execution/dspark.h"
#include "models/inkling2/execution/forward.h"
#include "models/inkling2/execution/parameters.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace infernix::models::inkling2 {

class DFlash2Drafter {
public:
    // The block: the anchor and kDrafts masks (the drafter's trained block of 8).
    static constexpr std::int32_t kWidth  = 8;
    static constexpr std::int32_t kDrafts = kWidth - 1;

    DFlash2Drafter(const Parameters& parameters, const Forward& forward, std::int32_t lanes, std::uint32_t max_context);
    ~DFlash2Drafter();
    DFlash2Drafter(const DFlash2Drafter&)            = delete;
    DFlash2Drafter& operator=(const DFlash2Drafter&) = delete;

    // The device memory a drafter of these lanes and context allocates: the rings, the io words and the workspace.
    [[nodiscard]] static std::size_t device_bytes(const Config& config, std::int32_t lanes, std::uint32_t max_context,
                                                  bool shortlist);

    // The lane state a prefix-cache snapshot copies with the target's image: every layer's K and V ring.
    [[nodiscard]] std::vector<DSparkStatePart> snapshot_parts() const;

    // ---- context (committed columns of the forward's last call) ----
    // One lane's sequential columns [first, first + count): their last window (earlier ones would be overwritten
    // before any read). Captured or eager.
    void context(std::int32_t first, std::int32_t count, cudaStream_t stream);
    // The same over explicit device columns: `count` sequential columns of features [count][features * H] (BF16), their
    // positions [count] and lanes [count] (one lane).
    void context(const std::uint16_t* features, const std::int32_t* positions, const std::int32_t* lanes,
                 std::int32_t count, cudaStream_t stream);
    // A plain round of `rows` lanes, one column each (column r is lane fb.lane[r]'s at fb.position[r]). Captured or eager.
    void context_columns(std::int32_t rows, cudaStream_t stream);
    // A verify call's committed columns: row r's counts[r] columns from column firsts[r] (sequential positions, at
    // most kWidth) into lane lanes[r] (0: none). The words are staged in pinned memory and copied in first; the columns
    // are gathered into a block of kWidth columns a row (the rows of one call differ in width: their drafts are dealt).
    void context_rows(std::span<const std::int32_t> counts, std::span<const std::int32_t> firsts,
                      std::span<const std::int32_t> lanes, cudaStream_t stream);
    // context_rows in its two halves, the second capturable per row count (a commit graph): the rows' words into
    // pinned memory (validated), then their copy in, the gather and the materializer of `rows` rows.
    void stage_commit(std::span<const std::int32_t> counts, std::span<const std::int32_t> firsts,
                      std::span<const std::int32_t> lanes);
    void commit_rows(std::int32_t rows, cudaStream_t stream);

    // ---- drafts ----
    // The rounds' words and each row's selector draw: `draws[r]` (empty: every row greedy) is a whole-vocabulary row's
    // temperature and seed with full_vocabulary set, its drafts coupled with the target's own draws by their Gumbel
    // noise (candidate_selector_path's couple_draws); any other row drafts greedily.
    void stage(const ops::inkling2::DSparkRound& round, const ops::SamplingConfig* draw = nullptr) noexcept {
        stage({&round, 1}, draw != nullptr ? std::span<const ops::SamplingConfig>(draw, 1)
                                           : std::span<const ops::SamplingConfig>());
    }
    void stage(std::span<const ops::inkling2::DSparkRound> rounds,
               std::span<const ops::SamplingConfig> draws = {}) noexcept;
    // The staged rows' blocks (their words copied in from pinned memory first): kDrafts drafts a row and their
    // probabilities in device memory. Captured per row count.
    void draft(cudaStream_t stream, std::int32_t rows = 1);
    // The drafts copied to pinned memory (after draft, on the same stream): host_tokens(r) / host_probabilities(r).
    void download(cudaStream_t stream, std::int32_t rows = 1);
    [[nodiscard]] const std::int32_t* host_tokens(std::int32_t row = 0) const noexcept {
        return host_drafts_ + static_cast<std::size_t>(row) * kDrafts;
    }
    [[nodiscard]] const float* host_probabilities(std::int32_t row = 0) const noexcept {
        return host_confidence_ + static_cast<std::size_t>(row) * kDrafts;
    }

private:
    // The materializer over [features * H, width, rows] features (row r's columns at r * width), positions [width,
    // rows], counts and slots [rows] (device I32).
    void materialize(const std::uint16_t* features, std::int32_t width, std::int32_t rows, const std::int32_t* positions,
                     const std::int32_t* counts, const std::int32_t* slots, std::uint32_t min_count,
                     std::uint32_t max_count, cudaStream_t stream);
    void head_candidates(const Tensor& hidden, Tensor& ids, Tensor& scores, cudaStream_t stream);

    const Parameters& p_;
    const DFlash2Operands& d_;
    const DFlash2Config& config_;
    const Forward& forward_;
    std::int32_t lanes_ = 0;
    std::uint32_t max_context_ = 0;
    std::int32_t head_rows_    = 0; // the shortlist's or the vocabulary's
    DeviceBuffer memory_;
    PinnedHostBuffer pinned_;
    std::unique_ptr<CyclicKVCache> ring_;
    std::vector<CyclicKVCacheLayerView> layers_;
    std::unique_ptr<WorkspaceArena> work_;
    // Device io words (one region): the round's (uploaded at once): anchors, frontiers, valid columns and slots [lanes]
    // each, the selector configs [lanes]; the commit's counts, firsts and slots [lanes] each; constant counts (ones
    // [lanes], the iota [window + 1]); ids and positions [kWidth][lanes], drafts [kDrafts][lanes], candidates and q
    // [16][kDrafts][lanes], confidence [kDrafts][lanes]. The commit's gathered context: features [F, kWidth, lanes]
    // and positions [kWidth, lanes].
    std::int32_t *anchors_ = nullptr, *frontiers_ = nullptr, *valid_ = nullptr, *slots_ = nullptr;
    std::int32_t *ids_ = nullptr, *positions_ = nullptr, *drafts_ = nullptr, *candidates_ = nullptr;
    float *q_ = nullptr, *confidence_ = nullptr;
    std::int32_t *commit_ = nullptr, *ones_ = nullptr, *iota_ = nullptr;
    std::uint16_t* gathered_          = nullptr;
    std::int32_t* gathered_positions_ = nullptr;
    ops::SamplingConfig* configs_ = nullptr;
    std::int32_t* head_x8_ = nullptr; // [kDrafts * lanes] 8-bit head input rows (Forward::project_head)
    // Pinned: the round words (anchors .. configs, in the device order), the commit words, the drafts and confidence.
    std::int32_t* host_words_      = nullptr;
    ops::SamplingConfig* host_configs_ = nullptr;
    std::int32_t* host_drafts_     = nullptr;
    float* host_confidence_        = nullptr;
    std::int32_t* host_commit_     = nullptr;
};

} // namespace infernix::models::inkling2
