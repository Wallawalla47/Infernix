#pragma once

// Inkling 2's DSpark drafter (design docs/maintainer/inkling2_mini-design.md §13; the reference arithmetic in .local's
// DSPARK_SPEC): its device state and a round's two parts.
//
// - append: the context rows of a forward call's columns (the taps it left in buffers().features): each layer's mixed,
//   normed context (dspark_mix) through its context K and V projections, the K head-normed and rotated at the
//   column's position, both into the layer's cache of the column's lane; the Markov context (mix row 3) into the lane's
//   ring by position. Position p's rows are its committed target row's once p is committed; later rows overwrite a
//   rejected verify column's before anything reads them.
// - draft: the block of 8 rows at positions F .. F + 7 (the anchor's normed target embedding, then the mask
//   embedding) through the 3 layers and the final norm, the target head over rows 1 .. steps (FP32, no softcap), and
//   the Markov head's steps: `steps` greedy drafts and their probabilities, in device memory. The round's words
//   (DSparkRound: F, the anchor, the lane) are copied from pinned memory first, so a captured draft replays per round.
//   A round of several rows (lanes decoding together) stages each row's words and drafts every row's block in one
//   chain: the projections, norms and SwiGLU over all the blocks' rows at once (each matrix read once a round), the
//   rotations, attention and Markov steps a launch over the blocks, each block its own round, lane and slots; the
//   target head once over the blocks' final rows (their drafts and probabilities into their own slots).
//
// The draft's projections run projections_dmma at every width (launches of 16 columns); the append's run it at most
// 16 columns and projections_mma wider (a prefill chunk's context rows). The drafter decides only proposals;
// verification decides every output.

#include "core/arena.h"
#include "infernix/ops/inkling2_dspark.h"
#include "infernix/ops/inkling2_projections.h"
#include "models/inkling2/execution/forward.h"
#include "models/inkling2/execution/parameters.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace infernix::models::inkling2 {

// The full layer's context K and V: two planes of the target's KV pages (page-major [pages][8][64][128] BF16), or with
// every other pool format four INT8 planes (ops::inkling2::DSparkInt8Cache: codes and side records; k and v null), and
// the lanes' block-table rows, which the Program owns (its pool's planes; a page holds the drafter's context with the
// target's, so the prefix cache shares and restores both).
struct DSparkPagedContext {
    std::uint16_t* k            = nullptr;
    std::uint16_t* v            = nullptr;
    ops::inkling2::DSparkInt8Cache int8;
    const std::int32_t* table   = nullptr;
    std::int32_t table_stride   = 0;
};

// One lane's part of the drafter's state a snapshot carries (the sliding layers' rings): `words` 16-bit words at
// `base + lane * lane_words`.
struct DSparkStatePart {
    std::uint16_t* base      = nullptr;
    std::int64_t lane_words  = 0;
};

class Drafter {
public:
    // `capacity`: the widest forward call whose context rows are appended at once (the forward's capacity).
    Drafter(const Parameters& parameters, const Forward& forward, std::int32_t lanes, std::uint32_t max_context,
            std::int32_t capacity, const DSparkPagedContext& full);
    ~Drafter() = default;
    Drafter(const Drafter&)            = delete;
    Drafter& operator=(const Drafter&) = delete;

    // The device memory a drafter of these lanes, context and capacity allocates (the full layer's context is the
    // Program's pages).
    [[nodiscard]] static std::size_t device_bytes(std::int32_t lanes, std::uint32_t max_context, std::int32_t capacity,
                                                  std::uint32_t vocab);

    // The lane state a prefix-cache snapshot copies with the target's image: the sliding layers' K and V rings (the
    // Markov ring needs none: a resumed request prefills at least one column, whose row the first draft reads).
    [[nodiscard]] std::array<DSparkStatePart, 4> snapshot_parts() const noexcept {
        return {DSparkStatePart{cache_k_[0], lane_words_[0]}, DSparkStatePart{cache_v_[0], lane_words_[0]},
                DSparkStatePart{cache_k_[2], lane_words_[2]}, DSparkStatePart{cache_v_[2], lane_words_[2]}};
    }

    // The context rows of columns [first, first + count) of the forward's last call (its positions and lanes).
    void append(std::int32_t first, std::int32_t count, cudaStream_t stream);

    // The round's words, staged in pinned memory before `draft` (or its graph) runs: one row, or `rounds.size()` rows
    // (at most the drafter's lanes, each a different lane).
    void stage(const ops::inkling2::DSparkRound& round) noexcept { host_round_[0] = round; }
    void stage(std::span<const ops::inkling2::DSparkRound> rounds);
    // A round's drafts for the staged rows 0 .. rows: row r's tokens and probabilities [0 .. steps) in its slot.
    void draft(std::int32_t steps, cudaStream_t stream, std::int32_t rows = 1);
    // The drafts copied to pinned memory (after draft, on the same stream): host_tokens(r) / host_probabilities(r).
    void download(std::int32_t steps, cudaStream_t stream, std::int32_t rows = 1);
    [[nodiscard]] const std::int32_t* host_tokens(std::int32_t row = 0) const noexcept {
        return host_tokens_ + static_cast<std::size_t>(row) * ops::inkling2::kDSparkBlock;
    }
    [[nodiscard]] const float* host_probabilities(std::int32_t row = 0) const noexcept {
        return host_probabilities_ + static_cast<std::size_t>(row) * ops::inkling2::kDSparkBlock;
    }

private:
    void project(ops::inkling2::ProjectionArgs& args, cudaStream_t stream) const;
    // The blocks of rows 0 .. blocks - 1 (consecutive rows' workspace slices: one [blocks x 8] buffer each).
    void draft_blocks(std::int32_t blocks, std::int32_t steps, cudaStream_t stream);

    const Parameters& p_;
    const DSparkOperands& d_;
    const Forward& forward_;
    std::int32_t lanes_ = 0, capacity_ = 0, full_splits_ = 0;
    std::uint32_t vocab_ = 0;
    DeviceBuffer memory_;
    PinnedHostBuffer pinned_;
    // Caches: the sliding layers' K and V, [lanes][8][ring][128] BF16, and the full layer's paged context (full_); the
    // Markov context ring [lanes][16][H].
    std::uint16_t* cache_k_[3] = {};
    std::uint16_t* cache_v_[3] = {};
    std::int64_t lane_words_[3] = {};
    DSparkPagedContext full_;
    std::uint16_t* f_ring_ = nullptr;
    // Append workspace: the mixed contexts [3][capacity][H], their K and V [3][capacity][1024].
    std::uint16_t* mixed_[3] = {};
    std::uint16_t* ctx_k_[3] = {};
    std::uint16_t* ctx_v_[3] = {};
    // A row's draft workspace (a lane's each; a round of several rows uses consecutive rows' slices as one buffer).
    struct Workspace {
        std::uint16_t *block_in = nullptr, *residual = nullptr, *normed = nullptr; // [8][H]
        std::uint16_t *q = nullptr, *k = nullptr, *v = nullptr, *qr = nullptr;     // [8][3072], [8][1024] x 2, [8][3072]
        std::uint16_t *attention = nullptr, *a = nullptr, *y = nullptr;            // [8][3072], [8][H], [8][H]
        std::uint16_t *gate = nullptr, *up = nullptr, *h = nullptr;                // [8][9216]
        std::uint16_t* final_rows = nullptr;                                       // [8][H]
        float* base = nullptr;                                                     // [8][V]
        float* attention_scratch = nullptr;
        float* markov_scratch = nullptr;
        std::int32_t* head_x8 = nullptr; // [8] 8-bit head input rows (Forward::project_head)
    };
    std::vector<Workspace> rows_;
    // The round words, drafts and probabilities a row each [lanes].
    ops::inkling2::DSparkRound* round_ = nullptr;
    std::int32_t* tokens_ = nullptr;
    float* probabilities_ = nullptr;
    float* inv_freq_ = nullptr; // the RoPE table [64]
    ops::inkling2::DSparkRound* host_round_ = nullptr;
    std::int32_t* host_tokens_ = nullptr;
    float* host_probabilities_ = nullptr;
};

} // namespace infernix::models::inkling2
