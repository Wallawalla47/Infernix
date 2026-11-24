#pragma once

// --lm-head-draft on Inkling 2 (design docs/maintainer/inkling2_mini-design.md §13.6): the DSpark drafter reads only the
// vocabulary's most frequent tokens. The artifact stores their rank order (`proposal/token_rank`, the measured ranking of
// tools/freq_corpus); the rows of the target head, the Fold and fold_bias are gathered at load and the two matrices made
// 8-bit. Row r of each table is the r-th listed token in ascending id, so ties break as over the whole vocabulary. Only
// the drafts depend on these tables: a draft is kept only when it equals the target's own token.

#include "core/arena.h"
#include "core/weight.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace infernix {
struct DeviceContext;
}

namespace infernix::models::inkling2 {

struct DraftHead {
    DeviceBuffer storage;
    std::int32_t rows = 0;
    Weight head;                               // Q8_G32_FP16 RowSplit [rows, H]
    const std::int8_t* fold_codes    = nullptr; // Q8 [rows][rank]
    const std::uint16_t* fold_scales = nullptr; // FP16 [rows][rank / 32]
    const std::uint16_t* fold_bias   = nullptr; // BF16 [rows]
    const std::int32_t* token_ids    = nullptr; // [rows], ascending
};

// The tokens a draft head of the artifact's ranking holds, ascending. `rank` must be a permutation of the vocabulary.
[[nodiscard]] std::vector<std::int32_t> draft_head_ids(std::span<const std::int32_t> rank);

// The device bytes of a draft head of `rows` tokens.
[[nodiscard]] std::uint64_t draft_head_bytes(std::int32_t rows, std::int32_t hidden, std::int32_t markov_rank);

// Gathers and converts the tables on the device. `head` and `fold` are BF16 Contiguous [V, H] and [V, rank].
[[nodiscard]] DraftHead build_draft_head(const Weight& head, const Weight& fold, const std::uint16_t* fold_bias,
                                         std::span<const std::int32_t> token_ids, DeviceContext& device);

} // namespace infernix::models::inkling2
