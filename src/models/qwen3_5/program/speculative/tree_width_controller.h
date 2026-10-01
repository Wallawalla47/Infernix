#pragma once

#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// Automatic DFlash2 tree verification: chooses each all-neural round's width among the chain and
// the tree widths captured for its batch size.
//
// The gain of a width is measured on the generated text itself. The tree builder adds columns
// best first, so a tree of w columns is the first w columns of any wider tree built for the same
// row, and the acceptance walk over it follows the same path until that path first uses a column
// >= w (the chain is the tree of draft_tokens + 1 columns). Every verified tree row therefore
// gives the tokens each narrower width would have emitted in that round on the same text, and a
// width's gain is its token total over the chain's on the same rows. The cost of a width is its
// measured round time per batch size and context bucket (the smallest of its first three rounds,
// then a bounded running average). Each round takes the width with the most tokens per second;
// widths not yet timed three times, the widest width every kExplorePeriod rounds and timings older
// than kStaleRounds rounds are explored. A choice depends only on earlier rounds, never on the
// round's own draws, so sampling stays exact; it does depend on measured time, so a seeded
// request's sampled text can differ between runs.
class TreeWidthController {
public:
    static constexpr std::size_t kContextBuckets    = 6; // <8K, <16K, <32K, <64K, <128K, longer
    static constexpr std::uint32_t kExplorePeriod   = 32;
    static constexpr std::uint32_t kStaleRounds     = 512;
    static constexpr double kGainDecay              = 1.0 - 1.0 / 512.0; // per observed tree row
    static constexpr std::uint64_t kMinimumGainRows = 32;        // rows before a gain counts
    static constexpr double kTimeWeight             = 1.0 / 8.0; // per observed round

    // chain_columns = draft_tokens + 1; trees[b] lists the tree widths (columns, ascending) that
    // rounds of b + 1 rows may verify.
    TreeWidthController(std::uint32_t chain_columns,
                        const std::array<std::vector<std::uint32_t>, kMaximumConcurrency>& trees);

    // Columns per row for the next all-neural round of batch_size rows whose largest execution
    // frontier is context; chain_columns selects chain verification.
    [[nodiscard]] std::uint32_t choose(std::uint32_t batch_size, std::uint32_t context);

    // One verified tree row of a round of `columns` columns: path holds its accepted columns
    // c_0 = 0, c_1 .. c_A.
    void observe_tree_row(std::span<const std::int32_t> path, std::uint32_t columns);

    // The wall time of a round choose() returned `columns` for, with the same batch size and
    // context.
    void observe_round(std::uint32_t batch_size, std::uint32_t context, std::uint32_t columns,
                       double seconds);

    // Gain of a tree width over the chain on the same text (1 for the chain), or 0 until enough
    // rows have been observed.
    [[nodiscard]] double gain(std::uint32_t columns) const;

    [[nodiscard]] static std::size_t context_bucket(std::uint32_t context) noexcept;

private:
    struct Gain {
        double tokens      = 0; // decayed tokens this width would have emitted
        double chain       = 0; // decayed tokens the chain would have emitted on the same rows
        std::uint64_t rows = 0;
    };

    struct Timing {
        double seconds         = 0;
        std::uint32_t samples  = 0;
        std::uint64_t measured = 0; // the cell round the timing was last refreshed at
    };

    struct Cell {
        std::uint64_t rounds = 0;
        std::vector<Timing> timings; // per candidate of the batch size
    };

    // Candidate columns of rounds of b + 1 rows: the chain, then the tree widths ascending.
    std::array<std::vector<std::uint32_t>, kMaximumConcurrency> candidates_;
    std::array<std::array<Cell, kContextBuckets>, kMaximumConcurrency> cells_;
    std::vector<std::uint32_t> tree_widths_; // every tree width, ascending
    std::vector<Gain> gains_;                // per tree_widths_ entry
    std::uint32_t chain_columns_ = 0;
};

} // namespace ninfer::models::qwen3_5::detail
