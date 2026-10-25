#pragma once

#include "infernix/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace infernix::models::qwen3_5::detail {

// Automatic tree verification: chooses each all-neural round's width among the chain and
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

    // TEMPORARY (INFERNIX_TMP_TREE_ROUND_WIDTH, candidates §13 PUB-03): a row of a round whose width
    // is chosen after its proposal (a tree block round): the drafter's probability of each chain
    // draft (DFlash2's selector probability), and whether a tree may extend the row (its chain is
    // the whole window).
    struct RoundRow {
        std::span<const float> probabilities;
        bool tree = false;
    };
    // The row's own estimates: expected chain tokens 1 + sum_j prod_{i<=j} p_i, and miss mass, the
    // chance the chain stops before its end, 1 - prod_j p_j, which a tree's side columns may recover.
    struct RoundEstimate {
        double chain = 1.0;
        double miss  = 0.0;
    };
    [[nodiscard]] static RoundEstimate estimate(const RoundRow& row) noexcept;
    // Whether rounds of batch_size rows choose among tree widths at all.
    [[nodiscard]] bool automatic(std::uint32_t batch_size) const noexcept;
    // choose() for a tree block round. Exploration is choose()'s; otherwise, once calibrated, a width
    // w's tokens are each row's beta x chain + (tree rows) alpha_w x miss, beta the measured chain
    // tokens over the estimated ones and alpha_w the measured tokens w adds over the chain per unit of
    // miss (each a running sum decaying by kGainDecay per row, counted after kMinimumGainRows rows),
    // and the round takes the width with the most tokens per second; uncalibrated, choose()'s rule.
    [[nodiscard]] std::uint32_t choose(std::uint32_t batch_size, std::uint32_t context,
                                       std::span<const RoundRow> rows);
    // One row of a tree block round verified at `columns`: its accepted columns c_0 = 0 .. c_A (a
    // chain row's are 0 .. A) calibrate beta and, for a tree row, alpha of every tree width up to
    // `columns`.
    void observe_round_row(const RoundRow& row, std::span<const std::int32_t> path, std::uint32_t columns);

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

    // A running ratio of decayed sums (TEMPORARY, INFERNIX_TMP_TREE_ROUND_WIDTH).
    struct Calibration {
        double measured    = 0;
        double estimated   = 0;
        std::uint64_t rows = 0;
        [[nodiscard]] bool ready() const noexcept { return rows >= kMinimumGainRows && estimated > 0; }
        [[nodiscard]] double ratio() const noexcept { return measured / estimated; }
        void add(double m, double e) noexcept {
            measured  = measured * kGainDecay + m;
            estimated = estimated * kGainDecay + e;
            rows += 1;
        }
    };

    // The exploration choose() makes before its rate rule, or 0 when it makes none.
    [[nodiscard]] std::uint32_t explore(std::uint32_t batch_size, std::uint32_t context);

    // Candidate columns of rounds of b + 1 rows: the chain, then the tree widths ascending.
    std::array<std::vector<std::uint32_t>, kMaximumConcurrency> candidates_;
    std::array<std::array<Cell, kContextBuckets>, kMaximumConcurrency> cells_;
    std::vector<std::uint32_t> tree_widths_; // every tree width, ascending
    std::vector<Gain> gains_;                // per tree_widths_ entry
    std::uint32_t chain_columns_ = 0;
    Calibration chain_tokens_;                   // beta
    std::vector<Calibration> miss_recovery_;     // alpha, per tree_widths_ entry
};

} // namespace infernix::models::qwen3_5::detail
