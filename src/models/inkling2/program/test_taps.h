#pragma once

// Test seams of Inkling 2's Program (for tests only): a stall written into the MoE workspace's error word, as a down
// CTA's lost count sets it (infernix_inkling2_engine_real_test: the round fails its requests instead of committing their
// tokens, and the next request is served), counts of rows drafted in rounds of several rows and of how they dealt their
// drafts, a skewed deal, and the RAM reading the Host tier's resize follows.

#include <cstdint>
#include <functional>
#include <span>

namespace infernix::models::inkling2::testing {

// The next `rounds` sampled calls of any Program (0: none) find kErrorDownStalled in the MoE workspace's error word, as
// a stalled down CTA leaves it. take_forced_stall reads one and counts it down; it is the Program's.
void force_moe_stall(std::uint32_t rounds) noexcept;
[[nodiscard]] bool take_forced_stall() noexcept;

// Rows that drafted in DSpark rounds of several rows (decode_dspark_rows), every Program of the process together: the
// Engine tests' proof that those rounds engaged.
void count_rows_drafted(std::uint64_t rows) noexcept;
[[nodiscard]] std::uint64_t rows_drafted_together() noexcept;

// How those rounds dealt their drafts (count_dealing: a verified round's drafts by row, of `steps` drafted): the rounds
// whose rows verified different counts, and the rows that verified none and all `steps`. The Engine tests' proof that
// rows of unequal widths, empty and full ones, kept their plain tokens.
void count_dealing(std::span<const std::int32_t> drafts, std::int32_t steps) noexcept;
struct Dealing {
    std::uint64_t unequal_rounds = 0, empty_rows = 0, full_rows = 0;
};
[[nodiscard]] Dealing dealing() noexcept;

// With `on`, every Program's rounds of several rows under --draft-tokens auto deal their n x K drafts from a row that
// rotates round by round, each row all it drafted until none are left (so rows verify none and all their drafts), in
// place of the expected tokens' deal. Off by default.
void set_skewed_dealing(bool on) noexcept;
[[nodiscard]] bool skewed_dealing() noexcept;

// The available RAM a Program's RAM monitor reads (start_ram_resize), for Programs started after the call: `reader`, or
// the system's reading when empty (the default). The prefix cache test drives a release and a growth with it.
void set_ram_available(std::function<std::uint64_t()> reader);
[[nodiscard]] std::function<std::uint64_t()> ram_available();

} // namespace infernix::models::inkling2::testing
