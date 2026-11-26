#pragma once

// Test seams of Inkling 2's Program (for tests only): a stall written into the MoE workspace's error word, as a down
// CTA's lost count sets it (infernix_inkling2_engine_real_test: the round fails its requests instead of committing their
// tokens, and the next request is served), a count of rows drafted in rounds of several rows, and the RAM reading the
// Host tier's resize follows.

#include <cstdint>
#include <functional>

namespace infernix::models::inkling2::testing {

// The next `rounds` sampled calls of any Program (0: none) find kErrorDownStalled in the MoE workspace's error word, as
// a stalled down CTA leaves it. take_forced_stall reads one and counts it down; it is the Program's.
void force_moe_stall(std::uint32_t rounds) noexcept;
[[nodiscard]] bool take_forced_stall() noexcept;

// Rows that drafted in DSpark rounds of several rows (decode_dspark_rows), every Program of the process together: the
// Engine tests' proof that those rounds engaged.
void count_rows_drafted(std::uint64_t rows) noexcept;
[[nodiscard]] std::uint64_t rows_drafted_together() noexcept;

// The available RAM a Program's RAM monitor reads (start_ram_resize), for Programs started after the call: `reader`, or
// the system's reading when empty (the default). The prefix cache test drives a release and a growth with it.
void set_ram_available(std::function<std::uint64_t()> reader);
[[nodiscard]] std::function<std::uint64_t()> ram_available();

} // namespace infernix::models::inkling2::testing
