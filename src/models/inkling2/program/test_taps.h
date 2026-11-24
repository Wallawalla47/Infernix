#pragma once

// Test seams of Inkling 2's Program (for tests only): a stall written into the MoE workspace's error word, as a down
// CTA's lost count sets it (infernix_inkling2_engine_real_test: the round fails its requests instead of committing their
// tokens, and the next request is served).

#include <cstdint>

namespace infernix::models::inkling2::testing {

// The next `rounds` sampled calls of any Program (0: none) find kErrorDownStalled in the MoE workspace's error word, as
// a stalled down CTA leaves it. take_forced_stall reads one and counts it down; it is the Program's.
void force_moe_stall(std::uint32_t rounds) noexcept;
[[nodiscard]] bool take_forced_stall() noexcept;

} // namespace infernix::models::inkling2::testing
