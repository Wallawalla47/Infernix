#pragma once

// A test seam of Qwen3.8-Flash-Next's prefill planner: an exact boundary tap at a chosen position, so a test can make a
// later request resume there (infernix_qwen4_exp_resume_width_real_test: the reproducible route's bits against a cold
// prefill for resumes off the chunk grid). Without free cache resume the tap moves to the 16-token grid at or before it.

#include <cstdint>
#include <optional>

namespace infernix::models::qwen4_exp::testing {

// Every prefill planned after the call (until it is reset with nullopt) taps at `position` when it lies inside the
// planned suffix. For tests only.
void set_forced_tap(std::optional<std::uint32_t> position) noexcept;
[[nodiscard]] std::optional<std::uint32_t> forced_tap() noexcept;

} // namespace infernix::models::qwen4_exp::testing
