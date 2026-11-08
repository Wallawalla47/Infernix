#pragma once

// Test seams of Qwen3.8-Flash-Next's prefix cache:
// - an exact boundary tap at a chosen position, so a test can make a later request resume there
//   (infernix_qwen4_exp_resume_width_real_test: the reproducible route's bits against a cold prefill for resumes off
//   the chunk grid). Without free cache resume the tap moves to the 16-token grid at or before it;
// - a delay before every Host restore's copies, so a consumer that does not wait for a restore reads its destinations
//   before they land every time rather than now and then (infernix_qwen4_exp_preemption_real_test: a resume without
//   replay decodes from a restored state).

#include <cstdint>
#include <optional>

namespace infernix::models::qwen4_exp::testing {

// Every prefill planned after the call (until it is reset with nullopt) taps at `position` when it lies inside the
// planned suffix. For tests only.
void set_forced_tap(std::optional<std::uint32_t> position) noexcept;
[[nodiscard]] std::optional<std::uint32_t> forced_tap() noexcept;

// Every Host restore submitted after the call (until it is reset with 0) holds its copy stream `microseconds` before
// its first copy. For tests only.
void set_restore_delay(std::uint32_t microseconds) noexcept;
[[nodiscard]] std::uint32_t restore_delay() noexcept;

} // namespace infernix::models::qwen4_exp::testing
