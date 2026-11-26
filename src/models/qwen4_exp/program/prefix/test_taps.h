#pragma once

// Test seams of Qwen3.8-Flash-Next's prefix cache:
// - an exact boundary tap at a chosen position, so a test can make a later request resume there
//   (infernix_qwen4_exp_resume_width_real_test: the reproducible route's bits against a cold prefill for resumes off
//   the chunk grid). Without free cache resume the tap moves to the 16-token grid at or before it;
// - a delay before every Host restore's copies, so a consumer that does not wait for a restore reads its destinations
//   before they land every time rather than now and then (infernix_qwen4_exp_preemption_real_test: a resume without
//   replay decodes from a restored state).

#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

namespace infernix::models::qwen4_exp::testing {

// Every prefill planned after the call (until it is reset with nullopt) taps at `position` when it lies inside the
// planned suffix. For tests only.
void set_forced_tap(std::optional<std::uint32_t> position) noexcept;
[[nodiscard]] std::optional<std::uint32_t> forced_tap() noexcept;

// Every Host restore submitted after the call (until it is reset with 0) holds its copy stream `microseconds` before
// its first copy. For tests only.
void set_restore_delay(std::uint32_t microseconds) noexcept;
[[nodiscard]] std::uint32_t restore_delay() noexcept;

// The next activation that stages a Host restore fails once its batch is open and staged, as a device error in a
// restore copy would (infernix_qwen4_exp_prefix_cache_real_test: the batch is aborted, and later restores and the
// shutdown save work). take_restore_fault reads and clears the request. For tests only.
void set_restore_fault() noexcept;
[[nodiscard]] bool take_restore_fault() noexcept;

// Whole drains of the prefix cache's transfer and restore streams (PrefixCache::drain) since the process started: a
// test reads the count around an operation that must not wait for every transfer. count_drain is the cache's.
// Blocks whose Host write started when a tap was taken (continuous write-through), not at their lane's release,
// since the process started. count_tap_block_writes is the Program's.
void count_tap_block_writes(std::uint64_t blocks) noexcept;
[[nodiscard]] std::uint64_t tap_block_writes() noexcept;

void count_drain() noexcept;
[[nodiscard]] std::uint64_t drains() noexcept;

// The capture results a Program's prefix cache kept after its latest lane release (prefix_release): what the slots
// keep for their next admission, however many requests came before (the prefix cache test's growth case).
void record_known_captures(std::uint64_t count) noexcept;
[[nodiscard]] std::uint64_t known_captures() noexcept;

// For `hold` after the call (zero: none), a binding's context transaction whose own work is done is reported
// unfinished: the binding spans more Engine cycles, as a Vision encode lent its window does (a layer group a cycle).
// And the finishes completed beside an open transaction (finish_held) since the process started; count_held_finish
// is the Program's (infernix_qwen4_exp_finish_beside_binding_real_test). For tests only.
void set_binding_hold(std::chrono::milliseconds hold) noexcept;
[[nodiscard]] bool binding_held() noexcept;
void count_held_finish() noexcept;
[[nodiscard]] std::uint64_t held_finishes() noexcept;

// The wide workspace scopes (ProgramImpl::WideWork, a prefill call wider than the static arena): where each was
// entered, the columns it was entered with, the widest call it ran (a layer walk's MoE runs over a group of chunks),
// the bytes the arena was given, the bytes the forward's own allocation plan gives it (0 while none is computed) and
// the arena's peak. Recorded while record_wide_scopes(true) (which clears what was recorded); note_wide_scope is the
// Program's. For tests only.
enum class WideSite : std::uint8_t {
    Call,     // one forward call (ProgramImpl::run)
    WalkSpan, // a layer walk's span setup: each chunk's embedding and first layer
    WalkStep, // a layer walk's step over its next layers
    PairCall, // a prompt's narrow head and opener tail as one call (run_split)
};
struct WideScope {
    WideSite site          = WideSite::Call;
    std::int32_t columns   = 0;
    std::int32_t inner     = 0;
    std::uint64_t given    = 0;
    std::uint64_t planned  = 0;
    std::uint64_t peak     = 0;
};
void record_wide_scopes(bool enabled);
[[nodiscard]] std::vector<WideScope> wide_scopes();
void note_wide_scope(const WideScope& scope);

// The staging slots of the last Program started, and whether its experts are all in RAM or VRAM (no record read from
// the artifact on use: every expert in RAM, or the exclusive layout's tier). A start short of RAM reads the rest from
// the artifact and keeps 64 slots by design (infernix_qwen4_exp_fork_route_real_test skips such a start). note_staging
// is the Program's (a test may clear it with 0 slots). For tests only.
void note_staging(std::int32_t slots, bool full_tier) noexcept;
[[nodiscard]] std::int32_t staging_slots() noexcept;
[[nodiscard]] bool staging_full_tier() noexcept;

} // namespace infernix::models::qwen4_exp::testing
