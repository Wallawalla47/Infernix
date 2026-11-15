#pragma once

// Test seams of Qwen3.5's prefix cache binding (defined in hybrid_cache.cpp):
// - a delay before every Host restore's copies, so a consumer that does not wait for a restore reads its destinations
//   before they land every time rather than now and then (infernix_qwen3_5_engine_preemption_real_test: a resume that
//   replays from a tap over restored blocks);
// - counts of the synchronous waits and the Vision encodes a test bounds: the waits for a transfer that room-making
//   took (a reclaim that cannot evict takes none), and the overlay Vision encodes bindings ran (a binding that
//   reports a shortage runs none).

// - the rotary offset each prefill-shaped text pass ran with (a prompt's, a Replay's, forced tokens'): a pass at a
//   base past 0 must carry its sequence's rope_delta (2026-11-15: all three ran at delta 0 after a member was
//   inserted into the pass's context, so an image request's text passes computed at the wrong positions).

#include <cstdint>
#include <vector>

namespace infernix::models::qwen3_5::testing {

// One text pass (execution::prefill_text_chunk): where it began and the rotary offset it was given.
struct TextPass {
    std::uint32_t base      = 0;
    std::int32_t rope_delta = 0;
};

// Starts (clearing what was recorded) or stops the recording of text passes. For tests only; off, a pass records
// nothing.
void record_text_passes(bool enabled);
[[nodiscard]] std::vector<TextPass> text_passes();
void note_text_pass(std::uint32_t base, std::int32_t rope_delta);

// Every Host restore opened after the call (until it is reset with 0) holds its copy stream `microseconds` before
// its first copy. For tests only.
void set_restore_delay(std::uint32_t microseconds) noexcept;
[[nodiscard]] std::uint32_t restore_delay() noexcept;

// Transfers room-making waited for (HybridPrefixCache::settle_oldest_transfer), since the process started.
[[nodiscard]] std::uint64_t transfer_waits() noexcept;
void count_transfer_wait() noexcept;

// Overlay Vision encodes run for a binding or a resume (ProgramImpl::preencode_overlay_vision), since the process
// started.
[[nodiscard]] std::uint64_t binding_vision_encodes() noexcept;
void count_binding_vision_encode() noexcept;

} // namespace infernix::models::qwen3_5::testing
