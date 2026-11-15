#pragma once

// Test seams of Qwen3.5's prefix cache binding (defined in hybrid_cache.cpp):
// - a delay before every Host restore's copies, so a consumer that does not wait for a restore reads its destinations
//   before they land every time rather than now and then (infernix_qwen3_5_engine_preemption_real_test: a resume that
//   replays from a tap over restored blocks);
// - counts of the synchronous waits and the Vision encodes a test bounds: the waits for a transfer that room-making
//   took (a reclaim that cannot evict takes none), and the overlay Vision encodes bindings ran (a binding that
//   reports a shortage runs none).

#include <cstdint>

namespace infernix::models::qwen3_5::testing {

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
