#pragma once

// How the expert offload stack applies its placement switches (core/temporary_switches.h,
// temporary_switches::offload(): INFERNIX_TMP_LINK_SHARE, _CPU_PLACEMENT, _STAGING_SLOTS), the plain
// decode divisor and the engine thread's isolation to Qwen3.8-Flash-Next's Program and
// ExpertOffload, which serves Qwen3.5's offloaded Programs. Placement only: every output bit is unchanged.

#include "core/temporary_switches.h"

#include <cstdint>
#include <string>
#include <vector>

namespace infernix::runtime::expert_offload {

using temporary_switches::OffloadSwitches;

// The link's share of a decode call's host-memory misses in 256ths (MoeCpuChannel::pcie_share; 0:
// the divisor rule), from the measured link (bytes/s), the CPU team's experts/s and the record bytes.
[[nodiscard]] std::int32_t link_share_256(const OffloadSwitches& switches, double link_bytes_per_second,
                                          double cpu_expert_rate, double record_bytes) noexcept;

// Plain decode's one-column divisor at `position` context tokens: 1 below kPlainDivisorContext, 2 from
// there (Flash-Next §19.3.12, measured on the GGUF IQ3_S import; NVFP4 banks keep their rate-based split,
// which measured 7-27 % faster than this divisor, 2026-10-28).
inline constexpr std::int32_t kPlainDivisorContext = 1536;
[[nodiscard]] constexpr std::int32_t plain_divisor(std::int32_t position) noexcept {
    return position >= kPlainDivisorContext ? 2 : 1;
}

// Where the CPU expert team and the engine thread run.
struct CpuPlacement {
    std::uint32_t workers = 0;          // the team's worker count
    std::vector<std::uint64_t> cpus;    // per worker its logical CPUs (empty: every worker unpinned)
    std::uint64_t engine  = 0;          // the engine thread's logical CPUs (0: unpinned)
    std::string text;                   // the startup line ("" with the default placement)
};
// The placement of a team of `configured` workers under the switches (by default unpinned). With
// `engine_isolate` (the Qwen3.5-122B's offloaded route: OffloadOptions::engine_isolate) the engine
// thread takes the last core of the fastest class (both its logical CPUs), which no worker may use.
[[nodiscard]] CpuPlacement cpu_placement(const OffloadSwitches& switches, std::uint32_t configured,
                                         bool engine_isolate = false);
// Pins the calling thread (the engine's round thread) to `cpus` (CpuPlacement::engine; 0: unpinned).
void place_engine_thread(std::uint64_t cpus) noexcept;

} // namespace infernix::runtime::expert_offload
