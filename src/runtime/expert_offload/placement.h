#pragma once

// How the expert offload stack applies its placement switches (core/temporary_switches.h,
// temporary_switches::offload(): INFERNIX_TMP_LINK_SHARE, _PLAIN_DIVISOR_ALL, _CPU_PLACEMENT,
// _ENGINE_ISOLATE, _PROMPT_SEED, _STAGING_SLOTS) to Qwen3.8-Flash-Next's Program and ExpertOffload,
// which serves Qwen3.5's offloaded Programs. Placement only: every output bit is unchanged.

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
// there (Flash-Next §19.3.12, measured on the GGUF IQ3_S import), or the switch's pair.
inline constexpr std::int32_t kPlainDivisorContext = 1536;
[[nodiscard]] std::int32_t plain_divisor(const OffloadSwitches& switches, std::int32_t position) noexcept;

// Where the CPU expert team and the engine thread run.
struct CpuPlacement {
    std::uint32_t workers = 0;          // the team's worker count
    std::vector<std::uint64_t> cpus;    // per worker its logical CPUs (empty: every worker unpinned)
    std::uint64_t engine  = 0;          // the engine thread's logical CPUs (0: unpinned)
    std::string text;                   // the startup line ("" with the default placement)
};
// The placement of a team of `configured` workers under the switches (today: unpinned, unchanged).
[[nodiscard]] CpuPlacement cpu_placement(const OffloadSwitches& switches, std::uint32_t configured);
// Pins the calling thread (the engine's round thread) when INFERNIX_TMP_ENGINE_ISOLATE is set.
void place_engine_thread() noexcept;

} // namespace infernix::runtime::expert_offload
