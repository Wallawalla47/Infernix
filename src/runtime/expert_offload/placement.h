#pragma once

// How the expert offload stack applies its placement switches (core/temporary_switches.h,
// temporary_switches::offload(): INFERNIX_TMP_LINK_SHARE, _CPU_PLACEMENT, _STAGING_SLOTS), the plain
// decode divisor and the engine thread's isolation to Qwen3.8-Flash-Next's Program and
// ExpertOffload, which serves Qwen3.5's offloaded Programs. Placement only: every output bit is unchanged.

#include "core/temporary_switches.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace infernix::runtime::expert_offload {

using temporary_switches::OffloadSwitches;

// The link's share of a decode call's host-memory misses in 256ths (MoeCpuChannel::pcie_share; 0:
// the divisor rule), from the measured link (bytes/s), the CPU team's experts/s and the record bytes.
[[nodiscard]] std::int32_t link_share_256(const OffloadSwitches& switches, double link_bytes_per_second,
                                          double cpu_expert_rate, double record_bytes) noexcept;

// ExpertOffload's decode split without INFERNIX_TMP_LINK_SHARE (the Qwen3.5-122B-A10B's offloaded experts): the
// link's share of a decode call's host-memory misses in 256ths, 256 / (1 + kDecodeLinkWeight C t), with C the CPU
// team's experts/s and t a record's seconds on the link (the two legs' balance); the stage keeps (M share) >> 8 of
// a call's M misses. Fitted on Ornith-1.5-122B (RTX 5090 at PCIe 5.0 x8, i9-13900K, 2026-10-30, design §11.4):
// real code continued greedily with MTP auto decoded fastest at 64/256 (C t 2.06) after 1,536, 8,192 and 32,768
// context tokens, 8.0-12.7 % faster than the divisor rule's 1/3 (M / 3), level with 43/256, 7-8 % slower at
// 107/256. Flash-Next's Program keeps its own divisor rule.
inline constexpr double kDecodeLinkWeight = 1.46;
[[nodiscard]] inline std::int32_t decode_link_share_256(double link_bytes_per_second, double cpu_expert_rate,
                                                        double record_bytes) noexcept {
    if (!(link_bytes_per_second > 0.0) || !(cpu_expert_rate > 0.0) || !(record_bytes > 0.0)) { return 0; }
    const double ct = cpu_expert_rate * record_bytes / link_bytes_per_second;
    return static_cast<std::int32_t>(std::clamp<long>(std::lround(256.0 / (1.0 + kDecodeLinkWeight * ct)), 1, 256));
}

// Plain decode's one-column divisor at `position` context tokens: 1 below kPlainDivisorContext, 2 from
// there (Flash-Next §19.3.12, measured on the GGUF IQ3_S import; NVFP4 banks keep their rate-based split,
// which measured 7-27 % faster than this divisor, 2026-10-28).
inline constexpr std::int32_t kPlainDivisorContext = 1536;
[[nodiscard]] constexpr std::int32_t plain_divisor(std::int32_t position) noexcept {
    return position >= kPlainDivisorContext ? 2 : 1;
}

// The logical CPUs a default team leaves to the engine's spinning round thread, the CUDA runtime and
// the OS. With the process held to 16 logical CPUs (one per core) of the i9-13900K, the GSQ-RCO IQ3_S
// with DFlash2 decoded at 61-87 tok/s with 16 GGUF workers, 189-217 with 8, and 268 with 16 on all 32
// CPUs (2026-10-30); a 16-vCPU VM decoded at 8.6-10.6 with 16 and 184-240 with 4-12 (GitHub issue #48).
inline constexpr std::uint32_t kReservedCpus = 4;

// The default team size: `configured` workers, at most `logical_cpus` - kReservedCpus and at least one
// (`logical_cpus` 0: unknown, `configured`; `configured` 0 keeps the team off).
[[nodiscard]] constexpr std::uint32_t default_workers(std::uint32_t configured, std::uint32_t logical_cpus) noexcept {
    if (configured == 0 || logical_cpus == 0) { return configured; }
    const std::uint32_t room = logical_cpus > kReservedCpus ? logical_cpus - kReservedCpus : 1U;
    return configured < room ? configured : room;
}

// Where the CPU expert team and the engine thread run.
struct CpuPlacement {
    std::uint32_t workers = 0;          // the team's worker count
    std::vector<std::uint64_t> cpus;    // per worker its logical CPUs (empty: every worker unpinned)
    std::uint64_t engine  = 0;          // the engine thread's logical CPUs (0: unpinned)
    std::string text;                   // the startup line ("" with the default placement)
};
// The placement of a team of `configured` workers under the switches (by default unpinned; without
// INFERNIX_TMP_CPU_PLACEMENT's count, default_workers of the process's logical CPUs). With
// `engine_isolate` (the Qwen3.5-122B's offloaded route: OffloadOptions::engine_isolate) the engine
// thread takes the last core of the fastest class (both its logical CPUs), which no worker may use.
[[nodiscard]] CpuPlacement cpu_placement(const OffloadSwitches& switches, std::uint32_t configured,
                                         bool engine_isolate = false);
// Pins the calling thread (the engine's round thread) to `cpus` (CpuPlacement::engine; 0: unpinned).
void place_engine_thread(std::uint64_t cpus) noexcept;

} // namespace infernix::runtime::expert_offload
