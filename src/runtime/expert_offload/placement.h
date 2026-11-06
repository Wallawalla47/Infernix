#pragma once

// How the expert offload stack applies its placement switches (core/temporary_switches.h,
// temporary_switches::offload(): INFERNIX_TMP_LINK_SHARE, _CPU_PLACEMENT, _STAGING_SLOTS), the plain
// decode divisor, the CPU team's settled rates and the engine thread's isolation to Qwen3.8-Flash-Next's
// Program and ExpertOffload, which serves Qwen3.5's offloaded Programs. Placement only: every output bit is
// unchanged.

#include "core/temporary_switches.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <thread>
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
// 107/256. Flash-Next's Program takes it for NVFP4 W4A16 banks (FP4-Local 67/256: struct +6.3 %, short context
// +5.4 %, after 16K +0.1 %, 2026-10-30) and keeps its divisor rule for the others; calls of eight columns or more of
// every NVFP4 bank keep 51/256 there (its wide_link_share, 2026-11-02).
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
    // Log `text` at Info, not Debug: a switch set the team (an A/B arm shows it), or the host cut
    // the default team (default_workers below the configured count).
    bool announce = false;
};
// The placement of a team of `configured` workers under the switches (by default unpinned; without
// INFERNIX_TMP_CPU_PLACEMENT's count, default_workers of the process's logical CPUs). With
// `engine_isolate` (the Qwen3.5-122B's offloaded route: OffloadOptions::engine_isolate) the engine
// thread takes the last core of the fastest class (both its logical CPUs), which no worker may use.
[[nodiscard]] CpuPlacement cpu_placement(const OffloadSwitches& switches, std::uint32_t configured,
                                         bool engine_isolate = false);
// Pins the calling thread (the engine's round thread) to `cpus` (CpuPlacement::engine; 0: unpinned).
void place_engine_thread(std::uint64_t cpus) noexcept;

// One reading of the CPU expert team's rates: experts/s on one-column jobs (decode's misses), expert columns/s on
// the narrow route's widest jobs (the prefill split's per-column cost), and how it was taken.
struct CpuRateReading {
    double experts    = 0.0;
    double columns    = 0.0;
    double busy_share = -1.0; // other processes' CPU share when they kept the CPU busy through it, else < 0
    std::string note;         // how it was taken (the startup report's parenthesis)
};
// The rates the decode miss split and the prefill CPU split take for the session.
struct SettledCpuRates {
    double experts = 0.0; // the best of the readings
    double columns = 0.0;
    std::vector<double> readings; // each reading's experts/s in order (none: no expert record in RAM, the
                                  // reference rates stand)
    double busy_share = -1.0;     // the last reading's when every reading was busy (the startup warns), else < 0
    std::string note;             // the last reading's
    // The startup report's account of the readings: "; the best of 2 readings (9377, 18942 experts/s)", or "".
    [[nodiscard]] std::string readings_text() const {
        if (readings.size() < 2) { return {}; }
        std::string text = "; the best of " + std::to_string(readings.size()) + " readings (";
        for (std::size_t i = 0; i < readings.size(); ++i) {
            text += (i > 0 ? ", " : "") + std::to_string(std::lround(readings[i]));
        }
        return text + " experts/s)";
    }
};
// Right after the expert banks' load or the SSD tier's fill (tens of GiB of direct reads), or while the OS clears
// memory another process freed, memory is slow for some seconds, and a low reading keeps too many decode misses on
// the link for the whole session: Flash-Next's Dense8 read 9,377 experts/s right after its tier's fill and 18,942
// 5 s later (decode -26 %, 2026-10-27); the Qwen3.5-122B read 4,923-6,027 against 9,515-12,805 settled (decode -24
// to -40 %, 2026-10-30). So the rates are read again kRateSettle after the first reading, and again while a reading
// beats the best experts/s by more than kRateSettleGain, at most kRateSettleReadings readings; the best of each
// rate stands. `read` takes one reading (nullopt without expert records in RAM); `pause` waits between readings
// (empty: kRateSettle).
inline constexpr std::chrono::seconds kRateSettle{4};
inline constexpr int kRateSettleReadings = 3;
inline constexpr double kRateSettleGain  = 0.10;
[[nodiscard]] inline SettledCpuRates settle_cpu_rates(const std::function<std::optional<CpuRateReading>()>& read,
                                                      const std::function<void()>& pause = {}) {
    SettledCpuRates out;
    bool quiet = false; // some reading was taken on a quiet CPU
    while (out.readings.size() < static_cast<std::size_t>(kRateSettleReadings)) {
        if (!out.readings.empty()) {
            if (pause) {
                pause();
            } else {
                std::this_thread::sleep_for(kRateSettle);
            }
        }
        const std::optional<CpuRateReading> reading = read();
        if (!reading) { break; }
        const bool improved = reading->experts > out.experts * (1.0 + kRateSettleGain);
        out.experts         = std::max(out.experts, reading->experts);
        out.columns         = std::max(out.columns, reading->columns);
        out.note            = reading->note;
        quiet               = quiet || reading->busy_share < 0.0;
        out.busy_share      = quiet ? -1.0 : reading->busy_share;
        out.readings.push_back(reading->experts);
        if (!improved) { break; }
    }
    return out;
}

} // namespace infernix::runtime::expert_offload
