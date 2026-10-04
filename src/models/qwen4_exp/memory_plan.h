#pragma once

// The startup Host RAM ledger of Qwen3.8-Flash-Next (docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.7): what the experts may take once a reserve stays free for the system and every other
// allocation the engine will make is planned. One reserve covers every pinned allocation.

#include "core/host_memory.h"
#include "ninfer/types.h"

#include <cstdint>
#include <string>

namespace ninfer::models::qwen4_exp {

inline constexpr std::uint64_t kLedgerGiB = 1ULL << 30;

struct HostMemoryDemand {
    // Physical memory left free for the OS and other programs (--ram-headroom-mib).
    std::uint64_t reserve = kDefaultRamHeadroomBytes;
    // Pinned expert banks (all layers).
    std::uint64_t expert_banks = 0;
    // The model's other pinned weights (the token embedding).
    std::uint64_t other_pinned = 0;
    // Pinned and mapped allocations made after the model loads (the Program's buffers).
    std::uint64_t later_pinned = 0;
    // Large pageable allocations made after planning (the n-gram row cache).
    std::uint64_t pageable = 0;
    // Transient load staging (bounce and upload slots), freed when the load ends.
    std::uint64_t load_staging = 0;
    // The process's pageable growth after planning.
    std::uint64_t process_growth = kLedgerGiB;
    // Page-frame and lock bookkeeping, as a fraction of every pinned byte.
    double lock_overhead = 0.0025;
    // Full mode needs this much beyond the banks, so the system is not run at the reserve's edge.
    std::uint64_t full_margin = kLedgerGiB;
};

enum class ExpertPlacement : std::uint8_t {
    Full, // every expert bank pinned in RAM (today's load path)
    Tier, // RAM holds part of the experts; the rest is read from the artifact
};

struct HostMemoryLedger {
    HostMemorySnapshot snapshot;
    HostMemoryDemand demand;
    // Everything but the experts: other and later pins with their lock overhead, pageable caches,
    // staging and growth.
    std::uint64_t planned = 0;
    // What the experts may take (pinned bytes including their lock overhead), never more than
    // full mode needs.
    std::uint64_t expert_ram = 0;
    // Full mode's need: the banks with their lock overhead plus the margin.
    std::uint64_t full_need = 0;
    // The commit limit, not physical memory, bounded expert_ram (a small page file).
    bool commit_limited = false;
    ExpertPlacement placement = ExpertPlacement::Tier;

    // One log line: the inputs and the outcome.
    [[nodiscard]] std::string describe() const;
};

[[nodiscard]] HostMemoryLedger plan_host_memory(const HostMemorySnapshot& snapshot, const HostMemoryDemand& demand);

} // namespace ninfer::models::qwen4_exp
