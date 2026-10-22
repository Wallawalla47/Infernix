#pragma once

// The startup memory plans of an Engine whose artifact offloads its routed experts (Qwen3.8-Flash-Next,
// docs/maintainer/qwen3_8-flash-next-design.md §19.3.7; Qwen3.5-122B-A10B, docs/maintainer/
// qwen3_5-122b-offload-design.md §11), around the shared component's ledger functions
// (runtime/expert_offload/memory_plan.h): the RAM ledger before anything is pinned or read, which
// places the experts in RAM or in the SSD tier, and the VRAM check before the weights are read.

#include "core/vram_budget.h"
#include "infernix/types.h"
#include "runtime/expert_offload/memory_plan.h"

#include <cstdint>
#include <string>

namespace infernix::runtime {

// Planning allowance for the Program's pinned and mapped buffers (I/O staging, sampling, the CPU
// expert service's mapped records); the measured total is well below it.
inline constexpr std::uint64_t kProgramPinnedAllowance = 256ULL << 20;
// The materializer's upload slots and direct-read bounce buffer during the load.
inline constexpr std::uint64_t kLoadStagingBytes = 4ULL * (64ULL << 20) + (8ULL << 20);
// Below this many expert frames most routed experts miss the VRAM cache and decode slows down
// markedly (Flash-Next design §19.2).
inline constexpr std::uint32_t kFewExpertFrames = 2048;

// What the artifact's load plan pins: the routed banks and everything else (the token embedding,
// an offloaded vision tower).
struct ExpertPinnedTerms {
    std::uint64_t expert_banks = 0;
    std::uint64_t other_pinned = 0;
};

// The RAM ledger's demand for these options: the reserve (--ram-headroom-mib), the load plan's pins,
// the Program's allowance, the prefix cache's pinned Host tier (`prefix_cache_bytes`), the load
// staging, --expert-ram-mib, --expert-pin-mib and what Windows still lets the process page-lock for
// the device. The caller adds its own terms (pageable caches, later pins) before planning.
[[nodiscard]] expert_offload::HostMemoryDemand expert_ram_demand(const EngineOptions& options,
                                                                 const ExpertPinnedTerms& pinned,
                                                                 std::uint64_t prefix_cache_bytes,
                                                                 const VramSnapshot& vram);

struct ExpertRamPlan {
    expert_offload::HostMemoryDemand demand;
    expert_offload::HostMemoryLedger ledger;
    // The SSD tier (ExpertPlacement::Tier): the banks stay in the artifact (LoadOptions::stream_experts).
    bool tier = false;
    // The tier's RAM for the Program's OffloadOptions (zero without the tier): pinned record slots
    // less their lock overhead, resident unpinned records, and the pinned bytes later allocations need.
    std::uint64_t expert_ram_bytes      = 0;
    std::uint64_t expert_unpinned_bytes = 0;
    std::uint64_t later_pinned_bytes    = 0;
};

// Plans `demand` against `snapshot`, reports the ledger (an Info summary, Debug details), refuses an
// --expert-ram-mib the ledger cannot give (std::runtime_error), and asks Windows to keep the
// process's pageable growth resident (a Warning when it will not).
[[nodiscard]] ExpertRamPlan plan_expert_ram(const EngineOptions& options, const expert_offload::HostMemoryDemand& demand,
                                            const HostMemorySnapshot& snapshot);

// The VRAM check before the weights are read: `fixed` (the dense weights and the Program's fixed
// allocations) and the graph reserve for `graphs` executables must leave the expert cache room for
// at least one frame of `frame_bytes` (std::runtime_error naming `model` and `contributors`
// otherwise); fewer than kFewExpertFrames frames is a Warning. Returns the sizing.
[[nodiscard]] expert_offload::VramSizing check_expert_vram(const EngineOptions& options, const std::string& model,
                                                           const VramSnapshot& before, std::uint64_t fixed,
                                                           std::uint32_t graphs, std::uint64_t frame_bytes,
                                                           std::uint32_t max_frames, const std::string& contributors);

// An Engine diagnostic, or stderr (Info and above) without an observer.
void report_engine(const EngineOptions& options, DiagnosticLevel level, const std::string& text);
void report_diagnostic(const DiagnosticObserver& observer, DiagnosticLevel level, const std::string& text);

} // namespace infernix::runtime
