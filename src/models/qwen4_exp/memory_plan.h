#pragma once

// The startup memory plans of Qwen3.8-Flash-Next (docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.7). Host RAM: what the experts may take once a reserve stays free for the system and every
// other allocation the engine will make is planned; one reserve covers every pinned allocation.
// Device memory: what the expert frames may take once the fixed allocations, a reserve for
// NInfer's own later allocations and a headroom for the display and other programs are left.

#include "core/host_memory.h"
#include "core/vram_budget.h"
#include "ninfer/types.h"

#include <cstdint>
#include <optional>
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
    // The prefix cache's pinned Host tier (--host-context-mib), pinned when the Program starts.
    std::uint64_t prefix_cache = 0;
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

// ---------------------------------------------------------------- device memory

inline constexpr std::uint64_t kLedgerMiB = 1ULL << 20;
// The margin kept below the OS budget (Windows WDDM). The display headroom applies to device free
// memory only, so a budget below free memory is not stacked on it.
inline constexpr std::uint64_t kVramBudgetMarginBytes = 64 * kLedgerMiB;
// Device memory one CUDA graph executable may take once instantiated (graphs are captured lazily,
// after sizing).
inline constexpr std::uint64_t kGraphExecutableBytes = 4 * kLedgerMiB;

struct VramDemand {
    // --vram-headroom-mib; empty selects it from the display state.
    std::optional<std::uint64_t> headroom;
    // The frame pool shrinks at runtime when the display or another program needs memory. Without
    // that nothing absorbs their growth, so the automatic headroom and the graph reserve are larger.
    bool elastic = false;
    // CUDA graph executables the Program may instantiate after sizing, and what each may take.
    std::uint32_t graphs      = 0;
    std::uint64_t graph_bytes = kGraphExecutableBytes;
    // Device allocations still to be made besides the frames (taken from free memory and budget).
    std::uint64_t fixed = 0;
    // One expert frame, and the most frames the residency uses.
    std::uint64_t frame_bytes = 0;
    std::uint32_t max_frames  = 0;
};

struct VramSizing {
    VramSnapshot snapshot;
    VramDemand demand;
    std::uint64_t headroom = 0; // left free for the display and other programs
    std::uint64_t reserve  = 0; // for NInfer's own allocations after sizing (graph executables)
    // What the frames may take; negative when the fixed allocations alone do not fit.
    std::int64_t frame_bytes = 0;
    bool budget_limited      = false; // the OS budget, not free memory, bounded frame_bytes
    std::uint32_t frames     = 0;

    // One log line: the inputs and the outcome.
    [[nodiscard]] std::string describe() const;
};

// CUDA graph executables a Program instantiates at most: one decode graph per batch size, and with
// speculation a verification graph per batch size and width, and the MTP drafter's catch-up (per
// batch size and width) and draft graphs (per batch size and step).
[[nodiscard]] std::uint32_t graph_bound(std::uint32_t lanes, std::uint32_t max_width, std::uint32_t mtp_draft_tokens);
// The headroom left for the display and other programs: the requested one, else 256 MiB without a
// display and, with one (or when unknown), 512 MiB for an elastic pool or 1 GiB for a fixed one.
[[nodiscard]] std::uint64_t display_headroom(DisplayState display, bool elastic, std::optional<std::uint64_t> requested);
// The reserve for graph executables instantiated after sizing: max(256 MiB, 128 MiB + one
// allowance per graph), the graph count capped at 32 for an elastic pool, which absorbs the rest.
[[nodiscard]] std::uint64_t internal_reserve(std::uint32_t graphs, std::uint64_t graph_bytes, bool elastic);
// frame_bytes = min(F - fixed - headroom, B - fixed - 64 MiB) - reserve, with F the device free
// memory and B the OS budget less this process's usage (no budget: F's bound alone); frames =
// clamp(frame_bytes / frame size, 0, max_frames).
[[nodiscard]] VramSizing size_expert_frames(const VramSnapshot& snapshot, const VramDemand& demand);

// The runtime law of an elastic frame pool (design §19.3.7): from the latest snapshot, the pool's
// size and the time, the frame count it should have. Pure and single-threaded; the Program guards
// it. The reserve is spent only by NInfer's own allocations after sizing (graph executables): with
// an OS budget, the growth of this process's usage other than the pool's.
class VramControl {
public:
    // `demand` as sized at startup (elastic); `usage` and `pool_bytes` at that moment.
    VramControl(const VramDemand& demand, std::uint64_t chunk_bytes, std::uint64_t usage, std::uint64_t pool_bytes,
                double grow_delay_seconds = 30.0);

    struct Decision {
        std::uint32_t frames = 0; // the frame count to resize to (the current one: no change)
        bool shrink          = false;
        bool grow            = false;
        bool pressure        = false; // free memory below half the headroom, or the budget exceeded
    };
    // Shrinks at once when free memory falls below half the headroom or this process is over its
    // OS budget, to one chunk below the sizing function's target; grows toward the target once it
    // has stayed two chunks above the pool for the grow delay, by at most four chunks a boundary
    // while rounds run and all at once when `idle`.
    [[nodiscard]] Decision decide(const VramSnapshot& snapshot, std::uint64_t pool_bytes, std::uint32_t frames,
                                  double now_seconds, bool idle);
    // The frame count the sizing function gives now (no hysteresis).
    [[nodiscard]] std::uint32_t target(const VramSnapshot& snapshot, std::uint64_t pool_bytes) const;
    [[nodiscard]] std::uint64_t headroom(DisplayState display) const;
    [[nodiscard]] std::uint64_t reserve_left(const VramSnapshot& snapshot, std::uint64_t pool_bytes) const;
    // A fixed allocation grew (positive) or shrank by `bytes` since sizing (the elastic KV pool, design
    // §19.3.11): it is not the reserve's use.
    void account_fixed(std::int64_t bytes) noexcept {
        usage_at_sizing_ = static_cast<std::uint64_t>(static_cast<std::int64_t>(usage_at_sizing_) + bytes);
    }

private:
    [[nodiscard]] std::int64_t frame_bytes(const VramSnapshot& snapshot, std::uint64_t pool_bytes) const;

    VramDemand demand_;
    std::uint64_t chunk_;
    std::uint64_t usage_at_sizing_;
    std::uint64_t pool_at_sizing_;
    double grow_delay_;
    double grow_since_ = -1.0; // when the target first stood two chunks above the pool; < 0: not now
};

} // namespace ninfer::models::qwen4_exp
