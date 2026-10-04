#include "models/qwen4_exp/memory_plan.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace ninfer::models::qwen4_exp {
namespace {

std::uint64_t with_lock_overhead(std::uint64_t pinned, double fraction) {
    return pinned + static_cast<std::uint64_t>(std::ceil(static_cast<double>(pinned) * fraction));
}

std::uint64_t saturating_sub(std::uint64_t a, std::uint64_t b) { return a > b ? a - b : 0; }

double gib(std::uint64_t bytes) { return static_cast<double>(bytes) / static_cast<double>(kLedgerGiB); }

} // namespace

HostMemoryLedger plan_host_memory(const HostMemorySnapshot& snapshot, const HostMemoryDemand& demand) {
    HostMemoryLedger out;
    out.snapshot = snapshot;
    out.demand   = demand;
    const std::uint64_t pinned_other =
        with_lock_overhead(demand.other_pinned + demand.later_pinned, demand.lock_overhead);
    out.planned   = pinned_other + demand.pageable + demand.load_staging + demand.process_growth;
    out.full_need = with_lock_overhead(demand.expert_banks, demand.lock_overhead) + demand.full_margin;
    // Physical memory: the reserve and everything planned come first.
    std::uint64_t room = saturating_sub(snapshot.available_physical, demand.reserve + out.planned);
    // Commit: every pin and the process's growth are committed too, and the reserve must survive.
    const std::uint64_t commit_room = saturating_sub(snapshot.available_commit, demand.reserve + out.planned);
    if (commit_room < room) {
        room               = commit_room;
        out.commit_limited = true;
    }
    out.expert_ram = std::min(room, out.full_need);
    out.placement  = room >= out.full_need ? ExpertPlacement::Full : ExpertPlacement::Tier;
    if (out.placement == ExpertPlacement::Full) { out.commit_limited = false; }
    return out;
}

std::string HostMemoryLedger::describe() const {
    char line[640];
    std::snprintf(line, sizeof(line),
                  "RAM ledger: available %.2f GiB of %.2f (commit %.2f); reserve %.2f; other pins %.2f; later pins "
                  "%.2f; caches %.2f; staging %.2f; growth %.2f -> experts %.2f GiB of %.2f needed (banks %.2f + margin "
                  "%.2f)%s; %s",
                  gib(snapshot.available_physical), gib(snapshot.total_physical), gib(snapshot.available_commit),
                  gib(demand.reserve), gib(demand.other_pinned), gib(demand.later_pinned), gib(demand.pageable),
                  gib(demand.load_staging),
                  gib(demand.process_growth), gib(expert_ram), gib(full_need), gib(demand.expert_banks),
                  gib(demand.full_margin), commit_limited ? ", limited by the page file" : "",
                  placement == ExpertPlacement::Full ? "full mode (every expert in RAM)"
                                                     : "too little RAM to hold every expert");
    return line;
}

std::uint32_t graph_bound(std::uint32_t lanes, std::uint32_t max_width, std::uint32_t mtp_draft_tokens) {
    std::uint32_t graphs = lanes;
    if (max_width > 1) { graphs += lanes * max_width; }
    if (mtp_draft_tokens > 0) { graphs += lanes * max_width + lanes * mtp_draft_tokens; }
    return graphs;
}

std::uint64_t display_headroom(DisplayState display, bool elastic, std::optional<std::uint64_t> requested) {
    if (requested) { return *requested; }
    if (display == DisplayState::Headless) { return 256 * kLedgerMiB; }
    return (elastic ? 512 : 1024) * kLedgerMiB;
}

std::uint64_t internal_reserve(std::uint32_t graphs, std::uint64_t graph_bytes, bool elastic) {
    const std::uint64_t counted = elastic ? std::min<std::uint32_t>(graphs, 32) : graphs;
    return std::max(256 * kLedgerMiB, 128 * kLedgerMiB + graph_bytes * counted);
}

VramSizing size_expert_frames(const VramSnapshot& snapshot, const VramDemand& demand) {
    VramSizing out;
    out.snapshot = snapshot;
    out.demand   = demand;
    out.headroom = display_headroom(snapshot.display, demand.elastic, demand.headroom);
    out.reserve  = internal_reserve(demand.graphs, demand.graph_bytes, demand.elastic);
    const auto signed_bytes = [](std::uint64_t v) { return static_cast<std::int64_t>(v); };
    std::int64_t room = signed_bytes(snapshot.device_free) - signed_bytes(demand.fixed) - signed_bytes(out.headroom);
    if (snapshot.has_budget) {
        const std::int64_t budget = signed_bytes(snapshot.local_budget) - signed_bytes(snapshot.local_usage) -
                                    signed_bytes(demand.fixed) - signed_bytes(kVramBudgetMarginBytes);
        if (budget < room) {
            room               = budget;
            out.budget_limited = true;
        }
    }
    out.frame_bytes = room - signed_bytes(out.reserve);
    if (out.frame_bytes > 0 && demand.frame_bytes > 0) {
        const std::uint64_t frames = static_cast<std::uint64_t>(out.frame_bytes) / demand.frame_bytes;
        out.frames = static_cast<std::uint32_t>(std::min<std::uint64_t>(frames, demand.max_frames));
    }
    return out;
}

VramControl::VramControl(const VramDemand& demand, std::uint64_t chunk_bytes, std::uint64_t usage,
                         std::uint64_t pool_bytes, double grow_delay_seconds)
    : demand_(demand), chunk_(chunk_bytes), usage_at_sizing_(usage), pool_at_sizing_(pool_bytes),
      grow_delay_(grow_delay_seconds) {
    demand_.elastic = true;
    demand_.fixed   = 0;
    if (chunk_ == 0 || demand_.frame_bytes == 0) { throw std::invalid_argument("VRAM control: empty chunks or frames"); }
}

std::uint64_t VramControl::headroom(DisplayState display) const {
    return display_headroom(display, true, demand_.headroom);
}

std::uint64_t VramControl::reserve_left(const VramSnapshot& snapshot, std::uint64_t pool_bytes) const {
    const std::uint64_t total = internal_reserve(demand_.graphs, demand_.graph_bytes, true);
    if (!snapshot.has_budget) { return total; }
    // This process's growth since sizing that is not the pool's (the budget usage counts CUDA).
    const auto grown = static_cast<std::int64_t>(snapshot.local_usage) - static_cast<std::int64_t>(usage_at_sizing_) -
                       (static_cast<std::int64_t>(pool_bytes) - static_cast<std::int64_t>(pool_at_sizing_));
    const std::uint64_t used = grown > 0 ? static_cast<std::uint64_t>(grown) : 0;
    return used >= total ? 0 : total - used;
}

std::int64_t VramControl::frame_bytes(const VramSnapshot& snapshot, std::uint64_t pool_bytes) const {
    const auto s = [](std::uint64_t v) { return static_cast<std::int64_t>(v); };
    std::int64_t room = s(snapshot.device_free) + s(pool_bytes) - s(headroom(snapshot.display));
    if (snapshot.has_budget) {
        room = std::min(room, s(snapshot.local_budget) - s(snapshot.local_usage) + s(pool_bytes) -
                                  s(kVramBudgetMarginBytes));
    }
    return room - s(reserve_left(snapshot, pool_bytes));
}

std::uint32_t VramControl::target(const VramSnapshot& snapshot, std::uint64_t pool_bytes) const {
    const std::int64_t bytes = frame_bytes(snapshot, pool_bytes);
    if (bytes <= 0) { return 0; }
    // Whole chunks: the pool maps and releases in chunks.
    const std::uint64_t chunks = static_cast<std::uint64_t>(bytes) / chunk_;
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(chunks * chunk_ / demand_.frame_bytes, demand_.max_frames));
}

VramControl::Decision VramControl::decide(const VramSnapshot& snapshot, std::uint64_t pool_bytes, std::uint32_t frames,
                                          double now_seconds, bool idle) {
    Decision out;
    out.frames = frames;
    const std::uint64_t headroom_bytes = headroom(snapshot.display);
    out.pressure = snapshot.device_free < headroom_bytes / 2 ||
                   (snapshot.has_budget && snapshot.local_usage > snapshot.local_budget);
    const std::int64_t bytes = frame_bytes(snapshot, pool_bytes);
    if (out.pressure) {
        grow_since_ = -1.0;
        // One chunk below the target, so the next reading does not shrink again.
        const std::int64_t chunks = bytes > 0 ? bytes / static_cast<std::int64_t>(chunk_) - 1 : 0;
        const std::uint64_t keep  = chunks > 0 ? static_cast<std::uint64_t>(chunks) * chunk_ : 0;
        const auto shrunk = static_cast<std::uint32_t>(std::min<std::uint64_t>(keep / demand_.frame_bytes, demand_.max_frames));
        if (shrunk < frames) {
            out.frames = shrunk;
            out.shrink = true;
        }
        return out;
    }
    const std::uint32_t goal = target(snapshot, pool_bytes);
    const std::uint64_t goal_bytes = static_cast<std::uint64_t>(goal) * demand_.frame_bytes;
    if (goal <= frames || goal_bytes < pool_bytes + 2 * chunk_) {
        grow_since_ = -1.0;
        return out;
    }
    if (grow_since_ < 0.0) { grow_since_ = now_seconds; }
    if (now_seconds - grow_since_ < grow_delay_) { return out; }
    std::uint64_t grown = goal_bytes;
    if (!idle) { grown = std::min<std::uint64_t>(grown, (pool_bytes / chunk_ + 4) * chunk_); }
    out.frames = static_cast<std::uint32_t>(std::max<std::uint64_t>(frames, grown / demand_.frame_bytes));
    // The timer keeps running until the pool reaches the target, so the engine's boundaries and the
    // monitor's readings see the same decision.
    out.grow = out.frames > frames;
    return out;
}

std::string VramSizing::describe() const {
    const auto mib = [](std::uint64_t bytes) { return static_cast<unsigned long long>(bytes / kLedgerMiB); };
    const char* display = snapshot.display == DisplayState::Headless   ? "no display"
                          : snapshot.display == DisplayState::Attached ? "display attached"
                                                                       : "display unknown (treated as attached)";
    char budget[96] = "";
    if (snapshot.has_budget) {
        std::snprintf(budget, sizeof(budget), "; OS budget %llu MiB, used %llu", mib(snapshot.local_budget),
                      mib(snapshot.local_usage));
    }
    char line[512];
    std::snprintf(line, sizeof(line),
                  "VRAM sizing: free %llu of %llu MiB; %s%s; to allocate %llu; reserve %llu (%u graphs); headroom "
                  "%llu (%s) -> expert frames %u (%.2f GiB)%s%s",
                  mib(snapshot.device_free), mib(snapshot.device_total), display, budget, mib(demand.fixed),
                  mib(reserve), demand.graphs, mib(headroom), demand.headroom ? "set" : "auto", frames,
                  gib(static_cast<std::uint64_t>(frames) * demand.frame_bytes),
                  budget_limited ? ", limited by the OS budget" : "",
                  frame_bytes < 0 ? "; the fixed allocations do not fit" : "");
    return line;
}

} // namespace ninfer::models::qwen4_exp
