#include "models/qwen4_exp/memory_plan.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

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

} // namespace ninfer::models::qwen4_exp
