// The Qwen3.8-Flash-Next startup RAM ledger (design §19.3.7, test RT1): the experts take what the
// reserve and every planned allocation leave, full mode needs the banks plus the margin, the commit
// limit bounds the experts like physical memory, and the reserve is applied once.

#include "models/qwen4_exp/memory_plan.h"

#include <cstdio>
#include <stdexcept>
#include <string>

using namespace ninfer;
using namespace ninfer::models::qwen4_exp;

namespace {

constexpr std::uint64_t kMiB = 1ULL << 20;
constexpr std::uint64_t kGiB = 1ULL << 30;

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

// This machine's model: 64,801 MiB of banks, the 1.18 GiB embedding.
HostMemoryDemand flash_next() {
    HostMemoryDemand d;
    d.reserve      = 2 * kGiB;
    d.expert_banks = 64'801 * kMiB;
    d.other_pinned = 1'212'825'600; // 248,320 x 2,560 BF16
    d.later_pinned = 256 * kMiB;
    d.pageable     = 160 * kMiB;
    d.load_staging = 264 * kMiB;
    return d;
}

HostMemorySnapshot snapshot(std::uint64_t available, std::uint64_t commit) {
    HostMemorySnapshot s;
    s.total_physical     = 96 * kGiB;
    s.available_physical = available;
    s.available_commit   = commit;
    return s;
}

std::uint64_t need_for_full(const HostMemoryDemand& d) {
    const HostMemoryLedger probe = plan_host_memory(snapshot(0, 0), d);
    return d.reserve + probe.planned + probe.full_need;
}

void test_full_mode_boundary() {
    const HostMemoryDemand d = flash_next();
    const std::uint64_t need = need_for_full(d);
    const HostMemoryLedger at = plan_host_memory(snapshot(need, 200 * kGiB), d);
    require(at.placement == ExpertPlacement::Full && at.expert_ram == at.full_need,
            "exactly enough RAM gives full mode with every expert");
    const HostMemoryLedger below = plan_host_memory(snapshot(need - 1, 200 * kGiB), d);
    require(below.placement == ExpertPlacement::Tier && below.expert_ram == at.full_need - 1 && !below.commit_limited,
            "one byte less leaves the experts one byte short of full mode");
    // The margin and the lock overhead are part of the need.
    require(at.full_need > d.expert_banks + d.full_margin, "the banks' lock overhead is counted");
    require(at.full_need - d.full_margin - d.expert_banks ==
                (d.expert_banks * 25 + 9'999) / 10'000,
            "lock overhead is 0.25 % of the banks, rounded up");
    // Today's machine: 95.8 GiB with ~78 GiB available holds every expert at the 2 GiB default,
    // but not at the old 8 GiB reserve plus the margin.
    require(plan_host_memory(snapshot(78 * kGiB, 120 * kGiB), d).placement == ExpertPlacement::Full,
            "78 GiB available runs in full mode");
    HostMemoryDemand old = d;
    old.reserve          = 12 * kGiB;
    require(plan_host_memory(snapshot(78 * kGiB, 120 * kGiB), old).placement == ExpertPlacement::Tier,
            "a 12 GiB reserve does not fit beside every expert");
}

void test_reserve_applied_once() {
    HostMemoryDemand d = flash_next();
    const std::uint64_t available = 50 * kGiB;
    const HostMemoryLedger a      = plan_host_memory(snapshot(available, 200 * kGiB), d);
    d.reserve += 3 * kGiB;
    const HostMemoryLedger b = plan_host_memory(snapshot(available, 200 * kGiB), d);
    require(a.expert_ram - b.expert_ram == 3 * kGiB, "raising the reserve by 3 GiB takes 3 GiB from the experts");
    require(a.expert_ram == available - flash_next().reserve - a.planned, "the experts take exactly what is left");
}

void test_planned_terms() {
    HostMemoryDemand d = flash_next();
    const HostMemoryLedger base = plan_host_memory(snapshot(50 * kGiB, 200 * kGiB), d);
    d.later_pinned += kGiB;
    const HostMemoryLedger more_pins = plan_host_memory(snapshot(50 * kGiB, 200 * kGiB), d);
    require(base.expert_ram - more_pins.expert_ram == kGiB + (kGiB * 25 + 9'999) / 10'000,
            "a later pin costs its bytes plus its lock overhead");
    d                   = flash_next();
    d.pageable += kGiB;
    const HostMemoryLedger more_cache = plan_host_memory(snapshot(50 * kGiB, 200 * kGiB), d);
    require(base.expert_ram - more_cache.expert_ram == kGiB, "a pageable cache costs its bytes, no lock overhead");
}

void test_commit_limit() {
    const HostMemoryDemand d = flash_next();
    const std::uint64_t need = need_for_full(d);
    // Plenty of physical memory, a small page file.
    const HostMemoryLedger small_commit = plan_host_memory(snapshot(need + 10 * kGiB, need - 5 * kGiB), d);
    require(small_commit.placement == ExpertPlacement::Tier && small_commit.commit_limited &&
                small_commit.expert_ram + 5 * kGiB == small_commit.full_need,
            "the commit limit bounds the experts and is reported");
    const HostMemoryLedger enough = plan_host_memory(snapshot(need + 10 * kGiB, need + kGiB), d);
    require(enough.placement == ExpertPlacement::Full && !enough.commit_limited, "enough commit leaves full mode");
}

void test_starved_and_describe() {
    const HostMemoryDemand d     = flash_next();
    const HostMemoryLedger empty = plan_host_memory(snapshot(kGiB, kGiB), d);
    require(empty.expert_ram == 0 && empty.placement == ExpertPlacement::Tier, "no room leaves no expert RAM");
    const std::string line = plan_host_memory(snapshot(78 * kGiB, 120 * kGiB), d).describe();
    require(line.find("RAM ledger: available 78.00 GiB") == 0 && line.find("reserve 2.00") != std::string::npos &&
                line.find("full mode") != std::string::npos,
            "the ledger line names the inputs and the outcome: " + line);
    require(empty.describe().find("too little RAM") != std::string::npos, "a short ledger says so");
}

} // namespace

int main() {
    try {
        test_full_mode_boundary();
        test_reserve_applied_once();
        test_planned_terms();
        test_commit_limit();
        test_starved_and_describe();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("qwen4_exp memory plan checks passed\n");
    return 0;
}
