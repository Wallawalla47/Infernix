// The Qwen3.8-Flash-Next startup memory plans (design §19.3.7). RT1, the RAM ledger: the experts
// take what the reserve and every planned allocation leave, full mode needs the banks plus the
// margin, the commit limit bounds the experts like physical memory, and the reserve is applied
// once. RT7, the VRAM sizing: the display headroom by display state, the OS budget without
// stacking the headroom on it, the graph reserve, fixed allocations that do not fit, the frame
// clamp, and the spill test with noisy free-memory readings.

#include "models/qwen4_exp/memory_plan.h"

#include <cstdio>
#include <deque>
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

// ---------------------------------------------------------------- RT7

constexpr std::uint64_t kRecord = 2'764'800; // one nvfp4_expert_rg16_v1 record of this model

VramSnapshot card(std::uint64_t free, DisplayState display) {
    VramSnapshot s;
    s.device_free  = free;
    s.device_total = 32'607 * kMiB;
    s.display      = display;
    s.outputs      = display == DisplayState::Attached ? 1 : display == DisplayState::Headless ? 0 : -1;
    return s;
}

VramDemand demand(std::uint64_t fixed, std::uint32_t graphs) {
    VramDemand d;
    d.graphs      = graphs;
    d.fixed       = fixed;
    d.frame_bytes = kRecord;
    d.max_frames  = 48 * 512 - 1;
    return d;
}

void test_graph_bound() {
    require(graph_bound(1, 1, 0) == 1, "plain C = 1: one decode graph");
    require(graph_bound(1, 8, 3) == 1 + 8 + 8 + 3, "C = 1, W = 8 with MTP 3");
    require(graph_bound(2, 16, 0) == 2 + 32, "n-gram only: decode and verification graphs");
    require(graph_bound(8, 16, 7) == 320, "C = 8 with MTP 7 and n-gram 15");
}

void test_display_headroom() {
    require(display_headroom(DisplayState::Headless, false, std::nullopt) == 256 * kMiB, "headless: 256 MiB");
    require(display_headroom(DisplayState::Attached, false, std::nullopt) == 1024 * kMiB,
            "a display with a fixed pool: 1 GiB");
    require(display_headroom(DisplayState::Attached, true, std::nullopt) == 512 * kMiB,
            "a display with an elastic pool: 512 MiB");
    require(display_headroom(DisplayState::Unknown, false, std::nullopt) == 1024 * kMiB,
            "an unknown display counts as attached");
    require(display_headroom(DisplayState::Attached, false, 0) == 0 &&
                display_headroom(DisplayState::Headless, false, 3 * kGiB) == 3 * kGiB,
            "a requested headroom wins in every state");
}

void test_internal_reserve() {
    require(internal_reserve(1, kGraphExecutableBytes, false) == 256 * kMiB, "few graphs: the 256 MiB floor");
    require(internal_reserve(40, kGraphExecutableBytes, false) == 128 * kMiB + 160 * kMiB,
            "a fixed pool reserves for every graph");
    require(internal_reserve(320, kGraphExecutableBytes, true) == 256 * kMiB,
            "an elastic pool counts at most 32 graphs");
}

void test_sizing() {
    // Headless, no OS budget: frames take free - fixed - headroom - reserve.
    const std::uint64_t free  = 28 * kGiB;
    const std::uint64_t fixed = 3 * kGiB;
    const VramSizing headless = size_expert_frames(card(free, DisplayState::Headless), demand(fixed, 20));
    const std::int64_t expect = static_cast<std::int64_t>(free - fixed - 256 * kMiB - 256 * kMiB);
    require(headless.frame_bytes == expect && headless.frames == static_cast<std::uint32_t>(expect / kRecord) &&
                !headless.budget_limited,
            "headless sizing");
    const VramSizing attached = size_expert_frames(card(free, DisplayState::Attached), demand(fixed, 20));
    require(attached.frame_bytes == expect - static_cast<std::int64_t>(768 * kMiB),
            "a display takes 768 MiB more with a fixed pool");

    // An OS budget below free memory bounds the frames with its 64 MiB margin only: the display
    // headroom is not stacked on top of it.
    VramSnapshot budgeted = card(free, DisplayState::Attached);
    budgeted.has_budget   = true;
    budgeted.local_budget = 26 * kGiB;
    budgeted.local_usage  = 0;
    const VramSizing low  = size_expert_frames(budgeted, demand(fixed, 20));
    require(low.budget_limited &&
                low.frame_bytes == static_cast<std::int64_t>(26 * kGiB - fixed - 64 * kMiB - 256 * kMiB),
            "the budget's bound: B - fixed - 64 MiB - reserve");
    budgeted.local_budget = 30 * kGiB; // above F - headroom: the free-memory bound applies
    const VramSizing high = size_expert_frames(budgeted, demand(fixed, 20));
    require(!high.budget_limited && high.frame_bytes == attached.frame_bytes, "a roomy budget changes nothing");
    budgeted.local_budget = 27 * kGiB;
    budgeted.local_usage  = 2 * kGiB; // usage counts against the budget
    require(size_expert_frames(budgeted, demand(fixed, 20)).frame_bytes ==
                static_cast<std::int64_t>(25 * kGiB - fixed - 64 * kMiB - 256 * kMiB),
            "the budget's bound subtracts this process's usage");

    // The fixed allocations do not fit: negative bytes, no frames, and the line says so.
    const VramSizing over = size_expert_frames(card(2 * kGiB, DisplayState::Headless), demand(fixed, 20));
    require(over.frame_bytes < 0 && over.frames == 0, "an overflow leaves no frames");
    require(over.describe().find("the fixed allocations do not fit") != std::string::npos,
            "an overflow is named: " + over.describe());

    // Every expert fits: the frames stop at max_frames.
    VramDemand all = demand(0, 1);
    all.max_frames = 100;
    require(size_expert_frames(card(free, DisplayState::Headless), all).frames == 100, "frames clamp to max_frames");

    VramDemand set   = demand(fixed, 20);
    set.headroom     = 0;
    const auto line  = size_expert_frames(card(free, DisplayState::Attached), set).describe();
    require(line.find("headroom 0 (set)") != std::string::npos && line.find("display attached") != std::string::npos,
            "the sizing line names the headroom's source and the display: " + line);
}

class ScriptedSource final : public VramBudgetSource {
public:
    explicit ScriptedSource(std::deque<std::uint64_t> readings) : readings_(std::move(readings)) {}
    VramSnapshot query() override {
        VramSnapshot s;
        s.device_free = readings_.front();
        if (readings_.size() > 1) { readings_.pop_front(); }
        return s;
    }

private:
    std::deque<std::uint64_t> readings_;
};

void test_spill() {
    const std::uint64_t n = 1 * kGiB;
    require(spill_shortfall(10 * kGiB, 9 * kGiB, n) == 0, "a resident step");
    require(spill_shortfall(10 * kGiB, 9 * kGiB + 30 * kMiB, n) == 0, "a shortfall within the tolerance");
    require(spill_shortfall(10 * kGiB, 10 * kGiB - 512 * kMiB, n) == 512 * kMiB, "half the step spilled");
    require(spill_shortfall(10 * kGiB, 10 * kGiB + kMiB, n) == n, "free memory rose: all of it missing");
    require(spill_shortfall(10 * kGiB, 8 * kGiB, n) == 0, "another program allocated meanwhile: no spill");

    // A transient high reading is re-read before a spill is reported.
    ScriptedSource noisy({10 * kGiB, 10 * kGiB, 9 * kGiB});
    SpillGuard guard(noisy);
    guard.begin();
    require(guard.end(n) == 0, "a second reading clears a transient shortfall");
    ScriptedSource spilled({10 * kGiB, 10 * kGiB - 100 * kMiB, 10 * kGiB - 100 * kMiB});
    SpillGuard confirm(spilled);
    confirm.begin();
    require(confirm.end(n) == n - 100 * kMiB, "a confirmed shortfall is reported");

    // The test seam replaces the sources opened afterwards.
    ninfer::testing::set_vram_budget_source([] { return card(5 * kGiB, DisplayState::Headless); });
    const VramSnapshot faked = open_vram_budget_source(0)->query();
    ninfer::testing::set_vram_budget_source({});
    require(faked.device_free == 5 * kGiB && faked.display == DisplayState::Headless, "the seam's source answers");
}

} // namespace

int main() {
    try {
        test_full_mode_boundary();
        test_reserve_applied_once();
        test_planned_terms();
        test_commit_limit();
        test_starved_and_describe();
        test_graph_bound();
        test_display_headroom();
        test_internal_reserve();
        test_sizing();
        test_spill();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("qwen4_exp memory plan checks passed\n");
    return 0;
}
