// The Qwen3.8-Flash-Next startup memory plans (design §19.3.7). RT1, the RAM ledger: the experts
// take what the reserve and every planned allocation leave, full mode needs the banks plus the
// margin, the commit limit bounds the experts like physical memory, and the reserve is applied
// once. RT7, the VRAM sizing: the display headroom by display state, the OS budget without
// stacking the headroom on it, the graph reserve, fixed allocations that do not fit, the frame
// clamp, and the spill test with noisy free-memory readings. RT9, the runtime law of the elastic
// pool: shrink on low free memory or an exceeded budget, hysteresis, the grow delay and per-boundary
// cap, display hot-plug, and the reserve spent by Infernix's own later allocations.

#include "models/qwen4_exp/memory_plan.h"

#include <cstdio>
#include <deque>
#include <stdexcept>
#include <string>

using namespace infernix;
using namespace infernix::models::qwen4_exp;

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
    d = flash_next();
    d.prefix_cache += kGiB;
    const HostMemoryLedger prefix_tier = plan_host_memory(snapshot(50 * kGiB, 200 * kGiB), d);
    require(prefix_tier.expert_ram == more_pins.expert_ram, "the prefix cache's Host tier is a pin like any other");
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
    require(empty.describe().find("SSD tier") != std::string::npos, "a short ledger names the SSD tier");
}

// --expert-ram-mib: a cap below what the ledger leaves selects the SSD tier with exactly the cap; a
// cap at or above the full need changes nothing; a cap the ledger cannot give is flagged.
void test_expert_cap() {
    HostMemoryDemand d       = flash_next();
    const std::uint64_t need = need_for_full(d);
    d.expert_cap             = 32 * kGiB;
    const HostMemoryLedger capped = plan_host_memory(snapshot(need + kGiB, 200 * kGiB), d);
    require(capped.placement == ExpertPlacement::Tier && capped.expert_ram == 32 * kGiB && capped.capped &&
                !capped.cap_too_large && capped.describe().find("capped by --expert-ram-mib") != std::string::npos,
            "a 32 GiB cap on a full-mode machine gives the SSD tier with 32 GiB");
    d.expert_cap = 100 * kGiB;
    const HostMemoryLedger above = plan_host_memory(snapshot(need + kGiB, 200 * kGiB), d);
    require(above.placement == ExpertPlacement::Full && !above.capped && !above.cap_too_large,
            "a cap above the full need leaves full mode");
    d.expert_cap = 40 * kGiB;
    const HostMemoryLedger short_room = plan_host_memory(snapshot(30 * kGiB, 200 * kGiB), d);
    require(short_room.cap_too_large && short_room.placement == ExpertPlacement::Tier,
            "a cap larger than the ledger leaves is flagged");
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
    infernix::testing::set_vram_budget_source([] { return card(5 * kGiB, DisplayState::Headless); });
    const VramSnapshot faked = open_vram_budget_source(0)->query();
    infernix::testing::set_vram_budget_source({});
    require(faked.device_free == 5 * kGiB && faked.display == DisplayState::Headless, "the seam's source answers");
}

// ---------------------------------------------------------------- RT9

constexpr std::uint64_t kChunk = 64 * kMiB;

// A card where this process has `pool` bytes of frames and `other` bytes of everything else, the OS
// budget is 31,419 MiB and free memory is budget - usage (as RM0a measured on the RTX 5090).
VramSnapshot wddm(std::uint64_t pool, std::uint64_t other, DisplayState display = DisplayState::Headless,
                  std::uint64_t budget = 31'419 * kMiB) {
    VramSnapshot s;
    s.device_total = 32'579 * kMiB;
    s.has_budget   = true;
    s.local_budget = budget;
    s.local_usage  = pool + other;
    s.device_free  = budget > s.local_usage ? budget - s.local_usage : 0;
    s.display      = display;
    s.outputs      = display == DisplayState::Attached ? 1 : 0;
    return s;
}

std::uint32_t frames_in(std::uint64_t bytes) { return static_cast<std::uint32_t>(bytes / kRecord); }

void test_control_law() {
    VramDemand d;
    d.graphs      = 20;
    d.frame_bytes = kRecord;
    d.max_frames  = 48 * 512 - 1;
    const std::uint64_t other = 6 * kGiB; // dense weights, KV, workspace, staging, context
    // Sizing left headroom (256) + reserve (256) free, on the chunk grid.
    const std::uint64_t pool = (31'419 * kMiB - other - 512 * kMiB) / kChunk * kChunk;
    VramControl control(d, kChunk, pool + other, pool);
    const std::uint32_t frames = frames_in(pool);
    double t = 0.0;

    auto steady = control.decide(wddm(pool, other), pool, frames, t, true);
    require(!steady.shrink && !steady.grow && steady.frames == frames, "the sized pool is stable when idle");

    // Another program takes 2 GiB: this process's budget falls by it, free memory to 0.
    const std::uint64_t squeezed_budget = 31'419 * kMiB - 2 * kGiB;
    auto shrink = control.decide(wddm(pool, other, DisplayState::Headless, squeezed_budget), pool, frames, t += 1, false);
    require(shrink.shrink && shrink.pressure && shrink.frames < frames, "low free memory shrinks at once");
    const std::uint64_t new_pool = (static_cast<std::uint64_t>(shrink.frames) * kRecord + kChunk - 1) / kChunk * kChunk;
    // After the shrink the same pressure no longer holds and the pool is stable.
    const VramSnapshot after = wddm(new_pool, other, DisplayState::Headless, squeezed_budget);
    auto settled             = control.decide(after, new_pool, shrink.frames, t += 1, false);
    require(!settled.shrink && !settled.pressure && settled.frames == shrink.frames,
            "one chunk below the target: the next reading does not shrink again");
    require(after.device_free >= control.headroom(DisplayState::Headless), "the shrink restores the headroom");

    // The other program exits: room for the startup pool again. No growth before the delay.
    VramSnapshot freed = wddm(new_pool, other);
    for (double step = 0; step < 29.5; step += 1.0) {
        const auto wait = control.decide(freed, new_pool, shrink.frames, t + step, false);
        require(!wait.grow && wait.frames == shrink.frames, "no growth before 30 s");
    }
    const auto grow = control.decide(freed, new_pool, shrink.frames, t + 30.0, false);
    require(grow.grow && grow.frames == frames_in(new_pool + 4 * kChunk),
            "during rounds the pool grows by at most four chunks per boundary");
    const auto idle_grow = control.decide(freed, new_pool, shrink.frames, t + 30.5, true);
    require(idle_grow.grow && idle_grow.frames >= frames - frames_in(kChunk) && idle_grow.frames <= frames,
            "an idle engine grows to the target at once");

    // Hysteresis: one chunk of room never grows the pool.
    VramControl tight(d, kChunk, pool + other, pool);
    const VramSnapshot one_chunk = wddm(pool - kChunk, other);
    for (double step = 0; step < 120; step += 10) {
        require(!tight.decide(one_chunk, pool - kChunk, frames_in(pool - kChunk), step, true).grow,
                "a target one chunk above the pool does not grow it");
    }

    // A display appears and the desktop takes 400 MiB: the headroom rises from 256 to 512 MiB
    // (elastic) and the remaining ~170 MiB free is below half of it.
    VramControl plugged(d, kChunk, pool + other, pool);
    const VramSnapshot display = wddm(pool, other, DisplayState::Attached, 31'419 * kMiB - 400 * kMiB);
    require(plugged.headroom(DisplayState::Attached) == 512 * kMiB, "elastic headroom with a display");
    require(plugged.decide(display, pool, frames, 0.0, false).shrink, "a display appearing shrinks the pool");

    // Infernix's own later allocations (graph executables, 64 MiB) spend the reserve, not the frames:
    // the target falls by what remains of the reserve's use, and growth never eats into it.
    VramControl reserve(d, kChunk, pool + other, pool);
    const VramSnapshot graphs = wddm(pool, other + 64 * kMiB);
    require(reserve.reserve_left(graphs, pool) == 256 * kMiB - 64 * kMiB, "graph memory spends the reserve");
    require(reserve.target(graphs, pool) <= frames, "an idle start never grows into the internal reserve");
    require(!reserve.decide(graphs, pool, frames, 100.0, true).shrink, "spending the reserve is not pressure");

    // The OS budget falls below this process's usage: shrink even with free memory reported.
    VramSnapshot trimmed = wddm(pool, other);
    trimmed.local_budget = pool + other - kGiB;
    trimmed.device_free  = kGiB;
    require(control.decide(trimmed, pool, frames, 200.0, false).shrink, "an exceeded budget shrinks the pool");
}

} // namespace

int main() {
    try {
        test_full_mode_boundary();
        test_reserve_applied_once();
        test_planned_terms();
        test_commit_limit();
        test_starved_and_describe();
        test_expert_cap();
        test_graph_bound();
        test_display_headroom();
        test_internal_reserve();
        test_sizing();
        test_spill();
        test_control_law();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("qwen4_exp memory plan checks passed\n");
    return 0;
}
