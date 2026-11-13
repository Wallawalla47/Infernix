// The Qwen3.8-Flash-Next startup memory plans (design §19.3.7). RT1, the RAM ledger: the experts
// take what the reserve and every planned allocation leave, full mode needs the banks plus the
// margin, the commit limit and the OS's pin bound limit the experts like physical memory, and the
// reserve is applied once. RT7, the VRAM sizing: the display headroom by display state, the OS budget without
// stacking the headroom on it, the graph reserve, fixed allocations that do not fit, the frame
// clamp, and the spill test with noisy free-memory readings. RT9, the runtime law of the elastic
// pool: shrink on low free memory or an exceeded budget, hysteresis, the grow delay and per-boundary
// cap, display hot-plug, and the reserve spent by Infernix's own later allocations. The exclusive RAM
// layout's tier for one record size (the Qwen3.5-122B-A10B's, plan_exclusive_tier): the shadows cover the
// loan, a sixteenth of the frames and the KV growth, the seeds never take the frames they cover, and RAM holds
// every other expert.

#include "runtime/expert_offload/memory_plan.h"
#include "runtime/expert_offload/ram_monitor.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <deque>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace infernix;
using namespace infernix::runtime::expert_offload;

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

// The page-locking limit (Windows: the GPU's shared system memory): the other pins draw on it first,
// the experts pin what is left, and the rest of their RAM stays unpinned; it never shrinks the RAM
// the experts get. Without a known limit nothing changes.
void test_pin_room() {
    HostMemoryDemand d       = flash_next();
    d.prefix_cache           = 4 * kGiB;
    const std::uint64_t need = need_for_full(d);
    const HostMemoryLedger unlimited = plan_host_memory(snapshot(need, 200 * kGiB), d);
    require(unlimited.placement == ExpertPlacement::Full && !unlimited.pin_limited && unlimited.expert_pageable == 0,
            "no pin limit: full mode as before");
    // A 96 GB machine whose limit is half of RAM (47.9 GiB, 1 GiB margin): every expert fits in RAM
    // but not in the pin room, so the SSD tier holds all of them, part unpinned.
    const std::uint64_t pins = plan_host_memory(snapshot(need, 200 * kGiB), d).planned; // unchanged by the room
    d.pin_room = 47 * kGiB;
    const HostMemoryLedger half = plan_host_memory(snapshot(need, 200 * kGiB), d);
    const std::uint64_t other =
        d.other_pinned + d.later_pinned + d.prefix_cache +
        ((d.other_pinned + d.later_pinned + d.prefix_cache) * 25 + 9'999) / 10'000;
    require(half.placement == ExpertPlacement::Tier && half.pin_limited && !half.pin_capped &&
                half.expert_ram == unlimited.expert_ram && half.expert_pinned == 47 * kGiB - other &&
                half.expert_pinned + half.expert_pageable == half.expert_ram && half.planned == pins,
            "a pin room below the banks splits the full need into pinned and unpinned RAM");
    require(half.summary().find("every expert fits in RAM") != std::string::npos &&
                half.summary().find("page-locked for the GPU") != std::string::npos,
            "the summary names the split: " + half.summary());
    require(half.describe().find("pin room 47.00") != std::string::npos &&
                half.describe().find("limited by the pin room") != std::string::npos,
            "the ledger line names the pin room: " + half.describe());
    // The 64 GB machine of the report: 52.7 GiB available, a limit of half of RAM. The RAM the
    // experts get does not change; the pinned part is what the room leaves.
    HostMemoryDemand small = d;
    small.pin_room         = 31 * kGiB;
    HostMemoryDemand open  = small;
    open.pin_room.reset();
    const HostMemoryLedger report = plan_host_memory(snapshot(52 * kGiB + 700 * kMiB, 60 * kGiB), small);
    const HostMemoryLedger before = plan_host_memory(snapshot(52 * kGiB + 700 * kMiB, 60 * kGiB), open);
    require(report.placement == ExpertPlacement::Tier && report.expert_ram == before.expert_ram &&
                report.expert_pinned == 31 * kGiB - other && report.expert_pageable == report.expert_ram - report.expert_pinned,
            "a small pin room moves expert RAM from pinned to unpinned, never from RAM to the SSD");
    // A room the other pins take whole leaves no pinned expert RAM (the tier then refuses to start).
    small.pin_room = 2 * kGiB;
    require(plan_host_memory(snapshot(52 * kGiB, 60 * kGiB), small).expert_pinned == 0,
            "the other pins come first");
    // A RAM-bound tier inside the room is not pin limited.
    small.pin_room = 60 * kGiB;
    const HostMemoryLedger ram_bound = plan_host_memory(snapshot(40 * kGiB, 60 * kGiB), small);
    require(ram_bound.placement == ExpertPlacement::Tier && !ram_bound.pin_limited && ram_bound.expert_pageable == 0,
            "a tier bounded by RAM pins all of its expert RAM");
    // --expert-pin-mib caps the pinned part below the room, and says so.
    HostMemoryDemand capped = flash_next();
    capped.expert_pin_cap   = 20 * kGiB;
    const HostMemoryLedger pin_cap = plan_host_memory(snapshot(need_for_full(capped), 200 * kGiB), capped);
    require(pin_cap.placement == ExpertPlacement::Tier && pin_cap.pin_capped && pin_cap.expert_pinned == 20 * kGiB &&
                pin_cap.summary().find("--expert-pin-mib") != std::string::npos,
            "--expert-pin-mib caps the pinned part");
    // The room from a VRAM snapshot: shared size less usage less the margin; unknown without DXGI.
    VramSnapshot vram;
    require(!pin_room(vram).has_value(), "no shared size: no pin limit");
    vram.shared_system  = 48 * kGiB;
    vram.nonlocal_usage = 3 * kGiB;
    require(pin_room(vram) == 48 * kGiB - 3 * kGiB - kPinRoomMarginBytes, "pin room = shared - usage - margin");
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
    require(graph_bound(8, 16, 7, true) == 320 + 56, "sampled drafts: a second draft graph per batch size and step");
    require(graph_bound(2, 8, 0, false, 7) == 2 + 16 + 2 + 16,
            "DFlash2: a proposal graph per batch size and a catch-up graph per batch size and width");
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
    require(internal_reserve(1, kGraphExecutableBytes, false) == 256 * kMiB, "a fixed pool, few graphs: the 256 MiB floor");
    require(internal_reserve(40, kGraphExecutableBytes, false) == 128 * kMiB + 160 * kMiB,
            "a fixed pool reserves for every graph");
    require(internal_reserve(1, kGraphExecutableBytes, true) == 128 * kMiB &&
                internal_reserve(320, kGraphExecutableBytes, true) == 128 * kMiB,
            "an elastic pool reserves 128 MiB at any graph count (its shrinks absorb the executables)");
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
    // An elastic pool reserves 128 MiB however many executables the Program may instantiate; a fixed
    // pool reserves for every one.
    VramDemand many = demand(fixed, 320);
    many.elastic    = true;
    require(size_expert_frames(card(free, DisplayState::Headless), many).reserve == 128 * kMiB,
            "an elastic pool: 128 MiB");
    many.elastic = false;
    require(size_expert_frames(card(free, DisplayState::Headless), many).reserve == 128 * kMiB + 320 * 4 * kMiB,
            "a fixed pool: the reserve covers every executable");

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

// Free device memory per query, and optionally this process's shared system memory use (a source
// with an adapter: Windows, WSL2) in step with it.
class ScriptedSource final : public VramBudgetSource {
public:
    explicit ScriptedSource(std::deque<std::uint64_t> readings, std::deque<std::uint64_t> nonlocal = {})
        : readings_(std::move(readings)), nonlocal_(std::move(nonlocal)) {}
    VramSnapshot query() override {
        VramSnapshot s;
        s.device_free = readings_.front();
        if (readings_.size() > 1) { readings_.pop_front(); }
        if (!nonlocal_.empty()) {
            s.shared_system  = 76 * kGiB;
            s.nonlocal_usage = nonlocal_.front();
            if (nonlocal_.size() > 1) { nonlocal_.pop_front(); }
        }
        return s;
    }

private:
    std::deque<std::uint64_t> readings_;
    std::deque<std::uint64_t> nonlocal_;
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
    require(confirm.readings(n) == "free device memory 10240 MiB before and 10140 MiB after 1024 MiB of allocations",
            "the failure names the free-memory readings it compared");
    // Where the source reads this process's shared system memory use (Windows, WSL2), the readings
    // carry it: a real sysmem fallback raises it, a misread free-memory figure does not.
    ScriptedSource shared({10 * kGiB, 10 * kGiB - 100 * kMiB, 10 * kGiB - 100 * kMiB},
                          {200 * kMiB, 1124 * kMiB, 1124 * kMiB});
    SpillGuard fallback(shared);
    fallback.begin();
    require(fallback.end(n) == n - 100 * kMiB, "a shortfall with shared-memory readings is reported");
    require(fallback.readings(n) == "free device memory 10240 MiB before and 10140 MiB after 1024 MiB of allocations; "
                                    "this process's shared system memory use 200 MiB before and 1124 MiB after",
            "the failure names the shared system memory readings where the source has them");

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
    // Sizing left headroom (256) + reserve (128, an elastic pool's) free, on the chunk grid.
    const std::uint64_t pool = (31'419 * kMiB - other - 384 * kMiB) / kChunk * kChunk;
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
    require(reserve.reserve_left(graphs, pool) == 128 * kMiB - 64 * kMiB, "graph memory spends the reserve");
    require(reserve.target(graphs, pool) <= frames, "an idle start never grows into the internal reserve");
    require(!reserve.decide(graphs, pool, frames, 100.0, true).shrink, "spending the reserve is not pressure");

    // The OS budget falls below this process's usage: shrink even with free memory reported.
    VramSnapshot trimmed = wddm(pool, other);
    trimmed.local_budget = pool + other - kGiB;
    trimmed.device_free  = kGiB;
    require(control.decide(trimmed, pool, frames, 200.0, false).shrink, "an exceeded budget shrinks the pool");
}

void test_exclusive_tier() {
    // The 122B: 48 x 256 experts, 4,993 frames, a ring of 2 x 256 + 1 and a 304-frame wide workspace, 57 frames of
    // KV growth.
    constexpr std::uint64_t keys = 48 * 256, frames = 4'993, loan = 2 * 256 + 1 + 304, kv = 57;
    const ExclusiveTier half = plan_exclusive_tier(keys, frames, loan, kv, 0.5);
    require(half.shadows == loan + frames / 16 + kv, "the shadows cover the loan, a sixteenth of the frames and the KV");
    require(half.seeded == frames / 2, "half the frames seeded");
    require(half.slots == keys - half.seeded + half.shadows, "RAM holds every key but the seeds, and the shadows");
    const ExclusiveTier rest = plan_exclusive_tier(keys, frames, loan, kv, 1.0);
    require(rest.seeded == frames - rest.shadows, "the seeds never take the frames the shadows cover");
    require(rest.slots == keys - rest.seeded + rest.shadows && rest.slots < half.slots, "more seeds, fewer slots");
    require(plan_exclusive_tier(keys, 1'000, 900, 100, 0.5).slots == 0, "shadows taking every frame: no exclusive tier");
    require(plan_exclusive_tier(keys, 0, loan, kv, 0.5).slots == 0, "no frames: no exclusive tier");
    // More frames than keys need: the slots never exceed the keys.
    const ExclusiveTier ample = plan_exclusive_tier(1'000, 100'000, 10, 0, 0.0);
    require(ample.seeded == 0 && ample.slots == 1'000, "no seeds: RAM holds every key, never more");
}

} // namespace

// What the prefill-chunk fits read (Flash-Next's start; the offloaded Qwen3.5's Engine plan, candidates RF-04 (c)):
// the reproducible route's nominal launch, and the lend capacity of a cache not built yet, split among the
// pools of its strides as resize splits it.
void test_nominal_and_lend() {
    VramSnapshot s;
    s.device_free  = 20 * kGiB;
    s.device_total = 32 * kGiB;
    s.has_budget   = true;
    s.display      = DisplayState::Headless;
    require(nominal_snapshot(s, std::nullopt).device_free == s.device_free && nominal_snapshot(s, std::nullopt).has_budget,
            "without a reading the snapshot stands");
    const VramSnapshot busier = nominal_snapshot(s, 5 * kGiB);
    require(busier.device_free == 22 * kGiB && !busier.has_budget && busier.display == DisplayState::Attached,
            "5 GiB in use before loading: sized as if 3 GiB, an attached display and no OS budget");
    require(nominal_snapshot(s, 1 * kGiB).device_free == 18 * kGiB, "1 GiB in use: 2 GiB less free than measured");
    VramSnapshot tight = s;
    tight.device_free  = 1 * kGiB;
    require(nominal_snapshot(tight, 0).device_free == 0, "the nominal free memory stops at zero");

    const std::vector<std::uint64_t> one(48, 10 * kMiB);
    const auto lend = lend_capacity(one, 256, 1000);
    require(lend.frames == 750 && lend.pitch == 10 * kMiB, "one pool lends what it keeps above a quarter");
    require(lend_capacity(one, 256, 1000000).frames == (48 * 256 - 1) - (48 * 256 - 1) / 4,
            "a pool lends at most what its limit keeps above a quarter");
    require(lend.lends(749 * 10 * kMiB) && !lend.lends(749 * 10 * kMiB + 1) && !lend.lends(750 * 10 * kMiB),
            "a run lends with a frame to spare");
    std::vector<std::uint64_t> two(40, 4 * kMiB);
    two.insert(two.end(), 8, 9 * kMiB);
    // 4,800 frames over 48 layers: 4,000 for the 40 layers of 4 MiB (lending 12,000 MiB), 800 for the 8 of 9 MiB
    // (5,400 MiB).
    const auto split = lend_capacity(two, 256, 4800);
    require(split.frames == 3000 && split.pitch == 4 * kMiB, "two pools: the one that lends the most bytes");
    require(!LendCapacity{}.lends(1), "an empty cache lends nothing");
}

// The RAM tiers' resize law (--ram-resize auto) for the reserve R (--ram-free-mib): below the pressure point
// max(R / 2, 2 GiB) a release in whole steps up to the relieve point max(R, pressure point + a step), capped by what
// the tiers can give; growth only after more than the growth floor (the relieve point plus a step) has held for 10 s
// since the last release, by what stays above it, its pieces then due one after another; between the two nothing (the
// band that keeps ordinary swings from cycling the tiers); nothing due when the tiers have no room, which is measured
// again every 30 s under pressure.
void test_ram_control() {
    using infernix::runtime::expert_offload::RamControl;
    RamControl control(8 * kGiB);
    require(control.pressure_point() == 4 * kGiB && control.relieve_point() == 8 * kGiB &&
                control.growth_floor() == 9 * kGiB,
            "R = 8 GiB: release below 4 GiB back to 8 GiB, grow above 9 GiB");
    control.set_room(10 * kGiB, 0);
    require(control.observe(3 * kGiB + kGiB / 2, 0.0) && control.decide(3 * kGiB + kGiB / 2, 0.0).release == 5 * kGiB &&
                control.decide(3 * kGiB + kGiB / 2, 0.0).grow == 0,
            "3.5 GiB available: release 5 GiB (whole steps to 8 GiB free)");
    require(control.decide(4 * kGiB - 1, 0.0).release == 5 * kGiB, "just under the pressure point: 4 GiB and a byte, five steps");
    require(!control.observe(4 * kGiB, 1.0) && control.decide(4 * kGiB, 1.0).release == 0, "at the pressure point: nothing");
    require(!control.observe(6 * kGiB, 1.5) && control.decide(6 * kGiB, 1.5).release == 0,
            "inside the band (below the reserve, above the pressure point): no release");
    control.set_room(kGiB, 0);
    require(control.decide(kGiB / 2, 2.0).release == kGiB, "the release stops at what the tiers can give");
    control.set_room(0, 0);
    control.checked(3.0);
    require(!control.observe(kGiB / 2, 3.0) && !control.observe(kGiB / 2, 32.0) &&
                control.decide(kGiB / 2, 33.0).release == 0,
            "no room: nothing due under pressure");
    require(control.observe(kGiB / 2, 33.0), "no room under pressure: the tiers are asked again after 30 s");
    require(!control.observe(5 * kGiB, 100.0), "no pressure, no room: nothing to ask");

    control.resized(100.0, true);
    control.set_room(0, 6 * kGiB);
    require(!control.observe(12 * kGiB, 100.0) && !control.observe(12 * kGiB, 105.0),
            "growth waits while the threshold has not held for 10 s");
    require(control.observe(12 * kGiB, 110.5) && control.decide(12 * kGiB, 110.5).grow == 3 * kGiB,
            "12 GiB available after 10 s: grow by what stays above the growth floor (3 GiB)");
    require(!control.observe(8 * kGiB + kGiB / 2, 111.0), "a dip below the growth floor restarts the hold");
    require(!control.observe(12 * kGiB, 113.0) && !control.observe(12 * kGiB, 120.0) &&
                control.observe(12 * kGiB, 123.5),
            "the hold restarts from the dip");
    control.resized(124.0, true);
    require(!control.observe(12 * kGiB, 130.0) && control.observe(12 * kGiB, 134.5), "a release restarts the hold");
    control.resized(135.0, false);
    require(control.observe(11 * kGiB, 135.0) && control.decide(11 * kGiB, 135.0).grow == 2 * kGiB,
            "a growth piece does not: the next is due at once");
    control.set_room(0, kGiB);
    require(control.decide(12 * kGiB, 135.0).grow == kGiB, "growth stops at the startup size");

    // A 4 GiB reserve: the pressure point is its half, 2 GiB.
    RamControl four(4 * kGiB);
    four.set_room(8 * kGiB, 0);
    require(four.pressure_point() == 2 * kGiB && four.relieve_point() == 4 * kGiB && four.growth_floor() == 5 * kGiB &&
                four.decide(kGiB + kGiB / 2, 0.0).release == 3 * kGiB && four.decide(3 * kGiB, 0.0).release == 0,
            "R = 4 GiB: release below 2 GiB (1.5: 3 GiB, back to 4.5), none at 3");

    // A reserve under 4 GiB: the pressure point stays at 2 GiB (Windows stalls near 1 GiB), a release brings at least
    // 3 GiB back and growth leaves 4 GiB.
    RamControl small(2 * kGiB);
    small.set_room(0, 8 * kGiB);
    require(small.pressure_point() == 2 * kGiB && small.relieve_point() == 3 * kGiB && small.growth_floor() == 4 * kGiB,
            "R = 2 GiB: release below 2 GiB back to 3 GiB, grow above 4 GiB");
    require(!small.observe(4 * kGiB, 0.0) && !small.observe(4 * kGiB, 20.0), "4 GiB available: nothing to grow into");
    require(!small.observe(6 * kGiB + kGiB / 2, 21.0) && small.observe(6 * kGiB + kGiB / 2, 31.0) &&
                small.decide(6 * kGiB + kGiB / 2, 31.0).grow == 2 * kGiB,
            "6.5 GiB for 10 s: grow 2 GiB, leaving 4.5");

    // A large reserve: the band scales with it.
    const RamControl large(16 * kGiB);
    require(large.pressure_point() == 8 * kGiB && large.relieve_point() == 16 * kGiB && large.growth_floor() == 17 * kGiB,
            "R = 16 GiB: release below 8 GiB back to 16 GiB, grow above 17 GiB");
}

// The monitor wakes the engine when a reading finds a resize due, which due() reports until decide() takes it, and
// every period while a resize is in flight (set_busy), within the band too; its worker runs jobs one at a time in the
// order posted, a job's exception lands in its future, and the destructor runs what is still queued.
void test_ram_monitor() {
    std::atomic<std::uint64_t> available{20 * kGiB};
    std::atomic<int> wakes{0};
    const auto wait_for = [](const auto& done) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!done() && std::chrono::steady_clock::now() < until) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        return done();
    };
    {
        RamMonitor monitor(8 * kGiB, -1, [&] { return available.load(); }, std::chrono::milliseconds(2));
        monitor.set_room(10 * kGiB, 0);
        monitor.set_waker([&] { ++wakes; });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        require(wakes.load() == 0 && !monitor.due(), "monitor: plenty available, no wake");
        available = 3 * kGiB;
        require(wait_for([&] { return wakes.load() > 0; }) && monitor.due(), "monitor: under the pressure point it wakes");
        require(monitor.decide(10 * kGiB, 0).release == 5 * kGiB, "monitor: the due release");
        // The band: once no tick still holds the old reading, decide takes what is due and nothing sets it again.
        available = 6 * kGiB;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        require(monitor.decide(10 * kGiB, 0).release == 0, "monitor: nothing due in the band");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        require(!monitor.due(), "monitor: decide takes the due resize, and the band sets nothing");
        const int before = wakes.load();
        monitor.set_busy(true);
        require(wait_for([&] { return wakes.load() >= before + 3; }) && !monitor.due(),
                "monitor: a resize in flight wakes it every period, nothing due");
        monitor.set_busy(false);

        std::vector<int> order;
        std::future<void> first  = monitor.worker().post([&] { order.push_back(1); });
        std::future<int> second  = monitor.worker().post([&] { order.push_back(2); return 7; });
        std::future<void> failed = monitor.worker().post([] { throw std::runtime_error("refused"); });
        require(second.get() == 7, "worker: a job's result");
        bool threw = false;
        try {
            failed.get();
        } catch (const std::runtime_error&) { threw = true; }
        first.get();
        require(threw && order == std::vector<int>{1, 2}, "worker: jobs run in order, an exception lands in its future");
    }
    std::atomic<int> ran{0};
    {
        HostMemoryWorker worker;
        for (int i = 0; i < 16; ++i) {
            (void)worker.post([&ran] {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                ++ran;
            });
        }
    }
    require(ran.load() == 16, "worker: the destructor runs the queued jobs (a queued release does not leak)");
}

int main() {
    try {
        test_nominal_and_lend();
        test_ram_control();
        test_ram_monitor();
        test_full_mode_boundary();
        test_reserve_applied_once();
        test_planned_terms();
        test_commit_limit();
        test_starved_and_describe();
        test_expert_cap();
        test_pin_room();
        test_graph_bound();
        test_display_headroom();
        test_internal_reserve();
        test_sizing();
        test_spill();
        test_control_law();
        test_exclusive_tier();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("qwen4_exp memory plan checks passed\n");
    return 0;
}
