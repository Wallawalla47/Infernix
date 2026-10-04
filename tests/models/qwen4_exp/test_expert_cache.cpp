// Host expert cache of Qwen3.8-Flash-Next (docs/maintainer/qwen3_8-flash-next-design.md §9):
// LFRU decisions equal tools/expert_cache_replay step for step, residency words round-trip,
// frames are never reused while a round that may read them is in flight, and a resized pool (design
// §19.3.7, RT8) evicts the lowest-score experts wherever they sit, moves the survivors below the new
// top and never names a frame above it.

#include "models/qwen4_exp/program/expert_cache/expert_cache.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NINFER_SOURCE_DIR
#    define NINFER_SOURCE_DIR "."
#endif

using namespace ninfer::models::qwen4_exp::expert_cache;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

void test_lfru_conformance() {
    const std::string path = std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/expert_cache/lfru_conformance.txt";
    std::ifstream in(path);
    if (!in) {
        check(false, "conformance fixture readable");
        return;
    }
    std::string line;
    std::getline(in, line);
    unsigned keys = 0, capacity = 0;
    long hits = 0, misses = 0;
    std::sscanf(line.c_str(), "# keys %u capacity %u hits %ld misses %ld", &keys, &capacity, &hits, &misses);
    LfruPolicy policy(keys, capacity);
    LfruPolicy::Step step;
    long got_hits = 0, got_misses = 0, groups = 0, mismatches = 0;
    while (std::getline(in, line)) {
        std::istringstream s(line);
        std::string token;
        s >> token; // "G"
        std::vector<std::uint32_t> group, victims;
        bool after_bar = false;
        while (s >> token) {
            if (token == "|") {
                after_bar = true;
                continue;
            }
            (after_bar ? victims : group).push_back(static_cast<std::uint32_t>(std::stoul(token)));
        }
        policy.step(group, step);
        got_hits += static_cast<long>(step.hits.size());
        got_misses += static_cast<long>(step.misses.size());
        if (step.victims != victims) { ++mismatches; }
        ++groups;
    }
    std::printf("LFRU conformance: %ld groups, %ld mismatching steps\n", groups, mismatches);
    check(groups > 1000 && mismatches == 0, "LFRU victims equal the replay tool's, step for step");
    check(got_hits == hits && got_misses == misses, "LFRU hit and miss totals equal the replay tool's");
}

void test_residency_entries() {
    ResidencyEntry e{ResidencyState::kReady, 63, ResidencyEntry::kMaxFrame};
    check(ResidencyEntry::decode(e.encode()).frame == ResidencyEntry::kMaxFrame, "frame round trip");
    check(e.encode() == 0xBFFFFFFFU, "entry bit layout");
    const auto absent = e.next(ResidencyState::kAbsent, 0);
    check(absent.generation == 0 && absent.frame == 0 && absent.state == ResidencyState::kAbsent, "generation wraps");
    bool threw = false;
    try {
        (void)ResidencyEntry::decode(0xC0000000U);
    } catch (const std::exception&) { threw = true; }
    check(threw, "state 3 is invalid");
}

void test_budgeted_admission() {
    std::mt19937 rng(4);
    LfruPolicy policy(1024, 100);
    LfruPolicy::Step step;
    for (int i = 0; i < 5000; ++i) {
        std::set<std::uint32_t> g;
        while (g.size() < 10) { g.insert(static_cast<std::uint32_t>(rng() % (i % 500 < 250 ? 300 : 1024))); }
        const std::vector<std::uint32_t> group(g.begin(), g.end());
        const std::size_t budget = i % 3;
        policy.step(group, step, budget);
        check(step.admitted.size() <= std::max<std::size_t>(budget, 0) || budget >= step.misses.size(), "budget respected");
        check(policy.resident_count() <= policy.capacity(), "capacity respected");
        for (std::uint32_t v : step.victims) { check(!g.count(v), "current group never evicted"); }
        if (g_failures) { return; }
    }
}

void test_frame_epochs() {
    FramePool pool(3, 1);
    auto a = pool.acquire(), b = pool.acquire(), c = pool.acquire();
    check(a && b && c && !pool.acquire(), "three frames, then none");
    pool.retire(*a, 10); // round_started sampled as 10 after the ABSENT write became visible
    pool.on_round_done(10);
    check(!pool.is_free(*a), "round 11 may still read the frame");
    pool.on_round_done(11);
    check(pool.is_free(*a) && pool.acquire() == a, "free once round_done >= r_s + D");
    pool.retire(*b, 50);
    pool.on_quiescent();
    check(pool.is_free(*b), "quiescence frees every pending frame");
}

void test_pool_resize() {
    FramePool pool(4, 1, 8);
    check(pool.backed() == 4 && pool.max_frames() == 8, "4 of 8 frames backed");
    std::vector<std::uint32_t> held;
    while (auto f = pool.acquire()) { held.push_back(*f); }
    check(held.size() == 4, "only backed frames are handed out");
    pool.resize(6);
    auto e = pool.acquire(), f = pool.acquire();
    check(e && f && *e >= 4 && *f >= 4 && !pool.acquire(), "growing frees exactly the new frames");
    pool.release_now(*e);
    pool.release_now(*f);
    pool.release_now(held[0]);
    const auto low = pool.acquire_below(4);
    check(low && *low < 4, "a relocation target lies below the limit");
    pool.release_now(*low);
    pool.resize(4);
    check(pool.backed() == 4 && pool.free_count() == 1, "shrinking drops the top frames from the free list");
    bool threw = false;
    try {
        pool.resize(2); // frames 2 and 3 hold experts
    } catch (const std::logic_error&) { threw = true; }
    check(threw, "a shrink over held frames is refused");
}

// Fills a 12-frame pool, gives the experts distinct scores, shrinks to 5 and grows back to 9:
// victims are the 7 lowest-score experts wherever their frames are; every survivor ends below 5
// (relocated by kRelocate commands from its old frame); growing loads queued experts only into the
// new frames.
void test_controller_resize() {
    constexpr std::uint32_t kKeys = 64;
    CacheController cache(kKeys, 12, 0, 1, 16);
    std::vector<CacheController::Command> cmds;
    std::vector<std::uint32_t> frame_of(kKeys, ~0U);
    const auto apply = [&] {
        for (const auto& c : cmds) {
            if (c.kind == CacheController::Command::Kind::kCopy) { frame_of[c.key] = c.frame; }
            if (c.kind == CacheController::Command::Kind::kRelocate) {
                check(frame_of[c.key] == c.source, "a relocation starts from the expert's frame");
                frame_of[c.key] = c.frame;
            }
            if (c.kind == CacheController::Command::Kind::kWriteEntry &&
                ResidencyEntry::decode(c.word).state == ResidencyState::kAbsent) {
                frame_of[c.key] = ~0U;
            }
        }
        cmds.clear();
    };
    std::uint64_t round = 0;
    // Key k (0..11) is routed k + 1 times: key 11 scores highest.
    for (std::uint32_t k = 0; k < 12; ++k) {
        for (std::uint32_t use = 0; use <= k; ++use) {
            const std::uint32_t group[] = {k};
            cache.on_route(group, ++round, cmds);
            cache.on_quiescent(cmds);
            apply();
        }
    }
    check(cache.policy().resident_count() == 12 && cache.queued_loads() == 0, "12 experts resident");
    cache.resize(5, cmds);
    const std::vector<CacheController::Command> shrink = cmds;
    apply();
    check(cache.frames() == 5 && cache.policy().resident_count() == 5, "the pool and the policy shrink to 5");
    for (std::uint32_t k = 0; k < 12; ++k) {
        const bool kept = k >= 7;
        check(cache.policy().resident(k) == kept, "the 7 lowest-score experts are evicted");
        check(kept ? (frame_of[k] < 5 && cache.entry(k).state == ResidencyState::kReady &&
                      cache.entry(k).frame == frame_of[k])
                   : (frame_of[k] == ~0U && cache.entry(k).state == ResidencyState::kAbsent),
              "survivors sit below the new top, victims are absent");
    }
    const bool relocations_last = std::is_partitioned(shrink.begin(), shrink.end(), [](const auto& c) {
        return c.kind == CacheController::Command::Kind::kWriteEntry;
    });
    check(relocations_last, "evictions precede relocations");
    // Growing makes room: the next routes' experts load into the new frames.
    cache.resize(9, cmds);
    apply();
    check(cache.frames() == 9, "the pool grows to 9");
    for (std::uint32_t k = 20; k < 24; ++k) {
        const std::uint32_t group[] = {k};
        cache.on_route(group, ++round, cmds);
        cache.on_quiescent(cmds);
        apply();
    }
    std::set<std::uint32_t> frames;
    for (std::uint32_t k = 0; k < kKeys; ++k) {
        if (frame_of[k] != ~0U) {
            check(frame_of[k] < 9, "no expert above the grown top");
            check(frames.insert(frame_of[k]).second, "one expert per frame");
        }
    }
    check(cache.policy().resident_count() == 9, "the grown pool fills");
    // Shrinking to zero evicts everything and leaves a working, empty cache.
    cache.resize(0, cmds);
    apply();
    check(cache.frames() == 0 && cache.policy().resident_count() == 0, "a pool can shrink to no frames");
    const std::uint32_t group[] = {30};
    cache.on_route(group, ++round, cmds);
    check(cmds.empty() && cache.queued_loads() == 0, "an empty pool admits nothing");
}

// An agent-and-device simulation. Each round reads the frames named by READY entries; the agent
// applies the route log while that round runs and its commands become visible to later rounds.
// Invariants: a READY entry's frame holds its expert, and a frame is overwritten only after every
// round that read its previous expert has finished (design §9.5).
void test_agent_simulation() {
    const std::uint32_t keys = 6 * 64, frames = 120;
    // Slack covers peak admissions per round times (D + 1): ~27 per round here, held one extra round.
    CacheController cache(keys, frames + 48, 48, 1);
    std::vector<std::uint32_t> device_entry(keys, 0);
    std::vector<std::int64_t> frame_content(frames + 48, -1);
    std::vector<std::uint64_t> frame_last_read(frames + 48, 0);
    std::vector<CacheController::Command> cmds;
    std::mt19937 rng(77);
    std::uint64_t round_done = 0;
    long cpu_served = 0, gpu_served = 0, policy_hits = 0, violations = 0;
    auto apply = [&](std::uint64_t done) {
        for (const auto& c : cmds) {
            if (c.kind == CacheController::Command::Kind::kCopy) {
                if (frame_content[c.frame] >= 0 && frame_last_read[c.frame] > done) { ++violations; }
                frame_content[c.frame] = c.key;
            } else {
                device_entry[c.key] = c.word;
            }
        }
        cmds.clear();
    };
    for (std::uint64_t round = 1; round <= 3000; ++round) {
        for (int layer = 0; layer < 6; ++layer) {
            std::set<std::uint32_t> g;
            while (g.size() < 8) { g.insert(layer * 64 + static_cast<std::uint32_t>(rng() % (round % 400 < 200 ? 24 : 64))); }
            for (std::uint32_t k : g) {
                policy_hits += cache.policy().resident(k) ? 1 : 0;
                const auto e = ResidencyEntry::decode(device_entry[k]);
                if (e.state == ResidencyState::kReady) {
                    if (frame_content[e.frame] != static_cast<std::int64_t>(k)) { ++violations; }
                    frame_last_read[e.frame] = round;
                    ++gpu_served;
                } else {
                    ++cpu_served;
                }
            }
            const std::vector<std::uint32_t> group(g.begin(), g.end());
            cache.on_route(group, round, cmds);
            apply(round_done);
        }
        round_done = (rng() % 2 == 0 || round == 1) ? round : round - 1; // completion lags by 0 or 1
        cache.on_round_done(round_done, cmds);
        apply(round_done);
    }
    std::printf("agent simulation: %ld GPU-served, %ld CPU-served, %ld policy hits, %zu loads still queued\n",
                gpu_served, cpu_served, policy_hits, cache.queued_loads());
    check(violations == 0, "no frame reused while a round may read it; READY frames hold their expert");
    // Queued loads lag the policy only while frames retire: the device serves almost every policy hit.
    check(gpu_served >= policy_hits * 95 / 100, "device residency tracks the policy");
}

} // namespace

int main() {
    test_residency_entries();
    test_lfru_conformance();
    test_budgeted_admission();
    test_frame_epochs();
    test_agent_simulation();
    test_pool_resize();
    test_controller_resize();
    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all expert cache checks passed\n");
    return 0;
}
