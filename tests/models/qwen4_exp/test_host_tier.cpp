// The SSD tier's RAM-level controller (design §19.3.7, plan memory-tiers.md §4.3, §4.7, §4.8; RT2):
// the decayed LFU against a naive sum of decayed uses, every slot transition T1-T12, and a long
// randomized run whose victims are checked against a naive scan, with the full audit after every
// step. Host only.
#include "runtime/expert_offload/host_tier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace infernix::runtime::expert_offload::expert_cache;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

constexpr std::uint32_t kUnlimited = std::numeric_limits<std::uint32_t>::max();

// ---- decayed LFU against the naive sum
void test_lfu() {
    constexpr std::uint32_t keys = 64;
    const double h               = 16;
    DecayedLfu lfu(keys, h);
    struct Use { std::uint32_t key; double t, w; };
    std::vector<Use> uses;
    std::mt19937_64 rng(3);
    double t = 0;
    for (int step = 0; step < 4000; ++step) {
        const double dt = (rng() % 4 == 0) ? 16.0 : 1.0;
        lfu.advance(dt);
        t += dt;
        for (int u = 0; u < 6; ++u) {
            const auto k = static_cast<std::uint32_t>(rng() % (step < 2000 ? keys : keys / 4));
            const double w = (rng() % 2) ? 1.0 : 1.0 / 64;
            lfu.use(k, w);
            uses.push_back({k, t, w});
        }
    }
    std::vector<double> naive(keys, 0.0);
    for (const auto& u : uses) { naive[u.key] += u.w * std::exp2(-(t - u.t) / h); }
    double worst = 0;
    for (std::uint32_t k = 0; k < keys; ++k) {
        if (naive[k] == 0) {
            check(lfu.score(k) == 0, "an unused key scores 0");
            continue;
        }
        worst = std::max(worst, std::abs(lfu.score(k) - naive[k]) / naive[k]);
    }
    check(worst < 1e-9, "decayed LFU equals the naive sum (relative error " + std::to_string(worst) + ")");
    // Ranking by the stored value equals ranking by the current score.
    for (std::uint32_t a = 0; a < keys; ++a) {
        for (std::uint32_t b = 0; b < keys; ++b) {
            if (naive[a] > 0 && naive[b] > 0 && naive[a] < naive[b] * (1 - 1e-9)) {
                check(lfu.value(a) < lfu.value(b), "ranking by value");
            }
        }
    }
}

HostTier::Config small_config() {
    HostTier::Config c;
    c.num_keys = 100;
    c.slots    = 14; // 2 ring + 1 prefetch + 3 demotion + 8 resident
    c.ring     = 2;
    c.prefetch = 1;
    c.demotion = 3;
    c.half_life = 8;
    return c;
}

std::vector<std::uint32_t> dirty(HostTier& tier) {
    std::vector<std::uint32_t> keys;
    tier.take_dirty(keys);
    std::sort(keys.begin(), keys.end());
    return keys;
}

// ---- transitions
void test_transitions() {
    HostTier tier(small_config());
    tier.check();
    // Pre-fill keys 0..7 (all resident slots); uses rank key k at weight k + 1.
    for (std::uint32_t k = 0; k < 8; ++k) {
        std::int32_t slot = -1;
        check(tier.prefill(k, slot) && slot >= 0, "prefill");
    }
    std::int32_t none = -1;
    check(!tier.prefill(50, none), "prefill stops when the resident slots are full");
    check(dirty(tier).size() == 8, "prefilled keys are dirty");
    for (std::uint32_t k = 0; k < 8; ++k) {
        const std::uint32_t one[] = {k};
        tier.record_uses(0, one, static_cast<double>(k + 1));
    }
    tier.check();

    // T10: a demand landing outranking the lowest resident (key 0) replaces it; the victim's slot
    // takes the landing's place in the ring.
    tier.begin_round(32);
    const std::uint32_t ring0 = tier.ring()[0];
    const std::uint64_t serial0 = tier.serial(ring0);
    {
        const std::uint32_t used[] = {40};
        tier.record_uses(1, used, 5.0);
    }
    const std::int32_t slot0 = tier.slot_of(0);
    const Landing landing{ring0, serial0, 40};
    check(tier.admit({&landing, 1}, false) == 1, "T10 demand landing admitted");
    check(tier.slot_of(40) == static_cast<std::int32_t>(ring0) && !tier.host_copy(0), "T10 victim is the lowest key");
    check(tier.ring()[0] == static_cast<std::uint32_t>(slot0), "T10 the victim's slot joins the ring");
    check(dirty(tier) == std::vector<std::uint32_t>{0, 40}, "T10/T12 dirty keys");
    tier.check();

    // A stale serial (an earlier round's ring) and a key that already has a host copy are discarded.
    tier.begin_round(32);
    const Landing stale{tier.ring()[1], serial0, 41};
    check(tier.admit({&stale, 1}, false) == 0, "T11 a landing of an earlier round is not admitted");
    const Landing twice{tier.ring()[1], tier.serial(tier.ring()[1]), 40};
    check(tier.admit({&twice, 1}, false) == 0, "a landing of a key with a host copy is discarded");
    // A demand landing that does not outrank the lowest resident is discarded.
    const Landing weak{tier.ring()[1], tier.serial(tier.ring()[1]), 42};
    check(tier.admit({&weak, 1}, false) == 0 && !tier.host_copy(42), "a weaker landing is discarded");
    tier.check();

    // T8/T1/T2: queue, promote, publish -> shadow; a shadow is the first victim, without a score
    // comparison.
    tier.queued(7);
    check((tier.pins(static_cast<std::uint32_t>(tier.slot_of(7))) & kPinQueue) != 0, "T8 Queue pin");
    tier.promotion_issued(7);
    const auto s7 = static_cast<std::uint32_t>(tier.slot_of(7));
    check(tier.pins(s7) == kPinH2D, "T1 Queue becomes H2D");
    tier.promotion_completed(7, true);
    check(tier.state(s7) == SlotState::kShadow && tier.pins(s7) == 0 && tier.shadow_count() == 1, "T2 shadow");
    tier.begin_round(32);
    const Landing weak2{tier.ring()[1], tier.serial(tier.ring()[1]), 43};
    check(tier.admit({&weak2, 1}, false) == 1 && !tier.host_copy(7) && tier.host_copy(43),
          "a shadow is evicted for any landing");
    tier.check();

    // T9: a queued key is never a victim; dropping it from the queue releases the pin.
    tier.queued(1);
    tier.begin_round(32);
    {
        const std::uint32_t used[] = {44};
        tier.record_uses(1, used, 100.0);
    }
    const Landing strong{tier.ring()[1], tier.serial(tier.ring()[1]), 44};
    check(tier.admit({&strong, 1}, false) == 1 && tier.host_copy(1), "T8 a queued key is not a victim");
    tier.unqueued(1);
    check(tier.pins(static_cast<std::uint32_t>(tier.slot_of(1))) == 0, "T9 Queue released");
    tier.check();

    // Use pins: a landing read this round is not a victim until the next round begins.
    const auto s44 = static_cast<std::uint32_t>(tier.slot_of(44));
    tier.mark_used(s44);
    check((tier.pins(s44) & kPinUse) != 0, "Use pin");
    tier.begin_round(32);
    check(tier.pins(s44) == 0, "Use released at the next round");

    // T3: VRAM eviction of a key with a host copy keeps it in RAM (a shadow becomes resident).
    tier.queued(2);
    tier.promotion_issued(2);
    tier.promotion_completed(2, true);
    std::uint32_t target = 0;
    check(tier.vram_evicted(2, true, target) == VramEviction::kKeptInRam &&
              tier.state(static_cast<std::uint32_t>(tier.slot_of(2))) == SlotState::kResident,
          "T3 kept in RAM");
    // A promotion that completes after its key left VRAM leaves the slot resident.
    tier.queued(3);
    tier.promotion_issued(3);
    tier.promotion_completed(3, false);
    check(tier.state(static_cast<std::uint32_t>(tier.slot_of(3))) == SlotState::kResident, "unpublished promotion");
    tier.check();

    // T4/T7: a VRAM-only key outranking the lowest resident by the margin is demoted.
    {
        const std::uint32_t used[] = {60, 61};
        tier.record_uses(1, used, 1000.0);
    }
    check(tier.demotion_worthy(60), "a hot key is demotion-worthy");
    check(tier.allow_evict(60), "the gate allows a demotion within the allowance");
    const std::size_t list_before = tier.demotion_slots();
    check(tier.vram_evicted(60, true, target) == VramEviction::kDemote && tier.state(target) == SlotState::kLanding &&
              tier.pins(target) == kPinD2H && tier.demotion_slots() == list_before - 1,
          "T4 demotion target");
    tier.check();
    tier.demotion_completed(60, target, false);
    check(tier.slot_of(60) == static_cast<std::int32_t>(target) && tier.state(target) == SlotState::kResident,
          "T7 resident after the copy");
    check(dirty(tier).back() == 60, "T7 publishes the host pointer");
    // T7 with the key back in VRAM: the slot becomes its shadow.
    check(tier.vram_evicted(61, true, target) == VramEviction::kDemote, "second demotion");
    tier.demotion_completed(61, target, true);
    check(tier.state(target) == SlotState::kShadow, "T7 shadow when re-admitted");
    tier.replenish_demotion_list();
    check(tier.demotion_slots() == 3, "the demotion list is replenished");
    tier.check();

    // T5: an unpublished frame or a cold key is dropped; T6: no allowance left blocks the eviction.
    check(tier.vram_evicted(70, false, target) == VramEviction::kDropped, "T5 unpublished frame dropped");
    check(!tier.demotion_worthy(71) && tier.vram_evicted(71, true, target) == VramEviction::kDropped,
          "T5 a cold key is dropped");
    {
        const std::uint32_t used[] = {72};
        tier.record_uses(0, used, 5000.0);
    }
    tier.begin_round(0);
    check(!tier.allow_evict(72), "T6 a demotion-worthy victim waits without allowance");
    check(tier.allow_evict(71), "T6 a cold victim may still be dropped");
    tier.begin_round(1);
    check(tier.allow_evict(72) && !tier.allow_evict(72), "the allowance is consumed");
    tier.begin_round(kUnlimited);
    check(tier.allow_evict(72) && tier.allow_evict(72), "unlimited allowance");
    tier.check();

    // Prefetch landings need the margin over the victim.
    tier.begin_round(32);
    const std::uint32_t pf = tier.prefetch_list()[0];
    const Landing reading{pf, tier.prefetch_started(pf), 80};
    check(tier.admit({&reading, 1}, true) == 0, "a prefetch still in flight is not admitted");
    tier.prefetch_ended(pf);
    check(tier.admit({&reading, 1}, true) == 0, "a cold prefetch landing is discarded");
    tier.check();
}

// A randomized run's RAM resize: six random slots in service retire together (when nothing refuses them) at round 30
// of each hundred and come back at round 80; the audit runs after each.
void resize_randomly(HostTier& tier, std::mt19937_64& rng, int round, std::size_t& retirements) {
    if (round % 100 == 30) {
        std::vector<std::uint32_t> set;
        while (set.size() < 6) {
            const auto s = static_cast<std::uint32_t>(rng() % tier.config().slots);
            if (tier.list(s) != SlotList::kRetired && std::find(set.begin(), set.end(), s) == set.end()) { set.push_back(s); }
        }
        const bool expected = tier.retirable(set);
        check(tier.retire(set) == expected, "retirable predicts retire (round " + std::to_string(round) + ")");
        retirements += expected ? 1 : 0;
        tier.check();
    } else if (round % 100 == 80) {
        std::vector<std::uint32_t> back;
        for (std::uint32_t s = 0; s < tier.config().slots; ++s) {
            if (tier.list(s) == SlotList::kRetired) { back.push_back(s); }
        }
        tier.restore(back);
        check(tier.retired() == 0, "every retired slot restored");
        tier.check();
    }
}

// ---- randomized: every admission's victim against a naive scan, the audit after every step
void test_random() {
    HostTier::Config c;
    c.num_keys  = 400;
    c.slots     = 96;
    c.ring      = 8;
    c.prefetch  = 2;
    c.demotion  = 4;
    c.half_life = 32;
    HostTier tier(c);
    std::mt19937_64 rng(11);
    std::vector<std::uint8_t> in_vram(c.num_keys, 0), queued(c.num_keys, 0), h2d(c.num_keys, 0);
    struct Demotion { std::uint32_t key, slot; };
    std::vector<Demotion> demotions;
    std::size_t admissions = 0, demoted = 0, checked_victims = 0, retirements = 0;
    for (int round = 0; round < 2000; ++round) {
        tier.begin_round(round % 10 == 0 ? kUnlimited : 4);
        // Complete earlier demotions and promotions.
        for (const auto& d : demotions) { tier.demotion_completed(d.key, d.slot, in_vram[d.key] != 0); }
        demotions.clear();
        for (std::uint32_t k = 0; k < c.num_keys; ++k) {
            if (h2d[k]) {
                h2d[k] = 0;
                tier.promotion_completed(k, in_vram[k] != 0);
            }
        }
        // Uses: a shifting hot set.
        std::vector<std::uint32_t> used;
        const std::uint32_t base = static_cast<std::uint32_t>(round / 200) * 37;
        for (int u = 0; u < 12; ++u) { used.push_back((base + static_cast<std::uint32_t>(rng() % 120)) % c.num_keys); }
        tier.record_uses(1, used, 1.0);
        // Landings into the ring, one at a time, against the naive victim.
        for (std::uint32_t i = 0; i < 3; ++i) {
            const std::uint32_t key  = used[rng() % used.size()];
            const std::uint32_t slot = tier.ring()[i];
            if (tier.host_copy(key)) { continue; }
            std::int32_t expected = -1;
            bool shadow = false;
            for (std::uint32_t s = 0; s < c.slots; ++s) {
                if (tier.pins(s) != 0 || tier.list(s) != SlotList::kNone) { continue; }
                const bool is_shadow = tier.state(s) == SlotState::kShadow;
                if (!is_shadow && tier.state(s) != SlotState::kResident) { continue; }
                const double v = tier.lfu().value(tier.key(s));
                if (expected < 0 || (is_shadow && !shadow)) {
                    expected = static_cast<std::int32_t>(s);
                    shadow   = is_shadow;
                    continue;
                }
                if (is_shadow != shadow) { continue; }
                // Residents go least-used first, shadows most-used first (their experts are VRAM's last victims).
                const double best = tier.lfu().value(tier.key(static_cast<std::uint32_t>(expected)));
                const bool before = shadow ? v > best : v < best;
                if (before || (v == best && s < static_cast<std::uint32_t>(expected))) { expected = static_cast<std::int32_t>(s); }
            }
            const std::uint32_t victim_key = expected >= 0 ? tier.key(static_cast<std::uint32_t>(expected)) : 0;
            const Landing l{slot, tier.serial(slot), key};
            const std::size_t before = tier.resident_count();
            if (tier.admit({&l, 1}, false) == 1) {
                ++admissions;
                if (tier.resident_count() == before && expected >= 0) { // a victim was evicted
                    check(!tier.host_copy(victim_key), "the naive scan's victim was evicted (round " + std::to_string(round) + ")");
                    ++checked_victims;
                }
            }
        }
        // VRAM traffic: promote some resident keys, evict some VRAM keys.
        for (int p = 0; p < 2; ++p) {
            const std::uint32_t key = used[rng() % used.size()];
            if (tier.host_copy(key) && !in_vram[key] && !queued[key] && !h2d[key]) {
                tier.queued(key);
                if (rng() % 4 == 0) {
                    tier.unqueued(key);
                } else {
                    tier.promotion_issued(key);
                    h2d[key]    = 1;
                    in_vram[key] = 1;
                }
            }
        }
        for (int e = 0; e < 2; ++e) {
            const auto key = static_cast<std::uint32_t>(rng() % c.num_keys);
            if (!in_vram[key] || !tier.allow_evict(key)) { continue; }
            std::uint32_t slot = 0;
            in_vram[key] = 0;
            if (tier.vram_evicted(key, !h2d[key], slot) == VramEviction::kDemote) {
                demotions.push_back({key, slot});
                ++demoted;
            }
        }
        // Some VRAM-only keys (as if loaded before this run).
        if (round % 7 == 0) { in_vram[(base + 200 + static_cast<std::uint32_t>(rng() % 50)) % c.num_keys] = 1; }
        tier.replenish_demotion_list();
        for (const std::uint32_t s : tier.ring()) {
            if (rng() % 8 == 0) { tier.mark_used(s); }
        }
        resize_randomly(tier, rng, round, retirements);
        tier.check();
    }
    std::cout << "random: " << admissions << " admissions (" << checked_victims << " victims checked), " << demoted
              << " demotions, " << retirements << " retirements, " << tier.resident_count() << " residents, "
              << tier.shadow_count() << " shadows\n";
    check(admissions > 500 && checked_victims > 100 && demoted > 20 && retirements > 5, "the random run exercised the tier");
}

// ---- prefill expert stream pins: a record the stream planned to copy keeps its slot until release,
// across the round boundaries of a layer walk's steps (the SSD-tier crash of 2026-10-07: an
// admission evicted a planned record before its copy was enqueued).
void test_stream_pins() {
    HostTier tier(small_config());
    for (std::uint32_t k = 0; k < 8; ++k) {
        std::int32_t slot = -1;
        check(tier.prefill(k, slot), "prefill");
        const std::uint32_t one[] = {k};
        tier.record_uses(0, one, static_cast<double>(k + 1));
    }
    (void)dirty(tier);
    // The stream plans keys 0 and 1, the two lowest-ranked residents, and an SSD-only key.
    const std::int32_t slot0 = tier.slot_of(0), slot1 = tier.slot_of(1);
    tier.stream_pin(0);
    tier.stream_pin(1);
    tier.stream_pin(60); // no host copy: not planned, ignored
    check((tier.pins(static_cast<std::uint32_t>(slot0)) & kPinStream) != 0, "stream pin set");
    tier.check();
    // Two step boundaries, each admitting a strong landing: the victims skip the pinned keys.
    for (const std::uint32_t key : {50U, 51U}) {
        tier.begin_round(0);
        const std::uint32_t used[] = {key};
        tier.record_uses(1, used, 100.0);
        const Landing landing{tier.ring()[0], tier.serial(tier.ring()[0]), key};
        check(tier.admit({&landing, 1}, false) == 1 && tier.host_copy(key), "a landing is admitted beside the pins");
        tier.check();
    }
    check(tier.host_copy(0) && tier.host_copy(1) && tier.slot_of(0) == slot0 && tier.slot_of(1) == slot1,
          "stream-pinned records keep their slots across boundaries");
    check(!tier.host_copy(2) && !tier.host_copy(3), "the next-lowest residents were the victims");
    check(dirty(tier) == std::vector<std::uint32_t>{2, 3, 50, 51}, "only the victims' and landings' pointers changed");
    // Released (the next chunk's boundary): key 0 is again the lowest-ranked victim.
    tier.release_stream_pins();
    check(tier.pins(static_cast<std::uint32_t>(slot0)) == 0 && tier.pins(static_cast<std::uint32_t>(slot1)) == 0,
          "stream pins released");
    tier.check();
    tier.begin_round(0);
    const std::uint32_t used[] = {52};
    tier.record_uses(1, used, 100.0);
    const Landing landing{tier.ring()[0], tier.serial(tier.ring()[0]), 52};
    check(tier.admit({&landing, 1}, false) == 1 && !tier.host_copy(0), "after release the lowest key is a victim again");
    tier.check();
}

// ---- the warm start's fill (seeded ranking, released shadows)
void test_warm_fill() {
    // Without seeds every pre-filled key scores 0 and the lowest slot, which the pre-fill gave the
    // best-ranked key, is the first victim.
    {
        HostTier tier(small_config());
        for (std::uint32_t k = 0; k < 8; ++k) {
            std::int32_t slot = -1;
            check(tier.prefill(k, slot), "prefill");
        }
        tier.begin_round(32);
        const std::uint32_t used[] = {40};
        tier.record_uses(1, used, 5.0);
        const Landing landing{tier.ring()[0], tier.serial(tier.ring()[0]), 40};
        check(tier.admit({&landing, 1}, false) == 1 && !tier.host_copy(0), "unseeded: the best-ranked key goes first");
    }
    // Seeded with the last session's counts (key k used 8 - k times), the least used one goes.
    HostTier tier(small_config());
    std::vector<std::uint32_t> counts(100, 0);
    for (std::uint32_t k = 0; k < 8; ++k) { counts[k] = 8 - k; }
    counts[50] = 100; // capped
    tier.seed_uses(counts, 16);
    check(tier.lfu().score(50) == 16.0 && tier.lfu().score(0) == 8.0 && tier.lfu().score(60) == 0.0,
          "seeds are capped counts at the current time");
    for (std::uint32_t k = 0; k < 8; ++k) {
        std::int32_t slot = -1;
        check(tier.prefill(k, slot), "prefill");
    }
    check(!tier.has_free_resident(), "the resident slots are full");
    tier.begin_round(32);
    {
        const std::uint32_t used[] = {40};
        tier.record_uses(1, used, 5.0);
    }
    const Landing landing{tier.ring()[0], tier.serial(tier.ring()[0]), 40};
    check(tier.admit({&landing, 1}, false) == 1 && !tier.host_copy(7) && tier.host_copy(0) && tier.host_copy(40),
          "seeded: the least-used key goes first");
    tier.check();
    (void)dirty(tier);

    // The frames' warm start loads keys 1 and 2 (published: shadows); key 3's load is still in flight.
    for (const std::uint32_t k : {1U, 2U, 3U}) {
        tier.queued(k);
        tier.promotion_issued(k);
    }
    tier.promotion_completed(1, true);
    tier.promotion_completed(2, true);
    check(tier.shadow_count() == 2, "two shadows");
    check(tier.release_shadows() == 2 && tier.shadow_count() == 0, "release_shadows frees the published copies");
    check(!tier.host_copy(1) && !tier.host_copy(2) && tier.host_copy(3), "a pinned slot keeps its key");
    check(dirty(tier) == std::vector<std::uint32_t>{1, 2}, "released keys are dirty");
    tier.check();
    std::int32_t slot = -1;
    check(tier.prefill(90, slot) && tier.prefill(91, slot) && !tier.has_free_resident(),
          "the released slots take the next keys");
    check(tier.release_shadows() == 0, "nothing more to release");
    // A released key VRAM evicts later is demoted like any key without a host copy (T4).
    {
        const std::uint32_t used[] = {1};
        tier.record_uses(0, used, 1000.0);
    }
    std::uint32_t target = 0;
    check(tier.vram_evicted(1, true, target) == VramEviction::kDemote, "a released seed is demoted when evicted");
    tier.demotion_completed(1, target, false);
    check(tier.host_copy(1), "the demoted seed is back in RAM");
    tier.check();
    bool threw = false;
    try {
        tier.seed_uses(std::vector<std::uint32_t>(99, 1), 16);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "seed counts of another key count are refused");
}

// ---- keys held outside the slots (unpinned records past the page-locking limit)
void test_held() {
    HostTier tier(small_config());
    for (std::uint32_t k = 0; k < 8; ++k) {
        std::int32_t slot = -1;
        check(tier.prefill(k, slot), "prefill");
    }
    check(!tier.hold(3), "a key with a slot copy is not held");
    check(tier.hold(60) && tier.hold(61) && !tier.hold(60) && tier.held(60) && tier.held_count() == 2,
          "hold marks keys once");
    check(!tier.host_copy(60) && tier.slot_of(60) < 0, "a held key has no slot");
    std::int32_t slot = -1;
    check(!tier.prefill(60, slot), "a held key is not pre-filled into a slot");
    tier.check();
    // A landing of a held key (a fetch copied its record into a ring slot) is never admitted, however
    // hot: the key keeps its record outside the slots, and no resident loses its copy.
    tier.begin_round(32);
    {
        const std::uint32_t used[] = {60};
        tier.record_uses(1, used, 1000.0);
    }
    const Landing landing{tier.ring()[0], tier.serial(tier.ring()[0]), 60};
    check(tier.admit({&landing, 1}, false) == 0 && tier.slot_of(60) < 0 && tier.resident_count() == 8,
          "a held key's landing is not admitted");
    // Promotions pin nothing and complete without a slot; a VRAM eviction keeps it (T3), consumes no
    // demotion allowance and takes no demotion slot.
    tier.queued(60);
    tier.promotion_issued(60);
    tier.promotion_completed(60, true);
    tier.unqueued(61);
    check(tier.shadow_count() == 0, "a held key never becomes a shadow");
    tier.begin_round(0); // no demotion allowance
    check(tier.allow_evict(60), "evicting a held key needs no allowance");
    const std::size_t demotion_slots = tier.demotion_slots();
    std::uint32_t target = 0;
    check(tier.vram_evicted(60, true, target) == VramEviction::kKeptInRam && tier.demotion_slots() == demotion_slots,
          "a VRAM eviction keeps a held key in RAM");
    tier.check();
    // A RAM shrink gives a held key's record back: the key is SSD-only, and its next landing may be admitted.
    (void)dirty(tier);
    check(tier.unhold(60) && !tier.held(60) && tier.held_count() == 1 && !tier.unhold(60), "unhold ends a hold once");
    check(dirty(tier) == std::vector<std::uint32_t>{60}, "an unheld key is dirty");
    tier.begin_round(32);
    const Landing unheld{tier.ring()[0], tier.serial(tier.ring()[0]), 60};
    check(tier.admit({&unheld, 1}, false) == 1 && tier.host_copy(60), "an unheld key's landing is admitted");
    tier.check();
}

// ---- retirement (a RAM shrink takes slots out of service, a growth restores them): residents and shadows leave RAM
// and are dirty, a list's slots are replaced by the lowest-ranked residents outside the set (the lists keep their
// sizes), a pinned slot refuses the whole set, a class keeps more slots than its demotion list, and restored slots
// come back free.
void test_retire() {
    HostTier tier(small_config()); // ring 0-1, prefetch 2, demotion 3-5, residents 6-13
    for (std::uint32_t k = 0; k < 8; ++k) {
        std::int32_t slot = -1;
        check(tier.prefill(k, slot) && slot == static_cast<std::int32_t>(6 + k), "prefill");
        const std::uint32_t one[] = {k};
        tier.record_uses(0, one, static_cast<double>(k + 1));
    }
    tier.queued(7); // key 7 (slot 13) becomes a shadow
    tier.promotion_issued(7);
    tier.promotion_completed(7, true);
    (void)dirty(tier);
    tier.queued(6); // slot 12
    const std::uint32_t with_pin[] = {7, 12};
    check(!tier.retirable(with_pin) && !tier.retire(with_pin) && tier.retired() == 0 && tier.host_copy(1) &&
              dirty(tier).empty(),
          "a pinned slot refuses the retirement, which changes nothing");
    tier.unqueued(6);
    tier.check();

    // Ring slot 0, the prefetch slot 2, demotion slot 3, key 1's resident slot 7 and key 7's shadow slot 13. The
    // demotion slot's replacement evicts key 0 (slot 6), the ring's key 2 (slot 8), the prefetch list's key 3 (slot 9).
    const std::vector<std::uint32_t> set{0, 2, 3, 7, 13};
    check(tier.retirable(set) && tier.retire(set), "a retirement of list, resident and shadow slots");
    tier.check();
    check(tier.ring().size() == 2 && tier.prefetch_list().size() == 1 && tier.demotion_slots() == 3, "the lists keep their sizes");
    check(tier.ring()[0] == 8 && tier.ring()[1] == 1 && tier.prefetch_list()[0] == 9 && tier.list(6) == SlotList::kDemotion &&
              tier.slot_class(8) == HostTier::kTransit && tier.slot_class(9) == HostTier::kTransit,
          "a list slot's replacement is the lowest-ranked resident outside the set");
    bool retired = true;
    for (const std::uint32_t s : set) {
        retired = retired && tier.list(s) == SlotList::kRetired && tier.state(s) == SlotState::kFree && tier.slot_class(s) == 0;
    }
    check(retired && tier.retired() == 5, "retired slots hold nothing; the retired ring and prefetch slots take the class");
    check(dirty(tier) == std::vector<std::uint32_t>{0, 1, 2, 3, 7}, "the keys that left RAM are dirty");
    check(tier.resident_count() == 3 && tier.shadow_count() == 0 && tier.losses().retired == 4 &&
              tier.losses().residents_evicted == 0,
          "four residents lost to the retirement (the shadow's key stays in VRAM)");

    // Admissions go on with the slots in service: key 40's landing replaces key 4 (slot 10), which joins the ring.
    tier.begin_round(32);
    {
        const std::uint32_t used[] = {40};
        tier.record_uses(1, used, 100.0);
    }
    const Landing landing{tier.ring()[0], tier.serial(tier.ring()[0]), 40};
    check(tier.admit({&landing, 1}, false) == 1 && tier.slot_of(40) == 8 && tier.ring()[0] == 10 && !tier.host_copy(4),
          "an admission swaps with the slots left");
    tier.check();
    // Six slots in service (5, 6, 4 listed; 8, 11, 12 resident), three in the demotion list: two more may go, not three.
    const std::uint32_t three[] = {8, 11, 12}, two[] = {11, 12};
    check(!tier.retirable(three) && !tier.retire(three) && tier.retire(two), "a class keeps a slot beyond its demotion list");
    tier.check();
    bool threw = false;
    try {
        const std::uint32_t twice[] = {11};
        (void)tier.retire(twice);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a retired slot cannot be retired again");

    // Restored slots come back as free resident slots; a slot in service cannot be restored.
    threw = false;
    try {
        const std::uint32_t serving[] = {8};
        tier.restore(serving);
    } catch (const std::logic_error&) {
        threw = true;
    }
    check(threw && tier.retired() == 7, "only retired slots are restored");
    const std::vector<std::uint32_t> back{0, 2, 3, 7, 13, 11, 12};
    tier.restore(back);
    tier.check();
    std::uint32_t refilled = 0;
    for (std::uint32_t k = 60; k < 70; ++k) {
        std::int32_t slot = -1;
        refilled += tier.prefill(k, slot) ? 1U : 0U;
    }
    check(tier.retired() == 0 && refilled == 7 && !tier.has_free_resident(), "the restored slots take keys again");
    tier.check();
}

// ---- size classes (records of different sizes): a class's slots hold only its keys; a landing is admitted by a
// copy into a slot of its class, or by a swap in a class of the transit slots' size; victims, demotion targets and
// loans stay within a class.
HostTier::Config two_classes() {
    HostTier::Config c;
    c.num_keys  = 100;
    c.ring      = 2;
    c.prefetch  = 1;
    c.half_life = 8;
    c.key_class.assign(100, 0);
    for (std::uint32_t k = 50; k < 100; ++k) { c.key_class[k] = 1; }
    // Class 0 (keys 0-49): smaller records, admitted by copy; 2 demotion + 4 resident slots. Class 1 (keys
    // 50-99): records of the transit size, admitted by swap; 1 demotion + 5 resident slots.
    c.classes = {{.slots = 6, .demotion = 2, .transit_pitch = false}, {.slots = 6, .demotion = 1, .transit_pitch = true}};
    c.slots   = 2 + 1 + 6 + 6;
    return c;
}

void test_classes() {
    HostTier tier(two_classes());
    tier.check();
    check(tier.class_count() == 2 && tier.class_of(3) == 0 && tier.class_of(60) == 1, "key classes");
    for (const std::uint32_t k : {0U, 1U, 2U, 3U, 50U, 51U, 52U, 53U}) {
        std::int32_t slot = -1;
        check(tier.prefill(k, slot) && tier.slot_class(static_cast<std::uint32_t>(slot)) == tier.class_of(k),
              "prefill into a slot of the key's class");
        const std::uint32_t one[] = {k};
        tier.record_uses(0, one, static_cast<double>(k % 50 + 1));
    }
    std::int32_t slot = -1;
    check(!tier.prefill(4, slot), "a full class takes no key, whatever the other classes hold");
    check(tier.has_free_resident() && tier.prefill(54, slot), "another class still has room");
    tier.check();

    // A copy admission: the landing of key 10 (class 0) outranks class 0's lowest key (0); the record goes into
    // that key's slot by a copy, and the ring keeps its slot.
    tier.begin_round(32);
    const std::uint32_t ring0 = tier.ring()[0];
    {
        const std::uint32_t used[] = {10};
        tier.record_uses(1, used, 50.0);
    }
    const std::int32_t slot0 = tier.slot_of(0);
    const Landing copied{ring0, tier.serial(ring0), 10};
    check(tier.admit({&copied, 1}, false) == 1, "a copy admission");
    check(tier.copies().size() == 1 && tier.copies()[0].from == ring0 &&
              tier.copies()[0].to == static_cast<std::uint32_t>(slot0) && tier.copies()[0].key == 10,
          "the copy goes from the ring slot into the victim's class slot");
    check(!tier.host_copy(0) && !tier.host_copy(10) && tier.state(static_cast<std::uint32_t>(slot0)) == SlotState::kLanding &&
              (tier.pins(ring0) & kPinCopy) != 0,
          "the victim left; the key has no host copy until the copy completes");
    tier.check();
    bool threw = false;
    try {
        tier.begin_round(32);
    } catch (const std::logic_error&) {
        threw = true;
    }
    check(threw, "a round does not begin while a copy is pending");
    tier.finish_copies();
    check(tier.slot_of(10) == slot0 && tier.ring()[0] == ring0 && tier.pins(ring0) == 0 && tier.copies().empty(),
          "after the copy the key is resident in its class slot and the ring keeps its slot");
    check(tier.host_copy(50) && tier.host_copy(51), "class 1 lost nothing to a class 0 landing");
    tier.check();

    // A swap admission in the transit-size class (full: 50-54): key 60's landing slot joins class 1, the victim's
    // (key 54's, never used) slot joins the ring.
    tier.begin_round(32);
    const std::uint32_t ring1 = tier.ring()[1];
    {
        const std::uint32_t used[] = {60};
        tier.record_uses(1, used, 50.0);
    }
    const std::int32_t slot54 = tier.slot_of(54);
    const Landing swapped{ring1, tier.serial(ring1), 60};
    check(tier.admit({&swapped, 1}, false) == 1 && tier.copies().empty(), "a swap admission needs no copy");
    check(tier.slot_of(60) == static_cast<std::int32_t>(ring1) && tier.slot_class(ring1) == 1 &&
              tier.ring()[1] == static_cast<std::uint32_t>(slot54) &&
              tier.slot_class(static_cast<std::uint32_t>(slot54)) == HostTier::kTransit && !tier.host_copy(54),
          "the swap exchanges the landing's and the victim's slots between the ring and the class");
    tier.check();

    // Demotions take a slot of the key's class.
    {
        const std::uint32_t used[] = {20, 70};
        tier.record_uses(1, used, 1000.0);
    }
    std::uint32_t target = 0;
    check(tier.vram_evicted(20, true, target) == VramEviction::kDemote && tier.slot_class(target) == 0,
          "a class 0 demotion target");
    tier.demotion_completed(20, target, false);
    check(tier.vram_evicted(70, true, target) == VramEviction::kDemote && tier.slot_class(target) == 1,
          "a class 1 demotion target");
    tier.demotion_completed(70, target, false);
    tier.replenish_demotion_list();
    check(tier.demotion_slots() == 3, "each class's demotion list is replenished");
    tier.check();

    // A loan takes each class's count from that class.
    const std::uint32_t asked[] = {1, 3};
    const std::vector<std::uint32_t> lent = tier.lend(asked);
    std::size_t per_class[2] = {0, 0};
    for (const std::uint32_t s : lent) { ++per_class[tier.slot_class(s)]; }
    check(lent.size() == 4 && per_class[0] == 1 && per_class[1] == 3, "a loan's slots come from their classes");
    tier.check();
    tier.give_back();
    tier.check();

    // A demotion whose class's list is empty takes a free slot of the class.
    {
        HostTier fresh(two_classes());
        fresh.begin_round(32);
        std::uint32_t t = 0;
        const std::uint32_t used[] = {20, 21, 22};
        fresh.record_uses(1, used, 1000.0);
        check(fresh.vram_evicted(20, true, t) == VramEviction::kDemote && fresh.vram_evicted(21, true, t) == VramEviction::kDemote,
              "the class's two demotion slots");
        check(fresh.demotion_slots() == 1 && fresh.allow_evict(22) &&
                  fresh.vram_evicted(22, true, t) == VramEviction::kDemote && fresh.slot_class(t) == 0,
              "a third demotion takes a free slot of its class");
        fresh.check();
    }

    // The shadow floor (the exclusive layout): shadows up to it are kept for a prefill's loan. Class 1 (keys 50-99)
    // with a floor of 2: its 2 shadows are not an admission's first victims, and a loan takes free slots only.
    {
        HostTier::Config fc = two_classes();
        fc.classes[1].shadow_floor = 2;
        HostTier floored(fc);
        for (const std::uint32_t k : {50U, 51U, 52U, 53U, 54U}) {
            std::int32_t s = -1;
            check(floored.prefill(k, s), "floored prefill");
            const std::uint32_t one[] = {k};
            floored.record_uses(0, one, static_cast<double>(k - 49));
        }
        for (const std::uint32_t k : {53U, 54U}) { // the two best become VRAM residents with shadows
            floored.queued(k);
            floored.promotion_issued(k);
            floored.promotion_completed(k, true);
        }
        check(floored.shadow_count() == 2, "two shadows, at the floor");
        floored.begin_round(32);
        {
            const std::uint32_t hot[] = {60};
            floored.record_uses(1, hot, 50.0);
        }
        const std::uint32_t r = floored.ring()[0];
        const Landing landing{r, floored.serial(r), 60};
        check(floored.admit({&landing, 1}, false) == 1 && floored.host_copy(53) && floored.host_copy(54) &&
                  !floored.host_copy(50),
              "at the floor a landing replaces the lowest resident, not a shadow");
        const std::uint32_t loan[] = {0, 3};
        check(floored.lend(loan).empty() && floored.shadow_count() == 2,
              "a floored class with no free slot lends nothing, keeping its residents and shadows");
        floored.check();
    }

    // Configurations whose slots do not add up are refused.
    HostTier::Config bad = two_classes();
    bad.slots += 1;
    threw = false;
    try {
        HostTier refused(bad);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "the ring, the prefetch list and the classes must hold every slot");
}

// ---- retirement with size classes: a ring slot's replacement comes from a class of the transit size (the retired
// slot takes that class), a demotion slot's from its own class; without a class of the transit size a ring slot cannot
// be retired; a class's shadow floor scales with its slots in service.
void test_retire_classes() {
    HostTier tier(two_classes()); // ring 0-1, prefetch 2; class 0: demotion 3-4, slots 5-8; class 1: demotion 9, slots 10-14
    for (const std::uint32_t k : {0U, 1U, 2U, 3U, 50U, 51U, 52U, 53U, 54U}) {
        std::int32_t slot = -1;
        check(tier.prefill(k, slot), "prefill");
        const std::uint32_t one[] = {k};
        tier.record_uses(0, one, static_cast<double>(k % 50 + 1));
    }
    (void)dirty(tier);
    const std::uint32_t set[] = {0, 3, 10}; // a ring slot, class 0's demotion slot, key 50's slot (class 1)
    check(tier.retire(set), "a retirement over the transit slots and two classes");
    check(tier.ring()[0] == 11 && tier.slot_class(11) == HostTier::kTransit && tier.slot_class(0) == 1 &&
              tier.list(5) == SlotList::kDemotion && tier.slot_class(5) == 0,
          "the ring's replacement comes from the transit-size class, the demotion list's from its own");
    check(dirty(tier) == std::vector<std::uint32_t>{0, 50, 51}, "each class's lowest key left for a replacement");
    tier.check();
    const std::uint32_t too_many[] = {6, 7, 8}; // class 0 would keep 2 slots, both its demotion list
    check(!tier.retire(too_many), "a class keeps a slot beyond its demotion list");
    tier.restore(set);
    tier.check();
    check(tier.slot_class(0) == 1 && tier.has_free_resident(), "a restored ring slot serves its new class");

    HostTier::Config no_transit = two_classes();
    no_transit.classes[1].transit_pitch = false;
    HostTier copies_only(no_transit);
    const std::uint32_t ring_slot[] = {1};
    check(!copies_only.retirable(ring_slot) && !copies_only.retire(ring_slot),
          "without a class of the transit size a ring slot has no replacement");
    copies_only.check();

    // Class 1's floor of 2 shadows (of 6 slots) keeps them from an admission's victims; with 3 slots in service the
    // floor is 1, so a shadow (the most used, key 54) goes first again.
    HostTier::Config fc = two_classes();
    fc.classes[1].shadow_floor = 2;
    HostTier floored(fc);
    for (const std::uint32_t k : {50U, 51U, 52U, 53U, 54U}) {
        std::int32_t s = -1;
        check(floored.prefill(k, s), "floored prefill");
        const std::uint32_t one[] = {k};
        floored.record_uses(0, one, static_cast<double>(k - 49));
    }
    for (const std::uint32_t k : {53U, 54U}) {
        floored.queued(k);
        floored.promotion_issued(k);
        floored.promotion_completed(k, true);
    }
    const std::uint32_t three[] = {10, 11, 12}; // keys 50-52
    check(floored.retire(three) && floored.shadow_count() == 2, "a retirement of class 1's residents");
    floored.check();
    floored.begin_round(32);
    {
        const std::uint32_t hot[] = {60};
        floored.record_uses(1, hot, 50.0);
    }
    const std::uint32_t r = floored.ring()[0];
    const Landing landing{r, floored.serial(r), 60};
    check(floored.admit({&landing, 1}, false) == 1 && !floored.host_copy(54) && floored.host_copy(53),
          "the scaled floor lets a shadow go first");
    floored.restore(three);
    floored.check();
}

// A demotion whose read failed (the SSD tier's refill, HostExpertTier::refill): its slot is free again and the key
// keeps no host copy.
void test_demotion_failed() {
    HostTier tier(small_config());
    for (std::uint32_t k = 0; k < 8; ++k) {
        std::int32_t slot = -1;
        check(tier.prefill(k, slot), "prefill");
    }
    const std::size_t lists = tier.demotion_slots();
    std::uint32_t target = 0;
    check(tier.vram_evicted(52, true, target) == VramEviction::kDemote && tier.demotion_slots() == lists - 1,
          "a published VRAM victim takes a demotion slot");
    tier.check();
    tier.demotion_failed(52, target);
    check(!tier.host_copy(52) && tier.state(target) == SlotState::kFree, "a failed read frees the slot");
    tier.check();
}

// ---- randomized over three classes (two admitted by copy, one by swap): every admission's victim against a naive
// scan of the landing key's class, the audit after every step.
void test_random_classes() {
    HostTier::Config c;
    c.num_keys  = 300;
    c.ring      = 6;
    c.prefetch  = 1;
    c.half_life = 32;
    c.key_class.resize(300);
    for (std::uint32_t k = 0; k < 300; ++k) { c.key_class[k] = static_cast<std::uint8_t>(k / 100); }
    c.classes = {{.slots = 30, .demotion = 2, .transit_pitch = false},
                 {.slots = 24, .demotion = 2, .transit_pitch = true},
                 {.slots = 27, .demotion = 3, .transit_pitch = false}};
    c.slots   = 6 + 1 + 30 + 24 + 27;
    HostTier tier(c);
    std::mt19937_64 rng(5);
    std::vector<std::uint8_t> in_vram(c.num_keys, 0), h2d(c.num_keys, 0);
    struct Demotion { std::uint32_t key, slot; };
    std::vector<Demotion> demotions;
    std::size_t admissions = 0, copies = 0, checked_victims = 0, demoted = 0, retirements = 0;
    for (int round = 0; round < 3000; ++round) {
        tier.begin_round(round % 10 == 0 ? kUnlimited : 4);
        for (const auto& d : demotions) { tier.demotion_completed(d.key, d.slot, in_vram[d.key] != 0); }
        demotions.clear();
        for (std::uint32_t k = 0; k < c.num_keys; ++k) {
            if (h2d[k]) {
                h2d[k] = 0;
                tier.promotion_completed(k, in_vram[k] != 0);
            }
        }
        std::vector<std::uint32_t> used;
        const std::uint32_t base = static_cast<std::uint32_t>(round / 300) * 41;
        for (int u = 0; u < 12; ++u) { used.push_back((base + static_cast<std::uint32_t>(rng() % 150)) % c.num_keys); }
        tier.record_uses(1, used, 1.0);
        std::vector<Landing> landings;
        std::vector<std::int32_t> expected;
        for (std::uint32_t i = 0; i < 4; ++i) {
            const std::uint32_t key  = used[rng() % used.size()];
            const std::uint32_t slot = tier.ring()[i];
            if (tier.host_copy(key)) { continue; }
            // One landing at a time so the naive victim is computed on the state the admission sees.
            std::int32_t naive = -1;
            bool shadow        = false;
            for (std::uint32_t s = 0; s < c.slots; ++s) {
                if (tier.slot_class(s) != tier.class_of(key) || tier.pins(s) != 0 || tier.list(s) != SlotList::kNone) {
                    continue;
                }
                const bool is_shadow = tier.state(s) == SlotState::kShadow;
                if (!is_shadow && tier.state(s) != SlotState::kResident) { continue; }
                if (naive < 0 || (is_shadow && !shadow)) {
                    naive  = static_cast<std::int32_t>(s);
                    shadow = is_shadow;
                    continue;
                }
                if (is_shadow != shadow) { continue; }
                const double v    = tier.lfu().value(tier.key(s));
                const double best = tier.lfu().value(tier.key(static_cast<std::uint32_t>(naive)));
                const bool before = shadow ? v > best : v < best; // shadows most-used first, residents least-used first
                if (before || (v == best && s < static_cast<std::uint32_t>(naive))) { naive = static_cast<std::int32_t>(s); }
            }
            const std::uint32_t victim_key = naive >= 0 ? tier.key(static_cast<std::uint32_t>(naive)) : 0;
            const Landing l{slot, tier.serial(slot), key};
            const std::size_t before = tier.resident_count();
            if (tier.admit({&l, 1}, false) == 1) {
                ++admissions;
                copies += tier.copies().size();
                tier.check();
                tier.finish_copies();
                if (tier.resident_count() == before && naive >= 0) {
                    check(!tier.host_copy(victim_key) && tier.class_of(victim_key) == tier.class_of(key),
                          "the class's naive victim was evicted (round " + std::to_string(round) + ")");
                    ++checked_victims;
                }
                check(tier.slot_class(static_cast<std::uint32_t>(tier.slot_of(key))) == tier.class_of(key),
                      "the admitted key sits in its class");
            }
        }
        for (int p = 0; p < 2; ++p) {
            const std::uint32_t key = used[rng() % used.size()];
            if (tier.host_copy(key) && !in_vram[key] && !h2d[key]) {
                tier.queued(key);
                tier.promotion_issued(key);
                h2d[key]     = 1;
                in_vram[key] = 1;
            }
        }
        for (int e = 0; e < 2; ++e) {
            const auto key = static_cast<std::uint32_t>(rng() % c.num_keys);
            if (!in_vram[key] || !tier.allow_evict(key)) { continue; }
            std::uint32_t slot = 0;
            in_vram[key] = 0;
            if (tier.vram_evicted(key, !h2d[key], slot) == VramEviction::kDemote) {
                check(tier.slot_class(slot) == tier.class_of(key), "a demotion target of the key's class");
                demotions.push_back({key, slot});
                ++demoted;
            }
        }
        if (round % 7 == 0) { in_vram[(base + 200 + static_cast<std::uint32_t>(rng() % 50)) % c.num_keys] = 1; }
        if (round % 50 == 0) {
            const std::uint32_t loan[] = {3, 3, 3};
            tier.lend(loan);
            tier.check();
            tier.give_back();
        }
        tier.replenish_demotion_list();
        resize_randomly(tier, rng, round, retirements);
        tier.check();
    }
    std::cout << "random classes: " << admissions << " admissions (" << copies << " by copy, " << checked_victims
              << " victims checked), " << demoted << " demotions, " << retirements << " retirements\n";
    check(admissions > 500 && copies > 200 && checked_victims > 100 && demoted > 20 && retirements > 5,
          "the random run exercised the classes");
}

} // namespace

int main() {
    try {
        test_lfu();
        test_demotion_failed();
        test_transitions();
        test_warm_fill();
        test_held();
        test_stream_pins();
        test_retire();
        test_random();
        test_classes();
        test_retire_classes();
        test_random_classes();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "host tier: all checks passed\n";
    return 0;
}
