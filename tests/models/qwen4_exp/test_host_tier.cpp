// The SSD tier's RAM-level controller (design §19.3.7, plan memory-tiers.md §4.3, §4.7, §4.8; RT2):
// the decayed LFU against a naive sum of decayed uses, every slot transition T1-T12, and a long
// randomized run whose victims are checked against a naive scan, with the full audit after every
// step. Host only.
#include "models/qwen4_exp/program/expert_cache/host_tier.h"

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

using namespace ninfer::models::qwen4_exp::expert_cache;

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
    std::size_t admissions = 0, demoted = 0, checked_victims = 0;
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
                const double best = tier.lfu().value(tier.key(static_cast<std::uint32_t>(expected)));
                if (v < best || (v == best && s < static_cast<std::uint32_t>(expected))) { expected = static_cast<std::int32_t>(s); }
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
        tier.check();
    }
    std::cout << "random: " << admissions << " admissions (" << checked_victims << " victims checked), " << demoted
              << " demotions, " << tier.resident_count() << " residents, " << tier.shadow_count() << " shadows\n";
    check(admissions > 500 && checked_victims > 100 && demoted > 20, "the random run exercised the tier");
}

} // namespace

int main() {
    try {
        test_lfu();
        test_transitions();
        test_random();
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
