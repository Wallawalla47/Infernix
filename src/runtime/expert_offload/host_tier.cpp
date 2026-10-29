#include "runtime/expert_offload/host_tier.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace infernix::runtime::expert_offload::expert_cache {

namespace {

// Victim heaps: min-heaps on (value, slot), the lowest value first, ties to the lower slot, with each
// entry's index in pos (per slot).
template <typename Entry>
bool heap_before(const Entry& a, const Entry& b) {
    return a.value < b.value || (a.value == b.value && a.slot < b.slot);
}

template <typename Entry>
void sift_up(std::vector<Entry>& heap, std::vector<std::int32_t>& pos, std::size_t i) {
    const Entry e = heap[i];
    while (i > 0) {
        const std::size_t parent = (i - 1) / 2;
        if (!heap_before(e, heap[parent])) { break; }
        heap[i]            = heap[parent];
        pos[heap[i].slot] = static_cast<std::int32_t>(i);
        i                  = parent;
    }
    heap[i]      = e;
    pos[e.slot] = static_cast<std::int32_t>(i);
}

template <typename Entry>
void sift_down(std::vector<Entry>& heap, std::vector<std::int32_t>& pos, std::size_t i) {
    const Entry e       = heap[i];
    const std::size_t n = heap.size();
    for (;;) {
        std::size_t child = 2 * i + 1;
        if (child >= n) { break; }
        if (child + 1 < n && heap_before(heap[child + 1], heap[child])) { ++child; }
        if (!heap_before(heap[child], e)) { break; }
        heap[i]            = heap[child];
        pos[heap[i].slot] = static_cast<std::int32_t>(i);
        i                  = child;
    }
    heap[i]      = e;
    pos[e.slot] = static_cast<std::int32_t>(i);
}

} // namespace

// ---------------------------------------------------------------------------- DecayedLfu

DecayedLfu::DecayedLfu(std::uint32_t num_keys, double half_life) : half_life_(half_life), value_(num_keys, 0.0) {
    if (!(half_life > 0)) { throw std::invalid_argument("DecayedLfu: the half-life must be positive"); }
}

int DecayedLfu::advance(double dt) {
    now_ += dt;
    int halvings = 0;
    if (const double span = (now_ - base_) / half_life_; span >= kRebase) {
        halvings = static_cast<int>(std::floor(span));
        base_ += static_cast<double>(halvings) * half_life_;
        for (double& v : value_) { v = std::ldexp(v, -halvings); }
    }
    growth_ = std::exp2((now_ - base_) / half_life_);
    return halvings;
}

// ---------------------------------------------------------------------------- HostTier

HostTier::HostTier(const Config& config)
    : config_(config), lfu_(config.num_keys, config.half_life), slots_(config.slots), slot_class_(config.slots, kTransit),
      key_class_(config.num_keys, 0), slot_of_(config.num_keys, -1), held_(config.num_keys, 0),
      copying_(config.num_keys, 0), dirty_mark_(config.num_keys, 0) {
    if (config.num_keys == 0) { throw std::invalid_argument("HostTier: no keys"); }
    for (auto& pos : heap_pos_) { pos.assign(config.slots, -1); }
    for (auto& cls : heap_class_) { cls.assign(config.slots, 0); }
    const std::uint64_t transit = static_cast<std::uint64_t>(config.ring) + config.prefetch;
    std::vector<Class> classes  = config.classes;
    if (classes.empty()) {
        if (!config.key_class.empty()) { throw std::invalid_argument("HostTier: key classes without classes"); }
        if (config.slots <= transit + config.demotion) {
            throw std::invalid_argument("HostTier: the slots must exceed the ring, prefetch and demotion lists");
        }
        classes.push_back({static_cast<std::uint32_t>(config.slots - transit), config.demotion, true, config.shadow_floor});
    } else {
        if (config.key_class.size() != config.num_keys) { throw std::invalid_argument("HostTier: a class for every key"); }
        if (classes.size() >= kTransit) { throw std::invalid_argument("HostTier: too many classes"); }
        std::uint64_t total = transit;
        for (const Class& c : classes) {
            if (c.slots <= c.demotion) { throw std::invalid_argument("HostTier: a class's slots must exceed its demotion list"); }
            total += c.slots;
        }
        if (total != config.slots) {
            throw std::invalid_argument("HostTier: the ring, the prefetch list and the classes must hold every slot");
        }
        for (const std::uint8_t c : config.key_class) {
            if (c >= classes.size()) { throw std::invalid_argument("HostTier: a key's class is out of range"); }
        }
        key_class_ = config.key_class;
    }
    std::uint32_t s = 0;
    for (std::uint32_t i = 0; i < config.ring; ++i, ++s) {
        slots_[s].list = SlotList::kRing;
        ring_.push_back(s);
    }
    for (std::uint32_t i = 0; i < config.prefetch; ++i, ++s) {
        slots_[s].list = SlotList::kPrefetch;
        prefetch_.push_back(s);
    }
    classes_.resize(classes.size());
    for (std::size_t c = 0; c < classes.size(); ++c) {
        ClassState& cs     = classes_[c];
        cs.demotion_target = classes[c].demotion;
        cs.transit_pitch   = classes[c].transit_pitch;
        cs.shadow_floor    = classes[c].shadow_floor;
        const std::uint32_t end = s + classes[c].slots;
        for (std::uint32_t i = 0; i < classes[c].demotion; ++i, ++s) {
            slots_[s].list = SlotList::kDemotion;
            slot_class_[s] = static_cast<std::uint8_t>(c);
            cs.demotion.push_back(s);
        }
        // Free resident slots are taken lowest first.
        for (std::uint32_t i = end; i-- > s;) {
            slot_class_[i] = static_cast<std::uint8_t>(c);
            cs.free.push_back(i);
        }
        s = end;
    }
}

std::size_t HostTier::demotion_slots() const {
    std::size_t n = 0;
    for (const ClassState& c : classes_) { n += c.demotion.size(); }
    return n;
}

bool HostTier::has_free_resident() const {
    return std::any_of(classes_.begin(), classes_.end(), [](const ClassState& c) { return !c.free.empty(); });
}

void HostTier::take_dirty(std::vector<std::uint32_t>& keys) {
    keys.insert(keys.end(), dirty_.begin(), dirty_.end());
    for (const auto k : dirty_) { dirty_mark_[k] = 0; }
    dirty_.clear();
}

void HostTier::mark_dirty(std::uint32_t key) {
    if (!dirty_mark_[key]) {
        dirty_mark_[key] = 1;
        dirty_.push_back(key);
    }
}

// Residents leave RAM least-used first. Shadows leave most-used first: a shadow saves a demotion only when its expert
// leaves VRAM, which takes the least-used experts first, so dropping the least-used experts' shadows (as residents)
// made nearly every VRAM victim a demotion. Dense8 struct: 24,318 -> 9,649 demotions a request, exclusive decode
// 327.2 -> 353.2 tok/s (full layout 355.1); with the artifact read at 800 MB/s (a slower SSD) IQ3_S struct
// 410.0 -> 450.0 and Dense8 16K 220.5 -> 237.0 (2026-10-29).
double HostTier::victim_value(std::uint32_t key, Victims kind) const {
    return kind == Victims::kShadow ? -lfu_.value(key) : lfu_.value(key);
}

bool HostTier::eligible(const Slot& s, Victims kind) const {
    return s.pins == 0 && s.list == SlotList::kNone &&
           s.state == (kind == Victims::kShadow ? SlotState::kShadow : SlotState::kResident);
}

void HostTier::push_victim(std::uint32_t slot) {
    const Slot& s        = slots_[slot];
    const std::uint8_t c = slot_class_[slot];
    for (const auto kind : {Victims::kShadow, Victims::kResident}) {
        place_victim(kind, slot, c, c != kTransit && eligible(s, kind));
    }
}

void HostTier::place_victim(Victims kind, std::uint32_t slot, std::uint8_t c, bool wanted) {
    const int k                    = static_cast<int>(kind);
    std::vector<std::int32_t>& pos = heap_pos_[k];
    if (pos[slot] >= 0 && (!wanted || heap_class_[k][slot] != c)) {
        auto& heap          = classes_[heap_class_[k][slot]].heap[k];
        const auto at       = static_cast<std::size_t>(pos[slot]);
        const HeapEntry last = heap.back();
        heap.pop_back();
        pos[slot] = -1;
        if (at < heap.size()) {
            heap[at]       = last;
            pos[last.slot] = static_cast<std::int32_t>(at);
            sift_up(heap, pos, at);
            sift_down(heap, pos, static_cast<std::size_t>(pos[last.slot]));
        }
    }
    if (!wanted) { return; }
    auto& heap     = classes_[c].heap[k];
    const double v = victim_value(slots_[slot].key, kind);
    if (pos[slot] >= 0) {
        const auto at = static_cast<std::size_t>(pos[slot]);
        if (heap[at].value == v) { return; }
        heap[at].value = v;
        sift_up(heap, pos, at);
        sift_down(heap, pos, static_cast<std::size_t>(pos[slot]));
        return;
    }
    heap_class_[k][slot] = c;
    heap.push_back({v, slot});
    sift_up(heap, pos, heap.size() - 1);
}

void HostTier::rescale_victims(int halvings) {
    for (ClassState& cs : classes_) {
        for (int k = 0; k < 2; ++k) {
            auto& heap = cs.heap[k];
            for (HeapEntry& e : heap) { e.value = std::ldexp(e.value, -halvings); }
            // Values that fell below the subnormals may now tie: restore the (value, slot) order.
            for (std::size_t i = heap.size() / 2; i-- > 0;) { sift_down(heap, heap_pos_[k], i); }
        }
    }
}

std::int32_t HostTier::peek_victim(std::uint8_t c, Victims kind) {
    auto& heap = classes_[c].heap[static_cast<int>(kind)];
    while (!heap.empty()) {
        const HeapEntry& top = heap.front();
        const double v       = victim_value(slots_[top.slot].key, kind);
        if (top.value != v) { // a resident used since it was placed: re-rank
            heap.front().value = v;
            sift_down(heap, heap_pos_[static_cast<int>(kind)], 0);
            continue;
        }
        return static_cast<std::int32_t>(top.slot);
    }
    return -1;
}

std::int32_t HostTier::victim_for_admission(std::uint8_t c, double& victim_score) {
    // A shadow goes without a comparison while the class has more than its floor: dropping it loses
    // nothing now. The floor's shadows are kept for a prefill's loan, which takes shadowed VRAM experts
    // first; below it a resident is ranked against the landing, and a shadow goes only when no resident can.
    if (shadows_spare(c)) {
        if (const std::int32_t shadow = peek_victim(c, Victims::kShadow); shadow >= 0) {
            victim_score = -1;
            return shadow;
        }
    }
    const std::int32_t resident = peek_victim(c, Victims::kResident);
    if (resident >= 0) {
        victim_score = lfu_.score(slots_[static_cast<std::uint32_t>(resident)].key);
        return resident;
    }
    victim_score = -1;
    return peek_victim(c, Victims::kShadow);
}

void HostTier::count_shadow(std::uint32_t slot, int delta) {
    std::uint32_t& n = classes_[slot_class_[slot]].shadows;
    n                = static_cast<std::uint32_t>(static_cast<int>(n) + delta);
    shadow_count_    = static_cast<std::size_t>(static_cast<std::int64_t>(shadow_count_) + delta);
}

std::int32_t HostTier::take_free(std::uint8_t c) {
    auto& free = classes_[c].free;
    if (free.empty()) { return -1; }
    const std::uint32_t s = free.back();
    free.pop_back();
    return static_cast<std::int32_t>(s);
}

void HostTier::set_resident(std::uint32_t slot, std::uint32_t key, SlotState state) {
    Slot& s  = slots_[slot];
    s.state  = state;
    s.key    = key;
    s.list   = SlotList::kNone;
    slot_of_[key] = static_cast<std::int32_t>(slot);
    ++resident_count_;
    if (state == SlotState::kShadow) { count_shadow(slot, 1); }
    mark_dirty(key);
    push_victim(slot);
}

void HostTier::evict(std::uint32_t slot) {
    Slot& s = slots_[slot];
    if (s.pins != 0 || (s.state != SlotState::kResident && s.state != SlotState::kShadow)) {
        throw std::logic_error("HostTier: evicting a pinned or empty slot");
    }
    slot_of_[s.key] = -1;
    mark_dirty(s.key);
    --resident_count_;
    if (s.state == SlotState::kShadow) { count_shadow(slot, -1); }
    if (s.state == SlotState::kResident) { ++losses_.residents_evicted; } // its only host copy
    s.state = SlotState::kFree;
    push_victim(slot); // invalidates its heap entries
}

void HostTier::replace_in_list(std::vector<std::uint32_t>& list, std::uint32_t old_slot, std::uint32_t new_slot) {
    const auto it = std::find(list.begin(), list.end(), old_slot);
    if (it == list.end()) { throw std::logic_error("HostTier: a landing slot is not in its list"); }
    *it = new_slot;
}

bool HostTier::prefill(std::uint32_t key, std::int32_t& slot) {
    if (host_copy(key) || held(key) || copying_[key]) { return false; }
    const std::int32_t s = take_free(key_class_[key]);
    if (s < 0) { return false; }
    set_resident(static_cast<std::uint32_t>(s), key, SlotState::kResident);
    slot = s;
    return true;
}

bool HostTier::prefill_shadow(std::uint32_t key, std::int32_t& slot) {
    if (host_copy(key) || held(key) || copying_[key]) { return false; }
    const std::int32_t s = take_free(key_class_[key]);
    if (s < 0) { return false; }
    set_resident(static_cast<std::uint32_t>(s), key, SlotState::kShadow);
    slot = s;
    return true;
}

bool HostTier::hold(std::uint32_t key) {
    if (host_copy(key) || held(key)) { return false; }
    held_[key] = 1;
    ++held_count_;
    return true;
}

void HostTier::seed_uses(std::span<const std::uint32_t> counts, std::uint32_t cap) {
    if (counts.size() != config_.num_keys) { throw std::invalid_argument("HostTier: the seed counts have another key count"); }
    for (std::uint32_t key = 0; key < config_.num_keys; ++key) {
        if (counts[key] != 0) {
            lfu_.use(key, static_cast<double>(std::min(counts[key], cap)));
            rerank_shadow(key);
        }
    }
    // Residents' heap entries carry their old values; peek_victim re-ranks them when it meets them.
}

void HostTier::rerank_shadow(std::uint32_t key) {
    // A use raises a shadow's rank (shadows leave most-used first), and peek_victim re-ranks lazily only entries that
    // fell: the shadow's entry moves now.
    const std::int32_t slot = slot_of_[key];
    if (slot >= 0 && heap_pos_[static_cast<int>(Victims::kShadow)][static_cast<std::uint32_t>(slot)] >= 0) {
        const auto s = static_cast<std::uint32_t>(slot);
        place_victim(Victims::kShadow, s, slot_class_[s], true);
    }
}

std::uint32_t HostTier::release_shadows() {
    std::uint32_t released = 0;
    for (std::uint32_t s = 0; s < slots_.size(); ++s) {
        const Slot& slot = slots_[s];
        if (slot.state != SlotState::kShadow || slot.pins != 0 || slot.list != SlotList::kNone) { continue; }
        evict(s);
        classes_[slot_class_[s]].free.push_back(s);
        ++released;
    }
    return released;
}

std::vector<std::uint32_t> HostTier::lend(std::span<const std::uint32_t> per_class) {
    if (per_class.size() != classes_.size()) { throw std::invalid_argument("HostTier: a loan names every class"); }
    std::vector<std::uint32_t> out;
    for (std::size_t c = 0; c < classes_.size(); ++c) {
        const auto cls     = static_cast<std::uint8_t>(c);
        const bool floored = classes_[c].shadow_floor > 0;
        for (std::uint32_t i = 0; i < per_class[c]; ++i) {
            std::int32_t s = take_free(cls);
            if (s < 0) {
                // With a floor (the exclusive layout) a loan evicts no expert RAM alone holds, and none of
                // the shadows the floor keeps for a prefill's loan.
                s = !floored || shadows_spare(cls) ? peek_victim(cls, Victims::kShadow) : -1;
                if (s < 0 && !floored) { s = peek_victim(cls, Victims::kResident); }
                if (s < 0) { break; }
                evict(static_cast<std::uint32_t>(s));
            }
            slots_[static_cast<std::uint32_t>(s)].list = SlotList::kLent;
            lent_.push_back(static_cast<std::uint32_t>(s));
            out.push_back(static_cast<std::uint32_t>(s));
        }
    }
    return out;
}

void HostTier::give_back() {
    for (const std::uint32_t s : lent_) {
        slots_[s].list = SlotList::kNone;
        classes_[slot_class_[s]].free.push_back(s);
    }
    lent_.clear();
}

void HostTier::mark_used(std::uint32_t slot) {
    if (!(slots_[slot].pins & kPinUse)) {
        slots_[slot].pins |= kPinUse;
        used_.push_back(slot);
        push_victim(slot);
    }
}

void HostTier::begin_round(std::uint32_t allowance) {
    if (!copies_.empty()) { throw std::logic_error("HostTier: admissions are still waiting for their copies"); }
    allowance_ = allowance;
    for (const auto s : used_) {
        slots_[s].pins &= static_cast<std::uint8_t>(~kPinUse);
        push_victim(s);
    }
    used_.clear();
    for (const auto s : ring_) { slots_[s].serial = next_serial_++; }
}

void HostTier::stream_pin(std::uint32_t key) {
    const std::int32_t s = slot_of_[key];
    if (s < 0) { return; }
    const auto slot = static_cast<std::uint32_t>(s);
    if (!(slots_[slot].pins & kPinStream)) {
        slots_[slot].pins |= kPinStream;
        streamed_.push_back(slot);
        push_victim(slot);
    }
}

void HostTier::release_stream_pins() {
    for (const auto s : streamed_) {
        slots_[s].pins &= static_cast<std::uint8_t>(~kPinStream);
        push_victim(s);
    }
    streamed_.clear();
}

void HostTier::record_uses(double dt, std::span<const std::uint32_t> keys, double weight) {
    if (const int halvings = lfu_.advance(dt); halvings > 0) { rescale_victims(halvings); }
    for (const auto k : keys) {
        lfu_.use(k, weight);
        rerank_shadow(k);
    }
}

std::size_t HostTier::admit(std::span<const Landing> landings, bool prefetch) {
    std::size_t admitted = 0;
    auto& list           = prefetch ? prefetch_ : ring_;
    const SlotList tag   = prefetch ? SlotList::kPrefetch : SlotList::kRing;
    for (const Landing& l : landings) {
        if (l.slot >= slots_.size()) { throw std::logic_error("HostTier: landing slot out of range"); }
        Slot& s = slots_[l.slot];
        if (s.list != tag || s.serial != l.serial || (s.pins & (kPinPrefetch | kPinCopy)) || host_copy(l.key) ||
            held(l.key) || copying_[l.key]) {
            continue;
        }
        const std::uint8_t c = key_class_[l.key];
        std::int32_t replacement = take_free(c);
        if (replacement < 0) {
            double victim_score       = 0;
            const std::int32_t victim = victim_for_admission(c, victim_score);
            if (victim < 0) { continue; }
            const double score = lfu_.score(l.key);
            if (victim_score >= 0 && !(prefetch ? score >= config_.prefetch_margin * victim_score : score > victim_score)) {
                continue;
            }
            replacement = victim;
            evict(static_cast<std::uint32_t>(replacement));
        }
        const auto r = static_cast<std::uint32_t>(replacement);
        if (classes_[c].transit_pitch) {
            // The landing's slot joins the class's residents; the replacement takes its place in the list.
            replace_in_list(list, l.slot, r);
            Slot& rs  = slots_[r];
            rs.list   = tag;
            rs.state  = SlotState::kFree;
            rs.serial = next_serial_++; // never matches a landing reported for the old occupant
            slot_class_[r]      = kTransit;
            slot_class_[l.slot] = c;
            const std::uint8_t use = s.pins & kPinUse;
            s.pins = use;
            set_resident(l.slot, l.key, SlotState::kResident);
        } else {
            // The record moves into a slot of its class's size: the caller copies it before the next round.
            Slot& t  = slots_[r];
            t.state  = SlotState::kLanding;
            t.key    = l.key;
            t.list   = SlotList::kNone;
            t.pins  |= kPinCopy;
            s.pins  |= kPinCopy;
            copying_[l.key] = 1;
            copies_.push_back({l.slot, r, l.key});
            push_victim(r);
        }
        ++admitted;
    }
    return admitted;
}

void HostTier::finish_copies() {
    for (const AdmissionCopy& a : copies_) {
        slots_[a.from].pins &= static_cast<std::uint8_t>(~kPinCopy);
        slots_[a.to].pins &= static_cast<std::uint8_t>(~kPinCopy);
        copying_[a.key] = 0;
        set_resident(a.to, a.key, SlotState::kResident);
    }
    copies_.clear();
}

void HostTier::replenish_demotion_list() {
    for (std::size_t c = 0; c < classes_.size(); ++c) {
        ClassState& cs       = classes_[c];
        const auto cls       = static_cast<std::uint8_t>(c);
        while (cs.demotion.size() < cs.demotion_target) {
            std::int32_t slot = take_free(cls);
            if (slot < 0) {
                double ignored = 0;
                slot           = victim_for_admission(cls, ignored);
                if (slot < 0) { break; }
                evict(static_cast<std::uint32_t>(slot));
            }
            slots_[static_cast<std::uint32_t>(slot)].list = SlotList::kDemotion;
            cs.demotion.push_back(static_cast<std::uint32_t>(slot));
        }
    }
}

void HostTier::queued(std::uint32_t key) {
    const std::int32_t s = slot_of_[key];
    if (s < 0 && held(key)) { return; } // its record never moves
    if (s < 0) { throw std::logic_error("HostTier: a key queued for VRAM has no host copy"); }
    slots_[static_cast<std::uint32_t>(s)].pins |= kPinQueue;
    push_victim(static_cast<std::uint32_t>(s));
}

void HostTier::unqueued(std::uint32_t key) {
    const std::int32_t s = slot_of_[key];
    if (s < 0) { return; }
    slots_[static_cast<std::uint32_t>(s)].pins &= static_cast<std::uint8_t>(~kPinQueue);
    push_victim(static_cast<std::uint32_t>(s));
}

void HostTier::promotion_issued(std::uint32_t key) {
    const std::int32_t s = slot_of_[key];
    if (s < 0 && held(key)) { return; }
    if (s < 0) { throw std::logic_error("HostTier: a promotion without a host copy"); }
    Slot& slot = slots_[static_cast<std::uint32_t>(s)];
    slot.pins  = static_cast<std::uint8_t>((slot.pins & ~kPinQueue) | kPinH2D);
    push_victim(static_cast<std::uint32_t>(s));
}

void HostTier::promotion_completed(std::uint32_t key, bool published) {
    const std::int32_t s = slot_of_[key];
    if (s < 0 && held(key)) { return; }
    if (s < 0) { throw std::logic_error("HostTier: a completed promotion without a host copy"); }
    Slot& slot = slots_[static_cast<std::uint32_t>(s)];
    slot.pins &= static_cast<std::uint8_t>(~kPinH2D);
    if (published && slot.state == SlotState::kResident) {
        slot.state = SlotState::kShadow;
        count_shadow(static_cast<std::uint32_t>(s), 1);
    }
    push_victim(static_cast<std::uint32_t>(s));
}

bool HostTier::demotion_worthy(std::uint32_t key) {
    const std::uint8_t c = key_class_[key];
    if (!classes_[c].free.empty() || (shadows_spare(c) && peek_victim(c, Victims::kShadow) >= 0)) { return true; }
    const std::int32_t victim = peek_victim(c, Victims::kResident);
    if (victim < 0) { return peek_victim(c, Victims::kShadow) >= 0; }
    return lfu_.score(key) >= config_.demotion_margin * lfu_.score(slots_[static_cast<std::uint32_t>(victim)].key);
}

bool HostTier::allow_evict(std::uint32_t key) {
    if (host_copy(key) || held(key) || !demotion_worthy(key)) { return true; }
    const ClassState& cs = classes_[key_class_[key]];
    if (allowance_ == 0 || (cs.demotion.empty() && cs.free.empty())) { return false; } // T6: the admission waits
    if (allowance_ != std::numeric_limits<std::uint32_t>::max()) { --allowance_; }
    return true;
}

VramEviction HostTier::vram_evicted(std::uint32_t key, bool published, std::uint32_t& slot) {
    if (const std::int32_t s = slot_of_[key]; s >= 0) { // T3
        Slot& kept = slots_[static_cast<std::uint32_t>(s)];
        if (kept.state == SlotState::kShadow) {
            kept.state = SlotState::kResident;
            count_shadow(static_cast<std::uint32_t>(s), -1);
        }
        push_victim(static_cast<std::uint32_t>(s));
        return VramEviction::kKeptInRam;
    }
    if (held(key)) { return VramEviction::kKeptInRam; } // T3
    if (!published) { // T5
        ++losses_.unpublished;
        return VramEviction::kDropped;
    }
    ClassState& cs = classes_[key_class_[key]];
    if (cs.demotion.empty() && cs.free.empty()) { // T5
        ++losses_.no_slot;
        return VramEviction::kDropped;
    }
    if (!demotion_worthy(key)) { // T5
        ++losses_.unworthy;
        return VramEviction::kDropped;
    }
    // T4: the class's demotion list, else one of its free slots (a list of a few slots, per record size, runs out
    // between boundaries where one list of 32 did not).
    if (!cs.demotion.empty()) {
        slot = cs.demotion.back();
        cs.demotion.pop_back();
    } else {
        slot = static_cast<std::uint32_t>(take_free(key_class_[key]));
    }
    Slot& target = slots_[slot];
    target.list  = SlotList::kNone;
    target.state = SlotState::kLanding;
    target.key   = key;
    target.pins |= kPinD2H;
    push_victim(slot);
    return VramEviction::kDemote;
}

void HostTier::demotion_completed(std::uint32_t key, std::uint32_t slot, bool in_vram) {
    Slot& s = slots_[slot];
    if (s.state != SlotState::kLanding || s.key != key || !(s.pins & kPinD2H)) {
        throw std::logic_error("HostTier: a demotion completed into a slot that was not its target");
    }
    s.pins &= static_cast<std::uint8_t>(~kPinD2H);
    if (host_copy(key)) { // landed meanwhile: the copy is redundant
        s.state = SlotState::kFree;
        classes_[slot_class_[slot]].free.push_back(slot);
        push_victim(slot);
        return;
    }
    set_resident(slot, key, in_vram ? SlotState::kShadow : SlotState::kResident); // T7
}

void HostTier::demotion_failed(std::uint32_t key, std::uint32_t slot) {
    Slot& s = slots_[slot];
    if (s.state != SlotState::kLanding || s.key != key || !(s.pins & kPinD2H)) {
        throw std::logic_error("HostTier: a failed demotion names a slot that was not its target");
    }
    s.pins &= static_cast<std::uint8_t>(~kPinD2H);
    s.state = SlotState::kFree;
    classes_[slot_class_[slot]].free.push_back(slot);
    push_victim(slot);
}

void HostTier::check() const {
    const auto fail = [](const std::string& what) { throw std::logic_error("HostTier audit: " + what); };
    std::vector<std::uint8_t> placed(slots_.size(), 0);
    const auto place = [&](std::uint32_t s, const char* where) {
        if (s >= slots_.size() || placed[s]++) { fail(std::string("slot placed twice or out of range in ") + where); }
    };
    const auto copying_slot = [&](std::uint32_t s, bool from) {
        return std::any_of(copies_.begin(), copies_.end(), [&](const AdmissionCopy& a) { return (from ? a.from : a.to) == s; });
    };
    for (const auto s : ring_) {
        place(s, "ring");
        if (slots_[s].list != SlotList::kRing || slots_[s].state != SlotState::kFree || slot_class_[s] != kTransit) {
            fail("ring slot state");
        }
        if ((slots_[s].pins & kPinCopy) && !copying_slot(s, true)) { fail("a ring slot's copy pin without its copy"); }
    }
    for (const auto s : prefetch_) {
        place(s, "prefetch");
        if (slots_[s].list != SlotList::kPrefetch || slots_[s].state != SlotState::kFree || slot_class_[s] != kTransit) {
            fail("prefetch slot state");
        }
    }
    std::vector<std::size_t> class_slots(classes_.size(), 0);
    for (std::uint32_t s = 0; s < slots_.size(); ++s) {
        if (slot_class_[s] != kTransit) {
            if (slot_class_[s] >= classes_.size()) { fail("slot class out of range"); }
            ++class_slots[slot_class_[s]];
        }
    }
    const std::size_t transit = std::count(slot_class_.begin(), slot_class_.end(), kTransit);
    if (transit != ring_.size() + prefetch_.size()) { fail("transit slots outside the ring and the prefetch list"); }
    for (std::size_t c = 0; c < classes_.size(); ++c) {
        const ClassState& cs = classes_[c];
        const std::uint32_t expected =
            config_.classes.empty() ? config_.slots - config_.ring - config_.prefetch : config_.classes[c].slots;
        if (class_slots[c] != expected) { fail("class " + std::to_string(c) + " gained or lost slots"); }
        for (const auto s : cs.demotion) {
            place(s, "demotion");
            if (slots_[s].list != SlotList::kDemotion || slots_[s].state != SlotState::kFree || slots_[s].pins != 0 ||
                slot_class_[s] != c) {
                fail("demotion slot state");
            }
        }
        for (const auto s : cs.free) {
            place(s, "free");
            if (slots_[s].list != SlotList::kNone || slots_[s].state != SlotState::kFree || slot_class_[s] != c) {
                fail("free slot state");
            }
        }
    }
    for (const auto s : lent_) {
        place(s, "lent");
        if (slots_[s].list != SlotList::kLent || slots_[s].state != SlotState::kFree || slots_[s].pins != 0 ||
            slot_class_[s] == kTransit) {
            fail("lent slot state");
        }
    }
    std::size_t residents = 0, shadows = 0;
    std::vector<std::uint32_t> class_shadows(classes_.size(), 0);
    for (std::uint32_t s = 0; s < slots_.size(); ++s) {
        const Slot& slot = slots_[s];
        if (slot.state == SlotState::kResident || slot.state == SlotState::kShadow) {
            place(s, "residents");
            ++residents;
            if (slot.state == SlotState::kShadow) {
                ++shadows;
                if (slot_class_[s] != kTransit) { ++class_shadows[slot_class_[s]]; }
            }
            if (slot.list != SlotList::kNone || slot_of_[slot.key] != static_cast<std::int32_t>(s)) { fail("resident map"); }
            if (slot_class_[s] != key_class_[slot.key]) { fail("a resident in another class's slot"); }
            if (slot.pins & kPinCopy) { fail("a resident with a copy pin"); }
        } else if (slot.state == SlotState::kLanding) {
            place(s, "landings in flight");
            const bool demotion = (slot.pins & kPinD2H) != 0, copy = (slot.pins & kPinCopy) != 0;
            if (demotion == copy || slot_of_[slot.key] >= 0) { fail("landing target"); }
            if (copy && (!copying_[slot.key] || !copying_slot(s, false))) { fail("a copy target without its copy"); }
            if (slot_class_[s] != key_class_[slot.key]) { fail("a landing in another class's slot"); }
        }
    }
    for (std::uint32_t s = 0; s < slots_.size(); ++s) {
        if (!placed[s]) { fail("slot " + std::to_string(s) + " is nowhere"); }
    }
    std::size_t copying_keys = 0;
    for (const std::uint8_t k : copying_) { copying_keys += k != 0 ? 1 : 0; }
    if (copying_keys != copies_.size()) { fail("pending copies"); }
    std::size_t stream_pins = 0;
    for (const auto s : streamed_) {
        const Slot& slot = slots_[s];
        if (!(slot.pins & kPinStream) || (slot.state != SlotState::kResident && slot.state != SlotState::kShadow)) {
            fail("stream pin");
        }
    }
    for (const Slot& slot : slots_) { stream_pins += (slot.pins & kPinStream) ? 1 : 0; }
    if (stream_pins != streamed_.size()) { fail("stream pins"); }
    if (residents != resident_count_ || shadows != shadow_count_) { fail("counts"); }
    for (std::size_t c = 0; c < classes_.size(); ++c) {
        if (class_shadows[c] != classes_[c].shadows) { fail("class " + std::to_string(c) + " shadow count"); }
    }
    // The victim heaps: every eligible slot of the class once, in (value, slot) order, a shadow at its current value
    // and a resident at most at it.
    std::size_t heaped[2] = {0, 0};
    for (std::size_t c = 0; c < classes_.size(); ++c) {
        for (const auto kind : {Victims::kShadow, Victims::kResident}) {
            const int k      = static_cast<int>(kind);
            const auto& heap = classes_[c].heap[k];
            std::size_t eligible_slots = 0;
            for (std::uint32_t s = 0; s < slots_.size(); ++s) {
                eligible_slots += slot_class_[s] == c && eligible(slots_[s], kind) ? 1 : 0;
            }
            if (heap.size() != eligible_slots) { fail("class " + std::to_string(c) + " victim heap size"); }
            heaped[k] += heap.size();
            for (std::size_t i = 0; i < heap.size(); ++i) {
                const HeapEntry& e = heap[i];
                if (heap_pos_[k][e.slot] != static_cast<std::int32_t>(i) || heap_class_[k][e.slot] != c ||
                    slot_class_[e.slot] != c || !eligible(slots_[e.slot], kind)) {
                    fail("victim heap entry");
                }
                if (i > 0 && heap_before(e, heap[(i - 1) / 2])) { fail("victim heap order"); }
                const double v = victim_value(slots_[e.slot].key, kind);
                if (kind == Victims::kShadow ? e.value != v : e.value > v) { fail("victim heap value"); }
            }
        }
    }
    for (int k = 0; k < 2; ++k) {
        const auto placed_slots = static_cast<std::size_t>(
            std::count_if(heap_pos_[k].begin(), heap_pos_[k].end(), [](std::int32_t p) { return p >= 0; }));
        if (placed_slots != heaped[k]) { fail("victim heap positions"); }
    }
    std::size_t held_keys = 0;
    for (std::uint32_t k = 0; k < slot_of_.size(); ++k) {
        const std::int32_t s = slot_of_[k];
        if (s >= 0 && slots_[static_cast<std::uint32_t>(s)].key != k) { fail("key map"); }
        if (held_[k] != 0 && s >= 0) { fail("a held key also has a slot"); }
        held_keys += held_[k] != 0 ? 1 : 0;
    }
    if (held_keys != held_count_) { fail("held count"); }
}

} // namespace infernix::runtime::expert_offload::expert_cache
