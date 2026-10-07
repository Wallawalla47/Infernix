#include "models/qwen4_exp/program/expert_cache/host_tier.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace infernix::models::qwen4_exp::expert_cache {

namespace {

constexpr double kNever = -std::numeric_limits<double>::infinity();

// Min-heap order on (value, slot): the lowest value first, ties to the lower slot.
bool heap_after(double va, std::uint32_t sa, double vb, std::uint32_t sb) {
    return va > vb || (va == vb && sa > sb);
}

} // namespace

// ---------------------------------------------------------------------------- DecayedLfu

DecayedLfu::DecayedLfu(std::uint32_t num_keys, double half_life) : half_life_(half_life), value_(num_keys, kNever) {
    if (!(half_life > 0)) { throw std::invalid_argument("DecayedLfu: the half-life must be positive"); }
}

void DecayedLfu::use(std::uint32_t key, double weight) {
    const double t = now_ / half_life_;
    double& v      = value_[key];
    // v = log2(2^(v - t/h) + w) + t/h, written to stay exact for the first use.
    v = v == kNever ? std::log2(weight) + t : std::log2(std::exp2(v - t) + weight) + t;
}

double DecayedLfu::score(std::uint32_t key) const {
    const double v = value_[key];
    return v == kNever ? 0.0 : std::exp2(v - now_ / half_life_);
}

// ---------------------------------------------------------------------------- HostTier

HostTier::HostTier(const Config& config)
    : config_(config), lfu_(config.num_keys, config.half_life), slots_(config.slots),
      slot_of_(config.num_keys, -1), dirty_mark_(config.num_keys, 0) {
    const std::uint64_t lists = static_cast<std::uint64_t>(config.ring) + config.prefetch + config.demotion;
    if (config.num_keys == 0 || config.slots <= lists) {
        throw std::invalid_argument("HostTier: the slots must exceed the ring, prefetch and demotion lists");
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
    for (std::uint32_t i = 0; i < config.demotion; ++i, ++s) {
        slots_[s].list = SlotList::kDemotion;
        demotion_.push_back(s);
    }
    // Free resident slots are taken lowest first.
    for (std::uint32_t i = config.slots; i-- > s;) { free_resident_.push_back(i); }
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

bool HostTier::eligible(const Slot& s, Victims kind) const {
    return s.pins == 0 && s.list == SlotList::kNone &&
           s.state == (kind == Victims::kShadow ? SlotState::kShadow : SlotState::kResident);
}

void HostTier::push_victim(std::uint32_t slot) {
    Slot& s = slots_[slot];
    ++s.stamp;
    for (const auto kind : {Victims::kShadow, Victims::kResident}) {
        if (!eligible(s, kind)) { continue; }
        auto& heap = heap_[static_cast<int>(kind)];
        heap.push_back({lfu_.value(s.key), slot, s.stamp});
        std::push_heap(heap.begin(), heap.end(), [](const HeapEntry& a, const HeapEntry& b) {
            return heap_after(a.value, a.slot, b.value, b.slot);
        });
    }
}

std::int32_t HostTier::peek_victim(Victims kind) {
    auto& heap      = heap_[static_cast<int>(kind)];
    const auto less = [](const HeapEntry& a, const HeapEntry& b) { return heap_after(a.value, a.slot, b.value, b.slot); };
    while (!heap.empty()) {
        const HeapEntry top = heap.front();
        const Slot& s       = slots_[top.slot];
        if (top.stamp != s.stamp || !eligible(s, kind)) {
            std::pop_heap(heap.begin(), heap.end(), less);
            heap.pop_back();
            continue;
        }
        const double v = lfu_.value(s.key);
        if (top.value != v) { // used since it was pushed: re-rank
            std::pop_heap(heap.begin(), heap.end(), less);
            heap.back().value = v;
            std::push_heap(heap.begin(), heap.end(), less);
            continue;
        }
        return static_cast<std::int32_t>(top.slot);
    }
    return -1;
}

std::int32_t HostTier::victim_for_admission(double& victim_score) {
    if (const std::int32_t shadow = peek_victim(Victims::kShadow); shadow >= 0) {
        victim_score = -1; // a shadow goes without a comparison: dropping it loses nothing now
        return shadow;
    }
    const std::int32_t resident = peek_victim(Victims::kResident);
    victim_score                = resident >= 0 ? lfu_.score(slots_[static_cast<std::uint32_t>(resident)].key) : 0;
    return resident;
}

void HostTier::set_resident(std::uint32_t slot, std::uint32_t key, SlotState state) {
    Slot& s  = slots_[slot];
    s.state  = state;
    s.key    = key;
    s.list   = SlotList::kNone;
    slot_of_[key] = static_cast<std::int32_t>(slot);
    ++resident_count_;
    if (state == SlotState::kShadow) { ++shadow_count_; }
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
    if (s.state == SlotState::kShadow) { --shadow_count_; }
    s.state = SlotState::kFree;
    push_victim(slot); // invalidates its heap entries
}

void HostTier::replace_in_list(std::vector<std::uint32_t>& list, std::uint32_t old_slot, std::uint32_t new_slot) {
    const auto it = std::find(list.begin(), list.end(), old_slot);
    if (it == list.end()) { throw std::logic_error("HostTier: a landing slot is not in its list"); }
    *it = new_slot;
}

bool HostTier::prefill(std::uint32_t key, std::int32_t& slot) {
    if (host_copy(key) || free_resident_.empty()) { return false; }
    const std::uint32_t s = free_resident_.back();
    free_resident_.pop_back();
    set_resident(s, key, SlotState::kResident);
    slot = static_cast<std::int32_t>(s);
    return true;
}

void HostTier::mark_used(std::uint32_t slot) {
    if (!(slots_[slot].pins & kPinUse)) {
        slots_[slot].pins |= kPinUse;
        used_.push_back(slot);
        push_victim(slot);
    }
}

void HostTier::begin_round(std::uint32_t allowance) {
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
    lfu_.advance(dt);
    for (const auto k : keys) { lfu_.use(k, weight); }
}

std::size_t HostTier::admit(std::span<const Landing> landings, bool prefetch) {
    std::size_t admitted = 0;
    auto& list           = prefetch ? prefetch_ : ring_;
    const SlotList tag   = prefetch ? SlotList::kPrefetch : SlotList::kRing;
    for (const Landing& l : landings) {
        if (l.slot >= slots_.size()) { throw std::logic_error("HostTier: landing slot out of range"); }
        Slot& s = slots_[l.slot];
        if (s.list != tag || s.serial != l.serial || (s.pins & kPinPrefetch) || host_copy(l.key)) { continue; }
        std::uint32_t replacement;
        if (!free_resident_.empty()) {
            replacement = free_resident_.back();
            free_resident_.pop_back();
        } else {
            double victim_score = 0;
            const std::int32_t victim = victim_for_admission(victim_score);
            if (victim < 0) { continue; }
            const double score = lfu_.score(l.key);
            if (victim_score >= 0 && !(prefetch ? score >= config_.prefetch_margin * victim_score : score > victim_score)) {
                continue;
            }
            replacement = static_cast<std::uint32_t>(victim);
            evict(replacement);
        }
        // The landing's slot joins the residents; the replacement takes its place in the list.
        replace_in_list(list, l.slot, replacement);
        Slot& r  = slots_[replacement];
        r.list   = tag;
        r.state  = SlotState::kFree;
        r.serial = next_serial_++; // never matches a landing reported for the old occupant
        const std::uint8_t use = s.pins & kPinUse;
        s.pins = use;
        set_resident(l.slot, l.key, SlotState::kResident);
        ++admitted;
    }
    return admitted;
}

void HostTier::replenish_demotion_list() {
    while (demotion_.size() < config_.demotion) {
        std::uint32_t slot;
        if (!free_resident_.empty()) {
            slot = free_resident_.back();
            free_resident_.pop_back();
        } else {
            double ignored = 0;
            const std::int32_t victim = victim_for_admission(ignored);
            if (victim < 0) { return; }
            slot = static_cast<std::uint32_t>(victim);
            evict(slot);
        }
        slots_[slot].list = SlotList::kDemotion;
        demotion_.push_back(slot);
    }
}

void HostTier::queued(std::uint32_t key) {
    const std::int32_t s = slot_of_[key];
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
    if (s < 0) { throw std::logic_error("HostTier: a promotion without a host copy"); }
    Slot& slot = slots_[static_cast<std::uint32_t>(s)];
    slot.pins  = static_cast<std::uint8_t>((slot.pins & ~kPinQueue) | kPinH2D);
    push_victim(static_cast<std::uint32_t>(s));
}

void HostTier::promotion_completed(std::uint32_t key, bool published) {
    const std::int32_t s = slot_of_[key];
    if (s < 0) { throw std::logic_error("HostTier: a completed promotion without a host copy"); }
    Slot& slot = slots_[static_cast<std::uint32_t>(s)];
    slot.pins &= static_cast<std::uint8_t>(~kPinH2D);
    if (published && slot.state == SlotState::kResident) {
        slot.state = SlotState::kShadow;
        ++shadow_count_;
    }
    push_victim(static_cast<std::uint32_t>(s));
}

bool HostTier::demotion_worthy(std::uint32_t key) {
    if (!free_resident_.empty() || peek_victim(Victims::kShadow) >= 0) { return true; }
    const std::int32_t victim = peek_victim(Victims::kResident);
    if (victim < 0) { return false; }
    return lfu_.score(key) >= config_.demotion_margin * lfu_.score(slots_[static_cast<std::uint32_t>(victim)].key);
}

bool HostTier::allow_evict(std::uint32_t key) {
    if (host_copy(key) || !demotion_worthy(key)) { return true; }
    if (allowance_ == 0 || demotion_.empty()) { return false; } // T6: the admission waits
    if (allowance_ != std::numeric_limits<std::uint32_t>::max()) { --allowance_; }
    return true;
}

VramEviction HostTier::vram_evicted(std::uint32_t key, bool published, std::uint32_t& slot) {
    if (const std::int32_t s = slot_of_[key]; s >= 0) { // T3
        Slot& held = slots_[static_cast<std::uint32_t>(s)];
        if (held.state == SlotState::kShadow) {
            held.state = SlotState::kResident;
            --shadow_count_;
        }
        push_victim(static_cast<std::uint32_t>(s));
        return VramEviction::kKeptInRam;
    }
    if (!published || demotion_.empty() || !demotion_worthy(key)) { return VramEviction::kDropped; } // T5
    slot = demotion_.back(); // T4
    demotion_.pop_back();
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
        free_resident_.push_back(slot);
        push_victim(slot);
        return;
    }
    set_resident(slot, key, in_vram ? SlotState::kShadow : SlotState::kResident); // T7
}

void HostTier::check() const {
    const auto fail = [](const std::string& what) { throw std::logic_error("HostTier audit: " + what); };
    std::vector<std::uint8_t> placed(slots_.size(), 0);
    const auto place = [&](std::uint32_t s, const char* where) {
        if (s >= slots_.size() || placed[s]++) { fail(std::string("slot placed twice or out of range in ") + where); }
    };
    for (const auto s : ring_) {
        place(s, "ring");
        if (slots_[s].list != SlotList::kRing || slots_[s].state != SlotState::kFree) { fail("ring slot state"); }
    }
    for (const auto s : prefetch_) {
        place(s, "prefetch");
        if (slots_[s].list != SlotList::kPrefetch || slots_[s].state != SlotState::kFree) { fail("prefetch slot state"); }
    }
    for (const auto s : demotion_) {
        place(s, "demotion");
        if (slots_[s].list != SlotList::kDemotion || slots_[s].state != SlotState::kFree || slots_[s].pins != 0) {
            fail("demotion slot state");
        }
    }
    for (const auto s : free_resident_) {
        place(s, "free");
        if (slots_[s].list != SlotList::kNone || slots_[s].state != SlotState::kFree) { fail("free slot state"); }
    }
    std::size_t residents = 0, shadows = 0;
    for (std::uint32_t s = 0; s < slots_.size(); ++s) {
        const Slot& slot = slots_[s];
        if (slot.state == SlotState::kResident || slot.state == SlotState::kShadow) {
            place(s, "residents");
            ++residents;
            shadows += slot.state == SlotState::kShadow ? 1 : 0;
            if (slot.list != SlotList::kNone || slot_of_[slot.key] != static_cast<std::int32_t>(s)) { fail("resident map"); }
        } else if (slot.state == SlotState::kLanding) {
            place(s, "demotions in flight");
            if (!(slot.pins & kPinD2H) || slot_of_[slot.key] >= 0) { fail("demotion target"); }
        }
    }
    for (std::uint32_t s = 0; s < slots_.size(); ++s) {
        if (!placed[s]) { fail("slot " + std::to_string(s) + " is nowhere"); }
    }
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
    for (std::uint32_t k = 0; k < slot_of_.size(); ++k) {
        const std::int32_t s = slot_of_[k];
        if (s >= 0 && slots_[static_cast<std::uint32_t>(s)].key != k) { fail("key map"); }
    }
}

} // namespace infernix::models::qwen4_exp::expert_cache
