#pragma once

// Host side of the SSD expert tier's RAM level (docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.7, plan local/workdirs/fn/plans/memory-tiers.md §4.3, §4.7, §4.8): the decayed-LFU ranking
// of every expert key, the pinned host slots with their states, pins and load serials, the landing
// lists (demand ring, prefetch list, demotion lists), the admission and demotion gates, and the keys
// held outside the slots (resident pageable records past the page-locking limit). Pure
// host logic with no CUDA dependency, driven by the engine thread at round boundaries; the Program
// turns its decisions into copies and host-pointer table uploads.
//
// Size classes. A GGUF import's layers have records of up to seven sizes (IQ3_S: 1.49-2.54 MiB), so
// slots of one size padded every record to the largest (29 % of the tier's RAM on IQ3_S). The slots
// are partitioned into classes, one per record size: a class's slots hold only its keys' records,
// and each class has its own free slots, victim ranking and demotion list, so admissions, demotions
// and loans exchange records of one size. The transit slots (the ring and the prefetch list) hold
// any key's record: a landing there is admitted by swapping its slot into the class when the class's
// slots have the transit size (one class of the largest record, as an NVFP4 or block-FP8 bank has),
// and otherwise by a copy into a class slot that the caller makes at the boundary (copies(),
// finish_copies()).

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace infernix::runtime::expert_offload::expert_cache {

// ---------------------------------------------------------------------------- decayed LFU

// score_k(t) = sum over uses of w * 2^-(t - t_use) / h. Each key keeps the time-invariant value
// v_k = sum over uses of w * 2^((t_use - base) / h), so a use is one multiply-add by the clock's
// growth 2^((t - base) / h) and ranking by v equals ranking by the current score; v only increases,
// except that a rebase divides every v by the same power of two (exactly) once the clock is kRebase
// half-lives past the base (design §4.7). The log2 form it replaced took two transcendentals a use,
// ~100 us of a decode round's host time (IQ3_S struct, 2026-10-29).
class DecayedLfu {
public:
    DecayedLfu(std::uint32_t num_keys, double half_life);

    // Advances the clock. Returns the halvings a rebase applied to every value (almost always 0); a
    // holder of copies of values divides them alike (std::ldexp(v, -halvings)).
    int advance(double dt);
    void use(std::uint32_t key, double weight) { value_[key] += weight * growth_; }

    [[nodiscard]] double now() const { return now_; }
    [[nodiscard]] double half_life() const { return half_life_; }
    // v_k; 0 for a key never used.
    [[nodiscard]] double value(std::uint32_t key) const { return value_[key]; }
    [[nodiscard]] double score(std::uint32_t key) const { return value_[key] / growth_; }
    [[nodiscard]] std::uint32_t num_keys() const { return static_cast<std::uint32_t>(value_.size()); }

private:
    static constexpr double kRebase = 256;
    double half_life_;
    double now_ = 0, base_ = 0;
    double growth_ = 1; // 2^((now - base) / h)
    std::vector<double> value_;
};

// ---------------------------------------------------------------------------- host slots

enum class SlotState : std::uint8_t { kFree, kLanding, kResident, kShadow };

// A slot is never a victim and never rewritten while any pin is held (design §4.3).
enum SlotPin : std::uint8_t {
    kPinH2D      = 1U << 0, // an in-flight promotion reads it
    kPinQueue    = 1U << 1, // its key is queued for a VRAM load
    kPinD2H      = 1U << 2, // an in-flight demotion writes it
    kPinUse      = 1U << 3, // a landing read by the current round
    kPinPrefetch = 1U << 4, // a prefetch read in flight
    kPinStream   = 1U << 5, // a prefill expert stream copies its record
    kPinCopy     = 1U << 6, // an admission's copy reads (the transit slot) or writes (the class slot) it
};

// The list a non-resident slot belongs to.
enum class SlotList : std::uint8_t { kNone, kRing, kPrefetch, kDemotion, kLent };

// One landing the agent reports at a boundary: the record of `key` was read into `slot` while the
// slot carried `serial`.
struct Landing {
    std::uint32_t slot   = 0;
    std::uint64_t serial = 0;
    std::uint32_t key    = 0;
};

// An admission by copy: the caller copies key's record from the transit slot `from` into the class
// slot `to`, then calls finish_copies().
struct AdmissionCopy {
    std::uint32_t from = 0, to = 0, key = 0;
};

// What happens to a key's bytes when the VRAM policy evicts it.
enum class VramEviction : std::uint8_t {
    kKeptInRam, // T3: it had a host copy (a shadow becomes resident)
    kDemote,    // T4: copy the frame into `slot` (D2H), keep the frame held until it completes
    kDropped,   // T5: it becomes SSD-only
};

class HostTier {
public:
    // The class of the transit slots (the ring and the prefetch list).
    static constexpr std::uint8_t kTransit = 0xFF;

    struct Class {
        std::uint32_t slots    = 0;     // its slots, its demotion list included
        std::uint32_t demotion = 0;     // its demotion list
        bool transit_pitch     = false; // its slots have the transit slots' size (admission swaps, no copy)
        // Shadows kept for a prefill's loan (the exclusive layout): at or below this many, a shadow is no
        // longer the first victim, and a loan takes free slots and shadows above it only.
        std::uint32_t shadow_floor = 0;
    };

    struct Config {
        std::uint32_t num_keys = 0;
        std::uint32_t slots    = 0; // all slots: the ring, the prefetch list and every class's
        std::uint32_t ring     = 128;
        std::uint32_t prefetch = 16;
        std::uint32_t demotion = 32; // the demotion list of the one class (without `classes`)
        double half_life       = 64;
        double prefetch_margin = 1.25; // a prefetch landing must outrank the victim by this factor
        double demotion_margin = 1.25; // a VRAM victim is demotion-worthy at this factor
        std::uint32_t decode_demotions = 32; // demotion allowance per decode boundary
        std::uint32_t shadow_floor     = 0;  // the one class's (without `classes`)
        // Size classes: key_class[k] is key k's class, classes[c] its slots. Empty: one class of every
        // slot past the ring and the prefetch list, `demotion` of them in its list, of the transit size.
        std::vector<std::uint8_t> key_class;
        std::vector<Class> classes;
    };

    explicit HostTier(const Config& config);

    // ---- queries
    [[nodiscard]] const Config& config() const { return config_; }
    [[nodiscard]] const DecayedLfu& lfu() const { return lfu_; }
    [[nodiscard]] std::uint32_t class_count() const { return static_cast<std::uint32_t>(classes_.size()); }
    [[nodiscard]] std::uint8_t class_of(std::uint32_t key) const { return key_class_[key]; }
    // A slot's class, or kTransit for the ring and prefetch slots. A swap moves a transit-size
    // class's slot into the ring and the ring's slot into the class.
    [[nodiscard]] std::uint8_t slot_class(std::uint32_t slot) const { return slot_class_[slot]; }
    // The slot holding key's bytes (resident or shadow), or -1.
    [[nodiscard]] std::int32_t slot_of(std::uint32_t key) const { return slot_of_[key]; }
    [[nodiscard]] bool host_copy(std::uint32_t key) const { return slot_of_[key] >= 0; }
    // A key held in RAM outside the slots for good (the SSD tier's resident pageable records): it is
    // never admitted into a slot, a VRAM eviction keeps it (T3), and its promotions pin nothing.
    [[nodiscard]] bool held(std::uint32_t key) const { return held_[key] != 0; }
    [[nodiscard]] std::size_t held_count() const { return held_count_; }
    [[nodiscard]] SlotState state(std::uint32_t slot) const { return slots_[slot].state; }
    [[nodiscard]] std::uint32_t key(std::uint32_t slot) const { return slots_[slot].key; }
    [[nodiscard]] std::uint8_t pins(std::uint32_t slot) const { return slots_[slot].pins; }
    [[nodiscard]] std::uint64_t serial(std::uint32_t slot) const { return slots_[slot].serial; }
    [[nodiscard]] SlotList list(std::uint32_t slot) const { return slots_[slot].list; }
    [[nodiscard]] std::span<const std::uint32_t> ring() const { return ring_; }
    [[nodiscard]] std::span<const std::uint32_t> prefetch_list() const { return prefetch_; }
    // Slots in every class's demotion list.
    [[nodiscard]] std::size_t demotion_slots() const;
    [[nodiscard]] std::size_t resident_count() const { return resident_count_; }
    [[nodiscard]] std::size_t shadow_count() const { return shadow_count_; }
    [[nodiscard]] std::uint32_t demotion_allowance() const { return allowance_; }
    // Keys that left both tiers (SSD-only afterwards), by cause, since start: VRAM victims dropped (T5) because
    // their frame was not published (a loan's or a shrink's, or one evicted while loading), no demotion slot was
    // left, or they were not worth one; and RAM residents (their only host copy) evicted for another key.
    struct Losses {
        std::uint64_t unpublished = 0, no_slot = 0, unworthy = 0, residents_evicted = 0;
    };
    [[nodiscard]] const Losses& losses() const { return losses_; }

    // Keys whose host slot changed since the last call (the device pointer table's dirty set).
    void take_dirty(std::vector<std::uint32_t>& keys);

    // ---- startup
    // Places `key`'s record into a free resident slot of its class (the warm pre-fill); false when
    // its class has none left or the key already has a host copy. The caller writes the bytes
    // before the first round.
    bool prefill(std::uint32_t key, std::int32_t& slot);
    // The same for a key VRAM holds (the warm start's spare slots): the slot is its shadow, so its
    // eviction from VRAM (a lend's or a decode victim's) needs no copy back (T3).
    bool prefill_shadow(std::uint32_t key, std::int32_t& slot);
    // Whether any class has a free resident slot.
    [[nodiscard]] bool has_free_resident() const;
    // Marks `key` held outside the slots (see held); false when it has a slot copy or is held.
    bool hold(std::uint32_t key);
    // Seeds the decayed LFU at the current time with a saved session's use counts, each capped at
    // `cap`, so the first admissions and demotions rank RAM residents by what that session used
    // instead of comparing all-zero scores (where the lowest slot, the best-ranked pre-fill, goes
    // first). The seeds decay like any use.
    void seed_uses(std::span<const std::uint32_t> counts, std::uint32_t cap);
    // Frees every unpinned shadow slot: its key stays readable in its VRAM frame, and a later VRAM
    // eviction demotes it (T4) or drops it (T5) like any key without a host copy. Returns how many.
    // The warm start uses it so RAM holds the keys ranked below the VRAM seeds, not copies of them.
    std::uint32_t release_shadows();
    // ---- lending (engine thread, no round in flight)
    // Lends up to per_class[c] resident slots of each class c (free ones first, then the lowest-ranked
    // unpinned residents, shadows before residents, which leave RAM; a class with a shadow floor lends
    // free slots and shadows above the floor only): they hold no key until give_back. Returns them;
    // slot_class tells which keys each can hold.
    std::vector<std::uint32_t> lend(std::span<const std::uint32_t> per_class);
    // Every lent slot becomes a free resident slot again.
    void give_back();
    [[nodiscard]] std::size_t lent() const { return lent_.size(); }

    // ---- round boundaries (engine thread, no round in flight)
    // A new round begins: every Use pin is released and each ring slot gets a fresh serial (an
    // earlier landing that was not admitted can no longer be). `allowance` is this boundary's
    // demotion allowance (decode_demotions at a decode boundary, UINT32_MAX for prefill and resize).
    void begin_round(std::uint32_t allowance);
    // The decayed-LFU clock and uses of one round: `dt` ticks, every routed key with its weight.
    void record_uses(double dt, std::span<const std::uint32_t> keys, double weight);
    // A landing read by the round just begun (its slot keeps Use until the next begin_round).
    void mark_used(std::uint32_t slot);
    // Admits the round's landings (T10): a demand landing when its class has a free slot, a shadow
    // can go or it outranks the class's lowest resident, a prefetch landing with prefetch_margin. A
    // landing with a stale serial or of a key that already has a host copy is discarded. In a class
    // of the transit size the victim's slot (T12) takes the landing's place in its list; in another
    // the landing is admitted by a copy (copies()), complete once finish_copies() is called. Returns
    // the number admitted.
    std::size_t admit(std::span<const Landing> landings, bool prefetch);
    // The admissions waiting for their copies, and their completion (before the next round begins).
    [[nodiscard]] std::span<const AdmissionCopy> copies() const { return copies_; }
    void finish_copies();
    // Keeps each class's demotion list full by evicting its lowest-ranked RAM residents (shadows first).
    void replenish_demotion_list();

    // ---- prefetch
    // A prefetch read into the prefetch slot `slot` starts (Prefetch pin; returns the serial its
    // landing must carry) or ends.
    std::uint64_t prefetch_started(std::uint32_t slot) {
        slots_[slot].pins |= kPinPrefetch;
        return slots_[slot].serial = next_serial_++;
    }
    void prefetch_ended(std::uint32_t slot) { slots_[slot].pins &= static_cast<std::uint8_t>(~kPinPrefetch); }

    // ---- prefill expert stream
    // The stream will copy `key`'s host record (resident or shadow): its slot keeps the record until
    // release_stream_pins, which the engine calls at a boundary after the stream's copies landed. The
    // stream plans a chunk's copies at its start and enqueues them layer by layer, across boundaries
    // in a layer walk, so an admission must not evict or rewrite a planned record. A key without a
    // host copy is not planned and is ignored here.
    void stream_pin(std::uint32_t key);
    void release_stream_pins();

    // ---- VRAM interplay (design §4.8)
    void queued(std::uint32_t key);             // T8
    void unqueued(std::uint32_t key);           // T9
    void promotion_issued(std::uint32_t key);   // T1: Queue -> H2D
    // T2: the promotion's copy completed; when its frame was published the slot becomes the key's
    // shadow, otherwise (evicted from VRAM while loading) it stays resident.
    void promotion_completed(std::uint32_t key, bool published);
    // The eviction gate: may the VRAM policy evict `key` now? Consumes one unit of the demotion
    // allowance when the eviction will be a demotion.
    [[nodiscard]] bool allow_evict(std::uint32_t key);
    // Whether evicting `key` (no host copy) would be worth a demotion.
    [[nodiscard]] bool demotion_worthy(std::uint32_t key);
    // T3 / T4 / T5. `published` tells whether key's frame is published (its load completed); only
    // then can it be demoted. On kDemote, `slot` receives the D2H target (pinned D2H), a slot of the
    // key's class: from its demotion list, else a free one.
    VramEviction vram_evicted(std::uint32_t key, bool published, std::uint32_t& slot);
    // T7: the demotion copy into `slot` completed; `in_vram` tells whether key was re-admitted
    // into VRAM meanwhile (then the slot is its shadow).
    void demotion_completed(std::uint32_t key, std::uint32_t slot, bool in_vram);
    // A demotion's read failed: its slot is free again and the key keeps no host copy.
    void demotion_failed(std::uint32_t key, std::uint32_t slot);

    // Audit: recomputes every invariant (states, pins, lists, classes, counts, maps); throws logic_error.
    void check() const;

private:
    struct Slot {
        SlotState state   = SlotState::kFree;
        std::uint8_t pins = 0;
        SlotList list     = SlotList::kNone;
        std::uint32_t key = 0;
        std::uint64_t serial = 0;
    };
    // A victim heap holds each eligible slot of its class and kind once (heap_pos_), at the value it had when last
    // placed: a shadow's is placed again at each use of its key, which raises its rank (most-used first); a
    // resident's only falls behind its current value, which a use raises, so peek_victim re-ranks a stale top. The
    // heaps grew by an entry a use before (re-ranks pushed copies), 2,000 a decode round.
    struct HeapEntry {
        double value;
        std::uint32_t slot;
    };
    enum class Victims : std::uint8_t { kShadow, kResident };
    // One size class's slots: its free resident slots, demotion list and victim heaps.
    struct ClassState {
        std::vector<std::uint32_t> free; // free slots of the resident partition
        std::vector<std::uint32_t> demotion;
        std::vector<HeapEntry> heap[2];
        std::uint32_t demotion_target = 0;
        std::uint32_t shadow_floor    = 0;
        std::uint32_t shadows         = 0;
        bool transit_pitch            = false;
    };
    // Whether class c's shadows are its first victims (more of them than its floor).
    [[nodiscard]] bool shadows_spare(std::uint8_t c) const {
        return classes_[c].shadows > classes_[c].shadow_floor;
    }
    // Shadow counts follow a slot's state changes (resident <-> shadow, eviction).
    void count_shadow(std::uint32_t slot, int delta);

    [[nodiscard]] bool eligible(const Slot& s, Victims kind) const;
    // A key's rank in its class's victim heap of `kind` (the lowest goes first).
    [[nodiscard]] double victim_value(std::uint32_t key, Victims kind) const;
    // Re-ranks an eligible shadow's victim heap entry after its key's use.
    void rerank_shadow(std::uint32_t key);
    // Places the slot in, re-ranks it in or removes it from its class's victim heaps after a change of its state,
    // pins, list, class or value.
    void push_victim(std::uint32_t slot);
    void place_victim(Victims kind, std::uint32_t slot, std::uint8_t c, bool wanted);
    // Applies an LFU rebase (DecayedLfu::advance) to the heaps' values, and restores their order.
    void rescale_victims(int halvings);
    // The lowest-ranked eligible victim of `kind` in class c (left in its heap), or -1.
    std::int32_t peek_victim(std::uint8_t c, Victims kind);
    // The victim a landing of class c would replace: a shadow, else the lowest resident; -1 when none.
    std::int32_t victim_for_admission(std::uint8_t c, double& victim_score);
    // A free slot of class c taken from its free list, else -1.
    std::int32_t take_free(std::uint8_t c);
    void set_resident(std::uint32_t slot, std::uint32_t key, SlotState state);
    // Removes the slot's key from RAM (T12) and returns it as a free slot.
    void evict(std::uint32_t slot);
    void mark_dirty(std::uint32_t key);
    void replace_in_list(std::vector<std::uint32_t>& list, std::uint32_t old_slot, std::uint32_t new_slot);

    Config config_;
    DecayedLfu lfu_;
    std::vector<Slot> slots_;
    std::vector<std::uint8_t> slot_class_;
    std::vector<std::uint8_t> key_class_;
    std::vector<ClassState> classes_;
    std::vector<std::int32_t> heap_pos_[2];   // per kind and slot: its index in its victim heap, or -1
    std::vector<std::uint8_t> heap_class_[2]; // per kind and slot: the class whose heap holds it
    std::vector<std::int32_t> slot_of_;
    std::vector<std::uint8_t> held_;
    std::size_t held_count_ = 0;
    std::vector<std::uint32_t> ring_, prefetch_;
    std::vector<std::uint32_t> lent_; // lend(): slots out of the classes' resident partitions
    std::vector<std::uint32_t> used_; // slots holding a Use pin
    std::vector<std::uint32_t> streamed_; // slots holding a Stream pin
    std::vector<AdmissionCopy> copies_;  // admissions waiting for their copies
    std::vector<std::uint8_t> copying_;  // per key: an admission copy is pending
    std::vector<std::uint8_t> dirty_mark_;
    std::vector<std::uint32_t> dirty_;
    std::uint64_t next_serial_ = 1;
    std::uint32_t allowance_   = 0;
    std::size_t resident_count_ = 0, shadow_count_ = 0;
    Losses losses_;
};

} // namespace infernix::runtime::expert_offload::expert_cache
