#pragma once

// Host side of the SSD expert tier's RAM level (docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.7, plan local/workdirs/fn/plans/memory-tiers.md §4.3, §4.7, §4.8): the decayed-LFU ranking
// of every expert key, the pinned host slots with their states, pins and load serials, the landing
// lists (demand ring, prefetch list, demotion list), and the admission and demotion gates. Pure
// host logic with no CUDA dependency, driven by the engine thread at round boundaries; the Program
// turns its decisions into copies and host-pointer table uploads.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace infernix::models::qwen4_exp::expert_cache {

// ---------------------------------------------------------------------------- decayed LFU

// score_k(t) = sum over uses of w * 2^-(t - t_use) / h. Each key keeps only the time-invariant
// value v_k = log2 score_k + t_k / h, so a use is O(1) and ranking by v equals ranking by the
// current score; v only increases (design §4.7).
class DecayedLfu {
public:
    DecayedLfu(std::uint32_t num_keys, double half_life);

    void advance(double dt) { now_ += dt; }
    void use(std::uint32_t key, double weight);

    [[nodiscard]] double now() const { return now_; }
    [[nodiscard]] double half_life() const { return half_life_; }
    // v_k; -infinity for a key never used.
    [[nodiscard]] double value(std::uint32_t key) const { return value_[key]; }
    [[nodiscard]] double score(std::uint32_t key) const;
    [[nodiscard]] std::uint32_t num_keys() const { return static_cast<std::uint32_t>(value_.size()); }

private:
    double half_life_;
    double now_ = 0;
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
};

// The list a non-resident slot belongs to.
enum class SlotList : std::uint8_t { kNone, kRing, kPrefetch, kDemotion };

// One landing the agent reports at a boundary: the record of `key` was read into `slot` while the
// slot carried `serial`.
struct Landing {
    std::uint32_t slot   = 0;
    std::uint64_t serial = 0;
    std::uint32_t key    = 0;
};

// What happens to a key's bytes when the VRAM policy evicts it.
enum class VramEviction : std::uint8_t {
    kKeptInRam, // T3: it had a host copy (a shadow becomes resident)
    kDemote,    // T4: copy the frame into `slot` (D2H), keep the frame held until it completes
    kDropped,   // T5: it becomes SSD-only
};

class HostTier {
public:
    struct Config {
        std::uint32_t num_keys = 0;
        std::uint32_t slots    = 0; // all pinned slots
        std::uint32_t ring     = 128;
        std::uint32_t prefetch = 16;
        std::uint32_t demotion = 32;
        double half_life       = 64;
        double prefetch_margin = 1.25; // a prefetch landing must outrank the victim by this factor
        double demotion_margin = 1.25; // a VRAM victim is demotion-worthy at this factor
        std::uint32_t decode_demotions = 32; // demotion allowance per decode boundary
    };

    explicit HostTier(const Config& config);

    // ---- queries
    [[nodiscard]] const Config& config() const { return config_; }
    [[nodiscard]] const DecayedLfu& lfu() const { return lfu_; }
    // The slot holding key's bytes (resident or shadow), or -1.
    [[nodiscard]] std::int32_t slot_of(std::uint32_t key) const { return slot_of_[key]; }
    [[nodiscard]] bool host_copy(std::uint32_t key) const { return slot_of_[key] >= 0; }
    [[nodiscard]] SlotState state(std::uint32_t slot) const { return slots_[slot].state; }
    [[nodiscard]] std::uint32_t key(std::uint32_t slot) const { return slots_[slot].key; }
    [[nodiscard]] std::uint8_t pins(std::uint32_t slot) const { return slots_[slot].pins; }
    [[nodiscard]] std::uint64_t serial(std::uint32_t slot) const { return slots_[slot].serial; }
    [[nodiscard]] SlotList list(std::uint32_t slot) const { return slots_[slot].list; }
    [[nodiscard]] std::span<const std::uint32_t> ring() const { return ring_; }
    [[nodiscard]] std::span<const std::uint32_t> prefetch_list() const { return prefetch_; }
    [[nodiscard]] std::size_t demotion_slots() const { return demotion_.size(); }
    [[nodiscard]] std::size_t resident_count() const { return resident_count_; }
    [[nodiscard]] std::size_t shadow_count() const { return shadow_count_; }
    [[nodiscard]] std::uint32_t demotion_allowance() const { return allowance_; }

    // Keys whose host slot changed since the last call (the device pointer table's dirty set).
    void take_dirty(std::vector<std::uint32_t>& keys);

    // ---- startup
    // Places `key`'s record into a free resident slot (the warm pre-fill); false when none is left
    // or the key already has a host copy. The caller writes the bytes before the first round.
    bool prefill(std::uint32_t key, std::int32_t& slot);
    [[nodiscard]] bool has_free_resident() const { return !free_resident_.empty(); }
    // Seeds the decayed LFU at the current time with a saved session's use counts, each capped at
    // `cap`, so the first admissions and demotions rank RAM residents by what that session used
    // instead of comparing all-zero scores (where the lowest slot, the best-ranked pre-fill, goes
    // first). The seeds decay like any use.
    void seed_uses(std::span<const std::uint32_t> counts, std::uint32_t cap);
    // Frees every unpinned shadow slot: its key stays readable in its VRAM frame, and a later VRAM
    // eviction demotes it (T4) or drops it (T5) like any key without a host copy. Returns how many.
    // The warm start uses it so RAM holds the keys ranked below the VRAM seeds, not copies of them.
    std::uint32_t release_shadows();

    // ---- round boundaries (engine thread, no round in flight)
    // A new round begins: every Use pin is released and each ring slot gets a fresh serial (an
    // earlier landing that was not admitted can no longer be). `allowance` is this boundary's
    // demotion allowance (decode_demotions at a decode boundary, UINT32_MAX for prefill and resize).
    void begin_round(std::uint32_t allowance);
    // The decayed-LFU clock and uses of one round: `dt` ticks, every routed key with its weight.
    void record_uses(double dt, std::span<const std::uint32_t> keys, double weight);
    // A landing read by the round just begun (its slot keeps Use until the next begin_round).
    void mark_used(std::uint32_t slot);
    // Admits the round's landings (T10): a demand landing when a slot is free, a shadow can go or
    // it outranks the lowest resident, a prefetch landing with prefetch_margin. A landing with a
    // stale serial or of a key that already has a host copy is discarded. The victim's slot (T12)
    // takes the landing's place in its list. Returns the number admitted.
    std::size_t admit(std::span<const Landing> landings, bool prefetch);
    // Keeps the demotion list full by evicting the lowest-ranked RAM residents (shadows first).
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
    // then can it be demoted. On kDemote, `slot` receives the D2H target (pinned D2H).
    VramEviction vram_evicted(std::uint32_t key, bool published, std::uint32_t& slot);
    // T7: the demotion copy into `slot` completed; `in_vram` tells whether key was re-admitted
    // into VRAM meanwhile (then the slot is its shadow).
    void demotion_completed(std::uint32_t key, std::uint32_t slot, bool in_vram);

    // Audit: recomputes every invariant (states, pins, lists, counts, maps); throws logic_error.
    void check() const;

private:
    struct Slot {
        SlotState state   = SlotState::kFree;
        std::uint8_t pins = 0;
        SlotList list     = SlotList::kNone;
        std::uint32_t key = 0;
        std::uint64_t serial = 0;
        std::uint32_t stamp  = 0; // victim-heap generation
    };
    struct HeapEntry {
        double value;
        std::uint32_t slot;
        std::uint32_t stamp;
    };
    enum class Victims : std::uint8_t { kShadow, kResident };

    [[nodiscard]] bool eligible(const Slot& s, Victims kind) const;
    void push_victim(std::uint32_t slot);
    // The lowest-ranked eligible victim of `kind` (left in its heap), or -1.
    std::int32_t peek_victim(Victims kind);
    // The victim a landing would replace: a shadow, else the lowest resident; -1 when none.
    std::int32_t victim_for_admission(double& victim_score);
    void set_resident(std::uint32_t slot, std::uint32_t key, SlotState state);
    // Removes the slot's key from RAM (T12) and returns it as a free slot.
    void evict(std::uint32_t slot);
    void mark_dirty(std::uint32_t key);
    void replace_in_list(std::vector<std::uint32_t>& list, std::uint32_t old_slot, std::uint32_t new_slot);

    Config config_;
    DecayedLfu lfu_;
    std::vector<Slot> slots_;
    std::vector<std::int32_t> slot_of_;
    std::vector<std::uint32_t> free_resident_; // free slots of the resident partition
    std::vector<std::uint32_t> ring_, prefetch_, demotion_;
    std::vector<std::uint32_t> used_; // slots holding a Use pin
    std::vector<std::uint32_t> streamed_; // slots holding a Stream pin
    std::vector<HeapEntry> heap_[2];
    std::vector<std::uint8_t> dirty_mark_;
    std::vector<std::uint32_t> dirty_;
    std::uint64_t next_serial_ = 1;
    std::uint32_t allowance_   = 0;
    std::size_t resident_count_ = 0, shadow_count_ = 0;
};

} // namespace infernix::models::qwen4_exp::expert_cache
