#pragma once

// Host side of the Qwen3.8-Flash-Next expert cache (docs/maintainer/qwen3_8-flash-next-design.md
// §9): the LFRU policy, the device residency-entry encoding, and the frame pool with epoch-based
// reuse. Pure host logic with no CUDA dependency; the transfer agent drives it and turns its
// decisions into copy-stream commands.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp::expert_cache {

// ---------------------------------------------------------------------------- residency entries

enum class ResidencyState : std::uint32_t { kAbsent = 0, kLoading = 1, kReady = 2 };

// residency[layer * 512 + expert], one u32 in device memory (design §9.4):
//   bits 31-30 state, 29-24 generation (incremented on every state change), 23-0 frame.
struct ResidencyEntry {
    ResidencyState state = ResidencyState::kAbsent;
    std::uint32_t generation = 0; // 6 bits
    std::uint32_t frame      = 0; // 24 bits

    static constexpr std::uint32_t kMaxFrame = (1U << 24) - 1;

    [[nodiscard]] std::uint32_t encode() const;
    [[nodiscard]] static ResidencyEntry decode(std::uint32_t word);
    // The next entry in the state machine; the generation advances modulo 64.
    [[nodiscard]] ResidencyEntry next(ResidencyState state, std::uint32_t frame) const;
};

// ---------------------------------------------------------------------------- LFRU

// Zhang (arXiv 2608.07911): score = f / (now - last + 1) with f a global use count that is never
// reset on eviction; the clock advances once per routed layer call. Victims are the lowest
// (score, key) residents outside the current group, scores compared as IEEE binary64, which is
// what tools/expert_cache_replay's LFRU does, decision for decision.
class LfruPolicy {
public:
    // Which missing keys may be admitted (the SSD tier: only keys with a host copy can be loaded).
    // Every routed key counts as used either way.
    using Admissible = std::function<bool(std::uint32_t)>;

    // halving_period > 0 halves every count each that many ticks (design §9.3, untested default).
    LfruPolicy(std::uint32_t num_keys, std::uint32_t capacity, std::uint32_t halving_period = 0);

    struct Step {
        std::vector<std::uint32_t> hits;
        std::vector<std::uint32_t> misses;   // every routed key that was not resident
        std::vector<std::uint32_t> admitted; // misses made resident by this step
        std::vector<std::uint32_t> victims;  // residents evicted to make room
    };

    // One routed layer call: `group` holds its distinct keys (the union over the round's columns).
    // `admission_budget` limits how many misses become resident (the DRAM/PCIe token bucket);
    // the highest-score misses are admitted first, and only if they outrank the victim they
    // replace. An unlimited budget reproduces the replay tool's on-demand LFRU exactly. With
    // `admissible`, misses it rejects stay non-resident (out.misses still lists them). With
    // `evictable` (the SSD tier's eviction gate) admission is always the budgeted one-victim loop,
    // and it stops at the first victim the gate refuses.
    void step(std::span<const std::uint32_t> group, Step& out,
              std::size_t admission_budget = static_cast<std::size_t>(-1), const Admissible* admissible = nullptr,
              const Admissible* evictable = nullptr);

    // Promote a non-resident key (for example a staged prefetch whose frame changes role).
    // Returns the victim, or nothing when the pool still has room.
    std::optional<std::uint32_t> promote(std::uint32_t key, std::span<const std::uint32_t> protect);

    [[nodiscard]] bool resident(std::uint32_t key) const { return slot_[key] != kNone; }
    [[nodiscard]] std::size_t resident_count() const { return residents_.size(); }
    [[nodiscard]] std::uint32_t capacity() const { return capacity_; }
    [[nodiscard]] std::uint64_t now() const { return now_; }
    [[nodiscard]] double score(std::uint32_t key) const;
    [[nodiscard]] std::uint32_t count(std::uint32_t key) const { return count_[key]; }
    [[nodiscard]] std::uint32_t num_keys() const { return static_cast<std::uint32_t>(count_.size()); }
    [[nodiscard]] std::span<const std::uint32_t> residents() const { return residents_; }

    // Seeds residency and counts, for example from a saved state or a shipped profile.
    void seed(std::span<const std::uint32_t> resident_keys, std::span<const std::uint32_t> counts);

    // Changes the capacity (the frame pool grew or shrank). Shrinking evicts the lowest-score
    // residents, wherever their frames are, into `victims`; uses keep their counts.
    void set_capacity(std::uint32_t capacity, std::vector<std::uint32_t>& victims);
    // Makes a resident key non-resident (its frame is lent); its count is kept.
    void evict(std::uint32_t key);

private:
    static constexpr std::uint32_t kNone = 0xFFFFFFFFU;
    void insert(std::uint32_t key);
    void erase(std::uint32_t key);
    void select_victims(std::size_t need, std::span<const std::uint32_t> protect,
                        std::vector<std::uint32_t>& out);

    std::uint32_t capacity_;
    std::uint32_t halving_period_;
    std::uint64_t now_ = 0;
    std::vector<std::uint32_t> count_;
    std::vector<std::uint64_t> last_;
    std::vector<std::uint32_t> slot_;      // index into residents_, or kNone
    std::vector<std::uint32_t> residents_; // dense list of resident keys
    // count_ and last_ of residents_[i] at i, as doubles, so a victim search is one contiguous
    // floating-point loop. Both are integers below 2^53, so they and now_ - last_ + 1 are exact in
    // binary64 and every score equals score()'s.
    std::vector<double> resident_count_;
    std::vector<double> resident_last_;
    std::vector<std::uint8_t> mark_;       // scratch: keys of the current group
    std::vector<std::pair<double, std::uint32_t>> scratch_;
    std::vector<double> scores_;           // scratch: residents' scores, in residents_ order
    std::vector<std::uint32_t> candidates_;
};

// What a warm start needs (design §19.3.5 S4b): every key's LFRU count, and the resident keys,
// highest score first.
struct SavedState {
    std::vector<std::uint32_t> counts;
    std::vector<std::uint32_t> ranked;
};

// ---------------------------------------------------------------------------- frames

// Frames of the device pool. A frame whose expert was evicted is reusable only once every round
// that might still read it has finished (design §9.5): the agent samples r_s = round_started
// after the ABSENT write is visible, and the frame returns when round_done >= r_s + D, D being the
// number of rounds that can be in flight. Frames [0, backed()) have memory; the pool can grow up
// to `max_frames` and shrink from the top (design §19.3.7).
class FramePool {
public:
    FramePool(std::uint32_t frames, std::uint32_t rounds_in_flight, std::uint32_t max_frames = 0);

    [[nodiscard]] std::optional<std::uint32_t> acquire();
    void release_now(std::uint32_t frame);                         // never held an expert
    void retire(std::uint32_t frame, std::uint64_t round_started_sample);
    void on_round_done(std::uint64_t round_done);
    void on_quiescent();                                            // no round in flight
    // A free frame below `limit` (a relocation target when the pool shrinks to `limit`).
    [[nodiscard]] std::optional<std::uint32_t> acquire_below(std::uint32_t limit);
    // Backs frames [0, frames). Growing frees the new frames; shrinking needs every frame at or
    // above `frames` free (the controller relocates their experts first).
    void resize(std::uint32_t frames);
    // Lends the free frames [first, first + count) to another user of the memory (design
    // §19.3.2, "Frame lending"); they are neither acquired nor resized until given back.
    void lend(std::uint32_t first, std::uint32_t count);
    void give_back(std::uint32_t first, std::uint32_t count);
    // A retired frame whose expert is being copied out (the SSD tier's demotion) is held: neither
    // freed by on_quiescent nor acquired until release_held.
    void hold(std::uint32_t frame);
    void release_held(std::uint32_t frame);

    [[nodiscard]] std::size_t free_count() const { return free_.size(); }
    [[nodiscard]] std::size_t pending_count() const { return pending_.size(); }
    [[nodiscard]] bool is_free(std::uint32_t frame) const { return state_[frame] == kFree; }
    [[nodiscard]] bool is_loaned(std::uint32_t frame) const { return state_[frame] == kLoaned; }
    [[nodiscard]] std::uint32_t loaned_count() const { return loaned_; }
    [[nodiscard]] std::uint32_t backed() const { return backed_; }
    [[nodiscard]] std::uint32_t max_frames() const { return static_cast<std::uint32_t>(state_.size()); }

private:
    enum : std::uint8_t { kFree, kHeld, kPending, kUnbacked, kLoaned, kDemoting };
    std::uint32_t rounds_in_flight_;
    std::uint32_t backed_ = 0;
    std::uint32_t loaned_ = 0;
    std::vector<std::uint32_t> free_;
    std::deque<std::pair<std::uint64_t, std::uint32_t>> pending_; // (reusable at round_done >=, frame)
    std::vector<std::uint8_t> state_;
};

// ---------------------------------------------------------------------------- controller

// The transfer agent's bookkeeping: applies LFRU decisions to the residency table and the frames,
// and emits the copy-stream commands that realize them (design §9.4). An admitted expert with no
// free frame stays queued, CPU-served meanwhile, and is loaded as soon as a frame retires; a queued
// expert that the policy evicts first is simply dropped from the queue.
class CacheController {
public:
    struct Command {
        enum class Kind : std::uint8_t {
            kWriteEntry, // cuStreamWriteValue32(residency[key], word)
            kCopy,       // DMA host record of `key` into `frame`
            kRelocate,   // device copy of `key`'s record from frame `source` into `frame`
        };
        Kind kind;
        std::uint32_t key;
        std::uint32_t frame;
        std::uint32_t word;
        std::uint32_t source = 0;
        std::uint64_t serial = 0; // kCopy: this load's serial (complete_load)
    };

    // `frames` backed now, up to `max_frames` (0: no growth).
    CacheController(std::uint32_t num_keys, std::uint32_t frames, std::uint32_t slack_frames,
                    std::uint32_t rounds_in_flight, std::uint32_t max_frames = 0);

    // The route log of one layer call. `round_started` is the agent's latest sample.
    void on_route(std::span<const std::uint32_t> group, std::uint64_t round_started,
                  std::vector<Command>& out, std::size_t admission_budget = static_cast<std::size_t>(-1),
                  const LfruPolicy::Admissible* admissible = nullptr, const LfruPolicy::Admissible* evictable = nullptr);
    // The SSD tier's demotion (design §19.3.7 T4/T7): after on_route evicted an expert and before
    // on_quiescent, its retired frame is held while its record is copied to RAM; release_held frees
    // it once the copy completed and loads queued experts into free frames.
    void hold(std::uint32_t frame) { frames_.hold(frame); }
    void release_held(std::uint32_t frame, std::vector<Command>& out);
    void on_round_done(std::uint64_t round_done, std::vector<Command>& out);
    void on_quiescent(std::vector<Command>& out);
    // Resizes the pool to `frames` while no round is in flight, every issued load has landed and no
    // frame is lent. Shrinking evicts the lowest-score residents (ABSENT writes) and relocates the
    // experts still above the new top into free frames below it (relocations precede no other
    // command); growing admits queued experts into the new frames.
    void resize(std::uint32_t frames, std::vector<Command>& out);

    // Frame lending (design §19.3.2): a contiguous run of frames serves another use of the memory
    // (the Vision window) and comes back later. choose_run picks the run of `count` frames whose
    // held experts have the minimum summed LFRU score (free frames score 0); lent frames are
    // ineligible, and runs over `busy` frames (targets of loads still in flight) are avoided while a
    // run without them exists. Nothing when no run fits.
    [[nodiscard]] std::optional<std::uint32_t> choose_run(std::uint32_t count, std::span<const std::uint8_t> busy) const;
    // At a quiescent boundary (no retiring frame): evicts every expert held in the run (ABSENT
    // writes), lends the run and shrinks the policy's capacity by `count` (further victims are
    // evicted elsewhere). Returns the number of evicted experts.
    std::uint32_t lend(std::uint32_t first, std::uint32_t count, std::vector<Command>& out);
    // Returns the run: capacity grows back and queued experts load into it.
    void give_back(std::uint32_t first, std::uint32_t count, std::vector<Command>& out);
    // Warm start (design §19.3.5 S4b): sets every key's count from `counts` (empty: unchanged), then
    // loads `keys`, best first, into free frames while the policy has capacity. Before the first
    // round only. Returns the number of keys loaded.
    std::uint32_t seed(std::span<const std::uint32_t> keys, std::span<const std::uint32_t> counts,
                       std::vector<Command>& out);
    // Fill-phase landing (design §19.3.5 S4). reserve_free pops up to `count` free frames into
    // `out` while the policy has room for that many more residents (the LFRU state is untouched)
    // and returns how many it reserved. adopt makes `key`, copied into the reserved `frame` during
    // a round, resident there: READY in the table, admitted by the policy (a victim outside
    // `protect` is evicted as in on_route), a queued load of it dropped; false when the key is
    // already loading or resident elsewhere (the caller then releases the frame). release returns
    // a reserved frame that received no expert.
    std::uint32_t reserve_free(std::uint32_t count, std::vector<std::uint32_t>& out);
    bool adopt(std::uint32_t key, std::uint32_t frame, std::span<const std::uint32_t> protect,
               std::uint64_t round_started, std::vector<Command>& out);
    void release(std::uint32_t frame);
    [[nodiscard]] std::uint32_t loaned_frames() const { return frames_.loaned_count(); }
    // The key held in `frame`, or nothing.
    [[nodiscard]] std::optional<std::uint32_t> frame_key(std::uint32_t frame) const;

    [[nodiscard]] std::uint32_t frames() const { return frames_.backed(); }
    [[nodiscard]] std::uint32_t free_frames() const { return static_cast<std::uint32_t>(frames_.free_count()); }
    [[nodiscard]] const ResidencyEntry& entry(std::uint32_t key) const { return table_[key]; }
    [[nodiscard]] const LfruPolicy& policy() const { return policy_; }
    [[nodiscard]] std::size_t queued_loads() const { return queued_.size(); }
    // Keys admitted and waiting for a free frame (their load is not issued yet).
    [[nodiscard]] const std::deque<std::uint32_t>& queued_keys() const { return queued_; }
    // Whether the copy issued as (key, frame, serial) may be published: the key is still READY in
    // that frame from that very load. A key can be evicted and re-admitted into the same frame
    // while an earlier copy is in flight (the ABA case); only the newest copy's data is current.
    [[nodiscard]] bool complete_load(std::uint32_t key, std::uint32_t frame, std::uint64_t serial) const;

private:
    void load(std::uint32_t key, std::uint32_t frame, std::vector<Command>& out);
    void drain_queue(std::vector<Command>& out);
    // ABSENT write for a resident key; its frame no longer holds it.
    std::uint32_t make_absent(std::uint32_t key, std::vector<Command>& out);

    static constexpr std::uint32_t kNoKey = 0xFFFFFFFFU;

    LfruPolicy policy_;
    std::uint32_t slack_;
    FramePool frames_;
    std::vector<ResidencyEntry> table_;
    std::deque<std::uint32_t> queued_;
    std::vector<std::uint8_t> is_queued_;
    std::vector<std::uint64_t> load_serial_; // per key: the serial of its latest issued copy
    std::vector<std::uint32_t> frame_key_;   // per frame: the key it holds, or kNoKey
    std::uint64_t next_serial_ = 0;
    LfruPolicy::Step step_;
};

} // namespace ninfer::models::qwen4_exp::expert_cache
