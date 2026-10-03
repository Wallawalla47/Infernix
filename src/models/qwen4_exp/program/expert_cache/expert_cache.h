#pragma once

// Host side of the Qwen3.8-Flash-Next expert cache (docs/maintainer/qwen3_8-flash-next-design.md
// §9): the LFRU policy, the device residency-entry encoding, and the frame pool with epoch-based
// reuse. Pure host logic with no CUDA dependency; the transfer agent drives it and turns its
// decisions into copy-stream commands.

#include <cstddef>
#include <cstdint>
#include <deque>
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
    // replace. An unlimited budget reproduces the replay tool's on-demand LFRU exactly.
    void step(std::span<const std::uint32_t> group, Step& out,
              std::size_t admission_budget = static_cast<std::size_t>(-1));

    // Promote a non-resident key (for example a staged prefetch whose frame changes role).
    // Returns the victim, or nothing when the pool still has room.
    std::optional<std::uint32_t> promote(std::uint32_t key, std::span<const std::uint32_t> protect);

    [[nodiscard]] bool resident(std::uint32_t key) const { return slot_[key] != kNone; }
    [[nodiscard]] std::size_t resident_count() const { return residents_.size(); }
    [[nodiscard]] std::uint32_t capacity() const { return capacity_; }
    [[nodiscard]] std::uint64_t now() const { return now_; }
    [[nodiscard]] double score(std::uint32_t key) const;
    [[nodiscard]] std::span<const std::uint32_t> residents() const { return residents_; }

    // Seeds residency and counts, for example from a saved state or a shipped profile.
    void seed(std::span<const std::uint32_t> resident_keys, std::span<const std::uint32_t> counts);

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
    std::vector<std::uint8_t> mark_;       // scratch: keys of the current group
    std::vector<std::pair<double, std::uint32_t>> scratch_;
};

// ---------------------------------------------------------------------------- frames

// Frames of the device pool. A frame whose expert was evicted is reusable only once every round
// that might still read it has finished (design §9.5): the agent samples r_s = round_started
// after the ABSENT write is visible, and the frame returns when round_done >= r_s + D, D being the
// number of rounds that can be in flight.
class FramePool {
public:
    FramePool(std::uint32_t frames, std::uint32_t rounds_in_flight);

    [[nodiscard]] std::optional<std::uint32_t> acquire();
    void release_now(std::uint32_t frame);                         // never held an expert
    void retire(std::uint32_t frame, std::uint64_t round_started_sample);
    void on_round_done(std::uint64_t round_done);
    void on_quiescent();                                            // no round in flight

    [[nodiscard]] std::size_t free_count() const { return free_.size(); }
    [[nodiscard]] std::size_t pending_count() const { return pending_.size(); }
    [[nodiscard]] bool is_free(std::uint32_t frame) const { return state_[frame] == kFree; }

private:
    enum : std::uint8_t { kFree, kHeld, kPending };
    std::uint32_t rounds_in_flight_;
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
        };
        Kind kind;
        std::uint32_t key;
        std::uint32_t frame;
        std::uint32_t word;
    };

    CacheController(std::uint32_t num_keys, std::uint32_t frames, std::uint32_t slack_frames,
                    std::uint32_t rounds_in_flight);

    // The route log of one layer call. `round_started` is the agent's latest sample.
    void on_route(std::span<const std::uint32_t> group, std::uint64_t round_started,
                  std::vector<Command>& out, std::size_t admission_budget = static_cast<std::size_t>(-1));
    void on_round_done(std::uint64_t round_done, std::vector<Command>& out);
    void on_quiescent(std::vector<Command>& out);

    [[nodiscard]] const ResidencyEntry& entry(std::uint32_t key) const { return table_[key]; }
    [[nodiscard]] const LfruPolicy& policy() const { return policy_; }
    [[nodiscard]] std::size_t queued_loads() const { return queued_.size(); }

private:
    void load(std::uint32_t key, std::uint32_t frame, std::vector<Command>& out);
    void drain_queue(std::vector<Command>& out);

    LfruPolicy policy_;
    FramePool frames_;
    std::vector<ResidencyEntry> table_;
    std::deque<std::uint32_t> queued_;
    std::vector<std::uint8_t> is_queued_;
    LfruPolicy::Step step_;
};

} // namespace ninfer::models::qwen4_exp::expert_cache
