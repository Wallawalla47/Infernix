#pragma once

// The SSD expert tier's host level (docs/maintainer/qwen3_8-flash-next-design.md §19.3.7; plan
// memory-tiers.md §4.3-§4.6): pinned slots holding expert records read in place from the artifact,
// the RAM-tier controller that decides which keys they hold (expert_cache::HostTier), and an agent
// thread that owns the unbuffered read queue. The engine thread drives the controller at round
// boundaries; during a round the agent serves demand reads (the CPU service's SSD-only experts,
// through RecordProvider) into the ring the boundary handed it. A landed record is admitted at the
// next boundary when it outranks the RAM victim.
//
// Threads: the engine thread calls every method except the RecordProvider ones, and only while no
// round is in flight (begin_round, record_uses, prefill, the controller); the CPU service thread
// calls demand / wait / done during a round. The agent is internal.

#include "core/direct_read_queue.h"
#include "models/qwen4_exp/expert_store.h"
#include "models/qwen4_exp/program/expert_cache/host_tier.h"
#include "ops/offloaded_sparse_moe/cpu/record_provider.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace ninfer::models::qwen4_exp {

class HostExpertTier final : public ops::offloaded_moe::RecordProvider {
public:
    struct Options {
        std::uint32_t slots     = 0; // every pinned slot: residents, ring, prefetch and demotion lists
        std::uint32_t ring      = 128;
        std::uint32_t prefetch  = 16;
        std::uint32_t demotion  = 32;
        double half_life        = 64;
        std::uint32_t slots_per_chunk = 512; // pinned in chunks of this many slots
        DirectReadQueue::Options io;
        // Pinned memory the device can read (default: cudaHostAlloc portable | mapped) and its release.
        std::function<void*(std::size_t)> allocate;
        std::function<void(void*)> release;
    };

    struct Stats {
        std::uint64_t demand_reads = 0, demand_failures = 0, prefill_reads = 0, admitted = 0, discarded = 0;
        std::uint64_t read_ns = 0; // demand reads, submit to landing, summed
    };

    HostExpertTier(const ExpertStore& store, Options options);
    ~HostExpertTier() override;

    HostExpertTier(const HostExpertTier&)            = delete;
    HostExpertTier& operator=(const HostExpertTier&) = delete;

    [[nodiscard]] std::uint32_t keys() const noexcept { return store_.layers() * store_.experts(); }
    [[nodiscard]] std::uint32_t key(std::uint32_t layer, std::uint32_t expert) const noexcept {
        return layer * store_.experts() + expert;
    }
    [[nodiscard]] std::size_t record_bytes() const noexcept { return store_.record_bytes(); }

    // ---- engine thread, no round in flight
    // Reads the first keys of `ranked` (best first) into free resident slots; returns how many.
    std::uint32_t prefill(std::span<const std::uint32_t> ranked);
    // A round boundary: admits the last round's landings (T10), then hands the agent a fresh ring.
    // `allowance` is the boundary's demotion allowance (HostTier::begin_round).
    void begin_round(std::uint32_t allowance);
    // The decayed-LFU clock and the round's routed keys.
    void record_uses(double dt, std::span<const std::uint32_t> keys, double weight) {
        tier_.record_uses(dt, keys, weight);
    }
    // The host copy of `key` (resident or shadow slot), or null for an SSD-only key.
    [[nodiscard]] const std::uint8_t* record(std::uint32_t key) const noexcept;
    // Keys whose host pointer changed since the last call (the device table's dirty set).
    void take_dirty(std::vector<std::uint32_t>& keys) { tier_.take_dirty(keys); }
    [[nodiscard]] expert_cache::HostTier& controller() noexcept { return tier_; }
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
    [[nodiscard]] const std::uint8_t* slot_bytes(std::uint32_t slot) const noexcept;

    // ---- RecordProvider: the CPU service thread, during a round
    std::uint32_t demand(int layer, int expert) noexcept override;
    const std::uint8_t* wait(std::uint32_t ticket, std::uint32_t& status) noexcept override;
    void done(std::uint32_t ticket) noexcept override;

    // A ticket demand() returns when a round has demanded more records than it can track (wait
    // then reports ENOBUFS).
    static constexpr std::uint32_t kNoTicket = 0xFFFFFFFFU;

private:
    struct Ticket {
        std::uint32_t key = 0, slot = 0;
        std::uint64_t serial = 0;
        std::uint32_t remaining = 0; // segments still in flight (agent only)
        std::uint64_t submitted_ns = 0;
        bool resident = false;       // the key was already in RAM: nothing read, nothing to admit
        bool recycled = false;       // its slot was reused for a later demand of the round
        std::atomic<bool> done{false};
        std::atomic<std::uint32_t> state{0}; // 0 pending, 1 landed, 2 failed
        std::atomic<std::uint32_t> status{0};
    };
    struct RingSlot {
        std::uint32_t slot = 0;
        std::uint64_t serial = 0;
    };

    void agent_main();
    // Agent: submits the reads of `key` into `slot` with tags (kind, index, segment).
    void submit(std::uint32_t key, std::uint32_t slot, std::uint64_t tag_base);
    // Waits until every request handed to the agent has completed.
    void drain();

    const ExpertStore& store_;
    Options options_;
    expert_cache::HostTier tier_;
    std::vector<std::byte*> chunks_;
    std::uint32_t per_chunk_ = 0;

    // Agent state. Guarded by mutex_: the queues of new work and the round's ring.
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stop_ = false;
    std::vector<std::uint32_t> new_demands_; // ticket indices
    std::vector<RingSlot> ring_;             // free ring slots of this round, taken from the back
    std::vector<std::pair<std::uint32_t, std::uint32_t>> prefill_jobs_; // (key, slot)
    std::uint32_t prefill_pending_ = 0;
    std::condition_variable prefill_done_;
    std::atomic<std::uint32_t> in_flight_{0}; // demands and prefill reads not yet completed
    std::vector<std::uint32_t> round_ticket_of_key_; // ticket + 1 of a key demanded this round, 0 none

    // Tickets of the current round (stable addresses; reset at each boundary).
    std::unique_ptr<Ticket[]> tickets_;
    std::uint32_t ticket_capacity_ = 0;
    std::atomic<std::uint32_t> ticket_count_{0};

    std::unique_ptr<DirectReadQueue> queue_;
    std::vector<std::uint32_t> files_;
    Stats stats_;
    // Written by the agent; read at boundaries.
    std::atomic<std::uint64_t> agent_reads_{0}, agent_failures_{0}, agent_read_ns_{0};
    std::uint32_t prefill_failures_ = 0; // published to prefill() by prefill_done_
    std::thread agent_;
};

} // namespace ninfer::models::qwen4_exp
