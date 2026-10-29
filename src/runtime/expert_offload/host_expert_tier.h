#pragma once

// The SSD expert tier's host level (docs/maintainer/qwen3_8-flash-next-design.md §19.3.7; plan
// memory-tiers.md §4.3-§4.6): pinned slots holding expert records read in place from the artifact,
// the RAM-tier controller that decides which keys they hold (expert_cache::HostTier), and an agent
// thread that owns the unbuffered read queue. The engine thread drives the controller at round
// boundaries; during a round the agent serves demand reads into the ring the boundary handed it:
// the CPU service's SSD-only experts (through RecordProvider) and the device's fetch requests
// (Options::fetch: the responder of ops::offloaded_moe::FetchChannel). A ring slot is reused within
// a round once its reader is done with it (the CPU job finished, or the device consumed the fetched
// record); a landed record still in its slot is admitted at the next boundary when it outranks the
// RAM victim.
//
// Unpinned records: Windows page-locks memory for the device only up to the adapter's shared system
// memory (often half of RAM), so the RAM past that holds a fixed set of records in pageable memory
// kept resident (ResidentHostBuffer), filled at startup after the slots (fill_unpinned) and never
// replaced (HostTier::held). The device cannot read them: they are absent from its record table, so
// the device treats them as SSD-only. A CPU demand is served in place, without a read; a fetch
// request copies the record into a ring slot (copy threads, memory speed), and the copy is never
// admitted (the key keeps its unpinned record). Promotions read them through host_record().
//
// Threads: the engine thread calls every method except the RecordProvider ones, and only while no
// round is in flight (begin_round, end_round, record_uses, prefill, the controller), except the
// stream slots' (stream_read, stream_wait, stream_release: the prefill stream's reads of SSD-only
// records, at any time); the CPU service thread calls demand / wait / done during a round. The
// agent is internal: it spins while a round is open (begin_round to end_round), so a request waits
// for no timer, and sleeps otherwise.
//
// Size classes (HostTier): when the layers' records differ in size (a GGUF import), the pinned slots
// are pooled by record size, each class's slots exactly its records' size, so the tier's RAM holds
// ~29 % more IQ3_S records than slots of the largest record did. The transit slots (the ring, the
// prefetch list and the bounce slots) take any record and keep the largest size; a landing admitted
// into a smaller class is copied there at the next boundary. Each layer gets the same share of its
// class's resident slots, as each layer gets the same share of the VRAM frames. Records of one size
// (NVFP4, block-FP8) keep one pool of slots with the transit slots carved from it.

#include "core/direct_read_queue.h"
#include "core/host_memory.h"
#include "runtime/expert_offload/expert_store.h"
#include "runtime/expert_offload/host_tier.h"
#include "ops/offloaded_sparse_moe/cpu/fetch_channel.h"
#include "ops/offloaded_sparse_moe/cpu/record_provider.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace infernix::runtime::expert_offload {

class HostExpertTier final : public ops::offloaded_moe::RecordProvider {
public:
    struct Options {
        // Every pinned slot: residents, ring, prefetch and demotion lists, bounce slots. With records of
        // several sizes a class slot is its record's size (slots_for_bytes converts a RAM budget).
        std::uint32_t slots     = 0;
        std::uint32_t ring      = 128;
        std::uint32_t prefetch  = 16;
        std::uint32_t demotion  = 32;
        double half_life        = 64;
        std::uint32_t slots_per_chunk = 512; // pinned in chunks of this many slots
        // Fewest pinned slots the tier runs with (default: the lists plus one resident slot).
        std::uint32_t min_slots = 0;
        // Unpinned records beyond the slots (at most the keys the resident slots leave).
        std::uint32_t unpinned = 0;
        // When the OS refuses to pin a chunk, the tier keeps the slots it pinned less this many bytes
        // (left for the pinned allocations made after it), and the slots it could not keep become
        // unpinned records.
        std::uint64_t pin_reserve = 0;
        // Threads copying unpinned records into pinned memory (0: a third of the hardware threads,
        // 2 to 8). Copies run at DRAM speed: 20 GB/s on one i9-13900K thread, 36 on four, 41 on
        // eight, against a 5.0 x16 link's ~50 GB/s.
        std::uint32_t copy_threads = 0;
        // Pinned slots set aside, when there are unpinned records, as the bounce buffer of their
        // VRAM promotions (two halves; see bounce_slot): a pageable source makes the driver stage
        // the copy synchronously at ~13 GB/s while it holds the calling thread.
        std::uint32_t bounce = 32;
        // Further pinned slots set aside, with unpinned records, as the prefill expert stream's
        // bounce (two halves of a layer's unpinned records each; ExpertStream::Staging).
        std::uint32_t stream_bounce = 0;
        // Shadows (RAM copies of VRAM experts) kept for a prefill's loan of VRAM frames, shared by the
        // classes by their layers (HostTier::Class::shadow_floor; the exclusive layout). 0: none kept.
        std::uint32_t shadow_floor = 0;
        DirectReadQueue::Options io;
        // Pinned memory the device can read (default: cudaHostAlloc portable | mapped), null when the
        // OS refuses, and its release.
        std::function<void*(std::size_t)> allocate;
        std::function<void(void*)> release;
        // The device's fetch channel this tier answers (owned by the caller; null: CPU demands only).
        // The ring must exceed the CPU jobs of one call plus the largest pass's fetched records.
        ops::offloaded_moe::FetchChannel* fetch = nullptr;
        // The most layer calls one round makes (0: one per layer). A layer call demands each of its
        // experts once, so a round holds this many layers' worth of tickets; a layer walk's step
        // calls its layers once per chunk of the span.
        std::uint32_t layer_calls = 0;
        // Read-ahead: a fetch request of at least `ahead_min` records (a wide prefill call, which
        // routes nearly every expert) also queues, behind its own reads, up to `ahead` of the next
        // layer's records without a host copy; the next request adopts those it names, the others
        // give their ring slots back. 0 disables it.
        std::uint32_t ahead_min = 64;
        std::uint32_t ahead     = 32;
        // The most resident slots lend_stream lends to the prefill expert stream for a layer walk's
        // span: SSD-only records it reads ahead (stream_read) and copies to the device once per span.
        // 0 disables it.
        std::uint32_t stream = 0;
    };

    struct Stats {
        std::uint64_t demand_reads = 0, demand_failures = 0, prefill_reads = 0, admitted = 0, discarded = 0;
        std::uint64_t copied_admissions = 0; // admissions copied into a class of smaller records
        std::uint64_t fetch_requests = 0, fetch_records = 0, fetch_from_ram = 0; // the fetch channel's
        std::uint64_t read_ns = 0; // demand reads, submit to landing, summed
        // Scatter reads of records that follow each other in a file, and the records they read.
        std::uint64_t run_reads = 0, run_records = 0;
        std::uint64_t stream_reads = 0; // records read into stream slots
        // Unpinned records served: in place to the CPU, and copied into ring slots for fetches.
        std::uint64_t unpinned_served = 0, unpinned_copies = 0;
    };

    HostExpertTier(const ExpertStore& store, Options options);
    ~HostExpertTier() override;

    // The Options::slots that `bytes` of pinned memory hold for `store`'s records with `options`'s
    // lists: one slot per largest record when the records share a size (at most every key plus the
    // ring, prefetch and demotion lists), else the transit slots of the largest record plus class
    // slots at the layers' mean record size (at most every key plus the classes' demotion lists).
    [[nodiscard]] static std::uint32_t slots_for_bytes(const ExpertStore& store, std::uint64_t bytes, const Options& options);
    // Layers a prefill stream holds reads for at once (ExpertStream::kReadAhead + 2): stream_layer_cap
    // shares each class's lent slots among min(its layers, this), so each layer's reads fit them.
    static constexpr std::uint32_t kStreamWindow = 4;

    HostExpertTier(const HostExpertTier&)            = delete;
    HostExpertTier& operator=(const HostExpertTier&) = delete;

    [[nodiscard]] std::uint32_t keys() const noexcept { return store_.layers() * store_.experts(); }
    [[nodiscard]] std::uint32_t key(std::uint32_t layer, std::uint32_t expert) const noexcept {
        return layer * store_.experts() + expert;
    }
    // The largest record: the size of a transit slot and of an unpinned record.
    [[nodiscard]] std::size_t record_bytes() const noexcept { return store_.record_bytes(); }

    // The pinned slots (Options::slots, less those the OS refused) and the unpinned records.
    [[nodiscard]] std::uint32_t pinned_slots() const noexcept { return pinned_.slots; }
    // The pinned slots' bytes (each class's slots at its record size).
    [[nodiscard]] std::uint64_t pinned_bytes() const noexcept { return pinned_.bytes; }
    [[nodiscard]] std::uint32_t unpinned_records() const noexcept { return unpinned_count_; }
    // Slots of Options::slots the OS refused to pin (they became unpinned records); 0 normally.
    [[nodiscard]] std::uint32_t refused_slots() const noexcept { return pinned_.refused; }
    // Empty when the unpinned records are locked in RAM, else the OS's reason (they may be paged).
    [[nodiscard]] const std::string& unpinned_lock_failure() const noexcept;

    // ---- engine thread, no round in flight
    // Reads the keys of `ranked` (best first) that have no host copy into free resident slots until
    // none is left; returns how many.
    std::uint32_t prefill(std::span<const std::uint32_t> ranked);
    // Reads the keys of `ranked` (best first) that have no host copy and that `skip` rejects into
    // unpinned records until none is left; they keep them for good. Returns how many.
    std::uint32_t fill_unpinned(std::span<const std::uint32_t> ranked, const std::function<bool(std::uint32_t)>& skip);
    // The warm start's spare pinned slots (every key outside VRAM already has a host copy): reads the
    // keys of `ranked` that VRAM holds, in its order (the caller passes the least-used first: a prefill
    // loan's and decode's next victims), as their shadows until no slot is left (HostTier::prefill_shadow).
    // Returns how many.
    std::uint32_t prefill_shadows(std::span<const std::uint32_t> ranked);
    // A round boundary: admits the last round's landings (T10), refills the demotion list, then hands
    // the agent a fresh ring and opens the round. `allowance` is the boundary's demotion allowance (HostTier::begin_round).
    void begin_round(std::uint32_t allowance);
    // The round's kernels have completed (or it was abandoned): the agent stops spinning.
    void end_round() noexcept;
    // Whether a round is open: only then does the agent answer the device's fetch requests.
    [[nodiscard]] bool round_open() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return open_;
    }
    // The decayed-LFU clock and the round's routed keys.
    void record_uses(double dt, std::span<const std::uint32_t> keys, double weight) {
        tier_.record_uses(dt, keys, weight);
    }
    // The device-readable host copy of `key` (resident or shadow slot), or null (SSD-only, or an
    // unpinned record).
    [[nodiscard]] const std::uint8_t* record(std::uint32_t key) const noexcept;
    // key's unpinned record, or null.
    [[nodiscard]] const std::uint8_t* unpinned_record(std::uint32_t key) const noexcept {
        const std::int32_t i = unpinned_of_[key];
        return i >= 0 ? reinterpret_cast<const std::uint8_t*>(unpinned_->data()) + static_cast<std::size_t>(i) * store_.record_bytes()
                      : nullptr;
    }
    // Any host copy of `key` the host may read (its slot, else its unpinned record), or null.
    [[nodiscard]] const std::uint8_t* host_record(std::uint32_t key) const noexcept {
        const std::uint8_t* slot = record(key);
        return slot != nullptr ? slot : unpinned_record(key);
    }
    // The bounce slots for promotions of unpinned records (0 without unpinned records); pinned,
    // outside the controller's slots.
    [[nodiscard]] std::uint32_t bounce_slots() const noexcept { return bounce_; }
    [[nodiscard]] std::uint8_t* bounce_slot(std::uint32_t i) const noexcept { return bounce_address_[i]; }
    // The prefill stream's bounce slots (0 without unpinned records), after the promotions'.
    [[nodiscard]] std::uint32_t stream_bounce_slots() const noexcept { return stream_bounce_; }
    [[nodiscard]] std::uint8_t* stream_bounce_slot(std::uint32_t i) const noexcept {
        return bounce_address_[bounce_ + i];
    }
    // Engine thread: copies whole records (destination, source) on the copy threads and this
    // thread, and returns once all have landed.
    void copy_records(std::span<const std::pair<std::uint8_t*, const std::uint8_t*>> copies);
    // The prefill expert stream's hold on the records it plans to copy (HostTier::stream_pin) and
    // its release at a boundary after the copies landed.
    void stream_pin(std::uint32_t key) { tier_.stream_pin(key); }
    void release_stream_pins() { tier_.release_stream_pins(); }
    // Keys whose host pointer changed since the last call (the device table's dirty set).
    void take_dirty(std::vector<std::uint32_t>& keys) { tier_.take_dirty(keys); }
    // ---- stream slots (Options::stream)
    // Engine thread, no round in flight: lends up to min(n, Options::stream) resident slots to the
    // stream, each class its share by its layers (free slots first, then the lowest-ranked RAM
    // residents, which leave RAM; HostTier::lend) and returns how many; and takes them back as free
    // resident slots (every stream slot released), which admissions refill.
    std::uint32_t lend_stream(std::uint32_t n);
    void return_stream();
    // The engine thread, during a round or between rounds: queues a read of `key`'s record into a
    // free stream slot of its class and returns the slot, or -1 when none is free (or the agent has
    // stopped); the record is the artifact's whatever the key's RAM state.
    std::int32_t stream_read(std::uint32_t key);
    // The record of stream slot `slot` once its read has landed (waits), or null when it failed.
    const std::uint8_t* stream_wait(std::int32_t slot) noexcept;
    // Gives a stream slot back: its read has landed and nothing reads the slot any more.
    void stream_release(std::int32_t slot);
    // Stream slots lent now.
    [[nodiscard]] std::uint32_t stream_slots() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<std::uint32_t>(stream_slot_.size());
    }
    // The stream reads a layer may hold at once: kStreamWindow layers' reads of its class fit the class's
    // lent slots.
    [[nodiscard]] std::uint32_t stream_layer_cap(std::uint32_t layer) const noexcept;
    // ---- refills (a demotion that reads the evicted expert's record from the artifact instead of copying its frame
    // back over the GPU's link)
    // The engine thread, between rounds: queues a read of `key`'s record into `slot` (the demotion slot
    // HostTier::vram_evicted chose) and returns its id, or kNoRefill when kRefills are in flight already.
    // refill_state(id) is 0 while the read is in flight, 1 once it landed, 2 when it failed; refill_release(id) lets
    // the id be reused once the state has been read.
    static constexpr std::uint32_t kRefills  = 1024;
    static constexpr std::uint32_t kNoRefill = 0xFFFFFFFFU;
    std::uint32_t refill(std::uint32_t key, std::uint32_t slot);
    [[nodiscard]] std::uint32_t refill_state(std::uint32_t id) const noexcept {
        return refill_state_[id].load(std::memory_order_acquire);
    }
    void refill_release(std::uint32_t id) { refill_free_.push_back(id); }
    [[nodiscard]] bool refill_available() const noexcept { return !refill_free_.empty(); }
    // Whether one more refill, behind those in flight, lands within the given seconds at the rate the startup fill read the
    // artifact (true before the fill measured it).
    [[nodiscard]] bool refill_within(double seconds) const noexcept;
    [[nodiscard]] expert_cache::HostTier& controller() noexcept { return tier_; }
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
    [[nodiscard]] const std::uint8_t* slot_bytes(std::uint32_t slot) const noexcept;

    // ---- RecordProvider: the CPU service thread, during a round
    std::uint32_t demand(int layer, int expert) noexcept override;
    [[nodiscard]] bool landed(std::uint32_t ticket) const noexcept override;
    const std::uint8_t* wait(std::uint32_t ticket, std::uint32_t& status) noexcept override;
    void done(std::uint32_t ticket) noexcept override;

    // A ticket demand() returns when a round has demanded more records than it can track (wait
    // then reports ENOBUFS).
    static constexpr std::uint32_t kNoTicket = 0xFFFFFFFFU;
    // Every key can be demanded once per round (a layer call reads each of its experts once).
    [[nodiscard]] std::uint32_t ticket_capacity() const noexcept { return ticket_capacity_; }
    // Why the agent thread stopped (an error of the read queue itself, not a failed read); empty
    // while it runs. Its pending and later demands then fail at once instead of waiting, and the
    // caller treats the tier as failed.
    [[nodiscard]] std::string failure() const;

private:
    struct Ticket {
        std::uint32_t key = 0, slot = 0;
        std::uint64_t serial = 0;
        std::uint32_t remaining = 0; // segments still in flight (agent only)
        std::uint64_t submitted_ns = 0;
        bool resident = false;       // the key was already in RAM: nothing read, nothing to admit
        bool recycled = false;       // its slot was reused for a later demand of the round
        std::int32_t fetch = -1;     // its index in the fetch request it belongs to (-1: a CPU demand)
        const std::uint8_t* copy_from = nullptr; // a fetch of an unpinned record: copied, never admitted
        const std::uint8_t* bytes     = nullptr; // the record once landed (state 1)
        std::atomic<bool> done{false};
        std::atomic<std::uint32_t> state{0}; // 0 pending, 1 landed, 2 failed
        std::atomic<std::uint32_t> status{0};
    };
    struct RingSlot {
        std::uint32_t slot = 0;
        std::uint64_t serial = 0;
    };
    // The pinned slots' layout: regions of slots of one size, pinned in chunks of slots_per_chunk
    // slots. Records of one size: one region (the controller's slots, then the bounce slots). Several
    // sizes: region 0 holds the transit slots (ring, prefetch, bounce, stream bounce) at the largest
    // record's size, region 1 + c class c's slots; `classes` sizes the controller's classes.
    struct Region {
        std::uint64_t pitch = 0;
        std::uint32_t slots = 0;
    };
    struct Plan {
        std::vector<Region> regions;
        std::vector<expert_cache::HostTier::Class> classes; // empty: records of one size
        std::vector<std::uint8_t> layer_class;
        std::vector<std::uint32_t> class_layers;
        std::uint32_t transit = 0; // region 0's slots outside the controller (several sizes): the bounce slots
    };
    // The pinned slots: chunks released on destruction, and each region's slot addresses in order.
    struct Pinned {
        std::vector<std::byte*> chunks;
        std::vector<std::vector<std::uint8_t*>> regions;
        std::uint32_t slots   = 0;
        std::uint32_t refused = 0; // slots the OS refused to pin
        std::uint64_t bytes   = 0;
        std::function<void(void*)> release;
        Pinned() = default;
        Pinned(Pinned&& other) noexcept;
        Pinned& operator=(Pinned&&) = delete;
        ~Pinned();
    };
    // A copy into pinned memory (copy threads): a fetch's ring slot from an unpinned record, tagged
    // as a demand read, or a batch (copy_records, the admissions' copies) counting down `left`.
    struct Copy {
        std::uint8_t* destination   = nullptr;
        const std::uint8_t* source  = nullptr;
        std::uint64_t tag           = 0;
        std::uint32_t* left         = nullptr; // a batch: guarded by copy_mutex_
        std::size_t bytes           = 0;       // 0: the largest record
    };
    // Runs `copy` and reports it (copy_mutex_ not held).
    void finish_copy(const Copy& copy);
    // Runs a batch of copies on the copy threads and this thread; returns once all have landed.
    void run_copies(std::vector<Copy>& batch);

    static Options normalized(Options options);
    static Plan plan_slots(const ExpertStore& store, const Options& options);
    static Pinned pin_slots(const Plan& plan, const Options& options);
    expert_cache::HostTier::Config controller_config() const;
    void copier_main() noexcept;

    // The agent thread: agent_loop, and agent_failed when it throws.
    void agent_main() noexcept;
    void agent_loop();
    // Records the failure and fails every pending ticket and the startup fill (mutex_ not held).
    void agent_failed(std::string what) noexcept;
    // A pending ticket that will never be read (its request failed, or the agent stopped).
    void fail_ticket(std::uint32_t t) noexcept;
    // Agent: submits the reads of `key` into `destination` with tags (kind, index, segment).
    void submit(std::uint32_t key, std::uint8_t* destination, std::uint64_t tag_base,
                DirectReadQueue::Priority priority = DirectReadQueue::Priority::Demand);
    // Whether `key`'s record starts where `before`'s ends, each in one segment of the same file.
    [[nodiscard]] bool follows(std::uint32_t before, std::uint32_t key) const;
    // Agent: one scatter read of a run of (key, destination) whose records follow each other in a file.
    void submit_run(std::span<const std::pair<std::uint32_t, std::uint8_t*>> run, std::uint64_t tag);
    // Reads (key, destination) records before the first round (prefill, fill_unpinned).
    void read_at_startup(const std::vector<std::pair<std::uint32_t, std::uint8_t*>>& jobs);
    // A new ticket of this round for `key` (kNoTicket when the round has run out); resident when the
    // key is in RAM, else pending a read.
    std::uint32_t new_ticket(std::uint32_t key, std::int32_t fetch);
    // Agent: a ring slot for ticket t (a free one, else the oldest holder whose reader is done).
    bool take_slot(std::uint32_t t);
    // Waits until every request handed to the agent has completed.
    void drain();

    const ExpertStore& store_;
    Options options_;
    Plan plan_;
    Pinned pinned_;
    std::uint32_t bounce_ = 0, stream_bounce_ = 0; // pinned slots outside the controller
    std::vector<std::uint8_t*> slot_address_;      // the controller's slots
    std::vector<std::uint8_t*> bounce_address_;    // the bounce slots, then the stream bounce slots
    expert_cache::HostTier tier_;
    // Unpinned records: key -> index (-1 none), the buffer, and how many are allocated and filled.
    std::vector<std::int32_t> unpinned_of_;
    std::unique_ptr<ResidentHostBuffer> unpinned_;
    std::uint32_t unpinned_count_ = 0, unpinned_filled_ = 0;
    // Copy threads and their queue (copier_main); completions are drained by the agent.
    std::mutex copy_mutex_;
    std::condition_variable copy_wake_, copy_done_;
    std::deque<Copy> copies_;
    std::vector<std::uint64_t> copied_; // tags of finished copies
    bool copy_stop_ = false;
    std::vector<std::thread> copiers_;

    // Agent state. Guarded by mutex_: the queues of new work and the round's ring.
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    bool stop_ = false;
    std::string agent_failure_; // non-empty once the agent thread has stopped on an error
    std::vector<std::uint32_t> new_demands_; // ticket indices
    std::vector<RingSlot> ring_;             // free ring slots of this round, taken from the back
    std::deque<std::uint32_t> holding_;      // tickets holding a ring slot, oldest first
    std::uint64_t round_ = 0;                // begin_round count
    bool open_           = false;            // a round is open: the agent spins
    std::vector<std::pair<std::uint32_t, std::uint8_t*>> prefill_jobs_; // (key, destination)
    std::uint32_t prefill_pending_ = 0;
    std::condition_variable prefill_done_;
    std::atomic<std::uint32_t> in_flight_{0}; // demands and prefill reads not yet completed
    std::vector<std::uint32_t> round_ticket_of_key_; // ticket + 1 of a key demanded this round, 0 none
    // Stream slots: the lent tier slots (stream slot s is slot stream_slot_[s]), queued reads
    // (key, stream slot), the free stream slots (all guarded by mutex_), and each stream slot's
    // state (0 pending, 1 landed, 2 failed).
    std::vector<std::uint32_t> stream_slot_;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> stream_jobs_;
    std::vector<std::vector<std::uint32_t>> stream_free_; // by class
    std::vector<std::uint32_t> stream_lent_;              // by class: lent stream slots
    std::vector<std::uint8_t> stream_class_;              // by stream slot: its class
    std::unique_ptr<std::atomic<std::uint32_t>[]> stream_state_;
    std::atomic<std::uint64_t> agent_stream_reads_{0};
    // Refills: queued reads (key, slot, id) (guarded by mutex_), each id's state (0 in flight, 1 landed, 2 failed),
    // and the free ids (the engine thread's).
    struct RefillJob {
        std::uint32_t key, slot, id;
    };
    std::vector<RefillJob> refill_jobs_;
    std::unique_ptr<std::atomic<std::uint32_t>[]> refill_state_;
    std::vector<std::uint32_t> refill_free_;
    double read_rate_ = 0; // bytes a second of the startup fill (prefill), 0 before it

    // Tickets of the current round (stable addresses; reset at each boundary).
    std::unique_ptr<Ticket[]> tickets_;
    std::uint32_t ticket_capacity_ = 0;
    std::atomic<std::uint32_t> ticket_count_{0};

    std::unique_ptr<DirectReadQueue> queue_;
    std::vector<std::uint32_t> files_;
    Stats stats_;
    // Written by the agent; read at boundaries.
    std::atomic<std::uint64_t> agent_reads_{0}, agent_failures_{0}, agent_read_ns_{0};
    std::atomic<std::uint64_t> agent_fetch_requests_{0}, agent_fetch_records_{0}, agent_fetch_from_ram_{0};
    std::atomic<std::uint64_t> agent_run_reads_{0}, agent_run_records_{0};
    std::atomic<std::uint64_t> unpinned_served_{0}, unpinned_copies_{0};
    std::uint32_t prefill_failures_ = 0; // published to prefill() by prefill_done_
    std::thread agent_;
};

} // namespace infernix::runtime::expert_offload
