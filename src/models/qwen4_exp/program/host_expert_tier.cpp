#include "models/qwen4_exp/program/host_expert_tier.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <stdexcept>
#include <string>

#if defined(__x86_64__) || defined(_M_X64)
#    include <immintrin.h>
#endif

namespace ninfer::models::qwen4_exp {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kDemandTag  = 0;
constexpr std::uint64_t kPrefillTag = 1;
// An open round with no work for this long (one that was abandoned without end_round) stops the
// agent's spinning; it then polls about once per millisecond.
constexpr auto kIdleSpin = std::chrono::seconds(2);

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

std::uint64_t tag_of(std::uint64_t kind, std::uint64_t index, std::uint64_t segment) {
    return (kind << 62U) | (index << 8U) | segment;
}

void relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

void* pinned_allocate(std::size_t bytes) {
    void* p = nullptr;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess) {
        (void)cudaGetLastError();
        throw std::runtime_error("SSD tier: cannot pin " + std::to_string(bytes >> 20) + " MiB of expert slots");
    }
    return p;
}

void pinned_release(void* p) { (void)cudaFreeHost(p); }

} // namespace

HostExpertTier::HostExpertTier(const ExpertStore& store, Options options)
    : store_(store), options_(std::move(options)),
      tier_(expert_cache::HostTier::Config{.num_keys  = store.layers() * store.experts(),
                                           .slots     = options_.slots,
                                           .ring      = options_.ring,
                                           .prefetch  = options_.prefetch,
                                           .demotion  = options_.demotion,
                                           .half_life = options_.half_life}),
      round_ticket_of_key_(static_cast<std::size_t>(store.layers()) * store.experts(), 0) {
    if (!options_.allocate) { options_.allocate = pinned_allocate; }
    if (!options_.release) { options_.release = pinned_release; }
    if (options_.slots_per_chunk == 0) { throw std::invalid_argument("SSD tier: slots per chunk must be positive"); }
    per_chunk_ = options_.slots_per_chunk;
    try {
        for (std::uint32_t s = 0; s < options_.slots; s += per_chunk_) {
            const std::uint32_t here = std::min(per_chunk_, options_.slots - s);
            auto* p = static_cast<std::byte*>(options_.allocate(static_cast<std::size_t>(here) * store_.record_bytes()));
            if (reinterpret_cast<std::uintptr_t>(p) % DirectReadQueue::kAlignment != 0) {
                options_.release(p);
                throw std::runtime_error("SSD tier: expert slots must be 4 KiB-aligned");
            }
            chunks_.push_back(p);
        }
        queue_ = std::make_unique<DirectReadQueue>(options_.io);
        for (const auto& path : store_.files()) { files_.push_back(queue_->open(path)); }
    } catch (...) {
        for (auto* p : chunks_) { options_.release(p); }
        throw;
    }
    // A layer call reads each of its experts once, so a round demands every key at most once.
    ticket_capacity_ = keys();
    tickets_         = std::make_unique<Ticket[]>(ticket_capacity_);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tier_.begin_round(0);
        for (const std::uint32_t s : tier_.ring()) { ring_.push_back({s, tier_.serial(s)}); }
    }
    agent_ = std::thread([this] { agent_main(); });
}

HostExpertTier::~HostExpertTier() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    if (agent_.joinable()) { agent_.join(); }
    queue_.reset(); // cancels and waits for any read still in flight
    for (auto* p : chunks_) { options_.release(p); }
}

const std::uint8_t* HostExpertTier::slot_bytes(std::uint32_t slot) const noexcept {
    return reinterpret_cast<const std::uint8_t*>(chunks_[slot / per_chunk_]) +
           static_cast<std::size_t>(slot % per_chunk_) * store_.record_bytes();
}

const std::uint8_t* HostExpertTier::record(std::uint32_t key) const noexcept {
    const std::int32_t slot = tier_.slot_of(key);
    return slot >= 0 ? slot_bytes(static_cast<std::uint32_t>(slot)) : nullptr;
}

void HostExpertTier::submit(std::uint32_t key, std::uint32_t slot, std::uint64_t tag_base) {
    const std::uint32_t layer = key / store_.experts(), expert = key % store_.experts();
    auto* destination = const_cast<std::uint8_t*>(slot_bytes(slot));
    const auto segments = store_.segments(layer, expert);
    for (std::size_t i = 0; i < segments.size(); ++i) {
        const auto& s = segments[i];
        queue_->submit({files_[s.file], s.offset, s.bytes, destination + s.at, DirectReadQueue::Priority::Demand,
                        tag_base | i});
    }
}

std::uint32_t HostExpertTier::prefill(std::span<const std::uint32_t> ranked) {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> jobs;
    for (const std::uint32_t key : ranked) {
        std::int32_t slot = -1;
        if (!tier_.prefill(key, slot)) { break; }
        jobs.emplace_back(key, static_cast<std::uint32_t>(slot));
    }
    if (jobs.empty()) { return 0; }
    {
        std::unique_lock<std::mutex> lock(mutex_);
        prefill_jobs_.insert(prefill_jobs_.end(), jobs.begin(), jobs.end());
        prefill_pending_ += static_cast<std::uint32_t>(jobs.size());
        wake_.notify_all();
        prefill_done_.wait(lock, [&] { return prefill_pending_ == 0; });
    }
    if (prefill_failures_ != 0) {
        throw std::runtime_error("SSD tier: " + std::to_string(prefill_failures_) +
                                 " expert records could not be read from the artifact at startup");
    }
    stats_.prefill_reads += jobs.size();
    return static_cast<std::uint32_t>(jobs.size());
}

void HostExpertTier::drain() {
    while (in_flight_.load(std::memory_order_acquire) != 0) { std::this_thread::yield(); }
}

void HostExpertTier::begin_round(std::uint32_t allowance) {
    drain();
    const std::uint32_t count = std::min(ticket_count_.load(std::memory_order_acquire), ticket_capacity_);
    std::vector<expert_cache::Landing> landings;
    for (std::uint32_t t = 0; t < count; ++t) {
        Ticket& ticket = tickets_[t];
        round_ticket_of_key_[ticket.key] = 0;
        if (ticket.state.load(std::memory_order_acquire) == 1 && !ticket.resident && !ticket.recycled) {
            landings.push_back({ticket.slot, ticket.serial, ticket.key});
        }
    }
    const std::size_t admitted = tier_.admit(landings, false);
    stats_.admitted += admitted;
    stats_.discarded += landings.size() - admitted;
    stats_.demand_reads    = agent_reads_.load(std::memory_order_relaxed);
    stats_.demand_failures = agent_failures_.load(std::memory_order_relaxed);
    stats_.read_ns         = agent_read_ns_.load(std::memory_order_relaxed);
    stats_.fetch_requests  = agent_fetch_requests_.load(std::memory_order_relaxed);
    stats_.fetch_records   = agent_fetch_records_.load(std::memory_order_relaxed);
    stats_.fetch_from_ram  = agent_fetch_from_ram_.load(std::memory_order_relaxed);
    tier_.begin_round(allowance);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ring_.clear();
        for (const std::uint32_t s : tier_.ring()) { ring_.push_back({s, tier_.serial(s)}); }
        holding_.clear();
        ticket_count_.store(0, std::memory_order_release);
        ++round_;
        open_ = true;
    }
    wake_.notify_all();
}

void HostExpertTier::end_round() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    open_ = false;
}

std::uint32_t HostExpertTier::new_ticket(std::uint32_t key, std::int32_t fetch) {
    const std::uint32_t t = ticket_count_.fetch_add(1, std::memory_order_acq_rel);
    if (t >= ticket_capacity_) { return kNoTicket; }
    Ticket& ticket = tickets_[t];
    ticket.key      = key;
    ticket.resident = false;
    ticket.recycled = false;
    ticket.fetch    = fetch;
    ticket.done.store(false, std::memory_order_relaxed);
    ticket.status.store(0, std::memory_order_relaxed);
    round_ticket_of_key_[key] = t + 1;
    if (const std::int32_t slot = tier_.slot_of(key); slot >= 0) { // already in RAM: nothing to read
        ticket.slot     = static_cast<std::uint32_t>(slot);
        ticket.resident = true;
        ticket.state.store(1, std::memory_order_release);
        return t;
    }
    ticket.state.store(0, std::memory_order_relaxed);
    in_flight_.fetch_add(1, std::memory_order_acq_rel);
    return t;
}

std::uint32_t HostExpertTier::demand(int layer, int expert) noexcept {
    const std::uint32_t key = this->key(static_cast<std::uint32_t>(layer), static_cast<std::uint32_t>(expert));
    if (const std::uint32_t known = round_ticket_of_key_[key]; known != 0) { return known - 1; }
    const std::uint32_t t = new_ticket(key, -1);
    if (t == kNoTicket || tickets_[t].resident) { return t; }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        new_demands_.push_back(t);
    }
    wake_.notify_all();
    return t;
}

bool HostExpertTier::take_slot(std::uint32_t t) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ring_.empty()) {
        // The oldest holder whose reader is done gives its slot back (its landing is then not admitted).
        for (auto it = holding_.begin(); it != holding_.end(); ++it) {
            Ticket& old = tickets_[*it];
            if (old.done.load(std::memory_order_acquire) && old.state.load(std::memory_order_acquire) != 0) {
                old.recycled = true;
                ring_.push_back({old.slot, old.serial});
                holding_.erase(it);
                break;
            }
        }
        if (ring_.empty()) { return false; }
    }
    Ticket& ticket = tickets_[t];
    ticket.slot    = ring_.back().slot;
    ticket.serial  = ring_.back().serial;
    ring_.pop_back();
    holding_.push_back(t);
    return true;
}

const std::uint8_t* HostExpertTier::wait(std::uint32_t t, std::uint32_t& status) noexcept {
    if (t >= ticket_capacity_) {
        status = ENOBUFS;
        return nullptr;
    }
    Ticket& ticket = tickets_[t];
    for (std::uint32_t spins = 0;; ++spins) {
        const std::uint32_t state = ticket.state.load(std::memory_order_acquire);
        if (state == 1) {
            status = 0;
            return slot_bytes(ticket.slot);
        }
        if (state == 2) {
            status = ticket.status.load(std::memory_order_relaxed);
            return nullptr;
        }
        if (spins < 4096) {
            relax();
        } else {
            std::this_thread::yield();
        }
    }
}

bool HostExpertTier::landed(std::uint32_t t) const noexcept {
    return t >= ticket_capacity_ || tickets_[t].state.load(std::memory_order_acquire) != 0;
}

void HostExpertTier::done(std::uint32_t t) noexcept {
    if (t < ticket_capacity_) { tickets_[t].done.store(true, std::memory_order_release); }
}

void HostExpertTier::agent_main() {
    struct PrefillEntry {
        std::uint32_t key, slot, remaining;
        bool failed;
    };
    std::vector<std::uint32_t> demands;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> prefills;
    std::vector<PrefillEntry> prefill_entries;
    std::uint32_t prefill_left = 0;
    std::vector<DirectReadQueue::Completion> done;
    std::size_t outstanding = 0;        // reads submitted and not completed
    std::deque<std::uint32_t> waiting;  // tickets of this round waiting for a ring slot, in order
    // The current fetch request: its tickets by index, and how many of them the device has released.
    std::vector<std::uint32_t> fetch_ticket;
    std::uint32_t released     = 0;
    bool fetch_failed          = false;
    std::uint64_t round        = 0;
    auto last_work             = Clock::now();
    auto* const fetch          = options_.fetch;
    const auto finish_ticket = [&](Ticket& ticket, std::uint32_t t, bool failed) {
        ticket.state.store(failed ? 2 : 1, std::memory_order_release);
        in_flight_.fetch_sub(1, std::memory_order_acq_rel);
        if (fetch == nullptr || ticket.fetch < 0 || static_cast<std::size_t>(ticket.fetch) >= fetch_ticket.size() ||
            fetch_ticket[static_cast<std::size_t>(ticket.fetch)] != t || fetch_failed) {
            return;
        }
        if (failed) {
            fetch->fail(EIO);
            fetch_failed = true;
        } else {
            fetch->land(static_cast<std::uint32_t>(ticket.fetch), slot_bytes(ticket.slot));
        }
    };
    for (;;) {
        bool open = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (outstanding == 0 && waiting.empty() && !open_) {
                wake_.wait(lock, [&] { return stop_ || open_ || !new_demands_.empty() || !prefill_jobs_.empty(); });
                last_work = Clock::now();
            }
            // Stopping abandons demands still waiting for a slot (their readers are gone).
            if (stop_ && outstanding == 0) { return; }
            demands.swap(new_demands_);
            prefills.swap(prefill_jobs_);
            open = open_;
            if (round_ != round) { // a boundary reset the tickets: the last request is over
                round = round_;
                fetch_ticket.clear();
                released     = 0;
                fetch_failed = false;
            }
        }
        if (fetch != nullptr && open) {
            fetch->beat();
            ops::offloaded_moe::FetchChannel::Request request;
            if (fetch->poll(request)) {
                // A new request means the device has copied every record of the previous one.
                for (const std::uint32_t t : fetch_ticket) {
                    if (t != kNoTicket) { tickets_[t].done.store(true, std::memory_order_release); }
                }
                fetch_ticket.assign(request.experts.size(), kNoTicket);
                released     = 0;
                fetch_failed = false;
                agent_fetch_requests_.fetch_add(1, std::memory_order_relaxed);
                agent_fetch_records_.fetch_add(request.experts.size(), std::memory_order_relaxed);
                for (std::size_t i = 0; i < request.experts.size(); ++i) {
                    const std::uint32_t t = new_ticket(key(static_cast<std::uint32_t>(request.layer),
                                                           static_cast<std::uint32_t>(request.experts[i])),
                                                       static_cast<std::int32_t>(i));
                    if (t == kNoTicket) {
                        fetch->fail(ENOBUFS);
                        fetch_failed = true;
                        break;
                    }
                    fetch_ticket[i] = t;
                    if (tickets_[t].resident) {
                        fetch->land(static_cast<std::uint32_t>(i), slot_bytes(tickets_[t].slot));
                        agent_fetch_from_ram_.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        waiting.push_back(t);
                    }
                }
                last_work = Clock::now();
            }
            // Records the device has copied: their ring slots may be reused.
            const std::uint32_t consumed = std::min<std::uint32_t>(fetch->consumed(), static_cast<std::uint32_t>(fetch_ticket.size()));
            for (; released < consumed; ++released) {
                if (fetch_ticket[released] != kNoTicket) {
                    tickets_[fetch_ticket[released]].done.store(true, std::memory_order_release);
                }
            }
        }
        waiting.insert(waiting.end(), demands.begin(), demands.end());
        demands.clear();
        while (!waiting.empty()) {
            const std::uint32_t t = waiting.front();
            Ticket& ticket        = tickets_[t];
            if (ticket.fetch >= 0 && fetch_failed) { // its request failed: nothing will read it
                waiting.pop_front();
                ticket.status.store(EIO, std::memory_order_relaxed);
                ticket.done.store(true, std::memory_order_release);
                ticket.state.store(2, std::memory_order_release);
                in_flight_.fetch_sub(1, std::memory_order_acq_rel);
                continue;
            }
            if (!take_slot(t)) { break; } // a slot frees up once the device or the CPU is done with one
            waiting.pop_front();
            ticket.submitted_ns = now_ns();
            ticket.remaining    = static_cast<std::uint32_t>(
                store_.segments(ticket.key / store_.experts(), ticket.key % store_.experts()).size());
            submit(ticket.key, ticket.slot, tag_of(kDemandTag, t, 0));
            outstanding += ticket.remaining;
            last_work = Clock::now();
        }
        for (const auto& [key, slot] : prefills) {
            const auto index = static_cast<std::uint32_t>(prefill_entries.size());
            const auto n = static_cast<std::uint32_t>(store_.segments(key / store_.experts(), key % store_.experts()).size());
            prefill_entries.push_back({key, slot, n, false});
            submit(key, slot, tag_of(kPrefillTag, index, 0));
            outstanding += n;
            ++prefill_left;
        }
        prefills.clear();
        done.clear();
        // While a round is open the agent spins (a fetch request must not wait for a timer).
        queue_->poll(done, outstanding != 0 && !open ? std::chrono::microseconds(200) : std::chrono::microseconds(0));
        for (const auto& c : done) {
            --outstanding;
            const std::uint64_t kind = c.tag >> 62U;
            const auto index         = static_cast<std::uint32_t>((c.tag >> 8U) & ((1ULL << 54U) - 1U));
            const bool ok            = c.status == DirectReadQueue::Status::Ok;
            if (kind == kDemandTag) {
                Ticket& ticket = tickets_[index];
                if (!ok) { ticket.status.store(EIO, std::memory_order_relaxed); }
                if (--ticket.remaining == 0) {
                    const bool failed = ticket.status.load(std::memory_order_relaxed) != 0;
                    agent_read_ns_.fetch_add(now_ns() - ticket.submitted_ns, std::memory_order_relaxed);
                    (failed ? agent_failures_ : agent_reads_).fetch_add(1, std::memory_order_relaxed);
                    finish_ticket(ticket, index, failed);
                }
            } else {
                PrefillEntry& entry = prefill_entries[index];
                entry.failed        = entry.failed || !ok;
                if (--entry.remaining == 0) {
                    if (entry.failed) { ++prefill_failures_; }
                    if (--prefill_left == 0) {
                        prefill_entries.clear();
                        std::lock_guard<std::mutex> lock(mutex_);
                        prefill_pending_ = 0;
                        prefill_done_.notify_all();
                    }
                }
            }
        }
        if (!done.empty()) { last_work = Clock::now(); }
        if (open && done.empty()) {
            if (Clock::now() - last_work > kIdleSpin) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
                relax();
            }
        } else if (!open && done.empty() && outstanding == 0 && !waiting.empty()) {
            std::this_thread::yield(); // tickets of a closed round wait for a reader to finish
        }
    }
}

} // namespace ninfer::models::qwen4_exp
