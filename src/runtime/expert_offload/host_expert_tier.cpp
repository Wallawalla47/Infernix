#include "runtime/expert_offload/host_expert_tier.h"

#include "core/cpu_topology.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(__x86_64__) || defined(_M_X64)
#    include <immintrin.h>
#endif

namespace infernix::runtime::expert_offload {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kDemandTag     = 0;
constexpr std::uint64_t kPrefillTag    = 1;
constexpr std::uint64_t kDemandRunTag  = 2; // index: a run of tickets
constexpr std::uint64_t kPrefillRunTag = 3; // index: a run of prefill entries
// Records that follow each other in a file are read as one scatter read (Strata 0.1.41's batched
// file-tier reads): on this machine's NVMe drive (Samsung 990 PRO), unbuffered 33 MB reads of
// neighbouring records reached 7.1 GB/s, against 6.7 GB/s for separate 2.6 MB record reads at
// depth 8 (RM0e). The startup fill takes long runs; a fetch request's runs stay short, so its
// first record lands soon.
constexpr std::size_t kFillRunBytes  = std::size_t{32} << 20;
constexpr std::size_t kFetchRunBytes = std::size_t{12} << 20;
// Read-ahead (Options::ahead_min, ahead) keeps the drive reading while the device finishes a layer
// and routes the next. Decode calls (a few records each) never read ahead.
constexpr std::int32_t kAheadFetch = -2; // Ticket::fetch of a read-ahead ticket not yet adopted
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

// Null when the OS refuses: on Windows page-locking stops at the adapter's shared system memory.
void* pinned_allocate(std::size_t bytes) {
    void* p = nullptr;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess) {
        (void)cudaGetLastError();
        return nullptr;
    }
    return p;
}

void pinned_release(void* p) { (void)cudaFreeHost(p); }

// The tier's agent (it spins while a round is open) and copier threads run on the CPU's slowest cores (a hybrid CPU's
// E-cores), leaving the fast cores to the CPU expert team (i9-13900K: 8 P-cores for 6 workers, the service, the engine
// thread and the agent). Dense8's exclusive layout 0.986x against 0.980x with them unpinned, the tier holding every
// expert 0.977x against 0.973x; IQ3_S unchanged (x_ab13, 2026-10-27). No change on a CPU of one class.
void place_on_slow_cores() noexcept {
    static const std::uint64_t slow = [] {
        const std::vector<PhysicalCore> cores = physical_cores();
        if (cores.empty()) { return std::uint64_t{0}; }
        std::uint8_t lowest = 255, highest = 0;
        for (const PhysicalCore& core : cores) {
            lowest  = std::min(lowest, core.efficiency);
            highest = std::max(highest, core.efficiency);
        }
        std::uint64_t mask = 0;
        if (lowest != highest) {
            for (const PhysicalCore& core : cores) { mask |= core.efficiency == lowest ? core.cpus : 0; }
        }
        return mask;
    }();
    if (slow != 0) { (void)set_current_thread_cpus(slow); }
}


} // namespace

HostExpertTier::Pinned::Pinned(Pinned&& other) noexcept
    : chunks(std::move(other.chunks)), regions(std::move(other.regions)), slots(other.slots), refused(other.refused),
      bytes(other.bytes), release(std::move(other.release)) {
    other.chunks.clear();
}

HostExpertTier::Pinned::~Pinned() {
    for (auto* p : chunks) { release(p); }
}

HostExpertTier::Options HostExpertTier::normalized(Options options) {
    if (!options.allocate) { options.allocate = pinned_allocate; }
    if (!options.release) { options.release = pinned_release; }
    if (options.slots_per_chunk == 0) { throw std::invalid_argument("SSD tier: slots per chunk must be positive"); }
    if (options.min_slots == 0) { options.min_slots = options.ring + options.prefetch + options.demotion + 1; }
    if (options.copy_threads == 0) {
        options.copy_threads = std::clamp(std::thread::hardware_concurrency() / 3, 2U, 8U);
    }
    // Two halves, at least one slot each: unpinned records promote through them, and the bounced
    // promotion loop advances by a half (ExpertResidency, zero would never advance).
    options.bounce        = std::max<std::uint32_t>(2, options.bounce / 2 * 2);
    options.stream_bounce = options.stream_bounce / 2 * 2;
    return options;
}

namespace {

// The layers' record sizes, largest first (a class per size).
std::vector<std::uint64_t> class_pitches(const ExpertStore& store) {
    std::vector<std::uint64_t> pitches;
    for (std::uint32_t l = 0; l < store.layers(); ++l) { pitches.push_back(store.record_bytes(l)); }
    std::sort(pitches.begin(), pitches.end(), std::greater<>());
    pitches.erase(std::unique(pitches.begin(), pitches.end()), pitches.end());
    return pitches;
}

// A class's demotion list: its layers' share of Options::demotion, at least one slot.
std::uint32_t class_demotion(std::uint32_t demotion, std::uint32_t class_layers, std::uint32_t layers) {
    return std::max<std::uint32_t>(1, (demotion * class_layers + layers - 1) / layers);
}

} // namespace

std::uint32_t HostExpertTier::slots_for_bytes(const ExpertStore& store, std::uint64_t bytes, const Options& raw) {
    const Options options       = normalized(raw);
    const std::uint64_t keys    = static_cast<std::uint64_t>(store.layers()) * store.experts();
    const auto pitches          = class_pitches(store);
    if (pitches.size() <= 1) {
        const std::uint64_t lists = static_cast<std::uint64_t>(options.ring) + options.prefetch + options.demotion;
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(bytes / store.record_bytes(), keys + lists));
    }
    const std::uint64_t transit =
        static_cast<std::uint64_t>(options.ring) + options.prefetch + options.bounce + options.stream_bounce;
    const std::uint64_t transit_bytes = transit * store.record_bytes();
    if (bytes <= transit_bytes) { return static_cast<std::uint32_t>(transit); }
    // Each layer gets the same share of its class's slots, so the class slots cost the layers' mean record size.
    std::uint64_t stride_sum = 0, demotions = 0;
    for (std::uint32_t l = 0; l < store.layers(); ++l) { stride_sum += store.record_bytes(l); }
    for (const std::uint64_t pitch : pitches) {
        std::uint32_t layers_here = 0;
        for (std::uint32_t l = 0; l < store.layers(); ++l) { layers_here += store.record_bytes(l) == pitch ? 1U : 0U; }
        demotions += class_demotion(options.demotion, layers_here, store.layers());
    }
    const std::uint64_t class_slots = (bytes - transit_bytes) * store.layers() / stride_sum;
    return static_cast<std::uint32_t>(transit + std::min(class_slots, keys + demotions));
}

HostExpertTier::Plan HostExpertTier::plan_slots(const ExpertStore& store, const Options& options) {
    if (store.record_bytes() % DirectReadQueue::kAlignment != 0) {
        throw std::invalid_argument("SSD tier: expert records must be a multiple of 4 KiB");
    }
    Plan plan;
    const std::uint32_t layers = store.layers(), experts = store.experts();
    const auto pitches         = class_pitches(store);
    if (pitches.size() <= 1) {
        plan.regions = {{store.record_bytes(), options.slots}};
        plan.layer_class.assign(layers, 0);
        plan.class_layers = {layers};
        return plan;
    }
    plan.layer_class.resize(layers);
    plan.class_layers.assign(pitches.size(), 0);
    for (std::uint32_t l = 0; l < layers; ++l) {
        const auto c = static_cast<std::uint8_t>(std::find(pitches.begin(), pitches.end(), store.record_bytes(l)) - pitches.begin());
        plan.layer_class[l] = c;
        ++plan.class_layers[c];
    }
    plan.transit                = options.bounce + options.stream_bounce;
    const std::uint64_t transit = static_cast<std::uint64_t>(options.ring) + options.prefetch + plan.transit;
    if (options.slots <= transit) {
        throw std::invalid_argument("SSD tier: the slots must exceed the ring, prefetch and bounce slots");
    }
    std::vector<std::uint32_t> demotion(pitches.size());
    std::uint64_t demotions = 0;
    for (std::size_t c = 0; c < pitches.size(); ++c) {
        demotion[c] = class_demotion(options.demotion, plan.class_layers[c], layers);
        demotions += demotion[c];
    }
    if (options.slots - transit <= demotions + pitches.size()) {
        throw std::invalid_argument("SSD tier: every record size needs a resident slot beyond its demotion list");
    }
    // Every layer the same share of resident slots (as each layer gets the same share of the VRAM
    // frames), then the rest one more slot per layer, class by class, while its layers have keys left.
    std::uint64_t left            = options.slots - transit - demotions;
    const std::uint64_t per_layer = std::min<std::uint64_t>(experts, left / layers);
    std::vector<std::uint64_t> residents(pitches.size());
    for (std::size_t c = 0; c < pitches.size(); ++c) {
        residents[c] = per_layer * plan.class_layers[c];
        left -= residents[c];
    }
    for (std::size_t c = 0; c < pitches.size() && left > 0 && per_layer < experts; ++c) {
        const std::uint64_t more = std::min<std::uint64_t>(left, plan.class_layers[c]);
        residents[c] += more;
        left -= more;
    }
    plan.regions.push_back({pitches[0], static_cast<std::uint32_t>(transit)});
    for (std::size_t c = 0; c < pitches.size(); ++c) {
        const auto slots = static_cast<std::uint32_t>(residents[c] + demotion[c]);
        plan.regions.push_back({pitches[c], slots});
        plan.classes.push_back({.slots = slots, .demotion = demotion[c], .transit_pitch = pitches[c] == pitches[0]});
    }
    return plan;
}

HostExpertTier::Pinned HostExpertTier::pin_slots(const Plan& plan, const Options& options) {
    Pinned out;
    out.release = options.release;
    out.regions.resize(plan.regions.size());
    struct Piece {
        std::uint32_t region, slots;
        std::uint64_t bytes;
    };
    std::vector<Piece> pieces;
    std::vector<std::uint32_t> next(plan.regions.size(), 0);
    // 1: a chunk of region r was pinned; 0: the region is complete; -1: the OS refused.
    const auto pin_chunk = [&](std::uint32_t r) {
        const Region& region = plan.regions[r];
        if (next[r] >= region.slots) { return 0; }
        const std::uint32_t here  = std::min(options.slots_per_chunk, region.slots - next[r]);
        const std::uint64_t bytes = static_cast<std::uint64_t>(here) * region.pitch;
        auto* p = static_cast<std::byte*>(options.allocate(static_cast<std::size_t>(bytes)));
        if (p == nullptr) { return -1; }
        if (reinterpret_cast<std::uintptr_t>(p) % DirectReadQueue::kAlignment != 0) {
            options.release(p);
            throw std::runtime_error("SSD tier: expert slots must be 4 KiB-aligned");
        }
        out.chunks.push_back(p);
        pieces.push_back({r, here, bytes});
        for (std::uint32_t i = 0; i < here; ++i) {
            out.regions[r].push_back(reinterpret_cast<std::uint8_t*>(p) + static_cast<std::size_t>(i) * region.pitch);
        }
        next[r] += here;
        out.slots += here;
        out.bytes += bytes;
        return 1;
    };
    // Region 0 first (the transit slots, or the one region), then the classes' chunks in turn, so a
    // refusal leaves every class a like share.
    bool refused = false;
    for (int got = 1; got == 1;) {
        got     = pin_chunk(0);
        refused = got < 0;
    }
    for (bool more = !refused; more && !refused;) {
        more = false;
        for (std::uint32_t r = 1; r < plan.regions.size() && !refused; ++r) {
            const int got = pin_chunk(r);
            refused       = got < 0;
            more          = more || got == 1;
        }
    }
    if (refused) {
        std::uint64_t planned = 0;
        for (const Region& region : plan.regions) { planned += region.slots; }
        // Whatever is pinned after the tier (the prefix cache's Host tier, the Program's buffers)
        // must still fit below the limit the OS just reached.
        std::uint64_t freed = 0;
        while (!out.chunks.empty() && freed < options.pin_reserve) {
            options.release(out.chunks.back());
            out.chunks.pop_back();
            const Piece piece = pieces.back();
            pieces.pop_back();
            out.regions[piece.region].resize(out.regions[piece.region].size() - piece.slots);
            freed += piece.bytes;
            out.slots -= piece.slots;
            out.bytes -= piece.bytes;
        }
        out.refused = static_cast<std::uint32_t>(planned - out.slots);
        // The refused slots become unpinned records, whose promotions need the bounce slots.
        const std::uint32_t needed = options.min_slots + options.bounce + options.stream_bounce;
        const std::uint64_t largest = plan.regions[0].pitch;
        if (out.slots < needed || (plan.regions.size() > 1 && out.regions[0].size() < plan.regions[0].slots)) {
            throw std::runtime_error("SSD tier: the OS let Infernix page-lock only " + std::to_string(out.bytes >> 20) +
                                     " MiB of expert slots once " + std::to_string(options.pin_reserve >> 20) +
                                     " MiB are left for later pinned buffers; at least " +
                                     std::to_string((static_cast<std::uint64_t>(needed) * largest) >> 20) +
                                     " MiB are needed (Windows limits page-locked memory to the GPU's shared memory: "
                                     "close programs that use it, or lower --host-context-mib)");
        }
    }
    return out;
}

expert_cache::HostTier::Config HostExpertTier::controller_config() const {
    expert_cache::HostTier::Config config{.num_keys  = store_.layers() * store_.experts(),
                                          .ring      = options_.ring,
                                          .prefetch  = options_.prefetch,
                                          .demotion  = options_.demotion,
                                          .half_life = options_.half_life};
    if (plan_.classes.empty()) {
        config.slots        = pinned_.slots - bounce_ - stream_bounce_;
        config.shadow_floor = options_.shadow_floor;
        return config;
    }
    config.key_class.resize(config.num_keys);
    for (std::uint32_t key = 0; key < config.num_keys; ++key) {
        config.key_class[key] = plan_.layer_class[key / store_.experts()];
    }
    config.slots = options_.ring + options_.prefetch;
    for (std::size_t c = 0; c < plan_.classes.size(); ++c) {
        auto pinned = plan_.classes[c];
        pinned.slots = static_cast<std::uint32_t>(pinned_.regions[1 + c].size()); // less any refused chunks
        pinned.shadow_floor = static_cast<std::uint32_t>(static_cast<std::uint64_t>(options_.shadow_floor) *
                                                         plan_.class_layers[c] / store_.layers());
        if (pinned.slots <= pinned.demotion) {
            throw std::runtime_error("SSD tier: too few expert slots of one record size could be page-locked");
        }
        config.slots += pinned.slots;
        config.classes.push_back(pinned);
    }
    return config;
}

HostExpertTier::HostExpertTier(const ExpertStore& store, Options options)
    : store_(store), options_(normalized(std::move(options))), plan_(plan_slots(store_, options_)),
      pinned_(pin_slots(plan_, options_)),
      bounce_(options_.unpinned > 0 || pinned_.refused > 0 ? options_.bounce : 0),
      stream_bounce_(options_.unpinned > 0 || pinned_.refused > 0 ? options_.stream_bounce : 0),
      tier_(controller_config()), unpinned_of_(static_cast<std::size_t>(store.layers()) * store.experts(), -1),
      round_ticket_of_key_(static_cast<std::size_t>(store.layers()) * store.experts(), 0) {
    // The controller's slots in its order (the ring, the prefetch list, then each class's), and the
    // bounce slots: the one region's last slots, or region 0's after the ring and the prefetch list.
    std::uint32_t resident = 0;
    if (plan_.classes.empty()) {
        const auto& all                = pinned_.regions[0];
        const std::uint32_t controller = pinned_.slots - bounce_ - stream_bounce_;
        slot_address_.assign(all.begin(), all.begin() + controller);
        bounce_address_.assign(all.begin() + controller, all.end());
        resident = controller - (options_.ring + options_.prefetch + options_.demotion);
    } else {
        const auto& transit       = pinned_.regions[0];
        const std::uint32_t lists = options_.ring + options_.prefetch;
        slot_address_.assign(transit.begin(), transit.begin() + lists);
        bounce_address_.assign(transit.begin() + lists, transit.end());
        for (std::size_t c = 0; c < plan_.classes.size(); ++c) {
            const auto& region = pinned_.regions[1 + c];
            slot_address_.insert(slot_address_.end(), region.begin(), region.end());
            resident += static_cast<std::uint32_t>(region.size()) - plan_.classes[c].demotion;
        }
    }
    stream_free_.assign(tier_.class_count(), {});
    stream_lent_.assign(tier_.class_count(), 0);
    // Unpinned records: those asked for and the slots the OS refused, at most the keys the
    // resident slots leave.
    const std::uint32_t uncovered = keys() > resident ? keys() - resident : 0;
    unpinned_count_ = std::min<std::uint32_t>(options_.unpinned + pinned_.refused, uncovered);
    if (unpinned_count_ > 0) {
        unpinned_ = std::make_unique<ResidentHostBuffer>(static_cast<std::size_t>(unpinned_count_) * store_.record_bytes());
    }
    queue_ = std::make_unique<DirectReadQueue>(options_.io);
    for (const auto& path : store_.files()) { files_.push_back(queue_->open(path)); }
    // A layer call reads each of its experts once, so a round demands at most its layer calls'
    // experts (every key once when a round calls each layer once), and each layer call's request
    // may leave Options::ahead read-ahead tickets unadopted.
    const std::uint32_t calls = options_.layer_calls > 0 ? options_.layer_calls : store_.layers();
    ticket_capacity_ = (options_.layer_calls > 0 ? options_.layer_calls * store_.experts() : keys()) + calls * options_.ahead;
    tickets_         = std::make_unique<Ticket[]>(ticket_capacity_);
    stream_state_    = std::make_unique<std::atomic<std::uint32_t>[]>(std::max<std::uint32_t>(options_.stream, 1));
    refill_state_    = std::make_unique<std::atomic<std::uint32_t>[]>(kRefills);
    for (std::uint32_t id = kRefills; id > 0; --id) { refill_free_.push_back(id - 1); }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tier_.begin_round(0);
        for (const std::uint32_t s : tier_.ring()) { ring_.push_back({s, tier_.serial(s)}); }
    }
    if (unpinned_count_ > 0) {
        for (std::uint32_t i = 0; i < std::max<std::uint32_t>(1, options_.copy_threads); ++i) {
            copiers_.emplace_back([this] { copier_main(); });
        }
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
    {
        std::lock_guard<std::mutex> lock(copy_mutex_);
        copy_stop_ = true;
    }
    copy_wake_.notify_all();
    for (auto& copier : copiers_) { copier.join(); }
    queue_.reset(); // cancels and waits for any read still in flight
}

const std::string& HostExpertTier::unpinned_lock_failure() const noexcept {
    static const std::string none;
    return unpinned_ ? unpinned_->lock_failure() : none;
}

const std::uint8_t* HostExpertTier::slot_bytes(std::uint32_t slot) const noexcept { return slot_address_[slot]; }

const std::uint8_t* HostExpertTier::record(std::uint32_t key) const noexcept {
    const std::int32_t slot = tier_.slot_of(key);
    return slot >= 0 ? slot_bytes(static_cast<std::uint32_t>(slot)) : nullptr;
}

void HostExpertTier::submit(std::uint32_t key, std::uint8_t* destination, std::uint64_t tag_base,
                            DirectReadQueue::Priority priority) {
    const std::uint32_t layer = key / store_.experts(), expert = key % store_.experts();
    const auto segments = store_.segments(layer, expert);
    for (std::size_t i = 0; i < segments.size(); ++i) {
        const auto& s = segments[i];
        queue_->submit({files_[s.file], s.offset, s.bytes, destination + s.at, priority, tag_base | i});
    }
}

// Whole records that follow each other in one file. A record split across files (one that straddles
// part files, or an expert stripe's head and tail) is read segment by segment.
bool HostExpertTier::follows(std::uint32_t before, std::uint32_t key) const {
    const auto a = store_.segments(before / store_.experts(), before % store_.experts());
    const auto b = store_.segments(key / store_.experts(), key % store_.experts());
    return a.size() == 1 && b.size() == 1 && a[0].file == b[0].file && a[0].offset + a[0].bytes == b[0].offset;
}

void HostExpertTier::submit_run(std::span<const std::pair<std::uint32_t, std::uint8_t*>> run, std::uint64_t tag) {
    DirectReadQueue::Read read;
    const auto first = store_.segments(run.front().first / store_.experts(), run.front().first % store_.experts())[0];
    read.file   = files_[first.file];
    read.offset = first.offset;
    read.tag    = tag;
    for (const auto& [key, destination] : run) {
        const auto s = store_.segments(key / store_.experts(), key % store_.experts())[0];
        read.scatter.push_back({destination + s.at, s.bytes});
        read.bytes += s.bytes;
    }
    queue_->submit(read);
    agent_run_reads_.fetch_add(1, std::memory_order_relaxed);
    agent_run_records_.fetch_add(run.size(), std::memory_order_relaxed);
}

void HostExpertTier::read_at_startup(const std::vector<std::pair<std::uint32_t, std::uint8_t*>>& jobs) {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!agent_failure_.empty()) { throw std::runtime_error("SSD tier: the read agent stopped: " + agent_failure_); }
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
    stats_.run_reads   = agent_run_reads_.load(std::memory_order_relaxed);
    stats_.run_records = agent_run_records_.load(std::memory_order_relaxed);
}

std::uint32_t HostExpertTier::prefill(std::span<const std::uint32_t> ranked,
                                      const std::function<bool(std::uint32_t)>& skip) {
    std::vector<std::pair<std::uint32_t, std::uint8_t*>> jobs;
    for (const std::uint32_t key : ranked) {
        if (!tier_.has_free_resident()) { break; }
        if (skip && skip(key)) { continue; }
        std::int32_t slot = -1;
        if (!tier_.prefill(key, slot)) { continue; } // already has a host copy
        jobs.emplace_back(key, const_cast<std::uint8_t*>(slot_bytes(static_cast<std::uint32_t>(slot))));
    }
    if (jobs.empty()) { return 0; }
    const auto start = std::chrono::steady_clock::now();
    read_at_startup(jobs);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (seconds > 0) { read_rate_ = static_cast<double>(jobs.size()) * static_cast<double>(store_.record_bytes()) / seconds; }
    return static_cast<std::uint32_t>(jobs.size());
}

bool HostExpertTier::refill_within(double seconds) const noexcept {
    if (read_rate_ <= 0) { return true; }
    const double queued = static_cast<double>(kRefills - refill_free_.size()) + 1.0;
    return queued * static_cast<double>(store_.record_bytes()) / read_rate_ <= seconds;
}

std::uint32_t HostExpertTier::prefill_shadows(std::span<const std::uint32_t> ranked) {
    std::vector<std::pair<std::uint32_t, std::uint8_t*>> jobs;
    for (const std::uint32_t key : ranked) {
        if (!tier_.has_free_resident()) { break; }
        std::int32_t slot = -1;
        if (!tier_.prefill_shadow(key, slot)) { continue; } // already has a host copy
        jobs.emplace_back(key, const_cast<std::uint8_t*>(slot_bytes(static_cast<std::uint32_t>(slot))));
    }
    if (jobs.empty()) { return 0; }
    read_at_startup(jobs);
    return static_cast<std::uint32_t>(jobs.size());
}

std::uint32_t HostExpertTier::fill_unpinned(std::span<const std::uint32_t> ranked,
                                            const std::function<bool(std::uint32_t)>& skip) {
    std::vector<std::pair<std::uint32_t, std::uint8_t*>> jobs;
    for (const std::uint32_t key : ranked) {
        if (unpinned_filled_ == unpinned_count_) { break; }
        if (tier_.host_copy(key) || tier_.held(key) || (skip && skip(key))) { continue; }
        const std::uint32_t index = unpinned_filled_++;
        unpinned_of_[key]         = static_cast<std::int32_t>(index);
        (void)tier_.hold(key);
        jobs.emplace_back(key, reinterpret_cast<std::uint8_t*>(unpinned_->data()) +
                                   static_cast<std::size_t>(index) * store_.record_bytes());
    }
    if (jobs.empty()) { return 0; }
    read_at_startup(jobs);
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
        // A copied unpinned record is not admitted: its key keeps the unpinned record.
        if (ticket.state.load(std::memory_order_acquire) == 1 && !ticket.resident && !ticket.recycled &&
            ticket.copy_from == nullptr) {
            landings.push_back({ticket.slot, ticket.serial, ticket.key});
        }
    }
    const std::size_t admitted = tier_.admit(landings, false);
    stats_.admitted += admitted;
    stats_.discarded += landings.size() - admitted;
    // Landings admitted into a class of smaller records are copied there from their ring slots (the
    // agent is drained, and the ring is handed out again only below).
    if (!tier_.copies().empty()) {
        std::vector<Copy> batch;
        for (const expert_cache::AdmissionCopy& a : tier_.copies()) {
            batch.push_back({.destination = slot_address_[a.to],
                             .source      = slot_address_[a.from],
                             .bytes       = static_cast<std::size_t>(store_.record_bytes(a.key / store_.experts()))});
        }
        run_copies(batch);
        tier_.finish_copies();
        stats_.copied_admissions += batch.size();
    }
    // The demotion list refills at every boundary (design §19.3.7 §4.8): without it the first 32
    // demotions used it up, after which every VRAM victim without a host copy was dropped to the
    // SSD and the eviction gate refused demotion-worthy victims outright (T6).
    tier_.replenish_demotion_list();
    stats_.demand_reads    = agent_reads_.load(std::memory_order_relaxed);
    stats_.demand_failures = agent_failures_.load(std::memory_order_relaxed);
    stats_.read_ns         = agent_read_ns_.load(std::memory_order_relaxed);
    stats_.fetch_requests  = agent_fetch_requests_.load(std::memory_order_relaxed);
    stats_.fetch_records   = agent_fetch_records_.load(std::memory_order_relaxed);
    stats_.fetch_from_ram  = agent_fetch_from_ram_.load(std::memory_order_relaxed);
    stats_.run_reads       = agent_run_reads_.load(std::memory_order_relaxed);
    stats_.run_records     = agent_run_records_.load(std::memory_order_relaxed);
    stats_.stream_reads    = agent_stream_reads_.load(std::memory_order_relaxed);
    stats_.unpinned_served = unpinned_served_.load(std::memory_order_relaxed);
    stats_.unpinned_copies = unpinned_copies_.load(std::memory_order_relaxed);
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
    ticket.key       = key;
    ticket.resident  = false;
    ticket.recycled  = false;
    ticket.fetch     = fetch;
    ticket.copy_from = nullptr;
    ticket.bytes     = nullptr;
    ticket.done.store(false, std::memory_order_relaxed);
    ticket.status.store(0, std::memory_order_relaxed);
    round_ticket_of_key_[key] = t + 1;
    if (const std::int32_t slot = tier_.slot_of(key); slot >= 0) { // already in RAM: nothing to read
        ticket.slot     = static_cast<std::uint32_t>(slot);
        ticket.bytes    = slot_bytes(ticket.slot);
        ticket.resident = true;
        ticket.state.store(1, std::memory_order_release);
        return t;
    }
    if (const std::uint8_t* unpinned = unpinned_record(key); unpinned != nullptr) {
        if (fetch < 0) { // the CPU reads it in place
            ticket.bytes    = unpinned;
            ticket.resident = true;
            unpinned_served_.fetch_add(1, std::memory_order_relaxed);
            ticket.state.store(1, std::memory_order_release);
            return t;
        }
        ticket.copy_from = unpinned; // the device reads pinned memory only: a ring slot takes a copy
    }
    ticket.state.store(0, std::memory_order_relaxed);
    in_flight_.fetch_add(1, std::memory_order_acq_rel);
    return t;
}

std::uint32_t HostExpertTier::lend_stream(std::uint32_t n) {
    const std::uint32_t total = std::min(n, options_.stream);
    // A class's share of the loan by its layers, as its RAM is (each layer the same share): the loan takes the
    // free slots the RAM plan left for it instead of evicting RAM-only experts.
    std::vector<std::uint32_t> per_class(tier_.class_count(), 0);
    for (std::size_t c = 0; c < per_class.size(); ++c) {
        per_class[c] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(total) * plan_.class_layers[c] / store_.layers());
    }
    const std::vector<std::uint32_t> slots = tier_.lend(per_class);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stream_slot_.empty()) { throw std::logic_error("SSD tier: stream slots are lent already"); }
    stream_slot_ = slots;
    stream_class_.resize(slots.size());
    for (auto& free : stream_free_) { free.clear(); }
    std::fill(stream_lent_.begin(), stream_lent_.end(), 0U);
    for (std::uint32_t s = static_cast<std::uint32_t>(slots.size()); s > 0; --s) {
        const std::uint8_t c = tier_.slot_class(slots[s - 1]);
        stream_class_[s - 1] = c;
        stream_free_[c].push_back(s - 1);
        ++stream_lent_[c];
    }
    return static_cast<std::uint32_t>(slots.size());
}

void HostExpertTier::return_stream() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stream_slot_.empty()) { return; }
        std::size_t free = 0;
        for (const auto& f : stream_free_) { free += f.size(); }
        if (free != stream_slot_.size()) { throw std::logic_error("SSD tier: a stream slot is still in use"); }
        stream_slot_.clear();
        for (auto& f : stream_free_) { f.clear(); }
        std::fill(stream_lent_.begin(), stream_lent_.end(), 0U);
    }
    tier_.give_back();
}

std::uint32_t HostExpertTier::stream_layer_cap(std::uint32_t layer) const noexcept {
    // Per layer, by its class: the least over every class left every layer without a read whenever one class lent
    // nothing.
    std::lock_guard<std::mutex> lock(mutex_);
    if (stream_lent_.empty() || layer >= plan_.layer_class.size()) { return 0; }
    const std::uint8_t c = plan_.layer_class[layer];
    return stream_lent_[c] / std::max<std::uint32_t>(1, std::min(plan_.class_layers[c], kStreamWindow));
}

std::int32_t HostExpertTier::stream_read(std::uint32_t key) {
    std::int32_t slot = -1;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& free = stream_free_[tier_.class_of(key)];
        if (!agent_failure_.empty() || free.empty()) { return -1; }
        slot = static_cast<std::int32_t>(free.back());
        free.pop_back();
        stream_state_[slot].store(0, std::memory_order_relaxed);
        stream_jobs_.emplace_back(key, static_cast<std::uint32_t>(slot));
    }
    wake_.notify_all();
    return slot;
}

const std::uint8_t* HostExpertTier::stream_wait(std::int32_t slot) noexcept {
    for (std::uint32_t spins = 0;; ++spins) {
        const std::uint32_t state = stream_state_[slot].load(std::memory_order_acquire);
        if (state == 1) { return slot_bytes(stream_slot_[static_cast<std::size_t>(slot)]); }
        if (state == 2) { return nullptr; }
        if (spins < 4096) {
            relax();
        } else {
            std::this_thread::yield();
        }
    }
}

std::uint32_t HostExpertTier::refill(std::uint32_t key, std::uint32_t slot) {
    if (refill_free_.empty()) { return kNoRefill; }
    const std::uint32_t id = refill_free_.back();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!agent_failure_.empty()) { return kNoRefill; } // nothing would read it: the caller copies the frame
        refill_free_.pop_back();
        refill_state_[id].store(0, std::memory_order_relaxed);
        refill_jobs_.push_back({key, slot, id});
    }
    wake_.notify_all();
    return id;
}

void HostExpertTier::stream_release(std::int32_t slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    stream_free_[stream_class_[static_cast<std::size_t>(slot)]].push_back(static_cast<std::uint32_t>(slot));
}

std::uint32_t HostExpertTier::demand(int layer, int expert) noexcept {
    const std::uint32_t key = this->key(static_cast<std::uint32_t>(layer), static_cast<std::uint32_t>(expert));
    if (const std::uint32_t known = round_ticket_of_key_[key]; known != 0) { return known - 1; }
    const std::uint32_t t = new_ticket(key, -1);
    if (t == kNoTicket || tickets_[t].resident) { return t; }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!agent_failure_.empty()) { // nothing will read it
            fail_ticket(t);
            return t;
        }
        new_demands_.push_back(t);
    }
    wake_.notify_all();
    return t;
}

void HostExpertTier::fail_ticket(std::uint32_t t) noexcept {
    Ticket& ticket          = tickets_[t];
    std::uint32_t pending   = 0;
    if (!ticket.state.compare_exchange_strong(pending, 2, std::memory_order_acq_rel)) { return; }
    ticket.status.store(EIO, std::memory_order_relaxed);
    ticket.done.store(true, std::memory_order_release);
    in_flight_.fetch_sub(1, std::memory_order_acq_rel);
}

std::string HostExpertTier::failure() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return agent_failure_;
}

void HostExpertTier::finish_copy(const Copy& copy) {
    std::memcpy(copy.destination, copy.source, copy.bytes != 0 ? copy.bytes : store_.record_bytes());
    std::lock_guard<std::mutex> lock(copy_mutex_);
    if (copy.left == nullptr) {
        copied_.push_back(copy.tag);
    } else if (--*copy.left == 0) {
        copy_done_.notify_all();
    }
}

void HostExpertTier::copier_main() noexcept {
    place_on_slow_cores();
    for (;;) {
        Copy copy;
        {
            std::unique_lock<std::mutex> lock(copy_mutex_);
            copy_wake_.wait(lock, [&] { return copy_stop_ || !copies_.empty(); });
            if (copies_.empty()) { return; } // stopping
            copy = copies_.front();
            copies_.pop_front();
        }
        finish_copy(copy);
    }
}

void HostExpertTier::copy_records(std::span<const std::pair<std::uint8_t*, const std::uint8_t*>> copies) {
    std::vector<Copy> batch;
    batch.reserve(copies.size());
    for (const auto& [destination, source] : copies) { batch.push_back({.destination = destination, .source = source}); }
    run_copies(batch);
}

void HostExpertTier::run_copies(std::vector<Copy>& batch) {
    if (batch.empty()) { return; }
    std::uint32_t left = static_cast<std::uint32_t>(batch.size());
    {
        std::lock_guard<std::mutex> lock(copy_mutex_);
        for (Copy& copy : batch) {
            copy.left = &left;
            copies_.push_back(copy);
        }
    }
    copy_wake_.notify_all();
    // This thread copies too, then waits for the copies the threads took.
    for (;;) {
        Copy copy;
        {
            std::unique_lock<std::mutex> lock(copy_mutex_);
            if (copies_.empty()) {
                copy_done_.wait(lock, [&] { return left == 0; });
                return;
            }
            copy = copies_.front();
            copies_.pop_front();
        }
        finish_copy(copy);
    }
}

void HostExpertTier::agent_main() noexcept {
    place_on_slow_cores();
    try {
        agent_loop();
    } catch (const std::exception& error) {
        agent_failed(error.what());
    } catch (...) {
        agent_failed("unknown exception");
    }
}

void HostExpertTier::agent_failed(std::string what) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        agent_failure_ = what.empty() ? std::string("unknown exception") : std::move(what);
        // Reads in flight may still land in their slots; nothing admits a failed ticket's slot.
        const std::uint32_t count = std::min(ticket_count_.load(std::memory_order_acquire), ticket_capacity_);
        for (std::uint32_t t = 0; t < count; ++t) { fail_ticket(t); }
        new_demands_.clear();
        prefill_jobs_.clear();
        stream_jobs_.clear();
        for (const RefillJob& job : refill_jobs_) { refill_state_[job.id].store(2, std::memory_order_release); }
        refill_jobs_.clear();
        for (std::uint32_t s = 0; s < options_.stream; ++s) { // pending stream reads fail
            std::uint32_t pending = 0;
            stream_state_[s].compare_exchange_strong(pending, 2, std::memory_order_acq_rel);
        }
        if (prefill_pending_ != 0) {
            prefill_failures_ += prefill_pending_;
            prefill_pending_ = 0;
            prefill_done_.notify_all();
        }
    } catch (...) {}
    if (options_.fetch != nullptr) { options_.fetch->fail(EIO); }
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
            return ticket.bytes;
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

void HostExpertTier::agent_loop() {
    struct PrefillEntry {
        std::uint32_t key, remaining;
        bool failed;
    };
    std::vector<std::uint32_t> demands;
    std::vector<std::pair<std::uint32_t, std::uint8_t*>> prefills;
    std::vector<PrefillEntry> prefill_entries;
    std::uint32_t prefill_left = 0;
    // Stream reads in flight: (stream slot, segments left, failed), tagged as prefill reads whose
    // index (or run member) carries kStreamEntry.
    struct StreamEntry {
        std::uint32_t slot, remaining;
        bool failed;
    };
    constexpr std::uint32_t kStreamEntry = 0x80000000U;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> streams;
    std::vector<StreamEntry> stream_entries;
    std::uint32_t stream_left = 0;
    // Refill reads in flight, by refill id (segments left, failed), tagged as prefill reads whose index carries
    // kRefillEntry.
    struct RefillEntry {
        std::uint32_t remaining = 0;
        bool failed             = false;
    };
    constexpr std::uint32_t kRefillEntry = 0x40000000U;
    std::vector<RefillJob> refills;
    std::vector<RefillEntry> refill_entries(kRefills);
    // The members of each scatter read in flight (tickets or prefill entries), by tag index.
    std::vector<std::vector<std::uint32_t>> runs;
    std::vector<DirectReadQueue::Completion> done;
    std::vector<std::uint64_t> copied;
    std::size_t outstanding = 0;        // reads and copies submitted and not completed
    std::deque<std::uint32_t> waiting;  // tickets of this round waiting for a ring slot, in order
    // The current fetch request: its tickets by index, and how many of them the device has released.
    std::vector<std::uint32_t> fetch_ticket;
    std::uint32_t released     = 0;
    bool fetch_failed          = false;
    // Read-ahead tickets of the last request's guess (fetch kAheadFetch until adopted), by key
    // (ticket + 1) for adoption. They stay out of round_ticket_of_key_ until adopted, so a CPU demand
    // never shares one (it reads its own copy).
    std::vector<std::pair<std::uint32_t, std::uint32_t>> ahead; // (ticket, key)
    std::vector<std::uint32_t> ahead_of_key(keys(), 0);
    const auto new_ahead_ticket = [&](std::uint32_t k) {
        const std::uint32_t t = ticket_count_.fetch_add(1, std::memory_order_acq_rel);
        if (t >= ticket_capacity_) { return kNoTicket; }
        Ticket& ticket  = tickets_[t];
        ticket.key      = k;
        ticket.resident = false;
        ticket.recycled = false;
        ticket.fetch    = kAheadFetch;
        // Every field new_ticket sets: indices recycle each round, so a stale copy_from would copy
        // another key's unpinned record into this key's ring slot. A key with an unpinned record of
        // its own is copied from it, as a fetch ticket is; any other is read from the artifact.
        ticket.copy_from = unpinned_record(k);
        ticket.bytes     = nullptr;
        ticket.done.store(false, std::memory_order_relaxed);
        ticket.status.store(0, std::memory_order_relaxed);
        ticket.state.store(0, std::memory_order_relaxed);
        in_flight_.fetch_add(1, std::memory_order_acq_rel);
        return t;
    };
    // Read-ahead tickets still waiting for a ring slot are dropped (nothing will read them).
    const auto drop_waiting_ahead = [&] {
        for (auto it = waiting.begin(); it != waiting.end();) {
            if (tickets_[*it].fetch == kAheadFetch) {
                if (ahead_of_key[tickets_[*it].key] == *it + 1) { ahead_of_key[tickets_[*it].key] = 0; }
                fail_ticket(*it);
                it = waiting.erase(it);
            } else {
                ++it;
            }
        }
    };
    std::uint64_t round        = 0;
    auto last_work             = Clock::now();
    auto* const fetch          = options_.fetch;
    const auto finish_ticket = [&](Ticket& ticket, std::uint32_t t, bool failed) {
        if (!failed) { ticket.bytes = slot_bytes(ticket.slot); }
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
                wake_.wait(lock, [&] {
                    return stop_ || open_ || !new_demands_.empty() || !prefill_jobs_.empty() || !stream_jobs_.empty() ||
                           !refill_jobs_.empty();
                });
                last_work = Clock::now();
            }
            // Stopping abandons demands still waiting for a slot (their readers are gone).
            if (stop_ && outstanding == 0) { return; }
            demands.swap(new_demands_);
            prefills.swap(prefill_jobs_);
            streams.swap(stream_jobs_);
            refills.swap(refill_jobs_);
            open = open_;
            if (round_ != round) { // a boundary reset the tickets: the last request is over
                round = round_;
                fetch_ticket.clear();
                for (const auto& [t, k] : ahead) { ahead_of_key[k] = 0; }
                ahead.clear();
                released     = 0;
                fetch_failed = false;
            }
        }
        if (!open) { drop_waiting_ahead(); } // the round is over: no request will adopt them
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
                    const std::uint32_t k = key(static_cast<std::uint32_t>(request.layer),
                                                static_cast<std::uint32_t>(request.experts[i]));
                    // A read-ahead ticket of this key is adopted: it lands as this request's record i
                    // (finish_ticket, or now if it already has), and becomes the round's ticket of k.
                    if (const std::uint32_t known = ahead_of_key[k]; known != 0) {
                        const std::uint32_t t = known - 1;
                        ahead_of_key[k] = 0;
                        Ticket& ticket = tickets_[t];
                        ticket.fetch    = static_cast<std::int32_t>(i);
                        fetch_ticket[i] = t;
                        round_ticket_of_key_[k] = t + 1;
                        const std::uint32_t state = ticket.state.load(std::memory_order_acquire);
                        if (state == 1) {
                            fetch->land(static_cast<std::uint32_t>(i), slot_bytes(ticket.slot));
                        } else if (state == 2 && !fetch_failed) {
                            fetch->fail(EIO);
                            fetch_failed = true;
                        }
                        continue;
                    }
                    const std::uint32_t t = new_ticket(k, static_cast<std::int32_t>(i));
                    if (t == kNoTicket) {
                        fetch->fail(ENOBUFS);
                        fetch_failed = true;
                        break;
                    }
                    fetch_ticket[i] = t;
                    if (tickets_[t].resident) {
                        fetch->land(static_cast<std::uint32_t>(i), tickets_[t].bytes);
                        agent_fetch_from_ram_.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        waiting.push_back(t);
                    }
                }
                // The last guess's records this request did not name: their slots may be reused.
                drop_waiting_ahead();
                for (const auto& [t, k] : ahead) {
                    if (ahead_of_key[k] == t + 1) { // not adopted
                        ahead_of_key[k] = 0;
                        tickets_[t].done.store(true, std::memory_order_release);
                    }
                }
                ahead.clear();
                // A wide request reads the next layer's records without a host copy behind its own.
                if (!fetch_failed && options_.ahead > 0 && request.experts.size() >= options_.ahead_min &&
                    static_cast<std::uint32_t>(request.layer) + 1 < store_.layers()) {
                    const auto next = static_cast<std::uint32_t>(request.layer) + 1;
                    for (std::uint32_t e = 0; e < store_.experts() && ahead.size() < options_.ahead; ++e) {
                        const std::uint32_t k = key(next, e);
                        if (round_ticket_of_key_[k] != 0 || tier_.slot_of(k) >= 0) { continue; }
                        const std::uint32_t t = new_ahead_ticket(k);
                        if (t == kNoTicket) { break; }
                        ahead_of_key[k] = t + 1;
                        ahead.emplace_back(t, k);
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
                fail_ticket(t);
                continue;
            }
            if (!take_slot(t)) { break; } // a slot frees up once the device or the CPU is done with one
            waiting.pop_front();
            ticket.submitted_ns = now_ns();
            if (ticket.copy_from != nullptr) { // an unpinned record: copied at memory speed
                ticket.remaining = 1;
                {
                    std::lock_guard<std::mutex> lock(copy_mutex_);
                    copies_.push_back({const_cast<std::uint8_t*>(slot_bytes(ticket.slot)), ticket.copy_from,
                                       tag_of(kDemandTag, t, 0)});
                }
                copy_wake_.notify_one();
                unpinned_copies_.fetch_add(1, std::memory_order_relaxed);
                ++outstanding;
                last_work = Clock::now();
                continue;
            }
            ticket.remaining    = static_cast<std::uint32_t>(
                store_.segments(ticket.key / store_.experts(), ticket.key % store_.experts()).size());
            // The next waiting tickets of the same request whose records follow this one in the file
            // join it in one read while a ring slot is free for each.
            std::vector<std::uint32_t> run{t};
            std::size_t bytes = store_.record_bytes();
            while (!waiting.empty() && ticket.remaining == 1 && ticket.fetch >= 0 && !fetch_failed &&
                   run.size() < DirectReadQueue::kMaxTargets && bytes + store_.record_bytes() <= kFetchRunBytes) {
                const std::uint32_t next = waiting.front();
                if (tickets_[next].fetch < 0 || tickets_[next].copy_from != nullptr ||
                    !follows(tickets_[run.back()].key, tickets_[next].key) || !take_slot(next)) {
                    break;
                }
                waiting.pop_front();
                tickets_[next].submitted_ns = ticket.submitted_ns;
                tickets_[next].remaining    = 1;
                run.push_back(next);
                bytes += store_.record_bytes();
            }
            if (run.size() == 1) {
                submit(ticket.key, const_cast<std::uint8_t*>(slot_bytes(ticket.slot)), tag_of(kDemandTag, t, 0));
                outstanding += ticket.remaining;
            } else {
                std::vector<std::pair<std::uint32_t, std::uint8_t*>> records;
                for (const std::uint32_t member : run) {
                    records.emplace_back(tickets_[member].key, const_cast<std::uint8_t*>(slot_bytes(tickets_[member].slot)));
                }
                const auto index = static_cast<std::uint32_t>(runs.size());
                runs.push_back(std::move(run));
                submit_run(records, tag_of(kDemandRunTag, index, 0));
                ++outstanding;
            }
            last_work = Clock::now();
        }
        if (!prefills.empty()) {
            // The startup fill lands whole before the first request, so its order is free: by file
            // offset, records that follow each other are read as one run.
            const auto offset_of = [&](std::uint32_t key) {
                const auto s = store_.segments(key / store_.experts(), key % store_.experts())[0];
                return std::pair<std::uint32_t, std::uint64_t>(s.file, s.offset);
            };
            std::sort(prefills.begin(), prefills.end(),
                      [&](const auto& a, const auto& b) { return offset_of(a.first) < offset_of(b.first); });
            for (std::size_t i = 0; i < prefills.size();) {
                std::size_t end   = i + 1;
                std::size_t bytes = store_.record_bytes();
                while (end < prefills.size() && end - i < DirectReadQueue::kMaxTargets &&
                       bytes + store_.record_bytes() <= kFillRunBytes && follows(prefills[end - 1].first, prefills[end].first)) {
                    bytes += store_.record_bytes();
                    ++end;
                }
                if (end - i == 1) {
                    const auto [key, destination] = prefills[i];
                    const auto index              = static_cast<std::uint32_t>(prefill_entries.size());
                    const auto n = static_cast<std::uint32_t>(store_.segments(key / store_.experts(), key % store_.experts()).size());
                    prefill_entries.push_back({key, n, false});
                    submit(key, destination, tag_of(kPrefillTag, index, 0));
                    outstanding += n;
                } else {
                    std::vector<std::uint32_t> members;
                    for (std::size_t k = i; k < end; ++k) {
                        members.push_back(static_cast<std::uint32_t>(prefill_entries.size()));
                        prefill_entries.push_back({prefills[k].first, 1, false});
                    }
                    const auto index = static_cast<std::uint32_t>(runs.size());
                    runs.push_back(std::move(members));
                    submit_run(std::span(prefills).subspan(i, end - i), tag_of(kPrefillRunTag, index, 0));
                    ++outstanding;
                }
                prefill_left += static_cast<std::uint32_t>(end - i);
                i = end;
            }
        }
        prefills.clear();
        // Stream reads in the order asked (a layer's experts, ascending): records that follow each
        // other in a file are read as one run.
        for (std::size_t i = 0; i < streams.size();) {
            std::size_t end   = i + 1;
            std::size_t bytes = store_.record_bytes();
            while (end < streams.size() && end - i < DirectReadQueue::kMaxTargets && bytes + store_.record_bytes() <= kFillRunBytes &&
                   follows(streams[end - 1].first, streams[end].first)) {
                bytes += store_.record_bytes();
                ++end;
            }
            if (end - i == 1) {
                const auto [key, slot] = streams[i];
                const auto index       = static_cast<std::uint32_t>(stream_entries.size());
                const auto n = static_cast<std::uint32_t>(store_.segments(key / store_.experts(), key % store_.experts()).size());
                stream_entries.push_back({slot, n, false});
                submit(key, const_cast<std::uint8_t*>(slot_bytes(stream_slot_[slot])), tag_of(kPrefillTag, index | kStreamEntry, 0));
                outstanding += n;
            } else {
                std::vector<std::uint32_t> members;
                std::vector<std::pair<std::uint32_t, std::uint8_t*>> records;
                for (std::size_t k = i; k < end; ++k) {
                    members.push_back(static_cast<std::uint32_t>(stream_entries.size()) | kStreamEntry);
                    stream_entries.push_back({streams[k].second, 1, false});
                    records.emplace_back(streams[k].first, const_cast<std::uint8_t*>(slot_bytes(stream_slot_[streams[k].second])));
                }
                const auto index = static_cast<std::uint32_t>(runs.size());
                runs.push_back(std::move(members));
                submit_run(records, tag_of(kPrefillRunTag, index, 0));
                ++outstanding;
            }
            stream_left += static_cast<std::uint32_t>(end - i);
            i = end;
        }
        streams.clear();
        // Refills: one read each, into the demotion slot the boundary chose (behind the demands submitted above).
        for (const RefillJob& job : refills) {
            const auto n = static_cast<std::uint32_t>(store_.segments(job.key / store_.experts(), job.key % store_.experts()).size());
            refill_entries[job.id] = {n, false};
            submit(job.key, const_cast<std::uint8_t*>(slot_bytes(job.slot)), tag_of(kPrefillTag, job.id | kRefillEntry, 0),
                   DirectReadQueue::Priority::Restore);
            outstanding += n;
        }
        if (!refills.empty()) { last_work = Clock::now(); }
        refills.clear();
        done.clear();
        // While a round is open the agent spins (a fetch request must not wait for a timer).
        queue_->poll(done, outstanding != 0 && !open ? std::chrono::microseconds(200) : std::chrono::microseconds(0));
        copied.clear();
        if (!copiers_.empty()) {
            std::lock_guard<std::mutex> lock(copy_mutex_);
            copied.swap(copied_);
        }
        const auto ticket_read = [&](std::uint32_t t, bool ok) {
            Ticket& ticket = tickets_[t];
            if (!ok) { ticket.status.store(EIO, std::memory_order_relaxed); }
            if (--ticket.remaining == 0) {
                const bool failed = ticket.status.load(std::memory_order_relaxed) != 0;
                agent_read_ns_.fetch_add(now_ns() - ticket.submitted_ns, std::memory_order_relaxed);
                (failed ? agent_failures_ : agent_reads_).fetch_add(1, std::memory_order_relaxed);
                finish_ticket(ticket, t, failed);
            }
        };
        const auto prefill_read = [&](std::uint32_t index, bool ok) {
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
        };
        const auto stream_done = [&](std::uint32_t index, bool ok) {
            StreamEntry& entry = stream_entries[index];
            entry.failed       = entry.failed || !ok;
            if (--entry.remaining == 0) {
                stream_state_[entry.slot].store(entry.failed ? 2 : 1, std::memory_order_release);
                agent_stream_reads_.fetch_add(1, std::memory_order_relaxed);
                if (--stream_left == 0) { stream_entries.clear(); }
            }
        };
        const auto refill_done = [&](std::uint32_t id, bool ok) {
            RefillEntry& entry = refill_entries[id];
            entry.failed       = entry.failed || !ok;
            if (--entry.remaining == 0) { refill_state_[id].store(entry.failed ? 2 : 1, std::memory_order_release); }
        };
        // A prefill-kind index or run member is a stream read's when it carries kStreamEntry, a refill's when it
        // carries kRefillEntry.
        const auto fill_read = [&](std::uint32_t index, bool ok) {
            if ((index & kStreamEntry) != 0) {
                stream_done(index & ~kStreamEntry, ok);
            } else if ((index & kRefillEntry) != 0) {
                refill_done(index & ~kRefillEntry, ok);
            } else {
                prefill_read(index, ok);
            }
        };
        for (const auto& c : done) {
            --outstanding;
            const std::uint64_t kind = c.tag >> 62U;
            const auto index         = static_cast<std::uint32_t>((c.tag >> 8U) & ((1ULL << 54U) - 1U));
            const bool ok            = c.status == DirectReadQueue::Status::Ok;
            if (kind == kDemandTag) {
                ticket_read(index, ok);
            } else if (kind == kPrefillTag) {
                fill_read(index, ok);
            } else {
                for (const std::uint32_t member : runs[index]) {
                    if (kind == kDemandRunTag) {
                        ticket_read(member, ok);
                    } else {
                        fill_read(member, ok);
                    }
                }
                runs[index].clear();
            }
        }
        for (const std::uint64_t tag : copied) { // not reads: the read statistics leave them out
            --outstanding;
            const auto t = static_cast<std::uint32_t>((tag >> 8U) & ((1ULL << 54U) - 1U));
            finish_ticket(tickets_[t], t, false);
        }
        if (outstanding == 0) { runs.clear(); } // no run is in flight: their indices start over
        if (!done.empty() || !copied.empty()) { last_work = Clock::now(); }
        if (open && done.empty() && copied.empty()) {
            // Spinning through an open round costs nothing measurable: sleeping 200 us when idle instead left the SSD
            // tier's Dense8 decode where it was (0.968x against 0.966x of the full layout, 2026-10-27 x_ab7).
            if (Clock::now() - last_work > kIdleSpin) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
                relax();
            }
        } else if (!open && done.empty() && copied.empty() && outstanding == 0 && !waiting.empty()) {
            std::this_thread::yield(); // tickets of a closed round wait for a reader to finish
        }
    }
}

} // namespace infernix::runtime::expert_offload
