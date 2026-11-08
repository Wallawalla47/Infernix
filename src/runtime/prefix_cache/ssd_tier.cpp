#include "runtime/prefix_cache/ssd_tier.h"

#include "runtime/prefix_cache/fp32_row_codec.h"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <new>
#include <stdexcept>
#include <thread>
#include <utility>

namespace infernix::runtime::prefix_cache {
namespace {

// Two write-backs at once keep the drive busy without a long queue a promotion could find ahead of
// it (the file issues no write while a read waits, but the writes already issued finish first).
constexpr std::uint32_t kWriteJobs = 2;
// After a refused write-back (nothing qualified, or no room) the idle Engine is not woken for another
// for this long (polls between rounds try every time; a landed promotion resets it).
constexpr std::chrono::milliseconds kWriteRetry{100};
// After this many failed writes the tier stops writing (reads go on).
constexpr std::uint64_t kWriteFailureLimit = 3;
// Consecutive slabs join one I/O up to this size: several in flight reach a drive's sequential rate.
constexpr std::size_t kMaxIoBytes = 8U << 20U;
// Only batches this large measure a rate (smaller ones measure latency).
constexpr std::uint64_t kRateBatchBytes = 16U << 20U;
// The write budget's level, kept across restarts beside the scratch files, saved this often.
constexpr const char* kBudgetFile = "infernix-ssd-write-budget.txt";
constexpr std::chrono::seconds kBudgetSaveInterval{60};
// Staging buffers kept for reuse (a fresh one costs its first-touch page faults on the read path).
constexpr std::size_t kSpareStaging = 1;

} // namespace

// The codec's worker threads: a FIFO of tasks, and a wait for all of them.
class SsdTier::Workers {
public:
    explicit Workers(unsigned count) {
        for (unsigned i = 0; i < count; ++i) {
            threads_.emplace_back([this] { run(); });
        }
    }
    ~Workers() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        for (std::thread& thread : threads_) { thread.join(); }
    }
    Workers(const Workers&)            = delete;
    Workers& operator=(const Workers&) = delete;

    [[nodiscard]] unsigned count() const noexcept { return static_cast<unsigned>(threads_.size()); }
    void post(std::function<void()> task) {
        {
            std::lock_guard lock(mutex_);
            tasks_.push_back(std::move(task));
        }
        wake_.notify_one();
    }
    void wait_idle() {
        std::unique_lock lock(mutex_);
        idle_.wait(lock, [&] { return tasks_.empty() && running_ == 0; });
    }

private:
    void run() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock lock(mutex_);
                wake_.wait(lock, [&] { return stop_ || !tasks_.empty(); });
                if (tasks_.empty()) { return; }
                task = std::move(tasks_.front());
                tasks_.pop_front();
                ++running_;
            }
            task();
            {
                std::lock_guard lock(mutex_);
                --running_;
                if (tasks_.empty() && running_ == 0) { idle_.notify_all(); }
            }
        }
    }

    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable wake_, idle_;
    std::deque<std::function<void()>> tasks_;
    unsigned running_ = 0;
    bool stop_        = false;
};

void SsdTier::AlignedFree::operator()(std::byte* p) const noexcept {
    ::operator delete(p, std::align_val_t{DirectScratchFile::kAlignment});
}

SsdTierStats ssd_tier_stats(const SsdTier* tier, const PrefixIndexStats& index, std::size_t slab_bytes,
                            std::uint32_t slabs) noexcept {
    SsdTierStats out;
    if (tier == nullptr) { return out; }
    const SsdTier::Counters& counters = tier->counters();
    out.capacity_bytes       = static_cast<std::uint64_t>(slabs) * slab_bytes;
    out.used_bytes           = static_cast<std::uint64_t>(slabs - std::min(slabs, index.ssd_free_slabs)) * slab_bytes;
    out.ssd_only_snapshots   = index.ssd_only_snapshots;
    out.write_bytes          = counters.write_bytes;
    out.read_bytes           = counters.read_bytes;
    out.image_writes         = counters.image_writes;
    out.block_writes         = counters.block_writes;
    out.promotions           = counters.promotions;
    out.promotion_room_waits = counters.promotions_waiting_for_room;
    out.host_image_demotions = index.host_image_demotions;
    out.host_block_demotions = index.host_block_demotions;
    out.snapshot_evictions   = index.ssd_snapshot_evictions;
    out.backup_drops         = index.ssd_backup_drops;
    out.dead_reclaims        = index.ssd_dead_reclaims;
    out.write_failures       = counters.write_failures;
    out.read_failures        = counters.read_failures;
    out.read_bytes_per_second = counters.read_bytes_per_second;
    out.slab_bytes            = slab_bytes;
    if (tier->budget()) {
        out.budget_bytes_per_hour  = tier->budget()->bytes_per_hour();
        out.budget_available_bytes = tier->budget()->available_bytes();
    }
    out.declined_snapshots = index.ssd_declined_snapshots;
    out.packed_images      = counters.packed_images;
    out.packed_saved_bytes = counters.packed_saved_bytes;
    out.policy_nanoseconds = counters.policy_nanoseconds;
    out.by_kind            = index.ssd_by_kind;
    return out;
}

SsdTier::SsdTier(const std::filesystem::path& directory, std::uint32_t slabs, std::size_t slab_bytes,
                 std::function<std::byte*(std::uint32_t)> host_slab, std::uint64_t write_bytes_per_hour, bool codec,
                 const std::filesystem::path& kept_file)
    : file_(directory, static_cast<std::uint64_t>(slabs) * slab_bytes, DirectScratchFile::Options{.kept_file = kept_file}),
      slab_bytes_(slab_bytes), kept_(!kept_file.empty()), host_slab_(std::move(host_slab)) {
    if (slab_bytes == 0 || slab_bytes % DirectScratchFile::kAlignment != 0 || slab_bytes % kCodecRowBytes != 0 ||
        !host_slab_) {
        throw std::invalid_argument("SSD tier slabs must be positive multiples of 4 KiB with a Host address");
    }
    if (kept_) {
        slab_hash_.assign(slabs, 0U);
        unverified_.assign(slabs, 0U);
    }
    if (write_bytes_per_hour != 0) {
        budget_.emplace(write_bytes_per_hour, directory / kBudgetFile, SsdWriteBudget::Clock::now());
        last_budget_save_ = std::chrono::steady_clock::now();
    }
    if (codec) {
        // Decoding is on a promotion's path: enough threads to keep up with a fast drive's reads.
        const unsigned hardware = std::max(1U, std::thread::hardware_concurrency());
        workers_ = std::make_unique<Workers>(std::clamp(hardware / 4U, 2U, 8U));
    }
}

SsdTier::~SsdTier() {
    if (workers_) { workers_->wait_idle(); }
    file_.wait_idle();
    if (budget_) { budget_->save(SsdWriteBudget::Clock::now()); }
}

void SsdTier::set_waker(std::function<void()> waker) {
    waker_ = waker;
    file_.set_waker(std::move(waker));
}

SsdTier::Staging SsdTier::take_staging() {
    if (!spare_staging_.empty()) {
        Staging staging = std::move(spare_staging_.back());
        spare_staging_.pop_back();
        return staging;
    }
    return Staging(static_cast<std::byte*>(::operator new(staging_bytes_, std::align_val_t{DirectScratchFile::kAlignment})));
}

void SsdTier::give_staging(Staging staging) noexcept {
    if (!staging) { return; }
    if (spare_staging_.size() < kSpareStaging) { spare_staging_.push_back(std::move(staging)); }
}

void SsdTier::codec_done(std::uint64_t tag) {
    {
        std::lock_guard lock(codec_mutex_);
        codec_finished_.push_back(tag);
    }
    if (waker_) { waker_(); }
}

void SsdTier::expect(std::uint32_t slab, std::uint64_t hash) {
    if (!kept_ || slab >= slab_hash_.size()) { throw std::invalid_argument("SSD tier: no kept slab " + std::to_string(slab)); }
    slab_hash_[slab] = hash;
    if (unverified_[slab] == 0U) {
        unverified_[slab] = 1U;
        ++unverified_count_;
    }
}

void SsdTier::flush() {
    if (!jobs_.empty()) { throw std::logic_error("SSD tier flush requires a drained tier"); }
    file_.flush();
}

bool SsdTier::verify_kept(const Job& job) {
    if (job.hashes.empty()) { return true; }
    for (std::size_t k = 0; k < job.hash_slabs.size(); ++k) {
        const std::uint32_t slab = job.hash_slabs[k];
        if (unverified_[slab] != 0U && job.hashes[k] != slab_hash_[slab]) {
            ++counters_.kept_slab_mismatches;
            return false;
        }
    }
    for (const std::uint32_t slab : job.hash_slabs) {
        if (unverified_[slab] != 0U) {
            unverified_[slab] = 0U;
            --unverified_count_;
        }
    }
    return true;
}

void SsdTier::record_written(const Job& job) {
    for (std::size_t k = 0; k < job.hash_slabs.size(); ++k) {
        const std::uint32_t slab = job.hash_slabs[k];
        slab_hash_[slab]         = job.hashes[k];
        if (unverified_[slab] != 0U) {
            unverified_[slab] = 0U;
            --unverified_count_;
        }
    }
}

DirectScratchFile::Batch SsdTier::batch(std::uint64_t tag, DirectScratchFile::Op op, Job& job) const {
    // Slab copies (blocks, a raw image, a tail) and a packed image's stream slabs from its staging.
    struct Piece {
        std::uint64_t offset;
        std::byte* memory;
    };
    std::vector<Piece> pieces;
    pieces.reserve(job.job.transfers.size() + job.job.image_ssd_slabs.size());
    for (const SsdTransfer& transfer : job.job.transfers) {
        pieces.push_back({static_cast<std::uint64_t>(transfer.ssd_slab) * slab_bytes_, host_slab_(transfer.host_slab)});
    }
    for (std::size_t k = 0; k < job.job.image_ssd_slabs.size(); ++k) {
        pieces.push_back({static_cast<std::uint64_t>(job.job.image_ssd_slabs[k]) * slab_bytes_, job.staging.get() + k * slab_bytes_});
    }
    std::sort(pieces.begin(), pieces.end(), [](const Piece& a, const Piece& b) { return a.offset < b.offset; });
    // A kept file hashes every slab it writes, and the slabs of the previous run a read takes.
    job.hash_slabs.clear();
    job.hashes.clear();
    bool hashed = false;
    if (kept_) {
        for (const Piece& piece : pieces) { job.hash_slabs.push_back(static_cast<std::uint32_t>(piece.offset / slab_bytes_)); }
        hashed = op == DirectScratchFile::Op::Write ||
                 (unverified_count_ != 0 &&
                  std::any_of(job.hash_slabs.begin(), job.hash_slabs.end(), [&](std::uint32_t s) { return unverified_[s] != 0U; }));
        if (hashed) {
            job.hashes.assign(pieces.size(), 0U);
        } else {
            job.hash_slabs.clear();
        }
    }
    DirectScratchFile::Batch out{.tag = tag, .op = op, .ios = {}};
    for (std::size_t k = 0; k < pieces.size(); ++k) {
        const Piece& piece = pieces[k];
        if (!out.ios.empty()) {
            DirectScratchFile::Io& last = out.ios.back();
            // Consecutive on the SSD and in memory (one pinned chunk, or a staging buffer): one I/O.
            if (last.offset + last.bytes == piece.offset && last.memory + last.bytes == piece.memory &&
                last.bytes + slab_bytes_ <= kMaxIoBytes) {
                last.bytes += slab_bytes_;
                continue;
            }
        }
        out.ios.push_back(DirectScratchFile::Io{.offset      = piece.offset,
                                                .bytes       = slab_bytes_,
                                                .memory      = piece.memory,
                                                .hashes      = hashed ? job.hashes.data() + k : nullptr,
                                                .hash_stride = hashed ? slab_bytes_ : 0U});
    }
    return out;
}

void SsdTier::land_promotion(PrefixCacheIndex& index, Job& job, bool ok) {
    --promotions_in_flight_;
    promotion_landed_ = true;
    if (ok) {
        // Image slabs the SSD copy leaves out hold zeros, never another slab's bytes.
        for (const std::uint32_t slab : job.job.zero_host_slabs) { std::memset(host_slab_(slab), 0, slab_bytes_); }
        counters_.block_reads += job.job.nodes.size();
        counters_.image_reads += job.job.image ? 1U : 0U;
        ++counters_.promotions;
        index.complete_ssd_promotion(job.job);
    } else {
        ++counters_.read_failures;
        index.abort_ssd_promotion(job.job);
    }
    give_staging(std::move(job.staging));
}

void SsdTier::start_decode(std::uint64_t tag, Job& job) {
    // The stream's payloads, one per kept image slab, located before any decoding starts.
    const std::size_t available = job.job.image_ssd_slabs.size() * slab_bytes_;
    std::vector<std::size_t> offsets;
    offsets.reserve(job.job.image_host_slabs.size());
    std::size_t offset = 0;
    for (std::size_t i = 0; i < job.job.image_host_slabs.size(); ++i) {
        if (offset + 4U > available) {
            job.failed = true;
            break;
        }
        offsets.push_back(offset);
        offset += payload_size(job.staging.get() + offset);
    }
    if (job.failed || offset > available) {
        job.failed = true;
        codec_done(tag);
        return;
    }
    std::vector<std::byte*> targets;
    targets.reserve(job.job.image_host_slabs.size());
    for (const std::uint32_t slab : job.job.image_host_slabs) { targets.push_back(host_slab_(slab)); }
    job.phase           = Phase::Decoding;
    const unsigned parts = std::min<unsigned>(workers_->count(), static_cast<unsigned>(targets.size()));
    job.parts            = parts;
    for (unsigned part = 0; part < parts; ++part) {
        const std::size_t begin = targets.size() * part / parts, end = targets.size() * (part + 1U) / parts;
        workers_->post([this, tag, &job, begin, end, offsets, targets, available] {
            for (std::size_t i = begin; i < end && !job.failed; ++i) {
                if (!decode_slab(job.staging.get() + offsets[i], available - offsets[i], targets[i], slab_bytes_)) {
                    job.failed = true;
                }
            }
            if (job.parts.fetch_sub(1) == 1U) { codec_done(tag); }
        });
    }
}

void SsdTier::issue_write(PrefixCacheIndex& index, std::uint64_t tag, Job& job) {
    job.phase = Phase::Io;
    if (job.staging) {
        // Packed when the stream takes fewer SSD slabs than the kept image; otherwise written raw.
        const auto stream_slabs = static_cast<std::uint32_t>((job.stream_bytes + slab_bytes_ - 1U) / slab_bytes_);
        const std::uint32_t kept = index.ssd_image_slab_count();
        if (stream_slabs < kept) {
            index.pack_ssd_write(job.job, stream_slabs);
            ++counters_.packed_images;
            counters_.packed_saved_bytes += static_cast<std::uint64_t>(kept - stream_slabs) * slab_bytes_;
        } else {
            give_staging(std::move(job.staging));
            packed_write_busy_ = false;
        }
    }
    try {
        file_.submit(batch(tag, DirectScratchFile::Op::Write, job));
    } catch (const std::exception& error) {
        // A transfer the file refuses fails like one the drive fails: the entries stay as they were.
        index.abort_ssd_write(job.job);
        ++counters_.write_failures;
        last_error_ = error.what();
        write_finished(job);
        if (job.staging) {
            give_staging(std::move(job.staging));
            packed_write_busy_ = false;
        }
        jobs_.erase(tag);
        return;
    }
    if (budget_) {
        budget_->charge((job.job.transfers.size() + job.job.image_ssd_slabs.size()) * slab_bytes_);
    }
}

void SsdTier::apply_codec(PrefixCacheIndex& index) {
    std::vector<std::uint64_t> finished;
    {
        std::lock_guard lock(codec_mutex_);
        finished.swap(codec_finished_);
    }
    for (const std::uint64_t tag : finished) {
        const auto found = jobs_.find(tag);
        if (found == jobs_.end()) { continue; } // abandoned
        Job& job = *found->second;
        if (job.promotion) {
            land_promotion(index, job, !job.failed);
            jobs_.erase(found);
        } else {
            try {
                issue_write(index, tag, job);
            } catch (const std::exception& error) {
                // An encoded write that cannot be issued fails like a refused transfer.
                last_error_ = error.what();
                ++counters_.write_failures;
                index.abort_ssd_write(job.job);
                write_finished(job);
                give_staging(std::move(job.staging));
                packed_write_busy_ = false;
                jobs_.erase(tag);
            }
        }
    }
}

void SsdTier::write_finished(const Job& job) noexcept {
    --writes_in_flight_;
    write_slabs_in_flight_ -= job.host_slabs;
    if (room_wanted_) {
        // A promotion waits for room (promote): whatever this write's outcome, it quotes again.
        room_wanted_      = false;
        promotion_landed_ = true;
    }
}

void SsdTier::apply(PrefixCacheIndex& index) {
    if (workers_) { apply_codec(index); }
    completions_.clear();
    file_.poll(completions_);
    for (const DirectScratchFile::Completion& done : completions_) {
        const auto found = jobs_.find(done.tag);
        if (found == jobs_.end()) { continue; } // abandoned
        Job& job             = *found->second;
        const double seconds = static_cast<double>(done.nanoseconds) * 1e-9;
        if (job.promotion) {
            if (done.ok) {
                counters_.read_bytes += done.bytes;
                if (done.bytes >= kRateBatchBytes && seconds > 0.0) {
                    counters_.read_bytes_per_second = static_cast<double>(done.bytes) / seconds;
                    index.set_ssd_rate(counters_.read_bytes_per_second);
                }
            }
            // A slab of the previous run whose bytes changed fails the promotion as a failed read does.
            const bool ok = done.ok && verify_kept(job);
            if (ok && job.job.packed) {
                start_decode(done.tag, job); // lands through apply_codec
                continue;
            }
            land_promotion(index, job, ok);
        } else {
            write_finished(job);
            if (done.ok) {
                counters_.write_bytes += done.bytes;
                counters_.block_writes += job.job.nodes.size();
                counters_.image_writes += job.job.image ? 1U : 0U;
                if (done.bytes >= kRateBatchBytes && seconds > 0.0) {
                    counters_.write_bytes_per_second = static_cast<double>(done.bytes) / seconds;
                }
                if (kept_) { record_written(job); }
                index.complete_ssd_write(job.job);
            } else {
                ++counters_.write_failures;
                index.abort_ssd_write(job.job);
            }
            if (job.staging) {
                give_staging(std::move(job.staging));
                packed_write_busy_ = false;
            }
        }
        jobs_.erase(found);
    }
    if (workers_) { apply_codec(index); } // decodes that finished meanwhile land in this pass
    due_.store(compute_due(index), std::memory_order_relaxed);
}

std::uint64_t SsdTier::write_back_target(const PrefixCacheIndex& index) const noexcept {
    // At most a quarter of the Host tier stays unwritten: one prefill unit can take far more than half
    // of it at once (its captures and its released blocks, or a promotion), before a write-back started
    // after it could land, and whatever it then displaces unwritten is lost. (Half unwritten wrote 12 %
    // less on the agentic replay for 1.1 points of cache hits, a trade the write budget makes better.)
    const std::uint64_t host = index.host_slab_limit();
    return host - host / 4U;
}

bool SsdTier::budget_still_blocked(const PrefixCacheIndex& index) const noexcept {
    return budget_blocked_ && index.snapshot_epoch() == blocked_epoch_ &&
           budget_->available_bytes() < blocked_available_ + blocked_retry_bytes_;
}

bool SsdTier::compute_due(const PrefixCacheIndex& index) const noexcept {
    const std::uint64_t room   = index.host_lossless_slabs() + write_slabs_in_flight_;
    const std::uint64_t target = write_back_target(index);
    return counters_.write_failures < kWriteFailureLimit && writes_in_flight_ < kWriteJobs &&
           !(packed_write_busy_ && 2U * room >= target) && !budget_still_blocked(index) &&
           std::chrono::steady_clock::now() >= next_write_attempt_ && room < target;
}

void SsdTier::start_write_backs(PrefixCacheIndex& index) {
    if (counters_.write_failures >= kWriteFailureLimit) { return; }
    if (budget_ && budget_still_blocked(index)) { return; }
    const auto began = std::chrono::steady_clock::now();
    struct Timed {
        Counters& counters;
        std::chrono::steady_clock::time_point began;
        ~Timed() {
            counters.policy_nanoseconds += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - began).count());
        }
    } timed{counters_, began};
    // Tried at every poll: a candidate is often blocked only for a moment (the last blocks of a request
    // that just ended still landing in the Host tier), and the next request's prefill must not find it
    // unwritten. A refusal only keeps the idle Engine from waking for it (compute_due) for a while.
    const auto now             = std::chrono::steady_clock::now();
    const std::uint64_t target = write_back_target(index);
    // With a write budget, a candidate's expected value is its GDSF value density times the share of
    // its kind's SSD images read back so far (a prior of one in two before any).
    std::function<bool(const SsdWriteCandidate&)> admit;
    std::array<double, kSnapshotKinds> read_share{};
    std::uint32_t declined = 0;
    if (budget_) {
        const auto& kinds = index.ssd_kind_stats();
        for (std::size_t k = 0; k < kinds.size(); ++k) {
            read_share[k] = (static_cast<double>(kinds[k].first_reads) + 1.0) / (static_cast<double>(kinds[k].writes) + 2.0);
        }
        const auto wall = SsdWriteBudget::Clock::now();
        admit = [&, wall](const SsdWriteCandidate& candidate) {
            const double value = candidate.value * read_share[static_cast<std::size_t>(candidate.kind)];
            const bool admitted =
                budget_->admit(value, static_cast<std::uint64_t>(candidate.slabs) * slab_bytes_, candidate.first_offer, wall);
            if (!admitted) { ++declined; }
            return admitted;
        };
    }
    // With the codec, write-back with slack (the Host tier's lossless room at least half its target) runs
    // one packed write at a time, encoded into the staging buffer before it is issued; under pressure
    // images go raw, beside a packed write in flight, so write-back keeps the raw path's pace (one packed
    // write at a time fell behind a small Host tier's captures). Raw writes beside a packed one while
    // there is slack made more content clean early and demoted it, losing a snapshot a later resume
    // needed: with slack the tier waits for the packed write.
    while (writes_in_flight_ < kWriteJobs && index.host_lossless_slabs() + write_slabs_in_flight_ < target) {
        const bool slack = 2U * (index.host_lossless_slabs() + write_slabs_in_flight_) >= target;
        if (packed_write_busy_ && slack) { break; }
        declined = 0;
        std::optional<SsdJob> job = index.begin_ssd_write(admit);
        if (!job) {
            next_write_attempt_ = now + kWriteRetry;
            if (budget_ && declined != 0) {
                // Every candidate the budget saw was passed over: nothing changes until a snapshot is
                // published or the budget allows another image.
                budget_blocked_      = true;
                blocked_epoch_       = index.snapshot_epoch();
                blocked_available_   = budget_->available_bytes();
                blocked_retry_bytes_ = static_cast<std::uint64_t>(index.ssd_image_slab_count()) * slab_bytes_;
            }
            return;
        }
        budget_blocked_ = false;
        const std::uint64_t tag = next_tag_++;
        auto entry              = std::make_unique<Job>();
        entry->job              = std::move(*job);
        entry->host_slabs       = entry->job.transfers.size();
        write_slabs_in_flight_ += entry->host_slabs;
        ++writes_in_flight_;
        Job& added = *jobs_.emplace(tag, std::move(entry)).first->second;
        if (workers_ && added.job.image && slack && !packed_write_busy_) {
            if (staging_bytes_ == 0) { staging_bytes_ = (index.ssd_image_slab_count() + 1U) * slab_bytes_; }
            added.staging      = take_staging();
            added.phase        = Phase::Encoding;
            packed_write_busy_ = true;
            // The kept image slabs follow the blocks' transfers.
            std::vector<const std::byte*> sources;
            for (std::uint32_t k = 0; k < index.ssd_image_slab_count(); ++k) {
                sources.push_back(host_slab_(added.job.transfers[added.job.nodes.size() + k].host_slab));
            }
            workers_->post([this, tag, &added, sources = std::move(sources)] {
                std::size_t offset = 0;
                for (const std::byte* source : sources) {
                    offset += encode_slab(source, slab_bytes_, added.staging.get() + offset);
                }
                added.stream_bytes = offset;
                codec_done(tag);
            });
            continue;
        }
        issue_write(index, tag, added);
    }
}

bool SsdTier::poll(PrefixCacheIndex& index) {
    apply(index);
    if (budget_ && std::chrono::steady_clock::now() - last_budget_save_ >= kBudgetSaveInterval) {
        budget_->save(SsdWriteBudget::Clock::now());
        last_budget_save_ = std::chrono::steady_clock::now();
    }
    const bool landed = std::exchange(promotion_landed_, false);
    if (landed) { next_write_attempt_ = {}; }
    start_write_backs(index);
    due_.store(compute_due(index), std::memory_order_relaxed);
    return landed;
}

SsdTier::Promotion SsdTier::promote(PrefixCacheIndex& index, SnapshotRef snapshot) {
    apply(index);
    for (const auto& [tag, job] : jobs_) {
        if (job->promotion && job->job.snapshot == snapshot) { return Promotion::Pending; }
    }
    SsdPromotionStart start = index.begin_ssd_promotion(snapshot);
    if (start.status == SsdPromotionStatus::Resident) { return Promotion::Ready; }
    if (start.status == SsdPromotionStatus::NoRoom) {
        // A write-back in flight pins its Host entries until it lands, when they become clean and yield
        // without loss (a restart refills the Host tier with entries the SSD does not hold yet): wait for it.
        if (writes_in_flight_ != 0) {
            room_wanted_ = true;
            ++counters_.promotions_waiting_for_room;
            return Promotion::Pending;
        }
        return Promotion::Unavailable;
    }
    if (start.job) {
        const std::uint64_t tag = next_tag_++;
        auto entry              = std::make_unique<Job>();
        entry->promotion        = true;
        entry->job              = std::move(*start.job);
        entry->host_slabs       = entry->job.transfers.size() + entry->job.image_host_slabs.size();
        if (entry->job.packed) {
            if (staging_bytes_ == 0) { staging_bytes_ = (index.ssd_image_slab_count() + 1U) * slab_bytes_; }
            entry->staging = take_staging();
        }
        try {
            file_.submit(batch(tag, DirectScratchFile::Op::Read, *entry));
        } catch (const std::exception& error) {
            ++counters_.read_failures;
            last_error_ = error.what();
            index.abort_ssd_promotion(entry->job);
            give_staging(std::move(entry->staging));
            return Promotion::Unavailable;
        }
        ++promotions_in_flight_;
        jobs_.emplace(tag, std::move(entry));
    }
    return Promotion::Pending;
}

void SsdTier::drain(PrefixCacheIndex& index) {
    // A finished encode issues its write and a finished read may start a decode: settle until none is left.
    while (!jobs_.empty()) {
        if (workers_) { workers_->wait_idle(); }
        file_.wait_idle();
        apply(index);
    }
}

void SsdTier::abandon() noexcept {
    if (workers_) { workers_->wait_idle(); }
    file_.wait_idle();
    completions_.clear();
    file_.poll(completions_);
    completions_.clear();
    {
        std::lock_guard lock(codec_mutex_);
        codec_finished_.clear();
    }
    for (auto& [tag, job] : jobs_) { give_staging(std::move(job->staging)); }
    jobs_.clear();
    promotions_in_flight_  = 0;
    writes_in_flight_      = 0;
    write_slabs_in_flight_ = 0;
    packed_write_busy_     = false;
    room_wanted_           = false;
    promotion_landed_      = true; // waiting requests quote the empty cache again
    due_.store(false, std::memory_order_relaxed);
}

} // namespace infernix::runtime::prefix_cache
