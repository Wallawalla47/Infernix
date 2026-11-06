#include "runtime/prefix_cache/ssd_tier.h"

#include <algorithm>
#include <stdexcept>
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

} // namespace

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
    out.host_image_demotions = index.host_image_demotions;
    out.host_block_demotions = index.host_block_demotions;
    out.snapshot_evictions   = index.ssd_snapshot_evictions;
    out.backup_drops         = index.ssd_backup_drops;
    out.dead_reclaims        = index.ssd_dead_reclaims;
    out.write_failures       = counters.write_failures;
    out.read_failures        = counters.read_failures;
    out.read_bytes_per_second = counters.read_bytes_per_second;
    return out;
}

SsdTier::SsdTier(const std::filesystem::path& directory, std::uint32_t slabs, std::size_t slab_bytes,
                 std::function<std::byte*(std::uint32_t)> host_slab)
    : file_(directory, static_cast<std::uint64_t>(slabs) * slab_bytes), slab_bytes_(slab_bytes),
      host_slab_(std::move(host_slab)) {
    if (slab_bytes == 0 || slab_bytes % DirectScratchFile::kAlignment != 0 || !host_slab_) {
        throw std::invalid_argument("SSD tier slabs must be positive multiples of 4 KiB with a Host address");
    }
}

SsdTier::~SsdTier() { file_.wait_idle(); }

void SsdTier::set_waker(std::function<void()> waker) { file_.set_waker(std::move(waker)); }

DirectScratchFile::Batch SsdTier::batch(std::uint64_t tag, DirectScratchFile::Op op, const SsdJob& job) const {
    std::vector<SsdTransfer> transfers = job.transfers;
    std::sort(transfers.begin(), transfers.end(),
              [](const SsdTransfer& a, const SsdTransfer& b) { return a.ssd_slab < b.ssd_slab; });
    DirectScratchFile::Batch out{.tag = tag, .op = op, .ios = {}};
    for (const SsdTransfer& transfer : transfers) {
        std::byte* memory           = host_slab_(transfer.host_slab);
        const std::uint64_t offset  = static_cast<std::uint64_t>(transfer.ssd_slab) * slab_bytes_;
        if (!out.ios.empty()) {
            DirectScratchFile::Io& last = out.ios.back();
            // Consecutive on the SSD and in Host memory (one pinned chunk): one I/O.
            if (last.offset + last.bytes == offset && last.memory + last.bytes == memory &&
                last.bytes + slab_bytes_ <= kMaxIoBytes) {
                last.bytes += slab_bytes_;
                continue;
            }
        }
        out.ios.push_back(DirectScratchFile::Io{.offset = offset, .bytes = slab_bytes_, .memory = memory});
    }
    return out;
}

void SsdTier::apply(PrefixCacheIndex& index) {
    completions_.clear();
    file_.poll(completions_);
    for (const DirectScratchFile::Completion& done : completions_) {
        const auto found = jobs_.find(done.tag);
        if (found == jobs_.end()) { continue; } // abandoned
        Job job = std::move(found->second);
        jobs_.erase(found);
        const double seconds = static_cast<double>(done.nanoseconds) * 1e-9;
        if (job.promotion) {
            --promotions_in_flight_;
            promotion_landed_ = true;
            if (done.ok) {
                counters_.read_bytes += done.bytes;
                counters_.block_reads += job.job.nodes.size();
                counters_.image_reads += job.job.image ? 1U : 0U;
                ++counters_.promotions;
                if (done.bytes >= kRateBatchBytes && seconds > 0.0) {
                    counters_.read_bytes_per_second = static_cast<double>(done.bytes) / seconds;
                    index.set_ssd_rate(counters_.read_bytes_per_second);
                }
                index.complete_ssd_promotion(job.job);
            } else {
                ++counters_.read_failures;
                index.abort_ssd_promotion(job.job);
            }
        } else {
            --writes_in_flight_;
            write_slabs_in_flight_ -= job.host_slabs;
            if (done.ok) {
                counters_.write_bytes += done.bytes;
                counters_.block_writes += job.job.nodes.size();
                counters_.image_writes += job.job.image ? 1U : 0U;
                if (done.bytes >= kRateBatchBytes && seconds > 0.0) {
                    counters_.write_bytes_per_second = static_cast<double>(done.bytes) / seconds;
                }
                index.complete_ssd_write(job.job);
            } else {
                ++counters_.write_failures;
                index.abort_ssd_write(job.job);
            }
        }
    }
    due_.store(compute_due(index), std::memory_order_relaxed);
}

std::uint64_t SsdTier::write_back_target(const PrefixCacheIndex& index) const noexcept {
    // At most a quarter of the Host tier stays unwritten: one prefill unit can take far more than half
    // of it at once (its captures and its released blocks, or a promotion), before a write-back started
    // after it could land, and whatever it then displaces unwritten is lost.
    const std::uint64_t host = index.host_slab_limit();
    return host - host / 4U;
}

bool SsdTier::compute_due(const PrefixCacheIndex& index) const noexcept {
    return counters_.write_failures < kWriteFailureLimit && writes_in_flight_ < kWriteJobs &&
           std::chrono::steady_clock::now() >= next_write_attempt_ &&
           index.host_lossless_slabs() + write_slabs_in_flight_ < write_back_target(index);
}

void SsdTier::start_write_backs(PrefixCacheIndex& index) {
    if (counters_.write_failures >= kWriteFailureLimit) { return; }
    // Tried at every poll: a candidate is often blocked only for a moment (the last blocks of a request
    // that just ended still landing in the Host tier), and the next request's prefill must not find it
    // unwritten. A refusal only keeps the idle Engine from waking for it (compute_due) for a while.
    const auto now             = std::chrono::steady_clock::now();
    const std::uint64_t target = write_back_target(index);
    while (writes_in_flight_ < kWriteJobs && index.host_lossless_slabs() + write_slabs_in_flight_ < target) {
        std::optional<SsdJob> job = index.begin_ssd_write();
        if (!job) {
            next_write_attempt_ = now + kWriteRetry;
            return;
        }
        const std::uint64_t tag = next_tag_++;
        Job entry{.promotion = false, .job = std::move(*job)};
        entry.host_slabs = entry.job.transfers.size();
        try {
            file_.submit(batch(tag, DirectScratchFile::Op::Write, entry.job));
        } catch (const std::exception& error) {
            // A transfer the file refuses fails like one the drive fails: the entries stay as they were.
            index.abort_ssd_write(entry.job);
            ++counters_.write_failures;
            last_error_ = error.what();
            return;
        }
        write_slabs_in_flight_ += entry.host_slabs;
        ++writes_in_flight_;
        jobs_.emplace(tag, std::move(entry));
    }
}

bool SsdTier::poll(PrefixCacheIndex& index) {
    apply(index);
    const bool landed = std::exchange(promotion_landed_, false);
    if (landed) { next_write_attempt_ = {}; }
    start_write_backs(index);
    due_.store(compute_due(index), std::memory_order_relaxed);
    return landed;
}

SsdTier::Promotion SsdTier::promote(PrefixCacheIndex& index, SnapshotRef snapshot) {
    apply(index);
    for (const auto& [tag, job] : jobs_) {
        if (job.promotion && job.job.snapshot == snapshot) { return Promotion::Pending; }
    }
    SsdPromotionStart start = index.begin_ssd_promotion(snapshot);
    if (start.status == SsdPromotionStatus::Resident) { return Promotion::Ready; }
    if (start.status == SsdPromotionStatus::NoRoom) { return Promotion::Unavailable; }
    if (start.job) {
        const std::uint64_t tag = next_tag_++;
        Job entry{.promotion = true, .job = std::move(*start.job)};
        entry.host_slabs   = entry.job.transfers.size();
        try {
            file_.submit(batch(tag, DirectScratchFile::Op::Read, entry.job));
        } catch (const std::exception& error) {
            ++counters_.read_failures;
            last_error_ = error.what();
            index.abort_ssd_promotion(entry.job);
            return Promotion::Unavailable;
        }
        ++promotions_in_flight_;
        jobs_.emplace(tag, std::move(entry));
    }
    return Promotion::Pending;
}

void SsdTier::drain(PrefixCacheIndex& index) {
    file_.wait_idle();
    apply(index);
}

void SsdTier::abandon() noexcept {
    file_.wait_idle();
    completions_.clear();
    file_.poll(completions_);
    completions_.clear();
    jobs_.clear();
    promotions_in_flight_  = 0;
    writes_in_flight_      = 0;
    write_slabs_in_flight_ = 0;
    promotion_landed_      = true; // waiting requests quote the empty cache again
    due_.store(false, std::memory_order_relaxed);
}

} // namespace infernix::runtime::prefix_cache
