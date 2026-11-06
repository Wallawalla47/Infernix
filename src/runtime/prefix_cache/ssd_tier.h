#pragma once

#include "core/direct_scratch_file.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/ssd_stats.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace infernix::runtime::prefix_cache {

class SsdTier;
[[nodiscard]] SsdTierStats ssd_tier_stats(const SsdTier* tier, const PrefixIndexStats& index, std::size_t slab_bytes,
                                          std::uint32_t slabs) noexcept;

// The prefix cache's SSD tier (docs/maintainer/hybrid-prefix-cache-spec.md §5.7), shared by every
// model binding: it owns the scratch file and moves the bytes of the index's SSD jobs between Host
// slabs and the file. The index decides what moves; this class decides when to write back, starts
// promotions for the sources the bindings choose, and applies landed transfers on the Engine worker
// (the index's only mutator). Transfers run on the file's own I/O threads, reads ahead of writes.
//
// Write-back runs while the Host tier's lossless room (PrefixCacheIndex::host_lossless_slabs) is
// below three quarters of it: at most a quarter of the Host tier holds content the SSD does not, so
// a burst that takes most of the tier at once (one prefill unit's captures and blocks, a promotion)
// displaces written copies rather than losing anything.
class SsdTier {
public:
    struct Counters {
        std::uint64_t write_bytes     = 0;
        std::uint64_t read_bytes      = 0;
        std::uint64_t image_writes    = 0;
        std::uint64_t block_writes    = 0;
        std::uint64_t image_reads     = 0;
        std::uint64_t block_reads     = 0;
        std::uint64_t promotions      = 0; // promotions that landed
        std::uint64_t write_failures  = 0;
        std::uint64_t read_failures   = 0;
        double read_bytes_per_second  = 0.0; // the last promotion's measured rate
        double write_bytes_per_second = 0.0; // the last write-back's
    };

    enum class Promotion : std::uint8_t {
        Ready,       // nothing of the source is only on the SSD
        Pending,     // reads are in flight: quote it again once poll reports a landing
        Unavailable, // the Host tier cannot take it now
    };

    // `host_slab(id)` is the address of Host slab `id` (4 KiB-aligned); a slab under a transfer stays
    // put because the index keeps its entry pinned. Creates the scratch file in `directory`.
    SsdTier(const std::filesystem::path& directory, std::uint32_t slabs, std::size_t slab_bytes,
            std::function<std::byte*(std::uint32_t)> host_slab);
    ~SsdTier(); // waits for the transfers in flight

    SsdTier(const SsdTier&)            = delete;
    SsdTier& operator=(const SsdTier&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return file_.path(); }
    [[nodiscard]] std::uint64_t capacity_bytes() const noexcept { return file_.bytes(); }
    [[nodiscard]] const Counters& counters() const noexcept { return counters_; }
    // Why the last refused transfer was refused (empty: none).
    [[nodiscard]] const std::string& last_error() const noexcept { return last_error_; }

    // Called on an I/O thread after each finished transfer batch (the Engine's maintenance waker).
    void set_waker(std::function<void()> waker);

    // Applies every landed transfer to `index`, then starts write-backs while the Host tier's
    // lossless room is below its target. Returns whether a promotion landed (or failed) since the
    // last call: a request waiting for it can be quoted again.
    bool poll(PrefixCacheIndex& index);
    // Applies every landed transfer, starting nothing (for polls inside an admission); the landings
    // are still reported by the next poll.
    void settle(PrefixCacheIndex& index) { apply(index); }
    // Starts reading the parts of `snapshot` only on the SSD into the Host tier.
    [[nodiscard]] Promotion promote(PrefixCacheIndex& index, SnapshotRef snapshot);
    [[nodiscard]] bool promoting() const noexcept { return promotions_in_flight_ != 0; }
    // Whether poll would start a write-back (the idle Engine runs its maintenance for it): the Host
    // tier's lossless room was below target with a write slot free and no recent refusal when the
    // tier last applied transfers. Lock-free: the Engine reads it outside its execution lock.
    [[nodiscard]] bool write_back_due() const noexcept { return due_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool busy() const noexcept { return !jobs_.empty(); }
    // Waits for every transfer in flight and applies it, starting nothing new (before a Host tier
    // resize, a save or the cache's teardown).
    void drain(PrefixCacheIndex& index);
    // Waits for the transfers in flight and forgets them: the index is being rebuilt empty.
    void abandon() noexcept;

private:
    struct Job {
        bool promotion = false;
        SsdJob job;
        std::uint64_t host_slabs = 0;
    };

    void apply(PrefixCacheIndex& index);
    void start_write_backs(PrefixCacheIndex& index);
    [[nodiscard]] std::uint64_t write_back_target(const PrefixCacheIndex& index) const noexcept;
    [[nodiscard]] bool compute_due(const PrefixCacheIndex& index) const noexcept;
    std::atomic<bool> due_{false};
    [[nodiscard]] DirectScratchFile::Batch batch(std::uint64_t tag, DirectScratchFile::Op op,
                                                 const SsdJob& job) const;

    DirectScratchFile file_;
    std::size_t slab_bytes_ = 0;
    std::function<std::byte*(std::uint32_t)> host_slab_;
    std::unordered_map<std::uint64_t, Job> jobs_;
    std::uint64_t next_tag_ = 1;
    std::uint32_t promotions_in_flight_ = 0;
    std::uint32_t writes_in_flight_     = 0;
    std::uint64_t write_slabs_in_flight_ = 0;
    bool promotion_landed_              = false;
    std::chrono::steady_clock::time_point next_write_attempt_{};
    std::vector<DirectScratchFile::Completion> completions_;
    Counters counters_;
    std::string last_error_;
};

} // namespace infernix::runtime::prefix_cache
