#pragma once

#include "core/direct_scratch_file.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/ssd_stats.h"
#include "runtime/prefix_cache/ssd_write_budget.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
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
//
// With the codec, an image is written as a stream of fp32_row_codec payloads (one per kept image
// slab) when that takes fewer SSD slabs: a worker thread encodes it into a staging buffer before the
// write is issued. While write-back has slack (lossless room at least half its target) one packed
// write runs at a time and nothing beside it; under pressure images go raw, beside a packed write in
// flight, so write-back keeps the raw pace. A packed promotion reads the stream into a staging buffer
// and decodes it into the Host image slabs on the codec workers.
class SsdTier {
public:
    struct Counters {
        std::uint64_t write_bytes        = 0;
        std::uint64_t read_bytes         = 0;
        std::uint64_t image_writes       = 0;
        std::uint64_t block_writes       = 0;
        std::uint64_t image_reads        = 0;
        std::uint64_t block_reads        = 0;
        std::uint64_t promotions         = 0; // promotions that landed
        std::uint64_t write_failures     = 0;
        std::uint64_t read_failures      = 0;
        std::uint64_t packed_images      = 0; // images written as codec streams
        std::uint64_t packed_saved_bytes = 0; // SSD bytes those streams saved
        std::uint64_t policy_nanoseconds = 0; // spent choosing write-backs (start_write_backs)
        double read_bytes_per_second     = 0.0; // the last promotion's measured rate
        double write_bytes_per_second    = 0.0; // the last write-back's
    };

    enum class Promotion : std::uint8_t {
        Ready,       // nothing of the source is only on the SSD
        Pending,     // reads are in flight: quote it again once poll reports a landing
        Unavailable, // the Host tier cannot take it now
    };

    // `host_slab(id)` is the address of Host slab `id` (4 KiB-aligned); a slab under a transfer stays
    // put because the index keeps its entry pinned. Creates the scratch file in `directory`, and with
    // a positive `write_bytes_per_hour` keeps the write budget's level there (SsdWriteBudget). `codec`
    // packs images (fp32_row_codec) when that saves SSD slabs.
    SsdTier(const std::filesystem::path& directory, std::uint32_t slabs, std::size_t slab_bytes,
            std::function<std::byte*(std::uint32_t)> host_slab, std::uint64_t write_bytes_per_hour = 0,
            bool codec = false);
    ~SsdTier(); // waits for the transfers and the codec work in flight

    SsdTier(const SsdTier&)            = delete;
    SsdTier& operator=(const SsdTier&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return file_.path(); }
    [[nodiscard]] std::uint64_t capacity_bytes() const noexcept { return file_.bytes(); }
    [[nodiscard]] const Counters& counters() const noexcept { return counters_; }
    // The write budget (absent: every candidate is written).
    [[nodiscard]] const std::optional<SsdWriteBudget>& budget() const noexcept { return budget_; }
    // Why the last refused transfer was refused (empty: none).
    [[nodiscard]] const std::string& last_error() const noexcept { return last_error_; }

    // Called on an I/O or codec thread after each finished transfer batch or codec job (the Engine's
    // maintenance waker).
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
    // Waits for every transfer and codec job in flight and applies it, starting nothing new (before a
    // Host tier resize, a save or the cache's teardown).
    void drain(PrefixCacheIndex& index);
    // Waits for the work in flight and forgets it: the index is being rebuilt empty.
    void abandon() noexcept;

private:
    class Workers;
    struct AlignedFree {
        void operator()(std::byte* p) const noexcept;
    };
    using Staging = std::unique_ptr<std::byte, AlignedFree>;
    enum class Phase : std::uint8_t { Io, Encoding, Decoding };
    struct Job {
        bool promotion = false;
        SsdJob job;
        std::uint64_t host_slabs = 0;
        Phase phase              = Phase::Io;
        Staging staging;              // the codec stream (packed images)
        std::size_t stream_bytes = 0; // its encoded length
        std::atomic<std::uint32_t> parts{0};
        std::atomic<bool> failed{false};
    };

    void apply(PrefixCacheIndex& index);
    void apply_codec(PrefixCacheIndex& index);
    void land_promotion(PrefixCacheIndex& index, Job& job, bool ok);
    void issue_write(PrefixCacheIndex& index, std::uint64_t tag, Job& job);
    void start_decode(std::uint64_t tag, Job& job);
    void start_write_backs(PrefixCacheIndex& index);
    [[nodiscard]] std::uint64_t write_back_target(const PrefixCacheIndex& index) const noexcept;
    [[nodiscard]] bool compute_due(const PrefixCacheIndex& index) const noexcept;
    [[nodiscard]] bool budget_still_blocked(const PrefixCacheIndex& index) const noexcept;
    [[nodiscard]] DirectScratchFile::Batch batch(std::uint64_t tag, DirectScratchFile::Op op, const Job& job) const;
    [[nodiscard]] Staging take_staging();
    void give_staging(Staging staging) noexcept;
    void codec_done(std::uint64_t tag);

    std::atomic<bool> due_{false};
    DirectScratchFile file_;
    std::size_t slab_bytes_ = 0;
    std::function<std::byte*(std::uint32_t)> host_slab_;
    std::unordered_map<std::uint64_t, std::unique_ptr<Job>> jobs_;
    std::uint64_t next_tag_              = 1;
    std::uint32_t promotions_in_flight_  = 0;
    std::uint32_t writes_in_flight_      = 0;
    std::uint64_t write_slabs_in_flight_ = 0;
    bool promotion_landed_               = false;
    std::chrono::steady_clock::time_point next_write_attempt_{};
    std::vector<DirectScratchFile::Completion> completions_;
    Counters counters_;
    std::string last_error_;
    std::optional<SsdWriteBudget> budget_;
    std::chrono::steady_clock::time_point last_budget_save_{};
    // A scan the budget declined entirely: not repeated before a snapshot is published (the index's
    // epoch moves) or the budget allows another image's bytes.
    bool budget_blocked_                = false;
    std::uint64_t blocked_epoch_        = 0;
    std::uint64_t blocked_available_    = 0;
    std::uint64_t blocked_retry_bytes_  = 0;
    // The codec: its workers, staging buffers (each holds a packed image's stream), whether a packed
    // write is in flight, and the jobs whose codec work finished (filled by the workers).
    std::unique_ptr<Workers> workers_;
    std::size_t staging_bytes_ = 0;
    std::vector<Staging> spare_staging_;
    bool packed_write_busy_ = false;
    std::mutex codec_mutex_;
    std::vector<std::uint64_t> codec_finished_;
    std::function<void()> waker_;
};

} // namespace infernix::runtime::prefix_cache
