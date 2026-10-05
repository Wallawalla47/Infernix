#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <system_error>
#include <vector>

namespace ninfer {

// Asynchronous unbuffered reads of file byte ranges into caller memory, for reading expert records
// from an artifact in place (design §19.3.7, the SSD expert tier). Files are opened unbuffered and
// without write sharing, so an in-place rewrite fails for the writer while a reader holds them.
//
// Reads are of two classes: demand reads are issued before prefetch reads, and each class has its
// own cap of reads in flight under a common cap. A prefetch read is issued as sub-reads of at most
// `prefetch_split` bytes, so a demand read never queues behind a whole prefetched record.
//
// Failures: a transient error (out of system resources or quota) is retried with backoff (1, 2, 4,
// ... ms) while the read's first error is younger than `transient_bound`; any other error, and a
// read still in flight after `timeout` (cancelled first), is retried once; a short read (end of
// file) fails at once. A request completes once, with its worst sub-read's status.
//
// The queue is used by one thread: submit, poll and cancel_all are not synchronized. On POSIX the
// reads run on a small pool of pread threads; an issued read cannot be cancelled there, so the
// timeout is not enforced and cancel_all waits for reads in flight.
class DirectReadQueue {
public:
    static constexpr std::size_t kAlignment = 4096; // offsets, sizes and destinations

    enum class Priority : std::uint8_t { Demand = 0, Prefetch = 1 };
    enum class Status : std::uint8_t { Ok, Failed, TimedOut, Cancelled };

    struct Options {
        std::uint32_t demand_depth   = 8;
        std::uint32_t prefetch_depth = 1;
        std::uint32_t total_depth    = 9;
        std::size_t prefetch_split   = 256 * 1024;
        std::chrono::milliseconds timeout{1500};
        std::chrono::milliseconds transient_bound{1500};
        std::uint32_t posix_workers = 4;
    };

    struct Read {
        std::uint32_t file   = 0; // from open()
        std::uint64_t offset = 0;
        std::size_t bytes    = 0;
        void* destination    = nullptr;
        Priority priority    = Priority::Demand;
        std::uint64_t tag    = 0; // returned with the completion
    };

    struct Completion {
        std::uint64_t tag = 0;
        Status status     = Status::Ok;
        std::error_code error;     // set unless Ok
        std::uint32_t attempts = 0; // issues of the sub-read issued most often
        std::uint64_t latency_ns = 0; // submit to completion
    };

    struct Stats {
        std::uint64_t issued = 0, retries = 0, transient_retries = 0, timeouts = 0;
    };

    // Test seam: issue number n (1-based, retries counted) fails for `fail_count` consecutive
    // issues with a transient or a hard error; issues from `stall_read` (`stall_count` of them)
    // never complete until their timeout cancels them; every completion is held until `delay`
    // after its issue. 0 disables a fault.
    struct Faults {
        std::uint64_t fail_read   = 0;
        std::uint32_t fail_count  = 1;
        bool transient            = false;
        std::uint64_t stall_read  = 0;
        std::uint32_t stall_count = 1;
        std::chrono::microseconds delay{0};
    };

    explicit DirectReadQueue(Options options);
    DirectReadQueue() : DirectReadQueue(Options{}) {}
    ~DirectReadQueue(); // cancels every read and waits for those in flight

    DirectReadQueue(const DirectReadQueue&)            = delete;
    DirectReadQueue& operator=(const DirectReadQueue&) = delete;

    // Opens a file for unbuffered reads (no write sharing); throws std::system_error.
    std::uint32_t open(const std::filesystem::path& path);
    [[nodiscard]] std::uint64_t file_bytes(std::uint32_t file) const;

    // Queues a read; throws std::invalid_argument for an unknown file, a misaligned or empty range,
    // or a misaligned destination.
    void submit(const Read& read);

    // Issues queued reads up to the caps, waits up to `wait` for the first completion, and
    // appends every request completed since the last call. Returns the number appended.
    std::size_t poll(std::vector<Completion>& out, std::chrono::microseconds wait);

    // Every queued read completes Cancelled; reads in flight are cancelled and waited for (each
    // completes Cancelled, or Ok when it finished first). The completions arrive with the next poll.
    void cancel_all();

    [[nodiscard]] std::size_t pending() const noexcept; // requests not yet returned by poll
    [[nodiscard]] const Stats& stats() const noexcept;
    void set_faults(const Faults& faults);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
