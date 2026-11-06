#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <system_error>
#include <vector>

namespace infernix {

// Unbuffered scratch storage on a local drive: the prefix cache's SSD tier
// (docs/maintainer/hybrid-prefix-cache-spec.md §5.7). The file is created in a directory with its
// whole size reserved, read and written without the OS page cache (so it holds no RAM), and removed
// when this object is destroyed or the process ends, a crash included (Windows deletes it on close,
// POSIX unlinks it at once). Its bytes are never made durable: it is a cache.
//
// Transfers are batches of reads or of writes between the file and caller memory (4 KiB-aligned
// offsets, sizes and addresses). Dedicated I/O threads run them: up to `read_depth` reads in flight,
// then up to `write_depth` writes, and no write is issued while a read waits, so reads never queue
// behind a backlog of writes. A failed I/O is retried `retries` times; a batch completes once, after
// all of its I/Os, with the first error of any. submit and poll may be called from any thread.
class DirectScratchFile {
public:
    static constexpr std::size_t kAlignment = 4096;

    struct Options {
        std::uint32_t read_depth  = 8;
        std::uint32_t write_depth = 2;
        std::uint32_t retries     = 2;
    };

    enum class Op : std::uint8_t { Read, Write };

    // One contiguous transfer: `bytes` at file `offset` from or into `memory`.
    struct Io {
        std::uint64_t offset = 0;
        std::size_t bytes    = 0;
        std::byte* memory    = nullptr;
    };

    struct Batch {
        std::uint64_t tag = 0; // returned with the completion
        Op op             = Op::Read;
        std::vector<Io> ios;
    };

    struct Completion {
        std::uint64_t tag = 0;
        Op op             = Op::Read;
        bool ok           = true;
        std::error_code error;        // the first failure, when !ok
        std::uint64_t bytes       = 0; // the batch's bytes
        std::uint64_t nanoseconds = 0; // first issue to last completion
    };

    // Creates the file (a new, uniquely named file in `directory`) and reserves `bytes` (a positive
    // multiple of 4 KiB) for it. Throws std::system_error (a missing directory, no space, a drive that
    // refuses unbuffered I/O) or std::invalid_argument.
    DirectScratchFile(const std::filesystem::path& directory, std::uint64_t bytes, Options options);
    DirectScratchFile(const std::filesystem::path& directory, std::uint64_t bytes)
        : DirectScratchFile(directory, bytes, Options{}) {}
    // Waits for the I/O in flight; batches not yet issued are dropped unreported.
    ~DirectScratchFile();

    DirectScratchFile(const DirectScratchFile&)            = delete;
    DirectScratchFile& operator=(const DirectScratchFile&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept;
    [[nodiscard]] std::uint64_t bytes() const noexcept;

    // Queues a batch; throws std::invalid_argument for an empty batch, a misaligned or empty I/O, or
    // one past the end of the file.
    void submit(Batch batch);
    // Appends every batch finished since the last call; never blocks. Returns the number appended.
    std::size_t poll(std::vector<Completion>& out);
    // Blocks until every submitted batch has finished (it is still returned by poll).
    void wait_idle();
    // Batches submitted and not yet returned by poll.
    [[nodiscard]] std::size_t outstanding() const noexcept;
    // Called on an I/O thread after each batch finishes (no lock held); empty to clear.
    void set_waker(std::function<void()> waker);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace infernix
