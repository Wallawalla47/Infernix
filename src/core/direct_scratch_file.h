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
// With Options::kept_file the file is the one named instead, opened for this process alone, kept when
// it closes and reused at the next start: its bytes stay (kept_content) when it already exists with
// this size, so a prefix cache file can name slabs in it (§5.5). flush() makes the writes before it
// durable.
//
// An Io may name a hash buffer: each `hash_stride` bytes of the transfer get their content_hash, of the
// memory before a write or of the bytes read after a read, computed on the I/O thread.
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
        // Empty: a uniquely named scratch file in the directory, removed at the end. Otherwise this file,
        // kept and reused (above).
        std::filesystem::path kept_file;
    };

    enum class Op : std::uint8_t { Read, Write };

    // One contiguous transfer: `bytes` at file `offset` from or into `memory`. With `hashes`, the I/O
    // thread writes bytes / hash_stride content hashes there (hash_stride a positive divisor of bytes).
    struct Io {
        std::uint64_t offset      = 0;
        std::size_t bytes         = 0;
        std::byte* memory         = nullptr;
        std::uint64_t* hashes     = nullptr;
        std::size_t hash_stride   = 0;
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

    // Creates the file (a new, uniquely named file in `directory`, or opens Options::kept_file) and
    // reserves `bytes` (a positive multiple of 4 KiB) for it. Throws std::system_error (a missing
    // directory, no space, a drive that refuses unbuffered I/O, a kept file another process holds) or
    // std::invalid_argument.
    DirectScratchFile(const std::filesystem::path& directory, std::uint64_t bytes, Options options);
    DirectScratchFile(const std::filesystem::path& directory, std::uint64_t bytes)
        : DirectScratchFile(directory, bytes, Options{}) {}
    // Waits for the I/O in flight; batches not yet issued are dropped unreported.
    ~DirectScratchFile();

    DirectScratchFile(const DirectScratchFile&)            = delete;
    DirectScratchFile& operator=(const DirectScratchFile&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept;
    [[nodiscard]] std::uint64_t bytes() const noexcept;
    // A kept file that already existed with this size: its bytes are the previous run's.
    [[nodiscard]] bool kept_content() const noexcept;
    // Makes every write completed before the call durable (FlushFileBuffers / fdatasync); throws
    // std::system_error when the drive refuses.
    void flush();

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
