#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>

namespace infernix {

// Whether a ReadOnlyFile maps the whole file. A file read only through read_direct* must not be
// mapped: on Windows, every unbuffered read of a file with a mapped data section pays the cache
// manager's coherency work, which cut random 4 KiB reads of the n-gram volume from ~220K/s to
// ~56K/s (Optane P5800X, 2026-10-05).
enum class FileMapping : std::uint8_t {
    Whole, // mapped_bytes() is the whole file
    None,  // nothing mapped: mapped_bytes() is empty; for read_direct* only
};

class ReadOnlyFile {
public:
    explicit ReadOnlyFile(const std::filesystem::path& path, FileMapping mapping = FileMapping::Whole);
    ~ReadOnlyFile();

    ReadOnlyFile(ReadOnlyFile&&) noexcept;
    ReadOnlyFile& operator=(ReadOnlyFile&&) noexcept;
    ReadOnlyFile(const ReadOnlyFile&)            = delete;
    ReadOnlyFile& operator=(const ReadOnlyFile&) = delete;

    std::span<const std::byte> mapped_bytes() const noexcept;
    // Size of the file on disk now; external truncation or extension after open is observed,
    // matching the short-read behavior of positional pread on POSIX.
    [[nodiscard]] std::uint64_t current_bytes() const noexcept;
    // Unbuffered read of `destination` (4 KiB-aligned offset, size and address) from `offset`;
    // returns the bytes read contiguously from `offset` (fewer only at the end of the file). On
    // Windows a read of more than 8 MiB keeps up to 16 such blocks in flight.
    std::size_t read_direct(std::uint64_t offset, std::span<std::byte> destination) const;

    // Unbuffered reads of `block_bytes` at each offset (read_direct's alignment rules), at most
    // ring.size() / block_bytes in flight, each into a free block of `ring` (block-aligned), so
    // their device latency overlaps (Windows: overlapped reads; Linux: kernel AIO; other POSIX
    // systems read serially). consume(i, bytes) runs on the calling thread as read i completes, in
    // completion order; its ring block is reused once consume returns. Returns when
    // every read has completed; a failed or short read stops issuing, waits for the reads in flight
    // and throws.
    void read_direct_blocks(std::span<const std::uint64_t> offsets, std::size_t block_bytes, std::span<std::byte> ring,
                            const std::function<void(std::size_t, std::span<const std::byte>)>& consume) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace infernix
