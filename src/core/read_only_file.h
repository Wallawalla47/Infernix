#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace ninfer {

class ReadOnlyFile {
public:
    explicit ReadOnlyFile(const std::filesystem::path& path);
    ~ReadOnlyFile();

    ReadOnlyFile(ReadOnlyFile&&) noexcept;
    ReadOnlyFile& operator=(ReadOnlyFile&&) noexcept;
    ReadOnlyFile(const ReadOnlyFile&)            = delete;
    ReadOnlyFile& operator=(const ReadOnlyFile&) = delete;

    std::span<const std::byte> mapped_bytes() const noexcept;
    // Size of the file on disk now; external truncation or extension after open is observed,
    // matching the short-read behavior of positional pread on POSIX.
    [[nodiscard]] std::uint64_t current_bytes() const noexcept;
    std::size_t read_direct(std::uint64_t offset, std::span<std::byte> destination) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
