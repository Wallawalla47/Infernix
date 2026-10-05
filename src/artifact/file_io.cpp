#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <cerrno>
#include <cstring>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

#ifndef _WIN32
[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}
#else
// The shared ReadOnlyFile abstraction reports platform failures as std::system_error; the
// artifact layer promises ArtifactError, mirroring the POSIX "path: open: ..." message.
ReadOnlyFile open_read_only(const std::filesystem::path& path) {
    try {
        return ReadOnlyFile(path, FileMapping::None);
    } catch (const std::exception& error) {
        throw ArtifactError(error.what());
    }
}
#endif

} // namespace

#ifdef _WIN32
// The Win32 path delegates to the shared read-only file abstraction with nothing mapped and no
// buffered handle: every read is unbuffered, so neither a mapping nor the system cache holds the
// file's data and the materializer's direct reads pay no cache-coherency work (a mapped file cut
// unbuffered read rates by 12 % for 8 MiB reads and 75 % for 4 KiB ones, design §19.3.8 F9). The
// POSIX path keeps positional pread on a raw fd.
InputFile::InputFile(std::filesystem::path path)
    : path_(std::move(path)), file_(open_read_only(path_)) {
    bytes_ = file_.current_bytes();
}
#endif

#ifndef _WIN32
InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail(path_, "open"); }

    struct stat status {};

    if (::fstat(fd_, &status) != 0) {
        const auto error = errno;
        ::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
}
#endif

InputFile::~InputFile() {
#ifndef _WIN32
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
#endif
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
#ifdef _WIN32
    // The mapped view keeps the size observed at open; a file truncated after open must still
    // reject reads past its current end, as the POSIX short read below does.
    const auto limit = std::min(bytes_, file_.current_bytes());
    if (offset > limit || destination.size() > limit - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    constexpr std::uint64_t kAlign = kPayloadAlignment;
    while (!destination.empty()) {
        // An aligned range of whole blocks reads straight into the destination.
        if (offset % kAlign == 0 && destination.size() % kAlign == 0 &&
            reinterpret_cast<std::uintptr_t>(destination.data()) % kAlign == 0) {
            const std::size_t got = file_.read_direct(offset, destination);
            if (got != destination.size()) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
            return;
        }
        // Otherwise through an aligned bounce buffer of at most 8 MiB per thread.
        constexpr std::size_t kBounce = 8ULL << 20;
        thread_local std::vector<std::byte> storage(kBounce + kAlign);
        const auto base     = reinterpret_cast<std::uintptr_t>(storage.data());
        std::byte* bounce   = storage.data() + (kAlign - base % kAlign) % kAlign;
        const std::uint64_t first = offset - offset % kAlign;
        const auto skip     = static_cast<std::size_t>(offset - first);
        const std::size_t span = std::min<std::size_t>(kBounce, (skip + destination.size() + kAlign - 1) / kAlign * kAlign);
        const std::size_t got  = file_.read_direct(first, std::span<std::byte>(bounce, span));
        if (got <= skip) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        const std::size_t count = std::min(destination.size(), got - skip);
        std::copy_n(bounce + skip, count, destination.data());
        offset += count;
        destination = destination.subspan(count);
    }
#else
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read  = ::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
#endif
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
#ifdef _WIN32
    return file_.read_direct(offset, destination);
#else
    if (direct_fd_ < 0) {
        direct_fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (direct_fd_ < 0) { fail(path_, "open direct"); }
    }
    ssize_t read;
    do {
        read = ::pread(direct_fd_, destination.data(), destination.size(), file_offset(offset));
    } while (read < 0 && errno == EINTR);
    if (read < 0) { fail(path_, "direct pread"); }
    return static_cast<std::size_t>(read);
#endif
}

} // namespace ninfer::artifact
