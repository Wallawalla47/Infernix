#include "core/direct_scratch_file.h"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <cerrno>
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace infernix {
namespace {

using Clock = std::chrono::steady_clock;

// One I/O call never moves more than this (a DWORD on Windows; the SSD tier issues at most 8 MiB).
constexpr std::size_t kMaxIoBytes = std::size_t{1} << 30;

std::atomic<std::uint64_t> next_file_serial{0};

std::filesystem::path unique_name(const std::filesystem::path& directory) {
#ifdef _WIN32
    const auto pid = static_cast<unsigned long>(::GetCurrentProcessId());
#else
    const auto pid = static_cast<unsigned long>(::getpid());
#endif
    return directory / ("infernix-context-" + std::to_string(pid) + "-" +
                        std::to_string(next_file_serial.fetch_add(1)) + ".bin");
}

#ifdef _WIN32
std::error_code win32_error(DWORD code) { return {static_cast<int>(code), std::system_category()}; }
[[noreturn]] void throw_win32(DWORD code, const std::string& what) {
    throw std::system_error(static_cast<int>(code), std::system_category(), what);
}
#else
[[noreturn]] void throw_errno(int code, const std::string& what) {
    throw std::system_error(code, std::generic_category(), what);
}
#endif

} // namespace

struct DirectScratchFile::Impl {
    struct BatchState {
        std::uint64_t tag = 0;
        Op op             = Op::Read;
        std::uint32_t remaining = 0;
        bool ok                 = true;
        std::error_code error;
        std::uint64_t bytes = 0;
        Clock::time_point first_issue{};
        bool issued = false;
    };
    struct Pending {
        std::size_t batch = 0;
        Io io;
    };

    Options options;
    std::filesystem::path file_path;
    std::uint64_t file_bytes = 0;
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif

    mutable std::mutex mutex;
    std::condition_variable work_ready;
    std::condition_variable idle;
    std::deque<Pending> reads;
    std::deque<Pending> writes;
    std::deque<BatchState> batches; // indexed by slot
    std::vector<std::size_t> free_batches;
    std::vector<Completion> done;
    std::uint32_t reads_in_flight  = 0;
    std::uint32_t writes_in_flight = 0;
    std::size_t live               = 0; // batches submitted and not finished
    bool stopping                  = false;
    std::function<void()> waker;
    std::vector<std::thread> threads;

    Impl(const std::filesystem::path& directory, std::uint64_t bytes, const Options& o) : options(o), file_bytes(bytes) {
        if (bytes == 0 || bytes % kAlignment != 0) {
            throw std::invalid_argument("DirectScratchFile: the size must be a positive multiple of 4 KiB");
        }
        if (options.read_depth == 0 || options.write_depth == 0) {
            throw std::invalid_argument("DirectScratchFile: the read and write depths must be positive");
        }
        std::error_code ec;
        if (!std::filesystem::is_directory(directory, ec)) {
            throw std::system_error(ec ? ec : std::make_error_code(std::errc::not_a_directory),
                                    "DirectScratchFile: " + directory.string() + " is not a directory");
        }
        file_path = unique_name(std::filesystem::absolute(directory));
        open_file();
        const std::uint32_t count = options.read_depth + options.write_depth;
        try {
            for (std::uint32_t i = 0; i < count; ++i) { threads.emplace_back([this] { work(); }); }
        } catch (...) {
            stop();
            close_file();
            throw;
        }
    }

    ~Impl() {
        stop();
        close_file();
    }

    void stop() noexcept {
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        work_ready.notify_all();
        for (auto& thread : threads) {
            if (thread.joinable()) { thread.join(); }
        }
        threads.clear();
    }

    // ---- platform -------------------------------------------------------------------------------
    void open_file() {
#ifdef _WIN32
        // No sharing, removed when the last handle closes (also at a crash: the OS closes it).
        handle = ::CreateFileW(file_path.c_str(), GENERIC_READ | GENERIC_WRITE | DELETE, 0, nullptr, CREATE_NEW,
                               FILE_ATTRIBUTE_TEMPORARY | FILE_ATTRIBUTE_NOT_CONTENT_INDEXED | FILE_FLAG_NO_BUFFERING |
                                   FILE_FLAG_OVERLAPPED | FILE_FLAG_DELETE_ON_CLOSE,
                               nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            throw_win32(::GetLastError(), "DirectScratchFile: CreateFileW " + file_path.string());
        }
        try {
            // Unbuffered transfers must be whole sectors; every I/O here is a multiple of 4 KiB.
            FILE_STORAGE_INFO storage{};
            if (::GetFileInformationByHandleEx(handle, FileStorageInfo, &storage, sizeof(storage)) &&
                (storage.LogicalBytesPerSector == 0 || kAlignment % storage.LogicalBytesPerSector != 0)) {
                throw std::system_error(std::make_error_code(std::errc::not_supported),
                                        "DirectScratchFile: " + file_path.string() + " has " +
                                            std::to_string(storage.LogicalBytesPerSector) + "-byte sectors");
            }
            // Reserving the allocation fails at once when the drive lacks the space, and keeps the file
            // in few extents. The valid data length stays 0: nothing is written to reserve it.
            FILE_ALLOCATION_INFO allocation{};
            allocation.AllocationSize.QuadPart = static_cast<LONGLONG>(file_bytes);
            if (!::SetFileInformationByHandle(handle, FileAllocationInfo, &allocation, sizeof(allocation))) {
                throw_win32(::GetLastError(), "DirectScratchFile: reserving " + std::to_string(file_bytes) +
                                                  " bytes for " + file_path.string());
            }
            FILE_END_OF_FILE_INFO end{};
            end.EndOfFile.QuadPart = static_cast<LONGLONG>(file_bytes);
            if (!::SetFileInformationByHandle(handle, FileEndOfFileInfo, &end, sizeof(end))) {
                throw_win32(::GetLastError(), "DirectScratchFile: sizing " + file_path.string());
            }
        } catch (...) {
            close_file();
            throw;
        }
#else
        int flags = O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC;
#    ifdef O_DIRECT
        flags |= O_DIRECT;
#    endif
        fd = ::open(file_path.c_str(), flags, 0600);
#    ifdef O_DIRECT
        // A filesystem without direct I/O (tmpfs, some FUSE mounts) refuses O_DIRECT: the file then
        // goes through the page cache with the same alignment rules.
        if (fd < 0 && (errno == EINVAL || errno == EOPNOTSUPP)) {
            fd = ::open(file_path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        }
#    endif
        if (fd < 0) { throw_errno(errno, "DirectScratchFile: open " + file_path.string()); }
        // Unlinked at once: the space returns when the descriptor closes, whatever ends the process.
        (void)::unlink(file_path.c_str());
        const int reserved = ::posix_fallocate(fd, 0, static_cast<off_t>(file_bytes));
        if (reserved != 0 && reserved != EOPNOTSUPP && reserved != EINVAL) {
            const int error = reserved;
            close_file();
            throw_errno(error, "DirectScratchFile: reserving " + std::to_string(file_bytes) + " bytes for " +
                                   file_path.string());
        }
        if (reserved != 0 && ::ftruncate(fd, static_cast<off_t>(file_bytes)) != 0) {
            const int error = errno;
            close_file();
            throw_errno(error, "DirectScratchFile: sizing " + file_path.string());
        }
#endif
    }

    void close_file() noexcept {
#ifdef _WIN32
        if (handle != INVALID_HANDLE_VALUE) {
            ::CloseHandle(handle);
            handle = INVALID_HANDLE_VALUE;
        }
#else
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
#endif
    }

#ifdef _WIN32
    // A positional transfer on the calling thread; `event` is the thread's own.
    std::error_code transfer(Op op, const Io& io, HANDLE event) const {
        OVERLAPPED overlapped{};
        overlapped.Offset     = static_cast<DWORD>(io.offset & 0xffffffffULL);
        overlapped.OffsetHigh = static_cast<DWORD>(io.offset >> 32U);
        overlapped.hEvent     = event;
        const auto count      = static_cast<DWORD>(io.bytes);
        // A write past the file's valid data length completes synchronously (NTFS zeroes from it to
        // the write first); the SSD tier fills the file from its start, so those writes are in order.
        const BOOL started = op == Op::Read ? ::ReadFile(handle, io.memory, count, nullptr, &overlapped)
                                            : ::WriteFile(handle, io.memory, count, nullptr, &overlapped);
        if (!started) {
            const DWORD error = ::GetLastError();
            if (error != ERROR_IO_PENDING) { return win32_error(error); }
        }
        DWORD moved = 0;
        if (!::GetOverlappedResult(handle, &overlapped, &moved, TRUE)) { return win32_error(::GetLastError()); }
        if (moved != count) { return std::make_error_code(std::errc::io_error); }
        return {};
    }
#else
    std::error_code transfer(Op op, const Io& io) const {
        std::size_t total = 0;
        while (total < io.bytes) {
            const ssize_t moved =
                op == Op::Read ? ::pread(fd, io.memory + total, io.bytes - total, static_cast<off_t>(io.offset + total))
                               : ::pwrite(fd, io.memory + total, io.bytes - total, static_cast<off_t>(io.offset + total));
            if (moved < 0) {
                if (errno == EINTR) { continue; }
                return {errno, std::generic_category()};
            }
            if (moved == 0) { return std::make_error_code(std::errc::io_error); }
            total += static_cast<std::size_t>(moved);
        }
        return {};
    }
#endif

    // ---- scheduling -----------------------------------------------------------------------------
    // The next I/O this thread may issue (mutex held): a read below its depth, else a write below its
    // depth while no read waits.
    bool take(Pending& out, Op& op) {
        if (!reads.empty() && reads_in_flight < options.read_depth) {
            out = reads.front();
            reads.pop_front();
            op = Op::Read;
            ++reads_in_flight;
            return true;
        }
        if (reads.empty() && !writes.empty() && writes_in_flight < options.write_depth) {
            out = writes.front();
            writes.pop_front();
            op = Op::Write;
            ++writes_in_flight;
            return true;
        }
        return false;
    }

    void work() {
#ifdef _WIN32
        HANDLE event = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
#endif
        for (;;) {
            Pending pending;
            Op op = Op::Read;
            {
                std::unique_lock lock(mutex);
                // A stop drops what is queued; an I/O already taken runs to its end.
                work_ready.wait(lock, [&] { return stopping || take(pending, op); });
                if (stopping) { break; }
                BatchState& batch = batches[pending.batch];
                if (!batch.issued) {
                    batch.issued      = true;
                    batch.first_issue = Clock::now();
                }
            }
            std::error_code error;
            for (std::uint32_t attempt = 0; attempt <= options.retries; ++attempt) {
#ifdef _WIN32
                if (event == nullptr) {
                    error = win32_error(::GetLastError());
                    break;
                }
                error = transfer(op, pending.io, event);
#else
                error = transfer(op, pending.io);
#endif
                if (!error) { break; }
            }
            finish(pending, op, error);
        }
#ifdef _WIN32
        if (event != nullptr) { ::CloseHandle(event); }
#endif
    }

    void finish(const Pending& pending, Op op, const std::error_code& error) {
        std::function<void()> wake;
        {
            std::lock_guard lock(mutex);
            --(op == Op::Read ? reads_in_flight : writes_in_flight);
            BatchState& batch = batches[pending.batch];
            if (error && batch.ok) {
                batch.ok    = false;
                batch.error = error;
            }
            if (--batch.remaining == 0) {
                done.push_back(Completion{
                    .tag         = batch.tag,
                    .op          = batch.op,
                    .ok          = batch.ok,
                    .error       = batch.error,
                    .bytes       = batch.bytes,
                    .nanoseconds = static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - batch.first_issue).count()),
                });
                free_batches.push_back(pending.batch);
                --live;
                wake = waker;
                if (live == 0) { idle.notify_all(); }
            }
        }
        // Another thread may take the next I/O now (a finished read can release a waiting write).
        work_ready.notify_all();
        if (wake) { wake(); }
    }

    void submit(Batch batch) {
        if (batch.ios.empty()) { throw std::invalid_argument("DirectScratchFile: an empty batch"); }
        std::uint64_t bytes = 0;
        for (const Io& io : batch.ios) {
            if (io.bytes == 0 || io.bytes > kMaxIoBytes || io.bytes % kAlignment != 0 || io.offset % kAlignment != 0 ||
                reinterpret_cast<std::uintptr_t>(io.memory) % kAlignment != 0 || io.offset > file_bytes ||
                io.bytes > file_bytes - io.offset) {
                throw std::invalid_argument(
                    "DirectScratchFile: transfers need 4 KiB-aligned offsets, sizes and addresses inside the file "
                    "(offset " + std::to_string(io.offset) + ", " + std::to_string(io.bytes) + " bytes, address " +
                    std::to_string(reinterpret_cast<std::uintptr_t>(io.memory)) + ", file " + std::to_string(file_bytes) +
                    " bytes)");
            }
            bytes += io.bytes;
        }
        {
            std::lock_guard lock(mutex);
            std::size_t slot = 0;
            if (!free_batches.empty()) {
                slot = free_batches.back();
                free_batches.pop_back();
            } else {
                slot = batches.size();
                batches.emplace_back();
            }
            batches[slot] = BatchState{.tag       = batch.tag,
                                       .op        = batch.op,
                                       .remaining = static_cast<std::uint32_t>(batch.ios.size()),
                                       .ok        = true,
                                       .error     = {},
                                       .bytes     = bytes};
            std::deque<Pending>& queue = batch.op == Op::Read ? reads : writes;
            for (const Io& io : batch.ios) { queue.push_back(Pending{slot, io}); }
            ++live;
        }
        work_ready.notify_all();
    }
};

DirectScratchFile::DirectScratchFile(const std::filesystem::path& directory, std::uint64_t bytes, Options options)
    : impl_(std::make_unique<Impl>(directory, bytes, options)) {}

DirectScratchFile::~DirectScratchFile() = default;

const std::filesystem::path& DirectScratchFile::path() const noexcept { return impl_->file_path; }

std::uint64_t DirectScratchFile::bytes() const noexcept { return impl_->file_bytes; }

void DirectScratchFile::submit(Batch batch) { impl_->submit(std::move(batch)); }

std::size_t DirectScratchFile::poll(std::vector<Completion>& out) {
    std::lock_guard lock(impl_->mutex);
    const std::size_t n = impl_->done.size();
    out.insert(out.end(), impl_->done.begin(), impl_->done.end());
    impl_->done.clear();
    return n;
}

void DirectScratchFile::wait_idle() {
    std::unique_lock lock(impl_->mutex);
    impl_->idle.wait(lock, [&] { return impl_->live == 0; });
}

std::size_t DirectScratchFile::outstanding() const noexcept {
    std::lock_guard lock(impl_->mutex);
    return impl_->live + impl_->done.size();
}

void DirectScratchFile::set_waker(std::function<void()> waker) {
    std::lock_guard lock(impl_->mutex);
    impl_->waker = std::move(waker);
}

} // namespace infernix
