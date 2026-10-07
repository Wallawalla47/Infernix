#include "core/direct_read_queue.h"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <cerrno>
#    include <condition_variable>
#    include <fcntl.h>
#    include <mutex>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

#include <algorithm>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace infernix {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kMaxReadBytes = std::size_t{1} << 30;

// Severity of a request's outcome: a request reports its worst sub-read.
int severity(DirectReadQueue::Status status) {
    switch (status) {
    case DirectReadQueue::Status::Ok: return 0;
    case DirectReadQueue::Status::Cancelled: return 1;
    case DirectReadQueue::Status::TimedOut: return 2;
    case DirectReadQueue::Status::Failed: return 3;
    }
    return 3;
}

#ifdef _WIN32
std::error_code win32_error(DWORD code) { return {static_cast<int>(code), std::system_category()}; }
bool is_transient(const std::error_code& ec) {
    if (ec.category() != std::system_category()) { return false; }
    const auto v = static_cast<DWORD>(ec.value());
    return v == ERROR_NO_SYSTEM_RESOURCES || v == ERROR_WORKING_SET_QUOTA || v == ERROR_NOT_ENOUGH_QUOTA ||
           v == ERROR_NOT_ENOUGH_MEMORY;
}
bool is_cancellation(const std::error_code& ec) {
    return ec.category() == std::system_category() && static_cast<DWORD>(ec.value()) == ERROR_OPERATION_ABORTED;
}
std::error_code cancellation_error() { return win32_error(ERROR_OPERATION_ABORTED); }
std::error_code fault_error(bool transient) { return win32_error(transient ? ERROR_NO_SYSTEM_RESOURCES : ERROR_CRC); }
#else
std::error_code errno_error(int code) { return {code, std::generic_category()}; }
bool is_transient(const std::error_code& ec) {
    return ec.category() == std::generic_category() &&
           (ec.value() == EAGAIN || ec.value() == ENOMEM || ec.value() == ENOBUFS);
}
bool is_cancellation(const std::error_code& ec) {
    return ec.category() == std::generic_category() && ec.value() == ECANCELED;
}
std::error_code cancellation_error() { return errno_error(ECANCELED); }
std::error_code fault_error(bool transient) { return errno_error(transient ? ENOBUFS : EIO); }
#endif

} // namespace

struct DirectReadQueue::Impl {
    struct File {
#ifdef _WIN32
        HANDLE handle = INVALID_HANDLE_VALUE;
#else
        int fd = -1;
#endif
        std::uint64_t bytes = 0;
    };

    struct Request {
        std::uint64_t tag       = 0;
        std::uint32_t remaining = 0;
        Status status           = Status::Ok;
        std::error_code error;
        std::uint32_t attempts = 0;
        Clock::time_point submitted{};
        bool cancelled = false;
    };

    // One read of the device: a whole demand read or one prefetch sub-read. Never moves while
    // issued (a deque element), since Windows writes its OVERLAPPED.
    struct Io {
#ifdef _WIN32
        OVERLAPPED overlapped{};
#endif
        std::size_t self = 0, request = 0;
        std::uint32_t file   = 0;
        std::uint64_t offset = 0;
        std::size_t bytes    = 0;
        std::byte* destination = nullptr;
        Priority priority      = Priority::Demand;
        std::uint32_t issues = 0, hard_failures = 0, timeouts = 0, transient_retries = 0;
        Clock::time_point issued{}, first_transient{}, not_before{};
        bool timed_out = false, stalled = false, transient_seen = false;
    };

    // A finished device read waiting to be applied (held by the delay fault, or synthetic).
    struct Finished {
        std::size_t io = 0;
        std::error_code error;
        std::size_t bytes = 0;
        Clock::time_point release{};
    };

    Options options;
    Faults faults;
    Stats stats;
    std::vector<File> files;
    std::deque<Request> requests;
    std::vector<std::size_t> free_requests;
    std::deque<Io> ios;
    std::vector<std::size_t> free_ios;
    std::deque<std::size_t> waiting[2];
    std::vector<std::size_t> backoff;
    std::vector<std::size_t> flight;
    std::vector<Finished> held;
    std::vector<Completion> ready;
    std::uint32_t in_class[2] = {};
    std::size_t live = 0;
    bool cancelling  = false;

#ifdef _WIN32
    HANDLE port = nullptr;
#else
    struct Job {
        std::size_t io = 0;
        int fd         = -1;
        std::uint64_t offset = 0;
        std::size_t bytes    = 0;
        std::byte* destination = nullptr;
    };
    std::mutex mutex;
    std::condition_variable work_ready, done_ready;
    std::deque<Job> jobs;
    std::deque<Finished> done;
    bool stopping = false;
    std::vector<std::thread> workers;
#endif

    explicit Impl(const Options& o) : options(o) {
        if (options.demand_depth == 0 || options.prefetch_depth == 0 || options.total_depth == 0 ||
            options.prefetch_split < kAlignment || options.prefetch_split % kAlignment != 0) {
            throw std::invalid_argument("DirectReadQueue: depths must be positive and the split a positive multiple of 4 KiB");
        }
#ifdef _WIN32
        port = ::CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 1);
        if (port == nullptr) {
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "CreateIoCompletionPort");
        }
#else
        const std::uint32_t n = std::max<std::uint32_t>(1, options.posix_workers);
        for (std::uint32_t i = 0; i < n; ++i) { workers.emplace_back([this] { work(); }); }
#endif
    }

    ~Impl() {
#ifdef _WIN32
        for (auto& f : files) {
            if (f.handle != INVALID_HANDLE_VALUE) { ::CloseHandle(f.handle); }
        }
        if (port != nullptr) { ::CloseHandle(port); }
#else
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        work_ready.notify_all();
        for (auto& w : workers) { w.join(); }
        for (auto& f : files) {
            if (f.fd >= 0) { ::close(f.fd); }
        }
#endif
    }

    // ---- platform
    std::uint32_t open_file(const std::filesystem::path& path) {
        File f;
#ifdef _WIN32
        // No write sharing: an in-place rewrite of the artifact fails while it is read. Delete
        // sharing keeps a rename-replace possible (this handle keeps the old file).
        f.handle = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                                 FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if (f.handle == INVALID_HANDLE_VALUE) {
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "DirectReadQueue: CreateFileW " + path.string());
        }
        LARGE_INTEGER size{};
        if (!::GetFileSizeEx(f.handle, &size) || ::CreateIoCompletionPort(f.handle, port, 0, 0) == nullptr) {
            const DWORD error = ::GetLastError();
            ::CloseHandle(f.handle);
            throw std::system_error(static_cast<int>(error), std::system_category(), "DirectReadQueue: open " + path.string());
        }
        f.bytes = static_cast<std::uint64_t>(size.QuadPart);
#else
        int flags = O_RDONLY | O_CLOEXEC;
#    ifdef O_DIRECT
        flags |= O_DIRECT;
#    endif
        f.fd = ::open(path.c_str(), flags);
        if (f.fd < 0) { throw std::system_error(errno, std::generic_category(), "DirectReadQueue: open " + path.string()); }
        struct stat st {};
        if (::fstat(f.fd, &st) != 0) {
            const int error = errno;
            ::close(f.fd);
            throw std::system_error(error, std::generic_category(), "DirectReadQueue: fstat " + path.string());
        }
        f.bytes = static_cast<std::uint64_t>(st.st_size);
#endif
        files.push_back(f);
        return static_cast<std::uint32_t>(files.size() - 1);
    }

    // Starts the device read of `io`; an immediate failure becomes a finished read.
    void platform_read(Io& io) {
#ifdef _WIN32
        io.overlapped            = OVERLAPPED{};
        io.overlapped.Offset     = static_cast<DWORD>(io.offset & 0xffffffffULL);
        io.overlapped.OffsetHigh = static_cast<DWORD>(io.offset >> 32U);
        if (!::ReadFile(files[io.file].handle, io.destination, static_cast<DWORD>(io.bytes), nullptr, &io.overlapped)) {
            const DWORD error = ::GetLastError();
            if (error != ERROR_IO_PENDING) { held.push_back({io.self, win32_error(error), 0, Clock::now()}); }
        }
#else
        {
            std::lock_guard<std::mutex> lock(mutex);
            jobs.push_back({io.self, files[io.file].fd, io.offset, io.bytes, io.destination});
        }
        work_ready.notify_one();
#endif
    }

    // Asks the device to abandon `io`; false when that is not possible (it completes normally).
    bool platform_cancel(Io& io) {
        if (io.stalled) {
            io.stalled = false;
            held.push_back({io.self, cancellation_error(), 0, Clock::now()});
            return true;
        }
#ifdef _WIN32
        return ::CancelIoEx(files[io.file].handle, &io.overlapped) != FALSE;
#else
        return false;
#endif
    }

    // Waits up to `wait` for device completions and appends them to `got`.
    void platform_wait(std::chrono::microseconds wait, std::vector<Finished>& got) {
#ifdef _WIN32
        OVERLAPPED_ENTRY entries[32];
        ULONG n = 0;
        const auto ms = static_cast<DWORD>(std::min<std::int64_t>((wait.count() + 999) / 1000, 60000));
        if (!::GetQueuedCompletionStatusEx(port, entries, 32, &n, ms, FALSE)) {
            const DWORD error = ::GetLastError();
            if (error == WAIT_TIMEOUT) { return; }
            throw std::system_error(static_cast<int>(error), std::system_category(), "GetQueuedCompletionStatusEx");
        }
        for (ULONG i = 0; i < n; ++i) {
            Io* io      = CONTAINING_RECORD(entries[i].lpOverlapped, Io, overlapped);
            DWORD bytes = 0;
            std::error_code error;
            if (!::GetOverlappedResult(files[io->file].handle, &io->overlapped, &bytes, FALSE)) {
                error = win32_error(::GetLastError());
            }
            got.push_back({io->self, error, bytes, Clock::now()});
        }
#else
        std::unique_lock<std::mutex> lock(mutex);
        done_ready.wait_for(lock, wait, [&] { return !done.empty(); });
        while (!done.empty()) {
            got.push_back(done.front());
            done.pop_front();
        }
#endif
    }

#ifndef _WIN32
    void work() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                work_ready.wait(lock, [&] { return stopping || !jobs.empty(); });
                if (stopping && jobs.empty()) { return; }
                job = jobs.front();
                jobs.pop_front();
            }
            std::size_t total = 0;
            std::error_code error;
            while (total < job.bytes) {
                const ssize_t got = ::pread(job.fd, job.destination + total, job.bytes - total,
                                            static_cast<off_t>(job.offset + total));
                if (got < 0) {
                    if (errno == EINTR) { continue; }
                    error = errno_error(errno);
                    break;
                }
                if (got == 0) { break; }
                total += static_cast<std::size_t>(got);
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                done.push_back({job.io, error, total, Clock::now()});
            }
            done_ready.notify_one();
        }
    }
#endif

    // ---- scheduling
    void submit(const Read& read) {
        if (read.file >= files.size()) { throw std::invalid_argument("DirectReadQueue: unknown file"); }
        if (read.bytes == 0 || read.bytes > kMaxReadBytes || read.bytes % kAlignment != 0 || read.offset % kAlignment != 0 ||
            reinterpret_cast<std::uintptr_t>(read.destination) % kAlignment != 0) {
            throw std::invalid_argument("DirectReadQueue: reads need 4 KiB-aligned offsets, sizes and destinations");
        }
        std::size_t r;
        if (!free_requests.empty()) {
            r = free_requests.back();
            free_requests.pop_back();
        } else {
            r = requests.size();
            requests.emplace_back();
        }
        const std::size_t piece =
            read.priority == Priority::Prefetch ? std::min(read.bytes, options.prefetch_split) : read.bytes;
        const auto pieces = static_cast<std::uint32_t>((read.bytes + piece - 1) / piece);
        requests[r]       = Request{.tag = read.tag, .remaining = pieces, .submitted = Clock::now()};
        for (std::uint32_t k = 0; k < pieces; ++k) {
            std::size_t i;
            if (!free_ios.empty()) {
                i = free_ios.back();
                free_ios.pop_back();
            } else {
                i = ios.size();
                ios.emplace_back();
            }
            Io& io         = ios[i];
            io             = Io{};
            io.self        = i;
            io.request     = r;
            io.file        = read.file;
            io.offset      = read.offset + static_cast<std::uint64_t>(k) * piece;
            io.bytes       = std::min(piece, read.bytes - static_cast<std::size_t>(k) * piece);
            io.destination = static_cast<std::byte*>(read.destination) + static_cast<std::size_t>(k) * piece;
            io.priority    = read.priority;
            waiting[static_cast<int>(read.priority)].push_back(i);
        }
        ++live;
    }

    std::size_t in_flight() const { return flight.size(); }

    void issue_ready(Clock::time_point now) {
        for (std::size_t k = 0; k < backoff.size();) {
            Io& io = ios[backoff[k]];
            if (io.not_before <= now) {
                waiting[static_cast<int>(io.priority)].push_front(io.self);
                backoff.erase(backoff.begin() + static_cast<std::ptrdiff_t>(k));
            } else {
                ++k;
            }
        }
        const std::uint32_t caps[2] = {options.demand_depth, options.prefetch_depth};
        while (in_flight() < options.total_depth) {
            int c = -1;
            for (int k = 0; k < 2 && c < 0; ++k) {
                if (!waiting[k].empty() && in_class[k] < caps[k]) { c = k; }
            }
            if (c < 0) { break; }
            const std::size_t i = waiting[c].front();
            waiting[c].pop_front();
            issue(ios[i], now);
        }
    }

    void issue(Io& io, Clock::time_point now) {
        ++io.issues;
        const std::uint64_t n = ++stats.issued;
        io.issued    = now;
        io.timed_out = false;
        flight.push_back(io.self);
        ++in_class[static_cast<int>(io.priority)];
        if (faults.fail_read != 0 && n >= faults.fail_read && n < faults.fail_read + faults.fail_count) {
            held.push_back({io.self, fault_error(faults.transient), 0, now});
            return;
        }
        if (faults.stall_read != 0 && n >= faults.stall_read && n < faults.stall_read + faults.stall_count) {
            io.stalled = true;
            return;
        }
        platform_read(io);
    }

    void check_timeouts(Clock::time_point now) {
        for (const std::size_t i : flight) {
            Io& io = ios[i];
            if (!io.timed_out && now - io.issued >= options.timeout && platform_cancel(io)) {
                io.timed_out = true;
                ++io.timeouts;
                ++stats.timeouts;
            }
        }
    }

    // A device read has finished: apply the delay fault, then its outcome.
    void arrive(const Finished& f) {
        const Io& io = ios[f.io];
        const auto release = io.issued + faults.delay;
        if (!cancelling && faults.delay.count() > 0 && release > f.release) {
            held.push_back({f.io, f.error, f.bytes, release});
        } else {
            finish(f.io, f.error, f.bytes);
        }
    }

    void finish(std::size_t i, const std::error_code& error, std::size_t bytes) {
        Io& io = ios[i];
        flight.erase(std::find(flight.begin(), flight.end(), i));
        --in_class[static_cast<int>(io.priority)];
        const Request& request = requests[io.request];
        const auto now         = Clock::now();
        if (!error && bytes == io.bytes) { return resolve(io, Status::Ok, {}); }
        if (request.cancelled) { return resolve(io, Status::Cancelled, cancellation_error()); }
        if (error && io.timed_out && is_cancellation(error)) {
            if (io.timeouts < 2) { return retry(io); }
            return resolve(io, Status::TimedOut, std::make_error_code(std::errc::timed_out));
        }
        if (error && is_transient(error)) {
            if (!io.transient_seen) {
                io.transient_seen  = true;
                io.first_transient = now;
            }
            if (now - io.first_transient < options.transient_bound) {
                io.not_before = now + std::chrono::milliseconds(1LL << std::min<std::uint32_t>(io.transient_retries, 10));
                ++io.transient_retries;
                ++stats.transient_retries;
                backoff.push_back(i);
                return;
            }
            return resolve(io, Status::Failed, error);
        }
        if (error) {
            if (io.hard_failures++ == 0) { return retry(io); }
            return resolve(io, Status::Failed, error);
        }
        // A short read: the range passes the end of the file.
        resolve(io, Status::Failed, std::make_error_code(std::errc::io_error));
    }

    void retry(Io& io) {
        ++stats.retries;
        waiting[static_cast<int>(io.priority)].push_front(io.self);
    }

    void resolve(Io& io, Status status, const std::error_code& error) {
        Request& request = requests[io.request];
        request.attempts = std::max(request.attempts, io.issues);
        if (severity(status) > severity(request.status)) {
            request.status = status;
            request.error  = error;
        }
        free_ios.push_back(io.self);
        if (--request.remaining > 0) { return; }
        Completion c;
        c.tag        = request.tag;
        c.status     = request.status;
        c.error      = request.error;
        c.attempts   = request.attempts;
        c.latency_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - request.submitted).count());
        ready.push_back(c);
        free_requests.push_back(io.request);
        --live;
    }

    void apply_held(Clock::time_point now) {
        for (std::size_t k = 0; k < held.size();) {
            if (cancelling || held[k].release <= now) {
                const Finished f = held[k];
                held.erase(held.begin() + static_cast<std::ptrdiff_t>(k));
                finish(f.io, f.error, f.bytes);
            } else {
                ++k;
            }
        }
    }

    // The next time something is due without a device completion.
    Clock::time_point next_due(Clock::time_point limit) const {
        Clock::time_point due = limit;
        for (const auto& f : held) { due = std::min(due, f.release); }
        for (const std::size_t i : backoff) { due = std::min(due, ios[i].not_before); }
        for (const std::size_t i : flight) {
            if (!ios[i].timed_out) { due = std::min(due, ios[i].issued + options.timeout); }
        }
        return due;
    }

    std::size_t poll(std::vector<Completion>& out, std::chrono::microseconds wait) {
        const auto deadline = Clock::now() + wait;
        std::vector<Finished> got;
        for (bool first = true;; first = false) {
            auto now = Clock::now();
            apply_held(now);
            issue_ready(now);
            check_timeouts(now);
            apply_held(now); // a cancelled stall finishes at once
            issue_ready(now);
            const bool busy = !flight.empty() || !held.empty() || !backoff.empty();
            if (!ready.empty() || !busy || (!first && now >= deadline)) { break; }
            const auto until = next_due(deadline);
            const auto span  = std::chrono::duration_cast<std::chrono::microseconds>(until > now ? until - now : Clock::duration{});
            got.clear();
            platform_wait(flight.empty() ? std::chrono::microseconds{0} : span, got);
            for (const auto& f : got) { arrive(f); }
            if (flight.empty() && got.empty() && until > Clock::now()) {
                // Only held completions or backoffs are due: sleep until the first.
                std::this_thread::sleep_until(until);
            }
        }
        const std::size_t n = ready.size();
        out.insert(out.end(), ready.begin(), ready.end());
        ready.clear();
        return n;
    }

    void cancel_all() {
        cancelling = true;
        for (std::size_t r = 0; r < requests.size(); ++r) { requests[r].cancelled = true; }
        for (auto& queue : waiting) {
            while (!queue.empty()) {
                const std::size_t i = queue.front();
                queue.pop_front();
                resolve(ios[i], Status::Cancelled, cancellation_error());
            }
        }
        for (const std::size_t i : backoff) { resolve(ios[i], Status::Cancelled, cancellation_error()); }
        backoff.clear();
        for (const std::size_t i : std::vector<std::size_t>(flight)) { platform_cancel(ios[i]); }
        std::vector<Finished> got;
        while (!flight.empty()) {
            apply_held(Clock::now());
            if (flight.empty()) { break; }
            got.clear();
            platform_wait(std::chrono::milliseconds(100), got);
            for (const auto& f : got) { finish(f.io, f.error, f.bytes); }
        }
        held.clear();
        for (auto& request : requests) { request.cancelled = false; }
        cancelling = false;
    }
};

DirectReadQueue::DirectReadQueue(Options options) : impl_(std::make_unique<Impl>(options)) {}

DirectReadQueue::~DirectReadQueue() {
    try {
        impl_->cancel_all();
    } catch (...) {}
}

std::uint32_t DirectReadQueue::open(const std::filesystem::path& path) { return impl_->open_file(path); }

std::uint64_t DirectReadQueue::file_bytes(std::uint32_t file) const {
    if (file >= impl_->files.size()) { throw std::invalid_argument("DirectReadQueue: unknown file"); }
    return impl_->files[file].bytes;
}

void DirectReadQueue::submit(const Read& read) { impl_->submit(read); }

std::size_t DirectReadQueue::poll(std::vector<Completion>& out, std::chrono::microseconds wait) {
    return impl_->poll(out, wait);
}

void DirectReadQueue::cancel_all() { impl_->cancel_all(); }

std::size_t DirectReadQueue::pending() const noexcept { return impl_->live + impl_->ready.size(); }

const DirectReadQueue::Stats& DirectReadQueue::stats() const noexcept { return impl_->stats; }

void DirectReadQueue::set_faults(const Faults& faults) { impl_->faults = faults; }

} // namespace infernix
