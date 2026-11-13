#include "core/host_memory.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/mman.h>
#include <sys/resource.h>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#endif

#if defined(_M_X64) || defined(__x86_64__)
#include <emmintrin.h>
#endif

namespace infernix {

void copy_streaming(void* destination, const void* source, std::size_t bytes) noexcept {
#if defined(_M_X64) || defined(__x86_64__)
    const auto to = reinterpret_cast<std::uintptr_t>(destination), from = reinterpret_cast<std::uintptr_t>(source);
    if (((to | from | bytes) & 15U) == 0) {
        auto* out          = static_cast<__m128i*>(destination);
        const auto* in     = static_cast<const __m128i*>(source);
        const std::size_t n = bytes / 16U;
        std::size_t i       = 0;
        for (; i + 4U <= n; i += 4U) { // a 64-byte line per step: the write-combining buffer fills whole
            const __m128i a = _mm_load_si128(in + i), b = _mm_load_si128(in + i + 1U);
            const __m128i c = _mm_load_si128(in + i + 2U), d = _mm_load_si128(in + i + 3U);
            _mm_stream_si128(out + i, a);
            _mm_stream_si128(out + i + 1U, b);
            _mm_stream_si128(out + i + 2U, c);
            _mm_stream_si128(out + i + 3U, d);
        }
        for (; i < n; ++i) { _mm_stream_si128(out + i, _mm_load_si128(in + i)); }
        _mm_sfence(); // the streamed stores are visible before whatever the caller publishes next
        return;
    }
#endif
    std::memcpy(destination, source, bytes);
}

namespace {

std::atomic<std::size_t>& refused_registrations() {
    static std::atomic<std::size_t> above{0};
    return above;
}

std::size_t refused_registration_above() noexcept { return refused_registrations().load(std::memory_order_relaxed); }

#ifndef _WIN32
// A "Key:   N kB" field of a /proc file, in bytes.
std::optional<std::uint64_t> proc_kib_field(const char* path, const std::string& key) {
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind(key, 0) == 0) {
            std::istringstream fields(line.substr(key.size()));
            std::uint64_t kib = 0;
            if (fields >> kib) { return kib * 1024; }
        }
    }
    return std::nullopt;
}

// A cgroup v2 memory file holding one byte count ("max" means unlimited).
std::optional<std::uint64_t> cgroup_bytes(const char* path) {
    std::ifstream file(path);
    std::string value;
    if (!(file >> value) || value == "max") { return std::nullopt; }
    try {
        return std::stoull(value);
    } catch (const std::exception&) { return std::nullopt; }
}
#endif

} // namespace

std::uint64_t available_host_memory_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!::GlobalMemoryStatusEx(&status)) {
        throw std::runtime_error("GlobalMemoryStatusEx failed");
    }
    return status.ullAvailPhys;
#else
    if (const auto available = proc_kib_field("/proc/meminfo", "MemAvailable:")) { return *available; }
    throw std::runtime_error("/proc/meminfo has no MemAvailable");
#endif
}

HostMemorySnapshot host_memory_snapshot() {
    HostMemorySnapshot out;
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!::GlobalMemoryStatusEx(&status)) { throw std::runtime_error("GlobalMemoryStatusEx failed"); }
    out.total_physical     = status.ullTotalPhys;
    out.available_physical = status.ullAvailPhys;
    out.available_commit   = status.ullAvailPageFile;
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (::GetProcessMemoryInfo(::GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                               sizeof(counters))) {
        out.process_private  = counters.PrivateUsage;
        out.process_resident = counters.WorkingSetSize;
    }
#else
    out.total_physical     = proc_kib_field("/proc/meminfo", "MemTotal:").value_or(0);
    out.available_physical = available_host_memory_bytes();
    if (const auto limit = cgroup_bytes("/sys/fs/cgroup/memory.max")) {
        const std::uint64_t used = cgroup_bytes("/sys/fs/cgroup/memory.current").value_or(0);
        out.available_physical   = std::min(out.available_physical, *limit > used ? *limit - used : 0);
    }
    out.available_commit = out.available_physical + proc_kib_field("/proc/meminfo", "SwapFree:").value_or(0);
    out.process_private  = proc_kib_field("/proc/self/status", "VmRSS:").value_or(0);
    out.process_resident = out.process_private;
#endif
    return out;
}

std::string reserve_process_working_set(std::uint64_t bytes) {
#ifdef _WIN32
    SIZE_T minimum = 0, maximum = 0;
    DWORD flags    = 0;
    if (!::GetProcessWorkingSetSizeEx(::GetCurrentProcess(), &minimum, &maximum, &flags)) {
        return "Windows error " + std::to_string(::GetLastError());
    }
    const auto wanted = static_cast<SIZE_T>(bytes);
    // Only the minimum is hard; the maximum stays soft so the process may grow past it. Windows
    // rejects a hard minimum equal to the maximum (ERROR_INVALID_PARAMETER), so the maximum is
    // kept above it.
    constexpr SIZE_T kMaximumAbove = SIZE_T{64} << 20;
    if (!::SetProcessWorkingSetSizeEx(::GetCurrentProcess(), wanted, std::max(maximum, wanted + kMaximumAbove),
                                      QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        return "Windows error " + std::to_string(::GetLastError());
    }
    return {};
#else
    // Linux has no per-process resident minimum short of locking pages.
    (void)bytes;
    return {};
#endif
}

RegisteredHostBuffer::RegisteredHostBuffer(std::size_t bytes) : size_(bytes) {
    if (bytes == 0) { throw std::invalid_argument("RegisteredHostBuffer size must be nonzero"); }
#ifdef _WIN32
    void* p = ::VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (p == nullptr) {
        throw std::runtime_error("VirtualAlloc of " + std::to_string(bytes) + " bytes failed: Windows error " +
                                 std::to_string(::GetLastError()));
    }
#else
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { throw std::runtime_error("mmap of " + std::to_string(bytes) + " bytes failed"); }
#endif
    data_ = static_cast<std::byte*>(p);
}

RegisteredHostBuffer::~RegisteredHostBuffer() {
    unregister();
#ifdef _WIN32
    (void)::VirtualFree(data_, 0, MEM_RELEASE);
#else
    (void)::munmap(data_, size_);
#endif
}

void RegisteredHostBuffer::prefault(std::size_t begin, std::size_t end) const noexcept {
    constexpr std::size_t kPage = 4096;
    for (std::size_t at = begin / kPage * kPage; at < std::min(end, size_); at += kPage) {
        *reinterpret_cast<volatile std::byte*>(data_ + std::max(at, begin)) = std::byte{0};
    }
}

bool RegisteredHostBuffer::register_slice(std::size_t begin, std::size_t bytes, std::string& failure) {
    std::byte* at = data_ + begin;
    if (const std::size_t limit = refused_registration_above(); limit != 0 && bytes > limit) {
        failure = "cudaHostRegister of " + std::to_string(bytes) + " bytes refused (test)";
        return false;
    }
    if (const auto status = cudaHostRegister(at, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
        status != cudaSuccess) {
        (void)cudaGetLastError();
        failure = "cudaHostRegister of " + std::to_string(bytes) + " bytes failed: " + cudaGetErrorString(status);
        return false;
    }
    void* device = nullptr;
    if (const auto status = cudaHostGetDevicePointer(&device, at, 0); status != cudaSuccess) {
        (void)cudaHostUnregister(at);
        throw std::runtime_error(std::string("cudaHostGetDevicePointer failed: ") + cudaGetErrorString(status));
    }
    slices_.push_back(Slice{.begin = begin, .bytes = bytes, .device = static_cast<std::byte*>(device)});
    return true;
}

void RegisteredHostBuffer::unregister() noexcept {
    for (const Slice& slice : slices_) { (void)cudaHostUnregister(data_ + slice.begin); }
    slices_.clear();
}

void RegisteredHostBuffer::pin(std::span<const std::size_t> boundaries, std::size_t slice_bytes) {
    if (!slices_.empty()) { throw std::logic_error("RegisteredHostBuffer is already pinned"); }
    std::string whole;
    if (register_slice(0, size_, whole)) { return; }
    // Slices that end at object boundaries, each as long as slice_bytes allows (an object larger
    // than that is one slice of its own); a slice never splits a page, which two registrations
    // cannot share.
    constexpr std::size_t kPage = 4096;
    std::vector<std::size_t> cuts;
    for (const std::size_t at : boundaries) {
        if (at % kPage == 0 && at > 0 && at < size_ && (cuts.empty() || at > cuts.back())) { cuts.push_back(at); }
    }
    if (cuts.empty()) { throw std::runtime_error(whole); }
    std::size_t begin = 0;
    std::size_t next  = 0;
    while (begin < size_) {
        std::size_t end = size_;
        while (next < cuts.size() && cuts[next] <= begin) { ++next; }
        if (end - begin > slice_bytes) {
            std::size_t cut = next;
            while (cut < cuts.size() && cuts[cut] - begin <= slice_bytes) { ++cut; }
            if (cut > next) {
                end = cuts[cut - 1];
            } else if (next < cuts.size()) {
                end = cuts[next];
            }
        }
        std::string failure;
        if (!register_slice(begin, end - begin, failure)) {
            unregister();
            throw std::runtime_error(whole + "; registered in slices of at most " + std::to_string(slice_bytes >> 20) +
                                     " MiB instead, a slice failed too: " + failure);
        }
        begin = end;
    }
}

std::byte* RegisteredHostBuffer::device_at(std::size_t offset) const noexcept {
    if (slices_.empty()) { return nullptr; }
    const auto after = std::upper_bound(slices_.begin(), slices_.end(), offset,
                                        [](std::size_t at, const Slice& slice) { return at < slice.begin; });
    const Slice& slice = *(after - 1);
    return slice.device + (offset - slice.begin);
}

#ifdef _WIN32
namespace {

// VirtualLock holds pages in the working set, so the hard minimum must cover them (with the process's own pages above
// it); the maximum stays soft.
constexpr SIZE_T kWorkingSetSlack = SIZE_T{64} << 20;

// Raises the hard minimum working set by `bytes` and locks [at, at + bytes); empty, or the OS's reason. `raised` grows
// by what the minimum grew.
std::string lock_resident(std::byte* at, std::size_t bytes, std::size_t& raised) {
    SIZE_T minimum = 0, maximum = 0;
    DWORD flags    = 0;
    if (!::GetProcessWorkingSetSizeEx(::GetCurrentProcess(), &minimum, &maximum, &flags)) {
        return "Windows error " + std::to_string(::GetLastError()) + " reading the working set";
    }
    const SIZE_T grow   = bytes + (raised == 0 ? kWorkingSetSlack : 0);
    const SIZE_T wanted = minimum + grow;
    if (!::SetProcessWorkingSetSizeEx(::GetCurrentProcess(), wanted, std::max(maximum, wanted + kWorkingSetSlack),
                                      QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        return "Windows error " + std::to_string(::GetLastError()) + " raising the working set";
    }
    raised += grow;
    constexpr std::size_t kPiece = std::size_t{1} << 30;
    for (std::size_t done = 0; done < bytes; done += kPiece) {
        if (!::VirtualLock(at + done, std::min(kPiece, bytes - done))) {
            return "Windows error " + std::to_string(::GetLastError()) + " locking pages";
        }
    }
    return {};
}

} // namespace
#else
namespace {

// A service started without LimitMEMLOCK, or a login shell's default, may lock as little as 8 MiB while its hard limit
// allows more: the soft limit is raised to the hard one once (no privilege needed) before the first lock (candidates
// WA-02, Strata 5a13c34f).
void raise_memlock_soft_limit() noexcept {
    static const bool raised = [] {
        rlimit limit{};
        if (::getrlimit(RLIMIT_MEMLOCK, &limit) == 0 && limit.rlim_cur != RLIM_INFINITY &&
            (limit.rlim_max == RLIM_INFINITY || limit.rlim_cur < limit.rlim_max)) {
            limit.rlim_cur = limit.rlim_max;
            (void)::setrlimit(RLIMIT_MEMLOCK, &limit);
        }
        return true;
    }();
    (void)raised;
}

// Locks [at, at + bytes); empty, or the reason with the limit and what raises it.
std::string lock_resident(std::byte* at, std::size_t bytes) {
    raise_memlock_soft_limit();
    if (::mlock(at, bytes) == 0) { return {}; }
    const int error = errno;
    std::string limit_text = "unknown";
    if (rlimit limit{}; ::getrlimit(RLIMIT_MEMLOCK, &limit) == 0) {
        const auto mib = [](rlim_t value) {
            return value == RLIM_INFINITY ? std::string("unlimited") : std::to_string(value >> 20) + " MiB";
        };
        limit_text = mib(limit.rlim_cur) + " (hard " + mib(limit.rlim_max) + ")";
    }
    return "mlock of " + std::to_string(bytes >> 20) + " MiB failed: " + std::strerror(error) + ", RLIMIT_MEMLOCK " +
           limit_text + "; raise the memlock limit (LimitMEMLOCK=infinity in a systemd unit, `ulimit -l unlimited` in "
                        "the launching shell, or a memlock line in /etc/security/limits.conf)";
}

} // namespace
#endif

ResidentHostBuffer::ResidentHostBuffer(std::size_t bytes) : size_(bytes) {
    if (bytes == 0) { throw std::invalid_argument("ResidentHostBuffer size must be nonzero"); }
#ifdef _WIN32
    void* p = ::VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (p == nullptr) {
        throw std::runtime_error("VirtualAlloc of " + std::to_string(bytes) + " bytes failed: Windows error " +
                                 std::to_string(::GetLastError()));
    }
    data_         = static_cast<std::byte*>(p);
    lock_failure_ = lock_resident(data_, bytes, raised_);
#else
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { throw std::runtime_error("mmap of " + std::to_string(bytes) + " bytes failed"); }
    data_         = static_cast<std::byte*>(p);
    lock_failure_ = lock_resident(data_, bytes);
#endif
}

ResidentHostBuffer::~ResidentHostBuffer() {
#ifdef _WIN32
    (void)::VirtualFree(data_, 0, MEM_RELEASE); // releases the locks
    if (raised_ != 0) {
        SIZE_T minimum = 0, maximum = 0;
        DWORD flags    = 0;
        if (::GetProcessWorkingSetSizeEx(::GetCurrentProcess(), &minimum, &maximum, &flags) && minimum > raised_) {
            (void)::SetProcessWorkingSetSizeEx(::GetCurrentProcess(), minimum - raised_, maximum, flags);
        }
    }
#else
    (void)::munmap(data_, size_);
#endif
}

void ResidentHostBuffer::decommit(std::size_t offset, std::size_t bytes) noexcept {
    if (bytes == 0) { return; }
    std::byte* at = data_ + offset;
#ifdef _WIN32
    (void)::VirtualUnlock(at, bytes); // fails on pages the OS never locked, which decommit alike
    (void)::VirtualFree(at, bytes, MEM_DECOMMIT);
    // The minimum shrinks by the pages it covered (never below the slack it was raised with).
    const std::size_t lowered = std::min(bytes, raised_ > kWorkingSetSlack ? raised_ - kWorkingSetSlack : 0);
    SIZE_T minimum = 0, maximum = 0;
    DWORD flags    = 0;
    if (lowered != 0 && ::GetProcessWorkingSetSizeEx(::GetCurrentProcess(), &minimum, &maximum, &flags) &&
        minimum > lowered && ::SetProcessWorkingSetSizeEx(::GetCurrentProcess(), minimum - lowered, maximum, flags)) {
        raised_ -= lowered;
    }
#else
    (void)::munlock(at, bytes);
    (void)::madvise(at, bytes, MADV_DONTNEED);
#endif
}

bool ResidentHostBuffer::recommit(std::size_t offset, std::size_t bytes) {
    if (bytes == 0) { return true; }
    std::byte* at = data_ + offset;
#ifdef _WIN32
    if (::VirtualAlloc(at, bytes, MEM_COMMIT, PAGE_READWRITE) == nullptr) { return false; }
    // Locked again only when the constructor raised the working set (otherwise the block stays pageable).
    if (raised_ != 0) {
        if (std::string failure = lock_resident(at, bytes, raised_); !failure.empty()) { lock_failure_ = std::move(failure); }
    }
#else
    if (std::string failure = lock_resident(at, bytes); !failure.empty()) { lock_failure_ = std::move(failure); }
#endif
    return true;
}

AddressReservation::AddressReservation(std::size_t bytes) : size_(bytes) {
    if (bytes == 0) { return; }
#ifdef _WIN32
    void* p = ::VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_NOACCESS);
    if (p == nullptr) {
        throw std::runtime_error("reserving " + std::to_string(bytes) + " bytes of address space failed (Windows error " +
                                 std::to_string(::GetLastError()) + ")");
    }
#else
    void* p = ::mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        throw std::runtime_error("reserving " + std::to_string(bytes) + " bytes of address space failed");
    }
#endif
    data_ = static_cast<std::byte*>(p);
}

AddressReservation::~AddressReservation() {
    if (data_ == nullptr) { return; }
#ifdef _WIN32
    (void)::VirtualFree(data_, 0, MEM_RELEASE);
#else
    (void)::munmap(data_, size_);
#endif
}

namespace testing {
void refuse_host_registrations_above(std::size_t bytes) {
    refused_registrations().store(bytes, std::memory_order_relaxed);
}
} // namespace testing

} // namespace infernix
