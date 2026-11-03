#include "core/host_memory.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
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

#include <fstream>
#include <sstream>
#include <string>
#endif

namespace infernix {
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

ResidentHostBuffer::ResidentHostBuffer(std::size_t bytes) : size_(bytes) {
    if (bytes == 0) { throw std::invalid_argument("ResidentHostBuffer size must be nonzero"); }
#ifdef _WIN32
    void* p = ::VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (p == nullptr) {
        throw std::runtime_error("VirtualAlloc of " + std::to_string(bytes) + " bytes failed: Windows error " +
                                 std::to_string(::GetLastError()));
    }
    data_ = static_cast<std::byte*>(p);
    // VirtualLock holds pages in the working set, so the hard minimum must cover them (with the
    // process's own pages above it); the maximum stays soft.
    constexpr SIZE_T kSlack = SIZE_T{64} << 20;
    SIZE_T minimum = 0, maximum = 0;
    DWORD flags    = 0;
    if (!::GetProcessWorkingSetSizeEx(::GetCurrentProcess(), &minimum, &maximum, &flags)) {
        lock_failure_ = "Windows error " + std::to_string(::GetLastError()) + " reading the working set";
        return;
    }
    const SIZE_T wanted = minimum + bytes + kSlack;
    if (!::SetProcessWorkingSetSizeEx(::GetCurrentProcess(), wanted, std::max(maximum, wanted + kSlack),
                                      QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        lock_failure_ = "Windows error " + std::to_string(::GetLastError()) + " raising the working set";
        return;
    }
    raised_ = bytes + kSlack;
    constexpr std::size_t kPiece = std::size_t{1} << 30;
    for (std::size_t at = 0; at < bytes; at += kPiece) {
        if (!::VirtualLock(data_ + at, std::min(kPiece, bytes - at))) {
            lock_failure_ = "Windows error " + std::to_string(::GetLastError()) + " locking pages";
            return;
        }
    }
#else
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { throw std::runtime_error("mmap of " + std::to_string(bytes) + " bytes failed"); }
    data_ = static_cast<std::byte*>(p);
    if (::mlock(data_, bytes) != 0) { lock_failure_ = "mlock failed (RLIMIT_MEMLOCK)"; }
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
