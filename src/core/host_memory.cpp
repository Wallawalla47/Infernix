#include "core/host_memory.h"

#include <algorithm>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>

#include <fstream>
#include <sstream>
#include <string>
#endif

namespace infernix {
namespace {

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
        out.process_private = counters.PrivateUsage;
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
    rlimit lock{};
    if (::getrlimit(RLIMIT_MEMLOCK, &lock) == 0 && lock.rlim_cur != RLIM_INFINITY) {
        out.lock_limit = static_cast<std::uint64_t>(lock.rlim_cur);
    }
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

} // namespace infernix
